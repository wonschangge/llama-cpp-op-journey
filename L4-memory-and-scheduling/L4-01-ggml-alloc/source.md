<!-- llama-coverage
ggml/include/ggml-alloc.h
ggml/src/ggml-alloc.c
-->

# L4-01 · 分配器 ggml-alloc：算子的内存从哪来 — 源文件

**一句话**：前面几课里，每个 `ggml_tensor` 都有一个 `data` 指针和一个 `buffer`；这一课回答**这两个字段是谁填的**。填它们的不是算子、也不是后端内核，而是 `ggml-alloc.c` 里的**图分配器**（graph allocator）。

它的核心事实只有一条：**图在开始执行之前就已经建好了，所以每个张量的"出生"和"死亡"时刻是已知的**。知道完整生命周期，就可以把互不重叠的生存期压进同一块内存 —— 而不是给每个张量各开一块。这就是验收点"为什么 ggml 需要自己管理内存"的答案。

本课只用两个源文件：`ggml/src/ggml-alloc.c`（1249 行）与 `ggml/include/ggml-alloc.h`（86 行）。文中提到的每个行号都出自这两个文件。

---

## 一、两张 API 面：tallocr 与 gallocr

`ggml-alloc.h` 开头就把分配器分成两种。上面那个 `struct ggml_tallocr` 是**线性分配器**：一个 buffer、一个 base、一个 alignment、一个 offset，全部状态就这四个字段；配套只有"新建"和"分配"两个函数，**没有释放**。下面那个 `ggml_gallocr` 是**图分配器**，本课的主角。

<!-- src: ggml/include/ggml-alloc.h -->
```c
// Tensor allocator
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);
```

## 二、图分配器的标准用法

这是 `ggml-alloc.h` 里那段 `Example usage` 注释，也是本课全部内容的路线图：`ggml_gallocr_new` 选 buffer 类型，`ggml_gallocr_reserve` 用最坏情况图先规划，`ggml_gallocr_alloc_graph` 对当前图落位，`ggml_gallocr_get_buffer_size` 报告结果，最后由后端 `ggml_backend_graph_compute` 执行。注释把 `reserve` 标为 optional：不调也能跑，只是可能在运行时触发重新分配。

<!-- src: ggml/include/ggml-alloc.h -->
```c
// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/
```

## 三、两个特殊 flag

分配器只为两类张量开了后门：`ggml_set_input()` 的张量**先落位且地址互不重叠**（它们在图外面被写数据），`ggml_set_output()` 的张量**永不被释放、也不会被覆盖**（图算完之后外面还要读它）。这两条正是 `ggml_gallocr_free_node` 里`GGML_TENSOR_FLAG_OUTPUT` 判断的依据。

<!-- src: ggml/include/ggml-alloc.h -->
```c
// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten
```

## 四、线性分配器 ggml_tallocr：只进不退

这就是"不复用"的完整实现。`ggml_tallocr_alloc` 先向后端问"这个张量要占多少字节"（同一个张量在不同后端上大小可能不同），对齐后检查是否越界，然后`talloc->offset += size` 把水位线往前推。**没有任何路径能让 offset 退回去**。所以它的内存账是"所有张量之和"，而不是"同时存活的峰值"。

<!-- src: ggml/src/ggml-alloc.c -->
```c
// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}
```

## 五、★ free_block：复用的数据结构

动态分配器 `ggml_dyn_tallocr` 不记录"哪个张量住在哪"，只记录**哪些字节是空的**。空闲块就是 `{offset, size}` 两个数（`struct free_block`），按地址有序地放在 `tallocr_chunk::free_blocks[MAX_FREE_BLOCKS]` 里。`max_size` 是这个 chunk 的水位线 —— 规划结束后 buffer 就按它开。`chunks[GGML_VBUFFER_MAX_CHUNKS]` 让一个分配器横跨多个后端 buffer，这是 L4-02 的基础。

<!-- src: ggml/src/ggml-alloc.c -->
```c
struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        struct buffer_address addr;
    } allocated_tensors[1024];
#endif
};
```

## 六、★ best-fit：从空闲表里挑一块

分配策略是 **best fit**：在所有装得下的空闲块里挑最小的那个。注意内层循环上界写成 `chunk->n_free_blocks - 1` —— **最后一个空闲块故意不参与挑选**，因为它代表"尚未使用的水位线以上"，用它意味着把 buffer 撑大（第 226-247 行才轮到它）。取块后把 `block->offset` 让出去、`block->size` 减掉，减到 0 就把这一项从表里删掉。

<!-- src: ggml/src/ggml-alloc.c -->
```c
static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }
```

## 七、还块：与相邻空闲块合并

还块时先看能不能"贴"在已有空闲块的尾部或头部，能贴就长大，长大之后顺手再看一眼另一侧的邻居能不能一起吞掉。两个方向都不沾，才新开一个空闲块 —— 这是空闲表变碎的唯一来源，也是 `MAX_FREE_BLOCKS`（第 14 行，256）这道上限存在的理由。

<!-- src: ggml/src/ggml-alloc.c -->
```c
// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}
```

## 八、★ 第一趟：把"谁还要用我"数出来

`ggml_gallocr_alloc_graph_impl` 的第一趟不分配，只做两件事：把所有叶子落位（它们不属于任何一次"执行"，可以认为活到最后），然后遍历每个节点的 `src[]`，把 `n_children` 加一；碰到视图则给源张量加 `n_views`。**这两个计数就是"生命周期"的量化形式**，也是第二趟能否释放的唯一判据。带 `GGML_TENSOR_FLAG_INPUT` 的张量在这里就抢先落位，避免地址被别人占掉。

<!-- src: ggml/src/ggml-alloc.c -->
```c
    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }
```

## 九、第二趟：按拓扑序分配，用完就还

第二趟沿 `graph->nodes[]` 的**拓扑序**走（这正是 L1-03 讲的顺序）：先保证父张量有地址，再分配自己，然后把每个父张量的 `n_children` 减一。当 `n_children == 0 && n_views == 0` 时，这个张量再也不会被读到，于是要么把源张量的 `n_views` 减一（视图的情况），要么直接 `ggml_gallocr_free_node`。**分配与释放共用同一个循环**：一进一出，水位线就被压在峰值上。

<!-- src: ggml/src/ggml-alloc.c -->
```c
    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_impl_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
```

## 十、free_node：输出张量永不释放

释放路径上的第一道判断是 `GGML_TENSOR_FLAG_OUTPUT`：图输出直接 return，既不还块也不改 `allocated`。注释写得很直白 —— graph outputs are never freed。其余张量走 `ggml_dyn_tallocr_free_bytes` 把地址还给空闲表。

<!-- src: ggml/src/ggml-alloc.c -->
```c
static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    // graph outputs are never freed
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        return;
    }

    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, node);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}
```

## 十一、视图：跳过分配，必要时接管地址

`ggml_gallocr_allocate_node` 的入口就是一条硬判据：`!ggml_impl_is_view(node)`。视图不拥有数据，所以永不从这里申请内存。更进一步：如果父张量只有 1 个消费者、0 个视图且 layout 与当前节点一致，当前节点可以直接**接管父张量的地址**（`hn->addr = p_hn->addr`），并把 `p_hn->allocated` 置 false 防止它被误释放。视图的情形还要额外检查 `view_src->data == parent->data`，确认视图正好覆盖源张量的全部数据。这一段呼应 L1-01 的 `view_src` / `view_offs`：零拷贝不只是省掉一次复制，还让分配器少发一块内存。

<!-- src: ggml/src/ggml-alloc.c -->
```c
static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_impl_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        return;
                    }
                }
            }
```

## 十二、图重放：needs_realloc 决定要不要重新规划

每次 `ggml_gallocr_alloc_graph` 都会先比一遍：节点数、叶子数、以及每个张量在规划时记下的 `size_max` 还够不够用。只要有一处不满足就返回 true。这就是"图形态没变就不用重规划"的实现 —— decode 每一步形状相同，所以这一步在稳态下几乎不花钱（L2-07 的 graph reuse 是它的上游）。

<!-- src: ggml/src/ggml-alloc.c -->
```c
static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            return true;
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                return true;
            }
        }
    }

    return false;
}
```

## 十三、落位：init_tensor 把规划结果发给张量

规划的产物是 `buffer_id` + `struct buffer_address`（chunk 下标 + 块内偏移），存在 `node_allocs` / `leaf_allocs` 里。`ggml_gallocr_init_tensor` 负责把这两个数翻译成张量的 `buffer` 与 `data`：普通张量走 `ggml_vbuffer_tensor_alloc`，视图走 `ggml_backend_view_init`（它自己不需要新内存，只要把地址算出来）。

<!-- src: ggml/src/ggml-alloc.c -->
```c
static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, struct tensor_alloc * tensor_alloc) {
    int buffer_id = tensor_alloc->buffer_id;
    assert(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    if (tensor->view_src != NULL) {
        if (tensor->buffer == NULL) {
            assert(tensor_alloc->addr.offset == SIZE_MAX);
            if (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
            ggml_backend_view_init(tensor);
        }
    } else {
        if (tensor->data == NULL) {
            assert(tensor_alloc->addr.offset != SIZE_MAX);
            assert(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } else {
            if (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
        }
    }
}
```

## 十四、图分配器的状态与多 buffer 版本

`struct ggml_gallocr` 的字段回答了"规划结果存在哪"：`buf_tallocs[n_buffers]` 是每个 buffer 一个动态分配器，`node_allocs` / `leaf_allocs` 是逐节点、逐叶子的落位表，`hash_set` + `hash_values` 是"张量 -> hash_node"的查找结构。`ggml_gallocr_new_n` 里的内层循环还做了一件小事：**同一个 buffer type 被给多次时共用同一个分配器**，避免同一块类型的内存被重复计算。

<!-- src: ggml/src/ggml-alloc.c -->
```c
struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    struct vbuffer ** buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, sizeof(struct vbuffer *));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}
```

## 十五、另一条路径（一）：在一段 buffer 里顺序摆张量

不是所有张量都属于某张图。权重、KV cache 这类"整场都活着"的东西走另一条路：`alloc_tensor_range` 开一个 buffer，用第四节那个线性 `ggml_tallocr` 沿着 context 的张量链一个挨一个摆下去。注意它对视图的处理与图分配器一致：`t->view_src == NULL` 才调 `ggml_tallocr_alloc`，否则只做 `ggml_backend_view_init`。已经带 data 的张量直接跳过 —— 这就是"不要重复分配别人的内存"。

<!-- src: ggml/src/ggml-alloc.c -->
```c
static bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
        free_buffers(buffers, n_buffers);
        return false;
    }

    *buffers = realloc(*buffers, sizeof(ggml_backend_buffer_t) * (*n_buffers + 1));
    (*buffers)[(*n_buffers)++] = buffer;

    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        enum ggml_status status = GGML_STATUS_SUCCESS;
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                status = ggml_tallocr_alloc(&tallocr, t);
            } else if (t->buffer == NULL) {
                status = ggml_backend_view_init(t);
            }
        } else {
            if (t->view_src != NULL && t->buffer == NULL) {
                // view of a pre-allocated tensor
                status = ggml_backend_view_init(t);
            }
        }
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t->name);
            free_buffers(buffers, n_buffers);
            return false;
        }
    }

    return true;
}
```

## 十六、另一条路径（二）：装不下就换一个 buffer

`ggml_backend_alloc_ctx_tensors_from_buft_impl` 沿 context 的张量链累加大小：一旦 `cur_buf_size + this_size` 超过这一种 buffer type 的 `max_size`，就把前面那一段交给 `alloc_tensor_range` 开出去，从头开始累加。因为视图与已分配的张量大小为 0（第 1183-1185 行），它们不会把 buffer 撑大。最后 `n_buffers > 1` 时用 `ggml_backend_multi_buffer_alloc_buffer` 合成一个句柄。

<!-- src: ggml/src/ggml-alloc.c -->
```c
static ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft_impl(
        struct ggml_context * ctx, ggml_backend_buffer_type_t buft, size_t * nbytes_total, bool no_alloc) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;
    *nbytes_total = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!no_alloc && !alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return NULL;
            }
            first = t;
            *nbytes_total += cur_buf_size;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        *nbytes_total += cur_buf_size;
        if (!no_alloc && !alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return NULL;
        }
    }

    if (no_alloc) {
        return NULL;
    }

    if (n_buffers == 0) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
#endif
        GGML_ASSERT(!buffers);
        return NULL;
    }

    ggml_backend_buffer_t buffer;
    if (n_buffers == 1) {
        buffer = buffers[0];
    } else {
        buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
    }
    if (buffers) {
        free(buffers); // can be NULL if context is empty or no_alloc
    }
    return buffer;
}
```

## 十七、规划的输出：ggml_gallocr_get_buffer_size

`ggml_gallocr_get_buffer_size` 返回的就是"这张图要多少执行缓冲"。它读的是 `ggml_vbuffer_size`，也就是各 chunk 水位线之和。注意里面那个去重循环：同一个 buffer type 被注册多次时，只有第一次出现才会报数，避免重复计入。

<!-- src: ggml/src/ggml-alloc.c -->
```c
size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->buffers[buffer_id] == NULL) {
        return 0;
    }

    for (int i = 0; i < buffer_id; i++) {
        if (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            return 0;
        }
    }

    return ggml_vbuffer_size(galloc->buffers[buffer_id]);
}
```

---

## 说明

- 本课只用 `ggml/src/ggml-alloc.c`（1249 行）与 `ggml/include/ggml-alloc.h`（86 行）两个文件，均计入覆盖率。
- 课件里提到的 `ggml_op_can_inplace`（第 22-51 行）、`MAX_FREE_BLOCKS`（第 14 行）、`GGML_VBUFFER_MAX_CHUNKS`（第 96 行）、`ggml_gallocr_reserve_n_impl`（第 825 行起）、`ggml_gallocr_node_needs_realloc`（第 997-1007 行）都在这两个文件里，属于"提到但不逐字引用"的行，不计入引用块数。
- 验收点"为什么 ggml 需要自己管理内存而不是逐个 ggml_new_tensor"的答案在第 2 幕：线性分配器只有一个单调递增的 offset（第 87 行），没有释放路径；而图分配器知道每个张量的完整生命周期，可以把死者的内存立刻发给下一个人。
