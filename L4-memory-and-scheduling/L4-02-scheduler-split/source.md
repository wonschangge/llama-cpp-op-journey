<!-- llama-coverage
ggml/include/ggml-backend.h
ggml/src/ggml-backend.cpp
-->

# L4-02 · ★ 调度器：把图切给不同后端 — 源文件

**一句话**：`ggml_backend_sched` 是"多后端执行"的总管。它把一张图**沿数据流切成若干连续段**，每段整体交给一个后端；段与段之间的张量，必须在下游后端的 buffer 里**再出现一份** —— 这份副本就是切分的产物，不是图里本来有的节点。

本课覆盖 `ggml/src/ggml-backend.cpp`（v0.5.0，共 2513 行）与 `ggml/include/ggml-backend.h`（437 行）。所有引用都按行号从上游抽取，行号只对 v0.5.0 有效。

---

## 一、调度器在头文件里的自述

三行注释定义了它的职责边界：**多设备协同、compute buffer 分配、张量归属、跨后端拷贝**。注意最后一条 —— 本课后半段会看到，那些拷贝用的目标张量是**切分时现场造出来的**，不是图里本来就有的节点。

<!-- src: ggml/include/ggml-backend.h -->
```c
    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
```

## 二、后端顺序就是优先级

头文件把约定写在了构造函数上方：**下标小的后端优先级高**。所以调用方传进来的顺序（GPU 在前、CPU 在后）直接决定了归属结果 —— 这也是 L3-02 里"设备枚举顺序会影响 offload"的根。

<!-- src: ggml/include/ggml-backend.h -->
```c
    // Initialize a backend scheduler, backends with low index are given priority over backends with high index
    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
```

## 三、最后一个后端必须是 CPU

构造函数用断言把上一条约定钉死：数组最后一个后端必须是 **CPU 类型设备**。后文多处依赖这一点，例如图输入默认落在 `n_backends - 1`（第 946 行）、以及 pass 2 里"跳过 CPU 只扩展 GPU"。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);
```

## 四、数据结构：split 与 sched

整个调度器的状态都在这两个结构体里。读法：`split` 回答"哪一段、吃什么"，`sched` 回答"每个张量归谁、副本在哪、内存谁管"。

| 字段 | 含义 |
|---|---|
| `split.backend_id` / `i_start` / `i_end` | 这一段归哪个后端、是图上哪一段 |
| `split.inputs[]` / `n_inputs` | 这一段开始前必须搬进来的张量清单 |
| `split.graph` | 这一段的子图视图，执行时整段交给一个后端 |
| `hv_tensor_backend_ids` | 张量 -> 后端 id |
| `hv_tensor_copies` | (张量, 后端, 副本) -> 副本张量 |
| `splits[]` / `n_splits` | 切分结果 |
| `galloc` | 分配器句柄（L4-01） |
| `op_offload` | 是否允许把 op 抢到更靠前的后端 |

<!-- src: ggml/src/ggml-backend.cpp -->
```c
struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;
```

## 五、归属（1）：权重所在 buffer 决定候选后端

`ggml_backend_sched_backend_from_buffer()` 是一个纯查询：把"张量在哪个 buffer"翻译成"哪个后端 id"。判据是两个条件同时成立 —— 后端**支持这个 buffer type**，并且**支持这个 op**。它按后端表顺序扫描，所以返回的永远是优先级最高的合格者；一个都没有时返回 `-1`，debug 模式下还会打一行警告说明"这个权重将需要被拷贝"。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}
```

## 六、归属（2）：pass 1 的优先级

`ggml_backend_sched_backend_id_from_cur()` 是 pass 1 的全部逻辑，顺序固定：

1. 张量已经有 buffer（权重、用户预分配）-> 用它的后端（`1.dst`）；
2. 视图张量 -> 跟 `view_src` 走（`1.vsrc`）；
3. 图输入 -> 最后一个后端，即 CPU（`1.inp`）；
4. 输入里有 `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` 的张量 -> 跟权重走（`1.wgt`）。

一条都不命中就返回 `-1`：这不是错误，而是"等后面的 pass 再定"。那些 `SET_CAUSE` 字符串是打开调试输出后能看到的归属理由。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
```

## 七、归属（3）：权重与 op_offload

第 4 条规则还有反转：如果按权重算出来的后端正好是最后一个（CPU）、权重又在 host 内存里，并且构造调度器时 `op_offload` 打开，那么更靠前的后端只要 `supports_op` + `offload_op` 都通过，就可以把这个 op **抢过去**（`1.off`）。

两个例外被显式跳过：`GGML_OP_ROPE`（rope freqs 张量太小）和 `GGML_OP_FLASH_ATTN_EXT`（sinks 张量太小）—— 源码注释说明理由：小张量不足以决定 op 的归属。

回顾 L2-05：`dev_layer` / `n_gpu_layers` 决定哪些层的权重放 GPU；这里决定剩下那些"权重留在 host"的层里，哪些 op 还能被抢上去。实战见 L8-02。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}
```

## 八、pass 2：把相邻节点并到同一个后端

pass 1 只给"有据可依"的节点定了归属，中间的激活张量大多是 `-1`。pass 2 用多轮扫描把它们补上：**先处理 GPU（非最后优先级的后端），沿图向下、向上各扫一遍；忽略 CPU**。下面引的是 GPU 的两遍，其余后端的两遍紧随其后（1172-1202）。

注释里的三条结论值得抄下来：

```text
assign the same backend to adjacent nodes
expand gpu backends up and down, ignoring cpu (the lowest priority backend)
thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
```

也就是说：**"能连成一片"本身就是切分想要的结果** —— pass 2 主动把同后端的节点连成段，pass 5 才只需要在真正连不起来的地方断刀。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
```

## 九、pass 4：视图跟着 view_src，兜底保证没有节点落空

pass 4 做两件事：把还没归属的**视图张量**绑到 `view_src` 的归属上，把还没归属的**源张量**绑到"使用它的那个节点"的归属上；最后给仍然落单的节点挑第一个支持它的后端，并用 `GGML_ASSERT(*cur_backend_id != -1)` 保证**没有节点落空**。

这条断言的另一面是：如果所有后端都不支持某个 op，程序会在这里直接挂掉，而不是悄悄算出错误结果。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }
```

## 十、pass 5：切段的判据

pass 5 是"切分"这件事真正发生的地方。它沿拓扑序走一遍，维护一个"当前段"：

- 节点的后端和当前段不同 -> 断段；
- 节点带着"别的后端上的权重"、而当前段后端**又不支持那个 buffer type** -> 也断段。源码注释给出理由：断开之后，上一段 offload 上去的权重内存可以更早被复用。

断段时旧段写 `i_end = i`、新段写 `i_start = i`，所以节点 `i` 属于新段 —— **边界是"切在节点之前"，每段永远是图上的连续区间**。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }
```

## 十一、搬运判据：buffer_supported

决定"要不要造副本"的是这个函数：张量如果**已经分配**，就看它所在 buffer 的 buffer type；否则看它已经被分配到的后端对应的 buffer type。然后问下游后端一句 `ggml_backend_supports_buft()` —— 不支持就得搬。

这就是验收点的技术根：**不同的后端用不同的 buffer type 管理内存**（显存 / pinned host / 普通内存，见 L4-03），一个后端的内核读不到另一个后端 buffer 里的数据。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}
```

## 十二、★ 边界造副本：copy 是切分的产物

当某个输入既不在本段后端上、本段后端又用不了它的 buffer type 时，调度器就地造副本：

1. `ggml_dup_tensor_layout()` 在调度器自己的 ctx 里造一个同形状同类型的新张量；
2. 把它按 `(张量, 后端, 副本)` 记进 `hv_tensor_copies`，每个组合只造一次；
3. 把**源**张量登记进 `split->inputs`（执行时的搬运清单）；
4. 把 `node->src[j]` 改成指向副本。

**关键：这不是往图里插一个 `GGML_OP_CPY` 节点。**图里的节点数不变，变的是某些节点的输入指针 —— 副本张量是"数据在哪"的改写，真正的搬运发生在 `ggml_backend_sched_compute_splits()` 里（下一节）。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
```

## 十三、分配：把切好的图交给 galloc

`ggml_backend_sched_alloc_splits()` 先判断**后端归属是否变化**：只有当 id 变了、或上一次分配失败时，才重新 `ggml_gallocr_reserve_n()`；随后 `ggml_gallocr_alloc_graph()` 真正落地地址。

注意两个 id 数组是给 `reserve_n` 的 —— **调度器通过 `node_backend_ids` / `leaf_backend_ids` 把"谁在哪"告诉分配器（L4-01），分配器再决定"地址在哪"**。重新 reserve 之前还会把所有后端同步一遍：因为重分配可能挪动 split 输入张量的地址。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }
```

## 十四、执行：先搬 inputs，再逐段 compute

每个 split 的执行是固定两步：

1. **搬 inputs**：优先走 `cpy_tensor_async`；后端不支持或拒绝时退回同步 `ggml_backend_tensor_copy()`，退回前会先同步源后端，保证数据真的写好；
2. **跑子图**：`ggml_backend_graph_compute_async(split_backend, &split->graph)`。注意传进去的是 `split->graph`（这一段的子图），不是原始整图。

回顾 L4-04：这个函数是单后端的执行入口；调度器只是按顺序对每一段各调一次。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
```

## 十五、reserve：用测量图先切一遍

`ggml_backend_sched_reserve()` 的用法在头文件示例里：拿一张"最大 batch"的图，走一遍 `split_graph()` + `ggml_gallocr_reserve_n()`，把每段的 buffer 先预留出来，然后 `reset` 掉。这样正式推理时不会因为要临时扩显存而失败或抖动。

注意它和 `alloc_graph()` 一样会调用 `ggml_backend_sched_split_graph()` —— **切分是每次分配都要重算的，不是一次性编译**。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}
```

## 十六、入口：alloc_graph 与 graph_compute_async

`ggml_backend_sched_alloc_graph()` 是"要开始跑了"的入口：先按 `next_copy` 轮转副本号（并行流水线时 `n_copies > 1`），再 `split_graph()`，然后 `alloc_splits()`，最后把 `is_alloc` 置真。

`graph_compute_async()` 的顺序是：没 reset 就先 reset；没分配就调 alloc_graph；然后 `compute_splits()`。所以**第一次 graph_compute 会自动带上切分与分配**，之后每步只执行。这两个函数的差别，就是 L4-04 的主题。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}
```

## 十七、段与段之间的握手：events

`compute_splits()` 的循环头能看到两件事：一是调度器**按 split 顺序逐段执行**；二是在特定情况下要等上一段：当前段没有输入、且上一段是别的后端时，必须等上一段的异步工作完成 —— 源码注释给了理由：**分配器可能在不同段之间复用了同一块 buffer 区域**（这正是 L4-01 的复用策略在调度器侧的代价）。

每段跑完则记录下来，供后面的段等待。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

```

## 十八、事件记录与副本

段执行完成之后记录事件，然后进入下一段。`events[][]` 的第一维是后端、第二维是副本号 —— 与 `hv_tensor_copies` 的第三维对应。

同一段代码上方还能看到环境变量 `GGML_SCHED_DEBUG`（第 1861 行）：它把 `sched->debug` 打开，于是 `ggml_backend_sched_print_assignments()` 会打印每个 split 的后端、输入清单，以及（`debug > 1` 时）每个节点的归属理由。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }
```

## 十九、并行副本与调试开关

`n_copies` 来自构造参数 `parallel`：并行（流水线）模式下取 `GGML_SCHED_MAX_COPIES`，否则为 1。这就是 `hv_tensor_copies` 里第三维的来源；同一段里 `GGML_SCHED_DEBUG` / `GGML_SCHED_DEBUG_REALLOC` 两个环境变量分别控制归属打印与"重分配是否属于意外"的断言。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
```

## 二十、用户可以手动指定归属

`ggml_backend_sched_set_tensor_backend()` 直接写 `tensor_backend_id(node)` 并清掉 `is_reset` 标记。pass 1 的两个循环里专门有 "do not overwrite user assignments" 的保护（第 1091、1100 行），所以手动指定的归属不会被算法覆盖 —— 这也是头文件示例里 "manually assign nodes to a backend (optional)" 的落点。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}
```

---

## 说明

- 本课引用 `ggml/src/ggml-backend.cpp` 与 `ggml/include/ggml-backend.h` 两个文件，均计入覆盖率。
- 第 6 幕里那张 6 节点小图是**示意**，不是源码：它的前提写在幕内（`op_offload` 关闭，所以带 host 权重的 `mul_mat` 留在 CPU）。示意里的节点名（n0/n1/...）只在本课内有效。
- 本课不引用 `ggml/src/ggml-backend-reg.cpp`（设备如何注册与排序）—— 那是 L3-02 的内容；也不引用 `ggml/src/ggml-alloc.c`（arena 怎么切）—— 那是 L4-01 的内容。
