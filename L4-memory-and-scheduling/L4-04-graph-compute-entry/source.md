<!-- llama-coverage
ggml/src/ggml-backend.cpp
ggml/src/ggml.c
-->

# L4-04 · 图执行入口：graph_compute 与异步 — 源文件

**一句话**：一张图怎么被执行，只剩两个入口 —— `ggml_backend_graph_compute()` 与它的异步版本。两者只差**一行**：同步版在返回前调用 `ggml_backend_synchronize()`。所以"同步"不是另一条执行路径，而是"异步提交 + 一次显式等待"。

这一课回答一个具体问题：**一次 async 提交之后，host 到底在哪一步才真正等设备算完**。答案不在 `ggml-backend.cpp` 的那两行里，而在三处：后端接口的 `synchronize`、调度器执行循环里的跨后端等待、以及调度器收尾时的 `ggml_backend_sched_synchronize()`。

同时要回答 `ggml.c` 在这一层的角色。实测：v0.5.0 的 `ggml.c` 里 `graph_compute` **出现 0 次** —— 执行入口已经整体搬到后端层；`ggml.c` 留下的是图的**容器与视图**，而"图视图"正是把一张图切成可单独提交的子图的那个工具。

---

## 一、★ 两个入口：同步只是"异步 + 一次等待"

`ggml_backend_graph_compute()` 的函数体只有三行：调异步版、调 `ggml_backend_synchronize()`、返回状态。`ggml_backend_graph_compute_async()` 只有一行转发。**两者的全部差别就是第 457 行**——所以"同步路径"不是另一套执行机制，而是"异步提交 + 一次显式等待"。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}
```

## 二、等待的真正落点：`iface.synchronize`

`ggml_backend_synchronize()` 自己不做任何等待，它只做三件事：断言、判空、转发。

判空这一条很关键：**后端的 `synchronize` 可以是 `NULL`**，那就直接返回。CPU 后端正是如此（`ggml/src/ggml-cpu/ggml-cpu.cpp:201` 写着 `.synchronize = NULL`，该文件属 L5-01），而 CUDA 后端把它实现成 `cudaStreamSynchronize()`（`ggml/src/ggml-cuda/ggml-cuda.cu:2544`，该文件属 L6-01）。

所以"同步与异步路径分别在哪一步等待"这个问题，必须先分清**哪个后端**：

| 后端 | `iface.synchronize` | 同步入口的行为 |
|---|---|---|
| CPU | `NULL` | 第 427-429 行空返回；`graph_compute` 返回时已经算完 |
| CUDA | `ggml_backend_cuda_synchronize` | 第 431 行进入 `cudaStreamSynchronize`，等 stream 排空 |

<!-- src: ggml/src/ggml-backend.cpp -->
```c
void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}
```

## 三、事件：把"完成"变成可等待的句柄

要让 host 不停下来，就必须能"事后"再问"那次提交完成了吗"。事件对象就是那个句柄，五个 API 各管一件事：

- `event_new(device)`：设备级对象；设备没实现就返回 `NULL`（第 536-538 行）。
- `event_record(event, backend)`：在一次提交之后插标记，**不阻塞**。
- `event_wait(backend, event)`：让**这个后端**等标记，host 继续跑。
- `event_synchronize(event)`：让 **host** 等标记。
- `event_free(event)`：`NULL` 安全。

两个等待主体的区别，是调度器能把多后端流水线排起来的关键。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
```

## 四、调度器层的同一对入口

调度器把"提交 / 等待"这对动作又包了一层。`ggml_backend_sched_graph_compute()` = 异步版 + `ggml_backend_sched_synchronize()`（后者循环 `sched->n_backends` 次）。

异步版里还藏着 graph reuse 的两个短路：`is_reset` / `is_alloc`。已经分配过（`is_alloc == true`）就跳过切分与分配，直接进 `ggml_backend_sched_compute_splits()`。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}
```

## 五、★ graph reuse 的缓存判据

第二次跑同一张图时，调度器要先回答"切分还算不算数"。判据写得很保守：**后端编号变了、并且 buffer type 也变了**，才认为需要重新分配（第 1594-1595 行）；leafs 同理（第 1600-1608 行）。判据不成立时，第 1611 行的 `ggml_gallocr_alloc_graph()` 会直接在预留好的块里重新指派地址（复用细节见 L4-01）。

三层缓存叠在一起：调度器层 `is_alloc` 短路 → 切分层的 id 比较 → 显存层的 gallocr 复用。真要重算时，`ggml_backend_sched_reset()`（第 1949 行）会清空 hash 表与后端映射。

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
```

## 六、★ 异步提交之后，host 在哪里等

每个 split 都用异步入口提交（第 1799 行），提交后立刻给事件盖章（第 1839 行）。真正的等待点只有下面几处：

| 位置 | 调用 | 等待主体 |
|---|---|---|
| 第 1665 / 1667 行 | `event_synchronize` / `ggml_backend_synchronize` | host（跨后端交接） |
| 第 1680 行 | `event_synchronize` | host（拷贝用户输入前） |
| 第 1688 行 | `event_wait` | 只有那个后端 |
| 第 2032-2043 行 | `ggml_backend_sched_synchronize` | host（收尾，循环所有后端） |

带 eval 回调时是例外：每个回调段提交后立刻同步（第 1821、1827 行），因为回调要立刻读数据 —— 本文件第 2298 行的 `ggml_backend_compare_graph_backend()` 则直接使用同步入口（第 2312-2313、2335-2336 行）。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

```

## 七、ggml.c 侧：执行入口已经搬走，留下的是图的容器与视图

实测：`grep -c graph_compute ggml/src/ggml.c` 的结果是 **0**。`ggml_graph_plan()` 现在定义在 `ggml/src/ggml-cpu/ggml-cpu.c:2815`，`ggml_graph_compute()` 在 `ggml-cpu.c:3399`（两者都属 L5-01）。

`ggml.c` 留下的是图的**表示层**：

- `ggml_graph_nbytes`（7451）/ `ggml_graph_overhead_custom`（7469）/ `ggml_new_graph_custom`（7477）：图的内存预算。
- `ggml_graph_view`（7526）：零拷贝切出节点数组的一段 —— 就是提交给后端的那张子图。
- `ggml_graph_clear`（7646）：就地复位 `n_nodes` / `n_leafs` / hash 表，复用同一块图内存。
- `ggml_status_to_string`（438）：把 `graph_compute` 的返回值翻成文字。

`ggml_graph_view` 的两个真实调用点：调度器给每个 split 造子图（`ggml-backend.cpp:1460`），以及逐节点对比时造单节点子图（`ggml-backend.cpp:2332`）。

<!-- src: ggml/src/ggml.c -->
```c
struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ NULL, // gradients would need visited_hash_set
        /*.grad_accs        =*/ NULL,
        /*.leafs            =*/ NULL,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    return cgraph;
}
```

## 八、状态码：执行入口的返回值

执行入口返回 `enum ggml_status`。四个取值的文字由 `ggml_status_to_string()` 给出，调度器与 `llama_context` 就是用它在出错时打日志的。

<!-- src: ggml/src/ggml.c -->
```c
const char * ggml_status_to_string(enum ggml_status status) {
    switch (status) {
        case GGML_STATUS_ALLOC_FAILED: return "GGML status: error (failed to allocate memory)";
        case GGML_STATUS_FAILED:       return "GGML status: error (operation failed)";
        case GGML_STATUS_SUCCESS:      return "GGML status: success";
        case GGML_STATUS_ABORTED:      return "GGML status: warning (operation aborted)";
    }

    return "GGML status: unknown";
}
```

---

## 说明

- 本课逐字引用 `ggml/src/ggml-backend.cpp` 与 `ggml/src/ggml.c` 两个文件，计入覆盖率。
- 文中按行号提到的 `ggml/src/ggml-cpu/ggml-cpu.cpp`、`ggml/src/ggml-cpu/ggml-cpu.c`、`ggml/src/ggml-cuda/ggml-cuda.cu`、`src/llama-context.cpp` 分别是 L5-01、L6-01、L2-07 的覆盖文件，本课只指路、不引用其源码，**不计入本课覆盖率**。
