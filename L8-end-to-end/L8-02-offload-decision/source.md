<!-- llama-coverage
src/llama-model.cpp
ggml/src/ggml-backend.cpp
src/llama-context.cpp
-->

# L8-02 · ★ 端到端：offload 决策实战 — 源文件

**一句话**：`--n-gpu-layers` 不是"把多少算力交给 GPU"，而是**把最高的 N 个层槽位的权重放到设备 buffer 里**；至于图最后被切成几段、切在哪，是调度器**逐节点**重新决定的另一回事。

本课把两次决策串成一条链：`n_gpu_layers` → `i_gpu_start` → `dev_layer[il]` → 权重的 `buffer` → 调度器的节点归属 → `split`。每一环都能在源码里指到具体行号，最后一幕用一个 27 层的真实模型走完整条链。

覆盖 `src/llama-model.cpp`、`ggml/src/ggml-backend.cpp`、`src/llama-context.cpp`（v0.5.0，commit `7fe450e19305`）。所有引用都按行号从上游抽取，行号只对 v0.5.0 有效。

---

## 一、入口：一个整数变成一条分界线

`n_gpu_layers` 在 `load_tensors()` 里先被读成一个局部量（`llama-model.cpp:1438`），随即化成 `i_gpu_start`。注意右边的 `+ 1`：**槽位总数是 `n_layer_all + 1`，多出来的那一个是输出层**。CPU 判据写在 `get_layer_buft_list` 里：`il < i_gpu_start` 就走 CPU 分支。

<!-- src: src/llama-model.cpp -->
```c
    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };
```

## 二、★ 层 -> 设备：dev_input / dev_layer[] / dev_output

这一段产出的是路由表 `dev_layer[]`。三处细节值得单独记：

- **输入层永远在 CPU**（1536-1537 行硬编码，源码注释给了理由：几乎没有好处）；
- **输出层不是特例**：它走的是同一个 `get_layer_buft_list(n_layer_all)`；
- **多卡时的层分配**用 `std::upper_bound(splits, ..., float(il - i_gpu_start)/act_gpu_layers)`，而 `splits` 是各卡空闲显存归一化后的前缀和（1511-1519 行）—— 这一条 L3-02 已经展开过。

回顾 L2-05：那一课引用过 `1521-1546` 这一段，讲的是"层怎么被分配给设备"；本课往下追的是"这个分配结果怎么影响图的形状"。

<!-- src: src/llama-model.cpp -->
```c
    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);
```

## 三、★ 层 -> buffer：路由表被查的那一次

`create_tensor` 收到层号 `tn.bid`，查 `dev_layer.at(tn.bid).buft_list`，把候选 buffer 类型表交给加载器去选。**"层在哪个设备"到这里就变成"张量在哪块 buffer"了。**

同一段代码随后给整批权重 buffer 打上 `WEIGHTS` 标记：

<!-- src: src/llama-model.cpp -->
```c
ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}
```

## 四、★ WEIGHTS 标记：两次决策之间唯一的接口

源码注释把用途写得很清楚：`this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight`。

<!-- src: src/llama-model.cpp -->
```c
        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
```

## 五、★ 调度器只认 buffer：usage == WEIGHTS

`ggml_backend_sched_backend_id_from_cur` 的权重分支。三条判据按顺序生效：

1. 输入张量的 buffer 带 `WEIGHTS` 标记（967 行）；
2. `ggml_backend_sched_backend_from_buffer` 从下标 0 开始找**优先级最高**且支持该 op 的后端（968 行）；
3. `op_offload` 允许更靠前的后端把权重在 host 上的 op 抢走（970-976 行）。

注意 955-959 行排除了 `ROPE` 与 `FLASH_ATTN_EXT` —— 它们的输入权重张量太小，不足以代表整个算子的归属，所以交给后面的扩张轮次。

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
```

## 六、★ 切分的粒度是节点：split 存的是 node 下标

段不是一个"层的集合"，而是一个**节点区间**。`split->i_start = i` 里的 `i` 是`graph->nodes[]` 的下标 —— 图上根本没有"层"这个对象。

边界一旦确定，跨后端的输入张量就要在下游后端的 buffer 里再出现一份（1399-1419 行）：

<!-- src: ggml/src/ggml-backend.cpp -->
```c
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

## 七、边界上的搬运：copy 是切分的产物

`node->src[j] = tensor_id_copy(...)` 这一行把图**改了**：下游节点读的不再是原张量，而是切分时现场造出来的副本。这也是 L4-02 的核心结论之一。

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

## 八、扩张轮次：CPU 是一堵墙

没有权重输入的节点在 pass 1 里返回 -1，归属全靠 pass 2 的扩张。四段扩张（down / up / rest down / rest up）的规则写在注释里，其中最关键的是这句：`cpu will never be used unless weights are on cpu`。实现上就是 1139-1141 行：碰到 CPU 归属的节点，把 `cur_backend_id` 清成 -1，扩张中断。

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
```

## 九、反例：边界不一定落在层边界上

`llama_context::graph_get_cb` 是一段**补丁**。源码注释承认了问题：`norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends`。解决办法是把 `norm` / `l_last` 手工钉回本层的设备（2616 行），但**只在 `ubatch.n_tokens < 32 || full_offload` 时生效**。

顺带注意 2609 行的 `full_offload = model.n_gpu_layers() > model.hparams.n_layer_all`：它与 1521 行 `i_gpu_start == 0` 的条件完全等价 —— 三处代码对同一个边界是一致的。

<!-- src: src/llama-context.cpp -->
```c
llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
```

## 十、算例的算术：27 层模型 + 三行日志

把 `n_gpu_layers` 从 0 走到 28，`i_gpu_start = max(28 - L, 0)`：

| L | i_gpu_start | GPU 槽位 | GPU 重复层 |
|---|---|---|---|
| 0 | 28 | （无） | 0 |
| 1 | 27 | 输出层 | 0 |
| 4 | 24 | 24,25,26 + 输出层 | 3 |
| 14 | 14 | 14..26 + 输出层 | 13 |
| 27 | 1 | 1..26 + 输出层 | 26 |
| 28 | 0 | 0..26 + 输出层 | 27 |

下面这段日志的算术需要单独提醒：1834 行取 `min(n_gpu_layers, n_layer_all)`，而 1522 行取 `min(n_gpu_layers, n_layer_all + 1)`。当 `n_gpu_layers >= n_layer_all + 1`（含默认值 -1）时，1834 行把计数卡在 27，于是 1841 行报出 26 个 repeating 层，而实际有 27 个重复层在 GPU 上；1846 行又按 `n_layer_all + 1` 报出 28/28。**三行日志口径不一致，不要用它反推层数或段数。**

<!-- src: src/llama-model.cpp -->
```c
    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_all);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = n_layer_all + 1;
        const int max_offloadable_layers       = n_layer_all + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }
```

---

## 说明

- 本课覆盖 3 个源文件，均计入覆盖率：`src/llama-model.cpp`、`ggml/src/ggml-backend.cpp`、`src/llama-context.cpp`。
- 算例用的 27 层取自 `src/models/deepseek2.cpp:7-8`（源码注释点名 DeepSeek-V2-Lite，判据 `hparams.n_layer() == 27`）。该文件不在本课声明里，**不计入本课覆盖率**。
- `LLAMA_MAX_LAYERS = 512`（`src/llama-hparams.h:11`）是 `n_layer_all` 的上界，`llama-model.cpp:1260` 用它做断言。该头文件不在本课声明里，不计入本课覆盖率。
- 本课的所有数字都是**源码算术**的结果（`i_gpu_start` 公式 + 槽位判据），不是运行时实测：本机的 llama.cpp 只构建了 CPU 后端（`llama_supports_gpu_offload()` 为假），`load_tensors()` 的 offload 日志分支因此不会执行。要实测请在带 GPU 后端的机器上跑并打开调试日志。
