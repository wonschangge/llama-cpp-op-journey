<!-- llama-coverage
src/llama-context.cpp
src/llama-context.h
-->

# L2-07 · 上下文与解码 llama-context — 源文件

**一句话**：`llama_decode` 是**图的生命周期管理器** —— 建图（`model.build_graph`）、分配（`ggml_backend_sched_alloc_graph`）、执行（`graph_compute` → `ggml_backend_sched_graph_compute_async`）、复用（`res->can_reuse`）这四件事都在 `llama_context::decode` 一个函数里闭环。

回顾 L2-06：那一课讲 `llm_graph_context` 与 `build_*` 原语 —— 也就是"图长什么样"。这一课讲**谁在什么时候调用它**：一个 `llama_batch` 进来，怎么变成若干张被调度器执行的图，以及为什么第二个 token 不用重新建图。

---

## 一、llama_context 的公共面：decode / encode / graph_compute / graph_reserve

这几个声明就是本课的全部入口。注意 `graph_compute` 的注释直接写明它返回的是 "`ggml_backend_sched_graph_compute_async` 执行的结果"，`graph_reserve` 的注释则说明它用**一个假 ubatch** 去预留图 —— 这正是 `sched_reserve()` 的底层动作。

<!-- src: src/llama-context.h -->
```cpp
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);
```

## 二、sched_reserve()：一次性预留

`sched_reserve()` 在 `decode` 的第 1800 行被调用，但它第一件事就是检查开关：`sched_need_reserve` 为 false 时**整段跳过**。需要预留时，它按最坏情况（`n_tokens = min(n_ctx, n_ubatch)`、`n_seqs = n_seq_max`）算出 `max_nodes`，然后重建调度器、重建 `gf_res_reserve`，并让 memory 用 `init_full()` 准备一个"整块"的 memory context。

这一段是本课"分配"侧的核心：**调度器与计算缓冲的容量在这里被钉死**，之后每个 token 都只是在这个容量里换一张图（细节见 L4-02）。

<!-- src: src/llama-context.cpp -->
```cpp
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }
```

## 三、graph_reserve()：预留路径用到的三个调度器函数

`sched_reserve()` 走的不是 `process_ubatch` 那条路，而是 `graph_reserve()`。同一个调度器对象，在预留路径上暴露三个不同的函数名 —— 这一段把"实际函数名"钉清楚：`ggml_backend_sched_reserve_size`（只算大小）、`ggml_backend_sched_split_graph`（只切分）、`ggml_backend_sched_reserve`（切分 + 分配）。

<!-- src: src/llama-context.cpp -->
```cpp
    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}
```

## 四、★ 复用槽有两个：gf_res_prev[n_outputs > 0]

`get_gf_res_prev()` 只有四行，但它是 graph reuse 的关键：**按"这一轮 ubatch 有没有输出"选不同的 arena**。头文件里对应的注释解释了原因 —— 分开的 arena 让"有输出/无输出"的批在 CUDA graph 缓存里拿到不同的 key（见下面的成员引用）。

<!-- src: src/llama-context.cpp -->
```cpp
llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
```

## 五、成员速查：sched / balloc / output_ids / sched_need_reserve

`llama_context` 里与本课直接相关的成员都在这几行。特别值得注意的是 `balloc` 的注释："reuse the batch_allocr to avoid unnecessary memory allocations" —— 批的解析对象也是复用的。

<!-- src: src/llama-context.h -->
```cpp
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;
```

## 六、复用相关的成员与开关

`gf_res_prev` 是一个长度为 2 的数组，`gf_res_reserve` 供预留使用，`gf_res_prev_active` 记住"当前哪一个 arena 里的图是有效的"。`graph_reuse_disable` 的注释写明它来自环境变量 `LLAMA_GRAPH_REUSE_DISABLE`（构造函数 280-281 行读取）。

<!-- src: src/llama-context.h -->
```cpp
    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;
```

## 七、输出：从图里取回 logits

图算完之后，context 把 `res->get_logits()` 指到的张量当成**源**，用 `ggml_backend_tensor_get_async` 拷进自己的 host 缓冲 `logits.data`。注意中间的断言：目标偏移 `n_outputs_prev*n_vocab` 加上本轮输出，不能越过 `logits.size`。

之后用户在 CPU 侧读 `llama_context::get_logits()` / `get_logits_ith(i)`（909、944 行；C API 侧的 `llama_get_logits` 在 3932 行），`get_logits_ith` 会先把批里的 token 序号用 `output_ids` 翻译成输出行号（`output_resolve_row`，915 行）。

<!-- src: src/llama-context.cpp -->
```cpp
        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }
```

## 八、复用计数进了 perf

`n_reused` 不只是个内部计数器：它被塞进 `llama_perf_context_data`，并最终由 `llama_perf_context_print` 打印成一行 `graphs reused = ...`（4364 行）。所以"这一轮跑下来复用了几次"是**可以实测的数字**，不是推测。

<!-- src: src/llama-context.cpp -->
```cpp
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}
```

---

## 说明

- 本课只引用 `src/llama-context.cpp` 与 `src/llama-context.h` 两个文件，二者都计入覆盖率。
- 文中提到的 `llm_graph_result::can_reuse` / `llm_graph_params::allow_reuse` / `llm_graph_result::set_inputs` 定义在 `src/llama-graph.{h,cpp}`（L2-06 覆盖），本课只引用 `llama-context.cpp:1405`、`1449` 这两个调用点，故 `src/llama-graph.h` 不计入本课覆盖率。
- 文中提到的 `ggml_backend_sched_*` 函数名与 `ggml_backend_tensor_get_async` 均取自本课引用的 两个文件正文，未从注释推断。
