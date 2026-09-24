<!-- llama-coverage
src/llama-graph.cpp
ggml/src/ggml.c
ggml/src/ggml-backend.cpp
ggml/src/ggml-cpu/ggml-cpu.c
src/models/llama.cpp
src/llama-context.cpp
ggml/src/ggml-cpu/ggml-cpu.cpp
ggml/src/ggml-cuda/ggml-cuda.cu
-->

# L8-01 · ★ 端到端：一个 mul_mat 的完整旅程 — 源文件

**一句话**：这一课不讲新机制，只做一件事 —— 挑 `build_attn` 里**一次真实的 `ggml_mul_mat` 调用**，从它在模型代码里被调用的那一刻起，一路走到 CPU 的 `vec_dot` 内核，并在**每一跳上指出判据**：谁决定下一步走哪。

这条链路过 **9 个文件**：`src/models/llama.cpp` → `src/llama-graph.cpp` → `ggml/src/ggml.c` → `src/llama-context.cpp` → `ggml/src/ggml-backend.cpp` → `ggml/src/ggml-cpu/ggml-cpu.cpp` → `ggml/src/ggml-cpu/ggml-cpu.c` → `ggml/src/ggml-cpu/arch/x86/quants.c`（对照：`ggml/src/ggml-cuda/ggml-cuda.cu`）。

主角选的是**输出投影**那一次：`build_attn`（`src/llama-graph.cpp:2766`）在第 2804 行调用 `build_lora_mm(wo, cur, wo_s)`，后者在第 1518 行调用 `ggml_mul_mat(ctx0, w, cur)`。选它的原因很实际：`wo` 是**权重**，它的 buffer 决定这次 mul_mat 归哪个后端 —— 这正是 L4-02 讲的那条判据；而同一段 `build_attn_mha` 里的另一次 mul_mat（第 2715 行）没有权重，归属判据完全不同（第 7 幕会对照）。

---

## 一、起点：build_attn 里的输出投影

模型侧（`src/models/llama.cpp`）在第 169 行把本层的 `wo / wo_b / wo_s` 与 Q/K/V 交给 `build_attn`；`build_attn`（`src/llama-graph.cpp:2766`）在第 2804 行调用 `build_lora_mm(wo, cur, wo_s)`；`build_lora_mm`（第 1514 行）的本体就是第 1518 行的 `ggml_mul_mat(ctx0, w, cur)`。

**这一串调用里没有任何计算**：它只是往 `ctx0` 里加了一个 `GGML_OP_MUL_MAT` 节点。下面是这三处的逐字引用（两个文件）。

<!-- src: src/models/llama.cpp -->
```cpp
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
//>> ---- src/models/llama.cpp:246-246 ----
    ggml_build_forward_expand(gf, cur);
```

## 二、这一次 mul_mat 的构造器

`ggml_mul_mat` 只做四件事：

1. 断言 `ggml_can_mul_mat(a, b)`（三条 `ne` 关系）与 `!ggml_is_transposed(a)`；
2. 算输出形状 `ne = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] }`；
3. `ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)` —— **输出恒为 F32**；
4. 填 `op = GGML_OP_MUL_MAT` 与 `src[0] = a`、`src[1] = b`。

第 4 步就是 L1-02 的结论："输入个数由构造器赋了几个 `src[i]` 决定"—— 这里赋了两个。

<!-- src: src/llama-graph.cpp -->
```cpp
    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }
//>> ---- src/llama-graph.cpp:1514-1518 ----
ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
```

## 三、入图：拓扑序遍历与 hash set

`ggml_build_forward_expand`（`ggml/src/ggml.c:7337`）只是入口，转手就调 `ggml_build_forward_impl` → `ggml_visit_parents_graph`（第 7236 行）。

递归的每一站先做一次 `ggml_hash_find(&cgraph->visited_hash_set, node)`：`used` 位已置就立即返回（第 7244 行）。这就是 L1-03 里"图是 DAG、公共子图只算一次"的实现。本篇的 mul_mat 节点正是由 `build_attn` 内部的局部 expand（`llama-graph.cpp:2735`、`2783`）与模型 build 函数末尾的 `ggml_build_forward_expand(gf, cur)`（`src/models/llama.cpp:246`）两类调用拉进图的。

<!-- src: ggml/src/ggml.c -->
```c
void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}
//>> ---- ggml/src/ggml.c:7236-7245 ----
static size_t ggml_visit_parents_graph(struct ggml_cgraph * cgraph, struct ggml_tensor * node, bool compute) {
    if (node->op != GGML_OP_NONE && compute) {
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    const size_t node_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
    GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);

    if (ggml_bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
        // already visited
```

## 四、进入执行：decode 的三个调用点

C API `llama_decode`（`src/llama-context.cpp:4326`）只有一行转发。`llama_context::decode`（第 1704 行）里与本链路有关的三个调用点是：建图（1425 `model.build_graph`）、分配（1435 `ggml_backend_sched_alloc_graph`）、计算（1454 `graph_compute` → 2588 `ggml_backend_sched_graph_compute_async`）。

注意第 1415 行的 `n_reused++`：图能复用就不重建，但**分配仍然每次都做**（L2-07 的结论）。

<!-- src: src/llama-context.cpp -->
```cpp
int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
//>> ---- src/llama-context.cpp:1433-1454 ----
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
//>> ---- src/llama-context.cpp:2583-2590 ----
    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
```

## 五、调度器：归属判据与逐段执行

`ggml_backend_sched_alloc_graph`（`ggml/src/ggml-backend.cpp:1992`）第一件事是 `ggml_backend_sched_split_graph`（2000）。切分时每个节点问 `ggml_backend_sched_backend_id_from_cur`（921），判据按固定优先级：预分配 buffer（923）> 视图的 buffer（930）> 图输入（945）> **输入里有权重**（967）。

本篇的 mul_mat 命中最后一条：`src[0] = wo` 的 buffer usage 是 `WEIGHTS`，于是归属就取 `ggml_backend_sched_backend_from_buffer`（888）找到的那个后端。执行阶段 `ggml_backend_sched_compute_splits`（1646）逐段调用 `ggml_backend_graph_compute_async`（1799）→ `backend->iface.graph_compute`（463）。

对照（同一段代码里的第二个 mul_mat）：`kqv = ggml_mul_mat(ctx0, v, kq)`（`src/llama-graph.cpp:2715`）两个输入都不是权重，`backend_id_from_cur` 对它返回 `-1`，改由邻居带走（`ggml_backend_sched_set_if_supported`，1058）或在 pass 3（1210-1246）按"支持输入最多的后端"分配 —— 判据不同，落点就可能不同。

<!-- src: ggml/src/ggml-backend.cpp -->
```cpp
bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

//>> ---- ggml/src/ggml-backend.cpp:921-948 ----
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
//>> ---- ggml/src/ggml-backend.cpp:961-969 ----
    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
//>> ---- ggml/src/ggml-backend.cpp:1646-1660 ----
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    int prev_backend_id = -1;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

//>> ---- ggml/src/ggml-backend.cpp:1798-1802 ----
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
//>> ---- ggml/src/ggml-backend.cpp:461-464 ----
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}
```

## 六、CPU 侧：接口表 → 入口 → 大 switch → vec_dot

CPU 后端的接口表把 `graph_compute` 槽填成 `ggml_backend_cpu_graph_compute`（`ggml/src/ggml-cpu/ggml-cpu.cpp:206`，函数体在 170）：它先 `ggml_graph_plan` 算 `cplan`，再调 `ggml_graph_compute`（`ggml/src/ggml-cpu/ggml-cpu.c:3399`）。

线程主体按 `cgraph->nodes[]` 的顺序逐个调用 `ggml_compute_forward`（第 1744 行），那里是一个 `switch (tensor->op)`；`GGML_OP_MUL_MAT` 的 case 在第 1869 行，指向 `ggml_compute_forward_mul_mat`（1255）→ `..._one_chunk`（1165）。

内核的选择在 `..._one_chunk` 里：`type_traits_cpu[type].vec_dot`（1182）。以 Q4_0 权重为例，表里绑定的是 `ggml_vec_dot_q4_0_q8_0`、`vec_dot_type = GGML_TYPE_Q8_0`（240-243）；该函数按架构由 CMake 从 `arch/<arch>/quants.c` 选一份编译（x86 的在 `ggml/src/ggml-cpu/arch/x86/quants.c:701`）。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.cpp -->
```cpp
static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
}
//>> ---- ggml/src/ggml-cpu/ggml-cpu.cpp:193-210 ----
static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};
```

## 七、CPU 分派与内核选择的逐字引用

第一段是分派入口（`ggml_compute_forward`），第二段是 `GGML_OP_MUL_MAT` 的 case，第三段是 Q4_0 在 `type_traits_cpu[]` 里的条目，第四段是 `..._one_chunk` 里 `vec_dot` 的绑定处（真正调用在 1244 行）。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.c -->
```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:1869-1872 ----
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:240-249 ----
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:1165-1183 ----
static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;
```

## 八、同一跳在 CUDA 侧（对照）

同一个 `iface.graph_compute` 槽，在 CUDA 后端里填的是 `ggml_backend_cuda_graph_compute`（`ggml/src/ggml-cuda/ggml-cuda.cu:4421`，接口表见 4853）。它内部的 `ggml_cuda_compute_forward`（2067）同样是一个 `switch (dst->op)`，`GGML_OP_MUL_MAT` 的 case 在第 2259 行，指向 `ggml_cuda_mul_mat`（1823）。

**两次分派的形状完全一样**：一次按 `op` 找到函数，再由函数内部按 `src0->type` 与形状挑具体内核。这就是 L6-01 与本课的交点。

<!-- src: ggml/src/ggml-cuda/ggml-cuda.cu -->
```cpp
static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:2067-2071 ----
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:2256-2262 ----
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:1823-1830 ----
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:4850-4854 ----
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
```

## 九、完整调用链表（每一步的判据）

把这一课压成一张表。左列是跳，中间是位置，右列是"谁决定下一步走哪"。

| # | 跳（函数） | 文件:行号 | 判据 |
|---|---|---|---|
| 0 | 模型 build → `build_attn` | `src/models/llama.cpp:169` | 层里有 `wo` → 走注意力块 |
| 1 | `build_attn` → `build_lora_mm(wo, cur)` | `src/llama-graph.cpp:2804` | `wo` 非空 |
| 2 | `build_lora_mm` → `ggml_mul_mat(w, cur)` | `src/llama-graph.cpp:1518` | 只是加一个节点 |
| 3 | 构造器：断言 + `ne[]` + `op/src` | `ggml/src/ggml.c:3333 / 3348 / 3351` | `ggml_can_mul_mat` 三条 ne 判据 |
| 4 | `ggml_build_forward_expand` → `ggml_visit_parents_graph` | `ggml/src/ggml.c:7337 → 7236` | `visited_hash_set` 没见过才继续 |
| 5 | `llama_decode` → `llama_context::decode` → `graph_compute` | `src/llama-context.cpp:4326 → 1704 → 2569` | 图没变就复用（1415），但分配照做 |
| 6 | `ggml_backend_sched_alloc_graph` → `split_graph` | `ggml/src/ggml-backend.cpp:1992 → 1066` | 切分在分配之前（2000） |
| 7 | `ggml_backend_sched_backend_id_from_cur` | `ggml/src/ggml-backend.cpp:921` | `src[0]` 的 buffer usage = WEIGHTS（967） |
| 8 | `compute_splits` → `iface.graph_compute` | `ggml/src/ggml-backend.cpp:1799 → 461` | `split->backend_id`（1658） |
| 9 | `ggml_backend_cpu_graph_compute` → `ggml_graph_compute` | `ggml-cpu/ggml-cpu.cpp:170` → `ggml-cpu/ggml-cpu.c:3399` | 接口表第 206 行 |
| 10 | `ggml_compute_forward` → `..._mul_mat` → `..._one_chunk` | `ggml-cpu/ggml-cpu.c:1744 → 1869 → 1255 → 1165` | `switch (tensor->op)` 与 `case GGML_OP_MUL_MAT`（1869） |
| 11 | `vec_dot` 内核 | `ggml-cpu/ggml-cpu.c:1182` → `ggml-cpu/arch/x86/quants.c:701` | `type_traits_cpu[src0->type].vec_dot`（240-243） |
| 12 | （对照）CUDA 同一跳 | `ggml-cuda/ggml-cuda.cu:2259 → 1823` | 同一个 iface 槽，另一套实现 |

表里的每个行号都在本课的逐字引用里出现过（第 0/4/5/9 跳的调用点除外 —— 它们分别是 `src/models/llama.cpp:169`、`src/models/llama.cpp:246`、`src/llama-context.cpp:1415`、`ggml/src/ggml-cpu/ggml-cpu.cpp:206`，也都在这几段引用区间内）。

---

## 说明

- 本课覆盖 **9 个源文件**，全部计入覆盖率。其中 `src/llama-graph.cpp` 是计划里本课的主文件；按"这条链路"追加了 3 个：`ggml/src/ggml.c`（`ggml_mul_mat` 构造器 + `ggml_build_forward_expand`/拓扑序）、`ggml/src/ggml-backend.cpp`（切分、归属与调度执行）、`ggml/src/ggml-cpu/ggml-cpu.c`（CPU 分派与 `vec_dot` 选择）。
- 另外 5 个文件是这条链路的相邻跳，本课只取"这一跳"的几行：`src/models/llama.cpp`（`build_attn` 的调用点与最终 expand）、`src/llama-context.cpp`（decode 与 graph_compute）、`ggml/src/ggml-cpu/ggml-cpu.cpp`（CPU 接口表与入口）、`ggml/src/ggml-cuda/ggml-cuda.cu`（同一跳的 CUDA 实现，作对照）、`ggml/src/ggml-cpu/arch/x86/quants.c`（x86 版 `ggml_vec_dot_q4_0_q8_0`）。这些文件各自的专课会逐字展开：L2-06 / L2-07 / L5-01 / L6-01 / L5-03。
- 说明：`ggml/src/ggml-cpu/arch/<arch>/quants.c` 的同名函数由 CMake 按架构选一份编译（`ggml/src/ggml-cpu/CMakeLists.txt` 的 `GGML_CPU_SOURCES`），本课引用的是 x86 那一份；换架构时 `type_traits_cpu[]` 里的绑定名不变，实现文件会换。
