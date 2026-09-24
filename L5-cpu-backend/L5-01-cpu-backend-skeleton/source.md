<!-- llama-coverage
ggml/src/ggml-cpu/ggml-cpu.cpp
ggml/src/ggml-cpu/ggml-cpu.c
ggml/include/ggml-cpu.h
-->

# L5-01 · CPU 后端骨架：从 graph_compute 到算子分派 — 源文件

**一句话**：CPU 后端 = **一个大 switch + 一个线程池**。`ggml_compute_forward()` 用 102 个 `case` 把 `tensor->op`（L1-02 讲的身份）分派到具体内核；`ggml_get_n_tasks()` 决定每个 op 开几个任务，线程池按这个数字启动 worker。

这一课**不展开任何内核**——只讲骨架：一个 op 从进入后端到调用内核，中间经过哪几层、每一层在哪个文件的哪一行、参数 `ith` / `nth` 是怎么传下去的。内核本身在 L5-02（一元/二元）、L5-03（向量化）、L5-04（量化与 repack）、L5-05（多架构 SIMD）里逐课展开。

---

## 一、入口：ggml_backend_cpu_graph_compute

CPU 后端的 `graph_compute` 槽指向这个函数。它只做三件事：

1. `ggml_graph_plan()` 算出 `cplan`（任务数 + work buffer 大小）；
2. 保证 work buffer 够大（够就复用后端持有的那块，不够才重分配）；
3. 调 `ggml_graph_compute()` 进入 `ggml-cpu.c`。

注意它是**同步**的：返回时整张图已经算完，`ggml_status` 是结果。

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
```

## 二、接口表：ggml_backend_cpu_i

16 个槽里 **6 个已实现、10 个是 NULL**。NULL 不是"没写完"，是"不需要"：
CPU 与数据在同一块内存上，异步传输、事件、图优化都没有意义。

这张表就是 L3（后端注册）的落点 —— 注册表拿到的是 `struct ggml_backend_i`，调度器（L4-02）只认这张表里的函数指针。

另外三个后端 API 也一并放在这里：`ggml_backend_cpu_init()` 建上下文并调 `ggml_cpu_init()`；`ggml_backend_cpu_set_n_threads()` 改的是**后端上下文里的数字**，下一张图才会生效。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.cpp -->
```cpp
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
//>> ---- ggml/src/ggml-cpu/ggml-cpu.cpp:217-247 ----
ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}
//>> ---- ggml/src/ggml-cpu/ggml-cpu.cpp:253-258 ----
void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}
```

## 三、★ 规划：ggml_graph_plan 逐节点问任务数

`ggml_graph_plan()` 是"任务数"的决策点。它按 `cgraph->nodes` 顺序遍历（即 L1-03 的拓扑序），对每个节点调 `ggml_get_n_tasks(node, n_threads)`，取全图最大值 `max_tasks`，并据此估算 work buffer 大小。

函数的最后三项产物就是 plan 交给 compute 的全部信息：

```text
cplan.threadpool = threadpool;
cplan.n_threads  = MIN(max_tasks, n_threads);   // 真正启动的线程数
cplan.work_size  = work_size;
```

**这就是"线程数不由用户决定"的地方**：`n_threads` 只是上限，真正启动几个线程要看图里最"想并行"的那个 op。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.c -->
```c
struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2842-2850 ----
    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:3055-3067 ----
        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
```

## 四、线程池与线程主体

`struct ggml_threadpool` 全局一份，`struct ggml_compute_state` 每线程一份 —— 两者唯一的区别就是 `ith`。

线程主体 `ggml_graph_compute_thread()` 先构造 `params`，再进入节点循环：
每个节点先试 `ggml_cpu_try_fuse_ops()`（CPU 侧的算子融合），没命中才调 `ggml_compute_forward()`；节点之间用 `ggml_barrier()` 同步。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.c -->
```c
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:3122-3167 ----
    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }
```

## 五、★ 大 switch：ggml_compute_forward

这是本课的核心。`ggml_compute_forward()` 用**一个 switch** 把 `tensor->op`（L1-02 讲的算子身份）映射成一次内核调用。

按行数数：**102 个 `case GGML_OP_*`**，其中

| 类别 | 个数 | 说明 |
|---|---|---|
| 直接调用内核 | 96 | `ggml_compute_forward_dup` / `_add` / `_mul_mat` ... |
| nop | 5 | `NONE` / `RESHAPE` / `PERMUTE` / `VIEW` / `TRANSPOSE` |
| 兜底 abort | 1 | `GGML_OP_COUNT` -> `GGML_ABORT("fatal error")` |

对比 GPU 后端：CUDA 是"每个 op 一个 kernel 启动"，CPU 是"一个 switch + 一个线程池"。

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
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2152-2177 ----
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}
```

## 六、★ ggml_get_n_tasks：哪类 op 开几个任务

同一个 switch 结构，但这里决定的是**任务数**。把 case 按返回值分组：

| 策略 | op 分支数 | 代表 |
|---|---|---|
| `= n_threads` | 72 | `ADD` `MUL_MAT` `RMS_NORM` `CONV_2D` `ROPE` ... |
| `= 1` | 48 | `SUM` `ARGMAX` `CLAMP` `POOL_2D` `GET_ROWS` ... |
| `= MIN(n_threads, ...)` | 4 | `SOFT_MAX` `RWKV_WKV6/7` `GATED_LINEAR_ATTN` |
| 由 `op_params` 决定 | 4 | `MAP_CUSTOM1/2/3` `CUSTOM` |
| abort | 1 | `GGML_OP_COUNT` |

数法：顶层 `case` 标签 102 个；`GGML_OP_UNARY`（内层 22 个子算子）与 `GGML_OP_GLU`（内层 7 个）是容器，展开后共 **129 个 op 分支**。

下面按组逐字引用真实实现。注意 `GET_ROWS` / `SET_ROWS` 那一组：源码把 `n_tasks = n_threads;` **注释掉了**，理由写在旁边的 FIXME 里。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.c -->
```c
static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2263-2288 ----
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2304-2338 ----
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2364-2377 ----
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2402-2405 ----
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2440-2446 ----
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2453-2462 ----
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:2500-2523 ----
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}
```

## 七、ith / nth 落到内核：mul_mat 的抢块

内核怎么用 `ith` / `nth`？以 `ggml_compute_forward_mul_mat()` 为例：它先按"谁的行多就切谁"决定 `nchunk0` / `nchunk1`，然后 **`ith` 认领第一块**，剩下的块用原子计数器 `current_chunk` 动态抢。

这不是唯一写法：多数逐元素内核是**静态切分**（按 `ith` 直接取行），因为逐元素计算量均匀，抢块的原子开销不划算。两种写法的对比在 L5-02 / L5-03。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.c -->
```c
void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:1430-1447 ----
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);
```

## 八、公共 API、初始化与特征检测

`ggml/include/ggml-cpu.h` 只有 152 行，却是 CPU 后端的全部公共面：`struct ggml_cplan`、线程池 5 件套、`ggml_graph_plan` / `ggml_graph_compute`、`ggml_cpu_has_*` 特征检测、`ggml_type_traits_cpu`、以及后端 API。

两个入口在 `ggml-cpu.c`：`ggml_cpu_init()`（3867 行）第一次调用时建 GELU / SILU / FP16 查表，并读 `GGML_CPU_DISABLE_FUSION` 环境变量；`ggml_cpu_has_avx()`（3643 行）这类特征检测**是编译期常量** —— 它返回的是"这份二进制是否编进了 AVX"，不是运行时探测。这条线索通向 L5-05（多架构 SIMD 与厂商加速）。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.c -->
```c
int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:3867-3879 ----
void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
```

## 九、头文件：声明与调用顺序

头文件把调用顺序写死在注释里：`ggml_graph_plan()` 必须在 `ggml_graph_compute()` 之前调用；当 `plan.work_size > 0` 时，调用者必须自己准备 `plan.work_data`。

<!-- src: ggml/include/ggml-cpu.h -->
```c
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;
    };
//>> ---- ggml/include/ggml-cpu.h:58-74 ----
    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
//>> ---- ggml/include/ggml-cpu.h:124-141 ----
    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);
```

---

## 说明

- 本课引用 3 个源文件：`ggml/src/ggml-cpu/ggml-cpu.c`（3944 行）、`ggml/src/ggml-cpu/ggml-cpu.cpp`（716 行）、`ggml/include/ggml-cpu.h`（152 行），全部计入覆盖率。
- 场景 3 的算例（n_threads = 8、5 个节点、max_tasks = 8）是按 `ggml_get_n_tasks()` 的真实逻辑手算的示例，不是实测输出。
- 文中提到但不逐字引用的位置：`struct ggml_compute_params` 定义在 `ggml/src/ggml-cpu/ggml-cpu-impl.h:18`；`ggml_cpu_extra_compute_forward()` 定义在 `ggml/src/ggml-cpu/traits.cpp:12`；`ggml_graph_compute_kickoff()` 在 `ggml-cpu.c:3288`；`struct ggml_threadpool_params` 在 `ggml/include/ggml.h:3003`。这些文件不计入本课覆盖率。
