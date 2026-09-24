<!-- llama-coverage
ggml/src/ggml-hexagon/ggml-hexagon.cpp
ggml/src/ggml-hexagon/htp-drv.cpp
ggml/src/ggml-hexagon/htp-drv.h
ggml/src/ggml-hexagon/htp-opnode.h
ggml/src/ggml-hexagon/libdl.h
ggml/include/ggml-hexagon.h
ggml/src/ggml-hexagon/htp/main.c
ggml/src/ggml-hexagon/htp/act-ops.c
ggml/src/ggml-hexagon/htp/allreduce-ops.c
ggml/src/ggml-hexagon/htp/allreduce-ops.h
ggml/src/ggml-hexagon/htp/argsort-ops.c
ggml/src/ggml-hexagon/htp/binary-ops.c
ggml/src/ggml-hexagon/htp/binary-ops.h
ggml/src/ggml-hexagon/htp/concat-ops.c
ggml/src/ggml-hexagon/htp/cpy-ops.c
ggml/src/ggml-hexagon/htp/cumsum-ops.c
ggml/src/ggml-hexagon/htp/diag-ops.c
ggml/src/ggml-hexagon/htp/fill-ops.c
ggml/src/ggml-hexagon/htp/flash-attn-ops.c
ggml/src/ggml-hexagon/htp/flash-attn-ops.h
ggml/src/ggml-hexagon/htp/gated-delta-net-ops.c
ggml/src/ggml-hexagon/htp/gated-delta-net-ops.h
ggml/src/ggml-hexagon/htp/get-rows-ops.c
ggml/src/ggml-hexagon/htp/get-rows-ops.h
ggml/src/ggml-hexagon/htp/im2col-ops.c
ggml/src/ggml-hexagon/htp/matmul-ops.c
ggml/src/ggml-hexagon/htp/matmul-ops.h
ggml/src/ggml-hexagon/htp/pad-ops.c
ggml/src/ggml-hexagon/htp/repeat-ops.c
ggml/src/ggml-hexagon/htp/roll-ops.c
ggml/src/ggml-hexagon/htp/rope-ops.c
ggml/src/ggml-hexagon/htp/rope-ops.h
ggml/src/ggml-hexagon/htp/set-rows-ops.c
ggml/src/ggml-hexagon/htp/set-rows-ops.h
ggml/src/ggml-hexagon/htp/softmax-ops.c
ggml/src/ggml-hexagon/htp/softmax-ops.h
ggml/src/ggml-hexagon/htp/solve-tri-ops.c
ggml/src/ggml-hexagon/htp/ssm-conv.c
ggml/src/ggml-hexagon/htp/ssm-conv.h
ggml/src/ggml-hexagon/htp/sum-rows-ops.c
ggml/src/ggml-hexagon/htp/unary-ops.c
ggml/src/ggml-hexagon/htp/unary-ops.h
ggml/src/ggml-hexagon/htp/htp-ctx.h
ggml/src/ggml-hexagon/htp/htp-fence.h
ggml/src/ggml-hexagon/htp/htp-ops.h
ggml/src/ggml-hexagon/htp/htp-tensor.c
ggml/src/ggml-hexagon/htp/htp-tensor.h
ggml/src/ggml-hexagon/htp/htp-vtcm.h
ggml/src/ggml-hexagon/htp/hvx-arith.h
ggml/src/ggml-hexagon/htp/hvx-base.h
ggml/src/ggml-hexagon/htp/hvx-copy.h
ggml/src/ggml-hexagon/htp/hvx-div.h
ggml/src/ggml-hexagon/htp/hvx-dump.h
ggml/src/ggml-hexagon/htp/hvx-exp.h
ggml/src/ggml-hexagon/htp/hvx-fa-kernels.h
ggml/src/ggml-hexagon/htp/hvx-flash-attn.h
ggml/src/ggml-hexagon/htp/hvx-floor.h
ggml/src/ggml-hexagon/htp/hvx-inverse.h
ggml/src/ggml-hexagon/htp/hvx-log.h
ggml/src/ggml-hexagon/htp/hvx-mm-kernels-float.h
ggml/src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h
ggml/src/ggml-hexagon/htp/hvx-norm.h
ggml/src/ggml-hexagon/htp/hvx-pow.h
ggml/src/ggml-hexagon/htp/hvx-quant.h
ggml/src/ggml-hexagon/htp/hvx-reduce.h
ggml/src/ggml-hexagon/htp/hvx-repl.h
ggml/src/ggml-hexagon/htp/hvx-scale.h
ggml/src/ggml-hexagon/htp/hvx-sigmoid.h
ggml/src/ggml-hexagon/htp/hvx-sin-cos.h
ggml/src/ggml-hexagon/htp/hvx-sqrt.h
ggml/src/ggml-hexagon/htp/hvx-types.h
ggml/src/ggml-hexagon/htp/hvx-utils.h
ggml/src/ggml-hexagon/htp/hmx-fa-kernels.h
ggml/src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h
ggml/src/ggml-hexagon/htp/hmx-queue.c
ggml/src/ggml-hexagon/htp/hmx-queue.h
ggml/src/ggml-hexagon/htp/hmx-utils.h
ggml/src/ggml-hexagon/htp/dma-queue.c
ggml/src/ggml-hexagon/htp/dma-queue.h
ggml/src/ggml-hexagon/htp/work-queue.c
ggml/src/ggml-hexagon/htp/work-queue.h
ggml/src/ggml-hexagon/htp/hex-bitmap.h
ggml/src/ggml-hexagon/htp/hex-common.h
ggml/src/ggml-hexagon/htp/hex-dump.h
ggml/src/ggml-hexagon/htp/hex-fastdiv.h
ggml/src/ggml-hexagon/htp/hex-profile.h
ggml/src/ggml-hexagon/htp/hex-utils.h
-->

# L7-02 · Hexagon 后端（Qualcomm NPU/DSP）：host 与 DSP 的边界 — 源文件

**一句话**：Hexagon 后端把算子送到**另一颗处理器**上执行 —— 主机侧（ARM/x86，跑 `llama.cpp` 的那个进程）只负责**建图、分内存、发命令、等结果**；真正算数的是 DSP 上的 HTP（Hexagon Tensor Processor），用 HVX 向量单元和 HMX 矩阵引擎跑内核。

**核心洞察**：正因为要跨两个处理器架构，这个后端对外的接口**不只是"提交 kernel"**，还必须包含三样别的东西 —— **远端内存分配**（`rpcmem` + `fastrpc_mmap`）、**命令队列**（`dspqueue`）、**显式同步**（`fence` + `dspqueue_write` 的收发配对）。同机的 GPU 后端（L6 系）用统一虚拟地址 + 驱动隐式同步就能解决的事，在这里都得由后端自己搭出来。

本课覆盖清单里的 **87 个文件全部计入覆盖率**（`llama-coverage` 块里逐条声明）。其中 **10 个核心文件被逐字引用**（场景里 8 个，source.md 里另加 2 个），其余按下面这张表按族说明 —— 它们分别是 host 侧的 C++ 与 DSP 侧的 C，编译进两个不同的二进制。

| 族 | 路径 / 命名 | 文件数 | 编译到 | 说明 |
|---|---|---|---|---|
| host 顶层 | `ggml/src/ggml-hexagon/*.{cpp,h}`（不含 `htp/`） | 5 | host | `ggml-hexagon.cpp`（8070 行，后端主体）、`htp-drv.cpp` / `htp-drv.h`（FastRPC 动态加载）、`htp-opnode.h`（HTP 图节点）、`libdl.h` |
| 公共 API | `ggml/include/ggml-hexagon.h` | 1 | host | 只导出 3 个函数 |
| DSP 入口 | `htp/main.c` | 1 | DSP | `htp_iface_*` 远端接口实现 + 主循环 + 算子分发 |
| DSP 算子内核 | `htp/*-ops.c`、`htp/*-ops.h`、`htp/ssm-conv.{c,h}` | 35 | DSP | 24 个 `.c` + 11 个 `.h`，一个算子一份实现 |
| DSP 核心结构 | `htp/htp-*.{h,c}` | 6 | DSP | `htp-ops.h`（线格式）、`htp-ctx.h`（DSP 上下文）、`htp-tensor.{h,c}`（cache 维护）、`htp-vtcm.h`（VTCM 布局）、`htp-fence.h`（栅栏） |
| HVX 内核 | `htp/hvx-*.h` | 24 | DSP | HVX 向量单元 intrinsics 层 |
| HMX 矩阵引擎 | `htp/hmx-*` | 5 | DSP | HMX 队列 + tiled matmul / flash-attn 参数 |
| 队列 | `htp/dma-queue.{c,h}`、`htp/work-queue.{c,h}` | 4 | DSP | 硬件 DMA 描述符 + 工作线程池 |
| 通用工具 | `htp/hex-*.h` | 6 | DSP | bitmap / fastdiv / profile / dump / common / utils |

合计 5 + 1 + 1 + 35 + 6 + 24 + 5 + 4 + 6 = **87**。

---

## 一、公共 API：只有三个函数

整个 Hexagon 后端对外只暴露三个函数。其余 8000 多行都藏在后端内部 —— **这正说明"后端"这个抽象层是有效的：设备再复杂，契约只有那么大。**

<!-- src: ggml/include/ggml-hexagon.h -->
```c
// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);
```

## 二、host 侧：注册与设备枚举

`reg_i` 四项与 L3-01 的后端契约逐字对应。关键在于 `htpdrv_init()`：它去 `dlopen` 厂商的 FastRPC 驱动，失败就返回 `NULL` —— **这个后端在非高通机器上不是"报错"，而是根本不出现在设备列表里。**

<!-- src: ggml/src/ggml-hexagon/ggml-hexagon.cpp -->
```cpp
static const struct ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            auto nErr = htpdrv_init();
            if (nErr != AEE_SUCCESS) {
                return NULL;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
```

## 三、host 侧：会话建立（远端句柄 + 命令队列）

一次会话要建两样东西：一个 FastRPC 远端句柄（控制面），一条 dspqueue（数据面）。队列在 host 侧创建后必须 `dspqueue_export()` 出一个 `queue_id`，DSP 才能导入同一条队列。源码里连请求/响应队列的字节数都是按 `opt_opqueue` 算出来的。

<!-- src: ggml/src/ggml-hexagon/ggml-hexagon.cpp -->
```cpp
    const size_t req_q_size = (sizeof(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    const size_t rsp_q_size = (sizeof(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(this->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (void *) this,  // Callback context
                          &queue);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", this->name.c_str(), (unsigned) err);
        throw std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    this->valid_queue = true;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &this->queue_id);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        throw std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }
```

## 四、host 侧：驱动是运行时加载的

`htpdrv.cpp` 不链接厂商库，而是运行时 `dlopen("libcdsprpc.so")`，再逐个 `dlsym` 把 FastRPC / dspqueue / rpcmem 的入口取出来。**这让同一个 ggml 二进制既能在没有 Hexagon 的机器上加载，又能在有 Hexagon 的机器上启用这个后端。**注意最后一行 `remote_system_request` 的 `ignore` 参数是 `true`：这个符号允许缺失。

<!-- src: ggml/src/ggml-hexagon/htp-drv.cpp -->
```cpp
int htpdrv_init() {
    static dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    static bool initialized = false;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#else
    std::string drv_path = "libcdsprpc.so";
#endif
    if (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        return AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    if (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        return AEE_EUNABLETOLOAD;
    }
```

## 五、host 侧：远端内存的三步

`ggml_hexagon_rpcmem_block` 是跨域内存的最小封装：`rpcmem_alloc2()` 分配（不是 `malloc`）、`rpcmem_to_fd()` 取 fd、真正的 `fastrpc_mmap()` 在 `ggml_hexagon_shared_buffer::mmap()` 里做。**指针不能跨域，fd 可以** —— 这就是 DSP 侧能拿到同一块内存的原因。

<!-- src: ggml/src/ggml-hexagon/ggml-hexagon.cpp -->
```cpp
struct ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    int       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        if (!base) {
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        if (fd < 0) {
            rpcmem_free(base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        this->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        if (base) {
            rpcmem_free(base);
        }
    }
};
```

## 六、host 侧：`fastrpc_mmap` 与 `htp_iface_munmap` 的配对

注意 `unmap()` 里的顺序：非 pinned 的 buffer 解除映射前，先调 `htp_iface_munmap()` **通知 DSP 放手**（源码注释：`HTP might still hold a reference, tell it drop it`），然后才 `fastrpc_munmap()`。跨域内存的生命周期必须两边都点头。

<!-- src: ggml/src/ggml-hexagon/ggml-hexagon.cpp -->
```cpp
    void unmap() {
        if (!this->mapped) return;

        if (!this->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        if (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (void *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) base(), size(), fd());

        this->mapped = false;
    }
```

## 七、DSP 侧：远端接口与队列导入

DSP 侧入口 `htp_iface_start()`：导入 host 创建的队列，探测硬件 HVX 线程数，再把 `htp_context` + 主线程栈 + `work_queue` + `dma_queue` + `hmx_queue` 按 offset 摆进**同一块 4K 对齐的内存**（`memalign(4096, footprint)`）。这么做的原因见 `htp-ctx.h`：一整块便于 cache 维护（`hex_l2fetch_block`）。

<!-- src: ggml/src/ggml-hexagon/htp/main.c -->
```c
    dspqueue_t dsp_queue = NULL;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    int err = dspqueue_import(dsp_queue_id, NULL, NULL, (void *) h, &dsp_queue);
    if (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(HIGH, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (void *) h, &dsp_queue);
    }

    if (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        return err;
    }
```

## 八、DSP 侧：一个 batch 的解码

`process_opbatch()` 从 dspqueue buffer 里按 `n_bufs / n_tensors / n_ops` 三个计数切出三个数组，先**整批刷一次 cache**（`qurt_mem_cache_clean` + `hex_l2fetch_block`），再 `prep_op_bufs()` 复用/新建 mmap，最后 `prep_tensors()` 把每个张量的「块内偏移」换成 DSP 自己的虚拟地址。

<!-- src: ggml/src/ggml-hexagon/htp/main.c -->
```c
static void process_opbatch(struct htp_context * ctx, const struct htp_opbatch_req * req, const struct dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    int err;

    const uint32_t n_bufs = req->n_bufs;
    const uint32_t n_tens = req->n_tensors;
    const uint32_t n_ops  = req->n_ops;

    const uint32_t b_size = sizeof(struct htp_buf_desc)  * n_bufs;
    const uint32_t t_size = sizeof(struct htp_tensor)    * n_tens;
    const uint32_t o_size = sizeof(struct htp_op_desc)   * n_ops;
    const uint32_t p_size = sizeof(struct htp_prof_desc) * n_ops;
    const uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * sizeof(struct htp_trace_desc);

    if (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        return;
    }

    FARF(HIGH, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    struct htp_buf_desc* bufs = (struct htp_buf_desc*)  m_ptr; m_ptr += b_size;
    struct htp_tensor*   tens = (struct htp_tensor*)    m_ptr; m_ptr += t_size;
    struct htp_op_desc*   ops = (struct htp_op_desc*)   m_ptr; m_ptr += o_size;
    struct htp_prof_desc* pds = (struct htp_prof_desc*) m_ptr;

    struct profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, sizeof(ctx->trace));
    if (ctx->profiler == HTP_PROF_TRACE) {
        struct htp_trace_desc * trace_events = (struct htp_trace_desc *) (m_ptr + p_size);
        for (int t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, sizeof(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);
```

## 九、DSP 侧：`prep_tensor` 与地址重定位

这一行是整条链上最关键的一句：`t->data = bufs[bi].base + offset;`。host 侧发过来的是**偏移**（`t->data` 在编码时被当作 offset 用），DSP 侧加上自己 mmap 出来的 `base`，才变成可解引用的指针。注释 `update data to the actual pointer` 写明了这件事。

<!-- src: ggml/src/ggml-hexagon/htp/main.c -->
```c
static void prep_tensor(struct htp_context *ctx, struct htp_buf_desc *bufs, struct htp_tensor *tens, uint32_t idx, struct htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(HIGH, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

static void prep_tensors(struct htp_context *ctx, struct htp_buf_desc *bufs, struct htp_tensor *tens, uint32_t n_tens) {
    for (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}
```

## 十、DSP 侧：`execute_op` 大 switch

算子的实际分发就在这里：`htp_op_code` → 具体的 `op_*()` 实现。对照 L5-01 的 CPU 后端 `ggml_compute_forward`：**两者是同一种设计的两个实例** —— 一个 switch 把"图上节点的身份"变成"某个函数调用"。

<!-- src: ggml/src/ggml-hexagon/htp/main.c -->
```c
static int execute_op(struct htp_ops_context * octx) {
    switch (octx->op) {
        case HTP_OP_MDEV_GROUP:
            return op_mdev_group(octx);

        case HTP_OP_FENCE:
            return op_fence(octx);

        case HTP_OP_ALLREDUCE:
        case HTP_OP_ALLREDUCE_ADD:
            return op_allreduce(octx);

        case HTP_OP_MUL_MAT:
        case HTP_OP_MUL_MAT_ADD:
            return op_matmul(octx);

        case HTP_OP_MUL_MAT_ID:
            return op_matmul_id(octx);

        case HTP_OP_MUL_MAT_ID_NX:
            return op_matmul_id_nx(octx);

        case HTP_OP_MUL_MAT_NX:
            return op_matmul_nx(octx);

        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_SUB:
        case HTP_OP_DIV:
        case HTP_OP_ADD_ID:
            return op_binary(octx);
```

## 十一、HTP 图：host 侧的节点包装

`htp_opnode` 是 host 侧的"HTP 图节点"。它不复制数据（`node` 仍是原 `ggml_tensor*`），只做三件事：换 opcode、挂上预计算的 `kernel_params`、把融合链接起来。`add_fused()` 里那几行 `inputs.erase` / `inputs.push_back` 就是"融合后重接边"的全部实现。

<!-- src: ggml/src/ggml-hexagon/htp-opnode.h -->
```cpp
struct htp_opnode {
    ggml_tensor * node   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<const ggml_tensor *> inputs;
    std::vector<const ggml_tensor *> outputs;
    std::string                      name;

    int n_active_src(const ggml_tensor * t) const {
        if (!t) return 0;
        for (int i = GGML_MAX_SRC - 1; i >= 0; i--) {
            if (t->src[i]) {
                return i + 1;
            }
        }
        return 0;
    }

    void init(ggml_tensor * node) {
        this->node = node;
        if (this->node) {
            this->name = ggml_op_desc(this->node);

            // Build inputs (preserving optional nullptrs)
            int n_inputs = n_active_src(this->node);
            this->inputs.resize(n_inputs, nullptr);
            for (int i = 0; i < n_inputs; i++) {
                this->inputs[i] = this->node->src[i];
            }

            // Build outputs
            this->outputs.push_back(this->dst());
        }
```

## 十二、线格式：`htp_tensor` 与 `htp_buf_desc`

这两个结构体是**边界的定义**。`htp_tensor::data` 的注释写得最清楚：`Buffer offset in the messages, and data pointer on the NPU` —— 同一字段，两端含义不同。`htp_op_desc` 里 `src[10] / dst[4]` 是**张量下标**（不是指针），`kernel_params[32]` 装的是 host 预先算好的参数。

<!-- src: ggml/src/ggml-hexagon/htp/htp-ops.h -->
```c
// Tensor descriptor
struct htp_tensor {
    uint64_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t size;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint32_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint16_t ti;                   // Tensor index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
struct htp_buf_desc {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // HTP_BUF_*
    uint32_t fd;       // file descriptor
};
```

## 十三、DSP 侧：线程池与跨设备栅栏

`work_queue_run()` 是 DSP 侧所有算子并行的唯一入口；`n <= 1` 时直接在本线程跑。栅栏用「一个 128 字节槽 = {seq, status}」表达，写侧**先 status 后 seq**，并用 `syncht` + `Q6_dccleaninva_A` 绕过 cache —— **因为没有硬件 cache 一致性，跨核可见性必须手工保证。**

<!-- src: ggml/src/ggml-hexagon/htp/work-queue.h -->
```c
#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size);
size_t       work_queue_alignof(void);
work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size);
void         work_queue_free(work_queue_t q);

void work_queue_wakeup(work_queue_t q);
void work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n);

static inline bool work_queue_run(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    if (n <= 1) {
        func(n, 0, data);
        return true;
    }
    return work_queue_run_async(q, func, data, n);
}
```

## （同上）栅栏的读写实现

`htp_fence_write` / `htp_fence_read` 的对称性值得逐行看：写侧 `atomic_store(&fence[1], status)` 在前、`atomic_store(&fence[0], seq)` 在后；读侧先 `Q6_dccleaninva_A` 再 `syncht`，然后才读 seq / status。

<!-- src: ggml/src/ggml-hexagon/htp/htp-fence.h -->
```c
static inline void htp_fence_write(void * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm volatile ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) fence);
}

static inline void htp_fence_read(const void * fence_ptr, uint32_t * seq, uint32_t * status) {
    const atomic_uint * fence = (const atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((void *) fence);
    asm volatile ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}
```

## 十四、VTCM：DSP 的片上暂存

`htp-vtcm.h` 只有 19 行，做一件事：**在一块 VTCM 里按顺序切格子**。`VTCM_LAYOUT_ALLOC` / `VTCM_LAYOUT_PTR` 这一对宏就是"先算偏移、再换指针"的全部约定。VTCM 的申请在 `htp/main.c` 的 `vtcm_alloc()`：`HAP_compute_res_acquire` + `HAP_compute_res_attr_set_vtcm_param_v2`，并注册 release 回调 —— 因为 VTCM 是**多会话共享**的稀缺资源，被别的会话抢走时必须能感知。

<!-- src: ggml/src/ggml-hexagon/htp/htp-vtcm.h -->
```c
static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) do { (L)->field = (off); (off) += (sz); } while (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) do { if (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } else { (L)->field = 0; } } while (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : NULL)
```

## 十五、DSP 侧的 cache 维护

`htp_tensor.h` 里的 `htp_tensor_is_contiguous` / `is_permuted` / `can_row_partition` 都是给内核判断「这块张量能不能按行切开并行」用的；真正的 cache 维护在 `htp-tensor.c`：`htp_tensor_dirty_all()` 标记输出、`htp_flush_dirty_ranges()` 在 batch 内按需回写、`htp_tensor_flush_all()` 在算子前把输入刷干净。**这是"两个地址空间"在性能上最贵的一处开销。**

<!-- src: ggml/src/ggml-hexagon/htp/htp-tensor.h -->
```c
static inline bool htp_tensor_is_contiguous(const struct htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    if (t->ne[0] != 1 && t->nb[0] != next_nb) {
        return false;
    }
    next_nb *= t->ne[0];
    for (int i = 1; i < HTP_OP_MAX_DIMS; i++) {
        if (t->ne[i] != 1 && t->nb[i] != next_nb) {
            return false;
        }
        next_nb *= t->ne[i];
    }
    return true;
}

static inline bool htp_tensor_is_permuted(const struct htp_tensor * t) {
    return t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

static inline bool htp_tensor_mdev_data_aligned(const struct htp_tensor * t) {
    return ((uintptr_t) t->data & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0;
}

static inline bool htp_tensor_can_row_partition(const struct htp_tensor * t, uint32_t elem_size) {
    if (!htp_tensor_mdev_data_aligned(t)) {
        return false;
    }
    if (t->ne[0] != 1 && t->nb[0] != elem_size) {
        return false;
    }
    if (htp_tensor_is_permuted(t)) {
        return false;
    }
    if (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) return false;
    if (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) return false;
    if (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) return false;
    return true;
}
```

## 十六、文件族普查：87 个文件都在哪些族里

上面逐字引用了 **10 个核心文件**：`ggml-hexagon.h`、`ggml-hexagon.cpp`、`htp-drv.cpp`、`htp-opnode.h`、`htp/main.c`、`htp/htp-ops.h`、`htp/work-queue.h`、`htp/htp-fence.h`、`htp/htp-tensor.h`、`htp/htp-vtcm.h`。其余 **77 个文件**按族列出如下 —— 它们都在 `llama-coverage` 块里声明、**计入覆盖率**，但本课不逐字引用其代码（这是诚实的不计入引用，不是不覆盖）。

| 族 | 文件数 | 代表文件 | 职责 |
|---|---|---|---|
| host 顶层 | 5 | `ggml-hexagon.cpp`（8070 行）、`htp-drv.cpp` / `.h`、`htp-opnode.h`、`libdl.h` | 后端主体、FastRPC 运行时加载、HTP 图节点包装、跨平台 `dlopen` 封装 |
| 公共 API | 1 | `ggml-hexagon.h` | 三个导出函数 |
| DSP 入口 | 1 | `htp/main.c`（1360 行） | `htp_iface_*` 远端接口 + 主循环 + 批处理 + 算子分发 |
| DSP 算子内核 | 35 | `matmul-ops.c`、`flash-attn-ops.c`、`rope-ops.c`、`ssm-conv.c` | 24 个 `.c` + 11 个 `.h`；每个算子一份 DSP 实现与它的参数头 |
| DSP 核心结构 | 6 | `htp-ctx.h`、`htp-ops.h`、`htp-tensor.{h,c}`、`htp-vtcm.h`、`htp-fence.h` | DSP 上下文、线格式、cache 维护、VTCM 布局、栅栏 |
| HVX 向量内核 | 24 | `hvx-mm-kernels-tiled.h`、`hvx-fa-kernels.h`、`hvx-quant.h` | HVX intrinsics 层：量化/反量化、`exp`/`log`/`sqrt`、归约、分块矩阵乘 |
| HMX 矩阵引擎 | 5 | `hmx-queue.{c,h}`、`hmx-mm-kernels-tiled.h`、`hmx-fa-kernels.h`、`hmx-utils.h` | HMX 硬件队列与 tiled 矩阵乘 / flash-attn 参数 |
| 队列 | 4 | `work-queue.{c,h}`、`dma-queue.{c,h}` | 工作线程池 + 硬件 DMA 描述符队列 |
| 通用工具 | 6 | `hex-bitmap.h`、`hex-fastdiv.h`、`hex-profile.h` 等 | bitmap / 快速除法 / trace 与 PMU profiling / dump / common / utils |

数量核对：5 + 1 + 1 + 35 + 6 + 24 + 5 + 4 + 6 = **87**，与 `python3 tools/plan_matrix.py --files` 给出的 L7-02 清单一致。按后缀分是 **29 个 `.c` + 56 个 `.h` + 2 个 `.cpp`**（`htp/` 单独数是 29 `.c` + 52 `.h`；host 侧另有 4 个 `.h` 与 2 个 `.cpp`）。

---

## 说明

- 本课覆盖域 = `ggml/include/ggml-hexagon.h`（1）+ `ggml/src/ggml-hexagon/` 下全部源文件（86）= **87**，与 `python3 tools/plan_matrix.py --files` 给出的 L7-02 清单逐条一致，全部计入覆盖率。
- `ggml/src/ggml-hexagon/htp/htp_iface.idl`、`htp/CMakeLists.txt`、`htp/cmake-toolchain.cmake` **不在覆盖域内**（后缀不在 `SOURCE_SUFFIXES` 里），本课只在散文里提到它们的作用，不计入覆盖率，也没有为它们做 `cover` 声明。
- 上游位置：`ggml/src/ggml-hexagon/`、`ggml/include/ggml-hexagon.h`，v0.5.0（commit `7fe450e19305`）。`ggml-hexagon.cpp` 8070 行、`htp/main.c` 1360 行（`wc -l` 计数；README 上方表格里的行数由 lessonkit 按换行切分统计，比 `wc -l` 多 1）。`htp/` 目录共 84 个文件，其中 29 个 `.c` + 52 个 `.h` 计入覆盖域，另有 `CMakeLists.txt`、`cmake-toolchain.cmake`、`htp_iface.idl` 三个非源文件不计入。本课不试图覆盖这些文件的全部内容。
