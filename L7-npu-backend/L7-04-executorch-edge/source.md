<!-- llama-coverage
ggml/include/ggml-et.h
ggml/src/ggml-et/ggml-et-common.h
ggml/src/ggml-et/ggml-et-cpu-compare.cpp
ggml/src/ggml-et/ggml-et-cpu-compare.h
ggml/src/ggml-et/ggml-et-kernels.cpp
ggml/src/ggml-et/ggml-et-kernels.h
ggml/src/ggml-et/ggml-et-memops.cpp
ggml/src/ggml-et/ggml-et-memops.h
ggml/src/ggml-et/ggml-et-ops.cpp
ggml/src/ggml-et/ggml-et-ops.h
ggml/src/ggml-et/ggml-et-uberkernel-common.h
ggml/src/ggml-et/ggml-et.cpp
ggml/src/ggml-et/et-kernels/src/block_ops.h
ggml/src/ggml-et/et-kernels/src/ggml_tensor.h
ggml/src/ggml-et/et-kernels/src/math_fp.h
ggml/src/ggml-et/et-kernels/src/platform.h
ggml/src/ggml-et/et-kernels/src/quants.h
ggml/src/ggml-et/et-kernels/src/tensor.h
ggml/src/ggml-et/et-kernels/src/clamp_f32.c
ggml/src/ggml-et/et-kernels/src/concat_f32.c
ggml/src/ggml-et/et-kernels/src/cont_f16.c
ggml/src/ggml-et/et-kernels/src/cont_f32.c
ggml/src/ggml-et/et-kernels/src/conv_2d_f32_me.c
ggml/src/ggml-et/et-kernels/src/cpy_f32_f16.c
ggml/src/ggml-et/et-kernels/src/cumsum_f32.c
ggml/src/ggml-et/et-kernels/src/diag_f32.c
ggml/src/ggml-et/et-kernels/src/el_map_f32.c
ggml/src/ggml-et/et-kernels/src/fill_f32.c
ggml/src/ggml-et/et-kernels/src/flash_attn_ext_f16_me.c
ggml/src/ggml-et/et-kernels/src/flash_attn_ext_f32.c
ggml/src/ggml-et/et-kernels/src/gated_delta_net_f32.c
ggml/src/ggml-et/et-kernels/src/get_rows_f32.c
ggml/src/ggml-et/et-kernels/src/glu_f32.c
ggml/src/ggml-et/et-kernels/src/group_norm_f32.c
ggml/src/ggml-et/et-kernels/src/im2col.c
ggml/src/ggml-et/et-kernels/src/l2_norm_f32.c
ggml/src/ggml-et/et-kernels/src/mean_f32.c
ggml/src/ggml-et/et-kernels/src/memops.c
ggml/src/ggml-et/et-kernels/src/mul_mat_Q4_0.c
ggml/src/ggml-et/et-kernels/src/mul_mat_Q4_0_matrix_engine.c
ggml/src/ggml-et/et-kernels/src/mul_mat_Q8_0.c
ggml/src/ggml-et/et-kernels/src/mul_mat_f16.c
ggml/src/ggml-et/et-kernels/src/mul_mat_f16_matrix_engine.c
ggml/src/ggml-et/et-kernels/src/mul_mat_f32.c
ggml/src/ggml-et/et-kernels/src/mul_mat_f32_matrix_engine.c
ggml/src/ggml-et/et-kernels/src/mul_mat_id_Q4_0.c
ggml/src/ggml-et/et-kernels/src/mul_mat_id_Q8_0.c
ggml/src/ggml-et/et-kernels/src/mul_mat_id_f32.c
ggml/src/ggml-et/et-kernels/src/norm_f32.c
ggml/src/ggml-et/et-kernels/src/pad_f32.c
ggml/src/ggml-et/et-kernels/src/repeat_f32.c
ggml/src/ggml-et/et-kernels/src/rms_norm_f32.c
ggml/src/ggml-et/et-kernels/src/rms_norm_mul_f32.c
ggml/src/ggml-et/et-kernels/src/rope_f32.c
ggml/src/ggml-et/et-kernels/src/rwkv_wkv6_f32.c
ggml/src/ggml-et/et-kernels/src/rwkv_wkv7_f32.c
ggml/src/ggml-et/et-kernels/src/scale_f32.c
ggml/src/ggml-et/et-kernels/src/set_f32.c
ggml/src/ggml-et/et-kernels/src/set_rows_f32.c
ggml/src/ggml-et/et-kernels/src/softmax_f32.c
ggml/src/ggml-et/et-kernels/src/solve_tri_f32.c
ggml/src/ggml-et/et-kernels/src/sqr_f32.c
ggml/src/ggml-et/et-kernels/src/ssm_conv_f32.c
ggml/src/ggml-et/et-kernels/src/ssm_scan_f32.c
ggml/src/ggml-et/et-kernels/src/sum_rows_f32.c
ggml/src/ggml-et/et-kernels/src/tri_f32.c
ggml/src/ggml-et/et-kernels/src/uberkernel.c
ggml/src/ggml-et/et-kernels/src/unary_f32.c
-->

# L7-04 · ExecuTorch 后端（边缘/移动端）：离线编译的设备内核与运行期派发 — 源文件

**一句话**：这个后端（`GGML_ET_NAME "ET"`）把 ggml 的算子交给 **ET-SoC** —— 一颗多核 RISC-V 加速器上的**预编译裸机内核**。它的"编译"**不发生在图里**：内核在**构建期**用 RISC-V 工具链交叉编译成 `*.elf`、嵌进宿主库，运行期只做"按名字取代码 -> 加载 -> 派发"；`graph_compute` 本身就是**一个 `for` 循环套一个 `switch`**，没有任何图级编译步骤。

这条路线与 L7-01（CANN 把 op 映射到 ACL 算子）、L7-03（OpenVINO 把子图翻成 ov::Model）的"**运行期映射**"不同：那两家在运行期把图翻译给厂商栈，ET 把翻译**提前到了构建期**，运行期只剩执行。

**验收点**：看完这一课，你要能说出它与 GPU 后端在**数据面抽象**上的不同 —— 关键一条是：跨 host/device 边界的不是"裸指针 + 形状 + 元素数"，而是**整个 `struct ggml_tensor` 按值**（`ggml-et-ops.cpp:282`），设备内核直接读它的 `ne[]` / `nb[]` / `type` / `data`（`scale_f32.c:32-40`）。

**关于课名的说明（实测修正）**：计划里这一课叫"ExecuTorch 后端"，讲解要点写的是"ExecuTorch 的委托机制、图导出与运行时"。在本仓库 v0.5.0 的 `ggml/src/ggml-et/` 里**搜不到** `executorch` / `torch` / `delegate` / `.pte` 任何一个符号：它用的是 ET 平台 SDK 的 `dev::IDeviceLayer` + `rt::IRuntime`（`ggml-et-common.h:6-8`；`ggml/src/ggml-et/CMakeLists.txt:239` 链接 `runtime::etrt_static deviceLayer::deviceLayer`），机制是"**预编译内核 + 运行期加载**"，不是"导出 `.pte` 再委托"。本课按代码实际机制写，并把这条差异显式讲出来。

---

## 一、公共 API：ggml-et.h 的全部内容

整个后端对外只有 8 个函数 + 1 个名字宏。注意最后两行：`ggml_backend_et_buffer_type()` 拿的是**设备内存**的缓冲类型，`ggml_backend_et_host_buffer_type()` 拿的是一个"host"缓冲类型 —— 但设备的 `get_host_buffer_type` 槽位返回的是 **CPU 的** buffer type（`ggml-et.cpp:1667-1670`），而 `is_host` 恒为 false（`ggml-et.cpp:437-440`）。读这一课时要盯住这条"ET 没有 host 内存路径"的线索。

<!-- src: ggml/include/ggml-et.h -->
```c
#define GGML_ET_NAME "ET"

// backend API
GGML_BACKEND_API ggml_guid_t     ggml_backend_et_guid(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_et_init(size_t devidx);

GGML_BACKEND_API bool ggml_backend_is_et(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_et_get_device_count(void);
GGML_BACKEND_API void ggml_backend_et_get_device_description(int devidx, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_et_get_device_memory(int devidx, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_et_buffer_type(size_t dev_num);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_et_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_et_reg(void);
```

## 二、注册与初始化：guid + iface + device

`ggml_backend_et_init()` 是 L3-02（注册表）与 L3-03（动态加载）的落点：`GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)`（文件最后一行）让这个后端能被 `dlopen` 加载。

建后端的动作只有三件事：造一个 `ggml_backend_et_context`（只存 devidx）、把 guid 与 `ggml_backend_et_i` 填进 `ggml_backend`、把注册表里的 device 挂上。**没有**编译产物缓存、**没有**算子集协商 —— 那些都发生在图之外。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
ggml_guid_t ggml_backend_et_guid(void) {
    static ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    return &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    if (!ggml_et_driver_init()) {
        return nullptr;
    }

    if (devidx >= (size_t) ggml_backend_et_get_device_count()) {
        return nullptr;
    }

    ggml_backend_et_context * ctx = new ggml_backend_et_context;
    ctx->devidx                   = (int) devidx;

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    return backend;
}
```

## 三、缓冲区虚表：ggml_backend_et_buffer_i

11 个槽位里 **4 个是 NULL**（`memset_tensor` / `set_tensor_2d` / `get_tensor_2d` / `reset`）、1 个恒 false（`cpy_tensor`，见 333-340）。真正干活的是：

- `free_buffer` -> `runtime->freeDevice()`（242-251）
- `get_base` -> buffer context 里的设备指针（253-256）
- `init_tensor` -> 只做一件事：把补齐出来的 padding 用**设备侧 memset 内核**清零（258-288）
- `set_tensor` / `get_tensor` -> `memcpyHostToDevice` / `memcpyDeviceToHost` + `waitForEvent`（290-331）
- `clear` -> 同样是设备侧 memset 内核（342-363）

这张表就是 L4-03 说的"后端对 buffer 的假设"：**内存只能来自 runtime，进出只能靠显式拷贝**。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
static const struct ggml_backend_buffer_i ggml_backend_et_buffer_i = {
    /* .free_buffer     = */ ggml_backend_et_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_et_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_et_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_et_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_et_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_et_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_et_buffer_clear,
    /* .reset           = */ NULL,
};
```

## 四、设备内存从哪来：alloc_buffer

分配路径三步：查 runtime -> 查 `rt::DeviceId` -> `runtime->mallocDevice()`。拿到的指针存进 buffer context，同时记住 `rtid`，`free_buffer` 时原样还回去。

注意这里**没有** mmap、没有 host 注册、没有统一地址空间的假设：`ggml_backend_dev_props.caps` 里 `host_buffer` 与 `buffer_from_host_ptr` 都是 false / NULL（`ggml-et.cpp:1647-1653`、`1681`）。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
static ggml_backend_buffer_t ggml_backend_et_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_et_buffer_type_context * btctx = (ggml_backend_et_buffer_type_context *) buft->context;

    ggml_backend_et_buffer_context * ctx = new ggml_backend_et_buffer_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        delete ctx;
        return nullptr;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    if (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        delete ctx;
        return nullptr;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->data = runtime->mallocDevice(ctx->rtid, size);
    if (ctx->data == nullptr) {
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_et_buffer_i, ctx, size);
}
```

## 五、数据进出：set_tensor / get_tensor

两个方向都是"建一个事件 -> 等事件"。`memcpyHostToDevice` 的第 5 个参数 `true` 是 barrier 语义，`runtime->waitForEvent()` 立刻同步等待 —— 也就是说这两个函数是**同步**的，与 L6-01 里 CUDA 用 stream 做异步拷贝的路子不同（ET 的异步留给 `set_tensor_async` 槽位）。

`ggml_backend_et_synchronize()`（553-571）则在需要时 `waitForStream` 并把流上累积的错误捞出来，有错直接 `abort()` —— 这也是一种"没有编译期检查，只能运行期兜底"的表现。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
static void ggml_backend_et_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                              ggml_tensor *         tensor,
                                              const void *          data,
                                              size_t                offset,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buffer->buft->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->data) + offset;
    const std::byte * src_ptr = static_cast<const std::byte *>(data);

    rt::EventId event = runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}

static void ggml_backend_et_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                              const ggml_tensor *   tensor,
                                              void *                data,
                                              size_t                offset,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buffer->buft->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    const std::byte * src_ptr = static_cast<const std::byte *>(tensor->data) + offset;
    std::byte *       dst_ptr = static_cast<std::byte *>(data);

    rt::EventId event = runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}
```

## 六、缓冲类型虚表：对齐与"我不是 host"

五个槽位回答了"这块内存长什么样"：

- `get_alignment` -> 设备属性里的 `cacheLineSize_`（413-422）；
- `get_max_size` -> 设备总内存（424-430）；
- `get_alloc_size` -> `ggml_nbytes_pad()`，即**按 cache line 补齐**的字节数（432-435）；
- `is_host` -> 恒 false（437-440）。

`ggml_nbytes_pad` 与第 7 幕的 `ne[0] % 16 == 0` 是同一件事的两面：ET 的硬件按 64 字节 cache line 搬运，所以"补齐"不是优化，是正确性前提。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
static size_t ggml_backend_et_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime || !buft->device) {
        return GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    return prop.cacheLineSize_;
}

static size_t ggml_backend_et_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    if (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        return dev_ctx->total_mem;
    }
    return SIZE_MAX;
}

static size_t ggml_backend_et_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes_pad(tensor);
}

static bool ggml_backend_et_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const struct ggml_backend_buffer_type_i ggml_backend_et_buffer_type_i = {
    /* .get_name         = */ ggml_backend_et_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_et_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_et_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_et_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_et_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_et_buffer_type_is_host,
};
```

## 七、后端虚表：ggml_backend_et_i

16 个槽位：**6 个实现、10 个 NULL**。这一点与 L5-01 的 CPU 后端数字一样，但含义不同：

- 没有 `graph_plan_*`：ET 不做"执行计划"（那是 CPU 后端的 work buffer 机制），  每次 `graph_compute` 现算；
- 没有 `event_*`：事件藏在 runtime 内部（`rt::EventId`），不暴露给 ggml；
- 没有 `graph_optimize`：**优化发生在构建期**（哪些 op 有内核、内核怎么写），  不在运行期做图变换。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
static const struct ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_et_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};
```

## 八、设备虚表：ggml_backend_et_device_i

这一层回答"调度器（L4-02）能对这个设备期待什么"：

- `get_type` 返回 `GGML_BACKEND_DEVICE_TYPE_GPU`（1635-1638）—— 尽管它不是 GPU，  这样调度器才会把它当"加速器"来切分图；
- `get_host_buffer_type` 返回 **CPU 的** buffer type（1667-1670）；
- `buffer_from_host_ptr` 是 NULL（1681）—— 明确拒绝"把 host 内存包成设备 buffer"；
- `supports_buft` 用**函数指针相等**来判断（1577-1580），即只有 ET 自己的 buffer 才算数；
- `offload_op` 对 `GET_ROWS` 返回 false，理由写在注释里：跨后端权重每次都要重拷，  把 266MB 的 embedding 表按 token 拷到设备不值得（1582-1596）。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
static const struct ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_memory,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ NULL,
    /* .event_free        = */ NULL,
    /* .event_synchronize = */ NULL,
};
```

## 九、支持性判定：supports_op 是运行期唯一的"守门人"

因为没有图级编译，**"哪些 op 能跑"必须在执行前由 `supports_op` 用 44 个 case 判完**（`ggml-et.cpp:868-1575`）。头几族已经把内核的硬件假设写成了前置条件：

- `SQR`：`ne[0] % 16 == 0` 且 dst/src0 都连续（873-876）；
- `UNARY`：**只**要求 `nb[0] == sizeof(float)`，注释明说更高维可以有任意步长，  内核按行用四个 `nb[]` 走（895-900）；
- `MUL` / `ADD` / `SUB`：行内连续，且 `nb[1] == ne[0] * sizeof(float)`（930-938）。

读这里的诀窍：**每一条 `&&` 都是设备内核对内存的一个假设**，与第 6、7 幕的行为一一对应。

<!-- src: ggml/src/ggml-et/ggml-et.cpp -->
```cpp
static bool ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    bool supported = false;
    switch (op->op) {
        case GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == sizeof(float) && ggml_is_contiguous(op);
            break;
        case GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_SUM_ROWS:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_MEAN:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            if (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == sizeof(float) && op->src[0]->nb[0] == sizeof(float)) {
```

## 十、懒加载：内核第一次用到才下载

`ggml_et_launch_kernel_internal()` 在真正 launch 前查一次 `loaded_kernels`：没有就 `ggml_et_load_kernel()`，加载完再查一次（防止加载失败静默继续）。

所以"加载 50 个内核"不是启动成本，而是**按模型实际用到的 op 逐个发生**的：跑一个小模型可能只碰到十几个内核。

<!-- src: ggml/src/ggml-et/ggml-et-kernels.cpp -->
```cpp
    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    if (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        if (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            return false;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        if (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            return false;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;
```

## 十一、uberkernel：把一段图打包成"一个设备内核"

ET 的图级融合有两条路：一是宿主侧的算子融合（`RMS_NORM+MUL`、`MUL_MAT+ADD`），二是 `GGML_ET_UBERKERNEL` 打开的 **uberkernel** —— 把整张图编译成"指令数组 + 参数块"，一次 launch 交给设备上的派发器。

这段代码是发射点：先 `memcpyHostToDevice` 两份缓冲（指令数组、参数块），再把两个**设备地址**填进 `ggml_et_uberkernel_params`，最后 launch 名字叫 `uberkernel` 的那个内核。注意参数里带的是 `reinterpret_cast<uint64_t>(slot.device_insts)` —— 又一次"裸设备地址过边界"。

<!-- src: ggml/src/ggml-et/ggml-et-kernels.cpp -->
```cpp
        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<const std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, true);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    true);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(sizeof(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, sizeof(params), shire_mask, false, false,
                                            &launch_event);
```

## 十二、设备侧派发器：uberkernel.c

设备侧的 uberkernel 就是一个 `switch (inst->kernel_id)`：取出第 i 条指令，按 `params_offset` 在参数块里定位它的参数，转成对应的参数结构体，调用该内核的入口。

`et_barrier_global(32ULL)` 是**设备侧的全局同步** —— 这正是 `IDeviceLayer` / `IRuntime` 那层存在的理由：子内核之间没有天然的可见性边界，必须显式同步（`docs/backend/ET.md:161-167` 明确写了"no natural memory visibility horizon"）。

对照 L4-02 的调度器：那是在 **host 侧按 buffer 类型切图**；uberkernel 是在 **设备侧按指令序执行**。两者解决的不是同一个问题。

<!-- src: ggml/src/ggml-et/et-kernels/src/uberkernel.c -->
```c
int entry_point(struct ggml_et_uberkernel_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

    struct ggml_et_uberkernel_inst * insts       = (struct ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    if (!insts || !params_blob || params->inst_stride < sizeof(struct ggml_et_uberkernel_inst)) {
        return -1;
    }

    for (uint32_t i = 0; i < params->num_insts; ++i) {
        struct ggml_et_uberkernel_inst * inst =
            (struct ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        void * inst_params = params_blob + inst->params_offset;
        int    rc          = -1;

        et_barrier_global(32ULL);

        switch (inst->kernel_id) {
```

## 十三、连 memset 都是一个设备内核

`ggml_et_memset()` 是宿主侧唯一的"内存操作"入口，它做的事只有一件：把 `memset_params` 填好，然后 `ggml_et_launch_kernel(dev_ctx, "memops", ...)`。

为什么不用 host 的 `memset`？因为这块内存是**设备内存**（`runtime->mallocDevice`），host 指针不能直接写。这也解释了 `init_tensor` 里"清 padding"为什么要绕这么大一圈（258-288）。

<!-- src: ggml/src/ggml-et/ggml-et-memops.cpp -->
```cpp
#include "ggml-et-memops.h"

#include "ggml-et-kernels.h"
#include "ggml-impl.h"

// Kernel parameter structure for memset operation
struct memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    void *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, void * dst_ptr, uint8_t value, size_t size) {
    if (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        return false;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    bool success = ggml_et_launch_kernel(dev_ctx, "memops", &params, sizeof(params));

    if (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        return false;
    }

    return true;
}
```

## 十四、设备侧的同名参数结构体

设备侧把同一份结构体**再声明一遍**，并用注释提醒"必须与宿主侧一致"。这就是本课说的"共享表示"的代价：没有共享头文件，靠**约定 + 注释**对齐字段。

注意 `void * dst_ptr` —— 参数里带的仍然是设备地址。

<!-- src: ggml/src/ggml-et/et-kernels/src/memops.c -->
```c
// Operation identifiers for memops kernel
enum ggml_et_memop_type {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation parameters (must match host-side struct in ggml-et-memops.cpp)
struct memset_params {
    uint32_t op_type;
    uint32_t value;
    void *   dst_ptr;
    size_t   size;
};
```

## 十五、设备侧的内存契约与量化块

设备侧也有一个"连续性判定"，但比宿主宽松：`ne[i] == 1` 的轴不参与比较，因为这些轴的步长不可观测。这与 `supports_op` 里"只要求 `nb[0]`"是同一个思路。

更有意思的是量化块：`quants.h` 直接 `#define GGML_COMMON_DECL_C` 然后 include `ggml-common.h` —— 设备内核用的 `block_q8_0` / `block_q4_0` / `block_q4_K` 就是 L1-04 讲的那份定义，**同一个头文件，两端共用**。

<!-- src: ggml/src/ggml-et/et-kernels/src/ggml_tensor.h -->
```c
static inline int ggml_tensor_is_contiguous(const struct ggml_tensor * t, int type_size) {
    int64_t expected = type_size;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            return 0;
        }
        expected *= t->ne[i];
    }
    return 1;
}
```

## 十六、设备侧复用量化块定义

同一件事的另一半证据在这里：`GGML_COMMON_DECL_C` + `#include "ggml-common.h"` 之后，下面三个 `dequantize_*_block` 用的就是 ggml 的块结构。

这解释了为什么 ET 只支持 `q8_0` / `q4_0`（部分 `q4_K`，见 `docs/backend/ET.md:22`）：设备侧要**逐个手写**反量化，支持一种格式就多一份裸机代码。

<!-- src: ggml/src/ggml-et/et-kernels/src/quants.h -->
```c
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16
```

## 十七、没有编译器，就靠 CPU 对拍

ET 后端自带一套"逐算子对照"设施：`ggml_et_cpu_compare_ctx` 同时持有 CPU 与 ET 两侧的张量、图和数据指针，先（`init_pre`）把输入拷到 CPU 侧，等 ET 内核跑完再（`compute_and_check`）用 CPU 后端算一遍并按容差比较。

每个 op 家族的配置（从 `ggml-et-ops.cpp:12` 的 rope 到 `2217` 的 gated_delta_net）都是 `enabled = false`：默认关闭，调试时打开。**一个没有图级编译器的后端，只能用对拍来定位精度问题** —— 这与 L5-04 里 CPU 侧靠 repack/量化内核的单元测试保证正确性形成对照。

<!-- src: ggml/src/ggml-et/ggml-et-cpu-compare.h -->
```c

// CPU comparison context for a single operation
struct ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_graph;
    void *         cpu_src0_data;
    void *         cpu_src1_data;
    void *         cpu_src2_data;
    void *         cpu_dst_data;
    void *         et_dst_data;
    size_t         src0_size;
    size_t         src1_size;
    size_t         src2_size;
    size_t         dst_size;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, const ggml_tensor * node, ggml_op op);
```

## 十八、内核加载策略（源码自述）

`ggml-et-kernels.h` 的注释把加载策略写得很清楚，也是本课"编译在图之外"这条结论的直接依据：先试 `${GGML_ET_KERNELS_PATH}/${kernel_name}.elf`（开发期覆盖），失败则回退到**嵌进库里的那份**；两者都没有才算失败。

注意第 24 行的默认路径提示 `/opt/et/ggml/kernels/` —— 同样的 ELF 既能被嵌进二进制，也能放在设备可见的目录里。

<!-- src: ggml/src/ggml-et/ggml-et-kernels.h -->
```cpp
#define ET_TRACE_BUFFER_SIZE (1024 * 1024 * 8UL)

// Load kernel from file or embedded data and store handle in device context
// Returns true on success, false on failure
//
// Loading strategy:
// - If GGML_ET_KERNELS_PATH env var is set: tries to load from ${GGML_ET_KERNELS_PATH}/${kernel_name}.elf
// - If file not found or env var not set: falls back to embedded kernel data
// - Returns false if kernel cannot be loaded from either source
//
// Kernel is loaded using the device's default stream
bool ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name);

// Launch kernel with parameters on device's default stream
// Performs lazy loading: automatically loads kernel if not already loaded
// Kernel path: ${GGML_ET_KERNELS_PATH}/${kernel_name}.elf (default: /opt/et/ggml/kernels/)
// Returns true on success, false on failure
// Execution is synchronous - waits for completion
bool ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           const std::string &              kernel_name,
                           void *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask       = 0xFFFFFFFF,
                           bool                             enable_print     = false,
                           bool                             sync_error_check = false);
```

## 十九、50 个设备内核按族分类

`et-kernels/CMakeLists.txt:28-79` 的 `KERNELS` 列表**正好 50 项**，与 `et-kernels/src/*.c` 的 50 个文件一一对应；每个文件用 `add_riscv_executable()` 编译成 `<名字>.elf`（`et-kernels/CMakeLists.txt:41-43`），编译选项里有 `-nostdlib` / `-ffreestanding` / `-march=rv64imf`，并且**构建后跑 `check_unimplemented_instructions.sh`**：一旦编译器生成了硬件没实现的浮点指令就直接让构建失败。

这 50 个文件按族分成 9 组（下表）；其中 5 个文件名里带 `_me` / `matrix_engine`，是走 **tensor engine（矩阵引擎）** 指令的版本，与走向量指令的通用版本并存。`uberkernel.c` 是唯一不实现算子的内核 —— 它是设备侧的派发器（第十二节）。

| 族 | 文件数 | 行数 | 代表 |
|---|---|---|---|
| mul_mat 稠密 | 7 | 1902 | `mul_mat_Q8_0.c` / `mul_mat_f32_matrix_engine.c` |
| mul_mat_id（MoE） | 3 | 617 | `mul_mat_id_Q4_0.c` |
| 归一化 | 5 | 1296 | `rms_norm_f32.c` / `rms_norm_mul_f32.c` |
| 注意力与位置 | 4 | 2571 | `flash_attn_ext_f32.c` / `rope_f32.c` / `softmax_f32.c` |
| 逐元素 | 6 | 1990 | `unary_f32.c` / `scale_f32.c` / `sqr_f32.c` |
| 布局/搬运/归约 | 16 | 2979 | `cont_f32.c` / `get_rows_f32.c` / `set_rows_f32.c` |
| 卷积 | 2 | 937 | `conv_2d_f32_me.c` / `im2col.c` |
| 循环/SSM | 5 | 1213 | `ssm_scan_f32.c` / `rwkv_wkv7_f32.c` / `gated_delta_net_f32.c` |
| 基础设施 | 2 | 678 | `memops.c`（memset）/ `uberkernel.c`（派发器） |

合计 **50 个文件 / 14183 行**（`wc -l` 实测）。这些内核目前**没有**统一的公共接口头：每个内核在自己的 `.c` 里定义参数结构体，宿主侧在 `ggml-et-ops.h` 里再写一遍（第十五、十六节点出了这个约定的两端）。

## 二十、uberkernel 的 host/device 共享 ABI

第十一节里那两个设备地址（指令数组、参数块）指向的内存，格式就是这 17 行：`ggml_et_uberkernel_inst` 是一条指令（内核编号 + 标志 + 参数在参数块里的偏移与长度），`ggml_et_uberkernel_params` 是整批指令的头。

这个头文件被**两端同时 include**：宿主侧经 `ggml-et-common.h:4` 传递到 `ggml-et-kernels.cpp:344-349`（构造 `ggml_et_uberkernel_params`），设备侧由 `uberkernel.c:1` 直接 include。它是本课"共享表示"最纯粹的例子 —— 连"指令流"这种通常属于编译器内部的东西，在这里也是一份两边都认的结构体。

<!-- src: ggml/src/ggml-et/ggml-et-uberkernel-common.h -->
```c
#pragma once

#include <stdint.h>

struct ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_offset;
    uint32_t params_size;
};

struct ggml_et_uberkernel_params {
    uint32_t num_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};
```

## 二十一、设备侧共享头：6 个文件的分工

上一节按行数把 68 个文件分成四族；这一节把"设备侧共享头"这一族再拆开 —— 它们都是**两端共用同一套语义**的体现（行数为 `wc -l` 实测）：

| 文件 | 行数 | 分工（取自文件自己的头部注释） |
|---|---|---|
| `ggml_tensor.h` | 44 | 内核参数结构体（与宿主同名）+ 连续性判定；第十五节逐字引用 |
| `quants.h` | 72 | 反量化助手 + 复用 `ggml-common.h` 的块定义；第十六节逐字引用 |
| `platform.h` | 545 | 裸机 HAL：hart 与线程数、屏障/信号量、tensor engine 等待、L1/L2 scratchpad 寻址 |
| `math_fp.h` | 299 | 硬件未实现指令的替代实现（FP16 转换、三角、除法）| 
| `block_ops.h` | 997 | 向量块运算库（建立在 `math_fp.h` + `quants.h` 之上）| 
| `tensor.h` | 897 | ET-SoC 张量指令的 CSR 封装：`tensor_load` / `tensor_store` / `tensor_fma` 等 |

值得留意的是 `platform.h` 里的两个常数：`SOC_MINIONS_PER_SHIRE 32`、`NUM_HARTS_PER_MINION 2`（`platform.h:17-18`）—— 它们决定了第六节里"线程数 = popcount(shire_mask) x 32 x 2"这条公式。

## 二十二、宿主侧的参数结构体与 CPU 对拍实现

`ggml-et-ops.h` 里每个算子家族都有自己的参数结构体 —— 一共 **37 个**（`grep -c "^struct ggml_et_.*_params {"` 实测）。注意字段写的是 `ggml_tensor` 而不是指针：**结构体按值**，这正是第五幕那条机制在头文件里的样子。设备侧 `ggml_tensor.h` 里是同一批结构体的另一份声明（多了 `struct` 关键字），两端靠字段一致对齐。

同一族的 `ggml-et-cpu-compare.cpp`（502 行）是第十七节那套对拍设施的**实现**：它 `ggml_backend_cpu_init()`（104）建一个临时的 CPU 后端，把输入拷到 CPU 侧（84-92），在 ET 内核跑完后再 `ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph)`（360）算一遍参考结果，最后取回 ET 的输出比较（376）。一个没有图级编译器的后端，只能这样用"另一个后端"来当参照。

<!-- src: ggml/src/ggml-et/ggml-et-ops.h -->
```cpp
struct ggml_et_binary_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

// Q8_0 mul_mat with optional residual bias.
// bias.data == NULL means "no bias" - kernel skips the add.
// When non-NULL, bias must have the same shape and strides as dst.
struct ggml_et_mm_q8_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    ggml_tensor bias;
};
```

---

## 说明

- 本课覆盖声明是 `tools/plan_matrix.py --files` 里 L7-04 的**全部 68 个文件**：`ggml/include/ggml-et.h`（1）+ `ggml/src/ggml-et/*.{cpp,h}`（11）+ `ggml/src/ggml-et/et-kernels/src/*.h`（6）+ `et-kernels/src/*.c`（50）。合计 23191 行。
- **正文引用但不计入覆盖率的文件**（不在本视角的源文件后缀/路径里，故未进覆盖声明）：`ggml/src/ggml-et/CMakeLists.txt`（内核清单与嵌入）、`ggml/src/ggml-et/et-kernels/CMakeLists.txt`（RISC-V 交叉编译）、`ggml/src/ggml-et/et-kernels/scripts/check_unimplemented_instructions.sh`、`docs/backend/ET.md`（上游文档）、`et-kernels/src/RunBackend.sh`、以及同目录下不在覆盖域的 `crt.S` / `linker.ld`。另有若干对照引用属于别的课：`ggml/src/ggml-cuda/scale.cu`（L6-01）、`ggml/src/ggml-cuda/ggml-cuda.cu`（L6-01）、`ggml/src/ggml-cann/acl_tensor.cpp`（L7-01）、`ggml/src/ggml-openvino/ggml-decoder.cpp`（L7-03）、`ggml/src/ggml-common.h`（L1-04）。
- **关于课名与计划口径的修正**：计划文档把这一课记为"ExecuTorch 后端"，讲解要点是"ExecuTorch 的委托机制、图导出与运行时"。实测（`grep -rni` 全目录）`ggml/src/ggml-et/` 里没有 `executorch` / `torch` / `delegate` / `.pte` 任何符号；实际机制是"ET 平台 SDK（`dev::IDeviceLayer` + `rt::IRuntime`）+ 构建期交叉编译的内核 ELF"。本课按代码实际机制讲解，并在第 3、8 幕显式对照这两种口径。
- **未实测的部分**（明确边界）：ET-SoC 是专用硬件，本机没有 `/opt/et` 平台与 RISC-V 工具链，因此本课**没有**真的构建或运行过这个后端。所有结论都来自**逐字引用的源码**与**行号可核对的注释**；凡是"注释声称"的地方（如硬件未实现指令、L2/L1 不连贯）本课都标明出处，未当作已实测的行为。
