<!-- llama-coverage
ggml/include/ggml-blas.h
ggml/include/ggml-opencl.h
ggml/include/ggml-rpc.h
ggml/include/ggml-webgpu.h
ggml/include/ggml-zdnn.h
ggml/include/ggml-zendnn.h
ggml/src/ggml-blas/ggml-blas.cpp
ggml/src/ggml-musa/mudnn.cu
ggml/src/ggml-musa/mudnn.cuh
ggml/src/ggml-opencl/cl-program-cache.cpp
ggml/src/ggml-opencl/cl-program-cache.h
ggml/src/ggml-opencl/fa_tune.h
ggml/src/ggml-opencl/ggml-opencl.cpp
ggml/src/ggml-opencl/libdl.h
ggml/src/ggml-rpc/ggml-rpc.cpp
ggml/src/ggml-rpc/transport-apple.cpp
ggml/src/ggml-rpc/transport-apple.h
ggml/src/ggml-rpc/transport.cpp
ggml/src/ggml-rpc/transport.h
ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
ggml/src/ggml-webgpu/ggml-webgpu.cpp
ggml/src/ggml-webgpu/pre_wgsl.hpp
ggml/src/ggml-zdnn/common.hpp
ggml/src/ggml-zdnn/ggml-zdnn.cpp
ggml/src/ggml-zdnn/mmf.cpp
ggml/src/ggml-zdnn/mmf.hpp
ggml/src/ggml-zdnn/utils.cpp
ggml/src/ggml-zdnn/utils.hpp
ggml/src/ggml-zendnn/ggml-zendnn.cpp
-->

# L7-06 · 小后端合集：OpenCL / WebGPU / MUSA / RPC / BLAS / ZenDNN / zDNN — 源文件

**一句话**：这一课把七个"小"后端放在一起看 —— 它们填的是**同一张** `ggml_backend_i` 虚表（L3-01 讲过的那三张表），但数据面的差别大到荒谬：BLAS 的数据就在 CPU 内存里、WebGPU 的数据在浏览器的 GPU 缓冲里、zDNN 的数据被变换成硬件专用布局、RPC 的数据在**另一台机器**上。

"小"指的是**要写多少自有代码**，不是能力：MUSA 目录只有 2 个文件（124 行）——因为整个后端复用 `ggml-cuda` 的源码，MUSA 只补了一层 MUDNN 胶水；而 OpenCL 单个 `.cpp` 就有 29502 行。两者在调度器眼里没有区别。

本课覆盖 `python3 tools/plan_matrix.py --files` 指派给 `L7-06` 的**全部 29 个文件**（7 个后端目录 + 6 个公共头；`ggml/src/ggml-musa/CMakeLists.txt` 只是构建脚本，用来解释"MUSA 为什么只有 2 个文件"，不计入覆盖声明）。

---

## 一、七个后端，七份"最小可行后端"清单

本课覆盖 `plan_matrix.py` 指派给 `L7-06` 的全部 29 个文件。规模实测（`wc -l`）：

| 后端 | 本课文件 | 总行数 | 特有的东西 |
|---|---|---|---|
| BLAS | 2 | 555 | 只有 1 个 `.cpp`：一个 switch + cblas |
| MUSA | 2 | 124 | 只有 MUDNN 胶水；算子实现复用 ggml-cuda |
| ZenDNN | 2 | 860 | 只翻译 matmul（含 MoE 的 group matmul） |
| zDNN | 7 | 904 | 私有 buffer type + ztensor 变换 |
| RPC | 6 | 3694 | 传输层（TCP / RDMA）+ 协议结构 |
| WebGPU | 4 | 9185 | WGSL shader 库 + WGSL 预处理器 |
| OpenCL | 6 | 30227 | 运行期编译 .cl / 磁盘二进制缓存 / Adreno 调参表 |

合计 **29 文件 / 45549 行**。MUSA 目录里只有 `mudnn.cu` 与 `mudnn.cuh` 两个自有源文件，其余 `.cu` 来自 `../ggml-cuda/`（构建脚本 `ggml/src/ggml-musa/CMakeLists.txt` 用 `file(GLOB GGML_SOURCES_MUSA "../ggml-cuda/*.cu")` 收集）—— 该构建脚本不计入本课覆盖声明。

公共头都很薄，形状只有两种：

<!-- src: ggml/include/ggml-blas.h -->
```c
// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_blas_init(void);

GGML_BACKEND_API bool ggml_backend_is_blas(ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_BACKEND_API void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_blas_reg(void);
```

## 二、WebGPU 的公共头：连 init 都只为 examples 服务

`ggml_backend_webgpu_init()` 的注释写着 "Needed for examples in ggml"：正式路径是注册表 + 设备（L3-02），这个入口只是给例子程序用的。

<!-- src: ggml/include/ggml-webgpu.h -->
```c
#define GGML_WEBGPU_NAME "WebGPU"

// Needed for examples in ggml
GGML_BACKEND_API ggml_backend_t ggml_backend_webgpu_init(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_webgpu_reg(void);
```

## 三、OpenCL 的公共头：连 host buffer type 都暴露出来

OpenCL 除了设备 buffer type，还导出一个 **host** buffer type —— 因为它的设备内存与宿主内存是两份，中间要显式搬运（`clEnqueueWriteBuffer`，见 `ggml-opencl.cpp:9786-9815`）。

<!-- src: ggml/include/ggml-opencl.h -->
```c
//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);
```

## 四、ZenDNN 的公共头：一个 init、一个 set_n_threads、一个 reg

ZenDNN 是 AMD CPU 上的算子库（源码注释：AMD optimized primitives backend for GGML）。它的公共 API 与 BLAS 几乎同形 —— 因为两者都是"复用宿主内存、只加速个别算子"的后端。

<!-- src: ggml/include/ggml-zendnn.h -->
```c
// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_zendnn_init(void);

GGML_BACKEND_API bool ggml_backend_is_zendnn(ggml_backend_t backend);

// number of threads used for zendnn operations
GGML_BACKEND_API void ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_zendnn_reg(void);
```

## 五、RPC 协议：版本号与一个 static_assert

RPC 是唯一一个**带协议版本**的后端：`GGML_OP_COUNT` 变化会改变 `rpc_tensor` 的含义，所以 `GGML_OP_COUNT` 一改，就必须同步更新 patch 版本号 —— 这条约束写进了头文件，编译期就会检查。

<!-- src: ggml/include/ggml-rpc.h -->
```c
#define RPC_PROTO_MAJOR_VERSION    7
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16
```

## 六、RPC 的传输层：一条 socket，两种实现

`socket_t` 是一个 pimpl：上层只看到 `send_data` / `recv_data` / `flush`。`flush` 的注释说明了下层的差别 —— RDMA 传输会把写入合并成固定大小的帧，尾帧必须在消息边界显式 post；TCP 上它是 no-op。

<!-- src: ggml/src/ggml-rpc/transport.h -->
```c
struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();
```

## 七、transport.cpp：socket_t 把一切转发给 impl

这三个函数是整条 RPC 数据通路的最底层。TCP 实现里 `create_server` 只是 `socket/bind/listen`（`transport.cpp:656`），`connect` 还会设置 `TCP_NODELAY`（`:683`）。

<!-- src: ggml/src/ggml-rpc/transport.cpp -->
```c
bool socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

bool socket_t::flush() {
    return pimpl->flush();
}
```

## 八、Apple 的 RDMA：为 Thunderbolt 单独写一个传输

Apple 的 RDMA 与 Linux 差别大到"值得单独一个实现"：UC 而非 RC 队列、固定 128 KiB stride、依赖硬件信用流控。注意 `flush()` 的语义正是被它逼出来的。

<!-- src: ggml/src/ggml-rpc/transport-apple.h -->
```c
struct apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    static std::unique_ptr<apple_rdma> probe(int fd, const uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(const uint8_t * caps);

    bool send(const void * data, size_t size);
    bool recv(void * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() const;

private:
    struct impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
```

## 九、transport-apple.cpp：常量里写满了硬件事实

这些常量不是随意取的：4 KiB 是 Thunderbolt 帧、128 KiB stride = 32 帧，一次 SEND 必须覆盖与对应 RECV 相同数量的帧 —— 所以"半满也要发满一个 stride"。

<!-- src: ggml/src/ggml-rpc/transport-apple.cpp -->
```c
// Apple RDMA-over-Thunderbolt (see Apple TN3205).
//
// Apple's RDMA is quite different from what's supported in Linux - deserving of its own transport implementation.
// see https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt for details
// at a high level the main differences are:
// UC(unreliable connection) on Apple vs RC(reliable connection) QP transport types on Linux (though in practice UC on Apple is still lossless)
// fixed 128KiB stride on Apple vs variable chunk size on Linux
// relying on Apple's hardware credit based flow control vs RNR NAKs + retries on Linux
//
// on Apple a SEND and its corresponding RECV must cover the same number of 4 KiB Thunderbolt frames,
// so every SEND posts a whole 128KiB stride over the wire, even when partially filled.
// (In testing 128KiB was the best performing among 32, 64, 128, 256)

static constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
static constexpr int      RDMA_NBUF        = 16;          // ring depth (frames per direction)
static constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
static constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
static constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
static constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
```

## 十、WebGPU 的 shader 库：WGSL 与一个自制的预处理器

`ggml-wgsl-shaders.hpp` 是构建期生成的内核库（不在本课覆盖域内），`pre_wgsl.hpp` 则是一个运行期的小预处理器：因为不同适配器的 limits 不同，同一个算子要按能力挑不同变体。

<!-- src: ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp -->
```c
#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "ggml-impl.h"
#include "ggml-wgsl-shaders.hpp"
#include "ggml.h"
#include "pre_wgsl.hpp"

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define GGML_WEBGPU_F16_SIZE_BYTES                   2
#define GGML_WEBGPU_F32_SIZE_BYTES                   4
#define GGML_WEBGPU_I32_SIZE_BYTES                   4
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
```

## 十一、pre_wgsl：只要 include 路径与宏，就够用了

整个预处理器只需要 `Options{include_path, macros}` 两个参数 —— 它服务的是 WGSL 源码拼接，不是通用编译器。

<!-- src: ggml/src/ggml-webgpu/pre_wgsl.hpp -->
```c
#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pre_wgsl {

//==============================================================
// Options
//==============================================================
struct Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};
```

## 十二、OpenCL 的 libdl.h：给厂商预编译内核留的入口

非 Windows 分支就是 `dlopen` / `dlsym` / `dlerror` 三件套。它在 OpenCL 后端里只服务一件事：运行期尝试加载 Adreno 的**预编译内核二进制库**（`libadreno-opencl-kernels.so` / `adreno-opencl-kernels.dll`，定义在 `ggml-opencl.cpp:21-23`）；加载失败就回退到内置内核源码（`ggml-opencl.cpp:6726-6744`）。注意 OpenCL 本身仍是构建期依赖（`find_package(OpenCL REQUIRED)`），这里动态加载的只是内核。

<!-- src: ggml/src/ggml-opencl/libdl.h -->
```c
using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

static inline dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

static inline void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

static inline const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}
```

## 十三、OpenCL 的磁盘缓存：键是怎么算出来的

缓存键把三样东西拼起来做 SHA-256：内核源码、编译选项、设备身份（`key_suffix` 里含 CL_DEVICE_NAME / CL_DRIVER_VERSION / CL_PLATFORM_VERSION 与格式版本）。所以换驱动、换编译选项都会自动 miss，不会读到错误的二进制。

<!-- src: ggml/src/ggml-opencl/cl-program-cache.cpp -->
```c
std::string compute_key(const std::string & key_suffix,
                        const char *        source,
                        const std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    static const uint8_t sep = 0;
    sha256_update(c, source,             strlen(source));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.data(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, key_suffix.data(),  key_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    return sha256_hex(digest);
}
```

## 十四、缓存开关与文件布局

缓存的激活方式、目录规则与文件头布局都写在头文件注释里（这里只引前 14 行）。

<!-- src: ggml/src/ggml-opencl/cl-program-cache.h -->
```c
// On-disk cache for OpenCL cl_program binaries. Lets a fresh process skip the
// expensive clBuildProgram-from-source step when a binary for the exact same
// (source, compile options, device, driver, platform) was previously saved.
//
// Activation: default on via GGML_OPENCL_KERNEL_CACHE_DIR:
//   unset / empty / "1" / "default"      : platform default cache dir
//                                          (%LOCALAPPDATA%\llama.cpp\cl-cache,
//                                          ~/Library/Caches/llama.cpp/cl-cache,
//                                          <temp dir>/llama.cpp/cl-cache elsewhere)
//   "0" / "off" / "none" / "disable(d)"  : disabled (all functions no-op)
//   any other value                      : used verbatim as the cache path
// If the chosen directory cannot be created/used, the cache silently disables
// itself for the process and falls back to source compile.
// GGML_OPENCL_KERNEL_CACHE_DEBUG=1 prints a HIT/MISS/SAVE trace (with a running
```

## 十五、fa_tune.h：把调参从代码里赶出去

Flash-Attention 的 tile 参数按 (dk, dv) 列表给出，注释写明覆盖 Adreno 7xx/8xx 与 X1 系列；抽成头文件的理由写在文件开头：调参数字好找、好改，而派发与编译逻辑留在主文件里。

<!-- src: ggml/src/ggml-opencl/fa_tune.h -->
```c
// Per-(dk, dv) FA config; shared by dispatch and supports_op.
struct ggml_opencl_fa_dim {
    int dk; int dv; int bm; int bn; int n_split; int nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
static const ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};
```

## 十六、MUSA：整个后端只有一份胶水头

MUSA 目录里唯一的头文件只声明了一个函数：`mudnnMemcpyAsync`。也就是说，这个后端对 ggml 的"新增能力"只有一个异步拷贝 —— 其余全部来自 ggml-cuda。

<!-- src: ggml/src/ggml-musa/mudnn.cuh -->
```c
#pragma once

#include "ggml-cuda/common.cuh"
#include "ggml.h"

// Asynchronously copies data from src tensor to dst tensor using the provided context.
// Returns a musaError_t indicating success or failure.
musaError_t mudnnMemcpyAsync(
    ggml_backend_cuda_context &ctx,
    const ggml_tensor *dst,
    const ggml_tensor *src
);
```

## 十七、mudnn.cu：把 ggml_tensor 翻译成 mudnn::Tensor

dims 直接取 `ne[]`，strides 取 `nb[] / element_size`（换成以元素为单位）—— 然后 `SetAddr` 把指针交出去。整个翻译过程没有任何数据搬运。

<!-- src: ggml/src/ggml-musa/mudnn.cu -->
```c
// Extracts dimensions and strides from a ggml_tensor
int get_ggml_dims_and_strides(const ggml_tensor* tensor,
                              std::vector<int64_t>& dims,
                              std::vector<int64_t>& strides) {
    const int ndims = ggml_n_dims(tensor);
    const size_t element_size = ggml_element_size(tensor);

    dims.resize(ndims);
    strides.resize(ndims);

    for (int i = 0; i < ndims; ++i) {
        dims[i] = tensor->ne[i];
        strides[i] = tensor->nb[i] / static_cast<int64_t>(element_size);
    }
```

## 十八、ZenDNN：翻译成一次 matmul_direct 调用

注释把每个参数的含义都写清楚了：行主序、不转置 B、转置 A（因为 ggml 是列主序）、alpha/beta、bias、is_weights_const。量化权重（block_q8_0）额外补上 scale 的维度。

<!-- src: ggml/src/ggml-zendnn/ggml-zendnn.cpp -->
```c
static bool ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               const TA * A, int64_t lda, const TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    if constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', false, true,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        true,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    if (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<int>(status));
        return false;
    }
    return true;
}
```

## 十九、ZenDNN 的算子覆盖与回退

和 BLAS 一样，它只认少数几个 op，default 走 `GGML_ABORT`；`supports_op` 里还有一个可关闭的"自适应回退"开关（函数在 `ggml-zendnn.cpp:688-692`，读取环境变量 `GGML_ZENDNN_ADAPTIVE_FALLBACK`）。

<!-- src: ggml/src/ggml-zendnn/ggml-zendnn.cpp -->
```c
static ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, node);
                break;
            case GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}
```

## 二十、zDNN：每个 buffer 都带一个硬件张量描述符

`ggml_backend_zdnn_buffer` 里除了 data/size，还挂着 `pre_tfm_desc`、`tfm_desc` 与 `zdnn_ztensor` —— 这三样就是"硬件专用数据面"的直接证据。

<!-- src: ggml/src/ggml-zdnn/common.hpp -->
```c
struct ggml_backend_zdnn_buffer {
    void * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    char name[GGML_MAX_NAME];
};
```

## 二十一、zDNN 的 set_tensor：拷贝之后再变换一次

先 `memcpy` 到自己的缓冲，再（若尚未变换）调用 `ggml_zdnn_load_tensor` 做硬件变换；计算 buffer 被复用时会先 `zdnn_reset_ztensor` 复位（源码注释指向一个真实 bug：LLAMA_SET_ROWS）。

<!-- src: ggml/src/ggml-zdnn/ggml-zdnn.cpp -->
```c
static void ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    if (tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE && extra->ztensor.is_transformed) zdnn_reset_ztensor(&extra->ztensor);
    if (extra->ztensor.is_transformed == false) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
```

## 二十二、zDNN 的 buffer type：明确声明"不是 host"

`is_host` 返回 false，注释解释得直白：*while it resides in host memory, additional transformation is needed*。对齐要求 256 字节。

<!-- src: ggml/src/ggml-zdnn/ggml-zdnn.cpp -->
```c
static bool ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    return false;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name       = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,  // defaults to ggml_nbytes
            /* .is_host        = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ NULL,
    };
```

## 二十三、mmf.cpp：真正落到硬件的那一次调用

前面一大堆断言都在校验"tensor 的 ne[] 与预先算好的 pre_tfm_desc 一致"；最后一行才是硬件调用，接着还要把输出从 DLF16 变换回 FP32（源码 TODO 承认这一步低效）。

<!-- src: ggml/src/ggml-zdnn/mmf.cpp -->
```c
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[1]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim1  == inputs->ne[0]  && "inputs_extra->pre_tfm_desc.dim1 must match inputs->ne[0]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim2  == inputs->ne[1]  && "inputs_extra->pre_tfm_desc.dim2 must match inputs->ne[1]");

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        false, true, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->data));
```

## 二十四、zDNN 的接口声明与类型映射

`mmf.hpp` 只有矩阵乘一个函数 —— 与 `supports_op` 里"只支持 MUL_MAT"完全对应；`utils.hpp` 暴露的是建 tensor / 载入 tensor / 初始化 tensor 三件事。

<!-- src: ggml/src/ggml-zdnn/mmf.hpp -->
```c
#ifndef GGML_ZDNN_MMF_HPP
#define GGML_ZDNN_MMF_HPP

#include "common.hpp"

void ggml_zdnn_mul_mat_f(
    const ggml_backend_zdnn_context * ctx,
    const               ggml_tensor * src0,
    const               ggml_tensor * src1,
                        ggml_tensor * dst);

#endif  // GGML_ZDNN_MMF_HPP
```

## 二十五、utils.hpp 与类型映射表

`ggml_zdnn_type_mapping` 把 ggml 类型映射到 zDNN 类型；遇到不支持的类型直接 abort。

<!-- src: ggml/src/ggml-zdnn/utils.hpp -->
```c
#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#include "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

void ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      const ggml_tensor       * src,
                      const int64_t           * ne,
                      const zdnn_data_layouts   layout);

void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, void * buffer);

```

---

## 说明

- 正文表格里的行数按 `wc -l` 实测（例如 `ggml-blas.cpp` = 530 行）；README 的「覆盖的源文件」表由 `lessonkit` 生成，按换行符切分计数，每个文件会比 `wc -l` 多 1（文件末尾的换行也算一行）。
- WebGPU 的 WGSL 内核文本来自构建期生成的 `ggml-wgsl-shaders.hpp`（源文件在 `ggml/src/ggml-webgpu/wgsl-shaders/*.wgsl`，由 `embed_wgsl.py` 嵌入），该生成头不在本课覆盖域内；本课引用的是使用它的 `ggml-webgpu-shader-lib.hpp`。
- OpenCL 本身是构建期依赖（`ggml/src/ggml-opencl/CMakeLists.txt` 里 `find_package(OpenCL REQUIRED)`）；`libdl.h` 动态加载的是 Adreno 预编译内核库，不是 OpenCL 运行时。构建脚本不计入本课覆盖声明。
- 本课覆盖 29 个文件，全部进覆盖声明；`ggml/src/ggml-musa/CMakeLists.txt` 只在第一段讲解里被引用来说明"MUSA 为什么只有 2 个自有源文件"，不计入覆盖声明。
- 所有行数来自 `wc -l`，所有行号来自上游 `v0.5.0`（commit `7fe450e19305`）；引用一律由 `lessonkit` 按行号抽取，未手抄。
