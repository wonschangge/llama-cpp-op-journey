<!-- llama-coverage
ggml/src/ggml-cuda/mma.cuh
ggml/src/ggml-cuda/mmq-load-tiles.cuh
ggml/src/ggml-cuda/mmq-vec-dot.cuh
ggml/src/ggml-cuda/mmq.cu
ggml/src/ggml-cuda/mmq.cuh
ggml/src/ggml-cuda/mmvf.cu
ggml/src/ggml-cuda/mmvf.cuh
ggml/src/ggml-cuda/mmvq.cu
ggml/src/ggml-cuda/mmvq.cuh
ggml/src/ggml-cuda/vecdotq.cuh
ggml/src/ggml-cuda/ggml-cuda.cu
-->

# L6-02 · ★ CUDA 量化矩阵乘：mmq / mmvq / mmvf — 源文件

**一句话**：`GGML_OP_MUL_MAT` 在 CUDA 后端上**不是一个内核，而是五个**。`ggml_cuda_mul_mat()` 按固定顺序试 `should_use_*` 判据，第一个为真的赢；决定胜负的关键量是 **`ne11`** —— src1 的列数，也就是"一次算几个 token"。

本课只讲量化与 float 矩阵乘这三条路（`mmq` / `mmvq` / `mmvf`）：它们的判据函数、阈值常量、tile 加载、量化点积、以及 batch 大小如何一路决定到**模板实例**的选取。`mmf`（Tensor Core 大矩阵路径）与 cuBLAS 兜底只作指路。

---

## 一、mmvf.cuh：8 行头文件里的两个事实

`mmvf.cuh` 只有 14 行，但它钉死了本课最容易被忽略的两个事实：**float 向量内核的 batch 上限是 8**，以及这个上限与 `MMVQ_MAX_BATCH_SIZE` 相等（`ggml-cuda.cu:1924` 处有 `static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);` 兜底）。

注意：这里的宏只给出**上限**；真正的判据在 `ggml_cuda_should_use_mmvf()` 里，还要先过三条内存对齐前提。

<!-- src: ggml/src/ggml-cuda/mmvf.cuh -->
```c
#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);
```

## 二、mma.cuh：把 PTX 的 mma 包成 tile

`mma.cuh` 是 Tensor Core 的抽象层：它把 PTX 的 `mma.sync` 指令包装成 `tile<I, J, T>` 类型，并规定 A 行主、B 列主、C 列主。mmq 的 tile 点积（`mmq-vec-dot.cuh`）就是直接拿 `tile<16,8,int>` 拼出来的。

文件头的注释把约定写死：`J` 的量纲是**物理 32 位元素**，不是逻辑元素。

<!-- src: ggml/src/ggml-cuda/mma.cuh -->
```c
#pragma once
// This file contains primitives that expose the tensor core PTX instructions for CUDA code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.
// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the PTX documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.
// The API in this file also assumes that the pointers for load_generic are aligned to 16 bytes, unaligned pointers are considered undefined behavior.
```

## 三、mma.cuh：load_generic 与 tile 的 (i, j) 映射

tile 的读写只有两个入口：`load_generic`（任意数据）与 `load_ldmatrix`（shared memory、16 字节对齐）。它们都靠 `t.get_i(l)` / `t.get_j(l)` 把"第 l 个物理元素"映射回 (i, j)。

<!-- src: ggml/src/ggml-cuda/mma.cuh -->
```c
    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
    }
```

## 四、mmq.cuh：tile 常量与 SRAM 布局约定

这一课的"prefetch/tile 加载"全部建立在这几行常量上。注释解释了两件事：**tile 尺寸与 WARP_SIZE 解耦**（为了兼容不同 warp 大小），以及**最后一维要 padding 以避免 shared memory bank conflict**。

<!-- src: ggml/src/ggml-cuda/mmq.cuh -->
```c
    int dm;
    int sc;
};

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.
#define MMQ_TILE_NE_K 32

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
#define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
#define MMQ_TILE_Y_FP4_K MMQ_TILE_Y_K
```

## 五、mmvq.cu：向量内核的骨架

`mul_mat_vec_q` 是一个以 `ncols_dst` 为模板参数的 kernel。`ncols_dst` 就是一次算几个 token，它同时决定 `nwarps` 与 `rows_per_cuda_block`（见 `calc_nwarps` / `calc_rows_per_block`），于是"batch 多大"在**编译期**就被固化进寄存器占用和 block 形状里。

<!-- src: ggml/src/ggml-cuda/mmvq.cu -->
```c
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
```

## 六、mmq.cuh：J 的 16 个模板实例

`mul_mat_q_switch_J` 算出 `J_best` 之后，用一个 switch 把它落到具体的模板实例上。16 个 `case` 就是 16 个独立的 kernel —— 这正是 L6-04 要讲的 `template-instances/` 机制在同一处的体现：**编译期穷举，运行时选择**。

<!-- src: ggml/src/ggml-cuda/mmq.cuh -->
```c
    switch (J_best) {
        case   8:
            launch_mul_mat_q<type,   8, fallback>(ctx, args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16, fallback>(ctx, args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24, fallback>(ctx, args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32, fallback>(ctx, args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40, fallback>(ctx, args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48, fallback>(ctx, args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56, fallback>(ctx, args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64, fallback>(ctx, args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72, fallback>(ctx, args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80, fallback>(ctx, args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88, fallback>(ctx, args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96, fallback>(ctx, args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104, fallback>(ctx, args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112, fallback>(ctx, args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120, fallback>(ctx, args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128, fallback>(ctx, args, stream);
            break;
        default:
            fprintf(stderr, "J_best=%d\n", J_best);
            GGML_ABORT("fatal error");
            break;
    }
```

---

## 说明

- 覆盖率声明含 11 个文件：计划清单里 L6-02 的 10 个，外加 `ggml/src/ggml-cuda/ggml-cuda.cu` —— 本课验收点（mmq / mmvq 在什么形状下被选中）的分派点 `ggml_cuda_mul_mat()` 就在该文件，必须逐字引用。该文件 L6-01 也已声明，两课共同覆盖同一文件是允许的（覆盖是并集）。
- 场景中的派生数字（如 `threads_per_row = 256/(4*2) = 32`、`MMQ_TILE_Y_K = 36`）由 `ggml/src/ggml-common.h` 的 `QK4_0 = 32` / `QR4_0 = 2` / `QR8_1 = 1` 推出，本课不引用该文件，故不计入本课覆盖率。
- `mmf`（Tensor Core 大矩阵路径）与 cuBLAS 兜底只在本课的分派表里出现名字与行号，其实现分别属于 L6-04 与 cuBLAS，本课不展开。
