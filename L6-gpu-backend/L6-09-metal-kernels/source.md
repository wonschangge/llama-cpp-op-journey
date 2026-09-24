<!-- llama-coverage
ggml/src/ggml-metal/kernels/common.h
ggml/src/ggml-metal/kernels/mul_mm.metal
ggml/src/ggml-metal/kernels/mul_mv.metal
ggml/src/ggml-metal/kernels/fa.metal
ggml/src/ggml-metal/kernels/dequantize.h
ggml/src/ggml-metal/kernels/rope.metal
ggml/src/ggml-metal/kernels/norm.metal
ggml/src/ggml-metal/kernels/softmax.metal
ggml/src/ggml-metal/kernels/quantize.h
ggml/src/ggml-metal/kernels/quantize.metal
ggml/src/ggml-metal/kernels/unary.metal
ggml/src/ggml-metal/kernels/binbcast.metal
ggml/src/ggml-metal/kernels/misc.metal
ggml/src/ggml-metal/kernels/conv.metal
ggml/src/ggml-metal/kernels/pool.metal
ggml/src/ggml-metal/kernels/upscale.metal
ggml/src/ggml-metal/kernels/reduce.metal
ggml/src/ggml-metal/kernels/argsort.metal
ggml/src/ggml-metal/kernels/tri.metal
ggml/src/ggml-metal/kernels/solve_tri.metal
ggml/src/ggml-metal/kernels/ssm.metal
ggml/src/ggml-metal/kernels/wkv.metal
ggml/src/ggml-metal/kernels/gated_delta_net.metal
-->

# L6-09 · Metal 后端：Metal 内核 — 源文件

**一句话**：Metal 后端的 `kernels/` 目录里是 23 个 `.metal` / `.h` 文件，共 13425 行；它们是这条流水线的最后一站 —— 主机端（L6-08）选好 pipeline，这里的内核才真正在 GPU 上算。

这一课的主线是**一个问题**：Metal 怎么用同一份源码覆盖二十多种量化类型？答案不是"每种类型写一个内核文件"，而是 **C++ 模板 + host 侧显式实例化**。这正是它与 CUDA 的 `template-instances/`（L6-04）不同的地方。

数字来自命令：`python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L6-09"{print $2}'` 给出 23 个文件；`wc -l` 给出 13425 行；`grep -c host_name` 给出 1017 处实例化。

---

## 一、kernels 目录全景：23 个文件、8 个族

本课覆盖的清单由计划矩阵给出，不是凭印象划的：

```text
python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L6-09"{print $2}'
```

共 **23 个文件**（`wc -l` 合计 13425 行），按"算什么"分成 8 个族：

| 族 | 文件数 | 行数 | host_name 实例化数 | 文件 |
|---|---|---|---|---|
| 矩阵乘 | 2 | 4365 | 245 | `mul_mm.metal` `mul_mv.metal` |
| 注意力与序列混合 | 4 | 3395 | 567 | `fa.metal` `ssm.metal` `wkv.metal` `gated_delta_net.metal` |
| 逐元素与形状 | 6 | 2337 | 61 | `unary.metal` `binbcast.metal` `misc.metal` `conv.metal` `pool.metal` `upscale.metal` |
| 量化与反量化 | 3 | 1477 | 98 | `dequantize.h` `quantize.h` `quantize.metal` |
| 归约 / 排序 / 三角 | 4 | 851 | 20 | `reduce.metal` `argsort.metal` `tri.metal` `solve_tri.metal` |
| 归一化与 softmax | 2 | 541 | 18 | `norm.metal` `softmax.metal` |
| 位置编码 | 1 | 333 | 8 | `rope.metal` |
| 公共头 | 1 | 126 | 0 | `common.h` |
| **合计** | **23** | **13425** | **1017** | |

三个计数都能用命令复现：`wc -l`（行数）、`grep -c host_name`（实例化数）、`python3 tools/plan_matrix.py --files`（清单）。

## 二、common.h：所有内核共享的前言

每个 `.metal` 的第一行都是 `#include "common.h"`。它做四件事：把**主机端的参数头**`ggml-metal-impl.h` 拉进来（内核与主机共享同一份 kargs 结构体声明）、打开 `metal_stdlib`、定义几个小宏（`MAX` / `MIN` / `SWAP` / `PAD2` / `FOR_UNROLL`）、以及钉死 `N_SIMDWIDTH = 32`。

第 23 行的注释是后面所有归约代码的前提：**一个 SIMD group 假设是 32 个线程**。`simd_sum` / `simd_max` / `simd_shuffle_down` 全都建立在这个宽度上。

<!-- src: ggml/src/ggml-metal/kernels/common.h -->
```cpp
#pragma once

#include "ggml-metal-impl.h"

#include <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32
```

## 三、mul_mm 的 tensor 路径：tA / tB / matmul2d

同一个文件的上半部分（`#ifdef GGML_METAL_HAS_TENSOR`）走的是另一条实现：用 `metal_tensor` 与 `mpp::tensor_ops::matmul2d` 把矩阵乘交给硬件的张量单元。

注意两处细节：**A 用线程组内存包成 tensor**（`tA`，因为 A 要先反量化），而 **B 直接从设备内存包**（`tB`，连 stride 都是按元素算的）；K 维被声明成 `dynamic_extent`，源码注释解释了原因 —— 静态 K tile 会在 `K % N_MM_NK_TOTAL != 0` 时越界读 src1。

<!-- src: ggml/src/ggml-metal/kernels/mul_mm.metal -->
```cpp
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    // note: K is dynamic_extent (clamped to the valid range in PHASE 2), since a static
    //       N_MM_NK_TOTAL K tile would read src1 out of bounds when K % N_MM_NK_TOTAL != 0
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;
```

## 四、mul_mv：per-row 的内层循环与 ext 的分发器

`mul_mv.metal` 里有两套矩阵向量内核，这里各引一段。

**第一段是 per-row 的内层循环**（第 265 到 291 行）：`yl[16]` 是 src1 那一行的寄存器缓存，外层按 16 个元素推进，内层 `FOR_UNROLL (short row = 0; row < NR0; row++)` 对 NR0 行 src0 各做一次 `block_q_n_dot_y` —— **一份 src1 数据被 NR0 行复用**。注意 `yl[i+1] = yb[i+1]/256.f` 这类预缩放：它把 q4_0 的高低 4 位共用同一个 scale，省掉了一次乘法。

**第二段是 ext 的分发器**（第 811 到 838 行）：真正的实现在 `kernel_mul_mv_ext_q4_f32_impl` 里，分发器只负责把"每块的元素数"换算成"float4 块数"（`epb/4`）或 "float4x4 块数"（`epb/16`），源码注释说明它"needed for compile-time nxpsg"。

<!-- src: ggml/src/ggml-metal/kernels/mul_mv.metal -->
```cpp
    float yl[16]; // src1 vector cache

    //device const float * yb = y + ix*QK4_0 + il;
    device const float * yb = y + ib0*QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    //for (int ib = ib0; ib < nb; ib += NSG*NQ) {
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
        //yb += NSG*NQ*QK4_0;
    }
//>> ---- ggml/src/ggml-metal/kernels/mul_mv.metal:811-838 ----
// dispatchers needed for compile-time nxpsg
// epb - elements per quantization block
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

template<short r1ptg, typename q_t, short epb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
kernel void kernel_mul_mv_ext_q4x4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4x4_f32_impl<r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;
typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>)    mul_mv_ext_q4x4_f32_t;
```

## 五、dequantize.h：同一个函数槽，量化与非量化都填

模板参数 `dequantize_func` 对非量化类型也要有个实现，否则模板没法实例化。`dequantize_f32` / `dequantize_f16` 就是这样的"占位"实现 —— 源码注释写得很清楚：**this is not dequantizing - we are simply fitting the template**。

<!-- src: ggml/src/ggml-metal/kernels/dequantize.h -->
```cpp

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}
```

## 六、dequantize.h：dequantize_q4_0 逐块展开

真正干活的反量化函数长这样：从一个 `block_q4_0` 里取 4-bit 量化值，乘上 scale 再减掉零点偏移（`md = -8 * xb->d`，对应 4-bit 的无符号偏移 8）。`il` 决定取低 4 位还是高 4 位，返回一个 4x4 的寄存器块。

这正是 L1-04 讲过的块布局在 Metal 侧的样子：**同一个 `block_q4_0`，同一段字节**。

<!-- src: ggml/src/ggml-metal/kernels/dequantize.h -->
```cpp
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}
```

## 七、fa.metal：KV 量化先反量化成 F16

flash attention 的主内核要求 K / V 是 F16。量化 KV 缓存的走法是：先用一个**独立的、每线程一个块**的内核把 K / V 反量化成连续的 F16，再跑 F16 主内核。源码注释写明它"dispatched separately for K and V"。

这里能直接看到 Metal 支持哪几种 KV 量化：`q4_0` / `q4_1` / `q5_0` / `q5_1` / `q8_0`，五种都只是同一个模板换参数。

<!-- src: ggml/src/ggml-metal/kernels/fa.metal -->
```cpp
// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
```

## 八、rope.metal：YaRN 的数学与 rope_norm 内核

rope 的内核先算 YaRN 的修正维度（`rope_yarn_corr_dims`），再按位置取 theta、算出 cos / sin，最后做二维旋转。`FC_rope_is_back` 是函数常量 —— 反向时把 sin 取负。

注意内核的循环步长是 `2*tptg.x`：**一个线程负责一对元素**（`i0` 与 `i0+1`），这正是 rope 的旋转是成对进行的直接体现。

<!-- src: ggml/src/ggml-metal/kernels/rope.metal -->
```cpp
template<typename T>
kernel void kernel_rope_norm(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
            const int iw = i0 - args.n_offs; // relative idx
            const int ic = iw/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);
```

## 九、norm.metal：两级 SIMD-group 归约

归一化要算一整行的均值与方差，而一行有 `ne00` 个元素、线程组只有有限线程。写法是标准的两级归约：**先 `simd_sum` 在 SIMD group 内归约**，**再用 `threadgroup float *` 把各组的和汇总**，然后再 `simd_sum` 一次。

模板参数 `F` 是融合开关（1 = 只有 norm，2 = norm+mul，3 = norm+mul+add）—— 回顾 L6-08：主机端的图融合最终就是选一个 `F` 值。

<!-- src: ggml/src/ggml-metal/kernels/norm.metal -->
```cpp
template <typename T, short F>
kernel void kernel_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    T sumft(0.0f);

    float sumf = 0.0f;

    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumft += x[i00];
    }
    sumf = dot(sumft, T(1.0f));
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);
```

## 十、softmax.metal：同一个归约范式，两次

softmax 要在一行里先求 max 再求 sum，所以把两级归约写了两遍：`simd_max` + 线程组内存，然后 `simd_sum` + 线程组内存。

第 50 行的 `if (tptg.x > N_SIMDWIDTH)` 是这段代码的关键：**只有当线程组比一个 SIMD group 宽时才需要跨组汇总**，否则纯寄存器归约就够了。第 75 行的 `threadgroup_barrier(mem_flags::mem_none)` 还带着一条注释，说明它是为了修一个测试失败而加的。

<!-- src: ggml/src/ggml-metal/kernels/softmax.metal -->
```cpp
    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

```

## 十一、quantize.h：把 float 装进 block

反量化是"块 -> float"，量化是反过来的"float -> 块"。`quantize.h` 是后者的实现集合，每个函数拿一个 `device const float *` 和一个 `device block_q &`。

注意 `quantize_q1_0` 的写法：scale 取绝对值的平均，然后**只存符号位**；`quantize_q2_0` 取绝对值最大值当 scale。它们与 `dequantize.h` 里的对应函数是一对逆运算。

<!-- src: ggml/src/ggml-metal/kernels/quantize.h -->
```cpp
#include "common.h"

void quantize_q1_0(device const float * src, device block_q1_0 & dst) {
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    for (int j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK1_0; j++) {
        if (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_q2_0(device const float * src, device block_q2_0 & dst) {
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; j++) {
        float a = fabs(src[j]);
        if (a > amax) amax = a;
```

## 十二、quantize.metal：cpy 与量化/反量化

`quantize.metal` 里不只是量化，还有张量拷贝（`kernel_cpy_t_t`）。被引用的这一段是**反量化式拷贝**：从 `block_q` 读、写出 `T4x4`。

它的模板参数与 `mul_mm` 是同一组：`block_q` / `nl` / `dequantize_func`。这说明"换类型不换正文"的做法在整个 kernels 目录里是一致的约定，不是矩阵乘的特例。

<!-- src: ggml/src/ggml-metal/kernels/quantize.metal -->
```cpp
template<typename T4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_cpy_q_f32(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_q * src_data = (device const block_q *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T4x4    * dst_data = (device       T4x4    *)(dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        T4x4 temp;
        dequantize_func(src_data + i00/nl, i00%nl, temp);
        dst_data[i00] = temp;

        break;
    }
}

typedef decltype(kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>) cpy_q_f_t;

```

## 十三、unary.metal：一元算子靠函数常量分流

`unary` 覆盖 GELU / SILU / EXP / NEG 等一大票一元算子。它不为每个算子写一个内核，而是用函数常量 `FC_unary_op` 在一个内核里分流。

`FC_unary_cnt` 则区分"连续张量"与"有步长的张量"两条寻址路径 —— 连续时可以直接按线性下标算，不必走 `nb[]`。

<!-- src: ggml/src/ggml-metal/kernels/unary.metal -->
```cpp
template <typename T0, typename T, typename TC>
kernel void kernel_unary_impl(
        constant ggml_metal_kargs_unary & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP  FC_unary_op
#define FC_CNT FC_unary_cnt

    device const T0 * src0_ptr;
    device       T  * dst_ptr;

    int i0;

    if (FC_CNT) {
        i0 = tgpig.x;

        src0_ptr = (device const T0 *) (src0);
        dst_ptr  = (device       T  *) (dst);
    } else {
        const int i03 = tgpig.z;
        const int i02 = tgpig.y;
        const int k0  = tgpig.x/args.ne01;
        const int i01 = tgpig.x - k0*args.ne01;

        i0 = k0*ntg.x + tpitg.x;

        src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
        dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1 );
    }

    {
        //threadgroup_barrier(mem_flags::mem_none);
```

## 十四、binbcast.metal：四种二元运算 + 广播

加 / 减 / 乘 / 除四个算子由函数常量 `FC_bin_op` 分流；`FC_bin_rb`（行广播）与 `FC_bin_cb`（列广播）决定 src1 的下标怎么算。

这是"内核正文与算子名解耦"的又一个例子：**同一段代码，靠常量选分支**。

<!-- src: ggml/src/ggml-metal/kernels/binbcast.metal -->
```cpp
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP FC_bin_op
#define FC_F  FC_bin_f
#define FC_RB FC_bin_rb
#define FC_CB FC_bin_cb

    if (FC_RB) {
        // row broadcast
        const uint i0 = tgpig.y*args.ne00 + tgpig.x;
        const uint i1 = FC_CB ? tgpig.x%args.ne10 : tgpig.x;

        device const T0 * src0_row = (device const T0 *) (src0);
        device       T  * dst_row  = (device       T  *) (dst);

        if (FC_F == 1) {
            device const T1 * src1_row = (device const T1 *) (src1 + args.o1[0]);

            if (FC_OP == 0) {
                dst_row[i0] = src0_row[i0] + src1_row[i1];
            }

            if (FC_OP == 1) {
                dst_row[i0] = src0_row[i0] - src1_row[i1];
            }
```

## 十五、misc.metal：argmax 与 MoE 辅助

`misc.metal` 是杂项集合。被引用的 `kernel_argmax_f32` 一行一个线程组，组内先用 `simd_max` 求最大值，再用 `select` 把对应的下标也取出来 —— **归约值的同时归约下标**，这是 argmax 这类算子的常见写法。

文件里还有 top-k、MoE 相关的辅助内核。

<!-- src: ggml/src/ggml-metal/kernels/misc.metal -->
```cpp
kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
```

## 十六、conv.metal：im2col 与卷积

卷积在 GPU 上的常见做法是先 `im2col` 把输入展开成矩阵，再走矩阵乘。这里能看到线程组的三个维度分别对应输入通道、卷积核的行与列 —— `KH` / `KW` 直接取自 `ntg[1]` / `ntg[2]`，**核尺寸就是线程组的形状**。

<!-- src: ggml/src/ggml-metal/kernels/conv.metal -->
```cpp
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
//    const int64_t IC = tgpg[0];
    const int64_t OH = tgpg[1];
    const int64_t OW = tgpg[2];

    const int64_t KH = ntg[1];
    const int64_t KW = ntg[2];

          int64_t in  = tpitg[0];
    const int64_t ikh = tpitg[1];
    const int64_t ikw = tpitg[2];

    const int64_t iic = tgpig[0];
```

## 十七、pool.metal：每线程一个输出元素

池化是最"朴素"的一类内核：`gid` 直接对应一个输出元素，算出窗口边界后逐个比较取最大值。没有线程组内存，也没有 SIMD 归约。

注意边界处理：`bh` / `eh` 用 `MAX` / `MIN` 把窗口裁进输入范围，于是 padding 不需要真的填数据。

<!-- src: ggml/src/ggml-metal/kernels/pool.metal -->
```cpp
kernel void kernel_pool_2d_max_f32(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);

    float res = -INFINITY;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            res = MAX(res, i_ptr[i * args.IW + j]);
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = res;
}

```

## 十八、upscale.metal：最近邻与双线性

上采样有两个内核：最近邻直接按放大倍数整除下标，双线性用 `bilinear_tri` 之类的权重函数做插值。`FC_upscale_aa` 决定是否走抗锯齿那条路。

被引用的是最近邻版本：`i00 = i0/args.sf0` —— **一个整除就是整个上采样**。

<!-- src: ggml/src/ggml-metal/kernels/upscale.metal -->
```cpp
kernel void kernel_upscale_nearest_f32(
    constant ggml_metal_kargs_upscale & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3/args.sf3;
    const int64_t i02 = i2/args.sf2;
    const int64_t i01 = i1/args.sf1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int64_t i00 = i0/args.sf0;

        device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1  +  i0*args.nb0);

        dst_ptr[0] = src0_ptr[0];
    }
}

```

## 十九、reduce.metal：求和也是两级归约

`kernel_op_sum_f32` 把两级归约写成了最直白的形式：线程组内先 `simd_sum`，每个 SIMD group 写一个部分和进线程组内存，再由第 0 组读回来做第二次 `simd_sum`。

`nsg` 是从 `ntg.x` 现算出来的（`(ntg.x + 31) / 32`），源码注释标注了"TODO: become function constant"。

<!-- src: ggml/src/ggml-metal/kernels/reduce.metal -->
```cpp
kernel void kernel_op_sum_f32(
        constant ggml_metal_kargs_sum & args,
        device const float * src0,
        device       float * dst,
        threadgroup  float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    if (args.np == 0) {
        return;
    }

    // TODO: become function constant
    const uint nsg = (ntg.x + 31) / 32;

    float sumf = 0;

    for (uint64_t i0 = tpitg.x; i0 < args.np; i0 += ntg.x) {
        sumf += src0[i0];
    }

    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0;

    if (sgitg == 0) {
        float v = 0;

        if (tpitg.x < nsg) {
            v = shmem_f32[tpitg.x];
        }

        total = simd_sum(v);

```

## 二十、argsort.metal：双调排序

排序用的是双调排序（bitonic sort），源码注释说明它"following the CUDA kernels as reference"。排序键与下标一起在线程组内存里排，所以 `shmem_i32` 存的是下标。

模板参数是 `ggml_sort_order`（升序 / 降序）—— **排序方向也进模板**，这样比较那一步就没有运行期分支。

<!-- src: ggml/src/ggml-metal/kernels/argsort.metal -->
```cpp
typedef void (argsort_t)(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
```

## 二十一、tri.metal：三角掩码的四种类型

`tri` 把矩阵按三角区域清零。四种类型（下三角、含对角下三角、上三角、含对角上三角）被写成 `_ggml_vec_tri_cmp` 的四个模板特化 —— **模板特化在这里替代了运行期的 if**，与 mul_mm 用模板参数替代多份源码是同一个思路。

<!-- src: ggml/src/ggml-metal/kernels/tri.metal -->
```cpp
template<uint32_t ttype>
bool _ggml_vec_tri_cmp(const int i, const int r);

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_LOWER */ 3>(const int i, const int r) {
    return i < r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_LOWER_DIAG */ 2>(const int i, const int r) {
    return i <= r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_UPPER */ 1>(const int i, const int r) {
    return i > r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_UPPER_DIAG */ 0>(const int i, const int r) {
    return i >= r;
}

template<typename T, int ttype>
kernel void kernel_tri(
        constant ggml_metal_kargs_tri & args,
        device const char * src0,
        device const char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
```

## 二十二、solve_tri.metal：三角方程求解

三角求解是逐行前代/回代，天然串行，所以这里的并行度放在"多列同时解"上：线程组的第 0 维取 `i01 = tgpig.x*NSG + sgitg`，即一个 SIMD group 负责一列。`N` / `K` / `NSG` 都是函数常量，`NP = PAD2(N, NW)` 把行长补齐到 32 的倍数。

<!-- src: ggml/src/ggml-metal/kernels/solve_tri.metal -->
```cpp
constant short FC_solve_tri_nsg [[function_constant(FC_SOLVE_TRI + 0)]];
constant short FC_solve_tri_n   [[function_constant(FC_SOLVE_TRI + 1)]];
constant short FC_solve_tri_k   [[function_constant(FC_SOLVE_TRI + 2)]];

kernel void kernel_solve_tri_f32(
        constant ggml_metal_kargs_solve_tri & args,
        device   const char * src0,
        device   const char * src1,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        ushort3 tgpig[[threadgroup_position_in_grid]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const short NSG = FC_solve_tri_nsg;
    const short N   = FC_solve_tri_n;
    const short K   = FC_solve_tri_k;
    const short NP  = PAD2(N, NW);

    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x*NSG + sgitg;

    threadgroup float * sh0 = (threadgroup float *) shmem;

    device const float * src0_ptr = (device const float *)(src0 + i02 * args.nb02 + i03 * args.nb03) + sgitg*N;
    device const float * src1_ptr = (device const float *)(src1 + i02 * args.nb12 + i03 * args.nb13) + i01;
    device       float * dst_ptr  = (device       float *)(dst  + i02 * args.nb2  + i03 * args.nb3)  + i01;

    for (short rr = 0; rr < N; rr += NSG) {
```

## 二十三、ssm.metal：Mamba 的卷积与扫描

`ssm` 是状态空间模型（Mamba）的算子。源码注释直接指路：**ref: ggml.c:ggml_compute_forward_ssm_conv_f32** —— GPU 内核与 CPU 参考实现是同一套公式的两种写法，这是 llama.cpp 后端的普遍做法。

<!-- src: ggml/src/ggml-metal/kernels/ssm.metal -->
```cpp
constant bool FC_ssm_conv_silu [[function_constant(FC_SSM_CONV + 1)]];
constant int  FC_ssm_conv_nc    [[function_constant(FC_SSM_CONV + 2)]];

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
kernel void kernel_ssm_conv_f32_f32(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = FC_ssm_conv_nc;
  //const int64_t ncs = args.ne00;
  //const int64_t nr  = args.ne01;
  //const int64_t n_t = args.ne1;
  //const int64_t n_s = args.ne2;

    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);
```

## 二十四、wkv.metal：RWKV 的 WKV

RWKV 的 WKV 算子是逐 head 递推的：`tgpig.x` 同时编码 batch 与 head（`batch_id = tgpig.x / H`，`head_id = tgpig.x % H`），线程组内再按 `tid` 分特征维。

注意 `head_size` 被硬编码成 64，源码注释写着"TODO: support head_size = 128"。**凡注释声称的约束，都要按注释读**——这里说明该内核目前只支持 64 维 head。

<!-- src: ggml/src/ggml-metal/kernels/wkv.metal -->
```cpp
kernel void kernel_rwkv_wkv6_f32(
    device const float * k,
    device const float * v,
    device const float * r,
    device const float * tf,
    device const float * td,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]])  {

    const uint head_size = 64; // TODO: support head_size = 128
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

```

## 二十五、gated_delta_net.metal：函数常量做形状参数

Gated Delta Net 的内核把三个形状参数做成函数常量：`ne20`（V 的维度）、`ne30`（G 的维度）与 `K`，再用 `#define` 起短名字在正文里用。

模板参数只有 `NSG`（SIMD group 数）—— 这是本目录里**形状走函数常量、并行度走模板**的典型分工：形状可以在 pipeline 创建时定，并行度要在编译期定。

<!-- src: ggml/src/ggml-metal/kernels/gated_delta_net.metal -->
```cpp
constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];

#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
```

---

## 说明

- 本课引用 `ggml/src/ggml-metal/kernels/` 下的 23 个文件，全部计入覆盖率；清单来自 `python3 tools/plan_matrix.py --files`。
- 第 6 幕提到的 `r1ptg` 选择逻辑（按 `ne11` 在 2 到 8 之间取 2/3/4/5）在 `ggml/src/ggml-metal/ggml-metal-ops.cpp` 里，该文件由 L6-08 覆盖；本课不引用其源码，故不计入本课覆盖率。
- `ggml/src/ggml-metal/ggml-metal-impl.h`（`N_MM_*` / `N_R0_*` / `N_SG_*` 常量）同样由 L6-08 覆盖，本课不引用其源码，不计入本课覆盖率。第 4 幕与第 9 幕出现的宏名来自 `common.h:3` 的 include。
- `ggml/src/ggml-common.h`（块结构体定义）由 L1-04 覆盖，本课只引用 `dequantize.h` 里打开它分支的那几行，不计入本课覆盖率。该头文件按 `GGML_COMMON_DECL_C` / `_CPP` / `_METAL` / `_CUDA` / `_HIP` / `_SYCL` 分支出各后端要的结构体定义 —— 第 8 幕的卡片讲的就是这件事。
