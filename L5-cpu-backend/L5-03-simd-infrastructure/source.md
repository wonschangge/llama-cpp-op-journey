<!-- llama-coverage
ggml/src/ggml-cpu/vec.h
ggml/src/ggml-cpu/simd-mappings.h
ggml/src/ggml-cpu/vec.cpp
ggml/src/ggml-cpu/simd-gemm.h
-->

# L5-03 · ★ 向量化基础设施：一份宏，十一个架构分支 — 源文件

**一句话**：`ggml_vec_dot_*` 家族不可能为每个 CPU 架构写一遍。llama.cpp 的做法是**先定义一组宏，再用这组宏写运算** —— 同一个 `.cpp` 源码，在 x86 上被展开成 AVX 的`_mm256_*`，在 ARM 上被展开成 NEON 的 `v*_q*`，在 RISC-V 上被展开成 RVV 内建函数。这一课只讲让这件事成立的四个文件。

四个文件的分工：`simd-mappings.h` 定义宏（十一个架构分支）、`vec.h` 声明家族与调优常数、`vec.cpp` 用宏写 ISA 无关实现、`simd-gemm.h` 用宏写出 GEMM 微内核。

读这一课之前请先确认一件事：内核看到的永远是 `ne[]` / `nb[]`（L1-01）与量化块布局（L1-04）；这一课回答的是"拿到这些之后，一次算几个元素、用哪个寄存器"。

---

## 一、四个文件怎么分工

`vec.h` 是本课的入口：它只声明家族、给出调优常数，并把**所有架构差异**托付给 `simd-mappings.h`。注意它自己的 include 列表里没有任何 `immintrin.h` / `arm_neon.h` —— 那些头由 `simd-mappings.h` 按架构吸进来。

<!-- src: ggml/src/ggml-cpu/vec.h -->
```c
// Vectorized functions for fundamental operations

#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
#include "ggml.h"
#include "ggml-cpu.h"
```

## 二、★ 抽象层宣言：一套宏，十一个架构分支

这是本课最重要的一段注释。它把整个设计压缩成三条：

```text
1. 定义一组 C 宏，按当前架构映射到具体 intrinsic
2. 下面的基本运算只用这些宏实现
3. 支持新架构 = 定义对应的 SIMD 宏（不动算法）
```

宏族一共九个名字（`GGML_F32_VEC` 与 `_ZERO` / `_SET1` / `_LOAD` / `_STORE` / `_FMA` / `_ADD` / `_MUL` / `_REDUCE`），每个架构分支都必须定义齐。`GGML_F32_EPR` 是"一个寄存器装几个元素"，`GGML_F32_STEP` 是"一步处理几个元素"。

<!-- src: ggml/src/ggml-cpu/simd-mappings.h -->
```c
// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// GGML_F32_STEP / GGML_F16_STEP
//   number of elements to process in a single step
//
// GGML_F32_EPR / GGML_F16_EPR
//   number of elements to fit in a single register
```

## 三、★ 三种 ISA 的展开

同一组宏名，在三个分支里被定义成三套不同的东西。判据只有一个：编译器预先定义了哪个宏。AVX-512 是 16 个 float / 寄存器、一步 64 个；AVX/AVX2 是 8 / 32；NEON 是 4 / 16。三条分支的 `STEP / EPR` 都等于 4 —— **一步固定开 4 个向量寄存器**，变的只是每个寄存器装多少。（这三段的真实行号见代码区的 `//>>` 分隔行。）

<!-- src: ggml/src/ggml-cpu/simd-mappings.h -->
```c
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#define GGML_SIMD

// F32 NEON

#define GGML_F32_STEP 16
#define GGML_F32_EPR  4

#define GGML_F32x4              float32x4_t
#define GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
//>> ---- ggml/src/ggml-cpu/simd-mappings.h:581-590 ----
#elif defined(__AVX__)

#define GGML_SIMD

// F32 AVX

#define GGML_F32_STEP 32
#define GGML_F32_EPR  8

#define GGML_F32x8         __m256
//>> ---- ggml/src/ggml-cpu/simd-mappings.h:446-456 ----
#elif defined(__AVX512F__)

#define GGML_SIMD

// F32 AVX512

#define GGML_F32_STEP 64
#define GGML_F32_EPR  16

#define GGML_F32x16         __m512
#define GGML_F32x16_ZERO    _mm512_setzero_ps()
```

## 四、家族样板：ggml_vec_dot_f32

只有 34 行，却是所有向量化点积的模板：先切掉尾巴得到 `np`，再让主循环按 `GGML_F32_STEP` 前进、内层开 `GGML_F32_ARR` 个寄存器做 FMA，最后归约并补齐尾巴。文件末尾的 `#else // scalar` 是完全没有 SIMD 时的完整实现 —— 也是正确性基准。

<!-- src: ggml/src/ggml-cpu/vec.cpp -->
```cpp
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #endif
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}
```

## 五、家族与调优常数

三个浮点点积的签名完全一致，只有 x / y 的元素类型不同：`(int n, float * s, size_t bs, const T * x, size_t bx, const T * y, size_t by, int nrc)`。`nrc` 是"一次算几行"；`ggml_vec_dot_f16_unroll` 把它固定成 `GGML_VEC_DOT_UNROLL`。累加类型 `ggml_float` 是 `double`。

<!-- src: ggml/src/ggml-cpu/vec.h -->
```c
// floating point type used to accumulate sums
typedef double ggml_float;
//>> ---- ggml/src/ggml-cpu/vec.h:46-51 ----
#define GGML_GELU_FP16
#define GGML_GELU_QUICK_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2
#define GGML_VEC_MAD_UNROLL  32
//>> ---- ggml/src/ggml-cpu/vec.h:71-73 ----
void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);
//>> ---- ggml/src/ggml-cpu/vec.h:140-151 ----
// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float * GGML_RESTRICT s, void * GGML_RESTRICT xv, ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(GGML_SIMD)
```

## 六、另一种写法：ISA 特化分支

`ggml_vec_dot_bf16` 不用宏，而是按 `#if` 给每个架构写死 intrinsic：AVX-512 BF16 用一条硬件`_mm512_dpbf16_ps`，AVX-512/AVX2 用移位把 bf16 当 float，PowerPC 与 s390 回到宏族。注意链尾没有 `#else`：最后的循环是所有分支共用的收尾，也是"一条都不命中"时的全部实现。

<!-- src: ggml/src/ggml-cpu/vec.cpp -->
```cpp
#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
//>> ---- ggml/src/ggml-cpu/vec.cpp:239-261 ----
#elif defined(__POWER9_VECTOR__) || defined(__VXE__) || defined(__VXE2__)
    const int np = (n & ~(GGML_BF16_STEP - 1));
    if (np > 0) {
        GGML_F32_VEC sum[4] = {GGML_F32_VEC_ZERO};
        for (; i < np; i += GGML_BF16_STEP) {
            GGML_BF16_VEC vx0 = GGML_BF16_VEC_LOAD(x + i);
            GGML_BF16_VEC vx1 = GGML_BF16_VEC_LOAD(x + i + 8);
            GGML_BF16_VEC vy0 = GGML_BF16_VEC_LOAD(y + i);
            GGML_BF16_VEC vy1 = GGML_BF16_VEC_LOAD(y + i + 8);
            GGML_BF16_FMA_LO(sum[0], vx0, vy0);
            GGML_BF16_FMA_HI(sum[1], vx0, vy0);
            GGML_BF16_FMA_LO(sum[2], vx1, vy1);
            GGML_BF16_FMA_HI(sum[3], vx1, vy1);
        }
        GGML_F32x4_REDUCE_4(sumf, sum[0], sum[1], sum[2], sum[3]);
    }
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
```

## 七、GEMM 微内核：分块常数按 ISA 变

`simd-gemm.h` 的微内核里没有一个 intrinsic：加载累加器、遍历 K、FMA、存回，全部只用 `GGML_F32_VEC_*`。变得只有分块常数 —— AVX-512 与 NEON 是 4 x 4，AVX2 / AVX 是 6 x 2，其余 2 x 2；源码注释把寄存器预算写成了算式。

<!-- src: ggml/src/ggml-cpu/simd-gemm.h -->
```cpp
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON__)
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        for (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}
```

## 八、同一个 simd_gemm，三份定义

整个头文件被 `#if` 切成三段，每段各定义一次同名函数：宏模板版、RISC-V RVV 版、标量版。RVV 是唯一使用 sizeless 类型的分支（`vfloat32m4_t`，宽度运行时由 `__riscv_vlenb()` 给出）；标量版同时兜住"SVE"与"无 SIMD"两种情况。

<!-- src: ggml/src/ggml-cpu/simd-gemm.h -->
```cpp
#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <int RM>
static inline void rvv_simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
//>> ---- ggml/src/ggml-cpu/simd-gemm.h:176-186 ----
static constexpr int GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    const int KN = (int)__riscv_vlenb();
    int64_t ii = 0;
//>> ---- ggml/src/ggml-cpu/simd-gemm.h:207-226 ----
#else // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
```

## 九、★ q4_0 x q8_0：AVX2 与 NEON 的差异

Q4_0 的每个块带一个 fp16 的 scale（L1-04），所以任何 ISA 上的 `ggml_vec_dot_q4_0_q8_0` 第一步都是"把 fp16 转成 fp32"。这一段引用展示的就是这一步的两套实现：ARM 走 `__fp16` 强转，x86 走 `__F16C__` 的硬件转换指令。**上层只写 `GGML_CPU_FP16_TO_FP32(x)`** —— 这正是"一份源码、多种 ISA"的意思。

<!-- src: ggml/src/ggml-cpu/simd-mappings.h -->
```c
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) neon_compute_fp16_to_fp32(x)
    #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) neon_compute_fp32_to_fp16(x)

    #define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)

    static inline float neon_compute_fp16_to_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        memcpy(&tmp, &h, sizeof(ggml_fp16_t));
        return (float)tmp;
    }

    static inline ggml_fp16_t neon_compute_fp32_to_fp16(float f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        memcpy(&res, &tmp, sizeof(ggml_fp16_t));
        return res;
    }
#elif defined(__F16C__)
    #ifdef _MSC_VER
        #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
    #else
        #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
        #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
    #endif
```

---

## 说明

- 本课引用的 4 个文件 —— `ggml/src/ggml-cpu/` 下的 `vec.h`、`vec.cpp`、`simd-mappings.h`、`simd-gemm.h` —— 全部计入覆盖率。
- 为保证"每一条断言都能在引用的代码里看到"，本课**不引用** `arch/x86/quants.c`、`arch/arm/quants.c`、`quants.c`、`quants.h`、`arch-fallback.h`、`CMakeLists.txt`、`ggml-cpu.c`、`ops.cpp`、`ggml-cpu-impl.h` 等文件。正文里对这些文件只有"指路"（附真实行号），**不计入本课覆盖率**；它们分别属于 L5-01 / L5-02 / L5-04 / L5-05 的覆盖范围。
