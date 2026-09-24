<!-- llama-coverage
ggml/src/ggml-cpu/arch-fallback.h
ggml/src/ggml-cpu/ggml-cpu-impl.h
ggml/src/ggml-cpu/common.h
ggml/src/ggml-cpu/arch/x86/quants.c
ggml/src/ggml-cpu/arch/x86/repack.cpp
ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
ggml/src/ggml-cpu/arch/arm/quants.c
ggml/src/ggml-cpu/arch/arm/repack.cpp
ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp
ggml/src/ggml-cpu/arch/riscv/quants.c
ggml/src/ggml-cpu/arch/riscv/repack.cpp
ggml/src/ggml-cpu/arch/riscv/cpu-feats.cpp
ggml/src/ggml-cpu/arch/s390/quants.c
ggml/src/ggml-cpu/arch/s390/repack.cpp
ggml/src/ggml-cpu/arch/s390/cpu-feats.cpp
ggml/src/ggml-cpu/arch/powerpc/quants.c
ggml/src/ggml-cpu/arch/powerpc/cpu-feats.cpp
ggml/src/ggml-cpu/arch/loongarch/quants.c
ggml/src/ggml-cpu/arch/wasm/quants.c
ggml/src/ggml-cpu/amx/amx.cpp
ggml/src/ggml-cpu/amx/amx.h
ggml/src/ggml-cpu/amx/common.h
ggml/src/ggml-cpu/amx/mmq.cpp
ggml/src/ggml-cpu/amx/mmq.h
ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
ggml/src/ggml-cpu/kleidiai/kernels.cpp
ggml/src/ggml-cpu/kleidiai/kernels.h
ggml/src/ggml-cpu/kleidiai/kleidiai.h
ggml/src/ggml-cpu/spacemit/ime.cpp
ggml/src/ggml-cpu/spacemit/ime.h
ggml/src/ggml-cpu/spacemit/ime_env.cpp
ggml/src/ggml-cpu/spacemit/ime_env.h
ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp
ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp
ggml/src/ggml-cpu/spacemit/ime_kernels.h
ggml/src/ggml-cpu/spacemit/repack.cpp
ggml/src/ggml-cpu/spacemit/repack.h
ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp
ggml/src/ggml-cpu/spacemit/rvv_kernels.h
ggml/src/ggml-cpu/spacemit/spine_barrier.h
ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp
ggml/src/ggml-cpu/spacemit/spine_mem_pool.h
ggml/src/ggml-cpu/spacemit/spine_tcm.h
ggml/src/ggml-cpu/llamafile/sgemm.cpp
ggml/src/ggml-cpu/llamafile/sgemm.h
ggml/src/ggml-cpu/hbm.cpp
ggml/src/ggml-cpu/hbm.h
ggml/src/ggml-cpu/iqp.cpp
ggml/src/ggml-cpu/iqp.h
-->

# L5-05 · ★ 多架构 SIMD 与厂商加速 — 源文件

**一句话**：同一个算子（比如 `GGML_OP_MUL_MAT` 配 Q4_0 权重）在 x86、ARM、RISC-V、s390、PowerPC、LoongArch、WASM 上跑的是**七个同名、不同文件的函数**；"选哪个 kernel"是一张**二维决策表** —— 编译期决定编进哪些 ISA 变体（`arch-fallback.h`），运行期按 CPU 特性选一个（`arch/*/cpu-feats.cpp` 的 score）。厂商加速库（AMX / KleidiAI / SpacemiT / llamafile）不是第三套内核，而是**在这张表上再插一层**：一个 buffer type + 一个 traits 钩子。

本课覆盖 `ggml/src/ggml-cpu/` 下的 **49 个源文件**（`plan_matrix.py` 分配给 L5-05 的全部文件）：`arch/` 子树 16 个、`amx/` 5 个、`kleidiai/` 4 个、`spacemit/` 15 个、`llamafile/` 2 个、`hbm.*` 2 个、`iqp.*` 2 个，加上 `arch-fallback.h`、`common.h`、`ggml-cpu-impl.h`。所有引用按行号从上游 v0.5.0 抽取，行号只对该版本有效。

---

## 一、总览：49 个文件怎么分工

本课的文件可以按**「它回答哪个问题」**分成四类，后面每一节都按这个顺序读：

| 类 | 文件数 | 回答的问题 |
|---|---|---|
| ISA 内核与探测 | 16 | 这个架构上，这个算子用哪条指令序列？ |
| 厂商/第三方加速 | 26 | 有没有更强的一整块硬件/库可以接管？ |
| 通用垫层 | 3 | 没有原生实现时，谁来顶？跨 ISA 的名字怎么统一？ |
| 专用旁路 | 4 | 不换指令、换内存或换 GEMM 组织，能不能更快？ |

`arch-fallback.h` 是理解前两类的钥匙：它把「编译期选择」写成了**差集**。

<!-- src: ggml/src/ggml-cpu/arch-fallback.h -->
```c
#pragma once

// Rename `_generic` functions if no native implementation is available.
// This effectively selects the generic implementation.

#if defined(GGML_CPU_GENERIC)
// quants.c
#define quantize_row_q8_0_generic quantize_row_q8_0
#define quantize_row_q8_1_generic quantize_row_q8_1
#define quantize_row_q8_K_generic quantize_row_q8_K
#define ggml_vec_dot_q4_0_q8_0_generic ggml_vec_dot_q4_0_q8_0
```

## 二、★ 编译期那一维：差集表

每个架构分支只列出**本架构没有的原生实现**。ARM 只有 8 条（全在 repack.cpp 段），x86 有 29 条（quants.c 段只缺 `q2_0` 一条），其余架构更多。统计口径：对该文件各分支的 `#define` 计数（本课实测）。

```text
GENERIC 68   wasm 55   s390x 50   loongarch64 48
powerpc 47   riscv 41  x86_64 29   aarch64 8
```

配合 `ggml-cpu/quants.c`（属 L5-04）里定义的 `*_generic` 版本，就能看出完整机制：**原生实现写正式名字，通用实现写 `_generic` 后缀，arch-fallback.h 决定这一份编译里谁叫正式名字。**

<!-- src: ggml/src/ggml-cpu/arch-fallback.h -->
```c
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
// quants.c
#define ggml_vec_dot_q2_0_q8_0_generic ggml_vec_dot_q2_0_q8_0
// repack.cpp
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_gemv_q1_0_4x4_q8_0_generic ggml_gemv_q1_0_4x4_q8_0
#define ggml_gemv_q1_0_4x8_q8_0_generic ggml_gemv_q1_0_4x8_q8_0
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q1_0_4x4_q8_0_generic ggml_gemm_q1_0_4x4_q8_0
#define ggml_gemm_q1_0_4x8_q8_0_generic ggml_gemm_q1_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
```

## 三、运行期那一维：变体打分

同一架构可以编出多份变体（不同的 `-march`），每份导出一个 `ggml_backend_score()`。分数是**位图 + 1**：基线 1 分，每满足一项或上一位。每个 `#ifdef` 都配一次 `if (!is.X()) return 0;` —— 缺一项不是降级，而是这份变体直接出局。

读分数的是 `ggml/src/ggml-backend-reg.cpp`（属 L3-02，本课不引用其文本）：遍历候选动态库、调 `ggml_backend_score`、取最高分并 `dlopen`。所以「选 kernel」在运行期至少发生两次：先选变体，再选内核。

<!-- src: ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp -->
```c
static int ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    int score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    if (!is.FMA()) { return 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    if (!is.F16C()) { return 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    if (!is.SSE42()) { return 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    if (!is.BMI2()) { return 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    if (!is.AVX()) { return 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    if (!is.AVX2()) { return 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    if (!is.AVX_VNNI()) { return 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    if (!is.AVX512F()) { return 0; }
    if (!is.AVX512CD()) { return 0; }
    if (!is.AVX512VL()) { return 0; }
    if (!is.AVX512DQ()) { return 0; }
    if (!is.AVX512BW()) { return 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    if (!is.AVX512_VBMI()) { return 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    if (!is.AVX512_BF16()) { return 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    if (!is.AVX512_VNNI()) { return 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    if (!is.AMX_INT8()) { return 0; }
    score += 1<<11;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)
```

## 四、厂商层之一：KleidiAI 的 required_cpu 表

KleidiAI（Arm）不往 traits 里堆 if，而是把「我能跑什么」写成**有序表**：每项有 `required_cpu` 位掩码 + lhs/rhs/op 三个类型，**第一条命中就返回**，全表不命中则返回 `nullptr`（这一层不接管，退回默认路径）。

<!-- src: ggml/src/ggml-cpu/kleidiai/kernels.cpp -->
```c
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, const ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = nullptr;

    if (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != nullptr && tensor->src[1] != nullptr) {
        auto try_table = [&](auto & table) {
            for (size_t i = 0; i < NELEMS(table) - 1; ++i) {
                if ((cpu_features & table[i].required_cpu) == table[i].required_cpu &&
                    table[i].lhs_type == tensor->src[1]->type &&
                    table[i].rhs_type == tensor->src[0]->type &&
                    table[i].op_type  == tensor->type) {
                    kernel = &table[i];
                    return true;
                }
            }
            return false;
        };

        if (tensor->src[0]->type == GGML_TYPE_Q8_0) {
            try_table(gemm_gemv_kernels_q8);
        } else if (tensor->src[0]->type == GGML_TYPE_F32) {
            try_table(ggml_kleidiai_kernels_f32);
        } else {
            try_table(gemm_gemv_kernels);
        }
    }

    return kernel;
}
```

## 五、KleidiAI：特性掩码从哪来

掩码由 aarch64 运行期探测拼出来：DOTPROD / I8MM / FP16 / SVE 各占一位（SVE 还要求 `sve_cnt` 正好等于 QK8_0）。这一段还展示了三个调试用的环境变量覆盖 （`GGML_KLEIDIAI_SME` / `GGML_TOTAL_THREADS` / `GGML_KLEIDIAI_CHUNK_MULTIPLIER`）：**厂商路径一定要能关**。

SME 家族的处理在这段之后（338-355 行）：先数出可用的流式模式计算单元（SMCU），数不到就保守地把 SME 线程上限设为 1 —— 因为 SME 的流式模式是有限硬件资源。

<!-- src: ggml/src/ggml-cpu/kleidiai/kleidiai.cpp -->
```c++
static void init_kleidiai_context(void) {
    ggml_critical_section_start();
    static bool initialized = false;

    if (!initialized) {
        initialized = true;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        const char *env_sme         = getenv("GGML_KLEIDIAI_SME");
        const char *env_threads     = getenv("GGML_TOTAL_THREADS");
        const char *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        const auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);
```

## 六、厂商层之二：SpacemiT 的三级选择链

RISC-V 厂商 SpacemiT 把矩阵扩展（IME1/IME2）做成了自定义指令。`forward_mul_mat` 里的选择链是三级：**编译期宏 -> 运行期核型号 -> 权重块类型**，每级用 `!set_kernel_impl &&` 串联（先到先得），全不中就 `GGML_ABORT`。

注意第一级（IME2 段）里 `INTER_SIZE == 256` 还会再分一次 `_hp`（高精度）路径：同一类型在不同块大小下也能走不同内核。

<!-- src: ggml/src/ggml-cpu/spacemit/ime.cpp -->
```c++
#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl    = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = true;
            }
        }
#endif
```

## 七、同名不同物（上）：x86 的 ggml_vec_dot_q4_0_q8_0

函数名和签名在七个架构文件里**逐字相同**：`(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc)`。x86 这份在本文件内再按 `__AVX2__` / `__AVX__` / SSE 分派，并断言 `nrc == 1`（一次一行）。

<!-- src: ggml/src/ggml-cpu/arch/x86/quants.c -->
```c
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps( GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        qx = _mm256_sub_epi8( qx, off );

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
```

## 八、同名不同物（下）：ARM 的同名函数，契约不同

ARM 这份在 `__ARM_FEATURE_MATMUL_INT8` 下允许 `nrc == 2`（一次两行），并从 `vx + bx`、`vy + by` 取第二行 —— 这正是 x86 那份 `UNUSED(bx)` 的字段。**同名函数在不同架构上的契约可以不同**，调用方必须知道这一点。

实验：把两份文件并排看，只用 `grep -n "void ggml_vec_dot_q4_0_q8_0" ggml/src/ggml-cpu/arch/*/quants.c` 就能拿到七个行号：x86:701、arm:297、riscv:222、s390:217、powerpc:144、loongarch:647、wasm:232。

<!-- src: ggml/src/ggml-cpu/arch/arm/quants.c -->
```c
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0 * GGML_RESTRICT vx0 = vx;
        const block_q4_0 * GGML_RESTRICT vx1 = (const block_q4_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * GGML_RESTRICT vy0 = vy;
        const block_q8_0 * GGML_RESTRICT vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

```

## 九、AMX：把整颗 MUL_MAT 接过去

AMX（Intel）不新增类型、也不改 traits 表，而是注册一个 buffer type（`get_name` 返回 `"AMX"`，见 121-125 行）：权重在 `set_tensor` 时被重排成 VNNI 打包格式，`compute_forward` 里只拦 `GGML_OP_MUL_MAT`。这是「厂商层插在二维表之上」的最清晰形态：**判据是形状与类型白名单，代价是权重格式被绑定**。

<!-- src: ggml/src/ggml-cpu/amx/amx.cpp -->
```c++
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            return true;
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::amx
```

## 十、AMX 的参数：tile 形状与类型白名单

tile 常量（16/16/32）、8 个 TMM 寄存器编号、以及「哪些量化类型有 AMX 内核」的白名单都在这里。白名单只有 7 种类型 —— 不在名单里的张量根本不进 AMX buffer。

<!-- src: ggml/src/ggml-cpu/amx/common.h -->
```c
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4

#define AMX_BLK_SIZE 32

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7
```

## 十一、AMX 的判据：supports_op 到底看什么

这段是「厂商层怎么决定接管哪颗算子」的完整答案：算子必须是 `GGML_OP_MUL_MAT`、两个输入都连续、`src0` 必须身处 AMX buffer type、`src1` 必须在主机内存、`ne[0]` 必须是 `TILE_N * 2` 的整数倍、类型必须落在白名单里（每种类型还有各自的对齐要求）。任何一条不满足就返回 `false`，这颗算子交回默认路径 —— 于是**两套路径能在同一个模型里共存**。

<!-- src: ggml/src/ggml-cpu/amx/amx.cpp -->
```c++
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            return false;
        }
        if (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            return false;
        }
        if (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            return false;
        }
        if (op->ne[0] % (TILE_N * 2)) {
            return false;
        }
        int alignment;
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q8_0:
                alignment = TILE_K;
                break;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                break;
            case GGML_TYPE_F16:
                alignment = 16;
                break;
            default:
                return false;
        }
        if (src0->ne[0] % alignment) {
            return false;
        }
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        return true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
```

## 十二、垫层：ggml-cpu-impl.h 与 common.h

`ggml-cpu-impl.h` 是整个 CPU 后端的**跨 ISA 垫层**：一个头文件把 7 种 ISA 的内建函数头都引进来，并在没有硬件指令时用 C 写出等价实现（例如没有 DOTPROD 时用 `vmull_s8` 拼一个 `ggml_vdotq_s32`）。`common.h` 则提供模板内核需要的类型转换表（f16/bf16/f32/i32 的 to_f32/from_f32）与线程切分函数。

<!-- src: ggml/src/ggml-cpu/ggml-cpu-impl.h -->
```c
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#elif defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#include <immintrin.h>
#endif

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif
```

## 十三、垫层的另一半：类型转换表与线程切分

`type_conversion_table<T>` 把「模板参数 T」映射到「怎么转 float」，`get_thread_range` 把行按线程数切开（每线程一段连续行）。这两个看起来平淡的工具，是后面所有 `template <typename T>` 内核的共同底座。

<!-- src: ggml/src/ggml-cpu/common.h -->
```c++
// TODO - merge this into the traits table, after using row-based conversions
template <class T>
struct type_conversion_table;

template <>
struct type_conversion_table<ggml_fp16_t> {
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
struct type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
struct type_conversion_table<ggml_bf16_t> {
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;
};

template <>
struct type_conversion_table<int32_t> {
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;
};

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    return {ir0, ir1};
}
```

## 十四、旁路之一：hbm.cpp 根本不碰 SIMD

`hbm.cpp` 只在 `GGML_USE_CPU_HBM` 下编译，做的是**另一种内存来源**：用 `hbw_posix_memalign` 从高带宽内存（HBM）分配，释放时用 `hbw_free`。它注册成一个 buffer type，名字 `CPU_HBM`；除了分配器，其余接口全部复用 CPU 默认实现。

**它不是 SIMD 优化，而是访存优化**；把它放在这一课，是因为它和 AMX/KleidiAI 用了同一个挂载点。

<!-- src: ggml/src/ggml-cpu/hbm.cpp -->
```c
#ifdef GGML_USE_CPU_HBM

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include "hbm.h"

// buffer type HBM

#include <hbwmalloc.h>
```

## 十五、旁路之二：iqp.cpp 换的是 GEMM 的组织方式

IQP 路径针对**基于网格（grid）的 IQ 类型**（IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS）。它不重排权重、不注册 buffer type，而是把 **8 行 src0 一次性解码成 int8 面板**（`block_iqp_x8`），再对所有 src1 列做整数 GEMM —— 面板让一次解码服务多列，因此**只在批量足够大时才划算**。

激活侧用 VNNI 时按无符号字节喂（y + 128）再用 `bias[]` 校正；没有 VNNI 时改用 maddubs 的符号技巧。

<!-- src: ggml/src/ggml-cpu/iqp.cpp -->
```c
#define IQP_NB_ROWS 8

#define IQP_SB_SIZE 16                    // weights per sub-block
#define IQP_NSB     (QK_K / IQP_SB_SIZE)  // sub-blocks per super-block

// one super-block of a grid based IQ type decoded to int8, 8 rows interleaved:
// dfac[row] * iscales[sb*8 + row] * qs is bit identical to dequantize_row_iq*
struct block_iqp_x8 {
    float   dfac[8];               // f32 super-block scale, d * 2^-k
    int32_t bias[8];               // 128 * sum(qs * iscale), see GGML_IQP_USE_BIAS
    int8_t  iscales[IQP_NSB * 8];  // integer sub-block scales, in [-32, 31]
    int8_t  qs[QK_K * 8];          // qs[sb*128 + g*32 + row*4 + k] = column sb*16 + g*4 + k
};

static_assert(sizeof(block_iqp_x8) == 8 * sizeof(float) + 8 * sizeof(int32_t) + IQP_NSB * 8 + QK_K * 8,
              "wrong iqp_x8 block size/padding");

// feed the activations to VNNI as unsigned bytes (y + 128) and correct with bias[]; without VNNI the kernels use the maddubs sign trick instead and bias[] is not filled
#if defined(__AVX2__) && ((defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__))
#    define GGML_IQP_USE_BIAS 1
#else
#    define GGML_IQP_USE_BIAS 0
#endif
```

## 十六、iqp.h：判据与入口

头文件把这条路径的契约写得很清楚：什么时候划算、节点级判据、每线程 scratch 大小、以及「必须等 src1 转成 q8_K 并同步之后」才能调用实现。

<!-- src: ggml/src/ggml-cpu/iqp.h -->
```c
// batched mul_mat path for the grid based IQ types: decode 8 src0 rows at a time into per thread scratch
// (block_iqp_x8, see iqp.cpp) and run an integer gemm over them against all src1 columns

#ifdef __cplusplus
extern "C" {
#endif

// whether cne1 rows of src1 are enough for the decode to pay for itself, per expert, for MUL_MAT_ID
bool ggml_cpu_iqp_mul_mat_id_min_batch(int64_t cne1);

bool ggml_cpu_iqp_supports_mul_mat(const struct ggml_tensor * dst);

// node level test only - per expert eligibility is decided with ggml_cpu_iqp_mul_mat_id_min_batch
bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst);

// per thread panel scratch bytes, padded
size_t ggml_cpu_iqp_scratch_size(const struct ggml_tensor * dst);

// must be called after src1 has been converted to q8_K into params->wdata and the threads have synchronized on it
void ggml_compute_forward_mul_mat_iqp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
```

## 十七、llamafile：小矩阵的第三套 SGEMM

`llamafile_sgemm` 是 Mozilla tinyBLAS 的入口：**先按 A/B 类型 switch，再按 ISA 分派**，同一类型组合在不同架构下实例化不同模板。它只在 `n >= 2`（提示处理）时接管，小矩阵形状不合适就返回 false 让默认路径接手。

本课引用其头文件签名与实现文件里的分派骨架，完整模板体（4165 行）留作课外阅读。

<!-- src: ggml/src/ggml-cpu/llamafile/sgemm.h -->
```c
bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);
```

## 十七、iqp.cpp：判据长什么样

IQP 的判据不是形状白名单，而是**一批同时成立的条件**：类型在 8 种 grid IQ 里、`vec_dot_type` 必须是 Q8_K（这条路径假设 src1 会转成 q8_K）、运行时必须真有 AVX2、`src1` 必须是 F32、`ne[0]` 要能整除 QK_K、`ne[1]` 要能整除 8、`src0` 连续、`dst` 是 F32 连续。外加一个**逃生开关**：环境变量 `GGML_NO_IQ_PANEL` 一旦设置，这条路径整体关闭（用于 A/B 对比）。

<!-- src: ggml/src/ggml-cpu/iqp.cpp -->
```c
static bool iqp_supported_common(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (!iqp_type_supported(src0->type)) {
        return false;
    }

    // the path assumes the src1 conversion type is q8_K
    if (ggml_get_type_traits_cpu(src0->type)->vec_dot_type != GGML_TYPE_Q8_K) {
        return false;
    }

    // escape hatch to A/B the panel against the plain vec_dot path without rebuilding (--no-repack does not cover this path)
    static const bool disabled = getenv("GGML_NO_IQ_PANEL") != nullptr;
    if (disabled) {
        return false;
    }

    if (!ggml_cpu_has_avx2()) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] % QK_K != 0 || src0->ne[1] % IQP_NB_ROWS != 0) {
        return false;
    }

    if (src0->ne[3] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src0)) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float)) {
        return false;
    }

    return true;
}

bool ggml_cpu_iqp_supports_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (!iqp_supported_common(dst)) {
        return false;
    }

    if (src1->ne[1] < GGML_IQP_MIN_BATCH) {
        return false;
    }

    // plain 2D weight matmuls only (src1 may still be batched over ne12)
    if (src0->ne[2] != 1) {
        return false;
    }

    return true;
}
```

## 十八、llamafile：按类型 switch，再按 ISA 分派

下面这段是 `llamafile_sgemm` 的骨架（以 Atype = Q8_0 为例）：**外层 switch 查类型组合，每个 case 内部再用 `#if defined(ISA)` 选模板实例**。三种 ISA 各有一个 tinyBLAS 实现（AVX / ARM DOTPROD / PowerPC MMA），都不匹配就 `return false`，让上层回到默认路径。

<!-- src: ggml/src/ggml-cpu/llamafile/sgemm.cpp -->
```c++
    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q8_0> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#elif defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        if (n < 8 && n != 4)
           return false;
        if (m < 8 && m != 4)
           return false;
        tinyBLAS_Q0_PPC<block_q8_0> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }
```

## 十九、llamafile：只在提示处理（n >= 2）时接管

入口处先做两件事：`Ctype` 必须是 F32，且（非 MMA 平台）`n >= 2` —— 也就是只在「一次算多列」的提示处理阶段才值得用它；解码阶段（n = 1）直接返回 false。这一行注释就是它的适用场景说明。

<!-- src: ggml/src/ggml-cpu/llamafile/sgemm.cpp -->
```c++
    // only enable sgemm for prompt processing
#if !defined(__MMA__)
    if (n < 2)
        return false;
#endif

    if (Ctype != GGML_TYPE_F32)
        return false;
```

## 二十、SpacemiT：核型号探测与绑核

SpacemiT 的路径要先知道「这颗 SoC 上哪些核是 x100/a100」：`ime_env.cpp` 读 `/proc/cpuinfo` 拿每个核的 arch_id，必要时用环境变量在 QEMU 下注入，再用 `sched_setaffinity` 把线程绑到首选核；共享内存/大页/TCM 的选择也在这里定。

<!-- src: ggml/src/ggml-cpu/spacemit/ime_env.cpp -->
```c++
    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    char * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;
```

## 二十一、SpacemiT 的内存池与屏障

`spine_mem_pool` 提供三种后端：`posix_memalign`、透明大页（`madvise(MADV_HUGEPAGE)`）、1G 大页（`/dev/hugetlb_1g` + ioctl + mmap），另有按核分配的 TCM（紧耦合内存，通过可 dlopen 的 `spine_tcm` 库头文件方式加载）。跨核同步不用 pthread，而是自己的自旋屏障。

<!-- src: ggml/src/ggml-cpu/spacemit/spine_mem_pool.h -->
```c++
enum class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

struct spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_num{ 0 };
    bool   is_fake_tcm{ false };
};
```

---

## 说明

- 本课覆盖 `plan_matrix.py` 分配给 L5-05 的全部 **49 个源文件**，逐一声明于上表；其中 `arch/` 子树 16 个（7 个 `quants.c` + 4 个 `repack.cpp` + 5 个 `cpu-feats.cpp`）、`amx/` 5 个、`kleidiai/` 4 个、`spacemit/` 15 个、`llamafile/` 2 个、`hbm.*` 2 个、`iqp.*` 2 个，加 `arch-fallback.h`、`common.h`、`ggml-cpu-impl.h`。
- 文中提到的 `ggml/src/ggml-cpu/traits.cpp`、`ggml/src/ggml-cpu/quants.c`（`*_generic` 定义）、`ggml/src/ggml-cpu/repack.cpp`（默认 buffer type 的 repack）都属于 **L5-04**；`ggml/src/ggml-backend-reg.cpp`（变体加载器）属 L3-02；`ggml/src/ggml-cpu/ggml-cpu.cpp`（extra buffer type 注册）属 L5-01。本课只在正文中引用它们的机制，**不引用其文本，故不计入本课覆盖率**。
- 本课所有「条数」统计（各架构 `vec_dot` 个数、repack 内核个数、arch-fallback.h 各分支 `#define` 条数、同名函数行号）都由 v0.5.0 源码实测得出，可用 `grep -c` / 行号核对；它们不是源码注释里的说法。
