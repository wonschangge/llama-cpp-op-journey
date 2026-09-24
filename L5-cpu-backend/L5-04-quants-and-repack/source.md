<!-- llama-coverage
ggml/src/ggml-cpu/quants.h
ggml/src/ggml-cpu/quants.c
ggml/src/ggml-cpu/repack.h
ggml/src/ggml-cpu/repack.cpp
ggml/src/ggml-cpu/traits.h
ggml/src/ggml-cpu/traits.cpp
-->

# L5-04 · ★ 量化与 repack — 源文件

**一句话**：`GGML_TYPE_Q4_0` 只规定了"32 个权重一块、一块 18 字节、怎么反量化"；它**没有规定这些块在内存里怎么排**。CPU 后端用 `repack` 缓冲把 N 行权重重排成**交错块**（`block_q4_0x4` / `x8` / `x16`），让一条 SIMD 载入就能同时拿到**多行的同一批列** —— 这就是 repack 加速点积的秘密。

这一课回答两个问题：**交错之后的布局为什么能加速点积**，以及 **traits 如何按 ISA 与形状选出 4x4 / 4x8 / 8x8 / 16x1 这些变体**。回顾 L1-04：原始块布局（`d` + `qs`）在那里逐字讲过；回顾 L1-01：`nb[]` 是 `type` 的函数，而 repack 换的是"块与块之间怎么摆"，不是类型号。

---

## 一、CPU 侧的量化 ABI：quants.h

这个头文件是 CPU 后端内部的 C 接口：一族 `quantize_row_*`（float -> 量化块）与一族 `ggml_vec_dot_*`（量化块 x 量化块 -> float）。注意 `quantize_row_q8_0` /`quantize_row_q8_K`：它们就是 `vec_dot_type` 指定的**激活侧类型**，repack 路径在没有变体可用时会退回到它们（见第五节）。

<!-- src: ggml/src/ggml-cpu/quants.h -->
```c
// GGML CPU internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
```

## 二、基线：原始块上的标量点积

先看"没有 repack"时长什么样。`ggml_vec_dot_q4_0_q8_0_generic` 把 `vx` 直接当 `block_q4_0[]`、`vy` 直接当 `block_q8_0[]`：

- `x[ib].qs[j]` 的**低半**是元素 `j`，**高半**是元素 `j+16`（一块 32 个权重、16 字节）；
- 取出来要 `& 0x0F` / `>> 4`，再**减 8** 才是有符号值；
- 高半配的是 `y[ib].qs[j + qk/2]` —— y 里相隔 16 的那个 int8。

也就是说：**原始布局下，一个字节里塞了两个相隔 16 的元素，载入之后必须现场拆**。这正是 repack 要消掉的成本。

<!-- src: ggml/src/ggml-cpu/quants.c -->
```c
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}
```

## 三、★ repack 的块布局：block<K,N>

`repack.h` 用一个模板描述所有交错块：`block<K,N>` —— `N` 就是"交错进同一块的行数"，别名 `block_q4_0x4 / x8 / x16` 分别对应 N = 4 / 8 / 16；`K` 通过 `QK_0<K>()` 选量化家族（1 -> `QK1_0`、4 -> `QK4_0`、8 -> `QK8_0`）。

下面的逐字引用是 7 条 `static_assert` 与 7 个别名：**字节总数和原始布局完全相等** —— `sizeof(block<4,4>) == 4 * sizeof(ggml_half) + QK8_0 * 2` 就是 `4 x 18 = 72` 字节。repack 不省空间、不改类型号，只改排列。块里的两个成员（`d[N]` 与 `qs`）在第四幕的 `make_block_q4_0x4` 里能看到实际用法（`out.d[i]` / `out.qs[...]`）。

<!-- src: ggml/src/ggml-cpu/repack.h -->
```c
// control size
static_assert(sizeof(block<1, 4>) == 4 * sizeof(ggml_half) + QK1_0 / 2, "wrong block<1,4> size/padding");
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<4, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<4,16> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");
static_assert(sizeof(block<8, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 16, "wrong block<8,16> size/padding");

using block_q1_0x4 = block<1, 4>;
using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q4_0x16 = block<4, 16>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
using block_q8_0x16 = block<8, 16>;
```

## 四、重排实现：4 行轮流搬 + 异或偏置

`make_block_q4_0x4` 是"交错"这个词的全部实现，三个下标就把事情说完了：

```text
src_id     = i % 4                   从哪里来：第 i%4 行
src_offset = (i / 4) * interleave    该行的第几个块
dst_offset = i * interleave          放到输出的哪里
```

另一处容易被忽略的细节是 `elems ^= xor_mask`（0x88 / 0x88888888...）：Q4_0 的 `qs` 存的是无符号 4-bit 码，异或之后变成 4-bit 补码，于是内核里 `<< 4` / `& 0xF0` 得到的就是"值 - 8"，**减 8 这一步被提前做掉了**。

<!-- src: ggml/src/ggml-cpu/repack.cpp -->
```c
static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, int blck_size_interleave) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 2 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        const uint64_t xor_mask = 0x8888888888888888ULL;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
        }
    } else if (blck_size_interleave == 4) {
        const uint32_t xor_mask = 0x88888888;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}
```

## 五、激活侧契约：quantize_mat_q8_0_4x4

权重只重排一次，激活每次前向都要重排。`ggml_quantize_mat_q8_0_4x4_generic` 就是激活侧的对应物：它一次吃 4 行（4 个 token），每行各算各的 `amax` 与 `d`，然后按**同一个** `blck_size_interleave = 4` 把 4 行的 int8 交错写进 `block_q8_0x4`。

所以交错是**双边契约**：两侧粒度必须一致，内核才能用同一条乘加指令同时消费4 行权重与 4 个 token。`forward_mul_mat` 里只有 `ne11` 是 4 的倍数时才走这条路径，余下的 1..3 行退回 `from_float`（即 `quantize_row_q8_0`）交给 gemv。

<!-- src: ggml/src/ggml-cpu/repack.cpp -->
```c
void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}
```

## 六、★ 内核怎么读交错块：4x4 tile

`ggml_gemm_q4_0_4x4_q8_0_generic` 是"为什么能加速"的答案（下面引用到累加循环结束，写回段在 1832-1835 行）：

- `a_ptr` 是 `block_q8_0x4`（4 个 token 交错），`b_ptr` 是 `block_q4_0x4`（4 行权重交错）；
- 权重下标 `k * 4 * 4 + j * 4 + i`：`j` 每加 1 只走 **4 字节** —— 因为 4 行是交错的；
- 于是**一条载入就覆盖 4 行的同一批列**，而不是一行的 16 字节；
- 一个 `(m, j)` 做完 `blocklen = 4` 次乘加，`sumf[4][4]` 一次出 **16 个部分和**；
- `>> 4` 是收尾：`v0` 被 `<< 4` 放大过，两个乘积一起右移还原。

这也是各架构内核（AVX2 / NEON-i8mm / RVV）能共用同一套布局的原因：布局把"行"变成了向量载入的一个维度。

<!-- src: ggml/src/ggml-cpu/repack.cpp -->
```c
void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
```

## 七、钩子协议：traits.h

CPU 后端的默认实现是 `ggml_compute_forward_mul_mat`。要让 repack 内核接手，必须有钩子：`extra_buffer_type` 负责**认领**（`supports_op` / `get_tensor_traits`），`tensor_traits` 负责**执行**（`work_size` / `compute_forward`）。源码注释直接写明 `tensor_traits` 注册在 `tensor->extra` 里。

<!-- src: ggml/src/ggml-cpu/traits.h -->
```c
namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  public:
    virtual ~tensor_traits();
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
};

class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu
```

## 八、变体选择：ISA + 形状

`ggml_repack_get_optimal_repack_type` 是"选 kernel 变体"的唯一入口，对每个类型号按 `if (ISA) { if (形状门槛) return &变体; }` 的顺序试。以 Q4_0 为例（下面的逐字引用）：AVX2 或 SVE+i8mm 且 `ne[1] % 8 == 0` -> `q4_0_8x8_q8_0`；NEON+i8mm 且 `ne[1] % 4 == 0` -> `q4_0_4x8_q8_0`；NEON+dotprod -> `q4_0_4x4_q8_0`；VXE -> `q4_0_4x4_q8_0`；RVV（`__riscv_vlenb() * 8 == 256`）-> `q4_0_16x1_q8_0`。都不满足则 `return nullptr`（5140），张量就留在普通缓冲里。

<!-- src: ggml/src/ggml-cpu/repack.cpp -->
```c
    if (cur->type == GGML_TYPE_Q4_0) {
        if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_vxe()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
```

## 九、认领顺序：traits.cpp

最后一块拼图：`ggml_cpu_extra_compute_forward` 遍历所有注册的 extra buffer types，拿到 `tensor_traits` 后调用 `compute_forward`；**谁先返回 true，这个 op 就归谁**，默认实现不再执行。`ggml_cpu_extra_work_size` 是同一个模式的"问大小"版本 —— 图的 wdata 预算就是这么算出来的。

<!-- src: ggml/src/ggml-cpu/traits.cpp -->
```c
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}
```

---

## 说明

- 本课只引用 `ggml/src/ggml-cpu/` 下的 6 个文件（quants.c / quants.h / repack.cpp / repack.h / traits.cpp / traits.h），全部计入覆盖率。
- 文中为了定位而提到的 `ggml-common.h`（L1-04）、`ggml-cpu.c`（L5-01）、`arch/x86/repack.cpp` 与 `arch-fallback.h`（L5-05）、`src/llama-model.cpp`（L2 系列）、`common/arg.cpp` 均**不引用其源码**，故不计入本课覆盖率。
- `QK4_0 = 32`、`QK8_0 = 32` 的定义在 `ggml-common.h:194,251`；本课引用的 `repack.cpp:136` 有 `assert(QK8_0 == 32)` 可交叉验证。
- 第 3 幕的字节条按 `repack.h:31-46` 的 `static_assert` 与别名画出：`d[4]` = 4 x `sizeof(ggml_half)` = 8 字节，`qs` = `QK8_0 * 2` = 64 字节，合计 72 = 4 x 18。成员名 `d` / `qs` 见第 4 幕 `make_block_q4_0x4` 的 `out.d` / `out.qs`。
- 两处引用被本仓库 lint 的"Python 元组写进 JS"检查误伤（该检查扫的是 `[(` 相邻形式，而 C 的数组下标恰好可能长成这样）：`repack.h:28` 的 `qs[(QK_0<K>() * N * K) / 8]` 与 `repack.cpp:1834` 的写回行。两处分别改引用 `repack.h:31-46` 与 `repack.cpp:1785-1831`；被略去的行在上文已用文字指明位置。
