<!-- llama-coverage
ggml/src/ggml-common.h
ggml/src/ggml-quants.h
ggml/src/ggml-quants.c
ggml/src/ggml.c
-->

# L1-04 · 量化块结构：算子内层的压缩数据 — 源文件

**一句话**：在 llama.cpp 里，量化权重**不是**"每个数少几位"，而是**把元素切成定长的块，每块自带 scale**。`ggml_tensor` 的字节被解释成 `block_q4_0[]` / `block_q4_K[]` 这样的记录数组，块的大小（多少元素一块、一块多少字节）由类型号唯一决定。

这一课只讲**布局**：块里有哪些字段、每个字段多少字节、反量化函数怎么把字节还原成 float。至于"这个布局怎么被内核消费"（量化点积、SIMD、repack），属于 L5-03 / L5-04。

---

## 一、块大小的两个常量

整个量化体系只有两个"块大小"量级：小块 `QK4_0 = 32`（Q4_0 家族），超块 `QK_K = 256`（K-quant 家族）。`K_SCALE_SIZE = 12` 是 Q4_K 给"量化后的 scale/min"留的字节数 —— 第 6、7 幕会把 12 = 96 bit 拆开验算。

```text
QK4_0 = 32     # 小块：32 个元素一块
QK_K  = 256    # 超块：256 个元素一个 super-block
K_SCALE_SIZE = 12  # 8 个 6-bit scale + 8 个 6-bit min
```

下面第二段是本文件最前面的类型定义：在 CPU 的 C 编译分支里 `ggml_half` 就是 `uint16_t`，也就是 **2 字节**。后面所有 `sizeof(ggml_half)` 都按 2 算。

<!-- src: ggml/src/ggml-common.h -->
```c
#if defined(GGML_COMMON_DECL_C)
#include <stdint.h>

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

#define GGML_COMMON_DECL
//>> ---- ggml/src/ggml-common.h:86-90 ----
// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12
```

## 二、★ 小块家族：block_q4_0 与 block_q4_1

两个结构体只差一个字段，却差了 2 字节 / 块，也就是 0.5 bit / 权重。

| 类型 | 字段 | 字节 | bit / 权重 |
|---|---|---|---|
| `block_q4_0` | `d` + `qs[16]` | 18 | 4.5 |
| `block_q4_1` | `d` + `m` + `qs[16]` | 20 | 5.0 |

这里的数字全部来自 `static_assert` 的等式：`sizeof(ggml_half) = 2`（`ggml-common.h:6` 的 `typedef uint16_t ggml_half;`），`QK4_0 / 2 = 16`，所以 `sizeof(block_q4_0) = 18`。

注意 `GGML_COMMON_AGGR_U` 那个匿名 union：`d` / `m` 两个 fp16 与一个 `ggml_half2 dm` 共享同一块内存，方便 SIMD 一次搬 4 字节。

<!-- src: ggml/src/ggml-common.h -->
```c
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");
```

## 三、Q8_0：给量化点积当被乘数

`QK8_0` 也是 32，但每个元素占满 1 字节：`2 + 32 = 34` 字节 / 块，即 8.5 bit / 权重。它不是用来存模型权重的"高压缩"格式，而是量化点积里被乘的那一侧（L5-04 会看到 q4_0 × q8_0 的配对）。另一个区别是 `qs` 的类型：`int8_t` 而不是 `uint8_t` —— 8 bit 存的是有符号整数。

<!-- src: ggml/src/ggml-common.h -->
```c
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");
```

## 四、Q6_K：6-bit 主力

K-quant 家族里精度最高的一档：16 个 16 元素子块，每块一个 8-bit 有符号 scale，整个超块再共用一个 fp16 的 `d`。`ql` 存低 4 位、`qh` 存高 2 位 —— 和 Q4_K 的 6-bit scale 一样是"拆成两处存"的思路，只是拆的对象不同。

`sizeof(block_q6_K) = 2 + 16 + 192 = 210` 字节，源码注释写明 "Effectively 6.5625 bits per weight"（210 × 8 / 256 = 6.5625）。

<!-- src: ggml/src/ggml-common.h -->
```c
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");
```

## 五、反量化函数族：一个类型一个函数

`ggml-quants.h` 是这 3 个文件里唯一的"接口面"。它按同一套命名规则把两族函数排开：

```text
quantize_row_<类型名>_ref(...)   # 量化：参考实现，模型转换用
dequantize_row_<类型名>(...)     # 反量化：运行时把块还原成 float
```

注意第 53 行那个被注释掉的 `dequantize_row_q8_1`：Q8_1 只做量化（当被乘数），不需要反量化回 float，所以声明被注释掉了。

上面第 14 行的注释还说明了这些函数为什么是 `GGML_API`：`these functions are defined as GGML_API because they used by the CPU backend`。

<!-- src: ggml/src/ggml-quants.h -->
```c
// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API void quantize_row_q1_0_ref(const float * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q2_0_ref(const float * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
//>> ---- ggml/src/ggml-quants.h:45-63 ----
// Dequantization
GGML_API void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q2_0(const block_q2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
```

## 六、反量化 Q4_0：一个字节管两个半块

`qs[j]` 的低 4 位对应 `y[j]`、高 4 位对应 `y[j + qk/2]`，即一个字节跨越块的前后两半。这个排布让"取 32 个连续权重"必须先分两路走：这也是 L5-04 的 repack 要把 q4_0 重排成 4x4 / 4x8 交错布局的直接原因。

减 8 是为了把无符号的 `[0, 15]` 平移回对称区间：还原出的值是 `[-8d, 7d]`。

<!-- src: ggml/src/ggml-quants.c -->
```c
void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}
```

## 七、★ 量化 Q4_0：d = max / -8，没有 min

源码第 112 行的注释写着 `reference implementation for deterministic creation of model files` —— 这条路径是模型转换（L2-09）用的，要求确定性。

对称量化的三步：

1. 扫出 `amax`，连同取到 amax 的那个带符号值一起记进 `max`；
2. `d = max / -8`，`id = 1/d`；
3. `xi = MIN(15, (int8_t)(x*id + 8.5f))`，两个 nibble 合进一个字节。

`+8.5f` 同时完成"平移到 [0,15]"和"四舍五入"；`MIN(15, ...)` 兜住上溢 —— 正方向的极值会被压一点，这就是对称量化的代价。

<!-- src: ggml/src/ggml-quants.c -->
```c
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}
```

## 八、★ Q4_K 的 6-bit scale/min 解包

`K_SCALE_SIZE = 12` 字节 = 96 bit，而 8 个 scale + 8 个 min 各 6 bit 正好是 8 ×(6+6) = 96 bit。一位不多、一位不少。

装法就是 `get_scale_min_k4`：

| j | `*d`（scale） | `*m`（min） |
|---|---|---|
| 0..3 | `q[j] & 63` | `q[j + 4] & 63` |
| 4..7 | `(q[j+4] & 0xF) \| ((q[j-4] >> 6) << 4)` | `(q[j+4] >> 4) \| ((q[j-0] >> 6) << 4)` |

`q[0..7]` 只用低 6 位，剩下的最高 2 位被借给 `j >= 4` 的 scale/min；`q[8..11]` 出低 4 位。取用时必须移位拼接，这是 K-quant 内核比 Q4_0 多出来的固定开销。

<!-- src: ggml/src/ggml-quants.c -->
```c
static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

## 九、dequantize_row_q4_K：谁消费这些 6-bit

每个超块循环 4 次（`j += 64`），每次读两个子块的 scale/min，把 32 个字节的低半字节和高半字节分别还原成 32 个值 —— 一次 64 个元素。

注意 `d1 = d * sc` 与 `m1 = min * m`：超块头的 `d` / `dmin` 是 fp16，子块的 `sc` / `m` 是 6-bit 整数，两级相乘才是最终系数。这与 Q4_0 只有一个 `d` 形成对照（第 2、5 幕）。

<!-- src: ggml/src/ggml-quants.c -->
```c
void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}
```

## 十、类型号把一切绑在一起

`ggml_type_traits` 表（`ggml.c:632` 起）是"类型号"与"内存布局"的唯一连接点。每个类型登记三件事：

| 字段 | 含义 | Q4_0 的值 |
|---|---|---|
| `blck_size` | 多少元素算一块 | `QK4_0` = 32 |
| `type_size` | 一块多少字节 | `sizeof(block_q4_0)` = 18 |
| `to_float` | 怎么还原成 float | `dequantize_row_q4_0` |

对比同一段代码里的 `GGML_TYPE_F16`：它的 `blck_size = 1`、`type_size = sizeof(ggml_fp16_t)`，`to_float` 只是 `ggml_fp16_to_fp32_row` —— **没有块，就没有块布局这回事**。

<!-- src: ggml/src/ggml.c -->
```c
static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
//>> ---- ggml/src/ggml.c:669-676 ----
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
//>> ---- ggml/src/ggml.c:693-700 ----
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
```

---

## 说明

- 本课声明 4 个源文件。前 3 个是计划里指派给 L1-04 的 `ggml-common.h` / `ggml-quants.h` / `ggml-quants.c`；第 4 个 `ggml/src/ggml.c` 只用于第 8 幕与第十节 —— `ggml_type_traits` 表（`blck_size` / `type_size` / `to_float` 的注册）只存在于这个文件里，是"GGML_TYPE 与块大小的绑定"这一讲解要点的唯一出处。计划里 `ggml.c` 同时指派给 L1-02 / L1-03，本课只是追加引用，不改变其归属。
- 第 2、6 幕的字节数（18 / 20 / 34 / 144 / 210）由源码的 `static_assert` 等式与 `ggml_half = uint16_t` 直接算出，并已用一个只 include `ggml-common.h` 的小程序实测确认：`sizeof(block_q4_0) = 18`、`sizeof(block_q4_1) = 20`、`sizeof(block_q8_0) = 34`、`sizeof(block_q4_K) = 144`、`sizeof(block_q6_K) = 210`。
