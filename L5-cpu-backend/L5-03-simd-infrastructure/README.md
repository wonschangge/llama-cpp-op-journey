# L5-03 · ★ 向量化基础设施：一份宏，十一个架构分支 — 课件说明

> 层：**L5 · CPU 后端执行** ｜ 前置课：`L5-02`（一元与二元算子内核）

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml_vec_dot_*` 家族有哪些成员、签名里的 `nrc` 是什么意思（对应验收点）；
2. 解释 `simd-mappings.h` 的设计："一套宏 + 十一个架构分支"，以及 `GGML_F32_EPR` / `GGML_F32_STEP` / `GGML_F32_ARR` 三者的关系（对应验收点）；
3. 区分两种复用机制：**一份源码 + 宏映射**（vec.cpp / vec.h / simd-gemm.h）与 **同名函数 + 每架构一份实现**（quantized dot 家族）；
4. 说出 `ggml_vec_dot_q4_0_q8_0` 在 AVX2 与 NEON 上的实现差异（验收点）。

## 覆盖的源文件（4 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-cpu/vec.h` | 1571 |
| `ggml/src/ggml-cpu/simd-mappings.h` | 1320 |
| `ggml/src/ggml-cpu/vec.cpp` | 614 |
| `ggml/src/ggml-cpu/simd-gemm.h` | 227 |

> **说明**：本课引用的 4 个文件 —— `ggml/src/ggml-cpu/` 下的 `vec.h`、`vec.cpp`、`simd-mappings.h`、`simd-gemm.h` —— 全部计入覆盖率。
> **说明**：为保证"每一条断言都能在引用的代码里看到"，本课**不引用** `arch/x86/quants.c`、`arch/arm/quants.c`、`quants.c`、`quants.h`、`arch-fallback.h`、`CMakeLists.txt`、`ggml-cpu.c`、`ops.cpp`、`ggml-cpu-impl.h` 等文件。正文里对这些文件只有"指路"（附真实行号），**不计入本课覆盖率**；它们分别属于 L5-01 / L5-02 / L5-04 / L5-05 的覆盖范围。

## 场景（9 幕）

1. **向量化基础设施：四块拼图** — 一条点积的调用链：算子内核 -> ggml_vec_dot_* -> GGML_F32_VEC_* -> 某个 ISA 的 intrinsic。
2. **一套宏，九个名字** — 源码注释把设计意图写死了：只定义一套宏，基本运算只用宏写，接新架构 = 加一组宏。
3. **★ 同一组宏，三种 ISA 的展开** — 把三个分支从同一个头文件里并排取出来：判据、寄存器宽度、每步元素数，全在宏里。
4. **ggml_vec_dot_f32：家族样板** — 34 行里包含了向量化点积的全部套路：切尾巴、开寄存器数组、FMA 累加、归约、标量补齐。
5. **家族：三个浮点点积 + 一组常数** — 签名统一是 (n, s, bs, x, bx, y, by, nrc)；nrc 让一次调用算多行。
6. **同一个文件里的 ISA 特化分支** — ggml_vec_dot_bf16 走另一条路：不用宏，直接按 #if 给每个架构写死 intrinsic。
7. **simd-gemm.h：循环体不变，分块按 ISA 变** — 50 行里没有一个 intrinsic：微内核只用 GGML_F32_VEC_*，但行/列分块常数由架构决定。
8. **simd_gemm：宏模板 / RVV / 标量** — 整个头文件被 #if 切成三段，每段各定义一次同名函数 —— 编译期三选一。
9. **★ q4_0 x q8_0：AVX2 与 NEON 差在哪** — 同一个符号名，两份实现；它们共享的正是本课这四个文件里的东西。

## 核心结论

### ★ 一份宏，十一个架构分支

`simd-mappings.h` 只做一件事：把九个宏名（`GGML_F32_VEC` 与 `_ZERO/_SET1/_LOAD/_STORE/_FMA/_ADD/_MUL/_REDUCE`）按当前架构映射到具体 intrinsic。分支从 `__ARM_FEATURE_SVE`、`__ARM_NEON`、`__AVX512F__`、`__AVX__`、`__POWER9_VECTOR__`、`__wasm_simd128__`、`__SSE3__`、`__loongarch_asx/sx`、`__VXE__/__VXE2__` 一直到 `__riscv_v_intrinsic`，共十一个。

| 分支 | EPR | STEP | 向量类型 |
|---|---|---|---|
| `__AVX512F__` | 16 | 64 | `__m512` |
| `__AVX__` | 8 | 32 | `__m256` |
| `__ARM_NEON` | 4 | 16 | `float32x4_t` |
| 标量 | 无 | 无 | `float` |

三条 SIMD 分支的 `STEP / EPR` 都等于 4（`GGML_F32_ARR`）：**一步固定 4 个寄存器**。

### 两种复用机制，别混起来

| 机制 | 长什么样 | 本课例子 | 谁选 |
|---|---|---|---|
| 一份源码 + 宏映射 | 只写 `GGML_F32_VEC_*` | `vec.cpp` / `vec.h` / `simd-gemm.h` | 预处理器按宏选分支 |
| 一份源码 + ISA 特化分支 | 直接写 `_mm512_*` 等 | `ggml_vec_dot_bf16` | `#if` 选分支 |
| 同名函数、多份实现 | `ggml_vec_dot_q4_0_q8_0` 写两遍 | `arch/x86/quants.c` 与 `arch/arm/quants.c` | 构建系统按架构选文件 |

第三种是量化点积家族的做法：`ggml/src/ggml-cpu/CMakeLists.txt` 每次构建只把当前架构那一份 `arch/<arch>/quants.c` 加进源文件列表（x86 在 `:245`、arm 在 `:103`、riscv 在 `:443`）；某个架构没实现时，`arch-fallback.h` 会把 `*_generic` 重命名成正式名字顶上去（`arch-fallback.h:12`）。本课只指路，展开见 L5-05。

### ★ ggml_vec_dot_q4_0_q8_0：AVX2 与 NEON 的真实差异

```text
① 函数体不同源：arch/x86/quants.c:701 与 arch/arm/quants.c:297，构建期二选一
② 向量宽度：AVX2 走 __AVX__ 分支（EPR 8 / STEP 32 / __m256）
             NEON 走 __ARM_NEON 分支（EPR 4 / STEP 16 / float32x4_t）
③ 块 scale 的 fp16 -> fp32：x86 用 __F16C__ 的 _cvtsh_ss（simd-mappings.h:63）
                             ARM 用 (float)(__fp16)（simd-mappings.h:46-49）
```

共同点：两边都只写 `GGML_F32_VEC_*` 与 `GGML_CPU_FP16_TO_FP32` 这些抽象名，算法结构（切块、FMA、归约、尾巴）完全共享。

### 标量路径是基准，不是废物

每个向量化函数都保留了标量实现：`vec.cpp` 的 `#else // scalar`、`simd-gemm.h` 的 `#else // scalar path`。它同时承担三件事：没有 SIMD 的架构、向量化覆盖不到的尾巴、以及"结果对不对"的对照基准。

## 验收点

- [x] 保真门禁：18 处引用 —— 18 个引用块 / 41 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 4 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
