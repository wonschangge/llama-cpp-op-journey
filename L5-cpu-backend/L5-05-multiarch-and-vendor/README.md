# L5-05 · ★ 多架构 SIMD 与厂商加速 — 课件说明

> 层：**L5 · CPU 后端执行** ｜ 前置课：`L5-04`（量化与 repack）

## 学习目标

看完这一课，你应该能：

1. 说出同一个算子（如 Q4_0 权重的 `MUL_MAT`）在不同架构上分别落到哪个 kernel 文件（对应验收点）；
2. 解释 `arch-fallback.h` 为什么用「差集」而不是「清单」来表达编译期选择；
3. 说明 `arch/*/cpu-feats.cpp` 的 score 如何决定运行期使用哪一份 CPU 变体；
4. 区分厂商层的两种失败策略：KleidiAI 返回 `nullptr` 退回默认路径 vs SpacemiT 直接 `GGML_ABORT`；
5. 说出 `hbm.cpp` 与 `iqp.cpp` 各自优化的是什么（内存来源 / GEMM 组织），以及它们为什么不算 SIMD 优化。

## 覆盖的源文件（49 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-cpu/arch-fallback.h` | 378 |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | 540 |
| `ggml/src/ggml-cpu/common.h` | 96 |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | 4109 |
| `ggml/src/ggml-cpu/arch/x86/repack.cpp` | 6408 |
| `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp` | 328 |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | 4320 |
| `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 5466 |
| `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp` | 42 |
| `ggml/src/ggml-cpu/arch/riscv/quants.c` | 6597 |
| `ggml/src/ggml-cpu/arch/riscv/repack.cpp` | 1704 |
| `ggml/src/ggml-cpu/arch/riscv/cpu-feats.cpp` | 39 |
| `ggml/src/ggml-cpu/arch/s390/quants.c` | 1535 |
| `ggml/src/ggml-cpu/arch/s390/repack.cpp` | 226 |
| `ggml/src/ggml-cpu/arch/s390/cpu-feats.cpp` | 51 |
| `ggml/src/ggml-cpu/arch/powerpc/quants.c` | 2305 |
| `ggml/src/ggml-cpu/arch/powerpc/cpu-feats.cpp` | 83 |
| `ggml/src/ggml-cpu/arch/loongarch/quants.c` | 2310 |
| `ggml/src/ggml-cpu/arch/wasm/quants.c` | 1293 |
| `ggml/src/ggml-cpu/amx/amx.cpp` | 250 |
| `ggml/src/ggml-cpu/amx/amx.h` | 9 |
| `ggml/src/ggml-cpu/amx/common.h` | 116 |
| `ggml/src/ggml-cpu/amx/mmq.cpp` | 2512 |
| `ggml/src/ggml-cpu/amx/mmq.h` | 11 |
| `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` | 1920 |
| `ggml/src/ggml-cpu/kleidiai/kernels.cpp` | 1102 |
| `ggml/src/ggml-cpu/kleidiai/kernels.h` | 102 |
| `ggml/src/ggml-cpu/kleidiai/kleidiai.h` | 18 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp` | 1743 |
| `ggml/src/ggml-cpu/spacemit/ime.h` | 22 |
| `ggml/src/ggml-cpu/spacemit/ime_env.cpp` | 321 |
| `ggml/src/ggml-cpu/spacemit/ime_env.h` | 56 |
| `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp` | 1028 |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp` | 5769 |
| `ggml/src/ggml-cpu/spacemit/ime_kernels.h` | 190 |
| `ggml/src/ggml-cpu/spacemit/repack.cpp` | 1796 |
| `ggml/src/ggml-cpu/spacemit/repack.h` | 15 |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp` | 3179 |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.h` | 96 |
| `ggml/src/ggml-cpu/spacemit/spine_barrier.h` | 35 |
| `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp` | 761 |
| `ggml/src/ggml-cpu/spacemit/spine_mem_pool.h` | 33 |
| `ggml/src/ggml-cpu/spacemit/spine_tcm.h` | 410 |
| `ggml/src/ggml-cpu/llamafile/sgemm.cpp` | 4165 |
| `ggml/src/ggml-cpu/llamafile/sgemm.h` | 26 |
| `ggml/src/ggml-cpu/hbm.cpp` | 56 |
| `ggml/src/ggml-cpu/hbm.h` | 9 |
| `ggml/src/ggml-cpu/iqp.cpp` | 1254 |
| `ggml/src/ggml-cpu/iqp.h` | 40 |

> **说明**：本课覆盖 `plan_matrix.py` 分配给 L5-05 的全部 **49 个源文件**，逐一声明于上表；其中 `arch/` 子树 16 个（7 个 `quants.c` + 4 个 `repack.cpp` + 5 个 `cpu-feats.cpp`）、`amx/` 5 个、`kleidiai/` 4 个、`spacemit/` 15 个、`llamafile/` 2 个、`hbm.*` 2 个、`iqp.*` 2 个，加 `arch-fallback.h`、`common.h`、`ggml-cpu-impl.h`。
> **说明**：文中提到的 `ggml/src/ggml-cpu/traits.cpp`、`ggml/src/ggml-cpu/quants.c`（`*_generic` 定义）、`ggml/src/ggml-cpu/repack.cpp`（默认 buffer type 的 repack）都属于 **L5-04**；`ggml/src/ggml-backend-reg.cpp`（变体加载器）属 L3-02；`ggml/src/ggml-cpu/ggml-cpu.cpp`（extra buffer type 注册）属 L5-01。本课只在正文中引用它们的机制，**不引用其文本，故不计入本课覆盖率**。
> **说明**：本课所有「条数」统计（各架构 `vec_dot` 个数、repack 内核个数、arch-fallback.h 各分支 `#define` 条数、同名函数行号）都由 v0.5.0 源码实测得出，可用 `grep -c` / 行号核对；它们不是源码注释里的说法。

## 场景（10 幕）

1. **★ 一个算子，两维选择：编译期 + 运行期** — 49 个文件分成四类：七架构的 ISA 内核、五份特性探测、四种厂商加速、两条旁路。
2. **编译期那一维：每个架构缺哪些内核** — arch-fallback.h 是整张表的编译期半边：它逐架构列出「本架构没有的原生实现」。
3. **运行期那一维：变体打分，缺一项就出局** — 同一架构可以编出多份 CPU 变体；每份导出一个 score 函数，谁分高谁上场。
4. **★ 厂商层：在二维表上再插一张表** — KleidiAI 的做法：把「需要哪些 CPU 特性」写成表项，按顺序试，第一条命中就赢。
5. **同一个 ggml_vec_dot_q4_0_q8_0，七份实现** — x86 这份在同名文件里按 AVX2/AVX 再分一次；ARM 那份连函数契约都不一样。
6. **十六个 arch/ 文件，各自的真实主题** — 七份点积内核 + 四份 repack 内核 + 五份特性探测；repack 只有 x86/arm/riscv/s390 写了。
7. **AMX：用 tensor_traits 接管整颗 MUL_MAT** — 不新增类型、不改进 traits 表：注册一个 buffer type，让权重在写入时就重排成 VNNI 格式。
8. **SpacemiT：编译期 × 运行期 × 类型 的三级选择链** — RISC-V 厂商把矩阵扩展（IME1/IME2）做成了自定义指令；选中不了就 abort，不静默降级。
9. **剩下 26 个厂商/第三方文件，各自的真实主题** — amx 5 个、kleidiai 4 个、spacemit 15 个、llamafile 2 个 —— 四家四种接法。
10. **收束：最后 7 个文件 + 这张表的四层** — 剩下的是三条垫层和两条旁路；hbm 与 iqp 都不碰 SIMD，它们改的是内存与 GEMM 组织。

## 核心结论

### ★ 核心：一张二维决策表 + 一层可插拔的厂商层

```text
维度一（编译期）：arch-fallback.h 决定本架构编进哪些原生实现
                  -> 链接后 ggml_vec_dot_* / ggml_gemv_* 各只有一个定义
维度二（运行期）：arch/*/cpu-feats.cpp 给每份 CPU 变体打分，最高分被 dlopen
                  -> 同一颗 CPU 上，二进制里那套实现被选中
厂商层（运行期）：buffer type + tensor_traits，用形状/类型判据接管整颗算子
                  -> AMX / KleidiAI / SpacemiT / llamafile 四种接法
```

记住一句话：**「选哪个 kernel」不是一次选择，而是两次；厂商加速是在这两次之上再插一层。**

### 七个架构，一个函数名

`ggml_vec_dot_q4_0_q8_0` 在 v0.5.0 有七份实现，行号分别是 x86:701、arm:297、riscv:222、s390:217、powerpc:144、loongarch:647、wasm:232，守卫宏各不相同；各架构的 `vec_dot` 个数为 riscv 31、arm 25、x86 24、powerpc 19、loongarch 18、s390 13、wasm 10。**名字相同不代表契约相同**：ARM 在 I8MM 下允许 `nrc == 2`，x86 只允许 `nrc == 1`。

### 厂商层的共同形状：buffer type + traits

AMX、KleidiAI、SpacemiT 三家的代码量差了一个数量级（250 行 / 1900 行 / 1740 行），但挂载点完全一样：注册一个 `ggml_backend_buffer_type`，在 `init_tensor` 里把 traits 挂到 `tensor->extra`，在 `set_tensor` 里重排权重，在 `compute_forward` 里拦下算子。差别只在失败策略与重排格式。

### 两条旁路提醒：性能不只有 SIMD

`hbm.cpp` 换的是内存来源（`hbw_posix_memalign` / `hbw_free`），`iqp.cpp` 换的是 GEMM 的组织方式（8 行权重解码成 int8 面板 + 整数 GEMM，批量 >= 8 才划算），`llamafile/sgemm.cpp` 换的是小矩阵算法（tinyBLAS）。三者都不增加向量宽度，但都在同一个挂载点体系里生效。

## 验收点

- [x] 保真门禁：32 处引用 —— 32 个引用块 / 82 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 49 项，无空课、无幻影；全局覆盖 1048/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
