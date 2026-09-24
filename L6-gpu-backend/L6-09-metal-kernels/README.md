# L6-09 · Metal 后端：Metal 内核 — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-08`（Metal 主机端与图融合：谁选 pipeline、谁决定融合）；相关：`L6-04`（CUDA 的 template-instances 对照）、`L1-04`（量化块的字节布局）

## 学习目标

看完这一课，你应该能：

1. 说出 `mul_mv` 相比 `mul_mm` 在**什么形状下更优**：`src1` 只有一行或少数几行时走 `mul_mv`，因为 `mul_mm` 的线程组 tile 列方向固定是 32 列（对应验收点）；
2. 解释 Metal 如何用**一份 `mul_mm.metal`** 覆盖 23 种量化类型（模板参数 + `host_name` 实例化），并说出它与 CUDA `template-instances/` 的异同；
3. 说出 `N_SIMDWIDTH = 32` 之下，`simd_sum` / `simd_max` 与 threadgroup 内存如何组成两级归约；
4. 说明 `dequantize.h` 为什么要 `#define GGML_COMMON_DECL_METAL` 再 include `ggml-common.h`；
5. 按族说出 kernels 目录 23 个文件各自的职责。

## 覆盖的源文件（23 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-metal/kernels/common.h` | 127 |
| `ggml/src/ggml-metal/kernels/mul_mm.metal` | 968 |
| `ggml/src/ggml-metal/kernels/mul_mv.metal` | 3399 |
| `ggml/src/ggml-metal/kernels/fa.metal` | 2477 |
| `ggml/src/ggml-metal/kernels/dequantize.h` | 736 |
| `ggml/src/ggml-metal/kernels/rope.metal` | 334 |
| `ggml/src/ggml-metal/kernels/norm.metal` | 319 |
| `ggml/src/ggml-metal/kernels/softmax.metal` | 224 |
| `ggml/src/ggml-metal/kernels/quantize.h` | 263 |
| `ggml/src/ggml-metal/kernels/quantize.metal` | 481 |
| `ggml/src/ggml-metal/kernels/unary.metal` | 401 |
| `ggml/src/ggml-metal/kernels/binbcast.metal` | 229 |
| `ggml/src/ggml-metal/kernels/misc.metal` | 659 |
| `ggml/src/ggml-metal/kernels/conv.metal` | 725 |
| `ggml/src/ggml-metal/kernels/pool.metal` | 149 |
| `ggml/src/ggml-metal/kernels/upscale.metal` | 180 |
| `ggml/src/ggml-metal/kernels/reduce.metal` | 229 |
| `ggml/src/ggml-metal/kernels/argsort.metal` | 480 |
| `ggml/src/ggml-metal/kernels/tri.metal` | 70 |
| `ggml/src/ggml-metal/kernels/solve_tri.metal` | 76 |
| `ggml/src/ggml-metal/kernels/ssm.metal` | 477 |
| `ggml/src/ggml-metal/kernels/wkv.metal` | 180 |
| `ggml/src/ggml-metal/kernels/gated_delta_net.metal` | 265 |

> **说明**：本课引用 `ggml/src/ggml-metal/kernels/` 下的 23 个文件，全部计入覆盖率；清单来自 `python3 tools/plan_matrix.py --files`。
> **说明**：第 6 幕提到的 `r1ptg` 选择逻辑（按 `ne11` 在 2 到 8 之间取 2/3/4/5）在 `ggml/src/ggml-metal/ggml-metal-ops.cpp` 里，该文件由 L6-08 覆盖；本课不引用其源码，故不计入本课覆盖率。
> **说明**：`ggml/src/ggml-metal/ggml-metal-impl.h`（`N_MM_*` / `N_R0_*` / `N_SG_*` 常量）同样由 L6-08 覆盖，本课不引用其源码，不计入本课覆盖率。第 4 幕与第 9 幕出现的宏名来自 `common.h:3` 的 include。
> **说明**：`ggml/src/ggml-common.h`（块结构体定义）由 L1-04 覆盖，本课只引用 `dequantize.h` 里打开它分支的那几行，不计入本课覆盖率。该头文件按 `GGML_COMMON_DECL_C` / `_CPP` / `_METAL` / `_CUDA` / `_HIP` / `_SYCL` 分支出各后端要的结构体定义 —— 第 8 幕的卡片讲的就是这件事。

## 场景（9 幕）

1. **23 个源文件、8 个族：内核层的全景** — 每个 .metal 都从 #include "common.h" 开始；common.h 再把主机的参数头 ggml-metal-impl.h 拉进来。
2. **★ 模板参数就是量化类型：一份源码，二十多种 block_q** — kernel_mul_mm 的模板参数里有 block_q（量化块类型）、nl（每 16 个权重跨几个块）和一个反量化函数指针。
3. **实例化清单 = 量化支持矩阵** — typedef decltype(...) 先固定住与类型无关的参数，之后每一行 host_name 就是一个可被主机端取用的 pipeline。
4. **mul_mm 的 tile：64 x 32 的输出块** — 每个线程组算输出矩阵的一块；A 先反量化进线程组内存，再用 simdgroup 矩阵乘累加。
5. **mul_mv 的 per-row 结构：一行 src1 对 NR0 行 src0** — 每个线程组只认 src1 的一行；这一行被缓存进寄存器，在 NR0 行 src0 之间反复复用。
6. **★ src1 有几行决定走哪条路** — mul_mv 的 ext 变体把 r1ptg 行 src1 绑进一个线程组；实例化里 r1ptg 只取 2 到 5 —— 它就是为"少数几行"设计的。
7. **flash attention：DK / DV 也是模板参数** — kernel_flash_attn_ext 的头维度是编译期常量；host_name 里 dk128_dv128 这样的后缀就是一对 (DK, DV)。
8. **同一个 block_q：CPU 与 Metal 共用一份定义** — dequantize.h 用 GGML_COMMON_DECL_METAL 打开 ggml-common.h 的 Metal 分支 —— 块结构体不是 Metal 特有的。
9. **把这一课压成一张表** — 内核在哪、怎么覆盖量化类型、怎么用 SIMD group、什么时候换另一条路 —— 四个问题。

## 核心结论

### 23 个文件、8 个族、13425 行

本课覆盖 `ggml/src/ggml-metal/kernels/` 全部 23 个文件，按"算什么"分成 8 个族：

| 族 | 文件数 | 行数 | host_name |
|---|---|---|---|
| 矩阵乘 | 2 | 4365 | 245 |
| 注意力与序列混合 | 4 | 3395 | 567 |
| 逐元素与形状 | 6 | 2337 | 61 |
| 量化与反量化 | 3 | 1477 | 98 |
| 归约 / 排序 / 三角 | 4 | 851 | 20 |
| 归一化与 softmax | 2 | 541 | 18 |
| 位置编码 | 1 | 333 | 8 |
| 公共头 | 1 | 126 | 0 |

三个数字都由命令数出来：`wc -l`、`grep -c host_name`、`tools/plan_matrix.py --files`。

### ★ 模板 + host 侧实例化 = 量化支持矩阵

`kernel_mul_mm` 的模板参数里，决定"支持哪种量化"的是三个：

```text
typename block_q                            // 块类型：block_q4_0 / block_q4_K / ...
short nl                                    // 一个块装 16*nl 个权重
void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &)
```

同一份 `mul_mm.metal` 里出现了 **23 种 block 类型**、**111 处 `host_name`**（51 处非 MoE）。主机端按 tensor 的 `type` 去取对应名字的 pipeline —— 名字就是两侧的接口。

**与 L6-04 的对照**：CUDA 把每种实例写进 `template-instances/` 的独立文件（几十个 `.cu`，编译单元之间靠显式实例化衔接）；Metal 把实例化清单直接写在同一个 `.metal` 末尾，用 `template [[host_name("...")]]` 标记。两者解决同一个问题（避免运行期分发），代价也一样：**实例数随类型数增长**。fa.metal 就是代价的极端例子 —— 559 处 `host_name`、16 种 `(DK, DV)` 组合，源码注释自己写着 this is quite ugly。

### ★ 形状判据：mul_mv 还是 mul_mm

**src1 的行数（`ne11`）决定走哪条内核。**

| 形状 | 更优 | 源码依据 |
|---|---|---|
| `ne11 == 1`（逐 token 解码） | `mul_mv` | 线程组只服务 src1 的一行：`r1 = tgpig.y`（`mul_mv.metal:237`）；该行缓存在 `float yl[16]`（`:265`）里被 NR0 行 src0 复用（`:285-287`） |
| `ne11` 为 2 到 5 | `mul_mv_ext` | `i11 = tgpig.y * r1ptg`（`mul_mv.metal:623`），一个线程组吃下 `r1ptg` 行 src1；实例化只覆盖 2/3/4/5 |
| `ne11` 大（prompt 批量） | `mul_mm` | tile 固定 64 x 32（`mul_mm.metal:164-165`，第 175 行注释写明 64x32）；A 的反量化结果进线程组内存（`:52`）后被整块 tile 复用 |

一句话：**`mul_mm` 用"固定 32 列"换 A 块的复用；`mul_mv` 用"每行一次点积"换掉列方向的空转。**src1 只有一行时，32 列里 31 列是白算的。

### SIMD group 是内核的基本单位

`common.h:23` 把 `N_SIMDWIDTH` 钉成 32。所有跨线程的归约都走同一个两级范式：

```text
第一级：simd_sum / simd_max / simd_shuffle_down   -- SIMD group 内（寄存器）
第二级：threadgroup float * + threadgroup_barrier -- 跨 SIMD group（线程组内存）
```

`norm.metal:42-53`、`softmax.metal:49-65`、`reduce.metal:28-45`、`misc.metal:26-33` 都是这个范式的实例。`softmax.metal:50` 的 `if (tptg.x > N_SIMDWIDTH)` 说明：**只有线程组比一个 SIMD group 宽时，第二级才需要**。

### 块定义只有一份

`dequantize.h:5-10` 用 `GGML_COMMON_DECL_METAL` 打开 `ggml-common.h` 的 Metal 分支，于是 Metal 看到的是与 CPU、CUDA 完全相同的 `block_q4_0` / `block_q4_K`。

**内核实现可以各写各的，块的内存布局必须共用一份** —— 否则 GGUF 文件就没法跨后端读。L1-04 讲这些块的字节布局，本课讲 Metal 怎么用它们。

## 验收点

- [x] 保真门禁：33 处引用 —— 33 个引用块 / 60 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 23 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 4 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
