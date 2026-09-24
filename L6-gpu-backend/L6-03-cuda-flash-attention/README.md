# L6-03 · ★ CUDA FlashAttention：三条路线与一张实例矩阵 — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-02`

## 学习目标

看完这一课，你应该能：

1. 说出 `fattn-vec` 与 `fattn-mma-f16` 各自适用的 **head size** 与 **K/V dtype**（对应验收点）；
2. 解释 `can_use_vector_kernel` 的三个形状条件，特别是 `% 64 == 0` 与 `!= 192` 各自的来历；
3. 说出三类实例文件名（vec / mma / tile）分别编码了哪些模板参数，以及为什么没被实例化的组合“不存在”；
4. 说明 `mask` / `max_bias`(ALiBi) / `logit_softcap` 三个修饰在 `launch_fattn` 里怎么被处理。

## 覆盖的源文件（15 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh` | 1303 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 2188 |
| `ggml/src/ggml-cuda/fattn-tile.cu` | 61 |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | 1356 |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | 610 |
| `ggml/src/ggml-cuda/fattn.cu` | 775 |
| `ggml/src/ggml-cuda/fattn.cuh` | 8 |
| `ggml/src/ggml-cuda/pad.cu` | 107 |
| `ggml/src/ggml-cuda/pad.cuh` | 6 |
| `ggml/src/ggml-cuda/pad_reflect_1d.cu` | 92 |
| `ggml/src/ggml-cuda/pad_reflect_1d.cuh` | 6 |
| `ggml/src/ggml-cuda/softcap.cu` | 38 |
| `ggml/src/ggml-cuda/softcap.cuh` | 6 |
| `ggml/src/ggml-cuda/unary.cu` | 722 |
| `ggml/src/ggml-cuda/unary.cuh` | 124 |

> **说明**：本课覆盖 plan 矩阵分配给 `L6-03` 的 15 个文件（`fattn*` / `pad*` / `unary*` / `softcap*`），全部计入覆盖率。
> **说明**：`template-instances/` 下的 82 个 fattn 实例文件按 plan 矩阵归属 L6-04，**不计入本课覆盖率**（正文中有说明与统计表）。
> **说明**：场景与正文里引用的 `ggml/src/ggml-cuda/ggml-cuda.cu`（算子分派与 softcap 融合）、`ggml/cmake/common.cmake`、`ggml/CMakeLists.txt`、`src/llama-graph.cpp`（L2-06 的 build_attn）都只作**跨课指路**，不作覆盖声明。

## 场景（10 幕）

1. **★ 不是一个 kernel：三条路线 + 一张实例矩阵** — 入口函数只做两件事：按形状与设备挑路线，再查表拿到那份已经编译进库的模板实例。
2. **第一步：head size 白名单** — 选择逻辑的第一段就是一个 switch (K->ne[0])；不在白名单里的 head size 直接返回 NONE。
3. **★ 路线判据：can_use_vector_kernel × 设备能力** — vec 只在小 batch 时赢；只要 GPU 有 Tensor Core 且 batch 够大，就走 mma。
4. **vec 的 dtype 矩阵：3 个 head size × 7 × 7** — DECL_FATTN_VEC_CASE 把 (head size, type_K, type_V) 三元组逐一声明成显式实例化；EXTERN 列表把 7×7 铺满。
5. **per-element：一个 block 128 线程，一次处理 1~2 个 Q 列** — vec 把 KQ 点积按“每个线程负责 D 的一段”切开；batch=1 时 cols_per_block=1，否则 2。
6. **tile：12 组 (DKQ, DV) 全是显式实例** — tile 是“没有 Tensor Core 也能跑”的通用路线：KQ 用 SIMT 的向量点积，覆盖全部 head size。
7. **mma：Tensor Core + ncols1 / ncols2 两个模板参数** — mma 的实例键是四元组 (DKQ, DV, ncols1, ncols2)：ncols1 是 Q 行方向的 tile 宽度，ncols2 是 GQA 方向的宽度。
8. **为什么 template-instances/ 必须存在：查不到就退化成 f16** — 没被显式实例化的组合，在编译期就不存在。运行期查表拿到 nullptr 时不是算错，而是慢。
9. **mask / ALiBi / softcap：三条路线共用的一段准备** — launch_fattn 从 op_params 取出 scale、max_bias、logit_softcap，再决定要不要转 f16、要不要切 stream-k。
10. **三条路线一张表 + 下一次看哪里** — 路线由 head size 与 dtype 决定；dtype 的支持范围由构建选项决定；具体 shape 组合靠模板实例覆盖。

## 核心结论

### ★ 三条路线，各管一段形状

| 路线 | 文件 | head size | K/V dtype | 被选中的条件 |
|---|---|---|---|---|
| `VEC` | `fattn-vec.cuh` | 64 / 128 / 256 | F16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,BF16（+F32 视作 F16） | `can_use_vector_kernel` 且 batch 很小（解码） |
| `TILE` | `fattn-tile.cuh` | 12 组 (DKQ,DV)，含 40 / 72 | 先转成 f16 | 无 Tensor Core，或 Volta 上中等 batch（兜底） |
| `MMA_F16` | `fattn-mma-f16.cuh` | 64…576（不含 40 / 72） | 先转成 f16（sparse 只有 4 组） | 有 Tensor Core 且 batch 够大（prefill） |

判据在 `ggml_cuda_get_best_fattn_kernel()`（`fattn.cu` 541-717 行）；入口 `ggml_cuda_flash_attn_ext()`（755-770 行）只是个 `switch`。

### ★ dtype 的支持范围是编译期决定的

`FATTN_VEC_CASE` 的实例化被 `if constexpr (GGML_CUDA_FA_<type_K>_<type_V>)` 包着；这个宏由 CMake 的 `GGML_CUDA_FA_QUANTS` 决定（默认四个组合）。所以“vec 支持哪些 dtype”有两层答案：**代码里声明了 7×7 种**（`fattn-vec.cuh` 574-609 行），**这次构建实际编进去的是其中一部分**。查不到时 `ggml_cuda_get_fattn_vec_case()` 返回 `nullptr`，`ggml_cuda_flash_attn_ext_vec()` 退化成 f16-f16 并打印警告 —— 结果对，速度差。

### mask / ALiBi / softcap 是三条路线共用的准备

三者都在节点 `op_params` 里（scale / max_bias / logit_softcap），由 `launch_fattn`（`fattn-common.cuh` 975 行起）统一取出：softcap 通过 `scale /= logit_softcap` 折进缩放，ALiBi 通过 `m0` / `m1` 变成每个 head 的 slope，mask 则必须是 F16 张量（1001 行的 assert），mma 路线还能把它压成 sparse 索引。

## 验收点

- [x] 保真门禁：31 处引用 —— 31 个引用块 / 37 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 15 项，无空课、无幻影；全局覆盖 1048/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
