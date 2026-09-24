# L6-02 · ★ CUDA 量化矩阵乘：mmq / mmvq / mmvf — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-01`（CUDA 后端骨架 —— 本课的分派点就在它的 `ggml_cuda_mul_mat()` 里）

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml_cuda_mul_mat()` 里判据的**求值顺序**，并指出 mmvq 与 mmq 谁先被问到（对应验收点）；
2. 背出 mmvq 的默认阈值 `MMVQ_MAX_BATCH_SIZE = 8` 与 mmq 的 `MMQ_DP4A_MAX_BATCH_SIZE = 64`，并说明为什么 mmq 的判据里还有一道 `turing_mma_available()` 短路；
3. 解释 `vecdotq.cuh` 为什么给同一类型定义两个 VDR（`_MMVQ` 窄、`_MMQ` 宽）；
4. 说明 `load_tiles` 如何把一个 warp 铺到 tile 的 K 方向与 I 方向上；
5. 解释 `mul_mat_q_switch_J` 为什么需要 16 个模板实例，以及它与 `template-instances/` 的关系。

## 覆盖的源文件（11 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-cuda/mma.cuh` | 1517 |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | 1769 |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | 1241 |
| `ggml/src/ggml-cuda/mmq.cu` | 394 |
| `ggml/src/ggml-cuda/mmq.cuh` | 1606 |
| `ggml/src/ggml-cuda/mmvf.cu` | 876 |
| `ggml/src/ggml-cuda/mmvf.cuh` | 15 |
| `ggml/src/ggml-cuda/mmvq.cu` | 1568 |
| `ggml/src/ggml-cuda/mmvq.cuh` | 19 |
| `ggml/src/ggml-cuda/vecdotq.cuh` | 1381 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5857 |

> **说明**：覆盖率声明含 11 个文件：计划清单里 L6-02 的 10 个，外加 `ggml/src/ggml-cuda/ggml-cuda.cu` —— 本课验收点（mmq / mmvq 在什么形状下被选中）的分派点 `ggml_cuda_mul_mat()` 就在该文件，必须逐字引用。该文件 L6-01 也已声明，两课共同覆盖同一文件是允许的（覆盖是并集）。
> **说明**：场景中的派生数字（如 `threads_per_row = 256/(4*2) = 32`、`MMQ_TILE_Y_K = 36`）由 `ggml/src/ggml-common.h` 的 `QK4_0 = 32` / `QR4_0 = 2` / `QR8_1 = 1` 推出，本课不引用该文件，故不计入本课覆盖率。
> **说明**：`mmf`（Tensor Core 大矩阵路径）与 cuBLAS 兜底只在本课的分派表里出现名字与行号，其实现分别属于 L6-04 与 cuBLAS，本课不展开。

## 场景（10 幕）

1. **一个 MUL_MAT，五条内核路线** — ggml_cuda_mul_mat() 里没有 switch(op)，只有一串 should_use_* 判据，按顺序短路。
2. **mmvq：ne11 不超过 8 就走向量内核** — 这个 8 写在头文件的宏里；部分架构和量化类型还会把它压得更小。
3. **★ mmq：有 Tensor Core 就一律走，否则 ne11 小于 64** — mmq 的门槛比 mmvq 宽得多，因为它把权重搬进 shared memory 之后可以被多个 token 复用。
4. **mmvf：float 路径先看内存对齐，再看 batch** — 三条对齐前提任意一条不满足就崩溃或走不通，所以它们排在 batch 阈值前面。
5. **同一份点积源码，两套展开宽度** — vecdotq.cuh 给每个量化类型定义两个 VDR：MMVQ 用窄的，MMQ 用宽的。
6. **tile 点积把 mma 当积木：16x8 乘 8x8** — mma.cuh 把 PTX 的 mma 指令包装成 tile 类型；mmq 的点积直接用它拼出一次 tile 乘。
7. **K 循环：搬一次，点两次** — 每轮 K 迭代把 x 的一整条 tile 搬进 shared memory，然后分两半喂给同一个点积函数。
8. **协作加载：一个 warp 摊平一整块 tile** — 线程编号先按 K 方向切（每个线程固定搬哪几块），再按 I 方向切（搬第几行）。
9. **★ batch 不只选内核，还选 tile 的宽度 J** — 进了 mmq 之后，还要在 16 个编译期 J 实例里挑一个：J 是"一次处理几个 token"。
10. **把三套内核压成一张表** — 同一个量化点积，三种形状假设。选择依据只有一条：矩阵有多瘦。

## 核心结论

### 验收点

- [x] 保真门禁：16 处引用 —— 16 个引用块 / 36 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 11 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
