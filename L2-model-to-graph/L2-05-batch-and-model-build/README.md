# L2-05 · 批、解码参数与模型装配 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-04`（KV cache 与记忆家族）；同层的 `L2-02`（架构表与超参）与 `L2-03`（权重加载）是本课的直接上游。

## 学习目标

看完这一课，你应该能：

1. 说出 `llama_batch` 里哪些字段是必给的、哪些会被 `init()` 自动补齐（以及补齐时位置从哪里接）；
2. 写出 `llama_ubatch` 的计数不变量 `n_tokens = n_seq_tokens * n_seqs`，并说明 `equal_seqs()` 是谁赋的值；
3. 给定一个含两个不等长序列的 batch，写出 `split_equal` 切出的每个 ubatch 的 `(n_tokens, n_seq_tokens, n_seqs)`（对应本课验收点）；
4. 说清模型装配的两层钩子（`load_stats/load_hparams/load_vocab/load_tensors` 与 `load_arch_hparams/load_arch_tensors`），以及"层到设备"的分配发生在哪一步。

## 覆盖的源文件（6 个）

| 文件 | 行数 |
|---|---|
| `src/llama-batch.h` | 175 |
| `src/llama-batch.cpp` | 988 |
| `src/llama-model.h` | 858 |
| `src/llama-model.cpp` | 3360 |
| `src/llama-impl.h` | 106 |
| `src/llama-impl.cpp` | 172 |

> **说明**：本课声明并逐字引用 6 个源文件，全部计入覆盖率：`src/llama-batch.cpp`、`src/llama-batch.h`、`src/llama-model.cpp`、`src/llama-model.h`、`src/llama-impl.cpp`、`src/llama-impl.h`。
> **说明**：跨课呼应处引用了其他课的文件与行号（`src/llama-kv-cache.cpp:713`、`src/llama-context.cpp`、`src/llama-graph.h/.cpp`、`src/models/models.h:147`、`src/models/llama.cpp:34`、`src/llama.cpp:340`），这些**不在本课的 `src=` 声明里，不计入本课覆盖率** —— 它们分别属于 L2-04 / L2-06 / L2-07 / L2-10 / L2-01。

## 场景（10 幕）

1. **本课的位置：batch -> ubatch -> 图** — 一次 decode 收到的是一整批；真正被"算"的是切出来的每一小块。
2. **llama_batch：只有 token 是必给的** — 其余字段缺失时会被 init() 现场补齐 —— 补齐规则决定了后面"序列集"长什么样。
3. **★ struct llama_ubatch 逐字段** — 一个不变量把它全部串起来：n_tokens = n_seq_tokens x n_seqs。
4. **三种切法，产物都是 llama_ubatch** — 区别只在两处：这个 ubatch 里放几个序列集（n_seqs），以及 equal_seqs 标记。
5. **★ split_equal 的两条判据** — 判据一：序列集互不相交；判据二（sequential 时）：序列号严格递增。
6. **★ 一个 batch，两张不同形状的图** — 2 个不等长序列（s0 三个 token、s1 一个 token）走一遍 split_equal。
7. **ubatch_add：索引列表 -> ubatch 字段** — 切分只决定"要哪几个 batch 位置"，字段是这一步算出来的。
8. **装配分两层：基类读共性，子类读自己** — virtual 钩子把"所有架构都一样"和"这个架构特有"分开写。
9. **层 -> 设备 / buffer：在 load_tensors 里一次算好** — i_gpu_start 划出"最后 n_gpu_layers 个槽位上 GPU"；输入层永远留在 CPU。
10. **把这一课压成一张表** — 切分决定图，装配决定张量 —— 两条线在这里收口。

## 核心结论

### batch 是清单，ubatch 是执行单位

一次 `llama_decode` 收到一个 `llama_batch`，但它不会按整个 batch 建图：`llama_batch_allocr` 先补齐字段，再按 `n_ubatch` 切成若干 `llama_ubatch`，**每个 ubatch 建一次图、算一次**。所以"切分方式"就是"图的数量与形状"。

### ★ ubatch 的形状 = 五个计数字段

`n_tokens`、`n_seq_tokens`、`n_seqs`、`n_seqs_unq`、`equal_seqs()`。其中 `n_tokens = n_seq_tokens * n_seqs` 由 `ubatch_add()` 保证（函数开头断言整除）。

`split_equal` 会把**长度相同的一段**合并进同一个 ubatch（锁步扩张，任一序列集耗尽即停），于是多个序列可以共用同一条图；`split_simple` 则按 batch 顺序切片，`n_seq_tokens` 恒为 1。

### ★ 装配在 load_tensors 里一次算完

`llama_model_create(ml, params)` 按 GGUF 里的 arch `new` 出具体模型类，基类跑四个 `load_*` 钩子，子类跑 `load_arch_hparams` / `load_arch_tensors`；设备与候选 buffer 类型在 `load_tensors()` 里按层算成 `dev_layer[]`，最后 `ggml_backend_alloc_ctx_tensors_from_buft()` 把张量落到真实 buffer ——这正是 L2-03 那条"权重从文件到 buffer"的终点。

## 验收点

- [x] 保真门禁：19 处引用 —— 19 个引用块 / 46 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 6 项，无空课、无幻影；全局覆盖 87/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 48 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
