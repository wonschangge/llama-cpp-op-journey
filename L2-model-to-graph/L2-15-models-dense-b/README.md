# L2-15 · 模型家族（六）：稠密 Transformer（下）与投机解码草稿模型 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-14`（模型家族（五）稠密 Transformer 上）；建议先看 `L2-06`（build_attn）、`L2-04`（KV cache 与 SWA 分层）

## 学习目标

看完这一课，你应该能：

1. 说出 GQA 在本仓库里的**唯一**表达方式，以及它在哪个结构体的哪个字段上（对应验收点）；
2. 指出 SWA 的三个超参字段名，并解释 `load_swa_pattern(ml, 4)` 生成的逐层开关长什么样；
3. 说明 MLA 的 `is_mla()` 判据，以及它在 KV cache 上造成的唯一差别（`has_v`）；
4. 说出草稿模型与主模型**共享 KV** 的图差异：`build_attn` 的哪两个实参变成 `nullptr`，为什么；
5. 区分"配置差异"（GQA / SWA / MLA）与"结构差异"（草稿模型），各举一条代码证据。

## 覆盖的源文件（43 个）

| 文件 | 行数 |
|---|---|
| `src/models/maincoder.cpp` | 152 |
| `src/models/mistral4.cpp` | 7 |
| `src/models/models.h` | 2663 |
| `src/models/modern-bert.cpp` | 173 |
| `src/models/mpt.cpp` | 171 |
| `src/models/muse-glimmer.cpp` | 204 |
| `src/models/nanbeige.cpp` | 186 |
| `src/models/nemotron.cpp` | 151 |
| `src/models/neo-bert.cpp` | 135 |
| `src/models/nomic-bert.cpp` | 52 |
| `src/models/olmo.cpp` | 143 |
| `src/models/olmo2.cpp` | 209 |
| `src/models/openelm.cpp` | 172 |
| `src/models/orion.cpp` | 142 |
| `src/models/paddleocr.cpp` | 108 |
| `src/models/pangu-embed.cpp` | 163 |
| `src/models/phi2.cpp` | 143 |
| `src/models/plamo.cpp` | 137 |
| `src/models/plamo3.cpp` | 196 |
| `src/models/qwen.cpp` | 141 |
| `src/models/qwen2.cpp` | 155 |
| `src/models/qwen2vl.cpp` | 144 |
| `src/models/qwen3.cpp` | 160 |
| `src/models/qwen3tts.cpp` | 4 |
| `src/models/qwen3vl.cpp` | 198 |
| `src/models/seed-oss.cpp` | 152 |
| `src/models/smollm3.cpp` | 153 |
| `src/models/spark2-5.cpp` | 147 |
| `src/models/stablelm.cpp` | 173 |
| `src/models/starcoder.cpp` | 146 |
| `src/models/starcoder2.cpp` | 159 |
| `src/models/t5.cpp` | 371 |
| `src/models/t5encoder.cpp` | 45 |
| `src/models/talkie.cpp` | 150 |
| `src/models/wavtokenizer-dec.cpp` | 265 |
| `src/models/xverse.cpp` | 137 |
| `src/llama-hparams.h` | 513 |
| `src/llama-kv-cache.cpp` | 2818 |
| `src/models/gemma4-assistant.cpp` | 201 |
| `src/llama-graph.h` | 1391 |
| `src/llama-model.cpp` | 3360 |
| `src/llama-graph.cpp` | 3916 |
| `src/llama-hparams.cpp` | 354 |

> **说明**：本课共覆盖 43 个源文件。其中 36 个是计划里 L2-15 的全部文件，**无一遗漏**。
> **说明**：另有 7 个源文件被本课引用并计入覆盖，它们在计划里归别的课：`src/llama-hparams.h` / `src/llama-hparams.cpp`（L2-02）、`src/llama-graph.h` / `src/llama-graph.cpp`（L2-06）、`src/llama-kv-cache.cpp`（L2-04）、`src/llama-model.cpp`（L2-05，L2-04 也已借用）、`src/models/gemma4-assistant.cpp`（L2-14）。本课只引用它们与本课论题直接相关的少量行；各自的主覆盖仍在那七课。
> **说明**：本课不讨论任何命令行参数，参数门禁的真值集来自真实二进制的帮助输出。

## 场景（10 幕）

1. **36 个模型文件，两类差异** — 本课一个视角：这 36 个文件之间的差异，是配置的差异，还是结构的差异。
2. **★ GQA：一个 struct，两个 head 数** — Q 的头数和 K/V 的头数写在同一个结构体里 —— 这就是 GQA 的全部：不是新算子，是两个数字不同。
3. **SWA：窗口是一个超参，不是一层新代码** — 窗口大小、窗口语义、哪些层用窗口 —— 三样东西全部落在 llama_hparams 里。
4. **★ 有 SWA 和没 SWA，共用同一份图** — Plamo3 把整份图写成一个模板：只有"注意力输入类型"这一个类型别名随窗口开关切换。
5. **MLA：压缩 KV，只体现为一个布尔量** — MLA 把 K/V 压成一份潜在向量再解压。在图这一层，它的全部痕迹是"V 那一张张量不分配"。
6. **一张表：36 个文件各自用哪种注意力** — 按"注意力输入类型"分五组。每一行的依据行，都是该文件里真正写下的那一行。
7. **文件名叫一个模型，内容可能一行图都没有** — models.h 里有 151 个模型类；其中一批用 using graph = ... 直接把别族的图借过来。
8. **草稿模型（一）：主模型先把每层输入导出来** — 草稿模型要"猜到主模型下一层会看到什么"，所以主模型的图必须把中间层输入挂成图输出。
9. **★ 共享 KV 的图差异：不传 K/V 的 build_attn** — 草稿模型只算 Q，把 k_cur / v_cur 两个实参写成 nullptr —— 于是它一个 K/V 张量都不需要。
10. **把这一课压成一张表** — 注意力变体改的是配置，草稿模型改的是结构 —— 这就是 L2-06 那句结论的边界。

## 核心结论

### 注意力变体 = 同一个算子家族的不同配置

三种变体在本仓库里的全部表达：

| 变体 | 配置在哪 | 图上的痕迹 |
|---|---|---|
| GQA | `hparams.n_head_kv_arr[]` | `build_qkv` 的第 6 个实参 |
| SWA | `swa_type` + `n_swa` + `is_swa_impl[]` | 一个类型别名 / 三处三元表达式 |
| MLA | `n_embd_head_k_mla_impl` / `n_embd_head_v_mla_impl` | `has_v = !is_mla` |

**没有一种新算子。** 这印证了 L2-06 幕 10 的结论：模型架构的差异 = 原语选择 + 超参。

### ★ 草稿模型是结构性差异

共享主模型 KV 的草稿模型与稠密主模型有三处结构性不同：

1. **权重清单更短**：只有 `wq` / `wo`，没有 `wk` / `wv`；
2. **`build_attn` 少传两个实参**：`k_cur` / `v_cur` 为 `nullptr`，K/V 从共享张量读；
3. **cache 不分配**：`layers.push_back(layer_share)` 直接挂主模型的层张量。

这不是把超参调小，而是这张图少了两个投影、少了一份 KV。L2-06 那句"差异 = 原语 + 超参"在这里**不成立** —— 这就是本课要划的边界。

## 验收点

- [x] 保真门禁：26 处引用 —— 26 个引用块 / 47 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 43 项，无空课、无幻影；全局覆盖 422/1290
- [ ] 参数门禁：存在非法命令行参数（见 check_flags.py 输出）
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
