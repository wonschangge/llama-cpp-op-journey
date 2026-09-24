# L2-12 · 模型家族（三）：稀疏专家 MoE（上） — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-06`（★ 计算图骨架 llama-graph：`llm_graph_context` 与 `build_ffn`）

## 学习目标

看完这一课，你应该能：

1. 说出 MoE 层在计算图上的**七个阶段**，以及每个阶段对应的真实 `ggml_*` 算子（对应验收点）；
2. 默画出 **top-k 路由 + 专家并行** 子图，并说明 `selected_experts` 在这张图里被用了几次、用在哪；
3. 解释"稀疏"在图上的确切含义：专家张量多一维 + 一次按索引收集，其余与稠密 FFN 同构；
4. 区分 `n_expert` / `n_expert_used`（形参）与 `hparams.n_expert_groups` / `hparams.n_group_used`（分组路由，来自 hparams）这两组参数；
5. 指出 24 个受检文件里哪几个名实不符，并说出各自的实情。

## 覆盖的源文件（25 个）

| 文件 | 行数 |
|---|---|
| `src/llama-graph.cpp` | 3916 |
| `src/models/deci.cpp` | 192 |
| `src/models/dbrx.cpp` | 155 |
| `src/models/grovemoe.cpp` | 194 |
| `src/models/afmoe.cpp` | 284 |
| `src/models/arctic.cpp` | 181 |
| `src/models/bailingmoe.cpp` | 181 |
| `src/models/bailingmoe2.cpp` | 212 |
| `src/models/bert.cpp` | 222 |
| `src/models/cohere2moe.cpp` | 440 |
| `src/models/deepseek.cpp` | 195 |
| `src/models/deepseek2ocr.cpp` | 81 |
| `src/models/dots1.cpp` | 194 |
| `src/models/ernie4-5-moe.cpp` | 134 |
| `src/models/ernie4-5.cpp` | 165 |
| `src/models/exaone-moe.cpp` | 240 |
| `src/models/glm4-moe.cpp` | 445 |
| `src/models/granite-moe.cpp` | 85 |
| `src/models/granite-swa.cpp` | 320 |
| `src/models/granite.cpp` | 321 |
| `src/models/grok.cpp` | 224 |
| `src/models/hunyuan-moe.cpp` | 188 |
| `src/models/hy-v3.cpp` | 391 |
| `src/models/laguna.cpp` | 332 |
| `src/models/lfm2moe.cpp` | 86 |

> **说明**：**覆盖声明**：本课声明的源文件是 **25 个** —— `src/models/` 下计划指派的 24 个 `.cpp`，外加 `src/llama-graph.cpp`（`build_moe_ffn` 的定义处）。
> **说明**：**`src/llama-graph.cpp` 属于 L2-06**（★ 计算图骨架 llama-graph）。本课显式引用并声明它，是因为 MoE 的公共原语定义在这里；覆盖度按并集计算，不会让 L2-06 的声明失配，也不会重复计数。
> **说明**：**不计入覆盖率**：本课还提到了 `src/llama-hparams.h`（`n_expert` / `n_expert_groups` / `n_group_used` / `n_expert_used()` 的声明处，属 L2-02）、`src/llama-arch.h`（`enum llm_tensor` 里的 `LLM_TENSOR_FFN_GATE_INP` / `_GATE_EXPS` / `_UP_EXPS` / `_DOWN_EXPS` / `_EXP_PROBS_B`）与 `ggml/include/ggml.h`（算子声明，属 L1-02）。这三处只作参数名与算子名的核对，**已在 lesson.spec.py 里通过 grep 确认，但未逐字引用**，故不计入本课覆盖。
> **说明**：**分组依据**：24 个文件来自 `tools/plan_matrix.py` 的静态扫描规则 `build_moe_ffn|ffn_gate_exps|ffn_gate_inp`（见 `tools/plan_matrix.py` 的 `MODEL_MARKERS`）。本课第四节逐个读并如实记录了 6 处名实不符。
> **说明**：**命名事实**（已 grep 确认，可复核）：`build_moe_ffn` 里 top-k 用的是 `ggml_argsort_top_k`（第 2089 / 2096 / 2109 行），**不是** `ggml_top_k`；`ggml_top_k` 在整个 `src/llama-graph.cpp` 里出现 0 次。专家矩阵乘用的是 `ggml_mul_mat_id`（经 `build_lora_mm_id`，第 1550 行）。

## 场景（10 幕）

1. **24 个文件，共用同一张子图** — MoE 家族的全部差异都在"怎么选专家"，而不在"专家怎么算"—— 后者被抽成了一个原语。
2. **★ ffn_gate_inp：一次打分，就是一次 ggml_mul_mat** — 路由不是特殊算子：它就是把隐状态乘上 [n_embd, n_expert] 的打分矩阵。
3. **门控函数：softmax / sigmoid / sqrt(softplus)** — 打分之后要变成概率。选哪个函数是模型架构决定的，不是引擎决定的。
4. **分组路由：n_expert_groups / n_group_used** — DeepSeek-V3 式的"先选组、再选专家"：把 n_expert 个专家切成若干组，只在被选中的组里做 top-k。
5. **★ top-k 选择的真实函数名是 ggml_argsort_top_k** — 源码里没有用 ggml_top_k —— 用的是"先 argsort 再 view"的那个版本。
6. **专家权重：get_rows → 归一化 → 缩放** — 拿到 weights 之后有三段可选的加工，各由一个 bool / float 开关控制。
7. **★ 子图：top-k 路由 + 专家并行** — 把这一课要能默画出来的子图画出来 —— 每个名字都是真实存在的 ggml 算子。
8. **激活、down 投影、以及用 add 做的并行归并** — n_expert_used 这一维在最后被 view + add 展平回 [n_embd, n_tokens]。
9. **24 个文件里，有 7 个需要点名** — 分组依据是"图构建代码命中了 MoE 图原语"。命中了不等于"这个文件就是 MoE"。
10. **压成一张表 + 一道练习** — 会画这张子图，L2-13 里所有 MoE 变体都只是它的参数组合。

## 核心结论

### MoE 的"稀疏"只在张量维度上

`build_moe_ffn` 的全部输出就是一个 `[n_embd, n_tokens]` 的稠密张量，与 `build_ffn` 的返回值形状完全一致，直接加回残差。"稀疏"只活在图内部：专家权重张量比稠密 FFN 多一维（`n_expert`），而计算时只沿 `n_expert_used` 这一维展开。

### ★ 真实算子序列（无专用 MoE 算子）

```text
ggml_mul_mat(ffn_gate_inp, cur)              -> logits   [n_expert, n_tokens]
ggml_soft_max | ggml_sigmoid | sqrt+softplus -> probs
ggml_add(probs, exp_probs_b)                 -> selection_probs   (可选)
ggml_argsort_top_k(selection_probs, k)       -> selected_experts  [k, n_tokens]
ggml_get_rows(probs3, selected_experts)      -> weights [1, k, n_tokens]
ggml_mul_mat_id(up_exps,   cur, sel)         -> up    [n_ff, k, n_tokens]
ggml_mul_mat_id(gate_exps, cur, sel)         -> gate  [n_ff, k, n_tokens]
ggml_swiglu_split(gate, up) | ggml_silu(up)  -> act   [n_ff, k, n_tokens]
ggml_mul_mat_id(down_exps, act, sel)         -> experts [n_embd, k, n_tokens]
ggml_mul(experts, weights)                   -> 加权
ggml_view_2d x k ; ggml_add x (k-1)          -> moe_out [n_embd, n_tokens]
```

**没有一个 `ggml_moe_*` 专用算子。** 这张表可以直接和 L2-06 讲的 `build_ffn` （`ggml_mul_mat` x3 + 一个激活）逐行对照 —— 多的只有"top-k 选择"与"按索引收集/乘"。

### ★ 24 个文件像，是因为差异都在参数上

家族的差异全部落在 `build_moe_ffn` 的 24 个参数上：有没有 `gate_exps`、`gating_op` 取哪个枚举、`norm_w` / `w_scale` 传什么、有没有 `gate_up_exps`、以及 `hparams.n_expert_groups` 是否大于 1。**没有谁另写一张图。**

这也是 L1-02 "op 的身份由 `enum ggml_op` 决定"的直接推论：算子身份与模型架构解耦，模型只决定算子的**组合方式与参数**。

### 名实不符必须如实写

24 个受检文件里：`deci.cpp` **完全不是 MoE**（只有一处空指针判断）；`bert.cpp` 是 MoE 但无 gate 专家；`deepseek2ocr.cpp` / `ernie4-5.cpp` / `granite-moe.cpp` / `lfm2moe.cpp` **只有加载面**；`ernie4-5-moe.cpp` 反过来只构图不建张量。

教训：`src/models/` 里**一个文件 ≠ 一个模型** —— 张量加载与图构建可以拆在不同的 TU。

## 验收点

- [x] 保真门禁：38 处引用 —— 38 个引用块 / 79 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 25 项，无空课、无幻影；全局覆盖 251/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 78 文件 14 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
