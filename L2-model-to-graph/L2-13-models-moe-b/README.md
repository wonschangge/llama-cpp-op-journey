# L2-13 · 模型家族（四）：稀疏专家 MoE（下） — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-12`（模型家族（三）稀疏专家 MoE（上）：`build_moe_ffn` 的基础序列）

## 学习目标

看完这一课，你应该能：

1. 说出 `build_moe_ffn` 的参数里哪些控制**专家选择**（对应验收点），哪些只控制加权；
2. 解释"分组路由"为什么在这 24 个文件里一次都搜不到（它由 GGUF 元数据决定）；
3. 说出共享专家的三种接法，并指出 `ffn_up_exps_s` 与 `ffn_gate_shexp` 的区别；
4. 判断一个 `src/models/*.cpp` 是不是"借图"（`using graph = ...`），以及这对读代码意味着什么。

## 覆盖的源文件（29 个）

| 文件 | 行数 |
|---|---|
| `src/models/llada-moe.cpp` | 164 |
| `src/models/llama.cpp` | 251 |
| `src/models/llama4.cpp` | 273 |
| `src/models/maple.cpp` | 151 |
| `src/models/mellum.cpp` | 220 |
| `src/models/mimo2.cpp` | 397 |
| `src/models/minicpm.cpp` | 90 |
| `src/models/minimax-01.cpp` | 485 |
| `src/models/minimax-m2.cpp` | 169 |
| `src/models/minimax-m3.cpp` | 609 |
| `src/models/mistral3.cpp` | 236 |
| `src/models/nemotron-h-moe.cpp` | 165 |
| `src/models/nomic-bert-moe.cpp` | 57 |
| `src/models/olmoe.cpp` | 174 |
| `src/models/openai-moe.cpp` | 176 |
| `src/models/phi3.cpp` | 197 |
| `src/models/phimoe.cpp` | 56 |
| `src/models/qwen2moe.cpp` | 195 |
| `src/models/qwen3moe.cpp` | 180 |
| `src/models/qwen3vlmoe.cpp` | 191 |
| `src/models/refact.cpp` | 161 |
| `src/models/rnd1.cpp` | 178 |
| `src/models/smallthinker.cpp` | 189 |
| `src/models/step35.cpp` | 561 |
| `src/llama-graph.cpp` | 3916 |
| `src/llama-graph.h` | 1391 |
| `src/models/models.h` | 2663 |
| `src/llama-hparams.h` | 513 |
| `src/llama-model.cpp` | 3360 |

> **说明**：本课计划内的 **24 个文件**全部来自 `python3 tools/plan_matrix.py --files` 中 `L2-13` 的分配，逐字引用并计入覆盖率。
> **说明**：`src/llama-graph.cpp`（`build_moe_ffn` 的实现）与 `src/llama-graph.h`（它的两条声明）按计划属于 `L2-06`；本课把"参数如何控制专家选择"作为核心引用对象，**计入覆盖率声明**，但这不改变计划的文件归属。
> **说明**：`src/models/models.h` 按计划属于 `L2-10` / `L2-15`；本课只引用它的三处 `using graph = ...` 别名（341 / 663 / 1739）来证明"借图"。
> **说明**：`src/llama-hparams.h`（`L2-02`）与 `src/llama-model.cpp`（`L2-05`）本课各引用一段，用来定位分组路由那两个字段的来路。
> **说明**：第 5 幕与第 9 幕的统计数字（21 处调用点、各取值的出现次数、`n_expert_groups` 0 次）由脚本对这 24 个文件静态扫描得出，不是目测。

## 场景（10 幕）

1. **24 个 MoE 文件，共用 一条骨架** — L2-12 讲这条骨架的基础序列；本课讲它的参数如何被 24 个模型文件选成不同的路由策略。
2. **★ 控制"专家选择"的是这 8 个参数** — 其余参数只改权重布局与专家内部激活，不改变"选哪些专家"。
3. **★ 分组路由的开关是 hparams，不是调用参数** — 24 个文件里 n_expert_groups / n_group_used 出现 0 次；它从 GGUF 读进 hparams，在函数体里生效。
4. **★ 共享专家不在参数表里 —— 它是 调用点外的一条支路** — qwen2moe 给它配了独立的门控；llama4 / step35 / minimax-m3 直接相加；llama / mistral3 / refact 声明了但没接。
5. **同一个函数，21 个调用点各选不同的开关** — 20 个文件共 21 处调用；下表每一格都是对这 24 个文件静态统计出来的。
6. **probs_in：路由已经在外面算好了** — 传了 probs_in，函数体就不再碰 gate_inp；smallthinker 连 gate_inp 都直接传 nullptr。
7. **借图：3 个文件自己没有 MoE 图** — phimoe.cpp（55 行）没有一行 build_moe_ffn —— 它的 MoE 行为由 phi3.cpp:153 决定。
8. **MoE 分支的入口条件也由模型文件说了算** — phi3.cpp 的图里稠密与 MoE 共用一条主干，靠 ffn_gate_inp == nullptr 分流；PhiMoE 借的就是这张图。
9. **把 24 个文件压成 一张表** — 每格都写了证据行号；共享专家与分组路由是两条互不相干的开关。
10. **一句话：MoE 的可调参数不在算子，在 路由策略** — 共享专家、分组上限、top-k 归一化都是开关；模型文件只是选不同的组合。

## 核心结论

### ★ 控制专家选择的 8 个参数

`llama-graph.h:1113-1132` 的签名里，真正改变"选哪些专家 / 权重怎么算"的是：

| 参数 | 生效行（llama-graph.cpp） | 作用 |
|---|---|---|
| `n_expert` | 2084 / 2120 | 候选专家总数 |
| `n_expert_used` | 2109 | top-k 的 k |
| `gating_op` | 2040-2059 | logits 变概率的函数 |
| `exp_probs_b` | 2066 | 选择偏置（只影响选谁） |
| `norm_w` | 2134-2148 | top-k 权重归一化 |
| `w_scale` | 2149-2151 | 路由权重缩放 |
| `probs_in` | 2024-2032 | 绕开 `gate_inp` |
| `selected_experts_in` | 2107-2111 | 绕开 top-k |

`type_op`（专家内部激活）、`il`、以及四个权重张量**不控制选择**。

### ★ 两个开关不在这张参数表里

**分组路由**：入口是 `hparams.n_expert_groups > 1`（`llama-graph.cpp:2083`），字段从 GGUF 读入（`llama-model.cpp:1266-1267`）。本课 24 个文件里这两个名字出现 **0 次**。

**共享专家**：调用点外的第二条支路。9 / 24 个文件声明了 `ffn_*_shexp` 张量，但接法有三种（自带门控 / 直接相加 / 只声明不接线），且没有一种经过 `build_moe_ffn` 的参数。

### 一个辨别技巧：`_exps_s` 不是 shexp

`ffn_up_exps_s` / `ffn_gate_exps_s` / `ffn_down_exps_s` 结尾的 `_s` 是 `build_lora_mm_id(w, cur, ids, w_s)` 的**每专家缩放**（`llama-graph.h:1059-1063`），与共享专家无关。判断一个文件是否真的用了共享专家，要看 `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` 这三个名字，并且要在**图里**找到使用它的支路 —— `llama.cpp` / `mistral3.cpp` / `refact.cpp` 都是声明了却没接线。

## 验收点

- [x] 保真门禁：42 处引用 —— 42 个引用块 / 45 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 29 项，无空课、无幻影；全局覆盖 251/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 78 文件 14 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
