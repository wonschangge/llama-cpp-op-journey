<!-- llama-coverage
src/models/llada-moe.cpp
src/models/llama.cpp
src/models/llama4.cpp
src/models/maple.cpp
src/models/mellum.cpp
src/models/mimo2.cpp
src/models/minicpm.cpp
src/models/minimax-01.cpp
src/models/minimax-m2.cpp
src/models/minimax-m3.cpp
src/models/mistral3.cpp
src/models/nemotron-h-moe.cpp
src/models/nomic-bert-moe.cpp
src/models/olmoe.cpp
src/models/openai-moe.cpp
src/models/phi3.cpp
src/models/phimoe.cpp
src/models/qwen2moe.cpp
src/models/qwen3moe.cpp
src/models/qwen3vlmoe.cpp
src/models/refact.cpp
src/models/rnd1.cpp
src/models/smallthinker.cpp
src/models/step35.cpp
src/llama-graph.cpp
src/llama-graph.h
src/models/models.h
src/llama-hparams.h
src/llama-model.cpp
-->

# L2-13 · 模型家族（四）：稀疏专家 MoE（下） — 源文件

**一句话**：MoE 在 llama.cpp 里只有**一条骨架** —— `llm_graph_context::build_moe_ffn()`。24 个模型文件做的事不是"实现 MoE"，而是**给这条骨架挑参数**：选几个专家、概率用什么函数、权重归不归一化、路由 logits 是自己算还是外面算。

L2-12 讲的是这条骨架的**基础序列**（logits -> top-k -> `mul_mat_id` -> 加权求和）；本课是它的下篇，只问一件事：**同样的骨架，这 24 个文件各自选了哪些开关**，以及哪些开关根本不在参数表里（共享专家、分组路由）。

本课覆盖的 24 个文件来自 `python3 tools/plan_matrix.py --files` 中 `L2-13` 的分配（静态扫描把 `src/models/*.cpp` 里命中 `build_moe_ffn|ffn_gate_exps|ffn_gate_inp` 的文件排序后对半分，本课是后半）。**分组依据是命中图原语，不是对文件内容的断言** —— 实测结果：其中 4 个文件全篇没有 `build_moe_ffn`，本课会逐个说明。

---

## 一、分组方法与本课的 24 个文件

本课覆盖的文件清单**不靠人工挑选**，而是由计划脚本给出：

```text
python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L2-13"{print $2}'
```

`tools/plan_model.py` 把 `src/models/*.cpp` 按图原语正则 `build_moe_ffn|ffn_gate_exps|ffn_gate_inp` 分到 `moe` 组，再排序对半分给 `L2-12` 与 `L2-13`。**分组依据是命中图原语，不是内容判定。** 实测三个后续事实：

| 事实 | 命令 | 结果 |
|---|---|---|
| 有 4 个文件全篇没有 MoE 调用 | `grep -c "build_moe_ffn(" src/models/*.cpp` | minicpm / nomic-bert-moe / phimoe / refact 各 0 次 |
| 有 3 个文件借别人的图 | `grep -n "using graph" src/models/models.h` | 341 / 663 / 1739 三行 |
| 全部 24 个文件都没写分组路由 | `grep -c "n_expert_groups" <24 文件>` | 全为 0 |

这三点分别在第 7 幕、第 3 幕与本节的表格里展开。

## 二、★ build_moe_ffn 的两条声明：哪些参数控制专家选择

`build_moe_ffn` 有**两条重载**：一条不带偏置张量（下面这段，其余参数带默认值），一条在 `gate_inp` / `up_exps` / `gate_exps` / `down_exps` 后面各插一个 `_b` 偏置。本课 24 个文件里只有 `openai-moe.cpp:132` 用了带偏置的那条，其余都走这条。

按"是否改变专家选择"把参数分成三组：

| 组 | 参数 | 依据 |
|---|---|---|
| **决定选谁** | `n_expert` `n_expert_used` `gating_op` `exp_probs_b` `probs_in` `selected_experts_in` | 2084/2120、2109、2040-2059、2066、2024-2032、2107-2111 |
| **只改加权** | `norm_w` `w_scale` | 2134-2148、2149-2151 |
| **不改选择** | `gate_inp` `up_exps` `gate_exps` `down_exps`（权重）、`type_op`（专家内部激活，2221）、`il`（`cb` 命名） | 权重与激活，不是路由 |

还有两个"选谁"的开关**不是参数**：`hparams.n_expert_groups`（2083）与 `hparams.n_group_used`（2096）。

<!-- src: src/llama-graph.h -->
```c
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr) const;
```

## 三、函数体逐段：从 logits 到 weights

第 2 幕引用的行号都在这段实现里。三段结构：

1. **算 logits 或直接用 probs_in**（2024-2032）：`probs_in == nullptr` 时才调用 `build_lora_mm(gate_inp, cur)`。
2. **logits 变概率**（2039-2068）：`gating_op` 的四分支 switch；随后 `exp_probs_b` **只加到 `selection_probs`**，源码注释写明"leave probs unbiased as it's later used to get expert weights"。
3. **top-k 与加权**（2106-2152）：`ggml_argsort_top_k(..., n_expert_used)` 选出专家，`ggml_get_rows` 取权重，然后才是可选的 `norm_w` 与 `w_scale`。

注意 2020 / 2072 / 2076 / 2114 / 2228 这几处 `arch == LLM_ARCH_*` 特判：它们是**架构属性**，模型文件同样没有选择权。

<!-- src: src/llama-graph.cpp -->
```c
    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }
```

## 四、★ 分组路由：开关在 GGUF，不在代码

DeepSeek V3 的 device-limited routing 在 llama.cpp 里的实现是 2081-2103 这一段，入口条件是 `hparams.n_expert_groups > 1`。三个 hparams 字段的来源：

<!-- src: src/llama-graph.cpp -->
```c
    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }
```

## 五、★ 共享专家：三种接法

共享专家（shared expert）是 DeepSeekMoE 提出的结构：**一部分专家对所有 token 都生效**。在 llama.cpp 里它**不是 `build_moe_ffn` 的参数**，而是调用点外的一条支路。本课 24 个文件里有 9 个声明了 `ffn_*_shexp` 张量，接法分三类：

| 接法 | 文件（证据行） | 形状 |
|---|---|---|
| 自带门控 | `qwen2moe.cpp:54,147-165` | 额外一个 `[n_embd]` 的 `ffn_gate_inp_shexp`，sigmoid 门乘在输出上 |
| 直接相加 | `llama4.cpp:232-240`、`step35.cpp:327-335`、`minimax-m3.cpp:575-581` | 一次普通 `build_ffn`，结果 `ggml_add` 进 `moe_out` |
| 只声明不接线 | `llama.cpp:85-88`、`minicpm.cpp:77-81`、`mistral3.cpp:80-83`、`refact.cpp:66-70` | `create_tensor` 建了张量，本文件的图里没有对应支路 |

**一个容易踩的坑**：`ffn_up_exps_s` / `ffn_gate_exps_s` / `ffn_down_exps_s` 不是共享专家。结尾的 `_s` 是 `build_lora_mm_id(w, cur, ids, w_s)` 的**每专家缩放**（`llama-graph.h:1059-1063`），`llama.cpp:215-217` 等 5 个调用点传的正是这三个。

<!-- src: src/models/qwen2moe.cpp -->
```c
        // FFN shared expert
        {
            ggml_tensor * cur_gate_inp = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

            // sigmoid
            ggml_tensor * cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
            cb(cur_gate, "ffn_shexp_gate", il);

            ggml_tensor * cur_ffn = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur_ffn, "ffn_shexp", il);

            ggml_tensor * ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
            cb(ffn_shexp_out, "ffn_shexp_out", il);

            moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
            cb(moe_out, "ffn_out", il);

            cur = moe_out;
```

## 六 · `llada-moe.cpp`（164 行，引用 126-137）

全篇唯一一处 build_moe_ffn；norm_w 写死 false，exp_probs_b 传 nullptr。文件里没有 shexp，也没有 n_expert_groups。

<!-- src: src/models/llada-moe.cpp -->
```c
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, false,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(cur, "ffn_moe_out", il);
```

## 六 · `llama.cpp`（251 行，引用 84-88）

`llama` 架构在 `hparams.n_ff_shexp > 0` 时声明三张共享专家张量。但它的图（196-217）传给 `build_moe_ffn` 的是 `ffn_*_exps_s`（每专家缩放），**不是 shexp** —— 全文件 `shexp` 出现 4 次，全在这 5 行的声明里。

<!-- src: src/models/llama.cpp -->
```c
            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
```

## 六 · `llama4.cpp`（273 行，引用 230-241）

共享专家走 `build_ffn` + `ggml_add`，没有自己的门控；MoE 侧 `gating_op` 是 `SIGMOID`、`norm_w` 是 `false`。llama4 在 `build_moe_ffn` 体内还有两处 arch 特判（`llama-graph.cpp:2020` 的 `weight_before_ffn`、2072-2074 的 `selection_probs = logits`）。

<!-- src: src/models/llama4.cpp -->
```c

            // Shared experts
            ggml_tensor * shexp_out = build_ffn(ffn_inp_normed,
                model.layers[il].ffn_up_shexp,   NULL, NULL,
                model.layers[il].ffn_gate_shexp, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(shexp_out, "ffn_moe_shexp", il);

            cur = ggml_add(ctx0, moe_out, shexp_out);
            cb(cur, "ffn_moe_out_merged", il);
```

## 六 · `maple.cpp`（151 行，引用 121-131）

唯一把 `w_scale` 写成字面量 `1.0f` 的调用点（其余 20 处都读 `hparams.expert_weights_scale`）。MAPLE 在 `build_moe_ffn` 里另有一处 arch 特判：`llama-graph.cpp:2228` 的 swiglu clamp 分支。

<!-- src: src/models/maple.cpp -->
```c
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                1.0f,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
```

## 六 · `mellum.cpp`（220 行，引用 172-188）

用 18 参数的重载：除权重外还传 `ffn_up_exps_s` / `ffn_gate_exps_s` / `ffn_down_exps_s` 三个每专家缩放张量。

<!-- src: src/models/mellum.cpp -->
```c
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);
```

## 六 · `mimo2.cpp`（397 行，引用 206-220）

传 `ffn_exp_probs_b`（选择偏置），`gating_op` 是 `SIGMOID`。同文件另有 `graph_mtp`（`models.h:2555`），MTP 分支里还要再走一次 FFN。

<!-- src: src/models/mimo2.cpp -->
```c
        } else {
            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
```

## 六 · `minicpm.cpp`（90 行，引用 70-81）

声明了 MoE 与共享专家张量，但 `build_arch_graph` 用的是 `llama_model_granite::graph`（`models.h:1739`）—— 全文件 0 处 `build_moe_ffn`。

<!-- src: src/models/minicpm.cpp -->
```c
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
            }
```

## 六 · `minimax-01.cpp`（485 行，引用 442-453）

`exp_probs_b` + `SOFTMAX` + `norm_w=true`；FFN 之后还有 `f_residual_scale`（455 行）。

<!-- src: src/models/minimax-01.cpp -->
```c
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(cur, "ffn_moe_out", il);
```

## 六 · `minimax-m2.cpp`（169 行，引用 130-140）

`gating_op` 不写死，转成 `hparams.expert_gating_func` —— 概率函数由 GGUF 决定。

<!-- src: src/models/minimax-m2.cpp -->
```c
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
```

## 六 · `minimax-m3.cpp`（609 行，引用 559-582）

路由专家用 `LLM_FFN_SWIGLU_OAI_MOE` 激活；`norm_w` 与 `gating_op` 都读 hparams。共享专家宽度是 `n_ff_exp * n_expert_shared`（79-81），它是本课 24 个文件里**唯一**读 `n_expert_shared` 的。

<!-- src: src/models/minimax-m3.cpp -->
```c
        } else {
            // routed experts (swigluoai MoE)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (swigluoai)
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

```

## 六 · `mistral3.cpp`（236 行，引用 193-207）

18 参数重载（每专家缩放）。共享专家张量在 80-83 声明，但 MoE 分支（186-209）里没有接 —— 全文件 `shexp` 只出现在声明处。

<!-- src: src/models/mistral3.cpp -->
```c
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
```

## 六 · `nemotron-h-moe.cpp`（165 行，引用 115-130）

`gate_exps` 传 `nullptr`（源码注释 "no gate"），激活换成 `LLM_FFN_RELU_SQR`；第 13 个参数传 `router_logits`，路由 logits 在 100 行已用 `build_lora_mm` 算好。

<!-- src: src/models/nemotron-h-moe.cpp -->
```c
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                nullptr, // no gate
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_RELU_SQR, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                il,
                router_logits, nullptr,
                layer.ffn_up_exps_s,
                nullptr, // no gate
                layer.ffn_down_exps_s);
```

## 六 · `nomic-bert-moe.cpp`（57 行，引用 37-46）

用 `hparams.moe_every_n_layers` 决定哪些层是 MoE（37 行），图借 `llama_model_bert::graph`（`models.h:341`）。

<!-- src: src/models/nomic-bert-moe.cpp -->
```c
        if (hparams.moe_every_n_layers > 0 && i % hparams.moe_every_n_layers == 1) {
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff,   n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff,   n_embd, n_expert}, 0);
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert}, 0);
        } else {
            layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);
        }
```

## 六 · `olmoe.cpp`（174 行，引用 136-146）

基础组合里最"素"的一个：`norm_w=false`、无偏置、`SOFTMAX`、无 shexp。

<!-- src: src/models/olmoe.cpp -->
```c
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, false,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
```

## 六 · `openai-moe.cpp`（176 行，引用 131-142）

**唯一**使用 17 参数"带偏置"重载的调用点（`ffn_gate_inp_b` 等 4 组偏置）；`gating_op` 是 `SOFTMAX_WEIGHT`，即 softmax 作用在**选中的权重**上（`llama-graph.cpp:2127-2132`）。

<!-- src: src/models/openai-moe.cpp -->
```c
        // MoE branch
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il);
```

## 六 · `phi3.cpp`（197 行，引用 143-164）

稠密与 MoE 共用同一张图，靠 `ffn_gate_inp == nullptr` 分流；`build_arch_graph` 按 `swa_type` 在 `graph<true>` / `graph<false>` 之间选（59-65）。PhiMoE 借的就是这张图。

<!-- src: src/models/phi3.cpp -->
```c
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
```

## 六 · `phimoe.cpp`（56 行，引用 38-41）

只有 hparams、张量声明与选图三段：38-41 声明四张 MoE 张量（这就是静态扫描把它分进 MoE 组的原因），48-54 按 `swa_type` 选 `graph<iswa>`。**0 处 `build_moe_ffn`**，MoE 行为完全由 `phi3.cpp:153` 决定。

<!-- src: src/models/phimoe.cpp -->
```c
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert},         0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);
```

## 六 · `qwen2moe.cpp`（195 行，引用 145-166）

共享专家有自己的门控：`ffn_gate_inp_shexp`（54 行，形状 `[n_embd]`）出标量，经 sigmoid 门（151）乘到共享专家输出上，再在 165 行加回 `moe_out`。24 个文件里只有它声明 `LLM_TENSOR_FFN_GATE_INP_SHEXP`。

<!-- src: src/models/qwen2moe.cpp -->
```c
        // FFN shared expert
        {
            ggml_tensor * cur_gate_inp = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

            // sigmoid
            ggml_tensor * cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
            cb(cur_gate, "ffn_shexp_gate", il);

            ggml_tensor * cur_ffn = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur_ffn, "ffn_shexp", il);

            ggml_tensor * ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
            cb(ffn_shexp_out, "ffn_shexp_out", il);

            moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
            cb(moe_out, "ffn_out", il);
```

## 六 · `qwen3moe.cpp`（180 行，引用 136-152）

18 参数重载 + `norm_w=true`；无 shexp、无偏置。

<!-- src: src/models/qwen3moe.cpp -->
```c
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);
```

## 六 · `qwen3vlmoe.cpp`（191 行，引用 144-156）

MoE 侧参数与 qwen3moe 逐字相同（SILU / true / hparams.expert_weights_scale / SOFTMAX）。差别不在 MoE：这个文件里 `clip_` / `vision` / `mmproj` 命中 0 次，视觉塔在 `tools/mtmd` 里（不计入本课覆盖率，见 L2-10）。

<!-- src: src/models/qwen3vlmoe.cpp -->
```c
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
        cb(moe_out, "ffn_moe_out", il);
```

## 六 · `refact.cpp`（161 行，引用 59-70）

按 `n_expert == 0`（50 行）分别声明稠密与 MoE 张量，MoE 分支里连共享专家张量都建了；但它的 `graph` 的 FFN 只有一条 `build_ffn`（128-133），全文件 0 处 `build_moe_ffn`。

<!-- src: src/models/refact.cpp -->
```c
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
            }
```

## 六 · `rnd1.cpp`（178 行，引用 138-150）

基础组合：`nullptr` 偏置、`norm_w=true`、`SOFTMAX`、无 shexp。

<!-- src: src/models/rnd1.cpp -->
```c
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
        cb(moe_out, "ffn_moe_out", il);
```

## 六 · `smallthinker.cpp`（189 行，引用 148-159）

`gate_inp` 传 `nullptr`，第 14 个参数传 `probs`（109 行算好）；激活是 `LLM_FFN_RELU`。

<!-- src: src/models/smallthinker.cpp -->
```c
        ggml_tensor * ffn_out =
            build_moe_ffn(cur,
                    nullptr,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_RELU, true,
                    hparams.expert_weights_scale,
                    static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
                    il, probs);
```

## 六 · `step35.cpp`（561 行，引用 512-533）

主干与 MTP 各一处 MoE 调用（313 / 512），参数全部读 hparams；共享专家用 `build_ffn` + `ggml_add`（327-335 / 525-533）。

<!-- src: src/models/step35.cpp -->
```c
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "mtp_ffn_moe_out", il);

        ggml_tensor * sh_out = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(sh_out, "mtp_ffn_shared_out", il);

        cur = ggml_add(ctx0, moe_out, sh_out);
```

## 七、借图：models.h 里的 `using graph = ...`

`llm_graph_context` 是抽象基类；每个架构要么在 `models.h` 里声明自己的 `struct graph`，要么用 `using` 别名借别人的。本课 24 个文件里 21 个是前者，3 个是后者。被借的三张图分别在 `bert.cpp` / `granite.cpp`（L2-12 覆盖）与 `phi3.cpp`（本课覆盖）。

所以"读一个模型文件"不足以保证理解它的图：`phimoe.cpp` 只有 55 行，它的 MoE 行为完全由 `phi3.cpp:153` 决定。

<!-- src: src/models/models.h -->
```c
struct llama_model_phimoe : public llama_model_base {
    llama_model_phimoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    using graph = llama_model_phi3::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

## 八、hparams 里的专家字段与它们的 GGUF 来源

`build_moe_ffn` 体内引用了 5 个 hparams 字段，它们都不是函数参数：

| 字段 | 声明 | 从 GGUF 读入 | 在 build_moe_ffn 里的行 |
|---|---|---|---|
| `n_expert_groups` | `llama-hparams.h:114` | `llama-model.cpp:1266` | 2083、2084、2087 |
| `n_group_used` | `llama-hparams.h:115` | `llama-model.cpp:1267` | 2096 |
| `n_group_experts` | `llama-hparams.h:116` | GROVEMOE 自己在 `grovemoe.cpp:7` 读 | 2117 |
| `expert_weights_scale` | `llama-hparams.h:124` | 各模型文件按架构读 | 2149 |
| `expert_weights_norm` | `llama-hparams.h:125` | 各模型文件按架构读 | 2134 |

`llama-model.cpp:1294-1305` 是一组硬断言，它把合法组合钉死：`n_expert_groups < n_expert`；一旦 `n_expert_groups > 1`，就必须`n_expert % n_expert_groups == 0` 且 `0 < n_group_used < n_expert_groups`。

<!-- src: src/llama-hparams.h -->
```c
    uint32_t n_ff_shexp         = 0;
    uint32_t n_ff_chexp         = 0;
    uint32_t n_expert_shared    = 0;
    uint32_t n_norm_groups      = 0;
    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_group_experts    = 0;

    // MLA + SWA (i.e. dots3note)
    uint32_t n_lora_kv_swa           = 0;
    uint32_t n_embd_head_k_mla_swa   = 0;
    uint32_t n_embd_head_v_mla_swa   = 0;

    float    expert_group_scale   = 0.05f;
    float    expert_weights_scale = 0.0f;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    uint32_t moe_every_n_layers   = 0;
```

## 九、GGUF 元数据到 hparams：两个字段的来路

分组路由的两个字段走的是通用读取路径，与架构无关 —— 任何架构只要 GGUF 里有`*.expert_group_count` / `*.expert_group_used_count` 就会被读进来。

<!-- src: src/llama-model.cpp -->
```c
    ml.get_key_or_arr(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    if (arch == LLM_ARCH_HUNYUAN_VL || arch == LLM_ARCH_HUNYUAN_DENSE) {
        if (hparams.n_expert <= 1) {
            hparams.n_expert = 0;
            std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
        }
    }

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH,  hparams.n_embd);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd_out_impl);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);

        GGML_ASSERT(hparams.posnet.n_layer   <= hparams.n_layer_all);
        GGML_ASSERT(hparams.convnext.n_layer <= hparams.n_layer_all);
    }

    // models may route a different number of experts per layer, so validate the maximum
    uint32_t n_expert_used_max = hparams.n_expert_used_max();

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(n_expert_used_max <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(n_expert_used_max > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(n_expert_used_max == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
```

## 十、`probs_in` 的短路判据

第 6 幕引用的"传了 `probs_in` 就不再碰 `gate_inp`"就来自这两行。顺带一提：`gate_inp_b`（路由偏置）在 `probs_in` 路径下**仍然会加**（2034-2037）。

<!-- src: src/llama-graph.cpp -->
```c
    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }
```

---

## 说明

- 本课计划内的 **24 个文件**全部来自 `python3 tools/plan_matrix.py --files` 中 `L2-13` 的分配，逐字引用并计入覆盖率。
- `src/llama-graph.cpp`（`build_moe_ffn` 的实现）与 `src/llama-graph.h`（它的两条声明）按计划属于 `L2-06`；本课把"参数如何控制专家选择"作为核心引用对象，**计入覆盖率声明**，但这不改变计划的文件归属。
- `src/models/models.h` 按计划属于 `L2-10` / `L2-15`；本课只引用它的三处 `using graph = ...` 别名（341 / 663 / 1739）来证明"借图"。
- `src/llama-hparams.h`（`L2-02`）与 `src/llama-model.cpp`（`L2-05`）本课各引用一段，用来定位分组路由那两个字段的来路。
- 第 5 幕与第 9 幕的统计数字（21 处调用点、各取值的出现次数、`n_expert_groups` 0 次）由脚本对这 24 个文件静态扫描得出，不是目测。
