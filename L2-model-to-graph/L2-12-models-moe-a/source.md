<!-- llama-coverage
src/llama-graph.cpp
src/models/deci.cpp
src/models/dbrx.cpp
src/models/grovemoe.cpp
src/models/afmoe.cpp
src/models/arctic.cpp
src/models/bailingmoe.cpp
src/models/bailingmoe2.cpp
src/models/bert.cpp
src/models/cohere2moe.cpp
src/models/deepseek.cpp
src/models/deepseek2ocr.cpp
src/models/dots1.cpp
src/models/ernie4-5-moe.cpp
src/models/ernie4-5.cpp
src/models/exaone-moe.cpp
src/models/glm4-moe.cpp
src/models/granite-moe.cpp
src/models/granite-swa.cpp
src/models/granite.cpp
src/models/grok.cpp
src/models/hunyuan-moe.cpp
src/models/hy-v3.cpp
src/models/laguna.cpp
src/models/lfm2moe.cpp
-->

# L2-12 · 模型家族（三）：稀疏专家 MoE（上） — 源文件

**一句话**：所谓"稀疏专家"，**稀疏只体现在专家张量多出来的那一维上**。图上的算子序列与稠密 FFN 几乎一样 —— 一次打分、一次 top-k 选择、k 条并行的"乘权重矩阵"支路、一次求和 —— 只不过那 k 条支路走的不是同一个权重，而是按 `selected_experts` **按索引收集**出来的不同专家。正因为如此，24 个 MoE 模型的图代码才能长得那么像。

本课覆盖 24 个源文件，全部在 `src/models/` 下。它们**不是按内容挑的**，而是静态扫描的结果：图构建代码里命中了 `build_moe_ffn` / `ffn_gate_exps` / `ffn_gate_inp` 之一。**分组依据是"命中了这些图原语"，不是"这个文件就是 MoE"** —— 实测结果见下方第四节，24 个里有 6 个名实不符，本课如实列出。

---

## 一、本课的分组方法（先读这一节）

本课覆盖的 24 个文件**不是按内容选的**，而是**静态扫描**的结果：凡是 `src/models/` 下 图构建代码里出现了 `build_moe_ffn` / `ffn_gate_exps` / `ffn_gate_inp` 之一的 `.cpp`，就被分到"稀疏专家 MoE（上）"这一课。

```text
python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L2-12"{print $2}'
-> 24 个文件，全部在 src/models/ 下
```

**分组依据是"命中了这些图原语"，不是"这个文件就是 MoE"。** 逐个读完之后，24 个里有 6 个名实不符（见第四节），本课如实列出，不为了数字好看而虚报。

而这 24 个文件之所以能被归成"一个家族"，是因为它们**都只是把同一个图原语接在残差上**。那个原语就是下面这段签名 —— 它在 `src/llama-graph.cpp`，属于 L2-06 的计算图骨架。**本课显式声明覆盖 `src/llama-graph.cpp`**：覆盖度是并集，这不影响 L2-06 的声明。

<!-- src: src/llama-graph.cpp -->
```c
ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
```

## 二、★ build_moe_ffn 的真实算子序列

下表是逐行读 `src/llama-graph.cpp:1993-2358` 得到的**真实**序列。"门控"和"分组路由"两段是可选的，由 `gating_op` 与 `hparams.n_expert_groups` 决定。

| # | 阶段 | 真实算子 | 行号 |
|---|---|---|---|
| 1 | 打分 | `build_lora_mm(gate_inp, cur)` -> `ggml_mul_mat` | 2025 |
| 2 | 选择偏置（可选） | `ggml_add(logits, gate_inp_b)` | 2035 |
| 3 | 门控 | `ggml_soft_max` / `ggml_sigmoid` / `ggml_sqrt(ggml_softplus(..))` | 2043 / 2047 / 2055 |
| 4 | 专家选择偏置 | `ggml_add(probs, exp_probs_b)` | 2066 |
| 5 | 分组路由（可选） | `ggml_reshape_3d` -> `ggml_argsort_top_k(..,2)` -> `ggml_get_rows` -> `ggml_sum_rows` -> `ggml_argsort_top_k(..,n_group_used)` -> `ggml_get_rows` -> `ggml_fill(-INFINITY)` -> `ggml_set_rows` -> `ggml_reshape_2d` | 2087-2102 |
| 6 | top-k | `ggml_argsort_top_k(selection_probs, n_expert_used)` | 2109 |
| 7 | 收集权重 | `ggml_reshape_3d` -> `ggml_get_rows(probs, selected_experts)` | 2120 / 2123 |
| 8 | 权重 softmax（可选） | `ggml_reshape_2d` -> `ggml_soft_max` -> `ggml_reshape_3d` | 2128-2130 |
| 9 | 权重归一化（可选） | `ggml_reshape_2d` -> `ggml_sum_rows` -> `ggml_clamp` -> `ggml_div` -> `ggml_reshape_3d` | 2135-2147 |
| 10 | 权重缩放（可选） | `ggml_scale(weights, w_scale)` | 2150 |
| 11 | 输入摊平 | `ggml_reshape_3d(cur, n_embd, 1, n_tokens)` | 2157 |
| 12a | 融合 gate_up 路 | `build_lora_mm_id` -> `ggml_mul_mat_id`，再两个 `ggml_view_3d` | 2171 / 2184 / 2186 |
| 12b | 分开 gate/up 路 | `build_lora_mm_id(up_exps,..)` + `build_lora_mm_id(gate_exps,..)` | 2190 / 2203 |
| 13 | 激活 | `ggml_swiglu_split` / `ggml_silu` / `ggml_geglu_split` / `ggml_reglu_split` / `ggml_swiglu_oai` / `ggml_swiglu_clamp` | 2246 / 2249 / 2269 / 2285 / 2280 / 2229 |
| 14 | down 投影 | `build_lora_mm_id(down_exps,..)` -> `ggml_mul_mat_id` | 2304 |
| 15 | 加权 | `ggml_mul(experts, weights)` | 2321 |
| 16 | 切分 | `ggml_view_2d(experts, n_embd, n_tokens, nb[2], i*nb[1])` x k | 2337 |
| 17 | 归并 | `ggml_add` x (k-1) | 2346 |
| 18 | 单专家特例 | `ggml_cont(moe_out)` | 2353 |

**没有 `ggml_moe_*` 这样的专用算子。** `ggml_top_k` 虽然存在于 `ggml/include/ggml.h`，但在整个 `src/llama-graph.cpp` 里一次都没出现 —— 源码用的是 `ggml_argsort_top_k`。

<!-- src: src/llama-graph.cpp -->
```c
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
```

## 三、三种调用形态（代表模型）

同一个 `build_moe_ffn`，24 个模型填出三种典型形态：

```text
形态 A（最常见）：七个参数一次给全
    build_moe_ffn(cur, ffn_gate_inp, ffn_up_exps, ffn_gate_exps,
                  ffn_down_exps, nullptr, n_expert, n_expert_used, ...)
形态 B（grovemoe）：自己先算好 logits，用 probs_in 传进去，gate_inp 传 nullptr
形态 C（新式）：多加 gate_up_exps / *_s 缩放张量，并把共享专家另算一条 build_ffn
```

形态 A 的代表是 `dbrx.cpp`（本课第 10 幕引用）；形态 B 是 `grovemoe.cpp`，它甚至把 `build_lora_mm(ffn_gate_inp, cur)` 直接写在自己的图代码里；形态 C 是 `hy-v3.cpp` 与 `cohere2moe.cpp`，多出来的 `ffn_gate_up_exps` 走的是`build_moe_ffn` 的"融合路"分支（`src/llama-graph.cpp:2169`）。

<!-- src: src/models/grovemoe.cpp -->
```c
        ggml_tensor * probs = build_lora_mm(model.layers[il].ffn_gate_inp, cur);  // [n_expert, n_tokens]
        cb(probs, "ffn_moe_logits", il);

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                nullptr,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il,
                probs);
```

## 四、名实核对：24 个文件逐个点名

分组依据是"命中了 MoE 图原语"。下面逐个文件给出**逐字引用**与判定。

| # | 文件 | 判定 |
|---|---|---|
| 1 | `src/models/afmoe.cpp` | 真 MoE |
| 2 | `src/models/arctic.cpp` | 真 MoE |
| 3 | `src/models/bailingmoe.cpp` | 真 MoE |
| 4 | `src/models/bailingmoe2.cpp` | 真 MoE |
| 5 | `src/models/bert.cpp` | MoE，但无 gate 专家 |
| 6 | `src/models/cohere2moe.cpp` | 真 MoE |
| 7 | `src/models/dbrx.cpp` | 真 MoE |
| 8 | `src/models/deci.cpp` | ★ 完全不是 MoE |
| 9 | `src/models/deepseek.cpp` | 真 MoE |
| 10 | `src/models/deepseek2ocr.cpp` | 只有加载面 |
| 11 | `src/models/dots1.cpp` | 真 MoE |
| 12 | `src/models/ernie4-5-moe.cpp` | 只构图，不建张量 |
| 13 | `src/models/ernie4-5.cpp` | 只有加载面 |
| 14 | `src/models/exaone-moe.cpp` | 真 MoE |
| 15 | `src/models/glm4-moe.cpp` | 真 MoE |
| 16 | `src/models/granite-moe.cpp` | 只有加载面 |
| 17 | `src/models/granite-swa.cpp` | 真 MoE |
| 18 | `src/models/granite.cpp` | 真 MoE |
| 19 | `src/models/grok.cpp` | 真 MoE |
| 20 | `src/models/grovemoe.cpp` | 真 MoE（形态 B） |
| 21 | `src/models/hunyuan-moe.cpp` | 真 MoE |
| 22 | `src/models/hy-v3.cpp` | 真 MoE（形态 C） |
| 23 | `src/models/laguna.cpp` | 真 MoE |
| 24 | `src/models/lfm2moe.cpp` | 只有加载面 |
| 25 | `src/models/deci.cpp` | （重引·上下文） |

统计：**18 个真 MoE**（张量与构图至少有一处在文件内）、**1 个 MoE 但无 gate 专家**（`bert.cpp`）、**4 个只有加载面**（`deepseek2ocr.cpp` / `ernie4-5.cpp` / `granite-moe.cpp` / `lfm2moe.cpp`）、**1 个完全不是 MoE**（`deci.cpp`）、再加 1 个"只构图不建张量"的反例（`ernie4-5-moe.cpp`）。

<!-- src: src/models/dbrx.cpp -->
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
                il);
```

## 四.1 src/models/afmoe.cpp —— 真 MoE

张量与构图都在本文件：`create_tensor` 建 `ffn_gate_inp` / 三组 `*_exps`，图里调 `build_moe_ffn`，门控函数取自 `hparams.expert_gating_func`。

<!-- src: src/models/afmoe.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,           // norm_w (route_norm=True)
                    hparams.expert_weights_scale,          // w_scale (route_scale=2.826)
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
```

## 四.2 src/models/arctic.cpp —— 真 MoE

与 afmoe 同构；`n_ff`（不是 `n_ff_exp`）作为专家中间维，说明专家宽度可以等于稠密宽度。

<!-- src: src/models/arctic.cpp -->
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
                il);
```

## 四.3 src/models/bailingmoe.cpp —— 真 MoE

把 `build_moe_ffn` 的返回值直接赋给 `cur`；`hparams.expert_weights_norm` 与 `scale` 逐层可变。

<!-- src: src/models/bailingmoe.cpp -->
```c
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
```

## 四.4 src/models/bailingmoe2.cpp —— 真 MoE

张量创建带 `flags` 变量（量化/放置标志），构图与 bailingmoe 一致。

<!-- src: src/models/bailingmoe2.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
```

## 四.5 src/models/bert.cpp —— MoE，但无 gate 专家

`gate_exps` 位置传的是 `nullptr`，且 `gate_up` 参数完全省略 —— 只有 up/down 两组专家，激活退化成 `ggml_silu`。触发条件是 `hparams.moe_every_n_layers > 0 && il % moe_every_n_layers == 1`。

<!-- src: src/models/bert.cpp -->
```c
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    nullptr,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    hparams.n_expert, hparams.n_expert_used(),
                    LLM_FFN_GELU, false,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
```

## 四.6 src/models/cohere2moe.cpp —— 真 MoE

形态 C：传了 `gate_up_exps` 与三个 `*_s` 缩放张量。同一文件里 MTP 块（381 行起）再调一次。

<!-- src: src/models/cohere2moe.cpp -->
```c
            cur = build_moe_ffn(ffn_inp,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr, layer.ffn_gate_up_exps,
                    layer.ffn_up_exps_s,
                    layer.ffn_gate_exps_s,
                    layer.ffn_down_exps_s);
```

## 四.7 src/models/dbrx.cpp —— 真 MoE

形态 A 的最短样例，本课第 1 幕与第 10 幕引用它。

<!-- src: src/models/dbrx.cpp -->
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
                il);
```

## 四.8 src/models/deci.cpp —— ★ 完全不是 MoE

`ffn_gate_inp` 只出现在一个空指针判断里，**从未被 `create_tensor` 创建**；判断为真时走的是稠密 `build_ffn`。整个文件 grep `moe|expert` 零命中（Deci/Nemotron 是稠密模型）。

<!-- src: src/models/deci.cpp -->
```c
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
```

## 四.9 src/models/deepseek.cpp —— 真 MoE

`moe_out` 之后还与共享专家/稠密分支相加，是"路由专家 + 共享专家"写法的早期形态。

<!-- src: src/models/deepseek.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
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

## 四.10 src/models/deepseek2ocr.cpp —— 只有加载面

文件共 80 行，只做 `load_arch_hparams` + `load_arch_tensors`；`build_arch_graph` 返回的 `graph` 类定义在别处。图代码里没有 `build_moe_ffn`。

<!-- src: src/models/deepseek2ocr.cpp -->
```c
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, 0);
```

## 四.11 src/models/dots1.cpp —— 真 MoE

与 deepseek.cpp 同构（dots1 是 DeepSeek 系衍生）。

<!-- src: src/models/dots1.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
```

## 四.12 src/models/ernie4-5-moe.cpp —— 只构图，不建张量

反向的不符：本文件只有 `build_moe_ffn` 调用，专家张量由 `ernie4-5.cpp` 的 `arch == LLM_ARCH_ERNIE4_5_MOE` 分支创建。这里传了 `ffn_exp_probs_b`（第 5 个位置参数）。

<!-- src: src/models/ernie4-5-moe.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
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
```

## 四.13 src/models/ernie4-5.cpp —— 只有加载面

`create_tensor` 建了 `ffn_gate_inp` 与三组 `*_exps`（`ffn_gate_exps` 标记 `TENSOR_NOT_REQUIRED`），但本文件的图代码只有稠密 `build_ffn` 分支。

<!-- src: src/models/ernie4-5.cpp -->
```c
        if (arch == LLM_ARCH_ERNIE4_5_MOE && static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) { // MoE layers
            int n_ff_exp = hparams.n_ff_exp();

            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
```

## 四.14 src/models/exaone-moe.cpp —— 真 MoE

构图前先判断 `model.layers[il].ffn_gate_inp == nullptr` 来决定走稠密还是 MoE。

<!-- src: src/models/exaone-moe.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
```

## 四.15 src/models/glm4-moe.cpp —— 真 MoE

同一文件里有两处调用：MTP 块（235 行）与主干（390 行）。

<!-- src: src/models/glm4-moe.cpp -->
```c
    ggml_tensor * routed_out = build_moe_ffn(cur,
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
```

## 四.16 src/models/granite-moe.cpp —— 只有加载面

文件共 84 行，只有 `load_arch_hparams` + `load_arch_tensors` + `build_arch_graph` 三件事；`graph` 类不在本文件。

<!-- src: src/models/granite-moe.cpp -->
```c
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);
```

## 四.17 src/models/granite-swa.cpp —— 真 MoE

多传了一个 `gate_up_exps` 实参（融合路）；专家张量由 `create_tensor_gate_up_exps()` 创建。

<!-- src: src/models/granite-swa.cpp -->
```c
        ggml_tensor * moe_out = build_moe_ffn(cur,
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
```

## 四.18 src/models/granite.cpp —— 真 MoE

与 granite-swa 同构，区别只在注意力（SWA vs 全注意力）。

<!-- src: src/models/granite.cpp -->
```c
        ggml_tensor * moe_out = build_moe_ffn(cur,
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
```

## 四.19 src/models/grok.cpp —— 真 MoE

`ffn_gate_exps` 标记 `TENSOR_NOT_REQUIRED`，`n_ff_exp` 作为专家宽度。

<!-- src: src/models/grok.cpp -->
```c
        ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_GELU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
```

## 四.20 src/models/grovemoe.cpp —— 真 MoE（形态 B）

★ 唯一在本文件里自己写 `ggml_mul_mat`（经 `build_lora_mm`）算路由打分的；随后把它作为 `probs_in` 传进 `build_moe_ffn`，`gate_inp` 位置传 `nullptr`。同一文件下面还有一次针对 `*_chexps`（chunk expert）的调用。

<!-- src: src/models/grovemoe.cpp -->
```c
        ggml_tensor * probs = build_lora_mm(model.layers[il].ffn_gate_inp, cur);  // [n_expert, n_tokens]
        cb(probs, "ffn_moe_logits", il);

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                nullptr,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il,
                probs);
```

## 四.21 src/models/hunyuan-moe.cpp —— 真 MoE

结果先存进 `cur_moe`，之后再与别的分支合并。

<!-- src: src/models/hunyuan-moe.cpp -->
```c
        ggml_tensor * cur_moe = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU,
                true, // norm_topk_prob
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
```

## 四.22 src/models/hy-v3.cpp —— 真 MoE（形态 C）

参数最多的一处：`gate_up_exps` + `ffn_exp_probs_b` + 三个 `*_s` 缩放张量；下面还紧跟一段共享专家的 `build_ffn`，最后 `ggml_add(moe_out, sh_out)`。

<!-- src: src/models/hy-v3.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr, model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
```

## 四.23 src/models/laguna.cpp —— 真 MoE

上一行注释写明 `routed_scaling_factor (all handled by build_moe_ffn)` —— 缩放是原语的活，不是模型文件的活。

<!-- src: src/models/laguna.cpp -->
```c
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
```

## 四.24 src/models/lfm2moe.cpp —— 只有加载面

文件共 85 行，只有 `load_arch_hparams` + `load_arch_tensors`；`build_arch_graph` 返回模板化的 `graph<true/false>`，定义在别处。

<!-- src: src/models/lfm2moe.cpp -->
```c
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),  {n_embd, n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, hparams.n_ff_exp(), n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp(),   n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),   {n_embd, hparams.n_ff_exp(), n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
```

## 四.25 src/models/deci.cpp —— （重引·上下文）

把 deci.cpp 的判断块连同前后文完整引一次，便于核对"没有 else 分支"这一点。

<!-- src: src/models/deci.cpp -->
```c
        // modified to support attention-free layer of Llama-3_1-Nemotron-51B
        ggml_tensor * ffn_inp = cur;
        if (n_head > 0) {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }
        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

```

---

## 说明

- **覆盖声明**：本课声明的源文件是 **25 个** —— `src/models/` 下计划指派的 24 个 `.cpp`，外加 `src/llama-graph.cpp`（`build_moe_ffn` 的定义处）。
- **`src/llama-graph.cpp` 属于 L2-06**（★ 计算图骨架 llama-graph）。本课显式引用并声明它，是因为 MoE 的公共原语定义在这里；覆盖度按并集计算，不会让 L2-06 的声明失配，也不会重复计数。
- **不计入覆盖率**：本课还提到了 `src/llama-hparams.h`（`n_expert` / `n_expert_groups` / `n_group_used` / `n_expert_used()` 的声明处，属 L2-02）、`src/llama-arch.h`（`enum llm_tensor` 里的 `LLM_TENSOR_FFN_GATE_INP` / `_GATE_EXPS` / `_UP_EXPS` / `_DOWN_EXPS` / `_EXP_PROBS_B`）与 `ggml/include/ggml.h`（算子声明，属 L1-02）。这三处只作参数名与算子名的核对，**已在 lesson.spec.py 里通过 grep 确认，但未逐字引用**，故不计入本课覆盖。
- **分组依据**：24 个文件来自 `tools/plan_matrix.py` 的静态扫描规则 `build_moe_ffn|ffn_gate_exps|ffn_gate_inp`（见 `tools/plan_matrix.py` 的 `MODEL_MARKERS`）。本课第四节逐个读并如实记录了 6 处名实不符。
- **命名事实**（已 grep 确认，可复核）：`build_moe_ffn` 里 top-k 用的是 `ggml_argsort_top_k`（第 2089 / 2096 / 2109 行），**不是** `ggml_top_k`；`ggml_top_k` 在整个 `src/llama-graph.cpp` 里出现 0 次。专家矩阵乘用的是 `ggml_mul_mat_id`（经 `build_lora_mm_id`，第 1550 行）。
