<!-- llama-coverage
src/models/apertus.cpp
src/models/arcee.cpp
src/models/baichuan.cpp
src/models/bitnet.cpp
src/models/bloom.cpp
src/models/chameleon.cpp
src/models/chatglm.cpp
src/models/codeshell.cpp
src/models/cogvlm.cpp
src/models/cohere2.cpp
src/models/command-r.cpp
src/models/dream.cpp
src/models/eagle3.cpp
src/models/eurobert.cpp
src/models/exaone.cpp
src/models/exaone4.cpp
src/models/falcon.cpp
src/models/gemma-embedding.cpp
src/models/gemma.cpp
src/models/gemma2.cpp
src/models/gemma3.cpp
src/models/gemma3n.cpp
src/models/gemma4-assistant.cpp
src/models/glm4.cpp
src/models/gpt2.cpp
src/models/gptneox.cpp
src/models/granite-switch.cpp
src/models/hrm-text.cpp
src/models/hunyuan-dense.cpp
src/models/hunyuan-vl.cpp
src/models/internlm2.cpp
src/models/jais.cpp
src/models/jais2.cpp
src/models/jina-bert-v2.cpp
src/models/jina-bert-v3.cpp
src/models/llada.cpp
src/models/llama-embed.cpp
src/llama-graph.cpp
src/llama-graph.h
src/models/models.h
src/models/llama.cpp
-->

# L2-14 · 模型家族（五）：稠密 Transformer（上） — 源文件

**一句话**：`src/models/` 下这 37 个文件里，**绝大多数画的是同一张图** —— 嵌入 → 逐层（norm → 注意力 → 残差 → norm → FFN → 残差）→ 末层 norm → lm_head。它们之所以是 37 个文件而不是 1 个，差别只在**几个"层开关"**：QKV 有没有 bias、norm 摆在残差前还是后、embedding 与 lm_head 是否 tied、注意力用不用 sliding window、FFN 用哪个激活、位置编码用 RoPE 还是 ALiBi。

本课的主干来自 `src/models/llama.cpp`。注意一个名实之差：计划文档写的是 `llm_build_llama` —— 那是旧版 llama.cpp 的自由函数名；在 v0.5.0 里它已经改成 **`llama_model_llama::graph<embed>` 的构造函数**（`src/models/llama.cpp:99`），名字变了，图的形状没变。下面一律用真实名字。

回顾 **L2-06**：`build_inp_embd` / `build_norm` / `build_qkv` / `build_attn` / `build_ffn` / `build_lora_mm` 这些原语的**实现**在 `src/llama-graph.cpp`，本课只讲**谁在什么顺序上调用它们**。回顾 **L2-02**：`n_layer` / `n_head` / `n_embd` 这些超参决定循环次数与张量形状，本课的循环直接读它们。下一课 **L2-15** 讲同一条主干上的注意力变体（GQA / SWA / MLA）与投机解码草稿模型。

---

## 一、本课覆盖：37 个文件从哪来

本课的 37 个文件全部取自 `src/models/`，是 `tools/plan_matrix.py` 里 **L2-14** 的清单。

**分组依据（如实写）**：计划把这批文件归到"稠密 Transformer"，不是因为在代码里找到了一个叫"稠密"的标记，而是因为**它们没有命中其它三组图原语**：

| 组 | 判据（代码层面） | 本课是否命中 |
|---|---|---|
| 多模态（L2-10） | 文件里出现视觉/音频塔（`clip`、`mmproj`、图像嵌入输入） | 少数命中：`cogvlm.cpp`、`gemma3n.cpp` |
| 状态空间 / 线性注意力（L2-11） | 调用 `build_rs`、RWKV/Mamba/SSM 原语 | 不命中 |
| 稀疏专家（L2-12 / L2-13） | `ffn_gate_inp` 非空、调用 `build_moe_ffn` | **零命中** |
| 其余 | —— | 本课这 37 个 |

所以"37 个稠密模型"是**排除法的结果**，不是同质的 37 份代码。逐个读完之后发现：其中有 4 个文件**根本没有主干**（图在别处），另有若干实际属于编码器 / 扩散 / 草稿模型家族。这些都在下面的清单表里逐行标出。

<!-- src: src/models/llama.cpp -->
```c
std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
}
```

## 二、★ 主干十站：llama.cpp 逐站读

主干写在 `llama_model_llama::graph<embed>` 的构造函数里（不是旧版的 `llm_build_llama` 自由函数）。下面是它的四个段落，逐字引用：

**第 1 段（输入，四站）** —— `build_inp_embd` / `build_inp_pos` / `build_attn_inp_kv` / `build_inp_out_ids`：

<!-- src: src/models/llama.cpp -->
```c
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();
```

## 三、主干第 2 段：逐层循环的注意力半边

循环体前半：`build_norm`（pre-norm）→ `build_qkv` → `ggml_rope_ext` → `build_attn`。四个调用全部是原语，模型文件里看不到任何 `ggml_*` 算子。

<!-- src: src/models/llama.cpp -->
```c
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }
```

## 四、主干第 3 段：残差与 FFN（dense / MoE 的分叉）

`if (model.layers[il].ffn_gate_inp == nullptr)` 这一行判空，就是"稠密 FFN"与"MoE"两条路的唯一分叉点。本课 37 个文件全部走稠密那一支。

<!-- src: src/models/llama.cpp -->
```c
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

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
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
```

## 五、主干第 4 段：末层 norm 与 lm_head

`build_norm(..., -1)` 是唯一的"层外"归一化；`if constexpr (!embed)` 是编译期开关，`llama-embed.cpp` 用 `graph<true>` 把 lm_head 整段编译掉。

<!-- src: src/models/llama.cpp -->
```c
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur, model.output_s);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}
```

## 六、★ 十站对应的 build_* 原语

主干上每一站都是一个原语。原语的**实现**在 `src/llama-graph.cpp`（L2-06 的主题），这里只引用**声明**，用来看清"一个原语的签名 = 它能表达多少种模型"。

特别注意 `build_qkv` 的两个重载与 `build_ffn` 那一长串可选指针（`*_b` 偏置、`*_s` 缩放）：**同一行调用，靠"传不传"就能表达不同模型** —— 这就是"层开关"的实现方式。

<!-- src: src/llama-graph.h -->
```c
    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora and per-tensor scale
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * w_s = nullptr) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const;

    // Set reshape to false to return contiguous projections before clamp/reshape.
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape = true) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
```

## 七、层开关对照：把 gemma3 拨到另一边

gemma3 是"开关拨得最多"的稠密模型之一，但它仍然走同一条主干十站：QK-norm（131/139）、调度用的 post-norm（162/185）、GELU 的 FFN（177）、以及 FFN 之后那第三个 norm（185）。对照 `llama.cpp:132/184`，差别只在"norm 摆在残差前还是后"。

<!-- src: src/models/gemma3.cpp -->
```c
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        cur = build_norm(sa_out,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, sa_out);
```

## 八、"名实不符"的文件：主干不在这个文件里

逐个读完 37 个文件后，有四类必须如实说明：

**(1) 只有 6 行、根本没有主干**：`hunyuan-dense.cpp` 与 `llama-embed.cpp` 各自只实现一个 `build_arch_graph()`，图靠 `models.h` 里一行 `using graph =` 继承过来。

<!-- src: src/models/models.h -->
```c
struct llama_model_llama_embed : public llama_model_llama {
    llama_model_llama_embed(const struct llama_model_params & params) : llama_model_llama(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_llama

    template <bool embed>
    using graph = llama_model_llama::graph<embed>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
//>> ---- src/models/models.h:2101-2108 ----
struct llama_model_hunyuan_dense : public llama_model_hunyuan_vl {
    llama_model_hunyuan_dense(const struct llama_model_params & params) : llama_model_hunyuan_vl(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_hunyuan_vl

    using graph = llama_model_hunyuan_vl::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

## 八（续）、图在别的源文件里的两个 BERT 模型

`jina-bert-v2.cpp` / `jina-bert-v3.cpp` 同样把图指到 `llama_model_bert::graph`（`bert.cpp`，不在本课清单内，故本课不引用其源码、不计入本课覆盖）。它们还带 `attn_out_norm` / `layer_out_norm` 这种 BERT 式后置归一化 —— 不是自回归主干。

<!-- src: src/models/models.h -->
```c
struct llama_model_jina_bert_v2 : public llama_model_base {
    llama_model_jina_bert_v2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jina_bert_v3 : public llama_model_base {
    llama_model_jina_bert_v3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

## 九、不属于稠密 Transformer 的其它文件（如实标注）

按逐行读到的证据分类：

| 文件 | 证据（行号） | 实际家族 |
|---|---|---|
| `dream.cpp` | `hparams.causal_attn = false;`（L15）+ `build_attn_inp_no_cache()`（L68） | 扩散式（双向、无 KV cache） |
| `llada.cpp` | 注释 `Non-causal attention for diffusion` + `build_attn_inp_no_cache()`（L82） | 扩散式 |
| `eurobert.cpp` | `build_attn_inp_no_cache()`（L50）、全文无 `build_lora_mm(model.output` | 编码器（无 lm_head） |
| `gemma-embedding.cpp` | `hparams.causal_attn = false;`（L7）+ no-cache（L89） | 句向量模型 |
| `jina-bert-v2/v3.cpp` | `using graph = llama_model_bert::graph;`（models.h L308/L319） | BERT 编码器 |
| `eagle3.cpp` | 两张图（L125 / L152）、encoder 只做 `build_lora_mm(model.fc, ...)`（L137） | 投机解码草稿模型 |
| `gemma4-assistant.cpp` | 只建 `n_layer_nextn` 层（L127）、`k_cur/v_cur = nullptr`（L151） | 投机解码草稿头 |
| `granite-switch.cpp` | 自定义输入类（L134-147）、手写 FFN（`build_ffn` 从未被调用，L411-415）、图内路由器（L265-287） | 可切换适配器（switch LoRA）家族 |
| `hrm-text.cpp` | 深度循环双栈（L185-196 嵌套循环）、无参数 RMSNorm（`nullptr` 权重，L107） | HRM 深度循环家族 |
| `cogvlm.cpp` | `is_text` 二选一，取 `visexp_attn_wqkv` / `visexp_ffn_*`（L85-97） | 多模态（视觉专家） |
| `gemma3n.cpp` | AltUp / Laurel / 逐层嵌入（L324-376）、FFN 手写（L214-224） | 多模态家族的特殊主干 |
| `hunyuan-vl.cpp` | M-RoPE 四段（L66-69 / L104-115）、视觉层输入 tap（L86） | 多模态的文本半边 |
| `hunyuan-dense.cpp` | 6 行，`using graph = llama_model_hunyuan_vl::graph`（models.h L2105） | 复用 `hunyuan-vl.cpp` 的图 |
| `llama-embed.cpp` | 6 行，`graph<true>`（embed 模板） | 复用 `llama.cpp` 主干、无 KV cache、无 lm_head |

**这只是标注，不是删减**：这 15 个文件依然被本课声明覆盖。把它们的"名"与"实"写清楚，比把它们硬说成稠密 Transformer 更有用。

<!-- src: src/models/bitnet.cpp -->
```c
    // lm_head
    // FIXME: do not use model.tok_embd directly, duplicate as model.output
    cur = build_lora_mm(model.tok_embd, cur);
```

## 十、37 个文件的差异点（逐行读出的结果）

下表逐行给出每个文件的**归一化类型 / 激活 / 真实差异点**，行号指向该文件自身。

| 文件 | 归一化 | 激活 | 真实差异点（行号在源文件里） |
|---|---|---|---|
| `apertus.cpp` | RMS | xIELU | FFN 手写：`up -> ggml_xielu -> down`（L138） |
| `arcee.cpp` | RMS | ReLU² | 无 gate，`LLM_FFN_SEQ`（L128） |
| `baichuan.cpp` | RMS | SILU | 13B 不接位置编码 + ALiBi（L13 / L90） |
| `bitnet.cpp` | RMS ×5 | SILU | 层内多两个 sub-norm（L103 / L137）；lm_head 用 `tok_embd`（L164） |
| `bloom.cpp` | LayerNorm | GELU | fused `wqkv` + bias（L45）；嵌入后先归一化（L77）；ALiBi（L18） |
| `chameleon.cpp` | RMS + LN | SILU | Q/K 用 LayerNorm 且可选（L91-104）；`swin_norm` 只改 norm 位置（L7） |
| `chatglm.cpp` | RMS | SwiGLU | gate 串行：up 出 2 倍宽（L48 / L133） |
| `codeshell.cpp` | LayerNorm | GELU | `wo_b` 偏置（L100）；反向 tie：`tok_embd` 复用 output（L15-20） |
| `cogvlm.cpp` | RMS | SILU | 文本/视觉两套权重按 ubatch 二选一（L72-97） |
| `cohere2.cpp` | LayerNorm | SILU | SWA 与全局层交错（L70）；并行残差（L131）；强制 tie（L29） |
| `command-r.cpp` | LayerNorm | SILU | 并行残差（L118）；无 `ffn_norm`；Q/K norm 仅 64 层以上（L28） |
| `dream.cpp` | RMS | SILU | 非因果 + 无 KV cache（L15 / L68） |
| `eagle3.cpp` | RMS | SILU | 两张图：encoder 只做 fc（L137）、decoder 先拼接（L220） |
| `eurobert.cpp` | RMS | SILU | no-cache（L50）、无 lm_head |
| `exaone.cpp` | RMS | SILU | 逐层 `rope_freqs`（L75）；输出可 tied（L24） |
| `exaone4.cpp` | RMS | SILU | 无 pre-norm，全 post-norm（L113）；SWA；跳过 nextn 层（L42） |
| `falcon.cpp` | LayerNorm | GELU | FFN 吃 `attn_norm` 而非 attn 输出（L125）；双残差（L134-135） |
| `gemma-embedding.cpp` | RMS | GELU | `causal_attn = false`（L7）+ 无 cache（L89） |
| `gemma.cpp` | RMS | GELU | 嵌入乘 `sqrt(n_embd)`（L49）；output 恒为副本（L20） |
| `gemma2.cpp` | RMS | GELU | post-norm + SWA（L74）+ 两处 softcap（L14 / L167） |
| `gemma3.cpp` | RMS | GELU | QK-norm（L131 / L139）+ post-norm + SWA + tied（L44） |
| `gemma3n.cpp` | RMS | GELU | AltUp/Laurel + 逐层嵌入（L324-376）；FFN 手写（L214-224） |
| `gemma4-assistant.cpp` | RMS | GELU | 只建 `n_layer_nextn` 层（L127），KV 借主模型（L151） |
| `glm4.cpp` | RMS | SwiGLU | post-norm（L140 / L162）+ mRoPE（L81） |
| `gpt2.cpp` | LayerNorm | GELU | fused `wqkv` + bias（L38）；学习式位置嵌入，无 RoPE（L74-77） |
| `gptneox.cpp` | LayerNorm | GELU | `use_par_res`：注意力与 FFN 并行（L145）；`wqkv_b`/`wo_b` 都带（L67 / L70） |
| `granite-switch.cpp` | RMS | SILU | 自定义输入 + 手写 FFN + 图内路由器（L134-147 / L411-415） |
| `hrm-text.cpp` | RMS（无参数） | SILU | 深度循环双栈交替跑同一批层（L101 / L168）；注意力带 sigmoid gate（L112） |
| `hunyuan-dense.cpp` | - | - | 只有 6 行：图继承 `hunyuan-vl`（models.h L2105） |
| `hunyuan-vl.cpp` | RMS | SILU | M-RoPE 四段（L66-69）+ 视觉层输入 tap（L86） |
| `internlm2.cpp` | RMS | SILU | 无 `wo_b`；QKV 走 `create_tensor_qkv`（L26） |
| `jais.cpp` | LayerNorm | SILU | ALiBi（L5）；不调 `build_inp_pos`（无 RoPE） |
| `jais2.cpp` | LayerNorm | ReLU² | 无 gate，`LLM_FFN_SEQ`（L130）；tied（L22-24） |
| `jina-bert-v2.cpp` | LayerNorm | - | 本文件无主干：图在 `bert.cpp`（models.h L308） |
| `jina-bert-v3.cpp` | LayerNorm | - | 49 行只有张量表；图在 `bert.cpp`（models.h L319） |
| `llada.cpp` | RMS | SILU | 非因果（L16）+ 无 KV cache（L82） |
| `llama-embed.cpp` | - | - | 只有 6 行：复用 `llama.cpp` 主干，`graph<true>` |

### 主干序列（可复述版）

```text
build_inp_embd(model.tok_embd)                     llama.cpp:108
build_inp_pos()                                    llama.cpp:111
build_attn_inp_kv()                                llama.cpp:119
build_inp_out_ids()                                llama.cpp:124
for il in 0 .. n_layer-1:                          llama.cpp:126
    build_norm(inpL, attn_norm, NULL, RMS, il)     llama.cpp:132
    build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il)   llama.cpp:143
    ggml_rope_ext(Q) / ggml_rope_ext(K)            llama.cpp:146 / 152
    build_attn(inp_attn, wo, wo_b, wo_s, Q,K,V, ...)           llama.cpp:169
    ggml_add(cur, inpSA)                  # 残差 1   llama.cpp:178
    build_norm(ffn_inp, ffn_norm, NULL, RMS, il)   llama.cpp:184
    build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il)  llama.cpp:189
    ggml_add(cur, ffn_inp)                # 残差 2   llama.cpp:220
    build_cvec(cur, il)                            llama.cpp:223
build_norm(cur, output_norm, NULL, RMS, -1)        llama.cpp:231
build_lora_mm(model.output, cur, model.output_s)   llama.cpp:240
ggml_build_forward_expand(gf, cur)                 llama.cpp:246
```

（上面是示意性文字块，不是源码引用；每一行的行号都指向本源文件里的逐字引用。`text` 围栏不参与保真门禁。）

---

## 说明

- 本课覆盖声明共 41 项 = **L2-14 清单里的 37 个 `src/models/*.cpp`**（逐个出现在上面某一节的逐字引用、或本课的场景 `src=` 里）+ `src/models/llama.cpp`（主干的标准答案；plan 把它归给 L2-13，本课以它为主干逐字引用）+ `src/llama-graph.cpp` 与 `src/llama-graph.h`（与 `build_*` 原语对照，L2-06 的主文件，本课有意交叉引用）+ `src/models/models.h`（三处 `using graph =`，用来解释"6 行的模型文件"；该文件同时属于 L2-10 / L2-15）。
- plan 文档写的验收点是"能默写出 `llm_build_llama` 的主干图"。`llm_build_llama` 是旧版 llama.cpp 的函数名；在 v0.5.0 里主干已改为 `llama_model_llama::graph<embed>` 的构造函数（`src/models/llama.cpp:99`）。本课按真实代码命名，并在末幕给出可默写的主干序列。
- `src/llama-hparams.h` 与 `src/llama-model.cpp` 里的事实（`f_max_alibi_bias`、`swa_type`/`is_swa`、`create_tensor_qkv` 的 `TENSOR_NOT_REQUIRED`）只在叙述中被引用，**本课不引用其源码、不计入本课覆盖**（它们分别属于 L2-02 与 L2-05）。
