<!-- llama-coverage
src/llama-graph.cpp
src/llama-graph.h
-->

# L2-06 · ★ 计算图骨架 llama-graph — 源文件

**一句话**：`src/llama-graph.{h,cpp}` 是"模型定义 → 计算图"之间唯一的翻译层。`src/models/` 下 156 个文件里有 137 个不直接调 `ggml_*`，而是调这里的 `build_*` 原语；原语负责把**超参**变成**具体算子**，把算子挂到一张 `ggml_cgraph` 上。余下 19 个（视觉塔 clip.cpp、BERT 系、RWKV/Mamba 系等）在自己的 `build()` 里直接调 ggml。

回顾 L1-01 / L1-03：`ggml_tensor` 是图上的节点（数据面 + 身份面），`ggml_cgraph` 是节点序列。本课讲的是：**这些节点是谁、按什么规则造出来的**。回顾 L2-02：`hparams` 决定层数、头数、专家数；回顾 L2-04：KV cache 的读写算子由 memory 上下文提供；回顾 L2-05：`ubatch` 决定这一批有多少 token。这四样东西在 `llm_graph_context` 里汇合。

---

## 一、这一课在链条上的位置

`src/llama-graph.{h,cpp}` 是"模型定义到 ggml 计算图"之间唯一的翻译层。`src/models/` 下 156 个文件里有 137 个不直接调用 `ggml_*`，而是调用这里的 `build_*` 原语。原语的产物只有两样：一个张量 arena（`ctx0`）和一张节点序列（`gf`）。

<!-- src: src/llama-graph.h -->
```c
    llm_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;
```

## 二、★ llm_graph_context：把超参摊平成常量的工作台

构造函数 44 行，全是 `成员 (来源)` 的形式。右列决定了图的形状：

| 来源 | 决定什么 | 例子 |
|---|---|---|
| `hparams` | 图的形状（结构） | `n_embd` `n_layer()` `n_head_kv()` `n_expert` |
| `cparams` | 算子的参数（运行期） | `n_ctx` `rope_freq_base` `yarn_*` |
| `ubatch` | 图的宽度 | `n_tokens` |
| `res` | 产物 | `ctx0 = res->get_ctx()` `gf = res->get_gf()` |

注意最后两行：`ctx0` 与 `gf` 不是新造的，而是从 `res` 里取的 —— 图复用（L2-07）靠的就是这一点。

<!-- src: src/llama-graph.cpp -->
```c
llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }
```

## 三、build_norm：三种归一化一个入口

`llm_norm_type` 决定用 `ggml_norm`、`ggml_rms_norm` 还是 `ggml_group_norm`；两个 eps（`f_norm_eps` 与 `f_norm_rms_eps`）是不同的 hparams 字段。归一化之后的 `ggml_mul`（权重）几乎人人都有，`ggml_add`（偏置）只有部分架构有 —— **有没有偏置就是 RMSNorm 家族与 LayerNorm 家族在图上的分界**。

<!-- src: src/llama-graph.cpp -->
```c
ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
```

## 四、build_ffn：三次矩阵乘 + 一个激活

`build_lora_mm` 负责 up / gate / down 三次投影（并顺带处理 LoRA 与 per-tensor scale）。`type_gate` 决定 gate 是接在 up 之后（`LLM_FFN_SEQ`）还是与 up 并行（`LLM_FFN_PAR`）；`type_op` 决定激活函数。最常用的 SILU + PAR 组合只用一行 `ggml_swiglu_split`。**同一个 `build_ffn`，不同架构只是换两个枚举值。**

<!-- src: src/llama-graph.cpp -->
```c
    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_swiglu_clamp(ctx0, cur, tmp, limit);
```

## 五、build_attn：KV cache 的写与读

这一层不做 Q/K/V 投影（那是 `build_qkv`，1619 行起），负责的是记账：用 `mctx_cur->cpy_k` 与 `cpy_v` 把本轮的 K/V 写进缓存，再用 `get_k` 与 `get_v` 把整段（历史 + 本轮）读回来。这两个函数由 memory 上下文提供，不是 `ggml.h` 里的算子 —— 接口定义见 L2-04。

**rope 不在这个文件里**：各模型架构在调用 `build_attn` 之前用 `ggml_rope_ext` 处理 Q/K（见 `src/models/` 下各架构文件，L2-10~L2-15）。

<!-- src: src/llama-graph.cpp -->
```c
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);
```

## 六、build_attn_mha：一个节点还是四个节点

`cparams.flash_attn` 为真且没有 KQ bias 时，整个注意力塌缩成一个 `ggml_flash_attn_ext` 节点（K 与 V 需要先 cast 成 F16）。慢路则展开成 `ggml_mul_mat` 加 `ggml_soft_max_ext` 再加 `ggml_mul_mat`。**同一个语义，算子粒度可以差四倍** —— 这是本视角下"算子粒度不固定"最清楚的例子。

<!-- src: src/llama-graph.cpp -->
```c
    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        ggml_prec_set_acc(cur, GGML_PREC_F32);
```

## 七、★ build_moe_ffn：一个函数，11 步，7 种 ggml 算子

MoE 层分四段展开（行号见第 9 幕的表）：路由打分、选专家与权重、专家并行、聚合。其中 `ggml_mul_mat_id` 是关键：一个算子同时对所有选中专家做矩阵乘，这就是"专家并行"在图上的样子。它由 `build_lora_mm_id`（1552 行）发出。

<!-- src: src/llama-graph.cpp -->
```c
    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
    if (arch == LLM_ARCH_MISTRAL4) {
        // src1 can exceed F16 range
        ggml_prec_set_src(experts, GGML_PREC_F32, 1);
    }
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    // Use per-layer n_expert_used to bound the graph even during warmup (avoids
    // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
    // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
    for (uint32_t i = 0; i < n_expert_used_il; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < n_expert_used_il; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used_il == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);
```

## 八、输入原语：只造占位张量，不算东西

`build_inp_*` 一族不参与计算，只负责**造占位输入张量**并登记到 `res`：token id、位置、输出索引、attention scale 等。它们在建图时是空张量，由解码流程在计算前填真实数据（L2-07）。

<!-- src: src/llama-graph.h -->
```c
    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;
    ggml_tensor * build_inp_pos() const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
```

---

## 说明

- 本课只引用 `src/llama-graph.cpp` 与 `src/llama-graph.h`，两者均计入覆盖率。
- 文中提到的 `ggml_*` 算子名（`ggml_mul_mat_id` / `ggml_argsort_top_k` / `ggml_swiglu_split` 等）来自 `ggml/include/ggml.h`；该文件不在本课覆盖声明内，故不计入本课覆盖率。
- `src/models/` 的统计数字（156 个文件 / 137 个出现原语调用 / 57 个调用 build_moe_ffn）由 grep 实测得到，不是估计值。余下 19 个文件（视觉塔 clip.cpp、BERT 系、RWKV/Mamba 系等）在自己的 build() 里直接调 ggml，不经过本课的原语。
