<!-- llama-coverage
src/models/arwkv7.cpp
src/models/bailingmoe3.cpp
src/models/deepseek2.cpp
src/models/deepseek32.cpp
src/models/delta-net-base.cpp
src/models/dflash.cpp
src/models/dots3note.cpp
src/models/falcon-h1.cpp
src/models/glm-dsa.cpp
src/models/granite-hybrid.cpp
src/models/hy-v4.cpp
src/models/jamba.cpp
src/models/kimi-k3.cpp
src/models/kimi-linear.cpp
src/models/lfm2.cpp
src/models/mamba2.cpp
src/models/mamba-base.cpp
src/models/mamba.cpp
src/models/minicpm3.cpp
src/models/nemotron-h.cpp
src/models/plamo2.cpp
src/models/plm.cpp
src/models/qwen35.cpp
src/models/qwen35moe.cpp
src/models/qwen3next.cpp
src/models/rwkv6-base.cpp
src/models/rwkv6.cpp
src/models/rwkv6qwen2.cpp
src/models/rwkv7-base.cpp
src/models/rwkv7.cpp
ggml/src/ggml-cpu/ops.cpp
src/llama-hparams.cpp
src/llama-memory-recurrent.h
ggml/include/ggml.h
ggml/src/ggml.c
src/llama-graph.cpp
src/llama-graph.h
-->

# L2-11 · 模型家族（二）：状态空间与线性注意力 — 源文件

**一句话**：状态空间模型（Mamba）与线性注意力（RWKV / gated delta net）在 llama.cpp 里不是新的图引擎，而是**换了一种"记忆"**：不用随上下文增长的 KV cache，改用一份**固定大小的状态张量**，把它交给一两个**融合算子**去读写。`llm_build_mamba_base` / `llm_build_rwkv6_base` / `llm_build_rwkv7_base` / `llm_build_delta_net_base` 四个基类，就是这套记忆的图侧接口。

回顾 L2-06：模型文件不直接调 `ggml_*`，而是调 `llama-graph` 的 `build_*` 原语。本课看到的 `build_rs` / `build_inp_mem_hybrid` 正是那一族里的"记忆"分支 —— 它们**不在** `src/llama-graph.cpp` 里另起一套，而是和 KV cache 的 `build_attn` 并列。回顾 L2-04：`llama_memory_recurrent` 提供 `get_r_l()` / `get_s_l()` 两个张量，本课讲的就是**这两个张量在图上怎么被读、怎么被写回**。

---

## 一、30 个文件的真实主题（分组依据与判定）

**分组依据是静态扫描，不是内容判定。** 命令：

```text
cd /data/WORKSPACE/llama.cpp-project/curriculum
python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L2-11"{print $2}'
```

这 30 个 `src/models/*.cpp` 的图构建代码里命中 `build_ssm|ssm_conv|ssm_scan|gated_delta|build_rwkv|wkv` 之一。**逐文件读过之后，22 个名副其实，8 个名实不符。**

| 文件 | 真实主题（读代码得出） | 判定 | 判断依据 |
|---|---|---|---|
| `arwkv7.cpp` | RWKV-7 变体：token-shift + `build_rwkv7_time_mix` + channel mix | 是 | `llm_build_rwkv7_base` + `ggml_rwkv_wkv7` |
| `bailingmoe3.cpp` | KDA 层 + MLA 层 + MoE | 是（混合） | `build_recurrent_attn`(290) + `ggml_ssm_conv`(213) + `get_s_l`(286) |
| `deepseek2.cpp` | DeepSeek-V2 的 MLA + MoE | 否 | 只有 `wkv_a_mqa`(112) / `wkv_b`(119) 命中；走 `build_attn_inp_kv` |
| `deepseek32.cpp` | DeepSeek-V3.2 的 DSA 稀疏注意力 + MLA | 否 | 3 处命中全是 `wkv_a_mqa`；走 KV cache |
| `delta-net-base.cpp` | gated delta net 公共基类 | 是（基类） | `ggml_gated_delta_net`(402/567) + `build_conv_state`(449) |
| `dflash.cpp` | DFlash 投机解码草稿图（DSV4 / MLA 双模式） | 否 | 命中的 `layer.wkv`(196) 是 `LLM_TENSOR_ATTN_KV`，普通 K/V 投影 |
| `dots3note.cpp` | MLA + DSA indexer | 否 | 源码注释写明 adapted from deepseek32；只有 `wkv_a_mqa`(96/315) |
| `falcon-h1.cpp` | 混合：Mamba2 层 + 注意力层 | 是（混合） | `build_inp_mem_hybrid`(125) + `build_mamba2_layer`(161) |
| `glm-dsa.cpp` | GLM 的 DSA 稀疏注意力 + MLA | 否 | 3 处命中全是 `wkv_a_mqa` |
| `granite-hybrid.cpp` | 混合：Mamba2 层 + 注意力层 | 是（混合） | `is_recr(il)`(162) + `build_mamba2_layer`(164) |
| `hy-v4.cpp` | Hunyuan V4 的 MLA + DSA | 否 | 3 处命中全是 `wkv_a_mqa` |
| `jamba.cpp` | 混合：Mamba-1 层 + 注意力层 + MoE | 是（混合） | `build_inp_mem_hybrid`(117) + `build_mamba_layer`(128) |
| `kimi-k3.cpp` | KDA 层 + MLA 层 | 是（混合） | `n_head_kv == 0` 标记 recurrent 层(30-32) + `build_recurrent_attn`(459) |
| `kimi-linear.cpp` | KDA 层 + MLA 层 | 是（混合） | `ggml_ssm_conv`(226) + `build_rs`(297/342) |
| `lfm2.cpp` | 短卷积块 + 注意力层 | 是（只卷积） | `get_r_l`(192) + `ggml_ssm_conv`(223)；**无 `ssm_scan`** |
| `mamba2.cpp` | Mamba-2 超参与权重加载；图复用 `llama_model_mamba::graph` | 是 | `ssm_conv1d`(69) + `using graph = llama_model_mamba::graph` |
| `mamba-base.cpp` | `llm_build_mamba_base`：`build_mamba_layer` / `build_mamba2_layer` | 是（基类） | `ggml_ssm_scan`(125/264) + `ggml_ssm_conv`(77/232) |
| `mamba.cpp` | Mamba-1/2 的图 | 是 | `llm_build_mamba_base` + 逐层 `build_mamba*_layer`(104/106) |
| `minicpm3.cpp` | MiniCPM3 的 MLA 图 | 否 | 只有 `wkv_a_mqa` / `wkv_b`；走 `build_attn_inp_kv` |
| `nemotron-h.cpp` | 混合：Mamba2 层 + 注意力层 + MoE | 是（混合） | `build_inp_mem_hybrid`(199) + `build_mamba2_layer`(215) |
| `plamo2.cpp` | 自带一份 Mamba 实现（`ggml_ssm_conv` + `ggml_ssm_scan` 内联） | 是 | `ggml_ssm_scan`(385) + `build_rs`(286/388) |
| `plm.cpp` | PLM 的 MLA 图 | 否 | 只有 `wkv_a_mqa` / `wkv_b`；走 `build_attn_inp_kv` |
| `qwen35.cpp` | gated delta net 层 + 注意力层 | 是（混合） | `ggml_ssm_conv`(391) + `build_recurrent_attn`(448) |
| `qwen35moe.cpp` | 同上 + MoE | 是（混合） | `ggml_ssm_conv`(415) + `build_recurrent_attn`(472) |
| `qwen3next.cpp` | gated delta net 层 + 注意力层 + MoE | 是（混合） | `ggml_ssm_conv`(471) + `build_recurrent_attn`(543) |
| `rwkv6-base.cpp` | `llm_build_rwkv6_base` | 是（基类） | `ggml_rwkv_wkv6`(139) / `ggml_gated_linear_attn`(137) |
| `rwkv6.cpp` | RWKV-6 的图 | 是 | `build_rwkv6_time_mix`(130) + `build_rwkv_token_shift_*`(116/148) |
| `rwkv6qwen2.cpp` | RWKV-6 time mix + Qwen2 式 FFN | 是（混合） | `build_rwkv6_time_mix`(116) + `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, ...)`(138) |
| `rwkv7-base.cpp` | `llm_build_rwkv7_base` | 是（基类） | `ggml_rwkv_wkv7`(107) + 状态 `ggml_cpy`(111-114) |
| `rwkv7.cpp` | RWKV-7 的图 | 是 | `build_rwkv7_time_mix`(161) + `build_rwkv7_channel_mix`(190) |

**判定用的三条硬指标**（8 个"否"三条全不满足）：

1. 有没有 `build_rs` / `build_inp_mem_hybrid` 这类 recurrent 记忆入口；
2. 有没有 `ggml_ssm_conv` / `ggml_ssm_scan` / `ggml_rwkv_wkv*` / `ggml_gated_*` 算子；
3. 状态是不是与 `n_ctx` 无关的定长张量。

`lfm2.cpp` 只满足第 2 条的 `ggml_ssm_conv` 与第 3 条的卷积状态，**没有 `ggml_ssm_scan`** —— 它是一个"只借用了卷积状态"的混合模型，本课如实标注为「是（只卷积）」。

## 二、★ recurrent 记忆与 KV cache 的分工

`llama_memory_recurrent_context` 给出的两个张量：`get_r_l(il)` 是卷积状态（大小 `n_embd_r()`），`get_s_l(il)` 是 SSM/WKV 状态（大小 `n_embd_s()`）。两者都不含 token 维。

`n_embd_s()` 是一个**纯结构常量**：RWKV 走 `n_embd * wkv_head_size`，KDA 走 `head_dim * head_dim * n_head()`（源码注释给了 128×128×32 = 524288），Mamba 走 `ssm_d_state * ssm_d_inner`。**没有一项依赖 `n_ctx`。**

对照 L2-04：那一课讲的是 memory 侧的接口与 cell/slot 机制；本课讲的是**图侧怎么把这两个张量取出来、算完、再写回去**。

<!-- src: src/llama-hparams.cpp -->
```c
uint32_t llama_hparams::n_embd_s() const {
    if (wkv_head_size != 0) {
        // corresponds to RWKV's wkv_states size
        return n_embd * wkv_head_size;
    }

    if (n_embd_head_kda != 0) {
        // for Kimi KDA layers
        // Full recurrent state: head_dim * head_dim * n_head
        // h tensor shape for delta attention: [head_dim, head_dim, n_head]
        return n_embd_head_kda * n_embd_head_kda * n_head();  // 128 * 128 * 32 = 524288
    }

    if (n_embd_head_la != 0) {
        // for MiniMax-Text-01 linear attention layers
        // Full recurrent state: head_dim * head_dim * n_head
        // tensor shape for linear attention: [head_dim, head_dim, n_head]
        return n_embd_head_la * n_embd_head_la * n_head();  // 128 * 128 * 64 = 1048576
    }

    // corresponds to Mamba's ssm_states size
    return ssm_d_state * ssm_d_inner;
}
```

## 三、memory 侧的接口：get_r_l / get_s_l

本课图侧的所有"读状态"最终都落到这两个函数上。`get_head()` 给出这一批序列在状态缓存里的起始行，`get_size()` 给出缓存的总行数 —— 与 L2-04 的 KV cache cell/slot 是同一套编号机制。注意还有第三个张量 `get_p_l(il)`（本课覆盖的 30 个文件都没有用到它）。

<!-- src: src/llama-memory-recurrent.h -->
```c
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;
```

## 四、ggml_ssm_scan 的 C API

8 个形参就是这一课验收点的前半部分。紧挨在上面的是 `ggml_ssm_conv`（只有 2 个输入：`sx` 是拼好状态的序列、`c` 是卷积权重）—— 卷积算子连状态张量都不接，状态进出完全由图上的 `ggml_concat` 与 `ggml_cpy` 负责。

<!-- src: ggml/include/ggml.h -->
```c
    GGML_API struct ggml_tensor * ggml_ssm_conv(
            struct ggml_context * ctx,
            struct ggml_tensor  * sx,
            struct ggml_tensor  * c);

    GGML_API struct ggml_tensor * ggml_ssm_scan(
            struct ggml_context * ctx,
            struct ggml_tensor  * s,
            struct ggml_tensor  * x,
            struct ggml_tensor  * dt,
            struct ggml_tensor  * A,
            struct ggml_tensor  * B,
            struct ggml_tensor  * C,
            struct ggml_tensor  * ids,
            int64_t               K);
```

## 五、ggml_ssm_scan 的算子契约（输入 / 输出 / 状态）

`ggml/include/ggml.h:2529-2538` 给出 C API 的 8 个形参；`ggml/src/ggml.c:5711-5751` 用一连串 `GGML_ASSERT` 把它们之间的形状关系钉死，最后构造输出张量 —— **输出是"y 拼上 K 份状态"的一维 F32 张量**。

<!-- src: ggml/src/ggml.c -->
```c
    {
        const int64_t d_state      = s->ne[0];
        const int64_t head_dim     = x->ne[0];
        const int64_t n_head       = x->ne[1];
        const int64_t n_seq_tokens = x->ne[2];
        const int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_head);
        GGML_ASSERT(dt->ne[1] == n_seq_tokens);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == head_dim);
        GGML_ASSERT(s->ne[2] == n_head);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_tokens);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_vector(ids));
        GGML_ASSERT(A->ne[1] == n_head);
        GGML_ASSERT(ggml_is_matrix(A));

        if (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);
```

## 六、状态更新在哪一步发生（验收点）

分两步，分别在两个地方：

**第一步（内核里）**：`ggml/src/ggml-cpu/ops.cpp` 的 `ggml_compute_forward_ssm_scan_f32` 在扫描循环里更新状态。它用 `ids[i3]` 定位旧状态的读指针 `s0`，把新状态的写指针 `s` 指向**算子自己的输出张量**后半段（`s_off = nelements(x) * element_size`），循环体里 `s[i] = state` 就是更新本身。注意：**算子不改持久缓存**。

**第二步（图上）**：模型文件里一个显式的 `ggml_cpy` 把输出张量后半段搬回 `ssm_states_all` 的 `kv_head` 行。Mamba 在 `src/models/mamba-base.cpp:274-279`，RWKV 在 `src/models/rwkv6-base.cpp:144-147`，delta net 在 `src/models/delta-net-base.cpp:555-558`。

<!-- src: src/models/mamba-base.cpp -->
```c
        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);
        const int64_t D            = d_state * d_inner;
        const int64_t n_written    = std::min<int64_t>(n_seq_tokens, K);
        const size_t  row_size     = ggml_row_size(ssm_states_all->type, D);
        const size_t  y_row_size   = ggml_row_size(y_ssm->type, D);
        const size_t  state_offset = ggml_nelements(x) * ggml_element_size(x);

        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0,
                         ggml_view_3d(ctx0, y_ssm, D, n_seqs, n_written,
                                      y_row_size, y_row_size * n_seqs, state_offset),
                         ggml_view_3d(ctx0, ssm_states_all, D, n_seqs, n_written,
                                      ssm_states_all->nb[1], (size_t) mem_size * row_size, kv_head * row_size)));
```

## 七、混合模型：is_recr(il) 把两种层拼进一个图

`jamba` / `granite-hybrid` / `nemotron-h` / `falcon-h1` / `qwen3next` / `qwen35` / `qwen35moe` / `kimi-linear` / `kimi-k3` / `bailingmoe3` 都是混合体：一部分层走 recurrent 记忆，另一部分层走 KV cache，由 `hparams.is_recr(il)` 在图上分支。`kimi-k3.cpp:30-32` 的源码注释明说：`n_head_kv == 0` 就标记为 KDA（recurrent）层。

<!-- src: src/models/granite-hybrid.cpp -->
```c
        if (hparams.is_recr(il)) {
            // ssm layer //
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else {
            // attention layer //
            cur = build_attention_layer(cur, inp_pos, inp->get_attn(), model, n_embd_head, il);
        }
```

## 八、RWKV 的"读-算-写"与 Mamba 的同构性

`rwkv6-base.cpp:133-147` 与 `mamba-base.cpp:258-279` 是同构的：都是"从 `get_s_l(il)` 取状态 → 交给一个融合算子 → 输出张量里同时含 y 与新状态 → `ggml_cpy` 写回"。差别只有两处：用哪个融合算子，以及**状态怎么取** —— RWKV 走 `build_rs` 的默认实现 `ggml_get_rows`（`llama-graph.h:1333`），Mamba / plamo2 传入自己的 lambda，把 `ids` 直接交给 `ggml_ssm_scan`。

<!-- src: src/models/rwkv7-base.cpp -->
```c
    ggml_tensor * wkv_state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

    ggml_tensor * wkv_output = ggml_rwkv_wkv7(ctx0, r, w, k, v, ggml_neg(ctx0, kk), ggml_mul(ctx0, kk, a), wkv_state);
    cur                      = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    ggml_build_forward_expand(
        gf, ggml_cpy(ctx0, wkv_state,
                     ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                                  hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il)))));
```

## 九、token shift：RWKV 的第二份状态

RWKV 每层除了 wkv 状态，还要存"上一个 token 的 norm 输出"（token shift）。它由 `build_rwkv_token_shift_load` / `build_rwkv_token_shift_store` 两个原语读写，占用的正是 `n_embd_r()` 那一份卷积状态。这两个原语的实现在 `src/llama-graph.cpp`（L2-06），下面是 load 那个（store 在 3579-3597，做法就是一个 `ggml_cpy` 回 `get_r_l(il)`）。

<!-- src: src/llama-graph.cpp -->
```c
ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}
```

## 十、build_rs：状态行怎么被取出来

`build_rs` 是 `llama-graph` 的"记忆读原语"，与 KV cache 的 `build_attn` 并列（L2-06）。它做三件事：把状态张量 reshape 成 `{state_size, n_rs}`、把一个槽位清零、然后按 `ids` 取出这一批要用的行。**默认的行选择就是 `ggml_get_rows`**；传了 lambda 的调用方（Mamba / plamo2）可以把 `ids` 一路传进融合算子，省掉这次拷贝。

<!-- src: src/llama-graph.h -->
```c
    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy_main,
            ggml_tensor * state_copy_extra,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    llm_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;
```

## 十一、卷积状态：一个定长移位寄存器

`ggml_ssm_conv` 是纯算子，只吃 `conv_x` 与权重；状态与输入的拼接由 `ggml_concat` 完成，写回由显式 `ggml_cpy` 完成。窗口长度恒为 `d_conv - 1`。`kimi-linear.cpp:216-225` 的注释额外说明了权重布局约定：`ggml_ssm_conv` 按 `c[conv_step + channel * d_conv]` 取权重。

<!-- src: src/models/kimi-linear.cpp -->
```c
    // Reshape conv weight: GGUF [d_conv, 1, d_inner, 1] -> ggml_ssm_conv expects [d_conv, d_inner]
    // GGUF stores as [d_conv, 1, d_inner, 1] with memory layout w[conv_step + channel * d_conv]
    // vLLM stores as [d_inner, d_conv] with memory layout w[channel * d_conv + conv_step]
    // ggml_ssm_conv computes: c[conv_step + channel * d_conv]
    // GGUF layout: [d_conv, 1, d_inner] or [d_conv, 1, d_inner, 1] -> reshape to [d_conv, d_inner]
    // Reshape conv weight from [d_conv, 1, d_inner, 1] to [d_conv, d_inner] for ggml_ssm_conv
    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    // Apply conv1d
    // ggml_ssm_conv output: {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    // Reshape to 2D for bias add: {d_inner, n_tokens}
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
```

---

## 说明

- **覆盖声明**：本课声明块里的 30 个 `src/models/*.cpp` 是本课负责的文件，全部计入覆盖率，与 `tools/plan_matrix.py --files` 里 L2-11 的清单逐字一致。
- **跨课引用（同样计入覆盖率，但它们的主课在别处）**：`ggml/include/ggml.h`（L1-01）、`ggml/src/ggml.c`（L1-02/03/04/05、L4-04）、`ggml/src/ggml-cpu/ops.cpp`（L5-02）、`src/llama-memory-recurrent.h`（L2-04）、`src/llama-graph.{h,cpp}`（L2-06）、`src/llama-hparams.cpp`（L2-02）。本课引用它们，是为了把"模型文件里的调用"对到"算子契约"与"内核实现"上 —— **图原语的出处是 `src/llama-graph.{h,cpp}`（L2-06），不在本课范围**；本课只讲模型文件怎么调用它们。此处明确声明，不虚报。
- **一个需要澄清的措辞**：本课的 `build_ssm_*` / `build_rwkv_*` 原语**不在** `src/llama-graph.cpp` 里。实测：`grep -rn "build_ssm_scan|build_ssm_conv" src/` 无输出；它们实际定义在 `src/models/mamba-base.cpp`（`build_mamba_layer` / `build_mamba2_layer`）、`src/models/rwkv6-base.cpp`（`build_rwkv6_*`）、`src/models/rwkv7-base.cpp`（`build_rwkv7_*`）、`src/models/delta-net-base.cpp`（`build_delta_net` / `build_recurrent_attn`）。`llama-graph.cpp` 里与本课相关的是 `build_rs`(3478) 与 `build_rwkv_token_shift_load/store`(3558/3579)。
- **统计口径**：本课的"22 是 / 8 否"由逐文件读图构建代码得出，判据写在第一节末尾的三条硬指标里；`grep -c` 的命中数只用于定位，不用于判定。
