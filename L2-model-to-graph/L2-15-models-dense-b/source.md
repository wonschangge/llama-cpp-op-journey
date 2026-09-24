<!-- llama-coverage
src/models/maincoder.cpp
src/models/mistral4.cpp
src/models/models.h
src/models/modern-bert.cpp
src/models/mpt.cpp
src/models/muse-glimmer.cpp
src/models/nanbeige.cpp
src/models/nemotron.cpp
src/models/neo-bert.cpp
src/models/nomic-bert.cpp
src/models/olmo.cpp
src/models/olmo2.cpp
src/models/openelm.cpp
src/models/orion.cpp
src/models/paddleocr.cpp
src/models/pangu-embed.cpp
src/models/phi2.cpp
src/models/plamo.cpp
src/models/plamo3.cpp
src/models/qwen.cpp
src/models/qwen2.cpp
src/models/qwen2vl.cpp
src/models/qwen3.cpp
src/models/qwen3tts.cpp
src/models/qwen3vl.cpp
src/models/seed-oss.cpp
src/models/smollm3.cpp
src/models/spark2-5.cpp
src/models/stablelm.cpp
src/models/starcoder.cpp
src/models/starcoder2.cpp
src/models/t5.cpp
src/models/t5encoder.cpp
src/models/talkie.cpp
src/models/wavtokenizer-dec.cpp
src/models/xverse.cpp
src/llama-hparams.h
src/llama-kv-cache.cpp
src/models/gemma4-assistant.cpp
src/llama-graph.h
src/llama-model.cpp
src/llama-graph.cpp
src/llama-hparams.cpp
-->

# L2-15 · 模型家族（六）：稠密 Transformer（下）与投机解码草稿模型 — 源文件

**一句话**：这一课看 `src/models/` 下的 36 个文件，把它们分成两类 —— **注意力变体**（GQA / SWA / MLA）只是同一个 `build_attn` 的不同配置，图代码里的差异小到可以写成一行 `std::conditional_t`；而**投机解码的草稿模型**是真正的结构性差异：层数少、张量清单短、而且它**不写自己的 KV cache**。

上一课 L2-14 讲了稠密主干序列（llama / gemma / gpt 那一批）。本课接着往下走，覆盖的是 qwen / olmo / plamo / bert / t5 这一批，以及草稿模型在主模型侧留下的钩子。

回顾 L2-06 幕 10 的结论：**"模型架构的差异 = 原语选择 + 超参"**。这一课要做的事，是给这句话划一条边界 —— 它对注意力变体成立，对草稿模型不成立。回顾 L2-04 幕 7：SWA 层的记忆只需要一个窗口，所以 iswa 是**两套 cache**；本课看它在模型文件里怎么被声明。

---

## 一、模型注册表：一个家族 = 一个类

`src/models/models.h` 里有 **151 个** `llama_model_*` 类，全部继承 `llama_model_base`。每个类只承诺三件事：读超参、读张量、给出自己的图类。本课覆盖的 36 个文件里，32 个 `.cpp` 把这三个函数都实现了；`paddleocr.cpp`（继承 `llama_model_ernie4_5`）与 `mistral4.cpp`（继承 `llama_model_deepseek2`）只写 `build_arch_graph`，`qwen3tts.cpp` 一个都不写 —— 剩下的是声明表 `models.h` 自己。

<!-- src: src/models/models.h -->
```cpp
//
// models
//

struct llama_model_llama : public llama_model_base {
    llama_model_llama(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool embed>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

## 二、GQA：形状上的两个 head 数

GQA（grouped-query attention）在代码里没有任何专属分支。它的全部表达就是 `llm_graph_qkv` 这个三元组里 K/V 与 Q 的**第二维不同**：Q 用 `n_head`，K/V 用 `n_head_kv`。任何"变体名"在图上都不存在 —— 存在的只是两个数字。

<!-- src: src/llama-graph.h -->
```cpp
struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};
```

## 三、一个稠密模型文件的注意力长什么样

把 Qwen2 的层循环摊开看：取注意力输入 -> 建 Q/K/V -> RoPE -> 交给 `build_attn`。本课 23 个文件都是这个骨架，差别只在 `build_qkv` 的第 6 个实参和 RoPE 的参数表。第 67 行的 `build_attn_inp_kv()` 就是"稠密因果 + 一套 KV cache"的声明。

<!-- src: src/models/qwen2.cpp -->
```cpp
    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);
```

## 四、SWA 的超参面

窗口语义、窗口宽度、逐层开关 —— 三样东西都在 `llama_hparams` 里。注意 `is_swa_impl` 是**逐层数组**：这就是"分层注意力"在数据上的全部表达。源码注释写明了约定：`is_swa_impl[il] == 1` 表示这一层是 SWA 层，0 表示稠密层。

<!-- src: src/llama-hparams.h -->
```cpp
    // Sliding Window Attention (SWA)
    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    // the size of the sliding window (0 - no SWA)
    uint32_t n_swa = 0;

    // see llama_non_causal_type
    // note: for SWA_FULL, older tokens (outside the current ubatch) are still window-clipped
    llama_non_causal_type non_causal_type = LLAMA_NON_CAUSAL_TYPE_ALL;

    // if is_swa_impl[il] == 1, then layer il is SWA
    // if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
    // by default, all layers are dense
    // note: using uint32_t type for compatibility reason
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_swa_impl;
```

## 五、周期怎么变成逐层开关

`load_swa_pattern()` 只做两件事：先试着从元数据里直接读一个逐层数组（`get_arr`），读不到就把"周期"展开。展开规则在 `llama_hparams::set_swa_pattern()` 里：`is_swa_impl[il] = n_pattern == 0 || (il % n_pattern < (n_pattern - 1))`。所以 `load_swa_pattern(ml, 4)` 得到的是"每 4 层里前 3 层 SWA、第 4 层全上下文"；`dense_first = true` 时判据反过来，周期里的第一层是稠密层 —— modern-bert 用的就是 `load_swa_pattern(ml, 3, true)`。

<!-- src: src/llama-model.cpp -->
```cpp
void llama_model_base::load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first) {
    if (ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, false)) {
        return;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern, false);
    hparams.set_swa_pattern(n_pattern, dense_first);
}
```

## 六、窗口只有四种语义

`is_masked_swa()` 是全仓库**唯一**判断"这个位置对那个位置可见吗"的地方 —— 无论有没有 KV cache 都走它。四个 `case` 就是四种窗口语义。注意它的参数是 `(n_swa, swa_type, p0, p1)`：只有位置，没有张量。

<!-- src: src/llama-hparams.h -->
```cpp
    // TODO: pack the SWA params in a struct?
    static bool is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1) {
        assert(p0 >= 0 && p1 >= 0);

        switch (swa_type) {
            case LLAMA_SWA_TYPE_NONE:
                {
                } break;
            case LLAMA_SWA_TYPE_STANDARD:
                {
                    if (p1 - p0 >= (int32_t) n_swa) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_CHUNKED:
                {
                    const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                    if (p0 < pos_chunk_start) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_SYMMETRIC:
                {
                    const int32_t half_n_swa = (int32_t) n_swa / 2;
                    const int32_t pos_diff = p1 - p0;

                    // Mask if outside the symmetric window
                    if (pos_diff < -half_n_swa || pos_diff > half_n_swa) {
                        return true;
```

## 七、窗口在图上的三个落点

`build_attn` 的 iswa 版里，`is_swa(il)` 只被用来做三次选择：读哪一套 cache（`get_swa()` / `get_base()`）、读哪一张 mask、往哪里写 K/V。这三行就是"SWA 在图上"的全部。编码器模型（`modern-bert` / `neo-bert`）走的是 `build_attn_inp_no_cache()` 那条路，但窗口判据仍然是同一个 `llama_hparams::is_masked_swa()`（`llama-graph.cpp:438`）。

<!-- src: src/llama-graph.cpp -->
```cpp
    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
```

## 八、MLA 是一组超参，不是一个算子

`is_mla()` 的判据是两个 impl 字段同时非零；`n_embd_head_k_mla()` / `n_embd_head_v_mla()` 在没有 MLA 时直接退回普通的 head 维度。源码在字段声明处留下的注释说明了这套表示的来历："deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA"。

<!-- src: src/llama-hparams.cpp -->
```cpp
bool llama_hparams::is_mla() const {
    assert((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
           (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));

    return n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0;
}

bool llama_hparams::is_indexer_full(uint32_t il) const {
    if (il < n_layer()) {
        return is_indexer_full_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer: %u)\n", __func__, il, n_layer());
}

uint32_t llama_hparams::n_embd_head_k_mla() const {
    return is_mla() ? n_embd_head_k_mla_impl : n_embd_head_k();
}

uint32_t llama_hparams::n_embd_head_v_mla() const {
    return is_mla() ? n_embd_head_v_mla_impl : n_embd_head_v();
}
```

## 九、MLA 在 KV cache 上的唯一痕迹

cache 构造循环里，MLA 与非 MLA 走同一段代码，只在两处分开：`if (!is_mla)` 包住"V 的 head 维度统计"，以及 `has_v = !is_mla`。`has_v` 为假时 V 张量是 `nullptr` —— 后续所有 `v_stream` 与 `get_v()` 都走空路。这就是"压缩 KV"落到内存上的样子：**少一张张量**。

<!-- src: src/llama-kv-cache.cpp -->
```cpp
        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);
```

## 十、36 个文件的注意力变体清单

依据行的含义：纯数字是该文件的行号；`hN` 指 `src/models/models.h` 第 N 行。

| 文件 | 注意力输入 / 变体 | 依据行 |
|---|---|---|
| `maincoder.cpp` | `build_attn_inp_kv()` + QK-norm | 61 |
| `mpt.cpp` | `build_attn_inp_kv()` + fused QKV | 78 |
| `nanbeige.cpp` | `build_attn_inp_kv()` + 导出层输入 | 97 |
| `nemotron.cpp` | `build_attn_inp_kv()` | 64 |
| `olmo.cpp` | `build_attn_inp_kv()` | 57 |
| `orion.cpp` | `build_attn_inp_kv()` | 57 |
| `paddleocr.cpp` | `build_attn_inp_kv()` | 28 |
| `pangu-embed.cpp` | `build_attn_inp_kv()` | 72 |
| `phi2.cpp` | `build_attn_inp_kv()` | 62 |
| `plamo.cpp` | `build_attn_inp_kv()` | 53 |
| `qwen.cpp` | `build_attn_inp_kv()` | 56 |
| `qwen2.cpp` | `build_attn_inp_kv()` | 67 |
| `qwen2vl.cpp` | `build_attn_inp_kv()` | 56 |
| `qwen3.cpp` | `build_attn_inp_kv()` + QK-norm + 导出层输入 | 67 |
| `qwen3vl.cpp` | `build_attn_inp_kv()` + QK-norm | 80 |
| `seed-oss.cpp` | `build_attn_inp_kv()` | 63 |
| `smollm3.cpp` | `build_attn_inp_kv()` | 60 |
| `talkie.cpp` | `build_attn_inp_kv()` | 58 |
| `starcoder.cpp` | `build_attn_inp_kv()` | 73 |
| `starcoder2.cpp` | `build_attn_inp_kv()` | 73 |
| `stablelm.cpp` | `build_attn_inp_kv()` + QK-norm | 64 |
| `openelm.cpp` | `build_attn_inp_kv()` + QK-norm | 62 |
| `xverse.cpp` | `build_attn_inp_kv()` | 55 |
| `muse-glimmer.cpp` | `build_attn_inp_kv_iswa()` + SWA(周期 4) | 73 |
| `spark2-5.cpp` | `build_attn_inp_kv_iswa()` + SWA(逐层数组) | 64 |
| `olmo2.cpp` | 模板二选一 (`std::conditional_t`) + SWA(周期 4) | 77 |
| `plamo3.cpp` | 模板二选一 (`std::conditional_t`) + SWA(周期 8) | 78 |
| `modern-bert.cpp` | `build_attn_inp_no_cache()` + SWA(SYMMETRIC) | 91 |
| `neo-bert.cpp` | `build_attn_inp_no_cache()` | 56 |
| `t5.cpp` | 自注意力 KV + 交叉注意力 + 编码器无 cache | 277 |
| `mistral4.cpp` | 复用 `llama_model_deepseek2::graph`（MLA） | h1397 |
| `qwen3tts.cpp` | 复用 `llama_model_qwen3vl` | h625 |
| `t5encoder.cpp` | 复用 `llama_model_t5::graph<true>` | h1478 |
| `nomic-bert.cpp` | 复用 `llama_model_bert::graph` | h330 |
| `wavtokenizer-dec.cpp` | posnet + convnext，无 `build_attn` | 118 |
| `models.h` | 151 个模型类声明（唯一不建图的源文件） | 147 |

统计：23 个文件用 `build_attn_inp_kv()`，4 个碰窗口（2 个直接用 iswa、2 个用模板二选一），3 个没有 KV cache，6 个不建注意力图。

## 十一、名实不符之一：`using graph = ...`

`models.h` 里有一批类不自己写图，而是用一行别名把别族的图借过来。本课涉及的四处：`llama_model_nomic_bert`（第 330 行，借 `llama_model_bert::graph`）、`llama_model_mistral4`（第 1397 行，借 `llama_model_deepseek2::graph`）、`llama_model_t5encoder`（第 1478 行，借 `llama_model_t5::graph<true>`）、以及用继承复用的 `llama_model_qwen3tts : public llama_model_qwen3vl`（第 625 行）。下面引的是 Mistral4 —— 它是 DeepSeek2 的子类，连超参和张量加载都不重写。

<!-- src: src/models/models.h -->
```cpp
struct llama_model_mistral4 : public llama_model_deepseek2 {
    llama_model_mistral4(const struct llama_model_params & params) : llama_model_deepseek2(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_deepseek2

    using graph = llama_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

## 十二、名实不符之二：只有六行的模型文件

`src/models/mistral4.cpp` 全文如下。它没有 `load_arch_hparams`、没有 `load_arch_tensors`、没有图类 —— 因为三样都从 `llama_model_deepseek2` 继承。同类还有 `qwen3tts.cpp`（3 行，第 3 行是注释：`// llama_model_qwen3tts reuses llama_model_qwen3vl's hparams/tensors/graph logic`）、`t5encoder.cpp`（44 行，只有张量加载，图用 `llama_model_t5::graph<true>`）、`nomic-bert.cpp`（51 行，同理）、`paddleocr.cpp`（继承 `llama_model_ernie4_5`，只重写 `build_arch_graph`）。

<!-- src: src/models/mistral4.cpp -->
```cpp
#include "models.h"

std::unique_ptr<llm_graph_context> llama_model_mistral4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

```

## 十三、草稿模型的声明：两个独立的模型类

草稿模型不是"主模型的一个开关"，而是 `models.h` 里**独立的两个类**：`llama_model_eagle3` 与 `llama_model_dflash`。它们的图都带一个 `template <bool is_enc>` 与一个主模型没有的输入构造函数 `build_inp_embd_enc()`：草稿模型要先跑一遍"编码器"把主模型的特征融合进来。`llama_model_dflash` 还多一个 `graph_dsv4`，继承自 `llama_model_deepseek4::graph`。

<!-- src: src/models/models.h -->
```cpp
struct llama_model_eagle3 : public llama_model_base {
    llama_model_eagle3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dflash : public llama_model_base {
    llama_model_dflash(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    struct graph_dsv4 : public llama_model_deepseek4::graph {
        graph_dsv4(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

## 十四、主模型侧的钩子：导出每层输入

草稿模型需要主模型中间层的隐藏状态。主模型的图为此只加了一行：在层循环开头把 `inpL` 存进 `res->t_layer_inp[il]`。`llm_graph_result::set_outputs()` 会对被请求的层调 `ggml_set_output()`（`llama-graph.cpp:1375-1383`），运行时再由 `llama_context::extract_layer_inputs()` 抽出。本课有 3 个文件写了这一行：`qwen3.cpp:72`、`nanbeige.cpp:106`、`muse-glimmer.cpp:80`。

<!-- src: src/models/qwen3.cpp -->
```cpp
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;
```

## 十五、共享 KV 的图差异：不传 K/V

这是本课验收点的直接证据。草稿模型的注意力只算 Q：权重清单里**没有 `wk` / `wv`**（`gemma4-assistant.cpp:59-60` 只创建 `wq` 与 `wo`），所以 `build_attn` 的第 6、7 个实参（`k_cur` / `v_cur`）只能是 `nullptr`。`build_attn` 内部 `if (k_cur)` / `if (v_cur)` 两段写 cache 的代码因此整段跳过，K/V 直接来自 `mctx_cur->get_k()/get_v()`。

<!-- src: src/models/gemma4-assistant.cpp -->
```cpp
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur_norm);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        ggml_tensor * freq_factors = is_swa ? nullptr : model.layers[il].rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
```

## 十六、共享 KV 的 cache 侧：层张量直接挂上

草稿模型"借"到的 K/V 张量是在 cache 构造时挂上的：`if (share && other)` 命中后执行 `layers.push_back(layer_share)`，草稿的第 `il` 层于是指向主模型某一层的**同一个** K/V 张量（日志里会打印两个指针）。提供层映射的 `share` 回调来自 `llama_model::create_memory()`，它把草稿的 SWA 层映射到主模型的倒数第 2 层、其余层映射到最后 1 层：

<!-- src: src/llama-model.cpp -->
```cpp
                        if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
                            llama_memory_t mem_other = llama_get_memory(cparams.ctx_other);

                            share = [&](int32_t il) {
                                const llama_model * model_other = llama_get_model(cparams.ctx_other);

                                if (hparams.is_swa(il)) {
                                    return llama_model_n_layer(model_other) - 2;
                                }

                                return llama_model_n_layer(model_other) - 1;
                            };
```

## 十七、挂上之后：cache 构造函数里的那一跳

上面那个 `share` 回调在 cache 构造循环里被消费。命中时**不 new 任何张量**，而是把主模型 cache 里的那一层整体 push 进自己的 `layers`：`map_layer_ids[il]` 因此指向一个共享项，`layers.back().il` 被改写成草稿的层号。这就是"共享主模型 KV"在代码里最终发生的那一行。

<!-- src: src/llama-kv-cache.cpp -->
```cpp
        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }
```

---

## 说明

- 本课共覆盖 43 个源文件。其中 36 个是计划里 L2-15 的全部文件，**无一遗漏**。
- 另有 7 个源文件被本课引用并计入覆盖，它们在计划里归别的课：`src/llama-hparams.h` / `src/llama-hparams.cpp`（L2-02）、`src/llama-graph.h` / `src/llama-graph.cpp`（L2-06）、`src/llama-kv-cache.cpp`（L2-04）、`src/llama-model.cpp`（L2-05，L2-04 也已借用）、`src/models/gemma4-assistant.cpp`（L2-14）。本课只引用它们与本课论题直接相关的少量行；各自的主覆盖仍在那七课。
- 本课不讨论任何命令行参数，参数门禁的真值集来自真实二进制的帮助输出。
