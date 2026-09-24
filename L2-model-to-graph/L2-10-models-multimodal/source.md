<!-- llama-coverage
src/models/clip.cpp
src/models/models.h
src/models/qwen4exp.cpp
src/models/gemma4.cpp
tools/mtmd/models/gemma4v.cpp
tools/mtmd/mtmd-helper-common.h
src/models/deepseek4.cpp
tools/mtmd/clip-graph.h
tools/mtmd/clip.cpp
tools/mtmd/mtmd.cpp
src/models/pockettts.cpp
-->

# L2-10 · 模型家族（一）：多模态与视觉编码 — 源文件

**一句话**：llama.cpp 里"多模态"不是一种新的图引擎 —— 视觉塔（或音频编码器）是**另一张 ggml 图**，它算完之后只留下**一个 F32 张量**；这个张量被塞进 `llama_batch` 的 `embd` 字段，当作"已经算好的 embedding"喂进**普通文本图**的入口。

本课覆盖的 6 个文件是**静态扫描**分出来的：图构建代码里出现了 `clip_` / `vision` / `mmproj` / `image_` 之一。**分组依据是命中了这些图原语，不是对文件内容的断言。**实测结果（见下方第一节）是：8 处命中全部落在注释或元数据键名上，`clip_` 在 6 个文件里出现 **0 次**。因此本课既讲"文本图这一侧怎么接多模态"，也如实说明"谁不是多模态"。

---

## 一、本课的分组方法（先读这一节）

本课覆盖的 6 个文件**不是按内容选的**，而是**静态扫描**的结果：凡是图构建代码里出现了 `clip_` / `vision` / `mmproj` / `image_` 之一的 `src/models/*.cpp|h`，就被分到"多模态与视觉编码"这一课。**分组依据是"命中了这些图原语"，不是"这个文件就是多模态模型"。**

我把这 6 个文件逐个读了一遍，实测结果如下。行数来自 `wc -l`；README 的文件表由生成器按另一种口径计数（`split("\n")` 的元素个数），每个文件会多 1 —— 两处数字对得上，只是口径不同：

| 文件 | 行数 | 命中处（原文所在行） | 真实主题 | 是多模态吗 |
|---|---|---|---|---|
| `src/models/clip.cpp` | 18 | 第 3 行注释里的 `mmproj` | CLIP 的**量化存根**：三个 `[[noreturn]]` 函数，没有任何图 | **不是**（真正的运行时在 `tools/mtmd/clip.cpp`） |
| `src/models/models.h` | 2662 | 第 399 行注释里的 `mmproj` | **公共声明头**：151 个 `struct llama_model_*` 与图基类声明 | **不是**（是被所有模型共用的头文件） |
| `src/models/deepseek4.cpp` | 1502 | 第 162 行注释里的 `vision` / `image` | DeepSeek-V4 文本图（MoE + 压缩注意力 + MTP） | 部分：**有 vision variant 的钩子** |
| `src/models/gemma4.cpp` | 498 | 第 23 行注释里的 `"vision"` | Gemma 4 文本图（SWA + 共享 KV + per-layer 嵌入） | 部分：**有图像输入的分支** |
| `src/models/pockettts.cpp` | 146 | 第 4 行注释里的 `mmproj` | PocketTTS 的文本主干（无 lm_head） | 部分是：**音频侧**，不是视觉 |
| `src/models/qwen4exp.cpp` | 1294 | 第 90、1072、1073 行的 `ple_image_token_id` | Qwen4 文本图（线性注意力 + PLE n-gram 嵌入） | 部分：**有图像批次的分支** |

两条要如实说清楚的事实：

1. **8 处命中里，5 处出现在注释中**（`clip.cpp:3`、`models.h:399`、`deepseek4.cpp:162`、`gemma4.cpp:23`、`pockettts.cpp:4`），**剩下 3 处都在 `qwen4exp.cpp`，且都是同一个元数据键名 `ple_image_token_id`**（第 90、1072、1073 行）。**没有一处是图原语调用。**
2. 扫描模式里的 `clip_` 在这 6 个文件里出现 **0 次**。6 个文件全部只 `#include "models.h"`（`clip.cpp` / `gemma4.cpp` / `pockettts.cpp` / `qwen4exp.cpp` 在第 1 行，`deepseek4.cpp` 在第 2 行），**没有一个是视觉编码器**。

**所以本课讲两件事**：（a）多模态在**文本图这一侧**是怎么被接进来的（这 6 个文件里能读到）；（b）视觉塔本身长什么样 —— 它在 `tools/mtmd/`，**不在本仓库的覆盖域内**，本课引用它只作对照，并在本节末尾与脚注里明确标注**不计入覆盖率**。

## 二、clip.cpp：一个 [[noreturn]] 的存根

这是本课最短的文件，也是最能说明问题的一个。它给出三条 `GGML_ABORT`，并在注释里写明自己的身份（`Stub to allow llama-quantize to open mmproj GGUFs`），最后一行直接指路：**CLIP 在 llama 分发路径里没有推理图，运行时在 `tools/mtmd/clip.cpp`**。

<!-- src: src/models/clip.cpp -->
```c
// Stub to allow llama-quantize to open mmproj GGUFs

[[noreturn]]
void llama_model_clip::load_arch_hparams(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_hparams should not be called");
}

[[noreturn]]
void llama_model_clip::load_arch_tensors(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_tensors should not be called");
}

[[noreturn]]
std::unique_ptr<llm_graph_context> llama_model_clip::build_arch_graph(const llm_graph_params &) const {
    GGML_ABORT("CLIP has no inference graph via llama_model dispatch; runtime lives in tools/mtmd/clip.cpp");
}
```

## 三、视觉塔的契约（tools/mtmd/clip-graph.h）

视觉塔的公共基类 `clip_graph` 把两件事写进了注释：ViT 的通用构建函数 `build_vit`，以及输入构建函数 `build_inp()` 的返回形状 —— **`[n_embd, n_patches]`**。另外，投影后的宽度 `n_mmproj_embd` 是基类的一个字段（`clip-graph.h:43`）。

> 这一节的文件在 `tools/`，**不在覆盖域，不计入覆盖率**。

<!-- src: tools/mtmd/clip-graph.h -->
```c
    // if your model has specific features, you should probably duplicate this function
    ggml_tensor * build_vit(
                ggml_tensor * inp,
                int64_t n_pos,
                norm_type norm_t,
                ffn_op_type ffn_t,
                ggml_tensor * learned_pos_embd,
                std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos,
                const build_vit_opts & opts = {});

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches]
    ggml_tensor * build_inp();
```

## 四、视觉塔图的出口：最后一个节点就是 embedding（tools/mtmd/clip.cpp）

视觉塔算完之后，代码并不去"找名字"，而是**取这张图的最后一个节点**当输出，并用它的 `ne[1]` 校验 token 数。这是"视觉塔输出一个张量"这句话最直接的证据。

> 同样在 `tools/`，不计入覆盖率。

<!-- src: tools/mtmd/clip.cpp -->
```c
    // the last node is the embedding tensor, code2wav has no out_embd
    ggml_tensor * embeddings = params->out_embd ? ggml_graph_node(gf, -1) : nullptr;

    if (embeddings != nullptr) {
        // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
        const int n_tokens_out = embeddings->ne[1];
        const int expected_n_tokens_out = clip_n_output_tokens(ctx, &imgs.entries[0]);
        if (n_tokens_out != expected_n_tokens_out) {
            LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
            GGML_ABORT("Invalid number of output tokens");
        }

        LOG_DBG("%s: output embedding shape [%d, %d, %d]\n", __func__,
            (int)embeddings->ne[0], (int)embeddings->ne[1], (int)embeddings->ne[2]);

        // copy output to user buffer if provided
```

## 五、两个 n_embd 必须相等（tools/mtmd/mtmd.cpp）

mtmd 在建上下文时就断言：mmproj 的输出宽度必须等于文本模型的输入宽度。不相等就直接抛错（提示语就是"你多半用错了 mmproj"）。

> 同样在 `tools/`，不计入覆盖率。

<!-- src: tools/mtmd/mtmd.cpp -->
```c
        // since we already validate n_embd of vision and audio mmproj,
        // we can safely assume that they are the same
        int n_embd_clip = clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a);
        if (n_embd_text > 0 && n_embd_text != n_embd_clip) {
            throw std::runtime_error(string_format(
                "mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                "hint: you may be using wrong mmproj\n",
                n_embd_text, n_embd_clip));
        }
```

## 六、拷成一条扁平缓冲（tools/mtmd/mtmd.cpp）

视觉塔的输出张量被拷成一条扁平 `std::vector<float>`，长度 = 宽度 x token 数。这条缓冲随后就是 `llama_batch.embd` 指向的内存。

> 同样在 `tools/`，不计入覆盖率。

<!-- src: tools/mtmd/mtmd.cpp -->
```c
static int32_t mtmd_encode_impl(mtmd_context * ctx, const mtmd_image_tokens * image_tokens, std::vector<float> & out_embd) {
    clip_ctx * ctx_clip = ctx->ctx_v;
    if (!ctx_clip) {
        LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
        return 1;
    }

    int n_embd_out = ctx->n_embd_out();
    auto n_tokens_out = image_tokens->n_tokens();
    out_embd.resize((size_t)n_embd_out * n_tokens_out);

    if (image_tokens->is_placeholder()) {
        LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
        return 1;
    }

    bool ok = clip_image_batch_encode(
        ctx_clip,
        ctx->n_threads,
        &image_tokens->batch_f32,
        out_embd);

    return ok ? 0 : 1;
}

```

## 七、音频侧：pockettts 的文本主干

PocketTTS 走的是同一条思路的另一半：llama 侧只跑"文本"主干，**音频 latent 由 mmproj 里的 flow net 生成**。主干甚至有输出头也不产 logits —— 源码直接复用了嵌入表，只为让采样器还能跑起来。

<!-- src: src/models/pockettts.cpp -->
```c
// backbone of the pocket-tts CALM pipeline: the "text" side of a flow language model.
// it has no lm_head, the audio latents are produced by the flow net inside the mmproj

void llama_model_pockettts::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer()) {
        case 6:  type = LLM_TYPE_109M; break;
        case 24: type = LLM_TYPE_335M; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_pockettts::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    // no output head, the logits are unused; reuse the embedding table so a sampler can still run
    output        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
```

## 八、gemma4：per-layer 输入的多模态分支

Gemma 4 的 per-layer 嵌入需要按 token 查表。图像批次没有 token id，于是源码走"多模态 embedding 路径"：用 padding token（ID=0）的嵌入，并保留了"这未必与 transformers 实现一致"的 TODO。

<!-- src: src/models/gemma4.cpp -->
```c
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation
        const int64_t embd_size = model.per_layer_tok_embd->ne[0];  // n_embd_per_layer * n_layer

        // Extract and dequantize padding token embedding (row 0)
        ggml_tensor * padding = ggml_view_1d(ctx0, model.per_layer_tok_embd, embd_size, 0);
        inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);

        // Reshape to [n_embd_per_layer, n_layer, 1]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }
    return inp_per_layer;
```

## 九、qwen4exp：图像 token 的 id 来自元数据

Qwen4 的图像占位 id 是一个**可选**元数据键：老文件里没有它，于是回退到 EOS token。这就是第 3 幕那个 `img_tok` 的来历。

<!-- src: src/models/qwen4exp.cpp -->
```c
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
```

## 十、deepseek4：vision variant 的 MoE 路由偏置

DeepSeek-V4 在每个 MoE 层上多声明了一个**可选**张量：vision variant 的路由偏置。图构建时用 `ubatch.embd` 判断"这一批是不是媒体输入"，是就换用它。

<!-- src: src/models/deepseek4.cpp -->
```c
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, flags | TENSOR_NOT_REQUIRED);
```

---

## 说明

- **覆盖声明**：本课覆盖域内（`src/`）的文件是 6 个 —— `src/models/clip.cpp`、`src/models/deepseek4.cpp`、`src/models/gemma4.cpp`、`src/models/pockettts.cpp`、`src/models/qwen4exp.cpp`、`src/models/models.h`。它们计入覆盖率。
- **不计入覆盖率**：本课另外引用了 `tools/mtmd/` 下的 5 个文件（`clip.cpp`、`clip-graph.h`、`mtmd.cpp`、`mtmd-helper-common.h`、`models/gemma4v.cpp`）作为"视觉塔真正在哪里"的对照。按 `tools/repo_universe.py` 的 `CORE_PATTERNS`，`tools/` 不在覆盖域内 —— 这些引用**被引用但不计入覆盖率**，此处明确声明，不虚报。
- **分组来源**：本课 6 个文件来自静态扫描（图构建代码里出现 `clip_` / `vision` / `mmproj` / `image_` 之一）。实测：8 处命中中 5 处在注释里，3 处是 `qwen4exp.cpp` 的同一个元数据键名，**没有一处是图原语调用**；`clip_` 在 6 个文件里出现 0 次。因此本课不声称"这 6 个文件都是多模态模型"。
- 第 5~7 幕的形状链逐字引自 `tools/mtmd` 的代码；本课只解读形状流转，视觉塔的完整实现（40+ 个模型文件）不在本课范围内。
