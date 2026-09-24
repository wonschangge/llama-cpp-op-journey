<!-- llama-coverage
src/llama-quant.cpp
src/llama-quant.h
src/llama-model-saver.cpp
src/llama-model-saver.h
src/llama-adapter.cpp
src/llama-adapter.h
src/llama-grammar.cpp
src/llama-grammar.h
src/llama-chat.cpp
src/llama-chat.h
src/llama-graph.cpp
-->

# L2-09 · 量化、导出与适配器 — 源文件

**一句话**：一个已经训练好的模型，在"进入图"之前会经过三条互相独立的改造路径 ——**量化**改的是权重的数值与类型，**导出**改的是容器（写回 GGUF），**适配器**一个字节的权重都不改，它只是在建图时往图里多插几次矩阵乘。

三条路各有一个入口函数：`llama_model_quantize_impl()`（`src/llama-quant.cpp:913`）、`llama_model_saver::save()`（`src/llama-model-saver.cpp:499`）、`llm_graph_context::build_lora_mm()`（`src/llama-graph.cpp:1514`）。本课把这 10 个文件读一遍，最后回答一个问题：**LoRA 到底是在哪一步被加进图的。**

---

## 一、量化：从一个 GGUF 到另一个 GGUF

量化的入口是 `llama_model_quantize_impl()`。它只接收两个文件名和一个参数结构体，第一件事就是把 `ftype` 换算成默认类型 —— 类型不合法就直接抛错。

注意参数结构体本身**不在这个文件里**：`llama-quant.h` 全文只有一行 `#pragma once`（见第十六节），`llama_model_quantize_params` 定义在 `include/llama.h:434`。

<!-- src: src/llama-quant.cpp -->
```c
static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

```

## 二、★ 一个 ftype 不是一种类型

`llama_tensor_get_type()` 的返回路径有六条：默认类型、词嵌入特例、输出层特例、张量名正则手工覆盖、按类别与层号的混合精度表、以及形状回退。下面这段是后三条的完整逻辑 —— 也就是说，**"混合精度量化"这个说法在这里才落地**。

<!-- src: src/llama-quant.cpp -->
```c
    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (ggml_is_quantized(default_type)) {
        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs.tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n",
                                       __func__, tensor_name.c_str(), ggml_type_name(new_type), ggml_type_name(qtype));
                        new_type = qtype;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual && !params->pure) {
            new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }

    return new_type;
}
```

## 三、★ imatrix：谁重要，是数据不是算法

哪些目标类型**非有 imatrix 不可**，被写死在这二十行里。IQ 家族全部要求；k-quant 家族只有 Q2_K 是例外，而且只在 Q2_K_S 档位下要求。

缺失时的后果不是"质量下降"，而是在收集阶段就抛异常：

<!-- src: src/llama-quant.cpp -->
```c
static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}
```

## 四、imatrix 是在哪里被用掉的

每个张量化之前，代码会按 `ne[0]*ne[2]` 的形状去 imatrix 表里找对应的向量，尺寸不符（且不是词嵌入）就直接抛错；找不到而目标类型又必须有它，同样抛错。这段还顺手挡住了"对已经量化过的张量再量化"这种操作。

<!-- src: src/llama-quant.cpp -->
```c
                const float * imatrix = nullptr;
                if (imatrix_data) {
                    auto it = imatrix_data->find(tm.remapped_imatrix_name);
                    if (it == imatrix_data->end()) {
                        LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (!tensor_name_match_token_embd(tensor->name)) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                        int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tm.requires_imatrix) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }
```

## 五、逐行量化：chunk 与专家边界

一张权重矩阵被按行切成 chunk 交给线程；行号是跨专家全局连续的，但每个专家有自己的 imatrix 切片，所以 **chunk 不允许跨越专家边界**。真正的量化调用是 `ggml_quantize_chunk()`，它的产物还会被逐块校验。

<!-- src: src/llama-quant.cpp -->
```c
// note: chunks never cross an expert boundary since each expert has its own imatrix slice
static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t first_row, int64_t nrows, int64_t nrows_per_expert, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    const size_t row_size = ggml_row_size(new_type, n_per_row);

    auto imatrix_for_row = [=](int64_t row_global) {
        return imatrix ? imatrix + (row_global / nrows_per_expert) * n_per_row : nullptr;
    };

    if (nthread < 2) {
        // single-thread
        size_t new_size = 0;
        for (int64_t row = 0; row < nrows;) {
            const int64_t row_global = first_row + row;
            const int64_t this_nrow  = std::min(nrows - row, nrows_per_expert - row_global % nrows_per_expert);
            void * this_data = (char *) new_data + row * row_size;
            size_t this_size = ggml_quantize_chunk(new_type, f32_data + row * n_per_row, this_data, 0, this_nrow, n_per_row, imatrix_for_row(row_global));
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                throw std::runtime_error("quantized data validation failed");
            }
            new_size += this_size;
            row += this_nrow;
        }
        return new_size;
```

## 六、导出：llama_model_saver 的三分工

导出侧的类很小：登记元数据（`add_kv` 的一族重载）、登记张量（`add_tensor`）、写盘（`save`）。`gguf_ctx` 就是"还没落盘的文件"，所有登记都进它。

<!-- src: src/llama-model-saver.h -->
```c
struct llama_model_saver {
    struct gguf_context * gguf_ctx = nullptr;
    const bool gguf_ctx_owned;
    const struct llama_model * model;
    const struct LLM_KV llm_kv;

    llama_model_saver(const struct llama_model * model);
    llama_model_saver(enum llm_arch arch, struct gguf_context * gguf_ctx);
    ~llama_model_saver();

    void add_kv(enum llm_kv key, uint32_t     value);
    void add_kv(enum llm_kv key, int32_t      value);
    void add_kv(enum llm_kv key, uint64_t     value);
    void add_kv(enum llm_kv key, float        value);
    void add_kv(enum llm_kv key, bool         value);
    void add_kv(enum llm_kv key, const char * value);

    [[noreturn]]
    void add_kv(enum llm_kv key, char value); // needed to make the template below compile

    template <typename Container>
    void add_kv(enum llm_kv key, const Container & value, bool per_layer = false);

    void add_kv(enum llm_kv key, const std::vector<std::string> & value);

    void add_tensor(const struct ggml_tensor * tensor);

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string & path_model);
    void save(FILE * file);
};
```

## 七、add_tensor 与 save 的实现

`add_tensor` 只做两件事：已存在同名张量就返回（只有三个 rope 相关张量允许重名），否则把指针登记进 `gguf_ctx`。**字节仍然在模型自己的 buffer 里。**

真正的落盘是 `save()`：一行调用，把上下文写成文件。

<!-- src: src/llama-model-saver.cpp -->
```c
void llama_model_saver::add_tensor(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    if (gguf_find_tensor(gguf_ctx, tensor->name) >= 0) {
        const std::string tensor_name = tensor->name;
        GGML_ASSERT(
            tensor_name == "rope_freqs.weight" || tensor_name == "rope_factors_long.weight" ||
            tensor_name == "rope_factors_short.weight"); // FIXME
        return;
    }
    gguf_add_tensor(gguf_ctx, tensor);
//>> ---- src/llama-model-saver.cpp:499-505 ----
void llama_model_saver::save(const std::string & path_model) {
    gguf_write_to_file(gguf_ctx, path_model.c_str(), false);
}

void llama_model_saver::save(FILE * file) {
    gguf_write_to_file_ptr(gguf_ctx, file, false);
}
```

## 八、★ LoRA 的图内插入点：build_lora_mm

这一段是本课验收点的直接证据。`build_lora_mm()` 的骨架是：

```text
res = ggml_mul_mat(ctx0, w, cur)        // 基座那一次，w 只读
for lora in *loras:                     // 每一套挂着的 adapter
    lw = lora.get_weight(w)             // 按 w 的名字查表
    if lw == nullptr: continue          // 这套 adapter 不管这个权重
    ab = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur))
    ab = ggml_scale(ctx0, ab, scale)
    res = ggml_add(ctx0, res, ab)       // 并回主干
```

**基座张量 `w` 从头到尾没有被改写。** 这段代码属于 `src/llama-graph.cpp`，计划里把它归给 L2-06（计算图骨架）；本课追加引用它，理由见文末说明。

<!-- src: src/llama-graph.cpp -->
```c
ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}
```

## 九、适配器的数据面

一个 adapter 在内存里就是 `ab_map`（基座张量名 -> 一对小矩阵）加一个 `alpha`。`get_scale()` 把 rank 从 `b` 的形状里读出来；`get_n_nodes()` 则把"图会多出多少节点"直接写成 `ab_map.size() * 6u`，注释列出了那 6 个节点。

<!-- src: src/llama-adapter.h -->
```c
//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};
```

## 十、适配器是怎么装进来的：靠名字后缀配对

加载 adapter 文件时，每个张量名必须以 `.lora_a` 或 `.lora_b` 结尾；去掉后缀剩下的就是**基座里的张量名**。两者在 `ab_map` 里配成一对。不认识的结尾直接抛错；`_norm.weight` 被显式跳过 —— 源码注释说明多数 adapter 不依赖它。

<!-- src: src/llama-adapter.cpp -->
```c
    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (str_endswith(name, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
```

## 十一、运行时按名字查表

建图时 `build_lora_mm()` 调用的就是这一个函数：拿基座张量的名字去 `ab_map` 里找，找不到返回 `nullptr`。**"这套 adapter 作用于哪些权重"完全是运行时按名字决定的**，不需要在加载时改写任何权重。

<!-- src: src/llama-adapter.cpp -->
```c
llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    const std::string name(w->name);

    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}
```

## 十二、grammar：约束是采样的事，不是图的事

`llama_grammar_apply_impl()` 遍历采样候选，把规则不接受的 token 的 logit 置为 `-INFINITY`；`llama_grammar_accept_token()` 则在采样之后，用**同一个 piece** 推进规则栈。两个函数都不碰 ggml 张量。

<!-- src: src/llama-grammar.cpp -->
```c
void llama_grammar_apply_impl(const struct llama_grammar & grammar, llama_token_data_array * cur_p) {
    GGML_ASSERT(grammar.vocab != nullptr);

    if (grammar.awaiting_trigger) {
        return;
    }

    bool allow_eog = false;
    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            allow_eog = true;
            break;
        }
    }

    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> candidates_decoded;
    candidates_decoded.reserve(cur_p->size);

    llama_grammar_candidates candidates_grammar;
    candidates_grammar.reserve(cur_p->size);

    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id      = cur_p->data[i].id;
        const std::string & piece = grammar.vocab->token_to_piece(id);

        if (grammar.vocab->is_eog(id)) {
            if (!allow_eog) {
                cur_p->data[i].logit = -INFINITY;
            }
        } else if (piece.empty() || piece[0] == 0) {
            cur_p->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(piece, grammar.partial_utf8));
            candidates_grammar.push_back({ i, candidates_decoded.back().first.data(), candidates_decoded.back().second, id });
        }
    }

    const auto rejects = llama_grammar_reject_candidates(grammar.rules, grammar.stacks, candidates_grammar);
    for (const auto & reject : rejects) {
        cur_p->data[reject.index].logit = -INFINITY;
    }
}
//>> ---- src/llama-grammar.cpp:1472-1500 ----
void llama_grammar_accept_token(struct llama_grammar & grammar, llama_token token, const std::string & piece) {
    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece, grammar.partial_utf8);
    const auto & code_points = decoded.first;

    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar.stacks.size());

    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            continue;
        }

        const llama_grammar_element * pos = stack.back();

        if (pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
            if (llama_grammar_match_token(pos, token)) {
                llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    new_stack.push_back(pos + 1);
                }
                llama_grammar_advance_stack(grammar.rules, new_stack, stacks_new);
            }
        } else {
            llama_grammar_stacks current_stacks = {stack};

            for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
                llama_grammar_stacks next_stacks;

```

## 十三、grammar 的数据结构

规则被编译成一串 `llama_grammar_element`：字符、字符区间、取反、规则引用、以及按 token id 匹配的元素。`llama_grammar` 本体持有规则表与一组栈，外加"懒 grammar"的触发词缓冲。

<!-- src: src/llama-grammar.h -->
```c

// grammar element type
enum llama_gretype {
    // end of rule definition
    LLAMA_GRETYPE_END            = 0,

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,
```

## 十四、对话模板：一张名字表 + 一条 if-else 链

名字表里每一项把模板名映射到枚举值（共 54 项）。模板字符串本身由 `llama_model_chat_template()`（声明在 `include/llama.h:647`）从模型里取出；名字对不上时，`llm_chat_detect_template()` 会退化成对模板字符串做子串特征匹配。

应用模板就是一条很长的分支链，下面是第一条分支（chatml）的原文。

<!-- src: src/llama-chat.cpp -->
```c
static const std::map<std::string, llm_chat_template> LLM_CHAT_TEMPLATES = {
    { "chatml",            LLM_CHAT_TEMPLATE_CHATML            },
    { "llama2",            LLM_CHAT_TEMPLATE_LLAMA_2           },
    { "llama2-sys",        LLM_CHAT_TEMPLATE_LLAMA_2_SYS       },
    { "llama2-sys-bos",    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS   },
    { "llama2-sys-strip",  LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP },
    { "mistral-v1",        LLM_CHAT_TEMPLATE_MISTRAL_V1        },
    { "mistral-v3",        LLM_CHAT_TEMPLATE_MISTRAL_V3        },
    { "mistral-v3-tekken", LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN },
    { "mistral-v7",        LLM_CHAT_TEMPLATE_MISTRAL_V7        },
    { "mistral-v7-tekken", LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN },
    { "phi3",              LLM_CHAT_TEMPLATE_PHI_3             },
    { "phi4",              LLM_CHAT_TEMPLATE_PHI_4             },
    { "falcon3",           LLM_CHAT_TEMPLATE_FALCON_3          },
    { "zephyr",            LLM_CHAT_TEMPLATE_ZEPHYR            },
    { "monarch",           LLM_CHAT_TEMPLATE_MONARCH           },
    { "gemma",             LLM_CHAT_TEMPLATE_GEMMA             },
    { "orion",             LLM_CHAT_TEMPLATE_ORION             },
//>> ---- src/llama-chat.cpp:244-258 ----
int32_t llm_chat_apply_template(
    llm_chat_template tmpl,
    const std::vector<const llama_chat_message *> & chat,
    std::string & dest, bool add_ass) {
    // Taken from the research: https://github.com/ggml-org/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == LLM_CHAT_TEMPLATE_CHATML) {
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7 || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN) {
```

## 十五、模板枚举

`llm_chat_template` 有 56 个值（含末尾的 UNKNOWN）；名字表覆盖其中 54 个。枚举第一项就是 chatml —— 也就是上一节引用的那条分支。

<!-- src: src/llama-chat.h -->
```c
enum llm_chat_template {
    LLM_CHAT_TEMPLATE_CHATML,
    LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,
    LLM_CHAT_TEMPLATE_MISTRAL_V1,
    LLM_CHAT_TEMPLATE_MISTRAL_V3,
    LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN,
    LLM_CHAT_TEMPLATE_MISTRAL_V7,
    LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN,
    LLM_CHAT_TEMPLATE_PHI_3,
    LLM_CHAT_TEMPLATE_PHI_4,
    LLM_CHAT_TEMPLATE_FALCON_3,
    LLM_CHAT_TEMPLATE_ZEPHYR,
    LLM_CHAT_TEMPLATE_MONARCH,
```

## 十六、llama-quant.h 全文

这个头文件全文只有一行。量化模块的对外接口不在它里面：参数结构体与入口函数都声明在 `include/llama.h`（`llama_model_quantize_params` 在第 434 行，`llama_model_quantize_default_params` 在第 477 行）。把它列进覆盖是因为它确实是这个模块的源文件 —— 引用它也是引用"事实"。

<!-- src: src/llama-quant.h -->
```c
#pragma once
```

---

## 说明

- 本课按计划覆盖 10 个文件，全部计入覆盖率：`src/llama-quant.{cpp,h}`、`src/llama-model-saver.{cpp,h}`、`src/llama-adapter.{cpp,h}`、`src/llama-grammar.{cpp,h}`、`src/llama-chat.{cpp,h}`。
- **额外引用 1 个清单外文件**：`src/llama-graph.cpp`（计划里归 L2-06「计算图骨架」）。理由：本课验收点是"能说出 LoRA 是在哪一步被加进图的"，而图内插入点 `llm_graph_context::build_lora_mm()` 只在 `src/llama-graph.cpp:1514` 定义。不引用它，这个结论就只能靠推断 —— 而本仓库的红线是"每个断言都必须能从源码看出来"。覆盖度门禁取的是并集，追加引用不会让任何文件失配（`llama-graph.cpp` 已由 L2-06 覆盖）。
- `src/llama-quant.h` 全文 1 行（`#pragma once`），本课在 source.md 第十六节逐字引用，不计为"未引用但声明"。
