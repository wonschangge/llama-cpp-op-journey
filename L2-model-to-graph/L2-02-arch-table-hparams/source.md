<!-- llama-coverage
src/llama-arch.h
src/llama-arch.cpp
src/llama-hparams.h
src/llama-hparams.cpp
src/llama-cparams.h
src/llama-cparams.cpp
-->

# L2-02 · 架构表与超参：模型长什么样的元数据 — 源文件

**一句话**：在 llama.cpp 里，"这是哪个模型"由 `enum llm_arch` 回答，"这个模型长什么样"由 `struct llama_hparams` 回答，"这次运行怎么配置"由 `struct llama_cparams` 回答。三者都在同一批文件里。

本课是 L2 层的地基：后面每一课——加载权重（L2-03）、装配模型（L2-05）、构建计算图（L2-06）——都要先问这三个问题。而且**张量的名字不是拼字符串拼出来的，是查表查出来的**，这一点决定了"新增一个模型架构"到底要改哪几处。

---

## 一、enum llm_arch：152 个家族 + 1 个哨兵

枚举从 `LLM_ARCH_CLIP` 开始（第 14 行），到第 166 行的 `LLM_ARCH_UNKNOWN` 结束，中间共 **153 个成员**。最后那个不是真架构，而是"查不到"的返回值：`llm_arch_from_string()` 找不到匹配时返回它。

<!-- src: src/llama-arch.h -->
```c
    LLM_ARCH_DFLASH,
    LLM_ARCH_NANBEIGE,
    LLM_ARCH_QWEN3TTS,
    LLM_ARCH_POCKETTTS,
    LLM_ARCH_MINIMAX_01,
    LLM_ARCH_HRM_TEXT,
    LLM_ARCH_UNKNOWN,
};

```

## 二、LLM_ARCH_NAMES：枚举值与 GGUF 字符串的对照表

这张 `std::map<llm_arch, const char *>` 共 **153 条**，与枚举成员一一对应。映射到右边的字符串**就是** GGUF 文件里 `general.architecture` 的值 —— 所以它同时也是"模型文件 ↔ 代码枚举"的接口。源码注释还标出一个特例：`clip` 是 dummy，只给 `llama-quantize` 用。

<!-- src: src/llama-arch.cpp -->
```c
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_CLIP,             "clip"             }, // dummy, only used by llama-quantize
    { LLM_ARCH_LLAMA,            "llama"            },
    { LLM_ARCH_LLAMA4,           "llama4"           },
    { LLM_ARCH_DECI,             "deci"             },
    { LLM_ARCH_FALCON,           "falcon"           },
```

## 三、LLM_KV_NAMES：metadata key 的模板

`LLM_KV_NAMES` 共 **243 条**（第 164-431 行）。前面的 `general.*` 与架构无关，后面那些带 `%s` 前缀的才是架构相关的模板 —— `%s` 就是架构名字符串的位置。

<!-- src: src/llama-arch.cpp -->
```c
static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                     "general.type"                          },
    { LLM_KV_GENERAL_ARCHITECTURE,             "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,     "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,                "general.alignment"                     },
    { LLM_KV_GENERAL_FILE_TYPE,                "general.file_type"                     },
    { LLM_KV_GENERAL_SAMPLING_SEQUENCE,        "general.sampling.sequence"             },
```

## 四、enum llm_tensor 与两张派生表

`enum llm_tensor`（`src/llama-arch.h:438-713`）共 **274** 项。它对应两张表：`LLM_TENSOR_NAMES`（`src/llama-arch.cpp:433-707`）**273** 条名字模板；`LLM_TENSOR_INFOS`（`src/llama-arch.cpp:719-998`）**268** 条，每条给 `llm_tensor_layer layer` 与 `ggml_op op` 两个字段。

三个数字并不相等：273 条初始化项里有 1 个重复 key（`LLM_TENSOR_SSM_NORM` 在 493 与 518 行各一次），所以不同的名字只有 272 个；另有 6 个枚举成员在 `LLM_TENSOR_INFOS` 里没有记录 —— `LLM_TENSOR_ATTN_ROT_EMBD`、`LLM_TENSOR_FFN_GATE_EXP`、`LLM_TENSOR_FFN_DOWN_EXP`、`LLM_TENSOR_FFN_UP_EXP`、`LLM_TENSOR_POST_ATTN_NORM`、`LLM_TENSOR_POST_MLP_NORM`。后两个（`src/llama-arch.h:496-497`）在整棵源码树里只出现在枚举定义处。

<!-- src: src/llama-arch.cpp -->
```c
static const std::map<llm_tensor, llm_tensor_info> LLM_TENSOR_INFOS = {
    {LLM_TENSOR_TOKEN_EMBD,                 {LLM_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS}},
    {LLM_TENSOR_POS_EMBD,                   {LLM_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS}},
    {LLM_TENSOR_TOKEN_TYPES,                {LLM_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS}},
    {LLM_TENSOR_HRM_Z_L_INIT,               {LLM_TENSOR_LAYER_INPUT,     GGML_OP_ADD}},
    {LLM_TENSOR_TOKEN_EMBD_NORM,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},  // do the norms on the first layer (not the input layer)
```

## 五、LLM_TN：张量名的构造入口

模型实现里不写字符串，而是 `LLM_TN(arch)` 加一个 `llm_tensor` 枚举值。源码把用法写在注释里：模板 + 后缀 + 层号（+ 专家号），最终的字符串由 `LLM_TN_IMPL::str()`（`src/llama-arch.cpp:1016-1028`）生成。

<!-- src: src/llama-arch.h -->
```c

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_LLAMA);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN_IMPL {
    const llm_arch arch;
```

## 六、llama_hparams：逐层的形状数组

为什么 `n_head` 之类的字段要放数组？因为**每一层的形状可以不一样**。数组长度是 `LLAMA_MAX_LAYERS = 512`（`src/llama-hparams.h:10`），getter 用层号索引；`n_head_kv_arr` 在 GGUF 没写时直接等于 `n_head_arr`（`src/llama-model.cpp:1334`）。

<!-- src: src/llama-hparams.h -->
```c
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    // per-layer expert feed-forward size
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_exp_arr;
    // per-layer top-k expert routing count
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_expert_used_arr;
```

## 七、llama_cparams：一次运行的配置

`llama_cparams` 里既有 `n_ctx` / `n_batch` 这类调用者可调的字段，也有一批 `fused_*` / `auto_*` 开关（运行时自动选择融合算子的结果）。注意第 28-30 行的注释：源码明确说 YaRN 的四个系数**不放进 GGUF**，因为所有现有模型取值相同 —— 这正好从反面说明"进不进 GGUF"是这两个结构的分界线。

<!-- src: src/llama-cparams.h -->
```c
    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
```

## 八、llama-cparams.cpp：整个文件只有 5 行

六个文件里最小的一个。它只导出一个函数：并行序列数上限。上限的值来自头文件里的 `LLAMA_MAX_SEQ`（`src/llama-cparams.h:8`）。

<!-- src: src/llama-cparams.cpp -->
```c
#include "llama-cparams.h"

size_t llama_max_parallel_sequences(void) {
    return LLAMA_MAX_SEQ;
}
```

---

## 说明

- 本课逐字引用 6 个文件：`src/llama-arch.cpp`、`src/llama-arch.h`、`src/llama-hparams.cpp`、`src/llama-hparams.h`、`src/llama-cparams.cpp`、`src/llama-cparams.h`，全部计入覆盖率。
- 场景与正文里用 `文件:行` 形式指路、但**没有逐字引用**的文件有：`src/llama-model.cpp`（1254/1255/1259/1260/1327/1328/1334/43 行）、`src/llama-model-loader.cpp`、`src/llama-context.cpp`（84/246 行）。它们不计入本课覆盖率，其行号由本课作者用 `grep -n` / `sed -n` 实地核对过。
