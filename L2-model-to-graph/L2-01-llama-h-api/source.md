<!-- llama-coverage
include/llama.h
src/llama.cpp
include/llama-cpp.h
src/llama-ext.h
-->

# L2-01 · llama.h 公共 API 全景 — 源文件

**一句话**：L1-06 结束时，权重还只是一只 GGUF 文件；L2 层要把它变成"待执行的图"。这一层的第一步，是先认清**谁在对外承诺什么** —— 那份承诺写在 `include/llama.h` 里。

本课的视角是"调用方看到的世界"：公共头文件里有哪些不透明类型、哪些参数结构体，以及一个程序从启动到出第一个 token 必须按什么顺序调用它们。内部实现散落在哪些文件，本课只点到位置，展开留给 L2-05 与 L2-07。

---

## 一、契约的边界：llama.h 的分段结构

`llama.h` 用 `extern "C"` 把整个文件包起来（51-53），然后立刻声明四个**不透明类型**。从这一行开始，调用方与实现之间只剩指针：`llama_model` / `llama_context` / `llama_sampler` 的字段一个字都不在公共头文件里，**改内部布局不会破坏 ABI**。

本课覆盖的四个文件，分工是：

| 文件 | 行数 | 角色 |
|---|---|---|
| `include/llama.h` | 1646 | 唯一对外契约（241 处 `LLAMA_API` 声明） |
| `include/llama-cpp.h` | 30 | C++ RAII 包装，开头 `#error` 拒绝 C |
| `src/llama.cpp` | 620 | 契约的实现入口与装配处 |
| `src/llama-ext.h` | 134 | staging 扩展头，允许破坏性变更 |


<!-- src: include/llama.h -->
```cpp
#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;

    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;
```

## 二、★ 三段式对象：三个不透明指针，三段生命周期

三个类的创建与释放函数**不对称**，这是读公共 API 时最容易踩的地方：

| 对象 | 创建 | 释放 |
|---|---|---|
| `struct llama_model` | `llama_model_load_from_file`（517） | `llama_model_free`（542） |
| `struct llama_context` | `llama_init_from_model`（544） | `llama_free`（554） |
| `struct llama_sampler` | `llama_sampler_chain_init`（1355） | `llama_sampler_free`（1350） |

默认值不写在结构体里，而是由四个 `default_params` 函数给出（474-477）——这样默认值可以随实现版本变化，而头文件的字段布局保持稳定。

<!-- src: include/llama.h -->
```cpp
    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLAMA_API struct llama_model_params          llama_model_default_params(void);
    LLAMA_API struct llama_context_params        llama_context_default_params(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);
```

## 三、参数结构体族

四个参数结构体按声明顺序排开：模型加载（314）、上下文创建（360）、离线量化（434）、采样链（457）。它们都是**按值传递**的，所以 `llama_context_params` 里专门写了一句注释：把 bool 集中放在结构体末尾，避免按值拷贝时错位。

注意 `n_ctx` 这类字段只是"请求值"：创建之后要用 `llama_n_ctx(ctx)`（570）一族查询实际值。

<!-- src: include/llama.h -->
```cpp
    struct llama_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs
        enum llama_load_mode  load_mode;  // how to load the model

        enum llama_lazy_mode lazy_mode; // on-demand reading of tensors marked by the arch

        // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
```

## 四、★ 六步调用顺序

这是本课的验收点。一个最小可用的 llama.cpp 程序按这个顺序调用：

```text
1  llama_backend_init()            进程级，一次
2  llama_model_load_from_file()    读 GGUF -> llama_model
3  llama_init_from_model()         建 llama_context（含 memory）
4  llama_tokenize()                文本 -> token id
5  llama_decode()                  跑图，产出 logits
6  llama_sampler_sample()          从 logits 选 token
```

代码区把这六处声明从 1646 行头文件的六个段落里抽出来拼在一起；区间分隔行标出了各自的真实行号。

<!-- src: include/llama.h -->
```cpp
    LLAMA_API void llama_backend_init(void);
//>> ---- include/llama.h:517-519 ----
    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
              struct llama_model_params   params);
//>> ---- include/llama.h:544-546 ----
    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);
//>> ---- include/llama.h:1180-1187 ----
    LLAMA_API int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);
//>> ---- include/llama.h:998-1000 ----
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);
//>> ---- include/llama.h:1548-1548 ----
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
```

## 五、输入与记忆：llama_batch 与 memory

`llama_batch` 的七个字段都是并行数组，长度统一为 `n_tokens`；`n_seq_id` 与 `seq_id` 是二维的，所以一个 batch 能同时推进多条序列。

`llama_decode` 的注释写明它**必须有 memory**（987），而 `llama_encode` 恰好相反（977）。memory 是 KV cache 之上的抽象：`llama_get_memory(ctx)`（585）取句柄，`llama_memory_seq_rm` / `seq_cp` / `seq_add`（756 / 765 / 780）按序列增删。

<!-- src: include/llama.h -->
```cpp
    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } llama_batch;
```

## 六、★ src/llama.cpp 是聚合入口

它只有 620 行，却 include 了全部内部模块。它的角色是**把公共契约翻译成内部调用序列**：进程初始化、模型加载编排、chat 模板、分片路径。真正的重活分散在别的文件里 ——`llama_init_from_model` 在 `src/llama-context.cpp:3739`，`llama_decode` 在 `src/llama-context.cpp:4326`，`llama_tokenize` 在 `src/llama-vocab.cpp:4496`，`llama_sampler_sample` 在 `src/llama-sampler.cpp:895`。

<!-- src: src/llama.cpp -->
```cpp
#include "llama.h"

#include "llama-impl.h"
#include "llama-version.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"
```

## 七、C++ RAII 包装 llama-cpp.h

30 行、四个 deleter、四个 `unique_ptr` 别名。它把"记得释放"从文档约定变成类型约束。注意 `llama_context_deleter` 调的是 `llama_free`（llama.h:554），而不是某个 `llama_context_free`。

<!-- src: include/llama-cpp.h -->
```cpp
#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "llama.h"

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora * adapter) { llama_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<llama_model, llama_model_deleter> llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
typedef std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter> llama_adapter_lora_ptr;
```

## 八、staging 头 llama-ext.h

开头三行自述：这是 staging header，允许破坏性变更，一切都算 WIP（3-5）。它声明了建图入口 `llama_graph_reserve`（13）与量化状态 API（22-28）。

第 5 行那句自律 ——"尽量别在代码库其它地方 include 这个头" —— **实测没做到**：`src/` 下已有 4 个文件 include 了它。源码注释不一定是真的，凡结论必实测。

<!-- src: src/llama-ext.h -->
```cpp
#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);
```

---

## 说明

- 本课引用 `include/llama.h`、`include/llama-cpp.h`、`src/llama.cpp`、`src/llama-ext.h` 四个文件，全部计入覆盖率。
- 散文与图示里提到的内部定义位置（`src/llama-context.cpp:3739`、`src/llama-context.cpp:4326`、`src/llama-vocab.cpp:4496`、`src/llama-sampler.cpp:895`、`src/llama-quant.cpp:847`）由 `grep -n` 实测确认，但**不构成本课的代码引用**，故不计入本课覆盖率。
