<!-- llama-coverage
src/llama-batch.h
src/llama-batch.cpp
src/llama-model.h
src/llama-model.cpp
src/llama-impl.h
src/llama-impl.cpp
-->

# L2-05 · 批、解码参数与模型装配 — 源文件

**一句话**：调用方交给 `llama_decode` 的是一个 `llama_batch`，但计算图不是按 batch 建的 —— 中间隔着一层切分：`llama_batch_allocr` 把 batch 切成若干 `llama_ubatch`，**每个 ubatch 建一次图、算一次**。所以"怎么切"直接决定"图长什么样、要建几张图"。

本课同时收束 L2 的另一条线：**模型装配**。GGUF 里的 arch 决定 `llama_model_create()` new 出哪一个模型类；基类读共性元数据，架构子类读自己的超参与张量表；层到设备/buffer 的分配在 `load_tensors()` 里一次性算好。

阅读顺序：先看 batch 怎么被补齐（第 2 幕），再看 ubatch 的字段（第 3 幕），然后是三种切法与 `split_equal` 的真实判据（第 4-6 幕），最后是装配与设备分配（第 8-9 幕）。

---

## 一、batch：只有 token 是必给的

`llama_batch_allocr::init()` 先校验（token 范围、seq_id 范围），再**补全缺失字段**。补出来的默认值决定了后面"序列集"的形态：`seq_id` 指向默认序列 0、`pos` 从记忆模块里该序列的最大位置往后接。

注意 `pos` 的补法：源码为每个序列维护一个游标 `p0[s]`，每给一个 token 就 `p0[seq_id] = pos[i] + 1`，所以**同一个序列在 batch 里的位置必须连续递增** —— 这条约束是后面 `split_equal` 判据二的前提。

<!-- src: src/llama-batch.cpp -->
```c
    if (!batch.n_seq_id) {
        n_seq_id.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            n_seq_id[i] = seq_id_0.size();
        }
        batch.n_seq_id = n_seq_id.data();
    }

    if (!batch.seq_id) {
        seq_id.resize(batch.n_tokens + 1);
        seq_id[batch.n_tokens] = NULL;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            seq_id[i] = seq_id_0.data();
        }
        batch.seq_id = seq_id.data();
    }
```

## 二、★ struct llama_ubatch：一个不变量串起所有字段

按用途分五组读：计数（`n_tokens` / `n_seq_tokens` / `n_seqs` / `n_seqs_unq` / `n_pos`）、数据（`token` / `embd` / `pos`）、归属（`n_seq_id` / `seq_id` / `seq_id_unq` / `seq_idx`）、输出（`output`）、标记（`b_equal_seqs`）。

**不变量**：`n_tokens = n_seq_tokens * n_seqs`。字段注释里就写着 `total tokens (n_seq_tokens * n_seqs)`，`ubatch_add()` 也是按 `n_tokens/n_seqs` 算出 `n_seq_tokens` 的。

`b_equal_seqs` 的类型值得一提：语义上是布尔，源码故意写成 `uint32_t`，注释给的理由是"否则 address sanitizer 会报错"（按 `int32_t` 对齐）。

<!-- src: src/llama-batch.h -->
```c
    uint32_t b_equal_seqs; // note: this is a boolean, but we use an int32_t for alignment
                           //       otherwise address sanitizer complains
    // TODO: whole_seqs for embeddings?

    uint32_t n_tokens;     // total tokens (n_seq_tokens * n_seqs)
    uint32_t n_seq_tokens; // tokens per sequence set
    uint32_t n_seqs;       // sequence sets in the ubatch
    uint32_t n_seqs_unq;   // unique sequence ids in the ubatch
    uint32_t n_pos;        // number of position inputs for each token/embedding

    // seq_id_unq: unique sequence ids in the ubatch
    // seq_idx:    indices of the unique sequence ids in the ubatch in [0, n_seqs_unq)
    //             used for extracting sequence pooled embeddings

    //                          // size               | idx | val
    llama_token  *  token;      // [n_tokens]         | i   | id, token
    float        *  embd;       // [n_embd, n_tokens] | i   | embd
    llama_pos    *  pos;        // [n_tokens*n_pos]   | i   | pos
    int32_t      *  n_seq_id;   // [n_tokens]         | i   | -
    llama_seq_id ** seq_id;     // [n_tokens]         | s   | s0, s1, seq_id
    llama_seq_id *  seq_id_unq; // [n_seqs_unq]       | s   | seq_id
    int32_t      *  seq_idx;    // [LLAMA_MAX_SEQ]    | -   | seq_idx
    int8_t       *  output;     // [n_tokens]         | i   | -
```

## 三、三种切法

头文件把三种切法的用途写在注释里：`split_simple` 面向"数量未知、长短不一的序列集"，`split_equal` 产出"等长序列集"的 ubatch，`split_seq` 每个 ubatch 只装一个序列集。

它们在实现上的差别只在传给 `ubatch_add()` 的 `n_seqs` 与 `equal_seqs` 上：

```text
split_simple : ubatch_add(idxs, idxs.size(), false)   -> n_seqs = n_tokens, n_seq_tokens = 1
split_equal  : ubatch_add(idxs, n_seqs, true)         -> 等长锁步切分
split_seq    : ubatch_add(idxs, 1, true)              -> n_seqs = 1
```

调用点按"流的条数"选：`llama_kv_cache::init_batch()` 里 `n_stream == 1` 用 `split_simple`，否则用 `split_equal`（`src/llama-kv-cache.cpp:713`，属于 L2-04 的覆盖范围，不计入本课覆盖率）。

<!-- src: src/llama-batch.h -->
```c
    // call once before splitting the batch to reset the internal state
    void split_reset();

    // simple split, unknown number of sequence sets of unequal lengths
    llama_ubatch split_simple(uint32_t n_ubatch);

    // make ubatches of equal-length sequences sets
    // if sequential == true, the tokens in the ubatch will have increasing sequential sequence ids
    // n_keep_tail = minimum trailing tokens of a seq that must land in the same ubatch
    llama_ubatch split_equal(uint32_t n_ubatch, bool sequential, uint32_t n_keep_tail);

    // sequence-set-wise split - each ubatch contains a single sequence-set
    llama_ubatch split_seq(uint32_t n_ubatch);
```

## 四、★ split_equal 的两条判据

第一个循环只做一件事：**选出这个 ubatch 的参与序列集**。两个判据都由它保证：

1. **互不相交**：`(cur_seq_set[s] & seq_set[i]).none()` —— 一个 token 不能同时属于本 ubatch 的两个序列集。相交意味着两条流共享 KV 位置，只能留到下一个 ubatch。
2. **序列号递增**（`sequential == true` 时）：`batch.seq_id[i][0] == last_seq_id + 1`。

选中之后 `n_seqs = cur_seq_set.size()`，接下来才是锁步扩张。

<!-- src: src/llama-batch.cpp -->
```c
    // determine the non-overlapping sequence sets participating in this ubatch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (used[i]) {
            continue;
        }

        bool add = true;

        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            // no overlap with existing sequence sets:
            if (!(cur_seq_set[s] & seq_set[i]).none()) {
                add = false;
                break;
            }
        }

        // accept only increasing sequence ids
        if (sequential) {
            add = add && (cur_seq_set.empty() || batch.seq_id[i][0] == last_seq_id + 1);
        }

        if (add) {
            cur_seq_set.push_back(seq_set[i]);

            last_seq_id = batch.seq_id[i][0];

            if (cur_seq_set.size() > n_ubatch) {
                break;
            }
        }
    }

    uint32_t n_seqs = cur_seq_set.size();
```

## 五、ubatch_add：从索引到字段

`ubatch_add()` 的输入是"要哪几个 batch 位置"（`idxs`）和"几个序列集"（`n_seqs`），开头一句 `assert(n_tokens%n_seqs == 0);` 就是等长切分的硬保证 —— 切分器不可能传进不整除的组合。

函数体后半段把这些搬进一个 `llama_ubatch res { ... }`：计数字段现算，数据指针指向 `udata`（`shared_ptr`）里的 vector。所以 `llama_ubatch` 本身可以随意拷贝，数据始终只有一份。

<!-- src: src/llama-batch.cpp -->
```c
llama_ubatch llama_batch_allocr::ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    const uint32_t n_tokens = idxs.size();

    assert(n_tokens%n_seqs == 0);

    auto udata = std::make_shared<llama_ubatch::data_t>();

    const int64_t n_embd_all = batch.embd ? (int64_t) n_tokens*n_embd : 0;
    const int64_t n_pos_all  =              (int64_t) n_tokens*n_pos_per_embd;
```

## 六、模型装配：五个 load 钩子 + 两个架构钩子

`llama_model` 把装配拆成两层：基类实现与架构无关的 `load_stats` / `load_hparams` / `load_vocab` / `load_tensors`，架构子类只实现 `load_arch_hparams` / `load_arch_tensors`（以及属于 L2-06 的 `build_arch_graph`）。

派发链是：`llama_model_create(llama_model_loader &)` 取 arch -> `llama_model_create(llm_arch, params)` -> `llama_model_mapping()` 的 `switch (arch)` 里 `new llama_model_*`（`src/llama-model.cpp:353` / `:44`）。

以 `LLM_ARCH_LLAMA` 为例，子类是 `llama_model_llama`（`src/models/models.h:147`，由 L2-10 覆盖），它把 `load_arch_tensors()` 写在 `src/models/llama.cpp:34`。

<!-- src: src/llama-model.h -->
```c
    ggml_cgraph * build_graph(const llm_graph_params & params) const;

    virtual void load_stats  (llama_model_loader & ml) = 0;
    virtual void load_hparams(llama_model_loader & ml) = 0;
    virtual void load_vocab  (llama_model_loader & ml) = 0;
    virtual bool load_tensors(llama_model_loader & ml) = 0; // returns false if cancelled by progress_callback

    // model must define these
    virtual void load_arch_hparams(llama_model_loader & ml) = 0;
    virtual void load_arch_tensors(llama_model_loader & ml) = 0;
    virtual std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const = 0;
```

## 七、设备与 buffer：每层的 layer_dev 在装配时定下来

`load_tensors()` 先枚举设备的 buffer 类型表（`make_cpu_buft_list()` / `make_gpu_buft_list()`，`src/llama-model.cpp:1061` / `:1123`），然后给每一层算一个 `layer_dev{dev, buft_list}`：

* `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`：从倒数第 `n_gpu_layers` 个槽位开始上 GPU；
* 输入层例外，`dev_input` 无条件设成 CPU（注释：offload 输入层收益很小）；
* 输出层用同一个 lambda、以 `il = n_layer_all` 单独算一次。

结果存进 `llama_model::impl::dev_layer`，之后建张量时按它选 buffer 类型，最后在 `load_tensors()` 的后半段由 `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)` 真正分配（`src/llama-model.cpp:1805`）。

<!-- src: src/llama-model.cpp -->
```c
    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);
```

## 八、可观测性：LLAMA_LOG_* 与 format()

装配和切分都不是"黑盒一步"：切分器用 `LLAMA_LOG_DEBUG` 打印每个 ubatch 的全部字段（`llama-batch.cpp` 里 35 处、`llama-model.cpp` 里 109 处 `LLAMA_LOG_*`），错误路径用 `format()` 拼消息（`llama-model.cpp` 里 10 处）。

宏定义在 `llama-impl.h`：日志级别宏转发到 `llama_log_internal()`，`LLAMA_ATTRIBUTE_FORMAT` 让编译器检查格式串 —— 这类日志正是排查"为什么切成了这样"的第一手材料。

<!-- src: src/llama-impl.h -->
```c
#define LLAMA_LOG(...)       llama_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)
```

## 九、装配计时：time_meas

`time_meas` 是一个 RAII 计时器：构造时取时间戳，析构时把差值累加进引用进来的累加器。模型加载路径用 `time_meas tm(model->t_load_us);` 统计装配耗时（`src/llama.cpp:340`，由 L2-01 覆盖），采样路径用同一件工具统计 `t_sample_us`。

它和本课主题的关系：装配是**一次性的大开销**（读文件 + 建张量 + 分配 buffer），切分则是**每次 decode 都要做的小开销** —— 两类成本分开计量，才有 `llama_model::t_load_us` 这类字段。

<!-- src: src/llama-impl.cpp -->
```c
time_meas::time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

time_meas::~time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}
```

---

## 说明

- 本课声明并逐字引用 6 个源文件，全部计入覆盖率：`src/llama-batch.cpp`、`src/llama-batch.h`、`src/llama-model.cpp`、`src/llama-model.h`、`src/llama-impl.cpp`、`src/llama-impl.h`。
- 跨课呼应处引用了其他课的文件与行号（`src/llama-kv-cache.cpp:713`、`src/llama-context.cpp`、`src/llama-graph.h/.cpp`、`src/models/models.h:147`、`src/models/llama.cpp:34`、`src/llama.cpp:340`），这些**不在本课的 `src=` 声明里，不计入本课覆盖率** —— 它们分别属于 L2-04 / L2-06 / L2-07 / L2-10 / L2-01。
