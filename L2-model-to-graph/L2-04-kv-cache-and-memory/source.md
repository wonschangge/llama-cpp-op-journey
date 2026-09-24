<!-- llama-coverage
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-kv-cache-iswa.cpp
src/llama-kv-cache-iswa.h
src/llama-kv-cache-dsa.cpp
src/llama-kv-cache-dsa.h
src/llama-kv-cache-dsa-iswa.cpp
src/llama-kv-cache-dsa-iswa.h
src/llama-kv-cache-dsv4.cpp
src/llama-kv-cache-dsv4.h
src/llama-kv-cache-msa.cpp
src/llama-kv-cache-msa.h
src/llama-kv-cells.h
src/llama-memory.cpp
src/llama-memory.h
src/llama-memory-hybrid.cpp
src/llama-memory-hybrid.h
src/llama-memory-hybrid-idx.cpp
src/llama-memory-hybrid-idx.h
src/llama-memory-hybrid-iswa.cpp
src/llama-memory-hybrid-iswa.h
src/llama-memory-recurrent.cpp
src/llama-memory-recurrent.h
src/llama-model.cpp
-->

# L2-04 · KV cache 与记忆家族 — 源文件

**一句话**：推理时的"记忆"不是一种东西，而是一族东西 —— 同一个 `llama_memory_i` 接口下，既有按 token 记 K/V 的 KV cache，也有 SSM/RNN 那种每层只有一份固定状态的 memory，还有把两者按层拼起来的 hybrid。

回顾 L2-03：那一课讲的是**权重**怎么从磁盘进内存（mmap）。这一课讲**运行期**的记忆：decode 过程中新产生的 K/V 或状态存在哪、谁能读写、谁来决定用哪一种。

---

## 一、为什么需要 memory 抽象

接口定义的第一段注释就回答了这个问题：**KV cache 只是一种 LLM memory**。同一个结构体上还挂着三类回调（层过滤、层复用、层共享）——差异从接口这一层就开始被表达，而不是等到某个实现内部再 `if` 出来。

<!-- src: src/llama-memory.h -->
```c
// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;
```

## 二、★ llama_memory_i 的全部虚函数

15 个纯虚函数，按职责分三组：

| 组 | 函数 | 作用 |
|---|---|---|
| 生命周期 | `init_batch` `init_full` `init_update` | 规划一批 token 的落点，返回 context |
| 序列操作 | `clear` `seq_rm` `seq_cp` `seq_keep` `seq_add` `seq_div` `seq_pos_min` `seq_pos_max` `get_can_shift` | 按 seq_id + 位置区间增删改查 |
| 度量与状态 | `memory_breakdown` `state_write` `state_read` | 显存占用、状态存盘与恢复 |

注意接口里**没有** `get_k` / `cpy_k`：数据面（K/V 张量、状态张量）不在这个接口上，只有 KV cache 一族的 context 才额外提供它们。

<!-- src: src/llama-memory.h -->
```c
    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};
```

## 三、★ context：apply() 是唯一改状态的入口

`init_*` 只做规划并返回一个 `llama_memory_context_i`，真正的写入发生在 `apply()`。源码注释把这一点写成硬约束：

```text
the only method that should mutate the memory and the memory context is apply()
```

组合型 memory（iswa / dsa / hybrid）对外要表现得像一个 memory，靠的是把子 context 的 `llama_memory_status` 合并：任一失败即失败，任一有更新即有更新。

<!-- src: src/llama-memory.h -->
```c
// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;
```

## 四、status 合并：组合型 memory 的粘合剂

`llama_memory_status_combine()` 是整个组合机制的基础设施：任一侧失败就直接返回那个失败，否则任一侧有更新就算有更新。接口声明旁的注释点名了它的用途 —— hybrid memory 类型（例如 iSWA）。

<!-- src: src/llama-memory.cpp -->
```c
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;
//>> ---- src/llama-memory.cpp:40-41 ----
    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
```

## 五、★ cell：元数据在 llama_kv_cells，K/V 在别处

`llama_kv_cells` 的成员全是元数据：`pos`（位置，`-1` 表示空）、`ext`（M-RoPE 的 x/y 与 token id）、`shift`（攒起来的位移，最后一次同步）、`seq`（位集合，一个 cell 可属于多个序列）、`used`（非空下标集合）、`seq_pos`（`(pos, cell)` 有序集合）。

K/V 不在这个类里 —— 它们在 `llama_kv_cache::layers[]` 的 ggml 张量中（`src/llama-kv-cache.h:250-260`）。**cell 是账本，张量是仓库。**

<!-- src: src/llama-kv-cells.h -->
```c
private:
    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // the set seq_pos[s] holds one (pos, cell) pair per cell that carries sequence s, ordered by position
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    // and upper_bound() on a position finds the nearest cell of the sequence in logarithmic time
    //
    // the cell index is part of the key because a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    std::set<std::pair<llama_pos, uint32_t>> seq_pos[LLAMA_MAX_SEQ];
```

## 六、slot_info 与 apply_ubatch

`slot_info` 把 "第 i 个 token 写进哪个 cell" 变成一张表。落笔时先清掉被覆盖的旧 cell（`cells.rm`），再 `cells.pos_set()` 写入新位置 —— 这就是 SWA 的环形复用能工作的原因。

<!-- src: src/llama-kv-cache.h -->
```c
    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]
```

## 七、同一张表的消费端：apply_ubatch

第 6 节那张表在 `apply_ubatch()` 里被消费：`sinfo.idxs[s][ii]` 直接当 cell 下标用。

<!-- src: src/llama-kv-cache.cpp -->
```c
            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);
```

## 八、K/V 的归属与读写

K/V 张量在 `llama_kv_cache` 构造时**按后端 buffer type 整块分配**，注释写明 `no_alloc` 模式下挂 dummy buffer 是为了让 backend scheduler 不去分配它。读走 `ggml_view_4d`（零拷贝视图），写走 `ggml_set_rows`（图上的一个节点，行号来自 slot_info）。

<!-- src: src/llama-kv-cache.cpp -->
```c
    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
//>> ---- src/llama-kv-cache.cpp:1266-1283 ----
ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
//>> ---- src/llama-kv-cache.cpp:1346-1351 ----
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

## 九、iswa：用两个 llama_kv_cache 表示"两种层"

类注释一句话说清结构：**两个 llama_kv_cache 实例**，第一个给非 SWA 层，第二个给 SWA 层。实现方式是链式 filter：`filter_base` 保留 `!is_swa(il)`，`filter_swa` 保留 `is_swa(il)`；SWA cache 的大小按窗口算（并 pad 到 256）。

<!-- src: src/llama-kv-cache-iswa.cpp -->
```c
    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
//>> ---- src/llama-kv-cache-iswa.cpp:95-105 ----
    kv_base = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, mem_other_swa, filter_swa, reuse, share);
```

## 十、iswa 的头文件：为什么是"两个实例"

类注释把这个类的全部设计压成一句话：**两个 llama_kv_cache 实例** —— 第一个给非 SWA 层，第二个给 SWA 层。

<!-- src: src/llama-kv-cache-iswa.h -->
```c
// utilizes two instances of llama_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers
```

## 十一、KV cache 一族的其余变体（注释原话）

**dsa**：两个 llama_kv_cache —— 一个存模型的 key，一个存 lightning indexer 的 key。

<!-- src: src/llama-kv-cache-dsa.h -->
```c
// utilizes two instances of llama_kv_cache:
// - the first instance is for caching key tensors of the model,
// - the second instance is for caching lightning indexer key tensors
```

## 十二、dsa 的 indexer cache 是怎么造出来的

第二个 cache 直接复用 `llama_kv_cache`：注释原话是 hand-tweaking some hparams —— 把每层的 KV 头数全部改成 1、头维度换成 `indexer_head_size`，同一个类就长出了 indexer key 的形状。

<!-- src: src/llama-kv-cache-dsa.cpp -->
```c
    // we use llama_kv_cache for caching indexer keys
    // by hand-tweaking some hparams we fool it to create
    // indexer key cache tensors with correct dimensions
    // https://github.com/ggml-org/llama.cpp/pull/21149#discussion_r3015940823

    // DSA lightning indexer uses MQA with single key head
    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;

    LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, nullptr, filter_lid, reuse, nullptr);
```

## 十三、dsa_iswa

两个子 memory：DSA 那一套给全注意力层，`llama_kv_cache` 那一套给 SWA 层。

<!-- src: src/llama-kv-cache-dsa-iswa.h -->
```c
// utilizes two child memories: llama_kv_cache_dsa for the full-attention (DSA) layers and llama_kv_cache for the SWA layers
```

## 十四、dsa_iswa 的链式 filter

和 iswa 同一套手法：先把上游 filter 串起来，再按 `is_swa(il)` 一分为二；DSA 那一半拿满 `kv_size`，SWA 那一半按窗口算。

<!-- src: src/llama-kv-cache-dsa-iswa.cpp -->
```c
        return !hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa = [&](int32_t il) {
        if (filter_mla && !filter_mla(il)) {
            return false;
        }

        return hparams.is_swa(il);
    };

    const uint32_t size_dsa = kv_size;
```

## 十五、msa

两个 llama_kv_cache：一个存 K/V，一个存 MSA indexer。两者接收**完全相同的序列操作与 ubatch**，所以 cell 布局保持同步；context 还会把 `llama_kv_cells` 里的 pos -> cell 映射暴露给图，做 position 空间的块选择。

<!-- src: src/llama-kv-cache-msa.h -->
```c
// uses two instances of llama_kv_cache, one for K/V tensors, and one for the MSA indexer tensors
// both receive identical sequence operations and identical ubatches, so their cell layouts stay in synced.
// the context also exposes per-ubatch pos - cell translation maps populated from llama_kv_cells via
// llama_kv_cache::get_cells(), which the model graph uses to run MSA block selection in position space
```

## 十六、msa 的 indexer cache 形状

MSA 的 indexer 也是"每层一个 key 头"：把 `n_head_kv_arr` 全填 1，头维度换成 `indexer_head_size`；注释特别说明 **rope 参数与主 cache 保持一致**。

<!-- src: src/llama-kv-cache-msa.cpp -->
```c
    // the MSA indexer uses a single key head per layer
    std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
    hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;
    // the rope parameters are kept identical to the main cache
```

## 十七、dsv4

一个普通的 raw/SWA token cache **加上压缩过的 K-only 块 cache**；压缩 cache 只做存储，可见性与块规划交给 context 与图输入去处理。压缩比是文件顶部的两个常量：`DSV4_CSA_RATIO = 4`、`DSV4_HCA_RATIO = 128`。

<!-- src: src/llama-kv-cache-dsv4.h -->
```c
// DSV4 uses a normal raw/SWA token cache plus compressed K-only block caches.
// The compressed caches are storage only; DSV4-specific visibility and block
// planning are handled by llama_kv_cache_dsv4_context / llm_graph_input_dsv4.
```

## 十八、dsv4 的两个压缩比常量

压缩 cache 的粒度由文件顶部两个常量决定：`DSV4_CSA_RATIO = 4` 与 `DSV4_HCA_RATIO = 128`。压缩块的行数由 `dsv4_comp_size(kv_size, ratio)` 按 `kv_size / ratio` 算出来。

<!-- src: src/llama-kv-cache-dsv4.cpp -->
```c
static constexpr uint32_t DSV4_CSA_RATIO = 4;
static constexpr uint32_t DSV4_HCA_RATIO = 128;
```

## 十九、recurrent：没有 KV，只有状态

`mem_cell` 的字段是 `pos` / `src` / `src0` / `tail` / `seq_id` —— 没有任何 K/V。数据面是每层的 `r_l` / `s_l`（有 PLE 卷积历史时再加 `p_l`）；`n_rs_seq` 是每个序列保留的回滚快照数，所以张量的行数是 `mem_size * (1 + n_rs_seq)`。

<!-- src: src/llama-memory-recurrent.h -->
```c
    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;
```

## 二十、recurrent 的张量形状（cpp 侧）

`r` / `s` 张量的第一个维度是 `hparams.n_embd_r()` / `n_embd_s()`，第二个维度是行数 —— **行数只跟 mem_size 与快照数有关，与序列长度无关**。

<!-- src: src/llama-memory-recurrent.cpp -->
```c
        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
```

## 二十一、hybrid：attention 层与 recurrent 层各一套子 memory

类注释：用 `llama_memory_recurrent` 与 `llama_kv_cache` 的组合，支持"每一层要么是 attention、要么是 recurrent"的模型。默认 filter 就是按 `hparams.is_recr(il)` 分工。

<!-- src: src/llama-memory-hybrid.cpp -->
```c
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
```

## 二十二、hybrid 的两个更专版本

`hybrid_idx`：在 hybrid 之上再加第三套 cache —— 每个 token 一个 indexer key，用于块稀疏注意力。indexer 是 attention cell 的**旁路缓冲**：大小、padding、stream、slot 全都一样，所以 cell j 在两边指的是同一个 token。

<!-- src: src/llama-memory-hybrid-idx.h -->
```c
// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both
```

## 二十三、hybrid_idx 的第三套 cache

`mem_idx` 只在给了 `filter_idx` 时才建（即 GGUF 里真的有 indexer 张量时）。构造方式同样是改 hparams 造形状，并且显式把 rope 设成 NONE、把 K/V 的 MLA 维度都设成 indexer 头维度 —— 注释说这是"骗 llama_kv_cache 不要缓存 V"。

<!-- src: src/llama-memory-hybrid-idx.cpp -->
```c
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;
```

## 二十四、hybrid_iswa

把 hybrid 的 attention 侧换成 `llama_kv_cache_iswa`：支持"每层要么是 attention（带 SWA）、要么是 recurrent"的模型。

<!-- src: src/llama-memory-hybrid-iswa.h -->
```c
// utilizes instances of llama_memory_recurrent and llama_kv_cache_iswa to
//   support models where each layer may be either attention-based (with SWA support) or recurrent
```

## 二十五、hybrid_iswa 的 attention 侧

`mem_attn` 的类型直接写成 `llama_kv_cache_iswa` —— "hybrid"与"iswa"两个维度的正交组合，在这里只是一行构造参数。

<!-- src: src/llama-memory-hybrid-iswa.cpp -->
```c
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_iswa(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
```

## 二十六、分派点：llama_model::create_memory()

一个 `switch (arch)` 决定用哪个 memory 实现。钩子是两个判定函数：`llm_arch_is_recurrent(arch)` 与 `llm_arch_is_hybrid(arch)`；hybrid 分支里再按架构挑 layer filter，最后返回 `llama_memory_i *`。

<!-- src: src/llama-model.cpp -->
```c
llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
//>> ---- src/llama-model.cpp:2547-2557 ----
                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            cparams.n_rs_seq,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron) {
```

---

## 说明

- 本课声明 24 个源文件：计划中 L2-04 的 23 个全部声明，另加 `src/llama-model.cpp`（memory 的分派点，`llama_model::create_memory()` 在 2274 行）。`src/llama-model.cpp` 在计划里归 L2-05，本课引用后一并计入覆盖，特此说明。
- 本课不引用 `src/llama-arch.cpp`，因此 `llm_arch_is_recurrent()` / `llm_arch_is_hybrid()` 里的具体架构清单不在本课展开，留给 L2-11（模型家族 · SSM 与线性注意力）。
- 本课不讨论任何命令行参数，参数门禁的真值集来自真实二进制的帮助输出。
