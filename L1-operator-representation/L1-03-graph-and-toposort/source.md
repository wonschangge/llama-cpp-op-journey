<!-- llama-coverage
ggml/src/ggml.c
ggml/src/ggml-impl.h
ggml/include/ggml-cpp.h
-->

# L1-03 · 计算图与拓扑遍历：节点怎么排成一队 — 源文件

**一句话**：`ggml_build_forward_expand(cgraph, tensor)` 把"一个输出张量"变成"一个有序的节点数组 `cgraph->nodes[]`"；排序靠**后序深度优先遍历**，去重靠**图自己持有的 hash set**。

为什么值得单独一课：L4-04 的后端 `graph_compute` 就是按 `nodes[0], nodes[1], ...` 的顺序逐个执行的。顺序错了，模型算出来的就是另一个东西。

验收点：**能解释为什么拓扑排序用 hash set 而不是在张量上打个标记位**。答案不在"hash 快"，而在"状态归谁"——标记位必须属于图，不能属于张量。

---

## 一、used[] 位图：去重状态长什么样

`ggml_hash_set` 的 `used` 是一张位图：每个 `ggml_bitset_t` 是 `uint32_t`，`BITSET_SHR = 5` 表示"每 32 个槽位一个元素"。三条位运算分别负责问、置、清 —— **图的全部 visited 状态就是这一块可以整块 `memset` 的位图**。这也是"为什么不用张量标记位"的第一层答案：状态是一个可以一次性清空的连续块，而不是散落在成千上万个张量里的字段。

<!-- src: ggml/src/ggml-impl.h -->
```c
typedef uint32_t ggml_bitset_t;

static_assert(sizeof(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (sizeof(ggml_bitset_t)*8 - 1)

static size_t ggml_bitset_size(size_t n) {
    return (n + BITSET_MASK) >> BITSET_SHR;
}

static inline bool ggml_bitset_get(const ggml_bitset_t * bitset, size_t i) {
    return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

static inline void ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

static inline void ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}
```

## 二、hash set 的分配与容量策略

表本身只有两个数组：`keys`（指针）与 `used`（位图），`ggml_hash_set_new()` 一次把两者建好。容量不走"两倍扩容"，而是查一张写死的质数表：注释说明这些数是 "next primes after powers of two"，二分找出第一个不小于需求的值；如果需求超出表长，就退化成 `min_sz | 1`（奇数）。建图时传给它的需求是 `size * 2` —— 因为一张表要同时装下 nodes 与 leafs（段落末尾的引用给出注释原文）。注意注释里的"next primes after powers of two"是有道理的：质数取模能让线性探测的聚集更均匀，而当前实现是"建图时定容、表满即 abort"，不做任何扩容。

<!-- src: ggml/src/ggml.c -->
```c
struct ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    struct ggml_hash_set result;
    result.size = size;
    result.keys = GGML_MALLOC(sizeof(struct ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), sizeof(ggml_bitset_t));
    return result;
}
//>> ---- ggml/src/ggml.c:6631-6655 ----
size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    static const size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    static const size_t n_primes = sizeof(primes)/sizeof(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    while (l < r) {
        size_t m = (l + r)/2;
        if (primes[m] < min_sz) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    return sz;
}
//>> ---- ggml/src/ggml.c:7482-7483 ----
    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);
```

## 三、为什么标记位不能打在张量上

`visited_hash_set` 是 `struct ggml_cgraph` 的成员，所以"见过没有"这份状态跟着图走。下面的引用给出两条证据。第一条：`ggml_graph_view()` 按值复制整个 `visited_hash_set`，让一个切片图与原图**共用同一张表**（注释还写明梯度拿不到表所以为 NULL）。第二条：`ggml_graph_dup()` 在给定 context 里建一张新图，`ggml_graph_cpy()` 把在用的 key **重新插进目标图自己的表**，并逐个重新发号。于是同一批 `ggml_tensor *` 可以同时属于多张图：切片图与原图**共享**同一张表，复制出来的图则各有一张表、各有各的槽位编号。无论哪种情况，"这张图见过谁"都是**图的状态**而不是张量的状态 —— 这正是标记位不能写在张量上的原因。

<!-- src: ggml/src/ggml.c -->
```c
struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ NULL, // gradients would need visited_hash_set
        /*.grad_accs        =*/ NULL,
        /*.leafs            =*/ NULL,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    return cgraph;
}
//>> ---- ggml/src/ggml.c:7561-7567 ----
    for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        if (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }
//>> ---- ggml/src/ggml.c:7591-7595 ----
struct ggml_cgraph * ggml_graph_dup(struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads) {
    struct ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    return result;
}
```

## 四、struct ggml_cgraph 的全部字段

图的定义本身在 `ggml-impl.h` 里：三个计数器、五个数组指针，一个 hash set，一个遍历顺序，外加一个可选的身份号 `uid`（注释说明 0 表示未设置）。注意 `use_counts` 的注释：**indexed by hash table slot** —— 使用次数不是按节点下标存的，而是按哈希槽位存的，这就是 hash set 顺带提供的"图内编号"。顺带一个诚实的观察：`ggml_graph_view()` 里的指示初始化注释写的是 `/*.visited_hash_set =*/`，而 `ggml_new_graph_custom()` 里同一条注释写的是 `/*.hash_table =*/` —— 后者是**过时的注释**（结构体里早已没有 `hash_table` 这个字段），说明源码注释也会滞后，一切以结构体定义为准。

<!-- src: ggml/src/ggml-impl.h -->
```c
struct ggml_cgraph {
    int size;    // maximum number of nodes/leafs/grads/grad_accs
    int n_nodes; // number of nodes currently in use
    int n_leafs; // number of leafs currently in use

    struct ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    struct ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    struct ggml_tensor ** grad_accs; // accumulators for node gradients
    struct ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    struct ggml_hash_set visited_hash_set;

    enum ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};
```

## 五、图的存储与清空

图的字节数由 `ggml_graph_nbytes()` 算死，`ggml_new_graph_custom()` 再按同一顺序切出来；数组紧密排在结构体之后。于是"清空一张图"不需要释放任何东西 —— `ggml_graph_clear()` 把两个计数器归零、把 `used` 位图整块清零，图就回到刚建好的状态。这也解释了 `ggml_build_forward_impl()` 里 `expand` 参数的含义：`expand = true` 时**不清图**，可以在已有内容上继续追加（第 2 幕源码第 7305 行）。

<!-- src: ggml/src/ggml.c -->
```c
    size_t hash_size = ggml_hash_size(size * 2);
    void * p = 0;
    incr_ptr_aligned(&p, sizeof(struct ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // hash keys
    if (grads) {
        incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));
//>> ---- ggml/src/ggml.c:7646-7650 ----
void ggml_graph_clear(struct ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}
```

## 六、C++ 包装层：谁负责释放

`ggml-cpp.h` 不是新的数据结构，只是一层 C++ 智能指针别名：为每种 C 句柄写一个 deleter，再用 `std::unique_ptr` 包起来。它对图本身没有影响，但解释了 `ggml_context` 这个 arena 的宿主是谁 —— 图、张量、图里的 hash set 都随 ctx 一起释放。文件开头的 `#error "This header is for C++ only"` 也说明它是给 C++ 调用方用的便利层。

<!-- src: ggml/include/ggml-cpp.h -->
```c
struct ggml_gallocr_deleter { void operator()(ggml_gallocr_t galloc) { ggml_gallocr_free(galloc); } };

typedef std::unique_ptr<ggml_gallocr, ggml_gallocr_deleter> ggml_gallocr_ptr;

// ggml-backend

struct ggml_backend_deleter        { void operator()(ggml_backend_t backend)       { ggml_backend_free(backend); } };
struct ggml_backend_buffer_deleter { void operator()(ggml_backend_buffer_t buffer) { ggml_backend_buffer_free(buffer); } };
struct ggml_backend_event_deleter  { void operator()(ggml_backend_event_t event)   { ggml_backend_event_free(event); } };
struct ggml_backend_sched_deleter  { void operator()(ggml_backend_sched_t sched)   { ggml_backend_sched_free(sched); } };

typedef std::unique_ptr<ggml_backend,        ggml_backend_deleter>        ggml_backend_ptr;
typedef std::unique_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter> ggml_backend_buffer_ptr;
typedef std::unique_ptr<ggml_backend_event,  ggml_backend_event_deleter>  ggml_backend_event_ptr;
typedef std::unique_ptr<ggml_backend_sched,  ggml_backend_sched_deleter>  ggml_backend_sched_ptr;
```

---

## 说明

- 本课声明 3 个源文件：`ggml/src/ggml.c`（主）、`ggml/src/ggml-impl.h`（`struct ggml_cgraph` 与 `struct ggml_hash_set` 的定义与实现所在）、`ggml/include/ggml-cpp.h`（C++ RAII 包装层）。三者都在覆盖域内，均计入覆盖率。
- `ggml_graph_plan()` / `ggml_graph_compute()` 这一版不在 `ggml.c` 里（它们属于 CPU 后端的 `ggml-cpu`）。本课只说"顺序被后端按 nodes[] 消费"，不引用其源码，具体展开见 L4-04。
- GGML_DEFAULT_GRAPH_SIZE（默认图容量）定义在 `ggml/include/ggml.h`，该文件由 L1-01 覆盖；本课不引用其行号，故不计入本课声明。
- 工程备注（历史）：本课写作期间 `tools/check_flags.py` 的 `strip_verbatim()` 把 lesson.js 的**内容**当路径传给 `dump_scenes.js`，node 回显整份内容到 stderr 并在 65536 字节处截断（实测 `stderr = 2 x 文件字节 + 51`）；截断点落在多字节字符中间时 Python 的 strict 解码会抛 `UnicodeDecodeError`，整个参数门禁失败。该缺陷已由 3385a08 修复（现在传的是路径，且解码用 `errors="replace"`、不再静默降级）。缺陷存在期间本课 `visual` 写得紧凑、把 lesson.js 压到 32 KB 上下；修复后这个限制已消失，较长的解释仍留在 `source.md`。
