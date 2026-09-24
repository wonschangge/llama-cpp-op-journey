<!-- llama-coverage
src/llama-sampler.cpp
src/llama-sampler.h
src/llama-vocab.cpp
src/llama-vocab.h
src/unicode.cpp
src/unicode.h
src/unicode-data.cpp
src/unicode-data.h
-->

# L2-08 · 采样器与词表：图算 logits，host 选 token — 源文件

**一句话**：`llama_decode()` 在图上算完，交出来的是 **logits**（每个 token 一个分数，长度 = 词表大小）。把它变成一个 token 的那一步叫**采样**：它发生在 **host（CPU）** 上，操作的是一个 `llama_token_data_array`（token id / logit / p 的数组），**不产生任何 ggml 算子**。

这一课同时补上模型的另一头：文本要先被**词表**切成 token 才能进图。分词（BPE 合并、unicode 规范化与类别判断）同样跑在 host 上，靠的是 `unicode-data.cpp` 里 7000 行生成好的码点表。

读法建议：前 4 幕走"入口"（词表 / BPE / unicode），后 5 幕走"出口"（采样链 / 顺序 / 为何不进图）。

---

## 一、词表的职责面

`llama_vocab` 是纯 host 对象：它把文本和 token id 互相翻译，并持有 BPE 的 merges 表。注意 `find_bpe_rank()` 与 `tokenize()` 都在这里声明 —— 分词算法的"知识"一半在代码里，一半在词表数据里。

<!-- src: src/llama-vocab.h -->
```c
    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum llama_vocab_type     get_type()     const;
    enum llama_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;
//>> ---- src/llama-vocab.h:160-176 ----
    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llama_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<llama_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;
```

## 二、分词入口：先切分，再分派

`tokenize()` 的第一件事不是查 merges，而是把文本装进 `fragment_buffer` 并调用 `tokenizer_st_partition()`：特殊 token（BOS/EOS/控制符）被优先摘出来，剩下的原始文本片段才逐段交给具体 tokenizer。这就是"prompt 里写了 `<|im_start|>` 为什么不会被切成普通字符"的原因。

<!-- src: src/llama-vocab.cpp -->
```c
std::vector<llama_token> llama_vocab::impl::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.");

    std::vector<llama_token> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(fragment_buffer, parse_special);
    }

    switch (get_type()) {
```

## 三、特殊 token 的优先切分

`tokenizer_st_partition()` 遍历 `cache_special_tokens`，对每个特殊 token 扫描所有尚未处理的文本片段。源码注释指出：当 `parse_special == false` 时，CONTROL 与 UNKNOWN 属性的 token 被跳过，而用户自定义 token 仍然参与预切分。

<!-- src: src/llama-vocab.cpp -->
```c
void llama_vocab::impl::tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const {
    // for each special token
    for (const llama_token special_id : cache_special_tokens) {
        const auto & data = vocab.get_token_data(special_id);
        const auto & text = data.text;

        if (!parse_special && (data.attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
            // User-defined tokens are still pre-tokenized before everything else
            // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
            // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
```

## 四、★ BPE：symbol 链 + 按 rank 排序的优先队列

BPE 的实现是"链 + 优先队列"：

```text
1. 把每个词按字符拆成 symbols（一个双向链表）
2. 每对相邻 symbol 查 find_bpe_rank()，查得到就入优先队列
3. 每次弹出 rank 最小的 bigram，合并，并只补两个新 bigram
```

`llama_priority_queue` 是 `std::priority_queue` 的一个薄包装，只多了一个 `pop_move()`（把 `pop()` 显式 `delete` 掉，避免误用拷贝语义）。排序准则写在 `llm_bigram_bpe::comparator` 里：先比 `rank`，rank 相同再比左端位置。

<!-- src: src/llama-vocab.cpp -->
```c
template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() =  delete;
};
//>> ---- src/llama-vocab.cpp:264-278 ----
struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};
```

## 五、BPE 主循环：合并后只补两个 bigram

这是 BPE 的性能关键：合并 `(left, right)` 之后，只有左邻居与右邻居可能产生新的可合并对，所以只调用两次 `add_new_bigram()`。注意第 673 行的"过期 bigram"检查 —— 队列里可能存着已经失效的 bigram，靠文本比对丢弃。

<!-- src: src/llama-vocab.cpp -->
```c
            for (int i = 1; i < (int) symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }
```

## 六、unicode 的三个查表函数

`unicode_len_utf8()` 只看首字节高 4 位；`unicode_cpt_from_utf8()` 按位掩码拼码点，非法字节直接 `throw std::invalid_argument`（上层 `unicode_cpts_from_utf8()` 会把它换成 U+FFFD）。`unicode_cpt_flags_from_cpt()` 与 `unicode_cpts_normalize_nfd()` 都是**查表**：一个查 `unicode_ranges_flags`（构建一张 0x110000 项的数组），一个在 `unicode_ranges_nfd` 上二分。

<!-- src: src/unicode.cpp -->
```c
size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
//>> ---- src/unicode.cpp:1117-1128 ----
std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}
//>> ---- src/unicode.cpp:1147-1160 ----
unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;  // undefined
    }
    size_t offset = 0;
    return unicode_cpt_flags_from_cpt(unicode_cpt_from_utf8(utf8, offset));
}
```

## 七、unicode.h：12 个标志位挤进一个 uint16

`unicode_cpt_flags` 用位域把 12 个布尔量塞进 16 位，并提供 `as_uint()` 双向转换。源码在这里写明了一个**可移植性陷阱**：转换依赖字节序（`__BYTE_ORDER__`），文件开头因此留着一条 TODO：reimplement this structure in endian-independent way。另外请注意，位域定义的顺序与 `as_uint()` 里的移位是**同一件事的两种写法**，所以这些标志可以被当作一个整数存进数据表。

<!-- src: src/unicode.h -->
```c
// TODO: reimplement this structure in endian-independent way
struct unicode_cpt_flags {
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,  // regex: \p{N}
        LETTER          = 0x0004,  // regex: \p{L}
        SEPARATOR       = 0x0008,  // regex: \p{Z}
        ACCENT_MARK     = 0x0010,  // regex: \p{M}
        PUNCTUATION     = 0x0020,  // regex: \p{P}
        SYMBOL          = 0x0040,  // regex: \p{S}
        CONTROL         = 0x0080,  // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
        WHITESPACE      = 0x0100,
        LOWERCASE       = 0x0200,
        UPPERCASE       = 0x0400,
        NFD             = 0x0800,
    };
//>> ---- src/unicode.h:25-48 ----
    // codepoint type
    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;  // regex: \p{N}
    uint16_t is_letter      : 1;  // regex: \p{L}
    uint16_t is_separator   : 1;  // regex: \p{Z}
    uint16_t is_accent_mark : 1;  // regex: \p{M}
    uint16_t is_punctuation : 1;  // regex: \p{P}
    uint16_t is_symbol      : 1;  // regex: \p{S}
    uint16_t is_control     : 1;  // regex: \p{C}
    // helper flags
    uint16_t is_whitespace  : 1;  // regex: \s
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    // decode from uint16
    inline unicode_cpt_flags(const uint16_t flags = 0) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        *reinterpret_cast<uint16_t*>(this) = flags;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        is_undefined   = (flags & UNDEFINED)   ? 1 : 0;
        is_number      = (flags & NUMBER)      ? 1 : 0;
        is_letter      = (flags & LETTER)      ? 1 : 0;
        is_separator   = (flags & SEPARATOR)   ? 1 : 0;
//>> ---- src/unicode.h:85-95 ----
    }

    inline uint16_t category_flag() const {
        return this->as_uint() & MASK_CATEGORIES;
    }
};

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8  (uint32_t cpt);
uint32_t    unicode_cpt_from_utf8(const std::string & utf8, size_t & offset);
```

## 八、unicode-data.h：五张表的对外声明

这个头文件只有 20 行，却定义了整套 unicode 支持的接口面：一个 `range_nfd` 结构体、一个 `MAX_CODEPOINTS` 常量，以及五个 `extern` 表。把"数据"和"算法"分开声明，是这一课能讲清楚 unicode 的前提。

<!-- src: src/unicode-data.h -->
```c
struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const std::initializer_list<range_nfd> unicode_ranges_nfd;
```

## 九、unicode-data.cpp：7009 行生成数据

第一行就写明它是 `scripts/gen-unicode-data.py` 的生成物。五张表的规模（用行号数出来的真实值）：

| 表 | 行号区间 | 行数 |
|---|---|---|
| `unicode_ranges_flags` | 11 - 2283 | 2273 |
| `unicode_set_whitespace` | 2287 - 2311 | 25 |
| `unicode_map_lowercase` | 2316 - 3748 | 1433 |
| `unicode_map_uppercase` | 3753 - 5202 | 1450 |
| `unicode_ranges_nfd` | 5206 - 7033 | 1828 |

`unicode_ranges_flags` 的注释说明了编码方式：每行是"区间起点 + 标志位"，`last = next_start - 1`；`unicode_cpt_flags_array()` 正是按这个约定把它展开成数组的。大小写两张表上面都写着同一句约束：list is always in ascending order, to enable binary search。

<!-- src: src/unicode-data.cpp -->
```c
// generated with scripts/gen-unicode-data.py

#include "unicode-data.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {  // start, flags // last=next_start-1
{0x000000, 0x0080},
{0x000020, 0x0008},
{0x000021, 0x0020},
//>> ---- src/unicode-data.cpp:2286-2292 ----
const std::unordered_set<uint32_t> unicode_set_whitespace = {
0x000009,
0x00000A,
0x00000B,
0x00000C,
0x00000D,
0x000020,
//>> ---- src/unicode-data.cpp:2314-2319 ----
// list is always in ascending order, to enable binary search
const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase = {
{0x000041, 0x000061},
{0x000042, 0x000062},
{0x000043, 0x000063},
{0x000044, 0x000064},
//>> ---- src/unicode-data.cpp:5205-5210 ----
const std::initializer_list<range_nfd> unicode_ranges_nfd = {  // start, last, nfd
{0x000000, 0x000000, 0x000000},
{0x0000C0, 0x0000C5, 0x000041},
{0x0000C7, 0x0000C7, 0x000043},
{0x0000C8, 0x0000CB, 0x000045},
{0x0000CC, 0x0000CF, 0x000049},
```

## 十、采样器家族：函数名与行号（全部来自 llama-sampler.cpp）

`llama_sampler_init_*` 一共 22 个定义。下面这份名单是把 `llama-sampler.cpp` 里所有 `struct llama_sampler * llama_sampler_init_*` 定义逐个数出来的，括号里是定义所在行：

| 家族 | 函数（行号） |
|---|---|
| 基础 | `llama_sampler_init_empty` (520)、`llama_sampler_init_greedy` (1107)、`llama_sampler_init_dist` (1399) |
| 截断 | `llama_sampler_init_top_k` (1519)、`llama_sampler_init_top_p` (1719)、`llama_sampler_init_min_p` (1882)、`llama_sampler_init_typical` (1994)、`llama_sampler_init_top_n_sigma` (3300) |
| 温度 | `llama_sampler_init_temp` (2104)、`llama_sampler_init_temp_ext` (2307)、`llama_sampler_init_xtc` (2416) |
| 自适应 | `llama_sampler_init_mirostat` (2537)、`llama_sampler_init_mirostat_v2` (2643)、`llama_sampler_init_adaptive_p` (3860) |
| 约束与偏置 | `llama_sampler_init_grammar` (2825)、`llama_sampler_init_grammar_lazy` (2832)、`llama_sampler_init_grammar_lazy_patterns` (2843)、`llama_sampler_init_logit_bias` (4043) |
| 重复抑制 | `llama_sampler_init_penalties` (3203)、`llama_sampler_init_dry` (3639)、`llama_sampler_init_dry_testing` (3690) |
| 填充 | `llama_sampler_init_infill` (4288) |

背后是两个"空实现"规则：参数等于默认值时，`llama_sampler_init_*` 会返回 `llama_sampler_init_empty("?top-k")` 这样的占位节点（例如 top_k 的 `k <= 0`、temp 的 `temp == 1.0f`、top_p 的 `p >= 1.0f`）。

## 十一、★ temp / top_k / top_p / dist 各自改什么

四个 `apply` 挤在一起看，就能解释"顺序即语义"：

- `temp`：正的 temp 只做 `logit /= temp`（除正数**保序**）；`temp <= 0` 是特例，把除最大值以外的 logit 全置 `-INFINITY`（退化成 greedy）。
- `top_k`：排序后直接写 `cur_p->size = k`，尾巴被**永久删掉**。
- `top_p`：先 `softmax` 算 `p`，再按累积和截断；因为 softmax 依赖尺度，**"temp 先还是 top_p 先"会得到不同的集合**。
- `dist`：`std::uniform_real_distribution<double>` 抽一次，然后线性扫描累积概率写 `selected`。

源码注释里还有一条工程细节：dist 用的是"边扫描边归一化"的单遍实现，注释说它在全量 gpt-oss 词表上比下面的双遍版本快约 3 倍。

<!-- src: src/llama-sampler.cpp -->
```c
static void llama_sampler_temp_impl(llama_token_data_array * cur_p, float temp) {
    if (cur_p->size == 0) {
        return;
    }

    if (temp <= 0.0f) {
        // find the token with the highest logit and set the rest to -inf
        size_t max_i = 0;
        float  max_l = cur_p->data[0].logit;

        for (size_t i = 1; i < cur_p->size; ++i) {
            if (cur_p->data[i    ].logit > max_l) {
                cur_p->data[max_i].logit = -INFINITY;
                max_i = i;
                max_l = cur_p->data[i].logit;
            } else {
                cur_p->data[i].logit = -INFINITY;
            }
        }

        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
}
//>> ---- src/llama-sampler.cpp:321-338 ----
static void llama_sampler_top_k_impl(llama_token_data_array * cur_p, int32_t k) {
    // if (k >= (int32_t)cur_p->size) {
    //     return;
    // }

    if (k <= 0) {
        return;
    }

    k = std::min(k, (int) cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        llama_token_data_array_partial_sort_inplace(cur_p, k);
    }

    cur_p->size = k;
}
//>> ---- src/llama-sampler.cpp:1549-1571 ----
static void llama_sampler_top_p_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_top_p *) smpl->ctx;

    if (ctx->p >= 1.0f) {
        return;
    }

    llama_sampler_softmax_impl(cur_p, false);

    size_t k = cur_p->size;
    auto * pdata = cur_p->data;

    auto & buf_sort = ctx->buf_sort;

    // if not sorted, try adaptive top-k sorting
    if (!cur_p->sorted && cur_p->size > 1024) {
        k = std::min<size_t>(256, cur_p->size);
        llama_token_data_array_partial_sort(*cur_p, k, buf_sort);
        pdata = buf_sort.data();
    } else if (!cur_p->sorted) {
        // small candidates -> sort inplace
        llama_token_data_array_partial_sort_inplace(cur_p, k);
    }
//>> ---- src/llama-sampler.cpp:1150-1214 ----
static void llama_sampler_dist_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_dist *) smpl->ctx;

    // edge cases
    if (cur_p->size == 0) {
        cur_p->selected = -1;
        return;
    }

    cur_p->selected = 0;

    std::uniform_real_distribution<double> dist(0.0f, 1.0f);

    if (cur_p->size == 1) {
        // keep the RNG state aligned with backend sampling, which draws once per output
        dist(ctx->rng);
        cur_p->data[0].p = 1.0f;
        return;
    }

    // max logit for numerical stability
    float max_l = cur_p->data[0].logit;
    if (!cur_p->sorted) {
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_l = std::max(max_l, cur_p->data[i].logit);
        }
    }

    // apply softmax to obtain the probabilities
    double sum_cum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum_cum += p;
    }

#if 1
    // sample from the obtained probabilities and normalize the probs in a single pass
    // this is ~3x faster on Mac with full gpt-oss vocab than the version below
    //
    const double rnd = dist(ctx->rng);

          double sum_run = 0.0f;
    const double sum_tgt = sum_cum*rnd;

    bool found = false;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (!found) {
            // accumulate probs until we reach the target sum
            sum_run += cur_p->data[i].p;
            if (sum_run >= sum_tgt) {
                cur_p->selected = i;
                found = true;
            }
        }

        // normalize probs
        cur_p->data[i].p /= sum_cum;
    }

    // fallback to the last token (don't think this can happen)
    assert(found);
    if (!found) {
        cur_p->selected = cur_p->size - 1;
    }
```

## 十二、grammar 与 dry：没有后端实现的那些节点

对比两张虚表就能看出"哪些采样器只能跑在 host 上"：`llama_sampler_grammar_i` 的 `backend_init` / `backend_accept` / `backend_apply` / `backend_set_input` / `backend_reset` / `copy_state` **全是 nullptr**；`llama_sampler_dry_i` 同样如此。（`penalties` 与 `logit_bias` 则是反例：它们在这一版里已经实现了 `backend_apply`。）

<!-- src: src/llama-sampler.cpp -->
```c
static struct llama_sampler_i llama_sampler_grammar_i = {
    /* .name              = */ llama_sampler_grammar_name,
    /* .accept            = */ llama_sampler_grammar_accept_impl,
    /* .apply             = */ llama_sampler_grammar_apply,
    /* .reset             = */ llama_sampler_grammar_reset,
    /* .clone             = */ llama_sampler_grammar_clone,
    /* .free              = */ llama_sampler_grammar_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};
//>> ---- src/llama-sampler.cpp:3624-3637 ----
static struct llama_sampler_i llama_sampler_dry_i = {
    /* .name              = */ llama_sampler_dry_name,
    /* .accept            = */ llama_sampler_dry_accept,
    /* .apply             = */ llama_sampler_dry_apply,
    /* .reset             = */ llama_sampler_dry_reset,
    /* .clone             = */ llama_sampler_dry_clone,
    /* .free              = */ llama_sampler_dry_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};
```

## 十三、一次采样的完整收尾

链跑完后，`llama_sampler_sample()` 断言 `selected` 落在 `[0, size)` 内，取出 token id，再调用 `llama_sampler_accept()` 把 token 交给链上每个节点更新状态。这条断言也解释了上一节的位置约束：会缩小 `size` 的节点（top_k / top_p）必须排在会写 `selected` 的节点（dist / greedy）**之前**。

<!-- src: src/llama-sampler.cpp -->
```c
    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    llama_sampler_apply(smpl, &cur_p);

    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);

    auto token = cur_p.data[cur_p.selected].id;

    llama_sampler_accept(smpl, token);

    return token;
```

---

## 说明

- 本课声明并引用了 8 个源文件：`llama-sampler.cpp` / `llama-sampler.h` / `llama-vocab.cpp` / `llama-vocab.h` / `unicode.cpp` / `unicode.h` / `unicode-data.cpp` / `unicode-data.h`，全部计入覆盖率。
- 散文里提到的 `include/llama.h`（`struct llama_sampler_i` 与 `llama_token_data_array` 的声明）、`src/llama-grammar.cpp`（grammar 的 apply 实现）与 `src/ggml-backend` 的 `ggml_backend_dev_supports_op()` 属于**其他课的覆盖域**，本课不引用其源码，因此不计入本课覆盖率。
- 第 3 幕的 BPE 合并演示（`(e,s) rank 4` 之类）是**示意值**，只用于说明"rank 小的先合并"这条规则；真实 rank 存在模型的 merges 表里，由 `llama_vocab::find_bpe_rank()` 查询。
