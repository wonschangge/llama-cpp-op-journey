# L2-08 · 采样器与词表：图算 logits，host 选 token — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-07`（logits 从哪来：decode 的图算完并把结果交回 host）

## 学习目标

看完这一课，你应该能：

1. 说出 `llama_sampler_chain` 里有哪些字段，并解释为什么它"不认识"任何一个具体采样器（对应验收点）；
2. 按执行顺序写出 `llama_sampler_chain_apply()` 做了什么，并说明为什么链的顺序会改变采样结果；
3. 说出 `temp` / `top_k` / `top_p` / `dist` 各自修改 `llama_token_data_array` 的哪个字段；
4. 用源码证据说明采样为什么跑在 host 上而不进图，以及 `grammar` / `dry` 为什么没有后端实现；
5. 描述 BPE 分词的三级结构（特殊 token 切分、按 rank 的优先队列合并、unicode 查表）。

## 覆盖的源文件（8 个）

| 文件 | 行数 |
|---|---|
| `src/llama-sampler.cpp` | 4386 |
| `src/llama-sampler.h` | 47 |
| `src/llama-vocab.cpp` | 4527 |
| `src/llama-vocab.h` | 207 |
| `src/unicode.cpp` | 1409 |
| `src/unicode.h` | 112 |
| `src/unicode-data.cpp` | 7035 |
| `src/unicode-data.h` | 21 |

> **说明**：本课声明并引用了 8 个源文件：`llama-sampler.cpp` / `llama-sampler.h` / `llama-vocab.cpp` / `llama-vocab.h` / `unicode.cpp` / `unicode.h` / `unicode-data.cpp` / `unicode-data.h`，全部计入覆盖率。
> **说明**：散文里提到的 `include/llama.h`（`struct llama_sampler_i` 与 `llama_token_data_array` 的声明）、`src/llama-grammar.cpp`（grammar 的 apply 实现）与 `src/ggml-backend` 的 `ggml_backend_dev_supports_op()` 属于**其他课的覆盖域**，本课不引用其源码，因此不计入本课覆盖率。
> **说明**：第 3 幕的 BPE 合并演示（`(e,s) rank 4` 之类）是**示意值**，只用于说明"rank 小的先合并"这条规则；真实 rank 存在模型的 merges 表里，由 `llama_vocab::find_bpe_rank()` 查询。

## 场景（9 幕）

1. **模型的两端：分词与采样** — 图只认 token id。进来时要分词，出去时要采样 —— 两端都在 host 上，都不在图里。
2. **分词第一步：特殊 token 先切开，再按类型分派** — tokenize() 先做一次切分（BOS/EOS、 这类特殊 token），再把每段交给具体 tokenizer。
3. **★ BPE 的合并顺序由一个 rank 决定** — llm_bigram_bpe 用优先队列按 rank 排序：rank 最小的相邻对最先合并，合并后只补两个新 bigram。
4. **unicode：类别判断与 NFD 折叠靠数据表** — 分词器要回答"这个字符是字母还是数字、要不要折叠重音"，答案是三张查表函数 + 五张生成表。
5. **一条链就是 host 上的一个 vector** — llama_sampler_chain 的全部状态：一个 samplers 数组、一块复用缓冲、两个计时计数。
6. **★ 链的执行顺序 = 数组顺序，一步不差** — llama_sampler_apply() 只做一次虚表转发；chain_apply() 按 samplers 的顺序对同一个 cur_p 反复调用。
7. **★ 顺序为何会改变结果：temp 保序，top_p 不保序** — temp 只是把 logits 除以 temp；top_k 截断数组；top_p 先 softmax 再按累积概率截断；dist 才是抽签。
8. **★ 为什么采样不进图：它操作的是 host 上的 float 数组** — 一次采样 = 把 logits 复制进 std::vector，然后在 host 上跑链。全程没有 ggml 算子。
9. **一次采样的全部：apply 一次，accept 一次** — 图给出 logits，采样器给出 token，并把 token 记进自己的状态（penalties 的计数、grammar 的栈）。

## 核心结论

### 图算 logits，host 选 token

`llama_decode()` 的产物是 `float logits[n_vocab]`。把它变成 token 的 `llama_sampler_sample()` 全程在 host 上：

```text
logits[]  ->  std::vector<llama_token_data> cur  ->  chain.apply(cur_p)  ->  cur_p.selected  ->  accept(token)
```

这个链上没有 ggml 算子、没有张量、没有 buffer —— 所以 CUDA / Metal 后端对它一无所知。

### ★ 链的顺序就是算法

`llama_sampler_chain_apply()` 是一个朴素的 for 循环，所有节点在**同一个** `llama_token_data_array` 上原地修改。因此：

- `temp`（除正数）与 `top_k`（只删尾巴）**保序**，可以互换位置；
- `top_p` 依赖 softmax 的**尺度**，与 `temp` 不可互换；
- `dist` / `greedy` 写 `selected`，必须排在会缩小 `size` 的节点之后 —— 否则 `llama_sampler_sample()` 的断言会被触发。

### ★ grammar 约束不需要后端参与

`llama_sampler_grammar_i` 的 `backend_init` / `backend_apply` 都是 `nullptr`：grammar 采样器只实现 host 侧的 `.apply()`，在 `llama_token_data_array` 上把不合语法的 token 的 `logit` 置为 `-INFINITY`（`llama_grammar_apply_impl()`，`src/llama-grammar.cpp:1355` 起，L2-09 逐字展开）。同理，`dry` / `xtc` / `mirostat` / `top_n_sigma` / `infill` / `adaptive_p` / `typical` 都没有后端实现。

这一版新增的 `[EXPERIMENTAL]` 后端采样是这条结论的正面证明：要让某个节点进图，必须为它单独建一张采样图（`llama_sampler_backend_probe_graph()`），并逐个算子问后端 `ggml_backend_dev_supports_op()`；而且链上只允许**前缀**上后端。

### 分词：一半是算法，一半是数据

`llama_vocab::impl::tokenize()` 分三段：`tokenizer_st_partition()` 先切特殊 token，再按 vocab 类型分派到 7 种 tokenizer，最后由具体 tokenizer 产出 token id。本课细看了 BPE：symbol 链 + 按 `rank` 排序的优先队列，rank 来自词表的 merges 表（`find_bpe_rank()`）。unicode 侧则是纯查表：`unicode_ranges_flags`（2273 行）、`unicode_ranges_nfd`（1828 行）、`unicode_map_lowercase` / `unicode_map_uppercase` （1433 / 1450 行）、`unicode_set_whitespace`（25 行）—— 合计 7009 行生成数据，放在 `unicode-data.cpp` 里。

## 验收点

- [x] 保真门禁：21 处引用 —— 21 个引用块 / 49 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 8 项，无空课、无幻影；全局覆盖 87/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 48 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
