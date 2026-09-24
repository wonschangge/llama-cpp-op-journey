# L2-04 · KV cache 与记忆家族 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-03`

## 学习目标

看完这一课，你应该能：

1. 列出 `llama_memory_i` 的虚函数，并把它们按"生命周期 / 序列操作 / 状态 IO"分成三组（对应验收点）；
2. 解释为什么 `init_batch()` 返回 context 而 `apply()` 才改状态，以及这个设计对"装不下"的意义；
3. 说出 `llama_kv_cells` 里存的是哪些元数据，并指出 K/V 张量其实在哪里；
4. 对照 iswa / dsa / msa / dsv4 / recurrent / hybrid 六种变体，各说出它解决的一个具体问题；
5. 解释为什么 SSM/RNN 模型不需要 KV cache，而需要一份固定大小的状态。

## 覆盖的源文件（24 个）

| 文件 | 行数 |
|---|---|
| `src/llama-kv-cache.cpp` | 2818 |
| `src/llama-kv-cache.h` | 465 |
| `src/llama-kv-cache-iswa.cpp` | 364 |
| `src/llama-kv-cache-iswa.h` | 156 |
| `src/llama-kv-cache-dsa.cpp` | 263 |
| `src/llama-kv-cache-dsa.h` | 140 |
| `src/llama-kv-cache-dsa-iswa.cpp` | 342 |
| `src/llama-kv-cache-dsa-iswa.h` | 135 |
| `src/llama-kv-cache-dsv4.cpp` | 2254 |
| `src/llama-kv-cache-dsv4.h` | 407 |
| `src/llama-kv-cache-msa.cpp` | 396 |
| `src/llama-kv-cache-msa.h` | 154 |
| `src/llama-kv-cells.h` | 558 |
| `src/llama-memory.cpp` | 60 |
| `src/llama-memory.h` | 130 |
| `src/llama-memory-hybrid.cpp` | 280 |
| `src/llama-memory-hybrid.h` | 141 |
| `src/llama-memory-hybrid-idx.cpp` | 684 |
| `src/llama-memory-hybrid-idx.h` | 161 |
| `src/llama-memory-hybrid-iswa.cpp` | 286 |
| `src/llama-memory-hybrid-iswa.h` | 142 |
| `src/llama-memory-recurrent.cpp` | 1325 |
| `src/llama-memory-recurrent.h` | 196 |
| `src/llama-model.cpp` | 3360 |

> **说明**：本课声明 24 个源文件：计划中 L2-04 的 23 个全部声明，另加 `src/llama-model.cpp`（memory 的分派点，`llama_model::create_memory()` 在 2274 行）。`src/llama-model.cpp` 在计划里归 L2-05，本课引用后一并计入覆盖，特此说明。
> **说明**：本课不引用 `src/llama-arch.cpp`，因此 `llm_arch_is_recurrent()` / `llm_arch_is_hybrid()` 里的具体架构清单不在本课展开，留给 L2-11（模型家族 · SSM 与线性注意力）。
> **说明**：本课不讨论任何命令行参数，参数门禁的真值集来自真实二进制的帮助输出。

## 场景（11 幕）

1. **为什么不是"一个 KV cache"** — KV cache 只是 LLM memory 的一种。源码把这一点写在接口的第一行注释里。
2. **★ llama_memory_i 的虚接口：三类职责** — 生命周期、序列操作、状态读写。整族 memory 的共同语言就是这 15 个纯虚函数。
3. **★ context 对象：apply() 是唯一改状态的入口** — init_* 只做规划，不落笔；真正写进 memory 的动作只有 apply()。
4. **★ cell：KV cache 的最小单位是"元数据"，不是 K/V** — 一个 cell 记的是"这个位置上是谁的 token、属于哪些序列"，K/V 本身在另一个张量里。
5. **slot_info：token 到 cell 的翻译表** — 建图之前必须先知道"第 i 个 token 写进哪个 cell"，这张表就是 slot_info。
6. **K/V 住在哪，图怎么读写它** — cache 自己拿 buffer，图只拿视图（get_k）、只写行（cpy_k）。
7. **iswa：把层切成两套 cache** — SWA 层的记忆只需要一个窗口，不需要整个上下文 —— 那就给它一个更小的 cache。
8. **一张表：六种变体各自解决什么问题** — 每一行的依据都是源文件里的注释原话，逐字收在 source.md 里。
9. **recurrent：没有 KV cache，只有固定状态** — SSM / RNN 的记忆与序列长度无关：每层一份固定大小的状态，按序列复制几行。
10. **hybrid：attention 层与 recurrent 层拼进一个 memory** — 混合架构的模型一层一个样：一部分层是 attention，一部分是 recurrent，memory 就按层分工。
11. **谁来选 memory：create_memory()** — 一张 switch(arch) 决定用哪个实现；剩下的课都在讲被它选中的那些类。

## 核心结论

### 抽象的是行为，不是数据布局

`llama_memory_i` 只描述"记忆对外的行为"：15 个虚函数，没有一个提到 K 或 V。数据面留在各自实现里 —— 所以同一个接口下既可以有按 token 记账的 KV cache，也可以有每层一份固定状态的 recurrent memory。

### ★ 规划与提交分离

`init_batch()` / `init_full()` / `init_update()` 只规划并返回 context；源码把"唯一能改 memory 的方法是 `apply()`"写成了硬约束。组合型 memory 靠 `llama_memory_status_combine()` 合并子 memory 的状态，对外表现得像一个 memory。

### ★ cell 是账本，张量是仓库

`llama_kv_cells` 存 `pos` / `ext` / `shift` / `seq` / `used` / `seq_pos`；K/V 在 `llama_kv_cache::layers[]` 的 ggml 张量里，由 cache 自己按后端 buffer type 整块分配（调度器不参与）。图侧只拿视图（`ggml_view_4d`）与写行（`ggml_set_rows`）。

### 一句话记住这一族

```text
KV cache   = 每个 token 一份记忆   -> llama_kv_cache 及其变体
recurrent  = 每个序列一份固定状态  -> llama_memory_recurrent
hybrid     = 按层路由到上面两者    -> llama_memory_hybrid
```

选择发生在 `llama_model::create_memory()`：一个 `switch (arch)`，加上 `llm_arch_is_recurrent()` / `llm_arch_is_hybrid()` 两个判定。

## 验收点

- [x] 保真门禁：37 处引用 —— 37 个引用块 / 57 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 24 项，无空课、无幻影；全局覆盖 844/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：11 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
