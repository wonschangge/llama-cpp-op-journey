# L2-14 · 模型家族（五）：稠密 Transformer（上） — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-06`（计算图骨架与 `build_*` 原语）、`L2-02`（架构表与超参）

## 学习目标

看完这一课，你应该能：

1. 按顺序默写 `llama_model_llama::graph<embed>` 的**主干十站**（函数名 + 行号 + 作用）；
2. 说出主干上每一站对应的 `build_*` 原语，以及在哪一课讲它的实现；
3. 列举至少 5 个"层开关"，并各举一个本课文件作为反例（QKV bias / norm 位置 / tied embedding / sliding window / 激活）；
4. 判断一个 `src/models/*.cpp` 文件是不是"标准稠密主干"：看它有没有 `build_inp_embd`、层循环、`build_attn`、`build_ffn`、`build_lora_mm`；
5. 说明为什么 37 个文件里有 4 个"根本没有主干"（`using graph =` 继承），以及它们分别继承了谁。

## 覆盖的源文件（41 个）

| 文件 | 行数 |
|---|---|
| `src/models/apertus.cpp` | 171 |
| `src/models/arcee.cpp` | 158 |
| `src/models/baichuan.cpp` | 156 |
| `src/models/bitnet.cpp` | 171 |
| `src/models/bloom.cpp` | 152 |
| `src/models/chameleon.cpp` | 205 |
| `src/models/chatglm.cpp` | 162 |
| `src/models/codeshell.cpp` | 154 |
| `src/models/cogvlm.cpp` | 159 |
| `src/models/cohere2.cpp` | 160 |
| `src/models/command-r.cpp` | 145 |
| `src/models/dream.cpp` | 139 |
| `src/models/eagle3.cpp` | 339 |
| `src/models/eurobert.cpp` | 125 |
| `src/models/exaone.cpp` | 137 |
| `src/models/exaone4.cpp` | 191 |
| `src/models/falcon.cpp` | 162 |
| `src/models/gemma-embedding.cpp` | 177 |
| `src/models/gemma.cpp` | 140 |
| `src/models/gemma2.cpp` | 176 |
| `src/models/gemma3.cpp` | 224 |
| `src/models/gemma3n.cpp` | 465 |
| `src/models/gemma4-assistant.cpp` | 201 |
| `src/models/glm4.cpp` | 187 |
| `src/models/gpt2.cpp` | 149 |
| `src/models/gptneox.cpp` | 220 |
| `src/models/granite-switch.cpp` | 428 |
| `src/models/hrm-text.cpp` | 214 |
| `src/models/hunyuan-dense.cpp` | 7 |
| `src/models/hunyuan-vl.cpp` | 192 |
| `src/models/internlm2.cpp` | 140 |
| `src/models/jais.cpp` | 133 |
| `src/models/jais2.cpp` | 156 |
| `src/models/jina-bert-v2.cpp` | 67 |
| `src/models/jina-bert-v3.cpp` | 50 |
| `src/models/llada.cpp` | 154 |
| `src/models/llama-embed.cpp` | 7 |
| `src/llama-graph.cpp` | 3916 |
| `src/llama-graph.h` | 1391 |
| `src/models/models.h` | 2663 |
| `src/models/llama.cpp` | 251 |

> **说明**：本课覆盖声明共 41 项 = **L2-14 清单里的 37 个 `src/models/*.cpp`**（逐个出现在上面某一节的逐字引用、或本课的场景 `src=` 里）+ `src/models/llama.cpp`（主干的标准答案；plan 把它归给 L2-13，本课以它为主干逐字引用）+ `src/llama-graph.cpp` 与 `src/llama-graph.h`（与 `build_*` 原语对照，L2-06 的主文件，本课有意交叉引用）+ `src/models/models.h`（三处 `using graph =`，用来解释"6 行的模型文件"；该文件同时属于 L2-10 / L2-15）。
> **说明**：plan 文档写的验收点是"能默写出 `llm_build_llama` 的主干图"。`llm_build_llama` 是旧版 llama.cpp 的函数名；在 v0.5.0 里主干已改为 `llama_model_llama::graph<embed>` 的构造函数（`src/models/llama.cpp:99`）。本课按真实代码命名，并在末幕给出可默写的主干序列。
> **说明**：`src/llama-hparams.h` 与 `src/llama-model.cpp` 里的事实（`f_max_alibi_bias`、`swa_type`/`is_swa`、`create_tensor_qkv` 的 `TENSOR_NOT_REQUIRED`）只在叙述中被引用，**本课不引用其源码、不计入本课覆盖**（它们分别属于 L2-02 与 L2-05）。

## 场景（10 幕）

1. **37 个文件，一条主干** — 本课覆盖 src/models/ 下的 37 个文件；主干只有一条，差异都在"层的开关"上。
2. **主干的第 1 段：四样输入** — 主干开始之前，先把这一批 token 需要的东西全部备好：嵌入、位置、KV 入口、输出行。
3. **★ 主干第 2 段：逐层循环的注意力半边** — n_layer 次循环，每次做四件事：norm、QKV、RoPE、注意力。全部通过 build_* 原语完成。
4. **主干第 3 段：残差 + FFN（dense/MoE 就在这里分叉）** — 两个残差相加夹着一次 norm + 一次 FFN；走哪条 FFN 分支由一个指针是否为空决定。
5. **主干第 4 段：末层 norm + lm_head** — 循环结束后：一次 output_norm，然后接 lm_head；embed 模板为 true 时两处都被换掉。
6. **★ 主干上的每一站，都是一个 build_* 原语** — 模型文件里没有一个 ggml_* 算子：图被拆成 6 个原语，模型只负责"按顺序调用 + 选开关"。
7. **★ 六个"层开关"：同一行调用，两种行为** — gemma3 的主干与 llama 逐站对应，但多了 QK-norm、post-norm、SWA、softcap。
8. **37 个文件的真实差异（上：19 个）** — 逐行读出来的差异点。列"家族"里带 ★ 的，实际上不属于稠密 Transformer。
9. **37 个文件的真实差异（下：18 个）** — 下半张表里出现了"根本没有主干"的文件 —— 它们的图被一行 using 继承走了。
10. **把主干默写下来** — 十站、七个原语、六个开关。能默写这张表，这一课的验收点就达成了。

## 核心结论

### 主干十站（可默写）

```text
build_inp_embd(model.tok_embd)                     llama.cpp:108
build_inp_pos()                                    llama.cpp:111
build_attn_inp_kv()                                llama.cpp:119
build_inp_out_ids()                                llama.cpp:124
for il in 0 .. n_layer-1:                          llama.cpp:126
    build_norm(attn_norm, LLM_NORM_RMS)            llama.cpp:132
    build_qkv(...)                                 llama.cpp:143
    ggml_rope_ext(Q) / ggml_rope_ext(K)            llama.cpp:146 / 152
    build_attn(...)                                llama.cpp:169
    ggml_add(cur, inpSA)                           llama.cpp:178
    build_norm(ffn_norm, LLM_NORM_RMS)             llama.cpp:184
    build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)      llama.cpp:189
    ggml_add(cur, ffn_inp)                         llama.cpp:220
    build_cvec(cur, il)                            llama.cpp:223
build_norm(output_norm, LLM_NORM_RMS, -1)          llama.cpp:231
build_lora_mm(model.output, cur, model.output_s)   llama.cpp:240
ggml_build_forward_expand(gf, cur)                 llama.cpp:246
```

### ★ 所有稠密模型共享同一条主干，差异只在层的开关

这是 37 个文件能并成一课的原因。逐行读出并用得上的开关有六个：

| 开关 | llama.cpp 的拨法 | 反例 |
|---|---|---|
| norm 位置 | pre-norm（L132 / L184） | gemma3 post-norm（L162）、exaone4 无 pre-norm（L113） |
| QKV / wo bias | `build_qkv` 里有张量才加（L143） | bloom fused `wqkv_b`（L45）、command-r 只加 Q/K norm |
| QK-norm | 无（只有 `use_kq_norm`，L162） | gemma3（L131 / L139）、apertus（L93）、exaone4（L122） |
| sliding window | 无：`build_attn_inp_kv`（L119） | gemma2 / gemma3 / cohere2 / exaone4：iswa 入口 + `is_swa(il)` |
| FFN 激活 / 门控 | SILU + PAR（L194） | GELU / SwiGLU / ReLU² / 手写 xIELU（apertus L138） |
| embedding tied | `output` 缺失才复用 `tok_embd`（L41-46） | 恒 tied：gemma L20、cohere2 L29、command-r L21、bitnet L164 |
| 残差结构 | 串行两次相加（L178 / L220） | falcon L134-135、gptneox `use_par_res` L145、cohere2 L131 |

### 37 个文件不等于 37 个稠密 Transformer

本课的分组依据是**排除法**：这 37 个文件没有命中 multimodal / ssm / moe 三组图原语。逐个读完后必须如实标注：

- **4 个文件没有主干**：`hunyuan-dense.cpp` / `llama-embed.cpp`（6 行，`using graph =` 继承）、`jina-bert-v2.cpp` / `jina-bert-v3.cpp`（图在 `bert.cpp`）。
- **扩散式（非因果、无 KV cache）**：`dream.cpp`（L15 / L68）、`llada.cpp`（L16 / L82）。
- **编码器 / 句向量**：`eurobert.cpp`（无 lm_head）、`gemma-embedding.cpp`（`causal_attn = false`）。
- **草稿模型**：`eagle3.cpp`（两张图）、`gemma4-assistant.cpp`（只建 nextn 层）。
- **自成一派**：`granite-switch.cpp`（图内路由器 + 手写 FFN）、`hrm-text.cpp`（深度循环双栈）。
- **多模态家族**：`cogvlm.cpp`（视觉专家）、`gemma3n.cpp`（AltUp/Laurel）、`hunyuan-vl.cpp`（文本半边）。

主干十站在这些文件里大多仍在，只是被机制盖住了 —— 这也正是下一课要拆的东西。

## 验收点

- [x] 保真门禁：20 处引用 —— 20 个引用块 / 60 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 41 项，无空课、无幻影；全局覆盖 887/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
