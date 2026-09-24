# L2-09 · 量化、导出与适配器 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-08`（采样器与词表 —— 本课第八幕的 grammar 就挂在它的采样链上）

## 学习目标

看完这一课，你应该能：

1. 说出 `llama_model_quantize_impl()` 里"类型"是在哪一步被逐个张量决定的；
2. 解释 imatrix 在量化中的角色，以及哪几类目标类型**必须**有它、缺了会怎样；
3. 说明 `llama_model_saver` 的 `add_kv` / `add_tensor` / `save` 三者的分工；
4. **说出 LoRA 是在哪一步、由哪个函数加进图的**（本课验收点），并说明基座权重有没有被改写；
5. 说明 grammar 与对话模板为什么都不在计算图里，它们各自改的是什么。

## 覆盖的源文件（11 个）

| 文件 | 行数 |
|---|---|
| `src/llama-quant.cpp` | 1487 |
| `src/llama-quant.h` | 2 |
| `src/llama-model-saver.cpp` | 506 |
| `src/llama-model-saver.h` | 46 |
| `src/llama-adapter.cpp` | 523 |
| `src/llama-adapter.h` | 92 |
| `src/llama-grammar.cpp` | 1527 |
| `src/llama-grammar.h` | 195 |
| `src/llama-chat.cpp` | 958 |
| `src/llama-chat.h` | 76 |
| `src/llama-graph.cpp` | 3916 |

> **说明**：本课按计划覆盖 10 个文件，全部计入覆盖率：`src/llama-quant.{cpp,h}`、`src/llama-model-saver.{cpp,h}`、`src/llama-adapter.{cpp,h}`、`src/llama-grammar.{cpp,h}`、`src/llama-chat.{cpp,h}`。
> **说明**：**额外引用 1 个清单外文件**：`src/llama-graph.cpp`（计划里归 L2-06「计算图骨架」）。理由：本课验收点是"能说出 LoRA 是在哪一步被加进图的"，而图内插入点 `llm_graph_context::build_lora_mm()` 只在 `src/llama-graph.cpp:1514` 定义。不引用它，这个结论就只能靠推断 —— 而本仓库的红线是"每个断言都必须能从源码看出来"。覆盖度门禁取的是并集，追加引用不会让任何文件失配（`llama-graph.cpp` 已由 L2-06 覆盖）。
> **说明**：`src/llama-quant.h` 全文 1 行（`#pragma once`），本课在 source.md 第十六节逐字引用，不计为"未引用但声明"。

## 场景（10 幕）

1. **模型进入图之前，有四件事会发生** — 量化改数值与类型，导出改容器，适配器改图；grammar 与对话模板则完全不碰图。
2. **★ 一个 ftype不是"所有张量同一个类型"** — ftype 只换算出一个默认类型；最终类型是逐张量算出来的：类别、层号、形状各有权重。
3. **★ imatrix：改的是误差往哪儿分，不是算法** — 低比特的 IQ* 家族必须有重要性矩阵，否则直接抛错 —— 因为"哪一列重要"本身是数据。
4. **逐行量化：chunk 不跨专家边界** — 一张权重矩阵按行切开，行号全局连续；但每个专家有自己的 imatrix 切片，所以切块要在边界处断开。
5. **导出：llama_model_saver 的三分工** — 登记元数据、登记张量、写盘是三个独立动作；结构体自己只持有"一个还没落盘的 GGUF"。
6. **★ LoRA 不是改权重，是在图里多插一次矩阵乘** — 基座权重 w 在 ggml_mul_mat 之后原封不动；每个命中的 adapter 再追加 a -> b -> scale -> add。
7. **适配器的全部状态：一张 ab_map** — 一套 adapter = "基座张量名 -> (lora_a, lora_b)" 的映射 + 一个 alpha；作用点由建图时按名字查表决定。
8. **grammar 不进图：它在采样链上改 logits** — 采样前把不可能的 token 打成 -INFINITY，采样后再按生成的文本推进规则栈 —— 两步都在图外。
9. **对话模板：一张 硬编码名字表 + 一条 if-else 链** — 模板不是从文件里解析出来的语法，而是几十个手写分支；每个分支把消息拼成一段 prompt 文本。
10. **把这一课压成一张表** — 量化与导出改的是"文件里的东西"，适配器与约束改的是"图与采样"——后者一个权重字节都不碰。

## 核心结论

### 三条改造路径，各自动什么

| 路径 | 入口 | 动的是什么 |
|---|---|---|
| 量化 | `llama_model_quantize_impl()` | 权重的**数值与 type**（还是 GGUF） |
| 导出 | `llama_model_saver::save()` | **容器**：把内存里的模型写成 GGUF |
| 适配器 | `llm_graph_context::build_lora_mm()` | **图**：多插几次矩阵乘，权重不变 |
| 约束 | `llama_grammar_apply_impl()` | 采样前的 **logits**（不进图） |
| 模板 | `llm_chat_apply_template()` | **prompt 文本**（决定图上看到什么 token） |

### ★ LoRA 是在哪一步进图的

在 `llama_context` **构建计算图**的那一步（`llama-context.cpp:1425` 的 `model.build_graph(gparams)`；图能复用时不会重建），由 `llm_graph_context::build_lora_mm()`（`src/llama-graph.cpp:1514`）完成：

```text
res = ggml_mul_mat(ctx0, w, cur)                       // 基座，w 只读
for lora in *loras:                                    // 每套挂着的 adapter
    lw = lora.get_weight(w)                            // 按名字查 ab_map
    if lw == nullptr: continue
    ab = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur))
    res = ggml_add(ctx0, res, ggml_scale(ctx0, ab, scale))
```

**它是图上的加法，不是权重里的改写。** 证据有三条：`w` 在函数里只作为 `ggml_mul_mat` 的输入出现；`*loras` 是一个循环，所以能同时挂多套；`llama_adapter_lora::get_n_nodes()` 直接把代价写成 `ab_map.size() * 6u`。

### imatrix 与逐行量化

imatrix 是"每一列有多重要"的**数据**（形状 `ne[0] x ne[2]`，按专家切片）。IQ 家族必须有它，Q2_K 只在 Q2_K_S 档位下需要，词嵌入与输出层永远豁免。

量化时一张矩阵按行切成 chunk，行号跨专家全局连续，但 **chunk 不跨专家边界** —— 所以 `imatrix_for_row()` 才能用 `row_global / nrows_per_expert` 直接算出该用哪张切片。

## 验收点

- [x] 保真门禁：26 处引用 —— 26 个引用块 / 49 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 11 项，无空课、无幻影；全局覆盖 87/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 48 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
