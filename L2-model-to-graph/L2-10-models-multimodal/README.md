# L2-10 · 模型家族（一）：多模态与视觉编码 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-06`（★ 计算图骨架 llama-graph）

## 学习目标

看完这一课，你应该能：

1. 说出本课 6 个文件各自的真实主题，以及它们为什么会落到"多模态"这一课（对应分组说明）；
2. 说出视觉塔的输出以什么形状喂给 LLM 部分（对应验收点）；
3. 解释 `llama_batch` 的 token 路径与 embd 路径在文本图的哪一步汇合；
4. 举出至少两个"文本图在构建时就区分媒体输入"的源码位置。

## 覆盖的源文件（11 个）

| 文件 | 行数 |
|---|---|
| `src/models/clip.cpp` | 19 |
| `src/models/models.h` | 2663 |
| `src/models/qwen4exp.cpp` | 1295 |
| `src/models/gemma4.cpp` | 499 |
| `tools/mtmd/models/gemma4v.cpp` | 134 |
| `tools/mtmd/mtmd-helper-common.h` | 185 |
| `src/models/deepseek4.cpp` | 1503 |
| `tools/mtmd/clip-graph.h` | 166 |
| `tools/mtmd/clip.cpp` | 6088 |
| `tools/mtmd/mtmd.cpp` | 2734 |
| `src/models/pockettts.cpp` | 147 |

> **说明**：**覆盖声明**：本课覆盖域内（`src/`）的文件是 6 个 —— `src/models/clip.cpp`、`src/models/deepseek4.cpp`、`src/models/gemma4.cpp`、`src/models/pockettts.cpp`、`src/models/qwen4exp.cpp`、`src/models/models.h`。它们计入覆盖率。
> **说明**：**不计入覆盖率**：本课另外引用了 `tools/mtmd/` 下的 5 个文件（`clip.cpp`、`clip-graph.h`、`mtmd.cpp`、`mtmd-helper-common.h`、`models/gemma4v.cpp`）作为"视觉塔真正在哪里"的对照。按 `tools/repo_universe.py` 的 `CORE_PATTERNS`，`tools/` 不在覆盖域内 —— 这些引用**被引用但不计入覆盖率**，此处明确声明，不虚报。
> **说明**：**分组来源**：本课 6 个文件来自静态扫描（图构建代码里出现 `clip_` / `vision` / `mmproj` / `image_` 之一）。实测：8 处命中中 5 处在注释里，3 处是 `qwen4exp.cpp` 的同一个元数据键名，**没有一处是图原语调用**；`clip_` 在 6 个文件里出现 0 次。因此本课不声称"这 6 个文件都是多模态模型"。
> **说明**：第 5~7 幕的形状链逐字引自 `tools/mtmd` 的代码；本课只解读形状流转，视觉塔的完整实现（40+ 个模型文件）不在本课范围内。

## 场景（9 幕）

1. **六个"多模态"文件，只有一个是 clip —— 而它是个存根** — 6 个文件来自静态扫描分组：图构建代码里出现 clip_ / vision / mmproj / image_ 之一。命中原语不等于"它就是多模态模型"。
2. **models.h：为什么 llama 侧必须留一个空壳** — 架构类都要满足 llama_model_base 的接口；CLIP 也用同一个接口声明，只是三个实现全部 [[noreturn]]。
3. **★ 图像是以"已算好的 embedding"进文本图的** — 源码注释把这句话写死了：an image arrives as an embd batch, so ubatch->token is null。
4. **★ 文本图在入口处分叉：ubatch.token ? sqrtf(n_embd) : 1.0f** — Gemma 4 用同一个指针判断"这一批是 token 还是图像 embedding"，并据此决定要不要缩放。
5. **视觉塔是另一张 ggml 图：先 conv2d 切 patch** — tools/mtmd 里的视觉塔有自己的 ggml_context 与 cgraph；它与文本图不共享节点，只共享"embedding 张量"这个概念。
6. **★ 形状全链：[n_embd, n_patches] 到 [n_mmproj_embd, n_tokens]** — 每一步的 ne 都来自源码：ViT -> pooler -> projector，最后成为文本图入口的宽度。
7. **拼接点：llama_batch 的 embd 字段** — 视觉塔的输出被拷成一条扁平 float 缓冲，再作为一个"没有 token 的 batch"交给 llama_decode。
8. **同一个 LLM，四种"多模态"接法** — 判据都不是 token id，而是 ubatch.embd 这个指针 —— 图在构建时就知道自己是不是在处理媒体输入。
9. **把这一课压成一张表** — 验收点：视觉塔的输出以什么形状喂给 LLM 部分。

## 核心结论

### ★ 视觉塔的输出形状

视觉塔是一张**独立的 ggml 图**。它的最后一个节点就是输出张量：

```text
ne = [n_mmproj_embd, n_tokens]      // F32
ne[0] = n_mmproj_embd  投影后的特征宽度
ne[1] = n_tokens       这张图产生的 token 数
```

这个张量被拷成一条扁平 float 缓冲（长度 `n_mmproj_embd * n_tokens`），填进 `llama_batch.embd`（此时 `tokens = nullptr`），并在文本图入口变成 `[n_embd, n_tokens]` 的输入张量。**宽度相等的约束由 mtmd 在启动时断言：`n_mmproj_embd == n_embd_inp`。**

### ★ 多模态不是新引擎，而是"多张图接在一起"

视觉塔有自己的 `ggml_context`、自己的 `ggml_cgraph`、自己的后端调度；它与文本图**不共享任何节点**，只共享一个交汇点：**一个已算好的 embedding 张量**。文本图对这件事的感知只有一处 —— 入口是 `ubatch.token` 还是 `ubatch.embd`。这与 L2-05 的 `llama_batch`（tokens / embd 二选一）是同一件事的两端。

### 本课 6 个文件的真实主题

| 文件 | 行数 | 真实主题 | 与多模态的关系 |
|---|---|---|---|
| `src/models/clip.cpp` | 18 | CLIP 量化存根 | 只是"让 llama-quantize 能打开 mmproj GGUF" |
| `src/models/models.h` | 2662 | 151 个模型结构体的公共声明头 | 存根的类型声明在这里 |
| `src/models/deepseek4.cpp` | 1502 | DeepSeek-V4 文本图 | vision variant 的 MoE 路由偏置 |
| `src/models/gemma4.cpp` | 498 | Gemma 4 文本图 | 图像 embedding 不做嵌入缩放的入口分支 |
| `src/models/pockettts.cpp` | 146 | PocketTTS 文本主干 | 音频侧：latent 由 mmproj 的 flow net 产生 |
| `src/models/qwen4exp.cpp` | 1294 | Qwen4 文本图 | 图像批次用占位 token id 顶替 |

**只有 `clip.cpp` 与"视觉"直接同名，而它恰恰是唯一没有图的那个。**

## 验收点

- [x] 保真门禁：18 处引用 —— 18 个引用块 / 24 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 11 项，无空课、无幻影；全局覆盖 87/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 48 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
