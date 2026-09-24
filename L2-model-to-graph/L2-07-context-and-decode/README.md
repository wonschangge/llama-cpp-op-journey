# L2-07 · 上下文与解码 llama-context — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-06`

## 学习目标

看完这一课，你应该能：

1. 按顺序列出 `llama_decode` 内部从建图到 `ggml_backend_sched_graph_compute_async` 的每一步（对应验收点）；
2. 说明 `sched_reserve()` 为什么不是每 token 的固定开销（`sched_need_reserve` 开关）；
3. 解释 `process_ubatch` 里那个 `if` 的三个条件，以及 graph reuse 命中时省掉了哪两件事；
4. 说出 `gf_res_prev` 为什么是长度为 2 的数组，以及 `output_ids` 在输出取出时的作用。

## 覆盖的源文件（2 个）

| 文件 | 行数 |
|---|---|
| `src/llama-context.cpp` | 4413 |
| `src/llama-context.h` | 402 |

> **说明**：本课只引用 `src/llama-context.cpp` 与 `src/llama-context.h` 两个文件，二者都计入覆盖率。
> **说明**：文中提到的 `llm_graph_result::can_reuse` / `llm_graph_params::allow_reuse` / `llm_graph_result::set_inputs` 定义在 `src/llama-graph.{h,cpp}`（L2-06 覆盖），本课只引用 `llama-context.cpp:1405`、`1449` 这两个调用点，故 `src/llama-graph.h` 不计入本课覆盖率。
> **说明**：文中提到的 `ggml_backend_sched_*` 函数名与 `ggml_backend_tensor_get_async` 均取自本课引用的 两个文件正文，未从注释推断。

## 场景（10 幕）

1. **★ llama_decode：图的生命周期管理器** — 建图、分配、执行、复用 —— 四件事都在 decode 里闭环。先看 llama_context 类自己的注释。
2. **C API llama_decode 只是转发** — 真正的逻辑在 llama_context::decode（1704 行）。C API 只负责打日志与把返回值传出去。
3. **balloc：batch 进，ubatch 的准备开始** — decode 的第一件事是把用户给的 llama_batch 校验并归一化；切分本身由 memory 模块接手。
4. **★ sched_reserve：最坏情况的图，只预留一次** — 它有一个开关 sched_need_reserve：第一次进来做完就关掉，之后每次 decode 直接返回。
5. **memory->init_batch：ubatch 的产地** — 切分这件事不在 context 里做，而是交给 memory 模块；context 只拿到一个 mctx 再问它要 ubatch。
6. **★ process_ubatch：复用还是重建，一个 if 决定** — 注释写得很直白：想复用它，图的完整拓扑必须由这些参数唯一确定。
7. **重建路径：建图 → 分配** — 只有这一条 else 分支里才会出现 build_graph 与 sched_alloc_graph —— 复用命中时它们根本不执行。
8. **喂输入：res->set_inputs → graph_compute** — 两条路径（复用 / 重建）在这里汇合：图已经就绪，接下来只需要把这一轮的 ubatch 填进去。
9. **终点：ggml_backend_sched_graph_compute_async** — graph_compute 自己不做计算：它只选线程池/线程数，然后把图整张交给调度器。
10. **一趟 decode 的结尾，与整条调用链** — 循环收尾后把 n_outputs 归位、建立 output_ids 映射，用户才能用 llama_get_logits_ith 取数。

## 核心结论

### ★ llama_decode 是图的生命周期管理器

`llama_context::decode`（1704 行）一个函数里完成图的完整生命周期：

| 阶段 | 调用 | 行号 |
|---|---|---|
| 建图 | `model.build_graph(gparams)` | 1425 |
| 分配 | `ggml_backend_sched_alloc_graph(sched, gf)` | 1435 |
| 执行 | `graph_compute` → `ggml_backend_sched_graph_compute_async` | 1454 → 2588 |
| 复用 | `res->can_reuse(gparams)` → `n_reused++` | 1405 / 1415 |

四件事都在 `process_ubatch`（1391 行）里收口，而每个 ubatch 只走一次。

### 一次 decode 的两级循环

- **逻辑批级（每 batch 一次）**：`balloc->init`(1765) → `sched_reserve`(1800) → `output_reserve`(1853) → `memory->init_batch`(1810)。
- **ubatch 级（每 ubatch 一次）**：`process_ubatch`(1887) → 建图/复用 → `set_inputs`(1449) → `graph_compute`(1454)。

循环由 `do { ... } while (mctx->next())`（1866 / 2041）驱动：**一个 `llama_batch` 可能产生多张图**，切分依据来自 memory 模块与 `cparams.n_ubatch`（回顾 L2-05）。

### ★ graph reuse：把"重建图"降到"复用图"

`process_ubatch` 里的判据是三个条件同时成立（1405 行）：

```text
!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)
```

命中时：不调用 `model.build_graph`、不调用 `ggml_backend_sched_alloc_graph`，只记录 `n_reused++`（1415）然后重填输入。
这就是逐 token 生成时每步开销能被压下来的地方。复用槽有**两个**（`gf_res_prev[n_outputs > 0]`，2423 行），让"有输出"和"无输出"的图各自稳定命中；可用环境变量 `LLAMA_GRAPH_REUSE_DISABLE` 整体关掉（280-281 行）。

## 验收点

- [x] 保真门禁：18 处引用 —— 18 个引用块 / 40 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 2 项，无空课、无幻影；全局覆盖 87/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 48 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
