# L8-01 · ★ 端到端：一个 mul_mat 的完整旅程 — 课件说明

> 层：**L8 · 端到端** ｜ 前置课：`L7-06`

## 学习目标

看完这一课，你应该能：

1. 不看代码，复述一次 `ggml_mul_mat` 从 `build_attn` 到 CPU `vec_dot` 的完整链路（对应验收点）；
2. 说出这条链上每一跳的**判据**：`ggml_can_mul_mat` / `visited_hash_set` / `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` / `iface.graph_compute` / `switch (tensor->op)` / `type_traits_cpu[src0->type].vec_dot`；
3. 解释为什么同一个 `build_attn` 里的两次 `ggml_mul_mat` 可能落在不同后端（有权重 vs 没有权重）；
4. 拿到任意一个 ggml 算子，能说出它落到 CPU/CUDA 内核要经过哪几个文件、哪几个函数。

## 覆盖的源文件（9 个）

| 文件 | 行数 |
|---|---|
| `src/llama-graph.cpp` | 3916 |
| `ggml/src/ggml.c` | 8168 |
| `ggml/src/ggml-backend.cpp` | 2514 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 3945 |
| `src/models/llama.cpp` | 251 |
| `src/llama-context.cpp` | 4413 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 717 |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | 4109 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5857 |

> **说明**：本课覆盖 **9 个源文件**，全部计入覆盖率。其中 `src/llama-graph.cpp` 是计划里本课的主文件；按"这条链路"追加了 3 个：`ggml/src/ggml.c`（`ggml_mul_mat` 构造器 + `ggml_build_forward_expand`/拓扑序）、`ggml/src/ggml-backend.cpp`（切分、归属与调度执行）、`ggml/src/ggml-cpu/ggml-cpu.c`（CPU 分派与 `vec_dot` 选择）。
> **说明**：另外 5 个文件是这条链路的相邻跳，本课只取"这一跳"的几行：`src/models/llama.cpp`（`build_attn` 的调用点与最终 expand）、`src/llama-context.cpp`（decode 与 graph_compute）、`ggml/src/ggml-cpu/ggml-cpu.cpp`（CPU 接口表与入口）、`ggml/src/ggml-cuda/ggml-cuda.cu`（同一跳的 CUDA 实现，作对照）、`ggml/src/ggml-cpu/arch/x86/quants.c`（x86 版 `ggml_vec_dot_q4_0_q8_0`）。这些文件各自的专课会逐字展开：L2-06 / L2-07 / L5-01 / L6-01 / L5-03。
> **说明**：`ggml/src/ggml-cpu/arch/<arch>/quants.c` 的同名函数由 CMake 按架构选一份编译（`ggml/src/ggml-cpu/CMakeLists.txt` 的 `GGML_CPU_SOURCES`），本课引用的是 x86 那一份；换架构时 `type_traits_cpu[]` 里的绑定名不变，实现文件会换。

## 场景（10 幕）

1. **一课走完一条链路：ggml_mul_mat 的完整旅程** — 一次 mul_mat 要穿过 4 个世界、9 个文件；每一跳都有一个明确的判据。
2. **起点：模型 build 函数调用 build_attn** — 第 169 行把这一层的 wo 权重交出去；第 246 行才把整条链拉进图。
3. **★ 图上没有"执行"：一次调用 = 一个节点** — build_attn 第 2804 行 → build_lora_mm 第 1514 行 → ggml_mul_mat 第 1518 行。
4. **构造器：三条断言 + 一个输出形状** — ggml_mul_mat 的实现只有 15 行；判据全在 ggml_can_mul_mat 里。
5. **入图：hash set 决定"要不要重算"** — ggml_build_forward_expand 是入口，真正的拓扑序遍历在 ggml_visit_parents_graph。
6. **执行入口：decode 的三件事** — C API 只转发；llama_context::decode 决定"重建图 / 分配 / 计算"。
7. **★ 判据：权重住在哪个 buffer，op 就归哪个后端** — 调度器按固定优先级给每个节点找归属；我们这次命中的是"输入里有权重"这一条。
8. **从 split 到接口：backend_id → iface.graph_compute** — 切分把图切成连续段；每段拿到一个后端指针，执行就是一次虚表调用。
9. **★ CPU 侧：switch (tensor->op) → mul_mat → vec_dot** — 大 switch 的判据是 op；vec_dot 的判据是 src0->type。两个字段，两次分派。
10. **把这一课压成一张表：每一层负责哪一跳** — L1~L8 各管一段；回到起点那一行，这条链就闭合了。

## 核心结论

### 这条链路的四个世界

一次 `ggml_mul_mat` 穿过四个世界，每个世界只回答一个问题：

| 世界 | 问题 | 谁回答 |
|---|---|---|
| 建图（L1/L2） | 节点是什么、什么形状 | `ggml_mul_mat` 构造器 |
| 入图（L1） | 节点进图了没有、顺序如何 | `ggml_visit_parents_graph` + hash set |
| 调度（L3/L4） | 跑在哪个后端、内存谁给 | `ggml_backend_sched_*` |
| 内核（L5/L6/L7） | 具体用哪个 kernel | 后端自己的 `switch (op)` |

### ★ 每一跳都有一个判据

把这一课真正要记住的东西压成一行：

```text
能不能乘        -> ggml_can_mul_mat（三条 ne 关系）
进图几次        -> visited_hash_set
归哪个后端      -> src[0] 的 buffer usage（权重优先）
调哪个后端实现  -> backend->iface.graph_compute
走哪个 case     -> tensor->op
用哪个 kernel   -> type_traits_cpu[src0->type].vec_dot
```

**六个判据，六次"看某个字段"** —— 这就是 llama.cpp 把模型跑到设备上的全部机制骨架。

### ★ 同一个 op，两种归属

`build_attn` 的调用栈里其实有两次 `ggml_mul_mat`：

- `cur = build_lora_mm(wo, cur, wo_s)`（`llama-graph.cpp:2804` → `1518`）：`src[0] = wo` 是**权重**，归属由权重所在的后端决定（`ggml-backend.cpp:967`）；
- `kqv = ggml_mul_mat(ctx0, v, kq)`（`llama-graph.cpp:2715`）：两个输入都是**激活**，`backend_id_from_cur` 返回 `-1`，归属改由邻居与"支持输入最多的后端"决定（1058 / 1210-1246）。

同一个算子、同一张图，判据不同，落点就可能不同 —— 这也解释了 `offload_kqv` 这类开关为什么能直接把这一段按在 CPU 上（`llama-graph.cpp:2729-2732`）。

## 验收点

- [x] 保真门禁：19 处引用 —— 19 个引用块 / 63 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 9 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
