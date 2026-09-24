# L5-01 · CPU 后端骨架：从 graph_compute 到算子分派 — 课件说明

> 层：**L5 · CPU 后端执行** ｜ 前置课：`L4-04`

## 学习目标

看完这一课，你应该能：

1. 说出一个 op 从进入 CPU 后端到调用具体内核经过的 **6 层**，以及每层所在的文件与函数（对应验收点）；
2. 解释 `cplan.n_threads = MIN(max_tasks, n_threads)` 的含义：为什么线程数由**图的内容**决定，而不是用户给的 `n_threads`；
3. 说出 `ggml_compute_forward()` 大 switch 里 102 个 case 的构成（96 内核 / 5 nop / 1 abort），并解释视图算子为什么是 nop；
4. 按 `ggml_get_n_tasks()` 的四种策略给一个 op 归类，说出它默认开几个任务；
5. 说明内核是怎么用 `params->ith` / `params->nth` 切分数据的，以及"静态切分"与"抢块"两种写法的取舍。

## 覆盖的源文件（3 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 717 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 3945 |
| `ggml/include/ggml-cpu.h` | 153 |

> **说明**：本课引用 3 个源文件：`ggml/src/ggml-cpu/ggml-cpu.c`（3944 行）、`ggml/src/ggml-cpu/ggml-cpu.cpp`（716 行）、`ggml/include/ggml-cpu.h`（152 行），全部计入覆盖率。
> **说明**：场景 3 的算例（n_threads = 8、5 个节点、max_tasks = 8）是按 `ggml_get_n_tasks()` 的真实逻辑手算的示例，不是实测输出。
> **说明**：文中提到但不逐字引用的位置：`struct ggml_compute_params` 定义在 `ggml/src/ggml-cpu/ggml-cpu-impl.h:18`；`ggml_cpu_extra_compute_forward()` 定义在 `ggml/src/ggml-cpu/traits.cpp:12`；`ggml_graph_compute_kickoff()` 在 `ggml-cpu.c:3288`；`struct ggml_threadpool_params` 在 `ggml/include/ggml.h:3003`。这些文件不计入本课覆盖率。

## 场景（10 幕）

1. **一个 op 到 CPU 后端，要穿过 6 层** — 本课只讲骨架：入口 -> 规划 -> 线程池 -> 分派 switch。具体内核在 L5-02~L5-05。
2. **接口表 16 个槽，只填了 6 个** — 10 个槽是 NULL：CPU 后端没有 async、没有 event、没有 graph_optimize。
3. **★ 规划阶段就把每个 op 的任务数问清楚了** — ggml_graph_plan 遍历整张图，逐节点问 ggml_get_n_tasks 要任务数，再据此算 work buffer。
4. **线程池：n_threads 个 worker 共享一个 cgraph** — 每个 worker 一份 ggml_compute_state（只有 ith 不同），共用同一个 threadpool。
5. **★ 每个线程跑同一个循环：params{ith, nth}** — 节点按拓扑序遍历，每个节点先试算子融合，没命中才交给 ggml_compute_forward。
6. **★ ggml_compute_forward：一个 switch 分派所有算子** — 102 个 case：96 个直接调用内核，5 个是 nop（视图算子），1 个是 COUNT 兜底。
7. **★ ggml_get_n_tasks：哪类 op 开几个任务** — 四种策略：全并行 n_threads、单任务 1、按数据量封顶 MIN(...)、由 op_params 决定。
8. **ith / nth 落到内核：抢块，不是静态切行** — mul_mat 用 ith 认领第一块，剩下的块用原子计数器 current_chunk 动态抢。
9. **ggml-cpu.h：CPU 后端的公共契约** — 线程池 5 个函数 + graph_plan/graph_compute 2 个函数 + 后端 API 5 个函数。
10. **把这一课压成一张表** — 一个 op 从进入 CPU 后端到调用内核，经过 6 层；其中两层决定"开几个任务"。

## 核心结论

### 一个 op 的 6 层旅程

| 层 | 函数 | 位置 |
|---|---|---|
| 1 接口 | `ggml_backend_cpu_graph_compute` | `ggml-cpu.cpp:170` |
| 2 规划 | `ggml_graph_plan` + `ggml_get_n_tasks` | `ggml-cpu.c:2815` / `2253` |
| 3 启动 | `ggml_graph_compute` | `ggml-cpu.c:3399` |
| 4 线程主体 | `ggml_graph_compute_thread` | `ggml-cpu.c:3109` |
| 5 分派 | `ggml_compute_forward`（大 switch） | `ggml-cpu.c:1744` |
| 6 内核 | `ggml_compute_forward_*` | `ops.cpp` / `unary-ops.cpp` ... |

其中**第 2 层和第 5 层**是本课的重点：一个决定"开几个任务"，一个决定"调哪个内核"。

### ★ CPU 后端 = 一个 switch + 一个线程池

`ggml_compute_forward()` 用 102 个 `case` 把 `tensor->op` 映射成内核调用：96 个直接调用、5 个 nop（`NONE` / `RESHAPE` / `PERMUTE` / `VIEW` / `TRANSPOSE`）、1 个 `GGML_OP_COUNT` 兜底 abort。

`ggml_get_n_tasks()` 用同样形状的 switch 决定任务数：**72 个 op 分支全并行、48 个单任务、4 个按数据量封顶、4 个由 `op_params` 决定**。

对照：GPU 后端是"每个 op 一个 kernel"，CPU 后端是"一个 switch 分派全部" —— 所以 CPU 后端没有 kernel 注册表，只有 `ggml_compute_forward` 和它的 96 个兄弟函数。

### ★ nth 是"最多开几个任务"，不是"必须开几个线程"

两层含义：

1. `ggml_graph_plan()` 把 `n_threads` 按全图最大任务数封顶：`cplan.n_threads = MIN(max_tasks, n_threads)`。一张只有归约算子的图，即使传 `n_threads = 16` 也只会开 1 个线程。
2. 内核拿到 `nth` 后**自己决定怎么切**：`mul_mat` 用 `ith` 认领第一块、再用原子 `current_chunk` 抢剩下的块（动态均衡）；多数逐元素内核则按 `ith` 静态取行。

这解释了 L1-02 那句"编号即身份"的最终消费者是谁 —— 就是这两个 switch。

## 验收点

- [x] 保真门禁：19 处引用 —— 19 个引用块 / 79 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 3 项，无空课、无幻影；全局覆盖 256/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 84 文件 15 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
