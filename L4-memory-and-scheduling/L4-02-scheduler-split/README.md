# L4-02 · ★ 调度器：把图切给不同后端 — 课件说明

> 层：**L4 · 内存与调度** ｜ 前置课：`L4-01`（分配器 ggml-alloc）；建议先看 `L2-03`（权重落在哪个 buffer）、`L3-02`（设备顺序）

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml_backend_sched` 里"张量 -> 后端 id"和"张量 -> 副本"分别存在哪个字段里；
2. 复述 pass 1 的归属优先级（预分配 / 视图 / 图输入 / 权重），并说出 `op_offload` 如何把它反转；
3. 解释切分边界为什么必须插 copy：**一段子图只由一个后端执行，它的 kernel 只认自己 buffer type 上的内存**（对应验收点）；
4. 说出副本张量在哪个函数里、用哪个 API 造出来，真正的数据搬运又发生在哪个函数里。

## 覆盖的源文件（2 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-backend.h` | 438 |
| `ggml/src/ggml-backend.cpp` | 2514 |

> **说明**：本课引用 `ggml/src/ggml-backend.cpp` 与 `ggml/include/ggml-backend.h` 两个文件，均计入覆盖率。
> **说明**：第 6 幕里那张 6 节点小图是**示意**，不是源码：它的前提写在幕内（`op_offload` 关闭，所以带 host 权重的 `mul_mat` 留在 CPU）。示意里的节点名（n0/n1/...）只在本课内有效。
> **说明**：本课不引用 `ggml/src/ggml-backend-reg.cpp`（设备如何注册与排序）—— 那是 L3-02 的内容；也不引用 `ggml/src/ggml-alloc.c`（arena 怎么切）—— 那是 L4-01 的内容。

## 场景（10 幕）

1. **★ 调度器：把一张图切给多个后端** — ggml_backend_sched 把三件事绑在一个对象上：节点归谁、段间怎么搬、每段的内存谁来分。
2. **split = 连续区间 + 输入清单；sched = 账本** — 切分的结果不是一个新图，而是"splits[] 数组 + 每个张量的后端 id"。
3. **★ 归属优先级：预分配 &gt; 视图 &gt; 图输入 &gt; 权重** — backend_id_from_cur() 按固定顺序挑后端；一条都命中不了就返回 -1，交给后面的 pass。
4. **权重在哪个 buffer，op 就归哪个后端** — backend_from_buffer()：在有序后端表里找第一个"既支持这个 buffer type、又支持这个 op"的后端。
5. **★ op_offload：GPU 可以把 op 从 CPU 权重手里抢走** — 权重在 host 内存时，只要更靠前的后端支持这个 op 且愿意 offload，op 就归它 —— 代价是数据得搬过去。
6. **★ 切分 = 沿图找连续的同后端段** — 遍历拓扑序：节点后端和当前段不同就断一刀；节点带着"别的后端的权重"而本段又用不了它时，也断一刀。
7. **★ copy 是切分的产物，不是图里预先有的节点** — 输入不在本段后端、本段后端又用不了它的 buffer 时：造一个本后端的副本张量，把 node->src[j] 改指向它。
8. **切完之后：galloc 按段分配内存** — alloc_graph() = split_graph() + alloc_splits()：内存分配整个包给 L4-01 的 galloc。
9. **真搬运发生在 compute_splits：先搬 inputs，再跑子图** — 每个 split：优先用后端的异步 copy 把 inputs 搬进本后端，然后把 split->graph 整段交给这个后端。
10. **把这一课压成一张表** — 入口只有三个：split_graph（切）、alloc_graph（分配）、graph_compute（执行）。

## 核心结论

### 调度器只做三件事

`ggml_backend_sched` 的职责可以压成一张表：

| 事 | 谁干 | 本课行号 |
|---|---|---|
| 归属：每个张量归哪个后端 | `ggml_backend_sched_backend_id_from_cur` + pass 2/3/4 | 921-1295 |
| 切分：沿拓扑序断成连续段 | `ggml_backend_sched_split_graph` pass 5 | 1313-1362 |
| 搬运：段边界上造副本并搬数据 | `ggml_dup_tensor_layout` + `ggml_backend_tensor_copy` | 1399-1420 / 1782-1799 |

内存分配不在这里 —— 那是 L4-01 的 `ggml_gallocr_alloc_graph`；段内怎么执行也不在这里 —— 那是 L4-04 的 `ggml_backend_graph_compute_async`。

### ★ 切分不是"给节点分组"，而是"沿数据流找连续段"

pass 2 先把同后端的相邻节点**主动连成一片**，pass 5 才在连不起来的地方断刀。结果就是 `splits[]` 里的一段 = 图上的一个**连续下标区间**（`i_start` .. `i_end`）+ 它需要的输入清单。

**copy 是切分的产物**：节点本身没有被改写成一个"拷贝算子"，而是它的 `src[]` 指针被换成了下游后端的副本张量，真正的搬运在 `compute_splits()` 里执行。所以"边界为什么必须插 copy"的答案是：**一个段整体交给一个后端，而它的 kernel 只能读自己 buffer type 上的内存**（`ggml_backend_supports_buft`）；上游段在上游 buffer 里产出的张量，下游段根本读不到。

### ★ 权重的位置决定图的形状

`ggml_backend_sched_backend_id_from_cur()` 的第 4 条规则说：带权重的 op 优先和权重同后端。所以"哪些权重放 GPU"这个在加载期做出的决定（L2-03、L2-05 的 `dev_layer`），直接决定了图的切分结果。`op_offload` 是唯一的反转开关：权重留在 host，但 op 可以被更靠前的后端抢走 —— 代价是权重数据要在边界被搬一次。

这也是 L8-02 要实测的东西：给定 `n_gpu_layers`，预测图会被切成几段。

## 验收点

- [x] 保真门禁：30 处引用 —— 30 个引用块 / 84 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 2 项，无空课、无幻影；全局覆盖 1048/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
