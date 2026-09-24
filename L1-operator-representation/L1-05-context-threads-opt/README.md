# L1-05 · 上下文、线程与优化器接口 — 课件说明

> 层：**L1 · 算子的表示** ｜ 前置课：`L1-01`（ggml 张量：算子的数据面）

## 学习目标

看完这一课，你应该能：

1. 画出 `ggml_context` 的 arena 布局：`mem_buffer` / `mem_size` / 对象头 / `offs` / `objects_end`（对应验收点）；
2. 说出"用 arena 而不是逐个 `malloc`"换来了什么、又换走了什么，并解释为什么这直接导致 `ggml-opt` 要同时持有四个 context；
3. 解释为什么 `mem_size` 可以按"张量个数 x `ggml_tensor_overhead()`"来预算（呼应 L1-01 的定长张量）；
4. 说出 `ggml-threading` 实际提供了什么、没有提供什么，以及它在 `ggml_init` 里的调用点；
5. 说出 `ggml_opt` 的两个训练入口：`ggml_opt_epoch`（一次训练/验证循环）与 `ggml_opt_fit`（最高层封装）。

## 覆盖的源文件（6 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-opt.h` | 257 |
| `ggml/src/ggml.c` | 8168 |
| `ggml/src/ggml-opt.cpp` | 1095 |
| `ggml/src/ggml-threading.cpp` | 13 |
| `ggml/src/ggml.cpp` | 27 |
| `ggml/src/ggml-threading.h` | 15 |

> **说明**：本课声明的源文件共 6 个：计划指派的 5 个（`ggml/src/ggml.cpp`、`ggml/src/ggml-threading.cpp`、`ggml/src/ggml-threading.h`、`ggml/src/ggml-opt.cpp`、`ggml/include/ggml-opt.h`），外加 `ggml/src/ggml.c`。额外引用后者的原因：`ggml_context` / `ggml_object` / `ggml_init` / `ggml_free` / `ggml_new_object` 的实现都在 `ggml.c` 里，不在前五个文件中；没有它，验收点"为什么用 arena 而非逐个 malloc"无法由引用代码直接支撑。`ggml.c` 本就由 L1-02 / L1-03 / L4-04 共享，重叠覆盖不影响覆盖度门禁（并集判定）。
> **说明**：**实测纠正记录**：计划文档原先把本课要点写作"`ggml_context` 的对象分配、**线程池**、`ggml-opt` 的训练接口"。实测结果：`ggml-threading.cpp` 只有 12 行，内容是全局 `std::mutex` + `ggml_critical_section_start/end` 两个封装，**没有线程池**；`ggml.cpp` 26 行，内容是进程级 `std::terminate` handler，**与 context 无关**。本课按实测内容撰写，并把这次纠正本身作为教学点（第 1、8、9、10 幕与第八、十节）。
> **说明**：本课不含任何命令行参数引用；提到 `GGML_NO_BACKTRACE` 是环境变量，不是 CLI 选项。

## 场景（10 幕）

1. **五个文件，一个问题：对象住在哪里** — context 管内存、threading 管并发、ggml-opt 管训练；ggml.cpp 其实是进程级兜底。
2. **★ struct ggml_context：一块内存 + 一条对象链表** — 七个字段就够了：有多大、在哪、是不是自己的、要不要分配数据、几个对象、头、尾。
3. **ggml_init：整块 arena 只申请一次** — context 结构自己是一次 malloc；arena 是一次 malloc；两件事都只做一次。
4. **★ 释放是 O(1)：一次 free，或者一次置空** — ggml_free 只 free 一次内存块；ggml_reset 连 free 都不做，把两个指针抹掉就整块可再用。
5. **★ ggml_new_object：bump 分配的全过程** — 取当前末端、对齐、算新位置、查预算、写对象头、串到链表尾 —— 没有查找，没有空闲表。
6. **★ arena 的代价：ggml-opt 为什么要四个 context** — 张量只能跟着整块 arena 一起死。要按生命周期分组，唯一的办法是分组开 context。
7. **arena 的另一面：内存要提前算出来** — mem_size 是 ggml_init 的参数，所以调用者必须先知道"会有几个张量"。
8. **ggml-threading 到底提供了什么：一把全局互斥锁** — 整个实现文件 12 行、整个头文件 14 行。它没有线程池，也没有任务队列。
9. **ggml.cpp：这个文件其实与 context 无关** — 26 行，只做一件事：没人接住的异常，先打 backtrace，再 abort。
10. **压成一张表：arena 换来了什么、换走了什么** — 一次分配、一次释放、O(1) 的建图；代价是没有按张量释放，以及内存要提前预算。

## 核心结论

### ★ 为什么是 arena，而不是逐个 malloc

`ggml_context` 把"对象分配"从"每次调用 malloc"变成"在一块预分配内存上做指针加法"：

| 维度 | 逐个 malloc | arena（ggml_context 的做法） |
|---|---|---|
| 分配 | 每次都要走分配器、找空闲块 | `mem_buffer + cur_end`，纯算术（`ggml_new_object`） |
| 释放 | 每个对象各自 free | 一次 `ggml_free`，或 `ggml_reset` 把指针抹掉 |
| 元数据 | 每个对象都可能带堆头 | 只有对象头 `offs` / `size` / `next` / `type` |
| 用量查询 | 需要记账 | `objects_end->offs + objects_end->size` |
| 按对象释放 | 可以 | **不可以** —— 只能整块来、整块走 |
| 内存大小 | 按需增长 | 必须提前预算 `mem_size`，超了就失败 |

最后两行就是代价。**`ggml-opt` 的四个 context（`ctx_static` / `ctx_cpu` / `ctx_compute` / `ctx_copy`）正是"不能按对象释放"的直接产物**：它用"分组开 context"来换"分组释放"。

### 三条 O(1)

```text
ggml_new_object : mem_buffer + cur_end          <- 分配 O(1)
ggml_free       : ggml_aligned_free(mem_buffer)  <- 释放 O(1)
ggml_reset      : begin = end = NULL             <- 复用 O(1)
```

**建图快到这个程度，是"把整张图放进一个 context"这个设计换来的。**但也正因为如此，L4-01 的 `ggml-alloc` 才必须存在：它要在 arena 之上再算一层"哪些张量可以复用同一段内存"，因为 arena 自己不会回收任何一个字节。

### 跨课呼应

| 本课的结论 | 依赖/展开于 |
|---|---|
| 张量能整块放进 arena | **L1-01**：`ggml_tensor` 是定长值类型，可 memcpy、可顺序摆放 |
| 图节点也住在 context 里 | **L1-03**：计算图与拓扑遍历（`ggml/src/ggml.c:7479` 的 `ggml_new_graph_custom` 同样走 `ggml_new_object`） |
| arena 之上还要一层复用 | **L4-01**：`ggml-alloc` 在 arena 之上做张量内存复用 |
| `no_alloc = true` 时数据在别处 | **L4-02 / L4-03**：后端 buffer 与调度器 |

## 验收点

- [x] 保真门禁：21 处引用 —— 21 个引用块 / 51 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 6 项，无空课、无幻影；全局覆盖 78/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 39 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
