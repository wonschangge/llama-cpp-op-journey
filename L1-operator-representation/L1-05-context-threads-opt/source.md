<!-- llama-coverage
ggml/include/ggml-opt.h
ggml/src/ggml.c
ggml/src/ggml-opt.cpp
ggml/src/ggml-threading.cpp
ggml/src/ggml.cpp
ggml/src/ggml-threading.h
-->

# L1-05 · 上下文、线程与优化器接口 — 源文件

**一句话**：`ggml_context` 不是一个"对象工厂"，而是**一块 arena（预分配的连续内存）+ 一条对象链表**。张量与计算图这些对象都**顺序摆进这块内存**，于是"建图"退化成指针加法，`ggml_free(ctx)` 是一次释放；代价是**没有释放单个张量的接口**。

本课先把 arena 讲透（它是后面所有内存话题的地基），再看同一层的另外三个文件：`ggml-threading` 提供了什么并发原语、`ggml-opt` 的训练接口长什么样、以及 `ggml.cpp` 里到底装了什么。

**一处实测纠正（重要）**：计划文档原本把这一课的要点写成"`ggml_context` 的对象分配、**线程池**"。逐行读完这五个文件后，实情是：`ggml-threading.cpp` **没有线程池**，它只提供一对全局临界区函数；`ggml_context` / `ggml_object` / `ggml_init` / `ggml_new_object` 的实现也**不在这五个文件里**，而在 `ggml/src/ggml.c`。这一课按实测来讲，并把这次纠正本身当成一个教学点：**按文件名或计划猜内容，会猜错。**

> 覆盖说明：本课需要 `ggml_context` 的实现来支撑"为什么用 arena"这条验收点，因此额外引用了 `ggml/src/ggml.c`（它本就由 L1-02 / L1-03 / L4-04 共享）。重叠覆盖不影响覆盖度门禁（并集判定）。

---

## 一、★ struct ggml_context：一块 arena + 一条对象链表

`ggml_context` 的全部状态就是这七个字段。它**不持有**任何"分配器"结构：没有空闲链表、没有分桶、没有引用计数 —— 只有一块内存（`mem_buffer` / `mem_size`）和一条把对象串起来的单链表（`objects_begin` / `objects_end`）。

每个对象前面都压着一个 `struct ggml_object` 头部，它记录两件事：**这个对象在 mem_buffer 里从哪开始**（`offs`）、**占多少字节**（`size`）。`GGML_OBJECT_SIZE` 就是头部自身的字节数 —— 后面所有的偏移都要加上它。

注意 `struct ggml_object` 末尾那个 `char padding[4]`：它不是数据，是为了让头部长度对齐（`ggml/src/ggml.c:1262` 另有一条 `static_assert`，断言头部大小是 `GGML_MEM_ALIGN` 的整数倍）。

<!-- src: ggml/src/ggml.c -->
```c
struct ggml_object {
    size_t offs;
    size_t size;

    struct ggml_object * next;

    enum ggml_object_type type;

    char padding[4];
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

//
// ggml context
//

struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

## 二、一次 malloc：ggml_init

`ggml_init` 里其实有**两次** malloc，但都不是"每个对象一次"：

```text
GGML_MALLOC(sizeof(struct ggml_context))   <- 结构体自己，一次
ggml_aligned_malloc(mem_size)              <- arena 整块，一次
```

第二次可以完全没有：如果调用者自带 `mem_buffer`，就直接借用，并把 `mem_buffer_owned` 置为 false（于是 `ggml_free` 也不会去 free 它）。

`mem_size` 为 0 是允许的，会被兜底成 `GGML_MEM_ALIGN`。函数开头那段临界区保护的是"首次调用时初始化时间系统"，这也是本课引用到的 `ggml-threading` 用法。

<!-- src: ggml/src/ggml.c -->
```c
struct ggml_context * ggml_init(struct ggml_init_params params) {
    static bool is_first_call = true;

    ggml_critical_section_start();

    if (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    struct ggml_context * ctx = GGML_MALLOC(sizeof(struct ggml_context));

    // allow to call ggml_init with 0 size
    if (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (struct ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
    };

    GGML_ASSERT(ctx->mem_buffer != NULL);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    return ctx;
}
```

## 三、O(1) 的释放：ggml_reset / ggml_free / ggml_used_mem

分配是一次，释放自然也是一次。三个函数都很短：

| 函数 | 做什么 | 复杂度 |
|---|---|---|
| `ggml_reset` | 把 `n_objects` / `objects_begin` / `objects_end` 各抹一次 | O(1)，且不 free |
| `ggml_free` | 若 `mem_buffer_owned` 则 free 一次，再放掉结构体自己 | O(1) |
| `ggml_used_mem` | 返回最后一个对象的末端 | O(1) |

`ggml_used_mem` 值得单独看一眼：**用量不需要记账，bump 指针的身高就是账本。**

<!-- src: ggml/src/ggml.c -->
```c
void ggml_reset(struct ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = NULL;
    ctx->objects_end   = NULL;
}

void ggml_free(struct ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(const struct ggml_context * ctx) {
    return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}
```

## 四、★ bump 分配：ggml_new_object

这是"建图 = 指针加法"的字面实现。整个函数只有六步，没有任何查找：

```text
cur_end  = 上一个对象的 offs + size        （空链表时为 0）
size_needed = GGML_PAD(size, GGML_MEM_ALIGN)
obj_new  = mem_buffer + cur_end            <- 新对象的位置
if (cur_end + size_needed + GGML_OBJECT_SIZE > mem_size) 失败
obj_new->offs = cur_end + GGML_OBJECT_SIZE  （跳过对象头）
obj_cur->next = obj_new；objects_end = obj_new
```

两处细节值得记住：

1. **越界只有失败，没有扩容。** 超出 `mem_size` 就告警返回 `NULL`（debug 版直接 `GGML_ABORT`）。
2. **`offs` 跳过对象头。** 负载从 `mem_buffer + obj_new->offs` 开始，而对象头本身在 `obj_new` 处 —— 这正是 `ggml_get_next_tensor` 里`(char *)tensor - GGML_OBJECT_SIZE` 能反推出头部的原因。

<!-- src: ggml/src/ggml.c -->
```c
static struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    if (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        return NULL;
    }
    if (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        return NULL;
    }

    if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        return NULL;
    }

    *obj_new = (struct ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;
```

## 五、★ arena 的两个后果：四个 context 与一份预算

`ggml-opt` 是 context 最重的客户。它的设计几乎全部由 arena 的性质决定：

**后果一：按生命周期分组。** `ggml_opt_context` 里有四个 `ggml_context` 字段。因为能释放的最小粒度就是"整个 context"，想让一批张量一起死，唯一办法是让它们共用一个 context。`ggml_opt_free` 里于是就是三次 `ggml_free`。

**后果二：内存必须提前预算。** `mem_size` 是 `ggml_init` 的参数，所以调用者必须先算出"会有几个张量"，再乘每个张量的固定开销 `ggml_tensor_overhead()`。这个乘法能成立，靠的是 L1-01 的结论：`ggml_tensor` 是定长值类型。

<!-- src: ggml/src/ggml-opt.cpp -->
```cpp
struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
//>> ---- ggml/src/ggml-opt.cpp:584-594 ----
void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}
//>> ---- ggml/src/ggml-opt.cpp:346-364 ----
    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
```

## 六、数据集：一个 arena + 一个后端 buffer

`ggml_opt_dataset_init` 把"元数据"和"数据"分开：

- `mem_size = 2 * ggml_tensor_overhead()`：这个 arena **只放两个张量的对象头**（data 与 labels），所以 `no_alloc = true`；
- 真正的数据由 `ggml_backend_alloc_ctx_tensors_from_buft` 按这两个张量的形状在后端 buffer 上分配。

于是 `ggml_opt_dataset_free` 也就是两件事：先还 buffer，再 `ggml_free(ctx)`。

<!-- src: ggml/src/ggml-opt.cpp -->
```cpp
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
```

## 七、复用：把整个 context 扔掉重建

arena 不能单独回收，但可以**整车换掉**。`ggml_opt_alloc` 在静态图路径下就是这么做的：先把旧的 `ctx_copy` 整个 `ggml_free`，再用新的预算 `ggml_init` 一个，然后在新 arena 里 `dup_graph` 出一份图副本。

这是 arena 世界里的"回收"：**没有 free 单个对象，只有 free 整个世界再开一个新的。**

<!-- src: ggml/src/ggml-opt.cpp -->
```cpp
    if (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
```

## 八、线程：ggml-threading 的全部内容

**实测纠正**：计划文档把这一课的要点写成"线程池"，但逐行读完这两个文件后可以确认 —— `ggml-threading.cpp` 一共 12 行，内容是一个全局 `std::mutex` 加两个薄封装；`ggml-threading.h` 一共 14 行，只声明这两个函数。**没有线程池、没有任务队列、没有 worker、没有原子计数器。**

本课引用到的调用点是 `ggml/src/ggml.c` 的 `ggml_init`，保护"首次调用初始化时间系统"那一小段。（CPU 侧真正的并行度在 L5 层的后端课里讲。）

<!-- src: ggml/src/ggml-threading.cpp -->
```cpp
#include "ggml-threading.h"
#include <mutex>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}
```

## 九、ggml-threading.h：整个头文件只有两个函数

把整个头文件贴在这里，因为"它只有两个函数"这句话只有看全文才能确认。`extern "C"` 包裹说明它是 C ABI —— 供 C 代码和别的语言绑定调用。

<!-- src: ggml/src/ggml-threading.h -->
```c
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

#ifdef __cplusplus
}
#endif
```

## 十、ggml.cpp：进程级兜底，与 context 无关

本课覆盖的五个文件里，`ggml.cpp` 是最容易被误判的一个。它 26 行，做的是把一个 `std::terminate` handler 装进进程：异常没人接住时先打 backtrace，再 abort。`GGML_NO_BACKTRACE` 环境变量存在时，这个 handler 干脆不装。

这也是"源码/计划不一定是真的"的另一个例子：**文件名暗示的内容与实际内容可以完全无关。**

<!-- src: ggml/src/ggml.cpp -->
```cpp
static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();
```

## 十一、训练接口：一个 epoch 里的三段调用

`ggml_opt_epoch` 是训练循环的骨架：前半段数据集做训练（`backward = true`），后半段做验证（`backward = false`）。每个 batch 都是同一套三步：`ggml_opt_alloc` 准备（必要时分配/重建图）-> `ggml_opt_dataset_get_batch` 拷数据 -> `ggml_opt_eval` 算。

最高层再包一层 `ggml_opt_fit`：它建 `ggml_opt_context`、按 epoch 循环、最后 `ggml_opt_free`。**文档里推荐的用法（`ggml-opt.h` 的 Intended Usage）就是"两个 context + `no_alloc`"。**

<!-- src: ggml/src/ggml-opt.cpp -->
```cpp
void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    struct ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}
```

---

## 说明

- 本课声明的源文件共 6 个：计划指派的 5 个（`ggml/src/ggml.cpp`、`ggml/src/ggml-threading.cpp`、`ggml/src/ggml-threading.h`、`ggml/src/ggml-opt.cpp`、`ggml/include/ggml-opt.h`），外加 `ggml/src/ggml.c`。额外引用后者的原因：`ggml_context` / `ggml_object` / `ggml_init` / `ggml_free` / `ggml_new_object` 的实现都在 `ggml.c` 里，不在前五个文件中；没有它，验收点"为什么用 arena 而非逐个 malloc"无法由引用代码直接支撑。`ggml.c` 本就由 L1-02 / L1-03 / L4-04 共享，重叠覆盖不影响覆盖度门禁（并集判定）。
- **实测纠正记录**：计划文档原先把本课要点写作"`ggml_context` 的对象分配、**线程池**、`ggml-opt` 的训练接口"。实测结果：`ggml-threading.cpp` 只有 12 行，内容是全局 `std::mutex` + `ggml_critical_section_start/end` 两个封装，**没有线程池**；`ggml.cpp` 26 行，内容是进程级 `std::terminate` handler，**与 context 无关**。本课按实测内容撰写，并把这次纠正本身作为教学点（第 1、8、9、10 幕与第八、十节）。
- 本课不含任何命令行参数引用；提到 `GGML_NO_BACKTRACE` 是环境变量，不是 CLI 选项。
