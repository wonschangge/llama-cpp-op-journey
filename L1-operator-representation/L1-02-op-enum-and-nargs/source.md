<!-- llama-coverage
ggml/src/ggml.c
ggml/src/ggml-impl.h
ggml/src/ggml.cpp
-->

# L1-02 · 算子枚举与元数据：算子的身份 — 源文件

**一句话**：在 ggml 里，一个算子不是一个函数指针，而是**枚举里的一个编号**；这个编号同时是名字表的下标、后端 `switch` 的判据、以及图节点上的 `op` 字段。

L1-01 讲的是同一个节点的**数据面**（`ne[]` / `nb[]` / `type`）。这一课讲**身份面**，回答三个问题：它是什么运算（`enum ggml_op`）、它有几个输入（构造器里的 `src[]`）、它的输出是什么形状（构造器里的 `ne[]`）。

---

## 一、身份的两张表：名字与符号

算子的身份是一个整数（`enum ggml_op`），但人要看名字、图要画符号 —— 这两个函数就是身份与文本之间的全部接口。注意它们都不做边界检查：`op` 越界会直接读越界内存，调用方必须保证传进来的是合法枚举值。

<!-- src: ggml/src/ggml.c -->
```c
const char * ggml_op_name(enum ggml_op op) {
    return GGML_OP_NAME[op];
//>> 直查表：op 编号就是数组下标（表在 ggml.c:991）
}

const char * ggml_op_symbol(enum ggml_op op) {
    return GGML_OP_SYMBOL[op];
//>> 供 ggml_graph_dump_dot() 画图用（ggml.c:7907）
}
```

## 二、★ GGML_OP_NAME：编号就是下标

名字表是"枚举值到字符串"的**直查表**：`GGML_OP_NAME[op]`。表的长度写成 `GGML_OP_COUNT`（枚举的哨兵值），表尾还有一次 `static_assert` 兜底。所以**加一个算子要改三处**：`enum ggml_op`、`GGML_OP_NAME`、`GGML_OP_SYMBOL`；枚举改了而表没跟上，编译直接失败。

本版本（`v0.5.0`）实测：枚举里 101 个真算子（`GGML_OP_NONE = 0` 到 `GGML_OP_GLU`，`ggml/include/ggml.h:493-602`），两张表各有 101 项，两处 `static_assert(GGML_OP_COUNT == 101, ...)`。

<!-- src: ggml/src/ggml.c -->
```c
static const char * GGML_OP_NAME[GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD_ID",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
//>> ---- ggml/src/ggml.c:1100-1106 ----

    "GLU",
};

static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT != 101");

static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {
```

## 三、输入个数：写在各算子的构造器里

课件正文里常说"查 nargs 表"，但**本版本的源码里没有这张表**：在 `ggml/` 目录下 grep `nargs` 无命中，`ggml_op_name()` / `ggml_op_symbol()` 之外也没有以 op 为下标的输入个数数组。真正的规则是：**构造器给几个 `src[i]` 赋值，这个算子就有几个输入**。

- `ggml_mul_mat`：2 个输入（`src[0] = a`、`src[1] = b`）；
- `ggml_add_id`：3 个输入（`src[0] = a`、`src[1] = b`、`src[2] = ids`）。

`GGML_MAX_SRC = 10`（`ggml/include/ggml.h:224`）只是 `src[]` 数组的长度上限，不是任何算子的输入个数。没被赋值的槽保持 `NULL`，因此 `src[]` 同时就是图的**入边表**。

<!-- src: ggml/src/ggml.c -->
```c
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
//>> ---- ggml/src/ggml.c:2121-2140 ----
struct ggml_tensor * ggml_add_id(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    return result;
}
```

## 四、输出形状规则：也在构造器里

`GGML_OP_MUL_MAT` 的完整规则只有两行：先用 `ggml_can_mul_mat()` 断言输入合法，再用 `ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] }` 决定输出形状，类型固定为 `GGML_TYPE_F32`（与输入类型无关）。

别的算子给别的答案：`ggml_add_id` 直接 `ggml_dup_tensor(ctx, a)`（同 a 的形状与类型），`ggml_glu_impl` 把第 0 维减半。**没有统一的形状表**，规则就写在各自构造器的那一两行里。

<!-- src: ggml/src/ggml.c -->
```c
static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
//>> ---- ggml/src/ggml.c:2906-2931 ----
static struct ggml_tensor * ggml_glu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        enum ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    if (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; for (int i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, NULL, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

## 五、不产生数据的算子：ggml_op_is_empty

同一个枚举还有第二种用法：**把 op 当分类标签**。`ggml_op_is_empty()` 只对 5 个 op 返回 `true`，其中 4 个是视图算子 —— 它们的构造器不新建数据，只改 `ne[]` / `nb[]`，张量的 `view_src` 因此非空（回顾 L1-01）。

<!-- src: ggml/src/ggml-impl.h -->
```c
static bool ggml_op_is_empty(enum ggml_op op) {
//>> 按 op 分类的第二种用法：这个算子会不会真的产生数据
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        default:
            return false;
    }
```

## 五之二、视图构造器长什么样：ggml_transpose

`ggml_transpose()` 是最短的一个视图构造器：先 `ggml_view_tensor(ctx, a)`（这一步让 `view_src` 指向 `a`），再交换 `ne[0]/ne[1]` 与 `nb[0]/nb[1]`，最后才写上 `op` 与 `src[0]`。**整个过程没有分配任何数据** —— 这正是 `ggml_op_is_empty()` 把它归为"空"的原因，也是 L4-01 的分配器可以跳过它的原因。

<!-- src: ggml/src/ggml.c -->
```c
struct ggml_tensor * ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);
//>> view_src = a：数据仍在 a 那里，这个张量只是换个读法
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
//>> 身份（op）与入边（src[0]）最后才写上
    result->src[0] = a;

    return result;
}
```

## 六、op_params：16 个 int32 的读写接口

算子的标量参数（轴号、eps、精度）不存在结构体里，而是塞进 `op_params` 这 64 字节。三个接口就是全部机制：一次 `memcpy` 写入，两个按 `int32` / `float` 取值的 getter。唯一的边界检查是那条 `assert`，而它只在 `assert` 打开的构建里生效。

<!-- src: ggml/src/ggml-impl.h -->
```c
static void ggml_set_op_params(struct ggml_tensor * tensor, const void * params, size_t params_size) {
    GGML_ASSERT(tensor != NULL); // silence -Warray-bounds warnings
    assert(params_size <= GGML_MAX_OP_PARAMS);
//>> 写入前先检查长度不超过 GGML_MAX_OP_PARAMS（64 字节）
    memcpy(tensor->op_params, params, params_size);
}

static int32_t ggml_get_op_params_i32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
//>> 取值前的边界检查：i 必须小于 16
    return ((const int32_t *)(tensor->op_params))[i];
}

static float ggml_get_op_params_f32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    return ((const float *)(tensor->op_params))[i];
}
```

## 七、★ 槽位约定只是注释：TAG_GGML_PREC

`op_params` 的槽位含义**没有类型系统表达**：它写在 `ggml-impl.h` 的一段注释里，由每个算子自己约定。这段注释只列了 `GGML_OP_MUL_MAT` 与 `GGML_OP_MUL_MAT_ID` 两个算子。

而实际写入者比注释更宽：`ggml_prec_set_acc()` 对 `GGML_OP_MUL_MAT` 写槽 `0`，对 `GGML_OP_FLASH_ATTN_EXT` 却写槽 `3`。**同一个槽号在不同算子下含义不同**，这正是"算子的身份决定它的元数据怎么解释"。（源码注释不一定是完整的：这里注释没有覆盖 `FLASH_ATTN_EXT`。）

<!-- src: ggml/src/ggml-impl.h -->
```c
// [TAG_GGML_PREC]
// - GGML_OP_MUL_MAT
//>> 约定按算子分节：同一个槽号在不同 op 下含义不同
//   0 - acc
//   1 - hint
//   2 - src0 precision
//   3 - src1 precision
//
// - GGML_OP_MUL_MAT_ID
//   0 - acc
//   1 - hint
//   2 - src0 precision
//   3 - src1 precision
static void ggml_set_op_params_i32(struct ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
//>> 唯一的边界检查：i < 16（GGML_MAX_OP_PARAMS / sizeof(int32_t)）
    ((int32_t *)(tensor->op_params))[i] = value;
}

static void ggml_set_op_params_f32(struct ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(tensor->op_params))[i] = value;
}
```

## 八、算子契约失败之后：GGML_ASSERT 与 terminate handler

每个构造器开头都用 `GGML_ASSERT` 写死了输入契约（例如 `ggml_mul_mat` 的 `GGML_ASSERT(ggml_can_mul_mat(a, b))`）。契约不满足时进程会 `abort()`。

`ggml/src/ggml.cpp` 是 C++ 侧的兜底：它在静态初始化时装一个 `std::terminate` 处理器，终止前先 `ggml_print_backtrace()` 打出回溯；`GGML_NO_BACKTRACE` 环境变量可以关掉这个行为。这就是"元数据写错 / 契约违反"时你实际看到的输出路径。

<!-- src: ggml/src/ggml.cpp -->
```c
#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

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

## 九、第七节的证据：同一个写入者的两个槽号

第七节说"同一个写入函数对另一个算子写另一个槽号"，证据就是下面这段：`ggml_prec_set_acc()` 的 `switch (a->op)` —— `case GGML_OP_MUL_MAT` 写槽 `0`，`case GGML_OP_FLASH_ATTN_EXT` 写槽 `3`，`default` 直接返回 `false`（表示这个算子不支持该精度设置）。**同一个槽号在两张"表"里意思不同**，判断依据只有 `a->op`。

<!-- src: ggml/src/ggml.c -->
```c
bool ggml_prec_set_acc(
        struct ggml_tensor * a,
        enum ggml_prec       prec) {
    switch (a->op) {
        case GGML_OP_MUL_MAT:
//>> switch 的判据是 a->op —— 身份决定元数据怎么解释
        case GGML_OP_MUL_MAT_ID:
            {
                const int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
//>> MUL_MAT / MUL_MAT_ID：精度写进槽 0
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                const int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
//>> FLASH_ATTN_EXT：精度写进槽 3
            }
            break;
        default:
//>> 不支持这个精度设置的算子返回 false
            return false;
    };

    return true;
}
```

---

## 说明

- 本课逐字引用 3 个文件：`ggml/src/ggml.c`、`ggml/src/ggml-impl.h`、`ggml/src/ggml.cpp`，三者都计入覆盖率（与计划 `plan/COVERAGE.md` 给 L1-02 指派的 3 个文件一致）。
- 正文与图示里有若干行号**只是指路**，它们没有被本课的引用块覆盖，也不计入覆盖率：`ggml/include/ggml.h:492`、`ggml/include/ggml.h:224`、`ggml/src/ggml-alloc.c:22`、`ggml/src/ggml-cpu/ggml-cpu.c:2253`、`ggml/src/ggml-cpu/ops.cpp:9351`、以及 `ggml/src/ggml.c` 内未被引用的定位行（7907 / 1844 / 1895 / 1963）。本课的覆盖率声明只有上一条那 3 个文件。
- "没有 nargs 表"是**实测结论**：在 `ggml/` 目录下 grep `nargs` 无命中。计划文档里"参数个数表 nargs"的措辞与本版本源码不符，本课按实测改写为"构造器里的 src[] 赋值"。
- 场景里的代码引用不使用注解行（`notes`），因为注解行会让代码区的行号槽整体错位；而本课的核心恰恰是"某条规则在哪一行"。注解改放在本篇 `source.md` 的引用块里。
