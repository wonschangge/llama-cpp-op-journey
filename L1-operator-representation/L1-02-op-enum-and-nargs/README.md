# L1-02 · 算子枚举与元数据：算子的身份 — 课件说明

> 层：**L1 · 算子的表示** ｜ 前置课：`L1-01`（ggml 张量：算子的数据面）

## 学习目标

看完这一课，你应该能：

1. 说出算子的身份由哪些东西定义：`enum ggml_op` 编号、`GGML_OP_NAME` / `GGML_OP_SYMBOL` 两张表（对应验收点）；
2. 解释为什么"输入个数"在本版本源码里不是查表得到的，而是散在每个算子的构造器里；
3. 指出任意一个算子的"输出形状规则"写在哪个函数里（构造器的 `ne[]`，或 `ggml_dup_tensor` / 视图构造器）；
4. 说明 `op_params` 是 16 个 `int32` 的无 schema 元数据，槽位含义由算子自己约定，且注释并不完整。

## 覆盖的源文件（3 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml.c` | 8168 |
| `ggml/src/ggml-impl.h` | 796 |
| `ggml/src/ggml.cpp` | 27 |

> **说明**：本课逐字引用 3 个文件：`ggml/src/ggml.c`、`ggml/src/ggml-impl.h`、`ggml/src/ggml.cpp`，三者都计入覆盖率（与计划 `plan/COVERAGE.md` 给 L1-02 指派的 3 个文件一致）。
> **说明**：正文与图示里有若干行号**只是指路**，它们没有被本课的引用块覆盖，也不计入覆盖率：`ggml/include/ggml.h:492`、`ggml/include/ggml.h:224`、`ggml/src/ggml-alloc.c:22`、`ggml/src/ggml-cpu/ggml-cpu.c:2253`、`ggml/src/ggml-cpu/ops.cpp:9351`、以及 `ggml/src/ggml.c` 内未被引用的定位行（7907 / 1844 / 1895 / 1963）。本课的覆盖率声明只有上一条那 3 个文件。
> **说明**："没有 nargs 表"是**实测结论**：在 `ggml/` 目录下 grep `nargs` 无命中。计划文档里"参数个数表 nargs"的措辞与本版本源码不符，本课按实测改写为"构造器里的 src[] 赋值"。
> **说明**：场景里的代码引用不使用注解行（`notes`），因为注解行会让代码区的行号槽整体错位；而本课的核心恰恰是"某条规则在哪一行"。注解改放在本篇 `source.md` 的引用块里。

## 场景（8 幕）

1. **算子的身份：一个编号 + 两张表** — enum ggml_op 给每个算子一个编号；两张静态表把编号翻译成"日志里的名字"和"图里的符号"。
2. **★ GGML_OP_NAME：编号就是下标** — 名字表是"枚举值到字符串"的直查表；表的长度和枚举的哨兵值必须一致。
3. **输入个数：写了几个 src[i] 就是几个** — ggml_mul_mat 赋了两个 src；ggml_add_id 赋了三个 —— 数量由各自的构造器决定。
4. **输出形状规则：每个算子自己算** — ggml_can_mul_mat 先断言输入合法，再由一行 ne[] 决定输出形状；没有统一的形状表。
5. **同一个枚举的第二种用法：ggml_op_is_empty** — 把 op 当分类标签：只有 5 个算子"什么都不做"，其余 96 个都要真的算。
6. **★ op_params：16 个 int32，schema 只是注释** — op_params 没有结构体类型；槽位含义由每个算子自己约定，唯一的边界检查是一条 assert。
7. **新张量的默认身份：GGML_OP_NONE** — 张量出生时 op = NONE、src 全 NULL；构造器随后把身份写进去，它才成为图上的节点。
8. **把这一课压成一张表** — 三个问题，三种答案的存放位置；两张表尾的 static_assert 负责让它们不脱节。

## 核心结论

### 算子的身份 = 一个编号 + 两张平行表

`enum ggml_op`（`ggml/include/ggml.h:492-605`）给每个算子一个编号，`GGML_OP_NAME`（`ggml.c:991-1102`）与 `GGML_OP_SYMBOL`（`ggml.c:1106-1218`）以编号为下标直查。两张表都以 `GGML_OP_COUNT` 为长度，并各由一次 `static_assert(GGML_OP_COUNT == 101, ...)` 钉住 —— **加算子必须同步改三处，否则编译不过**。

### ★ 输入个数与输出形状：都在构造器里，没有表

```text
输入个数 = 构造器里赋值了几个 result->src[i]
输出形状 = 构造器里算出的 ne[]（或 ggml_dup_tensor / 视图构造器）
GGML_MAX_SRC = 10   只是 src[] 数组长度上限，不是任何算子的输入个数
```

`ggml_mul_mat`：2 个输入、输出 `{ a->ne[1], b->ne[1], b->ne[2], b->ne[3] }`、类型固定 F32。`ggml_add_id`：3 个输入、输出 = `ggml_dup_tensor(ctx, a)`。要回答"某个 op 的输入个数与输出形状规则在哪定义"，答案是**它的构造器**，不是某张表。

### 自检：给一个 GGML_OP_*，去哪找它的输入个数与输出形状

| 要回答的问题 | 去哪找 | 本课引用的例子 |
|---|---|---|
| 它是什么运算 | `enum ggml_op`（`ggml/include/ggml.h:492-605`） | `GGML_OP_MUL_MAT` |
| 它叫什么名字 | `GGML_OP_NAME[op]`（`ggml.c:991-1102`）+ `ggml_op_name()` | `"MUL_MAT"` |
| 图里画成什么符号 | `GGML_OP_SYMBOL[op]`（`ggml.c:1106-1218`）+ `ggml_op_symbol()` | `"x*y"` |
| **它有几个输入** | **它的构造器**里 `result->src[i]` 赋值了几个 | `ggml_mul_mat` 2 个（`ggml.c:3352-3353`）；`ggml_add_id` 3 个（`ggml.c:2135-2137`） |
| **它的输出是什么形状** | **它的构造器**里的 `ne[]`，或 `ggml_dup_tensor()` / 视图构造器 | MUL_MAT `ggml.c:3348-3349`；ADD_ID `ggml.c:2132`；GLU `ggml.c:2920` |
| 它的标量参数 | `op_params[16]`，槽位约定见 `ggml-impl.h:163-174` | MUL_MAT 的槽 0/1/2/3 |

换个说法：**名字与符号查表，输入个数与输出形状查构造器**。

### ★ op_params：约定在注释里，边界只有一条 assert

`op_params` 是 64 字节（16 个 `int32`），写入靠 `ggml_set_op_params_i32()`，唯一检查是 `i < GGML_MAX_OP_PARAMS / sizeof(int32_t)`。槽位含义写在 `ggml-impl.h:163-174` 的 `[TAG_GGML_PREC]` 注释里，**只覆盖 `MUL_MAT` / `MUL_MAT_ID`**；`ggml_prec_set_acc()` 对 `FLASH_ATTN_EXT` 用的是槽 `3`。换一个 op，同一槽号的意思就变了 —— 这是"身份决定元数据怎么解释"。

### 同一枚举的第三种用法：分类

`ggml_op_is_empty()`（`ggml-impl.h:90`）只对 `NONE / RESHAPE / TRANSPOSE / VIEW / PERMUTE` 返回 `true`。同类开关还有 `ggml_op_can_inplace()`（`ggml-alloc.c:22`）与 CPU 后端的 `ggml_get_n_tasks()`（`ggml-cpu.c:2253`）；**L5-01 会看到后端按 op 分派的那个大 `switch`**，它就是本课"编号即身份"的最终消费者。

## 验收点

- [ ] 保真门禁：18 个引用块全部逐字来自声明的源文件，且位置连续
- [ ] 覆盖度门禁：3 个源文件均被声明
- [ ] 参数门禁：无非法命令行参数
- [ ] 语法检查：通过
- [ ] 渲染门禁：8 幕，无 JS 错误、无布局溢出、交互可用
- [ ] 自检：不看代码，能说出 `struct ggml_tensor` 里哪些字段决定一次内核调用的形状

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
