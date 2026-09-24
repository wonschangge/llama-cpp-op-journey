<!-- llama-coverage
ggml/src/ggml-cpu/unary-ops.h
ggml/src/ggml-cpu/ops.cpp
ggml/src/ggml-cpu/unary-ops.cpp
ggml/src/ggml-cpu/binary-ops.cpp
ggml/src/ggml-cpu/ops.h
-->

# L5-02 · ★ 一元与二元算子内核 — 源文件

**一句话**：一元与二元算子是"一个（或两个）元素进、一个元素出"的算子。正因为它们**没有归约、没有跨元素依赖**，CPU 后端才敢把"运算 x 类型"的笛卡尔积用 C++ 模板全部展开，让每个组合都变成一段独立、平凡、可向量化的循环。

本课覆盖 6 个文件：`unary-ops.{cpp,h}`（23 个一元内核）、`binary-ops.{cpp,h}`（加/减/乘/除）、`ops.{cpp,h}`（一元与二元的**分派器**，以及 DUP / ADD1 / SCALE / WIN_PART 这些特例）。`ops.cpp` 有 12206 行，本课只取其中 6 处，不试图覆盖全文件。

上游位置：`ggml/src/ggml-cpu/`，v0.5.0。上一课 L5-01 停在 `ggml_compute_forward` 的大 switch；本课就从那个 switch 把控制权交出去的地方开始。

---

## 一、一元算子的两条入口

`unary-ops.h` 声明了 23 个一元内核，签名完全一样：`(params, dst)`，没有返回值。但它们的"入口"有两条：其中 18 个由 `GGML_OP_UNARY` 的 switch 进入，另外 5 个（`sqr` / `sqrt` / `log` / `sin` / `cos`）在 `ggml-cpu.c` 里有自己的 `GGML_OP_*` case，直接调用同名函数。

这一点值得记住：**"一元算子"在 ggml 里不是一个单一的 op，而是一族 op 的统称**。`GGML_OP_UNARY` 只是其中"用 op_params 记一个子类型"的那一支。

<!-- src: ggml/src/ggml-cpu/unary-ops.h -->
```c
void ggml_compute_forward_abs(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sgn(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_neg(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_step(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_tanh(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_elu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_hardsigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_exp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_hardswish(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqr(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqrt(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sin(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_log(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_expm1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_softplus(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_floor(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_ceil(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_round(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_trunc(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_xielu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
```

## 二、★ 第一层分派：op 名 -> 函数名

`ggml_compute_forward_unary()` 的全部逻辑就是一个 switch。它唯一的输入是 `dst`：用 `ggml_get_unary_op(dst)` 从 `op_params` 里取出子类型，然后调用同名函数。

这一层**不看类型**，`GGML_TYPE_*` 的判断全部留给下一层。整个 switch 有 22 个 case，其中最后 4 个（GELU 家族与 SILU）指向 `ops.cpp`，其余 18 个指向 `unary-ops.cpp`。

<!-- src: ggml/src/ggml-cpu/ops.cpp -->
```c
//ggml_compute_forward_unary

void ggml_compute_forward_unary(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_unary_op op = ggml_get_unary_op(dst);

    switch (op) {
        case GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } break;
        case GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } break;
        case GGML_UNARY_OP_NEG:
```

## 三、★ 第二层与第三层：函数指针模板 x 类型对

`unary_op<op>` 的模板参数是**函数指针**，不是运行时参数。编译器看到 `unary_op<op_abs>` 时，会把 `op_abs` 直接内联进循环体，运行时没有一次间接调用。

函数体是一串 if-else：把 `(src0->type, dst->type)` 映射到 `apply_unary_op<op, src0_t, dst_t>`。一元算子只支持 5 种组合，其余组合走 `GGML_ABORT`。源码第 135 行的 TODO 原文是 `instead of a mass of 'if' conditions with long templates` —— 维护者自己把这条if-else 链记为技术债。

`binary-ops.cpp` 里是同一套写法，只是多了一个 `src1_t`（共 7 条组合）。

<!-- src: ggml/src/ggml-cpu/unary-ops.cpp -->
```c
// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}
```

## 四、★ 逐元素循环与行指针

内层循环一次只碰一个元素：`y[i] = f32_to_dst(op(src0_to_f32(x[i])))`。两个转换函数来自 `type_conversion_table<T>`，是 `constexpr` 的函数指针常量，会被内联掉。

外层是按行并行的循环：`get_thread_range` 给出本线程负责的行区间，行指针用 `i03*nb3 + i02*nb2 + i01*nb1` 算出 —— 这正是 L1-01 讲的 `ne[]` / `nb[]`。注意断言只要求**行内连续**（`ggml_is_contiguous_rows`），不要求整个张量连续。

<!-- src: ggml/src/ggml-cpu/unary-ops.cpp -->
```c
template <float (*op)(float), typename src0_t, typename dst_t>
static inline void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}
//>> ---- ggml/src/ggml-cpu/unary-ops.cpp:110-133 ----
template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}
```

## 五、二元算子：连续与广播

`vec_binary_op_contiguous` 与 `vec_binary_op_non_contiguous` 是两条循环。后者用 `i10 = i % ne10` 把 src1 的行**回绕复用** —— 广播在 CPU 上的实现就是重算指针，既不展开也不复制数据。

`apply_binary_op` 用 `ggml_is_contiguous_rows(src1)` 在运行时选一条路径；广播的合法性则由开头的 `GGML_ASSERT(ggml_can_repeat(src1, src0))` 把关（形状规则在 `ggml.c` 里，内核只是相信这个前提）。

<!-- src: ggml/src/ggml-cpu/binary-ops.cpp -->
```c
template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_contiguous(const int64_t n, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        int i10 = i % ne10;
        const src1_t * y_ptr = (const src1_t *)((const char *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);
    const bool is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);
```

## 六、二元的 7 条类型链，与唯一的 vDSP 例外

`binary_op<op>` 有 7 条类型组合分支。因为 `op` 是函数指针，`apply_binary_op` 里可以**直接拿它做比较**：在 Accelerate（macOS 系统框架）构建下，f32 的加/减/乘/除会被换成 `vDSP_vadd` / `vDSP_vsub` / `vDSP_vmul` / `vDSP_vdiv`。

这是二元算子里唯一的"按选项分派"，但它仍然是**编译期**开关（`GGML_USE_ACCELERATE`），不是运行时按 CPU 特性探测。

<!-- src: ggml/src/ggml-cpu/binary-ops.cpp -->
```c
#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (op == op_add) {
            vDSP_op = vDSP_vadd;
        } else if (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } else if (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } else if (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif
//>> ---- ggml/src/ggml-cpu/binary-ops.cpp:114-138 ----
// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float, float)>
static void binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    /*  */ if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_binary_op<op, float, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_binary_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_binary_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_binary_op<op, ggml_bf16_t, float, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_bf16_t, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_binary_op<op, ggml_fp16_t, float, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_fp16_t, float, float>(params, dst);
    } else {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}
```

## 七、ops.h / binary-ops.h：这一族的声明表

`ops.h` 是这一族的总目录：加/减/乘/除的入口、DUP、ADD1、SCALE，以及一元与二元的两个分派器（`ggml_compute_forward_unary` / `ggml_compute_forward_glu`）都在这里声明。所有函数都是 `extern "C"`，参数永远是 `(params, dst)`。

<!-- src: ggml/src/ggml-cpu/ops.h -->
```c
void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_acc(const struct ggml_compute_params * params, struct ggml_tensor * dst);
//>> ---- ggml/src/ggml-cpu/ops.h:51-51 ----
void ggml_compute_forward_scale(const struct ggml_compute_params * params, struct ggml_tensor * dst);
//>> ---- ggml/src/ggml-cpu/ops.h:96-99 ----
void ggml_compute_forward_win_part(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_unpart(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_unary(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
```

## 八、ops.cpp 里的三个特例

`ADD1`：第二个输入必须是**标量张量**（`ggml_is_scalar`），语义是"整行加同一个数"。

`SCALE`：**没有第二个输入张量**，scale 与 bias 从 `op_params` 里 memcpy 出来 —— L1-01 讲的那 64 字节在这里被真正使用。

`WIN_PART`：把一个张量切成 `w x w` 的窗口重排，索引是"窗口原点 + 窗口内偏移"，越界写 0。它标着 `TODO: optimize / multi-thread`，并且 `GGML_UNUSED(params)` —— 目前是单线程的。

关于 "sliding window"：这个词在 `ggml-cpu/` 下只出现一次（`ops.cpp:9734`，SSM 卷积的缓存窗口）。逐元素家族里的"窗口"变体是 `WIN_PART` / `WIN_UNPART` 这类分窗算子，不是注意力里的滑动窗口。

<!-- src: ggml/src/ggml-cpu/ops.cpp -->
```c
static void ggml_compute_forward_add1_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
//>> ---- ggml/src/ggml-cpu/ops.cpp:4697-4711 ----
static void ggml_compute_forward_scale_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float s; // scale factor
    float b; // bias

    memcpy(&s, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&b, (float *) dst->op_params + 1, sizeof(float));
//>> ---- ggml/src/ggml-cpu/ops.cpp:10030-10038 ----
    // TODO: optimize / multi-thread
    for (int py = 0; py < nep1; ++py) {
        for (int px = 0; px < nep0; ++px) {
            const int64_t i3 = py*nep0 + px;
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const int64_t i02 = py*w + i2;
                        const int64_t i01 = px*w + i1;
```

## 九、实测：这些 SIMD 到底从哪来

本课的向量化结论不是从源码"推断"的，而是在本机编译后看汇编得到的。命令与输出：

```text
$ g++ --version | head -1
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

$ g++ -O3 -mavx2 -mf16c -S -o /tmp/unary-ops.s \
      -I ggml/include -I ggml/src -I ggml/src/ggml-cpu \
      ggml/src/ggml-cpu/unary-ops.cpp        # 1.9 秒

$ sed -n "589,592p" /tmp/unary-ops.s        # ggml_compute_forward_abs 内部
.L14:
        vandps  (%rax,%rcx), %ymm2, %ymm0
        vmovups %ymm0, (%rdx,%rcx)
        addq    $32, %rcx

$ g++ -O3 -mavx2 -mf16c -c -o /tmp/unary-ops.o ... && size /tmp/unary-ops.o
   text    data     bss     dec     hex filename
  87060       0       0   87060   15414 /tmp/unary-ops.o

$ g++ -O3 -mavx2 -mf16c -c -o /tmp/binary-ops.o ... && size /tmp/binary-ops.o
   text    data     bss     dec     hex filename
  50519       0       0   50519    c557 /tmp/binary-ops.o
```

读数：**逐元素循环被自动向量化了**（`vandps` + ymm，一次 8 个 float，32 字节一组），而源码里一行 SIMD 内建函数都没有。同时可以看到模板展开的代价：单是 `unary-ops.cpp` 一个翻译单元，`.text` 就有 87060 字节。

**边界**：这是本机命令行编译的结果，用来说明"这份源码在这种编译方式下会被向量化"，不等于官方二进制一定是同样的指令选择（优化级别、目标 ISA、编译器版本都会改变结果）。

---

## 说明

- 本课引用 `unary-ops.cpp` / `unary-ops.h` / `binary-ops.cpp` / `binary-ops.h` / `ops.cpp` / `ops.h` 共 6 个文件，均计入覆盖率。
- 本课提到 `ggml-cpu.c`（L5-01 的覆盖面）、`vec.cpp` / `vec.h`（L5-03 的覆盖面）、`ggml.c` 与 `common.h` 时只给名字与行号，不引用其代码 —— 因此它们不计入本课覆盖率。
- 第九节的编译实测数据由本机 g++ 13.3.0 产出，不是上游构建产物；它的作用范围仅限"这份源码 + 这种编译方式"。
