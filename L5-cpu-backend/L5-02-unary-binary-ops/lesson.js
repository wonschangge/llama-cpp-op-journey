/* ==========================================================================
   L5-02 · ★ 一元与二元算子内核
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一元与二元：CPU 后端里最规整的一族 */
{
  kicker: "L5-02 · 全局",
  title: "一元与二元：CPU 后端里最规整的一族",
  sub: "23 个一元内核 + 4 个二元入口，共用同一套写法；本课把它们拆到模板层。",
  caption: "本课 6 个文件都在 ggml/src/ggml-cpu/ 下。L5-01 的 ggml-cpu.c 负责\"谁来做\"，本课负责\"怎么做\"。",
  src: "ggml/src/ggml-cpu/unary-ops.h",
  mark: [0, 1, 2, 3, 4, 5, 6],
  lineNo: 9,
  code: `void ggml_compute_forward_abs(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sgn(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_neg(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_step(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_tanh(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_elu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">L5-01 大 switch</span><span class="arrow">-></span>
        <span class="chip a">本课：一元 / 二元内核</span><span class="arrow">-></span>
        <span class="chip b">L5-03 vec.h SIMD</span>
      </div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'unary-ops.cpp / .h', b: '23 个逐元素一元内核，<br>共用一套模板与类型对', m: 'apply_unary_op<op, src0_t, dst_t>' },
      { c: 'b', t: 'binary-ops.cpp / .h', b: '加 / 减 / 乘 / 除 4 个入口，<br>连续与广播两条循环', m: 'binary_op<op_add>' },
      { c: 'c', t: 'ops.cpp / ops.h', b: '一元与二元的<b>分派器</b>，<br>外加 DUP / ADD1 / SCALE / WIN_PART', m: 'ggml_compute_forward_unary' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看左边这段声明：<span class="k">同一个签名，重复 23 次</span>。这不是复制粘贴的偶然，而是这一族算子的共同形状。',
      '<span class="v">unary-ops.h</span> 里 23 个函数：abs / sgn / neg / step / tanh / elu / relu / sigmoid / hardsigmoid / exp / hardswish / sqr / sqrt / sin / cos / log / expm1 / softplus / floor / ceil / round / trunc / xielu。',
      '<span class="v">binary-ops.h</span> 只有 4 个：add_non_quantized / sub / mul / div —— 二元算子几乎全部复用同一个模板。',
      '一元与二元的差别只有一处：<span class="k">一元读 1 个输入张量，二元读 2 个</span>。其余（线程切分、类型对展开、逐元素循环）写法完全一样。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15100, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '记住这个形状：<span class="v">fn(params, dst)</span> —— 没有返回值，输入从 dst->src[] 里取。';
    });
  }
},

/* ------------------------------------------------------ 2 ★ 第一层分派：<span class="hl-a">op 名 -> 函数名</span> */
{
  kicker: "L5-02 · 分派",
  title: "★ 第一层分派：<span class=\"hl-a\">op 名 -> 函数名</span>",
  sub: "ggml_compute_forward_unary 只做一件事：从 dst 里读出是哪个一元 op，然后 switch 到同名函数。",
  caption: "这一层不看类型。类型判断在下一层的模板里（第 3 幕）。注意 GELU / SILU 系列不在 unary-ops.cpp 里。",
  src: "ggml/src/ggml-cpu/ops.cpp",
  mark: [3, 7, 10, 11, 13, 18],
  lineNo: 10135,
  code: `//ggml_compute_forward_unary
//>> L5-01 的 ggml-cpu.c:2062 就是跳到这个函数

void ggml_compute_forward_unary(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_unary_op op = ggml_get_unary_op(dst);
//>> 唯一的输入：dst 自己记着这是哪个一元 op（L1-01 的 op_params 那 64 字节）

    switch (op) {
        case GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
//>> case 名 -> 同名函数，一一对应
            } break;
        case GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } break;
        case GGML_UNARY_OP_NEG:`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['GGML_UNARY_OP_*', 'ggml_compute_forward_*', '实现在'],
      [['ABS / SGN / NEG / STEP', 'abs / sgn / neg / step', 'unary-ops.cpp'],
       ['TANH / ELU / RELU', 'tanh / elu / relu', 'unary-ops.cpp'],
       ['SIGMOID / HARDSIGMOID / HARDSWISH', 'sigmoid / hardsigmoid / hardswish', 'unary-ops.cpp'],
       ['EXP / EXPM1 / SOFTPLUS', 'exp / expm1 / softplus', 'unary-ops.cpp'],
       ['FLOOR / CEIL / ROUND / TRUNC', 'floor / ceil / round / trunc', 'unary-ops.cpp'],
       ['XIELU', 'xielu（带 op_params 的函子）', 'unary-ops.cpp'],
       ['GELU / GELU_ERF / GELU_QUICK / SILU', 'gelu / gelu_erf / gelu_quick / silu', 'ops.cpp']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '整个函数的全部逻辑：<span class="k">取 op 号，switch</span>。22 个 case，每个 case 只做一次函数调用。',
      '<span class="v">ggml_get_unary_op(dst)</span> 把 dst->op_params 里的一个 int32 读出来 —— 这就是 L1-01 讲的"身份面"。',
      '前 6 组共 18 个 op 落在 <span class="k">unary-ops.cpp</span>：它们是"纯函数 + 类型对"的新族。',
      '最后一组 4 个（GELU 家族与 SILU）落在 <span class="k">ops.cpp</span>：它们是老族，走手写 SIMD 内核 —— 第 5 幕讲这个差别。',
      '<span class="v">unary-ops.h</span> 里那 23 个声明，有 5 个（sqr / sqrt / log / sin / cos）<b>不在这个 switch 里</b>：它们在 ggml-cpu.c 里有自己的 case（GGML_OP_SQR / GGML_OP_SQRT / GGML_OP_LOG / GGML_OP_SIN / GGML_OP_COS）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 3)];
    }));
    tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ 第二层分派：<span class="hl-a">函数指针模板</span> x <span class="hl-a">类型对</span> */
{
  kicker: "L5-02 · 核心",
  title: "★ 第二层分派：<span class=\"hl-a\">函数指针模板</span> x <span class=\"hl-a\">类型对</span>",
  sub: "unary_op<op_abs> 把\"做哪个运算\"变成模板参数；5 条 if-else 把\"(src0 类型, dst 类型)\"变成模板实参。",
  caption: "两层都是编译期的：op 是函数指针常量，类型对是 if-else 链 —— 运行时不做任何类型查表。",
  src: "ggml/src/ggml-cpu/unary-ops.cpp",
  mark: [1, 5, 6, 7, 14, 18, 22, 23],
  lineNo: 0,
  code: `// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
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
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}
//>> ---- ggml/src/ggml-cpu/unary-ops.cpp:237-243 ----
void ggml_compute_forward_abs(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia');
    const steps = [
      { c: 'a', t: 'ggml_compute_forward_abs', m: 'unary-ops.cpp:237' },
      { c: 'b', t: 'unary_op<op_abs>', m: 'unary-ops.cpp:136' },
      { c: 'c', t: 'apply_unary_op<op_abs, float, float>', m: 'unary-ops.cpp:110' }
    ];
    const els = steps.map(s => {
      const e = U.card({ c: s.c, t: s.t, m: s.m }, { style: 'width:196px' });
      dia.appendChild(e);
      if (s !== steps[steps.length - 1]) dia.appendChild(U.arrow('->'));
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');

    const host = wrap.querySelector('#cards');
    const defs = [
      { c: 'a', t: '23 个 op', b: 'unary-ops.h 的每个声明' },
      { c: 'b', t: 'x 5 种类型对', b: 'f32/f16/bf16 同型<br>+ bf16->f32 + f16->f32' },
      { c: 'c', t: '= 115 份特化', b: '每份都是一段独立机器码<br>（xielu 走函子版本）' },
      { c: 'd', t: '代价', b: '编译时间与二进制体积<br>源码自己写着 TODO' }
    ];
    const ces = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    ces.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把三层的名字连起来看：<span class="k">op 名 -> 模板实例 -> 类型对特化</span>。',
      '<span class="v">template &lt;float (*op)(float)&gt;</span>：模板参数是一个函数指针。编译器看到 <span class="k">unary_op&lt;op_abs&gt;</span> 就把 op_abs 的调用直接内联进去 —— 运行时没有一次间接调用。',
      '同一份模板被 5 条分支实例化：f32/f32、f16/f16、bf16/bf16、bf16/f32、f16/f32。<br><span class="k">多出来的组合（比如 f32 -> f16）直接 abort</span>，不是悄悄算错。',
      '23 个 op x 5 种类型对 = 115 份内层循环特化（其中 <span class="v">xielu</span> 走函子版本 <span class="v">vec_unary_op_functor</span>）。源码第 135 行的 TODO 原话是：<span class="m">instead of a mass of `if` conditions with long templates</span> —— 维护者自己也知道这是笔债。',
      '这一层是<b>编译期</b>的：类型对命中哪一条，在编译时就定下来。运行时只比较两个 <span class="v">enum ggml_type</span> 值。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    els.forEach((_, i) => tl.at(3000 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[2]; });
    ces.forEach((e, i) => tl.at(16400 + i * 1400, () => { e.style.opacity = '1'; }));
    tl.at(20500, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 ★ 为什么这一族最容易向量化：<span class="hl-c">平凡循环</span> + <span class="hl-c">行指针</span> */
{
  kicker: "L5-02 · 核心",
  title: "★ 为什么这一族最容易向量化：<span class=\"hl-c\">平凡循环</span> + <span class=\"hl-c\">行指针</span>",
  sub: "内层循环一次只碰一个元素、只依赖 x[i] —— 没有归约、没有跨元素依赖，编译器可以放手展开成 SIMD。",
  caption: "实测（本机 g++ 13.3，命令见 source.md 第九节）：ggml_compute_forward_abs 的 f32 循环被编译成 vandps ymm + vmovups，一次 8 个 float。",
  src: "ggml/src/ggml-cpu/unary-ops.cpp",
  mark: [1, 2, 4, 6, 7, 26, 31, 33, 35],
  lineNo: 100,
  code: `template <float (*op)(float), typename src0_t, typename dst_t>
static inline void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
//>> 源类型 -> f32 的转换函数，编译期常量
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
//>> 一行一个元素：只有 x[i] 参与，不读别的元素 —— 这是自动向量化的前提
    }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);
//>> 线程范围：每线程拿到一段连续的行（L5-01 的 nth/ith）

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
//>> 行指针 = data + i03*nb3 + i02*nb2 + i01*nb1 —— 就是 L1-01 讲的 ne/nb
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
//>> 按行调用：本线程负责的每一行，整行交给内层循环
    }
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="reg" style="gap:4px"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const reg = wrap.querySelector('#reg');
    const cells = [];
    for (let i = 0; i < 8; i++) {
      const c = U.el('div', { style: 'width:44px;height:30px;border:1px solid var(--border);border-radius:3px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--dim)' });
      c.textContent = 'x[' + i + ']';
      reg.appendChild(c); cells.push(c);
    }
    const lanecap = U.el('div', { class: 'formula', style: 'margin:0' });
    lanecap.innerHTML = '<span class="m">ymm</span>：一次处理 8 个 float（AVX2）';
    reg.appendChild(lanecap);

    const host = wrap.querySelector('#cards');
    const defs = [
      { c: 'a', t: '为什么能向量化', b: '每个输出元素只读对应的一个输入元素；<br>循环次数与数据无关。' },
      { c: 'c', t: '没有的东西', b: '没有累加器、没有跨 lane 的归约、<br>没有 data-dependent 的分支。' },
      { c: 'b', t: '实测', b: 'unary-ops.cpp 单独编译：<br>.text = 87060 字节，372 条 ymm 指令。' },
      { c: 'd', t: '谁在做', b: '编译器（自动向量化），<br>不是手写 kernel。' }
    ];
    const ces = defs.map(d => U.card(d, { style: 'width:163px' }));
    ces.forEach(e => host.appendChild(e));
    ces.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看内层循环：9 行，形状简单到不能再简单。',
      '<span class="v">y[i] = f32_to_dst(op(src0_to_f32(x[i])))</span>：读一个、算一个、写一个。<br>把两个转换函数内联掉之后，剩下的就是一个纯 elementwise 循环。',
      '外层再套一层<b>按行</b>的循环：行指针用 <span class="v">i03*nb3 + i02*nb2 + i01*nb1</span> 算出来（L1-01 的 nb[]）。<br>注意它不要求张量"完全连续"，只要求<b>行内连续</b>（第 114 行的 GGML_ASSERT）。',
      '于是编译器的活变得很轻松：把内层循环按 8 个 float 一组展开 —— <span class="k">vandps</span> 取绝对值，<span class="k">vmovups</span> 写回。',
      '对照一下：这里的 23 个一元算子，<b>没有一个</b>在源码里写 SIMD 内建函数。向量化来自编译器，而不是来自手写 kernel。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [6, 7, 9]); });
    tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, [2, 4, 7]); });
    tl.at(8600, () => {
      cells.forEach(c => { c.style.background = 'rgba(88,166,255,.18)'; c.style.borderColor = 'var(--a)'; c.style.color = 'var(--text)'; });
      msg.innerHTML = texts[2]; U.markLines(document, [26, 27, 28, 29, 31, 33]);
    });
    tl.at(14200, () => { msg.innerHTML = texts[3]; U.markLines(document, [35]); });
    tl.at(18800, () => {
      cells.forEach(c => { c.style.background = 'rgba(63,185,80,.18)'; c.style.borderColor = 'var(--b)'; });
      ces.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[4]; U.markLines(document, []);
    });
  }
},

/* ------------------------------------------------------ 5 ★ 两条 SIMD 路线：<span class="hl-a">编译器</span> vs <span class="hl-a">手写 vec_ 内核</span> */
{
  kicker: "L5-02 · 核心",
  title: "★ 两条 SIMD 路线：<span class=\"hl-a\">编译器</span> vs <span class=\"hl-a\">手写 vec_ 内核</span>",
  sub: "GELU 家族与 SILU 仍在 ops.cpp：它们调用 ggml_vec_* 手写内核，按编译期 ISA 宏分支。",
  caption: "所以回答\"unary op 的 SIMD 分发怎么做\"要分两半：新族靠自动向量化，老族靠 vec.h 的宏。两者都不是运行时按 CPU 特性选 kernel。",
  src: "ggml/src/ggml-cpu/ops.cpp",
  mark: [2, 6, 15, 33],
  lineNo: 2579,
  code: `static void ggml_compute_forward_silu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
//>> 老族只支持 f32 / f16 两种类型（第 2674 行的 switch 里只有这两个 case）

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
//>> ggml_vec_silu_f32：实现不在本文件，而在 vec.cpp / vec.h（L5-03 的主题）
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="paths" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const paths = wrap.querySelector('#paths');
    const left = U.el('div', { class: 'col grow', style: 'gap:7px' });
    left.innerHTML = '<div class="cap" style="font-size:9.5px;color:var(--dim)">新族：18 个 op（unary-ops.cpp）</div>';
    const right = U.el('div', { class: 'col grow', style: 'gap:7px' });
    right.innerHTML = '<div class="cap" style="font-size:9.5px;color:var(--dim)">老族：4 个 op（ops.cpp）</div>';
    paths.appendChild(left); paths.appendChild(right);

    const lc = U.card({ c: 'a', t: 'vec_unary_op', b: '普通 for 循环<br>无 SIMD 内建函数', m: 'unary-ops.cpp:101' }, { style: 'width:100%' });
    const lc2 = U.card({ c: 'b', t: '谁来向量化', b: '编译器自动向量化<br>（-O3 下 ymm 指令）', m: 'vandps / vmovups' }, { style: 'width:100%' });
    const rc = U.card({ c: 'c', t: 'ggml_vec_silu_f32', b: '手写内核，按 ISA 宏分支', m: 'vec.cpp:380' }, { style: 'width:100%' });
    const rc2 = U.card({ c: 'd', t: '谁来选分支', b: '编译期宏：AVX512F / AVX2+FMA /<br>SSE2 / SVE / NEON', m: '#if defined(__AVX2__)' }, { style: 'width:100%' });
    left.appendChild(lc); left.appendChild(lc2);
    right.appendChild(rc); right.appendChild(rc2);
    const allc = [lc, lc2, rc, rc2];
    allc.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '同一个 <span class="v">GGML_OP_UNARY</span>，在 CPU 后端里有两条实现路线。',
      '<b>新族</b>（18 个 op）：源码只有平凡循环，<span class="k">SIMD 由编译器生成</span>。想看代码里的向量化，得去看汇编。',
      '<b>老族</b>（GELU / GELU_ERF / GELU_QUICK / SILU）：调用 <span class="v">ggml_vec_gelu_f32</span> / <span class="v">ggml_vec_silu_f32</span>，那些函数里写着 _mm256_* 之类的内建函数。',
      '老族的"选分支"发生在<b>编译期</b>：同一份 vec.cpp 在 x86 构建里走 AVX2 分支，在 ARM 构建里走 NEON 分支。<br>它不是运行时按 CPU 特性选 kernel。',
      '要看<b>真正运行时</b>的 SIMD 分派（按 CPU 特性位选 kernel），得去 L5-04 的 repack / traits 与 L5-05 的多架构内核。'
    ];
    allc.forEach((e, i) => tl.at(800 + i * 3300, () => {
      allc.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14800, () => { allc.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 同一套展开，在 <span class="hl-b">ops.cpp</span> 里再来一份：DUP 的类型对表 */
{
  kicker: "L5-02 · 类型组合",
  title: "同一套展开，在 <span class=\"hl-b\">ops.cpp</span> 里再来一份：DUP 的类型对表",
  sub: "DUP 要支持\"任意非量化类型 -> 任意非量化类型\"，于是模板实参从 2 个变成 (src_t, dst_t)。",
  caption: "DUP 是 L1-04 与 L5-04 的接口：量化类型在这里被反量化或量化，块布局就在这些模板里。",
  src: "ggml/src/ggml-cpu/ops.cpp",
  mark: [0, 6, 15, 19],
  lineNo: 526,
  code: `void ggml_compute_forward_dup(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (src0->type == dst->type) {
//>> 同类型 -> 直接按字节搬（dup_bytes），根本不进模板
        ggml_compute_forward_dup_bytes(params, dst);
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                /**/ if (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
//>> 类型对写成了实参：<ggml_fp16_t, ggml_fp16_t>、<ggml_fp16_t, ggml_bf16_t> …
                else if (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                else if (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, float      >(params, dst);
                else ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
//>> 目标是量化类型 -> 走 dup_to_q（L1-04 的块布局在这里被写入）
            } break;
        case GGML_TYPE_BF16:`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['src0 \ dst', 'F32', 'F16', 'BF16', 'I32', '量化类型'],
      [['F32', 'flt<float,float>', 'flt<float,fp16>', 'flt<float,bf16>', 'flt<float,int32>', 'to_q<float>'],
       ['F16', 'flt<fp16,float>', 'flt<fp16,fp16>', 'flt<fp16,bf16>', '—', 'to_q<fp16>'],
       ['BF16', 'flt<bf16,float>', 'flt<bf16,fp16>', 'flt<bf16,bf16>', '—', 'to_q<bf16>'],
       ['量化类型', 'from_q<...>', '—', '—', '—', 'from_q / to_q']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这张表不是文档，而是<b>源码里手写的 if-else 链</b>（第 537 行的 switch + 每个 case 里的 4 条分支）。',
      '<span class="v">DUP</span> 是"类型转换"算子：同类型走 memcpy 快路，跨类型走 <span class="k">dup_flt&lt;src_t, dst_t&gt;</span>。',
      '所以模板实参的个数由算子的输入个数决定：一元 2 个（op + 1 个类型），DUP 2 个类型，二元 4 个（op + 3 个类型）。',
      '量化类型不在 fast 路径上：<span class="v">dup_to_q</span> / <span class="v">dup_from_q</span> 会调用类型的 to_float / from_float —— 这正是 L1-04 讲的那些块函数。',
      '一句话：<span class="k">"类型 x 运算"的笛卡尔积，在 CPU 后端里是逐条手写的</span>。代价是源码冗长，收益是零运行时开销。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 3000, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(15200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 二元算子：<span class="hl-d">连续</span> 与 <span class="hl-d">广播</span> 两条循环 */
{
  kicker: "L5-02 · 二元",
  title: "二元算子：<span class=\"hl-d\">连续</span> 与 <span class=\"hl-d\">广播</span> 两条循环",
  sub: "两个输入形状相同时指针同步走；src1 更小时用 i % ne10 回绕 —— 这就是广播。",
  caption: "广播的合法性由第 54 行的 GGML_ASSERT(ggml_can_repeat(src1, src0)) 把关；违反就直接 abort。",
  src: "ggml/src/ggml-cpu/binary-ops.cpp",
  mark: [1, 6, 7, 13, 19, 31, 40],
  lineNo: 25,
  code: `template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_contiguous(const int64_t n, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
//>> 连续路径：x、y、z 三个指针同步前进，一次一个元素
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        int i10 = i % ne10;
//>> i10 = i % ne10：src1 那一行被反复回绕使用 —— 广播的"滑动窗口"读法
        const src1_t * y_ptr = (const src1_t *)((const char *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));
//>> 广播契约：src1 必须能按维重复出 src0 的形状

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);
    const bool is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);
//>> 按行判断 src1 是否行内连续，决定这一行走哪条循环`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="lanes" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const lanes = wrap.querySelector('#lanes');
    function lane(title, color, mod) {
      const box = U.el('div', { class: 'col grow', style: 'gap:6px' });
      box.innerHTML = '<div class="cap" style="font-size:9.5px;color:var(--dim)">' + title + '</div>';
      const line = U.el('div', { class: 'row', style: 'gap:3px' });
      const cs = [];
      for (let i = 0; i < 8; i++) {
        const c = U.el('div', { style: 'width:30px;height:22px;border:1px solid var(--border);border-radius:3px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
        c.textContent = mod ? ('y[' + (i % mod) + ']') : ('y[' + i + ']');
        line.appendChild(c); cs.push(c);
      }
      box.appendChild(line);
      const note = U.el('div', { class: 'formula', style: 'margin:0;font-size:9.5px' });
      note.innerHTML = mod
        ? '<span class="m">i % ' + mod + '</span>：只有 ' + mod + ' 个不同的 y'
        : '<span class="m">i</span>：每个 z[i] 配一个自己的 y[i]';
      box.appendChild(note);
      box.style.borderLeft = '2px solid var(--' + color + ')';
      box.style.paddingLeft = '7px';
      lanes.appendChild(box);
      return cs;
    }
    const cA = lane('连续：ne10 == ne00', 'd', 0);
    const cB = lane('广播：ne10 == 4（src1 只有 4 个元素）', 'e', 4);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看左边的连续路径：<span class="v">vec_binary_op_contiguous</span>，9 行，和一元算子的循环几乎一样。',
      '广播路径多了一行 <span class="v">i10 = i % ne10</span>：<b>用取模把 src1 的行回绕</b>。这就是"广播"在 CPU 上的实现 —— 没有展开、没有复制，只是重新算指针。',
      '两条路径的选择在运行时的第 62 行：<span class="v">ggml_is_contiguous_rows(src1)</span>。行内连续就走连续循环，否则走回绕循环。',
      '广播的合法性不是这里检查的：第 54 行的断言要求 <span class="v">ggml_can_repeat(src1, src0)</span> —— 形状规则在 L1 的 ggml.c 里，内核只是<b>相信</b>这个前提。',
      '所以二元算子的内核只有两件事：<span class="k">定位行指针</span> + <span class="k">选择同步走还是回绕走</span>。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; cA.forEach(c => { c.style.borderColor = 'var(--d)'; }); });
    tl.at(4200, () => {
      cB.forEach((c, i) => { c.style.borderColor = 'var(--e)'; c.style.background = (i % 4 === 0) ? 'rgba(247,120,186,.16)' : '#10151b'; });
      msg.innerHTML = texts[1];
    });
    tl.at(9000, () => { msg.innerHTML = texts[2]; });
    tl.at(13000, () => { msg.innerHTML = texts[3]; });
    tl.at(16800, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 ★ 二元的 7 条类型链，和唯一的 <span class="hl-e">vDSP</span> 例外 */
{
  kicker: "L5-02 · 二元",
  title: "★ 二元的 7 条类型链，和唯一的 <span class=\"hl-e\">vDSP</span> 例外",
  sub: "binary_op<op_add> 有 7 条类型分支；Accelerate 构建里还会按 op 指针挑一个 vDSP 函数。",
  caption: "Accelerate 是 macOS 的系统框架，是否编译这段由编译期宏决定 —— 仍然是编译期开关，不是运行时探测。",
  src: "ggml/src/ggml-cpu/binary-ops.cpp",
  mark: [3, 4, 17, 22, 28, 36],
  lineNo: 0,
  code: `#ifdef GGML_USE_ACCELERATE
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
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '7 条类型组合', b: 'f32/f32/f32、f16 全同型、bf16 全同型，<br>以及 bf16/f32、f16/f32 的混合', m: 'apply_binary_op<op, s0, s1, d>' },
      { c: 'b', t: '模板实参 4 个', b: '1 个 op + 3 个类型<br>（src0、src1、dst）', m: 'binary_op<op_add>' },
      { c: 'e', t: 'vDSP 例外', b: 'macOS 上 f32 的加/减/乘/除<br>改调系统向量函数', m: 'vDSP_vadd / vsub / vmul / vdiv' },
      { c: 'c', t: '其余组合', b: 'GGML_ABORT —— 不支持的组合<br>是编译期可见的崩溃，不是静默降级', m: 'unsupported types' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '二元算子的类型组合比一元多：三个类型都得对上，于是 7 条分支。',
      '<span class="v">binary_op&lt;op_add&gt;</span> 一个模板，被 sub / mul / div 三个函数各实例化一遍（第 140-154 行）。',
      '<span class="v">op</span> 是函数指针，所以第 68 行可以直接用它做比较：<span class="k">if (op == op_add) vDSP_op = vDSP_vadd;</span> —— 编译器为每条实例各挑一次。',
      '注意这不是"运行时按 CPU 选 kernel"：<span class="v">GGML_USE_ACCELERATE</span> 是编译期宏，选出来的 vDSP 函数指针在每次调用时被复用。',
      '真正的运行时分派（按 CPU 特性位选 kernel）在 L5-04 的 repack / traits 与 L5-05 的多架构内核里 —— 那里才有了 <span class="v">ggml_cpu_has_*</span> 这类查询。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 三个特例：<span class="hl-c">ADD1</span>（标量）、<span class="hl-c">SCALE</span>（参数）、<span class="hl-c">WIN_PART</span>（窗口） */
{
  kicker: "L5-02 · 特例",
  title: "三个特例：<span class=\"hl-c\">ADD1</span>（标量）、<span class=\"hl-c\">SCALE</span>（参数）、<span class=\"hl-c\">WIN_PART</span>（窗口）",
  sub: "同样是逐元素，输入却可以来自标量张量、op_params、或一个滑动的窗口原点。",
  caption: "\"sliding window\" 这个词在 ggml-cpu 里只出现一次（ops.cpp:9734，SSM 卷积的缓存窗口）；逐元素家族里的\"窗口\"变体是 WIN_PART / WIN_UNPART 这类分窗算子。",
  src: "ggml/src/ggml-cpu/ops.cpp",
  mark: [8, 20, 24, 30],
  lineNo: 0,
  code: `static void ggml_compute_forward_add1_f32(
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
//>> ---- ggml/src/ggml-cpu/ops.cpp:4707-4711 ----
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
                        const int64_t i01 = px*w + i1;`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: 'GGML_OP_ADD1', b: 'src1 是<b>标量张量</b>（ggml_is_scalar）。<br>整行加同一个数，f32 走 ggml_vec_add1_f32。', m: 'ggml_compute_forward_add1_f32' },
      { c: 'd', t: 'GGML_OP_SCALE', b: 'scale 与 bias 从 <b>op_params</b> 里 memcpy 出来。<br>只有一个输入张量，却有两个标量参数。', m: 'ggml_compute_forward_scale_f32' },
      { c: 'e', t: 'GGML_OP_WIN_PART', b: '把一个张量切成 w x w 的窗口重排。<br>标了 TODO: optimize / multi-thread，<br>且 GGML_UNUSED(params) —— 单线程。', m: 'ggml_compute_forward_win_part_f32' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const m = msg;
    const texts = [
      '三个特例都还在"逐元素"这个族里，但每个都在某处破了一点常规。',
      '<span class="v">ADD1</span>：形状规则不是"同形状"，而是"src1 是标量"。<br>断言写在最前面：<span class="k">GGML_ASSERT(ggml_is_scalar(src1))</span>。',
      '<span class="v">SCALE</span>：<span class="k">没有第二个输入张量</span>，参数在 op_params 里。<br>所以它的循环里出现的是 s、b 两个 float，而不是 row 指针。',
      '<span class="v">WIN_PART</span>：循环里出现的是<b>窗口原点</b> <span class="v">py*w + i2</span> 与 <span class="v">px*w + i1</span>，越界就写 0。',
      '共同点还是那一句：<span class="k">一个输出元素只依赖有限的、位置可算的输入元素</span>。这就是逐元素族能被模板化、能被向量化的全部理由。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      m.innerHTML = texts[i];
    }));
    tl.at(15200, () => { els.forEach(e => e.style.opacity = '1'); m.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表 */
{
  kicker: "L5-02 · 收束",
  title: "把这一课压成一张表",
  sub: "三层分派 + 两条 SIMD 路线。验收点：\"unary op 的 SIMD 分发是怎么做的\"。",
  caption: "下一课 L5-03 进 vec.h：手写 SIMD 内核、ISA 映射、ggml_vec_dot_* 家族。",
  src: "ggml/src/ggml-cpu/ops.h",
  mark: [0, 1, 2, 3],
  lineNo: 96,
  code: `void ggml_compute_forward_win_part(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_unpart(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_unary(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['层', '做什么', '在哪'],
      [['1. op -> 函数', 'switch(ggml_get_unary_op(dst))，22 个 case', 'ops.cpp:10137'],
       ['2. 函数 -> 模板实例', 'unary_op<op_abs>，op 是函数指针模板参数', 'unary-ops.cpp:136'],
       ['3. 类型对 -> 特化', '5 条 if-else，落到 apply_unary_op<op, s0, d>', 'unary-ops.cpp:140'],
       ['4. 向量化（新族）', '平凡循环 + 编译器自动向量化（vandps ymm）', 'unary-ops.cpp:105'],
       ['4b. 向量化（老族）', 'GELU/SILU 调 ggml_vec_*，按编译期 ISA 宏分支', 'ops.cpp:2611 / vec.cpp:380']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '给定 <span class="mono">GGML_OP_UNARY + GGML_UNARY_OP_ABS</span>（f32）与 ' +
      '<span class="mono">GGML_UNARY_OP_SILU</span>（f32），从 CPU 后端的大 switch 一路到真正做运算的循环，' +
      '各经过哪几层？<b>SIMD 是从哪一层进来的？</b>',
      '<b>ABS</b>：ggml-cpu.c 的 GGML_OP_UNARY case -> ops.cpp:10144 的 switch -> ' +
      'unary-ops.cpp:237 <span class="mono">ggml_compute_forward_abs</span> -> ' +
      '<span class="mono">unary_op&lt;op_abs&gt;</span>（136 行）-> 140 行命中 f32/f32 -> ' +
      '<span class="mono">apply_unary_op&lt;op_abs, float, float&gt;</span> -> ' +
      '<span class="mono">vec_unary_op</span>（101 行）的平凡循环。<br>' +
      '<b>SIMD 在这一层由编译器生成</b>：源码里没有任何 SIMD 内建函数；实测 -O3 -mavx2 -mf16c 下 ' +
      '循环体是 <span class="mono">vandps ymm</span>，一次 8 个 float。<br><br>' +
      '<b>SILU</b> 不走 unary-ops.cpp：ops.cpp:10188 的 case -> ops.cpp:2674 ' +
      '<span class="mono">ggml_compute_forward_silu</span> -> 2579 的 <span class="mono">silu_f32</span> -> ' +
      '2611 调 <span class="mono">ggml_vec_silu_f32</span>。<br>' +
      '<b>SIMD 在这一层是手写的</b>：vec.cpp:380 里按编译期 ISA 宏（AVX512F / AVX2+FMA / SSE2 / SVE / NEON）' +
      '选一段内建函数循环。<br><br>' +
      '两者的共同点，也是本题的关键：<b>都没有运行时按 CPU 特性选 kernel</b>。' +
      '新族靠编译器，老族靠编译期宏。运行时的 kernel 分派要去 L5-04 的 traits / repack 找。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '一元的全链路，压成五行：',
      '<span class="k">第一层</span>在 ops.cpp：op 号 -> 函数名，22 个 case，不看类型。',
      '<span class="k">第二层</span>在 unary-ops.cpp：函数名 -> 模板实例，op 是编译期常量。',
      '<span class="k">第三层</span>还是 unary-ops.cpp：类型对 -> 模板特化，5 条分支。',
      '<span class="k">第四层</span>分岔：新族交给编译器自动向量化，老族（GELU/SILU）交给 vec.h 的手写内核 —— 这就是 L5-03 要展开的东西。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2900, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(16200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
