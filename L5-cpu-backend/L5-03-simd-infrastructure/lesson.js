/* ==========================================================================
   L5-03 · ★ 向量化基础设施：一份宏，十一个架构分支
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 向量化基础设施：<span class="hl-a">四块拼图</span> */
{
  kicker: "L5 · CPU 后端执行",
  title: "向量化基础设施：<span class=\"hl-a\">四块拼图</span>",
  sub: "一条点积的调用链：算子内核 -> ggml_vec_dot_* -> GGML_F32_VEC_* -> 某个 ISA 的 intrinsic。",
  caption: "回顾 L1-01：内核拿到的是 ne[]/nb[]；本课讲的是\"沿 ne[0] 连续的元素一次算几个\"。",
  src: "ggml/src/ggml-cpu/vec.h",
  mark: [0, 4, 5],
  lineNo: 1,
  code: `// Vectorized functions for fundamental operations

#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
//>> vec.h 只把架构差异托付给 simd-mappings.h 这一个头文件
#include "ggml.h"
#include "ggml-cpu.h"`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">算子内核</span><span class="arrow">-></span>
        <span class="chip a">ggml_vec_dot_*</span><span class="arrow">-></span>
        <span class="chip c">GGML_F32_VEC_*</span><span class="arrow">-></span>
        <span class="chip b">intrinsic</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'vec.h · 声明', b: '家族签名与调优常数；<br>架构差异全靠下面那个头。', m: 'ggml_vec_dot_f32(...)' },
      { c: 'b', t: 'vec.cpp · 实现', b: 'ISA 无关的循环 + 标量兜底；<br>同文件里还有 ISA 特化分支。', m: '#if defined(GGML_SIMD)' },
      { c: 'c', t: 'simd-mappings.h · 映射', b: '把一组宏映射到具体 intrinsic：<br>十一个架构分支。', m: 'GGML_F32_VEC_LOAD' },
      { c: 'd', t: 'simd-gemm.h · 微内核', b: '只用宏写成的 GEMM；<br>分块常数按 ISA 变。', m: 'simd_gemm_ukernel' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:160px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '四个文件合起来回答一个问题：<span class="k">同一份点积源码，怎么在 x86 / ARM / RISC-V / s390 / PowerPC / LoongArch / WASM 上都能编译出向量指令。</span>',
      '第一块 <span class="v">vec.h</span>：只声明家族（f32 / bf16 / f16）与调优常数，把架构相关的部分交给 <span class="m">simd-mappings.h</span>（第 6 行）。',
      '第二块 <span class="v">vec.cpp</span>：实现全部写在 <span class="m">#if defined(GGML_SIMD)</span> 之下 —— 有 SIMD 走宏，没有就走标量。',
      '第三块 <span class="v">simd-mappings.h</span>：本课的核心。它不实现算法，只把九个宏名翻译成当前 ISA 的 intrinsic。',
      '第四块 <span class="v">simd-gemm.h</span>：同一套宏还能写矩阵乘的微内核 —— 连分块大小都按 ISA 给。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(17700, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[4];
      U.markLines(document, []);
    });
  }
},

/* ------------------------------------------------------ 2 一套宏，<span class="hl-c">九个名字</span> */
{
  kicker: "L5-03 · 核心接口",
  title: "一套宏，<span class=\"hl-c\">九个名字</span>",
  sub: "源码注释把设计意图写死了：只定义一套宏，基本运算只用宏写，接新架构 = 加一组宏。",
  caption: "宏族一共九个名字：GGML_F32_VEC 与 _ZERO / _SET1 / _LOAD / _STORE / _FMA / _ADD / _MUL / _REDUCE —— 每个架构分支都要定义齐。",
  src: "ggml/src/ggml-cpu/simd-mappings.h",
  mark: [0, 1, 2, 8],
  lineNo: 161,
  code: `// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// GGML_F32_STEP / GGML_F16_STEP
//   number of elements to process in a single step
//
// GGML_F32_EPR / GGML_F16_EPR
//   number of elements to fit in a single register`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['上层只写这个宏', 'NEON 展开', 'AVX / AVX2 展开'],
      [['GGML_F32_VEC', 'float32x4_t', '__m256'],
       ['GGML_F32_VEC_LOAD', 'vld1q_f32', '_mm256_loadu_ps'],
       ['GGML_F32_VEC_FMA', 'vfmaq_f32', '_mm256_fmadd_ps'],
       ['GGML_F32_VEC_REDUCE', 'GGML_F32x4_REDUCE', 'GGML_F32x8_REDUCE']],
      { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '第一行注释就是全部设计：<span class="v">common set of C macros which map to specific intrinsics</span>。',
      '<span class="k">we then implement the fundamental computation operations below using only these macros</span> —— 只准用宏，不准写 intrinsic。',
      '<span class="k">adding support for new architectures requires to define the corresponding SIMD macros</span> —— 接新架构 = 加一组宏，不动算法。',
      '<span class="v">GGML_F32_EPR</span> = 一个寄存器装几个元素；<span class="v">GGML_F32_STEP</span> = 一步处理几个元素。上层只认这两个数。',
      '于是"支持 AVX2 / NEON / RVV / s390"这件事，被压缩成头文件里的十一个 <span class="m">#if / #elif</span> 分支。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [1]); });
    tl.at(6600, () => { msg.innerHTML = texts[2]; U.markLines(document, [2]); });
    tl.at(9600, () => { msg.innerHTML = texts[3]; U.markLines(document, [8]); });
    tl.at(12600, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[4]; U.markLines(document, [0, 1, 2, 8]); });
    rows.forEach((r, i) => tl.at(15600 + i * 600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
    }));
  }
},

/* ------------------------------------------------------ 3 ★ 同一组宏，<span class="hl-b">三种 ISA 的展开</span> */
{
  kicker: "L5-03 · ★ 核心洞察",
  title: "★ 同一组宏，<span class=\"hl-b\">三种 ISA 的展开</span>",
  sub: "把三个分支从同一个头文件里并排取出来：判据、寄存器宽度、每步元素数，全在宏里。",
  caption: "多段引用：三段分别来自 simd-mappings.h 的三个架构分支（行号见代码区 //>> 分隔行）。",
  src: "ggml/src/ggml-cpu/simd-mappings.h",
  mark: [0, 6, 7, 9, 12, 18, 19, 21, 23, 29, 30, 32],
  lineNo: 0,
  code: `#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#define GGML_SIMD

// F32 NEON

#define GGML_F32_STEP 16
#define GGML_F32_EPR  4

#define GGML_F32x4              float32x4_t
#define GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
//>> ---- ggml/src/ggml-cpu/simd-mappings.h:581-590 ----
#elif defined(__AVX__)

#define GGML_SIMD

// F32 AVX

#define GGML_F32_STEP 32
#define GGML_F32_EPR  8

#define GGML_F32x8         __m256
//>> ---- ggml/src/ggml-cpu/simd-mappings.h:446-456 ----
#elif defined(__AVX512F__)

#define GGML_SIMD

// F32 AVX512

#define GGML_F32_STEP 64
#define GGML_F32_EPR  16

#define GGML_F32x16         __m512
#define GGML_F32x16_ZERO    _mm512_setzero_ps()`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['ISA', '分支条件（simd-mappings.h）', 'EPR / STEP', '向量类型'],
      [['AVX-512', '#elif defined(__AVX512F__)', '16 / 64', '__m512'],
       ['AVX / AVX2', '#elif defined(__AVX__)', '8 / 32', '__m256'],
       ['NEON', '#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)', '4 / 16', 'float32x4_t'],
       ['标量', '一条都不命中 -> GGML_SIMD 未定义', '无宏', 'float']],
      { monoCols: [1, 2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '三个分支，三套定义。判据只有一个：<span class="k">编译器预先定义了哪个宏</span>。',
      '<span class="v">__AVX512F__</span>：一个寄存器 16 个 float，一步 64 个。',
      '<span class="v">__AVX__</span>：一个寄存器 8 个 float，一步 32 个。',
      '<span class="v">__ARM_NEON</span>：一个寄存器 4 个 float，一步 16 个。',
      '一条都不命中时 <span class="v">GGML_SIMD</span> 不会被定义 —— 上层源码自动退到标量分支。',
      '注意 <span class="k">STEP / EPR = 4</span>：这三条 SIMD 分支都是一步用 4 个向量寄存器。变的只是"每个寄存器装几个数"。'
    ];
    tl.at(700, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(13200, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[5]; });
    tl.at(15800, () => { rows.forEach((x, k) => { x.className = (k < 3) ? 'on' : ''; }); });
  }
},

/* ------------------------------------------------------ 4 <span class="hl-a">ggml_vec_dot_f32</span>：家族样板 */
{
  kicker: "L5-03 · 样板",
  title: "<span class=\"hl-a\">ggml_vec_dot_f32</span>：家族样板",
  sub: "34 行里包含了向量化点积的全部套路：切尾巴、开寄存器数组、FMA 累加、归约、标量补齐。",
  caption: "L5-02 会看到这个函数被 ops.cpp 里的算子内核直接调用；本幕只看它的形状。",
  src: "ggml/src/ggml-cpu/vec.cpp",
  mark: [0, 2, 7, 12, 17, 20, 24, 27],
  lineNo: 104,
  code: `        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #endif
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '① 切尾巴', b: 'np = n &amp; ~(STEP-1)：<br>能被一个步骤整除的部分。', m: 'const int np = ...' },
      { c: 'b', t: '② 开寄存器', b: 'GGML_F32_ARR 个累加器 + 两排输入，<br>个数由 STEP/EPR 推出。', m: 'GGML_F32_VEC sum[ARR]' },
      { c: 'c', t: '③ FMA 累加', b: '每个寄存器吃 EPR 个元素；<br>AVX 是 8 个 float。', m: 'GGML_F32_VEC_FMA(...)' },
      { c: 'd', t: '④ 归约 + 兜底', b: '向量收成标量；<br>没 SIMD 时整块走标量。', m: 'GGML_F32_VEC_REDUCE' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:160px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看四个动作，再看它们在代码里的位置。',
      '<span class="v">np = (n &amp; ~(GGML_F32_STEP - 1))</span>：主循环里因此<b>没有任何 if</b>。',
      '<span class="v">GGML_F32_ARR</span> 个累加器 = STEP/EPR：AVX2 是 4 个 __m256，NEON 是 4 个 float32x4_t。',
      '<span class="v">GGML_F32_VEC_FMA</span>：AVX2 展开成 _mm256_fmadd_ps，NEON 展开成 vfmaq_f32 —— 同一行源码。',
      '<span class="v">GGML_F32_VEC_REDUCE</span> 把 4 个向量收成 1 个标量；<span class="v">np..n</span> 的尾巴用标量补齐。',
      '最后 7 行（<span class="m">#else // scalar</span>）是完全没有 SIMD 的路径：<span class="k">同一个函数、同一个结果</span>。它是正确性基准，不是废物。'
    ];
    const hl = xs => U.markLines(document, xs);
    const plan = [[0, 2], [0, 2, 7], [7, 12], [17, 20]];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
      hl(plan[i]);
    }));
    tl.at(14200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[4];
      hl([17, 20, 24, 27]);
    });
    tl.at(17400, () => { msg.innerHTML = texts[5]; hl([24, 27]); });
    tl.at(20000, () => { msg.innerHTML = texts[5]; hl([0, 2, 7, 12, 17, 20, 24, 27]); });
  }
},

/* ------------------------------------------------------ 5 家族：<span class="hl-d">三个浮点点积</span> + 一组常数 */
{
  kicker: "L5-03 · 家族",
  title: "家族：<span class=\"hl-d\">三个浮点点积</span> + 一组常数",
  sub: "签名统一是 (n, s, bs, x, bx, y, by, nrc)；nrc 让一次调用算多行。",
  caption: "回顾 L1-04：量化类型把 32 个权重打包成一块；第 9 幕会看到块怎么进入点积。",
  src: "ggml/src/ggml-cpu/vec.h",
  mark: [1, 7, 10, 11, 12, 14, 16, 17],
  lineNo: 0,
  code: `// floating point type used to accumulate sums
typedef double ggml_float;
//>> ---- ggml/src/ggml-cpu/vec.h:46-51 ----
#define GGML_GELU_FP16
#define GGML_GELU_QUICK_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2
#define GGML_VEC_MAD_UNROLL  32
//>> ---- ggml/src/ggml-cpu/vec.h:71-73 ----
void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);
//>> ---- ggml/src/ggml-cpu/vec.h:140-151 ----
// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float * GGML_RESTRICT s, void * GGML_RESTRICT xv, ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(GGML_SIMD)`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['函数', '输入元素类型', '一次算几行', '本课哪里看'],
      [['ggml_vec_dot_f32', 'float', 'nrc', '第 4 幕'],
       ['ggml_vec_dot_bf16', 'ggml_bf16_t', 'nrc', '第 6 幕'],
       ['ggml_vec_dot_f16', 'ggml_fp16_t', 'nrc', 'vec.cpp'],
       ['ggml_vec_dot_f16_unroll', 'ggml_fp16_t', 'GGML_VEC_DOT_UNROLL', '本幕'],
       ['ggml_vec_dot_q4_0_q8_0', '量化块（见第 9 幕）', 'nrc', '第 9 幕 · L5-05']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '家族成员的签名完全一致，只有 x / y 的元素类型不同 —— 这是 traits 表能用一个函数指针类型装下它们的前提。',
      '<span class="v">nrc</span> = number of rows：一次调用同时算几行点积。量化点积在 ARM 上会用到 <span class="m">nrc == 2</span>。',
      '<span class="v">ggml_vec_dot_f16_unroll</span> 是家族里的多行变体：一次算 <span class="m">GGML_VEC_DOT_UNROLL</span> 行，行的跨度是 <span class="m">xs</span> 字节。',
      '累加类型 <span class="m">ggml_float</span> = <span class="v">double</span>（vec.h:15）；常数：<span class="m">SOFT_MAX_UNROLL</span> = <span class="v">4</span>　<span class="m">VEC_DOT_UNROLL</span> = <span class="v">2</span>　<span class="m">VEC_MAD_UNROLL</span> = <span class="v">32</span>（vec.h:46-51）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[1];
    }));
    tl.at(15800, () => { rows.forEach((x, k) => { x.className = (k === 3) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
    tl.at(18000, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 6 同一个文件里的 <span class="hl-e">ISA 特化分支</span> */
{
  kicker: "L5-03 · 另一种写法",
  title: "同一个文件里的 <span class=\"hl-e\">ISA 特化分支</span>",
  sub: "ggml_vec_dot_bf16 走另一条路：不用宏，直接按 #if 给每个架构写死 intrinsic。",
  caption: "链尾没有 #else：257 行的循环是共用收尾；一个分支都没命中时，它就是全部计算。中间还有 RISC-V 的 __riscv_zvfbfwma 分支（vec.cpp:198），本幕未引用。",
  src: "ggml/src/ggml-cpu/vec.cpp",
  mark: [0, 4, 12, 13, 24, 31, 36, 40, 45, 49, 50],
  lineNo: 0,
  code: `#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
//>> ---- ggml/src/ggml-cpu/vec.cpp:239-261 ----
#elif defined(__POWER9_VECTOR__) || defined(__VXE__) || defined(__VXE2__)
    const int np = (n & ~(GGML_BF16_STEP - 1));
    if (np > 0) {
        GGML_F32_VEC sum[4] = {GGML_F32_VEC_ZERO};
        for (; i < np; i += GGML_BF16_STEP) {
            GGML_BF16_VEC vx0 = GGML_BF16_VEC_LOAD(x + i);
            GGML_BF16_VEC vx1 = GGML_BF16_VEC_LOAD(x + i + 8);
            GGML_BF16_VEC vy0 = GGML_BF16_VEC_LOAD(y + i);
            GGML_BF16_VEC vy1 = GGML_BF16_VEC_LOAD(y + i + 8);
            GGML_BF16_FMA_LO(sum[0], vx0, vy0);
            GGML_BF16_FMA_HI(sum[1], vx0, vy0);
            GGML_BF16_FMA_LO(sum[2], vx1, vy1);
            GGML_BF16_FMA_HI(sum[3], vx1, vy1);
        }
        GGML_F32x4_REDUCE_4(sumf, sum[0], sum[1], sum[2], sum[3]);
    }
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['分支', '条件', '用什么算', '行号'],
      [['AVX-512 BF16', '#if defined(__AVX512BF16__)', '_mm512_dpbf16_ps', ':148'],
       ['AVX-512 通用', '#elif defined(__AVX512F__)', '_mm512_* + LOAD 宏', ':160'],
       ['AVX2 / AVX', '#elif defined(__AVX2__) || defined(__AVX__)', '_mm256_* / _mm_*', ':172'],
       ['PowerPC / s390', '#elif defined(__POWER9_VECTOR__) || defined(__VXE__) || defined(__VXE2__)', 'GGML_BF16_VEC_* 宏', ':239'],
       ['收尾（共用）', '链尾无 #else', 'GGML_BF16_TO_FP32 标量', ':257']],
      { monoCols: [1, 2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '同一个函数、五种算法。前三条是 x86，第四条同时覆盖 PowerPC 与 s390（<span class="m">__VXE2__</span>）。',
      '<span class="v">__AVX512BF16__</span>：硬件一条指令算 32 个 bf16 的乘加（<span class="m">_mm512_dpbf16_ps</span>）。',
      '<span class="v">__AVX512F__</span> 与 <span class="v">__AVX2__</span>：没有硬件 bf16 点积，就把 bf16 左移 16 位当成 float 再乘。',
      '<span class="v">__POWER9_VECTOR__</span> / <span class="v">__VXE2__</span>：回到宏族 —— <span class="m">GGML_BF16_VEC_LOAD</span>、<span class="m">GGML_BF16_FMA_LO/HI</span>。',
      '注意最后一行：<span class="k">它不是 else 分支</span>，而是所有分支共用的收尾循环（处理 n 除不尽的部分），也是"一条都不命中"时的全部实现。'
    ];
    tl.at(700, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2700, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(14800, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-a">simd-gemm.h</span>：循环体不变，分块按 ISA 变 */
{
  kicker: "L5-03 · GEMM 微内核",
  title: "<span class=\"hl-a\">simd-gemm.h</span>：循环体不变，分块按 ISA 变",
  sub: "50 行里没有一个 intrinsic：微内核只用 GGML_F32_VEC_*，但行/列分块常数由架构决定。",
  caption: "不满足开头那个条件（SVE 或 RISC-V RVV，或没有 SIMD）时，整个文件退化成 207-226 行的三重叠循环 —— 见下一幕。",
  src: "ggml/src/ggml-cpu/simd-gemm.h",
  mark: [0, 4, 6, 7, 8, 10, 11, 24, 27, 30, 37, 42, 49],
  lineNo: 8,
  code: `#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON__)
//>> AVX-512 与 NEON 共用这一档：4 个累加器行 x 4 个累加器列
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
//>> AVX2 / AVX 换一档：6 行 x 2 列 —— 注释给出了寄存器预算的算法
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;
//>> KN = GGML_F32_EPR：一行里一次能处理几个元素，由宏给出

    GGML_F32_VEC acc[RM][RN];
    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        for (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '不变的部分', b: '加载累加器 -> 遍历 K -> FMA -> 存回：<br>全部只用 GGML_F32_VEC_* 宏。', m: 'GGML_F32_VEC_FMA' },
      { c: 'b', t: '变了的部分（x86）', b: 'AVX2 / AVX：6 x 2。<br>源码注释：12+2+1 = 15/16。', m: 'GEMM_RM = 6, GEMM_RN = 2' },
      { c: 'c', t: '变了的部分（ARM）', b: 'NEON：4 x 4（与 AVX-512 同档）。<br>源码注释：16+4+1 = 25/32。', m: 'GEMM_RM = 4, GEMM_RN = 4' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '矩阵乘 C[M x N] += A[M x K] * B[K x N] 的微内核：把 M 和 N 各切一小块，块内用向量寄存器堆。',
      '<span class="v">GEMM_RM / GEMM_RN</span> 是"用几个寄存器行 x 几个寄存器列"的累加器阵列（单位是 GGML_F32_EPR）。',
      '<span class="v">KN = GGML_F32_EPR</span>：一行里一次处理几个 float —— 这个数在 AVX2 上是 8，在 NEON 上是 4。',
      '最内层只有一行计算：<span class="m">acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p)</span>。展开成什么指令，仍然由宏决定。',
      '所以 <span class="k">同一份微内核能同时服务 x86 与 ARM</span>：循环结构共享，只有分块常数不同档。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3500, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(18500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 <span class="hl-f">simd_gemm</span>：宏模板 / RVV / 标量 */
{
  kicker: "L5-03 · 同一名字，三份定义",
  title: "<span class=\"hl-f\">simd_gemm</span>：宏模板 / RVV / 标量",
  sub: "整个头文件被 #if 切成三段，每段各定义一次同名函数 —— 编译期三选一。",
  caption: "RVV 是唯一用 sizeless 类型的分支：vfloat32m4_t 的宽度运行时才由 __riscv_vlenb() 给出。",
  src: "ggml/src/ggml-cpu/simd-gemm.h",
  mark: [0, 3, 10, 12, 14, 17, 23, 26, 28, 34, 38],
  lineNo: 0,
  code: `#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <int RM>
static inline void rvv_simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
//>> ---- ggml/src/ggml-cpu/simd-gemm.h:176-186 ----
static constexpr int GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    const int KN = (int)__riscv_vlenb();
    int64_t ii = 0;
//>> ---- ggml/src/ggml-cpu/simd-gemm.h:207-226 ----
#else // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '分支 1 · 宏模板版', b: '用 GGML_F32_VEC_* 与<br>GEMM_RM/RN —— 见上一幕。', m: 'simd_gemm_ukernel<RM,RN>' },
      { c: 'c', t: '分支 2 · RISC-V RVV', b: 'sizeless 的 vfloat32m4_t；<br>宽度由 __riscv_vlenb() 给出。', m: 'GEMM_RM = 7  (:176)' },
      { c: 'd', t: '分支 3 · 标量版', b: '三重叠循环，逐元素乘加；<br>任何其他情况都落在这里。', m: '#else // scalar path  (:207)' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '三种实现不是三层：它们是 <span class="k">平行的三个分支</span>，编译期只留一个。',
      '分支 2（RVV）用 <span class="m">vfloat32m4_t</span> 做累加器，<span class="m">static_assert(RM &gt;= 1 &amp;&amp; RM &lt;= 7)</span> 把寄存器预算写进编译期断言。',
      'RVV 的分块常数是 <span class="v">GEMM_RM = 7</span>，且 N 方向不切块：<span class="m">KN = __riscv_vlenb()</span>，一次吃一整条向量寄存器。',
      '分支 3 是纯标量：<span class="m">sum += A[i * K + kk] * B[kk * N + j]</span>。它同时是"没有 SIMD"和"SVE"两条路的实现。',
      '一句话：<span class="k">宏模板负责大多数架构，RVV 单独写一份，剩下的都归标量。</span>'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => { msg.innerHTML = texts[3]; });
    tl.at(17000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 ★ <span class="hl-c">q4_0 x q8_0</span>：AVX2 与 NEON 差在哪 */
{
  kicker: "L5-03 · ★ 验收",
  title: "★ <span class=\"hl-c\">q4_0 x q8_0</span>：AVX2 与 NEON 差在哪",
  sub: "同一个符号名，两份实现；它们共享的正是本课这四个文件里的东西。",
  caption: "函数体在 ggml/src/ggml-cpu/arch/x86/quants.c:701 与 .../arch/arm/quants.c:297 —— 本课不引用这两个文件（属 L5-05），此处只作指路。",
  src: "ggml/src/ggml-cpu/simd-mappings.h",
  mark: [0, 2, 5, 8, 9, 10, 11, 20, 26],
  lineNo: 40,
  code: `#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
//>> ARM：把 fp16 装进 __fp16 再强转成 float，交给编译器选指令
    #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) neon_compute_fp16_to_fp32(x)
    #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) neon_compute_fp32_to_fp16(x)

    #define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
//>> 上层永远只写 GGML_CPU_FP16_TO_FP32(x) —— 怎么转由这一组宏决定

    static inline float neon_compute_fp16_to_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        memcpy(&tmp, &h, sizeof(ggml_fp16_t));
        return (float)tmp;
    }

    static inline ggml_fp16_t neon_compute_fp32_to_fp16(float f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        memcpy(&res, &tmp, sizeof(ggml_fp16_t));
        return res;
    }
#elif defined(__F16C__)
//>> x86：有 F16C 时走硬件转换指令（MSVC 与 GCC 写法不同）
    #ifdef _MSC_VER
        #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
    #else
        #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
        #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
    #endif`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['维度', 'AVX2 / x86', 'NEON / ARM'],
      [['函数体（指路）', 'arch/x86/quants.c:701', 'arch/arm/quants.c:297'],
       ['GGML_F32_EPR / STEP', '8 / 32', '4 / 16'],
       ['向量类型', '__m256 / __m256i', 'float32x4_t / int8x16_t'],
       ['FMA', '_mm256_fmadd_ps', 'vfmaq_f32'],
       ['块 scale：fp16 -> fp32', '_cvtsh_ss（__F16C__）', '(float)(__fp16)'],
       ['宏分支行号', 'simd-mappings.h:581', 'simd-mappings.h:331']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      'ggml_vec_dot_q4_0_q8_0 在 AVX2 机器与 NEON 机器上编译出来的实现，是同一份源码吗？差异体现在哪几处？',
      '<b>不是同一份源码。</b><br>' +
      '① 函数体两处：x86 在 <span class="mono">arch/x86/quants.c:701</span>，ARM 在 <span class="mono">arch/arm/quants.c:297</span>；' +
      '同名同签名，由构建系统按架构二选一（本课不引用这两个文件，见 L5-05）。<br>' +
      '② 向量宽度：AVX2 走 <span class="mono">simd-mappings.h:581</span> 的 <span class="mono">__AVX__</span> 分支' +
      '（EPR 8 / STEP 32 / <span class="mono">__m256</span>）；NEON 走 <span class="mono">:331</span>' +
      '（EPR 4 / STEP 16 / <span class="mono">float32x4_t</span>）。<br>' +
      '③ 块里的 fp16 scale：x86 用 <span class="mono">__F16C__</span> 的 <span class="mono">_cvtsh_ss</span>（<span class="mono">:63</span>），' +
      'ARM 用 <span class="mono">__fp16</span> 强转（<span class="mono">:46-49</span>）。<br>' +
      '共同点：两边都只写 <span class="mono">GGML_F32_VEC_*</span> 与 <span class="mono">GGML_CPU_FP16_TO_FP32</span> 这些抽象名' +
      '（设计宣言在 <span class="mono">:161-163</span>）。'));

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'Q4_0 块 = 1 个 fp16 scale + 16 字节 4-bit 权重（L1-04）。点积第一步永远是：<span class="k">把 scale 转成 float</span>。',
      '本幕引用的代码只回答这一件事：<span class="v">GGML_CPU_FP16_TO_FP32</span> 在 ARM 与 x86 上不是同一段实现。',
      'ARM：<span class="m">__fp16</span> + memcpy（:46-49）；x86：<span class="m">_cvtsh_ss</span>（:63）。上层写法完全一样。',
      '剩下的差异在宏分支那一层：<span class="v">EPR / STEP = 8/32 vs 4/16</span>，向量类型 __m256 vs float32x4_t。',
      '而函数体本身是两份源码（arch/x86 与 arch/arm），构建期二选一 —— 那是 <span class="k">L5-05</span> 的内容。'
    ];
    tl.at(700, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[0]; });
    tl.at(3300, () => { rows[4].className = 'on'; msg.innerHTML = texts[1]; });
    tl.at(6200, () => { rows[4].className = 'on'; msg.innerHTML = texts[2]; });
    tl.at(9100, () => { rows[4].className = ''; rows[1].className = 'on'; rows[2].className = 'on'; msg.innerHTML = texts[3]; });
    tl.at(12300, () => { rows.forEach(r => { r.className = ''; }); rows[0].className = 'on'; msg.innerHTML = texts[4]; });
    tl.at(16000, () => { rows[0].className = 'on'; rows[5].className = 'on'; msg.innerHTML = texts[4]; });
  }
},

];
