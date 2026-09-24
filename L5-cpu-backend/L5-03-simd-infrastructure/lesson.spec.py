#!/usr/bin/env python3
"""L5-03 · 向量化基础设施 —— 课件 spec。

运行：python3 L5-cpu-backend/L5-03-simd-infrastructure/lesson.spec.py

所有代码引用按【上游行号】从 llama.cpp 抽取，逐字保真由构造保证。
本课只引用 4 个文件（见 L.footnote_add），其余文件只作"指路"。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_VEC_H = 'ggml/src/ggml-cpu/vec.h'
SRC_VEC_CPP = 'ggml/src/ggml-cpu/vec.cpp'
SRC_MAP = 'ggml/src/ggml-cpu/simd-mappings.h'
SRC_GEMM = 'ggml/src/ggml-cpu/simd-gemm.h'

L = Lesson(
    id='L5-03',
    layer='L5 · CPU 后端执行',
    title='★ 向量化基础设施：一份宏，十一个架构分支',
    codecap='ggml/src/ggml-cpu/{vec.h, vec.cpp, simd-mappings.h, simd-gemm.h}（逐字引用）',
    nav={'prev': {'href': '../L5-02-unary-binary-ops/index.html',
                  'label': 'L5-02 ★ 一元与二元算子内核'},
         'next': {'href': '../L5-04-quants-and-repack/index.html',
                  'label': 'L5-04 ★ 量化与 repack'}},
)

L.note('**一句话**：`ggml_vec_dot_*` 家族不可能为每个 CPU 架构写一遍。llama.cpp 的做法是'
       '**先定义一组宏，再用这组宏写运算** —— 同一个 `.cpp` 源码，在 x86 上被展开成 AVX 的'
       '`_mm256_*`，在 ARM 上被展开成 NEON 的 `v*_q*`，在 RISC-V 上被展开成 RVV 内建函数。'
       '这一课只讲让这件事成立的四个文件。')
L.note('四个文件的分工：`simd-mappings.h` 定义宏（十一个架构分支）、`vec.h` 声明家族与调优常数、'
       '`vec.cpp` 用宏写 ISA 无关实现、`simd-gemm.h` 用宏写出 GEMM 微内核。')
L.note('读这一课之前请先确认一件事：内核看到的永远是 `ne[]` / `nb[]`（L1-01）与量化块布局'
       '（L1-04）；这一课回答的是"拿到这些之后，一次算几个元素、用哪个寄存器"。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L5 · CPU 后端执行',
    title='向量化基础设施：<span class="hl-a">四块拼图</span>',
    sub='一条点积的调用链：算子内核 -> ggml_vec_dot_* -> GGML_F32_VEC_* -> 某个 ISA 的 intrinsic。',
    caption='回顾 L1-01：内核拿到的是 ne[]/nb[]；本课讲的是"沿 ne[0] 连续的元素一次算几个"。',
    src=SRC_VEC_H, parts=[(1, 8)], duration=20000,
    mark_src=[1, 5, 6],
    notes_src={6: 'vec.h 只把架构差异托付给 simd-mappings.h 这一个头文件'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L5-03 · 核心接口',
    title='一套宏，<span class="hl-c">九个名字</span>',
    sub='源码注释把设计意图写死了：只定义一套宏，基本运算只用宏写，接新架构 = 加一组宏。',
    caption='宏族一共九个名字：GGML_F32_VEC 与 _ZERO / _SET1 / _LOAD / _STORE / _FMA / _ADD / _MUL / _REDUCE —— 每个架构分支都要定义齐。',
    src=SRC_MAP, parts=[(161, 169)], duration=18000,
    mark_src=[161, 162, 163, 169],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L5-03 · ★ 核心洞察',
    title='★ 同一组宏，<span class="hl-b">三种 ISA 的展开</span>',
    sub='把三个分支从同一个头文件里并排取出来：判据、寄存器宽度、每步元素数，全在宏里。',
    caption='多段引用：三段分别来自 simd-mappings.h 的三个架构分支（行号见代码区 //>> 分隔行）。',
    src=SRC_MAP, parts=[(331, 341), (581, 590), (446, 456)], duration=19000,
    mark_src=[331, 337, 338, 340, 581, 587, 588, 590, 446, 452, 453, 455],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L5-03 · 样板',
    title='<span class="hl-a">ggml_vec_dot_f32</span>：家族样板',
    sub='34 行里包含了向量化点积的全部套路：切尾巴、开寄存器数组、FMA 累加、归约、标量补齐。',
    caption='L5-02 会看到这个函数被 ops.cpp 里的算子内核直接调用；本幕只看它的形状。',
    src=SRC_VEC_CPP, parts=[(104, 137)], duration=22000,
    mark_src=[104, 106, 111, 116, 121, 124, 128, 131],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L5-03 · 家族',
    title='家族：<span class="hl-d">三个浮点点积</span> + 一组常数',
    sub='签名统一是 (n, s, bs, x, bx, y, by, nrc)；nrc 让一次调用算多行。',
    caption='回顾 L1-04：量化类型把 32 个权重打包成一块；第 9 幕会看到块怎么进入点积。',
    src=SRC_VEC_H, parts=[(14, 15), (46, 51), (71, 73), (140, 151)], duration=20000,
    mark_src=[15, 50, 71, 72, 73, 140, 142, 143],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L5-03 · 另一种写法',
    title='同一个文件里的 <span class="hl-e">ISA 特化分支</span>',
    sub='ggml_vec_dot_bf16 走另一条路：不用宏，直接按 #if 给每个架构写死 intrinsic。',
    caption='链尾没有 #else：257 行的循环是共用收尾；一个分支都没命中时，它就是全部计算。中间还有 RISC-V 的 __riscv_zvfbfwma 分支（vec.cpp:198），本幕未引用。',
    src=SRC_VEC_CPP, parts=[(148, 177), (239, 261)], duration=22000,
    mark_src=[148, 152, 160, 161, 172, 239, 244, 248, 253, 257, 258],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L5-03 · GEMM 微内核',
    title='<span class="hl-a">simd-gemm.h</span>：循环体不变，分块按 ISA 变',
    sub='50 行里没有一个 intrinsic：微内核只用 GGML_F32_VEC_*，但行/列分块常数由架构决定。',
    caption='不满足开头那个条件（SVE 或 RISC-V RVV，或没有 SIMD）时，整个文件退化成 207-226 行的三重叠循环 —— 见下一幕。',
    src=SRC_GEMM, parts=[(8, 57)], duration=24000,
    mark_src=[8, 12, 13, 14, 15, 16, 17, 30, 32, 35, 42, 47, 54],
    notes_src={12: 'AVX-512 与 NEON 共用这一档：4 个累加器行 x 4 个累加器列',
               15: 'AVX2 / AVX 换一档：6 行 x 2 列 —— 注释给出了寄存器预算的算法',
               30: 'KN = GGML_F32_EPR：一行里一次能处理几个元素，由宏给出'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L5-03 · 同一名字，三份定义',
    title='<span class="hl-f">simd_gemm</span>：宏模板 / RVV / 标量',
    sub='整个头文件被 #if 切成三段，每段各定义一次同名函数 —— 编译期三选一。',
    caption='RVV 是唯一用 sizeless 类型的分支：vfloat32m4_t 的宽度运行时才由 __riscv_vlenb() 给出。',
    src=SRC_GEMM, parts=[(112, 124), (176, 186), (207, 226)], duration=20000,
    mark_src=[112, 115, 122, 124, 176, 179, 185, 207, 209, 215, 219],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L5-03 · ★ 验收',
    title='★ <span class="hl-c">q4_0 x q8_0</span>：AVX2 与 NEON 差在哪',
    sub='同一个符号名，两份实现；它们共享的正是本课这四个文件里的东西。',
    caption='函数体在 ggml/src/ggml-cpu/arch/x86/quants.c:701 与 .../arch/arm/quants.c:297 —— 本课不引用这两个文件（属 L5-05），此处只作指路。',
    src=SRC_MAP, parts=[(40, 65)], duration=26000,
    mark_src=[40, 41, 44, 46, 47, 48, 49, 58, 63],
    notes_src={40: 'ARM：把 fp16 装进 __fp16 再强转成 float，交给编译器选指令',
               44: '上层永远只写 GGML_CPU_FP16_TO_FP32(x) —— 怎么转由这一组宏决定',
               58: 'x86：有 F16C 时走硬件转换指令（MSVC 与 GCC 写法不同）'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、四个文件怎么分工',
    '`vec.h` 是本课的入口：它只声明家族、给出调优常数，并把**所有架构差异**托付给 '
    '`simd-mappings.h`。注意它自己的 include 列表里没有任何 `immintrin.h` / `arm_neon.h` —— '
    '那些头由 `simd-mappings.h` 按架构吸进来。',
    src=SRC_VEC_H, parts=[(1, 8)], lang='c')

L.section(
    '二、★ 抽象层宣言：一套宏，十一个架构分支',
    '这是本课最重要的一段注释。它把整个设计压缩成三条：\n\n'
    '```text\n'
    '1. 定义一组 C 宏，按当前架构映射到具体 intrinsic\n'
    '2. 下面的基本运算只用这些宏实现\n'
    '3. 支持新架构 = 定义对应的 SIMD 宏（不动算法）\n'
    '```\n\n'
    '宏族一共九个名字（`GGML_F32_VEC` 与 `_ZERO` / `_SET1` / `_LOAD` / `_STORE` / `_FMA` / '
    '`_ADD` / `_MUL` / `_REDUCE`），每个架构分支都必须定义齐。'
    '`GGML_F32_EPR` 是"一个寄存器装几个元素"，`GGML_F32_STEP` 是"一步处理几个元素"。',
    src=SRC_MAP, parts=[(161, 169)], lang='c')

L.section(
    '三、★ 三种 ISA 的展开',
    '同一组宏名，在三个分支里被定义成三套不同的东西。判据只有一个：编译器预先定义了哪个宏。'
    'AVX-512 是 16 个 float / 寄存器、一步 64 个；AVX/AVX2 是 8 / 32；NEON 是 4 / 16。'
    '三条分支的 `STEP / EPR` 都等于 4 —— **一步固定开 4 个向量寄存器**，变的只是每个寄存器装多少。'
    '（这三段的真实行号见代码区的 `//>>` 分隔行。）',
    src=SRC_MAP, parts=[(331, 341), (581, 590), (446, 456)], lang='c')

L.section(
    '四、家族样板：ggml_vec_dot_f32',
    '只有 34 行，却是所有向量化点积的模板：先切掉尾巴得到 `np`，再让主循环按 `GGML_F32_STEP` '
    '前进、内层开 `GGML_F32_ARR` 个寄存器做 FMA，最后归约并补齐尾巴。'
    '文件末尾的 `#else // scalar` 是完全没有 SIMD 时的完整实现 —— 也是正确性基准。',
    src=SRC_VEC_CPP, parts=[(104, 137)], lang='cpp')

L.section(
    '五、家族与调优常数',
    '三个浮点点积的签名完全一致，只有 x / y 的元素类型不同：`(int n, float * s, size_t bs, '
    'const T * x, size_t bx, const T * y, size_t by, int nrc)`。'
    '`nrc` 是"一次算几行"；`ggml_vec_dot_f16_unroll` 把它固定成 `GGML_VEC_DOT_UNROLL`。'
    '累加类型 `ggml_float` 是 `double`。',
    src=SRC_VEC_H, parts=[(14, 15), (46, 51), (71, 73), (140, 151)], lang='c')

L.section(
    '六、另一种写法：ISA 特化分支',
    '`ggml_vec_dot_bf16` 不用宏，而是按 `#if` 给每个架构写死 intrinsic：AVX-512 BF16 用一条硬件'
    '`_mm512_dpbf16_ps`，AVX-512/AVX2 用移位把 bf16 当 float，PowerPC 与 s390 回到宏族。'
    '注意链尾没有 `#else`：最后的循环是所有分支共用的收尾，也是"一条都不命中"时的全部实现。',
    src=SRC_VEC_CPP, parts=[(148, 177), (239, 261)], lang='cpp')

L.section(
    '七、GEMM 微内核：分块常数按 ISA 变',
    '`simd-gemm.h` 的微内核里没有一个 intrinsic：加载累加器、遍历 K、FMA、存回，全部只用 '
    '`GGML_F32_VEC_*`。变得只有分块常数 —— AVX-512 与 NEON 是 4 x 4，AVX2 / AVX 是 6 x 2，'
    '其余 2 x 2；源码注释把寄存器预算写成了算式。',
    src=SRC_GEMM, parts=[(8, 57)], lang='cpp')

L.section(
    '八、同一个 simd_gemm，三份定义',
    '整个头文件被 `#if` 切成三段，每段各定义一次同名函数：宏模板版、RISC-V RVV 版、标量版。'
    'RVV 是唯一使用 sizeless 类型的分支（`vfloat32m4_t`，宽度运行时由 `__riscv_vlenb()` 给出）；'
    '标量版同时兜住"SVE"与"无 SIMD"两种情况。',
    src=SRC_GEMM, parts=[(112, 124), (176, 186), (207, 226)], lang='cpp')

L.section(
    '九、★ q4_0 x q8_0：AVX2 与 NEON 的差异',
    'Q4_0 的每个块带一个 fp16 的 scale（L1-04），所以任何 ISA 上的 `ggml_vec_dot_q4_0_q8_0` '
    '第一步都是"把 fp16 转成 fp32"。这一段引用展示的就是这一步的两套实现：ARM 走 `__fp16` 强转，'
    'x86 走 `__F16C__` 的硬件转换指令。**上层只写 `GGML_CPU_FP16_TO_FP32(x)`** —— '
    '这正是"一份源码、多种 ISA"的意思。',
    src=SRC_MAP, parts=[(40, 65)], lang='c')

L.footnote_add('本课引用的 4 个文件 —— `ggml/src/ggml-cpu/` 下的 `vec.h`、`vec.cpp`、'
               '`simd-mappings.h`、`simd-gemm.h` —— 全部计入覆盖率。')
L.footnote_add('为保证"每一条断言都能在引用的代码里看到"，本课**不引用** '
               '`arch/x86/quants.c`、`arch/arm/quants.c`、`quants.c`、`quants.h`、`arch-fallback.h`、'
               '`CMakeLists.txt`、`ggml-cpu.c`、`ops.cpp`、`ggml-cpu-impl.h` 等文件。'
               '正文里对这些文件只有"指路"（附真实行号），**不计入本课覆盖率**；'
               '它们分别属于 L5-01 / L5-02 / L5-04 / L5-05 的覆盖范围。')

L.prereqs('`L5-02`（一元与二元算子内核）')

L.goal(
    '说出 `ggml_vec_dot_*` 家族有哪些成员、签名里的 `nrc` 是什么意思（对应验收点）；',
    '解释 `simd-mappings.h` 的设计："一套宏 + 十一个架构分支"，以及 `GGML_F32_EPR` / '
    '`GGML_F32_STEP` / `GGML_F32_ARR` 三者的关系（对应验收点）；',
    '区分两种复用机制：**一份源码 + 宏映射**（vec.cpp / vec.h / simd-gemm.h）与 '
    '**同名函数 + 每架构一份实现**（quantized dot 家族）；',
    '说出 `ggml_vec_dot_q4_0_q8_0` 在 AVX2 与 NEON 上的实现差异（验收点）。')

L.conclusion(
    '★ 一份宏，十一个架构分支',
    '`simd-mappings.h` 只做一件事：把九个宏名（`GGML_F32_VEC` 与 `_ZERO/_SET1/_LOAD/_STORE/'
    '_FMA/_ADD/_MUL/_REDUCE`）按当前架构映射到具体 intrinsic。'
    '分支从 `__ARM_FEATURE_SVE`、`__ARM_NEON`、`__AVX512F__`、`__AVX__`、`__POWER9_VECTOR__`、'
    '`__wasm_simd128__`、`__SSE3__`、`__loongarch_asx/sx`、`__VXE__/__VXE2__` 一直到 '
    '`__riscv_v_intrinsic`，共十一个。\n\n'
    '| 分支 | EPR | STEP | 向量类型 |\n|---|---|---|---|\n'
    '| `__AVX512F__` | 16 | 64 | `__m512` |\n'
    '| `__AVX__` | 8 | 32 | `__m256` |\n'
    '| `__ARM_NEON` | 4 | 16 | `float32x4_t` |\n'
    '| 标量 | 无 | 无 | `float` |\n\n'
    '三条 SIMD 分支的 `STEP / EPR` 都等于 4（`GGML_F32_ARR`）：**一步固定 4 个寄存器**。')

L.conclusion(
    '两种复用机制，别混起来',
    '| 机制 | 长什么样 | 本课例子 | 谁选 |\n|---|---|---|---|\n'
    '| 一份源码 + 宏映射 | 只写 `GGML_F32_VEC_*` | `vec.cpp` / `vec.h` / `simd-gemm.h` | 预处理器按宏选分支 |\n'
    '| 一份源码 + ISA 特化分支 | 直接写 `_mm512_*` 等 | `ggml_vec_dot_bf16` | `#if` 选分支 |\n'
    '| 同名函数、多份实现 | `ggml_vec_dot_q4_0_q8_0` 写两遍 | `arch/x86/quants.c` 与 `arch/arm/quants.c` | 构建系统按架构选文件 |\n\n'
    '第三种是量化点积家族的做法：`ggml/src/ggml-cpu/CMakeLists.txt` 每次构建只把当前架构那一份 '
    '`arch/<arch>/quants.c` 加进源文件列表（x86 在 `:245`、arm 在 `:103`、riscv 在 `:443`）；'
    '某个架构没实现时，`arch-fallback.h` 会把 `*_generic` 重命名成正式名字顶上去'
    '（`arch-fallback.h:12`）。本课只指路，展开见 L5-05。')

L.conclusion(
    '★ ggml_vec_dot_q4_0_q8_0：AVX2 与 NEON 的真实差异',
    '```text\n'
    '① 函数体不同源：arch/x86/quants.c:701 与 arch/arm/quants.c:297，构建期二选一\n'
    '② 向量宽度：AVX2 走 __AVX__ 分支（EPR 8 / STEP 32 / __m256）\n'
    '             NEON 走 __ARM_NEON 分支（EPR 4 / STEP 16 / float32x4_t）\n'
    '③ 块 scale 的 fp16 -> fp32：x86 用 __F16C__ 的 _cvtsh_ss（simd-mappings.h:63）\n'
    '                             ARM 用 (float)(__fp16)（simd-mappings.h:46-49）\n'
    '```\n\n'
    '共同点：两边都只写 `GGML_F32_VEC_*` 与 `GGML_CPU_FP16_TO_FP32` 这些抽象名，'
    '算法结构（切块、FMA、归约、尾巴）完全共享。')

L.conclusion(
    '标量路径是基准，不是废物',
    '每个向量化函数都保留了标量实现：`vec.cpp` 的 `#else // scalar`、`simd-gemm.h` 的 '
    '`#else // scalar path`。它同时承担三件事：没有 SIMD 的架构、向量化覆盖不到的尾巴、'
    '以及"结果对不对"的对照基准。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
