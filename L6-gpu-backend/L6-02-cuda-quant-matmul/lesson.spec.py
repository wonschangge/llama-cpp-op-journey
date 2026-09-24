#!/usr/bin/env python3
"""L6-02 · ★ CUDA 量化矩阵乘：mmq / mmvq / mmvf —— 课件 spec。

运行：python3 L6-gpu-backend/L6-02-cuda-quant-matmul/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

D     = 'ggml/src/ggml-cuda/'
CU    = D + 'ggml-cuda.cu'
MMQC  = D + 'mmq.cu'
MMQH  = D + 'mmq.cuh'
MMQL  = D + 'mmq-load-tiles.cuh'
MMQV  = D + 'mmq-vec-dot.cuh'
MMVQC = D + 'mmvq.cu'
MMVQH = D + 'mmvq.cuh'
MMVFC = D + 'mmvf.cu'
MMVFH = D + 'mmvf.cuh'
VDQ   = D + 'vecdotq.cuh'
MMA   = D + 'mma.cuh'

L = Lesson(
    id='L6-02',
    layer='L6 · GPU 后端执行',
    title='★ CUDA 量化矩阵乘：mmq / mmvq / mmvf',
    kicker='L6 · GPU 后端执行',
    codecap='ggml/src/ggml-cuda/*（逐字引用）',
    nav={'prev': {'href': '../L6-01-cuda-skeleton/index.html', 'label': 'L6-01 CUDA 后端骨架'},
         'next': {'href': '../L6-03-cuda-flash-attention/index.html', 'label': 'L6-03 ★ CUDA FlashAttention'}},
)

# plan_matrix.py --files 里 L6-02 的 10 个文件，全部声明覆盖。
# 另加 ggml-cuda.cu：本课验收点（mmq / mmvq 分别在什么形状下被选中）的分派点
# ggml_cuda_mul_mat() 就在这个文件里，必须逐字引用，故一并声明（L6-01 也声明它）。
L.cover(MMA, MMQL, MMQV, MMQC, MMQH, MMVFC, MMVFH, MMVQC, MMVQH, VDQ, CU)

L.note('**一句话**：`GGML_OP_MUL_MAT` 在 CUDA 后端上**不是一个内核，而是五个**。'
       '`ggml_cuda_mul_mat()` 按固定顺序试 `should_use_*` 判据，第一个为真的赢；'
       '决定胜负的关键量是 **`ne11`** —— src1 的列数，也就是"一次算几个 token"。')
L.note('本课只讲量化与 float 矩阵乘这三条路（`mmq` / `mmvq` / `mmvf`）：'
       '它们的判据函数、阈值常量、tile 加载、量化点积、以及 batch 大小如何一路决定到'
       '**模板实例**的选取。`mmf`（Tensor Core 大矩阵路径）与 cuBLAS 兜底只作指路。')

# ================================================================== 第 1 幕

L.scene(
    kicker='L6-02 · 全局',
    title='一个 <span class="hl-a">MUL_MAT</span>，五条内核路线',
    sub='ggml_cuda_mul_mat() 里没有 switch(op)，只有一串 should_use_* 判据，按顺序短路。',
    caption='验收点的答案就在这张表里：mmvq 在第 4 条、mmq 在第 5 条，谁先命中谁上。分派点 ggml_cuda_mul_mat() 属于 L6-01 的后端骨架，本课只拆它的 MUL_MAT 分支。',
    src=CU, parts=[(1844, 1876)], duration=19000,
    mark_src=[1844, 1847, 1868, 1869, 1872, 1873, 1876],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', w: '163px', t: 'mmvf', m: 'ggml_cuda_mul_mat_vec_f', b: 'float 权重 × 向量<br>src0 极瘦时才划算' },
  { c: 'a', w: '163px', t: 'mmvq', m: 'ggml_cuda_mul_mat_vec_q', b: '量化权重 × 向量<br>每线程读一整行权重' },
  { c: 'c', w: '163px', t: 'mmq', m: 'ggml_cuda_mul_mat_q', b: '量化权重 × 小矩阵<br>权重搬进 shared tile' },
  { c: 'f', w: '163px', t: 'cuBLAS', m: 'ggml_cuda_mul_mat_cublas', b: '兜底：反量化 + GEMM<br>本课不展开' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const t = U.table(
  ['序', '判据（ggml-cuda.cu 行号）', '调用的内核'],
  [['1', 'ggml_cuda_should_use_mmvf(...)  :1844', 'ggml_cuda_mul_mat_vec_f  :1847'],
   ['2', 'ne01 == 1 且 ne11 &gt; 8  :1851', 'mul_mat_vec_f（转置后复用）  :1861'],
   ['3', 'ggml_cuda_should_use_mmf(...)  :1864', 'ggml_cuda_mul_mat_f  :1865'],
   ['4', 'ggml_cuda_should_use_mmvq(...)  :1868', 'ggml_cuda_mul_mat_vec_q  :1869'],
   ['5', 'ggml_cuda_should_use_mmq(...)  :1872', 'ggml_cuda_mul_mat_q  :1873'],
   ['6', '以上全不满足  :1876', 'ggml_cuda_mul_mat_cublas  :1876']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const trs = t.body.querySelectorAll('tr');
const texts = [
  'MUL_MAT 落到 CUDA 上，并不是无条件走同一个内核。',
  '六个判据按 <span class="k">固定顺序</span> 求值，<span class="v">第一个为真的赢</span>，后面的根本不会被问到。',
  '所以"选哪个内核"= "哪条判据先为真"。本课把第 4、5 条的判据函数逐字拆开。',
  '两台内核都做量化点积，却是两份代码：<span class="k">同一个数学，两种形状假设</span>。',
  '关键变量是 <span class="v">ne11</span>：src1 的列数，即一次乘几个 token。',
  '一句话：<span class="k">矩阵有多瘦，决定用哪套实现</span>。下一幕逐个看判据。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
[0, 1, 2, 3, 4, 5].forEach(i => tl.at(2600 + i * 2300, () => {
  trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  els.forEach((e, k) => { e.style.opacity = (k === [0, 3, 2, 1, 2, 3][i]) ? '1' : '.30'; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(17200, () => { trs.forEach(x => { x.className = ''; }); els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ================================================================== 第 2 幕

L.scene(
    kicker='L6-02 · 判据 1/3',
    title='mmvq：<span class="hl-a">ne11 不超过 8</span> 就走向量内核',
    sub='这个 8 写在头文件的宏里；部分架构和量化类型还会把它压得更小。',
    caption='MMVQ_MAX_BATCH_SIZE 定义在 mmvq.cuh:3。',
    src=MMVQC, parts=[(318, 331)], duration=17000,
    mark_src=[318, 319, 322, 324, 327, 331],
    notes_src={331: '默认落地：ne11 <= MMVQ_MAX_BATCH_SIZE，即 8（mmvq.cuh:3）',
               322: '源码注释给出理由：k-quant 解码贵，而 mvq 每个 token 都要重解码一遍'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '218px', t: '第 0 道门：必须是量化类型',
    m: 'ggml_is_quantized(type)',
    b: 'F32 / F16 / BF16 直接返回 false，<br>在本课里走 mmvf 或 cuBLAS。' },
  { c: 'b', w: '218px', t: '默认阈值：ne11 不超过 8',
    m: 'MMVQ_MAX_BATCH_SIZE = 8',
    b: 'ne11 是 src1 的列数，也就是<br>一次前向要算几个 token。' },
  { c: 'c', w: '218px', t: '按架构和类型再收紧',
    m: 'Ada: Q2_K 到 4 / Q3_K 到 6',
    b: '函数体里还有 Blackwell、Orin、<br>Volta、CDNA1/2 各自的调优阈值。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '先看 <span class="v">ne11</span> 这个量：它是 <span class="k">src1 的列数</span>，'
    + '在 MUL_MAT 里就是 batch / token 数。',
  '<span class="k">非量化类型</span> 在第 3 行就被挡掉 —— 向量内核只服务量化权重。',
  '<span class="v">默认阈值 8</span>：超过 8 个 token 就不再"一列一列地算"，'
    + '让位给能复用权重的 tile 内核。',
  '架构分支把阈值压得更小：<span class="v">Q2_K 到 4、Q3_K 到 6</span>（Ada）。'
    + '<br>理由写在源码注释里：<span class="k">k-quant 解码贵，而 mvq 每个 token 都重解码一遍</span>。',
  '收束：<span class="k">量化权重 × 至多 8 个 token</span> 是 mmvq 的适用形状。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1, 2, 3, 4]); });
tl.at(3400, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [0, 1]); });
tl.at(6800, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [6, 7, 8, 9, 10, 11, 12, 13]); });
tl.at(10200, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [5, 6, 7, 8, 9, 10, 11, 12]); });
tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; U.markLines(document, [6, 7, 8, 9, 10, 11, 12, 13]); });
'''
)

# ================================================================== 第 3 幕

L.scene(
    kicker='L6-02 · 判据 2/3',
    title='★ mmq：有 Tensor Core 就一律走，<span class="hl-a">否则 ne11 小于 64</span>',
    sub='mmq 的门槛比 mmvq 宽得多，因为它把权重搬进 shared memory 之后可以被多个 token 复用。',
    caption='MMQ_DP4A_MAX_BATCH_SIZE 定义在 mmq.cuh:8，值是 64。',
    src=MMQC, parts=[(310, 335)], duration=18000,
    mark_src=[310, 314, 319, 320, 326, 333, 334],
    notes_src={310: '硬性前提：每个 block 至少要 48 KiB shared memory，否则直接让给 BLAS',
               319: 'Turing 及以后（cc >= 7.5）有 mma 指令，mmq 无条件返回 true',
               333: '否则：没有 fp16 tensor core 的卡一律 mmq；有的卡只在 ne11 < 64 时用 mmq'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
  <div class="col" id="bars" style="gap:6px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'e', w: '163px', t: '门 1 · 类型白名单',
    m: 'mmq_supported', b: 'Q4_0/Q8_0/Q2_K~Q6_K/<br>IQ*/MXFP4/NVFP4 才算支持' },
  { c: 'f', w: '163px', t: '门 2 · 48 KiB SRAM',
    m: 'smpbo >= 48 * 1024', b: 'tile 要放进每 block 的<br>shared memory；装不下就退出' },
  { c: 'c', w: '163px', t: '门 3 · Turing 及以上',
    m: 'turing_mma_available(cc)', b: '有 tensor core 指令，<br>直接 return true（不看 ne11）' },
  { c: 'a', w: '163px', t: '门 4 · 老卡看 batch',
    m: 'ne11 &lt; 64', b: '没有 fp16 mma 的卡一律 mmq；<br>有的卡只在 batch 小于 64 时用' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const barsHost = wrap.querySelector('#bars');
barsHost.innerHTML = '<div class="cm" style="margin-bottom:2px">batch（ne11）轴上的胜负</div>';
const segs = [
  { l: 'ne11 1..8 → mmvq', w: '12%', c: 'a' },
  { l: 'ne11 9..63 → mmq（Turing 及以上）', w: '48%', c: 'c' },
  { l: 'ne11 >= 64 → mmq（有 mma）或 cuBLAS', w: '40%', c: 'f' }
];
const bars = segs.map(s => { const b = U.bar(s.l, s.c); barsHost.appendChild(b.el); return b; });
bars.forEach(b => { b.fill.style.width = '0%'; });

const msg = wrap.querySelector('#msg');
const texts = [
  'mmq 的判据比 mmvq 长：<span class="k">四道门</span>，任何一道不过就返回 false。',
  '门 1 是 <span class="v">类型白名单</span>：不在表里的量化类型不会走 mmq。',
  '门 2 是 <span class="k">资源门</span>：tile 常驻 shared memory，至少要 48 KiB。',
  '门 3 最干脆：<span class="v">Turing 及以后直接 true</span>，完全不看 batch。',
  '门 4 才是 batch 判据：<span class="v">ne11 小于 64</span>，这就是 MMQ_DP4A_MAX_BATCH_SIZE。',
  '注意顺序：<span class="k">mmvq 在 mmq 之前</span>被问，所以 1..8 个 token 仍然归 mmvq。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(2900 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  bars.forEach((b, k) => { b.fill.style.width = k === i ? segs[k].w : '0%'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(16500, () => { els.forEach(e => e.style.opacity = '1'); bars.forEach((b, k) => { b.fill.style.width = segs[k].w; }); msg.innerHTML = texts[5]; });
'''
)

# ================================================================== 第 4 幕

L.scene(
    kicker='L6-02 · 判据 3/3',
    title='mmvf：float 路径先看<span class="hl-b">内存对齐</span>，再看 batch',
    sub='三条对齐前提任意一条不满足就崩溃或走不通，所以它们排在 batch 阈值前面。',
    caption='MMVF_MAX_BATCH_SIZE 定义在 mmvf.cuh:3，值与 MMVQ_MAX_BATCH_SIZE 相同。',
    src=MMVFC, parts=[(792, 814)], duration=17000,
    mark_src=[792, 793, 798, 802, 804, 812, 813],
    notes_src={802: '源码注释：指针没按 half2 / nv_bfloat162 / float2 对齐会直接崩'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'e', w: '163px', t: '前提 1 · ne[0] 为偶数',
    m: 'src0_ne[0] % 2 == 0', b: '内核一次吃两个元素<br>（half2 / float2）' },
  { c: 'e', w: '163px', t: '前提 2 · 最内层连续',
    m: 'src0_nb[0] == type_size', b: '第 0 维没有 padding，<br>可以按类型宽度直接读' },
  { c: 'e', w: '163px', t: '前提 3 · 高维 2 元素对齐',
    m: 'nb[i] % (2*ts) == 0', b: '否则 half2 指针未对齐，<br>源码注释说会崩' },
  { c: 'a', w: '163px', t: '然后才看 batch',
    m: 'ne11 &lt;= 3（Ampere F32）', b: 'Turing 上到 4，无 fp32 mma<br>的 AMD 卡上到 8；上限 8' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  'mmvf 服务的是 <span class="k">非量化</span> 权重（F32 / F16 / BF16），'
    + '所以判据里没有"类型白名单"，只有对齐。',
  '<span class="v">ne[0] 必须是偶数</span>：内核按 half2 / float2 成对读，奇数长度无法覆盖。',
  '<span class="v">nb[0] 必须等于 type_size</span>：最内层要连续，'
    + '否则向量化的指针推进不成立。',
  '更高维还要 <span class="v">2 倍类型宽度对齐</span>。源码注释直接写明：'
    + '<span class="k">不对齐会 crash</span>。',
  '三条都过了才轮到 batch：<span class="v">Ampere 上 F32 只到 3</span>，'
    + 'Turing 到 4，上限 8。',
  '对比 mmvq 的 8：<span class="k">float 向量内核比量化向量内核更挑形状</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(2900 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(16000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ================================================================== 第 5 幕

L.scene(
    kicker='L6-02 · 量化点积',
    title='同一份点积源码，<span class="hl-c">两套展开宽度</span>',
    sub='vecdotq.cuh 给每个量化类型定义两个 VDR：MMVQ 用窄的，MMQ 用宽的。',
    caption='VDR 是每个线程一次点积处理的块数；它是唯一把 mmvq 与 mmq 写在同一个文件里的地方。',
    src=VDQ, parts=[(107, 137)], duration=18000,
    mark_src=[107, 115, 116, 118, 124, 129, 136],
    notes_src={115: 'MMVQ：每个线程一次只吃 2 块 Q4_0',
               116: 'MMQ：同一个类型，每个线程一次吃 4 块',
               118: '模板参数 vdr 让同一份展开代码被实例化成两个宽度'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['量化类型', 'VDR（mmvq 用）', 'VDR（mmq 用）', '源码行'],
  [['Q1_0', '1', '4', ':109-110'],
   ['Q2_0', '1', '2', ':112-113'],
   ['Q4_0', '2', '4', ':115-116'],
   ['Q8_0', '2', '8', ':243-244'],
   ['Q2_K', '1', '4', ':363-364'],
   ['Q3_K', '1', '2', ':446-447'],
   ['Q4_K', '2', '8', ':504-505'],
   ['Q6_K', '1', '8', ':623-624']],
  { monoCols: [0, 1, 2, 3] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const trs = t.body.querySelectorAll('tr');
const texts = [
  '同一个 <span class="m">vec_dot_&lt;类型&gt;_q8_1</span>，要同时服务向量内核和 tile 内核。',
  '<span class="v">MMVQ 列的 VDR 都比 MMQ 列小</span>：向量内核的并行度来自'
    + '"更多线程各算一点"，不是"每个线程多算几块"。',
  'tile 内核相反：数据已经在 shared memory 里，<span class="k">让每个线程多算几块</span>'
    + '才能摊薄指令开销。',
  'Q6_K 最极端：<span class="v">1 对 8</span>，相差 8 倍。',
  '实现上只是一个模板参数：<span class="m">vec_dot_q4_0_q8_1_impl&lt;vdr&gt;</span>，'
    + '循环 <span class="m">1..vdr</span> 次 <span class="m">ggml_cuda_dp4a</span>。',
  '所以"两套实现"不是两份代码，而是<span class="k">同一份代码的两个编译期宽度</span>；'
    + '宽度能取多少，由 L1-04 讲的块字节布局决定。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
[0, 1, 2, 3, 4, 5, 6, 7].forEach(i => tl.at(2400 + i * 1750, () => {
  trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(16600, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ================================================================== 第 6 幕

L.scene(
    kicker='L6-02 · tile 点积',
    title='tile 点积把 <span class="hl-d">mma</span> 当积木：16x8 乘 8x8',
    sub='mma.cuh 把 PTX 的 mma 指令包装成 tile 类型；mmq 的点积直接用它拼出一次 tile 乘。',
    caption='tile<> / load_ldmatrix / load_generic / mma 全部来自 mma.cuh（见本课 source.md 第二节）。',
    src=MMQV, parts=[(201, 253)], duration=20000,
    mark_src=[202, 203, 204, 208, 218, 219, 229, 252],
    notes_src={202: 'A 是 16x8 的 x 分片（i 方向 16 行）',
               203: 'B 是 8x8 的 y 分片（k 方向 8 个 int）',
               204: 'C 是 16x8 的累加器，一个 mma 指令直接算出整块'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="flow" style="gap:8px"></div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const flow = wrap.querySelector('#flow');
flow.innerHTML = '<span class="chip a">x 的 shared tile</span><span class="arrow">-></span>'
  + '<span class="chip d">load_ldmatrix</span><span class="arrow">-></span>'
  + '<span class="chip c">mma</span><span class="arrow">-></span>'
  + '<span class="chip b">sum[] 累加</span>';

const defs = [
  { c: 'a', w: '218px', t: 'A：x 的量化数据',
    m: 'tile<16, 8, int>', b: '从 shared memory 用 load_ldmatrix 读，<br>一次搬一整块 16x8。' },
  { c: 'd', w: '218px', t: 'B：y（激活）的分片',
    m: 'tile<8, 8, int>', b: '用 load_generic 读。源码注释：<br>faster than load_ldmatrix。' },
  { c: 'b', w: '218px', t: '结果再乘标量 scale',
    m: 'sum += C.x[l]*dA*dB', b: 'mma 只算整数累加，<br>量化 scale 在循环外补乘。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  'tile 内核的点积不再"一个线程算一个点"，而是<span class="k">一个 warp 算一整块</span>。',
  'A 分片从 shared memory 走 <span class="v">load_ldmatrix</span>：'
    + 'PTX 要求它必须指向 shared memory 且 16 字节对齐。',
  'B 分片走 <span class="v">load_generic</span>；源码注释说明它比 ldmatrix 更快。',
  '<span class="v">mma(C, A[n], B)</span> 一条指令覆盖 16x8 个输出；'
    + '这是 tile 内核吞吐的来源。',
  '整数点积由 mma 完成，<span class="k">量化 scale 在 C.x[l] 上补乘</span>：'
    + '<span class="m">C.x[l]*dA[n][l/2][..]*dB[l%2]</span>。',
  '对照 L5-03 的 CPU 点积：那边是标量乘加展开，这边是<span class="k">一条指令一整块</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3200 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(16800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ================================================================== 第 7 幕

L.scene(
    kicker='L6-02 · tile 加载',
    title='K 循环：<span class="hl-c">搬一次，点两次</span>',
    sub='每轮 K 迭代把 x 的一整条 tile 搬进 shared memory，然后分两半喂给同一个点积函数。',
    caption='两个 vec_dot 的 k00 分别是 0 与 MMQ_TILE_NE_K（= 32，mmq.cuh:116）。',
    src=MMQH, parts=[(904, 941)], duration=20000,
    mark_src=[908, 909, 913, 922, 929, 938],
    notes_src={909: 'load_tiles：把一个 K 迭代的 x tile 协作搬进 shared memory',
               922: '前 32 个元素上做点积（k00 = 0）',
               938: '后 32 个元素上做点积（k00 = MMQ_TILE_NE_K = 32）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="steps" style="gap:7px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '163px', t: '1 · load_tiles', m: 'load_tiles(x, tile_x, ...)', b: '协作把 x 的量化数据<br>搬进 shared memory' },
  { c: 'b', w: '163px', t: '2 · sync + dot', m: 'vec_dot(tile_x, tile_y, sum, 0)', b: 'k00 = 0，<br>前半个 K 分片' },
  { c: 'c', w: '163px', t: '3 · 续搬 tile_y', m: 'tile_y[l] = by0[l]', b: '只换 y 的下一段，<br>x tile 不重搬' },
  { c: 'd', w: '163px', t: '4 · sync + dot', m: 'vec_dot(..., MMQ_TILE_NE_K)', b: 'k00 = 32，<br>后半个 K 分片' }
];
const host = wrap.querySelector('#steps');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">ITER_K</span> 是每轮从显存搬多少 K（mmq.cuh:9 给出 256），'
    + '<span class="m">blocks_per_iter = ITER_K / qk</span>。',
  '关键成本在 <span class="k">x</span>：量化权重。它每轮只搬一次。',
  '搬完立刻在<span class="v">前半段</span>做点积，同时把 y 的后半段续进 tile_y。',
  '同一个 x tile 上做<span class="v">第二次点积</span>，k00 偏移 32 —— '
    + '这就是"搬一次、点两次"。',
  '两块 <span class="m">__syncthreads()</span> 把 shared memory 的读写严格分开：'
    + '<span class="k">写 tile_y 之前必须等上一次点积读完</span>。',
  '这就是 tile 内核摊薄载入成本的方式：<span class="k">权重搬一次，被 J 个 token 和 2 段 K 复用</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3200 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(16200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ================================================================== 第 8 幕

L.scene(
    kicker='L6-02 · tile 加载',
    title='协作加载：一个 warp 摊平一整块 tile',
    sub='线程编号先按 K 方向切（每个线程固定搬哪几块），再按 I 方向切（搬第几行）。',
    caption='MMQ_ITER_K = 256、QR4_0 = 2（ggml-common.h），故 threads_per_row = 256/(4*2) = 32，正好一个 warp。',
    src=MMQL, parts=[(187, 226)], duration=19000,
    mark_src=[187, 203, 204, 211, 217, 224],
    notes_src={203: '一个 warp 负责一整行的 K：256 / (4 * QR4_0) = 32 个线程',
               211: 'nrows == 1 时，行号直接就是 threadIdx.y',
               217: 'x + kbx0 + i*stride + kbx：拿到这一行这一块的源地址'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="cm" id="cap"></div>
  <div class="row" id="grid" style="gap:2px"></div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
wrap.querySelector('#cap').innerHTML =
  '一个 warp（32 线程）覆盖一行 x 的 256 个 K 元素 = 8 个 Q4_0 块，每块 4 个线程：';

const grid = wrap.querySelector('#grid');
const cells = [];
for (let i = 0; i < 32; i++) {
  const kbx = Math.floor(i / 4);
  const c = U.el('div', {
    style: 'width:19px;height:19px;border:1px solid var(--border);border-radius:3px;'
         + 'background:#10151b;display:flex;align-items:center;justify-content:center;'
         + 'font-size:8px;color:var(--dim);font-family:var(--mono)' });
  c.textContent = kbx;
  grid.appendChild(c); cells.push(c);
}

const t = U.table(['式子', '这一行算出什么'],
  [['threads_per_row = MMQ_ITER_K / (4 * QR4_0)', '32 —— 一个 warp 正好一行'],
   ['txi = threadIdx.x % threads_per_row', '我在这一行里的位置'],
   ['kbx = txi / QI4_0', '我负责第几个 Q4_0 块（0..7）'],
   ['kqsx = txi % QI4_0', '我负责块内的第几个 32 位字'],
   ['i = i0 + threadIdx.y', '我负责第几行（nrows == 1）'],
   ['x + kbx0 + i*stride + kbx', '源地址：第 i 行、kbx0 起的第 kbx 块']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const texts = [
  '加载不是"每个线程搬一个元素"，而是先<span class="k">把线程铺到 tile 的两个方向上</span>。',
  '<span class="v">K 方向</span>：8 个块 x 4 个线程 = 32，正好填满一个 warp。',
  '<span class="v">I 方向</span>：每 32 个线程负责一行，靠 <span class="m">i*stride</span> 跳到下一行。',
  '<span class="k">stride 来自 src0->nb[1]</span>：tile 里的行序必须和全局内存里的行序一致。',
  '<span class="m">fallback</span> 分支处理最后一行不满的情况：<span class="m">i = min(i, i_max)</span>。',
  '一格一格的全局读被换成<span class="k">整齐的块读</span>，这就是 tile 加载比 mmvq 更适合大 batch 的原因。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1, 2, 3, 4, 5]); });
tl.at(3200, () => {
  cells.forEach((c, i) => { if (i % 4 === 0) { c.style.background = 'rgba(88,166,255,.25)'; c.style.borderColor = 'var(--a)'; } });
  msg.innerHTML = texts[1];
});
tl.at(6200, () => {
  cells.forEach((c, i) => { c.style.background = (Math.floor(i / 4) % 2) ? 'rgba(210,153,34,.22)' : 'rgba(88,166,255,.25)'; });
  msg.innerHTML = texts[2];
});
tl.at(9200, () => { msg.innerHTML = texts[3]; U.markLines(document, [29]); });
tl.at(12200, () => { msg.innerHTML = texts[4]; U.markLines(document, [25, 26, 27]); });
tl.at(15200, () => { msg.innerHTML = texts[5]; U.markLines(document, [29, 35]); });
'''
)

# ================================================================== 第 9 幕

L.scene(
    kicker='L6-02 · ★ 洞察',
    title='★ batch 不只选内核，<span class="hl-a">还选 tile 的宽度 J</span>',
    sub='进了 mmq 之后，还要在 16 个编译期 J 实例里挑一个：J 是"一次处理几个 token"。',
    caption='J_best 的候选是 8,16,...,128；每个候选对应一个 launch_mul_mat_q<type, J> 模板实例（mmq.cuh:1504-1552）。下面的算例只推演 ntiles_x 公式，实际还要过 :1492 那道 shared memory 门。',
    src=MMQH, parts=[(1477, 1502)], duration=18000,
    mark_src=[1477, 1483, 1486, 1492, 1496, 1498],
    notes_src={1492: '装不进 shared memory 的 J 直接跳过',
               1496: 'ncols_opt 就是 batch：需要多少列 tile 才能盖住它',
               1498: '挑 tile 列数最少的 J；从 8 往上扫，先到先得'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="cm" id="cap"></div>
  <div class="col" id="bars" style="gap:5px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
wrap.querySelector('#cap').innerHTML =
  '以 <span class="m">ncols_opt = 40</span> 为例：ntiles_x = ceil(40 / J)，取最小的那个';

const host = wrap.querySelector('#bars');
const cand = [
  { j: 8, n: 5, c: 'e', ok: true }, { j: 16, n: 3, c: 'e', ok: true },
  { j: 24, n: 2, c: 'c', ok: true }, { j: 32, n: 2, c: 'c', ok: true },
  { j: 40, n: 1, c: 'b', ok: true }, { j: 48, n: 1, c: 'f', ok: false },
  { j: 56, n: 1, c: 'f', ok: false }, { j: 64, n: 1, c: 'f', ok: false }
];
const bars = cand.map(d => {
  const b = U.bar('J = ' + d.j, d.c);
  host.appendChild(b.el);
  b.fill.style.width = '0%';
  b.val.textContent = d.n + ' 列';
  return b;
});

const msg = wrap.querySelector('#msg');
const texts = [
  '进 mmq 只是第一步。内核还要在 <span class="k">16 个编译期 J 实例</span> 里挑一个。',
  'J 是 tile 的列宽，也就是<span class="v">一次搬进来几个 token 的 x</span>。',
  '<span class="m">ntiles_x = (ncols_opt + J - 1) / J</span>：盖住 batch 需要多少个列 tile。',
  '扫描从 J = 8 开始，只接受 <span class="v">ntiles_x 更小</span> 的候选 —— '
    + '所以 J 越宽，列 tile 越少。',
  '但 J 受 <span class="v">shared memory 上限</span> 约束：装不下的候选被 '
    + '<span class="m">continue</span> 跳过。',
  '结果：<span class="k">batch = 40 时，按 ntiles_x 公式 J_best 会落在 40</span>'
    + '（一个列 tile 盖住全部 token）；能不能真的用 40，由 :1492 的 SRAM 门决定。',
  '这与 L6-04 的 <span class="m">template-instances/</span> 是同一机制：'
    + '<span class="k">编译期把 J 摊成 16 份，运行时按 batch 选一份</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
[0, 1, 2, 3, 4, 5].forEach(i => tl.at(2900 + i * 2400, () => {
  bars.forEach((b, k) => {
    b.el.style.opacity = (k <= i + 1) ? '1' : '.30';
    if (k <= i + 1) { b.fill.style.width = Math.min(100, cand[k].n * 20) + '%'; }
  });
  msg.innerHTML = texts[i + 1];
}));
tl.at(16400, () => {
  bars.forEach((b, k) => { b.el.style.opacity = '1'; b.fill.style.width = Math.min(100, cand[k].n * 20) + '%'; });
  bars.forEach((b, k) => { if (k !== 4) { b.el.style.opacity = '.35'; } });
  msg.innerHTML = texts[6];
});
'''
)

# ================================================================== 第 10 幕

L.scene(
    kicker='L6-02 · 收束',
    title='把三套内核压成一张表',
    sub='同一个量化点积，三种形状假设。选择依据只有一条：矩阵有多瘦。',
    caption='下一课 L6-03 讲 FlashAttention —— 那里有完全一样的三条路线（vec / tile / mma）。',
    src=MMVQH, parts=[(1, 9)], duration=20000,
    mark_src=[3, 5],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['内核', '判据（函数 : 行）与阈值', '适用形状', '关键差异'],
  [['mmvf', { html: 'ggml_cuda_should_use_mmvf<br>mmvf.cu:792<br>ne11 &lt;= 3 / 4 / 8' },
    { html: 'float 权重 × 极瘦矩阵<br>（KQV 那类）' },
    { html: '不量化，按 half2/float2 直接读；<br>先过三条对齐前提' }],
   ['mmvq', { html: 'ggml_cuda_should_use_mmvq<br>mmvq.cu:318<br>ne11 &lt;= MMVQ_MAX_BATCH_SIZE = 8' },
    { html: '量化权重 × 1..8 个 token<br>（decode 阶段）' },
    { html: '每线程读一整行权重，权重按<br>token 数重复读；VDR 展开窄' }],
   ['mmq', { html: 'ggml_cuda_should_use_mmq<br>mmq.cu:266<br>turing_mma 一律 true，<br>否则 ne11 &lt; 64' },
    { html: '量化权重 × 9..64 个 token<br>（prefill 阶段）' },
    { html: '权重搬进 shared tile 被 J 个<br>token 复用；VDR 展开宽' }],
   ['cuBLAS', { html: 'ggml-cuda.cu:1876<br>以上都不满足' },
    { html: 'batch 更大，或类型<br>不被 mmq 支持' },
    { html: '先反量化再 GEMM；本课不展开' }]]);
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '在同一张 Turing 卡上，<span class="mono">src0</span> 是 Q4_K 量化权重、'
  + '<span class="mono">src1</span> 分别是 1 个 token 和 32 个 token。'
  + '两次分别走哪个内核？为什么？',
  '<span class="mono">ne11 = 1</span>：<span class="mono">ggml_cuda_should_use_mmvq</span> '
  + '（mmvq.cu:318）先被问到，量化类型且 <span class="mono">ne11 &lt;= MMVQ_MAX_BATCH_SIZE (= 8)</span>，'
  + '返回 true → 走 <span class="mono">ggml_cuda_mul_mat_vec_q</span>。<br>'
  + '<span class="mono">ne11 = 32</span>：mmvq 的判据为 false（32 &gt; 8）；'
  + '接着 <span class="mono">ggml_cuda_should_use_mmq</span>（mmq.cu:266）在 '
  + '<span class="mono">turing_mma_available(cc)</span> 处直接返回 true → 走 '
  + '<span class="mono">ggml_cuda_mul_mat_q</span>，并在其中用 '
  + '<span class="mono">ncols_opt = 32</span> 挑 J（mmq.cuh:1496）—— '
  + 'J 越宽，盖住 32 个 token 所需的列 tile 越少。<br>'
  + '一句话：<span class="mono">ne11</span> 一路决定到 tile 宽度。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '三套内核，一条判据轴：<span class="v">矩阵有多瘦</span>。',
  '<span class="k">瘦到 3 以内</span>：float 路径的 mmvf 最快。',
  '<span class="k">1..8 个 token</span>：mmvq —— 并行度来自"更多线程各读一行权重"。',
  '<span class="k">9..64 个 token</span>：mmq —— 权重搬进 shared tile，被多个 token 复用。',
  '再宽就交给 cuBLAS：<span class="k">tile 复用已经摊不动了</span>。',
  '记住落点：<span class="m">ggml_cuda_mul_mat() 在 ggml-cuda.cu:1823</span>，'
    + '判据函数分散在 mmvq.cu / mmq.cu / mmvf.cu —— 下一课在 FlashAttention 里会再见到同一套三分法。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(14200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ================================================================== source.md

L.section(
    '一、mmvf.cuh：8 行头文件里的两个事实',
    '`mmvf.cuh` 只有 14 行，但它钉死了本课最容易被忽略的两个事实：'
    '**float 向量内核的 batch 上限是 8**，以及这个上限与 `MMVQ_MAX_BATCH_SIZE` 相等'
    '（`ggml-cuda.cu:1924` 处有 `static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);` 兜底）。\n\n'
    '注意：这里的宏只给出**上限**；真正的判据在 `ggml_cuda_should_use_mmvf()` 里，'
    '还要先过三条内存对齐前提。',
    src=MMVFH, parts=[(1, 6)], lang='c')

L.section(
    '二、mma.cuh：把 PTX 的 mma 包成 tile',
    '`mma.cuh` 是 Tensor Core 的抽象层：它把 PTX 的 `mma.sync` 指令包装成 '
    '`tile<I, J, T>` 类型，并规定 A 行主、B 列主、C 列主。'
    'mmq 的 tile 点积（`mmq-vec-dot.cuh`）就是直接拿 `tile<16,8,int>` 拼出来的。\n\n'
    '文件头的注释把约定写死：`J` 的量纲是**物理 32 位元素**，不是逻辑元素。',
    src=MMA, parts=[(1, 17)], lang='c')

L.section(
    '三、mma.cuh：load_generic 与 tile 的 (i, j) 映射',
    'tile 的读写只有两个入口：`load_generic`（任意数据）与 `load_ldmatrix`（shared memory、'
    '16 字节对齐）。它们都靠 `t.get_i(l)` / `t.get_j(l)` 把"第 l 个物理元素"映射回 (i, j)。',
    src=MMA, parts=[(777, 783)], lang='c')

L.section(
    '四、mmq.cuh：tile 常量与 SRAM 布局约定',
    '这一课的"prefetch/tile 加载"全部建立在这几行常量上。注释解释了两件事：'
    '**tile 尺寸与 WARP_SIZE 解耦**（为了兼容不同 warp 大小），'
    '以及**最后一维要 padding 以避免 shared memory bank conflict**。',
    src=MMQH, parts=[(105, 120)], lang='c')

L.section(
    '五、mmvq.cu：向量内核的骨架',
    '`mul_mat_vec_q` 是一个以 `ncols_dst` 为模板参数的 kernel。'
    '`ncols_dst` 就是一次算几个 token，它同时决定 `nwarps` 与 '
    '`rows_per_cuda_block`（见 `calc_nwarps` / `calc_rows_per_block`），'
    '于是"batch 多大"在**编译期**就被固化进寄存器占用和 block 形状里。',
    src=MMVQC, parts=[(599, 630)], lang='c')

L.section(
    '六、mmq.cuh：J 的 16 个模板实例',
    '`mul_mat_q_switch_J` 算出 `J_best` 之后，用一个 switch 把它落到具体的模板实例上。'
    '16 个 `case` 就是 16 个独立的 kernel —— 这正是 L6-04 要讲的 '
    '`template-instances/` 机制在同一处的体现：**编译期穷举，运行时选择**。',
    src=MMQH, parts=[(1504, 1557)], lang='c')

L.footnote_add('覆盖率声明含 11 个文件：计划清单里 L6-02 的 10 个，外加 '
               '`ggml/src/ggml-cuda/ggml-cuda.cu` —— 本课验收点（mmq / mmvq 在什么形状下被选中）'
               '的分派点 `ggml_cuda_mul_mat()` 就在该文件，必须逐字引用。'
               '该文件 L6-01 也已声明，两课共同覆盖同一文件是允许的（覆盖是并集）。')
L.footnote_add('场景中的派生数字（如 `threads_per_row = 256/(4*2) = 32`、`MMQ_TILE_Y_K = 36`）'
               '由 `ggml/src/ggml-common.h` 的 `QK4_0 = 32` / `QR4_0 = 2` / `QR8_1 = 1` 推出，'
               '本课不引用该文件，故不计入本课覆盖率。')
L.footnote_add('`mmf`（Tensor Core 大矩阵路径）与 cuBLAS 兜底只在本课的分派表里出现名字与行号，'
               '其实现分别属于 L6-04 与 cuBLAS，本课不展开。')

L.prereqs('`L6-01`（CUDA 后端骨架 —— 本课的分派点就在它的 `ggml_cuda_mul_mat()` 里）')

L.goal(
    '说出 `ggml_cuda_mul_mat()` 里判据的**求值顺序**，并指出 mmvq 与 mmq 谁先被问到（对应验收点）；',
    '背出 mmvq 的默认阈值 `MMVQ_MAX_BATCH_SIZE = 8` 与 mmq 的 `MMQ_DP4A_MAX_BATCH_SIZE = 64`，'
    '并说明为什么 mmq 的判据里还有一道 `turing_mma_available()` 短路；',
    '解释 `vecdotq.cuh` 为什么给同一类型定义两个 VDR（`_MMVQ` 窄、`_MMQ` 宽）；',
    '说明 `load_tiles` 如何把一个 warp 铺到 tile 的 K 方向与 I 方向上；',
    '解释 `mul_mat_q_switch_J` 为什么需要 16 个模板实例，以及它与 `template-instances/` 的关系。')

L.conclusion(
    '验收点：mmq 与 mmvq 分别在什么形状下被选中',
    '判据在 `ggml_cuda_mul_mat()`（`ggml-cuda.cu:1823`）里按顺序求值，**先命中者胜**：\n\n'
    '```text\n'
    '1  should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)   // :1844\n'
    '2  ne01 == 1 且 ne11 > MMVF_MAX_BATCH_SIZE                     // :1851\n'
    '3  should_use_mmf(...)                                          // :1864\n'
    '4  should_use_mmvq(src0->type, cc, ne11)                        // :1868\n'
    '5  should_use_mmq(src0->type, cc, ne11, 0)                      // :1872\n'
    '6  否则 ggml_cuda_mul_mat_cublas                                // :1876\n'
    '```\n\n'
    '- **mmvq 被选中**：`ggml_cuda_should_use_mmvq()`（`mmvq.cu:318`）返回 true，'
    '即类型是量化类型 **且** `ne11 <= MMVQ_MAX_BATCH_SIZE`（默认 8，`mmvq.cuh:3`）；'
    'Ada/Blackwell/Orin/Volta/CDNA 上这个阈值对 k-quant 还会更小。\n'
    '- **mmq 被选中**：mmvq 先失配（`ne11 > 8`）后，'
    '`ggml_cuda_should_use_mmq()`（`mmq.cu:266`）返回 true —— '
    '类型在白名单里、`smpbo >= 48 KiB`，且要么 `turing_mma_available(cc)`（直接 true），'
    '要么 `ne11 < MMQ_DP4A_MAX_BATCH_SIZE`（64，`mmq.cuh:8`）。\n'
    '- 于是同一张 Turing 卡上：`ne11 = 1..8` 走 mmvq，`ne11 = 9..N` 走 mmq，'
    '类型不被 mmq 支持时才落到 cuBLAS。')

L.conclusion(
    '★ 同一数学，三套实现，判据是"矩阵有多瘦"',
    '| 内核 | 判据函数（行号） | 阈值 | 适用形状 |\n|---|---|---|---|\n'
    '| `mmvf` | `ggml_cuda_should_use_mmvf`（`mmvf.cu:792`） | `ne11 <= 3/4/8` | '
    'float 权重 × 极瘦矩阵（KQV）|\n'
    '| `mmvq` | `ggml_cuda_should_use_mmvq`（`mmvq.cu:318`） | `ne11 <= 8` | '
    '量化权重 × 1..8 个 token（decode）|\n'
    '| `mmq` | `ggml_cuda_should_use_mmq`（`mmq.cu:266`） | turing 一律 true，否则 `ne11 < 64` | '
    '量化权重 × 9..64 个 token（prefill）|\n\n'
    '**batch = 1 时向量内核更优**：并行度来自"更多线程各读一整行权重"；'
    '**batch 大时 tile 内核更优**：权重搬进 shared memory 后被 J 个 token 复用，载入成本被摊薄。')

L.conclusion(
    'batch 一路决定到 tile 宽度 J',
    '进了 mmq 之后，`mul_mat_q_switch_J()`（`mmq.cuh:1477`）还要挑一个 `J_best`：\n\n'
    '```text\n'
    'for J in 8, 16, ..., 128:\n'
    '    if 这个 J 的 kernel 不存在:            continue   // :1487-1490\n'
    '    if mmq_get_nbytes_shared(config) > smpbo: continue // :1492-1494\n'
    '    ntiles_x = ceil(ncols_opt / J)                     // :1496\n'
    '    保留 ntiles_x 更小的 J\n'
    '```\n\n'
    '`ncols_opt` 就是 batch（dense 路径下两个字段都取 `ne1`，见 `mmq.cu:174`）。'
    '最后 switch 到 16 个 `launch_mul_mat_q<type, J>` 之一（`mmq.cuh:1504-1552`）—— '
    '**编译期穷举、运行时选择**，与 L6-04 的 `template-instances/` 是同一机制。')

L.conclusion(
    '跨课呼应',
    '- **L1-04（量化块布局）**：`VDR_*` 的宽窄、`QI4_0` / `QR4_0` 这些常量，'
    '前提都是块的内存布局。tile 加载读的就是 `bxi->qs` 与 `bxi->d`。\n'
    '- **L5-03（CPU 侧 SIMD 基础设施）**：CPU 的 `ggml_vec_dot_*` 与这里的 '
    '`vec_dot_<type>_q8_1` 是同一个数学，只是这边把标量乘加换成了 `ggml_cuda_dp4a` / `mma`。\n'
    '- **L6-01（分派点）**：`ggml_cuda_mul_mat()` 是本课所有判据的宿主；'
    '本课只拆 MUL_MAT 这一条 op。\n'
    '- **L6-04（模板实例）**：`mul_mat_q_switch_J` 的 16 个 `case` 与 '
    '`template-instances/` 目录解决的是同一个问题 —— C++ 模板必须被显式实例化才能进 '
    '单独编译的 `.cu` 文件。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
