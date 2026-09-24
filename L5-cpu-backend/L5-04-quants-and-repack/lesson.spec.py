#!/usr/bin/env python3
"""L5-04 · 量化与 repack —— 课件 spec。

运行：python3 L5-cpu-backend/L5-04-quants-and-repack/lesson.spec.py

覆盖：ggml/src/ggml-cpu/{quants.c,quants.h,repack.cpp,repack.h,traits.cpp,traits.h}
所有代码引用都按【行号】从上游抽取，绝不手抄。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

QC = 'ggml/src/ggml-cpu/quants.c'
QH = 'ggml/src/ggml-cpu/quants.h'
RC = 'ggml/src/ggml-cpu/repack.cpp'
RH = 'ggml/src/ggml-cpu/repack.h'
TC = 'ggml/src/ggml-cpu/traits.cpp'
TH = 'ggml/src/ggml-cpu/traits.h'


def hl(parts, notes, *lns):
    """这些上游行号在渲染后的【下标】（0-based）；parts 是本幕引用的行区间列表。

    两件事会移动下标：注解行（插在上游行之后、不占行号）与多段引用之间的分隔行。
    所以必须按 parts / notes 重算 —— 直接手写 marks 下标迟早会数错。
    """
    def idx(ln):
        i = 0
        for pi, (a, b) in enumerate(parts):
            if pi:
                i += 1                                  # //>> ---- src:a-b ---- 分隔行
            if ln <= b:
                return i + (ln - a) + sum(1 for k in notes if k < ln)
            i += b - a + 1
        raise SystemExit(f'行号 {ln} 不在引用区间内: {parts}')
    return '[' + ','.join(str(idx(ln)) for ln in lns) + ']'


def fill(tpl, **kw):
    """把 @@NAME@@ 占位符换成实际值。

    不用 % 或 f-string：visual 里大量出现 %（ne[1] % 8）与 {}（JS/CSS），
    %-format 与 f-string 都会被这些字符咬到。
    """
    for k, v in kw.items():
        tpl = tpl.replace('@@' + k + '@@', v)
    return tpl


L = Lesson(
    id='L5-04',
    layer='L5 · CPU 后端执行',
    title='★ 量化与 repack',
    codecap='ggml-cpu/quants.{c,h} · repack.{c,h} · traits.{c,h}（逐字引用）',
    nav={'prev': {'href': '../L5-03-simd-infrastructure/index.html',
                  'label': 'L5-03 ★ 向量化基础设施'},
         'next': {'href': '../L5-05-multiarch-and-vendor/index.html',
                  'label': 'L5-05 ★ 多架构 SIMD 与厂商加速'}},
)

L.note('**一句话**：`GGML_TYPE_Q4_0` 只规定了"32 个权重一块、一块 18 字节、怎么反量化"；'
       '它**没有规定这些块在内存里怎么排**。CPU 后端用 `repack` 缓冲把 N 行权重重排成'
       '**交错块**（`block_q4_0x4` / `x8` / `x16`），让一条 SIMD 载入就能同时拿到'
       '**多行的同一批列** —— 这就是 repack 加速点积的秘密。')
L.note('这一课回答两个问题：**交错之后的布局为什么能加速点积**，以及 '
       '**traits 如何按 ISA 与形状选出 4x4 / 4x8 / 8x8 / 16x1 这些变体**。'
       '回顾 L1-04：原始块布局（`d` + `qs`）在那里逐字讲过；回顾 L1-01：'
       '`nb[]` 是 `type` 的函数，而 repack 换的是"块与块之间怎么摆"，不是类型号。')

# ------------------------------------------------------------------ 第 1 幕

V1 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">GGUF 权重</span><span class="arrow">-></span>
    <span class="chip a">repack 缓冲</span><span class="arrow">-></span>
    <span class="chip b">交错块 x4 / x8 / x16</span><span class="arrow">-></span>
    <span class="chip c">F32 激活</span><span class="arrow">-></span>
    <span class="chip d">gemv / gemm</span>
  </div>
  <div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'quants.{c,h}', b: 'CPU 侧的量化与点积 ABI：<br>quantize_row_* 与 ggml_vec_dot_*',
    m: 'quantize_row_q8_0()' },
  { c: 'b', t: 'repack.{c,h}', b: '交错布局的定义与重排实现：<br>block&lt;K,N&gt; 与 gemv/gemm',
    m: 'block_q4_0x4' },
  { c: 'c', t: 'traits.{c,h}', b: '钩子协议与变体选择：<br>这个 op 谁来算、用哪个变体',
    m: 'tensor_traits' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');
const msg = wrap.querySelector('#msg');
const texts = [
  '一次 MUL_MAT 落到 CPU 上要过四道手：<span class="k">权重布局 -> 激活量化 -> 分块内核 -> 写回 F32</span>。',
  'A 面是 <span class="v">GGML_TYPE_Q4_0</span>：块多大、一块几字节、怎么反量化 —— L1-04 讲的就是它。',
  'B 面是 <span class="k">这些块在内存里怎么摆</span>。类型号不管这件事，它归 CPU 后端的 <span class="v">repack</span> 缓冲类型管。',
  '同一个类型号可以有多种排法：<span class="v">4x4 / 4x8 / 8x8 / 16x1</span>，各对着一种 SIMD 宽度。<br>本课回答：<span class="k">交错布局为什么能加速点积</span>？<span class="k">谁按 ISA 选变体</span>？'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
  U.markLines(document, @@M0@@);
}));
tl.at(12400, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[3];
  U.markLines(document, @@M1@@);
});
'''

L.scene(
    kicker='L5-04 · 全局',
    title='同一份权重，两种排法：<span class="hl-a">类型号固定</span>，<span class="hl-b">布局可换</span>',
    sub='一个 Q4_0 张量既可以"每行每 32 个权重一块"顺序存，也可以"4 行交错"存；内核跟着换。',
    caption='本课覆盖 6 个文件：quants.{c,h}（标量基线与 ABI）、repack.{c,h}（交错布局与 gemv/gemm）、'
            'traits.{c,h}（谁来认领这个 op）。模型加载期是否启用 repack 缓冲由 `--no-repack` 控制。',
    src=QH, parts=[(8, 22)], duration=16000,
    mark_src=[14, 21, 22],
    visual=fill(V1, M0=hl([(8, 22)], {}, 14, 21), M1=hl([(8, 22)], {}, 14, 21, 22))
)

# ------------------------------------------------------------------ 第 2 幕

V2 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="s0"></div><div class="row" id="ops" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

function strip(host, caption, segs) {
  host.appendChild(U.el('div', { class: 'cm', text: caption, style: 'margin-bottom:3px' }));
  const s = U.el('div', { style: 'display:flex;height:26px;border:1px solid var(--border);border-radius:5px;overflow:hidden;width:100%' });
  segs.forEach(g => {
    const e = U.el('div', { style: 'flex:' + g.n + ' 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:8.5px;overflow:hidden;white-space:nowrap;background:' + g.bg + ';color:var(--' + g.c + ');border-right:1px solid var(--border)' });
    e.textContent = g.t;
    s.appendChild(e);
  });
  host.appendChild(s);
  return s;
}
const s0 = wrap.querySelector('#s0');
strip(s0, '原始布局：block_q4_0[] —— 每 32 个权重一块（18 字节），一块接一块',
  [{ n: 2,  t: 'd',      c: 'a', bg: 'rgba(88,166,255,.30)' },
   { n: 16, t: 'qs[16] —— 元素 0..15 与 16..31 挤在同一个字节里', c: 'b', bg: 'rgba(63,185,80,.16)' },
   { n: 2,  t: 'd',      c: 'a', bg: 'rgba(88,166,255,.30)' },
   { n: 16, t: 'qs[16]', c: 'b', bg: 'rgba(63,185,80,.16)' },
   { n: 2,  t: 'd',      c: 'a', bg: 'rgba(88,166,255,.30)' },
   { n: 16, t: 'qs[16]', c: 'b', bg: 'rgba(63,185,80,.16)' }]);

const opsHost = wrap.querySelector('#ops');
const ops = [
  { c: 'a', t: '1 载入', m: 'x[ib].qs[j]', b: '一个字节里两个元素' },
  { c: 'c', t: '2 低半', m: 'qs[j] &amp; 0x0F', b: '元素 j' },
  { c: 'c', t: '3 高半', m: 'qs[j] &gt;&gt; 4', b: '元素 j + 16' },
  { c: 'd', t: '4 还原', m: '- 8 后乘加', b: 'y[ib].qs[j] / qs[j + 16]' }
];
const opEls = ops.map(d => { const e = U.card(d, { style: 'width:160px' }); opsHost.appendChild(e); return e; });
opEls.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '基线内核直接按数组下标读：<span class="v">block_q4_0* x</span> 配 <span class="v">block_q8_0* y</span>。',
  '每个字节要拆成两个元素：<span class="v">(x[ib].qs[j] &amp; 0x0F) - 8</span> 取低半，<span class="v">(x[ib].qs[j] &gt;&gt; 4) - 8</span> 取高半。',
  '拆出来的两个元素配的是 y 里<span class="k">相隔 16 的两个 int8</span>：<span class="v">y[ib].qs[j]</span> 与 <span class="v">y[ib].qs[j + qk/2]</span>。',
  'SIMD 版本同样要拆：一条载入里<span class="k">有用的位是散开的</span>，得先做位运算再乘加。',
  '<span class="k">这就是 repack 要解决的问题</span>：把"必须现场拆"改成"存的时候就已经排好"。'
];
opEls.forEach((e, i) => tl.at(700 + i * 2600, () => {
  opEls.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
  U.markLines(document, @@M0@@);
}));
tl.at(11600, () => {
  opEls.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[4];
  U.markLines(document, @@M1@@);
});
'''

NS2 = {247: '低半是元素 j、高半是元素 j+16 —— 一个字节必须先拆成两个 int8'}

L.scene(
    kicker='L5-04 · 基线',
    title='基线：标量点积要<span class="hl-c">先拆 nibble</span>',
    sub='原始 Q4_0 里一个字节管相隔 16 的两个元素 —— 载入之后还得 mask / shift / 减 8 才能变成 int8。',
    caption='回顾 L1-04：block_q4_0 = 2 字节 d + 16 字节 qs；qs[j] 的低半是元素 j，高半是元素 j+16。',
    src=QC, parts=[(225, 259)], duration=18000,
    mark_src=[225, 236, 237, 247, 248, 250, 251],
    notes_src=NS2,
    visual=fill(V2, M0=hl([(225, 259)], NS2, 236, 237, 247, 248),
                M1=hl([(225, 259)], NS2, 250, 251))
)

# ------------------------------------------------------------------ 第 3 幕

V3 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="s0"></div><div id="s1"></div><div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

function strip(host, caption, segs) {
  host.appendChild(U.el('div', { class: 'cm', text: caption, style: 'margin-bottom:3px' }));
  const s = U.el('div', { style: 'display:flex;height:24px;border:1px solid var(--border);border-radius:5px;overflow:hidden;width:100%' });
  segs.forEach(g => {
    const e = U.el('div', { style: 'flex:' + g.n + ' 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:8.5px;overflow:hidden;white-space:nowrap;background:' + g.bg + ';color:var(--' + (g.c || 'muted') + ');border-right:1px solid var(--border)' });
    e.textContent = g.t || '';
    s.appendChild(e);
  });
  host.appendChild(s);
  return s;
}
const RBG = ['rgba(88,166,255,.30)', 'rgba(63,185,80,.26)', 'rgba(210,153,34,.26)', 'rgba(188,140,255,.26)'];
const RFAINT = ['rgba(88,166,255,.13)', 'rgba(63,185,80,.13)', 'rgba(210,153,34,.13)', 'rgba(188,140,255,.13)'];
const s0 = wrap.querySelector('#s0'), s1 = wrap.querySelector('#s1');
strip(s0, 'repack 前：4 个 block_q4_0 各自独立（每行一块，18 字节 x 4 = 72 字节）',
  [{ n: 2, t: 'd', c: 'a', bg: RBG[0] }, { n: 16, t: 'qs[16] 行 0', c: 'b', bg: RFAINT[0] },
   { n: 2, t: 'd', c: 'a', bg: RBG[1] }, { n: 16, t: 'qs[16] 行 1', c: 'b', bg: RFAINT[1] },
   { n: 2, t: 'd', c: 'a', bg: RBG[2] }, { n: 16, t: 'qs[16] 行 2', c: 'b', bg: RFAINT[2] },
   { n: 2, t: 'd', c: 'a', bg: RBG[3] }, { n: 16, t: 'qs[16] 行 3', c: 'b', bg: RFAINT[3] }]);
const segs1 = [{ n: 8, t: 'd[4]', c: 'a', bg: 'rgba(88,166,255,.30)' }];
for (let i = 0; i < 16; i++) {
  const r = i % 4;
  segs1.push({ n: 4, t: i < 4 ? ('r' + r) : '', c: 'b', bg: RFAINT[r] });
}
strip(s1, 'repack 后：1 个 block_q4_0x4 = block&lt;4,4&gt;（72 字节，d 集中 + qs 按 4 字节交错）', segs1);

const t = U.table(['等式', '左边', '右边'],
  [['sizeof(block_q4_0) x 4', '18 x 4', '72'],
   ['sizeof(block&lt;4,4&gt;)', '4 x 2（d）+ 32 x 2（qs）', '72']],
  { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  'repack 加的是<span class="k">新布局</span>，不是新类型号：<span class="v">block&lt;K,N&gt;</span> 是编译期模板。',
  '<span class="v">ggml_half d[N]</span>：N 行的 scale 先集中放在块首。',
  '<span class="v">int8_t qs[...]</span>：N 行的 4-bit 数据交错放在后面 —— "SIMD 友好"就来自这里。',
  '<span class="v">block_q4_0x4 = block&lt;4,4&gt;</span>、<span class="v">block_q4_0x8 = block&lt;4,8&gt;</span>、<span class="v">block_q4_0x16 = block&lt;4,16&gt;</span>：N = 交错几行。',
  '<span class="v">block_q8_0x4 / x8 / x16 = block&lt;8,N&gt;</span>：激活侧用 K = 8 的同一族模板。',
  'static_assert 把等式钉在编译期：<span class="v">4 x 2 + 32 x 2 = 72 = 4 x 18</span> —— <span class="k">字节数一个不多一个不少</span>，只是排列变了。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, @@M0@@); });
tl.at(3400, () => { msg.innerHTML = texts[1]; U.markLines(document, @@M1@@); });
tl.at(6400, () => { msg.innerHTML = texts[2]; U.markLines(document, @@M2@@); });
tl.at(9800, () => { msg.innerHTML = texts[3]; U.markLines(document, @@M2@@); });
tl.at(13200, () => { msg.innerHTML = texts[4]; U.markLines(document, @@M3@@); });
tl.at(16500, () => {
  rows.forEach(r => { r.className = 'on'; });
  msg.innerHTML = texts[5];
  U.markLines(document, @@M4@@);
});
'''

PRH = [(26, 27), (31, 46)]
NS3 = {27: '下一个成员是 qs：N 行的 4-bit 数据交错放在后面（总位数 = QK_0<K>() * N * K，'
           '即下面 static_assert 里的 QK8_0 * 2 / * 4 / * 8）；'
           'K 由 QK_0<K>() 选量化家族（1 -> QK1_0、4 -> QK4_0、8 -> QK8_0），N 是交错进同一块的行数'}

L.scene(
    kicker='L5-04 · 核心',
    title='★ <span class="hl-a">repack</span>：交错布局从类型号里搬进后端',
    sub='block<K,N>：一个块里放 N 行的 scale 和 N 行的 quants —— 字节总数不变，只是换了顺序。',
    caption='回顾 L1-01：旧式的 GGML_TYPE_Q4_0_4_4 把 4x4 交错编进了类型号（ggml.h:421 注明 support has been removed）；'
            '现在同一件事改由后端的 repack 缓冲做。引用区间为 repack.h:26-27 与 31-46（28-29 行是 qs 成员声明，见注解）。',
    src=RH, parts=PRH, duration=20000,
    mark_src=[26, 27, 33, 41, 42, 44, 46],
    notes_src=NS3,
    visual=fill(V3, M0=hl(PRH, NS3, 26), M1=hl(PRH, NS3, 27),
                M2=hl(PRH, NS3, 31, 33), M3=hl(PRH, NS3, 41, 42, 44, 46),
                M4=hl(PRH, NS3, 31, 33, 35, 37))
)

# ------------------------------------------------------------------ 第 4 幕

V4 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div id="src"></div><div id="dst"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const srcHost = wrap.querySelector('#src'), dstHost = wrap.querySelector('#dst');
const BGC = ['rgba(88,166,255,.30)', 'rgba(63,185,80,.26)', 'rgba(210,153,34,.26)', 'rgba(188,140,255,.26)'];
const CC  = ['a', 'b', 'c', 'd'];

srcHost.appendChild(U.el('div', { class: 'cm', text: '输入：4 个 block_q4_0 的 qs[16]（每行 16 字节，按 4 字节一组切）', style: 'margin-bottom:3px' }));
const srcCells = [];
for (let r = 0; r < 4; r++) {
  const row = U.el('div', { style: 'display:flex;align-items:center;gap:5px;margin-bottom:3px' });
  row.appendChild(U.el('div', { class: 'cm', style: 'width:24px;flex:0 0 auto;margin:0', text: 'r' + r }));
  const strip = U.el('div', { style: 'flex:1 1 auto;display:flex;height:16px;border:1px solid var(--border);border-radius:4px;overflow:hidden' });
  const cs = [];
  for (let c = 0; c < 4; c++) {
    const e = U.el('div', { style: 'flex:1 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--' + CC[r] + ');border-right:1px solid var(--border);background:#10151b' });
    e.textContent = r + ',' + c;
    strip.appendChild(e); cs.push(e);
  }
  row.appendChild(strip); srcHost.appendChild(row); srcCells.push(cs);
}
dstHost.appendChild(U.el('div', { class: 'cm', text: '输出：block_q4_0x4.qs[64] —— 第 i 个 4 字节块来自第 i%4 行（d[4] 另行集中搬）', style: 'margin-bottom:3px' }));
const dstStrip = U.el('div', { style: 'display:flex;height:20px;border:1px solid var(--border);border-radius:4px;overflow:hidden' });
const dstCells = [];
for (let i = 0; i < 16; i++) {
  const e = U.el('div', { style: 'flex:1 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:8px;border-right:1px solid var(--border);background:#10151b' });
  dstStrip.appendChild(e); dstCells.push(e);
}
dstHost.appendChild(dstStrip);

const msg = wrap.querySelector('#msg');
const texts = [
  '输入侧每行 16 字节，输出侧 64 字节：<span class="v">end = QK4_0 x 2 / blck_size_interleave = 16</span> 次循环。',
  '第 i 次循环搬第 <span class="k">i%4</span> 行的第 <span class="k">(i/4)</span> 个 4 字节，放到输出的第 <span class="k">i</span> 个 4 字节位置。',
  '搬运的同时 <span class="v">elems ^= 0x88888888</span>：把无符号 4-bit 码变成 4-bit 补码。',
  '于是内核拿到 nibble 后 <span class="v">&lt;&lt; 4</span> / <span class="v">&amp; 0xF0</span> 就是有符号值，<span class="k">不需要再减 8</span>（对比第 2 幕的标量基线）。',
  'x8 变体是同一个模式，只是粒度从 4 字节变成 8 字节：<span class="v">xor_mask = 0x8888888888888888</span>。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, @@M0@@); });
tl.at(3800, () => { msg.innerHTML = texts[1]; U.markLines(document, @@M1@@); });
for (let i = 0; i < 16; i++) {
  tl.at(7000 + i * 480, () => {
    const r = i % 4, c = Math.floor(i / 4);
    srcCells[r][c].style.background = BGC[r];
    dstCells[i].style.background = BGC[r];
    dstCells[i].style.color = 'var(--' + CC[r] + ')';
    dstCells[i].textContent = r + ',' + c;
  });
}
tl.at(15600, () => { msg.innerHTML = texts[2]; U.markLines(document, @@M2@@); });
tl.at(18400, () => { msg.innerHTML = texts[3]; U.markLines(document, @@M3@@); });
tl.at(20400, () => { msg.innerHTML = texts[4]; U.markLines(document, @@M4@@); });
'''

NS4 = {3102: 'xor 0x88：Q4_0 存的是无符号 4-bit 码（0..15），异或后变成 4-bit 补码 —— 内核里 << 4 / & 0xF0 直接得到有符号值，省掉减 8'}

L.scene(
    kicker='L5-04 · 重排',
    title='<span class="hl-d">make_block_q4_0x4</span>：4 行轮流搬，顺手异或掉偏置',
    sub='src_id = i % 4、src_offset = (i/4) x 4、dst_offset = i x 4 —— 每次只搬 4 字节，轮到下一行。',
    caption='同一个函数用 blck_size_interleave = 4 或 8 决定每次搬几个字节；x8 变体走 8 字节一组。',
    src=RC, parts=[(3083, 3122)], duration=22000,
    mark_src=[3083, 3086, 3087, 3094, 3095, 3096, 3097, 3101, 3102, 3103, 3106, 3107, 3114],
    notes_src=NS4,
    visual=fill(V4,
                M0=hl([(3083, 3122)], NS4, 3083, 3094, 3095, 3096, 3097),
                M1=hl([(3083, 3122)], NS4, 3094, 3095, 3096, 3097),
                M2=hl([(3083, 3122)], NS4, 3101, 3102, 3103),
                M3=hl([(3083, 3122)], NS4, 3106, 3107, 3114),
                M4=hl([(3083, 3122)], NS4, 3086, 3087))
)

# ------------------------------------------------------------------ 第 5 幕

V5 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="cards" style="gap:12px"></div><div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '权重侧（一次性）', b: '模型加载时把权重重排成交错块，之后只读。<br>谁来做：repack 缓冲的 set_tensor。',
    m: 'block_q4_0x4 / x8 / x16' },
  { c: 'c', t: '激活侧（每次前向）', b: '激活是运行时数据，每次前向都要量化一次；<br>4 行一组写进 wdata 临时区。',
    m: 'block_q8_0x4 / block_q8_Kx4' }
];
const cards = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:322px' }); cards.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.35');

const t = U.table(['走哪条路', '触发条件', '激活布局', '依据'],
  [['gemm', 'src1 行数 > 3', 'block_q8_0x4（4 行交错）', 'repack.cpp:1810'],
   ['gemv', '残留的 1..3 行', 'block_q8_0（每行独立）', 'repack.cpp:773'],
   ['from_float', '单行量化', 'quantize_row_q8_0（标量 ABI）', 'repack.cpp:4700-4705']],
  { monoCols: [0, 2, 3] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '权重重排是<span class="k">一次性成本</span>；激活重排是<span class="k">每次前向的成本</span> —— 所以只给"值得"的形状做。',
  '<span class="v">quantize_mat_q8_0_4x4</span> 一次吃 4 行：<span class="v">x[row_iter * k + ...]</span>，4 行的 amax 各算各的。',
  '4 行的 scale 写进 <span class="v">y[i].d[row_iter]</span>；4 行的 int8 按同一个 <span class="v">blck_size_interleave = 4</span> 交错写进 <span class="v">y[i].qs[j]</span>。',
  '交错公式与权重侧同构：<span class="v">src_id = (j % 16) / 4</span>、<span class="v">src_offset = (j / 16) * 4 + j % 4</span>。',
  '只有 <span class="v">ne11</span> 是 4 的倍数时才能整批交错；余下 1..3 行退回 <span class="v">from_float</span>，由 gemv 消费。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; els[0].style.opacity = '1'; });
tl.at(4000, () => { msg.innerHTML = texts[1]; els[1].style.opacity = '1'; U.markLines(document, @@M0@@); });
tl.at(7600, () => { msg.innerHTML = texts[2]; U.markLines(document, @@M1@@); });
tl.at(11200, () => { msg.innerHTML = texts[3]; U.markLines(document, @@M2@@); });
tl.at(14800, () => {
  msg.innerHTML = texts[4];
  rows.forEach(r => { r.className = 'on'; });
  U.markLines(document, @@M3@@);
});
'''

NS5 = {140: '目标类型是 block_q8_0x4 而不是 block_q8_0 —— 4 行的量化结果直接写成交错块'}

L.scene(
    kicker='L5-04 · 契约',
    title='交错是<span class="hl-c">双边</span>的：激活也要按 4 行一组量化',
    sub='权重在加载时重排一次；激活每次前向都要重排 —— 两侧必须用同一个交错粒度才配得上。',
    caption='gemm 内核把 vy 直接当 block_q8_0x4 读（repack.cpp:1810）并断言 nr % 4 == 0（1792）；'
            '单行残留走 gemv，读的是普通 block_q8_0（773）。两者的分工在 forward_mul_mat_one_chunk（4637-4647）。',
    src=RC, parts=[(135, 171)], duration=20000,
    mark_src=[135, 136, 140, 143, 147, 148, 152, 159, 162, 163, 164, 165, 167, 168],
    notes_src=NS5,
    visual=fill(V5,
                M0=hl([(135, 171)], NS5, 135, 140, 143),
                M1=hl([(135, 171)], NS5, 147, 148, 152, 159),
                M2=hl([(135, 171)], NS5, 162, 163, 164, 165, 167, 168),
                M3=hl([(135, 171)], NS5, 136))
)

# ------------------------------------------------------------------ 第 6 幕

V6 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:12px;align-items:flex-start">
    <div class="col" id="grid" style="gap:4px"></div>
    <div class="col grow" id="side" style="gap:6px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const grid = wrap.querySelector('#grid');
function gRow(label, isHead) {
  const r = U.el('div', { style: 'display:flex;gap:4px;align-items:center' });
  r.appendChild(U.el('div', { class: 'cm', style: 'width:22px;flex:0 0 auto;margin:0;text-align:right', text: label }));
  const rowCells = [];
  for (let j = 0; j < 4; j++) {
    const e = U.el('div', { style: 'width:40px;height:20px;flex:0 0 auto;display:flex;align-items:center;justify-content:center;font-size:8.5px;border:1px solid var(--border);border-radius:3px;background:#10151b;color:var(--dim)' });
    e.textContent = isHead ? ('j' + j) : '';
    r.appendChild(e); rowCells.push(e);
  }
  grid.appendChild(r);
  return rowCells;
}
gRow('', true);
const cells = [];
for (let m = 0; m < 4; m++) { cells.push(gRow('m' + m, false)); }

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="formula" style="font-size:9.5px">权重：<span class="m">b_ptr[l].qs[k * 4 * 4 + j * 4 + i]</span></div>' +
  '<div class="formula" style="font-size:9.5px">激活：<span class="m">a_ptr[l].qs[k * 4 * 4 + m * 4 + i]</span></div>' +
  '<div class="formula" style="font-size:9.5px">内层：<span class="m">((v0 * a[i]) + (v1 * a[i + 16])) &gt;&gt; 4</span></div>' +
  '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">' +
  '交错把"行"变成向量载入的一个维度</div><div class="cb">同一批列在 4 行上是<b>连续</b>的：' +
  'j 每加 1，地址只走 4 字节（G = 4）。所以一条 16 字节载入拿到的是"4 行 x 4 字节"，' +
  '而不是"1 行的 16 字节"。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '看清三个下标：<span class="v">k</span> 走列（每块 4 组）、<span class="v">j</span> 走权重行、<span class="v">m</span> 走 token。',
  '权重的第 j 行：<span class="v">k * 16 + j * 4 + i</span> —— j 每加 1 只挪 <span class="k">4 字节</span>。',
  '激活的第 m 个 token：<span class="v">k * 16 + m * 4 + i</span> —— 同一个交错规则。',
  '一个 <span class="v">(m, j)</span> 格子做完 <span class="v">blocklen = 4</span> 次乘加，得到 <span class="v">sumf[m][j]</span>。',
  '一次 4 x 4 循环出 <span class="k">16 个部分和</span>：这 16 个格子正好用满一条向量寄存器的乘加吞吐。',
  '<span class="v">&gt;&gt; 4</span> 是收尾：v0 被 <span class="v">&lt;&lt; 4</span> 放大过 16 倍，两个乘积一起右移还原。',
  '所以"加速"不是玄学：<span class="k">载入宽度 = 行数 x 行内字节</span>，交错让行数也进了向量。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, @@M0@@); });
tl.at(3400, () => { msg.innerHTML = texts[1]; U.markLines(document, @@M1@@); });
tl.at(6200, () => { msg.innerHTML = texts[2]; U.markLines(document, @@M2@@); });
[0, 1, 2, 3].forEach(j => tl.at(9000 + j * 1900, () => {
  msg.innerHTML = texts[3];
  for (let m = 0; m < 4; m++) {
    cells[m][j].style.background = 'rgba(88,166,255,.24)';
    cells[m][j].style.borderColor = 'var(--a)';
    cells[m][j].style.color = 'var(--text)';
  }
  U.markLines(document, @@M3@@);
}));
tl.at(17600, () => {
  for (let m = 0; m < 4; m++) for (let j = 0; j < 4; j++) {
    cells[m][j].textContent = 'sumf[' + m + '][' + j + ']';
    cells[m][j].style.background = 'rgba(63,185,80,.20)';
    cells[m][j].style.borderColor = 'var(--b)';
    cells[m][j].style.color = 'var(--b)';
  }
  msg.innerHTML = texts[4];
  U.markLines(document, @@M4@@);
});
tl.at(20800, () => { msg.innerHTML = texts[5]; U.markLines(document, @@M5@@); });
tl.at(22600, () => { msg.innerHTML = texts[6]; U.markLines(document, @@M6@@); });
'''

NS6 = {1822: '<< 4 把低 nibble 挪到高位、保住它的符号位；& 0xF0 原地取高 nibble —— 两者都已是"值-8"的补码',
       1827: 'sumf[m][j] = 第 m 个 token 与第 j 行权重的部分和；4 x 4 = 16 个一起出来',
       1831: '紧接着是写回段：把 sumf[4][4] 按 (y*4+m) 行、x*4+j 列写进 s（repack.cpp:1832-1835）'}

L.scene(
    kicker='L5-04 · 收益',
    title='★ 交错块怎么读：<span class="hl-a">偏移 = k·N·G + j·G + i</span>',
    sub='k 走列、j 走行、i 走行内字节（本例 N = G = 4）—— 一条向量载入就同时拿到多行的同一批列。',
    caption='这是 generic 实现（可读版，第 j 行权重配第 m 个 token，一次出 16 个部分和）；'
            'x86 / ARM 上的真身是 AVX2 / NEON-i8mm 版本，见 L5-05。',
    src=RC, parts=[(1785, 1831)], duration=24000,
    mark_src=[1805, 1810, 1812, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1827],
    notes_src=NS6,
    visual=fill(V6,
                M0=hl([(1785, 1831)], NS6, 1805, 1810, 1812),
                M1=hl([(1785, 1831)], NS6, 1812, 1816, 1817, 1818, 1819, 1820),
                M2=hl([(1785, 1831)], NS6, 1810, 1816, 1820),
                M3=hl([(1785, 1831)], NS6, 1821, 1822, 1823),
                M4=hl([(1785, 1831)], NS6, 1827),
                M5=hl([(1785, 1831)], NS6, 1824, 1825),
                M6=hl([(1785, 1831)], NS6, 1816, 1817, 1818))
)

# ------------------------------------------------------------------ 第 7 幕

V7 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">ggml_compute_forward(op)</span><span class="arrow">-></span>
    <span class="chip a">extra_compute_forward</span><span class="arrow">-></span>
    <span class="chip b">extra_buffer_type</span><span class="arrow">-></span>
    <span class="chip c">tensor_traits</span><span class="arrow">-></span>
    <span class="chip d">gemv / gemm</span>
  </div>
  <div class="row" id="cards" style="gap:12px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', t: 'class extra_buffer_type —— 认领层',
    m: 'supports_op(dev, op)<br>get_tensor_traits(op)',
    b: '注册在 buffer type 的 context 上；<br>回答"这个 op 我认领吗"。' },
  { c: 'c', t: 'class tensor_traits —— 执行层',
    m: 'work_size(n_threads, op, size)<br>compute_forward(params, op)',
    b: '注册在 tensor-&gt;extra 上；<br>回答"要多少 wdata、我来算"。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:322px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');
const msg = wrap.querySelector('#msg');
const texts = [
  '默认路径是 <span class="v">ggml_compute_forward_mul_mat</span>：逐行按 <span class="v">nb[]</span> 扫普通块。',
  'repack 想插队就得有钩子：<span class="k">traits.h 定义两个纯虚基类</span>。',
  '<span class="v">extra_buffer_type</span> 是认领层：<span class="v">get_tensor_traits</span> 直接返回 <span class="v">op-&gt;src[0]-&gt;extra</span>，也就是建张量时就选好的变体指针（repack.cpp:5225-5232）。',
  '<span class="v">tensor_traits</span> 是执行层：<span class="v">work_size</span> 报激活量化的临时区大小，<span class="v">compute_forward</span> 才是真正的 MUL_MAT 实现。',
  '调用点 <span class="v">ggml-cpu.c:1752</span>：<span class="k">谁认领成功就 return true</span>，默认实现根本不会执行。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; els[0].style.opacity = '1'; U.markLines(document, @@M0@@); });
tl.at(4000, () => { msg.innerHTML = texts[1]; els[0].style.opacity = '1'; els[1].style.opacity = '.32'; U.markLines(document, @@M1@@); });
tl.at(7400, () => { msg.innerHTML = texts[2]; els[0].style.opacity = '1'; els[1].style.opacity = '.32'; U.markLines(document, @@M2@@); });
tl.at(10800, () => { msg.innerHTML = texts[3]; els[0].style.opacity = '.32'; els[1].style.opacity = '1'; U.markLines(document, @@M3@@); });
tl.at(14300, () => { msg.innerHTML = texts[4]; els[0].style.opacity = '1'; els[1].style.opacity = '1'; U.markLines(document, @@M4@@); });
'''

L.scene(
    kicker='L5-04 · traits',
    title='<span class="hl-d">traits</span>：把 op 从默认路径接过来的钩子',
    sub='tensor_traits 只回答两件事：要多大 work 空间、这个 op 我来算；extra_buffer_type 负责认领。',
    caption='调用点在 ggml-cpu.c:1752（L5-01 覆盖）：每个 op 进来先被 extra buffer types 遍历一遍。',
    src=TH, parts=[(18, 33)], duration=18000,
    mark_src=[19, 20, 23, 24, 27, 30, 31],
    visual=fill(V7,
                M0=hl([(18, 33)], {}, 19, 20), M1=hl([(18, 33)], {}, 19, 20, 27),
                M2=hl([(18, 33)], {}, 27, 30, 31), M3=hl([(18, 33)], {}, 19, 20, 23, 24),
                M4=hl([(18, 33)], {}, 19, 20, 23, 24, 27, 30, 31))
)

# ------------------------------------------------------------------ 第 8 幕

V8 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(['选中的变体（触发 ISA）', 'NB_COLS x INTER_SIZE', '形状门槛'],
  [['q4_0_8x8_q8_0 —— AVX2 / SVE+i8mm', '8 x 8', 'ne[1] % 8 == 0'],
   ['q4_0_4x8_q8_0 —— NEON + i8mm', '4 x 8', 'ne[1] % 4 == 0'],
   ['q4_0_4x4_q8_0 —— NEON + dotprod', '4 x 4', 'ne[1] % 4 == 0'],
   ['q4_0_4x4_q8_0 —— VXE (s390x)', '4 x 4', 'ne[1] % 4 == 0'],
   ['q4_0_16x1_q8_0 —— RVV + zvfh, VLEN=256', '16 x 1', 'ne[1] % 16 == 0'],
   ['return nullptr —— 以上都不满足', '—', '留在普通缓冲，走默认 MUL_MAT']],
   { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '同一个 <span class="v">GGML_TYPE_Q4_0</span>，在这台机器上用哪个变体？<span class="k">建张量时就问一次</span>。',
  '<span class="v">ggml_cpu_has_avx2()</span>：256-bit 整数乘加 —— 选 <span class="v">8x8</span>（8 行交错）。',
  '<span class="v">ggml_cpu_has_neon() &amp;&amp; ggml_cpu_has_matmul_int8()</span>：ARM 的 i8mm —— 选 <span class="v">4x8</span>。',
  '<span class="v">ggml_cpu_has_neon() &amp;&amp; ggml_cpu_has_dotprod()</span>：ARM 的 sdot —— 选 <span class="v">4x4</span>。',
  'VXE（s390x）也走 4x4；RVV 分支看 <span class="v">__riscv_vlenb() * 8</span>，<span class="k">只有 256 位这一档实现了</span>（128 / 512 / 1024 还是 TODO）。',
  '每个分支都还有形状门槛：<span class="v">ne[1] % N == 0</span>。不满足就往下走，最后 <span class="v">return nullptr</span>。'
];
function on(i) { rows.forEach((r, k) => { r.className = k === i ? 'on' : ''; }); }
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, @@M0@@); });
tl.at(4000, () => { msg.innerHTML = texts[1]; on(0); U.markLines(document, @@M1@@); });
tl.at(7400, () => { msg.innerHTML = texts[2]; on(1); U.markLines(document, @@M2@@); });
tl.at(10800, () => { msg.innerHTML = texts[3]; on(2); on(3); U.markLines(document, @@M3@@); });
tl.at(14800, () => { msg.innerHTML = texts[4]; on(4); U.markLines(document, @@M4@@); });
tl.at(19000, () => { msg.innerHTML = texts[5]; on(5); U.markLines(document, @@M5@@); });
'''

NS8 = {4975: 'SVE 分支还要求向量长度正好等于 QK8_0（32 字节）—— 内核粒度要和寄存器宽度对齐'}

L.scene(
    kicker='L5-04 · 选择',
    title='变体选择：<span class="hl-b">ISA + 形状</span> 决定 4x4 / 4x8 / 8x8 / 16x1',
    sub='AVX2/SVE -> 8x8，NEON+i8mm -> 4x8，NEON+dotprod -> 4x4，VXE -> 4x4，RVV 256-bit -> 16x1；都不满足返回 nullptr。',
    caption='实例声明在同函数上方 repack.cpp:4927-4935 与 4966-4972：tensor_traits<block_q4_0, INTER_SIZE, NB_COLS, GGML_TYPE_Q8_0>，'
            '名字 = <NB_COLS>x<INTER_SIZE>。其余量化类型的分支从 5006 行起。',
    src=RC, parts=[(4973, 5005)], duration=22000,
    mark_src=[4973, 4975, 4977, 4980, 4982, 4985, 4987, 4990, 4992, 4995, 4999],
    notes_src=NS8,
    visual=fill(V8,
                M0=hl([(4973, 5005)], NS8, 4973),
                M1=hl([(4973, 5005)], NS8, 4975, 4977),
                M2=hl([(4973, 5005)], NS8, 4980, 4982),
                M3=hl([(4973, 5005)], NS8, 4985, 4987),
                M4=hl([(4973, 5005)], NS8, 4990, 4992, 4995, 4999),
                M5=hl([(4973, 5005)], NS8, 4973))
)

# ------------------------------------------------------------------ 第 9 幕

V9 = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(['量化类型', '选中的变体', '依据（repack.cpp）'],
  [['Q4_0', '8x8（AVX2 / SVE）· 4x8（NEON+i8mm）· 4x4（NEON+dotprod / VXE）· 16x1（RVV）', '4973-5005'],
   ['Q4_K', '8x8（AVX2 或 NEON+i8mm）· 8x4（NEON+dotprod）· 16x1（RVV）', '5006-5032'],
   ['Q2_K', '8x8（AVX512）· 16x1（RVV）', '5033-5049'],
   ['Q5_K / Q6_K', '8x8 或 8x4（只有 NEON 两条路）', '5050-5071'],
   ['IQ4_NL', '8x8（AVX2）· 4x4（NEON+dotprod）· 16x1（RVV）', '5072-5093'],
   ['MXFP4', '8x8（AVX2）· 4x4（NEON+dotprod）', '5094-5104'],
   ['Q8_0 / Q1_0', '4x8（NEON+i8mm）· 4x4（NEON+dotprod）· 16x1（仅 Q8_0，RVV）', '5105-5139']],
  { monoCols: [0, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '一个 Q4_0 权重张量，<span class="mono">ne = [4096, 8]</span>（8 行、每行 4096 个权重），跑在 AVX2 机器上。' +
  'repack 会选哪个变体？如果 <span class="mono">ne[1] = 7</span> 呢？',
  '选 <span class="mono">q4_0_8x8_q8_0</span>：AVX2 分支要求 <span class="mono">ne[1] % 8 == 0</span>，而 8 % 8 == 0，' +
  '于是走 <span class="mono">tensor_traits&lt;block_q4_0, 8, 8, GGML_TYPE_Q8_0&gt;</span>（NB_COLS = 8、INTER_SIZE = 8）。<br>' +
  '若 <span class="mono">ne[1] = 7</span>：形状门槛不成立（7 % 8 != 0），函数一路走到最后' +
  '<span class="mono">return nullptr</span>（repack.cpp:5140）—— 这个张量拿不到 repack 变体，' +
  'supports_op 也不会认领它（5197），于是走默认的 MUL_MAT。<br>' +
  '注意重排不改字节数：<span class="mono">block_q4_0x8 = 8 x 18 = 144</span> 字节。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '把这一课压成一张表：<span class="k">类型号决定"能不能重排"，ISA 与形状决定"用哪个变体"</span>。',
  '<span class="v">ggml_cpu_extra_compute_forward</span> 遍历所有 extra buffer types —— 这是唯一的插队入口。',
  '<span class="v">get_tensor_traits(op)</span> 返回建张量时就选好的 <span class="v">tensor_traits</span> 指针。',
  '<span class="v">compute_forward</span> 返回 <span class="v">true</span> 表示"这个 op 我算完了"，默认实现不再执行。',
  '一句话：<span class="k">权重重排一次、激活重排每次；交错让"行"变成向量载入的一个维度。</span>'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, @@M0@@); });
rows.forEach((r, i) => tl.at(3000 + i * 2100, () => {
  rows.forEach((x, k) => { x.className = k === i ? 'on' : ''; });
  msg.innerHTML = texts[1];
  U.markLines(document, @@M0@@);
}));
tl.at(15000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[2]; U.markLines(document, @@M1@@); });
tl.at(17500, () => { msg.innerHTML = texts[3]; U.markLines(document, @@M1@@); });
tl.at(19800, () => { msg.innerHTML = texts[4]; U.markLines(document, @@M2@@); });
'''

L.scene(
    kicker='L5-04 · 收束',
    title='谁先认领这个 op —— 与"变体 -> SIMD 宽度"总表',
    sub='每个 op 进入 CPU 后端，先遍历 extra buffer types；认领成功就走 repack 内核，否则回到默认实现。',
    caption='下一课 L5-05：这些变体在各架构上的真实实现（AVX2 / NEON-i8mm / RVV / AMX / KleidiAI）。',
    src=TC, parts=[(12, 23)], duration=22000,
    mark_src=[12, 13, 15, 16, 17, 18, 22],
    visual=fill(V9, M0=hl([(12, 23)], {}, 12, 13, 15, 16),
                M1=hl([(12, 23)], {}, 17, 18), M2=hl([(12, 23)], {}, 22))
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、CPU 侧的量化 ABI：quants.h',
    '这个头文件是 CPU 后端内部的 C 接口：一族 `quantize_row_*`（float -> 量化块）与'
    '一族 `ggml_vec_dot_*`（量化块 x 量化块 -> float）。注意 `quantize_row_q8_0` /'
    '`quantize_row_q8_K`：它们就是 `vec_dot_type` 指定的**激活侧类型**，'
    'repack 路径在没有变体可用时会退回到它们（见第五节）。',
    src=QH, parts=[(8, 22)], lang='c')

L.section(
    '二、基线：原始块上的标量点积',
    '先看"没有 repack"时长什么样。`ggml_vec_dot_q4_0_q8_0_generic` 把 `vx` 直接当 '
    '`block_q4_0[]`、`vy` 直接当 `block_q8_0[]`：\n\n'
    '- `x[ib].qs[j]` 的**低半**是元素 `j`，**高半**是元素 `j+16`（一块 32 个权重、16 字节）；\n'
    '- 取出来要 `& 0x0F` / `>> 4`，再**减 8** 才是有符号值；\n'
    '- 高半配的是 `y[ib].qs[j + qk/2]` —— y 里相隔 16 的那个 int8。\n\n'
    '也就是说：**原始布局下，一个字节里塞了两个相隔 16 的元素，载入之后必须现场拆**。'
    '这正是 repack 要消掉的成本。',
    src=QC, parts=[(225, 259)], lang='c')

L.section(
    '三、★ repack 的块布局：block<K,N>',
    '`repack.h` 用一个模板描述所有交错块：`ggml_half d[N]` 放 N 行的 scale，'
    '`int8_t qs[...]` 放 N 行交错后的量化数据。`N` 就是"交错进同一块的行数"，'
    '别名 `block_q4_0x4 / x8 / x16` 分别对应 N = 4 / 8 / 16；`K` 通过 `QK_0<K>()` '
    '选量化家族（1 -> `QK1_0`、4 -> `QK4_0`、8 -> `QK8_0`）。\n\n'
    '关键是那 7 条 `static_assert`：**字节总数和原始布局完全相等** —— '
    '`sizeof(block<4,4>) == 4 * sizeof(ggml_half) + QK8_0 * 2` 就是 `4 x 18 = 72` 字节。'
    'repack 不省空间、不改类型号，只改排列。',
    src=RH, parts=[(26, 46)], lang='c')

L.section(
    '四、重排实现：4 行轮流搬 + 异或偏置',
    '`make_block_q4_0x4` 是"交错"这个词的全部实现，三个下标就把事情说完了：\n\n'
    '```text\n'
    'src_id     = i % 4                   从哪里来：第 i%4 行\n'
    'src_offset = (i / 4) * interleave    该行的第几个块\n'
    'dst_offset = i * interleave          放到输出的哪里\n'
    '```\n\n'
    '另一处容易被忽略的细节是 `elems ^= xor_mask`（0x88 / 0x88888888...）：'
    'Q4_0 的 `qs` 存的是无符号 4-bit 码，异或之后变成 4-bit 补码，'
    '于是内核里 `<< 4` / `& 0xF0` 得到的就是"值 - 8"，**减 8 这一步被提前做掉了**。',
    src=RC, parts=[(3083, 3122)], lang='c')

L.section(
    '五、激活侧契约：quantize_mat_q8_0_4x4',
    '权重只重排一次，激活每次前向都要重排。`ggml_quantize_mat_q8_0_4x4_generic` '
    '就是激活侧的对应物：它一次吃 4 行（4 个 token），每行各算各的 `amax` 与 `d`，'
    '然后按**同一个** `blck_size_interleave = 4` 把 4 行的 int8 交错写进 '
    '`block_q8_0x4`。\n\n'
    '所以交错是**双边契约**：两侧粒度必须一致，内核才能用同一条乘加指令同时消费'
    '4 行权重与 4 个 token。`forward_mul_mat` 里只有 `ne11` 是 4 的倍数时才走这条路径，'
    '余下的 1..3 行退回 `from_float`（即 `quantize_row_q8_0`）交给 gemv。',
    src=RC, parts=[(135, 171)], lang='c')

L.section(
    '六、★ 内核怎么读交错块：4x4 tile',
    '`ggml_gemm_q4_0_4x4_q8_0_generic` 是"为什么能加速"的答案：\n\n'
    '- `a_ptr` 是 `block_q8_0x4`（4 个 token 交错），`b_ptr` 是 `block_q4_0x4`（4 行权重交错）；\n'
    '- 权重下标 `k * 4 * 4 + j * 4 + i`：`j` 每加 1 只走 **4 字节** —— 因为 4 行是交错的；\n'
    '- 于是**一条载入就覆盖 4 行的同一批列**，而不是一行的 16 字节；\n'
    '- 一个 `(m, j)` 做完 `blocklen = 4` 次乘加，`sumf[4][4]` 一次出 **16 个部分和**；\n'
    '- `>> 4` 是收尾：`v0` 被 `<< 4` 放大过，两个乘积一起右移还原。\n\n'
    '这也是各架构内核（AVX2 / NEON-i8mm / RVV）能共用同一套布局的原因：'
    '布局把"行"变成了向量载入的一个维度。',
    src=RC, parts=[(1785, 1839)], lang='c')

L.section(
    '七、钩子协议：traits.h',
    'CPU 后端的默认实现是 `ggml_compute_forward_mul_mat`。要让 repack 内核接手，'
    '必须有钩子：`extra_buffer_type` 负责**认领**（`supports_op` / `get_tensor_traits`），'
    '`tensor_traits` 负责**执行**（`work_size` / `compute_forward`）。'
    '源码注释直接写明 `tensor_traits` 注册在 `tensor->extra` 里。',
    src=TH, parts=[(18, 33)], lang='c')

L.section(
    '八、变体选择：ISA + 形状',
    '`ggml_repack_get_optimal_repack_type` 是"选 kernel 变体"的唯一入口，'
    '对每个类型号按 `if (ISA) { if (形状门槛) return &变体; }` 的顺序试。'
    '以 Q4_0 为例（下面的逐字引用）：AVX2 或 SVE+i8mm 且 `ne[1] % 8 == 0` -> `q4_0_8x8_q8_0`；'
    'NEON+i8mm 且 `ne[1] % 4 == 0` -> `q4_0_4x8_q8_0`；NEON+dotprod -> `q4_0_4x4_q8_0`；'
    'VXE -> `q4_0_4x4_q8_0`；RVV（`__riscv_vlenb() * 8 == 256`）-> `q4_0_16x1_q8_0`。'
    '都不满足则 `return nullptr`（5140），张量就留在普通缓冲里。',
    src=RC, parts=[(4973, 5005)], lang='c')

L.section(
    '九、认领顺序：traits.cpp',
    '最后一块拼图：`ggml_cpu_extra_compute_forward` 遍历所有注册的 extra buffer types，'
    '拿到 `tensor_traits` 后调用 `compute_forward`；**谁先返回 true，这个 op 就归谁**，'
    '默认实现不再执行。`ggml_cpu_extra_work_size` 是同一个模式的"问大小"版本 —— '
    '图的 wdata 预算就是这么算出来的。',
    src=TC, parts=[(12, 23)], lang='c')

L.footnote_add('本课只引用 `ggml/src/ggml-cpu/` 下的 6 个文件（quants.c / quants.h / repack.cpp / '
               'repack.h / traits.cpp / traits.h），全部计入覆盖率。')
L.footnote_add('文中为了定位而提到的 `ggml-common.h`（L1-04）、`ggml-cpu.c`（L5-01）、'
               '`arch/x86/repack.cpp` 与 `arch-fallback.h`（L5-05）、`src/llama-model.cpp`（L2 系列）、'
               '`common/arg.cpp` 均**不引用其源码**，故不计入本课覆盖率。')
L.footnote_add('`QK4_0 = 32`、`QK8_0 = 32` 的定义在 `ggml-common.h:194,251`；本课引用的 '
               '`repack.cpp:136` 有 `assert(QK8_0 == 32)` 可交叉验证。')
L.footnote_add('第 3 幕的字节条按 `repack.h:26-46` 的结构体字段与 `static_assert` 画出：'
               '`d[4]` = 4 x 2 字节，`qs` = `QK8_0 * 2` = 64 字节，合计 72 = 4 x 18。')

L.prereqs('`L5-03`（★ 向量化基础设施）')

L.goal(
    '说清 repack 与 `GGML_TYPE` 的分工：类型号管"块多大、怎么反量化"，后端管"块与块之间怎么摆"（对应验收点）；',
    '解释交错布局为什么能加速点积：偏移 `k·N·G + j·G + i` 让"行"变成向量载入的一个维度；',
    '说出 `block_q4_0x4` 与 `block_q4_0` 的字节数关系（72 = 4 x 18），以及 `xor 0x88` 在重排时顺手做掉了什么；',
    '说明 traits 的两层钩子（`extra_buffer_type` 认领 / `tensor_traits` 执行）在一个 op 上如何先后生效；',
    '按 ISA 与 `ne[1]` 门槛，判断一个 Q4_0 张量会选中 8x8 / 4x8 / 4x4 / 16x1 中的哪一个。')

L.conclusion(
    '★ repack = 把交错布局从类型号里搬进后端',
    '`block<K,N>` 是编译期模板，`block_q4_0x4 / x8 / x16` 只是 `N = 4 / 8 / 16` 的别名。'
    '7 条 `static_assert` 钉死一件事：**重排前后字节总数完全相等**（`4 x 18 = 72`）。\n\n'
    '这与 L1-01 里提过的旧式 `GGML_TYPE_Q4_0_4_4` 形成对照：交错曾经是类型号的一部分'
    '（`ggml.h:421` 注明 support has been removed），现在它是后端缓冲类型的属性 —— '
    '类型号保持"纯数据格式"，性能布局交给后端。')

L.conclusion(
    '★ 为什么交错能加速点积',
    '原布局里一个字节管相隔 16 的两个元素，载入后必须 `& 0x0F` / `>> 4` / `- 8`；'
    '交错布局把同一批列在 N 行上的数据排成连续地址：\n\n'
    '```text\n'
    '权重的第 j 行第 i 个字节：偏移 = k * N * G + j * G + i\n'
    '（k 走列、j 走行、G = 交错粒度；4x4 变体里 N = G = 4）\n'
    '```\n\n'
    '于是**一条向量载入同时覆盖多行的同一批列**，`j` 每加 1 只走 G 字节；'
    '4x4 的 gemm 一次算出 `sumf[4][4]` 共 16 个部分和。重排时顺手 `^= 0x88`，'
    '把"减 8"也提前做掉了，内核只需 `<< 4` / `& 0xF0` 与一次 `>> 4` 收尾。')

L.conclusion(
    '交错是双边契约，粒度必须一致',
    '权重在**加载时**重排一次（repack 缓冲的 set_tensor），激活在**每次前向**重排'
    '（`ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>`，4 行一组写进 `params->wdata`）。'
    '只有两侧的 `blck_size_interleave` 相同，gemm 才能用同一条乘加同时消费 4 行权重与 '
    '4 个 token；`ne11` 不是 4 的倍数时，余下的行退回 `from_float` + gemv。')

L.conclusion(
    '变体选择 = 类型号 + ISA + 形状',
    '`ggml_repack_get_optimal_repack_type` 对每个类型号按顺序试分支，'
    '`return` 出去的那个实例就是本次运行的 kernel 变体；都不满足则 `return nullptr`，'
    '张量留在普通缓冲走默认 `MUL_MAT`。\n\n'
    '| 类型 | 变体（触发 ISA） | 依据行 |\n|---|---|---|\n'
    '| Q4_0 | 8x8（AVX2/SVE+i8mm）· 4x8（NEON+i8mm）· 4x4（NEON+dotprod / VXE）· 16x1（RVV 256）| 4973-5005 |\n'
    '| Q4_K | 8x8（AVX2 或 NEON+i8mm）· 8x4（NEON+dotprod）· 16x1（RVV）| 5006-5032 |\n'
    '| Q2_K | 8x8（AVX512）· 16x1（RVV）| 5033-5049 |\n'
    '| Q5_K / Q6_K | 8x8 或 8x4（只有 NEON 两条路）| 5050-5071 |\n'
    '| IQ4_NL / MXFP4 | 8x8（AVX2）· 4x4（NEON+dotprod）· 16x1（仅 IQ4_NL，RVV）| 5072-5104 |\n'
    '| Q8_0 / Q1_0 | 4x8（NEON+i8mm）· 4x4（NEON+dotprod）· 16x1（仅 Q8_0，RVV）| 5105-5139 |\n\n'
    '如果 ISA 与形状都不满足，第 8 幕那句 `return nullptr` 就是答案 —— '
    '**repack 是可选项，不是必需项**。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
