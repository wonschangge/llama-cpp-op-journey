#!/usr/bin/env python3
"""L1-04 · 量化块结构：算子内层的压缩数据 —— 课件 spec。

运行：python3 L1-operator-representation/L1-04-quant-block-layout/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

COMMON = 'ggml/src/ggml-common.h'
QH = 'ggml/src/ggml-quants.h'
QC = 'ggml/src/ggml-quants.c'
GC = 'ggml/src/ggml.c'

L = Lesson(
    id='L1-04',
    layer='L1 · 算子的表示',
    title='量化块结构：算子内层的压缩数据',
    codecap='ggml-common.h / ggml-quants.{h,c}（逐字引用）',
    nav={'prev': {'href': '../L1-03-graph-and-toposort/index.html',
                  'label': 'L1-03 计算图与拓扑排序'},
         'next': {'href': '../L1-05-context-threads-opt/index.html',
                  'label': 'L1-05 上下文、线程与优化器'}},
)

L.note('**一句话**：在 llama.cpp 里，量化权重**不是**"每个数少几位"，而是'
       '**把元素切成定长的块，每块自带 scale**。`ggml_tensor` 的字节被解释成 '
       '`block_q4_0[]` / `block_q4_K[]` 这样的记录数组，'
       '块的大小（多少元素一块、一块多少字节）由类型号唯一决定。')
L.note('这一课只讲**布局**：块里有哪些字段、每个字段多少字节、反量化函数怎么把字节还原成 '
       'float。至于"这个布局怎么被内核消费"（量化点积、SIMD、repack），属于 L5-03 / L5-04。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L1-04 · 全局',
    title='量化权重是一串<span class="hl-a">定长块</span>，不是 float 数组',
    sub='一个块覆盖固定个数的元素并自带 scale；张量的字节数 = 块数 × 每块字节数。',
    caption='回顾 L1-01：nb[0] = ggml_type_size(type)。这里说的 type_size 就是"一块多少字节"。',
    src=COMMON, parts=[(86, 90)], duration=15000,
    marks=[0, 1, 3, 4],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">模型里的权重字节</span><span class="arrow">-></span>
    <span class="chip c">block_q4_0[]</span><span class="arrow">-></span>
    <span class="chip b">dequantize_row_q4_0()</span><span class="arrow">-></span>
    <span class="chip d">量化点积内核 · L5-04</span>
  </div>
  <div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '块 = 定长记录', b: '一个块覆盖固定个数的元素并自带 scale，块内字节数固定。', m: 'struct block_q4_0' },
  { c: 'c', t: '块大小由宏给出', b: 'QK4_0 = 32、QK_K = 256、K_SCALE_SIZE = 12。', m: 'ggml-common.h:89-90' },
  { c: 'b', t: '两个消费者', b: '先反量化成 float，或直接做量化点积（L5-03 / L5-04）。', m: 'dequantize_row_*()' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:216px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });

const msg = wrap.querySelector('#msg');
const texts = [
  '模型权重不是 float 数组：<span class="k">它是一串定长的块</span>。',
  '<span class="v">QK4_0 = 32</span>：32 个元素一块；<span class="v">QK_K = 256</span>：K-quant 是 256 个元素一个超块。<br>两个数量级的块共存于同一套张量布局里。',
  '<span class="v">K_SCALE_SIZE = 12</span>：Q4_K 里 8 个 scale 加 8 个 min 一共只占 12 字节。<br>第 6、7 幕会把这个数字拆开看。',
  '回顾 L1-01：<span class="v">nb[0] = ggml_type_size(type)</span>。<br>换类型就换块大小、换块字节数，也就换了内核。',
  '本课路线：块长什么样（2、6 幕）-> 怎么写/怎么读（3-5 幕）-> 6-bit 解包（7 幕）-> 绑定（8 幕）。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(9800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
tl.at(12800, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L1-04 · 小块',
    title='★ <span class="hl-a">18 字节</span> = 2 字节 scale + 16 字节 4-bit',
    sub='QK4_0 = 32：32 个 4-bit 正好 16 字节；再加一个 fp16 的 d，就是 18 字节。',
    caption='static_assert 把这条等式钉在编译期：sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0/2，'
            '而 ggml_half 就是 uint16_t（2 字节）。',
    src=COMMON, parts=[(194, 212)], duration=17000,
    marks=[0, 2, 3, 6, 8, 12, 13, 17, 19],
    notes={3: 'QK4_0 / 2 = 16 字节 —— 32 个 4-bit 权重正好塞满 16 字节，这就是块大小取 32 的原因'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div id="s0"></div>
  <div id="s1"></div>
  <div id="tbl"></div>
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
const s1 = wrap.querySelector('#s1');
const b0 = strip(s0, 'block_q4_0 —— 18 字节 / 32 个权重',
  [{ n: 2,  t: 'd',      c: 'a', bg: 'rgba(88,166,255,.30)' },
   { n: 16, t: 'qs[16] —— 32 个 4-bit nibble', c: 'b', bg: 'rgba(63,185,80,.16)' }]);
const b1 = strip(s1, 'block_q4_1 —— 20 字节 / 32 个权重',
  [{ n: 2,  t: 'd',      c: 'a', bg: 'rgba(88,166,255,.30)' },
   { n: 2,  t: 'm',      c: 'e', bg: 'rgba(247,120,186,.26)' },
   { n: 16, t: 'qs[16] —— 32 个 4-bit nibble', c: 'b', bg: 'rgba(63,185,80,.16)' }]);

const t = U.table(['类型', '块内字段', '字节 / 块', 'bit / 权重'],
  [['Q4_0', 'd(2) + qs(16)', '18', '4.5'],
   ['Q4_1', 'd(2) + m(2) + qs(16)', '20', '5.0']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  'QK4_0 = 32：<span class="k">32 个 4-bit 权重正好 16 字节</span>。块大小不是随手取的，它由"字节数要整齐"反推出来。',
  '再加一个 fp16 的 <span class="v">d</span>（2 字节）：<span class="v">sizeof(block_q4_0) = 2 + 16 = 18</span>。<br>18 × 8 / 32 = <span class="k">4.5 bit / 权重</span>。',
  'Q4_1 多一个 <span class="v">m</span>（min）：<span class="v">2 + 2 + 16 = 20 字节</span>，即 <span class="k">5.0 bit / 权重</span>。<br>多出来的 2 字节买的是"非对称量化"（第 5 幕对照）。',
  '<span class="v">static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, ...)</span>：<br>改了布局却忘了改断言，<span class="k">编译就过不去</span>。这与 L1-01 的 nb[] 约定是同一件事的两种写法。'
];
tl.at(600, () => { U.markLines(document, [0, 2, 3]); msg.innerHTML = texts[0]; });
tl.at(4200, () => { U.markLines(document, [6]); rows.forEach((r, k) => { r.className = k === 0 ? 'on' : ''; }); msg.innerHTML = texts[1]; });
tl.at(8400, () => { U.markLines(document, [8, 12, 13, 17]); rows.forEach((r, k) => { r.className = k === 1 ? 'on' : ''; }); msg.innerHTML = texts[2]; });
tl.at(13000, () => { U.markLines(document, [19]); rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L1-04 · 函数族',
    title='一个类型一个函数：<span class="hl-c">名字里的类型号就是 GGML_TYPE</span>',
    sub='ggml-quants.h 把反量化函数按同一套命名排开：dequantize_row_ 加类型名，参数就是对应的 block_ 结构体。',
    caption='同一个头文件的上半部分（17-34 行）用同样的规则排开量化函数 quantize_row_*_ref —— 见 source.md 第五节。',
    src=QH, parts=[(45, 63)], duration=16000,
    marks=[0, 3, 15, 17, 18],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">GGML_TYPE_Q4_0</span><span class="arrow">-></span>
    <span class="chip c">dequantize_row_q4_0()</span><span class="arrow">-></span>
    <span class="chip b">traits.to_float（第 8 幕）</span>
  </div>
  <div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'c', t: '反量化 · 15 个声明', b: '从 q1_0 一路排到 q8_K；每个都输出 float。第 53 行那个 q8_1 被注释掉了。', m: 'dequantize_row_q4_K()' },
  { c: 'a', t: '参数就是块类型', b: '入参是 block_<类型名> 指针，所以"块布局"在这里被写死。', m: 'const block_q4_0 * x' },
  { c: 'b', t: '声明与实现分开', b: '声明在 .h（本幕），实现在 ggml-quants.c（4-5、7 幕）。', m: 'GGML_API void' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:216px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });

const msg = wrap.querySelector('#msg');
const texts = [
  '先认名字：<span class="v">dequantize_row_</span> + <span class="k">类型名的小写形式</span>。',
  '一共 15 个反量化声明（46-63 行）：<br>小块 Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0，K-quant Q2_K..Q6_K/Q8_K，还有 mxfp4/nvfp4。',
  '<span class="k">每个函数的参数类型就是它对应的块结构体</span>：<br><span class="v">dequantize_row_q4_0(const block_q4_0 * x, ...)</span> —— 布局不是约定，是签名。',
  '<span class="v">dequantize_row_q4_K</span> 与 <span class="v">dequantize_row_q6_K</span> 也在这一串里：<br>第 6、7 幕看 Q4_K，它比 Q4_0 多一层"超块"。',
  '<span class="v">GGML_API</span>：这些函数被 CPU 后端直接链接（注释写明 "because they used by the CPU backend"），<br>再由 traits 表挂到类型号上 —— 第 8 幕。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
tl.at(3600, () => { els.forEach((e, k) => { e.style.opacity = k === 0 ? '1' : '.30'; }); msg.innerHTML = texts[1]; U.markLines(document, [3]); });
tl.at(7200, () => { els.forEach((e, k) => { e.style.opacity = k === 1 ? '1' : '.30'; }); msg.innerHTML = texts[2]; U.markLines(document, [3, 15]); });
tl.at(10800, () => { els.forEach((e, k) => { e.style.opacity = k === 2 ? '1' : '.30'; }); msg.innerHTML = texts[3]; U.markLines(document, [15, 17, 18]); });
tl.at(14000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [0, 3, 15, 17, 18]); });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L1-04 · 反量化',
    title='dequantize_row_q4_0：<span class="hl-b">一个字节管相隔 16 的两个元素</span>',
    sub='低 4 位给 y[j]，高 4 位给 y[j + qk/2]，也就是 y[j+16] —— 不是相邻两个。',
    caption='这个"前后半块交错"的排布，正是 L5-04 的 repack 要把 q4_0 重排成 4x4 交错布局的原因。',
    src=QC, parts=[(459, 477)], duration=17000,
    marks=[8, 12, 13, 15, 16],
    notes={8: 'd 从 fp16 解回 float —— "块自带 scale" 落到一行代码就是这一句'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `
  <div class="cm">16 个 qs 字节 → 32 个输出元素（一个 Q4_0 块）</div>
  <div class="row" id="rA" style="gap:3px"></div>
  <div class="row" id="rQ" style="gap:3px"></div>
  <div class="row" id="rB" style="gap:3px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

function cell(host, txt, color) {
  const e = U.el('div', { style: 'flex:1 1 0;min-width:0;height:19px;border:1px solid var(--border);border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:8.5px;overflow:hidden;color:var(--' + color + ');background:#10151b' });
  e.textContent = txt;
  host.appendChild(e);
  return e;
}
const rA = wrap.querySelector('#rA'), rQ = wrap.querySelector('#rQ'), rB = wrap.querySelector('#rB');
const A = [], Q = [], B = [];
for (let j = 0; j < 16; j++) { A.push(cell(rA, 'y[' + j + ']', 'a')); }
for (let j = 0; j < 16; j++) { Q.push(cell(rQ, 'qs[' + j + ']', 'c')); }
for (let j = 0; j < 16; j++) { B.push(cell(rB, 'y[' + (j + 16) + ']', 'b')); }

function focus(j) {
  A.forEach((e, k) => { e.style.borderColor = (k === j) ? 'var(--a)' : 'var(--border)'; e.style.background = (k === j) ? 'rgba(88,166,255,.20)' : '#10151b'; });
  Q.forEach((e, k) => { e.style.borderColor = (k === j) ? 'var(--c)' : 'var(--border)'; e.style.background = (k === j) ? 'rgba(210,153,34,.20)' : '#10151b'; });
  B.forEach((e, k) => { e.style.borderColor = (k === j) ? 'var(--b)' : 'var(--border)'; e.style.background = (k === j) ? 'rgba(63,185,80,.20)' : '#10151b'; });
}

const msg = wrap.querySelector('#msg');
tl.at(600, () => { U.markLines(document, [8]); msg.innerHTML = '先把 <span class="v">d</span> 从 fp16 解回来：<span class="k">块自带的那一个 scale，是全块共用的</span>。'; });
tl.at(3600, () => {
  focus(0); U.markLines(document, [12, 15]);
  msg.innerHTML = '取字节 j 的<span class="k">低 4 位</span>：<span class="v">x0 = (qs[j] & 0x0F) - 8</span>，写到 <span class="v">y[j]</span>。';
});
tl.at(7200, () => {
  focus(0); U.markLines(document, [13, 16]);
  msg.innerHTML = '取同一个字节的<span class="k">高 4 位</span>：<span class="v">x1 = (qs[j] >> 4) - 8</span>，写到 <span class="v">y[j + qk/2]</span>。';
});
tl.at(10800, () => {
  focus(5); U.markLines(document, [12, 13, 15, 16]);
  msg.innerHTML = '换成 j = 5 也一样：<span class="v">qs[5]</span> 的低位给 <span class="v">y[5]</span>、高位给 <span class="v">y[21]</span>。<br><span class="k">一个字节跨越半块</span>，16 个字节刚好铺满 32 个元素。';
});
tl.at(14200, () => {
  focus(0); U.markLines(document, [8, 12, 13, 15, 16]);
  msg.innerHTML = '减去 8 是因为 q 落在 <span class="v">[0, 15]</span>：还原出的范围是 <span class="v">[-8d, 7d]</span>。<br>这就是 Q4_0 的<span class="k">对称</span>量化 —— 第 5 幕看它怎么写出来。';
});
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L1-04 · 量化侧',
    title='★ <span class="hl-a">d = max / -8</span>：Q4_0 对称，没有 min',
    sub='先扫一遍求绝对最大 amax（连同它的符号一起记进 max），再算 d 与 id，最后把 x*id 四舍五入加 8 塞进 nibble。',
    caption='Q4_1 走的是另一条路（非对称）：d = (max-min)/15 且多存一个 m —— 这解释了第 2 幕里它为什么是 20 字节。',
    src=QC, parts=[(113, 148)], duration=22000,
    marks=[8, 13, 19, 23, 29, 30, 32, 33],
    notes={19: 'd = max / -8：d 的符号与 max 相反，于是 (q - 8) * d 能把 q 还原到 [-8d, 7d]'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">amax / max</span><span class="arrow">-></span>
    <span class="chip c">d = max / -8</span><span class="arrow">-></span>
    <span class="chip b">id = 1/d</span><span class="arrow">-></span>
    <span class="chip d">(int8_t)(x*id + 8.5)</span><span class="arrow">-></span>
    <span class="chip e">MIN(15, .)</span>
  </div>
  <div class="cm" id="cap"></div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
wrap.querySelector('#cap').textContent =
  '算例：这一块里 amax = 1.0、max = -1.0（绝对值最大的是 -1.0），于是 d = 0.125、id = 8；源码用 (int8_t)(x*id + 8.5f) 做四舍五入。';

const t = U.table(['x', 'x*id', '+8.5', '(int8_t)', 'MIN(15,·)', '(q-8)*d'],
  [['-1.000', '-8.0', '0.5',  '0',  '0',  '-1.000'],
   ['0.000',  '0.0',  '8.5',  '8',  '8',  '0.000'],
   ['0.875',  '7.0',  '15.5', '15', '15', '0.875'],
   ['1.000',  '8.0',  '16.5', '16', '15', '0.875']],
  { monoCols: [0, 1, 2, 3, 4, 5] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '第一步扫出 <span class="v">amax</span> 与取到 amax 的那个<span class="k">带符号值 max</span>。',
  '<span class="v">d = max / -8</span>、<span class="v">id = 1/d</span>，写回 <span class="v">y[i].d</span>（fp16）。<br>注意：<span class="k">全程没有 min</span> —— Q4_0 是围绕 0 对称的。',
  '取整用的是 <span class="v">MIN(15, (int8_t)(x0 + 8.5f))</span>：<br>+8.5 把 [-8, 7] 平移到 [0, 15] 且四舍五入，MIN 兜住上溢。',
  '两个 nibble 合进一个字节：<span class="v">qs[j] = xi0; qs[j] |= xi1 &lt;&lt; 4;</span><br>低位是本块前半的 <span class="v">x[j]</span>，高位是后半的 <span class="v">x[j+16]</span> —— 正好对上第 4 幕的解包。',
  '看表的最后一行：x = 1.000 被 MIN 截到 15，还原成 0.875。<br><span class="k">对称量化的代价就藏在这里</span>：正方向的极值会被压一点。Q4_1 用 min 换掉了这个偏差，代价是 2 字节。'
];
tl.at(600, () => { U.markLines(document, [8, 13]); msg.innerHTML = texts[0]; rows.forEach(r => { r.className = ''; }); });
tl.at(4600, () => { U.markLines(document, [19, 23]); msg.innerHTML = texts[1]; });
tl.at(9000, () => { U.markLines(document, [29, 30]); rows.forEach((r, k) => { r.className = (k === 0 || k === 1) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
tl.at(13400, () => { U.markLines(document, [32, 33]); rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[3]; });
tl.at(17800, () => { U.markLines(document, [29, 30, 32, 33]); rows.forEach((r, k) => { r.className = (k === 3) ? 'on' : ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L1-04 · 超块',
    title='★ Q4_K：<span class="hl-a">144 字节</span>装 256 个权重，12 字节装 16 个 scale/min',
    sub='8 个 32 元素子块共用一个超块头；每个子块的 scale 与 min 都只用 6 bit，8 ×(6+6) = 96 bit 正好 12 字节。',
    caption='对比第 2 幕：Q4_0 的 18/32 与 Q4_K 的 144/256 都是 0.5625 字节/权重 —— 同样的码率，花法完全不同。',
    src=COMMON, parts=[(323, 338)], duration=22000,
    marks=[1, 3, 7, 8, 12, 14, 16],
    notes={12: 'K_SCALE_SIZE = 12 字节 = 96 bit：8 个 6-bit scale + 8 个 6-bit min，一位不多一位不少'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="cm">block_q4_K —— 144 字节 / 256 个权重（= 8 个子块 × 32）</div>
  <div id="strip" style="display:flex;height:30px;border:1px solid var(--border);border-radius:5px;overflow:hidden;width:100%"></div>
  <div class="row" id="subs" style="gap:4px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const segs = [
  { n: 4,   t: 'dm',          c: 'a', bg: 'rgba(88,166,255,.30)' },
  { n: 12,  t: 'scales[12]',  c: 'c', bg: 'rgba(210,153,34,.30)' },
  { n: 128, t: 'qs[128] —— 256 个 4-bit', c: 'b', bg: 'rgba(63,185,80,.16)' }
];
const strip = wrap.querySelector('#strip');
segs.forEach(g => {
  const e = U.el('div', { style: 'flex:' + g.n + ' 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:9px;overflow:hidden;white-space:nowrap;background:' + g.bg + ';color:var(--' + g.c + ');border-right:1px solid var(--border)' });
  e.textContent = g.t;
  strip.appendChild(e);
});
const subHost = wrap.querySelector('#subs');
const subs = [];
for (let j = 0; j < 8; j++) {
  const e = U.el('div', { style: 'flex:1 1 0;min-width:0;height:32px;border:1px solid var(--border);border-radius:4px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:8.5px;color:var(--muted);background:#10151b;overflow:hidden' });
  e.innerHTML = '<span style="color:var(--c)">子块 ' + j + '</span><span>6-bit sc + 6-bit min</span>';
  subHost.appendChild(e);
  subs.push(e);
}

const msg = wrap.querySelector('#msg');
const texts = [
  '超块头是 <span class="v">dm</span>：<span class="k">d 与 dmin 两个 fp16</span>，共 4 字节。它们是"scale 的 scale"。',
  '<span class="v">scales[12]</span>：8 个子块各自的 scale 和 min，各 6 bit。<br>8 × (6 + 6) = 96 bit = <span class="k">12 字节</span>，这就是 K_SCALE_SIZE。',
  '<span class="v">qs[128]</span>：256 个 4-bit 权重。<br>4 + 12 + 128 = <span class="v">144 字节</span>，与 static_assert 的 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 一致。',
  '子块与 Q4_0 的小块同宽（32 个元素），但<span class="k">不再各自带 fp16 scale</span>，<br>而是共享超块头 + 一个 6-bit 的量化 scale。',
  '算总账：144 × 8 / 256 = <span class="k">4.5 bit / 权重</span>，与 Q4_0 完全相同。<br>源码注释也写着 "Effectively 4.5 bits per weight"。'
];
tl.at(600, () => { U.markLines(document, [7, 8]); msg.innerHTML = texts[0]; subs.forEach(e => { e.style.opacity = '.40'; }); });
tl.at(4800, () => { U.markLines(document, [12]); msg.innerHTML = texts[1]; });
tl.at(9000, () => { U.markLines(document, [13, 14]); msg.innerHTML = texts[2]; });
tl.at(13200, () => { U.markLines(document, [1, 16]); subs.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
tl.at(17600, () => { U.markLines(document, [1, 3, 12, 13, 14, 16]); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L1-04 · 解包',
    title='6-bit 的 scale/min 怎么解回来：<span class="hl-c">低位在自己字节，高位从前 4 个字节借</span>',
    sub='bytes[0..3] 的低 6 位是 scale 0..3，bytes[4..7] 的低 6 位是 min 0..3；它们的高 2 位被借给 scale/min 4..7，低 4 位放在 bytes[8..11]。',
    caption='反量化的消费者是 dequantize_row_q4_K：它每 64 个元素调两次 get_scale_min_k4，见 source.md 第九节。',
    src=QC, parts=[(880, 887)], duration=20000,
    marks=[0, 1, 2, 5, 7],
    notes={2: 'j < 4：低 6 位直接就是第 j 个子块的 scale 与 min',
           4: 'j >= 4：低 4 位在本字节，高 2 位从前 4 个字节的最高两位左移上来'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="cm">scales[12]：12 个字节 = 96 bit，装下 8 个 scale + 8 个 min（各 6 bit）</div>
  <div id="bytes" style="display:flex;gap:3px"></div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#bytes');
const bytes = [];
for (let j = 0; j < 12; j++) {
  const e = U.el('div', { style: 'flex:1 1 0;min-width:0;height:26px;border:1px solid var(--border);border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:8.5px;background:#10151b;color:var(--dim);overflow:hidden' });
  e.textContent = 'q[' + j + ']';
  host.appendChild(e);
  bytes.push(e);
}
function tint(list, color, rgba) {
  list.forEach(j => { bytes[j].style.borderColor = 'var(--' + color + ')'; bytes[j].style.background = rgba; bytes[j].style.color = 'var(--' + color + ')'; });
}
function clear() {
  bytes.forEach(e => { e.style.borderColor = 'var(--border)'; e.style.background = '#10151b'; e.style.color = 'var(--dim)'; });
}

const t = U.table(['j', '*d（scale，6 bit）', '*m（min，6 bit）'],
  [['0..3', 'q[j] & 63', 'q[j+4] & 63'],
   ['4..7', '(q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)', '(q[j+4] >> 4) | ((q[j] >> 6) << 4)']],
  { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
tl.at(600, () => {
  clear(); tint([0, 1, 2, 3], 'a', 'rgba(88,166,255,.20)');
  U.markLines(document, [1, 2]); rows.forEach((r, k) => { r.className = k === 0 ? 'on' : ''; });
  msg.innerHTML = '先看 <span class="v">j &lt; 4</span>：<span class="k">q[j] 的低 6 位</span>就是第 j 个 scale，<span class="k">q[j+4] 的低 6 位</span>就是第 j 个 min。<br>只用了 8 个字节里最普通的 6 bit。';
});
tl.at(5000, () => {
  clear(); tint([4, 5, 6, 7], 'b', 'rgba(63,185,80,.20)');
  U.markLines(document, [5, 7]); rows.forEach((r, k) => { r.className = k === 1 ? 'on' : ''; });
  msg.innerHTML = '再看 <span class="v">j &gt;= 4</span>：第 j 个 scale 的低 4 位在 <span class="v">q[j+4]</span> 的低半字节，<br>高 2 位则是 <span class="v">q[j-4]</span> 的最高两位左移 4 位拼上来。';
});
tl.at(10000, () => {
  clear(); tint([8, 9, 10, 11], 'c', 'rgba(210,153,34,.22)');
  msg.innerHTML = '被"借走"高 2 位的就是 <span class="v">q[0..7]</span> —— 它们本来就只用 6 bit，剩下 2 bit 正好给 4..7 用。';
});
tl.at(14000, () => {
  clear(); tint([0, 1, 2, 3], 'a', 'rgba(88,166,255,.20)'); tint([4, 5, 6, 7], 'b', 'rgba(63,185,80,.20)'); tint([8, 9, 10, 11], 'c', 'rgba(210,153,34,.22)');
  U.markLines(document, [0, 1, 2, 5, 7]);
  msg.innerHTML = '8 个字节各出 6 bit + 4 个字节各出 4 bit × 2 = 96 bit = <span class="k">12 字节，正好装满</span>。<br>没有任何浪费，代价是<span class="k">取用时要移位拼接</span>。';
});
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L1-04 · 收束',
    title='类型号 = <span class="hl-a">块大小</span> + <span class="hl-b">反量化函数</span>',
    sub='ggml_type_traits 把三件事绑在一个类型号上：blck_size（多少元素一块）、type_size（一块多少字节）、to_float（怎么还原成 float）。',
    caption='下一课 L1-05 讲上下文、线程与优化器接口。',
    src=GC, parts=[(677, 700)], duration=21000,
    marks=[16, 19, 21, 24, 25],
    notes={16: 'GGML_TYPE_Q4_0 的登记项 —— 块的"大小"和"还原函数"都在这里绑定',
           18: 'blck_size = QK4_0 = 32：多少元素算一块',
           19: 'type_size = sizeof(block_q4_0) = 18：一块多少字节'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div id="s0"></div>
  <div id="s1"></div>
  <div id="tbl"></div>
  <div id="ex"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

function rowStrip(host, labelHtml, groups) {
  const box = U.el('div', { class: 'col', style: 'gap:3px;width:100%' });
  box.appendChild(U.el('div', { class: 'cm', html: labelHtml }));
  const s = U.el('div', { style: 'display:flex;height:24px;border:1px solid var(--border);border-radius:5px;overflow:hidden;width:100%' });
  groups.forEach(g => {
    const outer = U.el('div', { style: 'flex:' + g.w + ' 1 0;min-width:0;display:flex;border-right:1px solid var(--border)' });
    (g.sub || [g]).forEach(ss => {
      const e = U.el('div', { style: 'flex:' + ss.w + ' 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:8.5px;overflow:hidden;white-space:nowrap;background:' + ss.bg + ';color:var(--' + ss.c + ')' });
      e.textContent = ss.t || '';
      outer.appendChild(e);
    });
    s.appendChild(outer);
  });
  box.appendChild(s);
  host.appendChild(box);
}

const g40 = [];
for (let i = 0; i < 8; i++) {
  g40.push({ w: 18, sub: [
    { w: 2,  t: '', c: 'a', bg: 'rgba(88,166,255,.55)' },
    { w: 16, t: (i === 0 ? 'qs 16B' : ''), c: 'b', bg: 'rgba(63,185,80,.16)' }
  ] });
}
rowStrip(wrap.querySelector('#s0'),
  'Q4_0 · 256 个权重 = 8 块 × 18 B = 144 B　<span style="color:var(--a)">■</span> d 2B×8 = 16B　<span style="color:var(--b)">■</span> qs 16B×8 = 128B',
  g40);
rowStrip(wrap.querySelector('#s1'),
  'Q4_K · 256 个权重 = 1 个超块 = 144 B　<span style="color:var(--a)">■</span> dm 4B　<span style="color:var(--c)">■</span> scales 12B　<span style="color:var(--b)">■</span> qs 128B',
  [{ w: 144, sub: [
    { w: 4,   t: '', c: 'a', bg: 'rgba(88,166,255,.55)' },
    { w: 12,  t: '', c: 'c', bg: 'rgba(210,153,34,.30)' },
    { w: 128, t: 'qs 128B', c: 'b', bg: 'rgba(63,185,80,.16)' }
  ] }]);

const t = U.table(['', 'Q4_0', 'Q4_K'],
  [['blck_size', '32', '256'],
   ['type_size', '18 B', '144 B'],
   ['bit / 权重', '4.5', '4.5'],
   ['scale 粒度', '1 个 fp16 / 32', '1 个 6-bit / 32'],
   ['min 的表示', '无（对称）', '1 个 6-bit / 32'],
   ['内核代价', '无需位运算', '每超块解 16 个 6-bit']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '一个 <span class="mono">ggml_tensor</span>，<span class="mono">type = GGML_TYPE_Q4_0</span>、' +
  '<span class="mono">ne = [4096, 1, 1, 1]</span>。它占多少字节？换成 ' +
  '<span class="mono">GGML_TYPE_Q4_K</span> 呢？',
  'Q4_0：<span class="mono">blck_size = 32</span>、<span class="mono">type_size = 18</span>，' +
  '4096 / 32 = 128 块，128 × 18 = <b>2304 字节</b>。<br>' +
  'Q4_K：<span class="mono">blck_size = 256</span>、<span class="mono">type_size = 144</span>，' +
  '4096 / 256 = 16 块，16 × 144 = <b>2304 字节</b>。<br>' +
  '两者都是 4.5 bit/权重，所以字节数一样 —— 差别只在<span class="mono">块怎么切、谁带 scale</span>，' +
  '以及内核要不要做位运算解包。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '同一张表里，每个类型三行：<span class="v">blck_size</span>、<span class="v">type_size</span>、<span class="v">to_float</span>。',
  '<span class="v">[GGML_TYPE_Q4_0]</span>：<span class="v">blck_size = QK4_0</span>（32）、<br><span class="v">type_size = sizeof(block_q4_0)</span>（18）。第 2 幕的字节数在这里生效。',
  '<span class="v">to_float = (ggml_to_float_t) dequantize_row_q4_0</span>：<br><span class="k">类型号 -> 块布局 -> 反量化函数</span>，一条链在这里闭合。',
  '同一行还有 <span class="v">from_float_ref = quantize_row_q4_0_ref</span> —— 第 5 幕那个参考实现。<br>F16 这类非量化类型的 to_float 只是一个 fp16->fp32 转换，没有块。',
  '回到验收点：Q4_0 与 Q4_K 在 256 个权重上都是 144 字节，<span class="k">参数 16 字节 + quants 128 字节</span>；<br>区别是 Q4_0 用 8 个 fp16 scale，Q4_K 用 4 + 12 字节换来了 per-32 的 6-bit scale <b>和</b> min。',
  '一句话：<span class="k">类型号决定的不是"多少位"，而是"块怎么切、scale 怎么存"</span>。<br>内核的一切代价都从这张表开始。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => { r.className = ''; }); U.markLines(document, [16]); });
tl.at(4200, () => { U.markLines(document, [19, 21]); rows.forEach((r, k) => { r.className = (k === 0 || k === 1) ? 'on' : ''; }); msg.innerHTML = texts[1]; });
tl.at(8000, () => { U.markLines(document, [24]); rows.forEach((r, k) => { r.className = (k === 2) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
tl.at(11800, () => { U.markLines(document, [18, 25]); rows.forEach((r, k) => { r.className = (k === 3 || k === 4) ? 'on' : ''; }); msg.innerHTML = texts[3]; });
tl.at(15600, () => { U.markLines(document, [16, 19, 21, 24, 25]); rows.forEach((r, k) => { r.className = (k === 5) ? 'on' : ''; }); msg.innerHTML = texts[4]; });
tl.at(18800, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、块大小的两个常量',
    '整个量化体系只有两个"块大小"量级：小块 `QK4_0 = 32`（Q4_0 家族），'
    '超块 `QK_K = 256`（K-quant 家族）。`K_SCALE_SIZE = 12` 是 Q4_K 给'
    '"量化后的 scale/min"留的字节数 —— 第 6、7 幕会把 12 = 96 bit 拆开验算。\n\n'
    '```text\n'
    'QK4_0 = 32     # 小块：32 个元素一块\n'
    'QK_K  = 256    # 超块：256 个元素一个 super-block\n'
    'K_SCALE_SIZE = 12  # 8 个 6-bit scale + 8 个 6-bit min\n'
    '```\n\n'
    '下面第二段是本文件最前面的类型定义：在 CPU 的 C 编译分支里 '
    '`ggml_half` 就是 `uint16_t`，也就是 **2 字节**。'
    '后面所有 `sizeof(ggml_half)` 都按 2 算。',
    src=COMMON, parts=[(3, 12), (86, 90)], lang='c')

L.section(
    '二、★ 小块家族：block_q4_0 与 block_q4_1',
    '两个结构体只差一个字段，却差了 2 字节 / 块，也就是 0.5 bit / 权重。\n\n'
    '| 类型 | 字段 | 字节 | bit / 权重 |\n|---|---|---|---|\n'
    '| `block_q4_0` | `d` + `qs[16]` | 18 | 4.5 |\n'
    '| `block_q4_1` | `d` + `m` + `qs[16]` | 20 | 5.0 |\n\n'
    '这里的数字全部来自 `static_assert` 的等式：`sizeof(ggml_half) = 2`（'
    '`ggml-common.h:6` 的 `typedef uint16_t ggml_half;`），'
    '`QK4_0 / 2 = 16`，所以 `sizeof(block_q4_0) = 18`。\n\n'
    '注意 `GGML_COMMON_AGGR_U` 那个匿名 union：`d` / `m` 两个 fp16 与一个 '
    '`ggml_half2 dm` 共享同一块内存，方便 SIMD 一次搬 4 字节。',
    src=COMMON, parts=[(194, 212)], lang='c')

L.section(
    '三、Q8_0：给量化点积当被乘数',
    '`QK8_0` 也是 32，但每个元素占满 1 字节：`2 + 32 = 34` 字节 / 块，'
    '即 8.5 bit / 权重。它不是用来存模型权重的"高压缩"格式，'
    '而是量化点积里被乘的那一侧（L5-04 会看到 q4_0 × q8_0 的配对）。'
    '另一个区别是 `qs` 的类型：`int8_t` 而不是 `uint8_t` —— 8 bit 存的是有符号整数。',
    src=COMMON, parts=[(251, 256)], lang='c')

L.section(
    '四、Q6_K：6-bit 主力',
    'K-quant 家族里精度最高的一档：16 个 16 元素子块，每块一个 8-bit 有符号 scale，'
    '整个超块再共用一个 fp16 的 `d`。`ql` 存低 4 位、`qh` 存高 2 位 —— '
    '和 Q4_K 的 6-bit scale 一样是"拆成两处存"的思路，只是拆的对象不同。\n\n'
    '`sizeof(block_q6_K) = 2 + 16 + 192 = 210` 字节，源码注释写明 '
    '"Effectively 6.5625 bits per weight"（210 × 8 / 256 = 6.5625）。',
    src=COMMON, parts=[(358, 368)], lang='c')

L.section(
    '五、反量化函数族：一个类型一个函数',
    '`ggml-quants.h` 是这 3 个文件里唯一的"接口面"。它按同一套命名规则'
    '把两族函数排开：\n\n'
    '```text\n'
    'quantize_row_<类型名>_ref(...)   # 量化：参考实现，模型转换用\n'
    'dequantize_row_<类型名>(...)     # 反量化：运行时把块还原成 float\n'
    '```\n\n'
    '注意第 53 行那个被注释掉的 `dequantize_row_q8_1`：Q8_1 只做量化（当被乘数），'
    '不需要反量化回 float，所以声明被注释掉了。\n\n'
    '上面第 14 行的注释还说明了这些函数为什么是 `GGML_API`：'
    '`these functions are defined as GGML_API because they used by the CPU backend`。',
    src=QH, parts=[(14, 34), (45, 63)], lang='c')

L.section(
    '六、反量化 Q4_0：一个字节管两个半块',
    '`qs[j]` 的低 4 位对应 `y[j]`、高 4 位对应 `y[j + qk/2]`，'
    '即一个字节跨越块的前后两半。这个排布让"取 32 个连续权重"必须先分两路走：'
    '这也是 L5-04 的 repack 要把 q4_0 重排成 4x4 / 4x8 交错布局的直接原因。\n\n'
    '减 8 是为了把无符号的 `[0, 15]` 平移回对称区间：还原出的值是 `[-8d, 7d]`。',
    src=QC, parts=[(459, 477)], lang='c')

L.section(
    '七、★ 量化 Q4_0：d = max / -8，没有 min',
    '源码第 112 行的注释写着 `reference implementation for deterministic creation of model files` —— '
    '这条路径是模型转换（L2-09）用的，要求确定性。\n\n'
    '对称量化的三步：\n\n'
    '1. 扫出 `amax`，连同取到 amax 的那个带符号值一起记进 `max`；\n'
    '2. `d = max / -8`，`id = 1/d`；\n'
    '3. `xi = MIN(15, (int8_t)(x*id + 8.5f))`，两个 nibble 合进一个字节。\n\n'
    '`+8.5f` 同时完成"平移到 [0,15]"和"四舍五入"；`MIN(15, ...)` 兜住上溢 —— '
    '正方向的极值会被压一点，这就是对称量化的代价。',
    src=QC, parts=[(113, 148)], lang='c')

L.section(
    '八、★ Q4_K 的 6-bit scale/min 解包',
    '`K_SCALE_SIZE = 12` 字节 = 96 bit，而 8 个 scale + 8 个 min 各 6 bit '
    '正好是 8 ×(6+6) = 96 bit。一位不多、一位不少。\n\n'
    '装法就是 `get_scale_min_k4`：\n\n'
    '| j | `*d`（scale） | `*m`（min） |\n|---|---|---|\n'
    '| 0..3 | `q[j] & 63` | `q[j + 4] & 63` |\n'
    '| 4..7 | `(q[j+4] & 0xF) \\| ((q[j-4] >> 6) << 4)` | '
    '`(q[j+4] >> 4) \\| ((q[j-0] >> 6) << 4)` |\n\n'
    '`q[0..7]` 只用低 6 位，剩下的最高 2 位被借给 `j >= 4` 的 scale/min；'
    '`q[8..11]` 出低 4 位。取用时必须移位拼接，这是 K-quant 内核比 Q4_0 多出来的固定开销。',
    src=QC, parts=[(880, 887)], lang='c')

L.section(
    '九、dequantize_row_q4_K：谁消费这些 6-bit',
    '每个超块循环 4 次（`j += 64`），每次读两个子块的 scale/min，'
    '把 32 个字节的低半字节和高半字节分别还原成 32 个值 —— '
    '一次 64 个元素。\n\n'
    '注意 `d1 = d * sc` 与 `m1 = min * m`：超块头的 `d` / `dmin` 是 fp16，'
    '子块的 `sc` / `m` 是 6-bit 整数，两级相乘才是最终系数。'
    '这与 Q4_0 只有一个 `d` 形成对照（第 2、5 幕）。',
    src=QC, parts=[(1529, 1551)], lang='c')

L.section(
    '十、类型号把一切绑在一起',
    '`ggml_type_traits` 表（`ggml.c:632` 起）是"类型号"与"内存布局"的唯一连接点。'
    '每个类型登记三件事：\n\n'
    '| 字段 | 含义 | Q4_0 的值 |\n|---|---|---|\n'
    '| `blck_size` | 多少元素算一块 | `QK4_0` = 32 |\n'
    '| `type_size` | 一块多少字节 | `sizeof(block_q4_0)` = 18 |\n'
    '| `to_float` | 怎么还原成 float | `dequantize_row_q4_0` |\n\n'
    '对比同一段代码里的 `GGML_TYPE_F16`：它的 `blck_size = 1`、'
    '`type_size = sizeof(ggml_fp16_t)`，`to_float` 只是 `ggml_fp16_to_fp32_row` —— '
    '**没有块，就没有块布局这回事**。',
    src=GC, parts=[(632, 634), (669, 676), (693, 700)], lang='c')

L.footnote_add('本课声明 4 个源文件。前 3 个是计划里指派给 L1-04 的 '
               '`ggml-common.h` / `ggml-quants.h` / `ggml-quants.c`；'
               '第 4 个 `ggml/src/ggml.c` 只用于第 8 幕与第十节 —— '
               '`ggml_type_traits` 表（`blck_size` / `type_size` / `to_float` 的注册）'
               '只存在于这个文件里，是"GGML_TYPE 与块大小的绑定"这一讲解要点的唯一出处。'
               '计划里 `ggml.c` 同时指派给 L1-02 / L1-03，本课只是追加引用，不改变其归属。')
L.footnote_add('第 2、6 幕的字节数（18 / 20 / 34 / 144 / 210）由源码的 `static_assert` 等式'
               '与 `ggml_half = uint16_t` 直接算出，并已用一个只 include `ggml-common.h` 的'
               '小程序实测确认：`sizeof(block_q4_0) = 18`、`sizeof(block_q4_1) = 20`、'
               '`sizeof(block_q8_0) = 34`、`sizeof(block_q4_K) = 144`、`sizeof(block_q6_K) = 210`。')

L.prereqs('`L1-01`（ggml 张量：算子的数据面）')

L.goal(
    '说出 `QK4_0 = 32` 与 `QK_K = 256` 各自对应哪个家族，以及块的字节数是怎么算出来的（对应验收点）；',
    '画出 Q4_0（2 + 16 = 18 字节）与 Q4_K（4 + 12 + 128 = 144 字节）的块内存布局，'
    '并说出这个差异给内核带来的代价；',
    '解释 Q4_0 的 `d = max / -8` 为什么是"对称"量化，以及 Q4_1 多出来的 2 字节买到了什么；',
    '说明 Q4_K 的 12 字节 `scales[]` 如何装下 8 个 6-bit scale 与 8 个 6-bit min，'
    '以及 `get_scale_min_k4` 的两次移位分别在做什么；',
    '指出 `ggml_type_traits` 表里哪三个字段把"类型号"变成了"块布局 + 反量化函数"。')

L.conclusion(
    '块大小 = 元素数 + 字节数，两者都由类型号决定',
    '| 类型 | blck_size | type_size | bit / 权重 |\n|---|---|---|---|\n'
    '| Q4_0 | `QK4_0` = 32 | 18 B | 4.5 |\n'
    '| Q4_1 | `QK4_1` = 32 | 20 B | 5.0 |\n'
    '| Q8_0 | `QK8_0` = 32 | 34 B | 8.5 |\n'
    '| Q4_K | `QK_K` = 256 | 144 B | 4.5 |\n'
    '| Q6_K | `QK_K` = 256 | 210 B | 6.5625 |\n\n'
    '字节数不是"元素数 × 位宽 / 8"，而是 `元素数 × 位宽 / 8 + 每块的 scale / min`。'
    '正是这个"额外开销"把名义位宽和实际位宽拉开（Q4_0 名义 4 bit，实际 4.5 bit）。')

L.conclusion(
    '★ 同样的 4.5 bit，两种花法',
    '在 256 个权重上，Q4_0 与 Q4_K **都是 144 字节**，而且都恰好是'
    '「参数 16 字节 + quants 128 字节」：\n\n'
    '```text\n'
    'Q4_0  8 块 × (d 2B + qs 16B)  =  16B 参数 + 128B quants\n'
    'Q4_K  1 超块 (dm 4B + scales 12B + qs 128B) = 16B 参数 + 128B quants\n'
    '```\n\n'
    '区别在参数怎么花：Q4_0 是 8 个 fp16 的 scale（每 32 个权重一个），'
    'Q4_K 是 2 个 fp16 的超块 scale + 12 字节的 6-bit scale/min（每 32 个权重'
    '一个 scale **和**一个 min）。同样的预算换来的是"更细的粒度 + 非对称"。')

L.conclusion(
    '对内核的影响：三件事',
    '1. **访问粒度**：Q4_0 的 `nb[0] = 18`，不是 4 或 16 的整数倍；内核必须'
    '按"2 字节 scale + 16 字节 nibble"的节奏走。\n'
    '2. **位运算开销**：Q4_0 只需 `& 0x0F` / `>> 4`；Q4_K 每个超块还要解 16 个 '
    '6-bit 的 scale/min（第 7 幕的移位拼接），这是固定开销。\n'
    '3. **SIMD 友好度**：Q4_0 的"字节 j 管 y[j] 与 y[j+16]"对 SIMD 不友好，'
    '所以 L5-04 的 repack 会把它重排成 4x4 / 4x8 的交错布局；'
    '这一步不改数值，只改字节顺序 —— 前提正是本课讲的布局。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
