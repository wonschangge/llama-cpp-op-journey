#!/usr/bin/env python3
"""L4-01 · 分配器 ggml-alloc：算子的内存从哪来 —— 课件 spec。

运行：python3 L4-memory-and-scheduling/L4-01-ggml-alloc/lesson.spec.py

所有代码引用按【上游行号】从 ggml/src/ggml-alloc.c 与 ggml/include/ggml-alloc.h
逐字抽取，spec 里不出现任何手打的代码文本。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC = 'ggml/src/ggml-alloc.c'
HSRC = 'ggml/include/ggml-alloc.h'


def ridx(parts, notes_src, lines):
    """把【上游行号】换算成 lessonkit 渲染后的行下标。

    注解行（//>>）会占一个渲染行，所以被注解行之前插入的每条注解都要累加。
    返回值是 JS 数组字面量，直接写进 visual 里的 U.markLines 调用。
    """
    base = parts[0][0]
    nl = sorted((notes_src or {}).keys())
    out = [ln - base + sum(1 for x in nl if x < ln) for ln in lines]
    return '[' + ', '.join(str(x) for x in out) + ']'


L = Lesson(
    id='L4-01',
    layer='L4 · 内存与调度',
    title='分配器 ggml-alloc：算子的内存从哪来',
    codecap='ggml/src/ggml-alloc.c 与 ggml/include/ggml-alloc.h（逐字引用）',
    nav={'prev': {'href': '../../L3-backend-registry/L3-04-backend-features/index.html',
                  'label': 'L3-04 后端能力探测'},
         'next': {'href': '../L4-02-scheduler-split/index.html',
                  'label': 'L4-02 ★ 调度器：把图切给不同后端'}},
)

L.note('**一句话**：前面几课里，每个 `ggml_tensor` 都有一个 `data` 指针和一个 `buffer`；'
       '这一课回答**这两个字段是谁填的**。填它们的不是算子、也不是后端内核，'
       '而是 `ggml-alloc.c` 里的**图分配器**（graph allocator）。')
L.note('它的核心事实只有一条：**图在开始执行之前就已经建好了，所以每个张量的"出生"和"死亡"'
       '时刻是已知的**。知道完整生命周期，就可以把互不重叠的生存期压进同一块内存 —— '
       '而不是给每个张量各开一块。这就是验收点"为什么 ggml 需要自己管理内存"的答案。')
L.note('本课只用两个源文件：`ggml/src/ggml-alloc.c`（1249 行）与 `ggml/include/ggml-alloc.h`'
       '（86 行）。文中提到的每个行号都出自这两个文件。')

# ------------------------------------------------------------------ 第 1 幕

P1 = [(24, 40)]
N1 = {29: '可选但关键：拿一张"最坏情况图"（例如 max_batch）先规划一次',
      33: '每次 decode 都重新建一张图 —— 但形状不变时 L2-07 会复用上一次的那张',
      38: '分配完成后才轮到后端执行；本课只讲 30 / 34 / 36 这三步'}

L.scene(
    kicker='L4-01 · 全局',
    title='内存规划发生在建图之后、执行之前',
    sub='ggml-alloc.h 的注释把标准用法写死了：选 buffer 类型 -> 预留 -> 分配 -> 查大小 -> 执行。',
    caption='下一课 L4-02 讲调度器怎么把同一张图切成多个 buffer；本课先把单 buffer 的规划讲透。',
    src=HSRC, parts=P1, duration=20000,
    mark_src=[27, 30, 34, 36, 39], notes_src=N1,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">建图</span><span class="arrow">-></span>
    <span class="chip a">reserve</span><span class="arrow">-></span>
    <span class="chip b">alloc_graph</span><span class="arrow">-></span>
    <span class="chip c">get_buffer_size</span><span class="arrow">-></span>
    <span class="chip d">graph_compute</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '先选 buffer 类型', b: 'CPU / CUDA / Metal ...；带 _n 的版本<br>一次给多种，对应 L4-02 的图切分', m: 'ggml_gallocr_new_n' },
  { c: 'b', t: '用最坏情况的图预留', b: '只做内存规划：<br>不改图、不碰任何张量', m: 'ggml_gallocr_reserve' },
  { c: 'c', t: '对当前图落位', b: '把每个张量的 data / buffer<br>指到规划好的地址上', m: 'ggml_gallocr_alloc_graph' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '219px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '这一段注释就是本课的地图：<span class="k">内存规划夹在建图与执行之间</span>。',
  '<span class="v">reserve</span> 用一张"最坏情况图"先规划一次，运行时就不必再重新分配 —— 它是可选的，但生产代码一定会调。',
  '<span class="v">alloc_graph</span> 才把地址真正发给张量；它用的是<b>上一次规划的结果</b>，不是重新算一遍。',
  '<span class="v">get_buffer_size</span> 把规划结果报出来：这就是"这个模型要占多少显存"的答案。',
  '为什么要有这一整层？因为<span class="k">让每个张量各自去要一块内存</span>，在 70 层 transformer 上是活不下去的 —— 下一幕算这笔账。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(17600, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

P2 = [(59, 92)]
N2 = {59: 'tallocr = tensor allocator，最小可用的那一种',
      77: '先问后端：这个张量按你的规矩要占多少字节',
      87: '这一行就是全部策略：offset 只会往前推，永不回退'}

L.scene(
    kicker='L4-01 · 动机',
    title='★ 只进不退的分配器 <span class="hl-g">活不过一张图</span>',
    sub='ggml_tallocr 是最简单的分配器：一个 buffer、一个 offset，只往前走；它没有配套的释放函数。',
    caption='对照 L1-05：context 的 arena 也是"只进不退"，但那是放张量元数据的；这里说的是放数据的 buffer。',
    src=SRC, parts=P2, duration=18000,
    mark_src=[61, 77, 87], notes_src=N2,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:12px;align-items:flex-start">
    <div class="col" style="gap:7px;width:340px" id="bars"></div>
    <div class="col grow" style="gap:7px" id="notes"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const bars = wrap.querySelector('#bars');
const b1 = U.bar('各占一块', 'g');
const b2 = U.bar('复用峰值', 'b');
bars.appendChild(b1.el); bars.appendChild(b2.el);
bars.appendChild(U.el('div', { class: 'cm', style: 'margin:0',
  text: '同一张 decode 图：每个中间张量各要一块 vs 按生命周期复用（示意）' }));
b1.fill.style.width = '0%'; b2.fill.style.width = '0%';

const notes = wrap.querySelector('#notes');
notes.innerHTML =
  '<div class="card" style="border-left-color:var(--g)">' +
  '<div class="ct" style="color:var(--g)">它为什么不够用</div>' +
  '<div class="cb">一条数据流上，<b>绝大多数中间张量只活一小会儿</b>：某个 RMS_NORM 的输出喂给下一个 MUL_MAT 之后就没用了。' +
  '但 offset 不会退回，那块内存在这张图剩下的时间里一直空占着。</div></div>' +
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">它也不是没用</div>' +
  '<div class="cb">权重这种<b>整场都活着</b>的张量，用它顺序摆一遍正合适 —— ' +
  'header 里的 <span class="cm" style="margin:0">ggml_backend_alloc_ctx_tensors</span> 走的就是这条路。</div></div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">没有 free 函数</div>' +
  '<div class="cb">整个 ggml-alloc.h 只有 <span class="cm" style="margin:0">ggml_tallocr_new</span> 与 ' +
  '<span class="cm" style="margin:0">ggml_tallocr_alloc</span> 两个 API —— "还块"这件事压根不存在。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '32 行代码就是一个完整的分配器：<span class="k">算大小 -> 对齐 -> offset 往前推 -> 告诉后端</span>。',
  '<span class="v">ggml_backend_buffer_get_alloc_size</span>（77 行）让后端参与决定大小 —— 同一个张量在 CPU 和 CUDA 上可能不一样大。',
  '<span class="v">talloc-&gt;offset += size</span>（87 行）：<b>单调递增</b>。这就是"不复用"的全部含义。',
  '把一张图从头跑到尾要多少内存？<span class="v">所有张量大小之和</span>。而其中同时活着的，可能只有十分之一。',
  '所以 ggml 需要另一套东西：<span class="k">知道谁什么时候死，然后把死者的内存立刻发给下一个人</span> —— 这就是 ggml_dyn_tallocr。'
];
tl.at(700, () => { b1.fill.style.width = '100%'; b1.val.textContent = '100%'; msg.innerHTML = texts[0]; });
tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, @@A@@); });
tl.at(7700, () => { msg.innerHTML = texts[2]; U.markLines(document, @@B@@); });
tl.at(11200, () => {
  b2.fill.style.width = '24%'; b2.val.textContent = '24%';
  msg.innerHTML = texts[3]; U.markLines(document, []);
});
tl.at(14600, () => { msg.innerHTML = texts[4]; });
'''.replace('@@A@@', ridx(P2, N2, [77])).replace('@@B@@', ridx(P2, N2, [87]))
)

# ------------------------------------------------------------------ 第 3 幕

P3 = [(110, 133)]
N3 = {112: '空闲块的粒度就是 offset + size 两个数 —— 它不需要知道谁曾住在里面',
      117: '块的数量是有限的：256 个。用完在第 136 行直接 GGML_ASSERT 失败',
      124: '一个 buffer 最多切成 16 个 chunk；显存被 max_size 切开时用得上（L4-03）'}

L.scene(
    kicker='L4-01 · 核心',
    title='★ <span class="hl-a">free_block</span>：复用靠的就是这张空闲表',
    sub='动态分配器不记录"哪些张量在"，只记录"哪些字节是空的" —— 于是张量之间可以直接换手。',
    caption='两个上限决定了这张表的形状：MAX_FREE_BLOCKS = 256（第 14 行）、GGML_VBUFFER_MAX_CHUNKS = 16（第 96 行）。',
    src=SRC, parts=P3, duration=20000,
    mark_src=[110, 115, 121], notes_src=N3,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="cm" id="cap" style="margin:0">同一个 chunk 里的一维地址空间（示意）</div>
  <div class="row" id="strip" style="gap:3px;justify-content:center;align-items:stretch"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'free_block', b: '一段空闲内存：<br>起点偏移 + 长度', m: 'offset, size' },
  { c: 'b', t: 'tallocr_chunk', b: '一个 chunk 的空闲表 + 水位线<br>（已用到过的最高地址）', m: 'free_blocks[256] · n_free_blocks · max_size' },
  { c: 'c', t: 'ggml_dyn_tallocr', b: '每个 buffer 一个动态分配器：<br>对齐、chunk 上限、chunk 列表', m: 'alignment · max_chunk_size · chunks[16] · n_chunks' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '219px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });

const strip = wrap.querySelector('#strip');
const segs = [
  { w: 96,  t: '已分配 (x)',  c: '#1f2630' },
  { w: 148, t: 'free 12288',  c: 'rgba(88,166,255,.22)' },
  { w: 84,  t: '已分配 (h)',  c: '#1f2630' },
  { w: 112, t: 'free 8192',   c: 'rgba(63,185,80,.22)' },
  { w: 186, t: '最后一个 free（可向外扩）', c: 'rgba(210,153,34,.18)' }
];
const sels = segs.map(s => {
  const e = U.el('div', { style: 'width:' + s.w + 'px;height:34px;border:1px solid var(--border);' +
    'border-radius:4px;background:' + s.c + ';display:flex;align-items:center;justify-content:center;' +
    'font-family:var(--mono);font-size:9px;color:var(--muted);padding:0 4px;overflow-wrap:anywhere' });
  e.textContent = s.t;
  strip.appendChild(e);
  return e;
});

const msg = wrap.querySelector('#msg');
const texts = [
  '分配器不维护"张量 -> 地址"的账本，它只维护<span class="k">空闲块表</span>。',
  '<span class="v">free_blocks[]</span> 按地址<span class="k">有序</span>存放（插入时就排好，见第 135-150 行）—— 有序是为了还块时能立刻判断"能不能和邻居合并"。',
  '<span class="v">max_size</span> 是这个 chunk 的<b>水位线</b>，也是最后开 buffer 时唯一要用的数：规划完它多大，buffer 就开多大（第 917-924 行）。',
  '<span class="v">chunks[16]</span> 让一个分配器可以横跨多个后端 buffer —— 这是 L4-02 多 buffer 调度的地基。',
  '一句话：<span class="k">地址空间被切成"已分配"与"空闲块"两种状态，空闲块就是可以被下一个人接手的部分</span>。'
];
const hi = [[1], [1, 3], [2], [3, 4], [1, 2, 3, 4]];
defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  sels.forEach((s, k) => { s.style.outline = hi[i].indexOf(k) >= 0 ? '1px solid var(--a)' : 'none'; });
  msg.innerHTML = texts[i];
}));
tl.at(18700, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  sels.forEach(s => { s.style.outline = 'none'; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

P4 = [(202, 224)]
N4 = {203: '先按分配器的 alignment 把请求大小对齐（aligned_offset，第 53-57 行）',
      211: '注释原话：find the best fitting free block besides the last block',
      218: '两个条件同时成立：装得下（size >= 请求），且比目前找到的更小'}

L.scene(
    kicker='L4-01 · 核心',
    title='★ <span class="hl-a">best-fit</span>：从空闲表里挑最小够用的那块',
    sub='注意循环上界那个减一：最后一个空闲块不参与挑选，因为它是"可以向外扩"的边界。',
    caption='挑选判据在源码注释里写得很清楚：装得下，且是装得下的里面最小的。这就是 best fit。',
    src=SRC, parts=P4, duration=19000,
    mark_src=[211, 215, 218], notes_src=N4,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="formula" id="ask"></div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

wrap.querySelector('#ask').innerHTML =
  '请求：<span class="v">ggml_dyn_tallocr_alloc(alloc, 12288, node)</span> —— ' +
  '要一块 <span class="k">12288 字节</span>（已按 alignment 对齐）';

const t = U.table(
  ['free_block', 'offset', 'size', '装得下?', '结果'],
  [['#0', '4096', '8192', '否（8192 小于 12288）', '跳过'],
   ['#1', '16384', '16384', '是', '★ 选中：装得下的里面最小的'],
   ['#2', '32768', '12288', '是（并列）', '不选：不比它更小'],
   ['#3（最后一个）', '45056', '很大', '—', '本轮不参与挑选']],
  { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '三个数先固定下来：<span class="v">best_fit_chunk = -1</span>、<span class="v">best_fit_block = -1</span>、<span class="v">best_fit_size = SIZE_MAX</span>（第 207-209 行）。',
  '内层循环从 0 走到 <span class="v">n_free_blocks - 1</span>（第 215 行）—— <span class="k">少扫的正好是最后一个块</span>：那是水位线以上、可以向外长的地方。',
  '每碰到一个装得下的块就比较 <span class="v">size &lt;= best_fit_size</span>（第 218 行）：<span class="k">只留更小的那个</span>，所以最后拿到的是最紧的一块。',
  '循环结束后第 263-265 行把选中块的 offset 让出去、size 减掉；减到 0 就把这一项从空闲表里删掉（第 266-269 行）。',
  '只有 <span class="v">best_fit_block == -1</span>（一个都装不下）时，第 226 行才开始看最后一个块 —— 用它意味着把 buffer 撑大。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3200 + i * 2700, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 3)];
  if (i === 1) { U.markLines(document, @@A@@); }
  if (i === 2) { U.markLines(document, @@B@@); }
}));
tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''.replace('@@A@@', ridx(P4, N4, [215])).replace('@@B@@', ridx(P4, N4, [218]))
)

# ------------------------------------------------------------------ 第 5 幕

P5 = [(311, 350)]
N5 = {317: '分两种情况：新空出来的块贴在已有空闲块的后面，还是前面',
      321: '情况一：addr 正好接在块尾 -> 直接长大',
      326: '长大之后顺手看一眼下一块能不能一起吞掉',
      334: '情况二：addr 正好接在块头 -> 往前吞',
      348: '两个方向都不沾（左右都是活人）-> 只能新开一个空闲块'}

L.scene(
    kicker='L4-01 · 核心',
    title='还块：与相邻空闲块<span class="hl-b">就地合并</span>',
    sub='张量用完只是把地址还给空闲表；不合并的话，空闲表很快会被碎片塞满（上限 256 块）。',
    caption='源码自评："this is a very naive implementation" —— 但它够用，因为一张图里的空闲块通常很少。',
    src=SRC, parts=P5, duration=19000,
    mark_src=[321, 326, 334, 349], notes_src=N5,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="cm" id="c1" style="margin:0">释放前：h 被两块空闲区夹在中间</div>
  <div class="row" id="before" style="gap:3px;justify-content:center"></div>
  <div class="cm" id="c2" style="margin:0">释放 h 之后：三块并成一块，下次要 24 KB 也能一次给出去</div>
  <div class="row" id="after" style="gap:3px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

function seg(host, w, text, bg) {
  const e = U.el('div', { style: 'width:' + w + 'px;height:32px;border:1px solid var(--border);' +
    'border-radius:4px;background:' + bg + ';display:flex;align-items:center;justify-content:center;' +
    'font-family:var(--mono);font-size:9px;color:var(--muted);transition:all .5s;padding:0 3px' });
  e.textContent = text;
  host.appendChild(e);
  return e;
}
const before = wrap.querySelector('#before');
const bFree1 = seg(before, 116, 'free 8192', 'rgba(88,166,255,.22)');
const bH     = seg(before, 130, 'h（已分配）', '#1f2630');
const bFree2 = seg(before, 96,  'free 4096', 'rgba(88,166,255,.22)');

const after = wrap.querySelector('#after');
const aMerged = seg(after, 116 + 130 + 96 + 6, 'free 24576（8192 + 12288 + 4096）', 'rgba(63,185,80,.22)');
aMerged.style.opacity = '.18';

const msg = wrap.querySelector('#msg');
const texts = [
  '还块要解决的是<span class="k">碎片</span>：一块 8 KB 的空闲区旁边就是一块 4 KB 的，单独都装不下 24 KB。',
  '第 321 行先试"贴在块尾"：<span class="v">block-&gt;offset + block-&gt;size == addr.offset</span>。',
  '成立就把自己接上去（第 322 行），再看一眼下一块能不能一起吞（第 326-329 行）。',
  '否则第 334 行试"贴在块头"：<span class="v">addr.offset + size == block-&gt;offset</span>，成立就往前吞（第 335-336 行），同样回头看上一块（第 340-343 行）。',
  '两个方向都不沾，才 <span class="v">ggml_dyn_tallocr_insert_block</span> 新开一块（第 349 行）—— 这是空闲表变碎的唯一来源。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3800, () => { bH.style.background = 'rgba(210,153,34,.25)'; bH.style.borderColor = 'var(--c)'; msg.innerHTML = texts[1]; });
tl.at(7300, () => { msg.innerHTML = texts[2]; });
tl.at(10800, () => {
  bH.textContent = 'free 12288'; bH.style.background = 'rgba(88,166,255,.22)'; bH.style.borderColor = 'var(--border)';
  msg.innerHTML = texts[3];
});
tl.at(14300, () => {
  bH.style.opacity = '.18'; bFree1.style.opacity = '.18'; bFree2.style.opacity = '.18';
  aMerged.style.opacity = '1';
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 6 幕

P6 = [(723, 761)]
N6 = {723: '所有叶子先落位 —— 权重、KV cache 这些"整场都活着"的东西',
      730: '这一趟除输入外不分配，只做两件事：数 n_children、数 n_views',
      739: '视图不改数据，但要替它记一笔：源张量多了一个"借住者"',
      754: '每出现一次 src 引用，消费者的计数加一 —— 这就是"生命周期"的量化形式'}

L.scene(
    kicker='L4-01 · 核心',
    title='★ 第一趟：先数清<span class="hl-c">谁还要用我</span>',
    sub='分配之前，先把每个张量被引用几次（n_children）、被几个视图指着（n_views）数出来。',
    caption='对照 L1-03：图的拓扑序在这里第一次被"用起来" —— 节点顺序就是生命周期的顺序。',
    src=SRC, parts=P6, duration=22000,
    mark_src=[723, 730, 739, 754], notes_src=N6,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip a">a 输入</span>
    <span class="chip b">W 权重</span>
    <span class="chip d">c 偏置</span>
    <span class="arrow">-></span>
    <span class="chip c">h = mul_mat(W, a)</span>
    <span class="arrow">-></span>
    <span class="chip d">y = add(h, c)</span>
    <span class="arrow">-></span>
    <span class="chip e">out</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['张量', 'n_children', 'n_views', '第一趟结束时它欠了几笔'],
  [['a（带 INPUT 标记）', '1', '0', 'mul_mat 用完就归零'],
   ['W（权重叶子）', '1', '0', 'mul_mat 用完就归零'],
   ['c（偏置叶子）', '1', '0', 'add 用完就归零'],
   ['h = mul_mat(W, a)', '1', '0', 'add 用完就归零（还会被原地接管）'],
   ['y = add(h, c)', '0', '0', '它是图输出，永不为零']],
  { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '这一趟的产物只有两个整数：<span class="v">n_children</span> 与 <span class="v">n_views</span>。',
  '<span class="v">graph-&gt;leafs[]</span> 里的张量先全部落位（第 725-727 行）—— 它们不属于任何一次"执行"，可以认为活到最后。',
  '带 <span class="v">GGML_TENSOR_FLAG_INPUT</span> 的张量在第二趟之前就抢先落位（第 744-746 行），免得地址被别人占掉。',
  '视图不给源张量加 n_children，而是加 <span class="v">n_views</span>（第 739-742 行）：<span class="k">借用数据的人，和消费数据的人，要分开计数</span>。',
  '其余每一条 <span class="v">src[j]</span> 引用让 n_children 加一（第 754 行）。<span class="k">两个计数同时归零，才允许释放</span> —— 这是第二趟的唯一判据。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(4200 + i * 3400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(21000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

P7 = [(763, 822)]
N7 = {805: '★ 两个计数同时归零 = 这个张量再也不会被读到；不是视图就直接还给空闲表（第 816-817 行）'}

L.scene(
    kicker='L4-01 · 核心',
    title='★ 第二趟：按拓扑序分配，<span class="hl-b">用完就还</span>',
    sub='a -> mul_mat -> add -> out 这张小图跑一遍，看内存怎么在四个格子之间换手。',
    caption='分配在 763-778 行，释放判据在 805-819 行 —— 同一个循环里一进一出。',
    src=SRC, parts=P7, duration=21000,
    mark_src=[768, 778, 800, 805, 816], notes_src=N7,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip a">a</span>
    <span class="chip b">W</span>
    <span class="chip d">c</span>
    <span class="arrow">-></span>
    <span class="chip c">h = mul_mat(W, a)</span>
    <span class="arrow">-></span>
    <span class="chip d">y = add(h, c)</span>
    <span class="arrow">-></span>
    <span class="chip e">out</span>
  </div>
  <div class="cm" id="cap" style="margin:0">同一个计算 buffer 里的 4 格（示意；每格对应一次分配）</div>
  <div class="row" id="strip" style="gap:6px;justify-content:center;align-items:stretch"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const strip = wrap.querySelector('#strip');
const cells = [
  { t: 'x = a', s: '4096 B' },
  { t: 'W', s: '32768 B' },
  { t: 'c', s: '4096 B' },
  { t: 'h = mul_mat', s: '32768 B' }
];
const els = cells.map(c => {
  const e = U.el('div', { style: 'width:146px;height:46px;border:1px solid var(--border);' +
    'border-radius:5px;background:#10151b;display:flex;flex-direction:column;align-items:center;' +
    'justify-content:center;gap:1px;transition:all .45s' });
  e.innerHTML = '<div style="font-family:var(--mono);font-size:9.5px;color:var(--muted)">' + c.t + '</div>' +
    '<div style="font-family:var(--mono);font-size:9px;color:var(--dim)">' + c.s + '</div>';
  strip.appendChild(e);
  return e;
});

const msg = wrap.querySelector('#msg');
const texts = [
  '第一趟已经把账算好，第二趟只做三件事：<span class="k">给父张量分配 -> 给自己分配 -> 把父张量的 n_children 减一</span>。',
  '叶子先落位：<span class="v">x</span>、<span class="v">W</span>、<span class="v">c</span> 各占一格（第 725-727 行，带 INPUT 的另见第 744-746 行）。',
  '<span class="v">h = mul_mat(W, a)</span> 落位到第 4 格：从空闲表里取一块（第 687 行）。',
  'mul_mat 执行完，<span class="v">x</span> 与 <span class="v">W</span> 的 n_children 减到 0（第 800 行），第 816-817 行把它们还给空闲表；<span class="k">视图</span>走的是另一条分支 —— 它自己没占内存，只把源张量的 n_views 减一（第 806-814 行）。',
  '<span class="v">y = add(h, c)</span>：<span class="k">它和 h 同 layout，h 又只有它一个消费者</span>，于是直接接管 h 的地址（第 671-677 行），<b>0 新字节</b>。',
  '总账：这块 buffer 只需要 4 格。若每步各要一块、且不回收，同样这张图要 5 格起步 —— 而真实模型是每层都重复这一幕。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3400, () => {
  [0, 1, 2].forEach(i => { els[i].style.background = 'rgba(88,166,255,.18)'; els[i].style.borderColor = 'var(--a)'; });
  msg.innerHTML = texts[1];
});
tl.at(6800, () => { els[3].style.background = 'rgba(63,185,80,.18)'; els[3].style.borderColor = 'var(--b)'; msg.innerHTML = texts[2]; });
tl.at(10300, () => {
  [0, 1].forEach(i => { els[i].style.background = '#10151b'; els[i].style.borderColor = 'var(--dim)'; els[i].style.opacity = '.45'; });
  msg.innerHTML = texts[3];
});
tl.at(13800, () => {
  els[3].style.borderColor = 'var(--d)';
  els[3].querySelector('div').textContent = 'y = add（接管 h）';
  els[3].querySelectorAll('div')[1].textContent = '0 新字节';
  msg.innerHTML = texts[4];
});
tl.at(17600, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

P8 = [(623, 680)]
N8 = {627: '两个条件同时成立才分配：没被分配过，且不是视图 —— 视图永远不从这里要内存',
      651: 'layout 不同就不能原地复用 —— 尺寸与步长对不上',
      661: '三件事都满足才敢接管：源张量只有这一个视图、没有别的消费者、且视图正好覆盖它的全部数据'}

L.scene(
    kicker='L4-01 · 视图',
    title='视图：<span class="hl-e">跳过分配</span>，必要时连地址一起接管',
    sub='判据只有一行：view_src 非空就不进分配分支；能复用的时候连父张量那格都省下来。',
    caption='回顾 L1-01：RESHAPE / VIEW / TRANSPOSE 只改 ne[]、nb[]、view_src、view_offs，不复制数据。',
    src=SRC, parts=P8, duration=21000,
    mark_src=[627, 657, 661, 671], notes_src=N8,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row center" style="gap:10px" id="dia"></div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const dia = wrap.querySelector('#dia');
const base = U.el('div', { class: 'card', style: 'width:286px;border-left-color:var(--a)' });
base.innerHTML = '<div class="ct" style="color:var(--a)">基张量 h</div>' +
  '<div class="cb">拥有 data 与 buffer<br><span class="cm" style="margin:0">view_src == NULL</span></div>' +
  '<div style="display:flex;gap:2px;margin-top:6px" id="cells"></div>';
const view = U.el('div', { class: 'card', style: 'width:286px;border-left-color:var(--e)' });
view.innerHTML = '<div class="ct" style="color:var(--e)">视图 v = ggml_view_2d(...)</div>' +
  '<div class="cb">占 0 字节：它没有自己的内存<br><span class="cm" style="margin:0">view_src = h</span><br>' +
  '<span class="cm" style="margin:0">view_offs = 0</span></div>';
dia.appendChild(base); dia.appendChild(U.arrow('->')); dia.appendChild(view);

const cells = base.querySelector('#cells');
const cs = [];
for (let i = 0; i < 6; i++) {
  const c = U.el('div', { style: 'width:36px;height:17px;border:1px solid var(--border);border-radius:3px;' +
    'background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
  c.textContent = i;
  cells.appendChild(c); cs.push(c);
}

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: '判据一：跳过分配', b: 'view_src 非空 -> 第 627 行整个分配分支不进', m: '!ggml_impl_is_view(node)' },
  { c: 'b', t: '判据二：接管父张量', b: 'n_children == 1 && n_views == 0 且同 layout', m: 'p_hn->n_children == 1' },
  { c: 'c', t: '判据三：接管视图的源', b: '源只有这一个视图、没有消费者、且视图覆盖全部数据', m: 'view_src->data == parent->data' }
];
const els2 = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '219px' }); host.appendChild(e); return e; });
els2.forEach(e => { e.style.opacity = '.30'; });

const msg = wrap.querySelector('#msg');
const texts = [
  '视图不拥有内存，所以分配器对它的处理和普通张量完全不同：<span class="k">它一进来就被排除在分配之外</span>。',
  '但这带来一个麻烦：视图<b>借住</b>在源张量身上，源张量就不能随便被回收 —— 所以第一趟要单独数 <span class="v">n_views</span>（第 739-742 行）。另外，带 OUTPUT 标记的父张量（以及它的视图）一律不许被覆盖（第 646 行）。',
  '更妙的是第 651-677 行：如果 h 只有 v 一个视图、v 只有 add 一个消费者、layout 又一致，<span class="k">add 的输出可以直接接管 h 的地址</span>（第 665 与 674 行）—— 前提是这个算子在 <span class="v">ggml_op_can_inplace</span> 的白名单里（第 22-51 行，ADD / MUL / RMS_NORM / SOFT_MAX ...）。',
  '接管时把 <span class="v">p_hn-&gt;allocated</span> 置为 false（第 666-667 行），等于告诉后面的释放逻辑"这块不是你的，别还" —— 否则会把正在用的地址还进空闲表。',
  'L1-01 讲的是视图<span class="k">不复制数据</span>；这里补上后半句：<span class="k">视图链上的地址可以在算子之间整块转让</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, @@A@@); });
tl.at(4500, () => { msg.innerHTML = texts[1]; U.markLines(document, []); });
tl.at(8600, () => {
  [0, 1, 2, 3, 4, 5].forEach(i => { cs[i].style.background = 'rgba(88,166,255,.18)'; cs[i].style.borderColor = 'var(--a)'; });
  msg.innerHTML = texts[2]; U.markLines(document, @@B@@);
});
tl.at(13000, () => { msg.innerHTML = texts[3]; U.markLines(document, @@C@@); });
tl.at(16800, () => { els2.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, []); });
'''.replace('@@A@@', ridx(P8, N8, [627]))
   .replace('@@B@@', ridx(P8, N8, [657, 661]))
   .replace('@@C@@', ridx(P8, N8, [665, 666, 667]))
)

# ------------------------------------------------------------------ 第 9 幕

P9 = [(1052, 1098)]
N9 = {1053: 'needs_realloc 逐节点比对：节点数、叶子数、每个张量需要的字节数（第 1009-1050 行）',
      1054: '★ 单 buffer 在第 1058 行自动重规划；多 buffer 在第 1061-1065 行直接返回 false，要求先调 reserve_n',
      1069: '命中复用：只把 buffer 清一遍，不重新算任何 offset',
      1076: '然后照上一次的规划结果逐个落位（init_tensor，第 970-995 行）'}

L.scene(
    kicker='L4-01 · 图重放',
    title='★ 图没变，就<span class="hl-f">不重新规划</span>',
    sub='alloc_graph 先问 needs_realloc：节点数、叶子数、每个张量的大小都没变，就直接用上次的规划落位。',
    caption='呼应 L2-07 的 graph reuse：decode 每一步形状相同 -> 图一样 -> 内存规划也一样，于是每步只做一次"发地址"。',
    src=SRC, parts=P9, duration=20000,
    mark_src=[1053, 1054, 1059, 1069, 1081], notes_src=N9,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div>
  <div class="row" style="gap:9px">
    <div class="col grow" style="gap:6px" id="l"></div>
    <div class="col grow" style="gap:6px" id="r"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['图变了没有', 'needs_realloc', 'alloc_graph 会做什么'],
  [['节点数 / 叶子数变了', 'true', '单 buffer 自动 reserve 重规划；多 buffer 返回 false，要你先调 reserve_n'],
   ['某个张量变大了', 'true', '同上 —— 老的 size_max 已经装不下'],
   ['完全一样（decode 的常态）', 'false', '只清一遍 buffer 再 init_tensor 落位，0 次规划']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#l').innerHTML =
  '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">第一次 / 图变了</div>' +
  '<div class="cb">reserve：清空 dyn_tallocr -> 两趟规划 -> 按水位线开 buffer。<br>' +
  '<span class="cm" style="margin:0">ggml_gallocr_reserve_n_impl</span></div></div>';
wrap.querySelector('#r').innerHTML =
  '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">之后每一步</div>' +
  '<div class="cb">alloc_graph：比一遍 -> 清 buffer -> 落位。<br>' +
  '规划结果一直存在 node_allocs / leaf_allocs 里，<br>' +
  '<span class="cm" style="margin:0">重放只花一次比对的代价</span></div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  'alloc_graph 的第一件事不是分配，是<b>先问一句"要不要重来"</b>（第 1053 行）。',
  '<span class="v">needs_realloc</span> 会逐节点问：老的 <span class="v">size_max</span> 还装得下你吗（第 997-1007 行）？只要有一处不满足，就整体重规划。',
  '单 buffer 时它替你重规划（第 1058 行）；<span class="k">多 buffer 时不猜</span>，直接返回 false（第 1061-1065 行）—— 因为节点该放哪个 buffer 是调度器的决定（L4-02）。',
  '命中复用后只做两件事：<span class="v">ggml_vbuffer_reset</span> 清一遍（第 1069-1074 行），再对每个叶子与节点调 <span class="v">ggml_gallocr_init_tensor</span> 落位（第 1076-1095 行）。',
  '这就是 L2-07 里 graph reuse 那笔账的另一半：<span class="k">图复用省掉建图，内存复用省掉规划</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3600 + i * 2900, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 3)];
}));
tl.at(17500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

P10 = [(70, 74)]

L.scene(
    kicker='L4-01 · 收束',
    title='内存规划的输出，与这一课的一张表',
    sub='get_buffer_size 报出的那个数，就是这张图的执行缓冲大小；多 buffer 时按 buffer 分别报。',
    caption='下一课 L4-02：调度器怎么决定每个节点进哪个 buffer，以及为什么切分处必须插 copy 节点。',
    src=HSRC, parts=P10, duration=20000,
    mark_src=[70, 71, 74],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', '答案', '证据（ggml-alloc.c 行号）'],
  [['为什么要自己管理内存？', '生命周期在建图时就已知 -> 一次规划出峰值，而不是每个张量各占一块', '723-761 / 763-822'],
   ['内存怎么复用？', 'free_block 空闲表 + best fit 取块；还块时与左右邻居就地合并', '110-133 / 202-224 / 311-350'],
   ['视图怎么处理？', 'view_src 非空 -> 跳过分配；条件满足时把源张量的地址整块转让', '627 / 657-670'],
   ['图重放为什么不用重规划？', 'needs_realloc 比对图形态，没变就只用上次的结果落位', '1009-1050 / 1052-1098'],
   ['多 buffer 怎么办？', 'ggml_gallocr_new_n + 每个节点的 buffer_id，交给 L4-02 的调度器', '482-496 / 498-532']],
  { monoCols: [2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '给定小图 <span class="mono">a -> h = mul_mat(W, a) -> y = add(h, c) -> out</span>' +
  '（a 是输入，W / c 是叶子，out 带 OUTPUT 标记）。' +
  '① 为什么 <span class="mono">y</span> 可以不要新内存？' +
  '② <span class="mono">a</span> 和 <span class="mono">W</span> 在哪一刻被还给空闲表？' +
  '③ 为什么"每个张量各 ggml_new_tensor 一块"在小图上没问题，在 70 层 transformer 上会爆？',
  '① <span class="mono">ggml_op_can_inplace(GGML_OP_ADD)</span> 为真（第 22-51 行），' +
  '而 <span class="mono">h</span> 只被 <span class="mono">add</span> 一个节点引用（n_children == 1）、没有视图（n_views == 0）、' +
  'layout 又与输出一致（第 651 行）—— 于是第 671-677 行直接把 <span class="mono">h</span> 的地址转给 ' +
  '<span class="mono">y</span>，并在第 675 行把 <span class="mono">p_hn-&gt;allocated</span> 置 false，免得它被当成"可释放"。<br><br>' +
  '② 执行完 <span class="mono">mul_mat</span> 之后，第 800 行的 <span class="mono">p_hn-&gt;n_children -= 1</span> 把它们的计数减到 0，' +
  '第 816-817 行随即调用 <span class="mono">ggml_gallocr_free_node</span> 把地址还给空闲表（a、W 都是叶子，只会被引用这一次）。<br><br>' +
  '③ 小图里"各占一块"只浪费一格；70 层里每一层都有若干中间张量，它们<b>不同时活着</b>，' +
  '而"各占一块"只按总和算 —— 总和对峰值是数量级的差距。' +
  'gallocr 用空闲表把死者让出的洞立刻发给下一个人（第 263-268 行取块、第 311-350 行还块），' +
  '所以 buffer 只需要按<b>同时存活的峰值</b>开一次。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '五句话收束这一课：',
  '<span class="k">为什么</span>：生命周期已知 -> 可以提前规划（723-761 数账，763-822 分配与释放）。',
  '<span class="k">怎么复用</span>：free_block 空闲表 + best fit 取块 + 就地合并还块。',
  '<span class="k">视图</span>：跳过分配；能整体转让时连源张量那一格都省下。',
  '<span class="k">代价</span>：每次图形态变化都要重规划 —— 单 buffer 自动做（第 1058 行），多 buffer 必须你自己先 reserve_n。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、两张 API 面：tallocr 与 gallocr',
    '`ggml-alloc.h` 开头就把分配器分成两种。上面那个 `struct ggml_tallocr` 是**线性分配器**：'
    '一个 buffer、一个 base、一个 alignment、一个 offset，全部状态就这四个字段；'
    '配套只有"新建"和"分配"两个函数，**没有释放**。'
    '下面那个 `ggml_gallocr` 是**图分配器**，本课的主角。',
    src=HSRC, parts=[(13, 22)], lang='c')

L.section(
    '二、图分配器的标准用法',
    '这是 `ggml-alloc.h` 里那段 `Example usage` 注释，也是本课全部内容的路线图：'
    '`ggml_gallocr_new` 选 buffer 类型，`ggml_gallocr_reserve` 用最坏情况图先规划，'
    '`ggml_gallocr_alloc_graph` 对当前图落位，`ggml_gallocr_get_buffer_size` 报告结果，'
    '最后由后端 `ggml_backend_graph_compute` 执行。'
    '注释把 `reserve` 标为 optional：不调也能跑，只是可能在运行时触发重新分配。',
    src=HSRC, parts=[(24, 40)], lang='c')

L.section(
    '三、两个特殊 flag',
    '分配器只为两类张量开了后门：`ggml_set_input()` 的张量**先落位且地址互不重叠**'
    '（它们在图外面被写数据），`ggml_set_output()` 的张量**永不被释放、也不会被覆盖**'
    '（图算完之后外面还要读它）。这两条正是 `ggml_gallocr_free_node` 里'
    '`GGML_TENSOR_FLAG_OUTPUT` 判断的依据。',
    src=HSRC, parts=[(42, 44)], lang='c')

L.section(
    '四、线性分配器 ggml_tallocr：只进不退',
    '这就是"不复用"的完整实现。`ggml_tallocr_alloc` 先向后端问"这个张量要占多少字节"'
    '（同一个张量在不同后端上大小可能不同），对齐后检查是否越界，然后'
    '`talloc->offset += size` 把水位线往前推。**没有任何路径能让 offset 退回去**。'
    '所以它的内存账是"所有张量之和"，而不是"同时存活的峰值"。',
    src=SRC, parts=[(59, 92)], lang='c')

L.section(
    '五、★ free_block：复用的数据结构',
    '动态分配器 `ggml_dyn_tallocr` 不记录"哪个张量住在哪"，只记录**哪些字节是空的**。'
    '空闲块就是 `{offset, size}` 两个数（`struct free_block`），按地址有序地放在 '
    '`tallocr_chunk::free_blocks[MAX_FREE_BLOCKS]` 里。'
    '`max_size` 是这个 chunk 的水位线 —— 规划结束后 buffer 就按它开。'
    '`chunks[GGML_VBUFFER_MAX_CHUNKS]` 让一个分配器横跨多个后端 buffer，这是 L4-02 的基础。',
    src=SRC, parts=[(110, 133)], lang='c')

L.section(
    '六、★ best-fit：从空闲表里挑一块',
    '分配策略是 **best fit**：在所有装得下的空闲块里挑最小的那个。'
    '注意内层循环上界写成 `chunk->n_free_blocks - 1` —— **最后一个空闲块故意不参与挑选**，'
    '因为它代表"尚未使用的水位线以上"，用它意味着把 buffer 撑大（第 226-247 行才轮到它）。'
    '取块后把 `block->offset` 让出去、`block->size` 减掉，减到 0 就把这一项从表里删掉。',
    src=SRC, parts=[(202, 224)], lang='c')

L.section(
    '七、还块：与相邻空闲块合并',
    '还块时先看能不能"贴"在已有空闲块的尾部或头部，能贴就长大，'
    '长大之后顺手再看一眼另一侧的邻居能不能一起吞掉。'
    '两个方向都不沾，才新开一个空闲块 —— 这是空闲表变碎的唯一来源，'
    '也是 `MAX_FREE_BLOCKS`（第 14 行，256）这道上限存在的理由。',
    src=SRC, parts=[(311, 350)], lang='c')

L.section(
    '八、★ 第一趟：把"谁还要用我"数出来',
    '`ggml_gallocr_alloc_graph_impl` 的第一趟不分配，只做两件事：'
    '把所有叶子落位（它们不属于任何一次"执行"，可以认为活到最后），'
    '然后遍历每个节点的 `src[]`，把 `n_children` 加一；碰到视图则给源张量加 `n_views`。'
    '**这两个计数就是"生命周期"的量化形式**，也是第二趟能否释放的唯一判据。'
    '带 `GGML_TENSOR_FLAG_INPUT` 的张量在这里就抢先落位，避免地址被别人占掉。',
    src=SRC, parts=[(723, 761)], lang='c')

L.section(
    '九、第二趟：按拓扑序分配，用完就还',
    '第二趟沿 `graph->nodes[]` 的**拓扑序**走（这正是 L1-03 讲的顺序）：'
    '先保证父张量有地址，再分配自己，然后把每个父张量的 `n_children` 减一。'
    '当 `n_children == 0 && n_views == 0` 时，这个张量再也不会被读到，'
    '于是要么把源张量的 `n_views` 减一（视图的情况），要么直接 `ggml_gallocr_free_node`。'
    '**分配与释放共用同一个循环**：一进一出，水位线就被压在峰值上。',
    src=SRC, parts=[(763, 822)], lang='c')

L.section(
    '十、free_node：输出张量永不释放',
    '释放路径上的第一道判断是 `GGML_TENSOR_FLAG_OUTPUT`：图输出直接 return，'
    '既不还块也不改 `allocated`。注释写得很直白 —— graph outputs are never freed。'
    '其余张量走 `ggml_dyn_tallocr_free_bytes` 把地址还给空闲表。',
    src=SRC, parts=[(691, 712)], lang='c')

L.section(
    '十一、视图：跳过分配，必要时接管地址',
    '`ggml_gallocr_allocate_node` 的入口就是一条硬判据：`!ggml_impl_is_view(node)`。'
    '视图不拥有数据，所以永不从这里申请内存。'
    '更进一步：如果父张量只有 1 个消费者、0 个视图且 layout 与当前节点一致，'
    '当前节点可以直接**接管父张量的地址**（`hn->addr = p_hn->addr`），'
    '并把 `p_hn->allocated` 置 false 防止它被误释放。'
    '视图的情形还要额外检查 `view_src->data == parent->data`，确认视图正好覆盖源张量的全部数据。'
    '这一段呼应 L1-01 的 `view_src` / `view_offs`：零拷贝不只是省掉一次复制，'
    '还让分配器少发一块内存。',
    src=SRC, parts=[(623, 680)], lang='c')

L.section(
    '十二、图重放：needs_realloc 决定要不要重新规划',
    '每次 `ggml_gallocr_alloc_graph` 都会先比一遍：节点数、叶子数、以及每个张量'
    '在规划时记下的 `size_max` 还够不够用。只要有一处不满足就返回 true。'
    '这就是"图形态没变就不用重规划"的实现 —— decode 每一步形状相同，'
    '所以这一步在稳态下几乎不花钱（L2-07 的 graph reuse 是它的上游）。',
    src=SRC, parts=[(1009, 1050)], lang='c')

L.section(
    '十三、落位：init_tensor 把规划结果发给张量',
    '规划的产物是 `buffer_id` + `struct buffer_address`（chunk 下标 + 块内偏移），'
    '存在 `node_allocs` / `leaf_allocs` 里。`ggml_gallocr_init_tensor` 负责把这两个数'
    '翻译成张量的 `buffer` 与 `data`：普通张量走 `ggml_vbuffer_tensor_alloc`，'
    '视图走 `ggml_backend_view_init`（它自己不需要新内存，只要把地址算出来）。',
    src=SRC, parts=[(970, 995)], lang='c')

L.section(
    '十四、图分配器的状态与多 buffer 版本',
    '`struct ggml_gallocr` 的字段回答了"规划结果存在哪"：'
    '`buf_tallocs[n_buffers]` 是每个 buffer 一个动态分配器，'
    '`node_allocs` / `leaf_allocs` 是逐节点、逐叶子的落位表，'
    '`hash_set` + `hash_values` 是"张量 -> hash_node"的查找结构。'
    '`ggml_gallocr_new_n` 里的内层循环还做了一件小事：'
    '**同一个 buffer type 被给多次时共用同一个分配器**，避免同一块类型的内存被重复计算。',
    src=SRC, parts=[(482, 532)], lang='c')

L.section(
    '十五、另一条路径（一）：在一段 buffer 里顺序摆张量',
    '不是所有张量都属于某张图。权重、KV cache 这类"整场都活着"的东西走另一条路：'
    '`alloc_tensor_range` 开一个 buffer，用第四节那个线性 `ggml_tallocr` 沿着 context 的'
    '张量链一个挨一个摆下去。注意它对视图的处理与图分配器一致：'
    '`t->view_src == NULL` 才调 `ggml_tallocr_alloc`，否则只做 `ggml_backend_view_init`。'
    '已经带 data 的张量直接跳过 —— 这就是"不要重复分配别人的内存"。',
    src=SRC, parts=[(1127, 1166)], lang='c')

L.section(
    '十六、另一条路径（二）：装不下就换一个 buffer',
    '`ggml_backend_alloc_ctx_tensors_from_buft_impl` 沿 context 的张量链累加大小：'
    '一旦 `cur_buf_size + this_size` 超过这一种 buffer type 的 `max_size`，'
    '就把前面那一段交给 `alloc_tensor_range` 开出去，从头开始累加。'
    '因为视图与已分配的张量大小为 0（第 1183-1185 行），它们不会把 buffer 撑大。'
    '最后 `n_buffers > 1` 时用 `ggml_backend_multi_buffer_alloc_buffer` 合成一个句柄。',
    src=SRC, parts=[(1168, 1230)], lang='c')

L.section(
    '十七、规划的输出：ggml_gallocr_get_buffer_size',
    '`ggml_gallocr_get_buffer_size` 返回的就是"这张图要多少执行缓冲"。'
    '它读的是 `ggml_vbuffer_size`，也就是各 chunk 水位线之和。'
    '注意里面那个去重循环：同一个 buffer type 被注册多次时，'
    '只有第一次出现才会报数，避免重复计入。',
    src=SRC, parts=[(1100, 1116)], lang='c')

L.footnote_add('本课只用 `ggml/src/ggml-alloc.c`（1249 行）与 `ggml/include/ggml-alloc.h`（86 行）'
               '两个文件，均计入覆盖率。')
L.footnote_add('课件里提到的 `ggml_op_can_inplace`（第 22-51 行）、`MAX_FREE_BLOCKS`（第 14 行）、'
               '`GGML_VBUFFER_MAX_CHUNKS`（第 96 行）、`ggml_gallocr_reserve_n_impl`（第 825 行起）、'
               '`ggml_gallocr_node_needs_realloc`（第 997-1007 行）都在这两个文件里，'
               '属于"提到但不逐字引用"的行，不计入引用块数。')
L.footnote_add('验收点"为什么 ggml 需要自己管理内存而不是逐个 ggml_new_tensor"的答案在第 2 幕：'
               '线性分配器只有一个单调递增的 offset（第 87 行），没有释放路径；'
               '而图分配器知道每个张量的完整生命周期，可以把死者的内存立刻发给下一个人。')

L.prereqs('`L1-01`、`L1-03`、`L1-05`、`L2-07`、`L3-04`')

L.goal(
    '说出 `ggml_tallocr`（线性）与 `ggml_dyn_tallocr`（动态）的区别，以及为什么前者不够用（对应验收点）；',
    '指出内存复用的三件套：`free_block` 空闲表、best fit 取块、还块时与邻居合并；',
    '解释 `n_children` / `n_views` 两个计数如何决定一个张量什么时候可以被释放；',
    '判断一个张量是不是视图，并说出分配器对视图的两种处理（跳过分配 / 接管源张量地址）；',
    '说明图重放时 `ggml_gallocr_alloc_graph` 为什么可以不重新规划，以及多 buffer 时为什么不行。')

L.conclusion(
    '★ 为什么必须自己管理内存',
    '因为**生命周期在建图时就已知**。图的拓扑序（L1-03）给出了每个张量的"出生"与"死亡"，'
    '于是分配器可以：先数一遍谁还要用谁（`n_children` / `n_views`），'
    '再沿拓扑序分配，并在计数归零时立刻把地址还给空闲表。\n\n'
    '```text\n'
    '逐个分配的内存账  = 所有张量大小之和\n'
    '规划后的内存账    = 同时存活张量的峰值\n'
    '```\n\n'
    '线性分配器 `ggml_tallocr` 只有一个单调递增的 `offset`（第 87 行），没有还块路径，'
    '所以它只能算前者。这就是 `ggml-alloc.c` 存在的理由。')

L.conclusion(
    '★ 复用的三个动作',
    '| 动作 | 实现 | 位置 |\n|---|---|---|\n'
    '| 取块 | best fit：装得下且最小 | `ggml_dyn_tallocr_alloc`，202-224 |\n'
    '| 还块 | 与相邻空闲块就地合并 | `ggml_dyn_tallocr_free_bytes`，311-350 |\n'
    '| 原地 | 同 layout 且父张量只剩一个消费者时直接换手 | `ggml_gallocr_allocate_node`，651-677 |\n\n'
    '第三个动作是"零字节"的：`y = add(h, c)` 的输出直接住在 `h` 的地址上。')

L.conclusion(
    '视图的两条规则',
    '判据是 `view_src != NULL`（`ggml_impl_is_view`）。于是：\n\n'
    '1. **跳过分配**：`ggml_gallocr_allocate_node` 的第一行判断就把它排除在外（第 627 行）；\n'
    '2. **单独计数**：视图给源张量加的是 `n_views` 而不是 `n_children`（第 739-742 行），'
    '两个计数同时归零才允许释放。\n\n'
    '这补上了 L1-01 的另一半：视图不复制数据，而且**视图链上的地址可以在算子之间整块转让**。')

L.conclusion(
    '代价：图变了就得重规划',
    '`ggml_gallocr_needs_realloc`（1009-1050）逐节点比对图形态与规划时记录的 `size_max`。'
    '变了以后：单 buffer 由 `ggml_gallocr_alloc_graph` 自动重规划（第 1058 行），'
    '**多 buffer 直接返回 false**（第 1061-1065 行），要求调用方先调 `ggml_gallocr_reserve_n` —— '
    '因为节点该进哪个 buffer 是调度器的决定，不是分配器能猜的。\n\n'
    '这正是 L4-02 的入口：调度器负责切分与 buffer 归属，分配器负责在给定归属下把地址排到最紧。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
