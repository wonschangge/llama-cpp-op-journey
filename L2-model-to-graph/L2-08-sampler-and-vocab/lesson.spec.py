#!/usr/bin/env python3
"""L2-08 · 采样器与词表 —— 课件 spec。

视角：图算出 logits 之后，谁把 logits 变成 token？
答案分两半 —— 入口的**词表**（文本 <-> token）与出口的**采样器**（logits -> token），
两者都跑在 host 上，都不进图。

运行：python3 L2-model-to-graph/L2-08-sampler-and-vocab/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SAMPLER_C = 'src/llama-sampler.cpp'
SAMPLER_H = 'src/llama-sampler.h'
VOCAB_C = 'src/llama-vocab.cpp'
VOCAB_H = 'src/llama-vocab.h'
UNI_C = 'src/unicode.cpp'
UNI_H = 'src/unicode.h'
UNI_DATA_C = 'src/unicode-data.cpp'
UNI_DATA_H = 'src/unicode-data.h'

L = Lesson(
    id='L2-08',
    layer='L2 · 从模型到图',
    title='采样器与词表：图算 logits，host 选 token',
    codecap='src/llama-sampler.cpp / llama-vocab.cpp / unicode.cpp（逐字引用）',
    nav={'prev': {'href': '../L2-07-context-and-decode/index.html',
                  'label': 'L2-07 ★ 上下文与解码'},
         'next': {'href': '../L2-09-quant-export-adapter/index.html',
                  'label': 'L2-09 量化、导出与适配器'}},
)

L.cover(SAMPLER_C, SAMPLER_H, VOCAB_C, VOCAB_H, UNI_C, UNI_H, UNI_DATA_C, UNI_DATA_H)

L.note('**一句话**：`llama_decode()` 在图上算完，交出来的是 **logits**（每个 token 一个分数，'
       '长度 = 词表大小）。把它变成一个 token 的那一步叫**采样**：它发生在 **host（CPU）** 上，'
       '操作的是一个 `llama_token_data_array`（token id / logit / p 的数组），'
       '**不产生任何 ggml 算子**。')
L.note('这一课同时补上模型的另一头：文本要先被**词表**切成 token 才能进图。'
       '分词（BPE 合并、unicode 规范化与类别判断）同样跑在 host 上，'
       '靠的是 `unicode-data.cpp` 里 7000 行生成好的码点表。')
L.note('读法建议：前 4 幕走"入口"（词表 / BPE / unicode），后 5 幕走"出口"（采样链 / 顺序 / 为何不进图）。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2 · 从模型到图',
    title='模型的两端：<span class="hl-a">分词</span>与<span class="hl-d">采样</span>',
    sub='图只认 token id。进来时要分词，出去时要采样 —— 两端都在 host 上，都不在图里。',
    caption='回顾 L2-07：decode 建图、算图，产出 logits；本课接在它的后面。',
    src=VOCAB_H, parts=[(165, 187)], duration=17000,
    marks=[0, 8, 14],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">文本</span><span class="arrow">-></span>
    <span class="chip a">tokenize</span><span class="arrow">-></span>
    <span class="chip b">图 decode</span><span class="arrow">-></span>
    <span class="chip c">logits</span><span class="arrow">-></span>
    <span class="chip d">sampler</span><span class="arrow">-></span>
    <span class="chip e">token</span>
  </div>
  <div class="row center" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '入口 · 词表', b: '文本 <-> token id。<br>tokenize() 与 token_to_piece() 是一对互逆操作。',
    m: 'src/llama-vocab.h' },
  { c: 'd', t: '出口 · 采样器', b: 'logits -> token id。<br>在 host 上按链式顺序逐步裁剪分布。',
    m: 'src/llama-sampler.cpp' },
  { c: 'f', t: '共同底座 · unicode', b: '码点类别、NFD 折叠、大小写映射<br>都是生成好的静态数据表。',
    m: 'src/unicode-data.cpp' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '模型只吃 <span class="k">token id</span>。左边必须有人把文本切碎，右边必须有人把分数变成 id。',
  '入口：<span class="v">tokenize()</span> 把文本变成 token 数组；<span class="v">token_to_piece()</span> 反过来把 token 变回文本片段。',
  '中间：图（L2-07）只做一件事 —— 把 token id 变成 logits。它对"文本"和"概率"一无所知。',
  '出口：采样器把 logits 变成<b>下一个</b> token id，再喂回图。这就是自回归的回路。',
  '本课主线：<span class="k">这两端为什么都在 host 上做，而不进图？</span>'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-08 · 词表',
    title='分词第一步：<span class="hl-c">特殊 token 先切开</span>，再按类型分派',
    sub='tokenize() 先做一次切分（BOS/EOS 这类特殊 token），再把每段交给具体 tokenizer。',
    caption='分派表里 7 种 tokenizer：SPM / BPE / WPM / UGM / RWKV / PLAMO2 / TEST。本课细看 BPE。',
    src=VOCAB_C, parts=[(3413, 3427), (3214, 3235)], duration=19000,
    marks=[0, 11, 16, 24],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">raw_text</span><span class="arrow">-></span>
    <span class="chip c">tokenizer_st_partition</span><span class="arrow">-></span>
    <span class="chip a">按 vocab 类型分派</span><span class="arrow">-></span>
    <span class="chip e">token id[]</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['vocab 类型', 'tokenizer 实现', '说明'],
  [['LLAMA_VOCAB_TYPE_SPM', 'llm_tokenizer_spm', 'SentencePiece；空串特例见源码注释'],
   ['LLAMA_VOCAB_TYPE_BPE', 'llm_tokenizer_bpe', '本课重点：按 merges 的 rank 合并'],
   ['LLAMA_VOCAB_TYPE_WPM', 'llm_tokenizer_wpm', 'word-piece'],
   ['LLAMA_VOCAB_TYPE_UGM', 'llm_tokenizer_ugm', '需要 precompiled_charsmap'],
   ['LLAMA_VOCAB_TYPE_RWKV', 'llm_tokenizer_rwkv', 'RWKV 世界模型'],
   ['LLAMA_VOCAB_TYPE_PLAMO2', 'llm_tokenizer_plamo2', 'PLAMO2'],
   ['LLAMA_VOCAB_TYPE_TEST', 'llm_tokenizer', '测试用的基类']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');
rows.forEach(r => { r.style.opacity = '.35'; });

const msg = wrap.querySelector('#msg');
const texts = [
  '分词不是一个函数，而是<b>三段</b>：切分 -> 分派 -> 各类型自己的算法。',
  '<span class="v">fragment_buffer</span>：先把整段文本装进一个前向链表，切成"片段"。',
  '<span class="v">tokenizer_st_partition()</span>：特殊 token（BOS/EOS、控制符、用户自定义标记）在这里被<b>优先</b>摘出来，剩下的才是原始文本。',
  '<span class="v">switch (get_type())</span>：按 vocab 类型选一条实现。7 条路各有各的算法。',
  '下面三幕只看 <span class="k">BPE</span> 这一条 —— 它是当前主流 LLM 的分词方式。'
];
let t0 = 700;
tl.at(t0, () => { msg.innerHTML = texts[0]; });
tl.at(t0 + 3000, () => { msg.innerHTML = texts[1]; U.markLines(document, [0]); });
tl.at(t0 + 6000, () => { msg.innerHTML = texts[2]; U.markLines(document, [11]); });
rows.forEach((r, i) => tl.at(t0 + 9000 + i * 1200, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = (i === 1) ? texts[3] : '第 ' + (i + 1) + ' 条路：' + r.querySelector('td').textContent;
  if (i === 1) U.markLines(document, [16, 24]);
}));
tl.at(17800, () => { rows.forEach(x => { x.className = ''; x.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [24]); });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-08 · BPE',
    title='★ BPE 的合并顺序由一个 <span class="hl-a">rank</span> 决定',
    sub='llm_bigram_bpe 用优先队列按 rank 排序：rank 最小的相邻对最先合并，合并后只补两个新 bigram。',
    caption='rank 来自词表的 merges 表（find_bpe_rank，见 source.md）；它随模型一起存进 GGUF，见 L1-06。',
    src=VOCAB_C, parts=[(264, 278), (657, 689)], duration=23000,
    marks=[0, 2, 21, 37],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row center" id="sym" style="gap:3px"></div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="q" style="gap:5px"></div>
    <div class="col grow" id="log" style="gap:5px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const sym = wrap.querySelector('#sym');
const cells = [];
['l', 'o', 'w', 'e', 's', 't'].forEach((ch, i) => {
  const c = U.el('div', { style: 'width:34px;height:26px;border:1px solid var(--border);border-radius:3px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:12px;color:var(--text)' });
  c.textContent = ch;
  sym.appendChild(c); cells.push(c);
});
const q = wrap.querySelector('#q');
q.innerHTML = '<div class="cm" style="margin:0 0 2px">优先队列（rank 小的先出）</div>';
const qrows = [
  { t: '(e,s)  rank 4', c: 'a' }, { t: '(l,o)  rank 7', c: 'b' }, { t: '(s,t)  rank 9', c: 'c' }
];
const qels = qrows.map(r => {
  const e = U.el('div', { class: 'formula', style: 'padding:3px 7px;font-size:10px' });
  e.innerHTML = '<span style="color:var(--' + r.c + ')">' + U.esc(r.t) + '</span>';
  q.appendChild(e); return e;
});
const log = wrap.querySelector('#log');
log.innerHTML = '<div class="cm" style="margin:0 0 2px">合并后新增的 bigram</div>' +
  '<div class="cb" id="lg" style="font-size:9.5px;color:var(--muted)">（还没开始）</div>';
const lg = log.querySelector('#lg');

const msg = wrap.querySelector('#msg');
const texts = [
  '把词先按<b>字符</b>拆成一串 symbol —— 这是 BPE 的起点。',
  '<span class="v">add_new_bigram()</span> 给每对相邻 symbol 查一次 <span class="k">find_bpe_rank()</span>，查得到才入队。',
  '<span class="v">rank 小的先合并</span>：队列按 rank 排序（rank 相同再比左端位置）。示例里的 rank 是示意值。',
  '<span class="v">(e,s) 合并成 es</span>。被合并的右半标记为 n = 0（出局）。',
  '合并只影响两侧：<span class="k">只补两个新 bigram</span>，不重扫全部 —— 这就是 BPE 快的来源。',
  '收束：<span class="k">分词的顺序完全由词表里的 rank 决定</span>，不由代码的循环顺序决定。'
];
qels.forEach(e => e.style.opacity = '.30');
const fill = ['', '', ''];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [16]); });
tl.at(7100, () => {
  msg.innerHTML = texts[2]; U.markLines(document, [0, 2]);
  qels.forEach((e, i) => { e.style.opacity = i === 0 ? '1' : '.30'; });
});
tl.at(10300, () => {
  msg.innerHTML = texts[3]; U.markLines(document, [37]);
  cells[4].style.background = 'rgba(88,166,255,.25)'; cells[4].style.borderColor = 'var(--a)';
  cells[5].style.opacity = '.30'; cells[5].style.textDecoration = 'line-through';
  lg.textContent = '(l,es)  (es,t)';
});
tl.at(13500, () => {
  msg.innerHTML = texts[4];
  qels.forEach(e => { e.style.opacity = '.30'; });
  lg.textContent = '(l,es)  (es,t)  ->  再合并 (es,t)，只补 (l,est) 与 (est,?) 两个';
});
tl.at(17500, () => { msg.innerHTML = texts[5]; U.markLines(document, []); });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-08 · unicode',
    title='unicode：<span class="hl-f">类别判断</span>与 <span class="hl-f">NFD 折叠</span>靠数据表',
    sub='分词器要回答"这个字符是字母还是数字、要不要折叠重音"，答案是三张查表函数 + 五张生成表。',
    caption='unicode-data.cpp 第 1 行写明：generated with scripts/gen-unicode-data.py —— 它是生成物，不要手改。',
    src=UNI_C, parts=[(16, 20), (30, 47), (1117, 1127), (1147, 1151)], duration=21000,
    marks=[0, 6, 25, 37],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">UTF-8 字节</span><span class="arrow">-></span>
    <span class="chip a">unicode_cpt_from_utf8</span><span class="arrow">-></span>
    <span class="chip f">unicode_cpt_flags_from_cpt</span><span class="arrow">-></span>
    <span class="chip c">unicode_cpts_normalize_nfd</span>
  </div>
  <div id="bars" class="bars"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#bars');
const tables = [
  { l: 'ranges_flags', n: 2273, c: 'a' },
  { l: 'map_lowercase', n: 1433, c: 'b' },
  { l: 'map_uppercase', n: 1450, c: 'd' },
  { l: 'ranges_nfd', n: 1828, c: 'c' },
  { l: 'set_whitespace', n: 25, c: 'e' }
];
const bars = tables.map(t => { const b = U.bar(t.l, t.c); host.appendChild(b.el); return b; });
bars.forEach((b, i) => {
  b.fill.style.width = (tables[i].n / 2273 * 100) + '%';
  b.val.textContent = tables[i].n;
  b.el.style.opacity = '.55';
});
const hi = k => bars.forEach((b, i) => { b.el.style.opacity = (k < 0 || i === k) ? '1' : '.35'; });

const msg = wrap.querySelector('#msg');
const texts = [
  '三个函数，一条链：<span class="k">字节 -> 码点 -> 类别 -> 折叠形态</span>。',
  '<span class="v">unicode_len_utf8()</span>：只看首字节的高 4 位就能查出一个 UTF-8 字符占几字节 —— 一张 16 项的查找表。',
  '<span class="v">unicode_cpt_from_utf8()</span>：按位掩码逐段拼出码点；遇到非法字节直接 throw。BPE 主循环用它决定"下一个 symbol 切多长"。',
  '<span class="v">unicode_cpt_flags_from_cpt()</span>：码点 -> 12 个标志位（字母/数字/标点/空白/大小写/是否 NFD），查的是右边第一张表。',
  '<span class="v">unicode_cpts_normalize_nfd()</span>：二分查找 <span class="v">unicode_ranges_nfd</span>，判断这个码点要不要折叠成基字符。',
  '五张表合计 <span class="k">7009 行</span>（2273 + 1433 + 1450 + 1828 + 25）。这就是 unicode 支持的真正体积。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; hi(-1); });
tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [0]); hi(-1); });
tl.at(7100, () => { msg.innerHTML = texts[2]; U.markLines(document, [6]); hi(-1); });
tl.at(10300, () => { msg.innerHTML = texts[3]; U.markLines(document, [37]); hi(0); });
tl.at(13500, () => { msg.innerHTML = texts[4]; U.markLines(document, [25]); hi(3); });
tl.at(17500, () => { msg.innerHTML = texts[5]; U.markLines(document, []); hi(-1); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-08 · 采样链',
    title='一条链就是 host 上的一个 <span class="hl-a">vector</span>',
    sub='llama_sampler_chain 的全部状态：一个 samplers 数组、一块复用缓冲、两个计时计数。',
    caption='注意没有 ggml_tensor、没有 buffer、没有 graph —— 链是纯 host 数据结构。',
    src=SAMPLER_H, parts=[(12, 36)], duration=18000,
    notes={4: 'is_init：是否已经为「后端采样」建过图；这里是 host 路径，保持 false',
           14: 'samplers：链的本体。每个元素是一个 {is_backend, ptr}',
           17: 'cur：复用缓冲，避免每次采样都为整个词表重新分配'},
    marks=[4, 15, 19],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row center" id="vec" style="gap:8px"></div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="l" style="gap:7px"></div>
    <div class="col grow" id="r" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const vec = wrap.querySelector('#vec');
const nodes = [
  { n: 'penalties', c: 'a' }, { n: 'grammar', c: 'd' }, { n: 'temp', c: 'c' },
  { n: 'top_k', c: 'b' }, { n: 'top_p', c: 'b' }, { n: 'dist', c: 'e' }
];
const nels = nodes.map((nd, i) => {
  const e = U.card({ c: nd.c, t: nd.n, b: 'is_backend = false', m: 'ptr' });
  e.style.width = '98px'; e.style.textAlign = 'center'; e.style.flex = '0 0 auto';
  e.querySelector('.ct').style.fontSize = '11px';
  vec.appendChild(e);
  if (i < nodes.length - 1) vec.appendChild(U.arrow('->'));
  return e;
});
nels.forEach(e => e.style.opacity = '.30');

const l = wrap.querySelector('#l');
l.innerHTML = '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">每个节点只是一个指针</div>' +
  '<div class="cb"><span class="cm" style="margin:0">struct info { bool is_backend; llama_sampler * ptr; }</span><br>' +
  '链不认识具体类型，只按 <b>加入顺序</b>依次调用 <span class="cm" style="margin:0">ptr</span> 的虚表。</div></div>';
const r = wrap.querySelector('#r');
r.innerHTML = '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">链自带一块缓冲</div>' +
  '<div class="cb"><span class="cm" style="margin:0">std::vector<llama_token_data> cur;</span><br>' +
  '一次采样要装下整个词表的候选。把它挂在链上，就不必每步重新分配。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '这条链有 6 个节点，但结构体里<b>看不到任何节点类型</b>。',
  '<span class="v">samplers</span> 是一个 vector：链 = 顺序 + 一个指针数组。',
  '<span class="v">is_backend</span> 是给「后端采样」用的开关，默认 false 表示这个节点在 host 上跑。',
  '<span class="v">cur</span> 是复用缓冲：<span class="k">一次采样 = 把整个词表铺成数组</span>。',
  '<span class="v">t_sample_us / n_sample</span>：两个计时字段，说明采样是被单独计量的一个阶段（perf 输出里的 samplers time）。'
];
let t = 700;
nels.forEach((e, i) => { tl.at(t, () => {
  nels.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.30'; });
  msg.innerHTML = texts[Math.min(i, 2)];
}); t += 2000; });
tl.at(13500, () => { msg.innerHTML = texts[3]; U.markLines(document, [19]); });
tl.at(15800, () => { msg.innerHTML = texts[4]; U.markLines(document, [25, 27]); });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-08 · 采样链',
    title='★ 链的执行顺序 = <span class="hl-b">数组顺序</span>，一步不差',
    sub='llama_sampler_apply() 只做一次虚表转发；chain_apply() 按 samplers 的顺序对同一个 cur_p 反复调用。',
    caption='每个节点都在<b>同一个</b> llama_token_data_array 上原地修改 —— 所以顺序决定结果。',
    src=SAMPLER_C, parts=[(382, 389), (681, 701)], duration=19000,
    marks=[0, 9, 16, 27],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">penalties</span><span class="arrow">-></span>
    <span class="chip d">grammar</span><span class="arrow">-></span>
    <span class="chip c">temp</span><span class="arrow">-></span>
    <span class="chip b">top_k</span><span class="arrow">-></span>
    <span class="chip b">top_p</span><span class="arrow">-></span>
    <span class="chip e">dist</span>
  </div>
  <div class="row" style="gap:9px">
    <div class="card grow" style="border-left-color:var(--a)">
      <div class="ct" style="color:var(--a)">同一份 cur_p</div>
      <div class="cb">所有节点共享同一个数组指针。<br>
      <span class="cm" style="margin:0">llama_token_data_array * cur_p</span><br>
      前一个节点改了 <span class="cm" style="margin:0">logit / p / size / sorted</span>，后一个节点直接看到。</div>
    </div>
    <div class="card grow" style="border-left-color:var(--b)">
      <div class="ct" style="color:var(--b)">没有并行、没有重排</div>
      <div class="cb">chain_apply() 就是一个 for 循环，按加入顺序<b>串行</b>调用。<br>
      所以 <span class="cm" style="margin:0">llama_sampler_chain_add()</span> 的调用顺序就是语义的一部分。</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">llama_sampler_apply()</span>：唯一的动作是转发到 <span class="k">iface->apply</span>（虚表），并断言它非空。',
  '<span class="v">llama_sampler_chain_apply()</span>：遍历 <span class="k">chain->samplers</span>，逐个转发。',
  '<span class="v">if (smpl.ptr->iface->apply == nullptr) continue;</span>：节点可以不实现 apply（例如纯 accept 型的节点）。',
  '<span class="v">is_backend</span> 前缀跳过：已经交给后端的节点不再在 host 上重复执行（第 8 幕讲）。',
  '结论：<span class="k">链的顺序不是配置细节，而是算法本身</span>。下一幕看顺序如何改变结果。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [9]); });
tl.at(7100, () => { msg.innerHTML = texts[2]; U.markLines(document, [22]); });
tl.at(10300, () => { msg.innerHTML = texts[3]; U.markLines(document, [14, 18]); });
tl.at(14500, () => { msg.innerHTML = texts[4]; U.markLines(document, [27]); });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-08 · 各节点的职责',
    title='★ 顺序为何会改变结果：<span class="hl-c">temp 保序</span>，<span class="hl-b">top_p 不保序</span>',
    sub='temp 只是把 logits 除以 temp；top_k 截断数组；top_p 先 softmax 再按累积概率截断；dist 才是抽签。',
    caption='源码里 llama_sampler_sample 断言 selected < size（第 9 幕逐字引用）—— 所以 dist 必须在 top_k / top_p 之后。',
    src=SAMPLER_C, parts=[(288, 290), (330, 337), (1556, 1571), (1190, 1204)], duration=25000,
    marks=[1, 11, 13, 30],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div id="tbl"></div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="ca" style="gap:5px"></div>
    <div class="col grow" id="cb" style="gap:5px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['节点', '它改什么', '保序？'],
  [['temp', 'logit /= temp（temp <= 0 时除最大值外全置 -INFINITY）', '保序（除以正数）'],
   ['top_k', '排序后 size = k，丢掉尾巴', '保序（只删）'],
   ['top_p', '先 softmax 算 p，再按累积和截断', '改 p，依赖尺度'],
   ['dist', '按 p 抽一次，写 selected', '只写 selected']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const ca = wrap.querySelector('#ca');
ca.innerHTML = '<div class="cm" style="margin:0">链 A：temp -> top_k -> top_p</div>';
const cb = wrap.querySelector('#cb');
cb.innerHTML = '<div class="cm" style="margin:0">链 B：top_k -> temp -> top_p</div>';
const mk = (host, txt, c) => {
  const e = U.el('div', { class: 'formula', style: 'padding:3px 7px;font-size:9.5px' });
  e.innerHTML = '<span style="color:var(--' + c + ')">' + U.esc(txt) + '</span>';
  host.appendChild(e); return e;
};
const a1 = mk(ca, 'logits / 0.7  ->  全部变大', 'c');
const a2 = mk(ca, 'top_k 取前 20  ->  集合 S', 'b');
const a3 = mk(ca, 'softmax(缩放后的 logits)  ->  截断', 'b');
const b1 = mk(cb, 'top_k 取前 20  ->  集合 S', 'b');
const b2 = mk(cb, 'logits / 0.7  ->  只改尺度', 'c');
const b3 = mk(cb, 'softmax 尺度不同  ->  截断点不同', 'b');
[a1, a2, a3, b1, b2, b3].forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '先记住：<span class="k">四个节点都在同一个数组上原地改</span>，所以它改什么决定了后面能看到什么。',
  '<span class="v">temp</span>：正的 temp 只是给所有 logit 同除一个数，<b>排序不变</b>。所以 top_k 放在 temp 前后，选出的集合一样。',
  '<span class="v">top_k</span> 直接改 <span class="v">size</span>：尾巴被丢掉，后面的节点再也看不到它们。',
  '<span class="v">top_p</span>：先 softmax。而 softmax 的概率<b>依赖尺度</b>，所以"temp 先还是 top_p 先"会得到不同的截断集合 —— 链 A 与链 B 在这里分叉。',
  '<span class="v">dist</span>：读 <span class="v">p</span> 抽一次，只写 <span class="v">selected</span>。若它后面还有节点把 <span class="v">size</span> 改小，sample 的断言就可能被触发。',
  '一句话：<span class="k">保序的节点可以换位，改尺度的节点不能</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows[0].className = 'on'; });
tl.at(3900, () => { msg.innerHTML = texts[1]; rows[1].className = 'on'; U.markLines(document, [1]); a1.style.opacity = '1'; b2.style.opacity = '1'; });
tl.at(7100, () => { msg.innerHTML = texts[2]; rows[2].className = 'on'; U.markLines(document, [11]); a2.style.opacity = '1'; b1.style.opacity = '1'; });
tl.at(10300, () => { msg.innerHTML = texts[3]; rows[3].className = 'on'; U.markLines(document, [13]); a3.style.opacity = '1'; b3.style.opacity = '1'; });
tl.at(15000, () => { msg.innerHTML = texts[4]; rows[4].className = ''; U.markLines(document, [30]); });
tl.at(20000, () => { msg.innerHTML = texts[5]; U.markLines(document, []); rows.forEach(r => { r.className = ''; }); });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-08 · 核心',
    title='★ 为什么采样不进图：<span class="hl-e">它操作的是 host 上的 float 数组</span>',
    sub='一次采样 = 把 logits 复制进 std::vector<llama_token_data>，然后在 host 上跑链。全程没有 ggml 算子。',
    caption='本版新增的 [EXPERIMENTAL] 后端采样反过来证明了这一点：要把采样放进图，必须为它单独建一张图并逐个算子探测后端支持。',
    src=SAMPLER_C, parts=[(938, 952), (583, 587), (649, 662), (746, 762)], duration=25000,
    marks=[3, 16, 25, 44],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:9px">
    <div class="col grow" id="in" style="gap:6px"></div>
    <div class="col grow" id="out" style="gap:6px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const lhs = wrap.querySelector('#in');
lhs.innerHTML = '<div class="cm" style="margin:0">图内（L2-07 / L5-01）</div>' +
  '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">llama_decode -> ggml 图</div>' +
  '<div class="cb">注意力、矩阵乘、norm ... 都是 ggml 算子，可以切给 CPU / CUDA / Metal。<br>' +
  '产出：<span class="cm" style="margin:0">float logits[n_vocab]</span>（host 可取）。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">图上没有采样算子</div>' +
  '<div class="cb">没有 GGML_OP_SAMPLE 这种东西。图不知道"概率"，也不知道"词表"。</div></div>';
const rhs = wrap.querySelector('#out');
rhs.innerHTML = '<div class="cm" style="margin:0">图外（本课）</div>' +
  '<div class="card" style="border-left-color:var(--e)"><div class="ct" style="color:var(--e)">host 上的数组</div>' +
  '<div class="cb"><span class="cm" style="margin:0">std::vector<llama_token_data> cur;</span><br>' +
  '每个元素 = {id, logit, p}。采样器只认这个数组。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)"><div class="ct" style="color:var(--d)">为什么放在 host 更合理</div>' +
  '<div class="cb">o(n_vocab) 的标量操作，相对一次前向可以忽略；<br>' +
  '而且 grammar / penalties / dry 依赖<b>历史 token</b>与<b>语法状态</b>，<br>' +
  '这些状态本来就在 host 上。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">llama_sampler_sample()</span> 从上下文取出 logits，铺成 <span class="k">std::vector<llama_token_data></span>。',
  '注意这里没有任何 <span class="cm" style="margin:0">ggml_*</span> 调用：没有新张量、没有 buffer、没有 compute。',
  '然后 <span class="v">llama_sampler_apply(smpl, &cur_p)</span> 在 host 上跑完整条链，取出 selected 对应的 id。',
  '[EXPERIMENTAL] 后端采样：要给节点加 <span class="v">backend_apply</span>，它收到的是 ggml context 与 cgraph —— 也就是说<b>它得自己建一张采样图</b>。',
  '还要逐个算子问后端：<span class="v">ggml_backend_dev_supports_op()</span> 返回 false 就退回 host。默认路径根本不走这套。',
  '链上只允许<b>前缀</b>上后端：第一个不支持 backend_init 的节点之后，全部留在 host（grammar、dry、xtc、mirostat、top_n_sigma、infill、adaptive_p、typical 都没有实现）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [3]); });
tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [5]); });
tl.at(7100, () => { msg.innerHTML = texts[2]; U.markLines(document, [9]); });
tl.at(11000, () => { msg.innerHTML = texts[3]; U.markLines(document, [16, 22]); });
tl.at(15000, () => { msg.innerHTML = texts[4]; U.markLines(document, [25, 31]); });
tl.at(19500, () => { msg.innerHTML = texts[5]; U.markLines(document, [44, 52]); });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-08 · 收束',
    title='一次采样的全部：<span class="hl-a">apply</span> 一次，<span class="hl-a">accept</span> 一次',
    sub='图给出 logits，采样器给出 token，并把 token 记进自己的状态（penalties 的计数、grammar 的栈）。',
    caption='下一课 L2-09：grammar 的实现（llama-grammar.cpp）与 chat 模板（llama-chat.cpp）。',
    src=SAMPLER_C, parts=[(947, 960)], duration=19000,
    marks=[0, 3, 7, 9],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip c">logits[]</span><span class="arrow">-></span>
    <span class="chip a">cur_p（host 数组）</span><span class="arrow">-></span>
    <span class="chip b">chain.apply()</span><span class="arrow">-></span>
    <span class="chip e">selected</span><span class="arrow">-></span>
    <span class="chip d">accept(token)</span>
  </div>
  <div id="tbl"></div>
  <div id="ex"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['环节', '跑在哪', '本课逐字引用的位置', '展开在哪课'],
  [['分词 tokenize', 'host (CPU)', 'llama-vocab.cpp:3413', '本课'],
   ['BPE 合并（rank）', 'host (CPU)', 'llama-vocab.cpp:264 / 657', '本课 / L1-06（merges 从 GGUF 读）'],
   ['unicode 类别与 NFD', 'host (CPU)', 'unicode.cpp:1147 / 1117', '本课'],
   ['decode 建图算 logits', '后端（CPU/GPU）', '图里（本课不引用）', '回顾 L2-07 / L5-01'],
   ['采样链 apply', 'host (CPU)', 'llama-sampler.cpp:681', '本课'],
   ['grammar / penalties / dry', 'host (CPU)', 'llama-sampler.cpp:2751（无 backend 实现）', 'L2-09 llama-grammar.cpp'],
   ['后端采样（实验）', '后端（另建一张图）', 'llama-sampler.cpp:733', 'L5-01 后端能力探测']],
  { monoCols: [2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '给一条 sampler chain 配上 grammar 约束。为什么 <b>不需要</b> CUDA / Metal 后端为 grammar 写 kernel？',
  '因为 grammar 采样器只实现了 host 侧的 <span class="mono">.apply()</span>：' +
  '<span class="mono">llama_sampler_grammar_i</span> 里 <span class="mono">backend_init</span> / ' +
  '<span class="mono">backend_apply</span> 都是 <span class="mono">nullptr</span>（llama-sampler.cpp:2751）。<br>' +
  '它拿到的是 host 内存里的 <span class="mono">llama_token_data_array</span>，把不合语法的 token 的 ' +
  '<span class="mono">logit</span> 置为 <span class="mono">-INFINITY</span>' +
  '（实现在 <span class="mono">llama_grammar_apply_impl()</span>，llama-grammar.cpp:1355 起，L2-09 逐字展开）。<br>' +
  '图只负责把 logits 算出来；约束是<b>图外</b>对分布的一次裁剪，所以后端对此一无所知。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">cur_p</span> 是一个纯 host 结构：指针 + size + selected + sorted。',
  '链跑完后，答案就在 <span class="v">cur_p.data[cur_p.selected].id</span>。',
  '<span class="v">accept()</span> 把选中的 token 告诉链上每个节点 —— penalties 记计数，grammar 推进语法栈。',
  '什么都不看的行：tokenize 与采样都在 host，图只负责 logits。这就是"采样不进图"的全部证据。'
];
rows.forEach(r => { r.style.opacity = '.35'; });
tl.at(700, () => { msg.innerHTML = texts[0]; rows[0].style.opacity = '1'; });
tl.at(3600, () => { msg.innerHTML = texts[0]; rows[1].style.opacity = '1'; rows[2].style.opacity = '1'; });
tl.at(6600, () => { msg.innerHTML = texts[0]; rows[3].style.opacity = '1'; });
tl.at(9600, () => { msg.innerHTML = texts[1]; rows[4].style.opacity = '1'; });
tl.at(12600, () => { msg.innerHTML = texts[2]; rows[5].style.opacity = '1'; rows[6].style.opacity = '1'; });
tl.at(16200, () => { msg.innerHTML = texts[3]; U.markLines(document, [7, 12]); });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、词表的职责面',
    '`llama_vocab` 是纯 host 对象：它把文本和 token id 互相翻译，并持有 BPE 的 merges 表。'
    '注意 `find_bpe_rank()` 与 `tokenize()` 都在这里声明 —— 分词算法的"知识"一半在代码里，'
    '一半在词表数据里。',
    src=VOCAB_H, parts=[(94, 101), (160, 176)], lang='c')

L.section(
    '二、分词入口：先切分，再分派',
    '`tokenize()` 的第一件事不是查 merges，而是把文本装进 `fragment_buffer` 并调用 '
    '`tokenizer_st_partition()`：特殊 token（BOS/EOS/控制符）被优先摘出来，'
    '剩下的原始文本片段才逐段交给具体 tokenizer。这就是"prompt 里写了 `<|im_start|>` '
    '为什么不会被切成普通字符"的原因。',
    src=VOCAB_C, parts=[(3413, 3427)], lang='c')

L.section(
    '三、特殊 token 的优先切分',
    '`tokenizer_st_partition()` 遍历 `cache_special_tokens`，对每个特殊 token 扫描所有尚未处理的'
    '文本片段。源码注释指出：当 `parse_special == false` 时，CONTROL 与 UNKNOWN 属性的 token 被跳过，'
    '而用户自定义 token 仍然参与预切分。',
    src=VOCAB_C, parts=[(3250, 3270)], lang='c')

L.section(
    '四、★ BPE：symbol 链 + 按 rank 排序的优先队列',
    'BPE 的实现是"链 + 优先队列"：\n\n'
    '```text\n'
    '1. 把每个词按字符拆成 symbols（一个双向链表）\n'
    '2. 每对相邻 symbol 查 find_bpe_rank()，查得到就入优先队列\n'
    '3. 每次弹出 rank 最小的 bigram，合并，并只补两个新 bigram\n'
    '```\n\n'
    '`llama_priority_queue` 是 `std::priority_queue` 的一个薄包装，'
    '只多了一个 `pop_move()`（把 `pop()` 显式 `delete` 掉，避免误用拷贝语义）。'
    '排序准则写在 `llm_bigram_bpe::comparator` 里：先比 `rank`，rank 相同再比左端位置。',
    src=VOCAB_C, parts=[(249, 262), (264, 278)], lang='c')

L.section(
    '五、BPE 主循环：合并后只补两个 bigram',
    '这是 BPE 的性能关键：合并 `(left, right)` 之后，只有左邻居与右邻居可能产生新的可合并对，'
    '所以只调用两次 `add_new_bigram()`。注意第 673 行的"过期 bigram"检查 —— '
    '队列里可能存着已经失效的 bigram，靠文本比对丢弃。',
    src=VOCAB_C, parts=[(657, 689)], lang='c')

L.section(
    '六、unicode 的三个查表函数',
    '`unicode_len_utf8()` 只看首字节高 4 位；`unicode_cpt_from_utf8()` 按位掩码拼码点，'
    '非法字节直接 `throw std::invalid_argument`（上层 `unicode_cpts_from_utf8()` 会把它换成 U+FFFD）。'
    '`unicode_cpt_flags_from_cpt()` 与 `unicode_cpts_normalize_nfd()` 都是**查表**：'
    '一个查 `unicode_ranges_flags`（构建一张 0x110000 项的数组），一个在 `unicode_ranges_nfd` 上二分。',
    src=UNI_C, parts=[(16, 47), (1117, 1128), (1147, 1160)], lang='c')

L.section(
    '七、unicode.h：12 个标志位挤进一个 uint16',
    '`unicode_cpt_flags` 用位域把 12 个布尔量塞进 16 位，并提供 `as_uint()` 双向转换。'
    '源码在这里写明了一个**可移植性陷阱**：转换依赖字节序（`__BYTE_ORDER__`），'
    '文件开头因此留着一条 TODO：reimplement this structure in endian-independent way。'
    '另外请注意，位域定义的顺序与 `as_uint()` 里的移位是**同一件事的两种写法**，'
    '所以这些标志可以被当作一个整数存进数据表。',
    src=UNI_H, parts=[(7, 23), (25, 48), (85, 95)], lang='c')

L.section(
    '八、unicode-data.h：五张表的对外声明',
    '这个头文件只有 20 行，却定义了整套 unicode 支持的接口面：一个 `range_nfd` 结构体、'
    '一个 `MAX_CODEPOINTS` 常量，以及五个 `extern` 表。'
    '把"数据"和"算法"分开声明，是这一课能讲清楚 unicode 的前提。',
    src=UNI_DATA_H, parts=[(8, 20)], lang='c')

L.section(
    '九、unicode-data.cpp：7009 行生成数据',
    '第一行就写明它是 `scripts/gen-unicode-data.py` 的生成物。五张表的规模（用行号数出来的真实值）：\n\n'
    '| 表 | 行号区间 | 行数 |\n|---|---|---|\n'
    '| `unicode_ranges_flags` | 11 - 2283 | 2273 |\n'
    '| `unicode_set_whitespace` | 2287 - 2311 | 25 |\n'
    '| `unicode_map_lowercase` | 2316 - 3748 | 1433 |\n'
    '| `unicode_map_uppercase` | 3753 - 5202 | 1450 |\n'
    '| `unicode_ranges_nfd` | 5206 - 7033 | 1828 |\n\n'
    '`unicode_ranges_flags` 的注释说明了编码方式：每行是"区间起点 + 标志位"，'
    '`last = next_start - 1`；`unicode_cpt_flags_array()` 正是按这个约定把它展开成数组的。'
    '大小写两张表上面都写着同一句约束：list is always in ascending order, to enable binary search。',
    src=UNI_DATA_C, parts=[(1, 13), (2286, 2292), (2314, 2319), (5205, 5210)], lang='c')

L.section(
    '十、采样器家族：函数名与行号（全部来自 llama-sampler.cpp）',
    '`llama_sampler_init_*` 一共 22 个定义。下面这份名单是把 `llama-sampler.cpp` 里'
    '所有 `struct llama_sampler * llama_sampler_init_*` 定义逐个数出来的，'
    '括号里是定义所在行：\n\n'
    '| 家族 | 函数（行号） |\n|---|---|\n'
    '| 基础 | `llama_sampler_init_empty` (520)、`llama_sampler_init_greedy` (1107)、'
    '`llama_sampler_init_dist` (1399) |\n'
    '| 截断 | `llama_sampler_init_top_k` (1519)、`llama_sampler_init_top_p` (1719)、'
    '`llama_sampler_init_min_p` (1882)、`llama_sampler_init_typical` (1994)、'
    '`llama_sampler_init_top_n_sigma` (3300) |\n'
    '| 温度 | `llama_sampler_init_temp` (2104)、`llama_sampler_init_temp_ext` (2307)、'
    '`llama_sampler_init_xtc` (2416) |\n'
    '| 自适应 | `llama_sampler_init_mirostat` (2537)、`llama_sampler_init_mirostat_v2` (2643)、'
    '`llama_sampler_init_adaptive_p` (3860) |\n'
    '| 约束与偏置 | `llama_sampler_init_grammar` (2825)、`llama_sampler_init_grammar_lazy` (2832)、'
    '`llama_sampler_init_grammar_lazy_patterns` (2843)、`llama_sampler_init_logit_bias` (4043) |\n'
    '| 重复抑制 | `llama_sampler_init_penalties` (3203)、`llama_sampler_init_dry` (3639)、'
    '`llama_sampler_init_dry_testing` (3690) |\n'
    '| 填充 | `llama_sampler_init_infill` (4288) |\n\n'
    '背后是两个"空实现"规则：参数等于默认值时，`llama_sampler_init_*` 会返回 '
    '`llama_sampler_init_empty("?top-k")` 这样的占位节点（例如 top_k 的 `k <= 0`、'
    'temp 的 `temp == 1.0f`、top_p 的 `p >= 1.0f`）。')

L.section(
    '十一、★ temp / top_k / top_p / dist 各自改什么',
    '四个 `apply` 挤在一起看，就能解释"顺序即语义"：\n\n'
    '- `temp`：正的 temp 只做 `logit /= temp`（除正数**保序**）；`temp <= 0` 是特例，'
    '把除最大值以外的 logit 全置 `-INFINITY`（退化成 greedy）。\n'
    '- `top_k`：排序后直接写 `cur_p->size = k`，尾巴被**永久删掉**。\n'
    '- `top_p`：先 `softmax` 算 `p`，再按累积和截断；因为 softmax 依赖尺度，'
    '**"temp 先还是 top_p 先"会得到不同的集合**。\n'
    '- `dist`：`std::uniform_real_distribution<double>` 抽一次，然后线性扫描累积概率写 `selected`。\n\n'
    '源码注释里还有一条工程细节：dist 用的是"边扫描边归一化"的单遍实现，'
    '注释说它在全量 gpt-oss 词表上比下面的双遍版本快约 3 倍。',
    src=SAMPLER_C, parts=[(265, 291), (321, 338), (1549, 1571), (1150, 1214)], lang='c')

L.section(
    '十二、grammar 与 dry：没有后端实现的那些节点',
    '对比两张虚表就能看出"哪些采样器只能跑在 host 上"：`llama_sampler_grammar_i` 的 '
    '`backend_init` / `backend_accept` / `backend_apply` / `backend_set_input` / '
    '`backend_reset` / `copy_state` **全是 nullptr**；`llama_sampler_dry_i` 同样如此。'
    '（`penalties` 与 `logit_bias` 则是反例：它们在这一版里已经实现了 `backend_apply`。）',
    src=SAMPLER_C, parts=[(2751, 2764), (3624, 3637)], lang='c')

L.section(
    '十三、一次采样的完整收尾',
    '链跑完后，`llama_sampler_sample()` 断言 `selected` 落在 `[0, size)` 内，取出 token id，'
    '再调用 `llama_sampler_accept()` 把 token 交给链上每个节点更新状态。'
    '这条断言也解释了上一节的位置约束：会缩小 `size` 的节点（top_k / top_p）'
    '必须排在会写 `selected` 的节点（dist / greedy）**之前**。',
    src=SAMPLER_C, parts=[(947, 962)], lang='c')

L.footnote_add('本课声明并引用了 8 个源文件：`llama-sampler.cpp` / `llama-sampler.h` / '
               '`llama-vocab.cpp` / `llama-vocab.h` / `unicode.cpp` / `unicode.h` / '
               '`unicode-data.cpp` / `unicode-data.h`，全部计入覆盖率。')
L.footnote_add('散文里提到的 `include/llama.h`（`struct llama_sampler_i` 与 `llama_token_data_array` 的声明）、'
               '`src/llama-grammar.cpp`（grammar 的 apply 实现）与 `src/ggml-backend` 的 '
               '`ggml_backend_dev_supports_op()` 属于**其他课的覆盖域**，本课不引用其源码，'
               '因此不计入本课覆盖率。')
L.footnote_add('第 3 幕的 BPE 合并演示（`(e,s) rank 4` 之类）是**示意值**，'
               '只用于说明"rank 小的先合并"这条规则；真实 rank 存在模型的 merges 表里，'
               '由 `llama_vocab::find_bpe_rank()` 查询。')

L.prereqs('`L2-07`（logits 从哪来：decode 的图算完并把结果交回 host）')

L.goal(
    '说出 `llama_sampler_chain` 里有哪些字段，并解释为什么它"不认识"任何一个具体采样器（对应验收点）；',
    '按执行顺序写出 `llama_sampler_chain_apply()` 做了什么，并说明为什么链的顺序会改变采样结果；',
    '说出 `temp` / `top_k` / `top_p` / `dist` 各自修改 `llama_token_data_array` 的哪个字段；',
    '用源码证据说明采样为什么跑在 host 上而不进图，以及 `grammar` / `dry` 为什么没有后端实现；',
    '描述 BPE 分词的三级结构（特殊 token 切分、按 rank 的优先队列合并、unicode 查表）。')

L.conclusion(
    '图算 logits，host 选 token',
    '`llama_decode()` 的产物是 `float logits[n_vocab]`。把它变成 token 的 '
    '`llama_sampler_sample()` 全程在 host 上：\n\n'
    '```text\n'
    'logits[]  ->  std::vector<llama_token_data> cur  ->  chain.apply(cur_p)  ->  cur_p.selected  ->  accept(token)\n'
    '```\n\n'
    '这个链上没有 ggml 算子、没有张量、没有 buffer —— 所以 CUDA / Metal 后端对它一无所知。')

L.conclusion(
    '★ 链的顺序就是算法',
    '`llama_sampler_chain_apply()` 是一个朴素的 for 循环，所有节点在**同一个** '
    '`llama_token_data_array` 上原地修改。因此：\n\n'
    '- `temp`（除正数）与 `top_k`（只删尾巴）**保序**，可以互换位置；\n'
    '- `top_p` 依赖 softmax 的**尺度**，与 `temp` 不可互换；\n'
    '- `dist` / `greedy` 写 `selected`，必须排在会缩小 `size` 的节点之后 —— '
    '否则 `llama_sampler_sample()` 的断言会被触发。')

L.conclusion(
    '★ grammar 约束不需要后端参与',
    '`llama_sampler_grammar_i` 的 `backend_init` / `backend_apply` 都是 `nullptr`：'
    'grammar 采样器只实现 host 侧的 `.apply()`，在 `llama_token_data_array` 上把不合语法的 '
    'token 的 `logit` 置为 `-INFINITY`（`llama_grammar_apply_impl()`，`src/llama-grammar.cpp:1355` 起，'
    'L2-09 逐字展开）。同理，`dry` / `xtc` / `mirostat` / `top_n_sigma` / `infill` / '
    '`adaptive_p` / `typical` 都没有后端实现。\n\n'
    '这一版新增的 `[EXPERIMENTAL]` 后端采样是这条结论的正面证明：要让某个节点进图，'
    '必须为它单独建一张采样图（`llama_sampler_backend_probe_graph()`），'
    '并逐个算子问后端 `ggml_backend_dev_supports_op()`；而且链上只允许**前缀**上后端。')

L.conclusion(
    '分词：一半是算法，一半是数据',
    '`llama_vocab::impl::tokenize()` 分三段：`tokenizer_st_partition()` 先切特殊 token，'
    '再按 vocab 类型分派到 7 种 tokenizer，最后由具体 tokenizer 产出 token id。'
    '本课细看了 BPE：symbol 链 + 按 `rank` 排序的优先队列，rank 来自词表的 merges 表'
    '（`find_bpe_rank()`）。unicode 侧则是纯查表：`unicode_ranges_flags`（2273 行）、'
    '`unicode_ranges_nfd`（1828 行）、`unicode_map_lowercase` / `unicode_map_uppercase` '
    '（1433 / 1450 行）、`unicode_set_whitespace`（25 行）—— 合计 7009 行生成数据，'
    '放在 `unicode-data.cpp` 里。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
