#!/usr/bin/env python3
"""L2-05 · 批、解码参数与模型装配 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-05-batch-and-model-build/lesson.spec.py

本课覆盖 6 个源文件（全部逐字引用，行号一律来自上游 v0.5.0）：
    src/llama-batch.cpp   src/llama-batch.h
    src/llama-model.cpp   src/llama-model.h
    src/llama-impl.cpp    src/llama-impl.h
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_BH = 'src/llama-batch.h'
SRC_BC = 'src/llama-batch.cpp'
SRC_MH = 'src/llama-model.h'
SRC_MC = 'src/llama-model.cpp'
SRC_IH = 'src/llama-impl.h'
SRC_IC = 'src/llama-impl.cpp'

L = Lesson(
    id='L2-05',
    layer='L2 · 从模型到图',
    title='批、解码参数与模型装配',
    codecap='src/llama-batch.{h,cpp} · src/llama-model.{h,cpp} · src/llama-impl.{h,cpp}（逐字引用）',
    nav={'prev': {'href': '../L2-04-kv-cache-and-memory/index.html',
                  'label': 'L2-04 KV cache 与记忆家族'},
         'next': {'href': '../L2-06-graph-skeleton/index.html',
                  'label': 'L2-06 ★ 计算图骨架 llama-graph'}},
)

L.note('**一句话**：调用方交给 `llama_decode` 的是一个 `llama_batch`，但计算图不是按 batch 建的 —— '
       '中间隔着一层切分：`llama_batch_allocr` 把 batch 切成若干 `llama_ubatch`，'
       '**每个 ubatch 建一次图、算一次**。所以"怎么切"直接决定"图长什么样、要建几张图"。')
L.note('本课同时收束 L2 的另一条线：**模型装配**。GGUF 里的 arch 决定 `llama_model_create()` new 出哪一个模型类；'
       '基类读共性元数据，架构子类读自己的超参与张量表；层到设备/buffer 的分配在 `load_tensors()` 里一次性算好。')
L.note('阅读顺序：先看 batch 怎么被补齐（第 2 幕），再看 ubatch 的字段（第 3 幕），'
       '然后是三种切法与 `split_equal` 的真实判据（第 4-6 幕），最后是装配与设备分配（第 8-9 幕）。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2 · 从模型到图',
    title='本课的位置：<span class="hl-a">batch</span> -> <span class="hl-b">ubatch</span> -> 图',
    sub='一次 decode 收到的是一整批；真正被"算"的是切出来的每一小块。',
    caption='上游 L2-02/L2-03 决定"有哪些张量、权重放在哪"；下游 L2-06/L2-07 用 ubatch 建图并执行。',
    src=SRC_BH, parts=[(14, 18)], duration=16000,
    mark_src=[15, 16, 17],
    notes={0: '注释就是设计约束：ubatch 是会被反复创建、拷贝、传递的值，必须保持轻量'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">llama_batch</span><span class="arrow">-></span>
    <span class="chip a">llama_batch_allocr</span><span class="arrow">-></span>
    <span class="chip b">llama_ubatch</span><span class="arrow">-></span>
    <span class="chip c">建图 + 计算</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '① 输入：一整批', b: 'token 或 embd、pos、n_seq_id/seq_id、logits<br>都在一个 <b>llama_batch</b> 里交给解码器。', m: 'llama_batch' },
  { c: 'b', t: '② 先切分，再执行', b: '切分器把 batch 切成若干 <b>ubatch</b>，<br>每个都不超过 n_ubatch 个 token。', m: 'llama_batch_allocr' },
  { c: 'd', t: '③ ubatch 才是执行单位', b: '收到一个 ubatch 就建一次图、算一次；<br>图输入的形状就是 ubatch 的几个计数字段。', m: 'llama_ubatch' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '本课只回答一个问题：<span class="k">一个 batch 是怎么变成 ubatch 的，这件事为什么影响图</span>。',
  '<span class="v">llama_batch</span> 是调用方的输入：n_tokens 个 token（或嵌入）+ 位置 + 序列归属 + 哪些位置要 logits。',
  '<span class="v">llama_batch_allocr</span> 负责校验、补齐、切分，产出 <span class="v">llama_ubatch</span>：本次要算的一小块。',
  '<span class="k">一个 ubatch 一张图</span>：切分方式直接决定要建几张图、每张图的输入多宽。',
  '回顾 L2-02（架构表决定超参与张量清单）与 L2-03（权重从文件到 buffer）—— 本课把它们装配起来。',
  '接着看 L2-06（用 ubatch 建图骨架）与 L2-07（decode 主流程），那里会看到这几个字段怎么变成张量形状。'
];
defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(9400, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [2]); });
tl.at(12600, () => { msg.innerHTML = texts[5]; U.markLines(document, [3, 4]); });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-05 · 输入',
    title='llama_batch：<span class="hl-c">只有 token 是必给的</span>',
    sub='其余字段缺失时会被 init() 现场补齐 —— 补齐规则决定了后面"序列集"长什么样。',
    caption='补齐动作写在 llama_batch_allocr::init() 里（llama-batch.cpp:69 起）。',
    src=SRC_BC, parts=[(69, 117)], duration=18000,
    mark_src=[73, 81, 90, 117],
    notes={4: 'n_seq_id 为空：每个 token 默认只属于 1 个序列',
           12: 'seq_id 为空：每个 token 都指向 seq_id_0（默认序列 0）',
           21: 'pos 为空：从该序列在记忆模块里的最大位置 + 1 接着排',
           26: 'p0[s] 逐 token 推进 —— 同一序列在 batch 里的位置必须连续递增',
           48: '补齐后的指针写回 batch，之后切分器只读这一份'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'token / embd', b: '要算什么：token id 或<br>直接喂嵌入（二选一）', m: '必给' },
  { c: 'b', t: 'pos', b: '每个 token 的位置；<br>不给就从记忆模块往后接', m: '可空' },
  { c: 'c', t: 'n_seq_id / seq_id', b: '这个 token 属于哪些序列；<br>不给就是序列 0', m: '可空' },
  { c: 'd', t: 'logits', b: '哪些位置要输出；<br>不给就只输出最后一个', m: '可空' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => U.card(d, { style: 'width:163px' }));
els.forEach(e => host.appendChild(e));
els.forEach(e => { e.style.opacity = '.30'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '一个 batch = 四个数组 + n_tokens。<span class="k">只有 token 或 embd 是必给的。</span>',
  '<span class="v">pos</span> 为空：有 KV cache 就从 <span class="v">seq_pos_max(s) + 1</span> 往后排；没有就从 0 开始。',
  '<span class="v">n_seq_id / seq_id</span> 为空：全部指向默认序列 0 —— 这是"单序列推理"最常见的形态。',
  '<span class="v">logits</span> 为空：默认只把最后一个 token 标成输出（同一段 init()，llama-batch.cpp:120 起）。',
  '补齐后的指针写回 batch（<span class="v">batch.pos = pos.data()</span>），之后切分器只读这份 batch。',
  '关键一步：<span class="k">切分的输入是补齐后的 batch</span> —— seq_id 与 pos 都齐了，才谈得上"序列集"。'
];
defs.forEach((_, i) => tl.at(700 + i * 2700, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  U.markLines(document, [[4], [23], [13], []][i]);   // 渲染下标：上游行 73 / 90 / 81
  msg.innerHTML = texts[i];
}));
tl.at(12000, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  U.markLines(document, []);
  msg.innerHTML = texts[4];
});
tl.at(15000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-05 · 核心',
    title='★ <span class="hl-a">struct llama_ubatch</span> 逐字段',
    sub='一个不变量把它全部串起来：n_tokens = n_seq_tokens x n_seqs。',
    caption='字段注释里写死了这条不变量；ubatch_add() 也是按它算出来的（llama-batch.cpp:821）。',
    src=SRC_BH, parts=[(30, 52)], duration=20000,
    mark_src=[30, 34, 35, 36, 37, 45, 46, 47, 48, 49, 50, 51, 52],
    notes={0: 'b_equal_seqs 语义上是 bool，故意用 uint32_t —— 注释：否则 address sanitizer 会报错',
           4: 'n_tokens = n_seq_tokens * n_seqs（字段注释里就是这条不变量）',
           6: 'n_seqs：这个 ubatch 里有几个"序列集"',
           7: 'n_seqs_unq：去重后的序列 id 个数',
           16: 'embd：直接喂嵌入时用，形状 [n_embd, n_tokens]',
           21: 'seq_idx：序列 id -> [0, n_seqs_unq) 的下标，取 pooled embedding 时用'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['组', '字段', '含义'],
  [['计数', 'n_tokens / n_seq_tokens / n_seqs', '总 token 数 = 每序列集 token 数 x 序列集个数'],
   ['计数', 'n_seqs_unq / n_pos', '去重序列 id 个数 / 每个 token 的位置个数（M-RoPE 时可大于 1）'],
   ['数据', 'token / embd / pos', '指向本 ubatch 自己的数据，不复制调用方的数组'],
   ['归属', 'n_seq_id / seq_id / seq_id_unq / seq_idx', '每个 token 属于哪些序列；seq_idx 是 id 到下标的映射'],
   ['输出', 'output', '本 ubatch 里哪些位置要回传 logits'],
   ['标记', 'b_equal_seqs', '这几个序列集是不是等长 —— 由切分函数决定']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '先记不变量：<span class="v">n_tokens = n_seq_tokens * n_seqs</span>（字段注释里写着的）。',
  '<span class="v">n_seq_tokens</span> 是"每个序列集出几个 token"，<span class="v">n_seqs</span> 是"有几个序列集"。',
  '<span class="v">n_seqs_unq</span> 才是去重后的序列 id 个数：简单切分下一个 token 就是一个序列集，<br>n_seqs 可以等于 n_tokens，而 n_seqs_unq = 1。',
  'token / embd / pos / output 都是裸指针，指向 <span class="v">data_t</span> 里的 vector —— ubatch 本身很轻。',
  '<span class="v">output</span> 决定哪些位置的 logits 要回传，采样器只看这些位置。',
  '<span class="v">equal_seqs()</span> 就是 <span class="v">b_equal_seqs != 0</span>：切分函数给它赋值，图构建器读它。',
  '★ 记住这张表：L2-06 的图输入张量、L2-07 的图复用判据，读的都是这几个字段。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [5]); });
rows.forEach((r, i) => tl.at(3400 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
  U.markLines(document, [[5, 7, 8], [10, 12], [19, 20, 22], [23, 24, 25, 26], [28], [0]][i]);
}));
tl.at(18900, () => {
  rows.forEach(x => { x.className = ''; });
  U.markLines(document, [0, 5, 7, 8, 10]);
  msg.innerHTML = texts[6];
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-05 · 切分器',
    title='三种切法，产物都是 <span class="hl-b">llama_ubatch</span>',
    sub='区别只在两处：这个 ubatch 里放几个序列集（n_seqs），以及 equal_seqs 标记。',
    caption='调用点按"流的条数"选：单流用 split_simple，多流用 split_equal（llama-kv-cache.cpp:713）。',
    src=SRC_BH, parts=[(99, 111)], duration=17000,
    mark_src=[100, 103, 108, 111],
    notes={3: 'split_simple：按 batch 顺序连续取最多 n_ubatch 个 token（实现见 llama-batch.cpp:476）',
           9: 'split_equal：按"等长的序列集"切（实现见 llama-batch.cpp:510）',
           12: 'split_seq：一个 ubatch 只装一个序列集'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', t: 'split_simple', m: 'ubatch_add(idxs, idxs.size(), false)',
    b: '按 batch 顺序连续取最多 n_ubatch 个 token。<br>结果：n_seqs = n_tokens，n_seq_tokens 恒为 1，<br>equal_seqs() = false。' },
  { c: 'a', t: 'split_equal', m: 'ubatch_add(idxs, n_seqs, true)',
    b: '选一组互不相交的序列集，<br>每个序列集出同样多的 token（锁步）。<br>equal_seqs() = true。' },
  { c: 'e', t: 'split_seq', m: 'ubatch_add(idxs, 1, true)',
    b: '一个 ubatch 里只有一个序列集，<br>n_seqs = 1，token 尽量取满 n_ubatch。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '三种切法的声明就在头文件里。产物同类型，差别在 <span class="v">n_seqs</span> 与 <span class="v">equal_seqs</span>。',
  '头文件注释：split_simple 面向"数量未知、长短不一的序列集"—— 它不保证等长，只保证不超过 n_ubatch。',
  'split_equal 就是"同长度的序列可以合并成一个 ubatch、走同一条图"的实现。',
  'split_seq 最保守：一个 ubatch 只有一个序列集（SSM / 混合记忆这类模型需要它）。',
  '★ 这三个函数是 ubatch 形状的唯一来源 —— 图那边不重新切分，只按字段建张量。'
];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  U.markLines(document, [[1], [5], [10], [14]][i]);
  msg.innerHTML = texts[i];
}));
tl.at(13700, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  U.markLines(document, [1, 5, 10, 14]);
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-05 · 核心',
    title='★ split_equal 的<span class="hl-a">两条判据</span>',
    sub='判据一：序列集互不相交；判据二（sequential 时）：序列号严格递增。',
    caption='"不相交"用 bitset 的按位与判空实现 —— seq_set[i] 是 token i 属于哪些序列的位集。',
    src=SRC_BC, parts=[(521, 553)], duration=20000,
    mark_src=[522, 531, 539, 547, 553],
    notes={10: '判据一：与已选中的每个序列集都不相交（位集按位与为空）',
           18: '判据二：sequential = true 时还要求序列号严格递增',
           26: '上限：序列集个数本身也不能超过 n_ubatch',
           32: '这个 ubatch 里最终参与几个序列集'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '判据一 · 互不相交', m: '(cur_seq_set[s] & seq_set[i]).none()',
    b: '一个 token 不能同时属于本 ubatch 里的两个序列集，<br>否则同一份 KV 会被两条流各算一遍。' },
  { c: 'c', t: '判据二 · 序列号递增', m: 'batch.seq_id[i][0] == last_seq_id + 1',
    b: '只有 sequential = true 时才要求，<br>保证 ubatch 内序列号是连续递增的一段。' },
  { c: 'e', t: '上限 · 序列集个数', m: 'cur_seq_set.size() > n_ubatch',
    b: '序列集个数本身不能超过 n_ubatch；<br>选中后 n_seqs = cur_seq_set.size()。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '切分前先"选人"：从左到右扫 batch，把还能进这个 ubatch 的序列集挑出来。',
  '例：token 0 只属于 {0}，token 3 只属于 {1} -> <span class="v">{0} &amp; {1} = 空</span> -> 可以并存。',
  '若某个 token 同时属于 {0,1}（源码把这种序列叫 <span class="v">coupled sequences</span>），它与已选中的 {0} 相交 -> 被跳过。',
  '为什么必须不相交：ubatch 的 token 数组是<span class="k">按序列集分块</span>拼出来的（每块 n_seq_tokens 个），<br>一个 token 只能落在一个块里 —— 同时属于两个序列集的 token 只能等下一个 ubatch。',
  '选中之后 <span class="v">n_seqs</span> 就定了 —— 它同时决定图里有几条注意力流。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  U.markLines(document, [[10], [1], [10], [19], [35]][i]);
  msg.innerHTML = texts[i];
}));
tl.at(17700, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  U.markLines(document, [1, 10, 19, 28, 35]);
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-05 · 算例',
    title='★ 一个 batch，两张不同形状的图',
    sub='2 个不等长序列（s0 三个 token、s1 一个 token）走一遍 split_equal。',
    caption='锁步扩张（llama-batch.cpp:573 起）：任一序列集耗尽就停，所以每个 ubatch 内各序列集等长。',
    src=SRC_BC, parts=[(569, 603)], duration=26000,
    mark_src=[573, 576, 579, 585, 592, 600],
    notes={7: '先假设每个序列集都还有 token 可分',
           10: '任一序列集耗尽 -> can_expand = false -> 这个 ubatch 到此为止（"等长"就是这么来的）',
           25: '每个序列集各取一个 token —— 锁步（lockstep）',
           34: '次数上限：n_seq_tokens x n_seqs 不能超过 n_ubatch'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="cm" style="margin:0">输入 batch：4 个 token，来自 2 个序列（n_ubatch 足够大，不触上限）</div>
  <div class="row" id="cells" style="gap:6px"></div>
  <div class="row" id="ubs" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const cells = wrap.querySelector('#cells');
const toks = [{ i: 0, s: 's0', c: 'a' }, { i: 1, s: 's0', c: 'a' },
              { i: 2, s: 's0', c: 'a' }, { i: 3, s: 's1', c: 'e' }];
const els = toks.map(t => {
  const e = U.el('div', { class: 'card', style: 'width:112px;padding:5px 7px;border-left-color:var(--border)' });
  e.innerHTML = '<div class="cm" style="margin:0">batch[' + t.i + ']</div>' +
    '<div class="ct" style="font-size:11px;color:var(--' + t.c + ')">' + t.s + '</div>';
  cells.appendChild(e);
  return e;
});

const ubs = wrap.querySelector('#ubs');
const mkU = (title, color, lines) => {
  const e = U.el('div', { class: 'card', style: 'width:330px;border-left-color:var(--' + color + ')' });
  e.innerHTML = '<div class="ct" style="color:var(--' + color + ')">' + title + '</div>' +
    '<div class="cb">' + lines + '</div>';
  ubs.appendChild(e);
  return e;
};
const u1 = mkU('ubatch #1  ·  idxs = [0, 3]', 'b',
  'n_tokens = 2 &nbsp; n_seq_tokens = 1 &nbsp; n_seqs = 2<br>' +
  'n_seqs_unq = 2 &nbsp; equal_seqs() = true');
const u2 = mkU('ubatch #2  ·  idxs = [1, 2]', 'c',
  'n_tokens = 2 &nbsp; n_seq_tokens = 2 &nbsp; n_seqs = 1<br>' +
  'n_seqs_unq = 1 &nbsp; equal_seqs() = true');
u1.style.opacity = '.25'; u2.style.opacity = '.25';

const msg = wrap.querySelector('#msg');
tl.at(700, () => {
  U.markLines(document, [4]);
  msg.innerHTML = 'batch 里 4 个 token：s0 在 batch[0..2]，s1 在 batch[3]。切分按 <span class="k">序列集</span>看，不按顺序。';
});
tl.at(3600, () => {
  [0, 1, 2].forEach(k => { els[k].style.background = 'rgba(88,166,255,.10)'; els[k].style.borderLeftColor = 'var(--a)'; });
  msg.innerHTML = 'seq_set_map[{0}] = [0, 1, 2] —— 序列集 {0} 的 token 在 batch 里的索引，按出现顺序。';
});
tl.at(6800, () => {
  els[3].style.background = 'rgba(247,120,186,.10)'; els[3].style.borderLeftColor = 'var(--e)';
  msg.innerHTML = 'seq_set_map[{1}] = [3]。两个集合互不相交 -> 判据一通过，都进这个 ubatch：<span class="v">n_seqs = 2</span>。';
});
tl.at(10000, () => {
  u1.style.opacity = '1';
  U.markLines(document, [25]);
  msg.innerHTML = '第 1 轮锁步：两个序列集各取 1 个 token -> idxs_per_seq = [[0], [3]]，拼起来 <span class="v">idxs = [0, 3]</span>。';
});
tl.at(13000, () => {
  U.markLines(document, [11]);
  msg.innerHTML = '第 2 轮：序列集 {1} 没有剩余 token -> <span class="k">can_expand = false</span> -> 不再扩张。';
});
tl.at(15800, () => {
  U.markLines(document, [7, 18]);
  msg.innerHTML = '所以 ubatch #1 只有 2 个 token：n_tokens = n_seq_tokens x n_seqs = 1 x 2。';
});
tl.at(18600, () => {
  u1.style.opacity = '.25'; u2.style.opacity = '1';
  msg.innerHTML = '第 2 次调用：只剩序列集 {0} 的 batch[1]、batch[2] -> 一个序列集出 2 个 token：n_seqs = 1、n_seq_tokens = 2。';
});
tl.at(21400, () => {
  U.markLines(document, [34]);
  msg.innerHTML = '第 3 次调用：used[] 全为 true -> 返回 n_tokens = 0 的空 ubatch，切分结束。';
});
tl.at(24000, () => {
  u1.style.opacity = '1'; u2.style.opacity = '1';
  U.markLines(document, [4, 11, 18, 25, 34]);
  msg.innerHTML = '同样是 2 个 token：左边 n_seqs = 2（两条序列各 1 个），右边 n_seqs = 1（单序列 2 个）。<span class="k">图形态不一样。</span>';
});
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-05 · 组装',
    title='ubatch_add：<span class="hl-c">索引列表</span> -> <span class="hl-a">ubatch 字段</span>',
    sub='切分只决定"要哪几个 batch 位置"，字段是这一步算出来的。',
    caption='函数开头有一句 assert(n_tokens % n_seqs == 0)（llama-batch.cpp:752）—— 等长切分的硬保证。',
    src=SRC_BC, parts=[(818, 835)], duration=17000,
    mark_src=[819, 820, 821, 822, 823],
    notes={1: 'equal_seqs 由切分函数传入：split_simple 传 false，split_equal / split_seq 传 true',
           3: 'n_tokens / n_seqs，函数开头断言整除（llama-batch.cpp:752）',
           16: 'udata 是 shared_ptr：裸指针指向它，ubatch 本身可随意拷贝'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row center" style="gap:10px">
    <div class="card" style="width:250px;border-left-color:var(--c)">
      <div class="ct" style="color:var(--c)">切分器的决策：idxs</div>
      <div class="cb">本 ubatch 要哪几个 batch 位置<br>
      <span class="cm" style="margin:0">idxs = [0, 3]  // n_seqs = 2</span></div>
    </div>
    <span class="arrow">-></span>
    <div class="card" style="width:330px;border-left-color:var(--a)">
      <div class="ct" style="color:var(--a)">llama_ubatch res</div>
      <div class="cb">n_tokens = 2 &nbsp; n_seq_tokens = 2/2 = 1 &nbsp; n_seqs = 2<br>
      b_equal_seqs = true &nbsp; n_seqs_unq = 2</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  'ubatch_add 收到的是 <span class="v">idxs</span>（batch 里的下标）和 <span class="v">n_seqs</span>。',
  '<span class="v">n_tokens = idxs.size()</span>，<span class="v">n_seq_tokens = n_tokens / n_seqs</span> —— 不变量在这里落地。',
  '<span class="v">b_equal_seqs</span> 直接抄切分函数传进来的参数，ubatch 自己不判断"等不等长"。',
  'token / embd / pos / n_seq_id / seq_id / output 都是按 idxs 从 batch 里搬过来的（同序，不重排顺序）。',
  '<span class="v">seq_id_unq</span> 与 <span class="v">seq_idx</span> 在这一步建立：去重序列 id，并给出 id -> 下标的映射。',
  '整个 ubatch 只有指针 + 一个 data 共享指针 —— 这就是头文件那句 "keep this struct lightweight"。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [3, 4]); });
tl.at(3700, () => { msg.innerHTML = texts[1]; U.markLines(document, [3, 4, 6]); });
tl.at(6700, () => { msg.innerHTML = texts[2]; U.markLines(document, [1]); });
tl.at(9700, () => { msg.innerHTML = texts[3]; U.markLines(document, [10, 11, 12, 14, 15, 16]); });
tl.at(12700, () => { msg.innerHTML = texts[4]; U.markLines(document, [14, 15, 16, 17]); });
tl.at(14700, () => { msg.innerHTML = texts[5]; U.markLines(document, [0, 18]); });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-05 · 装配',
    title='装配分两层：<span class="hl-b">基类读共性</span>，<span class="hl-c">子类读自己</span>',
    sub='virtual 钩子把"所有架构都一样"和"这个架构特有"分开写。',
    caption='派发在 llama_model_create()（llama-model.cpp:353）与 llama_model_mapping() 的 switch (arch) 里。',
    src=SRC_MH, parts=[(762, 772)], duration=18000,
    mark_src=[764, 765, 766, 767, 769, 770, 771],
    notes={5: 'load_tensors 返回 bool：装配过程可以被进度回调取消',
           8: '架构子类必须实现的两个钩子：自己的超参 + 自己的张量表'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">GGUF 元数据</span><span class="arrow">-></span>
    <span class="chip a">llama_model_create</span><span class="arrow">-></span>
    <span class="chip b">llama_model_llama</span><span class="arrow">-></span>
    <span class="chip c">layers[] 张量</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '基类：所有架构共用', b: 'load_stats / load_hparams / load_vocab / load_tensors<br>按 GGUF 元数据填 hparams 与词表。', m: 'llama_model_base' },
  { c: 'b', t: '子类：这个架构特有', b: 'load_arch_hparams / load_arch_tensors<br>例如 llama_model_llama（src/models/llama.cpp）。', m: 'load_arch_*' },
  { c: 'd', t: '派发：arch -> 具体类', b: 'llama_model_mapping() 的 switch 把<br>LLM_ARCH_* 变成一个 new 出来的模型类。', m: 'switch (arch)' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '装配的入口只有一条：拿到 arch，new 出对应的模型类，然后依次跑四个 load 钩子。',
  '<span class="v">load_stats / load_hparams / load_vocab / load_tensors</span>：基类实现，逻辑与架构无关。',
  '<span class="v">load_arch_hparams / load_arch_tensors</span>：纯虚函数，子类必须自己写。',
  '回顾 L2-02：架构表决定"这个架构有哪些超参、哪些张量"—— 落到代码就是这两个钩子。',
  '<span class="v">build_arch_graph</span> 是第三个纯虚函数，它属于 L2-06 的计算图骨架。',
  '回顾 L2-03：张量的数据来自 GGUF 文件（mmap 或读入）；这一步只建"张量对象"，数据在后面分配 buffer 时落地。'
];
defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  U.markLines(document, [[2, 3, 4, 5], [9, 11], [2, 3]][Math.min(i, 2)]);
  msg.innerHTML = texts[i];
}));
tl.at(9400, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; U.markLines(document, [2, 3, 4, 5]); });
tl.at(12400, () => { msg.innerHTML = texts[4]; U.markLines(document, [12]); });
tl.at(15200, () => { msg.innerHTML = texts[5]; U.markLines(document, [9, 11]); });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-05 · 分配',
    title='层 -> 设备 / buffer：在 <span class="hl-a">load_tensors</span> 里一次算好',
    sub='i_gpu_start 划出"最后 n_gpu_layers 个槽位上 GPU"；输入层永远留在 CPU。',
    caption='每个张量随后按所属层的 buft_list 选 buffer 类型，最终由 ggml_backend_alloc_ctx_tensors_from_buft() 落地（llama-model.cpp:1805）。',
    src=SRC_MC, parts=[(1521, 1546)], duration=21000,
    mark_src=[1521, 1522, 1525, 1527, 1532, 1537, 1540, 1542, 1546],
    notes={1: 'n_gpu_layers 决定有多少个槽位可以上 GPU',
           4: '不在 GPU 窗口内的层，返回 CPU 设备 + CPU buffer 类型表',
           16: '输入层固定放 CPU，源码注释给的理由：offload 输入层收益很小',
           25: '输出层按同一规则单独算一次（它的 il 就是 n_layer_all）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="cm" style="margin:0">示意：8 层模型 + 输入层 + 输出层；每格是 layer_dev{dev, buft_list}</div>
  <div class="row" id="strip" style="gap:4px"></div>
  <div class="row wrap" id="cards" style="gap:7px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const strip = wrap.querySelector('#strip');
const slots = ['in', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'out'];
const cells = slots.map(s => {
  const e = U.el('div', { class: 'card', style: 'width:62px;padding:5px 4px;text-align:center;border-left-width:3px;border-left-color:var(--b)' });
  e.innerHTML = '<div class="ct" style="font-size:10px;color:var(--b)">' + s + '</div>' +
    '<div class="cm" style="margin:0;font-size:8.5px">cpu</div>';
  strip.appendChild(e);
  return e;
});

const defs = [
  { c: 'a', t: 'llama_model::impl::layer_dev', m: 'dev + buft_list', b: '每层记住两件事：哪个设备、<br>候选 buffer 类型表。' },
  { c: 'c', t: 'cpu_buft_list / gpu_buft_list', m: 'make_cpu_buft_list / make_gpu_buft_list', b: '设备在装配开始时枚举一次，<br>含 ACCEL、host、extra 等候选。' },
  { c: 'e', t: '落 buffer', m: 'ggml_backend_alloc_ctx_tensors_from_buft', b: '张量对象按所属层的 buft_list 选一种<br>buffer 类型，然后一次性分配。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => { e.style.opacity = '.30'; });

const msg = wrap.querySelector('#msg');
tl.at(700, () => {
  msg.innerHTML = '先看左边界：<span class="v">i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)</span>。这里 8 层、n_gpu_layers = 0。';
});
tl.at(3700, () => {
  msg.innerHTML = '<span class="k">n_gpu_layers = 0</span> -> i_gpu_start = 9 -> 所有层（含输出层）都拿 CPU 的 buft_list。';
});
tl.at(6700, () => {
  for (let k = 5; k < cells.length; k++) {
    cells[k].style.borderLeftColor = 'var(--a)';
    cells[k].querySelector('.ct').style.color = 'var(--a)';
    cells[k].querySelector('.cm').textContent = 'gpu';
  }
  msg.innerHTML = '<span class="k">n_gpu_layers = 4</span> -> i_gpu_start = max(8 + 1 - 4, 0) = 5 -> 槽位 5..8（L5、L6、L7、输出层）上 GPU。';
});
tl.at(10400, () => {
  cells[0].style.borderLeftColor = 'var(--c)';
  cells[0].querySelector('.cm').textContent = 'cpu 固定';
  msg.innerHTML = '输入层不受这个窗口影响：<span class="v">dev_input</span> 被无条件设成 CPU（注释：offload 输入层收益很小）。';
});
tl.at(13400, () => {
  els[0].style.opacity = '1';
  msg.innerHTML = '每一层得到一个 <span class="v">layer_dev</span>：设备 + 候选 buffer 类型表，存进 <span class="v">dev_layer[]</span>。';
  U.markLines(document, [0, 1, 4, 5, 22, 24]);
});
tl.at(16400, () => {
  els[0].style.opacity = '.30'; els[1].style.opacity = '1';
  msg.innerHTML = '候选表是装配开始时枚举好的（make_cpu_buft_list / make_gpu_buft_list），<br>不在每个张量上重复枚举。';
  U.markLines(document, []);
});
tl.at(19200, () => {
  els[1].style.opacity = '.30'; els[2].style.opacity = '1';
  msg.innerHTML = '张量真正拿内存是在后面：按 buft_list 选一种类型，再由 <span class="v">ggml_backend_alloc_ctx_tensors_from_buft()</span> 整体分配。';
});
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L2-05 · 收束',
    title='把这一课压成一张表',
    sub='切分决定图，装配决定张量 —— 两条线在这里收口。',
    caption='下一课 L2-06 用 ubatch 的字段建出计算图骨架。',
    src=SRC_BH, parts=[(34, 38)], duration=19000,
    mark_src=[34, 35, 36, 37, 38],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['环节', '真实函数 / 字段', '这一课记什么', '在哪一课展开'],
  [['输入', 'llama_batch 的 token/embd/pos/n_seq_id/seq_id/logits', '缺的字段会被 init() 自动补齐', '本课 / L2-07'],
   ['切分', 'split_reset / split_simple / split_equal / split_seq', '三种切法，产物都是 llama_ubatch', '本课'],
   ['产物', 'n_tokens / n_seq_tokens / n_seqs / n_seqs_unq / equal_seqs()', 'n_tokens = n_seq_tokens x n_seqs', 'L2-06 图输入形状'],
   ['组装', 'ubatch_add() 里的 llama_ubatch res { ... }', '裸指针指向共享的 data_t，ubatch 很轻', 'L2-07 decode 循环'],
   ['装配', 'load_arch_hparams / load_arch_tensors', '架构表决定装哪些张量', 'L2-02 / L2-10'],
   ['分配', 'layer_dev / dev_layer / cpu_buft_list / gpu_buft_list', '每层落到设备与候选 buffer 类型', 'L2-03 / L3-02'],
   ['落地', 'ggml_backend_alloc_ctx_tensors_from_buft', '权重在这里真正拿到 buffer', 'L2-03 / L4-03']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '一个 batch 里有两个序列：<span class="mono">s0</span> 有 4 个 token、<span class="mono">s1</span> 有 2 个 token，' +
  '<span class="mono">n_ubatch</span> 足够大。用 <span class="mono">split_equal(n_ubatch, true, 0)</span> 切，' +
  '会切出几个 ubatch？每个的 <span class="mono">n_tokens / n_seq_tokens / n_seqs</span> 是多少？' +
  '<span class="mono">equal_seqs()</span> 返回什么？',
  '2 个 ubatch，<span class="mono">equal_seqs()</span> 都是 <b>true</b>（split_equal 恒定传 true）：<br>' +
  '① 锁步前两轮两个序列集都有 token，第 3 轮 <span class="mono">s1</span> 耗尽 -> ' +
  '<span class="mono">n_tokens = 4</span>、<span class="mono">n_seq_tokens = 2</span>、<span class="mono">n_seqs = 2</span>。<br>' +
  '② 只剩 <span class="mono">s0</span> 的 2 个 token -> ' +
  '<span class="mono">n_tokens = 2</span>、<span class="mono">n_seq_tokens = 2</span>、<span class="mono">n_seqs = 1</span>。<br>' +
  '③ 第 3 次调用 <span class="mono">used[]</span> 全为 true -> 返回 <span class="mono">n_tokens = 0</span>，切分结束。<br>' +
  '验算：两个 ubatch 都满足 <span class="mono">n_tokens = n_seq_tokens x n_seqs</span>（4 = 2 x 2，2 = 2 x 1）。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '七个环节，两条线（切分 / 装配）。',
  '<span class="k">切分线</span>：batch -> ubatch。同长度的序列集被合并进同一个 ubatch，走同一条图。',
  '<span class="k">产物线</span>：ubatch 的五个计数字段就是图输入的形状来源。',
  '<span class="k">装配线</span>：arch 决定类，钩子决定张量表，layer_dev 决定放哪。',
  '一句话：<span class="v">batch 是调用方的清单，ubatch 是执行单位，装配决定这些计算跑在哪些张量、哪个设备上</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2500 + i * 2100, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
}));
tl.at(17300, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、batch：只有 token 是必给的',
    '`llama_batch_allocr::init()` 先校验（token 范围、seq_id 范围），再**补全缺失字段**。'
    '补出来的默认值决定了后面"序列集"的形态：`seq_id` 指向默认序列 0、`pos` 从记忆模块里该序列的最大位置往后接。\n\n'
    '注意 `pos` 的补法：源码为每个序列维护一个游标 `p0[s]`，每给一个 token 就 `p0[seq_id] = pos[i] + 1`，'
    '所以**同一个序列在 batch 里的位置必须连续递增** —— 这条约束是后面 `split_equal` 判据二的前提。',
    src=SRC_BC, parts=[(73, 88)], lang='c')

L.section(
    '二、★ struct llama_ubatch：一个不变量串起所有字段',
    '按用途分五组读：计数（`n_tokens` / `n_seq_tokens` / `n_seqs` / `n_seqs_unq` / `n_pos`）、'
    '数据（`token` / `embd` / `pos`）、归属（`n_seq_id` / `seq_id` / `seq_id_unq` / `seq_idx`）、'
    '输出（`output`）、标记（`b_equal_seqs`）。\n\n'
    '**不变量**：`n_tokens = n_seq_tokens * n_seqs`。字段注释里就写着 `total tokens (n_seq_tokens * n_seqs)`，'
    '`ubatch_add()` 也是按 `n_tokens/n_seqs` 算出 `n_seq_tokens` 的。\n\n'
    '`b_equal_seqs` 的类型值得一提：语义上是布尔，源码故意写成 `uint32_t`，'
    '注释给的理由是"否则 address sanitizer 会报错"（按 `int32_t` 对齐）。',
    src=SRC_BH, parts=[(30, 52)], lang='c')

L.section(
    '三、三种切法',
    '头文件把三种切法的用途写在注释里：`split_simple` 面向"数量未知、长短不一的序列集"，'
    '`split_equal` 产出"等长序列集"的 ubatch，`split_seq` 每个 ubatch 只装一个序列集。\n\n'
    '它们在实现上的差别只在传给 `ubatch_add()` 的 `n_seqs` 与 `equal_seqs` 上：\n\n'
    '```text\n'
    'split_simple : ubatch_add(idxs, idxs.size(), false)   -> n_seqs = n_tokens, n_seq_tokens = 1\n'
    'split_equal  : ubatch_add(idxs, n_seqs, true)         -> 等长锁步切分\n'
    'split_seq    : ubatch_add(idxs, 1, true)              -> n_seqs = 1\n'
    '```\n\n'
    '调用点按"流的条数"选：`llama_kv_cache::init_batch()` 里 `n_stream == 1` 用 `split_simple`，'
    '否则用 `split_equal`（`src/llama-kv-cache.cpp:713`，属于 L2-04 的覆盖范围，不计入本课覆盖率）。',
    src=SRC_BH, parts=[(99, 111)], lang='c')

L.section(
    '四、★ split_equal 的两条判据',
    '第一个循环只做一件事：**选出这个 ubatch 的参与序列集**。两个判据都由它保证：\n\n'
    '1. **互不相交**：`(cur_seq_set[s] & seq_set[i]).none()` —— 一个 token 不能同时属于本 ubatch 的两个序列集。'
    '相交意味着两条流共享 KV 位置，只能留到下一个 ubatch。\n'
    '2. **序列号递增**（`sequential == true` 时）：`batch.seq_id[i][0] == last_seq_id + 1`。\n\n'
    '选中之后 `n_seqs = cur_seq_set.size()`，接下来才是锁步扩张。',
    src=SRC_BC, parts=[(521, 553)], lang='c')

L.section(
    '五、ubatch_add：从索引到字段',
    '`ubatch_add()` 的输入是"要哪几个 batch 位置"（`idxs`）和"几个序列集"（`n_seqs`），'
    '开头一句 `assert(n_tokens%n_seqs == 0);` 就是等长切分的硬保证 —— 切分器不可能传进不整除的组合。\n\n'
    '函数体后半段把这些搬进一个 `llama_ubatch res { ... }`：计数字段现算，数据指针指向 `udata`（`shared_ptr`）里的 vector。'
    '所以 `llama_ubatch` 本身可以随意拷贝，数据始终只有一份。',
    src=SRC_BC, parts=[(749, 757)], lang='c')

L.section(
    '六、模型装配：五个 load 钩子 + 两个架构钩子',
    '`llama_model` 把装配拆成两层：基类实现与架构无关的 `load_stats` / `load_hparams` / `load_vocab` / `load_tensors`，'
    '架构子类只实现 `load_arch_hparams` / `load_arch_tensors`（以及属于 L2-06 的 `build_arch_graph`）。\n\n'
    '派发链是：`llama_model_create(llama_model_loader &)` 取 arch -> `llama_model_create(llm_arch, params)` '
    '-> `llama_model_mapping()` 的 `switch (arch)` 里 `new llama_model_*`（`src/llama-model.cpp:353` / `:44`）。\n\n'
    '以 `LLM_ARCH_LLAMA` 为例，子类是 `llama_model_llama`（`src/models/models.h:147`，由 L2-10 覆盖），'
    '它把 `load_arch_tensors()` 写在 `src/models/llama.cpp:34`。',
    src=SRC_MH, parts=[(762, 772)], lang='c')

L.section(
    '七、设备与 buffer：每层的 layer_dev 在装配时定下来',
    '`load_tensors()` 先枚举设备的 buffer 类型表（`make_cpu_buft_list()` / `make_gpu_buft_list()`，'
    '`src/llama-model.cpp:1061` / `:1123`），然后给每一层算一个 `layer_dev{dev, buft_list}`：\n\n'
    '* `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`：从倒数第 `n_gpu_layers` 个槽位开始上 GPU；\n'
    '* 输入层例外，`dev_input` 无条件设成 CPU（注释：offload 输入层收益很小）；\n'
    '* 输出层用同一个 lambda、以 `il = n_layer_all` 单独算一次。\n\n'
    '结果存进 `llama_model::impl::dev_layer`，之后建张量时按它选 buffer 类型，'
    '最后在 `load_tensors()` 的后半段由 `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)` 真正分配'
    '（`src/llama-model.cpp:1805`）。',
    src=SRC_MC, parts=[(1521, 1546)], lang='c')

L.section(
    '八、可观测性：LLAMA_LOG_* 与 format()',
    '装配和切分都不是"黑盒一步"：切分器用 `LLAMA_LOG_DEBUG` 打印每个 ubatch 的全部字段'
    '（`llama-batch.cpp` 里 35 处、`llama-model.cpp` 里 109 处 `LLAMA_LOG_*`），'
    '错误路径用 `format()` 拼消息（`llama-model.cpp` 里 10 处）。\n\n'
    '宏定义在 `llama-impl.h`：日志级别宏转发到 `llama_log_internal()`，'
    '`LLAMA_ATTRIBUTE_FORMAT` 让编译器检查格式串 —— 这类日志正是排查"为什么切成了这样"的第一手材料。',
    src=SRC_IH, parts=[(27, 32)], lang='c')

L.section(
    '九、装配计时：time_meas',
    '`time_meas` 是一个 RAII 计时器：构造时取时间戳，析构时把差值累加进引用进来的累加器。'
    '模型加载路径用 `time_meas tm(model->t_load_us);` 统计装配耗时（`src/llama.cpp:340`，由 L2-01 覆盖），'
    '采样路径用同一件工具统计 `t_sample_us`。\n\n'
    '它和本课主题的关系：装配是**一次性的大开销**（读文件 + 建张量 + 分配 buffer），'
    '切分则是**每次 decode 都要做的小开销** —— 两类成本分开计量，才有 `llama_model::t_load_us` 这类字段。',
    src=SRC_IC, parts=[(20, 26)], lang='c')

L.footnote_add('本课声明并逐字引用 6 个源文件，全部计入覆盖率：`src/llama-batch.cpp`、`src/llama-batch.h`、'
               '`src/llama-model.cpp`、`src/llama-model.h`、`src/llama-impl.cpp`、`src/llama-impl.h`。')
L.footnote_add('跨课呼应处引用了其他课的文件与行号（`src/llama-kv-cache.cpp:713`、`src/llama-context.cpp`、'
               '`src/llama-graph.h/.cpp`、`src/models/models.h:147`、`src/models/llama.cpp:34`、`src/llama.cpp:340`），'
               '这些**不在本课的 `src=` 声明里，不计入本课覆盖率** —— 它们分别属于 L2-04 / L2-06 / L2-07 / L2-10 / L2-01。')

L.prereqs('`L2-04`（KV cache 与记忆家族）；同层的 `L2-02`（架构表与超参）与 `L2-03`（权重加载）是本课的直接上游。')

L.goal(
    '说出 `llama_batch` 里哪些字段是必给的、哪些会被 `init()` 自动补齐（以及补齐时位置从哪里接）；',
    '写出 `llama_ubatch` 的计数不变量 `n_tokens = n_seq_tokens * n_seqs`，并说明 `equal_seqs()` 是谁赋的值；',
    '给定一个含两个不等长序列的 batch，写出 `split_equal` 切出的每个 ubatch 的 `(n_tokens, n_seq_tokens, n_seqs)`'
    '（对应本课验收点）；',
    '说清模型装配的两层钩子（`load_stats/load_hparams/load_vocab/load_tensors` 与 '
    '`load_arch_hparams/load_arch_tensors`），以及"层到设备"的分配发生在哪一步。')

L.conclusion(
    'batch 是清单，ubatch 是执行单位',
    '一次 `llama_decode` 收到一个 `llama_batch`，但它不会按整个 batch 建图：'
    '`llama_batch_allocr` 先补齐字段，再按 `n_ubatch` 切成若干 `llama_ubatch`，'
    '**每个 ubatch 建一次图、算一次**。所以"切分方式"就是"图的数量与形状"。')
L.conclusion(
    '★ ubatch 的形状 = 五个计数字段',
    '`n_tokens`、`n_seq_tokens`、`n_seqs`、`n_seqs_unq`、`equal_seqs()`。'
    '其中 `n_tokens = n_seq_tokens * n_seqs` 由 `ubatch_add()` 保证（函数开头断言整除）。\n\n'
    '`split_equal` 会把**长度相同的一段**合并进同一个 ubatch（锁步扩张，任一序列集耗尽即停），'
    '于是多个序列可以共用同一条图；`split_simple` 则按 batch 顺序切片，`n_seq_tokens` 恒为 1。')
L.conclusion(
    '★ 装配在 load_tensors 里一次算完',
    '`llama_model_create(ml, params)` 按 GGUF 里的 arch `new` 出具体模型类，'
    '基类跑四个 `load_*` 钩子，子类跑 `load_arch_hparams` / `load_arch_tensors`；'
    '设备与候选 buffer 类型在 `load_tensors()` 里按层算成 `dev_layer[]`，'
    '最后 `ggml_backend_alloc_ctx_tensors_from_buft()` 把张量落到真实 buffer ——'
    '这正是 L2-03 那条"权重从文件到 buffer"的终点。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
