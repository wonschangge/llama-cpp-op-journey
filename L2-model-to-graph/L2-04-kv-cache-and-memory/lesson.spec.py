#!/usr/bin/env python3
"""L2-04 · KV cache 与记忆家族 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-04-kv-cache-and-memory/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC = 'src/llama-memory.h'

L = Lesson(
    id='L2-04',
    layer='L2 · 从模型到图',
    title='KV cache 与记忆家族',
    codecap='src/llama-memory*.{h,cpp} / llama-kv-cache*.{h,cpp}（逐字引用）',
    nav={'prev': {'href': '../L2-03-weight-loading-mmap/index.html', 'label': 'L2-03 权重加载与内存映射'},
         'next': {'href': '../L2-05-batch-and-model-build/index.html', 'label': 'L2-05 批、解码参数与模型装配'}},
)

# ---------------------------------------------------------------- 覆盖声明
# 计划里 L2-04 的 23 个文件，全量声明（本课是这一族的总课）。
L.cover(
    'src/llama-kv-cache.cpp', 'src/llama-kv-cache.h',
    'src/llama-kv-cache-iswa.cpp', 'src/llama-kv-cache-iswa.h',
    'src/llama-kv-cache-dsa.cpp', 'src/llama-kv-cache-dsa.h',
    'src/llama-kv-cache-dsa-iswa.cpp', 'src/llama-kv-cache-dsa-iswa.h',
    'src/llama-kv-cache-dsv4.cpp', 'src/llama-kv-cache-dsv4.h',
    'src/llama-kv-cache-msa.cpp', 'src/llama-kv-cache-msa.h',
    'src/llama-kv-cells.h',
    'src/llama-memory.cpp', 'src/llama-memory.h',
    'src/llama-memory-hybrid.cpp', 'src/llama-memory-hybrid.h',
    'src/llama-memory-hybrid-idx.cpp', 'src/llama-memory-hybrid-idx.h',
    'src/llama-memory-hybrid-iswa.cpp', 'src/llama-memory-hybrid-iswa.h',
    'src/llama-memory-recurrent.cpp', 'src/llama-memory-recurrent.h',
    # 额外一个：memory 实现的选择点（计划里归 L2-05，本课引用后同样计入覆盖）
    'src/llama-model.cpp',
)

L.note('**一句话**：推理时的"记忆"不是一种东西，而是一族东西 —— 同一个 `llama_memory_i` 接口下，'
       '既有按 token 记 K/V 的 KV cache，也有 SSM/RNN 那种每层只有一份固定状态的 memory，'
       '还有把两者按层拼起来的 hybrid。')
L.note('回顾 L2-03：那一课讲的是**权重**怎么从磁盘进内存（mmap）。这一课讲**运行期**的记忆：'
       'decode 过程中新产生的 K/V 或状态存在哪、谁能读写、谁来决定用哪一种。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2-04 · 全局',
    title='为什么不是"一个 KV cache"',
    sub='KV cache 只是 LLM memory 的一种。源码把这一点写在接口的第一行注释里。',
    caption='覆盖计划里 L2-04 的全部 23 个文件，另加分派点所在的 src/llama-model.cpp（create_memory 在 2274 行）。',
    src=SRC, parts=[(71, 83)], duration=20000,
    mark_src=[73, 75, 79, 81],
    notes={3: '三类回调：层过滤 / 层复用 / 层共享 —— 差异从这里开始被表达出来'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">模型架构</span><span class="arrow">-></span>
    <span class="chip a">create_memory()</span><span class="arrow">-></span>
    <span class="chip b">llama_memory_i</span><span class="arrow">-></span>
    <span class="chip c">K/V 张量 · 状态张量</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'g', t: '问题', b: '不是所有模型都有 KV：<br>SSM/RNN 每层只有一份固定状态；<br>SWA / 稀疏注意力只要窗口', m: 'mem_cell 里没有 K/V' },
  { c: 'a', t: '抽象', b: '一批虚函数，描述"记忆"对外的<br>全部行为：生命周期 / 序列操作 / 状态 IO', m: 'struct llama_memory_i' },
  { c: 'b', t: '实现家族', b: 'KV cache 一族 6 个类 + recurrent 1 个<br>+ hybrid 一族 3 个类', m: '10 个类，9 个直接继承' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => U.card(d, { style: 'width:220px' }));
els.forEach(e => host.appendChild(e));
els.forEach(e => e.style.opacity = '.32');
const msg = wrap.querySelector('#msg');
const texts = [
  '接口注释写得很直白：<span class="v">the KV cache is a type of LLM memory, but there can be other types</span>。',
  '<span class="k">问题</span>：Mamba/RWKV 这类模型没有 per-token 的 K/V；<br>SWA 层只需要最近 n_swa 个 token。硬塞进一个类会到处是 if。',
  '<span class="k">抽象</span>：只把"记忆对外的行为"写成虚函数；<br>数据面（K/V 张量、状态张量）留在各自实现里。',
  '<span class="k">实现</span>：本课逐个对照它们各自解决什么问题 —— 见第 8 幕的对照表。',
  '一句话：<span class="k">抽象的是"行为"，不是"数据布局"</span>。'
];
defs.forEach((_, i) => tl.at(800 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
}));
tl.at(12000, () => { msg.innerHTML = texts[3]; });
tl.at(16000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-04 · 核心',
    title='★ <span class="hl-a">llama_memory_i</span> 的虚接口：三类职责',
    sub='生命周期、序列操作、状态读写。整族 memory 的共同语言就是这 15 个纯虚函数。',
    caption='注意：接口里没有 push_kv() / get_k()。图和 cell 之间的数据面不在这个接口上。',
    src=SRC, parts=[(85, 127)], duration=26000,
    mark_src=[88, 94, 98, 108, 110, 116, 125],
    notes={3: 'init_batch：把一批 token 切成 ubatch，并检查能否装下；返回一个 context 对象',
           9: 'init_full：假装 cache 是满的，用于分配最坏情况的 compute buffer',
           13: 'init_update：把挂起的 shift / copy 等更新做成一个 context',
           23: 'clear(data)：data=true 时连数据一起清',
           40: 'state_write/read：序列状态的存盘与恢复 —— 会话保存走的就是它'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '生命周期', m: 'init_batch · init_full · init_update', b: '每次 decode 前先问它："这批 token 放得下吗、放在哪些 cell？"' },
  { c: 'b', t: '序列操作', m: 'clear · seq_rm/cp/keep/add/div · seq_pos_min/max', b: '按 seq_id 与位置区间增删改查 —— 多序列推理的基础' },
  { c: 'c', t: '度量与状态', m: 'get_can_shift · memory_breakdown · state_write/read', b: '能不能整体平移、占了多少显存、状态怎么存盘恢复' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => U.card(d, { style: 'width:220px' }));
els.forEach(e => host.appendChild(e));
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '15 个纯虚函数，按职责分三组看：',
  '<span class="v">生命周期</span>：三个 <span class="k">init_*</span> 都返回 <span class="k">llama_memory_context_ptr</span>，<br>而不是直接改状态 —— 这是第 3 幕的主题。',
  '<span class="v">序列操作</span>：全部以 <span class="k">(seq_id, p0, p1)</span> 为参数，<br>说明"记忆"是按<span class="k">序列 + 位置区间</span>管理的，不是按数组下标。',
  '<span class="v">度量与状态</span>：<span class="k">memory_breakdown()</span> 让上层能报告显存占用；<br><span class="k">state_write/read</span> 让整个记忆可序列化。',
  '★ 记住这一点：<span class="k">接口管"哪些 cell 属于哪个序列"，不管"K/V 长什么样"</span>。<br>所以 recurrent memory 可以用同一套接口，却完全没有 K/V。'
];
defs.forEach((_, i) => tl.at(800 + i * 3800, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(19000, () => { msg.innerHTML = texts[3]; });
tl.at(24000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-04 · 核心',
    title='★ <span class="hl-a">context</span> 对象：apply() 是唯一改状态的入口',
    sub='init_* 只做规划，不落笔；真正写进 memory 的动作只有 apply()。',
    caption='组合型 memory（iswa / dsa / hybrid）靠 status 的合并来伪装成一个 memory：src/llama-memory.h:37-39。',
    src=SRC, parts=[(44, 69)], duration=19000,
    mark_src=[50, 51, 56, 60, 63, 66],
    notes={6: '这一行是整个设计的核心断言'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">init_batch()</span><span class="arrow">-></span>
    <span class="chip b">context</span><span class="arrow">-></span>
    <span class="chip c">next() 逐个 ubatch</span><span class="arrow">-></span>
    <span class="chip d">apply() 落笔</span>
  </div>
  <div class="row" id="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
const right = wrap.querySelector('#right');
left.innerHTML = '<div class="cm" style="margin:0">context 的四个方法</div>';
const mets = [
  ['next()', '推进到下一个 ubatch；返回 false 表示没有了'],
  ['apply()', '把当前 ubatch 的记忆状态写进 memory'],
  ['get_ubatch()', '取出当前 ubatch，交给建图'],
  ['get_status()', '成功 / 无需更新 / 准备失败 / 计算失败']
];
const mEls = mets.map(m => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
  e.innerHTML = '<span class="m">' + U.esc(m[0]) + '</span> &nbsp;' + U.esc(m[1]);
  left.appendChild(e);
  return e;
});
mEls.forEach(e => e.style.opacity = '.30');

right.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">为什么先规划后落笔？</div>' +
  '<div class="cb">因为"能不能装下"必须在建图之前回答。context 里存着 <span class="cm" style="margin:0">slot_info_vec_t</span> ' +
  '与 ubatch 列表，规划失败就直接返回错误状态，memory 一个字节都没动。</div></div>' +
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">组合靠什么？</div>' +
  '<div class="cb">子 memory 各自返回 status，用 <span class="cm" style="margin:0">llama_memory_status_combine()</span> 合并：' +
  '任一失败即失败，任一有更新即有更新（src/llama-memory.cpp:3-41）。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '三步走：<span class="k">init_batch 规划</span> -> <span class="k">next() 遍历</span> -> <span class="k">apply() 落笔</span>。',
  '<span class="v">next()</span> 只是游标前进，不碰 memory；<br>所以一个 batch 的多个 ubatch 可以先全部规划好。',
  '<span class="v">apply()</span> 才把 cell 的 pos / seq / ext 写下去 ——<br>源码注释原话：<span class="k">the only method that should mutate the memory</span>。',
  '<span class="v">get_status()</span> 是错误处理的统一出口；<br>组合型 memory 把子 context 的 status 合并后再上报。',
  '一句话：<span class="k">规划与提交分离</span>，所以"放不下"可以优雅失败，而不会留下半个批次的状态。'
];
tl.at(800, () => { msg.innerHTML = texts[0]; });
mets.forEach((_, i) => tl.at(4000 + i * 2600, () => {
  mEls.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[Math.min(i + 1, 3)];
}));
tl.at(16000, () => { mEls.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-04 · 核心',
    title='★ <span class="hl-c">cell</span>：KV cache 的最小单位是"元数据"，不是 K/V',
    sub='一个 cell 记的是"这个位置上是谁的 token、属于哪些序列"，K/V 本身在另一个张量里。',
    caption='llama_kv_cells 的成员全是元数据：pos / ext / shift / seq / seq_pos / used。',
    src='src/llama-kv-cells.h', parts=[(485, 524)], duration=24000,
    mark_src=[486, 489, 491, 494, 511, 514, 524],
    notes={1: 'has_shift：本轮的 pos 变更还没同步到 K 张量（见下一幕的 set_input_k_shift）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:8px">
    <div class="col grow" id="diag" style="gap:6px"></div>
    <div class="col" id="side" style="gap:7px;width:246px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const diag = wrap.querySelector('#diag');
diag.innerHTML = '<div class="cm" style="margin:0">8 个 cell（每个 = 一个位置）</div>' +
  '<div class="row" id="cells" style="gap:5px"></div>' +
  '<div class="cm" style="margin:2px 0 0" id="cap">pos: -1 表示空 cell；seq 是 LLAMA_MAX_SEQ 位的位集合</div>';

const host = diag.querySelector('#cells');
const cs = [];
for (let i = 0; i < 8; i++) {
  const c = U.el('div', { style: 'width:66px;height:52px;border:1px solid var(--border);border-radius:5px;background:#10151b;padding:3px 4px;font-family:var(--mono);font-size:9px;color:var(--dim)' });
  c.innerHTML = '<div style="color:var(--dim)">#' + i + '</div><div class="pv">pos -</div><div class="sv">seq -</div>';
  host.appendChild(c); cs.push(c);
}

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--a)"><div class="ct" style="color:var(--a)">pos[]</div>' +
  '<div class="cb">每个 cell 一个位置；<span class="cm" style="margin:0">-1 = 空</span>。</div></div>' +
  '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">seq[]</div>' +
  '<div class="cb">位集合：一个 cell 可以同时属于多个序列（前缀共享）。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">ext[]</div>' +
  '<div class="cb">M-RoPE 的 x/y、多模态 token 号 —— 需要随状态存盘的额外信息。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)"><div class="ct" style="color:var(--d)">used / seq_pos</div>' +
  '<div class="cb">used 是"非空 cell 的下标集合"；seq_pos 是 (pos, cell) 有序集合，' +
  '<span class="cm" style="margin:0">O(log n)</span> 找某序列最近的 cell。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '一个 cell 只有元数据。K/V 在别处：<span class="k">llama_kv_cache::layers[].k/v</span>（llama-kv-cache.h:250-260）。',
  '<span class="v">pos[i] == -1</span> 就是"空 cell"。所有分配逻辑都在找连续的空 cell 区间。',
  '<span class="v">seq[i]</span> 是位集合 —— 所以<b>多个序列可以共享同一个 cell</b>，这是公共前缀只存一份的原因。',
  '<span class="v">ext[i]</span> 存 M-RoPE 的二维位置或 token id；<span class="k">has_cell_ext()</span> 决定存盘要不要带上它。',
  '<span class="v">shift[i]</span> 把"位置平移"攒起来，最后一次性同步到 K 张量，而不是每改一次就重算。',
  '一句话：<span class="k">cell 是"账本"，K/V 张量是"仓库"</span>。第 6 幕会看到两者怎么被图接口对上。'
];
tl.at(600, () => {
  msg.innerHTML = texts[0];
  [2, 3, 4].forEach(i => {
    cs[i].style.borderColor = 'var(--a)';
    cs[i].querySelector('.pv').textContent = 'pos ' + i;
    cs[i].querySelector('.sv').textContent = 'seq 0';
    cs[i].style.color = 'var(--text)';
  });
  cs[5].style.borderColor = 'var(--b)';
  cs[5].querySelector('.pv').textContent = 'pos 5';
  cs[5].querySelector('.sv').textContent = 'seq 0,1';
  cs[5].style.color = 'var(--text)';
});
tl.at(4200, () => { msg.innerHTML = texts[1]; });
tl.at(8000, () => { msg.innerHTML = texts[2]; cs[5].style.background = 'rgba(63,185,80,.14)'; });
tl.at(11800, () => { msg.innerHTML = texts[3]; });
tl.at(15600, () => {
  msg.innerHTML = texts[4];
  cs[2].style.background = 'rgba(210,153,34,.16)'; cs[3].style.background = 'rgba(210,153,34,.16)'; cs[4].style.background = 'rgba(210,153,34,.16)';
});
tl.at(20000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-04 · 机制',
    title='<span class="hl-b">slot_info</span>：token 到 cell 的翻译表',
    sub='建图之前必须先知道"第 i 个 token 写进哪个 cell"，这张表就是 slot_info。',
    caption='落笔的地方是 apply_ubatch：先 cells.rm() 再 cells.pos_set()（llama-kv-cache.cpp:1120-1131）。',
    src='src/llama-kv-cache.h', parts=[(32, 43)], duration=20000,
    mark_src=[33, 34, 36, 39, 40, 42, 43],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="cm" style="margin:0">ubatch：4 个 token</div>
  <div class="row" id="tok" style="gap:6px"></div>
  <div class="flow" style="justify-content:center"><span class="arrow">|</span><span class="chip a">slot_info.idxs[0]</span><span class="arrow">|</span></div>
  <div class="row" id="cel" style="gap:6px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const tokHost = wrap.querySelector('#tok');
const celHost = wrap.querySelector('#cel');
const toks = ['t0', 't1', 't2', 't3'];
const cels = ['#5', '#6', '#7', '#8'];
const tEls = toks.map((t, i) => {
  const e = U.el('div', { class: 'chip', style: 'width:74px;text-align:center', text: t + '  p=' + (i + 4) });
  tokHost.appendChild(e); return e;
});
const cEls = cels.map((c, i) => {
  const e = U.el('div', { class: 'chip b', style: 'width:74px;text-align:center', text: 'cell ' + c });
  celHost.appendChild(e); return e;
});
[tEls, cEls].forEach(arr => arr.forEach(e => e.style.opacity = '.35'));

const msg = wrap.querySelector('#msg');
const texts = [
  '源码注释把这张表的语义写死了：<span class="v">token[i] -> goes to cells[idxs[i]]</span>。',
  '<span class="v">s0 / s1</span>：这次 ubatch 覆盖的 stream 区间（<span class="k">ns = s1 - s0 + 1</span>）。' +
  '<br>unified 模式下 ns = 1，否则一个序列一个 stream。',
  '<span class="v">strm[] / idxs[]</span>：每个 stream 一行；idxs 里就是该 stream 每个 token 的目标 cell 下标。',
  '<span class="v">is_contiguous()</span>：判断这些 cell 是不是从 head() 开始连续 —— ' +
  '<br>连续时图里可以用更省的写法。',
  '一句话：<span class="k">slot_info 是"规划结果"的载体</span>，第 3 幕的 context 里存的就是它的向量。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3600, () => { msg.innerHTML = texts[1]; });
tl.at(7200, () => {
  msg.innerHTML = texts[2];
  tEls.forEach((e, i) => { e.style.opacity = '1'; });
});
tl.at(11000, () => {
  msg.innerHTML = texts[3];
  cEls.forEach((e, i) => { e.style.opacity = '1'; e.style.borderColor = 'var(--b)'; });
});
tl.at(15500, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-04 · 机制',
    title='K/V 住在哪，图怎么读写它',
    sub='cache 自己拿 buffer，图只拿视图（get_k）、只写行（cpy_k）。',
    caption='跨课：L2-06 会看到 build_attn 用 get_k/get_v 造 KQ、用 cpy_k/cpy_v 写回；'
            'L4-02 会看到调度器为什么不碰这块 buffer。',
    src='src/llama-kv-cache.cpp', parts=[(276, 295), (1266, 1283), (1346, 1351)],
    duration=24000,
    mark_src=[282, 285, 1278, 1350],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '归属：cache 自己分配', m: 'ggml_backend_alloc_ctx_tensors_from_buft', b: '每个后端 buffer type 一个 ggml context，' +
    '整块 K/V 张量一次性分配；<b>调度器不参与</b>。' },
  { c: 'b', t: '读：get_k / get_v', m: 'ggml_view_4d', b: '返回 K 张量的一个四维视图：' +
    '<br>[n_embd_head_k, n_head_kv, n_kv, ns]。<b>零拷贝</b>。' },
  { c: 'c', t: '写：cpy_k / cpy_v', m: 'ggml_set_rows', b: '把本次 ubatch 的 K 按 k_idxs（来自 slot_info）' +
    '<br>写进对应行。写入是图上的一个节点。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => U.card(d, { style: 'width:220px' }));
els.forEach(e => host.appendChild(e));
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '三个事实，对应左边三段引用：',
  '<span class="k">归属</span>：K/V 张量在 llama_kv_cache 构造时按 buffer type 整块分配（cpp:276-294）。' +
  '<br>no_alloc 模式下挂 dummy buffer，注释写明是为了让 <span class="v">backend scheduler</span> 不去分配它。',
  '<span class="k">读</span>：get_k 返回 <span class="v">ggml_view_4d</span> —— 只是换一套 ne/nb，' +
  '<br>所以 attention 读整个 cache 不产生任何拷贝（cpp:1278-1283）。',
  '<span class="k">写</span>：cpy_k 最终落到 <span class="v">ggml_set_rows(ctx, k, k_cur, k_idxs)</span>（cpp:1350）。' +
  '<br>k_idxs 就是第 5 幕 slot_info 里的 idxs —— 规划与执行在这里闭环。',
  '为什么重要：<span class="k">KV cache 的生命周期与计算图的中间张量完全不同</span>。' +
  '<br>它跨 decode 存活、自己管 buffer，所以不能交给 L4-01 的 arena 分配器。'
];
defs.forEach((_, i) => tl.at(800 + i * 4600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(16000, () => { msg.innerHTML = texts[3]; });
tl.at(21000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-04 · 变体',
    title='<span class="hl-d">iswa</span>：把层切成两套 cache',
    sub='SWA 层的记忆只需要一个窗口，不需要整个上下文 —— 那就给它一个更小的 cache。',
    caption='同一个类名 llama_kv_cache，构造参数不同（n_swa / swa_type），行为就不同。',
    src='src/llama-kv-cache-iswa.cpp', parts=[(52, 73), (95, 105)], duration=23000,
    mark_src=[58, 66, 73, 98, 105],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:10px">
    <div class="card grow" style="border-left-color:var(--a)" id="cb">
      <div class="ct" style="color:var(--a)">kv_base · 非 SWA 层</div>
      <div class="cb">filter_base: <span class="cm" style="margin:0">!is_swa(il)</span><br>
      size = kv_size（= n_ctx_seq）<br>n_swa = 0, swa_type = NONE</div>
      <div class="bt" style="height:12px;background:#10151b;border:1px solid var(--border);border-radius:3px;margin-top:6px;overflow:hidden">
        <div id="fb" style="height:100%;width:0;background:var(--a);transition:width .6s"></div></div>
    </div>
    <div class="card grow" style="border-left-color:var(--b)" id="cs">
      <div class="ct" style="color:var(--b)">kv_swa · SWA 层</div>
      <div class="cb">filter_swa: <span class="cm" style="margin:0">is_swa(il)</span><br>
      size = min(kv_size, n_swa*(unified?n_seq_max:1)+n_ubatch)<br>n_swa / swa_type 取自 hparams</div>
      <div class="bt" style="height:12px;background:#10151b;border:1px solid var(--border);border-radius:3px;margin-top:6px;overflow:hidden">
        <div id="fs" style="height:100%;width:0;background:var(--b);transition:width .6s"></div></div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  '两个 filter 把模型的层一分为二：<span class="v">is_swa(il)</span> 为真进 SWA cache，否则进 base。',
  '<span class="k">base cache</span> 拿满 <span class="v">kv_size</span>（整个上下文）；' +
  '构造时 <span class="v">n_swa = 0</span>、<span class="v">swa_type = NONE</span>，即"没有窗口"的普通 cache。',
  '<span class="k">SWA cache</span> 的大小只跟窗口有关：' +
  '<span class="v">GGML_PAD(min(size_base, n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256)</span>。' +
  '<br>注意末尾 pad 到 256，注释说是为了性能。',
  '<span class="v">swa_full</span> 开关会把 SWA cache 放大到和 base 一样大 —— 调试用，默认关。',
  '一句话：<span class="k">同一份代码，两种规模</span>。这就是"组合两个 llama_kv_cache"的全部秘密。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(4200, () => {
  msg.innerHTML = texts[1];
  wrap.querySelector('#fb').style.width = '100%';
});
tl.at(8600, () => {
  msg.innerHTML = texts[2];
  wrap.querySelector('#fs').style.width = '18%';
});
tl.at(15000, () => { msg.innerHTML = texts[3]; });
tl.at(19500, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-04 · 变体',
    title='一张表：六种变体各自解决什么问题',
    sub='每一行的依据都是源文件里的注释原话，逐字收在 source.md 里。',
    caption='共 10 个类：9 个直接继承 llama_memory_i，llama_memory_hybrid_idx 继承 hybrid。',
    src='src/llama-kv-cache-msa.h', parts=[(9, 12)], duration=24000,
    mark_src=[9, 10, 12],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['变体', '解决什么问题', '依据'],
  [['llama_kv_cache', '基类：cell 记账 + K/V 张量', 'llama-kv-cache.h:20'],
   ['llama_kv_cache_iswa', '非 SWA 层全量 / SWA 层窗口，两套 cache', 'llama-kv-cache-iswa.h:11'],
   ['llama_kv_cache_dsa', '主 K cache + 稀疏注意力的 indexer key cache', 'llama-kv-cache-dsa.h:11'],
   ['llama_kv_cache_dsa_iswa', 'DSA 层与 SWA 层再分两层', 'llama-kv-cache-dsa-iswa.h:11'],
   ['llama_kv_cache_msa', 'K/V cache + MSA indexer cache，cell 布局同步', 'llama-kv-cache-msa.h:9'],
   ['llama_kv_cache_dsv4', '原始 token cache + 压缩过的 K 块 cache', 'llama-kv-cache-dsv4.h:82'],
   ['llama_memory_recurrent', 'SSM/RNN：每层固定状态，没有 per-token K/V', 'llama-memory-recurrent.h:88'],
   ['llama_memory_hybrid', 'attention 层 + recurrent 层，各一套子 memory', 'llama-memory-hybrid.h:16'],
   ['llama_memory_hybrid_idx', 'hybrid 再加第三套 indexer cache', 'llama-memory-hybrid-idx.h:12'],
   ['llama_memory_hybrid_iswa', 'hybrid 的 attention 侧换成 iswa', 'llama-memory-hybrid-iswa.h:16']],
  { monoCols: [0, 2] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '10 行 = 10 个类。全部实现同一个接口。',
  'KV 一族的共同点：都建立在 <span class="k">llama_kv_cache</span> 上，靠组合而非继承来加能力。',
  'iswa / dsa_iswa / hybrid_iswa 解决的都是"<span class="k">不同层要用不同大小的 cache</span>"。',
  'dsa / msa / hybrid_idx 解决的是"<span class="k">除了 K/V，还要给稀疏注意力存一份 indexer</span>"。',
  'recurrent / hybrid 解决的是"<span class="k">有些层根本没有 K/V</span>"—— 见第 9、10 幕。',
  '对照表的用法：先问"这层的记忆形状是什么"，答案自然指向某一个类。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2100, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 4)];
}));
tl.at(23000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-04 · 变体',
    title='<span class="hl-e">recurrent</span>：没有 KV cache，只有固定状态',
    sub='SSM / RNN 的记忆与序列长度无关：每层一份固定大小的状态，按序列复制几行。',
    caption='跨课：L2-11（模型家族 · SSM 与线性注意力）讲这些层本身；本课只讲它们的记忆怎么存。',
    src='src/llama-memory-recurrent.h', parts=[(69, 77), (87, 115)], duration=25000,
    mark_src=[69, 73, 88, 89, 92, 109, 112, 113],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:9px">
    <div class="card grow" style="border-left-color:var(--a)">
      <div class="ct" style="color:var(--a)">KV cache：每个 token 一行</div>
      <div class="cb">行数 = 上下文长度（mem_size cells）<br>每行 = n_embd_k / n_embd_v 个数</div>
    </div>
    <div class="card grow" style="border-left-color:var(--e)">
      <div class="ct" style="color:var(--e)">recurrent：每个序列一份状态</div>
      <div class="cb">每层 r_l / s_l 是二维张量<br><span class="cm" style="margin:0">[n_embd_r, mem_size*(1 + n_rs_seq)]</span></div>
    </div>
  </div>
  <div id="rows"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const rows = wrap.querySelector('#rows');
rows.innerHTML = '<div class="cm" style="margin:0 0 3px">mem_cell 的字段（recurrent.h:88-107）</div>' +
  '<div class="row wrap" style="gap:6px">' +
  ['pos', 'src', 'src0', 'tail', 'seq_id'].map(f =>
    '<span class="chip ' + (f === 'seq_id' ? 'e' : 'c') + '">' + f + '</span>').join('') +
  '</div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '第一眼就能看出区别：recurrent 的 cell 里<span class="k">没有 K、没有 V</span>，只有"这份状态从哪来、发给谁"。',
  '<span class="v">pos</span> 是这个状态对应的位置；<span class="v">seq_id</span> 是它属于哪些序列。',
  '<span class="v">src / src0 / tail</span> 描述状态在序列之间怎么复制：' +
  '<br>src 说明从哪一行拷来（src0 只在设置输入时用，允许只拷一次），tail 指向该序列最新的一份状态。',
  '<span class="v">n_rs_seq</span>：每个序列保留几份历史快照，用于回滚。' +
  '<br>所以张量的行数是 <span class="v">mem_size * (1 + n_rs_seq)</span>（llama-memory-recurrent.cpp:101）。',
  '实际张量：每层两个（有 PLE 卷积历史时三个）—— <span class="v">r_l</span> / <span class="v">s_l</span> / <span class="v">p_l</span>。',
  '一句话：<span class="k">记忆大小与序列长度无关</span>，所以这类模型推长文本时显存不涨。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(4200, () => { msg.innerHTML = texts[1]; });
tl.at(8200, () => { msg.innerHTML = texts[2]; });
tl.at(12600, () => { msg.innerHTML = texts[3]; });
tl.at(17000, () => { msg.innerHTML = texts[4]; });
tl.at(21500, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L2-04 · 变体',
    title='<span class="hl-f">hybrid</span>：attention 层与 recurrent 层拼进一个 memory',
    sub='混合架构的模型一层一个样：一部分层是 attention，一部分是 recurrent，memory 就按层分工。',
    caption='默认 filter 就是 is_recr()：attn 侧收 !is_recr(il)，recurrent 侧收 is_recr(il)（llama-memory-hybrid.cpp:48-64）。',
    src='src/llama-memory-hybrid.h', parts=[(16, 19), (89, 90)], duration=22000,
    mark_src=[17, 19, 89, 90],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row" id="layers" style="gap:5px;justify-content:center"></div>
  <div class="row" style="gap:9px">
    <div class="card grow" style="border-left-color:var(--a)"><div class="ct" style="color:var(--a)">mem_attn : llama_kv_cache</div>
      <div class="cb">收 <span class="cm" style="margin:0">!is_recr(il)</span> 的层；cell 记账 + K/V 张量</div></div>
    <div class="card grow" style="border-left-color:var(--e)"><div class="ct" style="color:var(--e)">mem_recr : llama_memory_recurrent</div>
      <div class="cb">收 <span class="cm" style="margin:0">is_recr(il)</span> 的层；每层固定状态 r_l / s_l</div></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#layers');
const kinds = ['A', 'R', 'A', 'R', 'R', 'A', 'R', 'A'];
const lels = kinds.map((k, i) => {
  const e = U.el('div', { class: 'chip ' + (k === 'A' ? 'a' : 'e'), style: 'width:56px;text-align:center',
                          text: 'L' + i + ' ' + (k === 'A' ? 'attn' : 'recr') });
  host.appendChild(e); return e;
});

const msg = wrap.querySelector('#msg');
const texts = [
  '一个模型里两种层混排，memory 就分成两个子 memory，各自带一个 layer filter。',
  '<span class="k">filter 是构造参数</span>：默认按 <span class="v">hparams.is_recr(il)</span> 分，' +
  '<br>架构需要时可以在 create_memory 里换成更细的判据。',
  '分工之后，上层看到的是一个 memory：<span class="v">init_batch</span> / <span class="v">apply</span> / ' +
  '<span class="v">seq_*</span> 都同时作用到两个子 memory 上。',
  '两个更专的版本：<span class="v">llama_memory_hybrid_idx</span> 再加第三套 indexer cache（块稀疏注意力，' +
  '<span class="cm" style="margin:0">llama-memory-hybrid-idx.h:12</span>）；' +
  '<span class="v">llama_memory_hybrid_iswa</span> 把 attention 侧换成 iswa（<span class="cm" style="margin:0">llama-memory-hybrid-iswa.h:16</span>）。',
  '一句话：<span class="k">hybrid 不是新的记忆类型，而是"按层路由"</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
lels.forEach((e, i) => tl.at(3200 + i * 700, () => { e.style.opacity = '1'; }));
lels.forEach(e => e.style.opacity = '.35');
tl.at(9200, () => { msg.innerHTML = texts[1]; });
tl.at(13200, () => { msg.innerHTML = texts[2]; });
tl.at(17200, () => { msg.innerHTML = texts[3]; });
tl.at(20200, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 11 幕

L.scene(
    kicker='L2-04 · 收束',
    title='谁来选 memory：<span class="hl-a">create_memory()</span>',
    sub='一张 switch(arch) 决定用哪个实现；剩下的课都在讲被它选中的那些类。',
    caption='下一课 L2-05 讲 batch 与模型装配；L2-06 讲图怎么用这些 memory 建 attention。',
    src='src/llama-model.cpp', parts=[(2274, 2277), (2547, 2557), (2757, 2760)],
    duration=24000,
    mark_src=[2274, 2276, 2547, 2548, 2557, 2759],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', '答案', '在哪'],
  [['记忆的对外行为？', 'llama_memory_i 的 15 个虚函数', 'src/llama-memory.h'],
   ['一次 decode 怎么提交？', 'init_batch 规划 -> context.apply() 落笔', '第 3 幕'],
   ['一个 token 放哪？', 'slot_info：token[i] -> cells[idxs[i]]', '第 5 幕'],
   ['K/V 存在哪？', 'cache 自己的 buffer；图只拿 view / set_rows', '第 6 幕'],
   ['为什么有这么多实现？', '层不同、记忆形状不同；选择在 create_memory()', '第 8 / 11 幕']],
  { monoCols: [2] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '为什么不能只写一个 <span class="mono">llama_kv_cache</span> 类，把 SWA、稀疏注意力、SSM 都塞进去？',
  '三个层次的原因，都能在本课源码里指出来：<br>' +
  '1. <b>记忆的形状不同</b>。<span class="mono">llama_memory_recurrent::mem_cell</span> 里只有 ' +
  '<span class="mono">pos/src/src0/tail/seq_id</span>，没有 K/V；它的数据是每层的 ' +
  '<span class="mono">r_l/s_l</span>，行数是 <span class="mono">mem_size*(1+n_rs_seq)</span>，' +
  '与序列长度无关。硬塞进 KV cache 会让 cell 结构里出现大量永远为空的字段。<br>' +
  '2. <b>层与层之间也不同</b>。iswa 把层按 <span class="mono">is_swa(il)</span> 分成两套 cache，' +
  'hybrid 按 <span class="mono">is_recr(il)</span> 分成两个子 memory；dsa/msa/hybrid_idx 还要为稀疏注意力' +
  '多存一份 indexer。这些差异是<b>按层</b>的，不是整个模型一个开关。<br>' +
  '3. <b>但上层需要统一口径</b>。建图、序列操作、state 存盘恢复都只认 ' +
  '<span class="mono">llama_memory_i</span>；组合型 memory 用 ' +
  '<span class="mono">llama_memory_status_combine()</span> 把子 memory 的状态合并上报，' +
  '所以 10 个类对外长得一样。<br>' +
  '结论：抽象的是<b>行为</b>，不是数据布局。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '把这一课压成五问五答：',
  '<span class="k">对外行为</span>统一，<span class="k">数据布局</span>各异 —— 这就是整个设计。',
  'K/V 张量跨 decode 存活、自己管 buffer，所以它的生命周期与图中间张量完全不同。',
  '选择点只有一个：<span class="v">llama_model::create_memory()</span>，按 arch 分派。',
  '下一步：L2-05 看 batch 怎么被切成 ubatch；L2-06 看图怎么用这些 memory 建 attention。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
}));
tl.at(22000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、为什么需要 memory 抽象',
    '接口定义的第一段注释就回答了这个问题：**KV cache 只是一种 LLM memory**。'
    '同一个结构体上还挂着三类回调（层过滤、层复用、层共享）——'
    '差异从接口这一层就开始被表达，而不是等到某个实现内部再 `if` 出来。',
    src=SRC, parts=[(71, 83)], lang='c')

L.section(
    '二、★ llama_memory_i 的全部虚函数',
    '15 个纯虚函数，按职责分三组：\n\n'
    '| 组 | 函数 | 作用 |\n|---|---|---|\n'
    '| 生命周期 | `init_batch` `init_full` `init_update` | 规划一批 token 的落点，返回 context |\n'
    '| 序列操作 | `clear` `seq_rm` `seq_cp` `seq_keep` `seq_add` `seq_div` `seq_pos_min` `seq_pos_max` `get_can_shift` | 按 seq_id + 位置区间增删改查 |\n'
    '| 度量与状态 | `memory_breakdown` `state_write` `state_read` | 显存占用、状态存盘与恢复 |\n\n'
    '注意接口里**没有** `get_k` / `cpy_k`：数据面（K/V 张量、状态张量）不在这个接口上，'
    '只有 KV cache 一族的 context 才额外提供它们。',
    src=SRC, parts=[(85, 127)], lang='c')

L.section(
    '三、★ context：apply() 是唯一改状态的入口',
    '`init_*` 只做规划并返回一个 `llama_memory_context_i`，真正的写入发生在 `apply()`。'
    '源码注释把这一点写成硬约束：\n\n'
    '```text\n'
    'the only method that should mutate the memory and the memory context is apply()\n'
    '```\n\n'
    '组合型 memory（iswa / dsa / hybrid）对外要表现得像一个 memory，靠的是把子 context 的 '
    '`llama_memory_status` 合并：任一失败即失败，任一有更新即有更新。',
    src=SRC, parts=[(44, 69)], lang='c')

L.section(
    '四、status 合并：组合型 memory 的粘合剂',
    '`llama_memory_status_combine()` 是整个组合机制的基础设施：'
    '任一侧失败就直接返回那个失败，否则任一侧有更新就算有更新。'
    '接口声明旁的注释点名了它的用途 —— hybrid memory 类型（例如 iSWA）。',
    src='src/llama-memory.cpp', parts=[(3, 4), (40, 41)], lang='c')

L.section(
    '五、★ cell：元数据在 llama_kv_cells，K/V 在别处',
    '`llama_kv_cells` 的成员全是元数据：`pos`（位置，`-1` 表示空）、`ext`（M-RoPE 的 x/y 与 token id）、'
    '`shift`（攒起来的位移，最后一次同步）、`seq`（位集合，一个 cell 可属于多个序列）、'
    '`used`（非空下标集合）、`seq_pos`（`(pos, cell)` 有序集合）。\n\n'
    'K/V 不在这个类里 —— 它们在 `llama_kv_cache::layers[]` 的 ggml 张量中（`src/llama-kv-cache.h:250-260`）。'
    '**cell 是账本，张量是仓库。**',
    src='src/llama-kv-cells.h', parts=[(485, 524)], lang='c')

L.section(
    '六、slot_info 与 apply_ubatch',
    '`slot_info` 把 "第 i 个 token 写进哪个 cell" 变成一张表。'
    '落笔时先清掉被覆盖的旧 cell（`cells.rm`），再 `cells.pos_set()` 写入新位置 —— '
    '这就是 SWA 的环形复用能工作的原因。',
    src='src/llama-kv-cache.h', parts=[(32, 43)], lang='c')

L.section(
    '七、同一张表的消费端：apply_ubatch',
    '第 6 节那张表在 `apply_ubatch()` 里被消费：`sinfo.idxs[s][ii]` 直接当 cell 下标用。',
    src='src/llama-kv-cache.cpp', parts=[(1120, 1131)], lang='c')

L.section(
    '八、K/V 的归属与读写',
    'K/V 张量在 `llama_kv_cache` 构造时**按后端 buffer type 整块分配**，'
    '注释写明 `no_alloc` 模式下挂 dummy buffer 是为了让 backend scheduler 不去分配它。'
    '读走 `ggml_view_4d`（零拷贝视图），写走 `ggml_set_rows`（图上的一个节点，行号来自 slot_info）。',
    src='src/llama-kv-cache.cpp', parts=[(276, 295), (1266, 1283), (1346, 1351)], lang='c')

L.section(
    '九、iswa：用两个 llama_kv_cache 表示"两种层"',
    '类注释一句话说清结构：**两个 llama_kv_cache 实例**，第一个给非 SWA 层，第二个给 SWA 层。'
    '实现方式是链式 filter：`filter_base` 保留 `!is_swa(il)`，`filter_swa` 保留 `is_swa(il)`；'
    'SWA cache 的大小按窗口算（并 pad 到 256）。',
    src='src/llama-kv-cache-iswa.cpp', parts=[(52, 73), (95, 105)], lang='c')

L.section(
    '十、iswa 的头文件：为什么是"两个实例"',
    '类注释把这个类的全部设计压成一句话：**两个 llama_kv_cache 实例** —— '
    '第一个给非 SWA 层，第二个给 SWA 层。',
    src='src/llama-kv-cache-iswa.h', parts=[(11, 12)], lang='c')

L.section(
    '十一、KV cache 一族的其余变体（注释原话）',
    '**dsa**：两个 llama_kv_cache —— 一个存模型的 key，一个存 lightning indexer 的 key。',
    src='src/llama-kv-cache-dsa.h', parts=[(11, 13)], lang='c')

L.section(
    '十二、dsa 的 indexer cache 是怎么造出来的',
    '第二个 cache 直接复用 `llama_kv_cache`：注释原话是 hand-tweaking some hparams —— '
    '把每层的 KV 头数全部改成 1、头维度换成 `indexer_head_size`，'
    '同一个类就长出了 indexer key 的形状。',
    src='src/llama-kv-cache-dsa.cpp', parts=[(38, 53)], lang='c')

L.section(
    '十三、dsa_iswa',
    '两个子 memory：DSA 那一套给全注意力层，`llama_kv_cache` 那一套给 SWA 层。',
    src='src/llama-kv-cache-dsa-iswa.h', parts=[(11, 11)], lang='c')

L.section(
    '十四、dsa_iswa 的链式 filter',
    '和 iswa 同一套手法：先把上游 filter 串起来，再按 `is_swa(il)` 一分为二；'
    'DSA 那一半拿满 `kv_size`，SWA 那一半按窗口算。',
    src='src/llama-kv-cache-dsa-iswa.cpp', parts=[(38, 49)], lang='c')

L.section(
    '十五、msa',
    '两个 llama_kv_cache：一个存 K/V，一个存 MSA indexer。'
    '两者接收**完全相同的序列操作与 ubatch**，所以 cell 布局保持同步；'
    'context 还会把 `llama_kv_cells` 里的 pos -> cell 映射暴露给图，做 position 空间的块选择。',
    src='src/llama-kv-cache-msa.h', parts=[(9, 12)], lang='c')

L.section(
    '十六、msa 的 indexer cache 形状',
    'MSA 的 indexer 也是"每层一个 key 头"：把 `n_head_kv_arr` 全填 1，'
    '头维度换成 `indexer_head_size`；注释特别说明 **rope 参数与主 cache 保持一致**。',
    src='src/llama-kv-cache-msa.cpp', parts=[(39, 42)], lang='c')

L.section(
    '十七、dsv4',
    '一个普通的 raw/SWA token cache **加上压缩过的 K-only 块 cache**；压缩 cache 只做存储，'
    '可见性与块规划交给 context 与图输入去处理。压缩比是文件顶部的两个常量：'
    '`DSV4_CSA_RATIO = 4`、`DSV4_HCA_RATIO = 128`。',
    src='src/llama-kv-cache-dsv4.h', parts=[(82, 84)], lang='c')

L.section(
    '十八、dsv4 的两个压缩比常量',
    '压缩 cache 的粒度由文件顶部两个常量决定：`DSV4_CSA_RATIO = 4` 与 `DSV4_HCA_RATIO = 128`。'
    '压缩块的行数由 `dsv4_comp_size(kv_size, ratio)` 按 `kv_size / ratio` 算出来。',
    src='src/llama-kv-cache-dsv4.cpp', parts=[(18, 19)], lang='c')

L.section(
    '十九、recurrent：没有 KV，只有状态',
    '`mem_cell` 的字段是 `pos` / `src` / `src0` / `tail` / `seq_id` —— 没有任何 K/V。'
    '数据面是每层的 `r_l` / `s_l`（有 PLE 卷积历史时再加 `p_l`）；'
    '`n_rs_seq` 是每个序列保留的回滚快照数，所以张量的行数是 `mem_size * (1 + n_rs_seq)`。',
    src='src/llama-memory-recurrent.h', parts=[(87, 115)], lang='c')

L.section(
    '二十、recurrent 的张量形状（cpp 侧）',
    '`r` / `s` 张量的第一个维度是 `hparams.n_embd_r()` / `n_embd_s()`，第二个维度是行数 —— '
    '**行数只跟 mem_size 与快照数有关，与序列长度无关**。',
    src='src/llama-memory-recurrent.cpp', parts=[(101, 105)], lang='c')

L.section(
    '二十一、hybrid：attention 层与 recurrent 层各一套子 memory',
    '类注释：用 `llama_memory_recurrent` 与 `llama_kv_cache` 的组合，'
    '支持"每一层要么是 attention、要么是 recurrent"的模型。'
    '默认 filter 就是按 `hparams.is_recr(il)` 分工。',
    src='src/llama-memory-hybrid.cpp', parts=[(48, 64)], lang='c')

L.section(
    '二十二、hybrid 的两个更专版本',
    '`hybrid_idx`：在 hybrid 之上再加第三套 cache —— 每个 token 一个 indexer key，用于块稀疏注意力。'
    'indexer 是 attention cell 的**旁路缓冲**：大小、padding、stream、slot 全都一样，'
    '所以 cell j 在两边指的是同一个 token。',
    src='src/llama-memory-hybrid-idx.h', parts=[(12, 13)], lang='c')

L.section(
    '二十三、hybrid_idx 的第三套 cache',
    '`mem_idx` 只在给了 `filter_idx` 时才建（即 GGUF 里真的有 indexer 张量时）。'
    '构造方式同样是改 hparams 造形状，并且显式把 rope 设成 NONE、'
    '把 K/V 的 MLA 维度都设成 indexer 头维度 —— 注释说这是"骗 llama_kv_cache 不要缓存 V"。',
    src='src/llama-memory-hybrid-idx.cpp', parts=[(49, 60)], lang='c')

L.section(
    '二十四、hybrid_iswa',
    '把 hybrid 的 attention 侧换成 `llama_kv_cache_iswa`：'
    '支持"每层要么是 attention（带 SWA）、要么是 recurrent"的模型。',
    src='src/llama-memory-hybrid-iswa.h', parts=[(16, 17)], lang='c')

L.section(
    '二十五、hybrid_iswa 的 attention 侧',
    '`mem_attn` 的类型直接写成 `llama_kv_cache_iswa` —— '
    '"hybrid"与"iswa"两个维度的正交组合，在这里只是一行构造参数。',
    src='src/llama-memory-hybrid-iswa.cpp', parts=[(33, 41)], lang='c')

L.section(
    '二十六、分派点：llama_model::create_memory()',
    '一个 `switch (arch)` 决定用哪个 memory 实现。钩子是两个判定函数：'
    '`llm_arch_is_recurrent(arch)` 与 `llm_arch_is_hybrid(arch)`；'
    'hybrid 分支里再按架构挑 layer filter，最后返回 `llama_memory_i *`。',
    src='src/llama-model.cpp', parts=[(2274, 2277), (2547, 2557)], lang='c')

L.footnote_add('本课声明 24 个源文件：计划中 L2-04 的 23 个全部声明，另加 `src/llama-model.cpp`'
               '（memory 的分派点，`llama_model::create_memory()` 在 2274 行）。'
               '`src/llama-model.cpp` 在计划里归 L2-05，本课引用后一并计入覆盖，特此说明。')
L.footnote_add('本课不引用 `src/llama-arch.cpp`，因此 `llm_arch_is_recurrent()` / `llm_arch_is_hybrid()` '
               '里的具体架构清单不在本课展开，留给 L2-11（模型家族 · SSM 与线性注意力）。')
L.footnote_add('本课不讨论任何命令行参数，参数门禁的真值集来自真实二进制的帮助输出。')

L.prereqs('`L2-03`')

L.goal(
    '列出 `llama_memory_i` 的虚函数，并把它们按"生命周期 / 序列操作 / 状态 IO"分成三组（对应验收点）；',
    '解释为什么 `init_batch()` 返回 context 而 `apply()` 才改状态，以及这个设计对"装不下"的意义；',
    '说出 `llama_kv_cells` 里存的是哪些元数据，并指出 K/V 张量其实在哪里；',
    '对照 iswa / dsa / msa / dsv4 / recurrent / hybrid 六种变体，各说出它解决的一个具体问题；',
    '解释为什么 SSM/RNN 模型不需要 KV cache，而需要一份固定大小的状态。')

L.conclusion(
    '抽象的是行为，不是数据布局',
    '`llama_memory_i` 只描述"记忆对外的行为"：15 个虚函数，没有一个提到 K 或 V。'
    '数据面留在各自实现里 —— 所以同一个接口下既可以有按 token 记账的 KV cache，'
    '也可以有每层一份固定状态的 recurrent memory。')

L.conclusion(
    '★ 规划与提交分离',
    '`init_batch()` / `init_full()` / `init_update()` 只规划并返回 context；'
    '源码把"唯一能改 memory 的方法是 `apply()`"写成了硬约束。'
    '组合型 memory 靠 `llama_memory_status_combine()` 合并子 memory 的状态，'
    '对外表现得像一个 memory。')

L.conclusion(
    '★ cell 是账本，张量是仓库',
    '`llama_kv_cells` 存 `pos` / `ext` / `shift` / `seq` / `used` / `seq_pos`；'
    'K/V 在 `llama_kv_cache::layers[]` 的 ggml 张量里，'
    '由 cache 自己按后端 buffer type 整块分配（调度器不参与）。'
    '图侧只拿视图（`ggml_view_4d`）与写行（`ggml_set_rows`）。')

L.conclusion(
    '一句话记住这一族',
    '```text\n'
    'KV cache   = 每个 token 一份记忆   -> llama_kv_cache 及其变体\n'
    'recurrent  = 每个序列一份固定状态  -> llama_memory_recurrent\n'
    'hybrid     = 按层路由到上面两者    -> llama_memory_hybrid\n'
    '```\n\n'
    '选择发生在 `llama_model::create_memory()`：一个 `switch (arch)`，'
    '加上 `llm_arch_is_recurrent()` / `llm_arch_is_hybrid()` 两个判定。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
