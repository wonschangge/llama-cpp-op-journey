#!/usr/bin/env python3
"""L2-07 · 上下文与解码 llama-context —— 课件 spec。

运行：python3 L2-model-to-graph/L2-07-context-and-decode/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

CPP = 'src/llama-context.cpp'
HDR = 'src/llama-context.h'

L = Lesson(
    id='L2-07',
    layer='L2 · 从模型到图',
    title='上下文与解码 llama-context',
    codecap='src/llama-context.{h,cpp}（逐字引用）',
    nav={'prev': {'href': '../L2-06-graph-skeleton/index.html', 'label': 'L2-06 ★ 计算图骨架'},
         'next': {'href': '../L2-08-sampler-and-vocab/index.html', 'label': 'L2-08 采样器与词表'}},
)

# ---------------------------------------------------------------- 覆盖声明
L.cover(CPP, HDR)

L.note('**一句话**：`llama_decode` 是**图的生命周期管理器** —— 建图（`model.build_graph`）、'
       '分配（`ggml_backend_sched_alloc_graph`）、执行（`graph_compute` → '
       '`ggml_backend_sched_graph_compute_async`）、复用（`res->can_reuse`）'
       '这四件事都在 `llama_context::decode` 一个函数里闭环。')
L.note('回顾 L2-06：那一课讲 `llm_graph_context` 与 `build_*` 原语 —— 也就是"图长什么样"。'
       '这一课讲**谁在什么时候调用它**：一个 `llama_batch` 进来，怎么变成若干张被调度器执行的图，'
       '以及为什么第二个 token 不用重新建图。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2-07 · 全局',
    title='★ <span class="hl-a">llama_decode</span>：图的生命周期管理器',
    sub='建图、分配、执行、复用 —— 四件事都在 decode 里闭环。先看 llama_context 类自己的注释。',
    caption='本课覆盖 src/llama-context.cpp 与 src/llama-context.h 两个文件，二者都计入覆盖率。',
    src=HDR, parts=[(43, 57)], duration=17000,
    mark_src=[43, 44, 57],
    notes_src={
        44: '构造函数注释就把职责写清楚了：init scheduler and compute buffers, reserve worst-case graphs',
        57: 'sched_reserve() 是"按需重建调度器 + 重新预留"，不是每 token 都做',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">llama_decode</span><span class="arrow">-></span>
    <span class="chip b">decode</span><span class="arrow">-></span>
    <span class="chip c">process_ubatch</span><span class="arrow">-></span>
    <span class="chip d">graph_compute</span><span class="arrow">-></span>
    <span class="chip f">sched_graph_compute_async</span>
  </div>
  <div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '建图', b: 'model.build_graph(gparams)<br>按 ubatch 的拓扑建一张图', m: 'llama-context.cpp:1425', w: '163px' },
  { c: 'b', t: '分配', b: 'ggml_backend_sched_alloc_graph<br>把节点放进各后端的计算缓冲', m: 'llama-context.cpp:1435', w: '163px' },
  { c: 'c', t: '执行', b: 'graph_compute 取线程池，<br>再交给调度器异步提交', m: 'llama-context.cpp:1454 / 2588', w: '163px' },
  { c: 'd', t: '复用', b: '能复用时跳过建图与分配，<br>只重新填输入', m: 'llama-context.cpp:1405', w: '163px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '一个 <span class="k">llama_batch</span> 进来，出去的是"若干张已经在后端上算完的图"。',
  '<span class="k">建图</span>：<span class="v">model.build_graph(gparams)</span> —— 图长什么样属于 L2-06。',
  '<span class="k">分配</span>：<span class="v">ggml_backend_sched_alloc_graph</span> —— 内存规划属于 L4-01 / L4-02。',
  '<span class="k">执行</span>：<span class="v">graph_compute</span> 里那句 <span class="v">_async</span> 才是真正把图交出去的地方（L4-04）。',
  '记住这四个词：<span class="v">建图 / 分配 / 执行 / 复用</span>。后面每一幕对应其中一段。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-07 · 入口',
    title='C API <span class="hl-b">llama_decode</span> 只是转发',
    sub='真正的逻辑在 llama_context::decode（1704 行）。C API 只负责打日志与把返回值传出去。',
    caption='注意 4330 行的判据：返回 1 不算错误 —— 它是"这个 batch 暂时没位置"的软失败。',
    src=CPP, parts=[(4326, 4335)], duration=14000,
    mark_src=[4329],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', t: '0', b: '成功。<br>图已提交（异步时可能仍在跑）', m: 'llama-context.cpp:2096', w: '163px' },
  { c: 'c', t: '1', b: 'memory 找不到 slot。<br>软失败，llama_decode 不为它报错', m: 'llama-context.cpp:1839', w: '163px' },
  { c: 'g', t: '-1', b: '输入非法：n_tokens 为 0、<br>batch 初始化失败、pooled 要求全输出', m: 'llama-context.cpp:1716 / 1767', w: '163px' },
  { c: 'd', t: '-2', b: '分配失败：memory context、<br>输出缓冲或图分配失败', m: 'llama-context.cpp:1855 / 1914', w: '163px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '返回值不是随意取的 —— 它在 <span class="k">1704 起的 decode()</span> 里逐条 <span class="v">return</span> 出来。',
  '<span class="v">0</span> 表示这一批成功。<span class="v">1</span> 表示"这次挤不下"，调用方可以稍后重试。',
  '<span class="v">-1</span> 是<b>调用方写错了</b>：空 batch、batch 结构不对、embedding 模式下没有全输出。',
  '<span class="v">-2</span> 是<b>资源不够</b>：拿不到 memory context、预留不出输出缓冲、图分配失败。',
  '还有两个来自 ubatch 执行失败的分支：<span class="v">2</span> = GGML_STATUS_ABORTED，<span class="v">-3</span> = GGML_STATUS_FAILED（1913-1915）。'
];
defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(12300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-07 · 第 1 步',
    title='<span class="hl-c">balloc</span>：batch 进，ubatch 的准备开始',
    sub='decode 的第一件事是把用户给的 llama_batch 校验并归一化；切分本身由 memory 模块接手。',
    caption='回顾 L2-05：ubatch 是切图的依据 —— 同一个 ubatch 拓扑不变，所以图才能复用。',
    src=CPP, parts=[(1765, 1784)], duration=18000,
    mark_src=[1765, 1770, 1771, 1782, 1784],
    notes_src={
        1765: 'balloc 是 context 的成员（llama-context.h:334），复用同一个对象避免反复分配',
        1770: 'n_tokens_all：本逻辑批的总 token 数',
        1771: 'n_outputs_all：需要输出 logits/embd 的行数，后面 output_reserve 用它',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'llama_batch_allocr', b: '把 llama_batch 归一化成<br>统一内部表示，并切出 ubatch', m: 'llama-context.h:334', w: '220px' },
  { c: 'b', t: 'n_tokens_all', b: '这一批一共多少 token。<br>下面用两条断言守住上限', m: 'llama-context.cpp:1770', w: '220px' },
  { c: 'c', t: 'n_outputs_all', b: '要输出多少行。<br>output_reserve 按它预备缓冲', m: 'llama-context.cpp:1771', w: '220px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '先看清楚：<span class="k">ubatch 不是这里切出来的</span>，这里只做校验与归一化。',
  '<span class="v">balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)</span><br>' +
  '失败就直接 <span class="v">return -1</span> —— 这是 decode 的第一道闸门。',
  '<span class="v">n_tokens_all</span> 与 <span class="v">n_outputs_all</span> 是后面所有循环的分母：<br>输出缓冲大小、进度累计都按它们算。',
  '两条 <span class="k">GGML_ASSERT</span>（1782、1784）是硬约束：<br>一批不能超过 <span class="v">n_batch</span>；非因果注意力必须一次放得下 <span class="v">n_ubatch</span>。',
  '真正的切分在下一幕：<span class="v">memory->init_batch(*balloc, cparams.n_ubatch, output_all)</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13400, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-07 · 第 2 步',
    title='★ <span class="hl-a">sched_reserve</span>：最坏情况的图，只预留一次',
    sub='它有一个开关 sched_need_reserve：第一次进来做完就关掉，之后每次 decode 直接返回。',
    caption='这正是"每 token 开销"能被压下来的原因之一：max_nodes 与计算缓冲只规划一次。',
    src=CPP, parts=[(617, 635)], duration=20000,
    mark_src=[618, 622, 631, 633],
    notes_src={
        618: '开关：不需要预留就直接 return —— 这是常态路径',
        622: '置 false：做完这一次就关掉，除非有东西把它重新点亮',
        631: '按最坏情况取：n_tokens = min(n_ctx, n_ubatch)',
        633: 'max_nodes 决定调度器与图元数据 arena 的上限',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '开关', b: 'sched_need_reserve<br>false 就整段跳过', m: 'llama-context.h:350', w: '163px' },
  { c: 'b', t: '最坏规模', b: 'n_tokens = min(n_ctx, n_ubatch)<br>n_seqs = n_seq_max', m: 'llama-context.cpp:630 - 631', w: '163px' },
  { c: 'c', t: '节点上限', b: 'max_nodes =<br>graph_max_nodes(n_tokens)', m: 'llama-context.cpp:633', w: '163px' },
  { c: 'd', t: '重建调度器', b: 'ggml_backend_sched_new<br>按 max_nodes 建', m: 'llama-context.cpp:643', w: '163px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">sched_reserve()</span> 在 decode 的第 1800 行被调用，但它<b>大多数时候什么都不做</b>。',
  '看 <span class="k">618</span> 行：<span class="v">if (!sched_need_reserve) return;</span><br>' +
  '也就是说这个函数是"幂等的一次性预留"，不是每 token 的固定开销。',
  '需要预留时按<b>最坏情况</b>来：<span class="v">n_tokens = min(n_ctx, n_ubatch)</span>，<br>序列数取 <span class="v">n_seq_max</span>，节点上限交给 <span class="v">graph_max_nodes()</span>。',
  '然后按 max_nodes 重建调度器与 gf_res_reserve —— <span class="k">调度器的容量在这里被钉死</span>（细节见 L4-02）。',
  '谁会把它重新点亮？<span class="v">set_sampler</span>（1289）、<span class="v">set_adapters_lora</span>（1347）、' +
  '<span class="v">set_causal_attn</span>（1259）、<span class="v">set_adapter_cvec</span>（1386）等；<br>' +
  '换句话说：<span class="k">图形态可能要变的时候</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-07 · 第 3 步',
    title='<span class="hl-d">memory->init_batch</span>：ubatch 的产地',
    sub='切分这件事不在 context 里做，而是交给 memory 模块；context 只拿到一个 mctx 再问它要 ubatch。',
    caption='回顾 L2-04：memory 家族（KV cache / recurrent / hybrid）各自决定怎么切。',
    src=CPP, parts=[(1800, 1813)], duration=20000,
    mark_src=[1800, 1805, 1809, 1810],
    notes_src={
        1805: '处理悬空的 shift/copy —— KV cache 的搬运在这里落地',
        1810: 'n_ubatch 是上限：memory 可以切得更小',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">llama_batch_allocr</span><span class="arrow">+</span>
    <span class="chip b">llama_memory_i</span><span class="arrow">-></span>
    <span class="chip c">memory->init_batch</span><span class="arrow">-></span>
    <span class="chip d">llama_memory_context_i</span><span class="arrow">-></span>
    <span class="chip f">mctx->get_ubatch()</span>
  </div>
  <div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', t: 'memory_update(false)', b: '先把待处理的 shift/copy 做掉，<br>并让 memory 尝试优化一次', m: 'llama-context.cpp:1805', w: '220px' },
  { c: 'c', t: 'while (true) 重试', b: 'FAILED_PREPARE 时用<br>memory_update(true) 优化后重来一次', m: 'llama-context.cpp:1809 / 1830', w: '220px' },
  { c: 'd', t: 'mctx 给的是 ubatch 序列', b: 'do { get_ubatch() } while (mctx->next())<br>一个逻辑批可能拆成多个 ubatch', m: 'llama-context.cpp:1867 / 2041', w: '220px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  'decode 不自己切 batch：它把 <span class="k">balloc</span>、<span class="k">memory</span>、<span class="k">n_ubatch</span> 一起交给 memory 模块。',
  '第 <span class="v">1805</span> 行的 <span class="v">memory_update(false)</span> 先处理悬空搬运 —— 这是 L2-04 里 KV cache 的 shift/copy。',
  '拿到 <span class="v">mctx</span> 之后，真正的循环是 <span class="k">do / while (mctx->next())</span>：<br>一个逻辑批可能被拆成好几个 ubatch，每个走一次 process_ubatch。',
  '如果 memory 报 <span class="v">FAILED_PREPARE</span>，decode 会先尝试 <span class="v">memory_update(true)</span> 优化一次，<br>再重试；还不行就 <span class="v">return 1</span>（软失败）。',
  '所以"切图依据"这条链是：<span class="v">n_ubatch</span>（上限）→ memory 的策略 → <span class="v">mctx</span> → 每个 ubatch 一张图。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-07 · 第 4 步',
    title='★ <span class="hl-b">process_ubatch</span>：复用还是重建，一个 if 决定',
    sub='注释写得很直白：想复用它，图的完整拓扑必须由这些参数唯一确定。',
    caption='can_reuse 的实现属于 L2-06 的 llm_graph_result（src/llama-graph.{h,cpp}），本课只看调用点与判据。',
    src=CPP, parts=[(1398, 1420)], duration=22000,
    mark_src=[1403, 1405, 1412, 1415, 1418, 1420],
    notes_src={
        1403: 'gparams 里装着 ubatch、mctx、gtype、sched —— 复用判据就是比较它',
        1405: '三个条件同时成立才复用：开关没关、还是同一个 arena、拓扑参数一致',
        1415: '复用命中时唯一要做的事：记一笔账',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', t: '复用路径（命中）', b: '不建图、不分配。<br>只 ggml_backend_sched_synchronize（仅流水线并行时）<br>然后 n_reused++', m: 'llama-context.cpp:1412 / 1415', w: '220px' },
  { c: 'g', t: '重建路径（未命中）', b: 'gf_res_prev_active = nullptr<br>res->reset()<br>ggml_backend_sched_reset(sched)<br>再建图 + 分配', m: 'llama-context.cpp:1417 - 1420', w: '220px' },
  { c: 'd', t: '两个复用槽', b: 'gf_res_prev[n_outputs > 0]<br>有输出 / 无输出各一个 arena，<br>避免两类图互相顶掉', m: 'llama-context.cpp:2423', w: '220px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">process_ubatch</span> 是整个 decode 里最值得记住的函数：它是<b>图的入口</b>。',
  '先拿 <span class="k">res = get_gf_res_prev()</span>（1398），再算 <span class="k">gparams = graph_params(...)</span>（1403）—— ' +
  '参数里打包了 ubatch、mctx、gtype、sched。',
  '然后就是那个 <span class="k">if</span>（1405）：<span class="v">!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)</span>。<br>' +
  '三个条件缺一不可 —— 而判据本身写在 <span class="v">llm_graph_params::allow_reuse</span> 里（L2-06 讲过）。',
  '命中复用时，这一轮 decode 的开销只剩：<span class="k">同步（仅流水线并行）+ 重填输入 + 提交计算</span>。',
  '没命中就走 else：重置 arena、重置调度器、重建图、重新分配（下一幕）。<br>可用环境变量 <span class="v">LLAMA_GRAPH_REUSE_DISABLE=1</span> 关掉复用（280-281 行）。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(15200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-07 · 第 5 步',
    title='重建路径：<span class="hl-c">建图</span> → <span class="hl-a">分配</span>',
    sub='只有这一条 else 分支里才会出现 build_graph 与 sched_alloc_graph —— 复用命中时它们根本不执行。',
    caption='回顾 L2-06：build_graph 返回的就是 llm_graph_context 拼出来的 ggml_cgraph。',
    src=CPP, parts=[(1421, 1442)], duration=20000,
    mark_src=[1421, 1425, 1435, 1441],
    notes_src={
        1421: '先清掉上一张图的分配状态，并重新挂上 eval 回调',
        1425: '图从这里来 —— 拓扑由 gparams 唯一确定',
        1435: '分配：调度器把每个节点落到某个后端的计算缓冲里（L4-01 / L4-02）',
        1441: '只有成功建图并分配之后，才把这个 arena 标成"可复用"',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '1. 清场', b: 'res->reset() +<br>ggml_backend_sched_reset(sched)', m: 'llama-context.cpp:1418 / 1420', w: '163px' },
  { c: 'b', t: '2. 建图', b: 'model.build_graph(gparams)<br>返回 ggml_cgraph *', m: 'llama-context.cpp:1425', w: '163px' },
  { c: 'c', t: '3. 分配', b: 'ggml_backend_sched_alloc_graph<br>失败即 GGML_STATUS_ALLOC_FAILED', m: 'llama-context.cpp:1435', w: '163px' },
  { c: 'd', t: '4. 记账', b: 'gf_res_prev_active = res<br>下次 can_reuse 才有机会命中', m: 'llama-context.cpp:1441', w: '163px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '这四步是"重建一张图"的全部动作，顺序不能换。',
  '<span class="v">ggml_backend_sched_reset()</span> 会清掉调度器里上一张图的切分与分配信息 —— ' +
  '所以 <span class="k">reset 之后旧图一定不能复用</span>（graph_reserve 里 2499 行的注释就是这么说的）。',
  '<span class="v">model.build_graph(gparams)</span>：这一步的产物是 <span class="k">ggml_cgraph *</span>，' +
  '内容完全由 gparams 决定 —— 这就是"拓扑可复用"的前提。',
  '<span class="v">ggml_backend_sched_alloc_graph()</span> 做的是内存规划：节点算出的张量住哪个 buffer、' +
  '哪些可以复用同一块（L4-01 讲分配器，L4-02 讲切分）。',
  '最后 <span class="v">gf_res_prev_active = res</span>：把当前 arena 标记为"这张图有效"，' +
  '下一次 process_ubatch 才有机会走进复用分支。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-07 · 第 6 步',
    title='喂输入：<span class="hl-e">res->set_inputs</span> → <span class="hl-d">graph_compute</span>',
    sub='两条路径（复用 / 重建）在这里汇合：图已经就绪，接下来只需要把这一轮的 ubatch 填进去。',
    caption='第二个参数 batched 决定用哪套线程数：ubatch.n_tokens > 1 就是 prompt 处理（L4-04）。',
    src=CPP, parts=[(1444, 1454)], duration=16000,
    mark_src=[1449, 1454],
    notes_src={
        1449: '复用路径下这一步是唯一真正"更新"图内容的动作',
        1454: '返回值直接就是 ggml_status，失败时 process_ubatch 返回 nullptr',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'e', t: 'res->set_inputs(&ubatch)', b: '把这一轮的 ubatch 填进<br>图的所有输入对象', m: 'llama-context.cpp:1449', w: '220px' },
  { c: 'd', t: 'graph_compute(gf, batched)', b: 'batched = ubatch.n_tokens > 1<br>决定用 n_threads_batch 还是 n_threads', m: 'llama-context.cpp:1454', w: '220px' },
  { c: 'a', t: '两条路径在此汇合', b: '复用与重建的区别到此为止，<br>后面完全一样', m: 'llama-context.cpp:1442 - 1454', w: '220px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '注意 <span class="k">if / else</span> 在 1442 行就结束了 —— 从 1444 行开始，两条路径走的是同一段代码。',
  '<span class="v">res->set_inputs(&ubatch)</span> 遍历图的所有输入对象，把这一轮的数据拷进去；<br>' +
  '上一幕的 <span class="v">can_reuse</span> 检查的就是"这些输入张量还能不能装下新数据"。',
  '<span class="v">graph_compute(res->get_gf(), ubatch.n_tokens > 1)</span><br>用 <span class="v">res->get_gf()</span> 而不是局部变量 <span class="v">gf</span> —— ' +
  '因为复用路径下局部 <span class="v">gf</span> 还是 <span class="v">nullptr</span>。',
  '<span class="v">batched</span> 这个布尔量一边选线程池，一边选线程数：<br>prefill 用 batch 线程，逐 token 用普通线程（细节见 L4-04）。',
  '一句话：<span class="k">图的身份在这一步之前定下来，图的内容在这一步被填进去</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(12300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-07 · 第 7 步',
    title='终点：<span class="hl-f">ggml_backend_sched_graph_compute_async</span>',
    sub='graph_compute 自己不做计算：它只选线程池/线程数，然后把图整张交给调度器。',
    caption='在 llama-context.cpp 里，与"提交整图计算"相关的 ggml_backend_sched_ 调用只有 2588 这一处（且是 _async）。',
    src=CPP, parts=[(2569, 2596)], duration=18000,
    mark_src=[2572, 2588, 2589],
    notes_src={
        2572: 'batched 在这里变成具体线程数：n_threads_batch 或 n_threads',
        2588: '异步提交：函数返回时计算可能还在跑',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '1. 先设 CPU 线程池', b: '把 threadpool / threadpool_batch<br>塞给 CPU 后端', m: 'llama-context.cpp:2575 - 2581', w: '163px' },
  { c: 'b', t: '2. 再通知所有后端', b: '遍历 set_n_threads_fns，<br>逐个后端设置线程数', m: 'llama-context.cpp:2584', w: '163px' },
  { c: 'c', t: '3. 提交整图', b: 'ggml_backend_sched_graph_compute_async<br>(sched, gf)', m: 'llama-context.cpp:2588', w: '163px' },
  { c: 'd', t: '4. 返回状态', b: 'ggml_status 一路回到 process_ubatch，<br>失败即整体失败', m: 'llama-context.cpp:2595', w: '163px' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '这就是验收点里那句"到 ggml_backend_sched_graph_compute 为止"的真实落点：<b>函数名带 _async</b>。',
  '<span class="v">batched</span> 在这里生效：<span class="v">n_threads</span> 与 <span class="v">tp</span> 两个局部变量都靠它三目选择。',
  '线程池只对 CPU 后端有意义，所以先做一次<b>进程地址查询</b>拿到 <span class="v">ggml_backend_cpu_set_threadpool</span>；<br>' +
  '拿不到就跳过 —— 这解释了为什么它是 <span class="v">if (set_threadpool_fn)</span>。',
  '其余后端（GPU 等）走 <span class="v">set_n_threads_fns</span>：context 构造时就把每个后端"设线数的函数"收集好了（366-368 行）。',
  '交给调度器之后的事 —— 切分成几个 split、插哪些 copy 节点、什么时候同步 —— 是 L4-02 与 L4-04。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L2-07 · 收束',
    title='一趟 decode 的结尾，与整条调用链',
    sub='循环收尾后把 n_outputs 归位、建立 output_ids 映射，用户才能用 llama_get_logits_ith 取数。',
    caption='下一课 L2-08 采样器与词表：logits 拿到手之后，怎么变成下一个 token。',
    src=CPP, parts=[(2039, 2056)], duration=24000,
    mark_src=[2041, 2044, 2056],
    notes_src={
        2041: 'mctx->next() 为假时退出：一个逻辑批的全部 ubatch 都算完了',
        2044: '把 n_outputs 从"本 ubatch 的输出数"改回"整批的输出数"',
    },
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['步', '在 llama-context.cpp 的哪一步', '做什么'],
  [['1', 'llama_context::decode · 1704', '入口：校验 + 主循环'],
   ['2', 'balloc->init · 1765', '归一化 batch，取 n_tokens_all'],
   ['3', 'sched_reserve · 1800', '按需预留最坏情况图（通常直接返回）'],
   ['4', 'memory->init_batch · 1810', '拿到 mctx，ubatch 的产地'],
   ['5', 'process_ubatch · 1887', '每个 ubatch 一次'],
   ['6', 'model.build_graph · 1425', '建图（复用命中时跳过）'],
   ['7', 'ggml_backend_sched_alloc_graph · 1435', '分配计算缓冲（同上）'],
   ['8', 'res->set_inputs · 1449', '把 ubatch 填进输入张量'],
   ['9', 'graph_compute · 1454 → 2569', '选线程池与线程数'],
   ['10', 'ggml_backend_sched_graph_compute_async · 2588', '提交整图计算']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '从 <span class="mono">process_ubatch</span> 收到一个 ubatch 开始，到调度器真正接手计算，' +
  '按顺序说出经过的每一步（函数名 + 大致行为）。',
  '1. <span class="mono">get_gf_res_prev()</span>（1398）取这个 arena 里上一次的图结果；<br>' +
  '2. <span class="mono">graph_params(...)</span>（1403）打包 ubatch / mctx / gtype / sched；<br>' +
  '3. 判断 <span class="mono">res->can_reuse(gparams)</span>（1405）：<br>' +
  '&nbsp;&nbsp;&nbsp;命中 → 只做 <span class="mono">n_reused++</span>（1415），跳到第 8 步；<br>' +
  '&nbsp;&nbsp;&nbsp;未命中 → <span class="mono">res->reset()</span>（1418）+ <span class="mono">ggml_backend_sched_reset()</span>（1420）；<br>' +
  '4. <span class="mono">model.build_graph(gparams)</span>（1425）建图；<br>' +
  '5. <span class="mono">ggml_backend_sched_alloc_graph(sched, gf)</span>（1435）分配；<br>' +
  '6. <span class="mono">gf_res_prev_active = res</span>（1441）标记可复用；<br>' +
  '7. <span class="mono">res->set_inputs(&ubatch)</span>（1449）填输入；<br>' +
  '8. <span class="mono">graph_compute(gf, batched)</span>（1454）→ 进入 2569 行的实现；<br>' +
  '9. <span class="mono">ggml_backend_sched_graph_compute_async(sched, gf)</span>（2588）—— 计算在这里才真正被提交。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '先把链走一遍：<span class="v">decode → balloc → sched_reserve → init_batch → process_ubatch → build_graph → alloc_graph → set_inputs → compute</span>。',
  '<span class="k">前四步每"逻辑批"一次</span>：校验、预留、切分（第 2-5 行）。',
  '<span class="k">后五步每"ubatch"一次</span>：建图、分配、填输入、提交计算（第 6-10 行）。',
  '<span class="v">建图与分配只在复用未命中时发生</span> —— 这就是逐 token 生成时每步开销的大头被省掉的地方。',
  '回到出口：<span class="v">n_outputs = n_outputs_all</span>（2044）与 <span class="v">output_ids[out_id] = i</span>（2056），' +
  '把"批里的第几个 token"翻译成"输出缓冲的第几行"。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 1900, () => {
  rows.forEach(x => { x.className = ''; });
  r.className = 'on';
  msg.innerHTML = texts[i < 2 ? 1 : (i < 9 ? 2 : 3)];
}));
tl.at(22600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、llama_context 的公共面：decode / encode / graph_compute / graph_reserve',
    '这几个声明就是本课的全部入口。注意 `graph_compute` 的注释直接写明它返回的是'
    ' "`ggml_backend_sched_graph_compute_async` 执行的结果"，`graph_reserve` 的注释则说明'
    '它用**一个假 ubatch** 去预留图 —— 这正是 `sched_reserve()` 的底层动作。',
    src=HDR, parts=[(138, 145)], lang='cpp')

L.section(
    '二、sched_reserve()：一次性预留',
    '`sched_reserve()` 在 `decode` 的第 1800 行被调用，但它第一件事就是检查开关：'
    '`sched_need_reserve` 为 false 时**整段跳过**。需要预留时，它按最坏情况（'
    '`n_tokens = min(n_ctx, n_ubatch)`、`n_seqs = n_seq_max`）算出 `max_nodes`，'
    '然后重建调度器、重建 `gf_res_reserve`，并让 memory 用 `init_full()` 准备一个"整块"的 memory context。\n\n'
    '这一段是本课"分配"侧的核心：**调度器与计算缓冲的容量在这里被钉死**，'
    '之后每个 token 都只是在这个容量里换一张图（细节见 L4-02）。',
    src=CPP, parts=[(637, 652)], lang='cpp')

L.section(
    '三、graph_reserve()：预留路径用到的三个调度器函数',
    '`sched_reserve()` 走的不是 `process_ubatch` 那条路，而是 `graph_reserve()`。'
    '同一个调度器对象，在预留路径上暴露三个不同的函数名 —— 这一段把"实际函数名"钉清楚：'
    '`ggml_backend_sched_reserve_size`（只算大小）、`ggml_backend_sched_split_graph`（只切分）、'
    '`ggml_backend_sched_reserve`（切分 + 分配）。',
    src=CPP, parts=[(2529, 2543)], lang='cpp')

L.section(
    '四、★ 复用槽有两个：gf_res_prev[n_outputs > 0]',
    '`get_gf_res_prev()` 只有四行，但它是 graph reuse 的关键：'
    '**按"这一轮 ubatch 有没有输出"选不同的 arena**。'
    '头文件里对应的注释解释了原因 —— 分开的 arena 让"有输出/无输出"的批在 CUDA graph 缓存里'
    '拿到不同的 key（见下面的成员引用）。',
    src=CPP, parts=[(2422, 2427)], lang='cpp')

L.section(
    '五、成员速查：sched / balloc / output_ids / sched_need_reserve',
    '`llama_context` 里与本课直接相关的成员都在这几行。'
    '特别值得注意的是 `balloc` 的注释："reuse the batch_allocr to avoid unnecessary memory allocations" —— '
    '批的解析对象也是复用的。',
    src=HDR, parts=[(330, 350)], lang='cpp')

L.section(
    '六、复用相关的成员与开关',
    '`gf_res_prev` 是一个长度为 2 的数组，`gf_res_reserve` 供预留使用，'
    '`gf_res_prev_active` 记住"当前哪一个 arena 里的图是有效的"。'
    '`graph_reuse_disable` 的注释写明它来自环境变量 `LLAMA_GRAPH_REUSE_DISABLE`（构造函数 280-281 行读取）。',
    src=HDR, parts=[(366, 386)], lang='cpp')

L.section(
    '七、输出：从图里取回 logits',
    '图算完之后，context 把 `res->get_logits()` 指到的张量当成**源**，'
    '用 `ggml_backend_tensor_get_async` 拷进自己的 host 缓冲 `logits.data`。'
    '注意中间的断言：目标偏移 `n_outputs_prev*n_vocab` 加上本轮输出，不能越过 `logits.size`。\n\n'
    '之后用户在 CPU 侧读 `llama_context::get_logits()` / `get_logits_ith(i)`（909、944 行；'
    'C API 侧的 `llama_get_logits` 在 3932 行），'
    '`get_logits_ith` 会先把批里的 token 序号用 `output_ids` 翻译成输出行号（`output_resolve_row`，915 行）。',
    src=CPP, parts=[(1933, 1946)], lang='cpp')

L.section(
    '八、复用计数进了 perf',
    '`n_reused` 不只是个内部计数器：它被塞进 `llama_perf_context_data`，'
    '并最终由 `llama_perf_context_print` 打印成一行 `graphs reused = ...`（4364 行）。'
    '所以"这一轮跑下来复用了几次"是**可以实测的数字**，不是推测。',
    src=CPP, parts=[(3419, 3434)], lang='cpp')

L.footnote_add('本课只引用 `src/llama-context.cpp` 与 `src/llama-context.h` 两个文件，二者都计入覆盖率。')
L.footnote_add('文中提到的 `llm_graph_result::can_reuse` / `llm_graph_params::allow_reuse` / '
               '`llm_graph_result::set_inputs` 定义在 `src/llama-graph.{h,cpp}`（L2-06 覆盖），'
               '本课只引用 `llama-context.cpp:1405`、`1449` 这两个调用点，'
               '故 `src/llama-graph.h` 不计入本课覆盖率。')
L.footnote_add('文中提到的 `ggml_backend_sched_*` 函数名与 `ggml_backend_tensor_get_async` 均取自本课引用的 '
               '两个文件正文，未从注释推断。')

L.prereqs('`L2-06`')

L.goal(
    '按顺序列出 `llama_decode` 内部从建图到 `ggml_backend_sched_graph_compute_async` 的每一步（对应验收点）；',
    '说明 `sched_reserve()` 为什么不是每 token 的固定开销（`sched_need_reserve` 开关）；',
    '解释 `process_ubatch` 里那个 `if` 的三个条件，以及 graph reuse 命中时省掉了哪两件事；',
    '说出 `gf_res_prev` 为什么是长度为 2 的数组，以及 `output_ids` 在输出取出时的作用。')

L.conclusion(
    '★ llama_decode 是图的生命周期管理器',
    '`llama_context::decode`（1704 行）一个函数里完成图的完整生命周期：\n\n'
    '| 阶段 | 调用 | 行号 |\n|---|---|---|\n'
    '| 建图 | `model.build_graph(gparams)` | 1425 |\n'
    '| 分配 | `ggml_backend_sched_alloc_graph(sched, gf)` | 1435 |\n'
    '| 执行 | `graph_compute` → `ggml_backend_sched_graph_compute_async` | 1454 → 2588 |\n'
    '| 复用 | `res->can_reuse(gparams)` → `n_reused++` | 1405 / 1415 |\n\n'
    '四件事都在 `process_ubatch`（1391 行）里收口，而每个 ubatch 只走一次。')

L.conclusion(
    '一次 decode 的两级循环',
    '- **逻辑批级（每 batch 一次）**：`balloc->init`(1765) → `sched_reserve`(1800) → `output_reserve`(1853) → '
    '`memory->init_batch`(1810)。\n'
    '- **ubatch 级（每 ubatch 一次）**：`process_ubatch`(1887) → 建图/复用 → `set_inputs`(1449) → '
    '`graph_compute`(1454)。\n\n'
    '循环由 `do { ... } while (mctx->next())`（1866 / 2041）驱动：'
    '**一个 `llama_batch` 可能产生多张图**，切分依据来自 memory 模块与 `cparams.n_ubatch`（回顾 L2-05）。')

L.conclusion(
    '★ graph reuse：把"重建图"降到"复用图"',
    '`process_ubatch` 里的判据是三个条件同时成立（1405 行）：\n\n'
    '```text\n'
    '!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)\n'
    '```\n\n'
    '命中时：不调用 `model.build_graph`、不调用 `ggml_backend_sched_alloc_graph`，'
    '只记录 `n_reused++`（1415）然后重填输入。\n'
    '这就是逐 token 生成时每步开销能被压下来的地方。复用槽有**两个**（`gf_res_prev[n_outputs > 0]`，2423 行），'
    '让"有输出"和"无输出"的图各自稳定命中；可用环境变量 `LLAMA_GRAPH_REUSE_DISABLE` 整体关掉（280-281 行）。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
