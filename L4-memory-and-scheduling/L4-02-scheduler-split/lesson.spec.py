#!/usr/bin/env python3
"""L4-02 · 调度器：把图切给不同后端 —— 课件 spec。

运行：python3 L4-memory-and-scheduling/L4-02-scheduler-split/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

CPP = 'ggml/src/ggml-backend.cpp'
HDR = 'ggml/include/ggml-backend.h'

L = Lesson(
    id='L4-02',
    layer='L4 · 内存与调度',
    title='★ 调度器：把图切给不同后端',
    codecap='ggml/src/ggml-backend.cpp · ggml/include/ggml-backend.h（逐字引用）',
    nav={'prev': {'href': '../L4-01-ggml-alloc/index.html', 'label': 'L4-01 分配器 ggml-alloc'},
         'next': {'href': '../L4-03-buffers-and-types/index.html', 'label': 'L4-03 buffer 与 buffer type'}},
)

L.note('**一句话**：`ggml_backend_sched` 是"多后端执行"的总管。它把一张图**沿数据流切成若干连续段**，'
       '每段整体交给一个后端；段与段之间的张量，必须在下游后端的 buffer 里**再出现一份** —— '
       '这份副本就是切分的产物，不是图里本来有的节点。')
L.note('本课覆盖 `ggml/src/ggml-backend.cpp`（v0.5.0，共 2513 行）与 `ggml/include/ggml-backend.h`（437 行）。'
       '所有引用都按行号从上游抽取，行号只对 v0.5.0 有效。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L4 · 内存与调度',
    title='★ <span class="hl-a">调度器</span>：把一张图切给多个后端',
    sub='ggml_backend_sched 把三件事绑在一个对象上：节点归谁、段间怎么搬、每段的内存谁来分。',
    caption='回顾 L3-02：后端数组的顺序就是优先级（头文件第 318 行写明）。回顾 L4-01：真正分配内存的是 galloc。下一课 L4-03 讲 buffer type。',
    src=HDR, parts=[(266, 270)], duration=17000,
    mark_src=[266, 267, 268, 269, 270],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">一张计算图（拓扑序）</span><span class="arrow">-></span>
    <span class="chip a">归属：谁跑</span><span class="arrow">-></span>
    <span class="chip c">切分：断成连续段</span><span class="arrow">-></span>
    <span class="chip d">搬运：段间 copy</span><span class="arrow">-></span>
    <span class="chip b">逐段执行</span>
  </div>
  <div class="row center" id="cards1" style="gap:8px"></div>
  <div class="formula" id="msg1"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '163px', t: '归属', m: 'ggml_backend_supports_op', b: '哪个后端能跑这个节点：<br>看 op，也看权重在哪个 buffer' },
  { c: 'c', w: '163px', t: '切分', m: 'ggml_backend_sched_split_graph', b: '沿拓扑序切成"连续的同后端段"，<br>每段是一个独立子图' },
  { c: 'd', w: '163px', t: '搬运', m: 'ggml_backend_tensor_copy', b: '段边界上的张量，要在下游<br>后端的 buffer 里再出现一份' },
  { c: 'b', w: '163px', t: '分配', m: 'ggml_gallocr_alloc_graph', b: '每段的张量由分配器按后端<br>落到各自 buffer（L4-01）' }
];
const host = wrap.querySelector('#cards1');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg1');
const texts = [
  '先看调度器在头文件里的自述 —— 三行字说清了它的职责范围。',
  '<span class="v">multiple backend devices to be used together</span>：'
    + '一个模型可以同时用 GPU、第二个 GPU、CPU。',
  '<span class="v">compute buffer allocation</span>：分配内存（这件事包给 L4-01 的 galloc）；<br>'
    + '<span class="v">assignment of tensors to backends</span>：决定每个张量归谁。',
  '<span class="v">copying of tensors between backends</span>：跨后端搬运。<br>'
    + '后面会看到：<span class="k">搬运的目标张量是切分时现场造出来的</span>。',
  '选后端的两个依据也写在注释里：<span class="k">后端支持这个 op</span> 和 '
    + '<span class="k">预分配张量（权重）在哪</span>。这两条就是下一幕的入口。'
];
els.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13900, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L4-02 · 数据结构',
    title='<span class="hl-c">split</span> = 连续区间 + 输入清单；<span class="hl-c">sched</span> = 账本',
    sub='切分的结果不是一个新图，而是"splits[] 数组 + 每个张量的后端 id"。',
    caption='splits 是动态数组：满了就翻倍（见 1347-1356）。',
    src=CPP, parts=[(775, 813)], duration=20000,
    mark_src=[775, 776, 777, 778, 779, 780, 783, 787, 788, 790, 792, 793, 794,
              797, 798, 799, 801, 802, 808, 811, 812, 813],
    notes_src={775: '一个 split 有两句话：图上哪一段，以及这段要"吃"什么',
               779: 'inputs[]：这一段开始前必须已经在本后端 buffer 里的张量',
               783: 'graph：这一段自己的子图视图（i_start..i_end），执行时整段丢给后端',
               794: 'galloc：就是 L4-01 的分配器 —— 调度器自己不分配内存',
               798: 'hv_tensor_backend_ids：张量 -> 后端 id，pass 1..4 填的就是它',
               799: 'hv_tensor_copies：(张量, 后端, 副本) -> 副本张量，跨段搬运的目标',
               802: 'leaf_backend_ids：叶子的归属（权重、图输入也在 leafs 里）',
               811: 'splits[]：切分结果。每段一个连续区间，不是"节点的集合"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl2"></div><div class="formula" id="msg2"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['字段', '含义', '谁写'],
  [['backends[] / bufts[]', '有序后端表 + 各自对应的 buffer type', 'sched_new'],
   ['hv_tensor_backend_ids', '张量 -> 后端 id', 'pass 1..4'],
   ['hv_tensor_copies', '(张量, 后端, 副本) -> 副本张量', 'pass 5'],
   ['splits[] / n_splits', '每段 = 连续区间 + 输入清单', 'pass 5'],
   ['galloc', '分配器句柄 —— 内存不归调度器管（L4-01）', 'sched_new'],
   ['op_offload', '是否允许把 op 抢到更靠前的后端', 'sched_new']],
  { monoCols: [0] });
wrap.querySelector('#tbl2').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');
const msg = wrap.querySelector('#msg2');
const texts = [
  '调度器的全部状态就是这张表 —— 读源码时按这张表对号入座。',
  '<span class="v">backends[] / bufts[]</span>：后端是有序的，'
    + '<span class="k">下标就是优先级</span>（0 最高）。',
  '<span class="v">hv_tensor_backend_ids</span>：归属的答案存在这里。'
    + '它按张量指针哈希（见 844 行的 <span class="k">hash_id</span> 宏），一次归属查询就是一次查表。',
  '<span class="v">hv_tensor_copies</span>：<span class="k">跨段搬运的"目标"在这里</span>。'
    + '一个源张量，在每个可能需要它的后端上，各有一份副本。',
  '<span class="v">splits[]</span>：切分结果。注意 <span class="k">i_start / i_end 是图上的下标</span>，'
    + '所以一个 split 必然是连续的一段。',
  '一句话：<span class="k">调度器只维护账本，不碰内存、不碰 kernel</span>。'
    + '内存是 galloc 的（L4-01），kernel 是各后端的（L4-04）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L4-02 · 归属（pass 1）',
    title='★ 归属优先级：<span class="hl-a">预分配 &gt; 视图 &gt; 图输入 &gt; 权重</span>',
    sub='backend_id_from_cur() 按固定顺序挑后端；一条都命中不了就返回 -1，交给后面的 pass。',
    caption='挑不出来不是错误：中间张量本来就没有"天生"的后端，要靠 pass 2 从邻居那里扩展出来（见 1124-1170）。',
    src=CPP, parts=[(921, 948)], duration=20000,
    mark_src=[921, 923, 924, 925, 926, 930, 931, 932, 933, 934, 938, 939, 940, 941,
              945, 946, 947, 948],
    notes_src={923: '① 已经有 buffer（权重、用户预分配）-> 直接用它的后端',
               930: '② 视图张量跟着 view_src 走 —— 视图不拥有数据，数据在哪它就在哪',
               938: '已经有 buffer 但这个后端跑不了这个 op：直接 abort，不会偷偷替它搬',
               945: '③ 图输入默认落在最后一个后端，也就是 CPU',
               946: '注释写死了约定：最后一个后端假定是 CPU',
               948: '④ 都命中不了 -> -1，留给 pass 2 扩展 / pass 4 兜底'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="cards3" style="gap:8px"></div>
  <div class="formula" id="msg3"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '163px', t: '① 预分配 buffer', m: '1.dst', b: '张量已经有 buffer：<br>在后端表里找支持它的' },
  { c: 'b', w: '163px', t: '② 视图', m: '1.vsrc', b: 'view_src != NULL：<br>归属 = view_src 的归属' },
  { c: 'c', w: '163px', t: '③ 图输入', m: '1.inp', b: 'GGML_TENSOR_FLAG_INPUT：<br>放最后一个后端（CPU）' },
  { c: 'd', w: '163px', t: '④ 权重', m: '1.wgt', b: 'src 是 WEIGHTS buffer：<br>跟着那个权重走' }
];
const host = wrap.querySelector('#cards3');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg3');
const texts = [
  'pass 1 只做一件事：给每个节点和每个叶子挑一个后端 id。顺序是固定的。',
  '<span class="v">① 预分配</span>：张量已经躺在某个 buffer 里（权重、用户自己分配的张量）。<br>'
    + '<span class="k">它不能被搬走</span>，所以只能找"支持这个 buffer type 又能跑这个 op"的后端。',
  '<span class="v">② 视图</span>：RESHAPE / VIEW 这类张量不拥有数据，'
    + '归属必须和 <span class="k">view_src</span> 一致，否则读到的就是别人地址空间里的指针。',
  '<span class="v">③ 图输入</span>：用户喂进来的张量默认放最后一个后端 —— '
    + '<span class="k">约定最后一个后端是 CPU</span>（第 946 行的注释）。',
  '<span class="v">④ 权重</span>：节点的输入里有 WEIGHTS buffer 时，节点跟着权重走 —— '
    + '这是最后一条，也是下一幕的主题。挑不出来就是 -1。'
];
els.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L4-02 · 归属细节',
    title='权重在哪个 buffer，<span class="hl-b">op 就归哪个后端</span>',
    sub='backend_from_buffer()：在有序后端表里找第一个"既支持这个 buffer type、又支持这个 op"的后端。',
    caption='回顾 L2-03：权重加载时就决定了它落在哪个 buffer；这里只是把那个决定翻译成后端 id。下一课 L4-03 展开 buffer type 本身。',
    src=CPP, parts=[(888, 908)], duration=17000,
    mark_src=[888, 889, 890, 891, 895, 896, 897, 898, 903, 904, 907],
    notes_src={889: '视图张量的 buffer 要看 view_src 的',
               890: '没有 buffer 的中间张量 -> -1，这一步答不了',
               894: '注释：找优先级最高、且同时支持这个 buffer type 和这个 op 的后端',
               896: '两个条件必须同时成立：supports_buft + supports_op',
               907: '没有后端同时满足 -> -1（debug 模式下会打一行警告）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia4" style="gap:10px"></div>
  <div class="formula" id="msg4"></div>`;
root.appendChild(wrap);

const dia = wrap.querySelector('#dia4');
const a = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
a.innerHTML = '<div class="ct" style="color:var(--b)">输入：一个已经分配好的张量</div>' +
  '<div class="cb">它带着 buffer（例如权重 buffer）<br>' +
  '<span class="cm" style="margin:0">buffer-&gt;buft</span> 就是它所在的内存种类</div>';
const b = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
b.innerHTML = '<div class="ct" style="color:var(--a)">输出：后端 id</div>' +
  '<div class="cb">从 0 开始扫后端表，返回第一个<br>' +
  '<span class="cm" style="margin:0">supports_buft(x) &amp;&amp; supports_op(x)</span><br>都成立的下标</div>';
dia.appendChild(a); dia.appendChild(U.arrow('->')); dia.appendChild(b);

const msg = wrap.querySelector('#msg4');
const texts = [
  '这是一个纯粹的小工具函数：把"张量在哪"翻译成"哪个后端 id"。',
  '<span class="v">supports_buft</span>：这个后端能不能在这个 buffer type 上干活'
    + '（显存 / pinned host / 普通内存，见 L4-03）。',
  '<span class="v">supports_op</span>：这个后端有没有实现这个算子。<br>'
    + '<span class="k">两个都成立才返回</span>，所以顺序里的第一个命中者就是答案。',
  '返回 <span class="v">-1</span> 有两种含义：张量还没分配（中间张量），'
    + '或者没有后端能接这个活。前者交给后面的 pass，后者在 debug 模式下会打警告。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 3, 5]); });
tl.at(4000, () => { msg.innerHTML = texts[1]; U.markLines(document, [10]); });
tl.at(7300, () => { msg.innerHTML = texts[2]; U.markLines(document, [11, 13, 14]); });
tl.at(10600, () => { msg.innerHTML = texts[3]; U.markLines(document, [19, 20, 23]); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L4-02 · 归属细节',
    title='★ <span class="hl-d">op_offload</span>：GPU 可以把 op 从 CPU 权重手里抢走',
    sub='权重在 host 内存时，只要更靠前的后端支持这个 op 且愿意 offload，op 就归它 —— 代价是数据得搬过去。',
    caption='回顾 L2-05：dev_layer / n_gpu_layers 决定哪些层的权重放 GPU；这里决定剩下那些"权重留在 host"的层里，哪些 op 还能被抢上去。实战见 L8-02。',
    src=CPP, parts=[(951, 985)], duration=22000,
    mark_src=[951, 952, 953, 955, 956, 958, 959, 961, 962, 963, 967, 968, 969, 970,
              971, 972, 973, 978, 979, 984],
    notes_src={951: '注释：用权重的 op，优先和权重跑在同一个后端',
               953: 'allow 先设为 true —— 默认允许"跟着权重走"',
               956: '例外 1：ROPE 的 rope freqs 张量太小，不足以决定归属',
               959: '例外 2：FLASH_ATTN_EXT 的 sinks 张量太小',
               967: '只对 usage == WEIGHTS 的输入这么做',
               969: '注释：看看有没有优先级更高、又愿意 offload 这个 op 的后端',
               970: '三个条件：op_offload 打开、权重在 host、默认归属是最后一个后端',
               972: 'supports_op + offload_op 都通过 -> 直接返回这个更靠前的后端',
               979: '抢不到就回到权重所在的后端'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="cards5" style="gap:8px"></div>
  <div class="formula" id="msg5"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'd', w: '220px', t: '抢的三个条件', b: '① sched-&gt;op_offload 打开<br>② 权重在 host buffer 里<br>③ 更靠前的后端 supports_op 且 offload_op' },
  { c: 'e', w: '220px', t: '两个例外', b: 'ROPE：rope freqs 太小<br>FLASH_ATTN_EXT：sinks 太小<br>小张量不足以决定 op 的归属' },
  { c: 'c', w: '220px', t: '抢不到就回落', b: '返回权重所在的后端 id<br>（通常是 CPU）<br>此时 op 和权重在一起，不用搬权重' }
];
const host = wrap.querySelector('#cards5');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg5');
const texts = [
  '权重在 host 上，不代表 op 一定要在 CPU 上算 —— 这一段就是"抢 op"的逻辑。',
  '<span class="v">src_backend_id == n_backends - 1</span>：先算出"按权重该在哪个后端"，'
    + '如果它正好是最后一个（CPU）……',
  '……并且权重在 <span class="v">host</span> 内存里，'
    + '<span class="k">就从 0 开始找有没有更靠前的后端愿意接</span>（for b &lt; src_backend_id）。',
  '注意 <span class="v">offload_op</span> 是后端自己的意见：它和 supports_op 是两个不同的判断'
    + '（头文件第 110 行把两者分开声明）。',
  '抢到了会怎样？归属只决定"谁来算"，<span class="k">数据怎么到位是 pass 5 的事</span>：'
    + '这个 host 上的权重会在边界被造一份副本搬过去（第 12 节）。'
];
els.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14700, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L4-02 · pass 5',
    title='★ 切分 = 沿图找<span class="hl-a">连续的同后端段</span>',
    sub='遍历拓扑序：节点后端和当前段不同就断一刀；节点带着"别的后端的权重"而本段又用不了它时，也断一刀。',
    caption='断点处旧段写 i_end = i，新段从 i 开始 —— 所以 i 属于新段：段永远是图上的连续区间，不会跳着挑节点。视图节点直接跳过不参与切段。',
    src=CPP, parts=[(1313, 1362)], duration=24000,
    mark_src=[1313, 1316, 1317, 1320, 1322, 1324, 1326, 1327, 1332, 1333, 1334, 1336,
              1337, 1344, 1345, 1346, 1357, 1358, 1359, 1361],
    notes_src={1316: '视图算子不参与切段 —— 它不产生数据，且归属必然和 view_src 相同',
               1320: '每个节点在这里读出归属 —— pass 1..4 已经保证它不等于 -1',
               1326: '只有当"本节点属于当前段"且"当前段已经有输入"时才检查权重',
               1332: '注释：权重在别的、且不兼容的后端上',
               1333: '注释：断一刀，上一段 offload 的权重内存就能早点还回去',
               1336: '权重在别的后端 + 本段不支持它的 buffer type -> 断段',
               1344: '真正的断段条件：后端换了，或者上面那个权重原因',
               1345: '旧段在这里收尾（i_end 不含 i）',
               1358: '新段从这个节点开始'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="formula" id="dia6" style="padding:8px 9px"></div>
  <div class="row center" id="cards6" style="gap:8px"></div>
  <div class="formula" id="msg6"></div>`;
root.appendChild(wrap);

const dia = wrap.querySelector('#dia6');
dia.innerHTML = '<div class="cm" style="margin:0 0 6px">示意：一张 6 节点的小图（注意力 -&gt; FFN），'
  + 'n1 的权重在显存，n3 的权重在 host，且这次没把 op offload 出去</div>'
  + '<div class="row" id="segs" style="gap:6px;align-items:center;flex-wrap:wrap"></div>';
const segs = dia.querySelector('#segs');

const box = (id, op, c) => {
  const e = U.el('div', { style: 'width:62px;padding:3px 4px;border-radius:5px;'
    + 'border:1px solid var(--border);background:#10151b;text-align:center' });
  e.innerHTML = '<div style="font-family:var(--mono);font-size:8.5px;color:var(--' + c + ')">' + id + '</div>'
    + '<div style="font-size:8px;color:var(--muted)">' + op + '</div>';
  return e;
};
const segBox = (title, c, nodes) => {
  const e = U.el('div', { class: 'col', style: 'gap:3px;padding:4px 5px;border:1px dashed var(--' + c + ');border-radius:6px' });
  e.appendChild(U.el('div', { style: 'font-size:8px;color:var(--' + c + ')', text: title }));
  const row = U.el('div', { class: 'row', style: 'gap:3px' });
  nodes.forEach(n => row.appendChild(n));
  e.appendChild(row);
  return e;
};
const s0 = segBox('split #0 · GPU', 'a', [box('n0', 'norm', 'a'), box('n1', 'matmul wq', 'a'), box('n2', 'softmax', 'a')]);
const s1 = segBox('split #1 · CPU', 'c', [box('n3', 'matmul wf', 'c'), box('n4', 'add', 'c')]);
const s2 = segBox('split #2 · GPU', 'd', [box('n5', 'matmul wo', 'd')]);
const cp1 = U.chip('copy', 'e'), cp2 = U.chip('copy', 'e');
segs.appendChild(s0); segs.appendChild(cp1); segs.appendChild(s1); segs.appendChild(cp2); segs.appendChild(s2);

const cards = wrap.querySelector('#cards6');
const c1 = U.card({ c: 'a', w: '300px', t: '断段判据 1：后端换了',
  b: 'node_backend_id != cur_backend_id<br>图是拓扑序，所以同后端的节点天然连成一段' });
const c2 = U.card({ c: 'c', w: '300px', t: '断段判据 2：权重在别的后端',
  b: '输入是 WEIGHTS buffer，且 src_backend_id != cur_backend_id，而本段又用不了它的 buffer type' });
cards.appendChild(c1); cards.appendChild(c2);
[c1, c2].forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg6');
const texts = [
  '先把一张小图摆出来：n0..n2 在 GPU，n3..n4 因为权重在 host 而留在 CPU，n5 又回到 GPU。',
  '<span class="v">判据 1</span>：走到 n3 时 node_backend_id（CPU）!= cur_backend_id（GPU）—— '
    + '旧段在 n3 前收尾，新段从 n3 开始。',
  '<span class="v">判据 2</span>：就算两个节点的后端相同，只要某个输入是"别的后端上的权重"、'
    + '而本段后端用不了那个 buffer type，也要断开。',
  '为什么要为权重特意断一刀？<span class="k">注释给了理由</span>：'
    + '断段之后，上一段 offload 上去的权重内存可以提前还给分配器（L4-01 复用）。',
  '于是切分结果就是 3 个段：<span class="v">[n0,n1,n2]</span>、'
    + '<span class="v">[n3,n4]</span>、<span class="v">[n5]</span>，以及 2 个边界。',
  '边界上那两个 <span class="v">copy</span> 不是图里的节点 —— 下一幕看它们是怎么被造出来的。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(4300, () => { msg.innerHTML = texts[1]; s0.style.borderColor = 'var(--a)'; });
tl.at(7900, () => { msg.innerHTML = texts[2]; s1.style.borderColor = 'var(--c)'; });
tl.at(11500, () => { msg.innerHTML = texts[3]; });
tl.at(15100, () => { msg.innerHTML = texts[4]; s2.style.borderColor = 'var(--d)'; });
tl.at(18700, () => { msg.innerHTML = texts[5]; [c1, c2].forEach(e => e.style.opacity = '1');
  [cp1, cp2].forEach(e => e.style.borderColor = 'var(--e)'); });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L4-02 · pass 5 核心',
    title='★ <span class="hl-a">copy 是切分的产物</span>，不是图里预先有的节点',
    sub='输入不在本段后端、本段后端又用不了它的 buffer 时：造一个本后端的副本张量，把 node->src[j] 改指向它。',
    caption='注意：调度器没有往图里插 GGML_OP_CPY 节点 —— 它只换了指针；真正的搬运发生在 compute_splits（下一幕）。',
    src=CPP, parts=[(1399, 1420)], duration=22000,
    mark_src=[1399, 1400, 1401, 1402, 1404, 1405, 1410, 1413, 1416, 1417, 1419],
    notes_src={1399: '判据：src 在别的后端，且本段后端不支持它的 buffer type',
               1400: '注释：在本段后端造一份输入的副本',
               1401: '每个 (张量, 后端) 只造一次 —— tensor_id_copy 就是缓存',
               1404: 'ggml_dup_tensor_layout：同形状同类型的新张量',
               1405: '名字里带上后端与副本号，debug 打印时一眼看出',
               1410: '副本按 n_copies 各存一份（并行流水线时才 >1）',
               1413: '把【源】张量登记进 split->inputs',
               1417: '执行时按这个清单搬运',
               1419: '关键一行：节点的输入指针被改成副本 —— 图被改写了'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia7" style="gap:10px"></div>
  <div class="formula" id="msg7"></div>`;
root.appendChild(wrap);

const dia = wrap.querySelector('#dia7');
const before = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--c)' });
before.innerHTML = '<div class="ct" style="color:var(--c)">切分前</div>' +
  '<div class="cb">n3（CPU 段）的 src[0] 指向<br>n2 的输出 —— 它在 GPU 的 buffer 里<br>' +
  '<span class="cm" style="margin:0">n3.src[0] -&gt; GPU buffer</span></div>';
const after = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
after.innerHTML = '<div class="ct" style="color:var(--a)">切分后</div>' +
  '<div class="cb">同一个 n3，src[0] 指向本段的副本张量<br>它分配在 CPU 段的 buffer 里<br>' +
  '<span class="cm" style="margin:0">n3.src[0] -&gt; CPU#&lt;n2&gt;#0</span></div>';
const ar = U.arrow('->');
dia.appendChild(before); dia.appendChild(ar); dia.appendChild(after);

const msg = wrap.querySelector('#msg7');
const texts = [
  '先记住结论：<span class="k">图里的节点数没变</span>，变的是某些节点的 src[] 指针。',
  '<span class="v">判据</span>：src 在别的后端，而且本段后端不支持它的 buffer type'
    + '（supports_buft 为假）。换句话说：<span class="k">本段的内核读不到那块内存</span>。',
  '<span class="v">动作 1</span>：ggml_dup_tensor_layout 造一个同形状的副本张量 —— '
    + '它会被分配器放进本段后端的 buffer（L4-01 按 node_backend_ids 决定）。',
  '<span class="v">动作 2</span>：把源张量登记进 split-&gt;inputs。'
    + '这份清单就是"这一段开始前要搬什么"的待办列表。',
  '<span class="v">动作 3</span>：node-&gt;src[j] = 副本。'
    + '从这一刻起，本段的子图只看得见自己后端上的张量 —— 所以它才敢整段丢给一个后端。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; ar.textContent = '切分'; });
tl.at(4000, () => { msg.innerHTML = texts[1]; before.style.borderColor = 'var(--a)'; });
tl.at(7300, () => { msg.innerHTML = texts[2]; });
tl.at(10600, () => { msg.innerHTML = texts[3]; });
tl.at(13900, () => { msg.innerHTML = texts[4]; after.style.borderColor = 'var(--a)';
  ar.textContent = '->'; ar.style.color = 'var(--e)'; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L4-02 · 分配',
    title='切完之后：<span class="hl-e">galloc</span> 按段分配内存',
    sub='alloc_graph() = split_graph() + alloc_splits()：内存分配整个包给 L4-01 的 galloc。',
    caption='只有后端归属（或 buffer type）变了，才会重新 reserve；否则每步推理复用同一份内存（见 1591-1641）。',
    src=CPP, parts=[(1992, 2009)], duration=20000,
    mark_src=[1992, 1994, 1995, 1997, 1998, 2000, 2002, 2003, 2006, 2008],
    notes_src={1995: '同一个图不能分配两次（is_alloc 断言）',
               1997: '并行模式下轮转副本号：cur_copy / next_copy',
               2000: '第一步：切图（pass 1..5）',
               2002: '第二步：分配 —— 内部把 sched->graph 交给 galloc',
               2006: '分配完成。之后 graph_compute 发现 is_alloc 为真就直接跑'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip c">split_graph()</span><span class="arrow">-></span>
    <span class="chip a">alloc_splits()</span><span class="arrow">-></span>
    <span class="chip e">ggml_gallocr_alloc_graph()</span><span class="arrow">-></span>
    <span class="chip b">每段的 buffer 就位</span>
  </div>
  <div class="row center" id="cards8" style="gap:8px"></div>
  <div class="formula" id="msg8"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'e', w: '220px', t: '分配器怎么知道放哪', b: 'alloc_splits 把 node_backend_ids / leaf_backend_ids 交给 gallocr（1591-1641）' },
  { c: 'b', w: '220px', t: '什么时候重新分配', b: '后端 id 变了、或 buffer type 变了<br>才 reserve；否则复用' },
  { c: 'd', w: '220px', t: 'reserve 是什么', b: '用一张"最大 batch"的测量图先切一遍，<br>把每段的 buffer 预留出来（1975）' }
];
const host = wrap.querySelector('#cards8');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg8');
const texts = [
  'alloc_graph 是"要开始跑了"的入口。它只有两步，但第一步是整课的重点。',
  '<span class="v">第一步 split_graph()</span>：重新算归属、重新切段、重新造副本。'
    + '每分配一张新图都要重来一遍。',
  '<span class="v">第二步 alloc_splits()</span>：把带后端 id 的图交给 galloc —— '
    + '<span class="k">L4-01 的 arena 按这些 id 决定每段的张量落在哪个 buffer</span>。',
  '<span class="v">cur_copy / next_copy</span>：并行（流水线）模式下，每次 alloc 换一份副本，'
    + '让上一次计算还在跑的时候，下一次就能写另一份输入。',
  '分配完之后 <span class="v">is_alloc</span> 为真，graph_compute 就不再重复切分与分配 —— '
    + '这是解码循环里"每步只算一次"的关键。'
];
els.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L4-02 · 执行',
    title='真搬运发生在 <span class="hl-c">compute_splits</span>：先搬 inputs，再跑子图',
    sub='每个 split：优先用后端的异步 copy 把 inputs 搬进本后端，然后把 split->graph 整段交给这个后端。',
    caption='回顾 L4-04：ggml_backend_graph_compute_async 是单后端的执行入口，调度器只是逐段调用它。',
    src=CPP, parts=[(1782, 1799)], duration=20000,
    mark_src=[1782, 1783, 1785, 1786, 1787, 1790, 1792, 1799],
    notes_src={1783: '先试异步 copy —— 它会排在源后端已有的工作之后',
               1785: '后端没有异步接口，或者它拒绝了这次拷贝 -> 退回同步',
               1786: '同步路径要自己保证顺序：先同步源后端',
               1792: '真正的数据搬运：src 后端 -> 本段后端的副本',
               1799: 'inputs 都就位后，整段子图交给本段后端执行'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">段 A 算完</span><span class="arrow">-></span>
    <span class="chip c">record event</span><span class="arrow">-></span>
    <span class="chip d">搬 inputs 到段 B</span><span class="arrow">-></span>
    <span class="chip b">graph_compute_async(段 B 的子图)</span>
  </div>
  <div class="row center" id="cards9" style="gap:8px"></div>
  <div class="formula" id="msg9"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '220px', t: '同步靠事件', b: '每个 (后端, 副本) 一个 event：<br>搬之前等它，跑完记它（1838）' },
  { c: 'd', w: '220px', t: '优先异步搬运', b: 'cpy_tensor_async(src, dst, ...)<br>不行才退回同步 copy（1785）' },
  { c: 'c', w: '220px', t: '段内不再切分', b: '搬完之后，这一段就是一整个<br>后端自己认识的标准子图' }
];
const host = wrap.querySelector('#cards9');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg9');
const texts = [
  '执行阶段是一个 for 循环：按切分顺序，一段一段地跑（1656）。',
  '<span class="v">先搬 inputs</span>：每个 split 都带着自己的输入清单，'
    + '清单里的每一个张量都要在段开始前落到本后端的 buffer 里。',
  '<span class="v">异步优先</span>：能异步就异步（拷贝排在源后端的队列里），'
    + '不能就同步搬 —— <span class="k">无论哪条路，段开始前数据一定就位</span>。',
  '<span class="v">再跑子图</span>：ggml_backend_graph_compute_async(split_backend, &amp;split-&gt;graph)。'
    + '<span class="k">注意这里传的是 split-&gt;graph，不是原始整图</span>。',
  '这就是切分的最终收益：每个后端只看到"自己的那一段 + 已经搬好的输入"，'
    + '不需要理解别的设备。下一课 L4-04 讲这个入口内部怎么跑。'
];
els.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L4-02 · 收束',
    title='把这一课压成一张表',
    sub='入口只有三个：split_graph（切）、alloc_graph（分配）、graph_compute（执行）。',
    caption='下一课 L4-03：buffer 与 buffer type —— 决定"数据放在哪种内存里"，也就是本课所有 buffer type 判据的另一半。',
    src=HDR, parts=[(339, 346)], duration=20000,
    mark_src=[339, 340, 342, 343, 344, 345, 346],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl10"></div><div id="ex10"></div><div class="formula" id="msg10"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['阶段', '函数', '留下的数据', '本课行号'],
  [['归属', 'ggml_backend_sched_backend_id_from_cur', 'hv_tensor_backend_ids', '921-985'],
   ['扩展', 'pass 2 的四个循环', 'node_backend_ids', '1124-1202'],
   ['切段', 'ggml_backend_sched_split_graph', 'splits[].i_start / i_end', '1324-1362'],
   ['造副本', 'ggml_dup_tensor_layout', 'hv_tensor_copies / split.inputs', '1399-1420'],
   ['分配', 'ggml_backend_sched_alloc_graph', 'galloc 的 arena（L4-01）', '1591-1641 / 1992-2009'],
   ['搬运+执行', 'ggml_backend_sched_compute_splits', '每个 split 的输入清单', '1782-1799']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl10').appendChild(t.el);

wrap.querySelector('#ex10').appendChild(W.exercise(
  '一张图的拓扑序是 n0(GPU) -&gt; n1(GPU) -&gt; n2(CPU) -&gt; n3(CPU) -&gt; n4(GPU)。'
  + '调度器会切出几个 split？在哪里产生副本张量？',
  '3 个 split：<span class="mono">#0 = [n0, n1]</span> 在 GPU，<span class="mono">#1 = [n2, n3]</span> 在 CPU，'
  + '<span class="mono">#2 = [n4]</span> 在 GPU。<br>'
  + '边界由 <span class="mono">node_backend_id != cur_backend_id</span> 触发：旧段写 <span class="mono">i_end = i</span>，'
  + '新段 <span class="mono">i_start = i</span>（1344-1359）。<br>'
  + '两处边界各造一个副本张量：n1 的输出要在 CPU 段的 buffer 里有一份，n3 的输出要在 GPU 段的 buffer 里有一份'
  + '（<span class="mono">ggml_dup_tensor_layout</span> + <span class="mono">node-&gt;src[j] = 副本</span>，1399-1419）。<br>'
  + '注意：图里的节点数没变，还是 5 个 —— 变的只是 n2 和 n4 的 <span class="mono">src[]</span> 指针。'));

const msg = wrap.querySelector('#msg10');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '六个阶段，一条流水线：归属 -> 扩展 -> 切段 -> 造副本 -> 分配 -> 执行。',
  '<span class="k">归属</span>：权重 buffer、视图、图输入三条硬规则，加上 op_offload 的例外。',
  '<span class="k">切段</span>：沿拓扑序把同后端的节点连成连续区间；权重不兼容时特意断开，好让内存早点回收。',
  '<span class="k">造副本</span>：copy 不是图里预置的节点，而是切分时按需造出来的目标张量。',
  '<span class="k">分配 + 执行</span>：galloc 按后端 id 落 buffer（L4-01），compute_splits 逐段搬运并执行（L4-04）。',
  '验收点：<span class="v">段只由一个后端执行，它的 kernel 只认自己 buffer type 上的内存</span> —— '
    + '所以边界上必须把数据搬到下游后端的 buffer 里，这个"搬"就是副本张量存在的理由。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(16500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、调度器在头文件里的自述',
    '三行注释定义了它的职责边界：**多设备协同、compute buffer 分配、张量归属、跨后端拷贝**。'
    '注意最后一条 —— 本课后半段会看到，那些拷贝用的目标张量是**切分时现场造出来的**，'
    '不是图里本来就有的节点。',
    src=HDR, parts=[(266, 270)], lang='c')

L.section(
    '二、后端顺序就是优先级',
    '头文件把约定写在了构造函数上方：**下标小的后端优先级高**。'
    '所以调用方传进来的顺序（GPU 在前、CPU 在后）直接决定了归属结果 —— '
    '这也是 L3-02 里"设备枚举顺序会影响 offload"的根。',
    src=HDR, parts=[(318, 319)], lang='c')

L.section(
    '三、最后一个后端必须是 CPU',
    '构造函数用断言把上一条约定钉死：数组最后一个后端必须是 **CPU 类型设备**。'
    '后文多处依赖这一点，例如图输入默认落在 `n_backends - 1`（第 946 行）、'
    '以及 pass 2 里"跳过 CPU 只扩展 GPU"。',
    src=CPP, parts=[(1855, 1857)], lang='c')

L.section(
    '四、数据结构：split 与 sched',
    '整个调度器的状态都在这两个结构体里。读法：`split` 回答"哪一段、吃什么"，'
    '`sched` 回答"每个张量归谁、副本在哪、内存谁管"。\n\n'
    '| 字段 | 含义 |\n|---|---|\n'
    '| `split.backend_id` / `i_start` / `i_end` | 这一段归哪个后端、是图上哪一段 |\n'
    '| `split.inputs[]` / `n_inputs` | 这一段开始前必须搬进来的张量清单 |\n'
    '| `split.graph` | 这一段的子图视图，执行时整段交给一个后端 |\n'
    '| `hv_tensor_backend_ids` | 张量 -> 后端 id |\n'
    '| `hv_tensor_copies` | (张量, 后端, 副本) -> 副本张量 |\n'
    '| `splits[]` / `n_splits` | 切分结果 |\n'
    '| `galloc` | 分配器句柄（L4-01） |\n'
    '| `op_offload` | 是否允许把 op 抢到更靠前的后端 |',
    src=CPP, parts=[(775, 813)], lang='c')

L.section(
    '五、归属（1）：权重所在 buffer 决定候选后端',
    '`ggml_backend_sched_backend_from_buffer()` 是一个纯查询：把"张量在哪个 buffer"'
    '翻译成"哪个后端 id"。判据是两个条件同时成立 —— 后端**支持这个 buffer type**，'
    '并且**支持这个 op**。它按后端表顺序扫描，所以返回的永远是优先级最高的合格者；'
    '一个都没有时返回 `-1`，debug 模式下还会打一行警告说明"这个权重将需要被拷贝"。',
    src=CPP, parts=[(888, 908)], lang='c')

L.section(
    '六、归属（2）：pass 1 的优先级',
    '`ggml_backend_sched_backend_id_from_cur()` 是 pass 1 的全部逻辑，顺序固定：\n\n'
    '1. 张量已经有 buffer（权重、用户预分配）-> 用它的后端（`1.dst`）；\n'
    '2. 视图张量 -> 跟 `view_src` 走（`1.vsrc`）；\n'
    '3. 图输入 -> 最后一个后端，即 CPU（`1.inp`）；\n'
    '4. 输入里有 `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` 的张量 -> 跟权重走（`1.wgt`）。\n\n'
    '一条都不命中就返回 `-1`：这不是错误，而是"等后面的 pass 再定"。'
    '那些 `SET_CAUSE` 字符串是打开调试输出后能看到的归属理由。',
    src=CPP, parts=[(921, 948)], lang='c')

L.section(
    '七、归属（3）：权重与 op_offload',
    '第 4 条规则还有反转：如果按权重算出来的后端正好是最后一个（CPU）、权重又在 host 内存里，'
    '并且构造调度器时 `op_offload` 打开，那么更靠前的后端只要 `supports_op` + `offload_op` 都通过，'
    '就可以把这个 op **抢过去**（`1.off`）。\n\n'
    '两个例外被显式跳过：`GGML_OP_ROPE`（rope freqs 张量太小）和 `GGML_OP_FLASH_ATTN_EXT`'
    '（sinks 张量太小）—— 源码注释说明理由：小张量不足以决定 op 的归属。\n\n'
    '回顾 L2-05：`dev_layer` / `n_gpu_layers` 决定哪些层的权重放 GPU；'
    '这里决定剩下那些"权重留在 host"的层里，哪些 op 还能被抢上去。实战见 L8-02。',
    src=CPP, parts=[(951, 985)], lang='c')

L.section(
    '八、pass 2：把相邻节点并到同一个后端',
    'pass 1 只给"有据可依"的节点定了归属，中间的激活张量大多是 `-1`。'
    'pass 2 用多轮扫描把它们补上：**先处理 GPU（非最后优先级的后端），沿图向下、向上各扫一遍；'
    '忽略 CPU**。下面引的是 GPU 的两遍，其余后端的两遍紧随其后（1172-1202）。\n\n'
    '注释里的三条结论值得抄下来：\n\n'
    '```text\n'
    'assign the same backend to adjacent nodes\n'
    'expand gpu backends up and down, ignoring cpu (the lowest priority backend)\n'
    'thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops\n'
    '```\n\n'
    '也就是说：**"能连成一片"本身就是切分想要的结果** —— pass 2 主动把同后端的节点连成段，'
    'pass 5 才只需要在真正连不起来的地方断刀。',
    src=CPP, parts=[(1124, 1170)], lang='c')

L.section(
    '九、pass 4：视图跟着 view_src，兜底保证没有节点落空',
    'pass 4 做两件事：把还没归属的**视图张量**绑到 `view_src` 的归属上，'
    '把还没归属的**源张量**绑到"使用它的那个节点"的归属上；最后给仍然落单的节点'
    '挑第一个支持它的后端，并用 `GGML_ASSERT(*cur_backend_id != -1)` 保证**没有节点落空**。\n\n'
    '这条断言的另一面是：如果所有后端都不支持某个 op，程序会在这里直接挂掉，'
    '而不是悄悄算出错误结果。',
    src=CPP, parts=[(1265, 1295)], lang='c')

L.section(
    '十、pass 5：切段的判据',
    'pass 5 是"切分"这件事真正发生的地方。它沿拓扑序走一遍，维护一个"当前段"：\n\n'
    '- 节点的后端和当前段不同 -> 断段；\n'
    '- 节点带着"别的后端上的权重"、而当前段后端**又不支持那个 buffer type** -> 也断段。'
    '源码注释给出理由：断开之后，上一段 offload 上去的权重内存可以更早被复用。\n\n'
    '断段时旧段写 `i_end = i`、新段写 `i_start = i`，所以节点 `i` 属于新段 —— '
    '**边界是"切在节点之前"，每段永远是图上的连续区间**。',
    src=CPP, parts=[(1313, 1362)], lang='c')

L.section(
    '十一、搬运判据：buffer_supported',
    '决定"要不要造副本"的是这个函数：张量如果**已经分配**，就看它所在 buffer 的 buffer type；'
    '否则看它已经被分配到的后端对应的 buffer type。'
    '然后问下游后端一句 `ggml_backend_supports_buft()` —— 不支持就得搬。\n\n'
    '这就是验收点的技术根：**不同的后端用不同的 buffer type 管理内存**'
    '（显存 / pinned host / 普通内存，见 L4-03），一个后端的内核读不到另一个后端 buffer 里的数据。',
    src=CPP, parts=[(1037, 1056)], lang='c')

L.section(
    '十二、★ 边界造副本：copy 是切分的产物',
    '当某个输入既不在本段后端上、本段后端又用不了它的 buffer type 时，调度器就地造副本：\n\n'
    '1. `ggml_dup_tensor_layout()` 在调度器自己的 ctx 里造一个同形状同类型的新张量；\n'
    '2. 把它按 `(张量, 后端, 副本)` 记进 `hv_tensor_copies`，每个组合只造一次；\n'
    '3. 把**源**张量登记进 `split->inputs`（执行时的搬运清单）；\n'
    '4. 把 `node->src[j]` 改成指向副本。\n\n'
    '**关键：这不是往图里插一个 `GGML_OP_CPY` 节点。**'
    '图里的节点数不变，变的是某些节点的输入指针 —— 副本张量是"数据在哪"的改写，'
    '真正的搬运发生在 `ggml_backend_sched_compute_splits()` 里（下一节）。',
    src=CPP, parts=[(1399, 1420)], lang='c')

L.section(
    '十三、分配：把切好的图交给 galloc',
    '`ggml_backend_sched_alloc_splits()` 先判断**后端归属是否变化**：'
    '只有当 id 变了、或上一次分配失败时，才重新 `ggml_gallocr_reserve_n()`；'
    '随后 `ggml_gallocr_alloc_graph()` 真正落地地址。\n\n'
    '注意两个 id 数组是给 `reserve_n` 的 —— **调度器通过 `node_backend_ids` / `leaf_backend_ids` '
    '把"谁在哪"告诉分配器（L4-01），分配器再决定"地址在哪"**。'
    '重新 reserve 之前还会把所有后端同步一遍：因为重分配可能挪动 split 输入张量的地址。',
    src=CPP, parts=[(1591, 1641)], lang='c')

L.section(
    '十四、执行：先搬 inputs，再逐段 compute',
    '每个 split 的执行是固定两步：\n\n'
    '1. **搬 inputs**：优先走 `cpy_tensor_async`；后端不支持或拒绝时退回同步 `ggml_backend_tensor_copy()`，'
    '退回前会先同步源后端，保证数据真的写好；\n'
    '2. **跑子图**：`ggml_backend_graph_compute_async(split_backend, &split->graph)`。'
    '注意传进去的是 `split->graph`（这一段的子图），不是原始整图。\n\n'
    '回顾 L4-04：这个函数是单后端的执行入口；调度器只是按顺序对每一段各调一次。',
    src=CPP, parts=[(1782, 1799)], lang='c')

L.section(
    '十五、reserve：用测量图先切一遍',
    '`ggml_backend_sched_reserve()` 的用法在头文件示例里：拿一张"最大 batch"的图，'
    '走一遍 `split_graph()` + `ggml_gallocr_reserve_n()`，把每段的 buffer 先预留出来，'
    '然后 `reset` 掉。这样正式推理时不会因为要临时扩显存而失败或抖动。\n\n'
    '注意它和 `alloc_graph()` 一样会调用 `ggml_backend_sched_split_graph()` —— '
    '**切分是每次分配都要重算的，不是一次性编译**。',
    src=CPP, parts=[(1975, 1990)], lang='c')

L.section(
    '十六、入口：alloc_graph 与 graph_compute_async',
    '`ggml_backend_sched_alloc_graph()` 是"要开始跑了"的入口：'
    '先按 `next_copy` 轮转副本号（并行流水线时 `n_copies > 1`），再 `split_graph()`，'
    '然后 `alloc_splits()`，最后把 `is_alloc` 置真。\n\n'
    '`graph_compute_async()` 的顺序是：没 reset 就先 reset；没分配就调 alloc_graph；'
    '然后 `compute_splits()`。所以**第一次 graph_compute 会自动带上切分与分配**，'
    '之后每步只执行。这两个函数的差别，就是 L4-04 的主题。',
    src=CPP, parts=[(1992, 2009)], lang='c')

L.section(
    '十七、段与段之间的握手：events',
    '`compute_splits()` 的循环头能看到两件事：一是调度器**按 split 顺序逐段执行**；'
    '二是在特定情况下要等上一段：当前段没有输入、且上一段是别的后端时，'
    '必须等上一段的异步工作完成 —— 源码注释给了理由：'
    '**分配器可能在不同段之间复用了同一块 buffer 区域**（这正是 L4-01 的复用策略在调度器侧的代价）。\n\n'
    '每段跑完则记录下来，供后面的段等待。',
    src=CPP, parts=[(1656, 1670)], lang='c')

L.section(
    '十八、事件记录与副本',
    '段执行完成之后记录事件，然后进入下一段。'
    '`events[][]` 的第一维是后端、第二维是副本号 —— 与 `hv_tensor_copies` 的第三维对应。\n\n'
    '同一段代码上方还能看到环境变量 `GGML_SCHED_DEBUG`（第 1861 行）：它把 `sched->debug` 打开，'
    '于是 `ggml_backend_sched_print_assignments()` 会打印每个 split 的后端、输入清单，'
    '以及（`debug > 1` 时）每个节点的归属理由。',
    src=CPP, parts=[(1837, 1843)], lang='c')

L.section(
    '十九、并行副本与调试开关',
    '`n_copies` 来自构造参数 `parallel`：并行（流水线）模式下取 `GGML_SCHED_MAX_COPIES`，'
    '否则为 1。这就是 `hv_tensor_copies` 里第三维的来源；'
    '同一段里 `GGML_SCHED_DEBUG` / `GGML_SCHED_DEBUG_REALLOC` 两个环境变量分别控制'
    '归属打印与"重分配是否属于意外"的断言。',
    src=CPP, parts=[(1861, 1872)], lang='c')

L.section(
    '二十、用户可以手动指定归属',
    '`ggml_backend_sched_set_tensor_backend()` 直接写 `tensor_backend_id(node)` 并清掉 `is_reset` 标记。'
    'pass 1 的两个循环里专门有 "do not overwrite user assignments" 的保护（第 1091、1100 行），'
    '所以手动指定的归属不会被算法覆盖 —— 这也是头文件示例里 '
    '"manually assign nodes to a backend (optional)" 的落点。',
    src=CPP, parts=[(2088, 2095)], lang='c')

L.footnote_add('本课引用 `ggml/src/ggml-backend.cpp` 与 `ggml/include/ggml-backend.h` 两个文件，均计入覆盖率。')
L.footnote_add('第 6 幕里那张 6 节点小图是**示意**，不是源码：它的前提写在幕内（`op_offload` 关闭，'
               '所以带 host 权重的 `mul_mat` 留在 CPU）。示意里的节点名（n0/n1/...）只在本课内有效。')
L.footnote_add('本课不引用 `ggml/src/ggml-backend-reg.cpp`（设备如何注册与排序）—— 那是 L3-02 的内容；'
               '也不引用 `ggml/src/ggml-alloc.c`（arena 怎么切）—— 那是 L4-01 的内容。')

L.prereqs('`L4-01`（分配器 ggml-alloc）；建议先看 `L2-03`（权重落在哪个 buffer）、`L3-02`（设备顺序）')

L.goal(
    '说出 `ggml_backend_sched` 里"张量 -> 后端 id"和"张量 -> 副本"分别存在哪个字段里；',
    '复述 pass 1 的归属优先级（预分配 / 视图 / 图输入 / 权重），并说出 `op_offload` 如何把它反转；',
    '解释切分边界为什么必须插 copy：**一段子图只由一个后端执行，它的 kernel 只认自己 buffer type 上的内存**（对应验收点）；',
    '说出副本张量在哪个函数里、用哪个 API 造出来，真正的数据搬运又发生在哪个函数里。')

L.conclusion(
    '调度器只做三件事',
    '`ggml_backend_sched` 的职责可以压成一张表：\n\n'
    '| 事 | 谁干 | 本课行号 |\n|---|---|---|\n'
    '| 归属：每个张量归哪个后端 | `ggml_backend_sched_backend_id_from_cur` + pass 2/3/4 | 921-1295 |\n'
    '| 切分：沿拓扑序断成连续段 | `ggml_backend_sched_split_graph` pass 5 | 1313-1362 |\n'
    '| 搬运：段边界上造副本并搬数据 | `ggml_dup_tensor_layout` + `ggml_backend_tensor_copy` | 1399-1420 / 1782-1799 |\n\n'
    '内存分配不在这里 —— 那是 L4-01 的 `ggml_gallocr_alloc_graph`；'
    '段内怎么执行也不在这里 —— 那是 L4-04 的 `ggml_backend_graph_compute_async`。')

L.conclusion(
    '★ 切分不是"给节点分组"，而是"沿数据流找连续段"',
    'pass 2 先把同后端的相邻节点**主动连成一片**，pass 5 才在连不起来的地方断刀。'
    '结果就是 `splits[]` 里的一段 = 图上的一个**连续下标区间**（`i_start` .. `i_end`）+ 它需要的输入清单。\n\n'
    '**copy 是切分的产物**：节点本身没有被改写成一个"拷贝算子"，'
    '而是它的 `src[]` 指针被换成了下游后端的副本张量，真正的搬运在 `compute_splits()` 里执行。'
    '所以"边界为什么必须插 copy"的答案是：**一个段整体交给一个后端，'
    '而它的 kernel 只能读自己 buffer type 上的内存**（`ggml_backend_supports_buft`）；'
    '上游段在上游 buffer 里产出的张量，下游段根本读不到。')

L.conclusion(
    '★ 权重的位置决定图的形状',
    '`ggml_backend_sched_backend_id_from_cur()` 的第 4 条规则说：带权重的 op 优先和权重同后端。'
    '所以"哪些权重放 GPU"这个在加载期做出的决定（L2-03、L2-05 的 `dev_layer`），'
    '直接决定了图的切分结果。`op_offload` 是唯一的反转开关：'
    '权重留在 host，但 op 可以被更靠前的后端抢走 —— 代价是权重数据要在边界被搬一次。\n\n'
    '这也是 L8-02 要实测的东西：给定 `n_gpu_layers`，预测图会被切成几段。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
