#!/usr/bin/env python3
"""L1-01 · ggml 张量：算子的数据面 —— 课件 spec。

运行：python3 L1-operator-representation/L1-01-tensor-data-plane/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC = 'ggml/include/ggml.h'

L = Lesson(
    id='L1-01',
    layer='L1 · 算子的表示',
    title='ggml 张量：算子的数据面',
    codecap='ggml/include/ggml.h（逐字引用）',
    nav={'prev': {'href': '../../index.html', 'label': '门户'},
         'next': {'href': '../L1-02-op-enum-and-nargs/index.html', 'label': 'L1-02 算子枚举与元数据'}},
)

L.note('**一句话**：在 llama.cpp 里，一个"算子"不是一个函数调用，而是**计算图上的一个节点**；'
       '而这个节点同时有**数据面**（`struct ggml_tensor`：形状、类型、数据在哪）和'
       '**身份面**（`enum ggml_op`：它是什么运算）。这一课只讲数据面。')
L.note('读懂 `ggml_tensor` 是本视角的起点：后面每一课——图构建、后端切分、CPU/GPU 内核——'
       '都在读写这个结构体里的同样几个字段。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L1 · 算子的表示',
    title='一个算子，两个面',
    sub='数据面决定"算什么形状、什么精度、数据在哪"；身份面决定"做什么运算"。',
    caption='下一课 L1-02 讲身份面（GGML_OP_* 枚举与参数个数表）。',
    src=SRC, parts=[(684, 686)], duration=14000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">模型定义</span><span class="arrow">-></span>
    <span class="chip a">图上节点</span><span class="arrow">-></span>
    <span class="chip b">后端内核</span>
  </div>
  <div class="row center" id="faces" style="gap:11px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '数据面 · struct ggml_tensor', b: '形状 ne[]、步长 nb[]、类型 type、<br>数据指针 data、所在 buffer', m: 'struct ggml_tensor' },
  { c: 'c', t: '身份面 · enum ggml_op', b: '这个节点做什么运算：<br>MUL_MAT / RMS_NORM / SOFT_MAX ...', m: 'enum ggml_op op;' },
  { c: 'b', t: '执行面（后面的层）', b: '调度器把它切给某个后端，<br>后端用自己的 kernel 实现它', m: 'ggml_backend_graph_compute' }
];
const host = wrap.querySelector('#faces');
const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.33');
const msg = wrap.querySelector('#msg');
const texts = [
  '一个 ggml 张量 = <span class="k">形状</span> + <span class="k">类型</span> + <span class="k">一个 op</span>。它就是图上的节点。',
  '数据面回答：<span class="v">ne[]</span> 多少元素、<span class="v">nb[]</span> 怎么跳、<span class="v">type</span> 什么精度、<span class="v">data</span> 在哪。',
  '身份面回答：这个节点是 <span class="v">GGML_OP_MUL_MAT</span> 还是 <span class="v">GGML_OP_RMS_NORM</span>。',
  '后面的课：L2 讲节点怎么被建出来，L4 讲它怎么被切给后端，L5/L6/L7 讲内核怎么实现它。'
];
defs.forEach((_, i) => tl.at(600 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.33'; });
  msg.innerHTML = texts[i];
}));
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L1-01 · 常量',
    title='结构体的形状被几个 <span class="hl-c">宏</span>钉死',
    sub='这些上限不是随手取的：它们直接决定 struct 的布局，也决定后端内核能假设什么。',
    caption='GGML_MAX_OP_PARAMS = 64 字节，即最多 16 个 int32 参数。',
    src=SRC, parts=[(222, 226)], duration=16000,
    marks=[0, 2, 4],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { k: 'GGML_MAX_DIMS', v: '4', c: 'a', b: '最多 4 维。ne[]/nb[] 定长数组，<br>不堆分配 -> 结构体大小固定。' },
  { k: 'GGML_MAX_PARAMS', v: '2048', c: 'b', b: '算子的参数个数上限<br>（如 CONV 的 kernel 尺寸）。' },
  { k: 'GGML_MAX_SRC', v: '10', c: 'c', b: '一个节点的输入张量上限。<br>src[] 定长指针数组。' },
  { k: 'GGML_MAX_OP_PARAMS', v: '64', c: 'd', b: 'op_params 的字节数 = <br>16 个 int32（备注写明按 int32 对齐）。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.el('div', { class: 'card' });
  e.style.borderLeftColor = 'var(--' + d.c + ')';
  e.style.width = '163px';
  e.innerHTML = '<div class="cm" style="margin:0 0 2px">' + U.esc(d.k) + '</div>' +
    '<div class="ct" style="color:var(--' + d.c + ');font-size:16px">' + d.v + '</div>' +
    '<div class="cb">' + d.b + '</div>';
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '先看四个上限。它们出现在代码最前面，因为 <span class="k">struct 的字段都是定长数组</span>。',
  '<span class="v">GGML_MAX_DIMS = 4</span>：所以 <span class="k">ne[4]</span> / <span class="k">nb[4]</span> 是定长的，不是指针。<br>这意味着任何算子内核都能假设"最多 4 维"，不用处理任意秩。',
  '<span class="v">GGML_MAX_SRC = 10</span>：所以 <span class="k">src[10]</span> 装得下一个节点的全部输入。<br>图的边就是这些指针，不需要单独的边表。',
  '<span class="v">GGML_MAX_OP_PARAMS = 64</span>：所以 <span class="k">op_params[16]</span>（int32）。<br>算子的标量参数（轴、eps、scale）都塞在这 64 字节里。',
  '结论：<span class="k">ggml_tensor 是一个定长、可 memcpy、可放进 arena 的值类型</span>。<br>这一点在 L1-05（context 用 arena）和 L1-03（图遍历）里会反复用到。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  U.markLines(document, [i === 0 ? 0 : (i === 1 ? 2 : (i === 2 ? 4 : -1))].filter(x => x >= 0));
}));
tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [0, 2, 4]); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L1-01 · 核心',
    title='★ <span class="hl-a">struct ggml_tensor</span> 逐字段',
    sub='这就是"算子的数据面"的全部。后面所有后端都在读同样这几个字段。',
    caption='注意：结构体里同时有 src[]（输入）和 op（运算）—— 节点与边都在这一个结构里。',
    src=SRC, parts=[(685, 717)], duration=26000,
    marks=[1, 3, 5, 6, 12, 14, 17, 20, 23, 25, 27],
    notes={2: 'type 决定 nb[0] 与量化块大小，是后端选 kernel 的第一依据',
           4: 'buffer 指向数据实际所在的后端缓冲；NULL 表示还没分配',
           11: 'op_params：算子的标量参数。按 int32 对齐存放，共 16 个',
           13: 'flags：见 GGML_TENSOR_FLAG_*（如 OUTPUT / PARAM / LOSS）',
           18: 'view_src + view_offs：视图算子（VIEW/RESHAPE/TRANSPOSE）不拥有数据',
           22: 'name：给调试和 graph 打印用，长度 GGML_MAX_NAME',
           24: 'extra：后端私有数据，例如 ggml-cuda.cu 的额外信息'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '身份/精度', m: 'type · op · op_params[16] · flags', b: '它是什么类型、做什么运算、参数是什么' },
  { c: 'b', t: '形状', m: 'ne[4]', b: '每一维有多少元素' },
  { c: 'c', t: '步长', m: 'nb[4]', b: '每一维跨多少字节；与 type 绑定' },
  { c: 'd', t: '位置', m: 'buffer · data · extra', b: '数据在哪个后端缓冲、指针是多少' },
  { c: 'e', t: '连接', m: 'src[10]', b: '输入张量 —— 这就是图的边' },
  { c: 'f', t: '视图', m: 'view_src · view_offs', b: '不拥有数据时，指向源张量 + 偏移' },
  { c: 'g', t: '调试', m: 'name[64]', b: '图上打印、报错定位用' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m }, { style: 'width:218px' });
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '先按"用途"把 12 个字段分成 <span class="k">7 组</span>，再逐个看源码。',
  '<span class="v">type / op / op_params / flags</span>：一个节点的"是什么"。<br>op 是身份，type 是精度，op_params 是标量参数。',
  '<span class="v">ne[4]</span>：number of elements。<span class="k">ne[0] 是最内层（连续）维度</span>。<br>ggml 的约定是"列主序"式：ne[0] 变化最快。',
  '<span class="v">nb[4]</span>：stride in bytes。这是全课最关键的一处约定 —— 下一幕专门讲。',
  '<span class="v">buffer / data / extra</span>：<span class="k">算法与内存解耦</span>。<br>同一个 op，数据可以在 CPU 的 malloc 上，也可以在 CUDA 显存里。',
  '<span class="v">src[10]</span>：输入张量指针数组。<span class="k">图的边不需要额外结构</span> —— 就是这些指针。',
  '<span class="v">view_src / view_offs</span>：VIEW / RESHAPE / TRANSPOSE 这类算子<br><span class="k">不复制数据</span>，只是换一个"看"的方式。',
  '这正是 L4-01（ggml-alloc）能省内存的原因：<br>视图不占新内存，只是换个偏移。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(25200, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[7];
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L1-01 · 核心',
    title='★ <span class="hl-c">ne[] / nb[]</span>：形状与步长',
    sub='源码注释里已经把公式写死了。看懂这三行，就看懂了 ggml 的内存布局约定。',
    caption='padding 来自 GGML_MEM_ALIGN 对齐；量化类型下 ne[0] 必须是 blck_size 的整数倍。',
    src=SRC, parts=[(690, 694)], duration=20000,
    marks=[1, 2, 3, 4],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="calc" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const calc = wrap.querySelector('#calc');
const right = wrap.querySelector('#right');

calc.innerHTML = '<div class="cm" style="margin-bottom:4px">一个 Q4_0 张量：ne = [256, 3, 1, 1]</div>';
const rows = [
  { n: 'ggml_blck_size(Q4_0)', v: '32', c: 'a' },
  { n: 'ggml_type_size(Q4_0)', v: '18 字节/块', c: 'b' },
  { n: 'nb[0]', v: '18', c: 'c' },
  { n: 'nb[1]', v: '18 * (256/32) + pad = 144', c: 'd' },
  { n: 'nb[2]', v: '144 * 3 = 432', c: 'e' },
  { n: 'nb[3]', v: '432 * 1 = 432', c: 'f' }
];
const els = rows.map(r => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
  e.innerHTML = '<span class="m">' + U.esc(r.n) + '</span> = <span style="color:var(--' + r.c + ')">' + U.esc(r.v) + '</span>';
  calc.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

right.innerHTML = '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">为什么 nb[0] 不是 4 字节？</div>' +
  '<div class="cb">Q4_0 是<b>块量化</b>：32 个权重打包成一块（1 个 scale + 16 字节 4-bit）。' +
  '所以一"列"跨 18 字节，而不是元素大小 × 数量。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">为什么视角上"ne[0] 是最内层"？</div>' +
  '<div class="cb">因为 nb[0] 最小、ne[0] 变化最快。矩阵乘内核沿 ne[0] 连续读，' +
  '这正是 L5-03 里 <span class="cm" style="margin:0">ggml_vec_dot_*</span> 的前提。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '源码把三条不变量写在注释里。逐条推一遍：',
  '<span class="v">nb[0] = ggml_type_size(type)</span><br>注意：是 <b>type_size</b>（一"块"的字节数），不是元素大小。',
  '<span class="v">nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding</span><br>先按块算出一行有多少字节，再对齐。',
  '<span class="v">nb[i] = nb[i-1] * ne[i-1]</span><br>更高维就是逐维累乘 —— 所以是连续的、无孔洞的布局。',
  '这个布局让内核只需知道 <span class="k">nb[0]</span> 就能连续扫描一行，' +
  '也是"行大小"这个概念（ggml_row_size）的出处。'
];
let t = 700;
els.forEach((e, i) => { tl.at(t, () => {
  els.forEach((x, k) => x.style.opacity = k <= i ? '1' : '.28');
  msg.innerHTML = texts[Math.min(i, 2)];
}); t += 3000; });
tl.at(t, () => { msg.innerHTML = texts[3]; U.markLines(document, [3, 4]); });
tl.at(t + 3000, () => { msg.innerHTML = texts[4]; U.markLines(document, [1, 2]); });
tl.at(t + 6000, () => { msg.innerHTML = '一句话：<span class="k">nb[] 是 type 的函数</span> —— 换类型就换步长，换步长就换内核。' ; U.markLines(document, []); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L1-01 · 类型系统',
    title='<span class="hl-b">enum ggml_type</span>：精度谱系',
    sub='F32/F16/BF16 是浮点，Q*/IQ*/TQ* 是块量化，MXFP4/NVFP4 是新的 microscaling 格式。',
    caption='GGML_TYPE_COUNT = 43，且中间有编号空洞——注释说明了原因：删掉的类型不能复用编号。',
    src=SRC, parts=[(388, 434)], duration=22000,
    marks=[1, 13, 16, 41, 43],
    notes={2: '枚举从 0 开始，显式赋值 —— 这些数字会写进 GGUF 文件，不能随便变',
           43: '注释是硬约束：只能在末尾追加，否则旧 GGUF 文件会读错类型'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="bars" style="gap:6px"></div>
    <div class="col grow" id="notes" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const groups = [
  { l: '浮点 F32/F16/BF16/F64', n: 4, c: 'a', d: '每元素独立表示，无块' },
  { l: '传统块量化 Q4_0..Q8_K', n: 8, c: 'b', d: 'Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K..Q6_K' },
  { l: '重要性矩阵量化 IQ*', n: 8, c: 'c', d: 'IQ2_XXS..IQ4_XS，需要 imatrix 校准' },
  { l: '整数 I8/I16/I32/I64', n: 4, c: 'd', d: '量化中间表示与索引' },
  { l: '三值/新型格式 TQ*/MXFP4/NVFP4/Q1_0/Q2_0', n: 7, c: 'e', d: 'TQ1_0/TQ2_0 为三值；MXFP4/NVFP4 为 microscaling' }
];
const host = wrap.querySelector('#bars');
const bars = groups.map(g => {
  const b = U.bar(g.l + '  (' + g.n + ')', g.c);
  host.appendChild(b.el);
  return b;
});
const notes = wrap.querySelector('#notes');
notes.innerHTML =
  '<div class="card" style="border-left-color:var(--g)">' +
  '<div class="ct" style="color:var(--g)">编号只能追加，不能重排</div>' +
  '<div class="cb">源码注释：<span class="cm" style="margin:0">always add types at the end of the enum to keep backward compatibility</span>。' +
  '这些数字直接写进 GGUF，重排会让所有旧模型读错。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)">' +
  '<div class="ct" style="color:var(--d)">为什么有编号空洞？</div>' +
  '<div class="cb">源码里被注释掉的 <span class="cm" style="margin:0">GGML_TYPE_Q4_2 = 4</span> 等，' +
  '说明支持被移除了，但编号<b>不允许回收</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">嵌套在类型里的第二个维度</div>' +
  '<div class="cb">旧式的 <span class="cm" style="margin:0">Q4_0_4_4</span> 把 GPU 的 4x4 交错布局编进了类型号；' +
  '现在改由后端的 <b>repack</b> 处理（见 L5-04）。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '43 个类型，先按家族分五组看。',
  '<span class="v">F32 / F16 / BF16 / F64</span>：无块，<span class="k">blck_size = 1</span>。',
  '<span class="v">Q4_0 家族</span>：<span class="k">blck_size = 32</span>。一个块 = 1 个 scale + 32 个低位宽权重。',
  '<span class="v">IQ* 家族</span>：需要重要性矩阵（imatrix）校准，<span class="k">推理时反量化代价更高</span>，' +
  '但同码率下质量更好。',
  '<span class="v">MXFP4 / NVFP4</span>：microscaling 格式，块更小、带 E4M3 之类的小指数 scale。',
  'L1-04 会把这些块的内存布局逐个画出来 —— 那一课解释"类型号如何变成 nb[0]"。'
];
tl.at(700, () => { bars.forEach((b, i) => { b.fill.style.width = (18 + i * 8) + '%'; b.val.textContent = groups[i].n; }); msg.innerHTML = texts[0]; });
groups.forEach((g, i) => tl.at(3300 + i * 2900, () => {
  bars.forEach((b, k) => { b.el.style.opacity = k === i ? '1' : '.35'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(18200, () => { bars.forEach(b => { b.el.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L1-01 · 类型系统',
    title='三个度量函数：<span class="hl-a">blck_size</span> / <span class="hl-a">type_size</span> / <span class="hl-a">row_size</span>',
    sub='所有内核和分配器都靠这三个函数把"类型 + 形状"换算成字节数。',
    caption='ggml_type_sizef 已标记 GGML_DEPRECATED，注释要求改用 ggml_row_size()。',
    src=SRC, parts=[(759, 761)], duration=16000,
    marks=[0, 1, 2],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'ggml_blck_size(type)', m: 'int64_t', b: '一个块含多少元素。<br>F32 -> 1；Q4_0 -> 32；Q4_K -> 256。' },
  { c: 'b', t: 'ggml_type_size(type)', m: 'size_t', b: '一个块占多少字节。<br>源码注释：size in bytes for all elements in a block。' },
  { c: 'c', t: 'ggml_row_size(type, ne)', m: 'size_t', b: '一行（ne[0] 个元素）占多少字节。<br>内核扫描时用的就是它。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => U.card(d, { style: 'width:224px' }));
els.forEach(e => host.appendChild(e));
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '三个函数，一个换算链：',
  '<span class="v">blck_size</span>：<span class="k">元素 -> 块</span>。它把"多少个数"变成"多少块"。',
  '<span class="v">type_size</span>：<span class="k">块 -> 字节</span>。它和 blck_size 一起定义了类型的"密度"。',
  '<span class="v">row_size</span>：<span class="k">形状 -> 字节</span>。内核按行扫描，所以最常用的就是它。',
  '它们共同支撑了上一幕的 <span class="k">nb[] 公式</span>：<br><span class="v">nb[0] = type_size</span>，<span class="v">nb[1] ≈ row_size</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  U.markLines(document, [i]);
  msg.innerHTML = texts[i];
}));
tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [0, 1, 2]); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L1-01 · 视图',
    title='<span class="hl-e">view_src</span> / <span class="hl-e">view_offs</span>：不拥有数据的算子',
    sub='RESHAPE、VIEW、TRANSPOSE、PERMUTE 这些"算子"不产生新数据，只是换一个读法。',
    caption='L4-01 会看到：分配器正是靠 view_src 判断"这个张量不需要新内存"。',
    src=SRC, parts=[(706, 710)], duration=15000,
    marks=[1, 2, 4],
    notes={0: 'view_src 非 NULL 时，这个张量是"视图"——数据在别人那里',
           1: 'view_offs 是相对 view_src->data 的字节偏移'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia" style="gap:12px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const dia = wrap.querySelector('#dia');

const base = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
base.innerHTML = '<div class="ct" style="color:var(--a)">基张量 base</div>' +
  '<div class="cb">ne = [8, 4]，拥有 data 与 buffer<br>' +
  '<span class="cm" style="margin:0">view_src = NULL</span></div>' +
  '<div style="display:flex;gap:2px;margin-top:6px" id="cells"></div>';
const view = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--e)' });
view.innerHTML = '<div class="ct" style="color:var(--e)">视图 v = ggml_view_2d(...)</div>' +
  '<div class="cb">ne = [4, 2]，<b>不分配新内存</b><br>' +
  '<span class="cm" style="margin:0">view_src = base</span><br>' +
  '<span class="cm" style="margin:0">view_offs = 2 * 4 字节</span></div>';
dia.appendChild(base); dia.appendChild(U.arrow('->')); dia.appendChild(view);

const cells = base.querySelector('#cells');
const cs = [];
for (let i = 0; i < 8; i++) {
  const c = U.el('div', { style: 'width:22px;height:16px;border:1px solid var(--border);border-radius:2px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
  c.textContent = i;
  cells.appendChild(c); cs.push(c);
}

const msg = wrap.querySelector('#msg');
const texts = [
  '视图不复制数据：<span class="k">view_src 指向源张量，view_offs 是偏移</span>。',
  '视图自己的 <span class="v">data</span> 字段仍然会被算出来（= view_src->data + view_offs），' +
  '但内核读的还是同一块显存/内存。',
  '<span class="k">这就是 ggml 能做零拷贝 RESHAPE 的原因。</span>L4-01 的分配器会检查 view_src：' +
  '非空就跳过分配。',
  '代价：视图的 <span class="v">nb[]</span> 可能不再满足 <span class="v">nb[0] = type_size</span> 的默认约定 —— ' +
  '所以内核不能假设"连续"，必须用 <span class="v">nb[]</span> 走（见 L5-03）。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 2]); });
tl.at(3600, () => {
  [0, 1, 2, 3].forEach(i => { cs[i].style.background = 'rgba(88,166,255,.22)'; cs[i].style.borderColor = 'var(--a)'; });
  msg.innerHTML = texts[1];
});
tl.at(6800, () => {
  [2, 3, 4, 5].forEach(i => { cs[i].style.background = 'rgba(247,120,186,.22)'; cs[i].style.borderColor = 'var(--e)'; });
  msg.innerHTML = texts[2]; U.markLines(document, [0]);
});
tl.at(10200, () => { msg.innerHTML = texts[3]; U.markLines(document, [4]); });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L1-01 · 收束',
    title='把这一课压成一张表',
    sub='数据面 = 4 组字段。记住它们，后面 51 课都在读写这几组。',
    caption='下一课 L1-02 讲身份面：enum ggml_op 与参数个数表。',
    src=SRC, parts=[(685, 689)], duration=17000,
    marks=[0, 1, 2, 3, 4],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['字段组', '字段', '谁在读它', '在哪一课展开'],
  [['身份/精度', 'type, op, op_params, flags', '后端选 kernel、图打印', 'L1-02 / L5-01'],
   ['形状', 'ne[4]', '所有内核、分配器', '本课 / L4-01'],
   ['步长', 'nb[4]', '所有内核的数据访问', '本课 / L5-03'],
   ['位置', 'buffer, data, extra', '调度器、后端 buffer', 'L4-02 / L4-03'],
   ['连接', 'src[10]', '图遍历、拓扑排序', 'L1-03'],
   ['视图', 'view_src, view_offs', '分配器（跳过分配）', 'L1-01 / L4-01'],
   ['调试', 'name[64]', '报错定位、graph dump', 'L1-03']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '给定 <span class="mono">ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 512, 8)</span>，' +
  '<span class="mono">nb[0]</span> 和 <span class="mono">nb[1]</span> 各是多少？',
  'Q4_K 的 <span class="mono">blck_size = 256</span>、<span class="mono">type_size = 144</span> 字节。<br>' +
  '<span class="mono">nb[0] = 144</span>。<br>' +
  '<span class="mono">nb[1] = 144 * (512 / 256) = 288</span>（若 288 已满足 GGML_MEM_ALIGN 对齐则无额外 padding）。<br>' +
  '验算：这个张量共 512*8 = 4096 个元素 = 16 块，16 * 144 = 2304 字节 = 288 * 8。一致。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '六组字段，各自的读者不同：',
  '<span class="k">身份/精度</span> 组决定后端选哪个 kernel —— 见 L5-01 的大 switch。',
  '<span class="k">形状 + 步长</span> 组是所有数据访问的基础，也是量化块约定生效的地方。',
  '<span class="k">位置</span> 组把算法和内存解耦：同一个 op 可以跑在 CPU 内存或显存上。',
  '<span class="k">连接 + 视图</span> 组让"图"和"零拷贝"成为可能。',
  '记住一句话：<span class="v">ggml_tensor 是值类型，定长、可复制、可放进 arena</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(15500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、常量：结构体的形状被宏钉死',
    '`ggml_tensor` 的所有数组字段都是**定长**的，长度由这几个宏给出。'
    '这不是风格问题：定长意味着结构体可以 `memcpy`、可以整体放进 arena，'
    '也意味着**任何后端内核都可以假设秩不超过 4**，不必处理任意维。',
    src=SRC, parts=[(222, 226)], lang='c')

L.section(
    '二、★ struct ggml_tensor 逐字段',
    '这是"算子的数据面"的全部定义。按用途分成 7 组读：\n\n'
    '| 组 | 字段 | 含义 |\n|---|---|---|\n'
    '| 身份/精度 | `type` `op` `op_params` `flags` | 什么类型、什么运算、标量参数 |\n'
    '| 形状 | `ne[GGML_MAX_DIMS]` | 每维元素个数 |\n'
    '| 步长 | `nb[GGML_MAX_DIMS]` | 每维跨多少字节 |\n'
    '| 位置 | `buffer` `data` `extra` | 数据在哪个后端缓冲 |\n'
    '| 连接 | `src[GGML_MAX_SRC]` | 输入张量，即图的边 |\n'
    '| 视图 | `view_src` `view_offs` | 不拥有数据时的来源与偏移 |\n'
    '| 调试 | `name[GGML_MAX_NAME]` | 报错与图打印 |\n\n'
    '注意 `op_params` 的声明方式：源码注释写明是 "allocated as int32_t for alignment"，'
    '即 16 个 `int32_t`（`GGML_MAX_OP_PARAMS / sizeof(int32_t)`）。'
    '算子的标量参数（轴号、eps、scale）都按位塞进这 64 字节，详见下面的逐字引用。',
    src=SRC, parts=[(685, 717)], lang='c')

L.section(
    '三、★ ne[] / nb[]：三条不变量',
    '源码把约定直接写在注释里。三条规则合起来定义了 ggml 的内存布局：',
    src=SRC, parts=[(690, 694)], lang='c')

L.section(
    '四、enum ggml_type：编号即文件格式',
    '类型枚举的**数值**会写进 GGUF 文件，所以源码给了一条硬约束：只能往末尾追加。'
    '枚举里那些被注释掉的编号（如 `GGML_TYPE_Q4_2 = 4`）说明：支持可以移除，'
    '**编号不能回收**。',
    src=SRC, parts=[(388, 434)], lang='c')

L.section(
    '五、三个度量函数',
    '内核和分配器都不直接读 `type_size` 表，而是走这三个函数。'
    '注意 `ggml_type_sizef` 已经被标记 `GGML_DEPRECATED`，'
    '注释要求改用 `ggml_row_size()`。',
    src=SRC, parts=[(759, 761)], lang='c')

L.section(
    '六、view_src / view_offs：视图算子',
    '`RESHAPE` / `VIEW` / `TRANSPOSE` / `PERMUTE` 这类算子**不产生新数据**，'
    '只改变"怎么读"。源码在结构体里用两个字段表达这一点。',
    src=SRC, parts=[(706, 710)], lang='c')

L.footnote_add('本课只引用 `ggml/include/ggml.h` 一个文件，计入覆盖率。')
L.footnote_add('场景 4 的数值算例（Q4_0 / Q4_K 的 block size 与 type size）来自 L1-04 将要逐字展开的 '
               '`ggml/src/ggml-common.h`；本课不引用其源码，故不计入本课覆盖率。')

L.prereqs('无（本课是全套课件的起点）')

L.goal(
    '说出 `struct ggml_tensor` 里哪些字段决定一次内核调用的**形状**（对应验收点）；',
    '解释 `nb[0] = ggml_type_size(type)` 这条约定的来历，以及它与量化块大小的关系；',
    '判断一个张量是不是**视图**（`view_src != NULL`），并说出它对内存分配的含义；',
    '说明为什么 `enum ggml_type` 的编号**只能追加不能重排**。')

L.conclusion(
    '数据面 = 四组字段',
    '`struct ggml_tensor` 的 12 个字段按用途分成四组：\n\n'
    '| 组 | 字段 | 决定什么 |\n|---|---|---|\n'
    '| 身份/精度 | `type` `op` `op_params` `flags` | 后端选哪个 kernel |\n'
    '| 形状/步长 | `ne[4]` `nb[4]` | 内核怎么访问数据 |\n'
    '| 位置 | `buffer` `data` `extra` | 数据在哪个设备上 |\n'
    '| 连接/视图 | `src[10]` `view_src` `view_offs` | 图的结构与零拷贝 |')

L.conclusion(
    '★ nb[] 是 type 的函数',
    '三条不变量（源码注释里写死的）：\n\n'
    '```text\n'
    'nb[0] = ggml_type_size(type)\n'
    'nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding\n'
    'nb[i] = nb[i-1] * ne[i-1]\n'
    '```\n\n'
    '**换类型就换步长，换步长就换内核。** 这是后面 L1-04（块布局）、L5-03（向量化）、'
    'L5-04（repack）三课的共同前提。')

L.conclusion(
    '视图算子不拥有数据',
    '`RESHAPE` / `VIEW` / `TRANSPOSE` / `PERMUTE` 只改 `ne[]`、`nb[]` 与 '
    '`view_src`/`view_offs`，不复制数据。L4-01 的分配器据此跳过分配 —— 这是 ggml '
    '能在有限显存里跑大模型的原因之一。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
