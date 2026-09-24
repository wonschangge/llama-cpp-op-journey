#!/usr/bin/env python3
"""L6-09 · Metal 后端：Metal 内核 —— 课件 spec。

运行：python3 L6-gpu-backend/L6-09-metal-kernels/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC = 'ggml/src/ggml-metal/kernels/mul_mm.metal'

L = Lesson(
    id='L6-09',
    layer='L6 · GPU 后端执行',
    title='Metal 后端：Metal 内核',
    codecap='ggml/src/ggml-metal/kernels/（逐字引用）',
    nav={'prev': {'href': '../L6-08-metal-host-fusion/index.html',
                  'label': 'L6-08 Metal 主机端与图融合'},
         'next': {'href': '../../L7-npu-backend/L7-01-cann-ascend-npu/index.html',
                  'label': 'L7-01 CANN 后端'}},
)

L.note('**一句话**：Metal 后端的 `kernels/` 目录里是 23 个 `.metal` / `.h` 文件，共 13425 行；'
       '它们是这条流水线的最后一站 —— 主机端（L6-08）选好 pipeline，'
       '这里的内核才真正在 GPU 上算。')
L.note('这一课的主线是**一个问题**：Metal 怎么用同一份源码覆盖二十多种量化类型？'
       '答案不是"每种类型写一个内核文件"，而是 **C++ 模板 + host 侧显式实例化**。'
       '这正是它与 CUDA 的 `template-instances/`（L6-04）不同的地方。')
L.note('数字来自命令：`python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L6-09"{print $2}\'` '
       '给出 23 个文件；`wc -l` 给出 13425 行；`grep -c host_name` 给出 1017 处实例化。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L6-09 · 全局',
    title='23 个源文件、<span class="hl-a">8 个族</span>：内核层的全景',
    sub='每个 .metal 都从 #include "common.h" 开始；common.h 再把主机的参数头 ggml-metal-impl.h 拉进来。',
    caption='上游：L6-08 讲主机端怎么选 pipeline；本课只讲内核本身。',
    src='ggml/src/ggml-metal/kernels/common.h', parts=[(1, 23)], duration=22000,
    mark_src=[3, 13, 21, 23],
    notes_src={3: '主机端的 kargs 与 N_MM_* 常量都在这里，两边共享',
               21: 'FOR_UNROLL 把循环交给 clang 全展开，内核里到处在用',
               23: '一个 SIMD group = 32 个线程；后面所有 simd_sum / simd_max 都基于它'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const fams = [
  ['矩阵乘 matmul',       '2',  '4365', '245'],
  ['注意力与序列混合',    '4',  '3395', '567'],
  ['逐元素与形状',        '6',  '2337', '61'],
  ['量化与反量化',        '3',  '1477', '98'],
  ['归约 / 排序 / 三角',  '4',  '851',  '20'],
  ['归一化与 softmax',    '2',  '541',  '18'],
  ['位置编码 rope',       '1',  '333',  '8'],
  ['公共头 common.h',     '1',  '126',  '0']
];
const t = U.table(['族', '文件', '行数', 'host_name'], fams, { monoCols: [1, 2, 3] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');
const msg = wrap.querySelector('#msg');
const texts = [
  '23 个文件按"算什么"分成 8 个族。最大的一族是矩阵乘（2 个文件 4365 行）。',
  '<span class="v">mul_mm.metal + mul_mv.metal</span>：矩阵乘的两条路，也是本课的主角（第 2 到第 6 幕）。',
  '<span class="v">fa.metal</span> 一个文件就有 2476 行、559 个实例化 —— 头维度每变一次就要多一组实例。',
  '逐元素与形状族最杂：unary / binbcast / misc / conv / pool / upscale，写法最接近普通 GPU kernel。',
  '归约、排序、三角求解：都在线程组内做两级归约，靠 threadgroup 内存跨 SIMD group 汇总。',
  '归一化与 softmax 只有 541 行，却是最典型的 SIMD-group 归约范式（第 8 幕之外的源码小节）。',
  'rope 一个文件：把 YaRN 的数学直接搬进内核。',
  '<span class="k">合计 23 个文件 / 13425 行 / 1017 处 host_name 实例化。</span>'
];
rows.forEach((r, i) => tl.at(700 + i * 2500, () => {
  rows.forEach(x => { x.className = ''; });
  r.className = 'on';
  msg.innerHTML = texts[i];
}));
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L6-09 · 核心',
    title='★ <span class="hl-a">模板参数就是量化类型</span>：一份源码，二十多种 block_q',
    sub='kernel_mul_mm 的模板参数里有 block_q（量化块类型）、nl（每 16 个权重跨几个块）和一个反量化函数指针。',
    caption='第 13 行是条件编译：这条声明只在支持 simdgroup 矩阵乘的设备上生效；否则走文件下半部分的回退实现。',
    src=SRC, parts=[(12, 27)], duration=20000,
    mark_src=[12, 14, 17, 18, 19],
    notes_src={12: '这一行注释就是 nl 的含义：一个 block_q 装 16*nl 个权重',
               17: 'block_q 是块类型，nl 是块内 16 元素的分段数',
               18: 'T0 / T1 是 A、B 在设备内存里的元素类型；SA / SB 是线程组内存类型'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'block_q', m: 'typename block_q', b: '量化块类型：<br>block_q4_0 / block_q4_K / block_iq4_xs ...' },
  { c: 'b', t: 'nl', m: 'short nl', b: '一个块含 16*nl 个权重：<br>q4_0 -> 2（=32），Q4_K -> QK_NL=16（=256）' },
  { c: 'c', t: 'dequantize_func', m: 'void (*)(device const block_q *, short, thread SA_4x4 &)', b: '反量化函数指针：<br>dequantize_q4_0 / dequantize_q4_K ...' },
  { c: 'd', t: 'SA / SB', m: 'typename SA, typename SB', b: '线程组内存里的元素类型：<br>half / bfloat / float' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { w: '163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '模板参数一共 12 个。其中三个决定了"支持哪种量化"：',
  '<span class="v">block_q</span>：换一个块类型，就换一种量化格式。内核正文只按块读、按块反量化。',
  '<span class="v">nl</span>：告诉内核"16 个权重"在一个块里占第几段。<br>q4_0 是 2，K-quant 是 16 —— 见第 3 幕的实例化清单。',
  '<span class="v">dequantize_func</span>：把反量化做成函数指针参数。<br>于是内核正文完全不需要知道 q4_0 和 iq4_xs 有什么区别。',
  '非量化类型也走同一个槽：<span class="k">dequantize_f32</span> 只是把 float4x4 原样拷进寄存器（源码注释写着"this is not dequantizing"）。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(15000, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L6-09 · 实例化',
    title='实例化清单 = <span class="hl-c">量化支持矩阵</span>',
    sub='typedef decltype(...) 先固定住与类型无关的参数，之后每一行 host_name 就是一个可被主机端取用的 pipeline。',
    caption='回顾 L6-04：CUDA 把实例化写进 template-instances/ 的独立文件；Metal 直接写在同一份 .metal 的末尾。',
    src=SRC, parts=[(849, 866)], duration=24000,
    mark_src=[851, 853, 855, 858, 860, 866],
    notes_src={851: '先给一组"默认"参数定型：这就是 mul_mm_t',
               853: 'f32 走的就是 dequantize_f32 —— 同一个模板槽'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const rows = [
  ['f32',  '1',     'dequantize_f32',  '不反量化，只把 float4x4 拷进寄存器'],
  ['f16',  '1',     'dequantize_f16',  '同上，元素类型换成 half'],
  ['bf16', '1',     'dequantize_bf16', '被 GGML_METAL_HAS_BF16 包住'],
  ['q1_0', '8',     'dequantize_q1_0', '16*8 = 128 个权重一块'],
  ['q2_0', '4',     'dequantize_q2_0', '16*4 = 64 个权重一块'],
  ['q4_0', '2',     'dequantize_q4_0', '16*2 = 32 个权重一块'],
  ['q2_K', 'QK_NL', 'dequantize_q2_K', '16*16 = 256 个权重一块']
];
const t = U.table(['src0 类型', 'nl', '反量化函数', '含义'], rows, { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const trs = t.body.querySelectorAll('tr');
const msg = wrap.querySelector('#msg');
const texts = [
  '先看被引用的这几行：一张"类型 -> nl -> 反量化函数"的对照表。',
  '<span class="v">f32 / f16 / bf16</span>：nl = 1，反量化函数只是把向量原样搬进寄存器。',
  '<span class="v">q1_0 / q2_0</span>：nl = 8 / 4，即一个块 128 / 64 个权重。',
  '<span class="v">q4_0</span>：nl = 2，一个块 32 个权重 —— 与 L1-04 讲的 Q4_0 块布局一致。',
  '<span class="v">q2_K</span>：nl = QK_NL = 16，一个块 256 个权重，即 QK_K。',
  '同一个 mul_mm.metal 里共有 <span class="k">23 种 block 类型</span>、<span class="k">111 处 host_name</span>（其中 51 处非 MoE）。',
  '主机端按 tensor 的 type 取名字，取到哪个 host_name 就用哪个 pipeline —— 名字是两边的接口。'
];
trs.forEach((r, i) => tl.at(700 + i * 2900, () => {
  trs.forEach(x => { x.className = ''; });
  r.className = 'on';
  msg.innerHTML = texts[i];
}));
tl.at(21500, () => {
  trs.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[6];
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L6-09 · 矩阵乘',
    title='mul_mm 的 tile：<span class="hl-b">64 x 32</span> 的输出块',
    sub='每个线程组算输出矩阵的一块；A 先反量化进线程组内存，再用 simdgroup 矩阵乘累加。',
    caption='这是不支持 simdgroup 矩阵乘时的回退实现；上半部分（tensor 路径）的 tile 由 N_MM_* 宏给出。',
    src=SRC, parts=[(160, 178)], duration=18000,
    mark_src=[161, 162, 164, 165, 175, 176, 177],
    notes_src={164: 'NR0：沿输出行方向，一个线程组算 64 行',
               165: 'NR1：沿输出列方向，一个线程组算 32 列',
               175: '源码注释直接把 tile 形状写出来了',
               176: '边界块收窄：行方向取剩余的行数'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:4px">一个线程组负责的输出块</div>';
const grid = U.el('div', { style: 'display:flex;flex-wrap:wrap;gap:2px;width:196px' });
for (let i = 0; i < 8; i++) {
  const c = U.el('div', { style: 'width:22px;height:15px;border:1px solid var(--border);border-radius:2px;background:rgba(88,166,255,.20)' });
  grid.appendChild(c);
}
left.appendChild(grid);
left.appendChild(U.el('div', { class: 'cb', style: 'font-size:9.5px;color:var(--dim)', text: '8 x 8 = 64 格，示意 NR0=64 / NR1=32 的 tile 形状' }));

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">两阶段流水</div>' +
  '<div class="cb"><b>PHASE 1</b>：把 A 的一块反量化进线程组内存 sa（第 52 行）。<br>' +
  '<b>PHASE 2</b>：simdgroup 矩阵乘，把结果累加进 cT。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">K 方向分块</div>' +
  '<div class="cb">外层循环按 NK=32 走 K 维，每一轮都要一次 ' +
  '<span class="cm" style="margin:0">threadgroup_barrier</span> —— ' +
  '这就是 tile 方案的代价。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '先看 tile 的形状：NR0 = 64 行、NR1 = 32 列、NK = 32。',
  '<span class="v">r0 = tgpig.y * NR0</span>，<span class="v">r1 = tgpig.x * NR1</span>：<br>线程组在网格里的坐标直接乘出它在输出矩阵里的位置。',
  'A 的反量化结果放进线程组内存 <span class="v">sa</span>，B 的缓冲是 <span class="v">sb</span>（偏移 4096 字节）。',
  '关键问题：<span class="k">列方向固定 32 列</span>。<br>如果输出只有 1 列（一次只算一个 token），32 列里 31 列是空转的。',
  '第 175 到 177 行的收窄只解决"边界块"，解决不了"本来就只有 1 列"。<br>这就是下一幕 mul_mv 存在的理由。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [4, 6]); });
tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [12, 13]); });
tl.at(7000, () => { msg.innerHTML = texts[2]; U.markLines(document, [1, 2]); });
tl.at(10500, () => { msg.innerHTML = texts[3]; U.markLines(document, [4, 6]); });
tl.at(14500, () => { msg.innerHTML = texts[4]; U.markLines(document, [17, 19, 21]); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L6-09 · 矩阵乘',
    title='mul_mv 的 <span class="hl-e">per-row</span> 结构：一行 src1 对 NR0 行 src0',
    sub='每个线程组只认 src1 的一行；这一行被缓存进寄存器，在 NR0 行 src0 之间反复复用。',
    caption='NR0 与线程组里的 SIMD group 数由主机端给出（N_R0_* / N_SG_* 宏，见 L6-08）。',
    src='ggml/src/ggml-metal/kernels/mul_mv.metal', parts=[(228, 255)], duration=20000,
    mark_src=[233, 235, 237, 250, 251, 254],
    notes_src={237: 'r1 = tgpig.y：src1 的一行 = 线程组网格的一个 y 坐标',
               250: 'ax[NR0]：一次抓下 NR0 行 src0 的块指针',
               254: '每一行 src0 的起始地址按 nb01 步进'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const dia = wrap.querySelector('#dia');
const a = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--e)' });
a.innerHTML = '<div class="ct" style="color:var(--e)">src0：NR0 行量化权重</div>' +
  '<div class="cb">每行 = nb 个 block_q<br>' +
  '<span class="cm" style="margin:0">ax[row] = src0 + (r0+row)*nb01</span></div>';
const b = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
b.innerHTML = '<div class="ct" style="color:var(--b)">src1：1 行输入向量</div>' +
  '<div class="cb">缓存在寄存器 <span class="cm" style="margin:0">float yl[16]</span><br>' +
  '被 NR0 行 src0 复用</div>';
dia.appendChild(a); dia.appendChild(U.arrow('x')); dia.appendChild(b);

const msg = wrap.querySelector('#msg');
const texts = [
  'mul_mv 的网格是"行 x 行"：<span class="v">r0</span> 是 src0 的行，<span class="v">r1</span> 是 src1 的行。',
  '<span class="v">r0 = (tgpig.x*NSG + sgitg)*NR0</span>：<br>一个线程组里的每个 SIMD group 再分 NR0 行 src0。',
  '<span class="v">r1 = tgpig.y</span>：src1 的一行 = 一个 y 坐标。<br><span class="k">线程组粒度就是"一行 src1"</span> —— 没有列方向的空转。',
  '<span class="v">ax[NR0]</span>：把 NR0 行 src0 的块指针一次性算好，循环里只做 <span class="m">ax[row] + ib</span>。',
  '内层循环（第 265 到 291 行，见下面源码小节的第二段）：每次取 16 个 float 存进 ' +
  '<span class="m">yl[16]</span>，再对 NR0 行各做一次 <span class="m">block_q_n_dot_y</span> —— ' +
  '<span class="k">一份 src1 数据，NR0 次复用</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [7, 9]); });
tl.at(3700, () => { msg.innerHTML = texts[1]; U.markLines(document, [7]); });
tl.at(7400, () => { msg.innerHTML = texts[2]; U.markLines(document, [9]); });
tl.at(11000, () => { msg.innerHTML = texts[3]; U.markLines(document, [23, 25, 26, 28]); });
tl.at(15000, () => { msg.innerHTML = texts[4]; U.markLines(document, [7, 9]); });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L6-09 · 验收点',
    title='★ <span class="hl-a">src1 有几行</span>决定走哪条路',
    sub='mul_mv 的 ext 变体把 r1ptg 行 src1 绑进一个线程组；实例化里 r1ptg 只取 2 到 5 —— 它就是为"少数几行"设计的。',
    caption='主机端按 ne11 选 r1ptg、并决定走 ext 还是 mul_mm —— 那部分在 ggml-metal-ops.cpp，见 L6-08。',
    src='ggml/src/ggml-metal/kernels/mul_mv.metal', parts=[(602, 640)], duration=22000,
    mark_src=[602, 614, 622, 623, 634, 637, 640],
    notes_src={614: 'chpt = 4：每个线程一次处理 4 个 float4 块',
               623: 'i11 = tgpig.y * r1ptg：线程组的 y 坐标以 r1ptg 为步长',
               637: 'r1ptg 行 src1 的指针一次抓进寄存器数组'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const rows = [
  ['ne11 == 1',        'mul_mv',     '线程组只算 1 列，没有空转（第 237 行 r1 = tgpig.y）'],
  ['ne11 = 2 到 5',    'mul_mv_ext', '一个线程组吃下 r1ptg 行 src1（第 623 行），r1ptg 只实例化了 2/3/4/5'],
  ['ne11 大（批量）',  'mul_mm',     'tile 是 64 x 32，反量化后的 A 块被 32 列复用（第 164、165 行）']
];
const t = U.table(['src1 的行数（ne11）', '更优的内核', '源码依据'], rows, { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);
const trs = t.body.querySelectorAll('tr');
const msg = wrap.querySelector('#msg');
const texts = [
  '把这一课的问题压成一张表：<span class="k">形状决定内核</span>。',
  '<span class="v">ne11 == 1</span>（逐 token 解码）：mul_mv 每个线程组只服务 src1 的一行，没有一列是白算的。',
  '<span class="v">ne11 只有几行</span>：走 ext 变体，一个线程组同时算 r1ptg 行 —— 把"少数几行"也凑满一个线程组。',
  '<span class="v">ne11 很大</span>（prompt 批量）：mul_mm 更优 —— 它把 A 反量化进线程组内存后，整块 tile 一起用。',
  '注意这两条路的<span class="k">取舍是对称的</span>：mul_mv 用"每行一次点积"换掉列方向空转；mul_mm 用"固定 32 列"换 A 块的复用。'
];
trs.forEach((r, i) => tl.at(700 + i * 3300, () => {
  trs.forEach(x => { x.className = ''; });
  r.className = 'on';
  msg.innerHTML = texts[i];
}));
tl.at(15500, () => {
  trs.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[4];
});
tl.at(18500, () => { msg.innerHTML = texts[4] + '<br>验收点：能说出 mul_mv 在 <span class="v">src1 只有一行/少数几行</span>时更优，' +
  '因为 mul_mm 的 tile 列方向是固定的 32 列。'; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L6-09 · 注意力',
    title='flash attention：<span class="hl-d">DK / DV</span> 也是模板参数',
    sub='kernel_flash_attn_ext 的头维度是编译期常量；host_name 里 dk128_dv128 这样的后缀就是一对 (DK, DV)。',
    caption='fa.metal 共 559 处 host_name、16 种 (DK, DV) 组合 —— 头维度组合爆炸是模板方案的主要代价。',
    src='ggml/src/ggml-metal/kernels/fa.metal', parts=[(199, 229)], duration=18000,
    mark_src=[199, 218, 224, 225, 226, 227, 228],
    notes_src={199: '源码直接给出 flash-attention 的原始论文链接',
               218: 'kd4x4_t / nl_k / deq_k：K 在设备内存里的块类型、分段数与反量化函数',
               224: 'DK / DV：K 与 V 的头维度，都是 short 模板参数',
               227: 'C：每个线程组缓存多少条 KV'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'DK / DV', m: 'short DK, short DV', b: 'K 与 V 的头维度。<br>host_name 后缀 dk128_dv128 就是它' },
  { c: 'b', t: 'Q / C', m: 'short Q, short C', b: 'Q：每个线程组处理几个 query<br>C：缓存几条 KV' },
  { c: 'c', t: 'NSG', m: 'short NSG', b: '线程组里的 SIMD group 数，<br>实现里只实例化了 4 和 8' },
  { c: 'd', t: 'deq_k / deq_v', m: 'void (*)(device const kd4x4_t *, short, thread k4x4_t &)', b: 'K / V 的反量化函数指针 ——<br>与 mul_mm 完全同一个套路' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { w: '163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  'flash attention 的内核有 28 个模板参数，比 mul_mm 还长。',
  '<span class="v">DK / DV</span>：头维度进模板，是因为线程组内存的布局要按它算常量。<br>代价：每换一个头维度就多一组实例。',
  '<span class="v">Q / C / NSG</span>：分块策略也进模板。<br>第 894 行的源码注释说得很直白：<span class="m">this is quite ugly ... for now keep them as template</span>。',
  '<span class="v">deq_k / deq_v</span>：KV 量化的支持方式与 mul_mm 一致 —— 换函数指针，不换内核正文。',
  'KV 量化只覆盖 <span class="k">q4_0 / q4_1 / q5_0 / q5_1 / q8_0</span> 五种：先由 <span class="m">kernel_flash_attn_ext_kv_f16</span> 反量化成 F16，再跑 F16 主内核。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(15000, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L6-09 · 共享定义',
    title='同一个 <span class="hl-c">block_q</span>：CPU 与 Metal 共用一份定义',
    sub='dequantize.h 用 GGML_COMMON_DECL_METAL 打开 ggml-common.h 的 Metal 分支 —— 块结构体不是 Metal 特有的。',
    caption='回顾 L1-04：块的字节布局（scale + 低位宽权重）在那里逐字展开；本课只讲 Metal 怎么用它。',
    src='ggml/src/ggml-metal/kernels/dequantize.h', parts=[(1, 13)], duration=18000,
    mark_src=[5, 6, 10, 13],
    notes_src={5: '打开 ggml-common.h 的 Metal 分支',
               10: 'block_q4_0 / block_q4_K 这些结构体就是从这一行来的',
               13: 'QK_NL = 16，被 mul_mm 与 get_rows_q 的实例化共用'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip">ggml-common.h</span><span class="arrow">-></span>
    <span class="chip a">GGML_COMMON_DECL_METAL</span><span class="arrow">-></span>
    <span class="chip b">block_q4_0 / block_q4_K</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'CPU 侧', b: 'ggml-common.h 的 C / CPP 分支<br>同一个 block_q4_0' },
  { c: 'b', t: 'CUDA 侧', b: 'GGML_COMMON_DECL_CUDA 分支<br>同一个 block_q4_0' },
  { c: 'c', t: 'Metal 侧', b: 'GGML_COMMON_DECL_METAL 分支<br>还是同一个 block_q4_0' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { w: '214px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '量化块的<b>内存布局</b>只有一份定义，各后端只是各取所需地把它 include 进来。',
  '<span class="v">#define GGML_COMMON_DECL_METAL</span> 选分支；<span class="v">GGML_COMMON_IMPL_METAL</span> 还要实现部分。',
  '所以 <span class="k">Metal 不需要重新定义 block_q4_0</span>：它和 CPU 看到的是同一段字节。',
  '第 7 到 11 行是双路径：嵌入库时用 <span class="m">__embed_ggml-common.h__</span>，否则直接 include。',
  '<span class="v">QK_NL = 16</span> 也定义在这里，注释写明它被 mul_mm 与 get_rows_q 的实例化共用 —— 第 2 幕那个 nl 的取值就来自它。'
];
els.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13500, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[3];
});
tl.at(15500, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L6-09 · 收束',
    title='把这一课压成一张表',
    sub='内核在哪、怎么覆盖量化类型、怎么用 SIMD group、什么时候换另一条路 —— 四个问题。',
    caption='下一课 L7-01 换一个后端：CANN 怎么把 ggml op 映射到 Ascend NPU 的 ACL 算子。',
    src=SRC, parts=[(44, 56)], duration=24000,
    mark_src=[44, 45, 48, 49, 52, 56],
    notes_src={44: 'NRB：沿输出列方向的 tile 宽度，由三个宏相乘得到',
               45: 'NRA：沿输出行方向的 tile 高度',
               52: 'A 的反量化结果只占线程组内存 —— B 直接从设备内存读'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', '答案', '源码位置'],
  [['内核在哪', 'kernels/ 下 23 个文件、8 个族、13425 行', 'common.h:1-23'],
   ['怎么覆盖量化', 'C++ 模板 + host 侧 host_name 实例化（23 种 block_q）', 'mul_mm.metal:851-880'],
   ['怎么用 SIMD group', 'N_SIMDWIDTH = 32，simd_sum / simd_max + threadgroup 内存两级归约', 'common.h:23 / norm.metal:42'],
   ['什么时候换路', 'ne11 小走 mul_mv / ext，ne11 大走 mul_mm（tile 固定 32 列）', 'mul_mv.metal:237,623 / mul_mm.metal:165'],
   ['块定义从哪来', 'GGML_COMMON_DECL_METAL 打开 ggml-common.h，与 CPU 共用', 'dequantize.h:5-10']],
  { monoCols: [2] });
wrap.querySelector('#tbl').appendChild(t.el);
const trs = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '一次只解码一个 token（src1 只有 1 行）时，该走 <span class="mono">mul_mv</span> 还是 ' +
  '<span class="mono">mul_mm</span>？为什么？',
  '走 <span class="mono">mul_mv</span>。<br>' +
  '<b>1.</b> <span class="mono">mul_mm</span> 的线程组 tile 是 64 行 x 32 列' +
  '（<span class="mono">mul_mm.metal:164-165</span>，第 175 行的注释也写着 64x32）；' +
  '输出只有 1 列时，32 列里 31 列白算。<br>' +
  '<b>2.</b> <span class="mono">mul_mv</span> 的网格是"src0 的行 x src1 的行"：' +
  '<span class="mono">r1 = tgpig.y</span>（<span class="mono">mul_mv.metal:237</span>），' +
  '一个线程组只服务 src1 的一行，列方向没有空转。<br>' +
  '<b>3.</b> 它把这一行的 16 个 float 缓存在寄存器 ' +
  '<span class="mono">float yl[16]</span>（<span class="mono">mul_mv.metal:265</span>）里，' +
  '在 NR0 行 src0 之间复用（<span class="mono">mul_mv.metal:285-287</span>）。<br>' +
  '<b>反过来</b>：src1 有几十上百行（prompt 批量）时 <span class="mono">mul_mm</span> 更优 —— ' +
  'A 的反量化结果进了线程组内存（<span class="mono">mul_mm.metal:52</span>），' +
  '整块 tile 一起复用。<br>' +
  '中间地带（src1 只有 2 到 5 行）走 <span class="mono">mul_mv_ext</span>：' +
  '<span class="mono">i11 = tgpig.y * r1ptg</span>（<span class="mono">mul_mv.metal:623</span>），' +
  '一个线程组同时算 r1ptg 行。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '五个问题、五个答案。',
  '<span class="k">内核在哪</span>：23 个文件 8 个族；公共头 common.h 把主机参数头拉进来。',
  '<span class="k">怎么覆盖量化</span>：模板参数 block_q / nl / dequantize_func，加 host_name 实例化 —— 与 CUDA 的 template-instances 是两种解法。',
  '<span class="k">怎么用 SIMD group</span>：32 线程一组，先 simd_sum 组内归约，再用 threadgroup 内存跨组汇总。',
  '<span class="k">什么时候换路</span>：src1 的行数决定内核 —— 这是本课的验收点。',
  '一句话：<span class="v">Metal 用一份 .metal 加一张实例化清单，换掉 CUDA 的整个 template-instances 目录</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
trs.forEach((r, i) => tl.at(2400 + i * 3000, () => {
  trs.forEach(x => { x.className = ''; });
  r.className = 'on';
  msg.innerHTML = texts[i + 1];
}));
tl.at(19000, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、kernels 目录全景：23 个文件、8 个族',
    '本课覆盖的清单由计划矩阵给出，不是凭印象划的：\n\n'
    '```text\n'
    'python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L6-09"{print $2}\'\n'
    '```\n\n'
    '共 **23 个文件**（`wc -l` 合计 13425 行），按"算什么"分成 8 个族：\n\n'
    '| 族 | 文件数 | 行数 | host_name 实例化数 | 文件 |\n|---|---|---|---|---|\n'
    '| 矩阵乘 | 2 | 4365 | 245 | `mul_mm.metal` `mul_mv.metal` |\n'
    '| 注意力与序列混合 | 4 | 3395 | 567 | `fa.metal` `ssm.metal` `wkv.metal` `gated_delta_net.metal` |\n'
    '| 逐元素与形状 | 6 | 2337 | 61 | `unary.metal` `binbcast.metal` `misc.metal` `conv.metal` `pool.metal` `upscale.metal` |\n'
    '| 量化与反量化 | 3 | 1477 | 98 | `dequantize.h` `quantize.h` `quantize.metal` |\n'
    '| 归约 / 排序 / 三角 | 4 | 851 | 20 | `reduce.metal` `argsort.metal` `tri.metal` `solve_tri.metal` |\n'
    '| 归一化与 softmax | 2 | 541 | 18 | `norm.metal` `softmax.metal` |\n'
    '| 位置编码 | 1 | 333 | 8 | `rope.metal` |\n'
    '| 公共头 | 1 | 126 | 0 | `common.h` |\n'
    '| **合计** | **23** | **13425** | **1017** | |\n\n'
    '三个计数都能用命令复现：`wc -l`（行数）、`grep -c host_name`（实例化数）、'
    '`python3 tools/plan_matrix.py --files`（清单）。')

L.section(
    '二、common.h：所有内核共享的前言',
    '每个 `.metal` 的第一行都是 `#include "common.h"`。它做四件事：把**主机端的参数头**'
    '`ggml-metal-impl.h` 拉进来（内核与主机共享同一份 kargs 结构体声明）、'
    '打开 `metal_stdlib`、定义几个小宏（`MAX` / `MIN` / `SWAP` / `PAD2` / `FOR_UNROLL`）、'
    '以及钉死 `N_SIMDWIDTH = 32`。\n\n'
    '第 23 行的注释是后面所有归约代码的前提：**一个 SIMD group 假设是 32 个线程**。'
    '`simd_sum` / `simd_max` / `simd_shuffle_down` 全都建立在这个宽度上。',
    src='ggml/src/ggml-metal/kernels/common.h', parts=[(1, 23)], lang='cpp')

L.section(
    '三、mul_mm 的 tensor 路径：tA / tB / matmul2d',
    '同一个文件的上半部分（`#ifdef GGML_METAL_HAS_TENSOR`）走的是另一条实现：'
    '用 `metal_tensor` 与 `mpp::tensor_ops::matmul2d` 把矩阵乘交给硬件的张量单元。\n\n'
    '注意两处细节：**A 用线程组内存包成 tensor**（`tA`，因为 A 要先反量化），'
    '而 **B 直接从设备内存包**（`tB`，连 stride 都是按元素算的）；'
    'K 维被声明成 `dynamic_extent`，源码注释解释了原因 —— 静态 K tile 会在 '
    '`K % N_MM_NK_TOTAL != 0` 时越界读 src1。',
    src='ggml/src/ggml-metal/kernels/mul_mm.metal', parts=[(59, 74)], lang='cpp')

L.section(
    '四、mul_mv：per-row 的内层循环与 ext 的分发器',
    '`mul_mv.metal` 里有两套矩阵向量内核，这里各引一段。\n\n'
    '**第一段是 per-row 的内层循环**（第 265 到 291 行）：'
    '`yl[16]` 是 src1 那一行的寄存器缓存，外层按 16 个元素推进，'
    '内层 `FOR_UNROLL (short row = 0; row < NR0; row++)` 对 NR0 行 src0 各做一次 '
    '`block_q_n_dot_y` —— **一份 src1 数据被 NR0 行复用**。'
    '注意 `yl[i+1] = yb[i+1]/256.f` 这类预缩放：它把 q4_0 的高低 4 位共用同一个 scale，'
    '省掉了一次乘法。\n\n'
    '**第二段是 ext 的分发器**（第 811 到 838 行）：'
    '真正的实现在 `kernel_mul_mv_ext_q4_f32_impl` 里，分发器只负责把"每块的元素数"'
    '换算成"float4 块数"（`epb/4`）或 "float4x4 块数"（`epb/16`），'
    '源码注释说明它"needed for compile-time nxpsg"。',
    src='ggml/src/ggml-metal/kernels/mul_mv.metal',
    parts=[(265, 291), (811, 838)], lang='cpp')

L.section(
    '五、dequantize.h：同一个函数槽，量化与非量化都填',
    '模板参数 `dequantize_func` 对非量化类型也要有个实现，否则模板没法实例化。'
    '`dequantize_f32` / `dequantize_f16` 就是这样的"占位"实现 —— '
    '源码注释写得很清楚：**this is not dequantizing - we are simply fitting the template**。',
    src='ggml/src/ggml-metal/kernels/dequantize.h', parts=[(14, 24)], lang='cpp')

L.section(
    '六、dequantize.h：dequantize_q4_0 逐块展开',
    '真正干活的反量化函数长这样：从一个 `block_q4_0` 里取 4-bit 量化值，'
    '乘上 scale 再减掉零点偏移（`md = -8 * xb->d`，对应 4-bit 的无符号偏移 8）。'
    '`il` 决定取低 4 位还是高 4 位，返回一个 4x4 的寄存器块。\n\n'
    '这正是 L1-04 讲过的块布局在 Metal 侧的样子：**同一个 `block_q4_0`，同一段字节**。',
    src='ggml/src/ggml-metal/kernels/dequantize.h', parts=[(131, 148)], lang='cpp')

L.section(
    '七、fa.metal：KV 量化先反量化成 F16',
    'flash attention 的主内核要求 K / V 是 F16。量化 KV 缓存的走法是：'
    '先用一个**独立的、每线程一个块**的内核把 K / V 反量化成连续的 F16，再跑 F16 主内核。'
    '源码注释写明它"dispatched separately for K and V"。\n\n'
    '这里能直接看到 Metal 支持哪几种 KV 量化：`q4_0` / `q4_1` / `q5_0` / `q5_1` / `q8_0`，'
    '五种都只是同一个模板换参数。',
    src='ggml/src/ggml-metal/kernels/fa.metal', parts=[(4, 45)], lang='cpp')

L.section(
    '八、rope.metal：YaRN 的数学与 rope_norm 内核',
    'rope 的内核先算 YaRN 的修正维度（`rope_yarn_corr_dims`），再按位置取 theta、'
    '算出 cos / sin，最后做二维旋转。`FC_rope_is_back` 是函数常量 —— 反向时把 sin 取负。\n\n'
    '注意内核的循环步长是 `2*tptg.x`：**一个线程负责一对元素**（`i0` 与 `i0+1`），'
    '这正是 rope 的旋转是成对进行的直接体现。',
    src='ggml/src/ggml-metal/kernels/rope.metal', parts=[(47, 77)], lang='cpp')

L.section(
    '九、norm.metal：两级 SIMD-group 归约',
    '归一化要算一整行的均值与方差，而一行有 `ne00` 个元素、线程组只有有限线程。'
    '写法是标准的两级归约：**先 `simd_sum` 在 SIMD group 内归约**，'
    '**再用 `threadgroup float *` 把各组的和汇总**，然后再 `simd_sum` 一次。\n\n'
    '模板参数 `F` 是融合开关（1 = 只有 norm，2 = norm+mul，3 = norm+mul+add）—— '
    '回顾 L6-08：主机端的图融合最终就是选一个 `F` 值。',
    src='ggml/src/ggml-metal/kernels/norm.metal', parts=[(8, 53)], lang='cpp')

L.section(
    '十、softmax.metal：同一个归约范式，两次',
    'softmax 要在一行里先求 max 再求 sum，所以把两级归约写了两遍：'
    '`simd_max` + 线程组内存，然后 `simd_sum` + 线程组内存。\n\n'
    '第 50 行的 `if (tptg.x > N_SIMDWIDTH)` 是这段代码的关键：'
    '**只有当线程组比一个 SIMD group 宽时才需要跨组汇总**，否则纯寄存器归约就够了。'
    '第 75 行的 `threadgroup_barrier(mem_flags::mem_none)` 还带着一条注释，'
    '说明它是为了修一个测试失败而加的。',
    src='ggml/src/ggml-metal/kernels/softmax.metal', parts=[(41, 80)], lang='cpp')

L.section(
    '十一、quantize.h：把 float 装进 block',
    '反量化是"块 -> float"，量化是反过来的"float -> 块"。`quantize.h` 是后者的实现集合，'
    '每个函数拿一个 `device const float *` 和一个 `device block_q &`。\n\n'
    '注意 `quantize_q1_0` 的写法：scale 取绝对值的平均，然后**只存符号位**；'
    '`quantize_q2_0` 取绝对值最大值当 scale。它们与 `dequantize.h` 里的对应函数是一对逆运算。',
    src='ggml/src/ggml-metal/kernels/quantize.h', parts=[(3, 26)], lang='cpp')

L.section(
    '十二、quantize.metal：cpy 与量化/反量化',
    '`quantize.metal` 里不只是量化，还有张量拷贝（`kernel_cpy_t_t`）。'
    '被引用的这一段是**反量化式拷贝**：从 `block_q` 读、写出 `T4x4`。\n\n'
    '它的模板参数与 `mul_mm` 是同一组：`block_q` / `nl` / `dequantize_func`。'
    '这说明"换类型不换正文"的做法在整个 kernels 目录里是一致的约定，不是矩阵乘的特例。',
    src='ggml/src/ggml-metal/kernels/quantize.metal', parts=[(104, 141)], lang='cpp')

L.section(
    '十三、unary.metal：一元算子靠函数常量分流',
    '`unary` 覆盖 GELU / SILU / EXP / NEG 等一大票一元算子。它不为每个算子写一个内核，'
    '而是用函数常量 `FC_unary_op` 在一个内核里分流。\n\n'
    '`FC_unary_cnt` 则区分"连续张量"与"有步长的张量"两条寻址路径 —— '
    '连续时可以直接按线性下标算，不必走 `nb[]`。',
    src='ggml/src/ggml-metal/kernels/unary.metal', parts=[(6, 40)], lang='cpp')

L.section(
    '十四、binbcast.metal：四种二元运算 + 广播',
    '加 / 减 / 乘 / 除四个算子由函数常量 `FC_bin_op` 分流；'
    '`FC_bin_rb`（行广播）与 `FC_bin_cb`（列广播）决定 src1 的下标怎么算。\n\n'
    '这是"内核正文与算子名解耦"的又一个例子：**同一段代码，靠常量选分支**。',
    src='ggml/src/ggml-metal/kernels/binbcast.metal', parts=[(11, 40)], lang='cpp')

L.section(
    '十五、misc.metal：argmax 与 MoE 辅助',
    '`misc.metal` 是杂项集合。被引用的 `kernel_argmax_f32` 一行一个线程组，'
    '组内先用 `simd_max` 求最大值，再用 `select` 把对应的下标也取出来 —— '
    '**归约值的同时归约下标**，这是 argmax 这类算子的常见写法。\n\n'
    '文件里还有 top-k、MoE 相关的辅助内核。',
    src='ggml/src/ggml-metal/kernels/misc.metal', parts=[(3, 40)], lang='cpp')

L.section(
    '十六、conv.metal：im2col 与卷积',
    '卷积在 GPU 上的常见做法是先 `im2col` 把输入展开成矩阵，再走矩阵乘。'
    '这里能看到线程组的三个维度分别对应输入通道、卷积核的行与列 —— '
    '`KH` / `KW` 直接取自 `ntg[1]` / `ntg[2]`，**核尺寸就是线程组的形状**。',
    src='ggml/src/ggml-metal/kernels/conv.metal', parts=[(9, 32)], lang='cpp')

L.section(
    '十七、pool.metal：每线程一个输出元素',
    '池化是最"朴素"的一类内核：`gid` 直接对应一个输出元素，'
    '算出窗口边界后逐个比较取最大值。没有线程组内存，也没有 SIMD 归约。\n\n'
    '注意边界处理：`bh` / `eh` 用 `MAX` / `MIN` 把窗口裁进输入范围，'
    '于是 padding 不需要真的填数据。',
    src='ggml/src/ggml-metal/kernels/pool.metal', parts=[(3, 40)], lang='cpp')

L.section(
    '十八、upscale.metal：最近邻与双线性',
    '上采样有两个内核：最近邻直接按放大倍数整除下标，双线性用 `bilinear_tri` 之类的'
    '权重函数做插值。`FC_upscale_aa` 决定是否走抗锯齿那条路。\n\n'
    '被引用的是最近邻版本：`i00 = i0/args.sf0` —— **一个整除就是整个上采样**。',
    src='ggml/src/ggml-metal/kernels/upscale.metal', parts=[(5, 30)], lang='cpp')

L.section(
    '十九、reduce.metal：求和也是两级归约',
    '`kernel_op_sum_f32` 把两级归约写成了最直白的形式：'
    '线程组内先 `simd_sum`，每个 SIMD group 写一个部分和进线程组内存，'
    '再由第 0 组读回来做第二次 `simd_sum`。\n\n'
    '`nsg` 是从 `ntg.x` 现算出来的（`(ntg.x + 31) / 32`），'
    '源码注释标注了"TODO: become function constant"。',
    src='ggml/src/ggml-metal/kernels/reduce.metal', parts=[(3, 45)], lang='cpp')

L.section(
    '二十、argsort.metal：双调排序',
    '排序用的是双调排序（bitonic sort），源码注释说明它"following the CUDA kernels as reference"。'
    '排序键与下标一起在线程组内存里排，所以 `shmem_i32` 存的是下标。\n\n'
    '模板参数是 `ggml_sort_order`（升序 / 降序）—— **排序方向也进模板**，'
    '这样比较那一步就没有运行期分支。',
    src='ggml/src/ggml-metal/kernels/argsort.metal', parts=[(10, 45)], lang='cpp')

L.section(
    '二十一、tri.metal：三角掩码的四种类型',
    '`tri` 把矩阵按三角区域清零。四种类型（下三角、含对角下三角、上三角、含对角上三角）'
    '被写成 `_ggml_vec_tri_cmp` 的四个模板特化 —— '
    '**模板特化在这里替代了运行期的 if**，与 mul_mm 用模板参数替代多份源码是同一个思路。',
    src='ggml/src/ggml-metal/kernels/tri.metal', parts=[(3, 38)], lang='cpp')

L.section(
    '二十二、solve_tri.metal：三角方程求解',
    '三角求解是逐行前代/回代，天然串行，所以这里的并行度放在"多列同时解"上：'
    '线程组的第 0 维取 `i01 = tgpig.x*NSG + sgitg`，即一个 SIMD group 负责一列。'
    '`N` / `K` / `NSG` 都是函数常量，`NP = PAD2(N, NW)` 把行长补齐到 32 的倍数。',
    src='ggml/src/ggml-metal/kernels/solve_tri.metal', parts=[(3, 34)], lang='cpp')

L.section(
    '二十三、ssm.metal：Mamba 的卷积与扫描',
    '`ssm` 是状态空间模型（Mamba）的算子。源码注释直接指路：'
    '**ref: ggml.c:ggml_compute_forward_ssm_conv_f32** —— '
    'GPU 内核与 CPU 参考实现是同一套公式的两种写法，这是 llama.cpp 后端的普遍做法。',
    src='ggml/src/ggml-metal/kernels/ssm.metal', parts=[(3, 26)], lang='cpp')

L.section(
    '二十四、wkv.metal：RWKV 的 WKV',
    'RWKV 的 WKV 算子是逐 head 递推的：`tgpig.x` 同时编码 batch 与 head'
    '（`batch_id = tgpig.x / H`，`head_id = tgpig.x % H`），线程组内再按 `tid` 分特征维。\n\n'
    '注意 `head_size` 被硬编码成 64，源码注释写着"TODO: support head_size = 128"。'
    '**凡注释声称的约束，都要按注释读**——这里说明该内核目前只支持 64 维 head。',
    src='ggml/src/ggml-metal/kernels/wkv.metal', parts=[(3, 27)], lang='cpp')

L.section(
    '二十五、gated_delta_net.metal：函数常量做形状参数',
    'Gated Delta Net 的内核把三个形状参数做成函数常量：`ne20`（V 的维度）、'
    '`ne30`（G 的维度）与 `K`，再用 `#define` 起短名字在正文里用。\n\n'
    '模板参数只有 `NSG`（SIMD group 数）—— 这是本目录里**形状走函数常量、'
    '并行度走模板**的典型分工：形状可以在 pipeline 创建时定，并行度要在编译期定。',
    src='ggml/src/ggml-metal/kernels/gated_delta_net.metal', parts=[(3, 26)], lang='cpp')

# ------------------------------------------------------------------ 结论

L.conclusion(
    '23 个文件、8 个族、13425 行',
    '本课覆盖 `ggml/src/ggml-metal/kernels/` 全部 23 个文件，按"算什么"分成 8 个族：\n\n'
    '| 族 | 文件数 | 行数 | host_name |\n|---|---|---|---|\n'
    '| 矩阵乘 | 2 | 4365 | 245 |\n'
    '| 注意力与序列混合 | 4 | 3395 | 567 |\n'
    '| 逐元素与形状 | 6 | 2337 | 61 |\n'
    '| 量化与反量化 | 3 | 1477 | 98 |\n'
    '| 归约 / 排序 / 三角 | 4 | 851 | 20 |\n'
    '| 归一化与 softmax | 2 | 541 | 18 |\n'
    '| 位置编码 | 1 | 333 | 8 |\n'
    '| 公共头 | 1 | 126 | 0 |\n\n'
    '三个数字都由命令数出来：`wc -l`、`grep -c host_name`、`tools/plan_matrix.py --files`。')

L.conclusion(
    '★ 模板 + host 侧实例化 = 量化支持矩阵',
    '`kernel_mul_mm` 的模板参数里，决定"支持哪种量化"的是三个：\n\n'
    '```text\n'
    'typename block_q                            // 块类型：block_q4_0 / block_q4_K / ...\n'
    'short nl                                    // 一个块装 16*nl 个权重\n'
    'void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &)\n'
    '```\n\n'
    '同一份 `mul_mm.metal` 里出现了 **23 种 block 类型**、**111 处 `host_name`**（51 处非 MoE）。'
    '主机端按 tensor 的 `type` 去取对应名字的 pipeline —— 名字就是两侧的接口。\n\n'
    '**与 L6-04 的对照**：CUDA 把每种实例写进 `template-instances/` 的独立文件'
    '（几十个 `.cu`，编译单元之间靠显式实例化衔接）；Metal 把实例化清单直接写在同一个 '
    '`.metal` 末尾，用 `template [[host_name("...")]]` 标记。'
    '两者解决同一个问题（避免运行期分发），代价也一样：**实例数随类型数增长**。'
    'fa.metal 就是代价的极端例子 —— 559 处 `host_name`、16 种 `(DK, DV)` 组合，'
    '源码注释自己写着 this is quite ugly。')

L.conclusion(
    '★ 形状判据：mul_mv 还是 mul_mm',
    '**src1 的行数（`ne11`）决定走哪条内核。**\n\n'
    '| 形状 | 更优 | 源码依据 |\n|---|---|---|\n'
    '| `ne11 == 1`（逐 token 解码） | `mul_mv` | 线程组只服务 src1 的一行：`r1 = tgpig.y`（`mul_mv.metal:237`）；'
    '该行缓存在 `float yl[16]`（`:265`）里被 NR0 行 src0 复用（`:285-287`） |\n'
    '| `ne11` 为 2 到 5 | `mul_mv_ext` | `i11 = tgpig.y * r1ptg`（`mul_mv.metal:623`），'
    '一个线程组吃下 `r1ptg` 行 src1；实例化只覆盖 2/3/4/5 |\n'
    '| `ne11` 大（prompt 批量） | `mul_mm` | tile 固定 64 x 32（`mul_mm.metal:164-165`，'
    '第 175 行注释写明 64x32）；A 的反量化结果进线程组内存（`:52`）后被整块 tile 复用 |\n\n'
    '一句话：**`mul_mm` 用"固定 32 列"换 A 块的复用；`mul_mv` 用"每行一次点积"换掉列方向的空转。**'
    'src1 只有一行时，32 列里 31 列是白算的。')

L.conclusion(
    'SIMD group 是内核的基本单位',
    '`common.h:23` 把 `N_SIMDWIDTH` 钉成 32。所有跨线程的归约都走同一个两级范式：\n\n'
    '```text\n'
    '第一级：simd_sum / simd_max / simd_shuffle_down   -- SIMD group 内（寄存器）\n'
    '第二级：threadgroup float * + threadgroup_barrier -- 跨 SIMD group（线程组内存）\n'
    '```\n\n'
    '`norm.metal:42-53`、`softmax.metal:49-65`、`reduce.metal:28-45`、`misc.metal:26-33` '
    '都是这个范式的实例。`softmax.metal:50` 的 `if (tptg.x > N_SIMDWIDTH)` 说明：'
    '**只有线程组比一个 SIMD group 宽时，第二级才需要**。')

L.conclusion(
    '块定义只有一份',
    '`dequantize.h:5-10` 用 `GGML_COMMON_DECL_METAL` 打开 `ggml-common.h` 的 Metal 分支，'
    '于是 Metal 看到的是与 CPU、CUDA 完全相同的 `block_q4_0` / `block_q4_K`。\n\n'
    '**内核实现可以各写各的，块的内存布局必须共用一份** —— 否则 GGUF 文件就没法跨后端读。'
    'L1-04 讲这些块的字节布局，本课讲 Metal 怎么用它们。')

# ------------------------------------------------------------------ 页脚 / 前置 / 目标

L.footnote_add('本课引用 `ggml/src/ggml-metal/kernels/` 下的 23 个文件，全部计入覆盖率；'
               '清单来自 `python3 tools/plan_matrix.py --files`。')
L.footnote_add('第 6 幕提到的 `r1ptg` 选择逻辑（按 `ne11` 在 2 到 8 之间取 2/3/4/5）在 '
               '`ggml/src/ggml-metal/ggml-metal-ops.cpp` 里，该文件由 L6-08 覆盖；'
               '本课不引用其源码，故不计入本课覆盖率。')
L.footnote_add('`ggml/src/ggml-metal/ggml-metal-impl.h`（`N_MM_*` / `N_R0_*` / `N_SG_*` 常量）'
               '同样由 L6-08 覆盖，本课不引用其源码，不计入本课覆盖率。'
               '第 4 幕与第 9 幕出现的宏名来自 `common.h:3` 的 include。')
L.footnote_add('`ggml/src/ggml-common.h`（块结构体定义）由 L1-04 覆盖，'
               '本课只引用 `dequantize.h` 里打开它分支的那几行，不计入本课覆盖率。'
               '该头文件按 `GGML_COMMON_DECL_C` / `_CPP` / `_METAL` / `_CUDA` / `_HIP` / `_SYCL` '
               '分支出各后端要的结构体定义 —— 第 8 幕的卡片讲的就是这件事。')

L.prereqs('`L6-08`（Metal 主机端与图融合：谁选 pipeline、谁决定融合）；'
          '相关：`L6-04`（CUDA 的 template-instances 对照）、`L1-04`（量化块的字节布局）')

L.goal(
    '说出 `mul_mv` 相比 `mul_mm` 在**什么形状下更优**：`src1` 只有一行或少数几行时走 `mul_mv`，'
    '因为 `mul_mm` 的线程组 tile 列方向固定是 32 列（对应验收点）；',
    '解释 Metal 如何用**一份 `mul_mm.metal`** 覆盖 23 种量化类型（模板参数 + `host_name` 实例化），'
    '并说出它与 CUDA `template-instances/` 的异同；',
    '说出 `N_SIMDWIDTH = 32` 之下，`simd_sum` / `simd_max` 与 threadgroup 内存如何组成两级归约；',
    '说明 `dequantize.h` 为什么要 `#define GGML_COMMON_DECL_METAL` 再 include `ggml-common.h`；',
    '按族说出 kernels 目录 23 个文件各自的职责。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
