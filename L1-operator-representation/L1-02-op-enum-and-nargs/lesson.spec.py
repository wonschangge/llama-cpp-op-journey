#!/usr/bin/env python3
"""L1-02 · 算子枚举与元数据：算子的身份 —— 课件 spec。

运行：python3 L1-operator-representation/L1-02-op-enum-and-nargs/lesson.spec.py

本课引用的所有代码都按行号从上游抽取（lessonkit.q），spec 里不出现任何手打代码。
场景里不使用 notes：注解行会让代码区的行号槽整体错位，而本课的核心正是"某条规则在哪一行"。
注解放在 source.md 的引用块里（那里没有行号槽）。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC = 'ggml/src/ggml.c'
IMPL = 'ggml/src/ggml-impl.h'
CPP = 'ggml/src/ggml.cpp'

L = Lesson(
    id='L1-02',
    layer='L1 · 算子的表示',
    title='算子枚举与元数据：算子的身份',
    codecap='ggml/src/ggml.c 等 3 个文件（逐字引用）',
    nav={'prev': {'href': '../L1-01-tensor-data-plane/index.html', 'label': 'L1-01 ggml 张量'},
         'next': {'href': '../L1-03-graph-and-toposort/index.html', 'label': 'L1-03 计算图与拓扑遍历'}},
)

L.note('**一句话**：在 ggml 里，一个算子不是一个函数指针，而是**枚举里的一个编号**；'
       '这个编号同时是名字表的下标、后端 `switch` 的判据、以及图节点上的 `op` 字段。')
L.note('L1-01 讲的是同一个节点的**数据面**（`ne[]` / `nb[]` / `type`）。这一课讲**身份面**，'
       '回答三个问题：它是什么运算（`enum ggml_op`）、它有几个输入（构造器里的 `src[]`）、'
       '它的输出是什么形状（构造器里的 `ne[]`）。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L1 · 算子的表示',
    title='算子的身份：<span class="hl-a">一个编号</span> + 两张表',
    sub='enum ggml_op 给每个算子一个编号；两张静态表把编号翻译成"日志里的名字"和"图里的符号"。',
    caption='回顾 L1-01：那里讲张量的数据面（ne/nb）；本课讲身份面。下一课 L1-03 讲引用这些节点的计算图。',
    src=SRC, parts=[(1364, 1370)], duration=14000,
    marks=[1, 5],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">模型定义</span><span class="arrow">-></span>
    <span class="chip a">enum ggml_op 编号</span><span class="arrow">-></span>
    <span class="chip b">图上的一个节点</span><span class="arrow">-></span>
    <span class="chip d">后端 switch 分派</span>
  </div>
  <div class="row center" id="q3" style="gap:11px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '它是什么运算', b: '枚举里的一个编号：<br>从 <span class="cm" style="margin:0">GGML_OP_NONE = 0</span> 到最后一个真算子。', m: 'enum ggml_op' },
  { c: 'b', t: '它有几个输入', b: '构造器里赋了几个 <span class="cm" style="margin:0">src[i]</span>，<br>没有集中的表可查。', m: 'result->src[0] = a;' },
  { c: 'd', t: '它的输出什么形状', b: '构造器里算 <span class="cm" style="margin:0">ne[]</span>，<br>或者直接抄输入的形状。', m: 'ggml_new_tensor(...)' }
];
const host = wrap.querySelector('#q3');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');
const msg = wrap.querySelector('#msg');
const texts = [
  '本课回答三个问题：<span class="k">是什么</span>、<span class="k">几个输入</span>、<span class="k">输出什么形状</span>。',
  '两个函数就是身份与文本之间的全部接口：<span class="v">GGML_OP_NAME[op]</span> 是直查表（表在 ggml.c:991）；' +
    '<span class="v">GGML_OP_SYMBOL[op]</span> 给 ggml_graph_dump_dot() 画图用（ggml.c:7907）。',
  '输入个数与输出形状<b>不是</b>集中定义的 —— 它们写在各算子的构造器里（第 3 幕起逐个看）。',
  '回顾 L1-01：那里讲同一个节点的数据面（<span class="v">ne/nb/type</span>）。' +
    'L1-03 会用 <span class="v">src[]</span> 把这些节点排成拓扑序。'
];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
}));
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L1-02 · 身份',
    title='★ <span class="hl-c">GGML_OP_NAME</span>：编号就是下标',
    sub='名字表是"枚举值到字符串"的直查表；表的长度和枚举的哨兵值必须一致。',
    caption='表尾的 static_assert（ggml.c:1104）把长度钉在 101 项 —— 末幕会看到那一行。',
    src=SRC, parts=[(991, 1000)], duration=16000,
    marks=[0, 2, 4, 5],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:10px">
    <div class="col grow" id="tbl"></div>
    <div class="col grow" id="side" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const names = ['NONE', 'DUP', 'ADD', 'ADD_ID', 'ADD1', 'ACC', 'SUB', 'MUL'];
const t = U.table(['编号', 'GGML_OP_NAME[编号]'],
  names.map((n, i) => [String(i), '"' + n + '"']), { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#side').innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
    '<div class="ct" style="color:var(--a)">枚举 enum ggml_op</div>' +
    '<div class="cb">定义在 <b>ggml/include/ggml.h:492-605</b>，共 101 个真算子；' +
    'L1-01 引用的就是同一个头文件。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
    '<div class="ct" style="color:var(--c)">名字表 GGML_OP_NAME</div>' +
    '<div class="cb">ggml.c:991-1102，长度写成 <b>GGML_OP_COUNT</b>。' +
    'ggml_op_name() 直接按下标取字符串。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)">' +
    '<div class="ct" style="color:var(--d)">符号表 GGML_OP_SYMBOL</div>' +
    '<div class="cb">ggml.c:1106-1218，元素是 <span class="cm" style="margin:0">"x+y"</span> 这类数学写法，' +
    '给 ggml_graph_dump_dot() 画图用（ggml.c:7907）。</div></div>';

const msg = wrap.querySelector('#msg');
tl.at(700, () => { msg.innerHTML = '表的第 0 项是 <span class="v">"NONE"</span>，第 1 项是 <span class="v">"DUP"</span>…… 下标就是枚举值。' +
  '<span class="k">GGML_OP_COUNT</span> 是枚举的哨兵值，等于真算子的个数。'; });
names.forEach((n, i) => tl.at(2400 + i * 1450, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = '编号 <span class="v">' + i + '</span> 的名字是 <span class="k">' + U.esc(n) + '</span>' +
    (i === 0 ? ' —— 还没成为节点的张量就是这个身份（第 7 幕）。' : '');
}));
tl.at(14200, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = '★ 枚举、名字表、符号表是<b>三处必须同步</b>的平行数组；' +
    '两张表尾各有一次 <span class="k">static_assert(GGML_OP_COUNT == 101, ...)</span> 兜底 —— ' +
    '枚举一改、表没跟上就编译不过（末幕）。';
});
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L1-02 · 输入',
    title='输入个数：写了几个 <span class="hl-b">src[i]</span> 就是几个',
    sub='ggml_mul_mat 赋了两个 src；ggml_add_id 赋了三个 —— 数量由各自的构造器决定。',
    caption='GGML_MAX_SRC = 10（ggml/include/ggml.h:224）只是 src[] 数组的长度上限，不是算子的输入个数。',
    src=SRC, parts=[(3341, 3356)], duration=17000,
    marks=[10, 11, 12],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row" id="two" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

function mk(color, title, rows, note) {
  const e = U.el('div', { class: 'card', style: 'width:330px;border-left-color:var(--' + color + ')' });
  let h = '<div class="ct" style="color:var(--' + color + ')">' + title + '</div>';
  rows.forEach(r => { h += '<div class="cm" style="margin:1px 0 0">' + U.esc(r) + '</div>'; });
  h += '<div class="cb" style="margin-top:4px">' + note + '</div>';
  e.innerHTML = h;
  return e;
}
const host = wrap.querySelector('#two');
const c1 = mk('b', 'ggml_mul_mat(ctx, a, b)',
  ['result->src[0] = a;', 'result->src[1] = b;'],
  '2 个输入（ggml.c:3352-3353，本幕的逐字引用）');
const c2 = mk('d', 'ggml_add_id(ctx, a, b, ids)',
  ['result->src[0] = a;', 'result->src[1] = b;', 'result->src[2] = ids;'],
  '3 个输入（ggml.c:2135-2137，逐字引用见 source.md 第三节）');
host.appendChild(c1); host.appendChild(c2);
c1.style.opacity = '.35'; c2.style.opacity = '.35';

const msg = wrap.querySelector('#msg');
tl.at(700, () => { msg.innerHTML = '代码区第 3345 行是契约：<span class="v">GGML_ASSERT(ggml_can_mul_mat(a, b))</span> —— ' +
  '不满足就 abort，错误路径见 source.md 第八节的 ggml.cpp。'; });
tl.at(3900, () => { c1.style.opacity = '1'; c2.style.opacity = '.35';
  msg.innerHTML = '<span class="k">MUL_MAT</span>：矩阵乘，两个输入。写进 <span class="v">src[0]</span> / <span class="v">src[1]</span>。'; });
tl.at(7500, () => { c1.style.opacity = '.35'; c2.style.opacity = '1';
  msg.innerHTML = '<span class="k">ADD_ID</span>：按 ids 里的下标做加，三个输入 —— ' +
    '<b>同一个枚举，输入个数可以不同</b>。'; });
tl.at(11200, () => {
  c1.style.opacity = '1'; c2.style.opacity = '1';
  msg.innerHTML = '★ 本版本源码里<b>没有</b>名为 <span class="v">nargs</span> 的表' +
    '（v0.5.0 的 ggml/ 目录 grep 无命中）；<span class="v">GGML_MAX_SRC = 10</span> 只是数组长度，' +
    '不是任何算子的输入个数。';
});
tl.at(14800, () => {
  msg.innerHTML = '没被赋值的 src 槽保持 <span class="v">NULL</span> —— src[] 就是图上的"入边表"，' +
    'L1-03 沿它做拓扑排序。';
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L1-02 · 输出',
    title='输出形状规则：<span class="hl-d">每个算子自己算</span>',
    sub='ggml_can_mul_mat 先断言输入合法，再由一行 ne[] 决定输出形状；没有统一的形状表。',
    caption='对照 ggml_glu_impl（ggml.c:2920-2921）：GLU 的输出是 a->ne[0] / 2，规则完全不同。',
    src=SRC, parts=[(3333, 3349)], duration=17000,
    marks=[3, 5, 15, 16],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:10px">
    <div class="col" style="gap:7px;width:330px" id="left"></div>
    <div class="col grow" style="gap:7px" id="right"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
const rule = U.el('div', { class: 'card', style: 'border-left-color:var(--b)' });
rule.innerHTML =
  '<div class="ct" style="color:var(--b)">GGML_OP_MUL_MAT 的形状规则</div>' +
  '<div class="cm" style="margin:2px 0 0">契约：a-&gt;ne[0] == b-&gt;ne[0]（3336）</div>' +
  '<div class="cm" style="margin:2px 0 0">输出：ne = { a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3] }（3348）</div>' +
  '<div class="cm" style="margin:2px 0 0">类型：GGML_TYPE_F32，与 a 的类型无关（3349）</div>' +
  '<div class="cb" style="margin-top:5px">两行代码 = 全部规则。没有表，没有注册，没有虚函数。</div>';
left.appendChild(rule);

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--d)">' +
    '<div class="ct" style="color:var(--d)">ADD_ID：抄输入</div>' +
    '<div class="cb"><span class="cm" style="margin:0">ggml_dup_tensor(ctx, a)</span> —— ' +
    '与 a 同形状、同类型（ggml.c:2132）。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
    '<div class="ct" style="color:var(--c)">GLU：第 0 维减半</div>' +
    '<div class="cb">输出 <span class="cm" style="margin:0">ne[0] = a-&gt;ne[0] / 2</span>，' +
    '其余维度照抄（ggml.c:2920）。</div></div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
    '<div class="ct" style="color:var(--f)">视图类：连数据都不新建</div>' +
    '<div class="cb">RESHAPE / TRANSPOSE / VIEW / PERMUTE 只改 ne/nb（第五幕）。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '先看契约：<span class="v">ggml_can_mul_mat()</span> 检查三个条件（3336-3338），构造器第一行就断言它。',
  'ne[0] 必须相等 —— 这是"内积维度"；ne[2] / ne[3] 只要求整除，即允许广播。',
  '然后才是形状：<span class="k">输出 = { a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3] }</span>（3348）。',
  '★ 输出类型固定 <span class="v">F32</span>（3349），与输入类型无关 —— 量化权重乘出来的仍是浮点结果。',
  '★ 别的算子给别的答案：ADD_ID 抄 a、GLU 砍一半。<b>形状规则散在构造器里，没有统一表。</b>'
];
tl.at(700, () => { U.markLines(document, [3, 5]); msg.innerHTML = texts[0]; });
tl.at(4000, () => { msg.innerHTML = texts[1]; });
tl.at(7300, () => { U.markLines(document, [15]); msg.innerHTML = texts[2]; });
tl.at(10600, () => { U.markLines(document, [16]); msg.innerHTML = texts[3]; });
tl.at(13900, () => { U.markLines(document, [3, 5, 15, 16]); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L1-02 · 分类',
    title='同一个枚举的第二种用法：<span class="hl-e">ggml_op_is_empty</span>',
    sub='把 op 当分类标签：只有 5 个算子"什么都不做"，其余 96 个都要真的算。',
    caption='同类开关还有 ggml_op_can_inplace（ggml-alloc.c:22）与 CPU 的 ggml_get_n_tasks（ggml-cpu.c:2253）——L5-01 会看到按 op 分派的大 switch。',
    src=IMPL, parts=[(90, 100)], duration=15000,
    marks=[2, 3, 4, 5, 6],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row center" id="chips" style="gap:7px"></div>
  <div class="row" id="cards" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const ops = ['GGML_OP_NONE', 'GGML_OP_RESHAPE', 'GGML_OP_TRANSPOSE', 'GGML_OP_VIEW', 'GGML_OP_PERMUTE'];
const host = wrap.querySelector('#chips');
const cs = ops.map((n, i) => { const c = U.chip(n, i === 0 ? 'g' : 'e'); host.appendChild(c); return c; });
cs.forEach(c => { c.style.opacity = '.35'; });

const cards = wrap.querySelector('#cards');
cards.innerHTML =
  '<div class="card" style="width:330px;border-left-color:var(--a)">' +
    '<div class="ct" style="color:var(--a)">返回 true 的只有这 5 个</div>' +
    '<div class="cb">default 分支返回 false —— 其余 96 个 op 都要真的读写数据。' +
    '分配器与图遍历据此决定"能不能跳过"。</div></div>' +
  '<div class="card" style="width:330px;border-left-color:var(--e)">' +
    '<div class="ct" style="color:var(--e)">其中 4 个是视图算子</div>' +
    '<div class="cb">RESHAPE / TRANSPOSE / VIEW / PERMUTE 的构造器都不新建数据：' +
    'TRANSPOSE 走 <span class="cm" style="margin:0">ggml_view_tensor()</span>（ggml.c:3937），' +
    '于是 <span class="cm" style="margin:0">view_src != NULL</span> —— 回顾 L1-01。</div></div>';

const msg = wrap.querySelector('#msg');
tl.at(700, () => { msg.innerHTML = '这个函数<b>不产生数据</b>，它只回答一个分类问题：op 是不是"空"的。'; });
tl.at(3600, () => { cs.forEach((c, i) => { c.style.opacity = i > 0 ? '1' : '.35'; });
  msg.innerHTML = '<span class="v">GGML_OP_NONE</span> 是"还没成为节点"的张量（第 7 幕）。'; });
tl.at(7000, () => { cs.forEach(c => { c.style.opacity = '1'; });
  msg.innerHTML = '<span class="v">RESHAPE / TRANSPOSE / VIEW / PERMUTE</span>：不产生新数据，只换一个读法。'; });
tl.at(10500, () => {
  msg.innerHTML = '回顾 L1-01：这些算子的张量 <span class="v">view_src</span> 非空、不拥有数据；' +
    'L4-01 的分配器据此跳过分配。';
});
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L1-02 · 参数',
    title='★ <span class="hl-e">op_params</span>：16 个 int32，schema 只是注释',
    sub='op_params 没有结构体类型；槽位含义由每个算子自己约定，唯一的边界检查是一条 assert。',
    caption='同一槽号在不同算子下含义不同：MUL_MAT 的槽 0 是累加精度，FLASH_ATTN_EXT 的精度在槽 3（ggml.c:3282-3304，见 source.md 第九节）。',
    src=IMPL, parts=[(163, 183)], duration=18000,
    marks=[2, 3, 4, 5, 13],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row" id="slots" style="gap:3px"></div>
  <div class="row" id="cards" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#slots');
const cells = [];
for (let i = 0; i < 16; i++) {
  const c = U.el('div', { style: 'flex:1 1 0;height:22px;border:1px solid var(--border);border-radius:3px;background:#10151b;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:9px;color:var(--dim)' });
  c.textContent = String(i);
  host.appendChild(c); cells.push(c);
}

wrap.querySelector('#cards').innerHTML =
  '<div class="card" style="width:330px;border-left-color:var(--e)">' +
    '<div class="ct" style="color:var(--e)">GGML_OP_MUL_MAT 的槽位表（注释）</div>' +
    '<div class="cm" style="margin:2px 0 0">0 - acc &nbsp; 1 - hint</div>' +
    '<div class="cm" style="margin:2px 0 0">2 - src0 precision &nbsp; 3 - src1 precision</div>' +
    '<div class="cb" style="margin-top:4px">写在 ggml-impl.h:163-174，' +
    '<b>只覆盖 MUL_MAT 与 MUL_MAT_ID 两个算子</b>。</div></div>' +
  '<div class="card" style="width:330px;border-left-color:var(--g)">' +
    '<div class="ct" style="color:var(--g)">同一个写入函数，另一个槽号</div>' +
    '<div class="cb">ggml_prec_set_acc() 里：MUL_MAT 写槽 <b>0</b>，' +
    'FLASH_ATTN_EXT 写槽 <b>3</b>（ggml.c:3290 / 3296）—— 这段注释没写到这个算子。</div></div>';

const msg = wrap.querySelector('#msg');
tl.at(700, () => { msg.innerHTML = 'op_params 是 <span class="v">64 字节 = 16 个 int32</span>' +
  '（GGML_MAX_OP_PARAMS，L1-01 讲过它的来历）。'; });
tl.at(4000, () => {
  cells.forEach((c, i) => { c.style.borderColor = i < 4 ? 'var(--e)' : 'var(--border)'; });
  msg.innerHTML = '★ 它<b>没有结构体类型</b>：槽位含义只写在注释里（[TAG_GGML_PREC]），而且只覆盖了两个算子。';
});
tl.at(7800, () => {
  cells.forEach(c => { c.style.background = '#10151b'; });
  msg.innerHTML = '写入接口：<span class="v">ggml_set_op_params_i32(t, i, v)</span>。' +
    '唯一的边界检查是第 176 行的 <span class="k">i &lt; 16</span>，写错槽位编译器不会拦你。';
});
tl.at(11600, () => {
  cells[0].style.background = 'rgba(247,120,186,.22)';
  cells[3].style.background = 'rgba(255,123,114,.22)';
  msg.innerHTML = '★ 同一个槽号在不同算子下含义不同：MUL_MAT 的槽 <span class="v">0</span> 是累加精度，' +
    'FLASH_ATTN_EXT 的精度在槽 <span class="v">3</span>。';
});
tl.at(15000, () => {
  msg.innerHTML = '读者是后端内核：CPU 的 FlashAttention 实现直接 ' +
    '<span class="cm">switch (dst-&gt;op_params[3])</span>（ggml/src/ggml-cpu/ops.cpp:9351，本课不引用该文件）。';
});
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L1-02 · 成节点',
    title='新张量的默认身份：<span class="hl-g">GGML_OP_NONE</span>',
    sub='张量出生时 op = NONE、src 全 NULL；构造器随后把身份写进去，它才成为图上的节点。',
    caption='写完之后 ggml_build_forward_expand 才能沿 src[] 做拓扑排序 —— 那是 L1-03。',
    src=SRC, parts=[(1809, 1824)], duration=16000,
    marks=[5, 6, 8],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">ggml_new_tensor_impl</span><span class="arrow">-></span>
    <span class="chip g">op = NONE, src = NULL</span><span class="arrow">-></span>
    <span class="chip b">构造器写 op + src</span><span class="arrow">-></span>
    <span class="chip a">图上的节点</span>
  </div>
  <div class="row center" id="cards" style="gap:11px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'g', t: '出生：没有身份', b: 'ggml.c 里的张量创建都汇到这一个函数：' +
      'ggml_new_tensor 系列、dup、view（1844 / 1895 / 1963）。', m: 'GGML_OP_NONE' },
  { c: 'b', t: '构造器：写入身份', b: 'result-&gt;op = GGML_OP_XXX; 再填 src[i]。' +
      '这才是"这个张量是什么运算"。', m: 'result->op = ...;' },
  { c: 'a', t: '成为节点', b: 'src[] 非空 = 有入边，图遍历才能访问到它。' +
      '哪些张量算图的输出，是 L1-03 的事。', m: 'ggml_build_forward_expand' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');
const msg = wrap.querySelector('#msg');
const texts = [
  '同一段内存，两种状态：<span class="k">叶子张量</span>（权重、输入）与<span class="k">运算节点</span>。',
  '出生时 <span class="v">op = GGML_OP_NONE</span>（名字表第 0 项）、<span class="v">src</span> 全 NULL，' +
    '所以它<b>还不是</b>图上的节点。',
  '构造器写入 op 与 src 后，身份与入边同时确定 —— 它才是一个"算子节点"。',
  '回顾 L1-01：<span class="v">ggml_tensor</span> 是定长值类型，所以"建节点"就是改这几个字段。' +
    'L1-03 会沿 <span class="v">src[]</span> 做拓扑排序。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
}));
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L1-02 · 收束',
    title='把这一课压成一张表',
    sub='三个问题，三种答案的存放位置；两张表尾的 static_assert 负责让它们不脱节。',
    caption='下一课 L1-03 讲这些 src[] 如何被拓扑排序成一张可执行的图。',
    src=SRC, parts=[(1101, 1106)], duration=17000,
    marks=[3],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', '答案在哪（行号）', '本课', '后面谁用'],
  [['它是什么运算', 'enum ggml_op（ggml.h:492-605）', '第 1-2 幕', 'L5-01 后端分派'],
   ['它的字符串名', 'GGML_OP_NAME（ggml.c:991-1102）', '第 2 幕', 'L1-03 图打印'],
   ['图上的符号', 'GGML_OP_SYMBOL（ggml.c:1106-1218）', '第 1-2 幕', 'L1-03 dot 图'],
   ['有几个输入', '构造器里的 src[i]（3352-3353 / 2135-2137）', '第 3 幕', 'L1-03 拓扑排序'],
   ['输出什么形状', '构造器里的 ne[]（3348）或 dup（2132）', '第 4 幕', 'L5-xx 内核'],
   ['标量参数', 'op_params[16]（ggml-impl.h:163-174）', '第 6 幕', 'L5-01 后端读取'],
   ['不产生数据', 'ggml_op_is_empty（ggml-impl.h:90）', '第 5 幕', 'L4-01 分配器']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '给定 <span class="mono">ggml_add_id(ctx, a, b, ids)</span>：它有几个输入？输出形状由哪一行决定？<br>' +
  '再答：为什么 <span class="mono">ggml_mul_mat</span> 不能改用 <span class="mono">ggml_dup_tensor(ctx, a)</span>？',
  'ADD_ID 有 <b>3 个输入</b>：构造器依次写了 src[0]=a、src[1]=b、src[2]=ids（ggml.c:2135-2137）。<br>' +
  '输出形状由 <span class="mono">ggml_dup_tensor(ctx, a)</span> 决定（ggml.c:2132）—— 与 a 同形状、同类型。<br>' +
  'MUL_MAT 不行：它的输出是 <span class="mono">{ a->ne[1], b->ne[1], b->ne[2], b->ne[3] }</span>（ggml.c:3348），' +
  '类型固定 <span class="mono">GGML_TYPE_F32</span>（ggml.c:3349），都不是 a 自己的形状或类型。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '一个问题一行，注意"答案在哪"这一列：<b>只有名字与符号是查表</b>。',
  '前两行是两张静态表：下标就是枚举编号（第 2 幕）。',
  '中间两行没有任何表 —— 输入个数与输出形状散在各算子的构造器里（第 3、4 幕）。',
  '最后两行是标量参数与分类开关（第 5、6 幕）。代码区的 static_assert 说明：<b>枚举一改，两张表没跟上就编译不过</b>。',
  '一句话：<span class="k">op 是身份编号，src[] 是入边，ne[] 是形状 —— 三样都在同一张 ggml_tensor 里</span>（回顾 L1-01）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2100, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(15600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、身份的两张表：名字与符号',
    '算子的身份是一个整数（`enum ggml_op`），但人要看名字、图要画符号 —— '
    '这两个函数就是身份与文本之间的全部接口。注意它们都不做边界检查：'
    '`op` 越界会直接读越界内存，调用方必须保证传进来的是合法枚举值。',
    src=SRC, parts=[(1364, 1370)], lang='c',
    notes={1: '直查表：op 编号就是数组下标（表在 ggml.c:991）',
           5: '供 ggml_graph_dump_dot() 画图用（ggml.c:7907）'})

L.section(
    '二、★ GGML_OP_NAME：编号就是下标',
    '名字表是"枚举值到字符串"的**直查表**：`GGML_OP_NAME[op]`。'
    '表的长度写成 `GGML_OP_COUNT`（枚举的哨兵值），表尾还有一次 `static_assert` 兜底。'
    '所以**加一个算子要改三处**：`enum ggml_op`、`GGML_OP_NAME`、`GGML_OP_SYMBOL`；'
    '枚举改了而表没跟上，编译直接失败。\n\n'
    '本版本（`v0.5.0`）实测：枚举里 101 个真算子（`GGML_OP_NONE = 0` 到 `GGML_OP_GLU`，'
    '`ggml/include/ggml.h:493-602`），两张表各有 101 项，两处 `static_assert(GGML_OP_COUNT == 101, ...)`。',
    src=SRC, parts=[(991, 1000), (1100, 1106)], lang='c')

L.section(
    '三、输入个数：写在各算子的构造器里',
    '课件正文里常说"查 nargs 表"，但**本版本的源码里没有这张表**：'
    '在 `ggml/` 目录下 grep `nargs` 无命中，`ggml_op_name()` / `ggml_op_symbol()` 之外也没有'
    '以 op 为下标的输入个数数组。真正的规则是：**构造器给几个 `src[i]` 赋值，这个算子就有几个输入**。\n\n'
    '- `ggml_mul_mat`：2 个输入（`src[0] = a`、`src[1] = b`）；\n'
    '- `ggml_add_id`：3 个输入（`src[0] = a`、`src[1] = b`、`src[2] = ids`）。\n\n'
    '`GGML_MAX_SRC = 10`（`ggml/include/ggml.h:224`）只是 `src[]` 数组的长度上限，'
    '不是任何算子的输入个数。没被赋值的槽保持 `NULL`，因此 `src[]` 同时就是图的**入边表**。',
    src=SRC, parts=[(3341, 3356), (2121, 2140)], lang='c')

L.section(
    '四、输出形状规则：也在构造器里',
    '`GGML_OP_MUL_MAT` 的完整规则只有两行：先用 `ggml_can_mul_mat()` 断言输入合法，'
    '再用 `ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] }` 决定输出形状，'
    '类型固定为 `GGML_TYPE_F32`（与输入类型无关）。\n\n'
    '别的算子给别的答案：`ggml_add_id` 直接 `ggml_dup_tensor(ctx, a)`（同 a 的形状与类型），'
    '`ggml_glu_impl` 把第 0 维减半。**没有统一的形状表**，规则就写在各自构造器的那一两行里。',
    src=SRC, parts=[(3333, 3349), (2906, 2931)], lang='c')

L.section(
    '五、不产生数据的算子：ggml_op_is_empty',
    '同一个枚举还有第二种用法：**把 op 当分类标签**。`ggml_op_is_empty()` 只对 5 个 op 返回 `true`，'
    '其中 4 个是视图算子 —— 它们的构造器不新建数据，只改 `ne[]` / `nb[]`，'
    '张量的 `view_src` 因此非空（回顾 L1-01）。',
    src=IMPL, parts=[(90, 100)], lang='c',
    notes={0: '按 op 分类的第二种用法：这个算子会不会真的产生数据'})

L.section(
    '五之二、视图构造器长什么样：ggml_transpose',
    '`ggml_transpose()` 是最短的一个视图构造器：先 `ggml_view_tensor(ctx, a)`（这一步让 `view_src` 指向 '
    '`a`），再交换 `ne[0]/ne[1]` 与 `nb[0]/nb[1]`，最后才写上 `op` 与 `src[0]`。'
    '**整个过程没有分配任何数据** —— 这正是 `ggml_op_is_empty()` 把它归为"空"的原因，'
    '也是 L4-01 的分配器可以跳过它的原因。',
    src=SRC, parts=[(3934, 3950)], lang='c',
    notes={3: 'view_src = a：数据仍在 a 那里，这个张量只是换个读法',
           12: '身份（op）与入边（src[0]）最后才写上'})

L.section(
    '六、op_params：16 个 int32 的读写接口',
    '算子的标量参数（轴号、eps、精度）不存在结构体里，而是塞进 `op_params` 这 64 字节。'
    '三个接口就是全部机制：一次 `memcpy` 写入，两个按 `int32` / `float` 取值的 getter。'
    '唯一的边界检查是那条 `assert`，而它只在 `assert` 打开的构建里生效。',
    src=IMPL, parts=[(147, 161)], lang='c',
    notes={2: '写入前先检查长度不超过 GGML_MAX_OP_PARAMS（64 字节）',
           7: '取值前的边界检查：i 必须小于 16'})

L.section(
    '七、★ 槽位约定只是注释：TAG_GGML_PREC',
    '`op_params` 的槽位含义**没有类型系统表达**：它写在 `ggml-impl.h` 的一段注释里，'
    '由每个算子自己约定。这段注释只列了 `GGML_OP_MUL_MAT` 与 `GGML_OP_MUL_MAT_ID` 两个算子。\n\n'
    '而实际写入者比注释更宽：`ggml_prec_set_acc()` 对 `GGML_OP_MUL_MAT` 写槽 `0`，'
    '对 `GGML_OP_FLASH_ATTN_EXT` 却写槽 `3`。**同一个槽号在不同算子下含义不同**，'
    '这正是"算子的身份决定它的元数据怎么解释"。'
    '（源码注释不一定是完整的：这里注释没有覆盖 `FLASH_ATTN_EXT`。）',
    src=IMPL, parts=[(163, 183)], lang='c',
    notes={1: '约定按算子分节：同一个槽号在不同 op 下含义不同',
           13: '唯一的边界检查：i < 16（GGML_MAX_OP_PARAMS / sizeof(int32_t)）'})

L.section(
    '八、算子契约失败之后：GGML_ASSERT 与 terminate handler',
    '每个构造器开头都用 `GGML_ASSERT` 写死了输入契约（例如 `ggml_mul_mat` 的 '
    '`GGML_ASSERT(ggml_can_mul_mat(a, b))`）。契约不满足时进程会 `abort()`。\n\n'
    '`ggml/src/ggml.cpp` 是 C++ 侧的兜底：它在静态初始化时装一个 `std::terminate` 处理器，'
    '终止前先 `ggml_print_backtrace()` 打出回溯；`GGML_NO_BACKTRACE` 环境变量可以关掉这个行为。'
    '这就是"元数据写错 / 契约违反"时你实际看到的输出路径。',
    src=CPP, parts=[(1, 26)], lang='c')

L.section(
    '九、第七节的证据：同一个写入者的两个槽号',
    '第七节说"同一个写入函数对另一个算子写另一个槽号"，证据就是下面这段：'
    '`ggml_prec_set_acc()` 的 `switch (a->op)` —— `case GGML_OP_MUL_MAT` 写槽 `0`，'
    '`case GGML_OP_FLASH_ATTN_EXT` 写槽 `3`，`default` 直接返回 `false`（表示这个算子不支持该精度设置）。'
    '**同一个槽号在两张"表"里意思不同**，判断依据只有 `a->op`。',
    src=SRC, parts=[(3282, 3304)], lang='c',
    notes={4: 'switch 的判据是 a->op —— 身份决定元数据怎么解释',
           8: 'MUL_MAT / MUL_MAT_ID：精度写进槽 0',
           14: 'FLASH_ATTN_EXT：精度写进槽 3',
           17: '不支持这个精度设置的算子返回 false'})

L.footnote_add('本课逐字引用 3 个文件：`ggml/src/ggml.c`、`ggml/src/ggml-impl.h`、`ggml/src/ggml.cpp`，'
               '三者都计入覆盖率（与计划 `plan/COVERAGE.md` 给 L1-02 指派的 3 个文件一致）。')
L.footnote_add('正文与图示里有若干行号**只是指路**，它们没有被本课的引用块覆盖，也不计入覆盖率：'
               '`ggml/include/ggml.h:492`、`ggml/include/ggml.h:224`、`ggml/src/ggml-alloc.c:22`、'
               '`ggml/src/ggml-cpu/ggml-cpu.c:2253`、`ggml/src/ggml-cpu/ops.cpp:9351`、'
               '以及 `ggml/src/ggml.c` 内未被引用的定位行（7907 / 1844 / 1895 / 1963）。'
               '本课的覆盖率声明只有上一条那 3 个文件。')
L.footnote_add('"没有 nargs 表"是**实测结论**：在 `ggml/` 目录下 grep `nargs` 无命中。'
               '计划文档里"参数个数表 nargs"的措辞与本版本源码不符，本课按实测改写为'
               '"构造器里的 src[] 赋值"。')
L.footnote_add('场景里的代码引用不使用注解行（`notes`），因为注解行会让代码区的行号槽整体错位；'
               '而本课的核心恰恰是"某条规则在哪一行"。注解改放在本篇 `source.md` 的引用块里。')

L.prereqs('`L1-01`（ggml 张量：算子的数据面）')

L.goal(
    '说出算子的身份由哪些东西定义：`enum ggml_op` 编号、`GGML_OP_NAME` / `GGML_OP_SYMBOL` 两张表（对应验收点）；',
    '解释为什么"输入个数"在本版本源码里不是查表得到的，而是散在每个算子的构造器里；',
    '指出任意一个算子的"输出形状规则"写在哪个函数里（构造器的 `ne[]`，或 `ggml_dup_tensor` / 视图构造器）；',
    '说明 `op_params` 是 16 个 `int32` 的无 schema 元数据，槽位含义由算子自己约定，且注释并不完整。')

L.conclusion(
    '算子的身份 = 一个编号 + 两张平行表',
    '`enum ggml_op`（`ggml/include/ggml.h:492-605`）给每个算子一个编号，'
    '`GGML_OP_NAME`（`ggml.c:991-1102`）与 `GGML_OP_SYMBOL`（`ggml.c:1106-1218`）以编号为下标直查。'
    '两张表都以 `GGML_OP_COUNT` 为长度，并各由一次 `static_assert(GGML_OP_COUNT == 101, ...)` 钉住 —— '
    '**加算子必须同步改三处，否则编译不过**。')

L.conclusion(
    '★ 输入个数与输出形状：都在构造器里，没有表',
    '```text\n'
    '输入个数 = 构造器里赋值了几个 result->src[i]\n'
    '输出形状 = 构造器里算出的 ne[]（或 ggml_dup_tensor / 视图构造器）\n'
    'GGML_MAX_SRC = 10   只是 src[] 数组长度上限，不是任何算子的输入个数\n'
    '```\n\n'
    '`ggml_mul_mat`：2 个输入、输出 `{ a->ne[1], b->ne[1], b->ne[2], b->ne[3] }`、类型固定 F32。'
    '`ggml_add_id`：3 个输入、输出 = `ggml_dup_tensor(ctx, a)`。'
    '要回答"某个 op 的输入个数与输出形状规则在哪定义"，答案是**它的构造器**，不是某张表。')

L.conclusion(
    '自检：给一个 GGML_OP_*，去哪找它的输入个数与输出形状',
    '| 要回答的问题 | 去哪找 | 本课引用的例子 |\n'
    '|---|---|---|\n'
    '| 它是什么运算 | `enum ggml_op`（`ggml/include/ggml.h:492-605`） | `GGML_OP_MUL_MAT` |\n'
    '| 它叫什么名字 | `GGML_OP_NAME[op]`（`ggml.c:991-1102`）+ `ggml_op_name()` | `"MUL_MAT"` |\n'
    '| 图里画成什么符号 | `GGML_OP_SYMBOL[op]`（`ggml.c:1106-1218`）+ `ggml_op_symbol()` | `"x*y"` |\n'
    '| **它有几个输入** | **它的构造器**里 `result->src[i]` 赋值了几个 | `ggml_mul_mat` 2 个（`ggml.c:3352-3353`）；`ggml_add_id` 3 个（`ggml.c:2135-2137`） |\n'
    '| **它的输出是什么形状** | **它的构造器**里的 `ne[]`，或 `ggml_dup_tensor()` / 视图构造器 | MUL_MAT `ggml.c:3348-3349`；ADD_ID `ggml.c:2132`；GLU `ggml.c:2920` |\n'
    '| 它的标量参数 | `op_params[16]`，槽位约定见 `ggml-impl.h:163-174` | MUL_MAT 的槽 0/1/2/3 |\n\n'
    '换个说法：**名字与符号查表，输入个数与输出形状查构造器**。')

L.conclusion(
    '★ op_params：约定在注释里，边界只有一条 assert',
    '`op_params` 是 64 字节（16 个 `int32`），写入靠 `ggml_set_op_params_i32()`，'
    '唯一检查是 `i < GGML_MAX_OP_PARAMS / sizeof(int32_t)`。槽位含义写在 `ggml-impl.h:163-174` 的 '
    '`[TAG_GGML_PREC]` 注释里，**只覆盖 `MUL_MAT` / `MUL_MAT_ID`**；'
    '`ggml_prec_set_acc()` 对 `FLASH_ATTN_EXT` 用的是槽 `3`。'
    '换一个 op，同一槽号的意思就变了 —— 这是"身份决定元数据怎么解释"。')

L.conclusion(
    '同一枚举的第三种用法：分类',
    '`ggml_op_is_empty()`（`ggml-impl.h:90`）只对 `NONE / RESHAPE / TRANSPOSE / VIEW / PERMUTE` '
    '返回 `true`。同类开关还有 `ggml_op_can_inplace()`（`ggml-alloc.c:22`）与 CPU 后端的 '
    '`ggml_get_n_tasks()`（`ggml-cpu.c:2253`）；**L5-01 会看到后端按 op 分派的那个大 `switch`**，'
    '它就是本课"编号即身份"的最终消费者。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
