#!/usr/bin/env python3
"""L1-03 · 计算图与拓扑遍历 —— 课件 spec。

运行：python3 L1-operator-representation/L1-03-graph-and-toposort/lesson.spec.py

写作约定：visual 里的 JS 写得紧凑（不缩进、少空行、短标识符），
把较长的解释放进 source.md。这样 lesson.js 保持在 32 KB 上下，浏览器一次加载即完，
`dump_scenes.js` 求值也更快。历史背景：本课写作期间 tools/check_flags.py 有一个缺陷
（把 lesson.js 的【内容】当路径传给 dump_scenes.js，node 回显内容到 stderr 并在 65536
字节处截断，截断点多字节字符中间时 Python strict 解码抛 UnicodeDecodeError），
当时 lesson.js 必须 <= 32742 字节；该缺陷已在 3385a08 修复，这个上限不再存在。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC = 'ggml/src/ggml.c'                 # 主源文件
IMPL = 'ggml/src/ggml-impl.h'           # struct ggml_cgraph / struct ggml_hash_set 的家
CPP = 'ggml/include/ggml-cpp.h'         # C++ 侧 RAII 包装层

L = Lesson(
    id='L1-03',
    layer='L1 · 算子的表示',
    title='计算图与拓扑遍历：节点怎么排成一队',
    codecap='ggml/src/ggml.c · ggml/src/ggml-impl.h · ggml/include/ggml-cpp.h（逐字引用）',
    nav={'prev': {'href': '../L1-02-op-enum-and-nargs/index.html', 'label': 'L1-02 算子枚举与元数据'},
         'next': {'href': '../L1-04-quant-block-layout/index.html', 'label': 'L1-04 量化块结构'}},
)

L.note('**一句话**：`ggml_build_forward_expand(cgraph, tensor)` 把"一个输出张量"变成'
       '"一个有序的节点数组 `cgraph->nodes[]`"；排序靠**后序深度优先遍历**，'
       '去重靠**图自己持有的 hash set**。')
L.note('为什么值得单独一课：L4-04 的后端 `graph_compute` 就是按 `nodes[0], nodes[1], ...` '
       '的顺序逐个执行的。顺序错了，模型算出来的就是另一个东西。')
L.note('验收点：**能解释为什么拓扑排序用 hash set 而不是在张量上打个标记位**。'
       '答案不在"hash 快"，而在"状态归谁"——标记位必须属于图，不能属于张量。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L1 · 算子的表示',
    title='从张量到图：这一课看的是<span class="hl-a">遍历</span>',
    sub='L1-01 给了节点，L1-02 给了身份；本课看 ggml 如何把它们串成一张有序的、可执行的图。',
    caption='跨课：src[] 是图的边（L1-01）；op 决定算节点还是叶子（L1-02）；顺序被 L4-04 的 graph_compute 消费。',
    src=SRC, parts=[(7337, 7339)], duration=17000,
    marks=[1],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
w.innerHTML = `<div class="flow" style="justify-content:center">
<span class="chip">张量 + op</span><span class="arrow">-></span><span class="chip a">ggml_build_forward_expand</span><span class="arrow">-></span><span class="chip b">nodes[] 有序数组</span><span class="arrow">-></span><span class="chip c">graph_compute</span></div>
<div class="row center" id="c" style="gap:9px"></div><div class="formula" id="m"></div>`;
root.appendChild(w);
const m = w.querySelector('#m');
w.querySelector('#c').innerHTML = '<div class="formula" style="line-height:1.6">' +
 '<span class="a">输入</span> 一张（可能是空的）图 + 一个输出张量　|　' +
 '<span class="c">过程</span> 后序 DFS 沿 src[] 展开，hash set 保证每个张量只入列一次　|　' +
 '<span class="b">输出</span> cgraph-&gt;nodes[0 .. n_nodes)：先算输入，再算用到它们的节点</div>';
const T = ['入口只有三行：把活全部转给内部实现。',
 '输入 = <span class="k">一张图 + 一个输出张量</span>；图可能已有内容（所以叫 expand）。',
 '要解决两件事：<span class="k">重复</span>（同一张量被多处引用）与 <span class="k">顺序</span>（谁先算）。',
 '回顾 L1-01：边就是 <span class="v">src[10]</span>；回顾 L1-02：<span class="v">op</span> 决定节点身份。<br>产物被 L4-04 按 <span class="v">nodes[]</span> 顺序消费。'];
T.forEach((t, i) => tl.at(700 + i * 3600, () => { m.innerHTML = t; }));
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L1-03 · 排序',
    title='★ <span class="hl-a">后序 DFS</span>：nodes[] 出来就是拓扑序',
    sub='父节点在它的所有 src[] 之后入列；源码用一条断言把这个不变量钉死。',
    caption='n_new > 0 才断言：expand 允许对同一张图多次展开，但"最后入列的一定是本次起点"永远成立。',
    src=SRC, parts=[(7304, 7321)], duration=27000,
    marks=[6, 8, 10, 15],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
w.innerHTML = `<div class="row" style="gap:12px"><div class="col" id="d" style="gap:4px;width:246px"></div><div class="col grow" id="p" style="gap:4px"></div></div><div class="formula" id="m"></div>`;
root.appendChild(w);
const d = w.querySelector('#d'), p = w.querySelector('#p'), m = w.querySelector('#m');
const EG = [['N.src[0] L1', 'N'], ['N.src[1] L2', 'N'], ['O.src[0] N', 'O'], ['O.src[1] L2', 'O']];
d.innerHTML = '<div class="cm" style="margin:0">输出张量 O 的图（边 = src[]）</div>' + EG.map(e =>
 '<div class="formula"><span class="m">' + e[0] + '</span> -&gt; ' + e[1] + '</div>').join('') +
 '<div class="cb" style="font-size:9.5px">L2 被引用两次 —— "重复"就从这里来。</div>';
const G = d.querySelectorAll('.formula');
p.innerHTML = '<div class="cm" style="margin:0">leafs[] · op == GGML_OP_NONE</div><div class="col" id="lf" style="gap:3px"></div>' +
 '<div class="cm" style="margin:4px 0 0">nodes[] · 要算的</div><div class="col" id="nd" style="gap:3px"></div>';
const lf = p.querySelector('#lf'), nd = p.querySelector('#nd');
function put(x, t, c) { const e = U.el('div', { class: 'formula', style: 'padding:2px 7px;font-size:10px' });
  e.innerHTML = '<span class="' + c + '">' + t + '</span>'; x.appendChild(e); }
const S = [
 [700, -1, '', '', '从输出 <span class="k">O</span> 出发：O.src[0] = N，O.src[1] = L2。'],
 [4200, 2, '', '', 'dfs(O) -> 先递归 N；dfs(N) -> 再递归 L1、L2。'],
 [8000, 0, 'lf', 'leafs[0] = L1', '两个叶子 op 都是 NONE，进 <span class="v">leafs[]</span>。'],
 [11400, 1, 'lf', 'leafs[1] = L2', '<span class="k">两个输入都处理完</span>，N 才进 nodes[]。'],
 [14800, 3, '', '', '回到 O，src[1] 又指向 L2 —— 第二次到达，<span class="k">不能再入列一次</span>。'],
 [18200, -1, 'nd', 'nodes[0] = N', '最后 O 入列：<span class="v">nodes[] = [N, O]</span>，输入都在它前面。'],
 [21600, -1, '', '', '源码第 7319 行把它写成断言：最后入列的，必须正好是本次的起点张量。']];
S.forEach(s => tl.at(s[0], () => {
  G.forEach((x, k) => { x.style.opacity = (s[1] < 0 || k === s[1]) ? '1' : '.3'; });
  if (s[2]) { put(s[2] === 'lf' ? lf : nd, s[3], s[2] === 'lf' ? 'v' : 'k'); }
  m.innerHTML = s[4];
}));
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L1-03 · 核心',
    title='★ 每个张量只入列一次：<span class="hl-a">先查表，再插入</span>',
    sub='同一张量被多个节点引用时，第二次到达必须立刻返回 —— 否则它会在 nodes[]/leafs[] 里出现两次。',
    caption='第二次到达也不是空转：compute 为真时仍会递归给 src[] 补 COMPUTE 标志；去重针对的是"入列"。',
    src=SRC, parts=[(7236, 7263)], duration=24000,
    notes={8: 'used[pos] = 1：这张图收过它了 -> 直接 return',
           23: '首次见到：写 keys[]、置 used 位、清 use_counts'},
    marks=[5, 6, 8, 10, 22, 27, 28, 29],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
w.innerHTML = `<div class="row" style="gap:12px"><div class="col" id="g" style="width:320px"></div><div class="col grow" id="s" style="gap:6px"></div></div><div class="formula" id="m"></div>`;
root.appendChild(w);
const t = U.table(['到达顺序', 'used 位', '做什么'],
 [['O = f(N, L2)', '0 -> 1', '标记 + 递归 src[]'],
  ['N = g(L1, L2)', '0 -> 1', '标记 + 递归 src[]'],
  ['L1', '0 -> 1', '叶子 -> leafs[0]'],
  ['L2（第一次）', '0 -> 1', '叶子 -> leafs[1]'],
  ['L2（第二次）', '已是 1', 'return：不再入列']], { monoCols: [0, 1] });
w.querySelector('#g').appendChild(t.el);
const R = t.body.querySelectorAll('tr');
R.forEach(r => { r.style.opacity = '.3'; });
w.querySelector('#s').innerHTML =
 '<div class="formula" style="line-height:1.65">' +
 '① 算槽位：<span class="v">ggml_hash_find(&amp;cgraph-&gt;visited_hash_set, node)</span>；全满返回 GGML_HASHSET_FULL，遍历里紧跟断言<br>' +
 '② 看 used 位：为 1 = 本图收过它 -> <span class="k">直接 return</span>。同一个 L2 被引用两次，入列只一次<br>' +
 '③ 首次见到才写表：<span class="v">keys[pos] = node; used[pos] = 1; use_counts[pos] = 0;</span><br>' +
 '④ 槽位同时是 grads[] / grad_accs[] / use_counts[] 的下标</div>';
const m = w.querySelector('#m');
const T = ['看第 2 幕的 L2：被引用两次，却只能进图一次。',
 '槽位已用 -> <span class="k">already visited</span> -> 直接 return；否则 L2 会进 leafs[] 两次、被算两遍。',
 '首次见到才写表，顺手拿到"图内编号"：槽位就是 grads / use_counts 的下标。',
 '图的语义是"每个张量算一次"：重复入列费时间，也会让 use_counts 的账算错。'];
tl.at(600, () => { R[0].style.opacity = '1'; m.innerHTML = T[0]; });
tl.at(3600, () => { R[1].style.opacity = '1'; m.innerHTML = T[1]; });
tl.at(6600, () => { R[3].style.opacity = '1'; m.innerHTML = T[2]; });
tl.at(9600, () => { R[4].style.opacity = '1'; m.innerHTML = T[3]; });
tl.at(12600, () => { R.forEach(r => { r.style.opacity = '1'; }); m.innerHTML = T[3]; });
tl.at(15400, () => { R.forEach(r => { r.className = ''; }); R[4].className = 'on';
  m.innerHTML = '一句话：<span class="k">状态在图上（used 位），动作在遍历里（return）</span>。'; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L1-03 · 递归',
    title='<span class="hl-c">src[]</span> 就是边：递归只走这一个数组',
    sub='沿 src[0..9] 递归；所有输入处理完，才决定这个张量进 leafs[] 还是 nodes[]。',
    caption='回顾 L1-02：GGML_OP_NONE 且无 PARAM 标志的是常量（叶子）；其余才是要算的节点。',
    src=SRC, parts=[(7265, 7302)], duration=22000,
    notes={0: 'src[] 的长度上限 GGML_MAX_SRC = 10（见 L1-01）',
           11: 'use_counts 也按 hash 槽位索引'},
    marks=[0, 3, 7, 9, 12, 17, 25, 26, 28, 34, 35],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
w.innerHTML = `<div class="row" id="c" style="gap:9px"></div><div class="col" id="ph" style="gap:4px"></div><div class="formula" id="m"></div>`;
root.appendChild(w);
const h = w.querySelector('#c');
const D = [
 { c: 'c', t: 'src[] 是唯一的边', m: 'for (i = 0; i < GGML_MAX_SRC; ++i)',
   b: 'order 只决定按 i 还是按 GGML_MAX_SRC-1-i 走，两个方向都是"先子后父"。' },
 { c: 'e', t: '叶 vs 节点', m: 'op == GGML_OP_NONE && !(flags & PARAM)',
   b: '不满足 -> leafs[]；满足（含带 PARAM 的权重）-> nodes[]。' }];
const E = D.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m }, { style: 'width:300px' });
  h.appendChild(e); return e; });
E.forEach(e => { e.style.opacity = '.32'; });
const ph = w.querySelector('#ph');
const Q = ['① 标记：keys[pos] = node，used[pos] = 1，use_counts[pos] = 0',
 '② 递归每个非空 src[k]',
 '③ 计数：use_counts[src_pos]++',
 '④ 入列：常量进 leafs[]，其余进 nodes[]'].map(s => {
   const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
   e.innerHTML = '<span class="m">' + s + '</span>'; e.style.opacity = '.25';
   ph.appendChild(e); return e; });
const m = w.querySelector('#m');
const T = ['每个节点固定做四件事，顺序不能换。',
 '★ 只走 <span class="v">src[]</span>：边就是这些指针；换 order 只换递归方向。',
 '第 ③ 步的计数供融合判定使用（ggml_node_get_use_count）。',
 '第 ④ 步的分流靠 <span class="v">op</span> 与 <span class="v">flags</span> —— L1-02 的身份在这里第一次派上用场。',
 '入列前都有断言 <span class="v">n_nodes / n_leafs &lt; size</span>：图是定长 arena，装不下就炸，不扩容。'];
tl.at(600, () => { Q[0].style.opacity = '1'; m.innerHTML = T[0]; });
tl.at(3600, () => { E[0].style.opacity = '1'; Q[1].style.opacity = '1'; m.innerHTML = T[1]; });
tl.at(6600, () => { Q[2].style.opacity = '1'; m.innerHTML = T[2]; });
tl.at(9600, () => { E[1].style.opacity = '1'; Q[3].style.opacity = '1'; m.innerHTML = T[3]; });
tl.at(12600, () => { E.forEach(e => { e.style.opacity = '1'; }); Q.forEach(e => { e.style.opacity = '1'; }); m.innerHTML = T[4]; });
tl.at(15600, () => { m.innerHTML = '四步连起来：<span class="k">标记 -> 递归 -> 计数 -> 入列</span>。'; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L1-03 · 核心',
    title='★ 标记位属于<span class="hl-a">图</span>，不属于张量',
    sub='struct ggml_hash_set 只有三个成员：容量、指针数组、位图 —— 去重状态全在这里。',
    caption='对照 L1-01 的 struct ggml_tensor：12 个字段里没有"我在图上被访问过"的标记。位图实现见 source.md 第一节。',
    src=IMPL, parts=[(235, 242)], duration=24000,
    notes={5: '位图：第 i 位 = 1 表示槽位 i 已占用'},
    marks=[0, 1, 3, 4, 5, 7],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
w.innerHTML = `<div class="row" id="c" style="gap:9px"></div><div class="formula" id="m"></div><div class="row" id="x" style="gap:9px"></div>`;
root.appendChild(w);
const h = w.querySelector('#c');
const D = [
 { c: 'a', t: 'size', m: 'size_t size;', b: '槽位数。建图时按 size * 2 申请 —— 要同时装 nodes 和 leafs。' },
 { c: 'c', t: 'used[]', m: 'ggml_bitset_t * used;', b: '位图。注释：whether or not the keys are in use i.e. set' },
 { c: 'b', t: 'keys[]', m: 'struct ggml_tensor ** keys;', b: '槽位里的张量指针；注释：keys[i] 只在 used 位为 1 时有定义。' }];
const E = D.map(d => { const e = U.card(d, { style: 'width:220px' }); h.appendChild(e); return e; });
E.forEach(e => { e.style.opacity = '.3'; });
w.querySelector('#x').innerHTML =
 '<div class="formula" style="line-height:1.6">' +
 '<span class="e">若标记打在张量上</span>：同一批张量指针被多张图引用（ggml_graph_view / ggml_graph_dup），' +
 '标记必须每张图各清一次，漏清就"少算一个节点"。<br>' +
 '<span class="b">实际做法</span>：visited_hash_set 是 struct ggml_cgraph 的成员，清图只需重置计数器和这块位图。' +
 '</div>';
const m = w.querySelector('#m');
const T = ['三个成员就够：容量、位图、指针数组。',
 '<span class="v">keys[]</span> 与 <span class="v">used[]</span> 是一对：源码注释写明 keys[i] 只在 used 位为 1 时有效。',
 '★ 关键：它是 <span class="k">cgraph 的成员</span>，不是张量的成员 —— 状态留在图里，<span class="k">遍历不碰张量本身</span>。',
 '验收点的答案：<span class="k">状态属于哪张图，就存在哪张图里</span>。hash set 不只是"快"，它把 visited 放进了正确的容器。'];
tl.at(600, () => { E[0].style.opacity = '1'; m.innerHTML = T[0]; });
tl.at(3600, () => { E[1].style.opacity = '1'; E[2].style.opacity = '1'; m.innerHTML = T[1]; });
tl.at(6600, () => { E.forEach(e => { e.style.opacity = '1'; }); m.innerHTML = T[2]; });
tl.at(10600, () => { m.innerHTML = T[3]; });
tl.at(14600, () => { m.innerHTML = '一张权重会同时出现在很多张图里（切片、复制、每次 decode 重建）；visited 打在它身上 = 所有图共用一份脏状态。'; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L1-03 · 实现',
    title='<span class="hl-c">ggml_hash_find</span> / <span class="hl-c">ggml_hash_insert</span>：线性探测',
    sub='哈希函数只是把指针右移 4 位；冲突就 +1 找下一格，用 used[] 判断占用。',
    caption='find 返回"命中位置"或"该插入的位置"；insert 撞到已存在的 key 返回 ALREADY_EXISTS，表满则 abort。',
    src=IMPL, parts=[(266, 310)], duration=24000,
    marks=[1, 5, 8, 11, 14, 25, 31, 32, 37],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
w.innerHTML = `<div class="row" style="gap:12px"><div class="col" id="t" style="gap:6px;width:326px"></div><div class="col grow" id="s" style="gap:6px"></div></div><div class="formula" id="m"></div>`;
root.appendChild(w);
const h = w.querySelector('#t');
h.innerHTML = '<div class="cm" style="margin:0">槽位示意（真实容量是质数，见 source.md 第二节）</div>' +
 '<div class="row" id="kr" style="gap:4px"></div><div class="row" id="ur" style="gap:4px"></div>' +
 '<div class="cm" style="font-size:9px">上排 = keys[槽位]，下排 = used[槽位]</div>';
const kr = h.querySelector('#kr'), ur = h.querySelector('#ur'), K = [], Z = [];
for (let i = 0; i < 8; i++) {
  const a = U.el('div', { class: 'formula', style: 'width:36px;padding:5px 0;text-align:center;font-size:10px' });
  a.innerHTML = '<span class="m">·</span>'; kr.appendChild(a); K.push(a);
  const b = U.el('div', { class: 'formula', style: 'width:36px;padding:3px 0;text-align:center;font-size:9px' });
  b.innerHTML = '<span class="m">0</span>'; ur.appendChild(b); Z.push(b);
}
function slot(i, v, c) { K[i].innerHTML = '<span class="' + c + '">' + v + '</span>';
  Z[i].innerHTML = '<span class="' + c + '">1</span>'; }
function fo(i, on) { K[i].style.borderColor = Z[i].style.borderColor = on ? 'var(--c)' : 'var(--border)'; }
w.querySelector('#s').innerHTML =
 '<div class="formula" style="line-height:1.6">' +
 '哈希函数：<span class="v">return (size_t)(uintptr_t)p &gt;&gt; 4;</span> —— 注释说张量按对齐分配，低 4 位恒为 0。<br>' +
 'find 的循环条件把两件事写在一起：<span class="v">used[i] &amp;&amp; keys[i] != key</span> —— 空槽就停，绕回起点返回 FULL。<br>' +
 'insert：空槽则置 used 位、写 keys[i]、返回下标；撞到同一个 key 返回 <b>GGML_HASHSET_ALREADY_EXISTS</b>。' +
 '</div>';
const m = w.querySelector('#m');
const T = ['把表画成 8 格：insert(A) 落在 hash(A) % 8 = 5。',
 'insert(B) 也算出 5，但槽里已有 A —— 冲突，i = (i + 1) % 8 试下一格，6 空，放下。',
 'find(A) 从 5 开始：used[5] 为 1 且 keys[5] == A，立刻返回 5。find(C) 则一路 +1，绕回起点返回 FULL。',
 '八格里每个张量最多一个位置 —— <span class="k">"只入列一次"的物理保证</span>，槽位还兼作图内编号。'];
tl.at(600, () => { slot(5, 'A', 'v'); m.innerHTML = T[0]; });
tl.at(3900, () => { slot(6, 'B', 'v'); fo(5, true); fo(6, true); m.innerHTML = T[1]; });
tl.at(7600, () => { for (let i = 0; i < 8; i++) { fo(i, false); } fo(5, true); m.innerHTML = T[2]; });
tl.at(11600, () => { fo(6, true); m.innerHTML = T[2]; });
tl.at(14600, () => { for (let i = 0; i < 8; i++) { fo(i, false); } m.innerHTML = T[3]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L1-03 · 视图',
    title='<span class="hl-e">view_src</span> 不是图的边：遍历不走它',
    sub='一次 ggml_view_impl 调用建两条链：src[0] 是计算依赖，view_src 是数据归属。',
    caption='要视图的数据来源时得自己沿 view_src 链上溯（例如融合判定）；图外的 view_src 只允许是常量权重。',
    src=SRC, parts=[(3781, 3796), (7805, 7813)], duration=22000,
    marks=[6, 11, 12, 19, 21, 24],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
w.innerHTML = `<div class="row" style="gap:12px"><div class="col grow" id="a" style="gap:4px"></div><div class="col grow" id="b" style="gap:4px"></div></div><div class="formula" id="m"></div>`;
root.appendChild(w);
function chain(h, title, c, items, tag) {
  h.innerHTML = '<div class="cm" style="margin:0;color:var(--' + c + ')">' + title + '</div>' + items.map((it, i) =>
    (i ? '<div class="formula" style="padding:2px 6px;font-size:9px"><span class="m">' + tag + '</span></div>' : '') +
    '<div class="formula" style="padding:4px 7px;font-size:10px"><span class="' + c + '">' + it + '</span></div>').join('');
  h.style.opacity = '.3';
}
const a = w.querySelector('#a'), b = w.querySelector('#b');
chain(a, '计算链：拓扑遍历走这条', 'k', ['out', 'v (GGML_OP_VIEW)', 'base'], 'src[0]');
chain(b, '数据链：遍历不走这条', 'e', ['v (GGML_OP_VIEW)', 'base'], 'view_src');
const m = w.querySelector('#m');
const T = ['ggml_view_impl 一次调用建两条链。',
 '<span class="v">src[0] = a</span>（第 3793 行）：<span class="k">计算依赖</span> —— 拓扑遍历走这条。',
 '<span class="v">view_src = a</span>（第 3787 行，ggml_new_tensor_impl 第 5 个实参）：<span class="k">数据归属</span> —— v 借 a 的内存 + view_offs。',
 '遍历只跟 <span class="v">node-&gt;src[k]</span>：融合判定得自己沿 view_src 链上溯，并要求图外来源是常量权重。',
 '一句话：<span class="k">src[] 决定怎么算，view_src 决定内存归谁</span>；只有前者是图的边。'];
tl.at(600, () => { a.style.opacity = '1'; m.innerHTML = T[0]; });
tl.at(3600, () => { m.innerHTML = T[1]; });
tl.at(6600, () => { b.style.opacity = '1'; m.innerHTML = T[2]; });
tl.at(10600, () => { m.innerHTML = T[3]; });
tl.at(14600, () => { m.innerHTML = T[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L1-03 · 收束',
    title='收束：这张图归谁所有',
    sub='左表把本课压成六行；右边源码说明这些 C 结构在 C++ 侧由 unique_ptr 托管。',
    caption='下一课 L1-04 讲张量内层的量化块布局；图的执行顺序在 L4-04 的 graph_compute 里被消费。',
    src=CPP, parts=[(13, 21)], duration=20000,
    marks=[0, 4, 5, 7, 8],
    visual='''
const w = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
w.innerHTML = `<div id="t"></div><div id="e"></div><div class="formula" id="m"></div>`;
root.appendChild(w);
const t = U.table(['概念', '在哪', '一句话'],
 [['后序 DFS', 'ggml_visit_parents_graph', '父在子之后入列，所以 nodes[] 就是拓扑序'],
  ['去重', 'visited_hash_set.used', '第二次到达直接 return，只把 use_counts 加一'],
  ['hash set', 'keys[] + used[]', '标记位属于图，不属于张量'],
  ['容量', 'ggml_hash_size(size * 2)', '质数表 + 线性探测；nodes 与 leafs 共用一张表'],
  ['视图', 'src[0] vs view_src', '遍历走 src[]，view_src 是数据链'],
  ['宿主', 'ggml_context / unique_ptr', '图住在 ctx 的 arena 里（L1-05），C++ 侧用 RAII 托管']],
 { monoCols: [1] });
w.querySelector('#t').appendChild(t.el);
const R = t.body.querySelectorAll('tr');
w.querySelector('#e').appendChild(W.exercise(
 '权重 W 被 <span class="mono">mul_mat(A, W)</span> 与 <span class="mono">mul_mat(B, W)</span> 两个节点引用。遍历到第二个节点时 <span class="mono">ggml_visit_parents_graph</span> 对 W 做什么？若改成在 ggml_tensor 上打 visited 标志位，会坏在哪里？',
 '第一次到达：<span class="mono">ggml_hash_find()</span> 给出空槽 -> 写 <span class="mono">keys[pos] = W</span>、置 <span class="mono">used[pos] = 1</span>、清 <span class="mono">use_counts[pos]</span>，再按 op/flags 进 leafs[] 或 nodes[]。' +
 '第二次到达：<span class="mono">used[pos]</span> 已是 1 -> "already visited" 分支直接 return，W 不再入列，只在调用点 <span class="mono">use_counts[pos]++</span>。<br><br>' +
 '换成张量上的标志位，坏在两处：<br>' +
 '(1) <b>状态放错容器</b>：张量不属于任何一张图 —— graph_view 与原图共用 visited_hash_set，graph_dup + graph_cpy 让新图把 keys 重插进自己的表。' +
 '标记打在张量上，这些图互相污染、每次重建都要清；used 位图属于 cgraph，<span class="mono">ggml_graph_clear()</span> 一次重置。<br>' +
 '(2) <b>拿不到图内编号</b>：grads[] / grad_accs[] / use_counts[] 都按 hash 槽位索引（注释：indexed by hash table slot），' +
 '标记位给不出这个稳定下标，而 find/insert 返回的槽位正好两件事一起做。'));
const m = w.querySelector('#m');
const T = ['六件事，压成一张表。',
 '<span class="k">顺序</span>：后序 DFS 让"父在子之后"成为构造性事实，不需要事后再排序。',
 '<span class="k">去重</span>：hash set 的 used 位是"这张图收过它"的唯一真相。',
 '<span class="k">状态归属</span>：图的状态在图上（visited_hash_set / use_counts），张量始终是纯数据。',
 '<span class="k">宿主</span>：cgraph 和它的所有数组都住在 ggml_context 里；C++ 侧用 unique_ptr 管 ctx 与 backend 的生命周期。',
 '下一课 L1-04 换个尺度：进到张量内部，看量化块（block_q4_0 等）的字节布局。'];
tl.at(600, () => { m.innerHTML = T[0]; });
R.forEach((r, i) => tl.at(2400 + i * 2300, () => {
  R.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; }); m.innerHTML = T[Math.min(i + 1, 5)]; }));
tl.at(16800, () => { R.forEach(x => { x.className = ''; }); m.innerHTML = T[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、used[] 位图：去重状态长什么样',
    '`ggml_hash_set` 的 `used` 是一张位图：每个 `ggml_bitset_t` 是 `uint32_t`，'
    '`BITSET_SHR = 5` 表示"每 32 个槽位一个元素"。三条位运算分别负责问、置、清 —— '
    '**图的全部 visited 状态就是这一块可以整块 `memset` 的位图**。'
    '这也是"为什么不用张量标记位"的第一层答案：状态是一个可以一次性清空的连续块，'
    '而不是散落在成千上万个张量里的字段。',
    src=IMPL, parts=[(211, 231)], lang='c')

L.section(
    '二、hash set 的分配与容量策略',
    '表本身只有两个数组：`keys`（指针）与 `used`（位图），`ggml_hash_set_new()` 一次把两者建好。'
    '容量不走"两倍扩容"，而是查一张写死的质数表：注释说明这些数是 "next primes after powers of two"，'
    '二分找出第一个不小于需求的值；如果需求超出表长，就退化成 `min_sz | 1`（奇数）。'
    '建图时传给它的需求是 `size * 2` —— 因为一张表要同时装下 nodes 与 leafs（段落末尾的引用给出注释原文）。'
    '注意注释里的"next primes after powers of two"是有道理的：质数取模能让线性探测的聚集更均匀，'
    '而当前实现是"建图时定容、表满即 abort"，不做任何扩容。',
    src=SRC, parts=[(6613, 6620), (6631, 6655), (7482, 7483)], lang='c')

L.section(
    '三、为什么标记位不能打在张量上',
    '`visited_hash_set` 是 `struct ggml_cgraph` 的成员，所以"见过没有"这份状态跟着图走。'
    '下面的引用给出两条证据。第一条：`ggml_graph_view()` 按值复制整个 `visited_hash_set`，'
    '让一个切片图与原图**共用同一张表**（注释还写明梯度拿不到表所以为 NULL）。'
    '第二条：`ggml_graph_dup()` 在给定 context 里建一张新图，`ggml_graph_cpy()` 把在用的 key '
    '**重新插进目标图自己的表**，并逐个重新发号。'
    '于是同一批 `ggml_tensor *` 可以同时属于多张图：切片图与原图**共享**同一张表，'
    '复制出来的图则各有一张表、各有各的槽位编号。'
    '无论哪种情况，"这张图见过谁"都是**图的状态**而不是张量的状态 —— '
    '这正是标记位不能写在张量上的原因。',
    src=SRC, parts=[(7526, 7542), (7561, 7567), (7591, 7595)], lang='c')

L.section(
    '四、struct ggml_cgraph 的全部字段',
    '图的定义本身在 `ggml-impl.h` 里：三个计数器、五个数组指针，一个 hash set，一个遍历顺序，'
    '外加一个可选的身份号 `uid`（注释说明 0 表示未设置）。'
    '注意 `use_counts` 的注释：**indexed by hash table slot** —— 使用次数不是按节点下标存的，'
    '而是按哈希槽位存的，这就是 hash set 顺带提供的"图内编号"。'
    '顺带一个诚实的观察：`ggml_graph_view()` 里的指示初始化注释写的是 `/*.visited_hash_set =*/`，'
    '而 `ggml_new_graph_custom()` 里同一条注释写的是 `/*.hash_table =*/` —— 后者是**过时的注释**'
    '（结构体里早已没有 `hash_table` 这个字段），说明源码注释也会滞后，一切以结构体定义为准。',
    src=IMPL, parts=[(341, 359)], lang='c')

L.section(
    '五、图的存储与清空',
    '图的字节数由 `ggml_graph_nbytes()` 算死，`ggml_new_graph_custom()` 再按同一顺序切出来；'
    '数组紧密排在结构体之后。于是"清空一张图"不需要释放任何东西 —— '
    '`ggml_graph_clear()` 把两个计数器归零、把 `used` 位图整块清零，图就回到刚建好的状态。'
    '这也解释了 `ggml_build_forward_impl()` 里 `expand` 参数的含义：'
    '`expand = true` 时**不清图**，可以在已有内容上继续追加（第 2 幕源码第 7305 行）。',
    src=SRC, parts=[(7452, 7463), (7646, 7650)], lang='c')

L.section(
    '六、C++ 包装层：谁负责释放',
    '`ggml-cpp.h` 不是新的数据结构，只是一层 C++ 智能指针别名：为每种 C 句柄写一个 deleter，'
    '再用 `std::unique_ptr` 包起来。它对图本身没有影响，但解释了 `ggml_context` 这个 arena 的宿主是谁 —— '
    '图、张量、图里的 hash set 都随 ctx 一起释放。'
    '文件开头的 `#error "This header is for C++ only"` 也说明它是给 C++ 调用方用的便利层。',
    src=CPP, parts=[(25, 39)], lang='c')

L.footnote_add('本课声明 3 个源文件：`ggml/src/ggml.c`（主）、`ggml/src/ggml-impl.h`'
               '（`struct ggml_cgraph` 与 `struct ggml_hash_set` 的定义与实现所在）、'
               '`ggml/include/ggml-cpp.h`（C++ RAII 包装层）。三者都在覆盖域内，均计入覆盖率。')
L.footnote_add('`ggml_graph_plan()` / `ggml_graph_compute()` 这一版不在 `ggml.c` 里'
               '（它们属于 CPU 后端的 `ggml-cpu`）。本课只说"顺序被后端按 nodes[] 消费"，'
               '不引用其源码，具体展开见 L4-04。')
L.footnote_add('GGML_DEFAULT_GRAPH_SIZE（默认图容量）定义在 `ggml/include/ggml.h`，'
               '该文件由 L1-01 覆盖；本课不引用其行号，故不计入本课声明。')
L.footnote_add('工程备注（历史）：本课写作期间 `tools/check_flags.py` 的 `strip_verbatim()` '
               '把 lesson.js 的**内容**当路径传给 `dump_scenes.js`，node 回显整份内容到 stderr 并在 '
               '65536 字节处截断（实测 `stderr = 2 x 文件字节 + 51`）；截断点落在多字节字符中间时 '
               'Python 的 strict 解码会抛 `UnicodeDecodeError`，整个参数门禁失败。'
               '该缺陷已由 3385a08 修复（现在传的是路径，且解码用 `errors="replace"`、不再静默降级）。'
               '缺陷存在期间本课 `visual` 写得紧凑、把 lesson.js 压到 32 KB 上下；修复后这个限制已消失，'
               '较长的解释仍留在 `source.md`。')

L.prereqs('`L1-01`、`L1-02`')

L.goal(
    '说出 `ggml_build_forward_expand()` 展开出的 `nodes[]` 为什么天然是拓扑序（后序 DFS + 断言）；',
    '解释 `visited_hash_set` 如何用 `keys[] + used[]` 同时解决"去重"和"图内编号"两件事（对应验收点）；',
    '论证为什么 visited 标记**不能**打在 `ggml_tensor` 上（张量跨图复用、清标记、拿不到稳定槽位）；',
    '区分视图的两条链：`src[0]`（计算依赖，遍历会走）与 `view_src`（数据归属，遍历不走）；',
    '说出 `ggml_graph_clear()` 为什么只要重置计数器和 `used` 位图。')

L.conclusion(
    '★ 顺序：后序 DFS 让 nodes[] 直接可用',
    '`ggml_visit_parents_graph()` 是后序遍历：先递归所有 `src[]`，再把当前张量追加进 `leafs[]` 或 `nodes[]`。'
    '所以"输入永远排在使用它的节点之前"是构造性事实，不需要事后再排序。'
    '源码用一条断言把它钉死：\n\n```text\n本次最后入列的节点 == 本次的起点张量\n```\n\n'
    '后端（L4-04）按 `nodes[0], nodes[1], ...` 顺序派发内核即可。')

L.conclusion(
    '★ 去重：先查表，再插入',
    '每个张量在图上只占一个槽位。第二次到达同一个张量时，`used` 位已经是 1，'
    '遍历走 "already visited" 分支直接返回 —— 不再入列，只在调用点把 `use_counts` 加一。'
    '但这一支并非完全空转：`compute` 为真时它仍会递归给 `src[]` 补上 `GGML_TENSOR_FLAG_COMPUTE`，'
    '因为"标记为要算"和"入列一次"是两件事。'
    '如果不去重：同一张量会进 `nodes[]`/`leafs[]` 两次、被执行两遍，'
    '`use_counts` 与梯度累加的账也会算错。')

L.conclusion(
    '★ 为什么是 hash set 而不是标记位',
    '答案不是"hash 更快"，而是**状态归谁**：\n\n'
    '| 维度 | 张量上的标记位 | 图上的 hash set |\n|---|---|---|\n'
    '| 状态归属 | 属于张量，被所有图共享 | 属于 `cgraph`（`visited_hash_set`） |\n'
    '| 多图复用 | 两张图互相污染 | 每张图各一份；`graph_view` 可显式共享 |\n'
    '| 清空 | 必须逐个张量清 | `ggml_graph_clear()` 整块 `memset` |\n'
    '| 图内编号 | 给不出 | 槽位就是 `use_counts` / `grads` / `grad_accs` 的下标 |\n\n'
    '同一批 `ggml_tensor *` 会同时出现在多张图里（`ggml_graph_view` / `ggml_graph_dup`），'
    '而 `ggml_hash_set` 只有 3 个成员、容量取质数、冲突用线性探测 —— 结构简单，却把 visited 状态'
    '放在了正确的容器里。')

L.conclusion(
    '视图：src[] 是计算链，view_src 是数据链',
    '`ggml_view_impl()`（`ggml_view_1d/2d/3d/4d` 的共同实现）在一次调用里建两条链：\n\n'
    '```text\n'
    'ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset)   -> view_src = a, view_offs = offset\n'
    'result->src[0] = a                                          -> 计算依赖\n'
    '```\n\n'
    '拓扑遍历只跟 `src[]`，所以视图在图上是个普通节点；'
    '需要"视图的数据从哪来"的地方（例如子图融合判定）必须自己沿 `view_src` 链上溯，'
    '并要求图外的来源是常量权重（见第 7 幕引用的源码）。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
