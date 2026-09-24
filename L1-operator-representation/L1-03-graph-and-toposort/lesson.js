/* ==========================================================================
   L1-03 · 计算图与拓扑遍历：节点怎么排成一队
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 从张量到图：这一课看的是<span class="hl-a">遍历</span> */
{
  kicker: "L1 · 算子的表示",
  title: "从张量到图：这一课看的是<span class=\"hl-a\">遍历</span>",
  sub: "L1-01 给了节点（tensor），L1-02 给了身份（op）；本课看 ggml 如何把它们串成一张有序、可执行的图。",
  caption: "跨课：src[] 是图的边（L1-01）；op 决定这个张量算节点还是叶子（L1-02）；这里排好的顺序最终被 L4-04 的 graph_compute 消费。",
  src: "ggml/src/ggml.c",
  mark: [1],
  lineNo: 7337,
  code: `void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}`,
  duration: 17000,
  build(root, tl) {
    const w = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    w.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip">张量 + op</span><span class="arrow">-></span><span class="chip a">ggml_build_forward_expand</span><span class="arrow">-></span><span class="chip b">nodes[] 有序数组</span><span class="arrow">-></span><span class="chip c">graph_compute</span></div>
    <div class="row center" id="c" style="gap:9px"></div><div class="formula" id="m"></div>`;
    root.appendChild(w);
    const h = w.querySelector('#c'), m = w.querySelector('#m');
    const D = [
     { c: 'a', t: '输入', m: 'struct ggml_cgraph * cgraph', b: '一张（可能是空的）图 + 一个输出张量' },
     { c: 'c', t: '过程', m: '递归 + hash set', b: '后序 DFS 沿 src[] 展开；hash set 保证每个张量只入列一次' },
     { c: 'b', t: '输出', m: 'cgraph->nodes[0 .. n_nodes)', b: '一个顺序：先算输入，再算用到它们的节点' }];
    const E = D.map(d => { const e = U.card(d, { style: 'width:220px' }); h.appendChild(e); return e; });
    E.forEach(e => e.style.opacity = '.3');
    const T = ['入口只有三行：<span class="v">ggml_build_forward_expand()</span> 把活全部转给内部实现。',
     '它接受 <span class="k">一张图 + 一个输出张量</span>；图本身可能已经有内容（所以叫 expand）。',
     '展开要同时解决两件事：<span class="k">重复</span>（同一张量被多处引用）和 <span class="k">顺序</span>（谁先算）。',
     '回顾 L1-01：图的边就是 <span class="v">src[10]</span> 这些指针，没有单独的边表；<br>回顾 L1-02：<span class="v">op</span> 决定这个张量是"要算的节点"还是"常量叶子"。',
     '产物交给 L4-04：后端按 <span class="v">nodes[]</span> 的顺序逐个派发内核。本课只讲图怎么建。'];
    D.forEach((_, i) => tl.at(700 + i * 3300, () => {
      E.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.3'; }); m.innerHTML = T[i]; }));
    tl.at(13900, () => { E.forEach(e => { e.style.opacity = '1'; }); m.innerHTML = T[3]; });
    tl.at(15600, () => { m.innerHTML = T[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">后序 DFS</span>：nodes[] 出来就是拓扑序 */
{
  kicker: "L1-03 · 排序",
  title: "★ <span class=\"hl-a\">后序 DFS</span>：nodes[] 出来就是拓扑序",
  sub: "父节点在它的所有 src[] 之后入列；源码用一条断言把这个不变量钉死。",
  caption: "n_new > 0 才断言：expand 模式允许对同一张图多次展开，但\"本次最后入列的一定是本次的起点\"永远成立。",
  src: "ggml/src/ggml.c",
  mark: [6, 8, 10, 15],
  lineNo: 7304,
  code: `static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand, bool compute) {
    if (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    const int n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    const int n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}`,
  duration: 27000,
  build(root, tl) {
    const w = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    w.innerHTML = `<div class="row" style="gap:12px"><div class="col" id="d" style="gap:4px;width:246px"></div><div class="col grow" id="p" style="gap:4px"></div></div><div class="formula" id="m"></div>`;
    root.appendChild(w);
    const d = w.querySelector('#d'), p = w.querySelector('#p'), m = w.querySelector('#m');
    const EG = [['N.src[0]', 'L1', 'N'], ['N.src[1]', 'L2', 'N'], ['O.src[0]', 'N', 'O'], ['O.src[1]', 'L2', 'O']];
    d.innerHTML = '<div class="cm" style="margin:0">输出张量 O 的图（边 = src[]）</div>' + EG.map(e =>
     '<div class="formula" style="padding:3px 7px;font-size:10px"><span class="m" style="margin:0">' + e[0] +
     '</span> <span class="k">' + e[1] + '</span> <span class="m">-></span> <span class="v">' + e[2] + '</span></div>').join('') +
     '<div class="cb" style="font-size:9.5px">L2 被 N 和 O 各引用一次 —— "重复"就从这里来。</div>';
    const G = d.querySelectorAll('.formula');
    p.innerHTML = '<div class="cm" style="margin:0">leafs[] · op == GGML_OP_NONE</div><div class="col" id="lf" style="gap:3px"></div>' +
     '<div class="cm" style="margin:4px 0 0">nodes[] · 要算的</div><div class="col" id="nd" style="gap:3px"></div>';
    const lf = p.querySelector('#lf'), nd = p.querySelector('#nd');
    function put(x, t, c) { const e = U.el('div', { class: 'formula', style: 'padding:2px 7px;font-size:10px' });
      e.innerHTML = '<span class="' + c + '">' + t + '</span>'; x.appendChild(e); }
    const S = [
     [700, -1, '', '', '站在输出 <span class="k">O</span> 上：src[0] = N，src[1] = L2。'],
     [3800, 2, '', '', 'dfs(O)：先标记 O，再递归 src[0]。'],
     [6900, 0, 'lf', 'leafs[0] = L1', 'dfs(N)：标记 N，递归 src[0] = L1；L1 的 op 是 NONE，进 <span class="v">leafs[]</span>。'],
     [10000, 1, 'lf', 'leafs[1] = L2', '接着 src[1] = L2，同样进 <span class="v">leafs[]</span>。此时 N 还没入列。'],
     [13100, -1, 'nd', 'nodes[0] = N', '<span class="k">N 的输入全部处理完</span>，才轮到 N 自己进 nodes[]。'],
     [16200, 3, '', '', '回到 O，src[1] 又指向 L2 —— 第二次到达，<span class="k">不能再入列一次</span>。'],
     [19300, -1, 'nd', 'nodes[1] = O', '最后 O 入列：<span class="v">nodes[] = [N, O]</span>，每个节点的输入都排在它前面。'],
     [22400, -1, '', '', '源码第 7319 行把这条不变量写成断言：本次最后入列的，必须正好是本次的起点张量。']];
    S.forEach(s => tl.at(s[0], () => {
      G.forEach((x, k) => { x.style.opacity = (s[1] < 0 || k === s[1]) ? '1' : '.3'; });
      if (s[2] === 'lf') { put(lf, s[3], 'v'); }
      if (s[2] === 'nd') { put(nd, s[3], 'k'); }
      m.innerHTML = s[4];
    }));
  }
},

/* ------------------------------------------------------ 3 ★ 每个张量只入列一次：<span class="hl-a">先查表，再插入</span> */
{
  kicker: "L1-03 · 核心",
  title: "★ 每个张量只入列一次：<span class=\"hl-a\">先查表，再插入</span>",
  sub: "同一张量被多个下游节点引用时，第二次到达必须立刻返回 —— 否则它会在 nodes[]/leafs[] 里出现两次。",
  caption: "第二次到达也不是完全空转：compute 为真时它仍会递归给 src[] 补上 COMPUTE 标志；去重针对的是\"入列\"。",
  src: "ggml/src/ggml.c",
  mark: [5, 6, 8, 10, 22, 27, 28, 29],
  lineNo: 7236,
  code: `static size_t ggml_visit_parents_graph(struct ggml_cgraph * cgraph, struct ggml_tensor * node, bool compute) {
    if (node->op != GGML_OP_NONE && compute) {
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    const size_t node_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
    GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);

    if (ggml_bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
//>> used[pos] 已经是 1 = 这张图收过它了 —— 直接 return，不再递归、不再入列
        // already visited

        if (compute) {
            // update the compute flag regardless
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                struct ggml_tensor * src = node->src[i];
                if (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        return node_hash_pos;
    }

//>> 首次见到：先把指针写进 keys[]，再把 used 位置 1，最后清 use_counts
    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.keys[node_hash_pos] = node;
    ggml_bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
    cgraph->use_counts[node_hash_pos] = 0;`,
  duration: 24000,
  build(root, tl) {
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
     '<div class="card" style="border-left-color:var(--a)"><div class="ct" style="color:var(--a)">第一步：算槽位</div>' +
     '<div class="cb"><span class="cm" style="margin:0">ggml_hash_find(&amp;cgraph-&gt;visited_hash_set, node)</span> 返回槽位下标；全满返回 GGML_HASHSET_FULL，遍历里紧跟断言挡住。</div></div>' +
     '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">第二步：看 used 位</div>' +
     '<div class="cb">为 1 = 本图收过它 —— 直接 return。同一个 L2 被 N 和 O 各引用，入列只一次。</div></div>' +
     '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">首次见到：三个写操作</div>' +
     '<div class="cb"><span class="cm" style="margin:0">keys[pos] = node; used[pos] = 1; use_counts[pos] = 0;</span> 槽位同时是 grads[] / grad_accs[] / use_counts[] 的下标。</div></div>';
    const m = w.querySelector('#m');
    const T = ['把第 2 幕的 L2 单独拎出来看：它被两个节点引用，却只能进图一次。',
     '<span class="v">ggml_hash_find()</span> 先回答"它在不在表里、在哪个槽"。',
     '槽位已用 -> <span class="k">already visited</span> -> 直接 return。不返回的话 L2 会进 leafs[] 两次、被算两遍。',
     '首次见到才写表：<span class="v">keys[pos] = node; used[pos] = 1; use_counts[pos] = 0;</span>',
     '图的语义是"每个张量算一次"：重复入列既浪费执行时间，也会让 use_counts 与梯度累加的账算错。'];
    tl.at(600, () => { R[0].style.opacity = '1'; m.innerHTML = T[0]; });
    tl.at(3600, () => { R[1].style.opacity = '1'; m.innerHTML = T[1]; });
    tl.at(6600, () => { R[3].style.opacity = '1'; m.innerHTML = T[2]; });
    tl.at(9600, () => { R[4].style.opacity = '1'; m.innerHTML = T[3]; });
    tl.at(12600, () => { R.forEach(r => { r.style.opacity = '1'; }); m.innerHTML = T[4]; });
    tl.at(15600, () => { R.forEach(r => { r.className = ''; }); R[4].className = 'on';
      m.innerHTML = '一句话：<span class="k">去重的状态在图上（used 位），动作在遍历里（return）</span> —— 张量自己什么都不知道。'; });
  }
},

/* ------------------------------------------------------ 4 <span class="hl-c">src[]</span> 就是边：递归只走这一个数组 */
{
  kicker: "L1-03 · 递归",
  title: "<span class=\"hl-c\">src[]</span> 就是边：递归只走这一个数组",
  sub: "沿着 src[0..9] 递归；等所有输入都处理完，才决定这个张量进 leafs[] 还是 nodes[]。",
  caption: "回顾 L1-02：GGML_OP_NONE 且没有 PARAM 标志的张量是常量（叶子）；其余才是要算的节点。",
  src: "ggml/src/ggml.c",
  mark: [0, 3, 7, 9, 12, 17, 25, 26, 28, 34, 35],
  lineNo: 7265,
  code: `    for (int i = 0; i < GGML_MAX_SRC; ++i) {
//>> src[] 的长度上限 GGML_MAX_SRC = 10（见 L1-01）
        const int k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        struct ggml_tensor * src = node->src[k];
        if (src) {
            const size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
//>> use_counts 也按 hash 槽位索引：一次遍历顺手记下"每个张量被用了几次"
        }
    }

    if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        if (strlen(node->name) == 0) {
            ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        if (strlen(node->name) == 0) {
            ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->n_nodes++;
    }

    return node_hash_pos;
}`,
  duration: 22000,
  build(root, tl) {
    const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    w.innerHTML = `<div class="row" id="c" style="gap:9px"></div><div class="col" id="ph" style="gap:4px"></div><div class="formula" id="m"></div>`;
    root.appendChild(w);
    const h = w.querySelector('#c');
    const D = [
     { c: 'c', t: 'src[] 是唯一的边', m: 'for (i = 0; i < GGML_MAX_SRC; ++i)',
       b: 'order 只决定按 i 还是按 GGML_MAX_SRC-1-i 走；两个方向都满足"先子后父"。' },
     { c: 'e', t: '叶 vs 节点', m: 'op == GGML_OP_NONE && !(flags & PARAM)',
       b: '不满足就是叶子 -> leafs[]；满足（包括带 PARAM 的权重）-> nodes[]。' }];
    const E = D.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m }, { style: 'width:300px' });
      h.appendChild(e); return e; });
    E.forEach(e => { e.style.opacity = '.32'; });
    const ph = w.querySelector('#ph');
    const Q = ['① 标记自己：keys[pos] = node，used[pos] = 1，use_counts[pos] = 0',
     '② 递归输入：对每个非空 src[k] 再进 ggml_visit_parents_graph()',
     '③ 记使用次数：use_counts[src_pos]++',
     '④ 入列：常量进 leafs[]，其余进 nodes[]'].map(s => {
       const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
       e.innerHTML = '<span class="m" style="margin:0">' + s + '</span>'; e.style.opacity = '.25';
       ph.appendChild(e); return e; });
    const m = w.querySelector('#m');
    const T = ['每个节点固定做四件事，顺序不能换。',
     '★ 只走 <span class="v">src[]</span>：图的边没有独立结构，就是这些指针。<br>换 order 只是换"先递归哪个输入"，不改变后序性质。',
     '第 ③ 步给每个输入 +1 次使用计数 —— 融合判定用它判断"结果只有 N 个使用者"（ggml-impl.h 的 ggml_node_get_use_count）。',
     '第 ④ 步的分流全靠 <span class="v">op</span> 和 <span class="v">flags</span>：<br>这是 L1-02 讲的"身份"在图构建阶段的第一个用处。',
     '入列前都有断言 <span class="v">n_nodes / n_leafs &lt; size</span>：图是定长 arena，装不下就直接炸，不会偷偷扩容。'];
    tl.at(600, () => { Q[0].style.opacity = '1'; m.innerHTML = T[0]; });
    tl.at(3600, () => { E[0].style.opacity = '1'; Q[1].style.opacity = '1'; m.innerHTML = T[1]; });
    tl.at(6600, () => { Q[2].style.opacity = '1'; m.innerHTML = T[2]; });
    tl.at(9600, () => { E[1].style.opacity = '1'; Q[3].style.opacity = '1'; m.innerHTML = T[3]; });
    tl.at(12600, () => { E.forEach(e => { e.style.opacity = '1'; }); Q.forEach(e => { e.style.opacity = '1'; }); m.innerHTML = T[4]; });
    tl.at(15600, () => { m.innerHTML = '四步连起来：<span class="k">标记 -> 递归 -> 计数 -> 入列</span>，于是 nodes[] 里每个节点都排在它所有输入之后。'; });
  }
},

/* ------------------------------------------------------ 5 ★ 标记位属于<span class="hl-a">图</span>，不属于张量 */
{
  kicker: "L1-03 · 核心",
  title: "★ 标记位属于<span class=\"hl-a\">图</span>，不属于张量",
  sub: "struct ggml_hash_set 只有三个成员：容量、指针数组、位图。去重所需的状态全在这里。",
  caption: "对照 L1-01（struct ggml_tensor 逐字段）：那 12 个字段里没有任何\"我在图上被访问过\"的标记。used[] 的位运算实现见 source.md 第一节。",
  src: "ggml/src/ggml-impl.h",
  mark: [0, 1, 3, 4, 5, 7],
  lineNo: 235,
  code: `#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

struct ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
//>> 位图：第 i 位为 1 表示槽位 i 已被占用（位运算实现见 source.md 第一节）
    struct ggml_tensor ** keys; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};`,
  duration: 24000,
  build(root, tl) {
    const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    w.innerHTML = `<div class="row" id="c" style="gap:9px"></div><div class="formula" id="m"></div><div class="row" id="x" style="gap:9px"></div>`;
    root.appendChild(w);
    const h = w.querySelector('#c');
    const D = [
     { c: 'a', t: 'size', m: 'size_t size;', b: '槽位数。建图时按 size * 2 申请 —— 要同时装 nodes 和 leafs。' },
     { c: 'c', t: 'used[]', m: 'ggml_bitset_t * used;', b: '位图。源码注释：whether or not the keys are in use i.e. set' },
     { c: 'b', t: 'keys[]', m: 'struct ggml_tensor ** keys;', b: '槽位里的张量指针；源码注释：keys[i] 只在 used 位为 1 时有定义。' }];
    const E = D.map(d => { const e = U.card(d, { style: 'width:220px' }); h.appendChild(e); return e; });
    E.forEach(e => { e.style.opacity = '.3'; });
    const x = w.querySelector('#x');
    const bad = U.card({ c: 'e', t: '如果标记打在张量上', m: 'tensor->visited = 1  (不存在)',
      b: '同一批张量指针会被多张图引用：ggml_graph_view / ggml_graph_dup 会在同一批张量上再建一张图。标记必须每张图各清一次，漏清就会"少算一个节点"。' },
      { style: 'width:300px' });
    const good = U.card({ c: 'b', t: '实际做法：标记打在图上', m: 'cgraph->visited_hash_set.used',
      b: 'visited_hash_set 是 struct ggml_cgraph 的成员。清图只需重置计数器和这块位图（ggml_graph_clear）。' },
      { style: 'width:300px' });
    bad.style.opacity = good.style.opacity = '.3';
    x.appendChild(bad); x.appendChild(good);
    const m = w.querySelector('#m');
    const T = ['三个成员就够：容量、位图、指针数组。位图答"这个槽用没用"，指针数组答"槽里是谁"。',
     '<span class="v">keys[]</span> 与 <span class="v">used[]</span> 是一对：源码注释写明 keys[i] 只在 used 位为 1 时有效。',
     '★ 关键：这个结构是 <span class="k">cgraph 的成员</span>，不是张量的成员。所以"见过没有"的全部状态都留在图里 —— <span class="k">遍历不修改张量本身</span>。',
     '反过来看张量：一张权重会同时出现在很多张图里（切片、复制、每次 decode 重建），把 visited 打在它身上等于让所有图共用一份脏状态。',
     '这就是验收点的答案：<span class="k">状态属于哪张图，就存在哪张图里</span>。hash set 不只是"快"，它把 visited 状态放进了正确的容器。'];
    tl.at(600, () => { E[0].style.opacity = '1'; m.innerHTML = T[0]; });
    tl.at(3600, () => { E[1].style.opacity = '1'; E[2].style.opacity = '1'; m.innerHTML = T[1]; });
    tl.at(6600, () => { E.forEach(e => { e.style.opacity = '1'; }); m.innerHTML = T[2]; });
    tl.at(9600, () => { bad.style.opacity = '1'; m.innerHTML = T[3]; });
    tl.at(12600, () => { good.style.opacity = '1'; m.innerHTML = T[2]; });
    tl.at(15600, () => { m.innerHTML = T[4]; });
  }
},

/* ------------------------------------------------------ 6 <span class="hl-c">ggml_hash_find</span> / <span class="hl-c">ggml_hash_insert</span>：线性探测 */
{
  kicker: "L1-03 · 实现",
  title: "<span class=\"hl-c\">ggml_hash_find</span> / <span class=\"hl-c\">ggml_hash_insert</span>：线性探测",
  sub: "哈希函数只是把指针右移 4 位（对齐保证低 4 位为 0）；冲突就 +1 找下一格，用 used[] 判断占用。",
  caption: "find 返回\"命中位置\"或\"该插入的位置\"；insert 对已存在的 key 返回 GGML_HASHSET_ALREADY_EXISTS，表满则 abort。",
  src: "ggml/src/ggml-impl.h",
  mark: [1, 5, 8, 11, 14, 25, 31, 32, 37],
  lineNo: 266,
  code: `static inline size_t ggml_hash(const struct ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    return (size_t)(uintptr_t)p >> 4;
}

static size_t ggml_hash_find(const struct ggml_hash_set * hash_set, const struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    while (ggml_bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
        i = (i + 1) % hash_set->size;
        if (i == h) {
            // visited all hash table entries -> not found
            return GGML_HASHSET_FULL;
        }
    }
    return i;
}

static bool ggml_hash_contains(const struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t i = ggml_hash_find(hash_set, key);
    return i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

static size_t ggml_hash_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}`,
  duration: 24000,
  build(root, tl) {
    const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    w.innerHTML = `<div class="row" style="gap:12px"><div class="col" id="t" style="gap:6px;width:326px"></div><div class="col grow" id="s" style="gap:6px"></div></div><div class="formula" id="m"></div>`;
    root.appendChild(w);
    const h = w.querySelector('#t');
    h.innerHTML = '<div class="cm" style="margin:0">槽位示意（真实容量是质数，见 source.md 第二节）</div>' +
     '<div class="row" id="kr" style="gap:4px"></div><div class="row" id="ur" style="gap:4px"></div>' +
     '<div class="cm" style="margin:2px 0 0;font-size:9px">上排 = keys[槽位]，下排 = used[槽位]</div>';
    const kr = h.querySelector('#kr'), ur = h.querySelector('#ur'), K = [], Z = [];
    for (let i = 0; i < 8; i++) {
      const a = U.el('div', { class: 'formula', style: 'width:36px;padding:5px 0;text-align:center;font-size:10px' });
      a.innerHTML = '<span class="m" style="margin:0">·</span>'; kr.appendChild(a); K.push(a);
      const b = U.el('div', { class: 'formula', style: 'width:36px;padding:3px 0;text-align:center;font-size:9px' });
      b.innerHTML = '<span class="m" style="margin:0">0</span>'; ur.appendChild(b); Z.push(b);
    }
    function slot(i, v, c) { K[i].innerHTML = '<span class="' + c + '">' + v + '</span>';
      Z[i].innerHTML = '<span class="' + c + '">1</span>'; }
    function fo(i, on) { K[i].style.borderColor = Z[i].style.borderColor = on ? 'var(--c)' : 'var(--border)'; }
    w.querySelector('#s').innerHTML =
     '<div class="card" style="border-left-color:var(--a)"><div class="ct" style="color:var(--a)">哈希函数</div>' +
     '<div class="cb"><span class="cm" style="margin:0">return (size_t)(uintptr_t)p &gt;&gt; 4;</span><br>源码注释：张量按对齐分配，低 4 位恒为 0，直接丢掉。</div></div>' +
     '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">探测是"边走边判"</div>' +
     '<div class="cb">find 的循环条件把两件事写在一起：<span class="cm" style="margin:0">used[i] &amp;&amp; keys[i] != key</span> —— 空槽就停，绕回起点则返回 FULL。</div></div>' +
     '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">插入的结果</div>' +
     '<div class="cb">空槽：置 used 位、写 keys[i]、返回下标。撞到同一个 key：返回 <b>GGML_HASHSET_ALREADY_EXISTS</b>，不重复插入。</div></div>';
    const m = w.querySelector('#m');
    const T = ['把表画成 8 格：insert(A) 落在 hash(A) % 8 = 5。',
     'insert(B) 也算出 5，但槽里已有 A —— 冲突，i = (i + 1) % 8 试下一格，6 空，放下。',
     'find(A)：从 5 开始，used[5] 为 1 且 keys[5] == A，立刻返回 5。',
     'find(C)（hash 也是 5）：5、6 都占用且都不是 C，继续 +1；绕回起点说明全表扫过 -> 返回 GGML_HASHSET_FULL。',
     '八格里每个张量最多一个位置 —— <span class="k">这就是"每个张量只入列一次"的物理保证</span>。',
     '槽位下标同时被 use_counts / grads / grad_accs 复用：这张表不只是"集合"，还是<span class="k">图内编号</span>的发号器。'];
    tl.at(600, () => { slot(5, 'A', 'v'); m.innerHTML = T[0]; });
    tl.at(3900, () => { slot(6, 'B', 'v'); fo(5, true); fo(6, true); m.innerHTML = T[1]; });
    tl.at(7200, () => { for (let i = 0; i < 8; i++) { fo(i, false); } fo(5, true); m.innerHTML = T[2]; });
    tl.at(10500, () => { fo(6, true); m.innerHTML = T[3]; });
    tl.at(13800, () => { for (let i = 0; i < 8; i++) { fo(i, false); } m.innerHTML = T[4]; });
    tl.at(16800, () => { m.innerHTML = T[5]; });
  }
},

/* ------------------------------------------------------ 7 建图：哈希表容量<span class="hl-b">翻倍</span>，整张图是一个 arena 对象 */
{
  kicker: "L1-03 · 存储",
  title: "建图：哈希表容量<span class=\"hl-b\">翻倍</span>，整张图是一个 arena 对象",
  sub: "源码注释直接写了为什么翻倍：it needs to hold both nodes and leafs。",
  caption: "这一整块内存属于 ggml_context 这个 arena（L1-05）；图装不下只会触发断言，不会扩容。",
  src: "ggml/src/ggml.c",
  mark: [0, 1, 3, 5, 6, 7, 8, 12],
  lineNo: 7482,
  code: `    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    void * p = cgraph + 1;

    struct ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
    struct ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t));
    struct ggml_tensor ** hash_keys_ptr  =         incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
    struct ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)) : NULL;
    struct ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)) : NULL;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));`,
  duration: 20000,
  build(root, tl) {
    const w = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    w.innerHTML = `<div class="cm" style="margin:0">ggml_new_graph_custom()：一次分配，按顺序切出所有数组</div>
    <div class="row wrap" id="st" style="gap:5px"></div><div class="row" id="n" style="gap:9px"></div><div class="formula" id="m"></div>`;
    root.appendChild(w);
    const P = [
     ['cgraph', 'struct', 'a', '计数 + 各数组指针'],
     ['nodes[size]', 'size', 'c', '要算的节点，按拓扑序'],
     ['leafs[size]', 'size', 'c', '常量张量'],
     ['use_counts', 'hash_size', 'd', '按 hash 槽位索引'],
     ['keys[hash_size]', 'hash_size', 'e', 'hash set 的指针数组'],
     ['grads / grad_accs', 'hash_size', 'f', '仅训练图分配'],
     ['used 位图', 'hash_size/32', 'b', 'hash set 的位图']];
    const h = w.querySelector('#st');
    const E = P.map(p => { const e = U.el('div', { class: 'formula', style: 'width:92px;padding:4px;font-size:9px;text-align:center' });
      e.innerHTML = '<div style="color:var(--' + p[2] + ');font-size:9.5px">' + p[0] + '</div>' +
        '<div class="m" style="margin:0;font-size:8.5px">' + p[1] + '</div>' +
        '<div class="cb" style="font-size:8.5px;line-height:1.3">' + p[3] + '</div>';
      e.style.opacity = '.3'; h.appendChild(e); return e; });
    w.querySelector('#n').innerHTML =
     '<div class="card" style="border-left-color:var(--a);width:300px"><div class="ct" style="color:var(--a)">为什么是 size * 2</div>' +
     '<div class="cb">源码注释：the size of the hash table is doubled since it needs to hold both nodes and leafs —— 两类张量共用一张表。</div></div>' +
     '<div class="card" style="border-left-color:var(--b);width:300px"><div class="ct" style="color:var(--b)">为什么必须是质数</div>' +
     '<div class="cb">容量走 ggml_hash_size()：取质数表里第一个 &gt;= 需求的值；超出表长则退化为 <span class="cm" style="margin:0">min_sz | 1</span>。</div></div>';
    const m = w.querySelector('#m');
    const T = ['建图不是"new 一个 cgraph 再说"：先算总字节数，再一次从 arena 里切。',
     '容量按 <span class="v">size * 2</span> 算 —— 一张哈希表同时装节点和叶子。',
     '然后按固定顺序切：nodes、leafs、use_counts、keys、grads、grad_accs，最后是 used 位图。',
     '注意 <span class="v">p = cgraph + 1</span>：所有数组紧跟在结构体后面，没有一次单独 malloc。',
     '所以"清空一张图"很便宜：计数归零 + used 位图整块清零（ggml_graph_clear，见 source.md 第五节）。'];
    tl.at(600, () => { E[0].style.opacity = '1'; m.innerHTML = T[0]; });
    tl.at(3400, () => { E[3].style.opacity = E[4].style.opacity = E[6].style.opacity = '1'; m.innerHTML = T[1]; });
    tl.at(6600, () => { E.forEach(e => { e.style.opacity = '1'; }); m.innerHTML = T[2]; });
    tl.at(9600, () => { m.innerHTML = T[3]; });
    tl.at(12600, () => { m.innerHTML = T[4]; });
  }
},

/* ------------------------------------------------------ 8 <span class="hl-e">view_src</span> 不是图的边：遍历不走它 */
{
  kicker: "L1-03 · 视图",
  title: "<span class=\"hl-e\">view_src</span> 不是图的边：遍历不走它",
  sub: "一次 ggml_view_impl 调用同时建了两条链：src[0] 是计算依赖，view_src 是数据归属。",
  caption: "需要视图的数据来源时，代码必须自己沿 view_src 链上溯 —— 例如子图融合判定；外部的 view_src 只允许是常量权重。",
  src: "ggml/src/ggml.c",
  mark: [6, 11, 12, 19, 21, 24],
  lineNo: 0,
  code: `static struct ggml_tensor * ggml_view_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_dims,
        const int64_t       * ne,
        size_t                offset) {
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, sizeof(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    return result;
}
//>> ---- ggml/src/ggml.c:7805-7813 ----
        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        struct ggml_tensor * view_src = node->view_src;
        while (view_src) {
            if (ggml_node_list_find_tensor(cgraph, node_idxs, count, view_src) == -1 && !ggml_is_constant(view_src)) {
                return false;
            }
            view_src = view_src->view_src;
        }`,
  duration: 22000,
  build(root, tl) {
    const w = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    w.innerHTML = `<div class="row" style="gap:12px"><div class="col grow" id="a" style="gap:4px"></div><div class="col grow" id="b" style="gap:4px"></div></div><div class="formula" id="m"></div>`;
    root.appendChild(w);
    function chain(h, title, c, items, tag) {
      h.innerHTML = '<div class="cm" style="margin:0;color:var(--' + c + ')">' + title + '</div>' + items.map((it, i) =>
        (i ? '<div class="formula" style="padding:2px 6px;font-size:9px"><span class="m" style="margin:0">' + tag + '</span></div>' : '') +
        '<div class="formula" style="padding:4px 7px;font-size:10px"><span class="' + c + '">' + it + '</span></div>').join('');
      h.style.opacity = '.3';
    }
    const a = w.querySelector('#a'), b = w.querySelector('#b');
    chain(a, '计算链：拓扑遍历走这条', 'k', ['out', 'v (GGML_OP_VIEW)', 'base'], 'src[0]');
    chain(b, '数据链：遍历不走这条', 'e', ['v (GGML_OP_VIEW)', 'base'], 'view_src');
    const m = w.querySelector('#m');
    const T = ['ggml_view_impl 一次调用建两条链。',
     '<span class="v">src[0] = a</span>（第 3793 行）：<span class="k">计算依赖</span> —— v 要算，就得先算 a。拓扑遍历走这条。',
     '<span class="v">view_src = a</span>（第 3787 行，作为 ggml_new_tensor_impl 的第 5 个实参）：<span class="k">数据归属</span> —— v 借 a 的内存，加上 view_offs。',
     '两条链在图上分开：第 4 幕已经看到，遍历只跟 <span class="v">node-&gt;src[k]</span>，从不跟 view_src。',
     '所以要"视图的数据从哪来"的代码必须自己上溯（第二部分）：融合判定沿 view_src -> view_src 一直走，并要求图外的来源是常量权重。',
     '一句话：<span class="k">src[] 决定"怎么算"，view_src 决定"内存归谁"</span>。两者都是指针，但只有前者是图的边。'];
    tl.at(600, () => { a.style.opacity = '1'; m.innerHTML = T[0]; });
    tl.at(3600, () => { m.innerHTML = T[1]; });
    tl.at(6600, () => { b.style.opacity = '1'; m.innerHTML = T[2]; });
    tl.at(9600, () => { m.innerHTML = T[3]; });
    tl.at(12600, () => { m.innerHTML = T[4]; });
    tl.at(15600, () => { m.innerHTML = T[5]; });
  }
},

/* ------------------------------------------------------ 9 收束：这张图归谁所有 */
{
  kicker: "L1-03 · 收束",
  title: "收束：这张图归谁所有",
  sub: "左表把本课压成六行；右边源码说明这些 C 结构在 C++ 侧由 unique_ptr 托管（RAII）。",
  caption: "下一课 L1-04 讲张量内层的量化块布局；图的执行顺序则在 L4-04 的 graph_compute 里被消费。",
  src: "ggml/include/ggml-cpp.h",
  mark: [0, 4, 5, 7, 8],
  lineNo: 13,
  code: `// Smart pointers for ggml types

// ggml

struct ggml_context_deleter { void operator()(ggml_context * ctx) { ggml_free(ctx); } };
struct gguf_context_deleter { void operator()(gguf_context * ctx) { gguf_free(ctx); } };

typedef std::unique_ptr<ggml_context, ggml_context_deleter> ggml_context_ptr;
typedef std::unique_ptr<gguf_context, gguf_context_deleter> gguf_context_ptr;`,
  duration: 20000,
  build(root, tl) {
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
     '同一张权重 W 被 <span class="mono">mul_mat(A, W)</span> 与 <span class="mono">mul_mat(B, W)</span> 两个节点引用。遍历到第二个节点时，<span class="mono">ggml_visit_parents_graph</span> 对 W 做什么？如果改成"在 ggml_tensor 上打一个 visited 标志位"，会坏在哪里？',
     '第一次到达 W：<span class="mono">ggml_hash_find()</span> 给出空槽，于是写 <span class="mono">keys[pos] = W</span>、置 <span class="mono">used[pos] = 1</span>、<span class="mono">use_counts[pos] = 0</span>，再按 op/flags 进 leafs[] 或 nodes[]。<br>' +
     '第二次到达：<span class="mono">used[pos]</span> 已是 1，命中 "already visited" 分支直接 return —— W 不再入列，只在调用点 <span class="mono">use_counts[pos]++</span>。<br><br>' +
     '换成张量上的标志位会坏在两处：<br>' +
     '(1) <b>状态放错了容器</b>。张量不属于任何一张图：ggml_graph_view 按值复制 visited_hash_set 与原图共用表，ggml_graph_dup + ggml_graph_cpy 则让新图把 keys 重新插进自己的表。' +
     '标志位打在张量上，这些图就会互相污染，且每次重建都要清一遍；而 used 位图属于 cgraph，清图只要 <span class="mono">ggml_graph_clear()</span> 一次重置。<br>' +
     '(2) <b>拿不到图内编号</b>。grads[]、grad_accs[]、use_counts[] 都按 hash 槽位索引（源码注释：indexed by hash table slot）。' +
     '标记位只能回答"来过没有"，给不出这个稳定下标；hash set 的 find/insert 返回的槽位正好两件事一起做。'));
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
  }
},

];
