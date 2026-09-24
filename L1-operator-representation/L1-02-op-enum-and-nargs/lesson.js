/* ==========================================================================
   L1-02 · 算子枚举与元数据：算子的身份
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 算子的身份：<span class="hl-a">一个编号</span> + 两张表 */
{
  kicker: "L1 · 算子的表示",
  title: "算子的身份：<span class=\"hl-a\">一个编号</span> + 两张表",
  sub: "enum ggml_op 给每个算子一个编号；两张静态表把编号翻译成\"日志里的名字\"和\"图里的符号\"。",
  caption: "回顾 L1-01：那里讲张量的数据面（ne/nb）；本课讲身份面。下一课 L1-03 讲引用这些节点的计算图。",
  src: "ggml/src/ggml.c",
  mark: [1, 5],
  lineNo: 1364,
  code: `const char * ggml_op_name(enum ggml_op op) {
    return GGML_OP_NAME[op];
}

const char * ggml_op_symbol(enum ggml_op op) {
    return GGML_OP_SYMBOL[op];
}`,
  duration: 14000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-c">GGML_OP_NAME</span>：编号就是下标 */
{
  kicker: "L1-02 · 身份",
  title: "★ <span class=\"hl-c\">GGML_OP_NAME</span>：编号就是下标",
  sub: "名字表是\"枚举值到字符串\"的直查表；表的长度和枚举的哨兵值必须一致。",
  caption: "表尾的 static_assert（ggml.c:1104）把长度钉在 101 项 —— 末幕会看到那一行。",
  src: "ggml/src/ggml.c",
  mark: [0, 2, 4, 5],
  lineNo: 991,
  code: `static const char * GGML_OP_NAME[GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD_ID",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",`,
  duration: 16000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 3 输入个数：写了几个 <span class="hl-b">src[i]</span> 就是几个 */
{
  kicker: "L1-02 · 输入",
  title: "输入个数：写了几个 <span class=\"hl-b\">src[i]</span> 就是几个",
  sub: "ggml_mul_mat 赋了两个 src；ggml_add_id 赋了三个 —— 数量由各自的构造器决定。",
  caption: "GGML_MAX_SRC = 10（ggml/include/ggml.h:224）只是 src[] 数组的长度上限，不是算子的输入个数。",
  src: "ggml/src/ggml.c",
  mark: [10, 11, 12],
  lineNo: 3341,
  code: `struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}`,
  duration: 17000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 4 输出形状规则：<span class="hl-d">每个算子自己算</span> */
{
  kicker: "L1-02 · 输出",
  title: "输出形状规则：<span class=\"hl-d\">每个算子自己算</span>",
  sub: "ggml_can_mul_mat 先断言输入合法，再由一行 ne[] 决定输出形状；没有统一的形状表。",
  caption: "对照 ggml_glu_impl（ggml.c:2920-2921）：GLU 的输出是 a->ne[0] / 2，规则完全不同。",
  src: "ggml/src/ggml.c",
  mark: [3, 5, 15, 16],
  lineNo: 3333,
  code: `static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);`,
  duration: 17000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 5 同一个枚举的第二种用法：<span class="hl-e">ggml_op_is_empty</span> */
{
  kicker: "L1-02 · 分类",
  title: "同一个枚举的第二种用法：<span class=\"hl-e\">ggml_op_is_empty</span>",
  sub: "把 op 当分类标签：只有 5 个算子\"什么都不做\"，其余 96 个都要真的算。",
  caption: "同类开关还有 ggml_op_can_inplace（ggml-alloc.c:22）与 CPU 的 ggml_get_n_tasks（ggml-cpu.c:2253）——L5-01 会看到按 op 分派的大 switch。",
  src: "ggml/src/ggml-impl.h",
  mark: [2, 3, 4, 5, 6],
  lineNo: 90,
  code: `static bool ggml_op_is_empty(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        default:
            return false;
    }`,
  duration: 15000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 6 ★ <span class="hl-e">op_params</span>：16 个 int32，schema 只是注释 */
{
  kicker: "L1-02 · 参数",
  title: "★ <span class=\"hl-e\">op_params</span>：16 个 int32，schema 只是注释",
  sub: "op_params 没有结构体类型；槽位含义由每个算子自己约定，唯一的边界检查是一条 assert。",
  caption: "同一槽号在不同算子下含义不同：MUL_MAT 的槽 0 是累加精度，FLASH_ATTN_EXT 的精度在槽 3（ggml.c:3282-3304，见 source.md 第九节）。",
  src: "ggml/src/ggml-impl.h",
  mark: [2, 3, 4, 5, 13],
  lineNo: 163,
  code: `// [TAG_GGML_PREC]
// - GGML_OP_MUL_MAT
//   0 - acc
//   1 - hint
//   2 - src0 precision
//   3 - src1 precision
//
// - GGML_OP_MUL_MAT_ID
//   0 - acc
//   1 - hint
//   2 - src0 precision
//   3 - src1 precision
static void ggml_set_op_params_i32(struct ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    ((int32_t *)(tensor->op_params))[i] = value;
}

static void ggml_set_op_params_f32(struct ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(tensor->op_params))[i] = value;
}`,
  duration: 18000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 7 新张量的默认身份：<span class="hl-g">GGML_OP_NONE</span> */
{
  kicker: "L1-02 · 成节点",
  title: "新张量的默认身份：<span class=\"hl-g\">GGML_OP_NONE</span>",
  sub: "张量出生时 op = NONE、src 全 NULL；构造器随后把身份写进去，它才成为图上的节点。",
  caption: "写完之后 ggml_build_forward_expand 才能沿 src[] 做拓扑排序 —— 那是 L1-03。",
  src: "ggml/src/ggml.c",
  mark: [5, 6, 8],
  lineNo: 1809,
  code: `    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ NULL,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { NULL },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        /*.padding      =*/ { 0 },
    };`,
  duration: 16000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 8 把这一课压成一张表 */
{
  kicker: "L1-02 · 收束",
  title: "把这一课压成一张表",
  sub: "三个问题，三种答案的存放位置；两张表尾的 static_assert 负责让它们不脱节。",
  caption: "下一课 L1-03 讲这些 src[] 如何被拓扑排序成一张可执行的图。",
  src: "ggml/src/ggml.c",
  mark: [3],
  lineNo: 1101,
  code: `    "GLU",
};

static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT != 101");

static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {`,
  duration: 17000,
  build(root, tl) {
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
  }
},

];
