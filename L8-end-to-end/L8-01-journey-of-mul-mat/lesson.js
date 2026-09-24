/* ==========================================================================
   L8-01 · ★ 端到端：一个 mul_mat 的完整旅程
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一课走完一条链路：<span class="hl-a">ggml_mul_mat</span> 的完整旅程 */
{
  kicker: "L8 · 端到端",
  title: "一课走完一条链路：<span class=\"hl-a\">ggml_mul_mat</span> 的完整旅程",
  sub: "一次 mul_mat 要穿过 4 个世界、9 个文件；每一跳都有一个明确的判据。",
  caption: "前置课 L7-06（小后端合集）；下一课 L8-02 用真实模型参数走一遍 offload 决策。",
  src: "src/llama-graph.cpp",
  mark: [4],
  lineNo: 1514,
  code: `ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
//>> 本课的主角就是这一行：它在图上创建了一个 GGML_OP_MUL_MAT 节点`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">建图 L1/L2</span><span class="arrow">-&gt;</span>
        <span class="chip b">入图 L1</span><span class="arrow">-&gt;</span>
        <span class="chip c">调度 L3/L4</span><span class="arrow">-&gt;</span>
        <span class="chip d">内核 L5/L6/L7</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '第 1 段 · 建图', m: 'ggml_mul_mat',
        b: '模型 build 函数调用 build_attn；<br>一次调用只往图上加一个节点' },
      { c: 'b', t: '第 2 段 · 入图', m: 'ggml_build_forward_expand',
        b: '从根出发 DFS；<br>visited_hash_set 决定节点只进图一次' },
      { c: 'c', t: '第 3 段 · 调度', m: 'ggml_backend_sched_*',
        b: '切分 + 归属 + 分配：<br>决定这个节点跑在哪个后端' },
      { c: 'd', t: '第 4 段 · 内核', m: 'ggml_compute_forward',
        b: '后端自己的 switch 再分派：<br>vec_dot / CUDA kernel / NPU 算子' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '本课只做一件事：把一个 <span class="k">ggml_mul_mat</span> 从"被调用"到"算完"，逐跳走完。',
      '主角：<span class="v">build_lora_mm(wo, cur)</span> 里的 <span class="v">ggml_mul_mat(ctx0, w, cur)</span>（llama-graph.cpp:1518）。',
      '它要穿过 9 个文件：models/llama.cpp、llama-graph.cpp、ggml.c、llama-context.cpp、ggml-backend.cpp、ggml-cpu.cpp/c、arch/x86/quants.c。',
      '记住判据：<span class="k">can_mul_mat</span> → <span class="k">visited_hash_set</span> → <span class="k">weight buffer</span> → <span class="k">iface 槽</span> → <span class="k">tensor-&gt;op</span> → <span class="k">src0-&gt;type</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(17600, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 2 起点：<span class="hl-b">模型 build 函数</span>调用 build_attn */
{
  kicker: "L8-01 · 第 0 跳",
  title: "起点：<span class=\"hl-b\">模型 build 函数</span>调用 build_attn",
  sub: "第 169 行把这一层的 wo 权重交出去；第 246 行才把整条链拉进图。",
  caption: "回顾 L2-06：src/models/llama.cpp 的 graph<>() 是整张图的装配线。",
  src: "src/models/llama.cpp",
  mark: [],
  lineNo: 0,
  code: `            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
//>> ---- src/models/llama.cpp:246-246 ----
    ggml_build_forward_expand(gf, cur);`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:9px">
        <div class="col grow" id="stack" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const stack = wrap.querySelector('#stack');
    stack.innerHTML = '<div class="cm" style="margin-bottom:2px">模型侧：llama_model_llama::graph&lt;embed&gt;()</div>';
    const frames = [
      { n: 'for (int il = 0; il < n_layer; ++il)', c: 'a', d: '逐层装配（第 132 行起）' },
      { n: 'cur = build_attn(inp_attn, ...)', c: 'b', d: '把 wo / wo_b / wo_s 与 Q/K/V 交出去（169）' },
      { n: 'cur = build_lora_mm(wo, cur, wo_s)', c: 'c', d: 'build_attn 内部：输出投影（2804）' },
      { n: 'ggml_mul_mat(ctx0, w, cur)', c: 'd', d: '图上多一个 MUL_MAT 节点（1518）' }
    ];
    const els = frames.map(f => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
      e.innerHTML = '<span class="cm" style="margin:0">' + U.esc(f.n) + '</span><br>' +
        '<span style="color:var(--' + f.c + ');font-size:9px">' + U.esc(f.d) + '</span>';
      stack.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">wo 是权重，不是激活</div>' +
      '<div class="cb">它在模型加载时就被放进某个后端的 buffer，' +
      'usage = <span class="cm" style="margin:0">GGML_BACKEND_BUFFER_USAGE_WEIGHTS</span>。' +
      '这个事实会在第 7 幕决定这次 mul_mat 归谁。</div></div>' +
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">第 246 行：整条链入图</div>' +
      '<div class="cb">层循环里建出来的节点，最后靠 <span class="cm" style="margin:0">ggml_build_forward_expand(gf, cur)</span> ' +
      '（246）从根反向 DFS 才真正进入 <span class="cm" style="margin:0">cgraph</span>。' +
      '中间还有若干次局部的 expand（如 llama-graph.cpp:2735、2783）。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '起点不在 ggml 里，在<span class="k">模型代码</span>里：一次普通的 C++ 调用。',
      '<span class="v">build_attn</span> 拿到的是本层的 <span class="k">wo</span>（输出投影权重）与 Q/K/V 三个张量。',
      '<span class="v">build_lora_mm</span> 只是个薄封装：它把权重与激活交给 <span class="k">ggml_mul_mat</span>。',
      '注意这一步<span class="k">什么都没算</span>：只是往 <span class="v">ctx0</span> 里加了一个节点。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.28');
      msg.innerHTML = texts[i];
    }));
    tl.at(14800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 3 ★ 图上没有"执行"：<span class="hl-a">一次调用 = 一个节点</span> */
{
  kicker: "L8-01 · ★ 洞察",
  title: "★ 图上没有\"执行\"：<span class=\"hl-a\">一次调用 = 一个节点</span>",
  sub: "build_attn 第 2804 行 → build_lora_mm 第 1514 行 → ggml_mul_mat 第 1518 行。",
  caption: "对照：同一段 build_attn_mha 里的 kqv = ggml_mul_mat(v, kq)（2715）没有权重，归属判据不同（见第 7 幕与 L4-02）。",
  src: "src/llama-graph.cpp",
  mark: [],
  lineNo: 0,
  code: `    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }
//>> ---- src/llama-graph.cpp:1514-1518 ----
ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:11px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const dia = wrap.querySelector('#dia');

    const a = U.el('div', { class: 'card', style: 'width:210px;border-left-color:var(--a)' });
    a.innerHTML = '<div class="ct" style="color:var(--a)">build_attn(wo, cur)</div>' +
      '<div class="cb">llama-graph.cpp:2804<br>wo 非空 → 做输出投影<br>' +
      '<span class="cm" style="margin:0">cur = build_lora_mm(wo, cur, wo_s)</span></div>';
    const b = U.el('div', { class: 'card', style: 'width:210px;border-left-color:var(--c)' });
    b.innerHTML = '<div class="ct" style="color:var(--c)">build_lora_mm</div>' +
      '<div class="cb">llama-graph.cpp:1518<br>本体只有一行（外加 LoRA 分支）<br>' +
      '<span class="cm" style="margin:0">ggml_mul_mat(ctx0, w, cur)</span></div>';
    const c = U.el('div', { class: 'card', style: 'width:210px;border-left-color:var(--d)' });
    c.innerHTML = '<div class="ct" style="color:var(--d)">新节点 kqv_out</div>' +
      '<div class="cb">op = MUL_MAT<br>src[0] = w（权重）<br>src[1] = cur（激活）</div>';
    dia.appendChild(a); dia.appendChild(U.arrow('-&gt;')); dia.appendChild(b);
    dia.appendChild(U.arrow('-&gt;')); dia.appendChild(c);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把这三行连起来看：<span class="k">模型代码 → 薄封装 → 构造器</span>，中间没有任何计算。',
      '<span class="v">ggml_mul_mat</span> 返回的是一个 <span class="k">ggml_tensor *</span> —— 它是一个<em>句柄</em>，不是结果。',
      '节点的全部信息就是：<span class="v">op</span> + <span class="v">src[]</span> + <span class="v">ne[]</span> + <span class="v">type</span>。',
      '所以"什么时候算"这个问题，在图上根本不存在 —— 它由后面第 6 幕的 decode 决定。'
    ];
    [0, 1, 2].forEach(i => tl.at(700 + i * 3600, () => {
      msg.innerHTML = texts[i];
    }));
    tl.at(11600, () => { msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 4 构造器：<span class="hl-c">三条断言</span> + 一个输出形状 */
{
  kicker: "L8-01 · 第 1 跳",
  title: "构造器：<span class=\"hl-c\">三条断言</span> + 一个输出形状",
  sub: "ggml_mul_mat 的实现只有 15 行；判据全在 ggml_can_mul_mat 里。",
  caption: "回顾 L1-02：op 身份（3351）与输出形状规则（3348）都写在构造器里，没有单独的元数据表。",
  src: "ggml/src/ggml.c",
  mark: [0, 4, 6, 8, 12, 17, 19, 22, 24, 27, 29, 31],
  lineNo: 3333,
  code: `static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
//>> 判据函数：能不能乘，只看三个 ne 关系
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0]           == t1->ne[0])  &&
//>> ① 内维必须相等：a 的列数 == b 的列数
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
//>> ② a 必须在 dim2 上可广播（b 的 ne[2] 是 a 的整数倍）
           (t1->ne[3]%t0->ne[3] == 0);
//>> ③ 同上，dim3
}

struct ggml_tensor * ggml_mul_mat(
//>> 构造器本体从这里开始
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
//>> 不满足就直接断言失败 —— 这是"编译期之外"的图级检查
    GGML_ASSERT(!ggml_is_transposed(a));
//>> a 不能是转置视图：内核按 src0 的行扫描（L5-03 的前提）

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
//>> ★ 输出形状只有一行：{ a->ne[1], b->ne[1], b->ne[2], b->ne[3] }
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
//>> 输出类型固定 F32，与 a/b 的类型无关（量化权重也乘出 F32）

    result->op     = GGML_OP_MUL_MAT;
//>> 身份面：op 决定后面所有 switch 走哪个 case
    result->src[0] = a;
//>> 输入个数由这里赋了几个 src 决定（L1-02 实测：没有 nargs 表）
    result->src[1] = b;
//>> src[1] 是激活 —— 它在第 7 幕决定归属时帮不上忙

    return result;
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:2px">判据：ggml_can_mul_mat(a, b)</div>';
    const checks = [
      { n: 'a-&gt;ne[0] == b-&gt;ne[0]', c: 'a', d: '内维相等（K 维对齐）' },
      { n: 'b-&gt;ne[2] % a-&gt;ne[2] == 0', c: 'b', d: 'a 在 dim2 可广播' },
      { n: 'b-&gt;ne[3] % a-&gt;ne[3] == 0', c: 'c', d: 'a 在 dim3 可广播' },
      { n: '!ggml_is_transposed(a)', c: 'd', d: 'src0 不能是转置视图' }
    ];
    const els = checks.map(k => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
      e.innerHTML = '<span class="cm" style="margin:0;color:var(--' + k.c + ')">' + k.n + '</span>' +
        '<br><span style="font-size:9px">' + U.esc(k.d) + '</span>';
      left.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="formula" id="shape" style="font-size:10px"></div>' +
      '<div class="card" style="border-left-color:var(--e)">' +
      '<div class="ct" style="color:var(--e)">输出为什么是 F32？</div>' +
      '<div class="cb">因为 <span class="cm" style="margin:0">ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)</span> 写死了。' +
      '量化权重负责"省内存"，累加与结果一律走 F32 —— 这是 L1-04 与 L5-03 的共同前提。</div></div>' +
      '<div class="card" style="border-left-color:var(--f)">' +
      '<div class="ct" style="color:var(--f)">输入是"借来的"</div>' +
      '<div class="cb">构造器只存指针：<span class="cm" style="margin:0">src[0] = a</span>、' +
      '<span class="cm" style="margin:0">src[1] = b</span>。' +
      '所以这个节点与 <span class="cm" style="margin:0">wo</span>、与激活共享同一块内存。</div></div>';
    const shape = right.querySelector('#shape');
    shape.innerHTML = '<span class="m">ne</span> = { <span style="color:var(--a)">a-&gt;ne[1]</span>, ' +
      '<span style="color:var(--b)">b-&gt;ne[1]</span>, <span style="color:var(--c)">b-&gt;ne[2]</span>, ' +
      '<span style="color:var(--d)">b-&gt;ne[3]</span> }';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '构造器只做四件事：<span class="k">断言可乘</span>、<span class="k">算 ne[]</span>、<span class="k">建张量</span>、<span class="k">填 op 与 src</span>。',
      '三条 ne 判据 + 一条"src0 不能转置"，就是这张图能不能建出来的全部条件。',
      '输出形状 = <span class="v">{a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3]}</span>：把 a 的"行数"与 b 的其余维度拼起来。',
      '输出类型与输入类型<span class="k">无关</span>：永远是 F32。这就是"量化权重乘 F32 激活"能成立的原因。',
      '到这里，图上多了一个节点 —— 但还没有任何东西把它<span class="k">串进图</span>。下一幕。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    els.forEach((_, i) => tl.at(3300 + i * 3000, () => {
      els.forEach((e, k) => e.style.opacity = k <= i ? '1' : '.28');
      msg.innerHTML = texts[Math.min(i + 1, 3)];
    }));
    tl.at(16200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 入图：<span class="hl-e">hash set</span> 决定"要不要重算" */
{
  kicker: "L8-01 · 第 2 跳",
  title: "入图：<span class=\"hl-e\">hash set</span> 决定\"要不要重算\"",
  sub: "ggml_build_forward_expand 是入口，真正的拓扑序遍历在 ggml_visit_parents_graph。",
  caption: "回顾 L1-03：visited_hash_set 让每个节点只入图一次 —— 这也是\"图是 DAG 不是树\"的证据。",
  src: "ggml/src/ggml.c",
  mark: [],
  lineNo: 0,
  code: `void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}
//>> ---- ggml/src/ggml.c:7236-7245 ----
static size_t ggml_visit_parents_graph(struct ggml_cgraph * cgraph, struct ggml_tensor * node, bool compute) {
    if (node->op != GGML_OP_NONE && compute) {
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    const size_t node_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
    GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);

    if (ggml_bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
        // already visited`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip e">expand(cur)</span><span class="arrow">-&gt;</span>
        <span class="chip c">visit_parents</span><span class="arrow">-&gt;</span>
        <span class="chip b">hash_find</span><span class="arrow">-&gt;</span>
        <span class="chip a">入 cgraph-&gt;nodes</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:2px">一次 expand 的递归顺序（简化）</div>';
    const steps = [
      { n: 'mul_mat 节点（我们这次）', c: 'a' },
      { n: 'src[0] = w（权重，叶子）', c: 'b' },
      { n: 'src[1] = cur（上一层输出）', c: 'c' },
      { n: '…继续向上一层递归', c: 'd' }
    ];
    const els = steps.map(s => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
      e.innerHTML = '<span style="color:var(--' + s.c + ')">' + U.esc(s.n) + '</span>';
      left.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">先问"见过没有"</div>' +
      '<div class="cb"><span class="cm" style="margin:0">ggml_hash_find(&amp;cgraph-&gt;visited_hash_set, node)</span>（7241）。' +
      'used 位已置 → 直接返回，不再往下走（7244）。共用同一个权重的两个节点不会把它算两遍。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">叶子与节点分开存</div>' +
      '<div class="cb">op == NONE 且不是 PARAM 的进 <span class="cm" style="margin:0">cgraph-&gt;leafs</span>，' +
      '其余进 <span class="cm" style="margin:0">cgraph-&gt;nodes</span>。' +
      '后端只需要遍历 nodes（一次 switch）。</div></div>' +
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">顺序 = 拓扑序</div>' +
      '<div class="cb">先递归父节点再记录自己，所以 nodes[] 里"生产者一定排在消费者前面" —— ' +
      '第 9 幕 CPU 线程按这个顺序扫一遍即可，不需要再做依赖检查。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">ggml_build_forward_expand(gf, cur)</span>（7337）只转发到 <span class="v">ggml_build_forward_impl</span>。',
      'DFS 的每一站都先查 hash set：<span class="k">没见过 → 打标记 + 继续递归；见过 → 立刻返回</span>。',
      '我们的 mul_mat 节点就是这样入图的：src[1] 一路递归回上一层的输出，src[0] 递归到权重叶子。',
      '入图结束后，<span class="v">cgraph-&gt;nodes</span> 里就是一张排好序的待执行清单 —— 还没有后端的概念。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => e.style.opacity = k <= i ? '1' : '.28');
      msg.innerHTML = texts[Math.min(i, 3)];
    }));
    tl.at(16800, () => { msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 6 执行入口：<span class="hl-b">decode</span> 的三件事 */
{
  kicker: "L8-01 · 第 3 跳",
  title: "执行入口：<span class=\"hl-b\">decode</span> 的三件事",
  sub: "C API 只转发；llama_context::decode 决定\"重建图 / 分配 / 计算\"。",
  caption: "回顾 L2-07：K/V 写缓存、ubatch 准备都在这一层；本幕只看与这条链有关的三行。",
  src: "src/llama-context.cpp",
  mark: [],
  lineNo: 0,
  code: `int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\\n", __func__, ret);
//>> ---- src/llama-context.cpp:1433-1454 ----
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
//>> ---- src/llama-context.cpp:2583-2590 ----
    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\\n", __func__, status);`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '① 建图（1425）', m: 'model.build_graph(gparams)',
        b: '图的形状只取决于 ubatch 参数；<br>参数没变就复用上一张图（1415 n_reused++）' },
      { c: 'b', t: '② 分配（1435）', m: 'ggml_backend_sched_alloc_graph',
        b: '对调度器说"这张图要跑了"：<br>切分 + 归属 + buffer 一次做完' },
      { c: 'c', t: '③ 计算（1454 → 2588）', m: 'graph_compute → ..._async',
        b: 'llama_context::graph_compute（2569）设好线程数，<br>交给 ggml_backend_sched_graph_compute_async' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">llama_decode</span>（4326）只有一行转发：<span class="k">ctx-&gt;decode(batch)</span>。',
      '<span class="v">decode</span> 里与这条链有关的就是三个调用点：建图、分配、计算。',
      '注意<span class="k">分配</span>这一步：<span class="v">ggml_backend_sched_alloc_graph</span> 才第一次涉及"后端"。',
      '而 <span class="v">graph_compute</span> 只是设置线程数（2584-2586），真正干活的是下一幕的调度器。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
      els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.30');
      msg.innerHTML = texts[i];
    }));
    tl.at(17400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 7 ★ 判据：<span class="hl-a">权重住在哪个 buffer</span>，op 就归哪个后端 */
{
  kicker: "L8-01 · ★ 洞察",
  title: "★ 判据：<span class=\"hl-a\">权重住在哪个 buffer</span>，op 就归哪个后端",
  sub: "调度器按固定优先级给每个节点找归属；我们这次命中的是\"输入里有权重\"这一条。",
  caption: "回顾 L4-02：优先级四层与 copy 节点的产生都在那一课逐字展开；本幕只把它放回链路。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [],
  lineNo: 0,
  code: `bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

//>> ---- ggml/src/ggml-backend.cpp:921-948 ----
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
//>> ---- ggml/src/ggml-backend.cpp:961-969 ----
    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '① 预分配（923）', m: 'backend_from_buffer', b: '张量自己有 buffer：<br>直接用它的后端（权重走这条）' },
      { c: 'b', t: '② 视图（930）', m: 'view_src-&gt;buffer', b: '视图张量看 view_src：<br>不产生数据，归属必然跟着源' },
      { c: 'c', t: '③ 图输入（945）', m: 'GGML_TENSOR_FLAG_INPUT', b: '输入张量固定给<br>最后一个后端（约定是 CPU）' },
      { c: 'd', t: '④ 输入有权重（967）', m: 'BUFFER_USAGE_WEIGHTS', b: '★ 本次命中：src[0] = wo<br>用权重的后端，除非更高优先级后端愿意接' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.26');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '切分发生在分配之前：<span class="v">sched_alloc_graph</span>（1992）第一件事就是 <span class="v">split_graph</span>（2000）。',
      '每个节点先问 <span class="v">backend_id_from_cur</span>（921）：按 ①预分配 → ②视图 → ③图输入 → ④权重 的顺序找。',
      '我们这次落在 ④：<span class="k">src[0] = wo</span> 的 buffer usage 是 WEIGHTS → <span class="v">src_backend_id</span> 就是 wo 所在后端的编号（967-968）。',
      '如果那个后端支持这个 op，就<span class="k">一路跳过</span>准备拿图的邻居；否则会被 copy 节点隔开（L4-02 的产物）。',
      '对照：<span class="v">kqv = ggml_mul_mat(v, kq)</span>（llama-graph.cpp:2715）两个输入都不是权重 → 走到 921 的 <span class="k">-1</span>，改由邻居"带走"（1058 set_if_supported）或按"支持输入最多的后端"分配（pass 3，1210-1246）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(3400 + i * 3400, () => {
      els.forEach((e, k) => e.style.opacity = k <= i ? '1' : '.26');
      msg.innerHTML = texts[Math.min(i + 1, 3)];
    }));
    tl.at(17800, () => { msg.innerHTML = texts[3]; });
    tl.at(20200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 从 split 到接口：<span class="hl-c">backend_id → iface.graph_compute</span> */
{
  kicker: "L8-01 · 第 4 跳",
  title: "从 split 到接口：<span class=\"hl-c\">backend_id → iface.graph_compute</span>",
  sub: "切分把图切成连续段；每段拿到一个后端指针，执行就是一次虚表调用。",
  caption: "回顾 L3-01（后端契约）与 L4-04（图执行入口）：本幕是两者的交点。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [],
  lineNo: 0,
  code: `enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}
//>> ---- ggml/src/ggml-backend.cpp:1646-1660 ----
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    int prev_backend_id = -1;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

//>> ---- ggml/src/ggml-backend.cpp:1798-1802 ----
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip c">split[i]</span><span class="arrow">-&gt;</span>
        <span class="chip b">backend_id</span><span class="arrow">-&gt;</span>
        <span class="chip a">sched-&gt;backends[id]</span><span class="arrow">-&gt;</span>
        <span class="chip d">iface.graph_compute</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:2px">split 的两句话（L4-02 第 2 幕）</div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">这一段的图</div>' +
      '<div class="cb">split-&gt;graph：连续的同后端节点<br>' +
      '<span class="cm" style="margin:0">compute_splits</span> 直接把它交给后端（1799）</div></div>' +
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">这一段要吃什么</div>' +
      '<div class="cb">split-&gt;n_inputs / inputs[]：跨后端的张量<br>' +
      '先 copy，再算（1786-1792）</div></div>';
    left.appendChild(U.el('div', { class: 'formula', style: 'font-size:9.5px',
      html: '本次的 mul_mat 与 <span class="k">wo</span> 同属一段 —— 无需 copy' }));

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">唯一的"多态"点</div>' +
      '<div class="cb"><span class="cm" style="margin:0">backend-&gt;iface.graph_compute(backend, cgraph)</span>（463）' +
      '是全链路唯一一次"不知道会跳到哪"的调用。CPU 把它填成 ' +
      '<span class="cm" style="margin:0">ggml_backend_cpu_graph_compute</span>（ggml-cpu.cpp:206），' +
      'CUDA 填成 <span class="cm" style="margin:0">ggml_backend_cuda_graph_compute</span>（ggml-cuda.cu:4853）。</div></div>' +
      '<div class="card" style="border-left-color:var(--e)">' +
      '<div class="ct" style="color:var(--e)">同一张图可能被调多次</div>' +
      '<div class="cb">for 循环按 <span class="cm" style="margin:0">sched-&gt;n_splits</span> 逐段执行（1656）；' +
      '段与段之间如果需要，还会插入同步或事件等待。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">compute_splits</span>（1646）按段循环：CPU 段、GPU 段、再回到 CPU 段，各自调一次后端。',
      '每段的执行体是 <span class="v">ggml_backend_graph_compute_async(split_backend, &amp;split-&gt;graph)</span>（1799）。',
      '<span class="v">ggml_backend_graph_compute_async</span>（461）只有一行：<span class="k">backend-&gt;iface.graph_compute(...)</span>。',
      '到这里，"算哪个 op"这件事已经从链路里消失了 —— 后端拿到的是<span class="k">一整段图</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4600, () => { msg.innerHTML = texts[1]; });
    tl.at(9200, () => { msg.innerHTML = texts[2]; });
    tl.at(14000, () => { msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 9 CPU 侧：<span class="hl-d">switch (tensor-&gt;op)</span> → mul_mat → vec_dot */
{
  kicker: "L8-01 · ★ 洞察",
  title: "CPU 侧：<span class=\"hl-d\">switch (tensor-&gt;op)</span> → mul_mat → vec_dot",
  sub: "大 switch 的判据是 op；vec_dot 的判据是 src0->type。两个字段，两次分派。",
  caption: "回顾 L5-01（CPU 分派）与 L5-03（向量化内核）：本幕把这两课接回链路，并给出完整调用链表。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [],
  lineNo: 0,
  code: `static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:1869-1872 ----
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:240-249 ----
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
//>> ---- ggml/src/ggml-cpu/ggml-cpu.c:1165-1183 ----
static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div class="formula" id="msg" style="font-size:9.5px;padding:5px 8px"></div>
      <div id="tbl"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['跳（函数）', '文件:行号', '判据：谁决定下一步'],
      [['模型 build → build_attn', 'src/models/llama.cpp:169', '层里有 wo 权重 → 走注意力块'],
       ['build_attn → build_lora_mm(wo, cur)', 'src/llama-graph.cpp:2804', 'wo 非空 → 做输出投影'],
       ['build_lora_mm → ggml_mul_mat(w, cur)', 'src/llama-graph.cpp:1518', '这一次调用只是加节点'],
       ['构造器：断言 + ne[] + op/src', 'ggml/src/ggml.c:3333 / 3348', 'can_mul_mat 的三条 ne 判据'],
       ['ggml_build_forward_expand → visit_parents', 'ggml/src/ggml.c:7337 → 7236', 'visited_hash_set 没见过才继续 DFS'],
       ['llama_decode → llama_context::decode', 'src/llama-context.cpp:4326 → 1704', '图没变就复用（1415 n_reused）'],
       ['sched_alloc_graph → split_graph', 'ggml/src/ggml-backend.cpp:1992 → 1066', '切分在分配之前（2000）'],
       ['backend_id_from_cur（归属）', 'ggml/src/ggml-backend.cpp:921', 'src[0] 的 buffer = WEIGHTS（967）'],
       ['compute_splits → iface.graph_compute', 'ggml/src/ggml-backend.cpp:1799 → 461', 'split-&gt;backend_id（1658）'],
       ['cpu_graph_compute → ggml_graph_compute', 'ggml/src/ggml-cpu/ggml-cpu.cpp:170 → ggml-cpu.c:3399', 'CPU 接口表第 206 行填的就是它'],
       ['compute_forward → mul_mat → one_chunk', 'ggml/src/ggml-cpu/ggml-cpu.c:1744 → 1869 → 1255', 'switch (tensor-&gt;op)'],
       ['vec_dot 内核', 'ggml-cpu.c:1182 → arch/x86/quants.c:701', 'type_traits_cpu[src0-&gt;type].vec_dot'],
       ['（对照）CUDA 同一跳', 'ggml/src/ggml-cuda/ggml-cuda.cu:2259 → 1823', '同一个 iface 槽，填的是 CUDA 实现']],
      { monoCols: [1] });
    t.el.style.fontSize = '9.5px';
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    function hi(a, b) { rows.forEach(function (r, k) { r.className = (k >= a && k <= b) ? 'on' : ''; }); }
    const texts = [
      '<span class="k">第 1 段 · 建图</span>：模型代码 → build_attn → build_lora_mm → ggml_mul_mat。图上多了一个 MUL_MAT 节点。',
      '<span class="k">第 2 段 · 构造与入图</span>：can_mul_mat 决定形状；hash set 决定它进图一次、拓扑序正确。',
      '<span class="k">第 3 段 · 调度</span>：切分 → 归属（权重在哪）→ 分配。这一步之后"归哪个后端"就定了。',
      '<span class="k">第 4 段 · 内核</span>：接口槽 → CPU 入口 → switch(op) → mul_mat → vec_dot(src0->type)。',
      '一句话：这条链上每一个箭头，都是<span class="v">某个字段的一次判断</span>。'
    ];
    tl.at(700, () => { hi(0, 2); msg.innerHTML = texts[0]; });
    tl.at(6300, () => { hi(3, 4); msg.innerHTML = texts[1]; });
    tl.at(11800, () => { hi(5, 8); msg.innerHTML = texts[2]; });
    tl.at(17300, () => { hi(9, 12); msg.innerHTML = texts[3]; });
    tl.at(22600, () => { hi(0, 12); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表：<span class="hl-a">每一层负责哪一跳</span> */
{
  kicker: "L8-01 · 收束",
  title: "把这一课压成一张表：<span class=\"hl-a\">每一层负责哪一跳</span>",
  sub: "L1~L8 各管一段；回到起点那一行，这条链就闭合了。",
  caption: "下一课 L8-02：用真实模型参数走一遍 offload 决策，看这条链在 CPU/GPU 之间怎么切。",
  src: "src/llama-graph.cpp",
  mark: [4],
  lineNo: 1514,
  code: `ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['层', '在这条链路上负责哪一跳', '本课引用的位置'],
      [['L1 算子的表示', '节点身份（op）与输出形状（ne[]）的定义', 'ggml/src/ggml.c:3333-3356'],
       ['L2 从模型到图', '把权重接到图上：build_attn → build_lora_mm', 'models/llama.cpp:169 / llama-graph.cpp:2804'],
       ['L3 后端发现与注册', '接口表把 graph_compute 槽绑到具体实现', 'ggml-backend.cpp:461 → ggml-cpu.cpp:206'],
       ['L4 内存与调度', '切分 + 归属 + buffer：决定跑在哪个后端', 'ggml-backend.cpp:1992 / 921'],
       ['L5 CPU 后端执行', '大 switch 分派 + type_traits 选 vec_dot', 'ggml-cpu.c:1744 / 1182'],
       ['L6 GPU 后端执行', '同一跳的 CUDA 实现（契约不变）', 'ggml-cuda.cu:2259 → 1823'],
       ['L7 NPU 与加速器', '同一跳的厂商实现：CANN / Hexagon / OpenVINO …', '不引用源码：见 L7-01 ~ L7-06'],
       ['L8 端到端', '把各跳串成可复述的链，并给出每跳判据', '本课（上表 13 行）']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码，复述 <span class="mono">build_attn</span> 里输出投影那次 ' +
      '<span class="mono">ggml_mul_mat</span> 的五跳，并说出每一跳的判据。',
      '<b>① 构造</b>（<span class="mono">ggml/src/ggml.c:3341</span>）：' +
      '<span class="mono">ggml_can_mul_mat</span> 判三条 ne 关系（3336-3338）→ 输出 ' +
      '<span class="mono">ne = {a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3]}</span>（3348），' +
      '<span class="mono">op = GGML_OP_MUL_MAT</span>（3351）。<br>' +
      '<b>② 入图</b>（<span class="mono">ggml.c:7337 → 7236</span>）：' +
      '<span class="mono">visited_hash_set</span> 没见过才继续 DFS（7241-7244）。<br>' +
      '<b>③ 调度</b>（<span class="mono">ggml-backend.cpp:2000 → 1066 → 921</span>）：' +
      '<span class="mono">src[0] = wo</span> 的 buffer usage 是 ' +
      '<span class="mono">GGML_BACKEND_BUFFER_USAGE_WEIGHTS</span>（967）→ ' +
      '用 <span class="mono">ggml_backend_sched_backend_from_buffer</span> 找"支持该 buffer 类型且支持该 op"的后端（888-898）。<br>' +
      '<b>④ 执行</b>（<span class="mono">ggml-backend.cpp:1799 → 461</span>）：' +
      '<span class="mono">split-&gt;backend_id</span> 选出后端（1658），调 ' +
      '<span class="mono">backend-&gt;iface.graph_compute</span>（463）。<br>' +
      '<b>⑤ 内核</b>（<span class="mono">ggml-cpu.c:1744 → 1869 → 1255 → 1460 → 1182</span>）：' +
      '<span class="mono">switch (tensor-&gt;op)</span> → <span class="mono">case GGML_OP_MUL_MAT</span> → ' +
      '<span class="mono">type_traits_cpu[src0-&gt;type].vec_dot</span>；Q4_0 绑定到 ' +
      '<span class="mono">ggml_vec_dot_q4_0_q8_0</span>（240-243，x86 实现见 ' +
      '<span class="mono">arch/x86/quants.c:701</span>）。<br>' +
      '<b>反例</b>：<span class="mono">kqv = ggml_mul_mat(v, kq)</span>（llama-graph.cpp:2715）两个输入都不是权重，' +
      '第 ③ 跳走的是"邻居/最多支持输入"路径 —— 判据不同，落点就可能不同。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '八层，各管一段；这一课把它们串成了一条线。',
      '<span class="k">L1/L2</span> 负责"建"：op、形状、src[] 与装配线。',
      '<span class="k">L3/L4</span> 负责"分"：接口契约、切分、归属、buffer。',
      '<span class="k">L5/L6/L7</span> 负责"算"：同一个 op，三套实现。',
      '<span class="k">L8</span> 负责"串"：把判据连起来，才叫读懂了这条链。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2200 + i * 2200, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
    }));
    tl.at(20000, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[4] +
        ' ｜ 全套课件门户：<a href="../../index.html">index.html</a>' +
        ' ｜ 下一课：<a href="../L8-02-offload-decision/index.html">L8-02 ★ offload 决策实战</a>';
    });
  }
},

];
