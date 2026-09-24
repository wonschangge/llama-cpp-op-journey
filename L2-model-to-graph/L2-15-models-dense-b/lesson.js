/* ==========================================================================
   L2-15 · 模型家族（六）：稠密 Transformer（下）与投机解码草稿模型
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 36 个模型文件，两类差异 */
{
  kicker: "L2 · 从模型到图",
  title: "36 个模型文件，两类差异",
  sub: "本课一个视角：这 36 个文件之间的差异，是配置的差异，还是结构的差异。",
  caption: "L2-14 已覆盖 37 个文件（稠密主干上篇）；本课覆盖规划里剩下的 36 个，全部在 src/models/ 下。",
  src: "src/models/models.h",
  mark: [4, 6, 7, 9, 14],
  lineNo: 143,
  code: `//
// models
//

struct llama_model_llama : public llama_model_base {
    llama_model_llama(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool embed>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
//>> 每个模型家族只实现三件事：读超参、读张量、给出自己的图类
};`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">GGUF 元数据</span><span class="arrow">-></span>
        <span class="chip a">llama_model_xxx 类</span><span class="arrow">-></span>
        <span class="chip c">llama_hparams</span><span class="arrow">-></span>
        <span class="chip b">llm_graph_context</span>
      </div>
      <div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '本课覆盖 36 个文件', b: '全部在 src/models/ 下。<br>其中 <b>6 个不建注意力图</b>：有声明表，也有只有几行的壳。', m: 'L2-15 覆盖域' },
      { c: 'c', t: '差异分两类', b: '<b>配置差异</b>：GQA / SWA / MLA —— 同一个 build_attn，换几个超参。<br><b>结构差异</b>：草稿模型 —— 另一套图、另一份张量清单。', m: '本课的判据' },
      { c: 'b', t: '跨课位置', b: 'L2-06 讲 build_attn 本身；L2-04 讲 SWA 的两套 cache；<br>L2-14 讲稠密主干；本课讲变体与草稿。', m: 'L2-06 / L2-04 / L2-14' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看这一类长什么样：<span class="k">一个家族 = 一个类</span>，三个虚函数。',
      '<span class="v">load_arch_hparams</span> 读元数据，<span class="v">load_arch_tensors</span> 建权重张量，<br><span class="v">build_arch_graph</span> 交出这一族自己的图。',
      '这 36 个文件全部落在最后一步：<span class="k">交出一张图</span>。所以问题变成 —— <br>这些图彼此差多少？',
      '答案是：<span class="k">大部分只差超参</span>，<span class="k">草稿模型差结构</span>。接下来九幕把这个区别拆开。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">GQA</span>：一个 struct，两个 head 数 */
{
  kicker: "L2-15 · 变体一",
  title: "★ <span class=\"hl-a\">GQA</span>：一个 struct，两个 head 数",
  sub: "Q 的头数和 K/V 的头数写在同一个结构体里 —— 这就是 GQA 的全部：不是新算子，是两个数字不同。",
  caption: "回顾 L2-06 幕 4：build_qkv 只做投影和 reshape，注意力本身仍在 build_attn 里。",
  src: "src/models/qwen2.cpp",
  mark: [0, 17, 19],
  lineNo: 67,
  code: `    auto * inp_attn = build_attn_inp_kv();
//>> 同一个 build_attn_inp_kv()，Qwen2 用它，Qwen3 / Olmo / Plamo 等 22 个文件也用它

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
//>> build_qkv 的第 5/6 个实参：Q 头数 n_head、KV 头数 n_head_kv —— GQA 的全部差异就是第 6 个实参
                    n_embd_head, n_head, n_head_kv, il);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="col" id="shapes" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const host = wrap.querySelector('#shapes');

    const rows = [
      { n: 'Qcur', s: '[ n_embd_head,  n_head,     n_tokens ]', c: 'a', b: 'n_head 个 Q 头' },
      { n: 'Kcur', s: '[ n_embd_head,  n_head_kv,  n_tokens ]', c: 'b', b: 'n_head_kv 个 KV 头' },
      { n: 'Vcur', s: '[ n_embd_head,  n_head_kv,  n_tokens ]', c: 'b', b: 'n_head_kv 个 KV 头' },
      { n: 'kqv_out', s: '[ n_embd_head,  n_head,     n_tokens ]', c: 'c', b: '输出回到 Q 的头数' }
    ];
    const els = rows.map(r => {
      const e = U.el('div', { class: 'formula', style: 'padding:5px 9px' });
      e.innerHTML = '<span style="color:var(--' + r.c + ');font-weight:600">' + U.esc(r.n) + '</span>' +
        ' <span class="m">' + U.esc(r.s) + '</span>' +
        ' <span class="m">  <- ' + U.esc(r.b) + '</span>';
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');

    const note = U.el('div', { class: 'card', style: 'border-left-color:var(--d)' });
    note.innerHTML = '<div class="ct" style="color:var(--d)">n_head / n_head_kv = 一组 Q 头共用一个 KV 头</div>' +
      '<div class="cb">这个比值就是 GQA 的<b>组大小</b>；它等于 1 时退化成 MHA。' +
      'KV cache 的体量随之按 <b>n_head_kv / n_head</b> 缩小。<br>' +
      'KV 头数在 hparams 里是<b>逐层数组</b>（<span class="cm" style="margin:0">n_head_kv_arr[]</span>），' +
      '所以同一份 build_qkv 代码能同时表达 MHA（相等）和 GQA（不等）。</div>';
    host.appendChild(note);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先说形状。这三个张量的形状决定了后面一切 —— 注意 Q 和 K/V 的第一维相同，<span class="k">第二维不同</span>。',
      '<span class="v">Qcur</span> 的第二维是 n_head；<span class="v">Kcur / Vcur</span> 的第二维是 n_head_kv。',
      'MHA 时两者相等；<span class="k">GQA 时 n_head_kv &lt; n_head</span> —— 仅此而已。',
      '<span class="v">kqv_out</span> 又回到 n_head，所以下游的 wo 投影完全不用改。',
      '结论：<span class="k">GQA 不是新算子</span>。它是 build_qkv 的一个实参 + hparams 的一个数组。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k <= i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(16200, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      note.style.opacity = '1';
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 3 <span class="hl-c">SWA</span>：窗口是一个超参，不是一层新代码 */
{
  kicker: "L2-15 · 变体二",
  title: "<span class=\"hl-c\">SWA</span>：窗口是一个超参，不是一层新代码",
  sub: "窗口大小、窗口语义、哪些层用窗口 —— 三样东西全部落在 llama_hparams 里。",
  caption: "回顾 L2-04 幕 7：SWA 层在 memory 侧被切成独立的 cache；本幕只看模型文件声明了什么。",
  src: "src/llama-hparams.h",
  mark: [1, 4, 9, 15],
  lineNo: 170,
  code: `    // Sliding Window Attention (SWA)
    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
//>> 窗口语义：NONE / STANDARD / CHUNKED / SYMMETRIC，四个枚举值
    // the size of the sliding window (0 - no SWA)
    uint32_t n_swa = 0;
//>> 窗口大小。0 表示不用 SWA

    // see llama_non_causal_type
    // note: for SWA_FULL, older tokens (outside the current ubatch) are still window-clipped
    llama_non_causal_type non_causal_type = LLAMA_NON_CAUSAL_TYPE_ALL;

    // if is_swa_impl[il] == 1, then layer il is SWA
    // if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
    // by default, all layers are dense
    // note: using uint32_t type for compatibility reason
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_swa_impl;
//>> 真正的开关：is_swa_impl 是逐层数组，一个 uint32_t 决定这一层用不用窗口`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
      <div id="pat"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: 'swa_type', b: '窗口语义，四选一。<br>STANDARD 是"往前 n_swa 个 token"；<br>CHUNKED 按块对齐；SYMMETRIC 前后各半。', m: '4 个枚举值' },
      { c: 'a', t: 'n_swa', b: '窗口宽度。<br>0 表示整层退化成全上下文。', m: 'uint32_t' },
      { c: 'd', t: 'is_swa_impl[]', b: '逐层的开关数组。<br>由 load_swa_pattern(ml, 周期) 生成，或直接从元数据读数组。', m: 'LLAMA_MAX_LAYERS 项' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:226px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const pat = wrap.querySelector('#pat');
    pat.innerHTML = '<div class="cm" style="margin:2px 0 3px">load_swa_pattern(ml, 4) -> is_swa_impl[il] = (il % 4 < 3)</div>';
    const strip = U.el('div', { class: 'row', style: 'gap:4px' });
    const cells = [];
    for (let i = 0; i < 16; i++) {
      const swa = (i % 4) < 3;
      const c = U.el('div', { style: 'width:38px;height:20px;border:1px solid var(--border);border-radius:3px;' +
        'display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:9.5px;color:' +
        (swa ? 'var(--c)' : 'var(--b)') });
      c.textContent = swa ? 'SWA' : 'FULL';
      strip.appendChild(c);
      cells.push(c);
    }
    pat.appendChild(strip);

    const msg = wrap.querySelector('#msg');
    const texts = [
      'SWA 在模型文件里只有三处声明：<span class="k">语义</span>、<span class="k">宽度</span>、<span class="k">逐层开关</span>。',
      '<span class="v">swa_type</span>：<span class="k">怎么裁</span>。源码里只有四个值，本课的 5 个 SWA 文件用到两个 —— STANDARD 和 SYMMETRIC。',
      '<span class="v">n_swa</span>：<span class="k">裁多宽</span>。它同时是 mask 的判据和 cache 的容量上限。',
      '<span class="v">is_swa_impl[]</span>：<span class="k">哪几层裁</span>。这就是"分层注意力"的全部表达。',
      '周期 4 = 每 4 层里 3 层 SWA + 1 层全上下文。<br>muse-glimmer / olmo2 / spark2-5 用 4，plamo3 用 8，modern-bert 用 3 且首层为全上下文。',
      '一句话：<span class="k">窗口是一张逐层布尔表</span> —— 不是一层新代码。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(12500, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      cells.forEach((c, i) => { if (c.textContent === 'SWA') { c.style.background = 'rgba(210,153,34,.16)'; c.style.borderColor = 'var(--c)'; } });
      msg.innerHTML = texts[4];
    });
    tl.at(15800, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 4 ★ 有 SWA 和没 SWA，共用<span class="hl-a">同一份图</span> */
{
  kicker: "L2-15 · 变体二",
  title: "★ 有 SWA 和没 SWA，共用<span class=\"hl-a\">同一份图</span>",
  sub: "Plamo3 把整份图写成一个模板：只有\"注意力输入类型\"这一个类型别名随窗口开关切换。",
  caption: "图里的实际落点只有三处：取哪个 cache、取哪张 mask、K/V 从哪来 —— 见 source.md 第七节。",
  src: "src/models/plamo3.cpp",
  mark: [1, 2, 5, 10, 20, 24, 26, 28],
  lineNo: 60,
  code: `std::unique_ptr<llm_graph_context> llama_model_plamo3::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return std::make_unique<graph<true>> (*this, params);
//>> 有窗口 -> 用 iswa 版图（两套 cache）
    } else {
        return std::make_unique<graph<false>>(*this, params);
//>> 没有窗口 -> 用普通版图（一套 cache）。两者之间只有模板实参不同
    }
}

template <bool iswa>
llama_model_plamo3::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t head_dim_q = hparams.n_embd_head_k();
    const int64_t head_dim_v = hparams.n_embd_head_v();

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
//>> 整份图共用的类型别名：llm_graph_input_attn_kv_iswa 还是 llm_graph_input_attn_kv
    inp_attn_type * inp_attn = nullptr;

    if constexpr (iswa) {
//>> if constexpr —— 编译期二选一，下面的 for 循环体一个字都不用改
        inp_attn = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="br" style="gap:8px"></div>
      <div id="pts"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const br = wrap.querySelector('#br');
    const left = U.card({ c: 'c', t: 'swa_type != LLAMA_SWA_TYPE_NONE',
      b: '两条 return 的全部差异：<br>图类变成 <b>graph&lt;true&gt;</b>，<br>注意力输入变成 iswa 版。', m: 'graph<true>' },
      { style: 'width:296px' });
    const right = U.card({ c: 'b', t: 'swa_type == LLAMA_SWA_TYPE_NONE',
      b: '同一个图类模板的另一个实参：<br><b>graph&lt;false&gt;</b>，<br>注意力输入是普通 KV 版。', m: 'graph<false>' },
      { style: 'width:296px' });
    br.appendChild(left); br.appendChild(U.arrow('vs')); br.appendChild(right);
    [left, right].forEach(e => e.style.opacity = '.30');

    const pts = wrap.querySelector('#pts');
    pts.innerHTML = '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">图侧的三处三元表达式（llama-graph.cpp:3133 / 3148 / 3151-3152）</div>' +
      '<div class="cb">' +
      '<div class="cm" style="margin:2px 0">mctx_cur = is_swa ? mctx_iswa-&gt;get_swa() : mctx_iswa-&gt;get_base()</div>' +
      '<div class="cm" style="margin:2px 0">kq_mask  = is_swa ? inp-&gt;get_kq_mask_swa() : inp-&gt;get_kq_mask()</div>' +
      '<div class="cm" style="margin:2px 0">k = mctx_cur-&gt;get_k(ctx0, il);  v = mctx_cur-&gt;get_v(ctx0, il);</div>' +
      '<div style="margin-top:4px">选哪套 cache、选哪张 mask、K/V 从哪读 —— 有窗口和没窗口的图，' +
      '差别到此为止。</div></div></div>';
    pts.style.opacity = '.30';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'Plamo3 的入口先做一件事：<span class="k">按 swa_type 挑模板实参</span>。',
      '两条分支指向<span class="k">同一个图类模板</span> —— graph&lt;true&gt; 与 graph&lt;false&gt;。',
      '图体本身一个字都没变：<span class="v">build_inp_embd</span>、循环、<span class="v">build_qkv</span>、<span class="v">build_attn</span> 全部共用。',
      '唯一随实参变化的是<span class="k">注意力输入的类型别名</span>和一行 if constexpr。',
      '对比第 2 幕：GQA 换的是实参，SWA 换的是类型 —— 但两者都<span class="k">没有新算子</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; left.style.opacity = '1'; right.style.opacity = '1'; });
    tl.at(4100, () => { msg.innerHTML = texts[1]; });
    tl.at(7600, () => { msg.innerHTML = texts[2]; pts.style.opacity = '1'; });
    tl.at(11600, () => { msg.innerHTML = texts[3]; });
    tl.at(16600, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 <span class="hl-e">MLA</span>：压缩 KV，只体现为一个布尔量 */
{
  kicker: "L2-15 · 变体三",
  title: "<span class=\"hl-e\">MLA</span>：压缩 KV，只体现为一个布尔量",
  sub: "MLA 把 K/V 压成一份潜在向量再解压。在图这一层，它的全部痕迹是\"V 那一张张量不分配\"。",
  caption: "回顾 L2-04 幕 6：cache 自己拿 buffer，图只拿视图。MLA 少一张张量，就是少一份 buffer。",
  src: "src/llama-kv-cache.cpp",
  mark: [0, 1, 4, 5],
  lineNo: 230,
  code: `        const bool has_k = true;
        const bool has_v = !is_mla;
//>> is_mla 在构造函数上方第 163 行算出：const bool is_mla = hparams.is_mla();

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;
//>> has_v 为假时 v 就是 nullptr —— 后续所有 v_stream / get_v 都跟着走空路

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="lays"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const host = wrap.querySelector('#lays');

    const mk = (title, color, boxes, note) => {
      const d = U.el('div', { class: 'card', style: 'border-left-color:var(--' + color + ');width:100%' });
      d.innerHTML = '<div class="ct" style="color:var(--' + color + ')">' + title + '</div>' +
        '<div class="cb">' + note + '</div>';
      const row = U.el('div', { class: 'row', style: 'gap:6px;margin-top:6px' });
      boxes.forEach(b => {
        const e = U.el('div', { style: 'flex:1 1 auto;height:30px;border:1px dashed var(--' + b.c + ');border-radius:4px;' +
          'display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:9.5px;color:var(--' +
          b.c + ');opacity:' + (b.on ? '1' : '.45') });
        e.textContent = b.t;
        row.appendChild(e);
      });
      d.appendChild(row);
      host.appendChild(d);
      return d;
    };

    const a = mk('普通注意力：has_v = !is_mla = true', 'b',
      [{ t: 'cache_k_l0', c: 'b', on: 1 }, { t: 'cache_v_l0', c: 'b', on: 1 }],
      '每层两张张量：K 一张、V 一张。K 的头数是 n_embd_k_gqa，V 是 n_embd_v_gqa。');
    const b = mk('MLA：has_v = !is_mla = false', 'e',
      [{ t: 'cache_k_l0（压缩后的潜在向量）', c: 'e', on: 1 }, { t: 'v = nullptr（不分配）', c: 'dim', on: 0 }],
      '每层只有一张：K 里放的是压缩后的潜在向量，V 由它现场解压出来。');

    [a, b].forEach(d => d.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'MLA 不是一个新算子，也不是一条新的 build_attn 分支。',
      '它只改一件事：<span class="k">这一层的 V 张量到底分不分配</span>。',
      '<span class="v">is_mla()</span> 的判据是两个超参非零：<span class="v">n_embd_head_k_mla_impl</span> 与 <span class="v">n_embd_head_v_mla_impl</span>。',
      '于是 cache 构造里出现两行：<span class="v">has_k = true</span>、<span class="v">has_v = !is_mla</span>。<br>只差一个布尔量，省下的是整层 V 的 buffer。',
      '上一幕的 SWA 让图换了一个类型别名，这一幕的 MLA 让 cache 少一张张量 ——<br>两者都还是<span class="k">配置</span>，不是新结构。'
    ];
    tl.at(600, () => { a.style.opacity = '1'; msg.innerHTML = texts[0]; });
    tl.at(3900, () => { b.style.opacity = '1'; msg.innerHTML = texts[1]; });
    tl.at(7600, () => { msg.innerHTML = texts[2]; });
    tl.at(11300, () => { msg.innerHTML = texts[3]; });
    tl.at(15800, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 一张表：36 个文件各自用哪种注意力 */
{
  kicker: "L2-15 · 全景",
  title: "一张表：36 个文件各自用哪种注意力",
  sub: "按\"注意力输入类型\"分五组。每一行的依据行，都是该文件里真正写下的那一行。",
  caption: "依据行写法：数字 = 该文件行号；hN = src/models/models.h 第 N 行（复用与声明在那里）。",
  src: "src/llama-hparams.h",
  mark: [1, 3, 5],
  lineNo: 24,
  code: `enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,
//>> 本课的 5 个窗口文件里，4 个用 STANDARD、1 个用 SYMMETRIC；CHUNKED 没有出现
    LLAMA_SWA_TYPE_STANDARD  = 1,
    LLAMA_SWA_TYPE_CHUNKED   = 2,
    LLAMA_SWA_TYPE_SYMMETRIC = 3,
//>> SYMMETRIC 只被 modern-bert 用到（一个没有 KV cache 的编码器）
};`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div class="row" id="tbl" style="gap:9px;align-items:flex-start"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const host = wrap.querySelector('#tbl');

    const rowsA = [
      { f: 'maincoder.cpp', v: 'KV', l: '61', g: 'kv' },
      { f: 'mpt.cpp', v: 'KV', l: '78', g: 'kv' },
      { f: 'nanbeige.cpp', v: 'KV', l: '97', g: 'kv' },
      { f: 'nemotron.cpp', v: 'KV', l: '64', g: 'kv' },
      { f: 'olmo.cpp', v: 'KV', l: '57', g: 'kv' },
      { f: 'orion.cpp', v: 'KV', l: '57', g: 'kv' },
      { f: 'paddleocr.cpp', v: 'KV', l: '28', g: 'kv' },
      { f: 'pangu-embed.cpp', v: 'KV', l: '72', g: 'kv' },
      { f: 'phi2.cpp', v: 'KV', l: '62', g: 'kv' },
      { f: 'plamo.cpp', v: 'KV', l: '53', g: 'kv' },
      { f: 'qwen.cpp', v: 'KV', l: '56', g: 'kv' },
      { f: 'qwen2.cpp', v: 'KV', l: '67', g: 'kv' },
      { f: 'qwen2vl.cpp', v: 'KV', l: '56', g: 'kv' },
      { f: 'qwen3.cpp', v: 'KV', l: '67', g: 'kv' },
      { f: 'qwen3vl.cpp', v: 'KV', l: '80', g: 'kv' },
      { f: 'seed-oss.cpp', v: 'KV', l: '63', g: 'kv' },
      { f: 'smollm3.cpp', v: 'KV', l: '60', g: 'kv' },
      { f: 'talkie.cpp', v: 'KV', l: '58', g: 'kv' }
    ];
    const rowsB = [
      { f: 'starcoder.cpp', v: 'KV', l: '73', g: 'kv' },
      { f: 'starcoder2.cpp', v: 'KV', l: '73', g: 'kv' },
      { f: 'stablelm.cpp', v: 'KV', l: '64', g: 'kv' },
      { f: 'openelm.cpp', v: 'KV', l: '62', g: 'kv' },
      { f: 'xverse.cpp', v: 'KV', l: '55', g: 'kv' },
      { f: 'muse-glimmer.cpp', v: 'KV+SWA', l: '73', g: 'swa' },
      { f: 'spark2-5.cpp', v: 'KV+SWA', l: '64', g: 'swa' },
      { f: 'olmo2.cpp', v: '两个模板', l: '77', g: 'swa' },
      { f: 'plamo3.cpp', v: '两个模板', l: '78', g: 'swa' },
      { f: 'modern-bert.cpp', v: '无 cache', l: '91', g: 'enc' },
      { f: 'neo-bert.cpp', v: '无 cache', l: '56', g: 'enc' },
      { f: 't5.cpp', v: '无 cache+交叉', l: '277', g: 'enc' },
      { f: 'mistral4.cpp', v: '复用 MLA 图', l: 'h1397', g: 'shell' },
      { f: 'qwen3tts.cpp', v: '复用 qwen3vl', l: 'h625', g: 'shell' },
      { f: 't5encoder.cpp', v: '复用 t5 图', l: 'h1478', g: 'shell' },
      { f: 'nomic-bert.cpp', v: '复用 bert 图', l: 'h330', g: 'shell' },
      { f: 'wavtokenizer-dec.cpp', v: 'posnet+convnext', l: '118', g: 'shell' },
      { f: 'models.h', v: '151 个类声明', l: '147', g: 'shell' }
    ];
    const H = ['文件', '变体', '依据行'];
    const tA = U.table(H, rowsA.map(r => [r.f, r.v, r.l]), { monoCols: [0, 2] });
    const tB = U.table(H, rowsB.map(r => [r.f, r.v, r.l]), { monoCols: [0, 2] });
    const dA = U.el('div', { style: 'width:341px;flex:0 0 auto' }, tA.el);
    const dB = U.el('div', { style: 'width:341px;flex:0 0 auto' }, tB.el);
    host.appendChild(dA); host.appendChild(dB);

    const trA = tA.body.querySelectorAll('tr'), trB = tB.body.querySelectorAll('tr');
    const all = [];
    rowsA.forEach((r, i) => { r.tr = trA[i]; all.push(r); });
    rowsB.forEach((r, i) => { r.tr = trB[i]; all.push(r); });
    const show = g => all.forEach(r => { r.tr.className = (g === 'all' || r.g === g) ? 'on' : ''; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '36 行，五组。先整体看一眼：<span class="k">最右边那列是依据行</span>，可以逐条回源码核。',
      '<span class="k">23 个文件</span>写的是 build_attn_inp_kv() —— 稠密因果 + GQA 是默认形态。',
      '<span class="k">4 个文件</span>碰窗口：muse-glimmer / spark2-5 直接选 iswa，olmo2 / plamo3 用模板实参二选一。',
      '<span class="k">3 个文件</span>没有 KV cache：两个 BERT 编码器，加上 t5 的编码器分支（t5 另有交叉注意力）。',
      '<span class="k">6 个文件</span>不建注意力图：4 个壳 + 一个卷积音频解码器 + 一张 151 个类的声明表。',
      '把这张表和实现对照：<span class="v">23 + 4 + 3</span> 个建图的文件里，<span class="k">没有一种新算子</span>。'
    ];
    tl.at(600, () => { show('all'); msg.innerHTML = texts[0]; });
    tl.at(4300, () => { show('kv'); msg.innerHTML = texts[1]; });
    tl.at(8600, () => { show('swa'); msg.innerHTML = texts[2]; });
    tl.at(12900, () => { show('enc'); msg.innerHTML = texts[3]; });
    tl.at(17200, () => { show('shell'); msg.innerHTML = texts[4]; });
    tl.at(21200, () => { show('all'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 文件名叫一个模型，内容可能<span class="hl-d">一行图都没有</span> */
{
  kicker: "L2-15 · 名实不符",
  title: "文件名叫一个模型，内容可能<span class=\"hl-d\">一行图都没有</span>",
  sub: "models.h 里有 151 个模型类；其中一批用 using graph = ... 直接把别族的图借过来。",
  caption: "Mistral4 是 DeepSeek2 的子类：它连 load_arch_hparams / load_arch_tensors 都不重写。",
  src: "src/models/models.h",
  mark: [0, 3, 5],
  lineNo: 1393,
  code: `struct llama_model_mistral4 : public llama_model_deepseek2 {
//>> 继承自 llama_model_deepseek2 —— MLA 家族的图就是它的图
    llama_model_mistral4(const struct llama_model_params & params) : llama_model_deepseek2(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_deepseek2

    using graph = llama_model_deepseek2::graph;
//>> 一行别名，把整份 MLA 图借过来。mistral4.cpp 因此只剩 6 行

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'd', t: 'mistral4.cpp', b: '6 行。<br>只有 build_arch_graph，转调 deepseek2 的图。', m: 'using graph = llama_model_deepseek2::graph' },
      { c: 'g', t: 'qwen3tts.cpp', b: '3 行，其中一行是注释。<br>models.h 里直接继承 qwen3vl。', m: 'struct llama_model_qwen3tts : public llama_model_qwen3vl' },
      { c: 'a', t: 't5encoder.cpp', b: '44 行，只加载编码器张量。<br>图用 t5 的编码器那一半。', m: 'using graph = llama_model_t5::graph<true>' },
      { c: 'f', t: 'nomic-bert.cpp', b: '51 行，只读超参和张量。<br>图整份借用 bert。', m: 'using graph = llama_model_bert::graph' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一课的第一个"名实不符"：<span class="k">文件名叫一个模型，内容可能不建图</span>。',
      '<span class="v">mistral4.cpp</span> 全文 6 行 —— 因为它是 deepseek2 的子类，超参、张量、图三样全部继承。',
      '<span class="v">qwen3tts.cpp</span> 只有 3 行，其中一行还是注释。<br>它的"复用"写在 models.h 的继承列表里，不在 .cpp 里。',
      '<span class="v">t5encoder.cpp</span> 与 <span class="v">nomic-bert.cpp</span> 是另一种借法：用 using 起一个别名，指向别族的图类。',
      '还有两个不成"壳"的：<span class="v">wavtokenizer-dec.cpp</span> 264 行却没有 build_attn（posnet + convnext），<br><span class="v">models.h</span> 2662 行、151 个模型类，是本课唯一不建图的源文件。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 草稿模型（一）：主模型先把<span class="hl-c">每层输入</span>导出来 */
{
  kicker: "L2-15 · 草稿模型",
  title: "草稿模型（一）：主模型先把<span class=\"hl-c\">每层输入</span>导出来",
  sub: "草稿模型要\"猜到主模型下一层会看到什么\"，所以主模型的图必须把中间层输入挂成图输出。",
  caption: "三个目标模型文件这么做：qwen3.cpp:72 / nanbeige.cpp:106 / muse-glimmer.cpp:80。",
  src: "src/models/qwen3.cpp",
  mark: [5, 10, 11],
  lineNo: 62,
  code: `    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();
//>> 主模型自己的注意力输入不变，和 L2-14 的稠密模型一模一样

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
//>> res->t_layer_inp[il] 记下第 il 层的输入；llm_graph_result 在 set_outputs 里对它调 ggml_set_output（llama-graph.cpp:1379）

        ggml_tensor * inpSA = inpL;`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip c">inpL（第 il 层输入）</span><span class="arrow">-></span>
        <span class="chip a">res->t_layer_inp[il]</span><span class="arrow">-></span>
        <span class="chip d">ggml_set_output</span><span class="arrow">-></span>
        <span class="chip b">草稿模型的编码器输入</span>
      </div>
      <div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '目标侧：3 个文件', b: 'qwen3.cpp、nanbeige.cpp、muse-glimmer.cpp 在层循环开头写一行<br><span class="cm" style="margin:0">res-&gt;t_layer_inp[il] = inpL;</span>', m: '导出层输入' },
      { c: 'b', t: '草稿侧：models.h 声明', b: 'llama_model_eagle3 / llama_model_dflash 是<b>独立的模型类</b>，<br>各有一个 template&lt;bool is_enc&gt; 的图。', m: 'models.h:1357-1390' },
      { c: 'd', t: '接口侧：ctx_other', b: '草稿 context 通过 cparams.ctx_other 指回主模型 context，<br>才能借用它的 tok_embd / lm_head / KV。', m: 'llama-cparams.h:66' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '草稿模型不是"小一号的主模型"，它吃的输入不一样。',
      '它要读主模型<span class="k">若干个中间层</span>的输入。所以主模型必须在建图时把这些张量挂成输出。',
      '<span class="v">t_layer_inp[il]</span> 就是这个挂点：一出循环就被 <span class="v">ggml_set_output</span> 钉住，<br>运行时由 context 抽出来（llama-context.cpp 的 extract_layer_inputs）。',
      '注意：主模型的图<span class="k">一行都没为草稿改</span> —— 只是多存了一个指针。<br>这是"结构差异"里最轻的一处。',
      '接下来两幕看重的部分：草稿模型自己的图，和它跟主模型共享的 KV。'
    ];
    tl.at(600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; });
    tl.at(4000, () => { msg.innerHTML = texts[1]; });
    tl.at(7600, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[2]; });
    tl.at(11400, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(14800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 ★ 共享 KV 的图差异：<span class="hl-a">不传 K/V</span> 的 build_attn */
{
  kicker: "L2-15 · 草稿模型",
  title: "★ 共享 KV 的图差异：<span class=\"hl-a\">不传 K/V</span> 的 build_attn",
  sub: "草稿模型只算 Q，把 k_cur / v_cur 两个实参写成 nullptr —— 于是它一个 K/V 张量都不需要。",
  caption: "对照第 2 幕：稠密模型传的是 Qcur, Kcur, Vcur 三个张量；这里只有第一个非空。",
  src: "src/models/gemma4-assistant.cpp",
  mark: [0, 2, 3, 11, 13],
  lineNo: 140,
  code: `        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur_norm);
//>> 只有 wq：这一族的权重清单里没有 wk / wv（见 gemma4-assistant.cpp:59-60）
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        ggml_tensor * freq_factors = is_swa ? nullptr : model.layers[il].rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
//>> 第 6、7 个实参是 k_cur / v_cur —— 两个 nullptr
                Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
//>> build_attn 里 if (k_cur) / if (v_cur) 两段写 cache 的代码整段跳过（llama-graph.cpp:3136/3142）`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="chain" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'd', t: '① 谁跟谁共享', b: '草稿 context 的 <b>cparams.ctx_other</b> 指回主模型 context —— 在 common/speculative.cpp:2555 赋值。', m: 'cparams.ctx_other' },
      { c: 'c', t: '② 建 memory 时给出层映射', b: 'create_memory 为草稿装 <b>share</b> 回调：SWA 层映射到主模型倒数第 2 层，其余映射到最后 1 层。', m: 'llama-model.cpp:2690-2698' },
      { c: 'b', t: '③ cache 里不分配，直接挂上', b: 'share && other 命中时执行 <b>layers.push_back(layer_share)</b> —— 草稿层拿到的是主模型层的同一个 K/V 张量。', m: 'llama-kv-cache.cpp:176-190' },
      { c: 'a', t: '④ 图上只差两个实参', b: 'build_attn(..., Qcur, nullptr, nullptr, ...)：K/V 从共享张量读，草稿自己不写 cache。', m: 'gemma4-assistant.cpp:150-151' }
    ];
    const host = wrap.querySelector('#chain');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:340px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '共享 KV 不是一句口号，是四步：<span class="k">谁跟谁共享 -> 层怎么映射 -> 张量怎么挂上 -> 图上少传什么</span>。',
      '第一步：草稿 context 通过 <span class="v">cparams.ctx_other</span> 拿到主模型 context。',
      '第二步：<span class="v">create_memory</span> 给草稿的 cache 一个 <span class="v">share</span> 回调，<br>把草稿的第 il 层映射到主模型的某一层。',
      '第三步：cache 构造函数命中 share 时<span class="k">不 new 张量</span>，直接把主模型那一层的 k/v 指针 push 进自己的 layers。',
      '第四步：图侧的证据 —— 草稿的 <span class="v">build_attn</span> 把 k_cur / v_cur 都传 nullptr，<br>因为它根本没有 wk / wv 权重可算。',
      '所以"共享 KV"的图差异一句话就能说完：<span class="k">稠密模型传三个张量，共享 KV 的草稿只传一个</span>。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    els.forEach((e, i) => tl.at(4300 + i * 3900, () => {
      els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(21300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表 */
{
  kicker: "L2-15 · 收束",
  title: "把这一课压成一张表",
  sub: "注意力变体改的是配置，草稿模型改的是结构 —— 这就是 L2-06 那句结论的边界。",
  caption: "下一课 L3-01：后端接口。图建好之后，交给谁执行。",
  src: "src/models/models.h",
  mark: [5, 7, 10, 14],
  lineNo: 1357,
  code: `struct llama_model_eagle3 : public llama_model_base {
    llama_model_eagle3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
//>> template <bool is_enc>：草稿模型的图分"编码器"与"解码器"两半，这是主模型没有的形态
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
//>> build_inp_embd_enc() —— 草稿模型特有的输入：主模型中间层的特征
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['差异面', '稠密主模型', '注意力变体', '草稿模型'],
      [['Q/K/V 头数', 'n_head = n_head_kv', 'GQA：n_head_kv < n_head', '同主模型（借它的头数）'],
       ['注意力范围', '全上下文', 'SWA：每层一个窗口开关', '随主模型的层（SWA 层映射到 SWA 层）'],
       ['KV 张量', 'K 一张 + V 一张', 'MLA：只有 K 一张', '一张都不建，挂主模型的'],
       ['层数', 'n_layer 层', '不变', 'n_layer_nextn 层（草稿层）'],
       ['权重清单', 'wq / wk / wv / wo', '不变', '只有 wq / wo（没有 wk / wv）'],
       ['图代码差异', '——', '一个类型别名 / 一个布尔量', 'build_attn 少传两个实参']],
      { monoCols: [] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一个共享主模型 KV 的草稿模型，它的 <span class="mono">build_attn</span> 调用和本课第 2 幕那个稠密模型' +
      '（<span class="mono">qwen2.cpp</span>）有什么不同？为什么这个不同是<b>结构性</b>的？',
      '稠密模型传 <span class="mono">Qcur, Kcur, Vcur</span> 三个张量；共享 KV 的草稿模型把中间两个写成 ' +
      '<span class="mono">nullptr</span>（<span class="mono">gemma4-assistant.cpp:150-151</span>）。<br>' +
      '于是 <span class="mono">build_attn</span> 里 <span class="mono">if (k_cur)</span> / <span class="mono">if (v_cur)</span> ' +
      '两段写 cache 的代码整段跳过，K/V 直接来自 <span class="mono">mctx_cur-&gt;get_k()/get_v()</span>；' +
      '而这两个张量是主模型的层张量（<span class="mono">llama-kv-cache.cpp:180-187</span>）。<br>' +
      '说它结构性，是因为草稿模型的<b>权重清单里压根没有 wk / wv</b>（只有 wq 和 wo，' +
      '<span class="mono">gemma4-assistant.cpp:59-60</span>）—— 这不是把某个超参调小，' +
      '而是这张图少了两个投影、少了两个实参、少了自己的一份 KV。<br>' +
      '对照 GQA / SWA / MLA：它们改完后 build_attn 的实参个数和算子清单都不变。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '五行差异，前两行是配置，后三行是结构：',
      '<span class="k">头数</span>与<span class="k">范围</span>：GQA 换实参，SWA 换类型别名，MLA 换一个布尔量。',
      '<span class="k">KV 张量</span>：MLA 只是少一张，草稿模型是一张都不建。',
      '<span class="k">层数与权重清单</span>：草稿模型的 wk / wv 根本不存在 —— 这是结构差异。',
      '所以 L2-06 幕 10 的结论要加一条边界：<span class="v">"差异 = 原语 + 超参"</span> 对注意力变体成立，<br>对草稿模型不成立。',
      '下一课 L3-01 换一个视角：图建好之后，它被切给哪个后端、按什么接口执行。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2900 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(15600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
    tl.at(18900, () => { msg.innerHTML = texts[5]; });
  }
},

];
