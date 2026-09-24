/* ==========================================================================
   L2-12 · 模型家族（三）：稀疏专家 MoE（上）
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 24 个文件，共用<span class="hl-a">同一张子图</span> */
{
  kicker: "L2-12 · 全局",
  title: "24 个文件，共用<span class=\"hl-a\">同一张子图</span>",
  sub: "MoE 家族的全部差异都在\"怎么选专家\"，而不在\"专家怎么算\"—— 后者被抽成了一个原语。",
  caption: "回顾 L2-06：`build_moe_ffn()` 是 llm_graph_context 的成员函数，和 build_ffn / build_norm 并列。",
  src: "src/llama-graph.cpp",
  mark: [0, 2, 4, 7, 9, 11, 16],
  lineNo: 1949,
  code: `ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
//>> gate_inp = ffn_gate_inp：路由打分矩阵 [n_embd, n_expert]
         ggml_tensor * up_exps,
//>> up_exps / gate_exps / down_exps：三个"多了一维"的专家权重张量
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
//>> n_expert：这一层总共有多少专家（= 专家张量最后一维的长度）
             int64_t   n_expert_used,
//>> n_expert_used：每个 token 选几个（top-k 的 k）
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
//>> gating_op：softmax / sigmoid / sqrt_softplus / softmax_weight
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="row wrap" id="cards" style="gap:9px"></div>
      <div class="formula" id="fam" style="font-size:9.5px;overflow-wrap:anywhere;line-height:1.6"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '24 个 MoE 模型文件', b: '按静态扫描分组：图构建代码命中<br>' +
          '<span class="cm" style="margin:0">build_moe_ffn | ffn_gate_exps | ffn_gate_inp</span>',
        m: '分组依据 ≠ 内容断言' },
      { c: 'b', t: '一个公共图原语', b: '所有 MoE 模型都调它；<br>家族差异只落在它的参数上',
        m: 'llm_graph_context::build_moe_ffn' },
      { c: 'c', t: '定义在 L2-06 的地盘', b: 'src/llama-graph.cpp:1993<br>本课按并集声明该文件，不影响 L2-06',
        m: 'src/llama-graph.cpp' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const names = ['afmoe','arctic','bailingmoe','bailingmoe2','bert','cohere2moe','dbrx','deci',
      'deepseek','deepseek2ocr','dots1','ernie4-5-moe','ernie4-5','exaone-moe','glm4-moe',
      'granite-moe','granite-swa','granite','grok','grovemoe','hunyuan-moe','hy-v3','laguna','lfm2moe'];
    wrap.querySelector('#fam').innerHTML =
      '<span class="m">本课逐字引用并声明的 24 个文件：</span><br>' +
      names.map(n => '<span style="color:var(--b)">src/models/' + n + '.cpp</span>').join(' <span class="m">·</span> ');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '24 个文件，看起来是 24 个模型；<br>但它们<b>共用同一个图原语</b> —— 这就是本课要讲清的那一件事。',
      '<span class="v">build_moe_ffn</span> 的签名有 <span class="k">24 个参数</span>（另有 19 个参数的简版重载）：<br>' +
      '5 个张量（gate_inp + 三组 exps + exp_probs_b）、<span class="k">2 个整数</span>（n_expert / n_expert_used）、' +
      '门控与激活类型、归一化/缩放开关，以及一批可选张量。',
      '<span class="k">家族的差异全部落在参数上</span>：<br>' +
      '谁有 gate_exps、谁用 sigmoid、谁有分组路由、谁有共享专家 —— 都是传参不同，不是另写一张图。',
      '这就是"稀疏"在图上的全部含义：<span class="v">专家张量多一维</span> + <span class="v">按索引收集</span>。<br>' +
      'L2-13（MoE 下篇）会逐个模型看这些参数怎么被填。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14500, () => {
      els.forEach(e => e.style.opacity = '1');
      U.markLines(document, [0, 2, 3, 5, 7, 8, 12]);
      msg.innerHTML = texts[3];
    });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">ffn_gate_inp</span>：一次打分，就是一次 <span class="hl-a">ggml_mul_mat</span> */
{
  kicker: "L2-12 · 算子 1/7",
  title: "★ <span class=\"hl-a\">ffn_gate_inp</span>：一次打分，就是一次 <span class=\"hl-a\">ggml_mul_mat</span>",
  sub: "路由不是特殊算子：它就是把隐状态乘上 [n_embd, n_expert] 的打分矩阵。",
  caption: "回顾 L1-02：`ggml_mul_mat` 是一个普通的 `GGML_OP_*`，路由打分用的是它，专家里用的也是它的变体。",
  src: "src/llama-graph.cpp",
  mark: [0, 2, 9, 14, 21],
  lineNo: 2018,
  code: `    const int64_t n_embd   = cur->ne[0];
//>> n_embd = cur 的第 0 维（最内层，连续）
    const int64_t n_tokens = cur->ne[1];
//>> n_tokens = cur 的第 1 维；下面所有张量都带这一维
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
//>> build_lora_mm 内部就是 ggml_mul_mat；LoRA 只是可选分支
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
//>> cb() 给张量起调试名字 "ffn_moe_logits"，图 dump 时能看到
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['张量', '怎么来', '形状', '产生的算子'],
      [['cur', 'ffn_norm 的输出（上游）', '[n_embd, n_tokens]', '（上游）'],
       ['ffn_gate_inp', 'load_arch_tensors 里的权重张量', '[n_embd, n_expert]', 'create_tensor'],
       ['logits', 'build_lora_mm(gate_inp, cur)', '[n_expert, n_tokens]', 'ggml_mul_mat'],
       ['probs', 'ggml_soft_max(logits)', '[n_expert, n_tokens]', 'ggml_soft_max']],
      { monoCols: [0, 2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先把"路由"这件事拆成两个张量：<span class="v">gate_inp</span>（权重）和 <span class="v">cur</span>（数据）。',
      '<span class="v">ffn_gate_inp</span> 的形状是 <span class="k">[n_embd, n_expert]</span>：<br>' +
      '一行一个专家，代表"这个专家喜欢什么样的输入"。',
      '<span class="v">logits = build_lora_mm(gate_inp, cur)</span>：<br>' +
      '注意 build_lora_mm 内部就是 <span class="k">ggml_mul_mat</span>（grovemoe.cpp:133 直接这么写）—— ' +
      '路由打分的算子身份和普通 FFN 的 up/gate 投影<b>完全一样</b>。',
      '结果形状 <span class="v">[n_expert, n_tokens]</span>：<br>' +
      '每一列是一个 token 对全部专家的打分。这就是后面 top-k 的输入。',
      '所以"MoE 有特殊算子"是个错觉：<span class="k">打分 = ggml_mul_mat，收尾 = ggml_add</span>，都是 L1-02 里的老算子。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2900 + i * 3000, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(15200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 门控函数：<span class="hl-c">softmax</span> / <span class="hl-c">sigmoid</span> / <span class="hl-c">sqrt(softplus)</span> */
{
  kicker: "L2-12 · 算子 2/7",
  title: "门控函数：<span class=\"hl-c\">softmax</span> / <span class=\"hl-c\">sigmoid</span> / <span class=\"hl-c\">sqrt(softplus)</span>",
  sub: "打分之后要变成概率。选哪个函数是模型架构决定的，不是引擎决定的。",
  caption: "`exp_probs_b` 只加到\"选择\"用的那份概率上；用来算权重的 `probs` 保持无偏 —— 源码注释写明了原因。",
  src: "src/llama-graph.cpp",
  mark: [2, 4, 7, 9, 12, 14, 17, 19, 28, 30],
  lineNo: 2039,
  code: `    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
//>> 最常见：DeepSeek-V2 / DBRX / Qwen3-MoE 都用 softmax
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
//>> sigmoid：逐专家独立打分，总和不必为 1（Hy-V3 / Grovemoe 一类）
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
//>> SOFTMAX_WEIGHT：此处原样透传，延迟到第 6 幕的 weights 上再做 softmax
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
//>> Grovemoe 特殊：选择概率直接用 sigmoid(logits)，覆盖上面的结果
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', k: 'SOFTMAX', m: 'ggml_soft_max', b: '整行归一化，<br>概率和 = 1。<br>最主流。' },
      { c: 'd', k: 'SIGMOID', m: 'ggml_sigmoid', b: '逐元素独立，<br>和不必为 1。<br>DeepSeek-V3 起。' },
      { c: 'f', k: 'SQRT_SOFTPLUS', m: 'ggml_sqrt(ggml_softplus(x))', b: '开方 + softplus，<br>非负。<br>少见。' },
      { c: 'e', k: 'SOFTMAX_WEIGHT', m: 'probs = logits', b: '此处不门控，<br>挪到 weights 上做。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card' });
      e.style.borderLeftColor = 'var(--' + d.c + ')';
      e.style.width = '163px';
      e.innerHTML = '<div class="cm" style="margin:0 0 2px">' + U.esc(d.k) + '</div>' +
        '<div class="ct" style="color:var(--' + d.c + ');font-size:15px">' + U.esc(d.m) + '</div>' +
        '<div class="cb">' + d.b + '</div>';
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="k">gating_op</span> 是一个 4 值枚举，switch 出四种概率化方式。',
      '<span class="v">SOFTMAX</span> -> <span class="m">ggml_soft_max(logits)</span>：全局归一化。',
      '<span class="v">SIGMOID</span> -> <span class="m">ggml_sigmoid(logits)</span>：每个专家独立打分。',
      '<span class="v">SQRT_SOFTPLUS</span> -> <span class="m">ggml_sqrt(ggml_softplus(logits))</span>：' +
      '两个一元算子叠起来，都是 L1-02 里已有的 op。',
      '<span class="v">SOFTMAX_WEIGHT</span>：这一步 <span class="k">probs = logits</span> 什么都不做，' +
      '等 top-k 选完再对 <span class="v">weights</span> 做 softmax（第 6 幕）。'
    ];
    const mk = [[2], [4], [6], [8]];
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, mk[i]);
      msg.innerHTML = texts[i];
    }));
    tl.at(18800, () => {
      els.forEach(e => e.style.opacity = '1');
      U.markLines(document, [2, 4, 6, 8, 10]);
      msg.innerHTML = '然后是<b>选择偏置</b>：<span class="v">selection_probs = ggml_add(probs, exp_probs_b)</span>。' +
        '<br>源码注释点明：偏置只影响"选谁"，<span class="k">probs 本身保持无偏</span>，' +
        '因为它后面还要用来算专家权重。';
    });
  }
},

/* ------------------------------------------------------ 4 分组路由：<span class="hl-d">n_expert_groups</span> / <span class="hl-d">n_group_used</span> */
{
  kicker: "L2-12 · 算子 3/7",
  title: "分组路由：<span class=\"hl-d\">n_expert_groups</span> / <span class=\"hl-d\">n_group_used</span>",
  sub: "DeepSeek-V3 式的\"先选组、再选专家\"：把 n_expert 个专家切成若干组，只在被选中的组里做 top-k。",
  caption: "这两个参数不在 build_moe_ffn 的形参里 —— 它们来自 `hparams`（`src/llama-hparams.h`，属 L2-02，本课不计入覆盖）。",
  src: "src/llama-graph.cpp",
  mark: [2, 4, 7, 10, 15, 18, 24, 26],
  lineNo: 2081,
  code: `    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
//>> n_expert_groups <= 1 时整段跳过 —— 绝大多数 MoE 模型走的就是这条路
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]
//>> 把 [n_expert, n_tokens] 重排成 [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
//>> 每组只取分数最高的 2 个专家，求和当作"组分数"
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
//>> 在组维度上再做一次 top-k —— 这次 k 是 n_group_used
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
//>> 把没被选中的组整体填成 -INFINITY，等价于"永久落选"
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:9px">
        <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--d)">
          <div class="ct" style="color:var(--d)">hparams.n_expert_groups</div>
          <div class="cb">把 n_expert 个专家分成几组。<br>
          <span class="cm" style="margin:0">n_exp_per_group = n_expert / n_expert_groups</span><br>
          默认 0 -> 本段整段跳过。</div></div>
        <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--f)">
          <div class="ct" style="color:var(--f)">hparams.n_group_used</div>
          <div class="cb">每个 token 允许用几个<b>组</b>。<br>
          被选中组之外的概率被填成 <span class="cm" style="margin:0">-INFINITY</span>。</div></div>
      </div>
      <div class="flow" id="flow" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ops = ['ggml_reshape_3d', 'ggml_argsort_top_k(2)', 'ggml_get_rows', 'ggml_sum_rows',
                 'ggml_argsort_top_k(n_group_used)', 'ggml_set_rows', 'ggml_fill(-INF)',
                 'ggml_reshape_2d'];
    const flow = wrap.querySelector('#flow');
    const chips = ops.map((o, i) => {
      const c = U.chip(o, i === 1 || i === 4 ? 'd' : '');
      flow.appendChild(c);
      if (i < ops.length - 1) flow.appendChild(U.arrow('->'));
      return c;
    });
    chips.forEach(c => c.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '整段被 <span class="v">if (hparams.n_expert_groups &gt; 1 &amp;&amp; n_tokens &gt; 0)</span> 守着 —— 关掉时零开销。',
      '<span class="v">ggml_reshape_3d</span>：把概率按"组"重新排，得到 <span class="k">[n_exp_per_group, n_expert_groups, n_tokens]</span>。',
      '<span class="v">ggml_argsort_top_k(..., 2)</span>：每组取前 2 名专家，<br>再用 <span class="v">ggml_get_rows</span> + <span class="v">ggml_sum_rows</span> 求和，得到"组分数"。',
      '第二次 <span class="v">ggml_argsort_top_k</span>：在组维度上取前 <span class="k">n_group_used</span> 组。',
      '<span class="v">ggml_fill(..., -INFINITY)</span> + <span class="v">ggml_set_rows</span>：<br>' +
      '把落选组整体压成负无穷，于是第 5 幕的全局 top-k <b>不可能选到它们</b>。',
      '注意：<span class="k">分组路由没有引入任何新算子</span> —— 用的还是 reshape / argsort_top_k / get_rows / sum_rows / fill / set_rows。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    chips.forEach((c, i) => tl.at(2800 + i * 2100, () => {
      c.style.opacity = '1';
      msg.innerHTML = texts[Math.min(Math.floor(i / 2) + 1, 4)];
    }));
    tl.at(19600, () => { chips.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 ★ top-k 选择的真实函数名是 <span class="hl-b">ggml_argsort_top_k</span> */
{
  kicker: "L2-12 · 算子 4/7",
  title: "★ top-k 选择的真实函数名是 <span class=\"hl-b\">ggml_argsort_top_k</span>",
  sub: "源码里没有用 ggml_top_k —— 用的是\"先 argsort 再 view\"的那个版本。",
  caption: "`ggml_top_k` 确实存在于 ggml.h，但整个 `src/llama-graph.cpp` 里一次都没出现（grep 可复核）。",
  src: "src/llama-graph.cpp",
  mark: [1, 4, 6, 9, 17, 20, 22],
  lineNo: 2106,
  code: `    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
//>> selected_experts_in：允许调用方从外面传一份选中结果（某些模型复用别的路由）
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
//>> k = n_expert_used；输出 [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
//>> cb 挂在 src[0] 上：那个张量是 argsort 的完整结果，top-k 只是它的一个视图
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
//>> ggml_get_rows(probs, sel)：把"整行概率"收集成"选中专家的概率"
    cb(weights, "ffn_moe_weights", il);`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="row" style="gap:8px">
        <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">ggml_argsort_top_k(ctx, a, k)</div>
          <div class="cb">源码注释：<span class="cm" style="margin:0">similar to ggml_top_k but implemented as \`argsort\` + \`view\`</span><br>
          结果是一段<b>有序</b>索引。</div></div>
        <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--g)">
          <div class="ct" style="color:var(--g)">ggml_top_k(ctx, a, k)</div>
          <div class="cb">源码注释：<span class="cm" style="margin:0">the resulting top k indices are in no particular order</span><br>
          <b>本函数一次都没用它。</b></div></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['张量', '表达式', '形状'],
      [['selection_probs', '打分 / 门控 / 分组掩码之后的概率', '[n_expert, n_tokens]'],
       ['selected_experts', 'ggml_argsort_top_k(selection_probs, n_expert_used)', '[n_expert_used, n_tokens]'],
       ['probs (3d)', 'ggml_reshape_3d(probs, 1, n_expert, n_tokens)', '[1, n_expert, n_tokens]'],
       ['weights', 'ggml_get_rows(probs3, selected_experts)', '[1, n_expert_used, n_tokens]']],
      { monoCols: [0, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一步是全课最短、也最关键的一步：<span class="k">选中哪几个专家</span>。',
      '<span class="v">selection_probs</span> 是 [n_expert, n_tokens] —— 每列一个 token 的专家分数。',
      '<span class="v">ggml_argsort_top_k(selection_probs, n_expert_used)</span><br>' +
      '输出 <span class="k">[n_expert_used, n_tokens]</span> 的 <b>I32 专家编号表</b>。这张表就是"路由结果"。',
      '<span class="v">weights = ggml_get_rows(probs3, selected_experts)</span>：<br>' +
      '用编号表去"收集"概率，得到每个被选中专家的权重 [1, n_expert_used, n_tokens]。',
      '记住：<span class="v">selected_experts</span> 是下面所有专家计算的<b>索引参数</b> —— ' +
      '它同时也是 L2-13 里各家模型差异最大的地方（有的再乘个 scale、有的按组重映射）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2900 + i * 3100, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(15400, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 专家权重：<span class="hl-e">get_rows</span> → 归一化 → 缩放 */
{
  kicker: "L2-12 · 算子 5/7",
  title: "专家权重：<span class=\"hl-e\">get_rows</span> → 归一化 → 缩放",
  sub: "拿到 weights 之后有三段可选的加工，各由一个 bool / float 开关控制。",
  caption: "clamp 的下界 `6.103515625e-5`；源码注释逐字写着：`Avoid division by zero, clamp to smallest number representable by F16`。",
  src: "src/llama-graph.cpp",
  mark: [1, 3, 8, 10, 12, 16, 20, 25, 27],
  lineNo: 2126,
  code: `
    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
//>> norm_w：是否把 top-k 的权重归一化回和为 1
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
//>> clamp 下界 = F16 最小正规数，注释：Avoid division by zero
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
//>> w_scale：源码这里用的是 != 0.0f && != 1.0f —— 传 0 表示"不缩放"
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['分支', '触发条件', '算子', '形状变化'],
      [['SOFTMAX_WEIGHT', 'gating_op == SOFTMAX_WEIGHT', 'ggml_reshape_2d -> ggml_soft_max -> ggml_reshape_3d', '[1,k,T] 往返'],
       ['归一化 norm_w', 'norm_w == true', 'ggml_sum_rows -> ggml_clamp -> ggml_div', '[1,k,T] -> [k,T] -> [1,k,T]'],
       ['缩放 w_scale', 'w_scale != 0 && != 1', 'ggml_scale', '不变'],
       ['不缩放', '调用方传 0.0f', '（无）', '不变']],
      { monoCols: [2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '权重已经到手，但还不能直接用 —— 有三段可选加工。',
      '<span class="v">SOFTMAX_WEIGHT</span> 型：把 weights 临时压回 2 维做一次 <span class="k">ggml_soft_max</span>，再升回 3 维。',
      '<span class="v">norm_w = true</span>：<br><span class="v">ggml_sum_rows</span> 求 k 个权重之和 -> ' +
      '<span class="v">ggml_clamp</span> 防零 -> <span class="v">ggml_div</span> 除回去。',
      '<span class="v">w_scale</span>：一个 <span class="k">ggml_scale</span> 乘上常数（DeepSeek 系的 routed_scaling_factor 就走这里）。',
      '注意这三段都<b>不是必须的</b>：<span class="k">dbrx.cpp 传 norm_w=true、w_scale=expert_weights_scale</span>，' +
      '而 <span class="k">bert.cpp 传 false / 1.0f</span>。同一个原语，两种模型。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 3200, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 ★ 子图：<span class="hl-a">top-k 路由</span> + <span class="hl-b">专家并行</span> */
{
  kicker: "L2-12 · 算子 6/7",
  title: "★ 子图：<span class=\"hl-a\">top-k 路由</span> + <span class=\"hl-b\">专家并行</span>",
  sub: "把这一课要能默画出来的子图画出来 —— 每个名字都是真实存在的 ggml 算子。",
  caption: "`build_lora_mm_id` 内部就是 `ggml_mul_mat_id`（src/llama-graph.cpp:1550），它是\"按专家索引做矩阵乘\"的唯一入口。",
  src: "src/llama-graph.cpp",
  mark: [3, 6, 20, 23, 27, 39, 41],
  lineNo: 2166,
  code: `    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
//>> 两条路：gate_up 融合张量（一个 mul_mat_id）或 up / gate 分开（两个 mul_mat_id）
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
//>> build_lora_mm_id -> ggml_mul_mat_id(ctx0, w, cur, ids)
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
//>> 融合路里，gate 与 up 是同一块结果的左右两个 ggml_view_3d
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
//>> 分开路：gate_exps 为 nullptr 时就退化成一元激活（bert.cpp 正是这样）
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="cm" style="margin:0">① 路由（每个 token 选 k 个专家）—— 第 2~5 幕</div>
      <div class="flow" id="r1" style="gap:5px"></div>
      <div class="cm" style="margin:2px 0 0">② 专家并行（k 条支路，各乘自己的权重）—— 本幕</div>
      <div class="flow" id="r2" style="gap:5px"></div>
      <div class="flow" id="r3" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    function band(host, names, colors) {
      const out = [];
      names.forEach((n, i) => {
        const c = U.chip(n, colors[i] || '');
        host.appendChild(c);
        out.push(c);
        if (i < names.length - 1) host.appendChild(U.arrow('->'));
      });
      return out;
    }
    const r1 = band(wrap.querySelector('#r1'),
      ['ffn_gate_inp [E,n_embd]', 'ggml_mul_mat', 'ggml_soft_max', 'ggml_argsort_top_k', 'ggml_get_rows'],
      ['', 'a', 'a', 'b', 'b']);
    const r2 = band(wrap.querySelector('#r2'),
      ['ggml_reshape_3d', 'ggml_mul_mat_id (up_exps)', 'ggml_mul_mat_id (gate_exps)', 'ggml_swiglu_split'],
      ['', 'b', 'b', 'c']);
    const r3 = band(wrap.querySelector('#r3'),
      ['ggml_mul_mat_id (down_exps)', 'ggml_mul(weights)', 'ggml_view_2d x k', 'ggml_add x (k-1)'],
      ['b', 'e', 'e', 'e']);
    [r1, r2, r3].forEach(g => g.forEach(c => c.style.opacity = '.28'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '路由子图：打分 -> 概率 -> top-k 索引 -> 用索引收集权重。<br>' +
      '产出两张表：<span class="k">selected_experts</span>（选谁）和 <span class="v">weights</span>（各占多少）。',
      '① <span class="v">ggml_argsort_top_k</span> 输出的 [k, n_tokens] 索引，是下面专家支路的<b>全部路由信息</b>。',
      '② <span class="v">ggml_reshape_3d</span> 把输入摊成 <span class="k">[n_embd, 1, n_tokens]</span>，' +
      '再交给 <span class="v">ggml_mul_mat_id</span> —— 后者的第三个参数就是 selected_experts。',
      '关键：<span class="k">k 条支路不是 for 循环，而是同一个张量的第 1 维</span>。<br>' +
      'up / gate / down 的结果都带 <span class="v">n_expert_used</span> 这一维。',
      '③ 专家内 FFN 与稠密 FFN 完全同构：<span class="v">up</span> / <span class="v">gate</span> / 激活 / <span class="v">down</span>，' +
      '只是每次乘都多一个"按索引选权重"的动作。',
      '④ 最后 <span class="v">ggml_mul(experts, weights)</span> 加权，' +
      '<span class="v">ggml_view_2d</span> 沿 k 维切 k 份，<span class="v">ggml_add</span> 累加 —— ' +
      '<b>一次 add 就是一路专家并行</b>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3400, () => { r1.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[1]; });
    tl.at(7400, () => { r2.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[2]; });
    tl.at(11400, () => { msg.innerHTML = texts[3]; });
    tl.at(15400, () => { r3.slice(0, 2).forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[4]; });
    tl.at(19600, () => { r3.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 激活、down 投影、以及<span class="hl-c">用 add 做的并行归并</span> */
{
  kicker: "L2-12 · 算子 7/7",
  title: "激活、down 投影、以及<span class=\"hl-c\">用 add 做的并行归并</span>",
  sub: "n_expert_used 这一维在最后被 view + add 展平回 [n_embd, n_tokens]。",
  caption: "`n_expert_used_il == 1` 时要做一次 `ggml_cont` —— 源码注释：avoid returning a non-contiguous tensor。",
  src: "src/llama-graph.cpp",
  mark: [0, 18, 33, 35, 36, 43, 46, 54],
  lineNo: 2304,
  code: `    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
//>> down 投影：同样是 ggml_mul_mat_id，索引同一张 selected_experts
    if (arch == LLM_ARCH_MISTRAL4) {
        // src1 can exceed F16 range
        ggml_prec_set_src(experts, GGML_PREC_F32, 1);
    }
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
//>> ★ 加权：k 条支路各乘自己的权重 —— 这就是"加权求和"里的"加权"
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    // Use per-layer n_expert_used to bound the graph even during warmup (avoids
    // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
    // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
//>> 按【每层】的 n_expert_used 上界建图，避免热身后图规模突变（注释点明了原因）
    for (uint32_t i = 0; i < n_expert_used_il; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
//>> ★ 沿第 1 维（专家维）切出第 i 路的 [n_embd, n_tokens] 视图

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < n_expert_used_il; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
//>> ★ k-1 次 ggml_add —— 这就是专家并行的"求和"

        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used_il == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" id="flow" style="gap:5px"></div>
      <div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const acts = ['ggml_swiglu_split', 'ggml_silu', 'ggml_geglu_split', 'ggml_reglu_split',
                  'ggml_swiglu_oai', 'ggml_swiglu_clamp'];
    const flow = wrap.querySelector('#flow');
    const chips = acts.map((a, i) => {
      const c = U.chip(a, 'c');
      flow.appendChild(c);
      if (i < acts.length - 1) flow.appendChild(U.arrow('/'));
      return c;
    });

    const t = U.table(
      ['阶段', '算子', '形状'],
      [['up / gate', 'ggml_mul_mat_id', '[n_ff, k, n_tokens]'],
       ['激活', 'ggml_swiglu_split（或 silu / geglu / reglu）', '[n_ff, k, n_tokens]'],
       ['down', 'ggml_mul_mat_id', '[n_embd, k, n_tokens]'],
       ['加权', 'ggml_mul(experts, weights)', '[n_embd, k, n_tokens]'],
       ['切分', 'ggml_view_2d x k', '[n_embd, n_tokens] x k'],
       ['归并', 'ggml_add x (k-1)', '[n_embd, n_tokens]']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '激活函数是 switch 出来的，所有分支都是已存在的 ggml 算子：',
      '<span class="v">ggml_swiglu_split</span> 是最常见的一支（有 gate 时）。' +
      '<span class="v">ggml_silu</span> 是无 gate 的退化情形 —— <span class="k">bert.cpp 就是这一支</span>。',
      '<span class="v">ggml_mul_mat_id(down_exps, ...)</span> 把 [n_ff, k, T] 投回 [n_embd, k, T]，' +
      '索引还是同一张 <span class="v">selected_experts</span>。',
      '<span class="v">ggml_mul(experts, weights)</span>：<span class="k">k 路各乘自己的权重</span>。',
      '<span class="v">ggml_view_2d</span> 沿专家维切 k 份，<span class="v">ggml_add</span> 累加 k-1 次 —— ' +
      '<b>稀疏的 k 路在这里被压回一个稠密张量</b>，层与层之间看不出 MoE。',
      '于是这一层的输出和稠密 FFN 一模一样：<span class="v">[n_embd, n_tokens]</span>，' +
      '直接加回残差。MoE 的"稀疏"从头到尾只活在<span class="k">图内部</span>。'
    ];
    rows.forEach((r, i) => tl.at(600 + i * 3400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(21200, () => {
      rows.forEach(x => { x.className = ''; });
      chips.forEach(c => c.style.opacity = '1');
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 9 24 个文件里，有 <span class="hl-g">7 个需要点名</span> */
{
  kicker: "L2-12 · 如实报告",
  title: "24 个文件里，有 <span class=\"hl-g\">7 个需要点名</span>",
  sub: "分组依据是\"图构建代码命中了 MoE 图原语\"。命中了不等于\"这个文件就是 MoE\"。",
  caption: "下面这张表是逐个读出来的结论；每一行的原文引用见 source.md 第四节。",
  src: "src/models/deci.cpp",
  mark: [0, 5, 10],
  lineNo: 157,
  code: `        if (model.layers[il].ffn_gate_inp == nullptr) {
//>> deci.cpp 里 ffn_gate_inp 只出现在这个空指针判断里 —— 从来没有被创建过
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
//>> 走的是稠密 build_ffn：up / gate / down 三个普通权重
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
//>> 整个 deci.cpp 里 grep "moe|expert" 零命中（这是 Nemotron 稠密模型）
            cb(cur, "ffn_out", il);
        }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['文件', '文件里实际有什么', '判定'],
      [['src/models/deci.cpp', '只有一处 ffn_gate_inp == nullptr 判断；grep moe/expert 零命中', '完全不是 MoE'],
       ['src/models/deepseek2ocr.cpp', '只 create_tensor 专家张量，本文件不构图（80 行）', '图里没有 MoE'],
       ['src/models/ernie4-5.cpp', '只 create_tensor 专家张量；本文件的图是稠密分支', '图里没有 MoE'],
       ['src/models/granite-moe.cpp', '只 create_tensor 专家张量（84 行），图类不在本文件', '图里没有 MoE'],
       ['src/models/lfm2moe.cpp', '只 create_tensor 专家张量（85 行），图类不在本文件', '图里没有 MoE'],
       ['src/models/bert.cpp', '调 build_moe_ffn，但 gate_exps 传 nullptr', 'MoE，但无 gate 专家'],
       ['src/models/ernie4-5-moe.cpp', '只调 build_moe_ffn；张量由 ernie4-5.cpp 创建', '反向拆分']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '24 个文件里 <span class="k">17 个</span>是"专家张量与构图都在本文件"的真 MoE，<br>' +
      '另外 <span class="k">7 个</span>要单独点名（见上表）。',
      '<span class="v">deci.cpp</span> 最彻底：它是 Nemotron 稠密模型，<br>' +
      '文件里唯一的 MoE 痕迹是一个"专家张量为空就走稠密"的分支判断 —— 而且没有 else。',
      '<span class="v">deepseek2ocr.cpp</span> / <span class="v">ernie4-5.cpp</span> / ' +
      '<span class="v">granite-moe.cpp</span> / <span class="v">lfm2moe.cpp</span>：<br>' +
      '只负责 <span class="k">create_tensor</span>，图类定义在别的 TU 里。',
      '<span class="v">bert.cpp</span> 是 MoE（<span class="m">moe_every_n_layers</span>），' +
      '但没有 gate 专家张量：<br>build_moe_ffn 的 gate_exps 传 nullptr，激活退化成 <span class="k">ggml_silu</span>。',
      '还有反向的一例：<span class="v">ernie4-5-moe.cpp</span> 只负责构图，专家张量由 ' +
      '<span class="v">ernie4-5.cpp</span> 创建。<br>' +
      '所以"一个文件 = 一个模型"这个直觉在 <span class="v">src/models/</span> 里不成立。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => { rows.forEach((x, k) => { x.className = (k === 0) ? 'on' : ''; }); msg.innerHTML = texts[1]; });
    tl.at(7600, () => { rows.forEach((x, k) => { x.className = (k >= 1 && k <= 4) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
    tl.at(11800, () => { rows.forEach((x, k) => { x.className = (k === 5) ? 'on' : ''; }); msg.innerHTML = texts[3]; });
    tl.at(16200, () => { rows.forEach((x, k) => { x.className = (k === 6) ? 'on' : ''; }); msg.innerHTML = texts[4]; });
    tl.at(20200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 压成一张表 + 一道练习 */
{
  kicker: "L2-12 · 收束",
  title: "压成一张表 + 一道练习",
  sub: "会画这张子图，L2-13 里所有 MoE 变体都只是它的参数组合。",
  caption: "下一课 L2-13（MoE 下篇）逐个模型看：共享专家、MTP 块、chunk expert、gate_up 融合。",
  src: "src/models/dbrx.cpp",
  mark: [0, 1, 3, 5, 6, 8, 10, 12],
  lineNo: 115,
  code: `        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
//>> ffn_gate_inp：路由打分矩阵
                model.layers[il].ffn_up_exps,
//>> ffn_up_exps / ffn_gate_exps / ffn_down_exps：三组专家权重（各多一维）
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
//>> n_expert 与 n_expert_used 是这一课的两个核心整数
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
//>> LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX —— 第 3 幕的枚举
                il);`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['阶段', '真实算子', 'llama-graph.cpp 行号'],
      [['打分', 'build_lora_mm -> ggml_mul_mat', '2025'],
       ['门控', 'ggml_soft_max / ggml_sigmoid / ggml_sqrt(ggml_softplus)', '2043 / 2047 / 2055'],
       ['选择偏置', 'ggml_add(probs, exp_probs_b)', '2066'],
       ['分组路由', 'ggml_argsort_top_k x2 + ggml_fill + ggml_set_rows', '2089 / 2096 / 2101'],
       ['top-k', 'ggml_argsort_top_k(selection_probs, n_expert_used)', '2109'],
       ['收集权重', 'ggml_get_rows(probs3, selected_experts)', '2123'],
       ['归一化', 'ggml_sum_rows + ggml_clamp + ggml_div', '2137 / 2141 / 2144'],
       ['专家 up/gate', 'ggml_mul_mat_id（经 build_lora_mm_id）', '2190 / 2203'],
       ['激活', 'ggml_swiglu_split（或 ggml_silu）', '2246 / 2249'],
       ['专家 down', 'ggml_mul_mat_id', '2304'],
       ['加权', 'ggml_mul(experts, weights)', '2321'],
       ['归并', 'ggml_view_2d x k + ggml_add x (k-1)', '2337 / 2346']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '某 MoE 层：<span class="mono">n_embd=512</span>、<span class="mono">n_expert=64</span>、' +
      '<span class="mono">n_expert_used=6</span>、<span class="mono">n_ff_exp=1408</span>、' +
      '<span class="mono">n_tokens=8</span>，且 <span class="mono">gate_up_exps == nullptr</span>。<br>' +
      '（1）写出 <span class="mono">selected_experts</span> / <span class="mono">weights</span> 的形状；<br>' +
      '（2）专家支路里哪一个算子替代了稠密 FFN 的 <span class="mono">ggml_mul_mat</span>，它的第三个参数是什么？<br>' +
      '（3）最后归并用了哪两个算子，各几次？',
      '（1）<span class="mono">ggml_argsort_top_k(selection_probs, n_expert_used)</span> 输出 ' +
      '<span class="mono">[n_expert_used, n_tokens] = [6, 8]</span>；<br>' +
      'probs 先 <span class="mono">ggml_reshape_3d(probs, 1, n_expert, n_tokens)</span> 成 ' +
      '<span class="mono">[1, 64, 8]</span>，再 <span class="mono">ggml_get_rows</span> 得 ' +
      '<span class="mono">weights = [1, 6, 8]</span>（源码注释逐字写着这两个形状）。<br>' +
      '（2）是 <span class="mono">ggml_mul_mat_id</span>（经 <span class="mono">build_lora_mm_id</span>，' +
      'src/llama-graph.cpp:1550）；第三个参数是 <span class="mono">selected_experts</span>，即那张 [6, 8] 的索引表。<br>' +
      '（3）<span class="mono">ggml_view_2d</span> 切 <span class="mono">k = 6</span> 次，' +
      '<span class="mono">ggml_add</span> 累加 <span class="mono">k-1 = 5</span> 次；' +
      '若该层 <span class="mono">n_expert_used == 1</span> 还要补一次 <span class="mono">ggml_cont</span>。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '12 个阶段，全部是 L1-02 里已有 op 的<b>重新组合</b>。',
      '前四行都属于"路由"：<span class="k">决定谁能参与</span>。',
      '中间四行是"专家并行"：<span class="k">k 路同时算，索引同一张表</span>。',
      '最后四行是"归并"：<span class="k">k 路压回一个稠密张量</span>，加回残差。',
      '一句话收束：<span class="v">稀疏只体现在专家张量多出来的那一维</span>；' +
      '图上多出来的只有"一次 top-k 选择"和"一次按索引的收集"。<br>' +
      '所以 24 个 MoE 模型的图代码才能长得那么像。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2500 + i * 1250, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      if (i === 3) msg.innerHTML = texts[1];
      if (i === 7) msg.innerHTML = texts[2];
      if (i === 11) msg.innerHTML = texts[3];
    }));
    tl.at(19600, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[4];
    });
  }
},

];
