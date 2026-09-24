/* ==========================================================================
   L2-13 · 模型家族（四）：稀疏专家 MoE（下）
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 24 个 MoE 文件，共用 <span class="hl-a">一条骨架</span> */
{
  kicker: "L2-13 · 全局",
  title: "24 个 MoE 文件，共用 <span class=\"hl-a\">一条骨架</span>",
  sub: "L2-12 讲这条骨架的基础序列；本课讲它的参数如何被 24 个模型文件选成不同的路由策略。",
  caption: "回顾 L2-06：每个模型图都是 llm_graph_context 的子类，MoE 只是其中一个 build_* 调用。",
  src: "src/llama-graph.cpp",
  mark: [0, 10, 11, 12, 16, 18, 24],
  lineNo: 1993,
  code: `ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="faces" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '共享专家 shexp', b: '9 / 24 个文件声明了<br>ffn_*_shexp 张量', m: 'ffn_gate_shexp' },
      { c: 'c', t: '分组路由', b: '24 个文件里 0 处提到<br>n_expert_groups / n_group_used', m: 'hparams.n_expert_groups' },
      { c: 'e', t: '与注意力变体的组合', b: '同一个 MoE 分支，喂进来的 cur<br>来自 SWA / MSA / MTP 等不同的图', m: 'build_attn(...)' }
    ];
    const host = wrap.querySelector('#faces');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'L2-12 讲的是这条骨架的<b>基础序列</b>：logits -> top-k -> mul_mat_id -> 加权求和。',
      '本课只问一件事：<span class="k">同样的骨架，24 个模型文件各自选了哪些开关</span>。',
      '<span class="v">共享专家</span>：9 个文件声明了 shexp 张量，接线方式有三种。',
      '<span class="v">分组路由</span>：开关不在参数表里，在 <span class="v">hparams</span> 里。',
      '<span class="v">注意力变体</span>：MoE 分支的上游各不相同，但 build_moe_ffn 不关心。',
      '对照 L2-06：所有模型图都是 <span class="v">llm_graph_context</span> 的子类，MoE 只是其中一个 build_* 调用。'
    ];
    defs.forEach((_, i) => tl.at(600 + i * 2600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(12400, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[0] + '<br>' + texts[5];
    });
  }
},

/* ------------------------------------------------------ 2 ★ 控制"专家选择"的是这 <span class="hl-a">8 个参数</span> */
{
  kicker: "L2-13 · 核心",
  title: "★ 控制\"专家选择\"的是这 <span class=\"hl-a\">8 个参数</span>",
  sub: "其余参数只改权重布局与专家内部激活，不改变\"选哪些专家\"。",
  caption: "验收点：能说出 build_moe_ffn 的参数里哪些控制专家选择。",
  src: "src/llama-graph.cpp",
  mark: [0, 1, 2, 4, 5, 6, 8, 14],
  lineNo: 2003,
  code: `         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="flow" id="notroute" style="gap:6px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['参数', '函数体里生效于', '它决定什么'],
      [['n_expert', 'llama-graph.cpp:2084 / 2120', '候选专家总数：分组大小、probs 的维度'],
       ['n_expert_used', 'llama-graph.cpp:2109', 'top-k 的 k —— 直接决定选几个专家'],
       ['gating_op', 'llama-graph.cpp:2040-2059', 'logits 怎么变概率：softmax / sigmoid / softmax_weight / sqrt_softplus'],
       ['exp_probs_b', 'llama-graph.cpp:2066', '选择偏置：只加到 selection_probs，不加到权重'],
       ['norm_w', 'llama-graph.cpp:2134-2148', 'top-k 权重是否除以它们的和'],
       ['w_scale', 'llama-graph.cpp:2149-2151', '选中之后的路由权重整体缩放'],
       ['probs_in', 'llama-graph.cpp:2024-2032', '非空则跳过 gate_inp，用外部算好的 logits'],
       ['selected_experts_in', 'llama-graph.cpp:2107-2111', '非空则整段跳过 top-k']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const nr = wrap.querySelector('#notroute');
    nr.appendChild(U.chip('不控制选择：', 'g'));
    ['weight 类：gate_inp', 'up_exps', 'gate_exps', 'down_exps'].forEach(x => nr.appendChild(U.chip(x)));
    nr.appendChild(U.chip('type_op -> :2221', 'd'));
    nr.appendChild(U.chip('il -> cb 命名', 'd'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '8 个参数里，前两个是<b>基数与 k</b>，中间四个是<b>概率与权重</b>，最后两个是<b>外部接管通道</b>。',
      '<span class="v">n_expert</span> 与 <span class="v">n_expert_used</span>：一个定候选集合大小，一个定 top-k 的 k。',
      '<span class="v">gating_op</span>：四个分支在源码里写死（2040-2059），传错值会 GGML_ABORT。',
      '<span class="v">exp_probs_b</span> 只影响"选谁"，<span class="k">不影响选中之后的权重</span>（2063 的注释写明）。',
      '<span class="v">norm_w</span>：把 top-k 权重除以它们的和，除以零被 clamp 成 F16 最小值（2141）。',
      '<span class="v">w_scale</span>：<span class="k">只在它不等于 0.0f 且不等于 1.0f 时才生效</span>（2149）。',
      '<span class="v">probs_in</span> / <span class="v">selected_experts_in</span>：两条外部接管通道，可以分别绕开 gate_inp 与 top-k。',
      '结论：<span class="k">MoE 的可调旋钮不在"算子"，在"路由策略"</span> —— 这 8 个参数就是全部。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 8)];
    }));
    tl.at(22800, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[8];
    });
  }
},

/* ------------------------------------------------------ 3 ★ 分组路由的开关是 <span class="hl-a">hparams</span>，不是调用参数 */
{
  kicker: "L2-13 · 核心",
  title: "★ 分组路由的开关是 <span class=\"hl-a\">hparams</span>，不是调用参数",
  sub: "24 个文件里 n_expert_groups / n_group_used 出现 0 次；它从 GGUF 读进 hparams，在函数体里生效。",
  caption: "回顾 L2-12：deepseek2 走的就是这一段，模型文件同样一个字都没写。",
  src: "src/llama-graph.cpp",
  mark: [0, 2, 4, 7, 16, 22],
  lineNo: 2081,
  code: `    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
//>> hparams 是成员引用，不是参数 —— 模型文件无法在这一行做选择
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
//>> 组级 top-k：n_group_used 决定允许几个组
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
//>> 没被选中的组整片填成 -INFINITY，它们在下一步的 top-k 里必然落选
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">GGUF 元数据</span><span class="arrow">-></span>
        <span class="chip b">hparams.n_expert_groups</span><span class="arrow">-></span>
        <span class="chip c">大于 1 ?</span><span class="arrow">-></span>
        <span class="chip e">mask 掉没选中的组</span>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['hparams 字段', '读进 hparams 的地方', '在 build_moe_ffn 里怎么用'],
      [['n_expert_groups', 'llama-model.cpp:1266', 'llama-graph.cpp:2083 判断 / 2087 分组'],
       ['n_group_used', 'llama-model.cpp:1267', 'llama-graph.cpp:2096 组级 top-k'],
       ['n_group_experts', '本课 24 文件未出现', 'llama-graph.cpp:2117 GROVEMOE 专用重映射']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '分组路由（DeepSeek V3 的 device-limited routing）<span class="k">不是一个参数</span>。',
      '它从 GGUF 读进 <span class="v">hparams.n_expert_groups</span>，在函数体里用 <span class="v">if (hparams.n_expert_groups &gt; 1)</span> 判断。',
      '先按组内最高两个专家分求和（2089-2093），再对组做 top-<span class="v">n_group_used</span>。',
      '最后把没选中的组填成 <span class="v">-INFINITY</span>，后面那次 <span class="v">argsort_top_k</span> 自然选不到它们。',
      '本课 24 个文件里 <span class="v">n_expert_groups</span> / <span class="v">n_group_used</span> 出现 <span class="k">0 次</span>：<br>它们是模型（GGUF）的属性，不是代码里的选择。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 2]); });
    tl.at(3200, () => { msg.innerHTML = texts[1]; });
    tl.at(6000, () => { msg.innerHTML = texts[2]; U.markLines(document, [4, 7, 16, 22]); });
    tl.at(9200, () => { msg.innerHTML = texts[3]; });
    tl.at(11800, () => { msg.innerHTML = texts[4]; U.markLines(document, []); });
  }
},

/* ------------------------------------------------------ 4 ★ 共享专家不在参数表里 —— 它是 <span class="hl-a">调用点外的一条支路</span> */
{
  kicker: "L2-13 · 核心",
  title: "★ 共享专家不在参数表里 —— 它是 <span class=\"hl-a\">调用点外的一条支路</span>",
  sub: "qwen2moe 给它配了独立的门控；llama4 / step35 / minimax-m3 直接相加；llama / mistral3 / refact 声明了但没接。",
  caption: "24 个文件里只有 qwen2moe 声明 LLM_TENSOR_FFN_GATE_INP_SHEXP。",
  src: "src/models/qwen2moe.cpp",
  mark: [0, 2, 6, 9, 17, 20, 23],
  lineNo: 145,
  code: `        // FFN shared expert
        {
            ggml_tensor * cur_gate_inp = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

            // sigmoid
            ggml_tensor * cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
            cb(cur_gate, "ffn_shexp_gate", il);

            ggml_tensor * cur_ffn = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur_ffn, "ffn_shexp", il);

            ggml_tensor * ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
            cb(ffn_shexp_out, "ffn_shexp_out", il);

            moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
            cb(moe_out, "ffn_out", il);

            cur = moe_out;`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="faces" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '自带门控', b: 'qwen2moe:147-165<br>ffn_gate_inp_shexp 出一个标量门，<br>乘在共享专家输出上再相加', m: 'ggml_mul(cur_ffn, cur_gate)' },
      { c: 'b', t: '直接相加', b: 'llama4:232-240 · step35:327-335<br>minimax-m3:575-581<br>共享专家是一个普通 build_ffn', m: 'ggml_add(moe_out, sh_out)' },
      { c: 'd', t: '只声明不接线', b: 'llama:85-88 · minicpm:77-81<br>mistral3:80-83 · refact:66-70<br>张量建了，图里没有对应支路', m: 'ffn_gate_shexp = create_tensor(...)' }
    ];
    const host = wrap.querySelector('#faces');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="k">9 / 24</span> 个文件声明了 ffn_*_shexp 张量，但接法有三种。',
      'qwen2moe：<span class="v">ffn_gate_inp_shexp</span> 是一个 <span class="v">[n_embd]</span> 的向量（54 行），<br>算完 sigmoid 门后用 <span class="v">ggml_mul</span> 乘到共享专家输出上（162）。',
      'llama4 / step35 / minimax-m3：共享专家就是一次 <span class="v">build_ffn</span>，<br>结果与 <span class="v">moe_out</span> 直接 <span class="v">ggml_add</span>，没有门。',
      'llama / minicpm / mistral3 / refact：<span class="v">create_tensor</span> 建了张量，<br>但在本文件的图里找不到使用它的支路。',
      '关键：这三种接法<span class="k">一种都没有经过 build_moe_ffn 的参数</span> —— <br>共享专家是图上的<b>第二条支路</b>，不是 MoE 的一个开关。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(3400 + i * 4600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(17800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 5 同一个函数，<span class="hl-c">21 个调用点</span>各选不同的开关 */
{
  kicker: "L2-13 · 24 个调用点",
  title: "同一个函数，<span class=\"hl-c\">21 个调用点</span>各选不同的开关",
  sub: "20 个文件共 21 处调用；下表每一格都是对这 24 个文件静态统计出来的。",
  caption: "L2-12 覆盖的另外 12 个 MoE 文件走的是同一张参数表。",
  src: "src/models/step35.cpp",
  mark: [0, 5, 7, 8, 9],
  lineNo: 512,
  code: `        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['参数', '21 个调用点的取值分布', '代表调用点'],
      [['n_expert / n_expert_used', '21 处写法完全一样', 'llama-graph.cpp:1471-1472 提供'],
       ['gating_op', 'SOFTMAX 12 / 读 hparams 5 / SIGMOID 3 / SOFTMAX_WEIGHT 1', 'llama4:228 · openai-moe:141'],
       ['exp_probs_b', '传张量 7 / nullptr 14', 'mimo2:213 · step35:517'],
       ['norm_w', 'true 12 / false 5 / 读 hparams 4', 'llama:209 · step35:519'],
       ['w_scale', 'hparams.expert_weights_scale 20 / 字面量 1.0f 1', 'maple:129'],
       ['probs_in', '只有 2 处传（其余用默认 nullptr）', 'smallthinker:159 · nemotron-h-moe:127']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '统计口径：对 24 个文件里所有 <span class="v">build_moe_ffn(</span> 调用点逐个取参。<br>20 个文件共 <span class="k">21 处</span>（step35 主干与 MTP 各一处）。',
      '<span class="v">n_expert</span> / <span class="v">n_expert_used</span> 21 处一字不差 —— 它们是 <span class="v">llm_graph_context</span> 的成员，<br>来自 hparams；<span class="k">模型文件在这两个参数上没有自由度</span>。',
      '<span class="v">gating_op</span> 是分歧最大的一个：12 处写死 SOFTMAX，5 处转成 <span class="v">hparams.expert_gating_func</span>。',
      '<span class="v">norm_w</span> 是三态：写死 true、写死 false、或者读 <span class="v">hparams.expert_weights_norm</span>。',
      '<span class="v">probs_in</span> 只有 smallthinker 与 nemotron-h-moe 用到 —— 这两处把 top-k 的输入搬到了函数外。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 3000, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(18300, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[1]; });
  }
},

/* ------------------------------------------------------ 6 <span class="hl-e">probs_in</span>：路由已经在外面算好了 */
{
  kicker: "L2-13 · 两条接管通道",
  title: "<span class=\"hl-e\">probs_in</span>：路由已经在外面算好了",
  sub: "传了 probs_in，函数体就不再碰 gate_inp；smallthinker 连 gate_inp 都直接传 nullptr。",
  caption: "签名最后一位 selected_experts_in 能整段跳过 top-k；本课 24 个文件都没用它。",
  src: "src/models/nemotron-h-moe.cpp",
  mark: [1, 4, 16, 17, 20, 22, 24, 26, 28, 29],
  lineNo: 99,
  code: `    {
        ggml_tensor * router_logits = build_lora_mm(layer.ffn_gate_inp, cur);
        cb(router_logits, "mtp_ffn_moe_logits", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   NULL, layer.ffn_up_shexp_s,
                NULL,                 NULL, NULL,
                layer.ffn_down_shexp, NULL, layer.ffn_down_shexp_s,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        if (layer.ffn_latent_down) {
            cur = ggml_mul_mat(ctx0, layer.ffn_latent_down, cur);
        }

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                nullptr, // no gate
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_RELU_SQR, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                il,
                router_logits, nullptr,
                layer.ffn_up_exps_s,
                nullptr, // no gate
                layer.ffn_down_exps_s);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row center" style="gap:8px">
        <div class="card" style="width:214px;border-left-color:var(--a)">
          <div class="ct" style="color:var(--a)">MTP 层：外面先算</div>
          <div class="cb">nemotron-h-moe:100<br><span class="cm" style="margin:0">build_lora_mm(ffn_gate_inp, cur)</span><br>
          得到 router_logits，作为第 13 个参数传进去</div></div>
        <div class="card" style="width:214px;border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">函数体：短路</div>
          <div class="cb">llama-graph.cpp:2024<br><span class="cm" style="margin:0">if (probs_in == nullptr)</span><br>
          非空就直接 <span class="cm" style="margin:0">logits = probs_in</span>，gate_inp 根本不用</div></div>
        <div class="card" style="width:214px;border-left-color:var(--d)">
          <div class="ct" style="color:var(--d)">特例：smallthinker</div>
          <div class="cb">smallthinker:150 把 gate_inp 传成 <span class="cm" style="margin:0">nullptr</span>，<br>
          第 14 个参数传 <span class="cm" style="margin:0">probs</span>（109 行算好）</div></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    const texts = [
      '两条通道的意义：<span class="k">路由这件事可以完全搬到 build_moe_ffn 之外</span>。',
      'nemotron-h-moe 的 MTP 草稿头在 100 行自己算 <span class="v">router_logits</span>，<br>MoE 调用在 115 行才发生 —— 中间隔着共享专家的 build_ffn（103-109）。',
      '函数体的判据在 <span class="v">llama-graph.cpp:2024</span>：<span class="v">probs_in</span> 非空就不调用 <span class="v">build_lora_mm(gate_inp, cur)</span>；<br>115 行的调用把 <span class="v">gate_inp</span> 与 <span class="v">probs_in</span> 一起传进去，函数只认后者。',
      '<span class="v">selected_experts_in</span> 是同一思路的极端版：非空则连 <span class="v">argsort_top_k</span> 都跳过（2107-2111）。',
      '本课 24 个文件里，<span class="v">probs_in</span> 出现 2 次，<span class="v">selected_experts_in</span> <span class="k">0 次</span>。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [1]); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [1, 4]); });
    tl.at(7600, () => { msg.innerHTML = texts[2]; U.markLines(document, [16, 17]); });
    tl.at(11600, () => { msg.innerHTML = texts[3]; });
    tl.at(15600, () => { msg.innerHTML = texts[4]; U.markLines(document, [20, 22, 28, 29]); });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-a">借图</span>：3 个文件自己没有 MoE 图 */
{
  kicker: "L2-13 · 发现",
  title: "<span class=\"hl-a\">借图</span>：3 个文件自己没有 MoE 图",
  sub: "phimoe.cpp（55 行）没有一行 build_moe_ffn —— 它的 MoE 行为由 phi3.cpp:153 决定。",
  caption: "同类的还有 nomic-bert-moe（借 bert）、minicpm（借 granite）。",
  src: "src/models/models.h",
  mark: [0, 2, 3, 5, 6],
  lineNo: 657,
  code: `struct llama_model_phimoe : public llama_model_base {
    llama_model_phimoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    using graph = llama_model_phi3::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['本课文件', '在 models.h 里怎么声明图', 'MoE 行为实际由谁决定'],
      [['src/models/phimoe.cpp', 'models.h:663  using graph = llama_model_phi3::graph<iswa>', 'phi3.cpp:153 的 build_moe_ffn'],
       ['src/models/nomic-bert-moe.cpp', 'models.h:341  using graph = llama_model_bert::graph', 'bert.cpp 的图（本课不引用）'],
       ['src/models/minicpm.cpp', 'models.h:1739 using graph = llama_model_granite::graph', 'granite.cpp 的图（本课不引用）']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '24 个文件里 <span class="k">21 个</span>在 models.h 里声明了自己的 <span class="v">struct graph</span>；<br>剩下 <span class="k">3 个</span>只是把别人的图 <span class="v">using</span> 过来。',
      'phimoe.cpp 全文 55 行：hparams、张量声明、选图，仅此而已。',
      '<span class="k">文件行数少不等于模型简单</span>：PhiMoE 的路由图完全长在 phi3.cpp 里，<br>靠 <span class="v">ffn_gate_inp == nullptr</span> 分流（phi3.cpp:143）。',
      '这也是静态扫描会把它分到"MoE 文件"的原因 —— 它命中了 <span class="v">ffn_gate_inp</span>，<br>但命中的是<b>张量名</b>，不是图原语。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    tl.at(4000, () => { msg.innerHTML = texts[1]; });
    tl.at(7600, () => { msg.innerHTML = texts[2]; });
    tl.at(11800, () => { msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 8 MoE 分支的<span class="hl-c">入口条件</span>也由模型文件说了算 */
{
  kicker: "L2-13 · 组合",
  title: "MoE 分支的<span class=\"hl-c\">入口条件</span>也由模型文件说了算",
  sub: "phi3.cpp 的图里稠密与 MoE 共用一条主干，靠 ffn_gate_inp == nullptr 分流；PhiMoE 借的就是这张图。",
  caption: "回顾 L2-15：SWA / GQA / MLA 等注意力变体的完整谱系在那一课。",
  src: "src/models/phi3.cpp",
  mark: [0, 1, 8, 10, 17, 18, 19],
  lineNo: 143,
  code: `        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="faces" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'SWA 模板', b: 'phi3.cpp:60 按 swa_type 选<br>graph&lt;true&gt; / graph&lt;false&gt;<br>phimoe.cpp:49 选同一对', m: 'hparams.swa_type' },
      { c: 'c', t: 'MSA', b: 'minimax-m3.cpp 自带<br>llama_kv_cache_msa_context<br>与 build_attn_msa_fa', m: 'llama_model_minimax_m3::graph' },
      { c: 'e', t: 'MTP 草稿头', b: 'mimo2.cpp:85 · step35.cpp<br>nemotron-h-moe.cpp:4<br>都会切到 graph_mtp', m: 'LLM_GRAPH_TYPE_DECODER_MTP' }
    ];
    const host = wrap.querySelector('#faces');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'MoE 的上游是一段普通的残差流，<span class="k">但这段流可以是三种完全不同的注意力</span>。',
      'phi3 / phimoe：同一份图代码模板化两份，SWA 与非 SWA 各一份，MoE 分支两边都在。',
      'minimax-m3：注意力换成了自己实现的 MSA（608 行里 96 行起是它自己的 graph input），<br>MoE 调用仍然只有一处（561）。',
      'mimo2 / step35 / nemotron-h-moe：主干之外还有一个 <span class="v">graph_mtp</span>，<br>里面同样要再调一次 <span class="v">build_moe_ffn</span>。',
      '结论：<span class="k">build_moe_ffn 不关心上游是哪种注意力</span> —— 它只吃一个 [n_embd, n_tokens] 的张量。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(3400 + i * 4200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 9 把 24 个文件压成 <span class="hl-d">一张表</span> */
{
  kicker: "L2-13 · 总表",
  title: "把 24 个文件压成 <span class=\"hl-d\">一张表</span>",
  sub: "每格都写了证据行号；共享专家与分组路由是两条互不相干的开关。",
  caption: "完整的 24 行逐文件表在 source.md 的第五节。",
  src: "src/llama-hparams.h",
  mark: [0, 2, 4, 5, 14, 15, 16, 17],
  lineNo: 110,
  code: `    uint32_t n_ff_shexp         = 0;
    uint32_t n_ff_chexp         = 0;
    uint32_t n_expert_shared    = 0;
    uint32_t n_norm_groups      = 0;
    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_group_experts    = 0;

    // MLA + SWA (i.e. dots3note)
    uint32_t n_lora_kv_swa           = 0;
    uint32_t n_embd_head_k_mla_swa   = 0;
    uint32_t n_embd_head_v_mla_swa   = 0;

    float    expert_group_scale   = 0.05f;
    float    expert_weights_scale = 0.0f;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    uint32_t moe_every_n_layers   = 0;`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['变体', '命中文件（证据行）', '文件数'],
      [['调 build_moe_ffn', '20 个文件共 21 处调用（step35 主干 313 + MTP 512）', '20 / 24'],
       ['没有 MoE 调用', 'minicpm:71 · nomic-bert-moe:38 · phimoe:38 · refact:60', '4 / 24'],
       ['借别人的图', 'phimoe:663 · nomic-bert-moe:341 · minicpm:1739（models.h 行号）', '3 / 24'],
       ['声明共享专家', 'llama:85 · llama4:84 · minicpm:77 · minimax-m3:79 · mistral3:80 · nemotron-h-moe:103 · qwen2moe:54 · refact:66 · step35:110', '9 / 24'],
       ['shexp 自带门控', 'qwen2moe:54 的 ffn_gate_inp_shexp（24 个文件里唯一一个）', '1 / 24'],
       ['selection 偏置', 'mimo2:213 · minimax-01:447 · minimax-m2:135 · minimax-m3:566 · nemotron-h-moe:121 · step35:318/517', '6 / 24'],
       ['gating 读 hparams', 'minimax-m2:139 · minimax-m3:570 · smallthinker:158 · step35:322/521', '4 / 24'],
       ['probs_in 外部路由', 'smallthinker:159 · nemotron-h-moe:127', '2 / 24'],
       ['分组路由', 'n_expert_groups / n_group_used：24 个文件里 0 处', '0 / 24']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '九行覆盖全部 24 个文件；同一文件可以出现在多行（这些开关彼此独立）。',
      '20 个文件真的调了 <span class="v">build_moe_ffn</span>，4 个没有。',
      '共享专家 <span class="k">9 / 24</span>，其中只有 qwen2moe 给它配了独立门控。',
      '<span class="v">exp_probs_b</span> 这一行的来历写在源码注释里：<span class="v">llama-graph.cpp:2062</span> 注明 "introduced in DeepSeek V3"。',
      '分组路由这一行是 <span class="k">0 / 24</span>：它由 GGUF 元数据决定，代码里看不见。',
      '最后一行也是本课的结论：<span class="v">路由策略的开关，一半在参数表，一半在 hparams</span>。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2700, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(24000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 10 一句话：MoE 的可调参数不在算子，在 <span class="hl-a">路由策略</span> */
{
  kicker: "L2-13 · 收束",
  title: "一句话：MoE 的可调参数不在算子，在 <span class=\"hl-a\">路由策略</span>",
  sub: "共享专家、分组上限、top-k 归一化都是开关；模型文件只是选不同的组合。",
  caption: "下一课 L2-14 模型家族（五）稠密 Transformer（上）：没有专家时图长什么样。",
  src: "src/llama-graph.h",
  mark: [0, 1, 7, 8, 9, 11, 15, 20],
  lineNo: 1112,
  code: `    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr) const;`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问', '答'],
      [['控制专家选择的参数有哪些？', 'n_expert · n_expert_used · gating_op · exp_probs_b · norm_w · w_scale · probs_in · selected_experts_in'],
       ['哪一个决定"选几个专家"？', 'n_expert_used —— 2109 行 ggml_argsort_top_k(..., n_expert_used)'],
       ['分组路由怎么开？', '不是参数：hparams.n_expert_groups 大于 1 时自动生效（2083）'],
       ['共享专家在哪？', '不在 build_moe_ffn 里：调用点外的第二条支路（9/24 个文件）']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '在 <span class="mono">llama-graph.cpp:1993</span> 的签名里，哪些参数会改变"选中哪些专家"，' +
      '哪些只改变"选中之后怎么加权"？',
      '<b>改变选谁：</b><span class="mono">n_expert</span>（候选集合大小）、' +
      '<span class="mono">n_expert_used</span>（top-k 的 k，见 2109）、' +
      '<span class="mono">gating_op</span>（概率函数，2040-2059）、' +
      '<span class="mono">exp_probs_b</span>（选择偏置，2066，只加到 selection_probs）、' +
      '<span class="mono">probs_in</span>（绕开 gate_inp，2024-2032）、' +
      '<span class="mono">selected_experts_in</span>（绕开 top-k，2107-2111）。<br>' +
      '<b>只改加权：</b><span class="mono">norm_w</span>（2134-2148）、' +
      '<span class="mono">w_scale</span>（2149-2151）。<br>' +
      '<b>不改选择：</b>权重 <span class="mono">gate_inp/up_exps/gate_exps/down_exps</span>、' +
      '激活 <span class="mono">type_op</span>（2221 的 switch）、层号 <span class="mono">il</span>。<br>' +
      '另外 <span class="mono">hparams.n_expert_groups / n_group_used</span> 会在 2083-2103 段' +
      '把一部分专家直接 masking 成 <span class="mono">-INFINITY</span> —— 它也是"选谁"，但不是参数。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '把这一课压成四问四答：',
      '八个参数：两个定规模，四个定概率与权重，两个是外部接管通道。',
      '<span class="v">n_expert_used</span> 是唯一直接决定"几个专家"的参数。',
      '分组路由与共享专家都<b>不在这张参数表里</b> —— 一个在 hparams，一个是图上的第二条支路。',
      '下一课 <span class="k">L2-14</span>：没有专家的 Transformer，图画起来是什么样。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 3600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(18600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
