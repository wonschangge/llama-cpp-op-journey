/* ==========================================================================
   L2-14 · 模型家族（五）：稠密 Transformer（上）
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 37 个文件，<span class="hl-a">一条主干</span> */
{
  kicker: "L2-14 · 全局",
  title: "37 个文件，<span class=\"hl-a\">一条主干</span>",
  sub: "本课覆盖 src/models/ 下的 37 个文件；主干只有一条，差异都在\"层的开关\"上。",
  caption: "下一课 L2-15 接同一张清单的另外 36 个文件（注意力变体与草稿模型）。",
  src: "src/models/llama.cpp",
  mark: [1, 6, 16],
  lineNo: 94,
  code: `std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
//>> build_arch_graph：模型类把这个 arch 的图"造出来"的唯一入口（回顾 L2-05）
}

template <bool embed>
llama_model_llama::graph<embed>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
//>> ★ 主干就写在这个模板构造函数里；embed 是编译期开关（true = 只出句向量）
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
//>> 第 1 站：token id -> 嵌入向量`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">src/models/ 37 个文件</span><span class="arrow">-></span>
        <span class="chip a">同一条主干</span><span class="arrow">-></span>
        <span class="chip b">不同的层开关</span>
      </div>
      <div class="row" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '本课讲什么', b: '一个稠密 Transformer 的图<br>是怎么被一行行建出来的', m: 'llama_model_llama::graph' },
      { c: 'b', t: '为什么是 37 个', b: '主干相同，只有几个开关不同：<br>bias / norm 位置 / tied / SWA / 激活', m: 'build_ffn(..., LLM_FFN_*, ...)' },
      { c: 'c', t: '怎么读这一课', b: '先默写主干十站，再看<br>每个开关在源码里长什么样', m: '见末幕的练习' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '37 个文件，先别急着一个个读。它们的图**骨架相同**：<br><span class="k">嵌入 -> 逐层（norm -> 注意力 -> 残差 -> norm -> FFN -> 残差）-> norm -> lm_head</span>。',
      '<span class="v">llama.cpp:99</span> 是主干的入口：一个模板构造函数。<br><span class="k">模板参数 embed</span> 决定最后接不接 lm_head。',
      '<span class="v">llama.cpp:108</span> 起，主干一站一站往下走 —— 本课就用它当"标准答案"。',
      '记住这句话：<span class="k">37 个文件共享同一条主干，差异只在层的开关</span>。<br>这正是它们能并成一课的原因。',
      '下一站：把这十站逐行读出来。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 2 主干的第 1 段：<span class="hl-a">四样输入</span> */
{
  kicker: "L2-14 · 主干 1/4",
  title: "主干的第 1 段：<span class=\"hl-a\">四样输入</span>",
  sub: "主干开始之前，先把这一批 token 需要的东西全部备好：嵌入、位置、KV 入口、输出行。",
  caption: "回顾 L2-05：ubatch 决定这一批有多少 token；回顾 L2-04：KV cache 由 memory 上下文提供。",
  src: "src/models/llama.cpp",
  mark: [3, 7, 16, 23],
  lineNo: 105,
  code: `    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
//>> build_inp_embd：查表得到 [n_embd, n_tokens]，是图上第一个真算子（ggml_get_rows）

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();
//>> build_inp_pos：位置序列，RoPE 要用

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
//>> build_attn_inp_kv：不是 KV cache 本体，而是"这一层如何读写 cache + 掩码"的入口
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
//>> kq_scale：注意力缩放；超参给了就用超参（gemma 系靠它调 27B）

    ggml_tensor * inp_out_ids = build_inp_out_ids();
//>> build_inp_out_ids：只有最后要 logits 的那些 token 才需要算 head`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '嵌入', m: 'build_inp_embd(tok_embd)', b: 'token id -> 向量<br>llama.cpp:108' },
      { c: 'b', t: '位置', m: 'build_inp_pos()', b: '给 RoPE 用的位置序列<br>llama.cpp:111' },
      { c: 'c', t: '注意力入口', m: 'build_attn_inp_kv()', b: 'KV cache 的读写 + 掩码<br>llama.cpp:119' },
      { c: 'e', t: '输出行', m: 'build_inp_out_ids()', b: '只保留要出 logits 的行<br>llama.cpp:124' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '主干开跑前先备四样东西。注意它们都叫 <span class="v">build_inp_*</span>：<br><span class="k">在图上开一个"输入张量"</span>，运行时由 llama-context 填数据。',
      '<span class="v">build_inp_embd</span>：主干第 1 站，唯一一个真正算东西的输入（一次 <span class="k">ggml_get_rows</span> 查表）。',
      '<span class="v">build_inp_pos</span>：位置。<span class="k">只有用 RoPE 的模型需要它</span> —— bloom 就完全不调它（它用 ALiBi）。',
      '<span class="v">build_attn_inp_kv</span>：注意它<span class="k">不是 KV cache 本身</span>，而是这一层的 cache 读写入口 + 掩码。<br>换成 <span class="v">build_attn_inp_no_cache</span> 就变成无 cache 的双向注意力（dream / llada 就是这样）。',
      '<span class="v">build_inp_out_ids</span>：把"要出 logits 的行"挑出来。<br>这一招让最后只在 <span class="k">少量 token</span> 上跑 lm_head，是 prefill 提速的关键。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(17000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ 主干第 2 段：<span class="hl-a">逐层循环</span>的注意力半边 */
{
  kicker: "L2-14 · 主干 2/4",
  title: "★ 主干第 2 段：<span class=\"hl-a\">逐层循环</span>的注意力半边",
  sub: "n_layer 次循环，每次做四件事：norm、QKV、RoPE、注意力。全部通过 build_* 原语完成。",
  caption: "回顾 L2-02：n_layer 来自 hparams，从 GGUF 元数据读入；层数决定循环次数，不决定图的形状。",
  src: "src/models/llama.cpp",
  mark: [1, 4, 8, 20, 24, 49],
  lineNo: 126,
  code: `    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
//>> t_layer_inp：把每层输入存下来（投机解码 / 中间层输出要用）

        ggml_tensor * inpSA = inpL;
//>> inpSA：残差支路的起点，第 5 站要加回来

        // norm
        cur = build_norm(inpL,
//>> 第 2 站 build_norm —— 注意在注意力【之前】，这就是 pre-norm
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
//>> 第 3 站 build_qkv：一次调用拿走 Q/K/V，bias 加不加由张量是否存在决定
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
//>> 第 4 站 RoPE：位置信息在这里注入 Q 和 K
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
//>> use_kq_norm：少数模型（Llama4 文本塔）在 RoPE 后再做一次 Q/K RMSNorm
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            cur = build_attn(inp_attn,
//>> 第 5 站 build_attn：写 cache、读 cache、softmax、乘 V，一次调用全包
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div class="col" id="steps" style="gap:6px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = [
      { n: 'build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il)', l: 'llama.cpp:132', c: 'a',
        d: '注意力之前的归一化（pre-norm）' },
      { n: 'build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il)', l: 'llama.cpp:143', c: 'b',
        d: 'Q/K/V 投影；有 bias 张量就加 bias' },
      { n: 'ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, ...)', l: 'llama.cpp:146 / 152', c: 'c',
        d: '位置注入；Q、K 各做一次（V 不做）' },
      { n: 'build_attn(inp_attn, wo, wo_b, wo_s, Qcur, Kcur, Vcur, ...)', l: 'llama.cpp:169', c: 'd',
        d: '写 KV cache -> 取 mask -> softmax -> 输出投影' }
    ];
    const host = wrap.querySelector('#steps');
    const els = rows.map(r => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
      e.innerHTML = '<span class="cm" style="margin:0">' + U.esc(r.n) + '</span>' +
        ' <span style="color:var(--' + r.c + ')">' + U.esc(r.l) + '</span>' +
        '<div class="cb" style="margin-top:2px">' + r.d + '</div>';
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '一层的注意力半边 = <span class="k">四站</span>。四站全部走 build_* 原语，模型文件里看不到一个 ggml_* 算子。',
      '<span class="v">build_norm</span> 在注意力【之前】—— 这叫 <span class="k">pre-norm</span>，是 llama 系的做法。<br>gemma2 / gemma3 / exaone4 会在注意力【之后】再加一个（下一幕对照）。',
      '<span class="v">build_qkv</span> 一个调用返回 Q/K/V 三个张量。<br><span class="k">bias 加不加，由 GGUF 里有没有那个张量决定</span>（L2-06 的原语把它做成了可选）。',
      '<span class="v">ggml_rope_ext</span>：位置注入。<span class="k">只对 Q 和 K 做</span>，V 不动。<br>这就是"注意力知道谁在谁前面"的全部来源。',
      '<span class="v">build_attn</span>：把 Q/K/V 交给同一个原语，它负责写 cache、算 softmax、再乘 V。<br>换 mask、换 cache，注意力就换了个"变体" —— 那是 L2-15 的主题。'
    ];
    els.forEach((e, i) => tl.at(700 + i * 4200, () => {
      els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(20000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 主干第 3 段：残差 + FFN（<span class="hl-c">dense/MoE 就在这里分叉</span>） */
{
  kicker: "L2-14 · 主干 3/4",
  title: "主干第 3 段：残差 + FFN（<span class=\"hl-c\">dense/MoE 就在这里分叉</span>）",
  sub: "两个残差相加夹着一次 norm + 一次 FFN；走哪条 FFN 分支由一个指针是否为空决定。",
  caption: "MoE 分支的细节见 L2-12 / L2-13；本课只关心\"稠密这一支\"长什么样。",
  src: "src/models/llama.cpp",
  mark: [0, 5, 10, 19, 51, 55],
  lineNo: 174,
  code: `        if (il == n_layer - 1 && inp_out_ids) {
//>> inp_out_ids 只在这一层（最后一层）生效，省掉中间层的 head 开销
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
//>> 第 6 站残差相加：注意力输出 + 进层时的 inpSA
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {
//>> ★ 层的开关：ffn_gate_inp 为空 -> 稠密 FFN；否则 -> MoE 分支

            cur = build_norm(ffn_inp,
//>> 第 7 站 build_norm（ffn_norm）
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
//>> 第 8 站 build_ffn(SILU, PAR)：up + gate 并行，再乘 down
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

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
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
//>> 第 9 站第二个残差相加；223 行的 build_cvec 是适配器注入点，没有适配器时是恒等
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row" id="dia" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia');
    const left = U.el('div', { class: 'card', style: 'width:340px;border-left-color:var(--b)' });
    left.innerHTML = '<div class="ct" style="color:var(--b)">稠密分支（本课）</div>' +
      '<div class="cb"><span class="cm" style="margin:0">if (ffn_gate_inp == nullptr)</span><br>' +
      'build_norm -> build_ffn(SILU, PAR)<br>' +
      '一个 gate 张量、一个 up 张量、一个 down 张量<br>' +
      '<b>每个 token 都走同样的权重</b></div>';
    const right = U.el('div', { class: 'card', style: 'width:340px;border-left-color:var(--c)' });
    right.innerHTML = '<div class="ct" style="color:var(--c)">MoE 分支（L2-12 / L2-13）</div>' +
      '<div class="cb"><span class="cm" style="margin:0">else</span><br>' +
      'build_norm -> build_moe_ffn(...)<br>' +
      '专家权重是三维张量 [n_embd, n_ff, n_expert]<br>' +
      '<b>每个 token 只走 top-k 个专家</b></div>';
    dia.appendChild(left); dia.appendChild(right);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '注意力半边结束后，剩下的事只有三件：<span class="k">加残差、过 FFN、再加残差</span>。',
      '<span class="v">ffn_inp = ggml_add(cur, inpSA)</span>（llama.cpp:178）：<br>把注意力输出加回<span class="k">进层时的输入</span>，残差支路在这里闭合。',
      '<span class="v">if (ffn_gate_inp == nullptr)</span>（llama.cpp:182）：<br>★ <span class="k">整个稠密/MoE 的分叉就写在这一次判空上</span>。37 个文件全部走 dense 这一支。',
      '<span class="v">build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il)</span>：<br><span class="k">PAR</span> = gate 与 up 并行算（两条支路同一输入），再乘起来过 down。',
      '<span class="v">ggml_add(cur, ffn_inp)</span>（llama.cpp:220）：第二个残差。<br>到这里一层的形状就固定了：<span class="k">两次 norm、两次残差、一次注意力、一次 FFN</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[1]; });
    tl.at(7200, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[2]; });
    tl.at(11000, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[3]; });
    tl.at(15000, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[4]; });
    tl.at(20000, () => { left.style.opacity = '1'; right.style.opacity = '1'; msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 主干第 4 段：末层 norm + <span class="hl-b">lm_head</span> */
{
  kicker: "L2-14 · 主干 4/4",
  title: "主干第 4 段：末层 norm + <span class=\"hl-b\">lm_head</span>",
  sub: "循环结束后：一次 output_norm，然后接 lm_head；embed 模板为 true 时两处都被换掉。",
  caption: "回顾 L2-07：这段图产生的 t_logits / t_embd 就是 context 交给采样器或调用者的东西。",
  src: "src/models/llama.cpp",
  mark: [2, 8, 11, 14, 21],
  lineNo: 229,
  code: `    cur = inpL;

    cur = build_norm(cur,
//>> 第 10 站 output_norm：整个主干只有这一次是"层外"的归一化
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;
//>> t_embd：句向量出口。embedding 模型就到此为止

    if constexpr (!embed) {
//>> ★ 编译期开关：embed == true 就不建 lm_head（llama-embed.cpp 用的是 graph<true>）
        // lm_head
        cur = build_lora_mm(model.output, cur, model.output_s);
//>> lm_head：build_lora_mm(model.output, cur)；output 可能就是 tok_embd 的副本

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
//>> 把这一整条链挂到计算图上，图构建结束
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="dia" style="gap:10px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia');
    const a1 = U.el('div', { class: 'card', style: 'width:330px;border-left-color:var(--a)' });
    a1.innerHTML = '<div class="ct" style="color:var(--a)">graph&lt;false&gt; · 生成</div>' +
      '<div class="cb">注意力入口 = <b>build_attn_inp_kv()</b>（L119）<br>' +
      'output_norm -> <b>lm_head</b><br>' +
      '<span class="cm" style="margin:0">res-&gt;t_logits = cur;</span><br>' +
      'llama.cpp:240 / 243</div>';
    const a2 = U.el('div', { class: 'card', style: 'width:330px;border-left-color:var(--b)' });
    a2.innerHTML = '<div class="ct" style="color:var(--b)">graph&lt;true&gt; · 取句向量</div>' +
      '<div class="cb">注意力入口 = <b>build_attn_inp_no_cache()</b>（L117）<br>' +
      'output_norm -> <b>停</b><br>' +
      '<span class="cm" style="margin:0">res-&gt;t_embd = cur;</span><br>' +
      'llama-embed.cpp:4 用的就是这个特化</div>';
    dia.appendChild(a1); dia.appendChild(a2);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '循环结束，<span class="v">cur = inpL</span>。剩下的只有两站。',
      '<span class="v">build_norm(cur, output_norm, NULL, LLM_NORM_RMS, -1)</span>：<br>注意 <span class="k">il = -1</span>：它不是某一层的 norm，是整条主干出口的 norm。',
      '<span class="v">res-&gt;t_embd = cur</span>：句向量出口。<span class="k">embedding 模型到这里就结束</span>（gemma-embedding / eurobert 连 lm_head 都没有）。',
      '<span class="v">if constexpr (!embed)</span>：<span class="k">编译期开关</span>。<br>llama-embed.cpp 全文件只有 6 行，它用 <span class="v">graph&lt;true&gt;</span>，于是这一整段被编译掉。',
      '★ 开关不止一处：<span class="v">embed</span> 在 L113-117 还把注意力入口换成了 <span class="k">no_cache</span> 版。<br>所以 graph&lt;true&gt; = 不写 KV cache + 不出 logits。',
      '<span class="v">build_lora_mm(model.output, cur, model.output_s)</span>：最后一个矩阵乘。<br><span class="k">output 可能只是 tok_embd 的一个副本</span>（tied embedding），见下一幕的开关表。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3200, () => { msg.innerHTML = texts[1]; });
    tl.at(6000, () => { a2.style.opacity = '.35'; msg.innerHTML = texts[2]; });
    tl.at(9000, () => { a1.style.opacity = '.35'; a2.style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(12000, () => { a1.style.opacity = '1'; msg.innerHTML = texts[4]; });
    tl.at(15000, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 ★ 主干上的每一站，都是一个 <span class="hl-d">build_* 原语</span> */
{
  kicker: "L2-14 · 原语",
  title: "★ 主干上的每一站，都是一个 <span class=\"hl-d\">build_* 原语</span>",
  sub: "模型文件里没有一个 ggml_* 算子：图被拆成 6 个原语，模型只负责\"按顺序调用 + 选开关\"。",
  caption: "原语的实现见 L2-06；本幕只把它们与主干的站一一对上。",
  src: "src/llama-graph.h",
  mark: [0, 5, 18, 29, 51],
  lineNo: 1048,
  code: `    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora and per-tensor scale
    ggml_tensor * build_lora_mm(
//>> build_lora_mm：一次矩阵乘 + 可选 LoRA + 可选 per-tensor scale
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * w_s = nullptr) const;

    ggml_tensor * build_norm(
//>> build_norm：cur + 权重(+bias) + 三种归一化之一
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
//>> build_qkv：Q/K/V 三合一，bias 可选（见下一幕开关 ②）
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const;

    // Set reshape to false to return contiguous projections before clamp/reshape.
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape = true) const;

    ggml_tensor * build_ffn(
//>> build_ffn：up / gate / down 三块权重 + 激活类型 + 门控方式
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['主干上的站', '原语', '...cpp 里的实现行', '它替你做了几件事'],
      [['嵌入 / 输出行', 'build_inp_embd / build_inp_out_ids', '2362 / 2480', '开输入张量、查表、挑行'],
       ['归一化', 'build_norm', '1583', '一个 switch 分派 RMS / LayerNorm / GroupNorm，再乘权重加偏置'],
       ['Q/K/V 投影', 'build_qkv', '1633', 'fused 或分离两条路；bias 有就加，没有就跳过'],
       ['自注意力', 'build_attn', '2851', '写 KV cache、取掩码、softmax、输出投影'],
       ['FFN', 'build_ffn', '1748', 'up + gate（PAR 并行 / SEQ 串行）+ 激活 + down'],
       ['lm_head / 投影', 'build_lora_mm', '1514', '矩阵乘 + 可选 LoRA + 可选 scale'],
       ['控制向量', 'build_cvec', '1508', '适配器注入点，没有适配器时是恒等']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '把主干的十站压成 <span class="k">7 个原语</span>：模型文件只写"调用顺序"，不写算子。',
      '输入类原语（<span class="v">build_inp_*</span>）只负责<span class="k">在图上开张量</span>，真正的数据由运行时灌进来。',
      '<span class="v">build_norm</span> 是三条归一化的统一入口。<span class="k">用哪条由参数决定</span> —— 这是"层开关 ①"的实现方式。',
      '<span class="v">build_qkv</span> / <span class="v">build_ffn</span> 的签名里塞满了可选张量（bias、scale、LoRA）。<br><span class="k">同一个原语，靠"传不传"就能表达不同模型</span>。',
      '<span class="v">build_attn</span> 有 7 个重载（按 cache 类型分）：普通 KV、无 cache、SWA、cross、MLA……<br><span class="k">选哪个重载 = 选哪种注意力</span> —— 这是 L2-15 的主线。',
      '所以"模型家族"在这个仓库里的真实含义是：<span class="k">原语的选择 + 超参的取值</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(19000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 ★ 六个"层开关"：同一行调用，<span class="hl-e">两种行为</span> */
{
  kicker: "L2-14 · 核心",
  title: "★ 六个\"层开关\"：同一行调用，<span class=\"hl-e\">两种行为</span>",
  sub: "gemma3 的主干与 llama 逐站对应，但多了 QK-norm、post-norm、SWA、softcap。",
  caption: "gemma3 是\"开关拨得最多\"的稠密模型之一：它仍然走同一条主干十站。",
  src: "src/models/gemma3.cpp",
  mark: [0, 9, 33, 49, 57],
  lineNo: 131,
  code: `            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
//>> 开关②：Q 先做一次 RMSNorm 再进 RoPE（llama 没有这一步）
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
//>> K 同样处理；norm 的是「每个头的维度」n_embd_head_k
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
//>> 开关①：注意力【之后】再 norm 一次 —— post-norm，llama 没有
                model.layers[il].attn_post_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        cur = build_norm(sa_out,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
//>> FFN 之后还有第三个 norm —— gemma 的 sandwich 结构
                model.layers[il].ffn_post_norm, NULL,`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['层开关', 'llama.cpp 的拨法', '其它文件的拨法（本课 37 个里）'],
      [['① norm 位置', 'pre-norm：只在注意力/FFN 之前（L132/L184）',
        'post-norm：gemma2/gemma3 L162、glm4 L140、exaone4 L147（连 pre-norm 都没有）'],
       ['② QKV / wo bias', '有张量才加：build_qkv 里判空（llama-graph.cpp:1653）',
        'bloom / gpt2 恒有（fused wqkv+bias）；command-r / internlm2 的 wo_b 从未创建'],
       ['③ QK-norm', '无（只有 use_kq_norm 开关，L162）',
        'gemma3 L131/L139、apertus L93、exaone4 L122；command-r 可选且用 LayerNorm'],
       ['④ sliding window', '无：build_attn_inp_kv（L119）',
        'gemma2/gemma3/cohere2/exaone4/gemma-embedding：iswa 版入口 + is_swa(il)'],
       ['⑤ FFN 激活 / 门控', 'LLM_FFN_SILU + LLM_FFN_PAR（L194）',
        'GELU：bloom/gpt2/gemma 系；SwiGLU：chatglm/glm4；ReLU²：arcee/jais2；xIELU：apertus 手写'],
       ['⑥ embedding 是否 tied', 'output 缺失就复用 tok_embd（llama.cpp L41-46）',
        '恒 tied：gemma L20、cohere2 L29、command-r L21、bitnet L164；不 tied：baichuan、apertus']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '同一条主干，六处拨法不同 —— 这就是 37 个文件的全部差异来源。',
      '<span class="v">① norm 位置</span>：llama 是 pre-norm；<span class="k">gemma 系在残差前再加一次</span>（post-norm）。<br>exaone4 更极端：它<span class="k">根本没有 pre-norm</span>。',
      '<span class="v">② bias</span>：注意这不是"模型文件选"的 —— <span class="k">build_qkv 判的是张量在不在</span>，<br>而张量在不在由 GGUF 决定。同一个模型文件能同时支持两种。',
      '<span class="v">③ QK-norm</span>：norm 的是<span class="k">每个头的向量</span>（n_embd_head_k），不是整层。<br>它压的是注意力 logits 的量级。',
      '<span class="v">④ sliding window</span>：开关落在<span class="k">输入入口的选型</span>上（iswa 版 KV / 掩码），<br>而不是某个算子里 —— 所以主干十站一个都不少。',
      '<span class="v">⑤ 激活</span>：PAR 与 SEQ 的区别是 gate 吃谁的输出：<span class="k">PAR 吃输入，SEQ 吃 up 的输出</span>。',
      '<span class="v">⑥ tied embedding</span>：lm_head 与嵌入共享权重。<span class="k">省一份 [n_embd, n_vocab] 的显存</span>，<br>代价是输出与输入被绑在一起。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 3000, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(22000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[0]; });
  }
},

/* ------------------------------------------------------ 8 37 个文件的真实差异（上：19 个） */
{
  kicker: "L2-14 · 清单 1/2",
  title: "37 个文件的真实差异（上：19 个）",
  sub: "逐行读出来的差异点。列\"家族\"里带 ★ 的，实际上不属于稠密 Transformer。",
  caption: "\"归一化\"列写的是该文件图里真正调用的归一化类型；\"-\"表示这个文件里没有图。",
  src: "src/models/bitnet.cpp",
  mark: [6, 9, 16],
  lineNo: 149,
  code: `        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
//>> 和其它文件一模一样的收尾：output_norm
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    // FIXME: do not use model.tok_embd directly, duplicate as model.output
    cur = build_lora_mm(model.tok_embd, cur);
//>> ★ bitnet 的 lm_head 直接用 tok_embd（tied），而且它压根没建 output 张量

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:6px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg" style="font-size:10px"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['apertus.cpp', 'RMS', 'xIELU', 'FFN 手写：up -> ggml_xielu -> down（L138）'],
      ['arcee.cpp', 'RMS', 'ReLU2', '无 gate，SEQ 串行（L128）'],
      ['baichuan.cpp', 'RMS', 'SILU', '13B 不接位置编码 + ALiBi（L13 / L90）'],
      ['bitnet.cpp', 'RMS x5', 'SILU', '层内多两个 sub-norm（L103 / L137）'],
      ['bloom.cpp', 'LayerNorm', 'GELU', 'fused wqkv+bias（L45）；嵌入后先归一化（L77）'],
      ['chameleon.cpp', 'RMS + LN', 'SILU', 'Q/K 用 LayerNorm 且可选（L91-104）'],
      ['chatglm.cpp', 'RMS', 'SwiGLU', 'gate 串行：up 出 2 倍宽（L48 / L133）'],
      ['codeshell.cpp', 'LayerNorm', 'GELU', '反向 tie：tok_embd 复用 output（L15-20）'],
      ['cogvlm.cpp', 'RMS', 'SILU', '★ 文本/视觉两套权重按 ubatch 二选一（L72-97）'],
      ['cohere2.cpp', 'LayerNorm', 'SILU', 'SWA 与全局层交错（L70）；并行残差（L131）'],
      ['command-r.cpp', 'LayerNorm', 'SILU', '并行残差（L118）；无 ffn_norm；强制 tie（L21）'],
      ['dream.cpp', 'RMS', 'SILU', '★ 非因果 + 无 KV cache（L15 / L68）：扩散式'],
      ['eagle3.cpp', 'RMS', 'SILU', '★ 两张图：encoder 只做 fc（L137），草稿模型'],
      ['eurobert.cpp', 'RMS', 'SILU', '★ 无 cache（L50）、无 lm_head：编码器'],
      ['exaone.cpp', 'RMS', 'SILU', '逐层 rope_freqs（L75）；输出可 tied（L24）'],
      ['exaone4.cpp', 'RMS', 'SILU', '无 pre-norm，全 post-norm（L113）；SWA；跳过 nextn 层（L42）'],
      ['falcon.cpp', 'LayerNorm', 'GELU', 'FFN 吃 attn_norm 而非 attn 输出（L125）'],
      ['gemma-embedding.cpp', 'RMS', 'GELU', '★ causal_attn=false（L7）+ 无 cache（L89）'],
      ['gemma.cpp', 'RMS', 'GELU', '嵌入乘 sqrt(n_embd)（L49）；output 恒为副本（L20）']
    ];
    const t = U.table(['文件', '归一化', '激活', '真实差异点（行号在源文件里）'], rows, { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    t.el.style.fontSize = '9px';
    t.el.querySelectorAll('th, td').forEach(c => { c.style.padding = '1px 4px'; c.style.lineHeight = '1.22'; });

    const msg = wrap.querySelector('#msg');
    const trs = t.body.querySelectorAll('tr');
    const texts = [
      '这 19 个文件，主干都在；差异只有最后一列那一点点。',
      '★ 标记的三个（cogvlm / dream / eagle3）实际属于别的家族：<br>视觉专家、扩散式非因果、投机解码草稿。',
      '再看一遍最左列：<span class="k">归一化类型</span>基本就是"这个模型是新一代还是老一代"的分水岭。',
      '最后一行 gemma 最有代表性：<span class="k">output 无条件复用 tok_embd</span>（L20 直接 TENSOR_DUPLICATED）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    trs.forEach((r, i) => tl.at(2500 + i * 900, () => {
      if (i % 3 === 0) { r.className = 'on'; }
      if (i === 8) { msg.innerHTML = texts[1]; }
      if (i === 17) { msg.innerHTML = texts[2]; }
    }));
    tl.at(20000, () => { msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 9 37 个文件的真实差异（下：18 个） */
{
  kicker: "L2-14 · 清单 2/2",
  title: "37 个文件的真实差异（下：18 个）",
  sub: "下半张表里出现了\"根本没有主干\"的文件 —— 它们的图被一行 using 继承走了。",
  caption: "hunyuan-dense / llama-embed 各只有 6 行；jina-bert-v2/v3 的图在 bert.cpp（别的课）。",
  src: "src/models/apertus.cpp",
  mark: [6, 16, 21],
  lineNo: 123,
  code: `        // feed-forward network with xIELU activation
        {
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // Up projection
            ggml_tensor * up = build_lora_mm(model.layers[il].ffn_up, cur);
//>> up 投影还是原语
            cb(up, "ffn_up", il);

            float alpha_n_val = hparams.xielu_alpha_n[il];
            float alpha_p_val = hparams.xielu_alpha_p[il];
            float beta_val    = hparams.xielu_beta[il];
            float eps_val     = hparams.xielu_eps[il];

            // Apply xIELU activation
            ggml_tensor * activated = ggml_xielu(ctx0, up, alpha_n_val, alpha_p_val, beta_val, eps_val);
//>> ★ 中间这一步不走 build_ffn：xIELU 是 LLM_FFN_* 枚举里没有的激活
            cb(activated, "ffn_xielu", il);

            // Down projection
            cur = build_lora_mm(model.layers[il].ffn_down, activated);
//>> down 投影又回到原语；主干两侧没变，只换了中间那一步
            cb(cur, "ffn_down", il);
        }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:6px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg" style="font-size:10px"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['gemma2.cpp', 'RMS', 'GELU', 'post-norm + SWA（L74）+ 两处 softcap（L14 / L167）'],
      ['gemma3.cpp', 'RMS', 'GELU', 'QK-norm（L131 / L139）+ post-norm + SWA + tied（L44）'],
      ['gemma3n.cpp', 'RMS', 'GELU', '★ AltUp/Laurel + 逐层嵌入（L324-376）；FFN 手写'],
      ['gemma4-assistant.cpp', 'RMS', 'GELU', '★ 只建 n_layer_nextn 层（L127），KV 借主模型（L151）'],
      ['glm4.cpp', 'RMS', 'SwiGLU', 'post-norm（L140 / L162）+ mRoPE（L81）'],
      ['gpt2.cpp', 'LayerNorm', 'GELU', 'fused wqkv+bias（L38）；学习式位置，无 RoPE（L74-77）'],
      ['gptneox.cpp', 'LayerNorm', 'GELU', 'use_par_res：注意力与 FFN 并行（L145）；带 bias（L67/L70）'],
      ['granite-switch.cpp', 'RMS', 'SILU', '★ 自定义输入 + 手写 FFN + 图内路由器（L134-147 / L411-415）'],
      ['hrm-text.cpp', 'RMS', 'SILU', '★ 深度循环双栈交替跑同一批层（L101 / L168）'],
      ['hunyuan-dense.cpp', '-', '-', '只有 6 行：图继承 hunyuan-vl（models.h L2105）'],
      ['hunyuan-vl.cpp', 'RMS', 'SILU', '★ 多模态文本半边：M-RoPE 4 段（L66-69）+ 视觉 tap（L86）'],
      ['internlm2.cpp', 'RMS', 'SILU', '无 wo_b；QKV 走 create_tensor_qkv（L26）'],
      ['jais.cpp', 'LayerNorm', 'SILU', 'ALiBi（L5）；不调 build_inp_pos（无 RoPE）'],
      ['jais2.cpp', 'LayerNorm', 'ReLU2', '无 gate，SEQ（L130）；tied（L22-24）'],
      ['jina-bert-v2.cpp', 'LayerNorm', '-', '★ 本文件无主干：图在 bert.cpp（models.h L308）'],
      ['jina-bert-v3.cpp', 'LayerNorm', '-', '★ 49 行只有张量表；图在 bert.cpp（models.h L319）'],
      ['llada.cpp', 'RMS', 'SILU', '★ 非因果（L16）+ 无 KV cache（L82）：扩散式'],
      ['llama-embed.cpp', '-', '-', '只有 6 行：复用 llama.cpp 主干，graph<true>']
    ];
    const t = U.table(['文件', '归一化', '激活', '真实差异点（行号在源文件里）'], rows, { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    t.el.style.fontSize = '9px';
    t.el.querySelectorAll('th, td').forEach(c => { c.style.padding = '1px 4px'; c.style.lineHeight = '1.22'; });

    const msg = wrap.querySelector('#msg');
    const trs = t.body.querySelectorAll('tr');
    const texts = [
      '下半张表：★ 有九个 —— 它们才是"37 个"里真正需要解释的部分。',
      '★ 四个<span class="k">没有主干</span>：hunyuan-dense / llama-embed（各 6 行）、<br>jina-bert-v2 / jina-bert-v3（图在 bert.cpp）。',
      '★ 两个<span class="k">自成一派</span>：granite-switch（每 token 一套 LoRA + 图内路由器）、<br>hrm-text（深度循环双栈，同一批层跑多轮）。',
      '★ 三个<span class="k">别的家族</span>：gemma3n（多模态家族）、gemma4-assistant（草稿头）、llada（扩散式）。',
      '结论：<span class="k">"37 个稠密模型"是计划分组的结果，不是同质的 37 份代码</span>。<br>主干十站在它们里面大多还在，只是被机制盖住了。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    trs.forEach((r, i) => tl.at(2500 + i * 850, () => {
      if (i % 3 === 0) { r.className = 'on'; }
      if (i === 9) { msg.innerHTML = texts[1]; }
      if (i === 11) { msg.innerHTML = texts[2]; }
      if (i === 13) { msg.innerHTML = texts[3]; }
    }));
    tl.at(20000, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 把主干默写下来 */
{
  kicker: "L2-14 · 收束",
  title: "把主干默写下来",
  sub: "十站、七个原语、六个开关。能默写这张表，这一课的验收点就达成了。",
  caption: "下一课 L2-15：同一条主干上的注意力变体（GQA / SWA / MLA）与投机解码草稿模型。",
  src: "src/models/models.h",
  mark: [0, 4, 6],
  lineNo: 175,
  code: `struct llama_model_llama_embed : public llama_model_llama {
    llama_model_llama_embed(const struct llama_model_params & params) : llama_model_llama(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_llama

    template <bool embed>
//>> 注释说得很直白：hparams 与张量加载都复用 llama
    using graph = llama_model_llama::graph<embed>;
//>> ★ 一行 using，就把 llama.cpp 里那整条主干继承过来了

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['#', '主干十站', '行号', '原语'],
      [['1', 'build_inp_embd(model.tok_embd)', 'llama.cpp:108', 'build_inp_embd'],
       ['2', 'build_inp_pos()', 'llama.cpp:111', 'build_inp_pos'],
       ['3', 'build_attn_inp_kv()（KV 读写 + 掩码入口）', 'llama.cpp:119', 'build_attn_inp_kv'],
       ['4', 'build_inp_out_ids()（只留要输出的行）', 'llama.cpp:124', 'build_inp_out_ids'],
       ['5', 'for il in 0..n_layer-1', 'llama.cpp:126', '-'],
       ['5a', 'build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il)', 'llama.cpp:132', 'build_norm'],
       ['5b', 'build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il)', 'llama.cpp:143', 'build_qkv'],
       ['5c', 'ggml_rope_ext(Q) / ggml_rope_ext(K)', 'llama.cpp:146 / 152', 'ggml_rope_ext'],
       ['5d', 'build_attn(inp_attn, wo, wo_b, wo_s, Q, K, V, ...)', 'llama.cpp:169', 'build_attn'],
       ['5e', 'ggml_add(cur, inpSA)  <- 残差 1', 'llama.cpp:178', 'ggml_add'],
       ['5f', 'build_norm(ffn_inp, ffn_norm, NULL, LLM_NORM_RMS, il)', 'llama.cpp:184', 'build_norm'],
       ['5g', 'build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il)', 'llama.cpp:189', 'build_ffn'],
       ['5h', 'ggml_add(cur, ffn_inp)  <- 残差 2', 'llama.cpp:220', 'ggml_add'],
       ['6', 'build_norm(cur, output_norm, NULL, LLM_NORM_RMS, -1)', 'llama.cpp:231', 'build_norm'],
       ['7', 'build_lora_mm(model.output, cur, model.output_s)', 'llama.cpp:240', 'build_lora_mm'],
       ['8', 'ggml_build_forward_expand(gf, cur)', 'llama.cpp:246', 'ggml_build_forward_expand']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    t.el.style.fontSize = '9px';
    t.el.querySelectorAll('th, td').forEach(c => { c.style.padding = '1px 5px'; c.style.lineHeight = '1.25'; });

    wrap.querySelector('#ex').appendChild(W.exercise(
      '合上源码：写出 <span class="mono">llama_model_llama::graph</span> 的主干序列 —— ' +
      '每一步的<b>函数名 + 作用 + 是第几站</b>。再回答：<span class="mono">ffn_gate_inp</span> 判空是在决定什么？',
      '<b>十站</b>：<br>' +
      '1. <span class="mono">build_inp_embd(model.tok_embd)</span>（108）—— token 查表成向量<br>' +
      '2. <span class="mono">build_inp_pos()</span>（111）—— 位置序列<br>' +
      '3. <span class="mono">build_attn_inp_kv()</span>（119）—— KV cache 读写 + 掩码入口<br>' +
      '4. <span class="mono">build_inp_out_ids()</span>（124）—— 只要 logits 的行<br>' +
      '5. <span class="mono">for il in 0..n_layer-1</span>（126），层内依次：<br>' +
      '&nbsp;&nbsp;<span class="mono">build_norm(attn_norm, LLM_NORM_RMS)</span>（132）→ ' +
      '<span class="mono">build_qkv</span>（143）→ <span class="mono">ggml_rope_ext</span>（146/152）→ ' +
      '<span class="mono">build_attn</span>（169）→ <span class="mono">ggml_add(cur, inpSA)</span>（178）→ ' +
      '<span class="mono">build_norm(ffn_norm, LLM_NORM_RMS)</span>（184）→ ' +
      '<span class="mono">build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)</span>（189）→ ' +
      '<span class="mono">ggml_add(cur, ffn_inp)</span>（220）→ <span class="mono">build_cvec</span>（223）<br>' +
      '6. <span class="mono">build_norm(output_norm, LLM_NORM_RMS, -1)</span>（231）<br>' +
      '7. <span class="mono">build_lora_mm(model.output, cur, model.output_s)</span>（240）<br>' +
      '8. <span class="mono">ggml_build_forward_expand(gf, cur)</span>（246）<br><br>' +
      '<b>ffn_gate_inp 判空</b>是在决定这一层走<b>稠密 FFN</b>（为空，本课全部 37 个文件）' +
      '还是 <b>MoE 分支</b>（非空，L2-12 / L2-13 那一族）。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '十六行表 = 这一课的验收点。前四站是输入，中间是层循环，最后两站是出口。',
      '<span class="k">输入四站</span>：都叫 build_inp_*，只开张量不算数。',
      '<span class="k">层内九步</span>：两次 norm、两次残差、一次 QKV、一次 RoPE、一次注意力、一次 FFN。',
      '<span class="k">出口两站</span>：output_norm + lm_head（embed=true 时后者被编译掉）。',
      '记住一句话：<span class="v">所有稠密模型共享同一条主干，差异只在层的开关</span>。<br>QKV bias、norm 位置、tied embedding、sliding window、激活 —— 就这五个常见开关。',
      '下一课 L2-15 接着看：同一条主干上，注意力本身怎么换（GQA / SWA / MLA），<br>以及投机解码的草稿模型怎么把这条主干剪短。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3500, () => { [0, 1, 2, 3].forEach(i => { rows[i].className = 'on'; }); msg.innerHTML = texts[1]; });
    tl.at(8000, () => {
      rows.forEach(r => { r.className = ''; });
      for (let i = 4; i < 13; i++) { rows[i].className = 'on'; }
      msg.innerHTML = texts[2];
    });
    tl.at(14000, () => {
      rows.forEach(r => { r.className = ''; });
      rows[13].className = 'on'; rows[14].className = 'on'; rows[15].className = 'on';
      msg.innerHTML = texts[3];
    });
    tl.at(19000, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[4]; });
    tl.at(23000, () => { msg.innerHTML = texts[5]; });
  }
},

];
