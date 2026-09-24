/* ==========================================================================
   L2-11 · 模型家族（二）：状态空间与线性注意力
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 30 个文件里，<span class="hl-a">22 个</span>真是状态空间/线性注意力，<span class="hl-e">8 个</span>只是名字里有 wkv */
{
  kicker: "L2-11 · 全局",
  title: "30 个文件里，<span class=\"hl-a\">22 个</span>真是状态空间/线性注意力，<span class=\"hl-e\">8 个</span>只是名字里有 wkv",
  sub: "分组来自静态扫描：图构建代码里命中 build_ssm / ssm_conv / ssm_scan / gated_delta / build_rwkv / wkv 之一。命中原语不等于\"它就是 SSM 模型\"。",
  caption: "逐文件的真实主题见第 8、9 幕的表格，以及 source.md 第一节。",
  src: "src/models/mamba.cpp",
  mark: [4, 12, 22, 23, 26],
  lineNo: 83,
  code: `std::unique_ptr<llm_graph_context> llama_model_mamba::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_mamba::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_build_mamba_base(params) {
//>> 整个 Mamba 图继承 llm_build_mamba_base —— SSM 的图原语全在那个基类里
    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();
//>> build_rs_inp()：recurrent 记忆的输入占位，与 KV cache 的 build_attn_inp_kv 并列（L2-04）

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (model.arch == LLM_ARCH_MAMBA2) {
            cur = build_mamba2_layer(rs_inp, cur, model, ubatch, il);
//>> Mamba-2 走 build_mamba2_layer
        } else {
            cur = build_mamba_layer(rs_inp, cur, model, ubatch, il);
//>> Mamba-1 走 build_mamba_layer
        }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip a">30 个 src/models/*.cpp</span><span class="arrow">-&gt;</span>
        <span class="chip b">4 个基类</span><span class="arrow">-&gt;</span>
        <span class="chip c">3 个 ggml 算子</span><span class="arrow">-&gt;</span>
        <span class="chip d">固定大小状态张量</span>
      </div>
      <div class="row" style="gap:8px">
        <div class="card" style="border-left-color:var(--b);flex:1 1 0">
          <div class="ct" style="color:var(--b)">4 个图基类（本课的主角）</div>
          <div class="cb"><span class="cm" style="margin:0">llm_build_mamba_base</span>（Mamba-1/2 及 5 个混合模型）<br>
          <span class="cm" style="margin:0">llm_build_rwkv6_base</span> / <span class="cm" style="margin:0">llm_build_rwkv7_base</span><br>
          <span class="cm" style="margin:0">llm_build_delta_net_base</span>（gated delta net / KDA）</div>
        </div>
        <div class="card" style="border-left-color:var(--c);flex:1 1 0">
          <div class="ct" style="color:var(--c)">3 个融合算子</div>
          <div class="cb"><span class="cm" style="margin:0">ggml_ssm_conv</span> — 短卷积，读写卷积状态<br>
          <span class="cm" style="margin:0">ggml_ssm_scan</span> — 选择性扫描，Mamba 的核心<br>
          <span class="cm" style="margin:0">ggml_rwkv_wkv6 / wkv7 / gated_linear_attn / gated_delta_net</span></div>
        </div>
        <div class="card" style="border-left-color:var(--e);flex:1 1 0">
          <div class="ct" style="color:var(--e)">8 个名实不符</div>
          <div class="cb"><span class="cm" style="margin:0">deepseek2 · deepseek32 · dots3note<br>glm-dsa · hy-v4 · minicpm3 · plm · dflash</span><br>
          命中的是 MLA 的 <span class="cm" style="margin:0">wkv_a_mqa</span> 子串，走 KV cache。</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先给全局：本课要回答的是 <span class="k">一个 SSM/线性注意力层，在 ggml 图上长什么样</span>。',
      '入口非常统一：<span class="v">llama_model_mamba::graph</span> 继承 <span class="v">llm_build_mamba_base</span>，'
        + '逐层调 <span class="v">build_mamba_layer</span> 或 <span class="v">build_mamba2_layer</span>。<br>'
        + 'RWKV 与 delta net 各自有同构的基类。',
      '算子只有少数几个：<span class="v">ggml_ssm_conv</span> 管卷积状态，<span class="v">ggml_ssm_scan</span> 管 SSM 状态，'
        + '其它家族各有一个融合算子。',
      '<span class="k">四个基类 + 三个算子 = 22 个模型文件</span>。剩下 8 个是被 <span class="v">wkv</span> 子串扫进来的，'
        + '第 9 幕会逐个说明。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4200, () => { msg.innerHTML = texts[1]; });
    tl.at(9200, () => { msg.innerHTML = texts[2]; });
    tl.at(14800, () => { msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">SSM 不需要 KV cache</span>：历史被压进一份固定大小的状态 */
{
  kicker: "L2-11 · 核心",
  title: "★ <span class=\"hl-a\">SSM 不需要 KV cache</span>：历史被压进一份固定大小的状态",
  sub: "两种记忆各占一个张量：get_r_l() 是卷积状态，get_s_l() 是 SSM 状态。它们的大小由超参决定，与 n_ctx 无关。",
  caption: "对比 L2-04 第 10 幕：那一幕讲 memory 侧的接口，这一幕讲图侧怎么把它取出来用；两者的接口就是 get_r_l / get_s_l 这两个函数。",
  src: "src/models/mamba-base.cpp",
  mark: [2, 7, 8, 9, 13, 25, 27, 30],
  lineNo: 14,
  code: `    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();
//>> kv_head：这一批的每个序列映射到状态缓存的哪一行 —— 就是 L2-04 里 cell/slot 的同一套机制

    const auto & layer = model.layers[il];

    const int64_t d_conv         = hparams.ssm_d_conv;
    const int64_t d_inner        = hparams.ssm_d_inner;
    const int64_t d_state        = hparams.ssm_d_state;
    const int64_t dt_rank        = hparams.ssm_dt_rank;
    const int64_t n_head         = d_inner;
    const int64_t head_dim       = 1;
    const int64_t n_seqs         = ubatch.n_seqs;
//>> n_seqs 是这一批的序列数，不是上下文长度：state 里没有 token 维
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool    ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(d_inner % n_head == 0);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
//>> 卷积状态：形状由 n_embd_r() 决定
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
//>> SSM 状态：形状由 n_embd_s() 决定

    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
//>> build_rs() 把"状态缓存的若干行"取成"这一批要用的状态张量"（L2-06 的原语）
    conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="row" style="gap:8px">
        <div class="card" style="border-left-color:var(--g);flex:1 1 0">
          <div class="ct" style="color:var(--g)">KV cache：随上下文线性增长</div>
          <div class="cb">大小 ≈ <span class="cm" style="margin:0">n_ctx x n_layer x n_head_kv x head_dim</span><br>
          每来一个 token 就多一行；prompt 越长、显存越吃紧。</div>
        </div>
        <div class="card" style="border-left-color:var(--b);flex:1 1 0">
          <div class="ct" style="color:var(--b)">recurrent state：与 n_ctx 无关</div>
          <div class="cb">Mamba：<span class="cm" style="margin:0">ssm_d_state x ssm_d_inner</span><br>
          KDA：<span class="cm" style="margin:0">head_dim x head_dim x n_head</span><br>
          跑 10 个 token 和跑 10 万个 token，占的是同一块内存。</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['', '张量', '由谁给', '大小由什么决定'],
      [['卷积状态', 'get_r_l(il)', 'llama_memory_recurrent_context', 'n_embd_r()'],
       ['SSM 状态', 'get_s_l(il)', 'llama_memory_recurrent_context', 'n_embd_s()'],
       ['（对照）K/V', 'mctx->get_k / get_v', 'llama_kv_cache_context', 'n_ctx —— 随上下文增长']],
      { monoCols: [1, 3] });
    t.el.style.fontSize = '9.5px';
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先记住两句话：<span class="k">KV cache 的大小是 O(n_ctx)，recurrent state 的大小是 O(1)</span>。',
      '<span class="v">get_r_l(il)</span> 给的是<b>卷积状态</b>：只有 <span class="v">d_conv - 1</span> 步的窗口。',
      '<span class="v">get_s_l(il)</span> 给的是<b>SSM 状态</b>：整段历史的压缩表示，'
        + '形状 <span class="m">{d_state, head_dim, n_head, n_seqs+}</span> —— <span class="k">没有 token 维</span>。',
      '<span class="v">build_rs()</span> 把"缓存的若干行"变成"这一批要用的状态"。'
        + '它不复制状态本身，只是按 <span class="v">ids</span> 选行 —— 这就是 L2-04 里 slot 机制的图侧用法。',
      '所以 SSM 模型能跑很长的上下文而显存不涨；代价是<span class="k">历史被有损压缩</span>，'
        + '这正是"线性注意力"这个名字的来历。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
    tl.at(4500, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 0 ? 'on' : ''); });
    tl.at(9000, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = i === 1 ? 'on' : ''); });
    tl.at(14000, () => { msg.innerHTML = texts[3]; rows.forEach(r => r.className = ''); });
    tl.at(19000, () => { msg.innerHTML = texts[4]; rows.forEach((r, i) => r.className = i === 2 ? 'on' : ''); });
  }
},

/* ------------------------------------------------------ 3 <span class="hl-b">卷积状态</span>：一个 (d_conv − 1) 长的滑动窗口 */
{
  kicker: "L2-11 · 图原语 · 卷积",
  title: "<span class=\"hl-b\">卷积状态</span>：一个 (d_conv − 1) 长的滑动窗口",
  sub: "这是整条链上唯一保留\"原始 token 片段\"的地方，长度是常数，不是上下文长度。",
  caption: "注意 217-220 行：写回缓存用的不是\"最后一个位置\"，而是 n_seq_tokens - slot 处的快照 —— K 个槽位分别对应\"最终状态\"和\"回退 s 个 token 的状态\"（L2-04）。",
  src: "src/models/mamba-base.cpp",
  mark: [3, 14, 30, 34, 37],
  lineNo: 204,
  code: `    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner + 2*n_group*d_state, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);
//>> ggml_concat：把缓存的 d_conv-1 列拼到本轮输入前面 —— 状态与输入在这里拼成一条序列

        const int64_t row_count = (d_conv - 1) * (d_inner + 2 * n_group * d_state);
        const size_t  row_size  = ggml_row_size(conv_states_all->type, row_count);
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

        for (int64_t slot = 0; slot < n_written; ++slot) {
            ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs,
                                                   conv_x->nb[1], conv_x->nb[2], (n_seq_tokens - slot) * conv_x->nb[0]);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv,
//>> 把 conv_x 的尾部快照 cpy 回 conv_states_all（下一幕会看到同样的写法用于 SSM 状态）
                                                   ggml_view_2d(ctx0, conv_states_all, row_count, n_seqs,
                                                                conv_states_all->nb[1],
                                                                ((size_t) slot * mem_size + kv_head) * row_size)));
        }

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        xBC = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);
//>> ggml_ssm_conv：一个算子完成 1D 卷积（源码 224-231 行解释了它等价于什么）

        // bias
        xBC = ggml_add(ctx0, xBC, model.layers[il].ssm_conv1d_b);
//>> 卷积之后接 bias 与 SiLU，与普通卷积层的区别只在状态从哪来

        xBC = ggml_silu(ctx0, xBC);
    }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip d">conv_states_all（缓存）</span><span class="arrow">-&gt;</span>
        <span class="chip a">ggml_concat</span><span class="arrow">-&gt;</span>
        <span class="chip b">ggml_ssm_conv</span><span class="arrow">-&gt;</span>
        <span class="chip c">+bias / SiLU</span><span class="arrow">-&gt;</span>
        <span class="chip d">cpy 回缓存</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'd', t: '读', m: 'build_rs(inp, conv_states_all, n_embd_r(), n_seqs)',
        b: 'build_rs 按 ids 选出这一批序列的状态行；<br>再 reshape 成 {d_conv-1, d_inner, n_seqs}。' },
      { c: 'a', t: '拼', m: 'ggml_concat(conv, transpose(xBC), 0)',
        b: '状态在上面、本轮输入在下面，沿第 0 维拼接 —— <br>于是卷积"看得到"历史窗口。' },
      { c: 'b', t: '算', m: 'ggml_ssm_conv(ctx0, conv_x, ssm_conv1d)',
        b: '一个算子做完整条 1D 卷积。<br>权重形状 [d_conv, d_inner]。' },
      { c: 'c', t: '写', m: 'ggml_cpy(conv_x 的尾部快照, conv_states_all 的槽位)',
        b: '快照偏移是 n_seq_tokens - slot，<br>所以 K 个槽位里放的是不同回退深度的状态。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '卷积状态只有 <span class="v">d_conv - 1</span> 步 —— 它是一个定长移位寄存器，不是 cache。',
      '<span class="v">ggml_concat</span> 这一步很关键：状态与输入被拼成一条连续序列，'
        + '后面的卷积才不需要"跨图状态"。',
      '<span class="v">ggml_ssm_conv</span> 是纯函数：它只吃 conv_x 与权重，'
        + '<span class="k">不碰任何缓存</span>。状态进出全部由显式的读/写节点负责。',
      '写回用 <span class="v">ggml_cpy</span> 而不是 inplace —— 这样 L4-02 的调度器才能看清依赖顺序：'
        + '<span class="k">先读旧状态、再写新状态</span>。',
      '同样的"读-算-写"三拍，下一幕在 SSM 状态上再来一次。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(18400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 <span class="hl-c">ggml_ssm_scan</span> 的 7 个输入：形状写在注释里 */
{
  kicker: "L2-11 · 核心 · SSM",
  title: "<span class=\"hl-c\">ggml_ssm_scan</span> 的 7 个输入：形状写在注释里",
  sub: "内核开头七行就是权威的输入契约。8 个参数（含 K）分别描述状态、输入、步长、衰减与选择矩阵。",
  caption: "输出张量的构造在 ggml/src/ggml.c:5739-5751（逐字见 source.md 第三节）：长度 = nelements(x) + K * s->ne[0] * s->ne[1] * s->ne[2] * ids->ne[0]。",
  src: "ggml/src/ggml-cpu/ops.cpp",
  mark: [3, 5, 8, 10, 12, 13, 25, 29],
  lineNo: 9773,
  code: `static void ggml_compute_forward_ssm_scan_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
//>> s：状态。第 3 维是 n_seqs+ —— 比这一批的序列数多，多出来的行是缓存里别的序列
    const ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
//>> x：本轮输入（"SSM 里的 V"）。ne[1]=n_head 在 Mamba-1 里退化成 d_inner
    const ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    const ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
//>> A：衰减矩阵。[d_state, n_head] 或 [1, n_head]；后者是 Mamba-2 的"每头一个标量衰减"
    const ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
//>> B / C：选择矩阵，形状相同（源码 5708 行断言 ggml_are_same_shape(B, C)）
    const ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    const ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}
//>> ids：每个序列读状态缓存里的哪一行 —— 这就是 recurrent cache 的"slot 表"

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nc = src0->ne[0]; // d_state
    const int64_t nr = src0->ne[1]; // dim
    const int64_t nh = src1->ne[1]; // n_head
    const int64_t ng = src4->ne[1];
    const int64_t nt = src1->ne[2]; // number of tokens per sequence
    const int64_t ns = src1->ne[3]; // number of sequences in the batch
    const int64_t K  = ggml_get_op_params_i32(dst, 0);
//>> K：一个 int32 op_param，不是张量。K>1 时额外返回 K 份回退快照

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    const int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);
//>> s_off：状态在输出张量里的起始偏移 —— 输出是"y 拼上状态"的一维张量`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['参数', '形状', '在 Mamba 里是什么'],
      [['s', '{d_state, head_dim, n_head, n_seqs+}', '上一刻的状态（从缓存按 ids 选中）'],
       ['x', '{head_dim, n_head, n_seq_tokens, n_seqs}', '本轮输入，来自 ssm_conv 的输出'],
       ['dt', '{n_head, n_seq_tokens, n_seqs}', '每步的离散化步长，先 softplus 再加 bias'],
       ['A', '{d_state, n_head} 或 {1, n_head}', '衰减；Mamba-2 每头一个标量'],
       ['B', '{d_state, n_group, n_seq_tokens, n_seqs}', '选择矩阵（写）'],
       ['C', '{d_state, n_group, n_seq_tokens, n_seqs}', '选择矩阵（读）；必须与 B 同形'],
       ['ids', '{n_seqs}（I32 向量）', '每个序列对应状态缓存的哪一行'],
       ['K', 'int32 op_param（不是张量）', 'K = n_rs_seq + 1；K>1 时返回回退快照']],
      { monoCols: [0, 1] });
    t.el.style.fontSize = '9px';
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '验收点的前半：<span class="k">ssm_scan 有 7 个张量输入 + 1 个整数参数 K</span>。',
      '<span class="v">s</span> 的第 3 维是 <span class="v">n_seqs+</span>，'
        + '也就是说它<b>直接看着整份缓存</b>，靠 <span class="v">ids</span> 选行。',
      '<span class="v">dt / A</span> 一起决定衰减：内核里 <span class="m">dA = expf(softplus(dt) * A)</span>。',
      '<span class="v">B / C</span> 是"选择"的全部：同一份 x，靠 B、C 决定这一刻写进状态多少、读出多少。',
      '<span class="v">K</span> 是唯一的整数参数，用来一次多要 K 份历史快照（供回退用）。',
      '<span class="k">注意：没有任何一个输入带 n_ctx。SSM 的"上下文"就是 s 这张定长张量。</span>'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
    tl.at(4600, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 0 || i === 6 ? 'on' : ''); });
    tl.at(9600, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = i === 2 || i === 3 ? 'on' : ''); });
    tl.at(14400, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 4 || i === 5 ? 'on' : ''); });
    tl.at(19200, () => { msg.innerHTML = texts[4]; rows.forEach((r, i) => r.className = i === 7 ? 'on' : ''); });
    tl.at(22800, () => { msg.innerHTML = texts[5]; rows.forEach(r => r.className = ''); });
  }
},

/* ------------------------------------------------------ 5 ★ 状态更新发生在<span class="hl-a">内核的扫描循环里</span>，写回缓存是图上的一个 cpy */
{
  kicker: "L2-11 · 核心 · SSM",
  title: "★ 状态更新发生在<span class=\"hl-a\">内核的扫描循环里</span>，写回缓存是图上的一个 cpy",
  sub: "输入视图 → 交给 ssm_scan（拿 ids 直接读缓存）→ 输出张量里同时含 y 与新状态 → 显式 cpy 回 ssm_states_all。",
  caption: "验收点的后半：新状态由内核写在输出张量的后半段（ggml/src/ggml-cpu/ops.cpp:9819-9821 定位读写指针，9923 / 9976 处 s[i] = state），模型侧再用 274-279 行的 ggml_cpy 搬回持久缓存。",
  src: "src/models/mamba-base.cpp",
  mark: [3, 6, 9, 13, 16, 22, 29, 33, 43, 46],
  lineNo: 240,
  code: `    // ssm
    {
        // These correspond to V K Q in SSM/attention duality
        ggml_tensor * x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * xBC->nb[0],
//>> x：从卷积输出里切出 {head_dim, n_head, n_seq_tokens, n_seqs}
                                       xBC->nb[1], xBC->nb[2], 0);
        ggml_tensor * B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
//>> B、C：同一个张量 xBC 的两个 4 维视图，偏移不同 —— 不复制数据
                                       xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC));
        ggml_tensor * C = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                       xBC->nb[1], xBC->nb[2], (d_inner + n_group * d_state) * ggml_element_size(xBC));

        // {n_head, n_seq_tokens, n_seqs}
        dt = ggml_add(ctx0, ggml_cont(ctx0, dt), model.layers[il].ssm_dt_b);
//>> dt 先 cont 成连续张量，再加 dt_bias（ssm_scan 断言 dt 必须连续）

        ggml_tensor * A = model.layers[il].ssm_a;
//>> A 直接就是权重张量 ssm_a，没有额外处理

        // use the states and the indices provided by build_recurrent_state
        // (this is necessary in order to properly use the states before they are overwritten,
        //  while avoiding to make unnecessary copies of the states)
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
//>> 这个 lambda 就是 build_rs 的 get_state_rows 参数 —— 它决定"怎么把状态读出来"
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, state_slots);

            // TODO: use semistructured matrices to implement state-space duality
            // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
            // K > 1 asks the backend to return rollback snapshots in addition to the final state.
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids, K);
//>> ids 直接交给 ssm_scan：内核自己按 ids 去 s 里选行，省掉一次 get_rows 拷贝
        };

        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);
//>> build_rs 返回的就是 ssm_scan 的输出张量（y 与新状态拼在一起）
        const int64_t D            = d_state * d_inner;
        const int64_t n_written    = std::min<int64_t>(n_seq_tokens, K);
        const size_t  row_size     = ggml_row_size(ssm_states_all->type, D);
        const size_t  y_row_size   = ggml_row_size(y_ssm->type, D);
        const size_t  state_offset = ggml_nelements(x) * ggml_element_size(x);

        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0,
                         ggml_view_3d(ctx0, y_ssm, D, n_seqs, n_written,
//>> 取输出张量后半段：K 份新状态，偏移 state_offset = nelements(x) * element_size
                                      y_row_size, y_row_size * n_seqs, state_offset),
                         ggml_view_3d(ctx0, ssm_states_all, D, n_seqs, n_written,
//>> 目标视图落在 ssm_states_all 的 kv_head 行上 —— 状态就这样回到持久缓存
                                      ssm_states_all->nb[1], (size_t) mem_size * row_size, kv_head * row_size)));

        ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1], n_head * x->nb[1],
                                       n_seq_tokens * n_head * x->nb[1], 0);`,
  duration: 28000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip a">build_rs 读缓存</span><span class="arrow">-&gt;</span>
        <span class="chip b">ggml_ssm_scan</span><span class="arrow">-&gt;</span>
        <span class="chip c">y ⧺ K 份新状态</span><span class="arrow">-&gt;</span>
        <span class="chip d">ggml_cpy 回 ssm_states_all</span>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['步', '图上发生什么', '行号'],
      [['① 读', 'build_rs → reshape 状态缓存 → ssm_scan 按 ids 选行', '258-267'],
       ['② 算', '内核在 scan 循环里更新 s：s[i] = state（同时写出 y[ii]）', 'ops.cpp 9923 / 9976'],
       ['③ 出', '输出张量 = [ y（前 nelements(x) 个） | K 份新状态 ]', 'ggml.c 5739-5741'],
       ['④ 写回', 'ggml_cpy(y_ssm 的后半段 → ssm_states_all 的 kv_head 行)', '274-279'],
       ['⑤ 用 y', 'y = view_4d(y_ssm, 前半段) → + D*x → swiglu → ssm_out', '281-298']],
      { monoCols: [2] });
    t.el.style.fontSize = '9.5px';
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '四拍：<span class="k">读缓存 → 扫描 → 输出含新状态 → cpy 写回</span>。',
      '<span class="v">ggml_ssm_scan</span> 是纯算子：它<b>不</b>直接改 <span class="v">ssm_states_all</span>，'
        + '只把新状态写在<b>自己的输出张量</b>里。',
      '<span class="v">ids</span> 直接传进算子，是这里最巧的一处：内核用 '
        + '<span class="m">src0->data + ids[i3]*src0->nb[3]</span> 定位读地址 —— <span class="k">省掉一次 get_rows 拷贝</span>。',
      '写回用的是 <span class="v">ggml_cpy</span>，所以调度器能排出"先读后写"的顺序；'
        + '<span class="v">kv_head</span> 决定落在哪一行，与 L2-04 的 slot 是同一套编号。',
      '<span class="v">K = n_rs_seq + 1</span> 时输出里会有多份快照，分别落在缓存的相邻槽位 —— 这是投机解码回退的基础。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach((r, i) => r.className = i === 0 ? 'on' : ''); });
    tl.at(5200, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 1 ? 'on' : ''); });
    tl.at(10500, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
    tl.at(16000, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 2 || i === 3 ? 'on' : ''); });
    tl.at(22000, () => { msg.innerHTML = texts[4]; rows.forEach((r, i) => r.className = i === 4 ? 'on' : ''); });
  }
},

/* ------------------------------------------------------ 6 <span class="hl-d">RWKV</span>：同一套"读-算-写"，换一个融合算子 */
{
  kicker: "L2-11 · 家族 · RWKV",
  title: "<span class=\"hl-d\">RWKV</span>：同一套\"读-算-写\"，换一个融合算子",
  sub: "wkv6 与 wkv7 的差别只在算子；状态进出缓存的方式与 Mamba 完全同构。",
  caption: "对比第 5 幕：Mamba 把 ids 传进算子（内核自己选行），RWKV 用 build_rs 的默认实现 ggml_get_rows 先把状态行取出来（llama-graph.h:1333）。两条路都成立，代价不同。",
  src: "src/models/rwkv6-base.cpp",
  mark: [0, 5, 8, 11, 13, 17],
  lineNo: 133,
  code: `    ggml_tensor * wkv_state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);
//>> build_rs 不带 lambda —— 走默认的 ggml_get_rows，先取出 n_seqs 行状态

    ggml_tensor * wkv_output;
    if (is_qrwkv) {
        wkv_output = ggml_gated_linear_attn(ctx0, k, v, r, w, wkv_state, pow(head_size, -0.5f));
//>> is_qrwkv 时用 ggml_gated_linear_attn，否则用 ggml_rwkv_wkv6；两者签名不同但语义同位
    } else {
        wkv_output = ggml_rwkv_wkv6(ctx0, k, v, r, layer.time_mix_first, w, wkv_state);
//>> wkv_state 是输入，也是输出的一部分
    }
    cur       = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
//>> 前 n_embd * n_tokens 个元素是这一层的输出 y
    wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));
//>> 紧随其后的是新状态：n_embd * head_size * n_seqs 个 float

    ggml_build_forward_expand(
        gf, ggml_cpy(ctx0, wkv_state,
//>> 与 Mamba 第 5 幕同款的显式 cpy：把新状态搬回 mctx_cur->get_s_l(il)
                     ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                                  hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il)))));`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="row" style="gap:8px">
        <div class="card" style="border-left-color:var(--d);flex:1 1 0">
          <div class="ct" style="color:var(--d)">输出张量的切法</div>
          <div class="cb"><span class="cm" style="margin:0">view_1d(out, n_embd*n_tokens, 0)</span> → y<br>
          <span class="cm" style="margin:0">view_1d(out, n_embd*head_size*n_seqs, n_embd*n_tokens*4)</span> → 新状态<br>
          偏移用 <b>字节</b> 表达，不是元素个数 —— 与 Mamba 那一幕的 state_offset 同一个套路。</div>
        </div>
        <div class="card" style="border-left-color:var(--b);flex:1 1 0">
          <div class="ct" style="color:var(--b)">RWKV 还多一份状态：token shift</div>
          <div class="cb">RWKV 每层把"上一个 token 的 norm 输出"也存起来，
          由 <span class="cm" style="margin:0">build_rwkv_token_shift_load / store</span> 读写（L2-06 的原语）。<br>
          它是 <span class="cm" style="margin:0">n_embd_r()</span> 那一份卷积状态在 RWKV 里的对应物。</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['文件', '这一层的算子', '状态行号'],
      [['src/models/rwkv6-base.cpp', 'ggml_rwkv_wkv6 或 ggml_gated_linear_attn', '137 / 139'],
       ['src/models/rwkv7-base.cpp', 'ggml_rwkv_wkv7', '107'],
       ['src/models/delta-net-base.cpp', 'ggml_gated_delta_net', '402 / 567']],
      { monoCols: [0, 1, 2] });
    t.el.style.fontSize = '9.5px';
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把 Mamba 那一幕的骨架换成 RWKV，形状完全对得上：<span class="k">读状态 → 一个融合算子 → 输出含新状态 → cpy 写回</span>。',
      '<span class="v">ggml_rwkv_wkv6</span> 与 <span class="v">ggml_rwkv_wkv7</span> 的参数不同'
        + '（wkv7 多了 r/w/k/v/a 五路），但都返回"y 拼新状态"。',
      'RWKV 的 <span class="v">build_rs</span> 没传 lambda，走的是默认 '
        + '<span class="v">ggml_get_rows</span> —— 先把状态行拷出来，再交给算子。<br>'
        + 'Mamba 则把 <span class="v">ids</span> 交给算子省掉这次拷贝。',
      '因此这四个基类在图上长得几乎一样，<span class="k">差异被压缩成"用哪个融合算子 + 状态怎么取"两件事</span>。',
      '这条结论与 L2-06 一致：模型架构的差别，到原语这一层就只剩参数选择。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
    tl.at(4600, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i < 2 ? 'on' : ''); });
    tl.at(9500, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
    tl.at(14500, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 2 ? 'on' : ''); });
    tl.at(18500, () => { msg.innerHTML = texts[4]; rows.forEach(r => r.className = ''); });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-f">gated delta net</span>：状态是一整块 <span class="hl-f">矩阵</span>，不是向量 */
{
  kicker: "L2-11 · 家族 · delta net",
  title: "<span class=\"hl-f\">gated delta net</span>：状态是一整块 <span class=\"hl-f\">矩阵</span>，不是向量",
  sub: "delta net 的状态形状是 {S_v, S_v, H_v, n_seqs} —— 更像\"外积累加器\"，比 Mamba 的 {d_state, head_dim} 大得多。",
  caption: "Kimi Linear / Kimi K3 / BailingMoE3 / Qwen3-Next / Qwen3.5 的 recurrent 层都走这条路径；同一份代码用 is_recr(il) 与普通的注意力层交替。",
  src: "src/models/delta-net-base.cpp",
  mark: [10, 17, 21, 24, 32, 41, 46],
  lineNo: 527,
  code: `ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
//>> n_seq_tokens > 1 是预填充，== 1 是逐 token 解码 —— 两条路径的算子实现不同
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
//>> keep 为真时走 K>1 的分支：一次多要 K 份快照，供回退用
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;
//>> cparams.n_rs_seq 决定要不要保留回退快照

    if (!keep) {
//>> 不保留时 build_delta_net 返回 (output, new_state) 一对张量
        auto attn_out = build_delta_net(q, k, v, g, b, s, il);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
//>> 与 Mamba 同样的写回：ggml_cpy 到 ssm_states_all 的 kv_head 行
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
//>> 状态元素个数 D = S_v * S_v * H_v —— 二次方级，比 Mamba 的状态大得多
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
//>> K>1 分支：状态快照直接由算子写在输出张量里，按 n_written 个槽位切
    if (n_seq_tokens > 1) {`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:8px">
        <div class="card" style="border-left-color:var(--f);flex:1 1 0">
          <div class="ct" style="color:var(--f)">状态形状对比</div>
          <div class="cb">Mamba：<span class="cm" style="margin:0">{d_state, head_dim, n_head}</span> —— 每头一个向量<br>
          GDN：<span class="cm" style="margin:0">{S_v, S_v, H_v, n_seqs}</span> —— 每头一个方阵<br>
          KDA（Kimi）：<span class="cm" style="margin:0">head_dim x head_dim x n_head</span>，源码注释给了 128x128x32 = 524288。</div>
        </div>
        <div class="card" style="border-left-color:var(--b);flex:1 1 0">
          <div class="ct" style="color:var(--b)">三条实现路径</div>
          <div class="cb"><span class="cm" style="margin:0">build_delta_net_autoregressive</span>（n_seq_tokens == 1）<br>
          <span class="cm" style="margin:0">build_delta_net_chunking</span>（预填充，分块并行）<br>
          <span class="cm" style="margin:0">build_delta_net_fused</span>（后端有融合内核时）<br>
          由 <span class="cm" style="margin:0">cparams.fused_gdn_ar / fused_gdn_ch</span> 两个开关选。</div>
        </div>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['谁用它', '行号', '说明'],
      [['qwen3next.cpp', '543', 'gated delta net 层 + 注意力层 + MoE'],
       ['qwen35.cpp / qwen35moe.cpp', '448 / 472', '同族，MoE 版多一层专家'],
       ['kimi-linear.cpp', '342 起', 'KDA 层（有卷积状态）+ MLA 层'],
       ['bailingmoe3.cpp', '290', 'KDA 层走 build_recurrent_attn，卷积用 ggml_ssm_conv'],
       ['kimi-k3.cpp', '459', 'n_head_kv == 0 标记 recurrent 层']],
      { monoCols: [0, 1] });
    t.el.style.fontSize = '9px';
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'delta net 把"线性注意力"写成了一个递推：<span class="k">状态是一块方阵，每步做一次带门控的外积更新</span>。',
      '<span class="v">build_delta_net</span> 用 <span class="v">n_seq_tokens</span> 与两个 cparams 开关'
        + '在三条实现路径之间选一条 —— <span class="k">同一语义，算子粒度不同</span>（对照 L2-06 第 6 幕的 flash attention 快慢路）。',
      '<span class="v">K = cparams.n_rs_seq + 1</span> 与 Mamba 完全一致：这就是"投机解码要回退几步"的统一表达。',
      '这条路径被五个模型家族共用：<span class="v">qwen3next / qwen35 / qwen35moe / kimi-linear / kimi-k3 / bailingmoe3</span>。'
        + '<span class="k">一个基类撑起半个线性注意力家族。</span>',
      '回顾 L2-14 / L2-15：那些稠密 Transformer 里没有这一段 —— 它们的记忆全部在 KV cache 里。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
    tl.at(4800, () => { msg.innerHTML = texts[1]; rows.forEach(r => r.className = ''); });
    tl.at(10200, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
    tl.at(15200, () => { msg.innerHTML = texts[3]; rows.forEach(r => r.className = 'on'); });
    tl.at(20000, () => { msg.innerHTML = texts[4]; rows.forEach(r => r.className = ''); });
  }
},

/* ------------------------------------------------------ 8 15 个文件各自的<span class="hl-a">真实主题</span>（按扫描清单顺序） */
{
  kicker: "L2-11 · 清点（上）",
  title: "15 个文件各自的<span class=\"hl-a\">真实主题</span>（按扫描清单顺序）",
  sub: "逐文件读图构建代码后写下，不按文件名猜。第三列是本课的判定。",
  caption: "顺序即 plan_matrix.py --files 给出的顺序；后 15 个见下一幕。",
  src: "src/models/lfm2.cpp",
  mark: [1, 3, 33, 34],
  lineNo: 191,
  code: `        // read conv state
        auto * conv_state = mctx_cur->get_r_l(il);
//>> 读卷积状态：注意的是 get_r_l —— recurrent 那一份
        auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
//>> build_rs 取行；没有 get_ssm_rows lambda，因为 LFM2 没有 ssm_scan
        auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, hparams.n_embd, n_seqs);

        // causal prepends the state, non-causal pads symmetrically for a centered window
        if (hparams.causal_attn) {
            bx = ggml_concat(ctx0, conv, bx, 0);
        } else {
            const int64_t pad = (hparams.n_shortconv_l_cache - 1) / 2;
            auto * left = ggml_cont(ctx0,
                ggml_view_3d(ctx0, conv, pad, hparams.n_embd, n_seqs, conv->nb[1], conv->nb[2], (d_conv - pad) * conv->nb[0]));
            bx = ggml_pad_ext(ctx0, ggml_concat(ctx0, left, bx, 0), 0, pad, 0, 0, 0, 0, 0, 0);
        }
        GGML_ASSERT(bx->ne[0] > conv->ne[0]);

        // write conv states: slot 0 = the final state, slot s = the state s tokens back (partial rollback)
        const int64_t K         = hparams.causal_attn && cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
        const auto    mem_size  = mctx_cur->get_size();
        const size_t  row_size  = ggml_row_size(conv_state->type, (int64_t) d_conv * n_embd);

        for (int64_t slot = 0; slot < n_written; ++slot) {
            auto * conv_snap = ggml_view_3d(ctx0, bx, d_conv, bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2],
                                            (bx->ne[0] - d_conv - slot) * ggml_element_size(bx));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_snap,
                                                   ggml_view_2d(ctx0, conv_state, (int64_t) d_conv * n_embd, n_seqs,
                                                                conv_state->nb[1],
                                                                ((size_t) slot * mem_size + kv_head) * row_size)));
        }

        auto * conv_kernel = model.layers[il].shortconv.conv;
        auto * conv_out    = ggml_ssm_conv(ctx0, bx, conv_kernel);
//>> ggml_ssm_conv：LFM2 复用了 Mamba 的卷积算子，但只用它做短卷积
        cb(conv_out, "model.layers.{}.conv.conv", il);

        auto * y = ggml_mul(ctx0, c, conv_out);`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['文件', '真实主题（读代码得出）', 'SSM/线性注意力？'],
      [["arwkv7.cpp","RWKV-7 的变体：每层 token-shift + build_rwkv7_time_mix + channel mix","是"],["bailingmoe3.cpp","KDA 层 + MLA 层 + MoE：KDA 走 build_recurrent_attn，卷积用 ggml_ssm_conv","是（混合）"],["deepseek2.cpp","DeepSeek-V2 的 MLA + MoE：wkv_a_mqa / wkv_b 是低秩 KV 压缩投影","否"],["deepseek32.cpp","DeepSeek-V3.2 的 DSA 稀疏注意力 + MLA，走 KV cache","否"],["delta-net-base.cpp","gated delta net 公共基类：三种实现 + 卷积状态 + 状态写回","是（基类）"],["dflash.cpp","DFlash 投机解码草稿图：layer.wkv 只是一个普通 K/V 投影权重名","否"],["dots3note.cpp","MLA + DSA indexer（源码注释写明 adapted from deepseek32）","否"],["falcon-h1.cpp","混合：按 hparams.is_recr(il) 在 Mamba2 层与注意力层间切换","是（混合）"],["glm-dsa.cpp","GLM 的 DSA 稀疏注意力 + MLA，走 KV cache","否"],["granite-hybrid.cpp","混合：Mamba2 层 + 注意力层，inp->get_recr() 传状态","是（混合）"],["hy-v4.cpp","Hunyuan V4 的 MLA + DSA，走 KV cache","否"],["jamba.cpp","混合：Mamba-1 层 + 注意力层 + MoE","是（混合）"],["kimi-k3.cpp","KDA 层（n_head_kv == 0 标记）+ MLA 层","是（混合）"],["kimi-linear.cpp","KDA 层（ggml_ssm_conv + gated delta net）+ MLA 层","是（混合）"],["lfm2.cpp","短卷积块：卷积状态存 recurrent cache，用 ggml_ssm_conv；无 ssm_scan","是（只卷积）"]],
      { monoCols: [0] });
    t.el.style.fontSize = '9px';
    t.el.style.lineHeight = '1.15';
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这 15 个里，<span class="k">9 个名副其实，6 个不是</span>。',
      '<span class="v">delta-net-base.cpp</span> 是基类不是模型：它同时服务 kimi-linear / kimi-k3 / bailingmoe3 / qwen3next / qwen35。',
      '<span class="v">bailingmoe3 / kimi-k3 / kimi-linear</span> 是"KDA + MLA"的混合体：'
        + '<span class="k">同一份模型里既有 recurrent 层也有 KV cache 层</span>，由 is_recr(il) 分开。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
    tl.at(8000, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 4 ? 'on' : ''); });
    tl.at(15000, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = (i === 1 || i === 12 || i === 13) ? 'on' : ''); });
  }
},

/* ------------------------------------------------------ 9 ★ 余下 15 个，以及<span class="hl-e">8 个名实不符</span>的全部理由 */
{
  kicker: "L2-11 · 清点（下）",
  title: "★ 余下 15 个，以及<span class=\"hl-e\">8 个名实不符</span>的全部理由",
  sub: "8 个\"否\"的命中点都只是子串 wkv：MLA 的低秩 KV 压缩投影叫 wkv_a_mqa / wkv_b，dflash 里干脆只是一个普通 K/V 权重叫 wkv。",
  caption: "它们走 build_attn_inp_kv（KV cache），不调 build_rs / build_inp_mem_hybrid，也不出现任何一个 ggml_ssm_* 或 ggml_*wkv* 算子。",
  src: "src/models/deepseek2.cpp",
  mark: [0, 5, 7, 9],
  lineNo: 112,
  code: `        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);
//>> wkv_a_mqa：MLA 的"下投影"—— 把 n_embd 压到 kv_lora_rank + rope 维；与 RWKV 无关

        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
        if (is_mla) {
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
//>> wk_b / wv_b 是 MLA 的"上投影"
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);
        } else {
            layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)}, flags);
//>> 不分片的旧式写法里，这个上投影就叫 wkv_b —— 子串 wkv 就来自这里
        }`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['文件', '真实主题（读代码得出）', 'SSM/线性注意力？'],
      [["mamba2.cpp","Mamba-2 的超参与权重加载；图直接复用 llama_model_mamba::graph","是"],["mamba-base.cpp","llm_build_mamba_base：build_mamba_layer / build_mamba2_layer，ggml_ssm_scan 的唯一出处","是（基类）"],["mamba.cpp","Mamba-1/2 的图：逐层 build_mamba_layer 或 build_mamba2_layer + 残差","是"],["minicpm3.cpp","MiniCPM3 的 MLA 图，走 build_attn_inp_kv","否"],["nemotron-h.cpp","混合：Mamba2 层 + 注意力层 + MoE","是（混合）"],["plamo2.cpp","自带一份 Mamba 实现：ggml_ssm_conv + ggml_ssm_scan 内联在模型文件里","是"],["plm.cpp","PLM 的 MLA 图，走 build_attn_inp_kv","否"],["qwen35.cpp","gated delta net 层 + 注意力层混合","是（混合）"],["qwen35moe.cpp","同上 + MoE","是（混合）"],["qwen3next.cpp","gated delta net 层 + 注意力层 + MoE","是（混合）"],["rwkv6-base.cpp","llm_build_rwkv6_base：channel mix / time mix，wkv6 或 gated_linear_attn","是（基类）"],["rwkv6.cpp","RWKV-6 的图：token-shift + time mix + channel mix","是"],["rwkv6qwen2.cpp","RWKV-6 的 time mix + Qwen2 式 build_ffn（FFN_SILU / FFN_PAR）","是（混合）"],["rwkv7-base.cpp","llm_build_rwkv7_base：ggml_rwkv_wkv7 + 状态回写","是（基类）"],["rwkv7.cpp","RWKV-7 的图：token-shift + build_rwkv7_time_mix","是"]],
      { monoCols: [0] });
    t.el.style.fontSize = '9px';
    t.el.style.lineHeight = '1.15';
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '后 15 个里，<span class="k">13 个名副其实，2 个不是</span>。'
        + '加上上一幕：<span class="v">22 是 / 8 否</span>。',
      '8 个"否"的公共点：它们都在声明 <span class="v">LLM_TENSOR_ATTN_KV_A_MQA</span> / '
        + '<span class="v">LLM_TENSOR_ATTN_KV_B</span> 这类 MLA 张量，名字里带 wkv。',
      '判据不是名字，而是三件事：<span class="k">有没有 build_rs / build_inp_mem_hybrid、'
        + '有没有 ggml_ssm_* 或 ggml_*wkv* 算子、状态是不是定长张量</span>。这 8 个三条全不满足。',
      '<span class="v">dflash.cpp</span> 更彻底：它的 <span class="v">layer.wkv</span> 就是一次普通 K/V 投影'
        + '（<span class="v">LLM_TENSOR_ATTN_KV</span>），文件里连 MLA 都不是。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
    tl.at(8000, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = (i === 3 || i === 6) ? 'on' : ''); });
    tl.at(15000, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
    tl.at(21000, () => { msg.innerHTML = texts[3]; rows.forEach(r => r.className = ''); });
  }
},

/* ------------------------------------------------------ 10 一张表收束：<span class="hl-a">记忆</span>决定图的形状 */
{
  kicker: "L2-11 · 收束",
  title: "一张表收束：<span class=\"hl-a\">记忆</span>决定图的形状",
  sub: "同一层里，\"怎么记\"比\"算什么\"更能决定图长什么样 —— 这就是本课与稠密家族课的分界。",
  caption: "下一课 L2-12：MoE 家族（上），看稀疏专家怎么在图上展开；本课的混合模型（jamba / granite-hybrid / qwen3next 等）在那一课还会再出现。",
  src: "src/models/bailingmoe3.cpp",
  mark: [0, 5, 7, 10],
  lineNo: 255,
  code: `        if (hparams.is_recr(il)) {
//>> is_recr(il)：这一层是 recurrent 还是 attention，由 hparams 预先标好
            const auto * mctx_cur = inp_rs->mctx;
            const auto cache_head = mctx_cur->get_head();
            const auto mem_size = mctx_cur->get_size();
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
//>> recurrent 层：卷积状态，n_embd_r()
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
//>> build_rs 取这一批要用的状态行

            ggml_tensor * q = bailingmoe3_causal_conv1d(
//>> 然后一路 build_recurrent_attn —— 与纯 Mamba 走的是同一条路`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', 'KV cache 路线', 'recurrent 路线（本课）', '在哪一课'],
      [['记忆住哪', 'mctx->get_k / get_v（张量）', 'mctx->get_r_l / get_s_l（张量）', 'L2-04'],
       ['大小', 'O(n_ctx)，随上下文增长', 'O(1)，由结构超参决定', '本课 · 第 2 幕'],
       ['读', 'build_rs → get_rows 或把 ids 交给算子', 'build_rs / build_attn_inp_kv', 'L2-06 / 本课'],
       ['写', 'cpy_k / cpy_v', '模型里的 ggml_cpy 写回缓存行', '本课 · 第 5 幕'],
       ['算', 'mul_mat + soft_max（或 flash_attn）', 'ggml_ssm_scan / wkv6 / wkv7 / gdn', '本课 · 第 4~7 幕'],
       ['层怎么选', '全部层', 'hparams.is_recr(il)', '本课 · 第 10 幕'],
       ['代表模型', 'L2-14 / L2-15 的稠密家族、MLA 家族', 'mamba / rwkv / qwen3next / kimi', '—']],
      { monoCols: [1, 2] });
    t.el.style.fontSize = '9px';
    t.el.style.lineHeight = '1.2';
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码，说出 <span class="mono">ggml_ssm_scan</span> 的输入、输出，'
      + '以及<b>状态更新究竟发生在哪一步</b>？',
      '<b>输入</b>：7 个张量 + 1 个整数。<br>'
      + '<span class="mono">s {d_state, head_dim, n_head, n_seqs+}</span>（旧状态，'
      + '用 <span class="mono">ids {n_seqs}</span> 选行）、'
      + '<span class="mono">x {head_dim, n_head, n_seq_tokens, n_seqs}</span>、'
      + '<span class="mono">dt {n_head, n_seq_tokens, n_seqs}</span>、'
      + '<span class="mono">A {1, n_head} 或 {d_state, n_head}</span>、'
      + '<span class="mono">B / C {d_state, n_group, n_seq_tokens, n_seqs}</span>，'
      + '外加整数 <span class="mono">K</span>（op_param，不是张量）。<br>'
      + '<b>输出</b>：一个一维 F32 张量，长度 = '
      + '<span class="mono">nelements(x) + K * s->ne[0] * s->ne[1] * s->ne[2] * ids->ne[0]</span>；'
      + '前半段是 <span class="mono">y</span>，后半段是 <span class="mono">K</span> 份新状态'
      + '（<span class="mono">ggml/src/ggml.c:5739-5741</span>）。<br>'
      + '<b>状态更新发生在内核的扫描循环里</b>：'
      + '<span class="mono">ggml/src/ggml-cpu/ops.cpp:9819-9821</span> 用 '
      + '<span class="mono">ids[i3]</span> 定位旧状态读指针 <span class="mono">s0</span>，'
      + '并把新状态写指针 <span class="mono">s</span> 指向<b>算子自己的输出张量</b>后半段'
      + '（<span class="mono">s_off = nelements(x) * element_size</span>）；'
      + '循环体里 <span class="mono">s[i] = state</span>（9923 / 9976 行）就是更新本身。<br>'
      + '把新状态搬回持久缓存是<b>图上的另一个节点</b>：'
      + '<span class="mono">src/models/mamba-base.cpp:274-279</span> 的 '
      + '<span class="mono">ggml_cpy(view_3d(y_ssm, ..., state_offset), '
      + 'view_3d(ssm_states_all, ..., kv_head * row_size))</span>。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '把整课压成一张表：<span class="k">"怎么记"决定图的形状</span>。',
      '读的入口是 <span class="v">build_rs</span> / <span class="v">build_attn_inp_kv</span>，'
        + '两者都在 L2-06 那一族原语里 —— <span class="k">recurrent 不是旁路，是并列的一条</span>。',
      '写的入口永远是模型文件里一个显式的 <span class="v">ggml_cpy</span>：'
        + '<span class="k">算子不改缓存，只产出新状态</span>。',
      '混合模型靠 <span class="v">hparams.is_recr(il)</span> 在两种层之间切换 —— '
        + 'jamba / granite-hybrid / nemotron-h / falcon-h1 / lfm2 / qwen3next / qwen35 / kimi 全是这个套路。',
      '下一课 L2-12 讲 MoE：那是"怎么算"的另一半，与本课的"怎么记"正交。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
    tl.at(4200, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = (i === 2 || i === 3) ? 'on' : ''); });
    tl.at(8600, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = i === 3 ? 'on' : ''); });
    tl.at(13000, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 5 ? 'on' : ''); });
    tl.at(16500, () => { msg.innerHTML = texts[4]; rows.forEach(r => r.className = ''); });
  }
},

];
