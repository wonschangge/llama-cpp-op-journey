/* ==========================================================================
   L2-07 · 上下文与解码 llama-context
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 ★ <span class="hl-a">llama_decode</span>：图的生命周期管理器 */
{
  kicker: "L2-07 · 全局",
  title: "★ <span class=\"hl-a\">llama_decode</span>：图的生命周期管理器",
  sub: "建图、分配、执行、复用 —— 四件事都在 decode 里闭环。先看 llama_context 类自己的注释。",
  caption: "本课覆盖 src/llama-context.cpp 与 src/llama-context.h 两个文件，二者都计入覆盖率。",
  src: "src/llama-context.h",
  mark: [0, 1, 15],
  lineNo: 43,
  code: `struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
//>> 构造函数注释就把职责写清楚了：init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();
//>> sched_reserve() 是"按需重建调度器 + 重新预留"，不是每 token 都做`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">llama_decode</span><span class="arrow">-></span>
        <span class="chip b">decode</span><span class="arrow">-></span>
        <span class="chip c">process_ubatch</span><span class="arrow">-></span>
        <span class="chip d">graph_compute</span><span class="arrow">-></span>
        <span class="chip f">sched_graph_compute_async</span>
      </div>
      <div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '建图', b: 'model.build_graph(gparams)<br>按 ubatch 的拓扑建一张图', m: 'llama-context.cpp:1425', w: '163px' },
      { c: 'b', t: '分配', b: 'ggml_backend_sched_alloc_graph<br>把节点放进各后端的计算缓冲', m: 'llama-context.cpp:1435', w: '163px' },
      { c: 'c', t: '执行', b: 'graph_compute 取线程池，<br>再交给调度器异步提交', m: 'llama-context.cpp:1454 / 2588', w: '163px' },
      { c: 'd', t: '复用', b: '能复用时跳过建图与分配，<br>只重新填输入', m: 'llama-context.cpp:1405', w: '163px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个 <span class="k">llama_batch</span> 进来，出去的是"若干张已经在后端上算完的图"。',
      '<span class="k">建图</span>：<span class="v">model.build_graph(gparams)</span> —— 图长什么样属于 L2-06。',
      '<span class="k">分配</span>：<span class="v">ggml_backend_sched_alloc_graph</span> —— 内存规划属于 L4-01 / L4-02。',
      '<span class="k">执行</span>：<span class="v">graph_compute</span> 里那句 <span class="v">_async</span> 才是真正把图交出去的地方（L4-04）。',
      '记住这四个词：<span class="v">建图 / 分配 / 执行 / 复用</span>。后面每一幕对应其中一段。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 C API <span class="hl-b">llama_decode</span> 只是转发 */
{
  kicker: "L2-07 · 入口",
  title: "C API <span class=\"hl-b\">llama_decode</span> 只是转发",
  sub: "真正的逻辑在 llama_context::decode（1704 行）。C API 只负责打日志与把返回值传出去。",
  caption: "注意 4330 行的判据：返回 1 不算错误 —— 它是\"这个 batch 暂时没位置\"的软失败。",
  src: "src/llama-context.cpp",
  mark: [3],
  lineNo: 4326,
  code: `int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\\n", __func__, ret);
    }

    return ret;
}`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: '0', b: '成功。<br>图已提交（异步时可能仍在跑）', m: 'llama-context.cpp:2096', w: '163px' },
      { c: 'c', t: '1', b: 'memory 找不到 slot。<br>软失败，llama_decode 不为它报错', m: 'llama-context.cpp:1839', w: '163px' },
      { c: 'g', t: '-1', b: '输入非法：n_tokens 为 0、<br>batch 初始化失败、pooled 要求全输出', m: 'llama-context.cpp:1716 / 1767', w: '163px' },
      { c: 'd', t: '-2', b: '分配失败：memory context、<br>输出缓冲或图分配失败', m: 'llama-context.cpp:1855 / 1914', w: '163px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '返回值不是随意取的 —— 它在 <span class="k">1704 起的 decode()</span> 里逐条 <span class="v">return</span> 出来。',
      '<span class="v">0</span> 表示这一批成功。<span class="v">1</span> 表示"这次挤不下"，调用方可以稍后重试。',
      '<span class="v">-1</span> 是<b>调用方写错了</b>：空 batch、batch 结构不对、embedding 模式下没有全输出。',
      '<span class="v">-2</span> 是<b>资源不够</b>：拿不到 memory context、预留不出输出缓冲、图分配失败。',
      '还有两个来自 ubatch 执行失败的分支：<span class="v">2</span> = GGML_STATUS_ABORTED，<span class="v">-3</span> = GGML_STATUS_FAILED（1913-1915）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(12300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 <span class="hl-c">balloc</span>：batch 进，ubatch 的准备开始 */
{
  kicker: "L2-07 · 第 1 步",
  title: "<span class=\"hl-c\">balloc</span>：batch 进，ubatch 的准备开始",
  sub: "decode 的第一件事是把用户给的 llama_batch 校验并归一化；切分本身由 memory 模块接手。",
  caption: "回顾 L2-05：ubatch 是切图的依据 —— 同一个 ubatch 拓扑不变，所以图才能复用。",
  src: "src/llama-context.cpp",
  mark: [0, 6, 8, 20, 22],
  lineNo: 1765,
  code: `    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
//>> balloc 是 context 的成员（llama-context.h:334），复用同一个对象避免反复分配
        LLAMA_LOG_ERROR("%s: failed to initialize batch\\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
//>> n_tokens_all：本逻辑批的总 token 数
    const uint32_t n_outputs_all = balloc->get_n_outputs();
//>> n_outputs_all：需要输出 logits/embd 的行数，后面 output_reserve 用它

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'llama_batch_allocr', b: '把 llama_batch 归一化成<br>统一内部表示，并切出 ubatch', m: 'llama-context.h:334', w: '220px' },
      { c: 'b', t: 'n_tokens_all', b: '这一批一共多少 token。<br>下面用两条断言守住上限', m: 'llama-context.cpp:1770', w: '220px' },
      { c: 'c', t: 'n_outputs_all', b: '要输出多少行。<br>output_reserve 按它预备缓冲', m: 'llama-context.cpp:1771', w: '220px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看清楚：<span class="k">ubatch 不是这里切出来的</span>，这里只做校验与归一化。',
      '<span class="v">balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)</span><br>' +
      '失败就直接 <span class="v">return -1</span> —— 这是 decode 的第一道闸门。',
      '<span class="v">n_tokens_all</span> 与 <span class="v">n_outputs_all</span> 是后面所有循环的分母：<br>输出缓冲大小、进度累计都按它们算。',
      '两条 <span class="k">GGML_ASSERT</span>（1782、1784）是硬约束：<br>一批不能超过 <span class="v">n_batch</span>；非因果注意力必须一次放得下 <span class="v">n_ubatch</span>。',
      '真正的切分在下一幕：<span class="v">memory->init_batch(*balloc, cparams.n_ubatch, output_all)</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13400, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-a">sched_reserve</span>：最坏情况的图，只预留一次 */
{
  kicker: "L2-07 · 第 2 步",
  title: "★ <span class=\"hl-a\">sched_reserve</span>：最坏情况的图，只预留一次",
  sub: "它有一个开关 sched_need_reserve：第一次进来做完就关掉，之后每次 decode 直接返回。",
  caption: "这正是\"每 token 开销\"能被压下来的原因之一：max_nodes 与计算缓冲只规划一次。",
  src: "src/llama-context.cpp",
  mark: [1, 6, 16, 19],
  lineNo: 617,
  code: `void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
//>> 开关：不需要预留就直接 return —— 这是常态路径
        return;
    }

    sched_need_reserve = false;
//>> 置 false：做完这一次就关掉，除非有东西把它重新点亮

    LLAMA_LOG_INFO("%s: reserving ...\\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
//>> 按最坏情况取：n_tokens = min(n_ctx, n_ubatch)

    const size_t max_nodes = this->graph_max_nodes(n_tokens);
//>> max_nodes 决定调度器与图元数据 arena 的上限

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\\n", __func__, max_nodes);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '开关', b: 'sched_need_reserve<br>false 就整段跳过', m: 'llama-context.h:350', w: '163px' },
      { c: 'b', t: '最坏规模', b: 'n_tokens = min(n_ctx, n_ubatch)<br>n_seqs = n_seq_max', m: 'llama-context.cpp:630 - 631', w: '163px' },
      { c: 'c', t: '节点上限', b: 'max_nodes =<br>graph_max_nodes(n_tokens)', m: 'llama-context.cpp:633', w: '163px' },
      { c: 'd', t: '重建调度器', b: 'ggml_backend_sched_new<br>按 max_nodes 建', m: 'llama-context.cpp:643', w: '163px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">sched_reserve()</span> 在 decode 的第 1800 行被调用，但它<b>大多数时候什么都不做</b>。',
      '看 <span class="k">618</span> 行：<span class="v">if (!sched_need_reserve) return;</span><br>' +
      '也就是说这个函数是"幂等的一次性预留"，不是每 token 的固定开销。',
      '需要预留时按<b>最坏情况</b>来：<span class="v">n_tokens = min(n_ctx, n_ubatch)</span>，<br>序列数取 <span class="v">n_seq_max</span>，节点上限交给 <span class="v">graph_max_nodes()</span>。',
      '然后按 max_nodes 重建调度器与 gf_res_reserve —— <span class="k">调度器的容量在这里被钉死</span>（细节见 L4-02）。',
      '谁会把它重新点亮？<span class="v">set_sampler</span>（1289）、<span class="v">set_adapters_lora</span>（1347）、' +
      '<span class="v">set_causal_attn</span>（1259）、<span class="v">set_adapter_cvec</span>（1386）等；<br>' +
      '换句话说：<span class="k">图形态可能要变的时候</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 <span class="hl-d">memory->init_batch</span>：ubatch 的产地 */
{
  kicker: "L2-07 · 第 3 步",
  title: "<span class=\"hl-d\">memory->init_batch</span>：ubatch 的产地",
  sub: "切分这件事不在 context 里做，而是交给 memory 模块；context 只拿到一个 mctx 再问它要 ubatch。",
  caption: "回顾 L2-04：memory 家族（KV cache / recurrent / hybrid）各自决定怎么切。",
  src: "src/llama-context.cpp",
  mark: [0, 5, 10, 11],
  lineNo: 1800,
  code: `    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);
//>> 处理悬空的 shift/copy —— KV cache 的搬运在这里落地

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
//>> n_ubatch 是上限：memory 可以切得更小
        if (!mctx) {
            return -2;
        }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">llama_batch_allocr</span><span class="arrow">+</span>
        <span class="chip b">llama_memory_i</span><span class="arrow">-></span>
        <span class="chip c">memory->init_batch</span><span class="arrow">-></span>
        <span class="chip d">llama_memory_context_i</span><span class="arrow">-></span>
        <span class="chip f">mctx->get_ubatch()</span>
      </div>
      <div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: 'memory_update(false)', b: '先把待处理的 shift/copy 做掉，<br>并让 memory 尝试优化一次', m: 'llama-context.cpp:1805', w: '220px' },
      { c: 'c', t: 'while (true) 重试', b: 'FAILED_PREPARE 时用<br>memory_update(true) 优化后重来一次', m: 'llama-context.cpp:1809 / 1830', w: '220px' },
      { c: 'd', t: 'mctx 给的是 ubatch 序列', b: 'do { get_ubatch() } while (mctx->next())<br>一个逻辑批可能拆成多个 ubatch', m: 'llama-context.cpp:1867 / 2041', w: '220px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'decode 不自己切 batch：它把 <span class="k">balloc</span>、<span class="k">memory</span>、<span class="k">n_ubatch</span> 一起交给 memory 模块。',
      '第 <span class="v">1805</span> 行的 <span class="v">memory_update(false)</span> 先处理悬空搬运 —— 这是 L2-04 里 KV cache 的 shift/copy。',
      '拿到 <span class="v">mctx</span> 之后，真正的循环是 <span class="k">do / while (mctx->next())</span>：<br>一个逻辑批可能被拆成好几个 ubatch，每个走一次 process_ubatch。',
      '如果 memory 报 <span class="v">FAILED_PREPARE</span>，decode 会先尝试 <span class="v">memory_update(true)</span> 优化一次，<br>再重试；还不行就 <span class="v">return 1</span>（软失败）。',
      '所以"切图依据"这条链是：<span class="v">n_ubatch</span>（上限）→ memory 的策略 → <span class="v">mctx</span> → 每个 ubatch 一张图。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 ★ <span class="hl-b">process_ubatch</span>：复用还是重建，一个 if 决定 */
{
  kicker: "L2-07 · 第 4 步",
  title: "★ <span class=\"hl-b\">process_ubatch</span>：复用还是重建，一个 if 决定",
  sub: "注释写得很直白：想复用它，图的完整拓扑必须由这些参数唯一确定。",
  caption: "can_reuse 的实现属于 L2-06 的 llm_graph_result（src/llama-graph.{h,cpp}），本课只看调用点与判据。",
  src: "src/llama-context.cpp",
  mark: [5, 8, 16, 19, 23, 25],
  lineNo: 1398,
  code: `    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);
//>> gparams 里装着 ubatch、mctx、gtype、sched —— 复用判据就是比较它

    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
//>> 三个条件同时成立才复用：开关没关、还是同一个 arena、拓扑参数一致
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
//>> 复用命中时唯一要做的事：记一笔账
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: '复用路径（命中）', b: '不建图、不分配。<br>只 ggml_backend_sched_synchronize（仅流水线并行时）<br>然后 n_reused++', m: 'llama-context.cpp:1412 / 1415', w: '220px' },
      { c: 'g', t: '重建路径（未命中）', b: 'gf_res_prev_active = nullptr<br>res->reset()<br>ggml_backend_sched_reset(sched)<br>再建图 + 分配', m: 'llama-context.cpp:1417 - 1420', w: '220px' },
      { c: 'd', t: '两个复用槽', b: 'gf_res_prev[n_outputs > 0]<br>有输出 / 无输出各一个 arena，<br>避免两类图互相顶掉', m: 'llama-context.cpp:2423', w: '220px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">process_ubatch</span> 是整个 decode 里最值得记住的函数：它是<b>图的入口</b>。',
      '先拿 <span class="k">res = get_gf_res_prev()</span>（1398），再算 <span class="k">gparams = graph_params(...)</span>（1403）—— ' +
      '参数里打包了 ubatch、mctx、gtype、sched。',
      '然后就是那个 <span class="k">if</span>（1405）：<span class="v">!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)</span>。<br>' +
      '三个条件缺一不可 —— 而判据本身写在 <span class="v">llm_graph_params::allow_reuse</span> 里（L2-06 讲过）。',
      '命中复用时，这一轮 decode 的开销只剩：<span class="k">同步（仅流水线并行）+ 重填输入 + 提交计算</span>。',
      '没命中就走 else：重置 arena、重置调度器、重建图、重新分配（下一幕）。<br>可用环境变量 <span class="v">LLAMA_GRAPH_REUSE_DISABLE=1</span> 关掉复用（280-281 行）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 重建路径：<span class="hl-c">建图</span> → <span class="hl-a">分配</span> */
{
  kicker: "L2-07 · 第 5 步",
  title: "重建路径：<span class=\"hl-c\">建图</span> → <span class=\"hl-a\">分配</span>",
  sub: "只有这一条 else 分支里才会出现 build_graph 与 sched_alloc_graph —— 复用命中时它们根本不执行。",
  caption: "回顾 L2-06：build_graph 返回的就是 llm_graph_context 拼出来的 ggml_cgraph。",
  src: "src/llama-context.cpp",
  mark: [0, 5, 16, 23],
  lineNo: 1421,
  code: `        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
//>> 先清掉上一张图的分配状态，并重新挂上 eval 回调

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);
//>> 图从这里来 —— 拓扑由 gparams 唯一确定

        //LLAMA_LOG_INFO("graph build time: %.3f ms\\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
//>> 分配：调度器把每个节点落到某个后端的计算缓冲里（L4-01 / L4-02）
            LLAMA_LOG_ERROR("%s: failed to allocate graph\\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
//>> 只有成功建图并分配之后，才把这个 arena 标成"可复用"
    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '1. 清场', b: 'res->reset() +<br>ggml_backend_sched_reset(sched)', m: 'llama-context.cpp:1418 / 1420', w: '163px' },
      { c: 'b', t: '2. 建图', b: 'model.build_graph(gparams)<br>返回 ggml_cgraph *', m: 'llama-context.cpp:1425', w: '163px' },
      { c: 'c', t: '3. 分配', b: 'ggml_backend_sched_alloc_graph<br>失败即 GGML_STATUS_ALLOC_FAILED', m: 'llama-context.cpp:1435', w: '163px' },
      { c: 'd', t: '4. 记账', b: 'gf_res_prev_active = res<br>下次 can_reuse 才有机会命中', m: 'llama-context.cpp:1441', w: '163px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这四步是"重建一张图"的全部动作，顺序不能换。',
      '<span class="v">ggml_backend_sched_reset()</span> 会清掉调度器里上一张图的切分与分配信息 —— ' +
      '所以 <span class="k">reset 之后旧图一定不能复用</span>（graph_reserve 里 2499 行的注释就是这么说的）。',
      '<span class="v">model.build_graph(gparams)</span>：这一步的产物是 <span class="k">ggml_cgraph *</span>，' +
      '内容完全由 gparams 决定 —— 这就是"拓扑可复用"的前提。',
      '<span class="v">ggml_backend_sched_alloc_graph()</span> 做的是内存规划：节点算出的张量住哪个 buffer、' +
      '哪些可以复用同一块（L4-01 讲分配器，L4-02 讲切分）。',
      '最后 <span class="v">gf_res_prev_active = res</span>：把当前 arena 标记为"这张图有效"，' +
      '下一次 process_ubatch 才有机会走进复用分支。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 喂输入：<span class="hl-e">res->set_inputs</span> → <span class="hl-d">graph_compute</span> */
{
  kicker: "L2-07 · 第 6 步",
  title: "喂输入：<span class=\"hl-e\">res->set_inputs</span> → <span class=\"hl-d\">graph_compute</span>",
  sub: "两条路径（复用 / 重建）在这里汇合：图已经就绪，接下来只需要把这一轮的 ubatch 填进去。",
  caption: "第二个参数 batched 决定用哪套线程数：ubatch.n_tokens > 1 就是 prompt 处理（L4-04）。",
  src: "src/llama-context.cpp",
  mark: [5, 11],
  lineNo: 1444,
  code: `    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);
//>> 复用路径下这一步是唯一真正"更新"图内容的动作

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
//>> 返回值直接就是 ggml_status，失败时 process_ubatch 返回 nullptr`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'e', t: 'res->set_inputs(&ubatch)', b: '把这一轮的 ubatch 填进<br>图的所有输入对象', m: 'llama-context.cpp:1449', w: '220px' },
      { c: 'd', t: 'graph_compute(gf, batched)', b: 'batched = ubatch.n_tokens > 1<br>决定用 n_threads_batch 还是 n_threads', m: 'llama-context.cpp:1454', w: '220px' },
      { c: 'a', t: '两条路径在此汇合', b: '复用与重建的区别到此为止，<br>后面完全一样', m: 'llama-context.cpp:1442 - 1454', w: '220px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '注意 <span class="k">if / else</span> 在 1442 行就结束了 —— 从 1444 行开始，两条路径走的是同一段代码。',
      '<span class="v">res->set_inputs(&ubatch)</span> 遍历图的所有输入对象，把这一轮的数据拷进去；<br>' +
      '上一幕的 <span class="v">can_reuse</span> 检查的就是"这些输入张量还能不能装下新数据"。',
      '<span class="v">graph_compute(res->get_gf(), ubatch.n_tokens > 1)</span><br>用 <span class="v">res->get_gf()</span> 而不是局部变量 <span class="v">gf</span> —— ' +
      '因为复用路径下局部 <span class="v">gf</span> 还是 <span class="v">nullptr</span>。',
      '<span class="v">batched</span> 这个布尔量一边选线程池，一边选线程数：<br>prefill 用 batch 线程，逐 token 用普通线程（细节见 L4-04）。',
      '一句话：<span class="k">图的身份在这一步之前定下来，图的内容在这一步被填进去</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(12300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 终点：<span class="hl-f">ggml_backend_sched_graph_compute_async</span> */
{
  kicker: "L2-07 · 第 7 步",
  title: "终点：<span class=\"hl-f\">ggml_backend_sched_graph_compute_async</span>",
  sub: "graph_compute 自己不做计算：它只选线程池/线程数，然后把图整张交给调度器。",
  caption: "在 llama-context.cpp 里，与\"提交整图计算\"相关的 ggml_backend_sched_ 调用只有 2588 这一处（且是 _async）。",
  src: "src/llama-context.cpp",
  mark: [3, 20, 22],
  lineNo: 2569,
  code: `ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
//>> batched 在这里变成具体线程数：n_threads_batch 或 n_threads
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
//>> 异步提交：函数返回时计算可能还在跑
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '1. 先设 CPU 线程池', b: '把 threadpool / threadpool_batch<br>塞给 CPU 后端', m: 'llama-context.cpp:2575 - 2581', w: '163px' },
      { c: 'b', t: '2. 再通知所有后端', b: '遍历 set_n_threads_fns，<br>逐个后端设置线程数', m: 'llama-context.cpp:2584', w: '163px' },
      { c: 'c', t: '3. 提交整图', b: 'ggml_backend_sched_graph_compute_async<br>(sched, gf)', m: 'llama-context.cpp:2588', w: '163px' },
      { c: 'd', t: '4. 返回状态', b: 'ggml_status 一路回到 process_ubatch，<br>失败即整体失败', m: 'llama-context.cpp:2595', w: '163px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这就是验收点里那句"到 ggml_backend_sched_graph_compute 为止"的真实落点：<b>函数名带 _async</b>。',
      '<span class="v">batched</span> 在这里生效：<span class="v">n_threads</span> 与 <span class="v">tp</span> 两个局部变量都靠它三目选择。',
      '线程池只对 CPU 后端有意义，所以先做一次<b>进程地址查询</b>拿到 <span class="v">ggml_backend_cpu_set_threadpool</span>；<br>' +
      '拿不到就跳过 —— 这解释了为什么它是 <span class="v">if (set_threadpool_fn)</span>。',
      '其余后端（GPU 等）走 <span class="v">set_n_threads_fns</span>：context 构造时就把每个后端"设线数的函数"收集好了（366-368 行）。',
      '交给调度器之后的事 —— 切分成几个 split、插哪些 copy 节点、什么时候同步 —— 是 L4-02 与 L4-04。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14300, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 一趟 decode 的结尾，与整条调用链 */
{
  kicker: "L2-07 · 收束",
  title: "一趟 decode 的结尾，与整条调用链",
  sub: "循环收尾后把 n_outputs 归位、建立 output_ids 映射，用户才能用 llama_get_logits_ith 取数。",
  caption: "下一课 L2-08 采样器与词表：logits 拿到手之后，怎么变成下一个 token。",
  src: "src/llama-context.cpp",
  mark: [2, 6, 19],
  lineNo: 2039,
  code: `        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());
//>> mctx->next() 为假时退出：一个逻辑批的全部 ubatch 都算完了

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;
//>> 把 n_outputs 从"本 ubatch 的输出数"改回"整批的输出数"

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['步', '在 llama-context.cpp 的哪一步', '做什么'],
      [['1', 'llama_context::decode · 1704', '入口：校验 + 主循环'],
       ['2', 'balloc->init · 1765', '归一化 batch，取 n_tokens_all'],
       ['3', 'sched_reserve · 1800', '按需预留最坏情况图（通常直接返回）'],
       ['4', 'memory->init_batch · 1810', '拿到 mctx，ubatch 的产地'],
       ['5', 'process_ubatch · 1887', '每个 ubatch 一次'],
       ['6', 'model.build_graph · 1425', '建图（复用命中时跳过）'],
       ['7', 'ggml_backend_sched_alloc_graph · 1435', '分配计算缓冲（同上）'],
       ['8', 'res->set_inputs · 1449', '把 ubatch 填进输入张量'],
       ['9', 'graph_compute · 1454 → 2569', '选线程池与线程数'],
       ['10', 'ggml_backend_sched_graph_compute_async · 2588', '提交整图计算']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '从 <span class="mono">process_ubatch</span> 收到一个 ubatch 开始，到调度器真正接手计算，' +
      '按顺序说出经过的每一步（函数名 + 大致行为）。',
      '1. <span class="mono">get_gf_res_prev()</span>（1398）取这个 arena 里上一次的图结果；<br>' +
      '2. <span class="mono">graph_params(...)</span>（1403）打包 ubatch / mctx / gtype / sched；<br>' +
      '3. 判断 <span class="mono">res->can_reuse(gparams)</span>（1405）：<br>' +
      '&nbsp;&nbsp;&nbsp;命中 → 只做 <span class="mono">n_reused++</span>（1415），跳到第 8 步；<br>' +
      '&nbsp;&nbsp;&nbsp;未命中 → <span class="mono">res->reset()</span>（1418）+ <span class="mono">ggml_backend_sched_reset()</span>（1420）；<br>' +
      '4. <span class="mono">model.build_graph(gparams)</span>（1425）建图；<br>' +
      '5. <span class="mono">ggml_backend_sched_alloc_graph(sched, gf)</span>（1435）分配；<br>' +
      '6. <span class="mono">gf_res_prev_active = res</span>（1441）标记可复用；<br>' +
      '7. <span class="mono">res->set_inputs(&ubatch)</span>（1449）填输入；<br>' +
      '8. <span class="mono">graph_compute(gf, batched)</span>（1454）→ 进入 2569 行的实现；<br>' +
      '9. <span class="mono">ggml_backend_sched_graph_compute_async(sched, gf)</span>（2588）—— 计算在这里才真正被提交。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '先把链走一遍：<span class="v">decode → balloc → sched_reserve → init_batch → process_ubatch → build_graph → alloc_graph → set_inputs → compute</span>。',
      '<span class="k">前四步每"逻辑批"一次</span>：校验、预留、切分（第 2-5 行）。',
      '<span class="k">后五步每"ubatch"一次</span>：建图、分配、填输入、提交计算（第 6-10 行）。',
      '<span class="v">建图与分配只在复用未命中时发生</span> —— 这就是逐 token 生成时每步开销的大头被省掉的地方。',
      '回到出口：<span class="v">n_outputs = n_outputs_all</span>（2044）与 <span class="v">output_ids[out_id] = i</span>（2056），' +
      '把"批里的第几个 token"翻译成"输出缓冲的第几行"。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 1900, () => {
      rows.forEach(x => { x.className = ''; });
      r.className = 'on';
      msg.innerHTML = texts[i < 2 ? 1 : (i < 9 ? 2 : 3)];
    }));
    tl.at(22600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
