/* ==========================================================================
   L8-02 · ★ 端到端：offload 决策实战
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 <span class="hl-c">n_gpu_layers</span> 到 <span class="hl-a">split</span>：一条五站的决策链 */
{
  kicker: "L8 · 端到端",
  title: "<span class=\"hl-c\">n_gpu_layers</span> 到 <span class=\"hl-a\">split</span>：一条五站的决策链",
  sub: "用户只给了一个整数，图却被改了形状。这条链上有两次完全不同的决策。",
  caption: "回顾 L8-01：那一课跟的是一个 mul_mat 从头到尾；本课跟的是一个整数从头到尾。回顾 L4-02：调度器怎么切图已经在那一课讲过，这里只问\"它的输入从哪来\"。",
  src: "src/llama-model.cpp",
  mark: [0, 1],
  lineNo: 1521,
  code: `    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip c">整数 n_gpu_layers</span><span class="arrow">-></span>
        <span class="chip a">i_gpu_start</span><span class="arrow">-></span>
        <span class="chip a">dev_layer[il]</span><span class="arrow">-></span>
        <span class="chip b">权重 buffer</span><span class="arrow">-></span>
        <span class="chip d">节点归属</span><span class="arrow">-></span>
        <span class="chip e">split</span>
      </div>
      <div class="row center" id="c1" style="gap:8px"></div>
      <div class="formula" id="m1"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '第 1 次决策 · 加载期', m: 'llama_model_base::load_tensors', b: '按层分配设备：<br>第 1..3 站。粒度 = <b>层</b>。产出 <span class="cm" style="margin:0">dev_layer[]</span>。' },
      { c: 'd', t: '第 2 次决策 · 建图期', m: 'ggml_backend_sched_split_graph', b: '按节点分配后端：<br>第 4..5 站。粒度 = <b>节点</b>。产出 <span class="cm" style="margin:0">splits[]</span>。' },
      { c: 'b', t: '中间那一站', m: 'buffer->usage == WEIGHTS', b: '两次决策之间唯一的接口是<br><b>权重躺在哪块 buffer 里</b>。' },
      { c: 'c', t: '本课的验收点', m: 'i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)', b: '给定 <span class="cm" style="margin:0">n_gpu_layers</span>，<br>能说出哪些层上了 GPU。' }
    ];
    const host = wrap.querySelector('#c1');
    const els = defs.map(d => U.card(d, { style: 'width:163px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#m1');
    const texts = [
      '先看终点：等号左边是用户给的整数，右边是层的分界。<span class="k">一行算术，决定每层的权重去哪</span>。',
      '第 1 站到第 3 站都在 <span class="v">load_tensors()</span> 里，模型加载时一次性算完（回顾 L2-05）。',
      '第 4 站不是函数调用，而是<span class="k">一个既成事实</span>：权重张量的 <span class="v">buffer</span> 字段已经指向某块设备内存。',
      '第 5 站在建图之后发生，<span class="k">粒度换成了节点</span>。这就是本课的核心：两次决策不在同一个层次上。',
      '预测切分结果 = 先算出"哪些权重在哪"，再让调度器按它的规则读一遍。下一幕先把语义钉死。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">n_gpu_layers</span> = 最后 N 个槽位，不是前 N 层 */
{
  kicker: "L8-02 · 语义",
  title: "★ <span class=\"hl-a\">n_gpu_layers</span> = 最后 N 个槽位，不是前 N 层",
  sub: "槽位从 0 数到 n_layer_all，最后那一个槽位是输出层 —— 它也算在 N 里面。",
  caption: "常见误解：以为 --n-gpu-layers 4 是把第 0~3 层放上 GPU。源码里 il < i_gpu_start 走 CPU 分支，所以上 GPU 的是【末尾】那几层。回顾 L2-05：那里引用过同一段，本课走完整条链。",
  src: "src/llama-model.cpp",
  mark: [0, 2, 6, 9, 11, 15],
  lineNo: 1521,
  code: `    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
//>> 槽位总数是 n_layer_all + 1：重复层 0..n_layer_all-1，再加一个"输出层"槽位
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
//>> devices 为空（没有 GPU 后端）时是 0 —— 于是所有层都落到 CPU 分支
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
//>> CPU 判据：在分界之前，或超出可上 GPU 的层数
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
//>> 多卡时才用得上：按 (il - i_gpu_start) / act_gpu_layers 这个比例去 splits 里查第几张卡（L3-02）
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="formula" id="f2"></div>
      <div id="strip" style="display:flex;gap:2px;align-items:center;flex-wrap:nowrap;width:100%"></div>
      <div class="row" style="gap:8px">
        <div class="card" style="width:336px;border-left-color:var(--e)">
          <div class="ct" style="color:var(--e)">误解</div>
          <div class="cb">"前 N 层上 GPU" —— 于是以为第 0 层一定在显卡上。</div></div>
        <div class="card" style="width:336px;border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">源码事实</div>
          <div class="cb">上 GPU 的是<b>末尾</b>的槽位：<span class="cm" style="margin:0">il &lt; i_gpu_start</span> 一律 CPU。</div></div>
      </div>
      <div class="formula" id="m2"></div>`;
    root.appendChild(wrap);

    const N = 27;                       // 本课算例：n_layer_all
    const strip = wrap.querySelector('#strip');
    const cells = [];
    for (let s = 0; s <= N; s++) {
      const c = U.el('div', { style: 'width:20px;height:17px;border:1px solid var(--border);border-radius:2px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim);flex:0 0 auto' });
      c.textContent = s;
      if (s === N) c.style.marginLeft = '8px';
      strip.appendChild(c); cells.push(c);
    }
    const f2 = wrap.querySelector('#f2');
    const msg = wrap.querySelector('#m2');

    function paint(L) {
      const start = Math.max(N + 1 - L, 0);
      const act = Math.min(L, N + 1);
      cells.forEach((c, s) => {
        const on = s >= start && (s - start) < act;
        c.style.background = on ? 'rgba(63,185,80,.24)' : '#10151b';
        c.style.borderColor = on ? 'var(--b)' : 'var(--border)';
        c.style.color = on ? 'var(--b)' : 'var(--dim)';
      });
      const rep = Math.max(Math.min(L, N + 1) - 1, 0);
      f2.innerHTML = 'n_layer_all = <span class="v">27</span>　　n_gpu_layers = <span class="v">' + L +
        '</span>　　i_gpu_start = max(27 + 1 - ' + L + ', 0) = <span class="k">' + start +
        '</span>　　GPU 上的重复层 = <span class="k">' + rep + '</span>　　' +
        (start <= N ? '输出层槽位 27 -> <span class="k">GPU</span>' : '输出层槽位 27 -> <span class="m">CPU</span>');
      return { start: start, rep: rep };
    }

    const steps = [0, 1, 4, 14, 27, 28];
    const texts = [
      '先搭一把尺子：<span class="k">槽位 0..26 是 27 个重复层，槽位 27 是输出层</span>（输出层由 1546 行单独分配）。',
      '<span class="v">n_gpu_layers = 0</span>：i_gpu_start = 28，超出槽位范围 —— 连输出层也在 CPU。',
      '<span class="v">n_gpu_layers = 1</span>：i_gpu_start = 27，只有输出层槽位上 GPU，重复层 0 个（对应 1837-1841 的日志分支）。',
      '<span class="v">n_gpu_layers = 4</span>：i_gpu_start = 24，层 24/25/26 + 输出层共 4 个槽位。',
      '<span class="v">n_gpu_layers = 14</span>：i_gpu_start = 14，末尾 14 个槽位里 13 个是重复层。',
      '<span class="v">n_gpu_layers = 27</span>：i_gpu_start = 1 —— 注意 <span class="k">第 0 层仍然留在 CPU</span>。',
      '<span class="v">n_gpu_layers = 28</span>：i_gpu_start = 0，28 个槽位全上 GPU。',
      '别问"上 GPU 的最后一层是第几层"，要问"<span class="k">分界点 i_gpu_start 是几</span>" —— ' +
        '它是唯一的自由变量，其余全是从它推出来的。'
    ];
    tl.at(700, () => { paint(0); msg.innerHTML = texts[0]; });
    steps.forEach((Lv, i) => tl.at(2800 + i * 3300, () => {
      const r = paint(Lv);
      msg.innerHTML = texts[i + 1] + '　<span class="m">（i_gpu_start = ' + r.start + '，GPU 重复层 ' + r.rep + '）</span>';
    }));
    tl.at(2800 + steps.length * 3300, () => {
      paint(28);
      msg.innerHTML = texts[steps.length + 1];
    });
  }
},

/* ------------------------------------------------------ 3 层号 -> 设备 -> 权重的 buffer：<span class="hl-b">dev_layer[]</span> 是唯一的路由表 */
{
  kicker: "L8-02 · 落位",
  title: "层号 -> 设备 -> 权重的 buffer：<span class=\"hl-b\">dev_layer[]</span> 是唯一的路由表",
  sub: "每个重复层拿到一个 (设备, 候选 buffer 类型表)；输入层永远在 CPU，输出层由同一个函数算出。",
  caption: "回顾 L2-03：权重落位讲的是\"张量怎么进 buffer\"；这里讲的是\"哪个张量该进哪块 buffer\"由谁决定。",
  src: "src/llama-model.cpp",
  mark: [1, 2, 6, 7, 11, 14],
  lineNo: 0,
  code: `    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);
//>> ---- src/llama-model.cpp:1885-1890 ----
ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row center" id="r3" style="gap:8px"></div>
      <div class="formula" id="m3"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: 'dev_input', m: 'dev_input.buft_list', b: '输入层张量（token_embd）。<br>源码注释：几乎没有好处，<br>所以<b>永远留在 CPU</b>。' },
      { c: 'a', t: 'dev_layer[il]', m: 'std::vector<layer_dev>', b: '每个重复层一个 layer_dev<br>= 设备 + 候选 buffer 类型表。' },
      { c: 'b', t: 'dev_output', m: 'dev_output.buft_list', b: '输出层。由同一个<br>get_layer_buft_list(n_layer_all) 算出，<br>算法与重复层完全一致。' },
      { c: 'd', t: 'create_tensor', m: 'dev_layer.at(tn.bid).buft_list', b: '建权重张量时按层号 <b>bid</b><br>取回候选表，再从中选一种 buft。' }
    ];
    const host = wrap.querySelector('#r3');
    const els = defs.map(d => U.card(d, { style: 'width:163px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#m3');
    const texts = [
      '这一段的产出是 <span class="v">dev_layer[]</span> —— 一张"层号 -> 设备与候选 buffer 类型"的路由表。',
      '<span class="v">dev_input</span> 被硬编码成 CPU（1536-1537 两行）：<span class="k">无论 n_gpu_layers 多大，输入层都不上 GPU</span>。',
      '<span class="v">dev_layer[il]</span> 由 1541-1542 的循环逐层填出，每一格来自同一个 <span class="cm" style="margin:0">get_layer_buft_list(il)</span>。',
      '<span class="v">dev_output</span> 用的是 <span class="cm" style="margin:0">get_layer_buft_list(n_layer_all)</span> —— ' +
        '输出层不是特例，只是第 28 个槽位。',
      '建张量时（1886 行）路由表被查了一次：<span class="k">tn.bid 是层号，查出来的 buft_list 决定这块权重进哪种 buffer</span>。',
      '所以"层"这个粒度到这里为止：<span class="k">一层 = 一组张量 = 一种 buffer 类型</span>。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 4 调度器只认 buffer：<span class="hl-b">usage == WEIGHTS</span> 就是"层"信息的载体 */
{
  kicker: "L8-02 · 交接",
  title: "调度器只认 buffer：<span class=\"hl-b\">usage == WEIGHTS</span> 就是\"层\"信息的载体",
  sub: "\"这一层的权重在 GPU 上\"这句话，到调度器这里被翻译成\"这个 src 的 buffer 用了 WEIGHTS\"。",
  caption: "回顾 L4-02 第 4 幕：那一课讲过这段的优先级判据（预分配 > 视图 > 图输入 > 权重）。本课关心的是它【读的是哪个字段】：src->buffer->usage。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 17, 19, 21, 30, 31],
  lineNo: 951,
  code: `    // operations with weights are preferably run on the same backend as the weights
//>> 这一段的规则：有权重的算子，尽量跟权重待在同一个后端
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
//>> ★ 判据只有一条：这个输入张量的 buffer 被标成了 WEIGHTS
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
//>> op_offload：权重虽在 host，但更高优先级（更靠前）的后端愿意接这个 op，就抢过去
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip b">src->buffer->usage</span><span class="arrow">==</span>
        <span class="chip c">GGML_BACKEND_BUFFER_USAGE_WEIGHTS</span>
      </div>
      <div class="row wrap" id="r4" style="gap:8px"></div>
      <div class="formula" id="m4"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '1. 认标记', m: 'src->buffer->usage == WEIGHTS', b: '遍历 GGML_MAX_SRC 个输入，<br>谁的 buffer 带 WEIGHTS 标记，<br>这个节点的归属就跟着谁走。' },
      { c: 'b', t: '2. 查后端', m: 'ggml_backend_sched_backend_from_buffer', b: '按 buffer 的 buft 找<b>优先级最高</b>的<br>（下标最小）且支持该 op 的后端。' },
      { c: 'c', t: '3. 允许被抢', m: 'sched->op_offload', b: '若权重在 host，且更靠前的后端<br>支持这个 op 又想接，就让 GPU 抢走。' },
      { c: 'd', t: '4. 兜底返回', m: 'return src_backend_id', b: '抢不到就老老实实跟权重待在一起。<br>找不到带 WEIGHTS 的输入则返回 -1，<br>留给后面的扩张轮次。' }
    ];
    const host = wrap.querySelector('#r4');
    const els = defs.map(d => U.card(d, { style: 'width:163px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#m4');
    const texts = [
      '这就是两次决策之间的<span class="k">唯一接口</span>：调度器不看层号，也不看 n_gpu_layers。',
      '<span class="v">1. 认标记</span>：WEIGHTS 这个标记是加载期打上去的。'
        + '权重 buffer 的 <span class="cm" style="margin:0">usage</span> 被设成 WEIGHTS —— 谁分的 buffer，谁就决定了这里。',
      '<span class="v">2. 查后端</span>：<span class="cm" style="margin:0">backend_from_buffer</span> 从下标 0 开始扫，'
        + '第一个"支持这种 buft 且支持这个 op"的后端中标 —— <span class="k">下标就是优先级</span>（回顾 L3-02：GPU 在前、CPU 在最后）。',
      '<span class="v">3. 允许被抢</span>：只有权重在 host 内存里、且更靠前的后端愿意接，才会脱离权重的后端。'
        + '这是唯一一处"层"说了不算的地方。',
      '<span class="v">4. 兜底</span>：返回 -1 的节点（没有权重输入的算子，如 add / rms_norm / rope）'
        + '<span class="k">在 pass 1 里根本没有归属</span>，全靠下一幕的扩张轮次决定。',
      '记住这句：<span class="k">层的信息以 buffer 的形式传进来，节点归属以"遍历顺序 + 扩张"的形式算出来</span>。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 5 ★ offload 的粒度是<span class="hl-c">层</span>，切分的粒度是<span class="hl-a">节点</span> */
{
  kicker: "L8-02 · 核心",
  title: "★ offload 的粒度是<span class=\"hl-c\">层</span>，切分的粒度是<span class=\"hl-a\">节点</span>",
  sub: "段是\"图上连续的一段节点\"，边界落在 node 下标 i 上 —— 不是落在层号上。",
  caption: "回顾 L4-02 第 5/6 幕：段是沿拓扑序找连续的同后端节点；边界上的张量要现场造一份 copy。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [1, 13, 14, 15, 20, 40],
  lineNo: 0,
  code: `            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }
//>> ---- ggml/src/ggml-backend.cpp:1399-1420 ----
                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="formula" style="padding:5px 9px">
        <span class="m">层的粒度</span>　一层 = 一组权重张量 = 一种 buffer<br>
        <span class="m">节点的粒度</span>　一个 node = 图上一次运算 = 一个归属
      </div>
      <div id="nodes" style="display:flex;gap:3px;align-items:center;flex-wrap:nowrap;width:100%"></div>
      <div class="formula" id="m5"></div>`;
    root.appendChild(wrap);

    // 一条示意节点链：每层 3 个节点（matmul / norm / add），层边界用粗间距标出
    const host = wrap.querySelector('#nodes');
    const nl = [];
    const LAYERS = 6, PER = 3;
    for (let l = 0; l < LAYERS; l++) {
      for (let k = 0; k < PER; k++) {
        const gpu = l >= 4;
        const c = U.el('div', { style: 'width:29px;height:26px;flex:0 0 auto;border:1px solid ' +
          (gpu ? 'var(--b)' : 'var(--a)') + ';border-radius:3px;background:' +
          (gpu ? 'rgba(63,185,80,.16)' : 'rgba(88,166,255,.16)') +
          ';display:flex;align-items:center;justify-content:center;font-size:8px;color:' +
          (gpu ? 'var(--b)' : 'var(--a)') });
        c.textContent = 'L' + l;
        if (k === 0 && l > 0) c.style.marginLeft = '7px';
        host.appendChild(c); nl.push(c);
      }
    }
    const msg = wrap.querySelector('#m5');
    const texts = [
      '同一张图，用两把尺子量：左边那把刻到"层"，右边那把刻到"节点"。',
      '加载期用的是<b>左边</b>这把尺子：<span class="v">dev_layer[il]</span> 一格就是一整层的全部权重，'
        + '<span class="k">层是不可分割的单位</span>。',
      '建图期换成<b>右边</b>这把尺子：调度器遍历 <span class="v">graph->nodes[i]</span>，'
        + '一格是一个 node。图上并没有"层"这个对象。',
      '切段时记录的不是层号，而是 <span class="v">split->i_start = i</span>（1359 行）—— '
        + '<span class="k">一个节点下标</span>。段的定义就是"节点区间 [i_start, i_end)"。',
      '这就是为什么 <span class="v">ggml_backend_sched_get_n_splits()</span> 得到的是段数，'
        + '而它<span class="k">不是 n_gpu_layers 的函数</span>。',
      '边界上的张量要现场复制一份（1399-1419）：<span class="v">node->src[j] = copy</span>。'
        + 'copy 是切分的产物，不是图里本来就有的节点（回顾 L4-02）。',
      '一句话：<span class="k">offload 决定"权重在哪"，切分决定"节点在哪跑"</span>；'
        + '前者是后者的输入，但后者不受前者的粒度约束。'
    ];
    tl.at(500, () => { msg.innerHTML = texts[0]; });
    tl.at(3400, () => {
      msg.innerHTML = texts[1];
      nl.forEach((c, i) => { c.style.borderTop = (i % PER === 0) ? '3px solid var(--c)' : '1px solid var(--border)'; });
    });
    tl.at(6800, () => {
      msg.innerHTML = texts[2];
      nl.forEach(c => { c.style.borderTop = ''; });
    });
    tl.at(10300, () => {
      msg.innerHTML = texts[3];
      nl.forEach((c, i) => { c.style.outline = i === 11 ? '2px solid var(--e)' : 'none'; });
    });
    tl.at(14200, () => {
      msg.innerHTML = texts[4];
      nl.forEach((c, i) => { c.style.outline = i === 11 ? '2px solid var(--e)' : 'none'; });
    });
    tl.at(18000, () => {
      msg.innerHTML = texts[5];
      nl.forEach((c, i) => { c.style.opacity = (i === 11 || i === 12) ? '1' : '.45'; c.style.outline = 'none'; });
    });
    tl.at(21500, () => {
      msg.innerHTML = texts[6];
      nl.forEach(c => { c.style.opacity = '1'; });
    });
  }
},

/* ------------------------------------------------------ 6 没有权重的节点由<span class="hl-a">扩张</span>决定归属，而 <span class="hl-e">CPU 是一堵墙</span> */
{
  kicker: "L8-02 · 扩张",
  title: "没有权重的节点由<span class=\"hl-a\">扩张</span>决定归属，而 <span class=\"hl-e\">CPU 是一堵墙</span>",
  sub: "pass 2 把 GPU 归属向上下两个方向推开，碰到 CPU 归属的节点就把\"当前后端\"清空。",
  caption: "源码注释（1127 行）把结论写死了：cpu will never be used unless weights are on cpu。所以 n_gpu_layers 依然有效 —— 它决定了墙在哪。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 3, 14, 15, 17, 22],
  lineNo: 1124,
  code: `    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div id="lane" class="col" style="gap:4px;width:100%"></div>
      <div class="row" style="gap:8px">
        <div class="card" style="width:336px;border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">expand gpu down / up</div>
          <div class="cb">从已分配的 GPU 节点出发，向前、向后各扫一遍，
          把中间<b>还没归属</b>且后端支持该 op 的节点也标成 GPU（1146 行）。</div></div>
        <div class="card" style="width:336px;border-left-color:var(--e)">
          <div class="ct" style="color:var(--e)">CPU 是墙，不是桥</div>
          <div class="cb">遇到 CPU 归属的节点，<span class="cm" style="margin:0">cur_backend_id = -1</span>（1141 行）：
          扩张在这里<b>停止</b>，不会越过 CPU 节点把更远的节点也染成 GPU。</div></div>
      </div>
      <div class="formula" id="m6"></div>`;
    root.appendChild(wrap);

    const lane = wrap.querySelector('#lane');
    const KIND = ['w','-','w', 'w','-','w', 'w','-','w', 'w','-','w', 'w','-','w', 'w','-','w'];
    const GPU  = [ 0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0,   1,  1,  1,   1,  1,  1];
    const cells = [];
    const row = U.el('div', { style: 'display:flex;gap:3px;align-items:center;width:100%' });
    lane.appendChild(row);
    KIND.forEach((k, i) => {
      const g = GPU[i] === 1, hasW = k === 'w';
      const c = U.el('div', { style: 'width:30px;height:30px;flex:0 0 auto;border:1px solid ' +
        (hasW ? (g ? 'var(--b)' : 'var(--a)') : 'var(--border)') + ';border-radius:3px;background:' +
        (hasW ? (g ? 'rgba(63,185,80,.18)' : 'rgba(88,166,255,.18)') : '#10151b') +
        ';display:flex;align-items:center;justify-content:center;font-size:9px;color:' +
        (hasW ? (g ? 'var(--b)' : 'var(--a)') : 'var(--dim)') });
      c.textContent = Math.floor(i / 3);
      if (k === '-') { c.style.borderStyle = 'dashed'; }
      if (i > 0 && i % 3 === 0) c.style.marginLeft = '8px';
      row.appendChild(c); cells.push(c);
    });
    lane.appendChild(U.el('div', { class: 'formula', style: 'padding:3px 8px;font-size:9.5px',
      html: '格内数字 = 层号，一层 3 个节点。实线 = pass 1 已被权重<b>钉死</b>归属（蓝 = CPU / 绿 = GPU）；' +
            '虚线 = 没有权重输入，pass 1 返回 -1，归属未知' }));

    const msg = wrap.querySelector('#m6');
    function fill(i, c, col) {
      cells[i].style.background = c;
      cells[i].style.borderColor = col;
      cells[i].style.borderStyle = 'solid';
      cells[i].style.color = col;
    }
    const texts = [
      '初始态：有权重的节点在 pass 1 就定了归属（蓝 = CPU 权重，绿 = GPU 权重）；' +
        '<span class="k">没有权重的节点（虚线）还是 -1</span>。',
      '<span class="v">expand gpu down</span>（1131-1148）：从上往下扫，把 GPU 归属扩散给还没归属的邻居 —— ' +
        '层 4 内部那两个虚线节点被染成 GPU。',
      '扫到层 3 的最后一个权重节点（CPU 归属）时，<span class="v">cur_backend_id = -1</span>（1139-1141）：' +
        '<span class="k">扩张在这里断掉</span>，不会越过它去染层 0..3。',
      '层 0..3 里剩下的虚线节点由 <span class="v">expand rest down</span>（1171-1186）补上 —— ' +
        '它们前面是 CPU，于是也归 CPU。',
      '最终落在 <span class="k">两段</span>：节点 0..11 一段 CPU、节点 12..17 一段 GPU。' +
        '<span class="k">CPU 权重节点就是那堵墙</span>。',
      '源码注释（1127 行）：<span class="cm" style="margin:0">cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops</span>。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => {
      msg.innerHTML = texts[1];
      fill(13, 'rgba(63,185,80,.34)', 'var(--b)');
      fill(16, 'rgba(63,185,80,.34)', 'var(--b)');
    });
    tl.at(7200, () => {
      msg.innerHTML = texts[2];
      cells[11].style.outline = '2px solid var(--e)';
    });
    tl.at(10800, () => {
      msg.innerHTML = texts[3];
      [1,4,7,10].forEach(i => fill(i, 'rgba(88,166,255,.34)', 'var(--a)'));
    });
    tl.at(14000, () => {
      msg.innerHTML = texts[4];
      cells[11].style.outline = 'none';
      cells[12].style.outline = '2px solid var(--c)';
    });
    tl.at(17000, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 边界<span class="hl-e">不一定</span>落在层边界上：源码自己打了一个补丁 */
{
  kicker: "L8-02 · 反例",
  title: "边界<span class=\"hl-e\">不一定</span>落在层边界上：源码自己打了一个补丁",
  sub: "norm / l_last 可能被扩张轮次判给\"下一层\"的后端 —— 于是 llama_context 手工把它按回去。",
  caption: "这段只在 ubatch.n_tokens < 32 或完全 offload 时生效。也就是说：批量预填充（n_tokens 大）且非完全 offload 时，补丁不生效，边界可以劈开一层。",
  src: "src/llama-context.cpp",
  mark: [0, 8, 9, 11, 12, 18],
  lineNo: 2598,
  code: `llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div id="t7"></div>
      <div class="formula" id="m7"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['情形', '是否强制', '后果'],
      [['n_tokens < 32（逐 token 解码）', '强制把 norm / l_last 按回本层的设备', '边界对齐层边界'],
       ['full_offload（n_gpu_layers > n_layer_all）', '强制（同上，但本来就全在 GPU）', '边界无意义'],
       ['n_tokens >= 32（预填充）且非 full_offload', { html: '<b>不强制</b>' },
        'norm 可能落到下一层的后端，边界劈开一层']],
      { monoCols: [] });
    wrap.querySelector('#t7').appendChild(t.el);

    const msg = wrap.querySelector('#m7');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '源码注释把问题说得很直白：<span class="cm" style="margin:0">norm may be automatically assigned to the backend of the previous layer</span>。',
      '<span class="v">graph_get_cb</span> 是建图时的回调，每个张量建出来时都会经过它。',
      '第 2610 行的门：<span class="v">ubatch.n_tokens &lt; 32 || full_offload</span> —— 两种情形之外，这段代码根本不执行。',
      '第 2609 行：<span class="v">full_offload = model.n_gpu_layers() &gt; model.hparams.n_layer_all</span>。',
      '把这条判据和第 1521 行对一下：<span class="v">n_gpu_layers &gt; n_layer_all</span> 恰好就是 ' +
        '<span class="cm" style="margin:0">i_gpu_start == 0</span> 的条件 —— <span class="k">三处代码对同一个边界是一致的</span>。',
      '强制的手段是 <span class="v">ggml_backend_sched_set_tensor_backend</span>（2616 行）：'
        + '它在切分之前把一个节点<b>手工钉死</b>在指定后端上。',
      '所以"边界落在层边界上"是<span class="k">被修补出来的结果</span>，不是切分算法的性质。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3400 + i * 2900, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(12500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
    tl.at(16500, () => { msg.innerHTML = texts[5]; });
    tl.at(19500, () => { msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 8 把一个 27 层模型从 <span class="hl-v">n_gpu_layers = 0</span> 走到 28 */
{
  kicker: "L8-02 · 算例",
  title: "把一个 27 层模型从 <span class=\"hl-v\">n_gpu_layers = 0</span> 走到 28",
  sub: "算例模型取自源码里点名的真实架构：DeepSeek-V2-Lite，n_layer_all = 27。",
  caption: "27 来自 src/models/deepseek2.cpp:7-8 的注释与判据（\"lite variants include DeepSeek-V2-Lite\"，n_layer() == 27）；该文件不计入本课覆盖率。槽位总数 = 27 + 1 = 28。",
  src: "src/llama-model.cpp",
  mark: [0, 1, 4, 7, 9, 11, 13, 16],
  lineNo: 0,
  code: `    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
//>> ---- src/llama-model.cpp:1833-1847 ----
    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_all);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\\n", __func__, n_repeating);

        const int max_backend_supported_layers = n_layer_all + 1;
        const int max_offloadable_layers       = n_layer_all + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div id="t8"></div>
      <div id="strip8" style="display:flex;gap:2px;align-items:center;flex-wrap:nowrap;width:100%"></div>
      <div class="formula" id="m8"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['n_gpu_layers', 'i_gpu_start = max(28 - L, 0)', 'GPU 上的槽位', 'GPU 上的重复层'],
      [['0',  '28', '（无）',            '0'],
       ['1',  '27', '27 = 输出层',        '0'],
       ['4',  '24', '24,25,26 + 输出层',  '3'],
       ['14', '14', '14..26 + 输出层',    '13'],
       ['27', '1',  '1..26 + 输出层',     '26'],
       ['28', '0',  '0..26 + 输出层',     '27  <- 全上 GPU']],
      { monoCols: [0, 1] });
    wrap.querySelector('#t8').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const strip = wrap.querySelector('#strip8');
    const cells = [];
    for (let s = 0; s <= 27; s++) {
      const c = U.el('div', { style: 'width:20px;height:17px;border:1px solid var(--border);border-radius:2px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim);flex:0 0 auto' });
      c.textContent = s;
      if (s === 27) c.style.marginLeft = '8px';
      strip.appendChild(c); cells.push(c);
    }
    function paint(Lv) {
      const start = Math.max(28 - Lv, 0);
      cells.forEach((c, s) => {
        const on = s >= start;
        c.style.background = on ? 'rgba(63,185,80,.24)' : '#10151b';
        c.style.borderColor = on ? 'var(--b)' : 'var(--border)';
        c.style.color = on ? 'var(--b)' : 'var(--dim)';
      });
    }
    paint(0);

    const msg = wrap.querySelector('#m8');
    const texts = [
      '规则只有一条：<span class="v">i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)</span>（1521 行），' +
        '槽位号 <span class="k">&gt;= i_gpu_start</span> 的走 GPU。',
      '<span class="v">L = 0</span>：i_gpu_start = 28，比最大的槽位号 27 还大 —— <span class="k">全在 CPU</span>。',
      '<span class="v">L = 1</span>：i_gpu_start = 27。唯一上 GPU 的是输出层，重复层 0 个（对应 1837-1841 的日志分支）。',
      '<span class="v">L = 4</span>：i_gpu_start = 24，末尾 4 个槽位 —— 层 24/25/26 与输出层。',
      '<span class="v">L = 14</span>：i_gpu_start = 14，末尾 14 个槽位里 13 个是重复层（14..26），第 14 个是输出层。',
      '<span class="v">L = 27</span>：i_gpu_start = 1 —— 注意 <span class="k">第 0 层还在 CPU</span>。',
      '<span class="v">L = 28</span>：i_gpu_start = 0，28 个槽位全上 GPU。<span class="k">这才是"全部 offload"</span>。',
      '对照 1834 行：它算 n_gpu 时用的是 <span class="v">min(n_gpu_layers, n_layer_all)</span>，' +
        '而 1522 行用的是 <span class="v">min(n_gpu_layers, n_layer_all + 1)</span> —— 当 L = 28 时，' +
        '1834 行把计数卡在 27，1841 行于是报出 26 个 repeating 层（实际是 27），' +
        '1846 行又按 <span class="v">n_layer_all + 1</span> 报出 28/28。' +
        '<span class="k">三行日志口径不一致，别拿它反推切分</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2800 + i * 3300, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      paint(parseInt(r.children[0].textContent, 10));
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(2800 + 6 * 3300, () => {
      rows.forEach(x => { x.className = ''; });
      paint(28);
      msg.innerHTML = texts[7];
    });
  }
},

/* ------------------------------------------------------ 9 把决策链压成一张表 */
{
  kicker: "L8-02 · 收束",
  title: "把决策链压成一张表",
  sub: "五个决策点、各自的判据与产出；最后用一道题自测。",
  caption: "下一课 L8-03：写一个新后端要做什么 —— 那时你会站在这条链的另一端。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 3, 4],
  lineNo: 1124,
  code: `    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="t9"></div><div id="ex9"></div><div class="formula" id="m9"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['决策点', '判据（源码行）', '产出'],
      [['分界点', 'i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)（llama-model.cpp:1521）', '哪些槽位能上 GPU'],
       ['层 -> 设备', 'get_layer_buft_list(il) 写入 dev_layer[il]（1542）', 'dev_layer[]'],
       ['层 -> buffer', 'create_tensor 按 tn.bid 取 dev_layer[bid].buft_list（1886）', '权重 buffer + WEIGHTS 标记'],
       ['节点归属', 'src->buffer->usage == WEIGHTS（ggml-backend.cpp:967）', 'pass 1 的结果'],
       ['切段与搬运', 'split->i_start = i（1359）/ 边界造 copy（1419）', 'splits[]']],
      { monoCols: [] });
    wrap.querySelector('#t9').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex9').appendChild(W.exercise(
      '某模型 <span class="mono">n_layer_all = 27</span>，只挂 1 张 GPU，用 <span class="mono">--n-gpu-layers 4</span> 启动。' +
      '请说出：(1) <span class="mono">i_gpu_start</span> 是多少？(2) 哪些槽位在 GPU 上？' +
      '(3) 哪些重复层留在 CPU？(4) 改成 <span class="mono">--n-gpu-layers 27</span> 是不是所有层都上 GPU 了？',
      '(1) <span class="mono">i_gpu_start = max(27 + 1 - 4, 0) = 24</span>（llama-model.cpp:1521）。<br>' +
      '(2) 槽位 <span class="mono">24, 25, 26</span>（重复层）与槽位 <span class="mono">27</span>（输出层）—— 共 4 个，' +
      '因为槽位号 <span class="mono">&gt;= i_gpu_start</span> 才走 GPU（1525 行的 CPU 分支取反）。<br>' +
      '(3) 重复层 <span class="mono">0..23</span> 留在 CPU；此外 token_embd 所在的输入层' +
      '<b>永远在 CPU</b>（1536-1537 行硬编码），与 n_gpu_layers 无关。<br>' +
      '(4) <b>不是</b>。<span class="mono">--n-gpu-layers 27</span> 时 <span class="mono">i_gpu_start = 28 - 27 = 1</span>，' +
      '第 0 层仍留在 CPU。要 27 个重复层全上 GPU，需要 <span class="mono">n_gpu_layers = 28 = n_layer_all + 1</span>；' +
      '这也正是参数取 <span class="mono">-1</span>（默认）时 <span class="mono">n_gpu_layers()</span> 的返回值（1926 行）。'));

    const msg = wrap.querySelector('#m9');
    const texts = [
      '五个决策点，前三个在加载期（粒度 = 层），后两个在建图期（粒度 = 节点）。',
      '<span class="k">分界点</span>：一行算术，决定权重去哪。它是 n_gpu_layers 唯一真正影响的东西。',
      '<span class="k">层 -> 设备 -> buffer</span>：路由表 dev_layer[] 只被查一次，产物是"权重的 buffer 类型"。',
      '<span class="k">节点归属</span>：调度器读的是 buffer 的 usage 标记，不是层号 —— 这是两次决策的接口。',
      '<span class="k">切段与搬运</span>：段是节点区间，边界上的张量要现场复制。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(15000, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = '一句话：<span class="k">offload 决定权重在哪，切分决定节点在哪跑 —— 两次决策，两种粒度</span>。';
    });
  }
},

];
