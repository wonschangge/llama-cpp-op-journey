/* ==========================================================================
   L6-08 · Metal 后端：主机端与图融合
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 Metal 的融合：<span class="hl-a">一张表，两个阶段</span> */
{
  kicker: "L6 · GPU 后端执行",
  title: "Metal 的融合：<span class=\"hl-a\">一张表，两个阶段</span>",
  sub: "开头的这 6 行注释就是本课的全部内容：融合模式只声明一次，优化阶段和编码阶段读同一张表。",
  caption: "本课覆盖 16 个主机端文件。这个窗口是后端私有的，前端与其它后端都不感知。",
  src: "ggml/src/ggml-metal/ggml-metal-fusion.h",
  mark: [2, 3, 4, 5],
  lineNo: 1,
  code: `// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table, so the two
// phases can never disagree about what can be fused.`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip">前端建好的 ggml_cgraph</span><span class="arrow">-></span>
        <span class="chip a">优化阶段 · 按表打包</span><span class="arrow">-></span>
        <span class="chip b">编码阶段 · 合内核</span><span class="arrow">-></span>
        <span class="chip c">commit 给 GPU</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'ggml-metal-fusion.h / .cpp', m: '26 条表项 · 9 种融合 id', b: '融合表：优化阶段与编码阶段<br><b>共用的唯一判据</b>（本课核心）', w: '224px' },
      { c: 'b', t: 'ggml-metal-common.h / .cpp', m: 'ggml_graph_optimize()', b: '为融合而打包 -> 重排提并发<br>-> 再解包', w: '224px' },
      { c: 'c', t: 'ggml-metal-context.h / .m', m: 'ggml_metal_graph_compute()', b: '图编码循环：<br>idx += res - 1 跳过被吞的节点', w: '224px' },
      { c: 'd', t: 'ggml-metal-ops.h / .cpp', m: 'ggml_metal_op_encode()', b: '每个 op 一个编码器；<br>命中融合就改参数、加槽位', w: '224px' },
      { c: 'e', t: 'ggml-metal-device.h / .cpp / .m', m: 'ggml_metal_library_get_pipeline_*', b: '设备与 pipeline 缓存：<br>融合后的 kernel 名在这里拼出', w: '224px' },
      { c: 'f', t: 'ggml-metal-impl.h', m: 'ggml_metal_kargs_*', b: 'kernel 参数 ABI：<br>nef1[3] / nbf1[3] 是融合槽位', w: '224px' },
      { c: 'g', t: 'ggml-metal-tuning.h / .cpp', m: 'fa_vec_tuned_table', b: '另一类主机端决策：调参表。<br>与融合正交，不是一回事', w: '224px' },
      { c: 'e', t: 'ggml-metal.cpp', m: 'ggml_backend_metal_graph_compute', b: '后端接口实现：<br>融合对外的两个触发入口', w: '224px' },
      { c: 'd', t: 'ggml/include/ggml-metal.h', m: 'GGML_BACKEND_API', b: '对外 API 里<b>没有</b>融合 ——<br>它没有出现在任何公共头里', w: '224px' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.26');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看地图：16 个文件分成四组 —— <span class="k">表</span>（fusion）、<span class="k">两个阶段</span>（common / context）、<span class="k">编码器</span>（ops）、<span class="k">设备与内核参数</span>（device / impl / tuning）。',
      '<span class="k">表</span>只有一处：<span class="v">ggml_metal_fusions</span>。注释说得很直白 —— 模式声明一次，优化器和编码器都读它，两阶段<b>不可能互相矛盾</b>。',
      '<span class="k">两个阶段</span>：优化阶段按表把可融合的节点<b>临时打包</b>（common），编码阶段再按表把整组节点<b>合成一次调用</b>（context -> ops）。',
      '<span class="k">编码器与设备</span>：命中融合后，编码器把额外输入塞进 kernel 参数槽（impl.h），再到 device 里取那个融合 kernel 的 pipeline。',
      '收束：融合是 <span class="k">后端私有</span> 的优化。<span class="cm" style="margin:0">ggml/include/ggml-metal.h</span> 里没有任何 fusion 相关声明 —— 用这个后端的人看不见它。'
    ];
    const groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    groups.forEach((g, gi) => tl.at(2900 + gi * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = g.indexOf(k) >= 0 ? '1' : '.26'; });
      msg.innerHTML = texts[gi + 1];
    }));
    tl.at(12500, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ 融合只存在于 <span class="hl-a">"图已建好、还没编码"</span> 的窗口里 */
{
  kicker: "L6-08 · ★ 核心洞察",
  title: "★ 融合只存在于 <span class=\"hl-a\">\"图已建好、还没编码\"</span> 的窗口里",
  sub: "后端拿到 ggml_cgraph 之后、把节点编码成 Metal 命令之前 —— 这就是融合能存在的全部空间。",
  caption: "回顾 L4-04：graph_compute 是后端唯一的执行入口。本课看后端在这个入口内部又做了什么。",
  src: "ggml/src/ggml-metal/ggml-metal.cpp",
  mark: [0, 3],
  lineNo: 544,
  code: `static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="stages" style="gap:5px"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const stages = ['前端建图', '调度器切分 + 分配', '★ 融合窗口', '编码成 Metal 命令', 'GPU 执行'];
    const host = wrap.querySelector('#stages');
    const els = stages.map((s, i) => {
      const e = U.el('div', { class: 'chip' + (i === 2 ? ' a' : ''), text: s });
      host.appendChild(e);
      if (i < stages.length - 1) host.appendChild(U.arrow('->'));
      return e;
    });
    els.forEach(e => e.style.opacity = '.35');

    const defs = [
      { c: 'a', t: '窗口内能看到什么', m: 'ggml_cgraph * cgraph', b: '完整的节点序列、每个节点的 src[]、<br>张量已经分配好（buffer / data 都在）', w: '224px' },
      { c: 'c', t: '前端不感知', m: '图上节点一个不少', b: '融合吞掉的是<b>编码动作</b>，<br>不是图上的节点，也没有新 op 名字', w: '224px' },
      { c: 'b', t: '其它后端也看不到', m: '逻辑全在 ggml-metal-*', b: 'ggml 层只提供通用的"可消除"判据，<br>模式表完全由 Metal 自己维护', w: '224px' }
    ];
    const chost = wrap.querySelector('#cards');
    const cards = defs.map(d => { const e = U.card(d); chost.appendChild(e); return e; });
    cards.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'L4-04 讲过：<span class="cm" style="margin:0">ggml_backend_graph_compute(backend, cgraph)</span> 是后端唯一的执行入口。',
      'Metal 的实现只有 4 行 —— 但它转手把整个 <span class="v">ggml_cgraph</span> 交给了 <span class="cm" style="margin:0">ggml_metal_graph_compute()</span>。融合就发生在这之后。',
      '<span class="k">窗口的左边界</span>：图已建好、张量已分配；<span class="k">右边界</span>：节点还没被编码进 MTLCommandBuffer。中间这段，后端想怎么改就怎么改。',
      '<span class="k">关键推论</span>：融合是<b>后端私有的主机端优化</b>，所以前端、ggml 调度器、其它后端都看不到它 —— 这与 Vulkan / CUDA 的编码路径形成对照（见 L6-07、L6-01）。',
      '窗口内的产物只有两样：<span class="v">kernel 参数</span>和<span class="v">一次 dispatch</span>。没有新的 ggml op，没有新的节点。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, []); els[0].style.opacity = '1'; els[1].style.opacity = '1'; });
    tl.at(3400, () => { msg.innerHTML = texts[1]; els[2].style.opacity = '1'; U.markLines(document, [0]); });
    tl.at(6400, () => { msg.innerHTML = texts[2]; els[3].style.opacity = '1'; U.markLines(document, [3]); });
    tl.at(10200, () => { msg.innerHTML = texts[3]; cards.forEach(e => { e.style.opacity = '1'; }); els[4].style.opacity = '1'; });
    tl.at(14200, () => { msg.innerHTML = texts[4]; cards.forEach(e => { e.style.opacity = '.28'; }); cards[2].style.opacity = '1'; });
  }
},

/* ------------------------------------------------------ 3 同一张表，被问两次：<span class="hl-b">STRUCTURAL</span> 与 <span class="hl-c">FULL</span> */
{
  kicker: "L6-08 · 何时触发",
  title: "同一张表，被问两次：<span class=\"hl-b\">STRUCTURAL</span> 与 <span class=\"hl-c\">FULL</span>",
  sub: "模式枚举解释了\"何时\"：优化阶段张量还没分配，只能做结构判定；编码阶段才能做完整判定。",
  caption: "GGML_METAL_FUSION_MAX = 16 同时也是表里最长模式的长度（MoE 输出归约 8 专家那条）。",
  src: "ggml/src/ggml-metal/ggml-metal-fusion.h",
  mark: [2, 7, 10, 15, 17, 21, 23, 24, 25, 26],
  lineNo: 18,
  code: `// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#define GGML_METAL_FUSION_MAX 16

typedef enum ggml_metal_fusion_mode {
    // structural checks only; used by the graph optimizer, at which point the graph
    // tensors are not allocated yet, so buffer placement cannot be verified
    GGML_METAL_FUSION_STRUCTURAL = 0,
//>> STRUCTURAL：只做结构判定。注释写明此时"tensors are not allocated yet"
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_FULL,
//>> FULL：完整判定，编码阶段用
} ggml_metal_fusion_mode;

// identifier of each fusion pattern so the op encoders know which kernel to use
typedef enum ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_ADD, // NORM/RMS_NORM + MUL + ADD
//>> NORM/RMS_NORM + MUL + ADD 是同一个 id —— 融合的粒度是"内核能力"，不是 op 名字
    GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    GGML_METAL_FUSION_ADD_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
} ggml_metal_fusion_id;`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div id="ids" style="line-height:2.0"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['模式', '谁在问', '张量已分配？', '多做哪些检查'],
      [['GGML_METAL_FUSION_STRUCTURAL', 'ggml_metal_fusion_max()（优化）', '否',
        'op 序列精确匹配 + 链式 + 同形 + 可消除（子图外没人用中间结果）'],
       ['GGML_METAL_FUSION_FULL', 'ctx->can_fuse()（编码）', '是',
        '上面全部，外加模式 check 里的 ->data 检查与"外部源是否与融合输出重叠"的别名检查']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const ids = ['NORM_MUL', 'NORM_MUL_ADD', 'NORM_SCALE', 'ADD_CHAIN', 'SNAKE',
                 'GDN_CACHE', 'TOPK_MOE', 'MOE_REDUCE', 'SSM_CONV_SILU'];
    const idhost = wrap.querySelector('#ids');
    const chips = ids.map((s, i) => {
      const e = U.chip('GGML_METAL_FUSION_' + s, ['a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c'][i]);
      idhost.appendChild(e);
      idhost.appendChild(document.createTextNode(' '));
      return e;
    });
    chips.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '先看常量：<span class="v">GGML_METAL_FUSION_MAX = 16</span> —— 一次最多融合 16 个节点，这也是表里最长模式（8 专家 MoE 归约）的长度。',
      '<span class="k">STRUCTURAL 模式</span>：图刚被切分、张量<b>还没分配</b>，所以只能看形状与连接关系。注释把这一点写在枚举里。',
      '<span class="k">FULL 模式</span>：编码阶段才用。这时 <span class="v">buffer</span> / <span class="v">data</span> 已经就位，可以判"外部源会不会和融合输出撞车"。',
      '中间那一串 <span class="v">GGML_METAL_FUSION_*</span> 就是全部可融合组合的编号 —— 一共 <b>9 种</b>，下一幕逐个展开。',
      '注意 <span class="v">NORM_MUL_ADD</span> 一个 id 同时管 NORM 和 RMS_NORM 两条序列：融合的粒度是<b>内核能做什么</b>，不是 op 名字的字符串拼接。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => { msg.innerHTML = texts[1]; rows.forEach((x, k) => { x.className = k === 0 ? 'on' : ''; }); });
    tl.at(7000, () => { msg.innerHTML = texts[2]; rows.forEach((x, k) => { x.className = k === 1 ? 'on' : ''; }); });
    tl.at(10600, () => { msg.innerHTML = texts[3]; rows.forEach(x => { x.className = ''; }); });
    chips.forEach((c, i) => tl.at(11600 + i * 780, () => { c.style.opacity = '1'; }));
    tl.at(18800, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 先按表<b>打包</b>，再重排 —— <span class="hl-c">否则重排会把融合打断</span> */
{
  kicker: "L6-08 · 触发点一（优化阶段）",
  title: "先按表<b>打包</b>，再重排 —— <span class=\"hl-c\">否则重排会把融合打断</span>",
  sub: "调度器切分完图、做 graph_copy 之前，会对每个后端调用一次 graph_optimize 钩子。",
  caption: "回顾 L1-03：节点顺序就是拓扑序。重排是在不破坏依赖的前提下换顺序，而融合要求节点相邻 —— 两者会打架。",
  src: "ggml/src/ggml-metal/ggml-metal-common.cpp",
  mark: [6, 7, 8, 10, 11, 18],
  lineNo: 446,
  code: `void ggml_graph_optimize(ggml_cgraph * gf) {
    const int n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        const int f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        for (int k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            node.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(node));
    }`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="seq" style="gap:4px"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const seq = ['ADD', 'ADD', 'ADD', 'MUL_MAT', 'SILU', 'MUL', 'ADD', 'CPY'];
    const host = wrap.querySelector('#seq');
    const boxes = seq.map(s => {
      const e = U.el('div', { text: s });
      e.style.cssText = 'width:74px;padding:5px 0;text-align:center;border:1px solid var(--border);border-radius:5px;background:#10151b;font-family:var(--mono);font-size:9px;color:var(--muted)';
      host.appendChild(e);
      return e;
    });
    const sep = U.el('div', { text: '|' });
    sep.style.cssText = 'color:var(--dim);font-size:10px;padding:0 2px';
    host.insertBefore(sep, boxes[3]);

    const defs = [
      { c: 'c', t: '为什么先打包', b: '注释：不想让重排破坏融合，所以先把可融合的张量<b>打包成一个单位</b>，再以"组"为粒度重排', w: '224px' },
      { c: 'a', t: '打包用什么判据', m: 'ggml_metal_fusion_max(gf, i)', b: 'STRUCTURAL 模式：把模式<b>首尾相接</b>地匹配，返回原始下标跨度（中间的 VIEW 一起带上）', w: '224px' },
      { c: 'b', t: '谁调用这个钩子', m: 'ggml_backend_sched', b: '每个切分（split）各调一次；同时收集后端上报的分配依赖（见 ggml-metal.cpp:562）', w: '224px' }
    ];
    const chost = wrap.querySelector('#cards');
    const cards = defs.map(d => { const e = U.card(d); chost.appendChild(e); return e; });
    cards.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '示意一条 8 节点的序列。前三个 <span class="v">ADD</span> 正好是表里的 <span class="k">ADD_CHAIN</span> 模式（ADD x N，N 在 [2,7]）。',
      '<span class="k">问题</span>：重排是为了让互不干扰的节点早跑（提并发）；可融合要求节点<b>紧紧相邻</b> —— 重排一旦把 ADD 们拆散，编码阶段就再也匹配不上了。',
      '<span class="k">解法</span>：先把它们打包成一个"融合组"，重排时整组一起搬。看清楚：<b>打包只是优化阶段内部的临时表示</b>。',
      '<span class="v">ggml_metal_fusion_max()</span> 返回的是<b>原始下标跨度</b>（含中间被透传的 VIEW/RESHAPE），所以打包不会漏掉那些"空节点"。',
      '这一整段都是 STRUCTURAL 判定 —— 此刻张量还没分配，判不了 buffer 冲突，只能判形状与连接。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, []); });
    tl.at(3600, () => {
      [0, 1, 2].forEach(i => { boxes[i].style.borderColor = 'var(--c)'; boxes[i].style.color = 'var(--c)'; });
      msg.innerHTML = texts[1]; U.markLines(document, [6, 7, 8]);
    });
    tl.at(7600, () => {
      [0, 1, 2].forEach(i => { boxes[i].style.background = 'rgba(210,153,34,.16)'; });
      msg.innerHTML = texts[2]; cards[0].style.opacity = '1'; U.markLines(document, [10, 11]);
    });
    tl.at(11600, () => { msg.innerHTML = texts[3]; cards[1].style.opacity = '1'; U.markLines(document, [18]); });
    tl.at(15200, () => { msg.innerHTML = texts[4]; cards.forEach(e => { e.style.opacity = '1'; }); U.markLines(document, [18]); });
  }
},

/* ------------------------------------------------------ 5 编码循环：<span class="hl-a">idx += res - 1</span> 就是"吞掉"发生的地方 */
{
  kicker: "L6-08 · 触发点二（编码阶段）",
  title: "编码循环：<span class=\"hl-a\">idx += res - 1</span> 就是\"吞掉\"发生的地方",
  sub: "ggml_metal_op_encode() 返回它消费了几个节点；调用者据此跳过被融合的节点。",
  caption: "ctx->n_nodes() 数的是过滤掉 VIEW/RESHAPE 之后的非空节点；融合只在同一个编码器内进行，跨编码器会 GGML_ABORT。",
  src: "ggml/src/ggml-metal/ggml-metal-context.m",
  mark: [11, 12, 18],
  lineNo: 744,
  code: `        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute,
            ctx->debug_graph);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
//>> res = 本次编码消费掉的节点数（没融合就是 1）
            if (res == 0) {
                break;
            }

            idx += res - 1;
//>> 跳过被融合吞掉的节点 —— 它们不会被单独编码
        }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="cm" style="font-size:9.5px;color:var(--dim);text-align:center">示意：一次编码里看到的非空节点序列（含两个可融合组）</div>
      <div class="row center" id="seq" style="gap:4px"></div>
      <div id="ptr" class="formula" style="text-align:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const seq = ['ADD', 'ADD', 'ADD', 'MUL_MAT', 'RMS_NORM', 'MUL', 'ADD', 'CPY'];
    const host = wrap.querySelector('#seq');
    const boxes = seq.map(s => {
      const e = U.el('div', { text: s });
      e.style.cssText = 'width:78px;padding:6px 0;text-align:center;border:1px solid var(--border);border-radius:5px;background:#10151b;font-family:var(--mono);font-size:9px;color:var(--muted)';
      host.appendChild(e);
      return e;
    });
    const ptr = wrap.querySelector('#ptr');
    const msg = wrap.querySelector('#msg');

    function clearAll() {
      boxes.forEach(b => { b.style.background = '#10151b'; b.style.borderColor = 'var(--border)'; b.style.color = 'var(--muted)'; });
    }
    function mark(range, color, bg) {
      clearAll();
      range.forEach(i => { boxes[i].style.background = bg; boxes[i].style.borderColor = 'var(--' + color + ')'; boxes[i].style.color = 'var(--' + color + ')'; });
    }

    const BLUE = 'rgba(88,166,255,.16)';
    const AMBER = 'rgba(210,153,34,.18)';

    const texts = [
      '编码循环很朴素：对第 idx 个节点编码，拿到它消费的节点数 res，然后 <span class="v">idx += res - 1</span>。',
      '第 0 个节点 <span class="v">ADD</span>：命中 <span class="k">ADD_CHAIN</span>（表里 ADD x 3），<span class="v">res = 3</span>。',
      '<span class="v">idx += 3 - 1</span> -> 直接跳到下标 3。<b>三个 ADD 只编码了一次</b>，另外两个节点被"吞掉"了。',
      '下标 3 的 <span class="v">MUL_MAT</span>：表里没有以它开头的模式，<span class="v">res = 1</span>，照常编码。',
      '下标 4 起是 <span class="v">RMS_NORM + MUL + ADD</span>，命中 <span class="k">NORM_MUL_ADD</span>，<span class="v">res = 3</span>，一次 kernel 顶三次。',
      '收束：<span class="k">融合 = 编码循环少走几步</span>。图没变、节点没变、也没产生新的 ggml op —— 变的只是"几次 dispatch"。'
    ];
    tl.at(600, () => { mark([0], 'a', BLUE); ptr.innerHTML = 'idx = 0 &nbsp; | &nbsp; res = ?'; msg.innerHTML = texts[0]; U.markLines(document, [11]); });
    tl.at(3600, () => { mark([0, 1, 2], 'c', AMBER); ptr.innerHTML = 'idx = 0 &nbsp; | &nbsp; <span class="v">res = 3</span> &nbsp; | &nbsp; idx += 3 - 1 = 3'; msg.innerHTML = texts[1]; U.markLines(document, [12, 13]); });
    tl.at(7600, () => { mark([0, 1, 2], 'c', AMBER); ptr.innerHTML = 'idx = 3 &nbsp; <span class="k">（下标 1、2 直接被跳过）</span>'; msg.innerHTML = texts[2]; U.markLines(document, [17]); });
    tl.at(11600, () => { mark([3], 'a', BLUE); ptr.innerHTML = 'idx = 3 &nbsp; | &nbsp; res = 1 &nbsp; | &nbsp; idx = 4'; msg.innerHTML = texts[3]; U.markLines(document, []); });
    tl.at(15200, () => { mark([4, 5, 6], 'c', AMBER); ptr.innerHTML = 'idx = 4 &nbsp; | &nbsp; res = 3 &nbsp; | &nbsp; idx = 7'; msg.innerHTML = texts[4]; });
    tl.at(18600, () => { ptr.innerHTML = 'idx = 7'; msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 先过<span class="hl-b">并发判定</span>，才有资格谈融合 */
{
  kicker: "L6-08 · 由谁决定（一）",
  title: "先过<span class=\"hl-b\">并发判定</span>，才有资格谈融合",
  sub: "只有在\"这个节点能安全地和前面已编码的节点并发\"时，编码器才会去查融合表。",
  caption: "注意这一点：融合写的是整组最后一跳的 dst，所以并发性要按最后一跳重新判一次（第 236 行）。",
  src: "ggml/src/ggml-metal/ggml-metal-ops.cpp",
  mark: [11, 13, 15, 18, 22],
  lineNo: 218,
  code: `    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (is_concurrent && ctx->use_fusion()) {
            int n_fuse = 1;
            const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
            if (fusion) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml_mem_ranges_check(ctx->mem_ranges, ctx->node(idx + n_fuse - 1));
            }
        }

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="gate" style="gap:7px"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const g = wrap.querySelector('#gate');
    const steps = ['ggml_metal_op_concurrency_check()', 'ctx->use_fusion()', 'ctx->can_fuse(FULL)'];
    const gates = steps.map((s, i) => {
      const e = U.chip(s, i === 2 ? 'a' : '');
      g.appendChild(e);
      if (i < steps.length - 1) g.appendChild(U.arrow('&&'));
      return e;
    });
    gates.forEach(e => e.style.opacity = '.3');

    const defs = [
      { c: 'b', t: '并发判定的含义', b: '当前节点不能写任何此前 src/dst 的范围，<br>也不能读此前 dst 的范围（避免竞态）', w: '224px' },
      { c: 'd', t: '不许并发就下屏障', b: '不并发 -> <span class="cm" style="margin:0">ggml_metal_op_concurrency_reset()</span>，<br>同时也就<b>不会</b>尝试融合', w: '224px' },
      { c: 'a', t: '融合写最后一跳', b: '融合内核的输出是整组的<b>最后一个</b>节点，<br>所以要按 <span class="cm" style="margin:0">idx + n_fuse - 1</span> 重新判一次', w: '224px' }
    ];
    const chost = wrap.querySelector('#cards');
    const cards = defs.map(d => { const e = U.card(d); chost.appendChild(e); return e; });
    cards.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一幕回答"由谁决定"的第一半：<span class="k">没通过并发判定的节点，连查表的资格都没有</span>。',
      '第 229 行先做并发判定。这个判定和融合无关 —— 它是 Metal 为了把互不干扰的节点塞进同一个 command buffer 而做的内存范围分析。',
      '只有并发成立、并且融合总开关是开的（<span class="v">ctx->use_fusion()</span>，第 231 行），才会调 <span class="v">ctx->can_fuse(idx, FULL, &n_fuse)</span> 去查表。',
      '<span class="k">一个容易被忽略的细节</span>：融合内核的输出不是第一个节点的 dst，而是<b>最后一跳</b>的 dst。所以命中之后要用 <span class="v">idx + n_fuse - 1</span> 再判一次内存范围。',
      '如果并发不成立：清空内存范围（下屏障），这一轮就到此为止。融合不是"必须做的事"，而是<b>顺手能做的事</b>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, []); gates[0].style.opacity = '1'; });
    tl.at(3800, () => { msg.innerHTML = texts[1]; gates[0].style.opacity = '1'; cards[0].style.opacity = '1'; U.markLines(document, [11]); });
    tl.at(7400, () => { msg.innerHTML = texts[2]; gates[1].style.opacity = '1'; gates[2].style.opacity = '1'; U.markLines(document, [13, 15]); });
    tl.at(11000, () => { msg.innerHTML = texts[3]; cards[2].style.opacity = '1'; U.markLines(document, [18]); });
    tl.at(14200, () => { msg.innerHTML = texts[4]; cards[1].style.opacity = '1'; U.markLines(document, [22]); });
  }
},

/* ------------------------------------------------------ 7 决定权在这张 <span class="hl-c">26 条表项 / 9 种融合</span> 的表里 */
{
  kicker: "L6-08 · 由谁决定（二）",
  title: "决定权在这张 <span class=\"hl-c\">26 条表项 / 9 种融合</span> 的表里",
  sub: "ggml_metal_fusions 就是\"什么能融合\"的全部答案。数一数：26 条表项，归到 9 个融合 id。",
  caption: "表项第二个字段是完整原始序列 ops_all（含 VIEW/RESHAPE），第三个字段 outs 用来声明\"额外输出\"（MoE 路由要把专家下标也写出来）。",
  src: "ggml/src/ggml-metal/ggml-metal-fusion.cpp",
  mark: [1, 5, 9, 16, 17, 19, 24, 31],
  lineNo: 672,
  code: `static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     false, ggml_metal_fusion_check_norm },
//>> NORM + MUL：safe 模式，通用判据 + check_norm
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     false, ggml_metal_fusion_check_norm },
//>> RMS_NORM + MUL：同一个 fusion id —— 内核能同时干这两件事
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     false, ggml_metal_fusion_check_add_chain },
//>> ADD x 2；往下还有 ADD x 3..7，共 6 条表项
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     false, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     true,  ggml_metal_fusion_check_gdn_cache },
//>> GDN_CACHE 是 unsafe：它不走通用判据，由自己的 check 独家把关（写穿缓存，不是消除链）
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    true,  ggml_metal_fusion_check_topk_moe },
//>> TOPK_MOE 带 outs = {1}：融合内核要写两个输出（专家下标 + 路由权重）
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     false, ggml_metal_fusion_check_ssm_conv_silu },
//>> SSM_CONV + UNARY：很朴素的两节点融合
};`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['融合 id', '表里的 op 序列（ops）', '落到的 kernel（device.cpp）'],
      [['NORM_MUL', 'NORM + MUL ｜ RMS_NORM + MUL', 'kernel_norm_mul_f32 ｜ kernel_rms_norm_mul_f32'],
       ['NORM_MUL_ADD', 'NORM + MUL + ADD ｜ RMS_NORM + MUL + ADD', 'kernel_norm_mul_add_f32 ｜ kernel_rms_norm_mul_add_f32'],
       ['NORM_SCALE', 'NORM + SCALE ｜ RMS_NORM + SCALE', 'kernel_norm_mul_f32_use_scale ｜ kernel_rms_norm_mul_f32_use_scale'],
       ['ADD_CHAIN', 'ADD x N，N 在 [2, 7]（6 条表项）', 'kernel_bin_fuse_<t0>_<t1>_<t>_op=0_nf=N_rb=.._cb=..'],
       ['SNAKE', 'MUL + SIN + SQR + MUL + ADD', 'kernel_snake_<type>'],
       ['GDN_CACHE', 'GATED_DELTA_NET + CPY（写穿，unsafe）', '复用 kernel_gated_delta_net_<type>_<nsg>'],
       ['TOPK_MOE', 'SOFT_MAX + ARGSORT + GET_ROWS（+ norm 链 / + SCALE，4 条表项）', 'kernel_topk_moe_f32_n_expert=.._top_k=.._with_norm=..'],
       ['MOE_REDUCE', 'MUL + VIEW x k + ADD x k，2 到 8 专家（7 条表项）', 'kernel_moe_reduce_f32_n_expert_used=..'],
       ['SSM_CONV_SILU', 'SSM_CONV + UNARY(silu)', 'kernel_ssm_conv_<t0>_<t1>_nc=.._silu=1']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '26 条表项 = 9 种融合。表的每一行是 <span class="v">{ id, ops_all, outs, unsafe, check }</span>。',
      '<span class="k">NORM 家族（6 条）</span>：safe 模式 —— 走通用判据，再加一个 <span class="cm" style="margin:0">check_norm</span> 要求各跳的权重/偏置行连续、输出保持 F32。',
      '<span class="k">ADD_CHAIN（6 条）</span>：N 从 2 到 7。落到同一个 <span class="v">kernel_bin_fuse_*</span>，靠 pipeline 名里的 <span class="v">nf=</span> 区分要连加几跳。',
      '<span class="k">模型专用模式（SNAKE / GDN_CACHE / TOPK_MOE / MOE_REDUCE / SSM_CONV_SILU）</span>：不是通用逐元素链，而是把某个模型的固定子图一次算完。',
      '<span class="k">unsafe 标记（GDN_CACHE / TOPK_MOE / MOE_REDUCE）</span>：跳过通用"链式 + 同形 + 可消除"判据，由各自的 check 回调独家把关 —— 因为它们的中间节点<b>不是</b>典型的可消除链。',
      '<span class="k">outs 字段</span>：主输出是最后一个节点；<span class="v">outs = {1}</span> 表示"第 1 个节点也必须是输出"（MoE 路由要同时写出专家下标）。',
      '这张表就是答案：<b>融合由谁决定 —— 由后端维护的这张表决定</b>，而不是由某个 if 分支或某个 op 名字的拼接决定。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 6)];
    }));
    tl.at(22800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 8 ★ 命中的真判据：<span class="hl-a">连续、同形、可消除</span>，再加后端私有约束 */
{
  kicker: "L6-08 · ★ 判据",
  title: "★ 命中的真判据：<span class=\"hl-a\">连续、同形、可消除</span>，再加后端私有约束",
  sub: "表只声明\"模式长什么样\"；能不能真的融合，由这几道检查依次裁决。",
  caption: "ggml_can_fuse_subgraph_ext 的语义是\"子图之外没人用中间节点，所以它们可以被消除\"。",
  src: "ggml/src/ggml-metal/ggml-metal-fusion.cpp",
  mark: [0, 3, 5, 15, 19, 23, 39, 42, 44, 49, 50, 52, 56],
  lineNo: 1009,
  code: `        // the op sequence must match exactly
//>> ① op 序列必须逐位精确匹配表里的 ops（注意用的是过滤掉空节点后的 fusion.ops）
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
//>> ②③ safe 模式才有"链式 + 同形"两条约束；unsafe 模式整段跳过
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
//>> ④ 通用可消除判据：子图之外的消费者必须为空 —— 这一条来自 ggml 层，不是 Metal 私有
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == GGML_METAL_FUSION_FULL &&
//>> ⑥ 只有 FULL 模式才做：外部源与融合输出的内存别名检查（此刻张量已有地址）
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="list" class="col" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const checks = [
      ['①', 'op 序列精确匹配', 'nodes[j]->op != fusion.ops[j] 立即放弃 —— 表说什么就是什么'],
      ['②', '链式：每跳读上一跳', 'safe 模式要求 nodes[j]->src[0] 或 src[1] 就是 nodes[j-1]'],
      ['③', '同形：ggml_are_same_shape', '逐元素链上形状必须一致，否则偏移对不上'],
      ['④', '可消除：子图外无人使用中间结果', 'ggml_can_fuse_subgraph_ext() —— 这一条来自 ggml 层，不是 Metal 私有'],
      ['⑤', '模式自有的 check 回调', '例如 check_norm 要求权重行连续、输出 F32；unsafe 模式只有这一条'],
      ['⑥', '内存别名检查（仅 FULL）', '外部源不能与融合输出落在重叠地址，否则读写会打架']
    ];
    const host = wrap.querySelector('#list');
    const els = checks.map(c => {
      const e = U.el('div');
      e.style.cssText = 'display:flex;gap:7px;align-items:baseline;padding:4px 8px;border-left:3px solid var(--border);background:#10151b;border-radius:0 5px 5px 0';
      e.innerHTML = '<span style="color:var(--dim);font-size:10px">' + c[0] + '</span>' +
        '<span style="color:var(--text);font-size:10.5px;font-weight:600;flex:0 0 auto">' + U.esc(c[1]) + '</span>' +
        '<span style="color:var(--muted);font-size:9.5px">' + U.esc(c[2]) + '</span>';
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '表说完"能融合什么"，接下来是"这一次能不能融合"。<span class="k">六道检查，全过才算命中</span>。',
      '①②③ 是<b>逐元素链</b>的通用约束，只对 safe 模式生效 —— 源码把这三条放在 <span class="cm" style="margin:0">if (!fusion.unsafe)</span> 里面。',
      '④ 是关键的一条：<span class="v">ggml_can_fuse_subgraph_ext()</span>。它检查"中间节点的所有消费者都在子图内"，也就是<b>它们真的可以被消除</b>。',
      '注意 ④ 来自 ggml 层（<span class="cm" style="margin:0">ggml/src/ggml-impl.h</span>），不是 Metal 私有 —— Vulkan 也在用同一个判据（见 L6-07）。<b>通用判据在 ggml，模式与决策在后端</b>。',
      '⑤⑥ 是后端私有约束：<span class="v">fusion.check</span> 是 unsafe 模式的唯一裁判；<span class="v">check_memory_ranges</span> 只在 FULL 模式跑，因为只有那时才有地址。',
      '一句话：<span class="k">表定模式，check 定特例，ggml 定可消除性，模式枚举定严格程度</span> —— 四者合起来才是"由谁决定"。'
    ];
    els.forEach((e, i) => tl.at(700 + i * 3200, () => {
      els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; x.style.borderLeftColor = k === i ? 'var(--a)' : 'var(--border)'; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 5)];
    }));
    tl.at(20400, () => { els.forEach(e => { e.style.opacity = '1'; e.style.borderLeftColor = 'var(--border)'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 融合的最后一公里：<span class="hl-b">fusion id</span> 决定取哪个 pipeline */
{
  kicker: "L6-08 · 收束",
  title: "融合的最后一公里：<span class=\"hl-b\">fusion id</span> 决定取哪个 pipeline",
  sub: "命中之后，编码器告诉 device \"我要融合 N 跳\"，device 按名字取（或编译）对应的 kernel。",
  caption: "下一课 L6-09 讲这些 kernel 本身怎么写 —— 例如 kernel_norm_mul_add_f32 的三跳循环。",
  src: "ggml/src/ggml-metal/ggml-metal-device.cpp",
  mark: [8, 9, 11, 12, 18, 19, 20, 26, 27, 28],
  lineNo: 2034,
  code: `ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, const ggml_tensor * op, int n_fuse) {
    assert(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->ne[0] % 4 == 0) {
//>> ne[0] 能被 4 整除时走 float4 版本，名字加 _4 后缀
        suffix = "_4";
    }

    switch (op->op) {
        case GGML_OP_NORM:
            switch (n_fuse) {
//>> n_fuse = 1/2/3 分别对应不融合 / +MUL / +MUL+ADD —— 一个函数管三种 kernel
                case 1: snprintf(base, 256, "kernel_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        case GGML_OP_RMS_NORM:
//>> RMS_NORM 走同一套命名规则，只是前缀不同
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_rms_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案（都能在上面的代码里指出来）'],
      [['fusion 何时被触发？', '两次：优化阶段 ggml_metal_fusion_max()（STRUCTURAL），编码阶段 ctx->can_fuse()（FULL）'],
       ['由谁决定？', 'Metal 后端的融合表 ggml_metal_fusions（26 条 / 9 种 id）+ 模式 check + ggml 的 ggml_can_fuse_subgraph_ext'],
       ['图被改了吗？', '没有。优化阶段只是临时打包，重排完立刻解包（ggml-metal-common.cpp:487 起）'],
       ['落到哪个 kernel？', 'fusion id -> 编码器取 pipeline -> device 拼名字，如 n_fuse=3 的 RMS_NORM 得到 kernel_rms_norm_mul_add_f32'],
       ['怎么关掉它？', 'GGML_METAL_FUSION_DISABLE（device 初始化时读；另有 GGML_METAL_FUSION_DEBUG 开统计）']]);
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '同一张融合表被问了两次：分别在哪两个阶段、用什么模式、' +
      '为什么 <span class="mono">ggml_metal_fusion_check_moe_reduce()</span> 里的 ' +
      '<span class="mono">->data</span> 检查在其中一个阶段会被跳过？',
      '① <b>优化阶段</b>：调度器切分完图、做 graph_copy 之前调用 <span class="mono">graph_optimize</span> 钩子，' +
      'Metal 侧走 <span class="mono">ggml_graph_optimize()</span> -> <span class="mono">ggml_metal_fusion_max()</span>，' +
      '模式是 <span class="mono">GGML_METAL_FUSION_STRUCTURAL</span>。<br>' +
      '② <b>编码阶段</b>：<span class="mono">ggml_metal_op_encode_impl()</span> 里调 ' +
      '<span class="mono">ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &amp;n_fuse)</span>。<br>' +
      '跳过原因：STRUCTURAL 阶段张量<b>还没分配</b>，<span class="mono">->data</span> 可能还是空指针，' +
      '拿它做判据会误杀；源码因此把这段检查包在 ' +
      '<span class="mono">if (mode == GGML_METAL_FUSION_FULL)</span> 里。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '把这一课压成五个问答：',
      '<span class="k">触发</span>：两次查同一张表 —— 优化阶段定性、编码阶段定案。',
      '<span class="k">决定权</span>：表 + check + ggml 通用可消除判据；开关是环境变量。',
      '<span class="k">不变式</span>：图始终没被改过。融合只影响"编码几次"，不影响"图长什么样"。',
      '<span class="k">落点</span>：fusion id 最终变成一个 kernel 名字；这个名字是主机端拼出来的。',
      '对照 L6-01：CUDA 也融合（<span class="cm" style="margin:0">ggml-cuda.cu:4505</span> 同样有 graph_optimize 钩子），' +
      '但它的模式写在调用点的初始化列表上、两期各判各的（源码里还留着 TODO）；' +
      'Metal 用一张声明式表把两期统一。下一课 L6-09 看内核本身。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

];
