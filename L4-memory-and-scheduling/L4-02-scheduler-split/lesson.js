/* ==========================================================================
   L4-02 · ★ 调度器：把图切给不同后端
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 ★ <span class="hl-a">调度器</span>：把一张图切给多个后端 */
{
  kicker: "L4 · 内存与调度",
  title: "★ <span class=\"hl-a\">调度器</span>：把一张图切给多个后端",
  sub: "ggml_backend_sched 把三件事绑在一个对象上：节点归谁、段间怎么搬、每段的内存谁来分。",
  caption: "回顾 L3-02：后端数组的顺序就是优先级（头文件第 318 行写明）。回顾 L4-01：真正分配内存的是 galloc。下一课 L4-03 讲 buffer type。",
  src: "ggml/include/ggml-backend.h",
  mark: [0, 1, 2, 3, 4],
  lineNo: 266,
  code: `    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">一张计算图（拓扑序）</span><span class="arrow">-></span>
        <span class="chip a">归属：谁跑</span><span class="arrow">-></span>
        <span class="chip c">切分：断成连续段</span><span class="arrow">-></span>
        <span class="chip d">搬运：段间 copy</span><span class="arrow">-></span>
        <span class="chip b">逐段执行</span>
      </div>
      <div class="row center" id="cards1" style="gap:8px"></div>
      <div class="formula" id="msg1"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', w: '163px', t: '归属', m: 'ggml_backend_supports_op', b: '哪个后端能跑这个节点：<br>看 op，也看权重在哪个 buffer' },
      { c: 'c', w: '163px', t: '切分', m: 'ggml_backend_sched_split_graph', b: '沿拓扑序切成"连续的同后端段"，<br>每段是一个独立子图' },
      { c: 'd', w: '163px', t: '搬运', m: 'ggml_backend_tensor_copy', b: '段边界上的张量，要在下游<br>后端的 buffer 里再出现一份' },
      { c: 'b', w: '163px', t: '分配', m: 'ggml_gallocr_alloc_graph', b: '每段的张量由分配器按后端<br>落到各自 buffer（L4-01）' }
    ];
    const host = wrap.querySelector('#cards1');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg1');
    const texts = [
      '先看调度器在头文件里的自述 —— 三行字说清了它的职责范围。',
      '<span class="v">multiple backend devices to be used together</span>：'
        + '一个模型可以同时用 GPU、第二个 GPU、CPU。',
      '<span class="v">compute buffer allocation</span>：分配内存（这件事包给 L4-01 的 galloc）；<br>'
        + '<span class="v">assignment of tensors to backends</span>：决定每个张量归谁。',
      '<span class="v">copying of tensors between backends</span>：跨后端搬运。<br>'
        + '后面会看到：<span class="k">搬运的目标张量是切分时现场造出来的</span>。',
      '选后端的两个依据也写在注释里：<span class="k">后端支持这个 op</span> 和 '
        + '<span class="k">预分配张量（权重）在哪</span>。这两条就是下一幕的入口。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13900, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 <span class="hl-c">split</span> = 连续区间 + 输入清单；<span class="hl-c">sched</span> = 账本 */
{
  kicker: "L4-02 · 数据结构",
  title: "<span class=\"hl-c\">split</span> = 连续区间 + 输入清单；<span class=\"hl-c\">sched</span> = 账本",
  sub: "切分的结果不是一个新图，而是\"splits[] 数组 + 每个张量的后端 id\"。",
  caption: "splits 是动态数组：满了就翻倍（见 1347-1356）。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 3, 4, 5, 7, 10, 15, 16, 18, 20, 21, 22, 26, 27, 29, 32, 33, 40, 43, 45, 46],
  lineNo: 775,
  code: `struct ggml_backend_sched_split {
//>> 一个 split 有两句话：图上哪一段，以及这段要"吃"什么
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
//>> inputs[]：这一段开始前必须已经在本后端 buffer 里的张量
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
//>> graph：这一段自己的子图视图（i_start..i_end），执行时整段丢给后端
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;
//>> galloc：就是 L4-01 的分配器 —— 调度器自己不分配内存

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
//>> hv_tensor_backend_ids：张量 -> 后端 id，pass 1..4 填的就是它
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]
//>> hv_tensor_copies：(张量, 后端, 副本) -> 副本张量，跨段搬运的目标

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]
//>> leaf_backend_ids：叶子的归属（权重、图输入也在 leafs 里）

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
//>> splits[]：切分结果。每段一个连续区间，不是"节点的集合"
    int n_splits;
    int splits_capacity;`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl2"></div><div class="formula" id="msg2"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['字段', '含义', '谁写'],
      [['backends[] / bufts[]', '有序后端表 + 各自对应的 buffer type', 'sched_new'],
       ['hv_tensor_backend_ids', '张量 -> 后端 id', 'pass 1..4'],
       ['hv_tensor_copies', '(张量, 后端, 副本) -> 副本张量', 'pass 5'],
       ['splits[] / n_splits', '每段 = 连续区间 + 输入清单', 'pass 5'],
       ['galloc', '分配器句柄 —— 内存不归调度器管（L4-01）', 'sched_new'],
       ['op_offload', '是否允许把 op 抢到更靠前的后端', 'sched_new']],
      { monoCols: [0] });
    wrap.querySelector('#tbl2').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg2');
    const texts = [
      '调度器的全部状态就是这张表 —— 读源码时按这张表对号入座。',
      '<span class="v">backends[] / bufts[]</span>：后端是有序的，'
        + '<span class="k">下标就是优先级</span>（0 最高）。',
      '<span class="v">hv_tensor_backend_ids</span>：归属的答案存在这里。'
        + '它按张量指针哈希，所以整张图 5000 个节点也是一次哈希查找。',
      '<span class="v">hv_tensor_copies</span>：<span class="k">跨段搬运的"目标"在这里</span>。'
        + '一个源张量，在每个可能需要它的后端上，各有一份副本。',
      '<span class="v">splits[]</span>：切分结果。注意 <span class="k">i_start / i_end 是图上的下标</span>，'
        + '所以一个 split 必然是连续的一段。',
      '一句话：<span class="k">调度器只维护账本，不碰内存、不碰 kernel</span>。'
        + '内存是 galloc 的（L4-01），kernel 是各后端的（L4-04）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 3 ★ 归属优先级：<span class="hl-a">预分配 &gt; 视图 &gt; 图输入 &gt; 权重</span> */
{
  kicker: "L4-02 · 归属（pass 1）",
  title: "★ 归属优先级：<span class=\"hl-a\">预分配 &gt; 视图 &gt; 图输入 &gt; 权重</span>",
  sub: "backend_id_from_cur() 按固定顺序挑后端；一条都命中不了就返回 -1，交给后面的 pass。",
  caption: "挑不出来不是错误：中间张量本来就没有\"天生\"的后端，要靠 pass 2 从邻居那里扩展出来（见 1124-1170）。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 4, 5, 6, 10, 12, 13, 14, 15, 19, 21, 22, 23, 27, 29, 31, 32],
  lineNo: 921,
  code: `static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
//>> ① 已经有 buffer（权重、用户预分配）-> 直接用它的后端
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
//>> ② 视图张量跟着 view_src 走 —— 视图不拥有数据，数据在哪它就在哪
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
//>> 已经有 buffer 但这个后端跑不了这个 op：直接 abort，不会偷偷替它搬
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
//>> ③ 图输入默认落在最后一个后端，也就是 CPU
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
//>> 注释写死了约定：最后一个后端假定是 CPU
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
//>> ④ 都命中不了 -> -1，留给 pass 2 扩展 / pass 4 兜底`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards3" style="gap:8px"></div>
      <div class="formula" id="msg3"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', w: '163px', t: '① 预分配 buffer', m: '1.dst', b: '张量已经有 buffer：<br>在后端表里找支持它的' },
      { c: 'b', w: '163px', t: '② 视图', m: '1.vsrc', b: 'view_src != NULL：<br>归属 = view_src 的归属' },
      { c: 'c', w: '163px', t: '③ 图输入', m: '1.inp', b: 'GGML_TENSOR_FLAG_INPUT：<br>放最后一个后端（CPU）' },
      { c: 'd', w: '163px', t: '④ 权重', m: '1.wgt', b: 'src 是 WEIGHTS buffer：<br>跟着那个权重走' }
    ];
    const host = wrap.querySelector('#cards3');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg3');
    const texts = [
      'pass 1 只做一件事：给每个节点和每个叶子挑一个后端 id。顺序是固定的。',
      '<span class="v">① 预分配</span>：张量已经躺在某个 buffer 里（权重、用户自己分配的张量）。<br>'
        + '<span class="k">它不能被搬走</span>，所以只能找"支持这个 buffer type 又能跑这个 op"的后端。',
      '<span class="v">② 视图</span>：RESHAPE / VIEW 这类张量不拥有数据，'
        + '归属必须和 <span class="k">view_src</span> 一致，否则读到的就是别人地址空间里的指针。',
      '<span class="v">③ 图输入</span>：用户喂进来的张量默认放最后一个后端 —— '
        + '<span class="k">约定最后一个后端是 CPU</span>（第 946 行的注释）。',
      '<span class="v">④ 权重</span>：节点的输入里有 WEIGHTS buffer 时，节点跟着权重走 —— '
        + '这是最后一条，也是下一幕的主题。挑不出来就是 -1。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 权重在哪个 buffer，<span class="hl-b">op 就归哪个后端</span> */
{
  kicker: "L4-02 · 归属细节",
  title: "权重在哪个 buffer，<span class=\"hl-b\">op 就归哪个后端</span>",
  sub: "backend_from_buffer()：在有序后端表里找第一个\"既支持这个 buffer type、又支持这个 op\"的后端。",
  caption: "回顾 L2-03：权重加载时就决定了它落在哪个 buffer；这里只是把那个决定翻译成后端 id。下一课 L4-03 展开 buffer type 本身。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 1, 3, 5, 10, 11, 13, 14, 19, 20, 23],
  lineNo: 888,
  code: `static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
//>> 视图张量的 buffer 要看 view_src 的
    if (buffer == NULL) {
//>> 没有 buffer 的中间张量 -> -1，这一步答不了
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
//>> 注释：找优先级最高、且同时支持这个 buffer type 和这个 op 的后端
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
//>> 两个条件必须同时成立：supports_buft + supports_op
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
//>> 没有后端同时满足 -> -1（debug 模式下会打一行警告）
}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia4" style="gap:10px"></div>
      <div class="formula" id="msg4"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia4');
    const a = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
    a.innerHTML = '<div class="ct" style="color:var(--b)">输入：一个已经分配好的张量</div>' +
      '<div class="cb">它带着 buffer（例如权重 buffer）<br>' +
      '<span class="cm" style="margin:0">buffer-&gt;buft</span> 就是它所在的内存种类</div>';
    const b = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
    b.innerHTML = '<div class="ct" style="color:var(--a)">输出：后端 id</div>' +
      '<div class="cb">从 0 开始扫后端表，返回第一个<br>' +
      '<span class="cm" style="margin:0">supports_buft(x) &amp;&amp; supports_op(x)</span><br>都成立的下标</div>';
    dia.appendChild(a); dia.appendChild(U.arrow('->')); dia.appendChild(b);

    const msg = wrap.querySelector('#msg4');
    const texts = [
      '这是一个纯粹的小工具函数：把"张量在哪"翻译成"哪个后端 id"。',
      '<span class="v">supports_buft</span>：这个后端能不能在这个 buffer type 上干活'
        + '（显存 / pinned host / 普通内存，见 L4-03）。',
      '<span class="v">supports_op</span>：这个后端有没有实现这个算子。<br>'
        + '<span class="k">两个都成立才返回</span>，所以顺序里的第一个命中者就是答案。',
      '返回 <span class="v">-1</span> 有两种含义：张量还没分配（中间张量），'
        + '或者没有后端能接这个活。前者交给后面的 pass，后者在 debug 模式下会打警告。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 3, 5]); });
    tl.at(4000, () => { msg.innerHTML = texts[1]; U.markLines(document, [10]); });
    tl.at(7300, () => { msg.innerHTML = texts[2]; U.markLines(document, [11, 13, 14]); });
    tl.at(10600, () => { msg.innerHTML = texts[3]; U.markLines(document, [19, 20, 23]); });
  }
},

/* ------------------------------------------------------ 5 ★ <span class="hl-d">op_offload</span>：GPU 可以把 op 从 CPU 权重手里抢走 */
{
  kicker: "L4-02 · 归属细节",
  title: "★ <span class=\"hl-d\">op_offload</span>：GPU 可以把 op 从 CPU 权重手里抢走",
  sub: "权重在 host 内存时，只要更靠前的后端支持这个 op 且愿意 offload，op 就归它 —— 代价是数据得搬过去。",
  caption: "回顾 L2-05：dev_layer / n_gpu_layers 决定哪些层的权重放 GPU；这里决定剩下那些\"权重留在 host\"的层里，哪些 op 还能被抢上去。实战见 L8-02。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 3, 6, 7, 10, 11, 14, 15, 16, 20, 22, 23, 25, 27, 28, 30, 35, 36, 42],
  lineNo: 951,
  code: `    // operations with weights are preferably run on the same backend as the weights
//>> 注释：用权重的 op，优先和权重跑在同一个后端
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;
//>> allow 先设为 true —— 默认允许"跟着权重走"

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;
//>> 例外 1：ROPE 的 rope freqs 张量太小，不足以决定归属

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;
//>> 例外 2：FLASH_ATTN_EXT 的 sinks 张量太小

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
//>> 只对 usage == WEIGHTS 的输入这么做
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
//>> 注释：看看有没有优先级更高、又愿意 offload 这个 op 的后端
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
//>> 三个条件：op_offload 打开、权重在 host、默认归属是最后一个后端
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
//>> supports_op + offload_op 都通过 -> 直接返回这个更靠前的后端
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
//>> 抢不到就回到权重所在的后端
            }
        }
    }

    return -1;
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards5" style="gap:8px"></div>
      <div class="formula" id="msg5"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'd', w: '220px', t: '抢的三个条件', b: '① sched-&gt;op_offload 打开<br>② 权重在 host buffer 里<br>③ 更靠前的后端 supports_op 且 offload_op' },
      { c: 'e', w: '220px', t: '两个例外', b: 'ROPE：rope freqs 太小<br>FLASH_ATTN_EXT：sinks 太小<br>小张量不足以决定 op 的归属' },
      { c: 'c', w: '220px', t: '抢不到就回落', b: '返回权重所在的后端 id<br>（通常是 CPU）<br>此时 op 和权重在一起，不用搬权重' }
    ];
    const host = wrap.querySelector('#cards5');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg5');
    const texts = [
      '权重在 host 上，不代表 op 一定要在 CPU 上算 —— 这一段就是"抢 op"的逻辑。',
      '<span class="v">src_backend_id == n_backends - 1</span>：先算出"按权重该在哪个后端"，'
        + '如果它正好是最后一个（CPU）……',
      '……并且权重在 <span class="v">host</span> 内存里，'
        + '<span class="k">就从 0 开始找有没有更靠前的后端愿意接</span>（for b &lt; src_backend_id）。',
      '注意 <span class="v">offload_op</span> 是后端自己的意见：'
        + '有些后端对"值不值得"有判断（比如 op 太小、搬运不划算），它会说不。',
      '抢到了会怎样？归属只决定"谁来算"，<span class="k">数据怎么到位是 pass 5 的事</span>：'
        + '这个 host 上的权重会在边界被造一份副本搬过去（第 12 节）。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14700, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 ★ 切分 = 沿图找<span class="hl-a">连续的同后端段</span> */
{
  kicker: "L4-02 · pass 5",
  title: "★ 切分 = 沿图找<span class=\"hl-a\">连续的同后端段</span>",
  sub: "遍历拓扑序：节点后端和当前段不同就断一刀；节点带着\"别的后端的权重\"而本段又用不了它时，也断一刀。",
  caption: "断点处旧段写 i_end = i，新段从 i 开始 —— 所以 i 属于新段：段永远是图上的连续区间，不会跳着挑节点。视图节点直接跳过不参与切段。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 3, 5, 8, 11, 13, 15, 17, 22, 24, 26, 28, 30, 37, 39, 41, 52, 53, 55, 57],
  lineNo: 1313,
  code: `        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
//>> 视图算子不参与切段 —— 它不产生数据，且归属必然和 view_src 相同
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);
//>> 每个节点在这里读出归属 —— pass 1..4 已经保证它不等于 -1

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
//>> 只有当"本节点属于当前段"且"当前段已经有输入"时才检查权重
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
//>> 注释：权重在别的、且不兼容的后端上
                    // by starting a new split, the memory of the previously offloaded weights can be reused
//>> 注释：断一刀，上一段 offload 的权重内存就能早点还回去
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
//>> 权重在别的后端 + 本段不支持它的 buffer type -> 断段
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
//>> 真正的断段条件：后端换了，或者上面那个权重原因
                split->i_end = i;
//>> 旧段在这里收尾（i_end 不含 i）
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
//>> 新段从这个节点开始
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="formula" id="dia6" style="padding:8px 9px"></div>
      <div class="row center" id="cards6" style="gap:8px"></div>
      <div class="formula" id="msg6"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia6');
    dia.innerHTML = '<div class="cm" style="margin:0 0 6px">示意：一张 6 节点的小图（注意力 -&gt; FFN），'
      + 'n1 的权重在显存，n3 的权重在 host，且这次没把 op offload 出去</div>'
      + '<div class="row" id="segs" style="gap:6px;align-items:center;flex-wrap:wrap"></div>';
    const segs = dia.querySelector('#segs');

    const box = (id, op, c) => {
      const e = U.el('div', { style: 'width:62px;padding:3px 4px;border-radius:5px;'
        + 'border:1px solid var(--border);background:#10151b;text-align:center' });
      e.innerHTML = '<div style="font-family:var(--mono);font-size:8.5px;color:var(--' + c + ')">' + id + '</div>'
        + '<div style="font-size:8px;color:var(--muted)">' + op + '</div>';
      return e;
    };
    const segBox = (title, c, nodes) => {
      const e = U.el('div', { class: 'col', style: 'gap:3px;padding:4px 5px;border:1px dashed var(--' + c + ');border-radius:6px' });
      e.appendChild(U.el('div', { style: 'font-size:8px;color:var(--' + c + ')', text: title }));
      const row = U.el('div', { class: 'row', style: 'gap:3px' });
      nodes.forEach(n => row.appendChild(n));
      e.appendChild(row);
      return e;
    };
    const s0 = segBox('split #0 · GPU', 'a', [box('n0', 'norm', 'a'), box('n1', 'matmul wq', 'a'), box('n2', 'softmax', 'a')]);
    const s1 = segBox('split #1 · CPU', 'c', [box('n3', 'matmul wf', 'c'), box('n4', 'add', 'c')]);
    const s2 = segBox('split #2 · GPU', 'd', [box('n5', 'matmul wo', 'd')]);
    const cp1 = U.chip('copy', 'e'), cp2 = U.chip('copy', 'e');
    segs.appendChild(s0); segs.appendChild(cp1); segs.appendChild(s1); segs.appendChild(cp2); segs.appendChild(s2);

    const cards = wrap.querySelector('#cards6');
    const c1 = U.card({ c: 'a', w: '300px', t: '断段判据 1：后端换了',
      b: 'node_backend_id != cur_backend_id<br>图是拓扑序，所以同后端的节点天然连成一段' });
    const c2 = U.card({ c: 'c', w: '300px', t: '断段判据 2：权重在别的后端',
      b: '输入是 WEIGHTS buffer，且 src_backend_id != cur_backend_id，而本段又用不了它的 buffer type' });
    cards.appendChild(c1); cards.appendChild(c2);
    [c1, c2].forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg6');
    const texts = [
      '先把一张小图摆出来：n0..n2 在 GPU，n3..n4 因为权重在 host 而留在 CPU，n5 又回到 GPU。',
      '<span class="v">判据 1</span>：走到 n3 时 node_backend_id（CPU）!= cur_backend_id（GPU）—— '
        + '旧段在 n3 前收尾，新段从 n3 开始。',
      '<span class="v">判据 2</span>：就算两个节点的后端相同，只要某个输入是"别的后端上的权重"、'
        + '而本段后端用不了那个 buffer type，也要断开。',
      '为什么要为权重特意断一刀？<span class="k">注释给了理由</span>：'
        + '断段之后，上一段 offload 上去的权重内存可以提前还给分配器（L4-01 复用）。',
      '于是切分结果就是 3 个段：<span class="v">[n0,n1,n2]</span>、'
        + '<span class="v">[n3,n4]</span>、<span class="v">[n5]</span>，以及 2 个边界。',
      '边界上那两个 <span class="v">copy</span> 不是图里的节点 —— 下一幕看它们是怎么被造出来的。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4300, () => { msg.innerHTML = texts[1]; s0.style.borderColor = 'var(--a)'; });
    tl.at(7900, () => { msg.innerHTML = texts[2]; s1.style.borderColor = 'var(--c)'; });
    tl.at(11500, () => { msg.innerHTML = texts[3]; });
    tl.at(15100, () => { msg.innerHTML = texts[4]; s2.style.borderColor = 'var(--d)'; });
    tl.at(18700, () => { msg.innerHTML = texts[5]; [c1, c2].forEach(e => e.style.opacity = '1');
      [cp1, cp2].forEach(e => e.style.borderColor = 'var(--e)'); });
  }
},

/* ------------------------------------------------------ 7 ★ <span class="hl-a">copy 是切分的产物</span>，不是图里预先有的节点 */
{
  kicker: "L4-02 · pass 5 核心",
  title: "★ <span class=\"hl-a\">copy 是切分的产物</span>，不是图里预先有的节点",
  sub: "输入不在本段后端、本段后端又用不了它的 buffer 时：造一个本后端的副本张量，把 node->src[j] 改指向它。",
  caption: "注意：调度器没有往图里插 GGML_OP_CPY 节点 —— 它只换了指针；真正的搬运发生在 compute_splits（下一幕）。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 4, 6, 8, 10, 16, 20, 24, 25, 28],
  lineNo: 1399,
  code: `                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
//>> 判据：src 在别的后端，且本段后端不支持它的 buffer type
                    // create a copy of the input in the split's backend
//>> 注释：在本段后端造一份输入的副本
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
//>> 每个 (张量, 后端) 只造一次 —— tensor_id_copy 就是缓存
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
//>> ggml_dup_tensor_layout：同形状同类型的新张量
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
//>> 名字里带上后端与副本号，debug 打印时一眼看出
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
//>> 副本按 n_copies 各存一份（并行流水线时才 >1）
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
//>> 把【源】张量登记进 split->inputs
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
//>> 执行时按这个清单搬运
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
//>> 关键一行：节点的输入指针被改成副本 —— 图被改写了
                }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia7" style="gap:10px"></div>
      <div class="formula" id="msg7"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia7');
    const before = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--c)' });
    before.innerHTML = '<div class="ct" style="color:var(--c)">切分前</div>' +
      '<div class="cb">n3（CPU 段）的 src[0] 指向<br>n2 的输出 —— 它在 GPU 的 buffer 里<br>' +
      '<span class="cm" style="margin:0">n3.src[0] -&gt; GPU buffer</span></div>';
    const after = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
    after.innerHTML = '<div class="ct" style="color:var(--a)">切分后</div>' +
      '<div class="cb">同一个 n3，src[0] 指向本段的副本张量<br>它分配在 CPU 段的 buffer 里<br>' +
      '<span class="cm" style="margin:0">n3.src[0] -&gt; CPU#&lt;n2&gt;#0</span></div>';
    const ar = U.arrow('->');
    dia.appendChild(before); dia.appendChild(ar); dia.appendChild(after);

    const msg = wrap.querySelector('#msg7');
    const texts = [
      '先记住结论：<span class="k">图里的节点数没变</span>，变的是某些节点的 src[] 指针。',
      '<span class="v">判据</span>：src 在别的后端，而且本段后端不支持它的 buffer type'
        + '（supports_buft 为假）。换句话说：<span class="k">本段的内核读不到那块内存</span>。',
      '<span class="v">动作 1</span>：ggml_dup_tensor_layout 造一个同形状的副本张量 —— '
        + '它会被分配器放进本段后端的 buffer（L4-01 按 node_backend_ids 决定）。',
      '<span class="v">动作 2</span>：把源张量登记进 split-&gt;inputs。'
        + '这份清单就是"这一段开始前要搬什么"的待办列表。',
      '<span class="v">动作 3</span>：node-&gt;src[j] = 副本。'
        + '从这一刻起，本段的子图只看得见自己后端上的张量 —— 所以它才敢整段丢给一个后端。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; ar.textContent = '切分'; });
    tl.at(4000, () => { msg.innerHTML = texts[1]; before.style.borderColor = 'var(--a)'; });
    tl.at(7300, () => { msg.innerHTML = texts[2]; });
    tl.at(10600, () => { msg.innerHTML = texts[3]; });
    tl.at(13900, () => { msg.innerHTML = texts[4]; after.style.borderColor = 'var(--a)';
      ar.textContent = '->'; ar.style.color = 'var(--e)'; });
  }
},

/* ------------------------------------------------------ 8 切完之后：<span class="hl-e">galloc</span> 按段分配内存 */
{
  kicker: "L4-02 · 分配",
  title: "切完之后：<span class=\"hl-e\">galloc</span> 按段分配内存",
  sub: "alloc_graph() = split_graph() + alloc_splits()：内存分配整个包给 L4-01 的 galloc。",
  caption: "只有后端归属（或 buffer type）变了，才会重新 reserve；否则每步推理复用同一份内存（见 1591-1641）。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 3, 6, 8, 10, 13, 15, 18, 21],
  lineNo: 1992,
  code: `bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);
//>> 同一个图不能分配两次（is_alloc 断言）

    sched->cur_copy = sched->next_copy;
//>> 并行模式下轮转副本号：cur_copy / next_copy
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);
//>> 第一步：切图（pass 1..5）

    if (!ggml_backend_sched_alloc_splits(sched)) {
//>> 第二步：分配 —— 内部把 sched->graph 交给 galloc
        return false;
    }

    sched->is_alloc = true;
//>> 分配完成。之后 graph_compute 发现 is_alloc 为真就直接跑

    return true;
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip c">split_graph()</span><span class="arrow">-></span>
        <span class="chip a">alloc_splits()</span><span class="arrow">-></span>
        <span class="chip e">ggml_gallocr_alloc_graph()</span><span class="arrow">-></span>
        <span class="chip b">每段的 buffer 就位</span>
      </div>
      <div class="row center" id="cards8" style="gap:8px"></div>
      <div class="formula" id="msg8"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'e', w: '220px', t: '分配器怎么知道放哪', b: 'alloc_splits 把 node_backend_ids / leaf_backend_ids 交给 gallocr（1591-1641）' },
      { c: 'b', w: '220px', t: '什么时候重新分配', b: '后端 id 变了、或 buffer type 变了<br>才 reserve；否则复用' },
      { c: 'd', w: '220px', t: 'reserve 是什么', b: '用一张"最大 batch"的测量图先切一遍，<br>把每段的 buffer 预留出来（1975）' }
    ];
    const host = wrap.querySelector('#cards8');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg8');
    const texts = [
      'alloc_graph 是"要开始跑了"的入口。它只有两步，但第一步是整课的重点。',
      '<span class="v">第一步 split_graph()</span>：重新算归属、重新切段、重新造副本。'
        + '每换一张图（哪怕只是 batch 变了）都要重来一遍。',
      '<span class="v">第二步 alloc_splits()</span>：把带后端 id 的图交给 galloc —— '
        + '<span class="k">L4-01 的 arena 按这些 id 决定每段的张量落在哪个 buffer</span>。',
      '<span class="v">cur_copy / next_copy</span>：并行（流水线）模式下，每次 alloc 换一份副本，'
        + '让上一次计算还在跑的时候，下一次就能写另一份输入。',
      '分配完之后 <span class="v">is_alloc</span> 为真，graph_compute 就不再重复切分与分配 —— '
        + '这是解码循环里"每步只算一次"的关键。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 真搬运发生在 <span class="hl-c">compute_splits</span>：先搬 inputs，再跑子图 */
{
  kicker: "L4-02 · 执行",
  title: "真搬运发生在 <span class=\"hl-c\">compute_splits</span>：先搬 inputs，再跑子图",
  sub: "每个 split：优先用后端的异步 copy 把 inputs 搬进本后端，然后把 split->graph 整段交给这个后端。",
  caption: "回顾 L4-04：ggml_backend_graph_compute_async 是单后端的执行入口，调度器只是逐段调用它。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 1, 4, 6, 8, 11, 13, 21],
  lineNo: 1782,
  code: `                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
//>> 先试异步 copy —— 它会排在源后端已有的工作之后
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
//>> 后端没有异步接口，或者它拒绝了这次拷贝 -> 退回同步
                        ggml_backend_synchronize(input_backend);
//>> 同步路径要自己保证顺序：先同步源后端
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
//>> 真正的数据搬运：src 后端 -> 本段后端的副本
                    }
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
//>> inputs 都就位后，整段子图交给本段后端执行`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">段 A 算完</span><span class="arrow">-></span>
        <span class="chip c">record event</span><span class="arrow">-></span>
        <span class="chip d">搬 inputs 到段 B</span><span class="arrow">-></span>
        <span class="chip b">graph_compute_async(段 B 的子图)</span>
      </div>
      <div class="row center" id="cards9" style="gap:8px"></div>
      <div class="formula" id="msg9"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', w: '220px', t: '同步靠事件', b: '每个 (后端, 副本) 一个 event：<br>搬之前等它，跑完记它（1838）' },
      { c: 'd', w: '220px', t: '优先异步搬运', b: 'cpy_tensor_async(src, dst, ...)<br>不行才退回同步 copy（1785）' },
      { c: 'c', w: '220px', t: '段内不再切分', b: '搬完之后，这一段就是一整个<br>后端自己认识的标准子图' }
    ];
    const host = wrap.querySelector('#cards9');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg9');
    const texts = [
      '执行阶段是一个 for 循环：按切分顺序，一段一段地跑（1656）。',
      '<span class="v">先搬 inputs</span>：每个 split 都带着自己的输入清单，'
        + '清单里的每一个张量都要在段开始前落到本后端的 buffer 里。',
      '<span class="v">异步优先</span>：能异步就异步（拷贝排在源后端的队列里），'
        + '不能就同步搬 —— <span class="k">无论哪条路，段开始前数据一定就位</span>。',
      '<span class="v">再跑子图</span>：ggml_backend_graph_compute_async(split_backend, &amp;split-&gt;graph)。'
        + '<span class="k">注意这里传的是 split-&gt;graph，不是原始整图</span>。',
      '这就是切分的最终收益：每个后端只看到"自己的那一段 + 已经搬好的输入"，'
        + '不需要理解别的设备。下一课 L4-04 讲这个入口内部怎么跑。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表 */
{
  kicker: "L4-02 · 收束",
  title: "把这一课压成一张表",
  sub: "入口只有三个：split_graph（切）、alloc_graph（分配）、graph_compute（执行）。",
  caption: "下一课 L4-03：buffer 与 buffer type —— 决定\"数据放在哪种内存里\"，也就是本课所有 buffer type 判据的另一半。",
  src: "ggml/include/ggml-backend.h",
  mark: [0, 1, 3, 4, 5, 6, 7],
  lineNo: 339,
  code: `    // Split graph without allocating it
    GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    // Allocate and compute graph on the backend scheduler
    GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // returns success
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl10"></div><div id="ex10"></div><div class="formula" id="msg10"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['阶段', '函数', '留下的数据', '本课行号'],
      [['归属', 'ggml_backend_sched_backend_id_from_cur', 'hv_tensor_backend_ids', '921-985'],
       ['扩展', 'pass 2 的四个循环', 'node_backend_ids', '1124-1202'],
       ['切段', 'ggml_backend_sched_split_graph', 'splits[].i_start / i_end', '1324-1362'],
       ['造副本', 'ggml_dup_tensor_layout', 'hv_tensor_copies / split.inputs', '1399-1420'],
       ['分配', 'ggml_backend_sched_alloc_graph', 'galloc 的 arena（L4-01）', '1591-1641 / 1992-2009'],
       ['搬运+执行', 'ggml_backend_sched_compute_splits', '每个 split 的输入清单', '1782-1799']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl10').appendChild(t.el);

    wrap.querySelector('#ex10').appendChild(W.exercise(
      '一张图的拓扑序是 n0(GPU) -&gt; n1(GPU) -&gt; n2(CPU) -&gt; n3(CPU) -&gt; n4(GPU)。'
      + '调度器会切出几个 split？在哪里产生副本张量？',
      '3 个 split：<span class="mono">#0 = [n0, n1]</span> 在 GPU，<span class="mono">#1 = [n2, n3]</span> 在 CPU，'
      + '<span class="mono">#2 = [n4]</span> 在 GPU。<br>'
      + '边界由 <span class="mono">node_backend_id != cur_backend_id</span> 触发：旧段写 <span class="mono">i_end = i</span>，'
      + '新段 <span class="mono">i_start = i</span>（1344-1359）。<br>'
      + '两处边界各造一个副本张量：n1 的输出要在 CPU 段的 buffer 里有一份，n3 的输出要在 GPU 段的 buffer 里有一份'
      + '（<span class="mono">ggml_dup_tensor_layout</span> + <span class="mono">node-&gt;src[j] = 副本</span>，1399-1419）。<br>'
      + '注意：图里的节点数没变，还是 5 个 —— 变的只是 n2 和 n4 的 <span class="mono">src[]</span> 指针。'));

    const msg = wrap.querySelector('#msg10');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '六个阶段，一条流水线：归属 -> 扩展 -> 切段 -> 造副本 -> 分配 -> 执行。',
      '<span class="k">归属</span>：权重 buffer、视图、图输入三条硬规则，加上 op_offload 的例外。',
      '<span class="k">切段</span>：沿拓扑序把同后端的节点连成连续区间；权重不兼容时特意断开，好让内存早点回收。',
      '<span class="k">造副本</span>：copy 不是图里预置的节点，而是切分时按需造出来的目标张量。',
      '<span class="k">分配 + 执行</span>：galloc 按后端 id 落 buffer（L4-01），compute_splits 逐段搬运并执行（L4-04）。',
      '验收点：<span class="v">段只由一个后端执行，它的 kernel 只认自己 buffer type 上的内存</span> —— '
        + '所以边界上必须把数据搬到下游后端的 buffer 里，这个"搬"就是副本张量存在的理由。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(16500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

];
