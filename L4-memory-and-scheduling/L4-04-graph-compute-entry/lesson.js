/* ==========================================================================
   L4-04 · 图执行入口：graph_compute 与异步
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一个入口，两条路径：<span class="hl-a">提交</span> 与 <span class="hl-b">完成</span> */
{
  kicker: "L4 · 内存与调度",
  title: "一个入口，两条路径：<span class=\"hl-a\">提交</span> 与 <span class=\"hl-b\">完成</span>",
  sub: "两个函数体只差一行 —— 那一行就是\"等待\"。",
  caption: "本课验收点：能说出同步与异步路径分别在哪一步等待完成。相关课：L1-03（按 nodes[] 顺序）、L2-07（llama_decode 的调用点）、L4-02（切分）、L5-01 / L6-01（后端怎么实现）。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [2, 3, 10],
  lineNo: 455,
  code: `enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
//>> 同步入口：先异步提交，再同步等待，最后才把状态返回给调用者
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
//>> 异步入口：只有一个转发，没有任何等待
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">调用者</span><span class="arrow">-></span>
        <span class="chip a">graph_compute</span><span class="arrow">或</span>
        <span class="chip b">graph_compute_async</span><span class="arrow">-></span>
        <span class="chip">backend->iface.graph_compute</span><span class="arrow">-></span>
        <span class="chip c">CPU / CUDA / ...</span>
      </div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '同步入口 · 第 455-459 行', b: '调用 async 之后，<b>紧接着</b>调用 ggml_backend_synchronize()，然后才返回。<br>对调用者而言：返回 == 已经算完。', m: 'ggml_backend_graph_compute()' },
      { c: 'b', t: '异步入口 · 第 461-464 行', b: '只做一次转发：backend->iface.graph_compute()。<br>对调用者而言：返回 == 只是提交了。', m: 'ggml_backend_graph_compute_async()' },
      { c: 'c', t: '真正的执行者：后端接口', b: '同一段代码在 CPU 上跑和在 CUDA 上跑，差别全在 iface 的实现里。<br>见 L5-01 与 L6-01。', m: 'struct ggml_backend_i' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '本课只问一件事：<span class="k">一次执行请求从"提交"到"完成"，谁在等、在哪一行等</span>。',
      '同步入口（第 455 行）比异步入口多出来的那一步，<b>就是等待</b>：第 457 行。',
      '异步入口（第 461 行）没有等待：第 463 行把活交出去就返回。',
      '所以"异步"不是让后端跑得更快，而是把"完成"这件事交给别处的同步点 —— 事件、调度器收尾、或读回数据时。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
      U.markLines(document, [[1], [1], [2], []][i]);
    }));
  }
},

/* ------------------------------------------------------ 2 等待发生在 <span class="hl-c">iface.synchronize</span>，不在这里 */
{
  kicker: "L4-04 · 同步点",
  title: "等待发生在 <span class=\"hl-c\">iface.synchronize</span>，不在这里",
  sub: "这一层的全部内容是\"断言 + 判空 + 转发\"：后端没实现 synchronize，它就立刻返回。",
  caption: "对照：L5-01 的 ggml/src/ggml-cpu/ggml-cpu.cpp:201 写着 .synchronize = NULL；L6-01 的 ggml/src/ggml-cuda/ggml-cuda.cu:2544 ggml_backend_cuda_synchronize() 里是 cudaStreamSynchronize()。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [2, 4, 7],
  lineNo: 425,
  code: `void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
//>> 判空：后端可以完全不实现 synchronize —— 那就没有"异步未完成"这回事
        return;
    }

    backend->iface.synchronize(backend);
//>> 真正的等待：这一行会阻塞到后端的队列/流排空
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: '这一层只有三步', b: '断言、判空、转发。<br>没有循环、没有轮询、没有线程。<br>它<b>自己不等</b>，只是把"等"这件事交给后端。', m: '第 425-432 行' },
      { c: 'a', t: 'CPU：synchronize == NULL', b: 'L5-01：ggml-cpu.cpp:201 的 .synchronize 是 NULL。<br>判空命中第 427-429 行，<b>直接返回</b>。', m: 'ggml-cpu.cpp:201' },
      { c: 'b', t: 'CUDA：cudaStreamSynchronize', b: 'L6-01：ggml-cuda.cu:2544 等整个 stream 排空。<br>这一行才是 GPU 场景里 host 真正停住的地方。', m: 'ggml-cuda.cu:2544' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '第 431 行：<span class="k">backend->iface.synchronize(backend)</span>。同步路径的"等"，最终等在这里。',
      '但这一层是<b>可选的</b>：后端把自己的 synchronize 留成 NULL，第 427-429 行就变成一次空返回。',
      'CPU 就是这种情况：等待没有消失，而是<b>压根不需要</b> —— CPU 的 graph_compute 返回时已经算完（L5-01）。',
      '结论：问"同步路径在哪一步等"，必须先问"<span class="v">这个后端的 iface.synchronize 是什么</span>"。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
      U.markLines(document, [[2], [0, 1], [2], []] [i]);
    }));
  }
},

/* ------------------------------------------------------ 3 事件：把"完成"变成可等待的<span class="hl-d">句柄</span> */
{
  kicker: "L4-04 · 事件",
  title: "事件：把\"完成\"变成可等待的<span class=\"hl-d\">句柄</span>",
  sub: "record 在一次提交之后盖章；wait 让另一个后端等，synchronize 让 host 等。",
  caption: "注意 event_new 收的是 device，record / wait 收的是 backend —— 事件是设备级对象，能被不同后端共同引用。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 9, 16, 24, 31],
  lineNo: 534,
  code: `ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
//>> 设备不实现 event_new 时返回 NULL —— 调用方必须接受"没有事件"这种情况
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
//>> record：在当前后端已提交的工作后面插一个标记
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
//>> wait：只让这个后端等那个标记，host 不被阻塞
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip d">event_new(device)</span><span class="arrow">-></span>
        <span class="chip">event_record(backend)</span><span class="arrow">-></span>
        <span class="chip b">event_wait(backend)</span><span class="arrow">/</span>
        <span class="chip a">event_synchronize()</span><span class="arrow">-></span>
        <span class="chip">event_free()</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="card grow" style="border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">两个等待主体，别混</div>
          <div class="cb"><b>event_wait(backend, event)</b>（第 563-568 行）：让这个后端等，host 继续往下跑。<br>
          <b>event_synchronize(event)</b>（第 556-561 行）：让 host 等，直到标记被跨过。</div>
        </div>
        <div class="card grow" style="border-left-color:var(--c)">
          <div class="ct" style="color:var(--c)">事件可以是 NULL</div>
          <div class="cb">event_new 在设备没实现时返回 NULL（第 536-538 行）。调度器每个用点都先判空，
          退化到 ggml_backend_synchronize（第 1664-1668、1787-1791 行）。<br>
          纯 CPU 或没有事件能力的后端走的就是这条退化路径。</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    const texts = [
      '没有事件时，"异步"只能靠 host 停下来对齐 —— 这就是事件存在的理由。',
      '<span class="v">event_new(device)</span>：设备级资源；没有实现就返回 NULL，调用方必须容错。',
      '<span class="v">event_record(backend)</span>：把标记插进这个后端已提交的工作流里，<b>不阻塞</b>。',
      '<span class="v">event_wait / event_synchronize</span>：前者等的是后端，后者等的是 host。',
      '于是"完成"变成一个<b>可以被传递的对象</b>：提交者盖章，消费者按需等待。',
      '调度器正是这样做的：每个 split 提交后盖章（第 1839 行），下一个 split 需要这块数据时才等（第 1688 行）。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, []); });
    tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [0, 1]); });
    tl.at(7200, () => { msg.innerHTML = texts[2]; U.markLines(document, [2]); });
    tl.at(10500, () => { msg.innerHTML = texts[3]; U.markLines(document, [3, 4]); });
    tl.at(14000, () => { msg.innerHTML = texts[4]; U.markLines(document, [0, 1, 2, 3, 4]); });
    tl.at(17500, () => { msg.innerHTML = texts[5]; U.markLines(document, [2, 3]); });
  }
},

/* ------------------------------------------------------ 4 调度器把同一对入口又写了一遍 */
{
  kicker: "L4-04 · 调度器",
  title: "调度器把同一对入口又写了一遍",
  sub: "ggml_backend_sched_graph_compute() = 异步版 + 对<b>所有</b>后端各等一次。",
  caption: "llama.cpp 自己走异步入口：src/llama-context.cpp:2588 调 _async，src/llama-context.cpp:772 用 sched synchronize 兜底。同步入口的用户是\"必须逐步看结果\"的场景，例如本文件第 2298 行的 ggml_backend_compare_graph_backend。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [1, 2, 8, 13, 20],
  lineNo: 2011,
  code: `enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
//>> 只有"没复位且没分配"时才复位 —— 否则沿用上一次的状态
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
//>> 已经分配过就直接跳过：切分与 buffer 都复用上一次的（graph reuse）
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['入口', '行号', '返回时意味着什么'],
      [['ggml_backend_sched_graph_compute', '2011-2015', '所有后端都已完成：先 async，再 ggml_backend_sched_synchronize'],
       ['ggml_backend_sched_graph_compute_async', '2017-2030', '只完成"提交"；若上次的分配合格，连切分与分配都跳过（2019-2027）'],
       ['ggml_backend_sched_synchronize', '2032-2043', '循环 sched->n_backends 次，逐个 ggml_backend_synchronize']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '调度器层不是新的执行路径，只是同一套"提交 / 等待"多了一层编排。',
      '<span class="v">第 2012 行</span> 调异步版，<span class="v">第 2013 行</span> 对<b>所有</b>后端同步 —— 这是 sched 级同步路径的全部内容。',
      '<span class="v">第 2029 行</span> 才是真正干活的地方：ggml_backend_sched_compute_splits()，按 split 逐个提交。',
      '<span class="v">第 2019-2027 行</span> 是两个短路：已经复位就不重复复位，已经分配就直接复用 —— 这是 graph reuse 的入口。',
      '一句话：<span class="k">同步版 = 异步版 + 一次"等所有后端"</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 3400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
      U.markLines(document, [[0, 1], [2], [4, 5]][i]);
    }));
    tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; U.markLines(document, [1, 2]); });
  }
},

/* ------------------------------------------------------ 5 ★ graph reuse：第二次跑同一张图，<span class="hl-b">不重切、不重分配</span> */
{
  kicker: "L4-04 · 缓存",
  title: "★ graph reuse：第二次跑同一张图，<span class=\"hl-b\">不重切、不重分配</span>",
  sub: "判据是\"每个节点的后端归属有没有变\"；没变就落回上一次的分配结果。",
  caption: "复用的最后一跳在 ggml_gallocr_alloc_graph() 内部（L4-01）：在预留好的块里重新指派地址，成功就完全不需要 reserve。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [1, 3, 10, 21],
  lineNo: 1591,
  code: `static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
//>> node_backend_ids 与 prev_* 比较：后端换了、且 buffer type 也换了，才算"变了"
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
//>> 没变就直接 alloc —— 命中复用路径，不会走到下面的 reserve`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">第一次</span><span class="arrow">-></span>
        <span class="chip a">切分 + reserve + alloc</span><span class="arrow">|</span>
        <span class="chip">第二次</span><span class="arrow">-></span>
        <span class="chip b">比较 backend ids</span><span class="arrow">-></span>
        <span class="chip c">ggml_gallocr_alloc_graph 复用</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="card grow" style="border-left-color:var(--a)">
          <div class="ct" style="color:var(--a)">什么才算"变了"</div>
          <div class="cb">两个条件同时成立才置位：后端编号不同 <b>且</b> buffer type 也不同（第 1594-1595 行）。<br>
          leafs 用同一套判据（第 1600-1608 行）。只换编号、buffer type 一样，仍然复用。</div>
        </div>
        <div class="card grow" style="border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">三层缓存，层层短路</div>
          <div class="cb">① 调度器：is_alloc 为真则跳过 reset 与 alloc（第 2019-2027 行）。<br>
          ② 切分：本幕的 id 比较。③ 显存：gallocr 在预留块里重新指派（L4-01）。</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    const texts = [
      '同一张图、同一组输入反复跑，是推理的常态 —— 每一步 decode 都要跑一遍图。',
      '所以调度器把"上次怎么切的、内存分到哪"整套留下：<span class="k">第 1594 行</span> 只比后端归属。',
      '判据必须保守：<span class="v">后端编号变了 且 buffer type 变了</span> 才重算（第 1594-1595 行）。',
      '真要重算时，第 1949 行的 ggml_backend_sched_reset() 会把 hash 表与后端映射清空，下一轮重新切分。',
      '前提是图本身能复用 —— 这正是 L2-07 里 llama_context 的判据（src/llama-context.cpp:1405 res->can_reuse(gparams)）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, []); });
    tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [1, 2]); });
    tl.at(7300, () => { msg.innerHTML = texts[2]; U.markLines(document, [1, 2, 3]); });
    tl.at(10700, () => { msg.innerHTML = texts[3]; U.markLines(document, [5, 6]); });
    tl.at(14300, () => { msg.innerHTML = texts[4]; U.markLines(document, [4]); });
  }
},

/* ------------------------------------------------------ 6 同步路径的最后一步：对<span class="hl-a">每个后端</span>都等一次 */
{
  kicker: "L4-04 · 同步点",
  title: "同步路径的最后一步：对<span class=\"hl-a\">每个后端</span>都等一次",
  sub: "ggml_backend_sched_synchronize() 循环所有后端，顺带把 next_copy 归零。",
  caption: "多副本字段：GGML_SCHED_MAX_COPIES = 4（第 772 行）、n_copies / cur_copy / next_copy（第 816-818 行）。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [2, 4, 6, 10],
  lineNo: 2032,
  code: `void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
//>> 注意是 n_backends 次，不是 1 次 —— 一张图可能横跨多个设备
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
//>> 同步之后固定用副本 0：让每轮 decode 的图结构完全一致
    }
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">GPU0</span><span class="chip a">GPU1</span><span class="chip a">CPU</span>
        <span class="arrow">-></span>
        <span class="chip b">for i in 0..n_backends</span>
        <span class="arrow">-></span>
        <span class="chip c">ggml_backend_synchronize(backends[i])</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="card grow" style="border-left-color:var(--a)">
          <div class="ct" style="color:var(--a)">为什么必须每个都等</div>
          <div class="cb">一次图执行被切成多段跑在不同设备上（L4-02）。<br>
          只等其中一个，别的设备可能还在写显存里那块被下一个 split 复用的 buffer。</div>
        </div>
        <div class="card grow" style="border-left-color:var(--c)">
          <div class="ct" style="color:var(--c)">next_copy = 0 是为了图缓存</div>
          <div class="cb">源码注释写明原因：副本轮转会让图结构变化，<b>可能导致 CUDA 等后端的图缓存被禁用</b>（第 2038-2040 行）。</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const msg = wrap.querySelector('#msg');
    const texts = [
      '调度器版的等待没有魔法：<span class="k">循环所有后端，每个都 synchronize 一次</span>。',
      '第 2034-2036 行：n_backends 次调用，缺一个就可能读到别的设备还没写完的 buffer。',
      '第 2037-2041 行：同步之后把 next_copy 固定回 0 —— 多副本是为流水线重叠准备的，',
      '但副本轮转会让每轮的图长得不一样，图缓存（如 CUDA graph）就失效了，所以同步点顺手把图"摆正"。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [2, 3]); });
    tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, [2, 3]); });
    tl.at(8100, () => { msg.innerHTML = texts[2]; U.markLines(document, [5, 6]); });
    tl.at(12100, () => { msg.innerHTML = texts[3]; U.markLines(document, [5, 6, 8]); });
  }
},

/* ------------------------------------------------------ 7 ★ 异步把"提交"与"完成"分开：等待只剩<span class="hl-c">这几处</span> */
{
  kicker: "L4-04 · 核心",
  title: "★ 异步把\"提交\"与\"完成\"分开：等待只剩<span class=\"hl-c\">这几处</span>",
  sub: "每个 split 用 _async 提交（第 1799 行），提交后立刻盖章（第 1839 行）；host 只在必要时才等。",
  caption: "第 1799 / 1839 行在本幕代码区之外（同一文件里可直接核对）。带 eval 回调时是例外：每个回调段提交后立刻 ggml_backend_synchronize（第 1821、1827 行）。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [2, 5, 7, 21, 30],
  lineNo: 1661,
  code: `        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
//>> 换后端、且这一段的输入不需要拷贝时：先等上一段算完（分配器可能复用了同一块 buffer）
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
//>> 用户的输入张量：必须立刻等，否则用户可能在拷贝完成前改写它
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
//>> 不是用户的输入：让这个后端等事件即可 —— host 不参与
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
`,
  duration: 23000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['何处', '调用', '谁在等'],
      [['第 1665 行', 'event_synchronize(上一后端的事件)', 'host —— 等上一段落定'],
       ['第 1667 行', 'ggml_backend_synchronize(上一后端)', 'host —— 没有事件对象时的退化路径'],
       ['第 1680 行', 'event_synchronize(本 split 后端)', 'host —— 用户输入要立刻拷走'],
       ['第 1688 行', 'event_wait(本后端, 事件)', '只有这个后端等，host 不停'],
       ['第 1799 行', 'graph_compute_async(split)', '不等 —— 提交'],
       ['第 1839 行', 'event_record(事件)', '不等 —— 盖章']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '异步路径的"等"不是被删掉了，而是被<b>推迟</b>到必要的那一刻。',
      '<span class="v">跨后端交接</span>（第 1663-1669 行）：换设备了，而这一段没有输入要拷 —— 只能等上一段落定。',
      '<span class="v">覆盖输入之前</span>（第 1679-1691 行）：分两种情况 —— 用户输入立刻等 host，内部输入只让后端等事件。',
      '<span class="v">提交与盖章</span>（第 1799、1839 行）：都不等待。这正是 L4-02 讲的"切分后每段独立提交"能重叠的前提。',
      '最后一道保险是收尾：ggml_backend_sched_synchronize()（第 2032-2043 行）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3200 + i * 3200, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i < 4 ? i + 1 : 3];
    }));
    tl.at(20500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; U.markLines(document, [0, 1, 2, 3, 4]); });
  }
},

/* ------------------------------------------------------ 8 执行入口已经不在 <span class="hl-e">ggml.c</span> 里了 */
{
  kicker: "L4-04 · ggml.c 侧",
  title: "执行入口已经不在 <span class=\"hl-e\">ggml.c</span> 里了",
  sub: "v0.5.0 实测：ggml.c 里 graph_compute 出现 0 次；ggml_graph_plan 搬到了 ggml-cpu.c。",
  caption: "实测命令：grep -c graph_compute ggml/src/ggml.c 得到 0；ggml_graph_plan 定义在 ggml/src/ggml-cpu/ggml-cpu.c:2815，ggml_graph_compute 定义在 ggml-cpu.c:3399（都在 L5-01）。",
  src: "ggml/src/ggml.c",
  mark: [0, 3, 5, 11],
  lineNo: 7526,
  code: `struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
//>> 节点数组只是指针偏移：不复制任何节点，这是"零拷贝子图"
        /*.grads            =*/ NULL, // gradients would need visited_hash_set
        /*.grad_accs        =*/ NULL,
        /*.leafs            =*/ NULL,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
//>> use_counts 与 visited_hash_set 直接沿用父图的
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    return cgraph;
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">ggml.c</span><span class="arrow">:</span>
        <span class="chip a">图的容器与视图</span>
        <span class="arrow">|</span>
        <span class="chip">后端</span><span class="arrow">:</span>
        <span class="chip b">图执行</span>
      </div>
      <div class="row" style="gap:8px" id="cards"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const defs = [
      { c: 'a', t: '图的容器', b: 'ggml_graph_nbytes（7451）/ overhead_custom（7469）/ new_graph_custom（7477）：<br>图在 arena 里占多少字节，在创建时就定死。' },
      { c: 'b', t: '图视图', b: 'ggml_graph_view（7526）：零拷贝切出"节点数组的一段"，<br>就是提交给后端的那张子图。' },
      { c: 'c', t: '复用与状态', b: 'ggml_graph_clear（7646）就地复位节点数与 hash 表；<br>ggml_status_to_string（438）翻译执行结果。' },
      { c: 'd', t: '搬走的执行入口', b: 'ggml_graph_plan -> ggml-cpu.c:2815；<br>ggml_graph_compute -> ggml-cpu.c:3399。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'ggml.c 留下的是<b>图的表示</b>：容器、视图、状态字符串。执行入口整体搬去了后端层。',
      '视图是调度器切 split 的工具：ggml-backend.cpp:1460 对每段 split 调 ggml_graph_view。',
      '一次只算一个节点时也用它：ggml-backend.cpp:2332 的 compare 路径 ggml_graph_view(g1, i, i+1)。',
      '回到本课主线：<span class="k">提交给 graph_compute 的，永远是这种"视图"</span>，而不是整张图。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i, 1)];
      U.markLines(document, [[0], [0, 1, 2, 3], [0], [0]][i]);
    }));
    tl.at(15500, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[2];
      U.markLines(document, [0]);
    });
    tl.at(18000, () => { msg.innerHTML = texts[3]; U.markLines(document, [2, 3]); });
  }
},

/* ------------------------------------------------------ 9 把"在哪一步等"压成一张表 */
{
  kicker: "L4-04 · 收束",
  title: "把\"在哪一步等\"压成一张表",
  sub: "同步路径在 ggml_backend_synchronize 里等；异步路径把等待推迟到跨后端交接与收尾。",
  caption: "下一课 L5-01 看 CPU 后端如何实现 graph_compute；L6-01 看 CUDA 如何把提交做成真异步；L1-03 的 nodes[] 顺序就是这里 split 循环的顺序。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [1, 2],
  lineNo: 455,
  code: `enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['路径', '入口', '在哪一步等待'],
      [['同步（单后端）', 'ggml_backend_graph_compute（455）', '第 457 行 -> ggml_backend_synchronize 第 431 行 -> iface.synchronize；后端可为 NULL（CPU）'],
       ['同步（调度器）', 'ggml_backend_sched_graph_compute（2011）', '第 2013 行 -> 第 2034-2036 行循环所有后端各等一次'],
       ['异步', 'ggml_backend_graph_compute_async（461 / 调度器 2017）', '不等；只在跨后端交接（1663）、覆盖用户输入（1679）、收尾 synchronize（2032）时等']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一次 <span class="mono">ggml_backend_sched_graph_compute_async()</span> 返回之后，host 到哪些地方才会真的等设备算完？' +
      'CPU 后端和 CUDA 后端的答案一样吗？',
      '三处会让 host 停住：<br>' +
      '① 执行循环里：跨后端交接（第 1665 / 1667 行）、拷贝用户输入之前（第 1680 行）；<br>' +
      '② 收尾：<span class="mono">ggml_backend_sched_synchronize()</span>（第 2032-2043 行，循环所有后端）；<br>' +
      '③ 读回数据时：<span class="mono">ggml_backend_tensor_get_async()</span> 在后端没有异步实现时会先 synchronize（第 286 行）。<br>' +
      '答案不一样：CPU 的 <span class="mono">.synchronize = NULL</span>（ggml-cpu.cpp:201），第 427-429 行直接返回 —— ' +
      'CPU 上"提交"与"完成"本来就是同一件事（ggml-cpu.c:3399 返回时已算完）；' +
      'CUDA 在第 4477 行把 kernel 排进 stream 就返回，等待发生在 cudaStreamSynchronize（ggml-cuda.cu:2544）' +
      '或 cudaStreamWaitEvent（ggml-cuda.cu:4486）。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '一条主线上有三个同步点，别把它们混成一个：',
      '<span class="k">后端接口</span>（第 431 行）是唯一真正阻塞的地方；它的有无决定"异步"是否存在。',
      '<span class="v">事件</span>让等待可以被传递：提交者盖章（1839），消费者按需等（1688）。',
      '<span class="v">调度器收尾</span>（2032-2043）是最后一道保险，也是 sched 级同步路径的全部内容。',
      '记住一句话：<span class="k">异步执行把"提交"与"完成"分开，中间靠 event / synchronize 划界</span> —— ' +
      '这正是 llama.cpp 能在多后端流水线里重叠计算与拷贝的原因。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 3500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
      U.markLines(document, [[1], [1], []][i]);
    }));
    tl.at(15500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; U.markLines(document, [1]); });
  }
},

];
