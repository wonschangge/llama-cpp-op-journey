/* ==========================================================================
   L5-01 · CPU 后端骨架：从 graph_compute 到算子分派
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一个 op 到 CPU 后端，要穿过 <span class="hl-a">6 层</span> */
{
  kicker: "L5 · CPU 后端执行",
  title: "一个 op 到 CPU 后端，要穿过 <span class=\"hl-a\">6 层</span>",
  sub: "本课只讲骨架：入口 -> 规划 -> 线程池 -> 分派 switch。具体内核在 L5-02~L5-05。",
  caption: "★ 核心洞察：CPU 后端是\"一个 switch + 一个线程池\"；GPU 后端则是\"每个 op 一个 kernel 启动\"——这是两条不同的分派路线。",
  src: "ggml/src/ggml-cpu/ggml-cpu.cpp",
  mark: [0, 2, 4, 7, 17, 24],
  lineNo: 170,
  code: `static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
//>> 第 1 层 · 后端接口：调度器（L4-02）交给后端的入口就是它
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
//>> 第 2 层 · 规划：ggml_graph_plan 返回 cplan（任务数 + work buffer 大小）

    if (cpu_ctx->work_size < cplan.work_size) {
//>> work buffer 够用就复用，不够才重分配 —— 避免每张图都 malloc
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;
//>> 把后端持有的 work_data 挂进 cplan

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
//>> 第 3 层 · 启动：进入 ggml-cpu.c 的 ggml_graph_compute
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div class="col" id="layers" style="gap:4px"></div>' +
      '<div class="formula" id="msg"></div>' +
      '<div class="row wrap" id="xref" style="gap:6px"></div>';
    root.appendChild(wrap);

    const defs = [
      ['1', 'ggml_backend_cpu_graph_compute', 'ggml-cpu.cpp:170', 'a'],
      ['2', 'ggml_graph_plan', 'ggml-cpu.c:2815', 'b'],
      ['3', 'ggml_graph_compute', 'ggml-cpu.c:3399', 'c'],
      ['4', 'ggml_graph_compute_thread', 'ggml-cpu.c:3109', 'd'],
      ['5', 'ggml_compute_forward', 'ggml-cpu.c:1744', 'e'],
      ['6', 'ggml_compute_forward_*', 'ops.cpp / unary-ops.cpp ...', 'f']
    ];
    const host = wrap.querySelector('#layers');
    const els = defs.map(function (d) {
      const e = U.el('div', { class: 'formula', style: 'padding:2px 7px;font-size:10px' });
      e.innerHTML = '<span class="chip ' + d[3] + '">' + d[0] + '</span> ' +
        '<span style="color:var(--text)">' + U.esc(d[1]) + '</span> ' +
        '<span class="m">' + U.esc(d[2]) + '</span>';
      host.appendChild(e);
      return e;
    });
    els.forEach(function (e) { e.style.opacity = '.22'; });

    const xref = wrap.querySelector('#xref');
    [['回顾 L1-02 算子身份', 'b'], ['回顾 L1-03 拓扑序', 'b'],
     ['回顾 L4-04 图执行入口', 'c'], ['下一课 L5-02 内核', 'e']].forEach(function (x) {
      xref.appendChild(U.chip(x[0], x[1]));
    });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看全景：一个 op 要穿过 <span class="k">6 层</span>才落到内核。本课逐层走一遍。',
      '<span class="v">第 1 层 ggml_backend_cpu_graph_compute</span>：' +
        '后端接口层。它只做三件事：规划、备 work buffer、调用 compute。',
      '<span class="v">第 2 层 ggml_graph_plan</span>：' +
        '逐节点问 <span class="k">ggml_get_n_tasks()</span>，得到任务数，并算出 work buffer 大小。',
      '<span class="v">第 3 层 ggml_graph_compute</span>：' +
        '启动线程池。<span class="k">主线程自己也当一个 worker</span>（ith = 0）。',
      '<span class="v">第 4 层 ggml_graph_compute_thread</span>：' +
        '每个线程跑同一个循环：按 <span class="k">cgraph->nodes</span> 顺序遍历（L1-03 的拓扑序）。',
      '<span class="v">第 5 层 ggml_compute_forward</span>：' +
        '<span class="k">一个大 switch</span>，按 tensor->op 分派 —— 本课的核心。',
      '<span class="v">第 6 层</span>：真正的内核，按 <span class="k">ith / nth</span> 切分数据。' +
        '内核本身留给 L5-02~L5-05。'
    ];
    defs.forEach(function (_, i) { tl.at(700 + i * 3000, function () {
      els.forEach(function (e, k) { e.style.opacity = k <= i ? '1' : '.22'; });
      msg.innerHTML = texts[i];
    }); });
    tl.at(19600, function () {
      els.forEach(function (e) { e.style.opacity = '1'; });
      msg.innerHTML = texts[6];
    });
  }
},

/* ------------------------------------------------------ 2 接口表 16 个槽，只填了 <span class="hl-c">6 个</span> */
{
  kicker: "L5-01 · 后端接口",
  title: "接口表 16 个槽，只填了 <span class=\"hl-c\">6 个</span>",
  sub: "10 个槽是 NULL：CPU 后端没有 async、没有 event、没有 graph_optimize。",
  caption: "这张表就是 L3 后端注册的落点：注册表拿到的是 struct ggml_backend_i，调度器只认这张表里的函数指针。",
  src: "ggml/src/ggml-cpu/ggml-cpu.cpp",
  mark: [2, 11, 12, 15, 16],
  lineNo: 193,
  code: `static const struct ggml_backend_i ggml_backend_cpu_i = {
//>> struct ggml_backend_i 的一份常量实例 —— CPU 后端的全部对外能力
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
//>> 异步传输槽全为 NULL：CPU 访问自己的内存不需要异步
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
//>> graph_plan_update 为 NULL：CPU 的计划就是 cplan，不需要更新
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
//>> graph_compute —— 本课的主入口，也是唯一必填的槽
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['槽', '值', '为什么'],
      [['get_name / free', '已实现', '后端元信息与析构'],
       ['*_async 系列（6 个）', 'NULL', 'CPU 读写自己的内存，没有异步语义'],
       ['graph_plan_create / free', '已实现', '预编译一份 cplan（L4-04 的 plan 路径）'],
       ['graph_plan_update', 'NULL', 'cplan 由 ggml_graph_plan 一次算出，无需更新'],
       ['graph_plan_compute', '已实现', '执行预编译计划'],
       ['graph_compute', '已实现', '执行整张图 —— 本课主线'],
       ['event_record / event_wait', 'NULL', 'CPU 是同步后端，没有事件'],
       ['graph_optimize', 'NULL', 'CPU 侧的图优化只有算子融合，见第 5 幕']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '16 个槽：<span class="k">6 个已实现、10 个 NULL</span>。NULL 不是"没写完"，是"不需要"。',
      '<span class="v">*_async 全 NULL</span>：CPU 与数据在同一块内存上，异步没有意义。',
      '<span class="v">graph_plan_* 三件套</span>：CPU 保留了"预编译计划"的能力，' +
        '对应 L4-04 讲的 graph_plan 路径。',
      '<span class="v">graph_compute</span> 是主入口：一张 cgraph 进来，' +
        '一个 ggml_status 出去。',
      '对比 GPU 后端：CUDA 的接口表里有一堆 kernel 启动与流同步的槽，' +
        'CPU 只有这一个 <span class="k">graph_compute</span>。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) { tl.at(2400 + i * 2100, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 4)];
    }); });
    tl.at(16800, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 3 ★ 规划阶段就把每个 op 的<span class="hl-a">任务数</span>问清楚了 */
{
  kicker: "L5-01 · 规划",
  title: "★ 规划阶段就把每个 op 的<span class=\"hl-a\">任务数</span>问清楚了",
  sub: "ggml_graph_plan 遍历整张图，逐节点问 ggml_get_n_tasks 要任务数，再据此算 work buffer。",
  caption: "真正的线程数在 ggml-cpu.c:3063 封顶：cplan.n_threads = MIN(max_tasks, n_threads) —— 线程数不是用户说了算。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [0, 4, 6, 8, 11],
  lineNo: 2842,
  code: `    int max_tasks = 1;
//>> max_tasks 从 1 起步

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
//>> 按 cgraph->nodes 顺序遍历 —— 就是 L1-03 的拓扑序
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);
//>> 每个节点问一次：这个 op 想开几个任务？

        max_tasks = MAX(max_tasks, n_tasks);
//>> 取全图最大值：max_tasks = 整张图的最大并行度`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['节点', 'op', 'ggml_get_n_tasks(node, 8)', 'max_tasks'],
      [['nodes[0]', 'MUL_MAT', 'n_threads = 8', '8'],
       ['nodes[1]', 'ADD', 'n_threads = 8', '8'],
       ['nodes[2]', 'SUM', '1（归约，单任务）', '8'],
       ['nodes[3]', 'SOFT_MAX', 'MIN(8, ggml_nrows(src[0]) = 4) = 4', '8'],
       ['nodes[4]', 'RMS_NORM', 'n_threads = 8', '8']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '示例：n_threads = 8，图里 5 个节点。逐个问任务数。',
      '<span class="v">MUL_MAT / ADD / RMS_NORM</span> 这类 op 返回 n_threads —— 想全并行。',
      '<span class="v">SUM</span> 返回 1：归约算子的输出只有几个数，切了反而更慢。',
      '<span class="v">SOFT_MAX</span> 返回 MIN(n_threads, ggml_nrows(src[0]))：' +
        '按行切，行数不够就少开任务。',
      '全图扫完：<span class="k">max_tasks = 8</span> -> ' +
        '<span class="v">cplan.n_threads = MIN(8, 8) = 8</span>。',
      '若这张图只有 SUM 和 SOFT_MAX(4 行)，则 max_tasks = 4，' +
        '<span class="k">cplan.n_threads = 4</span> —— 线程数被图的内容封顶。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) { tl.at(2600 + i * 2600, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
    }); });
    tl.at(16200, function () { msg.innerHTML = texts[4]; });
    tl.at(17800, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 4 线程池：<span class="hl-b">n_threads</span> 个 worker 共享一个 cgraph */
{
  kicker: "L5-01 · 线程池",
  title: "线程池：<span class=\"hl-b\">n_threads</span> 个 worker 共享一个 cgraph",
  sub: "每个 worker 一份 ggml_compute_state（只有 ith 不同），共用同一个 threadpool。",
  caption: "ggml_barrier()（ggml-cpu.c:576）用 n_barrier / n_barrier_passed 做自旋屏障 —— n_threads == 1 时直接返回，不做任何同步。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [4, 6, 9, 13, 21, 23, 31, 37, 39, 40],
  lineNo: 481,
  code: `struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
//>> 当前正在算的图：所有 worker 共享同一个指针
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
//>> n_graph 同时编码"第几张图"和"本次活跃线程数"
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.
//>> current_chunk：matmul 抢块用的原子计数器（第 8 幕）

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
//>> workers 数组：每个线程一项
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
//>> cpumask：NUMA / 亲和性用
    struct ggml_threadpool * threadpool;
    int ith;
//>> ith —— 线程唯一编号，内核靠它切分数据
};`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div class="row" id="row1" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const row1 = wrap.querySelector('#row1');
    const a = U.card({ c: 'b', t: 'struct ggml_threadpool（481 行）',
      b: '全局一份。持有 cgraph / cplan 指针、<br>同步原语、workers 数组、n_threads。',
      m: 'n_graph · n_barrier · current_chunk' }, { style: 'width:330px' });
    const b = U.card({ c: 'd', t: 'struct ggml_compute_state（508 行）',
      b: '每个 worker 一份。<b>唯一的区别就是 ith</b>，<br>以及它属于哪个 threadpool。',
      m: 'int ith;  // 0 .. n_threads-1' }, { style: 'width:330px' });
    row1.appendChild(a); row1.appendChild(b);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '两个结构体分工：<span class="k">线程池是共享的，线程状态是每线程一份</span>。',
      '<span class="v">ggml_threadpool</span>：cgraph / cplan 指针 + 同步原语 + workers 数组 + n_threads。',
      '<span class="v">n_graph</span> 是打包值：高位是"第几张图"，低 16 位是本次活跃线程数' +
        '（GGML_THREADPOOL_N_THREADS_MASK，206 行）。',
      '<span class="v">ggml_compute_state</span>：每个 worker 只比别的 worker 多一个数字 —— ith。',
      '<span class="v">current_chunk</span> 是 matmul 的抢块计数器：' +
        '线程不预先分活，而是边算边抢（第 8 幕）。',
      '一句话：<span class="k">线程池提供并行度，ith 提供身份</span>。内核只认 ith / nth。'
    ];
    a.style.opacity = '.3'; b.style.opacity = '.3';
    tl.at(600, function () { a.style.opacity = '1'; msg.innerHTML = texts[0]; });
    tl.at(3200, function () { msg.innerHTML = texts[1]; });
    tl.at(5800, function () { msg.innerHTML = texts[2]; });
    tl.at(8400, function () { b.style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(11000, function () { msg.innerHTML = texts[4]; });
    tl.at(14000, function () { a.style.opacity = '1'; b.style.opacity = '1'; msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 ★ 每个线程跑同一个循环：<span class="hl-a">params{ith, nth}</span> */
{
  kicker: "L5-01 · 线程主体",
  title: "★ 每个线程跑同一个循环：<span class=\"hl-a\">params{ith, nth}</span>",
  sub: "节点按拓扑序遍历，每个节点先试算子融合，没命中才交给 ggml_compute_forward。",
  caption: "节点之间有一次 ggml_barrier（3165 行）：下一个节点可能要读上一个节点的输出。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [0, 1, 2, 16, 25, 32, 37, 48],
  lineNo: 3122,
  code: `    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
//>> nth = 本次图执行真正活跃的线程数（来自 cplan.n_threads）
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
//>> 按 cgraph->nodes 顺序遍历 —— 拓扑序保证依赖已就绪（L1-03）
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
//>> 没有 GGML_TENSOR_FLAG_COMPUTE 的节点直接跳过
            continue;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
//>> CPU 侧的算子融合：命中就把后续节点一起吃掉
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
//>> 没命中融合 —— 走大 switch（下一幕）
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
//>> 节点之间的屏障：所有线程到齐才进入下一个节点
        }
    }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
      '<div class="row" id="nodes" style="gap:5px;align-items:center"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'params.ith', b: '我是第几号线程。<br>内核靠它决定"我算哪一块"。', m: 'state->ith' },
      { c: 'b', t: 'params.nth', b: '一共有几个活跃线程。<br>内核靠它决定"切成几块"。', m: 'n_graph & MASK' },
      { c: 'c', t: 'params.wdata / wsize', b: '规划阶段算出的 work buffer，<br>量化 matmul 的中间结果放这里。', m: 'cplan->work_data' },
      { c: 'd', t: 'params.threadpool', b: '回指线程池 —— 抢块、屏障都要用它。', m: 'tp' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(function (d) { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(function (e) { e.style.opacity = '.3'; });

    const nodes = wrap.querySelector('#nodes');
    const ns = [];
    for (let i = 0; i < 6; i++) {
      const e = U.el('div', { class: 'formula', style: 'padding:2px 6px;font-size:9.5px' });
      e.innerHTML = '<span class="m">node' + i + '</span>';
      nodes.appendChild(e); ns.push(e);
      if (i < 5) nodes.appendChild(U.arrow('->'));
    }

    const msg = wrap.querySelector('#msg');
    const texts = [
      '线程启动时先构造 <span class="k">params</span>：这是内核能看到的全部上下文。',
      '<span class="v">ith</span> + <span class="v">nth</span>：内核切分数据的唯一依据。',
      '<span class="v">wdata / wsize</span>：共享的工作缓冲区，量化 matmul 会往里放反量化后的 src1。',
      '<span class="v">threadpool</span>：内核可以回指线程池 —— 抢块和屏障都靠它。',
      '然后进入节点循环：<span class="k">for node_n = 0 .. n_nodes-1</span>，' +
        '每个节点先试 <span class="v">ggml_cpu_try_fuse_ops</span>。',
      '没命中融合就调 <span class="v">ggml_compute_forward(&params, node)</span> —— 进入大 switch。',
      '每个节点结束时 <span class="k">ggml_barrier</span>：' +
        'n_threads == 1 时它直接返回，所以单线程没有同步开销。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    defs.forEach(function (_, i) { tl.at(2800 + i * 2600, function () {
      els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.3'; });
      msg.innerHTML = texts[i + 1];
    }); });
    tl.at(13800, function () {
      els.forEach(function (e) { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
      ns.forEach(function (e, i) { e.style.background = i === 0 ? 'rgba(88,166,255,.18)' : ''; });
    });
    tl.at(16800, function () {
      ns.forEach(function (e, i) { e.style.background = i === 1 ? 'rgba(247,120,186,.18)' : ''; });
      msg.innerHTML = texts[5];
    });
    tl.at(19600, function () {
      ns.forEach(function (e) { e.style.background = ''; });
      msg.innerHTML = texts[6];
    });
  }
},

/* ------------------------------------------------------ 6 ★ <span class="hl-e">ggml_compute_forward</span>：一个 switch 分派所有算子 */
{
  kicker: "L5-01 · 核心",
  title: "★ <span class=\"hl-e\">ggml_compute_forward</span>：一个 switch 分派所有算子",
  sub: "102 个 case：96 个直接调用内核，5 个是 nop（视图算子），1 个是 COUNT 兜底。",
  caption: "对照 GPU 后端：CUDA 走\"每个 op 一个 kernel 启动\"，CPU 走\"一个 switch + 线程池\"。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [0, 4, 10, 15, 17, 22],
  lineNo: 1744,
  code: `static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
//>> 分派层：参数只有 params 和 tensor —— 没有返回值，结果写回 tensor
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
//>> 空算子与 GGML_OP_NONE 直接返回 —— 视图算子不进内核
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
//>> 第 0 层钩子：AMX / KleidiAI / repack 可以在这里截胡（traits.cpp）
        return;
    }

    switch (tensor->op) {
//>> 整个 CPU 后端的算子分派就是这一个 switch
        case GGML_OP_DUP:
//>> 每个 case 一行调用，内核名 = ggml_compute_forward_ + op 小写
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div class="row" id="row1" style="gap:9px"><div class="col grow" id="bars" style="gap:6px"></div>' +
      '<div class="col" id="right" style="gap:7px;width:250px"></div></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const bars = wrap.querySelector('#bars');
    const groups = [
      { l: 'case 总数', n: 102, c: 'a' },
      { l: '调用内核', n: 96, c: 'b' },
      { l: 'nop 视图算子', n: 5, c: 'c' },
      { l: 'COUNT 兜底', n: 1, c: 'd' }
    ];
    const bs = groups.map(function (g) { const b = U.bar(g.l + '  (' + g.n + ')', g.c); bars.appendChild(b.el); return b; });

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--e)">' +
      '<div class="ct" style="color:var(--e)">5 个 nop 是谁？</div>' +
      '<div class="cb"><span class="cm" style="margin:0">NONE / RESHAPE / PERMUTE / VIEW / TRANSPOSE</span>' +
      ' —— 它们只改 ne[]/nb[]，没有数据要算（L1-01 的视图算子）。</div></div>' +
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">96 个内核去哪了？</div>' +
      '<div class="cb">一元/二元 -> <b>L5-02</b>；量化与 repack -> <b>L5-04</b>；' +
      'SIMD 分发 -> <b>L5-03 / L5-05</b>。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '数一数这个 switch：<span class="k">102 个 case</span>。',
      '其中 <span class="v">96 个</span>直接调用 ggml_compute_forward_* —— 每个 case 就是一次内核调用。',
      '<span class="v">5 个</span>是 nop：视图算子不需要计算，只改形状和步长。',
      '<span class="v">1 个</span>是 GGML_OP_COUNT：走到它就 GGML_ABORT("fatal error")。',
      '这个 switch <span class="k">没有 default</span>：102 个 case 与 <span class="v">enum ggml_op</span> 的' +
        '102 个枚举项一一对应（枚举本身见 L1-02 引的 ggml.h）—— 编译器能查漏，不需要兜底。',
      '对比：<span class="k">GPU 后端是"每个 op 一个 kernel"</span>，CPU 是' +
        '<span class="k">"一个 switch 分派全部"</span> —— 这正是 CPU 后端骨架的全部。'
    ];
    tl.at(700, function () {
      bs.forEach(function (b, i) { b.fill.style.width = (20 + i * 22) + '%'; b.val.textContent = groups[i].n; });
      msg.innerHTML = texts[0];
    });
    tl.at(3400, function () { msg.innerHTML = texts[1]; });
    tl.at(6400, function () { msg.innerHTML = texts[2]; });
    tl.at(9400, function () { msg.innerHTML = texts[3]; });
    tl.at(12400, function () { msg.innerHTML = texts[4]; });
    tl.at(15400, function () { msg.innerHTML = texts[5]; });
    tl.at(18400, function () { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 ★ <span class="hl-c">ggml_get_n_tasks</span>：哪类 op 开几个任务 */
{
  kicker: "L5-01 · 核心",
  title: "★ <span class=\"hl-c\">ggml_get_n_tasks</span>：哪类 op 开几个任务",
  sub: "四种策略：全并行 n_threads、单任务 1、按数据量封顶 MIN(...)、由 op_params 决定。",
  caption: "顶层 102 个 case；把 UNARY（22 个子算子）与 GLU（7 个）展开后是 129 个 op 分支：72 个全并行 / 48 个单任务 / 4 个 MIN / 4 个 op_params / 1 个 abort。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [0, 10, 12, 15, 25, 27],
  lineNo: 2263,
  code: `        case GGML_OP_CPY:
//>> 第一组：10 个 op 想全并行 —— 逐元素 / 复制 / 累加类
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
//>> n_tasks = n_threads：有多少线程就用多少
            } break;
        case GGML_OP_SUB:
//>> 第二组：10 个 op 只要 1 个任务
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
//>> 归约类算子的输出很小，多线程切分开销大于收益
            } break;`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['n_tasks 策略', 'op 分支数', '代表 op', '为什么'],
      [['= n_threads', '72', 'ADD / MUL_MAT / RMS_NORM / CONV_2D / ROPE', '数据量大，切得开'],
       ['= 1', '48', 'SUM / ARGMAX / CLAMP / POOL_2D / GET_ROWS', '输出小或本身是归约'],
       ['= MIN(n_threads, 数据)', '4', 'SOFT_MAX、RWKV_WKV6/7、GATED_LINEAR_ATTN', '按行/按头切，切不动就少开'],
       ['由 op_params 决定', '4', 'MAP_CUSTOM1/2/3、CUSTOM', '调用者自己指定 n_tasks'],
       ['abort', '1', 'GGML_OP_COUNT', '兜底：不允许出现']],
      { monoCols: [0, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '把 ggml_get_n_tasks 的 case 按"返回值"分五组。',
      '<span class="v">n_tasks = n_threads</span>（72 个）：逐元素、复制、matmul、卷积、RoPE —— ' +
        '这些 op 的数据量随模型规模增长，切得越开越快。',
      '<span class="v">n_tasks = 1</span>（48 个）：SUM / ARGMAX / MEAN 这类归约，' +
        '以及视图算子（它们根本不算）。',
      '<span class="v">MIN(n_threads, ...)</span>（4 个）：' +
        'SOFT_MAX 用 <span class="k">ggml_nrows(node->src[0])</span> 封顶；' +
        'RWKV 系列用 <span class="k">node->src[1]->ne[1]</span>（头数）封顶。',
      '<span class="v">op_params</span>（4 个）：CUSTOM 家族读 ' +
        '<span class="k">p.n_tasks</span>，等于 GGML_N_TASKS_MAX 时才用 n_threads。',
      '注意 GET_ROWS / SET_ROWS：源码里 <span class="k">//n_tasks = n_threads;</span> 被注释掉了，' +
        '理由写在旁边的 FIXME 里 —— 见 source.md。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) { tl.at(2600 + i * 3400, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }); });
    tl.at(20600, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 8 ith / nth 落到内核：<span class="hl-d">抢块</span>，不是静态切行 */
{
  kicker: "L5-01 · 任务到线程",
  title: "ith / nth 落到内核：<span class=\"hl-d\">抢块</span>，不是静态切行",
  sub: "mul_mat 用 ith 认领第一块，剩下的块用原子计数器 current_chunk 动态抢。",
  caption: "不是所有内核都这样：多数逐元素内核是静态切分（按 ith 取行）。两种写法的对比在 L5-02 / L5-03。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [0, 2, 3, 7, 12, 15, 17, 18],
  lineNo: 1430,
  code: `        // distribute the thread work across the inner or outer loop based on which one is larger
//>> 按哪一维切：谁的行多就切谁 —— 让每块的计算量尽量均匀
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
//>> 每块的元素数 = 总行数 / 块数（向上取整）
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;
//>> 第一块由 ith 认领 —— 这是唯一的"静态"部分

    while (current_chunk < nchunk0 * nchunk1) {
//>> 剩下的块由原子计数器动态分配：谁先算完谁先抢
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div class="row wrap" id="grid" style="gap:5px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const grid = wrap.querySelector('#grid');
    const cells = [];
    for (let i = 0; i < 8; i++) {
      const e = U.el('div', { style: 'width:74px;height:30px;border:1px solid var(--border);border-radius:3px;' +
        'background:#10151b;display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--dim)' });
      e.textContent = 'chunk ' + i;
      grid.appendChild(e); cells.push(e);
    }

    const msg = wrap.querySelector('#msg');
    const texts = [
      'nth = 4，块数 = 8。看线程怎么拿到自己的活。',
      '<span class="v">ith = 0..3</span> 各自认领一块：chunk 0 / 1 / 2 / 3 —— 这是起点。',
      '算完第一块后 <span class="k">atomic_fetch_add(current_chunk, 1)</span>：' +
        '抢到 4 / 5 / 6 / 7。',
      '谁算得快谁抢得多 —— <span class="k">动态负载均衡</span>。' +
        '行数不整除、量化类型快慢不一时，这比静态切分稳。',
      '为什么不是每个内核都这样？抢块有原子操作开销；' +
        '<span class="k">逐元素内核算得又快又均匀，静态切分更划算</span>。',
      '所以 nth 的语义是"<span class="k">最多开几个任务</span>"，' +
        '具体怎么切由内核自己决定 —— 这是本课要记住的一句话。'
    ];
    const colors = ['a', 'b', 'c', 'd'];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    tl.at(3000, function () {
      cells.forEach(function (e, i) { if (i < 4) { e.style.borderColor = 'var(--' + colors[i] + ')';
        e.style.color = 'var(--' + colors[i] + ')'; e.textContent = 'chunk ' + i + ' · ith' + i; } });
      msg.innerHTML = texts[1];
    });
    tl.at(6600, function () {
      cells.forEach(function (e, i) { if (i >= 4) { e.style.borderColor = 'var(--' + colors[i - 4] + ')';
        e.style.color = 'var(--' + colors[i - 4] + ')'; e.textContent = 'chunk ' + i + ' · ith' + (i - 4); } });
      msg.innerHTML = texts[2];
    });
    tl.at(10200, function () { msg.innerHTML = texts[3]; });
    tl.at(13400, function () { msg.innerHTML = texts[4]; });
    tl.at(16400, function () { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 <span class="hl-f">ggml-cpu.h</span>：CPU 后端的公共契约 */
{
  kicker: "L5-01 · 公共面",
  title: "<span class=\"hl-f\">ggml-cpu.h</span>：CPU 后端的公共契约",
  sub: "线程池 5 个函数 + graph_plan/graph_compute 2 个函数 + 后端 API 5 个函数。",
  caption: "ggml_cpu_init() 在 ggml-cpu.c:3867，特征检测 ggml_cpu_has_avx() 在 3643 —— 头文件声明、C 文件实现，是 L5-05（多架构 SIMD）的入口。",
  src: "ggml/include/ggml-cpu.h",
  mark: [0, 2, 3, 10, 15, 20],
  lineNo: 58,
  code: `    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
//>> 线程池 5 件套：新建 / 释放 / 查询 / 暂停 / 恢复
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
//>> 头文件把调用顺序写死在注释里：必须先 plan，再 compute
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
//>> ggml_graph_plan —— 第 3 幕的主角
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
//>> ggml_graph_compute —— 第 5 幕的入口

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
//>> 便利版：work_data 直接开在 ggml_context 里（L1-05 的 arena）`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['头文件里的区段', '行号', '内容', '本课/后续课'],
      [['struct ggml_cplan', '12-25', 'work_size / work_data / n_threads / threadpool', '第 3 幕'],
       ['线程池 API', '58-62', 'threadpool_new / free / get_n_threads / pause / resume', '第 4 幕'],
       ['plan + compute', '66-74', 'ggml_graph_plan / ggml_graph_compute', '第 3、5 幕'],
       ['特征检测', '80-110', 'ggml_cpu_has_avx / neon / sve / riscv_v ...', 'L5-05'],
       ['CPU 类型 traits', '117-124', 'ggml_type_traits_cpu（from_float / vec_dot）', 'L5-03 / L5-04'],
       ['初始化', '126', 'ggml_cpu_init()：建查表 + 读环境变量', 'ggml-cpu.c:3867'],
       ['后端 API', '132-141', 'backend_cpu_init / set_n_threads / reg ...', 'L3 后端注册']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'ggml-cpu.h 只有 152 行，却是 CPU 后端的全部公共面。',
      '<span class="v">struct ggml_cplan</span>：plan 与 compute 之间的唯一契约（L4-04 讲过它的调度侧）。',
      '<span class="v">线程池 5 件套</span>：谁想复用线程池，就调这几个函数 —— ' +
        'llama.cpp 就是这么做的（ggml-cpu.cpp:657 把 set_n_threads 暴露给后端注册表）。',
      '<span class="v">特征检测</span>：一组 ggml_cpu_has_* —— 声明在头文件、实现在 ggml-cpu.c。',
      '<span class="v">ggml_cpu_init()</span>：第一次调用时建 GELU/SILU/FP16 查表，' +
        '并读 <span class="k">GGML_CPU_DISABLE_FUSION</span> 环境变量。',
      '一句话：<span class="k">头文件是契约，ggml-cpu.c / ggml-cpu.cpp 是两份实现</span>。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) { tl.at(2600 + i * 2300, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 4)];
    }); });
    tl.at(18600, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表 */
{
  kicker: "L5-01 · 收束",
  title: "把这一课压成一张表",
  sub: "一个 op 从进入 CPU 后端到调用内核，经过 6 层；其中两层决定\"开几个任务\"。",
  caption: "下一课 L5-02 走进 switch 后面的第一个家族：一元与二元算子内核。",
  src: "ggml/src/ggml-cpu/ggml-cpu.c",
  mark: [0, 2, 4],
  lineNo: 3062,
  code: `    cplan.threadpool = threadpool;
//>> 规划阶段的三项产物，就是 plan 交给 compute 的全部信息
    cplan.n_threads  = MIN(max_tasks, n_threads);
//>> ★ 真正启动的线程数 = 全图最大任务数，被 n_threads 封顶
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['层', '函数', '位置', '下一课在哪展开'],
      [['1 接口', 'ggml_backend_cpu_graph_compute', 'ggml-cpu.cpp:170', 'L3 后端注册 / L4-04'],
       ['2 规划', 'ggml_graph_plan + ggml_get_n_tasks', 'ggml-cpu.c:2815 / 2253', '本课'],
       ['3 启动', 'ggml_graph_compute + kickoff', 'ggml-cpu.c:3399 / 3288', 'L4-04'],
       ['4 线程主体', 'ggml_graph_compute_thread', 'ggml-cpu.c:3109', 'L4-04'],
       ['5 分派', 'ggml_compute_forward（大 switch）', 'ggml-cpu.c:1744', '本课'],
       ['6 内核', 'ggml_compute_forward_*（ith/nth）', 'ops.cpp / unary-ops.cpp ...', 'L5-02 ~ L5-05']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一个 <span class="mono">GGML_OP_RMS_NORM</span> 节点，' +
      '从进入 CPU 后端到调用内核，经过哪几层？' +
      '如果 <span class="mono">n_threads = 16</span> 而图里只有它一个节点，会启动几个线程？',
      '<b>六层</b>：<span class="mono">ggml_backend_cpu_graph_compute</span>（ggml-cpu.cpp:170）' +
      ' -> <span class="mono">ggml_graph_plan</span>（2815）' +
      ' -> <span class="mono">ggml_graph_compute</span>（3399）' +
      ' -> <span class="mono">ggml_graph_compute_thread</span>（3109）' +
      ' -> <span class="mono">ggml_compute_forward</span>（1744）' +
      ' -> <span class="mono">ggml_compute_forward_rms_norm</span>。<br>' +
      '<b>线程数</b>：RMS_NORM 在 <span class="mono">ggml_get_n_tasks</span> 里属于 ' +
      '<span class="mono">n_tasks = n_threads</span> 那一组（2364-2369），' +
      '所以 <span class="mono">max_tasks = 16</span>，' +
      '<span class="mono">cplan.n_threads = MIN(16, 16) = 16</span>（3063）—— 会开 16 个。<br>' +
      '<b>注意</b>：如果那个节点是 <span class="mono">GGML_OP_SUM</span>，' +
      '<span class="mono">n_tasks = 1</span>，就只会开 1 个线程 —— ' +
      '<span class="mono">n_threads</span> 只是上限。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '六层，两层是本课的重点：<span class="k">规划层</span>和<span class="k">分派层</span>。',
      '<span class="k">规划层</span>：ggml_get_n_tasks 给每个 op 定任务数，' +
        'cplan.n_threads = MIN(max_tasks, n_threads)。',
      '<span class="k">分派层</span>：一个 102 个 case 的 switch，把 op 变成一次内核调用。',
      '这两层合起来就是 CPU 后端的骨架：<span class="v">一个 switch + 一个线程池</span>。',
      '下一课 L5-02：走进 switch 后面的第一个家族 —— 一元与二元算子内核。'
    ];
    tl.at(700, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) { tl.at(2400 + i * 2300, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 2)];
    }); });
    tl.at(16600, function () { msg.innerHTML = texts[2]; });
    tl.at(18600, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[3];
    });
    tl.at(20400, function () { msg.innerHTML = texts[4]; });
  }
},

];
