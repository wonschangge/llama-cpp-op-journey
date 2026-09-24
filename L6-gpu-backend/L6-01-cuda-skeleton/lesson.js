/* ==========================================================================
   L6-01 · CUDA 后端骨架：一个 switch + 一个 stream
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 CUDA 后端 = 四个 C 入口 + 三张虚表 */
{
  kicker: "L6-01 · 全局",
  title: "CUDA 后端 = 四个 C 入口 + 三张虚表",
  sub: "公共头只声明入口；真正的约定在 common.cuh 的上下文和 ggml-cuda.cu 的三张虚表里。",
  caption: "本课覆盖三个文件：ggml/include/ggml-cuda.h、ggml/src/ggml-cuda/common.cuh、ggml/src/ggml-cuda/ggml-cuda.cu（5856 行，本课只挑骨架）。",
  src: "ggml/include/ggml-cuda.h",
  mark: [0, 4, 7, 10, 17, 20, 23],
  lineNo: 20,
  code: `#define GGML_CUDA_MAX_DEVICES       16
//>> 设备数上限：所有按设备索引的数组都用它定长（如 streams[16][...]）

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);
//>> backend 对象：一个设备一个实例，内部就是 ggml_backend_cuda_context

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);
//>> buffer type：显存分配器的"型号"，第 4 幕的第三张虚表

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);
//>> host pinned 版本：给 CPU 后端做快速搬运的"中转站"

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
//>> 设备层：设备数 / 描述 / 显存 —— 能力判据就挂在这一层
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip">模型 / 计算图</span><span class="arrow">-></span>
        <span class="chip a">调度器（L4-02）</span><span class="arrow">-></span>
        <span class="chip b">CUDA 后端</span><span class="arrow">-></span>
        <span class="chip c">显存</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'backend', m: 'ggml_backend_cuda_init', b: '一个 device 一个后端实例；<br>上下文里装着 stream 与显存池。' },
      { c: 'b', t: 'buffer type', m: 'ggml_backend_cuda_buffer_type', b: '显存怎么分配、怎么对齐、<br>一个张量要多少字节。' },
      { c: 'c', t: 'device', m: 'ggml_backend_cuda_get_device_count', b: '设备数 / 描述 / 剩余显存；<br>supports_op 是设备虚表的槽。' },
      { c: 'd', t: 'reg', m: 'ggml_backend_cuda_reg', b: '注册表：把设备列表交给<br>ggml 的后端发现机制（L3-03）。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const mk = [0, 4, 7, 10];
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看公共头：CUDA 后端对外只有这几个 C 函数。<span class="k">每个函数的实现都只是去查一张虚表。</span>',
      '<span class="v">ggml_backend_cuda_init()</span>：建后端对象。它的 context 就是一个设备上的全部资源。',
      '<span class="v">buffer type</span>：调度器用它决定"张量放哪、占多少显存"—— 见 L4-01。',
      '<span class="v">device 层</span>：设备发现 + 能力判据。<span class="k">supports_op 就在这层</span>（第 9 幕）。',
      '<span class="v">reg</span>：注册表把设备列表交出去。回顾 L3-01：公共 API 是给人调的，干活的永远是虚表。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, [mk[i]]);
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [0, 20]); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 后端对象：一个 device，一组 stream，一组池 */
{
  kicker: "L6-01 · 上下文",
  title: "后端对象：一个 device，一组 stream，一组池",
  sub: "struct ggml_backend_cuda_context 就是\"CUDA 后端\"这个对象的全部状态。",
  caption: "池字段在 common.cuh:1559（pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]）；GGML_CUDA_MAX_STREAMS 在 common.cuh:188 定义为 8。",
  src: "ggml/src/ggml-cuda/common.cuh",
  mark: [1, 6, 12, 19],
  lineNo: 1455,
  code: `struct ggml_backend_cuda_context {
    int device;
//>> 设备号：后续所有 cudaMalloc / kernel 都发到这个设备上
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
//>> stream 不是一个，是「设备 x stream 槽」的二维表
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    void * cublas_workspaces[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    size_t cublas_workspace_sizes[GGML_CUDA_MAX_DEVICES] = {0};

    int curr_stream_no = 0;
//>> 当前用第几号 stream —— 并发分支执行期间会被临时改掉（第 6 幕）

#ifdef USE_CUDA_GRAPH
//>> CUDA graph 缓存被 USE_CUDA_GRAPH 包住：不编译这个宏就完全没有 capture 路径
    // Map from first_node_ptr to cuda_graph - allows multiple graphs per context
    // when the computation is split across CPU/GPU (e.g., with --n-cpu-moe)
    std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;
//>> 按「首节点指针」缓存图；下面注释点名了 CPU/GPU 混跑时会有多张图`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '一个设备', m: 'int device; std::string name', b: '上下文绑定一个设备号；<br>name 是后端实例名。' },
      { c: 'b', t: '一组 stream', m: 'streams[16][8]', b: '不是"一条 stream"。<br>默认只用 0 号，其余留给并发分支。' },
      { c: 'c', t: '一组显存池', m: 'pools[16][8]', b: '每个 (设备, stream 槽)<br>一个池 —— 池里的内存不跨流复用。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个后端实例 = 一个 <span class="k">ggml_backend_cuda_context</span>。它把"设备 + 资源"绑在一起。',
      '<span class="v">device</span>：所有设备相关的调用（set_device、分配、kernel）都发到它上面。',
      '<span class="v">streams[16][8]</span>：硬件上可以有多条流，所以上下文存的是一张二维表，而不是一个句柄。',
      '<span class="v">curr_stream_no</span>：默认 0。<span class="k">并发分支会把当前节点切到别的槽上</span>（第 6 幕 4310-4312）。',
      '<span class="v">cuda_graphs</span>：按图的首节点指针缓存；源码注释说明 CPU/GPU 混跑（例如 n-cpu-moe 那种切分）会有多张图。',
      '同构的还有 <span class="v">cublas_handles</span> 与 <span class="v">pools</span>：都是 <span class="k">每个 (设备, stream 槽) 一套</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, [[1], [1], [6]][i] || [6, 12, 19]);
      msg.innerHTML = texts[i];
    }));
    tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [12]); msg.innerHTML = texts[3]; });
    tl.at(16400, () => { U.markLines(document, [19]); msg.innerHTML = texts[4]; });
    tl.at(18000, () => { U.markLines(document, [6, 12, 19]); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 3 stream 是<span class="hl-a">懒创建</span>的，而且不是 legacy 默认流 */
{
  kicker: "L6-01 · stream",
  title: "stream 是<span class=\"hl-a\">懒创建</span>的，而且不是 legacy 默认流",
  sub: "没有\"新建 stream\"这种后端接口 —— stream 属于上下文，随用随建，随上下文一起销毁。",
  caption: "销毁在 ggml-cuda.cu:711（析构函数里 cudaStreamDestroy 扫全部 16 x 8 个槽）；并发分支用 event fork/join（ggml-cuda.cu:4205-4211）。",
  src: "ggml/src/ggml-cuda/common.cuh",
  mark: [0, 2, 5, 8, 11, 14],
  lineNo: 1528,
  code: `    cudaStream_t stream(int device, int stream) {
//>> 两个参数：哪个设备、第几号 stream 槽 —— 槽就是上面那张二维表的下标
        if (streams[device][stream] == nullptr) {
//>> 只有 nullptr 才建：这就是"懒创建"的全部
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
//>> cudaStreamNonBlocking：不跟 legacy default stream 隐式同步
        }
        return streams[device][stream];
    }

    cudaStream_t stream() { return stream(device, curr_stream_no); }
//>> 无参版本永远取 curr_stream_no 那条 —— kernel 发射默认走它

    ggml_cuda_stream_context & stream_context() { return concurrent_stream_context; }`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip a">第一次用到</span><span class="arrow">-></span>
        <span class="chip b">建 stream</span><span class="arrow">-></span>
        <span class="chip c">之后直接返回</span><span class="arrow">-></span>
        <span class="chip d">析构时销毁</span>
      </div>
      <div class="row wrap" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '建的时候', m: 'cudaStreamCreateWithFlags', b: '只在 streams[d][s] 还是<br>nullptr 时才调用。' },
      { c: 'b', t: '为什么是 NonBlocking', m: 'cudaStreamNonBlocking', b: '不跟 legacy 默认流隐式同步，<br>才能被 event fork 成并发分支。' },
      { c: 'c', t: '销毁的时候', m: 'cudaStreamDestroy', b: '上下文析构时统一销毁，<br>没有任何"free stream"接口。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'stream 的生命周期跟上下文绑定：<span class="k">用的时候才建，不用管释放</span>。',
      '<span class="v">stream(device, stream)</span>：先 ggml_cuda_set_device，再建 —— 保证 stream 建在正确的设备上。',
      '<span class="v">cudaStreamNonBlocking</span>：这一点很关键。0 号 stream 不是 legacy 默认流，所以能被 fork。',
      'fork 的写法（ggml-cuda.cu:4205-4211）：主 stream 记录 fork_event，各分支 cudaStreamWaitEvent 等它。',
      '收束：CUDA 后端没有"把活派给哪条流"的接口 —— <span class="k">发射 kernel 的代码自己调 ctx.stream()</span>。',
      '跨课呼应：L4-04 讲执行入口的异步与 event；这一课补上 CUDA 侧的载体。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, [[0], [2, 5], [5]][i] || []);
      msg.innerHTML = texts[i];
    }));
    tl.at(10200, () => { U.markLines(document, [8]); msg.innerHTML = texts[3]; });
    tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [11]); msg.innerHTML = texts[4]; });
    tl.at(15400, () => { U.markLines(document, [14]); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 4 三张虚表里的第一张：<span class="hl-b">backend 虚表</span> */
{
  kicker: "L6-01 · 虚表",
  title: "三张虚表里的第一张：<span class=\"hl-b\">backend 虚表</span>",
  sub: "16 个槽里有 4 个是 NULL —— CUDA 不用 ggml 的图计划接口，它走自己的 capture。",
  caption: "回顾 L3-01：三张虚表各管一层。CUDA 的另外两张：device 虚表在 ggml-cuda.cu:5651-5667，buffer_type 虚表在 ggml-cuda.cu:927-934。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [0, 9, 11, 16, 18, 20, 21],
  lineNo: 4840,
  code: `static const ggml_backend_i ggml_backend_cuda_interface = {
//>> 一张只装函数指针的 static const 表 —— 后端契约的 CUDA 实现
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
//>> 同步：见 ggml-cuda.cu:2547 的 cudaStreamSynchronize
    /* .graph_plan_create       = */ NULL,
//>> graph_plan_* 四个槽全是 NULL：这套接口 CUDA 没实现
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
//>> 整张图的执行入口，第 5、6、7 幕都从这里进去
    /* .event_record            = */ ggml_backend_cuda_event_record,
//>> 事件：把 ggml 的 event 映射成 cudaEvent（第 3 幕的 fork/join 用的就是它）
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['虚表槽', 'CUDA 实现', '这一槽干什么'],
      [['get_name / free', 'ggml_backend_cuda_get_name / _free', '名字与销毁'],
       ['set_tensor_async / get_tensor_async', 'ggml_backend_cuda_set_tensor_async ...', 'H2D / D2H 异步搬运'],
       ['set_tensor_2d_async / get_tensor_2d_async', '..._2d_async', '带 stride 的二维搬运'],
       ['cpy_tensor_async', 'ggml_backend_cuda_cpy_tensor_async', 'D2D / 跨设备，用 event 等流'],
       ['synchronize', 'ggml_backend_cuda_synchronize', 'cudaStreamSynchronize 当前流'],
       ['graph_plan_create / free / update / compute', 'NULL', '四个槽都没实现'],
       ['graph_compute', 'ggml_backend_cuda_graph_compute', '整张图的执行入口'],
       ['event_record / event_wait', 'ggml_backend_cuda_event_record / _wait', '映射到 cudaEvent'],
       ['graph_optimize', 'ggml_backend_cuda_graph_optimize', '执行前重排/融合图']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const mk = [[0], [], [], [], [9], [11], [16], [18, 20], [21]];
    const texts = [
      '16 个槽，按用途分四组看：搬运 / 同步 / 图 / 事件。',
      '<span class="k">搬运组</span>：set/get/cpy 是异步的 —— 它们只是发射，不等结果（L4-04）。',
      '<span class="k">同步组</span>：synchronize 把当前 stream 等干净。',
      '<span class="k">图组</span>：<span class="v">graph_plan_*</span> 四个槽是 NULL，说明"计划式执行"这套 CUDA 没用；<br>它自己用 cudaStreamBeginCapture 抓图（第 5 幕）。',
      '<span class="k">事件组</span>：event_record / event_wait 就是把 ggml 事件落到 cudaEvent 上。',
      '对照 L5-01：CPU 后端的同一个位置是一张填满的虚表 —— <span class="k">CUDA 用 NULL 明确表示"这一层能力不存在"</span>。'
    ];
    tl.at(700, () => { U.markLines(document, [0]); msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2100, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      U.markLines(document, mk[i]);
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(18000, () => { rows.forEach(x => { x.className = ''; }); U.markLines(document, [11]); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 图不是第一次就 capture：先<span class="hl-c">预热</span>，再录，之后只 launch */
{
  kicker: "L6-01 · graph capture",
  title: "图不是第一次就 capture：先<span class=\"hl-c\">预热</span>，再录，之后只 launch",
  sub: "同一张图连续两次\"属性没变\"，才值得录成 CUDA graph —— 因为图的地址和形状都被录死了。",
  caption: "capture 的起止：4472 起（cudaStreamBeginCapture）、4375 止（cudaStreamEndCapture）、4390 实例化（cudaGraphInstantiate）、4396 发射（cudaGraphLaunch）。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [0, 2, 7, 11, 14, 20, 26],
  lineNo: 4436,
  code: `    if (graph->is_enabled()) {
//>> 图的开关：第一张图建立时按架构决定（cc < Volta 直接禁用，见 4409）
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
//>> 兼容性检查：图里有节点不能进图（例如会同步 stream 的 MUL_MAT_ID 回退路径）就整张不用
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
//>> 预热阶段：第一次调用不录，直接执行
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
//>> 第二次属性没变 -> 预热完成，可以录了
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
//>> 属性变了（地址/形状变了）-> 退回预热，重新等一次稳定
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip a">第 1 次：直接执行</span><span class="arrow">-></span>
        <span class="chip b">第 2 次：属性没变</span><span class="arrow">-></span>
        <span class="chip c">capture</span><span class="arrow">-></span>
        <span class="chip d">之后：launch 重放</span>
      </div>
      <div class="row wrap" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '预热', m: 'warmup_complete = false', b: 'use_cuda_graph 保持 false，<br>按普通方式发 kernel。' },
      { c: 'b', t: '录图', m: 'cudaStreamBeginCapture', b: 'kernel 不执行，只被记进图；<br>录完 EndCapture + Instantiate。' },
      { c: 'c', t: '重放', m: 'cudaGraphLaunch', b: '一次调用重放整张图，<br>省的是 CPU 侧发射开销。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'capture 的前提是<span class="k">这张图稳定</span>：节点的指针、形状、属性连续两次一样。',
      '<span class="v">第 1 次</span>：properties_changed 为真 -> 不录，直接执行。',
      '<span class="v">第 2 次</span>：属性没变 -> warmup_complete = true，并打开 capture（4472 才开始录）。',
      '<span class="v">录的期间</span>：kernel 只是被记录，不产生计算。录完 4375 EndCapture、4390 Instantiate。',
      '<span class="v">之后</span>：4396 cudaGraphLaunch 重放整图。推理的 decode 阶段形状稳定，所以收益最大。',
      '两个"不录"的条件：架构低于 Volta（4409 cc &lt; GGML_CUDA_CC_VOLTA），或图里有会同步 stream 的节点（4437）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, [[0, 2], [7], [11, 14]][i] || []);
      msg.innerHTML = texts[i];
    }));
    tl.at(10900, () => { U.markLines(document, [11, 14]); msg.innerHTML = texts[3]; });
    tl.at(15500, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [26]); msg.innerHTML = texts[4]; });
    tl.at(19000, () => { U.markLines(document, [20]); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 一次遍历：跳过、融合、然后交给 <span class="hl-e">compute_forward</span> */
{
  kicker: "L6-01 · 执行",
  title: "一次遍历：跳过、融合、然后交给 <span class=\"hl-e\">compute_forward</span>",
  sub: "遍历 cgraph->n_nodes 的下标 i —— CUDA 后端的执行就是一个线性遍历。",
  caption: "注意 4359 的 GGML_ASSERT(ok)：compute_forward 返回 false 会直接断言失败 —— 所以\"支不支持\"必须在切图阶段就问清楚（L4-02）。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [2, 7, 12, 22, 43, 48],
  lineNo: 4316,
  code: `                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
//>> 视图/空节点直接跳过：它们不产生计算，只换读法（见 L1-01）
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
//>> 没有 GGML_TENSOR_FLAG_COMPUTE 的节点不算计算节点
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
//>> 融合：试着把从 i 开始的若干节点合成一个 kernel

                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
//>> 融合成功就跳过被吃掉的节点 —— 所以循环体里是 i += nodes_to_skip
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
//>> 真正的分派：一个节点一次 compute_forward，返回值表示"这个 op 我认不认"
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
//>> 断言：不认就直接挂。这是执行路径上的最后一道防线
`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="strip" style="gap:5px"></div>
      <div class="row wrap" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const names = ['NONE', 'VIEW', 'ADD', 'MUL_MAT', 'ROPE', 'RMS_NORM', 'SOFT_MAX', 'CPY'];
    const kind  = ['skip', 'skip', 'run', 'fuse', 'run', 'fuse', 'run', 'run'];
    const strip = wrap.querySelector('#strip');
    const nodes = names.map((n, i) => {
      const e = U.el('div', { class: 'card', style: 'width:78px;padding:5px 6px' });
      e.innerHTML = '<div class="cm" style="margin:0;font-size:9px">' + U.esc(n) + '</div>';
      strip.appendChild(e);
      return e;
    });
    nodes.forEach(e => { e.style.opacity = '.32'; });

    const defs = [
      { c: 'a', t: '跳过', b: 'VIEW / RESHAPE / TRANSPOSE / PERMUTE / NONE<br>以及没有 COMPUTE 标志的节点。' },
      { c: 'b', t: '融合', b: 'ggml_cuda_try_fuse() 返回要跳过的个数；<br>把多个节点合成一个 kernel。' },
      { c: 'c', t: '分派', b: '剩下的每个节点一次<br>ggml_cuda_compute_forward()。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'CUDA 后端拿到的是<span class="k">已经切好的子图</span>（L4-02 的产物），它只负责按序跑完。',
      '第一类：<span class="v">视图与空节点</span>直接跳过 —— 它们不产生计算（L1-01 的 view_src）。',
      '第二类：<span class="v">没有 GGML_TENSOR_FLAG_COMPUTE 的节点</span>不算计算节点，也跳过。',
      '第三类：<span class="v">融合</span>。4335 的 i += nodes_to_skip 就是"这几个节点已经合成一个 kernel 了"。',
      '剩下的每个节点：<span class="v">ggml_cuda_compute_forward(ctx, node)</span> —— 下一幕拆它的内部。',
      '如果它返回 false（没实现的 op），4359 的断言会直接终止程序。<span class="k">这是执行路径没有兜底的证据。</span>'
    ];
    tl.at(700, () => { U.markLines(document, []); msg.innerHTML = texts[0]; });
    tl.at(3000, () => {
      ['0', '1'].forEach(i => { nodes[+i].style.opacity = '1'; nodes[+i].style.borderLeftColor = 'var(--e)'; });
      msg.innerHTML = texts[1]; U.markLines(document, [2]);
    });
    tl.at(6400, () => { msg.innerHTML = texts[2]; U.markLines(document, [7]); });
    tl.at(9800, () => {
      [3, 5].forEach(i => { nodes[i].style.opacity = '1'; nodes[i].style.borderLeftColor = 'var(--c)'; });
      msg.innerHTML = texts[3]; U.markLines(document, [12, 22]);
    });
    tl.at(13800, () => {
      [2, 4, 6, 7].forEach(i => { nodes[i].style.opacity = '1'; nodes[i].style.borderLeftColor = 'var(--b)'; });
      msg.innerHTML = texts[4]; U.markLines(document, [43]);
    });
    tl.at(17800, () => { msg.innerHTML = texts[5]; U.markLines(document, [48]); });
  }
},

/* ------------------------------------------------------ 7 ★ 一个 <span class="hl-a">switch</span>：88 个 case 就是 CUDA 的 op 全表 */
{
  kicker: "L6-01 · 核心",
  title: "★ 一个 <span class=\"hl-a\">switch</span>：88 个 case 就是 CUDA 的 op 全表",
  sub: "ggml_cuda_compute_forward() 的全部内容：switch (dst->op)，每个 case 调一个 ggml_cuda_* 实现。",
  caption: "对照 L5-01：CPU 侧同形状的 switch 在 ggml/src/ggml-cpu/ggml-cpu.c:1744，签名是 void ggml_compute_forward(ggml_compute_params *, ggml_tensor *)。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [1, 3, 4, 15, 16, 21, 27],
  lineNo: 0,
  code: `static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
//>> 分派的全部依据：dst->op 一个字段（L1-02 的身份面）
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
//>> 每个 case 一行调用：真正的实现在各 ggml-cuda/*.cu 里（L6-02 / L6-03）
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:2415-2425 ----
        default:
            return false;
//>> 分派的全部依据：dst->op 一个字段（L1-02 的身份面）
    }

//>> 每个 case 一行调用：真正的实现在各 ggml-cuda/*.cu 里（L6-02 / L6-03）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;`,
  duration: 23000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ops = ['ARGMAX', 'COUNT_EQUAL', 'REPEAT', 'GET_ROWS', 'SET_ROWS', 'DUP', 'CPY', 'CONT',
                 'ADD', 'ADD1', 'ADD_ID', 'SUB', 'ACC', 'MUL', 'DIV', 'UNARY', 'GLU', 'NORM',
                 'RMS_NORM', 'CONCAT', 'PAD', 'MUL_MAT', 'MUL_MAT_ID', 'OUT_PROD', 'SOFT_MAX',
                 'ROPE', 'SSM_CONV', 'SSM_SCAN', 'FLASH_ATTN_EXT', 'OPT_STEP_ADAMW'];
    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 3px">案例（共 88 个 case 标签，行 2069-2412）</div>';
    const box = U.el('div', { class: 'row wrap', style: 'gap:4px' });
    left.appendChild(box);
    ops.forEach(o => box.appendChild(U.chip(o, 'a')));

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">一个 case = 一个算子家族</div>' +
      '<div class="cb">UNARY 与 GLU 在 case 里再套一层 switch（<span class="cm" style="margin:0">ggml_get_unary_op / ggml_get_glu_op</span>），' +
      '所以"88 个 case"对应的算子数还要多。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">default 只有两行</div>' +
      '<div class="cb"><span class="cm" style="margin:0">default: return false;</span>（2415-2416）' +
      '没有日志、没有回退 —— 认不认这个 op，答案就是 true / false。</div></div>' +
      '<div class="card" style="border-left-color:var(--g)">' +
      '<div class="ct" style="color:var(--g)">和 supports_op 是两份名单</div>' +
      '<div class="cb">这里分派"怎么跑"，第 9 幕的 supports_op 决定"切不切给我"。' +
      '两份名单不同步的后果就是第 6 幕那个断言。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '执行路径的核心就这一个 switch。<span class="k">88 个 case 标签，2069 到 2412 行</span>。',
      '分派依据只有一个字段：<span class="v">dst->op</span>。每个 case 一行调用，实现分散在各 ggml-cuda/*.cu。',
      '<span class="v">UNARY / GLU</span> 是嵌套 switch：同一个 GGML_OP_UNARY 下按 unary 子算子再分一次。',
      '<span class="v">default: return false</span>：没实现的 op 不会静默跳过，而是把 false 交回调用者。',
      '调用者（第 6 幕 4355-4359）把它变成 <span class="v">GGML_ASSERT(ok)</span> —— <span class="k">这就是"支持与否"必须提前判定的原因</span>。',
      '重点算子在哪：L6-02 讲 MUL_MAT 的量化路径（193/196 行的两个 case），L6-03 讲 FLASH_ATTN_EXT（301 行）。'
    ];
    tl.at(700, () => { U.markLines(document, []); msg.innerHTML = texts[0]; });
    tl.at(4000, () => { msg.innerHTML = texts[1]; U.markLines(document, [1, 3, 4]); });
    tl.at(8000, () => { msg.innerHTML = texts[2]; U.markLines(document, [3]); });
    tl.at(12000, () => { msg.innerHTML = texts[3]; U.markLines(document, [15, 16]); });
    tl.at(16000, () => { msg.innerHTML = texts[4]; U.markLines(document, [16]); });
    tl.at(19500, () => { msg.innerHTML = texts[5]; U.markLines(document, [21, 27]); });
  }
},

/* ------------------------------------------------------ 8 设备属性：<span class="hl-d">cc</span> 是一个整数，能力判断就是比大小 */
{
  kicker: "L6-01 · 设备属性",
  title: "设备属性：<span class=\"hl-d\">cc</span> 是一个整数，能力判断就是比大小",
  sub: "ggml_cuda_device_info 把每个设备的能力缓存下来；后面所有\"这个卡行不行\"都是读它。",
  caption: "cc 的编码：NVIDIA 走 ggml-cuda.cu:349 的 100*major + 10*minor；AMD / MUSA 各自加一个偏移（common.cuh:63-64）。",
  src: "ggml/src/ggml-cuda/common.cuh",
  mark: [5, 7, 11, 13, 17, 18, 25],
  lineNo: 1177,
  code: `struct ggml_cuda_device_info {
    int device_count;           // number of (possibly virtual) devices exposed to the rest of ggml
    int physical_device_count;  // number of physical CUDA devices actually present

    struct cuda_device_info {
        int     cc;                             // compute capability
//>> compute capability：所有能力判断的门槛值都在比它
        int     nsm;                            // number of streaming multiprocessors
//>> SM 个数：决定 kernel 的并行度参数
        size_t  smpb;                           // max. shared memory per block
        size_t  smpbo;                          // max. shared memory per block (with opt-in)
        bool    integrated;                     // Device is integrated as opposed to discrete
//>> 集显/独显：集显可以退回 host 可见内存（第 6 幕 4343 的断言就看它）
        bool    vmm;                            // virtual memory support
//>> VMM：显存池选 VMM 还是 legacy 就看这个字段（见 source.md 第四节）
        size_t  vmm_granularity;                // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;                      // Number of threads in a dispatch
        bool    supports_cooperative_launch;    // whether cooperative launch is supported
//>> 协作发射：需要跨 block 同步的 kernel 才用得上
        int     physical_device;                // backing physical CUDA device for this (virtual) device
        int     physical_share_count;           // number of (virtual) devices sharing this device's physical GPU
        int     virtual_index;                  // index of this (virtual) device among those sharing its physical GPU
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:9px"></div>
      <div class="row wrap" id="ladder" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'cc', m: 'int cc', b: '算力号：100 x 主版本 + 10 x 次版本。<br>7.5 -> 750。' },
      { c: 'b', t: '硬件规模', m: 'nsm · smpb · smpbo', b: 'SM 数、每 block 共享内存上限<br>（smpbo 是 opt-in 后的值）。' },
      { c: 'c', t: '形态与能力', m: 'integrated · vmm · warp_size', b: '集显/独显、虚拟内存支持、<br>一个 warp 多少线程。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const ladder = wrap.querySelector('#ladder');
    const steps = ['600 PASCAL', '610 DP4A', '700 VOLTA', '750 TURING', '800 AMPERE', '890 ADA', '900 HOPPER', '1200 BLACKWELL'];
    const chips = steps.map(s => { const c = U.chip(s, 'd'); ladder.appendChild(c); return c; });
    chips.forEach(c => { c.style.opacity = '.30'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '设备属性只在这里读一次（初始化时填进 info.devices[]），之后全靠 <span class="v">ggml_cuda_info()</span> 取。',
      '<span class="v">cc</span>：算力号。它是<span class="k">所有能力判断的共同货币</span> —— 比大小的判据都是它。',
      '<span class="v">nsm / smpb / smpbo</span>：决定 kernel 的网格与共享内存怎么开。',
      '<span class="v">integrated</span>：集显允许把输出放在 host 可见缓冲上（第 6 幕的断言专门给它开了口子）。',
      '能力号阶梯来自 common.cuh:50-62 的一组宏，本课 source.md 第二节逐字引用。',
      '能力判断用在哪：图 capture 的门槛 <span class="v">cc &lt; GGML_CUDA_CC_VOLTA</span>（4409）；<br>FlashAttention 等算子把 cc 交给专门的 *_supported()（5574）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, [[5], [7], [11, 13]][i] || []);
      msg.innerHTML = texts[i];
    }));
    tl.at(10300, () => { chips.forEach(c => { c.style.opacity = '1'; }); U.markLines(document, [11]); msg.innerHTML = texts[3]; });
    tl.at(14000, () => { els.forEach(e => { e.style.opacity = '1'; }); U.markLines(document, [25]); msg.innerHTML = texts[4]; });
    tl.at(17500, () => { U.markLines(document, [5]); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 ★ <span class="hl-a">supports_op</span>：能力号 + dtype 决定切不切给 CUDA */
{
  kicker: "L6-01 · 验收点",
  title: "★ <span class=\"hl-a\">supports_op</span>：能力号 + dtype 决定切不切给 CUDA",
  sub: "它只回 true / false，不加解释 —— 但 L4-02 的图切分完全建立在这一个函数上。",
  caption: "函数体：ggml-cuda.cu:5132-5591。它最后的兜底也是 default: return false（5588-5589）。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [0, 3, 6, 9, 10, 16],
  lineNo: 5132,
  code: `static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
//>> 第一道闸：数据在哪 —— 别的设备的 CUDA buffer 直接出局
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
//>> 只看 src 的 buffer type：是不是 CUDA 系的
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
//>> 设备号不一致就不支持：CUDA kernel 不会跨设备读
            }
        }
    }

    switch (op->op) {
//>> 第二道闸：op 白名单。这份名单要和第 7 幕的 switch 同步维护`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['判据', '在 supports_op 里的位置', '不满足就'],
      [['src 必须都在本设备', '5136-5143 遍历 GGML_MAX_SRC', 'false'],
       ['op 白名单', '5145 switch (op->op)', '落到 5588 false'],
       ['子算子白名单', '5146 UNARY / 5177 GLU 嵌套 switch', '5174 / 5188 false'],
       ['dtype 白名单', '5191 MUL_MAT 起 switch (a->type)', '不在列表 false'],
       ['布局约束', '5196 nb[0] != element_size', 'false'],
       ['参数约束', '5202 MUL_MAT_ID 且精度 F32', 'false'],
       ['设备能力号', '5206 读 devices[].cc（MUSA 分支）', 'false'],
       ['交给专门函数', '5574 FLASH_ATTN_EXT / 5586 LIGHTNING_INDEXER', '由 *_supported() 定']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const mk = [[3, 6, 9, 10], [16], [16], [], [], [], [], []];
    const texts = [
      'supports_op 是"图怎么切"的唯一依据。它没有"部分支持"这种答案。',
      '<span class="v">① 数据位置</span>：src 里有别的设备的 CUDA buffer 就不支持 —— 因为 kernel 不跨设备读。',
      '<span class="v">② op</span>：switch 里没有的 op 一路落到 5588 的 <span class="cm" style="margin:0">default: return false</span>。',
      '<span class="v">③ dtype</span>：MUL_MAT 只支持列表里的权重类型；<span class="k">列表外的量化类型会整体退回别的后端</span>（L4-02）。',
      '<span class="v">④ 布局 / 参数</span>：nb[0] 不是元素大小（被转置过）、或精度参数是 F32 的 MUL_MAT_ID，都直接 false。',
      '<span class="v">⑤ 能力号</span>：CUDA 下多数算子不看 cc；<span class="k">真正按卡分档的算子把 cc 交给专门的 *_supported()</span>（L6-03）。',
      '一句话：<span class="k">支持 = 数据在本设备 + op 在名单里 + dtype 在名单里 + 布局/参数合法 + 能力号够</span>。'
    ];
    tl.at(700, () => { U.markLines(document, [0]); msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      U.markLines(document, mk[i]);
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(21500, () => { rows.forEach(x => { x.className = ''; }); U.markLines(document, [0, 16]); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 10 把 CUDA 后端压成一张表 */
{
  kicker: "L6-01 · 收束",
  title: "把 CUDA 后端压成一张表",
  sub: "接口 → 上下文 → 图 → 分派 → 判据 → 显存：六件事，一条链。",
  caption: "下一课 L6-02 拆 MUL_MAT 在 CUDA 上的三条实现路线（mmq / mmvq / mmf）。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [0, 1, 3, 6],
  lineNo: 4421,
  code: `static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
//>> 一进来先 set_device：上下文绑哪个设备，活儿就发到哪

    bool use_cuda_graph             = false;
//>> 三个开关变量决定这次是"直接执行"还是"录图/重放"
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['环节', '关键代码', '一句话', '展开'],
      [['接口', 'ggml_backend_cuda_interface', '三张虚表里的第一张', 'L3-01'],
       ['上下文', 'stream(device, stream)', '懒创建、非阻塞、随上下文销毁', 'L4-04'],
       ['图', 'ggml_backend_cuda_graph_compute', '预热 → capture → launch', 'L4-04'],
       ['分派', 'ggml_cuda_compute_forward', '88 个 case 的 switch', 'L6-02 / L6-03'],
       ['判据', 'ggml_backend_cuda_device_supports_op', '位置 + op + dtype + 布局 + 能力', 'L4-02'],
       ['显存', 'ggml_cuda_pool', '每 (设备, stream) 一个池，RAII 归还', 'L4-01']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      'CUDA 后端怎么决定一个 op 支不支持？说出至少四条判据。',
      '<span class="mono">ggml_backend_cuda_device_supports_op()</span>（ggml-cuda.cu:5132-5591）按顺序判：<br>' +
      '① <b>数据位置</b>：遍历 <span class="mono">GGML_MAX_SRC</span> 个 src，只要有别的设备的 CUDA buffer 就 false（5136-5143）；<br>' +
      '② <b>op 白名单</b>：<span class="mono">switch (op->op)</span>，名单外落到 5588 的 <span class="mono">default: return false</span>；<br>' +
      '③ <b>dtype 白名单</b>：如 MUL_MAT 的 <span class="mono">switch (a->type)</span>（5191 起）不在列表就 false；<br>' +
      '④ <b>布局与参数</b>：<span class="mono">nb[0] != ggml_element_size(a)</span>（5196）、MUL_MAT_ID 且精度 F32（5202）都 false；<br>' +
      '⑤ <b>设备能力号</b>：部分算子读 <span class="mono">ggml_cuda_info().devices[].cc</span>（5206），' +
      '或交给 <span class="mono">ggml_cuda_flash_attn_ext_supported()</span>（5574）这类专门函数。<br>' +
      '结论直接决定 L4-02 的图切分：返回 false 的节点会被切给别的后端。'));

    wrap.querySelector('#ex').appendChild(W.exercise(
      '第 6 幕的遍历里，为什么 <span class="mono">GGML_ASSERT(ok)</span> 是"危险"的？',
      '因为执行路径<span class="k">没有兜底</span>：<span class="mono">ggml_cuda_compute_forward()</span> 的 ' +
      '<span class="mono">default</span> 只返回 false（2415-2416），遍历里的处理是打一条日志然后 ' +
      '<span class="mono">GGML_ASSERT(ok)</span>（4356-4359）—— 直接断言失败。<br>' +
      '所以"这个 op 我能不能跑"必须在切图阶段用 supports_op 判掉（第 9 幕 / L4-02），' +
      '而不是指望执行时能优雅退化。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '六个环节，一条链：虚表定契约，上下文管资源，graph_compute 管执行。',
      '<span class="k">接口与上下文</span>：三张虚表 + 一个 (设备, stream) 二维资源表。',
      '<span class="k">图与分派</span>：整图先决定要不要 capture，再逐节点 skip / fuse / compute_forward。',
      '<span class="k">判据</span>：supports_op 用位置 + op + dtype + 布局 + 能力号回答"支不支持"。',
      '<span class="k">显存</span>：池按 (设备, stream) 划分，RAII 归还 —— 见 L4-01 的分配器。',
      '离开这一课时记住一句话：<span class="v">CUDA 后端的架子 = 一个 switch + 一个 stream，加一个 supports_op 的判据</span>。'
    ];
    tl.at(700, () => { U.markLines(document, [0]); msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      U.markLines(document, i === 2 ? [3] : (i === 0 ? [1] : []));
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); U.markLines(document, [6]); msg.innerHTML = texts[5]; });
  }
},

];
