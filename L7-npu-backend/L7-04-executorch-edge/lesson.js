/* ==========================================================================
   L7-04 · ExecuTorch 后端（边缘/移动端）：离线编译的设备内核与运行期派发
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 ET 后端：<span class="hl-a">68 个文件</span>分成两个世界 */
{
  kicker: "L7 · NPU 与加速器后端",
  title: "ET 后端：<span class=\"hl-a\">68 个文件</span>分成两个世界",
  sub: "宿主侧 11 个文件说\"怎么派发\"，设备侧 56 个文件说\"内核怎么写、怎么被加载\"。",
  caption: "文件清单来自 tools/plan_matrix.py --files 里 L7-04 的 68 项，全部进本课覆盖声明。",
  src: "ggml/include/ggml-et.h",
  mark: [0, 5, 9, 14, 18],
  lineNo: 10,
  code: `#define GGML_ET_NAME "ET"
//>> 后端名字：注册表（L3-02）与 buft 名字都用它

// backend API
GGML_BACKEND_API ggml_guid_t     ggml_backend_et_guid(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_et_init(size_t devidx);
//>> 唯一的建后端入口：拿一个 devidx，返回 ggml_backend_t

GGML_BACKEND_API bool ggml_backend_is_et(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_et_get_device_count(void);
//>> 设备枚举：几个 ET 设备（runtime->getDevices()）
GGML_BACKEND_API void ggml_backend_et_get_device_description(int devidx, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_et_get_device_memory(int devidx, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_et_buffer_type(size_t dev_num);
//>> 设备缓冲类型：本课"数据面"的关键 —— 设备内存
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_et_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_et_reg(void);
//>> 注册入口：GGML_BACKEND_DL_IMPL 动态加载时找的就是它（L3-03）`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip">GGUF 权重</span><span class="arrow">-></span>
        <span class="chip a">ggml 计算图</span><span class="arrow">-></span>
        <span class="chip c">ET 后端（宿主侧 C++）</span><span class="arrow">-></span>
        <span class="chip b">rt::IRuntime</span><span class="arrow">-></span>
        <span class="chip d">ET-SoC 上的 RISC-V 内核</span>
      </div>
      <div id="fam"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['族', '文件', '行数', '是什么'],
      [['公共 API', '1', '28', 'ggml-et.h：init / is_et / buffer_type / reg'],
       ['宿主侧后端', '11', '6126', '虚表 + 内核加载 + 算子参数 + CPU 对拍'],
       ['设备侧共享头', '6', '2854', '同一份 ggml 张量结构 / 张量指令 / 量化块'],
       ['设备侧内核 .c', '50', '14183', '交叉编译成 ELF，跑到 ET-SoC 上']],
      { monoCols: [1, 2] });
    wrap.querySelector('#fam').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '五步流水线里，<span class="k">后两步</span>是本课：宿主侧怎么派发，设备侧内核从哪来。',
      '<span class="v">1 + 11</span> 个文件在宿主：公共 API + 三张虚表 + 内核加载 + 50 个算子的参数打包。',
      '<span class="v">6</span> 个设备侧共享头是"共同语言"：<span class="k">ggml_tensor.h</span> 与宿主 '
        + '<span class="k">struct ggml_tensor</span> 同名同义，<span class="k">quants.h</span> 直接 include '
        + '<span class="k">ggml-common.h</span>。',
      '<span class="v">50</span> 个 <span class="k">.c</span> 全部是裸机内核：没有 libc、没有 ggml 运行时，'
        + '只有结构体 + 裸指针 + 内联汇编。',
      '所以本课的视角是：<span class="k">这条流水线的最后两跳是怎么接上的</span>。'
    ];
    rows.forEach(function (r, i) { tl.at(600 + i * 2600, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i];
    }); });
    tl.at(11800, function () { msg.innerHTML = texts[4]; });
    tl.at(15000, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = '本课会反复回到这张表：<span class="k">宿主侧的 12 个文件</span>负责"翻译成内核调用"，'
        + '<span class="k">设备侧的 56 个文件</span>负责"在内核里把活干完"。';
    });
  }
},

/* ------------------------------------------------------ 2 运行期没有编译：一张图 = 一个 <span class="hl-c">for</span> + 一个 <span class="hl-c">switch</span> */
{
  kicker: "L7-04 · 运行期",
  title: "运行期没有编译：一张图 = 一个 <span class=\"hl-c\">for</span> + 一个 <span class=\"hl-c\">switch</span>",
  sub: "graph_compute 逐节点分派，只在两处做图级融合（RMS_NORM+MUL、MUL_MAT+ADD）。",
  caption: "★ 对照 L7-03：OpenVINO 在运行期把子图翻成 ov::Model 再编译；ET 这里没有任何\"编译\"调用。",
  src: "ggml/src/ggml-et/ggml-et.cpp",
  mark: [2, 5, 9, 16, 22],
  lineNo: 657,
  code: `static ggml_status ggml_backend_et_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_graph(&dev_ctx->uberkernel);
//>> uberkernel：本图要打包成"一个设备内核"时，先开一段

    for (int i = 0; i < cgraph->n_nodes; i++) {
//>> 遍历拓扑序（L1-03）里的每个节点；整张图的执行就是这个循环
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
//>> VIEW / RESHAPE / PERMUTE / TRANSPOSE 不产生内核调用，直接跳过
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        // --- Fusion checks (before regular dispatch) ---
        if (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
//>> 唯一的图级融合：RMS_NORM + MUL 合成一个内核 rms_norm_mul_f32
            ggml_et_op_rms_norm_mul(dev_ctx, node, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            continue;
        }
        if (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
//>> 第二个融合：MUL_MAT + ADD（bias）合成一次 mul_mat 调用
            ggml_et_op_mul_mat(dev_ctx, node, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            continue;
        }
`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="flow" id="nodes" style="justify-content:center"></div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const nodes = ['VIEW', 'MUL_MAT', 'ADD', 'RMS_NORM', 'MUL'];
    const host = wrap.querySelector('#nodes');
    const nels = nodes.map(function (n) {
      const c = U.chip(n);
      host.appendChild(c); host.appendChild(U.arrow('->'));
      return c;
    });
    const defs = [
      { c: 'a', t: 'for (i < cgraph->n_nodes)', b: '逐节点。跳过 VIEW 族，<br>其余每个节点一次内核调用。' },
      { c: 'b', t: 'switch (node->op)', b: '39 个 case + default。<br>每个 case 调一个 ggml_et_op_*。' },
      { c: 'e', t: 'default -> FAILED', b: '图里有不支持的 op 就整图失败，<br>运行期没有兜底编译。' }
    ];
    const cards = defs.map(function (d) { const e = U.card(d, { style: 'width:220px' }); wrap.querySelector('#cards').appendChild(e); return e; });
    cards.forEach(function (e) { e.style.opacity = '.30'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '五个节点，两次融合：<span class="v">MUL_MAT + ADD</span> 与 <span class="v">RMS_NORM + MUL</span>。',
      'i=0：VIEW 直接 <span class="k">continue</span> —— 没有内核，也没有内存。<br>这就是 L4-01 里"零拷贝视图"在执行期的样子。',
      'i=1..2：命中融合 <span class="v">MUL_MAT + ADD</span>，一次调用算完两个节点，<span class="k">i++</span> 跳过 ADD。',
      'i=3..4：命中融合 <span class="v">RMS_NORM + MUL</span>，同样一次调用。',
      '整张图算完，<span class="k">ggml_backend_et_graph_compute</span> 返回 <span class="v">SUCCESS</span> —— '
        + '中间<span class="k">没有编译、没有代码生成、没有图优化 pass</span>。'
    ];
    function mark(i) { nels.forEach(function (e, k) { e.className = 'chip' + (k === i ? ' a' : ''); }); }
    tl.at(600, function () { msg.innerHTML = texts[0]; mark(0); });
    tl.at(3000, function () { msg.innerHTML = texts[1]; mark(0); });
    tl.at(6200, function () { mark(1); });
    tl.at(8600, function () { msg.innerHTML = texts[2]; mark(1); });
    tl.at(11800, function () { mark(3); });
    tl.at(13600, function () { msg.innerHTML = texts[3]; mark(3); });
    tl.at(16400, function () { mark(4); cards.forEach(function (e) { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ "编译"发生在<span class="hl-a">图之外</span>：内核是预编译的 ELF，运行期按名字取 */
{
  kicker: "L7-04 · ★ 核心洞察",
  title: "★ \"编译\"发生在<span class=\"hl-a\">图之外</span>：内核是预编译的 ELF，运行期按名字取",
  sub: "这段代码里没有任何编译器：只有\"名字 -> 字节 -> runtime->loadCode() -> KernelId\"。",
  caption: "★ 与 CANN / OpenVINO 的\"运行期映射\"是两条路线：ET 把翻译提前到构建期（本幕末说明构建期的做法）。",
  src: "ggml/src/ggml-et/ggml-et-kernels.cpp",
  mark: [8, 15, 20, 33, 43, 48],
  lineNo: 133,
  code: `bool ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\\n");
        return false;
    }

    // Check if kernel already loaded
    if (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
//>> 已经加载过就直接返回 —— 名字就是缓存键
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\\n", kernel_name.c_str(), dev_ctx->devidx);
        return true;
    }

    std::vector<std::byte> kernel_data;
    const char *           kernels_path = getenv("GGML_ET_KERNELS_PATH");
//>> 先看环境变量 GGML_ET_KERNELS_PATH（开发期用文件覆盖内置内核）

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    if (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
//>> ★ 名字拼出来的是 .elf —— 一个已经编译好的设备可执行文件，不是一个"算子描述"
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        if (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\\n", kernel_name.c_str(), kernel_file.c_str());
        } else {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    if (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
//>> 没给路径（或文件不存在）就退回**嵌进宿主库的那份** ELF 字节
        if (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\\n", kernel_name.c_str());
            return false;
        }
    }

    try {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
//>> 把这段 ELF 字节交给运行时；由它下载到设备并返回句柄
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
//>> 句柄按名字存进 map，下次 ggml_et_launch_kernel 直接命中
        return true;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\\n", kernel_name.c_str(), e.what());
        return false;
    }
}`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards" style="gap:8px"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: '构建期', b: 'RISC-V 工具链把 src/*.c<br>交叉编译成 <b>50 个 .elf</b>', m: 'et-kernels/CMakeLists.txt' },
      { c: 'a', t: '嵌入', b: '每个 .elf 变成 C 数组，<br>编进 libggml-et 本体', m: 'ggml/src/ggml-et/CMakeLists.txt' },
      { c: 'b', t: '运行期 1', b: 'GGML_ET_KERNELS_PATH 下<br>有同名 .elf 就优先用它', m: 'getenv("GGML_ET_KERNELS_PATH")' },
      { c: 'd', t: '运行期 2', b: 'runtime->loadCode() 交给设备，<br>拿回 rt::KernelId 存进 map', m: 'loaded_kernels[name] = kernel_' }
    ];
    const cards = defs.map(function (d) { const e = U.card(d, { style: 'width:163px' }); wrap.querySelector('#cards').appendChild(e); return e; });
    cards.forEach(function (e) { e.style.opacity = '.30'; });

    const t = U.table(
      ['问题', '答案', '依据'],
      [['代码在哪编译？', '构建期，交叉编译成设备 ELF', 'et-kernels/CMakeLists.txt:41-43'],
       ['运行期编译什么？', '什么都不编译，只加载与派发', 'ggml-et-kernels.cpp:133-183'],
       ['"委托"发生在哪？', '不在图级：逐个 op 调用 + 名字查内核', 'ggml-et.cpp:681-842（39 个 case）'],
       ['有 .pte / torch 吗？', '没有：全目录搜不到这些符号', '本课第 8 幕的对照表']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '四个步骤里，<span class="k">前两步在构建期</span>，<span class="k">后两步在运行期</span>。'
        + '所以运行期的 graph_compute 只剩执行。',
      '构建期：<span class="v">50</span> 个 <span class="k">.c</span> 各自是一个 ELF（对应 CMake 的 KERNELS 列表 50 项）。',
      '嵌入：ELF 字节被 embed 成 C 数组编进宿主库 —— 部署时<span class="k">不需要带一堆 .elf 文件</span>。',
      '运行期：<span class="k">GGML_ET_KERNELS_PATH</span> 只用于开发期覆盖；生产走内置那份。',
      '加载：<span class="v">loadCode()</span> 之后才有 <span class="v">rt::KernelId</span>；'
        + '句柄按名字缓存，<span class="k">懒加载</span>发生在第一次 launch 时。'
    ];
    const seq = [0, 0, 1, 2, 3, 4];
    seq.forEach(function (idx, i) {
      tl.at(600 + i * 4200, function () {
        cards.forEach(function (e, k) { e.style.opacity = (k === Math.max(0, idx)) ? '1' : '.30'; });
        msg.innerHTML = texts[i];
        rows.forEach(function (r, k) { r.className = (k === Math.min(i, 3)) ? 'on' : ''; });
      });
    });
  }
},

/* ------------------------------------------------------ 4 真正的"运行时"是 SDK 的两个对象，不是 ggml */
{
  kicker: "L7-04 · 谁在派发",
  title: "真正的\"运行时\"是 SDK 的两个对象，不是 ggml",
  sub: "ggml 侧只持有两个 shared_ptr：设备层 + 运行时；流、内核句柄、事件都挂在设备上下文里。",
  caption: "跨课呼应 L3-01：ggml 的三张虚表是\"契约\"；这一层之下，ET SDK 自己还有一套对象模型。",
  src: "ggml/src/ggml-et/ggml-et.cpp",
  mark: [1, 3, 7],
  lineNo: 70,
  code: `static struct ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
//>> 设备层：屏蔽"真硬件 PCIe 卡"与"sysemu 模拟器"的差别
    std::shared_ptr<rt::IRuntime>                     runtime;
//>> 运行时：建流、分配设备内存、传数据、加载代码、启动内核
    std::unique_ptr<std::ofstream>                    profile_stream;
    std::unique_ptr<std::ofstream>                    kernel_id_stream;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
//>> profiling 时把"名字 -> KernelId"导出成 JSON（kernel_id.json）
    bool                                              profiling_enabled = false;
} _drv;`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['对象', '类型', '本课证据'],
      [['_drv.device_layer', 'dev::IDeviceLayer', 'createPcieDeviceLayer / createSysEmuDeviceLayer（153-157）'],
       ['_drv.runtime', 'rt::IRuntime', 'IRuntime::create(device_layer)（160）'],
       ['dev_ctx->default_stream', 'rt::StreamId', '一个设备一条默认流，内核按序执行（1771）'],
       ['dev_ctx->loaded_kernels', 'map<string, rt::KernelId>', '名字 -> 设备内核句柄（common.h:75, kernels.cpp:176）'],
       ['dev_ctx->uberkernel_enabled', 'bool', '由环境变量 GGML_ET_UBERKERNEL 打开（1759-1760）']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const defs = [
      { c: 'b', t: 'rt::IRuntime', b: 'mallocDevice / memcpyHostToDevice<br>loadCode / kernelLaunch / waitForEvent' },
      { c: 'd', t: 'dev::IDeviceLayer', b: 'createPcieDeviceLayer()<br>createSysEmuDeviceLayer(opts)' },
      { c: 'a', t: 'ggml 侧', b: '只是这两个对象的持有者：<br>虚表 + 设备上下文' }
    ];
    const cards = defs.map(function (d) { const e = U.card(d, { style: 'width:214px' }); wrap.querySelector('#cards').appendChild(e); return e; });
    cards.forEach(function (e) { e.style.opacity = '.30'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这三个名字是理解本课"边界在哪"的关键：<span class="k">ggml 只管契约，SDK 管设备</span>。',
      '<span class="v">dev::IDeviceLayer</span> 是同一份代码能跑在真卡与模拟器上的原因（编译期用 -DGGML_ET_SYSEMU=ON 选）。',
      '<span class="v">rt::IRuntime</span> 提供的是"设备级原语"：内存、流、事件、内核 —— 与 CUDA Runtime 同一层次。',
      '注意：<span class="k">没有</span> "compile(graph)" 这样的接口 —— 这套运行时只有 loadCode 与 kernelLaunch。',
      '所以派发路线是 <span class="v">ggml 节点 -> ggml_et_op_* -> ggml_et_launch_kernel -> rt::kernelLaunch</span>。'
    ];
    rows.forEach(function (r, i) { tl.at(600 + i * 3000, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      cards.forEach(function (e, k) { e.style.opacity = (k === 0 || k === 2) ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }); });
    tl.at(16000, function () { cards.forEach(function (e) { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 ★ 跨边界的不是裸指针，是<span class="hl-a">整个 struct ggml_tensor</span> */
{
  kicker: "L7-04 · ★ 数据面",
  title: "★ 跨边界的不是裸指针，是<span class=\"hl-a\">整个 struct ggml_tensor</span>",
  sub: "宿主侧把张量结构体按值拷进参数块；设备侧于是拿到同一套 ne/nb/type/data 词汇表。",
  caption: "★ 这就是验收点说的\"数据面抽象的不同\"：L6-01 的 CUDA 内核只收到裸指针 + 维度。",
  src: "ggml/src/ggml-et/ggml-et-ops.cpp",
  mark: [5, 7, 12],
  lineNo: 277,
  code: `    float scale, bias;
    memcpy(&scale, (const float *) node->op_params + 0, sizeof(float));
    memcpy(&bias, (const float *) node->op_params + 1, sizeof(float));

    ggml_et_scale_params params;
    params.src0  = *node->src[0];
//>> ★ 不是取 src0->data，而是把 src0 整个结构体拷进参数（含 data 里的设备地址）
    params.dst   = *node;
//>> ★ 输出张量同样整个拷过去：设备内核由此知道 dst->ne / dst->nb
    params.scale = scale;
    params.bias  = bias;

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, sizeof(params), 0xFFFFFFFF);
//>> 参数块整体交给 ggml_et_launch_kernel；名字 "scale_f32" 在设备侧对应一个 ELF`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip c">宿主: params.src0 = *node-&gt;src[0]</span>
        <span class="arrow">-&gt;</span>
        <span class="chip b">设备: struct ggml_tensor * src0 = &amp;params-&gt;src0</span>
      </div>
      <div class="row center" id="cards" style="gap:8px"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['字段', '宿主侧写的值', '设备内核拿它做什么'],
      [['ne[4]', '形状', '算总元素数、按 64B cache line 切分线程（scale_f32.c:55-61）'],
       ['nb[4]', '步长', '按行跳：unary_f32.c 用全部四个 nb[] 走 4D 视图（unary_f32.c:482-492）'],
       ['type', 'GGML_TYPE_F32', '类型检查，不匹配直接 return -1（scale_f32.c:35-37）'],
       ['data', '设备地址（mallocDevice 给的）', '直接当指针解引用（scale_f32.c:39-40）'],
       ['（全部字段）', 'op / op_params / name / buffer...', '原样拷过去；设备侧只读它需要的那几个']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const defs = [
      { c: 'a', t: 'CUDA（L6-01）', b: 'scale_f32(x, dst, scale, bias, nelements)<br>只有裸指针 + 标量', m: 'ggml/src/ggml-cuda/scale.cu:5' },
      { c: 'b', t: 'ET（本课）', b: 'params.src0 = *node->src[0]<br>整个结构体按值', m: 'ggml-et-ops.cpp:282' }
    ];
    const cards = defs.map(function (d) { const e = U.card(d, { style: 'width:300px' }); wrap.querySelector('#cards').appendChild(e); return e; });
    cards.forEach(function (e) { e.style.opacity = '.45'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '同一个 SCALE 算子：CUDA 传 5 个标量/指针，ET 传 2 个完整结构体 + 2 个 float。',
      '<span class="k">为什么可以这样？</span>因为设备侧有一份同名同义的 '
        + '<span class="v">struct ggml_tensor</span>（et-kernels/src/ggml_tensor.h 自己声明一遍）。',
      '结构体里的 <span class="v">data</span> 是宿主 <span class="k">runtime->mallocDevice()</span> 给的地址；'
        + '参数块被逐字节拷到设备后，这个地址在设备上<span class="k">直接可用</span>。',
      '这就是与 GPU 后端最本质的差别：<span class="k">元数据也跟着数据一起过边界</span>，'
        + '设备侧不用再翻译一遍形状语言。',
      '代价：设备内核里的 <span class="v">op_params / name / buffer</span> 等宿主指针毫无意义 —— '
        + '内核只能碰它真正需要的那几个字段（第 6 幕看设备侧怎么用）。'
    ];
    rows.forEach(function (r, i) { tl.at(600 + i * 3400, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i, 2)];
    }); });
    tl.at(17600, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[3] + '<br>' + texts[4];
    });
  }
},

/* ------------------------------------------------------ 6 设备内核：同一个结构体，<span class="hl-c">裸机</span>环境里再声明一遍 */
{
  kicker: "L7-04 · 设备侧",
  title: "设备内核：同一个结构体，<span class=\"hl-c\">裸机</span>环境里再声明一遍",
  sub: "entry_point(params, env)：参数是宿主打包好的那块内存，env 告诉它有多少个 hart 可用。",
  caption: "设备侧没有 malloc、没有 ggml 运行时、没有 libc：参数就是唯一输入，线程数由 shire_mask 推出。",
  src: "ggml/src/ggml-et/et-kernels/src/scale_f32.c",
  mark: [1, 3, 9, 17, 25, 33],
  lineNo: 11,
  code: `struct ggml_et_scale_params {
    struct ggml_tensor src0;   // F32 input tensor
//>> 与宿主 struct ggml_tensor 同名同义：两端各写一遍声明，靠"字段一致"对齐
    struct ggml_tensor dst;    // F32 output tensor
//>> 输出张量也是按值传进来的 —— 设备侧由此知道写多少、往哪写
    float              scale;  // Scale factor
    float              bias;   // Bias (additive offset)
};

int entry_point(struct ggml_et_scale_params * params, void * env) {
//>> 内核入口签名固定：int entry_point(<参数结构> *, void * env)
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
//>> env 里带着 shire_mask：本内核在多少个 hart 上并行，由调度它的那次 launch 决定
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor * src0 = &params->src0;
//>> 直接把参数块当成 ggml_tensor 用 —— 不需要任何反序列化
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    float * src0_data = (float *) src0->data;
//>> ★ data 是设备地址：这里解引用就是真的读到显存里的数据
    float * dst_data  = (float *) dst->data;

    if (!src0_data || !dst_data) {
        return -1;
    }
`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards" style="gap:8px"></div>
      <div id="bars" style="width:100%"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: 'entry_point(params, env)', b: '内核唯一入口。<br>params 是宿主打包的参数块。', m: 'int entry_point(...)' },
      { c: 'd', t: 'env = kernel_environment_t', b: '只带 shire_mask / frequency；<br>线程数 = popcount(mask) x 32 x 2', m: 'platform.h:99-102' },
      { c: 'a', t: '纯裸机', b: '-nostdlib / -ffreestanding；<br>向量运算全是内联汇编', m: 'flw.ps / fmadd.ps / fsw.ps' }
    ];
    const cards = defs.map(function (d) { const e = U.card(d, { style: 'width:214px' }); wrap.querySelector('#cards').appendChild(e); return e; });
    cards.forEach(function (e) { e.style.opacity = '.30'; });

    const host = wrap.querySelector('#bars');
    const bars = [];
    for (let i = 0; i < 4; i++) {
      const b = U.bar('hart ' + i, 'b');
      host.appendChild(b.el); bars.push(b);
    }
    bars.forEach(function (b) { b.fill.style.width = '0%'; b.val.textContent = '0%'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '宿主给的参数块在这里被<b>原地当成结构体</b>用；设备侧不做任何形状翻译。',
      '<span class="v">params-&gt;src0</span> / <span class="v">params-&gt;dst</span>：两个 ggml_tensor 就在参数里。',
      '每个 hart 先算出自己的 <span class="k">thread_id</span>，再按 <span class="k">shire_mask</span> 推出总线程数。',
      '分片单位不是元素，而是 <span class="v">64B cache line</span>（16 个 f32）—— '
        + '所以宿主侧才要求 <span class="k">ne[0] % 16 == 0</span>（第 7 幕）。',
      '最后 8 个一组做 <span class="v">fmadd.ps</span>：一条指令 8 个 float，'
        + '<span class="k">dst = src * scale + bias</span> 只用了三行汇编。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; cards[0].style.opacity = '1'; });
    tl.at(3400, function () { msg.innerHTML = texts[1]; cards[0].style.opacity = '1'; });
    tl.at(6800, function () { msg.innerHTML = texts[2]; cards[1].style.opacity = '1'; });
    tl.at(10200, function () {
      msg.innerHTML = texts[3]; cards[2].style.opacity = '1';
      bars.forEach(function (b) { b.fill.style.width = '25%'; b.val.textContent = '25%'; });
    });
    tl.at(14600, function () { msg.innerHTML = texts[4]; });
    tl.at(18000, function () {
      msg.innerHTML = '把这一端的契约记住：<span class="k">给我一块参数内存，我按 ggml 的字段语义干完活</span>。'
        + '下一幕看宿主侧为了保证这件事成立，加了哪些约束。';
    });
  }
},

/* ------------------------------------------------------ 7 内存契约：设备内存、<span class="hl-c">cache 对齐</span>、没有 host 映射 */
{
  kicker: "L7-04 · 数据面契约",
  title: "内存契约：设备内存、<span class=\"hl-c\">cache 对齐</span>、没有 host 映射",
  sub: "buffer_type 虚表把\"这块内存是怎么来的\"讲清楚了；supports_op 再把内核的假设写成前置条件。",
  caption: "跨课呼应 L4-03：buffer 是\"后端对内存的假设\"；ET 的假设比 GPU 后端更强。",
  src: "ggml/src/ggml-et/ggml-et.cpp",
  mark: [2, 8, 14, 19],
  lineNo: 432,
  code: `static size_t ggml_backend_et_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes_pad(tensor);
//>> 分配大小不是 nbytes，而是补齐到 cache line 的 nbytes_pad
}

static bool ggml_backend_et_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
//>> ★ 明确声明"我的 buffer 不是 host 内存" —— 与 CPU / CUDA pinned 路径区分开
}

static const struct ggml_backend_buffer_type_i ggml_backend_et_buffer_type_i = {
    /* .get_name         = */ ggml_backend_et_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_et_buffer_type_alloc_buffer,
//>> alloc_buffer 走 runtime->mallocDevice（384-411），不是 malloc
    /* .get_alignment    = */ ggml_backend_et_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_et_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_et_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_et_buffer_type_is_host,
//>> is_host 槽位就是上面那个恒 false 的函数
};`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards" style="gap:9px"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '设备内存', b: 'runtime->mallocDevice(rtid, size)<br>free 也是 runtime->freeDevice', m: 'ggml-et.cpp:404 / 247' },
      { c: 'c', t: '对齐', b: 'get_alignment 返回设备属性的<br>cacheLineSize', m: 'ggml-et.cpp:413-422' },
      { c: 'e', t: '无 host 映射', b: 'host_buffer / is_host 都是 false，<br>buffer_from_host_ptr 是 NULL', m: 'ggml-et.cpp:1649 / 439 / 1681' }
    ];
    const cards = defs.map(function (d) { const e = U.card(d, { style: 'width:220px' }); wrap.querySelector('#cards').appendChild(e); return e; });
    cards.forEach(function (e) { e.style.opacity = '.30'; });

    const t = U.table(
      ['内核的假设', '写在哪', '为什么'],
      [['nb[0] == sizeof(float)', '961-963 / 936', '行内连续，SIMD 载入才合法'],
       ['ne[0] % 16 == 0', '875 / 945-948', '64B cache line = 16 个 f32'],
       ['ggml_is_contiguous', '875 / 886 / 893', '按 cache line 平铺、一次写满'],
       ['buffer 只能是 ET 的 buft', '1577-1579', 'supports_buft 比的是 get_name 指针'],
       ['buffer 之间不能直接拷', '333-340', 'cpy_tensor 恒返回 false']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '三张卡片是"内存从哪来"，下面五条是"内核因此能假设什么"。',
      '<span class="v">nb[0] == sizeof(float)</span>：只要求行内连续；更高维可以是任意步长（unary_f32.c 用四个 nb[] 走行）。',
      '<span class="v">ne[0] % 16 == 0</span>：分片单位是 cache line，所以行宽必须是 16 个 f32 的整数倍。',
      '这两条一起解释了第 6 幕里那个 <span class="k">elements_per_cacheline = 16</span>。',
      '<span class="v">is_host = false</span> + <span class="v">host_buffer = false</span>：'
        + '宿主内存不能当 ET 的设备内存用，任何输入都必须 <span class="k">memcpyHostToDevice</span>（290-310）。',
      '还有一条隐性的：<span class="v">cpy_tensor</span> 恒 false —— 设备缓冲之间没有"直接拷"这条路，'
        + '要搬数据得回到 host（对照 CUDA 的 cpy_tensor_async）。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; cards[0].style.opacity = '1'; });
    tl.at(3600, function () { rows.forEach(function (r, k) { r.className = (k === 0) ? 'on' : ''; }); msg.innerHTML = texts[1]; });
    tl.at(7000, function () { rows.forEach(function (r, k) { r.className = (k === 1) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
    tl.at(10400, function () {
      rows.forEach(function (r) { r.className = ''; });
      cards[1].style.opacity = '1'; cards[2].style.opacity = '1';
      msg.innerHTML = texts[3];
    });
    tl.at(14200, function () { rows.forEach(function (r, k) { r.className = (k === 4) ? 'on' : ''; }); msg.innerHTML = texts[4]; });
    tl.at(17400, function () { rows.forEach(function (r) { r.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 与 GPU 后端比：<span class="hl-a">数据面抽象</span>差在哪 */
{
  kicker: "L7-04 · 验收点",
  title: "与 GPU 后端比：<span class=\"hl-a\">数据面抽象</span>差在哪",
  sub: "同一套 ggml_backend_i 契约之下，两边\"过边界的东西\"完全不同。",
  caption: "★ 一句话：CUDA 过边界的是\"指针 + 形状参数\"，ET 过边界的是\"ggml 自己的张量结构体\"。",
  src: "ggml/src/ggml-et/ggml-et-common.h",
  mark: [2, 5],
  lineNo: 23,
  code: `struct ggml_backend_et_buffer_context {
    int          devidx;
    void *       data;  // Device memory pointer
//>> ★ 注释写明：这是设备内存指针 —— 不是 host 指针，也不是 mmap 的文件页
    size_t       size;
    rt::DeviceId rtid;
//>> 每个 buffer 记住自己的 rt::DeviceId：以后 free / 传输都要用它
};

struct ggml_backend_et_context {
    int devidx;
};`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['数据面问题', 'ET（本课 L7-04）', 'CUDA（L6-01）', 'CANN / OpenVINO（L7-01 / L7-03）'],
      [['过边界传什么',
        'struct ggml_tensor 按值<br>ggml-et-ops.cpp:282',
        '裸指针 + 标量 + 元素数<br>scale.cu:5 / 37-50',
        '厂商描述符：aclCreateTensor（acl_tensor.cpp:89）；ov::Tensor(..., tensor->data)（ggml-decoder.cpp:1392）'],
       ['设备侧看到的元数据',
        'ne / nb / type / data 全在<br>scale_f32.c:32-40',
        '只有入参里的维度与指针',
        '厂商自己的 shape / stride（L7-01 / L7-03 展开）'],
       ['量化块布局',
        '设备侧直接 include ggml-common.h<br>quants.h:10-11',
        '复用 ggml 的 block_* 定义',
        '建图/映射时转成厂商格式（L7-01 / L7-03 展开）'],
       ['设备内存谁分配',
        'runtime->mallocDevice<br>ggml-et.cpp:404',
        'cudaMalloc（显存池）',
        '厂商 device 内存（L7-01 / L7-03 展开）'],
       ['host 内存能当设备内存吗',
        '不能：is_host=false（439）<br>host_buffer=false（1649）',
        '能：pinned host buft（1309-1317）<br>caps.host_buffer 默认 true（5099）',
        '不共享，靠拷贝（L7-05 讲虚拟化下更甚）'],
       ['缓冲区之间能直接拷吗',
        '不能：cpy_tensor 恒 false（339）',
        '能：cpy_tensor_async（2485）',
        '（L7-01 / L7-03 展开）']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '六行问题，三种答案。ET 这一列的共同点：<span class="k">把 ggml 的表示原样带过边界</span>。',
      '<span class="v">传什么</span>：ET 传结构体，CUDA 传指针 + 形状 —— 这是最本质的一行。',
      '<span class="v">元数据</span>：ET 的设备内核能自己读 ne/nb，所以内核可以通用地处理 4D 视图；'
        + 'CUDA 的 kernel 只能靠宿主算好参数。',
      '<span class="v">量化块</span>：ET 连 <span class="k">block_q8_0</span> 的定义都复用 ggml-common.h（L1-04 的那份）。',
      '<span class="v">内存</span>：ET 没有 host 内存路径、没有 buffer 互拷 —— '
        + '这两条在 L4-03 的 buffer 假设里是"后端能力"，在 ET 上直接是 false。',
      '结论：<span class="k">ET 的数据面是"共享 ggml 表示 + 独立设备内存"</span>；'
        + 'GPU 的数据面是"共享内存地址空间 + 独立形状表示"。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) {
      tl.at(3000 + i * 3000, function () {
        rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
        msg.innerHTML = texts[Math.min(1 + i, 4)];
      });
    });
    tl.at(20600, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L7-04 · 收束",
  title: "把这一课压成一张表",
  sub: "运行期只有执行与失败两种结局；所有\"翻译\"都发生在构建期。",
  caption: "下一课 L7-05：VirtGPU —— 客户机与宿主机之间的命令流、共享内存与同步。",
  src: "ggml/src/ggml-et/ggml-et.cpp",
  mark: [2, 7, 14],
  lineNo: 838,
  code: `            default:
                ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(node->op));
//>> 运行期遇到没实现（或 supports_op 判错）的 op：整图失败，没有兜底编译
                return GGML_STATUS_FAILED;
        }

        if (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
//>> uberkernel 打包失败同样整图失败 —— 不留半张图的结果
            ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
            return GGML_STATUS_FAILED;
        }
    }

    if (!ggml_et_uberkernel_end_graph(dev_ctx)) {
//>> "end_graph" 才是真正把 uberkernel 段发射出去的地方
        ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex" style="width:100%"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案', '依据'],
      [['这是什么后端', 'ET 名字下的 RISC-V manycore 加速器后端', 'ggml-et.h:10'],
       ['什么时候"编译"', '构建期：50 个 .c 交叉编译成 50 个 .elf 并嵌入宿主库', 'CMakeLists.txt:28-79, 172-207'],
       ['运行期做什么', 'for + switch：39 个 case 分派到 ggml_et_op_*', 'ggml-et.cpp:661-842'],
       ['"委托"发生在哪一级', '内核级：名字 -> loadCode -> KernelId；没有 .pte / torch', 'ggml-et-kernels.cpp:151-176'],
       ['数据面抽象', 'struct ggml_tensor 按值 + 设备地址；设备侧复用 ne/nb/type 词汇表', 'ggml-et-ops.cpp:282 / scale_f32.c:32-40'],
       ['与 GPU 最大差别', '无 host 内存路径（439）、无 buffer 互拷（339）', 'ggml-et.cpp:333-340, 437-440'],
       ['下一课', 'L7-05 VirtGPU：虚拟化下的命令流与共享内存', '../L7-05-virtgpu-virtualized/']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '这个后端的"编译"发生在什么时候？运行期 <span class="mono">graph_compute</span> 到底做哪几件事？',
      '<b>编译在构建期</b>：<span class="mono">et-kernels/CMakeLists.txt</span> 用 RISC-V 工具链把 '
      + '<span class="mono">et-kernels/src/*.c</span> 逐个编译成裸机 ELF（<span class="mono">-nostdlib</span>），'
      + '<span class="mono">ggml/src/ggml-et/CMakeLists.txt</span> 再把它们嵌进 libggml-et。<br>'
      + '<b>运行期只做三件事</b>：<span class="mono">ggml_backend_et_graph_compute</span> 逐节点走 '
      + '<span class="mono">for</span>（661）-> 两处融合判断（670/675）-> <span class="mono">switch</span> '
      + '分派到 <span class="mono">ggml_et_op_*</span>（681-842），每个 op 把参数打包后 '
      + '<span class="mono">ggml_et_launch_kernel</span>；内核第一次用到时懒加载 '
      + '（<span class="mono">ggml-et-kernels.cpp:199-216</span>）。<br>'
      + '<b>没有</b>图级编译 / 代码生成：搜遍 <span class="mono">ggml/src/ggml-et/</span> 没有 '
      + '<span class="mono">torch</span> / <span class="mono">delegate</span> / <span class="mono">.pte</span>。'));

    wrap.querySelector('#ex').appendChild(W.exercise(
      '同样一个 <span class="mono">GGML_OP_SCALE</span>，ET 后端和 CUDA 后端"送到设备上的东西"有什么不同？'
      + '这带来什么后果？',
      '<b>CUDA</b>：宿主算好指针与元素数，kernel 收 5 个参数 '
      + '<span class="mono">scale_f32(const float * x, float * dst, float scale, float bias, int64_t nelements)</span>'
      + '（<span class="mono">ggml/src/ggml-cuda/scale.cu:5</span>）；设备侧<b>看不到</b> ggml 的元数据。<br>'
      + '<b>ET</b>：宿主把整个张量结构体按值拷进参数块 —— '
      + '<span class="mono">params.src0 = *node-&gt;src[0]; params.dst = *node;</span>'
      + '（<span class="mono">ggml-et-ops.cpp:282-283</span>），设备内核直接 '
      + '<span class="mono">src0-&gt;ne / nb / type / data</span>（<span class="mono">scale_f32.c:32-40</span>）。<br>'
      + '<b>后果</b>：① 设备内核能自己处理 4D 步长（<span class="mono">unary_f32.c</span> 用全部四个 '
      + '<span class="mono">nb[]</span> 走行），不用宿主把形状拆成参数；'
      + '② 代价是<b>内核必须与 ggml 的结构体布局同步</b>（设备侧要再声明一遍 '
      + '<span class="mono">struct ggml_tensor</span>）；'
      + '③ 参数里那些宿主指针（<span class="mono">buffer</span> / <span class="mono">name</span>）在设备上毫无意义，内核只能读它需要的字段。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这张表就是本课的全部结论：<span class="k">翻译在构建期，执行在运行期</span>。',
      '第 2 行是本课的核心：50 个 ELF 在构建期就存在了，运行期只是把它们搬到设备上。',
      '第 4 行回答课名的疑问：这里的"委托"是<b>内核级</b>的，不是图级的 .pte 导出。',
      '第 5、6 行是验收点：数据面共享 ggml 表示，但内存完全独立。',
      '下一课看另一种"设备"：虚拟化 GPU —— 那里连设备内存都不能直接映射。'
    ];
    tl.at(600, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) { tl.at(3000 + i * 3200, function () {
      rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
    }); });
    tl.at(19600, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[4];
    });
  }
},

];
