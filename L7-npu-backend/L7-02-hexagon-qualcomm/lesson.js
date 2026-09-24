/* ==========================================================================
   L7-02 · Hexagon 后端（Qualcomm NPU/DSP）：host 与 DSP 的边界
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一个后端，<span class="hl-a">两个处理器</span> */
{
  kicker: "L7 · NPU 与加速器后端",
  title: "一个后端，<span class=\"hl-a\">两个处理器</span>",
  sub: "Hexagon 后端横跨 host（ARM/x86）与 DSP（Hexagon HTP）两颗核；本课覆盖的 87 个文件分属两边。",
  caption: "上游位置：ggml/src/ggml-hexagon/ 与 ggml/include/ggml-hexagon.h，v0.5.0。",
  src: "ggml/include/ggml-hexagon.h",
  mark: [1, 4, 7],
  lineNo: 10,
  code: `// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);
//>> 唯一的"造后端"入口；从没见过的设备名、域号都在内部处理

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);
//>> 判据是「虚表里的 get_name 是不是这个后端的函数」—— 见 L3-01 的三张虚表

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);
//>> 注册入口符号；L3-03 讲过它在动态库里怎么被 dlsym 找到`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">ggml 计算图</span><span class="arrow">-></span>
        <span class="chip a">host · ggml-hexagon.cpp</span><span class="arrow">-></span>
        <span class="chip c">DSP · htp/main.c</span><span class="arrow">-></span>
        <span class="chip b">HVX / HMX kernel</span>
      </div>
      <div class="row center" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '公共 API 只有 3 个', b: 'init / is_hexagon / reg。<br>其余 8000 多行都在后端内部。',
        m: 'ggml_backend_hexagon_reg()' },
      { c: 'c', t: '87 个文件分两边', b: 'host 侧 6 个源文件（2 .cpp + 4 .h，<br>含公共头）；DSP 侧 81 个 C 文件。',
        m: 'ggml/src/ggml-hexagon/htp/' },
      { c: 'b', t: '跨架构的代价', b: '内存要跨核分配、命令要排队、<br>同步要自己做 —— 本课的主线。',
        m: 'rpcmem / dspqueue / fence' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看这个后端的<b>公共面积</b>：只有三行声明。',
      '<span class="v">ggml_backend_hexagon_init()</span> 造一个后端实例，' +
      '<span class="v">ggml_backend_is_hexagon()</span> 认它，' +
      '<span class="v">ggml_backend_hexagon_reg()</span> 把自己注册给 ggml。',
      '但实现横跨<b>两颗处理器</b>：host 侧（<span class="k">ggml-hexagon.cpp</span>）与 ' +
      'DSP 侧（<span class="k">htp/</span> 下的 81 个文件）。左边那 6 个编译进 ' +
      '<span class="v">libggml-hexagon.so</span>，右边 81 个用 Hexagon 工具链单独编成 DSP 骨架库。',
      '所以这个后端的接口不只是"提交 kernel"：<br>' +
      '<span class="v">远端内存</span> + <span class="v">命令队列</span> + <span class="v">同步</span> ' +
      '都得它自己搭。这就是本课要讲的三件事。',
      '对照 L7-01 的 CANN：同是 NPU 后端，那边的控制面来自厂商运行时（ACL / aclnn），' +
      '这边的控制面（IDL + 队列 + 内存映射）是后端自己搭的。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15100, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 2 注册：HTP 就是一个<span class="hl-b">普通 ggml 后端</span> */
{
  kicker: "L7-02 · host 侧",
  title: "注册：HTP 就是一个<span class=\"hl-b\">普通 ggml 后端</span>",
  sub: "名字叫 \"HTP\"，虚表结构和别的后端一模一样 —— 特殊之处全在背后。",
  caption: "回顾 L3-01：`reg_i` / `dev_i` / `backend_i` 三张函数指针表就是后端契约；这一课看到它的第 N 个实现。",
  src: "ggml/src/ggml-hexagon/ggml-hexagon.cpp",
  mark: [0, 8, 19, 25, 35],
  lineNo: 8038,
  code: `static const struct ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
//>> 四行函数指针 = L3-01 的"注册虚表"，与 CUDA / CANN 完全同构
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            auto nErr = htpdrv_init();
//>> 先加载 FastRPC 驱动；加载不到就直接返回 NULL
            if (nErr != AEE_SUCCESS) {
                return NULL;
            }

            ggml_hexagon_init(&reg);
//>> 真正的初始化：探测设备、解析 domain、准备设备上下文
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '名字：HTP', b: 'reg->get_name() 返回字符串 "HTP"；<br>设备个数来自 opt_ndev（默认 1）。',
        m: 'return "HTP";' },
      { c: 'g', t: '驱动不在就整个消失', b: 'htpdrv_init() 失败 -> reg() 返回 NULL。<br>' +
           '后端不是"报错"，而是<b>在运行期不存在</b>。',
        m: 'if (nErr != AEE_SUCCESS) { return NULL; }' },
      { c: 'b', t: '然后才是普通后端', b: '有了设备上下文，后面就是<br>buffer / graph_compute / supports_op 的老套路。',
        m: 'GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">ggml_backend_hexagon_reg()</span> 只做一件事：把一个静态 ' +
      '<span class="v">ggml_backend_reg</span> 填好交出去。',
      '虚表里四项：<span class="k">get_name</span> / <span class="k">get_device_count</span> / ' +
      '<span class="k">get_device</span> / <span class="k">get_proc_address</span>。<br>' +
      '和 L3-01 讲的契约逐字对应 —— 这就是"后端"这个词的全部含义。',
      '关键差异在 <span class="v">htpdrv_init()</span>：它去 <span class="k">dlopen</span> 厂商的 ' +
      'FastRPC 驱动。驱动不在（非高通机器），这个后端<b>根本不出现</b>在 ' +
      '<span class="v">ggml_backend_reg_get_device</span> 的列表里。',
      '<span class="v">GGML_BACKEND_DL_IMPL</span> 导出符号 —— 见 L3-03：主库靠固定符号名找到它。',
      '一句话：<span class="k">对外它只是个普通后端；特殊之处是它背后站着第二颗处理器。</span>'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(18400, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ host 侧的会话：<span class="hl-a">句柄、队列、队列号</span> —— 接口不止是"提交 kernel" */
{
  kicker: "L7-02 · 核心",
  title: "★ host 侧的会话：<span class=\"hl-a\">句柄、队列、队列号</span> —— 接口不止是\"提交 kernel\"",
  sub: "一个 session 里同时住着\"远端句柄\"和\"命令队列\"两个东西，这在同一颗芯片的 GPU 后端里是看不到的。",
  caption: "对照 L6-06（Vulkan 主机端）：那边靠驱动的统一虚拟地址与隐式同步；这边每一样都要显式建。",
  src: "ggml/src/ggml-hexagon/ggml-hexagon.cpp",
  mark: [2, 4, 8, 17],
  lineNo: 457,
  code: `struct ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
//>> FastRPC 远端句柄：htp_iface_* 的所有调用都挂在这个句柄上
    dspqueue_t       queue;
//>> host 侧命令队列对象；DSP 用下面那个 queue_id 导入同一个队列
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
//>> dspqueue_export() 得到的跨域队列号 —— 32 位域里是它，不是指针
    int              phys_idx;
    int              virt_idx;
    bool             valid_session;
    bool             valid_handle;
    bool             valid_queue;
    bool             valid_iface;

    ggml_hexagon_opbatch* op_batch;
//>> 按批打包的算子请求；下一幕看它怎么被填满并写出去`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['字段', '类型', '跨边界', '它解决什么问题'],
      [['handle', 'remote_handle64', '→ DSP', '远端会话。open 之后所有 htp_iface_* 调用走它'],
       ['queue', 'dspqueue_t', '← 本地', '命令队列的 host 侧句柄，写请求 / 读响应'],
       ['queue_id', 'uint64_t', '→ DSP', 'dspqueue_export 出来的队列号，DSP 用它 import 同一个队列'],
       ['op_batch', 'ggml_hexagon_opbatch*', '← 本地', '攒算子的批缓冲，满了才写一次队列']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '先看这个结构体里<b>跨边界</b>的几个字段。',
      '<span class="v">remote_handle64 handle</span>：控制通道（既能调 htp_iface_* 函数，' +
      '也能设 QOS、开 profiler）。',
      '<span class="v">dspqueue_t queue</span>：数据通道。注意它是 <b>host 侧的对象</b>，' +
      '指针在 DSP 上毫无意义。',
      '<span class="v">uint64_t queue_id</span>：所以必须 <span class="k">dspqueue_export()</span> ' +
      '换一个跨域可见的号，DSP 才能 import 到同一条队列。',
      '<span class="v">op_batch</span>：一个 batch 里塞最多 <span class="v">opt_opbatch</span>（默认 1280）' +
      '个算子；攒批是为了摊薄跨核通信的开销。',
      '<span class="k">这就是核心洞察</span>：GPU 后端的"提交"是一句驱动调用；' +
      '这里的"提交"要先有一条<b>跨处理器的队列</b>，还要一个<b>远端句柄</b>管控制面。',
      '回顾 L3-01：后端接口是"虚表"；本课补的是 <b>虚表下面那层传输</b>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2800 + i * 3100, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
    tl.at(19500, () => { msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 4 远端内存：<span class="hl-b">rpcmem</span> → <span class="hl-b">fd</span> → <span class="hl-b">fastrpc_mmap</span> */
{
  kicker: "L7-02 · host 侧",
  title: "远端内存：<span class=\"hl-b\">rpcmem</span> → <span class=\"hl-b\">fd</span> → <span class=\"hl-b\">fastrpc_mmap</span>",
  sub: "DSP 看不到 host 的 malloc 堆，host 也递不过去一个指针 —— 唯一的通行证是「文件描述符」。",
  caption: "DSP 侧的对应动作在 htp/main.c 的 htp_iface_mmap()：收到 fd 后调 HAP_mmap 建自己的虚拟地址。",
  src: "ggml/src/ggml-hexagon/ggml-hexagon.cpp",
  mark: [8, 13, 20, 26],
  lineNo: 578,
  code: `struct ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    int       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
//>> ① rpcmem 分配：这块内存 host 与 DSP 都能访问（物理连续、可被 FastRPC 映射）
        if (!base) {
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
//>> ② 取 fd —— 这是唯一能跨过 FastRPC 边界的"钥匙"
        if (fd < 0) {
            rpcmem_free(base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        this->size = size;
    }
//>> ③ 记住 size；真正的映射在 ggml_hexagon_shared_buffer::mmap() 里做

    ~ggml_hexagon_rpcmem_block() {
//>> 析构时释放 —— 与 mmap/munmap 的配对在 shared_buffer 层
        if (base) {
            rpcmem_free(base);
        }
    }
};`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">rpcmem_alloc2</span><span class="arrow">-></span>
        <span class="chip a">base 指针</span><span class="arrow">-></span>
        <span class="chip c">rpcmem_to_fd</span><span class="arrow">-></span>
        <span class="chip d">fd</span><span class="arrow">-></span>
        <span class="chip b">fastrpc_mmap</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: '为什么不用 malloc', b: '普通堆内存不在 FastRPC 的可映射范围里；<br>要用厂商的 rpcmem 分配器。',
        m: 'RPCMEM_HEAP_ID_SYSTEM' },
      { c: 'd', t: 'fd 是跨域的钥匙', b: 'DMA-BUF 式的 fd 在两个域里<br>指向同一块物理内存。',
        m: 'fd = rpcmem_to_fd(base)' },
      { c: 'b', t: '映射还是各做各的', b: 'host 侧 fastrpc_mmap；DSP 侧收到 fd 后<br>自己 HAP_mmap 出一段虚拟地址。',
        m: 'fastrpc_mmap(domain_id, fd, ...)' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    const pick = i => els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '三步走。第一步：<span class="v">rpcmem_alloc2()</span> —— <b>不是 malloc</b>。',
      '第二步：<span class="v">rpcmem_to_fd(base)</span> 拿到 fd。' +
      '<span class="k">指针不能跨域，fd 可以。</span>',
      '第三步：<span class="v">fastrpc_mmap()</span> 把这块内存映射进 FastRPC 的域。' +
      '源码里还有一条：非 pinned 的 buffer 解除映射前要先通知 DSP ' +
      '（<span class="v">htp_iface_munmap</span>）。',
      'DSP 那边对称地做一遍：<span class="v">htp_iface_mmap(fd, size)</span> → ' +
      '<span class="v">HAP_mmap2(...)</span>，拿到 DSP 自己的虚拟地址。',
      '<span class="k">同一块内存，两个虚拟地址，靠 fd 对应起来。</span><br>' +
      '这就是"跨两个处理器"最直接的后果 —— 也是 L7-05 虚拟化 GPU 会再次遇到的同一个问题。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => { pick(i); msg.innerHTML = texts[i]; }));
    tl.at(10900, () => { pick(2); msg.innerHTML = texts[3]; });
    tl.at(14500, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 对岸：<span class="hl-c">dspqueue_import</span> 与一个连续块 */
{
  kicker: "L7-02 · DSP 侧",
  title: "对岸：<span class=\"hl-c\">dspqueue_import</span> 与一个连续块",
  sub: "DSP 侧入口 htp_iface_start() 收到队列号，导入同一条队列，然后把线程池、DMA 队列、VTCM 一次性摆好。",
  caption: "远端接口的签名由 IDL 定义（htp/htp_iface.idl，非源文件、不计入覆盖率）：start / stop / mmap / munmap / profiler / etm / hwinfo。",
  src: "ggml/src/ggml-hexagon/htp/main.c",
  mark: [4, 10],
  lineNo: 383,
  code: `    dspqueue_t dsp_queue = NULL;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    int err = dspqueue_import(dsp_queue_id, NULL, NULL, (void *) h, &dsp_queue);
//>> DSP 侧 import 主机创建的队列 —— 同一条队列，两个地址空间里的两个句柄
    if (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(HIGH, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (void *) h, &dsp_queue);
//>> 老设备不支持 NULL 回调时的回退：改用回调驱动 process_ops()
    }

    if (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        return err;
    }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '导入队列', b: '同一个 queue_id，DSP 侧得到<br>自己的 dspqueue_t。',
        m: 'dspqueue_import(queue_id, ...)' },
      { c: 'c', t: '问硬件要资源', b: 'qurt_sysenv_get_max_hw_threads +<br>qurt_hvx_get_units 得到可用 HVX 线程数。',
        m: 'n_hvx = (qurt_hvx_get_units() >> 8) & 0xFF' },
      { c: 'b', t: '一个连续块装下所有', b: 'ctx + 栈 + work_queue + dma_queue +<br>hmx_queue 按 offset 排进同一块内存。',
        m: 'memalign(4096, footprint)' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'host 侧的 <span class="v">htp_iface_start(handle, sess_id, dsp_queue_id, n_hvx, n_hmx, max_vmem)</span> ' +
      '是一次真实的远端调用；DSP 侧落在这个函数里。',
      '<span class="v">dspqueue_import()</span>：host 建、host 导出、DSP 导入 —— ' +
      '队列是<b>共享对象</b>，不是单向发送。',
      '导入之后才知道硬件有多宽：<span class="v">qurt_hvx_get_units()</span> 给出 HVX 单元数，' +
      '再被 <span class="v">hw_threads.max_hthreads</span> 和 <span class="v">HTP_MAX_NTHREADS</span> 依次夹紧。',
      '然后是一次性资源摆放：<span class="v">vtcm_alloc()</span> 拿 VTCM，' +
      '<span class="v">work_queue_init()</span> 建线程池，每个线程一个 <span class="v">dma_queue_init()</span>。',
      '最后 <span class="v">qurt_thread_create()</span> 起 <span class="k">htp-main</span> 线程，' +
      '循环 <span class="v">dspqueue_peek()</span> 等请求 —— 这就是 DSP 侧的"主循环"。',
      '回收也对称：<span class="v">htp_iface_close()</span> 逐个 <span class="v">htp_munmap</span> 掉 ' +
      '<span class="v">ctx-&gt;mmap[]</span> 里的映射，再释放整块内存。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(10800, () => { msg.innerHTML = texts[3]; });
    tl.at(14200, () => { msg.innerHTML = texts[4]; });
    tl.at(17000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 一次 <span class="hl-d">dspqueue_write</span> 里装了什么 */
{
  kicker: "L7-02 · 边界",
  title: "一次 <span class=\"hl-d\">dspqueue_write</span> 里装了什么",
  sub: "请求信封是一个定长头 + 紧跟其后的三个数组；DSP 侧用同一个公式把它切回来。",
  caption: "三个数组同处 dspqueue buffer 0：bufs → tensors → ops → prof。host 侧 `ggml_hexagon_opqueue::push()` 按同样顺序 memcpy。",
  src: "ggml/src/ggml-hexagon/htp/htp-ops.h",
  mark: [1, 2, 4, 5, 7],
  lineNo: 235,
  code: `struct htp_opbatch_req {
    uint64_t seq;         // Sequence number
    uint32_t n_bufs;      // Number of buffers
//>> 本次用到的内存块个数（去重后的 buffer 列表）
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
//>> 本次要跑多少个算子 —— 一个 batch 可以塞上千个
    uint32_t n_traces;    // Number of trace descriptors per thread
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
//>> 三个数组按 bufs / tensors / ops 顺序排在同一个 dspqueue buffer 里
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip d">htp_opbatch_req（头 24 字节）</span><span class="arrow">|</span>
        <span class="chip a">buf_desc × n_bufs</span><span class="arrow">|</span>
        <span class="chip b">htp_tensor × n_tensors</span><span class="arrow">|</span>
        <span class="chip c">htp_op_desc × n_ops</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'buf：内存块', b: '每个是 {base, size, flags, fd}。<br>DSP 侧按 fd 复用已有映射。',
        m: 'struct htp_buf_desc' },
      { c: 'b', t: 'tensor：张量描述', b: 'data 是<b>块内偏移</b>（不是指针），<br>DSP 侧再换成自己的虚拟地址。',
        m: 'struct htp_tensor' },
      { c: 'c', t: 'op：算子描述', b: 'opcode + 标量参数 + <b>host 预算好的</b> kernel 参数<br>+ src/dst 张量下标。',
        m: 'struct htp_op_desc' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '跨边界的<b>不是指针，是描述符数组</b>。',
      '<span class="v">n_bufs</span>：这次要用到几块内存。一个模型的权重被切在少数几个大块里，' +
      '所以这个数很小。',
      '<span class="v">n_tensors</span>：几张张量。上限 <span class="v">HTP_OP_MAX_TENSORS</span> = 8192，' +
      '注释写明"必须留在 uint16 以内"。',
      '<span class="v">n_ops</span>：几个算子。<span class="k">一个 batch 可以装上千个算子</span> —— ' +
      '攒批是为了把跨核通信成本摊薄。',
      '响应信封 <span class="v">htp_opbatch_rsp</span> 用同一个 <span class="v">seq</span> 配回来，' +
      '带上 status、耗时、每线程的 trace 事件数。',
      '<span class="k">一次 write = 一个 batch = 上千个算子。</span>' +
      '这就是"命令队列"在这个后端里的真实粒度。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13900, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 HTP 图 = ggml 图的<span class="hl-e">翻译结果</span>，不是另一张图 */
{
  kicker: "L7-02 · HTP 图",
  title: "HTP 图 = ggml 图的<span class=\"hl-e\">翻译结果</span>，不是另一张图",
  sub: "每个 ggml 节点被包成一个 htp_opnode：换成 HTP opcode、挂上预计算的 kernel 参数、接上融合链。",
  caption: "回顾 L1-03：图的结构（节点 + src[] 边）没变；变的是每个节点的“身份”从 `GGML_OP_*` 换成 `HTP_OP_*`。",
  src: "ggml/src/ggml-hexagon/htp-opnode.h",
  mark: [1, 3, 5, 8, 12, 14, 27, 35, 41],
  lineNo: 23,
  code: `struct htp_opnode {
    ggml_tensor * node   { nullptr };
//>> 仍然指回原来那个 ggml_tensor —— 不复制数据，只做包装
    htp_op_code   opcode { HTP_OP_INVALID };
//>> 身份换成 HTP 的 opcode（op_remap_to_htp 得到）
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};
//>> host 预先算好的内核参数；DSP 侧直接读，不再做这些判断

    std::vector<ggml_tensor *>                fused;
//>> 被融合进来的后续节点（如 MUL_MAT + ADD），替 DSP 省掉一次往返
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<const ggml_tensor *> inputs;
//>> 输入/输出都是 ggml_tensor 指针 —— 图的边原样保留
    std::vector<const ggml_tensor *> outputs;
    std::string                      name;

    int n_active_src(const ggml_tensor * t) const {
        if (!t) return 0;
        for (int i = GGML_MAX_SRC - 1; i >= 0; i--) {
            if (t->src[i]) {
                return i + 1;
            }
        }
        return 0;
    }

    void init(ggml_tensor * node) {
        this->node = node;
        if (this->node) {
            this->name = ggml_op_desc(this->node);

            // Build inputs (preserving optional nullptrs)
            int n_inputs = n_active_src(this->node);
            this->inputs.resize(n_inputs, nullptr);
            for (int i = 0; i < n_inputs; i++) {
//>> 输入来自 node->src[i]；这就是 L1-01 里说的"图的边"
                this->inputs[i] = this->node->src[i];
            }

            // Build outputs
            this->outputs.push_back(this->dst());
        }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['字段', '从哪来', 'DSP 侧看到的东西'],
      [['node', 'ggml 图上的那个节点', '不用（DSP 只收描述符）'],
       ['opcode', 'op_remap_to_htp(node)', 'htp_op_desc.opcode'],
       ['kernel_params', 'host 预计算（如 tiled 权重布局、分块数）', 'htp_op_desc.kernel_params[32]'],
       ['inputs / outputs', 'node->src[] 与 dst', 'htp_op_desc.src[10] / dst[4]（下标）'],
       ['fused / dummy', '融合链（RMS_NORM+MUL 等）', '合并成同一个 opcode']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '“HTP 图”这个词容易误解：它不是重新建一张图，而是<b>把 ggml 图逐节点翻译</b>。',
      '<span class="v">node</span> 保留 ggml_tensor 指针 —— 数据面完全没动（L1-01 的结构体还是那个）。',
      '<span class="v">opcode</span> 是翻译结果：<span class="k">GGML_OP_* → HTP_OP_*</span>。' +
      'DSP 上的 <span class="v">execute_op</span> 大 switch 就是按它分发的。',
      '<span class="v">kernel_params</span> 是这套设计的关键：<b>能用 host 算的，就不留给 DSP 算</b>。' +
      '例如矩阵乘的分块数与 tiled 行大小，host 在多线程里先算好。',
      '<span class="v">inputs / outputs</span> 仍是 ggml_tensor 指针，编码时才换成数组下标' +
      '（<span class="v">uint16_t src[10]</span>）—— 指针留在 host，下标过了边界。',
      '<span class="v">fused</span>：融合发生在建 opnode 的时候。<span class="v">add_fused()</span> ' +
      '把中间结果从输入里摘掉、把新输入接上 —— 于是<b>一次跨核往返干两件事</b>。',
      '这就是"图"在这个后端的角色：<span class="k">它的价值是让 host 能在发送前做完全局优化。</span>'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2900 + i * 3000, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(18100, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 8 ★ DSP 内部：<span class="hl-c">线程池</span> + <span class="hl-c">栅栏</span> */
{
  kicker: "L7-02 · 核心",
  title: "★ DSP 内部：<span class=\"hl-c\">线程池</span> + <span class=\"hl-c\">栅栏</span>",
  sub: "一个 batch 在 DSP 上被切成 n_threads 份并行跑；跑完靠一个\"序号 + 状态\"的栅栏对齐。",
  caption: "回顾 L5-01：CPU 后端用 `ggml_barrier` 对齐线程；这里是跨设备版本 —— 槽位放在共享内存里，两端都看得见。",
  src: "ggml/src/ggml-hexagon/htp/work-queue.h",
  mark: [0, 3, 6, 9, 11, 13, 16, 22],
  lineNo: 13,
  code: `#define WORK_QUEUE_MAX_N_THREADS      10
//>> 上限 10，与 HTP_MAX_NTHREADS 一致（main.c 里有 _Static_assert 把关）

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size);
//>> size/align/init 三件套：线程池本身也放在那块连续内存里
size_t       work_queue_alignof(void);
work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size);
void         work_queue_free(work_queue_t q);

void work_queue_wakeup(work_queue_t q);
//>> 唤醒 / 挂起：一个 batch 开始前唤醒，结束时挂起
void work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n);
//>> 把一个函数按 n 份并行展开 —— DSP 侧所有算子的多线程都走这一条路

static inline bool work_queue_run(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    if (n <= 1) {
//>> n <= 1 时直接在本线程执行，不起线程（省掉同步开销）
        func(n, 0, data);
        return true;
    }
    return work_queue_run_async(q, func, data, n);
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '三种“并发单元”', b: 'work_queue（HVX 线程池）+ dma_queue（每线程一个）<br>+ hmx_queue（矩阵引擎，最多 1 个）。',
        m: 'ctx->work_queue / ctx->dma[i] / ctx->hmx_queue' },
      { c: 'c', t: '栅栏：seq + status', b: '一个 128 字节槽存 {seq, status}。<br>多设备时按 mdev_idx 各占一格。',
        m: 'htp_fence_write / htp_fence_read' },
      { c: 'd', t: '绕过 cache 才看得见', b: '写侧先 status 后 seq，再 syncht +<br>Q6_dccleaninva_A；读侧同序。',
        m: 'asm volatile ("syncht")' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个 batch 到了 DSP，先唤醒线程池：<span class="v">work_queue_wakeup()</span>。',
      '算子内部调 <span class="v">work_queue_run(q, func, data, n)</span> 把工作切成 n 份。' +
      '<span class="k">n &lt;= 1 时根本不起线程</span> —— 小算子不走并发的路。',
      '线程数不是随便定的：<span class="v">WORK_QUEUE_MAX_N_THREADS</span> 必须 ≥ ' +
      '<span class="v">HTP_MAX_NTHREADS</span>，main.c 里有 ' +
      '<span class="v">_Static_assert</span> 在编译期把关。',
      '一个 batch 跑完：<span class="v">hmx_queue_suspend()</span> + <span class="v">hmx_queue_flush()</span> ' +
      '+ <span class="v">work_queue_suspend()</span>，然后刷 cache、过栅栏、回写响应。',
      '栅栏本身很小：一个槽里两个 <span class="v">atomic_uint</span> —— ' +
      '<span class="k">先写 status，再写 seq</span>。读方以 seq 为准，' +
      '所以"seq 到位"就意味着"status 也已写好"。',
      '<span class="v">asm volatile ("syncht")</span> + <span class="v">Q6_dccleaninva_A</span>：' +
      '这两条是为了让另一端<b>真的看见</b>更新 —— DSP 的 cache 与 host 之间没有硬件一致性。',
      '<span class="k">跨核同步必须显式、必须绕 cache</span>：这是"两颗处理器"这个事实最硬的一处体现。' +
      '（对比 L6-06：Vulkan 的 fence 由驱动管，主机端只写一句 wait。）'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3500, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(11200, () => { msg.innerHTML = texts[3]; });
    tl.at(14400, () => { msg.innerHTML = texts[4]; });
    tl.at(17600, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 9 把边界画成一张表：<span class="hl-a">host 管什么</span>、<span class="hl-c">DSP 管什么</span> */
{
  kicker: "L7-02 · 收束",
  title: "把边界画成一张表：<span class=\"hl-a\">host 管什么</span>、<span class=\"hl-c\">DSP 管什么</span>",
  sub: "验收点：看完这一课，应该能不看代码就把这张表说出来。",
  caption: "下一课 L7-03：OpenVINO 后端 —— 另一种\"把子图整体交出去\"的做法，对比这里的\"逐算子下发\"。",
  src: "ggml/src/ggml-hexagon/htp/htp-fence.h",
  mark: [0, 1, 2, 5, 6, 10, 14, 16],
  lineNo: 17,
  code: `static inline void htp_fence_write(void * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
//>> 先 status 后 seq：读方只需等 seq，不必担心读到半更新的状态
    atomic_store(&fence[0], seq);
    asm volatile ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) fence);
//>> Q6_dccleaninva_A + syncht：清 cache 行再同步线程，保证对端可见
}

static inline void htp_fence_read(const void * fence_ptr, uint32_t * seq, uint32_t * status) {
    const atomic_uint * fence = (const atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((void *) fence);
    asm volatile ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
//>> 读侧也先清 cache 再取 seq / status —— 顺序与写侧对称
    *status = atomic_load(&fence[1]);
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['侧', '负责什么', '源码依据'],
      [['host', '后端注册与驱动加载；驱动缺失则后端在运行期不存在', 'ggml-hexagon.cpp:8045 ggml_backend_hexagon_reg()'],
       ['host', '建远端会话：remote_handle64 + dspqueue_create / export', 'ggml-hexagon.cpp:3964 dspqueue_create()'],
       ['host', '共享内存三步：rpcmem_alloc2 → rpcmem_to_fd → fastrpc_mmap', 'ggml-hexagon.cpp:586 / :627'],
       ['host', '把 ggml 图翻成 htp_opnode，预计算 kernel_params 并做融合', 'htp-opnode.h:23 / ggml-hexagon.cpp:6400'],
       ['host', '攒批、打包三个描述符数组、dspqueue_write', 'ggml-hexagon.cpp:3234 / :3434'],
       ['DSP', '实现 htp_iface_*：open / start / mmap / munmap / profiler', 'htp/main.c:123 / :229 / :366'],
       ['DSP', '导入队列、探测 HVX 线程数、拿 VTCM、建线程池与 DMA 队列', 'htp/main.c:387 / :402 / :594'],
       ['DSP', '解包 batch：复用 mmap、刷 cache、把块内偏移换成 DSP 虚拟地址', 'htp/main.c:1198 / :1201'],
       ['DSP', 'execute_op 分发到算子内核；多线程切分；写栅栏；回写响应', 'htp/main.c:810 / work-queue.h:25 / htp-fence.h:17']],
      { monoCols: [0, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码回答：Hexagon 后端为什么要自己实现 <span class="mono">rpcmem</span> + ' +
      '<span class="mono">dspqueue</span> + <span class="mono">fence</span> 这三样，' +
      '而 CUDA / Vulkan 后端不用？',
      '因为算力在<b>另一颗处理器</b>上，两颗核不共享地址空间，也不共享 cache。' +
      '<span class="mono">rpcmem</span> + <span class="mono">fastrpc_mmap</span> 解决"同一块内存、两个虚拟地址"；' +
      '<span class="mono">dspqueue</span> 解决"两边都有句柄的一条命令队列"；' +
      '<span class="mono">fence</span> 解决"没有硬件 cache 一致性，同步必须显式且绕 cache"。<br>' +
      'GPU 后端把这三件事都交给了驱动和统一虚拟地址，所以它的接口看上去只是"提交 kernel"。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '上半张表是 host：<span class="k">注册、建会话、分内存、翻译图、发命令</span>。',
      '下半张表是 DSP：<span class="k">接命令、摆资源、解包、分发算子、算、同步、回话</span>。',
      '一句话划分：<b>host 做"决定"，DSP 做"执行"</b> —— ' +
      'host 上一个像素都不算，DSP 上一次内存分配都不做。',
      '两个反直觉的地方：<br>① <span class="v">kernel_params</span> 是 host 算的（能提前算的都提前算）；' +
      '② <span class="v">权重 repack 成 tiled 布局</span> 也是 host 做的（tensor 一 set 就转）。',
      '对照 L7-01（CANN）：那边 host 调 `aclnn*`、内存走 `aclrtMalloc`，控制面由厂商运行时给；' +
      '这边控制面（IDL + dspqueue + fastrpc 映射）全部自己搭。<br>' +
      '对照 L6-06（Vulkan）：那边也有队列与内存类型，但都在同一颗芯片上，驱动替你兜底。',
      '验收：合上代码，说出上面两行分工，并各举一条源码依据 —— 这一课就算过了。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => { msg.innerHTML = texts[1]; });
    rows.forEach((r, i) => tl.at(6400 + i * 1400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
    }));
    tl.at(19200, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[2];
    });
    tl.at(20800, () => { msg.innerHTML = texts[3]; });
    tl.at(22300, () => { msg.innerHTML = texts[4]; });
    tl.at(23300, () => { msg.innerHTML = texts[5]; });
  }
},

];
