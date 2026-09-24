/* ==========================================================================
   L7-06 · 小后端合集：OpenCL / WebGPU / MUSA / RPC / BLAS / ZenDNN / zDNN
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 七个后端，<span class="hl-a">一个契约</span> */
{
  kicker: "L7 · NPU 与加速器后端",
  title: "七个后端，<span class=\"hl-a\">一个契约</span>",
  sub: "回顾 L3-01：ggml 不认识任何硬件，只认识 ggml_backend_i 这一堆函数指针。",
  caption: "本课 9 幕：先量一遍规模，再看 BLAS / RPC / WebGPU / OpenCL，最后把三个\"厂商库转发\"型放在一起对比。",
  src: "ggml/include/ggml-zdnn.h",
  mark: [1, 3],
  lineNo: 10,
  code: `// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_zdnn_reg(void);
//>> zDNN 的公共头一共 2 个函数：要一块设备 buffer、要一个注册表项。契约的其余部分全在内部虚表里

#ifdef __cplusplus
}
#endif`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center;flex-wrap:wrap;gap:5px">
        <span class="chip a">ggml_backend_i</span><span class="arrow">-></span>
        <span class="chip">BLAS</span><span class="chip">RPC</span><span class="chip">WebGPU</span>
        <span class="chip">OpenCL</span><span class="chip">MUSA</span><span class="chip">ZenDNN</span>
        <span class="chip">zDNN</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '同一个契约', b: '七个后端都只填 L3-01 讲过的几张虚表：<br>backend_i / buffer_type_i / device_i / reg_i。',
        m: 'ggml_backend_i' },
      { c: 'c', t: '数据面差别极大', b: '数据可能在宿主内存、设备显存、<br>硬件专用布局，或在另一台机器上。',
        m: 'buffer_type_i' },
      { c: 'b', t: '实测规模：29 文件 / 45549 行', b: '最小的是 MUSA（2 文件 124 行），<br>最大的是 OpenCL（6 文件 30227 行）。',
        m: 'L7-06 coverage' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '七个后端，一个契约。<span class="k">谁把虚表填完，谁就是一个后端</span> —— 这是 L3-01 的结论。',
      '它们的公共头小得惊人：zDNN 只有 <span class="v">buffer_type()</span> 与 <span class="v">reg()</span> 两个函数（本幕引的第 10-17 行）。',
      '规模却差 240 倍：<span class="v">MUSA 124 行</span> vs <span class="v">OpenCL 30227 行</span>。差别不在契约，在<span class="k">各自要对付的数据面</span>。',
      '这七个后端和 <span class="k">L7-01~L7-05</span>（CANN / Hexagon / OpenVINO / ExecuTorch / VirtGPU）一起，构成 L7 的 312 个文件。'
    ];
    defs.forEach((_, i) => tl.at(600 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 2 一张表量完七个后端 */
{
  kicker: "L7-06 · 实测",
  title: "一张表量完七个后端",
  sub: "行数、文件数都来自 wc -l 实测；\"要写什么\"一句话来自逐文件阅读。",
  caption: "RPC 的 6 个文件里有 4 个是传输层（transport.h/.cpp + Apple RDMA 的 .h/.cpp）；OpenCL 的 6 个文件里含磁盘缓存与调参表。",
  src: "ggml/include/ggml-rpc.h",
  mark: [1, 4, 8, 12, 13],
  lineNo: 19,
  code: `// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
//>> 同一个库里既有客户端（init/buffer_type），也有服务端（start_server）—— 一眼看不出谁是"远端"
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);
//>> add_server：把一台 RPC 服务器变成一个可注册的后端，于是调度器可以像用 GPU 一样用它（见 L4-02）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['BLAS',    '2 / 555',   '只写一个 graph_compute 的 switch：MUL_MAT、OUT_PROD 转发给 cblas_sgemm'],
      ['MUSA',    '2 / 124',   '把 ggml_tensor 的 dims/strides/type 翻成 mudnn::Tensor；算子实现复用 ggml-cuda 源码'],
      ['ZenDNN',  '2 / 860',   '把 MUL_MAT / MUL_MAT_ID 翻译成 zendnnl 的 matmul_direct / group_matmul_direct'],
      ['zDNN',    '7 / 904',   '每个 tensor 建一个硬件 ztensor（变换布局）；只支持 MUL_MAT'],
      ['RPC',     '6 / 3694',  '把 tensor 与整张图序列化到 socket，远端执行；本机只把它当"设备"'],
      ['WebGPU',  '4 / 9185',  'WGSL 内核（3471 行的 shader 库）+ wgpu 缓冲；在浏览器里跑'],
      ['OpenCL',  '6 / 30227', '运行期编译 .cl（181 处 read_file）+ 磁盘二进制缓存 + Adreno 调参表'],
      [{ html: '<b>合计</b>' }, { html: '<b>29 / 45549</b>' },
       { html: '<b>同一个 ggml_backend_i，七张完全不同的数据面</b>' }]
    ];
    const t = U.table(['后端', '本课文件 / 行数', '最小可行实现要写什么（实测）'], rows, { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const trs = t.body.querySelectorAll('tr');
    const texts = [
      '先量规模。读法：<span class="k">文件数来自本课的覆盖声明，行数来自 wc -l</span>。',
      '<span class="v">BLAS</span> 只有 1 个 .cpp：它不实现任何算子，只是把两个 op 转给厂商库。',
      '<span class="v">MUSA</span> 只有 2 个文件，因为 <span class="k">整个后端复用 ggml-cuda 的源码</span>（构建脚本把 ../ggml-cuda/*.cu 一起编）。',
      '<span class="v">ZenDNN / zDNN</span> 各自只翻译矩阵乘：一个转成 zendnnl 的 matmul，一个转成 zDNN 的 ztensor 运算。',
      '<span class="v">RPC</span> 的 6 个文件里，3 个是传输层 —— 它要处理的不是算子，而是<span class="k">网络</span>。',
      '<span class="v">WebGPU / OpenCL</span> 最重：内核不是 C，是 WGSL / OpenCL C 源码，得在运行期编译。',
      '合计 <span class="v">29 个文件 / 45549 行</span>。记住这个数字：本课只需要它们证明一件事 —— 契约不变，数据面随便换。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    trs.forEach((r, i) => tl.at(3000 + i * 2100, () => {
      trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 6)];
    }));
    tl.at(15600, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 3 BLAS：整个后端就是<span class="hl-b">一个 switch</span> */
{
  kicker: "L7-06 · BLAS",
  title: "BLAS：整个后端就是<span class=\"hl-b\">一个 switch</span>",
  sub: "不新增数据面、不新增 buffer type —— 只是把两个 op 换成厂商 gemm。",
  caption: "数据面这一层的证据在 ggml-blas.cpp:381-382（device_get_buffer_type 直接返回 ggml_backend_cpu_buffer_type），第 9 幕会再引一次。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [0, 10, 11, 16, 27, 29],
  lineNo: 226,
  code: `static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
//>> 唯一被"加速"的 op：转给 ggml_backend_blas_mul_mat()，内部就是 cblas_sgemm（第 142 行）
                ggml_backend_blas_mul_mat(ctx, node);
                break;

            case GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
//>> default 直接 abort：送进来一个不支持的 op 就死。所以 supports_op 必须与这个 switch 严格一致
                GGML_ABORT("%s: unsupported op %s\\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '数据面：一行代码', b: 'device_get_buffer_type() 直接 <b>返回 CPU 的 buffer type</b>（第 381-382 行）：' +
          'BLAS 后端没有自己的内存，张量还躺在宿主内存里。', m: 'ggml_backend_cpu_buffer_type()' },
      { c: 'b', t: '只认两个 op', b: 'MUL_MAT 与 OUT_PROD 转给 cblas_sgemm；<br>其余 op 靠调度器分给别人（见 L4-02）。',
        m: 'GGML_OP_MUL_MAT' },
      { c: 'd', t: 'default 是 ABORT', b: 'supports_op 里还有一道门槛：只有 <b>大矩阵</b>才接（min_batch = 32，第 405-431 行）。',
        m: 'GGML_ABORT' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '第一个后端：<span class="k">BLAS</span>。它证明了"最小可行后端"能有多小。',
      '它不分配内存、不搬数据、不写内核 —— <span class="v">buffer type 直接复用 CPU 的</span>。',
      '<span class="v">MUL_MAT</span> 换成 <span class="v">cblas_sgemm</span>（第 142 行），别的 op 它根本不该收到。',
      '<span class="v">default: GGML_ABORT</span> 是契约的执行者：<span class="k">supports_op 说"不支持"，调度器就不会送过来</span>。',
      '所以一个后端的"能力声明"（supports_op）和"实现"（graph_compute）必须是一对 —— 这条在 L3-02 的设备发现里也要用到。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-a">rpc_tensor</span>：一个 tensor 在网线上的样子 */
{
  kicker: "L7-06 · RPC · 核心",
  title: "★ <span class=\"hl-a\">rpc_tensor</span>：一个 tensor 在网线上的样子",
  sub: "ggml_tensor 里有指针，不能直接过网；于是有一个逐字段对应的 POD 版本。",
  caption: "验收点就在这一幕和下一幕：tensor 先变成 rpc_tensor，再和裸数据拼成一块内存发给远端。",
  src: "ggml/src/ggml-rpc/ggml-rpc.cpp",
  mark: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 22],
  lineNo: 39,
  code: `// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
//>> buffer 不是本地指针：它是远端分配时记下的 remote_ptr（serialize_tensor，第 642 行）
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
//>> data 是【本机指针值】本身（第 643 行）—— 服务端把它当远端地址直接用（deserialize_tensor，第 1415 行）
    char name[GGML_MAX_NAME];

    int32_t use_count;
//>> use_count 来自图上的引用计数，服务端重建图时要用（第 1003-1009 行）
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");
//>> #pragma pack(push, 1) 下的定长结构：两端对同一份字节布局的约定，就是 RPC 协议`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['id / use_count', '本机 tensor 指针 + 图上引用计数', '图里节点的身份：服务端按 id 建节点、按 use_count 记引用'],
      ['type / ne / nb', '类型与形状、步长，逐个复制', '重建一个同形状同步长的 ggml_tensor（第 1384-1395 行）'],
      ['op / op_params / flags', '运算与标量参数', '服务端算出同一个算子（第 1410-1414 行）'],
      ['src / view_src / view_offs', '输入与视图关系，同样是指针值', '把图上的边在远端重新接起来'],
      ['buffer', '远端 buffer 句柄 remote_ptr', '查 buffers 表还原成远端的 ggml_backend_buffer_t'],
      ['data', '本机指针值（原样发过去）', '直接当远端地址用：必须落在远端 buffer 区间内（第 1401-1407 行断言）']
    ];
    const t = U.table(['rpc_tensor 字段', '网线上是什么', '服务端拿它做什么'], rows, { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const trs = t.body.querySelectorAll('tr');
    const texts = [
      '问题：<span class="v">ggml_tensor</span> 里有 <span class="v">data</span> / <span class="v">src[]</span> / <span class="v">buffer</span> 这些<span class="k">指针</span>，跨机器发过去毫无意义。',
      'RPC 的答案：定义一个<span class="k">定长、无指针歧义</span>的镜像结构 <span class="v">rpc_tensor</span>（第 40-56 行），逐字段填。',
      '形状、步长、op、参数、视图关系都是<span class="k">复制</span>：服务端凭它们重建一个结构等价的 tensor。',
      '<span class="v">buffer</span> 存的是远端分配时记下的 <span class="v">remote_ptr</span>（第 642 行）；服务端查表还原成自己的 buffer 句柄。',
      '最妙的是 <span class="v">data</span>：发过去的就是<span class="k">本机指针的数值</span>（第 643 行），服务端把它当远端地址直接用（第 1415 行）。',
      '前提是远端 buffer 的基址在两侧都被记下来了 —— 这就是 <span class="v">RPC_CMD_BUFFER_GET_BASE</span> 的作用（第 611-622 行）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    trs.forEach((r, i) => tl.at(3300 + i * 2500, () => {
      trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(17000, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 把 tensor 送过去：<span class="hl-c">打包</span> + 一条 socket */
{
  kicker: "L7-06 · RPC · 全链路",
  title: "把 tensor 送过去：<span class=\"hl-c\">打包</span> + 一条 socket",
  sub: "元数据 + 一个标志位 + 偏移 + 裸数据 = 一次 SET_TENSOR。整张图也走同一条路。",
  caption: "这一步是验收点的答案：客户端 serialize_set_tensor → dispatcher → socket → 服务端 recv_msg → deserialize → 写入远端 buffer。",
  src: "ggml/src/ggml-rpc/ggml-rpc.cpp",
  mark: [0, 2, 6, 7, 8, 9, 10, 20, 22, 40, 41],
  lineNo: 701,
  code: `static std::shared_ptr<uint8_t> serialize_set_tensor(const rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, const void * data, size_t size, size_t & input_size) {
//>> 一行注释写死了协议布局：| rpc_tensor | cache_flag(1B) | offset(8B) | data(size B) |
    input_size = sizeof(rpc_tensor) + sizeof(cache_flag) + sizeof(offset) + size;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * p = input;
    memcpy(p, &rpc_tensor, sizeof(rpc_tensor)); p += sizeof(rpc_tensor);
    memcpy(p, &cache_flag, sizeof(cache_flag)); p += sizeof(cache_flag);
    memcpy(p, &offset,     sizeof(offset));     p += sizeof(offset);
    memcpy(p, data, size);
    return std::shared_ptr<uint8_t>(input, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with \`rpc-server -c\` every ubatch above the threshold is written
// to the cache directory and later served from there.
static bool rpc_use_hash_cache(const ggml_tensor * tensor, size_t size) {
    return size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    if (rpc_use_hash_cache(tensor, size)) {
//>> 权重走哈希缓存：先把 FNV 哈希发给远端问"你有没有"（阈值见第 716-717 行，服务端分支见第 1987 行），命中就连数据都不用发
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t input_size;
    auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input, input_size);
//>> dispatcher->send() 把这块内存交给 socket；命令字 RPC_CMD_SET_TENSOR 是枚举里的第 68 行
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center;flex-wrap:wrap;gap:4px">
        <span class="chip a">serialize_tensor</span><span class="arrow">-></span>
        <span class="chip b">serialize_set_tensor</span><span class="arrow">-></span>
        <span class="chip c">socket</span><span class="arrow">-></span>
        <span class="chip d">recv_msg</span><span class="arrow">-></span>
        <span class="chip e">deserialize_tensor</span><span class="arrow">-></span>
        <span class="chip f">ggml_backend_tensor_set</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: '服务端只做三件事', b: '收下整块内存（第 1977-1985 行 case SET_TENSOR）→ ' +
          '按元数据重建 tensor（第 1442 行）→ 把裸数据写进远端 buffer（第 1472 行）。', m: 'ggml_backend_tensor_set' },
      { c: 'c', t: '整张图也走同一条 socket', b: 'serialize_graph 把节点与全部 tensor 拼成一块内存（第 1003-1028 行）；' +
          '服务端重建图后调用远端的 ggml_backend_graph_compute（第 1772 行）。', m: 'RPC_CMD_GRAPH_COMPUTE' },
      { c: 'e', t: '底下可以是 TCP，也可以是 RDMA', b: 'socket_t 只有 send_data / recv_data / flush（transport.h:16-21）；' +
          'Apple 上还有一条 Thunderbolt RDMA 通道（transport-apple.cpp:16-27）。', m: 'socket_t' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '打包格式只有三种东西：<span class="k">元数据</span>（定长）、<span class="k">标志位与偏移</span>、<span class="k">裸数据</span>。',
      '第 701 行的注释就是协议本身：<span class="v">| rpc_tensor | cache_flag | offset | data |</span>，客户端拼、服务端拆。',
      '发之前还有一步优化：<span class="v">权重</span>先发 FNV 哈希问远端有没有（第 724-737 行）—— 模型重载可以完全不传数据。',
      '服务端不需要理解 ggml 语义，它只是<span class="k">照着元数据把指针和偏移还原</span>，然后 memcpy 进远端设备内存。',
      '计算也一样：<span class="v">serialize_graph</span> 把整张图的节点与 tensor 序列化（第 1003-1028 行），远端重建并执行（第 1772 行）。',
      '<span class="k">参数与激活留在远端，本机只拿到一个"算完了"的状态</span>。所以 RPC 设备在调度器眼里就是一块普通的 GPU。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(16200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 WebGPU：内核是 <span class="hl-e">WGSL 源码字符串</span> */
{
  kicker: "L7-06 · WebGPU",
  title: "WebGPU：内核是 <span class=\"hl-e\">WGSL 源码字符串</span>",
  sub: "同一张虚表，但一半的槽是 NULL，且默认异步 —— 因为它跑在浏览器里。",
  caption: "回顾 L3-01：ggml_backend_i 里只有 get_name / free / graph_compute 是必需的，WebGPU 正好把这几个填满。",
  src: "ggml/src/ggml-webgpu/ggml-webgpu.cpp",
  mark: [0, 4, 5, 10, 14, 15, 21],
  lineNo: 333,
  code: `static webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   const char *                             shader_code,
                                                   const char *                             label,
                                                   const std::vector<wgpu::ConstantEntry> & constants = {}) {
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.nextInChain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = nullptr;  // nullptr means auto layout
    if (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.data();
        pipeline_desc.compute.constantCount = constants.size();
    }
    return { device.CreateComputePipeline(&pipeline_desc), label };
}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '内核不是 C', b: 'WGSL 源码字符串 → CreateShaderModule → CreateComputePipeline，入口固定叫 main。',
        m: 'wgpu::ShaderSourceWGSL' },
      { c: 'c', t: '内核住在 3471 行的库里', b: 'ggml-webgpu-shader-lib.hpp 把 WGSL 编译成 pipeline 并按设备能力挑变体；' +
          'WGSL 文本来自构建期生成的 ggml-wgsl-shaders.hpp（见脚注）。', m: 'ggml_webgpu_shader_lib' },
      { c: 'e', t: '数据在 wgpu::Buffer 里', b: 'buffer_set_tensor 走 queue.WriteBuffer（第 3777 行）；' +
          '读回要 CopyBufferToBuffer 到 staging buffer 再 map（第 3825-3851 行，map 辅助函数在第 536 行）。', m: 'queue.WriteBuffer' },
      { c: 'd', t: '默认异步', b: '虚表填的是 set_tensor_async / synchronize / event_record / event_wait（第 3700-3716 行）—— ' +
          'GPU 提交不阻塞 CPU。', m: 'ggml_backend_webgpu_i' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="k">WebGPU</span> 是唯一一个把后端编译进浏览器的：整份源码里有 <span class="v">__EMSCRIPTEN__</span> 分支（第 13-15 行）。',
      '它的"内核"是 <span class="v">WGSL 文本</span>：CreateShaderModule 吃源码字符串，ComputePipeline 的入口写死 <span class="v">main</span>。',
      '3471 行的 shader 库 + 808 行的 <span class="v">pre_wgsl</span> 预处理器：因为不同适配器的 limits 不同，同一个算子要挑不同变体。',
      '数据面是 <span class="v">wgpu::Buffer</span>：写入用 WriteBuffer，读回要 map staging buffer —— 不能像 CUDA 那样随意 memcpy。',
      '它把虚表里的异步槽位真的用上了：<span class="k">提交、同步、事件</span>四个函数都有实现，另半边则是 NULL。',
      '把这一课和前两幕对照：<span class="v">同一张表</span>，一边是网络，一边是浏览器 —— 这正是 L3-01 抽象成功的证据。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
    tl.at(15000, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 OpenCL：<span class="hl-d">运行期</span>加载、编译、缓存 */
{
  kicker: "L7-06 · OpenCL",
  title: "OpenCL：<span class=\"hl-d\">运行期</span>加载、编译、缓存",
  sub: "29502 行里绝大部分是内核与派发；真正\"后端骨架\"的部分只有几千行。",
  caption: "对照 L5-05：CPU 后端的内核是编译进二进制的 C；OpenCL 的内核是运行期才存在的字符串。",
  src: "ggml/src/ggml-opencl/ggml-opencl.cpp",
  mark: [0, 6, 8, 12, 16],
  lineNo: 1382,
  code: `static cl_program build_program_from_source(ggml_backend_opencl_context * backend_ctx, const char* program_buffer, const std::string &compile_opts) {
    cl_context   ctx = backend_ctx->context;
    cl_device_id dev = backend_ctx->device;

    // Try the on-disk binary cache first. Falls through silently on miss or
    // any failure; never blocks the build path. Disabled cache => nullptr.
    cl_program p_cached = cl_program_cache_try_load(
        backend_ctx->program_cache, ctx, dev, program_buffer, compile_opts);
    if (p_cached != nullptr) {
        return p_cached;
    }

    cl_program p = build_program_from_source_ex(ctx, dev, program_buffer, compile_opts, /*fatal=*/true);

    // Best-effort save of the freshly-built binary (no-op if cache disabled).
    if (p != nullptr) {
        cl_program_cache_try_save(backend_ctx->program_cache, p, dev, program_buffer, compile_opts);
    }
    return p;
}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: '预编译内核可动态加载', b: 'libdl.h 提供 dlopen / dlsym（第 66、71 行）：运行期尝试加载 Adreno 预编译内核库，' +
          '失败就回退到内置内核源码（ggml-opencl.cpp:6726-6744）。', m: 'dl_load_library()' },
      { c: 'a', t: '内核是字符串', b: '181 处 read_file("xxx.cl") 调用点把内核读进来编译（未嵌入内核时）；' +
          '编译选项带 -cl-mad-enable 等（第 1433-1435 行）。', m: 'build_program_from_source()' },
      { c: 'c', t: '磁盘二进制缓存', b: '缓存键 = SHA-256(源码 + 编译选项 + 设备/驱动/平台)（第 155-171 行）；' +
          '命中就跳过 clBuildProgram。', m: 'cl_program_cache_try_load()' },
      { c: 'd', t: '连调参表都外置了', b: 'fa_tune.h：每个 (dk,dv) 一组 bm/bn/n_split（第 16-24 行），' +
          '还能用环境变量覆盖（第 48 行）。', m: 'g_fa_dims_adreno_default[]' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="k">OpenCL</span> 是本课最大的后端：单文件 29502 行 —— 因为内核源码和派发逻辑都在里面。',
      '先看它的"后门"：<span class="v">libdl.h</span> 在运行期尝试 dlopen 厂商的 Adreno 预编译内核库 —— 成功就用二进制内核，失败就回退到内置源码。',
      '内核在<span class="k">运行期编译</span>：读 .cl 源码（或嵌入的内核头）→ clBuildProgram → clCreateKernel，共 484 处。',
      '编译很贵，所以有<span class="v">磁盘二进制缓存</span>：本幕引的 20 行就是缓存的 try_load / try_save 夹着编译。',
      '连 FA 的分块调参都抽成了数据表（<span class="v">fa_tune.h</span>）—— 不同 Adreno 代次一套参数，可现场覆盖。',
      '对比 BLAS：那边"最小可行后端"是 530 行，这边是 30227 行。<span class="k">契约一样，工作量差 50 倍</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
    tl.at(15000, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 ★ MUSA / ZenDNN / zDNN：<span class="hl-a">翻译深度</span>决定数据面 */
{
  kicker: "L7-06 · 厂商库转发",
  title: "★ MUSA / ZenDNN / zDNN：<span class=\"hl-a\">翻译深度</span>决定数据面",
  sub: "三个后端都在\"把 ggml 的布局翻译成厂商库要的布局\"，但翻译到哪一层，差别巨大。",
  caption: "对照 L7-01~L7-05：CANN / OpenVINO 等后端也在做同一件事，只是翻译的目标库不同。",
  src: "ggml/src/ggml-zdnn/utils.cpp",
  mark: [0, 5, 6, 13, 14, 18, 19],
  lineNo: 25,
  code: `void ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       const ggml_tensor       * src,
                       const int64_t           * ne,
                       const zdnn_data_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
//>> ztensor 的存储由 zDNN 自己 malloc：这块数据【不在】ggml 的 CPU buffer 里
}

void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, void * buffer) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, buffer));
//>> 写入时把 ggml 布局变换成硬件布局 —— 数据面就是在这里被改写的
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: 'MUSA：只翻译 dims/strides', b: '2 文件 124 行。把 ggml_tensor 的维度、步长、类型搬进 mudnn::Tensor' +
          '（mudnn.cu:53-66）；连 memcpy 都用 mudnn 的一元算子 IDENTITY（第 103 行）。', m: 'mudnn::Unary::Mode::IDENTITY' },
      { c: 'd', t: 'ZenDNN：只翻译矩阵乘', b: '2 文件 860 行。MUL_MAT / MUL_MAT_ID 直接投给 zendnnl 的 matmul_direct' +
          '（第 81-95 行），bias、alpha/beta、量化 scale 都写进参数。', m: 'zendnnl::lowoha::matmul' },
      { c: 'a', t: 'zDNN：连内存布局都翻译', b: '7 文件 904 行。每个 tensor 一个 zdnn_ztensor（common.hpp:43-45）；' +
          '写入时做硬件变换（本幕第 42-43 行）；buffer type 已经不是 host（ggml-zdnn.cpp:384-393）。', m: 'zdnn_transform_ztensor()' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '最后三个后端放在一起看，因为它们在干<span class="k">同一件事</span>：把 ggml 的布局翻译成厂商库要的布局。',
      '<span class="v">MUSA</span> 翻得最浅：只把 dims/strides/type 搬过去，算子实现直接复用 ggml-cuda 的源码。',
      '<span class="v">ZenDNN</span> 翻到"算子参数"这一层：行主序、是否转置、量化 scale 全部摊开写成 matmul_params。',
      '<span class="v">zDNN</span> 翻得最深：它给每个 tensor 造一个 <span class="v">zdnn_ztensor</span>，连内存布局都要变换成硬件要的样子。',
      '证据就是本幕的代码：<span class="v">zdnn_init_ztensor_with_malloc</span> 自己分配存储，<span class="v">zdnn_transform_ztensor</span> 把数据搬过去。',
      '三者的 buffer type 也据此不同：zDNN 明确回答 <span class="v">is_host = false</span>（ggml-zdnn.cpp:376-378），而 ZenDNN 直接复用 CPU 的。',
      '这就是本课的核心洞察：<span class="k">同一张虚表，能容纳从"零改动"到"改写内存布局"的全部实现</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L7-06 · 收束",
  title: "把这一课压成一张表",
  sub: "四种数据面，一套契约。下一课 L8-01 让一个 mul_mat 从模型一路走到内核。",
  caption: "末幕代码只引两行：BLAS 的 buffer type 就是 CPU 的 —— 数据面\"零改动\"的极端例子。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [0, 1],
  lineNo: 381,
  code: `static ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['数据面', '代表后端', 'tensor 的数据实际在哪', '谁负责搬'],
      [['宿主内存（零改动）', 'BLAS / ZenDNN', '就在 CPU 内存里，buffer type 直接复用 CPU 的', '没人搬 —— 只换了一个 gemm 实现'],
       ['设备/加速器内存', 'WebGPU / OpenCL', '驱动管理的设备缓冲（WriteBuffer / enqueueWriteBuffer）', '后端的 buffer_set_tensor'],
       ['硬件专用布局', 'zDNN', '变换后的 ztensor，由 zDNN 自己 malloc', 'zdnn_transform_ztensor()'],
       ['远端机器', 'RPC', '另一台机器上的 buffer，本机只有一个句柄', 'socket 上的 rpc_tensor + 裸 payload']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      'RPC 后端把一个 tensor 送到远端，中间经过了哪几步？',
      '① 客户端把 tensor 的元数据填进定长的 <span class="mono">rpc_tensor</span>' +
      '（<span class="mono">serialize_tensor</span>，ggml-rpc.cpp:628）；其中 <span class="mono">buffer</span> 是远端句柄 ' +
      '<span class="mono">remote_ptr</span>（:642），<span class="mono">data</span> 是本机指针值（:643）。<br>' +
      '② 拼成一块连续内存：<span class="mono">| rpc_tensor | cache_flag | offset | data |</span>' +
      '（<span class="mono">serialize_set_tensor</span>，:701-715）。<br>' +
      '③ 走 <span class="mono">dispatcher-&gt;send(RPC_CMD_SET_TENSOR, ...)</span>（:740）交给 ' +
      '<span class="mono">socket_t::send_data</span>（transport.cpp:603）。权重还会先用 FNV 哈希问远端有没有（:724-737）。<br>' +
      '④ 服务端 <span class="mono">recv_msg</span> 收下整块（:1977-1985）→ ' +
      '<span class="mono">deserialize_tensor</span> 还原指针（:1442）→ ' +
      '<span class="mono">ggml_backend_tensor_set</span> 写进远端 buffer（:1472）。<br>' +
      '计算同理：<span class="mono">serialize_graph</span>（:1003）把整张图发过去，远端重建后调用自己的 ' +
      '<span class="mono">ggml_backend_graph_compute</span>（:1772）。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '四种数据面，一套契约：',
      '<span class="k">零改动</span>：BLAS / ZenDNN 连 buffer type 都不换 —— 数据一直在宿主内存。',
      '<span class="k">设备内存</span>：WebGPU / OpenCL 的数据在驱动缓冲里，写入要过 WriteBuffer / enqueueWriteBuffer。',
      '<span class="k">硬件布局</span>：zDNN 把数据变换成 ztensor —— 唯一一个连内存表示都改写的后端。',
      '<span class="k">远端机器</span>：RPC 把"设备"这个概念推到了网络另一头，图与 tensor 都得序列化。',
      '所以 L3-01 的三张表是成功的抽象：<span class="v">后端之间差异有多大，契约就有多稳定</span>。',
      '下一课 <span class="v">L8-01 ★ 端到端</span>：跟着一个 mul_mat 走完全程，把 L1~L7 串成一条链。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 2900, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(15500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
    tl.at(17800, () => { msg.innerHTML = texts[6]; });
  }
},

];
