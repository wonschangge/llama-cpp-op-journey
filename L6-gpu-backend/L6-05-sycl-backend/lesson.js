/* ==========================================================================
   L6-05 · SYCL 后端（Intel GPU）
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 SYCL 后端：CUDA 的<span class="hl-a">平行实现</span> */
{
  kicker: "L6 · GPU 后端执行",
  title: "SYCL 后端：CUDA 的<span class=\"hl-a\">平行实现</span>",
  sub: "Intel GPU 走 SYCL；它没有另立一套算子体系，而是把 CUDA 侧那一套照搬过来。",
  caption: "回顾 L3-01：每个后端都要交出 registry / device / buffer type 三张接口。本课的对照物是 L6-01 ~ L6-04（CUDA 四条线）。",
  src: "ggml/include/ggml-sycl.h",
  mark: [1, 4, 7, 10],
  lineNo: 19,
  code: `// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_sycl_init(int device);

GGML_BACKEND_API bool ggml_backend_is_sycl(ggml_backend_t backend);

// devide buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(int main_device, const float * tensor_split);

// Tensor parallelism (--split-mode tensor): comm_init/free/allreduce_tensor
// trio queried by the meta-backend via ggml_backend_reg_get_proc_address.`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">ggml 计算图 L1-03</span><span class="arrow">-></span>
        <span class="chip a">调度器 L4-02</span><span class="arrow">-></span>
        <span class="chip b">ggml-sycl 后端</span><span class="arrow">-></span>
        <span class="chip c">Intel GPU（oneAPI / DPC++）</span>
      </div>
      <div class="row center" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '公共面：1 个头文件',
        b: 'init / is_sycl / buffer_type / split_buffer_type /<br>comm_* 三件套',
        m: 'ggml/include/ggml-sycl.h' },
      { c: 'b', t: '实现面：173 个文件',
        b: '主机框架、量化矩阵乘、FlashAttention、<br>算子内核、模板实例化',
        m: 'ggml/src/ggml-sycl/' },
      { c: 'd', t: '对照物：CUDA 后端',
        b: 'L6-01 骨架 / L6-02 量化矩阵乘 /<br>L6-03 FlashAttention / L6-04 其余算子',
        m: 'ggml/src/ggml-cuda/' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:224px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个算子（L1-02 的 <span class="v">GGML_OP_MUL_MAT</span>）从模型出发，经 L4-02 的调度器'
        + '切给某个后端，最后由后端自己的 kernel 落地。本课看 SYCL 这一格。',
      '<span class="k">公共面很薄</span>：SYCL 后端对外只暴露一个头文件里的十几个 C 函数。'
        + '后端要交出的三张接口（L3-01）实现都在 <span class="v">ggml-sycl.cpp</span> 里。',
      '<span class="k">实现面很厚</span>：173 个文件，按算子族切开，多数是一个算子一个 <span class="v">.cpp + .hpp</span> 对。'
        + '这个切法与 CUDA 侧基本一致（少数公共头如 quants.hpp / vecdotq.hpp 被多个算子共用）。',
      '为什么值得单独一课：<span class="k">它证明 ggml 的后端契约是可移植的</span> —— '
        + '换一套并行运行时（SYCL 替换 CUDA），算子层几乎不用重新设计。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14300, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[3];
    });
  }
},

/* ------------------------------------------------------ 2 174 个文件，<span class="hl-b">七族</span> */
{
  kicker: "L6-05 · 目录解剖",
  title: "174 个文件，<span class=\"hl-b\">七族</span>",
  sub: "家族的划分几乎照抄 CUDA：一算子一对文件、模板实例单独成目录、主机框架独立。",
  caption: "CMakeLists.txt 用通配收集 *.hpp / *.cpp，再单独 glob 两个 template-instances 前缀。该文件不是本视角的源文件后缀，因此被引用但不计入覆盖率。",
  src: "ggml/src/ggml-sycl/CMakeLists.txt",
  mark: [3, 4, 5, 7],
  lineNo: 22,
  code: `                         ggml-sycl.cpp
                         ../../include/ggml-sycl.h
                        )

file(GLOB   GGML_HEADERS_SYCL "*.hpp")
file(GLOB   GGML_SOURCES_SYCL "*.cpp")
file(GLOB   SRCS "template-instances/fattn-tile*.cpp")
list(APPEND GGML_SOURCES_SYCL \${SRCS})
file(GLOB   SRCS "template-instances/fattn-vec*.cpp")
list(APPEND GGML_SOURCES_SYCL \${SRCS})

target_sources(ggml-sycl PRIVATE \${GGML_HEADERS_SYCL} \${GGML_SOURCES_SYCL})`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="formula" style="padding:5px 8px">`
      + `174 个文件 = 1 个公共头（ggml/include/ggml-sycl.h）+ 173 个实现（ggml/src/ggml-sycl/）</div>
      <div id="bars"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const groups = [
      { l: '常规算子内核', n: 73, c: 'a', d: 'element_wise / norm / softmax / cpy / rope / conv2d / im2col / getrows …（多数是一个算子一对 .cpp+.hpp）' },
      { l: '模板实例化', n: 46, c: 'b', d: 'template-instances/：fattn-tile-* 10 个 + fattn-vec-* 36 个' },
      { l: '循环与融合', n: 18, c: 'c', d: 'ssm_scan / ssm_conv / wkv / gated_delta_net / gla / dsv4-hc / lightning-indexer / fusion' },
      { l: '主机框架', n: 14, c: 'd', d: 'ggml-sycl.cpp（全后端最大单文件）/ common / mem / memtrace / sycl_hw / presets / backend.hpp' },
      { l: '量化矩阵乘', n: 11, c: 'e', d: 'mmq / mmvq / dmmv / vecdotq / dequantize / quants / gemm / esimd' },
      { l: 'FlashAttn', n: 11, c: 'f', d: 'fattn / fattn-tile / fattn-vec / fattn-buffers / fattn-onednn / fattn-mkl' },
      { l: '公共 API', n: 1, c: 'g', d: 'ggml-sycl.h：唯一对外可见的头' }
    ];
    const host = wrap.querySelector('#bars');
    const bars = groups.map(g => { const b = U.bar(g.l, g.c); host.appendChild(b.el); return b; });
    bars.forEach(b => { b.fill.style.width = '0%'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先分类再讲细节。<span class="k">73 个常规算子内核</span>是最大一族：'
        + '多数是一个算子一个 .cpp + .hpp，与 CUDA 侧同名配对。',
      '<span class="k">46 个模板实例化文件</span>（26%）单独放在 template-instances/ —— '
        + '每个文件只有几行 <span class="v">DECL_*</span> 宏，用显式实例化把模板 kernel 编出来。',
      '<span class="k">循环 / 状态空间类算子</span>（ssm_scan、wkv、gated_delta_net…）18 个文件；'
        + 'fusion.cpp 负责图融合判定，对应 CUDA 侧的 ggml_cuda_can_fuse。',
      '<span class="k">主机框架 14 个文件</span>撑起整个后端：上下文、设备探测、内存池、trace。'
        + 'ggml-sycl.cpp 是最大的单文件，装下全部虚表与 graph compute。',
      '<span class="k">量化矩阵乘 11 个文件</span>是本课重点之一：mmq（大 batch）/ mmvq（小 batch）/ '
        + 'dmmv（单向量）三条路，加共用的 vecdotq / dequantize / quants。',
      '<span class="k">FlashAttention 11 个文件</span>：tile / vec 两套 kernel，'
        + '外加 oneDNN 与 MKL 两个厂商库版本 —— 这是 CUDA 侧没有的分支。',
      '结论：<span class="v">文件族 = 算子族</span>。CUDA 有什么，SYCL 目录里就能找到对应名字。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    groups.forEach((g, i) => tl.at(2400 + i * 2500, () => {
      bars.forEach((b, k) => {
        b.fill.style.width = (k <= i ? (groups[k].n / 73 * 100) : 0) + '%';
        b.val.textContent = k <= i ? groups[k].n : '';
        b.el.style.opacity = k === i ? '1' : '.42';
      });
      msg.innerHTML = texts[i];
    }));
    tl.at(18600, () => {
      bars.forEach(b => { b.el.style.opacity = '1'; });
      msg.innerHTML = texts[6];
    });
  }
},

/* ------------------------------------------------------ 3 ★ <span class="hl-b">同一批 kernel，同一套组织</span> */
{
  kicker: "L6-05 · 核心 ★",
  title: "★ <span class=\"hl-b\">同一批 kernel，同一套组织</span>",
  sub: "SYCL 的模板实例文件与 CUDA 的同名文件逐字相同，只有两处差异：.hpp/.cuh 与多一行 512。",
  caption: "8 行就是全部：一份 include、四条显式实例化。CUDA 侧同名文件 7 行（少了 512 那条）。",
  src: "ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q4_0.cpp",
  mark: [2, 7],
  lineNo: 1,
  code: `// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../fattn-vec.hpp"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_F16, GGML_TYPE_Q4_0);`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="top" style="gap:8px"></div>
      <div id="pairs"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const top = wrap.querySelector('#top');
    top.appendChild(U.card({ c: 'b', t: 'SYCL 侧：46 个实例文件',
      b: 'template-instances/*.cpp<br>10 个 tile + 36 个 vec',
      m: '#include "../fattn-vec.hpp"' }, { style: 'width:344px' }));
    top.appendChild(U.card({ c: 'd', t: 'CUDA 侧：121 个实例文件',
      b: 'template-instances/*.cu<br>多出 mmf / mmq / mma 三族',
      m: '#include "../fattn-vec.cuh"' }, { style: 'width:344px' }));

    const t = U.table(
      ['SYCL（.hpp / .cpp）', 'CUDA（.cuh / .cu）'],
      [['common.hpp', 'common.cuh'],
       ['mmq.cpp / mmvq.cpp / dmmv.cpp', 'mmq.cu / mmvq.cu /（v0.5.0 无 dmmv.cu）'],
       ['fattn-tile.hpp / fattn-vec.hpp', 'fattn-tile.cuh / fattn-vec.cuh'],
       ['template-instances/*.cpp', 'template-instances/*.cu'],
       ['ggml-sycl.cpp', 'ggml-cuda.cu']],
      { monoCols: [0, 1] });
    wrap.querySelector('#pairs').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    rows.forEach(r => { r.className = ''; });
    const texts = [
      '这个文件 8 行：一行注释、一行 include、四条 <span class="v">DECL_FATTN_VEC_CASE</span>。'
        + '它不实现任何东西，只负责把模板 kernel <span class="k">显式实例化</span>出来。',
      '<span class="k">D（头维度）64/128/256/512</span> 各实例化一次。'
        + 'CUDA 侧同名文件只有 64/128/256 三条 —— 这是两边真实的差异之一。',
      '左边一列是 SYCL，右边一列是 CUDA。'
        + '<span class="k">文件名去掉后缀就是同一张表</span>：一个算子族 = 一对文件。',
      '连"实例文件放在 template-instances/ 子目录、由 CMake 通配收集"这个做法都照搬。'
        + 'L6-04 讲过 CUDA 侧的同一套组织方式，这里是它的镜像。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4000, () => { msg.innerHTML = texts[1]; });
    rows.forEach((r, i) => tl.at(7300 + i * 1700, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[2];
    }));
    tl.at(16200, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[3];
    });
  }
},

/* ------------------------------------------------------ 4 CUDA ↔ SYCL <span class="hl-a">结构对照表</span> */
{
  kicker: "L6-05 · 核心",
  title: "CUDA ↔ SYCL <span class=\"hl-a\">结构对照表</span>",
  sub: "同一个算子，两边落在哪个文件、哪个函数。左边一列看完，对应关系就成立。",
  caption: "所有行号都来自上游 v0.5.0。SYCL 侧先看 buffer 虚表：字段顺序与 CUDA 侧一致。",
  src: "ggml/src/ggml-sycl/ggml-sycl.cpp",
  mark: [0, 4, 7],
  lineNo: 917,
  code: `static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const dim = s => '<span style="color:var(--dim)">' + s + '</span>';
    const t = U.table(
      ['环节', 'CUDA 侧（file:line）', 'SYCL 侧（file:line）'],
      [['后端上下文', 'ggml_backend_cuda_context<br>' + dim('common.cuh:1455'),
        'ggml_backend_sycl_context<br>' + dim('common.hpp:336')],
       ['执行流数组', 'cudaStream_t streams[][]<br>' + dim('common.cuh:1460'),
        'queue_ptr qptrs[][]<br>' + dim('common.hpp:341')],
       ['设备内存池', 'ggml_cuda_pool_alloc<br>' + dim('common.cuh:1215'),
        'ggml_sycl_pool_alloc<br>' + dim('common.hpp:263')],
       ['buffer 虚表', 'ggml_backend_cuda_buffer_interface<br>' + dim('ggml-cuda.cu:853'),
        'ggml_backend_sycl_buffer_interface<br>' + dim('ggml-sycl.cpp:917')],
       ['device 虚表', 'ggml_backend_cuda_device_interface<br>' + dim('ggml-cuda.cu:5651'),
        'ggml_backend_sycl_device_interface<br>' + dim('ggml-sycl.cpp:6903')],
       ['MUL_MAT 分派', 'ggml_cuda_mul_mat<br>' + dim('ggml-cuda.cu:1823'),
        'ggml_sycl_mul_mat<br>' + dim('ggml-sycl.cpp:4765')],
       ['图执行入口', 'ggml_backend_cuda_graph_compute<br>' + dim('ggml-cuda.cu:4421'),
        'ggml_backend_sycl_graph_compute<br>' + dim('ggml-sycl.cpp:6202')]],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '先看代码区：SYCL 的 buffer 虚表有 11 个字段，'
        + '<span class="v">ggml_backend_buffer_i</span> 定义在 L3-01 / ggml-backend-impl.h 里。',
      '<span class="k">上下文与执行流</span>：两边都是"设备 × 流"的二维数组，'
        + '只是元素类型从 cudaStream_t 换成 queue_ptr。',
      '<span class="k">内存池</span>：<span class="v">*_pool_alloc</span> 的 alloc/free 接口、'
        + 'RAII 析构、actual_size 记录，字段完全一样。',
      '<span class="k">虚表</span>：buffer / buffer_type / device / reg 四张，'
        + '字段顺序按 ggml-backend-impl.h 固定，两边逐项填。',
      '<span class="k">算子入口</span>：<span class="v">*_mul_mat</span> 承担同样的分派职责 —— '
        + '量化 / 非量化、单向量 / 大 batch 都在这里选路。',
      '<span class="k">图执行</span>：都先做兼容性判定，再决定"录制重放"还是"逐节点执行"。',
      '这张表就是本课的验收点：<span class="k">任给一个算子，你能同时说出两边的文件名与函数名</span>。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3200 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      U.markLines(document, []);
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(21000, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[6];
    });
  }
},

/* ------------------------------------------------------ 5 ★ 差异在<span class="hl-e">数据面</span>，不在算子层 */
{
  kicker: "L6-05 · 核心 ★",
  title: "★ 差异在<span class=\"hl-e\">数据面</span>，不在算子层",
  sub: "同一张算子表，换掉的是\"怎么发命令、内存从哪来、谁来算 GEMM\"。",
  caption: "编译链：oneAPI DPC++（icpx -fsycl）取代 nvcc；设备/队列/事件统一走 dpct:: 命名空间（定义在 ggml-sycl/dpct/helper.hpp:101）。",
  src: "ggml/src/ggml-sycl/common.hpp",
  mark: [5, 13, 14, 20],
  lineNo: 336,
  code: `struct ggml_backend_sycl_context {
    int device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(int device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(int device, int stream) {
        if (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        return qptrs[device][stream];
    }

    queue_ptr stream() {
        return stream(device, 0);
    }`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '执行流：stream -> queue',
        b: '两边都是"设备 × 流"的二维数组；<br>stream() 返回的是 sycl::queue 指针。',
        m: 'cudaStream_t  ->  sycl::queue *' },
      { c: 'b', t: '设备内存：cudaMalloc -> USM',
        b: '两边都有 RAII 的 *_pool_alloc；<br>USM 分配走 sycl::malloc_device，对齐都是 128 字节。',
        m: 'ggml_sycl_malloc_device / free_device' },
      { c: 'c', t: '事件：cudaEvent_t -> dpct::event_ptr',
        b: 'ggml_tensor_extra_gpu 里 events 的角色不变：<br>多设备之间的同步。',
        m: 'dpct::event_ptr events[dev][stream]' },
      { c: 'd', t: '数学库：cuBLAS -> oneDNN / oneMKL',
        b: 'gemm.hpp 用 dnnl::matmul 做兜底 GEMM；<br>FlashAttention 还多出 oneDNN SDPA 与 MKL 两条路。',
        m: 'DnnlGemmWrapper::gemm()' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:344px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const steps = [
      { card: -1, t: '代码区是 <span class="v">ggml_backend_sycl_context</span>：它只做三件事 —— '
        + '记住设备号、持有队列、管内存池。' },
      { card: 0, t: '<span class="k">执行流</span>：<span class="v">qptrs[device][stream]</span> 是个惰性缓存，'
        + '第一次用才去取 <span class="v">dpct::get_device(device).default_queue()</span>。' },
      { card: 1, t: '<span class="k">内存</span>：设备分配走 <span class="v">ggml_sycl_malloc_device</span>（内部 '
        + '<span class="v">sycl::malloc_device</span>），RAII 包装是 '
        + '<span class="v">ggml_sycl_pool_alloc</span>；对齐同样是 128。' },
      { card: 2, t: '<span class="k">事件</span>：<span class="v">ggml_tensor_extra_gpu</span> 里 events 的用途不变，'
        + '类型从 cudaEvent_t 换成 dpct::event_ptr。' },
      { card: 3, t: '<span class="k">数学库</span>：cuBLAS 的位置被 oneDNN（dnnl::matmul）与 oneMKL 顶上；'
        + '非量化兜底 GEMM 走 <span class="v">ggml_sycl_op_mul_mat_sycl</span>。' },
      { card: -1, t: '注意：<span class="k">惰性建流这一点 CUDA 侧也一样</span>'
        + '（common.cuh:1528 同为 if (nullptr) 才建）；差别在动作 —— CUDA 用 '
        + '<span class="v">cudaStreamCreateWithFlags</span> 新建一条流，SYCL 借设备的默认队列。' },
      { card: -1, t: '所以：<span class="k">算子层（switch、量化类型、tile 结构）可以照搬，数据面必须重写。</span>'
        + '这就是"结构同构 + 局部替换"的完整含义。' }
    ];
    steps.forEach((s, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => {
        e.style.opacity = (s.card < 0) ? '.55' : (k === s.card ? '1' : '.32');
      });
      msg.innerHTML = s.t;
    }));
    tl.at(19600, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = steps[6].t;
    });
  }
},

/* ------------------------------------------------------ 6 L3-01 的<span class="hl-d">虚表</span>在 SYCL 侧逐项填满 */
{
  kicker: "L6-05 · 后端契约",
  title: "L3-01 的<span class=\"hl-d\">虚表</span>在 SYCL 侧逐项填满",
  sub: "registry、device、buffer type（外加 buffer）四张接口，字段顺序由 ggml-backend-impl.h 钉死。",
  caption: "回顾 L3-01：算子能不能落到某个后端，取决于 supports_op / supports_buft 这两张表怎么说。",
  src: "ggml/src/ggml-sycl/ggml-sycl.cpp",
  mark: [3, 10, 11, 13],
  lineNo: 6903,
  code: `static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'buffer_i · 11 项', b: 'free_buffer / get_base / init_tensor /<br>memset_tensor / set_tensor / get_tensor /<br>set_tensor_2d / get_tensor_2d / cpy_tensor / clear / reset',
        m: 'ggml-sycl.cpp:917' },
      { c: 'b', t: 'buffer_type_i · 6 项', b: 'get_name / alloc_buffer /<br>get_alignment（返回 128）/<br>get_max_size / get_alloc_size / is_host',
        m: 'ggml-sycl.cpp:1055' },
      { c: 'c', t: 'device_i · 15 项', b: 'get_name / get_description / get_memory /<br>get_type / get_props / init_backend /<br>get_buffer_type / get_host_buffer_type /<br>buffer_from_host_ptr / supports_op /<br>supports_buft / offload_op / event_*',
        m: 'ggml-sycl.cpp:6903' },
      { c: 'd', t: 'reg_i · 4 项', b: 'get_name / get_device_count /<br>get_device /<br>get_proc_address（暴露 split buffer 与 comm_*）',
        m: 'ggml-sycl.cpp:7212' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:163px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'L3-01 说过：后端不是"函数集合"，而是<span class="k">几张函数指针表</span>。'
        + '代码区是其中的 device 表。',
      '<span class="v">buffer_i</span>：管一块显存里的张量读写。'
        + 'SYCL 侧的 set_tensor_2d / get_tensor_2d 是 NULL —— 与 CUDA 侧不同。',
      '<span class="v">buffer_type_i</span>：管分配。<span class="k">对齐 128 字节</span>，'
        + '与 CUDA 侧 get_alignment 返回的 128 一致。',
      '<span class="v">device_i</span>：这就是代码区这一段。'
        + '<span class="k">supports_op 决定调度器把哪些算子切过来</span>（L4-02）。',
      '<span class="v">reg_i</span>：注册入口。<span class="v">get_proc_address</span> 再额外暴露 '
        + 'split buffer 与 comm_* 三件套，对应 CUDA 侧的同名查询。',
      '四张表都由 <span class="v">ggml-sycl.cpp</span> 一个文件提供 —— '
        + '和 CUDA 侧全部塞在 ggml-cuda.cu 里是同一种做法。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13600, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 7 量化矩阵乘：<span class="hl-c">三条路</span>都搬过来了 */
{
  kicker: "L6-05 · 算子移植",
  title: "量化矩阵乘：<span class=\"hl-c\">三条路</span>都搬过来了",
  sub: "mmq（大 batch）/ mmvq（小 batch）/ dmmv（单向量）；三者共用同一套 q8_1 量化与 vec_dot。",
  caption: "回顾 L6-02：CUDA 侧同样按 batch 大小选 mmq / mmvq。SYCL 侧多保留了 dmmv 这一条。",
  src: "ggml/src/ggml-sycl/mmq.cpp",
  mark: [0, 2, 4],
  lineNo: 2987,
  code: `    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'DMMV · 单向量',
        b: '反量化后立刻乘向量；<br>只在 src1->ne[1] == 1 时可用。',
        m: 'ggml_sycl_op_dequantize_mul_mat_vec' },
      { c: 'b', t: 'MMVQ · 小 batch',
        b: '权重 × q8_1 点积；<br>批上限 MMVQ_MAX_BATCH_SIZE = 8。',
        m: 'ggml_sycl_op_mul_mat_vec_q' },
      { c: 'c', t: 'MMQ · 大 batch',
        b: 'tile 化 GEMM；<br>每个量化类型一个 case。',
        m: 'ggml_sycl_op_mul_mat_q' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:224px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '代码区是 mmq 的类型分派：<span class="v">switch (src0->type)</span>，'
        + '每个量化类型调一个 *_sycl kernel。',
      '第一段没有名字：<span class="v">ggml_sycl_op_mul_mat_q</span>（mmq.cpp:2962）'
        + '收下 row_low/row_high，把切分后的行段交给 per-type kernel。',
      '<span class="v">ggml_mul_mat_q4_0_q8_1_sycl</span>：<span class="k">权重用块量化格式，'
        + '激活统一量化成 q8_1</span>。这个"量化 × q8_1"的组合与 CUDA 侧 mmq 完全一致。',
      'CUDA 侧同样是一张 per-type 表：<span class="v">ggml_cuda_mul_mat_q_switch_type</span>'
        + '（mmq.cu:8）里 <span class="v">mul_mat_q_case&lt;GGML_TYPE_Q4_0&gt;</span>。',
      '三个 SYCL 入口的形参表是同一个形状：<span class="v">src0_dd_i / src1_ddq_i / dst_dd_i / row_low / '
        + 'row_high / src1_ncols / src1_padded_row_size / stream</span> —— '
        + 'CUDA 侧的 <span class="v">ggml_cuda_op_mul_mat_vec_q</span>（mmvq.cuh:14）逐项对应，'
        + '只有 stream 类型不同；CUDA 侧 mmq 的顶层入口是另一种形状'
        + '（<span class="v">ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst)</span>，mmq.cu:85）。',
      '差异在哪：CUDA 侧 v0.5.0 已经没有独立的 dmmv 文件，而 SYCL 侧把它留成 '
        + '<span class="v">dmmv.cpp</span> + 一组 <span class="v">*_reorder</span> 变体。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(9700, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
    tl.at(13100, () => { msg.innerHTML = texts[4]; });
    tl.at(16600, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 graph_compute：录制重放 vs 逐节点执行 */
{
  kicker: "L6-05 · 执行",
  title: "graph_compute：录制重放 vs 逐节点执行",
  sub: "和 CUDA 侧同一套判据：先问\"这张图能不能录\"，能录就用可更新的执行图，不能录就逐节点跑。",
  caption: "开关是 GGML_SYCL_ENABLE_GRAPH（默认 0）；还需要设备支持 ext_oneapi_limited_graph。",
  src: "ggml/src/ggml-sycl/ggml-sycl.cpp",
  mark: [6, 11, 17, 19],
  lineNo: 6202,
  code: `static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    bool use_sycl_graph = false;
    if (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    if (use_sycl_graph) {
        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['步骤', 'SYCL 侧调用', 'CUDA 侧对照物'],
      [['① 兼容性判定', 'check_graph_compatibility(cgraph)', 'ggml_cuda_graph_check_compability'],
       ['② 建图对象', 'sycl_ex::command_graph', 'ggml_cuda_graph 实例'],
       ['③ 录制', 'begin_recording / end_recording', 'cudaStreamBeginCapture / EndCapture'],
       ['④ 固化与更新', 'finalize(updatable) / exec_graph->update', 'cudaGraphExecUpdate'],
       ['⑤ 重放', 'ext_oneapi_graph(*exec_graph)', 'cudaGraphLaunch']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '代码区是入口函数。它先问两个问题：'
        + '<span class="v">g_ggml_sycl_enable_graph</span> 开了吗？这张图兼容吗？',
      '<span class="k">① 兼容性</span>：检查 cgraph 里有没有"录制图时不支持"的节点 —— '
        + 'CONCAT、MUL_MAT_ID，以及没有 async 内存扩展时的 MUL_MAT。',
      '<span class="k">②③ 建图与录制</span>：<span class="v">begin_recording</span> 之后，'
        + '原来的 <span class="v">graph_compute_impl</span> 一行不改地跑一遍，'
        + '所有 kernel 提交被记进图里。',
      '<span class="k">④ 固化与更新</span>：第一次 finalize 成可执行图，'
        + '之后每次 <span class="v">update</span> —— 形状不变就复用，省掉重复提交开销。',
      '<span class="k">⑤ 重放</span>：<span class="v">stream()->ext_oneapi_graph()</span> 提交整张图。',
      '设备不支持时直接退回：<span class="v">graph_compute_impl</span> 逐节点执行，'
        + '行为与没有图时完全一致。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3200 + i * 2800, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(18200, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 9 把 SYCL 后端压成一张表 */
{
  kicker: "L6-05 · 收束",
  title: "把 SYCL 后端压成一张表",
  sub: "同一个算子，两边各在哪；记住这张表，L6-06 起的其它 GPU 后端都是同一套问法。",
  caption: "下一课 L6-06：Vulkan 后端的主机端；它的 kernel 在 L6-07 里用 GLSL 计算着色器写。",
  src: "ggml/src/ggml-sycl/fattn.cpp",
  mark: [2, 3, 4, 5],
  lineNo: 97,
  code: `enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_MKL      = 300,
};`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const dim = s => '<span style="color:var(--dim)">' + s + '</span>';
    const t = U.table(
      ['算子 / 环节', 'CUDA 侧', 'SYCL 侧'],
      [['量化矩阵乘', 'mmq.cu / mmvq.cu<br>' + dim('ggml-cuda.cu:1823 分派'),
        'mmq.cpp / mmvq.cpp / dmmv.cpp<br>' + dim('ggml-sycl.cpp:4765 分派')],
       ['FlashAttention', 'fattn.cu:755 / fattn-tile.cu<br>' + dim('内核选择枚举 fattn.cu:517'),
        'fattn.cpp:276 / fattn-tile.cpp<br>' + dim('内核选择枚举 fattn.cpp:97')],
       ['模板实例化', 'DECL_FATTN_VEC_CASE<br>' + dim('fattn-vec.cuh:574'),
        'DECL_FATTN_VEC_CASE<br>' + dim('fattn-vec.hpp:644')],
       ['执行流 / 内存', 'cudaStream_t + cudaMalloc<br>' + dim('common.cuh:1460'),
        'queue_ptr + USM 指针<br>' + dim('common.hpp:341')],
       ['兜底 GEMM', 'ggml_cuda_mul_mat_cublas<br>' + dim('ggml-cuda.cu:1622'),
        'ggml_sycl_op_mul_mat_sycl<br>' + dim('ggml-sycl.cpp:2914')],
       ['图执行', 'ggml_backend_cuda_graph_compute<br>' + dim('ggml-cuda.cu:4421'),
        'ggml_backend_sycl_graph_compute<br>' + dim('ggml-sycl.cpp:6202')]],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '给一个算子，比如 <span class="mono">GGML_OP_MUL_MAT</span>：'
      + '要说出 SYCL 后端与 CUDA 后端在它上面的结构对应关系，你需要报出哪几项？',
      '四项，缺一不可：<br>'
      + '① <b>分派入口</b>：<span class="mono">ggml_cuda_mul_mat</span>（ggml-cuda.cu:1823）'
      + ' ↔ <span class="mono">ggml_sycl_mul_mat</span>（ggml-sycl.cpp:4765）；<br>'
      + '② <b>kernel 文件</b>：<span class="mono">mmq.cu/mmvq.cu</span>'
      + ' ↔ <span class="mono">mmq.cpp/mmvq.cpp/dmmv.cpp</span>；<br>'
      + '③ <b>回调形参表</b>：<span class="mono">ggml_cuda_op_mul_mat_vec_q</span>（mmvq.cuh:14）'
      + ' ↔ <span class="mono">ggml_sycl_op_mul_mat_vec_q</span>（mmvq.hpp:19）'
      + '（只差 <span class="mono">cudaStream_t</span> / <span class="mono">dpct::queue_ptr</span>）；<br>'
      + '④ <b>数据面</b>：stream/pointer ↔ queue/USM。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '收束成六行。前两行是本课的主角（量化矩阵乘、FlashAttention）。',
      '第三行是 <span class="k">★ 结构同构最硬的证据</span>：同一个宏名、同一个展开体，'
        + '只差 <span class="v">ggml_sycl_*</span> / <span class="v">ggml_cuda_*</span> 两个名字。',
      '代码区这一段说明两边<b>不是</b>完全一样：SYCL 多了 ONEDNN(150) 与 MKL(300)，'
        + 'CUDA 多了 MMA_F16(400)；共有的 NONE/VEC/TILE 取值相同。',
      '一句话：<span class="k">SYCL 后端 = CUDA 后端的算子层 + SYCL 的数据面 + 厂商库分支</span>。',
      '下一课 L6-06 换一个完全不同的后端：Vulkan 的主机端 —— 那里没有 C++ kernel，'
        + '算子在 L6-07 里要写成 GLSL 计算着色器。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(17400, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[3];
    });
    tl.at(19800, () => { msg.innerHTML = texts[4]; });
  }
},

];
