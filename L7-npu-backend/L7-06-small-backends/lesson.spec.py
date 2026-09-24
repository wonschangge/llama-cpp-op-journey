#!/usr/bin/env python3
"""L7-06 · 小后端合集：OpenCL / WebGPU / MUSA / RPC / BLAS / ZenDNN / zDNN —— 课件 spec。

运行：python3 L7-npu-backend/L7-06-small-backends/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

BLAS = 'ggml/src/ggml-blas/ggml-blas.cpp'
RPC = 'ggml/src/ggml-rpc/ggml-rpc.cpp'
OCL = 'ggml/src/ggml-opencl/ggml-opencl.cpp'
WGPU = 'ggml/src/ggml-webgpu/ggml-webgpu.cpp'
ZDNN = 'ggml/src/ggml-zdnn/ggml-zdnn.cpp'
ZENDNN = 'ggml/src/ggml-zendnn/ggml-zendnn.cpp'

L = Lesson(
    id='L7-06',
    layer='L7 · NPU 与加速器后端',
    title='小后端合集：OpenCL / WebGPU / MUSA / RPC / BLAS / ZenDNN / zDNN',
    codecap='ggml/src/ggml-{blas,rpc,opencl,webgpu,musa,zdnn,zendnn}/ 与 ggml/include/（逐字引用）',
    nav={'prev': {'href': '../L7-05-virtgpu-virtualized/index.html',
                  'label': 'L7-05 VirtGPU 后端'},
         'next': {'href': '../../L8-end-to-end/L8-01-journey-of-mul-mat/index.html',
                  'label': 'L8-01 ★ 端到端：一个 mul_mat 的完整旅程'}},
)

# 本课覆盖声明：plan_matrix.py 指派给 L7-06 的全部 29 个文件（先声明，后引用）。
L.cover(
    'ggml/include/ggml-blas.h',
    'ggml/include/ggml-opencl.h',
    'ggml/include/ggml-rpc.h',
    'ggml/include/ggml-webgpu.h',
    'ggml/include/ggml-zdnn.h',
    'ggml/include/ggml-zendnn.h',
    'ggml/src/ggml-blas/ggml-blas.cpp',
    'ggml/src/ggml-musa/mudnn.cu',
    'ggml/src/ggml-musa/mudnn.cuh',
    'ggml/src/ggml-opencl/cl-program-cache.cpp',
    'ggml/src/ggml-opencl/cl-program-cache.h',
    'ggml/src/ggml-opencl/fa_tune.h',
    'ggml/src/ggml-opencl/ggml-opencl.cpp',
    'ggml/src/ggml-opencl/libdl.h',
    'ggml/src/ggml-rpc/ggml-rpc.cpp',
    'ggml/src/ggml-rpc/transport-apple.cpp',
    'ggml/src/ggml-rpc/transport-apple.h',
    'ggml/src/ggml-rpc/transport.cpp',
    'ggml/src/ggml-rpc/transport.h',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp',
    'ggml/src/ggml-zdnn/common.hpp',
    'ggml/src/ggml-zdnn/ggml-zdnn.cpp',
    'ggml/src/ggml-zdnn/mmf.cpp',
    'ggml/src/ggml-zdnn/mmf.hpp',
    'ggml/src/ggml-zdnn/utils.cpp',
    'ggml/src/ggml-zdnn/utils.hpp',
    'ggml/src/ggml-zendnn/ggml-zendnn.cpp',
)

L.prereqs('`L7-05`')
L.goal('说出这 7 个后端各自"最少要写什么"，并给出实测的文件数与行数。',
       '解释同一份 `ggml_backend_i` 契约如何容纳差别极大的数据面（宿主内存 / 设备显存 / 硬件专用布局 / 远端机器）。',
       '复述 RPC 后端把一个 tensor 送到远端的完整路径：序列化 → 打包 → socket → 服务端还原并写入远端内存。',
       '指出 RPC 设备在调度器眼里就是一块 GPU（`supports_op` 目前恒真），因而是 L4-02 图切分的普通参与者。')

L.note('**一句话**：这一课把七个"小"后端放在一起看 —— 它们填的是**同一张** `ggml_backend_i` 虚表'
       '（L3-01 讲过的那三张表），但数据面的差别大到荒谬：BLAS 的数据就在 CPU 内存里、'
       'WebGPU 的数据在浏览器的 GPU 缓冲里、zDNN 的数据被变换成硬件专用布局、'
       'RPC 的数据在**另一台机器**上。')
L.note('"小"指的是**要写多少自有代码**，不是能力：MUSA 目录只有 2 个文件（124 行）——'
       '因为整个后端复用 `ggml-cuda` 的源码，MUSA 只补了一层 MUDNN 胶水；'
       '而 OpenCL 单个 `.cpp` 就有 29502 行。两者在调度器眼里没有区别。')
L.note('本课覆盖 `python3 tools/plan_matrix.py --files` 指派给 `L7-06` 的**全部 29 个文件**'
       '（7 个后端目录 + 6 个公共头；`ggml/src/ggml-musa/CMakeLists.txt` 只是构建脚本，'
       '用来解释"MUSA 为什么只有 2 个文件"，不计入覆盖声明）。')

# ================================================================== 第 1 幕

L.scene(
    kicker='L7 · NPU 与加速器后端',
    title='七个后端，<span class="hl-a">一个契约</span>',
    sub='回顾 L3-01：ggml 不认识任何硬件，只认识 ggml_backend_i 这一堆函数指针。',
    caption='本课 9 幕：先量一遍规模，再看 BLAS / RPC / WebGPU / OpenCL，最后把三个"厂商库转发"型放在一起对比。',
    src='ggml/include/ggml-zdnn.h', parts=[(10, 17)], duration=15000,
    mark_src=[11, 13],
    notes_src={13: 'zDNN 的公共头一共 2 个函数：要一块设备 buffer、要一个注册表项。契约的其余部分全在内部虚表里'},
    visual='''
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
'''
)

# ================================================================== 第 2 幕

L.scene(
    kicker='L7-06 · 实测',
    title='一张表量完七个后端',
    sub='行数、文件数都来自 wc -l 实测；"要写什么"一句话来自逐文件阅读。',
    caption='RPC 的 6 个文件里有 4 个是传输层（transport.h/.cpp + Apple RDMA 的 .h/.cpp）；OpenCL 的 6 个文件里含磁盘缓存与调参表。',
    src='ggml/include/ggml-rpc.h', parts=[(19, 31)], duration=17000,
    mark_src=[20, 23, 27, 30, 31],
    notes_src={27: '同一个库里既有客户端（init/buffer_type），也有服务端（start_server）—— 一眼看不出谁是"远端"',
               31: 'add_server：把一台 RPC 服务器变成一个可注册的后端，于是调度器可以像用 GPU 一样用它（见 L4-02）'},
    visual='''
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
'''
)

# ================================================================== 第 3 幕

L.scene(
    kicker='L7-06 · BLAS',
    title='BLAS：整个后端就是<span class="hl-b">一个 switch</span>',
    sub='不新增数据面、不新增 buffer type —— 只是把两个 op 换成厂商 gemm。',
    caption='数据面这一层的证据在 ggml-blas.cpp:381-382（device_get_buffer_type 直接返回 ggml_backend_cpu_buffer_type），第 9 幕会再引一次。',
    src=BLAS, parts=[(226, 260)], duration=16000,
    mark_src=[226, 236, 237, 241, 252, 253],
    notes_src={237: '唯一被"加速"的 op：转给 ggml_backend_blas_mul_mat()，内部就是 cblas_sgemm（第 142 行）',
               252: 'default 直接 abort：送进来一个不支持的 op 就死。所以 supports_op 必须与这个 switch 严格一致'},
    visual='''
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
'''
)

# ================================================================== 第 4 幕

L.scene(
    kicker='L7-06 · RPC · 核心',
    title='★ <span class="hl-a">rpc_tensor</span>：一个 tensor 在网线上的样子',
    sub='ggml_tensor 里有指针，不能直接过网；于是有一个逐字段对应的 POD 版本。',
    caption='验收点就在这一幕和下一幕：tensor 先变成 rpc_tensor，再和裸数据拼成一块内存发给远端。',
    src=RPC, parts=[(39, 58)], duration=18000,
    mark_src=[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 58],
    notes_src={43: 'buffer 不是本地指针：它是远端分配时记下的 remote_ptr（serialize_tensor，第 642 行）',
               52: 'data 是【本机指针值】本身（第 643 行）—— 服务端把它当远端地址直接用（deserialize_tensor，第 1415 行）',
               55: 'use_count 来自图上的引用计数，服务端重建图时要用（第 1003-1009 行）',
               58: '#pragma pack(push, 1) 下的定长结构：两端对同一份字节布局的约定，就是 RPC 协议'},
    visual='''
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
'''
)

# ================================================================== 第 5 幕

L.scene(
    kicker='L7-06 · RPC · 全链路',
    title='把 tensor 送过去：<span class="hl-c">打包</span> + 一条 socket',
    sub='元数据 + 一个标志位 + 偏移 + 裸数据 = 一次 SET_TENSOR。整张图也走同一条路。',
    caption='这一步是验收点的答案：客户端 serialize_set_tensor → dispatcher → socket → 服务端 recv_msg → deserialize → 写入远端 buffer。',
    src=RPC, parts=[(701, 741)], duration=19000,
    mark_src=[701, 702, 706, 707, 708, 709, 710, 720, 722, 739, 740],
    notes_src={701: '一行注释写死了协议布局：| rpc_tensor | cache_flag(1B) | offset(8B) | data(size B) |',
               724: '权重走哈希缓存：先把 FNV 哈希发给远端问"你有没有"（阈值见第 716-717 行，服务端分支见第 1987 行），命中就连数据都不用发',
               740: 'dispatcher->send() 把这块内存交给 socket；命令字 RPC_CMD_SET_TENSOR 是枚举里的第 68 行'},
    visual='''
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
'''
)

# ================================================================== 第 6 幕

L.scene(
    kicker='L7-06 · WebGPU',
    title='WebGPU：内核是 <span class="hl-e">WGSL 源码字符串</span>',
    sub='同一张虚表，但一半的槽是 NULL，且默认异步 —— 因为它跑在浏览器里。',
    caption='回顾 L3-01：ggml_backend_i 里只有 get_name / free / graph_compute 是必需的，WebGPU 正好把这几个填满。',
    src=WGPU, parts=[(333, 355)], duration=17000,
    mark_src=[333, 337, 338, 343, 347, 348, 354],
    visual='''
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
'''
)

# ================================================================== 第 7 幕

L.scene(
    kicker='L7-06 · OpenCL',
    title='OpenCL：<span class="hl-d">运行期</span>加载、编译、缓存',
    sub='29502 行里绝大部分是内核与派发；真正"后端骨架"的部分只有几千行。',
    caption='对照 L5-05：CPU 后端的内核是编译进二进制的 C；OpenCL 的内核是运行期才存在的字符串。',
    src=OCL, parts=[(1382, 1401)], duration=17000,
    mark_src=[1382, 1388, 1390, 1394, 1398],
    visual='''
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
'''
)

# ================================================================== 第 8 幕

L.scene(
    kicker='L7-06 · 厂商库转发',
    title='★ MUSA / ZenDNN / zDNN：<span class="hl-a">翻译深度</span>决定数据面',
    sub='三个后端都在"把 ggml 的布局翻译成厂商库要的布局"，但翻译到哪一层，差别巨大。',
    caption='对照 L7-01~L7-05：CANN / OpenVINO 等后端也在做同一件事，只是翻译的目标库不同。',
    src='ggml/src/ggml-zdnn/utils.cpp', parts=[(25, 44)], duration=18000,
    mark_src=[25, 30, 31, 38, 39, 42, 43],
    notes_src={39: 'ztensor 的存储由 zDNN 自己 malloc：这块数据【不在】ggml 的 CPU buffer 里',
               43: '写入时把 ggml 布局变换成硬件布局 —— 数据面就是在这里被改写的'},
    visual='''
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
'''
)

# ================================================================== 第 9 幕

L.scene(
    kicker='L7-06 · 收束',
    title='把这一课压成一张表',
    sub='四种数据面，一套契约。下一课 L8-01 让一个 mul_mat 从模型一路走到内核。',
    caption='末幕代码只引两行：BLAS 的 buffer type 就是 CPU 的 —— 数据面"零改动"的极端例子。',
    src=BLAS, parts=[(381, 382)], duration=20000,
    mark_src=[381, 382],
    visual='''
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
'''
)

# ================================================================== source.md 讲解

L.section(
    '一、七个后端，七份"最小可行后端"清单',
    '本课覆盖 `plan_matrix.py` 指派给 `L7-06` 的全部 29 个文件。规模实测（`wc -l`）：\n\n'
    '| 后端 | 本课文件 | 总行数 | 特有的东西 |\n|---|---|---|---|\n'
    '| BLAS | 2 | 555 | 只有 1 个 `.cpp`：一个 switch + cblas |\n'
    '| MUSA | 2 | 124 | 只有 MUDNN 胶水；算子实现复用 ggml-cuda |\n'
    '| ZenDNN | 2 | 860 | 只翻译 matmul（含 MoE 的 group matmul） |\n'
    '| zDNN | 7 | 904 | 私有 buffer type + ztensor 变换 |\n'
    '| RPC | 6 | 3694 | 传输层（TCP / RDMA）+ 协议结构 |\n'
    '| WebGPU | 4 | 9185 | WGSL shader 库 + WGSL 预处理器 |\n'
    '| OpenCL | 6 | 30227 | 运行期编译 .cl / 磁盘二进制缓存 / Adreno 调参表 |\n\n'
    '合计 **29 文件 / 45549 行**。MUSA 目录里只有 `mudnn.cu` 与 `mudnn.cuh` 两个自有源文件，'
    '其余 `.cu` 来自 `../ggml-cuda/`（构建脚本 `ggml/src/ggml-musa/CMakeLists.txt` 用 '
    '`file(GLOB GGML_SOURCES_MUSA "../ggml-cuda/*.cu")` 收集）—— 该构建脚本不计入本课覆盖声明。\n\n'
    '公共头都很薄，形状只有两种：',
    src='ggml/include/ggml-blas.h', parts=[(11, 20)], lang='c')

L.section(
    '二、WebGPU 的公共头：连 init 都只为 examples 服务',
    '`ggml_backend_webgpu_init()` 的注释写着 "Needed for examples in ggml"：'
    '正式路径是注册表 + 设备（L3-02），这个入口只是给例子程序用的。',
    src='ggml/include/ggml-webgpu.h', parts=[(10, 15)], lang='c')

L.section(
    '三、OpenCL 的公共头：连 host buffer type 都暴露出来',
    'OpenCL 除了设备 buffer type，还导出一个 **host** buffer type —— 因为它的设备内存与宿主内存是两份，'
    '中间要显式搬运（`clEnqueueWriteBuffer`，见 `ggml-opencl.cpp:9786-9815`）。',
    src='ggml/include/ggml-opencl.h', parts=[(11, 20)], lang='c')

L.section(
    '四、ZenDNN 的公共头：一个 init、一个 set_n_threads、一个 reg',
    'ZenDNN 是 AMD CPU 上的算子库（源码注释：AMD optimized primitives backend for GGML）。'
    '它的公共 API 与 BLAS 几乎同形 —— 因为两者都是"复用宿主内存、只加速个别算子"的后端。',
    src='ggml/include/ggml-zendnn.h', parts=[(10, 18)], lang='c')

L.section(
    '五、RPC 协议：版本号与一个 static_assert',
    'RPC 是唯一一个**带协议版本**的后端：`GGML_OP_COUNT` 变化会改变 `rpc_tensor` 的含义，'
    '所以 `GGML_OP_COUNT` 一改，就必须同步更新 patch 版本号 —— 这条约束写进了头文件，编译期就会检查。',
    src='ggml/include/ggml-rpc.h', parts=[(9, 17)], lang='c')

L.section(
    '六、RPC 的传输层：一条 socket，两种实现',
    '`socket_t` 是一个 pimpl：上层只看到 `send_data` / `recv_data` / `flush`。'
    '`flush` 的注释说明了下层的差别 —— RDMA 传输会把写入合并成固定大小的帧，'
    '尾帧必须在消息边界显式 post；TCP 上它是 no-op。',
    src='ggml/src/ggml-rpc/transport.h', parts=[(13, 21)], lang='c')

L.section(
    '七、transport.cpp：socket_t 把一切转发给 impl',
    '这三个函数是整条 RPC 数据通路的最底层。TCP 实现里 `create_server` 只是 '
    '`socket/bind/listen`（`transport.cpp:656`），`connect` 还会设置 `TCP_NODELAY`（`:683`）。',
    src='ggml/src/ggml-rpc/transport.cpp', parts=[(603, 613)], lang='c')

L.section(
    '八、Apple 的 RDMA：为 Thunderbolt 单独写一个传输',
    'Apple 的 RDMA 与 Linux 差别大到"值得单独一个实现"：UC 而非 RC 队列、'
    '固定 128 KiB stride、依赖硬件信用流控。注意 `flush()` 的语义正是被它逼出来的。',
    src='ggml/src/ggml-rpc/transport-apple.h', parts=[(7, 25)], lang='c')

L.section(
    '九、transport-apple.cpp：常量里写满了硬件事实',
    '这些常量不是随意取的：4 KiB 是 Thunderbolt 帧、128 KiB stride = 32 帧，'
    '一次 SEND 必须覆盖与对应 RECV 相同数量的帧 —— 所以"半满也要发满一个 stride"。',
    src='ggml/src/ggml-rpc/transport-apple.cpp', parts=[(16, 36)], lang='c')

L.section(
    '十、WebGPU 的 shader 库：WGSL 与一个自制的预处理器',
    '`ggml-wgsl-shaders.hpp` 是构建期生成的内核库（不在本课覆盖域内），'
    '`pre_wgsl.hpp` 则是一个运行期的小预处理器：因为不同适配器的 limits 不同，'
    '同一个算子要按能力挑不同变体。',
    src='ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp', parts=[(1, 20)], lang='c')

L.section(
    '十一、pre_wgsl：只要 include 路径与宏，就够用了',
    '整个预处理器只需要 `Options{include_path, macros}` 两个参数 —— 它服务的是 WGSL 源码拼接，'
    '不是通用编译器。',
    src='ggml/src/ggml-webgpu/pre_wgsl.hpp', parts=[(1, 22)], lang='c')

L.section(
    '十二、OpenCL 的 libdl.h：给厂商预编译内核留的入口',
    '非 Windows 分支就是 `dlopen` / `dlsym` / `dlerror` 三件套。它在 OpenCL 后端里只服务一件事：'
    '运行期尝试加载 Adreno 的**预编译内核二进制库**（`libadreno-opencl-kernels.so` / '
    '`adreno-opencl-kernels.dll`，定义在 `ggml-opencl.cpp:21-23`）；加载失败就回退到内置内核源码'
    '（`ggml-opencl.cpp:6726-6744`）。注意 OpenCL 本身仍是构建期依赖'
    '（`find_package(OpenCL REQUIRED)`），这里动态加载的只是内核。',
    src='ggml/src/ggml-opencl/libdl.h', parts=[(57, 77)], lang='c')

L.section(
    '十三、OpenCL 的磁盘缓存：键是怎么算出来的',
    '缓存键把三样东西拼起来做 SHA-256：内核源码、编译选项、设备身份（`key_suffix` 里含 '
    'CL_DEVICE_NAME / CL_DRIVER_VERSION / CL_PLATFORM_VERSION 与格式版本）。'
    '所以换驱动、换编译选项都会自动 miss，不会读到错误的二进制。',
    src='ggml/src/ggml-opencl/cl-program-cache.cpp', parts=[(155, 171)], lang='c')

L.section(
    '十四、缓存开关与文件布局',
    '缓存的激活方式、目录规则与文件头布局都写在头文件注释里（这里只引前 14 行）。',
    src='ggml/src/ggml-opencl/cl-program-cache.h', parts=[(1, 14)], lang='c')

L.section(
    '十五、fa_tune.h：把调参从代码里赶出去',
    'Flash-Attention 的 tile 参数按 (dk, dv) 列表给出，注释写明覆盖 Adreno 7xx/8xx 与 X1 系列；'
    '抽成头文件的理由写在文件开头：调参数字好找、好改，而派发与编译逻辑留在主文件里。',
    src='ggml/src/ggml-opencl/fa_tune.h', parts=[(9, 24)], lang='c')

L.section(
    '十六、MUSA：整个后端只有一份胶水头',
    'MUSA 目录里唯一的头文件只声明了一个函数：`mudnnMemcpyAsync`。'
    '也就是说，这个后端对 ggml 的"新增能力"只有一个异步拷贝 —— 其余全部来自 ggml-cuda。',
    src='ggml/src/ggml-musa/mudnn.cuh', parts=[(1, 12)], lang='c')

L.section(
    '十七、mudnn.cu：把 ggml_tensor 翻译成 mudnn::Tensor',
    'dims 直接取 `ne[]`，strides 取 `nb[] / element_size`（换成以元素为单位）—— '
    '然后 `SetAddr` 把指针交出去。整个翻译过程没有任何数据搬运。',
    src='ggml/src/ggml-musa/mudnn.cu', parts=[(53, 66)], lang='c')

L.section(
    '十八、ZenDNN：翻译成一次 matmul_direct 调用',
    '注释把每个参数的含义都写清楚了：行主序、不转置 B、转置 A（因为 ggml 是列主序）、'
    'alpha/beta、bias、is_weights_const。量化权重（block_q8_0）额外补上 scale 的维度。',
    src=ZENDNN, parts=[(69, 102)], lang='c')

L.section(
    '十九、ZenDNN 的算子覆盖与回退',
    '和 BLAS 一样，它只认少数几个 op，default 走 `GGML_ABORT`；'
    '`supports_op` 里还有一个可关闭的"自适应回退"开关（函数在 `ggml-zendnn.cpp:688-692`，'
    '读取环境变量 `GGML_ZENDNN_ADAPTIVE_FALLBACK`）。',
    src=ZENDNN, parts=[(535, 567)], lang='c')

L.section(
    '二十、zDNN：每个 buffer 都带一个硬件张量描述符',
    '`ggml_backend_zdnn_buffer` 里除了 data/size，还挂着 `pre_tfm_desc`、`tfm_desc` 与 '
    '`zdnn_ztensor` —— 这三样就是"硬件专用数据面"的直接证据。',
    src='ggml/src/ggml-zdnn/common.hpp', parts=[(38, 48)], lang='c')

L.section(
    '二十一、zDNN 的 set_tensor：拷贝之后再变换一次',
    '先 `memcpy` 到自己的缓冲，再（若尚未变换）调用 `ggml_zdnn_load_tensor` 做硬件变换；'
    '计算 buffer 被复用时会先 `zdnn_reset_ztensor` 复位（源码注释指向一个真实 bug：LLAMA_SET_ROWS）。',
    src=ZDNN, parts=[(284, 294)], lang='c')

L.section(
    '二十二、zDNN 的 buffer type：明确声明"不是 host"',
    '`is_host` 返回 false，注释解释得直白：*while it resides in host memory, additional '
    'transformation is needed*。对齐要求 256 字节。',
    src=ZDNN, parts=[(376, 395)], lang='c')

L.section(
    '二十三、mmf.cpp：真正落到硬件的那一次调用',
    '前面一大堆断言都在校验"tensor 的 ne[] 与预先算好的 pre_tfm_desc 一致"；'
    '最后一行才是硬件调用，接着还要把输出从 DLF16 变换回 FP32（源码 TODO 承认这一步低效）。',
    src='ggml/src/ggml-zdnn/mmf.cpp', parts=[(63, 71)], lang='c')

L.section(
    '二十四、zDNN 的接口声明与类型映射',
    '`mmf.hpp` 只有矩阵乘一个函数 —— 与 `supports_op` 里"只支持 MUL_MAT"完全对应；'
    '`utils.hpp` 暴露的是建 tensor / 载入 tensor / 初始化 tensor 三件事。',
    src='ggml/src/ggml-zdnn/mmf.hpp', parts=[(1, 12)], lang='c')

L.section(
    '二十五、utils.hpp 与类型映射表',
    '`ggml_zdnn_type_mapping` 把 ggml 类型映射到 zDNN 类型；遇到不支持的类型直接 abort。',
    src='ggml/src/ggml-zdnn/utils.hpp', parts=[(1, 16)], lang='c')

L.conclusion(
    '四种数据面，一套契约',
    '把这一课的 7 个后端按"数据在哪、谁搬"归类，只有四种答案：\n\n'
    '1. **零改动**：BLAS / ZenDNN —— buffer type 直接复用 CPU 的，只是把个别算子换成厂商库；\n'
    '2. **设备内存**：WebGPU / OpenCL —— 数据在驱动管理的缓冲里，写入要过 `WriteBuffer` / `clEnqueueWriteBuffer`；\n'
    '3. **硬件专用布局**：zDNN —— 每个 tensor 有自己的 `zdnn_ztensor`，写入时做 `zdnn_transform_ztensor`，`is_host = false`；\n'
    '4. **远端机器**：RPC —— 数据在另一台机器上，tensor 与整张图都要序列化过网。\n\n'
    '四种答案，填的是**同一张** `ggml_backend_i` / `ggml_backend_buffer_type_i` / `ggml_backend_device_i`。'
    '这正是 L3-01 那三张表的抽象力：契约不描述"数据在哪"，只描述"怎么问、怎么给、怎么算"。')

L.conclusion(
    'RPC：tensor 如何过网（验收点）',
    '1. 客户端 `serialize_tensor`（`ggml-rpc.cpp:628`）把 `ggml_tensor` 填进定长的 `rpc_tensor`：'
    '形状/步长/op/参数原样复制，`buffer` 写远端句柄 `remote_ptr`（`:642`），`data` 写**本机指针值**（`:643`）；\n'
    '2. `serialize_set_tensor`（`:701`）把它和 `cache_flag`、`offset`、裸数据拼成一块连续内存；\n'
    '3. `dispatcher->send(RPC_CMD_SET_TENSOR, ...)`（`:740`）经 `socket_t::send_data`（`transport.cpp:603`）发出；'
    '大权重先用 FNV 哈希问远端是否已有（`:724-737`）；\n'
    '4. 服务端 `recv_msg` 收下整块（`:1977`），`deserialize_tensor` 按元数据重建 tensor 并把 `data` '
    '还原成远端地址（`:1415`），最后 `ggml_backend_tensor_set` 写入远端 buffer（`:1472`）。\n\n'
    '计算走同一条路：`serialize_graph`（`:1003`）把节点与全部 tensor 发过去，'
    '服务端重建图并调用自己的 `ggml_backend_graph_compute`（`:1772`）。')

L.conclusion(
    'RPC 设备就是一块 GPU',
    '`ggml_backend_rpc_device_get_type()` 返回 `GGML_BACKEND_DEVICE_TYPE_GPU`（`ggml-rpc.cpp:2174`），'
    '`supports_op` 目前**恒真**（`:2209-2213`，源码 TODO 写着应当去问远端）。'
    '所以 L4-02 的调度器会像对待普通 GPU 一样把子图切给它 —— '
    '`ggml_backend_rpc_add_server`（`ggml-rpc.h:31`）让一台远端服务器变成一个可注册的设备。')

L.footnote_add('正文表格里的行数按 `wc -l` 实测（例如 `ggml-blas.cpp` = 530 行）；'
               'README 的「覆盖的源文件」表由 `lessonkit` 生成，按换行符切分计数，'
               '每个文件会比 `wc -l` 多 1（文件末尾的换行也算一行）。')
L.footnote_add('WebGPU 的 WGSL 内核文本来自构建期生成的 `ggml-wgsl-shaders.hpp`'
               '（源文件在 `ggml/src/ggml-webgpu/wgsl-shaders/*.wgsl`，由 `embed_wgsl.py` 嵌入），'
               '该生成头不在本课覆盖域内；本课引用的是使用它的 `ggml-webgpu-shader-lib.hpp`。')
L.footnote_add('OpenCL 本身是构建期依赖（`ggml/src/ggml-opencl/CMakeLists.txt` 里 '
               '`find_package(OpenCL REQUIRED)`）；`libdl.h` 动态加载的是 Adreno 预编译内核库，'
               '不是 OpenCL 运行时。构建脚本不计入本课覆盖声明。')
L.footnote_add('本课覆盖 29 个文件，全部进覆盖声明；`ggml/src/ggml-musa/CMakeLists.txt` 只在'
               '第一段讲解里被引用来说明"MUSA 为什么只有 2 个自有源文件"，不计入覆盖声明。')
L.footnote_add('所有行数来自 `wc -l`，所有行号来自上游 `v0.5.0`（commit `7fe450e19305`）；'
               '引用一律由 `lessonkit` 按行号抽取，未手抄。')

if __name__ == '__main__':
    L.build()
