#!/usr/bin/env python3
"""L7-04 · ExecuTorch 后端（边缘/移动端）—— 课件 spec。

运行：python3 L7-npu-backend/L7-04-executorch-edge/lesson.spec.py

引用范围全部按行号从上游 v0.5.0 抽取（ggml-et.cpp 1880 行 / ggml-et-ops.cpp 2584 行 /
ggml-et-kernels.cpp 508 行 / ggml-et-common.h 86 行 / ggml/include/ggml-et.h 28 行 /
et-kernels/src/scale_f32.c 94 行 等），逐字保真由 tools/lessonkit.py 的构造保证。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

ET = 'ggml/src/ggml-et/ggml-et.cpp'
OPSC = 'ggml/src/ggml-et/ggml-et-ops.cpp'
OPSH = 'ggml/src/ggml-et/ggml-et-ops.h'
KRN = 'ggml/src/ggml-et/ggml-et-kernels.cpp'
KRNH = 'ggml/src/ggml-et/ggml-et-kernels.h'
MEMC = 'ggml/src/ggml-et/ggml-et-memops.cpp'
CMPH = 'ggml/src/ggml-et/ggml-et-cpu-compare.h'
COMH = 'ggml/src/ggml-et/ggml-et-common.h'
UKH = 'ggml/src/ggml-et/ggml-et-uberkernel-common.h'
API = 'ggml/include/ggml-et.h'
K = 'ggml/src/ggml-et/et-kernels/src/'

# ---------------------------------------------------------------------------
# 覆盖声明：L7-04 的确切清单（tools/plan_matrix.py --files）
#   1 个公共 API 头 + 11 个宿主侧文件 + 6 个设备侧共享头 + 50 个设备内核 .c = 68
# ---------------------------------------------------------------------------

API_FILES = [
    'ggml/include/ggml-et.h',
]

HOST_FILES = [
    'ggml/src/ggml-et/ggml-et-common.h',
    'ggml/src/ggml-et/ggml-et-cpu-compare.cpp',
    'ggml/src/ggml-et/ggml-et-cpu-compare.h',
    'ggml/src/ggml-et/ggml-et-kernels.cpp',
    'ggml/src/ggml-et/ggml-et-kernels.h',
    'ggml/src/ggml-et/ggml-et-memops.cpp',
    'ggml/src/ggml-et/ggml-et-memops.h',
    'ggml/src/ggml-et/ggml-et-ops.cpp',
    'ggml/src/ggml-et/ggml-et-ops.h',
    'ggml/src/ggml-et/ggml-et-uberkernel-common.h',
    'ggml/src/ggml-et/ggml-et.cpp',
]

DEV_HDRS = [
    'ggml/src/ggml-et/et-kernels/src/block_ops.h',
    'ggml/src/ggml-et/et-kernels/src/ggml_tensor.h',
    'ggml/src/ggml-et/et-kernels/src/math_fp.h',
    'ggml/src/ggml-et/et-kernels/src/platform.h',
    'ggml/src/ggml-et/et-kernels/src/quants.h',
    'ggml/src/ggml-et/et-kernels/src/tensor.h',
]

DEV_KERNELS = [
    'ggml/src/ggml-et/et-kernels/src/clamp_f32.c',
    'ggml/src/ggml-et/et-kernels/src/concat_f32.c',
    'ggml/src/ggml-et/et-kernels/src/cont_f16.c',
    'ggml/src/ggml-et/et-kernels/src/cont_f32.c',
    'ggml/src/ggml-et/et-kernels/src/conv_2d_f32_me.c',
    'ggml/src/ggml-et/et-kernels/src/cpy_f32_f16.c',
    'ggml/src/ggml-et/et-kernels/src/cumsum_f32.c',
    'ggml/src/ggml-et/et-kernels/src/diag_f32.c',
    'ggml/src/ggml-et/et-kernels/src/el_map_f32.c',
    'ggml/src/ggml-et/et-kernels/src/fill_f32.c',
    'ggml/src/ggml-et/et-kernels/src/flash_attn_ext_f16_me.c',
    'ggml/src/ggml-et/et-kernels/src/flash_attn_ext_f32.c',
    'ggml/src/ggml-et/et-kernels/src/gated_delta_net_f32.c',
    'ggml/src/ggml-et/et-kernels/src/get_rows_f32.c',
    'ggml/src/ggml-et/et-kernels/src/glu_f32.c',
    'ggml/src/ggml-et/et-kernels/src/group_norm_f32.c',
    'ggml/src/ggml-et/et-kernels/src/im2col.c',
    'ggml/src/ggml-et/et-kernels/src/l2_norm_f32.c',
    'ggml/src/ggml-et/et-kernels/src/mean_f32.c',
    'ggml/src/ggml-et/et-kernels/src/memops.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_Q4_0.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_Q4_0_matrix_engine.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_Q8_0.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_f16.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_f16_matrix_engine.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_f32.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_f32_matrix_engine.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_id_Q4_0.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_id_Q8_0.c',
    'ggml/src/ggml-et/et-kernels/src/mul_mat_id_f32.c',
    'ggml/src/ggml-et/et-kernels/src/norm_f32.c',
    'ggml/src/ggml-et/et-kernels/src/pad_f32.c',
    'ggml/src/ggml-et/et-kernels/src/repeat_f32.c',
    'ggml/src/ggml-et/et-kernels/src/rms_norm_f32.c',
    'ggml/src/ggml-et/et-kernels/src/rms_norm_mul_f32.c',
    'ggml/src/ggml-et/et-kernels/src/rope_f32.c',
    'ggml/src/ggml-et/et-kernels/src/rwkv_wkv6_f32.c',
    'ggml/src/ggml-et/et-kernels/src/rwkv_wkv7_f32.c',
    'ggml/src/ggml-et/et-kernels/src/scale_f32.c',
    'ggml/src/ggml-et/et-kernels/src/set_f32.c',
    'ggml/src/ggml-et/et-kernels/src/set_rows_f32.c',
    'ggml/src/ggml-et/et-kernels/src/softmax_f32.c',
    'ggml/src/ggml-et/et-kernels/src/solve_tri_f32.c',
    'ggml/src/ggml-et/et-kernels/src/sqr_f32.c',
    'ggml/src/ggml-et/et-kernels/src/ssm_conv_f32.c',
    'ggml/src/ggml-et/et-kernels/src/ssm_scan_f32.c',
    'ggml/src/ggml-et/et-kernels/src/sum_rows_f32.c',
    'ggml/src/ggml-et/et-kernels/src/tri_f32.c',
    'ggml/src/ggml-et/et-kernels/src/uberkernel.c',
    'ggml/src/ggml-et/et-kernels/src/unary_f32.c',
]

ALL_FILES = API_FILES + HOST_FILES + DEV_HDRS + DEV_KERNELS

L = Lesson(
    id='L7-04',
    layer='L7 · NPU 与加速器后端',
    title='ExecuTorch 后端（边缘/移动端）：离线编译的设备内核与运行期派发',
    codecap='ggml-et.cpp / ggml-et-ops.cpp / ggml-et-kernels.cpp / ggml-et-common.h / '
            'ggml/include/ggml-et.h / et-kernels/src/*.c（逐字引用）',
    nav={'prev': {'href': '../L7-03-openvino-intel-npu/index.html',
                  'label': 'L7-03 OpenVINO 后端'},
         'next': {'href': '../L7-05-virtgpu-virtualized/index.html',
                  'label': 'L7-05 VirtGPU 后端'}},
)

L.cover(*ALL_FILES)

L.note('**一句话**：这个后端（`GGML_ET_NAME "ET"`）把 ggml 的算子交给 **ET-SoC** —— 一颗多核 '
       'RISC-V 加速器上的**预编译裸机内核**。它的"编译"**不发生在图里**：内核在**构建期**用 '
       'RISC-V 工具链交叉编译成 `*.elf`、嵌进宿主库，运行期只做"按名字取代码 -> 加载 -> 派发"；'
       '`graph_compute` 本身就是**一个 `for` 循环套一个 `switch`**，没有任何图级编译步骤。')
L.note('这条路线与 L7-01（CANN 把 op 映射到 ACL 算子）、L7-03（OpenVINO 把子图翻成 ov::Model）'
       '的"**运行期映射**"不同：那两家在运行期把图翻译给厂商栈，ET 把翻译**提前到了构建期**，'
       '运行期只剩执行。')
L.note('**验收点**：看完这一课，你要能说出它与 GPU 后端在**数据面抽象**上的不同 —— '
       '关键一条是：跨 host/device 边界的不是"裸指针 + 形状 + 元素数"，而是**整个 '
       '`struct ggml_tensor` 按值**（`ggml-et-ops.cpp:282`），设备内核直接读它的 '
       '`ne[]` / `nb[]` / `type` / `data`（`scale_f32.c:32-40`）。')
L.note('**关于课名的说明（实测修正）**：计划里这一课叫"ExecuTorch 后端"，讲解要点写的是'
       '"ExecuTorch 的委托机制、图导出与运行时"。在本仓库 v0.5.0 的 `ggml/src/ggml-et/` 里'
       '**搜不到** `executorch` / `torch` / `delegate` / `.pte` 任何一个符号：它用的是 ET 平台 SDK 的 '
       '`dev::IDeviceLayer` + `rt::IRuntime`（`ggml-et-common.h:6-8`；`ggml/src/ggml-et/CMakeLists.txt:239` 链接 '
       '`runtime::etrt_static deviceLayer::deviceLayer`），机制是"**预编译内核 + 运行期加载**"，'
       '不是"导出 `.pte` 再委托"。本课按代码实际机制写，并把这条差异显式讲出来。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L7 · NPU 与加速器后端',
    title='ET 后端：<span class="hl-a">68 个文件</span>分成两个世界',
    sub='宿主侧 11 个文件说"怎么派发"，设备侧 56 个文件说"内核怎么写、怎么被加载"。',
    caption='文件清单来自 tools/plan_matrix.py --files 里 L7-04 的 68 项，全部进本课覆盖声明。',
    src=API, parts=[(10, 24)], duration=20000,
    mark_src=[10, 14, 17, 21, 24],
    notes_src={10: '后端名字：注册表（L3-02）与 buft 名字都用它',
               14: '唯一的建后端入口：拿一个 devidx，返回 ggml_backend_t',
               17: '设备枚举：几个 ET 设备（runtime->getDevices()）',
               21: '设备缓冲类型：本课"数据面"的关键 —— 设备内存',
               24: '注册入口：GGML_BACKEND_DL_IMPL 动态加载时找的就是它（L3-03）'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L7-04 · 运行期',
    title='运行期没有编译：一张图 = 一个 <span class="hl-c">for</span> + 一个 <span class="hl-c">switch</span>',
    sub='graph_compute 逐节点分派，只在两处做图级融合（RMS_NORM+MUL、MUL_MAT+ADD）。',
    caption='★ 对照 L7-03：OpenVINO 在运行期把子图翻成 ov::Tensor / ov::op 组成的模型；ET 这里没有任何"编译"调用。',
    src=ET, parts=[(657, 680)], duration=22000,
    mark_src=[659, 661, 664, 670, 675],
    notes_src={659: 'uberkernel：本图要打包成"一个设备内核"时，先开一段',
               661: '遍历拓扑序（L1-03）里的每个节点；整张图的执行就是这个循环',
               664: 'VIEW / RESHAPE / PERMUTE / TRANSPOSE 不产生内核调用，直接跳过',
               670: '唯一的图级融合：RMS_NORM + MUL 合成一个内核 rms_norm_mul_f32',
               675: '第二个融合：MUL_MAT + ADD（bias）合成一次 mul_mat 调用'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L7-04 · ★ 核心洞察',
    title='★ "编译"发生在<span class="hl-a">图之外</span>：内核是预编译的 ELF，运行期按名字取',
    sub='这段代码里没有任何编译器：只有"名字 -> 字节 -> runtime->loadCode() -> KernelId"。',
    caption='★ 与 CANN / OpenVINO 的"运行期映射"是两条路线：ET 把翻译提前到构建期（本幕末说明构建期的做法）。',
    src=KRN, parts=[(133, 183)], duration=26000,
    mark_src=[141, 147, 151, 163, 172, 176],
    notes_src={141: '已经加载过就直接返回 —— 名字就是缓存键',
               147: '先看环境变量 GGML_ET_KERNELS_PATH（开发期用文件覆盖内置内核）',
               151: '★ 名字拼出来的是 .elf —— 一个已经编译好的设备可执行文件，不是一个"算子描述"',
               163: '没给路径（或文件不存在）就退回**嵌进宿主库的那份** ELF 字节',
               172: '把这段 ELF 字节交给运行时；由它下载到设备并返回句柄',
               176: '句柄按名字存进 map，下次 ggml_et_launch_kernel 直接命中'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L7-04 · 谁在派发',
    title='真正的"运行时"是 SDK 的两个对象，不是 ggml',
    sub='ggml 侧只持有两个 shared_ptr：设备层 + 运行时；流、内核句柄、事件都挂在设备上下文里。',
    caption='跨课呼应 L3-01：ggml 的三张虚表是"契约"；这一层之下，ET SDK 自己还有一套对象模型。',
    src=ET, parts=[(70, 77)], duration=18000,
    mark_src=[71, 72, 75],
    notes_src={71: '设备层：屏蔽"真硬件 PCIe 卡"与"sysemu 模拟器"的差别',
               72: '运行时：建流、分配设备内存、传数据、加载代码、启动内核',
               75: 'profiling 时把"名字 -> KernelId"导出成 JSON（kernel_id.json）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div>
  <div class="row center" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['对象', '类型', '本课证据'],
  [['_drv.device_layer', 'dev::IDeviceLayer', 'createPcieDeviceLayer / createSysEmuDeviceLayer（152-158）'],
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
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L7-04 · ★ 数据面',
    title='★ 跨边界的不是裸指针，是<span class="hl-a">整个 struct ggml_tensor</span>',
    sub='宿主侧把张量结构体按值拷进参数块；设备侧于是拿到同一套 ne/nb/type/data 词汇表。',
    caption='★ 这就是验收点说的"数据面抽象的不同"：L6-01 的 CUDA 内核只收到裸指针 + 维度。',
    src=OPSC, parts=[(277, 287)], duration=20000,
    mark_src=[282, 283, 287],
    notes_src={282: '★ 不是取 src0->data，而是把 src0 整个结构体拷进参数（含 data 里的设备地址）',
               283: '★ 输出张量同样整个拷过去：设备内核由此知道 dst->ne / dst->nb',
               287: '参数块整体交给 ggml_et_launch_kernel；名字 "scale_f32" 在设备侧对应一个 ELF'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L7-04 · 设备侧',
    title='设备内核：同一个结构体，<span class="hl-c">裸机</span>环境里再声明一遍',
    sub='entry_point(params, env)：参数是宿主打包好的那块内存，env 告诉它有多少个 hart 可用。',
    caption='设备侧没有 malloc、没有 ggml 运行时、没有 libc：参数就是唯一输入，线程数由 shire_mask 推出。',
    src=K + 'scale_f32.c', parts=[(11, 45)], duration=22000,
    mark_src=[12, 13, 18, 25, 32, 39],
    notes_src={12: '与宿主 struct ggml_tensor 同名同义：两端各写一遍声明，靠"字段一致"对齐',
               13: '输出张量也是按值传进来的 —— 设备侧由此知道写多少、往哪写',
               18: '内核入口签名固定：int entry_point(<参数结构> *, void * env)',
               25: 'env 里带着 shire_mask：本内核在多少个 hart 上并行，由调度它的那次 launch 决定',
               32: '直接把参数块当成 ggml_tensor 用 —— 不需要任何反序列化',
               39: '★ data 是设备地址：这里解引用就是真的读到显存里的数据'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L7-04 · 数据面契约',
    title='内存契约：设备内存、<span class="hl-c">cache 对齐</span>、没有 host 映射',
    sub='buffer_type 虚表把"这块内存是怎么来的"讲清楚了；supports_op 再把内核的假设写成前置条件。',
    caption='跨课呼应 L4-03：buffer 是"后端对内存的假设"；ET 的假设比 GPU 后端更强。',
    src=ET, parts=[(432, 449)], duration=20000,
    mark_src=[434, 439, 444, 448],
    notes_src={434: '分配大小不是 nbytes，而是补齐到 cache line 的 nbytes_pad',
               439: '★ 明确声明"我的 buffer 不是 host 内存" —— 与 CPU / CUDA pinned 路径区分开',
               444: 'alloc_buffer 走 runtime->mallocDevice（384-411），不是 malloc',
               448: 'is_host 槽位就是上面那个恒 false 的函数'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L7-04 · 验收点',
    title='与 GPU 后端比：<span class="hl-a">数据面抽象</span>差在哪',
    sub='同一套 ggml_backend_i 契约之下，两边"过边界的东西"完全不同。',
    caption='★ 一句话：CUDA 过边界的是"指针 + 形状参数"，ET 过边界的是"ggml 自己的张量结构体"。',
    src=COMH, parts=[(23, 32)], duration=22000,
    mark_src=[25, 27],
    notes_src={25: '★ 注释写明：这是设备内存指针 —— 不是 host 指针，也不是 mmap 的文件页',
               27: '每个 buffer 记住自己的 rt::DeviceId：以后 free / 传输都要用它'},
    visual='''
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
    '厂商自己的 shape / stride 表示（acl_tensor.cpp:87-90）'],
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
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L7-04 · 收束',
    title='把这一课压成一张表',
    sub='运行期只有执行与失败两种结局；所有"翻译"都发生在构建期。',
    caption='下一课 L7-05：VirtGPU —— 客户机与宿主机之间的命令流、共享内存与同步。',
    src=ET, parts=[(838, 856)], duration=22000,
    mark_src=[840, 844, 850],
    notes_src={840: '运行期遇到没实现（或 supports_op 判错）的 op：整图失败，没有兜底编译',
               844: 'uberkernel 打包失败同样整图失败 —— 不留半张图的结果',
               850: '"end_graph" 才是真正把 uberkernel 段发射出去的地方'},
    visual='''
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
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、公共 API：ggml-et.h 的全部内容',
    '整个后端对外只有 8 个函数 + 1 个名字宏。注意最后两行：`ggml_backend_et_buffer_type()` 拿的是'
    '**设备内存**的缓冲类型，`ggml_backend_et_host_buffer_type()` 拿的是一个"host"缓冲类型 —— '
    '但设备的 `get_host_buffer_type` 槽位返回的是 **CPU 的** buffer type（`ggml-et.cpp:1667-1670`），'
    '而 `is_host` 恒为 false（`ggml-et.cpp:437-440`）。读这一课时要盯住这条"ET 没有 host 内存路径"的线索。',
    src=API, parts=[(10, 24)], lang='c')

L.section(
    '二、注册与初始化：guid + iface + device',
    '`ggml_backend_et_init()` 是 L3-02（注册表）与 L3-03（动态加载）的落点：'
    '`GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)`（文件最后一行）让这个后端能被 `dlopen` 加载。\n\n'
    '建后端的动作只有三件事：造一个 `ggml_backend_et_context`（只存 devidx）、'
    '把 guid 与 `ggml_backend_et_i` 填进 `ggml_backend`、把注册表里的 device 挂上。'
    '**没有**编译产物缓存、**没有**算子集协商 —— 那些都发生在图之外。',
    src=ET, parts=[(1803, 1829)], lang='cpp')

L.section(
    '三、缓冲区虚表：ggml_backend_et_buffer_i',
    '11 个槽位里 **4 个是 NULL**（`memset_tensor` / `set_tensor_2d` / `get_tensor_2d` / `reset`）、'
    '1 个恒 false（`cpy_tensor`，见 333-340）。真正干活的是：\n\n'
    '- `free_buffer` -> `runtime->freeDevice()`（242-251）\n'
    '- `get_base` -> buffer context 里的设备指针（253-256）\n'
    '- `init_tensor` -> 只做一件事：把补齐出来的 padding 用**设备侧 memset 内核**清零（258-288）\n'
    '- `set_tensor` / `get_tensor` -> `memcpyHostToDevice` / `memcpyDeviceToHost` + `waitForEvent`（290-331）\n'
    '- `clear` -> 同样是设备侧 memset 内核（342-363）\n\n'
    '这张表就是 L4-03 说的"后端对 buffer 的假设"：**内存只能来自 runtime，进出只能靠显式拷贝**。',
    src=ET, parts=[(365, 377)], lang='cpp')

L.section(
    '四、设备内存从哪来：alloc_buffer',
    '分配路径三步：查 runtime -> 查 `rt::DeviceId` -> `runtime->mallocDevice()`。'
    '拿到的指针存进 buffer context，同时记住 `rtid`，`free_buffer` 时原样还回去。\n\n'
    '注意这里**没有** mmap、没有 host 注册、没有统一地址空间的假设：'
    '`ggml_backend_dev_props.caps` 里 `host_buffer` 与 `buffer_from_host_ptr` 都是 false / NULL'
    '（`ggml-et.cpp:1647-1653`、`1681`）。',
    src=ET, parts=[(384, 411)], lang='cpp')

L.section(
    '五、数据进出：set_tensor / get_tensor',
    '两个方向都是"建一个事件 -> 等事件"。`memcpyHostToDevice` 的第 5 个参数 `true` 是 barrier 语义，'
    '`runtime->waitForEvent()` 立刻同步等待 —— 也就是说这两个函数是**同步**的，'
    '与 L6-01 里 CUDA 用 stream 做异步拷贝的路子不同（ET 的异步留给 `set_tensor_async` 槽位）。\n\n'
    '`ggml_backend_et_synchronize()`（553-571）则在需要时 `waitForStream` 并把流上累积的错误捞出来，'
    '有错直接 `abort()` —— 这也是一种"没有编译期检查，只能运行期兜底"的表现。',
    src=ET, parts=[(290, 331)], lang='cpp')

L.section(
    '六、缓冲类型虚表：对齐与"我不是 host"',
    '五个槽位回答了"这块内存长什么样"：\n\n'
    '- `get_alignment` -> 设备属性里的 `cacheLineSize_`（413-422）；\n'
    '- `get_max_size` -> 设备总内存（424-430）；\n'
    '- `get_alloc_size` -> `ggml_nbytes_pad()`，即**按 cache line 补齐**的字节数（432-435）；\n'
    '- `is_host` -> 恒 false（437-440）。\n\n'
    '`ggml_nbytes_pad` 与第 7 幕的 `ne[0] % 16 == 0` 是同一件事的两面：'
    'ET 的硬件按 64 字节 cache line 搬运，所以"补齐"不是优化，是正确性前提。',
    src=ET, parts=[(413, 449)], lang='cpp')

L.section(
    '七、后端虚表：ggml_backend_et_i',
    '16 个槽位：**6 个实现、10 个 NULL**。这一点与 L5-01 的 CPU 后端数字一样，但含义不同：\n\n'
    '- 没有 `graph_plan_*`：ET 不做"执行计划"（那是 CPU 后端的 work buffer 机制），'
    '  每次 `graph_compute` 现算；\n'
    '- 没有 `event_*`：事件藏在 runtime 内部（`rt::EventId`），不暴露给 ggml；\n'
    '- 没有 `graph_optimize`：**优化发生在构建期**（哪些 op 有内核、内核怎么写），'
    '  不在运行期做图变换。',
    src=ET, parts=[(1598, 1615)], lang='cpp')

L.section(
    '八、设备虚表：ggml_backend_et_device_i',
    '这一层回答"调度器（L4-02）能对这个设备期待什么"：\n\n'
    '- `get_type` 返回 `GGML_BACKEND_DEVICE_TYPE_GPU`（1635-1638）—— 尽管它不是 GPU，'
    '  这样调度器才会把它当"加速器"来切分图；\n'
    '- `get_host_buffer_type` 返回 **CPU 的** buffer type（1667-1670）；\n'
    '- `buffer_from_host_ptr` 是 NULL（1681）—— 明确拒绝"把 host 内存包成设备 buffer"；\n'
    '- `supports_buft` 用**函数指针相等**来判断（1577-1580），即只有 ET 自己的 buffer 才算数；\n'
    '- `offload_op` 对 `GET_ROWS` 返回 false，理由写在注释里：跨后端权重每次都要重拷，'
    '  把 266MB 的 embedding 表按 token 拷到设备不值得（1582-1596）。',
    src=ET, parts=[(1672, 1688)], lang='cpp')

L.section(
    '九、支持性判定：supports_op 是运行期唯一的"守门人"',
    '因为没有图级编译，**"哪些 op 能跑"必须在执行前由 `supports_op` 用 44 个 case 判完**'
    '（`ggml-et.cpp:868-1575`）。头几族已经把内核的硬件假设写成了前置条件：\n\n'
    '- `SQR`：`ne[0] % 16 == 0` 且 dst/src0 都连续（873-876）；\n'
    '- `UNARY`：**只**要求 `nb[0] == sizeof(float)`，注释明说更高维可以有任意步长，'
    '  内核按行用四个 `nb[]` 走（895-900）；\n'
    '- `MUL` / `ADD` / `SUB`：行内连续，且 `nb[1] == ne[0] * sizeof(float)`（930-938）。\n\n'
    '读这里的诀窍：**每一条 `&&` 都是设备内核对内存的一个假设**，与第 6、7 幕的行为一一对应。',
    src=ET, parts=[(864, 900)], lang='cpp')

L.section(
    '十、懒加载：内核第一次用到才下载',
    '`ggml_et_launch_kernel_internal()` 在真正 launch 前查一次 `loaded_kernels`：'
    '没有就 `ggml_et_load_kernel()`，加载完再查一次（防止加载失败静默继续）。\n\n'
    '所以"加载 50 个内核"不是启动成本，而是**按模型实际用到的 op 逐个发生**的：'
    '跑一个小模型可能只碰到十几个内核。',
    src=KRN, parts=[(199, 216)], lang='cpp')

L.section(
    '十一、uberkernel：把一段图打包成"一个设备内核"',
    'ET 的图级融合有两条路：一是宿主侧的算子融合（`RMS_NORM+MUL`、`MUL_MAT+ADD`），'
    '二是 `GGML_ET_UBERKERNEL` 打开的 **uberkernel** —— 把整张图编译成"指令数组 + 参数块"，'
    '一次 launch 交给设备上的派发器。\n\n'
    '这段代码是发射点：先 `memcpyHostToDevice` 两份缓冲（指令数组、参数块），'
    '再把两个**设备地址**填进 `ggml_et_uberkernel_params`，最后 launch 名字叫 `uberkernel` 的那个内核。'
    '注意参数里带的是 `reinterpret_cast<uint64_t>(slot.device_insts)` —— 又一次"裸设备地址过边界"。',
    src=KRN, parts=[(335, 353)], lang='cpp')

L.section(
    '十二、设备侧派发器：uberkernel.c',
    '设备侧的 uberkernel 就是一个 `switch (inst->kernel_id)`：取出第 i 条指令，'
    '按 `params_offset` 在参数块里定位它的参数，转成对应的参数结构体，调用该内核的入口。\n\n'
    '`et_barrier_global(32ULL)` 是**设备侧的全局同步** —— 这正是 `IDeviceLayer` / `IRuntime` 那层'
    '存在的理由：子内核之间没有天然的可见性边界，必须显式同步（`docs/backend/ET.md:161-167` 明确写了'
    '"no natural memory visibility horizon"）。\n\n'
    '对照 L4-02 的调度器：那是在 **host 侧按 buffer 类型切图**；uberkernel 是在 **设备侧按指令序执行**。'
    '两者解决的不是同一个问题。',
    src=K + 'uberkernel.c', parts=[(225, 247)], lang='c')

L.section(
    '十三、连 memset 都是一个设备内核',
    '`ggml_et_memset()` 是宿主侧唯一的"内存操作"入口，它做的事只有一件：'
    '把 `memset_params` 填好，然后 `ggml_et_launch_kernel(dev_ctx, "memops", ...)`。\n\n'
    '为什么不用 host 的 `memset`？因为这块内存是**设备内存**（`runtime->mallocDevice`），'
    'host 指针不能直接写。这也解释了 `init_tensor` 里"清 padding"为什么要绕这么大一圈（258-288）。',
    src=MEMC, parts=[(1, 36)], lang='cpp')

L.section(
    '十四、设备侧的同名参数结构体',
    '设备侧把同一份结构体**再声明一遍**，并用注释提醒"必须与宿主侧一致"。'
    '这就是本课说的"共享表示"的代价：没有共享头文件，靠**约定 + 注释**对齐字段。\n\n'
    '注意 `void * dst_ptr` —— 参数里带的仍然是设备地址。',
    src=K + 'memops.c', parts=[(17, 28)], lang='c')

L.section(
    '十五、设备侧的内存契约与量化块',
    '设备侧也有一个"连续性判定"，但比宿主宽松：`ne[i] == 1` 的轴不参与比较，'
    '因为这些轴的步长不可观测。这与 `supports_op` 里"只要求 `nb[0]`"是同一个思路。\n\n'
    '更有意思的是量化块：`quants.h` 直接 `#define GGML_COMMON_DECL_C` 然后 include '
    '`ggml-common.h` —— 设备内核用的 `block_q8_0` / `block_q4_0` / `block_q4_K` '
    '就是 L1-04 讲的那份定义，**同一个头文件，两端共用**。',
    src=K + 'ggml_tensor.h', parts=[(33, 42)], lang='c')

L.section(
    '十六、设备侧复用量化块定义',
    '同一件事的另一半证据在这里：`GGML_COMMON_DECL_C` + `#include "ggml-common.h"` 之后，'
    '下面三个 `dequantize_*_block` 用的就是 ggml 的块结构。\n\n'
    '这解释了为什么 ET 只支持 `q8_0` / `q4_0`（部分 `q4_K`，见 `docs/backend/ET.md:22`）：'
    '设备侧要**逐个手写**反量化，支持一种格式就多一份裸机代码。',
    src=K + 'quants.h', parts=[(10, 15)], lang='c')

L.section(
    '十七、没有编译器，就靠 CPU 对拍',
    'ET 后端自带一套"逐算子对照"设施：`ggml_et_cpu_compare_ctx` 同时持有 CPU 与 ET 两侧的张量、'
    '图和数据指针，先（`init_pre`）把输入拷到 CPU 侧，等 ET 内核跑完再'
    '（`compute_and_check`）用 CPU 后端算一遍并按容差比较。\n\n'
    '每个 op 家族的配置（从 `ggml-et-ops.cpp:12` 的 rope 到 `2217` 的 gated_delta_net）'
    '都是 `enabled = false`：'
    '默认关闭，调试时打开。**一个没有图级编译器的后端，只能用对拍来定位精度问题** —— '
    '这与 L5-04 里 CPU 侧靠 repack/量化内核的单元测试保证正确性形成对照。',
    src=CMPH, parts=[(24, 46)], lang='c')

L.section(
    '十八、内核加载策略（源码自述）',
    '`ggml-et-kernels.h` 的注释把加载策略写得很清楚，也是本课"编译在图之外"这条结论的直接依据：'
    '先试 `${GGML_ET_KERNELS_PATH}/${kernel_name}.elf`（开发期覆盖），'
    '失败则回退到**嵌进库里的那份**；两者都没有才算失败。\n\n'
    '注意第 24 行的默认路径提示 `/opt/et/ggml/kernels/` —— '
    '同样的 ELF 既能被嵌进二进制，也能放在设备可见的目录里。',
    src=KRNH, parts=[(9, 33)], lang='cpp')

L.section(
    '十九、50 个设备内核按族分类',
    '`et-kernels/CMakeLists.txt:28-79` 的 `KERNELS` 列表**正好 50 项**，与 '
    '`et-kernels/src/*.c` 的 50 个文件一一对应；每个文件用 `add_riscv_executable()` 编译成 '
    '`<名字>.elf`（`et-kernels/CMakeLists.txt:41-43`），编译选项里有 `-nostdlib` / `-ffreestanding` / '
    '`-march=rv64imf`，并且**构建后跑 `check_unimplemented_instructions.sh`**：'
    '一旦编译器生成了硬件没实现的浮点指令就直接让构建失败。\n\n'
    '这 50 个文件按族分成 9 组（下表）；其中 5 个文件名里带 `_me` / `matrix_engine`，'
    '是走 **tensor engine（矩阵引擎）** 指令的版本，与走向量指令的通用版本并存。'
    '`uberkernel.c` 是唯一不实现算子的内核 —— 它是设备侧的派发器（第十二节）。\n\n'
    '| 族 | 文件数 | 行数 | 代表 |\n'
    '|---|---|---|---|\n'
    '| mul_mat 稠密 | 7 | 1902 | `mul_mat_Q8_0.c` / `mul_mat_f32_matrix_engine.c` |\n'
    '| mul_mat_id（MoE） | 3 | 617 | `mul_mat_id_Q4_0.c` |\n'
    '| 归一化 | 5 | 1296 | `rms_norm_f32.c` / `rms_norm_mul_f32.c` |\n'
    '| 注意力与位置 | 4 | 2571 | `flash_attn_ext_f32.c` / `rope_f32.c` / `softmax_f32.c` |\n'
    '| 逐元素 | 6 | 1990 | `unary_f32.c` / `scale_f32.c` / `sqr_f32.c` |\n'
    '| 布局/搬运/归约 | 16 | 2979 | `cont_f32.c` / `get_rows_f32.c` / `set_rows_f32.c` |\n'
    '| 卷积 | 2 | 937 | `conv_2d_f32_me.c` / `im2col.c` |\n'
    '| 循环/SSM | 5 | 1213 | `ssm_scan_f32.c` / `rwkv_wkv7_f32.c` / `gated_delta_net_f32.c` |\n'
    '| 基础设施 | 2 | 678 | `memops.c`（memset）/ `uberkernel.c`（派发器） |\n\n'
    '合计 **50 个文件 / 14183 行**（`wc -l` 实测）。这些内核目前**没有**统一的公共接口头：'
    '每个内核在自己的 `.c` 里定义参数结构体，宿主侧在 `ggml-et-ops.h` 里再写一遍'
    '（第十五、十六节点出了这个约定的两端）。',
    lang='text')

L.footnote_add('本课覆盖声明是 `tools/plan_matrix.py --files` 里 L7-04 的**全部 68 个文件**：'
               '`ggml/include/ggml-et.h`（1）+ `ggml/src/ggml-et/*.{cpp,h}`（11）+ '
               '`ggml/src/ggml-et/et-kernels/src/*.h`（6）+ `et-kernels/src/*.c`（50）。'
               '合计 23191 行。')
L.footnote_add('**正文引用但不计入覆盖率的文件**（不在本视角的源文件后缀/路径里，故未进覆盖声明）：'
               '`ggml/src/ggml-et/CMakeLists.txt`（内核清单与嵌入）、'
               '`ggml/src/ggml-et/et-kernels/CMakeLists.txt`（RISC-V 交叉编译）、'
               '`ggml/src/ggml-et/et-kernels/scripts/check_unimplemented_instructions.sh`、'
               '`docs/backend/ET.md`（上游文档）、`et-kernels/src/RunBackend.sh`、'
               '以及同目录下不在覆盖域的 `crt.S` / `linker.ld`。'
               '另有若干对照引用属于别的课：`ggml/src/ggml-cuda/scale.cu`（L6-01）、'
               '`ggml/src/ggml-cuda/ggml-cuda.cu`（L6-01）、`ggml/src/ggml-cann/acl_tensor.cpp`（L7-01）、'
               '`ggml/src/ggml-openvino/ggml-decoder.cpp`（L7-03）、`ggml/src/ggml-common.h`（L1-04）。')
L.footnote_add('**关于课名与计划口径的修正**：计划文档把这一课记为"ExecuTorch 后端"，'
               '讲解要点是"ExecuTorch 的委托机制、图导出与运行时"。实测（`grep -rni` 全目录）'
               '`ggml/src/ggml-et/` 里没有 `executorch` / `torch` / `delegate` / `.pte` 任何符号；'
               '实际机制是"ET 平台 SDK（`dev::IDeviceLayer` + `rt::IRuntime`）+ 构建期交叉编译的内核 ELF"。'
               '本课按代码实际机制讲解，并在第 3、8 幕显式对照这两种口径。')
L.footnote_add('**未实测的部分**（明确边界）：ET-SoC 是专用硬件，本机没有 `/opt/et` 平台与 RISC-V 工具链，'
               '因此本课**没有**真的构建或运行过这个后端。所有结论都来自**逐字引用的源码**与'
               '**行号可核对的注释**；凡是"注释声称"的地方（如硬件未实现指令、'
               'L2/L1 不连贯）本课都标明出处，未当作已实测的行为。')

L.prereqs('`L7-03`')

L.goal(
    '说出这个后端的"编译"发生在哪个阶段、运行期 `ggml_backend_et_graph_compute` 只做哪几件事；',
    '说出 `rt::IRuntime` / `dev::IDeviceLayer` 各自负责什么，以及"加载内核"这条路径上的 4 个步骤；',
    '说清 ET 与 GPU 后端在数据面抽象上的差别（跨边界传什么、设备侧能看到什么元数据、内存怎么来）；',
    '指出 ET 的 buffer 有哪些硬性假设（cache line 对齐、`nb[0]` 连续、`is_host = false`、`cpy_tensor = false`）；',
    '解释 uberkernel 与宿主侧算子融合的区别，以及它为什么要靠设备侧全局屏障。')

L.conclusion(
    '一句话',
    '**ET 后端的"编译"在图之外**：50 个裸机内核在构建期交叉编译成 ELF 并嵌进宿主库，'
    '运行期只有"按名字取代码 -> `loadCode` -> `kernelLaunch`"；`graph_compute` 是'
    '一个 `for` 套一个 `switch`（39 个 case），没有任何图级编译或代码生成。')

L.conclusion(
    '与 CANN / OpenVINO 的路线差别',
    'L7-01 的 CANN 与 L7-03 的 OpenVINO 都是**运行期映射**：前者把 ggml op 变成 ACL 描述符'
    '（`aclCreateTensor`，`ggml-cann/acl_tensor.cpp:89`），后者把 ggml 子图翻成 `ov::Tensor` / '
    '`ov::op` 组成的模型（`ggml-openvino/ggml-decoder.cpp:1392`）再交给 OpenVINO 运行时（L7-03 展开）。'
    'ET 把"翻译"提前到构建期，运行期只做执行 —— 代价是**算子集固定**：'
    '没有内核的 op 只能在 `supports_op` 阶段被拒（44 个 case），回退由调度器（L4-02）完成。')

L.conclusion(
    '数据面抽象：共享表示 + 独立内存',
    '**共享表示**：跨边界的是整个 `struct ggml_tensor`（`ggml-et-ops.cpp:282-283`），'
    '设备内核直接读 `ne[]` / `nb[]` / `type` / `data`（`et-kernels/src/scale_f32.c:32-40`），'
    '连量化块的 `block_q8_0` 都复用 `ggml-common.h`（`et-kernels/src/quants.h:10-11`）。\n\n'
    '**独立内存**：设备内存只能由 `runtime->mallocDevice` 给（`ggml-et.cpp:404`），'
    '`is_host = false`（439）、`host_buffer = false`（1649）、`buffer_from_host_ptr = NULL`（1681）、'
    '`cpy_tensor` 恒 false（333-340）。这与 CUDA 的 pinned host buffer（`ggml-cuda.cu:1309-1317`）'
    '和 `cpy_tensor_async`（2485）正好相反。')

if __name__ == '__main__':
    L.build()



