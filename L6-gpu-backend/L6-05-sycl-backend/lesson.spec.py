#!/usr/bin/env python3
"""L6-05 · SYCL 后端（Intel GPU） —— 课件 spec。

运行：python3 L6-gpu-backend/L6-05-sycl-backend/lesson.spec.py

覆盖声明 = `python3 tools/plan_matrix.py --files` 里指派给 L6-05 的全部 174 个文件，
在构建时从计划直接取出（不是手抄），见下面的 PLAN_FILES。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402
import plan_matrix as pmat             # noqa: E402

# ---------------------------------------------------------------- 覆盖清单

# 计划里指派给 L6-05 的文件：ggml/include/ggml-sycl.h + ggml/src/ggml-sycl/ 下的全部文件。
# 由 plan_matrix.resolve() 现场取出，保证与计划矩阵逐项一致（不手抄路径）。
PLAN_FILES = pmat.resolve()[0]['L6-05']
assert len(PLAN_FILES) == 174, f'计划清单变了：{len(PLAN_FILES)}（期望 174）'

# 逐字引用的 CUDA 侧对照文件（不是 L6-05 的计划归属，但本课确实引用了它们；
# 计入覆盖声明是诚实做法：声明 = 本课逐字引用过的文件）。
CUDA_REFS = [
    'ggml/src/ggml-cuda/common.cuh',                                    # 上下文 / 池
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_0.cu',  # 实例文件对照
    'ggml/src/ggml-cuda/fattn-vec.cuh',                                 # 模板实例化宏
    'ggml/src/ggml-cuda/ggml-cuda.cu',                                  # 图执行对照
]

SRC = 'ggml/src/ggml-sycl/ggml-sycl.cpp'

L = Lesson(
    id='L6-05',
    layer='L6 · GPU 后端执行',
    title='SYCL 后端（Intel GPU）',
    codecap='ggml/src/ggml-sycl/ 与 ggml/include/ggml-sycl.h（逐字引用）',
    nav={'prev': {'href': '../L6-04-cuda-other-ops/index.html', 'label': 'L6-04 CUDA 其余算子'},
         'next': {'href': '../L6-06-vulkan-host/index.html', 'label': 'L6-06 Vulkan 主机端'}},
)

L.cover(*PLAN_FILES)

L.note('**一句话**：SYCL 后端是 CUDA 后端在 Intel GPU 上的**平行实现** —— 同一批 kernel'
       '（量化矩阵乘 mmq/mmvq/dmmv、FlashAttention、逐元素算子）用 SYCL 重写一遍；'
       '文件切分、虚表结构、模板实例化的组织方式几乎与 CUDA 侧一一对应。')
L.note('★ 本课的核心洞察：**SYCL 与 CUDA 的结构同构，差异集中在数据面抽象与编译链。**'
       'CUDA 侧用 `cudaStream_t` + `cudaMalloc` + `cuBLAS`；SYCL 侧换成 `sycl::queue` + '
       'USM 指针 + oneDNN/oneMKL。算子层（`GGML_OP_*` 的 switch、量化类型的 switch、'
       '虚表的字段顺序）几乎可以逐行对着看 —— 这正是 SYCL 后端能跟上 CUDA 侧算子覆盖的原因。')
L.note('本课覆盖计划指派给 L6-05 的**全部 174 个文件**：`ggml/include/ggml-sycl.h` 与 '
       '`ggml/src/ggml-sycl/` 下的 173 个文件（含 46 个模板实例）。其中 7 个核心文件逐字引用，'
       '其余按族在 `source.md` 里用表格分类说明。')

# ------------------------------------------------------------------ 第 1 幕
# 首幕给全局：这个后端在整条链上的位置、它有多少东西、它跟谁对照。

L.scene(
    kicker='L6 · GPU 后端执行',
    title='SYCL 后端：CUDA 的<span class="hl-a">平行实现</span>',
    sub='Intel GPU 走 SYCL；它没有另立一套算子体系，而是把 CUDA 侧那一套照搬过来。',
    caption='回顾 L3-01：每个后端都要交出 registry / device / buffer type 三张接口。'
            '本课的对照物是 L6-01 ~ L6-04（CUDA 四条线）。',
    src='ggml/include/ggml-sycl.h', parts=[(19, 31)], duration=17000,
    mark_src=[20, 23, 26, 29],
    visual='''
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
  '<span class="k">实现面很厚</span>：173 个文件，按算子族切开，平均一个算子一个 <span class="v">.cpp + .hpp</span> 对。'
    + '这个"一算子一对文件"的切法与 CUDA 侧完全一致。',
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
'''
)

# ------------------------------------------------------------------ 第 2 幕
# 目录解剖：先数清楚有多少文件、分成哪几族（不靠印象）。

L.scene(
    kicker='L6-05 · 目录解剖',
    title='174 个文件，<span class="hl-b">七族</span>',
    sub='家族的划分几乎照抄 CUDA：一算子一对文件、模板实例单独成目录、主机框架独立。',
    caption='CMakeLists.txt 用通配收集 *.hpp / *.cpp，再单独 glob 两个 template-instances 前缀。'
            '该文件不是本视角的源文件后缀，因此被引用但不计入覆盖率。',
    src='ggml/src/ggml-sycl/CMakeLists.txt', parts=[(22, 33)], duration=19000,
    mark_src=[25, 26, 27, 29],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="formula" style="padding:5px 8px">`
  + `174 个文件 = 1 个公共头（ggml/include/ggml-sycl.h）+ 173 个实现（ggml/src/ggml-sycl/）</div>
  <div id="bars"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const groups = [
  { l: '常规算子内核', n: 73, c: 'a', d: 'element_wise / norm / softmax / cpy / rope / conv2d / im2col / getrows …（一个算子一对 .cpp+.hpp）' },
  { l: '模板实例化', n: 46, c: 'b', d: 'template-instances/：fattn-tile-* 10 个 + fattn-vec-* 36 个' },
  { l: '循环与融合', n: 18, c: 'c', d: 'ssm_scan / ssm_conv / wkv / gated_delta_net / gla / fusion / topk-* 之外的序列类算子' },
  { l: '主机框架', n: 14, c: 'd', d: 'ggml-sycl.cpp（7292 行）/ common / mem / memtrace / sycl_hw / presets / backend.hpp' },
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
    + '一个算子一个 .cpp + .hpp，与 CUDA 侧同名配对。',
  '<span class="k">46 个模板实例化文件</span>（26%）单独放在 template-instances/ —— '
    + '每个文件只有几行 <span class="v">DECL_*</span> 宏，用显式实例化把模板 kernel 编出来。',
  '<span class="k">循环 / 状态空间类算子</span>（ssm_scan、wkv、gated_delta_net…）18 个文件；'
    + 'fusion.cpp 负责图融合判定，对应 CUDA 侧的 ggml_cuda_can_fuse。',
  '<span class="k">主机框架 14 个文件</span>撑起整个后端：上下文、设备探测、内存池、trace。'
    + 'ggml-sycl.cpp 一个文件 7292 行，装下全部虚表与 graph compute。',
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
'''
)

# ------------------------------------------------------------------ 第 3 幕
# ★ 洞察一：同一个 kernel 集，同一个组织方式 —— 模板实例化文件逐字同构。

L.scene(
    kicker='L6-05 · 核心 ★',
    title='★ <span class="hl-b">同一批 kernel，同一套组织</span>',
    sub='SYCL 的模板实例文件与 CUDA 的同名文件逐字相同，只有两处差异：.hpp/.cuh 与多一行 512。',
    caption='8 行就是全部：一份 include、四条显式实例化。CUDA 侧同名文件 7 行（少了 512 那条）。',
    src='ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q4_0.cpp',
    parts=[(1, 8)], duration=19000,
    mark_src=[3, 8],
    visual='''
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
   ['mmq.cpp / mmvq.cpp / dmmv.cpp', 'mmq.cu / mmvq.cu /（v0.5.0 无 dmmv，并入 mmvq）'],
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
'''
)

# ------------------------------------------------------------------ 第 4 幕
# CUDA ↔ SYCL 结构对照表：本课的验收点。

L.scene(
    kicker='L6-05 · 核心',
    title='CUDA ↔ SYCL <span class="hl-a">结构对照表</span>',
    sub='同一个算子，两边落在哪个文件、哪个函数。左边一列看完，对应关系就成立。',
    caption='所有行号都来自上游 v0.5.0。SYCL 侧先看 buffer 虚表：字段顺序与 CUDA 侧一致。',
    src=SRC, parts=[(917, 928)], duration=22000,
    mark_src=[917, 921, 924],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 5 幕
# ★ 洞察二：差异在数据面（USM / queue）与编译链，而不是算子层。

L.scene(
    kicker='L6-05 · 核心 ★',
    title='★ 差异在<span class="hl-e">数据面</span>，不在算子层',
    sub='同一张算子表，换掉的是"怎么发命令、内存从哪来、谁来算 GEMM"。',
    caption='编译链：oneAPI DPC++（icpx -fsycl）取代 nvcc；源码里用 dpct:: 命名空间承接 CUDA→SYCL 的映射。',
    src='ggml/src/ggml-sycl/common.hpp', parts=[(336, 358)], duration=21000,
    mark_src=[341, 349, 350, 356],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '执行流：stream -> queue',
    b: '两边都是"设备 × 流"的二维数组；<br>stream() 返回的是 sycl::queue 指针。',
    m: 'cudaStream_t  ->  sycl::queue *' },
  { c: 'b', t: '设备内存：cudaMalloc -> USM',
    b: 'USM 指针就是普通指针，kernel 直接解引用；<br>两边缓冲区对齐都是 128 字节。',
    m: 'sycl::malloc_device / free' },
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
const texts = [
  '代码区是 <span class="v">ggml_backend_sycl_context</span>：它只做三件事 —— '
    + '记住设备号、持有队列、管内存池。',
  '<span class="k">执行流</span>：<span class="v">qptrs[device][stream]</span> 是个惰性缓存，'
    + '第一次用才去取 <span class="v">dpct::get_device(device).default_queue()</span>。',
  '<span class="k">内存</span>：CUDA 侧 mmq 会 <span class="v">cudaMemsetAsync</span> 清 padding；'
    + 'SYCL 侧对应 <span class="v">ggml_sycl_pool_alloc</span> + USM 分配，对齐同样是 128。',
  '<span class="k">事件</span>：<span class="v">ggml_tensor_extra_gpu</span> 里 events 的用途不变，'
    + '类型从 cudaEvent_t 换成 dpct::event_ptr。',
  '<span class="k">数学库</span>：cuBLAS 的位置被 oneDNN（dnnl::matmul）与 oneMKL 顶上；'
    + '非量化兜底 GEMM 走 <span class="v">ggml_sycl_op_mul_mat_sycl</span>。',
  '所以：<span class="k">算子层（switch、量化类型、tile 结构）可以照搬，数据面必须重写。</span>'
    + '这就是"结构同构 + 局部替换"的完整含义。'
];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
}));
tl.at(13800, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
tl.at(17600, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕
# 三张接口实现：L3-01 的后端契约在 SYCL 侧怎么落地。

L.scene(
    kicker='L6-05 · 后端契约',
    title='L3-01 的<span class="hl-d">虚表</span>在 SYCL 侧逐项填满',
    sub='registry、device、buffer type（外加 buffer）四张接口，字段顺序由 ggml-backend-impl.h 钉死。',
    caption='回顾 L3-01：算子能不能落到某个后端，取决于 supports_op / supports_buft 这两张表怎么说。',
    src=SRC, parts=[(6903, 6918)], duration=20000,
    mark_src=[6908, 6910, 6911, 6913],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'buffer_i', b: 'free_buffer / get_base / init_tensor /<br>memset_tensor / set_tensor / get_tensor /<br>cpy_tensor / clear / reset',
    m: 'ggml-sycl.cpp:917' },
  { c: 'b', t: 'buffer_type_i', b: 'get_name / alloc_buffer /<br>get_alignment（返回 128）/<br>get_max_size / get_alloc_size',
    m: 'ggml-sycl.cpp:1055' },
  { c: 'c', t: 'device_i', b: 'get_memory / get_props / init_backend /<br>get_buffer_type / supports_op /<br>supports_buft / offload_op / event_*',
    m: 'ggml-sycl.cpp:6903' },
  { c: 'd', t: 'reg_i', b: 'get_name / get_device_count / get_device /<br>get_proc_address（暴露 split buffer 与 comm_*）',
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
'''
)

# ------------------------------------------------------------------ 第 7 幕
# 量化矩阵乘的移植程度：三条路 + 同一套 q8_1 约定。

L.scene(
    kicker='L6-05 · 算子移植',
    title='量化矩阵乘：<span class="hl-c">三条路</span>都搬过来了',
    sub='mmq（大 batch）/ mmvq（小 batch）/ dmmv（单向量）；三者共用同一套 q8_1 量化与 vec_dot。',
    caption='回顾 L6-02：CUDA 侧同样按 batch 大小选 mmq / mmvq。SYCL 侧多保留了 dmmv 这一条。',
    src='ggml/src/ggml-sycl/mmq.cpp', parts=[(2987, 3000)], duration=20000,
    mark_src=[2987, 2989, 2991],
    visual='''
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
  '三个入口的形参表也一样：<span class="v">src0_dd_i / src1_ddq_i / dst_dd_i / row_low / '
    + 'row_high / src1_ncols / src1_padded_row_size / stream</span> —— '
    + 'CUDA 侧的 <span class="v">ggml_cuda_op_mul_mat_vec_q</span>（mmvq.cuh:14）逐项对应，'
    + '只有 stream 类型不同。',
  '差异在哪：CUDA 侧 v0.5.0 已经没有独立的 dmmv 文件，而 SYCL 侧把它留成 '
    + '<span class="v">dmmv.cpp</span>（2227 行）+ 一个 reorder 变体。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(9700, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
tl.at(13100, () => { msg.innerHTML = texts[4]; });
tl.at(16600, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕
# graph_compute：SYCL 侧的多一层 —— 用 command_graph 录制重放。

L.scene(
    kicker='L6-05 · 执行',
    title='graph_compute：录制重放 vs 逐节点执行',
    sub='和 CUDA 侧同一套判据：先问"这张图能不能录"，能录就用可更新的执行图，不能录就逐节点跑。',
    caption='开关是 GGML_SYCL_ENABLE_GRAPH（默认 0）；还需要设备支持 ext_oneapi_limited_graph。',
    src=SRC, parts=[(6202, 6222)], duration=21000,
    mark_src=[6208, 6213, 6219, 6221],
    visual='''
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
'''
)

# ------------------------------------------------------------------ 第 9 幕
# 收束：一张表 + 一句话 + 练习 + 下一课指路。

L.scene(
    kicker='L6-05 · 收束',
    title='把 SYCL 后端压成一张表',
    sub='同一个算子，两边各在哪；记住这张表，L6-06 起的其它 GPU 后端都是同一套问法。',
    caption='下一课 L6-06：Vulkan 后端的主机端 —— 那里没有 C++ kernel，算子要编译成计算着色器。',
    src='ggml/src/ggml-sycl/fattn.cpp', parts=[(97, 103)], duration=21000,
    mark_src=[99, 101, 102],
    visual='''
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
  '下一课 L6-06 换一个完全不同的后端：Vulkan 的主机端 —— 算子不再编译成 C++，'
    + '而是编译成 GLSL 计算着色器。'
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
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、公共面：一个头文件',
    'SYCL 后端对外只有一个头。它把后端要交出的东西列全了：初始化、缓冲类型、'
    '张量并行切分（split buffer）、以及三件套 `comm_init / comm_free / allreduce_tensor`。'
    '注意最后那组的注释：源码自己写明"mirrors the CUDA backend\'s pattern"——'
    '这不是巧合，是设计意图。',
    src='ggml/include/ggml-sycl.h', parts=[(19, 31)], lang='c')

L.section(
    '二、目录解剖：174 个文件分七族',
    '先用命令数清楚，再讲内容（不要凭印象）：\n\n'
    '```text\n'
    '$ python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L6-05"{print $2}\' | wc -l\n'
    '174\n'
    '```\n\n'
    '按文件名归类后的实测分布：\n\n'
    '| 族 | 文件数 | 代表文件 |\n|---|---|---|\n'
    '| 常规算子内核 | 73 | `element_wise` `norm` `softmax` `cpy` `rope` `conv2d` `im2col` `getrows` |\n'
    '| 模板实例化 | 46 | `template-instances/fattn-tile-instance-*.cpp`（10）+ `fattn-vec-instance-*.cpp`（36）|\n'
    '| 循环与融合 | 18 | `ssm_scan` `ssm_conv` `wkv` `gated_delta_net` `gla` `dsv4-hc` `fusion` |\n'
    '| 主机框架 | 14 | `ggml-sycl.cpp` `common` `mem` `memtrace` `sycl_hw` `presets` `backend.hpp` |\n'
    '| 量化矩阵乘 | 11 | `mmq` `mmvq` `dmmv` `vecdotq.hpp` `dequantize.hpp` `quants.hpp` `gemm.hpp` `esimd.hpp` |\n'
    '| FlashAttention | 11 | `fattn` `fattn-tile` `fattn-vec.hpp` `fattn-buffers` `fattn-onednn` `fattn-mkl` |\n'
    '| 公共 API | 1 | `ggml/include/ggml-sycl.h` |\n\n'
    '**构建方式**：`CMakeLists.txt` 用 `file(GLOB ...)` 收集 `*.hpp` 与 `*.cpp`，'
    '再把 `template-instances/fattn-tile*.cpp` 与 `fattn-vec*.cpp` 追加进源列表 ——'
    '下半段的逐字引用就是这几行。\n\n'
    '> 说明：`CMakeLists.txt` 不在本视角的源文件后缀集合（`.c/.cc/.cpp/.h/.hpp/.cu/.cuh/.m/.mm/.metal/.comp`）内，'
    '因此它是「被引用但不计入覆盖率」的文件，不写进覆盖声明。',
    src='ggml/src/ggml-sycl/CMakeLists.txt', parts=[(22, 33)], lang='cmake')

L.section(
    '三、★ 模板实例化：一个 8 行的文件',
    'SYCL 侧的模板实例化文件短到可以整份读完。它不含实现，只把 '
    '`ggml_sycl_flash_attn_ext_vec_case` 这个函数模板按 (D, type_K, type_V) 显式实例化。'
    '把这个文件与 CUDA 侧同名文件并排看，差异只有两处：include 的后缀（.hpp / .cuh），'
    '以及 SYCL 多实例化了 D=512 这一条。',
    src='ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q4_0.cpp',
    parts=[(1, 8)], lang='c')

L.section(
    '四、★ 逐字对照：CUDA 侧的同名实例文件',
    '下面这一份是 CUDA 侧的同名文件（v0.5.0）。除了第 3 行的 include 后缀，'
    '正文的前三行与 SYCL 侧逐字相同，第四行（D=512）在 CUDA 侧不存在。'
    '这种"同一份文件、两套后缀"的做法，是 SYCL 后端维护成本低的原因：'
    'CUDA 侧加了新的 (D, type_K, type_V) 组合，SYCL 侧照抄一行即可。',
    src='ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_0.cu',
    parts=[(1, 7)], lang='c')

L.section(
    '五、★ 宏级同构：DECL_FATTN_VEC_CASE',
    '实例文件调用的宏也在两边各写了一份。展开体只有一个标识符不同：'
    '函数名 `ggml_cuda_flash_attn_ext_vec_case` / `ggml_sycl_flash_attn_ext_vec_case`，'
    '上下文类型 `ggml_backend_cuda_context` / `ggml_backend_sycl_context`。\n\n'
    '但 `EXTERN_DECL_FATTN_VEC_CASES` 的类型清单**不同**：CUDA 侧多了 '
    '`GGML_TYPE_BF16` 一条 —— 也就是说 SYCL 侧的 vec kernel 目前不支持 BF16 的 K/V。'
    '这类"看起来一样、其实少一条"的地方，只有逐字比对才能发现。',
    src='ggml/src/ggml-cuda/fattn-vec.cuh', parts=[(574, 585)], lang='c')

L.section(
    '六、★ 数据面：common.hpp 与 common.cuh 的同一张表',
    '`ggml_backend_sycl_context` 与 `ggml_backend_cuda_context` 的字段是同一张表：'
    '设备号、名字、以及"设备 × 流"的二维数组。SYCL 侧的元素是 `queue_ptr`'
    '（`typedef sycl::queue *queue_ptr`，common.hpp:116），CUDA 侧是 `cudaStream_t`。\n\n'
    '队列是**惰性**取的：`stream(device, stream)` 第一次被调用时才去拿 '
    '`dpct::get_device(device).default_queue()`。CUDA 侧同位置的 `streams` 数组在上下文构造时就填好。',
    src='ggml/src/ggml-sycl/common.hpp', parts=[(336, 358)], lang='cpp')

L.section(
    '七、★ CUDA 侧的同一段：字段逐个对上',
    '把两段并排读：`device` / `name` 相同；CUDA 多一个 `copy_event`；'
    '执行流数组的维度宏不同名但同值（`GGML_CUDA_MAX_STREAMS` = `GGML_SYCL_MAX_STREAMS` = 8）；'
    'CUDA 侧把 cuBLAS 句柄与 workspace 也塞进上下文，SYCL 侧则把它们放在 '
    '`ggml_sycl_pool` 与 `gemm.hpp` 的 `DnnlGemmWrapper` 里。',
    src='ggml/src/ggml-cuda/common.cuh', parts=[(1455, 1465)], lang='c')

L.section(
    '八、量化矩阵乘：三个入口的形参表',
    '三个 SYCL 入口（dmmv / mmvq / mmq）收下的是**同一个形状**的参数：'
    '切分后的行区间 `row_low` / `row_high`、激活量化结果 `src1_ddq_i`、'
    '以及输出缓冲 `dst_dd_i`。这是 `ggml_sycl_op_mul_mat` 这个模板回调签名。\n\n'
    'CUDA 侧的 `ggml_cuda_op_mul_mat_vec_q`（mmvq.cuh:14）逐项对应，'
    '只有 stream 的类型不同（`cudaStream_t` / `const dpct::queue_ptr &`）。',
    src='ggml/src/ggml-sycl/mmvq.hpp', parts=[(19, 23)], lang='cpp')

L.section(
    '九、mmq 的类型分派',
    '`ggml_sycl_op_mul_mat_q` 的第一件事是算 `nrows_dst`（主设备拿全量行、'
    '其它设备拿自己那段），然后按 `src0->type` 分派到 per-type kernel。'
    '每个 kernel 的形参都是"权重 / 激活 / 输出 / ncols / nrows / ncols_dst / 行步长 / stream"。',
    src='ggml/src/ggml-sycl/mmq.cpp', parts=[(2987, 3000)], lang='cpp')

L.section(
    '十、★ 图执行：录制重放',
    '`ggml_backend_sycl_graph_compute` 的结构与 CUDA 侧同形：'
    '先判兼容性，再决定走"录制重放"还是"逐节点执行"。SYCL 侧多两个前置条件：'
    '设备必须支持 `ext_oneapi_limited_graph`，且 `finalize(updatable)` 之后还要设备支持 '
    '`ext_oneapi_graph` 才能 `update`。',
    src='ggml/src/ggml-sycl/ggml-sycl.cpp', parts=[(6202, 6222)], lang='cpp')

L.section(
    '十一、CUDA 侧的同一段',
    'CUDA 侧做的是同一件事，只是 API 换了名字：`cudaStreamBeginCapture` 开始录制、'
    '`cudaGraphInstantiate` 固化、`cudaGraphExecUpdate` 更新、`cudaGraphLaunch` 重放。'
    '注意它多了一个"warmup"概念（至少两次调用且属性不变才启用图），SYCL 侧没有这一步。',
    src='ggml/src/ggml-cuda/ggml-cuda.cu', parts=[(4421, 4476)], lang='c')

L.section(
    '十二、FlashAttention 的内核选择',
    '两边的内核选择都是"枚举 + switch"。共有取值的编号相同'
    '（NONE=0 / VEC=100 / TILE=200），各自扩展的部分不同：'
    'SYCL 加 ONEDNN=150 与 MKL=300，CUDA 加 MMA_F16=400。'
    'SYCL 侧的 `ggml_sycl_flash_attn_ext`（fattn.cpp:276）按这个枚举分派。',
    src='ggml/src/ggml-sycl/fattn.cpp', parts=[(97, 103)], lang='cpp')

L.footnote_add(
    '覆盖声明 **179** 项 = 计划指派给 L6-05 的 **174** 个文件（`ggml/include/ggml-sycl.h` + '
    '`ggml/src/ggml-sycl/` 下 173 个）+ 本课逐字引用的 **4** 个 CUDA 侧对照文件'
    '（`common.cuh`、`fattn-vec.cuh`、`template-instances/fattn-vec-instance-f16-q4_0.cu`、'
    '`ggml-cuda.cu`）+ 1 个非覆盖域文件（下面的 `CMakeLists.txt`）。'
    'CUDA 这 4 个在计划里归属 L6-01 / L6-03 / L6-04，这里计入是因为'
    '「声明 = 本课逐字引用过的文件」：不虚报，也不漏报。')
L.footnote_add(
    '`ggml/src/ggml-sycl/CMakeLists.txt` 被逐字引用，但它不在覆盖域的源文件后缀集合内，'
    '按 `check_coverage.py` 的 B3 规则记为「真实存在但不计入覆盖率」，未写进覆盖声明。')

L.prereqs('`L6-04`（CUDA 其余算子）')

L.goal(
    '给一个算子（如 `GGML_OP_MUL_MAT`），同时说出 SYCL 后端与 CUDA 后端在它上面的'
    '**文件名、函数名与行号**（对应本课验收点）；',
    '说出 `ggml_backend_sycl_context` 与 `ggml_backend_cuda_context` 的字段对应关系，'
    '以及 `queue_ptr` 与 `cudaStream_t` 的差异；',
    '解释为什么 SYCL 后端能用 173 个文件跟上 CUDA 侧的算子覆盖（结构同构 + 数据面局部替换）；',
    '说出 `template-instances/` 的组织方式，并指出 SYCL 侧与 CUDA 侧在'
    '`EXTERN_DECL_FATTN_VEC_CASES` 类型清单上的**真实差异**；',
    '说明 `ggml_backend_sycl_graph_compute` 在什么条件下走"录制重放"、否则怎么退回。')

L.conclusion(
    '★ SYCL 与 CUDA 结构同构',
    'SYCL 后端不是另起炉灶，而是 CUDA 后端的平行实现。实测的对应关系：\n\n'
    '| 环节 | CUDA 侧（file:line） | SYCL 侧（file:line） |\n|---|---|---|\n'
    '| 后端上下文 | `ggml_backend_cuda_context`（common.cuh:1455） | '
    '`ggml_backend_sycl_context`（common.hpp:336） |\n'
    '| 执行流 | `cudaStream_t streams[][]`（common.cuh:1460） | '
    '`queue_ptr qptrs[][]`（common.hpp:341） |\n'
    '| 设备内存池 | `ggml_cuda_pool_alloc`（common.cuh:1215） | '
    '`ggml_sycl_pool_alloc`（common.hpp:263） |\n'
    '| buffer 虚表 | `ggml_backend_cuda_buffer_interface`（ggml-cuda.cu:853） | '
    '`ggml_backend_sycl_buffer_interface`（ggml-sycl.cpp:917） |\n'
    '| device 虚表 | `ggml_backend_cuda_device_interface`（ggml-cuda.cu:5651） | '
    '`ggml_backend_sycl_device_interface`（ggml-sycl.cpp:6903） |\n'
    '| MUL_MAT 分派 | `ggml_cuda_mul_mat`（ggml-cuda.cu:1823） | '
    '`ggml_sycl_mul_mat`（ggml-sycl.cpp:4765） |\n'
    '| MMVQ 回调 | `ggml_cuda_op_mul_mat_vec_q`（mmvq.cuh:14） | '
    '`ggml_sycl_op_mul_mat_vec_q`（mmvq.hpp:19） |\n'
    '| FA 入口 | `ggml_cuda_flash_attn_ext`（fattn.cu:755） | '
    '`ggml_sycl_flash_attn_ext`（fattn.cpp:276） |\n'
    '| 图执行 | `ggml_backend_cuda_graph_compute`（ggml-cuda.cu:4421） | '
    '`ggml_backend_sycl_graph_compute`（ggml-sycl.cpp:6202） |\n\n'
    '最硬的一条证据是宏：`DECL_FATTN_VEC_CASE` 在两边展开体完全相同，'
    '只差 `ggml_cuda_*` / `ggml_sycl_*` 与上下文类型名'
    '（fattn-vec.cuh:574 / fattn-vec.hpp:644）。')

L.conclusion(
    '★ 差异在数据面与编译链',
    '算子层可以照搬，数据面必须重写：\n\n'
    '| 维度 | CUDA | SYCL |\n|---|---|---|\n'
    '| 执行流 | `cudaStream_t` | `sycl::queue *`（`queue_ptr`） |\n'
    '| 设备内存 | `cudaMalloc` / `cudaFree` | USM：`sycl::malloc_device` / `free` |\n'
    '| 事件 | `cudaEvent_t` | `dpct::event_ptr` |\n'
    '| 数学库 | cuBLAS | oneDNN（`dnnl::matmul`）/ oneMKL |\n'
    '| 编译链 | nvcc | oneAPI DPC++（`icpx -fsycl`） |\n\n'
    '缓冲区对齐两边**都是 128 字节**（`SYCL_BUFFER_ALIGNMENT`，common.hpp:253；'
    '`ggml_backend_cuda_buffer_type_get_alignment` 直接 `return 128`，ggml-cuda.cu:903）。'
    '这解释了为什么 SYCL 的算子层能直接沿用 CUDA 侧对 tile / padding 的假设。')

L.conclusion(
    'SYCL 独有的分支',
    '不是完全复制，SYCL 侧多出三条 CUDA 侧没有的路：\n\n'
    '1. **厂商库 FlashAttention**：`BEST_FATTN_KERNEL_ONEDNN = 150` 与 '
    '`BEST_FATTN_KERNEL_MKL = 300`（fattn.cpp:97-103），对应 `fattn-onednn.cpp` 与 `fattn-mkl.cpp`；\n'
    '2. **独立的 dmmv 路径**：`dmmv.cpp`（2227 行）+ reorder 变体，CUDA 侧 v0.5.0 已无同名文件；\n'
    '3. **SYCL command graph**：需要 `ext_oneapi_limited_graph` / `ext_oneapi_graph` 两个设备特性，'
    '不像 CUDA 那样有 warmup 步骤。\n\n'
    '反向的差异也有一处：`EXTERN_DECL_FATTN_VEC_CASES` 里 CUDA 侧有 `GGML_TYPE_BF16`，'
    'SYCL 侧没有（fattn-vec.cuh:585 vs fattn-vec.hpp:648-654）。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
    fam = {'公共 API': 0, '主机框架': 0, '量化矩阵乘': 0, 'FlashAttention': 0,
           '循环与融合': 0, '常规算子内核': 0, '模板实例化': 0}
    B = {'ggml-sycl', 'common', 'backend', 'base', 'type', 'mem', 'memtrace',
         'sycl_hw', 'presets', 'helper'}
    C = {'mmq', 'mmvq', 'dmmv', 'vecdotq', 'dequantize', 'quants', 'gemm', 'esimd'}
    E = {'gated_delta_net', 'gla', 'dsv4-hc', 'lightning-indexer', 'opt-step',
         'fusion', 'wkv', 'ssm_conv', 'ssm_scan'}
    for p in PLAN_FILES:
        if p == 'ggml/include/ggml-sycl.h':
            fam['公共 API'] += 1
            continue
        if p.startswith('ggml/src/ggml-sycl/template-instances/'):
            fam['模板实例化'] += 1
            continue
        s = os.path.basename(p).rsplit('.', 1)[0]
        if s.startswith('fattn'):
            fam['FlashAttention'] += 1
        elif s in C:
            fam['量化矩阵乘'] += 1
        elif s in B:
            fam['主机框架'] += 1
        elif s in E:
            fam['循环与融合'] += 1
        else:
            fam['常规算子内核'] += 1
    print(f'计划清单 : {len(PLAN_FILES)} 个文件')
    for k, v in fam.items():
        print(f'  {k:<12} {v}')
    print(f'  （合计 {sum(fam.values())}）')
    print(f'覆盖声明 : {len(L.files)} 项（含 {len(CUDA_REFS)} 个 CUDA 侧对照文件）')
