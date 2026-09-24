#!/usr/bin/env python3
"""L6-06 · Vulkan 后端：主机端 —— 课件 spec。

运行：python3 L6-gpu-backend/L6-06-vulkan-host/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

# ---- 本课计划覆盖的 7 个文件（plan_matrix --files 的 L6-06 清单）----
SRC_H = 'ggml/include/ggml-vulkan.h'
SRC_VK = 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
SRC_BUFS = 'ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp'
SRC_DBG = 'ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp'
SRC_PUSH = 'ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h'
SRC_TYPES = 'ggml/src/ggml-vulkan/ggml-vulkan-types.h'
SRC_COMMON = 'ggml/src/ggml-vulkan/ggml-vulkan-common.h'
# ---- 构建期一侧（非 L6-06 计划清单，见文末说明）----
SRC_GEN = 'ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp'
SRC_CMAKE = 'ggml/src/ggml-vulkan/CMakeLists.txt'

L = Lesson(
    id='L6-06',
    layer='L6 · GPU 后端执行',
    title='Vulkan 后端：主机端',
    codecap='ggml/src/ggml-vulkan/*（逐字引用）',
    nav={'prev': {'href': '../L6-05-sycl-backend/index.html', 'label': 'L6-05 SYCL 后端'},
         'next': {'href': '../L6-07-vulkan-shaders/index.html', 'label': 'L6-07 Vulkan 计算着色器'}},
)

L.cover(SRC_H, SRC_VK, SRC_BUFS, SRC_DBG, SRC_PUSH, SRC_TYPES, SRC_COMMON)

L.note('**一句话**：Vulkan 后端把 ggml 图上的一个节点翻译成三样 Vulkan 对象 —— **计算 pipeline**、'
       '**descriptor set**、**push constants**；而这三样东西的"内核一侧"是构建期就编好的 '
       'SPIR-V 字节数组，运行期只做对象创建与绑定。')
L.note('本课只看**主机端**（C++ 那一侧）。计算着色器本身的 GLSL 源码是下一课 L6-07 的主题；'
       '但"为什么 Vulkan 必须有独立的 shader 生成步骤"这个问题，只有在主机端这一侧才看得清 —— '
       '因为运行期能拿到的只有 `spv_data` / `spv_size` 两个值。')

L.prereqs('`L6-05`（SYCL 后端：对照另一种"主机端 + 内核"的分工）')

L.goal(
    '说出 Vulkan 后端为什么需要**预生成的 shader**（本课验收点），并背出 `.comp` 到 pipeline 的步骤链；',
    '说出一个 ggml 节点从"进后端"到"发 dispatch"要经过哪三样 Vulkan 对象，各自在哪个文件里创建；',
    '对照 `L3-01` 的三张虚表契约，指出 Vulkan 的三个实现分别在哪个文件、哪些槽位被显式置 `NULL`；',
    '解释 `descriptor set` 与 `push constants` 的分工：为什么前者可以整套复用，后者每次都重写。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L6 · GPU 后端执行',
    title='Vulkan 后端：主机端在做什么',
    sub='后端对外只有三张虚表；表后面是一条"构建期编 shader、运行期建对象"的流水线。',
    caption='下一课 L6-07 讲着色器内部的 GLSL；本课讲它外面的主机端 C++。',
    src=SRC_H, parts=[(10, 11), (13, 25)], duration=16000,
    mark_src=[10, 11, 14, 17, 19, 21, 23, 25],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML =
  '<div class="flow" style="justify-content:center">' +
    '<span class="chip">模型定义</span><span class="arrow">-></span>' +
    '<span class="chip">ggml 图</span><span class="arrow">-></span>' +
    '<span class="chip">调度器（L4-02）</span><span class="arrow">-></span>' +
    '<span class="chip a">Vulkan 后端</span>' +
  '</div>' +
  '<div class="flow" id="files" style="justify-content:center;gap:5px"></div>' +
  '<div class="row" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const files = wrap.querySelector('#files');
['ggml-vulkan.cpp', 'ggml-vulkan-buffers.cpp', 'ggml-vulkan-debug.cpp',
 'ggml-vulkan-types.h', 'ggml-vulkan-common.h', 'ggml-vulkan-push-constants.h',
 'ggml/include/ggml-vulkan.h'].forEach(function (f, i) {
  files.appendChild(U.chip(f, i < 3 ? 'a' : (i < 6 ? 'b' : 'c')));
});

const defs = [
  { c: 'a', w: '224px', t: '主机端的活', b: '建实例/设备、建 pipeline、分配 descriptor set、填 push constants、挑内存类型', m: 'ggml-vulkan.cpp（16274 行）' },
  { c: 'b', w: '224px', t: '着色器端（L6-07）', b: 'GLSL 计算着色器：mul_mm / flash_attn / 各量化类型的反量化', m: 'vulkan-shaders/*.comp' },
  { c: 'c', w: '224px', t: '对外的门面', b: '5 个导出函数 + 2 个 buffer type，形状与 CUDA/SYCL 完全同构', m: 'ggml/include/ggml-vulkan.h' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(function (d) { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(function (e) { e.style.opacity = '.32'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '一个 ggml 节点走到这里，后端要回答三个问题：<span class="k">用哪个 pipeline</span>、' +
  '<span class="k">它的输入输出放哪几个 buffer</span>、<span class="k">这次 dispatch 的标量参数是多少</span>。',
  '本课后半段就按这三个问题展开：pipeline（第 4-5 幕）、descriptor set（第 7 幕）、push constants（第 8 幕）。',
  '但第 2-3 幕先回答一个更靠前的问题：<span class="v">pipeline 里的 SPIR-V 是哪来的</span>。' +
  '答案是构建期生成的，不是运行期编的。',
  '对外的门面只有 5 个函数 + 2 个 buffer type —— 与 L3-01 的后端契约一一对应，' +
  '真正的复杂度全在实现里。'
];
defs.forEach(function (_, i) {
  tl.at(700 + i * 3300, function () {
    els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
    msg.innerHTML = texts[i];
  });
});
tl.at(13000, function () {
  els.forEach(function (e) { e.style.opacity = '1'; });
  msg.innerHTML = '本课覆盖 7 个文件：<span class="v">主机端 3 个 .cpp</span>、' +
    '<span class="v">4 个共享头</span>。';
});
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L6-06 · 核心',
    title='★ <span class="hl-a">Vulkan 只吃 SPIR-V</span>，所以 shader 必须在构建期编好',
    sub='主机端能拿到的只有一段字节数组；GLSL 前端是一个外部工具，只存在于构建期。',
    caption='把 .spv 逐字节嵌成数组那几行在 source.md 第二节；第 3 幕接着看构建规则。',
    src=SRC_GEN, parts=[(350, 353), (355, 357), (440, 443), (1398, 1412), (1421, 1428)],
    duration=22000,
    mark_src=[351, 356, 440, 441, 1398, 1399, 1401, 1402, 1404, 1405, 1407, 1408, 1410, 1411, 1421, 1428],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML =
  '<div class="row" id="steps" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '163px', k: '1', t: '.comp', b: 'GLSL 计算着色器源码<br>（L6-07 逐字讲）', m: 'vulkan-shaders/*.comp' },
  { c: 'c', w: '163px', k: '2', t: 'glslc', b: '外部编译器：<br>compute 阶段 + target-env', m: 'GLSL -> SPIR-V' },
  { c: 'd', w: '163px', k: '3', t: '<name>.spv', b: 'SPIR-V 二进制，<br>只存在于构建目录', m: 'vulkan-shaders.spv/' },
  { c: 'b', w: '163px', k: '4', t: '<name>_data[]', b: '生成器把它写成 C 数组<br>+ _len 长度', m: '<name>.comp.cpp + .hpp' }
];
const host = wrap.querySelector('#steps');
const els = defs.map(function (d, i) {
  const e = U.el('div', { class: 'card', style: 'width:163px;border-left-color:var(--' + d.c + ')' });
  e.innerHTML = '<div class="cm" style="margin:0 0 2px">STEP ' + d.k + '</div>' +
    '<div class="ct" style="color:var(--' + d.c + ')">' + U.esc(d.t) + '</div>' +
    '<div class="cb">' + d.b + '</div>' +
    '<div class="cm">' + U.esc(d.m) + '</div>';
  host.appendChild(e);
  return e;
});
els.forEach(function (e) { e.style.opacity = '.30'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '第一步：<span class="k">.comp</span> 是 GLSL 源码，属于下一课的范围。',
  '第二步：生成器拼出 <span class="v">glslc</span> 命令行 —— ' +
  '<span class="k">-fshader-stage=compute</span> 指明这是计算着色器，<span class="k">target-env</span> 指明目标 Vulkan 版本。',
  '第三步：glslc 输出 <span class="v">.spv</span> —— 这就是 SPIR-V 字节码。' +
  '注意它落在构建目录，<span class="k">源码仓库里没有它</span>。',
  '第四步：生成器把 <span class="v">.spv</span> 逐字节读回来，写成 ' +
  '<span class="k">const unsigned char &lt;name&gt;_data[]</span> 和 <span class="k">&lt;name&gt;_len</span>' +
  '（source.md 第二节逐字引用）。',
  '这四个成员（glslc / source / output-dir / target 两个）就是生成器的全部输入 —— ' +
  '它一次只处理一个 .comp，由构建系统循环调用。',
  '<span class="hl-a">Vulkan 规范只要求驱动接受 SPIR-V</span>；GLSL 前端不在 API 里，' +
  '仓库里也没有任何"运行期把 GLSL 编成 SPIR-V"的调用。'
];
defs.forEach(function (_, i) {
  tl.at(700 + i * 3400, function () {
    els.forEach(function (e, k) { e.style.opacity = k <= i ? '1' : '.30'; });
    msg.innerHTML = texts[i];
  });
});
tl.at(15200, function () {
  els.forEach(function (e) { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
tl.at(18600, function () { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L6-06 · 构建期',
    title='★ 构建系统把 <span class="hl-c">glslc</span> 变成一条编译规则',
    sub='每个 .comp 一条自定义命令；生成的 .cpp 进库，生成的 .hpp 被主机端头文件 include。',
    caption='带 glslc / source / target 参数的那条完整命令在 source.md 第三节（CMake 变量语法放在散文区更清楚）。',
    src=SRC_CMAKE, parts=[(9, 9), (209, 213), (227, 229), (232, 236), (238, 245), (251, 258)],
    duration=20000,
    mark_src=[9, 209, 210, 212, 227, 229, 234, 236, 238, 240, 242, 245, 254, 256],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
root.appendChild(wrap);

const t = U.table(
  ['构建产物', '谁生成它', '给谁用'],
  [['vulkan-shaders-gen（宿主可执行）', '先于库编出来的宿主工具：两条命令都 DEPENDS 它', '被下面两条命令调用'],
   ['ggml-vulkan-shaders.hpp', '一条自定义命令（不带 source）：只声明全部变体', '被 ggml-vulkan-types.h 引入'],
   ['每个 .comp 对应一个 .cpp', '每个 .comp 一条自定义命令：调 glslc 出 .spv，再嵌成数组', '编进 ggml-vulkan 目标']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  'configure 阶段就要求 glslc 存在（<span class="k">find_package(Vulkan COMPONENTS glslc REQUIRED)</span>）——' +
  '它是<span class="v">构建依赖</span>，不是运行期依赖。',
  '生成器是<span class="v">先于库</span>编出来的宿主工具 —— 两条命令的 DEPENDS 里都点名它。',
  '头文件那条命令不带 source：生成器在 source 为空时只登记名字、不编译（上一幕的 ' +
  '<span class="k">ggml-vulkan-shaders-gen.cpp:440</span>）。',
  '每个 .comp 一条命令：参数就是上一幕生成器解析的那一组，' +
  '完整命令行见 <span class="v">source.md 第三节</span>。',
  '产物通过 <span class="k">target_sources</span> 挂到 ggml-vulkan 目标上：' +
  '程序员写的 .cpp 与生成的 .cpp 一起被编译、一起进库。'
];
tl.at(700, function () { msg.innerHTML = texts[0]; });
rows.forEach(function (r, i) {
  tl.at(3000 + i * 3200, function () {
    rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
    msg.innerHTML = texts[i + 1];
  });
});
tl.at(14200, function () {
  rows.forEach(function (x) { x.className = ''; });
  msg.innerHTML = texts[4];
});
tl.at(17200, function () {
  msg.innerHTML = '所以：<span class="v">改一个 .comp 就是改一次构建输入</span> —— ' +
    '没有"运行期加载新内核"这条路。';
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L6-06 · 运行期',
    title='运行期只做三件事：<span class="hl-b">module</span> / <span class="hl-b">layout</span> / <span class="hl-b">pipeline</span>',
    sub='SPIR-V 是数据，不是代码：主机端把它交给驱动去创建对象，自己不编译。',
    caption='注意 parameter_count 的断言：一个 pipeline 最多绑定 12 个 buffer（第 7 幕展开）。',
    src=SRC_VK, parts=[(554, 564), (669, 678), (743, 749), (798, 804)],
    duration=20000,
    mark_src=[554, 560, 561, 564, 669, 671, 674, 677, 678, 744, 800, 801],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row" id="steps" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const defs = [
  { c: 'a', k: '1', t: 'ShaderModule', b: '把内嵌的 SPIR-V 数组<br>交给驱动', m: 'createShaderModule' },
  { c: 'c', k: '2', t: 'PipelineLayout', b: '一套 descriptor 布局<br>+ 一段 push constant 范围', m: 'createPipelineLayout' },
  { c: 'd', k: '3', t: 'Pipeline', b: '计算管线：<br>入口点固定为 main', m: 'createComputePipeline' },
  { c: 'b', k: '4', t: '登记缓存', b: '压进 all_pipelines，<br>置 compiled 并唤醒等待者', m: 'compile_cv.notify_all()' }
];
const host = wrap.querySelector('#steps');
const els = defs.map(function (d) {
  const e = U.el('div', { class: 'card', style: 'width:163px;border-left-color:var(--' + d.c + ')' });
  e.innerHTML = '<div class="cm" style="margin:0 0 2px">STEP ' + d.k + '</div>' +
    '<div class="ct" style="color:var(--' + d.c + ')">' + U.esc(d.t) + '</div>' +
    '<div class="cb">' + d.b + '</div>' +
    '<div class="cm">' + U.esc(d.m) + '</div>';
  host.appendChild(e);
  return e;
});
els.forEach(function (e) { e.style.opacity = '.30'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '函数签名里最关键的两个参数是 <span class="k">spv_size</span> 与 <span class="k">spv_data</span>：' +
  '运行期看到的是<span class="v">字节</span>，不是 GLSL 源码。',
  '第 1 步：<span class="v">ShaderModuleCreateInfo</span> 直接吃这段 uint32 数组。' +
  '此处还有一小段"打补丁"：按设备能力往 SPIR-V 里插 float control 能力位。',
  '第 2 步：pipeline layout = 设备的公共 descriptor 布局 + 这个 pipeline 的 push constant 范围（长度来自 ' +
  '<span class="k">push_constant_size</span>）。',
  '第 3 步：<span class="v">createComputePipeline</span> 若失败会打印 pipeline 名字再抛出 —— ' +
  '因为名字是唯一的定位线索。',
  '第 4 步：加锁登记 <span class="k">all_pipelines</span>、把 <span class="k">compiled</span> 置真并唤醒等待者。' +
  '第 5 幕看这套"谁在等"的机制。'
];
defs.forEach(function (_, i) {
  tl.at(700 + i * 3300, function () {
    els.forEach(function (e, k) { e.style.opacity = k <= i ? '1' : '.30'; });
    msg.innerHTML = texts[i];
  });
});
tl.at(13800, function () {
  els.forEach(function (e) { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L6-06 · 缓存',
    title='pipeline 是<span class="hl-d">对象</span>，也是<span class="hl-d">缓存项</span>',
    sub='设备启动时一次建齐，运行期只查表；没编完的才补编，并用条件变量等它。',
    caption='懒编译入口 ggml_pipeline_request_descriptor_sets()：未 compiled 就再调一次 ggml_vk_load_shaders（见 source.md 第三节）。',
    src=SRC_TYPES, parts=[(245, 270), (791, 791), (798, 801)],
    duration=20000,
    mark_src=[245, 247, 248, 249, 250, 251, 252, 255, 258, 260, 269, 791, 798, 800],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '一个 pipeline 的全部', b: 'shader_module / layout / pipeline 三个句柄，' +
      '外加 push_constant_size、parameter_count、wg_denoms、align', m: 'struct vk_pipeline_struct' },
  { c: 'c', t: '三态', b: 'initialized（字段填好没）<br>compile_pending（有人正在编）<br>compiled（可以用了）', m: 'bool + std::atomic<bool>' },
  { c: 'd', t: '缓存形式', b: '设备对象里成百个字段/映射：<br>矩阵乘用 map 以 (类型A,类型B,是否 MoE,是否 f16acc) 为键', m: 'device->pipeline_matmul / pipeline_dequant[]' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(function (d) { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(function (e) { e.style.opacity = '.32'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '先看这个结构体：它同时是"一个 Vulkan 对象的所有权"和"一个缓存项的状态"。',
  '<span class="k">shader_module / layout / pipeline</span>：三个句柄一起建、一起销毁（第 4 幕的三个步骤）。',
  '<span class="k">push_constant_size / parameter_count</span>：运行期要用它们做断言 —— ' +
  '给的 buffer 数、push constant 大小必须和建表时一致。',
  '三个状态位决定了并发行为：<span class="v">compile_pending</span> 去重，<span class="v">compiled</span> 是等待条件。',
  '缓存不是"按需新建"，而是<span class="k">预先建齐 + 惰性补齐</span>：设备初始化时就调一次 ' +
  '<span class="v">ggml_vk_load_shaders(device)</span>，运行期只是查表和等。',
  '这也解释了为什么 ops 的选择表（ggml_vk_op_get_pipeline 的大 switch）里只有"查"，没有"建"。'
];
defs.forEach(function (_, i) {
  tl.at(700 + i * 3000, function () {
    els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
    msg.innerHTML = texts[i];
  });
});
tl.at(12200, function () {
  els.forEach(function (e) { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
tl.at(15600, function () { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L6-06 · 接口',
    title='三张虚表，与 CUDA / SYCL <span class="hl-e">同构</span>',
    sub='L3-01 定义契约，各后端填表。Vulkan 的两个 buffer 面表就在 buffers.cpp 里。',
    caption='表里的 CUDA / SYCL 行号来自本仓库直接检索，用于对照；本课不引用它们的源码（见文末说明）。',
    src=SRC_BUFS, parts=[(3, 10), (745, 756)], duration=18000,
    mark_src=[3, 4, 5, 9, 745, 746, 748, 750, 755, 756],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
root.appendChild(wrap);

const t = U.table(
  ['虚表（L3-01 的契约）', 'Vulkan 实现处', '要点', 'CUDA / SYCL 的同名表'],
  [['ggml_backend_buffer_type_i', 'ggml-vulkan-buffers.cpp:3', '6 个槽位；is_host 显式 NULL', 'ggml-cuda.cu:927 / ggml-sycl.cpp:1055'],
   ['ggml_backend_buffer_i', 'ggml-vulkan-buffers.cpp:745', '11 个槽位；reset 显式 NULL', 'ggml-cuda.cu:853 / ggml-sycl.cpp:917'],
   ['ggml_backend_i', 'ggml-vulkan.cpp:14830', '16 个槽位；四个 graph_plan_* 显式 NULL', 'ggml-cuda.cu:4840 / ggml-sycl.cpp:6284']],
  { monoCols: [0, 1, 3] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '回顾 L3-01：ggml 后端就是三张函数指针表。表填好了，调度器就能用它执行图。',
  '<span class="k">buffer_type_i</span>：回答"这种 buffer 怎么分配、对齐多少、上限多少"。' +
  'Vulkan 的 <span class="v">is_host</span> 是 NULL —— 它不是主机内存。',
  '<span class="k">buffer_i</span>：回答"怎么往 buffer 里读/写张量"。Vulkan 的 ' +
  '<span class="v">reset</span> 是 NULL（没有"重置整块"的语义）。',
  '<span class="k">backend_i</span>：回答"怎么执行图"。Vulkan 不做图计划缓存（四个 graph_plan_* 为 NULL），' +
  '图执行入口是 <span class="v">graph_compute</span>。',
  '三张表的<span class="k">形状与 CUDA / SYCL 完全一致</span> —— 后端之间的差异在实现，不在契约。' +
  '这正是 L3-01 那一课的价值。'
];
tl.at(700, function () { msg.innerHTML = texts[0]; });
rows.forEach(function (r, i) {
  tl.at(2600 + i * 2900, function () {
    rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
    msg.innerHTML = texts[i + 1];
  });
});
tl.at(12800, function () {
  rows.forEach(function (x) { x.className = ''; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L6-06 · descriptor set',
    title='一套布局，一个池，<span class="hl-a">按需增长</span>',
    sub='所有 pipeline 共用同一个 descriptor set layout：12 个 storage buffer 槽位。',
    caption='消费侧见 ggml-vulkan-common.h 的 ggml_vk_dispatch_pipeline 模板（source.md 第五节）。',
    src=SRC_VK, parts=[(4647, 4660), (825, 861)], duration=20000,
    mark_src=[4647, 4649, 4650, 4660, 827, 834, 835, 846, 847, 853, 856, 857],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '一套布局', b: '按 MAX_PARAMETER_COUNT（=12）循环：每个 binding 都是 compute 阶段的 storage buffer', m: 'device->dsl' },
  { c: 'c', t: '一个池 256 个', b: '每池按 12 x 256 个 storage buffer 描述符申请', m: 'VK_DEVICE_DESCRIPTOR_POOL_SIZE' },
  { c: 'd', t: '按需增长', b: '需求先累加；不足时按"当前 1.5 倍"扩，一次分配一批', m: 'Grow by 50%' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(function (d) { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(function (e) { e.style.opacity = '.32'; });
const msg = wrap.querySelector('#msg');
const texts = [
  '先看布局：<span class="k">12 个 binding，全是 storage buffer</span>，' +
  '于是任何 pipeline 的 descriptor 都可以用同一个 layout。',
  '第 4 幕里每个 pipeline layout 都带上了 <span class="v">device-&gt;dsl</span> —— ' +
  'layout 只有一份，pipeline 之间共享。',
  '分配是批量的：<span class="k">descriptor_sets</span> 是个向量，池按 256 个一组追加。',
  '每次请求把需求累加到 <span class="v">pipeline_descriptor_set_requirements</span>；' +
  '够用就立刻返回，不够才扩。',
  '扩张策略是"<span class="k">当前规模的 1.5 倍</span>取大者"，避免每来一个算子就分配一次。',
  '一次 dispatch 取走一个 set（<span class="v">descriptor_set_idx++</span>），' +
  '把这次的 buffer 写进去就绑 —— 下一幕看完整动作。'
];
defs.forEach(function (_, i) {
  tl.at(700 + i * 3100, function () {
    els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
    msg.innerHTML = texts[i];
  });
});
tl.at(10500, function () { els.forEach(function (e) { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
tl.at(14000, function () { msg.innerHTML = texts[4]; });
tl.at(17200, function () { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L6-06 · push constants',
    title='push constants：每个 op 一张 <span class="hl-f">POD 结构</span>',
    sub='形状、步长、标量参数都塞在这里；上限由源码的 static_assert 钉死。',
    caption='数据缓冲走 descriptor set，标量参数走 push constants —— 两者在同一次 dispatch 里配齐。',
    src=SRC_PUSH, parts=[(398, 405), (413, 425), (958, 972)], duration=18000,
    mark_src=[398, 399, 400, 402, 404, 413, 418, 423, 425, 958, 961, 970],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const defs = [
  { c: 'f', t: '二元算子的那张', b: '一个 ne（元素个数）+ 三组 ne/nb（三张张量的形状与步长）' +
      ' + misalign_offsets + 三个标量参数', m: 'vk_op_binary_push_constants' },
  { c: 'a', t: '融合加法的那张', b: 'dst 的形状 + <b>nb[12][4]</b>：一次 dispatch 最多 12 张张量的步长', m: 'vk_op_multi_add_push_constants' },
  { c: 'b', t: '怎么变成参数字节', b: '两个模板：size 取 sizeof（或 vector/array 的长度），data 取地址（或 data()）', m: 'push_constant_size() / push_constant_data()' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(function (d) { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(function (e) { e.style.opacity = '.32'; });
const msg = wrap.querySelector('#msg');
const texts = [
  'push constants 是 Vulkan 里"每次 dispatch 都要重写的一小段常量"，所以它按 op 逐个定义结构体。',
  '第一个静态断言把 <span class="k">MAX_PARAMETER_COUNT == 12</span> 写死在编译期 —— ' +
  '它就是第 7 幕 descriptor 槽位的数量。',
  '第二个断言给结构体封顶：<span class="v">sizeof(...) &lt;= 256</span>。' +
  'push constants 是稀缺资源（最少保证 128 字节），源码自己把它限制在 256。',
  '<span class="k">nb[12][4]</span> 是"多输入融合"的代价与红利：一次 dispatch 能带上 12 张张量的步长，' +
  '所以能把连续的 ADD/RMS 融合进同一次分派。',
  '两个模板函数的存在只为一件事：让 <span class="v">pushConstants(...)</span> 的实参在编译期就和 ' +
  '<span class="v">pipeline-&gt;push_constant_size</span> 对得上。',
  '记住分工：<span class="k">descriptor set = 数据在哪</span>，' +
  '<span class="k">push constants = 这次怎么算</span>。'
];
defs.forEach(function (_, i) {
  tl.at(700 + i * 3100, function () {
    els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
    msg.innerHTML = texts[i];
  });
});
tl.at(10500, function () { els.forEach(function (e) { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
tl.at(13800, function () { msg.innerHTML = texts[4]; });
tl.at(16600, function () { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L6-06 · 收束',
    title='一次 dispatch 的全部零件',
    sub='这个模板函数就是本课的落点：pipeline + descriptor set + push constants -> 一次分派。',
    caption='下一课 L6-07 打开 pipeline 里的 SPIR-V：那些 .comp 到底怎么写。',
    src=SRC_COMMON, parts=[(248, 281)], duration=22000,
    mark_src=[249, 250, 262, 263, 264, 266, 267, 268, 270, 271, 272, 279],
    notes_src={250: 'workgroup 数 = 元素数 ÷ 每个 workgroup 覆盖的元素数（建 pipeline 时定下的 wg_denoms）',
               258: '先断言不超过设备的分派上限 —— 超了要换 tile 配置，而不是硬发',
               263: 'pipeline 建表时声明要几个 buffer，这里就必须给几个：parameter_count 是契约',
               264: 'push constant 大小也要和建表时一致：这就是那两个模板函数存在的理由',
               267: '一次写入把 parameter_count 个 storage buffer 挂到 set 的 0 号 binding 起',
               270: 'push constants 单独走一条路径：它不属于 descriptor set',
               279: '到这里才真正分派：前面的都是准备动作'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>';
root.appendChild(wrap);

const t = U.table(
  ['零件', '谁创建、什么时候', '每次 dispatch 是否变化'],
  [['pipeline（含 SPIR-V module）', '设备初始化时 ggml_vk_load_shaders 建齐；缺的惰性补齐', '不变（首次补齐）'],
   ['pipeline layout / descriptor set layout', 'device->dsl：一套布局给所有 pipeline 共享', '不变'],
   ['descriptor set', '从池里取一个，写入这次的 parameter_count 个 buffer', '每次变'],
   ['push constants', '按 op 的 POD 结构体填好，整块推给驱动', '每次变'],
   ['workgroup 数量', 'elements ÷ wg_denoms，向上取整', '每次变']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  'Vulkan 后端为什么必须在构建期把 <span class="mono">.comp</span> 预编译成 SPIR-V？' +
  '若要让 Vulkan 后端支持一个新的算子内核，最少要动哪几处？',
  '因为主机端只拿得到 <span class="mono">spv_data</span> / <span class="mono">spv_size</span> 两个值' +
  '（<span class="mono">ggml-vulkan.cpp:554</span>、<span class="mono">:564</span>），' +
  '仓库里没有任何"运行期把 GLSL 编成 SPIR-V"的路径；GLSL 前端是外部工具 ' +
  '<span class="mono">glslc</span>，由构建期调用（<span class="mono">vulkan-shaders-gen.cpp:350</span>、' +
  '<span class="mono">CMakeLists.txt:242</span>）。<br>' +
  '最少三处：① 写 <span class="mono">vulkan-shaders/&lt;op&gt;.comp</span>；' +
  '② 在生成器的 <span class="mono">process_shaders()</span> 里用 ' +
  '<span class="mono">string_to_spv()</span> 登记这个变体（否则生成的 .hpp 里没有它的 ' +
  '<span class="mono">_data</span> / <span class="mono">_len</span>）；' +
  '③ 在 <span class="mono">ggml-vulkan.cpp</span> 的 <span class="mono">ggml_vk_load_shaders()</span> 里' +
  '为它创建 pipeline，并在 <span class="mono">ggml_vk_op_get_pipeline()</span> 的选择表里把 op 映射过去。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '把本课压成一张表：三种零件，各自的创建时机与"是否每次变化"。',
  '<span class="k">pipeline 与 layout 是"不变的"</span>：构建期把 SPIR-V 编好，运行期只建一次对象。',
  '<span class="k">descriptor set 与 push constants 是"每次变的"</span>：前者换 buffer，后者换标量。',
  '这解释了为什么第 7 幕的 set 要按需增长、第 8 幕的 push constants 要按 op 分结构体。',
  '下一课 L6-07 打开 pipeline 里的 SPIR-V：<span class="v">mul_mm / mul_mmq / flash_attn</span> ' +
  '这些 .comp 怎么写、宏怎么展开。'
];
tl.at(700, function () { msg.innerHTML = texts[0]; });
rows.forEach(function (r, i) {
  tl.at(2600 + i * 2600, function () {
    rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
    msg.innerHTML = texts[Math.min(i + 1, 3)];
  });
});
tl.at(16000, function () {
  rows.forEach(function (x) { x.className = ''; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、三张虚表：ggml_backend_i 的 Vulkan 实现',
    '回顾 L3-01：一个 ggml 后端就是三张函数指针表（`ggml_backend_i` / `ggml_backend_buffer_i` / '
    '`ggml_backend_buffer_type_i`）。Vulkan 把 `backend` 那张放在 `ggml-vulkan.cpp` 末尾，'
    '两个 buffer 面的表放在 `ggml-vulkan-buffers.cpp`。\n\n'
    '值得注意的不是填了什么，而是**哪些槽位被显式置 `NULL`**：四个 `graph_plan_*` 全为空，'
    '说明 Vulkan 不走"图计划缓存"那条路径；`event_record` / `event_wait` 有实现，'
    '用来和其它后端做跨后端的流水线同步。',
    src=SRC_VK, parts=[(14830, 14847)], lang='c')

L.section(
    '二、生成器把 .spv 嵌成 C 数组',
    '这是"预生成"这条链路的最后一环：生成器把刚编译出来的 `.spv` 文件读成二进制，'
    '先声明 `_len` 与 `_data`（写进 `.hpp`），再把每个字节以十六进制字面量写进 `_data[]`（写进 `.cpp`）。'
    '注意数组的字节数**就是文件长度**：SPIR-V 是数据，没有别的包装。\n\n'
    '头文件那一遍（`source` 为空）只走到第 1253 行的声明，因为没有 `input_filepath` 就没有可读的 `.spv`。',
    src=SRC_GEN, parts=[(1252, 1253), (1261, 1268)], lang='c')

L.section(
    '三、构建规则：每个 .comp 一条自定义命令',
    '这条命令把上一节与下一节接起来：四个参数分别给出外部编译器（glslc）、这一个 `.comp`、'
    '"给它专用的"嵌入文件（所以 `add_custom_command` 的 `OUTPUT` 是逐文件生成的名字），'
    '以及所有 `.comp` 共享的同一个头文件。\n\n'
    '`DEPFILE` 让 CMake 知道这个 `.cpp` 依赖哪些被 `#include` 的 GLSL 片段；'
    '`DEPENDS` 里同时列了 `.comp` 本身、生成器源码和生成器目标 —— 三者任一变化都会触发重编。',
    src=SRC_CMAKE, parts=[(242, 256)], lang='cmake')

L.section(
    '四、生成的头文件怎么进入编译单元',
    '这三行是整条构建期链路的落点：`ggml-vulkan-shaders.hpp` **不在源码仓库里**，'
    '它是 `vulkan-shaders-gen` 在构建目录里生成的，内容只有每个 shader 变体的 '
    '`extern const uint64_t <name>_len;` 与 `extern const unsigned char <name>_data[];` 声明。\n\n'
    '`ggml-vulkan-types.h` 是 Vulkan 源文件的第一层公共头（`ggml-vulkan-common.h` 又引入它），'
    '所以每个编译单元都能看到这些数组 —— 于是 `ggml-vulkan.cpp` 里可以直接写 '
    '`matmul_f16_cm2_len, matmul_f16_cm2_data` 这样的标识符。',
    src=SRC_TYPES, parts=[(112, 114)], lang='c')

L.section(
    '五、pipeline 的缓存与懒编译',
    '设备初始化时会调一次 `ggml_vk_load_shaders(device)`，把整套 pipeline 建出来存进 `device->pipeline_*`。'
    '运行期第一次用到某个 pipeline 时，`ggml_pipeline_request_descriptor_sets()` 先看它的 `compiled`：'
    '没编完，就把这一个 pipeline 再交给 `ggml_vk_load_shaders(device, pipeline)`（只处理它），'
    '然后由条件变量等待编译完成。\n\n'
    '`ggml_vk_create_pipeline` 里的 `initialized` 判断保证：重复调用不会覆盖已经填好的字段，'
    '只会走"要不要编译"的那一半逻辑。',
    src=SRC_VK, parts=[(816, 823), (1841, 1854)], lang='c')

L.section(
    '六、buffer 的内存类型选择',
    'Vulkan 的内存类型是**设备暴露出来的**（`memoryTypes` 加上每个资源的 `memoryTypeBits` 掩码），'
    '所以分配必须"按属性挑 + 逐个试"。`ggml_vk_find_memory_properties()` 做第一层过滤：'
    '资源允许这个类型、属性位满足要求、堆还放得下。\n\n'
    '`ggml_vk_create_buffer_device()` 则是策略表：优先主机内存、UMA 设备、'
    '禁用 host-visible 显存、能否用 rebar —— 每一档都给出"首选属性 + 备选属性"，'
    '`ggml_vk_create_buffer()` 会按顺序试，全失败才抛 `OutOfDeviceMemoryError`。'
    '这是"同一份 ggml 代码在不同 GPU 上拿到不同内存"的原因。',
    src=SRC_BUFS, parts=[(12, 24), (193, 210)], lang='c')

L.section(
    '七、一次 dispatch 的组装：descriptor set + push constants',
    '`ggml_vk_dispatch_pipeline()` 是主机端所有算子的公共出口。它把本课的三种零件按固定顺序装上：'
    '取一个 descriptor set 并写入 `parameter_count` 个 storage buffer，'
    '推送 push constants，绑定 pipeline 与 descriptor set，最后 dispatch。\n\n'
    '四个断言就是"契约检查"：buffer 数不能超过 12、实际给的 buffer 数必须等于建表时的 `parameter_count`、'
    'push constant 的字节数必须等于建表时的 `push_constant_size`、workgroup 数不能超过设备上限。'
    '这正是第 4 幕与第 8 幕在编译期/建表期就要把这两个数定下来的原因。',
    src=SRC_COMMON, parts=[(248, 281)], lang='c')

L.section(
    '八、主机端的可观测性：内存日志与结果校验',
    '`ggml-vulkan-debug.cpp` 装的是三个 logger（内存 / 性能 / 同步）与结果校验路径。'
    '内存 logger 用内存属性位把每次分配分成 `device` 与 `host` 两类累加，'
    '这正是第四节那张"策略表"的可观测侧面。\n\n'
    '`ggml_vk_get_op_batch_size()` 给每个算子一个"批量"估计：`GET_ROWS` 返回 0（不值得卸载）、'
    '`MUL_MAT` 返回 `ne[1]`、`MUL_MAT_ID` / `ROPE` 返回 `ne[2]`、其余返回 `ggml_nrows()`。'
    '它是设备侧卸载判据的输入（见 `ggml-vulkan.cpp:15693` 的 `ggml_backend_vk_device_offload_op`）。',
    src=SRC_DBG, parts=[(15, 27), (766, 779)], lang='c')

L.footnote_add('本课覆盖计划清单里的 7 个文件（`ggml/include/ggml-vulkan.h` + `ggml/src/ggml-vulkan/` 下 6 个），'
               '全部计入覆盖率。')
L.footnote_add('本课另逐字引用了两个**构建期**文件：`ggml/src/ggml-vulkan/CMakeLists.txt` 与 '
               '`ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`。'
               '后者按计划归属 L6-07（整个 `vulkan-shaders/` 目录都在那一课）；本课只在"生成期与主机端如何使用它"'
               '这一侧引用它 —— 因为"为什么需要预生成 shader"这条验收点，只有把两处放在一起才看得出来。')
L.footnote_add('第 6 幕对照表里的 CUDA / SYCL 行号来自本仓库直接检索（`ggml-cuda.cu` / `ggml-sycl.cpp`），'
               '本课不引用它们的源码，故不计入本课覆盖率 —— 它们是 L6-01 / L6-05 的覆盖范围。')

L.conclusion(
    '★ 为什么 Vulkan 必须预生成 shader',
    '主机端拿到的是**字节**，不是源码：`ggml_vk_create_pipeline_func()` 的入参就是 '
    '`spv_size` / `spv_data`，第 564 行直接把这段 uint32 数组交给 `ShaderModuleCreateInfo`。\n\n'
    '完整的步骤链（每一步都有逐字引用）：\n\n'
    '```text\n'
    'vulkan-shaders/<name>.comp            GLSL 计算着色器源码（L6-07）\n'
    '  -> process_shaders() 登记变体       vulkan-shaders-gen.cpp:675 起\n'
    '  -> string_to_spv_func() 调 glslc    vulkan-shaders-gen.cpp:350-357\n'
    '  -> <name>.spv（SPIR-V 字节码）      构建目录，不在源码仓库\n'
    '  -> write_output_files() 嵌成数组    vulkan-shaders-gen.cpp:1261-1268\n'
    '  -> <name>.comp.cpp + .hpp           由 CMake 自定义命令产出\n'
    '  -> #include "ggml-vulkan-shaders.hpp"   ggml-vulkan-types.h:114\n'
    '  -> ggml_vk_load_shaders() 取用       ggml-vulkan.cpp:1592 起\n'
    '  -> createShaderModule / Pipeline     ggml-vulkan.cpp:669 / :744\n'
    '```\n\n'
    '**推论的边界**：运行期能选的只是"已经生成好的变体"——`ggml_vk_op_get_pipeline()` 那张大 switch '
    '和 `ggml_vk_load_shaders()` 里成排的 `_len` / `_data` 就是可选集合的全部。'
    '想要一个新内核，必须改 `.comp`、在生成器里登记变体、再重新构建。')

L.conclusion(
    '三张虚表 + 三种零件',
    '| 面 | Vulkan 实现 | 关键槽位 |\n|---|---|---|\n'
    '| `ggml_backend_buffer_type_i` | `ggml-vulkan-buffers.cpp:3` | `is_host = NULL` |\n'
    '| `ggml_backend_buffer_i` | `ggml-vulkan-buffers.cpp:745` | `reset = NULL` |\n'
    '| `ggml_backend_i` | `ggml-vulkan.cpp:14830` | 四个 `graph_plan_* = NULL` |\n\n'
    '一次 dispatch 的三种零件：**pipeline**（不变）、**descriptor set**（每次变，装数据缓冲）、'
    '**push constants**（每次变，装标量参数）。三者由 `ggml_vk_dispatch_pipeline()` 按固定顺序装上。')

L.conclusion(
    '跨课位置',
    '- **L6-05（SYCL）**：同样是"主机端 + 内核"，但内核与主机端是同一门语言（C++），没有独立的着色器编译步骤。\n'
    '- **L6-07（Vulkan 计算着色器）**：接手本课留下的 `.comp`，讲 `mul_mm` / `mul_mmq` / `flash_attn` 怎么写。\n'
    '- **L3-01（后端契约）**：本课第 6 幕的三张表就是那份契约的 Vulkan 实现。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
