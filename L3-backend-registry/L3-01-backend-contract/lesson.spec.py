#!/usr/bin/env python3
"""L3-01 · 后端接口 ggml-backend.h：后端的契约 —— 课件 spec。

运行：python3 L3-backend-registry/L3-01-backend-contract/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

PUB = 'ggml/include/ggml-backend.h'
IMPL = 'ggml/src/ggml-backend-impl.h'

L = Lesson(
    id='L3-01',
    layer='L3 · 后端发现与注册',
    title='后端接口 ggml-backend.h：后端的契约',
    codecap='ggml/include/ggml-backend.h 与 ggml/src/ggml-backend-impl.h（逐字引用）',
    nav={'prev': {'href': '../../L2-model-to-graph/L2-15-models-dense-b/index.html',
                  'label': 'L2-15 模型家族（六）稠密下'},
         'next': {'href': '../L3-02-registry-and-devices/index.html',
                  'label': 'L3-02 后端注册表与设备发现'}},
)

L.note('**一句话**：ggml 不认识任何一块硬件。它只认识一份**契约** —— 一堆函数指针。'
       '谁把这份契约填完，谁就是一个后端；CPU、CUDA、Metal、Vulkan 之间的差别，'
       '在 ggml 这一侧全部消失了。')
L.note('本课覆盖两个文件：公共头 `ggml/include/ggml-backend.h`（模型代码能看到的）'
       '与内部头 `ggml/src/ggml-backend-impl.h`（**后端实现者**才需要看的）。'
       '前者的 115 处 `GGML_API` 只是转发，真正的约定是后者里的五张虚表；'
       '这一课重点拆三张：`ggml_backend_device_i`（发现与创建）、'
       '`ggml_backend_buffer_type_i`（内存怎么分配）、`ggml_backend_i`（计算怎么执行）。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L3 · 后端发现与注册',
    title='一个后端，先要签一份<span class="hl-a">契约</span>',
    sub='公共头里没有结构体，只有 7 个不透明句柄：模型代码永远看不到后端的内部。',
    caption='本幕只引 ggml-backend.h 第 22-30 行。下一幕进内部头 ggml-backend-impl.h 看虚表本身。',
    src=PUB, parts=[(24, 30)], duration=16000,
    mark_src=[24, 25, 27, 29, 30],
    notes_src={28: 'graph_plan_t 是唯一的例外：它不是 struct 指针，而是 void* —— 一个纯不透明的执行计划',
               24: '6 个 struct 指针 + 1 个 void*，就是公共头对"后端"的全部认知'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip">计算图</span><span class="arrow">-></span>
    <span class="chip a">调度器</span><span class="arrow">-></span>
    <span class="chip c">后端契约</span><span class="arrow">-></span>
    <span class="chip b">CPU / CUDA / Metal</span>
  </div>
  <div class="row wrap" id="handles" style="gap:6px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { t: 'ggml_backend_dev_t', k: '设备' },
  { t: 'ggml_backend_t', k: '后端（计算流）' },
  { t: 'ggml_backend_buffer_type_t', k: '内存类型' },
  { t: 'ggml_backend_buffer_t', k: '一块内存' },
  { t: 'ggml_backend_reg_t', k: '注册项' },
  { t: 'ggml_backend_event_t', k: '事件' },
  { t: 'ggml_backend_graph_plan_t', k: '图计划（void*）' }
];
const host = wrap.querySelector('#handles');
const els = defs.map(d => { const e = U.chip(d.t + ' · ' + d.k); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');
const msg = wrap.querySelector('#msg');
const texts = [
  '这一课只回答一个问题：<span class="k">别人写的后端，凭什么能被 ggml 调用？</span>',
  '答案是一份契约。公共头用 <span class="v">7 个 typedef</span> 把实现藏起来：<br>其中 <span class="k">6 个是不透明 struct 指针</span>，只有 <span class="v">ggml_backend_graph_plan_t</span> 是 void*。',
  '结构体一个都不在公共头里 —— 它们全在 <span class="v">ggml/src/ggml-backend-impl.h</span>，<br>那是给"写后端的人"看的内部契约头。',
  '公共头的 <span class="v">115 处 GGML_API</span> 只做转发：函数名好记，干活的是虚表。',
  '记住这条分工：<span class="k">公共 API 是给人调的，虚表是给后端实现的。</span>'
];
defs.forEach((_, i) => tl.at(600 + i * 2500, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
}));
tl.at(14200, () => { els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L3-01 · 虚表的形状',
    title='★ <span class="hl-a">三张虚表</span> = 三个层次',
    sub='设备层回答"我是谁、能不能跑"，内存层回答"内存怎么分配"，计算层回答"图怎么算出来"。',
    caption='数出来的：device_i 15 个、buffer_type_i 6 个、backend_i 16 个函数指针。',
    src=IMPL, parts=[(1, 35)], duration=15000,
    mark_src=[3, 11, 17, 32, 33],
    notes_src={3: '文件自己声明：这是 internal header —— 公共头之外的"后端实现者契约"',
               11: '契约带版本号；L3-03 会看到动态加载后端时怎么校验它',
               32: '宿主对象把虚表放在第一个字段 —— 拿到 buft 就等于拿到了它的虚表',
               33: '每个 buffer type 记住自己属于哪个设备，L4-03 分配内存时用它',
               17: '一张虚表就是一个只装函数指针的 struct，可以 static const 初始化'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="layers" style="gap:8px"></div>
  <div class="formula" id="dia"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '设备层 · ggml_backend_device_i', m: '15 个函数指针',
    b: '发现与创建：我是谁 →<br>能不能跑这个算子 → 把我变成后端' },
  { c: 'c', t: '内存层 · ggml_backend_buffer_type_i', m: '6 个函数指针',
    b: '内存怎么分配：对齐多少、<br>多大的块、算不算 host 内存' },
  { c: 'b', t: '计算层 · ggml_backend_i', m: '16 个函数指针',
    b: '计算怎么执行：图怎么算出来、<br>数据怎么搬、什么时候同步' }
];
const host = wrap.querySelector('#layers');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const dia = wrap.querySelector('#dia');
dia.innerHTML = '<span class="m">struct ggml_backend_buffer_type_i</span> —— 只装函数指针的表<br>' +
  '&nbsp;&nbsp;&darr; <span class="k">作为第一个字段嵌进宿主对象</span><br>' +
  '<span class="m">struct ggml_backend_buffer_type</span> { <span class="v">iface</span>; <span class="v">device</span>; <span class="v">context</span>; }';
const msg = wrap.querySelector('#msg');
const texts = [
  '先看形状：每个 `_i` 结构体 <span class="k">只装函数指针</span>，不装数据。<br>数据在同名的宿主结构体里（iface + device + context）。',
  '<span class="v">设备层</span>：从"机器上有什么"出发，产出可执行的后端和内存类型。<br>L3-02 讲设备从哪来，L3-04 讲能力怎么探测。',
  '<span class="v">内存层</span>：只有一个职责 —— 把一段字节变成一块张量内存。<br>L4-03 会展开 buffer 与 buffer type 的分工。',
  '<span class="v">计算层</span>：L5-01 会从这里的 graph_compute 一路走到 CPU 内核的大 switch。',
  '一句话：<span class="k">新后端 = 填完这几张表</span>。表外的世界（图、张量、调度）不用重写。'
];
defs.forEach((_, i) => tl.at(600 + i * 2600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(9000, () => { msg.innerHTML = texts[3]; U.markLines(document, [0, 1, 2]); });
tl.at(12000, () => { els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4]; U.markLines(document, [0, 1, 2, 3, 4]); });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L3-01 · 设备层',
    title='★ <span class="hl-b">ggml_backend_device_i</span>：15 个函数指针',
    sub='自我介绍 5 个、创建与内存 4 个、能力判据 3 个、事件 3 个 —— 其中 6 个标了 (optional)。',
    caption='回顾 L2-03：权重用 mmap 直接映射进来时，走的就是这里的 buffer_from_host_ptr。',
    src=IMPL, parts=[(176, 218)], duration=18000,
    mark_src=[178, 184, 187, 193, 196, 205, 208, 212],
    notes_src={178: '自我介绍五件套从这里开始：name / description / memory / type / props，全部必需',
               184: 'get_memory 报告空闲与总字节数；注释说明返回 0 表示"不上报"',
               193: 'init_backend：设备 -> 后端。公共 API ggml_backend_dev_init() 转发到这里',
               196: 'get_buffer_type：设备首选的内存类型 —— 之后所有张量都分配在它上面',
               205: 'supports_op：调度器切图的判据 —— "这个算子我能不能跑"',
               208: 'supports_buft：反向判据 —— "这份权重所在的内存我能不能用"',
               212: 'offload_op 标了 (optional)：即使权重在别的内存里，也愿意把它搬过来算'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '自我介绍 ×5', m: 'get_name · get_description',
    b: 'get_memory · get_type · get_props<br><span class="k">全部必需</span>' },
  { c: 'c', t: '创建与内存 ×4', m: 'init_backend · get_buffer_type',
    b: 'get_host_buffer_type · buffer_from_host_ptr<br>后两个标了 (optional)' },
  { c: 'b', t: '能力判据 ×3', m: 'supports_op · supports_buft',
    b: 'offload_op（optional）<br><span class="k">调度器切图就靠这三个</span>' },
  { c: 'd', t: '事件 ×3', m: 'event_new · event_free',
    b: 'event_synchronize<br>整组 (optional)：不支持就不填' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '这一张表有 <span class="v">15 个函数指针</span>。按职责分四组，一眼能数完。',
  '<span class="v">自我介绍 5 个</span>（全部必需）：名字、描述、显存、类型、属性。<br>公共头把它们原样暴露成 ggml_backend_dev_name / memory / type / get_props。',
  '<span class="v">创建与内存 4 个</span>：<span class="k">init_backend 才是"设备 -> 可执行后端"的那一步</span>；<br>get_buffer_type 给出设备首选的内存类型。',
  '<span class="v">buffer_from_host_ptr</span> 标了 (optional)，注释写明用途：<br>memory mapped models —— 这正是 L2-03 里权重落位的那条路。',
  '<span class="v">能力判据 3 个</span>：supports_op / supports_buft 是必需，offload_op 是可选。<br>第 6 幕会看到：后端级同名公共 API 已被注释标记为"将移除"。',
  '事件三件套整组是 (optional)：<span class="k">不需要跨流同步的后端可以直接不填。</span>'
];
defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(12500, () => { msg.innerHTML = texts[4]; U.markLines(document, [3, 4, 5, 6]); });
tl.at(15500, () => { els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[5]; U.markLines(document, [0, 1, 2]); });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L3-01 · 计算层',
    title='★ <span class="hl-c">ggml_backend_i</span>：16 个函数指针，必需的只有 3 个',
    sub='get_name / free / graph_compute —— 其余 13 个都落在带 (optional) 的注释组里。',
    caption='数出来的必需项：16 - 13 = 3。异步五件套整组标了 (optional)。',
    src=IMPL, parts=[(121, 156)], duration=17000,
    mark_src=[122, 124, 131, 134, 138, 146, 155],
    notes_src={122: 'get_name：必需。调度器、日志、报错信息都用它标识后端',
               124: 'free：必需。设备层的 init_backend 负责产出后端，这里负责销毁它',
               131: '异步五件套（set/get 各两件 + cpy_tensor_async）整组标了 (optional)',
               134: 'synchronize：注释写明 —— 支持异步操作的后端就必须填它',
               138: 'graph_plan 四件套：注释直接写着 "not used currently"，是预留位',
               146: 'graph_compute：唯一必需的计算入口，返回 enum ggml_status',
               155: 'graph_optimize 也标了 (optional)：愿意重排节点就填'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', t: '必需 ×3', m: 'get_name · free',
    b: 'graph_compute<br><span class="k">唯一必需的计算入口</span>' },
  { c: 'a', t: '异步与同步 ×6', m: '五个 async 数据访问',
    b: 'synchronize：注释写明<br>"required if the backend supports async operations"' },
  { c: 'd', t: '预留与可选 ×7', m: 'graph_plan 四件套',
    b: 'event_record / event_wait / graph_optimize<br>graph plan 注释写 "not used currently"' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">16 个函数指针</span>，但数一遍带 (optional) 的注释组就知道：<br>真正非填不可的只有 <span class="k">3 个</span>。',
  '<span class="v">get_name</span> 是给人和日志看的；<span class="v">free</span> 是生命周期；<br><span class="v">graph_compute</span> 是整个后端的唯一计算入口。',
  '算一次图 = 调一次 <span class="v">graph_compute</span>，返回 <span class="v">enum ggml_status</span>。<br>L5-01 会从这里一路追到 CPU 内核的大 switch。',
  '五个 async 数据访问 + synchronize 是一组：<span class="k">要么都填，要么都不用。</span><br>不填时上层走 buffer 层的同步路径（第 5 幕）。',
  'graph_plan 四件套是历史预留位 —— 源码注释直说 "not used currently"。<br>读到这里可以放心跳过它们。',
  '结论：<span class="k">一个"能算图"的后端，计算层只要 3 个函数。</span>难的是算得对、算得快。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(10200, () => { msg.innerHTML = texts[3]; U.markLines(document, [2, 3]); });
tl.at(13200, () => { els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[5]; U.markLines(document, [0, 4]); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L3-01 · 内存层',
    title='<span class="hl-e">buffer type</span> 与 <span class="hl-e">buffer</span>：分配前与分配后',
    sub='buffer_type_i 有 6 个函数指针（3 必需），buffer_i 有 11 个（5 必需）；alloc_buffer 把前者变成后者。',
    caption='回顾 L1-01：ggml_tensor 的 buffer 字段指向的就是这里的 ggml_backend_buffer；init_tensor 会往 extra 字段填后端私有数据。',
    src=IMPL, parts=[(42, 67)], duration=17000,
    mark_src=[48, 50, 52, 54, 57, 62, 64, 66],
    notes_src={48: 'free_buffer 标了 (optional)：内存由外层统一管理时可以不填',
               50: 'get_base：必需。给出这块内存的起始地址',
               52: 'init_tensor：给张量补后端私有信息 —— 填的就是 L1-01 里的 extra 字段',
               54: 'memset / set / get：必需的三件数据搬运，签名都带 offset 与 size —— 可以只动张量的一段',
               57: '两个 2d 拷贝标了 (optional)：批量跨步拷贝的快捷方式',
               62: 'cpy_tensor 标了 (optional)，注释说明 src 可以来自任何后端的 buffer',
               64: 'clear：必需。整块内存刷成同一个字节值',
               66: 'reset：init_tensor 的反向操作，也标了 (optional)'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const dia = wrap.querySelector('#dia');

const bt = U.el('div', { class: 'card', style: 'width:300px' });
bt.innerHTML = '<div class="cm" style="margin:0 0 3px">分配前 · 内存的"型号"</div>' +
  '<div class="ct" style="color:var(--c)">ggml_backend_buffer_type_i</div>' +
  '<div class="cb"><span class="k">6 个函数指针</span><br>' +
  '必需 3：get_name · alloc_buffer · get_alignment<br>' +
  '可选 3：get_max_size · get_alloc_size · is_host</div>';

const bf = U.el('div', { class: 'card', style: 'width:300px' });
bf.innerHTML = '<div class="cm" style="margin:0 0 3px">分配后 · 一块具体的内存</div>' +
  '<div class="ct" style="color:var(--e)">ggml_backend_buffer_i</div>' +
  '<div class="cb"><span class="k">11 个函数指针</span><br>' +
  '必需 5：get_base · memset_tensor · set_tensor · get_tensor · clear<br>' +
  '可选 6：free_buffer · init_tensor · set_tensor_2d · get_tensor_2d · cpy_tensor · reset</div>';

dia.appendChild(bt); dia.appendChild(U.arrow('->')); dia.appendChild(bf);

const msg = wrap.querySelector('#msg');
const texts = [
  '一行看懂两者关系：<span class="v">buft-&gt;iface.alloc_buffer(buft, size)</span> 返回一个 buffer。',
  '<span class="v">buffer type</span> 是"型号"：对齐多少、最大能开多大、<br>算不算 host 内存（is_host 为真时数据能被 CPU 直接读）。',
  '<span class="v">buffer</span> 是"实物"：一块已经拿到的内存，外加怎么往里写、怎么读出来。<br>必需的 5 个全在数据面上。',
  'L1-01 里 ggml_tensor 的 <span class="v">buffer</span> 字段指向的就是这里的 buffer；<br><span class="v">init_tensor</span> 顺手把后端私有数据塞进 <span class="v">extra</span> 字段。',
  'L4-03 会把这两个虚表放在一起讲：<span class="k">什么时候选型号，什么时候开实物。</span>'
];
tl.at(700, () => { bt.style.opacity = '.35'; bf.style.opacity = '.35'; msg.innerHTML = texts[0]; });
tl.at(3400, () => { bt.style.opacity = '1'; bf.style.opacity = '.35'; msg.innerHTML = texts[1]; });
tl.at(6400, () => { bt.style.opacity = '.35'; bf.style.opacity = '1'; msg.innerHTML = texts[2]; });
tl.at(9600, () => { bt.style.opacity = '1'; bf.style.opacity = '1'; msg.innerHTML = texts[3]; });
tl.at(13000, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L3-01 · 两层分工',
    title='公共 API 是虚表的<span class="hl-a">镜像</span>',
    sub='模型代码调的是 ggml_backend_* 系列；它们只是把参数原样转发给某张虚表。',
    caption='注意第 107 行的注释：后端级的 supports_op / supports_buft / offload_op 将被移除 —— 能力判据归设备层。',
    src=PUB, parts=[(104, 118)], duration=16000,
    mark_src=[104, 107, 108, 109, 113, 118],
    notes_src={107: '注释写明：这三个后端级 API 将被移除，改用设备版本 —— 判据只存于 device_i',
               108: 'supports_op / supports_buft / offload_op 在 ggml_backend_i 里根本没有对应函数',
               113: '异步拷贝注释：不支持时自动回退成同步拷贝',
               118: '每个后端都能反查自己的设备：对应 impl 里 struct ggml_backend 的 device 字段'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['公共 API（本幕引用的这一段）', '真正干活的地方', '备注'],
  [['ggml_backend_graph_compute', 'backend_i.graph_compute', '第 4 幕第 146 行'],
   ['ggml_backend_graph_compute_async', 'backend_i.graph_compute', '异步由 synchronize 收口'],
   ['ggml_backend_supports_op', 'device_i.supports_op', '第 3 幕第 205 行'],
   ['ggml_backend_supports_buft', 'device_i.supports_buft', '第 3 幕第 208 行'],
   ['ggml_backend_offload_op', 'device_i.offload_op', '第 3 幕第 212 行（optional）'],
   ['ggml_backend_tensor_copy_async', 'backend_i.cpy_tensor_async', '不支持时回退同步'],
   ['ggml_backend_get_device', 'struct ggml_backend 的 device', '后端反查自己的设备']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '数出来的：<span class="v">ggml-backend.h 有 115 处 GGML_API</span>，几乎每个都能在虚表里找到落点。',
  '三个 compute 系列都落到同一个 <span class="v">graph_compute</span>。<br>区别只在"什么时候算完"：<span class="k">异步与否由 synchronize 收口。</span>',
  '注意 supports_op / supports_buft / offload_op 三行：<br>它们在 <span class="v">ggml_backend_i</span> 里<b>没有</b>对应函数，<span class="k">只在 device_i 里</span>。',
  '这就是第 107 行那句注释的含义：<span class="k">能力判据属于设备层。</span><br>第 4 幕那张 16 个指针的表里，确实一个 supports_* 都没有。',
  '最后一行是反向指针：后端知道自己属于哪个设备。<br>这条边让"调度器只拿到一个 backend"时仍能问出设备能力。',
  '一句话：<span class="k">读公共头能知道"能做什么"，读内部头才知道"谁来做的"。</span>'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2100, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(14000, () => { rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[5]; U.markLines(document, [3, 4, 5]); });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L3-01 · 注册面',
    title='还有一张表：<span class="hl-d">ggml_backend_reg_i</span>（4 个函数指针）',
    sub='一个后端 = 一组设备 + 一个版本号 + 一个可选扩展入口；它是 L3-02 与 L3-03 的接口。',
    caption='注册与设备枚举是下一课 L3-02 的主题；动态库导出的是本幕引用的这两个 typedef。',
    src=IMPL, parts=[(226, 254)], duration=15000,
    mark_src=[231, 234, 235, 239, 243, 251, 254],
    notes_src={231: 'get_name / get_device_count / get_device 三个必需 —— 设备枚举就靠它们',
               239: 'get_proc_address 标了 (optional)：后端私有扩展函数的唯一入口',
               243: '注释写明这个字段要初始化成 GGML_BACKEND_API_VERSION（第 2 幕第 11 行那个宏）',
               251: '动态加载时后端库必须导出 ggml_backend_init，返回的就是 ggml_backend_reg_t',
               254: '可选的 ggml_backend_score：分数高的后端优先，0 表示当前系统不支持'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'd', t: 'ggml_backend_reg_i ×4', m: 'get_name · get_device_count',
    b: 'get_device · get_proc_address(optional)<br><span class="k">后端 = 设备的集合</span>' },
  { c: 'a', t: 'struct ggml_backend_reg', m: 'api_version · iface · context',
    b: 'api_version 注释要求初始化成<br>GGML_BACKEND_API_VERSION（= 2）' },
  { c: 'b', t: '给下一课的接口', m: 'ggml_backend_init / _score',
    b: 'L3-02：谁把 reg 注册进全局表<br>L3-03：动态库怎么导出这两个符号' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '填完前面三张表，还差一步：<span class="k">让 ggml 知道你的存在。</span>',
  '<span class="v">reg_i 只有 4 个函数指针</span>：报名字、数设备、取第 i 个设备，<br>再加一个可选的后端私有扩展入口 get_proc_address。',
  '<span class="v">get_device_count + get_device</span> 就是设备发现的全部机制 ——<br>上层不需要知道设备是从 PCI 枚举来的还是写死的。',
  '<span class="v">api_version</span> 是契约的版本号：<br>它和 ggml_backend_init 一起构成动态加载时的握手（L3-03）。',
  '本课到此为止：<span class="k">接口长什么样已经全部看到。</span>下一课回答"它们从哪来、按什么顺序被选"。'
];
defs.forEach((_, i) => tl.at(600 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13200, () => { els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L3-01 · 落地',
    title='★ 实现一个新后端，最少要填哪些函数',
    sub='按虚表分组数一遍：设备 9 + 计算 3 + 内存类型 3 = 15 个必需项；加上数据面与注册面共 23 个。',
    caption='这张表就是本课的验收点；L8-03 会把它变成一份可勾选的"新后端清单"。',
    src=IMPL, parts=[(17, 29)], duration=19000,
    mark_src=[18, 20, 22, 24, 26, 28],
    notes_src={19: 'alloc_buffer 是这一层的核心：把字节数变成一块 buffer',
               23: '三个 (optional) 都有默认行为：max_size 默认 SIZE_MAX、alloc_size 默认 ggml_nbytes、is_host 默认 false',
               27: 'is_host 为真意味着"数据在系统内存里且布局标准" —— CPU 能直接读'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['虚表（层次）', '指针数', '必需', '可留空'],
  [['ggml_backend_device_i（发现与创建）', '15', '9', '6'],
   ['ggml_backend_i（计算执行）', '16', '3', '13'],
   ['ggml_backend_buffer_type_i（内存分配）', '6', '3', '3'],
   ['ggml_backend_buffer_i（内存读写）', '11', '5', '6'],
   ['ggml_backend_reg_i（注册）', '4', '3', '1'],
   ['合计', '52', '23', '29']],
  { monoCols: [0, 1, 2, 3] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '<span class="k">必需</span>= 不在任何 (optional) 注释组里；<span class="k">可留空</span>= 源码注释标了 (optional)。',
  '设备层必需 9：<span class="v">get_name · get_description · get_memory · get_type · get_props</span>'
  + ' + init_backend · get_buffer_type · supports_op · supports_buft。',
  '计算层必需 3：<span class="v">get_name · free · graph_compute</span>。<br>异步、图计划、事件、优化全部可选。',
  '内存类型层必需 3：<span class="v">get_name · alloc_buffer · get_alignment</span>；<br>max_size / alloc_size / is_host 都有默认行为。',
  '数据面必需 5：<span class="v">get_base · memset_tensor · set_tensor · get_tensor · clear</span>。<br>要在设备间搬张量，还得补 cpy_tensor 或异步版本。',
  '注册面必需 3：<span class="v">get_name · get_device_count · get_device</span>。<br>这一层怎么挂进全局表，是下一课 L3-02。',
  '验收答案：<span class="k">三张主虚表 15 个必需项</span>；算上数据面与注册面共 23 个。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 6)];
}));
tl.at(17500, () => { rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[6]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L3-01 · 收束',
    title='把这一课压成一张表',
    sub='五张虚表、52 个函数指针、23 个必需项；契约之外的一切都不用重写。',
    caption='下一课 L3-02：这些设备从哪来、按什么顺序被枚举。',
    src=PUB, parts=[(37, 43)], duration=20000,
    mark_src=[37, 38, 39, 40, 41, 42, 43],
    notes_src={41: '这七个公共函数一一对应 buffer_type_i 的 6 个函数指针，外加一个 get_device',
               43: 'get_device 暴露的是宿主对象里的 device 字段 —— 公共 API 只做转发'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['层次', '虚表', '本课第几幕', '在哪一课展开'],
  [['契约入口', '7 个不透明句柄', '1', '本课'],
   ['发现与创建', 'ggml_backend_device_i（15）', '3', 'L3-02 / L3-04'],
   ['计算执行', 'ggml_backend_i（16）', '4', 'L5-01 / L6 / L7'],
   ['内存分配', 'buffer_type_i（6）+ buffer_i（11）', '5', 'L4-03'],
   ['注册', 'ggml_backend_reg_i（4）', '7', 'L3-02 / L3-03'],
   ['落地清单', '23 个必需函数', '8', 'L8-03']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '不看代码回答：实现一个能真正跑图的新后端，最少要实现哪些接口函数？'
  + '（先按<b>层次</b>说，再补函数名）',
  '按三张主虚表：<br>'
  + '1) <span class="mono">ggml_backend_device_i</span> 里 9 个必需 —— '
  + '自我介绍 5（<span class="mono">get_name / get_description / get_memory / get_type / get_props</span>）'
  + ' + <span class="mono">init_backend</span> + <span class="mono">get_buffer_type</span>'
  + ' + <span class="mono">supports_op</span> + <span class="mono">supports_buft</span>；<br>'
  + '2) <span class="mono">ggml_backend_i</span> 里 3 个 —— '
  + '<span class="mono">get_name / free / graph_compute</span>；<br>'
  + '3) <span class="mono">ggml_backend_buffer_type_i</span> 里 3 个 —— '
  + '<span class="mono">get_name / alloc_buffer / get_alignment</span>。<br>'
  + '这三张表合计 <b>15</b> 个必需项。要真的搬数据，再加 '
  + '<span class="mono">ggml_backend_buffer_i</span> 的 5 个必需项（'
  + '<span class="mono">get_base / memset_tensor / set_tensor / get_tensor / clear</span>）；'
  + '要能被发现，再加 <span class="mono">ggml_backend_reg_i</span> 的 3 个必需项 —— '
  + '合计 <b>23</b> 个。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '五张虚表，52 个函数指针，23 个必需项 —— 这就是"后端"的全部义务。',
  '<span class="k">设备层</span>解决"发现与创建"：L3-02 讲注册与枚举顺序，L3-04 讲能力探测。',
  '<span class="k">计算层</span>只有 3 个必需函数，但 graph_compute 里面是整个后端的实现。',
  '<span class="k">内存层</span>是权重落位的地方：L2-03（mmap）与 L4-03（buffer 家族）都在这条线上。',
  '注册面把前四层挂进全局表；动态加载（L3-03）导出的是 ggml_backend_init。',
  '最后一句：<span class="v">写一个新后端，就是把这 23 个空填完</span> —— L8-03 会逐项对照。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i];
}));
tl.at(18500, () => { rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、为什么需要契约：公共头只有不透明句柄',
    '模型代码从头到尾只拿到指针，拿不到结构体。这不是封装洁癖，而是**二进制兼容**的代价控制：'
    '后端的结构体可以随便改，只要公共头里的 7 个 typedef 和函数签名不变。\n\n'
    '注意 `ggml_backend_graph_plan_t` 是个例外 —— 它是 `void *`，因为"执行计划"的形状'
    '完全由后端自己决定。',
    src=PUB, parts=[(24, 30)], lang='c')

L.section(
    '二、★ 三张虚表 = 三个层次',
    '虚表的形状是固定的：一个只装函数指针的 `struct`，名字以 `_i`（interface）结尾；'
    '同名的宿主结构体把这张表放在**第一个字段** `iface`，再挂上归属关系和私有 `context`。\n\n'
    '| 虚表 | 层次 | 回答的问题 | 函数指针 |\n|---|---|---|---|\n'
    '| `ggml_backend_device_i` | 设备 | 我是谁、能不能跑、怎么变成后端 | 15 |\n'
    '| `ggml_backend_buffer_type_i` | 内存 | 一块张量内存怎么分配 | 6 |\n'
    '| `ggml_backend_i` | 计算 | 一张图怎么算出来 | 16 |\n\n'
    '文件第一行就声明自己是 internal header；第 11 行的 `GGML_BACKEND_API_VERSION` 是'
    '这份契约的版本号，动态加载后端时要用（L3-03）。',
    src=IMPL, parts=[(1, 35)], lang='c')

L.section(
    '三、★ 设备层：ggml_backend_device_i 的 15 个函数指针',
    '把 15 个指针按职责分四组：自我介绍 5（`get_name` `get_description` `get_memory` `get_type` '
    '`get_props`）、创建与内存 4（`init_backend` `get_buffer_type` `get_host_buffer_type` '
    '`buffer_from_host_ptr`）、能力判据 3（`supports_op` `supports_buft` `offload_op`）、'
    '事件 3（`event_new` `event_free` `event_synchronize`）。\n\n'
    '源码注释给 6 个标了 `(optional)`：`get_host_buffer_type`、`buffer_from_host_ptr`、'
    '`offload_op` 和事件三件套。剩下的 9 个必须填。\n\n'
    '`buffer_from_host_ptr` 的注释写明用途是 "memory mapped models" —— 这正是 L2-03 里'
    '权重直接从磁盘映射进内存的那条路。',
    src=IMPL, parts=[(176, 218)], lang='c')

L.section(
    '四、★ 计算层：ggml_backend_i 的 16 个函数指针，必需只有 3 个',
    '16 个指针里，没有落在 `(optional)` 注释组里的只有 `get_name`、`free`、`graph_compute`：\n\n'
    '| 组 | 个数 | 说明 |\n|---|---|---|\n'
    '| 必需 | 3 | `get_name` / `free` / `graph_compute` |\n'
    '| 异步与同步 | 6 | 五个 async 数据访问 + `synchronize`（支持异步就必须填） |\n'
    '| 预留与可选 | 7 | `graph_plan` 四件套（注释写 "not used currently"）+ 事件两件 + '
    '`graph_optimize` |\n\n'
    '值得注意：这张表里**没有** `supports_op`。能力判据在设备虚表里 —— 见下一节。',
    src=IMPL, parts=[(121, 156)], lang='c')

L.section(
    '五、内存层（分配前）：ggml_backend_buffer_type_i',
    '6 个函数指针，3 个必需：`get_name`、`alloc_buffer`、`get_alignment`。'
    '三个可选的注释都给出了默认行为：`get_max_size` 默认 `SIZE_MAX`、'
    '`get_alloc_size` 默认 `ggml_nbytes`、`is_host` 默认 `false`。\n\n'
    '`alloc_buffer` 是这一层的产出：它把字节数变成一块 buffer。',
    src=IMPL, parts=[(17, 29)], lang='c')

L.section(
    '六、内存层（分配后）：ggml_backend_buffer_i',
    '11 个函数指针，5 个必需：`get_base`、`memset_tensor`、`set_tensor`、`get_tensor`、`clear`。'
    '可选的 6 个是 `free_buffer`、`init_tensor`、两个 2d 拷贝、`cpy_tensor`、`reset`。\n\n'
    '`init_tensor` 的注释写着 "eg. add tensor extras" —— 它填的正是 L1-01 里 `ggml_tensor` 的 '
    '`extra` 字段。回顾 L1-01：张量的 `buffer` 字段指向的就是这里的 `ggml_backend_buffer`。',
    src=IMPL, parts=[(42, 67)], lang='c')

L.section(
    '七、公共 API 是虚表的镜像',
    '这一段把"哪一层负责能力判据"讲透了：`ggml_backend_supports_op` / `ggml_backend_supports_buft` / '
    '`ggml_backend_offload_op` 三个后端级公共 API 上面有一行注释 —— "will be removed, use device '
    'version instead"。翻回计算层虚表（第 121-156 行）可以验证：那里确实一个 `supports_*` 都没有。\n\n'
    '最后一行 `ggml_backend_get_device` 是反向指针：从后端找回它的设备，'
    '对应内部头里 `struct ggml_backend` 的 `device` 字段。',
    src=PUB, parts=[(104, 118)], lang='c')

L.section(
    '八、注册面与动态加载入口',
    '`ggml_backend_reg_i` 只有 4 个函数指针：3 个必需（`get_name`、`get_device_count`、'
    '`get_device`）加 1 个可选（`get_proc_address`）。宿主结构体 `struct ggml_backend_reg` 的第一个'
    '字段是 `api_version`，注释要求初始化成 `GGML_BACKEND_API_VERSION`。\n\n'
    '这一段末尾还给出了动态加载的握手符号：后端动态库必须导出 `ggml_backend_init`'
    '（返回 `ggml_backend_reg_t`），可选导出 `ggml_backend_score`（分数高的后端优先，0 表示不支持）。'
    '注册流程与设备枚举顺序是 L3-02 的主题，动态库的符号解析是 L3-03 的主题。',
    src=IMPL, parts=[(226, 254)], lang='c')

L.section(
    '九、把"实现一个新后端"压成一张表',
    '把五个虚表的函数指针个数与必需项数一遍，就是本课验收点的答案：\n\n'
    '| 虚表（层次） | 指针数 | 必需 | 可留空 |\n|---|---|---|---|\n'
    '| `ggml_backend_device_i`（发现与创建） | 15 | 9 | 6 |\n'
    '| `ggml_backend_i`（计算执行） | 16 | 3 | 13 |\n'
    '| `ggml_backend_buffer_type_i`（内存分配） | 6 | 3 | 3 |\n'
    '| `ggml_backend_buffer_i`（内存读写） | 11 | 5 | 6 |\n'
    '| `ggml_backend_reg_i`（注册） | 4 | 3 | 1 |\n'
    '| **合计** | **52** | **23** | **29** |\n\n'
    '判据只有一条：**源码注释标了 `(optional)` 的可以留空，其余必须实现**。'
    '三张主虚表（前三行）合计 15 个必需项 —— 这就是"最少要写多少函数"的答案；'
    '要真的搬张量、要能被发现，就再加数据面 5 个与注册面 3 个。\n\n'
    '```text\n'
    '最小落地顺序（L8-03 会逐项对照）：\n'
    '  1. buffer_type_i: get_name / alloc_buffer / get_alignment\n'
    '  2. buffer_i:      get_base / memset_tensor / set_tensor / get_tensor / clear\n'
    '  3. backend_i:     get_name / free / graph_compute\n'
    '  4. device_i:      自我介绍 5 + init_backend / get_buffer_type / supports_op / supports_buft\n'
    '  5. reg_i:         get_name / get_device_count / get_device\n'
    '```')

L.footnote_add('本课逐字引用 `ggml/include/ggml-backend.h` 与 `ggml/src/ggml-backend-impl.h` 两个文件，'
               '两者都计入覆盖率。')
L.footnote_add('文中的函数指针个数、必需/可留空个数，均由脚本按 iface 结构体的起止行统计 '
               '`(*name)` 形态的成员，并以 "(optional)" 注释组为判据数出，不是凭印象写的。')
L.footnote_add('跨课指路只给课号与主题（L1-01 / L2-03 / L3-02 / L3-03 / L3-04 / L4-03 / L5-01 / L8-03），'
               '不引用它们的源码，故不计入本课覆盖率。')

L.prereqs('`L2-15`')

L.goal(
    '说出 `ggml_backend_device_i` / `ggml_backend_buffer_type_i` / `ggml_backend_i` 三张虚表'
    '各自对应哪一层次（对应验收点）；',
    '说出实现一个新后端最少要实现哪些函数：三张主虚表 9 + 3 + 3 = 15 个必需项，'
    '加上数据面 5 个与注册面 3 个共 23 个；',
    '解释为什么 `supports_op` / `supports_buft` / `offload_op` 只存在于设备虚表，'
    '而后端级的同名公共 API 被注释标记为"将移除"；',
    '说明 `buffer_type_i.alloc_buffer` 与 `buffer_i` 的分工，'
    '以及 `buffer_from_host_ptr` 与 mmap 权重加载（L2-03）的关系；',
    '复述 `iface` + 宿主对象（`iface` / `device` / `context`）的模式，'
    '并指出注册面（L3-02）与动态加载（L3-03）从哪里接进来。')

L.conclusion(
    '★ 三张虚表 = 三个层次',
    '| 虚表 | 层次 | 函数指针 | 必需 | 回答的问题 |\n|---|---|---|---|---|\n'
    '| `ggml_backend_device_i` | 设备 | 15 | 9 | 发现与创建 |\n'
    '| `ggml_backend_buffer_type_i` | 内存 | 6 | 3 | 内存怎么分配 |\n'
    '| `ggml_backend_i` | 计算 | 16 | 3 | 计算怎么执行 |\n\n'
    '另有数据面 `ggml_backend_buffer_i`（11 / 5）与注册面 `ggml_backend_reg_i`（4 / 3）。'
    '新后端只要填这几张表，图、张量、调度器都不用改。')

L.conclusion(
    '★ 最少 23 个必需项',
    '按"注释里有没有标 `(optional)`"数：设备 9 + 计算 3 + 内存类型 3 = **15**（三张主虚表）；'
    '再加数据面 5 与注册面 3，合计 **23** 个必需项、29 个可留空项。'
    '三张主虚表里最能省的是计算层 —— `graph_compute` 之外几乎全是可选项。')

L.conclusion(
    '虚表的形状：iface + 宿主对象',
    '每个 `_i` 结构体只装函数指针（可 `static const` 初始化、零运行时开销）；'
    '同名的宿主结构体把虚表放在第一个字段 `iface`，再接上归属关系（`device` / '
    '`reg`）与私有 `context`。拿到句柄就等于拿到了虚表 —— 这是整套接口的总模式。')

L.conclusion(
    '契约的边界：能力判据归设备层',
    '`supports_op` / `supports_buft` / `offload_op` 只在 `ggml_backend_device_i` 里；'
    '公共头里后端级的同名 API 带着注释 "will be removed, use device version instead"。'
    '这解释了后面几课的走向：调度器手里握着的是设备，不是后端（L3-02 / L4 层）。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
