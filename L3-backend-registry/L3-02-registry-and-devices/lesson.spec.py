#!/usr/bin/env python3
"""L3-02 · ★ 后端注册表与设备发现 —— 课件 spec。

运行：python3 L3-backend-registry/L3-02-registry-and-devices/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

REG   = 'ggml/src/ggml-backend-reg.cpp'
META  = 'ggml/src/ggml-backend-meta.cpp'
LLAMA = 'src/llama.cpp'

L = Lesson(
    id='L3-02',
    layer='L3 · 后端发现与注册',
    title='★ 后端注册表与设备发现',
    codecap='ggml/src/ggml-backend-reg.cpp / ggml-backend-meta.cpp（逐字引用）',
    nav={'prev': {'href': '../L3-01-backend-contract/index.html', 'label': 'L3-01 后端接口'},
         'next': {'href': '../L3-03-dynamic-loading/index.html', 'label': 'L3-03 动态加载后端'}},
)

L.note('**一句话**：`ggml_backend_registry` 就是**两个 `std::vector`** —— 一个装后端，一个装设备。'
       '这一课讲它们**被谁、按什么顺序**填满：**有哪些后端是编译期 `#ifdef` 决定的**，'
       '运行期只做动态加载补充；而**设备在数组里的下标，直接决定 offload 时“第一个 GPU”是谁**。')
L.note('阅读顺序：先看注册表结构（第 1 幕），再看静态注册的 16 个开关（第 2 幕）与注册动作（第 3 幕），'
       '然后是注册时机（第 4 幕）与设备枚举顺序（第 5 幕）—— 第 6 幕把“顺序”接到 offload 上，'
       '第 7 幕讲动态加载为什么只能追加，第 8、9 幕讲 Meta 设备与设备命名。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L3 · 后端发现与注册',
    title='注册表：这台机器上有什么',
    sub='结构体里只有两个 vector：后端列表，和它们贡献的设备列表。',
    caption='回顾 L3-01：`ggml_backend_reg_i` / `ggml_backend_device_i` 是后端要填的虚表 —— 这一课看它们怎么被收集起来。',
    src=REG, parts=[(110, 118)], duration=16000,
    mark_src=[111, 112, 116, 117],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">编译期 #ifdef</span><span class="arrow">-></span>
    <span class="chip a">register_backend()</span><span class="arrow">-></span>
    <span class="chip b">backends[] / devices[]</span><span class="arrow">-></span>
    <span class="chip c">llama.cpp 选设备</span>
  </div>
  <div class="row center" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '注册', b: '静态：构造函数里的 #ifdef<br>动态：load_backend() 之后追加', m: 'register_backend(reg)' },
  { c: 'b', t: '设备', b: '注册一个后端，顺带把它<br>报出来的设备登记进 devices', m: 'ggml_backend_dev_get(i)' },
  { c: 'c', t: '顺序', b: 'devices 的顺序 = 注册顺序，<br>它就是 offload 时“设备 0”的定义', m: 'devices[i]' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');
const msg = wrap.querySelector('#msg');
const texts = [
  '一个后端注册表项 = 后端句柄 + 动态库句柄（<span class="v">reg</span> + <span class="v">handle</span>）。',
  '注册表本体只有两个 <span class="k">std::vector</span>：<span class="v">backends</span> 装后端，<span class="v">devices</span> 装设备。<br>' +
    '<span class="k">没有排序、没有优先队列</span> —— 后面所有“顺序”问题都从这里来。',
  '<span class="v">register_backend()</span> 是唯一入口：静态注册和动态加载最后都走它，<br>所以两条路的差别只在“什么时候调用”。',
  '这一课的落点：<span class="k">devices[i] 的下标</span>如何一路变成 offload 时“哪张卡放哪些层”。',
  '下一幕先看静态注册：<span class="v">16 个 #ifdef</span> 就把这台机器“有哪些后端”定死了。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 2]); });
tl.at(3600, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [5, 6]); });
tl.at(6800, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [5, 6]); });
tl.at(10000, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, []); });
tl.at(13000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L3-02 · 核心',
    title='★ 有哪些后端，是<span class="hl-a">编译期</span>定下的',
    sub='注册表构造函数里 16 个 #ifdef；开关打开一个，就多注册一个后端。',
    caption='开关个数是数出来的：`grep -c "GGML_USE_" ggml/src/ggml-backend-reg.cpp` = 32 处；去重后 16 个（16 处 include + 16 处构造函数）。',
    src=REG, parts=[(119, 136)], duration=22000,
    mark_src=[120, 121, 123, 126, 129, 131],
    notes_src={129: '全文件唯一一处“编译进来了、运行期还能关掉”的开关：GGML_DISABLE_VULKAN（这个文件的 getenv 只有 131 与 601 两处）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const rows = [
  ['GGML_USE_CUDA',            '120', 'ggml_backend_cuda_reg()'],
  ['GGML_USE_METAL',           '123', 'ggml_backend_metal_reg()'],
  ['GGML_USE_SYCL',            '126', 'ggml_backend_sycl_reg()'],
  ['GGML_USE_VULKAN',          '129', 'ggml_backend_vk_reg()'],
  ['GGML_USE_WEBGPU',          '137', 'ggml_backend_webgpu_reg()'],
  ['GGML_USE_ZDNN',            '140', 'ggml_backend_zdnn_reg()'],
  ['GGML_USE_VIRTGPU_FRONTEND','143', 'ggml_backend_virtgpu_reg()'],
  ['GGML_USE_OPENCL',          '147', 'ggml_backend_opencl_reg()'],
  ['GGML_USE_ZENDNN',          '150', 'ggml_backend_zendnn_reg()'],
  ['GGML_USE_HEXAGON',         '153', 'ggml_backend_hexagon_reg()'],
  ['GGML_USE_CANN',            '156', 'ggml_backend_cann_reg()'],
  ['GGML_USE_BLAS',            '159', 'ggml_backend_blas_reg()'],
  ['GGML_USE_RPC',             '162', 'ggml_backend_rpc_reg()'],
  ['GGML_USE_OPENVINO',        '165', 'ggml_backend_openvino_reg()'],
  ['GGML_USE_ET',              '168', 'ggml_backend_et_reg()'],
  ['GGML_USE_CPU',             '171', 'ggml_backend_cpu_reg()']
];
const t = U.table(['#ifdef 开关（16 个）', '构造函数行', '注册入口'], rows, { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const trs = t.body.querySelectorAll('tr');
const msg = wrap.querySelector('#msg');
const texts = [
  '16 个开关，<span class="k">从上到下就是注册顺序</span>：CUDA 在最前（120 行），CPU 在最后（171 行）。',
  '前 4 个是 GPU 家族：<span class="v">CUDA</span> / <span class="v">METAL</span> / <span class="v">SYCL</span> / <span class="v">VULKAN</span>。<br>' +
    '注意 <span class="v">GGML_USE_CUDA</span> 这个宏在 HIP 构建（ROCm）和 MUSA 构建下也被打开 —— 见 L6-01。',
  '中间 6 个是各家加速器：<span class="v">WEBGPU</span> / <span class="v">ZDNN</span> / <span class="v">VIRTGPU</span> / <span class="v">OPENCL</span> / <span class="v">ZENDNN</span> / <span class="v">HEXAGON</span>。',
  '后 6 个：<span class="v">CANN</span>（昇腾 NPU，L7-01）/ <span class="v">BLAS</span> / <span class="v">RPC</span> / <span class="v">OPENVINO</span> / <span class="v">ET</span> / <span class="v">CPU</span>。',
  '宏是谁定义的？<span class="k">ggml/src/CMakeLists.txt:428-439</span>：只有 <span class="v">GGML_BACKEND_DL=OFF</span> 时才给 ggml 目标加这些宏。<br>' +
    '<span class="k">打开 DL 开关，这一整段 #ifdef 就全空了</span> —— 后端改由运行期加载（第 7 幕）。',
  '所以“机器上有哪些后端”有两个答案：<span class="k">编译期决定的静态集合</span> + 运行期追加的动态集合。'
];
function light(i) { trs.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; }); }
tl.at(700,  () => { light(0);  msg.innerHTML = texts[0]; });
tl.at(3600, () => { light(3);  msg.innerHTML = texts[1]; });
tl.at(6500, () => { light(8);  msg.innerHTML = texts[2]; });
tl.at(9400, () => { light(13); msg.innerHTML = texts[3]; });
tl.at(12300,() => { trs.forEach(r => { r.className = ''; }); msg.innerHTML = texts[4]; });
tl.at(15500,() => { light(15); msg.innerHTML = texts[5]; });
tl.at(19000,() => { trs.forEach(r => { r.className = ''; }); });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L3-02 · 注册动作',
    title='注册 = <span class="hl-c">追加到末尾</span> + 去重',
    sub='register_backend() 把后端 push_back，再按它报的设备数逐个 push_back 设备。',
    caption='实测（本机 CPU-only 构建，自建 harness 直接调 ggml_backend_register）：先注册名为 ZZZ 的假后端、再注册 AAA，得到 dev[1]=ZZZ0、dev[2]=AAA0 —— 顺序是注册顺序，不是字典序；重复注册同一个 reg，设备数不变。',
    src=REG, parts=[(186, 205)], duration=18000,
    mark_src=[187, 191, 192, 201, 202, 203],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:6px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:2px">devices[] 的生长过程（本机实测）</div>';
const steps = [
  { t: '初始',                 v: '[]' },
  { t: 'register(cuda_reg)',   v: '[CUDA0, CUDA1]' },
  { t: 'register(cpu_reg)',    v: '[CUDA0, CUDA1, CPU]' },
  { t: '再 register(cuda_reg)',v: '不变（去重）' },
  { t: 'register(zzz), register(aaa)', v: '[..., ZZZ0, AAA0]' }
];
const els = steps.map(s => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
  e.innerHTML = '<span class="m">' + U.esc(s.t) + '</span> <span class="arrow">-></span> ' +
                '<span style="color:var(--b)">' + U.esc(s.v) + '</span>';
  left.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">追加，不插入</div>' +
  '<div class="cb">两个循环都只有 <span class="cm" style="margin:0">push_back</span>（201、203 行）。' +
  '没有排序、没有按名字或显存重排 —— <b>谁先注册，谁的下标就小</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)">' +
  '<div class="ct" style="color:var(--d)">后端可以去重，设备也可以</div>' +
  '<div class="cb">同一个 <span class="cm" style="margin:0">reg</span> 注册两次直接 return（191-195）；' +
  '同一个设备指针也去重（207-212，源码里的 <span class="cm" style="margin:0">register_device</span>）。</div></div>' +
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">可以注册成功、却贡献 0 个设备</div>' +
  '<div class="cb">设备循环的次数来自后端自报的 <span class="cm" style="margin:0">ggml_backend_reg_dev_count(reg)</span>。' +
  'RPC 后端的设备数来自它的 context（<span class="cm" style="margin:0">ggml-rpc.cpp:2273-2276</span>）：' +
  '还没加服务器时就是 0。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '先看第 187-189 行的空指针守卫，再看第 191-195 行的去重 —— 然后才是真正的注册。',
  '后端入列：<span class="v">backends.push_back({ reg, handle })</span>。',
  '设备入列：<span class="v">for (i &lt; ggml_backend_reg_dev_count(reg))</span> 逐个 <span class="v">register_device(...)</span>（202-204）。',
  '于是 <span class="k">一个后端在后端列表里的位置，决定了它的设备在设备列表里的位置</span>。',
  '这就是第 6 幕要用到的全部机制：<span class="k">devices[] 的下标 = 注册顺序</span>。'
];
tl.at(700,  () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; U.markLines(document, [1, 5]); });
tl.at(3700, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [15]); });
tl.at(6700, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [16, 17]); });
tl.at(9700, () => { els[3].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [5]); });
tl.at(12700,() => { els[4].style.opacity = '1'; msg.innerHTML = texts[4]; U.markLines(document, []); });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L3-02 · 时机',
    title='<span class="hl-e">get_reg()</span>：注册发生在第一次触碰注册表时',
    sub='函数内 static 单例；静态后端在构造函数里就位，动态加载只是随后补票。',
    caption='实测：本机 CPU-only 构建里，调用 ggml_backend_load_all() 之前 ggml_backend_reg_count() 已经是 1（reg[0] = CPU）。',
    src=REG, parts=[(292, 300)], duration=16000,
    mark_src=[292, 293, 298, 299],
    notes_src={293: '函数内 static：第一次调用时构造 —— “这台机器有哪些后端”就是在这一行被回答的'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow wrap" style="justify-content:center">
    <span class="chip">进程启动</span><span class="arrow">-></span>
    <span class="chip a">第一次调用 ggml_backend_*</span><span class="arrow">-></span>
    <span class="chip b">get_reg() 构造 registry</span><span class="arrow">-></span>
    <span class="chip c">16 个 #ifdef 注册静态后端</span><span class="arrow">-></span>
    <span class="chip d">load_all() 追加动态后端</span>
  </div>
  <div class="row center" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'c', t: '注册表是懒的', b: '没有任何全局初始化函数：<br>第一次触碰才构造，构造即注册。' },
  { c: 'd', t: '空注册表的下场', b: 'src/llama.cpp:406-408 直接报错：<br>no backends are loaded' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">get_reg()</span> 返回一个函数内 <span class="k">static</span> 注册表（292-295）—— C++ 保证它只被构造一次。',
  '构造函数（第 2 幕那段 #ifdef）<span class="k">在任何动态加载之前</span>就想好了静态集合。<br>' +
    '本机实测：<span class="v">load_all()</span> 之前 <span class="v">reg_count()</span> 已经是 1。',
  '<span class="v">ggml_backend_register()</span> / <span class="v">ggml_backend_device_register()</span>（298-304）是内部 API：<br>' +
    '树外的后端（比如 RPC 服务器，<span class="v">common/arg.cpp:1180</span>）也能把自己塞进同一个注册表。',
  '于是“有哪些后端”= <span class="k">编译期静态集合</span> ∪ <span class="k">运行期加载的集合</span>，且前者总是先来。',
  '顺序也由此定死：<span class="k">静态后端永远排在动态后端前面</span>。'
];
tl.at(700,  () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1]); });
tl.at(3700, () => { msg.innerHTML = texts[1]; U.markLines(document, [0]); });
tl.at(6700, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [6, 7, 8]); });
tl.at(9700, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, []); });
tl.at(12700,() => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L3-02 · 设备枚举',
    title='设备枚举：<span class="hl-b">下标就是设备号</span>',
    sub='ggml_backend_dev_get(i) 返回 devices[i]；顺序由注册顺序决定，与名字无关。',
    caption='按名字查找走 by_name（大小写不敏感的比较在 307-314 的 striequals，返回第一个匹配的循环在 345-353）。',
    src=REG, parts=[(336, 343)], duration=16000,
    mark_src=[336, 337, 340, 341, 342],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row center" id="strip" style="gap:7px"></div>
  <div class="row center" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const strip = wrap.querySelector('#strip');
const items = [
  { k: 'dev[0]', v: 'CPU',  c: 'a' },
  { k: 'dev[1]', v: 'ZZZ0', c: 'c' },
  { k: 'dev[2]', v: 'AAA0', c: 'd' }
];
const els = items.map(it => {
  const e = U.el('div', { class: 'card', style: 'width:150px;border-left-color:var(--' + it.c + ')' });
  e.innerHTML = '<div class="cm" style="margin:0">' + U.esc(it.k) + '</div>' +
                '<div class="ct" style="color:var(--' + it.c + ');font-size:14px">' + U.esc(it.v) + '</div>';
  strip.appendChild(e);
  return e;
});
strip.insertBefore(U.el('div', { class: 'cm', style: 'margin-right:2px' }), els[0]);
strip.firstChild.innerHTML = '本机实测：<br>先注册 CPU，<br>再注册 ZZZ、AAA';
els.forEach(e => e.style.opacity = '.30');

const cards = wrap.querySelector('#cards');
const defs = [
  { c: 'b', t: '下标从 0 开始数', b: 'dev_count() / dev_get(i) 就是<br>对 devices 向量的直接转发。' },
  { c: 'e', t: '名字不参与排序', b: 'AAA 排在 ZZZ 之后 ——<br>顺序只跟“谁先注册”有关。' }
];
const cels = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); cards.appendChild(e); return e; });
cels.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '设备枚举有两个入口：<span class="v">ggml_backend_dev_count()</span> 与 <span class="v">ggml_backend_dev_get(i)</span>。',
  '两者都只是转发到注册表的 <span class="v">devices.size()</span> / <span class="v">devices[index]</span> —— <span class="k">顺序在这里没有被重新计算的机会</span>。',
  '本机实测（第 3 幕那个 harness）：先注册的后端，它的设备下标就小。<br>名字叫 AAA 也不会被排到前面。',
  '<span class="v">ggml_backend_dev_by_type()</span>（355-363）与 <span class="v">by_name()</span>（345-353）都是<b>从头扫、返回第一个匹配</b>，<br>所以“同一个类型有多块设备时返回谁”，答案还是顺序。',
  '记住：<span class="k">这里的下标不是硬件编号，是注册顺序</span>。下一幕看它的后果。'
];
tl.at(700,  () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1]); });
tl.at(3600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [4, 5, 6]); });
tl.at(6600, () => { els[1].style.opacity = '1'; els[2].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, []); });
tl.at(9600, () => { cels[0].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, []); });
tl.at(12600,() => { cels[1].style.opacity = '1'; msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L3-02 · 后果',
    title='★ 注册顺序怎么变成 <span class="hl-a">offload 结果</span>',
    sub='llama.cpp 按注册顺序把 GPU 型设备收进 model->devices；tensor-split / main-gpu 的下标就是它。',
    caption='接着看 L2-05 逐字引用过的 src/llama-model.cpp:1521-1542：splits[] 经 upper_bound() 定出每层的设备，写进 dev_layer[il]。',
    src=LLAMA, parts=[(222, 231)], duration=22000,
    mark_src=[222, 223, 224, 230],
    notes_src={223: '按注册顺序遍历设备 —— “设备 0/1/2”这个编号就是在这里被消费的'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow wrap" style="justify-content:center">
    <span class="chip a">devices[i]</span><span class="arrow">-></span>
    <span class="chip b">gpus[] 按遇到顺序 push_back</span><span class="arrow">-></span>
    <span class="chip c">model-&gt;devices</span><span class="arrow">-></span>
    <span class="chip d">main_gpu / tensor_split 下标</span><span class="arrow">-></span>
    <span class="chip e">dev_layer[il]</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', t: '① 收集', b: 'GPU 型设备按遇到的顺序 <span class="cm" style="margin:0">gpus.push_back</span>（254）；<br>RPC 设备单独收集。' },
  { c: 'c', t: '② 排序', b: 'model-&gt;devices = RPC 前置（276）+ GPU 追加（279）；<br>没有独显时才补 iGPU（283-285）。' },
  { c: 'd', t: '③ 消费', b: 'split-mode=none 只留 <span class="cm" style="margin:0">devices[main_gpu]</span>（297-299）；<br>layer/row 模式按这个下标切层。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '注册表给的顺序，第一个消费者就是 llama.cpp 的默认设备选择：<span class="k">从 i = 0 扫到 dev_count()-1</span>。',
  '<span class="v">case GGML_BACKEND_DEVICE_TYPE_GPU</span>（230）：GPU 型设备被收进 <span class="v">gpus</span>；<br>' +
    'CPU / ACCEL 型被跳过（225-228）—— 它们由 CPU buffer 列表单独处理（L2-05）。',
  '注意 269-270 行：这个循环里遇到 <span class="v">META</span> 型设备会 <span class="k">GGML_ABORT("fatal error")</span> —— ' +
    '因为 Meta 设备根本不在注册表里（第 8 幕）。',
  '接着：<span class="v">model->devices</span> 的顺序 = 注册表里 GPU 设备的顺序；<br>' +
    '<span class="v">--main-gpu N</span> 是 <span class="v">model->devices</span> 的下标（297-299），<span class="v">--tensor-split</span> 的第 j 项对应 <span class="v">devices[j]</span>（llama-model.cpp:1488-1508）。',
  '最后落到层：<span class="v">i_gpu_start</span> 划出上 GPU 的层，<span class="v">upper_bound(splits, ...)</span> 决定每层给哪块卡（llama-model.cpp:1529），' +
    '结果写进 <span class="v">dev_layer[il]</span>（1540-1542），权重张量再按它选 buffer 类型（L2-03）。',
  '所以：<span class="k">同一份 --tensor-split 3,1，在“设备顺序不同”的两台机器上，会把层分到不同的卡上</span>。',
  '顺序还会被写死的两张表改写：静态注册表与动态加载表顺序不同（下一幕），<span class="v">CANN</span> 与 <span class="v">CUDA</span> 就是一个真实的反例。'
];
tl.at(700,  () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1]); });
tl.at(3600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [8, 9]); });
tl.at(6600, () => { msg.innerHTML = texts[2]; U.markLines(document, []); });
tl.at(9600, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [9]); });
tl.at(12600,() => { els[2].style.opacity = '1'; msg.innerHTML = texts[4]; });
tl.at(15600,() => { msg.innerHTML = texts[5]; });
tl.at(18600,() => { msg.innerHTML = texts[6]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L3-02 · 动态加载',
    title='★ 动态加载只能<span class="hl-c">追加</span>到静态后端之后',
    sub='load_all_from_path() 按写死的顺序找 15 个名字的动态库；找到的都 push_back 到数组末尾。',
    caption='动态库的 dlopen/符号解析/版本校验细节在 L3-03；本幕只关心它对“顺序”的影响。',
    src=REG, parts=[(578, 604)], duration=20000,
    mark_src=[579, 583, 585, 589, 593, 599, 601, 603],
    notes_src={579: 'Release（NDEBUG）构建下 silent = true：加载失败的日志被吞掉',
               601: 'GGML_BACKEND_PATH：出树（out-of-tree）后端的入口，唯一由用户指定路径的加载'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['', '静态注册（构造函数）', '动态加载（load_all）'],
  [['谁决定', '编译期宏 GGML_USE_*', '运行期磁盘上有什么 .so'],
   ['时机', '第一次触碰注册表时', 'llama_backend_init() 之后'],
   ['顺序', 'CUDA→METAL→SYCL→VULKAN…→CPU（120-171）', 'blas→zendnn→cann→cuda→hip→…→cpu（585-599）'],
   ['失败表现', '根本不存在这个后端', '静默跳过（Release 下 silent=true）'],
   ['进了哪个数组', 'backends[] / devices[] 的头部', '同一个数组的尾部（push_back）']],
  { monoCols: [] });
wrap.querySelector('#tbl').appendChild(t.el);
const trs = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">ggml_backend_load_all()</span> 只是 <span class="v">load_all_from_path(nullptr)</span> 的转发（574-576）。',
  '搜索路径：<span class="v">GGML_BACKEND_DIR</span>（编译期，若定义）→ 可执行文件目录 → 当前目录（488-499）；<br>' +
    '每个路径下找 <span class="v">libggml-&lt;name&gt;-*.so</span>（464-478 的前后缀），按 <span class="v">ggml_backend_score()</span> 选分最高的（504-543）。',
  '候选名字 <span class="k">15 个</span>，顺序写死：blas, zendnn, cann, cuda, hip, metal, rpc, sycl, vulkan, virtgpu, opencl, hexagon, musa, openvino, cpu。<br>' +
    '（命令数出来：<span class="v">grep -c \'ggml_backend_load_best("\' ggml/src/ggml-backend-reg.cpp</span> = 15）',
  '★ 关键：<span class="k">这张表与构造函数的 #ifdef 顺序不是同一张表</span>。<br>' +
    '例如 <span class="v">CANN</span>：静态顺序在 CUDA 之后（156 vs 120），动态顺序在 CUDA 之前（587 vs 588）。',
  '而 <span class="v">register_backend()</span> 只做 push_back（201）—— 所以“同一个后端是编译进来的还是加载进来的”，<br>会改变它在 devices[] 里的下标。',
  '一个只在运行期出现的后端，<span class="k">不可能挤到静态后端前面</span>；要改设备编号，只能改编译期开关的组合。'
];
function light(i) { trs.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; }); }
tl.at(700,  () => { msg.innerHTML = texts[0]; light(0); });
tl.at(3600, () => { msg.innerHTML = texts[1]; light(1); });
tl.at(6600, () => { msg.innerHTML = texts[2]; light(2); });
tl.at(9600, () => { msg.innerHTML = texts[3]; light(2); });
tl.at(12600,() => { msg.innerHTML = texts[4]; light(4); });
tl.at(15600,() => { trs.forEach(r => { r.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L3-02 · Meta',
    title='<span class="hl-e">Meta</span> backend：把 N 张卡合成一个设备',
    sub='ggml_backend_meta_device() 造出来的设备不在注册表里：reg = nullptr。',
    caption='它由调用方按需创建：张量并行（split-mode = tensor）时，llama.cpp 把枚举到的设备打包成一个 Meta 设备（src/llama.cpp:193-220）。',
    src=META, parts=[(215, 245)], duration=20000,
    mark_src=[216, 219, 220, 229, 237, 239],
    notes_src={218: '源码自己标注：这个函数不是线程安全的',
               239: 'reg = nullptr —— 它不属于任何后端注册表项，因此永远不出现在 ggml_backend_dev_get() 的结果里'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '它是聚合', b: 'simple_devs 是入参那串设备（54-57）；<br>Meta 设备的全部行为都转发给它们。' },
  { c: 'b', t: '内存 = 相加', b: 'get_memory 把各设备的 free/total<br>累加（99-110）—— 报出来的是总和。' },
  { c: 'c', t: '能力 = 取交集', b: 'supports_op 用 <span class="cm" style="margin:0">all_of</span>（154-159）：<br>只有每个简单设备都支持，Meta 才支持。' },
  { c: 'd', t: '类型是 META', b: 'get_type 恒返回<br><span class="cm" style="margin:0">GGML_BACKEND_DEVICE_TYPE_META</span>（112-116）。' },
  { c: 'e', t: '同一组设备只造一次', b: '静态 map 去重（229-234）：<br>同样的设备组合返回同一个指针。' },
  { c: 'f', t: '谁在用', b: '张量并行需要“一个逻辑设备跨多张卡”，<br>上层算子不用知道几条流水线（L4-02）。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '第 8 幕起讲一个“不是注册表成员”的设备：<span class="v">Meta</span>。',
  '它由 <span class="v">ggml_backend_meta_device(devs, n_devs, ...)</span> 现场造出来（215-216），<br>' +
    '<span class="k">全文件没有任何 ggml_backend_register() 调用</span> —— 这是它和第 2 幕那些后端的根本区别。',
  '它内部的 <span class="v">ggml_backend_device</span> 结构体里 <span class="v">reg = nullptr</span>（237-241）：<br>设备存在，但不挂在任何后端项下。',
  '于是 <span class="v">ggml_backend_dev_count()</span> 永远数不到它（第 5 幕），<br>而 llama.cpp 的设备选择循环里遇到 META 型设备会 abort（src/llama.cpp:269-270）。',
  '它的价值：把“数据怎么切到多张卡”这件事<b>藏在一个设备后面</b>；<br>调度器仍然以为自己在跟一个设备打交道（L4-02）。拆开看是 <span class="v">simple_devs</span>。'
];
tl.at(700,  () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1, 2]); });
tl.at(3600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [3, 5]); });
tl.at(6600, () => { els[3].style.opacity = '1'; els[4].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [23, 24, 25, 27, 28]); });
tl.at(9600, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [23, 24, 25, 27, 28]); });
tl.at(12600,() => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; });
tl.at(15600,() => { els[5].style.opacity = '1'; msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L3-02 · 命名',
    title='多后端共存时，设备名是谁起的',
    sub='注册表不发明名字：它只转手后端给的 get_name；Meta 设备的名字是拼出来的。',
    caption='实例：CUDA 后端用 `GGML_CUDA_NAME + 序号`（ggml-cuda.cu:5798），该宏在 HIP 构建下是 "ROCm"、MUSA 下是 "MUSA"（ggml/include/ggml-cuda.h:11/14/17）；CPU 设备固定叫 "CPU"（ggml-cpu.cpp:353-357）；Metal 是 "MTL" + 序号（ggml-metal.cpp:307）。',
    src=META, parts=[(65, 77)], duration=18000,
    mark_src=[65, 66, 67, 71, 72, 75, 76],
    notes_src={72: '简单设备的名字按登记顺序用逗号连接 —— 名字里保留了“谁在左”的信息'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="strip" style="gap:8px"></div>
  <div class="row center" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const strip = wrap.querySelector('#strip');
strip.innerHTML =
  '<span class="chip a">"CUDA0"</span><span class="arrow">,</span>' +
  '<span class="chip a">"CUDA1"</span><span class="arrow">-></span>' +
  '<span class="chip e" style="font-size:12px">"Meta(CUDA0,CUDA1)"</span>';
const chipEls = strip.querySelectorAll('.chip');

const cards = wrap.querySelector('#cards');
const defs = [
  { c: 'b', t: '名字由后端自报', b: '注册表只用 <span class="cm" style="margin:0">ggml_backend_dev_name()</span> 转发<br>（215 行的日志、by_name 的查找）。' },
  { c: 'c', t: '查找：大小写不敏感、取第一个', b: '<span class="cm" style="margin:0">striequals</span>（307-314）+ 从头扫（345-353）：<br>重名时后面的设备永远查不到。' }
];
const cels = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); cards.appendChild(e); return e; });
cels.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  'Meta 设备的名字不是常量，是构造出来的：<span class="v">"Meta(" + 名字 + "," + 名字 + ")"</span>。',
  '注意 name（65-72）与 description（66-73）是两份不同的字符串：<br>前者是标识符，后者给日志看。',
  '<span class="v">Meta(CUDA0,CUDA1)</span> 里的顺序 = 传给工厂函数的设备顺序 = <span class="k">注册表里的顺序</span>。' +
    'meta buffer type 也照同样的规则拼（256-268，用 buffer type 的名字）。',
  '普通后端同理：<span class="v">CUDA0</span> 这个名字来自 CUDA 后端自己（ggml-cuda.cu:5798），' +
    '<span class="k">它在注册表里的下标与名字里的 0 没有必然关系</span>。',
  '实践建议：要钉住某块卡，用名字而不是下标 —— <span class="v">--device</span> 内部就是 by_name（common/arg.cpp:1116-1136），<br>它不随注册顺序漂移。'
];
tl.at(700,  () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1, 2]); });
tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [2, 8, 10]); });
tl.at(6600, () => { cels[0].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [5, 6, 7]); });
tl.at(9600, () => { chipEls.forEach(c => { c.style.opacity = '1'; }); msg.innerHTML = texts[3]; U.markLines(document, []); });
tl.at(12600,() => { cels[1].style.opacity = '1'; msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L3-02 · 收束',
    title='把这一课压成一张表',
    sub='三句话：有哪些后端看编译期；谁是设备 0 看注册顺序；Meta 设备不在注册表里。',
    caption='下一课 L3-03 讲动态加载的细节：dlopen、符号解析、版本校验。',
    src=REG, parts=[(382, 390)], duration=20000,
    mark_src=[383, 384, 385],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', '答案', '依据'],
  [['机器上有哪些后端？', '编译期 16 个 GGML_USE_* 决定；运行期动态加载补充', 'reg.cpp:119-173 + CMakeLists.txt:428-439'],
   ['谁在什么时候注册？', '第一次触碰注册表时，构造函数里完成', 'reg.cpp:292-295'],
   ['devices[] 的顺序？', '注册顺序（静态在前、动态在后），只追加不重排', 'reg.cpp:201-204'],
   ['“第一个 GPU”是谁？', '注册顺序里第一个 GPU 型设备 → model->devices[0]', 'src/llama.cpp:222-231 / 254 / 276-279'],
   ['device 0 影响什么？', 'main-gpu 的下标、tensor-split 的第 0 项、每层落哪张卡', 'src/llama.cpp:297-299 + llama-model.cpp:1529'],
   ['设备名谁起？', '后端自己的 get_name；Meta 设备名是拼出来的', 'meta.cpp:65-77 / ggml-cuda.cu:5798'],
   ['Meta 设备在哪？', '不在注册表：工厂直接造，reg = nullptr', 'meta.cpp:215-245']],
  { monoCols: [2] });
wrap.querySelector('#tbl').appendChild(t.el);
const trs = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '一台机器上有两张 NVIDIA 卡。构建 A 打开了 <span class="mono">GGML_USE_CUDA</span>（静态注册），' +
  '构建 B 只静态编译了 CPU，CUDA 由运行期动态加载。为什么两次运行的 <span class="mono">--tensor-split</span> 可能指向不同的卡？' +
  '要钉住某一张卡，应该用什么？',
  '关键在 <span class="mono">devices[]</span> 里 GPU 设备的<b>顺序</b>：<br>' +
  '· 构建 A：CUDA 后端在构造函数里注册（<span class="mono">reg.cpp:120-122</span>），它的卡先进数组；<br>' +
  '· 构建 B：CUDA 由 <span class="mono">load_backend()</span> 追加（<span class="mono">reg.cpp:201</span>），排在所有静态后端之后。<br>' +
  'llama.cpp 按 <span class="mono">ggml_backend_dev_get(i)</span> 的顺序收集 GPU（<span class="mono">src/llama.cpp:222-231, 254</span>），' +
  '<span class="mono">--main-gpu N</span> 与 <span class="mono">--tensor-split</span> 的第 j 项都是 <span class="mono">model->devices</span> 的下标' +
  '（<span class="mono">297-299</span>），最终决定每层落在哪张卡（<span class="mono">llama-model.cpp:1529</span>）。<br>' +
  '机器上还有第二种 GPU 后端在册时（例如 CANN 与 CUDA 同时构建），静态与动态两张顺序表不同这一点就会直接改变映射。' +
  '所以要用<b>名字</b>钉住设备：<span class="mono">--device CUDA1</span>，它内部走 ' +
  '<span class="mono">ggml_backend_dev_by_name()</span>（<span class="mono">reg.cpp:345-353</span>，经 <span class="mono">common/arg.cpp:1127</span>）。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '先补一句本节引用的代码：<span class="v">ggml_backend_init_best()</span> 是按<b>类型</b>挑最合适的设备（GPU → IGPU → CPU），',
  '—— 这是“不想管顺序”时的兜底入口；而 llama.cpp 的默认设备列表是按<b>顺序</b>建的。两条路都存在，别混淆。',
  '一句话收束：<span class="k">编译期决定“有哪些”，注册顺序决定“谁是 0”</span>。',
  '而顺序规则很朴素：<span class="v">register_backend() 只做 push_back</span>，先静态后动态。',
  'L2-03 讲权重落位、L2-05 讲 dev_layer、L4-02 讲调度器怎么按设备列表切图 —— 它们都吃这一课的输出。'
];
tl.at(700,  () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 2, 3]); });
tl.at(3700, () => { msg.innerHTML = texts[1]; U.markLines(document, []); });
trs.forEach((r, i) => tl.at(6400 + i * 1600, () => {
  trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[2];
}));
tl.at(17800, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
tl.at(19000, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、注册表：两个 vector',
    '`ggml_backend_registry` 的全部状态就是两个 `std::vector`：一个装后端注册项，一个装设备。'
    '后端项多带一个动态库句柄 —— 这是第 7 幕要用的东西。\n\n'
    '注意这里**没有**任何排序字段、优先级或哈希：`devices` 的顺序完全由“谁先被 push_back”决定，'
    '而这正是第 5、6 幕的主题。',
    src=REG, parts=[(110, 118)], lang='c')

L.section(
    '二、★ 静态注册：构造函数里的 16 个开关',
    '注册表的构造函数是一串 `#ifdef GGML_USE_*`（120-173 行）。开关一打开，对应的后端就在'
    '**第一次触碰注册表时**被注册进来。\n\n'
    '开关个数（命令口径）：\n\n'
    '```text\n'
    '$ grep -c "GGML_USE_" ggml/src/ggml-backend-reg.cpp\n'
    '32\n'
    '$ grep -o "GGML_USE_[A-Z0-9_]*" ggml/src/ggml-backend-reg.cpp | sort -u | wc -l\n'
    '16\n'
    '```\n\n'
    '32 = 16 处 include 守卫（29-91 行）+ 16 处构造函数守卫（120-173 行）。'
    '完整清单与行号见第 2 幕的表格。\n\n'
    '这些宏是 CMake 加的：`ggml/src/CMakeLists.txt:428-439` 的 `ggml_add_backend()` 里，'
    '只有 `GGML_BACKEND_DL=OFF` 时才执行 `target_compile_definitions(ggml PUBLIC GGML_USE_<BACKEND>)`。'
    '**打开 `GGML_BACKEND_DL`，这一段 `#ifdef` 就全空了** —— 后端改由运行期加载（第 7 幕、L3-03）。',
    src=REG, parts=[(119, 136)], lang='c')

L.section(
    '三、注册动作：追加 + 去重',
    '`register_backend()` 只做两件事：把后端 `push_back` 到 `backends`，'
    '再按这个后端自报的设备数，把设备逐个交给 `register_device()`。\n\n'
    '两个循环都**只有 `push_back`**，两侧都有“已经存在就直接 return”的去重（191-195、207-212）。'
    '去重按指针比较，不做任何排序或合并 —— 所以两个不同后端各自报出的设备会依次排在数组里。',
    src=REG, parts=[(186, 205)], lang='c')

L.section(
    '四、单例与内部 API：注册发生在什么时候',
    '`get_reg()` 是一个函数内 `static` 单例：**注册不是启动时的一次全局初始化，而是第一次触碰注册表时发生的**。'
    '`ggml_backend_load_all()` 与所有 `ggml_backend_*` 查询最终都会先走到这里。\n\n'
    '实测（本机 CPU-only 构建，`-DGGML_USE_CPU`）：\n\n'
    '```text\n'
    'reg_count before load_all = 1     # 静态注册已经完成\n'
    'reg_count after  load_all = 1     # 没有可加载的动态后端\n'
    'reg[0] = CPU (devs=1)\n'
    'dev[0] = CPU | AMD Ryzen 7 5700X 8-Core Processor | type=0\n'
    '```\n\n'
    '`ggml_backend_register()` / `ggml_backend_device_register()` 是给“树外”调用者用的内部 API：'
    '例如 RPC 客户端在连上服务器后，把新后端注册进同一个注册表（`common/arg.cpp:1180`）。',
    src=REG, parts=[(292, 300)], lang='c')

L.section(
    '五、★ 设备枚举顺序：下标 = 注册顺序',
    '设备枚举的全部实现就是这两行转发。没有排序、没有过滤、没有重算 —— '
    '`devices[index]` 的下标是**唯一**的“设备号”。\n\n'
    '按名字（345-353）与按类型（355-363）查找也都是**从头扫、返回第一个匹配**：'
    '当同类型有多个设备时，返回谁仍然由顺序决定。名字比较用 `striequals`（307-314），大小写不敏感。',
    src=REG, parts=[(336, 363)], lang='c')

L.section(
    '六、动态加载：准入校验与两张写死的顺序表',
    '`load_backend()` 是动态加载的**准入闸门**：先 `dl_load_library()`，再要求可选的 `ggml_backend_score()` 非 0，'
    '然后必须找到 `ggml_backend_init`，最后校验 `reg->api_version` **精确等于** `GGML_BACKEND_API_VERSION`。'
    '任何一步不过就返回 `nullptr`，注册表不受影响。\n\n'
    '（这些原语的实现见 L3-03；本课只关心“它什么时候、往哪儿追加”。）',
    src=REG, parts=[(220, 264)], lang='c')

L.section(
    '七、load_all 的候选名字表',
    '`ggml_backend_load_all_from_path()` 依次尝试 **15 个**候选名字（命令口径：'
    '`grep -c \'ggml_backend_load_best("\' ggml/src/ggml-backend-reg.cpp` = 15）：\n\n'
    '```text\n'
    'blas, zendnn, cann, cuda, hip, metal, rpc, sycl,\n'
    'vulkan, virtgpu, opencl, hexagon, musa, openvino, cpu\n'
    '```\n\n'
    '★ 这张表与构造函数里的 `#ifdef` 顺序**不是同一张表**。例如 `CANN`：静态顺序排在 CUDA 之后'
    '（156 vs 120 行），动态顺序排在 CUDA 之前（587 vs 588 行）。'
    '由于注册一律是 `push_back`，“同一个后端是编译进来的还是加载进来的”，会改变它在 `devices[]` 里的下标。\n\n'
    '最后一段是环境变量 `GGML_BACKEND_PATH`：这是唯一由用户指定路径的加载入口（601-604）。'
    '与之相对，`GGML_DISABLE_VULKAN`（131 行）是唯一能在运行期关掉一个**已编译**后端的开关。',
    src=REG, parts=[(574, 604)], lang='c')

L.section(
    '八、★ Meta backend：合成的设备',
    '`ggml_backend_meta_device()` 把一组“简单设备”打包成一个设备：内存相加（99-110）、'
    '能力取交集（154-159）、类型固定为 `GGML_BACKEND_DEVICE_TYPE_META`（112-116）。\n\n'
    '它**不是注册表成员**：全文件没有任何 `ggml_backend_register()` 调用，造出来的 `ggml_backend_device` 里 '
    '`reg = nullptr`（237-241）。所以 `ggml_backend_dev_count()` 数不到它，'
    'llama.cpp 的设备选择循环里遇到 META 型设备会直接 `GGML_ABORT`（`src/llama.cpp:269-270`）——'
    '它由调用方按需创建：张量并行时把枚举到的设备打包（`src/llama.cpp:193-220`）。\n\n'
    '设备名是拼出来的：`"Meta(" + 各简单设备名 + ")"`（65-77），buffer type 同理（256-268）。'
    '普通后端的名字则完全由后端自己给：CUDA 用 `GGML_CUDA_NAME + 序号`（`ggml-cuda.cu:5798`），'
    'CPU 设备固定 `"CPU"`（`ggml-cpu.cpp:353-357`）。',
    src=META, parts=[(215, 245)], lang='c')

L.section(
    '九、消费侧：注册顺序怎么变成 offload',
    '注册表本身不 offload。真正的消费者是 `llama.cpp` 的默认设备选择：它按注册顺序遍历设备，'
    '把 GPU 型设备收进 `gpus`（254 行 `push_back`），拼成 `model->devices`（RPC 前置 275-277、GPU 追加 278-279）。\n\n'
    '之后：`--main-gpu N` 取 `model->devices[N]`（297-299）；layer/row 模式下 `tensor_split` 的第 j 项'
    '对应 `model->devices[j]`（`llama-model.cpp:1488-1508`），归一化后由 `upper_bound(splits, ...)` '
    '决定每层交给哪块卡（`llama-model.cpp:1529`），写进 `dev_layer[il]`（1540-1542），'
    '权重张量再按它选 buffer 类型完成落位（L2-03）。',
    src=LLAMA, parts=[(222, 231)], lang='cpp')

L.footnote_add('本课逐字引用 3 个文件：`ggml/src/ggml-backend-reg.cpp`、`ggml/src/ggml-backend-meta.cpp`（计划里本课的 2 个），'
               '外加 `src/llama.cpp` —— 只为一处：设备顺序的消费侧（第 6 幕 / 第九节）。'
               '该文件本就由 L2-01 覆盖，本课是重复声明（门禁口径：至少被一课声明）。')
L.footnote_add('按行号指路、未引用原文因而**不计入本课覆盖率**的文件：'
               '`ggml/src/CMakeLists.txt:428-439`（谁定义 GGML_USE_*）、`common/arg.cpp:1116-1180`（设备名解析与 RPC 注册）、'
               '`ggml/src/ggml-cuda/ggml-cuda.cu:5798` 与 `ggml/include/ggml-cuda.h:11/14/17`（CUDA/ROCm/MUSA 设备名）、'
               '`ggml/src/ggml-cpu/ggml-cpu.cpp:353-357`（CPU 设备名）、`ggml/src/ggml-metal/ggml-metal.cpp:307`、'
               '`ggml/src/ggml-rpc/ggml-rpc.cpp:2273-2276`（RPC 设备数来自 context）、'
               '`ggml/src/ggml-cann/ggml-cann.cpp:2807-2810`（CANN 设备是 GPU 型）、'
               '`src/llama-model.cpp:1488-1542`（splits 与 dev_layer，L2-05 已逐字引用）。')
L.footnote_add('第 3、5 幕的“本机实测”来自作者自建的临时 harness（直接调用 `ggml_backend_register()` 注册两个假后端，'
               '再读 `ggml_backend_dev_count()` / `ggml_backend_dev_get(i)`），跑在本仓库 `build/` 下已有的 CPU-only 构建上；'
               '该 harness 不属于本课件仓库，也不计入覆盖率。')

L.prereqs('`L3-01`')

L.goal(
    '说出 `ggml_backend_registry` 的两个成员，以及“设备 0”这个编号是在哪里被定下来的；',
    '列出至少 6 个编译期后端开关，并说明它们由谁（哪个 CMake 函数的哪一段）决定；',
    '解释为什么“同一份 `--tensor-split` 参数在两次运行里可能指向不同的卡”，以及为什么用 `--device` 更稳；',
    '说明 Meta 设备与注册表里普通设备的区别（谁创建它、它的 `reg` 字段是什么、为什么 `dev_count()` 数不到它）。')

L.conclusion(
    '★ 有哪些后端 = 编译期开关',
    '`ggml_backend_registry` 的构造函数里有 **16 个** `#ifdef GGML_USE_*`（120-173 行），'
    '每个开关注册一个后端家族。这些宏由 `ggml/src/CMakeLists.txt:428-439` 在 `GGML_BACKEND_DL=OFF` 时加给 `ggml` 目标。\n\n'
    '运行期的 `ggml_backend_load_all_from_path()`（574-604）尝试 15 个候选名字的动态库，'
    '是在这个静态集合**之外**做补充 —— 而且它排在同一批数组的**后面**。')

L.conclusion(
    '★ 设备顺序 = 注册顺序，且无人重排',
    '`register_backend()` 只做 `push_back`（201），`register_device()` 也只做 `push_back`（217）；'
    '`ggml_backend_dev_get(i)` 直接返回 `devices[i]`（340-343）。'
    '没有任何一步按名字、显存或性能重排。\n\n'
    '实测（本机）：先注册 `ZZZ` 再注册 `AAA`，得到 `dev[0]=CPU, dev[1]=ZZZ0, dev[2]=AAA0` —— '
    '**顺序 = 注册顺序，不是字典序**；重复注册同一个 `reg`，设备数不变。')

L.conclusion(
    '★ 顺序决定 offload：谁是“第一个 GPU”',
    '`llama.cpp` 按 `ggml_backend_dev_get(i)` 的顺序收集 GPU 型设备（`src/llama.cpp:222-231`），'
    '拼成 `model->devices`。`--main-gpu N` 是它的下标（297-299），`--tensor-split` 的第 j 项对应 `devices[j]`'
    '（`llama-model.cpp:1488-1508`），最终由 `upper_bound(splits, ...)` 决定每层落在哪块卡（1529），'
    '写进 `dev_layer[il]`（1540-1542）。\n\n'
    '静态注册表与动态加载表是**两张不同的写死顺序表**（`CANN` 与 `CUDA` 的先后正好相反），'
    '所以“某个后端是编译进来的还是加载进来的”会改变设备下标 —— 这也是 `--device`（走名字，`by_name`）'
    '比依赖下标更稳的原因（L2-03 权重落位、L2-05 dev_layer、L4-02 调度器都建立在这份顺序上）。')

L.conclusion(
    'Meta 设备：合出来的设备，不在注册表里',
    '`ggml_backend_meta_device()`（215-245）把 N 个简单设备合成一个设备：内存相加、`supports_op` 取交集、'
    '类型为 `GGML_BACKEND_DEVICE_TYPE_META`；它造的 `ggml_backend_device` 里 `reg = nullptr`，'
    '全文件没有注册调用，因此不出现在 `ggml_backend_dev_count()` 的结果里，'
    '而由调用方（张量并行的 llama.cpp）按需创建。名字是拼出来的：`Meta(CUDA0,CUDA1)`。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
