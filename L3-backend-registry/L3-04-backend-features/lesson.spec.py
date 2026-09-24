#!/usr/bin/env python3
"""L3-04 · ggml-feats：后端能力探测 —— 课件 spec。

运行：python3 L3-backend-registry/L3-04-backend-features/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

FEATS = 'ggml/src/ggml-feats.h'
SCORE = 'ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp'
IMPL  = 'ggml/src/ggml-backend-impl.h'
REG   = 'ggml/src/ggml-backend-reg.cpp'
CPP   = 'ggml/src/ggml-cpu/ggml-cpu.cpp'
BEND  = 'ggml/include/ggml-backend.h'
LLAMA = 'src/llama.cpp'
CPUC  = 'ggml/src/ggml-cpu/ggml-cpu.c'
KLEI  = 'ggml/src/ggml-cpu/kleidiai/kleidiai.cpp'
CMAKE = 'ggml/src/ggml-cpu/CMakeLists.txt'

L = Lesson(
    id='L3-04',
    layer='L3 · 后端发现与注册',
    title='ggml-feats：后端能力探测',
    codecap='ggml/src/ggml-feats.h（逐字引用）',
    nav={'prev': {'href': '../L3-03-dynamic-loading/index.html', 'label': 'L3-03 动态加载后端'},
         'next': {'href': '../../L4-memory-and-scheduling/L4-01-ggml-alloc/index.html',
                  'label': 'L4-01 分配器 ggml-alloc'}},
)

L.note('**一句话**：`ggml-feats.h` 是"运行期"那一半 —— 它回答"**这台机器的 CPU 支持什么**"，'
       '而 CMake 里的 `GGML_USE_*` 是"编译期"那一半，回答"**这份二进制是按什么指令集编出来的**"。'
       '后端能不能被加载，取决于这两个答案的**乘积**。')
L.note('这个头文件只有 166 行，而且**整个文件被一道 `#if defined(__aarch64__) || defined(_M_ARM64)` 包住** —— '
       '在 x86 上它是一个空文件。它小，是因为它只做一件事：把平台 API 问出来的能力位，'
       '装进一个 8 字段的小结构体。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L3 · 后端发现与注册',
    title='166 行，只回答一个问题：<span class="hl-a">这台机器支持什么</span>',
    sub='探测结果为真时加分、为假时出局；编译期宏决定"要问哪些问题"，运行期探测决定"答案是什么"。',
    caption='本课的主角是 ggml/src/ggml-feats.h。它引出的路径会用到 L3-01/L3-02/L3-03 三课的概念。',
    src=FEATS, parts=[(1, 11)], duration=20000,
    mark_src=[3, 5, 6],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip c">CMake: GGML_USE_*</span><span class="arrow">x</span>
    <span class="chip b">运行期 has_*</span><span class="arrow">-></span>
    <span class="chip a">后端自评 score</span><span class="arrow">-></span>
    <span class="chip e">加载器选库</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: '这 166 行是什么', b: '一个<b>只有 ARM64 才有内容</b>的头文件：<br>把 CPU 的能力位读成 8 个字段。', m: 'ggml_feats_get_arch64_runtime' },
  { c: 'b', t: '谁 include 它', b: '全仓库只有两处：<br>cpu-feats.cpp（后端自评）、kleidiai.cpp（内核选择）。', m: '两个使用点，本课都逐字引用' },
  { c: 'c', t: '为什么值得一课', b: '它是"编译期开关 x 运行期探测"<br>这个乘积里<b>运行期的那一半</b>。', m: 'struct ggml_feats_arch64_runtime' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '先看文件的头三行：<span class="v">#pragma once</span> + 一道 <span class="k">架构门</span>。<br>非 ARM64 平台上，这个文件编译出来是空的。',
  '它回答的问题只有一个：<span class="k">这台机器的 CPU 到底支持哪些向量指令</span>。',
  '答案装在一个 8 字段的小结构体里，由 <span class="v">ggml_feats_get_arch64_runtime()</span> <b>按值返回</b>。',
  '这份答案有两条去处：<br>1) <span class="k">后端自评 score</span> —— 决定这个 .so 要不要被加载；<br>2) <span class="k">能力上报表</span> —— 决定应用看得见哪些特性。',
  '回顾 L3-01（后端契约）、L3-02（注册表与设备）、L3-03（动态加载）：<br>它们讲的是"有哪些后端"；本课补上"<span class="k">这台机器配得上哪个后端</span>"。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(17600, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L3-04 · 常量补丁',
    title='自带一份 <span class="hl-c">ABI 常量补丁</span>：先问系统头有没有',
    sub='Linux 侧要用的 9 个常量，每一个都写成 #if !defined(...) 才定义。',
    caption='这一段和主体逻辑无关，却占了文件近三分之一：它解决的是"这份代码要在别人的旧系统上编译"。',
    src=FEATS, parts=[(13, 47)], duration=19000,
    mark_src=[14, 18, 22, 26, 30, 34, 38, 42, 46],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['补丁常量', '位', 'struct 字段', '从哪个位图取'],
  [['HWCAP_ASIMDDP', '1 << 20', 'has_dotprod', 'AT_HWCAP'],
   ['HWCAP_FPHP / HWCAP_ASIMDHP', '1 << 9 / 1 << 10', 'has_fp16', 'AT_HWCAP'],
   ['HWCAP_SVE', '1 << 22', 'has_sve', 'AT_HWCAP'],
   ['HWCAP2_SVE2', '1ULL << 1', 'has_sve2', 'AT_HWCAP2'],
   ['HWCAP2_I8MM', '1ULL << 13', 'has_i8mm', 'AT_HWCAP2'],
   ['HWCAP2_SME / HWCAP2_SME2', '1ULL << 23 / 1ULL << 37', 'has_sme / has_sme2', 'AT_HWCAP2'],
   ['PR_SVE_GET_VL', '51', 'sve_cnt', 'prctl 的请求号'],
   ['PR_SVE_VL_LEN_MASK', '0xffff', 'sve_cnt', 'prctl 返回值的掩码']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '常量先摆在这里，逐条对到后面 struct 的字段上。',
  '<span class="v">AT_HWCAP</span> 是内核在进程启动时填好的位图，一次 <span class="k">getauxval</span> 就能取回。',
  '<span class="v">AT_HWCAP2</span> 装的是后来才加的能力（SVE2 / I8MM / SME），所以要取第二次。',
  '<span class="v">PR_SVE_GET_VL</span> 与掩码不是位，而是 <span class="k">prctl</span> 的请求号与返回值的切法。',
  '★ 每个常量外面都套着 <span class="v">#if !defined(...)</span>：<br>系统头里已经有名字就跳过，没有才补一份。',
  '这个写法本身就说明了它要解决的问题：<b>编译这套代码的机器，其系统头不一定认识这些位</b>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2100, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(16500, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L3-04 · 数据结构',
    title='探测结果的形状：<span class="hl-a">7 个布尔 + 1 个长度</span>',
    sub='值语义、按值返回、static inline —— 这三个性质决定了它不需要 ggml 导出任何符号。',
    caption='★ 按值返回 + static inline 是动态加载能成立的前提：每个后端 .so 各自编译一份探测代码。',
    src=FEATS, parts=[(84, 93)], duration=18000,
    mark_src=[85, 86, 87, 88, 89, 90, 91, 92],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: 'HWCAP 侧（3 个）', b: 'has_dotprod<br>has_fp16<br>has_sve', m: '来自 AT_HWCAP' },
  { c: 'b', t: 'HWCAP2 侧（4 个）', b: 'has_sve2 / has_i8mm<br>has_sme / has_sme2', m: '来自 AT_HWCAP2' },
  { c: 'c', t: '唯一不是布尔的', b: 'sve_cnt：SVE 向量长度<br>（int，不是 0/1）', m: '来自 prctl(PR_SVE_GET_VL)' },
  { c: 'd', t: '怎么被返回', b: 'static inline + 按值返回<br><b>不导出符号、不依赖链接</b>', m: 'ggml_feats_arch64_runtime_t' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '7 个布尔 + 1 个整数。这份结构体就是"运行期探测"的全部产出。',
  '<span class="v">AT_HWCAP</span> 那一侧产生三个：dotprod / fp16 / sve。',
  '<span class="v">AT_HWCAP2</span> 那一侧产生四个：sve2 / i8mm / sme / sme2。',
  '<span class="v">sve_cnt</span> 是唯一不是布尔的字段：<span class="k">向量长度是运行期才知道的数</span>，不是 0/1。',
  '★ 它按值返回、而且是 <span class="v">static inline</span>：<br>谁 include 谁就编译出自己的一份，<b>不需要 ggml 导出符号</b>。',
  '这正是 L3-03 里那个被 dlopen 进来的后端 .so 能"自己给自己打分"的前提。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(15800, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L3-04 · 探测实现',
    title='★ Linux：一次 <span class="hl-b">getauxval</span>，读两个位图',
    sub='能力位是内核填好的；唯一需要"再问一次"的是 SVE 的向量长度。',
    caption='这一函数是整份头文件的核心：98-115 行，Linux 分支的全部逻辑。',
    src=FEATS, parts=[(95, 115)], duration=22000,
    mark_src=[96, 99, 100, 102, 103, 104, 105, 106, 110, 111, 113],
    notes_src={99: '两次 getauxval，把内核填好的两个位图一次取回',
               103: 'has_fp16 要两位同时为真：FPHP（半精度）与 ASIMDHP（向量半精度算术）',
               110: '只有 has_sve 为真时才去问长度 —— 没有 SVE 就没有"当前向量长度"这回事',
               111: 'prctl 返回的是"当前线程"的 SVE 向量长度（字节）',
               113: '用 PR_SVE_VL_LEN_MASK 把长度位切出来，其余位丢弃'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:6px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:4px">两个位图 -> 七个布尔</div>';
const rows = [
  { n: 'hwcap  & HWCAP_ASIMDDP', v: 'has_dotprod', c: 'a' },
  { n: 'hwcap  & HWCAP_FPHP', v: 'has_fp16 (之一)', c: 'b' },
  { n: 'hwcap  & HWCAP_ASIMDHP', v: 'has_fp16 (之二)', c: 'b' },
  { n: 'hwcap  & HWCAP_SVE', v: 'has_sve', c: 'c' },
  { n: 'hwcap2 & HWCAP2_SVE2', v: 'has_sve2', c: 'd' },
  { n: 'hwcap2 & HWCAP2_I8MM', v: 'has_i8mm', c: 'e' },
  { n: 'hwcap2 & HWCAP2_SME', v: 'has_sme', c: 'f' },
  { n: 'hwcap2 & HWCAP2_SME2', v: 'has_sme2', c: 'g' }
];
const els = rows.map(r => {
  const e = U.el('div', { class: 'formula', style: 'padding:3px 8px;font-size:9.5px' });
  e.innerHTML = '<span class="m">' + U.esc(r.n) + '</span> -> <span style="color:var(--' + r.c + ')">' + U.esc(r.v) + '</span>';
  left.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">为什么 fp16 要两位一起看？</div>' +
  '<div class="cb">源码把两个位 <b>与</b> 起来：<span class="cm" style="margin:0">!!(hwcap &amp; HWCAP_FPHP) &amp;&amp; !!(hwcap &amp; HWCAP_ASIMDHP)</span>。' +
  '一个是半精度能力，一个是向量半精度算术能力 —— 后端要的是后者。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">为什么长度要单独问？</div>' +
  '<div class="cb">SVE 的向量长度是 <b>运行期</b> 决定的（同架构可有不同实现），' +
  '位图里没有这个数，只能走 <span class="cm" style="margin:0">prctl(PR_SVE_GET_VL)</span>。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  'Linux 分支只做两件事：<span class="k">读位图</span>、<span class="k">问长度</span>。',
  '能力位是内核填的，进程启动后 <span class="v">getauxval</span> 一取即得，没有系统调用开销。',
  '<span class="v">has_fp16</span> 是唯一需要两个位同时为真的字段：<b>能力的"复合条件"被写在探测端</b>，查询端只看到一个布尔。',
  '<span class="v">sve_cnt</span> 只在 <span class="k">has_sve</span> 为真时才去问，避免对不支持 SVE 的机器调 prctl。',
  '★ 长度是运行期量：<b>同一个二进制在不同 SVE 长度的机器上会得到不同的 sve_cnt</b> ——<br>这就是"运行期探测"不能省的原因。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
els.forEach((e, i) => tl.at(2600 + i * 1900, () => {
  els.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.28'; });
  msg.innerHTML = texts[1];
}));
tl.at(17600, () => { msg.innerHTML = texts[2]; });
tl.at(19500, () => { msg.innerHTML = texts[3]; });
tl.at(21000, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L3-04 · 探测实现',
    title='同一份结构体，<span class="hl-e">三种问法</span>',
    sub='macOS 按名字问 sysctl，Windows 按编号问 IsProcessorFeaturePresent；两侧都把 sve_cnt 记为 0。',
    caption='探测不到的量不猜：源码注释给出了两处 sve_cnt = 0 的理由。',
    src=FEATS, parts=[(116, 161)], duration=21000,
    mark_src=[120, 124, 148, 149, 151, 155, 159, 160],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: 'Linux', b: 'getauxval(AT_HWCAP / AT_HWCAP2)<br>+ prctl(PR_SVE_GET_VL)<br><b>sve_cnt = prctl 的结果</b>', m: '位图' },
  { c: 'e', t: 'macOS', b: 'sysctlbyname("hw.optional.arm.FEAT_*")<br>逐个名字问<br><b>sve_cnt = 0</b>', m: '名字' },
  { c: 'f', t: 'Windows', b: 'IsProcessorFeaturePresent(PF_ARM_*)<br>按编号问<br><b>sve_cnt = 0</b>', m: '编号' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '同一个结构体，三个平台三种问法：<span class="k">位图 / 名字 / 编号</span>。',
  '<span class="v">#elif defined(__APPLE__)</span> 之后是 7 个 sysctlbyname 调用，一个能力一个名字。',
  '每次调用都判 <span class="v">== 0</span>，即"取到了才写"；失败就保持结构体初始化时的 0。',
  '<span class="v">#elif defined (_WIN32)</span> 之后是 7 行 IsProcessorFeaturePresent，PF_ARM_* 也是这文件自己补的常量。',
  '★ 两处 <span class="v">sve_cnt = 0</span> 都配了一行源码注释：Apple 的理由是"不支持用户态 non-streaming SVE"，Windows 的理由是"这里不暴露运行期向量长度"。',
  '记 0 而不是猜一个值 —— 查询端（cpu-feats.cpp）只把 sve_cnt 当作"长度"用，0 就等于"没有"。'
];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(10300, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[3];
});
tl.at(15000, () => { msg.innerHTML = texts[4]; });
tl.at(18500, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L3-04 · 路径 A',
    title='★ 特性开关的路径：<span class="hl-a">编译期 #ifdef</span> x <span class="hl-b">运行期探测</span>',
    sub='cpu-feats.cpp 是唯二使用点之一：它把探测结果翻译成"这个后端变体值多少分"。',
    caption='注意 has_sme2：头文件探测了它，但这个 score 函数没有读它。',
    src=SCORE, parts=[(1, 39)], duration=26000,
    mark_src=[2, 6, 7, 8, 11, 12, 13, 39],
    notes_src={2: '整个文件只 include 两个头：本课的主角 + ggml-backend-impl.h',
               8: '一次调用拿到 7 个布尔 + 1 个长度，之后全在本地判断',
               12: '编译期要求 dotprod，运行期没有 -> 直接返回 0（= 本机不支持）',
               39: '把上面的 score 函数导出成 .so 的 ggml_backend_score 符号'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['编译期宏（这个变体的前提）', '运行期字段', '不满足时', '满足时加分'],
  [['GGML_USE_DOTPROD', 'af.has_dotprod', 'return 0', '1 << 1'],
   ['GGML_USE_FP16_VECTOR_ARITHMETIC', 'af.has_fp16', 'return 0', '1 << 2'],
   ['GGML_USE_SVE', 'af.has_sve', 'return 0', '1 << 3'],
   ['GGML_USE_MATMUL_INT8', 'af.has_i8mm', 'return 0', '1 << 4'],
   ['GGML_USE_SVE2', 'af.has_sve2', 'return 0', '1 << 5'],
   ['GGML_USE_SME', 'af.has_sme', 'return 0', '1 << 6']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '这是本课的枢纽：<span class="k">编译期开关</span> 与 <span class="k">运行期探测</span> 在这里相乘。',
  '<span class="v">#ifdef GGML_USE_DOTPROD</span> 表示这份 .so 是用 dotprod 指令编出来的 —— 它<b>要求</b>运行环境必须有 dotprod。',
  '<span class="v">if (!af.has_dotprod) { return 0; }</span>：要求落空就出局。<span class="k">0 的含义就是"本机不支持"</span>。',
  '满足则按位加分，基准分 1：<span class="v">score = 1 + sigma(1 &lt;&lt; k)</span>。',
  '分数越高越"专用"，也就越优先被加载 —— 同一台机器上可能有好几个变体都能跑。',
  '★ 注意 <span class="v">has_sme2</span>：头文件探测了它，但 score 函数（以及 kleidiai）都没有读它。<br>能力先探测、使用后来跟上，是这类头文件的常态。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2500 + i * 2400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(17500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
tl.at(22000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L3-04 · 路径 A',
    title='路径的终点：<span class="hl-c">加载器</span>在后端被加载之前就问它',
    sub='score 为 0 的 .so 直接 return nullptr；不是崩溃，是安静地跳过。',
    caption='另一处（source.md 引用 534-542 行）：扫描目录时对每个 .so 求分，留下最高的那个。',
    src=REG, parts=[(220, 235)], duration=20000,
    mark_src=[229, 230, 234],
    notes_src={229: '从刚 dlopen 进来的 .so 里取 ggml_backend_score 符号',
               230: 'score 为 0 -> 这个后端在这台机器上不可用，直接放弃'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">dl_load_library</span><span class="arrow">-></span>
    <span class="chip c">dl_get_sym("ggml_backend_score")</span><span class="arrow">-></span>
    <span class="chip b">score()</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'b', t: 'score != 0：继续', b: '接着去找 <b>ggml_backend_init</b>，<br>真正把后端注册进来。', m: '第 237 行往下' },
  { c: 'e', t: 'score == 0：放弃', b: '日志一行 "not supported"，<br><b>return nullptr</b>，不注册。', m: '第 230-234 行' },
  { c: 'c', t: '多个候选怎么办', b: '同名的变体各自算分，<br>留 <b>best_score</b> 最大的那个。', m: '第 534-542 行（见 source.md）' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '这条路径的终点在动态加载器里，而不是在后端内部。',
  '<span class="v">dl_get_sym(handle.get(), "ggml_backend_score")</span>：把自评函数从刚打开的 .so 里取出来。',
  '<span class="v">if (score_fn &amp;&amp; score_fn() == 0)</span> -> <span class="k">return nullptr</span>：这个后端不被加载。',
  '★ 所以"特性开关"的第一次被查询，发生在<b>后端还没有被加载</b>的时候 —— 这正是 L3-03 动态加载要解决的"选哪一个"。',
  '回顾 L3-03：那个被 dlopen 的后端库必须导出 <span class="v">ggml_backend_init</span>；<br>如果还想参与"选库"，就必须<b>额外</b>导出 <span class="v">ggml_backend_score</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13800, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[3];
});
tl.at(17200, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L3-04 · 路径 B',
    title='★ 第二条路径：把能力<span class="hl-b">报</span>出去，而不是用来打分',
    sub='ggml_backend_cpu_get_features() 返回一张以 { nullptr, nullptr } 结尾的特性表。',
    caption='源码注释写明动机：取代 ggml_cpu_has_* 系列，并让别的后端也能用同一套 API 暴露自己的特性。',
    src=CPP, parts=[(536, 548)], duration=22000,
    mark_src=[536, 537, 538, 539, 543, 544],
    notes_src={537: '一句话说清了两件事：取代 ggml_cpu_has_*，以及"其它后端也能用同一套 API"',
               539: '整张表只算一次（static + lambda），之后每次查询都是同一个数组'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip b">运行期 has_*</span><span class="arrow">+</span>
    <span class="chip c">#ifdef GGML_USE_*</span><span class="arrow">-></span>
    <span class="chip a">ggml_backend_feature[]</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: '表的形状', b: 'struct ggml_backend_feature<br>{ const char * name; const char * value; }', m: 'name/value 两个字符串' },
  { c: 'b', t: '谁填进去（一）', b: 'ggml_cpu_has_sse3() / has_avx2() /<br>has_dotprod() / has_sve() ...', m: '编译期宏决定真假' },
  { c: 'c', t: '谁填进去（二）', b: '#ifdef GGML_USE_ACCELERATE /<br>OPENMP / KLEIDIAI / REPACK', m: '直接按编译期追加' },
  { c: 'e', t: '唯一的数字项', b: 'if (ggml_cpu_get_sve_cnt() > 0)<br>push_back({ "SVE_CNT", ... })', m: '运行期问出来的值' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '第二条路径不是为了决定"能不能加载"，而是为了回答"你支持什么"。',
  '<span class="v">ggml_backend_cpu_get_features()</span> 签名里有一个 <span class="k">ggml_backend_reg_t</span> 参数：<br>它属于"注册表"这一层，而不是设备层。',
  '表里的每一项都是 <span class="k">名字 + 值</span> 两个字符串，最后用 <span class="v">{ nullptr, nullptr }</span> 结尾。',
  '这些名字就是应用能看到的字符串：<span class="v">SSE3</span> / <span class="v">AVX2</span> / <span class="v">NEON</span> / <span class="v">DOTPROD</span> / <span class="v">SVE_CNT</span> / <span class="v">KLEIDIAI</span> …',
  '★ 同一张表里混着两种来源：<span class="k">ggml_cpu_has_*()</span>（编译期宏）与 <span class="v">#ifdef GGML_USE_*</span>（直接追加）；<br>唯独 <span class="v">SVE_CNT</span> 带的是运行期问出来的数字。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13100, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[3];
});
tl.at(17000, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L3-04 · 收束',
    title='把这一课压成一张"谁定义 / 谁读"的表',
    sub='一条路径决定"要不要加载你"，另一条路径决定"怎么描述你"。',
    caption='下一课 L4-01 分配器 ggml-alloc：从"能力"转到"内存"。',
    src=FEATS, parts=[(161, 166)], duration=20000,
    mark_src=[163],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['层', '谁定义', '谁读', '在哪一课展开'],
  [['编译期开关', 'CMake 的 ARCH_DEFINITIONS（GGML_USE_*）', 'cpu-feats.cpp 的 #ifdef', 'L3-02'],
   ['运行期探测', 'ggml-feats.h 的 ggml_feats_get_arch64_runtime()', 'cpu-feats.cpp / kleidiai.cpp', '本课'],
   ['后端自评', 'ggml_backend_score()（动态库导出符号）', 'ggml-backend-reg.cpp 加载器', 'L3-03'],
   ['能力上报', 'ggml_backend_cpu_get_features()', 'reg_get_proc_address + llama_print_system_info', '本课 / L8'],
   ['内核选择', '运行期特性位', 'kleidiai 的 ctx.features 位掩码', 'L5-04 / L5-05']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '一台 Linux/ARM64 机器上 <span class="mono">getauxval(AT_HWCAP) &amp; HWCAP_SVE == 0</span>。' +
  '目录里有一个用 <span class="mono">GGML_USE_SVE</span> 编译出来的 CPU 后端变体，加载器会怎么处理它？',
  '<span class="mono">ggml_backend_cpu_aarch64_score()</span> 里 <span class="mono">#ifdef GGML_USE_SVE</span> 生效，' +
  '<span class="mono">af.has_sve</span> 为假 -> <span class="mono">return 0</span>。<br>' +
  '加载器判 <span class="mono">score_fn() == 0</span>，直接 <span class="mono">return nullptr</span>：' +
  '这个 .so 不会被加载，也不报错，只在日志里留一行 "not supported"。<br>' +
  '同目录里其它变体各自算分（满足一个特性加一个二进制位），<span class="mono">best_score</span> 最大的那个被选中。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '五层，各自的读者不同。',
  '<span class="k">编译期开关</span> 回答"我按什么指令集编的"；<span class="k">运行期探测</span> 回答"这台机器有什么"。',
  '两者在 <span class="v">ggml_backend_cpu_aarch64_score()</span> 里相乘，结果决定 L3-03 的加载器选哪一个 .so。',
  '<span class="k">能力上报</span> 是另一条路：同一份探测结果被写成字符串表，供应用查询。',
  '★ 一句话：<span class="v">编译期决定"能不能跑"，运行期决定"给多少分"</span>；<br>而两者都不满足时，后端的表现是"安静地不被加载"，不是崩溃。',
  '下一课 L4-01 分配器 ggml-alloc：算子的内存从哪来。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(14200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
tl.at(17500, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、两道门：架构门与平台门',
    '整个文件被一道 `#if defined(__aarch64__) || defined(_M_ARM64)` 包住（第 3 行），'
    '最后一行（第 166 行）是它的 `#endif`。这意味着：**在 x86 / RISC-V 上，这个文件编译出来是空的**，'
    '不会有任何函数被定义。门里面又按平台分成三支：`__linux__` / `__APPLE__` / `_WIN32`。',
    src=FEATS, parts=[(1, 11)], lang='c')

L.section(
    '二、Linux 侧的常量补丁：为什么要自己写 9 个 `#define`',
    '这些常量本该由 `<sys/auxv.h>` / `<sys/prctl.h>` 提供，但每个外面都套着 `#if !defined(...)`：'
    '**系统头里已经有名字就跳过，没有才补一份**。这个写法本身说明了它要解决的问题 —— '
    '编译这套代码的机器，其系统头不一定认识这些位（`HWCAP2_SME2 = 1ULL << 37` 是相当新的位）。',
    src=FEATS, parts=[(13, 47)], lang='c')

L.section(
    '三、macOS / Windows 侧的常量补丁',
    '另外两个平台分支只 include 平台头，然后补 `PF_ARM_*` 常量（Windows 用的处理器特性编号）。'
    '注意 Apple 分支不需要补任何常量 —— 它按**字符串名字**问 sysctl。',
    src=FEATS, parts=[(49, 82)], lang='c')

L.section(
    '四、探测结果的形状',
    '7 个布尔 + 1 个 `int`。这是"运行期探测"的全部产出，'
    '**按值返回、`static inline`** —— 谁 include 谁就编译出自己的一份，不需要 ggml 导出任何符号。'
    '这正是 L3-03 里被 `dlopen` 进来的后端 `.so` 能自己给自己打分的前提。\n\n'
    '★ 注意 `has_sme2`：它被探测了，但本课引用的两个使用点（`cpu-feats.cpp` 的 score 函数、'
    '`kleidiai.cpp` 的 `ctx.features`）都没有读它。',
    src=FEATS, parts=[(84, 93)], lang='c')

L.section(
    '五、★ Linux 路径：getauxval + prctl',
    '两次 `getauxval` 把内核填好的位图取回，然后逐个位映射成字段；'
    '唯一需要"再问一次"的是 SVE 向量长度（位图里没有这个数）。',
    src=FEATS, parts=[(98, 115)], lang='c')

L.section(
    '六、macOS / Windows 路径',
    '同一个结构体，三种问法：Linux 读位图、macOS 按名字问 sysctl、Windows 按编号问 '
    '`IsProcessorFeaturePresent`。两处 `sve_cnt = 0` 都配了源码注释说明理由'
    '（**注释声称**：Apple 不支持用户态 non-streaming SVE；Windows 这边不暴露运行期向量长度）。',
    src=FEATS, parts=[(116, 161)], lang='c')

L.section(
    '七、编译期开关是从哪来的',
    '`GGML_USE_*` 不是源文件里定义的，而是 CMake 在该变体的编译定义里追加的。'
    '下面的片段来自 `GGML_CPU_ALL_VARIANTS` 分支：每开一个 ARM 特性，就同时改 `-march` 目标、'
    '加一个 `ARCH_TAGS` 标签、追加一条 `ARCH_DEFINITIONS`。'
    '**这三件事必须一起发生**：指令集、库名后缀、以及源文件里那个 `#ifdef`。',
    src=CMAKE, parts=[(181, 195)], lang='cmake')

L.section(
    '八、★ 路径 A 的第一步：编译期与运行期相乘',
    '`ggml_backend_cpu_aarch64_score()` 是本课的枢纽：'
    '`#ifdef GGML_USE_X` 表示"这份二进制是按 X 编的，它**要求**运行环境有 X"；'
    '`if (!af.has_x) { return 0; }` 表示"要求落空就出局"。满足则按位加分（基准分 1）。\n\n'
    '★ `0` 不是"分数低"，而是**约定的"本机不支持"**（见下一节 `ggml_backend_score_t` 的注释）。',
    src=SCORE, parts=[(1, 39)], lang='c')

L.section(
    '九、路径 A 的第二步：score 怎么变成一个可被查询的符号',
    '`GGML_BACKEND_DL_SCORE_IMPL(score_fn)` 展开出 `extern "C"` 的 `ggml_backend_score()`，'
    '它只是把上面那个静态函数转发出去。注意 `#ifdef GGML_BACKEND_DL`：'
    '**只有在动态加载构建下这个符号才存在**；静态构建时宏展开为空。',
    src=IMPL, parts=[(250, 254)], lang='c')

L.section(
    '十、路径 A 的第三步：加载器按 score 择库',
    '`load_backend()` 里：取不到 `ggml_backend_score` 就照常加载（静态后端没有这个符号）；'
    '取得到且返回 0，就**放弃这个 .so**。\n\n'
    '另一处是目录扫描：对每个候选 `.so` 求分，`if (s > best_score)` 留下最高的一个；'
    '`best_score` 仍为 0 时，才回退去加载基础版后端。',
    src=REG, parts=[(534, 543)], lang='c')

L.section(
    '十一、★ 路径 B：能力被"报"出去',
    '第二条路径不参与"选库"，只回答"你支持什么"。契约在公共头文件里：'
    '一个 `name`/`value` 的两字符串结构体，外加一个返回**以 `nullptr` 结尾的数组**的函数指针类型。'
    '注意取这个函数的方式是 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")` —— '
    '**按符号名字符串查询**，而不是直接链接。',
    src=BEND, parts=[(203, 203), (219, 225)], lang='c')

L.section(
    '十二、路径 B 的实现：表里混着两种来源',
    '`ggml_backend_cpu_get_features()` 里，绝大多数项来自 `ggml_cpu_has_*()`（编译期宏）；'
    '表尾几项则是直接 `#ifdef GGML_USE_*` 追加的；`SVE_CNT` 是唯一带运行期数值的项。'
    '整张表用 `static` + lambda 只构建一次，最后以 `{ nullptr, nullptr }` 结尾。',
    src=CPP, parts=[(588, 604)], lang='cpp')

L.section(
    '十三、路径 B 的表尾：编译期追加项',
    '这几个 `#ifdef` 项说明：**后端特性表不只是"硬件能力表"**，'
    '它同时描述了这份构建打开了哪些可选组件（OpenMP、KleidiAI、repack 等）。',
    src=CPP, parts=[(629, 650)], lang='cpp')

L.section(
    '十四、谁在读这张表',
    '`llama_print_system_info()` 遍历注册表里的每个后端，'
    '用同一个符号名取出特性函数，再把 `name = value` 拼成一行字符串。'
    '这就是启动时打印的 system_info。',
    src=LLAMA, parts=[(598, 617)], lang='cpp')

L.section(
    '十五、对照：`ggml_cpu_has_*` 是编译期宏，不是运行期探测',
    '同一个头文件名叫"feats"，但 `ggml_cpu_has_dotprod()` 这类函数查的是 **编译期** 的 `__ARM_FEATURE_DOTPROD`，'
    '不是运行期位图。它回答的是"我被编成了什么样"。\n\n'
    '运行期那一侧另有实现：SVE 向量长度由 `svcntb()` 在初始化时问出来，'
    '同样只在 `__ARM_FEATURE_SVE` 编进来时才存在。',
    src=CPUC, parts=[(3819, 3825)], lang='c')

L.section(
    '十六、运行期特性如何决定内核',
    '第二个使用点：KleidiAI 后端把运行期探测结果压成一个位掩码 `ctx.features`，'
    '内核实现按这个掩码选择。这是本课与 L5/L6（后端如何用特性决定 kernel）的接口。',
    src=KLEI, parts=[(313, 320)], lang='cpp')

L.footnote_add(
    '本课的主文件是 `ggml/src/ggml-feats.h`（166 行）。它只有两个使用点，'
    '但"一个特性开关从定义到被后端查询的路径"要跨 5 个文件才能走完，'
    '因此本课**额外声明覆盖**：`ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`（唯一把探测结果变成后端分数的文件）、'
    '`ggml/src/ggml-backend-impl.h`（`ggml_backend_score_t` 与导出宏）、'
    '`ggml/src/ggml-backend-reg.cpp`（加载器读 score）、'
    '`ggml/include/ggml-backend.h`（特性表契约）、'
    '`ggml/src/ggml-cpu/ggml-cpu.cpp`（特性表实现）、'
    '`src/llama.cpp`（特性表的消费者）、'
    '`ggml/src/ggml-cpu/ggml-cpu.c`（编译期查询的对照）、'
    '`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`（运行期特性 -> 内核）。'
    '追加引用不会让任何原文件失配：覆盖度门禁取的是并集。')
L.footnote_add(
    '`ggml/src/ggml-cpu/CMakeLists.txt` 也出现在本课引用里，用于说明 `GGML_USE_*` 的来源。'
    '它不在本视角的覆盖域后缀内（存在，但不计入覆盖率），属于"诚实的不计入"。')
L.footnote_add(
    '本课没有引用 `ggml_backend_dev_get_features` 这个符号：它在本版本里不存在。'
    '真实的查询方式是 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")`，'
    '返回类型是 `ggml_backend_feature *`。')

L.prereqs('`L3-02`（注册表与设备）、`L3-03`（动态加载后端）')

L.goal(
    '说出一个特性开关的完整路径：`GGML_USE_*`（CMake 编译定义）-> `#ifdef` + 运行期 `has_*` 相与 -> '
    '`ggml_backend_score()` -> 加载器按分数选库（对应验收点）；',
    '解释 `ggml-feats.h` 为什么整份内容被一道架构门包住，以及它在非 ARM64 平台上的形态；',
    '说明同一份探测结果的两条去处：后端自评（决定是否加载）与能力上报（`ggml_backend_feature` 数组）；',
    '区分编译期查询（`ggml_cpu_has_dotprod()` 查 `__ARM_FEATURE_DOTPROD`）与运行期探测'
    '（`ggml_feats_get_arch64_runtime()` 查 `AT_HWCAP`/`AT_HWCAP2`/`sysctl`/`IsProcessorFeaturePresent`）。')

L.conclusion(
    '`ggml-feats.h` = 运行期那一半',
    '它只有 166 行，整份内容被 `#if defined(__aarch64__) || defined(_M_ARM64)` 包住，'
    '按平台分成 Linux（`getauxval` + `prctl`）、macOS（`sysctlbyname`）、Windows'
    '（`IsProcessorFeaturePresent`）三支，产出一个 8 字段的值语义结构体。'
    '**它不是"后端特性表"，而是"CPU 能力探测器"** —— 后端特性表是它的下游消费者之一。')

L.conclusion(
    '★ 一个特性开关的完整路径',
    '```text\n'
    'CMake: ARCH_DEFINITIONS += GGML_USE_DOTPROD      (ggml/src/ggml-cpu/CMakeLists.txt)\n'
    '  -> #ifdef GGML_USE_DOTPROD                     (cpu-feats.cpp)\n'
    '     + af.has_dotprod                            (ggml-feats.h: hwcap & HWCAP_ASIMDDP)\n'
    '     -> 相与：不满足 return 0，满足 score += 1<<k\n'
    '  -> ggml_backend_score()                        (GGML_BACKEND_DL_SCORE_IMPL 导出)\n'
    '  -> dl_get_sym + score_fn() == 0 ? 放弃 : 加载  (ggml-backend-reg.cpp)\n'
    '```\n\n'
    '编译期开关决定"这份二进制的前提"，运行期探测决定"这台机器满不满足前提"，'
    '两者都不满足时的表现是**安静地不被加载**。')

L.conclusion(
    '两条路径，同一份探测结果',
    '| 路径 | 入口 | 终点 | 回答的问题 |\n|---|---|---|---|\n'
    '| A 后端自评 | `ggml_backend_cpu_aarch64_score()` | `ggml-backend-reg.cpp` 的加载器 | 这个后端要不要加载 |\n'
    '| B 能力上报 | `ggml_backend_cpu_get_features()` | `llama_print_system_info()` | 这个后端支持什么 |\n\n'
    'B 路径的取值方式是 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")`，'
    '返回以 `nullptr` 结尾的 `ggml_backend_feature[]`。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
