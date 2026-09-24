#!/usr/bin/env python3
"""L5-05 · ★ 多架构 SIMD 与厂商加速 —— 课件 spec。

运行：python3 L5-cpu-backend/L5-05-multiarch-and-vendor/lesson.spec.py

覆盖（plan_matrix.py 分配，共 49 个文件）：
  arch/ 子树 16（7 quants + 4 repack + 5 cpu-feats）、arch-fallback.h、
  amx/ 5、kleidiai/ 4、spacemit/ 15、llamafile/ 2、hbm 2、iqp 2、
  common.h、ggml-cpu-impl.h。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

# ---------------------------------------------------------------- 覆盖清单

FB = 'ggml/src/ggml-cpu/arch-fallback.h'
IMPL = 'ggml/src/ggml-cpu/ggml-cpu-impl.h'
CMN = 'ggml/src/ggml-cpu/common.h'

XQ = 'ggml/src/ggml-cpu/arch/x86/quants.c'
XR = 'ggml/src/ggml-cpu/arch/x86/repack.cpp'
XF = 'ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp'
AQ = 'ggml/src/ggml-cpu/arch/arm/quants.c'
AR = 'ggml/src/ggml-cpu/arch/arm/repack.cpp'
AF = 'ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp'
RQ = 'ggml/src/ggml-cpu/arch/riscv/quants.c'
RR = 'ggml/src/ggml-cpu/arch/riscv/repack.cpp'
RF = 'ggml/src/ggml-cpu/arch/riscv/cpu-feats.cpp'
SQ = 'ggml/src/ggml-cpu/arch/s390/quants.c'
SR = 'ggml/src/ggml-cpu/arch/s390/repack.cpp'
SF = 'ggml/src/ggml-cpu/arch/s390/cpu-feats.cpp'
PQ = 'ggml/src/ggml-cpu/arch/powerpc/quants.c'
PF = 'ggml/src/ggml-cpu/arch/powerpc/cpu-feats.cpp'
LQ = 'ggml/src/ggml-cpu/arch/loongarch/quants.c'
WQ = 'ggml/src/ggml-cpu/arch/wasm/quants.c'

AX = 'ggml/src/ggml-cpu/amx/amx.cpp'
AXH = 'ggml/src/ggml-cpu/amx/amx.h'
AXC = 'ggml/src/ggml-cpu/amx/common.h'
MMQ = 'ggml/src/ggml-cpu/amx/mmq.cpp'
MMQH = 'ggml/src/ggml-cpu/amx/mmq.h'

KLD = 'ggml/src/ggml-cpu/kleidiai/kleidiai.cpp'
KLK = 'ggml/src/ggml-cpu/kleidiai/kernels.cpp'
KLKH = 'ggml/src/ggml-cpu/kleidiai/kernels.h'
KLDH = 'ggml/src/ggml-cpu/kleidiai/kleidiai.h'

SMI = 'ggml/src/ggml-cpu/spacemit/ime.cpp'
SMIH = 'ggml/src/ggml-cpu/spacemit/ime.h'
SME = 'ggml/src/ggml-cpu/spacemit/ime_env.cpp'
SMEH = 'ggml/src/ggml-cpu/spacemit/ime_env.h'
SMK1 = 'ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp'
SMK2 = 'ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp'
SMKH = 'ggml/src/ggml-cpu/spacemit/ime_kernels.h'
SMR = 'ggml/src/ggml-cpu/spacemit/repack.cpp'
SMRH = 'ggml/src/ggml-cpu/spacemit/repack.h'
SMV = 'ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp'
SMVH = 'ggml/src/ggml-cpu/spacemit/rvv_kernels.h'
SMB = 'ggml/src/ggml-cpu/spacemit/spine_barrier.h'
SMM = 'ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp'
SMMH = 'ggml/src/ggml-cpu/spacemit/spine_mem_pool.h'
SMT = 'ggml/src/ggml-cpu/spacemit/spine_tcm.h'

LSG = 'ggml/src/ggml-cpu/llamafile/sgemm.cpp'
LSGH = 'ggml/src/ggml-cpu/llamafile/sgemm.h'

HBM = 'ggml/src/ggml-cpu/hbm.cpp'
HBMH = 'ggml/src/ggml-cpu/hbm.h'
IQP = 'ggml/src/ggml-cpu/iqp.cpp'
IQPH = 'ggml/src/ggml-cpu/iqp.h'

ALL = [FB, IMPL, CMN,
       XQ, XR, XF, AQ, AR, AF, RQ, RR, RF, SQ, SR, SF, PQ, PF, LQ, WQ,
       AX, AXH, AXC, MMQ, MMQH,
       KLD, KLK, KLKH, KLDH,
       SMI, SMIH, SME, SMEH, SMK1, SMK2, SMKH, SMR, SMRH, SMV, SMVH,
       SMB, SMM, SMMH, SMT,
       LSG, LSGH, HBM, HBMH, IQP, IQPH]

L = Lesson(
    id='L5-05',
    layer='L5 · CPU 后端执行',
    title='★ 多架构 SIMD 与厂商加速',
    codecap='ggml/src/ggml-cpu/**（逐字引用，49 个文件）',
    nav={'prev': {'href': '../L5-04-quants-and-repack/index.html', 'label': 'L5-04 ★ 量化与 repack'},
         'next': {'href': '../../L6-gpu-backend/L6-01-cuda-skeleton/index.html', 'label': 'L6-01 CUDA 后端骨架'}},
)

L.cover(*ALL)

L.note('**一句话**：同一个算子（比如 `GGML_OP_MUL_MAT` 配 Q4_0 权重）在 x86、ARM、RISC-V、s390、'
       'PowerPC、LoongArch、WASM 上跑的是**七个同名、不同文件的函数**；"选哪个 kernel"是一张'
       '**二维决策表** —— 编译期决定编进哪些 ISA 变体（`arch-fallback.h`），运行期按 CPU 特性'
       '选一个（`arch/*/cpu-feats.cpp` 的 score）。厂商加速库（AMX / KleidiAI / SpacemiT / llamafile）'
       '不是第三套内核，而是**在这张表上再插一层**：一个 buffer type + 一个 traits 钩子。')
L.note('本课覆盖 `ggml/src/ggml-cpu/` 下的 **49 个源文件**（`plan_matrix.py` 分配给 L5-05 的全部文件）：'
       '`arch/` 子树 16 个、`amx/` 5 个、`kleidiai/` 4 个、`spacemit/` 15 个、`llamafile/` 2 个、'
       '`hbm.*` 2 个、`iqp.*` 2 个，加上 `arch-fallback.h`、`common.h`、`ggml-cpu-impl.h`。'
       '所有引用按行号从上游 v0.5.0 抽取，行号只对该版本有效。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L5 · CPU 后端执行',
    title='★ 一个算子，<span class="hl-a">两维选择</span>：编译期 + 运行期',
    sub='49 个文件分成四类：七架构的 ISA 内核、五份特性探测、四种厂商加速、两条旁路。',
    caption='回顾 L5-04：traits 表把「类型 -> kernel 函数指针」查出来。'
            '本课讲那些函数指针指向的、按架构各写一份的实现体是怎么被编进来、又是怎么被选中的。',
    src=FB, parts=[(2, 12)], duration=20000,
    mark_src=[4, 7, 9, 12],
    notes_src={4: 'arch-fallback.h 的用途只有一句话：本架构没有原生实现时，把 _generic 版本改名顶上',
               7: 'GGML_CPU_GENERIC 分支：通用构建，68 条全部要走改名',
               9: '改名后 quantize_row_q8_0 这个名字指向通用实现（不是第三份代码）',
               12: '每个名字在链接期只能有一个定义 —— 所以“选实现”发生在编译期，运行期改不了'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">同一张图</span><span class="arrow">-></span>
    <span class="chip a">编译期：编进哪些变体</span><span class="arrow">-></span>
    <span class="chip b">运行期：选一个</span><span class="arrow">-></span>
    <span class="chip c">厂商层：接管整颗算子</span>
  </div>
  <div class="row" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'ISA 内核 · 16 文件', b: '七份 quants.c<br>四份 repack.cpp<br>五份 cpu-feats.cpp', m: 'arch/<isa>/' },
  { c: 'b', t: '厂商/第三方 · 26', b: 'amx 5 · kleidiai 4<br>spacemit 15 · llamafile 2', m: 'buffer type + traits' },
  { c: 'c', t: '通用垫层 · 3', b: 'arch-fallback.h<br>common.h · ggml-cpu-impl.h', m: '跨 ISA 兼容' },
  { c: 'd', t: '专用旁路 · 4', b: 'hbm.cpp/h：换内存来源<br>iqp.cpp/h：换 GEMM 组织', m: '不是 SIMD' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:160px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '一个 <span class="m">MUL_MAT</span> 落到 CPU 上，名字叫 <span class="m">ggml_vec_dot_q4_0_q8_0</span>；' +
  '<b>这个名字有七份实现体</b>。',
  '<b>编译期</b>：每份 arch 文件只在对应 ISA 下参与编译，缺的用 generic 顶上（arch-fallback.h）。',
  '<b>运行期</b>：同一架构可以编出多份变体，各自用 CPUID/hwprobe/auxv 探测后自报一个 score。',
  '<b>厂商层</b>：AMX / KleidiAI / SpacemiT 直接注册一个 buffer type，用 traits 拦下整个算子。',
  '还有两条不碰 SIMD 的旁路：hbm 换内存来源，iqp 换小型 GEMM 的组织方式 —— 本课最后两个文件讲它们。',
  '验收点：看到一个算子 + 一个目标架构，你能说出它最终走的是哪个 kernel。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(19000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L5-05 · 编译期',
    title='<span class="hl-a">编译期那一维</span>：每个架构缺哪些内核',
    sub='arch-fallback.h 是整张表的编译期半边：它逐架构列出「本架构没有的原生实现」。',
    caption='右侧条数由本课统计（各分支 #define 计数）：GENERIC 68、wasm 55、s390x 50、'
            'loongarch64 48、powerpc 47、riscv 41、x86_64 29、aarch64 8。',
    src=FB, parts=[(78, 90)], duration=24000,
    mark_src=[78, 79, 80, 88, 89, 90],
    notes_src={78: 'ARM 分支：只有 8 条，而且全在 repack.cpp 段 —— quants.c 一条都不缺',
               79: '段注释直接说明这 8 条属于哪个文件（repack 的 4x4/4x8 GEMM 变体）',
               88: 'x86 分支：29 条。注意 89 行 —— x86 的 quants.c 段只缺 1 条',
               90: 'q2_0 是后加的类型，x86 还没写原生实现，于是用 generic 顶上'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:12px;align-items:flex-start">
    <div class="col grow" id="bars" style="gap:2px"></div>
    <div class="col" id="side" style="width:252px;gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const data = [
  { l: 'GENERIC', n: 68, c: 'g', t: '通用构建：一条原生实现都不用' },
  { l: 'wasm', n: 55, c: 'f', t: 'simd128 只有 128 位' },
  { l: 's390x', n: 50, c: 'e', t: 'VXE/VXE2 起步晚' },
  { l: 'loongarch64', n: 48, c: 'd', t: 'LSX/LASX 覆盖有限' },
  { l: 'powerpc', n: 47, c: 'c', t: '__POWER9_VECTOR__ 之外靠通用' },
  { l: 'riscv', n: 41, c: 'b', t: 'RVV 用 vsetvl 写，条数多' },
  { l: 'x86_64', n: 29, c: 'a', t: '缺的是后加类型（如 q2_0）' },
  { l: 'aarch64', n: 8, c: 'a', t: '只缺 8 条 repack' }
];
const host = wrap.querySelector('#bars');
host.innerHTML = '<div class="cm" style="margin:0 0 3px">arch-fallback.h 各分支条数（缺 = 走 generic）</div>';
const bars = data.map(d => { const b = U.bar(d.l, d.c); host.appendChild(b.el); return b; });
bars.forEach(b => { b.fill.style.width = '0%'; b.val.textContent = '0'; });

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">为什么是「缺」而不是「有」？</div>' +
  '<div class="cb">列「有」需要把七个架构的所有内核名都写一遍，而且新类型一加就要改七处；' +
  '列「缺」只需要写差集，<b>默认全部走 generic</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">这才是「七个同名函数」的真相</div>' +
  '<div class="cb">arch/<span class="m">isa</span>/quants.c 写原生实现，ggml-cpu/quants.c 写 ' +
  '<span class="m">_generic</span> 版本，arch-fallback.h 决定这一份编译里谁叫正式名字。' +
  '<br>三者缺一不可 —— 缺了 fallback 就会 <b>undefined reference</b>。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '先看统计：<b>越「新」的架构，缺得越多</b>（wasm 55、s390x 50），ARM 只缺 8 条。',
  '<span class="v">GENERIC 68</span>：完全不写原生实现也能跑 —— 这是 llama.cpp 能在任何平台上编出来的原因。',
  '<span class="v">aarch64 8</span>：ARM 把 quants.c 写满了（25 个 vec_dot），只在 repack 上留了 8 条缺口。',
  '<span class="v">x86_64 29</span>：连主战场 x86 也有缺口 —— 新增类型总是先有 generic，再有 SIMD。',
  '缺口的分布决定了各平台的相对速度：<b>缺口越少 = 越多的算子走向量路径</b>。',
  '本幕结论：哪些实现体被编进二进制，是<b>编译期</b>定死的；看一下半张表（运行期）怎么选变体。'
];
data.forEach((d, i) => tl.at(700 + i * 2500, () => {
  bars.forEach((b, k) => {
    b.fill.style.width = k <= i ? Math.round(data[k].n / 68 * 100) + '%' : '0%';
    b.val.textContent = k <= i ? String(data[k].n) : '0';
    b.el.style.opacity = k === i ? '1' : (k < i ? '.72' : '.30');
  });
  msg.innerHTML = (i < 4 ? texts[i] : texts[Math.min(i, 4)]);
}));
tl.at(21500, () => { bars.forEach(b => { b.el.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
tl.at(23000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L5-05 · 运行期',
    title='运行期那一维：<span class="hl-b">变体打分</span>，缺一项就出局',
    sub='同一架构可以编出多份 CPU 变体；每份导出一个 score 函数，谁分高谁上场。',
    caption='读分数的是后端注册器（ggml/src/ggml-backend-reg.cpp，属 L3-02，本课不引用其文本）：'
            '遍历候选动态库、调 ggml_backend_score、取最高分。',
    src=XF, parts=[(297, 325)], duration=22000,
    mark_src=[297, 298, 303, 313, 315, 317, 319, 325],
    notes_src={297: '#ifdef GGML_AVX512：这一段代码只在「编 AVX512 变体」时存在',
               298: '每个 #ifdef 都配一次 if (!is.X()) return 0 —— 缺一项不是降级，是这份变体直接 0 分出局',
               303: '每一项命中往一个位上打 1：AVX512 拿 1<<7',
               315: 'AVX512_VNNI 再往 1<<10 打一位',
               319: 'AMX_INT8 是当前 x86 变体的最高位 1<<11',
               325: 'GGML_BACKEND_DL_SCORE_IMPL 把这个函数导出成动态库符号 ggml_backend_score'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:10px;align-items:flex-start">
    <div class="grow" id="tbl"></div>
    <div class="col" id="side" style="width:238px;gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['本变体要求的宏', '探测函数', '命中得分'],
  [['GGML_FMA', 'is.FMA()', '+1'],
   ['GGML_F16C', 'is.F16C()', '+1<<1'],
   ['GGML_AVX2', 'is.AVX2()', '+1<<5'],
   ['GGML_AVX_VNNI', 'is.AVX_VNNI()', '+1<<6'],
   ['GGML_AVX512', 'F/CD/VL/DQ/BW 五项全要', '+1<<7'],
   ['GGML_AVX512_VNNI', 'is.AVX512_VNNI()', '+1<<10'],
   ['GGML_AMX_INT8', 'is.AMX_INT8()', '+1<<11']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">score 是什么形状？</div>' +
  '<div class="cb">一个 int：<b>位图 + 1</b>。基线给 1 分，每满足一项或上一位。' +
  '于是「支持更多特性」的变体分数<b>严格更大</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)">' +
  '<div class="ct" style="color:var(--d)">五种架构同一套写法</div>' +
  '<div class="cb">x86 用 CPUID、aarch64 用 ggml_feats_get_arch64_runtime、riscv 用 hwprobe 系统调用、' +
  's390x 用 getauxval、powerpc 解析 /proc 风格的平台串。</div></div>';

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '左边这张表就是运行期那一维的「打分规则」——它和上一幕的编译期表<b>正交</b>。',
  '两项硬约束：<span class="v">编译期宏</span>决定这段代码是否存在，<span class="v">运行期探测</span>决定它是否拿分。',
  '注意 AVX512 一行要 <b>五项全命中</b>才给分：指令集变体是按「一整组」用的，不能拼。',
  '分数最高的那份动态库被 dlopen，里面就是这张编译期表的<b>另一个实例</b>（宏不同、代码不同）。',
  '所以「选 kernel」不是一次选择，而是两次：<b>先选变体（运行期），再选内核（编译期已成事实）</b>。',
  '下一幕看厂商层怎么在这张表上再插一张表。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2900 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i < 3 ? i + 1 : 2];
}));
tl.at(20000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
tl.at(21500, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L5-05 · 厂商层',
    title='★ <span class="hl-c">厂商层</span>：在二维表上再插一张表',
    sub='KleidiAI 的做法：把「需要哪些 CPU 特性」写成表项，按顺序试，第一条命中就赢。',
    caption='三张同构的表：Q4_0 权重（330-707 行）、Q8_0（758-923）、F32（974-1033）。'
            '表尾各有一个 { } 哨兵项，所以循环写成 NELEMS(table) - 1。',
    src=KLK, parts=[(1035, 1062)], duration=24000,
    mark_src=[1038, 1039, 1040, 1041, 1042, 1045, 1046, 1049],
    notes_src={1038: '只接管 MUL_MAT，而且 src0/src1 都在才算数',
               1041: '第一个条件：特性位掩码必须覆盖表项要求的全部位（按位与后相等）',
               1042: '后三个条件：lhs/rhs/op 三个类型都要对上',
               1045: '第一条命中就返回 —— 表的顺序就是优先级，不是「分数最高者胜」',
               1049: '全表试完还没命中就返回 false，交给上面那两维的默认路径'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:10px;align-items:flex-start">
    <div class="grow" id="tbl"></div>
    <div class="col" id="side" style="width:236px;gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['表内序', 'required_cpu（源文件行）', '权重类型'],
  [['1', 'SME2 | FP16   (380)', 'Q4_0'],
   ['2', 'SME2          (433)', 'F16'],
   ['3', 'DOTPROD       (487)', 'Q4_0'],
   ['4', 'I8MM|DOTPROD  (540)', 'Q4_0'],
   ['5', 'SVE|I8MM|DOTPROD (594)', 'Q4_0'],
   ['6', 'I8MM|DOTPROD  (647)', 'Q4_0'],
   ['7', 'DOTPROD       (700)', 'Q4_0']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">这一层为什么必要？</div>' +
  '<div class="cb">Arm 的 KleidiAI 是<b>外部库</b>：kernel 用 kai_* 命名、有 SME/SME2 这种' +
  ' ggml 自己还没写的路径。用表描述「我能跑什么」，比在 traits 里堆 if 更好维护。</div></div>' +
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">表项命中之后</div>' +
  '<div class="cb">选中的表项挂在 <span class="m">ctx.kernels_q4</span> 上，' +
  '再由 buffer type 在 <span class="m">set_tensor</span> 时按它 repack 权重。</div></div>';

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '假设这颗 CPU 报出的特性掩码是 <span class="v">DOTPROD | I8MM | FP16</span>。',
  '序 1：需要 <span class="v">SME2</span> —— 掩码里没有，跳过。',
  '序 2：需要 SME2 —— 跳过。序 3：需要 DOTPROD —— <b>命中</b>，取第 3 项。',
  '注意：I8MM 比 DOTPROD 更强，但序 3 排在前面，所以<b>这一张表的顺序就是预算好的优先级</b>。',
  '换一颗只有 DOTPROD 的 CPU：序 1-6 全跳过，序 7 命中 —— <b>同一份二进制，不同的 kernel</b>。',
  '这张表不是替代前两维，而是架在它们之上：没有命中就退回默认路径。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3200 + i * 2800, () => {
  rows.forEach((x, k) => {
    x.className = (k === i) ? 'on' : '';
    x.style.opacity = (k <= i) ? '.55' : '1';
  });
  if (i === 0) { rows[0].style.opacity = '.55'; }
  msg.innerHTML = texts[i + 1];
}));
tl.at(22400, () => {
  rows.forEach(x => { x.className = ''; x.style.opacity = '1'; });
  rows[6].className = 'on';
  msg.innerHTML = texts[4];
});
tl.at(23500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L5-05 · 同名不同物',
    title='同一个 <span class="hl-d">ggml_vec_dot_q4_0_q8_0</span>，七份实现',
    sub='x86 这份在同名文件里按 AVX2/AVX 再分一次；ARM 那份连函数契约都不一样。',
    caption='七份实现的行号（v0.5.0 实测）：x86:701、arm:297、riscv:222、s390:217、'
            'powerpc:144、loongarch:647、wasm:232。它们的守卫宏分别是 __AVX2__、__ARM_NEON、'
            '__riscv_v、__VXE__、__POWER9_VECTOR__、__loongarch_sx、__wasm_simd128__。',
    src=XQ, parts=[(701, 746)], duration=24000,
    mark_src=[701, 706, 712, 713, 718, 720, 727, 731, 735, 738],
    notes_src={701: '函数名与签名：七个架构文件里逐字相同（bx/by/bs/nrc 都在）',
               706: 'x86 的前置条件：一次只算一行（nrc == 1）',
               712: 'vx 是 Q4_0 权重块数组，vy 是 Q8_0 激活块数组',
               718: '#if defined(__AVX2__)：同一个函数体内部，再按编译期宏分一次',
               727: 'bytes_from_nibbles_32：把 4 bit 解包成 32 个字节',
               731: '重新居中到 [-8, +7]：这是 Q4_0 的零点约定',
               738: '_mm256_fmadd_ps：一次做 8 个 float 的融合乘加'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:10px;align-items:flex-start">
    <div class="grow" id="tbl"></div>
    <div class="col" id="side" style="width:226px;gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '同名函数所在行', '守卫宏'],
  [['arch/x86/quants.c', '701', '__AVX2__'],
   ['arch/arm/quants.c', '297', '__ARM_NEON'],
   ['arch/riscv/quants.c', '222', '__riscv_v'],
   ['arch/s390/quants.c', '217', '__VXE__'],
   ['arch/powerpc/quants.c', '144', '__POWER9_VECTOR__'],
   ['arch/loongarch/quants.c', '647', '__loongarch_sx'],
   ['arch/wasm/quants.c', '232', '__wasm_simd128__']],
  { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--d)">' +
  '<div class="ct" style="color:var(--d)">vec_dot 个数（同文件实测）</div>' +
  '<div class="cb">riscv 31 · arm 25 · x86 24 · powerpc 19 · loongarch 18 · s390 13 · wasm 10。' +
  'RVV 用 <span class="m">vsetvl</span> 写「一次处理多少」，所以能覆盖更多类型组合。</div></div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">内核数 != 速度</div>' +
  '<div class="cb">条数多只说明覆盖广；每条能吃多宽（128/256/512 位）是另一回事。' +
  '下游还有 repack 后的 <span class="m">4x4 / 8x8</span> 版本，见 L5-04。</div></div>';

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '同一个签名，七个文件 —— <b>编译期只会留下其中一个</b>。',
  'x86 这份的 AVX2 路径：解包、居中、乘加、按块累加，全部 256 位。',
  'AVX2 不可用时，同一个函数体里还有 __AVX__ 路径：循环改成 <span class="m">ib += 2</span>，' +
  '<b>两块一起算</b> —— 这就是函数内的第二层分派。',
  'ARM 那份更极端：它有 I8MM 时允许一次算 <b>两</b> 行（nrc == 2），x86 这份 assert(nrc == 1)。',
  '<b>同名函数在不同架构上的「契约」都可能不同</b>；调用方（traits 表）必须知道这一点。',
  '这也解释了为什么要 repack：把权重换成布局友好的形式，同名函数才有 4x4/8x8 的快速版本。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2700 + i * 2900, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(22000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L5-05 · ISA 内核',
    title='十六个 <span class="hl-a">arch/</span> 文件，各自的真实主题',
    sub='七份点积内核 + 四份 repack 内核 + 五份特性探测；repack 只有 x86/arm/riscv/s390 写了。',
    caption='路径省略公共前缀 ggml/src/ggml-cpu/。各文件的 vec_dot / gemv+gemm 条数为本课实测统计。',
    src=WQ, parts=[(26, 39)], duration=24000,
    mark_src=[26, 27, 37, 38],
    notes_src={26: 'wasm 的守卫：__wasm_simd128__（128 位定长）',
               27: 'B1..B8 宏在编译期展开出 256 项的查表常量 —— wasm 没有 NEON 的 vqtbl',
               37: 'table_b2b_0：把 8 bit 展开成 8 字节的预计算表',
               38: 'table_b2b_1：反向（取反再展开）用的第二张表'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div class="formula" id="msg" style="padding:4px 9px"></div>
  <div class="row" style="gap:8px;align-items:flex-start">
    <div class="grow" id="t1"></div>
    <div class="grow" id="t2"></div>
  </div>`;
root.appendChild(wrap);

const rowsA = [
  ['arch/x86/quants.c', 'AVX2/AVX512 点积 · 24'],
  ['arch/x86/repack.cpp', 'VNNI repack 内核 · 12'],
  ['arch/x86/cpu-feats.cpp', 'CPUID 探测 + 打分'],
  ['arch/arm/quants.c', 'NEON/i8mm 点积 · 25'],
  ['arch/arm/repack.cpp', 'ARM repack 内核 · 32'],
  ['arch/arm/cpu-feats.cpp', 'aarch64 特性打分'],
  ['arch/riscv/quants.c', 'RVV 点积（vsetvl）· 31'],
  ['arch/riscv/repack.cpp', 'RVV repack，含 16x1 · 13']
];
const rowsB = [
  ['arch/riscv/cpu-feats.cpp', 'hwprobe 探 RVV'],
  ['arch/s390/quants.c', 'VXE/VXE2 点积 · 13'],
  ['arch/s390/repack.cpp', '最小 repack 集 · 3'],
  ['arch/s390/cpu-feats.cpp', 'auxv 探 VXE2/NNPA'],
  ['arch/powerpc/quants.c', 'VSX 点积 · 19'],
  ['arch/powerpc/cpu-feats.cpp', '按平台版本打分'],
  ['arch/loongarch/quants.c', 'LSX/LASX 点积 · 18'],
  ['arch/wasm/quants.c', 'wasm simd128 点积 · 10']
];
const t1 = U.table(['文件', '真实主题'], rowsA, { monoCols: [0] });
const t2 = U.table(['文件', '真实主题'], rowsB, { monoCols: [0] });
wrap.querySelector('#t1').appendChild(t1.el);
wrap.querySelector('#t2').appendChild(t2.el);

const msg = wrap.querySelector('#msg');
const texts = [
  '先记住三类：<b>quants.c 管点积，repack.cpp 管重排后的 GEMM，cpu-feats.cpp 管打分</b>。',
  'repack 的覆盖面小得多：<b>只有 x86 / arm / riscv / s390 四家写了</b>，其余架构只有 generic。',
  '本幕底部引用的 wasm/quants.c 是「同一个点积、不同工具」的好例子：没有查表指令就用宏展开常量表。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(8200, () => { msg.innerHTML = texts[1]; });
tl.at(16800, () => { msg.innerHTML = texts[2]; });
tl.at(22000, () => { msg.innerHTML = '下一步：厂商加速怎么把整颗算子接过去。'; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L5-05 · AMX',
    title='AMX：用 <span class="hl-e">tensor_traits</span> 接管整颗 MUL_MAT',
    sub='不新增类型、不改进 traits 表：注册一个 buffer type，让权重在写入时就重排成 VNNI 格式。',
    caption='mmq.cpp 的矩阵乘用 8 个 tile 寄存器（TMM0-TMM7）：_tile_loadd 装 A/B，_tile_dpbssd 做点积，'
            '_tile_stored 存回；每个线程先 ggml_tile_config_init() 配一次 tile 形状（2477-2479 行）。',
    src=AX, parts=[(19, 42)], duration=22000,
    mark_src=[19, 23, 24, 29, 30, 31, 38, 39, 40],
    notes_src={19: '整份 AMX 支持都在这一个编译期条件里：__AMX_INT8__ 且 __AVX512VNNI__',
               23: 'tensor_traits：ggml 的算子级钩子（L5-04 讲的 traits 机制）',
               24: 'work_size：这次 MUL_MAT 需要多少 Op 工作区',
               30: '只拦 GGML_OP_MUL_MAT；其它算子返回 false，交回默认路径',
               31: '真正的实现在 mmq.cpp：ggml_backend_amx_mul_mat',
               39: '一个静态实例：AMX 路径下所有张量共用同一套 traits'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px;align-items:flex-start">
    <div class="col grow" id="chain" style="gap:6px"></div>
    <div class="col" id="side" style="width:250px;gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const chain = wrap.querySelector('#chain');
chain.innerHTML = '<div class="cm" style="margin:0 0 2px">一个权重张量在 AMX 路径上的一生</div>';
const steps = [
  { c: 'a', t: '① ggml_backend_amx_buffer_type()', b: '注册 buffer type（名字 AMX，121-125 行）；初始化时申请 XTILEDATA 权限' },
  { c: 'b', t: '② init_tensor -> tensor->extra', b: '张量一进 AMX buffer，就挂上 tensor_traits 实例' },
  { c: 'c', t: '③ set_tensor -> convert_weight', b: '权重写入时直接重排成 VNNI 打包格式（不是原始块布局）' },
  { c: 'd', t: '④ supports_op 判形状', b: 'ne[0] % (TILE_N*2) 必须为 0；类型要在白名单里（qtype_has_amx_kernels）' },
  { c: 'e', t: '⑤ compute_forward 拦 MUL_MAT', b: 'tile 矩阵乘：TILE_M=16, TILE_N=16, TILE_K=32' }
];
const els = steps.map(s => { const e = U.card(s, { style: 'width:100%' }); chain.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">AMX 支持的量化类型</div>' +
  '<div class="cb"><span class="m">qtype_has_amx_kernels()</span>（amx/common.h）只认 7 种：' +
  'Q4_0 / Q4_1 / Q8_0 / Q4_K / Q5_K / Q6_K / IQ4_XS。' +
  '不在名单里就退回普通 buffer type。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">和上一课的关系</div>' +
  '<div class="cb">L5-04 讲的 repack 是「默认 buffer type 里的重排」；AMX 把它换成了' +
  '<b>另一个 buffer type + 另一张 traits 表</b>，粒度更粗、也更强。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  'AMX 不是「再加一个 SIMD 分支」，而是<b>换掉数据放置方式 + 换掉算子实现</b>。',
  '关键在 ③：权重在 set_tensor 时就被重排，之后的每次 MUL_MAT 都直接吃打包格式。',
  '关键在 ④：<span class="v">supports_op</span> 用一个<b>形状判据</b>决定这颗算子归不归 AMX —— ' +
  '这就是「厂商层是在二维表上再插一层」的具体形态。',
  '没命中判据的张量仍然走默认 buffer type，两套路径可以在同一个模型里共存。',
  '代价：权重被打包成 AMX 专用格式，<b>不能再给别的后端用</b>（buffer type 绑定）。'
];
els.forEach((e, i) => tl.at(700 + i * 3300, () => {
  els.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.30'; });
  msg.innerHTML = texts[Math.min(i, 3)];
}));
tl.at(20000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L5-05 · SpacemiT',
    title='SpacemiT：<span class="hl-f">编译期 × 运行期 × 类型</span> 的三级选择链',
    sub='RISC-V 厂商把矩阵扩展（IME1/IME2）做成了自定义指令；选中不了就 abort，不静默降级。',
    caption='本幕代码窗是链条的最后两级（315-329 行）。上一级（266-313 行，IME2 分支）见 source.md 的逐字引文。',
    src=SMI, parts=[(315, 329)], duration=22000,
    mark_src=[315, 316, 317, 320, 322, 323, 327, 328],
    notes_src={315: '第一级（编译期）：只有定义了 RISCV64_SPACEMIT_IME1 这段才存在',
               316: '第二级（运行期）：use_ime1 来自 /proc/cpuinfo 的核型号探测，且要求前面没人选中（!set_kernel_impl）',
               317: 'RVV 负责把激活量化成 int8 —— IME 只做整数矩阵乘，前处理仍走 RVV',
               320: '第三级（类型）：constexpr 判断权重块类型，只有 q4 家族交给 IME1',
               322: 'gemm_kernel_i8i4：IME1 的整数 GEMM 实现',
               327: '任何一级不匹配都走到这里',
               328: 'GGML_ABORT —— 宁可崩，也不悄悄退回慢路径（否则性能问题会查不出来）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:10px;align-items:flex-start">
    <div class="col grow" id="lvl" style="gap:6px"></div>
    <div class="col" id="side" style="width:244px;gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const lvl = wrap.querySelector('#lvl');
lvl.innerHTML = '<div class="cm" style="margin:0 0 2px">forward_mul_mat 里的三级选择（自上而下）</div>';
const lv = [
  { c: 'a', t: 'L1 编译期宏', b: '#if defined(RISCV64_SPACEMIT_IME2) / IME1 —— 决定哪些代码存在' },
  { c: 'b', t: 'L2 运行期核型号', b: 'use_ime2 / use_ime1：探测出的首选核架构是 a100 还是 a60/x100' },
  { c: 'c', t: 'L3 权重块类型', b: 'if constexpr 逐个匹配 block_q4_0 / q6_K / q2_K / q3_K / mxfp4 ...' },
  { c: 'd', t: '兜底', b: 'set_kernel_impl 仍为 false -> GGML_ABORT（不降级）' }
];
const els = lv.map(s => { const e = U.card(s, { style: 'width:100%' }); lvl.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">核型号从哪来？</div>' +
  '<div class="cb">ime_env.cpp 读 <span class="m">/proc/cpuinfo</span> 的 processor/marchid 对，' +
  '映射成 <span class="m">spine_core_arch_id</span> 枚举（x60/x100/x200/a60/a100/a200），' +
  '再由 ime.cpp 用 <span class="m">pthread_setaffinity_np</span> 把线程绑到首选核（1711 行）。</div></div>' +
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">"面板 + RVV 前处理"</div>' +
  '<div class="cb">权重在 repack 时变成 <span class="m">nrow_block_*</span> 布局（ime_kernels.h），' +
  '激活由 rvv_kernels.cpp 量化成 int8 —— <b>IME 只吃两边都排好的整数</b>。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '同一台机器上可能同时有 IME2 核和 IME1 核：<b>选哪一级既要看编译了什么，也要看跑在哪个核上</b>。',
  '这不是「二维表」，而是把二维表<b>按类型再展开一次</b>：BLOC_TYPE 是模板参数，每个类型一份实例。',
  '注意每一级都用 <span class="v">!set_kernel_impl &amp;&amp;</span> 串联：先到先得，后面的不覆盖前面的。',
  '最后一行是这节课最值得记的工程选择：<b>没有可用的 IME 内核就直接 abort</b>，' +
  '因为「悄悄变慢」比「明确失败」更难查。',
  '对照 KleidiAI：同样是厂商层，它选择「没命中就返回 false，退回默认路径」—— 两种策略各有代价。'
];
els.forEach((e, i) => tl.at(700 + i * 3400, () => {
  els.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.32'; });
  msg.innerHTML = texts[Math.min(i, 3)];
}));
tl.at(19000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L5-05 · 厂商层',
    title='剩下 26 个厂商/第三方文件，各自的真实主题',
    sub='amx 5 个、kleidiai 4 个、spacemit 15 个、llamafile 2 个 —— 四家四种接法。',
    caption='路径省略公共前缀 ggml/src/ggml-cpu/。llamafile 是 Mozilla 的 tinyBLAS：'
            '一份小矩阵 SGEMM，按 A/B 类型 switch、再按 ISA 分派（见 source.md 引文）。',
    src=IQPH, parts=[(8, 27)], duration=24000,
    mark_src=[8, 9, 15, 16, 18, 21, 24, 27],
    notes_src={8: 'iqp.cpp/h 是「IQ 面板」路径：按 8 行权重解码成 int8 面板，再做整数 GEMM',
               9: 'block_iqp_x8 就是那个面板：8 行的 int8 权重交错存放',
               16: '按专家（expert）算的最小批量门槛',
               18: '节点级判据：这个算子能不能走 IQP',
               24: '每线程面板 scratch 大小',
               27: '真正的实现入口：src1 必须先转成 q8_K 并同步好'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div class="formula" id="msg" style="padding:4px 9px"></div>
  <div class="row" style="gap:8px;align-items:flex-start">
    <div class="grow" id="t1"></div>
    <div class="grow" id="t2"></div>
  </div>`;
root.appendChild(wrap);

const rowsA = [
  ['amx/mmq.cpp', 'AMX tile 矩阵乘内核'],
  ['amx/amx.cpp', 'AMX buffer type + traits'],
  ['amx/common.h', 'tile 常量与并行原语'],
  ['amx/mmq.h', 'AMX 四个入口声明'],
  ['amx/amx.h', 'buffer type 声明'],
  ['kleidiai/kleidiai.cpp', '特性探测 + 权重 repack'],
  ['kleidiai/kernels.cpp', '按 CPU 特性选 kernel'],
  ['kleidiai/kernels.h', 'kernel 描述符结构'],
  ['kleidiai/kleidiai.h', 'buffer type 声明'],
  ['spacemit/ime.cpp', 'IME 三级分派 + traits'],
  ['spacemit/ime_env.cpp', 'Spine 核探测与绑核'],
  ['spacemit/ime_env.h', 'core_arch_id 枚举'],
  ['spacemit/ime1_kernels.cpp', 'IME1 GEMM 内联汇编']
];
const rowsB = [
  ['spacemit/ime2_kernels.cpp', 'IME2 vmadot 汇编内核'],
  ['spacemit/ime_kernels.h', 'nrow 重排块布局'],
  ['spacemit/ime.h', 'SpacemiT 入口声明'],
  ['spacemit/rvv_kernels.cpp', 'RVV flash-attn/量化'],
  ['spacemit/rvv_kernels.h', 'RVV 内核声明'],
  ['spacemit/repack.cpp', '权重重排成 nrow 块'],
  ['spacemit/repack.h', 'repack 模板声明'],
  ['spacemit/spine_mem_pool.cpp', 'THP/hugetlb/TCM 池'],
  ['spacemit/spine_mem_pool.h', '内存池后端枚举'],
  ['spacemit/spine_barrier.h', '跨核自旋屏障'],
  ['spacemit/spine_tcm.h', 'TCM 头文件加载器'],
  ['llamafile/sgemm.cpp', 'tinyBLAS 小矩阵 SGEMM'],
  ['llamafile/sgemm.h', 'llamafile_sgemm 声明']
];
const t1 = U.table(['文件', '真实主题'], rowsA, { monoCols: [0] });
const t2 = U.table(['文件', '真实主题'], rowsB, { monoCols: [0] });
wrap.querySelector('#t1').appendChild(t1.el);
wrap.querySelector('#t2').appendChild(t2.el);

const msg = wrap.querySelector('#msg');
const texts = [
  '四种接法的共同点：<b>都是 buffer type + traits</b>；差别在「谁来做重排」和「不命中怎么办」。',
  'AMX 把重排做在 set_tensor；KleidiAI 用 kai_* 库函数重排；SpacemiT 有自己的 nrow 块格式；' +
  'llamafile 干脆不重排，只在<b>小矩阵</b>上换一套 GEMM。',
  '底部引用的 iqp.h 是第五种接法：不改 buffer、不改 traits，只在批量够大时<b>换掉 GEMM 的组织方式</b>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(9000, () => { msg.innerHTML = texts[1]; });
tl.at(18000, () => { msg.innerHTML = texts[2]; });
tl.at(22500, () => { msg.innerHTML = '最后一幕：把 49 个文件压成两栏，并做一道练习。'; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L5-05 · 收束',
    title='收束：最后 7 个文件 + 这张表的四层',
    sub='剩下的是三条垫层和两条旁路；hbm 与 iqp 都不碰 SIMD，它们改的是内存与 GEMM 组织。',
    caption='下一课 L6-01 换到 GPU：同样的「一个算子多种内核」问题，在 CUDA 里由 stream 与 kernel 选择解决。',
    src=HBM, parts=[(40, 54)], duration=24000,
    mark_src=[40, 41, 43, 44, 45, 46, 53],
    notes_src={40: 'hbm.cpp 与 SIMD 无关：它是一个 buffer type 的注册函数',
               41: '和 AMX / KleidiAI / SpacemiT 一样，是一个 static 的 ggml_backend_buffer_type',
               44: '分配走 hbw_posix_memalign（高带宽内存分配器，memkind）',
               45: '对齐复用 CPU 默认 buffer type 的对齐要求',
               46: 'get_max_size / get_alloc_size 留空 = 用默认值（SIZE_MAX / ggml_nbytes）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px;align-items:flex-start">
    <div class="grow" id="t1"></div>
    <div class="grow" id="t2"></div>
  </div>
  <div id="ex"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const rowsA = [
  ['arch-fallback.h', '缺失内核改名表'],
  ['common.h', '类型转换表 + 线程切分'],
  ['ggml-cpu-impl.h', '跨 ISA 内建函数垫层'],
  ['hbm.cpp', 'HBM buffer（hbwmalloc）'],
  ['hbm.h', 'HBM buffer 声明'],
  ['iqp.cpp', 'IQ 面板 int8 GEMM'],
  ['iqp.h', 'IQP 入口与判据']
];
const rowsB = [
  ['1 · ISA 内核', '编译期：arch/<isa>/'],
  ['2 · 内核变体', '运行期：变体打分'],
  ['3 · 厂商加速', '运行期：buffer type'],
  ['4 · 专用旁路', '运行期：判据 + 换路']
];
const t1 = U.table(['文件（共 7 个）', '真实主题'], rowsA, { monoCols: [0] });
const t2 = U.table(['决策表的四层', '什么时候定 / 载体'], rowsB);
wrap.querySelector('#t1').appendChild(t1.el);
wrap.querySelector('#t2').appendChild(t2.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '同一台 x86 机器上，<span class="mono">GGML_OP_MUL_MAT</span>（权重 Q4_0）可能落到哪几个不同的 kernel 上？' +
  '说出至少三条互不相同的路径，并指出每条路径的<b>判据</b>写在哪个文件、属于哪一层。',
  '本课引用的候选（顺序由 L5-01 的 ggml-cpu.c 决定，此处不排序）：<br>' +
  '① <b>AMX</b>：权重在 AMX buffer type 里（<span class="mono">src0->buffer->buft == ggml_backend_amx_buffer_type()</span>）、' +
  '两个输入都连续、<span class="mono">ne[0] % (TILE_N * 2) == 0</span>、类型在 7 种白名单内 —— ' +
  '判据在 <span class="mono">amx.cpp</span> 的 <span class="mono">extra_buffer_type::supports_op</span>（145-190 行，厂商层）。<br>' +
  '② <b>IQP 面板路径</b>：类型是 8 种 grid IQ 之一、<span class="mono">vec_dot_type</span> 是 Q8_K、' +
  '运行时 <span class="mono">ggml_cpu_has_avx2()</span> 为真、src1 批量 >= 8 —— ' +
  '判据在 <span class="mono">iqp.cpp</span> 的 <span class="mono">ggml_cpu_iqp_supports_mul_mat</span>（1095-1113 行，旁路）。<br>' +
  '③ <b>llamafile tinyBLAS</b>：Atype=Q4_0 时选 <span class="mono">tinyBLAS_Q0_AVX</span>；' +
  '<span class="mono">n &lt; 2</span> 或 <span class="mono">Ctype != F32</span> 时直接 <span class="mono">return false</span> —— ' +
  '判据在 <span class="mono">sgemm.cpp</span>（3820 行、4078-4113 行，第三方层）。<br>' +
  '④ <b>兜底</b>：<span class="mono">ggml_vec_dot_q4_0_q8_0</span>（x86 那份的 AVX2 分支）—— ' +
  '编译期由 <span class="mono">arch-fallback.h</span> 与 <span class="mono">__AVX2__</span> 定死（ISA 内核层）。<br>' +
  '要点：四个候选的判据分属四层，而且<b>只有 ④ 是编译期决定的</b>。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '49 个文件 = <b>16 个 ISA 内核/探测 + 4 类厂商加速 + 3 个垫层 + 2 条旁路</b>。',
  '一句话：<b>「选哪个 kernel」是编译期 + 运行期两张表；厂商加速是在这两张表上再插一层。</b>',
  '最后两条旁路提醒你：性能不只有 SIMD —— hbm 换内存来源，iqp 换 GEMM 组织，' +
  'llamafile 换小矩阵算法，它们和向量宽度无关。',
  '复习线索：L5-03 讲 vec_dot 的 SIMD 抽象层；L5-04 讲 traits 与 repack；L3-04 讲后端特性上报；' +
  '本课讲这些指针背后按架构各写一份的实现体。',
  '下一课 L6-01：换到 GPU，同一个问题变成「一个算子对应哪些 CUDA kernel」。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(6000, () => { msg.innerHTML = texts[1]; });
tl.at(11000, () => { msg.innerHTML = texts[2]; });
tl.at(16500, () => { msg.innerHTML = texts[3]; });
tl.at(21500, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、总览：49 个文件怎么分工',
    '本课的文件可以按**「它回答哪个问题」**分成四类，后面每一节都按这个顺序读：\n\n'
    '| 类 | 文件数 | 回答的问题 |\n|---|---|---|\n'
    '| ISA 内核与探测 | 16 | 这个架构上，这个算子用哪条指令序列？ |\n'
    '| 厂商/第三方加速 | 26 | 有没有更强的一整块硬件/库可以接管？ |\n'
    '| 通用垫层 | 3 | 没有原生实现时，谁来顶？跨 ISA 的名字怎么统一？ |\n'
    '| 专用旁路 | 4 | 不换指令、换内存或换 GEMM 组织，能不能更快？ |\n\n'
    '`arch-fallback.h` 是理解前两类的钥匙：它把「编译期选择」写成了**差集**。',
    src=FB, parts=[(2, 12)], lang='c')

L.section(
    '二、★ 编译期那一维：差集表',
    '每个架构分支只列出**本架构没有的原生实现**。ARM 只有 8 条（全在 repack.cpp 段），'
    'x86 有 29 条（quants.c 段只缺 `q2_0` 一条），其余架构更多。'
    '统计口径：对该文件各分支的 `#define` 计数（本课实测）。\n\n'
    '```text\n'
    'GENERIC 68   wasm 55   s390x 50   loongarch64 48\n'
    'powerpc 47   riscv 41  x86_64 29   aarch64 8\n'
    '```\n\n'
    '配合 `ggml-cpu/quants.c`（属 L5-04）里定义的 `*_generic` 版本，就能看出完整机制：'
    '**原生实现写正式名字，通用实现写 `_generic` 后缀，arch-fallback.h 决定这一份编译里谁叫正式名字。**',
    src=FB, parts=[(88, 119)], lang='c')

L.section(
    '三、运行期那一维：变体打分',
    '同一架构可以编出多份变体（不同的 `-march`），每份导出一个 `ggml_backend_score()`。'
    '分数是**位图 + 1**：基线 1 分，每满足一项或上一位。'
    '每个 `#ifdef` 都配一次 `if (!is.X()) return 0;` —— 缺一项不是降级，而是这份变体直接出局。\n\n'
    '读分数的是 `ggml/src/ggml-backend-reg.cpp`（属 L3-02，本课不引用其文本）：'
    '遍历候选动态库、调 `ggml_backend_score`、取最高分并 `dlopen`。'
    '所以「选 kernel」在运行期至少发生两次：先选变体，再选内核。',
    src=XF, parts=[(263, 325)], lang='c')

L.section(
    '四、厂商层之一：KleidiAI 的 required_cpu 表',
    'KleidiAI（Arm）不往 traits 里堆 if，而是把「我能跑什么」写成**有序表**：'
    '每项有 `required_cpu` 位掩码 + lhs/rhs/op 三个类型，**第一条命中就返回**，'
    '全表不命中则返回 `nullptr`（这一层不接管，退回默认路径）。',
    src=KLK, parts=[(1035, 1062)], lang='c')

L.section(
    '五、KleidiAI：特性掩码从哪来',
    '掩码由 aarch64 运行期探测拼出来：DOTPROD / I8MM / FP16 / SVE 各占一位（SVE 还要求 '
    '`sve_cnt` 正好等于 QK8_0）。这一段还展示了三个调试用的环境变量覆盖 '
    '（`GGML_KLEIDIAI_SME` / `GGML_TOTAL_THREADS` / `GGML_KLEIDIAI_CHUNK_MULTIPLIER`）：'
    '**厂商路径一定要能关**。\n\n'
    'SME 家族的处理在这段之后（338-355 行）：先数出可用的流式模式计算单元（SMCU），'
    '数不到就保守地把 SME 线程上限设为 1 —— 因为 SME 的流式模式是有限硬件资源。',
    src=KLD, parts=[(301, 320)], lang='c++')

L.section(
    '六、厂商层之二：SpacemiT 的三级选择链',
    'RISC-V 厂商 SpacemiT 把矩阵扩展（IME1/IME2）做成了自定义指令。'
    '`forward_mul_mat` 里的选择链是三级：**编译期宏 -> 运行期核型号 -> 权重块类型**，'
    '每级用 `!set_kernel_impl &&` 串联（先到先得），全不中就 `GGML_ABORT`。\n\n'
    '注意第一级（IME2 段）里 `INTER_SIZE == 256` 还会再分一次 `_hp`（高精度）路径：'
    '同一类型在不同块大小下也能走不同内核。',
    src=SMI, parts=[(266, 313)], lang='c++')

L.section(
    '七、同名不同物（上）：x86 的 ggml_vec_dot_q4_0_q8_0',
    '函数名和签名在七个架构文件里**逐字相同**：`(int n, float * s, size_t bs, const void * vx, '
    'size_t bx, const void * vy, size_t by, int nrc)`。'
    'x86 这份在本文件内再按 `__AVX2__` / `__AVX__` / SSE 分派，并断言 `nrc == 1`（一次一行）。',
    src=XQ, parts=[(701, 746)], lang='c')

L.section(
    '八、同名不同物（下）：ARM 的同名函数，契约不同',
    'ARM 这份在 `__ARM_FEATURE_MATMUL_INT8` 下允许 `nrc == 2`（一次两行），'
    '并从 `vx + bx`、`vy + by` 取第二行 —— 这正是 x86 那份 `UNUSED(bx)` 的字段。'
    '**同名函数在不同架构上的契约可以不同**，调用方必须知道这一点。\n\n'
    '实验：把两份文件并排看，只用 `grep -n "void ggml_vec_dot_q4_0_q8_0" ggml/src/ggml-cpu/arch/*/quants.c` '
    '就能拿到七个行号：x86:701、arm:297、riscv:222、s390:217、powerpc:144、loongarch:647、wasm:232。',
    src=AQ, parts=[(297, 321)], lang='c')

L.section(
    '九、AMX：把整颗 MUL_MAT 接过去',
    'AMX（Intel）不新增类型、也不改 traits 表，而是注册一个 buffer type'
    '（`get_name` 返回 `"AMX"`，见 121-125 行）：'
    '权重在 `set_tensor` 时被重排成 VNNI 打包格式，`compute_forward` 里只拦 `GGML_OP_MUL_MAT`。'
    '这是「厂商层插在二维表之上」的最清晰形态：**判据是形状与类型白名单，代价是权重格式被绑定**。',
    src=AX, parts=[(19, 42)], lang='c++')

L.section(
    '十、AMX 的参数：tile 形状与类型白名单',
    'tile 常量（16/16/32）、8 个 TMM 寄存器编号、以及「哪些量化类型有 AMX 内核」的白名单都在这里。'
    '白名单只有 7 种类型 —— 不在名单里的张量根本不进 AMX buffer。',
    src=AXC, parts=[(16, 30)], lang='c')

L.section(
    '十一、AMX 的判据：supports_op 到底看什么',
    '这段是「厂商层怎么决定接管哪颗算子」的完整答案：算子必须是 `GGML_OP_MUL_MAT`、'
    '两个输入都连续、`src0` 必须身处 AMX buffer type、`src1` 必须在主机内存、'
    '`ne[0]` 必须是 `TILE_N * 2` 的整数倍、类型必须落在白名单里（每种类型还有各自的对齐要求）。'
    '任何一条不满足就返回 `false`，这颗算子交回默认路径 —— 于是**两套路径能在同一个模型里共存**。',
    src=AX, parts=[(144, 199)], lang='c++')

L.section(
    '十二、垫层：ggml-cpu-impl.h 与 common.h',
    '`ggml-cpu-impl.h` 是整个 CPU 后端的**跨 ISA 垫层**：一个头文件把 7 种 ISA 的内建函数头都引进来，'
    '并在没有硬件指令时用 C 写出等价实现（例如没有 DOTPROD 时用 `vmull_s8` 拼一个 `ggml_vdotq_s32`）。'
    '`common.h` 则提供模板内核需要的类型转换表（f16/bf16/f32/i32 的 to_f32/from_f32）与线程切分函数。',
    src=IMPL, parts=[(342, 350)], lang='c')

L.section(
    '十三、垫层的另一半：类型转换表与线程切分',
    '`type_conversion_table<T>` 把「模板参数 T」映射到「怎么转 float」，'
    '`get_thread_range` 把行按线程数切开（每线程一段连续行）。'
    '这两个看起来平淡的工具，是后面所有 `template <typename T>` 内核的共同底座。',
    src=CMN, parts=[(46, 88)], lang='c++')

L.section(
    '十四、旁路之一：hbm.cpp 根本不碰 SIMD',
    '`hbm.cpp` 只在 `GGML_USE_CPU_HBM` 下编译，做的是**另一种内存来源**：'
    '用 `hbw_posix_memalign` 从高带宽内存（HBM）分配，释放时用 `hbw_free`。'
    '它注册成一个 buffer type，名字 `CPU_HBM`；除了分配器，其余接口全部复用 CPU 默认实现。\n\n'
    '**它不是 SIMD 优化，而是访存优化**；把它放在这一课，是因为它和 AMX/KleidiAI 用了同一个挂载点。',
    src=HBM, parts=[(1, 12)], lang='c')

L.section(
    '十五、旁路之二：iqp.cpp 换的是 GEMM 的组织方式',
    'IQP 路径针对**基于网格（grid）的 IQ 类型**（IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS）。'
    '它不重排权重、不注册 buffer type，而是把 **8 行 src0 一次性解码成 int8 面板**（`block_iqp_x8`），'
    '再对所有 src1 列做整数 GEMM —— 面板让一次解码服务多列，因此**只在批量足够大时才划算**。\n\n'
    '激活侧用 VNNI 时按无符号字节喂（y + 128）再用 `bias[]` 校正；没有 VNNI 时改用 maddubs 的符号技巧。',
    src=IQP, parts=[(30, 52)], lang='c')

L.section(
    '十六、iqp.h：判据与入口',
    '头文件把这条路径的契约写得很清楚：什么时候划算、节点级判据、每线程 scratch 大小、'
    '以及「必须等 src1 转成 q8_K 并同步之后」才能调用实现。',
    src=IQPH, parts=[(8, 27)], lang='c')

L.section(
    '十七、iqp.cpp：判据长什么样',
    'IQP 的判据不是形状白名单，而是**一批同时成立的条件**：类型在 8 种 grid IQ 里、'
    '`vec_dot_type` 必须是 Q8_K（这条路径假设 src1 会转成 q8_K）、运行时必须真有 AVX2、'
    '`src1` 必须是 F32、`ne[0]` 要能整除 QK_K、`ne[1]` 要能整除 8、`src0` 连续、`dst` 是 F32 连续。'
    '外加一个**逃生开关**：环境变量 `GGML_NO_IQ_PANEL` 一旦设置，这条路径整体关闭（用于 A/B 对比）。',
    src=IQP, parts=[(1053, 1113)], lang='c')

L.section(
    '十八、llamafile：小矩阵的第三套 SGEMM',
    '`llamafile_sgemm` 是 Mozilla tinyBLAS 的入口：**先按 A/B 类型 switch，再按 ISA 分派**，'
    '同一类型组合在不同架构下实例化不同模板。它只在 `n >= 2`（提示处理）时接管，'
    '小矩阵形状不合适就返回 false 让默认路径接手。\n\n'
    '本课引用其头文件签名与实现文件里的分派骨架，完整模板体（4165 行）留作课外阅读。',
    src=LSGH, parts=[(19, 21)], lang='c')

L.section(
    '十九、llamafile：按类型 switch、再按 ISA 分派',
    '下面这段是 `llamafile_sgemm` 的骨架（以 Atype = Q8_0 为例）：**外层 switch 查类型组合，'
    '每个 case 内部再用 `#if defined(ISA)` 选模板实例**。'
    '三种 ISA 各有一个 tinyBLAS 实现（AVX / ARM DOTPROD / PowerPC MMA），'
    '都不匹配就 `return false`，让上层回到默认路径。',
    src=LSG, parts=[(4041, 4076)], lang='c++')

L.section(
    '二十、llamafile：只在提示处理（n >= 2）时接管',
    '入口处先做两件事：`Ctype` 必须是 F32，且（非 MMA 平台）`n >= 2` —— '
    '也就是只在「一次算多列」的提示处理阶段才值得用它；解码阶段（n = 1）直接返回 false。'
    '这一行注释就是它的适用场景说明。',
    src=LSG, parts=[(3818, 3825)], lang='c++')

L.section(
    '二十一、SpacemiT：核型号探测与绑核',
    'SpacemiT 的路径要先知道「这颗 SoC 上哪些核是 x100/a100」：'
    '`ime_env.cpp` 逐行读 `/proc/cpuinfo`，把 `processor` 与 `marchid` 配成对，'
    '再映射成 `spine_core_arch_id` 枚举（x60/x100/x200/a60/a100/a200）；'
    '`/proc/cpuinfo` 读不到时（例如 QEMU）改用环境变量注入。绑核发生在 ime.cpp：'
    '用 `pthread_setaffinity_np` 把线程钉在首选核上（1711 行）；共享内存/大页/TCM 的选择也在这里定。',
    src=SME, parts=[(259, 266)], lang='c++')

L.section(
    '二十二、SpacemiT 的内存池与屏障',
    '`spine_mem_pool` 提供三种后端：`posix_memalign`、透明大页（`madvise(MADV_HUGEPAGE)`）、'
    '1G 大页（`/dev/hugetlb_1g` + ioctl + mmap），另有按核分配的 TCM（紧耦合内存，'
    '通过可 dlopen 的 `spine_tcm` 库头文件方式加载）。跨核同步不用 pthread，而是自己的自旋屏障。',
    src=SMMH, parts=[(8, 20)], lang='c++')

L.footnote_add('本课覆盖 `plan_matrix.py` 分配给 L5-05 的全部 **49 个源文件**，逐一声明于上表；'
               '其中 `arch/` 子树 16 个（7 个 `quants.c` + 4 个 `repack.cpp` + 5 个 `cpu-feats.cpp`）、'
               '`amx/` 5 个、`kleidiai/` 4 个、`spacemit/` 15 个、`llamafile/` 2 个、'
               '`hbm.*` 2 个、`iqp.*` 2 个，加 `arch-fallback.h`、`common.h`、`ggml-cpu-impl.h`。')
L.footnote_add('文中提到的 `ggml/src/ggml-cpu/traits.cpp`、`ggml/src/ggml-cpu/quants.c`（`*_generic` 定义）、'
               '`ggml/src/ggml-cpu/repack.cpp`（默认 buffer type 的 repack）都属于 **L5-04**；'
               '`ggml/src/ggml-backend-reg.cpp`（变体加载器）属 L3-02；`ggml/src/ggml-cpu/ggml-cpu.cpp`'
               '（extra buffer type 注册）属 L5-01。本课只在正文中引用它们的机制，'
               '**不引用其文本，故不计入本课覆盖率**。')
L.footnote_add('本课所有「条数」统计（各架构 `vec_dot` 个数、repack 内核个数、arch-fallback.h 各分支 '
               '`#define` 条数、同名函数行号）都由 v0.5.0 源码实测得出，可用 `grep -c` / 行号核对；'
               '它们不是源码注释里的说法。')

L.prereqs('`L5-04`（量化与 repack）')

L.goal(
    '说出同一个算子（如 Q4_0 权重的 `MUL_MAT`）在不同架构上分别落到哪个 kernel 文件（对应验收点）；',
    '解释 `arch-fallback.h` 为什么用「差集」而不是「清单」来表达编译期选择；',
    '说明 `arch/*/cpu-feats.cpp` 的 score 如何决定运行期使用哪一份 CPU 变体；',
    '区分厂商层的两种失败策略：KleidiAI 返回 `nullptr` 退回默认路径 vs SpacemiT 直接 `GGML_ABORT`；',
    '说出 `hbm.cpp` 与 `iqp.cpp` 各自优化的是什么（内存来源 / GEMM 组织），以及它们为什么不算 SIMD 优化。')

L.conclusion(
    '★ 核心：一张二维决策表 + 一层可插拔的厂商层',
    '```text\n'
    '维度一（编译期）：arch-fallback.h 决定本架构编进哪些原生实现\n'
    '                  -> 链接后 ggml_vec_dot_* / ggml_gemv_* 各只有一个定义\n'
    '维度二（运行期）：arch/*/cpu-feats.cpp 给每份 CPU 变体打分，最高分被 dlopen\n'
    '                  -> 同一颗 CPU 上，二进制里那套实现被选中\n'
    '厂商层（运行期）：buffer type + tensor_traits，用形状/类型判据接管整颗算子\n'
    '                  -> AMX / KleidiAI / SpacemiT / llamafile 四种接法\n'
    '```\n\n'
    '记住一句话：**「选哪个 kernel」不是一次选择，而是两次；厂商加速是在这两次之上再插一层。**')

L.conclusion(
    '七个架构，一个函数名',
    '`ggml_vec_dot_q4_0_q8_0` 在 v0.5.0 有七份实现，行号分别是 x86:701、arm:297、riscv:222、'
    's390:217、powerpc:144、loongarch:647、wasm:232，守卫宏各不相同；'
    '各架构的 `vec_dot` 个数为 riscv 31、arm 25、x86 24、powerpc 19、loongarch 18、s390 13、wasm 10。'
    '**名字相同不代表契约相同**：ARM 在 I8MM 下允许 `nrc == 2`，x86 只允许 `nrc == 1`。')

L.conclusion(
    '厂商层的共同形状：buffer type + traits',
    'AMX、KleidiAI、SpacemiT 三家的代码量差了一个数量级（250 行 / 1900 行 / 1740 行），'
    '但挂载点完全一样：注册一个 `ggml_backend_buffer_type`，在 `init_tensor` 里把 traits 挂到 '
    '`tensor->extra`，在 `set_tensor` 里重排权重，在 `compute_forward` 里拦下算子。'
    '差别只在失败策略与重排格式。')

L.conclusion(
    '两条旁路提醒：性能不只有 SIMD',
    '`hbm.cpp` 换的是内存来源（`hbw_posix_memalign` / `hbw_free`），`iqp.cpp` 换的是 GEMM 的组织方式'
    '（8 行权重解码成 int8 面板 + 整数 GEMM，批量 >= 8 才划算），'
    '`llamafile/sgemm.cpp` 换的是小矩阵算法（tinyBLAS）。三者都不增加向量宽度，'
    '但都在同一个挂载点体系里生效。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
