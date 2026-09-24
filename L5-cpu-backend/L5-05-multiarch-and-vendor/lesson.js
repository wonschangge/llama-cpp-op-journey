/* ==========================================================================
   L5-05 · ★ 多架构 SIMD 与厂商加速
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 ★ 一个算子，<span class="hl-a">两维选择</span>：编译期 + 运行期 */
{
  kicker: "L5 · CPU 后端执行",
  title: "★ 一个算子，<span class=\"hl-a\">两维选择</span>：编译期 + 运行期",
  sub: "49 个文件分成四类：七架构的 ISA 内核、五份特性探测、四种厂商加速、两条旁路。",
  caption: "回顾 L5-04：traits 表把「类型 -> kernel 函数指针」查出来。本课讲那些函数指针指向的、按架构各写一份的实现体是怎么被编进来、又是怎么被选中的。",
  src: "ggml/src/ggml-cpu/arch-fallback.h",
  mark: [2, 6, 9, 13],
  lineNo: 2,
  code: `#pragma once

// Rename \`_generic\` functions if no native implementation is available.
//>> arch-fallback.h 的用途只有一句话：本架构没有原生实现时，把 _generic 版本改名顶上
// This effectively selects the generic implementation.

#if defined(GGML_CPU_GENERIC)
//>> GGML_CPU_GENERIC 分支：通用构建，68 条全部要走改名
// quants.c
#define quantize_row_q8_0_generic quantize_row_q8_0
//>> 改名后 quantize_row_q8_0 这个名字指向通用实现（不是第三份代码）
#define quantize_row_q8_1_generic quantize_row_q8_1
#define quantize_row_q8_K_generic quantize_row_q8_K
#define ggml_vec_dot_q4_0_q8_0_generic ggml_vec_dot_q4_0_q8_0
//>> 每个名字在链接期只能有一个定义 —— 所以“选实现”发生在编译期，运行期改不了`,
  duration: 20000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 2 <span class="hl-a">编译期那一维</span>：每个架构缺哪些内核 */
{
  kicker: "L5-05 · 编译期",
  title: "<span class=\"hl-a\">编译期那一维</span>：每个架构缺哪些内核",
  sub: "arch-fallback.h 是整张表的编译期半边：它逐架构列出「本架构没有的原生实现」。",
  caption: "右侧条数由本课统计（各分支 #define 计数）：GENERIC 68、wasm 55、s390x 50、loongarch64 48、powerpc 47、riscv 41、x86_64 29、aarch64 8。",
  src: "ggml/src/ggml-cpu/arch-fallback.h",
  mark: [0, 2, 4, 12, 14, 15],
  lineNo: 78,
  code: `#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64)
//>> ARM 分支：只有 8 条，而且全在 repack.cpp 段 —— quants.c 一条都不缺
// repack.cpp
//>> 段注释直接说明这 8 条属于哪个文件（repack 的 4x4/4x8 GEMM 变体）
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
//>> x86 分支：29 条。注意 89 行 —— x86 的 quants.c 段只缺 1 条
// quants.c
#define ggml_vec_dot_q2_0_q8_0_generic ggml_vec_dot_q2_0_q8_0
//>> q2_0 是后加的类型，x86 还没写原生实现，于是用 generic 顶上`,
  duration: 24000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 3 运行期那一维：<span class="hl-b">变体打分</span>，缺一项就出局 */
{
  kicker: "L5-05 · 运行期",
  title: "运行期那一维：<span class=\"hl-b\">变体打分</span>，缺一项就出局",
  sub: "同一架构可以编出多份 CPU 变体；每份导出一个 score 函数，谁分高谁上场。",
  caption: "读分数的是后端注册器（ggml/src/ggml-backend-reg.cpp，属 L3-02，本课不引用其文本）：遍历候选动态库、调 ggml_backend_score、取最高分。",
  src: "ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp",
  mark: [0, 2, 8, 19, 21, 24, 26, 33],
  lineNo: 297,
  code: `#ifdef GGML_AVX512
//>> #ifdef GGML_AVX512：这一段代码只在「编 AVX512 变体」时存在
    if (!is.AVX512F()) { return 0; }
//>> 每个 #ifdef 都配一次 if (!is.X()) return 0 —— 缺一项不是降级，是这份变体直接 0 分出局
    if (!is.AVX512CD()) { return 0; }
    if (!is.AVX512VL()) { return 0; }
    if (!is.AVX512DQ()) { return 0; }
    if (!is.AVX512BW()) { return 0; }
    score += 1<<7;
//>> 每一项命中往一个位上打 1：AVX512 拿 1<<7
#endif
#ifdef GGML_AVX512_VBMI
    if (!is.AVX512_VBMI()) { return 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    if (!is.AVX512_BF16()) { return 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    if (!is.AVX512_VNNI()) { return 0; }
    score += 1<<10;
//>> AVX512_VNNI 再往 1<<10 打一位
#endif
#ifdef GGML_AMX_INT8
    if (!is.AMX_INT8()) { return 0; }
    score += 1<<11;
//>> AMX_INT8 是当前 x86 变体的最高位 1<<11
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)
//>> GGML_BACKEND_DL_SCORE_IMPL 把这个函数导出成动态库符号 ggml_backend_score`,
  duration: 22000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-c">厂商层</span>：在二维表上再插一张表 */
{
  kicker: "L5-05 · 厂商层",
  title: "★ <span class=\"hl-c\">厂商层</span>：在二维表上再插一张表",
  sub: "KleidiAI 的做法：把「需要哪些 CPU 特性」写成表项，按顺序试，第一条命中就赢。",
  caption: "三张同构的表：Q4_0 权重（330-707 行）、Q8_0（758-923）、F32（974-1033）。表尾各有一个 { } 哨兵项，所以循环写成 NELEMS(table) - 1。",
  src: "ggml/src/ggml-cpu/kleidiai/kernels.cpp",
  mark: [3, 5, 6, 7, 9, 13, 15, 18],
  lineNo: 1035,
  code: `ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, const ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = nullptr;

    if (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != nullptr && tensor->src[1] != nullptr) {
//>> 只接管 MUL_MAT，而且 src0/src1 都在才算数
        auto try_table = [&](auto & table) {
            for (size_t i = 0; i < NELEMS(table) - 1; ++i) {
                if ((cpu_features & table[i].required_cpu) == table[i].required_cpu &&
//>> 第一个条件：特性位掩码必须覆盖表项要求的全部位（按位与后相等）
                    table[i].lhs_type == tensor->src[1]->type &&
//>> 后三个条件：lhs/rhs/op 三个类型都要对上
                    table[i].rhs_type == tensor->src[0]->type &&
                    table[i].op_type  == tensor->type) {
                    kernel = &table[i];
//>> 第一条命中就返回 —— 表的顺序就是优先级，不是「分数最高者胜」
                    return true;
                }
            }
            return false;
//>> 全表试完还没命中就返回 false，交给上面那两维的默认路径
        };

        if (tensor->src[0]->type == GGML_TYPE_Q8_0) {
            try_table(gemm_gemv_kernels_q8);
        } else if (tensor->src[0]->type == GGML_TYPE_F32) {
            try_table(ggml_kleidiai_kernels_f32);
        } else {
            try_table(gemm_gemv_kernels);
        }
    }

    return kernel;
}`,
  duration: 24000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 5 同一个 <span class="hl-d">ggml_vec_dot_q4_0_q8_0</span>，七份实现 */
{
  kicker: "L5-05 · 同名不同物",
  title: "同一个 <span class=\"hl-d\">ggml_vec_dot_q4_0_q8_0</span>，七份实现",
  sub: "x86 这份在同名文件里按 AVX2/AVX 再分一次；ARM 那份连函数契约都不一样。",
  caption: "七份实现的行号（v0.5.0 实测）：x86:701、arm:297、riscv:222、s390:217、powerpc:144、loongarch:647、wasm:232。它们的守卫宏分别是 __AVX2__、__ARM_NEON、__riscv_v、__VXE__、__POWER9_VECTOR__、__loongarch_sx、__wasm_simd128__。",
  src: "ggml/src/ggml-cpu/arch/x86/quants.c",
  mark: [0, 6, 13, 15, 20, 23, 30, 35, 40, 43],
  lineNo: 701,
  code: `void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
//>> 函数名与签名：七个架构文件里逐字相同（bx/by/bs/nrc 都在）
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
//>> x86 的前置条件：一次只算一行（nrc == 1）
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
//>> vx 是 Q4_0 权重块数组，vy 是 Q8_0 激活块数组
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
//>> #if defined(__AVX2__)：同一个函数体内部，再按编译期宏分一次
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps( GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
//>> bytes_from_nibbles_32：把 4 bit 解包成 32 个字节

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        qx = _mm256_sub_epi8( qx, off );
//>> 重新居中到 [-8, +7]：这是 Q4_0 的零点约定

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
//>> _mm256_fmadd_ps：一次做 8 个 float 的融合乘加
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);`,
  duration: 24000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 6 十六个 <span class="hl-a">arch/</span> 文件，各自的真实主题 */
{
  kicker: "L5-05 · ISA 内核",
  title: "十六个 <span class=\"hl-a\">arch/</span> 文件，各自的真实主题",
  sub: "七份点积内核 + 四份 repack 内核 + 五份特性探测；repack 只有 x86/arm/riscv/s390 写了。",
  caption: "路径省略公共前缀 ggml/src/ggml-cpu/。各文件的 vec_dot / gemv+gemm 条数为本课实测统计。",
  src: "ggml/src/ggml-cpu/arch/wasm/quants.c",
  mark: [0, 2, 13, 15],
  lineNo: 26,
  code: `#if defined(__wasm_simd128__)
//>> wasm 的守卫：__wasm_simd128__（128 位定长）
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
//>> B1..B8 宏在编译期展开出 256 项的查表常量 —— wasm 没有 NEON 的 vqtbl
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
//>> table_b2b_0：把 8 bit 展开成 8 字节的预计算表
static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
//>> table_b2b_1：反向（取反再展开）用的第二张表
#endif`,
  duration: 24000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 7 AMX：用 <span class="hl-e">tensor_traits</span> 接管整颗 MUL_MAT */
{
  kicker: "L5-05 · AMX",
  title: "AMX：用 <span class=\"hl-e\">tensor_traits</span> 接管整颗 MUL_MAT",
  sub: "不新增类型、不改进 traits 表：注册一个 buffer type，让权重在写入时就重排成 VNNI 格式。",
  caption: "mmq.cpp 的矩阵乘用 8 个 tile 寄存器（TMM0-TMM7）：_tile_loadd 装 A/B，_tile_dpbssd 做点积，_tile_stored 存回；每个线程先 ggml_tile_config_init() 配一次 tile 形状（2477-2479 行）。",
  src: "ggml/src/ggml-cpu/amx/amx.cpp",
  mark: [0, 5, 7, 13, 14, 16, 24, 25, 27],
  lineNo: 19,
  code: `#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
//>> 整份 AMX 支持都在这一个编译期条件里：__AMX_INT8__ 且 __AVX512VNNI__

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
//>> tensor_traits：ggml 的算子级钩子（L5-04 讲的 traits 机制）
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
//>> work_size：这次 MUL_MAT 需要多少 Op 工作区
        size = ggml_backend_amx_desired_wsize(op);
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
//>> 只拦 GGML_OP_MUL_MAT；其它算子返回 false，交回默认路径
            ggml_backend_amx_mul_mat(params, op);
//>> 真正的实现在 mmq.cpp：ggml_backend_amx_mul_mat
            return true;
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
//>> 一个静态实例：AMX 路径下所有张量共用同一套 traits
    return &traits;
}
}  // namespace ggml::cpu::amx`,
  duration: 22000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 8 SpacemiT：<span class="hl-f">编译期 × 运行期 × 类型</span> 的三级选择链 */
{
  kicker: "L5-05 · SpacemiT",
  title: "SpacemiT：<span class=\"hl-f\">编译期 × 运行期 × 类型</span> 的三级选择链",
  sub: "RISC-V 厂商把矩阵扩展（IME1/IME2）做成了自定义指令；选中不了就 abort，不静默降级。",
  caption: "本幕代码窗是链条的最后两级（315-329 行）。上一级（266-313 行，IME2 分支）见 source.md 的逐字引文。",
  src: "ggml/src/ggml-cpu/spacemit/ime.cpp",
  mark: [0, 2, 4, 8, 11, 13, 17, 19],
  lineNo: 315,
  code: `#if defined(RISCV64_SPACEMIT_IME1)
//>> 第一级（编译期）：只有定义了 RISCV64_SPACEMIT_IME1 这段才存在
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
//>> 第二级（运行期）：use_ime1 来自 /proc/cpuinfo 的核型号探测，且要求前面没人选中（!set_kernel_impl）
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
//>> RVV 负责把激活量化成 int8 —— IME 只做整数矩阵乘，前处理仍走 RVV
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
//>> 第三级（类型）：constexpr 判断权重块类型，只有 q4 家族交给 IME1
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
//>> gemm_kernel_i8i4：IME1 的整数 GEMM 实现
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
//>> 任何一级不匹配都走到这里
            GGML_ABORT("no kernel implementation found for the block type");
//>> GGML_ABORT —— 宁可崩，也不悄悄退回慢路径（否则性能问题会查不出来）
        }`,
  duration: 22000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 9 剩下 26 个厂商/第三方文件，各自的真实主题 */
{
  kicker: "L5-05 · 厂商层",
  title: "剩下 26 个厂商/第三方文件，各自的真实主题",
  sub: "amx 5 个、kleidiai 4 个、spacemit 15 个、llamafile 2 个 —— 四家四种接法。",
  caption: "路径省略公共前缀 ggml/src/ggml-cpu/。llamafile 是 Mozilla 的 tinyBLAS：一份小矩阵 SGEMM，按 A/B 类型 switch、再按 ISA 分派（见 source.md 引文）。",
  src: "ggml/src/ggml-cpu/iqp.h",
  mark: [0, 2, 9, 10, 13, 17, 20, 24],
  lineNo: 8,
  code: `// batched mul_mat path for the grid based IQ types: decode 8 src0 rows at a time into per thread scratch
//>> iqp.cpp/h 是「IQ 面板」路径：按 8 行权重解码成 int8 面板，再做整数 GEMM
// (block_iqp_x8, see iqp.cpp) and run an integer gemm over them against all src1 columns
//>> block_iqp_x8 就是那个面板：8 行的 int8 权重交错存放

#ifdef __cplusplus
extern "C" {
#endif

// whether cne1 rows of src1 are enough for the decode to pay for itself, per expert, for MUL_MAT_ID
bool ggml_cpu_iqp_mul_mat_id_min_batch(int64_t cne1);
//>> 按专家（expert）算的最小批量门槛

bool ggml_cpu_iqp_supports_mul_mat(const struct ggml_tensor * dst);
//>> 节点级判据：这个算子能不能走 IQP

// node level test only - per expert eligibility is decided with ggml_cpu_iqp_mul_mat_id_min_batch
bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst);

// per thread panel scratch bytes, padded
size_t ggml_cpu_iqp_scratch_size(const struct ggml_tensor * dst);
//>> 每线程面板 scratch 大小

// must be called after src1 has been converted to q8_K into params->wdata and the threads have synchronized on it
void ggml_compute_forward_mul_mat_iqp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
//>> 真正的实现入口：src1 必须先转成 q8_K 并同步好`,
  duration: 24000,
  build(root, tl) {
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
  }
},

/* ------------------------------------------------------ 10 收束：最后 7 个文件 + 这张表的四层 */
{
  kicker: "L5-05 · 收束",
  title: "收束：最后 7 个文件 + 这张表的四层",
  sub: "剩下的是三条垫层和两条旁路；hbm 与 iqp 都不碰 SIMD，它们改的是内存与 GEMM 组织。",
  caption: "下一课 L6-01 换到 GPU：同样的「一个算子多种内核」问题，在 CUDA 里由 stream 与 kernel 选择解决。",
  src: "ggml/src/ggml-cpu/hbm.cpp",
  mark: [0, 2, 5, 6, 8, 10, 18],
  lineNo: 40,
  code: `ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void) {
//>> hbm.cpp 与 SIMD 无关：它是一个 buffer type 的注册函数
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
//>> 和 AMX / KleidiAI / SpacemiT 一样，是一个 static 的 ggml_backend_buffer_type
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
//>> 分配走 hbw_posix_memalign（高带宽内存分配器，memkind）
                           /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
//>> 对齐复用 CPU 默认 buffer type 的对齐要求
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
//>> get_max_size / get_alloc_size 留空 = 用默认值（SIZE_MAX / ggml_nbytes）
                           /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
                           /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cpu_buffer_type_hbm;
}`,
  duration: 24000,
  build(root, tl) {
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
  }
},

];
