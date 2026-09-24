/* ==========================================================================
   L6-09 · Metal 后端：Metal 内核
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 23 个源文件、<span class="hl-a">8 个族</span>：内核层的全景 */
{
  kicker: "L6-09 · 全局",
  title: "23 个源文件、<span class=\"hl-a\">8 个族</span>：内核层的全景",
  sub: "每个 .metal 都从 #include \"common.h\" 开始；common.h 再把主机的参数头 ggml-metal-impl.h 拉进来。",
  caption: "上游：L6-08 讲主机端怎么选 pipeline；本课只讲内核本身。",
  src: "ggml/src/ggml-metal/kernels/common.h",
  mark: [2, 13, 21, 24],
  lineNo: 1,
  code: `#pragma once

#include "ggml-metal-impl.h"
//>> 主机端的 kargs 与 N_MM_* 常量都在这里，两边共享

#include <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
//>> FOR_UNROLL 把循环交给 clang 全展开，内核里到处在用

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32
//>> 一个 SIMD group = 32 个线程；后面所有 simd_sum / simd_max 都基于它`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const fams = [
      ['矩阵乘 matmul',       '2',  '4365', '245'],
      ['注意力与序列混合',    '4',  '3395', '567'],
      ['逐元素与形状',        '6',  '2337', '61'],
      ['量化与反量化',        '3',  '1477', '98'],
      ['归约 / 排序 / 三角',  '4',  '851',  '20'],
      ['归一化与 softmax',    '2',  '541',  '18'],
      ['位置编码 rope',       '1',  '333',  '8'],
      ['公共头 common.h',     '1',  '126',  '0']
    ];
    const t = U.table(['族', '文件', '行数', 'host_name'], fams, { monoCols: [1, 2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '23 个文件按"算什么"分成 8 个族。最大的一族是矩阵乘（2 个文件 4365 行）。',
      '<span class="v">mul_mm.metal + mul_mv.metal</span>：矩阵乘的两条路，也是本课的主角（第 2 到第 6 幕）。',
      '<span class="v">fa.metal</span> 一个文件就有 2476 行、559 个实例化 —— 头维度每变一次就要多一组实例。',
      '逐元素与形状族最杂：unary / binbcast / misc / conv / pool / upscale，写法最接近普通 GPU kernel。',
      '归约、排序、三角求解：都在线程组内做两级归约，靠 threadgroup 内存跨 SIMD group 汇总。',
      '归一化与 softmax 只有 541 行，却是最典型的 SIMD-group 归约范式（第 8 幕之外的源码小节）。',
      'rope 一个文件：把 YaRN 的数学直接搬进内核。',
      '<span class="k">合计 23 个文件 / 13425 行 / 1017 处 host_name 实例化。</span>'
    ];
    rows.forEach((r, i) => tl.at(700 + i * 2500, () => {
      rows.forEach(x => { x.className = ''; });
      r.className = 'on';
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">模板参数就是量化类型</span>：一份源码，二十多种 block_q */
{
  kicker: "L6-09 · 核心",
  title: "★ <span class=\"hl-a\">模板参数就是量化类型</span>：一份源码，二十多种 block_q",
  sub: "kernel_mul_mm 的模板参数里有 block_q（量化块类型）、nl（每 16 个权重跨几个块）和一个反量化函数指针。",
  caption: "第 13 行是条件编译：这条声明只在支持 simdgroup 矩阵乘的设备上生效；否则走文件下半部分的回退实现。",
  src: "ggml/src/ggml-metal/kernels/mul_mm.metal",
  mark: [0, 3, 6, 8, 10],
  lineNo: 12,
  code: `// each block_q contains 16*nl weights
//>> 这一行注释就是 nl 的含义：一个 block_q 装 16*nl 个权重
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
//>> block_q 是块类型，nl 是块内 16 元素的分段数
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
//>> T0 / T1 是 A、B 在设备内存里的元素类型；SA / SB 是线程组内存类型
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'block_q', m: 'typename block_q', b: '量化块类型：<br>block_q4_0 / block_q4_K / block_iq4_xs ...' },
      { c: 'b', t: 'nl', m: 'short nl', b: '一个块含 16*nl 个权重：<br>q4_0 -> 2（=32），Q4_K -> QK_NL=16（=256）' },
      { c: 'c', t: 'dequantize_func', m: 'void (*)(device const block_q *, short, thread SA_4x4 &)', b: '反量化函数指针：<br>dequantize_q4_0 / dequantize_q4_K ...' },
      { c: 'd', t: 'SA / SB', m: 'typename SA, typename SB', b: '线程组内存里的元素类型：<br>half / bfloat / float' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { w: '163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '模板参数一共 12 个。其中三个决定了"支持哪种量化"：',
      '<span class="v">block_q</span>：换一个块类型，就换一种量化格式。内核正文只按块读、按块反量化。',
      '<span class="v">nl</span>：告诉内核"16 个权重"在一个块里占第几段。<br>q4_0 是 2，K-quant 是 16 —— 见第 3 幕的实例化清单。',
      '<span class="v">dequantize_func</span>：把反量化做成函数指针参数。<br>于是内核正文完全不需要知道 q4_0 和 iq4_xs 有什么区别。',
      '非量化类型也走同一个槽：<span class="k">dequantize_f32</span> 只是把 float4x4 原样拷进寄存器（源码注释写着"this is not dequantizing"）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15000, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 3 实例化清单 = <span class="hl-c">量化支持矩阵</span> */
{
  kicker: "L6-09 · 实例化",
  title: "实例化清单 = <span class=\"hl-c\">量化支持矩阵</span>",
  sub: "typedef decltype(...) 先固定住与类型无关的参数，之后每一行 host_name 就是一个可被主机端取用的 pipeline。",
  caption: "回顾 L6-04：CUDA 把实例化写进 template-instances/ 的独立文件；Metal 直接写在同一份 .metal 的末尾。",
  src: "ggml/src/ggml-metal/kernels/mul_mm.metal",
  mark: [2, 5, 8, 11, 13, 19],
  lineNo: 849,
  code: `//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;
//>> 先给一组"默认"参数定型：这就是 mul_mm_t

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
//>> f32 走的就是 dequantize_f32 —— 同一个模板槽
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['f32',  '1',     'dequantize_f32',  '不反量化，只把 float4x4 拷进寄存器'],
      ['f16',  '1',     'dequantize_f16',  '同上，元素类型换成 half'],
      ['bf16', '1',     'dequantize_bf16', '被 GGML_METAL_HAS_BF16 包住'],
      ['q1_0', '8',     'dequantize_q1_0', '16*8 = 128 个权重一块'],
      ['q2_0', '4',     'dequantize_q2_0', '16*4 = 64 个权重一块'],
      ['q4_0', '2',     'dequantize_q4_0', '16*2 = 32 个权重一块'],
      ['q2_K', 'QK_NL', 'dequantize_q2_K', '16*16 = 256 个权重一块']
    ];
    const t = U.table(['src0 类型', 'nl', '反量化函数', '含义'], rows, { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const trs = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看被引用的这几行：一张"类型 -> nl -> 反量化函数"的对照表。',
      '<span class="v">f32 / f16 / bf16</span>：nl = 1，反量化函数只是把向量原样搬进寄存器。',
      '<span class="v">q1_0 / q2_0</span>：nl = 8 / 4，即一个块 128 / 64 个权重。',
      '<span class="v">q4_0</span>：nl = 2，一个块 32 个权重 —— 与 L1-04 讲的 Q4_0 块布局一致。',
      '<span class="v">q2_K</span>：nl = QK_NL = 16，一个块 256 个权重，即 QK_K。',
      '同一个 mul_mm.metal 里共有 <span class="k">23 种 block 类型</span>、<span class="k">111 处 host_name</span>（其中 51 处非 MoE）。',
      '主机端按 tensor 的 type 取名字，取到哪个 host_name 就用哪个 pipeline —— 名字是两边的接口。'
    ];
    trs.forEach((r, i) => tl.at(700 + i * 2900, () => {
      trs.forEach(x => { x.className = ''; });
      r.className = 'on';
      msg.innerHTML = texts[i];
    }));
    tl.at(21500, () => {
      trs.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[6];
    });
  }
},

/* ------------------------------------------------------ 4 mul_mm 的 tile：<span class="hl-b">64 x 32</span> 的输出块 */
{
  kicker: "L6-09 · 矩阵乘",
  title: "mul_mm 的 tile：<span class=\"hl-b\">64 x 32</span> 的输出块",
  sub: "每个线程组算输出矩阵的一块；A 先反量化进线程组内存，再用 simdgroup 矩阵乘累加。",
  caption: "这是不支持 simdgroup 矩阵乘时的回退实现；上半部分（tensor 路径）的 tile 由 N_MM_* 宏给出。",
  src: "ggml/src/ggml-metal/kernels/mul_mm.metal",
  mark: [1, 2, 4, 6, 17, 19, 21],
  lineNo: 160,
  code: `
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
//>> NR0：沿输出行方向，一个线程组算 64 行
    constexpr int NR1 = 32;
//>> NR1：沿输出列方向，一个线程组算 32 列

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
//>> 源码注释直接把 tile 形状写出来了
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
//>> 边界块收窄：行方向取剩余的行数
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;
`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:4px">一个线程组负责的输出块</div>';
    const grid = U.el('div', { style: 'display:flex;flex-wrap:wrap;gap:2px;width:196px' });
    for (let i = 0; i < 8; i++) {
      const c = U.el('div', { style: 'width:22px;height:15px;border:1px solid var(--border);border-radius:2px;background:rgba(88,166,255,.20)' });
      grid.appendChild(c);
    }
    left.appendChild(grid);
    left.appendChild(U.el('div', { class: 'cb', style: 'font-size:9.5px;color:var(--dim)', text: '8 x 8 = 64 格，示意 NR0=64 / NR1=32 的 tile 形状' }));

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">两阶段流水</div>' +
      '<div class="cb"><b>PHASE 1</b>：把 A 的一块反量化进线程组内存 sa（第 52 行）。<br>' +
      '<b>PHASE 2</b>：simdgroup 矩阵乘，把结果累加进 cT。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">K 方向分块</div>' +
      '<div class="cb">外层循环按 NK=32 走 K 维，每一轮都要一次 ' +
      '<span class="cm" style="margin:0">threadgroup_barrier</span> —— ' +
      '这就是 tile 方案的代价。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看 tile 的形状：NR0 = 64 行、NR1 = 32 列、NK = 32。',
      '<span class="v">r0 = tgpig.y * NR0</span>，<span class="v">r1 = tgpig.x * NR1</span>：<br>线程组在网格里的坐标直接乘出它在输出矩阵里的位置。',
      'A 的反量化结果放进线程组内存 <span class="v">sa</span>，B 的缓冲是 <span class="v">sb</span>（偏移 4096 字节）。',
      '关键问题：<span class="k">列方向固定 32 列</span>。<br>如果输出只有 1 列（一次只算一个 token），32 列里 31 列是空转的。',
      '第 175 到 177 行的收窄只解决"边界块"，解决不了"本来就只有 1 列"。<br>这就是下一幕 mul_mv 存在的理由。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [4, 6]); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [12, 13]); });
    tl.at(7000, () => { msg.innerHTML = texts[2]; U.markLines(document, [1, 2]); });
    tl.at(10500, () => { msg.innerHTML = texts[3]; U.markLines(document, [4, 6]); });
    tl.at(14500, () => { msg.innerHTML = texts[4]; U.markLines(document, [17, 19, 21]); });
  }
},

/* ------------------------------------------------------ 5 mul_mv 的 <span class="hl-e">per-row</span> 结构：一行 src1 对 NR0 行 src0 */
{
  kicker: "L6-09 · 矩阵乘",
  title: "mul_mv 的 <span class=\"hl-e\">per-row</span> 结构：一行 src1 对 NR0 行 src0",
  sub: "每个线程组只认 src1 的一行；这一行被缓存进寄存器，在 NR0 行 src0 之间反复复用。",
  caption: "NR0 与线程组里的 SIMD group 数由主机端给出（N_R0_* / N_SG_* 宏，见 L6-08）。",
  src: "ggml/src/ggml-metal/kernels/mul_mv.metal",
  mark: [5, 7, 9, 23, 25, 28],
  lineNo: 228,
  code: `    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
  //const int r0 =  tgpig.x*NR0;
    const int r1 =  tgpig.y;
//>> r1 = tgpig.y：src1 的一行 = 线程组网格的一个 y 坐标
    const int im =  tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q_type * x = (device const block_q_type *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[NR0];
//>> ax[NR0]：一次抓下 NR0 行 src0 的块指针
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
//>> 每一行 src0 的起始地址按 nb01 步进
    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia');
    const a = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--e)' });
    a.innerHTML = '<div class="ct" style="color:var(--e)">src0：NR0 行量化权重</div>' +
      '<div class="cb">每行 = nb 个 block_q<br>' +
      '<span class="cm" style="margin:0">ax[row] = src0 + (r0+row)*nb01</span></div>';
    const b = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
    b.innerHTML = '<div class="ct" style="color:var(--b)">src1：1 行输入向量</div>' +
      '<div class="cb">缓存在寄存器 <span class="cm" style="margin:0">float yl[16]</span><br>' +
      '被 NR0 行 src0 复用</div>';
    dia.appendChild(a); dia.appendChild(U.arrow('x')); dia.appendChild(b);

    const msg = wrap.querySelector('#msg');
    const texts = [
      'mul_mv 的网格是"行 x 行"：<span class="v">r0</span> 是 src0 的行，<span class="v">r1</span> 是 src1 的行。',
      '<span class="v">r0 = (tgpig.x*NSG + sgitg)*NR0</span>：<br>一个线程组里的每个 SIMD group 再分 NR0 行 src0。',
      '<span class="v">r1 = tgpig.y</span>：src1 的一行 = 一个 y 坐标。<br><span class="k">线程组粒度就是"一行 src1"</span> —— 没有列方向的空转。',
      '<span class="v">ax[NR0]</span>：把 NR0 行 src0 的块指针一次性算好，循环里只做 <span class="m">ax[row] + ib</span>。',
      '内层循环（第 265 到 291 行，见下面源码小节的第二段）：每次取 16 个 float 存进 ' +
      '<span class="m">yl[16]</span>，再对 NR0 行各做一次 <span class="m">block_q_n_dot_y</span> —— ' +
      '<span class="k">一份 src1 数据，NR0 次复用</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [7, 9]); });
    tl.at(3700, () => { msg.innerHTML = texts[1]; U.markLines(document, [7]); });
    tl.at(7400, () => { msg.innerHTML = texts[2]; U.markLines(document, [9]); });
    tl.at(11000, () => { msg.innerHTML = texts[3]; U.markLines(document, [23, 25, 26, 28]); });
    tl.at(15000, () => { msg.innerHTML = texts[4]; U.markLines(document, [7, 9]); });
  }
},

/* ------------------------------------------------------ 6 ★ <span class="hl-a">src1 有几行</span>决定走哪条路 */
{
  kicker: "L6-09 · 验收点",
  title: "★ <span class=\"hl-a\">src1 有几行</span>决定走哪条路",
  sub: "mul_mv 的 ext 变体把 r1ptg 行 src1 绑进一个线程组；实例化里 r1ptg 只取 2 到 5 —— 它就是为\"少数几行\"设计的。",
  caption: "主机端按 ne11 选 r1ptg、并决定走 ext 还是 mul_mm —— 那部分在 ggml-metal-ops.cpp，见 L6-08。",
  src: "ggml/src/ggml-metal/kernels/mul_mv.metal",
  mark: [0, 12, 21, 22, 34, 37, 41],
  lineNo: 602,
  code: `template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread
//>> chpt = 4：每个线程一次处理 4 个 float4 块

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
//>> i11 = tgpig.y * r1ptg：线程组的 y 坐标以 r1ptg 为步长
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
//>> r1ptg 行 src1 的指针一次抓进寄存器数组
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['ne11 == 1',        'mul_mv',     '线程组只算 1 列，没有空转（第 237 行 r1 = tgpig.y）'],
      ['ne11 = 2 到 5',    'mul_mv_ext', '一个线程组吃下 r1ptg 行 src1（第 623 行），r1ptg 只实例化了 2/3/4/5'],
      ['ne11 大（批量）',  'mul_mm',     'tile 是 64 x 32，反量化后的 A 块被 32 列复用（第 164、165 行）']
    ];
    const t = U.table(['src1 的行数（ne11）', '更优的内核', '源码依据'], rows, { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const trs = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '把这一课的问题压成一张表：<span class="k">形状决定内核</span>。',
      '<span class="v">ne11 == 1</span>（逐 token 解码）：mul_mv 每个线程组只服务 src1 的一行，没有一列是白算的。',
      '<span class="v">ne11 只有几行</span>：走 ext 变体，一个线程组同时算 r1ptg 行 —— 把"少数几行"也凑满一个线程组。',
      '<span class="v">ne11 很大</span>（prompt 批量）：mul_mm 更优 —— 它把 A 反量化进线程组内存后，整块 tile 一起用。',
      '注意这两条路的<span class="k">取舍是对称的</span>：mul_mv 用"每行一次点积"换掉列方向空转；mul_mm 用"固定 32 列"换 A 块的复用。'
    ];
    trs.forEach((r, i) => tl.at(700 + i * 3300, () => {
      trs.forEach(x => { x.className = ''; });
      r.className = 'on';
      msg.innerHTML = texts[i];
    }));
    tl.at(15500, () => {
      trs.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[4];
    });
    tl.at(18500, () => { msg.innerHTML = texts[4] + '<br>验收点：能说出 mul_mv 在 <span class="v">src1 只有一行/少数几行</span>时更优，' +
      '因为 mul_mm 的 tile 列方向是固定的 32 列。'; });
  }
},

/* ------------------------------------------------------ 7 flash attention：<span class="hl-d">DK / DV</span> 也是模板参数 */
{
  kicker: "L6-09 · 注意力",
  title: "flash attention：<span class=\"hl-d\">DK / DV</span> 也是模板参数",
  sub: "kernel_flash_attn_ext 的头维度是编译期常量；host_name 里 dk128_dv128 这样的后缀就是一对 (DK, DV)。",
  caption: "fa.metal 共 559 处 host_name、16 种 (DK, DV) 组合 —— 头维度组合爆炸是模板方案的主要代价。",
  src: "ggml/src/ggml-metal/kernels/fa.metal",
  mark: [0, 20, 27, 29, 30, 31, 33],
  lineNo: 199,
  code: `// ref: https://arxiv.org/pdf/2307.08691.pdf
//>> 源码直接给出 flash-attention 的原始论文链接
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
//>> kd4x4_t / nl_k / deq_k：K 在设备内存里的块类型、分段数与反量化函数
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
//>> DK / DV：K 与 V 的头维度，都是 short 模板参数
    short DV,         // V head size
    short Q,          // queries per threadgroup
    short C,          // cache items per threadgroup
//>> C：每个线程组缓存多少条 KV
    short NSG>        // number of simd groups
void kernel_flash_attn_ext_impl(`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'DK / DV', m: 'short DK, short DV', b: 'K 与 V 的头维度。<br>host_name 后缀 dk128_dv128 就是它' },
      { c: 'b', t: 'Q / C', m: 'short Q, short C', b: 'Q：每个线程组处理几个 query<br>C：缓存几条 KV' },
      { c: 'c', t: 'NSG', m: 'short NSG', b: '线程组里的 SIMD group 数，<br>实现里只实例化了 4 和 8' },
      { c: 'd', t: 'deq_k / deq_v', m: 'void (*)(device const kd4x4_t *, short, thread k4x4_t &)', b: 'K / V 的反量化函数指针 ——<br>与 mul_mm 完全同一个套路' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { w: '163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'flash attention 的内核有 28 个模板参数，比 mul_mm 还长。',
      '<span class="v">DK / DV</span>：头维度进模板，是因为线程组内存的布局要按它算常量。<br>代价：每换一个头维度就多一组实例。',
      '<span class="v">Q / C / NSG</span>：分块策略也进模板。<br>第 894 行的源码注释说得很直白：<span class="m">this is quite ugly ... for now keep them as template</span>。',
      '<span class="v">deq_k / deq_v</span>：KV 量化的支持方式与 mul_mm 一致 —— 换函数指针，不换内核正文。',
      'KV 量化只覆盖 <span class="k">q4_0 / q4_1 / q5_0 / q5_1 / q8_0</span> 五种：先由 <span class="m">kernel_flash_attn_ext_kv_f16</span> 反量化成 F16，再跑 F16 主内核。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15000, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 8 同一个 <span class="hl-c">block_q</span>：CPU 与 Metal 共用一份定义 */
{
  kicker: "L6-09 · 共享定义",
  title: "同一个 <span class=\"hl-c\">block_q</span>：CPU 与 Metal 共用一份定义",
  sub: "dequantize.h 用 GGML_COMMON_DECL_METAL 打开 ggml-common.h 的 Metal 分支 —— 块结构体不是 Metal 特有的。",
  caption: "回顾 L1-04：块的字节布局（scale + 低位宽权重）在那里逐字展开；本课只讲 Metal 怎么用它。",
  src: "ggml/src/ggml-metal/kernels/dequantize.h",
  mark: [4, 6, 10, 14],
  lineNo: 1,
  code: `#pragma once

#include "common.h"

#define GGML_COMMON_DECL_METAL
//>> 打开 ggml-common.h 的 Metal 分支
#define GGML_COMMON_IMPL_METAL
#if defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#else
#include "ggml-common.h"
//>> block_q4_0 / block_q4_K 这些结构体就是从这一行来的
#endif

#define QK_NL 16 // shared by mul_mm and get_rows_q instantiations
//>> QK_NL = 16，被 mul_mm 与 get_rows_q 的实例化共用`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip">ggml-common.h</span><span class="arrow">-></span>
        <span class="chip a">GGML_COMMON_DECL_METAL</span><span class="arrow">-></span>
        <span class="chip b">block_q4_0 / block_q4_K</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'CPU 侧', b: 'ggml-common.h 的 C / CPP 分支<br>同一个 block_q4_0' },
      { c: 'b', t: 'CUDA 侧', b: 'GGML_COMMON_DECL_CUDA 分支<br>同一个 block_q4_0' },
      { c: 'c', t: 'Metal 侧', b: 'GGML_COMMON_DECL_METAL 分支<br>还是同一个 block_q4_0' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { w: '214px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '量化块的<b>内存布局</b>只有一份定义，各后端只是各取所需地把它 include 进来。',
      '<span class="v">#define GGML_COMMON_DECL_METAL</span> 选分支；<span class="v">GGML_COMMON_IMPL_METAL</span> 还要实现部分。',
      '所以 <span class="k">Metal 不需要重新定义 block_q4_0</span>：它和 CPU 看到的是同一段字节。',
      '第 7 到 11 行是双路径：嵌入库时用 <span class="m">__embed_ggml-common.h__</span>，否则直接 include。',
      '<span class="v">QK_NL = 16</span> 也定义在这里，注释写明它被 mul_mm 与 get_rows_q 的实例化共用 —— 第 2 幕那个 nl 的取值就来自它。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13500, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[3];
    });
    tl.at(15500, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L6-09 · 收束",
  title: "把这一课压成一张表",
  sub: "内核在哪、怎么覆盖量化类型、怎么用 SIMD group、什么时候换另一条路 —— 四个问题。",
  caption: "下一课 L7-01 换一个后端：CANN 怎么把 ggml op 映射到 Ascend NPU 的 ACL 算子。",
  src: "ggml/src/ggml-metal/kernels/mul_mm.metal",
  mark: [0, 2, 6, 7, 10, 15],
  lineNo: 44,
  code: `    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
//>> NRB：沿输出列方向的 tile 宽度，由三个宏相乘得到
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
//>> NRA：沿输出行方向的 tile 高度

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);
//>> A 的反量化结果只占线程组内存 —— B 直接从设备内存读

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案', '源码位置'],
      [['内核在哪', 'kernels/ 下 23 个文件、8 个族、13425 行', 'common.h:1-23'],
       ['怎么覆盖量化', 'C++ 模板 + host 侧 host_name 实例化（23 种 block_q）', 'mul_mm.metal:851-880'],
       ['怎么用 SIMD group', 'N_SIMDWIDTH = 32，simd_sum / simd_max + threadgroup 内存两级归约', 'common.h:23 / norm.metal:42'],
       ['什么时候换路', 'ne11 小走 mul_mv / ext，ne11 大走 mul_mm（tile 固定 32 列）', 'mul_mv.metal:237,623 / mul_mm.metal:165'],
       ['块定义从哪来', 'GGML_COMMON_DECL_METAL 打开 ggml-common.h，与 CPU 共用', 'dequantize.h:5-10']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const trs = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一次只解码一个 token（src1 只有 1 行）时，该走 <span class="mono">mul_mv</span> 还是 ' +
      '<span class="mono">mul_mm</span>？为什么？',
      '走 <span class="mono">mul_mv</span>。<br>' +
      '<b>1.</b> <span class="mono">mul_mm</span> 的线程组 tile 是 64 行 x 32 列' +
      '（<span class="mono">mul_mm.metal:164-165</span>，第 175 行的注释也写着 64x32）；' +
      '输出只有 1 列时，32 列里 31 列白算。<br>' +
      '<b>2.</b> <span class="mono">mul_mv</span> 的网格是"src0 的行 x src1 的行"：' +
      '<span class="mono">r1 = tgpig.y</span>（<span class="mono">mul_mv.metal:237</span>），' +
      '一个线程组只服务 src1 的一行，列方向没有空转。<br>' +
      '<b>3.</b> 它把这一行的 16 个 float 缓存在寄存器 ' +
      '<span class="mono">float yl[16]</span>（<span class="mono">mul_mv.metal:265</span>）里，' +
      '在 NR0 行 src0 之间复用（<span class="mono">mul_mv.metal:285-287</span>）。<br>' +
      '<b>反过来</b>：src1 有几十上百行（prompt 批量）时 <span class="mono">mul_mm</span> 更优 —— ' +
      'A 的反量化结果进了线程组内存（<span class="mono">mul_mm.metal:52</span>），' +
      '整块 tile 一起复用。<br>' +
      '中间地带（src1 只有 2 到 5 行）走 <span class="mono">mul_mv_ext</span>：' +
      '<span class="mono">i11 = tgpig.y * r1ptg</span>（<span class="mono">mul_mv.metal:623</span>），' +
      '一个线程组同时算 r1ptg 行。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '五个问题、五个答案。',
      '<span class="k">内核在哪</span>：23 个文件 8 个族；公共头 common.h 把主机参数头拉进来。',
      '<span class="k">怎么覆盖量化</span>：模板参数 block_q / nl / dequantize_func，加 host_name 实例化 —— 与 CUDA 的 template-instances 是两种解法。',
      '<span class="k">怎么用 SIMD group</span>：32 线程一组，先 simd_sum 组内归约，再用 threadgroup 内存跨组汇总。',
      '<span class="k">什么时候换路</span>：src1 的行数决定内核 —— 这是本课的验收点。',
      '一句话：<span class="v">Metal 用一份 .metal 加一张实例化清单，换掉 CUDA 的整个 template-instances 目录</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    trs.forEach((r, i) => tl.at(2400 + i * 3000, () => {
      trs.forEach(x => { x.className = ''; });
      r.className = 'on';
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(19000, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

];
