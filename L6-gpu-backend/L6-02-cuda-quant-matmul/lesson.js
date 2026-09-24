/* ==========================================================================
   L6-02 · ★ CUDA 量化矩阵乘：mmq / mmvq / mmvf
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一个 <span class="hl-a">MUL_MAT</span>，五条内核路线 */
{
  kicker: "L6-02 · 全局",
  title: "一个 <span class=\"hl-a\">MUL_MAT</span>，五条内核路线",
  sub: "ggml_cuda_mul_mat() 里没有 switch(op)，只有一串 should_use_* 判据，按顺序短路。",
  caption: "验收点的答案就在这张表里：mmvq 在第 4 条、mmq 在第 5 条，谁先命中谁上。分派点 ggml_cuda_mul_mat() 属于 L6-01 的后端骨架，本课只拆它的 MUL_MAT 分支。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [0, 3, 24, 25, 28, 29, 32],
  lineNo: 1844,
  code: `    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', w: '163px', t: 'mmvf', m: 'ggml_cuda_mul_mat_vec_f', b: 'float 权重 × 向量<br>src0 极瘦时才划算' },
      { c: 'a', w: '163px', t: 'mmvq', m: 'ggml_cuda_mul_mat_vec_q', b: '量化权重 × 向量<br>每线程读一整行权重' },
      { c: 'c', w: '163px', t: 'mmq', m: 'ggml_cuda_mul_mat_q', b: '量化权重 × 小矩阵<br>权重搬进 shared tile' },
      { c: 'f', w: '163px', t: 'cuBLAS', m: 'ggml_cuda_mul_mat_cublas', b: '兜底：反量化 + GEMM<br>本课不展开' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const t = U.table(
      ['序', '判据（ggml-cuda.cu 行号）', '调用的内核'],
      [['1', 'ggml_cuda_should_use_mmvf(...)  :1844', 'ggml_cuda_mul_mat_vec_f  :1847'],
       ['2', 'ne01 == 1 且 ne11 &gt; 8  :1851', 'mul_mat_vec_f（转置后复用）  :1861'],
       ['3', 'ggml_cuda_should_use_mmf(...)  :1864', 'ggml_cuda_mul_mat_f  :1865'],
       ['4', 'ggml_cuda_should_use_mmvq(...)  :1868', 'ggml_cuda_mul_mat_vec_q  :1869'],
       ['5', 'ggml_cuda_should_use_mmq(...)  :1872', 'ggml_cuda_mul_mat_q  :1873'],
       ['6', '以上全不满足  :1876', 'ggml_cuda_mul_mat_cublas  :1876']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const trs = t.body.querySelectorAll('tr');
    const texts = [
      'MUL_MAT 落到 CUDA 上，并不是无条件走同一个内核。',
      '六个判据按 <span class="k">固定顺序</span> 求值，<span class="v">第一个为真的赢</span>，后面的根本不会被问到。',
      '所以"选哪个内核"= "哪条判据先为真"。本课把第 4、5 条的判据函数逐字拆开。',
      '两台内核都做量化点积，却是两份代码：<span class="k">同一个数学，两种形状假设</span>。',
      '关键变量是 <span class="v">ne11</span>：src1 的列数，即一次乘几个 token。',
      '一句话：<span class="k">矩阵有多瘦，决定用哪套实现</span>。下一幕逐个看判据。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    [0, 1, 2, 3, 4, 5].forEach(i => tl.at(2600 + i * 2300, () => {
      trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      els.forEach((e, k) => { e.style.opacity = (k === [0, 3, 2, 1, 2, 3][i]) ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(17200, () => { trs.forEach(x => { x.className = ''; }); els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 2 mmvq：<span class="hl-a">ne11 不超过 8</span> 就走向量内核 */
{
  kicker: "L6-02 · 判据 1/3",
  title: "mmvq：<span class=\"hl-a\">ne11 不超过 8</span> 就走向量内核",
  sub: "这个 8 写在头文件的宏里；部分架构和量化类型还会把它压得更小。",
  caption: "MMVQ_MAX_BATCH_SIZE 定义在 mmvq.cuh:3。",
  src: "ggml/src/ggml-cuda/mmvq.cu",
  mark: [0, 1, 4, 7, 10, 14],
  lineNo: 318,
  code: `bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
//>> 源码注释给出理由：k-quant 解码贵，而 mvq 每个 token 都要重解码一遍
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
//>> 默认落地：ne11 <= MMVQ_MAX_BATCH_SIZE，即 8（mmvq.cuh:3）`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', w: '218px', t: '第 0 道门：必须是量化类型',
        m: 'ggml_is_quantized(type)',
        b: 'F32 / F16 / BF16 直接返回 false，<br>在本课里走 mmvf 或 cuBLAS。' },
      { c: 'b', w: '218px', t: '默认阈值：ne11 不超过 8',
        m: 'MMVQ_MAX_BATCH_SIZE = 8',
        b: 'ne11 是 src1 的列数，也就是<br>一次前向要算几个 token。' },
      { c: 'c', w: '218px', t: '按架构和类型再收紧',
        m: 'Ada: Q2_K 到 4 / Q3_K 到 6',
        b: '函数体里还有 Blackwell、Orin、<br>Volta、CDNA1/2 各自的调优阈值。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看 <span class="v">ne11</span> 这个量：它是 <span class="k">src1 的列数</span>，'
        + '在 MUL_MAT 里就是 batch / token 数。',
      '<span class="k">非量化类型</span> 在第 3 行就被挡掉 —— 向量内核只服务量化权重。',
      '<span class="v">默认阈值 8</span>：超过 8 个 token 就不再"一列一列地算"，'
        + '让位给能复用权重的 tile 内核。',
      '架构分支把阈值压得更小：<span class="v">Q2_K 到 4、Q3_K 到 6</span>（Ada）。'
        + '<br>理由写在源码注释里：<span class="k">k-quant 解码贵，而 mvq 每个 token 都重解码一遍</span>。',
      '收束：<span class="k">量化权重 × 至多 8 个 token</span> 是 mmvq 的适用形状。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1, 2, 3, 4]); });
    tl.at(3400, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [0, 1]); });
    tl.at(6800, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [6, 7, 8, 9, 10, 11, 12, 13]); });
    tl.at(10200, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [5, 6, 7, 8, 9, 10, 11, 12]); });
    tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; U.markLines(document, [6, 7, 8, 9, 10, 11, 12, 13]); });
  }
},

/* ------------------------------------------------------ 3 ★ mmq：有 Tensor Core 就一律走，<span class="hl-a">否则 ne11 小于 64</span> */
{
  kicker: "L6-02 · 判据 2/3",
  title: "★ mmq：有 Tensor Core 就一律走，<span class=\"hl-a\">否则 ne11 小于 64</span>",
  sub: "mmq 的门槛比 mmvq 宽得多，因为它把权重搬进 shared memory 之后可以被多个 token 复用。",
  caption: "MMQ_DP4A_MAX_BATCH_SIZE 定义在 mmq.cuh:8，值是 64。",
  src: "ggml/src/ggml-cuda/mmq.cu",
  mark: [0, 5, 10, 12, 18, 25, 27],
  lineNo: 310,
  code: `    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
//>> 硬性前提：每个 block 至少要 48 KiB shared memory，否则直接让给 BLAS
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
//>> Turing 及以后（cc >= 7.5）有 mma 指令，mmq 无条件返回 true
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
//>> 否则：没有 fp16 tensor core 的卡一律 mmq；有的卡只在 ne11 < 64 时用 mmq
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
      <div class="col" id="bars" style="gap:6px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'e', w: '163px', t: '门 1 · 类型白名单',
        m: 'mmq_supported', b: 'Q4_0/Q8_0/Q2_K~Q6_K/<br>IQ*/MXFP4/NVFP4 才算支持' },
      { c: 'f', w: '163px', t: '门 2 · 48 KiB SRAM',
        m: 'smpbo >= 48 * 1024', b: 'tile 要放进每 block 的<br>shared memory；装不下就退出' },
      { c: 'c', w: '163px', t: '门 3 · Turing 及以上',
        m: 'turing_mma_available(cc)', b: '有 tensor core 指令，<br>直接 return true（不看 ne11）' },
      { c: 'a', w: '163px', t: '门 4 · 老卡看 batch',
        m: 'ne11 &lt; 64', b: '没有 fp16 mma 的卡一律 mmq；<br>有的卡只在 batch 小于 64 时用' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const barsHost = wrap.querySelector('#bars');
    barsHost.innerHTML = '<div class="cm" style="margin-bottom:2px">batch（ne11）轴上的胜负</div>';
    const segs = [
      { l: 'ne11 1..8 → mmvq', w: '12%', c: 'a' },
      { l: 'ne11 9..63 → mmq（Turing 及以上）', w: '48%', c: 'c' },
      { l: 'ne11 >= 64 → mmq（有 mma）或 cuBLAS', w: '40%', c: 'f' }
    ];
    const bars = segs.map(s => { const b = U.bar(s.l, s.c); barsHost.appendChild(b.el); return b; });
    bars.forEach(b => { b.fill.style.width = '0%'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      'mmq 的判据比 mmvq 长：<span class="k">四道门</span>，任何一道不过就返回 false。',
      '门 1 是 <span class="v">类型白名单</span>：不在表里的量化类型不会走 mmq。',
      '门 2 是 <span class="k">资源门</span>：tile 常驻 shared memory，至少要 48 KiB。',
      '门 3 最干脆：<span class="v">Turing 及以后直接 true</span>，完全不看 batch。',
      '门 4 才是 batch 判据：<span class="v">ne11 小于 64</span>，这就是 MMQ_DP4A_MAX_BATCH_SIZE。',
      '注意顺序：<span class="k">mmvq 在 mmq 之前</span>被问，所以 1..8 个 token 仍然归 mmvq。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(2900 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      bars.forEach((b, k) => { b.fill.style.width = k === i ? segs[k].w : '0%'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16500, () => { els.forEach(e => e.style.opacity = '1'); bars.forEach((b, k) => { b.fill.style.width = segs[k].w; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 4 mmvf：float 路径先看<span class="hl-b">内存对齐</span>，再看 batch */
{
  kicker: "L6-02 · 判据 3/3",
  title: "mmvf：float 路径先看<span class=\"hl-b\">内存对齐</span>，再看 batch",
  sub: "三条对齐前提任意一条不满足就崩溃或走不通，所以它们排在 batch 阈值前面。",
  caption: "MMVF_MAX_BATCH_SIZE 定义在 mmvf.cuh:3，值与 MMVQ_MAX_BATCH_SIZE 相同。",
  src: "ggml/src/ggml-cuda/mmvf.cu",
  mark: [0, 1, 6, 10, 13, 21, 22],
  lineNo: 792,
  code: `bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
//>> 源码注释：指针没按 half2 / nv_bfloat162 / float2 对齐会直接崩
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'e', w: '163px', t: '前提 1 · ne[0] 为偶数',
        m: 'src0_ne[0] % 2 == 0', b: '内核一次吃两个元素<br>（half2 / float2）' },
      { c: 'e', w: '163px', t: '前提 2 · 最内层连续',
        m: 'src0_nb[0] == type_size', b: '第 0 维没有 padding，<br>可以按类型宽度直接读' },
      { c: 'e', w: '163px', t: '前提 3 · 高维 2 元素对齐',
        m: 'nb[i] % (2*ts) == 0', b: '否则 half2 指针未对齐，<br>源码注释说会崩' },
      { c: 'a', w: '163px', t: '然后才看 batch',
        m: 'ne11 &lt;= 3（Ampere F32）', b: 'Turing 上到 4，无 fp32 mma<br>的 AMD 卡上到 8；上限 8' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'mmvf 服务的是 <span class="k">非量化</span> 权重（F32 / F16 / BF16），'
        + '所以判据里没有"类型白名单"，只有对齐。',
      '<span class="v">ne[0] 必须是偶数</span>：内核按 half2 / float2 成对读，奇数长度无法覆盖。',
      '<span class="v">nb[0] 必须等于 type_size</span>：最内层要连续，'
        + '否则向量化的指针推进不成立。',
      '更高维还要 <span class="v">2 倍类型宽度对齐</span>。源码注释直接写明：'
        + '<span class="k">不对齐会 crash</span>。',
      '三条都过了才轮到 batch：<span class="v">Ampere 上 F32 只到 3</span>，'
        + 'Turing 到 4，上限 8。',
      '对比 mmvq 的 8：<span class="k">float 向量内核比量化向量内核更挑形状</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(2900 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 同一份点积源码，<span class="hl-c">两套展开宽度</span> */
{
  kicker: "L6-02 · 量化点积",
  title: "同一份点积源码，<span class=\"hl-c\">两套展开宽度</span>",
  sub: "vecdotq.cuh 给每个量化类型定义两个 VDR：MMVQ 用窄的，MMQ 用宽的。",
  caption: "VDR 是每个线程一次点积处理的块数；它是唯一把 mmvq 与 mmq 写在同一个文件里的地方。",
  src: "ggml/src/ggml-cuda/vecdotq.cuh",
  mark: [0, 8, 10, 13, 20, 25, 32],
  lineNo: 107,
  code: `// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q1_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism
#define VDR_Q1_0_Q8_1_MMQ  4  // Q1_0 has 128 bits (4 ints) per block

#define VDR_Q2_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism
#define VDR_Q2_0_Q8_1_MMQ  2  // Q2_0 group 64: 128 bits (4 ints) per block, 2 32-element chunks

#define VDR_Q4_0_Q8_1_MMVQ 2
//>> MMVQ：每个线程一次只吃 2 块 Q4_0
#define VDR_Q4_0_Q8_1_MMQ  4
//>> MMQ：同一个类型，每个线程一次吃 4 块

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
//>> 模板参数 vdr 让同一份展开代码被实例化成两个宽度
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['量化类型', 'VDR（mmvq 用）', 'VDR（mmq 用）', '源码行'],
      [['Q1_0', '1', '4', ':109-110'],
       ['Q2_0', '1', '2', ':112-113'],
       ['Q4_0', '2', '4', ':115-116'],
       ['Q8_0', '2', '8', ':243-244'],
       ['Q2_K', '1', '4', ':363-364'],
       ['Q3_K', '1', '2', ':446-447'],
       ['Q4_K', '2', '8', ':504-505'],
       ['Q6_K', '1', '8', ':623-624']],
      { monoCols: [0, 1, 2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const trs = t.body.querySelectorAll('tr');
    const texts = [
      '同一个 <span class="m">vec_dot_&lt;类型&gt;_q8_1</span>，要同时服务向量内核和 tile 内核。',
      '<span class="v">MMVQ 列的 VDR 都比 MMQ 列小</span>：向量内核的并行度来自'
        + '"更多线程各算一点"，不是"每个线程多算几块"。',
      'tile 内核相反：数据已经在 shared memory 里，<span class="k">让每个线程多算几块</span>'
        + '才能摊薄指令开销。',
      'Q6_K 最极端：<span class="v">1 对 8</span>，相差 8 倍。',
      '实现上只是一个模板参数：<span class="m">vec_dot_q4_0_q8_1_impl&lt;vdr&gt;</span>，'
        + '循环 <span class="m">1..vdr</span> 次 <span class="m">ggml_cuda_dp4a</span>。',
      '所以"两套实现"不是两份代码，而是<span class="k">同一份代码的两个编译期宽度</span>；'
        + '宽度能取多少，由 L1-04 讲的块字节布局决定。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    [0, 1, 2, 3, 4, 5, 6, 7].forEach(i => tl.at(2400 + i * 1750, () => {
      trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(16600, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 tile 点积把 <span class="hl-d">mma</span> 当积木：16x8 乘 8x8 */
{
  kicker: "L6-02 · tile 点积",
  title: "tile 点积把 <span class=\"hl-d\">mma</span> 当积木：16x8 乘 8x8",
  sub: "mma.cuh 把 PTX 的 mma 指令包装成 tile 类型；mmq 的点积直接用它拼出一次 tile 乘。",
  caption: "tile<> / load_ldmatrix / load_generic / mma 全部来自 mma.cuh（见本课 source.md 第二节）。",
  src: "ggml/src/ggml-cuda/mmq-vec-dot.cuh",
  mark: [1, 3, 5, 10, 20, 21, 31, 54],
  lineNo: 201,
  code: `#else
    typedef tile<16, 8, int> tile_A;
//>> A 是 16x8 的 x 分片（i 方向 16 行）
    typedef tile< 8, 8, int> tile_B;
//>> B 是 8x8 的 y 分片（k 方向 8 个 int）
    typedef tile<16, 8, int> tile_C;
//>> C 是 16x8 的累加器，一个 mma 指令直接算出整块

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*sram_stride + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix
`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="flow" style="gap:8px"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const flow = wrap.querySelector('#flow');
    flow.innerHTML = '<span class="chip a">x 的 shared tile</span><span class="arrow">-></span>'
      + '<span class="chip d">load_ldmatrix</span><span class="arrow">-></span>'
      + '<span class="chip c">mma</span><span class="arrow">-></span>'
      + '<span class="chip b">sum[] 累加</span>';

    const defs = [
      { c: 'a', w: '218px', t: 'A：x 的量化数据',
        m: 'tile<16, 8, int>', b: '从 shared memory 用 load_ldmatrix 读，<br>一次搬一整块 16x8。' },
      { c: 'd', w: '218px', t: 'B：y（激活）的分片',
        m: 'tile<8, 8, int>', b: '用 load_generic 读。源码注释：<br>faster than load_ldmatrix。' },
      { c: 'b', w: '218px', t: '结果再乘标量 scale',
        m: 'sum += C.x[l]*dA*dB', b: 'mma 只算整数累加，<br>量化 scale 在循环外补乘。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'tile 内核的点积不再"一个线程算一个点"，而是<span class="k">一个 warp 算一整块</span>。',
      'A 分片从 shared memory 走 <span class="v">load_ldmatrix</span>：'
        + 'PTX 要求它必须指向 shared memory 且 16 字节对齐。',
      'B 分片走 <span class="v">load_generic</span>；源码注释说明它比 ldmatrix 更快。',
      '<span class="v">mma(C, A[n], B)</span> 一条指令覆盖 16x8 个输出；'
        + '这是 tile 内核吞吐的来源。',
      '整数点积由 mma 完成，<span class="k">量化 scale 在 C.x[l] 上补乘</span>：'
        + '<span class="m">C.x[l]*dA[n][l/2][..]*dB[l%2]</span>。',
      '对照 L5-03 的 CPU 点积：那边是标量乘加展开，这边是<span class="k">一条指令一整块</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(3200 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 K 循环：<span class="hl-c">搬一次，点两次</span> */
{
  kicker: "L6-02 · tile 加载",
  title: "K 循环：<span class=\"hl-c\">搬一次，点两次</span>",
  sub: "每轮 K 迭代把 x 的一整条 tile 搬进 shared memory，然后分两半喂给同一个点积函数。",
  caption: "两个 vec_dot 的 k00 分别是 0 与 MMQ_TILE_NE_K（= 32，mmq.cuh:116）。",
  src: "ggml/src/ggml-cuda/mmq.cuh",
  mark: [4, 5, 10, 19, 27, 36],
  lineNo: 904,
  code: `    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
//>> load_tiles：把一个 K 迭代的 x tile 协作搬进 shared memory
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);
//>> 前 32 个元素上做点积（k00 = 0）

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
//>> 后 32 个元素上做点积（k00 = MMQ_TILE_NE_K = 32）

        __syncthreads();
    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="steps" style="gap:7px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', w: '163px', t: '1 · load_tiles', m: 'load_tiles(x, tile_x, ...)', b: '协作把 x 的量化数据<br>搬进 shared memory' },
      { c: 'b', w: '163px', t: '2 · sync + dot', m: 'vec_dot(tile_x, tile_y, sum, 0)', b: 'k00 = 0，<br>前半个 K 分片' },
      { c: 'c', w: '163px', t: '3 · 续搬 tile_y', m: 'tile_y[l] = by0[l]', b: '只换 y 的下一段，<br>x tile 不重搬' },
      { c: 'd', w: '163px', t: '4 · sync + dot', m: 'vec_dot(..., MMQ_TILE_NE_K)', b: 'k00 = 32，<br>后半个 K 分片' }
    ];
    const host = wrap.querySelector('#steps');
    const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">ITER_K</span> 是每轮从显存搬多少 K（mmq.cuh:9 给出 256），'
        + '<span class="m">blocks_per_iter = ITER_K / qk</span>。',
      '关键成本在 <span class="k">x</span>：量化权重。它每轮只搬一次。',
      '搬完立刻在<span class="v">前半段</span>做点积，同时把 y 的后半段续进 tile_y。',
      '同一个 x tile 上做<span class="v">第二次点积</span>，k00 偏移 32 —— '
        + '这就是"搬一次、点两次"。',
      '两块 <span class="m">__syncthreads()</span> 把 shared memory 的读写严格分开：'
        + '<span class="k">写 tile_y 之前必须等上一次点积读完</span>。',
      '这就是 tile 内核摊薄载入成本的方式：<span class="k">权重搬一次，被 J 个 token 和 2 段 K 复用</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(3200 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 协作加载：一个 warp 摊平一整块 tile */
{
  kicker: "L6-02 · tile 加载",
  title: "协作加载：一个 warp 摊平一整块 tile",
  sub: "线程编号先按 K 方向切（每个线程固定搬哪几块），再按 I 方向切（搬第几行）。",
  caption: "MMQ_ITER_K = 256、QR4_0 = 2（ggml-common.h），故 threads_per_row = 256/(4*2) = 32，正好一个 warp。",
  src: "ggml/src/ggml-cuda/mmq-load-tiles.cuh",
  mark: [0, 16, 18, 25, 32, 40],
  lineNo: 187,
  code: `template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);
//>> 一个 warp 负责一整行的 K：256 / (4 * QR4_0) = 32 个线程
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_0;
    const int kqsx = txi % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
//>> nrows == 1 时，行号直接就是 threadIdx.y

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
//>> x + kbx0 + i*stride + kbx：拿到这一行这一块的源地址
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i*sram_stride + kbx*(2*QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    }`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="cm" id="cap"></div>
      <div class="row" id="grid" style="gap:2px"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    wrap.querySelector('#cap').innerHTML =
      '一个 warp（32 线程）覆盖一行 x 的 256 个 K 元素 = 8 个 Q4_0 块，每块 4 个线程：';

    const grid = wrap.querySelector('#grid');
    const cells = [];
    for (let i = 0; i < 32; i++) {
      const kbx = Math.floor(i / 4);
      const c = U.el('div', {
        style: 'width:19px;height:19px;border:1px solid var(--border);border-radius:3px;'
             + 'background:#10151b;display:flex;align-items:center;justify-content:center;'
             + 'font-size:8px;color:var(--dim);font-family:var(--mono)' });
      c.textContent = kbx;
      grid.appendChild(c); cells.push(c);
    }

    const t = U.table(['式子', '这一行算出什么'],
      [['threads_per_row = MMQ_ITER_K / (4 * QR4_0)', '32 —— 一个 warp 正好一行'],
       ['txi = threadIdx.x % threads_per_row', '我在这一行里的位置'],
       ['kbx = txi / QI4_0', '我负责第几个 Q4_0 块（0..7）'],
       ['kqsx = txi % QI4_0', '我负责块内的第几个 32 位字'],
       ['i = i0 + threadIdx.y', '我负责第几行（nrows == 1）'],
       ['x + kbx0 + i*stride + kbx', '源地址：第 i 行、kbx0 起的第 kbx 块']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '加载不是"每个线程搬一个元素"，而是先<span class="k">把线程铺到 tile 的两个方向上</span>。',
      '<span class="v">K 方向</span>：8 个块 x 4 个线程 = 32，正好填满一个 warp。',
      '<span class="v">I 方向</span>：每 32 个线程负责一行，靠 <span class="m">i*stride</span> 跳到下一行。',
      '<span class="k">stride 来自 src0->nb[1]</span>：tile 里的行序必须和全局内存里的行序一致。',
      '<span class="m">fallback</span> 分支处理最后一行不满的情况：<span class="m">i = min(i, i_max)</span>。',
      '一格一格的全局读被换成<span class="k">整齐的块读</span>，这就是 tile 加载比 mmvq 更适合大 batch 的原因。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 1, 2, 3, 4, 5]); });
    tl.at(3200, () => {
      cells.forEach((c, i) => { if (i % 4 === 0) { c.style.background = 'rgba(88,166,255,.25)'; c.style.borderColor = 'var(--a)'; } });
      msg.innerHTML = texts[1];
    });
    tl.at(6200, () => {
      cells.forEach((c, i) => { c.style.background = (Math.floor(i / 4) % 2) ? 'rgba(210,153,34,.22)' : 'rgba(88,166,255,.25)'; });
      msg.innerHTML = texts[2];
    });
    tl.at(9200, () => { msg.innerHTML = texts[3]; U.markLines(document, [29]); });
    tl.at(12200, () => { msg.innerHTML = texts[4]; U.markLines(document, [25, 26, 27]); });
    tl.at(15200, () => { msg.innerHTML = texts[5]; U.markLines(document, [29, 35]); });
  }
},

/* ------------------------------------------------------ 9 ★ batch 不只选内核，<span class="hl-a">还选 tile 的宽度 J</span> */
{
  kicker: "L6-02 · ★ 洞察",
  title: "★ batch 不只选内核，<span class=\"hl-a\">还选 tile 的宽度 J</span>",
  sub: "进了 mmq 之后，还要在 16 个编译期 J 实例里挑一个：J 是\"一次处理几个 token\"。",
  caption: "J_best 的候选是 8,16,...,128；每个候选对应一个 launch_mul_mat_q<type, J> 模板实例（mmq.cuh:1504-1552）。下面的算例只推演 ntiles_x 公式，实际还要过 :1492 那道 shared memory 门。",
  src: "ggml/src/ggml-cuda/mmq.cuh",
  mark: [0, 6, 9, 15, 20, 23],
  lineNo: 1477,
  code: `template <ggml_type type, bool fallback>
void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
//>> 装不进 shared memory 的 J 直接跳过
            continue;
        }

        const int ntiles_x = (args.ncols_opt + config.J - 1) / config.J;
//>> ncols_opt 就是 batch：需要多少列 tile 才能盖住它

        if (ntiles_x < ntiles_J_best) {
//>> 挑 tile 列数最少的 J；从 8 往上扫，先到先得
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="cm" id="cap"></div>
      <div class="col" id="bars" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    wrap.querySelector('#cap').innerHTML =
      '以 <span class="m">ncols_opt = 40</span> 为例：ntiles_x = ceil(40 / J)，取最小的那个';

    const host = wrap.querySelector('#bars');
    const cand = [
      { j: 8, n: 5, c: 'e', ok: true }, { j: 16, n: 3, c: 'e', ok: true },
      { j: 24, n: 2, c: 'c', ok: true }, { j: 32, n: 2, c: 'c', ok: true },
      { j: 40, n: 1, c: 'b', ok: true }, { j: 48, n: 1, c: 'f', ok: false },
      { j: 56, n: 1, c: 'f', ok: false }, { j: 64, n: 1, c: 'f', ok: false }
    ];
    const bars = cand.map(d => {
      const b = U.bar('J = ' + d.j, d.c);
      host.appendChild(b.el);
      b.fill.style.width = '0%';
      b.val.textContent = d.n + ' 列';
      return b;
    });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '进 mmq 只是第一步。内核还要在 <span class="k">16 个编译期 J 实例</span> 里挑一个。',
      'J 是 tile 的列宽，也就是<span class="v">一次搬进来几个 token 的 x</span>。',
      '<span class="m">ntiles_x = (ncols_opt + J - 1) / J</span>：盖住 batch 需要多少个列 tile。',
      '扫描从 J = 8 开始，只接受 <span class="v">ntiles_x 更小</span> 的候选 —— '
        + '所以 J 越宽，列 tile 越少。',
      '但 J 受 <span class="v">shared memory 上限</span> 约束：装不下的候选被 '
        + '<span class="m">continue</span> 跳过。',
      '结果：<span class="k">batch = 40 时，按 ntiles_x 公式 J_best 会落在 40</span>'
        + '（一个列 tile 盖住全部 token）；能不能真的用 40，由 :1492 的 SRAM 门决定。',
      '这与 L6-04 的 <span class="m">template-instances/</span> 是同一机制：'
        + '<span class="k">编译期把 J 摊成 16 份，运行时按 batch 选一份</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    [0, 1, 2, 3, 4, 5].forEach(i => tl.at(2900 + i * 2400, () => {
      bars.forEach((b, k) => {
        b.el.style.opacity = (k <= i + 1) ? '1' : '.30';
        if (k <= i + 1) { b.fill.style.width = Math.min(100, cand[k].n * 20) + '%'; }
      });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16400, () => {
      bars.forEach((b, k) => { b.el.style.opacity = '1'; b.fill.style.width = Math.min(100, cand[k].n * 20) + '%'; });
      bars.forEach((b, k) => { if (k !== 4) { b.el.style.opacity = '.35'; } });
      msg.innerHTML = texts[6];
    });
  }
},

/* ------------------------------------------------------ 10 把三套内核压成一张表 */
{
  kicker: "L6-02 · 收束",
  title: "把三套内核压成一张表",
  sub: "同一个量化点积，三种形状假设。选择依据只有一条：矩阵有多瘦。",
  caption: "下一课 L6-03 讲 FlashAttention —— 那里有完全一样的三条路线（vec / tile / mma）。",
  src: "ggml/src/ggml-cuda/mmvq.cuh",
  mark: [2, 4],
  lineNo: 1,
  code: `#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['内核', '判据（函数 : 行）与阈值', '适用形状', '关键差异'],
      [['mmvf', { html: 'ggml_cuda_should_use_mmvf<br>mmvf.cu:792<br>ne11 &lt;= 3 / 4 / 8' },
        { html: 'float 权重 × 极瘦矩阵<br>（KQV 那类）' },
        { html: '不量化，按 half2/float2 直接读；<br>先过三条对齐前提' }],
       ['mmvq', { html: 'ggml_cuda_should_use_mmvq<br>mmvq.cu:318<br>ne11 &lt;= MMVQ_MAX_BATCH_SIZE = 8' },
        { html: '量化权重 × 1..8 个 token<br>（decode 阶段）' },
        { html: '每线程读一整行权重，权重按<br>token 数重复读；VDR 展开窄' }],
       ['mmq', { html: 'ggml_cuda_should_use_mmq<br>mmq.cu:266<br>turing_mma 一律 true，<br>否则 ne11 &lt; 64' },
        { html: '量化权重 × 9..64 个 token<br>（prefill 阶段）' },
        { html: '权重搬进 shared tile 被 J 个<br>token 复用；VDR 展开宽' }],
       ['cuBLAS', { html: 'ggml-cuda.cu:1876<br>以上都不满足' },
        { html: 'batch 更大，或类型<br>不被 mmq 支持' },
        { html: '先反量化再 GEMM；本课不展开' }]]);
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '在同一张 Turing 卡上，<span class="mono">src0</span> 是 Q4_K 量化权重、'
      + '<span class="mono">src1</span> 分别是 1 个 token 和 32 个 token。'
      + '两次分别走哪个内核？为什么？',
      '<span class="mono">ne11 = 1</span>：<span class="mono">ggml_cuda_should_use_mmvq</span> '
      + '（mmvq.cu:318）先被问到，量化类型且 <span class="mono">ne11 &lt;= MMVQ_MAX_BATCH_SIZE (= 8)</span>，'
      + '返回 true → 走 <span class="mono">ggml_cuda_mul_mat_vec_q</span>。<br>'
      + '<span class="mono">ne11 = 32</span>：mmvq 的判据为 false（32 &gt; 8）；'
      + '接着 <span class="mono">ggml_cuda_should_use_mmq</span>（mmq.cu:266）在 '
      + '<span class="mono">turing_mma_available(cc)</span> 处直接返回 true → 走 '
      + '<span class="mono">ggml_cuda_mul_mat_q</span>，并在其中用 '
      + '<span class="mono">ncols_opt = 32</span> 挑 J（mmq.cuh:1496）—— '
      + 'J 越宽，盖住 32 个 token 所需的列 tile 越少。<br>'
      + '一句话：<span class="mono">ne11</span> 一路决定到 tile 宽度。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '三套内核，一条判据轴：<span class="v">矩阵有多瘦</span>。',
      '<span class="k">瘦到 3 以内</span>：float 路径的 mmvf 最快。',
      '<span class="k">1..8 个 token</span>：mmvq —— 并行度来自"更多线程各读一行权重"。',
      '<span class="k">9..64 个 token</span>：mmq —— 权重搬进 shared tile，被多个 token 复用。',
      '再宽就交给 cuBLAS：<span class="k">tile 复用已经摊不动了</span>。',
      '记住落点：<span class="m">ggml_cuda_mul_mat() 在 ggml-cuda.cu:1823</span>，'
        + '判据函数分散在 mmvq.cu / mmq.cu / mmvf.cu —— 下一课在 FlashAttention 里会再见到同一套三分法。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(14200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

];
