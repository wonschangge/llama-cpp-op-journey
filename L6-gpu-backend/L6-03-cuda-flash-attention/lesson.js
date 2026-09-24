/* ==========================================================================
   L6-03 · ★ CUDA FlashAttention：三条路线与一张实例矩阵
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 ★ 不是一个 kernel：<span class="hl-a">三条路线</span> + <span class="hl-c">一张实例矩阵</span> */
{
  kicker: "L6 · GPU 后端执行",
  title: "★ 不是一个 kernel：<span class=\"hl-a\">三条路线</span> + <span class=\"hl-c\">一张实例矩阵</span>",
  sub: "入口函数只做两件事：按形状与设备挑路线，再查表拿到那份已经编译进库的模板实例。",
  caption: "回顾 L6-01：ggml-cuda.cu 的大 switch 里 case GGML_OP_FLASH_ATTN_EXT 调用的就是下面这个函数。回顾 L6-02：mmq/mmvq 是同一个模式 —— 按形状在多个已实例化的 kernel 之间选一个。",
  src: "ggml/src/ggml-cuda/fattn.cu",
  mark: [2, 5, 6, 8, 9, 11, 12],
  lineNo: 755,
  code: `void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">图上的 FLASH_ATTN_EXT 节点</span><span class="arrow">-></span>
        <span class="chip a">get_best_fattn_kernel()</span><span class="arrow">-></span>
        <span class="chip c">三条路线之一</span><span class="arrow">-></span>
        <span class="chip b">模板实例</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'BEST_FATTN_KERNEL_VEC', b: 'per-element 路线，小 batch 最快<br>head size 只有 64 / 128 / 256', m: 'fattn-vec.cuh' },
      { c: 'c', t: 'BEST_FATTN_KERNEL_TILE', b: 'SIMT tile 路线，不需要 Tensor Core<br>覆盖全部 head size（含 40 / 72）', m: 'fattn-tile.cuh' },
      { c: 'b', t: 'BEST_FATTN_KERNEL_MMA_F16', b: 'Tensor Core 路线，大 batch 最快<br>K/V 先转成 f16，ncols1/ncols2 分块', m: 'fattn-mma-f16.cuh' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '同一个节点，三种实现。<span class="k">先把三条路线分开看，再合起来看它们怎么被选中。</span>',
      '<span class="v">NONE</span>：不在白名单的 head size / 类型，直接返回 0 —— 这不是错误，' +
      '而是告诉调度器“CUDA 后端不支持这个 FLASH_ATTN_EXT”，节点会被切给别的后端（L4-02）。',
      '<span class="v">VEC</span>：解码（batch 小）时最快。它的 dtype 支持面最宽（含量化 K/V）。',
      '<span class="v">TILE</span>：<span class="k">兜底路线</span> —— 没有 Tensor Core 时也能跑，' +
      '也是 K/V 必须转 f16 的两条路线之一。',
      '<span class="v">MMA_F16</span>：Tensor Core 路线，prefill（batch 大）时最快；' +
      '实例键是四元组 (DKQ, DV, ncols1, ncols2)。',
      '记住这张图：<span class="k">路线 = f(head size, dtype, GPU 能力, batch)</span>；' +
      '而每条路线内部还有一张<span class="k">实例矩阵</span>决定哪些组合真的存在。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 2700, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14500, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      U.markLines(document, [2, 5, 8, 11]);
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 2 第一步：<span class="hl-b">head size 白名单</span> */
{
  kicker: "L6-03 · 进门条件",
  title: "第一步：<span class=\"hl-b\">head size 白名单</span>",
  sub: "选择逻辑的第一段就是一个 switch (K->ne[0])；不在白名单里的 head size 直接返回 NONE。",
  caption: "这张白名单就是“CUDA FlashAttention 支持哪些 head size”的唯一真值。NONE 会让 ggml_cuda_flash_attn_ext_supported() 返回 false（同文件 772-774 行）。",
  src: "ggml/src/ggml-cuda/fattn.cu",
  mark: [1, 6, 13, 14, 21, 22, 29, 30, 33, 37, 38, 41, 45, 46],
  lineNo: 576,
  code: `    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['K->ne[0]', 'V->ne[0] 的要求', '附加条件'],
      [['40 / 64 / 72 / 80 / 96 / 112 / 128 / 256', 'V->ne[0] == K->ne[0]', '无（对称 head size）'],
       ['192', 'V->ne[0] == 128', 'gqa_opt_applies 且 gqa_ratio % 8 == 0'],
       ['320', 'V->ne[0] == 256', 'gqa_opt_applies 且 gqa_ratio % 32 == 0'],
       ['512', 'V->ne[0] == 512', 'gqa_opt_applies'],
       ['576', 'V->ne[0] == 512', 'gqa_opt_applies'],
       ['其它', '—', 'default: return BEST_FATTN_KERNEL_NONE']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看最外层：<span class="v">switch (K->ne[0])</span>，不在白名单里就直接出局。',
      '<span class="k">8 个对称 head size</span>（含常见的 64 / 128 / 256）：只要求 V 与 K 同宽。',
      '<span class="k">192 / 320 / 576 是“K 宽 V 窄”的不对称组合</span>：' +
      '576 的 K、V 宽度分别是 576 与 512（MLA 类的模型就是这种形状）。',
      '<span class="v">gqa_opt_applies</span>（561 行）= gqa_ratio ≥ 2 <b>且</b> 有 mask <b>且</b> ' +
      '<b>max_bias == 0</b>（没有 ALiBi）<b>且</b> K->ne[1] % FATTN_KQ_STRIDE == 0。',
      '换句话说：<span class="k">ALiBi 或没有 mask 时，GQA 优化不生效</span>，' +
      '192/320/512/576 这几个不对称 head size 直接不支持。',
      '白名单不过 → NONE → 调度器把节点切给别的后端（L4-02 的切分逻辑）。' +
      '<span class="k">能进门 ≠ 能走三条路线</span>，下一幕才是路线判据。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => {
      rows.forEach((x, k) => { x.className = k === 0 ? 'on' : ''; });
      msg.innerHTML = texts[1];
    });
    tl.at(7300, () => {
      rows.forEach((x, k) => { x.className = (k >= 1 && k <= 4) ? 'on' : ''; });
      msg.innerHTML = texts[2];
    });
    tl.at(11200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
    tl.at(14800, () => { rows.forEach((x, k) => { x.className = k === 2 ? 'on' : ''; }); msg.innerHTML = texts[4]; });
    tl.at(18400, () => {
      rows.forEach((x, k) => { x.className = k === 5 ? 'on' : ''; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 3 ★ 路线判据：<span class="hl-a">can_use_vector_kernel</span> × 设备能力 */
{
  kicker: "L6-03 · 核心",
  title: "★ 路线判据：<span class=\"hl-a\">can_use_vector_kernel</span> × 设备能力",
  sub: "vec 只在小 batch 时赢；只要 GPU 有 Tensor Core 且 batch 够大，就走 mma。",
  caption: "这一段是验收点的答案：vec 与 mma 各自的适用条件。excerpt 之外还有 Volta / AMD / 无 Tensor Core 三个分支（同文件 673-716 行），下一步的文字里逐个说。",
  src: "ggml/src/ggml-cuda/fattn.cu",
  mark: [2, 5, 6, 9, 11, 14, 16, 19, 21, 22, 26, 27, 31, 32, 35],
  lineNo: 633,
  code: `    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;
//>> vec 路线的形状门槛：Q->ne[0] ≤ 256、是 64 的倍数、且不是 192（192 没有 vec 实例）

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
//>> Turing 及更新的 NVIDIA（含 Ada / Hopper / Blackwell）走这个分支
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
//>> batch = 1 的纯解码：未量化 K/V 的唯一 vec 入口
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
//>> Ada 且 K/V 量化：batch ≤ 2 也还给 vec
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
//>> 走到这里就是 Tensor Core（mma）路线
    }`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['设备能力（按判据顺序）', '小 batch', '大 batch'],
      [['turing_mma_available（Turing 及更新）', 'VEC（形状满足 + batch 够小）', 'MMA_F16'],
       ['volta_mma_available', 'VEC；中等 batch 用 TILE', 'MMA_F16'],
       ['amd_mfma_available / amd_wmma_available', 'VEC 或 TILE', 'MMA_F16（batch 够大时）'],
       ['没有 Tensor Core（兜底）', 'VEC', 'TILE']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '判据一：<span class="v">can_use_vector_kernel</span>（635 行）= ' +
      'Q->ne[0] ≤ 256 <b>且</b> % 64 == 0 <b>且</b> ≠ 192 <b>且</b> K->ne[1] % FATTN_KQ_STRIDE == 0。',
      '判据二：<span class="v">turing_mma_available(cc)</span>（638 行）。有 Tensor Core 时，' +
      '只有 batch 很小才轮到 vec —— 这就是第一行。',
      '未量化 K/V：Ada 上仅 <span class="v">Q->ne[1] == 1</span>（645-648 行）；' +
      '量化 K/V 更宽：Ada 上 <span class="v">Q->ne[1] ≤ 2</span>，更老的卡只给 <span class="v">== 1</span>（650-658 行）。',
      '都不满足 → <span class="v">return BEST_FATTN_KERNEL_MMA_F16</span>（664 行）。' +
      '<span class="k">mma 是 Turing 及更新显卡上的默认路线。</span>',
      '<span class="v">Volta</span>（673-681 行）：tensor core 只在大矩阵上划算，' +
      '所以中等 batch（Q->ne[1]*gqa_ratio_eff ≤ 16）交给 TILE。',
      '<span class="v">AMD MFMA / WMMA</span>（684-700 行）：batch 超过阈值（≤64 头时 >8 / ≤128 时 >16 / ' +
      '≤256 时 >64）才用 mma，否则落在 TILE。',
      '没有任何 Tensor Core（702-716 行）：<span class="k">小 batch 用 vec，其余全走 tile</span> —— ' +
      'tile 是三条路线里的兜底路线。'
    ];
    rows.forEach((r, i) => tl.at(700 + i * 3400, () => {
      rows.forEach((x, k) => { x.className = k === i ? 'on' : ''; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14300, () => { msg.innerHTML = texts[4]; });
    tl.at(17700, () => { msg.innerHTML = texts[5]; });
    tl.at(21100, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 4 vec 的 dtype 矩阵：<span class="hl-b">3 个 head size × 7 × 7</span> */
{
  kicker: "L6-03 · vec 路线",
  title: "vec 的 dtype 矩阵：<span class=\"hl-b\">3 个 head size × 7 × 7</span>",
  sub: "DECL_FATTN_VEC_CASE 把 (head size, type_K, type_V) 三元组逐一声明成显式实例化；EXTERN 列表把 7×7 铺满。",
  caption: "head size 只有 64 / 128 / 256 三种 —— 这就是“fattn-vec 适用什么 head size”的答案。实例文件名直接编码了 dtype 组合：fattn-vec-instance-q8_0-q4_0.cu 之类。",
  src: "ggml/src/ggml-cuda/fattn-vec.cuh",
  mark: [0, 4, 5, 11, 13, 21, 29],
  lineNo: 574,
  code: `#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \\
    template void ggml_cuda_flash_attn_ext_vec_case                         \\
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \\

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \\
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \\
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \\
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \\
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \\
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \\
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \\
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \\

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="grid"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const types = ['F16', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'BF16'];
    const grid = U.el('div', { style: 'display:grid;grid-template-columns:52px repeat(7,1fr);gap:3px;font-size:9px;font-family:var(--mono)' });
    const cells = [];
    function cell(txt, color) {
      const e = U.el('div', { text: txt, style: 'padding:3px 2px;text-align:center;border:1px solid var(--border);border-radius:3px;color:' + (color || 'var(--muted)') });
      grid.appendChild(e);
      return e;
    }
    cell('K \\ V', 'var(--dim)');
    types.forEach(v => cell(v, 'var(--dim)'));
    types.forEach(k => {
      cell(k, 'var(--dim)');
      types.forEach(v => { cells.push({ k: k, v: v, el: cell(k + '-' + v) }); });
    });
    wrap.querySelector('#grid').appendChild(grid);
    cells.forEach(c => { c.el.style.opacity = '.30'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '7 种 K 类型 × 7 种 V 类型 = <span class="k">49 个 (type_K, type_V) 组合</span>，每一个组合都是一个 .cu 文件。',
      '每个 .cu 文件里有 <span class="k">3 行 DECL_FATTN_VEC_CASE</span>（D = 64 / 128 / 256，' +
      '见 587-609 行的 EXTERN 列表）→ 49 × 3 = <span class="v">147 份显式实例化</span>。',
      '宏（419-426 行）还额外接受 <span class="v">F32</span>：' +
      '“type_K == GGML_TYPE_F32 && 实例类型 == F16”时按 F16 实例用它。',
      '注意：这里只声明了 <b>哪些组合可以存在</b>。' +
      '<span class="k">真正被编译进库的是 CMake 选中的一部分</span>（第 8 幕）。',
      '回到验收点：<span class="v">fattn-vec</span> = head size ∈ {64, 128, 256}，' +
      'K/V ∈ {F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16}（+F32 视作 F16）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4200, () => {
      cells.forEach(c => { c.el.style.opacity = '1'; c.el.style.borderColor = 'var(--b)'; });
      msg.innerHTML = texts[1];
    });
    tl.at(8200, () => {
      cells.forEach(c => { c.el.style.opacity = (c.k === 'F16') ? '1' : '.30'; });
      msg.innerHTML = texts[2];
    });
    tl.at(12200, () => {
      cells.forEach(c => { c.el.style.opacity = '1'; c.el.style.borderColor = 'var(--border)'; });
      msg.innerHTML = texts[3];
    });
    tl.at(16200, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 <span class="hl-b">per-element</span>：一个 block 128 线程，一次处理 1~2 个 Q 列 */
{
  kicker: "L6-03 · vec 路线",
  title: "<span class=\"hl-b\">per-element</span>：一个 block 128 线程，一次处理 1~2 个 Q 列",
  sub: "vec 把 KQ 点积按“每个线程负责 D 的一段”切开；batch=1 时 cols_per_block=1，否则 2。",
  caption: "launch_fattn<D, cols_per_block, 1> 里 ncols2 = 1、stream_k = false：vec 不做 GQA 方向的分块，也不切 K/V 方向 —— 这正是它只适合小 batch 的原因。",
  src: "ggml/src/ggml-cuda/fattn-vec.cuh",
  mark: [1, 8, 9, 20, 23],
  lineNo: 544,
  code: `template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'nthreads = 128', b: 'ggml_cuda_fattn_vec_get_nthreads_host()<br>固定返回 128（4 个 warp）', m: 'fattn-vec.cuh:4-7' },
      { c: 'b', t: 'cols_per_block ∈ {1, 2}', b: 'Q->ne[1] == 1 → 1<br>否则 → 2（552 / 564 行）', m: 'cols_per_block' },
      { c: 'c', t: 'vec_dot_KQ', b: 'get_vec_dot_KQ<type_K, D, nthreads_KQ>()<br>KQ 点积按 K 的类型选实现', m: 'fattn-vec.cuh:96' },
      { c: 'd', t: 'VKQ 累加器', b: '每个线程持有的累加器：<br>D=128 时是 1 个 half2 / 线程', m: 'VKQ[ncols][D/2/nthreads_V]' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'vec 的模板参数只有 <span class="v">D</span> 与 <span class="v">cols_per_block</span> 是整数：' +
      'dtype 是类型参数（type_K / type_V），softcap 是 bool。',
      'nthreads_KQ / nthreads_V 由 <span class="k">K/V 的类型</span>决定（87-88 行）：' +
      'f16/bf16 直接算 128/cpy_nb，量化类型走 nthreads_*_q。',
      '<span class="k">D % (2*WARP_SIZE) == 0</span> 是一条 static_assert（117 行）——' +
      'D 必须是 64 的倍数。这就是第 3 幕里 <span class="v">% 64 == 0</span> 门槛的物理原因。',
      '每个 block 的 <span class="v">Q += ... + nb01*ic0</span>：' +
      'blockIdx.x 直接对应 Q 的列（104-111 行），<span class="k">没有 K/V 方向的并行</span>。',
      '所以 vec 只赢在“<span class="k">Q 列很少</span>”：Q 一多，同样的 K/V 会被重复读很多遍 —— ' +
      '那正是 tile / mma 用 ncols1/ncols2 分块要解决的问题。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13500, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 tile：<span class="hl-c">12 组 (DKQ, DV)</span> 全是显式实例 */
{
  kicker: "L6-03 · tile 路线",
  title: "tile：<span class=\"hl-c\">12 组 (DKQ, DV)</span> 全是显式实例",
  sub: "tile 是“没有 Tensor Core 也能跑”的通用路线：KQ 用 SIMT 的向量点积，覆盖全部 head size。",
  caption: "DECL_FATTN_TILE_CASE 的 12 行就是 tile 的全部 head size 组合；实例文件名 dkq192-dv128.cu 之类直接编码了这些组合。注意 40 与 72 只存在于 tile 路线（mma 没有它们）。",
  src: "ggml/src/ggml-cuda/fattn-tile.cuh",
  mark: [1, 7, 18, 22, 29, 30, 33],
  lineNo: 1322,
  code: `template <int DKQ, int DV>
void ggml_cuda_flash_attn_ext_tile_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \\
    template void ggml_cuda_flash_attn_ext_tile_case              \\
    <DKQ, DV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \\

extern DECL_FATTN_TILE_CASE( 40,  40);
extern DECL_FATTN_TILE_CASE( 64,  64);
extern DECL_FATTN_TILE_CASE( 72,  72);
extern DECL_FATTN_TILE_CASE( 80,  80);
extern DECL_FATTN_TILE_CASE( 96,  96);
extern DECL_FATTN_TILE_CASE(112, 112);
extern DECL_FATTN_TILE_CASE(128, 128);
extern DECL_FATTN_TILE_CASE(192, 128);
extern DECL_FATTN_TILE_CASE(256, 256);
extern DECL_FATTN_TILE_CASE(320, 256);
extern DECL_FATTN_TILE_CASE(512, 512);
extern DECL_FATTN_TILE_CASE(576, 512);`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['DKQ', 'DV', '形状', '只有 tile 支持？'],
      [['40', '40', '对称', '是（mma 的 switch 里没有 40）'],
       ['64', '64', '对称', '否'],
       ['72', '72', '对称', '是（mma 的 switch 里没有 72）'],
       ['80 / 96 / 112', '同 DKQ', '对称', '否'],
       ['128', '128', '对称', '否'],
       ['192', '128', 'K 宽 V 窄', '否'],
       ['256', '256', '对称', '否'],
       ['320', '256', 'K 宽 V 窄', '否'],
       ['512', '512', '对称', '否'],
       ['576', '512', 'K 宽 V 窄', '否']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      'tile 的实例键是二元组 <span class="v">(DKQ, DV)</span>：12 行 extern 声明 = 12 个 .cu 文件。',
      '<span class="k">40 与 72 是 tile 独占的</span>：mma 的 switch（fattn.cu 288-416 行）没有这两个 case，' +
      '选择逻辑也用 <span class="v">Q->ne[0] != 40 &amp;&amp; Q->ne[0] != 72</span> 把它们挡在 Tensor Core 路线之外。',
      '192 / 320 / 576 是 <span class="k">K 宽 V 窄</span>的不对称组合：' +
      'KQ 用 DKQ、VKQ 用 DV，模板参数必须分开写。',
      'tile 也要 f16：它的 case 只把 <span class="v">use_logit_softcap</span> 编成两个版本（1329-1335 行），' +
      'K/V 的 dtype 不参与模板 —— 非 f16 的 K/V 在 launch_fattn 里先被转成 f16（第 8 幕）。',
      '一句话：<span class="k">tile = 形状覆盖最全、但对 dtype 最不敏感的那条路线</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4000, () => {
      rows.forEach((x, k) => { x.className = (k === 0 || k === 2) ? 'on' : ''; });
      msg.innerHTML = texts[1];
    });
    tl.at(7900, () => {
      rows.forEach((x, k) => { x.className = (k >= 5) ? 'on' : ''; });
      msg.innerHTML = texts[2];
    });
    tl.at(11800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
    tl.at(15400, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 mma：<span class="hl-a">Tensor Core</span> + <span class="hl-d">ncols1 / ncols2</span> 两个模板参数 */
{
  kicker: "L6-03 · mma 路线",
  title: "mma：<span class=\"hl-a\">Tensor Core</span> + <span class=\"hl-d\">ncols1 / ncols2</span> 两个模板参数",
  sub: "mma 的实例键是四元组 (DKQ, DV, ncols1, ncols2)：ncols1 是 Q 行方向的 tile 宽度，ncols2 是 GQA 方向的宽度。",
  caption: "launch_fattn<DV, ncols1, ncols2>(..., true, true, true) 里的前两个 true 是 need_f16_K / need_f16_V：mma 路线只吃 f16 的 K/V。第三个 true 是 stream_k —— 大 batch 时沿 KV 方向再切一刀。",
  src: "ggml/src/ggml-cuda/fattn-mma-f16.cuh",
  mark: [0, 1, 6, 10, 11, 15],
  lineNo: 2116,
  code: `    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, use_sparse, warp_size_host);
//>> 第 7~9 个实参依次是 need_f16_K / need_f16_V / stream_k
}


#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \\
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \\
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \\

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \\
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \\
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \\
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \\
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \\
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \\

//>> ALL_NCOLS2 一次展开 5 个 ncols2，ncols1 = ncols / ncols2`,
  duration: 23000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px;align-items:flex-start">
        <div id="grid"></div>
        <div class="col grow" id="note" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const cols2 = [1, 2, 4, 8, 16, 32];
    const rows1 = [1, 2, 4, 8, 16, 32, 64];
    const have = { '1': [8, 16, 32, 64], '2': [4, 8, 16, 32], '4': [2, 4, 8, 16], '8': [1, 2, 4, 8], '16': [1, 2, 4], '32': [1, 2] };
    const grid = U.el('div', { style: 'display:grid;grid-template-columns:48px repeat(6,38px);gap:3px;font-size:9px;font-family:var(--mono)' });
    const cells = [];
    function cell(txt, color, bg) {
      const e = U.el('div', { text: txt, style: 'padding:3px 0;text-align:center;border:1px solid var(--border);border-radius:3px;color:' + (color || 'var(--dim)') + (bg ? ';background:' + bg : '') });
      grid.appendChild(e);
      return e;
    }
    cell('ncols1\\ncols2', 'var(--dim)');
    cols2.forEach(c => cell(String(c), 'var(--dim)'));
    rows1.forEach(r => {
      cell(String(r), 'var(--dim)');
      cols2.forEach(c => {
        const ok = (have[String(c)] || []).indexOf(r) >= 0;
        cells.push({ r: r, c: c, ok: ok, el: cell(ok ? '●' : '·', ok ? 'var(--a)' : 'var(--border)') });
      });
    });
    wrap.querySelector('#grid').appendChild(grid);
    wrap.querySelector('#note').innerHTML =
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">21 个 (ncols1, ncols2) 组合</div>' +
      '<div class="cb">每个组合一个 .cu 文件，21 个文件里再逐 (DKQ, DV) 展开，' +
      '合计 126 行 DECL_FATTN_MMA_F16_CASE。' +
      '<span class="cm" style="margin:0">ncols1 * ncols2 &lt;= 64</span></div></div>' +
      '<div class="card" style="border-left-color:var(--d)">' +
      '<div class="ct" style="color:var(--d)">ncols2 是 GQA 方向</div>' +
      '<div class="cb">ncols2 个 Q head 共享同一份 K/V；' +
      'ncols2 越大，K/V 复用越多，但只对 gqa_ratio 大的模型有意义。' +
      '<span class="cm" style="margin:0">ncols = ncols1 * ncols2</span></div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '每个 ● 都是一个真的存在的实例文件：<span class="v">fattn-mma-f16-instance-ncols1_8-ncols2_8.cu</span> 这样命名。',
      '模板参数 <span class="v">ncols1 * ncols2</span> = 一个 block 一次处理的 Q 列数。' +
      '实例只覆盖乘积 <span class="k">8 ~ 64</span> 的那些组合（1/2/4 的乘积没有生成）。',
      '为什么乘积上界是 64：<span class="v">switch_ncols1</span>（fattn.cu 170-190 行）只会去取 ' +
      '<span class="v">8/ncols2、16/ncols2、32/ncols2、64/ncols2</span> 这几种 ncols1。',
      'ncols2 由 <span class="v">switch_ncols2</span>（fattn.cu 193-416 行）按 head size 与 gqa_ratio 选，' +
      '每个 head size 只生成“用得上”的 ncols2：<span class="k">576 只生成 4 / 16 / 32</span>。',
      '回到验收点：<span class="v">fattn-mma</span> 的 head size 是 64…576（不含 40 / 72），' +
      'K/V 必须是 f16（非 f16 会先转换），因此 dtype 不进入模板参数 —— 进模板的是 (DKQ, DV, ncols1, ncols2)。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4500, () => {
      cells.forEach(c => { c.el.style.opacity = c.ok ? '1' : '.25'; });
      msg.innerHTML = texts[1];
    });
    tl.at(8800, () => { msg.innerHTML = texts[2]; });
    tl.at(13200, () => {
      cells.forEach(c => { c.el.style.opacity = (c.ok && c.c === 4) ? '1' : '.25'; });
      msg.innerHTML = texts[3];
    });
    tl.at(17800, () => {
      cells.forEach(c => { c.el.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 8 为什么 <span class="hl-c">template-instances/</span> 必须存在：查不到就退化成 f16 */
{
  kicker: "L6-03 · 实例矩阵",
  title: "为什么 <span class=\"hl-c\">template-instances/</span> 必须存在：查不到就退化成 f16",
  sub: "没被显式实例化的组合，在编译期就不存在。运行期查表拿到 nullptr 时不是算错，而是慢。",
  caption: "与 L6-04 呼应：那一课讲 template-instances 的全貌（mmq / mmf 也在同一个目录）。这里的 49 个 vec 实例只是其中一族。",
  src: "ggml/src/ggml-cuda/fattn.cu",
  mark: [5, 6, 9, 10, 11, 14, 16, 17],
  lineNo: 496,
  code: `static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \\"%s-%s\\" to GGML_CUDA_FA_QUANTS to compile it.\\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip c">CMake 选择组合</span><span class="arrow">-></span>
        <span class="chip a">编入库的实例</span><span class="arrow">-></span>
        <span class="chip b">运行期查表</span><span class="arrow">-></span>
        <span class="chip e">命中 / 回退</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: 'GGML_CUDA_FA_QUANTS', b: 'CMake 变量决定编译哪些 <br>&lt;type_K&gt;-&lt;type_V&gt; 组合；' +
          '默认是 q4_0-q4_0 / q8_0-q8_0 / f16-f16 / bf16-bf16 四个。', m: 'ggml/cmake/common.cmake' },
      { c: 'a', t: 'GGML_CUDA_FA_*_* = 0/1', b: '每个组合变成一个编译期宏；' +
          'FATTN_VEC_CASE 用 <br>if constexpr 决定要不要实例化。', m: 'fattn.cu:419-426' },
      { c: 'b', t: 'ggml_cuda_get_fattn_vec_case', b: '运行期按 (head_size, type_K, type_V) <br>依次试宏，返回函数指针或 nullptr。', m: 'fattn.cu:436-494' },
      { c: 'e', t: 'nullptr → 换 f16-f16', b: '回退到 f16 实例并打印一行警告，' +
          'K/V 在 launch_fattn 里被转成 f16 —— <br>结果对，速度差。', m: 'fattn.cu:502-511' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这条链解释了 template-instances 的存在理由：<span class="k">C++ 模板要有人写出实例，才会变成机器码</span>。',
      'CMake 只把选中的组合加进源文件列表，并把 <span class="v">GGML_CUDA_FA_Q8_0_Q4_0</span> 之类的宏置 0/1。',
      '宏为 0 时，<span class="v">if constexpr</span> 让那段声明整个消失 —— ' +
      '<span class="k">这个组合在二进制里根本不存在</span>，不是“跑得慢”，是“没有”。',
      '所以运行期必须查表：查到了就用，查不到就退到 <span class="v">f16-f16</span>（永远会被编译）。',
      '警告原文（505-507 行）：<span class="v">no FlashAttention vector kernel compiled for K/V types ... ' +
      'converting K and V to f16 instead (slow)</span>。',
      '这就把“<span class="k">dtype 支持是编译期决定的</span>”这句话落到了实处：' +
      '<span class="v">fattn-vec</span> 支持哪些 dtype，取决于你构建时给了 <span class="v">GGML_CUDA_FA_QUANTS</span> 什么值。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13900, () => { msg.innerHTML = texts[4]; });
    tl.at(17500, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 mask / ALiBi / softcap：三条路线共用的一段准备 */
{
  kicker: "L6-03 · 公共处理",
  title: "mask / ALiBi / softcap：三条路线共用的一段准备",
  sub: "launch_fattn 从 op_params 取出 scale、max_bias、logit_softcap，再决定要不要转 f16、要不要切 stream-k。",
  caption: "mask 必须是 f16：launch_fattn 里有一条 GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16)（同文件 1001 行）。ALiBi 就是这里的 max_bias：m0/m1 是 slope 的底数，内核按 head 取用（如 fattn-vec.cuh:115 的 get_alibi_slope）。",
  src: "ggml/src/ggml-cuda/fattn-common.cuh",
  mark: [0, 4, 5, 6, 8, 9, 15, 16],
  lineNo: 1215,
  code: `    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['修饰', '从哪来', '三条路线怎么用它'],
      [['scale', 'op_params[0]', 'Q·K 的缩放；softcap 非 0 时先除以 logit_softcap（1223-1225 行）'],
       ['max_bias（ALiBi）', 'op_params[1]', '算出 m0 / m1（1230-1231 行），内核按 head 取 slope 加到 KQ 上'],
       ['logit_softcap', 'op_params[2]', '编译期开关 use_logit_softcap：三条路线的 case 都各编两份'],
       ['mask', 'dst->src[3]，必须 f16', '元素级加到 KQ 上；mma 还能把它压实成 sparse 索引（fattn.cu:128-152）'],
       ['sinks / KV_max', 'dst->src[4] / 额外显存', 'sinks 是 attention sink；KV_max 是每行最后一个非 -inf 的 mask 位置（后面整段不用算）']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '三条路线的 launcher 都只是“选 kernel + 定分块”，真正的公共准备全在 launch_fattn。',
      '<span class="v">scale / max_bias / logit_softcap</span> 三个 float 就躺在节点的 op_params 里 —— ' +
      '回顾 L2-06：build_attn() 调 ggml_flash_attn_ext() 时把 kq_scale、f_max_alibi_bias、' +
      'logit_softcap 一起传进来。',
      '<span class="k">softcap 是通过 scale 实现的</span>：先 scale /= logit_softcap，' +
      '内核再做 tanh(x)*logit_softcap。没有 FlashAttention 时，图的非 FA 分支手工发出 scale→tanh→scale 三个节点。',
      '<span class="v">mask 必须转成 f16</span>：不是 CUDA 内核自己去转，而是图上先有一个 F16 的 mask 张量；' +
      'assert 只是把这条契约写死。',
      '<span class="v">KV_max</span>：每行最后一个非 -inf 的 mask 位置之后整段都不用算，' +
      '内核从那里往前收敛循环（<span class="v">flash_attn_mask_to_KV_max</span>，同文件 664-719 行）；' +
      'mma 更进一步，把小 mask 直接压成索引（sparse gather）。',
      '所以“mask 与 alibi 的处理”不是一个 kernel 的细节，' +
      '而是 <span class="k">launch_fattn 这一层与图构建层（L2-06）之间的契约</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3800, () => { rows.forEach((x, k) => { x.className = k <= 2 ? 'on' : ''; }); msg.innerHTML = texts[1]; });
    tl.at(8400, () => { rows.forEach((x, k) => { x.className = k === 2 ? 'on' : ''; }); msg.innerHTML = texts[2]; });
    tl.at(12500, () => { rows.forEach((x, k) => { x.className = k === 3 ? 'on' : ''; }); msg.innerHTML = texts[3]; });
    tl.at(15800, () => { rows.forEach((x, k) => { x.className = k === 4 ? 'on' : ''; }); msg.innerHTML = texts[4]; });
    tl.at(18400, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 10 三条路线一张表 + 下一次看哪里 */
{
  kicker: "L6-03 · 收束",
  title: "三条路线一张表 + 下一次看哪里",
  sub: "路线由 head size 与 dtype 决定；dtype 的支持范围由构建选项决定；具体 shape 组合靠模板实例覆盖。",
  caption: "下一课 L6-04 讲 CUDA 其余算子与 template-instances 的全貌（mmq / mmf 也在那个目录）。",
  src: "ggml/src/ggml-cuda/fattn.cu",
  mark: [1, 2, 3, 4, 5],
  lineNo: 516,
  code: `// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['路线（枚举值）', 'head size', 'K/V dtype', '什么时候被选中'],
      [['VEC (100)', '64 / 128 / 256', 'F16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,BF16（+F32 视作 F16）',
        'can_use_vector_kernel 为真且 batch 很小（解码）'],
       ['TILE (200)', '12 组 (DKQ,DV)，含 40 / 72', '先转成 f16 再用',
        '没有 Tensor Core，或 Volta 上中等 batch；兜底路线'],
       ['MMA_F16 (400)', '64…576（不含 40 / 72）', '先转成 f16 再用（sparse 只有 4 组）',
        '有 Tensor Core 且 batch 够大（prefill）']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '解码场景：<span class="mono">Q-&gt;ne[0] = 128</span>、<span class="mono">Q-&gt;ne[1] = 1</span>，' +
      'K/V 类型都是 <span class="mono">Q8_0</span>，GPU 是 Ada。走哪条路线？为什么？',
      '走 <span class="mono">BEST_FATTN_KERNEL_VEC</span>（fattn.cu:650-653）。<br>' +
      '<span class="mono">can_use_vector_kernel</span>（635 行）要求 Q-&gt;ne[0] ≤ 256 且 % 64 == 0 且 ≠ 192 ' +
      '且 K-&gt;ne[1] % FATTN_KQ_STRIDE == 0 → 128 满足。<br>' +
      'Ada 走 <span class="mono">turing_mma_available</span> 分支；K/V 是量化类型，' +
      '且 cc ≥ GGML_CUDA_CC_ADA_LOVELACE、Q-&gt;ne[1] ≤ 2 → 返回 VEC。<br>' +
      '但前提是编译期把 <span class="mono">q8_0-q8_0</span> 放进 GGML_CUDA_FA_QUANTS；' +
      '否则 ggml_cuda_get_fattn_vec_case() 返回 nullptr，退化成 f16-f16 并打印警告（502-511 行）。'));

    wrap.querySelector('#ex').appendChild(W.exercise(
      '预填充场景：<span class="mono">Q-&gt;ne[0] = 576</span>、<span class="mono">V-&gt;ne[0] = 512</span>，' +
      'GPU 有 Tensor Core，batch 很大。走哪条路线？',
      '走 <span class="mono">BEST_FATTN_KERNEL_MMA_F16</span>（fattn.cu:664）。<br>' +
      '576 能过白名单，但要满足 <span class="mono">V-&gt;ne[0] == 512</span> 且 gqa_opt_applies（613-620 行）；' +
      '<span class="mono">can_use_vector_kernel</span> 为 false（576 &gt; 256），于是 Turing 分支最后一行返回 MMA_F16。<br>' +
      'mma 只吃 f16：<span class="mono">launch_fattn&lt;DV, ncols1, ncols2&gt;(..., true, true, true)</span> ' +
      '（fattn-mma-f16.cuh:2116-2117），K/V 先被转成 f16（fattn-common.cuh:1026-1052）。<br>' +
      '实例键是 (576, 512, ncols1, ncols2)，只有 ncols2 ∈ {4, 16, 32} 的组合被生成。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '把三条路线并排看：它们的差别就是 <span class="k">head size 覆盖</span>、' +
      '<span class="k">dtype 处理方式</span>、<span class="k">被选中的形状</span>三件事。',
      '<span class="v">VEC</span>：唯一会把量化 K/V 直接喂进内核的路线（KQ 点积支持 7 种类型）。',
      '<span class="v">TILE</span>：形状覆盖最全（40 / 72 只有它有），但 K/V 先转 f16。',
      '<span class="v">MMA_F16</span>：batch 大时最快；实例矩阵的维度最多（DKQ × DV × ncols1 × ncols2）。',
      '下一课 L6-04：CUDA 其余算子，以及 <span class="k">template-instances 目录为什么必须存在</span> —— ' +
      '本课第 8 幕讲的机制，在那一课会看到全貌。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 3000, () => {
      rows.forEach((x, k) => { x.className = k === i ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(19500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
