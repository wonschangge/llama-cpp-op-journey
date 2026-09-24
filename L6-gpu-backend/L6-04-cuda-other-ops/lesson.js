/* ==========================================================================
   L6-04 · CUDA 其余算子与模板实例化
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 本课的地图：<span class="hl-a">280</span> 个文件，两类来源 */
{
  kicker: "L6-04 · 全局",
  title: "本课的地图：<span class=\"hl-a\">280</span> 个文件，两类来源",
  sub: "ggml/src/ggml-cuda/ 下的文件分成两类：人写的算子实现，和脚本生成的模板实例。",
  caption: "计数命令与逐族清单见 source.md 第一节与第十一节；数字全部来自命令输出。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [30, 31, 38, 39, 44],
  lineNo: 5,
  code: `#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row" id="stats" style="gap:9px"></div>
      <div class="row wrap" id="ops" style="gap:6px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const N = [280, 160, 120];

    const stats = [
      { n: String(N[0]), t: '本课清单文件数', s: N[1] + ' 手写 + ' + N[2] + ' 生成', c: 'a' },
      { n: String(N[1]), t: '手写 .cu / .cuh / .h', s: '68 .cu + 89 .cuh + 3 vendors/*.h', c: 'b' },
      { n: String(N[2]), t: 'template-instances/*.cu', s: '全部由脚本生成', c: 'c' }
    ];
    const host = wrap.querySelector('#stats');
    const els = stats.map(d => {
      const e = U.el('div', { class: 'card', style: 'width:218px' });
      e.style.borderLeftColor = 'var(--' + d.c + ')';
      e.innerHTML = '<div class="ct" style="font-size:22px;color:var(--' + d.c + ')">' + d.n + '</div>' +
        '<div class="cb"><b>' + d.t + '</b><br>' + d.s + '</div>';
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.35');

    const opHost = wrap.querySelector('#ops');
    ['norm', 'rope', 'softmax', 'ssm-scan', 'top-k', 'argsort', 'quantize',
     'moe-weighted-reduction', 'wkv', 'gla', 'conv2d', 'cumsum'].forEach(n => {
      opHost.appendChild(U.chip(n, 'b'));
    });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '右边是 `ggml-cuda.cu` 的 include 清单（第 5-49 行）：<b>一个 .cuh 对应一组算子</b>。',
      '本课清单 = <span class="v">' + N[0] + '</span> 个文件：<span class="k">' + N[1] +
        '</span> 个人写、<span class="k">' + N[2] + '</span> 个生成。生成的那部分全在 ' +
        '<span class="v">template-instances/</span>。',
      '手写的那些里，绝大多数是一对 <span class="v">xxx.cu</span> + <span class="v">xxx.cuh</span>：' +
        '前者是 kernel 与入口，后者是声明。',
      '下面按"一个算子一幕"走：先看它们怎么被分派，再看 6 个代表性算子的 kernel 结构，' +
        '最后回答 template-instances 为什么必须存在。'
    ];
    tl.at(600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; });
    tl.at(4200, () => { els[1].style.opacity = '1'; U.markLines(document, [0, 1, 2, 3, 4]); msg.innerHTML = texts[1]; });
    tl.at(8000, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[2]; });
    tl.at(11800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 2 和 L6-01 同一个 <span class="hl-a">switch</span>：每个 op 一个入口 */
{
  kicker: "L6-04 · 分派",
  title: "和 L6-01 同一个 <span class=\"hl-a\">switch</span>：每个 op 一个入口",
  sub: "非矩阵乘的算子没有特殊通道 —— 它们和 MUL_MAT 挤在同一张 dst->op 分派表里。",
  caption: "回顾 L6-01：ggml_cuda_compute_forward() 是 CUDA 后端唯一的算子入口。",
  src: "ggml/src/ggml-cuda/ggml-cuda.cu",
  mark: [0, 3, 6, 9],
  lineNo: 2355,
  code: `        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['上游行号', 'case（GGML_OP_*）', '入口函数', '本课哪一幕'],
      [['2355', 'SSM_CONV', 'ggml_cuda_op_ssm_conv', 'L2-11 的图原语'],
       ['2358', 'SSM_SCAN', 'ggml_cuda_op_ssm_scan', '第 6 幕'],
       ['2361', 'TOP_K', 'ggml_cuda_op_top_k', '第 7 幕'],
       ['2364', 'ARGSORT', 'ggml_cuda_op_argsort', '第 7 幕'],
       ['2301', 'SOFT_MAX', 'ggml_cuda_op_soft_max', '第 4 幕'],
       ['2307', 'ROPE', 'ggml_cuda_op_rope', '第 5 幕'],
       ['2253', 'RMS_NORM', 'ggml_cuda_op_rms_norm', '第 3 幕']],
      { monoCols: [0, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '这 12 行是同一个 switch 里的一段：<span class="k">case -> 一个 ggml_cuda_op_* 入口</span>。',
      '入口函数的命名是统一的：<span class="v">ggml_cuda_op_</span> 加上算子名的小写形式。',
      '<span class="v">SSM_SCAN</span> / <span class="v">TOP_K</span> / <span class="v">ARGSORT</span> ' +
        '都是这一课的内容。<span class="k">"多 kernel 选一"就发生在入口函数内部</span>：' +
        '按形状与类型挑具体 kernel。',
      '表中其它行号（2301 SOFT_MAX / 2307 ROPE / 2253 RMS_NORM）见 source.md 第二节的逐字引用。',
      '跨课呼应：<span class="k">L6-02（mmq/mmvq/mmf）与 L6-03（fattn）也是这张表上的 case</span>，' +
        '它们内部又各有"多 kernel 选一"的判据 —— 本课只看非矩阵乘的。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2000 + i * 700, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[1];
    }));
    tl.at(8000, () => { msg.innerHTML = texts[2]; });
    tl.at(11000, () => { msg.innerHTML = texts[3]; });
    tl.at(13000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 norm.cu：<span class="hl-a">一个 block 管一行</span>，规约在 block 内闭合 */
{
  kicker: "L6-04 · norm.cu",
  title: "norm.cu：<span class=\"hl-a\">一个 block 管一行</span>，规约在 block 内闭合",
  sub: "RMSNorm 在 GPU 上是一个\"行内规约\"问题：一行的平方和算完，才能写回这一行。",
  caption: "同一个 kernel 还能顺手融合 mul（和 add）—— 靠的是编译期模板参数，不是运行期分支。",
  src: "ggml/src/ggml-cuda/norm.cu",
  mark: [0, 2, 6, 7, 9, 10, 13, 16, 20, 21],
  lineNo: 131,
  code: `    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const left = wrap.querySelector('#left');
    const right = wrap.querySelector('#right');

    left.innerHTML = '<div class="cm" style="margin:0 0 4px">四步，全在一个 block 内</div>';
    const steps = [
      { c: 'a', t: '① 平方和', b: '每个线程按 <span class="cm" style="margin:0">col += block_size</span> 跨步扫自己那几列，累加 xi*xi' },
      { c: 'b', t: '② block 内规约', b: '<span class="cm" style="margin:0">block_reduce&lt;SUM&gt;(tmp, s_sum)</span>：共享内存 + warp 规约' },
      { c: 'c', t: '③ 求 scale', b: '<span class="cm" style="margin:0">rsqrtf(mean + eps)</span>，其中 mean = tmp / ncols' },
      { c: 'd', t: '④ 再扫一遍写回', b: '按编译期分支决定是否顺带乘 mul / 加 add' }
    ];
    const els = steps.map(s => { const e = U.card(s, { style: 'width:100%' }); left.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.3');

    right.innerHTML =
      '<div class="card" style="border-left-color:var(--e)">' +
      '<div class="ct" style="color:var(--e)">do_multiply / do_add 是模板参数</div>' +
      '<div class="cb">kernel 模板头是 <span class="cm" style="margin:0">template &lt;int block_size, ' +
      'bool do_multiply = false, bool do_add = false&gt;</span>（见 source.md 第三节）。<br>' +
      '所以 <span class="cm" style="margin:0">if constexpr</span> 的三个分支在编译期就只剩一个 —— ' +
      '<b>融合不是运行期判断</b>。</div></div>' +
      '<div class="card" style="border-left-color:var(--f)">' +
      '<div class="ct" style="color:var(--f)">谁决定 grid？</div>' +
      '<div class="cb"><span class="cm" style="margin:0">blocks_num(nrows, nchannels, nsamples)</span>：' +
      'grid 的三个维度就是张量的三个外维 —— 一行一个 block，不跨行规约。' +
      '线程数按 ncols 是否小于 1024 在 256 / 1024 之间选（见 source.md 第三节）。</div></div>' +
      '<div class="card" style="border-left-color:var(--g)">' +
      '<div class="ct" style="color:var(--g)">跨课呼应</div>' +
      '<div class="cb">L5-01 的 CPU 版 rms_norm 是同样的"两遍扫描"，差别只在规约用 SIMD 而不是 ' +
      'block_reduce。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'RMSNorm = 在一行里做一次平方和规约，再按 <span class="v">scale</span> 缩放写回。',
      '<span class="v">for (col = tid; col &lt; ncols; col += block_size)</span>：' +
      '线程按块大小跨步 —— 这是所有"行内规约"kernel 的写法。',
      '<span class="v">block_reduce&lt;SUM&gt;</span> 把每线程的部分和压成一个数；' +
      '之后 <span class="v">scale = rsqrtf(mean + eps)</span>。',
      '<span class="v">if constexpr</span> 三个分支：只乘 mul / 乘 mul 再加 add / 什么都不做。' +
      'L2 的稠密模型里 <span class="cm" style="margin:0">RMS_NORM + MUL</span> 就是这么合并的。',
      '一句话：<span class="k">运行期的形状决定 grid，编译期的开关决定融合</span>。'
    ];
    let t = 700;
    els.forEach((e, i) => { tl.at(t, () => {
      els.forEach((x, k) => x.style.opacity = k <= i ? '1' : '.3');
      msg.innerHTML = texts[i];
    }); t += 3200; });
    tl.at(t, () => { msg.innerHTML = texts[4]; U.markLines(document, [13, 14, 15, 16, 17, 20, 21]); });
  }
},

/* ------------------------------------------------------ 4 softmax.cu：把 <span class="hl-c">ncols</span> 折进编译期，但不建目录 */
{
  kicker: "L6-04 · softmax.cu",
  title: "softmax.cu：把 <span class=\"hl-c\">ncols</span> 折进编译期，但不建目录",
  sub: "同一份 kernel 源码，对若干常见列数做编译期特化；选择靠一条折叠表达式。",
  caption: "调用点写的是一串常量：launch_soft_max_kernels<32, 64, ..., 4096>（见 source.md 第四节）。",
  src: "ggml/src/ggml-cuda/softmax.cu",
  mark: [0, 7, 8, 9, 11, 21, 25, 26],
  lineNo: 278,
  code: `template<int... Ns, typename T>
static void launch_soft_max_kernels(const float * x, const T * mask, const float * sinks, float * dst,
                             const soft_max_params & p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, ncols, block, T>), smpbo);
            soft_max_f32<true, ncols, block><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, mask, sinks, dst, p);
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    //default case
    CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, 0, 0, T>), smpbo);`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 4px">调用点给的候选（source.md 第四节）</div>';
    const cand = [32, 64, 128, 256, 512, 1024, 2048, 4096];
    const cels = cand.map(n => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
      e.innerHTML = '<span class="m">ncols == ' + n + '</span> -> <span style="color:var(--c)">soft_max_f32&lt;true, ' + n + ', block&gt;</span>';
      left.appendChild(e);
      return e;
    });
    const dflt = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
    dflt.innerHTML = '<span class="m">其它列数</span> -> <span style="color:var(--d)">soft_max_f32&lt;true, 0, 0&gt;</span>';
    left.appendChild(dflt);
    const all = cels.concat([dflt]);
    all.forEach(e => e.style.opacity = '.28');

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">选择机制：一条折叠表达式</div>' +
      '<div class="cb">每个候选值包成 <span class="cm" style="margin:0">std::integral_constant&lt;int, Ns&gt;</span>，' +
      '由 <span class="cm" style="margin:0">(launch_kernel(...) || ...)</span> 折叠依次试；' +
      '命中就 <span class="cm" style="margin:0">return</span>。</div></div>' +
      '<div class="card" style="border-left-color:var(--d)">' +
      '<div class="ct" style="color:var(--d)">兜底：ncols_template = 0</div>' +
      '<div class="cb">没命中就用 <span class="cm" style="margin:0">soft_max_f32&lt;true, 0, 0&gt;</span>：' +
      '源码注释写了，此时循环边界未知、不能展开。</div></div>' +
      '<div class="card" style="border-left-color:var(--e)">' +
      '<div class="ct" style="color:var(--e)">和 template-instances/ 的差别</div>' +
      '<div class="cb">这里 8 个特化<b>全在同一个 .cu 里实例化</b>；' +
      'mmq 的 22 个类型则是<b>一个类型一个文件</b>（第 9 幕）。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'softmax 的列数是个热点形状：常见值就那几个。',
      '于是源码把它们做成<span class="k">编译期常量</span>：ncols_template 一确定，' +
      '循环就能展开、共享内存大小也能定死。',
      '选择发生在<b>一次折叠</b>里：候选依次尝试，命中即返回。' +
      '这是"多 kernel 选一"的第三种写法（前两种在 L6-02 / L6-03）。',
      '都没命中 -> <span class="v">&lt;true, 0, 0&gt;</span> 通用版兜底。' +
      '<span class="k">特化是加速，兜底是正确性。</span>',
      '注意：这个文件<b>没有</b>对应的 template-instances 目录 —— 因为实例化量小，' +
      '放在同一个 TU 里编译得动。第 9 幕回答"什么时候必须拆文件"。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [8, 9]); });
    cels.forEach((e, i) => tl.at(6200 + i * 900, () => {
      all.forEach((x, k) => x.style.opacity = k === i ? '1' : '.28');
      msg.innerHTML = texts[2];
    }));
    tl.at(14200, () => { all.forEach(x => x.style.opacity = '.28'); dflt.style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(16200, () => { all.forEach(x => x.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 rope.cu：一个 <span class="hl-b">mode</span> 整数，四种成对方式 */
{
  kicker: "L6-04 · rope.cu",
  title: "rope.cu：一个 <span class=\"hl-b\">mode</span> 整数，四种成对方式",
  sub: "RoPE 的变体不是四份图，而是 op_params 里 mode 的几个位 —— 但落地时确实是四个 kernel。",
  caption: "每个 kernel 再按 src0/dst 的 dtype 组合分派（F32/F32、F32/F16、F16/F16），见 source.md 第五节。",
  src: "ggml/src/ggml-cuda/rope.cu",
  mark: [0, 1, 2, 3, 9, 10],
  lineNo: 606,
  code: `    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;
`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'rope_norm', b: '交错式（GPT-J 风格）：<br>相邻两维成对旋转', m: '四路里的 else 分支' },
      { c: 'b', t: 'rope_neox', b: '半区式（GPT-NeoX 风格）：<br>前后半区对应维成对', m: 'mode & GGML_ROPE_TYPE_NEOX' },
      { c: 'c', t: 'rope_multi', b: 'M-RoPE：三个 section<br>各用不同的位置增量', m: 'mode & GGML_ROPE_TYPE_MROPE' },
      { c: 'd', t: 'rope_vision', b: '视觉塔：旋转对跨越整行，<br>不支持 offset', m: 'mode == GGML_ROPE_TYPE_VISION' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:163px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.3');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '四个 bool 从 <span class="v">mode</span> 里解出来 —— mode 来自 ' +
        '<span class="v">dst->op_params[2]</span>（L1-02 讲的那 16 个 int32）。',
      '<span class="v">is_neox</span>：是不是 NeoX 的成对方式；' +
        '<span class="v">is_mrope</span> / <span class="v">is_imrope</span>：多模态的 section 式位置。',
      '<span class="v">is_vision</span> 带两条例外断言：<span class="k">n_dims 必须是半个头</span>、' +
        '<span class="k">不支持 offset</span> —— 因为它的旋转对跨整个行。',
      '接着是 <span class="v">if (is_neox) ... else if (is_mrope &amp;&amp; !is_vision) ... ' +
        'else if (is_vision) ... else ...</span> 四路，每路再按 dtype 三选一。',
      '跨课呼应：<span class="k">同一个 GGML_OP_ROPE，在图上是同一个节点</span>，' +
        '差别全在 op_params。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.3'; });
      msg.innerHTML = texts[Math.min(i, 2)];
    }));
    tl.at(12400, () => { msg.innerHTML = texts[3]; U.markLines(document, [0, 1, 2, 3]); });
    tl.at(14500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 ssm-scan.cu：57 行 switch —— <span class="hl-a">手写的实例表</span> */
{
  kicker: "L6-04 · ssm-scan.cu",
  title: "ssm-scan.cu：57 行 switch —— <span class=\"hl-a\">手写的实例表</span>",
  sub: "同一个 kernel 模板，按 token 数 1..8 各实例化一份（循环可完全展开），其它走通用版。",
  caption: "这就是 template-instances/ 想自动化掉的东西 —— 只不过这里组合少，源码选择了手写。",
  src: "ggml/src/ggml-cuda/ssm-scan.cu",
  mark: [0, 2, 3, 44, 50, 51],
  lineNo: 282,
  code: `            switch (n_tok)
            {
            case 1:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 1>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 2:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 2>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 3:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 3>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 4:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 4>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 5:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 5>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 6:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 6>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 7:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 7>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 8:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 8>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            default:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 0>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="lane" style="gap:6px"></div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const lane = wrap.querySelector('#lane');
    const chips = ['1', '2', '3', '4', '5', '6', '7', '8', '其它'].map((n, i) => {
      const c = U.chip('n_tok = ' + n, i < 8 ? 'a' : 'd');
      lane.appendChild(c); return c;
    });

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 4px">模板参数（source.md 第六节）</div>' +
      '<div class="card" style="border-left-color:var(--b)"><div class="cb">' +
      '<span class="cm" style="margin:0">template &lt;size_t splitD, size_t N, size_t L_template&gt;</span><br>' +
      '<b>L_template</b> = 序列长度：1..8 时循环长度已知，编译器可展开；<br>' +
      '<b>L_template = 0</b> 时长度走运行期参数 <span class="cm" style="margin:0">L_param</span>。' +
      '</div></div>';
    const right = wrap.querySelector('#right');
    right.innerHTML = '<div class="cm" style="margin:0 0 4px">同一文件里的第二条路</div>' +
      '<div class="card" style="border-left-color:var(--e)"><div class="cb">' +
      '长序列且是 Mamba-2（<span class="cm" style="margin:0">n_tok &gt; SSM_SSD_MIN_TOKENS</span>，' +
      '阈值 128）时改走 <b>SSD</b> 路径：把 scan 变成 matmul，还要卡设备能力 ' +
      '<span class="cm" style="margin:0">cc &gt;= GGML_CUDA_CC_TURING</span>。' +
      '</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'SSM 的 scan 是<b>顺序依赖</b>的：第 t 步要第 t-1 步的 state。',
      '源码的应对：把 <span class="k">序列长度做进模板</span>。' +
        '<span class="v">ssm_scan_f32&lt;threads, 16, 1&gt;</span> 到 <span class="v">&lt;threads, 16, 8&gt;</span> 各一份。',
      '<span class="v">case 1..8</span> 分别调用这 8 份；<span class="v">default</span> 用 ' +
        '<span class="v">L_template = 0</span> 的通用版。',
      '这就是<b>手写的实例表</b>：一个模板参数取值 -> 一行 case。' +
        '<span class="k">第 9 幕的 template-instances/ 是同一件事，只是组合太多、写不下，改用脚本生成。</span>',
      '同时它还有第二条路：<span class="v">use_ssd</span> 成立时整个 scan 变成 matmul ' +
        '—— 和 L6-02 的主题接上。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    chips.forEach((c, i) => tl.at(3000 + i * 650, () => c.style.opacity = (i < 8 ? '1' : '.35')));
    tl.at(9200, () => { msg.innerHTML = texts[1]; U.markLines(document, [2, 3]); });
    tl.at(11500, () => { msg.innerHTML = texts[2]; U.markLines(document, [3, 44, 50]); chips.forEach(c => c.style.opacity = '1'); });
    tl.at(14200, () => { msg.innerHTML = texts[3]; });
    tl.at(16200, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 top-k.cu：一条 <span class="hl-c">多段式</span> 的 kernel 流水线 */
{
  kicker: "L6-04 · top-k.cu",
  title: "top-k.cu：一条 <span class=\"hl-c\">多段式</span> 的 kernel 流水线",
  sub: "排序类算子在 GPU 上常常不是\"一个 kernel\"，而是几个小 kernel 依次跑，中间结果放显存。",
  caption: "同一个文件里还有两条备选路径（CUB / 按位排序），由编译期宏决定，见 source.md 第七节。",
  src: "ggml/src/ggml-cuda/top-k.cu",
  mark: [0, 3, 4, 5, 6, 13, 15, 16, 17, 19, 20, 25, 26],
  lineNo: 180,
  code: `static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="flow" style="gap:5px"></div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const flow = wrap.querySelector('#flow');
    const stages = ['init', 'histogram', 'select', 'x4 轮', 'reset', 'gather'];
    const chips = stages.map((s, i) => {
      const c = U.chip(s, i === 3 ? 'd' : 'c');
      flow.appendChild(c);
      if (i < stages.length - 1) flow.appendChild(U.arrow('->'));
      return c;
    });

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 4px">基数选择（radix select）</div>' +
      '<div class="card" style="border-left-color:var(--a)"><div class="cb">' +
      '<span class="cm" style="margin:0">RADIX_BITS = 8</span>：每轮看 8 个 bit；' +
      'float 是 32 bit，所以 <span class="cm" style="margin:0">32 / 8 = 4</span> 轮。<br>' +
      '每轮：先按这一段 bit 统计直方图（<b>histogram</b>），再决定第 k 大落在哪个桶（<b>select</b>）。' +
      '</div></div>' +
      '<div class="card" style="border-left-color:var(--b)"><div class="cb">' +
      '中间状态 <span class="cm" style="margin:0">top_k_radix_state</span> 与直方图都从 ' +
      '<span class="cm" style="margin:0">ctx.pool()</span> 分配 —— 复用后端的内存池。' +
      '</div></div>';
    const right = wrap.querySelector('#right');
    right.innerHTML = '<div class="cm" style="margin:0 0 4px">为什么不是一个 kernel？</div>' +
      '<div class="card" style="border-left-color:var(--e)"><div class="cb">' +
      '每一轮都要<b>全局同步</b>：直方图必须全部写完，select 才能读。' +
      '拆成多个 kernel、放进同一条 stream，用 stream 的顺序性换同步 —— ' +
      '这是 GPU 上做迭代算法的常见办法。' +
      '</div></div>' +
      '<div class="card" style="border-left-color:var(--f)"><div class="cb">' +
      '最后 <span class="cm" style="margin:0">top_k_radix_gather</span> 把选中的 k 个下标写进 dst。' +
      '</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'top-k 的第一步是<b>找第 k 大的值</b>，第二步才是收集下标。',
      '<span class="v">RADIX_BITS = 8</span>：一次处理 8 个 bit，32 位浮点要 <span class="k">4 轮</span>。',
      '每轮两段：<span class="v">histogram</span>（多 block 统计）-> <span class="v">select</span>（单 block 决策）。',
      '轮次之间需要全局可见性，所以只能是<b>多个 kernel 顺序启动</b>，同一 stream 保证次序。',
      '一句话：<span class="k">GPU 上的"排序算子"通常是一串 kernel，而不是一个大 kernel</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3400, () => { chips[3].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [3, 4, 15]); });
    tl.at(7000, () => { msg.innerHTML = texts[2]; U.markLines(document, [15, 16, 17, 18, 19]); });
    tl.at(11000, () => { msg.innerHTML = texts[3]; U.markLines(document, [0, 1, 2, 3, 4]); });
    tl.at(15000, () => { msg.innerHTML = texts[4]; chips.forEach(c => c.style.opacity = '1'); });
  }
},

/* ------------------------------------------------------ 8 ★ <span class="hl-a">不是每个 .cu 都是一个 op</span>：moe 归约是个融合 */
{
  kicker: "L6-04 · 洞察",
  title: "★ <span class=\"hl-a\">不是每个 .cu 都是一个 op</span>：moe 归约是个融合",
  sub: "moe-weighted-reduction.cu 里没有 GGML_OP —— 它由图级模式匹配触发，跳过一整段子图。",
  caption: "匹配规则在 ggml-cuda.cu 的 ggml_cuda_try_fuse() 里（source.md 第九节逐字引用）。",
  src: "ggml/src/ggml-cuda/moe-weighted-reduction.cu",
  mark: [0, 6, 7, 12, 13, 14, 16, 19, 21, 24, 32, 33],
  lineNo: 3,
  code: `static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="roles" style="gap:8px"></div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '① op 实现', b: '被 compute_forward 的 switch 调<br>norm / rope / softmax / ssm-scan / top-k', m: 'ggml_cuda_op_*' },
      { c: 'd', t: '② 别人的助手', b: '不被 switch 调，被别的 kernel 调<br>quantize.cu 给 mmq / mmvq 准备 Q8_1 输入', m: 'quantize_mmq_q8_1_cuda' },
      { c: 'e', t: '③ 图级融合', b: '由模式匹配触发，吃掉一整段子图<br>moe-weighted-reduction.cu 就是这一类', m: 'ggml_cuda_try_fuse' }
    ];
    const host = wrap.querySelector('#roles');
    const els = defs.map(d => U.card(d, { style: 'width:218px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.3');

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 4px">kernel 在算什么</div>' +
      '<div class="card" style="border-left-color:var(--e)"><div class="cb">' +
      '一个线程负责<b>一个输出列</b>（<span class="cm" style="margin:0">col = blockIdx.y*blockDim.x + threadIdx.x</span>），' +
      '然后<b>串行累加 n_expert_used 个专家</b>。' +
      '</div></div>';
    const right = wrap.querySelector('#right');
    right.innerHTML = '<div class="cm" style="margin:0 0 4px">融合掉了什么</div>' +
      '<div class="card" style="border-left-color:var(--a)"><div class="cb">' +
      '长形式是 <b>2k+1</b> 个节点（k = 专家数）；源码注释写明它从 ' +
      '<span class="cm" style="margin:0">GGML_OP_MUL</span> 起头，上限 ' +
      '<span class="cm" style="margin:0">MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15</span>。' +
      '</div></div>' +
      '<div class="card" style="border-left-color:var(--b)"><div class="cb">' +
      '融合成功则一次 kernel 调用替代 2k+1 次节点计算 —— ' +
      '<b>省的是 kernel 启动与中间张量的读写</b>。' +
      '</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把 ggml-cuda/ 里的 .cu 按"谁调它"分成三类。',
      '<span class="v">ggml_cuda_op_moe_weighted_reduction</span> 的入口收的是 experts / weights ' +
        '两个张量，不是图节点 —— 它没有 <span class="v">dst->op</span> 可查。',
      '在 <span class="v">compute_forward</span> 里找不到 ' +
        '<span class="v">GGML_OP_MOE_WEIGHTED</span>：<span class="k">它不是算子，是融合规则</span>。',
      '它由 <span class="v">ggml_cuda_try_fuse()</span> 在遍历图时匹配：' +
        '<span class="v">GGML_OP_MUL</span> 起头 + 形状判据（weights 的 ne[0] == 1 且与 MUL 同形）。',
      '所以"读 CUDA 后端"不能只读分派表：<span class="k">还有一条图级的路径</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    els.forEach((e, i) => tl.at(3600 + i * 3200, () => {
      els.forEach((x, k) => x.style.opacity = k === i ? '1' : '.3');
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(16800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 ★ <span class="hl-a">template-instances/</span>：为什么这个目录必须存在 */
{
  kicker: "L6-04 · 洞察",
  title: "★ <span class=\"hl-a\">template-instances/</span>：为什么这个目录必须存在",
  sub: "一个实例文件只有 5 行：一句\"脚本生成\"，一个 include，一行不带 extern 的宏。",
  caption: "头文件里是带 extern 的同名宏 —— 声明与定义分家，正是显式实例化的标准写法。",
  src: "ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu",
  mark: [0, 2, 4],
  lineNo: 1,
  code: `// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_Q4_0);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const TI = [22, 16, 49, 12, 21];

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 4px">头文件那一半（mmq.cuh，见第十节）</div>' +
      '<div class="card" style="border-left-color:var(--a)"><div class="cb">' +
      '<span class="cm" style="margin:0">#define DECL_MMQ_CASE(type) template void mul_mat_q_case&lt;type&gt;(...)</span><br>' +
      '<span class="cm" style="margin:0">extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);</span> ...<br>' +
      '带 <b>extern</b> = <b>显式实例化声明</b>：告诉每个包含 mmq.cuh 的 TU' +
      '"别在这儿实例化，符号在别处"。' +
      '</div></div>';
    const right = wrap.querySelector('#right');
    right.innerHTML = '<div class="cm" style="margin:0 0 4px">实例文件这一半（左边代码）</div>' +
      '<div class="card" style="border-left-color:var(--c)"><div class="cb">' +
      '<span class="cm" style="margin:0">#include "../mmq.cuh"</span><br>' +
      '<span class="cm" style="margin:0">DECL_MMQ_CASE(GGML_TYPE_Q4_0);</span><br>' +
      '同一个宏，<b>不带 extern</b> = <b>显式实例化定义</b>：这个 TU 负责生成 ' +
      '<span class="cm" style="margin:0">mul_mat_q_case&lt;GGML_TYPE_Q4_0&gt;</span> 的代码。' +
      '</div></div>' +
      '<div class="card" style="border-left-color:var(--d)"><div class="cb">' +
      '删掉目录会怎样：这个名字<b>没有任何 TU 提供定义</b>，而 extern 声明又抑制了隐式实例化 ' +
      '—— 链接期 <span class="cm" style="margin:0">undefined reference</span>。' +
      '</div></div>';

    const t = U.table(
      ['族（前缀）', '文件数', '每个文件实例化什么', '模板参数空间'],
      [['mmq-instance-*', String(TI[0]), 'DECL_MMQ_CASE(一个量化类型)', '22 个量化类型'],
       ['mmf-instance-ncols_*', String(TI[1]), 'DECL_MMF_CASE(n)', 'n = 1..16'],
       ['fattn-vec-instance-*', String(TI[2]), '3 行 DECL_FATTN_VEC_CASE(64/128/256)', '7 个 type_K x 7 个 type_V'],
       ['fattn-tile-instance-*', String(TI[3]), '一行 DECL_FATTN_TILE_CASE(DKQ, DV)', '12 组 (DKQ, DV)'],
       ['fattn-mma-f16-instance-*', String(TI[4]), '2-8 行 DECL_FATTN_MMA_F16_CASE(...)', '21 组 (ncols1, ncols2)']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这个文件一共 5 行：一句"脚本生成、别手改"，一个 include，一行宏。' +
      '<span class="k">它的全部内容就是"我要实例化这一种组合"。</span>',
      '因为 CUDA 模板（含 <span class="v">__global__</span> 模板）只有在被实例化时才生成设备代码，' +
      '而实例化必须发生在某个 TU 里。',
      'mmq.cuh 用 <b>extern 声明</b>把 22 个类型全部"挂起来"，' +
      '于是定义只能来自那 22 个文件 —— 一个文件一个类型。',
      '<span class="k">模板参数空间有多大，文件就有多少</span>：' +
      'fattn 的 82 个文件 = head size x ncols x dtype 的组合数（L6-03）。',
      '一句话：<span class="k">template-instances/ 是"显式实例化的落地点"，不是"重复代码"</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 4]); });
    tl.at(4600, () => { msg.innerHTML = texts[1]; });
    tl.at(8600, () => { msg.innerHTML = texts[2]; });
    rows.forEach((r, i) => tl.at(11000 + i * 1300, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
    }));
    tl.at(17800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
    tl.at(19200, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表 */
{
  kicker: "L6-04 · 收束",
  title: "把这一课压成一张表",
  sub: "六个算子、三类角色、一个目录。",
  caption: "下一课 L6-05 换一个后端：SYCL 如何用同一套算子结构重写一遍。",
  src: "ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu",
  mark: [0, 4, 5, 6],
  lineNo: 1,
  code: `// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['文件', '角色', '这一课看到的关键点'],
      [['ggml-cuda.cu', '分派', 'dst->op 一个大 switch，每个 case 一个 ggml_cuda_op_* 入口'],
       ['norm.cu', 'op 实现', '一行一个 block + block_reduce<SUM>；do_multiply/do_add 是模板开关'],
       ['softmax.cu', 'op 实现', 'integral_constant 折叠挑编译期 ncols；<true,0,0> 兜底'],
       ['rope.cu', 'op 实现', 'mode 的四个位 -> 四种成对方式，再按 dtype 分派'],
       ['ssm-scan.cu', 'op 实现', '57 行 switch = 手写实例表；长序列改走 SSD matmul'],
       ['top-k.cu', 'op 实现', '基数选择拆成多个 kernel，靠 stream 顺序做同步'],
       ['quantize.cu', '助手', '不是 op：给 mmq / mmvq 准备 Q8_1 输入'],
       ['moe-weighted-reduction.cu', '图级融合', '由 ggml_cuda_try_fuse 匹配子图，不在 switch 里'],
       ['template-instances/', '生成', '显式实例化定义，一类组合一个 TU']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '为什么 <span class="mono">ggml/src/ggml-cuda/template-instances/</span> 这个目录必须存在？' +
      '把其中 <span class="mono">mmq-instance-q4_0.cu</span> 删掉会发生什么？',
      '因为"只在头文件里写模板"在这里行不通：<br>' +
      '1. <span class="mono">mmq.cuh</span> 对 22 个量化类型写的是<b>带 extern</b> 的 ' +
      '<span class="mono">DECL_MMQ_CASE(...)</span>（显式实例化<b>声明</b>，mmq.cuh:1571-1580）；<br>' +
      '2. extern 声明会<b>抑制</b>包含它的 TU 里的隐式实例化 —— 符号必须由别处提供；<br>' +
      '3. 提供符号的就是 <span class="mono">template-instances/</span> 里那 22 个文件：' +
      '每个文件一行<b>不带 extern</b> 的同一个宏（显式实例化<b>定义</b>）。<br>' +
      '所以删掉 <span class="mono">mmq-instance-q4_0.cu</span>，' +
      '<span class="mono">mul_mat_q_case&lt;GGML_TYPE_Q4_0&gt;</span> 就没有任何 TU 提供定义，' +
      '链接期报 undefined reference。<br>' +
      '更大的意义：文件数 = 模板参数空间的组合数（mmq 22 / mmf 16 / fattn 82，共 120）。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '最后回到验收点：<span class="k">template-instances/ 是显式实例化的落地点</span>。',
      '六个算子的共同点：<span class="v">运行期形状 -> 选 kernel / 定 grid</span>，' +
        '<span class="v">编译期参数 -> 定能不能展开、要不要融合</span>。',
      '三类角色的共同点：<span class="k">不是所有 .cu 都挂在 dst->op 上</span>。',
      '下一课 L6-05：SYCL 后端怎么用同一套算子结构重写一遍。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2200 + i * 1500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
    }));
    tl.at(16000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[1]; });
    tl.at(18500, () => { msg.innerHTML = texts[2]; });
    tl.at(20500, () => { msg.innerHTML = texts[3]; });
  }
},

];
