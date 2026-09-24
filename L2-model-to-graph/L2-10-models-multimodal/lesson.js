/* ==========================================================================
   L2-10 · 模型家族（一）：多模态与视觉编码
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 六个"多模态"文件，只有一个是 <span class="hl-a">clip</span> —— 而它是个存根 */
{
  kicker: "L2-10 · 全局",
  title: "六个\"多模态\"文件，只有一个是 <span class=\"hl-a\">clip</span> —— 而它是个存根",
  sub: "6 个文件来自静态扫描分组：图构建代码里出现 clip_ / vision / mmproj / image_ 之一。命中原语不等于\"它就是多模态模型\"。",
  caption: "逐文件的真实主题与实测行数见 source.md 第一节；本幕末尾给出结论。",
  src: "src/models/clip.cpp",
  mark: [2, 5, 10, 15, 16],
  lineNo: 1,
  code: `#include "models.h"

// Stub to allow llama-quantize to open mmproj GGUFs

[[noreturn]]
void llama_model_clip::load_arch_hparams(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_hparams should not be called");
}

[[noreturn]]
void llama_model_clip::load_arch_tensors(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_tensors should not be called");
}

[[noreturn]]
std::unique_ptr<llm_graph_context> llama_model_clip::build_arch_graph(const llm_graph_params &) const {
    GGML_ABORT("CLIP has no inference graph via llama_model dispatch; runtime lives in tools/mtmd/clip.cpp");
//>> 这一行就是本课的第一个事实：CLIP 在 llama 侧没有推理图，运行时在 tools/mtmd/clip.cpp
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'clip.cpp · 18 行', b: '三个 [[noreturn]] 存根函数，<br>没有任何图构建代码', m: 'llama_model_clip' },
      { c: 'c', t: 'models.h · 2662 行', b: '151 个模型结构体的公共声明头，<br>含上面那个存根的类型声明', m: 'struct llama_model_clip' },
      { c: 'd', t: 'deepseek4.cpp · 1502 行', b: 'DeepSeek-V4 文本图（MoE + 压缩注意力）<br>命中：vision variant 的 MoE 路由偏置', m: 'ffn_exp_probs_b_vl' },
      { c: 'b', t: 'gemma4.cpp · 498 行', b: 'Gemma 4 文本图（SWA + 共享 KV）<br>命中：use_bidirectional_attention == "vision"', m: 'non_causal_type' },
      { c: 'e', t: 'qwen4exp.cpp · 1294 行', b: 'Qwen4 文本图（线性注意力 + PLE）<br>命中：一个元数据键 ple_image_token_id', m: 'LLM_KV_PLE_IMAGE_TOKEN_ID' },
      { c: 'f', t: 'pockettts.cpp · 146 行', b: 'PocketTTS 的"文本"主干（无 lm_head）<br>命中：音频 latent 由 mmproj 里的 flow net 产生', m: 'llama_model_pockettts' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看结论：<span class="k">这 6 个文件里没有一个是视觉塔</span>。真正的视觉编码器在 tools/mtmd（不在覆盖域）。',
      '<span class="v">clip.cpp</span> 只有 18 行：三处 GGML_ABORT。注释说它是 stub，只为让 llama-quantize 能打开 mmproj GGUF。',
      '<span class="v">models.h</span> 是 6 个文件共用的声明头（6 个文件全都包含它）。存根的类型声明就在这里。',
      '<span class="v">deepseek4.cpp</span> 命中 <span class="m">ffn_exp_probs_b_vl</span>：图像 token 用另一套 MoE 路由偏置。',
      '<span class="v">gemma4.cpp</span> 命中 "<span class="m">vision</span>"：HF 配置里的双向注意力开关，落在 SWA 层上。',
      '<span class="v">qwen4exp.cpp</span> 命中 <span class="m">image</span>：元数据键 <span class="m">ple_image_token_id</span>。',
      '<span class="v">pockettts.cpp</span> 命中 <span class="m">mmproj</span>：音频主干不产 logits，latent 由 mmproj 侧生成。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(900 + i * 2400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(15600, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '6 个文件里 <span class="v">1 个存根 + 1 个声明头 + 4 个文本图</span>。'
        + '多模态的"视觉那一半"要在 <span class="k">tools/mtmd</span> 里看，本课用它做对照。';
    });
  }
},

/* ------------------------------------------------------ 2 <span class="hl-c">models.h</span>：为什么 llama 侧必须留一个空壳 */
{
  kicker: "L2-10 · 声明",
  title: "<span class=\"hl-c\">models.h</span>：为什么 llama 侧必须留一个空壳",
  sub: "架构类都要满足 llama_model_base 的接口；CLIP 也用同一个接口声明，只是三个实现全部 [[noreturn]]。",
  caption: "回顾 L2-06：build_arch_graph 返回 llm_graph_context —— 这是所有模型图构建的统一入口。",
  src: "src/models/models.h",
  mark: [1, 3, 6, 9, 12],
  lineNo: 399,
  code: `// Quant-only stub for mmproj GGUFs
// none of these are ever called, they only exist to satisfy the llama_model_base interface
//>> 这句注释是本课第一个可核对的事实：这些函数永远不会被调用
struct llama_model_clip : public llama_model_base {
    llama_model_clip(const struct llama_model_params & params) : llama_model_base(params) {}

    [[noreturn]]
    void load_arch_hparams(llama_model_loader & ml) override;

    [[noreturn]]
    void load_arch_tensors(llama_model_loader & ml) override;

    [[noreturn]]
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
//>> 三个函数都是 [[noreturn]]：CLIP 在 llama 侧没有超参加载、没有张量加载、没有图
};`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '接口契约', b: '模型基类要求实现三个虚函数：<br>load_arch_hparams / load_arch_tensors / build_arch_graph',
        m: 'struct llama_model_clip : public llama_model_base' },
      { c: 'c', t: '为什么不能干脆删掉', b: 'llama-quantize 要打开 mmproj GGUF，就得按架构号查到<b>一个类型</b>；<br>没有类型，量化工具就打不开多模态模型',
        m: '// Stub to allow llama-quantize to open mmproj GGUFs' },
      { c: 'e', t: '为什么不能在这里实现', b: '真正的 CLIP 运行时在 tools/mtmd/clip.cpp（不在覆盖域）；<br>在 llama 侧再写一遍就是两套实现',
        m: 'GGML_ABORT' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'models.h 是公共声明头：151 个 <span class="m">struct llama_model_*</span>，每个模型一个。',
      '<span class="v">三个虚函数</span>是所有架构都必须满足的接口：超参、权重张量、图构建。',
      'CLIP 也声明了这个类型，但 <span class="k">三个实现都 [[noreturn]]</span>：它从不被调用。',
      '它存在的唯一理由是：<span class="k">让量化工具能按架构号打开 mmproj GGUF</span>。',
      '运行时在哪？下一幕看那一行注释指的路。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; });
    defs.forEach((_, i) => tl.at(900 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      U.markLines(document, [i === 0 ? 3 : (i === 1 ? 1 : 12)]);
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(13200, () => { els.forEach(e => { e.style.opacity = '1'; }); U.markLines(document, [1, 3, 12, 13]); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ <span class="hl-a">图像是以"已算好的 embedding"进文本图的</span> */
{
  kicker: "L2-10 · 核心",
  title: "★ <span class=\"hl-a\">图像是以\"已算好的 embedding\"进文本图的</span>",
  sub: "源码注释把这句话写死了：an image arrives as an embd batch, so ubatch->token is null。",
  caption: "对应 L2-05 的 llama_batch：tokens 与 embd 是二选一的两条输入路径。",
  src: "src/models/qwen4exp.cpp",
  mark: [3, 4, 5, 10],
  lineNo: 1066,
  code: `void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };
//>> tok_of() 是本课的关键：没有 token 时用占位 id，说明"这一批不是 token 批"`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="card" id="c1" style="width:300px;border-left-color:var(--c)">
          <div class="ct" style="color:var(--c)">token 路径（普通文本）</div>
          <div class="cb">ubatch-&gt;token != nullptr<br>
          每个位置有一个 token id，<br>文本图用 ggml_get_rows 查嵌入表得到输入。</div>
          <div class="cm">ubatch-&gt;token[k]</div>
        </div>
        <div class="card" id="c2" style="width:300px;border-left-color:var(--a)">
          <div class="ct" style="color:var(--a)">embd 路径（图像 / 音频）</div>
          <div class="cb">ubatch-&gt;token == nullptr<br>
          每个位置没有 token id，<br>输入是别人<b>已经算好</b>的一整块 embedding。</div>
          <div class="cm">ubatch-&gt;embd</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const c1 = wrap.querySelector('#c1'), c2 = wrap.querySelector('#c2');
    c1.style.opacity = '.30'; c2.style.opacity = '.30';
    const msg = wrap.querySelector('#msg');
    const texts = [
      'Qwen4 的 PLE 输入要按 n-gram 回溯历史 token；但<b>图像批次里根本没有 token</b>。',
      '所以源码先给出一句判断：<span class="v">an image arrives as an embd batch</span> —— 图像是以 embd 批的形式到达的。',
      '在没有 token id 的情况下，它只能<b>顶替一个 id</b>：元数据里的 ple_image_token_id，缺省用 EOS。',
      '最后一行把这个约定收成一个三元表达式：<br><span class="v">ubatch-&gt;token ? ubatch-&gt;token[k] : img_tok</span>。',
      '结论：<span class="k">文本图并不区分"文字"和"图像"</span> —— 它只区分"有 token id"和"有现成 embedding"。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; });
    tl.at(3000, () => { c1.style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [3, 4]); });
    tl.at(6600, () => { msg.innerHTML = texts[2]; U.markLines(document, [5, 6]); });
    tl.at(10200, () => { c2.style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [10]); });
    tl.at(14000, () => { c1.style.opacity = '1'; msg.innerHTML = texts[4]; U.markLines(document, [3, 4, 10]); });
  }
},

/* ------------------------------------------------------ 4 ★ 文本图在入口处分叉：<span class="hl-b">ubatch.token ? sqrtf(n_embd) : 1.0f</span> */
{
  kicker: "L2-10 · 核心",
  title: "★ 文本图在入口处分叉：<span class=\"hl-b\">ubatch.token ? sqrtf(n_embd) : 1.0f</span>",
  sub: "Gemma 4 用同一个指针判断\"这一批是 token 还是图像 embedding\"，并据此决定要不要缩放。",
  caption: "\"encoded image emdeddings\" 是源码注释原话（原文拼写如此）。这句判断的位置就在图的入口。",
  src: "src/models/gemma4.cpp",
  mark: [3],
  lineNo: 161,
  code: `    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
//>> ubatch.token == nullptr 就是"这一批是图像 embedding"的判据
    cb(inpL, "inp_scaled", -1);`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip a">build_inp_embd(tok_embd)</span>
        <span class="arrow">-&gt;</span>
        <span class="chip c">ubatch.token ? sqrtf(n_embd) : 1.0f</span>
        <span class="arrow">-&gt;</span>
        <span class="chip b">inpL</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="card" id="c1" style="width:300px;border-left-color:var(--c)">
          <div class="ct" style="color:var(--c)">token 批：要缩放</div>
          <div class="cb">token 嵌入要乘 sqrt(n_embd) —— 这是 Gemma 的嵌入归一化约定。</div>
          <div class="cm">ubatch.token ? sqrtf(n_embd)</div>
        </div>
        <div class="card" id="c2" style="width:300px;border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">embd 批：不缩放</div>
          <div class="cb">图像 embedding 已经是"算好的"，<b>不能再乘一次</b>，所以因子取 1.0f。</div>
          <div class="cm">ubatch.token ? sqrtf(n_embd) : 1.0f</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const c1 = wrap.querySelector('#c1'), c2 = wrap.querySelector('#c2');
    c1.style.opacity = '.30'; c2.style.opacity = '.30';
    const msg = wrap.querySelector('#msg');
    const texts = [
      '同一行里出现了两个分支：<span class="m">ubatch.token ? sqrtf(n_embd) : 1.0f</span>。',
      '有 token：按 Gemma 的约定乘 <span class="v">sqrt(n_embd)</span>。',
      '没有 token（图像 embedding）：乘 <span class="v">1.0f</span>，即原样通过。',
      '注释写明理由：<span class="k">raw embeddings input (i.e. encoded image emdeddings)</span> 不做归一化。',
      '两个分支<strong>汇合到同一个 inpL</strong>：后面的层完全不知道输入是文字还是图像。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [3]); });
    tl.at(3200, () => { c1.style.opacity = '1'; msg.innerHTML = texts[1]; });
    tl.at(6200, () => { c2.style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [2]); });
    tl.at(9400, () => { msg.innerHTML = texts[3]; });
    tl.at(12200, () => { c1.style.opacity = '1'; c2.style.opacity = '1'; msg.innerHTML = texts[4]; U.markLines(document, [0, 3]); });
  }
},

/* ------------------------------------------------------ 5 视觉塔是<b>另一张 ggml 图</b>：先 conv2d 切 patch */
{
  kicker: "L2-10 · 视觉塔",
  title: "视觉塔是<b>另一张 ggml 图</b>：先 conv2d 切 patch",
  sub: "tools/mtmd 里的视觉塔有自己的 ggml_context 与 cgraph；它与文本图不共享节点，只共享\"embedding 张量\"这个概念。",
  caption: "clip-graph.h 的契约写着：build_inp() 返回 shape [n_embd, n_patches]（见 source.md 第三节）。",
  src: "tools/mtmd/models/gemma4v.cpp",
  mark: [8, 9, 10],
  lineNo: 4,
  code: `ggml_cgraph * clip_graph_gemma4v::build() {
    ggml_tensor * inp_raw = build_inp_raw();

    // patches = 2 * (patches - 0.5)
    // equivalent to: patches * 2 - 1
    inp_raw = ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f);
    ggml_set_name(inp_raw, "inp_raw_scaled");

    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
//>> 转置 + cont 之后 ne = [n_embd, n_patches]，与 clip-graph.h 的契约一致
    ggml_set_name(inp, "inp");
    // note: no patch bias`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow wrap" id="chain" style="gap:6px">
        <span class="chip a">inp_raw 像素</span>
        <span class="arrow">-&gt;</span>
        <span class="chip c">scale_bias x2 -1</span>
        <span class="arrow">-&gt;</span>
        <span class="chip d">conv_2d(patch_size, patch_size, stride 1)</span>
        <span class="arrow">-&gt;</span>
        <span class="chip e">reshape_3d [n_patches, n_embd, n_batch]</span>
        <span class="arrow">-&gt;</span>
        <span class="chip b">transpose + cont [n_embd, n_patches]</span>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '视觉塔的入口是 <span class="v">build_inp_raw()</span>：一张原始图像张量（HWC 像素）。',
      'gemma4v 的注释给出归一化公式 <span class="m">patches * 2 - 1</span>，用一次 scale_bias 完成。',
      '<span class="v">ggml_conv_2d</span> 用 patch_size 的核、步长 1 切出 patch —— 这就是 patch embedding。',
      '卷积结果的维度顺序是 <span class="m">[n_patches, n_embd, n_batch]</span>。',
      '转置 + cont 之后变成 <span class="k">ne = [n_embd, n_patches]</span>：<br>ne[0] 是特征宽度，ne[1] 是 patch 数 —— 与 L1-01 的 ne 约定一致。',
      '从这里开始，整条塔都在 <span class="m">[n_embd, n_patches]</span> 这个布局上推进。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [1]); });
    tl.at(3200, () => { msg.innerHTML = texts[1]; U.markLines(document, [3, 4, 5]); });
    tl.at(6400, () => { msg.innerHTML = texts[2]; U.markLines(document, [8]); });
    tl.at(9400, () => { msg.innerHTML = texts[3]; U.markLines(document, [9]); });
    tl.at(12200, () => { msg.innerHTML = texts[4]; U.markLines(document, [10]); });
    tl.at(14600, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 ★ 形状全链：<span class="hl-a">[n_embd, n_patches]</span> 到 <span class="hl-a">[n_mmproj_embd, n_tokens]</span> */
{
  kicker: "L2-10 · 视觉塔",
  title: "★ 形状全链：<span class=\"hl-a\">[n_embd, n_patches]</span> 到 <span class=\"hl-a\">[n_mmproj_embd, n_tokens]</span>",
  sub: "每一步的 ne 都来自源码：ViT → pooler → projector，最后成为文本图入口的宽度。",
  caption: "术语回顾 L1-01：ne[0] 是最内层维度。这里 ne[0] 是特征宽度，ne[1] 是 token 数。",
  src: "tools/mtmd/models/gemma4v.cpp",
  mark: [0, 14, 19, 35, 36, 40],
  lineNo: 76,
  code: `    ggml_tensor * cur = build_vit(
                        inp, n_patches,
                        NORM_TYPE_RMS,
                        hparams.ffn_op,
                        nullptr, // pos embd is already handled above
                        add_pos);

    // Gemma4VisionPooler
    {
        const int kernel_size = hparams.n_merge;
        GGML_ASSERT(kernel_size > 0);

        // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, n_batch]
        cur = ggml_cont_4d(ctx0, ggml_transpose(ctx0, cur), n_patches_x, n_patches_y, n_embd, n_batch);
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG,
                           kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        const int out_x = n_patches_x / kernel_size;
        const int out_y = n_patches_y / kernel_size;
        // [out_x, out_y, n_embd, n_batch] -> [n_embd, out_x * out_y, n_batch]
        cur = ggml_reshape_3d(ctx0, cur, out_x * out_y, n_embd, n_batch);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_scale(ctx0, cur, sqrtf((float)n_embd));
        cb(cur, "pooled", -1);
    }

    // hidden_states = (hidden_states - self.std_bias) * self.std_scale
    if (model.std_bias && model.std_scale) {
        cur = ggml_sub(ctx0, cur, model.std_bias);
        cur = ggml_mul(ctx0, cur, model.std_scale);
        cb(cur, "std_scaled", -1);
    }

    // Gemma4MultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = build_mm(model.mm_input_proj_w, cur);
        cb(cur, "projected", -1);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="flow wrap" id="chain" style="gap:5px">
        <span class="chip a">[n_embd, n_patches]</span>
        <span class="arrow">-&gt;</span>
        <span class="chip d">build_vit x n_layer</span>
        <span class="arrow">-&gt;</span>
        <span class="chip c">kernel_size = hparams.n_merge</span>
        <span class="arrow">-&gt;</span>
        <span class="chip e">[n_embd, out_x * out_y]</span>
        <span class="arrow">-&gt;</span>
        <span class="chip f">mm_input_proj_w</span>
        <span class="arrow">-&gt;</span>
        <span class="chip b">[n_mmproj_embd, n_tokens]</span>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['步骤', '张量 ne', '出处（逐字引用见 source.md）'],
      [['patch 嵌入后', '[n_embd, n_patches]', 'clip-graph.h:98-100 / models/gemma4v.cpp:13-14'],
       ['ViT 输出', '[n_embd, n_patches]（形状不变）', 'models/gemma4v.cpp:76-96'],
       ['pooler 之后', '[n_embd, out_x * out_y]', 'models/gemma4v.cpp:92-96'],
       ['projector 之后（图的最后一个节点）', '[n_mmproj_embd, n_tokens]', 'models/gemma4v.cpp:111-113 / clip.cpp:5771-5782'],
       ['host 侧扁平缓冲', 'n_mmproj_embd * n_tokens 个 float', 'mtmd.cpp:1781-1783'],
       ['文本图入口张量', '[n_embd, n_tokens]', 'src/models/gemma4.cpp:475'],
       ['宽度约束', 'n_mmproj_embd == n_embd_inp', 'mtmd.cpp:603-608']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '从 <span class="v">[n_embd, n_patches]</span> 开始：n_embd 是视觉塔的隐藏宽度，n_patches 是 patch 数。',
      'ViT 层（<span class="m">build_vit</span>）只改数值，<b>不改形状</b>。',
      'pooler 用 <span class="m">n_merge</span> 的核做平均池化：token 数从 n_patches 降到 <span class="v">out_x * out_y</span>。',
      'projector（<span class="m">mm_input_proj_w</span>）把宽度从 n_embd 换成 <span class="v">n_mmproj_embd</span>。',
      '它同时是<b>整张视觉图的最后一个节点</b>：clip.cpp 直接取 gf 的最后一个节点，并用 ne[1] 校验 token 数。',
      'mtmd 把它拷成一条扁平 float 缓冲：<span class="m">n_embd_out * n_tokens_out</span> 个元素。',
      '文本图那侧看到的输入张量是 <span class="v">[n_embd, n_tokens]</span> —— 宽度必须等于 n_mmproj_embd。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
    tl.at(3000, () => { msg.innerHTML = texts[1]; U.markLines(document, [0]); });
    rows.forEach((r, i) => tl.at(5400 + i * 2200, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 2, 6)];
      U.markLines(document, i < 4 ? [0, 14, 19, 35, 36, 40].slice(0, i + 1) : [40]);
    }));
    tl.at(20800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 7 拼接点：<span class="hl-e">llama_batch</span> 的 <span class="hl-e">embd</span> 字段 */
{
  kicker: "L2-10 · 拼接",
  title: "拼接点：<span class=\"hl-e\">llama_batch</span> 的 <span class=\"hl-e\">embd</span> 字段",
  sub: "视觉塔的输出被拷成一条扁平 float 缓冲，再作为一个\"没有 token 的 batch\"交给 llama_decode。",
  caption: "mtmd 启动时会断言 n_mmproj_embd == 文本模型的 n_embd_inp，否则直接报错（source.md 第五节）。",
  src: "tools/mtmd/mtmd-helper-common.h",
  mark: [19, 20, 21],
  lineNo: 73,
  code: `struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        GGML_ASSERT(n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0);
        pos     .resize((size_t) n_tokens * (size_t) n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
//>> tokens 为 nullptr、embd 指向视觉塔输出 —— "拼接"就发生在这三行
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
    }`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" style="gap:10px">
        <div class="card" id="c1" style="width:290px;border-left-color:var(--a)">
          <div class="ct" style="color:var(--a)">视觉塔侧</div>
          <div class="cb">out_embd：一条扁平 float 缓冲<br>长度 = n_embd_out * n_tokens_out<br>
          （每 token 一段连续的特征）</div>
          <div class="cm">std::vector&lt;float&gt; out_embd</div>
        </div>
        <span class="arrow">-&gt;</span>
        <div class="card" id="c2" style="width:290px;border-left-color:var(--e)">
          <div class="ct" style="color:var(--e)">文本图侧</div>
          <div class="cb">llama_batch：tokens = nullptr<br>embd = 上面那条缓冲的指针<br>
          n_tokens = 图像 token 数</div>
          <div class="cm">llama_batch batch</div>
        </div>
      </div>
      <div class="flow" style="justify-content:center">
        <span class="chip c">n_mmproj_embd = llama_model_n_embd_inp(model)</span>
        <span class="arrow">-&gt;</span>
        <span class="chip b">llama_decode(lctx, batch_embd_view)</span>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const c1 = wrap.querySelector('#c1'), c2 = wrap.querySelector('#c2');
    c1.style.opacity = '.30'; c2.style.opacity = '.30';
    const msg = wrap.querySelector('#msg');
    const texts = [
      '视觉塔算完，留在 host 内存里的只有一条 float 缓冲。',
      '它被交给一个专门的小结构：<span class="v">decode_embd_batch</span>。',
      '构造函数里三行是本课的拼接点：<br><span class="k">n_tokens / tokens = nullptr / embd = embd</span>。',
      'tokens 为 nullptr 是<b>合法</b>的：文本图的入口会走 embd 分支（第 3、4 幕）。',
      '宽度由 <span class="m">llama_model_n_embd_inp(model)</span> 给出：图像 embedding 的宽度必须与文本模型一致。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; });
    tl.at(3000, () => { c1.style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [9]); });
    tl.at(6600, () => { c2.style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [19, 20, 21]); });
    tl.at(10200, () => { msg.innerHTML = texts[3]; U.markLines(document, [20]); });
    tl.at(13500, () => { msg.innerHTML = texts[4]; U.markLines(document, [11]); });
  }
},

/* ------------------------------------------------------ 8 同一个 LLM，四种"多模态"接法 */
{
  kicker: "L2-10 · 家族差异",
  title: "同一个 LLM，四种\"多模态\"接法",
  sub: "判据都不是 token id，而是 ubatch.embd 这个指针 —— 图在构建时就知道自己是不是在处理媒体输入。",
  caption: "L2-11~L2-15 继续看其它家族：SSM 与线性注意力、稀疏 MoE、稠密 Transformer。",
  src: "src/models/deepseek4.cpp",
  mark: [4, 5, 8, 11],
  lineNo: 1280,
  code: `        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl is input is from mtmd
        const bool is_media = ubatch.embd != nullptr;
        if (is_media) {
            if (layer.ffn_exp_probs_b_vl) {
                exp_probs_b = layer.ffn_exp_probs_b_vl;
            }
        } else if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
        }`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="formula" id="msg"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'd', t: 'deepseek4', b: 'ubatch.embd != nullptr 时换用 <b>ffn_exp_probs_b_vl</b>：<br>图像 token 走另一套 MoE 路由偏置',
        m: 'is_media = ubatch.embd != nullptr' },
      { c: 'b', t: 'gemma4', b: 'HF 配置 use_bidirectional_attention == "vision" 时，<br>SWA 层改双向注意力',
        m: 'non_causal_type' },
      { c: 'e', t: 'qwen4exp', b: '图像批次没有 token id，用 <b>ple_image_token_id</b><br>（缺省 EOS）顶替',
        m: 'ubatch->token ? ubatch->token[k] : img_tok' },
      { c: 'f', t: 'pockettts', b: '音频侧反过来：主干不产 logits，<br>latent 由 mmproj 里的 flow net 生成',
        m: 'no lm_head' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'DeepSeek-V4 的图在 MoE 层前问一句：<span class="v">is_media = ubatch.embd != nullptr</span>。',
      '有媒体输入：用 vision variant 的路由偏置张量；<br>否则：哈希路由层用 token id 查表，其余层用普通偏置。',
      '同一个 LLM 主干，<span class="k">只在少数节点上分叉</span>。',
      '其它家族的"多模态钩子"各不相同：双向注意力开关、占位 token id、音频 latent 生成。',
      '共同点只有一个：<span class="k">媒体输入都被翻译成"已算好的 embedding"</span>，从 embd 路径进图。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [4, 5]); });
    tl.at(3400, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [8, 11]); });
    tl.at(7000, () => { msg.innerHTML = texts[2]; });
    defs.forEach((_, i) => { if (i > 0) tl.at(9200 + (i - 1) * 2400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[3];
    }); });
    tl.at(15200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L2-10 · 收束",
  title: "把这一课压成一张表",
  sub: "验收点：视觉塔的输出以什么形状喂给 LLM 部分。",
  caption: "下一课 L2-11 模型家族（二）：状态空间与线性注意力。",
  src: "src/models/gemma4.cpp",
  mark: [2],
  lineNo: 473,
  code: `// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案（每一格都能在源码里指到）'],
      [['视觉塔输出什么形状', 'ne = [n_mmproj_embd, n_tokens]，F32；它就是视觉图 gf 的最后一个节点'],
       ['怎么喂给 LLM 部分', '拷成 host 侧扁平缓冲，填进 llama_batch.embd，且 tokens = nullptr'],
       ['宽度谁保证一致', 'mtmd 断言 mmproj 的 n_mmproj_embd == 文本模型的 n_embd_inp'],
       ['文本图看到什么', '入口张量 [n_embd, n_tokens]，与 token 路径在同一处汇合'],
       ['token 数谁定', 'clip_n_output_tokens()；clip.cpp 用 embeddings->ne[1] 校验']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一张图像经过 mmproj 之后，喂给文本图的张量是什么形状？它和普通 token 路径在哪一步汇合？',
      '形状是 <span class="mono">ne = [n_mmproj_embd, n_tokens]</span> 的 F32 张量：'
      + '<span class="mono">ne[0]</span> 是投影后的特征宽度，<span class="mono">ne[1]</span> 是这张图产生的 token 数'
      + '（clip.cpp:5776 用 <span class="mono">embeddings-&gt;ne[1]</span> 校验）。<br>'
      + '它<b>不走</b> <span class="mono">ggml_get_rows(tok_embd, tokens)</span>，而是被拷成一条扁平 float 缓冲、'
      + '填进 <span class="mono">llama_batch.embd</span>（此时 <span class="mono">tokens = nullptr</span>）。<br>'
      + '文本图在入口 <span class="mono">build_inp_embd</span> 处得到 <span class="mono">[n_embd, n_tokens]</span> 的输入张量，'
      + 'token 路径与 embd 路径就在这里汇合 —— 后面的层分不清输入是文字还是图像。<br>'
      + '前提是 <span class="mono">n_mmproj_embd == n_embd_inp</span>，否则 mtmd 在启动时直接报错。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '视觉塔的输出不是一个"特别的类型"，就是一个普通张量：<span class="v">ne = [n_mmproj_embd, n_tokens]</span>。',
      '它进文本图的通道是 <span class="k">llama_batch.embd</span> —— L2-05 讲过的那条"现成 embedding"路径。',
      '宽度必须等于文本模型的 <span class="v">n_embd_inp</span>；文本图入口的输入张量写着 <span class="m">[n_embd, n_tokens]</span>。',
      '所以：<span class="k">多模态不是新引擎，而是"多张图接在一起"</span>。',
      '下一课看另一类家族：状态空间与线性注意力 —— 它们的"记忆"在图上长什么样。'
    ];
    tl.at(400, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(14800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
