/* ==========================================================================
   L2-09 · 量化、导出与适配器
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 模型进入图之前，有四件事会发生 */
{
  kicker: "L2-09 · 全局",
  title: "模型进入图之前，有四件事会发生",
  sub: "量化改数值与类型，导出改容器，适配器改图；grammar 与对话模板则完全不碰图。",
  caption: "本课逐字引用计划里的 10 个文件；另追加 src/llama-graph.cpp 作为\"LoRA 在图里的插入点\"的证据（见 README 说明）。",
  src: "src/llama-quant.cpp",
  mark: [0, 10, 11, 12],
  lineNo: 913,
  code: `static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
//>> 量化是"读一个 GGUF、写一个 GGUF"；中间全是逐张量的决策
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw std::runtime_error(format("invalid output file type %d\\n", ftype));
    }
`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">权重文件</span><span class="arrow">-></span>
        <span class="chip b">量化 · 导出</span><span class="arrow">-></span>
        <span class="chip c">GGUF</span><span class="arrow">-></span>
        <span class="chip d">建图</span><span class="arrow">-></span>
        <span class="chip e">适配器 · 约束</span>
      </div>
      <div class="row wrap" id="s1cards" style="gap:8px"></div>
      <div class="formula" id="s1msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#s1cards');
    const defs = [
      { c: 'a', t: '量化 · llama-quant.cpp', m: 'llama_model_quantize_impl',
        b: '逐张量选类型，再把浮点权重量化成块。<br>写出的还是 GGUF，但数值和 type 都变了。' },
      { c: 'b', t: '导出 · llama-model-saver', m: 'llama_model_saver::save',
        b: '把内存里的模型登记进一个 gguf_context，<br>只有 save() 才真正落盘。' },
      { c: 'c', t: '适配器 · llama-adapter', m: 'llama_adapter_lora::ab_map',
        b: '权重一个字节都不动。<br>建图时给命中的权重多插一次矩阵乘。' },
      { c: 'd', t: '约束 · grammar / chat', m: 'llama_grammar_apply_impl',
        b: '也不进图：一个把非法 token 的 logit 打成 -INFINITY，<br>一个把对话拼成 prompt 文本。' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#s1msg');
    const texts = [
      '本课覆盖 10 个文件（外加 1 个追加引用）。先看四条支路各自动了什么。',
      '<span class="k">量化</span>：入口参数只有一个 <span class="v">ftype</span>，' +
      '但真正被决定的是<b>每一个张量</b>的目标类型 —— 第 2 幕展开。',
      '<span class="k">导出</span>：<span class="v">add_kv</span> 写元数据、' +
      '<span class="v">add_tensor</span> 登记张量指针、<span class="v">save()</span> 落盘，三件事分开做。',
      '<span class="k">适配器</span>：这是本课的核心洞察 —— ' +
      'LoRA 不是"改权重"，是<b>在图里多插一次矩阵乘</b>，所以同一个基座能同时挂多套。',
      '<span class="k">grammar 与对话模板</span>都不在图里：' +
      '前者改 logits（回顾 L2-08），后者决定 prompt 文本。'
    ];
    defs.forEach((_, i) => tl.at(600 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">一个 ftype</span>不是"所有张量同一个类型" */
{
  kicker: "L2-09 · 量化",
  title: "★ <span class=\"hl-a\">一个 ftype</span>不是\"所有张量同一个类型\"",
  sub: "ftype 只换算出一个默认类型；最终类型是逐张量算出来的：类别、层号、形状各有权重。",
  caption: "下一幕讲 imatrix：为什么低比特类型必须靠它校准。",
  src: "src/llama-quant.cpp",
  mark: [0, 3, 7, 23, 25, 29],
  lineNo: 708,
  code: `    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (ggml_is_quantized(default_type)) {
//>> 默认类型之上，还有三种"手工/特例"覆盖
        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs.tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\\n",
                                       __func__, tensor_name.c_str(), ggml_type_name(new_type), ggml_type_name(qtype));
                        new_type = qtype;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual && !params->pure) {
//>> 没被手工覆盖、也不是"纯量化"时，才走按类别与层号的混合精度表
            new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
//>> 形状装不下目标类型时回退到可用类型（tensor_type_fallback）
    }

    return new_type;
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="s2tbl"></div><div class="formula" id="s2msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['次序', '谁说话', '代码里的名字'],
      [['1 默认',     'ftype 换算出一个默认类型',        'default_type'],
       ['2 词嵌入',   'category 是 TOKEN_EMBD 时覆盖',   'token_embedding_type'],
       ['3 输出层',   'category 是 OUTPUT 时覆盖',       'output_tensor_type'],
       ['4 手工覆盖', '张量名正则，命中即生效',          'tensor_type_patterns'],
       ['5 混合精度', '按张量类别 + 层号决定多给比特',   'llama_tensor_get_type_impl'],
       ['6 形状回退', '目标类型装不下这个形状',          'tensor_type_fallback']],
      { monoCols: [2] });
    wrap.querySelector('#s2tbl').appendChild(t.el);

    const msg = wrap.querySelector('#s2msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '同一个 <span class="v">ftype</span>（比如一个 4 比特混合精度档），落到不同张量上可以是不同类型。顺序就是这张表。',
      '<span class="v">default_type</span> 只是起点：<span class="k">ftype -> 默认 ggml_type</span> 的一次换算。',
      '<span class="k">词嵌入</span>与<span class="k">输出层</span>是两个可以单独指定的张量：' +
      '它们对质量最敏感，所以给了独立的覆盖入口。',
      '<span class="k">手工覆盖</span>按张量名正则生效；命中后连标准策略都不再走。',
      '<span class="k">混合精度</span>才是"哪里多给比特"的真正算法：它按张量类别（attn_v / ffn_down / …）' +
      '和层号（前 1/8、后 1/8 之类）决定升到 Q5_K、Q6_K 还是保持 Q4_K。',
      '最后一步是<span class="k">形状回退</span>：某些形状无法用目标类型表示，必须换一个能用的类型。',
      '所以"我用了某个量化档"这句话，展开后是<b>几百个张量各自的目标类型</b>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2500 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(18600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 3 ★ <span class="hl-c">imatrix</span>：改的是误差往哪儿分，不是算法 */
{
  kicker: "L2-09 · 量化",
  title: "★ <span class=\"hl-c\">imatrix</span>：改的是误差往哪儿分，不是算法",
  sub: "低比特的 IQ* 家族必须有重要性矩阵，否则直接抛错 —— 因为\"哪一列重要\"本身是数据。",
  caption: "\"imatrix\" 的形状是 ne[0] x ne[2]（按行 1247 校验）；下一幕看它怎么被按行取用。",
  src: "src/llama-quant.cpp",
  mark: [0, 1, 5, 13, 17],
  lineNo: 822,
  code: `static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
//>> 两个豁免：词嵌入与 output 权重
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
//>> k-quant 家族唯一的例外：Q2_K 只在 Q2_K_S 这种档位下才要 imatrix
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="s3cards" style="gap:8px"></div>
      <div class="formula" id="s3msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#s3cards');
    const defs = [
      { c: 'c', t: '必须有', b: 'IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS /<br>IQ2_S / IQ3_XXS —— 六个 IQ 类型' },
      { c: 'd', t: '唯一的 k-quant 例外', b: 'Q2_K 只在 Q2_K_S 这种档位下才要；<br>源码注释：k 系量化一般不要求 imatrix' },
      { c: 'b', t: '永远豁免', b: '词嵌入与 output 权重：<br>它们本来就不参与这套判断' },
      { c: 'g', t: '缺了会怎样', b: '收集阶段直接抛 runtime_error；<br>日志写着 The result will be garbage' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#s3msg');
    const texts = [
      '<span class="v">tensor_requires_imatrix()</span> 只有二十行，却把"哪些类型非它不可"写死了。',
      '六个 <span class="k">IQ*</span> 类型返回 true：它们的码本/码率太激进，' +
      '必须靠重要性矩阵告诉量化器<b>哪些列更值钱</b>。',
      '<span class="k">Q2_K</span> 是唯一的例外，而且取决于档位：只有 Q2_K_S 才要求。',
      '<span class="k">词嵌入与 output 权重</span>不参与 —— 所以这两类张量即使在 IQ 档位下也不报错。',
      '缺 imatrix 的后果不是"质量差点"：<span class="v">llama_model_quantize_impl</span> 在' +
      '收集阶段就抛错（第 1099-1109 行），日志原话是 The result will be garbage。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 逐行量化：<span class="hl-b">chunk</span> 不跨专家边界 */
{
  kicker: "L2-09 · 量化",
  title: "逐行量化：<span class=\"hl-b\">chunk</span> 不跨专家边界",
  sub: "一张权重矩阵按行切开，行号全局连续；但每个专家有自己的 imatrix 切片，所以切块要在边界处断开。",
  caption: "回顾 L1-04：Q4_0 的\"32 个权重 + 1 个 scale\"这类块布局，正是在 ggml_quantize_chunk 里落地的。",
  src: "src/llama-quant.cpp",
  mark: [0, 4, 5, 15],
  lineNo: 746,
  code: `// note: chunks never cross an expert boundary since each expert has its own imatrix slice
static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t first_row, int64_t nrows, int64_t nrows_per_expert, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    const size_t row_size = ggml_row_size(new_type, n_per_row);

    auto imatrix_for_row = [=](int64_t row_global) {
        return imatrix ? imatrix + (row_global / nrows_per_expert) * n_per_row : nullptr;
    };

    if (nthread < 2) {
        // single-thread
        size_t new_size = 0;
        for (int64_t row = 0; row < nrows;) {
            const int64_t row_global = first_row + row;
            const int64_t this_nrow  = std::min(nrows - row, nrows_per_expert - row_global % nrows_per_expert);
            void * this_data = (char *) new_data + row * row_size;
            size_t this_size = ggml_quantize_chunk(new_type, f32_data + row * n_per_row, this_data, 0, this_nrow, n_per_row, imatrix_for_row(row_global));
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
//>> 量化结果还要逐块校验：校验不过就整体抛错
                throw std::runtime_error("quantized data validation failed");
            }
            new_size += this_size;
            row += this_nrow;
        }
        return new_size;`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">nrows_total = 24</span><span class="arrow">=</span>
        <span class="chip b">nrows_per_expert = 8</span><span class="arrow">x</span>
        <span class="chip c">3 个专家</span>
      </div>
      <div id="s4ruler" style="display:flex;gap:6px;justify-content:center"></div>
      <div class="formula" id="s4msg"></div>`;
    root.appendChild(wrap);

    const ruler = wrap.querySelector('#s4ruler');
    const cells = [];
    for (let e = 0; e < 3; e++) {
      const seg = U.el('div', { style: 'display:flex;gap:2px;padding-right:5px;margin-right:3px;border-right:2px solid var(--c)' });
      for (let r = 0; r < 8; r++) {
        const c = U.el('div', { style: 'width:19px;height:17px;border:1px solid var(--border);border-radius:2px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
        c.textContent = String(e * 8 + r);
        seg.appendChild(c); cells.push(c);
      }
      ruler.appendChild(seg);
    }

    const msg = wrap.querySelector('#s4msg');
    const texts = [
      '一张权重矩阵的行是<b>全局连续编号</b>的：专家的行也排在同一条行号轴上。',
      '一次 <span class="v">ggml_quantize_chunk</span> 处理若干行（chunk）；多线程时按 chunk 抢任务，' +
      '所以线程数会被 chunk 数限制。',
      '<span class="k">chunk 在专家边界处强制断开</span>：<span class="v">this_nrow</span> 同时受' +
      '<span class="v">nrows_per_chunk</span> 与 <span class="v">nrows_per_expert</span> 约束。',
      '原因在这一行 lambda：imatrix 是<b>按专家切片</b>存的，' +
      '<span class="v">row_global / nrows_per_expert</span> 就是要落到第几张切片。',
      '<span class="v">ggml_row_size(new_type, n_per_row)</span> 决定每行多少字节 ——' +
      '这正是 L1-04 讲的块布局，也是 L1-01 讲的 nb[0]。'
    ];
    tl.at(700, () => {
      msg.innerHTML = texts[0];
      [0, 1, 2, 3, 4].forEach(i => { cells[i].style.background = 'rgba(88,166,255,.22)'; cells[i].style.borderColor = 'var(--a)'; });
    });
    tl.at(3700, () => {
      [5, 6, 7].forEach(i => { cells[i].style.background = 'rgba(63,185,80,.22)'; cells[i].style.borderColor = 'var(--b)'; });
      msg.innerHTML = texts[1];
    });
    tl.at(6700, () => {
      cells[7].style.borderColor = 'var(--c)'; cells[7].style.borderRight = '3px solid var(--c)';
      msg.innerHTML = texts[2];
    });
    tl.at(9700, () => { msg.innerHTML = texts[3]; });
    tl.at(12700, () => { [8, 9, 10, 11, 12].forEach(i => { cells[i].style.background = 'rgba(210,153,34,.22)'; }); msg.innerHTML = texts[3]; });
    tl.at(15700, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 导出：<span class="hl-d">llama_model_saver</span> 的三分工 */
{
  kicker: "L2-09 · 导出",
  title: "导出：<span class=\"hl-d\">llama_model_saver</span> 的三分工",
  sub: "登记元数据、登记张量、写盘是三个独立动作；结构体自己只持有\"一个还没落盘的 GGUF\"。",
  caption: "回顾 L1-06：GGUF 是\"元数据 KV + 张量表 + 数据段\"的容器；这一课看它的写出侧。",
  src: "src/llama-model-saver.h",
  mark: [1, 3, 11, 26, 29, 31, 33],
  lineNo: 12,
  code: `struct llama_model_saver {
    struct gguf_context * gguf_ctx = nullptr;
    const bool gguf_ctx_owned;
    const struct llama_model * model;
//>> gguf_ctx 就是"还没落盘的文件"：所有登记都进这里
    const struct LLM_KV llm_kv;

    llama_model_saver(const struct llama_model * model);
    llama_model_saver(enum llm_arch arch, struct gguf_context * gguf_ctx);
    ~llama_model_saver();

    void add_kv(enum llm_kv key, uint32_t     value);
    void add_kv(enum llm_kv key, int32_t      value);
    void add_kv(enum llm_kv key, uint64_t     value);
    void add_kv(enum llm_kv key, float        value);
    void add_kv(enum llm_kv key, bool         value);
    void add_kv(enum llm_kv key, const char * value);

    [[noreturn]]
    void add_kv(enum llm_kv key, char value); // needed to make the template below compile

    template <typename Container>
    void add_kv(enum llm_kv key, const Container & value, bool per_layer = false);

    void add_kv(enum llm_kv key, const std::vector<std::string> & value);

    void add_tensor(const struct ggml_tensor * tensor);
//>> add_tensor 只登记张量指针，不拷贝字节

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string & path_model);
//>> 真正的写盘只有这两行
    void save(FILE * file);
};`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">add_kv_from_model()</span>
        <span class="arrow">+</span>
        <span class="chip b">add_tensors_from_model()</span>
        <span class="arrow">-></span>
        <span class="chip c">gguf_context</span>
        <span class="arrow">-></span>
        <span class="chip d">save(path)</span>
        <span class="arrow">-></span>
        <span class="chip e">.gguf</span>
      </div>
      <div class="row wrap" id="s5cards" style="gap:8px"></div>
      <div class="formula" id="s5msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#s5cards');
    const defs = [
      { c: 'a', t: 'add_kv 的重载族', b: 'u32 / i32 / u64 / f32 / bool / 字符串 /<br>容器（还可以按层展开）' },
      { c: 'b', t: 'add_tensor', b: '名字已存在就跳过（只有 rope_freqs<br>那三个是例外，见 137 行断言）' },
      { c: 'c', t: 'save', b: 'gguf_write_to_file(gguf_ctx, path, false)<br>—— 到这一步才产生字节' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#s5msg');
    const texts = [
      '这个结构体只有 4 个字段：一个 gguf_context、一个"是否自己拥有它"、一个模型指针、一个 KV 名字生成器。',
      '<span class="v">gguf_ctx</span> 承载整个导出过程 —— 它此时还只是内存里的一个 GGUF 描述。',
      '<span class="k">元数据</span>侧：<span class="v">add_kv</span> 按类型重载，容器版本还能按层展开成数组。',
      '<span class="k">张量</span>侧：<span class="v">add_tensor</span> 把已有张量<b>登记</b>进上下文；' +
      '字节仍然在模型自己的 buffer 里，直到 save() 才被写出去。',
      '<span class="k">写盘</span>侧：<span class="v">save()</span> 只有一行调用 —— ' +
      '这也是 L1-06 里那张"容器 -> 文件"的落地处。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 ★ LoRA 不是改权重，是在图里<span class="hl-a">多插一次矩阵乘</span> */
{
  kicker: "L2-09 · 适配器",
  title: "★ LoRA 不是改权重，是在图里<span class=\"hl-a\">多插一次矩阵乘</span>",
  sub: "基座权重 w 在 ggml_mul_mat 之后原封不动；每个命中的 adapter 再追加 a -> b -> scale -> add。",
  caption: "本幕引用的 src/llama-graph.cpp 不在计划给 L2-09 的 10 个文件里 —— 它是\"图内插入点\"的唯一源码证据，故追加引用。",
  src: "src/llama-graph.cpp",
  mark: [4, 8, 12, 14, 15, 20, 22, 23, 24, 28, 29],
  lineNo: 1514,
  code: `ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
//>> 基座路径：w 在这里被读一次，之后再也不被写

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
//>> w_s 是逐张量缩放，和 LoRA 无关，只是搭同一趟车
    }

    for (const auto & lora : *loras) {
//>> 循环：挂在同一基座上的每一套 adapter 都各追加一次
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
//>> 内层把 cur 投到 rank 维，外层再投回 n_embd
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:8px">
        <div class="col grow" style="gap:6px" id="s6base"></div>
        <div class="col grow" style="gap:6px" id="s6lora"></div>
      </div>
      <div class="formula" id="s6msg"></div>`;
    root.appendChild(wrap);

    function nodeRow(host, items, cls) {
      const r = U.el('div', { class: 'flow', style: 'gap:4px' });
      items.forEach((t, i) => {
        if (i) r.appendChild(U.arrow('->'));
        r.appendChild(U.chip(t, cls));
      });
      host.appendChild(r);
      return r;
    }
    const base = wrap.querySelector('#s6base');
    base.innerHTML = '<div class="cm" style="margin:0 0 2px">没有 adapter</div>';
    const b1 = nodeRow(base, ['cur', 'mul_mat(w, cur)', 'res'], 'a');
    base.appendChild(U.el('div', { class: 'cb', text: '1 个矩阵乘算子' }));

    const lora = wrap.querySelector('#s6lora');
    lora.innerHTML = '<div class="cm" style="margin:0 0 2px">挂了一套 adapter</div>';
    const l1 = nodeRow(lora, ['cur', 'mul_mat(w, cur)', 'res'], 'a');
    const l2 = nodeRow(lora, ['cur', 'mul_mat(a, cur)', 'mul_mat(b, .)', 'scale'], 'e');
    const l3 = nodeRow(lora, ['res', 'add', "res'"], 'b');
    lora.appendChild(U.el('div', { class: 'cb', text: '多出 5 个节点 + 1 次加法（源码注释算作 6 个）' }));

    const msg = wrap.querySelector('#s6msg');
    const texts = [
      '先看没有 adapter 的情况：<span class="v">build_lora_mm</span> 就是一次普通的 <span class="k">ggml_mul_mat(ctx0, w, cur)</span>。',
      '这个函数名里的 <span class="v">lora</span> 说明了它的地位：<b>所有走它的权重才可能被 LoRA 作用</b>。',
      '<span class="v">get_weight(w)</span> 拿基座张量的<b>名字</b>去查表；查不到就 <span class="k">continue</span>，' +
      '这套 adapter 与这个权重无关。',
      '命中时追加一条<b>并行支路</b>：<span class="v">mul_mat(a, cur)</span> 降到 rank 维，' +
      '<span class="v">mul_mat(b, .)</span> 再升回输出维，然后乘上 scale。',
      '最后 <span class="v">ggml_add(ctx0, res, ab_cur)</span> 把支路并回主干 —— ' +
      '<b>基座权重 w 从头到尾只被读，没有被改写。</b>',
      '所以同一个基座可以同时挂多套 adapter：<span class="v">for (const auto & lora : *loras)</span> 每套各加一次，' +
      '代价只是图上多出节点。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [4]); });
    tl.at(4000, () => { msg.innerHTML = texts[1]; U.markLines(document, [4, 8]); });
    tl.at(7300, () => { msg.innerHTML = texts[2]; U.markLines(document, [12]); });
    tl.at(10600, () => { msg.innerHTML = texts[3]; U.markLines(document, [22, 24]); });
    tl.at(13900, () => { msg.innerHTML = texts[4]; U.markLines(document, [28, 29]); });
    tl.at(17200, () => { msg.innerHTML = texts[5]; U.markLines(document, [4, 12, 22, 29]); });
  }
},

/* ------------------------------------------------------ 7 适配器的全部状态：一张 <span class="hl-e">ab_map</span> */
{
  kicker: "L2-09 · 适配器",
  title: "适配器的全部状态：一张 <span class=\"hl-e\">ab_map</span>",
  sub: "一套 adapter = \"基座张量名 -> (lora_a, lora_b)\" 的映射 + 一个 alpha；作用点由建图时按名字查表决定。",
  caption: "回顾 L2-06：llm_graph_context 是 build_* 原语族的宿主；build_lora_mm 就是它的一个原语。",
  src: "src/llama-adapter.h",
  mark: [4, 10, 11, 20, 24, 41, 44],
  lineNo: 44,
  code: `//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
//>> a、b 两个小矩阵就是 LoRA 的全部数据

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;
//>> ab_map 是这套 adapter 的全部状态：名字 -> (a, b)

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
//>> 这行注释就是"图会多出多少节点"的账本
    }
};`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="s7cards" style="gap:8px"></div>
      <div class="formula" id="s7msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#s7cards');
    const defs = [
      { c: 'e', t: 'ab_map', m: 'name -> (a, b)', b: '一套 adapter 一张表。<br>每套 adapter 还带一个 alpha。' },
      { c: 'a', t: 'get_weight(w)', m: 'w->name 查表', b: '查不到返回 nullptr ——<br>这套 adapter 就跳过这个权重。' },
      { c: 'c', t: 'get_scale', m: 's * alpha / rank', b: 'rank 取自 b->ne[0]；<br>alpha 为 0 时只用传入的 scale。' },
      { c: 'b', t: 'get_n_nodes()', m: 'ab_map.size() * 6u', b: '命中多少个权重，<br>图上就多出 6 倍个节点。' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#s7msg');
    const texts = [
      '适配器的"数据面"只有这两个结构体：一个小矩阵对，加一张名字表。',
      '<span class="v">llama_adapter_lora_weight</span> 就是 <span class="v">a</span> 与 ' +
      '<span class="v">b</span> 两个张量指针 —— 没有别的。',
      '<span class="v">get_scale()</span> 把 rank 从形状里读出来：' +
      '<span class="v">rank = b->ne[0]</span>，再按 alpha 折算实际缩放。',
      '<span class="v">ab_map</span> 的键是<b>基座张量的名字</b>；' +
      '<span class="v">get_weight(w)</span> 就是一次 <span class="v">find</span>。',
      '<span class="v">get_n_nodes()</span> 把上一幕的图变化量化了：' +
      '每命中一个权重 6 个节点（a、b、scale、add、两次 mul_mat）。',
      '所以"多套 adapter"在图上就是<b>多次 add</b>，在权重的字节里什么痕迹都没有。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
    tl.at(16500, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 grammar 不进图：它在采样链上<span class="hl-c">改 logits</span> */
{
  kicker: "L2-09 · 约束",
  title: "grammar 不进图：它在采样链上<span class=\"hl-c\">改 logits</span>",
  sub: "采样前把不可能的 token 打成 -INFINITY，采样后再按生成的文本推进规则栈 —— 两步都在图外。",
  caption: "回顾 L2-08：采样在 CPU 上做、不进图；grammar 正是这条链上的一环，所以它能直接改 logits。",
  src: "src/llama-grammar.cpp",
  mark: [0, 3, 8, 26, 28, 38, 40],
  lineNo: 1355,
  code: `void llama_grammar_apply_impl(const struct llama_grammar & grammar, llama_token_data_array * cur_p) {
    GGML_ASSERT(grammar.vocab != nullptr);

    if (grammar.awaiting_trigger) {
//>> awaiting_trigger：懒 grammar 在触发词出现前完全不约束
        return;
    }

    bool allow_eog = false;
    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            allow_eog = true;
            break;
        }
    }

    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> candidates_decoded;
    candidates_decoded.reserve(cur_p->size);

    llama_grammar_candidates candidates_grammar;
    candidates_grammar.reserve(cur_p->size);

    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id      = cur_p->data[i].id;
        const std::string & piece = grammar.vocab->token_to_piece(id);

        if (grammar.vocab->is_eog(id)) {
            if (!allow_eog) {
                cur_p->data[i].logit = -INFINITY;
            }
        } else if (piece.empty() || piece[0] == 0) {
            cur_p->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(piece, grammar.partial_utf8));
            candidates_grammar.push_back({ i, candidates_decoded.back().first.data(), candidates_decoded.back().second, id });
        }
    }

    const auto rejects = llama_grammar_reject_candidates(grammar.rules, grammar.stacks, candidates_grammar);
    for (const auto & reject : rejects) {
        cur_p->data[reject.index].logit = -INFINITY;
//>> 被拒绝的 token 不是被删掉，而是把 logit 打成 -INFINITY
    }
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">logits</span><span class="arrow">-></span>
        <span class="chip c">grammar 拒绝的置 -INFINITY</span><span class="arrow">-></span>
        <span class="chip b">采样（L2-08）</span><span class="arrow">-></span>
        <span class="chip d">accept_token 推进栈</span>
      </div>
      <div class="row wrap" id="s8cards" style="gap:8px"></div>
      <div class="formula" id="s8msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#s8cards');
    const defs = [
      { c: 'a', t: 'stacks', b: '一组下推自动机的栈；<br>空了才表示"这个规则已经满足"。' },
      { c: 'b', t: 'allow_eog', b: '只要有任意一个栈为空，<br>才允许结束符活着。' },
      { c: 'c', t: 'candidates', b: '每个 token 先取出 piece 并解码成<br>code point，再拿去和规则比。' },
      { c: 'd', t: 'accept_token', b: '采样之后按同一个 piece<br>推进栈（1472 行）。' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#s8msg');
    const texts = [
      '整个函数没有出现任何 ggml 张量 —— 它操作的是采样器的候选数组。',
      '<span class="v">awaiting_trigger</span> 为真时直接返回：<b>懒 grammar 在触发词出现前一个 token 都不约束</b>。',
      '<span class="k">allow_eog</span> 由栈是否为空算出；不允许结束时，所有结束符的 logit 被置为 -INFINITY。',
      '其余 token 走解码 + 规则匹配，被拒绝的同样置 <span class="v">-INFINITY</span> —— ' +
      '<b>不是删除，软屏蔽</b>，后面的采样器仍看到完整候选表。',
      '采样结束后 <span class="v">llama_grammar_accept_token</span> 用<b>同一段 piece</b> 推进栈：' +
      '生成与约束用的是同一个字符串，所以不会错位。',
      '这也解释了 L2-08 的验收点：<b>约束发生在采样这一步，而采样不在图里</b>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
    tl.at(17000, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 对话模板：一张 <span class="hl-f">硬编码名字表</span> + 一条 if-else 链 */
{
  kicker: "L2-09 · 对话模板",
  title: "对话模板：一张 <span class=\"hl-f\">硬编码名字表</span> + 一条 if-else 链",
  sub: "模板不是从文件里解析出来的语法，而是几十个手写分支；每个分支把消息拼成一段 prompt 文本。",
  caption: "这段文本随后被分词成 inp_tokens —— 图的第一批输入；模板决定了\"图上看到什么 token\"。",
  src: "src/llama-chat.cpp",
  mark: [0, 6, 10, 12, 13],
  lineNo: 244,
  code: `int32_t llm_chat_apply_template(
    llm_chat_template tmpl,
    const std::vector<const llama_chat_message *> & chat,
    std::string & dest, bool add_ass) {
    // Taken from the research: https://github.com/ggml-org/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == LLM_CHAT_TEMPLATE_CHATML) {
//>> 这一支只处理 chatml。往下还有几十个 else if，一个模型一族
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "\\n" << message->content << "<|im_end|>\\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7 || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN) {`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">messages</span><span class="arrow">-></span>
        <span class="chip c">模板分支拼串</span><span class="arrow">-></span>
        <span class="chip b">prompt 文本</span><span class="arrow">-></span>
        <span class="chip d">分词</span><span class="arrow">-></span>
        <span class="chip e">inp_tokens</span>
      </div>
      <div class="row wrap" id="s9cards" style="gap:8px"></div>
      <div class="formula" id="s9msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#s9cards');
    const defs = [
      { c: 'a', t: '名字表 54 项', b: 'LLM_CHAT_TEMPLATES（28-83 行）<br>从 chatml 到 solar-open。' },
      { c: 'c', t: 'detect 兜底', b: '名字查不到时，按模板字符串里的<br>子串特征猜（89 行起）。' },
      { c: 'b', t: 'add_ass', b: '为真时补一个 assistant 头，<br>让模型从这里接着生成。' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#s9msg');
    const texts = [
      '模板的输入是消息数组，输出是一段字符串；它<b>不产生任何算子</b>。',
      '<span class="v">llm_chat_apply_template</span> 从头到尾就是一条 <span class="k">if / else if</span> 链，' +
      '每个内置模板一个分支。',
      '以 <span class="k">chatml</span> 为例：每条消息拼成 ' +
      '<span class="v">im_start + role + 内容 + im_end</span>，用字符串流一路写下去。',
      '<span class="v">add_ass</span> 为真时再补一个 assistant 头 —— 这就是"让模型接着说"的那一步。',
      '模板决定了图上看到什么 token：这段字符串被分词后就是 ' +
      '<span class="v">inp_tokens</span>（回顾 L2-06/L2-07 的图输入）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表 */
{
  kicker: "L2-09 · 收束",
  title: "把这一课压成一张表",
  sub: "量化与导出改的是\"文件里的东西\"，适配器与约束改的是\"图与采样\"——后者一个权重字节都不碰。",
  caption: "下一课 L2-10：模型家族（一）多模态 —— 视觉编码器的输出怎么接进同一张图。",
  src: "src/llama-quant.cpp",
  mark: [3, 6, 7, 11, 12],
  lineNo: 1365,
  code: `llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.max_buf_size                =*/ LLAMA_QUANT_MAX_BUF_SIZE
    };

    return result;
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="s10tbl"></div><div id="s10ex"></div><div class="formula" id="s10msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['文件', '它动的是什么', '关键函数'],
      [['llama-quant.cpp/.h', '权重的数值与 type', 'llama_model_quantize_impl'],
       ['llama-model-saver.cpp/.h', 'GGUF 容器', 'add_kv / add_tensor / save'],
       ['llama-adapter.cpp/.h', '名字 -> (a, b) 映射', 'get_weight / get_scale'],
       ['llama-graph.cpp（追加）', '图本身：多插矩阵乘', 'build_lora_mm'],
       ['llama-grammar.cpp/.h', '采样前的 logits', 'llama_grammar_apply_impl'],
       ['llama-chat.cpp/.h', 'prompt 文本', 'llm_chat_apply_template']],
      { monoCols: [2] });
    wrap.querySelector('#s10tbl').appendChild(t.el);

    wrap.querySelector('#s10ex').appendChild(W.exercise(
      'LoRA 是在哪一步被加进图的？基座权重张量有没有被改写？',
      '在<b>建图那一步</b>（llama-context.cpp:1425 的 <span class="mono">model.build_graph(gparams)</span>；' +
      '图能复用时不会重建），由 <span class="mono">llm_graph_context::build_lora_mm()</span> ' +
      '（<span class="mono">src/llama-graph.cpp:1514</span>）完成。<br>' +
      '它先做正常的 <span class="mono">ggml_mul_mat(ctx0, w, cur)</span>，再对 ' +
      '<span class="mono">*loras</span> 里每一个能查到该权重的 adapter 追加 ' +
      '<span class="mono">ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur))</span> -> ' +
      '<span class="mono">ggml_scale</span> -> <span class="mono">ggml_add</span>。<br>' +
      '基座权重 <span class="mono">w</span> 只被读，没有被改写；所以同一个基座能同时挂多套 adapter，' +
      '代价是图上多出节点（源码注释：每命中一个权重 6 个）。'));

    const msg = wrap.querySelector('#s10msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '十个文件，按"动什么"分成三组。',
      '<span class="k">量化 + 导出</span>动的是<b>文件</b>：数值、类型、容器。',
      '<span class="k">适配器</span>动的是<b>图</b>：权重不变，节点变多。',
      '<span class="k">grammar + 对话模板</span>动的是<b>图和采样之外</b>的两端：logits 与 prompt 文本。',
      '一句话：<span class="v">LoRA 是图上的加法，不是权重里的改写</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
    }));
    tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
