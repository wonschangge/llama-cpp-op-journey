/* ==========================================================================
   L2-01 · llama.h 公共 API 全景
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 <span class="hl-a">一个头文件</span>，四个文件的分工 */
{
  kicker: "L2 · 从模型到图",
  title: "<span class=\"hl-a\">一个头文件</span>，四个文件的分工",
  sub: "公共 API 的边界在哪里：契约、实现入口、C++ 包装、未定型的扩展。",
  caption: "本课覆盖 include/llama.h、include/llama-cpp.h、src/llama.cpp、src/llama-ext.h 四个文件，全部计入覆盖率。",
  src: "include/llama.h",
  mark: [10, 11, 12, 13, 16],
  lineNo: 51,
  code: `#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;
//>> 四个不透明类型的唯一公共声明处：调用方只拿得到指针，看不到任何字段

    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">权重文件 GGUF</span><span class="arrow">-></span>
        <span class="chip a">公共 API 契约</span><span class="arrow">-></span>
        <span class="chip b">内部模块实现</span><span class="arrow">-></span>
        <span class="chip d">待执行的图</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    const defs = [
      { c: 'a', t: 'include/llama.h',   m: '1646 行', b: '唯一对外契约<br>所有 llama_* 在这里声明' },
      { c: 'b', t: 'src/llama.cpp',     m: '620 行',  b: '契约的实现入口<br>初始化、加载、装配' },
      { c: 'c', t: 'include/llama-cpp.h', m: '30 行', b: '只给 C++ 用<br>智能指针 RAII 包装' },
      { c: 'd', t: 'src/llama-ext.h',   m: '134 行',  b: 'staging：尚未定型<br>允许破坏性变更' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.card(d, { style: 'width:163px;flex:0 0 auto' });
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');
    hi([]);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '从一个 GGUF 文件到一张待执行的图，要穿过 <span class="k">四层</span>。第一层就是公共 API 契约。',
      '<span class="v">include/llama.h</span>：1646 行，241 处 <span class="k">LLAMA_API</span> 声明。<br>它<b>只有声明</b>，一个函数体都没有 —— 这就是"契约"。',
      '<span class="v">src/llama.cpp</span>：只有 620 行，却 include 了全部内部模块。<br>它的角色是"接线"，第 6 幕专门讲。',
      '<span class="v">include/llama-cpp.h</span>：30 行，开头就 <span class="k">#error</span> 拒绝 C。<br>它把 C 的所有权规则翻译成 C++ 的 RAII。',
      '<span class="v">src/llama-ext.h</span>：自称 staging 头，允许破坏性变更。<br>新 API 先在这里长出来，稳定后再搬进 llama.h。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => {
      els.forEach(e => e.style.opacity = '1');
      hi('all');
      msg.innerHTML = '右边这段就是契约的开头：<span class="k">四个不透明类型</span>，加上一个 memory 句柄。<br>'
        + '有了它们，内部结构体怎么改都不会破坏调用方。';
    });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">三个不透明指针</span>，三段生命周期 */
{
  kicker: "L2-01 · 三段式",
  title: "★ <span class=\"hl-a\">三个不透明指针</span>，三段生命周期",
  sub: "模型 / 上下文 / 采样器各有独立的创建与释放函数；默认值由函数给出，不写在字段里。",
  caption: "上下文参数里的 n_ctx / n_batch 只是\"请求值\"，实际值要用 llama_n_ctx(ctx) 一族查询（llama.h:567-575）。",
  src: "include/llama.h",
  mark: [2, 3, 4, 5],
  lineNo: 472,
  code: `    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLAMA_API struct llama_model_params          llama_model_default_params(void);
    LLAMA_API struct llama_context_params        llama_context_default_params(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row" id="cards" style="gap:8px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    const defs = [
      { c: 'a', t: 'struct llama_model', b: '权重 + 超参 + 词表<br>只读，可被多个上下文共享',
        m: '建 517 · 释 542' },
      { c: 'c', t: 'struct llama_context', b: '一次推理的运行时状态<br>memory / KV cache / 后端调度器',
        m: '建 544 · 释 554' },
      { c: 'b', t: 'struct llama_sampler', b: '采样链，在 CPU 上跑<br>生命周期不依附上下文',
        m: '建 1355 · 释 1350' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.card(d, { style: 'width:219px;flex:0 0 auto' });
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');
    hi([]);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '三段式：<span class="k">模型</span> → <span class="k">上下文</span> → <span class="k">采样器</span>。三者生命周期互相独立。',
      '<span class="v">struct llama_model</span>：从文件读出来的东西，<b>只读</b>。<br>建：<span class="m">llama_model_load_from_file</span>（517）；释：<span class="m">llama_model_free</span>（542）。',
      '<span class="v">struct llama_context</span>：一次推理的全部运行时状态。<br>建：<span class="m">llama_init_from_model</span>（544）；释：<span class="m">llama_free</span>（554）—— 释放函数不叫 llama_context_free。',
      '<span class="v">struct llama_sampler</span>：采样链。<br>建：<span class="m">llama_sampler_chain_init</span>（1355）；释：<span class="m">llama_sampler_free</span>（1350）。',
      '每个类都给一个 <span class="k">default_params</span> 函数（右边四行），而不是在头文件里写字段默认值 ——<br>默认值由实现决定，可以随版本变；字段布局却保持稳定。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      hi([i]);
      msg.innerHTML = texts[i];
    }));
    tl.at(14300, () => {
      els.forEach(e => e.style.opacity = '1');
      hi('all');
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 3 参数结构体族：<span class="hl-c">4 个 struct</span> + 4 个默认值函数 */
{
  kicker: "L2-01 · 参数族",
  title: "参数结构体族：<span class=\"hl-c\">4 个 struct</span> + 4 个默认值函数",
  sub: "llama.cpp 不用\"一堆散参数\"，而是\"一个按值传的结构体 + 一个取默认值的函数\"。",
  caption: "结构体按值传（by value），所以里面有注释专门说明：bool 集中放末尾，避免拷贝时错位。",
  src: "include/llama.h",
  mark: [0, 2, 6, 8, 9, 10, 13],
  lineNo: 314,
  code: `    struct llama_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;
//>> devices：NULL 结尾的设备列表；为 NULL 表示用全部可用设备（设备发现见 L3-02）

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs
        enum llama_load_mode  load_mode;  // how to load the model
//>> load_mode：auto / none / mmap / mlock / mmap+mlock / dio —— 决定权重怎么进内存（L2-03）

        enum llama_lazy_mode lazy_mode; // on-demand reading of tensors marked by the arch
//>> lazy_mode：配合 mmap 按需读取张量，而不是一次性全部读进来

        // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    hi('all');

    const t = U.table(
      ['参数结构体', '行', '用在', '代表字段'],
      [['llama_model_params',          '314', '加载模型',   'devices / n_gpu_layers / split_mode / load_mode / vocab_only'],
       ['llama_context_params',        '360', '创建上下文', 'n_ctx / n_batch / n_ubatch / n_seq_max / type_k / type_v'],
       ['llama_model_quantize_params', '434', '量化模型',   'ftype / imatrix / kv_overrides / tt_overrides'],
       ['llama_sampler_chain_params',  '457', '创建采样链', 'no_perf（唯一字段，默认 true）']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '四个结构体，四个"入口参数包"。按声明顺序看：模型怎么进来 → 上下文多大 → 怎么量化 → 采样链要不要计时。',
      '<span class="v">llama_model_params</span>（314）：决定模型怎么被读进来 —— 放哪几张卡、怎么切、要不要 mmap。',
      '<span class="v">llama_context_params</span>（360）：决定上下文多大、批多大、KV cache 用什么精度。<br>它同时是 <span class="k">n_ctx</span> 这类"请求值"的载体，真实值要另外查询（570）。',
      '<span class="v">llama_model_quantize_params</span>（434）与 <span class="v">llama_sampler_chain_params</span>（457）：<br>一个管离线量化，一个只管采样链要不要计时。',
      '一句话：<span class="k">结构体是"输入契约"，默认值函数是"实现说了算"</span>。<br>llama.h:472-477 那四行 default_params 就是这条约定的落点。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3300 + i * 2800, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(15000, () => {
      rows.forEach(x => { x.className = ''; });
      hi('all');
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-a">六步调用顺序</span>：从 backend_init 到 sampler_sample */
{
  kicker: "L2-01 · 核心",
  title: "★ <span class=\"hl-a\">六步调用顺序</span>：从 backend_init 到 sampler_sample",
  sub: "一个用 llama.cpp 的程序，最小骨架就是这六个调用。顺序不能换。",
  caption: "这六处声明散布在 1646 行头文件的六个不同段落 —— 代码区用区间分隔行标出各自的真实位置。",
  src: "include/llama.h",
  mark: [0, 2, 6, 10, 19, 23],
  lineNo: 0,
  code: `    LLAMA_API void llama_backend_init(void);
//>> ---- include/llama.h:517-519 ----
    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
              struct llama_model_params   params);
//>> ---- include/llama.h:544-546 ----
    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);
//>> ---- include/llama.h:1180-1187 ----
    LLAMA_API int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);
//>> ---- include/llama.h:998-1000 ----
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);
//>> ---- include/llama.h:1548-1548 ----
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row" id="r1" style="gap:6px;justify-content:center"></div>
      <div class="row" id="r2" style="gap:6px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    const defs = [
      { c: 'a', t: '1 · llama_backend_init',         b: '初始化时间与后端注册表',     m: 'llama.h:482' },
      { c: 'a', t: '2 · llama_model_load_from_file', b: '读 GGUF，建模型与设备',      m: 'llama.h:517' },
      { c: 'c', t: '3 · llama_init_from_model',      b: '建上下文：memory 与调度器',  m: 'llama.h:544' },
      { c: 'b', t: '4 · llama_tokenize',             b: '文本切成 token id',          m: 'llama.h:1180' },
      { c: 'd', t: '5 · llama_decode',               b: '跑一遍图，产出 logits',      m: 'llama.h:998' },
      { c: 'e', t: '6 · llama_sampler_sample',       b: '从 logits 选出下一个 token', m: 'llama.h:1548' }
    ];
    const r1 = wrap.querySelector('#r1'), r2 = wrap.querySelector('#r2');
    const els = [];
    defs.forEach((d, i) => {
      const host = i < 3 ? r1 : r2;
      if (i % 3 !== 0) host.appendChild(U.arrow('->'));
      const e = U.card(d, { style: 'width:196px;flex:0 0 auto' });
      host.appendChild(e); els.push(e);
    });
    els.forEach(e => e.style.opacity = '.28');
    hi([]);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">llama_backend_init</span>：进程级，只调一次。它注册后端、初始化 f16 表（实现在 src/llama.cpp:122）。',
      '<span class="v">llama_model_load_from_file</span>：把 GGUF 读成 <span class="k">llama_model</span>。<br>权重从哪来？回顾 L1-06 —— 就是那一课拆开的容器文件。',
      '<span class="v">llama_init_from_model</span>：模型只读，上下文才是"这次的运行状态"。<br>KV cache（memory）在这一步建出来 —— 展开在 L2-04。',
      '<span class="v">llama_tokenize</span>：第一个参数是 <span class="m">const struct llama_vocab *</span>，不是 model。<br>词表要用 <span class="m">llama_model_get_vocab(model)</span>（588）取。',
      '<span class="v">llama_decode</span>：把 <span class="m">llama_batch</span> 喂进去。<br>它<b>必须有 memory</b>；返回值 1 = 找不到 KV 槽位，-1 = 输入非法（998 上方的注释写全了）。',
      '<span class="v">llama_sampler_sample</span>：第 5 步产出的 logits 在这里被消费。<br>它是 <span class="m">get_logits_ith → apply → accept</span> 的简写（1541-1546）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      hi([i]);
      msg.innerHTML = texts[i];
    }));
    tl.at(21500, () => {
      els.forEach(e => e.style.opacity = '1');
      hi('all');
      msg.innerHTML = '六步里 <span class="k">只有第 5 步会碰图</span>；第 6 步在 CPU 上、不进图。<br>'
        + '所以"从模型到图"这条主线，真正的主角是第 2 步与第 5 步 —— 见 L2-05、L2-07。';
    });
  }
},

/* ------------------------------------------------------ 5 <span class="hl-b">llama_batch</span>：喂给 decode 的那一张表 */
{
  kicker: "L2-01 · 输入",
  title: "<span class=\"hl-b\">llama_batch</span>：喂给 decode 的那一张表",
  sub: "七个字段全是并行数组，长度都等于 n_tokens；这是 L2 层\"数据进图\"的统一入口。",
  caption: "decode 的注释明写 Requires the context to have a memory（987）；memory 用 llama_get_memory(ctx)（585）取，再用 llama_memory_seq_* 一族操作。",
  src: "include/llama.h",
  mark: [16, 18, 20, 22, 25, 29],
  lineNo: 0,
  code: `    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
//>> embd 与 token 二选一：要么给 token id，要么直接给词向量
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
//>> logits 为 0 的 token 不输出 logits —— 只要最后一个 token 时，其余全填 0
    } llama_batch;
//>> ---- include/llama.h:742-749 ----
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    LLAMA_API void llama_memory_clear(
            llama_memory_t mem,
                      bool data);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 2px">struct llama_batch —— 7 个字段</div>';
    const fields = [
      ['n_tokens',  '本批 token 数；其余数组的长度'],
      ['token',     'token id 数组（与 embd 二选一）'],
      ['embd',      '词向量数组（与 token 二选一）'],
      ['pos',       '每个 token 在序列中的位置'],
      ['n_seq_id',  '每个 token 归属几个序列'],
      ['seq_id',    '每个 token 归属的序列 id 列表'],
      ['logits',    '该 token 是否要输出 logits']
    ];
    const rows = fields.map(f => {
      const e = U.el('div', { class: 'formula', style: 'padding:3px 8px;font-size:10px' });
      e.innerHTML = '<span class="m">' + U.esc(f[0]) + '</span> &nbsp;' + U.esc(f[1]);
      left.appendChild(e);
      return e;
    });
    rows.forEach(e => e.style.opacity = '.30');

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">为什么是"数组的数组"</div>' +
      '<div class="cb">一个 batch 可以同时装多条序列（n_seq_id / seq_id 都是二维），' +
      '所以<b>一次 decode 能并行推进多个请求</b>。切分与 ubatch 见 <b>L2-05</b>。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">decode 需要 memory</div>' +
      '<div class="cb"><span class="cm" style="margin:0">Requires the context to have a memory</span>（987）。' +
      'memory 是 KV cache 之上的抽象：<span class="cm" style="margin:0">llama_get_memory</span>（585）取句柄，' +
      '再用 <span class="cm" style="margin:0">llama_memory_seq_rm</span>（756）按序列增删。展开在 <b>L2-04</b>。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">llama_batch</span> 是一条"记录表"：每列长度都等于 <span class="k">n_tokens</span>。',
      '<span class="v">n_tokens</span> + <span class="v">token</span>：最常见的用法 —— 给一串 token id。',
      '<span class="v">embd</span> 与 <span class="v">token</span> 互斥：要直接喂向量就填 embd，把 token 置 NULL。',
      '<span class="v">pos</span> 为 NULL 时位置自动跟踪；<span class="v">seq_id</span> 为 NULL 时当作序列 0。',
      '<span class="v">n_seq_id</span> / <span class="v">seq_id</span>：把"一个 token 属于哪些序列"表达成二维数组 ——<br>这是多序列并行推理的数据基础（L2-05）。',
      '<span class="v">logits</span> 是唯一的输出开关：<br>为 NULL 且非 embeddings 时，<b>只输出最后一个 token</b>。',
      'batch 与 memory 是 L2 层的两块"输入状态"：一个描述<b>这次算什么</b>，一个描述<b>之前算过什么</b>。'
    ];
    const mk = [[0], [1], [2], [3], [], [], [4]];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3200 + i * 2400, () => {
      rows.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.30'; });
      hi(mk[i]);
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(19000, () => {
      rows.forEach(r => { r.style.opacity = '1'; });
      hi('all');
      msg.innerHTML = texts[6];
    });
  }
},

/* ------------------------------------------------------ 6 ★ <span class="hl-a">src/llama.cpp</span>：620 行的接线板 */
{
  kicker: "L2-01 · 聚合入口",
  title: "★ <span class=\"hl-a\">src/llama.cpp</span>：620 行的接线板",
  sub: "它 include 了所有内部模块，自己却几乎不算数：只做初始化、加载编排、chat 模板与分片路径。",
  caption: "\"聚合入口\"的含义：它是唯一把全部内部模块拉在一起的翻译单元，加载与初始化在这里被翻译成内部调用；六个主调用里只有两个落在这个文件。",
  src: "src/llama.cpp",
  mark: [0, 5, 19, 25, 31, 39, 47, 55],
  lineNo: 0,
  code: `#include "llama.h"

#include "llama-impl.h"
#include "llama-version.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"
//>> ---- src/llama.cpp:34-36 ----
//
// interface implementation
//
//>> ---- src/llama.cpp:316-327 ----
static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.load_mode,
            params.check_tensors, params.no_alloc, params.load_mtp, params.kv_overrides, params.tensor_buft_overrides);

        ml.lazy.mode = params.lazy_mode;

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get());
//>> ---- src/llama.cpp:344-362 ----
        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();
//>> ---- src/llama.cpp:369-369 ----
        if (!model->load_tensors(ml)) {`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">llama_model_loader</span><span class="arrow">-></span>
        <span class="chip c">llama_model_create</span><span class="arrow">-></span>
        <span class="chip d">load_hparams</span><span class="arrow">-></span>
        <span class="chip d">load_vocab</span><span class="arrow">-></span>
        <span class="chip b">load_tensors</span>
      </div>
      <div class="row" id="cards" style="gap:8px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    const defs = [
      { c: 'a', t: '它 include 了一切', b: '9 个内部头 + 4 个 ggml/gguf 头<br>llama-context.h / llama-vocab.h / llama-model.h …',
        m: '第 1-17 行' },
      { c: 'c', t: '它自己只做接线', b: 'backend 初始化、模型加载编排、<br>chat 模板、分片路径',
        m: '// interface implementation' },
      { c: 'b', t: '重活在别的文件', b: 'llama_init_from_model 在 llama-context.cpp:3739<br>llama_tokenize 在 llama-vocab.cpp:4496',
        m: 'llama_sampler_sample 在 llama-sampler.cpp:895' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.card(d, { style: 'width:219px;flex:0 0 auto' });
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');
    hi([]);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这个文件只有 <span class="k">620 行</span>，却 include 了全部内部模块 —— 典型的"聚合入口"。',
      '第 1-17 行是它的身份证明：<span class="v">llama.h</span> + 9 个内部头 + 4 个 ggml/gguf 头。<br>其中 llama-chat.h 与 llama-model-saver.h 在 src/ 下只被各自的实现文件和它 include。',
      '它实现的公共函数只有这几类：<span class="m">llama_backend_init</span>（122）、模型加载（465）、<br><span class="m">llama_chat_apply_template</span>（509）、<span class="m">llama_split_path</span>（544）。',
      '<span class="v">llama_model_load</span>（316）是加载编排：<br>构造 loader → 建模型 → 挑设备 → 读 hparams / vocab / stats → 搬张量。',
      '逐步对照跨课：<span class="k">hparams</span> 是 L2-02 的主题，<span class="k">load_tensors</span> 是 L2-03 的主题。',
      '所以"llama.cpp 是聚合入口"不是修辞：<span class="k">它把 C ABI 的调用翻译成内部模块的调用序列</span>，自己不含任何算子。'
    ];
    const mk = [[0], [1], [2], [3, 4], [5, 6], [7]];
    defs.forEach((_, i) => tl.at(700 + i * 3500, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      hi(mk[Math.min(i, 5)]);
      msg.innerHTML = texts[i];
    }));
    tl.at(18200, () => {
      els.forEach(e => e.style.opacity = '1');
      hi('all');
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-b">llama-cpp.h</span>：把所有权写进类型 */
{
  kicker: "L2-01 · C++ 包装",
  title: "<span class=\"hl-b\">llama-cpp.h</span>：把所有权写进类型",
  sub: "30 行，四个 deleter，四个 unique_ptr 别名 —— C 的\"记得手动释放\"变成 C++ 的\"不可能忘记\"。",
  caption: "注意对称性：创建用哪个函数，就决定了释放用哪个函数；而 context 的释放函数叫 llama_free。",
  src: "include/llama-cpp.h",
  mark: [3, 11, 16, 20, 24, 27, 28, 29, 30],
  lineNo: 1,
  code: `#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "llama.h"

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
//>> 四个 deleter 一一对应 llama.h 里的四个释放函数（542 / 554 / 1350 / 712）
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora * adapter) { llama_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<llama_model, llama_model_deleter> llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
typedef std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter> llama_adapter_lora_ptr;
//>> 拿到 llama_model_ptr，就等于"离开作用域自动 llama_model_free"—— 泄漏在类型层面被排除`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row wrap" id="cards" style="gap:8px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    const defs = [
      { c: 'a', t: 'llama_model_ptr',   b: '模型；<br>按值共享最省事', m: 'llama_model_free' },
      { c: 'c', t: 'llama_context_ptr', b: '上下文；<br>名字与释放函数不对称', m: 'llama_free' },
      { c: 'b', t: 'llama_sampler_ptr', b: '采样器 / 采样链；<br>进链后不再自己释放', m: 'llama_sampler_free' },
      { c: 'd', t: 'llama_adapter_lora_ptr', b: 'LoRA 适配器；<br>寿命不得超过模型', m: 'llama_adapter_lora_free' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.card(d, { style: 'width:163px;flex:0 0 auto' });
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');
    hi([]);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '第一件事是拒绝 C：<span class="k">#ifndef __cplusplus</span> 紧接一个 <span class="k">#error</span>（3-5）。',
      '<span class="v">llama_model_ptr</span>：deleter 调 <span class="m">llama_model_free</span>（llama.h:542）。',
      '<span class="v">llama_context_ptr</span>：deleter 调 <span class="m">llama_free</span>（554）。<br>创建函数叫 <span class="m">llama_init_from_model</span>，释放函数却叫 <span class="m">llama_free</span> —— 公共 API 的历史命名。',
      '<span class="v">llama_sampler_ptr</span>：deleter 调 <span class="m">llama_sampler_free</span>（1350）。<br>但一旦交给 <span class="m">llama_sampler_chain_add</span>（1358），所有权就归链，不能再自己释放。',
      '<span class="v">llama_adapter_lora_ptr</span>：LoRA 的释放函数（712）。<br>llama.h 里那句注释仍然成立：适配器的寿命不能超过模型。',
      '最后四个 <span class="k">typedef</span> 才是这个头的全部产出：<br>C 的所有权约定是文档，C++ 的所有权约定是类型。'
    ];
    const mk = [[0], [1], [2], [3], [4], [5, 6, 7, 8]];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      hi(mk[Math.min(i, 5)]);
      msg.innerHTML = texts[i];
    }));
    tl.at(16200, () => {
      els.forEach(e => e.style.opacity = '1');
      hi('all');
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 8 <span class="hl-c">llama-ext.h</span>：试验场，与它没做到的自律 */
{
  kicker: "L2-01 · 扩展头",
  title: "<span class=\"hl-c\">llama-ext.h</span>：试验场，与它没做到的自律",
  sub: "新 API 先在这里长出来，稳定了再搬进 llama.h；但它开头那句\"尽量别被 include\"已经被打破了。",
  caption: "本课只讲这个文件的角色；它声明的建图与量化 API 分别在 L2-06 / L2-07 与 L2-09 展开。",
  src: "src/llama-ext.h",
  mark: [2, 3, 4, 13, 20, 22],
  lineNo: 1,
  code: `#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase
//>> 实测：src/ 下已有 4 个文件 include 了它（llama-context.h、llama-model.cpp、llama-context.cpp、llama-quant.cpp）

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row wrap" id="cards" style="gap:8px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    const defs = [
      { c: 'a', t: 'llama_graph_reserve', b: '预留一张计算图，直到下次调用前有效。<br>建图入口在这里露头 —— 见 L2-06 / L2-07',
        m: 'llama-ext.h:13' },
      { c: 'c', t: 'llama_ftype_get_default_type', b: '给定 ftype 返回默认 ggml_type。<br>量化类型选择 —— 见 L2-09',
        m: 'llama-ext.h:20' },
      { c: 'd', t: 'quantize_state_impl', b: 'C++ 的 struct，没有不透明包装。<br>所以这个头文件天然只能 C++ 用',
        m: 'llama-ext.h:22' },
      { c: 'g', t: '注释与实测不一致', b: '第 5 行写"尽量别 include 这个头"，<br>但 src/ 下已有 4 处 include',
        m: 'grep 实测' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.card(d, { style: 'width:163px;flex:0 0 auto' });
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');
    hi([]);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '第 3-5 行是这个文件的自我介绍：<span class="k">staging header</span>，允许破坏性变更，一切都算 WIP。',
      '所以读它的方式和读 llama.h <b>不同</b>：<span class="v">llama.h</span> 是承诺，<span class="v">llama-ext.h</span> 是"当前状态"。',
      '<span class="v">llama_graph_reserve</span>（13）是建图入口：给定 n_tokens / n_seqs / n_outputs 预留一张图。<br>这就是"从模型到图"那句主题词在公共 API 上的落点。',
      '<span class="v">quantize_state_impl</span>（22）是个 C++ 类型，连前置声明都直接写着。<br>它和文件后面的 <span class="m">llama_memory_breakdown</span>（84，用 std::map）一起，决定了这个头不能给 C 用。',
      '同一段里还有 <span class="m">llama_memory_breakdown_data</span>（67）：模型 / 上下文 / 计算缓冲各占多少。<br>这是选后端时要看的账（L3-02 / L4-03）。',
      '最后一条要记牢：<span class="k">源码注释不一定是真的</span>。<br>第 5 行的自律，grep 一下就知道没做到 —— 凡结论必实测。'
    ];
    const mk = [[0, 1, 2], [0, 1, 2], [3], [5], [], [2]];
    defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      hi(mk[Math.min(i, 5)]);
      msg.innerHTML = texts[i];
    }));
    tl.at(17200, () => {
      els.forEach(e => e.style.opacity = '1');
      hi('all');
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L2-01 · 收束",
  title: "把这一课压成一张表",
  sub: "头文件自己就写了正确用法：先建链，再循环 decode → sample。",
  caption: "下一课 L2-02 进入架构表与超参：llama_model 里那些 hparams 究竟从 GGUF 的哪个键读出来。",
  src: "include/llama.h",
  mark: [0, 5, 7, 23, 26],
  lineNo: 1243,
  code: `    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const ML = document.querySelectorAll('mark.ln-mark');
    function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
      const all = (ks === 'all');
      ML.forEach((m, i) => {
        const on = all || ks.indexOf(i) >= 0;
        m.className = on ? 'ln-mark on' : 'ln-mark';
        m.style.display = on ? 'inline-block' : 'none';
      });
    }

    hi([]);

    const t = U.table(
      ['文件', '它是什么', '关键内容', '接着看'],
      [['include/llama.h',     '公共 C ABI 契约', '三段式对象 / 参数结构体族 / 六步调用序',    '全课'],
       ['include/llama-cpp.h', 'C++ RAII 包装',   '4 个 deleter + 4 个 unique_ptr 别名',      'common/ 与 tests/'],
       ['src/llama.cpp',       '实现与装配入口',  'backend 初始化 / 模型加载编排 / chat / split', 'L2-02 · L2-03'],
       ['src/llama-ext.h',     'staging 扩展头',  'llama_graph_reserve / llama_quant_*',      'L2-06 · L2-07']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码，把"从加载模型到拿到第一个 token"的调用链按顺序写出来；' +
      '并说出哪一步要求上下文里<b>必须有 memory</b>。',
      '依次是 <span class="mono">llama_backend_init</span> → ' +
      '<span class="mono">llama_model_load_from_file</span> → ' +
      '<span class="mono">llama_init_from_model</span> → ' +
      '<span class="mono">llama_tokenize</span> → ' +
      '<span class="mono">llama_decode</span> → ' +
      '<span class="mono">llama_sampler_sample</span>。<br>' +
      '要求有 memory 的是 <span class="mono">llama_decode</span>（llama.h:986-987 的注释：' +
      '<span class="mono">Requires the context to have a memory</span>）；' +
      '<span class="mono">llama_encode</span> 恰好相反，注释里明写它不用 KV cache（977）。<br>' +
      '批次用 <span class="mono">llama_batch_init</span>（968）准备，用完 ' +
      '<span class="mono">llama_batch_free</span>（974）释放。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '四个文件，四种角色 —— 记住分工，比记住函数名有用。',
      '<span class="k">llama.h</span> 是唯一承诺：其它三个都不该被调用方直接依赖（llama-ext.h 除外，它本来就是给外部工具用的）。',
      '<span class="k">llama-cpp.h</span> 只服务 C++；用 C 写程序就得自己管释放。',
      '<span class="k">src/llama.cpp</span> 是入口，<span class="k">src/llama-ext.h</span> 是出口 —— 新 API 从这里长大。',
      '头文件注释里那段示例（1245-1274）已经把顺序写好了：<span class="v">建链 → decode → sample</span>。<br>本课的验收点就是能默画出这六步。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(13500, () => {
      rows.forEach(x => { x.className = ''; });
      hi('all');
      msg.innerHTML = texts[4];
    });
    tl.at(16800, () => {
      hi([2, 4]);
      msg.innerHTML = '最后一句留给下一课：<span class="k">本课只回答了"谁在承诺"</span>，'
        + '还没回答"模型里到底存了什么"。L2-02 从 hparams 开始拆。';
    });
  }
},

];
