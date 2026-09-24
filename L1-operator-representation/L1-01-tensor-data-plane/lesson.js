/* ==========================================================================
   L1-01 · ggml 张量：算子的数据面
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一个算子，两个面 */
{
  kicker: "L1 · 算子的表示",
  title: "一个算子，两个面",
  sub: "数据面决定\"算什么形状、什么精度、数据在哪\"；身份面决定\"做什么运算\"。",
  caption: "下一课 L1-02 讲身份面（GGML_OP_* 枚举与参数个数表）。",
  src: "ggml/include/ggml.h",
  mark: [],
  lineNo: 684,
  code: `    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type;`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">模型定义</span><span class="arrow">-></span>
        <span class="chip a">图上节点</span><span class="arrow">-></span>
        <span class="chip b">后端内核</span>
      </div>
      <div class="row center" id="faces" style="gap:11px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '数据面 · struct ggml_tensor', b: '形状 ne[]、步长 nb[]、类型 type、<br>数据指针 data、所在 buffer', m: 'struct ggml_tensor' },
      { c: 'c', t: '身份面 · enum ggml_op', b: '这个节点做什么运算：<br>MUL_MAT / RMS_NORM / SOFT_MAX ...', m: 'enum ggml_op op;' },
      { c: 'b', t: '执行面（后面的层）', b: '调度器把它切给某个后端，<br>后端用自己的 kernel 实现它', m: 'ggml_backend_graph_compute' }
    ];
    const host = wrap.querySelector('#faces');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.33');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个 ggml 张量 = <span class="k">形状</span> + <span class="k">类型</span> + <span class="k">一个 op</span>。它就是图上的节点。',
      '数据面回答：<span class="v">ne[]</span> 多少元素、<span class="v">nb[]</span> 怎么跳、<span class="v">type</span> 什么精度、<span class="v">data</span> 在哪。',
      '身份面回答：这个节点是 <span class="v">GGML_OP_MUL_MAT</span> 还是 <span class="v">GGML_OP_RMS_NORM</span>。',
      '后面的课：L2 讲节点怎么被建出来，L4 讲它怎么被切给后端，L5/L6/L7 讲内核怎么实现它。'
    ];
    defs.forEach((_, i) => tl.at(600 + i * 3300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.33'; });
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 2 结构体的形状被几个 <span class="hl-c">宏</span>钉死 */
{
  kicker: "L1-01 · 常量",
  title: "结构体的形状被几个 <span class=\"hl-c\">宏</span>钉死",
  sub: "这些上限不是随手取的：它们直接决定 struct 的布局，也决定后端内核能假设什么。",
  caption: "GGML_MAX_OP_PARAMS = 64 字节，即最多 16 个 int32 参数。",
  src: "ggml/include/ggml.h",
  mark: [0, 2, 4],
  lineNo: 222,
  code: `#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         2048
#define GGML_MAX_SRC            10
#define GGML_MAX_N_THREADS      512
#define GGML_MAX_OP_PARAMS      64`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { k: 'GGML_MAX_DIMS', v: '4', c: 'a', b: '最多 4 维。ne[]/nb[] 定长数组，<br>不堆分配 -> 结构体大小固定。' },
      { k: 'GGML_MAX_PARAMS', v: '2048', c: 'b', b: '算子的参数个数上限<br>（如 CONV 的 kernel 尺寸）。' },
      { k: 'GGML_MAX_SRC', v: '10', c: 'c', b: '一个节点的输入张量上限。<br>src[] 定长指针数组。' },
      { k: 'GGML_MAX_OP_PARAMS', v: '64', c: 'd', b: 'op_params 的字节数 = <br>16 个 int32（备注写明按 int32 对齐）。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.el('div', { class: 'card' });
      e.style.borderLeftColor = 'var(--' + d.c + ')';
      e.style.width = '163px';
      e.innerHTML = '<div class="cm" style="margin:0 0 2px">' + U.esc(d.k) + '</div>' +
        '<div class="ct" style="color:var(--' + d.c + ');font-size:16px">' + d.v + '</div>' +
        '<div class="cb">' + d.b + '</div>';
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看四个上限。它们出现在代码最前面，因为 <span class="k">struct 的字段都是定长数组</span>。',
      '<span class="v">GGML_MAX_DIMS = 4</span>：所以 <span class="k">ne[4]</span> / <span class="k">nb[4]</span> 是定长的，不是指针。<br>这意味着任何算子内核都能假设"最多 4 维"，不用处理任意秩。',
      '<span class="v">GGML_MAX_SRC = 10</span>：所以 <span class="k">src[10]</span> 装得下一个节点的全部输入。<br>图的边就是这些指针，不需要单独的边表。',
      '<span class="v">GGML_MAX_OP_PARAMS = 64</span>：所以 <span class="k">op_params[16]</span>（int32）。<br>算子的标量参数（轴、eps、scale）都塞在这 64 字节里。',
      '结论：<span class="k">ggml_tensor 是一个定长、可 memcpy、可放进 arena 的值类型</span>。<br>这一点在 L1-05（context 用 arena）和 L1-03（图遍历）里会反复用到。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, [i === 0 ? 0 : (i === 1 ? 2 : (i === 2 ? 4 : -1))].filter(x => x >= 0));
    }));
    tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [0, 2, 4]); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ <span class="hl-a">struct ggml_tensor</span> 逐字段 */
{
  kicker: "L1-01 · 核心",
  title: "★ <span class=\"hl-a\">struct ggml_tensor</span> 逐字段",
  sub: "这就是\"算子的数据面\"的全部。后面所有后端都在读同样这几个字段。",
  caption: "注意：结构体里同时有 src[]（输入）和 op（运算）—— 节点与边都在这一个结构里。",
  src: "ggml/include/ggml.h",
  mark: [1, 3, 5, 6, 12, 14, 17, 20, 23, 25, 27],
  lineNo: 685,
  code: `    struct ggml_tensor {
        enum ggml_type type;

//>> type 决定 nb[0] 与量化块大小，是后端选 kernel 的第一依据
        struct ggml_backend_buffer * buffer;

//>> buffer 指向数据实际所在的后端缓冲；NULL 表示还没分配
        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
//>> op_params：算子的标量参数。按 int32 对齐存放，共 16 个
        enum ggml_op op;

//>> flags：见 GGML_TENSOR_FLAG_*（如 OUTPUT / PARAM / LOSS）
        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

//>> view_src + view_offs：视图算子（VIEW/RESHAPE/TRANSPOSE）不拥有数据
        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
//>> name：给调试和 graph 打印用，长度 GGML_MAX_NAME
        size_t               view_offs;

//>> extra：后端私有数据，例如 ggml-cuda.cu 的额外信息
        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '身份/精度', m: 'type · op · op_params[16] · flags', b: '它是什么类型、做什么运算、参数是什么' },
      { c: 'b', t: '形状', m: 'ne[4]', b: '每一维有多少元素' },
      { c: 'c', t: '步长', m: 'nb[4]', b: '每一维跨多少字节；与 type 绑定' },
      { c: 'd', t: '位置', m: 'buffer · data · extra', b: '数据在哪个后端缓冲、指针是多少' },
      { c: 'e', t: '连接', m: 'src[10]', b: '输入张量 —— 这就是图的边' },
      { c: 'f', t: '视图', m: 'view_src · view_offs', b: '不拥有数据时，指向源张量 + 偏移' },
      { c: 'g', t: '调试', m: 'name[64]', b: '图上打印、报错定位用' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => {
      const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m }, { style: 'width:218px' });
      host.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先按"用途"把 12 个字段分成 <span class="k">7 组</span>，再逐个看源码。',
      '<span class="v">type / op / op_params / flags</span>：一个节点的"是什么"。<br>op 是身份，type 是精度，op_params 是标量参数。',
      '<span class="v">ne[4]</span>：number of elements。<span class="k">ne[0] 是最内层（连续）维度</span>。<br>ggml 的约定是"列主序"式：ne[0] 变化最快。',
      '<span class="v">nb[4]</span>：stride in bytes。这是全课最关键的一处约定 —— 下一幕专门讲。',
      '<span class="v">buffer / data / extra</span>：<span class="k">算法与内存解耦</span>。<br>同一个 op，数据可以在 CPU 的 malloc 上，也可以在 CUDA 显存里。',
      '<span class="v">src[10]</span>：输入张量指针数组。<span class="k">图的边不需要额外结构</span> —— 就是这些指针。',
      '<span class="v">view_src / view_offs</span>：VIEW / RESHAPE / TRANSPOSE 这类算子<br><span class="k">不复制数据</span>，只是换一个"看"的方式。',
      '这正是 L4-01（ggml-alloc）能省内存的原因：<br>视图不占新内存，只是换个偏移。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(25200, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = texts[7];
    });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-c">ne[] / nb[]</span>：形状与步长 */
{
  kicker: "L1-01 · 核心",
  title: "★ <span class=\"hl-c\">ne[] / nb[]</span>：形状与步长",
  sub: "源码注释里已经把公式写死了。看懂这三行，就看懂了 ggml 的内存布局约定。",
  caption: "padding 来自 GGML_MEM_ALIGN 对齐；量化类型下 ne[0] 必须是 blck_size 的整数倍。",
  src: "ggml/include/ggml.h",
  mark: [1, 2, 3, 4],
  lineNo: 690,
  code: `        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="calc" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const calc = wrap.querySelector('#calc');
    const right = wrap.querySelector('#right');

    calc.innerHTML = '<div class="cm" style="margin-bottom:4px">一个 Q4_0 张量：ne = [256, 3, 1, 1]</div>';
    const rows = [
      { n: 'ggml_blck_size(Q4_0)', v: '32', c: 'a' },
      { n: 'ggml_type_size(Q4_0)', v: '18 字节/块', c: 'b' },
      { n: 'nb[0]', v: '18', c: 'c' },
      { n: 'nb[1]', v: '18 * (256/32) + pad = 144', c: 'd' },
      { n: 'nb[2]', v: '144 * 3 = 432', c: 'e' },
      { n: 'nb[3]', v: '432 * 1 = 432', c: 'f' }
    ];
    const els = rows.map(r => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
      e.innerHTML = '<span class="m">' + U.esc(r.n) + '</span> = <span style="color:var(--' + r.c + ')">' + U.esc(r.v) + '</span>';
      calc.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');

    right.innerHTML = '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">为什么 nb[0] 不是 4 字节？</div>' +
      '<div class="cb">Q4_0 是<b>块量化</b>：32 个权重打包成一块（1 个 scale + 16 字节 4-bit）。' +
      '所以一"列"跨 18 字节，而不是元素大小 × 数量。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">为什么视角上"ne[0] 是最内层"？</div>' +
      '<div class="cb">因为 nb[0] 最小、ne[0] 变化最快。矩阵乘内核沿 ne[0] 连续读，' +
      '这正是 L5-03 里 <span class="cm" style="margin:0">ggml_vec_dot_*</span> 的前提。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '源码把三条不变量写在注释里。逐条推一遍：',
      '<span class="v">nb[0] = ggml_type_size(type)</span><br>注意：是 <b>type_size</b>（一"块"的字节数），不是元素大小。',
      '<span class="v">nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding</span><br>先按块算出一行有多少字节，再对齐。',
      '<span class="v">nb[i] = nb[i-1] * ne[i-1]</span><br>更高维就是逐维累乘 —— 所以是连续的、无孔洞的布局。',
      '这个布局让内核只需知道 <span class="k">nb[0]</span> 就能连续扫描一行，' +
      '也是"行大小"这个概念（ggml_row_size）的出处。'
    ];
    let t = 700;
    els.forEach((e, i) => { tl.at(t, () => {
      els.forEach((x, k) => x.style.opacity = k <= i ? '1' : '.28');
      msg.innerHTML = texts[Math.min(i, 2)];
    }); t += 3000; });
    tl.at(t, () => { msg.innerHTML = texts[3]; U.markLines(document, [3, 4]); });
    tl.at(t + 3000, () => { msg.innerHTML = texts[4]; U.markLines(document, [1, 2]); });
    tl.at(t + 6000, () => { msg.innerHTML = '一句话：<span class="k">nb[] 是 type 的函数</span> —— 换类型就换步长，换步长就换内核。' ; U.markLines(document, []); });
  }
},

/* ------------------------------------------------------ 5 <span class="hl-b">enum ggml_type</span>：精度谱系 */
{
  kicker: "L1-01 · 类型系统",
  title: "<span class=\"hl-b\">enum ggml_type</span>：精度谱系",
  sub: "F32/F16/BF16 是浮点，Q*/IQ*/TQ* 是块量化，MXFP4/NVFP4 是新的 microscaling 格式。",
  caption: "GGML_TYPE_COUNT = 43，且中间有编号空洞——注释说明了原因：删掉的类型不能复用编号。",
  src: "ggml/include/ggml.h",
  mark: [1, 13, 16, 41, 43],
  lineNo: 388,
  code: `    // NOTE: always add types at the end of the enum to keep backward compatibility
    enum ggml_type {
        GGML_TYPE_F32     = 0,
//>> 枚举从 0 开始，显式赋值 —— 这些数字会写进 GGUF 文件，不能随便变
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0    = 41,
//>> 注释是硬约束：只能在末尾追加，否则旧 GGUF 文件会读错类型
        GGML_TYPE_Q2_0    = 42,
        GGML_TYPE_COUNT   = 43,
    };`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="bars" style="gap:6px"></div>
        <div class="col grow" id="notes" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const groups = [
      { l: '浮点 F32/F16/BF16/F64', n: 4, c: 'a', d: '每元素独立表示，无块' },
      { l: '传统块量化 Q4_0..Q8_K', n: 8, c: 'b', d: 'Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K..Q6_K' },
      { l: '重要性矩阵量化 IQ*', n: 8, c: 'c', d: 'IQ2_XXS..IQ4_XS，需要 imatrix 校准' },
      { l: '整数 I8/I16/I32/I64', n: 4, c: 'd', d: '量化中间表示与索引' },
      { l: '三值/新型格式 TQ*/MXFP4/NVFP4/Q1_0/Q2_0', n: 7, c: 'e', d: 'TQ1_0/TQ2_0 为三值；MXFP4/NVFP4 为 microscaling' }
    ];
    const host = wrap.querySelector('#bars');
    const bars = groups.map(g => {
      const b = U.bar(g.l + '  (' + g.n + ')', g.c);
      host.appendChild(b.el);
      return b;
    });
    const notes = wrap.querySelector('#notes');
    notes.innerHTML =
      '<div class="card" style="border-left-color:var(--g)">' +
      '<div class="ct" style="color:var(--g)">编号只能追加，不能重排</div>' +
      '<div class="cb">源码注释：<span class="cm" style="margin:0">always add types at the end of the enum to keep backward compatibility</span>。' +
      '这些数字直接写进 GGUF，重排会让所有旧模型读错。</div></div>' +
      '<div class="card" style="border-left-color:var(--d)">' +
      '<div class="ct" style="color:var(--d)">为什么有编号空洞？</div>' +
      '<div class="cb">源码里被注释掉的 <span class="cm" style="margin:0">GGML_TYPE_Q4_2 = 4</span> 等，' +
      '说明支持被移除了，但编号<b>不允许回收</b>。</div></div>' +
      '<div class="card" style="border-left-color:var(--f)">' +
      '<div class="ct" style="color:var(--f)">嵌套在类型里的第二个维度</div>' +
      '<div class="cb">旧式的 <span class="cm" style="margin:0">Q4_0_4_4</span> 把 GPU 的 4x4 交错布局编进了类型号；' +
      '现在改由后端的 <b>repack</b> 处理（见 L5-04）。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '43 个类型，先按家族分五组看。',
      '<span class="v">F32 / F16 / BF16 / F64</span>：无块，<span class="k">blck_size = 1</span>。',
      '<span class="v">Q4_0 家族</span>：<span class="k">blck_size = 32</span>。一个块 = 1 个 scale + 32 个低位宽权重。',
      '<span class="v">IQ* 家族</span>：需要重要性矩阵（imatrix）校准，<span class="k">推理时反量化代价更高</span>，' +
      '但同码率下质量更好。',
      '<span class="v">MXFP4 / NVFP4</span>：microscaling 格式，块更小、带 E4M3 之类的小指数 scale。',
      'L1-04 会把这些块的内存布局逐个画出来 —— 那一课解释"类型号如何变成 nb[0]"。'
    ];
    tl.at(700, () => { bars.forEach((b, i) => { b.fill.style.width = (18 + i * 8) + '%'; b.val.textContent = groups[i].n; }); msg.innerHTML = texts[0]; });
    groups.forEach((g, i) => tl.at(3300 + i * 2900, () => {
      bars.forEach((b, k) => { b.el.style.opacity = k === i ? '1' : '.35'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(18200, () => { bars.forEach(b => { b.el.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 三个度量函数：<span class="hl-a">blck_size</span> / <span class="hl-a">type_size</span> / <span class="hl-a">row_size</span> */
{
  kicker: "L1-01 · 类型系统",
  title: "三个度量函数：<span class=\"hl-a\">blck_size</span> / <span class=\"hl-a\">type_size</span> / <span class=\"hl-a\">row_size</span>",
  sub: "所有内核和分配器都靠这三个函数把\"类型 + 形状\"换算成字节数。",
  caption: "ggml_type_sizef 已标记 GGML_DEPRECATED，注释要求改用 ggml_row_size()。",
  src: "ggml/include/ggml.h",
  mark: [0, 1, 2],
  lineNo: 759,
  code: `    GGML_API int64_t ggml_blck_size(enum ggml_type type);
    GGML_API size_t  ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
    GGML_API size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'ggml_blck_size(type)', m: 'int64_t', b: '一个块含多少元素。<br>F32 -> 1；Q4_0 -> 32；Q4_K -> 256。' },
      { c: 'b', t: 'ggml_type_size(type)', m: 'size_t', b: '一个块占多少字节。<br>源码注释：size in bytes for all elements in a block。' },
      { c: 'c', t: 'ggml_row_size(type, ne)', m: 'size_t', b: '一行（ne[0] 个元素）占多少字节。<br>内核扫描时用的就是它。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:224px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '三个函数，一个换算链：',
      '<span class="v">blck_size</span>：<span class="k">元素 -> 块</span>。它把"多少个数"变成"多少块"。',
      '<span class="v">type_size</span>：<span class="k">块 -> 字节</span>。它和 blck_size 一起定义了类型的"密度"。',
      '<span class="v">row_size</span>：<span class="k">形状 -> 字节</span>。内核按行扫描，所以最常用的就是它。',
      '它们共同支撑了上一幕的 <span class="k">nb[] 公式</span>：<br><span class="v">nb[0] = type_size</span>，<span class="v">nb[1] ≈ row_size</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      U.markLines(document, [i]);
      msg.innerHTML = texts[i];
    }));
    tl.at(13500, () => { els.forEach(e => e.style.opacity = '1'); U.markLines(document, [0, 1, 2]); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-e">view_src</span> / <span class="hl-e">view_offs</span>：不拥有数据的算子 */
{
  kicker: "L1-01 · 视图",
  title: "<span class=\"hl-e\">view_src</span> / <span class=\"hl-e\">view_offs</span>：不拥有数据的算子",
  sub: "RESHAPE、VIEW、TRANSPOSE、PERMUTE 这些\"算子\"不产生新数据，只是换一个读法。",
  caption: "L4-01 会看到：分配器正是靠 view_src 判断\"这个张量不需要新内存\"。",
  src: "ggml/include/ggml.h",
  mark: [1, 2, 4],
  lineNo: 706,
  code: `        // source tensor and offset for views
//>> view_src 非 NULL 时，这个张量是"视图"——数据在别人那里
        struct ggml_tensor * view_src;
//>> view_offs 是相对 view_src->data 的字节偏移
        size_t               view_offs;

        void * data;`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:12px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const dia = wrap.querySelector('#dia');

    const base = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
    base.innerHTML = '<div class="ct" style="color:var(--a)">基张量 base</div>' +
      '<div class="cb">ne = [8, 4]，拥有 data 与 buffer<br>' +
      '<span class="cm" style="margin:0">view_src = NULL</span></div>' +
      '<div style="display:flex;gap:2px;margin-top:6px" id="cells"></div>';
    const view = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--e)' });
    view.innerHTML = '<div class="ct" style="color:var(--e)">视图 v = ggml_view_2d(...)</div>' +
      '<div class="cb">ne = [4, 2]，<b>不分配新内存</b><br>' +
      '<span class="cm" style="margin:0">view_src = base</span><br>' +
      '<span class="cm" style="margin:0">view_offs = 2 * 4 字节</span></div>';
    dia.appendChild(base); dia.appendChild(U.arrow('->')); dia.appendChild(view);

    const cells = base.querySelector('#cells');
    const cs = [];
    for (let i = 0; i < 8; i++) {
      const c = U.el('div', { style: 'width:22px;height:16px;border:1px solid var(--border);border-radius:2px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
      c.textContent = i;
      cells.appendChild(c); cs.push(c);
    }

    const msg = wrap.querySelector('#msg');
    const texts = [
      '视图不复制数据：<span class="k">view_src 指向源张量，view_offs 是偏移</span>。',
      '视图自己的 <span class="v">data</span> 字段仍然会被算出来（= view_src->data + view_offs），' +
      '但内核读的还是同一块显存/内存。',
      '<span class="k">这就是 ggml 能做零拷贝 RESHAPE 的原因。</span>L4-01 的分配器会检查 view_src：' +
      '非空就跳过分配。',
      '代价：视图的 <span class="v">nb[]</span> 可能不再满足 <span class="v">nb[0] = type_size</span> 的默认约定 —— ' +
      '所以内核不能假设"连续"，必须用 <span class="v">nb[]</span> 走（见 L5-03）。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 2]); });
    tl.at(3600, () => {
      [0, 1, 2, 3].forEach(i => { cs[i].style.background = 'rgba(88,166,255,.22)'; cs[i].style.borderColor = 'var(--a)'; });
      msg.innerHTML = texts[1];
    });
    tl.at(6800, () => {
      [2, 3, 4, 5].forEach(i => { cs[i].style.background = 'rgba(247,120,186,.22)'; cs[i].style.borderColor = 'var(--e)'; });
      msg.innerHTML = texts[2]; U.markLines(document, [0]);
    });
    tl.at(10200, () => { msg.innerHTML = texts[3]; U.markLines(document, [4]); });
  }
},

/* ------------------------------------------------------ 8 把这一课压成一张表 */
{
  kicker: "L1-01 · 收束",
  title: "把这一课压成一张表",
  sub: "数据面 = 4 组字段。记住它们，后面 51 课都在读写这几组。",
  caption: "下一课 L1-02 讲身份面：enum ggml_op 与参数个数表。",
  src: "ggml/include/ggml.h",
  mark: [0, 1, 2, 3, 4],
  lineNo: 685,
  code: `    struct ggml_tensor {
        enum ggml_type type;

        struct ggml_backend_buffer * buffer;
`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['字段组', '字段', '谁在读它', '在哪一课展开'],
      [['身份/精度', 'type, op, op_params, flags', '后端选 kernel、图打印', 'L1-02 / L5-01'],
       ['形状', 'ne[4]', '所有内核、分配器', '本课 / L4-01'],
       ['步长', 'nb[4]', '所有内核的数据访问', '本课 / L5-03'],
       ['位置', 'buffer, data, extra', '调度器、后端 buffer', 'L4-02 / L4-03'],
       ['连接', 'src[10]', '图遍历、拓扑排序', 'L1-03'],
       ['视图', 'view_src, view_offs', '分配器（跳过分配）', 'L1-01 / L4-01'],
       ['调试', 'name[64]', '报错定位、graph dump', 'L1-03']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '给定 <span class="mono">ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 512, 8)</span>，' +
      '<span class="mono">nb[0]</span> 和 <span class="mono">nb[1]</span> 各是多少？',
      'Q4_K 的 <span class="mono">blck_size = 256</span>、<span class="mono">type_size = 144</span> 字节。<br>' +
      '<span class="mono">nb[0] = 144</span>。<br>' +
      '<span class="mono">nb[1] = 144 * (512 / 256) = 288</span>（若 288 已满足 GGML_MEM_ALIGN 对齐则无额外 padding）。<br>' +
      '验算：这个张量共 512*8 = 4096 个元素 = 16 块，16 * 144 = 2304 字节 = 288 * 8。一致。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '六组字段，各自的读者不同：',
      '<span class="k">身份/精度</span> 组决定后端选哪个 kernel —— 见 L5-01 的大 switch。',
      '<span class="k">形状 + 步长</span> 组是所有数据访问的基础，也是量化块约定生效的地方。',
      '<span class="k">位置</span> 组把算法和内存解耦：同一个 op 可以跑在 CPU 内存或显存上。',
      '<span class="k">连接 + 视图</span> 组让"图"和"零拷贝"成为可能。',
      '记住一句话：<span class="v">ggml_tensor 是值类型，定长、可复制、可放进 arena</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(15500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

];
