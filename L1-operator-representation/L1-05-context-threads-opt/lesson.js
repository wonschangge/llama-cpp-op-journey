/* ==========================================================================
   L1-05 · 上下文、线程与优化器接口
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 五个文件，一个问题：<span class="hl-a">对象住在哪里</span> */
{
  kicker: "L1 · 算子的表示",
  title: "五个文件，一个问题：<span class=\"hl-a\">对象住在哪里</span>",
  sub: "context 管内存、threading 管并发、ggml-opt 管训练；ggml.cpp 其实是进程级兜底。",
  caption: "回顾 L1-01：ggml_tensor 是定长值类型 —— 正因为定长，它才能被整块塞进 arena。",
  src: "ggml/include/ggml-opt.h",
  mark: [5, 8],
  lineNo: 196,
  code: `    // ====== Intended Usage ======
    //
    // 1. Select the appropriate loss for your problem.
    // 2. Create a dataset and set the data for the "data" tensor. Also set the "labels" tensor if your loss needs them.
    //    Setting the shard size to 1 will be fine, it's the granularity with which data is shuffled/loaded (bigger values are faster).
    // 3. Create a GGML graph for your model with no_alloc == true. Use two separate contexts for the tensors.
//>> no_alloc == true：图里的张量只占"对象头"，数据不从这里出
    //    The first context should contain the model parameters and inputs and be allocated statically in user code.
    //    The second context should contain all other tensors and will be (re)allocated automatically.
//>> 两个 context：一个放参数与输入（静态），一个放其余（自动重分配）
    //    Due to this automated allocation the data of the second context is not defined when accessed in user code.
    //    Note that the second dimension of the inputs/outputs are interpreted as the number of datapoints in those tensors.
    // 4. Call ggml_opt_fit. If you need more control you can use ggml_opt_epoch instead.`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="flow" style="justify-content:center">' +
        '<span class="chip">模型定义</span><span class="arrow">-></span>' +
        '<span class="chip a">ggml_context</span><span class="arrow">-></span>' +
        '<span class="chip b">ggml_cgraph</span><span class="arrow">-></span>' +
        '<span class="chip c">后端执行</span></div>' +
      '<div class="row wrap center" id="faces" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'ggml/src/ggml.c —— 主角', b: 'context 的实现：<br>一块 arena + 一条对象链表', m: 'ggml_init · ggml_new_object' },
      { c: 'b', t: 'ggml-opt.cpp / ggml-opt.h', b: '训练接口：dataset、opt_context、<br>epoch、fit', m: 'ggml_opt_fit' },
      { c: 'f', t: 'ggml-threading.cpp / .h', b: '并发原语：<br>只有一对临界区函数', m: 'ggml_critical_section_start' },
      { c: 'g', t: 'ggml/src/ggml.cpp', b: '与 context 无关：<br>进程级 terminate 兜底', m: 'ggml_uncaught_exception' }
    ];
    const host = wrap.querySelector('#faces');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看四个角色的分工。第一张卡就提醒一件事：<span class="v">ggml_context</span> 的实现<b>不在</b>本课的五个文件里。',
      'arena 的实现在 <span class="v">ggml/src/ggml.c</span> —— 本课额外引用了它，否则"为什么用 arena"讲不出证据。',
      '<span class="v">ggml-opt</span> 是 context 最大的一位客户：<b>它自己同时管四个 context</b>。',
      '<span class="v">ggml-threading</span> 一共 12 行。本课如实讲它提供了什么，也讲它<b>没有</b>提供什么 —— 它不是线程池。',
      '五个文件，三个问题：<b>对象住在哪里、谁能并发碰它、训练时怎么用它</b>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14500, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">struct ggml_context</span>：一块内存 + 一条对象链表 */
{
  kicker: "L1-05 · context",
  title: "★ <span class=\"hl-a\">struct ggml_context</span>：一块内存 + 一条对象链表",
  sub: "七个字段就够了：有多大、在哪、是不是自己的、要不要分配数据、几个对象、头、尾。",
  caption: "示意图：对象头 + 对齐后的负载，从 mem_buffer 头部起一个挨一个往后摆，没有空闲链表。",
  src: "ggml/src/ggml.c",
  mark: [1, 2, 5, 7, 12, 19, 20, 22, 23, 24, 29, 30],
  lineNo: 958,
  code: `struct ggml_object {
    size_t offs;
    size_t size;
//>> offs / size：这个对象在 mem_buffer 里的偏移，与对齐后占用的字节数

    struct ggml_object * next;

    enum ggml_object_type type;

    char padding[4];
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);
//>> GGML_OBJECT_SIZE：对象头本身的字节数，后面每个偏移都要加上它

//
// ggml context
//

struct ggml_context {
    size_t mem_size;
//>> mem_size：这块 arena 一共多少字节，ggml_init 之后不再变
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
//>> no_alloc：true 时只放对象头，张量数据交给后端 buffer（ggml-opt 全程用它）

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
//>> objects_begin / objects_end：链表首尾；end 同时就是下一幕要用的 bump 指针
};`,
  duration: 25000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = '<div class="cm" style="margin:0">mem_buffer —— 一次分配拿到的整块 arena（mem_size 字节）</div>' +
      '<div class="row" id="arena" style="gap:3px;align-items:stretch"></div>' +
      '<div class="row center" id="ptr" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);
    const focus = ls => {
      const ms = document.querySelectorAll('mark.ln-mark');
      let first = null;
      for (let i = 0; i < ms.length; i++) {
        const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
        ms[i].className = on ? 'ln-mark on' : 'ln-mark';
        ms[i].style.display = 'inline-block';
        if (on && !first) first = ms[i];
      }
      if (first) first.scrollIntoView({ block: 'nearest' });
    };

    const segs = [
      { t: 'obj 头', w: 52, c: 'a' },
      { t: 'ggml_tensor', w: 100, c: 'b' },
      { t: 'obj 头', w: 52, c: 'a' },
      { t: 'ggml_tensor', w: 100, c: 'b' },
      { t: 'obj 头', w: 52, c: 'a' },
      { t: 'ggml_cgraph', w: 100, c: 'd' },
      { t: '还没走到', w: 170, c: 'x' }
    ];
    const arena = wrap.querySelector('#arena');
    const boxes = segs.map(s => {
      const free = (s.c === 'x');
      const e = U.el('div', { style: 'width:' + s.w + 'px;height:34px;border:1px solid var(--' +
        (free ? 'border' : s.c) + ');border-radius:4px;background:' +
        (free ? '#10151b' : 'rgba(88,166,255,.10)') +
        ';display:flex;align-items:center;justify-content:center;font-family:var(--mono);' +
        'font-size:8.5px;color:var(--muted);text-align:center;padding:2px' });
      e.textContent = s.t;
      arena.appendChild(e);
      return e;
    });
    const ptr = wrap.querySelector('#ptr');
    ptr.innerHTML = '<span class="chip a">objects_begin</span>' +
      '<span class="chip d">objects_end = bump 指针</span>' +
      '<span class="chip">n_objects</span>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'arena 里<b>没有空闲链表</b>：对象一个挨一个往后摆，靠 <span class="k">offs + size</span> 找到自己。',
      '<span class="v">struct ggml_object</span> 就是对象头：<span class="v">offs</span>（在 mem_buffer 里的偏移）、' +
        '<span class="v">size</span>（对齐后的字节数）、<span class="v">next</span>（链到下一个对象）、' +
        '<span class="v">type</span>（是张量、图，还是工作缓冲）。',
      '<span class="v">GGML_OBJECT_SIZE = sizeof(struct ggml_object)</span>：每段负载前面都要先让出这么多字节。<br>' +
        '末尾那个 <span class="v">char padding[4]</span> 就是用来把头部长度补齐对齐的。',
      '<span class="v">struct ggml_context</span> 只有 7 个字段：<br>' +
        '有多大（<span class="v">mem_size</span>）、在哪（<span class="v">mem_buffer</span>）、' +
        '是不是自己的（<span class="v">mem_buffer_owned</span>）、要不要分配数据（<span class="v">no_alloc</span>）、' +
        '几个对象（<span class="v">n_objects</span>）、链表头、链表尾。',
      '<span class="v">objects_end</span> 同时就是 <b>bump 指针</b>：下一个对象从它后面开始。<br>' +
        '下一幕的 <span class="v">ggml_new_object</span> 就是照这个算的 —— <b>建图 = 指针加法</b>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; focus([1, 2, 5, 7]); });
    tl.at(5200, () => { msg.innerHTML = texts[1]; focus([5, 7]); });
    tl.at(9800, () => { msg.innerHTML = texts[2]; focus([12]); });
    tl.at(14400, () => { msg.innerHTML = texts[3]; focus([19, 20, 22, 23, 24, 29, 30]); });
    tl.at(20200, () => {
      boxes[0].style.background = 'rgba(88,166,255,.30)';
      boxes[2].style.background = 'rgba(88,166,255,.30)';
      boxes[4].style.background = 'rgba(88,166,255,.30)';
      msg.innerHTML = texts[4];
      focus([29, 30]);
    });
  }
},

/* ------------------------------------------------------ 3 <span class="hl-b">ggml_init</span>：整块 arena 只申请一次 */
{
  kicker: "L1-05 · context",
  title: "<span class=\"hl-b\">ggml_init</span>：整块 arena 只申请一次",
  sub: "context 结构自己是一次 malloc；arena 是一次 malloc；两件事都只做一次。",
  caption: "注意第一个动作是进临界区 —— 那是本课里 threading 文件的用处，第 8 幕展开。",
  src: "ggml/src/ggml.c",
  mark: [3, 14, 22, 27, 29, 31, 35, 39, 41],
  lineNo: 1611,
  code: `struct ggml_context * ggml_init(struct ggml_init_params params) {
    static bool is_first_call = true;

    ggml_critical_section_start();

    if (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    struct ggml_context * ctx = GGML_MALLOC(sizeof(struct ggml_context));

    // allow to call ggml_init with 0 size
    if (params.mem_size == 0) {
//>> mem_size 为 0 时给一个最小兜底，所以 ggml_init 允许"0 大小"的 context
        params.mem_size = GGML_MEM_ALIGN;
    }

    const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);
//>> 调用者给了 mem_buffer 就原样用；没给就把 mem_size 补齐对齐后自己分配

    *ctx = (struct ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
//>> mem_buffer 是 arena 的基址，后面所有对象都在它上面做偏移
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
//>> mem_buffer_owned：决定 ggml_free 到底要不要真去 free 这块内存
        /*.no_alloc           =*/ params.no_alloc,
//>> no_alloc 原样抄进 context：它控制张量数据是否也从 arena 里出
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
//>> 链表头尾都是 NULL —— 这是一个刚开的、空的对象世界
    };

    GGML_ASSERT(ctx->mem_buffer != NULL);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\\n", __func__);

    return ctx;
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
      '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);
    const focus = ls => {
      const ms = document.querySelectorAll('mark.ln-mark');
      let first = null;
      for (let i = 0; i < ms.length; i++) {
        const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
        ms[i].className = on ? 'ln-mark on' : 'ln-mark';
        ms[i].style.display = 'inline-block';
        if (on && !first) first = ms[i];
      }
      if (first) first.scrollIntoView({ block: 'nearest' });
    };

    const defs = [
      { c: 'c', t: '第一步 · 临界区', m: 'ggml_critical_section_start()', b: '只保护"首次调用初始化时间系统"这一段，保护完立刻退出' },
      { c: 'a', t: '第二步 · context 自己', m: 'GGML_MALLOC(sizeof(struct ggml_context))', b: '注意：这一次 malloc 拿到的只是那 7 个字段，不是 arena' },
      { c: 'b', t: '第三步 · arena', m: 'ggml_aligned_malloc(mem_size)', b: '整块内存<b>一次</b>拿到；调用者自带 mem_buffer 时这一步跳过' },
      { c: 'e', t: '第四步 · 变成空世界', m: 'objects_begin = objects_end = NULL', b: '还没有任何对象，链表的头尾都是空' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const t = U.table(
      ['ggml_init_params 字段', '在这里变成什么', '谁说了算'],
      [['mem_size', 'arena 的字节数（0 会被兜底）', '调用者预算是多少'],
       ['mem_buffer', 'arena 的基址（NULL 就自己分配）', '调用者带不带内存来'],
       ['no_alloc', '原样抄进 context', '调用者要不要 arena 存数据']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    rows.forEach(r => { r.className = ''; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      'ggml_init 只做四件事。先把"内存"和"对象"分开看：',
      '<span class="v">GGML_MALLOC(sizeof(struct ggml_context))</span>：<b>这一次 malloc 拿的是结构体本身</b>，不是 arena。',
      '<span class="v">mem_size</span> 是调用者给的预算。给 0 也允许 —— 会被兜底成 <span class="v">GGML_MEM_ALIGN</span>。',
      '<span class="v">ggml_aligned_malloc(mem_size)</span>：<b>arena 整块只申请这一次</b>。<br>' +
        '调用者自带 <span class="v">mem_buffer</span> 时就借用，连这一次都省掉（于是 <span class="v">mem_buffer_owned = false</span>）。',
      '收尾：<span class="v">n_objects = 0</span>，链表头尾都是 <span class="v">NULL</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
      if (i === 1) { rows.forEach((r, k) => { r.className = (k === 0) ? 'on' : ''; }); focus([14]); }
      if (i === 2) { rows.forEach((r, k) => { r.className = (k <= 1) ? 'on' : ''; }); focus([22]); }
      if (i === 3) { rows.forEach(r => { r.className = 'on'; }); focus([27, 29, 31, 35, 39, 41]); }
      if (i === 0) { focus([3]); }
    }));
    tl.at(16600, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      rows.forEach(r => { r.className = ''; });
      msg.innerHTML = texts[4];
      focus([35]);
    });
  }
},

/* ------------------------------------------------------ 4 ★ 释放是 <span class="hl-c">O(1)</span>：一次 free，或者一次置空 */
{
  kicker: "L1-05 · context",
  title: "★ 释放是 <span class=\"hl-c\">O(1)</span>：一次 free，或者一次置空",
  sub: "ggml_free 只 free 一次内存块；ggml_reset 连 free 都不做，把两个指针抹掉就整块可再用。",
  caption: "对照上一幕：既然只有\"一次分配\"，释放自然也只需要\"一次释放\"。",
  src: "ggml/src/ggml.c",
  mark: [5, 6, 7, 16, 17, 21, 26],
  lineNo: 1653,
  code: `void ggml_reset(struct ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = NULL;
    ctx->objects_end   = NULL;
//>> reset：只把首尾指针置 NULL —— 整块内存立刻可再用，一个对象都没 free
}

void ggml_free(struct ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }
//>> free 块里只有这一次真释放；借来的 buffer（owned = false）连这次都没有

    GGML_FREE(ctx);
//>> GGML_FREE(ctx)：连那 7 个字段的结构体自己也一起放掉
}

size_t ggml_used_mem(const struct ggml_context * ctx) {
    return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>' +
      '<div class="flow center" id="flow" style="justify-content:center"></div>';
    root.appendChild(wrap);
    const focus = ls => {
      const ms = document.querySelectorAll('mark.ln-mark');
      let first = null;
      for (let i = 0; i < ms.length; i++) {
        const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
        ms[i].className = on ? 'ln-mark on' : 'ln-mark';
        ms[i].style.display = 'inline-block';
        if (on && !first) first = ms[i];
      }
      if (first) first.scrollIntoView({ block: 'nearest' });
    };

    const defs = [
      { c: 'c', t: 'ggml_reset(ctx)', m: 'n_objects = 0; begin = end = NULL', b: '整块 arena <b>当场作废、当场可再用</b>。<br>不 free、不清零、不看内容 —— 代价是里面所有旧指针一起失效。' },
      { c: 'b', t: 'ggml_free(ctx)', m: 'ggml_aligned_free(mem_buffer, mem_size)', b: '整块只释放这一次。<br>借来的 buffer（<span class="cm" style="margin:0">owned = false</span>）连这一次都没有。' },
      { c: 'd', t: 'ggml_used_mem(ctx)', m: 'objects_end->offs + objects_end->size', b: '用了多少 = 最后一个对象的末端。<br><b>bump 指针的身高就是用量表。</b>' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const flow = wrap.querySelector('#flow');
    flow.innerHTML = '<span class="chip c">ggml_reset</span><span class="arrow">:</span>' +
      '<span class="chip">O(1)</span><span class="arrow">|</span>' +
      '<span class="chip b">ggml_free</span><span class="arrow">:</span>' +
      '<span class="chip">O(1)</span><span class="arrow">|</span>' +
      '<span class="chip d">ggml_used_mem</span><span class="arrow">:</span>' +
      '<span class="chip">O(1)</span>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '释放这一侧和分配一样短。三个函数：',
      '<span class="v">ggml_reset</span> 把 <span class="v">n_objects / objects_begin / objects_end</span> 各抹一次 —— ' +
        '<b>整块内存立刻可再用</b>，但一个对象也没被 free。',
      '<span class="v">ggml_free</span> 只在 <span class="v">mem_buffer_owned</span> 为真时 free 一次，' +
        '然后 <span class="v">GGML_FREE(ctx)</span> 放掉结构体自己。',
      '<span class="v">ggml_used_mem</span> 直接返回最后一个对象的末端：' +
        '<span class="v">objects_end->offs + objects_end->size</span>。<b>用量不用记账，指针自己就是账本。</b>',
      '一句话：<span class="k">arena 的分配与释放都是 O(1)，代价是没有"释放其中一个对象"这回事</span>。<br>' +
        '下一幕看 ggml-opt 为这个代价付出了什么。'
    ];
    tl.at(700, () => {
      els.forEach((e, k) => { e.style.opacity = k === 0 ? '1' : '.30'; });
      msg.innerHTML = texts[0];
      focus([5, 6, 7]);
    });
    tl.at(4600, () => {
      els.forEach((e, k) => { e.style.opacity = k === 1 ? '1' : '.30'; });
      msg.innerHTML = texts[1];
      focus([5, 6, 7]);
    });
    tl.at(8500, () => {
      els.forEach((e, k) => { e.style.opacity = k === 2 ? '1' : '.30'; });
      msg.innerHTML = texts[2];
      focus([16, 17, 21]);
    });
    tl.at(12400, () => {
      msg.innerHTML = texts[3];
      focus([26]);
    });
    tl.at(15800, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
      focus([]);
    });
  }
},

/* ------------------------------------------------------ 5 ★ <span class="hl-a">ggml_new_object</span>：bump 分配的全过程 */
{
  kicker: "L1-05 · 核心",
  title: "★ <span class=\"hl-a\">ggml_new_object</span>：bump 分配的全过程",
  sub: "取当前末端、对齐、算新位置、查预算、写对象头、串到链表尾 —— 没有查找，没有空闲表。",
  caption: "这就是\"建图 = 指针加法\"的字面实现：新对象地址 = mem_buffer + cur_end。",
  src: "ggml/src/ggml.c",
  mark: [2, 4, 6, 11, 14, 15, 29, 33, 40, 50, 53, 56],
  lineNo: 1708,
  code: `static struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;
//>> cur_end = 上一个对象的 offs + size —— 这就是 bump 指针的当前值

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);
//>> 先对齐到 GGML_MEM_ALIGN，再谈放得下放不下

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);
//>> 没有任何查找：新对象直接落在 mem_buffer + cur_end

    // integer overflow checks
    if (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\\n", __func__, cur_end, size_needed);
        return NULL;
    }
    if (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        return NULL;
    }

    if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        return NULL;
    }
//>> 放不下就告警并返回 NULL（debug 版直接 abort）—— arena 不会扩容

    *obj_new = (struct ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
//>> offs 特意跳过对象头本身：负载从对象头之后开始
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;`,
  duration: 27000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);
    const focus = ls => {
      const ms = document.querySelectorAll('mark.ln-mark');
      let first = null;
      for (let i = 0; i < ms.length; i++) {
        const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
        ms[i].className = on ? 'ln-mark on' : 'ln-mark';
        ms[i].style.display = 'inline-block';
        if (on && !first) first = ms[i];
      }
      if (first) first.scrollIntoView({ block: 'nearest' });
    };

    const defs = [
      { c: 'a', t: '1 · 取末端', m: 'cur_end = offs + size', b: '从 <span class="cm" style="margin:0">ctx->objects_end</span> 读。<br>空链表时是 0。' },
      { c: 'b', t: '2 · 对齐', m: 'GGML_PAD(size, GGML_MEM_ALIGN)', b: '先把要放的字节数补齐，<br>后面所有偏移都按它算。' },
      { c: 'c', t: '3 · 算地址', m: 'mem_buffer + cur_end', b: '新对象头就落在这儿。<br><b>没有查找，没有空闲表。</b>' },
      { c: 'f', t: '4 · 查预算', m: 'cur_end + size + OBJECT_SIZE > mem_size', b: '超出就告警并返回 NULL。<br><b>arena 不会扩容。</b>' },
      { c: 'd', t: '5 · 写头', m: 'offs = cur_end + GGML_OBJECT_SIZE', b: 'offs 跳过对象头；<br>负载紧跟在头后面。' },
      { c: 'e', t: '6 · 串链', m: 'obj_cur->next = obj_new', b: '首对象时改写 <span class="cm" style="margin:0">objects_begin</span>；<br>无论如何都改写 <span class="cm" style="margin:0">objects_end</span>。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.26');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '整个分配过程六步，全程只有算术和一次比较。',
      '<span class="v">cur_offs / cur_size / cur_end</span>：bump 指针的当前值就是"上一个对象的末端"。',
      '<span class="v">GGML_PAD(size, GGML_MEM_ALIGN)</span>：先对齐，再判断装不装得下。',
      '<span class="v">mem_buffer + cur_end</span>：<b>新对象的位置是一次指针加法算出来的</b>，没有任何查找。',
      '<span class="v">cur_end + size_needed + GGML_OBJECT_SIZE &gt; ctx->mem_size</span> 就告警返回 NULL。' +
        '<br>这就是 arena 的"预算超支"：<b>没有扩容，只有失败</b>。',
      '<span class="v">.offs = cur_end + GGML_OBJECT_SIZE</span>：负载从对象头之后开始，所以 <span class="v">offs</span> 要跳过头部。',
      '最后把它挂到链表尾：<b>对象链表就是 arena 里唯一的索引结构</b>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.26'; });
      msg.innerHTML = texts[i];
      focus([[2, 4, 6], [6], [11], [14, 15], [29, 33], [40], [50, 53, 56]][i]);
    }));
    tl.at(23000, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[6];
      focus([]);
    });
  }
},

/* ------------------------------------------------------ 6 ★ arena 的代价：<span class="hl-e">ggml-opt</span> 为什么要四个 context */
{
  kicker: "L1-05 · 代价",
  title: "★ arena 的代价：<span class=\"hl-e\">ggml-opt</span> 为什么要四个 context",
  sub: "张量只能跟着整块 arena 一起死。要按生命周期分组，唯一的办法是分组开 context。",
  caption: "这里的关键选择不是数据结构，而是\"哪些张量共用一次生死\"。",
  src: "ggml/src/ggml-opt.cpp",
  mark: [],
  lineNo: 0,
  code: `struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
//>> ---- ggml/src/ggml-opt.cpp:584-594 ----
void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = '<div class="row center" id="ctxs" style="gap:7px"></div>' +
      '<div class="row center" id="frees" style="gap:6px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'ctx_static', b: '梯度、优化器动量、loss' },
      { c: 'b', t: 'ctx_cpu', b: '优化器参数（1 个张量）' },
      { c: 'c', t: 'ctx_compute', b: '调用者的临时张量' },
      { c: 'd', t: 'ctx_copy', b: '静态图的副本' }
    ];
    const host = wrap.querySelector('#ctxs');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const frees = wrap.querySelector('#frees');
    frees.innerHTML = '<span class="chip">ggml_backend_buffer_free x2</span>' +
      '<span class="arrow">-></span><span class="chip a">ggml_free(ctx_static)</span>' +
      '<span class="chip b">ggml_free(ctx_cpu)</span>' +
      '<span class="chip d">ggml_free(ctx_copy)</span>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'ggml_opt_context 里有 <b>四个</b> ggml_context 字段。这不是随手写的。',
      '回顾第 4 幕：能释放的粒度只有"整个 context"。<span class="v">ggml_reset</span>/<span class="v">ggml_free</span> 都不认识单个张量。',
      '所以"想让一批张量一起死"，就得让它们<b>共用一个 context</b>：<br>' +
        '<span class="v">ctx_static</span> 放梯度与动量，<span class="v">ctx_cpu</span> 只放一个优化器参数张量，<span class="v">ctx_copy</span> 放静态图副本。',
      '<span class="v">ggml_opt_free</span> 里就是三次 <span class="v">ggml_free</span> —— ' +
        '<b>释放一个训练上下文 = 释放三块 arena</b>，仍然是 O(1) 次调用。',
      '这就是 arena 的取舍：<span class="k">分配与释放极快，但"按张量释放"这个能力被换掉了</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4300, () => { msg.innerHTML = texts[1]; });
    els.forEach((_, i) => tl.at(8000 + i * 2400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[2];
    }));
    tl.at(17600, () => { msg.innerHTML = texts[3]; });
    tl.at(19000, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 7 arena 的另一面：内存要<span class="hl-c">提前算出来</span> */
{
  kicker: "L1-05 · 代价",
  title: "arena 的另一面：内存要<span class=\"hl-c\">提前算出来</span>",
  sub: "mem_size 是 ggml_init 的参数，所以调用者必须先知道\"会有几个张量\"。",
  caption: "呼应 L1-01：张量是定长值类型，在 arena 里的占用不随形状变 —— 预算才能按\"个数\"算。",
  src: "ggml/src/ggml-opt.cpp",
  mark: [1, 8, 9, 10, 11, 14, 16, 18],
  lineNo: 346,
  code: `    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
//>> size_meta：先数张量个数，再乘每个张量的固定开销 —— arena 的预算是这么来的
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
//>> 拿算出来的字节数去 ggml_init：arena 的大小是"算"出来的，不是"长"出来的
    }`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);
    const focus = ls => {
      const ms = document.querySelectorAll('mark.ln-mark');
      let first = null;
      for (let i = 0; i < ms.length; i++) {
        const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
        ms[i].className = on ? 'ln-mark on' : 'ln-mark';
        ms[i].style.display = 'inline-block';
        if (on && !first) first = ms[i];
      }
      if (first) first.scrollIntoView({ block: 'nearest' });
    };

    const t = U.table(
      ['预算项', '数量', '它是什么'],
      [['n_loss', '1', '损失项（每个 loss 一个梯度）'],
       ['tensors_per_param x n_param', '(1 或 0) + (2 或 0)', '每个参数：累积梯度 1 个；AdamW 再要 m / v 两个动量'],
       ['tensors_const', 'static_graphs ? 9 : 0', 'labels / loss / pred / ncorrect 等固定项'],
       ['x ggml_tensor_overhead()', '每个的固定开销', '对象头 + 定长张量体（与形状无关）']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'arena 要一次 malloc，就必须先知道要多少字节。于是有了这份预算表：',
      '<span class="v">n_param</span> 是遍历 forward 图数出来的 —— ' +
        '带 <span class="v">GGML_TENSOR_FLAG_PARAM</span> 的节点有几个。',
      '<span class="v">tensors_per_param</span> 取决于两个开关：要不要累积梯度、优化器是不是 AdamW（要两个动量）。',
      '<span class="v">tensors_const</span> 是 loss / labels / pred / ncorrect 这一批固定项。',
      '三者相加，<b>再乘每个张量的固定开销</b> <span class="v">ggml_tensor_overhead()</span> —— 就是 <span class="v">mem_size</span>。',
      '关键点：<span class="k">这个乘法成立，是因为一个张量在 arena 里的占用不随形状变</span>（L1-01：ne/nb/src 都是定长数组）。'
    ];
    function show(i) {
      rows.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i];
    }
    tl.at(700, () => { show(0); rows.forEach(r => { r.className = ''; }); });
    tl.at(4300, () => { show(1); rows.forEach((r, k) => { r.className = (k <= 0) ? 'on' : ''; }); });
    tl.at(8000, () => { show(2); rows.forEach((r, k) => { r.className = (k === 1) ? 'on' : ''; }); });
    tl.at(11700, () => { show(3); rows.forEach((r, k) => { r.className = (k === 2) ? 'on' : ''; }); });
    tl.at(15400, () => { show(4); rows.forEach((r, k) => { r.className = (k === 3) ? 'on' : ''; }); focus([8, 9, 10, 11, 14, 16, 18]); });
    tl.at(18500, () => { show(5); rows.forEach((r, k) => { r.className = (k === 3) ? 'on' : ''; }); });
  }
},

/* ------------------------------------------------------ 8 <span class="hl-f">ggml-threading</span> 到底提供了什么：一把全局互斥锁 */
{
  kicker: "L1-05 · 线程",
  title: "<span class=\"hl-f\">ggml-threading</span> 到底提供了什么：一把全局互斥锁",
  sub: "整个实现文件 12 行、整个头文件 14 行。它没有线程池，也没有任务队列。",
  caption: "本课引用到的调用点：ggml/src/ggml.c 的 ggml_init（第 3 幕代码块第 4、13 行）；它不是全仓唯一一处。",
  src: "ggml/src/ggml-threading.cpp",
  mark: [0, 3, 5, 6, 9, 10],
  lineNo: 1,
  code: `#include "ggml-threading.h"
#include <mutex>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);
    const focus = ls => {
      const ms = document.querySelectorAll('mark.ln-mark');
      let first = null;
      for (let i = 0; i < ms.length; i++) {
        const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
        ms[i].className = on ? 'ln-mark on' : 'ln-mark';
        ms[i].style.display = 'inline-block';
        if (on && !first) first = ms[i];
      }
      if (first) first.scrollIntoView({ block: 'nearest' });
    };

    const defs = [
      { c: 'f', t: '就这些（.cpp，12 行）', b: '一个 <span class="cm" style="margin:0">std::mutex</span>，<br>两个薄封装函数。', m: 'ggml_critical_section_start/end' },
      { c: 'a', t: '就这些（.h，14 行）', b: '头文件里只声明这两个函数，<br>没有别的公开 API。', m: 'GGML_API void ggml_critical_section_start' },
      { c: 'g', t: '它<b>没有</b>提供的', b: '线程池、任务队列、worker、<br>原子计数器 —— 一个都没有。', m: '（实测：grep 与逐行阅读）' },
      { c: 'c', t: '本课看到的调用点', b: 'ggml_init 里保护"首次初始化<br>时间系统"那一段。', m: 'ggml/src/ggml.c:1614, 1623' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把整个文件读完只需要 12 行。<b>它提供的东西少得让人意外。</b>',
      '<span class="v">std::mutex ggml_critical_section_mutex</span>：一个进程级的全局锁，没有名字空间、没有分层。',
      '<span class="v">start()</span> 就是 <span class="v">lock()</span>，<span class="v">end()</span> 就是 <span class="v">unlock()</span> —— ' +
        '不含 RAII、不含超时、不含重入。<b>调用者必须自己保证配对。</b>',
      '这把锁在仓库里还有别的调用点（不在本课引用范围内），形态都一样：<b>保护"首次调用时初始化一张共享表"</b>。',
      '这一课的教训：<span class="k">计划/目录名会骗人，只有逐行读过的文件不会</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; focus([0, 3]); });
    tl.at(4200, () => { msg.innerHTML = texts[1]; focus([3]); });
    tl.at(7700, () => { msg.innerHTML = texts[2]; focus([5, 6, 9, 10]); });
    tl.at(11200, () => { msg.innerHTML = texts[3]; focus([]); });
    tl.at(13800, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 <span class="hl-g">ggml.cpp</span>：这个文件其实与 context 无关 */
{
  kicker: "L1-05 · 兜底",
  title: "<span class=\"hl-g\">ggml.cpp</span>：这个文件其实与 context 无关",
  sub: "26 行，只做一件事：没人接住的异常，先打 backtrace，再 abort。",
  caption: "本课覆盖的五个文件中，只有它不参与对象分配 —— 如实讲它实际做了什么。",
  src: "ggml/src/ggml.cpp",
  mark: [2, 7, 12, 20, 21],
  lineNo: 6,
  code: `static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
//>> 打印调用栈之后仍然 abort —— 这个兜底不吞异常，只让它留下线索
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
//>> GGML_NO_BACKTRACE 环境变量：设了就完全不装这个 handler
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);
    const focus = ls => {
      const ms = document.querySelectorAll('mark.ln-mark');
      let first = null;
      for (let i = 0; i < ms.length; i++) {
        const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
        ms[i].className = on ? 'ln-mark on' : 'ln-mark';
        ms[i].style.display = 'inline-block';
        if (on && !first) first = ms[i];
      }
      if (first) first.scrollIntoView({ block: 'nearest' });
    };

    const defs = [
      { c: 'g', t: '它做了什么', b: '静态初始化时把 <span class="cm" style="margin:0">std::terminate</span> 换掉；<br>异常没人接住时先 <span class="cm" style="margin:0">ggml_print_backtrace()</span>。', m: 'ggml_uncaught_exception' },
      { c: 'c', t: '它不做什么', b: '不分配、不释放、<br>不碰 context、不碰线程。', m: '（26 行全文）' },
      { c: 'b', t: '开关', b: '<span class="cm" style="margin:0">GGML_NO_BACKTRACE</span> 环境变量存在，<br>就直接不装 handler。', m: 'getenv("GGML_NO_BACKTRACE")' },
      { c: 'd', t: '为什么值得看', b: '读完它才知道：<br><b>文件名不等于内容</b>。', m: 'static bool ... = []{ ... }()' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一课覆盖的五个文件里，ggml.cpp 是最容易被误判的一个。',
      '<span class="v">GGML_NORETURN static void ggml_uncaught_exception()</span>：' +
        '先 <span class="v">ggml_print_backtrace()</span>，再调用上一个 terminate handler，最后 <span class="v">abort()</span>。',
      '<span class="v">previous_terminate_handler</span> 保存原来的处理器，' +
        '<span class="v">std::set_terminate(ggml_uncaught_exception)</span> 把它换掉 —— <b>在静态初始化阶段完成</b>。',
      '注意那个 <span class="v">GGML_ASSERT(prev != ggml_uncaught_exception)</span>：' +
        '<b>防止这个 handler 被装两次造成自递归</b>。',
      '所以它和 context、线程、优化器都无关 —— 它是整个 ggml 的<b>进程级兜底</b>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3700, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
      focus([[2, 7], [], [12], [20, 21]][i]);
    }));
    tl.at(15200, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
      focus([]);
    });
  }
},

/* ------------------------------------------------------ 10 压成一张表：arena 换来了什么、换走了什么 */
{
  kicker: "L1-05 · 收束",
  title: "压成一张表：arena 换来了什么、换走了什么",
  sub: "一次分配、一次释放、O(1) 的建图；代价是没有按张量释放，以及内存要提前预算。",
  caption: "下一课 L1-06：这些张量的权重从 GGUF 文件里来。",
  src: "ggml/include/ggml-opt.h",
  mark: [1, 3, 12, 13],
  lineNo: 238,
  code: `    // fit model defined by inputs and outputs to dataset
    GGML_API void ggml_opt_fit(
            ggml_backend_sched_t            backend_sched,  // backend scheduler for constructing the compute graphs
            struct ggml_context           * ctx_compute,    // context with temporarily allocated tensors to calculate the outputs
//>> ctx_compute：只放"临时张量"的那个 context；参数与输入的那个在调用者手上
            struct ggml_tensor            * inputs,         // input tensor with shape [ne_datapoint, ndata_batch]
            struct ggml_tensor            * outputs,        // output tensor, must have shape [ne_label, ndata_batch] if labels are used
            ggml_opt_dataset_t              dataset,        // dataset with data and optionally also labels
            enum ggml_opt_loss_type         loss_type,      // loss to minimize
            enum ggml_opt_optimizer_type    optimizer,      // sgd or adamw
            ggml_opt_get_optimizer_params   get_opt_pars,   // callback to get optimizer params, userdata is pointer to epoch (of type int64_t)
            int64_t                         nepoch,         // how many times the dataset should be iterated over
            int64_t                         nbatch_logical, // datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs
            float                           val_split,      // fraction of the dataset to use for validation, must be in [0.0f, 1.0f)
//>> val_split：数据集后段按这个比例做验证，对应 ggml_opt_epoch 的 idata_split
            bool                            silent);        // whether or not info prints to stderr should be suppressed`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['机制', '在这一课里看到的实际做法', '去哪一课展开'],
      [['arena', '一块内存 + 一条对象链表；ggml_init 只 malloc 一次', '本课 · 第 2/3 幕'],
       ['bump 分配', 'ggml_new_object：mem_buffer + cur_end，顺序摆放', '本课 · 第 5 幕'],
       ['O(1) 释放', 'ggml_free 只 free 一次；ggml_reset 只把两个指针置空', '本课 · 第 4 幕'],
       ['不能单独释放', 'ggml-opt 用四个 context 按生命周期分组', 'L4-01 分配器'],
       ['内存要预算', 'mem_size = 张量个数 x ggml_tensor_overhead()', 'L1-01 定长张量'],
       ['并发原语', '一个全局 std::mutex，两个薄封装函数', '本课 · 第 8 幕']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '<span class="mono">ggml_opt</span> 要给静态 context 预留多少字节？为什么可以这样算？',
      '代码里是 <span class="mono">size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead()</span>：' +
      '<b>先数出会创建多少个张量，再乘每个张量的固定开销</b>。<br>' +
      '能这样算的前提是 <b>L1-01</b>：<span class="mono">ggml_tensor</span> 是<b>定长值类型</b>' +
      '（<span class="mono">ne[4]</span> / <span class="mono">nb[4]</span> / <span class="mono">src[10]</span> 都是定长数组），' +
      '它在 arena 里的占用不随形状变化。<br>' +
      '换句话说：<b>arena 的预算是按"个数"算的，不是按"形状"算的</b>。若张量是变长的，这个乘法就不成立了。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '六行，就是这一课的全部：',
      '<span class="k">一次分配</span> + <span class="k">一次释放</span> 换来 <span class="k">指针加法的建图速度</span>。',
      '换走的是两件事：<b>不能单独释放一个张量</b>，以及 <b>mem_size 必须提前算出来</b>。',
      '所以 L4-01 的 ggml-alloc 要在 arena 之上再做一层：它负责把"同时活着的张量"复用同一段内存。',
      '记忆锚点：<span class="v">ggml_context 是对象世界的边界，ggml_free 是这个世界的唯一出口</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(17600, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[4];
    });
  }
},

];
