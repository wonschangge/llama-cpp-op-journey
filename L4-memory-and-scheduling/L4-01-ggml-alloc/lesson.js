/* ==========================================================================
   L4-01 · 分配器 ggml-alloc：算子的内存从哪来
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 内存规划发生在建图之后、执行之前 */
{
  kicker: "L4-01 · 全局",
  title: "内存规划发生在建图之后、执行之前",
  sub: "ggml-alloc.h 的注释把标准用法写死了：选 buffer 类型 -> 预留 -> 分配 -> 查大小 -> 执行。",
  caption: "下一课 L4-02 讲调度器怎么把同一张图切成多个 buffer；本课先把单 buffer 的规划讲透。",
  src: "ggml/include/ggml-alloc.h",
  mark: [3, 7, 12, 14, 18],
  lineNo: 24,
  code: `// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
//>> 可选但关键：拿一张"最坏情况图"（例如 max_batch）先规划一次
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
//>> 每次 decode 都重新建一张图 —— 但形状不变时 L2-07 会复用上一次的那张
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
//>> 分配完成后才轮到后端执行；本课只讲 30 / 34 / 36 这三步
    ggml_backend_graph_compute(backend, graph);
*/`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">建图</span><span class="arrow">-></span>
        <span class="chip a">reserve</span><span class="arrow">-></span>
        <span class="chip b">alloc_graph</span><span class="arrow">-></span>
        <span class="chip c">get_buffer_size</span><span class="arrow">-></span>
        <span class="chip d">graph_compute</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '先选 buffer 类型', b: 'CPU / CUDA / Metal ...；带 _n 的版本<br>一次给多种，对应 L4-02 的图切分', m: 'ggml_gallocr_new_n' },
      { c: 'b', t: '用最坏情况的图预留', b: '只做内存规划：<br>不改图、不碰任何张量', m: 'ggml_gallocr_reserve' },
      { c: 'c', t: '对当前图落位', b: '把每个张量的 data / buffer<br>指到规划好的地址上', m: 'ggml_gallocr_alloc_graph' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '219px' }); host.appendChild(e); return e; });
    els.forEach(e => { e.style.opacity = '.30'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一段注释就是本课的地图：<span class="k">内存规划夹在建图与执行之间</span>。',
      '<span class="v">reserve</span> 用一张"最坏情况图"先规划一次，运行时就不必再重新分配 —— 它是可选的，但生产代码一定会调。',
      '<span class="v">alloc_graph</span> 才把地址真正发给张量；它用的是<b>上一次规划的结果</b>，不是重新算一遍。',
      '<span class="v">get_buffer_size</span> 把规划结果报出来：这就是"这个模型要占多少显存"的答案。',
      '为什么要有这一整层？因为<span class="k">让每个张量各自去要一块内存</span>，在 70 层 transformer 上是活不下去的 —— 下一幕算这笔账。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(17600, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ 只进不退的分配器 <span class="hl-g">活不过一张图</span> */
{
  kicker: "L4-01 · 动机",
  title: "★ 只进不退的分配器 <span class=\"hl-g\">活不过一张图</span>",
  sub: "ggml_tallocr 是最简单的分配器：一个 buffer、一个 offset，只往前走；它没有配套的释放函数。",
  caption: "对照 L1-05：context 的 arena 也是\"只进不退\"，但那是放张量元数据的；这里说的是放数据的 buffer。",
  src: "ggml/src/ggml-alloc.c",
  mark: [3, 19, 30],
  lineNo: 59,
  code: `// tallocr
//>> tallocr = tensor allocator，最小可用的那一种

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
//>> 先问后端：这个张量按你的规矩要占多少字节
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;
//>> 这一行就是全部策略：offset 只会往前推，永不回退

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px;align-items:flex-start">
        <div class="col" style="gap:7px;width:340px" id="bars"></div>
        <div class="col grow" style="gap:7px" id="notes"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const bars = wrap.querySelector('#bars');
    const b1 = U.bar('各占一块', 'g');
    const b2 = U.bar('复用峰值', 'b');
    bars.appendChild(b1.el); bars.appendChild(b2.el);
    bars.appendChild(U.el('div', { class: 'cm', style: 'margin:0',
      text: '同一张 decode 图：每个中间张量各要一块 vs 按生命周期复用（示意）' }));
    b1.fill.style.width = '0%'; b2.fill.style.width = '0%';

    const notes = wrap.querySelector('#notes');
    notes.innerHTML =
      '<div class="card" style="border-left-color:var(--g)">' +
      '<div class="ct" style="color:var(--g)">它为什么不够用</div>' +
      '<div class="cb">一条数据流上，<b>绝大多数中间张量只活一小会儿</b>：某个 RMS_NORM 的输出喂给下一个 MUL_MAT 之后就没用了。' +
      '但 offset 不会退回，那块内存在这张图剩下的时间里一直空占着。</div></div>' +
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">它也不是没用</div>' +
      '<div class="cb">权重这种<b>整场都活着</b>的张量，用它顺序摆一遍正合适 —— ' +
      'header 里的 <span class="cm" style="margin:0">ggml_backend_alloc_ctx_tensors</span> 走的就是这条路。</div></div>' +
      '<div class="card" style="border-left-color:var(--f)">' +
      '<div class="ct" style="color:var(--f)">没有 free 函数</div>' +
      '<div class="cb">整个 ggml-alloc.h 只有 <span class="cm" style="margin:0">ggml_tallocr_new</span> 与 ' +
      '<span class="cm" style="margin:0">ggml_tallocr_alloc</span> 两个 API —— "还块"这件事压根不存在。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '32 行代码就是一个完整的分配器：<span class="k">算大小 -> 对齐 -> offset 往前推 -> 告诉后端</span>。',
      '<span class="v">ggml_backend_buffer_get_alloc_size</span>（77 行）让后端参与决定大小 —— 同一个张量在 CPU 和 CUDA 上可能不一样大。',
      '<span class="v">talloc-&gt;offset += size</span>（87 行）：<b>单调递增</b>。这就是"不复用"的全部含义。',
      '把一张图从头跑到尾要多少内存？<span class="v">所有张量大小之和</span>。而其中同时活着的，可能只有十分之一。',
      '所以 ggml 需要另一套东西：<span class="k">知道谁什么时候死，然后把死者的内存立刻发给下一个人</span> —— 这就是 ggml_dyn_tallocr。'
    ];
    tl.at(700, () => { b1.fill.style.width = '100%'; b1.val.textContent = '100%'; msg.innerHTML = texts[0]; });
    tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, [19]); });
    tl.at(7700, () => { msg.innerHTML = texts[2]; U.markLines(document, [30]); });
    tl.at(11200, () => {
      b2.fill.style.width = '24%'; b2.val.textContent = '24%';
      msg.innerHTML = texts[3]; U.markLines(document, []);
    });
    tl.at(14600, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ <span class="hl-a">free_block</span>：复用靠的就是这张空闲表 */
{
  kicker: "L4-01 · 核心",
  title: "★ <span class=\"hl-a\">free_block</span>：复用靠的就是这张空闲表",
  sub: "动态分配器不记录\"哪些张量在\"，只记录\"哪些字节是空的\" —— 于是张量之间可以直接换手。",
  caption: "两个上限决定了这张表的形状：MAX_FREE_BLOCKS = 256（第 14 行）、GGML_VBUFFER_MAX_CHUNKS = 16（第 96 行）。",
  src: "ggml/src/ggml-alloc.c",
  mark: [0, 6, 13],
  lineNo: 110,
  code: `struct free_block {
    size_t offset;
    size_t size;
//>> 空闲块的粒度就是 offset + size 两个数 —— 它不需要知道谁曾住在里面
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
//>> 块的数量是有限的：256 个。用完在第 136 行直接 GGML_ASSERT 失败
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
//>> 一个 buffer 最多切成 16 个 chunk；显存被 max_size 切开时用得上（L4-03）
    int n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        struct buffer_address addr;
    } allocated_tensors[1024];
#endif
};`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="cm" id="cap" style="margin:0">同一个 chunk 里的一维地址空间（示意）</div>
      <div class="row" id="strip" style="gap:3px;justify-content:center;align-items:stretch"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'free_block', b: '一段空闲内存：<br>起点偏移 + 长度', m: 'offset, size' },
      { c: 'b', t: 'tallocr_chunk', b: '一个 chunk 的空闲表 + 水位线<br>（已用到过的最高地址）', m: 'free_blocks[256] · n_free_blocks · max_size' },
      { c: 'c', t: 'ggml_dyn_tallocr', b: '每个 buffer 一个动态分配器：<br>对齐、chunk 上限、chunk 列表', m: 'alignment · max_chunk_size · chunks[16] · n_chunks' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '219px' }); host.appendChild(e); return e; });
    els.forEach(e => { e.style.opacity = '.30'; });

    const strip = wrap.querySelector('#strip');
    const segs = [
      { w: 96,  t: '已分配 (x)',  c: '#1f2630' },
      { w: 148, t: 'free 12288',  c: 'rgba(88,166,255,.22)' },
      { w: 84,  t: '已分配 (h)',  c: '#1f2630' },
      { w: 112, t: 'free 8192',   c: 'rgba(63,185,80,.22)' },
      { w: 186, t: '最后一个 free（可向外扩）', c: 'rgba(210,153,34,.18)' }
    ];
    const sels = segs.map(s => {
      const e = U.el('div', { style: 'width:' + s.w + 'px;height:34px;border:1px solid var(--border);' +
        'border-radius:4px;background:' + s.c + ';display:flex;align-items:center;justify-content:center;' +
        'font-family:var(--mono);font-size:9px;color:var(--muted);padding:0 4px;overflow-wrap:anywhere' });
      e.textContent = s.t;
      strip.appendChild(e);
      return e;
    });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '分配器不维护"张量 -> 地址"的账本，它只维护<span class="k">空闲块表</span>。',
      '<span class="v">free_blocks[]</span> 按地址<span class="k">有序</span>存放（插入时就排好，见第 135-150 行）—— 有序是为了还块时能立刻判断"能不能和邻居合并"。',
      '<span class="v">max_size</span> 是这个 chunk 的<b>水位线</b>，也是最后开 buffer 时唯一要用的数：规划完它多大，buffer 就开多大（第 917-924 行）。',
      '<span class="v">chunks[16]</span> 让一个分配器可以横跨多个后端 buffer —— 这是 L4-02 多 buffer 调度的地基。',
      '一句话：<span class="k">地址空间被切成"已分配"与"空闲块"两种状态，空闲块就是可以被下一个人接手的部分</span>。'
    ];
    const hi = [[1], [1, 3], [2], [3, 4], [1, 2, 3, 4]];
    defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      sels.forEach((s, k) => { s.style.outline = hi[i].indexOf(k) >= 0 ? '1px solid var(--a)' : 'none'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(18700, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      sels.forEach(s => { s.style.outline = 'none'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-a">best-fit</span>：从空闲表里挑最小够用的那块 */
{
  kicker: "L4-01 · 核心",
  title: "★ <span class=\"hl-a\">best-fit</span>：从空闲表里挑最小够用的那块",
  sub: "注意循环上界那个减一：最后一个空闲块不参与挑选，因为它是\"可以向外扩\"的边界。",
  caption: "挑选判据在源码注释里写得很清楚：装得下，且是装得下的里面最小的。这就是 best fit。",
  src: "ggml/src/ggml-alloc.c",
  mark: [10, 15, 18],
  lineNo: 202,
  code: `static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);
//>> 先按分配器的 alignment 把请求大小对齐（aligned_offset，第 53-57 行）

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
//>> 注释原话：find the best fitting free block besides the last block
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
//>> 两个条件同时成立：装得下（size >= 请求），且比目前找到的更小
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="formula" id="ask"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    wrap.querySelector('#ask').innerHTML =
      '请求：<span class="v">ggml_dyn_tallocr_alloc(alloc, 12288, node)</span> —— ' +
      '要一块 <span class="k">12288 字节</span>（已按 alignment 对齐）';

    const t = U.table(
      ['free_block', 'offset', 'size', '装得下?', '结果'],
      [['#0', '4096', '8192', '否（8192 小于 12288）', '跳过'],
       ['#1', '16384', '16384', '是', '★ 选中：装得下的里面最小的'],
       ['#2', '32768', '12288', '是（并列）', '不选：不比它更小'],
       ['#3（最后一个）', '45056', '很大', '—', '本轮不参与挑选']],
      { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '三个数先固定下来：<span class="v">best_fit_chunk = -1</span>、<span class="v">best_fit_block = -1</span>、<span class="v">best_fit_size = SIZE_MAX</span>（第 207-209 行）。',
      '内层循环从 0 走到 <span class="v">n_free_blocks - 1</span>（第 215 行）—— <span class="k">少扫的正好是最后一个块</span>：那是水位线以上、可以向外长的地方。',
      '每碰到一个装得下的块就比较 <span class="v">size &lt;= best_fit_size</span>（第 218 行）：<span class="k">只留更小的那个</span>，所以最后拿到的是最紧的一块。',
      '循环结束后第 263-265 行把选中块的 offset 让出去、size 减掉；减到 0 就把这一项从空闲表里删掉（第 266-269 行）。',
      '只有 <span class="v">best_fit_block == -1</span>（一个都装不下）时，第 226 行才开始看最后一个块 —— 用它意味着把 buffer 撑大。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3200 + i * 2700, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 3)];
      if (i === 1) { U.markLines(document, [15]); }
      if (i === 2) { U.markLines(document, [18]); }
    }));
    tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 还块：与相邻空闲块<span class="hl-b">就地合并</span> */
{
  kicker: "L4-01 · 核心",
  title: "还块：与相邻空闲块<span class=\"hl-b\">就地合并</span>",
  sub: "张量用完只是把地址还给空闲表；不合并的话，空闲表很快会被碎片塞满（上限 256 块）。",
  caption: "源码自评：\"this is a very naive implementation\" —— 但它够用，因为一张图里的空闲块通常很少。",
  src: "ggml/src/ggml-alloc.c",
  mark: [11, 17, 26, 43],
  lineNo: 311,
  code: `// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
//>> 分两种情况：新空出来的块贴在已有空闲块的后面，还是前面
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == addr.offset) {
//>> 情况一：addr 正好接在块尾 -> 直接长大
            block->size += size;
            // check if we can merge with the next block
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
//>> 长大之后顺手看一眼下一块能不能一起吞掉
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (addr.offset + size == block->offset) {
//>> 情况二：addr 正好接在块头 -> 往前吞
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }
    // otherwise, add a new block
//>> 两个方向都不沾（左右都是活人）-> 只能新开一个空闲块
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="cm" id="c1" style="margin:0">释放前：h 被两块空闲区夹在中间</div>
      <div class="row" id="before" style="gap:3px;justify-content:center"></div>
      <div class="cm" id="c2" style="margin:0">释放 h 之后：三块并成一块，下次要 24 KB 也能一次给出去</div>
      <div class="row" id="after" style="gap:3px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    function seg(host, w, text, bg) {
      const e = U.el('div', { style: 'width:' + w + 'px;height:32px;border:1px solid var(--border);' +
        'border-radius:4px;background:' + bg + ';display:flex;align-items:center;justify-content:center;' +
        'font-family:var(--mono);font-size:9px;color:var(--muted);transition:all .5s;padding:0 3px' });
      e.textContent = text;
      host.appendChild(e);
      return e;
    }
    const before = wrap.querySelector('#before');
    const bFree1 = seg(before, 116, 'free 8192', 'rgba(88,166,255,.22)');
    const bH     = seg(before, 130, 'h（已分配）', '#1f2630');
    const bFree2 = seg(before, 96,  'free 4096', 'rgba(88,166,255,.22)');

    const after = wrap.querySelector('#after');
    const aMerged = seg(after, 116 + 130 + 96 + 6, 'free 24576（8192 + 12288 + 4096）', 'rgba(63,185,80,.22)');
    aMerged.style.opacity = '.18';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '还块要解决的是<span class="k">碎片</span>：一块 8 KB 的空闲区旁边就是一块 4 KB 的，单独都装不下 24 KB。',
      '第 321 行先试"贴在块尾"：<span class="v">block-&gt;offset + block-&gt;size == addr.offset</span>。',
      '成立就把自己接上去（第 322 行），再看一眼下一块能不能一起吞（第 326-329 行）。',
      '否则第 334 行试"贴在块头"：<span class="v">addr.offset + size == block-&gt;offset</span>，成立就往前吞（第 335-336 行），同样回头看上一块（第 340-343 行）。',
      '两个方向都不沾，才 <span class="v">ggml_dyn_tallocr_insert_block</span> 新开一块（第 349 行）—— 这是空闲表变碎的唯一来源。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3800, () => { bH.style.background = 'rgba(210,153,34,.25)'; bH.style.borderColor = 'var(--c)'; msg.innerHTML = texts[1]; });
    tl.at(7300, () => { msg.innerHTML = texts[2]; });
    tl.at(10800, () => {
      bH.textContent = 'free 12288'; bH.style.background = 'rgba(88,166,255,.22)'; bH.style.borderColor = 'var(--border)';
      msg.innerHTML = texts[3];
    });
    tl.at(14300, () => {
      bH.style.opacity = '.18'; bFree1.style.opacity = '.18'; bFree2.style.opacity = '.18';
      aMerged.style.opacity = '1';
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 6 ★ 第一趟：先数清<span class="hl-c">谁还要用我</span> */
{
  kicker: "L4-01 · 核心",
  title: "★ 第一趟：先数清<span class=\"hl-c\">谁还要用我</span>",
  sub: "分配之前，先把每个张量被引用几次（n_children）、被几个视图指着（n_views）数出来。",
  caption: "对照 L1-03：图的拓扑序在这里第一次被\"用起来\" —— 节点顺序就是生命周期的顺序。",
  src: "ggml/src/ggml-alloc.c",
  mark: [0, 8, 18, 34],
  lineNo: 723,
  code: `    // allocate leafs
//>> 所有叶子先落位 —— 权重、KV cache 这些"整场都活着"的东西
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
//>> 这一趟除输入外不分配，只做两件事：数 n_children、数 n_views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in \`src\`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
//>> 视图不改数据，但要替它记一笔：源张量多了一个"借住者"
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;
//>> 每出现一次 src 引用，消费者的计数加一 —— 这就是"生命周期"的量化形式

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip a">a 输入</span>
        <span class="chip b">W 权重</span>
        <span class="chip d">c 偏置</span>
        <span class="arrow">-></span>
        <span class="chip c">h = mul_mat(W, a)</span>
        <span class="arrow">-></span>
        <span class="chip d">y = add(h, c)</span>
        <span class="arrow">-></span>
        <span class="chip e">out</span>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['张量', 'n_children', 'n_views', '第一趟结束时它欠了几笔'],
      [['a（带 INPUT 标记）', '1', '0', 'mul_mat 用完就归零'],
       ['W（权重叶子）', '1', '0', 'mul_mat 用完就归零'],
       ['c（偏置叶子）', '1', '0', 'add 用完就归零'],
       ['h = mul_mat(W, a)', '1', '0', 'add 用完就归零（还会被原地接管）'],
       ['y = add(h, c)', '0', '0', '它是图输出，永不为零']],
      { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一趟的产物只有两个整数：<span class="v">n_children</span> 与 <span class="v">n_views</span>。',
      '<span class="v">graph-&gt;leafs[]</span> 里的张量先全部落位（第 725-727 行）—— 它们不属于任何一次"执行"，可以认为活到最后。',
      '带 <span class="v">GGML_TENSOR_FLAG_INPUT</span> 的张量在第二趟之前就抢先落位（第 744-746 行），免得地址被别人占掉。',
      '视图不给源张量加 n_children，而是加 <span class="v">n_views</span>（第 739-742 行）：<span class="k">借用数据的人，和消费数据的人，要分开计数</span>。',
      '其余每一条 <span class="v">src[j]</span> 引用让 n_children 加一（第 754 行）。<span class="k">两个计数同时归零，才允许释放</span> —— 这是第二趟的唯一判据。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(4200 + i * 3400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(21000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 ★ 第二趟：按拓扑序分配，<span class="hl-b">用完就还</span> */
{
  kicker: "L4-01 · 核心",
  title: "★ 第二趟：按拓扑序分配，<span class=\"hl-b\">用完就还</span>",
  sub: "a -> mul_mat -> add -> out 这张小图跑一遍，看内存怎么在四个格子之间换手。",
  caption: "分配在 763-778 行，释放判据在 805-819 行 —— 同一个循环里一进一出。",
  src: "ggml/src/ggml-alloc.c",
  mark: [5, 15, 37, 42, 54],
  lineNo: 763,
  code: `    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
//>> ★ 两个计数同时归零 = 这个张量再也不会被读到；不是视图就直接还给空闲表（第 816-817 行）
                if (ggml_impl_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\\n");
        }
    }`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip a">a</span>
        <span class="chip b">W</span>
        <span class="chip d">c</span>
        <span class="arrow">-></span>
        <span class="chip c">h = mul_mat(W, a)</span>
        <span class="arrow">-></span>
        <span class="chip d">y = add(h, c)</span>
        <span class="arrow">-></span>
        <span class="chip e">out</span>
      </div>
      <div class="cm" id="cap" style="margin:0">同一个计算 buffer 里的 4 格（示意；每格对应一次分配）</div>
      <div class="row" id="strip" style="gap:6px;justify-content:center;align-items:stretch"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const strip = wrap.querySelector('#strip');
    const cells = [
      { t: 'x = a', s: '4096 B' },
      { t: 'W', s: '32768 B' },
      { t: 'c', s: '4096 B' },
      { t: 'h = mul_mat', s: '32768 B' }
    ];
    const els = cells.map(c => {
      const e = U.el('div', { style: 'width:146px;height:46px;border:1px solid var(--border);' +
        'border-radius:5px;background:#10151b;display:flex;flex-direction:column;align-items:center;' +
        'justify-content:center;gap:1px;transition:all .45s' });
      e.innerHTML = '<div style="font-family:var(--mono);font-size:9.5px;color:var(--muted)">' + c.t + '</div>' +
        '<div style="font-family:var(--mono);font-size:9px;color:var(--dim)">' + c.s + '</div>';
      strip.appendChild(e);
      return e;
    });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '第一趟已经把账算好，第二趟只做三件事：<span class="k">给父张量分配 -> 给自己分配 -> 把父张量的 n_children 减一</span>。',
      '叶子先落位：<span class="v">x</span>、<span class="v">W</span>、<span class="v">c</span> 各占一格（第 725-727 行，带 INPUT 的另见第 744-746 行）。',
      '<span class="v">h = mul_mat(W, a)</span> 落位到第 4 格：从空闲表里取一块（第 687 行）。',
      'mul_mat 执行完，<span class="v">x</span> 与 <span class="v">W</span> 的 n_children 减到 0（第 800 行），第 816-817 行把它们还给空闲表；<span class="k">视图</span>走的是另一条分支 —— 它自己没占内存，只把源张量的 n_views 减一（第 806-814 行）。',
      '<span class="v">y = add(h, c)</span>：<span class="k">它和 h 同 layout，h 又只有它一个消费者</span>，于是直接接管 h 的地址（第 671-677 行），<b>0 新字节</b>。',
      '总账：这块 buffer 只需要 4 格。若每步各要一块、且不回收，同样这张图要 5 格起步 —— 而真实模型是每层都重复这一幕。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3400, () => {
      [0, 1, 2].forEach(i => { els[i].style.background = 'rgba(88,166,255,.18)'; els[i].style.borderColor = 'var(--a)'; });
      msg.innerHTML = texts[1];
    });
    tl.at(6800, () => { els[3].style.background = 'rgba(63,185,80,.18)'; els[3].style.borderColor = 'var(--b)'; msg.innerHTML = texts[2]; });
    tl.at(10300, () => {
      [0, 1].forEach(i => { els[i].style.background = '#10151b'; els[i].style.borderColor = 'var(--dim)'; els[i].style.opacity = '.45'; });
      msg.innerHTML = texts[3];
    });
    tl.at(13800, () => {
      els[3].style.borderColor = 'var(--d)';
      els[3].querySelector('div').textContent = 'y = add（接管 h）';
      els[3].querySelectorAll('div')[1].textContent = '0 新字节';
      msg.innerHTML = texts[4];
    });
    tl.at(17600, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 视图：<span class="hl-e">跳过分配</span>，必要时连地址一起接管 */
{
  kicker: "L4-01 · 视图",
  title: "视图：<span class=\"hl-e\">跳过分配</span>，必要时连地址一起接管",
  sub: "判据只有一行：view_src 非空就不进分配分支；能复用的时候连父张量那格都省下来。",
  caption: "回顾 L1-01：RESHAPE / VIEW / TRANSPOSE 只改 ne[]、nb[]、view_src、view_offs，不复制数据。",
  src: "ggml/src/ggml-alloc.c",
  mark: [4, 36, 40, 51],
  lineNo: 623,
  code: `static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
//>> 两个条件同时成立才分配：没被分配过，且不是视图 —— 视图永远不从这里要内存
        hn->allocated = true;
        assert(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
//>> layout 不同就不能原地复用 —— 尺寸与步长对不上
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_impl_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
//>> 三件事都满足才敢接管：源张量只有这一个视图、没有别的消费者、且视图正好覆盖它的全部数据
                            AT_PRINTF("reusing view parent %s (%s) for %s\\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        return;
                    }
                }
            }`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row center" style="gap:10px" id="dia"></div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const dia = wrap.querySelector('#dia');
    const base = U.el('div', { class: 'card', style: 'width:286px;border-left-color:var(--a)' });
    base.innerHTML = '<div class="ct" style="color:var(--a)">基张量 h</div>' +
      '<div class="cb">拥有 data 与 buffer<br><span class="cm" style="margin:0">view_src == NULL</span></div>' +
      '<div style="display:flex;gap:2px;margin-top:6px" id="cells"></div>';
    const view = U.el('div', { class: 'card', style: 'width:286px;border-left-color:var(--e)' });
    view.innerHTML = '<div class="ct" style="color:var(--e)">视图 v = ggml_view_2d(...)</div>' +
      '<div class="cb">占 0 字节：它没有自己的内存<br><span class="cm" style="margin:0">view_src = h</span><br>' +
      '<span class="cm" style="margin:0">view_offs = 0</span></div>';
    dia.appendChild(base); dia.appendChild(U.arrow('->')); dia.appendChild(view);

    const cells = base.querySelector('#cells');
    const cs = [];
    for (let i = 0; i < 6; i++) {
      const c = U.el('div', { style: 'width:36px;height:17px;border:1px solid var(--border);border-radius:3px;' +
        'background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
      c.textContent = i;
      cells.appendChild(c); cs.push(c);
    }

    const host = wrap.querySelector('#cards');
    const defs = [
      { c: 'a', t: '判据一：跳过分配', b: 'view_src 非空 -> 第 627 行整个分配分支不进', m: '!ggml_impl_is_view(node)' },
      { c: 'b', t: '判据二：接管父张量', b: 'n_children == 1 && n_views == 0 且同 layout', m: 'p_hn->n_children == 1' },
      { c: 'c', t: '判据三：接管视图的源', b: '源只有这一个视图、没有消费者、且视图覆盖全部数据', m: 'view_src->data == parent->data' }
    ];
    const els2 = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '219px' }); host.appendChild(e); return e; });
    els2.forEach(e => { e.style.opacity = '.30'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '视图不拥有内存，所以分配器对它的处理和普通张量完全不同：<span class="k">它一进来就被排除在分配之外</span>。',
      '但这带来一个麻烦：视图<b>借住</b>在源张量身上，源张量就不能随便被回收 —— 所以第一趟要单独数 <span class="v">n_views</span>（第 739-742 行）。另外，带 OUTPUT 标记的父张量（以及它的视图）一律不许被覆盖（第 646 行）。',
      '更妙的是第 651-677 行：如果 h 只有 v 一个视图、v 只有 add 一个消费者、layout 又一致，<span class="k">add 的输出可以直接接管 h 的地址</span>（第 665 与 674 行）—— 前提是这个算子在 <span class="v">ggml_op_can_inplace</span> 的白名单里（第 22-51 行，ADD / MUL / RMS_NORM / SOFT_MAX ...）。',
      '接管时把 <span class="v">p_hn-&gt;allocated</span> 置为 false（第 666-667 行），等于告诉后面的释放逻辑"这块不是你的，别还" —— 否则会把正在用的地址还进空闲表。',
      'L1-01 讲的是视图<span class="k">不复制数据</span>；这里补上后半句：<span class="k">视图链上的地址可以在算子之间整块转让</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [4]); });
    tl.at(4500, () => { msg.innerHTML = texts[1]; U.markLines(document, []); });
    tl.at(8600, () => {
      [0, 1, 2, 3, 4, 5].forEach(i => { cs[i].style.background = 'rgba(88,166,255,.18)'; cs[i].style.borderColor = 'var(--a)'; });
      msg.innerHTML = texts[2]; U.markLines(document, [36, 40]);
    });
    tl.at(13000, () => { msg.innerHTML = texts[3]; U.markLines(document, [45, 46, 47]); });
    tl.at(16800, () => { els2.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, []); });
  }
},

/* ------------------------------------------------------ 9 ★ 图没变，就<span class="hl-f">不重新规划</span> */
{
  kicker: "L4-01 · 图重放",
  title: "★ 图没变，就<span class=\"hl-f\">不重新规划</span>",
  sub: "alloc_graph 先问 needs_realloc：节点数、叶子数、每个张量的大小都没变，就直接用上次的规划落位。",
  caption: "呼应 L2-07 的 graph reuse：decode 每一步形状相同 -> 图一样 -> 内存规划也一样，于是每步只做一次\"发地址\"。",
  src: "ggml/src/ggml-alloc.c",
  mark: [1, 3, 9, 19, 33],
  lineNo: 1052,
  code: `bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
//>> needs_realloc 逐节点比对：节点数、叶子数、每个张量需要的字节数（第 1009-1050 行）
        if (galloc->n_buffers == 1) {
//>> ★ 单 buffer 在第 1058 行自动重规划；多 buffer 在第 1061-1065 行直接返回 false，要求先调 reserve_n
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
//>> 命中复用：只把 buffer 清一遍，不重新算任何 offset
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
//>> 然后照上一次的规划结果逐个落位（init_tensor，第 970-995 行）
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="row" style="gap:9px">
        <div class="col grow" style="gap:6px" id="l"></div>
        <div class="col grow" style="gap:6px" id="r"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['图变了没有', 'needs_realloc', 'alloc_graph 会做什么'],
      [['节点数 / 叶子数变了', 'true', '单 buffer 自动 reserve 重规划；多 buffer 返回 false，要你先调 reserve_n'],
       ['某个张量变大了', 'true', '同上 —— 老的 size_max 已经装不下'],
       ['完全一样（decode 的常态）', 'false', '只清一遍 buffer 再 init_tensor 落位，0 次规划']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#l').innerHTML =
      '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">第一次 / 图变了</div>' +
      '<div class="cb">reserve：清空 dyn_tallocr -> 两趟规划 -> 按水位线开 buffer。<br>' +
      '<span class="cm" style="margin:0">ggml_gallocr_reserve_n_impl</span></div></div>';
    wrap.querySelector('#r').innerHTML =
      '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">之后每一步</div>' +
      '<div class="cb">alloc_graph：比一遍 -> 清 buffer -> 落位。<br>' +
      '规划结果一直存在 node_allocs / leaf_allocs 里，<br>' +
      '<span class="cm" style="margin:0">重放只花一次比对的代价</span></div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'alloc_graph 的第一件事不是分配，是<b>先问一句"要不要重来"</b>（第 1053 行）。',
      '<span class="v">needs_realloc</span> 会逐节点问：老的 <span class="v">size_max</span> 还装得下你吗（第 997-1007 行）？只要有一处不满足，就整体重规划。',
      '单 buffer 时它替你重规划（第 1058 行）；<span class="k">多 buffer 时不猜</span>，直接返回 false（第 1061-1065 行）—— 因为节点该放哪个 buffer 是调度器的决定（L4-02）。',
      '命中复用后只做两件事：<span class="v">ggml_vbuffer_reset</span> 清一遍（第 1069-1074 行），再对每个叶子与节点调 <span class="v">ggml_gallocr_init_tensor</span> 落位（第 1076-1095 行）。',
      '这就是 L2-07 里 graph reuse 那笔账的另一半：<span class="k">图复用省掉建图，内存复用省掉规划</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3600 + i * 2900, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 3)];
    }));
    tl.at(17500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 内存规划的输出，与这一课的一张表 */
{
  kicker: "L4-01 · 收束",
  title: "内存规划的输出，与这一课的一张表",
  sub: "get_buffer_size 报出的那个数，就是这张图的执行缓冲大小；多 buffer 时按 buffer 分别报。",
  caption: "下一课 L4-02：调度器怎么决定每个节点进哪个 buffer，以及为什么切分处必须插 copy 节点。",
  src: "ggml/include/ggml-alloc.h",
  mark: [0, 1, 4],
  lineNo: 70,
  code: `// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案', '证据（ggml-alloc.c 行号）'],
      [['为什么要自己管理内存？', '生命周期在建图时就已知 -> 一次规划出峰值，而不是每个张量各占一块', '723-761 / 763-822'],
       ['内存怎么复用？', 'free_block 空闲表 + best fit 取块；还块时与左右邻居就地合并', '110-133 / 202-224 / 311-350'],
       ['视图怎么处理？', 'view_src 非空 -> 跳过分配；条件满足时把源张量的地址整块转让', '627 / 657-670'],
       ['图重放为什么不用重规划？', 'needs_realloc 比对图形态，没变就只用上次的结果落位', '1009-1050 / 1052-1098'],
       ['多 buffer 怎么办？', 'ggml_gallocr_new_n + 每个节点的 buffer_id，交给 L4-02 的调度器', '482-496 / 498-532']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '给定小图 <span class="mono">a -> h = mul_mat(W, a) -> y = add(h, c) -> out</span>' +
      '（a 是输入，W / c 是叶子，out 带 OUTPUT 标记）。' +
      '① 为什么 <span class="mono">y</span> 可以不要新内存？' +
      '② <span class="mono">a</span> 和 <span class="mono">W</span> 在哪一刻被还给空闲表？' +
      '③ 为什么"每个张量各 ggml_new_tensor 一块"在小图上没问题，在 70 层 transformer 上会爆？',
      '① <span class="mono">ggml_op_can_inplace(GGML_OP_ADD)</span> 为真（第 22-51 行），' +
      '而 <span class="mono">h</span> 只被 <span class="mono">add</span> 一个节点引用（n_children == 1）、没有视图（n_views == 0）、' +
      'layout 又与输出一致（第 651 行）—— 于是第 671-677 行直接把 <span class="mono">h</span> 的地址转给 ' +
      '<span class="mono">y</span>，并在第 675 行把 <span class="mono">p_hn-&gt;allocated</span> 置 false，免得它被当成"可释放"。<br><br>' +
      '② 执行完 <span class="mono">mul_mat</span> 之后，第 800 行的 <span class="mono">p_hn-&gt;n_children -= 1</span> 把它们的计数减到 0，' +
      '第 816-817 行随即调用 <span class="mono">ggml_gallocr_free_node</span> 把地址还给空闲表（a、W 都是叶子，只会被引用这一次）。<br><br>' +
      '③ 小图里"各占一块"只浪费一格；70 层里每一层都有若干中间张量，它们<b>不同时活着</b>，' +
      '而"各占一块"只按总和算 —— 总和对峰值是数量级的差距。' +
      'gallocr 用空闲表把死者让出的洞立刻发给下一个人（第 263-268 行取块、第 311-350 行还块），' +
      '所以 buffer 只需要按<b>同时存活的峰值</b>开一次。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '五句话收束这一课：',
      '<span class="k">为什么</span>：生命周期已知 -> 可以提前规划（723-761 数账，763-822 分配与释放）。',
      '<span class="k">怎么复用</span>：free_block 空闲表 + best fit 取块 + 就地合并还块。',
      '<span class="k">视图</span>：跳过分配；能整体转让时连源张量那一格都省下。',
      '<span class="k">代价</span>：每次图形态变化都要重规划 —— 单 buffer 自动做（第 1058 行），多 buffer 必须你自己先 reserve_n。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
