/* ==========================================================================
   L2-03 · 权重加载与内存映射
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一个权重 = 一段字节区间 + 一个张量 */
{
  kicker: "L2 · 从模型到图",
  title: "一个权重 = 一段字节区间 + 一个张量",
  sub: "loader 在建图之前就记下每个权重的（文件号，文件内偏移）；落到哪由后面决定。",
  caption: "回顾 L1-01：`data` / `buffer` 是 ggml_tensor 的\"位置\"字段 —— 本课讲它们怎么被填上。",
  src: "src/llama-model-loader.h",
  mark: [1, 2, 5, 13],
  lineNo: 34,
  code: `    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file
//>> 文件号：分片模型（split）由多个 .gguf 组成，idx 指是哪一片

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
//>> offs 是文件内的【绝对字节偏移】= 数据段基址 + 该张量在数据段内的偏移
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }`,
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">GGUF 分片文件</span><span class="arrow">-></span>
        <span class="chip a">llama_tensor_weight</span><span class="arrow">-></span>
        <span class="chip b">ggml_tensor</span><span class="arrow">-></span>
        <span class="chip c">后端 buffer</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#cards');
    const defs = [
      { c: 'a', t: '从哪来', b: '文件号 idx + 文件内偏移 offs。<br>构造时就查 GGUF 并做边界检查。', m: 'uint16_t idx;  size_t offs;' },
      { c: 'b', t: '落到谁', b: 'loader 只认张量名字：<br>名字 -> 权重的表就是 weights_map。', m: 'ggml_tensor * tensor;' },
      { c: 'c', t: '落到哪', b: '由建图时选出的 buffer type<br>决定，GGUF 里没有任何设备信息。', m: 'tensor->buffer / tensor->data' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一课回答三个问题：<span class="v">从哪来</span>、<span class="v">怎么搬</span>、<span class="v">落到哪</span>。',
      '<span class="k">从哪来</span>：权重 = (文件号 idx, 文件内偏移 offs)，<br>再加一个等着被填充的张量指针。',
      '构造时还有一道 <span class="v">file->size()</span> 边界检查：<br>数据必须真的落在文件里，否则模型被判为损坏。',
      '<span class="k">落到哪</span>没法从文件回答 —— 所以才有第 7 幕的"谁决定 buffer type"。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
  }
},

/* ------------------------------------------------------ 2 <span class="hl-c">load_mode</span> 只折成两个布尔 */
{
  kicker: "L2-03 · 开关",
  title: "<span class=\"hl-c\">load_mode</span> 只折成两个布尔",
  sub: "要不要 mmap、要不要 direct I/O，在构造 loader 时就定死，之后整条加载路径都按它分支。",
  caption: "命令行上的对应开关是 --load-mode（实测 llama-cli --help 里有 -lm, --load-mode MODE）。",
  src: "src/llama-model-loader.cpp",
  mark: [0, 1],
  lineNo: 559,
  code: `    this->use_mmap      = load_mode == LLAMA_LOAD_MODE_MMAP || load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK || load_mode == LLAMA_LOAD_MODE_AUTO;
    this->use_direct_io = load_mode == LLAMA_LOAD_MODE_DIRECT_IO;
//>> direct I/O 走的是 O_DIRECT 整块读，与 mmap 互斥；两者都不开就是普通 read`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['load_mode', 'use_mmap', 'use_direct_io', '加载手段'],
      [['LLAMA_LOAD_MODE_AUTO  (-1)', 'true', 'false', '默认；平台不支持时自动关掉 mmap'],
       ['LLAMA_LOAD_MODE_NONE  (0)', 'false', 'false', '普通 read'],
       ['LLAMA_LOAD_MODE_MMAP  (1)', 'true', 'false', '映射整个文件'],
       ['LLAMA_LOAD_MODE_MLOCK  (2)', 'false', 'false', '普通 read + 给页加锁'],
       ['LLAMA_LOAD_MODE_MMAP_MLOCK (3)', 'true', 'false', '映射 + 给页加锁'],
       ['LLAMA_LOAD_MODE_DIRECT_IO (4)', 'false', 'true', '绕开页缓存的整块读']],
      { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '六个 API 取值，落到代码里只是这一行折出来的两个布尔：',
      '<span class="v">AUTO / MMAP / MMAP_MLOCK</span> -> <span class="k">use_mmap = true</span>；只有 AUTO 带"自动判定"的语义。',
      '<span class="v">NONE / MLOCK</span> 都是普通 read，区别只在调用方要不要把页锁住（mlock 由调用方传 mlock_mmaps 进来）。',
      '<span class="v">DIRECT_IO</span> 打开 O_DIRECT（llama-mmap.cpp:198-216），读偏移和长度都要按块对齐 —— 第 6 幕会看到它的影响。',
      '平台不支持时（<span class="v">llama_mmap::SUPPORTED == false</span>）构造末尾把 <span class="v">use_mmap</span> 强行置回 false（llama-model-loader.cpp:829-832）。',
      '所以"用不用 mmap"是【构造时】定下的，运行期不再变化。'
    ];
    const rowtext = [1, 2, 1, 2, 1, 3];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[rowtext[i]];
    }));
    tl.at(16600, () => {
      rows.forEach((x, k) => { x.className = (k === 0 || k === 2 || k === 4) ? 'on' : ''; });
      msg.innerHTML = texts[4];
    });
    tl.at(18600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 3 llama_mmap 的门面：三个操作 */
{
  kicker: "L2-03 · 映射",
  title: "llama_mmap 的门面：三个操作",
  sub: "建立映射、拿基地址、按页归还。三个平台分支（POSIX / Win32 / 不支持）藏在 pimpl 后面。",
  caption: "loader 持有 `llama_mmaps mappings`（llama-model-loader.h:124），构造它的地方是 init_mappings（同文件 1410-1447 行）。",
  src: "src/llama-mmap.h",
  mark: [0, 5, 9, 10, 13, 15],
  lineNo: 44,
  code: `struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;
//>> addr() 是零拷贝的入口：权重数据地址 = addr() + 文件内偏移

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;
//>> SUPPORTED 是编译期常量；不支持时构造会直接抛 "mmap not supported"

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#cards');
    const defs = [
      { c: 'a', t: '① 建立映射', b: '构造即映射：整个文件一段，<br>prefetch / numa / lazy_ranges 都是提示。', m: 'llama_mmap(file, prefetch, numa, lazy_ranges)' },
      { c: 'b', t: '② 拿基地址', b: '映射的虚拟地址 + size。<br>权重数据地址 = addr() + offs。', m: 'void * addr() const;' },
      { c: 'c', t: '③ 归还页', b: '按页粒度还回去，加载结束后<br>把元数据、未用张量的页释放掉。', m: 'void unmap_fragment(size_t first, size_t last);' },
      { c: 'd', t: '④ 平台门', b: 'POSIX / Win32 / 不支持三分支，<br>对外只暴露 SUPPORTED 一个常量。', m: 'static const bool SUPPORTED;' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先把接口读成三个动作 + 一个平台门。',
      '<span class="k">构造即映射</span>：没有单独的 open() —— 对象活着的期间映射就有效，析构函数负责 munmap。',
      '<span class="v">addr()</span> 是后面 load_all_data 里 <span class="v">mapping->addr() + weight->offs</span> 的来源。',
      '<span class="v">unmap_fragment</span> 的存在说明映射不是"要么全在要么全不在"：可以只把某段页还回去。',
      '注释写明 ranges 是 <span class="v">[first, last)</span> 区间 —— lazy 读要把"哪些段留随机访问"告诉映射层。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13400, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-a">mmap</span>：省掉的是拷贝，不是必然的 I/O */
{
  kicker: "L2-03 · 核心",
  title: "★ <span class=\"hl-a\">mmap</span>：省掉的是拷贝，不是必然的 I/O",
  sub: "映射只改页表；字节什么时候从磁盘进来，由 prefetch / lazy / numa 三个提示决定。",
  caption: "回顾 L1-06：映射要求张量数据在文件里按对齐偏移连续排布 —— GGUF 的数据段正好如此（对比 llama-model-loader.cpp:688-693 的对齐检查）。",
  src: "src/llama-mmap.cpp",
  mark: [4, 12, 14],
  lineNo: 469,
  code: `    impl(struct llama_file * file, size_t prefetch, bool numa, const llama_mmap::ranges & lazy_ranges) {
//>> 映射粒度是整个文件；prefetch / numa / lazy_ranges 只影响"页什么时候进来"
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\\n",
                    strerror(errno));
        }
        // MAP_POPULATE would fault in the lazy ranges too
        if (prefetch && lazy_ranges.empty()) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
//>> PROT_READ：只读映射。内核知道没人会写，这些页可以与文件页缓存共享
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin:0 0 2px">同样读一遍权重，两条路的差别</div>';
    const t = U.table(
      ['', 'mmap', 'read'],
      [['加载时做什么', '建映射（改页表）', '把字节读进用户缓冲'],
       ['谁付磁盘 I/O', '第一次访问该页时', '加载时一次付清'],
       ['页缓存', '直接用文件的页', '页 -> 用户缓冲，多一次拷贝'],
       ['常驻内存', '被访问过的页', '全部权重'],
       ['可调项', 'prefetch / numa / lazy', '基本没有']],
      { monoCols: [0] });
    left.appendChild(t.el);

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">MAP_POPULATE（480 行）</div>' +
      '<div class="cb">prefetch 非 0 且没有 lazy 区间时才加：一次把页表建好。' +
      '只要有 lazy 区间就必须关掉 —— 否则会把本该懒读的页也拉进来。</div></div>' +
      '<div class="card" style="border-left-color:var(--f)">' +
      '<div class="ct" style="color:var(--f)">madvise（500-507 行）</div>' +
      '<div class="cb">prefetch 区间提示 <span class="cm" style="margin:0">POSIX_MADV_WILLNEED</span>（顺序读），' +
      'lazy 区间提示 <span class="cm" style="margin:0">POSIX_MADV_RANDOM</span>（别看太远）。</div></div>' +
      '<div class="card" style="border-left-color:var(--d)">' +
      '<div class="ct" style="color:var(--d)">numa（473、508 行）</div>' +
      '<div class="cb">NUMA 机器上直接否决预取：<span class="cm" style="margin:0">prefetch = 0</span>，' +
      '并对全文件用 MADV_RANDOM —— 避免把远端内存拉满。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '核心只有一行：<span class="v">mmap(NULL, file->size(), PROT_READ, MAP_SHARED, fd, 0)</span>。',
      '<span class="k">只读映射</span>：内核知道这些页不会被写，可以直接与文件页缓存共享，不需要私有副本。',
      '映射建立是 O(1)，但<span class="k">字节什么时候进来是可调的</span>：MAP_POPULATE 立刻预取，否则留到访问时。',
      '源码注释把边界说清楚了：<span class="v">MAP_POPULATE would fault in the lazy ranges too</span> —— 预取与懒读互斥。',
      '因此准确的说法是：mmap 省掉【用户态缓冲 + 一次拷贝】，并把 I/O 变成按页触发；<br>它并不自动等于"加载时零 I/O"。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [14]); });
    tl.at(4000, () => { msg.innerHTML = texts[1]; U.markLines(document, [14]); });
    tl.at(7600, () => { msg.innerHTML = texts[2]; U.markLines(document, [12]); });
    tl.at(11200, () => { msg.innerHTML = texts[3]; U.markLines(document, [12]); });
    tl.at(14800, () => { msg.innerHTML = texts[4]; U.markLines(document, [4, 12, 14]); });
  }
},

/* ------------------------------------------------------ 5 ★ <span class="hl-a">指过去</span>还是<span class="hl-b">拷过去</span> */
{
  kicker: "L2-03 · 核心",
  title: "★ <span class=\"hl-a\">指过去</span>还是<span class=\"hl-b\">拷过去</span>",
  sub: "load_all_data 的第一句判据：数据是\"已经有地址\"，还是\"还得从文件搬\"。",
  caption: "lazy 读（TENSOR_READ_LAZY）也走这一支：即使整体不用 mmap，被标记的大张量仍会拿到映射指针。",
  src: "src/llama-model-loader.cpp",
  mark: [2, 11, 20, 21, 33, 34],
  lineNo: 1638,
  code: `        size_t n_size = ggml_nbytes(cur);

        const bool from_mapping = use_mmap || lazy.has(cur);
//>> 一行判据：走映射（use_mmap 或这个张量被标记为懒读），还是走 read

        if (from_mapping) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs.count(weight->idx)) {
                buf_mmap = bufs.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);
//>> 把张量直接指到映射里的那个地址 —— 零拷贝落位，页错误推迟到内核读它的时候

                // locking a lazy tensor would fault all of it in, which is what lazy avoids
                if (lmlocks && !lazy.has(cur)) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const dia = wrap.querySelector('#dia');

    const box = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
    box.innerHTML = '<div class="ct" style="color:var(--a)">分支 A · 在映射里</div>' +
      '<div class="cb">data = mapping->addr() + weight->offs<br>' +
      '<b>cur->data == nullptr</b> 时直接把它挂上去：</div>' +
      '<div class="cm" style="margin:4px 0 0">ggml_backend_tensor_alloc(buf_mmap, cur, data);</div>' +
      '<div class="cb" style="margin-top:4px">代价：<b>页错误</b>推迟到后端内核第一次读它。</div>';

    const box2 = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
    box2.innerHTML = '<div class="ct" style="color:var(--b)">分支 B · 已经有 buffer</div>' +
      '<div class="cb">张量已经分配在自己的 buffer 里，<br>映射里的数据要拷过去：</div>' +
      '<div class="cm" style="margin:4px 0 0">ggml_backend_tensor_set(cur, data, 0, n_size);</div>' +
      '<div class="cb" style="margin-top:4px">代价：一次 memcpy，但页用完就可以放。</div>';

    dia.appendChild(box); dia.appendChild(U.arrow('->')); dia.appendChild(box2);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '映射里的数据要变成张量的 data，有两条出路，判据是"张量是不是还没地方放"。',
      '<span class="k">分支 A</span>：<span class="v">buf_mmap && cur->data == nullptr</span> —— 这个文件已经有对应的后端 buffer，直接把 data 指过去。',
      '指过去之后，源码顺手记下这个文件里【真正用过】的页区间（<span class="v">mmaps_used</span>，1666-1668 行），收尾时按它归还。',
      '<span class="k">分支 B</span>：张量已经有自己的 buffer（例如显存里），那就 <span class="v">ggml_backend_tensor_set</span> 拷一次。',
      '两条路都不做"选后端"这件事 —— buffer 是上一幕之前就按 buffer type 分好的（见第 7 幕）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [2]); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [20, 21]); });
    tl.at(7200, () => { msg.innerHTML = texts[2]; U.markLines(document, [11]); });
    tl.at(10800, () => { msg.innerHTML = texts[3]; U.markLines(document, [33, 34]); });
    tl.at(14400, () => { msg.innerHTML = texts[4]; U.markLines(document, [2, 21, 34]); });
  }
},

/* ------------------------------------------------------ 6 不走映射时：三条搬运路径 */
{
  kicker: "L2-03 · read 侧",
  title: "不走映射时：三条搬运路径",
  sub: "host buffer 直接读；设备 buffer 有异步能力就走 pinned host buffer 中转；没有就现开临时缓冲。",
  caption: "张量会按\"需要中转的排前面、大的排前面\"稳定排序（同一文件 1614-1623 行），让暂存峰值最小。",
  src: "src/llama-model-loader.cpp",
  mark: [2, 4, 13, 23, 26],
  lineNo: 0,
  code: `            const auto & file = files.at(weight->idx);

            if (ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
//>> 设备 buffer：后端支持异步时，先把文件块读进 pinned host buffer，再异步上传 —— 这就是 host 中转
                if (upload_backend) {
                    size_t offset = weight->offs;
                    alignment = file->read_alignment();
//>> direct I/O 模式下 alignment > 1，偏移与长度都要按块对齐（read_alignment() 来自 st_blksize）
                    size_t aligned_offset = offset & ~(alignment - 1);
                    size_t offset_from_alignment = offset - aligned_offset;
                    file->seek(aligned_offset, SEEK_SET);
//>> ---- src/llama-model-loader.cpp:1737-1746 ----
                } else {
                    // scoped to one tensor so only one staging buffer is alive at a time
                    std::vector<no_init<uint8_t>> read_buf(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#cards');
    const defs = [
      { c: 'b', t: '① host buffer：直读', b: '<b>ggml_backend_buffer_is_host</b> 为真时，' +
          'seek 到偏移后直接 read 进张量的 data —— 没有中转缓冲。', m: 'file->read_raw(cur->data, n_size);' },
      { c: 'a', t: '② 设备 buffer + 异步', b: '先建 4 个 pinned host buffer（1MB，' +
          'direct I/O 时 64MB + 对齐），分块读进来再异步上传。', m: 'ggml_backend_tensor_set_async(...)' },
      { c: 'c', t: '③ 设备 buffer + 同步', b: '没有异步能力就现开一个临时缓冲：读一份、set 过去、释放。' +
          '注释说明它按张量作用域分配。', m: 'std::vector<no_init<uint8_t>> read_buf(n_size);' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '非 mmap 路径要先问一句：这个张量住的 buffer 是主存还是设备？',
      '<span class="k">主存（host buffer）</span>：文件字节直接落到张量上，读一次就完事。',
      '<span class="k">设备</span>且后端有 async / host_buffer / events 三项能力（1526-1598 行）：走 pinned host buffer 中转。',
      '中转是为了让 <span class="v">ggml_backend_tensor_set_async</span> 与下一次磁盘读重叠 —— 代价是一份额外的 pinned 内存。',
      '都没有时退化成最朴素的一版：临时 buffer 读一份、<span class="v">ggml_backend_tensor_set</span> 拷到设备。',
      '三条路的共同点：<span class="k">张量早就落好位了</span>，这里只负责把字节送过去。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(12800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 7 ★ 谁决定权重落到哪个后端 */
{
  kicker: "L2-03 · 核心",
  title: "★ 谁决定权重落到哪个后端",
  sub: "答案是一串判据 + 一个函数：按 buft_list 的顺序问\"这个 buffer type 能不能跑用到这个权重的算子\"。",
  caption: "对照 L3-01：`ggml_backend_buffer_type_t` 与 `ggml_backend_dev_supports_op` 是后端契约的一部分；这里的判据完全建立在它之上。",
  src: "src/llama-model-loader.cpp",
  mark: [1, 4, 7, 9],
  lineNo: 1066,
  code: `// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t * buft_list) {
//>> buft_list 是"这一层可以用的 buffer type 列表"，顺序就是优先级（list 由调用方按 offload 参数排好）
    GGML_ASSERT(!buft_list->empty());
    for (const auto & cur : *buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
//>> 判据不在这个函数里：weight_buft_supported 给权重造一个假算子，挂 0 字节哑 buffer，再问后端 supports_op
            return cur_buft;
        }
    }

    return nullptr;
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="flow" style="gap:4px"></div>
        <div class="col grow" id="side" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const steps = [
      ['1', 'a', 'load_mode 折出 use_mmap / use_direct_io（559-560 行）'],
      ['2', 'c', '张量属于哪一层：input / output / repeating -> 取对应的 buft_list（1215-1230 行）'],
      ['3', 'd', 'tensor_buft_overrides 命中？命中就用它；命中 CPU 时再走 select_weight_buft（1235-1260 行）'],
      ['4', 'b', '否则 select_weight_buft：按列表顺序试（1067-1075 行）'],
      ['5', 'b', '判据：假算子 + 0 字节哑 buffer -> ggml_backend_dev_supports_op（1056-1059 行）'],
      ['6', 'c', '修正：use_mmap 且选中的是设备的 host buffer -> 换成 CPU 的 buffer type（1269-1277 行）'],
      ['7', 'd', '产物：按 buft 取（或新建）一个 ggml context，张量就建在那里（1122-1152、1380 行）'],
      ['8', 'a', '执行：load_all_data 指过去 / 拷过去 / 中转（1638-1747 行）']
    ];
    const host = wrap.querySelector('#flow');
    host.innerHTML = '<div class="cm" style="margin:0 0 2px">决定链（从上到下）</div>';
    const els = steps.map(s => {
      const e = U.el('div', { class: 'formula', style: 'padding:3px 7px;display:flex;gap:6px;align-items:baseline' });
      e.innerHTML = '<span class="chip ' + s[1] + '" style="flex:0 0 auto">' + s[0] + '</span><span>' + s[2] + '</span>';
      host.appendChild(e);
      return e;
    });
    els.forEach(e => { e.style.opacity = '.32'; });

    const side = wrap.querySelector('#side');
    side.innerHTML =
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">★ 决定的时刻</div>' +
      '<div class="cb">建图阶段：每拿到一个权重就调用一次 <span class="cm" style="margin:0">create_tensor()</span>。' +
      '此时形状已经知道（来自 GGUF 元数据），<b>数据一个字节都还没读</b>。</div></div>' +
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">为什么不用数据来判断</div>' +
      '<div class="cb">判据是【算子的形状 + 后端能力】，不是数据内容：' +
      'weight_buft_supported 按 op 造一个假算子（MUL_MAT / GET_ROWS / ROPE ...），' +
      '再挂一个 0 字节哑 buffer 让后端自己回答。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">一个都选不上会怎样</div>' +
      '<div class="cb">列表全试完仍无解就抛 "failed to find a compatible buffer type"（1264-1266 行）；' +
      '退到非首选时会记账，收尾时打一条 debug 日志（1280-1287、1403-1407 行）。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把决定链摊开：这是本课的验收点所在。',
      '★ 答案：决定发生在 <span class="k">create_tensor()</span> 被调用的那一刻 —— 建图阶段，数据还没读。',
      '第 5 步是全部判据的核心：不只是"这个 buffer type 存不存在"，而是"用它跑这个算子行不行"。',
      '第 6 步是 use_mmap 唯一影响选择结果的地方：映射的权重宁可留在 CPU，也不放进设备的 host buffer。',
      '第 8 步只执行落位；拿到手的 <span class="v">bufs</span> 已经按文件号分好，不再挑后端。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    els.forEach((e, i) => tl.at(2600 + i * 2100, () => {
      els.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.32'; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 4)];
    }));
    tl.at(19400, () => {
      els.forEach(x => { x.style.opacity = '1'; });
      msg.innerHTML = 'L4-02 的调度器再根据"权重在哪个 buffer 里"决定算子放哪 —— 前提正是这里的落位。';
    });
  }
},

/* ------------------------------------------------------ 8 同一个工程里的另一套 I/O */
{
  kicker: "L2-03 · 对照",
  title: "同一个工程里的另一套 I/O",
  sub: "llama-io.h 的虚接口不服务权重，它服务\"状态\"：KV 与序列的存取。",
  caption: "实现类 llama_io_write_host / _file / _device / _dummy 在 src/llama-context.cpp（L2-07 覆盖），本课不引用其代码。",
  src: "src/llama-io.h",
  mark: [5, 7, 10, 12, 15, 20, 22],
  lineNo: 9,
  code: `class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
//>> 读侧与写侧对称：只有 5 个纯虚函数 —— 裸字节 + 张量分片 + 计数
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
//>> 注意 read_tensor 带 (offset, size)：状态是按张量区间搬的，不是整块
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['', '权重 I/O（本课前 7 幕）', '状态 I/O（llama-io）'],
      [['方向', '只读', '既读又写'],
       ['粒度', '整个张量（或按行懒读）', '按 (offset, size) 分片'],
       ['载体', 'llama_file + llama_mmap', 'llama_io_read_i / llama_io_write_i'],
       ['实现', 'llama-mmap.cpp', 'llama-context.cpp 的 host / file / device 三个实现'],
       ['用途', '加载 GGUF 权重', '保存与恢复 KV cache、序列状态（见 L2-04）']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '为什么权重不走这套虚接口？因为它是【只读、一次、巨大】的 —— 虚函数帮不上忙。',
      '<span class="k">方向</span>：权重只读；状态要能存能取，才需要 read / write 两侧对称的接口。',
      '<span class="k">粒度</span>：<span class="v">read_tensor/write_tensor(tensor, offset, size)</span> —— 状态按张量区间搬，可以只存某几个 cell。',
      '<span class="k">载体与实现</span>：权重用 llama_file + llama_mmap；状态用虚接口。llama-io.cpp 全文只实现带 <span class="v">uint32_t</span> 长度前缀的字符串 —— 自描述字节流。',
      '<span class="k">用途</span>：权重加载 GGUF；状态保存与恢复 KV cache、序列状态（见 L2-04）。',
      '分界很清楚：<span class="k">文件里躺着的</span>走 mmap / read，<span class="k">运行期反复存取的</span>走 llama-io。'
    ];
    const rowtext = [1, 2, 3, 3, 4];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2900 + i * 2700, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[rowtext[i]];
    }));
    tl.at(16400, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L2-03 · 收束",
  title: "把这一课压成一张表",
  sub: "两个输入、两个决策、两个执行动作。",
  caption: "下一课 L2-04：权重落位之后，运行时那块内存（KV cache 与记忆家族）怎么组织。",
  src: "src/llama-model-loader.h",
  mark: [0, 2],
  lineNo: 255,
  code: `    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
//>> bufs 的键是【文件号】：哪个文件的数据放在哪个后端 buffer 里；mmap 时它还可以是"按映射区建的 buffer"
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案', '源码锚点'],
      [['权重从哪来', 'GGUF 数据段里的字节区间（文件号 + 偏移）', 'llama-model-loader.h:34-50'],
       ['mmap 还是 read', 'load_mode 折成 use_mmap / use_direct_io 两个布尔', 'llama-model-loader.cpp:559-560'],
       ['mmap 怎么变地址', '建只读映射；addr() + offs 就是张量数据地址', 'llama-mmap.cpp:482，llama-model-loader.cpp:1648'],
       ['谁决定落到哪个后端', 'create_tensor -> buft_list -> select_weight_buft -> supports_op', 'llama-model-loader.cpp:1066-1078'],
       ['什么时候决定', '建图阶段；load_all_data 只执行落位', 'llama-model-loader.cpp:1493-1797'],
       ['要不要 host 中转', 'host buffer 直读；否则 pinned / 临时缓冲中转', 'llama-model-loader.cpp:1675-1746']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一个权重张量在什么时刻决定自己落到哪个后端？把加载模式从 mmap 改成普通 read，这个决定会变吗？',
      '决定发生在<b>建图阶段</b>：模型构建代码每拿到一个权重就调用一次 ' +
      '<span class="mono">create_tensor()</span>，它按张量所属层取出 buft_list（1215-1230 行），' +
      '再用 <span class="mono">select_weight_buft()</span>（1067-1075 行）按顺序询问；' +
      '判据是 <span class="mono">weight_buft_supported()</span>（927-1064 行）：' +
      '给权重造一个假算子、挂一个 0 字节哑 buffer，再问后端 ' +
      '<span class="mono">ggml_backend_dev_supports_op()</span>（1058-1059 行）。此时数据一个字节都没读。<br>' +
      '改加载模式确实会改变这个决定：<span class="mono">use_mmap</span> 为真时，' +
      '若选中的是设备的 host buffer，会被强制换成 CPU 的 buffer type（1269-1277 行）。<br>' +
      '而"落位动作"是稍后的 <span class="mono">load_all_data()</span> 做的：' +
      '映射就直接把 <span class="mono">cur->data</span> 指到映射地址（1658 行），' +
      '否则拷贝或中转（1670 / 1742 行）。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '六行表：前两行是输入，中间两行是决策，最后两行是执行。',
      '<span class="k">输入</span>：文件里的字节区间、以及"要不要 mmap"。',
      '<span class="k">决策</span>：谁决定（create_tensor 里的 select_weight_buft）、什么时候决定（建图阶段）。',
      '<span class="k">执行</span>：load_all_data 只负责把字节送到已经定好的位置上。',
      '记住一句话：<span class="v">mmap 决定"怎么搬"，buft_list 决定"搬到哪"，两者在建图时就都定了</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2300, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(Math.floor(i / 2) + 1, 3)];
    }));
    tl.at(16200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

];
