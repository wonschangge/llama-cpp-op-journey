/* ==========================================================================
   L2-08 · 采样器与词表：图算 logits，host 选 token
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 模型的两端：<span class="hl-a">分词</span>与<span class="hl-d">采样</span> */
{
  kicker: "L2 · 从模型到图",
  title: "模型的两端：<span class=\"hl-a\">分词</span>与<span class=\"hl-d\">采样</span>",
  sub: "图只认 token id。进来时要分词，出去时要采样 —— 两端都在 host 上，都不在图里。",
  caption: "回顾 L2-07：decode 建图、算图，产出 logits；本课接在它的后面。",
  src: "src/llama-vocab.h",
  mark: [0, 8, 14],
  lineNo: 165,
  code: `    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llama_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<llama_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  llama_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(llama_token token) const;`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">文本</span><span class="arrow">-></span>
        <span class="chip a">tokenize</span><span class="arrow">-></span>
        <span class="chip b">图 decode</span><span class="arrow">-></span>
        <span class="chip c">logits</span><span class="arrow">-></span>
        <span class="chip d">sampler</span><span class="arrow">-></span>
        <span class="chip e">token</span>
      </div>
      <div class="row center" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '入口 · 词表', b: '文本 <-> token id。<br>tokenize() 与 token_to_piece() 是一对互逆操作。',
        m: 'src/llama-vocab.h' },
      { c: 'd', t: '出口 · 采样器', b: 'logits -> token id。<br>在 host 上按链式顺序逐步裁剪分布。',
        m: 'src/llama-sampler.cpp' },
      { c: 'f', t: '共同底座 · unicode', b: '码点类别、NFD 折叠、大小写映射<br>都是生成好的静态数据表。',
        m: 'src/unicode-data.cpp' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card({ c: d.c, t: d.t, b: d.b, m: d.m, w: '220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '模型只吃 <span class="k">token id</span>。左边必须有人把文本切碎，右边必须有人把分数变成 id。',
      '入口：<span class="v">tokenize()</span> 把文本变成 token 数组；<span class="v">token_to_piece()</span> 反过来把 token 变回文本片段。',
      '中间：图（L2-07）只做一件事 —— 把 token id 变成 logits。它对"文本"和"概率"一无所知。',
      '出口：采样器把 logits 变成<b>下一个</b> token id，再喂回图。这就是自回归的回路。',
      '本课主线：<span class="k">这两端为什么都在 host 上做，而不进图？</span>'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 分词第一步：<span class="hl-c">特殊 token 先切开</span>，再按类型分派 */
{
  kicker: "L2-08 · 词表",
  title: "分词第一步：<span class=\"hl-c\">特殊 token 先切开</span>，再按类型分派",
  sub: "tokenize() 先做一次切分（BOS/EOS 这类特殊 token），再把每段交给具体 tokenizer。",
  caption: "分派表里 7 种 tokenizer：SPM / BPE / WPM / UGM / RWKV / PLAMO2 / TEST。本课细看 BPE。",
  src: "src/llama-vocab.cpp",
  mark: [0, 11, 16, 24],
  lineNo: 0,
  code: `std::vector<llama_token> llama_vocab::impl::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.");

    std::vector<llama_token> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(fragment_buffer, parse_special);
    }

    switch (get_type()) {
//>> ---- src/llama-vocab.cpp:3214-3235 ----
void llama_vocab::impl::init_tokenizer(enum llama_vocab_type type) {
    LLAMA_LOG_DEBUG("%s: initializing tokenizer for type %d\\n", __func__, type);

    switch (type) {
        case LLAMA_VOCAB_TYPE_SPM:
            tokenizer = std::make_unique<llm_tokenizer_spm>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_BPE:
            tokenizer = std::make_unique<llm_tokenizer_bpe>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_WPM:
            tokenizer = std::make_unique<llm_tokenizer_wpm>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_UGM:
            tokenizer = std::make_unique<llm_tokenizer_ugm>(vocab, precompiled_charsmap);
            break;
        case LLAMA_VOCAB_TYPE_RWKV:
            tokenizer = std::make_unique<llm_tokenizer_rwkv>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_PLAMO2:
            tokenizer = std::make_unique<llm_tokenizer_plamo2>(vocab);
            break;`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">raw_text</span><span class="arrow">-></span>
        <span class="chip c">tokenizer_st_partition</span><span class="arrow">-></span>
        <span class="chip a">按 vocab 类型分派</span><span class="arrow">-></span>
        <span class="chip e">token id[]</span>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['vocab 类型', 'tokenizer 实现', '说明'],
      [['LLAMA_VOCAB_TYPE_SPM', 'llm_tokenizer_spm', 'SentencePiece；空串特例见源码注释'],
       ['LLAMA_VOCAB_TYPE_BPE', 'llm_tokenizer_bpe', '本课重点：按 merges 的 rank 合并'],
       ['LLAMA_VOCAB_TYPE_WPM', 'llm_tokenizer_wpm', 'word-piece'],
       ['LLAMA_VOCAB_TYPE_UGM', 'llm_tokenizer_ugm', '需要 precompiled_charsmap'],
       ['LLAMA_VOCAB_TYPE_RWKV', 'llm_tokenizer_rwkv', 'RWKV 世界模型'],
       ['LLAMA_VOCAB_TYPE_PLAMO2', 'llm_tokenizer_plamo2', 'PLAMO2'],
       ['LLAMA_VOCAB_TYPE_TEST', 'llm_tokenizer', '测试用的基类']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');
    rows.forEach(r => { r.style.opacity = '.35'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '分词不是一个函数，而是<b>三段</b>：切分 -> 分派 -> 各类型自己的算法。',
      '<span class="v">fragment_buffer</span>：先把整段文本装进一个前向链表，切成"片段"。',
      '<span class="v">tokenizer_st_partition()</span>：特殊 token（BOS/EOS、控制符、用户自定义标记）在这里被<b>优先</b>摘出来，剩下的才是原始文本。',
      '<span class="v">switch (get_type())</span>：按 vocab 类型选一条实现。7 条路各有各的算法。',
      '下面三幕只看 <span class="k">BPE</span> 这一条 —— 它是当前主流 LLM 的分词方式。'
    ];
    let t0 = 700;
    tl.at(t0, () => { msg.innerHTML = texts[0]; });
    tl.at(t0 + 3000, () => { msg.innerHTML = texts[1]; U.markLines(document, [0]); });
    tl.at(t0 + 6000, () => { msg.innerHTML = texts[2]; U.markLines(document, [11]); });
    rows.forEach((r, i) => tl.at(t0 + 9000 + i * 1200, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = (i === 1) ? texts[3] : '第 ' + (i + 1) + ' 条路：' + r.querySelector('td').textContent;
      if (i === 1) U.markLines(document, [16, 24]);
    }));
    tl.at(17800, () => { rows.forEach(x => { x.className = ''; x.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [24]); });
  }
},

/* ------------------------------------------------------ 3 ★ BPE 的合并顺序由一个 <span class="hl-a">rank</span> 决定 */
{
  kicker: "L2-08 · BPE",
  title: "★ BPE 的合并顺序由一个 <span class=\"hl-a\">rank</span> 决定",
  sub: "llm_bigram_bpe 用优先队列按 rank 排序：rank 最小的相邻对最先合并，合并后只补两个新 bigram。",
  caption: "rank 来自词表的 merges 表（find_bpe_rank，见 source.md）；它随模型一起存进 GGUF，见 L1-06。",
  src: "src/llama-vocab.cpp",
  mark: [0, 2, 21, 37],
  lineNo: 0,
  code: `struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};
//>> ---- src/llama-vocab.cpp:657-689 ----
            for (int i = 1; i < (int) symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }`,
  duration: 23000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row center" id="sym" style="gap:3px"></div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="q" style="gap:5px"></div>
        <div class="col grow" id="log" style="gap:5px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const sym = wrap.querySelector('#sym');
    const cells = [];
    ['l', 'o', 'w', 'e', 's', 't'].forEach((ch, i) => {
      const c = U.el('div', { style: 'width:34px;height:26px;border:1px solid var(--border);border-radius:3px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:12px;color:var(--text)' });
      c.textContent = ch;
      sym.appendChild(c); cells.push(c);
    });
    const q = wrap.querySelector('#q');
    q.innerHTML = '<div class="cm" style="margin:0 0 2px">优先队列（rank 小的先出）</div>';
    const qrows = [
      { t: '(e,s)  rank 4', c: 'a' }, { t: '(l,o)  rank 7', c: 'b' }, { t: '(s,t)  rank 9', c: 'c' }
    ];
    const qels = qrows.map(r => {
      const e = U.el('div', { class: 'formula', style: 'padding:3px 7px;font-size:10px' });
      e.innerHTML = '<span style="color:var(--' + r.c + ')">' + U.esc(r.t) + '</span>';
      q.appendChild(e); return e;
    });
    const log = wrap.querySelector('#log');
    log.innerHTML = '<div class="cm" style="margin:0 0 2px">合并后新增的 bigram</div>' +
      '<div class="cb" id="lg" style="font-size:9.5px;color:var(--muted)">（还没开始）</div>';
    const lg = log.querySelector('#lg');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把词先按<b>字符</b>拆成一串 symbol —— 这是 BPE 的起点。',
      '<span class="v">add_new_bigram()</span> 给每对相邻 symbol 查一次 <span class="k">find_bpe_rank()</span>，查得到才入队。',
      '<span class="v">rank 小的先合并</span>：队列按 rank 排序（rank 相同再比左端位置）。示例里的 rank 是示意值。',
      '<span class="v">(e,s) 合并成 es</span>。被合并的右半标记为 n = 0（出局）。',
      '合并只影响两侧：<span class="k">只补两个新 bigram</span>，不重扫全部 —— 这就是 BPE 快的来源。',
      '收束：<span class="k">分词的顺序完全由词表里的 rank 决定</span>，不由代码的循环顺序决定。'
    ];
    qels.forEach(e => e.style.opacity = '.30');
    const fill = ['', '', ''];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [16]); });
    tl.at(7100, () => {
      msg.innerHTML = texts[2]; U.markLines(document, [0, 2]);
      qels.forEach((e, i) => { e.style.opacity = i === 0 ? '1' : '.30'; });
    });
    tl.at(10300, () => {
      msg.innerHTML = texts[3]; U.markLines(document, [37]);
      cells[4].style.background = 'rgba(88,166,255,.25)'; cells[4].style.borderColor = 'var(--a)';
      cells[5].style.opacity = '.30'; cells[5].style.textDecoration = 'line-through';
      lg.textContent = '(l,es)  (es,t)';
    });
    tl.at(13500, () => {
      msg.innerHTML = texts[4];
      qels.forEach(e => { e.style.opacity = '.30'; });
      lg.textContent = '(l,es)  (es,t)  ->  再合并 (es,t)，只补 (l,est) 与 (est,?) 两个';
    });
    tl.at(17500, () => { msg.innerHTML = texts[5]; U.markLines(document, []); });
  }
},

/* ------------------------------------------------------ 4 unicode：<span class="hl-f">类别判断</span>与 <span class="hl-f">NFD 折叠</span>靠数据表 */
{
  kicker: "L2-08 · unicode",
  title: "unicode：<span class=\"hl-f\">类别判断</span>与 <span class=\"hl-f\">NFD 折叠</span>靠数据表",
  sub: "分词器要回答\"这个字符是字母还是数字、要不要折叠重音\"，答案是三张查表函数 + 五张生成表。",
  caption: "unicode-data.cpp 第 1 行写明：generated with scripts/gen-unicode-data.py —— 它是生成物，不要手改。",
  src: "src/unicode.cpp",
  mark: [0, 6, 25, 37],
  lineNo: 0,
  code: `size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
//>> ---- src/unicode.cpp:30-47 ----
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
//>> ---- src/unicode.cpp:1117-1127 ----
std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
//>> ---- src/unicode.cpp:1147-1151 ----
unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">UTF-8 字节</span><span class="arrow">-></span>
        <span class="chip a">unicode_cpt_from_utf8</span><span class="arrow">-></span>
        <span class="chip f">unicode_cpt_flags_from_cpt</span><span class="arrow">-></span>
        <span class="chip c">unicode_cpts_normalize_nfd</span>
      </div>
      <div id="bars" class="bars"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#bars');
    const tables = [
      { l: 'ranges_flags', n: 2273, c: 'a' },
      { l: 'map_lowercase', n: 1433, c: 'b' },
      { l: 'map_uppercase', n: 1450, c: 'd' },
      { l: 'ranges_nfd', n: 1828, c: 'c' },
      { l: 'set_whitespace', n: 25, c: 'e' }
    ];
    const bars = tables.map(t => { const b = U.bar(t.l, t.c); host.appendChild(b.el); return b; });
    bars.forEach((b, i) => {
      b.fill.style.width = (tables[i].n / 2273 * 100) + '%';
      b.val.textContent = tables[i].n;
      b.el.style.opacity = '.55';
    });
    const hi = k => bars.forEach((b, i) => { b.el.style.opacity = (k < 0 || i === k) ? '1' : '.35'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '三个函数，一条链：<span class="k">字节 -> 码点 -> 类别 -> 折叠形态</span>。',
      '<span class="v">unicode_len_utf8()</span>：只看首字节的高 4 位就能查出一个 UTF-8 字符占几字节 —— 一张 16 项的查找表。',
      '<span class="v">unicode_cpt_from_utf8()</span>：按位掩码逐段拼出码点；遇到非法字节直接 throw。BPE 主循环用它决定"下一个 symbol 切多长"。',
      '<span class="v">unicode_cpt_flags_from_cpt()</span>：码点 -> 12 个标志位（字母/数字/标点/空白/大小写/是否 NFD），查的是右边第一张表。',
      '<span class="v">unicode_cpts_normalize_nfd()</span>：二分查找 <span class="v">unicode_ranges_nfd</span>，判断这个码点要不要折叠成基字符。',
      '五张表合计 <span class="k">7009 行</span>（2273 + 1433 + 1450 + 1828 + 25）。这就是 unicode 支持的真正体积。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; hi(-1); });
    tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [0]); hi(-1); });
    tl.at(7100, () => { msg.innerHTML = texts[2]; U.markLines(document, [6]); hi(-1); });
    tl.at(10300, () => { msg.innerHTML = texts[3]; U.markLines(document, [37]); hi(0); });
    tl.at(13500, () => { msg.innerHTML = texts[4]; U.markLines(document, [25]); hi(3); });
    tl.at(17500, () => { msg.innerHTML = texts[5]; U.markLines(document, []); hi(-1); });
  }
},

/* ------------------------------------------------------ 5 一条链就是 host 上的一个 <span class="hl-a">vector</span> */
{
  kicker: "L2-08 · 采样链",
  title: "一条链就是 host 上的一个 <span class=\"hl-a\">vector</span>",
  sub: "llama_sampler_chain 的全部状态：一个 samplers 数组、一块复用缓冲、两个计时计数。",
  caption: "注意没有 ggml_tensor、没有 buffer、没有 graph —— 链是纯 host 数据结构。",
  src: "src/llama-sampler.h",
  mark: [4, 15, 19],
  lineNo: 12,
  code: `struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;
//>> is_init：是否已经为「后端采样」建过图；这里是 host 路径，保持 false

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;
//>> samplers：链的本体。每个元素是一个 {is_backend, ptr}

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;
//>> cur：复用缓冲，避免每次采样都为整个词表重新分配

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row center" id="vec" style="gap:8px"></div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="l" style="gap:7px"></div>
        <div class="col grow" id="r" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const vec = wrap.querySelector('#vec');
    const nodes = [
      { n: 'penalties', c: 'a' }, { n: 'grammar', c: 'd' }, { n: 'temp', c: 'c' },
      { n: 'top_k', c: 'b' }, { n: 'top_p', c: 'b' }, { n: 'dist', c: 'e' }
    ];
    const nels = nodes.map((nd, i) => {
      const e = U.card({ c: nd.c, t: nd.n, b: 'is_backend = false', m: 'ptr' });
      e.style.width = '98px'; e.style.textAlign = 'center'; e.style.flex = '0 0 auto';
      e.querySelector('.ct').style.fontSize = '11px';
      vec.appendChild(e);
      if (i < nodes.length - 1) vec.appendChild(U.arrow('->'));
      return e;
    });
    nels.forEach(e => e.style.opacity = '.30');

    const l = wrap.querySelector('#l');
    l.innerHTML = '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">每个节点只是一个指针</div>' +
      '<div class="cb"><span class="cm" style="margin:0">struct info { bool is_backend; llama_sampler * ptr; }</span><br>' +
      '链不认识具体类型，只按 <b>加入顺序</b>依次调用 <span class="cm" style="margin:0">ptr</span> 的虚表。</div></div>';
    const r = wrap.querySelector('#r');
    r.innerHTML = '<div class="card" style="border-left-color:var(--f)">' +
      '<div class="ct" style="color:var(--f)">链自带一块缓冲</div>' +
      '<div class="cb"><span class="cm" style="margin:0">std::vector<llama_token_data> cur;</span><br>' +
      '一次采样要装下整个词表的候选。把它挂在链上，就不必每步重新分配。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '这条链有 6 个节点，但结构体里<b>看不到任何节点类型</b>。',
      '<span class="v">samplers</span> 是一个 vector：链 = 顺序 + 一个指针数组。',
      '<span class="v">is_backend</span> 是给「后端采样」用的开关，默认 false 表示这个节点在 host 上跑。',
      '<span class="v">cur</span> 是复用缓冲：<span class="k">一次采样 = 把整个词表铺成数组</span>。',
      '<span class="v">t_sample_us / n_sample</span>：两个计时字段，说明采样是被单独计量的一个阶段（perf 输出里的 samplers time）。'
    ];
    let t = 700;
    nels.forEach((e, i) => { tl.at(t, () => {
      nels.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i, 2)];
    }); t += 2000; });
    tl.at(13500, () => { msg.innerHTML = texts[3]; U.markLines(document, [19]); });
    tl.at(15800, () => { msg.innerHTML = texts[4]; U.markLines(document, [25, 27]); });
  }
},

/* ------------------------------------------------------ 6 ★ 链的执行顺序 = <span class="hl-b">数组顺序</span>，一步不差 */
{
  kicker: "L2-08 · 采样链",
  title: "★ 链的执行顺序 = <span class=\"hl-b\">数组顺序</span>，一步不差",
  sub: "llama_sampler_apply() 只做一次虚表转发；chain_apply() 按 samplers 的顺序对同一个 cur_p 反复调用。",
  caption: "每个节点都在<b>同一个</b> llama_token_data_array 上原地修改 —— 所以顺序决定结果。",
  src: "src/llama-sampler.cpp",
  mark: [0, 9, 16, 27],
  lineNo: 0,
  code: `void llama_sampler_apply(struct llama_sampler * smpl, struct llama_token_data_array * cur_p) {
    if (!smpl) {
        return;
    }

    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}
//>> ---- src/llama-sampler.cpp:681-701 ----
static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    bool is_backend = chain->is_init;

    for (auto & smpl : chain->samplers) {
        if (is_backend && smpl.is_backend) {
            continue;
        }

        is_backend = false;

        if (smpl.ptr->iface->apply == nullptr) {
            continue;
        }

        llama_sampler_apply(smpl.ptr, cur_p);
    }
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">penalties</span><span class="arrow">-></span>
        <span class="chip d">grammar</span><span class="arrow">-></span>
        <span class="chip c">temp</span><span class="arrow">-></span>
        <span class="chip b">top_k</span><span class="arrow">-></span>
        <span class="chip b">top_p</span><span class="arrow">-></span>
        <span class="chip e">dist</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="card grow" style="border-left-color:var(--a)">
          <div class="ct" style="color:var(--a)">同一份 cur_p</div>
          <div class="cb">所有节点共享同一个数组指针。<br>
          <span class="cm" style="margin:0">llama_token_data_array * cur_p</span><br>
          前一个节点改了 <span class="cm" style="margin:0">logit / p / size / sorted</span>，后一个节点直接看到。</div>
        </div>
        <div class="card grow" style="border-left-color:var(--b)">
          <div class="ct" style="color:var(--b)">没有并行、没有重排</div>
          <div class="cb">chain_apply() 就是一个 for 循环，按加入顺序<b>串行</b>调用。<br>
          所以 <span class="cm" style="margin:0">llama_sampler_chain_add()</span> 的调用顺序就是语义的一部分。</div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">llama_sampler_apply()</span>：唯一的动作是转发到 <span class="k">iface->apply</span>（虚表），并断言它非空。',
      '<span class="v">llama_sampler_chain_apply()</span>：遍历 <span class="k">chain->samplers</span>，逐个转发。',
      '<span class="v">if (smpl.ptr->iface->apply == nullptr) continue;</span>：节点可以不实现 apply（例如纯 accept 型的节点）。',
      '<span class="v">is_backend</span> 前缀跳过：已经交给后端的节点不再在 host 上重复执行（第 8 幕讲）。',
      '结论：<span class="k">链的顺序不是配置细节，而是算法本身</span>。下一幕看顺序如何改变结果。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
    tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [9]); });
    tl.at(7100, () => { msg.innerHTML = texts[2]; U.markLines(document, [22]); });
    tl.at(10300, () => { msg.innerHTML = texts[3]; U.markLines(document, [14, 18]); });
    tl.at(14500, () => { msg.innerHTML = texts[4]; U.markLines(document, [27]); });
  }
},

/* ------------------------------------------------------ 7 ★ 顺序为何会改变结果：<span class="hl-c">temp 保序</span>，<span class="hl-b">top_p 不保序</span> */
{
  kicker: "L2-08 · 各节点的职责",
  title: "★ 顺序为何会改变结果：<span class=\"hl-c\">temp 保序</span>，<span class=\"hl-b\">top_p 不保序</span>",
  sub: "temp 只是把 logits 除以 temp；top_k 截断数组；top_p 先 softmax 再按累积概率截断；dist 才是抽签。",
  caption: "源码里 llama_sampler_sample 断言 selected < size（第 9 幕逐字引用）—— 所以 dist 必须在 top_k / top_p 之后。",
  src: "src/llama-sampler.cpp",
  mark: [1, 11, 13, 30],
  lineNo: 0,
  code: `    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
//>> ---- src/llama-sampler.cpp:330-337 ----
    k = std::min(k, (int) cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        llama_token_data_array_partial_sort_inplace(cur_p, k);
    }

    cur_p->size = k;
//>> ---- src/llama-sampler.cpp:1556-1571 ----
    llama_sampler_softmax_impl(cur_p, false);

    size_t k = cur_p->size;
    auto * pdata = cur_p->data;

    auto & buf_sort = ctx->buf_sort;

    // if not sorted, try adaptive top-k sorting
    if (!cur_p->sorted && cur_p->size > 1024) {
        k = std::min<size_t>(256, cur_p->size);
        llama_token_data_array_partial_sort(*cur_p, k, buf_sort);
        pdata = buf_sort.data();
    } else if (!cur_p->sorted) {
        // small candidates -> sort inplace
        llama_token_data_array_partial_sort_inplace(cur_p, k);
    }
//>> ---- src/llama-sampler.cpp:1190-1204 ----
    const double rnd = dist(ctx->rng);

          double sum_run = 0.0f;
    const double sum_tgt = sum_cum*rnd;

    bool found = false;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (!found) {
            // accumulate probs until we reach the target sum
            sum_run += cur_p->data[i].p;
            if (sum_run >= sum_tgt) {
                cur_p->selected = i;
                found = true;
            }
        }`,
  duration: 25000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div id="tbl"></div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="ca" style="gap:5px"></div>
        <div class="col grow" id="cb" style="gap:5px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['节点', '它改什么', '保序？'],
      [['temp', 'logit /= temp（temp <= 0 时除最大值外全置 -INFINITY）', '保序（除以正数）'],
       ['top_k', '排序后 size = k，丢掉尾巴', '保序（只删）'],
       ['top_p', '先 softmax 算 p，再按累积和截断', '改 p，依赖尺度'],
       ['dist', '按 p 抽一次，写 selected', '只写 selected']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const ca = wrap.querySelector('#ca');
    ca.innerHTML = '<div class="cm" style="margin:0">链 A：temp -> top_k -> top_p</div>';
    const cb = wrap.querySelector('#cb');
    cb.innerHTML = '<div class="cm" style="margin:0">链 B：top_k -> temp -> top_p</div>';
    const mk = (host, txt, c) => {
      const e = U.el('div', { class: 'formula', style: 'padding:3px 7px;font-size:9.5px' });
      e.innerHTML = '<span style="color:var(--' + c + ')">' + U.esc(txt) + '</span>';
      host.appendChild(e); return e;
    };
    const a1 = mk(ca, 'logits / 0.7  ->  全部变大', 'c');
    const a2 = mk(ca, 'top_k 取前 20  ->  集合 S', 'b');
    const a3 = mk(ca, 'softmax(缩放后的 logits)  ->  截断', 'b');
    const b1 = mk(cb, 'top_k 取前 20  ->  集合 S', 'b');
    const b2 = mk(cb, 'logits / 0.7  ->  只改尺度', 'c');
    const b3 = mk(cb, 'softmax 尺度不同  ->  截断点不同', 'b');
    [a1, a2, a3, b1, b2, b3].forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先记住：<span class="k">四个节点都在同一个数组上原地改</span>，所以它改什么决定了后面能看到什么。',
      '<span class="v">temp</span>：正的 temp 只是给所有 logit 同除一个数，<b>排序不变</b>。所以 top_k 放在 temp 前后，选出的集合一样。',
      '<span class="v">top_k</span> 直接改 <span class="v">size</span>：尾巴被丢掉，后面的节点再也看不到它们。',
      '<span class="v">top_p</span>：先 softmax。而 softmax 的概率<b>依赖尺度</b>，所以"temp 先还是 top_p 先"会得到不同的截断集合 —— 链 A 与链 B 在这里分叉。',
      '<span class="v">dist</span>：读 <span class="v">p</span> 抽一次，只写 <span class="v">selected</span>。若它后面还有节点把 <span class="v">size</span> 改小，sample 的断言就可能被触发。',
      '一句话：<span class="k">保序的节点可以换位，改尺度的节点不能</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows[0].className = 'on'; });
    tl.at(3900, () => { msg.innerHTML = texts[1]; rows[1].className = 'on'; U.markLines(document, [1]); a1.style.opacity = '1'; b2.style.opacity = '1'; });
    tl.at(7100, () => { msg.innerHTML = texts[2]; rows[2].className = 'on'; U.markLines(document, [11]); a2.style.opacity = '1'; b1.style.opacity = '1'; });
    tl.at(10300, () => { msg.innerHTML = texts[3]; rows[3].className = 'on'; U.markLines(document, [13]); a3.style.opacity = '1'; b3.style.opacity = '1'; });
    tl.at(15000, () => { msg.innerHTML = texts[4]; rows[4].className = ''; U.markLines(document, [30]); });
    tl.at(20000, () => { msg.innerHTML = texts[5]; U.markLines(document, []); rows.forEach(r => { r.className = ''; }); });
  }
},

/* ------------------------------------------------------ 8 ★ 为什么采样不进图：<span class="hl-e">它操作的是 host 上的 float 数组</span> */
{
  kicker: "L2-08 · 核心",
  title: "★ 为什么采样不进图：<span class=\"hl-e\">它操作的是 host 上的 float 数组</span>",
  sub: "一次采样 = 把 logits 复制进 std::vector<llama_token_data>，然后在 host 上跑链。全程没有 ggml 算子。",
  caption: "本版新增的 [EXPERIMENTAL] 后端采样反过来证明了这一点：要把采样放进图，必须为它单独建一张图并逐个算子探测后端支持。",
  src: "src/llama-sampler.cpp",
  mark: [3, 16, 25, 44],
  lineNo: 0,
  code: `    } else {
        const auto * logits = llama_get_logits_ith(ctx, idx);
        GGML_ASSERT(logits != nullptr);
        cur.resize(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };
//>> ---- src/llama-sampler.cpp:583-587 ----
static llama_sampler_backend_probe llama_sampler_backend_probe_graph(
        llama_sampler * sampler,
        int64_t         n_candidates,
        uint32_t        max_nodes,
        bool            with_candidates) {
//>> ---- src/llama-sampler.cpp:649-662 ----
    for (int i = 0; i < ggml_graph_n_nodes(probe.gf); i++) {
        struct ggml_tensor * op = ggml_graph_node(probe.gf, i);

        if (!ggml_backend_dev_supports_op(device, op)) {
            LLAMA_LOG_WARN("%s: device '%s' does not have support for op %s needed for sampler '%s'\\n",
                    __func__, ggml_backend_dev_name(device), ggml_op_name(op->op), smpl->iface->name(smpl));

            return false;
        }
    }

    return true;
}

//>> ---- src/llama-sampler.cpp:746-762 ----
    for (auto & smpl : chain->samplers) {
        bool cur_prefix = backend_prefix;

        // to be able to run a sampler on the backend, it has to:
        // - have the .backend_init() API implemented
        // - return true during .backend_init()
        // - support the requested per-sequence output limit
        if (cur_prefix && smpl.ptr->iface->backend_init) {
            if (!smpl.ptr->iface->backend_init(smpl.ptr, buft, n_outputs_max_per_seq)) {
                cur_prefix = false;
            }
        } else {
            cur_prefix = false;
        }

        smpl.is_backend = cur_prefix;
        backend_prefix = cur_prefix;`,
  duration: 25000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:9px">
        <div class="col grow" id="in" style="gap:6px"></div>
        <div class="col grow" id="out" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const lhs = wrap.querySelector('#in');
    lhs.innerHTML = '<div class="cm" style="margin:0">图内（L2-07 / L5-01）</div>' +
      '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">llama_decode -> ggml 图</div>' +
      '<div class="cb">注意力、矩阵乘、norm ... 都是 ggml 算子，可以切给 CPU / CUDA / Metal。<br>' +
      '产出：<span class="cm" style="margin:0">float logits[n_vocab]</span>（host 可取）。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">图上没有采样算子</div>' +
      '<div class="cb">没有 GGML_OP_SAMPLE 这种东西。图不知道"概率"，也不知道"词表"。</div></div>';
    const rhs = wrap.querySelector('#out');
    rhs.innerHTML = '<div class="cm" style="margin:0">图外（本课）</div>' +
      '<div class="card" style="border-left-color:var(--e)"><div class="ct" style="color:var(--e)">host 上的数组</div>' +
      '<div class="cb"><span class="cm" style="margin:0">std::vector<llama_token_data> cur;</span><br>' +
      '每个元素 = {id, logit, p}。采样器只认这个数组。</div></div>' +
      '<div class="card" style="border-left-color:var(--d)"><div class="ct" style="color:var(--d)">为什么放在 host 更合理</div>' +
      '<div class="cb">o(n_vocab) 的标量操作，相对一次前向可以忽略；<br>' +
      '而且 grammar / penalties / dry 依赖<b>历史 token</b>与<b>语法状态</b>，<br>' +
      '这些状态本来就在 host 上。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">llama_sampler_sample()</span> 从上下文取出 logits，铺成 <span class="k">std::vector<llama_token_data></span>。',
      '注意这里没有任何 <span class="cm" style="margin:0">ggml_*</span> 调用：没有新张量、没有 buffer、没有 compute。',
      '然后 <span class="v">llama_sampler_apply(smpl, &cur_p)</span> 在 host 上跑完整条链，取出 selected 对应的 id。',
      '[EXPERIMENTAL] 后端采样：要给节点加 <span class="v">backend_apply</span>，它收到的是 ggml context 与 cgraph —— 也就是说<b>它得自己建一张采样图</b>。',
      '还要逐个算子问后端：<span class="v">ggml_backend_dev_supports_op()</span> 返回 false 就退回 host。默认路径根本不走这套。',
      '链上只允许<b>前缀</b>上后端：第一个不支持 backend_init 的节点之后，全部留在 host（grammar、dry、xtc、mirostat、top_n_sigma、infill、adaptive_p、typical 都没有实现）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [3]); });
    tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, [5]); });
    tl.at(7100, () => { msg.innerHTML = texts[2]; U.markLines(document, [9]); });
    tl.at(11000, () => { msg.innerHTML = texts[3]; U.markLines(document, [16, 22]); });
    tl.at(15000, () => { msg.innerHTML = texts[4]; U.markLines(document, [25, 31]); });
    tl.at(19500, () => { msg.innerHTML = texts[5]; U.markLines(document, [44, 52]); });
  }
},

/* ------------------------------------------------------ 9 一次采样的全部：<span class="hl-a">apply</span> 一次，<span class="hl-a">accept</span> 一次 */
{
  kicker: "L2-08 · 收束",
  title: "一次采样的全部：<span class=\"hl-a\">apply</span> 一次，<span class=\"hl-a\">accept</span> 一次",
  sub: "图给出 logits，采样器给出 token，并把 token 记进自己的状态（penalties 的计数、grammar 的栈）。",
  caption: "下一课 L2-09：grammar 的实现（llama-grammar.cpp）与 chat 模板（llama-chat.cpp）。",
  src: "src/llama-sampler.cpp",
  mark: [0, 3, 7, 9],
  lineNo: 947,
  code: `    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    llama_sampler_apply(smpl, &cur_p);

    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);

    auto token = cur_p.data[cur_p.selected].id;

    llama_sampler_accept(smpl, token);`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip c">logits[]</span><span class="arrow">-></span>
        <span class="chip a">cur_p（host 数组）</span><span class="arrow">-></span>
        <span class="chip b">chain.apply()</span><span class="arrow">-></span>
        <span class="chip e">selected</span><span class="arrow">-></span>
        <span class="chip d">accept(token)</span>
      </div>
      <div id="tbl"></div>
      <div id="ex"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['环节', '跑在哪', '本课逐字引用的位置', '展开在哪课'],
      [['分词 tokenize', 'host (CPU)', 'llama-vocab.cpp:3413', '本课'],
       ['BPE 合并（rank）', 'host (CPU)', 'llama-vocab.cpp:264 / 657', '本课 / L1-06（merges 从 GGUF 读）'],
       ['unicode 类别与 NFD', 'host (CPU)', 'unicode.cpp:1147 / 1117', '本课'],
       ['decode 建图算 logits', '后端（CPU/GPU）', '图里（本课不引用）', '回顾 L2-07 / L5-01'],
       ['采样链 apply', 'host (CPU)', 'llama-sampler.cpp:681', '本课'],
       ['grammar / penalties / dry', 'host (CPU)', 'llama-sampler.cpp:2751（无 backend 实现）', 'L2-09 llama-grammar.cpp'],
       ['后端采样（实验）', '后端（另建一张图）', 'llama-sampler.cpp:733', 'L5-01 后端能力探测']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '给一条 sampler chain 配上 grammar 约束。为什么 <b>不需要</b> CUDA / Metal 后端为 grammar 写 kernel？',
      '因为 grammar 采样器只实现了 host 侧的 <span class="mono">.apply()</span>：' +
      '<span class="mono">llama_sampler_grammar_i</span> 里 <span class="mono">backend_init</span> / ' +
      '<span class="mono">backend_apply</span> 都是 <span class="mono">nullptr</span>（llama-sampler.cpp:2751）。<br>' +
      '它拿到的是 host 内存里的 <span class="mono">llama_token_data_array</span>，把不合语法的 token 的 ' +
      '<span class="mono">logit</span> 置为 <span class="mono">-INFINITY</span>' +
      '（实现在 <span class="mono">llama_grammar_apply_impl()</span>，llama-grammar.cpp:1355 起，L2-09 逐字展开）。<br>' +
      '图只负责把 logits 算出来；约束是<b>图外</b>对分布的一次裁剪，所以后端对此一无所知。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">cur_p</span> 是一个纯 host 结构：指针 + size + selected + sorted。',
      '链跑完后，答案就在 <span class="v">cur_p.data[cur_p.selected].id</span>。',
      '<span class="v">accept()</span> 把选中的 token 告诉链上每个节点 —— penalties 记计数，grammar 推进语法栈。',
      '什么都不看的行：tokenize 与采样都在 host，图只负责 logits。这就是"采样不进图"的全部证据。'
    ];
    rows.forEach(r => { r.style.opacity = '.35'; });
    tl.at(700, () => { msg.innerHTML = texts[0]; rows[0].style.opacity = '1'; });
    tl.at(3600, () => { msg.innerHTML = texts[0]; rows[1].style.opacity = '1'; rows[2].style.opacity = '1'; });
    tl.at(6600, () => { msg.innerHTML = texts[0]; rows[3].style.opacity = '1'; });
    tl.at(9600, () => { msg.innerHTML = texts[1]; rows[4].style.opacity = '1'; });
    tl.at(12600, () => { msg.innerHTML = texts[2]; rows[5].style.opacity = '1'; rows[6].style.opacity = '1'; });
    tl.at(16200, () => { msg.innerHTML = texts[3]; U.markLines(document, [7, 12]); });
  }
},

];
