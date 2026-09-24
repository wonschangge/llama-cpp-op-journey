/* ==========================================================================
   L2-04 · KV cache 与记忆家族
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 为什么不是"一个 KV cache" */
{
  kicker: "L2-04 · 全局",
  title: "为什么不是\"一个 KV cache\"",
  sub: "KV cache 只是 LLM memory 的一种。源码把这一点写在接口的第一行注释里。",
  caption: "覆盖计划里 L2-04 的全部 23 个文件，另加分派点所在的 src/llama-model.cpp（create_memory 在 2274 行）。",
  src: "src/llama-memory.h",
  mark: [2, 5, 9, 11],
  lineNo: 71,
  code: `// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
//>> 三类回调：层过滤 / 层复用 / 层共享 —— 差异从这里开始被表达出来
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">模型架构</span><span class="arrow">-></span>
        <span class="chip a">create_memory()</span><span class="arrow">-></span>
        <span class="chip b">llama_memory_i</span><span class="arrow">-></span>
        <span class="chip c">K/V 张量 · 状态张量</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'g', t: '问题', b: '不是所有模型都有 KV：<br>SSM/RNN 每层只有一份固定状态；<br>SWA / 稀疏注意力只要窗口', m: 'mem_cell 里没有 K/V' },
      { c: 'a', t: '抽象', b: '一批虚函数，描述"记忆"对外的<br>全部行为：生命周期 / 序列操作 / 状态 IO', m: 'struct llama_memory_i' },
      { c: 'b', t: '实现家族', b: 'KV cache 一族 6 个类 + recurrent 1 个<br>+ hybrid 一族 3 个类', m: '10 个类，9 个直接继承' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:220px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '接口注释写得很直白：<span class="v">the KV cache is a type of LLM memory, but there can be other types</span>。',
      '<span class="k">问题</span>：Mamba/RWKV 这类模型没有 per-token 的 K/V；<br>SWA 层只需要最近 n_swa 个 token。硬塞进一个类会到处是 if。',
      '<span class="k">抽象</span>：只把"记忆对外的行为"写成虚函数；<br>数据面（K/V 张量、状态张量）留在各自实现里。',
      '<span class="k">实现</span>：本课逐个对照它们各自解决什么问题 —— 见第 8 幕的对照表。',
      '一句话：<span class="k">抽象的是"行为"，不是"数据布局"</span>。'
    ];
    defs.forEach((_, i) => tl.at(800 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(12000, () => { msg.innerHTML = texts[3]; });
    tl.at(16000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">llama_memory_i</span> 的虚接口：三类职责 */
{
  kicker: "L2-04 · 核心",
  title: "★ <span class=\"hl-a\">llama_memory_i</span> 的虚接口：三类职责",
  sub: "生命周期、序列操作、状态读写。整族 memory 的共同语言就是这 15 个纯虚函数。",
  caption: "注意：接口里没有 push_kv() / get_k()。图和 cell 之间的数据面不在这个接口上。",
  src: "src/llama-memory.h",
  mark: [3, 10, 15, 26, 29, 35, 44],
  lineNo: 85,
  code: `    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
//>> init_batch：把一批 token 切成 ubatch，并检查能否装下；返回一个 context 对象
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;
//>> init_full：假装 cache 是满的，用于分配最坏情况的 compute buffer

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;
//>> init_update：把挂起的 shift / copy 等更新做成一个 context

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;
//>> clear(data)：data=true 时连数据一起清

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
//>> state_write/read：序列状态的存盘与恢复 —— 会话保存走的就是它
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '生命周期', m: 'init_batch · init_full · init_update', b: '每次 decode 前先问它："这批 token 放得下吗、放在哪些 cell？"' },
      { c: 'b', t: '序列操作', m: 'clear · seq_rm/cp/keep/add/div · seq_pos_min/max', b: '按 seq_id 与位置区间增删改查 —— 多序列推理的基础' },
      { c: 'c', t: '度量与状态', m: 'get_can_shift · memory_breakdown · state_write/read', b: '能不能整体平移、占了多少显存、状态怎么存盘恢复' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:220px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '15 个纯虚函数，按职责分三组看：',
      '<span class="v">生命周期</span>：三个 <span class="k">init_*</span> 都返回 <span class="k">llama_memory_context_ptr</span>，<br>而不是直接改状态 —— 这是第 3 幕的主题。',
      '<span class="v">序列操作</span>：全部以 <span class="k">(seq_id, p0, p1)</span> 为参数，<br>说明"记忆"是按<span class="k">序列 + 位置区间</span>管理的，不是按数组下标。',
      '<span class="v">度量与状态</span>：<span class="k">memory_breakdown()</span> 让上层能报告显存占用；<br><span class="k">state_write/read</span> 让整个记忆可序列化。',
      '★ 记住这一点：<span class="k">接口管"哪些 cell 属于哪个序列"，不管"K/V 长什么样"</span>。<br>所以 recurrent memory 可以用同一套接口，却完全没有 K/V。'
    ];
    defs.forEach((_, i) => tl.at(800 + i * 3800, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(19000, () => { msg.innerHTML = texts[3]; });
    tl.at(24000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 3 ★ <span class="hl-a">context</span> 对象：apply() 是唯一改状态的入口 */
{
  kicker: "L2-04 · 核心",
  title: "★ <span class=\"hl-a\">context</span> 对象：apply() 是唯一改状态的入口",
  sub: "init_* 只做规划，不落笔；真正写进 memory 的动作只有 apply()。",
  caption: "组合型 memory（iswa / dsa / hybrid）靠 status 的合并来伪装成一个 memory：src/llama-memory.h:37-39。",
  src: "src/llama-memory.h",
  mark: [6, 8, 13, 17, 20, 23],
  lineNo: 44,
  code: `// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
//>> 这一行是整个设计的核心断言
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">init_batch()</span><span class="arrow">-></span>
        <span class="chip b">context</span><span class="arrow">-></span>
        <span class="chip c">next() 逐个 ubatch</span><span class="arrow">-></span>
        <span class="chip d">apply() 落笔</span>
      </div>
      <div class="row" id="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    const right = wrap.querySelector('#right');
    left.innerHTML = '<div class="cm" style="margin:0">context 的四个方法</div>';
    const mets = [
      ['next()', '推进到下一个 ubatch；返回 false 表示没有了'],
      ['apply()', '把当前 ubatch 的记忆状态写进 memory'],
      ['get_ubatch()', '取出当前 ubatch，交给建图'],
      ['get_status()', '成功 / 无需更新 / 准备失败 / 计算失败']
    ];
    const mEls = mets.map(m => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
      e.innerHTML = '<span class="m">' + U.esc(m[0]) + '</span> &nbsp;' + U.esc(m[1]);
      left.appendChild(e);
      return e;
    });
    mEls.forEach(e => e.style.opacity = '.30');

    right.innerHTML =
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">为什么先规划后落笔？</div>' +
      '<div class="cb">因为"能不能装下"必须在建图之前回答。context 里存着 <span class="cm" style="margin:0">slot_info_vec_t</span> ' +
      '与 ubatch 列表，规划失败就直接返回错误状态，memory 一个字节都没动。</div></div>' +
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">组合靠什么？</div>' +
      '<div class="cb">子 memory 各自返回 status，用 <span class="cm" style="margin:0">llama_memory_status_combine()</span> 合并：' +
      '任一失败即失败，任一有更新即有更新（src/llama-memory.cpp:3-41）。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '三步走：<span class="k">init_batch 规划</span> -> <span class="k">next() 遍历</span> -> <span class="k">apply() 落笔</span>。',
      '<span class="v">next()</span> 只是游标前进，不碰 memory；<br>所以一个 batch 的多个 ubatch 可以先全部规划好。',
      '<span class="v">apply()</span> 才把 cell 的 pos / seq / ext 写下去 ——<br>源码注释原话：<span class="k">the only method that should mutate the memory</span>。',
      '<span class="v">get_status()</span> 是错误处理的统一出口；<br>组合型 memory 把子 context 的 status 合并后再上报。',
      '一句话：<span class="k">规划与提交分离</span>，所以"放不下"可以优雅失败，而不会留下半个批次的状态。'
    ];
    tl.at(800, () => { msg.innerHTML = texts[0]; });
    mets.forEach((_, i) => tl.at(4000 + i * 2600, () => {
      mEls.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i + 1, 3)];
    }));
    tl.at(16000, () => { mEls.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-c">cell</span>：KV cache 的最小单位是"元数据"，不是 K/V */
{
  kicker: "L2-04 · 核心",
  title: "★ <span class=\"hl-c\">cell</span>：KV cache 的最小单位是\"元数据\"，不是 K/V",
  sub: "一个 cell 记的是\"这个位置上是谁的 token、属于哪些序列\"，K/V 本身在另一个张量里。",
  caption: "llama_kv_cells 的成员全是元数据：pos / ext / shift / seq / seq_pos / used。",
  src: "src/llama-kv-cells.h",
  mark: [1, 5, 7, 10, 27, 30, 40],
  lineNo: 485,
  code: `private:
    bool has_shift = false;
//>> has_shift：本轮的 pos 变更还没同步到 K 张量（见下一幕的 set_input_k_shift）

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // the set seq_pos[s] holds one (pos, cell) pair per cell that carries sequence s, ordered by position
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    // and upper_bound() on a position finds the nearest cell of the sequence in logarithmic time
    //
    // the cell index is part of the key because a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    std::set<std::pair<llama_pos, uint32_t>> seq_pos[LLAMA_MAX_SEQ];`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:8px">
        <div class="col grow" id="diag" style="gap:6px"></div>
        <div class="col" id="side" style="gap:7px;width:246px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const diag = wrap.querySelector('#diag');
    diag.innerHTML = '<div class="cm" style="margin:0">8 个 cell（每个 = 一个位置）</div>' +
      '<div class="row" id="cells" style="gap:5px"></div>' +
      '<div class="cm" style="margin:2px 0 0" id="cap">pos: -1 表示空 cell；seq 是 LLAMA_MAX_SEQ 位的位集合</div>';

    const host = diag.querySelector('#cells');
    const cs = [];
    for (let i = 0; i < 8; i++) {
      const c = U.el('div', { style: 'width:66px;height:52px;border:1px solid var(--border);border-radius:5px;background:#10151b;padding:3px 4px;font-family:var(--mono);font-size:9px;color:var(--dim)' });
      c.innerHTML = '<div style="color:var(--dim)">#' + i + '</div><div class="pv">pos -</div><div class="sv">seq -</div>';
      host.appendChild(c); cs.push(c);
    }

    const side = wrap.querySelector('#side');
    side.innerHTML =
      '<div class="card" style="border-left-color:var(--a)"><div class="ct" style="color:var(--a)">pos[]</div>' +
      '<div class="cb">每个 cell 一个位置；<span class="cm" style="margin:0">-1 = 空</span>。</div></div>' +
      '<div class="card" style="border-left-color:var(--b)"><div class="ct" style="color:var(--b)">seq[]</div>' +
      '<div class="cb">位集合：一个 cell 可以同时属于多个序列（前缀共享）。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)"><div class="ct" style="color:var(--c)">ext[]</div>' +
      '<div class="cb">M-RoPE 的 x/y、多模态 token 号 —— 需要随状态存盘的额外信息。</div></div>' +
      '<div class="card" style="border-left-color:var(--d)"><div class="ct" style="color:var(--d)">used / seq_pos</div>' +
      '<div class="cb">used 是"非空 cell 的下标集合"；seq_pos 是 (pos, cell) 有序集合，' +
      '<span class="cm" style="margin:0">O(log n)</span> 找某序列最近的 cell。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个 cell 只有元数据。K/V 在别处：<span class="k">llama_kv_cache::layers[].k/v</span>（llama-kv-cache.h:250-260）。',
      '<span class="v">pos[i] == -1</span> 就是"空 cell"。所有分配逻辑都在找连续的空 cell 区间。',
      '<span class="v">seq[i]</span> 是位集合 —— 所以<b>多个序列可以共享同一个 cell</b>，这是公共前缀只存一份的原因。',
      '<span class="v">ext[i]</span> 存 M-RoPE 的二维位置或 token id；<span class="k">has_cell_ext()</span> 决定存盘要不要带上它。',
      '<span class="v">shift[i]</span> 把"位置平移"攒起来，最后一次性同步到 K 张量，而不是每改一次就重算。',
      '一句话：<span class="k">cell 是"账本"，K/V 张量是"仓库"</span>。第 6 幕会看到两者怎么被图接口对上。'
    ];
    tl.at(600, () => {
      msg.innerHTML = texts[0];
      [2, 3, 4].forEach(i => {
        cs[i].style.borderColor = 'var(--a)';
        cs[i].querySelector('.pv').textContent = 'pos ' + i;
        cs[i].querySelector('.sv').textContent = 'seq 0';
        cs[i].style.color = 'var(--text)';
      });
      cs[5].style.borderColor = 'var(--b)';
      cs[5].querySelector('.pv').textContent = 'pos 5';
      cs[5].querySelector('.sv').textContent = 'seq 0,1';
      cs[5].style.color = 'var(--text)';
    });
    tl.at(4200, () => { msg.innerHTML = texts[1]; });
    tl.at(8000, () => { msg.innerHTML = texts[2]; cs[5].style.background = 'rgba(63,185,80,.14)'; });
    tl.at(11800, () => { msg.innerHTML = texts[3]; });
    tl.at(15600, () => {
      msg.innerHTML = texts[4];
      cs[2].style.background = 'rgba(210,153,34,.16)'; cs[3].style.background = 'rgba(210,153,34,.16)'; cs[4].style.background = 'rgba(210,153,34,.16)';
    });
    tl.at(20000, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 <span class="hl-b">slot_info</span>：token 到 cell 的翻译表 */
{
  kicker: "L2-04 · 机制",
  title: "<span class=\"hl-b\">slot_info</span>：token 到 cell 的翻译表",
  sub: "建图之前必须先知道\"第 i 个 token 写进哪个 cell\"，这张表就是 slot_info。",
  caption: "落笔的地方是 apply_ubatch：先 cells.rm() 再 cells.pos_set()（llama-kv-cache.cpp:1120-1131）。",
  src: "src/llama-kv-cache.h",
  mark: [1, 2, 4, 7, 8, 10, 11],
  lineNo: 32,
  code: `    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="cm" style="margin:0">ubatch：4 个 token</div>
      <div class="row" id="tok" style="gap:6px"></div>
      <div class="flow" style="justify-content:center"><span class="arrow">|</span><span class="chip a">slot_info.idxs[0]</span><span class="arrow">|</span></div>
      <div class="row" id="cel" style="gap:6px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const tokHost = wrap.querySelector('#tok');
    const celHost = wrap.querySelector('#cel');
    const toks = ['t0', 't1', 't2', 't3'];
    const cels = ['#5', '#6', '#7', '#8'];
    const tEls = toks.map((t, i) => {
      const e = U.el('div', { class: 'chip', style: 'width:74px;text-align:center', text: t + '  p=' + (i + 4) });
      tokHost.appendChild(e); return e;
    });
    const cEls = cels.map((c, i) => {
      const e = U.el('div', { class: 'chip b', style: 'width:74px;text-align:center', text: 'cell ' + c });
      celHost.appendChild(e); return e;
    });
    [tEls, cEls].forEach(arr => arr.forEach(e => e.style.opacity = '.35'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '源码注释把这张表的语义写死了：<span class="v">token[i] -> goes to cells[idxs[i]]</span>。',
      '<span class="v">s0 / s1</span>：这次 ubatch 覆盖的 stream 区间（<span class="k">ns = s1 - s0 + 1</span>）。' +
      '<br>unified 模式下 ns = 1，否则一个序列一个 stream。',
      '<span class="v">strm[] / idxs[]</span>：每个 stream 一行；idxs 里就是该 stream 每个 token 的目标 cell 下标。',
      '<span class="v">is_contiguous()</span>：判断这些 cell 是不是从 head() 开始连续 —— ' +
      '<br>连续时图里可以用更省的写法。',
      '一句话：<span class="k">slot_info 是"规划结果"的载体</span>，第 3 幕的 context 里存的就是它的向量。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => { msg.innerHTML = texts[1]; });
    tl.at(7200, () => {
      msg.innerHTML = texts[2];
      tEls.forEach((e, i) => { e.style.opacity = '1'; });
    });
    tl.at(11000, () => {
      msg.innerHTML = texts[3];
      cEls.forEach((e, i) => { e.style.opacity = '1'; e.style.borderColor = 'var(--b)'; });
    });
    tl.at(15500, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 K/V 住在哪，图怎么读写它 */
{
  kicker: "L2-04 · 机制",
  title: "K/V 住在哪，图怎么读写它",
  sub: "cache 自己拿 buffer，图只拿视图（get_k）、只写行（cpy_k）。",
  caption: "跨课：L2-06 会看到 build_attn 用 get_k/get_v 造 KQ、用 cpy_k/cpy_v 写回；L4-02 会看到调度器为什么不碰这块 buffer。",
  src: "src/llama-kv-cache.cpp",
  mark: [6, 9, 33, 44],
  lineNo: 0,
  code: `    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
//>> ---- src/llama-kv-cache.cpp:1266-1283 ----
ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
//>> ---- src/llama-kv-cache.cpp:1346-1351 ----
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '归属：cache 自己分配', m: 'ggml_backend_alloc_ctx_tensors_from_buft', b: '每个后端 buffer type 一个 ggml context，' +
        '整块 K/V 张量一次性分配；<b>调度器不参与</b>。' },
      { c: 'b', t: '读：get_k / get_v', m: 'ggml_view_4d', b: '返回 K 张量的一个四维视图：' +
        '<br>[n_embd_head_k, n_head_kv, n_kv, ns]。<b>零拷贝</b>。' },
      { c: 'c', t: '写：cpy_k / cpy_v', m: 'ggml_set_rows', b: '把本次 ubatch 的 K 按 k_idxs（来自 slot_info）' +
        '<br>写进对应行。写入是图上的一个节点。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => U.card(d, { style: 'width:220px' }));
    els.forEach(e => host.appendChild(e));
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '三个事实，对应左边三段引用：',
      '<span class="k">归属</span>：K/V 张量在 llama_kv_cache 构造时按 buffer type 整块分配（cpp:276-294）。' +
      '<br>no_alloc 模式下挂 dummy buffer，注释写明是为了让 <span class="v">backend scheduler</span> 不去分配它。',
      '<span class="k">读</span>：get_k 返回 <span class="v">ggml_view_4d</span> —— 只是换一套 ne/nb，' +
      '<br>所以 attention 读整个 cache 不产生任何拷贝（cpp:1278-1283）。',
      '<span class="k">写</span>：cpy_k 最终落到 <span class="v">ggml_set_rows(ctx, k, k_cur, k_idxs)</span>（cpp:1350）。' +
      '<br>k_idxs 就是第 5 幕 slot_info 里的 idxs —— 规划与执行在这里闭环。',
      '为什么重要：<span class="k">KV cache 的生命周期与计算图的中间张量完全不同</span>。' +
      '<br>它跨 decode 存活、自己管 buffer，所以不能交给 L4-01 的 arena 分配器。'
    ];
    defs.forEach((_, i) => tl.at(800 + i * 4600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(16000, () => { msg.innerHTML = texts[3]; });
    tl.at(21000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-d">iswa</span>：把层切成两套 cache */
{
  kicker: "L2-04 · 变体",
  title: "<span class=\"hl-d\">iswa</span>：把层切成两套 cache",
  sub: "SWA 层的记忆只需要一个窗口，不需要整个上下文 —— 那就给它一个更小的 cache。",
  caption: "同一个类名 llama_kv_cache，构造参数不同（n_swa / swa_type），行为就不同。",
  src: "src/llama-kv-cache-iswa.cpp",
  mark: [6, 14, 21, 26, 33],
  lineNo: 0,
  code: `    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
//>> ---- src/llama-kv-cache-iswa.cpp:95-105 ----
    kv_base = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, mem_other_swa, filter_swa, reuse, share);`,
  duration: 23000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:10px">
        <div class="card grow" style="border-left-color:var(--a)" id="cb">
          <div class="ct" style="color:var(--a)">kv_base · 非 SWA 层</div>
          <div class="cb">filter_base: <span class="cm" style="margin:0">!is_swa(il)</span><br>
          size = kv_size（= n_ctx_seq）<br>n_swa = 0, swa_type = NONE</div>
          <div class="bt" style="height:12px;background:#10151b;border:1px solid var(--border);border-radius:3px;margin-top:6px;overflow:hidden">
            <div id="fb" style="height:100%;width:0;background:var(--a);transition:width .6s"></div></div>
        </div>
        <div class="card grow" style="border-left-color:var(--b)" id="cs">
          <div class="ct" style="color:var(--b)">kv_swa · SWA 层</div>
          <div class="cb">filter_swa: <span class="cm" style="margin:0">is_swa(il)</span><br>
          size = min(kv_size, n_swa*(unified?n_seq_max:1)+n_ubatch)<br>n_swa / swa_type 取自 hparams</div>
          <div class="bt" style="height:12px;background:#10151b;border:1px solid var(--border);border-radius:3px;margin-top:6px;overflow:hidden">
            <div id="fs" style="height:100%;width:0;background:var(--b);transition:width .6s"></div></div>
        </div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '两个 filter 把模型的层一分为二：<span class="v">is_swa(il)</span> 为真进 SWA cache，否则进 base。',
      '<span class="k">base cache</span> 拿满 <span class="v">kv_size</span>（整个上下文）；' +
      '构造时 <span class="v">n_swa = 0</span>、<span class="v">swa_type = NONE</span>，即"没有窗口"的普通 cache。',
      '<span class="k">SWA cache</span> 的大小只跟窗口有关：' +
      '<span class="v">GGML_PAD(min(size_base, n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256)</span>。' +
      '<br>注意末尾 pad 到 256，注释说是为了性能。',
      '<span class="v">swa_full</span> 开关会把 SWA cache 放大到和 base 一样大 —— 调试用，默认关。',
      '一句话：<span class="k">同一份代码，两种规模</span>。这就是"组合两个 llama_kv_cache"的全部秘密。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4200, () => {
      msg.innerHTML = texts[1];
      wrap.querySelector('#fb').style.width = '100%';
    });
    tl.at(8600, () => {
      msg.innerHTML = texts[2];
      wrap.querySelector('#fs').style.width = '18%';
    });
    tl.at(15000, () => { msg.innerHTML = texts[3]; });
    tl.at(19500, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 一张表：六种变体各自解决什么问题 */
{
  kicker: "L2-04 · 变体",
  title: "一张表：六种变体各自解决什么问题",
  sub: "每一行的依据都是源文件里的注释原话，逐字收在 source.md 里。",
  caption: "共 10 个类：9 个直接继承 llama_memory_i，llama_memory_hybrid_idx 继承 hybrid。",
  src: "src/llama-kv-cache-msa.h",
  mark: [0, 1, 3],
  lineNo: 9,
  code: `// uses two instances of llama_kv_cache, one for K/V tensors, and one for the MSA indexer tensors
// both receive identical sequence operations and identical ubatches, so their cell layouts stay in synced.
// the context also exposes per-ubatch pos - cell translation maps populated from llama_kv_cells via
// llama_kv_cache::get_cells(), which the model graph uses to run MSA block selection in position space`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['变体', '解决什么问题', '依据'],
      [['llama_kv_cache', '基类：cell 记账 + K/V 张量', 'llama-kv-cache.h:20'],
       ['llama_kv_cache_iswa', '非 SWA 层全量 / SWA 层窗口，两套 cache', 'llama-kv-cache-iswa.h:11'],
       ['llama_kv_cache_dsa', '主 K cache + 稀疏注意力的 indexer key cache', 'llama-kv-cache-dsa.h:11'],
       ['llama_kv_cache_dsa_iswa', 'DSA 层与 SWA 层再分两层', 'llama-kv-cache-dsa-iswa.h:11'],
       ['llama_kv_cache_msa', 'K/V cache + MSA indexer cache，cell 布局同步', 'llama-kv-cache-msa.h:9'],
       ['llama_kv_cache_dsv4', '原始 token cache + 压缩过的 K 块 cache', 'llama-kv-cache-dsv4.h:82'],
       ['llama_memory_recurrent', 'SSM/RNN：每层固定状态，没有 per-token K/V', 'llama-memory-recurrent.h:88'],
       ['llama_memory_hybrid', 'attention 层 + recurrent 层，各一套子 memory', 'llama-memory-hybrid.h:16'],
       ['llama_memory_hybrid_idx', 'hybrid 再加第三套 indexer cache', 'llama-memory-hybrid-idx.h:12'],
       ['llama_memory_hybrid_iswa', 'hybrid 的 attention 侧换成 iswa', 'llama-memory-hybrid-iswa.h:16']],
      { monoCols: [0, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '10 行 = 10 个类。全部实现同一个接口。',
      'KV 一族的共同点：都建立在 <span class="k">llama_kv_cache</span> 上，靠组合而非继承来加能力。',
      'iswa / dsa_iswa / hybrid_iswa 解决的都是"<span class="k">不同层要用不同大小的 cache</span>"。',
      'dsa / msa / hybrid_idx 解决的是"<span class="k">除了 K/V，还要给稀疏注意力存一份 indexer</span>"。',
      'recurrent / hybrid 解决的是"<span class="k">有些层根本没有 K/V</span>"—— 见第 9、10 幕。',
      '对照表的用法：先问"这层的记忆形状是什么"，答案自然指向某一个类。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2100, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 4)];
    }));
    tl.at(23000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 <span class="hl-e">recurrent</span>：没有 KV cache，只有固定状态 */
{
  kicker: "L2-04 · 变体",
  title: "<span class=\"hl-e\">recurrent</span>：没有 KV cache，只有固定状态",
  sub: "SSM / RNN 的记忆与序列长度无关：每层一份固定大小的状态，按序列复制几行。",
  caption: "跨课：L2-11（模型家族 · SSM 与线性注意力）讲这些层本身；本课只讲它们的记忆怎么存。",
  src: "src/llama-memory-recurrent.h",
  mark: [0, 4, 11, 12, 15, 32, 35, 36],
  lineNo: 0,
  code: `    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;
//>> ---- src/llama-memory-recurrent.h:87-115 ----
    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;`,
  duration: 25000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:9px">
        <div class="card grow" style="border-left-color:var(--a)">
          <div class="ct" style="color:var(--a)">KV cache：每个 token 一行</div>
          <div class="cb">行数 = 上下文长度（mem_size cells）<br>每行 = n_embd_k / n_embd_v 个数</div>
        </div>
        <div class="card grow" style="border-left-color:var(--e)">
          <div class="ct" style="color:var(--e)">recurrent：每个序列一份状态</div>
          <div class="cb">每层 r_l / s_l 是二维张量<br><span class="cm" style="margin:0">[n_embd_r, mem_size*(1 + n_rs_seq)]</span></div>
        </div>
      </div>
      <div id="rows"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = wrap.querySelector('#rows');
    rows.innerHTML = '<div class="cm" style="margin:0 0 3px">mem_cell 的字段（recurrent.h:88-107）</div>' +
      '<div class="row wrap" style="gap:6px">' +
      ['pos', 'src', 'src0', 'tail', 'seq_id'].map(f =>
        '<span class="chip ' + (f === 'seq_id' ? 'e' : 'c') + '">' + f + '</span>').join('') +
      '</div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '第一眼就能看出区别：recurrent 的 cell 里<span class="k">没有 K、没有 V</span>，只有"这份状态从哪来、发给谁"。',
      '<span class="v">pos</span> 是这个状态对应的位置；<span class="v">seq_id</span> 是它属于哪些序列。',
      '<span class="v">src / src0 / tail</span> 描述状态在序列之间怎么复制：' +
      '<br>src 说明从哪一行拷来（src0 只在设置输入时用，允许只拷一次），tail 指向该序列最新的一份状态。',
      '<span class="v">n_rs_seq</span>：每个序列保留几份历史快照，用于回滚。' +
      '<br>所以张量的行数是 <span class="v">mem_size * (1 + n_rs_seq)</span>（llama-memory-recurrent.cpp:101）。',
      '实际张量：每层两个（有 PLE 卷积历史时三个）—— <span class="v">r_l</span> / <span class="v">s_l</span> / <span class="v">p_l</span>。',
      '一句话：<span class="k">记忆大小与序列长度无关</span>，所以这类模型推长文本时显存不涨。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(4200, () => { msg.innerHTML = texts[1]; });
    tl.at(8200, () => { msg.innerHTML = texts[2]; });
    tl.at(12600, () => { msg.innerHTML = texts[3]; });
    tl.at(17000, () => { msg.innerHTML = texts[4]; });
    tl.at(21500, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 10 <span class="hl-f">hybrid</span>：attention 层与 recurrent 层拼进一个 memory */
{
  kicker: "L2-04 · 变体",
  title: "<span class=\"hl-f\">hybrid</span>：attention 层与 recurrent 层拼进一个 memory",
  sub: "混合架构的模型一层一个样：一部分层是 attention，一部分是 recurrent，memory 就按层分工。",
  caption: "默认 filter 就是 is_recr()：attn 侧收 !is_recr(il)，recurrent 侧收 is_recr(il)（llama-memory-hybrid.cpp:48-64）。",
  src: "src/llama-memory-hybrid.h",
  mark: [1, 3, 5, 6],
  lineNo: 0,
  code: `// utilizes instances of llama_memory_recurrent and llama_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class llama_memory_hybrid : public llama_memory_i {
//>> ---- src/llama-memory-hybrid.h:89-90 ----
    const std::unique_ptr<llama_kv_cache> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row" id="layers" style="gap:5px;justify-content:center"></div>
      <div class="row" style="gap:9px">
        <div class="card grow" style="border-left-color:var(--a)"><div class="ct" style="color:var(--a)">mem_attn : llama_kv_cache</div>
          <div class="cb">收 <span class="cm" style="margin:0">!is_recr(il)</span> 的层；cell 记账 + K/V 张量</div></div>
        <div class="card grow" style="border-left-color:var(--e)"><div class="ct" style="color:var(--e)">mem_recr : llama_memory_recurrent</div>
          <div class="cb">收 <span class="cm" style="margin:0">is_recr(il)</span> 的层；每层固定状态 r_l / s_l</div></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#layers');
    const kinds = ['A', 'R', 'A', 'R', 'R', 'A', 'R', 'A'];
    const lels = kinds.map((k, i) => {
      const e = U.el('div', { class: 'chip ' + (k === 'A' ? 'a' : 'e'), style: 'width:56px;text-align:center',
                              text: 'L' + i + ' ' + (k === 'A' ? 'attn' : 'recr') });
      host.appendChild(e); return e;
    });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个模型里两种层混排，memory 就分成两个子 memory，各自带一个 layer filter。',
      '<span class="k">filter 是构造参数</span>：默认按 <span class="v">hparams.is_recr(il)</span> 分，' +
      '<br>架构需要时可以在 create_memory 里换成更细的判据。',
      '分工之后，上层看到的是一个 memory：<span class="v">init_batch</span> / <span class="v">apply</span> / ' +
      '<span class="v">seq_*</span> 都同时作用到两个子 memory 上。',
      '两个更专的版本：<span class="v">llama_memory_hybrid_idx</span> 再加第三套 indexer cache（块稀疏注意力，' +
      '<span class="cm" style="margin:0">llama-memory-hybrid-idx.h:12</span>）；' +
      '<span class="v">llama_memory_hybrid_iswa</span> 把 attention 侧换成 iswa（<span class="cm" style="margin:0">llama-memory-hybrid-iswa.h:16</span>）。',
      '一句话：<span class="k">hybrid 不是新的记忆类型，而是"按层路由"</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    lels.forEach((e, i) => tl.at(3200 + i * 700, () => { e.style.opacity = '1'; }));
    lels.forEach(e => e.style.opacity = '.35');
    tl.at(9200, () => { msg.innerHTML = texts[1]; });
    tl.at(13200, () => { msg.innerHTML = texts[2]; });
    tl.at(17200, () => { msg.innerHTML = texts[3]; });
    tl.at(20200, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 11 谁来选 memory：<span class="hl-a">create_memory()</span> */
{
  kicker: "L2-04 · 收束",
  title: "谁来选 memory：<span class=\"hl-a\">create_memory()</span>",
  sub: "一张 switch(arch) 决定用哪个实现；剩下的课都在讲被它选中的那些类。",
  caption: "下一课 L2-05 讲 batch 与模型装配；L2-06 讲图怎么用这些 memory 建 attention。",
  src: "src/llama-model.cpp",
  mark: [0, 2, 5, 6, 15, 19],
  lineNo: 0,
  code: `llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
//>> ---- src/llama-model.cpp:2547-2557 ----
                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            cparams.n_rs_seq,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron) {
//>> ---- src/llama-model.cpp:2757-2760 ----
    }

    return res;
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案', '在哪'],
      [['记忆的对外行为？', 'llama_memory_i 的 15 个虚函数', 'src/llama-memory.h'],
       ['一次 decode 怎么提交？', 'init_batch 规划 -> context.apply() 落笔', '第 3 幕'],
       ['一个 token 放哪？', 'slot_info：token[i] -> cells[idxs[i]]', '第 5 幕'],
       ['K/V 存在哪？', 'cache 自己的 buffer；图只拿 view / set_rows', '第 6 幕'],
       ['为什么有这么多实现？', '层不同、记忆形状不同；选择在 create_memory()', '第 8 / 11 幕']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '为什么不能只写一个 <span class="mono">llama_kv_cache</span> 类，把 SWA、稀疏注意力、SSM 都塞进去？',
      '三个层次的原因，都能在本课源码里指出来：<br>' +
      '1. <b>记忆的形状不同</b>。<span class="mono">llama_memory_recurrent::mem_cell</span> 里只有 ' +
      '<span class="mono">pos/src/src0/tail/seq_id</span>，没有 K/V；它的数据是每层的 ' +
      '<span class="mono">r_l/s_l</span>，行数是 <span class="mono">mem_size*(1+n_rs_seq)</span>，' +
      '与序列长度无关。硬塞进 KV cache 会让 cell 结构里出现大量永远为空的字段。<br>' +
      '2. <b>层与层之间也不同</b>。iswa 把层按 <span class="mono">is_swa(il)</span> 分成两套 cache，' +
      'hybrid 按 <span class="mono">is_recr(il)</span> 分成两个子 memory；dsa/msa/hybrid_idx 还要为稀疏注意力' +
      '多存一份 indexer。这些差异是<b>按层</b>的，不是整个模型一个开关。<br>' +
      '3. <b>但上层需要统一口径</b>。建图、序列操作、state 存盘恢复都只认 ' +
      '<span class="mono">llama_memory_i</span>；组合型 memory 用 ' +
      '<span class="mono">llama_memory_status_combine()</span> 把子 memory 的状态合并上报，' +
      '所以 10 个类对外长得一样。<br>' +
      '结论：抽象的是<b>行为</b>，不是数据布局。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '把这一课压成五问五答：',
      '<span class="k">对外行为</span>统一，<span class="k">数据布局</span>各异 —— 这就是整个设计。',
      'K/V 张量跨 decode 存活、自己管 buffer，所以它的生命周期与图中间张量完全不同。',
      '选择点只有一个：<span class="v">llama_model::create_memory()</span>，按 arch 分派。',
      '下一步：L2-05 看 batch 怎么被切成 ubatch；L2-06 看图怎么用这些 memory 建 attention。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2400, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
    }));
    tl.at(22000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
  }
},

];
