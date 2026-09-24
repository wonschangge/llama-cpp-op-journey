/* ==========================================================================
   L1-04 · 量化块结构：算子内层的压缩数据
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 量化权重是一串<span class="hl-a">定长块</span>，不是 float 数组 */
{
  kicker: "L1-04 · 全局",
  title: "量化权重是一串<span class=\"hl-a\">定长块</span>，不是 float 数组",
  sub: "一个块覆盖固定个数的元素并自带 scale；张量的字节数 = 块数 × 每块字节数。",
  caption: "回顾 L1-01：nb[0] = ggml_type_size(type)。这里说的 type_size 就是\"一块多少字节\"。",
  src: "ggml/src/ggml-common.h",
  mark: [0, 1, 3, 4],
  lineNo: 86,
  code: `// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">模型里的权重字节</span><span class="arrow">-></span>
        <span class="chip c">block_q4_0[]</span><span class="arrow">-></span>
        <span class="chip b">dequantize_row_q4_0()</span><span class="arrow">-></span>
        <span class="chip d">量化点积内核 · L5-04</span>
      </div>
      <div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '块 = 定长记录', b: '一个块覆盖固定个数的元素并自带 scale，块内字节数固定。', m: 'struct block_q4_0' },
      { c: 'c', t: '块大小由宏给出', b: 'QK4_0 = 32、QK_K = 256、K_SCALE_SIZE = 12。', m: 'ggml-common.h:89-90' },
      { c: 'b', t: '两个消费者', b: '先反量化成 float，或直接做量化点积（L5-03 / L5-04）。', m: 'dequantize_row_*()' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:216px' }); host.appendChild(e); return e; });
    els.forEach(e => { e.style.opacity = '.30'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '模型权重不是 float 数组：<span class="k">它是一串定长的块</span>。',
      '<span class="v">QK4_0 = 32</span>：32 个元素一块；<span class="v">QK_K = 256</span>：K-quant 是 256 个元素一个超块。<br>两个数量级的块共存于同一套张量布局里。',
      '<span class="v">K_SCALE_SIZE = 12</span>：Q4_K 里 8 个 scale 加 8 个 min 一共只占 12 字节。<br>第 6、7 幕会把这个数字拆开看。',
      '回顾 L1-01：<span class="v">nb[0] = ggml_type_size(type)</span>。<br>换类型就换块大小、换块字节数，也就换了内核。',
      '本课路线：块长什么样（2、6 幕）-> 怎么写/怎么读（3-5 幕）-> 6-bit 解包（7 幕）-> 绑定（8 幕）。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(9800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
    tl.at(12800, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">18 字节</span> = 2 字节 scale + 16 字节 4-bit */
{
  kicker: "L1-04 · 小块",
  title: "★ <span class=\"hl-a\">18 字节</span> = 2 字节 scale + 16 字节 4-bit",
  sub: "QK4_0 = 32：32 个 4-bit 正好 16 字节；再加一个 fp16 的 d，就是 18 字节。",
  caption: "static_assert 把这条等式钉在编译期：sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0/2，而 ggml_half 就是 uint16_t（2 字节）。",
  src: "ggml/src/ggml-common.h",
  mark: [0, 2, 3, 6, 8, 12, 13, 17, 19],
  lineNo: 194,
  code: `#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
//>> QK4_0 / 2 = 16 字节 —— 32 个 4-bit 权重正好塞满 16 字节，这就是块大小取 32 的原因
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div id="s0"></div>
      <div id="s1"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    function strip(host, caption, segs) {
      host.appendChild(U.el('div', { class: 'cm', text: caption, style: 'margin-bottom:3px' }));
      const s = U.el('div', { style: 'display:flex;height:26px;border:1px solid var(--border);border-radius:5px;overflow:hidden;width:100%' });
      segs.forEach(g => {
        const e = U.el('div', { style: 'flex:' + g.n + ' 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:8.5px;overflow:hidden;white-space:nowrap;background:' + g.bg + ';color:var(--' + g.c + ');border-right:1px solid var(--border)' });
        e.textContent = g.t;
        s.appendChild(e);
      });
      host.appendChild(s);
      return s;
    }
    const s0 = wrap.querySelector('#s0');
    const s1 = wrap.querySelector('#s1');
    const b0 = strip(s0, 'block_q4_0 —— 18 字节 / 32 个权重',
      [{ n: 2,  t: 'd',      c: 'a', bg: 'rgba(88,166,255,.30)' },
       { n: 16, t: 'qs[16] —— 32 个 4-bit nibble', c: 'b', bg: 'rgba(63,185,80,.16)' }]);
    const b1 = strip(s1, 'block_q4_1 —— 20 字节 / 32 个权重',
      [{ n: 2,  t: 'd',      c: 'a', bg: 'rgba(88,166,255,.30)' },
       { n: 2,  t: 'm',      c: 'e', bg: 'rgba(247,120,186,.26)' },
       { n: 16, t: 'qs[16] —— 32 个 4-bit nibble', c: 'b', bg: 'rgba(63,185,80,.16)' }]);

    const t = U.table(['类型', '块内字段', '字节 / 块', 'bit / 权重'],
      [['Q4_0', 'd(2) + qs(16)', '18', '4.5'],
       ['Q4_1', 'd(2) + m(2) + qs(16)', '20', '5.0']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'QK4_0 = 32：<span class="k">32 个 4-bit 权重正好 16 字节</span>。块大小不是随手取的，它由"字节数要整齐"反推出来。',
      '再加一个 fp16 的 <span class="v">d</span>（2 字节）：<span class="v">sizeof(block_q4_0) = 2 + 16 = 18</span>。<br>18 × 8 / 32 = <span class="k">4.5 bit / 权重</span>。',
      'Q4_1 多一个 <span class="v">m</span>（min）：<span class="v">2 + 2 + 16 = 20 字节</span>，即 <span class="k">5.0 bit / 权重</span>。<br>多出来的 2 字节买的是"非对称量化"（第 5 幕对照）。',
      '<span class="v">static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, ...)</span>：<br>改了布局却忘了改断言，<span class="k">编译就过不去</span>。这与 L1-01 的 nb[] 约定是同一件事的两种写法。'
    ];
    tl.at(600, () => { U.markLines(document, [0, 2, 3]); msg.innerHTML = texts[0]; });
    tl.at(4200, () => { U.markLines(document, [6]); rows.forEach((r, k) => { r.className = k === 0 ? 'on' : ''; }); msg.innerHTML = texts[1]; });
    tl.at(8400, () => { U.markLines(document, [8, 12, 13, 17]); rows.forEach((r, k) => { r.className = k === 1 ? 'on' : ''; }); msg.innerHTML = texts[2]; });
    tl.at(13000, () => { U.markLines(document, [19]); rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 3 一个类型一个函数：<span class="hl-c">名字里的类型号就是 GGML_TYPE</span> */
{
  kicker: "L1-04 · 函数族",
  title: "一个类型一个函数：<span class=\"hl-c\">名字里的类型号就是 GGML_TYPE</span>",
  sub: "ggml-quants.h 把反量化函数按同一套命名排开：dequantize_row_ 加类型名，参数就是对应的 block_ 结构体。",
  caption: "同一个头文件的上半部分（17-34 行）用同样的规则排开量化函数 quantize_row_*_ref —— 见 source.md 第五节。",
  src: "ggml/src/ggml-quants.h",
  mark: [0, 3, 15, 17, 18],
  lineNo: 45,
  code: `// Dequantization
GGML_API void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q2_0(const block_q2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">GGML_TYPE_Q4_0</span><span class="arrow">-></span>
        <span class="chip c">dequantize_row_q4_0()</span><span class="arrow">-></span>
        <span class="chip b">traits.to_float（第 8 幕）</span>
      </div>
      <div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'c', t: '反量化 · 15 个声明', b: '从 q1_0 一路排到 q8_K；每个都输出 float。第 53 行那个 q8_1 被注释掉了。', m: 'dequantize_row_q4_K()' },
      { c: 'a', t: '参数就是块类型', b: '入参是 block_<类型名> 指针，所以"块布局"在这里被写死。', m: 'const block_q4_0 * x' },
      { c: 'b', t: '声明与实现分开', b: '声明在 .h（本幕），实现在 ggml-quants.c（4-5、7 幕）。', m: 'GGML_API void' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:216px' }); host.appendChild(e); return e; });
    els.forEach(e => { e.style.opacity = '.30'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先认名字：<span class="v">dequantize_row_</span> + <span class="k">类型名的小写形式</span>。',
      '一共 15 个反量化声明（46-63 行）：<br>小块 Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0，K-quant Q2_K..Q6_K/Q8_K，还有 mxfp4/nvfp4。',
      '<span class="k">每个函数的参数类型就是它对应的块结构体</span>：<br><span class="v">dequantize_row_q4_0(const block_q4_0 * x, ...)</span> —— 布局不是约定，是签名。',
      '<span class="v">dequantize_row_q4_K</span> 与 <span class="v">dequantize_row_q6_K</span> 也在这一串里：<br>第 6、7 幕看 Q4_K，它比 Q4_0 多一层"超块"。',
      '<span class="v">GGML_API</span>：这些函数被 CPU 后端直接链接（注释写明 "because they used by the CPU backend"），<br>再由 traits 表挂到类型号上 —— 第 8 幕。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
    tl.at(3600, () => { els.forEach((e, k) => { e.style.opacity = k === 0 ? '1' : '.30'; }); msg.innerHTML = texts[1]; U.markLines(document, [3]); });
    tl.at(7200, () => { els.forEach((e, k) => { e.style.opacity = k === 1 ? '1' : '.30'; }); msg.innerHTML = texts[2]; U.markLines(document, [3, 15]); });
    tl.at(10800, () => { els.forEach((e, k) => { e.style.opacity = k === 2 ? '1' : '.30'; }); msg.innerHTML = texts[3]; U.markLines(document, [15, 17, 18]); });
    tl.at(14000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [0, 3, 15, 17, 18]); });
  }
},

/* ------------------------------------------------------ 4 dequantize_row_q4_0：<span class="hl-b">一个字节管相隔 16 的两个元素</span> */
{
  kicker: "L1-04 · 反量化",
  title: "dequantize_row_q4_0：<span class=\"hl-b\">一个字节管相隔 16 的两个元素</span>",
  sub: "低 4 位给 y[j]，高 4 位给 y[j + qk/2]，也就是 y[j+16] —— 不是相邻两个。",
  caption: "这个\"前后半块交错\"的排布，正是 L5-04 的 repack 要把 q4_0 重排成 4x4 交错布局的原因。",
  src: "ggml/src/ggml-quants.c",
  mark: [8, 12, 13, 15, 16],
  lineNo: 459,
  code: `void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
//>> d 从 fp16 解回 float —— "块自带 scale" 落到一行代码就是这一句

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `
      <div class="cm">16 个 qs 字节 → 32 个输出元素（一个 Q4_0 块）</div>
      <div class="row" id="rA" style="gap:3px"></div>
      <div class="row" id="rQ" style="gap:3px"></div>
      <div class="row" id="rB" style="gap:3px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    function cell(host, txt, color) {
      const e = U.el('div', { style: 'flex:1 1 0;min-width:0;height:19px;border:1px solid var(--border);border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:8.5px;overflow:hidden;color:var(--' + color + ');background:#10151b' });
      e.textContent = txt;
      host.appendChild(e);
      return e;
    }
    const rA = wrap.querySelector('#rA'), rQ = wrap.querySelector('#rQ'), rB = wrap.querySelector('#rB');
    const A = [], Q = [], B = [];
    for (let j = 0; j < 16; j++) { A.push(cell(rA, 'y[' + j + ']', 'a')); }
    for (let j = 0; j < 16; j++) { Q.push(cell(rQ, 'qs[' + j + ']', 'c')); }
    for (let j = 0; j < 16; j++) { B.push(cell(rB, 'y[' + (j + 16) + ']', 'b')); }

    function focus(j) {
      A.forEach((e, k) => { e.style.borderColor = (k === j) ? 'var(--a)' : 'var(--border)'; e.style.background = (k === j) ? 'rgba(88,166,255,.20)' : '#10151b'; });
      Q.forEach((e, k) => { e.style.borderColor = (k === j) ? 'var(--c)' : 'var(--border)'; e.style.background = (k === j) ? 'rgba(210,153,34,.20)' : '#10151b'; });
      B.forEach((e, k) => { e.style.borderColor = (k === j) ? 'var(--b)' : 'var(--border)'; e.style.background = (k === j) ? 'rgba(63,185,80,.20)' : '#10151b'; });
    }

    const msg = wrap.querySelector('#msg');
    tl.at(600, () => { U.markLines(document, [8]); msg.innerHTML = '先把 <span class="v">d</span> 从 fp16 解回来：<span class="k">块自带的那一个 scale，是全块共用的</span>。'; });
    tl.at(3600, () => {
      focus(0); U.markLines(document, [12, 15]);
      msg.innerHTML = '取字节 j 的<span class="k">低 4 位</span>：<span class="v">x0 = (qs[j] & 0x0F) - 8</span>，写到 <span class="v">y[j]</span>。';
    });
    tl.at(7200, () => {
      focus(0); U.markLines(document, [13, 16]);
      msg.innerHTML = '取同一个字节的<span class="k">高 4 位</span>：<span class="v">x1 = (qs[j] >> 4) - 8</span>，写到 <span class="v">y[j + qk/2]</span>。';
    });
    tl.at(10800, () => {
      focus(5); U.markLines(document, [12, 13, 15, 16]);
      msg.innerHTML = '换成 j = 5 也一样：<span class="v">qs[5]</span> 的低位给 <span class="v">y[5]</span>、高位给 <span class="v">y[21]</span>。<br><span class="k">一个字节跨越半块</span>，16 个字节刚好铺满 32 个元素。';
    });
    tl.at(14200, () => {
      focus(0); U.markLines(document, [8, 12, 13, 15, 16]);
      msg.innerHTML = '减去 8 是因为 q 落在 <span class="v">[0, 15]</span>：还原出的范围是 <span class="v">[-8d, 7d]</span>。<br>这就是 Q4_0 的<span class="k">对称</span>量化 —— 第 5 幕看它怎么写出来。';
    });
  }
},

/* ------------------------------------------------------ 5 ★ <span class="hl-a">d = max / -8</span>：Q4_0 对称，没有 min */
{
  kicker: "L1-04 · 量化侧",
  title: "★ <span class=\"hl-a\">d = max / -8</span>：Q4_0 对称，没有 min",
  sub: "先扫一遍求绝对最大 amax（连同它的符号一起记进 max），再算 d 与 id，最后把 x*id 四舍五入加 8 塞进 nibble。",
  caption: "Q4_1 走的是另一条路（非对称）：d = (max-min)/15 且多存一个 m —— 这解释了第 2 幕里它为什么是 20 字节。",
  src: "ggml/src/ggml-quants.c",
  mark: [8, 13, 19, 23, 29, 30, 32, 33],
  lineNo: 113,
  code: `void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
//>> d = max / -8：d 的符号与 max 相反，于是 (q - 8) * d 能把 q 还原到 [-8d, 7d]
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">amax / max</span><span class="arrow">-></span>
        <span class="chip c">d = max / -8</span><span class="arrow">-></span>
        <span class="chip b">id = 1/d</span><span class="arrow">-></span>
        <span class="chip d">(int8_t)(x*id + 8.5)</span><span class="arrow">-></span>
        <span class="chip e">MIN(15, .)</span>
      </div>
      <div class="cm" id="cap"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    wrap.querySelector('#cap').textContent =
      '算例：这一块里 amax = 1.0、max = -1.0（绝对值最大的是 -1.0），于是 d = 0.125、id = 8；源码用 (int8_t)(x*id + 8.5f) 做四舍五入。';

    const t = U.table(['x', 'x*id', '+8.5', '(int8_t)', 'MIN(15,·)', '(q-8)*d'],
      [['-1.000', '-8.0', '0.5',  '0',  '0',  '-1.000'],
       ['0.000',  '0.0',  '8.5',  '8',  '8',  '0.000'],
       ['0.875',  '7.0',  '15.5', '15', '15', '0.875'],
       ['1.000',  '8.0',  '16.5', '16', '15', '0.875']],
      { monoCols: [0, 1, 2, 3, 4, 5] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '第一步扫出 <span class="v">amax</span> 与取到 amax 的那个<span class="k">带符号值 max</span>。',
      '<span class="v">d = max / -8</span>、<span class="v">id = 1/d</span>，写回 <span class="v">y[i].d</span>（fp16）。<br>注意：<span class="k">全程没有 min</span> —— Q4_0 是围绕 0 对称的。',
      '取整用的是 <span class="v">MIN(15, (int8_t)(x0 + 8.5f))</span>：<br>+8.5 把 [-8, 7] 平移到 [0, 15] 且四舍五入，MIN 兜住上溢。',
      '两个 nibble 合进一个字节：<span class="v">qs[j] = xi0; qs[j] |= xi1 &lt;&lt; 4;</span><br>低位是本块前半的 <span class="v">x[j]</span>，高位是后半的 <span class="v">x[j+16]</span> —— 正好对上第 4 幕的解包。',
      '看表的最后一行：x = 1.000 被 MIN 截到 15，还原成 0.875。<br><span class="k">对称量化的代价就藏在这里</span>：正方向的极值会被压一点。Q4_1 用 min 换掉了这个偏差，代价是 2 字节。'
    ];
    tl.at(600, () => { U.markLines(document, [8, 13]); msg.innerHTML = texts[0]; rows.forEach(r => { r.className = ''; }); });
    tl.at(4600, () => { U.markLines(document, [19, 23]); msg.innerHTML = texts[1]; });
    tl.at(9000, () => { U.markLines(document, [29, 30]); rows.forEach((r, k) => { r.className = (k === 0 || k === 1) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
    tl.at(13400, () => { U.markLines(document, [32, 33]); rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[3]; });
    tl.at(17800, () => { U.markLines(document, [29, 30, 32, 33]); rows.forEach((r, k) => { r.className = (k === 3) ? 'on' : ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 ★ Q4_K：<span class="hl-a">144 字节</span>装 256 个权重，12 字节装 16 个 scale/min */
{
  kicker: "L1-04 · 超块",
  title: "★ Q4_K：<span class=\"hl-a\">144 字节</span>装 256 个权重，12 字节装 16 个 scale/min",
  sub: "8 个 32 元素子块共用一个超块头；每个子块的 scale 与 min 都只用 6 bit，8 ×(6+6) = 96 bit 正好 12 字节。",
  caption: "对比第 2 幕：Q4_0 的 18/32 与 Q4_K 的 144/256 都是 0.5625 字节/权重 —— 同样的码率，花法完全不同。",
  src: "ggml/src/ggml-common.h",
  mark: [1, 3, 7, 8, 12, 14, 16],
  lineNo: 323,
  code: `// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
//>> K_SCALE_SIZE = 12 字节 = 96 bit：8 个 6-bit scale + 8 个 6-bit min，一位不多一位不少
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="cm">block_q4_K —— 144 字节 / 256 个权重（= 8 个子块 × 32）</div>
      <div id="strip" style="display:flex;height:30px;border:1px solid var(--border);border-radius:5px;overflow:hidden;width:100%"></div>
      <div class="row" id="subs" style="gap:4px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const segs = [
      { n: 4,   t: 'dm',          c: 'a', bg: 'rgba(88,166,255,.30)' },
      { n: 12,  t: 'scales[12]',  c: 'c', bg: 'rgba(210,153,34,.30)' },
      { n: 128, t: 'qs[128] —— 256 个 4-bit', c: 'b', bg: 'rgba(63,185,80,.16)' }
    ];
    const strip = wrap.querySelector('#strip');
    segs.forEach(g => {
      const e = U.el('div', { style: 'flex:' + g.n + ' 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:9px;overflow:hidden;white-space:nowrap;background:' + g.bg + ';color:var(--' + g.c + ');border-right:1px solid var(--border)' });
      e.textContent = g.t;
      strip.appendChild(e);
    });
    const subHost = wrap.querySelector('#subs');
    const subs = [];
    for (let j = 0; j < 8; j++) {
      const e = U.el('div', { style: 'flex:1 1 0;min-width:0;height:32px;border:1px solid var(--border);border-radius:4px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:8.5px;color:var(--muted);background:#10151b;overflow:hidden' });
      e.innerHTML = '<span style="color:var(--c)">子块 ' + j + '</span><span>6-bit sc + 6-bit min</span>';
      subHost.appendChild(e);
      subs.push(e);
    }

    const msg = wrap.querySelector('#msg');
    const texts = [
      '超块头是 <span class="v">dm</span>：<span class="k">d 与 dmin 两个 fp16</span>，共 4 字节。它们是"scale 的 scale"。',
      '<span class="v">scales[12]</span>：8 个子块各自的 scale 和 min，各 6 bit。<br>8 × (6 + 6) = 96 bit = <span class="k">12 字节</span>，这就是 K_SCALE_SIZE。',
      '<span class="v">qs[128]</span>：256 个 4-bit 权重。<br>4 + 12 + 128 = <span class="v">144 字节</span>，与 static_assert 的 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 一致。',
      '子块与 Q4_0 的小块同宽（32 个元素），但<span class="k">不再各自带 fp16 scale</span>，<br>而是共享超块头 + 一个 6-bit 的量化 scale。',
      '算总账：144 × 8 / 256 = <span class="k">4.5 bit / 权重</span>，与 Q4_0 完全相同。<br>源码注释也写着 "Effectively 4.5 bits per weight"。'
    ];
    tl.at(600, () => { U.markLines(document, [7, 8]); msg.innerHTML = texts[0]; subs.forEach(e => { e.style.opacity = '.40'; }); });
    tl.at(4800, () => { U.markLines(document, [12]); msg.innerHTML = texts[1]; });
    tl.at(9000, () => { U.markLines(document, [13, 14]); msg.innerHTML = texts[2]; });
    tl.at(13200, () => { U.markLines(document, [1, 16]); subs.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
    tl.at(17600, () => { U.markLines(document, [1, 3, 12, 13, 14, 16]); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 6-bit 的 scale/min 怎么解回来：<span class="hl-c">低位在自己字节，高位从前 4 个字节借</span> */
{
  kicker: "L1-04 · 解包",
  title: "6-bit 的 scale/min 怎么解回来：<span class=\"hl-c\">低位在自己字节，高位从前 4 个字节借</span>",
  sub: "bytes[0..3] 的低 6 位是 scale 0..3，bytes[4..7] 的低 6 位是 min 0..3；它们的高 2 位被借给 scale/min 4..7，低 4 位放在 bytes[8..11]。",
  caption: "反量化的消费者是 dequantize_row_q4_K：它每 64 个元素调两次 get_scale_min_k4，见 source.md 第九节。",
  src: "ggml/src/ggml-quants.c",
  mark: [0, 1, 2, 5, 7],
  lineNo: 880,
  code: `static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
//>> j < 4：低 6 位直接就是第 j 个子块的 scale 与 min
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
//>> j >= 4：低 4 位在本字节，高 2 位从前 4 个字节的最高两位左移上来
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="cm">scales[12]：12 个字节 = 96 bit，装下 8 个 scale + 8 个 min（各 6 bit）</div>
      <div id="bytes" style="display:flex;gap:3px"></div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#bytes');
    const bytes = [];
    for (let j = 0; j < 12; j++) {
      const e = U.el('div', { style: 'flex:1 1 0;min-width:0;height:26px;border:1px solid var(--border);border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:8.5px;background:#10151b;color:var(--dim);overflow:hidden' });
      e.textContent = 'q[' + j + ']';
      host.appendChild(e);
      bytes.push(e);
    }
    function tint(list, color, rgba) {
      list.forEach(j => { bytes[j].style.borderColor = 'var(--' + color + ')'; bytes[j].style.background = rgba; bytes[j].style.color = 'var(--' + color + ')'; });
    }
    function clear() {
      bytes.forEach(e => { e.style.borderColor = 'var(--border)'; e.style.background = '#10151b'; e.style.color = 'var(--dim)'; });
    }

    const t = U.table(['j', '*d（scale，6 bit）', '*m（min，6 bit）'],
      [['0..3', 'q[j] & 63', 'q[j+4] & 63'],
       ['4..7', '(q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)', '(q[j+4] >> 4) | ((q[j] >> 6) << 4)']],
      { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    tl.at(600, () => {
      clear(); tint([0, 1, 2, 3], 'a', 'rgba(88,166,255,.20)');
      U.markLines(document, [1, 2]); rows.forEach((r, k) => { r.className = k === 0 ? 'on' : ''; });
      msg.innerHTML = '先看 <span class="v">j &lt; 4</span>：<span class="k">q[j] 的低 6 位</span>就是第 j 个 scale，<span class="k">q[j+4] 的低 6 位</span>就是第 j 个 min。<br>只用了 8 个字节里最普通的 6 bit。';
    });
    tl.at(5000, () => {
      clear(); tint([4, 5, 6, 7], 'b', 'rgba(63,185,80,.20)');
      U.markLines(document, [5, 7]); rows.forEach((r, k) => { r.className = k === 1 ? 'on' : ''; });
      msg.innerHTML = '再看 <span class="v">j &gt;= 4</span>：第 j 个 scale 的低 4 位在 <span class="v">q[j+4]</span> 的低半字节，<br>高 2 位则是 <span class="v">q[j-4]</span> 的最高两位左移 4 位拼上来。';
    });
    tl.at(10000, () => {
      clear(); tint([8, 9, 10, 11], 'c', 'rgba(210,153,34,.22)');
      msg.innerHTML = '被"借走"高 2 位的就是 <span class="v">q[0..7]</span> —— 它们本来就只用 6 bit，剩下 2 bit 正好给 4..7 用。';
    });
    tl.at(14000, () => {
      clear(); tint([0, 1, 2, 3], 'a', 'rgba(88,166,255,.20)'); tint([4, 5, 6, 7], 'b', 'rgba(63,185,80,.20)'); tint([8, 9, 10, 11], 'c', 'rgba(210,153,34,.22)');
      U.markLines(document, [0, 1, 2, 5, 7]);
      msg.innerHTML = '8 个字节各出 6 bit + 4 个字节各出 4 bit × 2 = 96 bit = <span class="k">12 字节，正好装满</span>。<br>没有任何浪费，代价是<span class="k">取用时要移位拼接</span>。';
    });
  }
},

/* ------------------------------------------------------ 8 类型号 = <span class="hl-a">块大小</span> + <span class="hl-b">反量化函数</span> */
{
  kicker: "L1-04 · 收束",
  title: "类型号 = <span class=\"hl-a\">块大小</span> + <span class=\"hl-b\">反量化函数</span>",
  sub: "ggml_type_traits 把三件事绑在一个类型号上：blck_size（多少元素一块）、type_size（一块多少字节）、to_float（怎么还原成 float）。",
  caption: "下一课 L1-05 讲上下文、线程与优化器接口。",
  src: "ggml/src/ggml.c",
  mark: [16, 19, 21, 24, 25],
  lineNo: 677,
  code: `    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = sizeof(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = sizeof(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
//>> GGML_TYPE_Q4_0 的登记项 —— 块的"大小"和"还原函数"都在这里绑定
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
//>> blck_size = QK4_0 = 32：多少元素算一块
        .type_size                = sizeof(block_q4_0),
//>> type_size = sizeof(block_q4_0) = 18：一块多少字节
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div id="s0"></div>
      <div id="s1"></div>
      <div id="tbl"></div>
      <div id="ex"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    function rowStrip(host, labelHtml, groups) {
      const box = U.el('div', { class: 'col', style: 'gap:3px;width:100%' });
      box.appendChild(U.el('div', { class: 'cm', html: labelHtml }));
      const s = U.el('div', { style: 'display:flex;height:24px;border:1px solid var(--border);border-radius:5px;overflow:hidden;width:100%' });
      groups.forEach(g => {
        const outer = U.el('div', { style: 'flex:' + g.w + ' 1 0;min-width:0;display:flex;border-right:1px solid var(--border)' });
        (g.sub || [g]).forEach(ss => {
          const e = U.el('div', { style: 'flex:' + ss.w + ' 1 0;min-width:0;display:flex;align-items:center;justify-content:center;font-size:8.5px;overflow:hidden;white-space:nowrap;background:' + ss.bg + ';color:var(--' + ss.c + ')' });
          e.textContent = ss.t || '';
          outer.appendChild(e);
        });
        s.appendChild(outer);
      });
      box.appendChild(s);
      host.appendChild(box);
    }

    const g40 = [];
    for (let i = 0; i < 8; i++) {
      g40.push({ w: 18, sub: [
        { w: 2,  t: '', c: 'a', bg: 'rgba(88,166,255,.55)' },
        { w: 16, t: (i === 0 ? 'qs 16B' : ''), c: 'b', bg: 'rgba(63,185,80,.16)' }
      ] });
    }
    rowStrip(wrap.querySelector('#s0'),
      'Q4_0 · 256 个权重 = 8 块 × 18 B = 144 B　<span style="color:var(--a)">■</span> d 2B×8 = 16B　<span style="color:var(--b)">■</span> qs 16B×8 = 128B',
      g40);
    rowStrip(wrap.querySelector('#s1'),
      'Q4_K · 256 个权重 = 1 个超块 = 144 B　<span style="color:var(--a)">■</span> dm 4B　<span style="color:var(--c)">■</span> scales 12B　<span style="color:var(--b)">■</span> qs 128B',
      [{ w: 144, sub: [
        { w: 4,   t: '', c: 'a', bg: 'rgba(88,166,255,.55)' },
        { w: 12,  t: '', c: 'c', bg: 'rgba(210,153,34,.30)' },
        { w: 128, t: 'qs 128B', c: 'b', bg: 'rgba(63,185,80,.16)' }
      ] }]);

    const t = U.table(['', 'Q4_0', 'Q4_K'],
      [['blck_size', '32', '256'],
       ['type_size', '18 B', '144 B'],
       ['bit / 权重', '4.5', '4.5'],
       ['scale 粒度', '1 个 fp16 / 32', '1 个 6-bit / 32'],
       ['min 的表示', '无（对称）', '1 个 6-bit / 32'],
       ['内核代价', '无需位运算', '每超块解 16 个 6-bit']],
      { monoCols: [1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一个 <span class="mono">ggml_tensor</span>，<span class="mono">type = GGML_TYPE_Q4_0</span>、' +
      '<span class="mono">ne = [4096, 1, 1, 1]</span>。它占多少字节？换成 ' +
      '<span class="mono">GGML_TYPE_Q4_K</span> 呢？',
      'Q4_0：<span class="mono">blck_size = 32</span>、<span class="mono">type_size = 18</span>，' +
      '4096 / 32 = 128 块，128 × 18 = <b>2304 字节</b>。<br>' +
      'Q4_K：<span class="mono">blck_size = 256</span>、<span class="mono">type_size = 144</span>，' +
      '4096 / 256 = 16 块，16 × 144 = <b>2304 字节</b>。<br>' +
      '两者都是 4.5 bit/权重，所以字节数一样 —— 差别只在<span class="mono">块怎么切、谁带 scale</span>，' +
      '以及内核要不要做位运算解包。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '同一张表里，每个类型三行：<span class="v">blck_size</span>、<span class="v">type_size</span>、<span class="v">to_float</span>。',
      '<span class="v">[GGML_TYPE_Q4_0]</span>：<span class="v">blck_size = QK4_0</span>（32）、<br><span class="v">type_size = sizeof(block_q4_0)</span>（18）。第 2 幕的字节数在这里生效。',
      '<span class="v">to_float = (ggml_to_float_t) dequantize_row_q4_0</span>：<br><span class="k">类型号 -> 块布局 -> 反量化函数</span>，一条链在这里闭合。',
      '同一行还有 <span class="v">from_float_ref = quantize_row_q4_0_ref</span> —— 第 5 幕那个参考实现。<br>F16 这类非量化类型的 to_float 只是一个 fp16->fp32 转换，没有块。',
      '回到验收点：Q4_0 与 Q4_K 在 256 个权重上都是 144 字节，<span class="k">参数 16 字节 + quants 128 字节</span>；<br>区别是 Q4_0 用 8 个 fp16 scale，Q4_K 用 4 + 12 字节换来了 per-32 的 6-bit scale <b>和</b> min。',
      '一句话：<span class="k">类型号决定的不是"多少位"，而是"块怎么切、scale 怎么存"</span>。<br>内核的一切代价都从这张表开始。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => { r.className = ''; }); U.markLines(document, [16]); });
    tl.at(4200, () => { U.markLines(document, [19, 21]); rows.forEach((r, k) => { r.className = (k === 0 || k === 1) ? 'on' : ''; }); msg.innerHTML = texts[1]; });
    tl.at(8000, () => { U.markLines(document, [24]); rows.forEach((r, k) => { r.className = (k === 2) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
    tl.at(11800, () => { U.markLines(document, [18, 25]); rows.forEach((r, k) => { r.className = (k === 3 || k === 4) ? 'on' : ''; }); msg.innerHTML = texts[3]; });
    tl.at(15600, () => { U.markLines(document, [16, 19, 21, 24, 25]); rows.forEach((r, k) => { r.className = (k === 5) ? 'on' : ''; }); msg.innerHTML = texts[4]; });
    tl.at(18800, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

];
