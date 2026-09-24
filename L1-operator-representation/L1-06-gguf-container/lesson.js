/* ==========================================================================
   L1-06 · GGUF：算子的权重从文件来
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 算子的权重，从文件来 */
{
  kicker: "L1 · 算子的表示",
  title: "算子的权重，从文件来",
  sub: "GGUF 是\"自描述\"的容器：元数据与张量信息写在文件里，读的人不需要任何额外约定。",
  caption: "本课覆盖 ggml/include/gguf.h（格式定义 + API）与 ggml/src/gguf.cpp（读写实现）。它在 L1 里的位置：L1-01..L1-05 讲内存里的算子长什么样，本课讲这些算子的权重怎么从磁盘进内存。",
  src: "ggml/include/gguf.h",
  mark: [],
  lineNo: 1,
  code: `// This file contains functionality related to "GGUF" files, the binary file format used by ggml.`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">模型文件 .gguf</span><span class="arrow">-></span>
        <span class="chip a">gguf_init_from_file</span><span class="arrow">-></span>
        <span class="chip b">ggml_context</span><span class="arrow">-></span>
        <span class="chip c">计算图</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '自描述 · KV 段', b: '任意键值对：超参、词表、chat template、<br>"general.alignment" —— <b>读的人不必预先知道</b>',
        m: 'std::vector<gguf_kv> kv' },
      { c: 'b', t: '结构 · 张量信息段', b: '每个张量一条：名字、维数、形状、类型、<br>以及它在数据区里的字节偏移',
        m: 'std::vector<gguf_tensor_info> info' },
      { c: 'c', t: '对齐 · 数据区', b: '张量数据拼成一块连续区间；<br>起始偏移与每个张量都按 alignment 补齐',
        m: 'size_t alignment = GGUF_DEFAULT_ALIGNMENT' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '权重的旅程：<span class="k">磁盘上的 .gguf</span> -> <span class="k">gguf_context</span>（元数据 + 张量信息）-> <span class="k">ggml_context</span>（真的 ggml_tensor）-> 图。',
      '第一层：<span class="v">KV 段</span>。键、类型、值都写进文件，所以 GGUF 是<b>自描述</b>的 —— 新增一个超参不需要改格式。',
      '第二层：<span class="v">张量信息段</span>。它把"名字"和"数据在第几字节"绑在一起，读的人据此造出 ggml_tensor。',
      '第三层：<span class="v">数据区</span>。所有张量数据连续摆放、按 alignment 对齐 —— 这是 L2-03 能用 mmap 直接指向它、零拷贝的前提。',
      '★ 记住这条主线：<span class="k">元数据在前，数据在后，中间按 alignment 对齐</span>。后面每一幕都是它的一个切片。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(12900, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 文件头：<span class="hl-a">magic</span> · <span class="hl-a">version</span> · <span class="hl-a">n_tensors</span> · <span class="hl-a">n_kv</span> */
{
  kicker: "L1-06 · 文件头",
  title: "文件头：<span class=\"hl-a\">magic</span> · <span class=\"hl-a\">version</span> · <span class=\"hl-a\">n_tensors</span> · <span class=\"hl-a\">n_kv</span>",
  sub: "4 + 4 + 8 + 8 = 24 字节。按顺序读这四样，任何一样不对就直接返回 NULL。",
  caption: "magic 是字符串常量 \"GGUF\"（4 字节）；version 是 uint32；n_tensors / n_kv 是 int64。头里没有\"段长度\"之类的冗余信息 —— 布局完全靠这个固定顺序。",
  src: "ggml/src/gguf.cpp",
  mark: [1, 2, 3, 5, 8, 9, 10, 11, 13, 14],
  lineNo: 0,
  code: `    {
        std::vector<char> magic;
        ok = ok && gr.read(magic, 4);

        if (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        for (uint32_t i = 0; i < magic.size(); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                char c0 = isprint(magic[0]) ? magic[0] : '?';
                char c1 = isprint(magic[1]) ? magic[1] : '?';
                char c2 = isprint(magic[2]) ? magic[2] : '?';
                char c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                return nullptr;
//>> version 依次否掉四种情况：0、字节序不符（低 16 位为 0）、已废弃的 v1、比本软件支持的还新
            }
//>> ---- ggml/src/gguf.cpp:502-539 ----
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        if (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\\n", __func__, ctx->version);
            ok = false;
        }

        if (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\\n", __func__);
            ok = false;
        }
        if (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = false;
        }
    } else {
        ok = false;
//>> version 依次否掉四种情况：0、字节序不符（低 16 位为 0）、已废弃的 v1、比本软件支持的还新
    }

    if (ok && gr.read(n_tensors)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/sizeof(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\\n",
                __func__, n_tensors, SIZE_MAX/sizeof(gguf_tensor_info));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_kv)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_kv < 0 || n_kv > int64_t(SIZE_MAX/sizeof(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\\n",
                    __func__, n_kv, SIZE_MAX/sizeof(gguf_kv));
            ok = false;`,
  duration: 23000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="col" id="layout" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const layout = wrap.querySelector('#layout');
    const rows = [
      { off: '0', n: '4', hex: '47 47 55 46', what: 'magic "GGUF"', c: 'a' },
      { off: '4', n: '4', hex: '03 00 00 00', what: 'version = 3 (uint32)', c: 'b' },
      { off: '8', n: '8', hex: 'n_tensors (int64)', what: '张量个数', c: 'c' },
      { off: '16', n: '8', hex: 'n_kv (int64)', what: '键值对个数', c: 'd' },
      { off: '24', n: '..', hex: 'KV 段 / 张量信息段 ...', what: '见后面几幕', c: 'e' }
    ];
    const els = rows.map(r => {
      const e = U.el('div', { style: 'display:flex;gap:8px;align-items:center;border-left:3px solid var(--' + r.c + ');background:#10151b;border-radius:5px;padding:4px 7px' });
      e.innerHTML = '<span class="cm" style="margin:0;width:54px;flex:0 0 auto">@' + U.esc(r.off) + '</span>' +
        '<span class="cm" style="margin:0;width:34px;flex:0 0 auto;color:var(--' + r.c + ')">' + U.esc(r.n + 'B') + '</span>' +
        '<span style="font-family:var(--mono);font-size:10px;flex:1 1 auto;overflow-wrap:anywhere">' + U.esc(r.hex) + '</span>' +
        '<span class="cb" style="flex:0 0 auto">' + U.esc(r.what) + '</span>';
      layout.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '文件头一共 <span class="v">24 字节</span>，四个字段依次读。先看读 magic 的那一段：',
      '<span class="v">gr.read(magic, 4)</span> 读 4 字节，再逐字节与 <span class="k">GGUF_MAGIC[i]</span> 比较。' +
      '不一致就打印实际读到的四个字符（不可打印的显示成 ?）并<b>返回 nullptr</b>。',
      '然后是版本：<span class="v">ctx->version</span> 直接读进 context。四道关卡依次否掉 ' +
      '<span class="k">0</span>、<span class="k">字节序不符</span>、<span class="k">v1</span>、<span class="k">> GGUF_VERSION</span>。',
      '接着读 <span class="v">n_tensors</span>，再读 <span class="v">n_kv</span>。两者都要夹在 ' +
      '<span class="k">[0, SIZE_MAX/sizeof(...)]</span> 之内 —— 它们会被用来分配容器。',
      '四处读失败都归到同一个出口：<span class="k">failed to read header</span>，' +
      '然后 gguf_free(ctx) + return nullptr。头都没读对，后面的字节就没有解释的依据了。',
      '★ 一张 4+4+8+8 的字节图就是全部：<span class="v">magic -> version -> n_tensors -> n_kv</span>，' +
      '之后才轮到 KV 段。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i, texts.length - 1)];
    }));
    tl.at(16200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
    tl.at(19500, () => { msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 3 ★ KV 段：GGUF 为什么是<span class="hl-a">自描述</span>的 */
{
  kicker: "L1-06 · KV 段",
  title: "★ KV 段：GGUF 为什么是<span class=\"hl-a\">自描述</span>的",
  sub: "键是\"长度 + 字节\"，类型是 int32 枚举，\"元素个数\"只在数组时出现。任何元数据都能塞进来。",
  caption: "编码规则（gguf.h 注释第 26-28 行）：字符串 = uint64 长度 + 去掉结尾 \\0 的字节；枚举一律 int32；bool 一律 int8。所以同一个文件里既有超参，也能有整段 chat template。",
  src: "ggml/include/gguf.h",
  mark: [0, 1, 3, 4, 6, 13, 15, 18, 20, 28],
  lineNo: 0,
  code: `// Strings are serialized as the string length (uint64_t) followed by the C string without the null terminator.
// All enums are stored as int32_t.
// All bool values are stored as int8_t.
//>> general.alignment 是"元数据"，但它决定数据区的字节布局
// If the special key "general.alignment" (uint32_t) is defined it is used for alignment,
//   otherwise GGUF_DEFAULT_ALIGNMENT is used.
//>> u8=0 起，u64/i64/f64 在末尾补齐到 13 个类型
//>> ---- ggml/include/gguf.h:41-42 ----
#define GGUF_MAGIC   "GGUF"
#define GGUF_VERSION 3
//>> ---- ggml/include/gguf.h:44-46 ----
#define GGUF_KEY_GENERAL_ALIGNMENT "general.alignment"

#define GGUF_DEFAULT_ALIGNMENT 32
//>> general.alignment 是"元数据"，但它决定数据区的字节布局
//>> ---- ggml/include/gguf.h:53-68 ----
    enum gguf_type {
        GGUF_TYPE_UINT8   = 0,
        GGUF_TYPE_INT8    = 1,
//>> general.alignment 是"元数据"，但它决定数据区的字节布局
        GGUF_TYPE_UINT16  = 2,
        GGUF_TYPE_INT16   = 3,
//>> u8=0 起，u64/i64/f64 在末尾补齐到 13 个类型
        GGUF_TYPE_UINT32  = 4,
        GGUF_TYPE_INT32   = 5,
        GGUF_TYPE_FLOAT32 = 6,
        GGUF_TYPE_BOOL    = 7,
        GGUF_TYPE_STRING  = 8,
        GGUF_TYPE_ARRAY   = 9,
        GGUF_TYPE_UINT64  = 10,
        GGUF_TYPE_INT64   = 11,
        GGUF_TYPE_FLOAT64 = 12,
        GGUF_TYPE_COUNT,       // marks the end of the enum
    };`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:7px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:3px">一条 KV 在文件里的字节序列</div>';
    const kv = [
      { t: 'key 长度', b: 'uint64', c: 'a' },
      { t: 'key 字节', b: '不含结尾 \0', c: 'a' },
      { t: 'value 类型', b: 'gguf_type（int32）', c: 'b' },
      { t: '数组？', b: '若是 ARRAY：再读 元素类型 + 元素个数', c: 'c' },
      { t: 'value 字节', b: '按类型定长；string 自己带长度', c: 'd' }
    ];
    const kvEls = kv.map(k => {
      const e = U.el('div', { style: 'border-left:3px solid var(--' + k.c + ');background:#10151b;border-radius:5px;padding:4px 7px' });
      e.innerHTML = '<div style="font-size:10.5px;font-weight:620">' + U.esc(k.t) + '</div>' +
        '<div class="cm" style="margin:0">' + U.esc(k.b) + '</div>';
      left.appendChild(e);
      return e;
    });

    const right = wrap.querySelector('#right');
    right.innerHTML = '<div class="cm" style="margin-bottom:3px">能塞进 KV 段的东西（都是同一个机制）</div>';
    const ex = [
      { c: 'a', t: '超参', b: 'llama.block_count / attention.head_count ...' },
      { c: 'b', t: '词表', b: 'tokenizer.ggml.tokens（字符串数组）' },
      { c: 'c', t: '★ 整段 template', b: 'tokenizer.chat.ggml.template：一个长 string，' +
           '读的人不必认识它 —— 这就是"自描述"' },
      { c: 'd', t: '对齐参数', b: 'general.alignment：值本身是元数据，含义却作用于数据区' }
    ];
    const exEls = ex.map(x => { const e = U.card(x, { style: 'width:100%' }); e.style.width = '100%'; right.appendChild(e); return e; });

    kvEls.forEach(e => e.style.opacity = '.35');
    exEls.forEach(e => e.style.opacity = '.35');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先读 key：<span class="v">长度（uint64）+ 字节</span>，然后读 <span class="v">value 的类型</span>（int32）。',
      '<span class="k">类型是写进文件的数字</span>：u8=0 … f64=12。读的人按这个数字决定后面读几字节。',
      '类型不同读法不同：定长类型直接按宽度读；<span class="v">STRING</span> 自己再带一个长度；' +
      '<span class="v">ARRAY</span> 先读元素类型与元素个数，再循环。',
      '这套机制<b>不挑内容</b>：超参、词表、<b>整段 chat template</b> 全走同一条路。',
      '★ 这就是"自描述"：<span class="k">文件自己说明自己有哪些元数据、每个元数据是什么类型</span>。' +
      '新增一个元数据不需要改格式、不需要改旧读者。',
      '一个特例值得记住：<span class="v">general.alignment</span> 也是普通 KV，' +
      '但读完之后它会被提升成 <span class="k">ctx->alignment</span>，直接决定数据区的字节布局。',
      '回顾 <b>L1-04</b>：量化块布局（block_q4_0 / Q4_K…）决定一个"块"占几字节；' +
      '本课讲这些块<b>从第几字节开始</b>。两课合起来才是"权重在文件里长什么样"。'
    ];
    kvEls.forEach((_, i) => tl.at(700 + i * 2500, () => {
      kvEls.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.35'; });
      exEls.forEach(e => { e.style.opacity = '.35'; });
      msg.innerHTML = texts[Math.min(i, 2)];
    }));
    tl.at(8500, () => { kvEls.forEach(e => { e.style.opacity = '1'; });
      exEls[0].style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(11500, () => {
      exEls.forEach((e, k) => { e.style.opacity = k >= 1 ? '1' : '.35'; });
      msg.innerHTML = texts[4]; });
    tl.at(14500, () => { exEls.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
    tl.at(17500, () => { msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 4 ★ 文件布局：<span class="hl-c">头 -> KV 段 -> 张量信息段 -> padding -> 张量数据</span> */
{
  kicker: "L1-06 · 全局",
  title: "★ 文件布局：<span class=\"hl-c\">头 -> KV 段 -> 张量信息段 -> padding -> 张量数据</span>",
  sub: "上游把整个布局按 1..7 条写成了注释。本课其余各幕，就是把这张图逐格讲完。",
  caption: "注释第 7 条的关键词是 \"optional, aligned\"：数据区可以没有（只写元数据），但只要有张量，它就必须对齐。",
  src: "ggml/include/gguf.h",
  mark: [2, 3, 4, 5, 6, 16, 23],
  lineNo: 2,
  code: `// GGUF files have the following structure:
//>> 这段注释就是验收点要你复述的那张图 —— 它与 gguf.cpp 的读写顺序一一对应
//
// 1. File magic "GGUF" (4 bytes).
// 2. File version (uint32_t).
// 3. Number of ggml tensors in file (int64_t).
// 4. Number of key-value-pairs in file (int64_t).
// 5. For each KV pair:
//   1. The key (string).
//   2. The value type (gguf_type).
//   3a. If the value type is GGUF_TYPE_ARRAY:
//     1. The type of the array (gguf_type).
//     2. The number of elements in the array (uint64_t).
//     3. The binary representation of each element in the array.
//   3b. Otherwise:
//     1. The binary representation of the value.
// 6. For each ggml tensor:
//   1. The tensor name (string).
//   2. The number of dimensions of the tensor (uint32_t).
//   3. For each dimension:
//     1. The size of the tensor in the dimension (int64_t).
//   4. The tensor data type (ggml_type).
//   5. The tensor data offset in the tensor data binary blob (uint64_t).
// 7. The tensor data binary blob (optional, aligned).`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="col" id="stack" style="gap:4px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const stack = wrap.querySelector('#stack');
    const blocks = [
      { t: '1. magic "GGUF"', n: '4 B', c: 'a' },
      { t: '2. version', n: '4 B', c: 'a' },
      { t: '3. n_tensors', n: '8 B', c: 'a' },
      { t: '4. n_kv', n: '8 B', c: 'a' },
      { t: '5. KV 段（n_kv 条）', n: '变长', c: 'b' },
      { t: '6. 张量信息段（n_tensors 条）', n: '变长', c: 'c' },
      { t: '（padding）', n: '0..alignment-1 B', c: 'd' },
      { t: '7. 张量数据区（optional）', n: '各张量按 alignment 补齐', c: 'e' }
    ];
    const els = blocks.map(p => {
      const e = U.el('div', { style: 'display:flex;gap:9px;align-items:center;border-left:3px solid var(--' + p.c + ');background:#10151b;border-radius:5px;padding:3px 7px' });
      e.innerHTML = '<span style="font-size:10.5px;flex:1 1 auto">' + U.esc(p.t) + '</span>' +
        '<span class="cm" style="margin:0;flex:0 0 auto">' + U.esc(p.n) + '</span>';
      stack.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '上游注释第 1 条到第 4 条：文件头四个字段，共 24 字节。',
      '第 5 条：<span class="v">KV 段</span>。每条 = key（string）+ value 类型（gguf_type）+ value；数组多带"元素类型 + 元素个数"。',
      '第 6 条：<span class="v">张量信息段</span>。每个张量 = 名字（string）+ 维数（uint32）+ 各维大小（int64）+ 类型（ggml_type）+ 数据区偏移（uint64）。',
      '<span class="v">padding</span>：注释第 24 行那句 "the tensor data binary blob (optional, aligned)" —— ' +
      '数据区起点必须落在 alignment 的整数倍上，中间用 0 补齐。',
      '第 7 条：<span class="v">张量数据区</span>。所有张量的数据拼在一起，每个张量内部再按 alignment 补齐。',
      '★ 这张图就是验收点：<span class="k">头(24B) -> KV 段 -> 张量信息段 -> padding -> 张量数据</span>。' +
      '接下来几幕逐个字段展开。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[Math.min(i, 4)];
    }));
    tl.at(16800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 张量信息段：名字 + 形状 + <span class="hl-e">类型</span> + <span class="hl-e">offset</span> */
{
  kicker: "L1-06 · 张量信息",
  title: "张量信息段：名字 + 形状 + <span class=\"hl-e\">类型</span> + <span class=\"hl-e\">offset</span>",
  sub: "读进来的是一个 gguf_tensor_info：里面已经有一个 ggml_tensor（装信息），外加一个数据区偏移。",
  caption: "offset 是从\"数据区起点\"算起的字节偏移，不是从文件开头。结构体注释写它 must be a multiple of ALIGNMENT。",
  src: "ggml/src/gguf.cpp",
  mark: [1, 3, 6, 9, 10, 19],
  lineNo: 0,
  code: `struct gguf_tensor_info {
//>> struct ggml_tensor 在这里只是"装信息"：形状、类型文件里都有，不必分配数据
    struct ggml_tensor t; // for holding the equivalent info
//>> offset 的单位是字节，基准是数据区起点（ctx->offset），不是文件开头
    uint64_t offset;      // offset from start of \`data\`, must be a multiple of \`ALIGNMENT\`
};
//>> ---- ggml/src/gguf.cpp:652-672 ----
                ok = false;
//>> struct ggml_tensor 在这里只是"装信息"：形状、类型文件里都有，不必分配数据
            }
//>> offset 的单位是字节，基准是数据区起点（ctx->offset），不是文件开头
            if (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = false;
                break;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            for (int64_t j = 0; ok && j < i; ++j) {
                if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                    GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\\n", __func__, info.t.name, j, i);
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '名字', m: 'write(info.t.name)', b: 'string：uint64 长度 + 字节。<br>读时查空名字、重名、长度 < GGML_MAX_NAME。' },
      { c: 'c', t: '形状', m: 'n_dims + ne[j]', b: '维数 uint32，随后每维一个 int64。<br>读时挡住 n_dims > GGML_MAX_DIMS 与负数。' },
      { c: 'e', t: '类型 + offset', m: 'write(info.t.type); write(info.offset);', b: '类型是 int32 的 ggml_type（L1-04 的编号表）；<br>offset 是数据区内的字节位置。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '每条张量信息固定读四样：名字、形状、类型、offset。',
      '<span class="v">名字</span>是 string 编码；读的时候顺带检查 <span class="k">空名字</span>与' +
      '<span class="k">重名</span>（重名会让 gguf_find_tensor 失去意义）。',
      '<span class="v">形状</span>：先读维数，再逐维读 int64 大小。文件里的 ne[] 直接进 ' +
      '<span class="k">info.t.ne[]</span> —— 这就是为什么不必另存一份形状表。',
      '<span class="v">类型</span>：一个 int32 的 <span class="k">ggml_type</span>。' +
      '<b>回顾 L1-04</b>：这个编号决定块大小与每块字节数，所以枚举只能追加、不能重排。',
      '<span class="v">offset</span>：同一段代码里紧接着读 <span class="k">info.offset</span>（uint64）。' +
      '它告诉读的人：这个张量的数据在数据区里的第几字节。',
      '★ 一段信息里同时有"语义"（名字、形状、类型）和"位置"（offset）—— ' +
      '<span class="k">语义在元数据里，字节在数据区里，两者靠 offset 挂钩</span>。'
    ];
    els.forEach((_, i) => tl.at(700 + i * 3100, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[Math.min(i, 2)];
    }));
    tl.at(9800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
    tl.at(13200, () => { msg.innerHTML = texts[4]; });
    tl.at(16200, () => { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 ★ 对齐：<span class="hl-d">GGML_PAD</span> 把数据区推到 alignment 的整数倍 */
{
  kicker: "L1-06 · 核心",
  title: "★ 对齐：<span class=\"hl-d\">GGML_PAD</span> 把数据区推到 alignment 的整数倍",
  sub: "读的时候靠 seek 把文件指针推到对齐位置；写的时候靠 pad 补 0 —— 两边算出同一个数。",
  caption: "GGML_PAD(x, n) = ((x) + (n) - 1) & ~((n) - 1)（定义在 ggml/include/ggml.h），要求 n 是 2 的幂 —— 所以读的时候会检查 alignment 是 2 的幂。",
  src: "ggml/src/gguf.cpp",
  mark: [0, 1, 5, 7, 8, 9, 12],
  lineNo: 773,
  code: `    if (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\\n", __func__);
//>> gr.tell() 是当前位置，gr.start() 是本次读的起点（GGUF 可以不从文件第 0 字节开始）—— 所以对齐的是"相对起点"的偏移
        gguf_free(ctx);
        return nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

//>> 每个张量的期望 offset 必须严格等于累计值：既不能有洞，也不能重叠
    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            size_t padded_size = GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
            if (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\\n",`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px">
        <div class="col grow" id="left" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:2px">算例：元数据读完时，文件指针在 1140 字节处</div>';
    const steps = [
      { t: 'alignment', v: '32', c: 'd' },
      { t: 'GGML_PAD(1140, 32)', v: '1152', c: 'd' },
      { t: 'ctx->offset（数据区起点）', v: '1152', c: 'a' },
      { t: '张量 A：884736 B -> 补齐', v: '884736', c: 'b' },
      { t: '张量 B 的 offset', v: '1152 + 884736 = 885888', c: 'c' }
    ];
    const els = steps.map(s => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
      e.innerHTML = '<span class="m">' + U.esc(s.t) + '</span> = ' +
        '<span style="color:var(--' + s.c + ')">' + U.esc(s.v) + '</span>';
      left.appendChild(e);
      return e;
    });

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--d)">' +
      '<div class="ct" style="color:var(--d)">对齐的是"相对起点"的偏移</div>' +
      '<div class="cb">GGUF 可以嵌在别的文件中间，所以 ' +
      '<span class="cm" style="margin:0">GGML_PAD(gr.tell() - gr.start(), alignment)</span> ' +
      '算的不是文件绝对位置。</div></div>' +
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">连续性是<b>被断言</b>的</div>' +
      '<div class="cb">循环要求 <span class="cm" style="margin:0">ti.offset == ctx->size</span>：' +
      '每个张量的 offset 必须等于前面所有张量补齐后的累计值 —— 数据区由此成为' +
      '<b>无洞、无重叠</b>的连续区间。</div></div>' +
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">对齐让 mmap 成为可能</div>' +
      '<div class="cb">连续 + 起点对齐 => 可以直接把文件映射进内存，' +
      '把每个张量的 data 指到 <span class="cm" style="margin:0">blob + offset</span>。' +
      '这正是 <b>L2-03</b> 讲的权重加载与 mmap。</div></div>';

    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '元数据全部读完之后，文件指针停在某个任意字节上 —— 而数据区要求对齐。',
      '<span class="v">GGML_PAD(当前位置 - 起点, alignment)</span> 把它推到 alignment 的整数倍，' +
      '再用 seek 跳过去 —— 中间的字节不读、也不关心内容。',
      '然后 <span class="k">ctx->offset = gr.tell()</span> 把"数据区从哪开始"记下来（相对文件开头），' +
      '所以调用方知道数据区在文件里的位置。',
      '接着算数据区总长度：对每个张量 <span class="v">padded_size = GGML_PAD(ggml_nbytes(&ti.t), alignment)</span>，' +
      '累加进 <span class="k">ctx->size</span>。',
      '★ 顺序断言：<span class="k">ti.offset != ctx->size</span> 直接报错退出。' +
      '所以"张量数据是一块连续区间"是<b>被强制</b>的，而不是约定俗成。',
      '把读和写对上：写侧（gguf_writer_base::pad）也是"补 0 直到 written_bytes % alignment == 0"。' +
      '<span class="k">两侧算出同一个数，文件才能被对方读回来。</span>',
      '★ 收束：<span class="v">对齐 + 连续</span> 这四个字，就是 L2-03 能用 mmap 把张量直接指向文件的全部原因。'
    ];
    els.forEach((_, i) => tl.at(600 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k <= i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15600, () => { msg.innerHTML = texts[5]; });
    tl.at(19000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 7 写的一侧：<span class="hl-b">gguf_set_tensor_data</span> 只记指针，<span class="hl-b">offset</span> 由前一个张量决定 */
{
  kicker: "L1-06 · 写与读",
  title: "写的一侧：<span class=\"hl-b\">gguf_set_tensor_data</span> 只记指针，<span class=\"hl-b\">offset</span> 由前一个张量决定",
  sub: "写入顺序与读取顺序严格对称：头 -> KV -> 张量信息 -> pad -> 数据。",
  caption: "注意：张量数据并不\"存在 gguf_context 里\" —— context 只保存指针，真正落盘发生在 gguf_write_to_file 的写数据阶段。",
  src: "ggml/src/gguf.cpp",
  mark: [2, 3, 4, 5, 6, 12, 13, 14, 17, 18, 24],
  lineNo: 0,
  code: `void gguf_add_tensor(
             struct gguf_context * ctx,
        const struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
//>> offset 是推导出来的：前一个张量的 offset + 它补齐后的字节数。第一个张量是 0
    if (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    struct gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
//>> set_tensor_data 只写指针，不拷贝数据 —— 调用方必须保证这块内存在写文件时还有效
}

//>> ---- ggml/src/gguf.cpp:1432-1439 ----
void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name);
    if (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
//>> offset 是推导出来的：前一个张量的 offset + 它补齐后的字节数。第一个张量是 0
    }

    ctx->info[tensor_id].t.data = (void *)(uintptr_t)data; // double cast suppresses warning about casting away const
}
//>> ---- ggml/src/gguf.cpp:1634-1654 ----
    // write header
    gw.write(GGUF_MAGIC[0]);
    gw.write(GGUF_MAGIC[1]);
    gw.write(GGUF_MAGIC[2]);
//>> offset 是推导出来的：前一个张量的 offset + 它补齐后的字节数。第一个张量是 0
    gw.write(GGUF_MAGIC[3]);
    gw.write(ctx->version);
    gw.write(n_tensors);
    gw.write(n_kv);

    // write key-value pairs
    for (int64_t i = 0; i < n_kv; ++i) {
        gw.write(ctx->kv[i]);
    }
//>> set_tensor_data 只写指针，不拷贝数据 —— 调用方必须保证这块内存在写文件时还有效

    // write tensor info
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);
//>> ---- ggml/src/gguf.cpp:1660-1665 ----
    const size_t offset_data = gw.written_bytes;

    // write tensor data
    for (int64_t i = 0; i < n_tensors; ++i) {
//>> offset 是推导出来的：前一个张量的 offset + 它补齐后的字节数。第一个张量是 0
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:10px">
        <div class="col grow" id="left" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:2px">写入顺序（gguf_write_out）</div>';
    const ws = [
      { t: 'GGUF_MAGIC[0..3]', c: 'a' },
      { t: 'ctx->version', c: 'a' },
      { t: 'n_tensors', c: 'a' },
      { t: 'n_kv', c: 'a' },
      { t: '每条 KV（gw.write(kv)）', c: 'b' },
      { t: '每条张量信息（write_tensor_meta）', c: 'c' },
      { t: 'gw.pad(ctx->alignment)', c: 'd' },
      { t: '每个张量数据（write_tensor_data）+ pad', c: 'e' }
    ];
    const wEls = ws.map(w => {
      const e = U.el('div', { style: 'border-left:3px solid var(--' + w.c + ');background:#10151b;border-radius:5px;padding:3px 7px;font-family:var(--mono);font-size:9.5px;overflow-wrap:anywhere' });
      e.textContent = w.t;
      left.appendChild(e);
      return e;
    });

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">offset 是累计推导的</div>' +
      '<div class="cb">第一个张量 offset = 0；其后每个 = ' +
      '<span class="cm" style="margin:0">前一个.offset + GGML_PAD(前一个字节数, alignment)</span>。' +
      '改类型时（gguf_set_tensor_type）会立刻重算后续所有 offset，保证数据仍是一整块。</div></div>' +
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">三种写法，同一个布局</div>' +
      '<div class="cb">头文件列了三种：一次写完；<span class="cm" style="margin:0">only_meta=true</span> ' +
      '只写元数据再追加数据；或先留占位再回填元数据。三者产出的字节布局相同。</div></div>' +
      '<div class="card" style="border-left-color:var(--d)">' +
      '<div class="ct" style="color:var(--d)">写失败也要能报出来</div>' +
      '<div class="cb">写路径把 <span class="cm" style="margin:0">fputc / fwrite</span> 的返回值与期望比较，' +
      '不符就抛 <span class="cm" style="margin:0">std::runtime_error</span>；' +
      '<span class="cm" style="margin:0">gguf_write_to_file_ptr</span> 捕获后打日志并返回 false。</div></div>';

    wEls.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '写的一侧从 gguf_add_tensor 开始：把一个 ggml_tensor 的信息塞进 gguf_tensor_info。',
      '<span class="v">offset</span> 不是调用方给的，是 context 推出来的：' +
      '第一个张量从 0 开始，后面每个接在前一个"补齐后"的末尾。',
      '于是<b>数据区连续</b>这件事在写入侧就已经成立了 —— 读侧的 ti.offset == ctx->size 断言只是把它验一遍。',
      '<span class="v">gguf_set_tensor_data</span> 只做一件事：把 <span class="k">info.t.data</span> 指向你给的那块内存。' +
      '<b>不拷贝</b>。',
      '所以真正落盘发生在写文件阶段：<span class="k">write_tensor_data</span> 里按 nbytes 把数据写出去，然后 pad 到 alignment。',
      '回顾 <b>L1-01</b>：ggml_tensor 里 <span class="v">data</span> 只是一个指针 —— ' +
      'GGUF 的写入侧正是利用了这一点：有 buffer 就走后端读取，没有就直接 memcpy。',
      '★ 读写对称：<span class="k">头 -> KV -> 张量信息 -> padding -> 数据</span>。' +
      '读的人按同一顺序走，就能把文件还原成 ggml_context。'
    ];
    wEls.forEach((_, i) => tl.at(600 + i * 2700, () => {
      wEls.forEach((e, k) => { e.style.opacity = k <= i ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i, 4)];
    }));
    tl.at(13000, () => { msg.innerHTML = texts[5]; });
    tl.at(16500, () => { wEls.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 8 把这一课压成一张<span class="hl-a">字节表</span> */
{
  kicker: "L1-06 · 收束",
  title: "把这一课压成一张<span class=\"hl-a\">字节表</span>",
  sub: "从文件第 0 字节到张量数据第一字节，逐段过一遍；表里的每一项都能在源码里找到出处。",
  caption: "下一课 L2-01：llama.h 的公共 API —— 看 llama.cpp 怎么用 gguf_* 这套 API 把模型装起来。",
  src: "ggml/src/gguf.cpp",
  mark: [3, 6, 8, 10],
  lineNo: 0,
  code: `struct gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<struct gguf_kv> kv;
//>> version 是 context 的默认值，读进来后被文件里的值覆盖
    std::vector<struct gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of \`data\` from beginning of file
    size_t size      = 0; // size of \`data\` in bytes
//>> alignment 的默认值来自 GGUF_DEFAULT_ALIGNMENT（32），除非文件里写了 general.alignment

    void * data = nullptr;
//>> data 指向数据区起点；no_alloc 的调用方靠它直接映射文件
};
//>> ---- ggml/src/gguf.cpp:1205-1207 ----

const int64_t * gguf_get_tensor_ne(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['从文件第', '长度', '内容', '谁读它', '在哪一幕'],
      [['0', '4 B', 'magic "GGUF"', '逐字节比对 GGUF_MAGIC', '第 2 幕'],
       ['4', '4 B', 'version（uint32，当前 3）', '版本四道关卡', '第 2 幕'],
       ['8', '8 B', 'n_tensors（int64）', '范围检查 + 张量信息循环', '第 2 幕'],
       ['16', '8 B', 'n_kv（int64）', '范围检查 + KV 循环', '第 2 幕'],
       ['24', '变长', 'KV 段：key + 类型 + 值', 'gguf_get_val_* / general.alignment', '第 3 幕'],
       ['…', '变长', '张量信息段：名 / 形状 / 类型 / offset', '建 ggml_tensor、算数据指针', '第 5 幕'],
       ['…', '0..31 B', 'padding 补 0 到 alignment', 'gguf_get_data_offset', '第 6 幕'],
       ['对齐后', '变长', '张量数据区（连续，逐张量按对齐补齐）', 'blob + info.offset', '第 6 幕']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一个 GGUF 文件：<span class="mono">n_kv = 0</span>、<span class="mono">n_tensors = 1</span>，' +
      '张量名 <span class="mono">"blk.0.weight"</span>（13 字节），形状 <span class="mono">ne = [256, 4, 1, 1]</span>，' +
      '类型 <span class="mono">F32</span>，且文件里没有 <span class="mono">general.alignment</span>。' +
      '算出数据区起点（从文件第 0 字节算起）。',
      '按本课的字节表逐段累加（小端；读元数据时<b>不</b>插入任何对齐，只有数据区之前有一次 padding）：<br>' +
      '<span class="mono">头 = 4 + 4 + 8 + 8 = 24</span>；<br>' +
      '<span class="mono">KV 段 = 0</span>（n_kv = 0）；<br>' +
      '<span class="mono">张量信息 = 8（名字长度）+ 13（名字字节）+ 4（n_dims）+ 4×8（ne[0..3]）' +
      ' + 4（ggml_type = int32）+ 8（offset = uint64）= 69</span>；<br>' +
      '<span class="mono">元数据末尾 = 24 + 69 = 93</span>；<br>' +
      '<span class="mono">alignment = GGUF_DEFAULT_ALIGNMENT = 32</span>，' +
      '<span class="mono">GGML_PAD(93, 32) = 96</span>。<br>' +
      '<b>数据区起点 = 96</b> —— 这就是 <span class="mono">gguf_get_data_offset()</span> 的返回值，' +
      '也是第 6 幕那个 seek 的目标。<br>' +
      '顺带验连续性：这个张量 <span class="mono">nbytes = 256×4×4 = 4096</span>，' +
      '<span class="mono">GGML_PAD(4096, 32) = 4096</span>，所以它的 offset = 0，' +
      '下一个张量（若有）offset = 4096。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '验收点要求"按字节说出布局"。这张表就是答案，八行一句话：',
      '<span class="k">头 24 字节</span>：magic、version、n_tensors、n_kv。任何一样不对就返回 nullptr。',
      '<span class="k">KV 段</span>：自描述的元数据。键是 string，类型是 int32；数组多带元素类型与个数。',
      '<span class="k">张量信息段</span>：每个张量"名字 + 形状 + 类型 + offset"，offset 是数据区内的字节位置。',
      '<span class="k">padding</span>：数据区起点被 GGML_PAD 推到 alignment（默认 32）的整数倍。',
      '<span class="k">数据区</span>：连续、逐张量对齐 —— 于是 L2-03 的 mmap 可以零拷贝把张量指向文件。',
      '★ 一句话：<span class="v">GGUF = 自描述元数据 + 连续且对齐的数据区</span>。' +
      '下一课 L2-01 看 llama.cpp 怎么用这套 API 把模型装起来。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 1900, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(18200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

];
