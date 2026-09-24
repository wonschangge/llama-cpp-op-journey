/* ==========================================================================
   L4-03 · buffer 与 buffer type
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 张量脚下那块内存：<span class="hl-a">buft</span> 是厂，<span class="hl-b">buffer</span> 是地 */
{
  kicker: "L4 · 内存与调度",
  title: "张量脚下那块内存：<span class=\"hl-a\">buft</span> 是厂，<span class=\"hl-b\">buffer</span> 是地",
  sub: "buffer type 回答\"这类内存怎么造、有什么属性\"；buffer 回答\"就是这一块\"。",
  caption: "回顾 L1-01：tensor 的 buffer / data 两个字段，指的就是本课这两个对象。",
  src: "ggml/include/ggml-backend.h",
  mark: [4, 5, 6, 7, 8, 9, 10],
  lineNo: 33,
  code: `    //
    // Backend buffer type
    //

    GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer  (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API size_t                ggml_backend_buft_get_alignment (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_max_size  (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    GGML_API bool                  ggml_backend_buft_is_host       (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">模型 / 中间结果</span><span class="arrow">-&gt;</span>
        <span class="chip a">buft（类型）</span><span class="arrow">-&gt;</span>
        <span class="chip b">buffer（一块内存）</span><span class="arrow">-&gt;</span>
        <span class="chip c">tensor.buffer / data</span>
      </div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'buft · buffer type', b: '这类内存<b>怎么造</b>、有什么属性：<br>对齐、上限、是不是 host', m: 'ggml_backend_buffer_type_t' },
      { c: 'b', t: 'buffer · 一块内存', b: '已经分配好的那一块：<br>基址、大小、用途', m: 'ggml_backend_buffer_t' },
      { c: 'c', t: 'tensor 的两个字段', b: '回顾 L1-01：<b>buffer</b> 指向这一块，<br><b>data</b> 是里面的地址', m: 'tensor->buffer / tensor->data' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先分清两个词：<span class="k">type</span> 是"哪一类内存"，<span class="k">buffer</span> 是"这一块内存"。',
      '回顾 L1-01：每个 <span class="v">ggml_tensor</span> 都有 <span class="v">buffer</span> 与 <span class="v">data</span> 两个字段 —— 它们就是本课的主角。',
      '回顾 L3-01：这七个公共函数就是 <span class="v">buffer_type_i</span> 虚表的镜像。这里看它们的<b>实现与语义</b>。',
      '本课路线：七个函数 → buft 与 buffer 的绑定 → 数据面 → host / device 分工 → <span class="k">pinned host buffer</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(700 + 4 * 3400, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      U.markLines(document, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      msg.innerHTML = texts[3];
    });
  }
},

/* ------------------------------------------------------ 2 buft 的<span class="hl-a">三个必需函数</span>：名字、分配、对齐 */
{
  kicker: "L4-03 · 契约",
  title: "buft 的<span class=\"hl-a\">三个必需函数</span>：名字、分配、对齐",
  sub: "公共 API 全是虚表转发：断言 buft 非空，然后调 buft->iface —— 一行都没有业务逻辑。",
  caption: "七个函数的全景在右边的表里；这一段是它们的实现在 ggml/src/ggml-backend.cpp:34-52。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 6, 8, 10, 16],
  lineNo: 34,
  code: `const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}
//>> get_name：必需。日志与报错都用它标识这块内存（"CPU" / "CUDA0"）

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}
//>> alloc_buffer：必需，也是这一层唯一的产出 —— 把"字节数"变成一块 buffer

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}
//>> get_alignment：必需，没有默认值 —— 分配器必须知道对齐要求（L4-01）
`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['公共函数', '作用', '必需性'],
      [['ggml_backend_buft_name', '这块内存类型叫什么', '必需'],
       ['ggml_backend_buft_alloc_buffer', '把字节数变成一块 buffer', '必需'],
       ['ggml_backend_buft_get_alignment', '这张量数据的对齐要求', '必需'],
       ['ggml_backend_buft_get_max_size', '单块 buffer 的尺寸上限', '可选：默认 SIZE_MAX'],
       ['ggml_backend_buft_get_alloc_size', '一个张量实际要占多少字节', '可选：默认 ggml_nbytes'],
       ['ggml_backend_buft_is_host', '这块内存 CPU 能不能直接读写', '可选：默认 false'],
       ['ggml_backend_buft_get_device', '这个 buft 属于哪个设备', '字段直读']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const rows = Array.from(t.body.querySelectorAll('tr'));
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这是 buft 的全部公共 API —— 与 L3-01 的 <span class="v">buffer_type_i</span> 虚表一一对应。',
      '<span class="v">ggml_backend_buft_name</span>：必需。日志与报错都用它标识这块内存。',
      '<span class="v">ggml_backend_buft_alloc_buffer</span>：必需，也是这一层唯一的产出。第 41-44 行是零尺寸分支：返回一个哑 buffer，不真分配。',
      '<span class="v">ggml_backend_buft_get_alignment</span>：必需，同样没有默认值 —— 对齐是分配器的硬前提（L4-01 用它给每个节点补齐）。',
      '剩下四项：三个可选（下一幕看它们的默认值），<span class="v">get_device</span> 只是一行字段直读。'
    ];
    const steps = [[0], [6, 8, 10], [16]];
    steps.forEach((st, i) => tl.at(700 + i * 3400, () => {
      rows.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; });
      U.markLines(document, st);
      msg.innerHTML = texts[i];
    }));
    tl.at(700 + 3 * 3400, () => {
      rows.forEach((r, k) => { r.className = (k >= 3) ? 'on' : ''; });
      U.markLines(document, [0, 6, 16]);
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 3 三个可选函数：<span class="hl-c">默认值也是契约</span> */
{
  kicker: "L4-03 · 契约",
  title: "三个可选函数：<span class=\"hl-c\">默认值也是契约</span>",
  sub: "get_max_size 默认 SIZE_MAX、get_alloc_size 默认 ggml_nbytes、is_host 默认 false。",
  caption: "这三个默认值决定了后端能少写多少代码；也决定了分配器必须用哪个函数算尺寸。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 3, 5, 7, 10, 13, 15, 16, 20, 23, 24, 29, 32, 34, 38],
  lineNo: 53,
  code: `size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
//>> get_max_size：可选 —— 先判空再调；iface 没填就直接返回 SIZE_MAX
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
//>> get_alloc_size：可选 —— iface 没填就用 ggml_nbytes(tensor)
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
//>> 断言：返回值不能小于 ggml_nbytes，但【可以更大】
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));
//>> TAG_ALLOC_SIZE_EXPAND：量化类型与"可能扩张"的算子允许分配尺寸更大

        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
//>> is_host：可选 —— 同样先判空；没填一律按 false 处理
        return buft->iface.is_host(buft);
    }
    return false;
//>> 所以"没实现 is_host"不等于"这是 host 内存"，而是被当成 device buffer
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="cards" style="gap:11px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'get_max_size', b: '不填 <span class="cm" style="margin:0">SIZE_MAX</span><br>单块 buffer 不限大小', m: 'if (buft->iface.get_max_size)' },
      { c: 'b', t: 'get_alloc_size', b: '不填 <span class="cm" style="margin:0">ggml_nbytes</span><br>但允许返回<b>更大</b>的值', m: 'if (buft->iface.get_alloc_size)' },
      { c: 'c', t: 'is_host', b: '不填 <span class="cm" style="margin:0">false</span><br>缺省 = 当成 device buffer', m: 'if (buft->iface.is_host)' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '三个可选项各自有安全默认值 —— 这是后端可以"少填"的原因。',
      '<span class="v">get_max_size</span>：先判空再调（第 56-58 行），没填就是 <span class="v">SIZE_MAX</span>，注释把默认值写在旁边。',
      '<span class="v">get_alloc_size</span>：默认 <span class="v">ggml_nbytes</span>，但断言明确允许返回<b>更大</b>的值 —— 量化类型、repack、以及带 <span class="v">TAG_ALLOC_SIZE_EXPAND</span> 标记的算子都要额外空间。',
      '<span class="v">is_host</span>：同样先判空，没填就 <span class="v">return false</span> —— 缺省语义是"这不是 host 内存"。',
      '所以任何分配路径都必须用 <span class="v">get_alloc_size</span>，不能拿 <span class="v">ggml_nbytes</span> 顶替。'
    ];
    const groups = [[53, 55, 56, 57, 59], [62, 65, 66, 67, 71, 73, 74, 78], [81, 83, 86]];
    groups.forEach((g, i) => tl.at(700 + i * 4300, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      U.markLines(document, i === 0 ? [0, 3, 5, 7] : (i === 1 ? [10, 13, 20, 23, 24, 29] : [32, 34, 38]));
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(700 + 3 * 4300, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      U.markLines(document, [3, 13, 20, 24, 34]);
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-b">buffer</span> 只是 <span class="hl-a">buft</span> 的句柄 */
{
  kicker: "L4-03 · 绑定",
  title: "★ <span class=\"hl-b\">buffer</span> 只是 <span class=\"hl-a\">buft</span> 的句柄",
  sub: "buffer 里只存了虚表、buft 指针、context、size、usage；它的属性全部要回问 buft。",
  caption: "对比 L3-01：那里看虚表长什么样，这里看虚表怎么被一个具体对象持有。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [1, 2, 7, 8, 9, 10, 17, 21, 25, 29, 34],
  lineNo: 0,
  code: `ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}
//>> ---- ggml/src/ggml-backend.cpp:171-185 ----
size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}
//>> ---- ggml/src/ggml-backend.cpp:204-207 ----
ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:3px">struct ggml_backend_buffer 的五个字段</div>';
    [['iface', '这块 buffer 的虚表：读写、清空、释放都在这'],
     ['buft', '它属于哪一类内存 —— 唯一记录'],
     ['context', '后端私有数据（例如显存的指针）'],
     ['size', '这块内存多少字节'],
     ['usage', 'ANY / WEIGHTS / COMPUTE']].forEach(f => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
      e.innerHTML = '<span class="m">' + U.esc(f[0]) + '</span> — ' + f[1];
      left.appendChild(e);
    });

    const right = wrap.querySelector('#right');
    const t = U.table(
      ['buffer 侧 API', '转发目标'],
      [['ggml_backend_buffer_get_alignment', 'buft_get_alignment'],
       ['ggml_backend_buffer_get_max_size', 'buft_get_max_size'],
       ['ggml_backend_buffer_get_alloc_size', 'buft_get_alloc_size'],
       ['ggml_backend_buffer_is_host', 'buft_is_host'],
       ['ggml_backend_buffer_get_type', '不转发：返回 buffer->buft']],
      { monoCols: [0] });
    right.appendChild(t.el);
    const rows = Array.from(t.body.querySelectorAll('tr'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看 <span class="v">ggml_backend_buffer_init</span>：它接收 buft，把它原样存进 <span class="v">.buft</span>，再记下 size 与 usage。',
      '所以 buffer = 虚表 + <b>buft 指针</b> + context + size + usage。它自己<b>不存</b>对齐、上限、host 属性。',
      '证据在回程：<span class="v">ggml_backend_buffer_get_alignment</span> 只有一行 —— 先取 <span class="v">buffer-&gt;buft</span>，再问它。',
      '四个查询全是这个形状：<span class="v">max_size</span> / <span class="v">alloc_size</span> / <span class="v">is_host</span> 都一样 —— 同一个 buft 造出来的 buffer，答案必然相同。',
      '唯一不转发的是 <span class="v">ggml_backend_buffer_get_type</span>：直接返回 <span class="v">buffer-&gt;buft</span>。有了它，拿到 buffer 的人才能反查 buft。',
      '回顾 L4-02：调度器的 <span class="v">backend_from_buffer()</span> 读的就是这个回程指针 —— 先知道"哪类内存"，再决定"哪个后端能读它"。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 7, 10]); });
    tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, [8, 9, 10]); });
    const pairs = [[17], [21], [25, 29], [34]];
    pairs.forEach((st, i) => tl.at(8000 + i * 2800, () => {
      rows.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 2];
      U.markLines(document, st);
    }));
    tl.at(8000 + 4 * 2800 + 900, () => {
      rows.forEach(r => { r.className = ''; });
      msg.innerHTML = texts[5];
      U.markLines(document, [17, 21, 25, 29, 34]);
    });
  }
},

/* ------------------------------------------------------ 5 数据的进出：<span class="hl-d">set</span> / <span class="hl-e">get</span> / <span class="hl-f">memset</span> 都先解析 <span class="v">tensor-&gt;buffer</span> */
{
  kicker: "L4-03 · 数据面",
  title: "数据的进出：<span class=\"hl-d\">set</span> / <span class=\"hl-e\">get</span> / <span class=\"hl-f\">memset</span> 都先解析 <span class=\"v\">tensor-&gt;buffer</span>",
  sub: "三个函数的开头一行完全相同：有 view_src 就用源张量的 buffer，否则用自己的。",
  caption: "回顾 L1-01：view_src 非空表示\"数据在别人那里\"—— 这里就是它影响读写的时刻。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [1, 2, 3, 10, 12, 17, 18, 20, 25, 27, 32, 38, 41, 43],
  lineNo: 0,
  code: `void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}
//>> ---- ggml/src/ggml-backend.cpp:350-363 ----
void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}
//>> ---- ggml/src/ggml-backend.cpp:409-423 ----
void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">tensor</span><span class="arrow">-&gt;</span>
        <span class="chip a">view_src ? view_src-&gt;buffer : tensor-&gt;buffer</span><span class="arrow">-&gt;</span>
        <span class="chip b">buffer-&gt;iface.*</span>
      </div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'd', t: 'ggml_backend_tensor_set', b: '写入：把 host 内存的数据<br>写进张量所在的 buffer', m: 'iface.set_tensor' },
      { c: 'e', t: 'ggml_backend_tensor_get', b: '读出：从张量所在的 buffer<br>拷回 host 内存', m: 'iface.get_tensor' },
      { c: 'f', t: 'ggml_backend_tensor_memset', b: '填充：要求后端实现<br>memset_tensor，否则断言失败', m: 'iface.memset_tensor' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '三个函数，同一个开头：<span class="v">tensor-&gt;view_src ? tensor-&gt;view_src-&gt;buffer : tensor-&gt;buffer</span>。',
      '回顾 L1-01：<span class="v">view_src</span> 非空 = 这个张量不拥有数据。所以读写要去找<b>源张量</b>的 buffer。',
      '<span class="v">set</span> 还要过越界断言（<span class="v">offset + size</span> 不能超过 <span class="v">ggml_nbytes</span>），然后转发给 buffer 虚表的 <span class="v">set_tensor</span>。',
      '<span class="v">get</span> 是 set 的镜像：同样的解析、同样的越界检查，只是方向相反。',
      '<span class="v">memset</span> 多一条断言：<span class="v">buf-&gt;iface.memset_tensor != NULL</span> —— 填充能力<b>不是每个 buffer 都有</b>。',
      '一句话：<span class="k">张量的读写从来不是直接 memcpy，而是"找到 buffer，再让 buffer 自己搬"</span>。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
      U.markLines(document, i === 0 ? [1, 2, 3, 12] : (i === 1 ? [17, 18, 25, 27] : [32, 38, 41, 43]));
    }));
    tl.at(700 + 3 * 3400, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[5];
      U.markLines(document, [2, 12, 27, 41]);
    });
  }
},

/* ------------------------------------------------------ 6 <span class="hl-e">host buffer</span> 与 <span class="hl-f">device buffer</span> 的分工 */
{
  kicker: "L4-03 · 分工",
  title: "<span class=\"hl-e\">host buffer</span> 与 <span class=\"hl-f\">device buffer</span> 的分工",
  sub: "跨 buffer 搬张量时，is_host 决定\"谁来搬\"：host 侧用 set / get，device 之间才走后端 cpy_tensor。",
  caption: "回顾 L2-03：权重从磁盘到显存正是这条路 —— host 侧读出来、set 进去。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [1, 2, 3, 9, 15, 16, 17, 18, 19, 21, 24, 25, 26, 31, 32, 37, 45],
  lineNo: 0,
  code: `bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}
//>> ---- ggml/src/ggml-backend.cpp:488-509 ----
void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}
//>> ---- ggml/src/ggml-backend.cpp:2464-2478 ----
static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,`,
  duration: 28000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="formula" style="text-align:center">
        判据只有一个：<span class="m">ggml_backend_buffer_is_host(buffer)</span>
        <span class="arrow">-&gt;</span> <span class="m">buft-&gt;iface.is_host(buft)</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '① src 在 host', b: '直接 <span class="cm" style="margin:0">set</span> 进 dst：读的一端是 CPU 能碰的内存', m: 'ggml_backend_tensor_set' },
      { c: 'b', t: '② dst 在 host', b: '直接从 src <span class="cm" style="margin:0">get</span> 出来：写的一端是 CPU 能碰的内存', m: 'ggml_backend_tensor_get' },
      { c: 'c', t: '③ 两边都不是 host', b: '先试目的 buffer 的 <span class="cm" style="margin:0">cpy_tensor</span>；没有就 malloc 中转', m: 'ggml_backend_buffer_copy_tensor' },
      { c: 'd', t: 'host buffer 是什么', b: 'CPU 的默认 buft 里 <span class="cm" style="margin:0">is_host</span> 返回 true —— 它就是 system memory', m: 'ggml_backend_cpu_buffer_type_is_host' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">ggml_backend_tensor_copy</span> 只有 20 行，却是 host / device 分工的全部判据。',
      '<span class="v">ggml_backend_buffer_copy_tensor</span> 取的是 <b>dst</b> 的 <span class="v">cpy_tensor</span>（视图则取 view_src 的）—— 能力由目的地提供。',
      '<span class="k">src 在 host</span>：直接 set 到 dst。这正是 L2-03 权重上传的形状。',
      '<span class="k">dst 在 host</span>：直接从 src get 出来 —— 例如把显存里的结果取回 CPU。',
      '两边都不在 host：先试后端的 <span class="v">cpy_tensor</span>；连它都没有就 malloc 一块内存中转，调试版会打印 <span class="v">slow copy</span> 警告。',
      '回到定义：CPU 的 buft 里 <span class="v">is_host</span> 直接 <span class="v">return true</span>（第 2464-2468 行）—— 因为它的 buffer 就是 system memory。',
      '所以 host 与 device 的差别，说到底就是 <span class="k">"CPU 能不能直接解引用这块内存"</span>；也只有 host 侧能当跨设备搬运的跳板。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [1, 2, 9]); });
    tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, [1, 3]); });
    defs.forEach((_, i) => tl.at(7600 + i * 3600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i + 2];
      U.markLines(document, i === 0 ? [15, 16] : (i === 1 ? [17, 18] : (i === 2 ? [3, 19, 21, 24, 25, 26] : [31, 32, 37, 45])));
    }));
    tl.at(7600 + 4 * 3600, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[6];
      U.markLines(document, [1, 3, 15, 17, 19, 24, 32]);
    });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-g">手工装配</span>：宿主内存 → buffer → 张量的 data */
{
  kicker: "L4-03 · 装配",
  title: "<span class=\"hl-g\">手工装配</span>：宿主内存 → buffer → 张量的 data",
  sub: "设备能给出三种 buffer；把已经存在的地址装成张量，靠的是 tensor_alloc。",
  caption: "回顾 L2-03：mmap 权重落位走的就是 buffer_from_host_ptr + tensor_alloc 这两步。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 3, 4, 9, 11, 14, 16, 20, 25, 28, 35, 36, 40, 45, 50, 51, 52],
  lineNo: 0,
  code: `enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}
//>> ---- ggml/src/ggml-backend.cpp:615-632 ----
ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}
//>> ---- ggml/src/ggml-backend.cpp:2122-2147 ----
enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}`,
  duration: 28000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">mmap 的地址</span><span class="arrow">-&gt;</span>
        <span class="chip a">buffer_from_host_ptr</span><span class="arrow">-&gt;</span>
        <span class="chip b">buffer</span><span class="arrow">-&gt;</span>
        <span class="chip c">tensor_alloc</span><span class="arrow">-&gt;</span>
        <span class="chip d">tensor-&gt;data</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '设备的默认 buft', b: '设备内存（例如显存）；后端层的分配入口都指向它', m: 'dev_buffer_type' },
      { c: 'c', t: '设备的 host buft', b: '宿主内存；设备可以不提供 —— 没实现就返回 NULL', m: 'dev_host_buffer_type' },
      { c: 'g', t: '宿主指针包成 buffer', b: '地址已经在手上（mmap / 页缓存），直接包一块 buffer', m: 'dev_buffer_from_host_ptr' },
      { c: 'd', t: '挂成张量', b: 'tensor_alloc 把张量钉到 buffer 内的某个地址上', m: 'ggml_backend_tensor_alloc' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '设备的三个 buffer 入口：默认（设备内存）、host（宿主内存，可为 NULL）、把宿主指针包成 buffer。',
      '<span class="v">ggml_backend_dev_host_buffer_type</span> 只有 8 行，关键是那句 <span class="v">if (... == NULL) return NULL</span>：这个能力是<b>可选</b>的。',
      '<span class="v">ggml_backend_dev_buffer_from_host_ptr</span>：宿主内存已经在手 —— 不为它再分配，直接包成 buffer（mmap 走这条）。',
      '<span class="v">ggml_backend_tensor_alloc(buffer, tensor, addr)</span>：把已有地址挂成张量的 data，最后调用 <span class="v">init_tensor</span>。',
      '四条断言把前提钉死：tensor 必须还没有 buffer / data / view_src，且 addr 必须落在 buffer 区间内（meta buffer 除外）。',
      '对照 <span class="v">ggml_backend_view_init</span>：视图不分配 —— 直接借用源张量的 buffer，data = 源 data + view_offs。',
      '回顾 L2-03：mmap 的权重就是这样落位的 —— 先把映射的地址包成 buffer，再把每个张量钉到各自偏移上。'
    ];
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; U.markLines(document, [9, 11]); });
    tl.at(4200, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [14, 16, 20]); });
    tl.at(7700, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [25]); });
    tl.at(11200, () => { els[3].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [0, 4, 40, 50, 51, 52]); });
    tl.at(15200, () => { msg.innerHTML = texts[4]; U.markLines(document, [45, 46]); });
    tl.at(19200, () => { msg.innerHTML = texts[5]; U.markLines(document, [28, 35, 36]); });
    tl.at(23200, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[6];
      U.markLines(document, [11, 16, 25, 4, 50, 35]);
    });
  }
},

/* ------------------------------------------------------ 8 ★ <span class="hl-a">pinned host buffer</span> 到底用在什么场景 */
{
  kicker: "L4-03 · 验收点",
  title: "★ <span class=\"hl-a\">pinned host buffer</span> 到底用在什么场景",
  sub: "不用 mmap 加载模型时：为了异步上传权重到显存，先把权重读进设备的 pinned host buffer。",
  caption: "这一段在 src/llama-model-loader.cpp:1530-1578（load_all_data 的 upload_backend lambda）；该文件的逐字展开见 L2-03。",
  src: "src/llama-model-loader.cpp",
  mark: [0, 2, 9, 11, 18, 25, 27, 34, 42, 45, 54, 56],
  lineNo: 1530,
  code: `        // When not using mmaped io use async uploads from pinned memory to GPU memory.
//>> 前提：use_mmap 与 check_tensors 都为假 —— 否则直接返回 nullptr
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
//>> 从一块已有 buffer 反查 buft、再反查 device —— 第 2/4 幕的 API 在这里被用上
        auto * dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\\n", func,
                ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
//>> 只有"默认 buft"才走异步上传：host buft 自己不算
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\\n", func,
                ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
//>> 三个能力位一起查：async + host_buffer + events
            LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
//>> 向设备要它的 host buft —— 就是第 7 幕的 dev_host_buffer_type
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
//>> 注释直接写明：create pinned memory buffers
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
//>> 用同一个 buft_alloc_buffer 分配 pinned 缓冲（不是 malloc）

            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
//>> 缓冲留下来备用：host_buffers
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));
//>> buffer_get_base 拿到基址：文件内容就读进这块固定内存`,
  duration: 32000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">GGUF 文件</span><span class="arrow">-&gt;</span>
        <span class="chip a">pinned host buffer</span><span class="arrow">-&gt;</span>
        <span class="chip b">显存 buffer</span><span class="arrow">-&gt;</span>
        <span class="chip c">图算起来</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '什么时候走这条路', b: '不用 mmap、也不校验张量时（第 1527 行的门槛）：权重必须从文件读进来', m: 'use_mmap == false' },
      { c: 'c', t: '三个能力位一起查', b: 'async + host_buffer + events，缺一个就退回同步加载', m: 'props.caps.*' },
      { c: 'b', t: '中转站', b: '向设备要 host buft，分配 pinned 内存并配一个事件', m: 'dev_host_buffer_type' },
      { c: 'd', t: '怎么用掉', b: '文件读进 pinned 内存，再 set_async 上传到显存', m: 'ggml_backend_tensor_set_async' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先给答案：<b>不用 mmap 加载模型时</b>，权重要从文件读进显存 —— 中间那块中转内存就是 pinned host buffer。',
      '为什么要 pinned：只有固定（不换页）的宿主内存才能被设备直接 DMA，异步上传才有意义。',
      '判据是三个能力位：<span class="v">caps.async</span> + <span class="v">caps.host_buffer</span> + <span class="v">caps.events</span>（第 1554 行）—— <span class="v">host_buffer</span> 就是 pinned 的开关。',
      '拿到 <span class="v">host_buft</span> 之后，用的是<b>同一个</b> <span class="v">ggml_backend_buft_alloc_buffer</span> —— pinned 内存与显存走同一套 API。',
      '分配完立刻拿基址（第 1578 行）；紧接着还给每块缓冲配一个事件（第 1580-1587 行，见 source.md 第十一节），复用前先等上一次上传结束。',
      '上传在 1727 行：<span class="v">ggml_backend_tensor_set_async</span>。一句话：<span class="k">pinned host buffer = 非 mmap 路径下的异步上传中转站</span>。'
    ];
    tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; U.markLines(document, [0, 2, 9, 11]); });
    tl.at(5200, () => { msg.innerHTML = texts[1]; U.markLines(document, [27]); });
    tl.at(9200, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [25, 34]); });
    tl.at(13400, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [42, 45, 54, 56]); });
    tl.at(17600, () => { els[3].style.opacity = '1'; msg.innerHTML = texts[4]; U.markLines(document, [18, 34, 45]); });
    tl.at(21800, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[5];
      U.markLines(document, [27, 34, 45, 56]);
    });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L4-03 · 收束",
  title: "把这一课压成一张表",
  sub: "buft 七个函数、buffer 一套句柄、host / device 一条判据 —— 最后都收在\"默认 buft\"这一个入口上。",
  caption: "下一课 L4-04：图执行入口 ggml_backend_graph_compute 与异步。",
  src: "ggml/src/ggml-backend.cpp",
  mark: [0, 2, 6, 7, 11, 12, 16, 17],
  lineNo: 248,
  code: `ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
//>> 后端层的入口：默认 buft = 设备的 buft
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
//>> ggml_backend_alloc_buffer = 默认 buft 的 alloc_buffer
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
//>> 对齐与最大尺寸也走默认 buft —— 这两个值正是分配器（L4-01）要的
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
//>> 所以"给后端分配内存"这条路，从头到尾不需要知道 buft 是什么
}`,
  duration: 26000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['环节', '走哪个函数', '本课第几幕', '谁在调用'],
      [['造一块内存', 'ggml_backend_buft_alloc_buffer', '2', '模型加载、分配器、调度器'],
       ['问属性', 'get_alignment / get_max_size / get_alloc_size / is_host', '2 / 3', '分配器 L4-01、拷贝路由'],
       ['读写数据', 'tensor_set / tensor_get / tensor_memset', '5', '权重上传、KV cache、输出'],
       ['跨内存搬运', 'is_host + cpy_tensor', '6', '调度器的跨设备副本 L4-02'],
       ['挂上已有内存', 'buffer_from_host_ptr + tensor_alloc', '7', 'mmap 权重落位 L2-03'],
       ['异步中转', 'dev_host_buffer_type（pinned）', '8', '非 mmap 权重上传']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码回答：<b>pinned host buffer 在什么场景下被用到</b>？为什么必须是 pinned？',
      '场景：<b>不用 mmap 加载权重</b>时，权重需要从文件读进显存 —— 中间那块中转内存就是 pinned host buffer。<br>'
      + '证据链（<span class="mono">src/llama-model-loader.cpp</span> 的 upload_backend lambda）：<br>'
      + '1) 门槛：<span class="mono">use_mmap || check_tensors</span> 为真就直接返回（1527 行）；<br>'
      + '2) 能力检查：<span class="mono">props.caps.async &amp;&amp; props.caps.host_buffer &amp;&amp; props.caps.events</span>（1554 行）；<br>'
      + '3) 取设备的 host buft：<span class="mono">ggml_backend_dev_host_buffer_type(dev)</span>（1560 行）；<br>'
      + '4) 分配 pinned 缓冲：<span class="mono">ggml_backend_buft_alloc_buffer(host_buft, buffer_size)</span>（1569 行），基址来自 <span class="mono">ggml_backend_buffer_get_base</span>（1578 行）；<br>'
      + '5) 配事件：<span class="mono">ggml_backend_event_new</span>（1580 行），复用前先 <span class="mono">ggml_backend_event_synchronize</span>（1706 行）；<br>'
      + '6) 真正的上传：<span class="mono">ggml_backend_tensor_set_async</span>（1727 行）。<br>'
      + '为什么 pinned：只有固定不换页的宿主内存才能被设备直接 DMA，读文件与上传显存才能重叠。<br>'
      + '对照：走 mmap 时不需要它 —— 数据本来就在页缓存里，<span class="mono">buffer_from_host_ptr</span> 直接把映射包成 buffer（L2-03）。'));

    const msg = wrap.querySelector('#msg');
    const rows = Array.from(t.body.querySelectorAll('tr'));
    const texts = [
      '回头看最后一组函数：后端层的四个便捷入口。',
      '<span class="v">ggml_backend_get_default_buffer_type</span> 就是"问设备要它的默认 buft"。',
      '<span class="v">ggml_backend_alloc_buffer</span> = 默认 buft 的 <span class="v">alloc_buffer</span>：调用方连 buft 都不用拿。',
      '<span class="v">get_alignment</span> / <span class="v">get_max_size</span> 也走默认 buft —— 这两个值正是 L4-01 的分配器要的：用 alignment 补齐节点、一旦超过 max_size 就换一块 buffer。',
      '所以整条链是：<span class="v">backend -&gt; device -&gt; 默认 buft -&gt; buffer -&gt; tensor.data</span>。',
      '记住一句话：<span class="k">buft 是"哪类内存"的契约，buffer 是"这一块内存"的句柄；host 与 device 的分工只由一个 is_host 决定</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(2900, () => { msg.innerHTML = texts[1]; U.markLines(document, [0, 2]); rows[0].className = 'on'; });
    tl.at(6400, () => { msg.innerHTML = texts[2]; U.markLines(document, [6, 7]); rows[0].className = ''; rows[1].className = 'on'; });
    tl.at(10400, () => { msg.innerHTML = texts[3]; U.markLines(document, [11, 12]); rows[1].className = ''; rows[2].className = 'on'; });
    tl.at(15400, () => { msg.innerHTML = texts[4]; U.markLines(document, [16, 17]); rows[2].className = ''; rows[3].className = 'on'; rows[4].className = 'on'; rows[5].className = 'on'; });
    tl.at(21000, () => {
      rows.forEach(r => { r.className = ''; });
      U.markLines(document, [2, 7, 12, 17]);
      msg.innerHTML = texts[5];
    });
  }
},

];
