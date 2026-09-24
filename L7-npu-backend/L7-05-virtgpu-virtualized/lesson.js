/* ==========================================================================
   L7-05 · VirtGPU 后端：虚拟化 GPU
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一台 VM 里的 llama.cpp，GPU 在墙的另一边 */
{
  kicker: "L7 · NPU 与加速器后端",
  title: "一台 VM 里的 llama.cpp，GPU 在墙的另一边",
  sub: "39 个文件分成四族。看源码之前，先记住一句话：数据只能搬过去，不能指过去。",
  caption: "这一课的第一手依据是这个协议头：它自己声明\"本文件的其余部分必须与 virglrenderer 的 apir-protocol.h 一致\"。",
  src: "ggml/src/ggml-virtgpu/backend/shared/api_remoting.h",
  mark: [0, 7, 10, 12, 15],
  lineNo: 3,
  code: `/* the rest of this file must match virglrenderer/src/apir-protocol.h */
//>> 这一行框定了本课的边界：协议的真相在 virglrenderer 那一侧，这里只是镜像

#include <unistd.h>

#include <cstdint>

#define APIR_PROTOCOL_MAJOR 0
#define APIR_PROTOCOL_MINOR 2

#define APIR_HANDSHAKE_MAGIC 0xab1e

enum ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,
//>> 23 条后端命令全部从这一条 FORWARD 走 —— 见第 4 幕

    APIR_COMMAND_TYPE_LENGTH = 3,
};`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip a">客户机 VM：libggml-virtgpu</span><span class="arrow">-></span>
        <span class="chip c">/dev/dri + 共享窗口</span><span class="arrow">-></span>
        <span class="chip b">宿主机：libggml-virtgpu-backend</span>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['族', '文件', '跑在哪', '本课怎么讲'],
      [['客户机侧 frontend', '18', 'VM 里的 ggml-virtgpu 库', '幕 2/5/6/7/9，第七/十三节'],
       ['宿主机侧 backend', '11', 'virglrenderer 加载的库', '幕 3，第六/十二节'],
       ['两侧共用协议', '6', 'backend/shared/，编译进两边', '幕 1/4，第二节'],
       ['通用工具', '4', '客户机侧：窗口 + 稀疏数组', '幕 8，第十四节']]);
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '本课覆盖 39 个文件：<span class="k">ggml/src/ggml-virtgpu/</span> 下 38 个 + 公共头 <span class="k">ggml/include/ggml-virtgpu.h</span>，wc -l 合计 4414 行。',
      '客户机侧 18 个文件：三个 ggml 虚表（device / buffer_type / buffer）+ 一层转发（virtgpu-forward-*）。',
      '宿主机侧 11 个文件：把同一批调用在真后端上重放（backend-dispatched-*）。',
      '两侧共用 6 个头：命令枚举、编码器、tensor 的线格式都在 <span class="k">backend/shared/</span>。',
      '剩下 4 个是通用工具：共享内存窗口（virtgpu-shm）与稀疏数组（virtgpu-utils）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(14000, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = '一句话：<span class="k">数据只能搬过去，不能指过去。</span> 下一幕看这句话在代码里的第一处痕迹。';
    });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">指针退化成偏移</span>：过墙前先减掉 buffer base */
{
  kicker: "L7-05 · 核心",
  title: "★ <span class=\"hl-a\">指针退化成偏移</span>：过墙前先减掉 buffer base",
  sub: "其它后端把 tensor->data 直接交给内核去解引用；这里同一个字段必须先变成一个相对量。",
  caption: "回顾 L1-01：那一课说\"data 就是数据在哪\"的答案。这个前提在虚拟化下第一个失效。",
  src: "ggml/src/ggml-virtgpu/apir_cs_ggml-rpc-front.cpp",
  mark: [0, 3, 6],
  lineNo: 36,
  code: `    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    if (tensor->data) {
        if (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
//>> 减掉的是 buffer 的 base。这不是优化：墙那边的地址空间里，这个数只有"相对"才有意义`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="two" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#two');
    const left = U.card({ c: 'b', t: '其它 16 个后端',
      b: '<b>data 是本机可访问的地址</b>：<br>CPU 的 malloc、CUDA 的显存、Metal 的 buffer，<br>内核拿到它就能解引用。',
      m: 'kernel(gpu_ptr, tensor->data)' }, { style: 'width:330px' });
    const right = U.card({ c: 'a', t: 'ggml-virtgpu',
      b: '<b>data 过墙前要变成偏移</b>：<br>客户机持有的这个值属于宿主机的地址空间，<br>在 VM 里解引用没有意义。',
      m: 'result.data -= buffer_base' }, { style: 'width:330px' });
    host.appendChild(left); host.appendChild(right);
    left.style.opacity = '.35'; right.style.opacity = '.35';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '一切从 <span class="k">apir_serialize_tensor()</span> 开始：把一个 <span class="v">ggml_tensor</span> 变成一个定长 POD。',
      '左边这条路，整个仓库走了 16 遍：<span class="v">tensor->data</span> 就是设备指针，内核直接读。',
      '<span class="k">到了这里不行。</span> 客户机里的 <span class="v">tensor->data</span> 是宿主机地址空间里的一个数值 —— 它指向的内存，客户机没有映射。',
      '于是最后一行把它减掉 buffer 的 base：<span class="v">data - base = 该 tensor 在 buffer 内的字节偏移</span>。这是唯一能过墙的形式。',
      '宿主侧再把偏移加回自己的 base（下一幕）。<span class="k">跨墙传的是"第几字节"，不是"在哪里"。</span>'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    tl.at(3400, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[1]; U.markLines(document, [0]); });
    tl.at(7000, () => { left.style.opacity = '.35'; right.style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [0, 3]); });
    tl.at(10500, () => { msg.innerHTML = texts[3]; U.markLines(document, [0, 3, 6]); });
    tl.at(13500, () => { msg.innerHTML = texts[4]; U.markLines(document, [0, 3, 6]); });
  }
},

/* ------------------------------------------------------ 3 墙的另一边：偏移 + 宿主的 <span class="hl-d">buffer_start</span> = 真指针 */
{
  kicker: "L7-05 · 核心",
  title: "墙的另一边：偏移 + 宿主的 <span class=\"hl-d\">buffer_start</span> = 真指针",
  sub: "宿主机反解出指针，并且顺手做了隔着一堵墙唯一还能做的地址校验。",
  caption: "这两条断言是本课里唯一\"客户机说什么都不能全信\"的地方：偏移越界会在宿主侧被拦下。",
  src: "ggml/src/ggml-virtgpu/backend/apir_cs_ggml-rpc-back.cpp",
  mark: [0, 4, 9, 12, 13],
  lineNo: 42,
  code: `    uint64_t tensor_data = tensor->data;
    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size  = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
//>> 宿主自己的 base：由真后端的 buffer 给出，客户机拿不到它的映射，只能拿到这个数值
        uint64_t buffer_size  = (uint64_t) ggml_backend_buffer_get_size(result->buffer);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_data += buffer_start;
//>> 加回来 —— 这一步之后，tensor->data 才重新是一个可解引用的设备指针

        GGML_ASSERT(tensor_data + tensor_size >= tensor_data);  // check for overflow
        GGML_ASSERT(tensor_data >= buffer_start && tensor_data + tensor_size <= buffer_start + buffer_size);
//>> 同一条断言同时卡住下界与上界：偏移为负、或越过 buffer 末尾，都在这里被拒`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip c">线上：偏移</span><span class="arrow">+</span>
        <span class="chip d">宿主：buffer_start</span><span class="arrow">=</span>
        <span class="chip b">真指针 tensor_data</span>
      </div>
      <div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['第 52 / 53 行的断言', '挡住什么'],
      [['tensor_data + tensor_size >= tensor_data', '无符号加法回绕（越界的第一种伪装）'],
       ['tensor_data      >= buffer_start', '偏移为负 / 落在 buffer 之前'],
       ['tensor_data + tensor_size <= buffer_start + buffer_size', '越过 buffer 末尾']]);
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '解序列化的第一件事：把线格式里的 <span class="v">uint64_t data</span> 取出来。',
      '<span class="k">tensor_size</span> 来自 <span class="v">ggml_nbytes()</span>，<span class="k">buffer_start / buffer_size</span> 来自宿主自己的真 buffer。',
      '第 52 行先做一次溢出检查：<span class="v">a + b >= a</span> 不成立就说明回绕了。',
      '第 53 行再做一次区间检查：偏移必须落在 <span class="v">[buffer_start, buffer_start + buffer_size)</span> 内。',
      '注意校验的是<b>字节区间</b>，不是指针本身 —— 因为跨墙传过来的从来就不是指针。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
    tl.at(3300, () => { msg.innerHTML = texts[1]; U.markLines(document, [0, 4, 9]); });
    rows.forEach((r, i) => tl.at(6200 + i * 2900, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 2];
      U.markLines(document, [0, 4, 9, 12, 13]);
    }));
    tl.at(15000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; U.markLines(document, [0, 4, 9, 12, 13]); });
  }
},

/* ------------------------------------------------------ 4 <span class="hl-c">23 条命令</span>：整个后端就是一张枚举表 */
{
  kicker: "L7-05 · 命令流",
  title: "<span class=\"hl-c\">23 条命令</span>：整个后端就是一张枚举表",
  sub: "ggml 的 device / buffer_type / buffer / backend 四个面，被压成 0..22 号命令。",
  caption: "这个文件是生成的（来源见末幕说明）。它和宿主侧的分发表 backend-dispatched.gen.h 必须逐条对齐。",
  src: "ggml/src/ggml-virtgpu/backend/shared/apir_backend.gen.h",
  mark: [0, 4, 10, 16, 24, 32, 35],
  lineNo: 1,
  code: `typedef enum ApirBackendCommandType {
//>> 协议的另一半在这里：命令号是双方的契约，改一个就要两边一起改

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_COUNT        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_MEMORY       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE   = 12,
    APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER   = 14,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE = 15,

    /* buffer */
    APIR_COMMAND_TYPE_BUFFER_GET_BASE    = 16,
    APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = 17,
    APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_BUFFER_CLEAR       = 20,
    APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_TABLE_COUNT = 23,
//>> COUNT = 最后一个命令号 + 1；宿主侧用它做下标越界的上界检查
} ApirBackendCommandType;`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'device · 0-9', b: '设备名、类型、显存、<br>supports_op、buffer_from_ptr', m: '10 条' },
      { c: 'b', t: 'buffer-type · 10-15', b: '名字、对齐、上限、<br>分配 buffer、算 alloc_size', m: '6 条' },
      { c: 'c', t: 'buffer · 16-21', b: '数据面：get_base、<br>set/get/cpy_tensor、clear', m: '6 条' },
      { c: 'd', t: 'backend · 22', b: '执行：<br>graph_compute', m: '1 条' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '四个面共 <span class="k">23 条命令</span>（0..22 号）+ 一个计数常量：先按面看它们的分工。',
      '<span class="v">device</span> 组最厚：10 条。设备的每一项能力都要问一遍宿主，答案在墙那边。',
      '<span class="v">buffer-type</span> 组 6 条。第 13 号 <span class="k">is_host</span> 已废弃，但编号保留、槽位保留。',
      '<span class="v">buffer</span> 组 6 条，这一组就是数据面：<span class="k">SET_TENSOR / GET_TENSOR</span> 是全部数据搬运入口。',
      '<span class="v">backend</span> 组只有 1 条：<span class="k">GRAPH_COMPUTE</span>。整张图一次性过去，不是逐算子往返。',
      '所以"命令流"并不细碎：<span class="v">一次建 buffer、一次传数据、一次算整图</span>。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    els.forEach((_, i) => tl.at(2900 + i * 2700, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(14800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 5 一条命令的固定前缀：<span class="hl-b">类型 / 标志 / 回复窗口</span> */
{
  kicker: "L7-05 · 命令流",
  title: "一条命令的固定前缀：<span class=\"hl-b\">类型 / 标志 / 回复窗口</span>",
  sub: "所有命令共用同一个编码器，前三段是固定的 —— 第三条告诉宿主把回复写到哪里。",
  caption: "编码缓冲区是 thread_local 的 4KiB 栈数组：命令流不为此分配堆内存。",
  src: "ggml/src/ggml-virtgpu/virtgpu.cpp",
  mark: [5, 23, 30, 31, 33, 35],
  lineNo: 378,
  code: `apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local char encoder_buffer[4096];
//>> thread_local + 栈数组：每个线程一个 4KiB 编码缓冲，不碰堆

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_buffer,
        .start = encoder_buffer,
        .end   = encoder_buffer + sizeof(encoder_buffer),
        .fatal = false,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    if (!gpu->use_apir_capset) {
//>> 过渡期的兼容开关：老 capset 下命令号整体平移 331（见 virtgpu.h 的 VENUS_COMMAND_TYPE_LENGTH）
        cmd_type += VENUS_COMMAND_TYPE_LENGTH;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t reply_res_id = gpu->reply_shmem.res_id;
//>> 第一条业务参数永远是 reply 窗口的 res_id —— 宿主靠它知道该往哪块内存写回复
    apir_encode_uint32_t(&enc, &reply_res_id);

    return &enc;`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="stream"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'int32 cmd_type', b: 'FORWARD / HANDSHAKE /<br>LOADLIBRARY 三选一', m: 'apir_encode_int32_t' },
      { c: 'b', t: 'int32 cmd_flags', b: '转发层的标志位；<br>业务命令这里传 0', m: 'apir_encode_int32_t' },
      { c: 'c', t: 'uint32 reply_res_id', b: '回复窗口的 res_id：<br>宿主往这块内存写结果', m: 'gpu->reply_shmem.res_id' },
      { c: 'd', t: '业务参数…', b: '各命令自己的字段，<br>按 apir_cs.h 的编码器追加', m: 'apir_encode_*' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const stream = wrap.querySelector('#stream');
    stream.innerHTML = '<span class="m">字节序：</span>' +
      '<span style="color:var(--a)">[cmd_type]</span>' +
      '<span style="color:var(--b)">[cmd_flags]</span>' +
      '<span style="color:var(--c)">[reply_res_id]</span>' +
      '<span style="color:var(--d)">[arg0][arg1]…</span>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '一次远程调用 = 往一个 4KiB 缓冲里按顺序写字段，然后把这块缓冲交给 ioctl。',
      '<span class="v">cmd_type</span>：只有三个值。业务命令全部包在 <span class="k">FORWARD</span> 里，具体是第几号命令放在 <span class="v">cmd_flags</span>。',
      '<span class="v">cmd_flags</span>：转发层的标志位，各业务命令自己传 0。',
      '<span class="v">reply_res_id</span>：<span class="k">这是同步机制的关键</span> —— 它把"回复写哪块内存"写进了命令本身。',
      '此后每个业务参数都由 <span class="v">apir_cs.h</span> 里的编码器逐字段追加；头部固定 12 字节，从不变。',
      '所以一条命令 = <span class="v">12 字节公共头 + 各命令自己的参数</span>。下一幕看这块缓冲怎么交给内核。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    const mLines = [[23, 30], [23, 30, 31], [23, 30, 31, 33, 35], [23, 30, 31, 33, 35]];
    els.forEach((_, i) => tl.at(3000 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
      U.markLines(document, mLines[Math.min(i, 3)]);
    }));
    tl.at(15200, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[5];
      U.markLines(document, [5, 23, 30, 31, 33, 35]);
    });
  }
},

/* ------------------------------------------------------ 6 提交与等待：<span class="hl-e">零个 fence</span>，一个 atomic 计数器 */
{
  kicker: "L7-05 · 同步",
  title: "提交与等待：<span class=\"hl-e\">零个 fence</span>，一个 atomic 计数器",
  sub: "execbuffer 的三个同步字段全是 0；回复靠轮询 reply 窗口的第一个 4 字节。",
  caption: "这解释了为什么 ggml_backend_i 里 synchronize / event_record / event_wait 三个槽位在这个后端上全是 NULL。",
  src: "ggml/src/ggml-virtgpu/virtgpu.cpp",
  mark: [0, 15, 19, 20, 27, 32, 34, 38, 40, 48],
  lineNo: 0,
  code: `    volatile std::atomic_uint * atomic_reply_notif = (volatile std::atomic_uint *) gpu->reply_shmem.mmap_ptr;
    *atomic_reply_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer args = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .num_bo_handles = 0,

        .fence_fd         = 0,
//>> fence_fd = 0：不注册 fence，也不要求内核回一个 fd
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .num_in_syncobjs  = 0,
        .num_out_syncobjs = 0,
        .in_syncobjs      = 0,
        .out_syncobjs     = 0,
    };

    *decoder = NULL;

    int ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &args);
//>> ---- ggml/src/ggml-virtgpu/virtgpu.cpp:487-510 ----
    bool     timedout    = false;
    uint32_t notif_value = 0;
    while (true) {
        notif_value = std::atomic_load_explicit(atomic_reply_notif, std::memory_order_acquire);

        if (notif_value != 0) {
            break;
        }

        int64_t base_sleep_us = 15;

        os_time_sleep(base_sleep_us);

        if (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            long long end_time    = (long long) ts_end.tv_sec * 1000000000LL + ts_end.tv_nsec;
//>> fence_fd = 0：不注册 fence，也不要求内核回一个 fd
            float     duration_ms = (end_time - start_time) / 1000000;

            if (duration_ms > max_wait_ms) {
                timedout = true;
                break;
            }
        }
    }`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row" id="top" style="gap:9px">
        <div class="col grow" id="fields"></div>
        <div class="col grow" id="poll"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    wrap.querySelector('#fields').innerHTML =
      '<div class="cm" style="margin-bottom:4px">drm_virtgpu_execbuffer 的同步字段</div>' +
      ['fence_fd = 0', 'num_in_syncobjs = 0', 'num_out_syncobjs = 0', 'in_syncobjs = 0', 'out_syncobjs = 0']
        .map(s => '<div class="formula" style="padding:4px 8px;font-size:10px"><span class="v">' + s + '</span></div>')
        .join('');

    const poll = wrap.querySelector('#poll');
    poll.innerHTML = '<div class="cm" style="margin-bottom:4px">reply 窗口的第一个 4 字节</div>' +
      '<div class="formula" style="padding:5px 8px;font-size:11px">' +
      '<span class="m">notif_value</span> = <span id="cval" style="color:var(--e)">0</span></div>' +
      '<div class="card" style="border-left-color:var(--f);margin-top:5px">' +
      '<div class="cb">读到非 0 就跳出循环。源码注释写明：<b>extract the actual return value from the notif flag</b>，<br>' +
      '<span class="cm" style="margin:0">returned_value = notif_value - 1</span></div></div>';

    const cval = poll.querySelector('#cval');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '客户机把命令准备好，然后 <span class="k">drmIoctl(DRM_IOCTL_VIRTGPU_EXECBUFFER)</span> 把命令交给内核 → hypervisor → 宿主。',
      '同步字段全为 0：<span class="v">fence_fd</span>、<span class="v">num_in_syncobjs</span>、<span class="v">num_out_syncobjs</span> 都不参与。',
      'context init 阶段是同一个选择：<span class="v">VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK</span> 的值写 0，注释写明"不要在 fence 信号时产生 drm_events"。',
      '等待发生在 <span class="k">reply_shmem.mmap_ptr</span> 的第一个 4 字节上：它是一个 <span class="v">atomic_uint</span>，用 acquire 语义加载。',
      '没到就 <span class="v">os_time_sleep(15µs)</span> 再来一次；带超时的调用（如握手）还有一个毫秒级上限。',
      '醒来后：<span class="v">returned_value = notif_value - 1</span>，回复正文紧接着那 4 字节，交给解码器继续读。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 27]); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [0, 15, 19, 20, 27]); });
    tl.at(6800, () => { msg.innerHTML = texts[2]; U.markLines(document, [0, 15, 19, 20, 27]); });
    const seq = [0, 0, 0, 0, 7];
    seq.forEach((v, i) => tl.at(9800 + i * 1400, () => {
      cval.textContent = String(v);
      msg.innerHTML = texts[3];
      U.markLines(document, [0, 15, 19, 20, 27, 32, 34]);
    }));
    tl.at(17200, () => { msg.innerHTML = texts[4]; U.markLines(document, [0, 15, 19, 20, 27, 32, 34, 38, 40]); });
    tl.at(19800, () => { msg.innerHTML = texts[5]; U.markLines(document, [0, 15, 19, 20, 27, 32, 34, 38, 40, 48]); });
  }
},

/* ------------------------------------------------------ 7 ★ <span class="hl-c">数据只能搬过去</span>：一次 memcpy + 一条命令 */
{
  kicker: "L7-05 · 数据面",
  title: "★ <span class=\"hl-c\">数据只能搬过去</span>：一次 memcpy + 一条命令",
  sub: "set_tensor 的全部动作：挑窗口、上锁、把字节拷进客户机自己 mmap 到的内存、把 res_id 放进命令。",
  caption: "宿主侧的对应实现是 backend-dispatched-buffer.cpp 的 backend_buffer_set_tensor：解出 res_id → get_shmem_ptr → buffer->iface.set_tensor。",
  src: "ggml/src/ggml-virtgpu/virtgpu-forward-buffer.cpp",
  mark: [0, 9, 12, 17, 23, 25, 29],
  lineNo: 32,
  code: `    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_SET_TENSOR);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   temp_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &temp_shmem;
    bool            using_shared_shmem = false;

    if (size <= gpu->data_shmem.mmap_size) {
//>> 小于 24MiB 走常驻窗口 data_shmem；更大的临时建一个专属 blob
        // Lock mutex before using shared data_shmem buffer
        if (mtx_lock(&gpu->data_shmem_mutex) != thrd_success) {
//>> 常驻窗口是所有线程共用的，用之前必须先上锁
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = true;
        shmem              = &gpu->data_shmem;

    } else if (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    memcpy(shmem->mmap_ptr, data, size);
//>> 数据本身的搬运就在这一行：一次 memcpy，写进客户机自己 mmap 到的窗口
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
//>> 命令里带的是窗口的 res_id，不是指针 —— 宿主用同一个 id 换回它那一侧的地址

    apir_encode_size_t(encoder, &offset);
    apir_encode_size_t(encoder, &size);`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
    wrap.innerHTML = `
      <div class="row" id="steps" style="gap:7px"></div>
      <div class="row" id="side" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const steps = wrap.querySelector('#steps');
    const defs = [
      { c: 'a', t: '1 编头部', b: '命令号 + buffer 句柄 + tensor' },
      { c: 'b', t: '2 挑窗口', b: '小数据走常驻窗口，<br>大数据现建 blob' },
      { c: 'c', t: '3 上锁', b: '常驻窗口是共享的' },
      { c: 'd', t: '4 搬运', b: 'memcpy 进客户机的<br> mmap 指针' },
      { c: 'e', t: '5 发命令', b: 'res_id + offset + size' },
      { c: 'f', t: '6 收尾', b: '解锁 / 销毁临时窗口' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:109px' }); steps.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');

    const side = wrap.querySelector('#side');
    const g = U.card({ c: 'c', t: '客户机这一侧写下的是字节', b: '<span class="cm" style="margin:0">memcpy(shmem->mmap_ptr, data, size)</span>', m: 'virtgpu-forward-buffer.cpp:53' }, { style: 'width:336px' });
    const h = U.card({ c: 'b', t: '宿主这一侧换回的是指针', b: '<span class="cm" style="margin:0">get_shmem_ptr(ctx_id, shmem_res_id)</span>', m: 'backend-dispatched-buffer.cpp:64' }, { style: 'width:336px' });
    side.appendChild(g); side.appendChild(h);
    g.style.opacity = '.35'; h.style.opacity = '.35';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '对照 L4-03 的 buffer 契约：<span class="v">set_tensor(buffer, tensor, data, offset, size)</span> 的五个参数原样保留，只是实现换成了过墙。',
      '第 41 行先算大小：<span class="v">size <= data_shmem.mmap_size</span> 就直接复用常驻窗口。',
      '不复用的情况（第 49 行）现建一个 <span class="v">virtgpu_shmem</span>，用完在收尾阶段销毁。',
      '<span class="k">第 53 行是全部数据搬运</span>：一次 memcpy，目标是自己 mmap 出来的窗口地址 —— 在客户机里它是普通内存。',
      '第 54 行命令里带 <span class="v">res_id</span>；宿主侧第 64 行用同一个 id 拿回它那一侧的指针，第 71 行交给真后端写进设备内存。',
      '所以链路上没有任何一处"把设备内存映射进客户机"。<span class="k">搬的是字节，不是映射。</span>'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, [9]); });
    tl.at(3300, () => { msg.innerHTML = texts[1]; els.forEach((e, k) => { e.style.opacity = k <= 1 ? '1' : '.28'; }); U.markLines(document, [9]); });
    tl.at(6300, () => { msg.innerHTML = texts[2]; els.forEach((e, k) => { e.style.opacity = k === 2 ? '1' : '.28'; }); U.markLines(document, [9, 12, 17]); });
    tl.at(9600, () => { msg.innerHTML = texts[3]; els.forEach((e, k) => { e.style.opacity = k === 3 ? '1' : '.28'; }); g.style.opacity = '1'; U.markLines(document, [9, 12, 17, 23]); });
    tl.at(13200, () => { msg.innerHTML = texts[4]; els.forEach((e, k) => { e.style.opacity = k >= 4 ? '1' : '.28'; }); h.style.opacity = '1'; U.markLines(document, [9, 12, 17, 23, 25]); });
    tl.at(17800, () => { msg.innerHTML = texts[5]; els.forEach(e => { e.style.opacity = '1'; }); g.style.opacity = '1'; h.style.opacity = '1'; U.markLines(document, [0, 9, 12, 17, 23, 25, 29]); });
  }
},

/* ------------------------------------------------------ 8 窗口是怎么造出来的：一个 <span class="hl-b">HOST3D blob</span> + 一次 mmap */
{
  kicker: "L7-05 · 数据面",
  title: "窗口是怎么造出来的：一个 <span class=\"hl-b\">HOST3D blob</span> + 一次 mmap",
  sub: "客户机向 DRM 申请一块可映射的 blob，拿到 res_id 与一个本地指针；res_id 就是这块内存在墙那边的名字。",
  caption: "宿主侧声明自己需要的能力就是两个回调：get_config 与 get_shmem_ptr —— 后者按 res_id 换回宿主指针。",
  src: "ggml/src/ggml-virtgpu/virtgpu-shm.cpp",
  mark: [1, 4, 6, 12, 20, 23],
  lineNo: 75,
  code: `int virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
//>> HOST3D + USE_MAPPABLE：内存由宿主的 GPU 栈分配，但客户机可以映射它
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    if (!gem_handle) {
        return 1;
    }

    void * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
//>> 映射出来的指针落在客户机自己的地址空间里 —— 这就是"窗口"两个字的全部意思
    if (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\\n", __func__);
        return 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_size  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    return 0;`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div id="sz" class="row" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '1 对齐', b: 'size 先对齐到<br>16384 字节', m: 'align64(size, 16384)' },
      { c: 'b', t: '2 建 blob', b: 'RESOURCE_CREATE_BLOB<br>→ res_id + gem_handle', m: 'virtgpu_ioctl_resource_create_blob' },
      { c: 'c', t: '3 导出偏移', b: 'DRM_IOCTL_VIRTGPU_MAP<br>换一个 mmap 用的 offset', m: 'virtgpu_ioctl_map' },
      { c: 'd', t: '4 映射', b: 'mmap(PROT_READ|PROT_WRITE,<br>MAP_SHARED)', m: 'void * ptr = mmap(...)' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    wrap.querySelector('#sz').innerHTML =
      '<div class="card" style="width:336px;border-left-color:var(--e)">' +
      '<div class="ct" style="color:var(--e)">常驻数据窗口</div>' +
      '<div class="cb">SHMEM_DATA_SIZE = 0x1830000（源码注释写 24MiB）。' +
      '所有 set/get_tensor 默认都走它，所以每次用之前要上锁。</div></div>' +
      '<div class="card" style="width:336px;border-left-color:var(--f)">' +
      '<div class="ct" style="color:var(--f)">常驻回复窗口</div>' +
      '<div class="cb">SHMEM_REPLY_SIZE = 0x4000（16KiB）。' +
      '第一个 4 字节是 atomic 计数器，之后是回复正文。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '四个步骤全部在客户机侧完成：<span class="k">建窗口这件事本身不需要宿主配合</span>。',
      '<span class="v">align64(size, 16384)</span> 先对齐，之后才建 blob。',
      '<span class="k">VIRTGPU_BLOB_MEM_HOST3D</span>：内存归宿主的 GPU 栈管；<span class="k">VIRTGPU_BLOB_FLAG_USE_MAPPABLE</span>：客户机可以映射它。',
      '<span class="v">mmap</span> 出来的指针落在客户机自己的地址空间里 —— 这就是"窗口"两个字的全部意思。',
      '第 96 行把 <span class="v">gem_handle</span> 也存下来：销毁时用 <span class="k">DRM_IOCTL_GEM_CLOSE</span> 归还。',
      '最后回到宿主侧：<span class="v">virgl_apir_callbacks</span> 只有两个函数，其中 <span class="k">get_shmem_ptr(ctx_id, res_id)</span> 就是"按名字换指针"。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    els.forEach((_, i) => tl.at(2900 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(15400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 墙的两侧，各管一段 */
{
  kicker: "L7-05 · 收束",
  title: "墙的两侧，各管一段",
  sub: "客户机侧实现 ggml 的契约并把调用翻译成命令；宿主侧把命令翻译回 ggml 的调用。",
  caption: "下一课 L7-06 是小后端合集（RPC 也是\"把远端当设备\"）。对照 L7-02 Hexagon：计划里那一课的要点是 host 与 DSP 的边界，而这里隔的是虚拟化边界。",
  src: "ggml/include/ggml-virtgpu.h",
  mark: [9],
  lineNo: 1,
  code: `#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_virtgpu_reg();
//>> 客户机侧对外暴露的全部 API 就这一行：注册一个 backend。其余全在墙后面

#ifdef  __cplusplus
}
#endif`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['环节', '客户机侧（VM 里）', '宿主侧（hypervisor）', '源码依据'],
      [['设备能力', '缓存 name / type / memory', '向真后端查询并回传', 'ggml-backend-reg.cpp:34-58'],
       ['buffer 分配', '只发命令，拿回一个句柄', '真后端 alloc_buffer（记入跟踪表）', 'backend-dispatched-buffer-type.cpp:60-79'],
       ['数据写入', 'memcpy 进共享窗口', 'get_shmem_ptr 换指针后写设备', 'virtgpu-forward-buffer.cpp:53 / backend-dispatched-buffer.cpp:64,71'],
       ['图执行', '整图序列化后一次发走', '反序列化后交给真后端 graph_compute', 'virtgpu-forward-backend.cpp:17 / backend-dispatched-backend.cpp:56,93'],
       ['同步', '轮询 reply 窗口的 atomic', '真后端是异步的则补一次 synchronize', 'virtgpu.cpp:490 / backend-dispatched-backend.cpp:95-97'],
       ['地址', '只持有偏移与句柄', '持有真指针与真 buffer', 'apir_cs_ggml-rpc-front.cpp:42 / …-back.cpp:50']],
      { monoCols: [] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '这台 VM 里的 llama.cpp 能不能直接把宿主机 GPU 的显存映射进来，让 <span class="mono">tensor->data</span> 指过去？' +
      '不能的话，一个 tensor 的数据是怎么到宿主 GPU 的？',
      '不能。三条理由，都在代码里：<br>' +
      '① <span class="mono">apir_buffer_get_base()</span> 送回客户机的是一个 <span class="mono">uintptr_t</span>（宿主的地址），' +
      '它在 VM 的地址空间里没有映射，解引用就是野指针；<br>' +
      '② 所以 <span class="mono">tensor->data</span> 过墙前被减去 buffer base，变成 <b>buffer 内偏移</b>' +
      '（<span class="mono">apir_cs_ggml-rpc-front.cpp:42</span>），宿主再加回自己的 <span class="mono">buffer_start</span> 并断言没越界' +
      '（<span class="mono">apir_cs_ggml-rpc-back.cpp:50-53</span>）；<br>' +
      '③ 字节本身走共享内存窗口：客户机建一个 <span class="mono">HOST3D + USE_MAPPABLE</span> 的 blob 并 mmap 它' +
      '（<span class="mono">virtgpu-shm.cpp:79,86</span>），把数据 memcpy 进去（<span class="mono">virtgpu-forward-buffer.cpp:53</span>），' +
      '命令里只带 <span class="mono">res_id + offset + size</span>（<span class="mono">:54-57</span>），' +
      '宿主用 res_id 换回自己那一侧的指针（<span class="mono">backend-dispatched-buffer.cpp:64</span>）再写进真 buffer（<span class="mono">:71</span>）。<br>' +
      '一句话：<b>搬的是字节，不是映射；传的是偏移，不是指针。</b>'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '按"谁做这件事"把六条链路列出来，每一条都能在源码里指到行。',
      '<span class="k">设备能力</span>：客户机缓存一次，之后不再往返 —— 这是它少有的"不每次过墙"的地方。',
      '<span class="k">buffer 分配</span>：客户机只有一个句柄，宿主侧才真正持有 buffer 对象（还在跟踪表里）。',
      '<span class="k">数据写入</span>：客户机做的是 memcpy，宿主做的是"换指针 + 写设备"。',
      '<span class="k">图执行</span>：一次往返算整张图，命令流并不细碎。',
      '<span class="k">同步</span>：客户机这一侧只有一个 atomic 计数器；宿主只在真后端异步时补一次 synchronize。',
      '记住验收点：<span class="v">设备内存在宿主机上，客户机拿不到指针</span>，所以只能"共享内存窗口 + 命令流"。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 6)];
    }));
    tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

];
