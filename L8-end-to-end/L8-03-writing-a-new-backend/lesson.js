/* ==========================================================================
   L8-03 · 端到端：新后端要做什么
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 最后一课：把七层压成<span class="hl-a">一份清单</span> */
{
  kicker: "L8 · 端到端",
  title: "最后一课：把七层压成<span class=\"hl-a\">一份清单</span>",
  sub: "新增一个后端要回答四个问题：接口函数写哪些、哪些可以留空、必须导出什么符号、必须跑通哪些测试。",
  caption: "本课是整套课件的收束：L1-L8 的每一层都会在这里被引用一次；末幕给一张\"从模型到后端\"的总图。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [1, 4, 20],
  lineNo: 1,
  code: `#include "ggml.h"
#include "ggml-impl.h"
//>> 内部工具头：GGML_ASSERT / GGML_LOG_* 在这里（后端实现者不必自己造轮子）
#include "ggml-blas.h"
#include "ggml-backend-impl.h"
//>> 契约头：五张虚表的定义全在 ggml-backend-impl.h —— 这就是 L3-01 讲的那份契约

#include <future>
#include <vector>
#include <cstring>

#if defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
//>> BLAS 库的选择在编译期决定；device 的 description 字符串用同一组宏（第 329-341 行）
#endif`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">模型定义</span><span class="arrow">-></span>
        <span class="chip a">计算图</span><span class="arrow">-></span>
        <span class="chip c">调度器切分</span><span class="arrow">-></span>
        <span class="chip b">后端虚表</span><span class="arrow">-></span>
        <span class="chip d">内核</span>
      </div>
      <div class="row wrap" id="qs" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '这一课回答什么', b: '把 L1-L7 的每一层收成一份<b>可勾选的清单</b>：<br>新增一个后端，必须写哪些函数、必须导出什么符号、必须跑通哪些测试。', m: 'ggml-backend-impl.h' },
      { c: 'b', t: '证据：最小的真实后端', b: 'BLAS 后端只有 <b>1 个文件 / 530 行</b>，却是一个能跑 MUL_MAT 的完整后端；<br>它的规模就是"最少要写多少"的答案。', m: 'ggml-blas/ggml-blas.cpp' },
      { c: 'c', t: '跨课呼应', b: 'L3-01 三张虚表 · L3-02 注册 · L3-03 动态加载符号<br>L4-02 supports_op 切分 · L4-03 buffer type · L8-01 链路终点', m: 'L3-01 / L3-02 / L3-03 / L4-02 / L4-03 / L8-01' }
    ];
    const host = wrap.querySelector('#qs');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看一个后端的<b>第一屏</b>：四个 include。第 4 行就是契约头 —— <span class="k">写后端 = 填虚表</span>。',
      '这一课不再讲"某一层怎么工作"，而是回答：<span class="v">如果我要加一个新后端，最少要做什么？</span>',
      '答案的<b>上限</b>由契约给出（L3-01 数的 23 个必需项），<b>下限</b>由 BLAS 给出 —— 它只填了 17 个指针。',
      '四个问题：接口函数（必需 / 可留空）、导出符号（注册与动态加载）、验证步骤（跑哪些现成工具）。<br>下面逐段展开，最后压成一张清单。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(11200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">不是从零开始</span>：BLAS 后端就是活证据 */
{
  kicker: "L8-03 · 核心",
  title: "★ <span class=\"hl-a\">不是从零开始</span>：BLAS 后端就是活证据",
  sub: "530 行里只有三张虚表被填；内存层的 17 个指针一行都没写 —— 它直接借了 CPU 的 buffer type。",
  caption: "对照：ggml-zendnn 也只有 1 个文件（838 行），虚表形状与 BLAS 完全一样。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [0, 2, 14, 21],
  lineNo: 262,
  code: `static struct ggml_backend_i blas_backend_i = {
//>> 计算层 16 个函数指针，这里只有 3 个非 NULL：get_name / free / graph_compute
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
//>> graph_compute 是唯一带实现逻辑的指针（第 226-260 行）—— 后端真正干活的地方
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_blas_guid(void) {
//>> guid：后端的唯一标识，ggml_backend_is_blas() 靠它判断"这是不是我"（第 309-311 行）
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:12px">
        <div class="col grow" id="bars" style="gap:5px"></div>
        <div class="col grow" id="side" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#bars');
    host.innerHTML = '<div class="cm" style="margin-bottom:4px">BLAS 填了几个函数指针（槽位 / 非 NULL）</div>';
    const rows = [
      { l: 'device_i  (15)',   v: 10, c: 'a' },
      { l: 'backend_i  (16)',  v: 3,  c: 'b' },
      { l: 'reg_i  (4)',       v: 4,  c: 'c' },
      { l: 'buffer_type_i  (6)', v: 0, c: 'g' },
      { l: 'buffer_i  (11)',   v: 0,  c: 'g' }
    ];
    const bars = rows.map(r => { const b = U.bar(r.l, r.c); host.appendChild(b.el); return b; });
    const side = wrap.querySelector('#side');
    side.innerHTML =
      '<div class="card" style="border-left-color:var(--b)">' +
      '<div class="ct" style="color:var(--b)">530 行 · 1 个文件</div>' +
      '<div class="cb">这是一个<b>能跑 MUL_MAT 的完整后端</b>的全部规模。<br>' +
      '对照 ggml-cuda：277 个 .cu/.cuh、45718 行。</div></div>' +
      '<div class="card" style="border-left-color:var(--g)">' +
      '<div class="ct" style="color:var(--g)">内存层一行都没写</div>' +
      '<div class="cb">全文出现 <span class="cm" style="margin:0">ggml_backend_buffer_i</span> / ' +
      '<span class="cm" style="margin:0">buffer_type_i</span> 的次数是 <b>0</b>；<br>' +
      'get_buffer_type 直接返回 <span class="cm" style="margin:0">ggml_backend_cpu_buffer_type()</span>（第 382 行）。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先数一遍：BLAS 一共填了 <span class="v">3 + 10 + 4 = 17</span> 个非 NULL 指针。',
      '<span class="k">device_i 10 / 15</span>：自我介绍 5 个 + init_backend + get_buffer_type + buffer_from_host_ptr + supports_op + supports_buft。',
      '<span class="k">backend_i 3 / 16</span>：计算层只需要 get_name / free / graph_compute —— 其余 13 个全是 (optional)。',
      '<span class="k">reg_i 4 / 4</span>：注册层只有 4 个槽位，全填。',
      '<span class="k">内存层 0 / 17</span>：buffer_type_i 与 buffer_i 一个函数指针都没写 —— ' +
      '它把 get_buffer_type 接到 CPU 的 buffer type 上（L4-03 的 CPU buffer 家族）。',
      '所以"写一个新后端"的真实工作量 = <span class="v">三张虚表 + 一个导出宏</span>，' +
      '不是把 L3-01 的 23 个必需项全写一遍。'
    ];
    bars.forEach((b, i) => tl.at(700 + i * 2900, () => {
      b.fill.style.width = (rows[i].v / 16 * 100) + '%';
      b.val.textContent = rows[i].v;
      bars.forEach((x, k) => { x.el.style.opacity = k === i ? '1' : '.35'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15200, () => {
      bars.forEach(x => { x.el.style.opacity = '1'; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 3 三张虚表逐张看：<span class="hl-b">device_i</span> 是最大的一张 */
{
  kicker: "L8-03 · 接口面",
  title: "三张虚表逐张看：<span class=\"hl-b\">device_i</span> 是最大的一张",
  sub: "设备层 10 个、计算层 3 个、注册层 4 个；内存层与数据面直接沿用 CPU 的实现。",
  caption: "L3-01 数过：五张虚表共 52 个函数指针，其中 23 个必需。本幕看 BLAS 实际填了哪些。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [0, 8, 11, 13, 15],
  lineNo: 456,
  code: `static const struct ggml_backend_device_i ggml_backend_blas_device_i = {
//>> device_i：15 个槽位，这里 10 个非 NULL —— 三张表里填得最多的一张
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
//>> get_buffer_type 返回 CPU 的 buffer type（第 381-385 行）：内存层就此"借"出去了
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
//>> buffer_from_host_ptr 是 (optional)，BLAS 填了它 —— mmap 加载的模型可以直接用
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ NULL,
//>> offload_op 留空：不主动抢"权重不在自己 buffer 里"的算子（L4-02 的归属优先级）
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['虚表', 'BLAS 里的位置', '槽位', '非 NULL', '它负责什么'],
      [['ggml_backend_device_i', '456-472', '15', '10', '发现与创建：我是谁、能不能跑'],
       ['ggml_backend_i', '262-279', '16', '3', '计算执行：get_name / free / graph_compute'],
       ['ggml_backend_reg_i', '513-518', '4', '4', '注册：名字 / 设备数 / 取设备 / 扩展入口'],
       ['buffer_type_i + buffer_i', '不出现', '17', '0', '内存：整个借 CPU 的 buffer type（L4-03）'],
       ['注册与导出', '520-530', '—', '—', 'ggml_backend_blas_reg() + GGML_BACKEND_DL_IMPL']],
      { monoCols: [0, 1, 2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);
    t.el.style.fontSize = '9.5px';

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '按 L3-01 的分组逐张核对：每张表有多少槽位、BLAS 填了几个。',
      '<span class="k">device_i 10/15</span>：自我介绍 5 + init_backend + get_buffer_type + buffer_from_host_ptr + supports_op + supports_buft。',
      '<span class="k">backend_i 3/16</span>：只有 get_name / free / graph_compute 非 NULL，其余 13 个是 NULL。',
      '<span class="k">reg_i 4/4</span>：get_name / get_device_count / get_device / get_proc_address。',
      '<span class="k">内存层 0/17</span>：17 个槽位一个都没填 —— 它返回 <span class="v">ggml_backend_cpu_buffer_type()</span>。',
      '结论：<span class="v">52 个函数指针里，BLAS 只碰了 17 个</span>；必需项也因此从 23 降到 15。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2700 + i * 2700, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(17200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-c">supports_op</span>：新后端的第一等公民 */
{
  kicker: "L8-03 · 核心",
  title: "★ <span class=\"hl-c\">supports_op</span>：新后端的第一等公民",
  sub: "切分器只问它一个问题：这个算子你能跑吗？说 false 的算子会被留给别的后端 —— 这就是\"回退\"。",
  caption: "回顾 L4-02：调度器对每个节点从上到下问 supports_op + offload_op，第一个说\"能\"的后端拿走它。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [0, 5, 11, 26, 31, 40, 54],
  lineNo: 394,
  code: `static bool ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
//>> supports_op 的签名：给一个算子节点，回答能不能跑 —— 切分的唯一依据（L4-02）
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
//>> 5 个视图算子直接 true：它们不产生计算，任何后端都能"跑"

        case GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 32;
//>> min_batch = 32：矩阵太小的时候 BLAS 反而不如 CPU，主动让出去

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
//>> 带 Hadamard 提示的 MUL_MAT 让给 CPU —— 源码注释指向 issue 25565
                return false;
            }

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
//>> src0 可以是量化类型，只要它的 type traits 有 to_float（第 68-70 行会先反量化）
        }

        case GGML_OP_OUT_PROD:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

        default:
            return false;
//>> default: return false —— 一行就是"我不会，交给别人"

    }

    GGML_UNUSED(dev);
}`,
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['判据（按代码顺序）', '行', '不满足时'],
      [['op 是 5 个视图算子之一：NONE / RESHAPE / VIEW / PERMUTE / TRANSPOSE', '399-404', '继续往下判'],
       ['src0 与 src1 都连续（没有 permute 过）', '426-427', 'false'],
       ['src1 是 F32（激活值）', '428', 'false'],
       ['ne0 / ne1 / ne10 都 >= min_batch = 32', '429', 'false：小矩阵交回 CPU'],
       ['src0 是 F32，或它有 to_float 可反量化', '430', 'false'],
       ['op 参数不是 GGML_HINT_SRC0_IS_HADAMARD', '422', 'false：默认走 CPU 快路径'],
       ['OUT_PROD 另有 6 条判据（矩阵、连续、类型）', '433-440', 'false'],
       ['以上都不匹配', '442-443', 'default: return false']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);
    t.el.style.fontSize = '9.5px';

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      'supports_op 是一串<b>从宽到严</b>的判据。逐条看它什么时候说 true。',
      '视图算子（NONE / RESHAPE / VIEW / PERMUTE / TRANSPOSE）无条件 true —— 它们不需要计算。',
      'MUL_MAT 的硬条件：<span class="v">两个输入都连续</span>。permute 过的张量直接交回 CPU。',
      '再看规模：<span class="v">min_batch = 32</span> 是一道经验阈值（源码注释：BLAS usually is only faster for large matrices）。',
      '<span class="v">src0->type == F32 或 to_float != NULL</span>：量化权重也接，代价是先反量化成 F32（第 67-117 行）。',
      '<span class="v">GGML_HINT_SRC0_IS_HADAMARD</span> 时让给 CPU —— 这是从真实 issue 换来的判据，注释里有链接。',
      'OUT_PROD 是第二个支持的算子，判据更严（还接受转置过的 src1）。',
      '最后一行 <span class="k">default: return false</span>：<b>这句话写下去，回退就自动发生了</b> —— 切分器会把节点给别人。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3000 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(22500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[7]; });
  }
},

/* ------------------------------------------------------ 5 graph_compute 的 default 是 <span class="hl-g">GGML_ABORT</span> */
{
  kicker: "L8-03 · 核心",
  title: "graph_compute 的 default 是 <span class=\"hl-g\">GGML_ABORT</span>",
  sub: "后端自己不做回退：说 false 的算子根本不会进来；进来的算子如果没实现，就直接 abort。",
  caption: "所以 supports_op 与 graph_compute 必须说同一句话 —— 这是新后端最容易踩的坑。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [0, 7, 13, 17, 30],
  lineNo: 226,
  code: `static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
//>> graph_compute：后端唯一的"执行"入口，签名由 ggml_backend_i 定死（L3-01）
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
//>> 没有 GGML_TENSOR_FLAG_COMPUTE 的节点跳过 —— 视图类算子不产生计算（L1-03）
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, node);
                break;

            case GGML_OP_OUT_PROD:
//>> OUT_PROD 分支（实现见第 151-210 行）—— 只支持两个算子也要写清楚
                ggml_backend_blas_out_prod(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\\n", __func__, ggml_op_desc(node));
//>> default 分支是 GGML_ABORT：不认识的算子会直接终止进程，不是悄悄跳过
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:11px">
        <div class="col grow" id="sw" style="gap:6px"></div>
        <div class="col grow" id="side" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const sw = wrap.querySelector('#sw');
    sw.innerHTML = '<div class="cm" style="margin-bottom:4px">graph_compute 里的 switch（第 236-254 行）</div>';
    const cases = [
      { n: 'case GGML_OP_MUL_MAT', v: '-> ggml_backend_blas_mul_mat', c: 'b' },
      { n: 'case GGML_OP_OUT_PROD', v: '-> ggml_backend_blas_out_prod', c: 'b' },
      { n: 'case NONE / RESHAPE / VIEW /', v: 'break（什么都不做）', c: 'c' },
      { n: '     PERMUTE / TRANSPOSE', v: 'break（什么都不做）', c: 'c' },
      { n: 'default', v: 'GGML_ABORT（进程终止）', c: 'g' }
    ];
    const els = cases.map(cc => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
      e.innerHTML = '<span class="m">' + U.esc(cc.n) + '</span> <span style="color:var(--' + cc.c + ')">' +
        U.esc(cc.v) + '</span>';
      sw.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.30');

    const side = wrap.querySelector('#side');
    side.innerHTML =
      '<div class="card" style="border-left-color:var(--g)">' +
      '<div class="ct" style="color:var(--g)">回退不在这里发生</div>' +
      '<div class="cb">BLAS 没有"遇到不会的算子就转给 CPU"的代码 —— 它的 default 是 <b>abort</b>。' +
      '保证不越界的是<b>切分阶段</b>：L4-02 里调度器按 supports_op 把不支持的节点留给下一个后端。</div></div>' +
      '<div class="card" style="border-left-color:var(--a)">' +
      '<div class="ct" style="color:var(--a)">契约一致性</div>' +
      '<div class="cb">supports_op 说 true 的算子，graph_compute 必须真的实现；<br>' +
      '否则第一次推理就会 abort。这是新后端最常见的失败方式。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      'graph_compute 接收整张图，逐节点分发。</br>BLAS 只认两个算子，其余靠切分挡在外面。',
      '<span class="v">第 232 行先跳过</span>没有 GGML_TENSOR_FLAG_COMPUTE 的节点 —— 视图类算子不产生计算（L1-03）。',
      '两个真分支：<span class="k">MUL_MAT</span> 与 <span class="k">OUT_PROD</span>，各对应一个静态函数。',
      '5 个视图算子显式 <span class="m">break</span>：它们"被支持"但不需要做任何事 —— 与 supports_op 的 true 对应。',
      '<span class="k">default 是 GGML_ABORT</span>：这就是为什么 supports_op 不能说假话。',
      '一句话：<span class="v">supports_op 是承诺，graph_compute 是兑现</span>。两者必须一致。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    cases.forEach((_, i) => tl.at(2900 + i * 2700, () => {
      els.forEach((e, k) => { e.style.opacity = k <= i ? '1' : '.30'; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(16800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 一个工厂函数 + 一个导出宏，就是<span class="hl-d">注册</span>的全部 */
{
  kicker: "L8-03 · 注册面",
  title: "一个工厂函数 + 一个导出宏，就是<span class=\"hl-d\">注册</span>的全部",
  sub: "静态构建时 reg.cpp 直接调用 reg()；动态构建时 GGML_BACKEND_DL_IMPL 把同一个 reg() 导出成 ggml_backend_init。",
  caption: "回顾 L3-02（注册表）与 L3-03（动态加载）：两条路，同一个入口函数。",
  src: "ggml/src/ggml-blas/ggml-blas.cpp",
  mark: [0, 3, 8, 13],
  lineNo: 520,
  code: `ggml_backend_reg_t ggml_backend_blas_reg(void) {
//>> 工厂函数：返回一个静态的 ggml_backend_reg —— 注册表最终拿到的就是它
    static struct ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
//>> api_version 必须等于 GGML_BACKEND_API_VERSION（第 11 行）；reg.cpp 第 246 行会逐字校验
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ NULL,
    };

//>> reg 结构只有三个字段：api_version / iface / context（第 242-246 行）
    return &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
//>> 这一行在 GGML_BACKEND_DL 打开时生成 extern "C" 的 ggml_backend_init（第 256-264 行）`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center;gap:5px">
        <span class="chip a">ggml_backend_blas_reg()</span><span class="arrow">-></span>
        <span class="chip c">ggml_backend_reg{api_version, iface}</span><span class="arrow">-></span>
        <span class="chip b">register_backend()</span><span class="arrow">-></span>
        <span class="chip d">全局 backends[]</span>
      </div>
      <div class="row" style="gap:9px">
        <div class="col grow" id="a" style="gap:6px"></div>
        <div class="col grow" id="b" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const A = wrap.querySelector('#a');
    A.innerHTML = '<div class="cm" style="margin-bottom:4px">静态构建（GGML_BACKEND_DL=OFF）</div>';
    [['reg.cpp:65-66', 'GGML_USE_BLAS -> #include "ggml-blas.h"'],
     ['reg.cpp:160', 'register_backend(ggml_backend_blas_reg())'],
     ['CMakeLists.txt:589', 'ggml_add_backend(BLAS) 打开 GGML_USE_BLAS']].forEach(r => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
      e.innerHTML = '<span class="m">' + U.esc(r[0]) + '</span> ' + U.esc(r[1]);
      A.appendChild(e);
    });

    const B = wrap.querySelector('#b');
    B.innerHTML = '<div class="cm" style="margin-bottom:4px">动态构建（GGML_BACKEND_DL=ON）</div>';
    [['impl.h:258-264', 'GGML_BACKEND_DL_IMPL 展开出 ggml_backend_init'],
     ['reg.cpp:229-237', 'dl_get_sym 先找 score，再找 init'],
     ['reg.cpp:246', 'api_version != GGML_BACKEND_API_VERSION 就丢弃']].forEach(r => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
      e.innerHTML = '<span class="m">' + U.esc(r[0]) + '</span> ' + U.esc(r[1]);
      B.appendChild(e);
    });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '注册要写的东西很少：一个返回 reg 的工厂函数，加一行导出宏。',
      '<span class="v">静态路线</span>：编译期定义 GGML_USE_BLAS，reg.cpp 在构造注册表时直接调用 reg()（L3-02）。',
      '<span class="v">动态路线</span>：CMake 定义 GGML_BACKEND_DL，宏展开出 <span class="k">ggml_backend_init</span>，由 dlopen + dlsym 找到（L3-03）。',
      '两条路的交汇点是同一个 reg 结构；<span class="k">api_version 是硬门槛</span>，不匹配的后端会被丢弃并打日志。',
      '还有两个可选符号：<span class="v">ggml_backend_score</span>（打分，0 = 不支持本机）与 ' +
      '<span class="v">get_proc_address</span>（扩展入口，测试工具会来问 ggml_backend_set_n_threads）。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3900, () => { msg.innerHTML = texts[1]; A.style.opacity = '1'; B.style.opacity = '.35'; });
    tl.at(6900, () => { msg.innerHTML = texts[2]; A.style.opacity = '.35'; B.style.opacity = '1'; });
    tl.at(9900, () => { msg.innerHTML = texts[3]; A.style.opacity = '1'; B.style.opacity = '1'; });
    tl.at(13000, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 7 清单（a）必须实现 ·（b）可以留空 */
{
  kicker: "L8-03 · 清单 (a)(b)",
  title: "清单（a）必须实现 ·（b）可以留空",
  sub: "按 L3-01 的三张主虚表分组：必需 23 项、可留空 29 项；BLAS 的最小做法是\"借掉 8 个必需项\"。",
  caption: "\"必需\"的判据只有一个：源码注释里没有写 (optional)。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [0, 4, 27, 37],
  lineNo: 121,
  code: `    struct ggml_backend_i {
//>> (a) ggml_backend_i：16 个槽位，必需 3 个（get_name / free / graph_compute）
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);
//>> free：必需 —— 后端自己要负责释放 context 与 backend 结构

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
//>> graph_compute：必需 —— 唯一的执行入口，也是整个后端实现所在

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params);
//>> graph_optimize 是 (optional)：只有需要重排节点/跨流执行的后端才实现
    };`,
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['虚表', '(a) 必须实现', '(b) 可留空', '依据'],
      [['ggml_backend_device_i', '9：自我介绍 5（get_name / get_description / get_memory / get_type / get_props）+ init_backend + get_buffer_type + supports_op + supports_buft',
        '6：get_host_buffer_type / buffer_from_host_ptr / offload_op / event_new / event_free / event_synchronize', 'impl.h 176-218'],
       ['ggml_backend_i', '3：get_name / free / graph_compute',
        '13：异步 5 / synchronize / graph_plan 4 / event 2 / graph_optimize', 'impl.h 121-156'],
       ['ggml_backend_buffer_type_i', '3：get_name / alloc_buffer / get_alignment',
        '3：get_max_size / get_alloc_size / is_host（都有默认行为）', 'impl.h 17-29'],
       ['ggml_backend_buffer_i', '5：get_base / memset_tensor / set_tensor / get_tensor / clear',
        '6：free_buffer / init_tensor / set_tensor_2d / get_tensor_2d / cpy_tensor / reset', 'impl.h 46-67'],
       ['ggml_backend_reg_i', '3：get_name / get_device_count / get_device',
        '1：get_proc_address', 'impl.h 230-240'],
       ['合计', '23 个必需项（三张主虚表 15 + 数据面 5 + 注册面 3）',
        '29 个可留空 —— BLAS 更进一步：连内存层 8 个必需项也借了 CPU 的', 'L3-01 第 8 幕']],
      { monoCols: [] });
    wrap.querySelector('#tbl').appendChild(t.el);
    t.el.style.fontSize = '9px';

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '把 L3-01 数过的 52 个函数指针分成两列：必需的写下名字，可留空的只写个数。',
      'device_i：<span class="k">必需 9</span>。前 5 个是自我介绍，后 4 个决定"怎么被创建、怎么判能力"。',
      'backend_i：<span class="k">必需 3</span>。异步、图计划、事件、优化全部可以留空。',
      'buffer_type_i：<span class="k">必需 3</span>。alloc_buffer 把"字节数"变成"一块 buffer"。',
      'buffer_i：<span class="k">必需 5</span>。要在设备之间搬张量，才需要额外实现 cpy_tensor 或异步版本。',
      'reg_i：<span class="k">必需 3</span>。这一层只回答"我叫什么、我有几个设备"。',
      '验收答案：<span class="v">23 个必需项</span>；但 BLAS 证明其中 8 个（内存层）可以直接借 CPU 的。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2800 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(18500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 8 清单（c）必须导出的符号 ·（d）必须跑通的测试 */
{
  kicker: "L8-03 · 清单 (c)(d)",
  title: "清单（c）必须导出的符号 ·（d）必须跑通的测试",
  sub: "符号面只有 4 个名字；测试面从\"设备出现\"到\"端到端推理\"共 4 步，全部用仓库现成工具。",
  caption: "这 4 步就是本课的验收点：能列出它们，也就说明前七层都串起来了。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [3, 6, 12, 15, 21],
  lineNo: 248,
  code: `    // Add backend dynamic loading support to the backend

    // Initialize the backend
    typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);
//>> (c) 加载器要找的符号类型：ggml_backend_init —— 返回一个 reg
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
//>> score 是可选符号：返回 0 表示"这台机器上别用我"
    typedef int                (*ggml_backend_score_t)(void);

#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \\
//>> GGML_BACKEND_DL_IMPL：只有 GGML_BACKEND_DL 打开时才展开成真函数
            extern "C" {                                                 \\
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void); \\
//>> 导出的符号名固定为 ggml_backend_init，后端不能改名（reg.cpp 第 237 行按名查找）
            }                                                            \\
            ggml_backend_reg_t ggml_backend_init(void) {                 \\
                return reg_fn();                                         \\
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \\
//>> GGML_BACKEND_DL_SCORE_IMPL：可选的第二个导出符号 ggml_backend_score
            extern "C" {                                   \\
            GGML_BACKEND_API int ggml_backend_score(void); \\
            }                                              \\
            int ggml_backend_score(void) {                 \\
                return score_fn();                         \\
            }
#    else
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \\
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);  \\
            ggml_backend_reg_t                  ggml_backend_init(void) { \\
                return reg_fn();                                          \\
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \\
            GGML_BACKEND_API int ggml_backend_score(void);  \\
            int                  ggml_backend_score(void) { \\
                return score_fn();                          \\
            }
#    endif
#else
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="t1"></div><div id="t2"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t1 = U.table(
      ['(c) 必须注册 / 导出的符号', '谁在等它（依据）'],
      [['ggml_backend_reg_t <name>_reg(void)', '静态构建：reg.cpp:160 直接调用它'],
       ['reg->api_version = GGML_BACKEND_API_VERSION', '加载时校验：reg.cpp:246，不匹配就丢弃'],
       ['ggml_backend_init（由 GGML_BACKEND_DL_IMPL 生成）', '动态加载：reg.cpp:237 用 dl_get_sym 找这个名字'],
       ['ggml_backend_score（可选，GGML_BACKEND_DL_SCORE_IMPL）', '自动挑选：reg.cpp:229 与 534 用它打分'],
       ['get_proc_address("ggml_backend_set_n_threads")', '测试工具会来问：test-backend-ops.cpp:12145']],
      { monoCols: [0] });
    const t2 = U.table(
      ['(d) 验证步骤', '命令', '判据 / 依据'],
      [['1 设备出现', 'llama-cli --list-devices', '打印 name: description（common/arg.cpp:1157；只列非 CPU 设备）'],
       ['2 能力探测', 'test-backend-ops support -b BLAS', '逐算子打印 yes / no，就是 supports_op 的答案（第 1736 行）'],
       ['3 正确性', 'test-backend-ops test -b BLAS', '与 CPU 后端逐元素比误差（第 1530-1549 行）'],
       ['4 端到端', 'llama-cli -m model.gguf -ngl N', '真实模型跑通；BLAS 设备类型是 ACCEL（第 355 行）']],
      { monoCols: [1] });
    wrap.querySelector('#t1').appendChild(t1.el);
    wrap.querySelector('#t2').appendChild(t2.el);
    t1.el.style.fontSize = '9px';
    t2.el.style.fontSize = '9px';

    const msg = wrap.querySelector('#msg');
    const rows = t2.body.querySelectorAll('tr');
    const texts = [
      '符号面只有 4 个名字：reg() / api_version / ggml_backend_init / ggml_backend_score，外加一个可选扩展入口。',
      '<span class="v">第一步</span>：设备要能被列出来 —— 这是 device_i 与 reg_i 填对了的最直接证据。',
      '<span class="v">第二步</span>：support 模式把每个算子的 supports_op 答案打出来，先确认"承诺"的范围。',
      '<span class="v">第三步</span>：test 模式与 CPU 后端逐元素比误差 —— 这是新后端必须过的硬门禁。',
      '第四步：接进 llama-cli / llama-bench 跑真实模型。注意 BLAS 是 ACCEL 类型，不会被默认跳过。'
    ];
    tl.at(900, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(3600 + i * 3200, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(19500, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 从模型到后端：整套课件的一张总图 */
{
  kicker: "L8 · 收束",
  title: "从模型到后端：整套课件的一张总图",
  sub: "一个算子从模型定义出发，经过建图、切分、后端契约、内存、内核，最后落到某个设备上 —— 这条链就是全套课件。",
  caption: "回到门户 index.html 可以按层重看。L8-01 走完了一个 mul_mat 的旅程，L8-03 收在\"怎么写一个新的\"。",
  src: "ggml/include/ggml-blas.h",
  mark: [1, 4, 9, 12],
  lineNo: 11,
  code: `// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_blas_init(void);
//>> (c) 后端对外的第一个函数：创建 backend（内部就是 new ggml_backend{...}，第 286-307 行）

GGML_BACKEND_API bool ggml_backend_is_blas(ggml_backend_t backend);
//>> 类型判断：靠 guid 比对（第 309-311 行）—— 后面所有 "是不是某个后端" 都这么写

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_BACKEND_API void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);
//>> 扩展入口：set_n_threads 不在虚表里，通过 get_proc_address 暴露（第 503-511 行）

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_blas_reg(void);
//>> (c) 注册入口：静态构建时 reg.cpp:160 调它；动态构建时由 GGML_BACKEND_DL_IMPL 导出`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center;gap:5px">
        <span class="chip">模型定义</span><span class="arrow">-></span>
        <span class="chip a">计算图</span><span class="arrow">-></span>
        <span class="chip c">切分</span><span class="arrow">-></span>
        <span class="chip b">后端虚表</span><span class="arrow">-></span>
        <span class="chip d">内核</span><span class="arrow">-></span>
        <span class="chip e">设备</span>
      </div>
      <div class="flow" style="justify-content:center;gap:5px">
        <span class="chip">L2 建图</span><span class="arrow">-></span>
        <span class="chip">L4-02 supports_op 切分</span><span class="arrow">-></span>
        <span class="chip">L3-01/02/03 契约·注册·加载</span><span class="arrow">-></span>
        <span class="chip">L5/L6/L7 graph_compute</span><span class="arrow">-></span>
        <span class="chip">L4-03 buffer type</span>
      </div>
      <div id="ex"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码回答：<b>新增一个后端</b>最少要实现哪些函数？必须跑通哪些测试？' +
      '（提示：先按虚表说，再说"哪些可以借"）',
      '接口：三张主虚表的 <span class="mono">15</span> 个必需项 —— ' +
      '<span class="mono">device_i</span> 9（自我介绍 5 + <span class="mono">init_backend</span> + ' +
      '<span class="mono">get_buffer_type</span> + <span class="mono">supports_op</span> + ' +
      '<span class="mono">supports_buft</span>）+ <span class="mono">backend_i</span> 3（' +
      '<span class="mono">get_name / free / graph_compute</span>）+ ' +
      '<span class="mono">reg_i</span> 3（<span class="mono">get_name / get_device_count / get_device</span>）。<br>' +
      '可以借的：内存层 <span class="mono">buffer_type_i</span> 3 + <span class="mono">buffer_i</span> 5 = 8 个必需项 —— ' +
      '<span class="mono">get_buffer_type</span> 直接返回 <span class="mono">ggml_backend_cpu_buffer_type()</span>，' +
      'BLAS 就是这么做的（全文 0 处 buffer_i / buffer_type_i）。<br>' +
      '必须导出的符号：<span class="mono">reg()</span>、<span class="mono">api_version</span>、' +
      '<span class="mono">ggml_backend_init</span>（DL 宏生成），可选 ' +
      '<span class="mono">ggml_backend_score</span>。<br>' +
      '必须跑通：<span class="mono">llama-cli --list-devices</span>（设备出现）-> ' +
      '<span class="mono">test-backend-ops support -b NAME</span>（能力探测）-> ' +
      '<span class="mono">test-backend-ops test -b NAME</span>（与 CPU 比误差）-> ' +
      '真实模型端到端。<br>总工作量参照：BLAS 后端 1 个文件、530 行、17 个非 NULL 指针。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '整套课件就是这条链：<span class="k">模型 -> 图 -> 切分 -> 契约 -> 内存 -> 内核 -> 设备</span>。',
      '上半行是<b>数据方向</b>，下半行是<b>课件里的层</b>：L2 建图、L4-02 切分、L3 契约与注册、L5-L7 内核、L4-03 内存。',
      '新后端接入的位置是这条链的<b>后半段</b>：它只负责"我能不能跑"（supports_op）与"怎么跑"（graph_compute）。',
      '<span class="v">L8-01</span> 走完了一个 mul_mat 的完整旅程；<span class="v">L8-02</span> 用真实参数做了切分决策；' +
      '这一课收在"怎么写一个新的"。',
      '验收点：你能列出上面这份清单（接口 / 可留空 / 符号 / 测试），这一课就完成了 —— 整套课件也到此结束。',
      '从门户重看：<span class="k">index.html</span> 支持按层过滤与关键词搜索；每一课都可以双击离线打开。'
    ];
    tl.at(900, () => { msg.innerHTML = texts[0]; });
    tl.at(4200, () => { msg.innerHTML = texts[1]; });
    tl.at(7500, () => { msg.innerHTML = texts[2]; });
    tl.at(10800, () => { msg.innerHTML = texts[3]; });
    tl.at(14100, () => { msg.innerHTML = texts[4]; });
    tl.at(17400, () => { msg.innerHTML = texts[5]; });
  }
},

];
