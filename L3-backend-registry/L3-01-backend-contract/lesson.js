/* ==========================================================================
   L3-01 · 后端接口 ggml-backend.h：后端的契约
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 一个后端，先要签一份<span class="hl-a">契约</span> */
{
  kicker: "L3 · 后端发现与注册",
  title: "一个后端，先要签一份<span class=\"hl-a\">契约</span>",
  sub: "公共头里没有结构体，只有 7 个不透明句柄：模型代码永远看不到后端的内部。",
  caption: "本幕只引 ggml-backend.h 第 22-30 行。下一幕进内部头 ggml-backend-impl.h 看虚表本身。",
  src: "ggml/include/ggml-backend.h",
  mark: [0, 2, 4, 7, 8],
  lineNo: 24,
  code: `    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
//>> 6 个 struct 指针 + 1 个 void*，就是公共头对"后端"的全部认知
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
    typedef struct ggml_backend_event * ggml_backend_event_t;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;
//>> graph_plan_t 是唯一的例外：它不是 struct 指针，而是 void* —— 一个纯不透明的执行计划
    typedef struct ggml_backend_reg * ggml_backend_reg_t;
    typedef struct ggml_backend_device * ggml_backend_dev_t;`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="flow" style="justify-content:center">
        <span class="chip">计算图</span><span class="arrow">-></span>
        <span class="chip a">调度器</span><span class="arrow">-></span>
        <span class="chip c">后端契约</span><span class="arrow">-></span>
        <span class="chip b">CPU / CUDA / Metal</span>
      </div>
      <div class="row wrap" id="handles" style="gap:6px;justify-content:center"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { t: 'ggml_backend_dev_t', k: '设备' },
      { t: 'ggml_backend_t', k: '后端（计算流）' },
      { t: 'ggml_backend_buffer_type_t', k: '内存类型' },
      { t: 'ggml_backend_buffer_t', k: '一块内存' },
      { t: 'ggml_backend_reg_t', k: '注册项' },
      { t: 'ggml_backend_event_t', k: '事件' },
      { t: 'ggml_backend_graph_plan_t', k: '图计划（void*）' }
    ];
    const host = wrap.querySelector('#handles');
    const els = defs.map(d => { const e = U.chip(d.t + ' · ' + d.k); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一课只回答一个问题：<span class="k">别人写的后端，凭什么能被 ggml 调用？</span>',
      '答案是一份契约。公共头用 <span class="v">7 个 typedef</span> 把实现藏起来：<br>其中 <span class="k">6 个是不透明 struct 指针</span>，只有 <span class="v">ggml_backend_graph_plan_t</span> 是 void*。',
      '结构体一个都不在公共头里 —— 它们全在 <span class="v">ggml/src/ggml-backend-impl.h</span>，<br>那是给"写后端的人"看的内部契约头。',
      '公共头的 <span class="v">115 处 GGML_API</span> 只做转发：函数名好记，干活的是虚表。',
      '记住这条分工：<span class="k">公共 API 是给人调的，虚表是给后端实现的。</span>'
    ];
    defs.forEach((_, i) => tl.at(600 + i * 2500, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(14200, () => { els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">三张虚表</span> = 三个层次 */
{
  kicker: "L3-01 · 虚表的形状",
  title: "★ <span class=\"hl-a\">三张虚表</span> = 三个层次",
  sub: "设备层回答\"我是谁、能不能跑\"，内存层回答\"内存怎么分配\"，计算层回答\"图怎么算出来\"。",
  caption: "数出来的：device_i 15 个、buffer_type_i 6 个、backend_i 16 个函数指针。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [2, 11, 18, 34, 36],
  lineNo: 1,
  code: `#pragma once

// ggml-backend internal header
//>> 文件自己声明：这是 internal header —— 公共头之外的"后端实现者契约"

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    #define GGML_BACKEND_API_VERSION 2
//>> 契约带版本号；L3-03 会看到动态加载后端时怎么校验它

    //
    // Backend buffer type
    //

    struct ggml_backend_buffer_type_i {
//>> 一张虚表就是一个只装函数指针的 struct，可以 static const 初始化
        const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
//>> 宿主对象把虚表放在第一个字段 —— 拿到 buft 就等于拿到了它的虚表
        ggml_backend_dev_t device;
//>> 每个 buffer type 记住自己属于哪个设备，L4-03 分配内存时用它
        void * context;
    };`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="layers" style="gap:8px"></div>
      <div class="formula" id="dia"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '设备层 · ggml_backend_device_i', m: '15 个函数指针',
        b: '发现与创建：我是谁 →<br>能不能跑这个算子 → 把我变成后端' },
      { c: 'c', t: '内存层 · ggml_backend_buffer_type_i', m: '6 个函数指针',
        b: '内存怎么分配：对齐多少、<br>多大的块、算不算 host 内存' },
      { c: 'b', t: '计算层 · ggml_backend_i', m: '16 个函数指针',
        b: '计算怎么执行：图怎么算出来、<br>数据怎么搬、什么时候同步' }
    ];
    const host = wrap.querySelector('#layers');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const dia = wrap.querySelector('#dia');
    dia.innerHTML = '<span class="m">struct ggml_backend_buffer_type_i</span> —— 只装函数指针的表<br>' +
      '&nbsp;&nbsp;&darr; <span class="k">作为第一个字段嵌进宿主对象</span><br>' +
      '<span class="m">struct ggml_backend_buffer_type</span> { <span class="v">iface</span>; <span class="v">device</span>; <span class="v">context</span>; }';
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看形状：每个 `_i` 结构体 <span class="k">只装函数指针</span>，不装数据。<br>数据在同名的宿主结构体里（iface + device + context）。',
      '<span class="v">设备层</span>：从"机器上有什么"出发，产出可执行的后端和内存类型。<br>L3-02 讲设备从哪来，L3-04 讲能力怎么探测。',
      '<span class="v">内存层</span>：只有一个职责 —— 把一段字节变成一块张量内存。<br>L4-03 会展开 buffer 与 buffer type 的分工。',
      '<span class="v">计算层</span>：L5-01 会从这里的 graph_compute 一路走到 CPU 内核的大 switch。',
      '一句话：<span class="k">新后端 = 填完这几张表</span>。表外的世界（图、张量、调度）不用重写。'
    ];
    defs.forEach((_, i) => tl.at(600 + i * 2600, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(9000, () => { msg.innerHTML = texts[3]; U.markLines(document, [0, 1, 2]); });
    tl.at(12000, () => { els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4]; U.markLines(document, [0, 1, 2, 3, 4]); });
  }
},

/* ------------------------------------------------------ 3 ★ <span class="hl-b">ggml_backend_device_i</span>：15 个函数指针 */
{
  kicker: "L3-01 · 设备层",
  title: "★ <span class=\"hl-b\">ggml_backend_device_i</span>：15 个函数指针",
  sub: "自我介绍 5 个、创建与内存 4 个、能力判据 3 个、事件 3 个 —— 其中 6 个标了 (optional)。",
  caption: "回顾 L2-03：权重用 mmap 直接映射进来时，走的就是这里的 buffer_from_host_ptr。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [2, 9, 13, 19, 23, 33, 37, 42],
  lineNo: 176,
  code: `    struct ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        const char * (*get_name)(ggml_backend_dev_t dev);
//>> 自我介绍五件套从这里开始：name / description / memory / type / props，全部必需

        // device description: short informative description of the device, could be the model name
        const char * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        void         (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);
//>> get_memory 报告空闲与总字节数；注释说明返回 0 表示"不上报"

        // device type
        enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        void (*get_props)(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, const char * params);
//>> init_backend：设备 -> 后端。公共 API ggml_backend_dev_init() 转发到这里

        // preferred buffer type
        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);
//>> get_buffer_type：设备首选的内存类型 —— 之后所有张量都分配在它上面

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);
//>> supports_op：调度器切图的判据 —— "这个算子我能不能跑"

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
//>> supports_buft：反向判据 —— "这份权重所在的内存我能不能用"

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);
//>> offload_op 标了 (optional)：即使权重在别的内存里，也愿意把它搬过来算

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '自我介绍 ×5', m: 'get_name · get_description',
        b: 'get_memory · get_type · get_props<br><span class="k">全部必需</span>' },
      { c: 'c', t: '创建与内存 ×4', m: 'init_backend · get_buffer_type',
        b: 'get_host_buffer_type · buffer_from_host_ptr<br>后两个标了 (optional)' },
      { c: 'b', t: '能力判据 ×3', m: 'supports_op · supports_buft',
        b: 'offload_op（optional）<br><span class="k">调度器切图就靠这三个</span>' },
      { c: 'd', t: '事件 ×3', m: 'event_new · event_free',
        b: 'event_synchronize<br>整组 (optional)：不支持就不填' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '这一张表有 <span class="v">15 个函数指针</span>。按职责分四组，一眼能数完。',
      '<span class="v">自我介绍 5 个</span>（全部必需）：名字、描述、显存、类型、属性。<br>公共头把它们原样暴露成 ggml_backend_dev_name / memory / type / get_props。',
      '<span class="v">创建与内存 4 个</span>：<span class="k">init_backend 才是"设备 -> 可执行后端"的那一步</span>；<br>get_buffer_type 给出设备首选的内存类型。',
      '<span class="v">buffer_from_host_ptr</span> 标了 (optional)，注释写明用途：<br>memory mapped models —— 这正是 L2-03 里权重落位的那条路。',
      '<span class="v">能力判据 3 个</span>：supports_op / supports_buft 是必需，offload_op 是可选。<br>第 6 幕会看到：后端级同名公共 API 已被注释标记为"将移除"。',
      '事件三件套整组是 (optional)：<span class="k">不需要跨流同步的后端可以直接不填。</span>'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(12500, () => { msg.innerHTML = texts[4]; U.markLines(document, [3, 4, 5, 6]); });
    tl.at(15500, () => { els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[5]; U.markLines(document, [0, 1, 2]); });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-c">ggml_backend_i</span>：16 个函数指针，必需的只有 3 个 */
{
  kicker: "L3-01 · 计算层",
  title: "★ <span class=\"hl-c\">ggml_backend_i</span>：16 个函数指针，必需的只有 3 个",
  sub: "get_name / free / graph_compute —— 其余 13 个都落在带 (optional) 的注释组里。",
  caption: "数出来的必需项：16 - 13 = 3。异步五件套整组标了 (optional)。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [1, 4, 12, 16, 21, 30, 40],
  lineNo: 121,
  code: `    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);
//>> get_name：必需。调度器、日志、报错信息都用它标识后端

        void (*free)(ggml_backend_t backend);
//>> free：必需。设备层的 init_backend 负责产出后端，这里负责销毁它

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
//>> 异步五件套（set/get 各两件 + cpy_tensor_async）整组标了 (optional)

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(ggml_backend_t backend);
//>> synchronize：注释写明 —— 支持异步操作的后端就必须填它

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
//>> graph_plan 四件套：注释直接写着 "not used currently"，是预留位
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
//>> graph_compute：唯一必需的计算入口，返回 enum ggml_status

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params);
//>> graph_optimize 也标了 (optional)：愿意重排节点就填
    };`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'b', t: '必需 ×3', m: 'get_name · free',
        b: 'graph_compute<br><span class="k">唯一必需的计算入口</span>' },
      { c: 'a', t: '异步与同步 ×6', m: '五个 async 数据访问',
        b: 'synchronize：注释写明<br>"required if the backend supports async operations"' },
      { c: 'd', t: '预留与可选 ×7', m: 'graph_plan 四件套',
        b: 'event_record / event_wait / graph_optimize<br>graph plan 注释写 "not used currently"' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">16 个函数指针</span>，但数一遍带 (optional) 的注释组就知道：<br>真正非填不可的只有 <span class="k">3 个</span>。',
      '<span class="v">get_name</span> 是给人和日志看的；<span class="v">free</span> 是生命周期；<br><span class="v">graph_compute</span> 是整个后端的唯一计算入口。',
      '算一次图 = 调一次 <span class="v">graph_compute</span>，返回 <span class="v">enum ggml_status</span>。<br>L5-01 会从这里一路追到 CPU 内核的大 switch。',
      '五个 async 数据访问 + synchronize 是一组：<span class="k">要么都填，要么都不用。</span><br>不填时上层走 buffer 层的同步路径（第 5 幕）。',
      'graph_plan 四件套是历史预留位 —— 源码注释直说 "not used currently"。<br>读到这里可以放心跳过它们。',
      '结论：<span class="k">一个"能算图"的后端，计算层只要 3 个函数。</span>难的是算得对、算得快。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(10200, () => { msg.innerHTML = texts[3]; U.markLines(document, [2, 3]); });
    tl.at(13200, () => { els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[5]; U.markLines(document, [0, 4]); });
  }
},

/* ------------------------------------------------------ 5 <span class="hl-e">buffer type</span> 与 <span class="hl-e">buffer</span>：分配前与分配后 */
{
  kicker: "L3-01 · 内存层",
  title: "<span class=\"hl-e\">buffer type</span> 与 <span class=\"hl-e\">buffer</span>：分配前与分配后",
  sub: "buffer_type_i 有 6 个函数指针（3 必需），buffer_i 有 11 个（5 必需）；alloc_buffer 把前者变成后者。",
  caption: "回顾 L1-01：ggml_tensor 的 buffer 字段指向的就是这里的 ggml_backend_buffer；init_tensor 会往 extra 字段填后端私有数据。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [6, 9, 12, 15, 19, 25, 28, 31],
  lineNo: 42,
  code: `    //
    // Backend buffer
    //

    struct ggml_backend_buffer_i {
        // (optional) free the buffer
        void         (*free_buffer)  (ggml_backend_buffer_t buffer);
//>> free_buffer 标了 (optional)：内存由外层统一管理时可以不填
        // base address of the buffer
        void *       (*get_base)     (ggml_backend_buffer_t buffer);
//>> get_base：必需。给出这块内存的起始地址
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
//>> init_tensor：给张量补后端私有信息 —— 填的就是 L1-01 里的 extra 字段
        // tensor data access
        void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
//>> memset / set / get：必需的三件数据搬运，签名都带 offset 与 size —— 可以只动张量的一段
        void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) 2d data copies
//>> 两个 2d 拷贝标了 (optional)：批量跨步拷贝的快捷方式
        void         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
//>> cpy_tensor 标了 (optional)，注释说明 src 可以来自任何后端的 buffer
        // clear the entire buffer
        void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
//>> clear：必需。整块内存刷成同一个字节值
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        void         (*reset)        (ggml_backend_buffer_t buffer);
//>> reset：init_tensor 的反向操作，也标了 (optional)
    };`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const dia = wrap.querySelector('#dia');

    const bt = U.el('div', { class: 'card', style: 'width:300px' });
    bt.innerHTML = '<div class="cm" style="margin:0 0 3px">分配前 · 内存的"型号"</div>' +
      '<div class="ct" style="color:var(--c)">ggml_backend_buffer_type_i</div>' +
      '<div class="cb"><span class="k">6 个函数指针</span><br>' +
      '必需 3：get_name · alloc_buffer · get_alignment<br>' +
      '可选 3：get_max_size · get_alloc_size · is_host</div>';

    const bf = U.el('div', { class: 'card', style: 'width:300px' });
    bf.innerHTML = '<div class="cm" style="margin:0 0 3px">分配后 · 一块具体的内存</div>' +
      '<div class="ct" style="color:var(--e)">ggml_backend_buffer_i</div>' +
      '<div class="cb"><span class="k">11 个函数指针</span><br>' +
      '必需 5：get_base · memset_tensor · set_tensor · get_tensor · clear<br>' +
      '可选 6：free_buffer · init_tensor · set_tensor_2d · get_tensor_2d · cpy_tensor · reset</div>';

    dia.appendChild(bt); dia.appendChild(U.arrow('->')); dia.appendChild(bf);

    const msg = wrap.querySelector('#msg');
    const texts = [
      '一行看懂两者关系：<span class="v">buft-&gt;iface.alloc_buffer(buft, size)</span> 返回一个 buffer。',
      '<span class="v">buffer type</span> 是"型号"：对齐多少、最大能开多大、<br>算不算 host 内存（is_host 为真时数据能被 CPU 直接读）。',
      '<span class="v">buffer</span> 是"实物"：一块已经拿到的内存，外加怎么往里写、怎么读出来。<br>必需的 5 个全在数据面上。',
      'L1-01 里 ggml_tensor 的 <span class="v">buffer</span> 字段指向的就是这里的 buffer；<br><span class="v">init_tensor</span> 顺手把后端私有数据塞进 <span class="v">extra</span> 字段。',
      'L4-03 会把这两个虚表放在一起讲：<span class="k">什么时候选型号，什么时候开实物。</span>'
    ];
    tl.at(700, () => { bt.style.opacity = '.35'; bf.style.opacity = '.35'; msg.innerHTML = texts[0]; });
    tl.at(3400, () => { bt.style.opacity = '1'; bf.style.opacity = '.35'; msg.innerHTML = texts[1]; });
    tl.at(6400, () => { bt.style.opacity = '.35'; bf.style.opacity = '1'; msg.innerHTML = texts[2]; });
    tl.at(9600, () => { bt.style.opacity = '1'; bf.style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(13000, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 公共 API 是虚表的<span class="hl-a">镜像</span> */
{
  kicker: "L3-01 · 两层分工",
  title: "公共 API 是虚表的<span class=\"hl-a\">镜像</span>",
  sub: "模型代码调的是 ggml_backend_* 系列；它们只是把参数原样转发给某张虚表。",
  caption: "注意第 107 行的注释：后端级的 supports_op / supports_buft / offload_op 将被移除 —— 能力判据归设备层。",
  src: "ggml/include/ggml-backend.h",
  mark: [0, 3, 5, 7, 11, 17],
  lineNo: 104,
  code: `    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);

    // NOTE: will be removed, use device version instead
//>> 注释写明：这三个后端级 API 将被移除，改用设备版本 —— 判据只存于 device_i
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
//>> supports_op / supports_buft / offload_op 在 ggml_backend_i 里根本没有对应函数
    GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
//>> 异步拷贝注释：不支持时自动回退成同步拷贝
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);
//>> 每个后端都能反查自己的设备：对应 impl 里 struct ggml_backend 的 device 字段`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['公共 API（本幕引用的这一段）', '真正干活的地方', '备注'],
      [['ggml_backend_graph_compute', 'backend_i.graph_compute', '第 4 幕第 146 行'],
       ['ggml_backend_graph_compute_async', 'backend_i.graph_compute', '异步由 synchronize 收口'],
       ['ggml_backend_supports_op', 'device_i.supports_op', '第 3 幕第 205 行'],
       ['ggml_backend_supports_buft', 'device_i.supports_buft', '第 3 幕第 208 行'],
       ['ggml_backend_offload_op', 'device_i.offload_op', '第 3 幕第 212 行（optional）'],
       ['ggml_backend_tensor_copy_async', 'backend_i.cpy_tensor_async', '不支持时回退同步'],
       ['ggml_backend_get_device', 'struct ggml_backend 的 device', '后端反查自己的设备']],
      { monoCols: [0, 1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '数出来的：<span class="v">ggml-backend.h 有 115 处 GGML_API</span>，几乎每个都能在虚表里找到落点。',
      '三个 compute 系列都落到同一个 <span class="v">graph_compute</span>。<br>区别只在"什么时候算完"：<span class="k">异步与否由 synchronize 收口。</span>',
      '注意 supports_op / supports_buft / offload_op 三行：<br>它们在 <span class="v">ggml_backend_i</span> 里<b>没有</b>对应函数，<span class="k">只在 device_i 里</span>。',
      '这就是第 107 行那句注释的含义：<span class="k">能力判据属于设备层。</span><br>第 4 幕那张 16 个指针的表里，确实一个 supports_* 都没有。',
      '最后一行是反向指针：后端知道自己属于哪个设备。<br>这条边让"调度器只拿到一个 backend"时仍能问出设备能力。',
      '一句话：<span class="k">读公共头能知道"能做什么"，读内部头才知道"谁来做的"。</span>'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2100, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 4)];
    }));
    tl.at(14000, () => { rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[5]; U.markLines(document, [3, 4, 5]); });
  }
},

/* ------------------------------------------------------ 7 还有一张表：<span class="hl-d">ggml_backend_reg_i</span>（4 个函数指针） */
{
  kicker: "L3-01 · 注册面",
  title: "还有一张表：<span class=\"hl-d\">ggml_backend_reg_i</span>（4 个函数指针）",
  sub: "一个后端 = 一组设备 + 一个版本号 + 一个可选扩展入口；它是 L3-02 与 L3-03 的接口。",
  caption: "注册与设备枚举是下一课 L3-02 的主题；动态库导出的是本幕引用的这两个 typedef。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [5, 9, 10, 14, 19, 28, 32],
  lineNo: 226,
  code: `    //
    // Backend (reg)
    //

    struct ggml_backend_reg_i {
        const char * (*get_name)(ggml_backend_reg_t reg);
//>> get_name / get_device_count / get_device 三个必需 —— 设备枚举就靠它们

        // enumerate available devices
        size_t             (*get_device_count)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
//>> get_proc_address 标了 (optional)：后端私有扩展函数的唯一入口
    };

    struct ggml_backend_reg {
        int api_version; // initialize to GGML_BACKEND_API_VERSION
//>> 注释写明这个字段要初始化成 GGML_BACKEND_API_VERSION（第 2 幕第 11 行那个宏）
        struct ggml_backend_reg_i iface;
        void * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);
//>> 动态加载时后端库必须导出 ggml_backend_init，返回的就是 ggml_backend_reg_t
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    typedef int                (*ggml_backend_score_t)(void);
//>> 可选的 ggml_backend_score：分数高的后端优先，0 表示当前系统不支持`,
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'd', t: 'ggml_backend_reg_i ×4', m: 'get_name · get_device_count',
        b: 'get_device · get_proc_address(optional)<br><span class="k">后端 = 设备的集合</span>' },
      { c: 'a', t: 'struct ggml_backend_reg', m: 'api_version · iface · context',
        b: 'api_version 注释要求初始化成<br>GGML_BACKEND_API_VERSION（= 2）' },
      { c: 'b', t: '给下一课的接口', m: 'ggml_backend_init / _score',
        b: 'L3-02：谁把 reg 注册进全局表<br>L3-03：动态库怎么导出这两个符号' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '填完前面三张表，还差一步：<span class="k">让 ggml 知道你的存在。</span>',
      '<span class="v">reg_i 只有 4 个函数指针</span>：报名字、数设备、取第 i 个设备，<br>再加一个可选的后端私有扩展入口 get_proc_address。',
      '<span class="v">get_device_count + get_device</span> 就是设备发现的全部机制 ——<br>上层不需要知道设备是从 PCI 枚举来的还是写死的。',
      '<span class="v">api_version</span> 是契约的版本号：<br>它和 ggml_backend_init 一起构成动态加载时的握手（L3-03）。',
      '本课到此为止：<span class="k">接口长什么样已经全部看到。</span>下一课回答"它们从哪来、按什么顺序被选"。'
    ];
    defs.forEach((_, i) => tl.at(600 + i * 3200, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(13200, () => { els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 ★ 实现一个新后端，最少要填哪些函数 */
{
  kicker: "L3-01 · 落地",
  title: "★ 实现一个新后端，最少要填哪些函数",
  sub: "按虚表分组数一遍：设备 9 + 计算 3 + 内存类型 3 = 15 个必需项；加上数据面与注册面共 23 个。",
  caption: "这张表就是本课的验收点；L8-03 会把它变成一份可勾选的\"新后端清单\"。",
  src: "ggml/src/ggml-backend-impl.h",
  mark: [1, 4, 6, 9, 11, 14],
  lineNo: 17,
  code: `    struct ggml_backend_buffer_type_i {
        const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
//>> alloc_buffer 是这一层的核心：把字节数变成一块 buffer
        ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
//>> 三个 (optional) 都有默认行为：max_size 默认 SIZE_MAX、alloc_size 默认 ggml_nbytes、is_host 默认 false
        size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
//>> is_host 为真意味着"数据在系统内存里且布局标准" —— CPU 能直接读
        bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
    };`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['虚表（层次）', '指针数', '必需', '可留空'],
      [['ggml_backend_device_i（发现与创建）', '15', '9', '6'],
       ['ggml_backend_i（计算执行）', '16', '3', '13'],
       ['ggml_backend_buffer_type_i（内存分配）', '6', '3', '3'],
       ['ggml_backend_buffer_i（内存读写）', '11', '5', '6'],
       ['ggml_backend_reg_i（注册）', '4', '3', '1'],
       ['合计', '52', '23', '29']],
      { monoCols: [0, 1, 2, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '<span class="k">必需</span>= 不在任何 (optional) 注释组里；<span class="k">可留空</span>= 源码注释标了 (optional)。',
      '设备层必需 9：<span class="v">get_name · get_description · get_memory · get_type · get_props</span>'
      + ' + init_backend · get_buffer_type · supports_op · supports_buft。',
      '计算层必需 3：<span class="v">get_name · free · graph_compute</span>。<br>异步、图计划、事件、优化全部可选。',
      '内存类型层必需 3：<span class="v">get_name · alloc_buffer · get_alignment</span>；<br>max_size / alloc_size / is_host 都有默认行为。',
      '数据面必需 5：<span class="v">get_base · memset_tensor · set_tensor · get_tensor · clear</span>。<br>要在设备间搬张量，还得补 cpy_tensor 或异步版本。',
      '注册面必需 3：<span class="v">get_name · get_device_count · get_device</span>。<br>这一层怎么挂进全局表，是下一课 L3-02。',
      '验收答案：<span class="k">三张主虚表 15 个必需项</span>；算上数据面与注册面共 23 个。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 6)];
    }));
    tl.at(17500, () => { rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L3-01 · 收束",
  title: "把这一课压成一张表",
  sub: "五张虚表、52 个函数指针、23 个必需项；契约之外的一切都不用重写。",
  caption: "下一课 L3-02：这些设备从哪来、按什么顺序被枚举。",
  src: "ggml/include/ggml-backend.h",
  mark: [0, 1, 2, 3, 4, 6, 7],
  lineNo: 37,
  code: `    GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer  (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API size_t                ggml_backend_buft_get_alignment (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_max_size  (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
//>> 这七个公共函数一一对应 buffer_type_i 的 6 个函数指针，外加一个 get_device
    GGML_API bool                  ggml_backend_buft_is_host       (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);
//>> get_device 暴露的是宿主对象里的 device 字段 —— 公共 API 只做转发`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['层次', '虚表', '本课第几幕', '在哪一课展开'],
      [['契约入口', '7 个不透明句柄', '1', '本课'],
       ['发现与创建', 'ggml_backend_device_i（15）', '3', 'L3-02 / L3-04'],
       ['计算执行', 'ggml_backend_i（16）', '4', 'L5-01 / L6 / L7'],
       ['内存分配', 'buffer_type_i（6）+ buffer_i（11）', '5', 'L4-03'],
       ['注册', 'ggml_backend_reg_i（4）', '7', 'L3-02 / L3-03'],
       ['落地清单', '23 个必需函数', '8', 'L8-03']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码回答：实现一个能真正跑图的新后端，最少要实现哪些接口函数？'
      + '（先按<b>层次</b>说，再补函数名）',
      '按三张主虚表：<br>'
      + '1) <span class="mono">ggml_backend_device_i</span> 里 9 个必需 —— '
      + '自我介绍 5（<span class="mono">get_name / get_description / get_memory / get_type / get_props</span>）'
      + ' + <span class="mono">init_backend</span> + <span class="mono">get_buffer_type</span>'
      + ' + <span class="mono">supports_op</span> + <span class="mono">supports_buft</span>；<br>'
      + '2) <span class="mono">ggml_backend_i</span> 里 3 个 —— '
      + '<span class="mono">get_name / free / graph_compute</span>；<br>'
      + '3) <span class="mono">ggml_backend_buffer_type_i</span> 里 3 个 —— '
      + '<span class="mono">get_name / alloc_buffer / get_alignment</span>。<br>'
      + '这三张表合计 <b>15</b> 个必需项。要真的搬数据，再加 '
      + '<span class="mono">ggml_backend_buffer_i</span> 的 5 个必需项（'
      + '<span class="mono">get_base / memset_tensor / set_tensor / get_tensor / clear</span>）；'
      + '要能被发现，再加 <span class="mono">ggml_backend_reg_i</span> 的 3 个必需项 —— '
      + '合计 <b>23</b> 个。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '五张虚表，52 个函数指针，23 个必需项 —— 这就是"后端"的全部义务。',
      '<span class="k">设备层</span>解决"发现与创建"：L3-02 讲注册与枚举顺序，L3-04 讲能力探测。',
      '<span class="k">计算层</span>只有 3 个必需函数，但 graph_compute 里面是整个后端的实现。',
      '<span class="k">内存层</span>是权重落位的地方：L2-03（mmap）与 L4-03（buffer 家族）都在这条线上。',
      '注册面把前四层挂进全局表；动态加载（L3-03）导出的是 ggml_backend_init。',
      '最后一句：<span class="v">写一个新后端，就是把这 23 个空填完</span> —— L8-03 会逐项对照。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2500, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[i];
    }));
    tl.at(18500, () => { rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[5]; });
  }
},

];
