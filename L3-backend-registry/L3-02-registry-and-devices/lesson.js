/* ==========================================================================
   L3-02 · ★ 后端注册表与设备发现
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 注册表：这台机器上有什么 */
{
  kicker: "L3 · 后端发现与注册",
  title: "注册表：这台机器上有什么",
  sub: "结构体里只有两个 vector：后端列表，和它们贡献的设备列表。",
  caption: "回顾 L3-01：`ggml_backend_reg_i` / `ggml_backend_device_i` 是后端要填的虚表 —— 这一课看它们怎么被收集起来。",
  src: "ggml/src/ggml-backend-reg.cpp",
  mark: [1, 2, 6, 7],
  lineNo: 110,
  code: `struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;
`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">编译期 #ifdef</span><span class="arrow">-></span>
        <span class="chip a">register_backend()</span><span class="arrow">-></span>
        <span class="chip b">backends[] / devices[]</span><span class="arrow">-></span>
        <span class="chip c">llama.cpp 选设备</span>
      </div>
      <div class="row center" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const defs = [
      { c: 'a', t: '注册', b: '静态：构造函数里的 #ifdef<br>动态：load_backend() 之后追加', m: 'register_backend(reg)' },
      { c: 'b', t: '设备', b: '注册一个后端，顺带把它<br>报出来的设备登记进 devices', m: 'ggml_backend_dev_get(i)' },
      { c: 'c', t: '顺序', b: 'devices 的顺序 = 注册顺序，<br>它就是 offload 时“设备 0”的定义', m: 'devices[i]' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个后端注册表项 = 后端句柄 + 动态库句柄（<span class="v">reg</span> + <span class="v">handle</span>）。',
      '注册表本体只有两个 <span class="k">std::vector</span>：<span class="v">backends</span> 装后端，<span class="v">devices</span> 装设备。<br>' +
        '<span class="k">没有排序、没有优先队列</span> —— 后面所有“顺序”问题都从这里来。',
      '<span class="v">register_backend()</span> 是唯一入口：静态注册和动态加载最后都走它，<br>所以两条路的差别只在“什么时候调用”。',
      '这一课的落点：<span class="k">devices[i] 的下标</span>如何一路变成 offload 时“哪张卡放哪些层”。',
      '下一幕先看静态注册：<span class="v">16 个 #ifdef</span> 就把这台机器“有哪些后端”定死了。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3600, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[1]; lightAll(); });
    tl.at(6800, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[2]; lightAll(); });
    tl.at(10000, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; lightAll(); });
    tl.at(13000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 2 ★ 有哪些后端，是<span class="hl-a">编译期</span>定下的 */
{
  kicker: "L3-02 · 核心",
  title: "★ 有哪些后端，是<span class=\"hl-a\">编译期</span>定下的",
  sub: "注册表构造函数里 16 个 #ifdef；开关打开一个，就多注册一个后端。",
  caption: "开关个数是数出来的：`grep -c \"GGML_USE_\" ggml/src/ggml-backend-reg.cpp` = 32 处；去重后 16 个（16 处 include + 16 处构造函数）。",
  src: "ggml/src/ggml-backend-reg.cpp",
  mark: [1, 2, 4, 7, 10, 13],
  lineNo: 119,
  code: `    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
//>> 全文件唯一一处“编译进来了、运行期还能关掉”的开关：GGML_DISABLE_VULKAN（这个文件的 getenv 只有 131 与 601 两处）
    // Add runtime disable check
    if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } else {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\\n");
    }
#endif`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['GGML_USE_CUDA',            '120', 'ggml_backend_cuda_reg()'],
      ['GGML_USE_METAL',           '123', 'ggml_backend_metal_reg()'],
      ['GGML_USE_SYCL',            '126', 'ggml_backend_sycl_reg()'],
      ['GGML_USE_VULKAN',          '129', 'ggml_backend_vk_reg()'],
      ['GGML_USE_WEBGPU',          '137', 'ggml_backend_webgpu_reg()'],
      ['GGML_USE_ZDNN',            '140', 'ggml_backend_zdnn_reg()'],
      ['GGML_USE_VIRTGPU_FRONTEND','143', 'ggml_backend_virtgpu_reg()'],
      ['GGML_USE_OPENCL',          '147', 'ggml_backend_opencl_reg()'],
      ['GGML_USE_ZENDNN',          '150', 'ggml_backend_zendnn_reg()'],
      ['GGML_USE_HEXAGON',         '153', 'ggml_backend_hexagon_reg()'],
      ['GGML_USE_CANN',            '156', 'ggml_backend_cann_reg()'],
      ['GGML_USE_BLAS',            '159', 'ggml_backend_blas_reg()'],
      ['GGML_USE_RPC',             '162', 'ggml_backend_rpc_reg()'],
      ['GGML_USE_OPENVINO',        '165', 'ggml_backend_openvino_reg()'],
      ['GGML_USE_ET',              '168', 'ggml_backend_et_reg()'],
      ['GGML_USE_CPU',             '171', 'ggml_backend_cpu_reg()']
    ];
    const t = U.table(['#ifdef 开关（16 个）', '构造函数行', '注册入口'], rows, { monoCols: [0, 1, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const trs = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '16 个开关，<span class="k">从上到下就是注册顺序</span>：CUDA 在最前（120 行），CPU 在最后（171 行）。',
      '前 4 个是 GPU 家族：<span class="v">CUDA</span> / <span class="v">METAL</span> / <span class="v">SYCL</span> / <span class="v">VULKAN</span>。<br>' +
        '注意 <span class="v">GGML_USE_CUDA</span> 这个宏在 HIP 构建（ROCm）和 MUSA 构建下也被打开 —— 见 L6-01。',
      '中间 6 个是各家加速器：<span class="v">WEBGPU</span> / <span class="v">ZDNN</span> / <span class="v">VIRTGPU</span> / <span class="v">OPENCL</span> / <span class="v">ZENDNN</span> / <span class="v">HEXAGON</span>。',
      '后 6 个：<span class="v">CANN</span>（昇腾 NPU，L7-01）/ <span class="v">BLAS</span> / <span class="v">RPC</span> / <span class="v">OPENVINO</span> / <span class="v">ET</span> / <span class="v">CPU</span>。',
      '宏是谁定义的？<span class="k">ggml/src/CMakeLists.txt:428-439</span>：只有 <span class="v">GGML_BACKEND_DL=OFF</span> 时才给 ggml 目标加这些宏。<br>' +
        '<span class="k">打开 DL 开关，这一整段 #ifdef 就全空了</span> —— 后端改由运行期加载（第 7 幕）。',
      '所以“机器上有哪些后端”有两个答案：<span class="k">编译期决定的静态集合</span> + 运行期追加的动态集合。'
    ];
    function light(i) { trs.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; }); }
    tl.at(700,  () => { light(0);  msg.innerHTML = texts[0]; });
    tl.at(3600, () => { light(3);  msg.innerHTML = texts[1]; });
    tl.at(6500, () => { light(8);  msg.innerHTML = texts[2]; });
    tl.at(9400, () => { light(13); msg.innerHTML = texts[3]; });
    tl.at(12300,() => { trs.forEach(r => { r.className = ''; }); msg.innerHTML = texts[4]; });
    tl.at(15500,() => { light(15); msg.innerHTML = texts[5]; });
    tl.at(19000,() => { trs.forEach(r => { r.className = ''; }); });
  }
},

/* ------------------------------------------------------ 3 注册 = <span class="hl-c">追加到末尾</span> + 去重 */
{
  kicker: "L3-02 · 注册动作",
  title: "注册 = <span class=\"hl-c\">追加到末尾</span> + 去重",
  sub: "register_backend() 把后端 push_back，再按它报的设备数逐个 push_back 设备。",
  caption: "实测（本机 CPU-only 构建，自建 harness 直接调 ggml_backend_register）：先注册名为 ZZZ 的假后端、再注册 AAA，得到 dev[1]=ZZZ0、dev[2]=AAA0 —— 顺序是注册顺序，不是字典序；重复注册同一个 reg，设备数不变。",
  src: "ggml/src/ggml-backend-reg.cpp",
  mark: [1, 5, 6, 15, 16, 17],
  lineNo: 186,
  code: `    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        if (!reg) {
            return;
        }

        for (auto & entry : backends) {
            if (entry.reg == reg) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" style="gap:9px">
        <div class="col grow" id="left" style="gap:6px"></div>
        <div class="col grow" id="right" style="gap:7px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const left = wrap.querySelector('#left');
    left.innerHTML = '<div class="cm" style="margin-bottom:2px">devices[] 的生长过程（本机实测）</div>';
    const steps = [
      { t: '初始',                 v: '[]' },
      { t: 'register(cuda_reg)',   v: '[CUDA0, CUDA1]' },
      { t: 'register(cpu_reg)',    v: '[CUDA0, CUDA1, CPU]' },
      { t: '再 register(cuda_reg)',v: '不变（去重）' },
      { t: 'register(zzz), register(aaa)', v: '[..., ZZZ0, AAA0]' }
    ];
    const els = steps.map(s => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
      e.innerHTML = '<span class="m">' + U.esc(s.t) + '</span> <span class="arrow">-></span> ' +
                    '<span style="color:var(--b)">' + U.esc(s.v) + '</span>';
      left.appendChild(e);
      return e;
    });
    els.forEach(e => e.style.opacity = '.28');

    const right = wrap.querySelector('#right');
    right.innerHTML =
      '<div class="card" style="border-left-color:var(--c)">' +
      '<div class="ct" style="color:var(--c)">追加，不插入</div>' +
      '<div class="cb">两个循环都只有 <span class="cm" style="margin:0">push_back</span>（201、203 行）。' +
      '没有排序、没有按名字或显存重排 —— <b>谁先注册，谁的下标就小</b>。</div></div>' +
      '<div class="card" style="border-left-color:var(--d)">' +
      '<div class="ct" style="color:var(--d)">后端可以去重，设备也可以</div>' +
      '<div class="cb">同一个 <span class="cm" style="margin:0">reg</span> 注册两次直接 return（191-195）；' +
      '同一个设备指针也去重（207-212，源码里的 <span class="cm" style="margin:0">register_device</span>）。</div></div>' +
      '<div class="card" style="border-left-color:var(--e)">' +
      '<div class="ct" style="color:var(--e)">可以注册成功、却贡献 0 个设备</div>' +
      '<div class="cb">设备循环的次数来自后端自报的 <span class="cm" style="margin:0">ggml_backend_reg_dev_count(reg)</span>。' +
      'RPC 后端的设备数来自它的 context（<span class="cm" style="margin:0">ggml-rpc.cpp:2273-2276</span>）：' +
      '还没加服务器时就是 0。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看第 187-189 行的空指针守卫，再看第 191-195 行的去重 —— 然后才是真正的注册。',
      '后端入列：<span class="v">backends.push_back({ reg, handle })</span>。',
      '设备入列：<span class="v">for (i &lt; ggml_backend_reg_dev_count(reg))</span> 逐个 <span class="v">register_device(...)</span>（202-204）。',
      '于是 <span class="k">一个后端在后端列表里的位置，决定了它的设备在设备列表里的位置</span>。',
      '这就是第 6 幕要用到的全部机制：<span class="k">devices[] 的下标 = 注册顺序</span>。'
    ];
    tl.at(700,  () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3700, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[1]; lightAll(); });
    tl.at(6700, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[2]; lightAll(); });
    tl.at(9700, () => { els[3].style.opacity = '1'; msg.innerHTML = texts[3]; lightAll(); });
    tl.at(12700,() => { els[4].style.opacity = '1'; msg.innerHTML = texts[4]; lightAll(); });
  }
},

/* ------------------------------------------------------ 4 <span class="hl-e">get_reg()</span>：注册发生在第一次触碰注册表时 */
{
  kicker: "L3-02 · 时机",
  title: "<span class=\"hl-e\">get_reg()</span>：注册发生在第一次触碰注册表时",
  sub: "函数内 static 单例；静态后端在构造函数里就位，动态加载只是随后补票。",
  caption: "实测：本机 CPU-only 构建里，调用 ggml_backend_load_all() 之前 ggml_backend_reg_count() 已经是 1（reg[0] = CPU）。",
  src: "ggml/src/ggml-backend-reg.cpp",
  mark: [0, 1, 7, 8],
  lineNo: 292,
  code: `static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
//>> 函数内 static：第一次调用时构造 —— “这台机器有哪些后端”就是在这一行被回答的
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="flow wrap" style="justify-content:center">
        <span class="chip">进程启动</span><span class="arrow">-></span>
        <span class="chip a">第一次调用 ggml_backend_*</span><span class="arrow">-></span>
        <span class="chip b">get_reg() 构造 registry</span><span class="arrow">-></span>
        <span class="chip c">16 个 #ifdef 注册静态后端</span><span class="arrow">-></span>
        <span class="chip d">load_all() 追加动态后端</span>
      </div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const defs = [
      { c: 'c', t: '注册表是懒的', b: '没有任何全局初始化函数：<br>第一次触碰才构造，构造即注册。' },
      { c: 'd', t: '空注册表的下场', b: 'src/llama.cpp:406-408 直接报错：<br>no backends are loaded' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">get_reg()</span> 返回一个函数内 <span class="k">static</span> 注册表（292-295）—— C++ 保证它只被构造一次。',
      '构造函数（第 2 幕那段 #ifdef）<span class="k">在任何动态加载之前</span>就想好了静态集合。<br>' +
        '本机实测：<span class="v">load_all()</span> 之前 <span class="v">reg_count()</span> 已经是 1。',
      '<span class="v">ggml_backend_register()</span> / <span class="v">ggml_backend_device_register()</span>（298-304）是内部 API：<br>' +
        '树外的后端（比如 RPC 服务器，<span class="v">common/arg.cpp:1180</span>）也能把自己塞进同一个注册表。',
      '于是“有哪些后端”= <span class="k">编译期静态集合</span> ∪ <span class="k">运行期加载的集合</span>，且前者总是先来。',
      '顺序也由此定死：<span class="k">静态后端永远排在动态后端前面</span>。'
    ];
    tl.at(700,  () => { msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3700, () => { msg.innerHTML = texts[1]; lightAll(); });
    tl.at(6700, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[2]; lightAll(); });
    tl.at(9700, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[3]; lightAll(); });
    tl.at(12700,() => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 设备枚举：<span class="hl-b">下标就是设备号</span> */
{
  kicker: "L3-02 · 设备枚举",
  title: "设备枚举：<span class=\"hl-b\">下标就是设备号</span>",
  sub: "ggml_backend_dev_get(i) 返回 devices[i]；顺序由注册顺序决定，与名字无关。",
  caption: "按名字查找走 by_name（大小写不敏感的比较在 307-314 的 striequals，返回第一个匹配的循环在 345-353）。",
  src: "ggml/src/ggml-backend-reg.cpp",
  mark: [0, 1, 4, 5, 6],
  lineNo: 336,
  code: `size_t ggml_backend_dev_count() {
    return get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    return get_reg().devices[index];
}`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="strip" style="gap:7px"></div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const strip = wrap.querySelector('#strip');
    const items = [
      { k: 'dev[0]', v: 'CPU',  c: 'a' },
      { k: 'dev[1]', v: 'ZZZ0', c: 'c' },
      { k: 'dev[2]', v: 'AAA0', c: 'd' }
    ];
    const els = items.map(it => {
      const e = U.el('div', { class: 'card', style: 'width:150px;border-left-color:var(--' + it.c + ')' });
      e.innerHTML = '<div class="cm" style="margin:0">' + U.esc(it.k) + '</div>' +
                    '<div class="ct" style="color:var(--' + it.c + ');font-size:14px">' + U.esc(it.v) + '</div>';
      strip.appendChild(e);
      return e;
    });
    strip.insertBefore(U.el('div', { class: 'cm', style: 'margin-right:2px' }), els[0]);
    strip.firstChild.innerHTML = '本机实测：<br>先注册 CPU，<br>再注册 ZZZ、AAA';
    els.forEach(e => e.style.opacity = '.30');

    const cards = wrap.querySelector('#cards');
    const defs = [
      { c: 'b', t: '下标从 0 开始数', b: 'dev_count() / dev_get(i) 就是<br>对 devices 向量的直接转发。' },
      { c: 'e', t: '名字不参与排序', b: 'AAA 排在 ZZZ 之后 ——<br>顺序只跟“谁先注册”有关。' }
    ];
    const cels = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); cards.appendChild(e); return e; });
    cels.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '设备枚举有两个入口：<span class="v">ggml_backend_dev_count()</span> 与 <span class="v">ggml_backend_dev_get(i)</span>。',
      '两者都只是转发到注册表的 <span class="v">devices.size()</span> / <span class="v">devices[index]</span> —— <span class="k">顺序在这里没有被重新计算的机会</span>。',
      '本机实测（第 3 幕那个 harness）：先注册的后端，它的设备下标就小。<br>名字叫 AAA 也不会被排到前面。',
      '<span class="v">ggml_backend_dev_by_type()</span>（355-363）与 <span class="v">by_name()</span>（345-353）都是<b>从头扫、返回第一个匹配</b>，<br>所以“同一个类型有多块设备时返回谁”，答案还是顺序。',
      '记住：<span class="k">这里的下标不是硬件编号，是注册顺序</span>。下一幕看它的后果。'
    ];
    tl.at(700,  () => { msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; lightAll(); });
    tl.at(6600, () => { els[1].style.opacity = '1'; els[2].style.opacity = '1'; msg.innerHTML = texts[2]; lightAll(); });
    tl.at(9600, () => { cels[0].style.opacity = '1'; msg.innerHTML = texts[3]; lightAll(); });
    tl.at(12600,() => { cels[1].style.opacity = '1'; msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 ★ 注册顺序怎么变成 <span class="hl-a">offload 结果</span> */
{
  kicker: "L3-02 · 后果",
  title: "★ 注册顺序怎么变成 <span class=\"hl-a\">offload 结果</span>",
  sub: "llama.cpp 按注册顺序把 GPU 型设备收进 model->devices；tensor-split / main-gpu 的下标就是它。",
  caption: "接着看 L2-05 逐字引用过的 src/llama-model.cpp:1521-1542：splits[] 经 upper_bound() 定出每层的设备，写进 dev_layer[il]。",
  src: "src/llama.cpp",
  mark: [0, 1, 3, 9],
  lineNo: 222,
  code: `            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
//>> 按注册顺序遍历设备 —— “设备 0/1/2”这个编号就是在这里被消费的
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow wrap" style="justify-content:center">
        <span class="chip a">devices[i]</span><span class="arrow">-></span>
        <span class="chip b">gpus[] 按遇到顺序 push_back</span><span class="arrow">-></span>
        <span class="chip c">model-&gt;devices</span><span class="arrow">-></span>
        <span class="chip d">main_gpu / tensor_split 下标</span><span class="arrow">-></span>
        <span class="chip e">dev_layer[il]</span>
      </div>
      <div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const defs = [
      { c: 'b', t: '① 收集', b: 'GPU 型设备按遇到的顺序 <span class="cm" style="margin:0">gpus.push_back</span>（254）；<br>RPC 设备单独收集。' },
      { c: 'c', t: '② 排序', b: 'model-&gt;devices = RPC 前置（276）+ GPU 追加（279）；<br>没有独显时才补 iGPU（283-285）。' },
      { c: 'd', t: '③ 消费', b: 'split-mode=none 只留 <span class="cm" style="margin:0">devices[main_gpu]</span>（297-299）；<br>layer/row 模式按这个下标切层。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.32');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '注册表给的顺序，第一个消费者就是 llama.cpp 的默认设备选择：<span class="k">从 i = 0 扫到 dev_count()-1</span>。',
      '<span class="v">case GGML_BACKEND_DEVICE_TYPE_GPU</span>（230）：GPU 型设备被收进 <span class="v">gpus</span>；<br>' +
        'CPU / ACCEL 型被跳过（225-228）—— 它们由 CPU buffer 列表单独处理（L2-05）。',
      '注意 269-270 行：这个循环里遇到 <span class="v">META</span> 型设备会 <span class="k">GGML_ABORT("fatal error")</span> —— ' +
        '因为 Meta 设备根本不在注册表里（第 8 幕）。',
      '接着：<span class="v">model->devices</span> 的顺序 = 注册表里 GPU 设备的顺序；<br>' +
        '<span class="v">--main-gpu N</span> 是 <span class="v">model->devices</span> 的下标（297-299），<span class="v">--tensor-split</span> 的第 j 项对应 <span class="v">devices[j]</span>（llama-model.cpp:1488-1508）。',
      '最后落到层：<span class="v">i_gpu_start</span> 划出上 GPU 的层，<span class="v">upper_bound(splits, ...)</span> 决定每层给哪块卡（llama-model.cpp:1529），' +
        '结果写进 <span class="v">dev_layer[il]</span>（1540-1542），权重张量再按它选 buffer 类型（L2-03）。',
      '所以：<span class="k">同一份 --tensor-split 3,1，在“设备顺序不同”的两台机器上，会把层分到不同的卡上</span>。',
      '顺序还会被写死的两张表改写：静态注册表与动态加载表顺序不同（下一幕），<span class="v">CANN</span> 与 <span class="v">CUDA</span> 就是一个真实的反例。'
    ];
    tl.at(700,  () => { msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; lightAll(); });
    tl.at(6600, () => { msg.innerHTML = texts[2]; lightAll(); });
    tl.at(9600, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[3]; lightAll(); });
    tl.at(12600,() => { els[2].style.opacity = '1'; msg.innerHTML = texts[4]; });
    tl.at(15600,() => { msg.innerHTML = texts[5]; });
    tl.at(18600,() => { msg.innerHTML = texts[6]; });
  }
},

/* ------------------------------------------------------ 7 ★ 动态加载只能<span class="hl-c">追加</span>到静态后端之后 */
{
  kicker: "L3-02 · 动态加载",
  title: "★ 动态加载只能<span class=\"hl-c\">追加</span>到静态后端之后",
  sub: "load_all_from_path() 按写死的顺序找 15 个名字的动态库；找到的都 push_back 到数组末尾。",
  caption: "动态库的 dlopen/符号解析/版本校验细节在 L3-03；本幕只关心它对“顺序”的影响。",
  src: "ggml/src/ggml-backend-reg.cpp",
  mark: [1, 6, 8, 12, 16, 22, 24, 27],
  lineNo: 578,
  code: `void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
//>> Release（NDEBUG）构建下 silent = true：加载失败的日志被吞掉
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("hexagon", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
//>> GGML_BACKEND_PATH：出树（out-of-tree）后端的入口，唯一由用户指定路径的加载
    if (backend_path) {
        ggml_backend_load(backend_path);
    }`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['', '静态注册（构造函数）', '动态加载（load_all）'],
      [['谁决定', '编译期宏 GGML_USE_*', '运行期磁盘上有什么 .so'],
       ['时机', '第一次触碰注册表时', 'llama_backend_init() 之后'],
       ['顺序', 'CUDA→METAL→SYCL→VULKAN…→CPU（120-171）', 'blas→zendnn→cann→cuda→hip→…→cpu（585-599）'],
       ['失败表现', '根本不存在这个后端', '静默跳过（Release 下 silent=true）'],
       ['进了哪个数组', 'backends[] / devices[] 的头部', '同一个数组的尾部（push_back）']],
      { monoCols: [] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const trs = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">ggml_backend_load_all()</span> 只是 <span class="v">load_all_from_path(nullptr)</span> 的转发（574-576）。',
      '搜索路径：<span class="v">GGML_BACKEND_DIR</span>（编译期，若定义）→ 可执行文件目录 → 当前目录（488-499）；<br>' +
        '每个路径下找 <span class="v">libggml-&lt;name&gt;-*.so</span>（464-478 的前后缀），按 <span class="v">ggml_backend_score()</span> 选分最高的（504-543）。',
      '候选名字 <span class="k">15 个</span>，顺序写死：blas, zendnn, cann, cuda, hip, metal, rpc, sycl, vulkan, virtgpu, opencl, hexagon, musa, openvino, cpu。<br>' +
        '（命令数出来：<span class="v">grep -c "ggml_backend_load_best(" ggml/src/ggml-backend-reg.cpp</span> = 15）',
      '★ 关键：<span class="k">这张表与构造函数的 #ifdef 顺序不是同一张表</span>。<br>' +
        '例如 <span class="v">CANN</span>：静态顺序在 CUDA 之后（156 vs 120），动态顺序在 CUDA 之前（587 vs 588）。',
      '而 <span class="v">register_backend()</span> 只做 push_back（201）—— 所以“同一个后端是编译进来的还是加载进来的”，<br>会改变它在 devices[] 里的下标。',
      '一个只在运行期出现的后端，<span class="k">不可能挤到静态后端前面</span>；要改设备编号，只能改编译期开关的组合。'
    ];
    function light(i) { trs.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; }); }
    tl.at(700,  () => { msg.innerHTML = texts[0]; light(0); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; light(1); });
    tl.at(6600, () => { msg.innerHTML = texts[2]; light(2); });
    tl.at(9600, () => { msg.innerHTML = texts[3]; light(2); });
    tl.at(12600,() => { msg.innerHTML = texts[4]; light(4); });
    tl.at(15600,() => { trs.forEach(r => { r.className = ''; }); msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 <span class="hl-e">Meta</span> backend：把 N 张卡合成一个设备 */
{
  kicker: "L3-02 · Meta",
  title: "<span class=\"hl-e\">Meta</span> backend：把 N 张卡合成一个设备",
  sub: "ggml_backend_meta_device() 造出来的设备不在注册表里：reg = nullptr。",
  caption: "它由调用方按需创建：张量并行（split-mode = tensor）时，llama.cpp 把枚举到的设备打包成一个 Meta 设备（src/llama.cpp:193-220）。",
  src: "ggml/src/ggml-backend-meta.cpp",
  mark: [1, 5, 6, 15, 23, 25],
  lineNo: 215,
  code: `ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
//>> 源码自己标注：这个函数不是线程安全的
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
//>> reg = nullptr —— 它不属于任何后端注册表项，因此永远不出现在 ggml_backend_dev_get() 的结果里
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const defs = [
      { c: 'a', t: '它是聚合', b: 'simple_devs 是入参那串设备（54-57）；<br>Meta 设备的全部行为都转发给它们。' },
      { c: 'b', t: '内存 = 相加', b: 'get_memory 把各设备的 free/total<br>累加（99-110）—— 报出来的是总和。' },
      { c: 'c', t: '能力 = 取交集', b: 'supports_op 用 <span class="cm" style="margin:0">all_of</span>（154-159）：<br>只有每个简单设备都支持，Meta 才支持。' },
      { c: 'd', t: '类型是 META', b: 'get_type 恒返回<br><span class="cm" style="margin:0">GGML_BACKEND_DEVICE_TYPE_META</span>（112-116）。' },
      { c: 'e', t: '同一组设备只造一次', b: '静态 map 去重（229-234）：<br>同样的设备组合返回同一个指针。' },
      { c: 'f', t: '谁在用', b: '张量并行需要“一个逻辑设备跨多张卡”，<br>上层算子不用知道几条流水线（L4-02）。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '第 8 幕起讲一个“不是注册表成员”的设备：<span class="v">Meta</span>。',
      '它由 <span class="v">ggml_backend_meta_device(devs, n_devs, ...)</span> 现场造出来（215-216），<br>' +
        '<span class="k">全文件没有任何 ggml_backend_register() 调用</span> —— 这是它和第 2 幕那些后端的根本区别。',
      '它内部的 <span class="v">ggml_backend_device</span> 结构体里 <span class="v">reg = nullptr</span>（237-241）：<br>设备存在，但不挂在任何后端项下。',
      '于是 <span class="v">ggml_backend_dev_count()</span> 永远数不到它（第 5 幕），<br>而 llama.cpp 的设备选择循环里遇到 META 型设备会 abort（src/llama.cpp:269-270）。',
      '它的价值：把“数据怎么切到多张卡”这件事<b>藏在一个设备后面</b>；<br>调度器仍然以为自己在跟一个设备打交道（L4-02）。拆开看是 <span class="v">simple_devs</span>。'
    ];
    tl.at(700,  () => { msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; lightAll(); });
    tl.at(6600, () => { els[3].style.opacity = '1'; els[4].style.opacity = '1'; msg.innerHTML = texts[2]; lightAll(); });
    tl.at(9600, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[3]; lightAll(); });
    tl.at(12600,() => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(15600,() => { els[5].style.opacity = '1'; msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 9 多后端共存时，设备名是谁起的 */
{
  kicker: "L3-02 · 命名",
  title: "多后端共存时，设备名是谁起的",
  sub: "注册表不发明名字：它只转手后端给的 get_name；Meta 设备的名字是拼出来的。",
  caption: "实例：CUDA 后端用 `GGML_CUDA_NAME + 序号`（ggml-cuda.cu:5798），该宏在 HIP 构建下是 \"ROCm\"、MUSA 下是 \"MUSA\"（ggml/include/ggml-cuda.h:11/14/17）；CPU 设备固定叫 \"CPU\"（ggml-cpu.cpp:353-357）；Metal 是 \"MTL\" + 序号（ggml-metal.cpp:307）。",
  src: "ggml/src/ggml-backend-meta.cpp",
  mark: [0, 1, 2, 6, 7, 11, 12],
  lineNo: 65,
  code: `        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
//>> 简单设备的名字按登记顺序用逗号连接 —— 名字里保留了“谁在左”的信息
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="strip" style="gap:8px"></div>
      <div class="row center" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const strip = wrap.querySelector('#strip');
    strip.innerHTML =
      '<span class="chip a">"CUDA0"</span><span class="arrow">,</span>' +
      '<span class="chip a">"CUDA1"</span><span class="arrow">-></span>' +
      '<span class="chip e" style="font-size:12px">"Meta(CUDA0,CUDA1)"</span>';
    const chipEls = strip.querySelectorAll('.chip');

    const cards = wrap.querySelector('#cards');
    const defs = [
      { c: 'b', t: '名字由后端自报', b: '注册表只用 <span class="cm" style="margin:0">ggml_backend_dev_name()</span> 转发<br>（215 行的日志、by_name 的查找）。' },
      { c: 'c', t: '查找：大小写不敏感、取第一个', b: '<span class="cm" style="margin:0">striequals</span>（307-314）+ 从头扫（345-353）：<br>重名时后面的设备永远查不到。' }
    ];
    const cels = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); cards.appendChild(e); return e; });
    cels.forEach(e => e.style.opacity = '.30');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'Meta 设备的名字不是常量，是构造出来的：<span class="v">"Meta(" + 名字 + "," + 名字 + ")"</span>。',
      '注意 name（65-72）与 description（66-73）是两份不同的字符串：<br>前者是标识符，后者给日志看。',
      '<span class="v">Meta(CUDA0,CUDA1)</span> 里的顺序 = 传给工厂函数的设备顺序 = <span class="k">注册表里的顺序</span>。' +
        'meta buffer type 也照同样的规则拼（256-268，用 buffer type 的名字）。',
      '普通后端同理：<span class="v">CUDA0</span> 这个名字来自 CUDA 后端自己（ggml-cuda.cu:5798），' +
        '<span class="k">它在注册表里的下标与名字里的 0 没有必然关系</span>。',
      '实践建议：要钉住某块卡，用名字而不是下标 —— <span class="v">--device</span> 内部就是 by_name（common/arg.cpp:1116-1136），<br>它不随注册顺序漂移。'
    ];
    tl.at(700,  () => { msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; lightAll(); });
    tl.at(6600, () => { cels[0].style.opacity = '1'; msg.innerHTML = texts[2]; lightAll(); });
    tl.at(9600, () => { chipEls.forEach(c => { c.style.opacity = '1'; }); msg.innerHTML = texts[3]; lightAll(); });
    tl.at(12600,() => { cels[1].style.opacity = '1'; msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 10 把这一课压成一张表 */
{
  kicker: "L3-02 · 收束",
  title: "把这一课压成一张表",
  sub: "三句话：有哪些后端看编译期；谁是设备 0 看注册顺序；Meta 设备不在注册表里。",
  caption: "下一课 L3-03 讲动态加载的细节：dlopen、符号解析、版本校验。",
  src: "ggml/src/ggml-backend-reg.cpp",
  mark: [1, 2, 3],
  lineNo: 382,
  code: `ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    // 点亮本幕 mark_src 标记的全部行；不传子集，避免把未选中的行 display:none 掉
    const lightAll = () => U.markLines(document, Array.from(document.querySelectorAll('#code mark.ln-mark'))
      .map(m => +m.getAttribute('data-l')));


    const t = U.table(
      ['问题', '答案', '依据'],
      [['机器上有哪些后端？', '编译期 16 个 GGML_USE_* 决定；运行期动态加载补充', 'reg.cpp:119-173 + CMakeLists.txt:428-439'],
       ['谁在什么时候注册？', '第一次触碰注册表时，构造函数里完成', 'reg.cpp:292-295'],
       ['devices[] 的顺序？', '注册顺序（静态在前、动态在后），只追加不重排', 'reg.cpp:201-204'],
       ['“第一个 GPU”是谁？', '注册顺序里第一个 GPU 型设备 → model->devices[0]', 'src/llama.cpp:222-231 / 254 / 276-279'],
       ['device 0 影响什么？', 'main-gpu 的下标、tensor-split 的第 0 项、每层落哪张卡', 'src/llama.cpp:297-299 + llama-model.cpp:1529'],
       ['设备名谁起？', '后端自己的 get_name；Meta 设备名是拼出来的', 'meta.cpp:65-77 / ggml-cuda.cu:5798'],
       ['Meta 设备在哪？', '不在注册表：工厂直接造，reg = nullptr', 'meta.cpp:215-245']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const trs = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      '一台机器上既有 NVIDIA 卡（CUDA 后端）又有昇腾 NPU（CANN 后端）。构建 A 把两个后端都静态编译进来，' +
      '构建 B 里两者都由运行期动态加载。两次运行都写 <span class="mono">--main-gpu 0</span>，会选中同一块硬件吗？' +
      '要稳定钉住某一块，应该怎么做？',
      '不一定 —— 关键在 <span class="mono">devices[]</span> 里两个后端的<b>先后</b>，而两张顺序表正好相反：<br>' +
      '· 静态（构建 A）：构造函数里 CUDA 在 120 行、CANN 在 156 行 → <span class="mono">[CUDA0, CANN0]</span>；<br>' +
      '· 动态（构建 B）：候选名字表里 cann 在 587 行、cuda 在 588 行 → <span class="mono">[CANN0, CUDA0]</span>。<br>' +
      '注册只做 <span class="mono">push_back</span>（<span class="mono">reg.cpp:201</span>），所以两种构建的设备下标是反的。' +
      'llama.cpp 按 <span class="mono">ggml_backend_dev_get(i)</span> 的顺序收集 GPU 型设备（<span class="mono">src/llama.cpp:222-231, 254</span>），' +
      '<span class="mono">--main-gpu N</span> 与 <span class="mono">--tensor-split</span> 的第 j 项都是 <span class="mono">model->devices</span> 的下标' +
      '（<span class="mono">297-299</span>、<span class="mono">llama-model.cpp:1488-1508</span>），最终决定每层落在哪块设备（<span class="mono">1529</span>）。<br>' +
      '稳定做法是用<b>名字</b>：<span class="mono">--device</span> 内部走 ' +
      '<span class="mono">ggml_backend_dev_by_name()</span>（<span class="mono">reg.cpp:345-353</span>，经 <span class="mono">common/arg.cpp:1127</span>），' +
      '而且给出的顺序就是 <span class="mono">model->devices</span> 的顺序（<span class="mono">src/llama.cpp:181-183</span>）。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先补一句本节引用的代码：<span class="v">ggml_backend_init_best()</span> 是按<b>类型</b>挑最合适的设备（GPU → IGPU → CPU），',
      '—— 这是“不想管顺序”时的兜底入口；而 llama.cpp 的默认设备列表是按<b>顺序</b>建的。两条路都存在，别混淆。',
      '一句话收束：<span class="k">编译期决定“有哪些”，注册顺序决定“谁是 0”</span>。',
      '而顺序规则很朴素：<span class="v">register_backend() 只做 push_back</span>，先静态后动态。',
      'L2-03 讲权重落位、L2-05 讲 dev_layer、L4-02 讲调度器怎么按设备列表切图 —— 它们都吃这一课的输出。'
    ];
    tl.at(700,  () => { msg.innerHTML = texts[0]; lightAll(); });
    tl.at(3700, () => { msg.innerHTML = texts[1]; lightAll(); });
    trs.forEach((r, i) => tl.at(6400 + i * 1600, () => {
      trs.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[2];
    }));
    tl.at(17800, () => { trs.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
    tl.at(19000, () => { msg.innerHTML = texts[4]; });
  }
},

];
