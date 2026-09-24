/* ==========================================================================
   L6-06 · Vulkan 后端：主机端
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 Vulkan 后端：主机端在做什么 */
{
  kicker: "L6 · GPU 后端执行",
  title: "Vulkan 后端：主机端在做什么",
  sub: "后端对外只有三张虚表；表后面是一条\"构建期编 shader、运行期建对象\"的流水线。",
  caption: "下一课 L6-07 讲着色器内部的 GLSL；本课讲它外面的主机端 C++。",
  src: "ggml/include/ggml-vulkan.h",
  mark: [0, 1, 4, 7, 9, 11, 13, 15],
  lineNo: 0,
  code: `#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16
//>> ---- ggml/include/ggml-vulkan.h:13-25 ----
// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);`,
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML =
      '<div class="flow" style="justify-content:center">' +
        '<span class="chip">模型定义</span><span class="arrow">-></span>' +
        '<span class="chip">ggml 图</span><span class="arrow">-></span>' +
        '<span class="chip">调度器（L4-02）</span><span class="arrow">-></span>' +
        '<span class="chip a">Vulkan 后端</span>' +
      '</div>' +
      '<div class="flow" id="files" style="justify-content:center;gap:5px"></div>' +
      '<div class="row" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const files = wrap.querySelector('#files');
    ['ggml-vulkan.cpp', 'ggml-vulkan-buffers.cpp', 'ggml-vulkan-debug.cpp',
     'ggml-vulkan-types.h', 'ggml-vulkan-common.h', 'ggml-vulkan-push-constants.h',
     'ggml/include/ggml-vulkan.h'].forEach(function (f, i) {
      files.appendChild(U.chip(f, i < 3 ? 'a' : (i < 6 ? 'b' : 'c')));
    });

    const defs = [
      { c: 'a', w: '224px', t: '主机端的活', b: '建实例/设备、建 pipeline、分配 descriptor set、填 push constants、挑内存类型', m: 'ggml-vulkan.cpp（16274 行）' },
      { c: 'b', w: '224px', t: '着色器端（L6-07）', b: 'GLSL 计算着色器：mul_mm / flash_attn / 各量化类型的反量化', m: 'vulkan-shaders/*.comp' },
      { c: 'c', w: '224px', t: '对外的门面', b: '5 个导出函数 + 2 个 buffer type，形状与 CUDA/SYCL 完全同构', m: 'ggml/include/ggml-vulkan.h' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(function (d) { const e = U.card(d); host.appendChild(e); return e; });
    els.forEach(function (e) { e.style.opacity = '.32'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '一个 ggml 节点走到这里，后端要回答三个问题：<span class="k">用哪个 pipeline</span>、' +
      '<span class="k">它的输入输出放哪几个 buffer</span>、<span class="k">这次 dispatch 的标量参数是多少</span>。',
      '本课后半段就按这三个问题展开：pipeline（第 4-5 幕）、descriptor set（第 7 幕）、push constants（第 8 幕）。',
      '但第 2-3 幕先回答一个更靠前的问题：<span class="v">pipeline 里的 SPIR-V 是哪来的</span>。' +
      '答案是构建期生成的，不是运行期编的。',
      '对外的门面只有 5 个函数 + 2 个 buffer type —— 与 L3-01 的后端契约一一对应，' +
      '真正的复杂度全在实现里。'
    ];
    defs.forEach(function (_, i) {
      tl.at(700 + i * 3300, function () {
        els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
        msg.innerHTML = texts[i];
      });
    });
    tl.at(13000, function () {
      els.forEach(function (e) { e.style.opacity = '1'; });
      msg.innerHTML = '本课覆盖 7 个文件：<span class="v">主机端 3 个 .cpp</span>、' +
        '<span class="v">4 个共享头</span>。';
    });
  }
},

/* ------------------------------------------------------ 2 ★ <span class="hl-a">Vulkan 只吃 SPIR-V</span>，所以 shader 必须在构建期编好 */
{
  kicker: "L6-06 · 核心",
  title: "★ <span class=\"hl-a\">Vulkan 只吃 SPIR-V</span>，所以 shader 必须在构建期编好",
  sub: "主机端能拿到的只有一段字节数组；GLSL 前端是一个外部工具，只存在于构建期。",
  caption: "把 .spv 逐字节嵌成数组那几行在 source.md 第二节；第 3 幕接着看构建规则。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
  mark: [1, 6, 9, 10, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 30, 37],
  lineNo: 0,
  code: `void string_to_spv_func(std::string name, std::string in_path, std::string out_path, std::map<std::string, std::string> defines, bool coopmat, bool dep_file, compile_count_guard slot) {
    std::string target_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #ifdef _WIN32
//>> ---- ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp:355-357 ----
    #else
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, in_path, "-o", out_path};
    #endif
//>> ---- ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp:440-443 ----
    if (input_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, out_path));
        return;
//>> ---- ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp:1398-1412 ----
    if (args.find("--glslc") != args.end()) {
        GLSLC = args["--glslc"]; // Path to glslc
    }
    if (args.find("--source") != args.end()) {
        input_filepath = args["--source"]; // The shader source file to compile
    }
    if (args.find("--output-dir") != args.end()) {
        output_dir = args["--output-dir"]; // Directory for containing SPIR-V output
    }
    if (args.find("--target-hpp") != args.end()) {
        target_hpp = args["--target-hpp"]; // Path to generated header file
    }
    if (args.find("--target-cpp") != args.end()) {
        target_cpp = args["--target-cpp"]; // Path to generated cpp file
    }
//>> ---- ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp:1421-1428 ----
    process_shaders();

    if (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::endl;
        return EXIT_FAILURE;
    }

    write_output_files();`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML =
      '<div class="row" id="steps" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'a', w: '163px', k: '1', t: '.comp', b: 'GLSL 计算着色器源码<br>（L6-07 逐字讲）', m: 'vulkan-shaders/*.comp' },
      { c: 'c', w: '163px', k: '2', t: 'glslc', b: '外部编译器：<br>compute 阶段 + target-env', m: 'GLSL -> SPIR-V' },
      { c: 'd', w: '163px', k: '3', t: '<name>.spv', b: 'SPIR-V 二进制，<br>只存在于构建目录', m: 'vulkan-shaders.spv/' },
      { c: 'b', w: '163px', k: '4', t: '<name>_data[]', b: '生成器把它写成 C 数组<br>+ _len 长度', m: '<name>.comp.cpp + .hpp' }
    ];
    const host = wrap.querySelector('#steps');
    const els = defs.map(function (d, i) {
      const e = U.el('div', { class: 'card', style: 'width:163px;border-left-color:var(--' + d.c + ')' });
      e.innerHTML = '<div class="cm" style="margin:0 0 2px">STEP ' + d.k + '</div>' +
        '<div class="ct" style="color:var(--' + d.c + ')">' + U.esc(d.t) + '</div>' +
        '<div class="cb">' + d.b + '</div>' +
        '<div class="cm">' + U.esc(d.m) + '</div>';
      host.appendChild(e);
      return e;
    });
    els.forEach(function (e) { e.style.opacity = '.30'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '第一步：<span class="k">.comp</span> 是 GLSL 源码，属于下一课的范围。',
      '第二步：生成器拼出 <span class="v">glslc</span> 命令行 —— ' +
      '<span class="k">-fshader-stage=compute</span> 指明这是计算着色器，<span class="k">target-env</span> 指明目标 Vulkan 版本。',
      '第三步：glslc 输出 <span class="v">.spv</span> —— 这就是 SPIR-V 字节码。' +
      '注意它落在构建目录，<span class="k">源码仓库里没有它</span>。',
      '第四步：生成器把 <span class="v">.spv</span> 逐字节读回来，写成 ' +
      '<span class="k">const unsigned char &lt;name&gt;_data[]</span> 和 <span class="k">&lt;name&gt;_len</span>' +
      '（source.md 第二节逐字引用）。',
      '这四个成员（glslc / source / output-dir / target 两个）就是生成器的全部输入 —— ' +
      '它一次只处理一个 .comp，由构建系统循环调用。',
      '<span class="hl-a">Vulkan 规范只要求驱动接受 SPIR-V</span>；GLSL 前端不在 API 里，' +
      '仓库里也没有任何"运行期把 GLSL 编成 SPIR-V"的调用。'
    ];
    defs.forEach(function (_, i) {
      tl.at(700 + i * 3400, function () {
        els.forEach(function (e, k) { e.style.opacity = k <= i ? '1' : '.30'; });
        msg.innerHTML = texts[i];
      });
    });
    tl.at(15200, function () {
      els.forEach(function (e) { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
    tl.at(18600, function () { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 3 ★ 构建系统把 <span class="hl-c">glslc</span> 变成一条编译规则 */
{
  kicker: "L6-06 · 构建期",
  title: "★ 构建系统把 <span class=\"hl-c\">glslc</span> 变成一条编译规则",
  sub: "每个 .comp 一条自定义命令；生成的 .cpp 进库，生成的 .hpp 被主机端头文件 include。",
  caption: "带 glslc / source / target 参数的那条完整命令在 source.md 第三节（CMake 变量语法放在散文区更清楚）。",
  src: "ggml/src/ggml-vulkan/CMakeLists.txt",
  mark: [0, 2, 3, 5, 8, 10, 14, 16, 18, 20, 22, 25, 30, 32],
  lineNo: 0,
  code: `find_package(Vulkan COMPONENTS glslc REQUIRED)
//>> ---- ggml/src/ggml-vulkan/CMakeLists.txt:209-213 ----
    set (_ggml_vk_genshaders_cmd "\${_ggml_vk_genshaders_dir}/vulkan-shaders-gen\${_ggml_vk_host_suffix}")
    set (_ggml_vk_header     "\${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp")
    set (_ggml_vk_input_dir  "\${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders")
    set (_ggml_vk_output_dir "\${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv")
    set (_ggml_vk_generated_shader_files \${_ggml_vk_header})
//>> ---- ggml/src/ggml-vulkan/CMakeLists.txt:227-229 ----
    add_custom_command(
        OUTPUT \${_ggml_vk_header}
        COMMAND \${_ggml_vk_genshaders_cmd}
//>> ---- ggml/src/ggml-vulkan/CMakeLists.txt:232-236 ----
        DEPENDS \${_ggml_vk_shaders_gen_sources}
                vulkan-shaders-gen
        COMMENT "Generate vulkan shaders header"
    )
    target_sources(ggml-vulkan PRIVATE \${_ggml_vk_header})
//>> ---- ggml/src/ggml-vulkan/CMakeLists.txt:238-245 ----
    foreach (file_full \${_ggml_vk_shader_files})
        get_filename_component(file \${file_full} NAME)
        set (_ggml_vk_target_cpp "\${CMAKE_CURRENT_BINARY_DIR}/\${file}.cpp")

        add_custom_command(
            OUTPUT  \${_ggml_vk_target_cpp}
            DEPFILE \${_ggml_vk_target_cpp}.d
            COMMAND \${_ggml_vk_genshaders_cmd}
//>> ---- ggml/src/ggml-vulkan/CMakeLists.txt:251-258 ----
            DEPENDS \${file_full}
                    \${_ggml_vk_shaders_gen_sources}
                    vulkan-shaders-gen
            COMMENT "Generate vulkan shaders for \${file}"
        )
        target_sources(ggml-vulkan PRIVATE \${_ggml_vk_target_cpp})
        list(APPEND _ggml_vk_generated_shader_files \${_ggml_vk_target_cpp})
    endforeach()`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['构建产物', '谁生成它', '给谁用'],
      [['vulkan-shaders-gen（宿主可执行）', '先于库编出来的宿主工具：两条命令都 DEPENDS 它', '被下面两条命令调用'],
       ['ggml-vulkan-shaders.hpp', '一条自定义命令（不带 source）：只声明全部变体', '被 ggml-vulkan-types.h 引入'],
       ['每个 .comp 对应一个 .cpp', '每个 .comp 一条自定义命令：调 glslc 出 .spv，再嵌成数组', '编进 ggml-vulkan 目标']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      'configure 阶段就要求 glslc 存在（<span class="k">find_package(Vulkan COMPONENTS glslc REQUIRED)</span>）——' +
      '它是<span class="v">构建依赖</span>，不是运行期依赖。',
      '生成器是<span class="v">先于库</span>编出来的宿主工具 —— 两条命令的 DEPENDS 里都点名它。',
      '头文件那条命令不带 source：生成器在 source 为空时只登记名字、不编译（上一幕的 ' +
      '<span class="k">ggml-vulkan-shaders-gen.cpp:440</span>）。',
      '每个 .comp 一条命令：参数就是上一幕生成器解析的那一组，' +
      '完整命令行见 <span class="v">source.md 第三节</span>。',
      '产物通过 <span class="k">target_sources</span> 挂到 ggml-vulkan 目标上：' +
      '程序员写的 .cpp 与生成的 .cpp 一起被编译、一起进库。'
    ];
    tl.at(700, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) {
      tl.at(3000 + i * 3200, function () {
        rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
        msg.innerHTML = texts[i + 1];
      });
    });
    tl.at(14200, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[4];
    });
    tl.at(17200, function () {
      msg.innerHTML = '所以：<span class="v">改一个 .comp 就是改一次构建输入</span> —— ' +
        '没有"运行期加载新内核"这条路。';
    });
  }
},

/* ------------------------------------------------------ 4 运行期只做三件事：<span class="hl-b">module</span> / <span class="hl-b">layout</span> / <span class="hl-b">pipeline</span> */
{
  kicker: "L6-06 · 运行期",
  title: "运行期只做三件事：<span class=\"hl-b\">module</span> / <span class=\"hl-b\">layout</span> / <span class=\"hl-b\">pipeline</span>",
  sub: "SPIR-V 是数据，不是代码：主机端把它交给驱动去创建对象，自己不编译。",
  caption: "注意 parameter_count 的断言：一个 pipeline 最多绑定 12 个 buffer（第 7 幕展开）。",
  src: "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
  mark: [0, 6, 7, 10, 12, 14, 17, 20, 21, 24, 33, 34],
  lineNo: 0,
  code: `static void ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_size, const void* spv_data, const std::string entrypoint,
                                         uint32_t parameter_count, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness, bool require_full_subgroups, uint32_t required_subgroup_size) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_count <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(parameter_count <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan.cpp:669-678 ----
    pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->push_constant_size
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan.cpp:743-749 ----
    try {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan.cpp:798-804 ----
    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = true;
        pipeline->compile_pending = false;
    }
    device->compile_cv.notify_all();`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row" id="steps" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'a', k: '1', t: 'ShaderModule', b: '把内嵌的 SPIR-V 数组<br>交给驱动', m: 'createShaderModule' },
      { c: 'c', k: '2', t: 'PipelineLayout', b: '一套 descriptor 布局<br>+ 一段 push constant 范围', m: 'createPipelineLayout' },
      { c: 'd', k: '3', t: 'Pipeline', b: '计算管线：<br>入口点固定为 main', m: 'createComputePipeline' },
      { c: 'b', k: '4', t: '登记缓存', b: '压进 all_pipelines，<br>置 compiled 并唤醒等待者', m: 'compile_cv.notify_all()' }
    ];
    const host = wrap.querySelector('#steps');
    const els = defs.map(function (d) {
      const e = U.el('div', { class: 'card', style: 'width:163px;border-left-color:var(--' + d.c + ')' });
      e.innerHTML = '<div class="cm" style="margin:0 0 2px">STEP ' + d.k + '</div>' +
        '<div class="ct" style="color:var(--' + d.c + ')">' + U.esc(d.t) + '</div>' +
        '<div class="cb">' + d.b + '</div>' +
        '<div class="cm">' + U.esc(d.m) + '</div>';
      host.appendChild(e);
      return e;
    });
    els.forEach(function (e) { e.style.opacity = '.30'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '函数签名里最关键的两个参数是 <span class="k">spv_size</span> 与 <span class="k">spv_data</span>：' +
      '运行期看到的是<span class="v">字节</span>，不是 GLSL 源码。',
      '第 1 步：<span class="v">ShaderModuleCreateInfo</span> 直接吃这段 uint32 数组。' +
      '此处还有一小段"打补丁"：按设备能力往 SPIR-V 里插 float control 能力位。',
      '第 2 步：pipeline layout = 设备的公共 descriptor 布局 + 这个 pipeline 的 push constant 范围（长度来自 ' +
      '<span class="k">push_constant_size</span>）。',
      '第 3 步：<span class="v">createComputePipeline</span> 若失败会打印 pipeline 名字再抛出 —— ' +
      '因为名字是唯一的定位线索。',
      '第 4 步：加锁登记 <span class="k">all_pipelines</span>、把 <span class="k">compiled</span> 置真并唤醒等待者。' +
      '第 5 幕看这套"谁在等"的机制。'
    ];
    defs.forEach(function (_, i) {
      tl.at(700 + i * 3300, function () {
        els.forEach(function (e, k) { e.style.opacity = k <= i ? '1' : '.30'; });
        msg.innerHTML = texts[i];
      });
    });
    tl.at(13800, function () {
      els.forEach(function (e) { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 5 pipeline 是<span class="hl-d">对象</span>，也是<span class="hl-d">缓存项</span> */
{
  kicker: "L6-06 · 缓存",
  title: "pipeline 是<span class=\"hl-d\">对象</span>，也是<span class=\"hl-d\">缓存项</span>",
  sub: "设备启动时一次建齐，运行期只查表；没编完的才补编，并用条件变量等它。",
  caption: "懒编译入口 ggml_pipeline_request_descriptor_sets()：未 compiled 就再调一次 ggml_vk_load_shaders（见 source.md 第三节）。",
  src: "ggml/src/ggml-vulkan/ggml-vulkan-types.h",
  mark: [0, 2, 3, 4, 5, 6, 7, 10, 13, 15, 24, 27, 29, 31],
  lineNo: 0,
  code: `struct vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t push_constant_size;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
    // true if fields have been set by ggml_vk_create_pipeline
    bool initialized {};
    // true while a compile is in flight, used to dedupe concurrent claims.
    // Protected by device->compile_mutex.
    bool compile_pending {};
    // set to true when the shader has been compiled
    std::atomic<bool> compiled {};
    // number of registers used, extracted from pipeline executable properties
    uint32_t register_count {};

#if defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#endif
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline next;
};
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan-types.h:791-791 ----
    std::map<vk_matmul_pipeline_key, std::vector<vk_matmul_pipeline_pair>> pipeline_matmul;
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan-types.h:798-801 ----
    vk_pipeline pipeline_dequant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_COUNT]; // fused dequant+transpose for FA quant-KV
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '一个 pipeline 的全部', b: 'shader_module / layout / pipeline 三个句柄，' +
          '外加 push_constant_size、parameter_count、wg_denoms、align', m: 'struct vk_pipeline_struct' },
      { c: 'c', t: '三态', b: 'initialized（字段填好没）<br>compile_pending（有人正在编）<br>compiled（可以用了）', m: 'bool + std::atomic<bool>' },
      { c: 'd', t: '缓存形式', b: '设备对象里成百个字段/映射：<br>矩阵乘用 map 以 (类型A,类型B,是否 MoE,是否 f16acc) 为键', m: 'device->pipeline_matmul / pipeline_dequant[]' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(function (d) { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(function (e) { e.style.opacity = '.32'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看这个结构体：它同时是"一个 Vulkan 对象的所有权"和"一个缓存项的状态"。',
      '<span class="k">shader_module / layout / pipeline</span>：三个句柄一起建、一起销毁（第 4 幕的三个步骤）。',
      '<span class="k">push_constant_size / parameter_count</span>：运行期要用它们做断言 —— ' +
      '给的 buffer 数、push constant 大小必须和建表时一致。',
      '三个状态位决定了并发行为：<span class="v">compile_pending</span> 去重，<span class="v">compiled</span> 是等待条件。',
      '缓存不是"按需新建"，而是<span class="k">预先建齐 + 惰性补齐</span>：设备初始化时就调一次 ' +
      '<span class="v">ggml_vk_load_shaders(device)</span>，运行期只是查表和等。',
      '这也解释了为什么 ops 的选择表（ggml_vk_op_get_pipeline 的大 switch）里只有"查"，没有"建"。'
    ];
    defs.forEach(function (_, i) {
      tl.at(700 + i * 3000, function () {
        els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
        msg.innerHTML = texts[i];
      });
    });
    tl.at(12200, function () {
      els.forEach(function (e) { e.style.opacity = '1'; });
      msg.innerHTML = texts[4];
    });
    tl.at(15600, function () { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 6 三张虚表，与 CUDA / SYCL <span class="hl-e">同构</span> */
{
  kicker: "L6-06 · 接口",
  title: "三张虚表，与 CUDA / SYCL <span class=\"hl-e\">同构</span>",
  sub: "L3-01 定义契约，各后端填表。Vulkan 的两个 buffer 面表就在 buffers.cpp 里。",
  caption: "表里的 CUDA / SYCL 行号来自本仓库直接检索，用于对照；本课不引用它们的源码（见文末说明）。",
  src: "ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp",
  mark: [0, 1, 2, 6, 9, 10, 12, 14, 19, 20],
  lineNo: 0,
  code: `ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp:745-756 ----
ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['虚表（L3-01 的契约）', 'Vulkan 实现处', '要点', 'CUDA / SYCL 的同名表'],
      [['ggml_backend_buffer_type_i', 'ggml-vulkan-buffers.cpp:3', '6 个槽位；is_host 显式 NULL', 'ggml-cuda.cu:927 / ggml-sycl.cpp:1055'],
       ['ggml_backend_buffer_i', 'ggml-vulkan-buffers.cpp:745', '11 个槽位；reset 显式 NULL', 'ggml-cuda.cu:853 / ggml-sycl.cpp:917'],
       ['ggml_backend_i', 'ggml-vulkan.cpp:14830', '16 个槽位；四个 graph_plan_* 显式 NULL', 'ggml-cuda.cu:4840 / ggml-sycl.cpp:6284']],
      { monoCols: [0, 1, 3] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '回顾 L3-01：ggml 后端就是三张函数指针表。表填好了，调度器就能用它执行图。',
      '<span class="k">buffer_type_i</span>：回答"这种 buffer 怎么分配、对齐多少、上限多少"。' +
      'Vulkan 的 <span class="v">is_host</span> 是 NULL —— 它不是主机内存。',
      '<span class="k">buffer_i</span>：回答"怎么往 buffer 里读/写张量"。Vulkan 的 ' +
      '<span class="v">reset</span> 是 NULL（没有"重置整块"的语义）。',
      '<span class="k">backend_i</span>：回答"怎么执行图"。Vulkan 不做图计划缓存（四个 graph_plan_* 为 NULL），' +
      '图执行入口是 <span class="v">graph_compute</span>。',
      '三张表的<span class="k">形状与 CUDA / SYCL 完全一致</span> —— 后端之间的差异在实现，不在契约。' +
      '这正是 L3-01 那一课的价值。'
    ];
    tl.at(700, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) {
      tl.at(2600 + i * 2900, function () {
        rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
        msg.innerHTML = texts[i + 1];
      });
    });
    tl.at(12800, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[4];
    });
  }
},

/* ------------------------------------------------------ 7 一套布局，一个池，<span class="hl-a">按需增长</span> */
{
  kicker: "L6-06 · descriptor set",
  title: "一套布局，一个池，<span class=\"hl-a\">按需增长</span>",
  sub: "所有 pipeline 共用同一个 descriptor set layout：12 个 storage buffer 槽位。",
  caption: "消费侧见 ggml-vulkan-common.h 的 ggml_vk_dispatch_pipeline 模板（source.md 第五节）。",
  src: "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
  mark: [0, 2, 3, 13, 17, 24, 25, 36, 37, 43, 46, 47],
  lineNo: 0,
  code: `        std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        for (uint32_t i = 0; i < MAX_PARAMETER_COUNT; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = { dsl_binding_flags };

        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
            {},
            dsl_binding);
        descriptor_set_layout_create_info.setPNext(&dslbfci);
        device->dsl = device->device.createDescriptorSetLayout(descriptor_set_layout_create_info);
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan.cpp:825-861 ----
void ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        return;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_set_requirements});
    uint32_t to_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_SIZE;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    while (to_alloc > 0) {
        const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_count;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

        if (pool_idx >= ctx->descriptor_pools.size()) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '一套布局', b: '按 MAX_PARAMETER_COUNT（=12）循环：每个 binding 都是 compute 阶段的 storage buffer', m: 'device->dsl' },
      { c: 'c', t: '一个池 256 个', b: '每池按 12 x 256 个 storage buffer 描述符申请', m: 'VK_DEVICE_DESCRIPTOR_POOL_SIZE' },
      { c: 'd', t: '按需增长', b: '需求先累加；不足时按"当前 1.5 倍"扩，一次分配一批', m: 'Grow by 50%' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(function (d) { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(function (e) { e.style.opacity = '.32'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看布局：<span class="k">12 个 binding，全是 storage buffer</span>，' +
      '于是任何 pipeline 的 descriptor 都可以用同一个 layout。',
      '第 4 幕里每个 pipeline layout 都带上了 <span class="v">device-&gt;dsl</span> —— ' +
      'layout 只有一份，pipeline 之间共享。',
      '分配是批量的：<span class="k">descriptor_sets</span> 是个向量，池按 256 个一组追加。',
      '每次请求把需求累加到 <span class="v">pipeline_descriptor_set_requirements</span>；' +
      '够用就立刻返回，不够才扩。',
      '扩张策略是"<span class="k">当前规模的 1.5 倍</span>取大者"，避免每来一个算子就分配一次。',
      '一次 dispatch 取走一个 set（<span class="v">descriptor_set_idx++</span>），' +
      '把这次的 buffer 写进去就绑 —— 下一幕看完整动作。'
    ];
    defs.forEach(function (_, i) {
      tl.at(700 + i * 3100, function () {
        els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
        msg.innerHTML = texts[i];
      });
    });
    tl.at(10500, function () { els.forEach(function (e) { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
    tl.at(14000, function () { msg.innerHTML = texts[4]; });
    tl.at(17200, function () { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 8 push constants：每个 op 一张 <span class="hl-f">POD 结构</span> */
{
  kicker: "L6-06 · push constants",
  title: "push constants：每个 op 一张 <span class=\"hl-f\">POD 结构</span>",
  sub: "形状、步长、标量参数都塞在这里；上限由源码的 static_assert 钉死。",
  caption: "数据缓冲走 descriptor set，标量参数走 push constants —— 两者在同一次 dispatch 里配齐。",
  src: "ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h",
  mark: [0, 1, 2, 4, 6, 9, 14, 19, 21, 23, 26, 35],
  lineNo: 0,
  code: `struct vk_op_binary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    float param1; float param2; int32_t param3;
};
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h:413-425 ----
struct vk_op_multi_add_push_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_COUNT][4];

    uint32_t rms_partials;
};

static_assert(MAX_PARAMETER_COUNT == 12);

static_assert(sizeof(vk_op_multi_add_push_constants) <= 256);
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h:958-972 ----
template <typename T> size_t push_constant_size(const T &t) {
    static_assert(std::is_class<T>::value, "T must be a struct/class");
    GGML_UNUSED(t);
    return sizeof(T);
}

template <typename T> size_t push_constant_size(const std::vector<T> &t) {
    GGML_UNUSED(t);
    return sizeof(T) * t.size();
}

template <typename T, uint32_t N> size_t push_constant_size(const std::array<T, N> &t) {
    GGML_UNUSED(t);
    return sizeof(T) * N;
}`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div class="row" id="cards" style="gap:8px"></div>' +
      '<div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const defs = [
      { c: 'f', t: '二元算子的那张', b: '一个 ne（元素个数）+ 三组 ne/nb（三张张量的形状与步长）' +
          ' + misalign_offsets + 三个标量参数', m: 'vk_op_binary_push_constants' },
      { c: 'a', t: '融合加法的那张', b: 'dst 的形状 + <b>nb[12][4]</b>：一次 dispatch 最多 12 张张量的步长', m: 'vk_op_multi_add_push_constants' },
      { c: 'b', t: '怎么变成参数字节', b: '两个模板：size 取 sizeof（或 vector/array 的长度），data 取地址（或 data()）', m: 'push_constant_size() / push_constant_data()' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(function (d) { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
    els.forEach(function (e) { e.style.opacity = '.32'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      'push constants 是 Vulkan 里"每次 dispatch 都要重写的一小段常量"，所以它按 op 逐个定义结构体。',
      '第一个静态断言把 <span class="k">MAX_PARAMETER_COUNT == 12</span> 写死在编译期 —— ' +
      '它就是第 7 幕 descriptor 槽位的数量。',
      '第二个断言给结构体封顶：<span class="v">sizeof(...) &lt;= 256</span>。' +
      'push constants 是稀缺资源（最少保证 128 字节），源码自己把它限制在 256。',
      '<span class="k">nb[12][4]</span> 是"多输入融合"的代价与红利：一次 dispatch 能带上 12 张张量的步长，' +
      '所以能把连续的 ADD/RMS 融合进同一次分派。',
      '两个模板函数的存在只为一件事：让 <span class="v">pushConstants(...)</span> 的实参在编译期就和 ' +
      '<span class="v">pipeline-&gt;push_constant_size</span> 对得上。',
      '记住分工：<span class="k">descriptor set = 数据在哪</span>，' +
      '<span class="k">push constants = 这次怎么算</span>。'
    ];
    defs.forEach(function (_, i) {
      tl.at(700 + i * 3100, function () {
        els.forEach(function (e, k) { e.style.opacity = k === i ? '1' : '.32'; });
        msg.innerHTML = texts[i];
      });
    });
    tl.at(10500, function () { els.forEach(function (e) { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
    tl.at(13800, function () { msg.innerHTML = texts[4]; });
    tl.at(16600, function () { msg.innerHTML = texts[5]; });
  }
},

/* ------------------------------------------------------ 9 一次 dispatch 的全部零件 */
{
  kicker: "L6-06 · 收束",
  title: "一次 dispatch 的全部零件",
  sub: "这个模板函数就是本课的落点：pipeline + descriptor set + push constants -> 一次分派。",
  caption: "下一课 L6-07 打开 pipeline 里的 SPIR-V：那些 .comp 到底怎么写。",
  src: "ggml/src/ggml-vulkan/ggml-vulkan-common.h",
  mark: [1, 2, 16, 17, 19, 22, 23, 25, 27, 29, 30, 37],
  lineNo: 248,
  code: `template <typename T>
inline void ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos, const T &push_constants, std::array<uint32_t, 3> elements) {
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
//>> workgroup 数 = 元素数 ÷ 每个 workgroup 覆盖的元素数（建 pipeline 时定下的 wg_denoms）
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    for (auto& buffer : descriptor_buffer_infos) {
        std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.maxComputeWorkGroupCount[0] &&
//>> 先断言不超过设备的分派上限 —— 超了要换 tile 配置，而不是硬发
                wg1 <= ctx->device->properties.limits.maxComputeWorkGroupCount[1] &&
                wg2 <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
    GGML_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size());
//>> pipeline 建表时声明要几个 buffer，这里就必须给几个：parameter_count 是契约
    GGML_ASSERT(pipeline->push_constant_size == push_constant_size(push_constants));
//>> push constant 大小也要和建表时一致：这就是那两个模板函数存在的理由

    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
    vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_count, vk::DescriptorType::eStorageBuffer, nullptr, descriptor_buffer_infos.begin() };
//>> 一次写入把 parameter_count 个 storage buffer 挂到 set 的 0 号 binding 起
    ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});

    subctx->s->buffer->buf.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, push_constant_size(push_constants), push_constant_data(push_constants));
//>> push constants 单独走一条路径：它不属于 descriptor set
    subctx->s->buffer->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_set },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->buffer->buf.dispatch(wg0, wg1, wg2);
//>> 到这里才真正分派：前面的都是准备动作
    }
}`,
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = '<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>';
    root.appendChild(wrap);

    const t = U.table(
      ['零件', '谁创建、什么时候', '每次 dispatch 是否变化'],
      [['pipeline（含 SPIR-V module）', '设备初始化时 ggml_vk_load_shaders 建齐；缺的惰性补齐', '不变（首次补齐）'],
       ['pipeline layout / descriptor set layout', 'device->dsl：一套布局给所有 pipeline 共享', '不变'],
       ['descriptor set', '从池里取一个，写入这次的 parameter_count 个 buffer', '每次变'],
       ['push constants', '按 op 的 POD 结构体填好，整块推给驱动', '每次变'],
       ['workgroup 数量', 'elements ÷ wg_denoms，向上取整', '每次变']],
      { monoCols: [0] });
    wrap.querySelector('#tbl').appendChild(t.el);
    const rows = t.body.querySelectorAll('tr');

    wrap.querySelector('#ex').appendChild(W.exercise(
      'Vulkan 后端为什么必须在构建期把 <span class="mono">.comp</span> 预编译成 SPIR-V？' +
      '若要让 Vulkan 后端支持一个新的算子内核，最少要动哪几处？',
      '因为主机端只拿得到 <span class="mono">spv_data</span> / <span class="mono">spv_size</span> 两个值' +
      '（<span class="mono">ggml-vulkan.cpp:554</span>、<span class="mono">:564</span>），' +
      '仓库里没有任何"运行期把 GLSL 编成 SPIR-V"的路径；GLSL 前端是外部工具 ' +
      '<span class="mono">glslc</span>，由构建期调用（<span class="mono">vulkan-shaders-gen.cpp:350</span>、' +
      '<span class="mono">CMakeLists.txt:242</span>）。<br>' +
      '最少三处：① 写 <span class="mono">vulkan-shaders/&lt;op&gt;.comp</span>；' +
      '② 在生成器的 <span class="mono">process_shaders()</span> 里用 ' +
      '<span class="mono">string_to_spv()</span> 登记这个变体（否则生成的 .hpp 里没有它的 ' +
      '<span class="mono">_data</span> / <span class="mono">_len</span>）；' +
      '③ 在 <span class="mono">ggml-vulkan.cpp</span> 的 <span class="mono">ggml_vk_load_shaders()</span> 里' +
      '为它创建 pipeline，并在 <span class="mono">ggml_vk_op_get_pipeline()</span> 的选择表里把 op 映射过去。'));

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把本课压成一张表：三种零件，各自的创建时机与"是否每次变化"。',
      '<span class="k">pipeline 与 layout 是"不变的"</span>：构建期把 SPIR-V 编好，运行期只建一次对象。',
      '<span class="k">descriptor set 与 push constants 是"每次变的"</span>：前者换 buffer，后者换标量。',
      '这解释了为什么第 7 幕的 set 要按需增长、第 8 幕的 push constants 要按 op 分结构体。',
      '下一课 L6-07 打开 pipeline 里的 SPIR-V：<span class="v">mul_mm / mul_mmq / flash_attn</span> ' +
      '这些 .comp 怎么写、宏怎么展开。'
    ];
    tl.at(700, function () { msg.innerHTML = texts[0]; });
    rows.forEach(function (r, i) {
      tl.at(2600 + i * 2600, function () {
        rows.forEach(function (x, k) { x.className = (k === i) ? 'on' : ''; });
        msg.innerHTML = texts[Math.min(i + 1, 3)];
      });
    });
    tl.at(16000, function () {
      rows.forEach(function (x) { x.className = ''; });
      msg.innerHTML = texts[4];
    });
  }
},

];
