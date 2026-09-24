/* ==========================================================================
   L6-07 · Vulkan 后端：计算着色器
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 <span class="hl-a">161 个 .comp</span>：Vulkan 的算子实现全在这里 */
{
  kicker: "L6 · GPU 后端执行",
  title: "<span class=\"hl-a\">161 个 .comp</span>：Vulkan 的算子实现全在这里",
  sub: "先按名字分族数一遍。族名就是 ggml op 的名字 —— 这是本仓库最一致的命名约定之一。",
  caption: "计数命令：ls ggml/src/ggml-vulkan/vulkan-shaders/ 下的 .comp（含 feature-tests/ 子目录）= 161。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
  mark: [0, 6, 10, 13, 25, 29],
  lineNo: 49,
  code: `const std::vector<std::string> type_names = {
//>> 类型清单：每个名字都会在每个着色器族里派生出若干变体
    "f32",
    "f16",
    "q1_0",
    "q2_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "nvfp4",
    "tq1_0",
    "tq2_0",
    "bf16",
//>> bf16 在最后 —— 注释没有写"只能追加"，但这个顺序是稳定约定
};`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const rows = [
      ['反量化 dequant_*', '26', 'dequant_q4_0.comp', '反量化成 f16（MUL_MAT 的 f16 路径前置）'],
      ['矩阵乘 mul_mat_vec*', '19', 'mul_mat_vec_q4_k.comp', 'GGML_OP_MUL_MAT（小 M）/ MUL_MAT_ID（MoE）'],
      ['矩阵乘 mul_mm* / mul_mmq', '4', 'mul_mm.comp', 'GGML_OP_MUL_MAT（大 M）；split-K 归约'],
      ['注意力 flash_attn*', '8', 'flash_attn.comp', 'GGML_OP_FLASH_ATTN_EXT'],
      ['soft_max*', '5', 'soft_max.comp', 'GGML_OP_SOFT_MAX'],
      ['归一化（rms_norm* / norm / l2_norm / group_norm）', '6', 'rms_norm.comp',
       'RMS_NORM / NORM / L2_NORM / GROUP_NORM'],
      ['rope_*', '4', 'rope_norm.comp', 'GGML_OP_ROPE'],
      ['逐元素 / 激活族', '26', 'unary.comp', 'UNARY / GLU / ADD / MUL / SCALE ...'],
      ['索引 / 排序 / 归约族', '17', 'topk_moe.comp', 'TOP_K / ARGSORT / GET_ROWS / SUM_ROWS / CUMSUM'],
      ['搬运 / 拷贝族', '13', 'copy.comp', 'DUP / CPY / CONT / PAD / REPEAT / ROLL'],
      ['SSM / 线性注意力族', '10', 'ssm_scan.comp', 'SSM_SCAN / SSM_CONV / RWKV_WKV6 / GATED_DELTA_NET'],
      ['卷积 / 池化 / 图像族', '9', 'conv2d_mm.comp', 'CONV_2D / CONV_2D_DW / CONV_3D / IM2COL / POOL_2D'],
      ['feature-tests/*', '7', 'coopmat2.comp', '不是算子：编译期探测设备能力'],
      ['优化器 / 其他', '7', 'opt_step_adamw.comp', 'OPT_STEP_ADAMW / OPT_STEP_SGD / OUT_PROD']
    ];
    const t = U.table(['族（前缀）', '个数', '代表 .comp', '实现哪个 ggml op'], rows,
                      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const trs = t.body.querySelectorAll('tr');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '161 个 <span class="v">.comp</span> 按前缀分成 14 个族。' +
      '最大的族是 <span class="k">dequant_*</span>（26）—— 每个量化类型一个。',
      '三个矩阵乘族加起来 23 个：<span class="k">mul_mm*</span> 管大 M、' +
      '<span class="k">mul_mat_vec*</span> 管小 M、<span class="k">mul_mmq</span> 管整数点积。',
      '注意力与归一化：<span class="k">flash_attn*</span>（8）里有 cm1/cm2/decode/split-k 等变体，' +
      '<span class="k">rms_norm</span> 一个文件还要管三种融合。',
      '其余 89 个是逐元素、索引、搬运、SSM、卷积、优化器 —— ' +
      '命名约定一致：<span class="v">族名即 ggml op 名</span>。'
    ];
    const groups = [[0, 1, 2], [3, 4, 5, 6], [7, 8], [9, 10, 11, 12, 13]];
    groups.forEach((g, i) => tl.at(700 + i * 3600, () => {
      trs.forEach((r, k) => { r.className = g.indexOf(k) >= 0 ? 'on' : ''; });
      msg.innerHTML = texts[i];
    }));
    tl.at(15600, () => {
      trs.forEach(r => { r.className = ''; });
      msg.innerHTML = '一句话：<span class="k">一个 ggml op 在 Vulkan 上 = 一个 .comp 族</span>。';
    });
  }
},

/* ------------------------------------------------------ 2 ★ 一个 <span class="hl-c">.comp</span> 是<span class="hl-a">模板</span>：<span class="hl-a">string_to_spv</span> 是变体工厂 */
{
  kicker: "L6-07 · 核心",
  title: "★ 一个 <span class=\"hl-c\">.comp</span> 是<span class=\"hl-a\">模板</span>：<span class=\"hl-a\">string_to_spv</span> 是变体工厂",
  sub: "验收点：说出一个 .comp 从生成到被 pipeline 使用经过哪几步。这 16 行是第 2、3 步。",
  caption: "注意第 444 行：进程只编译与自己文件名匹配的那个 .comp —— CMake 为每个 .comp 起一个生成器进程。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
  mark: [0, 1, 3, 10, 17],
  lineNo: 436,
  code: `void string_to_spv(std::string name, const std::string& source, const std::map<std::string, std::string>& defines, bool fp16 = true, bool coopmat = false, bool coopmat2 = false, bool f16acc = false, const std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
//>> 变体名 = 名字 + 精度后缀（_f16acc / _cm1 / _cm2 / _fp32）+ 自定义 suffix
    std::string out_path = join_paths(output_dir, name + ".spv");
//>> 输出路径 = <output_dir>/<变体名>.spv —— 一个变体一个 SPIR-V 文件

    if (input_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, out_path));
        return;
    } else if (basename(input_filepath) != source) {
//>> 只编译与输入文件名匹配的变体；实际编译的是 CMake 为每个 .comp 起的那个进程
        // Only compile shader variants matching the input filename
        return;
    }

    compile_count_guard slot = acquire_compile_slot();
    compiles.push_back(std::async(
//>> 异步扔进线程池：成千上万个变体是并发编译出来的
        string_to_spv_func, name, input_filepath, out_path, defines, coopmat, generate_dep_file, std::move(slot)));`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: '1 · process_shaders()', b: '遍历 (着色器 × 类型 × 变体)：<br>f32/f16 × 量化类型 × coopmat 档', m: 'process_shaders()' },
      { c: 'c', t: '2 · string_to_spv()', b: '拼出<b>变体名</b>与 <b>.spv 输出路径</b>；<br>同一份 .comp 因此有几百个 SPIR-V', m: 'name + ".spv"' },
      { c: 'e', t: '3 · glslc', b: '把每个 <b>-D 宏</b>展开成命令行参数，<br>异步编译出 .spv', m: 'glslc ... -o name.spv' },
      { c: 'd', t: '4 · write_output_files()', b: '读回 .spv 字节，写成<br><b>&lt;name&gt;_data[]</b> 与 <b>&lt;name&gt;_len</b>', m: 'const unsigned char ..._data[]' },
      { c: 'b', t: '5 · 主机端建 pipeline', b: 'createShaderModule(spv_data)<br>→ vkCreateComputePipelines', m: 'ggml_vk_create_pipeline_func()' },
      { c: 'f', t: '6 · 运行时查表 dispatch', b: '按 (权重类型, 列数, workgroup 档)<br>从 pipeline 表里取，然后 dispatch', m: 'ggml_vk_dispatch_pipeline()' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先记住这六步。左边代码是第 <span class="k">2</span> 步：' +
      '<span class="v">string_to_spv()</span> 把 (着色器, 宏定义) 变成 (变体名, .spv 路径)。',
      '第 <span class="k">1</span> 步在上游：<span class="v">process_shaders()</span> 决定要哪些变体。' +
      '同一份 <span class="v">mul_mm.comp</span> 会被点名几十次。',
      '第 <span class="k">2</span> 步的 <span class="v">name</span> 后缀是真信息：' +
      '<span class="v">_cm1</span>=coopmat、<span class="v">_cm2</span>=coopmat2、' +
      '<span class="v">_f16acc</span>=f16 累加、<span class="v">_fp32</span>=fp32 累加。',
      '第 <span class="k">3</span> 步：<span class="v">defines</span> 里的每一项都变成命令行上的一个宏。' +
      '所以"一个 .comp"其实是一个<b>参数化的家族</b>。',
      '第 <span class="k">4</span> 步：<span class="v">.spv</span> 是二进制，落地成 C 数组才能编进静态库 —— ' +
      '这一步让 Vulkan 后端<b>不需要运行时编译着色器</b>。',
      '第 <span class="k">5、6</span> 步在主机端（<span class="v">ggml-vulkan.cpp</span>，L6-06 的地盘）：' +
      '字节数组 → shader module → compute pipeline → 按类型查表 dispatch。'
    ];
    defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
      els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(18800, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '六步一句话：<span class="k">.comp --(defines)--> .spv --(C 数组)--> pipeline --(查表)--> dispatch</span>';
    });
  }
},

/* ------------------------------------------------------ 3 宏定义变成 <span class="hl-c">glslc</span> 命令行 */
{
  kicker: "L6-07 · 生成链（第 3 步）",
  title: "宏定义变成 <span class=\"hl-c\">glslc</span> 命令行",
  sub: "这一步没有任何魔法：一个 define 一个 -D，最后 -o 出 .spv（下面是非 Windows 分支）。",
  caption: "上一节 source.md 给出完整的 350-384 行：第 351 行按变体名决定目标环境，第 382 行把每个宏变成一个 -D。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
  mark: [1, 9, 11, 15],
  lineNo: 355,
  code: `    #else
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, in_path, "-o", out_path};
//>> 整条命令在这里拼出来：阶段 + 目标环境 + 输入 + 输出
    #endif

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    if (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
//>> 四类变体跳过 spirv-opt：注释逐条给了上游 issue 编号
        cmd.push_back("-O");
//>> -O 只在安全时加 —— 这是一处"优化会改坏着色器"的现场记录
    }

    if (dep_file) {
//>> 同一个函数还顺带产出依赖文件（给构建系统用）
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#ifdef _WIN32`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="flow" id="cmd" style="justify-content:center"></div>
      <div class="row" id="cards" style="gap:9px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const cmd = wrap.querySelector('#cmd');
    const parts = [
      { t: 'glslc', c: 'a' },
      { t: '-fshader-stage=compute', c: '' },
      { t: 'target-env', c: 'c' },
      { t: '-D DATA_A_Q4_K=1', c: 'e' },
      { t: '-D LOAD_VEC_A=4', c: 'e' },
      { t: '-D FLOAT_TYPE=float16_t', c: 'e' },
      { t: '-D ...', c: 'e' },
      { t: '-o mul_mm_q4_k_f16.spv', c: 'b' }
    ];
    const chips = parts.map(p => { const e = U.chip(p.t, p.c); cmd.appendChild(e); return e; });
    chips.forEach(e => e.style.opacity = '.25');

    const defs = [
      { c: 'e', t: 'defines 是谁给的？', b: '上游 <span class="cm" style="margin:0">matmul_shaders()</span> 里的 <b>merge_maps(...)</b>：' +
          '基础字典（FLOAT_TYPE 系列）+ 类型键（DATA_A_Q4_K）+ 变体键（MULMAT_QUANT 等）。' },
      { c: 'c', t: '目标环境由变体名决定', b: '名字里含 <b>_cm2</b> 就按 Vulkan 1.3 编译，否则 1.2。' +
          'coopmat2 的着色器用到 1.3 才有的能力。' },
      { c: 'd', t: '依赖文件与调试信息', b: '可选地让 glslc 同时产出 <b>.d</b>（给 CMake 用）与 <b>调试信息</b>；' +
          '都由生成器的编译期开关控制。' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.3');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先把命令拼出来：<span class="v">GLSLC</span> 默认就是 <span class="k">glslc</span>（可用命令行覆盖）。',
      '<span class="v">-fshader-stage=compute</span>：所有 .comp 都是 <b>compute</b> 阶段 —— ' +
      '没有顶点/片元着色器。',
      '接下来是 <span class="k">target-env</span>：由变体名里有没有 <span class="v">_cm2</span> 决定。',
      '然后是 <span class="k">一长串 -D</span>：这就是"同一个 .comp 编译出几百个 SPIR-V"的全部机制。',
      '可选开关：<span class="k">依赖文件</span>与<span class="k">调试信息</span>，' +
      '还有去掉 spirv-opt 的那个分支（第 363 行）。',
      '最后 <span class="v">-o &lt;变体名&gt;.spv</span>。命令拼好后交给 <span class="k">execute_command</span> 跑。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    chips.forEach((c, i) => tl.at(2400 + i * 1500, () => {
      chips.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.25'; });
      msg.innerHTML = texts[Math.min(i, 5)];
    }));
    tl.at(15600, () => {
      els.forEach(e => e.style.opacity = '1');
      msg.innerHTML = '一句话：<span class="k">变体名 + defines 决定命令行，命令行决定 .spv</span>。';
    });
  }
},

/* ------------------------------------------------------ 4 <span class="hl-b">.spv</span> 变成 C 数组：<span class="hl-a">&lt;name&gt;_data[]</span> / <span class="hl-a">&lt;name&gt;_len</span> */
{
  kicker: "L6-07 · 生成链（第 4 步）",
  title: "<span class=\"hl-b\">.spv</span> 变成 C 数组：<span class=\"hl-a\">&lt;name&gt;_data[]</span> / <span class=\"hl-a\">&lt;name&gt;_len</span>",
  sub: "这一步之后，SPIR-V 就是普通的目标文件内容 —— 跟着 ggml-vulkan 一起被链接。",
  caption: "对应的 extern 声明写在 <name>_data[] 头文件里（第 1252-1253 行），主机端 include 它就能用。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
  mark: [0, 2, 5, 11, 13, 17],
  lineNo: 1252,
  code: `        hdr << "extern const uint64_t " << name << "_len;\\n";
//>> 头文件里只放 extern 声明：<name>_len 与 <name>_data[]
        hdr << "extern const unsigned char " << name << "_data[];\\n\\n";

        if (input_filepath != "") {
            std::string data = read_binary_file(path);
//>> 把上一步 glslc 产出的 .spv 读回内存
            if (data.empty()) {
                continue;
            }

            src << "const uint64_t " << name << "_len = " << data.size() << ";\\n";
//>> 字节数写成 <name>_len
            src << "const unsigned char " << name << "_data[" << data.size() << "] = {\\n" << std::hex;
//>> 整个 SPIR-V 二进制逐字节写成 C 数组 <name>_data[]
            auto bytes = reinterpret_cast<const uint8_t*>(data.data());
            for (size_t i = 0; i < data.size(); ++i) {
                src << "0x" << static_cast<int>(bytes[i]) << ",";
                if ((i + 1) % 12 == 0) src << "\\n";
            }
            src << std::dec << "\\n};\\n\\n";
        }
    }`,
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const dia = wrap.querySelector('#dia');

    const left = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--c)' });
    left.innerHTML = '<div class="ct" style="color:var(--c)">磁盘上的中间产物</div>' +
      '<div class="cb">每个变体一个文件：<br>' +
      '<span class="cm" style="margin:0">&lt;output_dir&gt;/&lt;变体名&gt;.spv</span><br>' +
      '名字里带着类型与精度后缀，例如某个 <b>mul_mat_vec_q4_k</b> 变体。</div>' +
      '<div class="cm" id="spv" style="margin-top:5px">.spv = SPIR-V 二进制</div>';

    const right = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
    right.innerHTML = '<div class="ct" style="color:var(--b)">生成的 C 源文件</div>' +
      '<div class="cb">每个变体两行：<br>' +
      '<span class="cm" style="margin:0">const uint64_t &lt;name&gt;_len = ...;</span><br>' +
      '<span class="cm" style="margin:0">const unsigned char &lt;name&gt;_data[N] = {...};</span><br>' +
      '编进静态库，<b>不需要运行时编译着色器</b>。</div>' +
      '<div class="cm" id="arr" style="margin-top:5px">逐字节写成 0x.. 列表</div>';

    dia.appendChild(left); dia.appendChild(U.arrow('->')); dia.appendChild(right);
    left.style.opacity = '.35'; right.style.opacity = '.35';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">write_output_files()</span> 遍历 <span class="k">shader_fnames</span> —— ' +
      '第 2 幕里 <span class="v">string_to_spv()</span> 每登记一个变体，这里就产出一对符号。',
      '第 <span class="k">1256</span> 行把 <span class="v">.spv</span> 读回来；' +
      '<span class="v">read_binary_file()</span> 读的是上一步 glslc 的产物。',
      '第 <span class="k">1261-1262</span> 行写出两个符号：<span class="v">&lt;name&gt;_len</span> 与 ' +
      '<span class="v">&lt;name&gt;_data[]</span>。',
      '头文件（第 1252-1253 行）只放 <span class="k">extern 声明</span>：' +
      '主机端 <span class="v">#include</span> 它就能引用到全部变体。',
      '这就是第 2 幕第 5 步的输入：<span class="v">ggml_vk_create_pipeline_func()</span> 收到 ' +
      '<span class="v">spv_size</span> 与 <span class="v">spv_data</span>，建 shader module。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    tl.at(3600, () => { left.style.opacity = '1'; msg.innerHTML = texts[1]; });
    tl.at(6600, () => { msg.innerHTML = texts[2]; });
    tl.at(9600, () => { right.style.opacity = '1'; msg.innerHTML = texts[3]; });
    tl.at(12600, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 5 ★ 量化块定义：<span class="hl-a">GLSL</span> 与 <span class="hl-e">C</span> 是<span class="hl-d">两份</span>定义 */
{
  kicker: "L6-07 · 核心",
  title: "★ 量化块定义：<span class=\"hl-a\">GLSL</span> 与 <span class=\"hl-e\">C</span> 是<span class=\"hl-d\">两份</span>定义",
  sub: "形状完全一样，但写在两个文件里、由两套代码各自维护 —— 这是本课最值得记住的一件事。",
  caption: "对照 source.md 第九节：CPU 侧 ggml-common.h 的 block_q4_0 逐字引用。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/types.glsl",
  mark: [0, 2, 4, 7, 9, 13, 15, 20, 23, 24],
  lineNo: 56,
  code: `#define QUANT_K_Q4_0 32
//>> 块大小 32：与 CPU 的 QK4_0 完全一致
#define QUANT_R_Q4_0 2

struct block_q4_0
//>> GLSL 的块结构体：一个 f16 scale + 16 字节 nibble
{
    float16_t d;
//>> float16_t 在这里是"存储类型"，不是计算精度
    uint8_t qs[16];
};
struct block_q4_0_packed16
{
    float16_t d;
//>> packed16 视图：同一块内存，按 uint16 读，方便一次取 4 个 nibble
    uint16_t qs[16/2];
};

#if defined(DATA_A_Q4_0)
//>> 这一组宏把"类型号"变成 QUANT_K / A_TYPE，供整个着色器复用
#define QUANT_K QUANT_K_Q4_0
#define QUANT_R QUANT_R_Q4_0
#define QUANT_AUXF 1
#define A_TYPE block_q4_0
#define A_TYPE_PACKED16 block_q4_0_packed16
#define DATA_A_QUANT_LEGACY
#endif`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);
    const dia = wrap.querySelector('#dia');

    const glsl = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
    glsl.innerHTML = '<div class="ct" style="color:var(--a)">GLSL 侧 · types.glsl</div>' +
      '<div class="cb">' +
      '<span class="cm" style="margin:0">struct block_q4_0 {</span><br>' +
      '<span class="cm" style="margin:0">&nbsp;&nbsp;float16_t d;</span><br>' +
      '<span class="cm" style="margin:0">&nbsp;&nbsp;uint8_t qs[16];</span><br>' +
      '<span class="cm" style="margin:0">};</span><br>' +
      '<span class="cm" style="margin:0">#define QUANT_K_Q4_0 32</span></div>' +
      '<div class="cb" style="margin-top:5px">被 <b>所有</b> Vulkan 着色器 include。</div>';

    const cside = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--e)' });
    cside.innerHTML = '<div class="ct" style="color:var(--e)">C 侧 · ggml-common.h（L1-04）</div>' +
      '<div class="cb">' +
      '<span class="cm" style="margin:0">typedef struct {</span><br>' +
      '<span class="cm" style="margin:0">&nbsp;&nbsp;ggml_half d;</span><br>' +
      '<span class="cm" style="margin:0">&nbsp;&nbsp;uint8_t qs[QK4_0 / 2];</span><br>' +
      '<span class="cm" style="margin:0">} block_q4_0;</span><br>' +
      '<span class="cm" style="margin:0">#define QK4_0 32</span></div>' +
      '<div class="cb" style="margin-top:5px">CPU 内核、CUDA、Metal……各自还有一份。</div>';

    dia.appendChild(glsl); dia.appendChild(U.arrow('=')); dia.appendChild(cside);
    glsl.style.opacity = '.35'; cside.style.opacity = '.35';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '<span class="v">types.glsl</span> 是 GLSL 侧的"量化块字典"：每个类型一个 struct。',
      '<span class="k">Q4_0</span>：32 个元素一块，块内 = 1 个 f16 scale + 16 字节 nibble。' +
      '与左边 C 侧的字段<b>逐个对应</b>。',
      '<span class="k">packed16</span> 视图不是新数据：同一块内存换个读法，' +
      '让 <span class="v">dequantize4()</span> 一次取 4 个 nibble。',
      '第 70-77 行的 <span class="v">#if defined(DATA_A_Q4_0)</span> 是<b>编译期</b>选择：' +
      '一个着色器只认识一种 A 类型。',
      '结论：<span class="k">新增一个量化类型 = 改多处</span> —— ' +
      'CPU 块定义、GLSL 块定义、dequant 着色器、生成器的类型清单，以及每个后端各自的 kernel。'
    ];
    tl.at(600, () => { glsl.style.opacity = '1'; msg.innerHTML = texts[0]; });
    tl.at(3400, () => { msg.innerHTML = texts[1]; });
    tl.at(6400, () => { msg.innerHTML = texts[2]; });
    tl.at(9400, () => { msg.innerHTML = texts[3]; });
    tl.at(12400, () => { cside.style.opacity = '1'; msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 6 <span class="hl-c">dequant_q4_0.comp</span>：一个反量化内核的全部 30 行 */
{
  kicker: "L6-07 · 反量化族（26 个）",
  title: "<span class=\"hl-c\">dequant_q4_0.comp</span>：一个反量化内核的全部 30 行",
  sub: "最大的族、最短的文件。看懂这 30 行，其余的 dequant_*.comp 都是同一个骨架换个块布局。",
  caption: "回顾 L1-04：Q4_0 的 32 个权重 = 1 个 f16 scale + 32 个 4-bit。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_0.comp",
  mark: [4, 7, 9, 12, 15, 17, 20, 22, 29, 33, 35],
  lineNo: 1,
  code: `#version 450

#include "dequant_head.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
//>> 256 个线程：每 64 个线程负责一个 block，于是一个 workgroup 处理 4 个 block

layout (binding = 0) readonly buffer A {block_q4_0 data_a[];};
//>> binding 0 读量化块数组，binding 1 写 f16 输出
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;
//>> i = 本 workgroup 的第几个 block（每批 4 个）

    const uint tid = gl_LocalInvocationID.x % 64;
//>> tid 在 block 内的编号 0..63
    const uint il  = tid/32;
//>> il ∈ {0,1}：决定从 qs 的哪 8 个字节取值（0..7 还是 8..15）
    const uint ir  = tid%32;
    const uint ib = 32*i + ir;
//>> ib：全局块号；第 17 行越界就返回
    if (ib >= p.nel / 32) {
        return;
    }

    const uint q_idx = 8*il;
    const uint b_idx = 1024*i + 32*ir + q_idx;

    const float d = float(data_a[ib].d);
//>> 每块一个 f16 scale，转成 float 后乘在结果上

    [[unroll]] for (uint l = 0; l < 8; ++l) {
        data_b[b_idx + l +  0] = D_TYPE(d * ((data_a[ib].qs[q_idx + l] & 0xF) - 8.0f));
//>> 低 nibble 减 8 —— 这是 Q4_0 的零点（对照 L1-04）
        data_b[b_idx + l + 16] = D_TYPE(d * ((data_a[ib].qs[q_idx + l] >>  4) - 8.0f));
//>> 高 nibble 同理：同一字节的另一半
    }
}`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="top" style="gap:9px;align-items:center"></div>
      <div class="col" id="rows" style="gap:6px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const top = wrap.querySelector('#top');
    top.innerHTML = '<span class="chip c">1 个 block_q4_0</span>' +
      '<span class="cm" style="margin:0">d: f16</span>' +
      '<span class="cm" style="margin:0">qs[16]</span>' +
      '<span class="arrow">-></span>' +
      '<span class="chip b">32 个 f16</span>' +
      '<span class="cm" style="margin:0">data_b[b_idx + 0..31]</span>';

    const rows = wrap.querySelector('#rows');
    function mkRow(label, cls) {
      const r = U.el('div', { class: 'row', style: 'gap:4px;align-items:center' });
      r.appendChild(U.el('span', { class: 'cm', style: 'margin:0;width:132px;font-size:9px', text: label }));
      const cells = [];
      for (let i = 0; i < 16; i++) {
        const c = U.el('div', { style: 'width:24px;height:17px;border:1px solid var(--border);border-radius:2px;' +
          'background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
        c.textContent = '--';
        r.appendChild(c); cells.push(c);
      }
      rows.appendChild(r);
      return { r: r, cells: cells, cls: cls };
    }
    const lo = mkRow('+l  <- 低 nibble', 'a');
    const hi = mkRow('+l+16  <- 高 nibble', 'e');
    lo.r.style.opacity = '.35'; hi.r.style.opacity = '.35';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '输入是一个 <span class="k">block_q4_0</span>：一个 f16 scale + 16 字节（32 个 nibble）。' +
      '输出是 32 个 f16。',
      '第 <span class="v">11</span> 行：一个 workgroup 认领 <b>4 个块</b>，' +
      '<span class="v">tid/64</span> 决定自己管哪一个；第 <span class="v">13-14</span> 行再把 64 个线程对半分。',
      '第 <span class="v">27-28</span> 行每次 unroll 写<b>两个</b>元素：' +
      '<span class="v">+l</span> 取低 4 位、<span class="v">+l+16</span> 取高 4 位 —— ' +
      '所以两行分别是输出的前 16 个与后 16 个。',
      '<span class="v">il=0</span> 的线程读 <span class="v">qs[0..7]</span>（<span class="v">q_idx=0</span>），' +
      '填两行的<b>前 8 格</b>。',
      '<span class="v">il=1</span> 的线程读 <span class="v">qs[8..15]</span>（<span class="v">q_idx=8</span>），' +
      '填两行的<b>后 8 格</b> —— 32 个元素这才齐。',
      '一句话：<span class="k">dequant_*.comp 只有块布局不同</span>，骨架完全一致。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    tl.at(3400, () => { msg.innerHTML = texts[1]; lo.r.style.opacity = '1'; hi.r.style.opacity = '1'; });
    tl.at(6400, () => {
      lo.cells.forEach(c => { c.textContent = '?'; });
      hi.cells.forEach(c => { c.textContent = '?'; });
      msg.innerHTML = texts[2];
    });
    tl.at(9400, () => {
      lo.cells.slice(0, 8).forEach((c, i) => { c.style.background = 'rgba(88,166,255,.20)'; c.style.borderColor = 'var(--a)'; c.textContent = 'q' + i; });
      hi.cells.slice(0, 8).forEach((c, i) => { c.style.background = 'rgba(88,166,255,.20)'; c.style.borderColor = 'var(--a)'; c.textContent = 'q' + i; });
      msg.innerHTML = texts[3];
    });
    tl.at(12400, () => {
      lo.cells.slice(8).forEach((c, i) => { c.style.background = 'rgba(247,120,186,.20)'; c.style.borderColor = 'var(--e)'; c.textContent = 'q' + (i + 8); });
      hi.cells.slice(8).forEach((c, i) => { c.style.background = 'rgba(247,120,186,.20)'; c.style.borderColor = 'var(--e)'; c.textContent = 'q' + (i + 8); });
      msg.innerHTML = texts[4];
    });
    tl.at(16400, () => {
      lo.cells.forEach(c => { c.style.background = 'rgba(63,185,80,.20)'; c.style.borderColor = 'var(--b)'; });
      hi.cells.forEach(c => { c.style.background = 'rgba(63,185,80,.20)'; c.style.borderColor = 'var(--b)'; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 7 <span class="hl-b">mul_mmq.comp</span>：B 侧先量化成 <span class="hl-a">q8_1</span>，再整数点积 */
{
  kicker: "L6-07 · 矩阵乘族（23 个）",
  title: "<span class=\"hl-b\">mul_mmq.comp</span>：B 侧先量化成 <span class=\"hl-a\">q8_1</span>，再整数点积",
  sub: "三个矩阵乘族分工不同：大 M 走 mul_mm，小 M 走 mul_mat_vec，整数点积走 mul_mmq。",
  caption: "B 侧的 q8_1 由 quantize_q8_1.comp 现场算出，再喂给 mul_mmq —— 权重始终是量化态，只有激活被压成 int8。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp",
  mark: [0, 3, 5, 8, 10, 12, 15, 17],
  lineNo: 26,
  code: `layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
//>> workgroup 大小也是 spec constant：同一个 SPIR-V 服务多种设备

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif
layout (binding = 1) readonly buffer B {block_q8_1_x4_packed128 data_b[];};
//>> B 侧不是 f32/f16，而是 4 个 q8_1 块打包成的 128 字节块
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

#ifdef MUL_MAT_ID
layout (binding = 3) readonly buffer IDS {int data_ids[];};
//>> MUL_MAT_ID（MoE 的按专家矩阵乘）多两个 binding：专家 id 与计数
layout (binding = 4) readonly buffer Counts {int data_expert_count[];};
#endif`,
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const defs = [
      { c: 'a', t: 'mul_mm.comp', b: '大 M（prompt / 训练式批量）：<br>block 级分块 + 共享内存，' +
          '非 coopmat 时每种 A 类型<b>单独编译</b>。', m: 'GGML_OP_MUL_MAT（大 M）' },
      { c: 'c', t: 'mul_mat_vec_q4_k.comp', b: '小 M（逐 token 解码，最常见）：<br>' +
          '一个 workgroup 算一行，K-quant 用 super-block。', m: 'GGML_OP_MUL_MAT / MUL_MAT_ID' },
      { c: 'e', t: 'mul_mmq.comp', b: '整数点积路径：<br>B 量化成 q8_1，A 保持量化，' +
          '用 <b>dot</b> 指令累加。', m: 'GGML_OP_MUL_MAT（量化×量化）' }
    ];
    const host = wrap.querySelector('#cards');
    const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
    els.forEach(e => e.style.opacity = '.28');

    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看左边的 binding：<span class="k">A</span> 是量化权重（有 packed16 / packed32 视图），' +
      '<span class="k">B</span> 是 <span class="v">block_q8_1_x4_packed128</span>，<span class="k">D</span> 是输出。',
      '为什么要 q8_1？因为这条路径用的是 <span class="k">整数点积</span>（第 7 行的扩展）：' +
      'A 的 nibble 与 B 的 int8 直接做 <b>dot</b>，比反量化成浮点再 FMA 更省。',
      '<span class="v">B</span> 不是权重 —— 是激活。权重（A）保持量化态不被展开，' +
      '这是 Vulkan 后端省显存/带宽的关键。',
      '三个族的<b>分工边界</b>不同：<span class="v">mul_mat_vec_*</span> 的 pipeline 表按' +
      '<b>列数</b>索引（source.md 第七节：<span class="v">[dmmv_wg][a_type][num_cols-1]</span>），' +
      '<span class="v">mul_mm*</span> 按 M/N 分块，<span class="v">mul_mmq</span> 走整数点积。',
      '<span class="v">MUL_MAT_ID</span> 分支（第 38-41 行）多两个 binding：' +
      '专家 id 与每专家行数 —— MoE 的 <span class="v">GGML_OP_MUL_MAT_ID</span> 就落在这里。'
    ];
    els.forEach((e, i) => tl.at(700 + i * 3200, () => {
      els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; });
      msg.innerHTML = texts[i];
    }));
    tl.at(11500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
    tl.at(14800, () => { msg.innerHTML = texts[4]; });
  }
},

/* ------------------------------------------------------ 8 <span class="hl-d">mul_mat_vec_q4_k.comp</span>：6-bit scale 是<span class="hl-a">位运算</span>拼出来的 */
{
  kicker: "L6-07 · K-quant 的写法",
  title: "<span class=\"hl-d\">mul_mat_vec_q4_k.comp</span>：6-bit scale 是<span class=\"hl-a\">位运算</span>拼出来的",
  sub: "K-quant 的 super-block 有 256 个元素，8 个 6-bit scale 被打包进 12 字节 —— 所以解包就是移位与掩码。",
  caption: "对照 L1-04：Q4_K 的块是 dm(2×f16) + scales[12] + qs[128] = 144 字节 / 256 个元素。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp",
  mark: [0, 2, 4, 6, 8, 11, 13, 16, 18, 19, 21],
  lineNo: 11,
  code: `void calc_superblock(const uint a_offset, const uint b_offset, const uint v_im, const uint q_offset, const uint y_offset, const uint i, const uint num_blocks_per_row, const uint first_row, const uint num_rows) {
//>> calc_superblock：一次算一个 256 元素的 super-block
    const uint y1_idx = i * QUANT_K + y_offset;
//>> y1_idx / y2_idx 相差 128 —— 一个 super-block 被劈成前后两半
    const uint y2_idx = y1_idx + 128;

    [[unroll]] for (uint n = 0; n < num_rows; ++n) {
        const uint ib0 = a_offset + (first_row+n)*num_blocks_per_row;
        const FLOAT_TYPEV2 dm = FLOAT_TYPEV2(data_a[ib0 + i].dm);
//>> dm 是两个 f16：d（scale）与 dmin

        const uint32_t scale0_u32 = data_a_packed16[ib0 + i].scales[v_im    ];
//>> scales 的 12 个字节里塞了 8 个 6-bit scale + 8 个 6-bit min
        const uint32_t scale4_u32 = data_a_packed16[ib0 + i].scales[v_im + 2];
        const uint32_t scale8_u32 = data_a_packed16[ib0 + i].scales[v_im + 4];

        const uint32_t scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
//>> 把相邻两个 16-bit 槽拼成一个 uint32
        const uint32_t scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0) >> 2;
        const vec4 scale_0_4_l_f = vec4(unpack8(scale_0_4_l & 0x3F3F3F3F));
//>> unpack8 + 掩码 0x3F3F3F3F：从一个 uint32 里取出 4 个 6-bit scale
        const vec4 scale8_f = vec4(unpack8((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0F) | scale_0_4_h));
//>> 后 4 个 scale 连同刚才溢出的高位一起拼出来

        const FLOAT_TYPE sc0 = scale_0_4_l_f.x;
        const FLOAT_TYPE sc1 = scale_0_4_l_f.y;
        const FLOAT_TYPE sc2 = scale_0_4_l_f.z;
        const FLOAT_TYPE sc3 = scale_0_4_l_f.w;
        const FLOAT_TYPE sc4 = scale8_f.x;
        const FLOAT_TYPE sc5 = scale8_f.y;
        const FLOAT_TYPE sc6 = scale8_f.z;
        const FLOAT_TYPE sc7 = scale8_f.w;`,
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row" id="top" style="gap:6px;align-items:center"></div>
      <div class="row wrap" id="subs" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const top = wrap.querySelector('#top');
    top.innerHTML = '<span class="chip d">1 个 super-block = 256 个元素</span>' +
      '<span class="cm" style="margin:0">= dm(2 x f16) + scales[12] + qs[128] = 144 字节</span>' +
      '<span class="arrow">-></span>' +
      '<span class="chip a">8 个 sub-block x 32 个 4-bit</span>';

    const subs = wrap.querySelector('#subs');
    const cells = [];
    for (let i = 0; i < 8; i++) {
      const c = U.el('div', { style: 'width:78px;height:34px;border:1px solid var(--border);border-radius:3px;' +
        'background:#10151b;display:flex;flex-direction:column;align-items:center;justify-content:center;' +
        'font-size:9px;color:var(--dim)' });
      c.innerHTML = '<b style="color:var(--d)">sub ' + i + '</b><span>6-bit scale</span>';
      subs.appendChild(c); cells.push(c);
    }

    const msg = wrap.querySelector('#msg');
    const texts = [
      'K-quant 的块是 <span class="k">super-block</span>：256 个元素，8 个 sub-block 各 32 个。',
      '每个 sub-block 有自己的 6-bit scale。8 个 x 6 bit = 48 bit，正好塞进 ' +
      '<span class="v">scales[12]</span>（12 字节 = 96 bit，另一半留给 min）。',
      '第 <span class="v">19-21</span> 行一次取 3 个 16-bit 槽' +
      '（<span class="v">scales[]</span> 一共 6 个 uint16 = 12 字节），' +
      '第 <span class="v">23-26</span> 行用移位/掩码把它们摊成 8 个 6-bit scale。',
      '第 <span class="v">12-13</span> 行：一个 super-block 的数据被劈成 ' +
      '<span class="v">y1_idx</span>（前 128 元素）与 <span class="v">y2_idx</span>（后 128）。',
      '第 <span class="v">37-38</span> 行取 qs：<span class="v">q_offset/4</span> 与 <span class="v">+16</span> —— ' +
      '同样是"一次读 4 个字节、拆出 8 个 nibble"的手法。',
      '一句话：<span class="k">K-quant 在着色器里 = 移位 + 掩码 + fma</span>，没有查表魔法。'
    ];
    tl.at(600, () => { msg.innerHTML = texts[0]; });
    tl.at(3400, () => { msg.innerHTML = texts[1]; });
    tl.at(6400, () => {
      cells.forEach((c, i) => { c.style.borderColor = 'var(--d)'; c.style.background = 'rgba(188,140,255,.15)'; });
      msg.innerHTML = texts[2];
    });
    tl.at(9600, () => {
      cells.forEach((c, i) => { c.style.background = i < 4 ? 'rgba(88,166,255,.18)' : 'rgba(247,120,186,.18)'; });
      msg.innerHTML = texts[3];
    });
    tl.at(12600, () => { msg.innerHTML = texts[4]; });
    tl.at(16000, () => {
      cells.forEach(c => { c.style.background = 'rgba(63,185,80,.18)'; c.style.borderColor = 'var(--b)'; });
      msg.innerHTML = texts[5];
    });
  }
},

/* ------------------------------------------------------ 9 把这一课压成一张表 */
{
  kicker: "L6-07 · 收束",
  title: "把这一课压成一张表",
  sub: "六步链 + 三个数字。回到第 2 幕的那 16 行 —— 整条链的枢纽就在那里。",
  caption: "下一课 L6-08：Metal 后端怎么把多个 ggml op 融合进一个 kernel。",
  src: "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
  mark: [1, 2, 8, 14],
  lineNo: 436,
  code: `void string_to_spv(std::string name, const std::string& source, const std::map<std::string, std::string>& defines, bool fp16 = true, bool coopmat = false, bool coopmat2 = false, bool f16acc = false, const std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string out_path = join_paths(output_dir, name + ".spv");

    if (input_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, out_path));
        return;
    } else if (basename(input_filepath) != source) {
        // Only compile shader variants matching the input filename
        return;
    }

    compile_count_guard slot = acquire_compile_slot();
    compiles.push_back(std::async(
        string_to_spv_func, name, input_filepath, out_path, defines, coopmat, generate_dep_file, std::move(slot)));`,
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['步', '在哪', '产物'],
      [['1', 'process_shaders()', '要编译的 (着色器, 类型, 变体) 清单'],
       ['2', 'string_to_spv()', '变体名 + <output_dir>/<name>.spv 路径'],
       ['3', 'glslc（string_to_spv_func）', '<name>.spv（SPIR-V 二进制）'],
       ['4', 'write_output_files()', '<name>_data[] / <name>_len（C 数组）'],
       ['5', 'ggml_vk_create_pipeline_func()', 'shader module → compute pipeline'],
       ['6', 'ggml_vk_get_dequantize_*', '运行时按类型查表取 pipeline → dispatch']],
      { monoCols: [1] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '不看代码，说出一个 <span class="mono">.comp</span> 从生成到被 pipeline 使用经过哪几步？',
      '<b>六步</b>：<br>' +
      '1. <span class="mono">process_shaders()</span> 决定要哪些变体；<br>' +
      '2. <span class="mono">string_to_spv()</span> 拼变体名与 <span class="mono">.spv</span> 输出路径；<br>' +
      '3. <span class="mono">glslc</span> 带着一串 <span class="mono">-D</span> 宏把 <span class="mono">.comp</span> 编译成 <span class="mono">.spv</span>；<br>' +
      '4. <span class="mono">write_output_files()</span> 把 <span class="mono">.spv</span> 读成 ' +
      '<span class="mono">&lt;name&gt;_data[]</span> 与 <span class="mono">&lt;name&gt;_len</span>；<br>' +
      '5. 主机端 <span class="mono">createShaderModule()</span> → <span class="mono">vkCreateComputePipelines</span>；<br>' +
      '6. 运行时按 (权重类型, 列数, workgroup 档) 查表取 pipeline，再 dispatch。<br>' +
      '关键：<b>第 2 步到第 3 步之间是"一变多"</b> —— 一个 <span class="mono">.comp</span> 变成几百个 SPIR-V。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '六步，每步一个产物。',
      '<span class="k">1-2</span>：同一个 <span class="v">.comp</span> 被点名很多次，' +
      '每次一组不同的宏 —— 这就是"变体"。',
      '<span class="k">3</span>：<span class="v">glslc</span> 是唯一的编译器；' +
      '<span class="v">.spv</span> 是它的产物。',
      '<span class="k">4</span>：<span class="v">.spv</span> 变成 C 数组，' +
      '于是<b>不需要运行时编译着色器</b>。',
      '<span class="k">5-6</span>：主机端建 pipeline，运行时按类型查表 —— ' +
      '这就是 L6-06 讲的那一半。',
      '三个数字：<span class="v">161</span> 个 <span class="v">.comp</span>、' +
      '<span class="v">26</span> 个反量化着色器、<span class="v">4</span> 个矩阵乘/归约着色器文件' +
      '撑起 <span class="v">MUL_MAT</span> 的全部形状。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2600 + i * 2200, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(17500, () => {
      rows.forEach(x => { x.className = ''; });
      msg.innerHTML = texts[5];
    });
  }
},

];
