/* ==========================================================================
   L7-03 · OpenVINO 后端：把 ggml 图翻译成另一套图 IR
   --------------------------------------------------------------------------
   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。
   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；
   最终产物仍由 tools/check_ir_fidelity.py 独立校验。
   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。
   ========================================================================== */
'use strict';

const SCENES = [
/* ------------------------------------------------------ 1 OpenVINO 后端：<span class="hl-a">翻译图</span>，而不是写 kernel */
{
  kicker: "L7 · NPU 与加速器后端",
  title: "OpenVINO 后端：<span class=\"hl-a\">翻译图</span>，而不是写 kernel",
  sub: "公共头只有 23 行有效内容、11 个 C 函数。真正的实现是 81 个 C++ 文件构成的\"前端\"。",
  caption: "L7-01 CANN 走的是同一路线（映射到厂商算子），L7-02 Hexagon 则把整图丢给远端 DSP；三课对照着看。",
  src: "ggml/include/ggml-openvino.h",
  mark: [0, 4, 9, 19, 22, 25],
  lineNo: 11,
  code: `#define GGML_OPENVINO_NAME "OPENVINO"
//>> 后端名字：注册表里的 key 就是它（见 L3-02）

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device);
//>> 唯一的创建入口：device 号进去，backend 句柄出来

GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer);
//>> buffer 判定 —— 调度器靠它识别"这个张量在 OpenVINO 手里"

GGML_BACKEND_API bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API size_t ggml_backend_openvino_buffer_get_ctx_id(ggml_backend_buffer_t buffer);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device);
//>> device buffer type：这个后端的默认 buffer 类型（见 L4-03）

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device);
//>> host buffer type：另一类 buffer，调度器按需在两者间搬运

GGML_BACKEND_API int ggml_backend_openvino_get_device_count(void);
//>> device count：告诉调度器这个后端有几台设备可以切（第 6 幕展开）

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void);`,
  duration: 21000,
  build(root, tl) {
    const LN = {"11": 0, "12": 2, "13": 3, "14": 4, "15": 6, "16": 7, "17": 8, "18": 9, "19": 11, "20": 12, "21": 13, "22": 14, "23": 15, "24": 16, "25": 17, "26": 18, "27": 19, "28": 21, "29": 22, "30": 24, "31": 25, "32": 27, "33": 28};
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center">
        <span class="chip">ggml cgraph</span><span class="arrow">-></span>
        <span class="chip a">GgmlOvDecoder</span><span class="arrow">-></span>
        <span class="chip b">ov::Model</span><span class="arrow">-></span>
        <span class="chip c">compile_model</span><span class="arrow">-></span>
        <span class="chip d">NPU / GPU / CPU</span>
      </div>
      <div class="bars" id="bars"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const groups = [
      { l: '顶层胶水', n: 12, c: 'a' },
      { l: 'frontend 骨架', n: 12, c: 'b' },
      { l: 'op 翻译器', n: 45, c: 'c' },
      { l: 'OV pass', n: 12, c: 'd' },
      { l: 'rt_info', n: 1, c: 'e' }
    ];
    const host = wrap.querySelector('#bars');
    const bars = groups.map(g => {
      const b = U.bar(g.l + '  (' + g.n + ')', g.c);
      host.appendChild(b.el);
      return b;
    });
    const msg = wrap.querySelector('#msg');
    const texts = [
      'ggml 侧的入口只有 <span class="k">graph_compute</span> 一个。它拿到的是<b>一段子图</b>，不是单个算子。',
      '<span class="v">顶层胶水 12 个</span>：虚表实现、cgraph→decoder、设备配置、权重量化、编译缓存。',
      '<span class="v">frontend 骨架 12 个</span>：<span class="k">op_table</span> 是注册表，' +
        '<span class="k">translate_session</span> 是调度器，<span class="k">node_context.h</span> 是翻译器的输入视图。',
      '<span class="v">op 翻译器 45 个</span>：42 个 .cpp + 3 个 .hpp，一个（族）ggml 算子一个文件。' +
        '<b>这个后端的工程量几乎全在这里</b>。',
      '<span class="v">OV pass 12 个</span>：翻译完还要改 OV 图 —— 融合成 Conv/SDPA、给去量化链打标记、KV state 重排。',
      '<span class="v">rt_info 1 个</span>：给大 <span class="k">Constant</span> 打标记，避免 NPUW 在编译时重复拷贝权重。',
      '<span class="v">ov::Model</span> 是 OpenVINO 的图 IR：算子换成 ov::op，边换成 Output&lt;Node&gt;，权重换成 Constant。' +
        '<span class="k">这里没有一行 kernel</span>。'
    ];
    tl.at(700, () => {
      bars.forEach((b, i) => { b.fill.style.width = (12 + i * 17) + '%'; b.val.textContent = groups[i].n; });
      msg.innerHTML = texts[0];
      U.markLines(document, [LN[11], LN[14]]);
    });
    groups.forEach((g, i) => tl.at(3400 + i * 3000, () => {
      bars.forEach((b, k) => { b.el.style.opacity = k === i ? '1' : '.32'; });
      msg.innerHTML = texts[i + 1];
    }));
    tl.at(18400, () => {
      bars.forEach(b => { b.el.style.opacity = '1'; });
      msg.innerHTML = texts[6];
      U.markLines(document, [LN[11], LN[14], LN[18], LN[27], LN[29], LN[31]]);
    });
  }
},

/* ------------------------------------------------------ 2 ★ 一个 ggml 节点 = <span class="hl-a">一次查表</span> */
{
  kicker: "L7-03 · 核心洞察",
  title: "★ 一个 ggml 节点 = <span class=\"hl-a\">一次查表</span>",
  sub: "翻译的骨架只有十几行：取 op 名字，去表里找翻译函数，调用它，把结果登记进 tensor_map。",
  caption: "对比 L5 系列：CPU 后端在同样的位置是 ggml_compute_forward 里一个按 tensor->op 分支的大 switch，每个 case 调一个 kernel。这里没有 switch。",
  src: "ggml/src/ggml-openvino/openvino/translate_session.cpp",
  mark: [0, 2, 4, 9, 11, 14, 15],
  lineNo: 297,
  code: `    auto translate_node = [&](const std::shared_ptr<GgmlDecoder> & decoder, int node_idx) {
//>> 这个 lambda 就是"翻译一个节点"的全部
        auto operation_type = decoder->get_op_type(node_idx);
//>> decoder->get_op_type() 返回的是【宏名】，如 "GGML_OP_MUL_MAT" —— 与 L1-02 的 ggml_op_name() 同源
        if (operation_type == "GGML_OP_NONE") {
//>> GGML_OP_NONE 不算算子，直接跳过（L1-02 幕 7：新张量的默认身份）
            return ov::OutputVector{};
        }

        auto it = m_translator_map.find(operation_type);
//>> ★ 支持与翻译共用同一张表：能不能算 = 表里有没有这个名字
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
//>> 查不到就直接抛异常 —— 所以 supports_op 必须在切分阶段就把它挡在外面
                                      " is not implemented.");
        NodeContext node_context(decoder, tensor_map, node_idx, this);
        ov::OutputVector converted_outputs = it->second(node_context);
//>> 翻译函数返回 ov::OutputVector：一个 ggml 节点可以变成好几个 ov::op`,
  duration: 19000,
  build(root, tl) {
    const LN = {"297": 0, "298": 2, "299": 4, "300": 6, "301": 7, "302": 8, "303": 9, "304": 11, "305": 13, "306": 14, "307": 15};
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row center" id="lane" style="gap:6px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const lane = wrap.querySelector('#lane');
    const steps = [
      { c: 'a', t: 'node->op', b: 'GGML_OP_MUL_MAT' },
      { c: 'b', t: 'get_op_type()', b: '"GGML_OP_MUL_MAT"' },
      { c: 'c', t: '查 translator_map', b: '.find(operation_type)' },
      { c: 'd', t: '调用翻译器', b: 'it->second(node_context)' },
      { c: 'e', t: 'ov::OutputVector', b: '登记进 tensor_map' }
    ];
    const els = steps.map(s => {
      const e = U.card(s, { style: 'width:122px' });
      e.querySelector('.ct').style.fontSize = '10px';
      lane.appendChild(e);
      lane.appendChild(U.arrow('->'));
      return e;
    });
    els.forEach(e => { e.style.opacity = '.26'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '翻译一个节点，五步。第一步是<b>取身份</b>。',
      '<span class="v">decoder->get_op_type(node_idx)</span> 返回的就是 L1-02 里那张 <span class="k">GGML_OP_NAME</span> 表的输出，' +
        '前缀 <span class="k">GGML_OP_</span> 由 decoder 拼上。',
      '拿到的字符串去 <span class="v">m_translator_map</span> 里查。这张表就是下一幕要逐字读的 <span class="k">op_table.cpp</span>。',
      '<span class="k">★ 关键</span>：这里没有"这个算子怎么算"的代码，只有"这个算子交给谁翻译"。<br>' +
        '计算语义全部在 <span class="v">it->second(context)</span> 里 —— 也就是 45 个 op 翻译器文件。',
      '翻译器返回 <span class="v">ov::OutputVector</span>。一个 ggml 节点可以吐出一个、也可以吐出六个 ov 节点 ——<br>' +
        '所以这不是"算子一一对应"，而是<b>两张图之间的映射</b>。',
      '一句话：<span class="k">ggml 的 op 枚举 = OpenVINO 后端的"函数指针表索引"</span>。表的缺失项 = 不支持。'
    ];
    tl.at(700, () => { els.forEach((e, k) => { e.style.opacity = k === 0 ? '1' : '.26'; }); msg.innerHTML = texts[0]; U.markLines(document, [LN[297]]); });
    tl.at(3800, () => { els.forEach((e, k) => { e.style.opacity = k <= 1 ? '1' : '.26'; }); msg.innerHTML = texts[1]; U.markLines(document, [LN[298], LN[299]]); });
    tl.at(7300, () => { els.forEach((e, k) => { e.style.opacity = k <= 2 ? '1' : '.26'; }); msg.innerHTML = texts[2]; U.markLines(document, [LN[303], LN[304]]); });
    tl.at(11000, () => { els.forEach((e, k) => { e.style.opacity = k <= 3 ? '1' : '.26'; }); msg.innerHTML = texts[3]; U.markLines(document, [LN[306], LN[307]]); });
    tl.at(14700, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [LN[307]]); });
    tl.at(17600, () => { msg.innerHTML = texts[5]; U.markLines(document, [LN[297], LN[298], LN[303], LN[304], LN[307]]); });
  }
},

/* ------------------------------------------------------ 3 <span class="hl-c">op_table.cpp</span>：后端能力的唯一真相来源 */
{
  kicker: "L7-03 · 翻译表",
  title: "<span class=\"hl-c\">op_table.cpp</span>：后端能力的唯一真相来源",
  sub: "一张 unordered_map：key 是 ggml 的 op 宏名，value 是翻译函数。表里有什么，后端就能算什么。",
  caption: "第 61-80 行还有 SWIGLU / GEGLU / SET_ROWS / CPY / CLAMP / PAD / REPEAT / DIAG / TRI / SET / ROLL 等；完整 55 项见 source.md 的逐字引用。",
  src: "ggml/src/ggml-openvino/openvino/op_table.cpp",
  mark: [0, 4, 6, 13, 14, 15, 17, 21, 23, 24, 31, 32, 33, 35, 38, 44],
  lineNo: 23,
  code: `std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
//>> 整张表就是一次 return {...}，没有分支、没有条件编译
    using namespace ov::op;
    return {
        {"GGML_OP_ADD",             op::translate_add                              },
//>> ADD 要处理 MoE 的专家平面求和与类型混用，所以有专属实现（add.cpp）
        {"GGML_OP_ADD1",            op::translate_1to1_match_2_inputs<v1::Add>     },
//>> ★ ADD1 走 1to1 模板 —— 标量加法与张量加法在 ov 侧是同一个 v1::Add
        {"GGML_OP_ADD_ID",          op::translate_add_id                           },
        {"GGML_OP_CONCAT",          op::translate_concat                           },
        {"GGML_OP_CONT",            op::translate_cont                             },
        {"GGML_OP_DIV",             op::translate_div                              },
        {"GGML_OP_FILL",            op::translate_fill                             },
        {"GGML_OP_GET_ROWS",        op::translate_get_rows                         },
        {"GGML_OP_IM2COL",          op::translate_im2col                           },
        {"GGML_OP_MUL",             op::translate_1to1_match_2_inputs<v1::Multiply>},
//>> ★ MUL 只需要一个 ov::op::v1::Multiply —— 模板自动补齐输入检查与类型转换
        {"GGML_OP_MUL_MAT",         op::translate_mulmat                           },
//>> MUL_MAT 是 45 个翻译器里最关键的一个，输出 ov::op::v0::MatMul（见 mulmat.cpp）
        {"GGML_OP_MUL_MAT_ID",      op::translate_mul_mat_id                       },
        {"GGML_OP_PERMUTE",         op::translate_permute                          },
        {"GGML_OP_RESHAPE",         op::translate_reshape                          },
//>> RESHAPE / PERMUTE / VIEW 这些"视图算子"也有翻译器：OpenVINO 图里必须显式表达
        {"GGML_OP_RMS_NORM",        op::translate_rms_norm                         },
        {"GGML_OP_NORM",            op::translate_norm                             },
        {"GGML_OP_L2_NORM",         op::translate_l2_norm                          },
        {"GGML_OP_SUM_ROWS",        op::translate_sum_rows                         },
        {"GGML_OP_ROPE",            op::translate_rope                             },
        {"GGML_OP_SCALE",           op::translate_scale                            },
        {"GGML_OP_SQR",             op::translate_sqr                              },
        {"GGML_OP_SQRT",            op::translate_sqrt                             },
        {"GGML_OP_SOFT_MAX",        op::translate_soft_max                         },
        {"GGML_OP_ARGSORT",         op::translate_argsort                          },
        {"GGML_OP_SUB",             op::translate_1to1_match_2_inputs<v1::Subtract>},
        {"GGML_OP_TRANSPOSE",       op::translate_transpose                        },
        {"GGML_UNARY_OP_GELU",      op::translate_1to1_match_1_input<v7::Gelu>     },
//>> GELU / SIGMOID / SILU / TANH / EXP / NEG / RELU 全部是一行模板特化
        {"GGML_UNARY_OP_SIGMOID",   op::translate_1to1_match_1_input<v0::Sigmoid>  },
        {"GGML_UNARY_OP_SILU",      op::translate_1to1_match_1_input<v4::Swish>    },
        {"GGML_UNARY_OP_SOFTPLUS",  op::translate_unary_softplus                   },
        {"GGML_UNARY_OP_TANH",      op::translate_1to1_match_1_input<v0::Tanh>     },
        {"GGML_UNARY_OP_EXP",       op::translate_1to1_match_1_input<v0::Exp>      },
        {"GGML_UNARY_OP_NEG",       op::translate_1to1_match_1_input<v0::Negative> },
        {"GGML_UNARY_OP_RELU",      op::translate_1to1_match_1_input<v0::Relu>     },
        {"GGML_OP_VIEW",            op::translate_view                             },
//>> VIEW 有专属实现，因为它要把 ggml 的 nb[] 语义折算成 Slice + Reshape（view.cpp）`,
  duration: 24000,
  build(root, tl) {
    const LN = {"23": 0, "24": 2, "25": 3, "26": 4, "27": 6, "28": 8, "29": 9, "30": 10, "31": 11, "32": 12, "33": 13, "34": 14, "35": 15, "36": 17, "37": 19, "38": 20, "39": 21, "40": 23, "41": 24, "42": 25, "43": 26, "44": 27, "45": 28, "46": 29, "47": 30, "48": 31, "49": 32, "50": 33, "51": 34, "52": 35, "53": 37, "54": 38, "55": 39, "56": 40, "57": 41, "58": 42, "59": 43, "60": 44};
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['ggml op', 'op_table.cpp', '翻译成什么 ov::op', '类别'],
      [['GGML_OP_MUL', '35', 'v1::Multiply', '1to1 模板'],
       ['GGML_OP_ADD1', '27', 'v1::Add', '1to1 模板'],
       ['GGML_OP_SUB', '50', 'v1::Subtract', '1to1 模板'],
       ['GGML_UNARY_OP_SILU', '54', 'v4::Swish', '1to1 模板'],
       ['GGML_UNARY_OP_GELU', '52', 'v7::Gelu', '1to1 模板'],
       ['GGML_OP_RMS_NORM', '40', 'v1::Multiply · v1::ReduceMean · v1::Add · v0::Sqrt · v1::Divide', '1 op -> 一串'],
       ['GGML_OP_NORM', '41', 'v6::MVN', '1 op -> 1 融合算子'],
       ['GGML_OP_FLASH_ATTN_EXT', '68', 'v13::ScaledDotProductAttention', '1 op -> 1 融合算子'],
       ['GGML_OP_MUL_MAT', '36', 'v0::MatMul', '专属翻译器'],
       ['GGML_OP_SOFT_MAX', '48', 'v8::Softmax', '专属翻译器'],
       ['GGML_OP_GET_ROWS', '33', 'v8::Gather', '专属翻译器'],
       ['GGML_OP_ARGSORT', '49', 'v11::TopK', '专属翻译器'],
       ['GGML_OP_IM2COL', '34', 'v3::ExtractImagePatches', '专属翻译器'],
       ['GGML_OP_POOL_2D', '79', 'v1::MaxPool / v1::AvgPool', '专属翻译器'],
       ['GGML_OP_CUMSUM', '74', 'v0::CumSum', '专属翻译器'],
       ['GGML_OP_SSM_CONV', '71', 'v1::GroupConvolution', '专属翻译器']],
      { monoCols: [0, 2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '先读右半：<span class="k">value 全是 op::translate_*</span>。函数名与 ggml 的 op 名一一对应。',
      '第一类：<span class="v">translate_1to1_match_2_inputs&lt;T&gt;</span> / <span class="v">&lt;1_input&gt;</span>。<br>' +
        '一个 ggml op 直接落成一个 ov::op —— 输入检查与类型转换由模板包办。',
      '第二类：一个 ggml op 展开成<b>一串</b> ov::op。RMS_NORM 是 6 个节点（下一幕逐字读）。',
      '第三类：反过来，一个 ggml op 落进<b>一个融合算子</b> —— NORM → MVN、FLASH_ATTN_EXT → SDPA。',
      '剩下的是专属翻译器：形状 / 索引语义太特别，模板盖不住，一个文件一个。',
      '所以"支持"的粒度是 <span class="k">ggml op</span>，不是 ov 节点：<br>只要某个 ggml op 有翻译器，它就能进子图。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    tl.at(3000, () => {
      [0, 1, 2, 3, 4].forEach(k => { rows[k].className = 'on'; });
      msg.innerHTML = texts[1];
      U.markLines(document, [LN[35], LN[27], LN[50], LN[54], LN[52]]);
    });
    tl.at(8000, () => {
      rows.forEach(r => { r.className = ''; });
      [5].forEach(k => { rows[k].className = 'on'; });
      msg.innerHTML = texts[2];
      U.markLines(document, [LN[40]]);
    });
    tl.at(12000, () => {
      rows.forEach(r => { r.className = ''; });
      [6, 7].forEach(k => { rows[k].className = 'on'; });
      msg.innerHTML = texts[3];
      U.markLines(document, [LN[41]]);
    });
    tl.at(16400, () => {
      rows.forEach(r => { r.className = ''; });
      [8, 9, 10, 11, 12, 13, 14, 15].forEach(k => { rows[k].className = 'on'; });
      msg.innerHTML = texts[4];
      U.markLines(document, [LN[36], LN[48], LN[33], LN[49], LN[34]]);
    });
    tl.at(20600, () => {
      rows.forEach(r => { r.className = ''; });
      msg.innerHTML = texts[5];
      U.markLines(document, [LN[23], LN[60]]);
    });
  }
},

/* ------------------------------------------------------ 4 ★ <span class="hl-a">supported_ops</span> 不是手写的清单，是从翻译表派生的 */
{
  kicker: "L7-03 · 支持判据（一）",
  title: "★ <span class=\"hl-a\">supported_ops</span> 不是手写的清单，是从翻译表派生的",
  sub: "三道门槛的第一道：这个 op（或 unary / glu 子枚举）的名字，在 op_table 里有没有？",
  caption: "验收点答案就藏在 build_supported_sets 这个 lambda 里 —— 它回答\"什么形状的子图会被整体交给 OpenVINO\"。",
  src: "ggml/src/ggml-openvino/ggml-openvino.cpp",
  mark: [0, 4, 6, 7, 14, 16, 24, 26, 30, 31, 33, 40, 48],
  lineNo: 1449,
  code: `static ggml_openvino_op_support ggml_backend_openvino_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op) {
//>> 返回 (bool, reason)：不支持时带一句人能读的理由，便于定位
    GGML_ASSERT(dev->reg != nullptr);

    static std::unordered_set<ggml_type> supported_types{
//>> ★ 张量类型白名单，13 种。量化类型只有 Q4_0/Q4_1/Q4_K/Q5_1/Q5_K/Q6_K/Q8_0/MXFP4
        GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_I64,  GGML_TYPE_I32,  GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,
        GGML_TYPE_MXFP4};

    // derive supported op sets from the op_table map, keys in
    // the map use the full macro name (e.g. "GGML_OP_ADD"), while
    // the ggml_*_op_name() helpers return only the trailing part (e.g. "ADD").
    // each set is built once and cached.
    static const auto build_supported_sets = [] {
//>> 取的就是 op_table.cpp 里那张表 —— 没有第二份"支持列表"
        const auto & table = ov::frontend::ggml::get_supported_ops();
        std::unordered_set<ggml_op> ops;
        std::unordered_set<ggml_unary_op> unary_ops;
        std::unordered_set<ggml_glu_op> glu_ops;

        // GGML_OP_NONE has no translator but is always safe to add to the supported set.
        ops.insert(GGML_OP_NONE);

        for (int i = 0; i < GGML_OP_COUNT; ++i) {
//>> 遍历 ggml 的全部 op 枚举，逐个拼 key 去查表
            const std::string key = std::string("GGML_OP_") + ggml_op_name(static_cast<ggml_op>(i));
//>> key 形如 "GGML_OP_MUL_MAT"；这正是 translate_session 查表用的同一个字符串
            if (table.count(key)) {
                ops.insert(static_cast<ggml_op>(i));
            }
        }
//>> unary 与 glu 是嵌在 GGML_OP_UNARY / GGML_OP_GLU 里的子枚举，所以各建一张集合
        for (int i = 0; i < GGML_UNARY_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_UNARY_OP_") + ggml_unary_op_name(static_cast<ggml_unary_op>(i));
            if (table.count(key)) {
                unary_ops.insert(static_cast<ggml_unary_op>(i));
            }
        }
        for (int i = 0; i < GGML_GLU_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_GLU_OP_") + ggml_glu_op_name(static_cast<ggml_glu_op>(i));
            if (table.count(key)) {
                glu_ops.insert(static_cast<ggml_glu_op>(i));
            }
        }
        return std::make_tuple(ops, unary_ops, glu_ops);
//>> GGML_OP_NONE 没有翻译器，但它不算算子，显式放行
    };
    static const auto supported_sets = build_supported_sets();
//>> 函数局部 static：整个进程只建一次（表是编译期常量，不会变）
    static const auto & supported_ops = std::get<0>(supported_sets);
    static const auto & supported_unary_ops = std::get<1>(supported_sets);
    static const auto & supported_glu_ops = std::get<2>(supported_sets);`,
  duration: 24000,
  build(root, tl) {
    const LN = {"1449": 0, "1450": 2, "1451": 3, "1452": 4, "1453": 6, "1454": 7, "1455": 8, "1456": 9, "1457": 10, "1458": 11, "1459": 12, "1460": 13, "1461": 14, "1462": 16, "1463": 17, "1464": 18, "1465": 19, "1466": 20, "1467": 21, "1468": 22, "1469": 23, "1470": 24, "1471": 26, "1472": 28, "1473": 29, "1474": 30, "1475": 31, "1476": 33, "1477": 34, "1478": 35, "1479": 36, "1480": 37, "1481": 38, "1482": 39, "1483": 40, "1484": 41, "1485": 42, "1486": 43, "1487": 44, "1488": 45, "1489": 47, "1490": 48, "1491": 50, "1492": 51, "1493": 52};
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="row center" id="gate" style="gap:7px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const gate = wrap.querySelector('#gate');
    const g = [
      { c: 'a', t: 'op_table 的 key', b: 'GGML_OP_* / GGML_UNARY_OP_* / GGML_GLU_OP_*' },
      { c: 'b', t: '枚举反查', b: 'ggml_op_name(i) 拼出同名 key' },
      { c: 'c', t: '三张集合', b: 'supported_ops / _unary_ops / _glu_ops' },
      { c: 'd', t: '第一道门槛', b: '查得到 -> 通过；<br>查不到 -> "has no op translator"' }
    ];
    const els = g.map(x => {
      const e = U.card(x, { style: 'width:158px' });
      e.querySelector('.ct').style.fontSize = '10px';
      gate.appendChild(e);
      return e;
    });
    els.forEach(e => { e.style.opacity = '.28'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '先看形状：<span class="k">supports_op 一次只看一个 op</span>。它返回 (bool, reason)，是 L4-02 切分的输入。',
      '<span class="v">ggml_op_name()</span> 的返回值前面拼上 <span class="k">GGML_OP_</span>，正好等于 op_table 的 key —— ' +
        '两张表是<b>同一份枚举的两个视图</b>（回顾 L1-02）。',
      'unary 与 glu 是嵌在 <span class="v">GGML_OP_UNARY</span> / <span class="v">GGML_OP_GLU</span> 里的子枚举，' +
        '不占 ggml_op 的编号，所以各建一张集合。',
      '★ 洞察：<span class="v">支持判据 = 翻译器的存在性</span>。第一道门槛问的不是"能不能算得快"，' +
        '而是 <span class="k">table.count(key)</span>。',
      '因此 <b>改 op_table.cpp 一行 = 改变后端的算力边界</b>；' +
        '13 种类型的白名单同理（下一幕）。'
    ];
    tl.at(700, () => { els.forEach((e, k) => { e.style.opacity = k === 0 ? '1' : '.28'; }); msg.innerHTML = texts[0]; U.markLines(document, [LN[1449]]); });
    tl.at(5600, () => { els.forEach((e, k) => { e.style.opacity = k <= 1 ? '1' : '.28'; }); msg.innerHTML = texts[1]; U.markLines(document, [LN[1461], LN[1462], LN[1470], LN[1471], LN[1476]]); });
    tl.at(11200, () => { els.forEach((e, k) => { e.style.opacity = k <= 2 ? '1' : '.28'; }); msg.innerHTML = texts[2]; U.markLines(document, [LN[1474], LN[1475], LN[1483]]); });
    tl.at(16600, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; U.markLines(document, [LN[1452], LN[1453], LN[1454]]); });
    tl.at(21200, () => { msg.innerHTML = texts[4]; U.markLines(document, [LN[1490]]); });
  }
},

/* ------------------------------------------------------ 5 第二、三道门槛：<span class="hl-b">类型</span> × <span class="hl-b">形状</span> × 设备例外 */
{
  kicker: "L7-03 · 支持判据（二）",
  title: "第二、三道门槛：<span class=\"hl-b\">类型</span> × <span class=\"hl-b\">形状</span> × 设备例外",
  sub: "op 有翻译器还不够：算子与全部输入的类型必须在 13 种白名单里，量化张量还不许是 3D。",
  caption: "\"什么形状的子图交给 OpenVINO\"的完整答案 = 每个节点都过这三道门槛的最长连续区间。",
  src: "ggml/src/ggml-openvino/ggml-openvino.cpp",
  mark: [0, 2, 3, 15, 31, 33, 34, 43, 45, 52, 54, 56, 58, 59, 61, 65, 67],
  lineNo: 1495,
  code: `    switch (op->op) {
//>> 按 op 分类走三条不同的分支 —— 因为 unary/glu 的子枚举不占 ggml_op 的编号
    case GGML_OP_UNARY: {
        auto supported = supported_unary_ops.find(ggml_get_unary_op(op)) != supported_unary_ops.end();
//>> UNARY：查 supported_unary_ops
        if (!supported) {
            return {false, "unary op " + std::string(ggml_unary_op_name(ggml_get_unary_op(op))) + " has no op translator"};
        }
        if (ggml_get_unary_op(op) == GGML_UNARY_OP_EXP && op->type == GGML_TYPE_F32) {
//>> 例外一：F32 的 EXP 没有翻译器能覆盖（类型级例外）
            return {false, "UNARY_EXP with F32 type is not supported"};
        }
        break;
    }
    case GGML_OP_GLU: {
        auto supported = supported_glu_ops.find(ggml_get_glu_op(op)) != supported_glu_ops.end();
//>> GLU：查 supported_glu_ops
        if (!supported) {
            return {false, "GLU op " + std::string(ggml_glu_op_name(ggml_get_glu_op(op))) + " has no op translator"};
        }
        // if (has_view_op_input(op)) {
        //     return {false, "GLU op " + std::string(ggml_glu_op_name(ggml_get_glu_op(op))) + " with view input is not supported"};
        // }
        if (op->src[1] == nullptr && op->src[0]->ne[0] % 2 != 0) {
//>> 例外二：src1 为空时 ne[0] 必须是偶数 —— 源码注释说是 ov gpu 的 bug
            // triggers bug in ov gpu
            return {false, "GLU op with odd src0 ne[0] and null src1 is not supported"};
        }
        break;
    }
    default: {
        auto supported = supported_ops.find(op->op) != supported_ops.end();
//>> default：普通算子查 supported_ops
        if (!supported) {
            return {false, "op " + std::string(ggml_op_name(op->op)) + " has no op translator"};
        }
        static std::set<ggml_op> ops_not_support_view_input{};
        if (ops_not_support_view_input.find(op->op) != ops_not_support_view_input.end() && has_view_op_input(op)) {
            return {false, "op " + std::string(ggml_op_name(op->op)) + " with VIEW input is not supported"};
        }
    }
    }

    if (supported_types.find(op->type) == supported_types.end()) {
//>> ★ 第二道门槛：输出张量类型必须在白名单里
        return {false, "tensor type " + std::string(ggml_type_name(op->type)) + " is not supported"};
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto * src = op->src[i];
        if (src == nullptr) {
            break;
        }
        if (supported_types.find(src->type) == supported_types.end()) {
//>> 每个非空输入的类型也必须在白名单里 —— 判据遍历 src[]，不只是 src[0]
            return {false, "src[" + std::to_string(i) + "] type " + std::string(ggml_type_name(src->type)) + " is not supported"};
        }
        const bool is_supported_3d_moe_expert =
//>> 唯一放行的 3D 量化例外：MUL_MAT_ID 的专家权重（MoE）
            op->op == GGML_OP_MUL_MAT_ID && i == 0 && (src->type == GGML_TYPE_MXFP4 || src->ne[3] == 1);
        if (ggml_is_quantized(src->type) && src->ne[2] != 1 && !is_supported_3d_moe_expert) {
//>> ★ 第三道门槛：量化张量的 ne[2] 必须为 1，否则整个 op 落到别的后端
            return {false, "3D quantized tensor for src[" + std::to_string(i) + "] is not supported"};
        }
    }

    auto op_support_case = is_op_supported_case(op);
//>> 最后是设备 / 形状级的例外表（is_op_supported_case，1129 行起）：ROPE 的 ne[3]、TRANSPOSE 的 BF16、GPU 上的 REPEAT 等
    if (!op_support_case.is_supported) {
        return op_support_case;`,
  duration: 26000,
  build(root, tl) {
    const LN = {"1495": 0, "1496": 2, "1497": 3, "1498": 5, "1499": 6, "1500": 7, "1501": 8, "1502": 10, "1503": 11, "1504": 12, "1505": 13, "1506": 14, "1507": 15, "1508": 17, "1509": 18, "1510": 19, "1511": 20, "1512": 21, "1513": 22, "1514": 23, "1515": 25, "1516": 26, "1517": 27, "1518": 28, "1519": 29, "1520": 30, "1521": 31, "1522": 33, "1523": 34, "1524": 35, "1525": 36, "1526": 37, "1527": 38, "1528": 39, "1529": 40, "1530": 41, "1531": 42, "1532": 43, "1533": 45, "1534": 46, "1535": 47, "1536": 48, "1537": 49, "1538": 50, "1539": 51, "1540": 52, "1541": 54, "1542": 55, "1543": 56, "1544": 58, "1545": 59, "1546": 61, "1547": 62, "1548": 63, "1549": 64, "1550": 65, "1551": 67, "1552": 68};
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `
      <div class="row" style="gap:8px">
        <div class="col grow" id="cut" style="gap:6px"></div>
        <div class="col grow" id="why" style="gap:6px"></div>
      </div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const cut = wrap.querySelector('#cut');
    cut.innerHTML = '<div class="cm" style="margin-bottom:3px">一段子图（拓扑序）</div>';
    const seq = ['MUL_MAT', 'RMS_NORM', 'ROPE', 'SOFT_MAX', 'FLASH_ATTN_EXT', 'GET_ROWS', 'ADD'];
    const cells = seq.map(n => {
      const e = U.el('div', { class: 'formula', style: 'padding:3px 7px;font-size:9.5px' });
      e.innerHTML = '<span style="color:var(--b)">&#10003;</span> ' + U.esc(n);
      cut.appendChild(e);
      return e;
    });
    const why = wrap.querySelector('#why');
    why.innerHTML =
      '<div class="card" style="border-left-color:var(--a);width:100%">' +
      '<div class="ct" style="color:var(--a);font-size:10px">第一道：有翻译器吗</div>' +
      '<div class="cb">table.count("GGML_OP_" + ggml_op_name(op))。<br>缺失 = 这个 op 永远进不来。</div></div>' +
      '<div class="card" style="border-left-color:var(--b);width:100%">' +
      '<div class="ct" style="color:var(--b);font-size:10px">第二道：类型在白名单吗</div>' +
      '<div class="cb">op-&gt;type 与每个 src[i]-&gt;type 都要在 13 种里：<br>' +
      'F32 F16 BF16 I32 I64 · Q4_0 Q4_1 Q4_K Q5_1 Q5_K Q6_K Q8_0 MXFP4</div></div>' +
      '<div class="card" style="border-left-color:var(--c);width:100%">' +
      '<div class="ct" style="color:var(--c);font-size:10px">第三道：形状 / 设备例外</div>' +
      '<div class="cb">量化张量 ne[2] != 1 直接否决（MoE 的 MUL_MAT_ID 除外）；<br>' +
      '再叠加 is_op_supported_case 的设备例外。</div></div>';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '把三道门槛叠在一段子图上：<b>任何一格被否决，调度器就在那里断一刀</b>（L4-02 幕 6）。',
      '第一道是布尔查表：<span class="v">supported_ops / supported_unary_ops / supported_glu_ops</span>。' +
        '<span class="k">有翻译器</span>才谈得上后面两道。',
      '第二道遍历 <span class="v">op-&gt;type</span> 和每个非空 <span class="v">src[i]-&gt;type</span>。' +
        '它解释了为什么同一个 op 有时能 offload、有时不能 —— <b>精度决定归属</b>。',
      '第三道是形状：<span class="v">ggml_is_quantized(src-&gt;type) &amp;&amp; src-&gt;ne[2] != 1</span> 一律否决。' +
        '量化的 3D 专家权重是唯一例外（MUL_MAT_ID）。',
      '所以"什么形状的子图交给 OpenVINO"的答案是：<span class="k">全部节点都过三道门槛的、最长的连续区间</span>。<br>' +
        '一格不过，那一段就整段留在别的后端上 —— 最后还有设备级例外表兜底。'
    ];
    tl.at(700, () => {
      msg.innerHTML = texts[0];
      cells.forEach((c, i) => { c.style.opacity = i === 0 ? '1' : '.35'; });
      U.markLines(document, [LN[1495], LN[1496]]);
    });
    tl.at(4400, () => { msg.innerHTML = texts[1]; U.markLines(document, [LN[1497], LN[1507], LN[1521], LN[1522], LN[1523]]); });
    tl.at(9600, () => { msg.innerHTML = texts[2]; U.markLines(document, [LN[1532], LN[1533], LN[1540], LN[1541]]); });
    tl.at(15200, () => { msg.innerHTML = texts[3]; U.markLines(document, [LN[1543], LN[1544], LN[1545], LN[1546]]); });
    tl.at(20000, () => {
      cells.forEach(c => { c.style.opacity = '1'; });
      msg.innerHTML = texts[4];
      U.markLines(document, [LN[1550], LN[1551]]);
    });
  }
},

/* ------------------------------------------------------ 6 一个后端，三种设备：<span class="hl-c">NPU</span> / <span class="hl-c">GPU</span> / <span class="hl-c">CPU</span> */
{
  kicker: "L7-03 · 设备选择",
  title: "一个后端，三种设备：<span class=\"hl-c\">NPU</span> / <span class=\"hl-c\">GPU</span> / <span class=\"hl-c\">CPU</span>",
  sub: "设备名从 GGML_OPENVINO_DEVICE 读，默认 CPU；不可用就回退 CPU。设备一旦定下，编译配置与判据分支都跟着变。",
  caption: "设备名不只是\"选硬件\"：它直接进入 supports_op 的例外判断（如 GPU 上的 BF16 TRANSPOSE / REPEAT、NPU 上的 BF16 输入）。",
  src: "ggml/src/ggml-openvino/ggml-openvino-extra.cpp",
  mark: [0, 2, 4, 6, 7, 9, 13, 15, 17, 18, 36],
  lineNo: 74,
  code: `    device_name = ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE", "CPU");
//>> 默认值是 "CPU" —— 不设环境变量时，这个后端跑在 OpenVINO 的 CPU 插件上
    auto available_devices = ov_singleton_core().get_available_devices();
//>> 可用性来自 OpenVINO Core 自己枚举，不是 ggml 猜的
    if (std::find(available_devices.begin(), available_devices.end(), device_name) == available_devices.end()) {
//>> 不可用就只警告一声、退回 CPU —— 后端不会因此拒绝初始化
        GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\\n", device_name.c_str());
        device_name = "CPU";
    }
    is_npu = (device_name == "NPU");
//>> is_npu 是热路径上的一个 bool（utils.cpp:1470 用它选 static/dynamic 执行路径）

    const char * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    if (device_name == "NPU") {
//>> ★ NPU 分支：整份 compile_config 只在设备名是 NPU 时才填
        compile_config = {
//>> NPUW = NPU 插件的权重银行 / 函数调用模式；这一组 key 是 NPU 能跑起来的全部前提
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        if (cache_dir && strlen(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        const char * compilation_mode_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        if (compilation_mode_params && strlen(compilation_mode_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_mode_params;
        }
    } else if (cache_dir && strlen(cache_dir) > 0) {
//>> 非 NPU 分支只有缓存配置 —— GPU 走的是另一条路（建 OpenCL remote context，115-148 行）`,
  duration: 22000,
  build(root, tl) {
    const LN = {"74": 0, "75": 2, "76": 4, "77": 6, "78": 7, "79": 8, "80": 9, "81": 11, "82": 12, "83": 13, "84": 15, "85": 17, "86": 18, "87": 19, "88": 20, "89": 21, "90": 22, "91": 23, "92": 24, "93": 25, "94": 26, "95": 27, "96": 28, "97": 29, "98": 30, "99": 31, "100": 32, "101": 33, "102": 34, "103": 35, "104": 36};
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `<div class="row wrap" id="dev" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const host = wrap.querySelector('#dev');
    const devs = [
      { c: 'a', t: 'NPU', b: 'compile_config 装 NPU_USE_NPUW / NPUW_* / 动态量化；<br>执行走 <b>static</b> 形状路径；<br>KV cache 不做 stateful。', m: 'GGML_OPENVINO_DEVICE=NPU' },
      { c: 'b', t: 'GPU', b: '建 OpenCL context/queue 并共享给 OV（115-148 行）；<br>执行走 <b>dynamic</b> 形状路径；<br>可开 stateful、权重可释放。', m: 'GGML_OPENVINO_DEVICE=GPU' },
      { c: 'c', t: 'CPU', b: '默认值，也是所有回退的落点；<br>没有 remote context、没有 NPUW；<br>用于无 Intel 加速硬件时验证翻译是否正确。', m: 'GGML_OPENVINO_DEVICE=CPU' }
    ];
    const els = devs.map(d => { const e = U.card(d, { style: 'width:216px' }); host.appendChild(e); return e; });
    els.forEach(e => { e.style.opacity = '.30'; });
    const msg = wrap.querySelector('#msg');
    const texts = [
      '设备名只有一个来源：环境变量 <span class="v">GGML_OPENVINO_DEVICE</span>，默认 <span class="k">CPU</span>。',
      '<span class="v">NPU</span>：唯一需要"改编译配置"的设备。NPUW 那一组 key 说明 NPU 不是直接吃 IR，' +
        '而是先被 <b>NPU 插件切分、折叠、权重入库</b>。',
      '<span class="v">GPU</span>：唯一需要"共享队列"的设备 —— 建 OpenCL context/queue 交给 OV，避免每步同步。' +
        '它同时是 NPU 分支的 else。',
      '<span class="v">CPU</span>：既是默认值，也是 <span class="k">get_available_devices()</span> 查不到时的回退目标。',
      '★ 设备名会渗透进 <span class="k">supports_op</span>：同一张图在 GPU 上可能比在 NPU 上多几个节点被接受。<br>' +
        '所以"支持什么"是 <b>(op, type, shape, device)</b> 四元组的函数，不是 op 的函数。'
    ];
    tl.at(700, () => { els.forEach((e, i) => { e.style.opacity = i === 2 ? '1' : '.30'; }); msg.innerHTML = texts[0]; U.markLines(document, [LN[74], LN[75]]); });
    tl.at(4600, () => { els.forEach((e, i) => { e.style.opacity = i === 0 ? '1' : '.30'; }); msg.innerHTML = texts[1]; U.markLines(document, [LN[83], LN[84], LN[85], LN[86]]); });
    tl.at(9800, () => { els.forEach((e, i) => { e.style.opacity = i === 1 ? '1' : '.30'; }); msg.innerHTML = texts[2]; U.markLines(document, [LN[104]]); });
    tl.at(14600, () => { els.forEach((e, i) => { e.style.opacity = i === 2 ? '1' : '.30'; }); msg.innerHTML = texts[3]; U.markLines(document, [LN[75], LN[76], LN[77], LN[78]]); });
    tl.at(19000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [LN[80]]); });
  }
},

/* ------------------------------------------------------ 7 不支持的 op <span class="hl-a">不需要回退代码</span>：它根本不会被切进来 */
{
  kicker: "L7-03 · 回退策略",
  title: "不支持的 op <span class=\"hl-a\">不需要回退代码</span>：它根本不会被切进来",
  sub: "运行时真正要判断的，是\"这次拿到的是整图、还是被切碎的一段\" —— 后者不能走缓存与 naive 捷径。",
  caption: "回顾 L4-02 幕 6：切分 = 沿图找连续的同后端段。supports_op 就是那把刀。",
  src: "ggml/src/ggml-openvino/utils.cpp",
  mark: [0, 6, 8, 11, 14, 17, 19, 21],
  lineNo: 640,
  code: `    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
//>> 注释直说：这个检测是 O(n^2)，在 Llama-1B 的 decode 图上要 ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    graph_key key(cgraph);
//>> graph_key 从 cgraph 构造；注释说它只要几百微秒，比逐节点比较便宜得多
    bool key_seen = false;
    if (!cache_disabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        key_seen = r_ctx->decoder_cache.find(key) != r_ctx->decoder_cache.end();
    }

    bool model_is_splitted = key_seen ? false : is_model_splitted(cgraph);
//>> 只有在缓存里没见过时才真的去算 is_model_splitted

    if (is_naive(cgraph)) {
//>> naive = 节点数 < 20 的小图（is_naive 在 1566 行）：直接一股脑翻译，不做权重提取与缓存
        if (!model_is_splitted) {
//>> 小图 + 没被切过 -> 走 naive_compute，一次性翻译整图
            return naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
//>> 被切过的小图不享受这条捷径：碎片图之间可能存在后端边界，缓存与权重假设都不成立
        }
    }

    auto start_time = ggml_time_us();`,
  duration: 20000,
  build(root, tl) {
    const LN = {"640": 0, "641": 2, "642": 3, "643": 4, "644": 5, "645": 6, "646": 8, "647": 9, "648": 10, "649": 11, "650": 12, "651": 13, "652": 14, "653": 16, "654": 17, "655": 19, "656": 21, "657": 23, "658": 24, "659": 25, "660": 26};
    const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
    wrap.innerHTML = `
      <div class="flow" style="justify-content:center" id="flow2"></div>
      <div class="row wrap" id="cards2" style="gap:8px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const f = wrap.querySelector('#flow2');
    [
      { t: 'supports_op', c: 'a' }, { t: 'scheduler 切分', c: 'b' },
      { t: 'graph_compute 收到一段', c: 'c' }, { t: 'is_model_splitted', c: 'd' },
      { t: 'naive / 缓存路径', c: 'e' }
    ].forEach((s, i) => {
      const e = U.chip(s.t, s.c);
      e.style.fontSize = '9.5px';
      f.appendChild(e);
      if (i < 4) f.appendChild(U.arrow('->'));
    });

    const host = wrap.querySelector('#cards2');
    const defs = [
      { c: 'a', t: '不属于本后端的 op', b: '不写"软件实现"。它由 L4-02 的 scheduler 判给别的后端，<br>跨段数据由 scheduler 插 copy 搬运。' },
      { c: 'b', t: '属于本后端但形状被拒', b: '同样在切分阶段就出局 —— 判据是同一份代码。<br>reason 字符串只在打开日志开关时打印。' },
      { c: 'c', t: 'GGML_OPENVINO_ENABLE_FALLBACK', b: '开启后（1492 行起）才做 is_model_splitted 检测：<br>用 use_count 与输入引用数对不上来识别"这是碎片"。' }
    ];
    const els = defs.map(d => { const e = U.card(d, { style: 'width:216px' }); host.appendChild(e); return e; });
    els.forEach(e => { e.style.opacity = '.30'; });

    const msg = wrap.querySelector('#msg');
    const texts = [
      '回退不是一个 try/catch，而是<b>两个阶段的分工</b>：切分阶段筛掉，执行阶段识别碎片。',
      '阶段一：<span class="k">supports_op 说不</span> -> 调度器把节点判给别的后端。OpenVINO 这边什么代码都不用写。',
      '阶段二：真的进来了但形状被拒？不会发生 —— 判据是同一份代码。日志开关只影响"说不说得出来为什么"。',
      '<span class="v">is_model_splitted</span> 是在<b>运行时</b>反推"我拿到的是一段还是全图"：<br>' +
        '节点的 <span class="k">use_count</span> 与"有多少节点把它当输入"对不上，就是碎片。',
      '<span class="v">is_naive</span> 另有一条捷径：节点数 &lt; 20 且没被切过，就一次性翻译整图（naive_compute）。',
      '★ 结论：OpenVINO 后端的"回退"是 <b>缺席</b>，不是分支。这跟写 kernel 的后端（L5/L6）完全相反。'
    ];
    tl.at(700, () => { els.forEach((e, i) => { e.style.opacity = i === 0 ? '1' : '.30'; }); msg.innerHTML = texts[0]; U.markLines(document, [LN[640]]); });
    tl.at(3800, () => { msg.innerHTML = texts[1]; U.markLines(document, [LN[652]]); });
    tl.at(7200, () => { els.forEach((e, i) => { e.style.opacity = i === 1 ? '1' : '.30'; }); msg.innerHTML = texts[2]; U.markLines(document, [LN[640]]); });
    tl.at(10600, () => { els.forEach((e, i) => { e.style.opacity = i === 2 ? '1' : '.30'; }); msg.innerHTML = texts[3]; U.markLines(document, [LN[645], LN[646], LN[649]]); });
    tl.at(14200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; U.markLines(document, [LN[654], LN[655], LN[656]]); });
    tl.at(17600, () => { msg.innerHTML = texts[5]; U.markLines(document, [LN[640], LN[652], LN[654]]); });
  }
},

/* ------------------------------------------------------ 8 一个 ggml op 进去，<span class="hl-e">一串</span> ov::op 出来 */
{
  kicker: "L7-03 · 翻译粒度",
  title: "一个 ggml op 进去，<span class=\"hl-e\">一串</span> ov::op 出来",
  sub: "RMS_NORM 是最典型的例子：6 个 ov 节点、零个 kernel。这是\"翻译\"这个视角最直接的证据。",
  caption: "对照组：FLASH_ATTN_EXT 反过来 —— 一个 ggml op 落进一个融合算子 v13::ScaledDotProductAttention（flash_attn_ext.cpp:228/232）。",
  src: "ggml/src/ggml-openvino/openvino/op/rms_norm.cpp",
  mark: [0, 3, 5, 7, 9, 11, 13, 15, 17, 19],
  lineNo: 57,
  code: `    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);
//>> square = input * input —— 用 Multiply 自乘表达

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
//>> ReduceMean over {-1}，keep_dims=true：这就是 ggml 的"最后维求均值"
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    float eps;
//>> eps 从 op_params 里按 float 读出来 —— 与 L1-02 幕 6 说的"16 个 int32 的 schema 只是注释"呼应
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
//>> sqrt(mean + eps)：Add + Sqrt
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
//>> 1 / rms：Divide
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);
//>> 最后乘回输入 —— 到这里已经 6 个 ov 节点了，而 ggml 侧只有 1 个

    return rename_outputs_with_suffix({res}, context.get_name());`,
  duration: 20000,
  build(root, tl) {
    const LN = {"57": 0, "58": 2, "59": 3, "60": 5, "61": 6, "62": 7, "63": 9, "64": 10, "65": 11, "66": 13, "67": 14, "68": 15, "69": 17, "70": 18, "71": 19, "72": 21, "73": 22};
    const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
    wrap.innerHTML = `
      <div class="row center" id="row1" style="gap:6px"></div>
      <div class="row wrap center" id="row2" style="gap:5px"></div>
      <div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const mk = (host, txt, cls) => {
      const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px;white-space:nowrap' });
      e.innerHTML = '<span style="color:var(--' + cls + ')">' + U.esc(txt) + '</span>';
      host.appendChild(e);
      return e;
    };
    const r1 = wrap.querySelector('#row1');
    const r2 = wrap.querySelector('#row2');

    const ggmlSide = mk(r1, '1 x GGML_OP_RMS_NORM', 'a');
    r1.appendChild(U.arrow('->'));
    const note = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
    note.innerHTML = '<span class="m">translate_rms_norm()</span>';
    r1.appendChild(note);

    const flat = [];
    [['v1::Multiply', 'b'], ['v1::ReduceMean', 'c'], ['v1::Add', 'd'],
     ['v0::Sqrt', 'e'], ['v1::Divide', 'f'], ['v1::Multiply', 'g']].forEach((p, i) => {
      if (i) r2.appendChild(U.arrow('+'));
      flat.push(mk(r2, p[0], p[1]));
    });
    const cnt = mk(r2, '= 6 个 ov 节点', 'a');
    cnt.style.opacity = '.8';

    const msg = wrap.querySelector('#msg');
    const texts = [
      '左边 1 个 ggml 节点，右边 6 个 ov 节点。中间<b>没有一行 kernel 代码</b>。',
      '<span class="v">input * input</span>：用 Multiply 自乘表达"平方"。',
      '<span class="v">v1::ReduceMean(square, {-1}, true)</span>：ggml 的 rms_norm 永远沿最后一维，所以轴是常量 -1。',
      '<span class="v">Add(mean, eps)</span> 的 eps 从 <span class="k">op_params</span> 读 —— ' +
        '标量参数靠 memcpy 出来，没有类型检查（L1-02 幕 6）。',
      '<span class="v">Sqrt -&gt; Divide(1, rms) -&gt; Multiply(input)</span> 收尾。' +
        '★ 每一步都是"翻译"，没有一步是"实现"。',
      '反过来也成立：<span class="k">flash_attn_ext</span> 整块注意力落进一个融合算子 SDPA（第 228/232 行）。' +
        '粒度由 ov 算子集决定，不由 ggml 决定。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; flat.forEach(e => { e.style.opacity = '.30'; }); U.markLines(document, [LN[57]]); });
    tl.at(3600, () => { msg.innerHTML = texts[1]; flat[0].style.opacity = '1'; U.markLines(document, [LN[57]]); });
    tl.at(7000, () => { msg.innerHTML = texts[2]; flat[1].style.opacity = '1'; U.markLines(document, [LN[59], LN[60]]); });
    tl.at(10600, () => { msg.innerHTML = texts[3]; flat[2].style.opacity = '1'; U.markLines(document, [LN[62], LN[63]]); });
    tl.at(14200, () => { msg.innerHTML = texts[4]; flat.forEach(e => { e.style.opacity = '1'; }); U.markLines(document, [LN[65], LN[66], LN[68], LN[69], LN[71]]); });
    tl.at(17600, () => { msg.innerHTML = texts[5]; cnt.style.opacity = '1'; U.markLines(document, [LN[57], LN[71]]); });
  }
},

/* ------------------------------------------------------ 9 一张表说清 OpenVINO 后端 */
{
  kicker: "L7-03 · 收束",
  title: "一张表说清 OpenVINO 后端",
  sub: "支持判据、设备、回退、粒度 —— 四个问题，四个答案，都在同一个设计决定下。",
  caption: "下一课 L7-04：ExecuTorch 用\"委托\"做同一件事 —— 但委托的边界是另一套抽象。",
  src: "ggml/src/ggml-openvino/openvino/op_table.cpp",
  mark: [0, 1, 2, 3, 4, 6, 8],
  lineNo: 76,
  code: `        {"GGML_OP_DIAG",            op::translate_diag                             },
        {"GGML_OP_TRI",             op::translate_tri                              },
        {"GGML_OP_SET",             op::translate_set                              },
        {"GGML_OP_POOL_2D",         op::translate_pool_2d                          },
        {"GGML_OP_ROLL",            op::translate_roll                             },
//>> ROLL 是这张表的最后一项
        // solve_tri has accuracy issues on GPU
//>> ★ 注释也是能力边界：把一行注释掉 = 把这个算子永久踢出 OpenVINO 子图
        // {"GGML_OP_SOLVE_TRI",       op::translate_solve_tri                        },
//>> SOLVE_TRI 的翻译器 translate_solve_tri 仍然存在（solve_tri.cpp:33），但没被登记进表里 —— 于是 supports_op 会拒绝这个 op，翻译器成了死代码
    };`,
  duration: 20000,
  build(root, tl) {
    const LN = {"76": 0, "77": 1, "78": 2, "79": 3, "80": 4, "81": 6, "82": 8, "83": 10};
    const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
    wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
    root.appendChild(wrap);

    const t = U.table(
      ['问题', '答案', '代码位置'],
      [['谁来决定一个 op 能不能算？', 'op_table.cpp 里有没有它的 key', 'ggml-openvino.cpp:1470'],
       ['什么形状的子图整体交给它？', '全部节点都过三道门槛的最长连续区间', 'ggml-openvino.cpp:1532-1548'],
       ['跑在哪台设备上？', 'GGML_OPENVINO_DEVICE，默认 CPU，不可用回退 CPU', 'ggml-openvino-extra.cpp:74-80'],
       ['不支持的 op 怎么办？', '不写回退代码，调度器根本不会切给它', 'utils.cpp:640-658'],
       ['一个 op 翻译成几个 ov::op？', '0 到 N 个，由 ov 算子集决定（RMS_NORM=6，FA=1）', 'op/rms_norm.cpp:57-73'],
       ['后端到底写了什么？', '45 个翻译器 + 12 个 OV pass，没有 kernel', 'ggml/src/ggml-openvino/']],
      { monoCols: [2] });
    wrap.querySelector('#tbl').appendChild(t.el);

    wrap.querySelector('#ex').appendChild(W.exercise(
      '给定一个 ggml 节点 <span class="mono">GGML_OP_REPEAT</span>，' +
      '<span class="mono">op-&gt;type = GGML_TYPE_BF16</span>，设备是 ' +
      '<span class="mono">GGML_OPENVINO_DEVICE=GPU</span>。它会被 OpenVINO 接受吗？',
      '<b>不会。</b><br>' +
      'REPEAT 在 op_table.cpp:73 有翻译器，BF16 也在 13 种白名单里，所以前两道门槛都过。<br>' +
      '但第三道 <span class="mono">is_op_supported_case</span> 里有一条例外：' +
      '<span class="mono">ggml_openvino_get_device_name() == "GPU" &amp;&amp; op-&gt;type == GGML_TYPE_BF16</span> ' +
      '时返回 <span class="mono">{false, "REPEAT with BF16 type is not supported on GPU"}</span>。<br>' +
      '它被拒之后，L4-02 的调度器会在这一点上断一刀 —— OpenVINO 后端不会为它写任何回退代码。'));

    const msg = wrap.querySelector('#msg');
    const rows = t.body.querySelectorAll('tr');
    const texts = [
      '六个问题。前两个是本课的验收点，后四个是它的推论。',
      '<span class="k">能不能算</span> = 表里有没有；<span class="k">什么形状</span> = 三道门槛的交集。',
      '<span class="k">设备</span> 既是选择也是判据的一部分：同一张图在 NPU/GPU/CPU 上边界不同。',
      '<span class="k">回退</span> 在这个后端里是"缺席" —— 与 L5/L6 写 kernel 的后端相反。',
      '<span class="k">粒度</span> 由 OpenVINO 的算子集决定：翻译器可以把 1 个拆成 6 个，也可以把 1 个塞进 1 个。',
      '记住一句话：<span class="v">OpenVINO 后端是一台"图到图"的编译器前端</span>，' +
        '它的能力边界就写在一张 55 项的 map 里。'
    ];
    tl.at(700, () => { msg.innerHTML = texts[0]; });
    rows.forEach((r, i) => tl.at(2400 + i * 2600, () => {
      rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
      msg.innerHTML = texts[Math.min(i + 1, 5)];
    }));
    tl.at(18200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; U.markLines(document, [LN[81], LN[82]]); });
  }
},

];
