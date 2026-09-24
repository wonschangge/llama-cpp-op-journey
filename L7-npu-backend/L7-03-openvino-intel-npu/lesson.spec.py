#!/usr/bin/env python3
"""L7-03 · OpenVINO 后端（Intel NPU/GPU）—— 课件 spec。

运行：python3 L7-npu-backend/L7-03-openvino-intel-npu/lesson.spec.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402
import plan_matrix as pmat             # noqa: E402

# ------------------------------------------------------------------ 覆盖清单
#
# 82 个文件从 plan_matrix 的指派里【现取】，不手抄 —— 手抄一份清单迟早会和计划漂移。
#   python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L7-03"{print $2}'
_ASSIGN, _UNASSIGNED = pmat.resolve()
MANIFEST = sorted(_ASSIGN['L7-03'])
assert len(MANIFEST) == 82, f'L7-03 清单应为 82 个文件，实际 {len(MANIFEST)}'
assert not _UNASSIGNED, f'计划里还有未指派文件: {_UNASSIGNED[:5]}'

_MOD = 'ggml/src/ggml-openvino'


def _family(p):
    """按目录把 82 个文件分成五族（计数由 FAMILY_COUNT 断言，实测得出）。"""
    if p.startswith(_MOD + '/openvino/op/'):
        return 'op 翻译器'
    if p.startswith(_MOD + '/openvino/pass/'):
        return 'OV pass'
    if '/rt_info/' in p:
        return 'rt_info'
    if p.startswith(_MOD + '/openvino/'):
        return 'frontend 骨架'
    return '顶层胶水'


FAMILY_ORDER = ['顶层胶水', 'frontend 骨架', 'op 翻译器', 'OV pass', 'rt_info']
FAMILY_COUNT = {f: sum(1 for p in MANIFEST if _family(p) == f) for f in FAMILY_ORDER}
assert FAMILY_COUNT == {'顶层胶水': 12, 'frontend 骨架': 12,
                        'op 翻译器': 45, 'OV pass': 12, 'rt_info': 1}, FAMILY_COUNT
MANIFEST_SORTED = sorted(MANIFEST, key=lambda p: (FAMILY_ORDER.index(_family(p)), p))

OW = 'ggml/src/ggml-openvino/ggml-openvino.cpp'
EXTRA = 'ggml/src/ggml-openvino/ggml-openvino-extra.cpp'
UTILS = 'ggml/src/ggml-openvino/utils.cpp'
HDR = 'ggml/include/ggml-openvino.h'
OPT = 'ggml/src/ggml-openvino/openvino/op_table.cpp'
SESSION = 'ggml/src/ggml-openvino/openvino/translate_session.cpp'
RMS = 'ggml/src/ggml-openvino/openvino/op/rms_norm.cpp'
FA = 'ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp'


def ridx(parts, notes_src, lines):
    """上游行号 -> 【渲染下标】的映射（与 lessonkit._render_code 同一算法）。

    为什么需要它：`U.markLines(document, [...])` 吃的是**渲染下标**（含注解行），
    而注解行是由 notes_src 插进去的 —— 手数下标必然错位（AUTHORING.md §3.1 的坑）。
    这里直接按 lessonkit 的同一套规则算出来，注入到 visual 里。
    """
    notes = notes_src or {}
    srcmap = []
    for pi, (a, b) in enumerate(parts):
        if pi:
            srcmap.append(None)
        for ln in range(a, b + 1):
            srcmap.append(ln)
            if ln in notes:
                srcmap.append(None)
    return {str(ln): srcmap.index(ln) for ln in lines}


def mklines(parts, notes_src):
    """把整段引用做成一整张 行号->下标 表，交给 JS 里的 LN[...] 用。"""
    lines = []
    for (a, b) in parts:
        lines += list(range(a, b + 1))
    return json.dumps(ridx(parts, notes_src, lines))


L = Lesson(
    id='L7-03',
    layer='L7 · NPU 与加速器后端',
    title='OpenVINO 后端：把 ggml 图翻译成另一套图 IR',
    codecap='ggml/src/ggml-openvino/**（逐字引用）',
    nav={'prev': {'href': '../L7-02-hexagon-qualcomm/index.html', 'label': 'L7-02 Hexagon 后端'},
         'next': {'href': '../L7-04-executorch-edge/index.html', 'label': 'L7-04 ExecuTorch 后端'}},
)

L.note('**一句话**：OpenVINO 后端不写 kernel。它把调度器切给它的那段 ggml 子图'
       '**翻译**成 OpenVINO 的图 IR（`ov::Model`），再交给 Intel 的 NPU / GPU / CPU 插件去编译。')
L.note('这个视角解释了本课后面所有的"奇怪"设计：\n\n'
       '- 支持判据不是"我能不能算得快"，而是"**这个 op 有没有对应的 `ov::op`**"；\n'
       '- 所以判据是**一次问一个 op**（`supports_op`），但**一次交出一整段子图**（`graph_compute`）；\n'
       '- 不支持的 op 不需要"软件回退"，它**根本不会被切进来** —— 切分发生在 L4-02 的调度器里。')
L.note('本课覆盖 **82 个文件**：`ggml/include/ggml-openvino.h` + `ggml/src/ggml-openvino/` 下的 81 个。'
       '其中 45 个在 `openvino/op/` 下，一个文件（最多）负责一个（或一族）ggml 算子到 `ov::op` 的翻译。')

# ==================================================================== 第 1 幕

VIS1 = '''
const LN = /*__LN1__*/null;
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
'''

P1 = [(11, 33)]
N1 = {11: '后端名字：注册表里的 key 就是它（见 L3-02）',
      14: '唯一的创建入口：device 号进去，backend 句柄出来',
      18: 'buffer 判定 —— 调度器靠它识别"这个张量在 OpenVINO 手里"',
      27: 'device buffer type：这个后端的默认 buffer 类型（见 L4-03）',
      29: 'host buffer type：另一类 buffer，调度器按需在两者间搬运',
      31: 'device count：告诉调度器这个后端有几台设备可以切（第 6 幕展开）'}

L.scene(
    kicker='L7 · NPU 与加速器后端',
    title='OpenVINO 后端：<span class="hl-a">翻译图</span>，而不是写 kernel',
    sub='公共头只有 23 行有效内容、11 个 C 函数。真正的实现是 81 个 C++ 文件构成的"前端"。',
    caption='L7-01 CANN 走的是同一路线（映射到厂商算子），L7-02 Hexagon 则把整图丢给远端 DSP；三课对照着看。',
    src=HDR, parts=P1, duration=21000,
    mark_src=[11, 14, 18, 27, 29, 31],
    notes_src=N1,
    visual=VIS1.replace('/*__LN1__*/null', mklines(P1, N1))
)

# ==================================================================== 第 2 幕

VIS2 = '''
const LN = /*__LN2__*/null;
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
'''

P2 = [(297, 307)]
N2 = {297: '这个 lambda 就是"翻译一个节点"的全部',
      298: 'decoder->get_op_type() 返回的是【宏名】，如 "GGML_OP_MUL_MAT" —— 与 L1-02 的 ggml_op_name() 同源',
      299: 'GGML_OP_NONE 不算算子，直接跳过（L1-02 幕 7：新张量的默认身份）',
      303: '★ 支持与翻译共用同一张表：能不能算 = 表里有没有这个名字',
      304: '查不到就直接抛异常 —— 所以 supports_op 必须在切分阶段就把它挡在外面',
      307: '翻译函数返回 ov::OutputVector：一个 ggml 节点可以变成好几个 ov::op'}

L.scene(
    kicker='L7-03 · 核心洞察',
    title='★ 一个 ggml 节点 = <span class="hl-a">一次查表</span>',
    sub='翻译的骨架只有十几行：取 op 名字，去表里找翻译函数，调用它，把结果登记进 tensor_map。',
    caption='对比 L5 系列：CPU 后端在同样的位置是 ggml_compute_forward 里一个按 tensor->op 分支的大 switch，每个 case 调一个 kernel。这里没有 switch。',
    src=SESSION, parts=P2, duration=19000,
    mark_src=[297, 298, 299, 303, 304, 306, 307],
    notes_src=N2,
    visual=VIS2.replace('/*__LN2__*/null', mklines(P2, N2))
)

# ==================================================================== 第 3 幕

VIS3 = '''
const LN = /*__LN3__*/null;
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
'''

P3 = [(23, 60)]
N3 = {23: '整张表就是一次 return {...}，没有分支、没有条件编译',
      26: 'ADD 要处理 MoE 的专家平面求和与类型混用，所以有专属实现（add.cpp）',
      27: '★ ADD1 走 1to1 模板 —— 标量加法与张量加法在 ov 侧是同一个 v1::Add',
      35: '★ MUL 只需要一个 ov::op::v1::Multiply —— 模板自动补齐输入检查与类型转换',
      36: 'MUL_MAT 是 45 个翻译器里最关键的一个，输出 ov::op::v0::MatMul（见 mulmat.cpp）',
      39: 'RESHAPE / PERMUTE / VIEW 这些"视图算子"也有翻译器：OpenVINO 图里必须显式表达',
      52: 'GELU / SIGMOID / SILU / TANH / EXP / NEG / RELU 全部是一行模板特化',
      60: 'VIEW 有专属实现，因为它要把 ggml 的 nb[] 语义折算成 Slice + Reshape（view.cpp）'}

L.scene(
    kicker='L7-03 · 翻译表',
    title='<span class="hl-c">op_table.cpp</span>：后端能力的唯一真相来源',
    sub='一张 unordered_map：key 是 ggml 的 op 宏名，value 是翻译函数。表里有什么，后端就能算什么。',
    caption='第 61-80 行还有 SWIGLU / GEGLU / SET_ROWS / CPY / CLAMP / PAD / REPEAT / DIAG / TRI / SET / ROLL 等；完整 55 项见 source.md 的逐字引用。',
    src=OPT, parts=P3, duration=24000,
    mark_src=[23, 26, 27, 33, 34, 35, 36, 39, 40, 41, 48, 49, 50, 52, 54, 60],
    notes_src=N3,
    visual=VIS3.replace('/*__LN3__*/null', mklines(P3, N3))
)

# ==================================================================== 第 4 幕

VIS4 = '''
const LN = /*__LN4__*/null;
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
'''

P4 = [(1449, 1493)]
N4 = {1449: '返回 (bool, reason)：不支持时带一句人能读的理由，便于定位',
      1452: '★ 张量类型白名单，13 种。量化类型只有 Q4_0/Q4_1/Q4_K/Q5_1/Q5_K/Q6_K/Q8_0/MXFP4',
      1461: '取的就是 op_table.cpp 里那张表 —— 没有第二份"支持列表"',
      1470: '遍历 ggml 的全部 op 枚举，逐个拼 key 去查表',
      1471: 'key 形如 "GGML_OP_MUL_MAT"；这正是 translate_session 查表用的同一个字符串',
      1475: 'unary 与 glu 是嵌在 GGML_OP_UNARY / GGML_OP_GLU 里的子枚举，所以各建一张集合',
      1488: 'GGML_OP_NONE 没有翻译器，但它不算算子，显式放行',
      1490: '函数局部 static：整个进程只建一次（表是编译期常量，不会变）'}

L.scene(
    kicker='L7-03 · 支持判据（一）',
    title='★ <span class="hl-a">supported_ops</span> 不是手写的清单，是从翻译表派生的',
    sub='三道门槛的第一道：这个 op（或 unary / glu 子枚举）的名字，在 op_table 里有没有？',
    caption='验收点答案就藏在 build_supported_sets 这个 lambda 里 —— 它回答"什么形状的子图会被整体交给 OpenVINO"。',
    src=OW, parts=P4, duration=24000,
    mark_src=[1449, 1452, 1453, 1454, 1461, 1462, 1470, 1471, 1474, 1475, 1476, 1483, 1490],
    notes_src=N4,
    visual=VIS4.replace('/*__LN4__*/null', mklines(P4, N4))
)

# ==================================================================== 第 5 幕

VIS5 = '''
const LN = /*__LN5__*/null;
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
'''

P5 = [(1495, 1552)]
N5 = {1495: '按 op 分类走三条不同的分支 —— 因为 unary/glu 的子枚举不占 ggml_op 的编号',
      1497: 'UNARY：查 supported_unary_ops',
      1501: '例外一：F32 的 EXP 没有翻译器能覆盖（类型级例外）',
      1507: 'GLU：查 supported_glu_ops',
      1514: '例外二：src1 为空时 ne[0] 必须是偶数 —— 源码注释说是 ov gpu 的 bug',
      1521: 'default：普通算子查 supported_ops',
      1532: '★ 第二道门槛：输出张量类型必须在白名单里',
      1540: '每个非空输入的类型也必须在白名单里 —— 判据遍历 src[]，不只是 src[0]',
      1543: '唯一放行的 3D 量化例外：MUL_MAT_ID 的专家权重（MoE）',
      1545: '★ 第三道门槛：量化张量的 ne[2] 必须为 1，否则整个 op 落到别的后端',
      1550: '最后是设备 / 形状级的例外表（is_op_supported_case，1129 行起）：ROPE 的 ne[3]、' +
            'TRANSPOSE 的 BF16、GPU 上的 REPEAT 等'}

L.scene(
    kicker='L7-03 · 支持判据（二）',
    title='第二、三道门槛：<span class="hl-b">类型</span> × <span class="hl-b">形状</span> × 设备例外',
    sub='op 有翻译器还不够：算子与全部输入的类型必须在 13 种白名单里，量化张量还不许是 3D。',
    caption='"什么形状的子图交给 OpenVINO"的完整答案 = 每个节点都过这三道门槛的最长连续区间。',
    src=OW, parts=P5, duration=26000,
    mark_src=[1495, 1496, 1497, 1507, 1521, 1522, 1523, 1532, 1533, 1540, 1541,
              1543, 1544, 1545, 1546, 1550, 1551],
    notes_src=N5,
    visual=VIS5.replace('/*__LN5__*/null', mklines(P5, N5))
)

# ==================================================================== 第 6 幕

VIS6 = '''
const LN = /*__LN6__*/null;
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
'''

P6 = [(74, 104)]
N6 = {74: '默认值是 "CPU" —— 不设环境变量时，这个后端跑在 OpenVINO 的 CPU 插件上',
      75: '可用性来自 OpenVINO Core 自己枚举，不是 ggml 猜的',
      76: '不可用就只警告一声、退回 CPU —— 后端不会因此拒绝初始化',
      80: 'is_npu 是热路径上的一个 bool（utils.cpp:1470 用它选 static/dynamic 执行路径）',
      83: '★ NPU 分支：整份 compile_config 只在设备名是 NPU 时才填',
      84: 'NPUW = NPU 插件的权重银行 / 函数调用模式；这一组 key 是 NPU 能跑起来的全部前提',
      104: '非 NPU 分支只有缓存配置 —— GPU 走的是另一条路（建 OpenCL remote context，115-148 行）'}

L.scene(
    kicker='L7-03 · 设备选择',
    title='一个后端，三种设备：<span class="hl-c">NPU</span> / <span class="hl-c">GPU</span> / <span class="hl-c">CPU</span>',
    sub='设备名从 GGML_OPENVINO_DEVICE 读，默认 CPU；不可用就回退 CPU。设备一旦定下，编译配置与判据分支都跟着变。',
    caption='设备名不只是"选硬件"：它直接进入 supports_op 的例外判断（如 GPU 上的 BF16 TRANSPOSE / REPEAT、NPU 上的 BF16 输入）。',
    src=EXTRA, parts=P6, duration=22000,
    mark_src=[74, 75, 76, 77, 78, 80, 83, 84, 85, 86, 104],
    notes_src=N6,
    visual=VIS6.replace('/*__LN6__*/null', mklines(P6, N6))
)

# ==================================================================== 第 7 幕

VIS7 = '''
const LN = /*__LN7__*/null;
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
'''

P7 = [(640, 660)]
N7 = {640: '注释直说：这个检测是 O(n^2)，在 Llama-1B 的 decode 图上要 ~20 ms',
      645: 'graph_key 从 cgraph 构造；注释说它只要几百微秒，比逐节点比较便宜得多',
      652: '只有在缓存里没见过时才真的去算 is_model_splitted',
      654: 'naive = 节点数 < 20 的小图（is_naive 在 1566 行）：直接一股脑翻译，不做权重提取与缓存',
      655: '小图 + 没被切过 -> 走 naive_compute，一次性翻译整图',
      656: '被切过的小图不享受这条捷径：碎片图之间可能存在后端边界，缓存与权重假设都不成立'}

L.scene(
    kicker='L7-03 · 回退策略',
    title='不支持的 op <span class="hl-a">不需要回退代码</span>：它根本不会被切进来',
    sub='运行时真正要判断的，是"这次拿到的是整图、还是被切碎的一段" —— 后者不能走缓存与 naive 捷径。',
    caption='回顾 L4-02 幕 6：切分 = 沿图找连续的同后端段。supports_op 就是那把刀。',
    src=UTILS, parts=P7, duration=20000,
    mark_src=[640, 645, 646, 649, 652, 654, 655, 656],
    notes_src=N7,
    visual=VIS7.replace('/*__LN7__*/null', mklines(P7, N7))
)

# ==================================================================== 第 8 幕

VIS8 = '''
const LN = /*__LN8__*/null;
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
'''

P8 = [(57, 73)]
N8 = {57: 'square = input * input —— 用 Multiply 自乘表达',
      59: 'ReduceMean over {-1}，keep_dims=true：这就是 ggml 的"最后维求均值"',
      62: 'eps 从 op_params 里按 float 读出来 —— 与 L1-02 幕 6 说的"16 个 int32 的 schema 只是注释"呼应',
      65: 'sqrt(mean + eps)：Add + Sqrt',
      68: '1 / rms：Divide',
      71: '最后乘回输入 —— 到这里已经 6 个 ov 节点了，而 ggml 侧只有 1 个'}

L.scene(
    kicker='L7-03 · 翻译粒度',
    title='一个 ggml op 进去，<span class="hl-e">一串</span> ov::op 出来',
    sub='RMS_NORM 是最典型的例子：6 个 ov 节点、零个 kernel。这是"翻译"这个视角最直接的证据。',
    caption='对照组：FLASH_ATTN_EXT 反过来 —— 一个 ggml op 落进一个融合算子 v13::ScaledDotProductAttention（flash_attn_ext.cpp:228/232）。',
    src=RMS, parts=P8, duration=20000,
    mark_src=[57, 59, 60, 62, 63, 65, 66, 68, 69, 71],
    notes_src=N8,
    visual=VIS8.replace('/*__LN8__*/null', mklines(P8, N8))
)

# ==================================================================== 第 9 幕

VIS9 = '''
const LN = /*__LN9__*/null;
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
'''

P9 = [(76, 83)]
N9 = {80: 'ROLL 是这张表的最后一项',
      81: '★ 注释也是能力边界：把一行注释掉 = 把这个算子永久踢出 OpenVINO 子图',
      82: 'SOLVE_TRI 的翻译器 translate_solve_tri 仍然存在（solve_tri.cpp:33），' +
          '但没被登记进表里 —— 于是 supports_op 会拒绝这个 op，翻译器成了死代码'}

L.scene(
    kicker='L7-03 · 收束',
    title='一张表说清 OpenVINO 后端',
    sub='支持判据、设备、回退、粒度 —— 四个问题，四个答案，都在同一个设计决定下。',
    caption='下一课 L7-04：ExecuTorch 用"委托"做同一件事 —— 但委托的边界是另一套抽象。',
    src=OPT, parts=P9, duration=20000,
    mark_src=[76, 77, 78, 79, 80, 81, 82],
    notes_src=N9,
    visual=VIS9.replace('/*__LN9__*/null', mklines(P9, N9))
)

# ==================================================================== source.md

L.section(
    '一、82 个文件：五族分工',
    '`L7-03` 的覆盖清单由 `plan_matrix.py` 指派，实测 **82 个文件**：'
    '公共头 1 个 + `ggml/src/ggml-openvino/` 下 81 个。按目录分成五族：\n\n'
    '| 族 | 个数 | 目录 / 代表文件 | 职责 |\n|---|---|---|---|\n'
    '| 顶层胶水 | 12 | `ggml-openvino.cpp` `ggml-decoder.*` `ggml-openvino-extra.*` `ggml-quants.*` `model-cache.*` `utils.*` + `include/ggml-openvino.h` | '
    '虚表实现、cgraph→decoder、设备配置、权重量化 / 重打包、编译缓存 |\n'
    '| frontend 骨架 | 12 | `openvino/op_table.*` `translate_session.*` `input_model.*` `node_context.h` `decoder.h` `frontend.*` `utils.*` | '
    '翻译器的注册表与调度：节点 → 翻译函数 → `ov::Model` |\n'
    '| op 翻译器 | 45 | `openvino/op/` 下 42 个 `.cpp` + 3 个 `.hpp` | '
    '每个（族）ggml 算子一个文件，把 `NodeContext` 变成 `ov::OutputVector` |\n'
    '| OV pass | 12 | `openvino/pass/` | 翻译完成后的图改写：`FuseToConv` `FuseToSdpa` `SqueezeMatmul` `KVStateSeqAxis` 等 |\n'
    '| rt_info | 1 | `openvino/rt_info/weightless_caching_attributes.hpp` | 给大 `Constant` 打标记，避免 NPUW 重复拷贝权重 |\n\n'
    '**读法**：想知道"某个 ggml 算子在这个后端上会变成什么"，先查 `op_table.cpp` 有没有它，'
    '再打开 `openvino/op/` 下对应的那个文件。')
L.cover(*MANIFEST_SORTED)

L.section(
    '二、★ 翻译表：后端能力的唯一真相来源',
    '`get_supported_ops()` 返回一张 `std::unordered_map<std::string, CreatorFunction>`。'
    'key 是 **ggml 的 op 宏名**（`GGML_OP_*` / `GGML_UNARY_OP_*` / `GGML_GLU_OP_*`），'
    'value 是翻译函数。表里共 **55 项**（第 26-80 行）。\n\n'
    '两个值得注意的细节（都能从下面的逐字引用读出来）：\n\n'
    '- `GGML_OP_FILL` 出现了**两次**（第 32 行与第 75 行）—— 两处都指向 `op::translate_fill`，'
    '所以无论哪一条生效，结果都相同。\n'
    '- 第 81-82 行是**被注释掉**的 `GGML_OP_SOLVE_TRI`：`solve_tri.cpp:33` 里的 `translate_solve_tri` '
    '仍然存在，但没有登记进表 —— 于是它永远不会被调用（`supports_op` 会说"没有翻译器"）。'
    '这说明**能力边界由这张表决定，而不是由存在哪些文件决定**。',
    src=OPT, parts=[(23, 84)], lang='cpp')

L.section(
    '三、★ 支持判据：三道门槛（验收点）',
    '`ggml_backend_openvino_device_supports_op_impl()` 是**验收点答案**所在。'
    '它一次只看**一个** op，返回 `(bool, reason)`：\n\n'
    '```text\n'
    '门槛一：op 名字在 build_supported_sets() 派生出的集合里吗\n'
    '        （集合来自 op_table 的 key；UNARY/GLU 走各自的子枚举集合）\n'
    '门槛二：op->type 与全部非空 src[i]->type 都在 13 种白名单里吗\n'
    '门槛三：量化张量的 ne[2] == 1 吗（MUL_MAT_ID 的专家权重除外）\n'
    '过完三道，再过 is_op_supported_case() 的设备 / 形状例外表\n'
    '```\n\n'
    '**所以"什么形状的子图会被整体交给 OpenVINO"的答案是**：'
    '在图里按拓扑序取**最长的、每个节点都过三道门槛的连续区间**。'
    '调度器（L4-02）在每个"过不去"的节点前面断一刀，跨段的数据由调度器插 copy。'
    'OpenVINO 后端自己**不写任何软件回退**。',
    src=OW, parts=[(1449, 1552)], lang='cpp')

L.section(
    '四、设备选择：NPU / GPU / CPU',
    '设备名只有一个来源：环境变量 `GGML_OPENVINO_DEVICE`，默认 `CPU`；'
    '如果 `ov::Core::get_available_devices()` 里没有它，就警告一句并回退到 `CPU`。\n\n'
    '设备名定下之后有三处后果：\n\n'
    '1. **编译配置**：只有 `NPU` 会填 `NPU_COMPILER_DYNAMIC_QUANTIZATION` 与一整组 `NPUW_*` key；\n'
    '2. **内存通道**：只有 `GPU` 会建 OpenCL context/queue 并把 remote context 交给 OV；\n'
    '3. **支持判据**：`supports_op` 的例外表里大量出现 `ggml_openvino_get_device_name() == "GPU"` '
    '`== "NPU"` 的比较 —— 同一个 op 在不同设备上的答案可以不同。\n\n'
    '另外 `is_npu` 直接决定执行路径：`utils.cpp:1470` 用 `ggml_openvino_is_npu()` 在 '
    '**static（NPU 形状固定）** 与 **dynamic** 两条路径之间二选一；'
    '`ggml-openvino.cpp:837` 里 stateful 执行也被显式禁用在 NPU 上。',
    src=EXTRA, parts=[(74, 104)], lang='cpp')

L.section(
    '五、回退策略：缺席，而不是分支',
    '不支持的 op 由**切分阶段**解决：`supports_op` 说不，调度器就不把它切进来。'
    '所以 OpenVINO 后端里找不到"这个算子我自己实现一个慢版本"的代码。\n\n'
    '运行时真正要判断的是另一件事：**这次拿到的是整图，还是被切碎的一段**。'
    '`is_model_splitted()` 用"节点的 `use_count` 是否等于把它当输入的节点数"来反推；'
    '`is_naive()` 用节点数是否 `< 20` 来判断是否小图；只有"小图且没被切过"才走 '
    '`naive_compute()` 这条一次性翻译的捷径。这两个检测都由 '
    '`GGML_OPENVINO_ENABLE_FALLBACK`（`utils.cpp:1492` 起）与缓存命中情况控制。',
    src=UTILS, parts=[(640, 660)], lang='cpp')

L.section(
    '六、翻译粒度：一个 ggml op → 0..N 个 ov::op',
    '`translate_*` 的返回类型是 `ov::OutputVector`，长度不受限。'
    '`translate_rms_norm()` 的收尾这一段就是 6 个 ov 节点（Multiply / ReduceMean / Add / Sqrt / Divide / Multiply）：',
    src=RMS, parts=[(57, 73)], lang='cpp')

L.section(
    '七、另一头：一个 ggml op → 一个融合 ov::op',
    '`FLASH_ATTN_EXT` 的翻译器把整块注意力交给 OpenVINO 的 '
    '`ov::op::v13::ScaledDotProductAttention`（有 mask 与无 mask 两个分支）。'
    '这是"粒度由 ov 算子集决定"的证明：ggml 侧一个节点，ov 侧一个算子。',
    src=FA, parts=[(226, 234)], lang='cpp')

L.section(
    '八、三张虚表：和别的后端签的是同一份契约',
    'OpenVINO 后端填的表与 L3-01 讲的三层契约完全一致，'
    '只是 `ggml_backend_i` 里除了 `get_name` / `free` / `graph_compute` 全是 `NULL`，'
    '而 `ggml_backend_device_i` 里 `supports_op` 指向本课的主角。',
    src=OW, parts=[(1574, 1590)], lang='cpp')

L.footnote_add(
    '本课覆盖清单里的 **82 个文件全部计入覆盖率**，其中 **8 个核心文件被逐字引用**：'
    '`ggml/include/ggml-openvino.h`、`ggml-openvino.cpp`、`ggml-openvino-extra.cpp`、`utils.cpp`、'
    '`openvino/op_table.cpp`、`openvino/translate_session.cpp`、'
    '`openvino/op/rms_norm.cpp`、`openvino/op/flash_attn_ext.cpp`。'
    '其余 74 个文件按族在第一节的表格里说明分工，不逐个引用。')
L.footnote_add(
    '场景 3 的映射表里每一行的"翻译成什么 ov::op"都来自实测，不是凭印象写的：'
    '走一行模板特化（`translate_1to1_match_*<T>`）的，类名直接写在 `op_table.cpp` 里；'
    '专属翻译器的类名来自对该文件的 `grep -o "ov::op::v[0-9]*::[A-Za-z_0-9]*"`。'
    '表里只列了最常见的 16 行，完整 55 项以 `op_table.cpp` 第 23-84 行的逐字引用为准。')
L.footnote_add(
    '本课**不展开**权重重打包（`ggml-quants.cpp`）与编译缓存（`model-cache.cpp`）的算法细节 —— '
    '这两个文件已进覆盖声明，但它们属于"数据面"而非"图翻译"，留给后续修订。')

L.prereqs('`L7-02`（Hexagon 后端）；建议先看 L3-01（虚表契约）与 L4-02（切分）')

L.goal(
    '说出 OpenVINO 后端在 `ggml_backend_device_i.supports_op` 里的**三道判据**（对应验收点）；',
    '据此判断**什么形状的子图会被整体交给 OpenVINO** —— 最长的、每节点都通过判据的连续区间；',
    '解释为什么这个后端**没有软件回退代码**，以及"回退"实际发生在哪一层；',
    '说出 `GGML_OPENVINO_DEVICE` 的三个取值各自改变了什么（编译配置 / 内存通道 / 判据例外）；',
    '举出"一个 ggml op → 多个 ov::op"和"一个 ggml op → 一个融合 ov::op"各一个例子。')

L.conclusion(
    '★ 支持判据 = 翻译器的存在性',
    '`supports_op` 的第一道门槛就是\n\n'
    '```text\n'
    'table.count("GGML_OP_" + ggml_op_name(op))   // op_table.cpp 的 key\n'
    '```\n\n'
    '它一次只回答**一个 op**，返回 `(bool, reason)`。三道门槛合起来是：\n\n'
    '| # | 判据 | 代码位置 |\n|---|---|---|\n'
    '| 一 | op（/ unary / glu）名字在 op_table 派生的集合里 | `ggml-openvino.cpp:1470-1488` |\n'
    '| 二 | `op->type` 与全部 `src[i]->type` ∈ 13 种白名单 | `ggml-openvino.cpp:1532-1542` |\n'
    '| 三 | 量化张量的 `ne[2] == 1`（MUL_MAT_ID 专家权重除外） | `ggml-openvino.cpp:1543-1547` |\n'
    '| + | 设备 / 形状例外表 | `is_op_supported_case`，`1129` 行起 |')

L.conclusion(
    '★ 一次交出一个子图，而不是一个算子',
    '`supports_op` 是**逐节点**问的，`graph_compute` 收到的却是**一整段子图**。'
    '两者由 L4-02 的调度器连接：它在每个"说不"的节点前断一刀，'
    '于是切给 OpenVINO 的永远是**连续的同后端段**。\n\n'
    '所以验收点的完整答案是：\n\n'
    '> 在图里按拓扑序取**最长的、每个节点都通过三道判据（且满足设备例外）的连续区间** —— '
    '这一段会被整体翻译成一个 `ov::Model`，一次 `compile_model`，一次推理。\n\n'
    '任何一格不过，就在那里断开；跨段的数据由调度器插 copy，OpenVINO 后端不写回退代码。')

L.conclusion(
    '这是一台"图到图"的编译器前端，不是一组 kernel',
    '| 维度 | 写 kernel 的后端（L5/L6） | OpenVINO 后端（L7-03） |\n|---|---|---|\n'
    '| 主要工程量 | 每个算子的 SIMD / 线程实现 | 45 个 op 翻译器 |\n'
    '| "支持"的含义 | 我实现了这个 kernel | 我有这个 op 的 `ov::op` 映射 |\n'
    '| 一个 ggml op | 一次 kernel 调用 | 0..N 个 `ov::op` 节点 |\n'
    '| 不支持的 op | 常要写标量回退 | 不写任何代码，交给切分 |\n'
    '| 性能来源 | 手写向量化 | OpenVINO 插件的图优化与设备编译 |\n\n'
    '与 L7-01（CANN 映射到 ACL 算子）同属"映射到厂商算子"这一路线；'
    'L7-02（Hexagon）则是把整图交给远端 DSP。三者的差别不在契约（都由 L3-01 定），'
    '而在**这段子图被翻译成了什么**。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
    print('covered files:', len(L.files))
