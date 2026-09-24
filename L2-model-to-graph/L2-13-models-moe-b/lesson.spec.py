#!/usr/bin/env python3
"""L2-13 · 模型家族（四）：稀疏专家 MoE（下）—— 课件 spec。

运行：python3 L2-model-to-graph/L2-13-models-moe-b/lesson.spec.py

本课是 L2-12（MoE 上）的下篇。L2-12 覆盖静态扫描分出的前 12 个 MoE 文件，
讲 `build_moe_ffn` 的**基础序列**；本课覆盖后 12 个（清单由
`python3 tools/plan_matrix.py --files` 里 `L2-13` 这一行给出，共 24 个文件），
讲同一骨架的**参数**如何被不同模型选成不同的路由策略：
共享专家、分组路由、以及 MoE 与注意力变体的组合。

`src/llama-graph.cpp` / `src/llama-graph.h`（`build_moe_ffn` 的定义与声明）、
`src/models/models.h`（`using graph = ...` 别名）、`src/llama-hparams.h`、
`src/llama-model.cpp` 是按计划分配给别课的公共文件，本课作为**对照**逐字引用，
见文末「说明」。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

# ---------------------------------------------------------------- 本课计划内的 24 个文件

COVERED = [
    'src/models/llada-moe.cpp',
    'src/models/llama.cpp',
    'src/models/llama4.cpp',
    'src/models/maple.cpp',
    'src/models/mellum.cpp',
    'src/models/mimo2.cpp',
    'src/models/minicpm.cpp',
    'src/models/minimax-01.cpp',
    'src/models/minimax-m2.cpp',
    'src/models/minimax-m3.cpp',
    'src/models/mistral3.cpp',
    'src/models/nemotron-h-moe.cpp',
    'src/models/nomic-bert-moe.cpp',
    'src/models/olmoe.cpp',
    'src/models/openai-moe.cpp',
    'src/models/phi3.cpp',
    'src/models/phimoe.cpp',
    'src/models/qwen2moe.cpp',
    'src/models/qwen3moe.cpp',
    'src/models/qwen3vlmoe.cpp',
    'src/models/refact.cpp',
    'src/models/rnd1.cpp',
    'src/models/smallthinker.cpp',
    'src/models/step35.cpp',
]

# ---------------------------------------------------------------- 计划外、仅作对照的公共文件

GRAPH_CPP = 'src/llama-graph.cpp'    # build_moe_ffn 的实现（L2-06 / L8-01 的计划文件）
GRAPH_H   = 'src/llama-graph.h'      # build_moe_ffn 的两条声明（L2-06 的计划文件）
MODELS_H  = 'src/models/models.h'    # using graph = ... 别名（L2-10 / L2-15 的计划文件）
HPARAMS_H = 'src/llama-hparams.h'    # 专家相关超参（L2-02 的计划文件）
MODEL_CPP = 'src/llama-model.cpp'    # GGUF 元数据 -> hparams（L2-05 的计划文件）


def ridx(parts, notes_src, lines):
    """上游行号 -> codeBlock 渲染下标（注解行不占上游行号但占渲染位）。"""
    base = parts[0][0]
    ns = sorted((notes_src or {}).keys())
    return [ln - base + sum(1 for x in ns if x < ln) for ln in lines]


L = Lesson(
    id='L2-13',
    layer='L2 · 从模型到图',
    title='模型家族（四）：稀疏专家 MoE（下）',
    codecap='src/models/*（24 个，逐字引用）+ src/llama-graph.cpp 对照',
    nav={'prev': {'href': '../L2-12-models-moe-a/index.html',
                  'label': 'L2-12 模型家族（三）MoE 上'},
         'next': {'href': '../L2-14-models-dense-a/index.html',
                  'label': 'L2-14 模型家族（五）稠密上'}},
)

L.note('**一句话**：MoE 在 llama.cpp 里只有**一条骨架** —— `llm_graph_context::build_moe_ffn()`。'
       '24 个模型文件做的事不是"实现 MoE"，而是**给这条骨架挑参数**：'
       '选几个专家、概率用什么函数、权重归不归一化、路由 logits 是自己算还是外面算。')
L.note('L2-12 讲的是这条骨架的**基础序列**（logits -> top-k -> `mul_mat_id` -> 加权求和）；'
       '本课是它的下篇，只问一件事：**同样的骨架，这 24 个文件各自选了哪些开关**，'
       '以及哪些开关根本不在参数表里（共享专家、分组路由）。')
L.note('本课覆盖的 24 个文件来自 `python3 tools/plan_matrix.py --files` 中 `L2-13` 的分配'
       '（静态扫描把 `src/models/*.cpp` 里命中 `build_moe_ffn|ffn_gate_exps|ffn_gate_inp` '
       '的文件排序后对半分，本课是后半）。**分组依据是命中图原语，不是对文件内容的断言** —— '
       '实测结果：其中 4 个文件全篇没有 `build_moe_ffn`，本课会逐个说明。')

L.cover(*COVERED)
L.cover(GRAPH_CPP, GRAPH_H, MODELS_H, HPARAMS_H, MODEL_CPP)

# ==================================================================== 第 1 幕 · 全局

L.scene(
    kicker='L2-13 · 全局',
    title='24 个 MoE 文件，共用 <span class="hl-a">一条骨架</span>',
    sub='L2-12 讲这条骨架的基础序列；本课讲它的参数如何被 24 个模型文件选成不同的路由策略。',
    caption='回顾 L2-06：每个模型图都是 llm_graph_context 的子类，MoE 只是其中一个 build_* 调用。',
    src=GRAPH_CPP, parts=[(1993, 2017)], duration=15000,
    mark_src=[1993, 2003, 2004, 2005, 2009, 2011, 2017],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row center" id="faces" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '共享专家 shexp', b: '9 / 24 个文件声明了<br>ffn_*_shexp 张量', m: 'ffn_gate_shexp' },
  { c: 'c', t: '分组路由', b: '24 个文件里 0 处提到<br>n_expert_groups / n_group_used', m: 'hparams.n_expert_groups' },
  { c: 'e', t: '与注意力变体的组合', b: '同一个 MoE 分支，喂进来的 cur<br>来自 SWA / MSA / MTP 等不同的图', m: 'build_attn(...)' }
];
const host = wrap.querySelector('#faces');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  'L2-12 讲的是这条骨架的<b>基础序列</b>：logits -> top-k -> mul_mat_id -> 加权求和。',
  '本课只问一件事：<span class="k">同样的骨架，24 个模型文件各自选了哪些开关</span>。',
  '<span class="v">共享专家</span>：9 个文件声明了 shexp 张量，接线方式有三种。',
  '<span class="v">分组路由</span>：开关不在参数表里，在 <span class="v">hparams</span> 里。',
  '<span class="v">注意力变体</span>：MoE 分支的上游各不相同，但 build_moe_ffn 不关心。',
  '对照 L2-06：所有模型图都是 <span class="v">llm_graph_context</span> 的子类，MoE 只是其中一个 build_* 调用。'
];
defs.forEach((_, i) => tl.at(600 + i * 2600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(12400, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[0] + '<br>' + texts[5];
});
'''
)

# ==================================================================== 第 2 幕 · ★ 参数

L.scene(
    kicker='L2-13 · 核心',
    title='★ 控制"专家选择"的是这 <span class="hl-a">8 个参数</span>',
    sub='其余参数只改权重布局与专家内部激活，不改变"选哪些专家"。',
    caption='验收点：能说出 build_moe_ffn 的参数里哪些控制专家选择。',
    src=GRAPH_CPP, parts=[(2003, 2017)], duration=24000,
    mark_src=[2003, 2004, 2005, 2007, 2008, 2009, 2011, 2017],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div>
  <div class="flow" id="notroute" style="gap:6px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['参数', '函数体里生效于', '它决定什么'],
  [['n_expert', 'llama-graph.cpp:2084 / 2120', '候选专家总数：分组大小、probs 的维度'],
   ['n_expert_used', 'llama-graph.cpp:2109', 'top-k 的 k —— 直接决定选几个专家'],
   ['gating_op', 'llama-graph.cpp:2040-2059', 'logits 怎么变概率：softmax / sigmoid / softmax_weight / sqrt_softplus'],
   ['exp_probs_b', 'llama-graph.cpp:2066', '选择偏置：只加到 selection_probs，不加到权重'],
   ['norm_w', 'llama-graph.cpp:2134-2148', 'top-k 权重是否除以它们的和'],
   ['w_scale', 'llama-graph.cpp:2149-2151', '选中之后的路由权重整体缩放'],
   ['probs_in', 'llama-graph.cpp:2024-2032', '非空则跳过 gate_inp，用外部算好的 logits'],
   ['selected_experts_in', 'llama-graph.cpp:2107-2111', '非空则整段跳过 top-k']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const nr = wrap.querySelector('#notroute');
nr.appendChild(U.chip('不控制选择：', 'g'));
['weight 类：gate_inp', 'up_exps', 'gate_exps', 'down_exps'].forEach(x => nr.appendChild(U.chip(x)));
nr.appendChild(U.chip('type_op -> :2221', 'd'));
nr.appendChild(U.chip('il -> cb 命名', 'd'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '8 个参数里，前两个是<b>基数与 k</b>，中间四个是<b>概率与权重</b>，最后两个是<b>外部接管通道</b>。',
  '<span class="v">n_expert</span> 与 <span class="v">n_expert_used</span>：一个定候选集合大小，一个定 top-k 的 k。',
  '<span class="v">gating_op</span>：四个分支在源码里写死（2040-2059），传错值会 GGML_ABORT。',
  '<span class="v">exp_probs_b</span> 只影响"选谁"，<span class="k">不影响选中之后的权重</span>（2063 的注释写明）。',
  '<span class="v">norm_w</span>：把 top-k 权重除以它们的和，除以零被 clamp 成 F16 最小值（2141）。',
  '<span class="v">w_scale</span>：<span class="k">只在它不等于 0.0f 且不等于 1.0f 时才生效</span>（2149）。',
  '<span class="v">probs_in</span> / <span class="v">selected_experts_in</span>：两条外部接管通道，可以分别绕开 gate_inp 与 top-k。',
  '结论：<span class="k">MoE 的可调旋钮不在"算子"，在"路由策略"</span> —— 这 8 个参数就是全部。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 8)];
}));
tl.at(22800, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[8];
});
'''
)

# ==================================================================== 第 3 幕 · ★ 分组路由

_GROUP_PARTS = [(2081, 2104)]
_GROUP_NOTES = {
    2083: 'hparams 是成员引用，不是参数 —— 模型文件无法在这一行做选择',
    2096: '组级 top-k：n_group_used 决定允许几个组',
    2101: '没被选中的组整片填成 -INFINITY，它们在下一步的 top-k 里必然落选',
}
_GROUP_M1 = ridx(_GROUP_PARTS, _GROUP_NOTES, [2081, 2083])
_GROUP_M2 = ridx(_GROUP_PARTS, _GROUP_NOTES, [2084, 2087, 2096, 2101])

L.scene(
    kicker='L2-13 · 核心',
    title='★ 分组路由的开关是 <span class="hl-a">hparams</span>，不是调用参数',
    sub='24 个文件里 n_expert_groups / n_group_used 出现 0 次；它从 GGUF 读进 hparams，在函数体里生效。',
    caption='回顾 L2-12：deepseek2 走的就是这一段，模型文件同样一个字都没写。',
    src=GRAPH_CPP, parts=_GROUP_PARTS, duration=14000,
    mark_src=[2081, 2083, 2084, 2087, 2096, 2101],
    notes_src=_GROUP_NOTES,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">GGUF 元数据</span><span class="arrow">-></span>
    <span class="chip b">hparams.n_expert_groups</span><span class="arrow">-></span>
    <span class="chip c">大于 1 ?</span><span class="arrow">-></span>
    <span class="chip e">mask 掉没选中的组</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['hparams 字段', '读进 hparams 的地方', '在 build_moe_ffn 里怎么用'],
  [['n_expert_groups', 'llama-model.cpp:1266', 'llama-graph.cpp:2083 判断 / 2087 分组'],
   ['n_group_used', 'llama-model.cpp:1267', 'llama-graph.cpp:2096 组级 top-k'],
   ['n_group_experts', '本课 24 文件未出现', 'llama-graph.cpp:2117 GROVEMOE 专用重映射']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const texts = [
  '分组路由（DeepSeek V3 的 device-limited routing）<span class="k">不是一个参数</span>。',
  '它从 GGUF 读进 <span class="v">hparams.n_expert_groups</span>，在函数体里用 <span class="v">if (hparams.n_expert_groups &gt; 1)</span> 判断。',
  '先按组内最高两个专家分求和（2089-2093），再对组做 top-<span class="v">n_group_used</span>。',
  '最后把没选中的组填成 <span class="v">-INFINITY</span>，后面那次 <span class="v">argsort_top_k</span> 自然选不到它们。',
  '本课 24 个文件里 <span class="v">n_expert_groups</span> / <span class="v">n_group_used</span> 出现 <span class="k">0 次</span>：<br>它们是模型（GGUF）的属性，不是代码里的选择。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, __M1__); });
tl.at(3200, () => { msg.innerHTML = texts[1]; });
tl.at(6000, () => { msg.innerHTML = texts[2]; U.markLines(document, __M2__); });
tl.at(9200, () => { msg.innerHTML = texts[3]; });
tl.at(11800, () => { msg.innerHTML = texts[4]; U.markLines(document, []); });
'''.replace('__M1__', repr(_GROUP_M1)).replace('__M2__', repr(_GROUP_M2))
)

# ==================================================================== 第 4 幕 · ★ 共享专家

L.scene(
    kicker='L2-13 · 核心',
    title='★ 共享专家不在参数表里 —— 它是 <span class="hl-a">调用点外的一条支路</span>',
    sub='qwen2moe 给它配了独立的门控；llama4 / step35 / minimax-m3 直接相加；llama / mistral3 / refact 声明了但没接。',
    caption='24 个文件里只有 qwen2moe 声明 LLM_TENSOR_FFN_GATE_INP_SHEXP。',
    src='src/models/qwen2moe.cpp', parts=[(145, 168)], duration=22000,
    mark_src=[145, 147, 151, 154, 162, 165, 168],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="faces" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '自带门控', b: 'qwen2moe:147-165<br>ffn_gate_inp_shexp 出一个标量门，<br>乘在共享专家输出上再相加', m: 'ggml_mul(cur_ffn, cur_gate)' },
  { c: 'b', t: '直接相加', b: 'llama4:232-240 · step35:327-335<br>minimax-m3:575-581<br>共享专家是一个普通 build_ffn', m: 'ggml_add(moe_out, sh_out)' },
  { c: 'd', t: '只声明不接线', b: 'llama:85-88 · minicpm:77-81<br>mistral3:80-83 · refact:66-70<br>张量建了，图里没有对应支路', m: 'ffn_gate_shexp = create_tensor(...)' }
];
const host = wrap.querySelector('#faces');
const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="k">9 / 24</span> 个文件声明了 ffn_*_shexp 张量，但接法有三种。',
  'qwen2moe：<span class="v">ffn_gate_inp_shexp</span> 是一个 <span class="v">[n_embd]</span> 的向量（54 行），<br>算完 sigmoid 门后用 <span class="v">ggml_mul</span> 乘到共享专家输出上（162）。',
  'llama4 / step35 / minimax-m3：共享专家就是一次 <span class="v">build_ffn</span>，<br>结果与 <span class="v">moe_out</span> 直接 <span class="v">ggml_add</span>，没有门。',
  'llama / minicpm / mistral3 / refact：<span class="v">create_tensor</span> 建了张量，<br>但在本文件的图里找不到使用它的支路。',
  '关键：这三种接法<span class="k">一种都没有经过 build_moe_ffn 的参数</span> —— <br>共享专家是图上的<b>第二条支路</b>，不是 MoE 的一个开关。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3400 + i * 4600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(17800, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[4];
});
'''
)

# ==================================================================== 第 5 幕 · 调用点对照

L.scene(
    kicker='L2-13 · 24 个调用点',
    title='同一个函数，<span class="hl-c">21 个调用点</span>各选不同的开关',
    sub='20 个文件共 21 处调用；下表每一格都是对这 24 个文件静态统计出来的。',
    caption='L2-12 覆盖的另外 12 个 MoE 文件走的是同一张参数表。',
    src='src/models/step35.cpp', parts=[(512, 522)], duration=20000,
    mark_src=[512, 517, 519, 520, 521],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['参数', '21 个调用点的取值分布', '代表调用点'],
  [['n_expert / n_expert_used', '21 处写法完全一样', 'llama-graph.cpp:1471-1472 提供'],
   ['gating_op', 'SOFTMAX 12 / 读 hparams 5 / SIGMOID 3 / SOFTMAX_WEIGHT 1', 'llama4:228 · openai-moe:141'],
   ['exp_probs_b', '传张量 7 / nullptr 14', 'mimo2:213 · step35:517'],
   ['norm_w', 'true 12 / false 5 / 读 hparams 4', 'llama:209 · step35:519'],
   ['w_scale', 'hparams.expert_weights_scale 20 / 字面量 1.0f 1', 'maple:129'],
   ['probs_in', '只有 2 处传（其余用默认 nullptr）', 'smallthinker:159 · nemotron-h-moe:127']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '统计口径：对 24 个文件里所有 <span class="v">build_moe_ffn(</span> 调用点逐个取参。<br>20 个文件共 <span class="k">21 处</span>（step35 主干与 MTP 各一处）。',
  '<span class="v">n_expert</span> / <span class="v">n_expert_used</span> 21 处一字不差 —— 它们是 <span class="v">llm_graph_context</span> 的成员，<br>来自 hparams；<span class="k">模型文件在这两个参数上没有自由度</span>。',
  '<span class="v">gating_op</span> 是分歧最大的一个：12 处写死 SOFTMAX，5 处转成 <span class="v">hparams.expert_gating_func</span>。',
  '<span class="v">norm_w</span> 是三态：写死 true、写死 false、或者读 <span class="v">hparams.expert_weights_norm</span>。',
  '<span class="v">probs_in</span> 只有 smallthinker 与 nemotron-h-moe 用到 —— 这两处把 top-k 的输入搬到了函数外。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3000 + i * 3000, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(18300, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[1]; });
'''
)

# ==================================================================== 第 6 幕 · probs_in

_MTP_PARTS = [(99, 130)]
_MTP_A = ridx(_MTP_PARTS, None, [100])
_MTP_B = ridx(_MTP_PARTS, None, [100, 103])
_MTP_C = ridx(_MTP_PARTS, None, [115, 116])
_MTP_D = ridx(_MTP_PARTS, None, [119, 121, 127, 128])

L.scene(
    kicker='L2-13 · 两条接管通道',
    title='<span class="hl-e">probs_in</span>：路由已经在外面算好了',
    sub='传了 probs_in，函数体就不再碰 gate_inp；smallthinker 连 gate_inp 都直接传 nullptr。',
    caption='签名最后一位 selected_experts_in 能整段跳过 top-k；本课 24 个文件都没用它。',
    src='src/models/nemotron-h-moe.cpp', parts=[(99, 130)], duration=20000,
    mark_src=[100, 103, 115, 116, 119, 121, 123, 125, 127, 128],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row center" style="gap:8px">
    <div class="card" style="width:214px;border-left-color:var(--a)">
      <div class="ct" style="color:var(--a)">MTP 层：外面先算</div>
      <div class="cb">nemotron-h-moe:100<br><span class="cm" style="margin:0">build_lora_mm(ffn_gate_inp, cur)</span><br>
      得到 router_logits，作为第 13 个参数传进去</div></div>
    <div class="card" style="width:214px;border-left-color:var(--b)">
      <div class="ct" style="color:var(--b)">函数体：短路</div>
      <div class="cb">llama-graph.cpp:2024<br><span class="cm" style="margin:0">if (probs_in == nullptr)</span><br>
      非空就直接 <span class="cm" style="margin:0">logits = probs_in</span>，gate_inp 根本不用</div></div>
    <div class="card" style="width:214px;border-left-color:var(--d)">
      <div class="ct" style="color:var(--d)">特例：smallthinker</div>
      <div class="cb">smallthinker:150 把 gate_inp 传成 <span class="cm" style="margin:0">nullptr</span>，<br>
      第 14 个参数传 <span class="cm" style="margin:0">probs</span>（109 行算好）</div></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const msg = wrap.querySelector('#msg');
const texts = [
  '两条通道的意义：<span class="k">路由这件事可以完全搬到 build_moe_ffn 之外</span>。',
  'nemotron-h-moe 的 MTP 草稿头在 100 行自己算 <span class="v">router_logits</span>，<br>MoE 调用在 115 行才发生 —— 中间隔着共享专家的 build_ffn（103-109）。',
  '函数体的判据在 <span class="v">llama-graph.cpp:2024</span>：<span class="v">probs_in</span> 非空就不调用 <span class="v">build_lora_mm(gate_inp, cur)</span>；<br>115 行的调用把 <span class="v">gate_inp</span> 与 <span class="v">probs_in</span> 一起传进去，函数只认后者。',
  '<span class="v">selected_experts_in</span> 是同一思路的极端版：非空则连 <span class="v">argsort_top_k</span> 都跳过（2107-2111）。',
  '本课 24 个文件里，<span class="v">probs_in</span> 出现 2 次，<span class="v">selected_experts_in</span> <span class="k">0 次</span>。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, __MA__); });
tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, __MB__); });
tl.at(7600, () => { msg.innerHTML = texts[2]; U.markLines(document, __MC__); });
tl.at(11600, () => { msg.innerHTML = texts[3]; });
tl.at(15600, () => { msg.innerHTML = texts[4]; U.markLines(document, __MD__); });
'''.replace('__MA__', repr(_MTP_A)).replace('__MB__', repr(_MTP_B))
   .replace('__MC__', repr(_MTP_C)).replace('__MD__', repr(_MTP_D))
)

# ==================================================================== 第 7 幕 · 借图

L.scene(
    kicker='L2-13 · 发现',
    title='<span class="hl-a">借图</span>：3 个文件自己没有 MoE 图',
    sub='phimoe.cpp（55 行）没有一行 build_moe_ffn —— 它的 MoE 行为由 phi3.cpp:153 决定。',
    caption='同类的还有 nomic-bert-moe（借 bert）、minicpm（借 granite）。',
    src=MODELS_H, parts=[(657, 666)], duration=16000,
    mark_src=[657, 659, 660, 662, 663],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['本课文件', '在 models.h 里怎么声明图', 'MoE 行为实际由谁决定'],
  [['src/models/phimoe.cpp', 'models.h:663  using graph = llama_model_phi3::graph<iswa>', 'phi3.cpp:153 的 build_moe_ffn'],
   ['src/models/nomic-bert-moe.cpp', 'models.h:341  using graph = llama_model_bert::graph', 'bert.cpp 的图（本课不引用）'],
   ['src/models/minicpm.cpp', 'models.h:1739 using graph = llama_model_granite::graph', 'granite.cpp 的图（本课不引用）']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const texts = [
  '24 个文件里 <span class="k">21 个</span>在 models.h 里声明了自己的 <span class="v">struct graph</span>；<br>剩下 <span class="k">3 个</span>只是把别人的图 <span class="v">using</span> 过来。',
  'phimoe.cpp 全文 55 行：hparams、张量声明、选图，仅此而已。',
  '<span class="k">文件行数少不等于模型简单</span>：PhiMoE 的路由图完全长在 phi3.cpp 里，<br>靠 <span class="v">ffn_gate_inp == nullptr</span> 分流（phi3.cpp:143）。',
  '这也是静态扫描会把它分到"MoE 文件"的原因 —— 它命中了 <span class="v">ffn_gate_inp</span>，<br>但命中的是<b>张量名</b>，不是图原语。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
tl.at(4000, () => { msg.innerHTML = texts[1]; });
tl.at(7600, () => { msg.innerHTML = texts[2]; });
tl.at(11800, () => { msg.innerHTML = texts[3]; });
'''
)

# ==================================================================== 第 8 幕 · 与注意力变体的组合

L.scene(
    kicker='L2-13 · 组合',
    title='MoE 分支的<span class="hl-c">入口条件</span>也由模型文件说了算',
    sub='phi3.cpp 的图里稠密与 MoE 共用一条主干，靠 ffn_gate_inp == nullptr 分流；PhiMoE 借的就是这张图。',
    caption='回顾 L2-15：SWA / GQA / MLA 等注意力变体的完整谱系在那一课。',
    src='src/models/phi3.cpp', parts=[(143, 165)], duration=20000,
    mark_src=[143, 144, 151, 153, 160, 161, 162],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="faces" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'SWA 模板', b: 'phi3.cpp:60 按 swa_type 选<br>graph&lt;true&gt; / graph&lt;false&gt;<br>phimoe.cpp:49 选同一对', m: 'hparams.swa_type' },
  { c: 'c', t: 'MSA', b: 'minimax-m3.cpp 自带<br>llama_kv_cache_msa_context<br>与 build_attn_msa_fa', m: 'llama_model_minimax_m3::graph' },
  { c: 'e', t: 'MTP 草稿头', b: 'mimo2.cpp:85 · step35.cpp<br>nemotron-h-moe.cpp:4<br>都会切到 graph_mtp', m: 'LLM_GRAPH_TYPE_DECODER_MTP' }
];
const host = wrap.querySelector('#faces');
const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  'MoE 的上游是一段普通的残差流，<span class="k">但这段流可以是三种完全不同的注意力</span>。',
  'phi3 / phimoe：同一份图代码模板化两份，SWA 与非 SWA 各一份，MoE 分支两边都在。',
  'minimax-m3：注意力换成了自己实现的 MSA（608 行里 96 行起是它自己的 graph input），<br>MoE 调用仍然只有一处（561）。',
  'mimo2 / step35 / nemotron-h-moe：主干之外还有一个 <span class="v">graph_mtp</span>，<br>里面同样要再调一次 <span class="v">build_moe_ffn</span>。',
  '结论：<span class="k">build_moe_ffn 不关心上游是哪种注意力</span> —— 它只吃一个 [n_embd, n_tokens] 的张量。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3400 + i * 4200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(16200, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = texts[4];
});
'''
)

# ==================================================================== 第 9 幕 · 变体总表

L.scene(
    kicker='L2-13 · 总表',
    title='把 24 个文件压成 <span class="hl-d">一张表</span>',
    sub='每格都写了证据行号；共享专家与分组路由是两条互不相干的开关。',
    caption='完整的 24 行逐文件表在 source.md 的第五节。',
    src=HPARAMS_H, parts=[(110, 127)], duration=26000,
    mark_src=[110, 112, 114, 115, 124, 125, 126, 127],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['变体', '命中文件（证据行）', '文件数'],
  [['调 build_moe_ffn', '20 个文件共 21 处调用（step35 主干 313 + MTP 512）', '20 / 24'],
   ['没有 MoE 调用', 'minicpm:71 · nomic-bert-moe:38 · phimoe:38 · refact:60', '4 / 24'],
   ['借别人的图', 'phimoe:663 · nomic-bert-moe:341 · minicpm:1739（models.h 行号）', '3 / 24'],
   ['声明共享专家', 'llama:85 · llama4:84 · minicpm:77 · minimax-m3:79 · mistral3:80 · nemotron-h-moe:103 · qwen2moe:54 · refact:66 · step35:110', '9 / 24'],
   ['shexp 自带门控', 'qwen2moe:54 的 ffn_gate_inp_shexp（24 个文件里唯一一个）', '1 / 24'],
   ['selection 偏置', 'mimo2:213 · minimax-01:447 · minimax-m2:135 · minimax-m3:566 · nemotron-h-moe:121 · step35:318/517', '6 / 24'],
   ['gating 读 hparams', 'minimax-m2:139 · minimax-m3:570 · smallthinker:158 · step35:322/521', '4 / 24'],
   ['probs_in 外部路由', 'smallthinker:159 · nemotron-h-moe:127', '2 / 24'],
   ['分组路由', 'n_expert_groups / n_group_used：24 个文件里 0 处', '0 / 24']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '九行覆盖全部 24 个文件；同一文件可以出现在多行（这些开关彼此独立）。',
  '20 个文件真的调了 <span class="v">build_moe_ffn</span>，4 个没有。',
  '共享专家 <span class="k">9 / 24</span>，其中只有 qwen2moe 给它配了独立门控。',
  '<span class="v">exp_probs_b</span> 这一行的来历写在源码注释里：<span class="v">llama-graph.cpp:2062</span> 注明 "introduced in DeepSeek V3"。',
  '分组路由这一行是 <span class="k">0 / 24</span>：它由 GGUF 元数据决定，代码里看不见。',
  '最后一行也是本课的结论：<span class="v">路由策略的开关，一半在参数表，一半在 hparams</span>。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2700, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(24000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ==================================================================== 第 10 幕 · 收束

L.scene(
    kicker='L2-13 · 收束',
    title='一句话：MoE 的可调参数不在算子，在 <span class="hl-a">路由策略</span>',
    sub='共享专家、分组上限、top-k 归一化都是开关；模型文件只是选不同的组合。',
    caption='下一课 L2-14 模型家族（五）稠密 Transformer（上）：没有专家时图长什么样。',
    src=GRAPH_H, parts=[(1112, 1132)], duration=22000,
    mark_src=[1112, 1113, 1119, 1120, 1121, 1123, 1127, 1132],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问', '答'],
  [['控制专家选择的参数有哪些？', 'n_expert · n_expert_used · gating_op · exp_probs_b · norm_w · w_scale · probs_in · selected_experts_in'],
   ['哪一个决定"选几个专家"？', 'n_expert_used —— 2109 行 ggml_argsort_top_k(..., n_expert_used)'],
   ['分组路由怎么开？', '不是参数：hparams.n_expert_groups 大于 1 时自动生效（2083）'],
   ['共享专家在哪？', '不在 build_moe_ffn 里：调用点外的第二条支路（9/24 个文件）']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '在 <span class="mono">llama-graph.cpp:1993</span> 的签名里，哪些参数会改变"选中哪些专家"，' +
  '哪些只改变"选中之后怎么加权"？',
  '<b>改变选谁：</b><span class="mono">n_expert</span>（候选集合大小）、' +
  '<span class="mono">n_expert_used</span>（top-k 的 k，见 2109）、' +
  '<span class="mono">gating_op</span>（概率函数，2040-2059）、' +
  '<span class="mono">exp_probs_b</span>（选择偏置，2066，只加到 selection_probs）、' +
  '<span class="mono">probs_in</span>（绕开 gate_inp，2024-2032）、' +
  '<span class="mono">selected_experts_in</span>（绕开 top-k，2107-2111）。<br>' +
  '<b>只改加权：</b><span class="mono">norm_w</span>（2134-2148）、' +
  '<span class="mono">w_scale</span>（2149-2151）。<br>' +
  '<b>不改选择：</b>权重 <span class="mono">gate_inp/up_exps/gate_exps/down_exps</span>、' +
  '激活 <span class="mono">type_op</span>（2221 的 switch）、层号 <span class="mono">il</span>。<br>' +
  '另外 <span class="mono">hparams.n_expert_groups / n_group_used</span> 会在 2083-2103 段' +
  '把一部分专家直接 masking 成 <span class="mono">-INFINITY</span> —— 它也是"选谁"，但不是参数。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '把这一课压成四问四答：',
  '八个参数：两个定规模，四个定概率与权重，两个是外部接管通道。',
  '<span class="v">n_expert_used</span> 是唯一直接决定"几个专家"的参数。',
  '分组路由与共享专家都<b>不在这张参数表里</b> —— 一个在 hparams，一个是图上的第二条支路。',
  '下一课 <span class="k">L2-14</span>：没有专家的 Transformer，图画起来是什么样。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 3600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(18600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== source.md

L.section(
    '一、分组方法与本课的 24 个文件',
    '本课覆盖的文件清单**不靠人工挑选**，而是由计划脚本给出：\n\n'
    '```text\n'
    'python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L2-13"{print $2}\'\n'
    '```\n\n'
    '`tools/plan_model.py` 把 `src/models/*.cpp` 按图原语正则 `build_moe_ffn|ffn_gate_exps|'
    'ffn_gate_inp` 分到 `moe` 组，再排序对半分给 `L2-12` 与 `L2-13`。'
    '**分组依据是命中图原语，不是内容判定。** 实测三个后续事实：\n\n'
    '| 事实 | 命令 | 结果 |\n|---|---|---|\n'
    '| 有 4 个文件全篇没有 MoE 调用 | `grep -c "build_moe_ffn(" src/models/*.cpp` | minicpm / nomic-bert-moe / phimoe / refact 各 0 次 |\n'
    '| 有 3 个文件借别人的图 | `grep -n "using graph" src/models/models.h` | 341 / 663 / 1739 三行 |\n'
    '| 全部 24 个文件都没写分组路由 | `grep -c "n_expert_groups" <24 文件>` | 全为 0 |\n\n'
    '这三点分别在第 7 幕、第 3 幕与本节的表格里展开。')

L.section(
    '二、★ build_moe_ffn 的两条声明：哪些参数控制专家选择',
    '`build_moe_ffn` 有**两条重载**：一条不带偏置张量（下面这段，其余参数带默认值），'
    '一条在 `gate_inp` / `up_exps` / `gate_exps` / `down_exps` 后面各插一个 `_b` 偏置。'
    '本课 24 个文件里只有 `openai-moe.cpp:132` 用了带偏置的那条，其余都走这条。\n\n'
    '按"是否改变专家选择"把参数分成三组：\n\n'
    '| 组 | 参数 | 依据 |\n|---|---|---|\n'
    '| **决定选谁** | `n_expert` `n_expert_used` `gating_op` `exp_probs_b` `probs_in` `selected_experts_in` | 2084/2120、2109、2040-2059、2066、2024-2032、2107-2111 |\n'
    '| **只改加权** | `norm_w` `w_scale` | 2134-2148、2149-2151 |\n'
    '| **不改选择** | `gate_inp` `up_exps` `gate_exps` `down_exps`（权重）、`type_op`（专家内部激活，2221）、`il`（`cb` 命名） | 权重与激活，不是路由 |\n\n'
    '还有两个"选谁"的开关**不是参数**：`hparams.n_expert_groups`（2083）与 `hparams.n_group_used`（2096）。',
    src=GRAPH_H, parts=[(1113, 1132)], lang='c')

L.section(
    '三、函数体逐段：从 logits 到 weights',
    '第 2 幕引用的行号都在这段实现里。三段结构：\n\n'
    '1. **算 logits 或直接用 probs_in**（2024-2032）：`probs_in == nullptr` 时才调用 `build_lora_mm(gate_inp, cur)`。\n'
    '2. **logits 变概率**（2039-2068）：`gating_op` 的四分支 switch；随后 `exp_probs_b` **只加到 `selection_probs`**，'
    '源码注释写明"leave probs unbiased as it\'s later used to get expert weights"。\n'
    '3. **top-k 与加权**（2106-2152）：`ggml_argsort_top_k(..., n_expert_used)` 选出专家，'
    '`ggml_get_rows` 取权重，然后才是可选的 `norm_w` 与 `w_scale`。\n\n'
    '注意 2020 / 2072 / 2076 / 2114 / 2228 这几处 `arch == LLM_ARCH_*` 特判：'
    '它们是**架构属性**，模型文件同样没有选择权。',
    src=GRAPH_CPP, parts=[(2039, 2068)], lang='c')

L.section(
    '四、★ 分组路由：开关在 GGUF，不在代码',
    'DeepSeek V3 的 device-limited routing 在 llama.cpp 里的实现是 2081-2103 这一段，'
    '入口条件是 `hparams.n_expert_groups > 1`。三个 hparams 字段的来源：',
    src=GRAPH_CPP, parts=[(2081, 2104)], lang='c')

L.section(
    '五、★ 共享专家：三种接法',
    '共享专家（shared expert）是 DeepSeekMoE 提出的结构：**一部分专家对所有 token 都生效**。'
    '在 llama.cpp 里它**不是 `build_moe_ffn` 的参数**，而是调用点外的一条支路。'
    '本课 24 个文件里有 9 个声明了 `ffn_*_shexp` 张量，接法分三类：\n\n'
    '| 接法 | 文件（证据行） | 形状 |\n|---|---|---|\n'
    '| 自带门控 | `qwen2moe.cpp:54,147-165` | 额外一个 `[n_embd]` 的 `ffn_gate_inp_shexp`，sigmoid 门乘在输出上 |\n'
    '| 直接相加 | `llama4.cpp:232-240`、`step35.cpp:327-335`、`minimax-m3.cpp:575-581` | 一次普通 `build_ffn`，结果 `ggml_add` 进 `moe_out` |\n'
    '| 只声明不接线 | `llama.cpp:85-88`、`minicpm.cpp:77-81`、`mistral3.cpp:80-83`、`refact.cpp:66-70` | `create_tensor` 建了张量，本文件的图里没有对应支路 |\n\n'
    '**一个容易踩的坑**：`ffn_up_exps_s` / `ffn_gate_exps_s` / `ffn_down_exps_s` 不是共享专家。'
    '结尾的 `_s` 是 `build_lora_mm_id(w, cur, ids, w_s)` 的**每专家缩放**（`llama-graph.h:1059-1063`），'
    '`llama.cpp:215-217` 等 5 个调用点传的正是这三个。',
    src='src/models/qwen2moe.cpp', parts=[(145, 168)], lang='c')

# ---------------------------------------------------------------- 逐文件证据（24 个）

_PER_FILE = [
    ('llada-moe.cpp', 163, '全篇唯一一处 build_moe_ffn；norm_w 写死 false，exp_probs_b 传 nullptr。文件里没有 shexp，也没有 n_expert_groups。', [(126, 137)]),
    ('llama.cpp', 250, '`llama` 架构在 `hparams.n_ff_shexp > 0` 时声明三张共享专家张量。但它的图（196-217）传给 `build_moe_ffn` 的是 `ffn_*_exps_s`（每专家缩放），**不是 shexp** —— 全文件 `shexp` 出现 4 次，全在这 5 行的声明里。', [(84, 88)]),
    ('llama4.cpp', 272, '共享专家走 `build_ffn` + `ggml_add`，没有自己的门控；MoE 侧 `gating_op` 是 `SIGMOID`、`norm_w` 是 `false`。llama4 在 `build_moe_ffn` 体内还有两处 arch 特判（`llama-graph.cpp:2020` 的 `weight_before_ffn`、2072-2074 的 `selection_probs = logits`）。', [(230, 241)]),
    ('maple.cpp', 150, '唯一把 `w_scale` 写成字面量 `1.0f` 的调用点（其余 20 处都读 `hparams.expert_weights_scale`）。MAPLE 在 `build_moe_ffn` 里另有一处 arch 特判：`llama-graph.cpp:2228` 的 swiglu clamp 分支。', [(121, 131)]),
    ('mellum.cpp', 219, '用 18 参数的重载：除权重外还传 `ffn_up_exps_s` / `ffn_gate_exps_s` / `ffn_down_exps_s` 三个每专家缩放张量。', [(172, 188)]),
    ('mimo2.cpp', 396, '传 `ffn_exp_probs_b`（选择偏置），`gating_op` 是 `SIGMOID`。同文件另有 `graph_mtp`（`models.h:2555`），MTP 分支里还要再走一次 FFN。', [(206, 220)]),
    ('minicpm.cpp', 89, '声明了 MoE 与共享专家张量，但 `build_arch_graph` 用的是 `llama_model_granite::graph`（`models.h:1739`）—— 全文件 0 处 `build_moe_ffn`。', [(70, 81)]),
    ('minimax-01.cpp', 484, '`exp_probs_b` + `SOFTMAX` + `norm_w=true`；FFN 之后还有 `f_residual_scale`（455 行）。', [(442, 453)]),
    ('minimax-m2.cpp', 168, '`gating_op` 不写死，转成 `hparams.expert_gating_func` —— 概率函数由 GGUF 决定。', [(130, 140)]),
    ('minimax-m3.cpp', 608, '路由专家用 `LLM_FFN_SWIGLU_OAI_MOE` 激活；`norm_w` 与 `gating_op` 都读 hparams。共享专家宽度是 `n_ff_exp * n_expert_shared`（79-81），它是本课 24 个文件里**唯一**读 `n_expert_shared` 的。', [(559, 582)]),
    ('mistral3.cpp', 235, '18 参数重载（每专家缩放）。共享专家张量在 80-83 声明，但 MoE 分支（186-209）里没有接 —— 全文件 `shexp` 只出现在声明处。', [(193, 207)]),
    ('nemotron-h-moe.cpp', 164, '`gate_exps` 传 `nullptr`（源码注释 "no gate"），激活换成 `LLM_FFN_RELU_SQR`；第 13 个参数传 `router_logits`，路由 logits 在 100 行已用 `build_lora_mm` 算好。', [(115, 130)]),
    ('nomic-bert-moe.cpp', 56, '用 `hparams.moe_every_n_layers` 决定哪些层是 MoE（37 行），图借 `llama_model_bert::graph`（`models.h:341`）。', [(37, 46)]),
    ('olmoe.cpp', 173, '基础组合里最"素"的一个：`norm_w=false`、无偏置、`SOFTMAX`、无 shexp。', [(136, 146)]),
    ('openai-moe.cpp', 175, '**唯一**使用 17 参数"带偏置"重载的调用点（`ffn_gate_inp_b` 等 4 组偏置）；`gating_op` 是 `SOFTMAX_WEIGHT`，即 softmax 作用在**选中的权重**上（`llama-graph.cpp:2127-2132`）。', [(131, 142)]),
    ('phi3.cpp', 196, '稠密与 MoE 共用同一张图，靠 `ffn_gate_inp == nullptr` 分流；`build_arch_graph` 按 `swa_type` 在 `graph<true>` / `graph<false>` 之间选（59-65）。PhiMoE 借的就是这张图。', [(143, 164)]),
    ('phimoe.cpp', 55, '只有 hparams、张量声明与选图三段：38-41 声明四张 MoE 张量（这就是静态扫描把它分进 MoE 组的原因），48-54 按 `swa_type` 选 `graph<iswa>`。**0 处 `build_moe_ffn`**，MoE 行为完全由 `phi3.cpp:153` 决定。', [(38, 41)]),
    ('qwen2moe.cpp', 194, '共享专家有自己的门控：`ffn_gate_inp_shexp`（54 行，形状 `[n_embd]`）出标量，经 sigmoid 门（151）乘到共享专家输出上，再在 165 行加回 `moe_out`。24 个文件里只有它声明 `LLM_TENSOR_FFN_GATE_INP_SHEXP`。', [(145, 166)]),
    ('qwen3moe.cpp', 179, '18 参数重载 + `norm_w=true`；无 shexp、无偏置。', [(136, 152)]),
    ('qwen3vlmoe.cpp', 190, 'MoE 侧参数与 qwen3moe 逐字相同（SILU / true / hparams.expert_weights_scale / SOFTMAX）。差别不在 MoE：这个文件里 `clip_` / `vision` / `mmproj` 命中 0 次，视觉塔在 `tools/mtmd` 里（不计入本课覆盖率，见 L2-10）。', [(144, 156)]),
    ('refact.cpp', 160, '按 `n_expert == 0`（50 行）分别声明稠密与 MoE 张量，MoE 分支里连共享专家张量都建了；但它的 `graph` 的 FFN 只有一条 `build_ffn`（128-133），全文件 0 处 `build_moe_ffn`。', [(59, 70)]),
    ('rnd1.cpp', 177, '基础组合：`nullptr` 偏置、`norm_w=true`、`SOFTMAX`、无 shexp。', [(138, 150)]),
    ('smallthinker.cpp', 188, '`gate_inp` 传 `nullptr`，第 14 个参数传 `probs`（109 行算好）；激活是 `LLM_FFN_RELU`。', [(148, 159)]),
    ('step35.cpp', 560, '主干与 MTP 各一处 MoE 调用（313 / 512），参数全部读 hparams；共享专家用 `build_ffn` + `ggml_add`（327-335 / 525-533）。', [(512, 533)]),
]

for _name, _nlines, _prose, _parts in _PER_FILE:
    _rel = 'src/models/' + _name
    assert _rel in COVERED, _rel
    _blk = _parts[0]
    L.section(
        f'六 · `{_name}`（{_nlines} 行，引用 {_blk[0]}-{_blk[1]}）',
        _prose,
        src=_rel, parts=_parts, lang='c')

L.section(
    '七、借图：models.h 里的 `using graph = ...`',
    '`llm_graph_context` 是抽象基类；每个架构要么在 `models.h` 里声明自己的 `struct graph`，'
    '要么用 `using` 别名借别人的。本课 24 个文件里 21 个是前者，3 个是后者。'
    '被借的三张图分别在 `bert.cpp` / `granite.cpp`（L2-12 覆盖）与 `phi3.cpp`（本课覆盖）。\n\n'
    '所以"读一个模型文件"不足以保证理解它的图：`phimoe.cpp` 只有 55 行，'
    '它的 MoE 行为完全由 `phi3.cpp:153` 决定。',
    src=MODELS_H, parts=[(657, 666)], lang='c')

L.section(
    '八、hparams 里的专家字段与它们的 GGUF 来源',
    '`build_moe_ffn` 体内引用了 5 个 hparams 字段，它们都不是函数参数：\n\n'
    '| 字段 | 声明 | 从 GGUF 读入 | 在 build_moe_ffn 里的行 |\n|---|---|---|---|\n'
    '| `n_expert_groups` | `llama-hparams.h:114` | `llama-model.cpp:1266` | 2083、2084、2087 |\n'
    '| `n_group_used` | `llama-hparams.h:115` | `llama-model.cpp:1267` | 2096 |\n'
    '| `n_group_experts` | `llama-hparams.h:116` | GROVEMOE 自己在 `grovemoe.cpp:7` 读 | 2117 |\n'
    '| `expert_weights_scale` | `llama-hparams.h:124` | 各模型文件按架构读 | 2149 |\n'
    '| `expert_weights_norm` | `llama-hparams.h:125` | 各模型文件按架构读 | 2134 |\n\n'
    '`llama-model.cpp:1294-1305` 是一组硬断言，它把合法组合钉死：'
    '`n_expert_groups < n_expert`；一旦 `n_expert_groups > 1`，就必须'
    '`n_expert % n_expert_groups == 0` 且 `0 < n_group_used < n_expert_groups`。',
    src=HPARAMS_H, parts=[(110, 127)], lang='c')

L.section(
    '九、GGUF 元数据到 hparams：两个字段的来路',
    '分组路由的两个字段走的是通用读取路径，与架构无关 —— 任何架构只要 GGUF 里有'
    '`*.expert_group_count` / `*.expert_group_used_count` 就会被读进来。',
    src=MODEL_CPP, parts=[(1265, 1305)], lang='c')

L.section(
    '十、`probs_in` 的短路判据',
    '第 6 幕引用的"传了 `probs_in` 就不再碰 `gate_inp`"就来自这两行。'
    '顺带一提：`gate_inp_b`（路由偏置）在 `probs_in` 路径下**仍然会加**（2034-2037）。',
    src=GRAPH_CPP, parts=[(2022, 2037)], lang='c')

# ---------------------------------------------------------------- 说明 / 前置 / 目标 / 结论

L.footnote_add('本课计划内的 **24 个文件**全部来自 `python3 tools/plan_matrix.py --files` 中 '
               '`L2-13` 的分配，逐字引用并计入覆盖率。')
L.footnote_add('`src/llama-graph.cpp`（`build_moe_ffn` 的实现）与 `src/llama-graph.h`（它的两条声明）'
               '按计划属于 `L2-06`；本课把"参数如何控制专家选择"作为核心引用对象，'
               '**计入覆盖率声明**，但这不改变计划的文件归属。')
L.footnote_add('`src/models/models.h` 按计划属于 `L2-10` / `L2-15`；本课只引用它的三处 '
               '`using graph = ...` 别名（341 / 663 / 1739）来证明"借图"。')
L.footnote_add('`src/llama-hparams.h`（`L2-02`）与 `src/llama-model.cpp`（`L2-05`）本课各引用一段，'
               '用来定位分组路由那两个字段的来路。')
L.footnote_add('第 5 幕与第 9 幕的统计数字（21 处调用点、各取值的出现次数、'
               '`n_expert_groups` 0 次）由脚本对这 24 个文件静态扫描得出，不是目测。')

L.prereqs('`L2-12`（模型家族（三）稀疏专家 MoE（上）：`build_moe_ffn` 的基础序列）')

L.goal(
    '说出 `build_moe_ffn` 的参数里哪些控制**专家选择**（对应验收点），哪些只控制加权；',
    '解释"分组路由"为什么在这 24 个文件里一次都搜不到（它由 GGUF 元数据决定）；',
    '说出共享专家的三种接法，并指出 `ffn_up_exps_s` 与 `ffn_gate_shexp` 的区别；',
    '判断一个 `src/models/*.cpp` 是不是"借图"（`using graph = ...`），以及这对读代码意味着什么。')

L.conclusion(
    '★ 控制专家选择的 8 个参数',
    '`llama-graph.h:1113-1132` 的签名里，真正改变"选哪些专家 / 权重怎么算"的是：\n\n'
    '| 参数 | 生效行（llama-graph.cpp） | 作用 |\n|---|---|---|\n'
    '| `n_expert` | 2084 / 2120 | 候选专家总数 |\n'
    '| `n_expert_used` | 2109 | top-k 的 k |\n'
    '| `gating_op` | 2040-2059 | logits 变概率的函数 |\n'
    '| `exp_probs_b` | 2066 | 选择偏置（只影响选谁） |\n'
    '| `norm_w` | 2134-2148 | top-k 权重归一化 |\n'
    '| `w_scale` | 2149-2151 | 路由权重缩放 |\n'
    '| `probs_in` | 2024-2032 | 绕开 `gate_inp` |\n'
    '| `selected_experts_in` | 2107-2111 | 绕开 top-k |\n\n'
    '`type_op`（专家内部激活）、`il`、以及四个权重张量**不控制选择**。')

L.conclusion(
    '★ 两个开关不在这张参数表里',
    '**分组路由**：入口是 `hparams.n_expert_groups > 1`（`llama-graph.cpp:2083`），'
    '字段从 GGUF 读入（`llama-model.cpp:1266-1267`）。本课 24 个文件里这两个名字出现 **0 次**。\n\n'
    '**共享专家**：调用点外的第二条支路。9 / 24 个文件声明了 `ffn_*_shexp` 张量，'
    '但接法有三种（自带门控 / 直接相加 / 只声明不接线），且没有一种经过 `build_moe_ffn` 的参数。')

L.conclusion(
    '一个辨别技巧：`_exps_s` 不是 shexp',
    '`ffn_up_exps_s` / `ffn_gate_exps_s` / `ffn_down_exps_s` 结尾的 `_s` 是 '
    '`build_lora_mm_id(w, cur, ids, w_s)` 的**每专家缩放**（`llama-graph.h:1059-1063`），'
    '与共享专家无关。判断一个文件是否真的用了共享专家，要看 `ffn_gate_shexp` / '
    '`ffn_up_shexp` / `ffn_down_shexp` 这三个名字，并且要在**图里**找到使用它的支路 —— '
    '`llama.cpp` / `mistral3.cpp` / `refact.cpp` 都是声明了却没接线。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
