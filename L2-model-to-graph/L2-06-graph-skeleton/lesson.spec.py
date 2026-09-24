#!/usr/bin/env python3
"""L2-06 · ★ 计算图骨架 llama-graph —— 课件 spec。

运行：python3 L2-model-to-graph/L2-06-graph-skeleton/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_CPP = 'src/llama-graph.cpp'
SRC_H = 'src/llama-graph.h'


def M(a, notes, hls):
    """把【上游行号】换算成渲染后的 code 行下标。

    注解行（//>> 开头的 notes）会插进 code 里但不占上游行号，
    所以它之后的行在渲染结果里要整体后移 —— marks 用错了就会高亮错行。
    """
    out = []
    for ln in hls:
        k = ln - a
        out.append(k + sum(1 for n in notes if n < k))
    return sorted(out)


def JS(lst):
    """把 Python 列表写成 JS 数组字面量，用于注入 visual 里的 U.markLines。"""
    return '[' + ', '.join(str(x) for x in lst) + ']'


L = Lesson(
    id='L2-06',
    layer='L2 · 从模型到图',
    title='★ 计算图骨架 llama-graph',
    codecap='src/llama-graph.{cpp,h}（逐字引用）',
    nav={'prev': {'href': '../L2-05-batch-and-model-build/index.html', 'label': 'L2-05 批、解码参数与模型装配'},
         'next': {'href': '../L2-07-context-and-decode/index.html', 'label': 'L2-07 ★ 上下文与解码'}},
)

# ---------------------------------------------------------------- 覆盖声明
L.cover(SRC_CPP, SRC_H)

L.note('**一句话**：`src/llama-graph.{h,cpp}` 是"模型定义 → 计算图"之间唯一的翻译层。'
       '`src/models/` 下 156 个文件里有 137 个不直接调 `ggml_*`，而是调这里的 `build_*` 原语；'
       '原语负责把**超参**变成**具体算子**，把算子挂到一张 `ggml_cgraph` 上。'
       '余下 19 个（视觉塔 clip.cpp、BERT 系、RWKV/Mamba 系等）在自己的 `build()` 里直接调 ggml。')
L.note('回顾 L1-01 / L1-03：`ggml_tensor` 是图上的节点（数据面 + 身份面），`ggml_cgraph` 是节点序列。'
       '本课讲的是：**这些节点是谁、按什么规则造出来的**。回顾 L2-02：`hparams` 决定层数、头数、专家数；'
       '回顾 L2-04：KV cache 的读写算子由 memory 上下文提供；回顾 L2-05：`ubatch` 决定这一批有多少 token。'
       '这四样东西在 `llm_graph_context` 里汇合。')

# ------------------------------------------------------------------ 第 1 幕

N1 = {0: 'res 持有"这一轮推理的结果"：张量 arena、计算图、以及全部输入张量',
      2: 'ctx0：这张图的张量 arena。所有 build_* 都往它里面 ggml_new_tensor',
      3: 'gf：要执行的节点序列。build_* 每造一个节点就 ggml_build_forward_expand 挂上去',
      8: 'cb 是给节点起名字的钩子；名字只用于调试与图打印，不参与计算'}

L.scene(
    kicker='L2-06 · 全局',
    title='llama-graph：模型的最后一公里',
    sub='左边三个只读输入，右边两样产物：一处张量 arena（ctx0）和一张计算图（gf）。',
    caption='本课只引用 src/llama-graph.cpp 与 src/llama-graph.h 两个文件，均计入覆盖率。'
            '下游消费这张图的是 L2-07（建图与解码）与 L4-02（切给后端）。',
    src=SRC_H, parts=[(1034, 1042)], duration=18000,
    notes=N1, marks=M(1034, N1, [1034, 1036, 1037, 1042]),
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip a">hparams · cparams · ubatch</span><span class="arrow">-&gt;</span>
    <span class="chip b">llm_graph_context</span><span class="arrow">-&gt;</span>
    <span class="chip c">ctx0 + gf</span><span class="arrow">-&gt;</span>
    <span class="chip d">调度器 / 后端</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '输入 · 三个只读引用', m: 'hparams / cparams / ubatch',
    b: '结构超参、运行参数、<br>这一批的 token 数' },
  { c: 'b', t: '主体 · 一个上下文对象', m: 'struct llm_graph_context',
    b: 'src/models/ 下 137 个文件<br>出现它的 build_* 调用' },
  { c: 'c', t: '产物 · arena + 图', m: 'ggml_context * ctx0 / ggml_cgraph * gf',
    b: 'ctx0 装张量，gf 装节点<br>（L1-03 讲过 gf 怎么遍历）' },
  { c: 'd', t: '下游 · 谁消费它', m: 'sched / backend_cpu',
    b: 'L2-07 建图，L4-02 把 gf<br>切给 CPU / CUDA / Metal' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '先看全局：<span class="k">llama-graph</span> 夹在"模型定义"和"ggml 计算图"之间，是两个世界唯一的接口。',
  '<span class="v">llm_graph_context</span> 一造出来就握着三样只读输入：<br>' +
    '<span class="k">hparams</span>（L2-02）、<span class="k">cparams</span>、<span class="k">ubatch</span>（L2-05）。',
  '它的任务只有一个：<span class="k">把超参翻译成算子</span>。<br>' +
    '翻译的产物落在两个地方 —— <span class="v">ctx0</span>（张量）和 <span class="v">gf</span>（节点）。',
  '<span class="v">cb(cur, "ffn_up", il)</span> 这类调用只给节点起名字，方便出错时定位；<br>' +
    '它不影响计算结果，但贯穿了整份 llama-graph.cpp。',
  '<span class="v">res</span> 还兼管输入：占位张量（token、位置、mask）由 build_inp_* 造出并登记，' +
    '解码前由 L2-07 填真实数据。'
];
defs.forEach((_, i) => tl.at(600 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(16600, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

N2 = {4: 'ubatch 是这一轮真正要算的 token 集合（L2-05 切出来的）',
      5: 'n_embd：隐藏维。所有 build_* 的矩阵乘都围绕它转',
      6: 'n_layer()：层数。它就是模型 build() 里那个 for 循环的上界（L2-02）',
      16: 'n_expert：MoE 的路由宽度。稠密模型的这一项是 0',
      26: 'n_tokens：本幕之后所有 build_* 的最外层维度',
      40: 'ctx0 不 new 出来，而是从 res 里取 —— 图复用时 arena 也一起复用',
      41: 'gf 同理：图复用的判据就是"参数没变就沿用同一张 gf"'}

HL2 = M(1455, N2, [1460, 1461, 1463, 1471, 1472, 1479, 1480, 1481, 1495, 1496])

L.scene(
    kicker='L2-06 · 核心',
    title='★ <span class="hl-a">llm_graph_context</span>：把超参摊平成常量的工作台',
    sub='构造函数一行一个字段 —— 左边是成员，右边是来源。看右列就知道"谁决定了图的形状"。',
    caption='注意 n_expert_used：cparams.warmup 为真时它直接用 hparams.n_expert（全部专家），'
            '而不是 hparams.n_expert_used()。',
    src=SRC_CPP, parts=[(1455, 1498)], duration=26000,
    notes=N2, marks=HL2,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '来自 hparams（结构）', m: 'n_embd · n_layer() · n_head_kv() · n_expert · f_norm_rms_eps',
    b: '权重加载后就定了，<br>整轮推理里不再变（L2-02）' },
  { c: 'b', t: '来自 cparams（运行）', m: 'n_ctx · rope_freq_base · yarn_* · pooling_type',
    b: '用户命令行给的推理参数；<br>改它们就要重建图' },
  { c: 'c', t: '来自 ubatch（这一批）', m: 'n_tokens',
    b: '图的最外层维度：<br>prompt 阶段几百，decode 阶段是 1（L2-05）' },
  { c: 'd', t: '来自 res（产物）', m: 'ctx0 = res-&gt;get_ctx() / gf = res-&gt;get_gf()',
    b: 'arena 与计算图都由外层结果对象持有，<br>图复用靠的就是它们' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '构造函数 44 行，全是 <span class="k">成员 (来源)</span>。读右列即可：',
  '<span class="v">hparams</span> 决定"图长什么样"：n_embd 定宽度、n_layer() 定层数、' +
    'n_head_kv() 定 KV 头数、n_expert 定路由宽度。',
  '<span class="v">cparams</span> 决定"算子带什么参数"：rope 的 freq_base / freq_scale、' +
    'yarn 的缩放因子、pooling 方式。',
  '<span class="v">ubatch.n_tokens</span> 决定"图有多宽"：<br>' +
    '所有 <span class="m">[n_embd, n_tokens]</span> 里的第二维就是它。',
  '<span class="v">ctx0</span> 与 <span class="v">gf</span> 是从 <span class="v">res</span> 里取的，' +
    '不是新造的 —— <span class="k">这就是图能复用的一半原因</span>（另一半在 L2-07）。',
  '一句话：<span class="k">模型架构的差异，到这一层就已经被压缩成"哪些字段、什么值"</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(20500, () => {
  els.forEach(e => e.style.opacity = '1');
  U.markLines(document, ''' + JS(HL2) + ''');
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 3 幕

N3 = {7: 'LLM_NORM：标准 LayerNorm，eps 来自 hparams.f_norm_eps',
      8: 'LLM_NORM_RMS：RMSNorm，eps 来自 hparams.f_norm_rms_eps —— 两者的 eps 是不同的字段',
      9: 'LLM_NORM_GROUP：组归一化，需要 3 维输入，所以前后各夹一次 reshape',
      21: 'mw 是归一化权重（每个模型都有）；mb 是偏置（只有部分架构有）',
      28: '有 mb 时再 ggml_add 一次 —— 这就是 LayerNorm 与 RMSNorm 在图上的差别'}

HL3 = M(1583, N3, [1590, 1591, 1592, 1604, 1605, 1611, 1612])

L.scene(
    kicker='L2-06 · 原语 · 归一化',
    title='<span class="hl-b">build_norm</span>：一个 switch 分派三种归一化',
    sub='选哪个算子由 llm_norm_type 决定；eps 从 hparams 来；权重与偏置是可选的第二步。',
    caption='LLM_NORM_GROUP 要先把 [n_embd, n_tokens] reshape 成 3 维才能做组归一化，做完再 reshape 回来。',
    src=SRC_CPP, parts=[(1583, 1615)], duration=20000,
    notes=N3, marks=HL3,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'LLM_NORM', m: 'ggml_norm(ctx0, cur, hparams.f_norm_eps)',
    b: '标准 LayerNorm。<br>后面通常还会跟 ggml_mul + ggml_add。' },
  { c: 'b', t: 'LLM_NORM_RMS', m: 'ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps)',
    b: 'RMSNorm，没有减均值那一步。<br>Llama 系全用它。' },
  { c: 'c', t: 'LLM_NORM_GROUP', m: 'ggml_reshape_3d + ggml_group_norm + ggml_reshape_2d',
    b: '组归一化：把隐藏维按<br>hparams.n_norm_groups 分组。' },
  { c: 'd', t: '可选第二步', m: 'ggml_mul(cur, mw) [+ ggml_add(cur, mb)]',
    b: 'mw（缩放）几乎人人都有；<br>mb（偏置）只有部分架构有。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '一个 <span class="k">switch (type)</span>，三个 case，三种归一化。',
  '<span class="v">LLM_NORM</span> 与 <span class="v">LLM_NORM_RMS</span> 只差一个函数名和<b>一个 eps 字段</b>：<br>' +
    '<span class="m">f_norm_eps</span> 与 <span class="m">f_norm_rms_eps</span>，都来自 hparams。',
  '<span class="v">LLM_NORM_GROUP</span> 特殊：先 reshape 成 3 维、做完再 reshape 回来，<br>' +
    '因为组归一化要在"组"这一维上算统计量。',
  '归一化之后的 <span class="v">ggml_mul(cur, mw)</span> 才是"权重"；<br>' +
    '这一步几乎人人都有，所以它写在 switch 外面。',
  '<span class="v">ggml_add(cur, mb)</span> 只有 mb 非空时才加 —— <span class="k">有没有偏置，' +
    '就是 RMSNorm 家族与 LayerNorm 家族在图上的分界</span>。',
  '同一个 build_norm，四条不同架构分支；到 ggml 层就变成 2 到 4 个节点。<br>' +
    '这正是 L2-14 / L2-15 里那些稠密 Transformer 长得像的原因。'
];
defs.forEach((_, i) => tl.at(600 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(16000, () => {
  els.forEach(e => e.style.opacity = '1');
  U.markLines(document, ''' + JS(HL3) + ''');
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

N4 = {0: 'up 投影：稠密 FFN 的第一个矩阵乘（走 build_lora_mm，可顺带叠加 LoRA）',
      13: 'gate 投影有两种接法：SEQ 接在 up 之后，PAR 与 up 并行',
      17: 'LLM_FFN_SEQ：gate 吃的是 tmp（已经算完的 up）',
      22: 'LLM_FFN_PAR：gate 吃的是 cur（原来的隐藏状态），这就是 GLU 家族',
      41: '激活函数在这里分派：SILU / GELU / RELU / GEGLU / SWIGLU / SWIGLU_OAI_MOE ...',
      42: '最常用的 SILU 分支：有 gate 且并行时用 ggml_swiglu_split 一步算完'}

HL4 = M(1786, N4, [1786, 1799, 1803, 1808, 1827, 1828, 1835])

L.scene(
    kicker='L2-06 · 原语 · 前馈',
    title='<span class="hl-c">build_ffn</span>：三个矩阵乘 + 一个激活',
    sub='up / gate / down 三次矩阵乘都由 build_lora_mm 发出；激活由 type_op 分派，GLU 变体由 type_gate 决定。',
    caption='down 投影在本幕区间之外（1925 行起）：cur = build_lora_mm(down, cur)。',
    src=SRC_CPP, parts=[(1786, 1835)], duration=24000,
    notes=N4, marks=HL4,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '① up 投影', m: 'build_lora_mm(up, cur) -&gt; ffn_up',
    b: '把隐藏状态升到中间维。<br>up_b / up_s 是可选的偏置与缩放。' },
  { c: 'b', t: '② gate 投影', m: 'LLM_FFN_SEQ / LLM_FFN_PAR',
    b: 'PAR（并行）是 GLU 家族：<br>gate 与 up 各算一份，再相乘。' },
  { c: 'c', t: '③ 激活', m: 'ggml_swiglu_split / ggml_silu / ggml_gelu',
    b: '由 type_op 分派；GLU 变体<br>一次拿到 gate 与 up 两份输入。' },
  { c: 'd', t: '④ down 投影', m: 'build_lora_mm(down, cur)（1925 行起）',
    b: '降回 n_embd。<br>GLM4 / JAIS2 会被强制成 F32 累加。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '稠密 FFN 的骨架：<span class="k">up -&gt; gate -&gt; 激活 -&gt; down</span>。',
  '<span class="v">build_lora_mm(up, cur)</span>：一次矩阵乘，外加可选的 LoRA 与 per-tensor scale。<br>' +
    '注意它同时覆盖了"有权重"和"没有 up（只有 gate 的模型）"两种情况。',
  '<span class="v">type_gate</span> 决定 gate 的位置：<br>' +
    '<span class="m">SEQ</span> 吃 up 的输出，<span class="m">PAR</span> 直接吃隐藏状态 —— 后者才叫 GLU。',
  '<span class="v">switch (type_op)</span> 是这一课最长的一个 switch：<br>' +
    'SILU / GELU / RELU / RELU_SQR / SWIGLU / SWIGLU_OAI_MOE / GEGLU / REGLU。',
  '最常走的 SILU + PAR 分支只用一行：<span class="v">ggml_swiglu_split(ctx0, cur, tmp)</span>。<br>' +
    '它还带一个按层可配的限幅（swiglu_clamp_shexp），超过阈值就改走 clamp + silu + mul。',
  '<span class="k">同一个 build_ffn，不同架构只是换 type_op / type_gate 两个枚举值</span>；' +
    '图上的节点形状完全一致。'
];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(17500, () => {
  els.forEach(e => e.style.opacity = '1');
  U.markLines(document, ''' + JS(HL4) + ''');
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 5 幕

N5 = {3: '三个 build_forward_expand 连在一起写：不让图重排，减少后端切分次数',
      14: '写：把本轮算出的 k_cur 按 k_idxs 摊进 KV cache（cpy_k 由 memory 上下文提供）',
      15: '写：v_cur 同理。索引表来自 ubatch（L2-04 / L2-05）',
      21: '读：get_k 取出的是"历史 + 本轮"的完整 K —— 图上的张量因此随上下文变长',
      24: '真正的注意力计算交给 build_attn_mha（下一幕）'}

HL5_ORDER = M(2875, N5, [2878])
HL5_WRITE = M(2875, N5, [2889, 2890])
HL5_READ = M(2875, N5, [2896, 2897])
HL5_MHA = M(2875, N5, [2899])
HL5 = M(2875, N5, [2878, 2889, 2890, 2896, 2897, 2899])

L.scene(
    kicker='L2-06 · 原语 · 注意力',
    title='<span class="hl-d">build_attn</span>：先写进 KV cache，再整段读回来',
    sub='Q/K/V 投影不在这一层（由 build_qkv 做）；这一层负责记账：把本轮的 K/V 写进缓存，再连历史一起读出来。',
    caption='2877 行的源码注释说：k 特意晚一点挂进图，好让 rope 融合后直接写进 KV cache。'
            'rope 本身不在 llama-graph.cpp 里 —— 各模型文件在调用 build_attn 之前用 ggml_rope_ext 处理 Q/K'
            '（见 src/models/ 下各架构文件，L2-10~L2-15）。',
    src=SRC_CPP, parts=[(2875, 2900)], duration=22000,
    notes=N5, marks=HL5,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip a">q_cur / k_cur / v_cur</span><span class="arrow">-&gt;</span>
    <span class="chip c">cpy_k / cpy_v</span><span class="arrow">-&gt;</span>
    <span class="chip b">k / v（含历史）</span><span class="arrow">-&gt;</span>
    <span class="chip d">build_attn_mha</span>
  </div>
  <div class="row" style="gap:9px">
    <div class="card" style="border-left-color:var(--c);flex:1 1 0">
      <div class="ct" style="color:var(--c)">写路径（本轮新增）</div>
      <div class="cb">mctx_cur-&gt;cpy_k(ctx0, k_cur, k_idxs, il) 与 cpy_v。<br>
      它们不是 ggml.h 里的算子，而是 <b>memory 上下文</b>给出的写算子 —— 接口见 L2-04。<br>
      索引表 k_idxs / v_idxs 由 build_attn_inp_kv 这类函数按 ubatch 算好。</div>
    </div>
    <div class="card" style="border-left-color:var(--b);flex:1 1 0">
      <div class="ct" style="color:var(--b)">读路径（历史 + 本轮）</div>
      <div class="cb">mctx_cur-&gt;get_k(ctx0, il) 返回整段 K 的<b>视图</b>，长度等于当前上下文。<br>
      所以同一份 build() 代码，prompt 阶段与 decode 阶段建出的图<b>形状不同</b> —— 这就是 L2-07 图复用的难点。</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  'Q/K/V 是上一层的产物（build_qkv，1619 行起）。这一幕只看 <span class="k">KV cache 的记账</span>。',
  '写：<span class="v">cpy_k</span> / <span class="v">cpy_v</span> 把本轮的 K/V 按索引表写进缓存。' +
    '它们由 memory 上下文提供，<span class="k">不是 ggml.h 里的算子</span>（L2-04）。',
  '读：<span class="v">get_k</span> / <span class="v">get_v</span> 返回的是<b>整段</b> K/V 的视图，' +
    '长度随上下文增长。',
  '回顾 L2-05：索引表来自 <span class="v">ubatch</span> 的切分；<br>' +
    '回顾 L2-04：缓存的物理布局、cell、slot 都在那一课。',
  '最后一行把注意力交给 <span class="v">build_attn_mha</span>，' +
    '再把输出经 <span class="v">wo</span> 投影回 n_embd。下一幕拆这个函数。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3800, () => { msg.innerHTML = texts[1]; U.markLines(document, ''' + JS(HL5_WRITE) + '''); });
tl.at(7600, () => { msg.innerHTML = texts[2]; U.markLines(document, ''' + JS(HL5_READ) + '''); });
tl.at(11200, () => { msg.innerHTML = texts[3]; U.markLines(document, ''' + JS(HL5_ORDER) + '''); });
tl.at(15000, () => { msg.innerHTML = texts[4]; U.markLines(document, ''' + JS(HL5_MHA) + '''); });
'''
)

# ------------------------------------------------------------------ 第 6 幕

N6 = {0: '两个条件：用户开了 flash attention，且这一层没有 KQ bias',
      1: '快路：整个注意力塌缩成一个节点；代价是 K 与 V 必须先转成 F16（看下面两处 ggml_cast）',
      17: '整个注意力 = 一个节点。mask、scale、alibi、softcap 全部塞进 op_params',
      19: '把它登记成"融合节点"；后续按名字回捞（例如投机解码要取中间结果）'}

HL6_FLAG = M(2626, N6, [2626])
HL6_FA = M(2626, N6, [2643])
HL6_NODE = M(2626, N6, [2645])
HL6_SLOW = M(2626, N6, [2627])
HL6 = M(2626, N6, [2626, 2627, 2643, 2645])

L.scene(
    kicker='L2-06 · 原语 · 注意力',
    title='<span class="hl-e">build_attn_mha</span>：快路与慢路',
    sub='cparams.flash_attn 有值且没有 KQ bias 时，整个注意力塌缩成一个 ggml_flash_attn_ext 节点。',
    caption='慢路在 2671 行起：ggml_mul_mat(k, q) 到 ggml_soft_max_ext 再到 ggml_mul_mat(v, kq)，'
            '是 L1-01 里"节点 + 边"的直接体现。',
    src=SRC_CPP, parts=[(2626, 2650)], duration=20000,
    notes=N6, marks=HL6,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="card" style="border-left-color:var(--b);flex:1 1 0">
      <div class="ct" style="color:var(--b)">快路 · 1 个节点</div>
      <div class="cb"><span class="cm" style="margin:0">ggml_flash_attn_ext(q, k, v, mask, scale, ...)</span><br>
      把 QK^T、缩放、mask、softmax、乘 V 全部融进<b>一个</b>节点。<br>
      代价：K/V 必须是 F16（2635 / 2639 行先 cast），所以多了两个类型转换节点。</div>
    </div>
    <div class="card" style="border-left-color:var(--c);flex:1 1 0">
      <div class="ct" style="color:var(--c)">慢路 · 4 个节点</div>
      <div class="cb"><span class="cm" style="margin:0">ggml_mul_mat(k, q)</span> 到 <span class="cm" style="margin:0">ggml_soft_max_ext</span> 再到 <span class="cm" style="margin:0">ggml_mul_mat(v, kq)</span>，最后 <span class="cm" style="margin:0">ggml_permute</span>。<br>
      KQ bias、softcap、MLA 的 v_mla 吸收都只在慢路里实现。</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  '分叉点只有一行：<span class="v">const bool use_flash_attn = cparams.flash_attn &amp;&amp; kq_b == nullptr;</span>',
  '快路：先按需 <span class="v">ggml_cast</span> 成 F16，然后<b>一行</b>算完整个注意力。',
  '<span class="k">同一个注意力，在图上可以是 1 个节点，也可以是 4 个节点。</span><br>' +
    '这是本视角下"算子粒度不固定"最清楚的例子 —— L5-01 起的内核课会看到两者各自的内核。',
  '<span class="v">res-&gt;add_fused_node(...)</span> 把融合节点登记下来：' +
    '投机解码、embedding 输出等场景要靠名字回捞这个节点。',
  '无论走哪条路，最后都 <span class="v">ggml_build_forward_expand(gf, cur)</span> 挂进图，' +
    '交给 L4-02 的调度器切给后端。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, ''' + JS(HL6_FLAG) + '''); });
tl.at(3900, () => { msg.innerHTML = texts[1]; U.markLines(document, ''' + JS(HL6_FA) + '''); });
tl.at(7500, () => { msg.innerHTML = texts[2]; U.markLines(document, []); });
tl.at(11200, () => { msg.innerHTML = texts[3]; U.markLines(document, ''' + JS(HL6_NODE) + '''); });
tl.at(14800, () => { msg.innerHTML = texts[4]; U.markLines(document, ''' + JS(HL6_SLOW) + '''); });
'''
)

# ------------------------------------------------------------------ 第 7 幕

N7 = {0: 'n_embd / n_tokens 从输入张量的形状直接读出来，不用再查 hparams',
      2: '只有 Llama4 例外：它把权重乘在 FFN 之前，所以这个分支要单独记一个标志',
      7: '路由打分就是一次普通矩阵乘：gate_inp 形状是 [n_expert, n_embd]',
      22: '门控函数四选一，由模型的 gating_op 决定',
      25: 'SOFTMAX：在整个专家维上归一化，权重之和恒为 1',
      29: 'SIGMOID：各专家独立打分，不做全局归一；选完 top-k 之后再归一',
      37: 'SQRT_SOFTPLUS：用 ggml_sqrt 套 ggml_softplus 拼出来的门控'}

HL7 = M(2018, N7, [2018, 2020, 2025, 2040, 2043, 2047, 2055])

L.scene(
    kicker='L2-06 · 核心 · MoE',
    title='<span class="hl-a">build_moe_ffn</span> 第一步：路由打分与门控',
    sub='MoE 的第一件事不是算专家，而是"每个 token 该去哪些专家" —— 一次矩阵乘 + 一个门控函数。',
    caption='n_expert 与 n_expert_used 都来自 hparams（见第 2 幕的 1471 / 1472 行）：'
            'n_expert 是路由宽度，n_expert_used 是每个 token 真正激活的专家数。',
    src=SRC_CPP, parts=[(2018, 2056)], duration=22000,
    notes=N7, marks=HL7,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '① 打分', m: 'build_lora_mm(gate_inp, cur)',
    b: 'gate_inp 形状 [n_expert, n_embd]，<br>输出 logits 形状 [n_expert, n_tokens]。' },
  { c: 'b', t: '② 门控 · SOFTMAX', m: 'ggml_soft_max(ctx0, logits)',
    b: '在整个专家维上归一化，<br>权重之和恒为 1。' },
  { c: 'c', t: '② 门控 · SIGMOID', m: 'ggml_sigmoid(ctx0, logits)',
    b: '各专家独立打分，不做全局归一；<br>选完 top-k 之后再归一。' },
  { c: 'd', t: '② 门控 · SOFTMAX_WEIGHT', m: 'probs = logits（不在这里算）',
    b: '推到取完 top-k 之后再 softmax，<br>只对选中的专家做。' },
  { c: 'e', t: '② 门控 · SQRT_SOFTPLUS', m: 'ggml_sqrt(ctx0, ggml_softplus(ctx0, logits))',
    b: '两个已有算子拼出来的门控，<br>不需要新增 ggml 算子。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  'MoE 层的输入和稠密层完全一样：<span class="v">cur = [n_embd, n_tokens]</span>。<br>' +
    '区别从第一行开始 —— 先问"该找谁"，而不是先算。',
  '<span class="v">logits = build_lora_mm(gate_inp, cur)</span>：<br>' +
    '一次 <span class="m">[n_expert, n_embd] x [n_embd, n_tokens]</span> 的矩阵乘，得到每个 token 对所有专家的打分。',
  '门控函数是<span class="k">四选一</span>：SOFTMAX / SIGMOID / SOFTMAX_WEIGHT / SQRT_SOFTPLUS。' +
    '注意最后一种是用 <span class="m">ggml_sqrt</span> 与 <span class="m">ggml_softplus</span> 拼出来的。',
  '还有一层 <span class="v">exp_probs_b</span>：源码注释说它是 DeepSeek V3 引入的<b>选择偏置</b>，<br>' +
    '只加给"选谁"，不加给"权重多少"（注释写得很明确）。',
  '所以路由这一步在图上是：<span class="k">1 次 mul_mat + 1 个门控算子</span>。' +
    '下一幕看选专家与权重。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[Math.min(i, 3)];
}));
tl.at(16000, () => {
  els.forEach(e => e.style.opacity = '1');
  U.markLines(document, ''' + JS(HL7) + ''');
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 8 幕

N8 = {3: 'argsort_top_k 同时完成"排序"和"取前 k"，输出的是索引而不是值',
      14: 'probs 补一个前置维，从 [n_expert, n_tokens] 变成 [1, n_expert, n_tokens]',
      17: 'get_rows 按索引取出每个 token 选中专家的权重：[1, n_expert_used, n_tokens]',
      28: 'norm_w 为真时要把选中的权重重新归一化 —— 否则 sigmoid 门控的权重和不为 1',
      31: 'sum_rows 把 n_expert_used 这一维求和，得到 [1, n_tokens]',
      35: '源码注释说明：下界 6.103515625e-5 是 F16 能表示的最小值，用来防除零',
      38: 'div：权重逐个除以和，完成归一化'}

HL8_TOPK = M(2106, N8, [2109])
HL8_GET = M(2106, N8, [2109, 2120, 2123])
HL8_ROWS = M(2106, N8, [2123])
HL8_NORM = M(2106, N8, [2134, 2137, 2141, 2144])
HL8_LAST = M(2106, N8, [2109, 2123, 2144])
HL8 = M(2106, N8, [2109, 2120, 2123, 2134, 2137, 2141, 2144])

L.scene(
    kicker='L2-06 · 核心 · MoE',
    title='<span class="hl-c">build_moe_ffn</span> 第二步：top-k 与权重归一',
    sub='选出 n_expert_used 个专家，取出它们的权重，再决定要不要重新归一化。',
    caption='argsort_top_k 的返回值是索引（I32），所以后面要 reshape 成 [1, n_expert, n_tokens] 才能用 get_rows 去取权重。',
    src=SRC_CPP, parts=[(2106, 2151)], duration=24000,
    notes=N8, marks=HL8,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip">probs [n_expert, n_tokens]</span><span class="arrow">-&gt;</span>
    <span class="chip a">argsort_top_k</span><span class="arrow">-&gt;</span>
    <span class="chip b">get_rows</span><span class="arrow">-&gt;</span>
    <span class="chip c">sum_rows / clamp / div</span><span class="arrow">-&gt;</span>
    <span class="chip d">weights [1, k, n_tokens]</span>
  </div>
  <div class="row" style="gap:9px">
    <div class="card" style="border-left-color:var(--a);flex:1 1 0">
      <div class="ct" style="color:var(--a)">选谁 · 索引</div>
      <div class="cb"><span class="cm" style="margin:0">ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)</span><br>
      selection_probs 可能是"加了选择偏置"的版本；而取权重时用的一定是<b>没加偏置</b>的 probs。<br>
      k = n_expert_used，来自 hparams（见第 2 幕 1472 行）。</div>
    </div>
    <div class="card" style="border-left-color:var(--c);flex:1 1 0">
      <div class="ct" style="color:var(--c)">多少权重 · 值</div>
      <div class="cb"><span class="cm" style="margin:0">ggml_get_rows(ctx0, probs, selected_experts)</span><br>
      权重只取被选中专家那一列，形状 [1, n_expert_used, n_tokens]；<br>
      归一化那一段只有在 norm_w 为真时才执行。</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  '四步流水线：<span class="k">选专家 -&gt; 取权重 -&gt; 归一 -&gt; 缩放</span>。',
  '<span class="v">ggml_argsort_top_k</span> 一次完成排序与前 k 截取，输出的是索引。',
  '<span class="v">ggml_get_rows</span> 用索引把权重捞出来，形状 <span class="m">[1, n_expert_used, n_tokens]</span>。',
  '<span class="v">norm_w</span> 为真时才做归一化：<span class="m">ggml_sum_rows</span> 求和、' +
    '<span class="m">ggml_clamp</span> 防除零、<span class="m">ggml_div</span> 相除。',
  '最后 <span class="v">ggml_scale</span> 做一次可选的 w_scale。<br>' +
    '到这里，每个 token 该去哪些专家、各占多少权重，就全变成图上的张量了。',
  '下一幕把这些权重真正乘到专家输出上。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, ''' + JS(HL8_TOPK) + '''); });
tl.at(3700, () => { msg.innerHTML = texts[1]; U.markLines(document, ''' + JS(HL8_GET) + '''); });
tl.at(7400, () => { msg.innerHTML = texts[2]; U.markLines(document, ''' + JS(HL8_ROWS) + '''); });
tl.at(11000, () => { msg.innerHTML = texts[3]; U.markLines(document, ''' + JS(HL8_NORM) + '''); });
tl.at(15500, () => { msg.innerHTML = texts[4]; U.markLines(document, []); });
tl.at(19500, () => { msg.innerHTML = texts[5]; U.markLines(document, ''' + JS(HL8_LAST) + '''); });
'''
)

# ------------------------------------------------------------------ 第 9 幕

N9 = {0: 'down 投影：第三个（也是最后一个）mul_mat_id，输出回到 n_embd',
      17: '加权：experts 与 weights 逐元素相乘，这一步把"路由"与"计算"接起来',
      33: '按索引把每个专家的结果切成 [n_embd, n_tokens] 的视图',
      42: '聚合：k 个专家的输出直接相加（权重已经在前面乘过了）',
      49: 'k = 1 时要 cont 一次，保证返回的是连续张量'}

HL9 = M(2304, N9, [2304, 2321, 2337, 2346, 2353])

L.scene(
    kicker='L2-06 · 核心 · MoE',
    title='★ <span class="hl-e">build_moe_ffn</span> 展开成的 ggml 算子清单',
    sub='这就是本课的验收点：一个 MoE 层 = 路由 1 次 + 门控 1 个 + top-k 1 个 + 三次 mul_mat_id + 聚合若干。',
    caption='表里的"行号"是 build_moe_ffn 里的调用点；算子本体由原语发出：'
            'build_lora_mm 在 1518 行调 ggml_mul_mat，build_lora_mm_id 在 1552 行调 ggml_mul_mat_id。'
            'ggml_mul_mat_id 一个算子就同时对所有选中专家做矩阵乘 —— 这才是"专家并行"在图上的样子。'
            '表里的算子名都取自 ggml/include/ggml.h。',
    src=SRC_CPP, parts=[(2304, 2356)], duration=28000,
    notes=N9, marks=HL9,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['阶段', '调用的原语', '展开成的 ggml 算子', '行号'],
  [['路由打分', 'build_lora_mm(gate_inp)', 'ggml_mul_mat', '2025'],
   ['门控（四选一）', 'softmax / sigmoid / sqrt + softplus', 'ggml_soft_max · ggml_sigmoid · ggml_softplus', '2043 · 2047 · 2055'],
   ['选专家', 'argsort_top_k', 'ggml_argsort_top_k', '2109'],
   ['取权重', 'reshape_3d + get_rows', 'ggml_reshape_3d · ggml_get_rows', '2120 · 2123'],
   ['权重归一', 'sum_rows + clamp + div', 'ggml_sum_rows · ggml_clamp · ggml_div', '2137 · 2141 · 2144'],
   ['gate / up 投影', 'build_lora_mm_id x2', 'ggml_mul_mat_id', '2190 · 2203'],
   ['GLU 激活', 'swiglu_split(gate, up)', 'ggml_swiglu_split', '2246'],
   ['down 投影', 'build_lora_mm_id', 'ggml_mul_mat_id', '2304'],
   ['加权', 'mul(experts, weights)', 'ggml_mul', '2321'],
   ['逐专家切开', 'view_2d(..., i * nb[1])', 'ggml_view_2d', '2337'],
   ['相加聚合', 'add(moe_out, cur_experts[i])', 'ggml_add', '2346']],
  { monoCols: [1, 2, 3] });
t.el.style.fontSize = '9.5px';
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '整张表就是 <span class="k">build_moe_ffn 一个函数</span>展开出来的东西 —— 11 步，7 种 ggml 算子。',
  '<span class="v">路由 + 门控 + top-k</span>（前 5 行）是"决定"，只跟 n_expert 有关，与专家权重无关。',
  '<span class="v">ggml_mul_mat_id</span>（第 6 / 8 行）是核心：一个算子，一次算完所有选中专家。<br>' +
    '它由 <span class="v">build_lora_mm_id</span> 发出，源码在 1552 行。',
  '<span class="v">加权 + 切开 + 相加</span>（后 3 行）把 k 份专家输出合回一份。<br>' +
    '这几个 <span class="v">ggml_view_2d</span> 是纯视图，不占新内存（L1-01 讲过 view_src）。',
  '所以验收点的答案是：<span class="k">mul_mat，soft_max 或 sigmoid 或 softplus，argsort_top_k，' +
    'reshape_3d，get_rows，sum_rows，clamp，div，mul_mat_id 三次，swiglu_split，mul，view_2d，add</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(4200, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i < 5 ? 'on' : ''); });
tl.at(9600, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = (i === 5 || i === 7) ? 'on' : ''); });
tl.at(15200, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i >= 8 ? 'on' : ''); });
tl.at(21500, () => { msg.innerHTML = texts[4]; rows.forEach(r => r.className = ''); });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L2-06 · 收束',
    title='★ 模型架构的差异 = <span class="hl-a">原语选择</span> + <span class="hl-b">超参</span>',
    sub='同一个 build_* 原语，靠枚举值与 hparams 分支出几十种架构 —— 这就是 156 个模型文件长得像的原因。',
    caption='下一课 L2-07 讲这张 gf 怎么被分配、复用、算出来；再往后 L4-02 讲它怎么被切给后端。',
    src=SRC_H, parts=[(1052, 1070)], duration=20000,
    marks=[1, 7, 13],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['原语', '展开成什么', '在哪一课展开'],
  [['build_lora_mm / build_lora_mm_id', 'ggml_mul_mat / ggml_mul_mat_id（+ 可选 LoRA、per-tensor scale）', 'L2-09 / L2-12'],
   ['build_norm', 'ggml_norm / ggml_rms_norm / ggml_group_norm（+ ggml_mul、ggml_add）', '本课 · 第 3 幕'],
   ['build_ffn', 'up / gate / down 三次 ggml_mul_mat + ggml_swiglu_split 等激活', 'L2-14 / L2-15'],
   ['build_moe_ffn', '上一幕那 11 步（7 种 ggml 算子）', '本课 · 第 9 幕，L2-12 / L2-13'],
   ['build_attn / build_attn_mha', 'ggml_flash_attn_ext 一个节点，或 mul_mat + soft_max_ext + mul_mat', 'L2-04 / 本课 · 第 6 幕'],
   ['build_inp_*', 'ggml_new_tensor + ggml_set_input 的占位输入，解码前填数据', 'L2-05 / L2-07']],
  { monoCols: [0, 1] });
t.el.style.fontSize = '9.5px';
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '不看代码，说出 <span class="mono">build_moe_ffn</span> 把一个 MoE 层展开成了哪些 ggml 算子？',
  '分四段：<br>' +
  '<b>① 路由</b>：<span class="mono">ggml_mul_mat</span>（gate_inp，2025 行），门控四选一 ' +
  '<span class="mono">ggml_soft_max</span>（2043）/ <span class="mono">ggml_sigmoid</span>（2047）/ ' +
  '<span class="mono">ggml_softplus</span>（2055）。<br>' +
  '<b>② 选专家与权重</b>：<span class="mono">ggml_argsort_top_k</span>（2109），' +
  '<span class="mono">ggml_reshape_3d</span>（2120）+ <span class="mono">ggml_get_rows</span>（2123），' +
  '归一化 <span class="mono">ggml_sum_rows</span>（2137）+ <span class="mono">ggml_clamp</span>（2141）+ ' +
  '<span class="mono">ggml_div</span>（2144）。<br>' +
  '<b>③ 专家并行</b>：<span class="mono">ggml_mul_mat_id</span> 三次（up 2190 / gate 2203 / down 2304，' +
  '由 <span class="mono">build_lora_mm_id</span> 在 1552 行发出），激活 ' +
  '<span class="mono">ggml_swiglu_split</span>（2246），加权 <span class="mono">ggml_mul</span>（2321）。<br>' +
  '<b>④ 聚合</b>：<span class="mono">ggml_view_2d</span>（2337）逐专家切开，' +
  '<span class="mono">ggml_add</span>（2346）相加，k 等于 1 时补一个 <span class="mono">ggml_cont</span>（2353）。<br>' +
  '另外每个中间张量都用 <span class="mono">ggml_build_forward_expand</span> 挂进 <span class="mono">gf</span>' +
  '（2155 / 2325 / 2339 / 2348 行）。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '把这一课压成一张表：<span class="k">六个原语族</span>覆盖了 Transformer 的全部结构。',
  '<span class="v">build_lora_mm</span> 与 <span class="v">build_norm</span> 是最底层两块砖：' +
    '一个管矩阵乘，一个管归一化。',
  '<span class="v">build_ffn</span> 与 <span class="v">build_moe_ffn</span> 是同一件事的稠密版与稀疏版 —— ' +
    '<span class="k">区别只在中间那层有几个矩阵乘、怎么选</span>。',
  '<span class="v">build_attn</span> 负责 KV cache 记账，<span class="v">build_inp_*</span> 负责占位输入。',
  '实测：<span class="m">src/models/</span> 下 156 个文件里，137 个出现这四族原语的调用，' +
    '57 个模型文件调用 <span class="v">build_moe_ffn</span>。<br>' +
    '<span class="k">所以"模型架构差异"到这一层就只剩两件事：调哪些原语、传什么超参。</span>',
  '下一课 L2-07：这张 <span class="v">gf</span> 被分配内存、被复用、被算出来。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、这一课在链条上的位置',
    '`src/llama-graph.{h,cpp}` 是"模型定义到 ggml 计算图"之间唯一的翻译层。'
    '`src/models/` 下 156 个文件里有 137 个不直接调用 `ggml_*`，而是调用这里的 `build_*` 原语。'
    '原语的产物只有两样：一个张量 arena（`ctx0`）和一张节点序列（`gf`）。',
    src=SRC_H, parts=[(1034, 1042)], lang='c')

L.section(
    '二、★ llm_graph_context：把超参摊平成常量的工作台',
    '构造函数 44 行，全是 `成员 (来源)` 的形式。右列决定了图的形状：\n\n'
    '| 来源 | 决定什么 | 例子 |\n|---|---|---|\n'
    '| `hparams` | 图的形状（结构） | `n_embd` `n_layer()` `n_head_kv()` `n_expert` |\n'
    '| `cparams` | 算子的参数（运行期） | `n_ctx` `rope_freq_base` `yarn_*` |\n'
    '| `ubatch` | 图的宽度 | `n_tokens` |\n'
    '| `res` | 产物 | `ctx0 = res->get_ctx()` `gf = res->get_gf()` |\n\n'
    '注意最后两行：`ctx0` 与 `gf` 不是新造的，而是从 `res` 里取的 —— '
    '图复用（L2-07）靠的就是这一点。',
    src=SRC_CPP, parts=[(1455, 1498)], lang='c')

L.section(
    '三、build_norm：三种归一化一个入口',
    '`llm_norm_type` 决定用 `ggml_norm`、`ggml_rms_norm` 还是 `ggml_group_norm`；'
    '两个 eps（`f_norm_eps` 与 `f_norm_rms_eps`）是不同的 hparams 字段。'
    '归一化之后的 `ggml_mul`（权重）几乎人人都有，`ggml_add`（偏置）只有部分架构有 —— '
    '**有没有偏置就是 RMSNorm 家族与 LayerNorm 家族在图上的分界**。',
    src=SRC_CPP, parts=[(1583, 1615)], lang='c')

L.section(
    '四、build_ffn：三次矩阵乘 + 一个激活',
    '`build_lora_mm` 负责 up / gate / down 三次投影（并顺带处理 LoRA 与 per-tensor scale）。'
    '`type_gate` 决定 gate 是接在 up 之后（`LLM_FFN_SEQ`）还是与 up 并行（`LLM_FFN_PAR`）；'
    '`type_op` 决定激活函数。最常用的 SILU + PAR 组合只用一行 `ggml_swiglu_split`。'
    '**同一个 `build_ffn`，不同架构只是换两个枚举值。**',
    src=SRC_CPP, parts=[(1786, 1835)], lang='c')

L.section(
    '五、build_attn：KV cache 的写与读',
    '这一层不做 Q/K/V 投影（那是 `build_qkv`，1619 行起），负责的是记账：'
    '用 `mctx_cur->cpy_k` 与 `cpy_v` 把本轮的 K/V 写进缓存，再用 `get_k` 与 `get_v` 把整段（历史 + 本轮）读回来。'
    '这两个函数由 memory 上下文提供，不是 `ggml.h` 里的算子 —— 接口定义见 L2-04。\n\n'
    '**rope 不在这个文件里**：各模型架构在调用 `build_attn` 之前用 `ggml_rope_ext` 处理 Q/K'
    '（见 `src/models/` 下各架构文件，L2-10~L2-15）。',
    src=SRC_CPP, parts=[(2875, 2900)], lang='c')

L.section(
    '六、build_attn_mha：一个节点还是四个节点',
    '`cparams.flash_attn` 为真且没有 KQ bias 时，整个注意力塌缩成一个 `ggml_flash_attn_ext` 节点'
    '（K 与 V 需要先 cast 成 F16）。慢路则展开成 `ggml_mul_mat` 加 `ggml_soft_max_ext` 再加 `ggml_mul_mat`。'
    '**同一个语义，算子粒度可以差四倍** —— 这是本视角下"算子粒度不固定"最清楚的例子。',
    src=SRC_CPP, parts=[(2626, 2650)], lang='c')

L.section(
    '七、★ build_moe_ffn：一个函数，11 步，7 种 ggml 算子',
    'MoE 层分四段展开（行号见第 9 幕的表）：路由打分、选专家与权重、专家并行、聚合。'
    '其中 `ggml_mul_mat_id` 是关键：一个算子同时对所有选中专家做矩阵乘，'
    '这就是"专家并行"在图上的样子。它由 `build_lora_mm_id`（1552 行）发出。',
    src=SRC_CPP, parts=[(2304, 2356)], lang='c')

L.section(
    '八、输入原语：只造占位张量，不算东西',
    '`build_inp_*` 一族不参与计算，只负责**造占位输入张量**并登记到 `res`：'
    'token id、位置、输出索引、attention scale 等。它们在建图时是空张量，'
    '由解码流程在计算前填真实数据（L2-07）。',
    src=SRC_H, parts=[(1164, 1173)], lang='c')

L.footnote_add('本课只引用 `src/llama-graph.cpp` 与 `src/llama-graph.h`，两者均计入覆盖率。')
L.footnote_add('文中提到的 `ggml_*` 算子名（`ggml_mul_mat_id` / `ggml_argsort_top_k` / `ggml_swiglu_split` 等）'
               '来自 `ggml/include/ggml.h`；该文件不在本课覆盖声明内，故不计入本课覆盖率。')
L.footnote_add('`src/models/` 的统计数字（156 个文件 / 137 个出现原语调用 / 57 个调用 build_moe_ffn）'
               '由 grep 实测得到，不是估计值。余下 19 个文件（视觉塔 clip.cpp、BERT 系、RWKV/Mamba 系等）'
               '在自己的 build() 里直接调 ggml，不经过本课的原语。')

L.prereqs('`L2-05`')

L.goal(
    '说出 `llm_graph_context` 的成员各自由 `hparams` / `cparams` / `ubatch` 中哪一类决定；',
    '解释 `build_norm` 如何用同一个入口分派 RMSNorm 与 LayerNorm，以及两者的 eps 来自不同字段；',
    '说明 `build_attn` 里 KV cache 的写路径与读路径分别由谁提供；',
    '**完整说出 `build_moe_ffn` 把一个 MoE 层展开成了哪些 ggml 算子**（本课验收点）；',
    '解释为什么"模型架构差异"在这套原语下只剩"调哪些原语 + 传什么超参"两个自由度。')

L.conclusion(
    '六个原语族 = Transformer 的全部结构',
    '| 原语 | 展开成什么 |\n|---|---|\n'
    '| `build_lora_mm` / `build_lora_mm_id` | `ggml_mul_mat` / `ggml_mul_mat_id` |\n'
    '| `build_norm` | `ggml_norm` / `ggml_rms_norm` / `ggml_group_norm`，再 `ggml_mul`（+`ggml_add`） |\n'
    '| `build_ffn` | up / gate / down 三次矩阵乘 + 激活（`ggml_swiglu_split` 等） |\n'
    '| `build_moe_ffn` | 路由 + top-k + 三次 `ggml_mul_mat_id` + 加权聚合 |\n'
    '| `build_attn` / `build_attn_mha` | KV cache 读写 + `ggml_flash_attn_ext`（或 4 节点慢路） |\n'
    '| `build_inp_*` | `ggml_new_tensor` + `ggml_set_input` 的占位输入 |')

L.conclusion(
    '★ build_moe_ffn 展开出的 ggml 算子（验收点）',
    '```text\n'
    '路由      ggml_mul_mat          (2025 调用点 / build_lora_mm 内 1518)\n'
    '门控      ggml_soft_max (2043) | ggml_sigmoid (2047) | ggml_softplus (2055)\n'
    '选专家    ggml_argsort_top_k                        (2109)\n'
    '取权重    ggml_reshape_3d (2120) + ggml_get_rows    (2123)\n'
    '归一化    ggml_sum_rows (2137) + ggml_clamp (2141) + ggml_div (2144)\n'
    '专家并行  ggml_mul_mat_id x3   (up 2190 / gate 2203 / down 2304；在 build_lora_mm_id 内 1552)\n'
    '激活      ggml_swiglu_split                         (2246)\n'
    '加权      ggml_mul                                 (2321)\n'
    '聚合      ggml_view_2d (2337) + ggml_add (2346)   [k = 1 时再加 ggml_cont (2353)]\n'
    '```')

L.conclusion(
    '★ 模型差异被压缩成"原语选择 + 超参"',
    '实测：`src/models/` 下 156 个文件里，137 个出现 `build_norm` / `build_ffn` / `build_attn` / '
    '`build_moe_ffn` 的调用，57 个模型文件调用 `build_moe_ffn`。\n\n'
    '原因在构造函数里就写明了：图的一切形状都由 `hparams`（结构）、`cparams`（运行参数）、'
    '`ubatch`（这一批的 token 数）三者决定，而原语把这些值翻译成算子。'
    '于是"新架构"往往只是换几个枚举值加几个超参，不需要新算子 —— '
    '这正是 L2-10~L2-15 六课能覆盖全部模型家族的前提。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
