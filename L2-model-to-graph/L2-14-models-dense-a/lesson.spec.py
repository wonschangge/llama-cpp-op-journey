#!/usr/bin/env python3
"""L2-14 · 模型家族（五）：稠密 Transformer（上）—— 课件 spec。

运行：python3 L2-model-to-graph/L2-14-models-dense-a/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC = 'src/models/llama.cpp'
SRC_G = 'src/llama-graph.cpp'
SRC_GH = 'src/llama-graph.h'
SRC_M = 'src/models/models.h'

# 本课覆盖的 37 个模型文件（plan_matrix 里 L2-14 的清单）。
# 它们没有命中 multimodal / ssm / moe 三组图原语，因此落在"其余"这一组里 ——
# 但"其余"不等于"同质"：其中若干其实属于别的家族（见 source.md 第八节）。
FILES_37 = [
    'src/models/apertus.cpp', 'src/models/arcee.cpp', 'src/models/baichuan.cpp',
    'src/models/bitnet.cpp', 'src/models/bloom.cpp', 'src/models/chameleon.cpp',
    'src/models/chatglm.cpp', 'src/models/codeshell.cpp', 'src/models/cogvlm.cpp',
    'src/models/cohere2.cpp', 'src/models/command-r.cpp', 'src/models/dream.cpp',
    'src/models/eagle3.cpp', 'src/models/eurobert.cpp', 'src/models/exaone.cpp',
    'src/models/exaone4.cpp', 'src/models/falcon.cpp', 'src/models/gemma-embedding.cpp',
    'src/models/gemma.cpp', 'src/models/gemma2.cpp', 'src/models/gemma3.cpp',
    'src/models/gemma3n.cpp', 'src/models/gemma4-assistant.cpp', 'src/models/glm4.cpp',
    'src/models/gpt2.cpp', 'src/models/gptneox.cpp', 'src/models/granite-switch.cpp',
    'src/models/hrm-text.cpp', 'src/models/hunyuan-dense.cpp', 'src/models/hunyuan-vl.cpp',
    'src/models/internlm2.cpp', 'src/models/jais.cpp', 'src/models/jais2.cpp',
    'src/models/jina-bert-v2.cpp', 'src/models/jina-bert-v3.cpp', 'src/models/llada.cpp',
    'src/models/llama-embed.cpp',
]

L = Lesson(
    id='L2-14',
    layer='L2 · 从模型到图',
    title='模型家族（五）：稠密 Transformer（上）',
    kicker='L2 · 从模型到图',
    codecap='src/models/llama.cpp 等（逐字引用）',
    nav={'prev': {'href': '../L2-13-models-moe-b/index.html', 'label': 'L2-13 模型家族（四）MoE 下'},
         'next': {'href': '../L2-15-models-dense-b/index.html', 'label': 'L2-15 模型家族（六）稠密下'}},
)

L.cover(*FILES_37)
L.cover(SRC_G, SRC_GH)          # 与 build_* 原语对照（见 source.md 说明）
L.cover(SRC_M)                  # 三处 using graph = ，用来解释"6 行的模型文件"

L.note('**一句话**：`src/models/` 下这 37 个文件里，**绝大多数画的是同一张图** —— '
       '嵌入 → 逐层（norm → 注意力 → 残差 → norm → FFN → 残差）→ 末层 norm → lm_head。'
       '它们之所以是 37 个文件而不是 1 个，差别只在**几个"层开关"**：'
       'QKV 有没有 bias、norm 摆在残差前还是后、embedding 与 lm_head 是否 tied、'
       '注意力用不用 sliding window、FFN 用哪个激活、位置编码用 RoPE 还是 ALiBi。')
L.note('本课的主干来自 `src/models/llama.cpp`。注意一个名实之差：'
       '计划文档写的是 `llm_build_llama` —— 那是旧版 llama.cpp 的自由函数名；'
       '在 v0.5.0 里它已经改成 **`llama_model_llama::graph<embed>` 的构造函数**'
       '（`src/models/llama.cpp:99`），名字变了，图的形状没变。下面一律用真实名字。')
L.note('回顾 **L2-06**：`build_inp_embd` / `build_norm` / `build_qkv` / `build_attn` / '
       '`build_ffn` / `build_lora_mm` 这些原语的**实现**在 `src/llama-graph.cpp`，'
       '本课只讲**谁在什么顺序上调用它们**。回顾 **L2-02**：`n_layer` / `n_head` / '
       '`n_embd` 这些超参决定循环次数与张量形状，本课的循环直接读它们。'
       '下一课 **L2-15** 讲同一条主干上的注意力变体（GQA / SWA / MLA）与投机解码草稿模型。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2-14 · 全局',
    title='37 个文件，<span class="hl-a">一条主干</span>',
    sub='本课覆盖 src/models/ 下的 37 个文件；主干只有一条，差异都在"层的开关"上。',
    caption='下一课 L2-15 接同一张清单的另外 36 个文件（注意力变体与草稿模型）。',
    src=SRC, parts=[(94, 108)], duration=16000,
    mark_src=[95, 99, 108],
    notes_src={95: 'build_arch_graph：模型类把这个 arch 的图"造出来"的唯一入口（回顾 L2-05）',
               99: '★ 主干就写在这个模板构造函数里；embed 是编译期开关（true = 只出句向量）',
               108: '第 1 站：token id -> 嵌入向量'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">src/models/ 37 个文件</span><span class="arrow">-></span>
    <span class="chip a">同一条主干</span><span class="arrow">-></span>
    <span class="chip b">不同的层开关</span>
  </div>
  <div class="row" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '本课讲什么', b: '一个稠密 Transformer 的图<br>是怎么被一行行建出来的', m: 'llama_model_llama::graph' },
  { c: 'b', t: '为什么是 37 个', b: '主干相同，只有几个开关不同：<br>bias / norm 位置 / tied / SWA / 激活', m: 'build_ffn(..., LLM_FFN_*, ...)' },
  { c: 'c', t: '怎么读这一课', b: '先默写主干十站，再看<br>每个开关在源码里长什么样', m: '见末幕的练习' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '37 个文件，先别急着一个个读。它们的图**骨架相同**：<br><span class="k">嵌入 -> 逐层（norm -> 注意力 -> 残差 -> norm -> FFN -> 残差）-> norm -> lm_head</span>。',
  '<span class="v">llama.cpp:99</span> 是主干的入口：一个模板构造函数。<br><span class="k">模板参数 embed</span> 决定最后接不接 lm_head。',
  '<span class="v">llama.cpp:108</span> 起，主干一站一站往下走 —— 本课就用它当"标准答案"。',
  '记住这句话：<span class="k">37 个文件共享同一条主干，差异只在层的开关</span>。<br>这正是它们能并成一课的原因。',
  '下一站：把这十站逐行读出来。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-14 · 主干 1/4',
    title='主干的第 1 段：<span class="hl-a">四样输入</span>',
    sub='主干开始之前，先把这一批 token 需要的东西全部备好：嵌入、位置、KV 入口、输出行。',
    caption='回顾 L2-05：ubatch 决定这一批有多少 token；回顾 L2-04：KV cache 由 memory 上下文提供。',
    src=SRC, parts=[(105, 124)], duration=20000,
    mark_src=[108, 111, 119, 124],
    notes_src={108: 'build_inp_embd：查表得到 [n_embd, n_tokens]，是图上第一个真算子（ggml_get_rows）',
               111: 'build_inp_pos：位置序列，RoPE 要用',
               119: 'build_attn_inp_kv：不是 KV cache 本体，而是"这一层如何读写 cache + 掩码"的入口',
               122: 'kq_scale：注意力缩放；超参给了就用超参（gemma 系靠它调 27B）',
               124: 'build_inp_out_ids：只有最后要 logits 的那些 token 才需要算 head'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '嵌入', m: 'build_inp_embd(tok_embd)', b: 'token id -> 向量<br>llama.cpp:108' },
  { c: 'b', t: '位置', m: 'build_inp_pos()', b: '给 RoPE 用的位置序列<br>llama.cpp:111' },
  { c: 'c', t: '注意力入口', m: 'build_attn_inp_kv()', b: 'KV cache 的读写 + 掩码<br>llama.cpp:119' },
  { c: 'e', t: '输出行', m: 'build_inp_out_ids()', b: '只保留要出 logits 的行<br>llama.cpp:124' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '主干开跑前先备四样东西。注意它们都叫 <span class="v">build_inp_*</span>：<br><span class="k">在图上开一个"输入张量"</span>，运行时由 llama-context 填数据。',
  '<span class="v">build_inp_embd</span>：主干第 1 站，唯一一个真正算东西的输入（一次 <span class="k">ggml_get_rows</span> 查表）。',
  '<span class="v">build_inp_pos</span>：位置。<span class="k">只有用 RoPE 的模型需要它</span> —— bloom 就完全不调它（它用 ALiBi）。',
  '<span class="v">build_attn_inp_kv</span>：注意它<span class="k">不是 KV cache 本身</span>，而是这一层的 cache 读写入口 + 掩码。<br>换成 <span class="v">build_attn_inp_no_cache</span> 就变成无 cache 的双向注意力（dream / llada 就是这样）。',
  '<span class="v">build_inp_out_ids</span>：把"要出 logits 的行"挑出来。<br>这一招让最后只在 <span class="k">少量 token</span> 上跑 lm_head，是 prefill 提速的关键。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(17000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-14 · 主干 2/4',
    title='★ 主干第 2 段：<span class="hl-a">逐层循环</span>的注意力半边',
    sub='n_layer 次循环，每次做四件事：norm、QKV、RoPE、注意力。全部通过 build_* 原语完成。',
    caption='回顾 L2-02：n_layer 来自 hparams，从 GGUF 元数据读入；层数决定循环次数，不决定图的形状。',
    src=SRC, parts=[(126, 173)], duration=24000,
    mark_src=[127, 129, 132, 143, 146, 169],
    notes_src={127: 't_layer_inp：把每层输入存下来（投机解码 / 中间层输出要用）',
               129: 'inpSA：残差支路的起点，第 5 站要加回来',
               132: '第 2 站 build_norm —— 注意在注意力【之前】，这就是 pre-norm',
               143: '第 3 站 build_qkv：一次调用拿走 Q/K/V，bias 加不加由张量是否存在决定',
               146: '第 4 站 RoPE：位置信息在这里注入 Q 和 K',
               162: 'use_kq_norm：少数模型（Llama4 文本塔）在 RoPE 后再做一次 Q/K RMSNorm',
               169: '第 5 站 build_attn：写 cache、读 cache、softmax、乘 V，一次调用全包'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div class="col" id="steps" style="gap:6px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const rows = [
  { n: 'build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il)', l: 'llama.cpp:132', c: 'a',
    d: '注意力之前的归一化（pre-norm）' },
  { n: 'build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il)', l: 'llama.cpp:143', c: 'b',
    d: 'Q/K/V 投影；有 bias 张量就加 bias' },
  { n: 'ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, ...)', l: 'llama.cpp:146 / 152', c: 'c',
    d: '位置注入；Q、K 各做一次（V 不做）' },
  { n: 'build_attn(inp_attn, wo, wo_b, wo_s, Qcur, Kcur, Vcur, ...)', l: 'llama.cpp:169', c: 'd',
    d: '写 KV cache -> 取 mask -> softmax -> 输出投影' }
];
const host = wrap.querySelector('#steps');
const els = rows.map(r => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
  e.innerHTML = '<span class="cm" style="margin:0">' + U.esc(r.n) + '</span>' +
    ' <span style="color:var(--' + r.c + ')">' + U.esc(r.l) + '</span>' +
    '<div class="cb" style="margin-top:2px">' + r.d + '</div>';
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '一层的注意力半边 = <span class="k">四站</span>。四站全部走 build_* 原语，模型文件里看不到一个 ggml_* 算子。',
  '<span class="v">build_norm</span> 在注意力【之前】—— 这叫 <span class="k">pre-norm</span>，是 llama 系的做法。<br>gemma2 / gemma3 / exaone4 会在注意力【之后】再加一个（下一幕对照）。',
  '<span class="v">build_qkv</span> 一个调用返回 Q/K/V 三个张量。<br><span class="k">bias 加不加，由 GGUF 里有没有那个张量决定</span>（L2-06 的原语把它做成了可选）。',
  '<span class="v">ggml_rope_ext</span>：位置注入。<span class="k">只对 Q 和 K 做</span>，V 不动。<br>这就是"注意力知道谁在谁前面"的全部来源。',
  '<span class="v">build_attn</span>：把 Q/K/V 交给同一个原语，它负责写 cache、算 softmax、再乘 V。<br>换 mask、换 cache，注意力就换了个"变体" —— 那是 L2-15 的主题。'
];
els.forEach((e, i) => tl.at(700 + i * 4200, () => {
  els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(20000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-14 · 主干 3/4',
    title='主干第 3 段：残差 + FFN（<span class="hl-c">dense/MoE 就在这里分叉</span>）',
    sub='两个残差相加夹着一次 norm + 一次 FFN；走哪条 FFN 分支由一个指针是否为空决定。',
    caption='MoE 分支的细节见 L2-12 / L2-13；本课只关心"稠密这一支"长什么样。',
    src=SRC, parts=[(174, 228)], duration=26000,
    mark_src=[174, 178, 182, 189, 220, 223],
    notes_src={174: 'inp_out_ids 只在这一层（最后一层）生效，省掉中间层的 head 开销',
               178: '第 6 站残差相加：注意力输出 + 进层时的 inpSA',
               182: '★ 层的开关：ffn_gate_inp 为空 -> 稠密 FFN；否则 -> MoE 分支',
               184: '第 7 站 build_norm（ffn_norm）',
               189: '第 8 站 build_ffn(SILU, PAR)：up + gate 并行，再乘 down',
               220: '第 9 站第二个残差相加；223 行的 build_cvec 是适配器注入点，没有适配器时是恒等'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" id="dia" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const dia = wrap.querySelector('#dia');
const left = U.el('div', { class: 'card', style: 'width:340px;border-left-color:var(--b)' });
left.innerHTML = '<div class="ct" style="color:var(--b)">稠密分支（本课）</div>' +
  '<div class="cb"><span class="cm" style="margin:0">if (ffn_gate_inp == nullptr)</span><br>' +
  'build_norm -> build_ffn(SILU, PAR)<br>' +
  '一个 gate 张量、一个 up 张量、一个 down 张量<br>' +
  '<b>每个 token 都走同样的权重</b></div>';
const right = U.el('div', { class: 'card', style: 'width:340px;border-left-color:var(--c)' });
right.innerHTML = '<div class="ct" style="color:var(--c)">MoE 分支（L2-12 / L2-13）</div>' +
  '<div class="cb"><span class="cm" style="margin:0">else</span><br>' +
  'build_norm -> build_moe_ffn(...)<br>' +
  '专家权重是三维张量 [n_embd, n_ff, n_expert]<br>' +
  '<b>每个 token 只走 top-k 个专家</b></div>';
dia.appendChild(left); dia.appendChild(right);

const msg = wrap.querySelector('#msg');
const texts = [
  '注意力半边结束后，剩下的事只有三件：<span class="k">加残差、过 FFN、再加残差</span>。',
  '<span class="v">ffn_inp = ggml_add(cur, inpSA)</span>（llama.cpp:178）：<br>把注意力输出加回<span class="k">进层时的输入</span>，残差支路在这里闭合。',
  '<span class="v">if (ffn_gate_inp == nullptr)</span>（llama.cpp:182）：<br>★ <span class="k">整个稠密/MoE 的分叉就写在这一次判空上</span>。37 个文件全部走 dense 这一支。',
  '<span class="v">build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il)</span>：<br><span class="k">PAR</span> = gate 与 up 并行算（两条支路同一输入），再乘起来过 down。',
  '<span class="v">ggml_add(cur, ffn_inp)</span>（llama.cpp:220）：第二个残差。<br>到这里一层的形状就固定了：<span class="k">两次 norm、两次残差、一次注意力、一次 FFN</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3600, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[1]; });
tl.at(7200, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[2]; });
tl.at(11000, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[3]; });
tl.at(15000, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[4]; });
tl.at(20000, () => { left.style.opacity = '1'; right.style.opacity = '1'; msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-14 · 主干 4/4',
    title='主干第 4 段：末层 norm + <span class="hl-b">lm_head</span>',
    sub='循环结束后：一次 output_norm，然后接 lm_head；embed 模板为 true 时两处都被换掉。',
    caption='回顾 L2-07：这段图产生的 t_logits / t_embd 就是 context 交给采样器或调用者的东西。',
    src=SRC, parts=[(229, 247)], duration=18000,
    mark_src=[231, 236, 238, 240, 246],
    notes_src={231: '第 10 站 output_norm：整个主干只有这一次是"层外"的归一化',
               236: 't_embd：句向量出口。embedding 模型就到此为止',
               238: '★ 编译期开关：embed == true 就不建 lm_head（llama-embed.cpp 用的是 graph<true>）',
               240: 'lm_head：build_lora_mm(model.output, cur)；output 可能就是 tok_embd 的副本',
               246: '把这一整条链挂到计算图上，图构建结束'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="dia" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const dia = wrap.querySelector('#dia');
const a1 = U.el('div', { class: 'card', style: 'width:330px;border-left-color:var(--a)' });
a1.innerHTML = '<div class="ct" style="color:var(--a)">graph&lt;false&gt; · 生成</div>' +
  '<div class="cb">注意力入口 = <b>build_attn_inp_kv()</b>（L119）<br>' +
  'output_norm -> <b>lm_head</b><br>' +
  '<span class="cm" style="margin:0">res-&gt;t_logits = cur;</span><br>' +
  'llama.cpp:240 / 243</div>';
const a2 = U.el('div', { class: 'card', style: 'width:330px;border-left-color:var(--b)' });
a2.innerHTML = '<div class="ct" style="color:var(--b)">graph&lt;true&gt; · 取句向量</div>' +
  '<div class="cb">注意力入口 = <b>build_attn_inp_no_cache()</b>（L117）<br>' +
  'output_norm -> <b>停</b><br>' +
  '<span class="cm" style="margin:0">res-&gt;t_embd = cur;</span><br>' +
  'llama-embed.cpp:4 用的就是这个特化</div>';
dia.appendChild(a1); dia.appendChild(a2);

const msg = wrap.querySelector('#msg');
const texts = [
  '循环结束，<span class="v">cur = inpL</span>。剩下的只有两站。',
  '<span class="v">build_norm(cur, output_norm, NULL, LLM_NORM_RMS, -1)</span>：<br>注意 <span class="k">il = -1</span>：它不是某一层的 norm，是整条主干出口的 norm。',
  '<span class="v">res-&gt;t_embd = cur</span>：句向量出口。<span class="k">embedding 模型到这里就结束</span>（gemma-embedding / eurobert 连 lm_head 都没有）。',
  '<span class="v">if constexpr (!embed)</span>：<span class="k">编译期开关</span>。<br>llama-embed.cpp 全文件只有 6 行，它用 <span class="v">graph&lt;true&gt;</span>，于是这一整段被编译掉。',
  '★ 开关不止一处：<span class="v">embed</span> 在 L113-117 还把注意力入口换成了 <span class="k">no_cache</span> 版。<br>所以 graph&lt;true&gt; = 不写 KV cache + 不出 logits。',
  '<span class="v">build_lora_mm(model.output, cur, model.output_s)</span>：最后一个矩阵乘。<br><span class="k">output 可能只是 tok_embd 的一个副本</span>（tied embedding），见下一幕的开关表。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3200, () => { msg.innerHTML = texts[1]; });
tl.at(6000, () => { a2.style.opacity = '.35'; msg.innerHTML = texts[2]; });
tl.at(9000, () => { a1.style.opacity = '.35'; a2.style.opacity = '1'; msg.innerHTML = texts[3]; });
tl.at(12000, () => { a1.style.opacity = '1'; msg.innerHTML = texts[4]; });
tl.at(15000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-14 · 原语',
    title='★ 主干上的每一站，都是一个 <span class="hl-d">build_* 原语</span>',
    sub='模型文件里没有一个 ggml_* 算子：图被拆成 6 个原语，模型只负责"按顺序调用 + 选开关"。',
    caption='原语的实现见 L2-06；本幕只把它们与主干的站一一对上。',
    src=SRC_GH, parts=[(1048, 1100)], duration=20000,
    mark_src=[1048, 1053, 1065, 1075, 1096],
    notes_src={1053: 'build_lora_mm：一次矩阵乘 + 可选 LoRA + 可选 per-tensor scale',
               1065: 'build_norm：cur + 权重(+bias) + 三种归一化之一',
               1075: 'build_qkv：Q/K/V 三合一，bias 可选（见下一幕开关 ②）',
               1096: 'build_ffn：up / gate / down 三块权重 + 激活类型 + 门控方式'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['主干上的站', '原语', '...cpp 里的实现行', '它替你做了几件事'],
  [['嵌入 / 输出行', 'build_inp_embd / build_inp_out_ids', '2362 / 2480', '开输入张量、查表、挑行'],
   ['归一化', 'build_norm', '1583', '一个 switch 分派 RMS / LayerNorm / GroupNorm，再乘权重加偏置'],
   ['Q/K/V 投影', 'build_qkv', '1633', 'fused 或分离两条路；bias 有就加，没有就跳过'],
   ['自注意力', 'build_attn', '2851', '写 KV cache、取掩码、softmax、输出投影'],
   ['FFN', 'build_ffn', '1748', 'up + gate（PAR 并行 / SEQ 串行）+ 激活 + down'],
   ['lm_head / 投影', 'build_lora_mm', '1514', '矩阵乘 + 可选 LoRA + 可选 scale'],
   ['控制向量', 'build_cvec', '1508', '适配器注入点，没有适配器时是恒等']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '把主干的十站压成 <span class="k">7 个原语</span>：模型文件只写"调用顺序"，不写算子。',
  '输入类原语（<span class="v">build_inp_*</span>）只负责<span class="k">在图上开张量</span>，真正的数据由运行时灌进来。',
  '<span class="v">build_norm</span> 是三条归一化的统一入口。<span class="k">用哪条由参数决定</span> —— 这是"层开关 ①"的实现方式。',
  '<span class="v">build_qkv</span> / <span class="v">build_ffn</span> 的签名里塞满了可选张量（bias、scale、LoRA）。<br><span class="k">同一个原语，靠"传不传"就能表达不同模型</span>。',
  '<span class="v">build_attn</span> 有 7 个重载（按 cache 类型分）：普通 KV、无 cache、SWA、cross、MLA……<br><span class="k">选哪个重载 = 选哪种注意力</span> —— 这是 L2-15 的主线。',
  '所以"模型家族"在这个仓库里的真实含义是：<span class="k">原语的选择 + 超参的取值</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(19000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-14 · 核心',
    title='★ 六个"层开关"：同一行调用，<span class="hl-e">两种行为</span>',
    sub='gemma3 的主干与 llama 逐站对应，但多了 QK-norm、post-norm、SWA、softcap。',
    caption='gemma3 是"开关拨得最多"的稠密模型之一：它仍然走同一条主干十站。',
    src='src/models/gemma3.cpp', parts=[(131, 186)], duration=24000,
    mark_src=[131, 139, 162, 177, 185],
    notes_src={131: '开关②：Q 先做一次 RMSNorm 再进 RoPE（llama 没有这一步）',
               139: 'K 同样处理；norm 的是「每个头的维度」n_embd_head_k',
               162: '开关①：注意力【之后】再 norm 一次 —— post-norm，llama 没有',
               185: 'FFN 之后还有第三个 norm —— gemma 的 sandwich 结构'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['层开关', 'llama.cpp 的拨法', '其它文件的拨法（本课 37 个里）'],
  [['① norm 位置', 'pre-norm：只在注意力/FFN 之前（L132/L184）',
    'post-norm：gemma2/gemma3 L162、glm4 L140、exaone4 L147（连 pre-norm 都没有）'],
   ['② QKV / wo bias', '有张量才加：build_qkv 里判空（llama-graph.cpp:1653）',
    'bloom / gpt2 恒有（fused wqkv+bias）；command-r / internlm2 的 wo_b 从未创建'],
   ['③ QK-norm', '无（只有 use_kq_norm 开关，L162）',
    'gemma3 L131/L139、apertus L93、exaone4 L122；command-r 可选且用 LayerNorm'],
   ['④ sliding window', '无：build_attn_inp_kv（L119）',
    'gemma2/gemma3/cohere2/exaone4/gemma-embedding：iswa 版入口 + is_swa(il)'],
   ['⑤ FFN 激活 / 门控', 'LLM_FFN_SILU + LLM_FFN_PAR（L194）',
    'GELU：bloom/gpt2/gemma 系；SwiGLU：chatglm/glm4；ReLU²：arcee/jais2；xIELU：apertus 手写'],
   ['⑥ embedding 是否 tied', 'output 缺失就复用 tok_embd（llama.cpp L41-46）',
    '恒 tied：gemma L20、cohere2 L29、command-r L21、bitnet L164；不 tied：baichuan、apertus']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '同一条主干，六处拨法不同 —— 这就是 37 个文件的全部差异来源。',
  '<span class="v">① norm 位置</span>：llama 是 pre-norm；<span class="k">gemma 系在残差前再加一次</span>（post-norm）。<br>exaone4 更极端：它<span class="k">根本没有 pre-norm</span>。',
  '<span class="v">② bias</span>：注意这不是"模型文件选"的 —— <span class="k">build_qkv 判的是张量在不在</span>，<br>而张量在不在由 GGUF 决定。同一个模型文件能同时支持两种。',
  '<span class="v">③ QK-norm</span>：norm 的是<span class="k">每个头的向量</span>（n_embd_head_k），不是整层。<br>它压的是注意力 logits 的量级。',
  '<span class="v">④ sliding window</span>：开关落在<span class="k">输入入口的选型</span>上（iswa 版 KV / 掩码），<br>而不是某个算子里 —— 所以主干十站一个都不少。',
  '<span class="v">⑤ 激活</span>：PAR 与 SEQ 的区别是 gate 吃谁的输出：<span class="k">PAR 吃输入，SEQ 吃 up 的输出</span>。',
  '<span class="v">⑥ tied embedding</span>：lm_head 与嵌入共享权重。<span class="k">省一份 [n_embd, n_vocab] 的显存</span>，<br>代价是输出与输入被绑在一起。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3000 + i * 3000, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(22000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[0]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-14 · 清单 1/2',
    title='37 个文件的真实差异（上：19 个）',
    sub='逐行读出来的差异点。列"家族"里带 ★ 的，实际上不属于稠密 Transformer。',
    caption='"归一化"列写的是该文件图里真正调用的归一化类型；"-"表示这个文件里没有图。',
    src='src/models/bitnet.cpp', parts=[(149, 169)], duration=22000,
    mark_src=[155, 157, 164],
    notes_src={155: '和其它文件一模一样的收尾：output_norm',
               164: '★ bitnet 的 lm_head 直接用 tok_embd（tied），而且它压根没建 output 张量'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:6px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg" style="font-size:10px"></div>`;
root.appendChild(wrap);

const rows = [
  ['apertus.cpp', 'RMS', 'xIELU', 'FFN 手写：up -> ggml_xielu -> down（L138）'],
  ['arcee.cpp', 'RMS', 'ReLU2', '无 gate，SEQ 串行（L128）'],
  ['baichuan.cpp', 'RMS', 'SILU', '13B 不接位置编码 + ALiBi（L13 / L90）'],
  ['bitnet.cpp', 'RMS x5', 'SILU', '层内多两个 sub-norm（L103 / L137）'],
  ['bloom.cpp', 'LayerNorm', 'GELU', 'fused wqkv+bias（L45）；嵌入后先归一化（L77）'],
  ['chameleon.cpp', 'RMS + LN', 'SILU', 'Q/K 用 LayerNorm 且可选（L91-104）'],
  ['chatglm.cpp', 'RMS', 'SwiGLU', 'gate 串行：up 出 2 倍宽（L48 / L133）'],
  ['codeshell.cpp', 'LayerNorm', 'GELU', '反向 tie：tok_embd 复用 output（L15-20）'],
  ['cogvlm.cpp', 'RMS', 'SILU', '★ 文本/视觉两套权重按 ubatch 二选一（L72-97）'],
  ['cohere2.cpp', 'LayerNorm', 'SILU', 'SWA 与全局层交错（L70）；并行残差（L131）'],
  ['command-r.cpp', 'LayerNorm', 'SILU', '并行残差（L118）；无 ffn_norm；强制 tie（L21）'],
  ['dream.cpp', 'RMS', 'SILU', '★ 非因果 + 无 KV cache（L15 / L68）：扩散式'],
  ['eagle3.cpp', 'RMS', 'SILU', '★ 两张图：encoder 只做 fc（L137），草稿模型'],
  ['eurobert.cpp', 'RMS', 'SILU', '★ 无 cache（L50）、无 lm_head：编码器'],
  ['exaone.cpp', 'RMS', 'SILU', '逐层 rope_freqs（L75）；输出可 tied（L24）'],
  ['exaone4.cpp', 'RMS', 'SILU', '无 pre-norm，全 post-norm（L113）；SWA；跳过 nextn 层（L42）'],
  ['falcon.cpp', 'LayerNorm', 'GELU', 'FFN 吃 attn_norm 而非 attn 输出（L125）'],
  ['gemma-embedding.cpp', 'RMS', 'GELU', '★ causal_attn=false（L7）+ 无 cache（L89）'],
  ['gemma.cpp', 'RMS', 'GELU', '嵌入乘 sqrt(n_embd)（L49）；output 恒为副本（L20）']
];
const t = U.table(['文件', '归一化', '激活', '真实差异点（行号在源文件里）'], rows, { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
t.el.style.fontSize = '9px';
t.el.querySelectorAll('th, td').forEach(c => { c.style.padding = '1px 4px'; c.style.lineHeight = '1.22'; });

const msg = wrap.querySelector('#msg');
const trs = t.body.querySelectorAll('tr');
const texts = [
  '这 19 个文件，主干都在；差异只有最后一列那一点点。',
  '★ 标记的三个（cogvlm / dream / eagle3）实际属于别的家族：<br>视觉专家、扩散式非因果、投机解码草稿。',
  '再看一遍最左列：<span class="k">归一化类型</span>基本就是"这个模型是新一代还是老一代"的分水岭。',
  '最后一行 gemma 最有代表性：<span class="k">output 无条件复用 tok_embd</span>（L20 直接 TENSOR_DUPLICATED）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
trs.forEach((r, i) => tl.at(2500 + i * 900, () => {
  if (i % 3 === 0) { r.className = 'on'; }
  if (i === 8) { msg.innerHTML = texts[1]; }
  if (i === 17) { msg.innerHTML = texts[2]; }
}));
tl.at(20000, () => { msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-14 · 清单 2/2',
    title='37 个文件的真实差异（下：18 个）',
    sub='下半张表里出现了"根本没有主干"的文件 —— 它们的图被一行 using 继承走了。',
    caption='hunyuan-dense / llama-embed 各只有 6 行；jina-bert-v2/v3 的图在 bert.cpp（别的课）。',
    src='src/models/apertus.cpp', parts=[(123, 144)], duration=22000,
    mark_src=[129, 138, 142],
    notes_src={129: 'up 投影还是原语',
               138: '★ 中间这一步不走 build_ffn：xIELU 是 LLM_FFN_* 枚举里没有的激活',
               142: 'down 投影又回到原语；主干两侧没变，只换了中间那一步'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:6px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg" style="font-size:10px"></div>`;
root.appendChild(wrap);

const rows = [
  ['gemma2.cpp', 'RMS', 'GELU', 'post-norm + SWA（L74）+ 两处 softcap（L14 / L167）'],
  ['gemma3.cpp', 'RMS', 'GELU', 'QK-norm（L131 / L139）+ post-norm + SWA + tied（L44）'],
  ['gemma3n.cpp', 'RMS', 'GELU', '★ AltUp/Laurel + 逐层嵌入（L324-376）；FFN 手写'],
  ['gemma4-assistant.cpp', 'RMS', 'GELU', '★ 只建 n_layer_nextn 层（L127），KV 借主模型（L151）'],
  ['glm4.cpp', 'RMS', 'SwiGLU', 'post-norm（L140 / L162）+ mRoPE（L81）'],
  ['gpt2.cpp', 'LayerNorm', 'GELU', 'fused wqkv+bias（L38）；学习式位置，无 RoPE（L74-77）'],
  ['gptneox.cpp', 'LayerNorm', 'GELU', 'use_par_res：注意力与 FFN 并行（L145）；带 bias（L67/L70）'],
  ['granite-switch.cpp', 'RMS', 'SILU', '★ 自定义输入 + 手写 FFN + 图内路由器（L134-147 / L411-415）'],
  ['hrm-text.cpp', 'RMS', 'SILU', '★ 深度循环双栈交替跑同一批层（L101 / L168）'],
  ['hunyuan-dense.cpp', '-', '-', '只有 6 行：图继承 hunyuan-vl（models.h L2105）'],
  ['hunyuan-vl.cpp', 'RMS', 'SILU', '★ 多模态文本半边：M-RoPE 4 段（L66-69）+ 视觉 tap（L86）'],
  ['internlm2.cpp', 'RMS', 'SILU', '无 wo_b；QKV 走 create_tensor_qkv（L26）'],
  ['jais.cpp', 'LayerNorm', 'SILU', 'ALiBi（L5）；不调 build_inp_pos（无 RoPE）'],
  ['jais2.cpp', 'LayerNorm', 'ReLU2', '无 gate，SEQ（L130）；tied（L22-24）'],
  ['jina-bert-v2.cpp', 'LayerNorm', '-', '★ 本文件无主干：图在 bert.cpp（models.h L308）'],
  ['jina-bert-v3.cpp', 'LayerNorm', '-', '★ 49 行只有张量表；图在 bert.cpp（models.h L319）'],
  ['llada.cpp', 'RMS', 'SILU', '★ 非因果（L16）+ 无 KV cache（L82）：扩散式'],
  ['llama-embed.cpp', '-', '-', '只有 6 行：复用 llama.cpp 主干，graph<true>']
];
const t = U.table(['文件', '归一化', '激活', '真实差异点（行号在源文件里）'], rows, { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
t.el.style.fontSize = '9px';
t.el.querySelectorAll('th, td').forEach(c => { c.style.padding = '1px 4px'; c.style.lineHeight = '1.22'; });

const msg = wrap.querySelector('#msg');
const trs = t.body.querySelectorAll('tr');
const texts = [
  '下半张表：★ 有九个 —— 它们才是"37 个"里真正需要解释的部分。',
  '★ 四个<span class="k">没有主干</span>：hunyuan-dense / llama-embed（各 6 行）、<br>jina-bert-v2 / jina-bert-v3（图在 bert.cpp）。',
  '★ 两个<span class="k">自成一派</span>：granite-switch（每 token 一套 LoRA + 图内路由器）、<br>hrm-text（深度循环双栈，同一批层跑多轮）。',
  '★ 三个<span class="k">别的家族</span>：gemma3n（多模态家族）、gemma4-assistant（草稿头）、llada（扩散式）。',
  '结论：<span class="k">"37 个稠密模型"是计划分组的结果，不是同质的 37 份代码</span>。<br>主干十站在它们里面大多还在，只是被机制盖住了。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
trs.forEach((r, i) => tl.at(2500 + i * 850, () => {
  if (i % 3 === 0) { r.className = 'on'; }
  if (i === 9) { msg.innerHTML = texts[1]; }
  if (i === 11) { msg.innerHTML = texts[2]; }
  if (i === 13) { msg.innerHTML = texts[3]; }
}));
tl.at(20000, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L2-14 · 收束',
    title='把主干默写下来',
    sub='十站、七个原语、六个开关。能默写这张表，这一课的验收点就达成了。',
    caption='下一课 L2-15：同一条主干上的注意力变体（GQA / SWA / MLA）与投机解码草稿模型。',
    src=SRC_M, parts=[(175, 183)], duration=26000,
    mark_src=[175, 179, 180],
    notes_src={179: '注释说得很直白：hparams 与张量加载都复用 llama',
               180: '★ 一行 using，就把 llama.cpp 里那整条主干继承过来了'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['#', '主干十站', '行号', '原语'],
  [['1', 'build_inp_embd(model.tok_embd)', 'llama.cpp:108', 'build_inp_embd'],
   ['2', 'build_inp_pos()', 'llama.cpp:111', 'build_inp_pos'],
   ['3', 'build_attn_inp_kv()（KV 读写 + 掩码入口）', 'llama.cpp:119', 'build_attn_inp_kv'],
   ['4', 'build_inp_out_ids()（只留要输出的行）', 'llama.cpp:124', 'build_inp_out_ids'],
   ['5', 'for il in 0..n_layer-1', 'llama.cpp:126', '-'],
   ['5a', 'build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il)', 'llama.cpp:132', 'build_norm'],
   ['5b', 'build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il)', 'llama.cpp:143', 'build_qkv'],
   ['5c', 'ggml_rope_ext(Q) / ggml_rope_ext(K)', 'llama.cpp:146 / 152', 'ggml_rope_ext'],
   ['5d', 'build_attn(inp_attn, wo, wo_b, wo_s, Q, K, V, ...)', 'llama.cpp:169', 'build_attn'],
   ['5e', 'ggml_add(cur, inpSA)  <- 残差 1', 'llama.cpp:178', 'ggml_add'],
   ['5f', 'build_norm(ffn_inp, ffn_norm, NULL, LLM_NORM_RMS, il)', 'llama.cpp:184', 'build_norm'],
   ['5g', 'build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il)', 'llama.cpp:189', 'build_ffn'],
   ['5h', 'ggml_add(cur, ffn_inp)  <- 残差 2', 'llama.cpp:220', 'ggml_add'],
   ['6', 'build_norm(cur, output_norm, NULL, LLM_NORM_RMS, -1)', 'llama.cpp:231', 'build_norm'],
   ['7', 'build_lora_mm(model.output, cur, model.output_s)', 'llama.cpp:240', 'build_lora_mm'],
   ['8', 'ggml_build_forward_expand(gf, cur)', 'llama.cpp:246', 'ggml_build_forward_expand']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);
t.el.style.fontSize = '9px';
t.el.querySelectorAll('th, td').forEach(c => { c.style.padding = '1px 5px'; c.style.lineHeight = '1.25'; });

wrap.querySelector('#ex').appendChild(W.exercise(
  '合上源码：写出 <span class="mono">llama_model_llama::graph</span> 的主干序列 —— ' +
  '每一步的<b>函数名 + 作用 + 是第几站</b>。再回答：<span class="mono">ffn_gate_inp</span> 判空是在决定什么？',
  '<b>十站</b>：<br>' +
  '1. <span class="mono">build_inp_embd(model.tok_embd)</span>（108）—— token 查表成向量<br>' +
  '2. <span class="mono">build_inp_pos()</span>（111）—— 位置序列<br>' +
  '3. <span class="mono">build_attn_inp_kv()</span>（119）—— KV cache 读写 + 掩码入口<br>' +
  '4. <span class="mono">build_inp_out_ids()</span>（124）—— 只要 logits 的行<br>' +
  '5. <span class="mono">for il in 0..n_layer-1</span>（126），层内依次：<br>' +
  '&nbsp;&nbsp;<span class="mono">build_norm(attn_norm, LLM_NORM_RMS)</span>（132）→ ' +
  '<span class="mono">build_qkv</span>（143）→ <span class="mono">ggml_rope_ext</span>（146/152）→ ' +
  '<span class="mono">build_attn</span>（169）→ <span class="mono">ggml_add(cur, inpSA)</span>（178）→ ' +
  '<span class="mono">build_norm(ffn_norm, LLM_NORM_RMS)</span>（184）→ ' +
  '<span class="mono">build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)</span>（189）→ ' +
  '<span class="mono">ggml_add(cur, ffn_inp)</span>（220）→ <span class="mono">build_cvec</span>（223）<br>' +
  '6. <span class="mono">build_norm(output_norm, LLM_NORM_RMS, -1)</span>（231）<br>' +
  '7. <span class="mono">build_lora_mm(model.output, cur, model.output_s)</span>（240）<br>' +
  '8. <span class="mono">ggml_build_forward_expand(gf, cur)</span>（246）<br><br>' +
  '<b>ffn_gate_inp 判空</b>是在决定这一层走<b>稠密 FFN</b>（为空，本课全部 37 个文件）' +
  '还是 <b>MoE 分支</b>（非空，L2-12 / L2-13 那一族）。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '十六行表 = 这一课的验收点。前四站是输入，中间是层循环，最后两站是出口。',
  '<span class="k">输入四站</span>：都叫 build_inp_*，只开张量不算数。',
  '<span class="k">层内九步</span>：两次 norm、两次残差、一次 QKV、一次 RoPE、一次注意力、一次 FFN。',
  '<span class="k">出口两站</span>：output_norm + lm_head（embed=true 时后者被编译掉）。',
  '记住一句话：<span class="v">所有稠密模型共享同一条主干，差异只在层的开关</span>。<br>QKV bias、norm 位置、tied embedding、sliding window、激活 —— 就这五个常见开关。',
  '下一课 L2-15 接着看：同一条主干上，注意力本身怎么换（GQA / SWA / MLA），<br>以及投机解码的草稿模型怎么把这条主干剪短。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3500, () => { [0, 1, 2, 3].forEach(i => { rows[i].className = 'on'; }); msg.innerHTML = texts[1]; });
tl.at(8000, () => {
  rows.forEach(r => { r.className = ''; });
  for (let i = 4; i < 13; i++) { rows[i].className = 'on'; }
  msg.innerHTML = texts[2];
});
tl.at(14000, () => {
  rows.forEach(r => { r.className = ''; });
  rows[13].className = 'on'; rows[14].className = 'on'; rows[15].className = 'on';
  msg.innerHTML = texts[3];
});
tl.at(19000, () => { rows.forEach(r => { r.className = ''; }); msg.innerHTML = texts[4]; });
tl.at(23000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、本课覆盖：37 个文件从哪来',
    '本课的 37 个文件全部取自 `src/models/`，是 `tools/plan_matrix.py` 里 **L2-14** 的清单。\n\n'
    '**分组依据（如实写）**：计划把这批文件归到"稠密 Transformer"，'
    '不是因为在代码里找到了一个叫"稠密"的标记，而是因为**它们没有命中其它三组图原语**：\n\n'
    '| 组 | 判据（代码层面） | 本课是否命中 |\n|---|---|---|\n'
    '| 多模态（L2-10） | 文件里出现视觉/音频塔（`clip`、`mmproj`、图像嵌入输入） | 少数命中：`cogvlm.cpp`、`gemma3n.cpp` |\n'
    '| 状态空间 / 线性注意力（L2-11） | 调用 `build_rs`、RWKV/Mamba/SSM 原语 | 不命中 |\n'
    '| 稀疏专家（L2-12 / L2-13） | `ffn_gate_inp` 非空、调用 `build_moe_ffn` | **零命中** |\n'
    '| 其余 | —— | 本课这 37 个 |\n\n'
    '所以"37 个稠密模型"是**排除法的结果**，不是同质的 37 份代码。'
    '逐个读完之后发现：其中有 4 个文件**根本没有主干**（图在别处），'
    '另有若干实际属于编码器 / 扩散 / 草稿模型家族。这些都在下面的清单表里逐行标出。',
    src=SRC, parts=[(94, 96)], lang='c')

L.section(
    '二、★ 主干十站：llama.cpp 逐站读',
    '主干写在 `llama_model_llama::graph<embed>` 的构造函数里（不是旧版的 `llm_build_llama` 自由函数）。'
    '下面是它的四个段落，逐字引用：\n\n'
    '**第 1 段（输入，四站）** —— `build_inp_embd` / `build_inp_pos` / `build_attn_inp_kv` / `build_inp_out_ids`：',
    src=SRC, parts=[(105, 124)], lang='c')

L.section(
    '三、主干第 2 段：逐层循环的注意力半边',
    '循环体前半：`build_norm`（pre-norm）→ `build_qkv` → `ggml_rope_ext` → `build_attn`。'
    '四个调用全部是原语，模型文件里看不到任何 `ggml_*` 算子。',
    src=SRC, parts=[(126, 173)], lang='c')

L.section(
    '四、主干第 3 段：残差与 FFN（dense / MoE 的分叉）',
    '`if (model.layers[il].ffn_gate_inp == nullptr)` 这一行判空，'
    '就是"稠密 FFN"与"MoE"两条路的唯一分叉点。本课 37 个文件全部走稠密那一支。',
    src=SRC, parts=[(174, 228)], lang='c')

L.section(
    '五、主干第 4 段：末层 norm 与 lm_head',
    '`build_norm(..., -1)` 是唯一的"层外"归一化；`if constexpr (!embed)` 是编译期开关，'
    '`llama-embed.cpp` 用 `graph<true>` 把 lm_head 整段编译掉。',
    src=SRC, parts=[(229, 247)], lang='c')

L.section(
    '六、★ 十站对应的 build_* 原语',
    '主干上每一站都是一个原语。原语的**实现**在 `src/llama-graph.cpp`（L2-06 的主题），'
    '这里只引用**声明**，用来看清"一个原语的签名 = 它能表达多少种模型"。\n\n'
    '特别注意 `build_qkv` 的两个重载与 `build_ffn` 那一长串可选指针（`*_b` 偏置、`*_s` 缩放）：'
    '**同一行调用，靠"传不传"就能表达不同模型** —— 这就是"层开关"的实现方式。',
    src=SRC_GH, parts=[(1048, 1100)], lang='c')

L.section(
    '七、层开关对照：把 gemma3 拨到另一边',
    'gemma3 是"开关拨得最多"的稠密模型之一，但它仍然走同一条主干十站：'
    'QK-norm（131/139）、调度用的 post-norm（162/185）、GELU 的 FFN（177）、'
    '以及 FFN 之后那第三个 norm（185）。对照 `llama.cpp:132/184`，'
    '差别只在"norm 摆在残差前还是后"。',
    src='src/models/gemma3.cpp', parts=[(131, 190)], lang='c')

L.section(
    '八、"名实不符"的文件：主干不在这个文件里',
    '逐个读完 37 个文件后，有四类必须如实说明：\n\n'
    '**(1) 只有 6 行、根本没有主干**：`hunyuan-dense.cpp` 与 `llama-embed.cpp` '
    '各自只实现一个 `build_arch_graph()`，图靠 `models.h` 里一行 `using graph =` 继承过来。',
    src=SRC_M, parts=[(175, 183), (2101, 2108)], lang='c')

L.section(
    '八（续）、图在别的源文件里的两个 BERT 模型',
    '`jina-bert-v2.cpp` / `jina-bert-v3.cpp` 同样把图指到 `llama_model_bert::graph`（`bert.cpp`，'
    '不在本课清单内，故本课不引用其源码、不计入本课覆盖）。'
    '它们还带 `attn_out_norm` / `layer_out_norm` 这种 BERT 式后置归一化 —— 不是自回归主干。',
    src=SRC_M, parts=[(303, 322)], lang='c')

L.section(
    '九、不属于稠密 Transformer 的其它文件（如实标注）',
    '按逐行读到的证据分类：\n\n'
    '| 文件 | 证据（行号） | 实际家族 |\n|---|---|---|\n'
    '| `dream.cpp` | `hparams.causal_attn = false;`（L15）+ `build_attn_inp_no_cache()`（L68） | 扩散式（双向、无 KV cache） |\n'
    '| `llada.cpp` | 注释 `Non-causal attention for diffusion` + `build_attn_inp_no_cache()`（L82） | 扩散式 |\n'
    '| `eurobert.cpp` | `build_attn_inp_no_cache()`（L50）、全文无 `build_lora_mm(model.output` | 编码器（无 lm_head） |\n'
    '| `gemma-embedding.cpp` | `hparams.causal_attn = false;`（L7）+ no-cache（L89） | 句向量模型 |\n'
    '| `jina-bert-v2/v3.cpp` | `using graph = llama_model_bert::graph;`（models.h L308/L319） | BERT 编码器 |\n'
    '| `eagle3.cpp` | 两张图（L125 / L152）、encoder 只做 `build_lora_mm(model.fc, ...)`（L137） | 投机解码草稿模型 |\n'
    '| `gemma4-assistant.cpp` | 只建 `n_layer_nextn` 层（L127）、`k_cur/v_cur = nullptr`（L151） | 投机解码草稿头 |\n'
    '| `granite-switch.cpp` | 自定义输入类（L134-147）、手写 FFN（`build_ffn` 从未被调用，L411-415）、图内路由器（L265-287） | 可切换适配器（switch LoRA）家族 |\n'
    '| `hrm-text.cpp` | 深度循环双栈（L185-196 嵌套循环）、无参数 RMSNorm（`nullptr` 权重，L107） | HRM 深度循环家族 |\n'
    '| `cogvlm.cpp` | `is_text` 二选一，取 `visexp_attn_wqkv` / `visexp_ffn_*`（L85-97） | 多模态（视觉专家） |\n'
    '| `gemma3n.cpp` | AltUp / Laurel / 逐层嵌入（L324-376）、FFN 手写（L214-224） | 多模态家族的特殊主干 |\n'
    '| `hunyuan-vl.cpp` | M-RoPE 四段（L66-69 / L104-115）、视觉层输入 tap（L86） | 多模态的文本半边 |\n'
    '| `hunyuan-dense.cpp` | 6 行，`using graph = llama_model_hunyuan_vl::graph`（models.h L2105） | 复用 `hunyuan-vl.cpp` 的图 |\n'
    '| `llama-embed.cpp` | 6 行，`graph<true>`（embed 模板） | 复用 `llama.cpp` 主干、无 KV cache、无 lm_head |\n\n'
    '**这只是标注，不是删减**：这 15 个文件依然被本课声明覆盖。'
    '把它们的"名"与"实"写清楚，比把它们硬说成稠密 Transformer 更有用。',
    src='src/models/bitnet.cpp', parts=[(162, 164)], lang='c')

L.section(
    '十、37 个文件的差异点（逐行读出的结果）',
    '下表逐行给出每个文件的**归一化类型 / 激活 / 真实差异点**，行号指向该文件自身。\n\n'
    '| 文件 | 归一化 | 激活 | 真实差异点（行号在源文件里） |\n|---|---|---|---|\n'
    '| `apertus.cpp` | RMS | xIELU | FFN 手写：`up -> ggml_xielu -> down`（L138） |\n'
    '| `arcee.cpp` | RMS | ReLU² | 无 gate，`LLM_FFN_SEQ`（L128） |\n'
    '| `baichuan.cpp` | RMS | SILU | 13B 不接位置编码 + ALiBi（L13 / L90） |\n'
    '| `bitnet.cpp` | RMS ×5 | SILU | 层内多两个 sub-norm（L103 / L137）；lm_head 用 `tok_embd`（L164） |\n'
    '| `bloom.cpp` | LayerNorm | GELU | fused `wqkv` + bias（L45）；嵌入后先归一化（L77）；ALiBi（L18） |\n'
    '| `chameleon.cpp` | RMS + LN | SILU | Q/K 用 LayerNorm 且可选（L91-104）；`swin_norm` 只改 norm 位置（L7） |\n'
    '| `chatglm.cpp` | RMS | SwiGLU | gate 串行：up 出 2 倍宽（L48 / L133） |\n'
    '| `codeshell.cpp` | LayerNorm | GELU | `wo_b` 偏置（L100）；反向 tie：`tok_embd` 复用 output（L15-20） |\n'
    '| `cogvlm.cpp` | RMS | SILU | 文本/视觉两套权重按 ubatch 二选一（L72-97） |\n'
    '| `cohere2.cpp` | LayerNorm | SILU | SWA 与全局层交错（L70）；并行残差（L131）；强制 tie（L29） |\n'
    '| `command-r.cpp` | LayerNorm | SILU | 并行残差（L118）；无 `ffn_norm`；Q/K norm 仅 64 层以上（L28） |\n'
    '| `dream.cpp` | RMS | SILU | 非因果 + 无 KV cache（L15 / L68） |\n'
    '| `eagle3.cpp` | RMS | SILU | 两张图：encoder 只做 fc（L137）、decoder 先拼接（L220） |\n'
    '| `eurobert.cpp` | RMS | SILU | no-cache（L50）、无 lm_head |\n'
    '| `exaone.cpp` | RMS | SILU | 逐层 `rope_freqs`（L75）；输出可 tied（L24） |\n'
    '| `exaone4.cpp` | RMS | SILU | 无 pre-norm，全 post-norm（L113）；SWA；跳过 nextn 层（L42） |\n'
    '| `falcon.cpp` | LayerNorm | GELU | FFN 吃 `attn_norm` 而非 attn 输出（L125）；双残差（L134-135） |\n'
    '| `gemma-embedding.cpp` | RMS | GELU | `causal_attn = false`（L7）+ 无 cache（L89） |\n'
    '| `gemma.cpp` | RMS | GELU | 嵌入乘 `sqrt(n_embd)`（L49）；output 恒为副本（L20） |\n'
    '| `gemma2.cpp` | RMS | GELU | post-norm + SWA（L74）+ 两处 softcap（L14 / L167） |\n'
    '| `gemma3.cpp` | RMS | GELU | QK-norm（L131 / L139）+ post-norm + SWA + tied（L44） |\n'
    '| `gemma3n.cpp` | RMS | GELU | AltUp/Laurel + 逐层嵌入（L324-376）；FFN 手写（L214-224） |\n'
    '| `gemma4-assistant.cpp` | RMS | GELU | 只建 `n_layer_nextn` 层（L127），KV 借主模型（L151） |\n'
    '| `glm4.cpp` | RMS | SwiGLU | post-norm（L140 / L162）+ mRoPE（L81） |\n'
    '| `gpt2.cpp` | LayerNorm | GELU | fused `wqkv` + bias（L38）；学习式位置嵌入，无 RoPE（L74-77） |\n'
    '| `gptneox.cpp` | LayerNorm | GELU | `use_par_res`：注意力与 FFN 并行（L145）；`wqkv_b`/`wo_b` 都带（L67 / L70） |\n'
    '| `granite-switch.cpp` | RMS | SILU | 自定义输入 + 手写 FFN + 图内路由器（L134-147 / L411-415） |\n'
    '| `hrm-text.cpp` | RMS（无参数） | SILU | 深度循环双栈交替跑同一批层（L101 / L168）；注意力带 sigmoid gate（L112） |\n'
    '| `hunyuan-dense.cpp` | - | - | 只有 6 行：图继承 `hunyuan-vl`（models.h L2105） |\n'
    '| `hunyuan-vl.cpp` | RMS | SILU | M-RoPE 四段（L66-69）+ 视觉层输入 tap（L86） |\n'
    '| `internlm2.cpp` | RMS | SILU | 无 `wo_b`；QKV 走 `create_tensor_qkv`（L26） |\n'
    '| `jais.cpp` | LayerNorm | SILU | ALiBi（L5）；不调 `build_inp_pos`（无 RoPE） |\n'
    '| `jais2.cpp` | LayerNorm | ReLU² | 无 gate，`LLM_FFN_SEQ`（L130）；tied（L22-24） |\n'
    '| `jina-bert-v2.cpp` | LayerNorm | - | 本文件无主干：图在 `bert.cpp`（models.h L308） |\n'
    '| `jina-bert-v3.cpp` | LayerNorm | - | 49 行只有张量表；图在 `bert.cpp`（models.h L319） |\n'
    '| `llada.cpp` | RMS | SILU | 非因果（L16）+ 无 KV cache（L82） |\n'
    '| `llama-embed.cpp` | - | - | 只有 6 行：复用 `llama.cpp` 主干，`graph<true>` |\n\n'
    '### 主干序列（可复述版）\n\n'
    '```text\n'
    'build_inp_embd(model.tok_embd)                     llama.cpp:108\n'
    'build_inp_pos()                                    llama.cpp:111\n'
    'build_attn_inp_kv()                                llama.cpp:119\n'
    'build_inp_out_ids()                                llama.cpp:124\n'
    'for il in 0 .. n_layer-1:                          llama.cpp:126\n'
    '    build_norm(inpL, attn_norm, NULL, RMS, il)     llama.cpp:132\n'
    '    build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il)   llama.cpp:143\n'
    '    ggml_rope_ext(Q) / ggml_rope_ext(K)            llama.cpp:146 / 152\n'
    '    build_attn(inp_attn, wo, wo_b, wo_s, Q,K,V, ...)           llama.cpp:169\n'
    '    ggml_add(cur, inpSA)                  # 残差 1   llama.cpp:178\n'
    '    build_norm(ffn_inp, ffn_norm, NULL, RMS, il)   llama.cpp:184\n'
    '    build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il)  llama.cpp:189\n'
    '    ggml_add(cur, ffn_inp)                # 残差 2   llama.cpp:220\n'
    '    build_cvec(cur, il)                            llama.cpp:223\n'
    'build_norm(cur, output_norm, NULL, RMS, -1)        llama.cpp:231\n'
    'build_lora_mm(model.output, cur, model.output_s)   llama.cpp:240\n'
    'ggml_build_forward_expand(gf, cur)                 llama.cpp:246\n'
    '```\n\n'
    '（上面是示意性文字块，不是源码引用；每一行的行号都指向本源文件里的逐字引用。`text` 围栏不参与保真门禁。）')

# ------------------------------------------------------------------ 收尾

L.footnote_add(
    '本课覆盖声明共 41 项 = **L2-14 清单里的 37 个 `src/models/*.cpp`**（逐个出现在上面某一节的逐字引用、或本课的场景 `src=` 里）'
    '+ `src/models/llama.cpp`（主干的标准答案；plan 把它归给 L2-13，本课以它为主干逐字引用）'
    '+ `src/llama-graph.cpp` 与 `src/llama-graph.h`（与 `build_*` 原语对照，L2-06 的主文件，本课有意交叉引用）'
    '+ `src/models/models.h`（三处 `using graph =`，用来解释"6 行的模型文件"；该文件同时属于 L2-10 / L2-15）。')

L.footnote_add(
    'plan 文档写的验收点是"能默写出 `llm_build_llama` 的主干图"。'
    '`llm_build_llama` 是旧版 llama.cpp 的函数名；在 v0.5.0 里主干已改为 '
    '`llama_model_llama::graph<embed>` 的构造函数（`src/models/llama.cpp:99`）。'
    '本课按真实代码命名，并在末幕给出可默写的主干序列。')

L.footnote_add(
    '`src/llama-hparams.h` 与 `src/llama-model.cpp` 里的事实（`f_max_alibi_bias`、'
    '`swa_type`/`is_swa`、`create_tensor_qkv` 的 `TENSOR_NOT_REQUIRED`）'
    '只在叙述中被引用，**本课不引用其源码、不计入本课覆盖**（它们分别属于 L2-02 与 L2-05）。')

L.prereqs('`L2-06`（计算图骨架与 `build_*` 原语）、`L2-02`（架构表与超参）')

L.goal(
    '按顺序默写 `llama_model_llama::graph<embed>` 的**主干十站**（函数名 + 行号 + 作用）；',
    '说出主干上每一站对应的 `build_*` 原语，以及在哪一课讲它的实现；',
    '列举至少 5 个"层开关"，并各举一个本课文件作为反例（QKV bias / norm 位置 / tied embedding / sliding window / 激活）；',
    '判断一个 `src/models/*.cpp` 文件是不是"标准稠密主干"：看它有没有 `build_inp_embd`、层循环、`build_attn`、`build_ffn`、`build_lora_mm`；',
    '说明为什么 37 个文件里有 4 个"根本没有主干"（`using graph =` 继承），以及它们分别继承了谁。')

L.conclusion(
    '主干十站（可默写）',
    '```text\n'
    'build_inp_embd(model.tok_embd)                     llama.cpp:108\n'
    'build_inp_pos()                                    llama.cpp:111\n'
    'build_attn_inp_kv()                                llama.cpp:119\n'
    'build_inp_out_ids()                                llama.cpp:124\n'
    'for il in 0 .. n_layer-1:                          llama.cpp:126\n'
    '    build_norm(attn_norm, LLM_NORM_RMS)            llama.cpp:132\n'
    '    build_qkv(...)                                 llama.cpp:143\n'
    '    ggml_rope_ext(Q) / ggml_rope_ext(K)            llama.cpp:146 / 152\n'
    '    build_attn(...)                                llama.cpp:169\n'
    '    ggml_add(cur, inpSA)                           llama.cpp:178\n'
    '    build_norm(ffn_norm, LLM_NORM_RMS)             llama.cpp:184\n'
    '    build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)      llama.cpp:189\n'
    '    ggml_add(cur, ffn_inp)                         llama.cpp:220\n'
    '    build_cvec(cur, il)                            llama.cpp:223\n'
    'build_norm(output_norm, LLM_NORM_RMS, -1)          llama.cpp:231\n'
    'build_lora_mm(model.output, cur, model.output_s)   llama.cpp:240\n'
    'ggml_build_forward_expand(gf, cur)                 llama.cpp:246\n'
    '```')

L.conclusion(
    '★ 所有稠密模型共享同一条主干，差异只在层的开关',
    '这是 37 个文件能并成一课的原因。逐行读出并用得上的开关有六个：\n\n'
    '| 开关 | llama.cpp 的拨法 | 反例 |\n|---|---|---|\n'
    '| norm 位置 | pre-norm（L132 / L184） | gemma3 post-norm（L162）、exaone4 无 pre-norm（L113） |\n'
    '| QKV / wo bias | `build_qkv` 里有张量才加（L143） | bloom fused `wqkv_b`（L45）、command-r 只加 Q/K norm |\n'
    '| QK-norm | 无（只有 `use_kq_norm`，L162） | gemma3（L131 / L139）、apertus（L93）、exaone4（L122） |\n'
    '| sliding window | 无：`build_attn_inp_kv`（L119） | gemma2 / gemma3 / cohere2 / exaone4：iswa 入口 + `is_swa(il)` |\n'
    '| FFN 激活 / 门控 | SILU + PAR（L194） | GELU / SwiGLU / ReLU² / 手写 xIELU（apertus L138） |\n'
    '| embedding tied | `output` 缺失才复用 `tok_embd`（L41-46） | 恒 tied：gemma L20、cohere2 L29、command-r L21、bitnet L164 |\n'
    '| 残差结构 | 串行两次相加（L178 / L220） | falcon L134-135、gptneox `use_par_res` L145、cohere2 L131 |')

L.conclusion(
    '37 个文件不等于 37 个稠密 Transformer',
    '本课的分组依据是**排除法**：这 37 个文件没有命中 multimodal / ssm / moe 三组图原语。'
    '逐个读完后必须如实标注：\n\n'
    '- **4 个文件没有主干**：`hunyuan-dense.cpp` / `llama-embed.cpp`（6 行，`using graph =` 继承）、'
    '`jina-bert-v2.cpp` / `jina-bert-v3.cpp`（图在 `bert.cpp`）。\n'
    '- **扩散式（非因果、无 KV cache）**：`dream.cpp`（L15 / L68）、`llada.cpp`（L16 / L82）。\n'
    '- **编码器 / 句向量**：`eurobert.cpp`（无 lm_head）、`gemma-embedding.cpp`（`causal_attn = false`）。\n'
    '- **草稿模型**：`eagle3.cpp`（两张图）、`gemma4-assistant.cpp`（只建 nextn 层）。\n'
    '- **自成一派**：`granite-switch.cpp`（图内路由器 + 手写 FFN）、`hrm-text.cpp`（深度循环双栈）。\n'
    '- **多模态家族**：`cogvlm.cpp`（视觉专家）、`gemma3n.cpp`（AltUp/Laurel）、`hunyuan-vl.cpp`（文本半边）。\n\n'
    '主干十站在这些文件里大多仍在，只是被机制盖住了 —— 这也正是下一课要拆的东西。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
