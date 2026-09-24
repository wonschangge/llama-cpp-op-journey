#!/usr/bin/env python3
"""L2-12 · 模型家族（三）：稀疏专家 MoE（上）—— 课件 spec。

运行：python3 L2-model-to-graph/L2-12-models-moe-a/lesson.spec.py

视角：24 个 MoE 模型文件的图代码为什么长得那么像 —— 因为它们都只是把
      `llm_graph_context::build_moe_ffn()` 这一张子图接在自己的残差上。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

G = 'src/llama-graph.cpp'              # build_moe_ffn 的定义处（L2-06 的主文件）
D = 'src/models/dbrx.cpp'              # 最典型的调用点

L = Lesson(
    id='L2-12',
    layer='L2 · 从模型到图',
    title='模型家族（三）：稀疏专家 MoE（上）',
    codecap='src/llama-graph.cpp + src/models/*.cpp（逐字引用）',
    nav={'prev': {'href': '../L2-11-models-ssm-linear/index.html',
                  'label': 'L2-11 模型家族（二）SSM'},
         'next': {'href': '../L2-13-models-moe-b/index.html',
                  'label': 'L2-13 模型家族（四）MoE 下'}},
)

L.note('**一句话**：所谓"稀疏专家"，**稀疏只体现在专家张量多出来的那一维上**。'
       '图上的算子序列与稠密 FFN 几乎一样 —— 一次打分、一次 top-k 选择、'
       'k 条并行的"乘权重矩阵"支路、一次求和 —— 只不过那 k 条支路走的不是同一个权重，'
       '而是按 `selected_experts` **按索引收集**出来的不同专家。'
       '正因为如此，24 个 MoE 模型的图代码才能长得那么像。')
L.note('本课覆盖 24 个源文件，全部在 `src/models/` 下。它们**不是按内容挑的**，'
       '而是静态扫描的结果：图构建代码里命中了 `build_moe_ffn` / `ffn_gate_exps` / '
       '`ffn_gate_inp` 之一。**分组依据是"命中了这些图原语"，不是"这个文件就是 MoE"** —— '
       '实测结果见下方第四节，24 个里有 6 个名实不符，本课如实列出。')

# ==================================================================== 第 1 幕

L.scene(
    kicker='L2-12 · 全局',
    title='24 个文件，共用<span class="hl-a">同一张子图</span>',
    sub='MoE 家族的全部差异都在"怎么选专家"，而不在"专家怎么算"—— 后者被抽成了一个原语。',
    caption='回顾 L2-06：`build_moe_ffn()` 是 llm_graph_context 的成员函数，和 build_ffn / build_norm 并列。',
    src=G, parts=[(1949, 1968)], duration=19000,
    mark_src=[1949, 1951, 1952, 1954, 1956, 1957, 1961],
    notes_src={1951: 'gate_inp = ffn_gate_inp：路由打分矩阵 [n_embd, n_expert]',
               1952: 'up_exps / gate_exps / down_exps：三个"多了一维"的专家权重张量',
               1956: 'n_expert：这一层总共有多少专家（= 专家张量最后一维的长度）',
               1957: 'n_expert_used：每个 token 选几个（top-k 的 k）',
               1961: 'gating_op：softmax / sigmoid / sqrt_softplus / softmax_weight'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="row wrap" id="cards" style="gap:9px"></div>
  <div class="formula" id="fam" style="font-size:9.5px;overflow-wrap:anywhere;line-height:1.6"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '24 个 MoE 模型文件', b: '按静态扫描分组：图构建代码命中<br>' +
      '<span class="cm" style="margin:0">build_moe_ffn | ffn_gate_exps | ffn_gate_inp</span>',
    m: '分组依据 ≠ 内容断言' },
  { c: 'b', t: '一个公共图原语', b: '所有 MoE 模型都调它；<br>家族差异只落在它的参数上',
    m: 'llm_graph_context::build_moe_ffn' },
  { c: 'c', t: '定义在 L2-06 的地盘', b: 'src/llama-graph.cpp:1993<br>本课按并集声明该文件，不影响 L2-06',
    m: 'src/llama-graph.cpp' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const names = ['afmoe','arctic','bailingmoe','bailingmoe2','bert','cohere2moe','dbrx','deci',
  'deepseek','deepseek2ocr','dots1','ernie4-5-moe','ernie4-5','exaone-moe','glm4-moe',
  'granite-moe','granite-swa','granite','grok','grovemoe','hunyuan-moe','hy-v3','laguna','lfm2moe'];
wrap.querySelector('#fam').innerHTML =
  '<span class="m">本课逐字引用并声明的 24 个文件：</span><br>' +
  names.map(n => '<span style="color:var(--b)">src/models/' + n + '.cpp</span>').join(' <span class="m">·</span> ');

const msg = wrap.querySelector('#msg');
const texts = [
  '24 个文件，看起来是 24 个模型；<br>但它们<b>共用同一个图原语</b> —— 这就是本课要讲清的那一件事。',
  '<span class="v">build_moe_ffn</span> 的签名有 <span class="k">24 个参数</span>（另有 19 个参数的简版重载）：<br>' +
  '5 个张量（gate_inp + 三组 exps + exp_probs_b）、<span class="k">2 个整数</span>（n_expert / n_expert_used）、' +
  '门控与激活类型、归一化/缩放开关，以及一批可选张量。',
  '<span class="k">家族的差异全部落在参数上</span>：<br>' +
  '谁有 gate_exps、谁用 sigmoid、谁有分组路由、谁有共享专家 —— 都是传参不同，不是另写一张图。',
  '这就是"稀疏"在图上的全部含义：<span class="v">专家张量多一维</span> + <span class="v">按索引收集</span>。<br>' +
  'L2-13（MoE 下篇）会逐个模型看这些参数怎么被填。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
}));
tl.at(14500, () => {
  els.forEach(e => e.style.opacity = '1');
  U.markLines(document, [0, 2, 3, 5, 7, 8, 12]);
  msg.innerHTML = texts[3];
});
'''
)

# ==================================================================== 第 2 幕

L.scene(
    kicker='L2-12 · 算子 1/7',
    title='★ <span class="hl-a">ffn_gate_inp</span>：一次打分，就是一次 <span class="hl-a">ggml_mul_mat</span>',
    sub='路由不是特殊算子：它就是把隐状态乘上 [n_embd, n_expert] 的打分矩阵。',
    caption='回顾 L1-02：`ggml_mul_mat` 是一个普通的 `GGML_OP_*`，路由打分用的是它，专家里用的也是它的变体。',
    src=G, parts=[(2018, 2037)], duration=20000,
    mark_src=[2018, 2019, 2025, 2029, 2035],
    notes_src={2018: 'n_embd = cur 的第 0 维（最内层，连续）',
               2019: 'n_tokens = cur 的第 1 维；下面所有张量都带这一维',
               2025: 'build_lora_mm 内部就是 ggml_mul_mat；LoRA 只是可选分支',
               2029: 'cb() 给张量起调试名字 "ffn_moe_logits"，图 dump 时能看到'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['张量', '怎么来', '形状', '产生的算子'],
  [['cur', 'ffn_norm 的输出（上游）', '[n_embd, n_tokens]', '（上游）'],
   ['ffn_gate_inp', 'load_arch_tensors 里的权重张量', '[n_embd, n_expert]', 'create_tensor'],
   ['logits', 'build_lora_mm(gate_inp, cur)', '[n_expert, n_tokens]', 'ggml_mul_mat'],
   ['probs', 'ggml_soft_max(logits)', '[n_expert, n_tokens]', 'ggml_soft_max']],
  { monoCols: [0, 2, 3] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '先把"路由"这件事拆成两个张量：<span class="v">gate_inp</span>（权重）和 <span class="v">cur</span>（数据）。',
  '<span class="v">ffn_gate_inp</span> 的形状是 <span class="k">[n_embd, n_expert]</span>：<br>' +
  '一行一个专家，代表"这个专家喜欢什么样的输入"。',
  '<span class="v">logits = build_lora_mm(gate_inp, cur)</span>：<br>' +
  '注意 build_lora_mm 内部就是 <span class="k">ggml_mul_mat</span>（grovemoe.cpp:133 直接这么写）—— ' +
  '路由打分的算子身份和普通 FFN 的 up/gate 投影<b>完全一样</b>。',
  '结果形状 <span class="v">[n_expert, n_tokens]</span>：<br>' +
  '每一列是一个 token 对全部专家的打分。这就是后面 top-k 的输入。',
  '所以"MoE 有特殊算子"是个错觉：<span class="k">打分 = ggml_mul_mat，收尾 = ggml_add</span>，都是 L1-02 里的老算子。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2900 + i * 3000, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== 第 3 幕

L.scene(
    kicker='L2-12 · 算子 2/7',
    title='门控函数：<span class="hl-c">softmax</span> / <span class="hl-c">sigmoid</span> / <span class="hl-c">sqrt(softplus)</span>',
    sub='打分之后要变成概率。选哪个函数是模型架构决定的，不是引擎决定的。',
    caption='`exp_probs_b` 只加到"选择"用的那份概率上；用来算权重的 `probs` 保持无偏 —— 源码注释写明了原因。',
    src=G, parts=[(2039, 2079)], duration=22000,
    mark_src=[2041, 2043, 2045, 2047, 2049, 2051, 2053, 2055, 2064, 2066],
    notes_src={2043: '最常见：DeepSeek-V2 / DBRX / Qwen3-MoE 都用 softmax',
               2047: 'sigmoid：逐专家独立打分，总和不必为 1（Hy-V3 / Grovemoe 一类）',
               2051: 'SOFTMAX_WEIGHT：此处原样透传，延迟到第 6 幕的 weights 上再做 softmax',
               2077: 'Grovemoe 特殊：选择概率直接用 sigmoid(logits)，覆盖上面的结果'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', k: 'SOFTMAX', m: 'ggml_soft_max', b: '整行归一化，<br>概率和 = 1。<br>最主流。' },
  { c: 'd', k: 'SIGMOID', m: 'ggml_sigmoid', b: '逐元素独立，<br>和不必为 1。<br>DeepSeek-V3 起。' },
  { c: 'f', k: 'SQRT_SOFTPLUS', m: 'ggml_sqrt(ggml_softplus(x))', b: '开方 + softplus，<br>非负。<br>少见。' },
  { c: 'e', k: 'SOFTMAX_WEIGHT', m: 'probs = logits', b: '此处不门控，<br>挪到 weights 上做。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.el('div', { class: 'card' });
  e.style.borderLeftColor = 'var(--' + d.c + ')';
  e.style.width = '163px';
  e.innerHTML = '<div class="cm" style="margin:0 0 2px">' + U.esc(d.k) + '</div>' +
    '<div class="ct" style="color:var(--' + d.c + ');font-size:15px">' + U.esc(d.m) + '</div>' +
    '<div class="cb">' + d.b + '</div>';
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="k">gating_op</span> 是一个 4 值枚举，switch 出四种概率化方式。',
  '<span class="v">SOFTMAX</span> -> <span class="m">ggml_soft_max(logits)</span>：全局归一化。',
  '<span class="v">SIGMOID</span> -> <span class="m">ggml_sigmoid(logits)</span>：每个专家独立打分。',
  '<span class="v">SQRT_SOFTPLUS</span> -> <span class="m">ggml_sqrt(ggml_softplus(logits))</span>：' +
  '两个一元算子叠起来，都是 L1-02 里已有的 op。',
  '<span class="v">SOFTMAX_WEIGHT</span>：这一步 <span class="k">probs = logits</span> 什么都不做，' +
  '等 top-k 选完再对 <span class="v">weights</span> 做 softmax（第 6 幕）。'
];
const mk = [[2], [4], [6], [8]];
defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  U.markLines(document, mk[i]);
  msg.innerHTML = texts[i];
}));
tl.at(18800, () => {
  els.forEach(e => e.style.opacity = '1');
  U.markLines(document, [2, 4, 6, 8, 10]);
  msg.innerHTML = '然后是<b>选择偏置</b>：<span class="v">selection_probs = ggml_add(probs, exp_probs_b)</span>。' +
    '<br>源码注释点明：偏置只影响"选谁"，<span class="k">probs 本身保持无偏</span>，' +
    '因为它后面还要用来算专家权重。';
});
'''
)

# ==================================================================== 第 4 幕

L.scene(
    kicker='L2-12 · 算子 3/7',
    title='分组路由：<span class="hl-d">n_expert_groups</span> / <span class="hl-d">n_group_used</span>',
    sub='DeepSeek-V3 式的"先选组、再选专家"：把 n_expert 个专家切成若干组，只在被选中的组里做 top-k。',
    caption='这两个参数不在 build_moe_ffn 的形参里 —— 它们来自 `hparams`（`src/llama-hparams.h`，属 L2-02，本课不计入覆盖）。',
    src=G, parts=[(2081, 2104)], duration=20000,
    mark_src=[2083, 2084, 2087, 2089, 2093, 2096, 2101, 2102],
    notes_src={2083: 'n_expert_groups <= 1 时整段跳过 —— 绝大多数 MoE 模型走的就是这条路',
               2087: '把 [n_expert, n_tokens] 重排成 [n_exp_per_group, n_expert_groups, n_tokens]',
               2089: '每组只取分数最高的 2 个专家，求和当作"组分数"',
               2096: '在组维度上再做一次 top-k —— 这次 k 是 n_group_used',
               2101: '把没被选中的组整体填成 -INFINITY，等价于"永久落选"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:9px">
    <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--d)">
      <div class="ct" style="color:var(--d)">hparams.n_expert_groups</div>
      <div class="cb">把 n_expert 个专家分成几组。<br>
      <span class="cm" style="margin:0">n_exp_per_group = n_expert / n_expert_groups</span><br>
      默认 0 -> 本段整段跳过。</div></div>
    <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--f)">
      <div class="ct" style="color:var(--f)">hparams.n_group_used</div>
      <div class="cb">每个 token 允许用几个<b>组</b>。<br>
      被选中组之外的概率被填成 <span class="cm" style="margin:0">-INFINITY</span>。</div></div>
  </div>
  <div class="flow" id="flow" style="gap:5px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const ops = ['ggml_reshape_3d', 'ggml_argsort_top_k(2)', 'ggml_get_rows', 'ggml_sum_rows',
             'ggml_argsort_top_k(n_group_used)', 'ggml_set_rows', 'ggml_fill(-INF)',
             'ggml_reshape_2d'];
const flow = wrap.querySelector('#flow');
const chips = ops.map((o, i) => {
  const c = U.chip(o, i === 1 || i === 4 ? 'd' : '');
  flow.appendChild(c);
  if (i < ops.length - 1) flow.appendChild(U.arrow('->'));
  return c;
});
chips.forEach(c => c.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '整段被 <span class="v">if (hparams.n_expert_groups &gt; 1 &amp;&amp; n_tokens &gt; 0)</span> 守着 —— 关掉时零开销。',
  '<span class="v">ggml_reshape_3d</span>：把概率按"组"重新排，得到 <span class="k">[n_exp_per_group, n_expert_groups, n_tokens]</span>。',
  '<span class="v">ggml_argsort_top_k(..., 2)</span>：每组取前 2 名专家，<br>再用 <span class="v">ggml_get_rows</span> + <span class="v">ggml_sum_rows</span> 求和，得到"组分数"。',
  '第二次 <span class="v">ggml_argsort_top_k</span>：在组维度上取前 <span class="k">n_group_used</span> 组。',
  '<span class="v">ggml_fill(..., -INFINITY)</span> + <span class="v">ggml_set_rows</span>：<br>' +
  '把落选组整体压成负无穷，于是第 5 幕的全局 top-k <b>不可能选到它们</b>。',
  '注意：<span class="k">分组路由没有引入任何新算子</span> —— 用的还是 reshape / argsort_top_k / get_rows / sum_rows / fill / set_rows。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
chips.forEach((c, i) => tl.at(2800 + i * 2100, () => {
  c.style.opacity = '1';
  msg.innerHTML = texts[Math.min(Math.floor(i / 2) + 1, 4)];
}));
tl.at(19600, () => { chips.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ==================================================================== 第 5 幕

L.scene(
    kicker='L2-12 · 算子 4/7',
    title='★ top-k 选择的真实函数名是 <span class="hl-b">ggml_argsort_top_k</span>',
    sub='源码里没有用 ggml_top_k —— 用的是"先 argsort 再 view"的那个版本。',
    caption='`ggml_top_k` 确实存在于 ggml.h，但整个 `src/llama-graph.cpp` 里一次都没出现（grep 可复核）。',
    src=G, parts=[(2106, 2124)], duration=18000,
    mark_src=[2107, 2109, 2110, 2112, 2120, 2123, 2124],
    notes_src={2107: 'selected_experts_in：允许调用方从外面传一份选中结果（某些模型复用别的路由）',
               2109: 'k = n_expert_used；输出 [n_expert_used, n_tokens]',
               2110: 'cb 挂在 src[0] 上：那个张量是 argsort 的完整结果，top-k 只是它的一个视图',
               2123: 'ggml_get_rows(probs, sel)：把"整行概率"收集成"选中专家的概率"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div>
  <div class="row" style="gap:8px">
    <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--b)">
      <div class="ct" style="color:var(--b)">ggml_argsort_top_k(ctx, a, k)</div>
      <div class="cb">源码注释：<span class="cm" style="margin:0">similar to ggml_top_k but implemented as \\`argsort\\` + \\`view\\`</span><br>
      结果是一段<b>有序</b>索引。</div></div>
    <div class="card" style="flex:1 1 0;min-width:0;border-left-color:var(--g)">
      <div class="ct" style="color:var(--g)">ggml_top_k(ctx, a, k)</div>
      <div class="cb">源码注释：<span class="cm" style="margin:0">the resulting top k indices are in no particular order</span><br>
      <b>本函数一次都没用它。</b></div></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['张量', '表达式', '形状'],
  [['selection_probs', '打分 / 门控 / 分组掩码之后的概率', '[n_expert, n_tokens]'],
   ['selected_experts', 'ggml_argsort_top_k(selection_probs, n_expert_used)', '[n_expert_used, n_tokens]'],
   ['probs (3d)', 'ggml_reshape_3d(probs, 1, n_expert, n_tokens)', '[1, n_expert, n_tokens]'],
   ['weights', 'ggml_get_rows(probs3, selected_experts)', '[1, n_expert_used, n_tokens]']],
  { monoCols: [0, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '这一步是全课最短、也最关键的一步：<span class="k">选中哪几个专家</span>。',
  '<span class="v">selection_probs</span> 是 [n_expert, n_tokens] —— 每列一个 token 的专家分数。',
  '<span class="v">ggml_argsort_top_k(selection_probs, n_expert_used)</span><br>' +
  '输出 <span class="k">[n_expert_used, n_tokens]</span> 的 <b>I32 专家编号表</b>。这张表就是"路由结果"。',
  '<span class="v">weights = ggml_get_rows(probs3, selected_experts)</span>：<br>' +
  '用编号表去"收集"概率，得到每个被选中专家的权重 [1, n_expert_used, n_tokens]。',
  '记住：<span class="v">selected_experts</span> 是下面所有专家计算的<b>索引参数</b> —— ' +
  '它同时也是 L2-13 里各家模型差异最大的地方（有的再乘个 scale、有的按组重映射）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2900 + i * 3100, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15400, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== 第 6 幕

L.scene(
    kicker='L2-12 · 算子 5/7',
    title='专家权重：<span class="hl-e">get_rows</span> → 归一化 → 缩放',
    sub='拿到 weights 之后有三段可选的加工，各由一个 bool / float 开关控制。',
    caption='clamp 的下界 `6.103515625e-5`；源码注释逐字写着：`Avoid division by zero, clamp to smallest number representable by F16`。',
    src=G, parts=[(2126, 2152)], duration=20000,
    mark_src=[2127, 2129, 2134, 2135, 2137, 2141, 2144, 2149, 2150],
    notes_src={2134: 'norm_w：是否把 top-k 的权重归一化回和为 1',
               2141: 'clamp 下界 = F16 最小正规数，注释：Avoid division by zero',
               2149: 'w_scale：源码这里用的是 != 0.0f && != 1.0f —— 传 0 表示"不缩放"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['分支', '触发条件', '算子', '形状变化'],
  [['SOFTMAX_WEIGHT', 'gating_op == SOFTMAX_WEIGHT', 'ggml_reshape_2d -> ggml_soft_max -> ggml_reshape_3d', '[1,k,T] 往返'],
   ['归一化 norm_w', 'norm_w == true', 'ggml_sum_rows -> ggml_clamp -> ggml_div', '[1,k,T] -> [k,T] -> [1,k,T]'],
   ['缩放 w_scale', 'w_scale != 0 && != 1', 'ggml_scale', '不变'],
   ['不缩放', '调用方传 0.0f', '（无）', '不变']],
  { monoCols: [2, 3] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '权重已经到手，但还不能直接用 —— 有三段可选加工。',
  '<span class="v">SOFTMAX_WEIGHT</span> 型：把 weights 临时压回 2 维做一次 <span class="k">ggml_soft_max</span>，再升回 3 维。',
  '<span class="v">norm_w = true</span>：<br><span class="v">ggml_sum_rows</span> 求 k 个权重之和 -> ' +
  '<span class="v">ggml_clamp</span> 防零 -> <span class="v">ggml_div</span> 除回去。',
  '<span class="v">w_scale</span>：一个 <span class="k">ggml_scale</span> 乘上常数（DeepSeek 系的 routed_scaling_factor 就走这里）。',
  '注意这三段都<b>不是必须的</b>：<span class="k">dbrx.cpp 传 norm_w=true、w_scale=expert_weights_scale</span>，' +
  '而 <span class="k">bert.cpp 传 false / 1.0f</span>。同一个原语，两种模型。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3000 + i * 3200, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(16600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== 第 7 幕

L.scene(
    kicker='L2-12 · 算子 6/7',
    title='★ 子图：<span class="hl-a">top-k 路由</span> + <span class="hl-b">专家并行</span>',
    sub='把这一课要能默画出来的子图画出来 —— 每个名字都是真实存在的 ggml 算子。',
    caption='`build_lora_mm_id` 内部就是 `ggml_mul_mat_id`（src/llama-graph.cpp:1550），它是"按专家索引做矩阵乘"的唯一入口。',
    src=G, parts=[(2166, 2217)], duration=26000,
    mark_src=[2169, 2171, 2184, 2186, 2190, 2202, 2203],
    notes_src={2169: '两条路：gate_up 融合张量（一个 mul_mat_id）或 up / gate 分开（两个 mul_mat_id）',
               2171: 'build_lora_mm_id -> ggml_mul_mat_id(ctx0, w, cur, ids)',
               2184: '融合路里，gate 与 up 是同一块结果的左右两个 ggml_view_3d',
               2202: '分开路：gate_exps 为 nullptr 时就退化成一元激活（bert.cpp 正是这样）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="cm" style="margin:0">① 路由（每个 token 选 k 个专家）—— 第 2~5 幕</div>
  <div class="flow" id="r1" style="gap:5px"></div>
  <div class="cm" style="margin:2px 0 0">② 专家并行（k 条支路，各乘自己的权重）—— 本幕</div>
  <div class="flow" id="r2" style="gap:5px"></div>
  <div class="flow" id="r3" style="gap:5px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

function band(host, names, colors) {
  const out = [];
  names.forEach((n, i) => {
    const c = U.chip(n, colors[i] || '');
    host.appendChild(c);
    out.push(c);
    if (i < names.length - 1) host.appendChild(U.arrow('->'));
  });
  return out;
}
const r1 = band(wrap.querySelector('#r1'),
  ['ffn_gate_inp [E,n_embd]', 'ggml_mul_mat', 'ggml_soft_max', 'ggml_argsort_top_k', 'ggml_get_rows'],
  ['', 'a', 'a', 'b', 'b']);
const r2 = band(wrap.querySelector('#r2'),
  ['ggml_reshape_3d', 'ggml_mul_mat_id (up_exps)', 'ggml_mul_mat_id (gate_exps)', 'ggml_swiglu_split'],
  ['', 'b', 'b', 'c']);
const r3 = band(wrap.querySelector('#r3'),
  ['ggml_mul_mat_id (down_exps)', 'ggml_mul(weights)', 'ggml_view_2d x k', 'ggml_add x (k-1)'],
  ['b', 'e', 'e', 'e']);
[r1, r2, r3].forEach(g => g.forEach(c => c.style.opacity = '.28'));

const msg = wrap.querySelector('#msg');
const texts = [
  '路由子图：打分 -> 概率 -> top-k 索引 -> 用索引收集权重。<br>' +
  '产出两张表：<span class="k">selected_experts</span>（选谁）和 <span class="v">weights</span>（各占多少）。',
  '① <span class="v">ggml_argsort_top_k</span> 输出的 [k, n_tokens] 索引，是下面专家支路的<b>全部路由信息</b>。',
  '② <span class="v">ggml_reshape_3d</span> 把输入摊成 <span class="k">[n_embd, 1, n_tokens]</span>，' +
  '再交给 <span class="v">ggml_mul_mat_id</span> —— 后者的第三个参数就是 selected_experts。',
  '关键：<span class="k">k 条支路不是 for 循环，而是同一个张量的第 1 维</span>。<br>' +
  'up / gate / down 的结果都带 <span class="v">n_expert_used</span> 这一维。',
  '③ 专家内 FFN 与稠密 FFN 完全同构：<span class="v">up</span> / <span class="v">gate</span> / 激活 / <span class="v">down</span>，' +
  '只是每次乘都多一个"按索引选权重"的动作。',
  '④ 最后 <span class="v">ggml_mul(experts, weights)</span> 加权，' +
  '<span class="v">ggml_view_2d</span> 沿 k 维切 k 份，<span class="v">ggml_add</span> 累加 —— ' +
  '<b>一次 add 就是一路专家并行</b>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3400, () => { r1.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[1]; });
tl.at(7400, () => { r2.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[2]; });
tl.at(11400, () => { msg.innerHTML = texts[3]; });
tl.at(15400, () => { r3.slice(0, 2).forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[4]; });
tl.at(19600, () => { r3.forEach(c => c.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ==================================================================== 第 8 幕

L.scene(
    kicker='L2-12 · 算子 7/7',
    title='激活、down 投影、以及<span class="hl-c">用 add 做的并行归并</span>',
    sub='n_expert_used 这一维在最后被 view + add 展平回 [n_embd, n_tokens]。',
    caption='`n_expert_used_il == 1` 时要做一次 `ggml_cont` —— 源码注释：avoid returning a non-contiguous tensor。',
    src=G, parts=[(2304, 2358)], duration=24000,
    mark_src=[2304, 2321, 2335, 2336, 2337, 2343, 2346, 2353],
    notes_src={2304: 'down 投影：同样是 ggml_mul_mat_id，索引同一张 selected_experts',
               2321: '★ 加权：k 条支路各乘自己的权重 —— 这就是"加权求和"里的"加权"',
               2335: '按【每层】的 n_expert_used 上界建图，避免热身后图规模突变（注释点明了原因）',
               2337: '★ 沿第 1 维（专家维）切出第 i 路的 [n_embd, n_tokens] 视图',
               2346: '★ k-1 次 ggml_add —— 这就是专家并行的"求和"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" id="flow" style="gap:5px"></div>
  <div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const acts = ['ggml_swiglu_split', 'ggml_silu', 'ggml_geglu_split', 'ggml_reglu_split',
              'ggml_swiglu_oai', 'ggml_swiglu_clamp'];
const flow = wrap.querySelector('#flow');
const chips = acts.map((a, i) => {
  const c = U.chip(a, 'c');
  flow.appendChild(c);
  if (i < acts.length - 1) flow.appendChild(U.arrow('/'));
  return c;
});

const t = U.table(
  ['阶段', '算子', '形状'],
  [['up / gate', 'ggml_mul_mat_id', '[n_ff, k, n_tokens]'],
   ['激活', 'ggml_swiglu_split（或 silu / geglu / reglu）', '[n_ff, k, n_tokens]'],
   ['down', 'ggml_mul_mat_id', '[n_embd, k, n_tokens]'],
   ['加权', 'ggml_mul(experts, weights)', '[n_embd, k, n_tokens]'],
   ['切分', 'ggml_view_2d x k', '[n_embd, n_tokens] x k'],
   ['归并', 'ggml_add x (k-1)', '[n_embd, n_tokens]']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '激活函数是 switch 出来的，所有分支都是已存在的 ggml 算子：',
  '<span class="v">ggml_swiglu_split</span> 是最常见的一支（有 gate 时）。' +
  '<span class="v">ggml_silu</span> 是无 gate 的退化情形 —— <span class="k">bert.cpp 就是这一支</span>。',
  '<span class="v">ggml_mul_mat_id(down_exps, ...)</span> 把 [n_ff, k, T] 投回 [n_embd, k, T]，' +
  '索引还是同一张 <span class="v">selected_experts</span>。',
  '<span class="v">ggml_mul(experts, weights)</span>：<span class="k">k 路各乘自己的权重</span>。',
  '<span class="v">ggml_view_2d</span> 沿专家维切 k 份，<span class="v">ggml_add</span> 累加 k-1 次 —— ' +
  '<b>稀疏的 k 路在这里被压回一个稠密张量</b>，层与层之间看不出 MoE。',
  '于是这一层的输出和稠密 FFN 一模一样：<span class="v">[n_embd, n_tokens]</span>，' +
  '直接加回残差。MoE 的"稀疏"从头到尾只活在<span class="k">图内部</span>。'
];
rows.forEach((r, i) => tl.at(600 + i * 3400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(21200, () => {
  rows.forEach(x => { x.className = ''; });
  chips.forEach(c => c.style.opacity = '1');
  msg.innerHTML = texts[5];
});
'''
)

# ==================================================================== 第 9 幕

L.scene(
    kicker='L2-12 · 如实报告',
    title='24 个文件里，有 <span class="hl-g">7 个需要点名</span>',
    sub='分组依据是"图构建代码命中了 MoE 图原语"。命中了不等于"这个文件就是 MoE"。',
    caption='下面这张表是逐个读出来的结论；每一行的原文引用见 source.md 第四节。',
    src='src/models/deci.cpp', parts=[(157, 167)], duration=22000,
    mark_src=[157, 161, 165],
    notes_src={157: 'deci.cpp 里 ffn_gate_inp 只出现在这个空指针判断里 —— 从来没有被创建过',
               161: '走的是稠密 build_ffn：up / gate / down 三个普通权重',
               165: '整个 deci.cpp 里 grep "moe|expert" 零命中（这是 Nemotron 稠密模型）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '文件里实际有什么', '判定'],
  [['src/models/deci.cpp', '只有一处 ffn_gate_inp == nullptr 判断；grep moe/expert 零命中', '完全不是 MoE'],
   ['src/models/deepseek2ocr.cpp', '只 create_tensor 专家张量，本文件不构图（80 行）', '图里没有 MoE'],
   ['src/models/ernie4-5.cpp', '只 create_tensor 专家张量；本文件的图是稠密分支', '图里没有 MoE'],
   ['src/models/granite-moe.cpp', '只 create_tensor 专家张量（84 行），图类不在本文件', '图里没有 MoE'],
   ['src/models/lfm2moe.cpp', '只 create_tensor 专家张量（85 行），图类不在本文件', '图里没有 MoE'],
   ['src/models/bert.cpp', '调 build_moe_ffn，但 gate_exps 传 nullptr', 'MoE，但无 gate 专家'],
   ['src/models/ernie4-5-moe.cpp', '只调 build_moe_ffn；张量由 ernie4-5.cpp 创建', '反向拆分']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '24 个文件里 <span class="k">17 个</span>是"专家张量与构图都在本文件"的真 MoE，<br>' +
  '另外 <span class="k">7 个</span>要单独点名（见上表）。',
  '<span class="v">deci.cpp</span> 最彻底：它是 Nemotron 稠密模型，<br>' +
  '文件里唯一的 MoE 痕迹是一个"专家张量为空就走稠密"的分支判断 —— 而且没有 else。',
  '<span class="v">deepseek2ocr.cpp</span> / <span class="v">ernie4-5.cpp</span> / ' +
  '<span class="v">granite-moe.cpp</span> / <span class="v">lfm2moe.cpp</span>：<br>' +
  '只负责 <span class="k">create_tensor</span>，图类定义在别的 TU 里。',
  '<span class="v">bert.cpp</span> 是 MoE（<span class="m">moe_every_n_layers</span>），' +
  '但没有 gate 专家张量：<br>build_moe_ffn 的 gate_exps 传 nullptr，激活退化成 <span class="k">ggml_silu</span>。',
  '还有反向的一例：<span class="v">ernie4-5-moe.cpp</span> 只负责构图，专家张量由 ' +
  '<span class="v">ernie4-5.cpp</span> 创建。<br>' +
  '所以"一个文件 = 一个模型"这个直觉在 <span class="v">src/models/</span> 里不成立。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3600, () => { rows.forEach((x, k) => { x.className = (k === 0) ? 'on' : ''; }); msg.innerHTML = texts[1]; });
tl.at(7600, () => { rows.forEach((x, k) => { x.className = (k >= 1 && k <= 4) ? 'on' : ''; }); msg.innerHTML = texts[2]; });
tl.at(11800, () => { rows.forEach((x, k) => { x.className = (k === 5) ? 'on' : ''; }); msg.innerHTML = texts[3]; });
tl.at(16200, () => { rows.forEach((x, k) => { x.className = (k === 6) ? 'on' : ''; }); msg.innerHTML = texts[4]; });
tl.at(20200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== 第 10 幕

L.scene(
    kicker='L2-12 · 收束',
    title='压成一张表 + 一道练习',
    sub='会画这张子图，L2-13 里所有 MoE 变体都只是它的参数组合。',
    caption='下一课 L2-13（MoE 下篇）逐个模型看：共享专家、MTP 块、chunk expert、gate_up 融合。',
    src=D, parts=[(115, 125)], duration=22000,
    mark_src=[115, 116, 117, 118, 119, 121, 122, 124],
    notes_src={116: 'ffn_gate_inp：路由打分矩阵',
               117: 'ffn_up_exps / ffn_gate_exps / ffn_down_exps：三组专家权重（各多一维）',
               121: 'n_expert 与 n_expert_used 是这一课的两个核心整数',
               124: 'LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX —— 第 3 幕的枚举'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['阶段', '真实算子', 'llama-graph.cpp 行号'],
  [['打分', 'build_lora_mm -> ggml_mul_mat', '2025'],
   ['门控', 'ggml_soft_max / ggml_sigmoid / ggml_sqrt(ggml_softplus)', '2043 / 2047 / 2055'],
   ['选择偏置', 'ggml_add(probs, exp_probs_b)', '2066'],
   ['分组路由', 'ggml_argsort_top_k x2 + ggml_fill + ggml_set_rows', '2089 / 2096 / 2101'],
   ['top-k', 'ggml_argsort_top_k(selection_probs, n_expert_used)', '2109'],
   ['收集权重', 'ggml_get_rows(probs3, selected_experts)', '2123'],
   ['归一化', 'ggml_sum_rows + ggml_clamp + ggml_div', '2137 / 2141 / 2144'],
   ['专家 up/gate', 'ggml_mul_mat_id（经 build_lora_mm_id）', '2190 / 2203'],
   ['激活', 'ggml_swiglu_split（或 ggml_silu）', '2246 / 2249'],
   ['专家 down', 'ggml_mul_mat_id', '2304'],
   ['加权', 'ggml_mul(experts, weights)', '2321'],
   ['归并', 'ggml_view_2d x k + ggml_add x (k-1)', '2337 / 2346']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '某 MoE 层：<span class="mono">n_embd=512</span>、<span class="mono">n_expert=64</span>、' +
  '<span class="mono">n_expert_used=6</span>、<span class="mono">n_ff_exp=1408</span>、' +
  '<span class="mono">n_tokens=8</span>，且 <span class="mono">gate_up_exps == nullptr</span>。<br>' +
  '（1）写出 <span class="mono">selected_experts</span> / <span class="mono">weights</span> 的形状；<br>' +
  '（2）专家支路里哪一个算子替代了稠密 FFN 的 <span class="mono">ggml_mul_mat</span>，它的第三个参数是什么？<br>' +
  '（3）最后归并用了哪两个算子，各几次？',
  '（1）<span class="mono">ggml_argsort_top_k(selection_probs, n_expert_used)</span> 输出 ' +
  '<span class="mono">[n_expert_used, n_tokens] = [6, 8]</span>；<br>' +
  'probs 先 <span class="mono">ggml_reshape_3d(probs, 1, n_expert, n_tokens)</span> 成 ' +
  '<span class="mono">[1, 64, 8]</span>，再 <span class="mono">ggml_get_rows</span> 得 ' +
  '<span class="mono">weights = [1, 6, 8]</span>（源码注释逐字写着这两个形状）。<br>' +
  '（2）是 <span class="mono">ggml_mul_mat_id</span>（经 <span class="mono">build_lora_mm_id</span>，' +
  'src/llama-graph.cpp:1550）；第三个参数是 <span class="mono">selected_experts</span>，即那张 [6, 8] 的索引表。<br>' +
  '（3）<span class="mono">ggml_view_2d</span> 切 <span class="mono">k = 6</span> 次，' +
  '<span class="mono">ggml_add</span> 累加 <span class="mono">k-1 = 5</span> 次；' +
  '若该层 <span class="mono">n_expert_used == 1</span> 还要补一次 <span class="mono">ggml_cont</span>。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '12 个阶段，全部是 L1-02 里已有 op 的<b>重新组合</b>。',
  '前四行都属于"路由"：<span class="k">决定谁能参与</span>。',
  '中间四行是"专家并行"：<span class="k">k 路同时算，索引同一张表</span>。',
  '最后四行是"归并"：<span class="k">k 路压回一个稠密张量</span>，加回残差。',
  '一句话收束：<span class="v">稀疏只体现在专家张量多出来的那一维</span>；' +
  '图上多出来的只有"一次 top-k 选择"和"一次按索引的收集"。<br>' +
  '所以 24 个 MoE 模型的图代码才能长得那么像。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2500 + i * 1250, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  if (i === 3) msg.innerHTML = texts[1];
  if (i === 7) msg.innerHTML = texts[2];
  if (i === 11) msg.innerHTML = texts[3];
}));
tl.at(19600, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[4];
});
'''
)

# ================================================================== source.md

L.section(
    '一、本课的分组方法（先读这一节）',
    '本课覆盖的 24 个文件**不是按内容选的**，而是**静态扫描**的结果：凡是 `src/models/` 下 '
    '图构建代码里出现了 `build_moe_ffn` / `ffn_gate_exps` / `ffn_gate_inp` 之一的 `.cpp`，'
    '就被分到"稀疏专家 MoE（上）"这一课。\n\n'
    '```text\n'
    'python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L2-12"{print $2}\'\n'
    '-> 24 个文件，全部在 src/models/ 下\n'
    '```\n\n'
    '**分组依据是"命中了这些图原语"，不是"这个文件就是 MoE"。** 逐个读完之后，'
    '24 个里有 6 个名实不符（见第四节），本课如实列出，不为了数字好看而虚报。\n\n'
    '而这 24 个文件之所以能被归成"一个家族"，是因为它们**都只是把同一个图原语接在残差上**。'
    '那个原语就是下面这段签名 —— 它在 `src/llama-graph.cpp`，属于 L2-06 的计算图骨架。'
    '**本课显式声明覆盖 `src/llama-graph.cpp`**：覆盖度是并集，这不影响 L2-06 的声明。',
    src=G, parts=[(1993, 2017)], lang='c')

L.section(
    '二、★ build_moe_ffn 的真实算子序列',
    '下表是逐行读 `src/llama-graph.cpp:1993-2358` 得到的**真实**序列。'
    '"门控"和"分组路由"两段是可选的，由 `gating_op` 与 `hparams.n_expert_groups` 决定。\n\n'
    '| # | 阶段 | 真实算子 | 行号 |\n|---|---|---|---|\n'
    '| 1 | 打分 | `build_lora_mm(gate_inp, cur)` -> `ggml_mul_mat` | 2025 |\n'
    '| 2 | 选择偏置（可选） | `ggml_add(logits, gate_inp_b)` | 2035 |\n'
    '| 3 | 门控 | `ggml_soft_max` / `ggml_sigmoid` / `ggml_sqrt(ggml_softplus(..))` | 2043 / 2047 / 2055 |\n'
    '| 4 | 专家选择偏置 | `ggml_add(probs, exp_probs_b)` | 2066 |\n'
    '| 5 | 分组路由（可选） | `ggml_reshape_3d` -> `ggml_argsort_top_k(..,2)` -> `ggml_get_rows` -> `ggml_sum_rows` -> `ggml_argsort_top_k(..,n_group_used)` -> `ggml_get_rows` -> `ggml_fill(-INFINITY)` -> `ggml_set_rows` -> `ggml_reshape_2d` | 2087-2102 |\n'
    '| 6 | top-k | `ggml_argsort_top_k(selection_probs, n_expert_used)` | 2109 |\n'
    '| 7 | 收集权重 | `ggml_reshape_3d` -> `ggml_get_rows(probs, selected_experts)` | 2120 / 2123 |\n'
    '| 8 | 权重 softmax（可选） | `ggml_reshape_2d` -> `ggml_soft_max` -> `ggml_reshape_3d` | 2128-2130 |\n'
    '| 9 | 权重归一化（可选） | `ggml_reshape_2d` -> `ggml_sum_rows` -> `ggml_clamp` -> `ggml_div` -> `ggml_reshape_3d` | 2135-2147 |\n'
    '| 10 | 权重缩放（可选） | `ggml_scale(weights, w_scale)` | 2150 |\n'
    '| 11 | 输入摊平 | `ggml_reshape_3d(cur, n_embd, 1, n_tokens)` | 2157 |\n'
    '| 12a | 融合 gate_up 路 | `build_lora_mm_id` -> `ggml_mul_mat_id`，再两个 `ggml_view_3d` | 2171 / 2184 / 2186 |\n'
    '| 12b | 分开 gate/up 路 | `build_lora_mm_id(up_exps,..)` + `build_lora_mm_id(gate_exps,..)` | 2190 / 2203 |\n'
    '| 13 | 激活 | `ggml_swiglu_split` / `ggml_silu` / `ggml_geglu_split` / `ggml_reglu_split` / `ggml_swiglu_oai` / `ggml_swiglu_clamp` | 2246 / 2249 / 2269 / 2285 / 2280 / 2229 |\n'
    '| 14 | down 投影 | `build_lora_mm_id(down_exps,..)` -> `ggml_mul_mat_id` | 2304 |\n'
    '| 15 | 加权 | `ggml_mul(experts, weights)` | 2321 |\n'
    '| 16 | 切分 | `ggml_view_2d(experts, n_embd, n_tokens, nb[2], i*nb[1])` x k | 2337 |\n'
    '| 17 | 归并 | `ggml_add` x (k-1) | 2346 |\n'
    '| 18 | 单专家特例 | `ggml_cont(moe_out)` | 2353 |\n\n'
    '**没有 `ggml_moe_*` 这样的专用算子。** `ggml_top_k` 虽然存在于 `ggml/include/ggml.h`，'
    '但在整个 `src/llama-graph.cpp` 里一次都没出现 —— 源码用的是 `ggml_argsort_top_k`。',
    src=G, parts=[(2024, 2060)], lang='c')

L.section(
    '三、三种调用形态（代表模型）',
    '同一个 `build_moe_ffn`，24 个模型填出三种典型形态：\n\n'
    '```text\n'
    '形态 A（最常见）：七个参数一次给全\n'
    '    build_moe_ffn(cur, ffn_gate_inp, ffn_up_exps, ffn_gate_exps,\n'
    '                  ffn_down_exps, nullptr, n_expert, n_expert_used, ...)\n'
    '形态 B（grovemoe）：自己先算好 logits，用 probs_in 传进去，gate_inp 传 nullptr\n'
    '形态 C（新式）：多加 gate_up_exps / *_s 缩放张量，并把共享专家另算一条 build_ffn\n'
    '```\n\n'
    '形态 A 的代表是 `dbrx.cpp`（本课第 10 幕引用）；形态 B 是 `grovemoe.cpp`，'
    '它甚至把 `build_lora_mm(ffn_gate_inp, cur)` 直接写在自己的图代码里；'
    '形态 C 是 `hy-v3.cpp` 与 `cohere2moe.cpp`，多出来的 `ffn_gate_up_exps` 走的是'
    '`build_moe_ffn` 的"融合路"分支（`src/llama-graph.cpp:2169`）。',
    src='src/models/grovemoe.cpp', parts=[(133, 148)], lang='c')

# ----------------------------------------------------------- 24 个文件逐个点名
# (文件, 起始行, 结束行, 判定, 说明)
_FILES = [
    ('src/models/afmoe.cpp', 214, 224, '真 MoE',
     '张量与构图都在本文件：`create_tensor` 建 `ffn_gate_inp` / 三组 `*_exps`，'
     '图里调 `build_moe_ffn`，门控函数取自 `hparams.expert_gating_func`。'),
    ('src/models/arctic.cpp', 141, 151, '真 MoE',
     '与 afmoe 同构；`n_ff`（不是 `n_ff_exp`）作为专家中间维，说明专家宽度可以等于稠密宽度。'),
    ('src/models/bailingmoe.cpp', 128, 138, '真 MoE',
     '把 `build_moe_ffn` 的返回值直接赋给 `cur`；`hparams.expert_weights_norm` 与 `scale` 逐层可变。'),
    ('src/models/bailingmoe2.cpp', 161, 171, '真 MoE',
     '张量创建带 `flags` 变量（量化/放置标志），构图与 bailingmoe 一致。'),
    ('src/models/bert.cpp', 167, 178, 'MoE，但无 gate 专家',
     '`gate_exps` 位置传的是 `nullptr`，且 `gate_up` 参数完全省略 —— 只有 up/down 两组专家，'
     '激活退化成 `ggml_silu`。触发条件是 `hparams.moe_every_n_layers > 0 && il % moe_every_n_layers == 1`。'),
    ('src/models/cohere2moe.cpp', 226, 240, '真 MoE',
     '形态 C：传了 `gate_up_exps` 与三个 `*_s` 缩放张量。同一文件里 MTP 块（381 行起）再调一次。'),
    ('src/models/dbrx.cpp', 115, 125, '真 MoE',
     '形态 A 的最短样例，本课第 1 幕与第 10 幕引用它。'),
    ('src/models/deci.cpp', 157, 167, '★ 完全不是 MoE',
     '`ffn_gate_inp` 只出现在一个空指针判断里，**从未被 `create_tensor` 创建**；'
     '判断为真时走的是稠密 `build_ffn`。整个文件 grep `moe|expert` 零命中（Deci/Nemotron 是稠密模型）。'),
    ('src/models/deepseek.cpp', 145, 155, '真 MoE',
     '`moe_out` 之后还与共享专家/稠密分支相加，是"路由专家 + 共享专家"写法的早期形态。'),
    ('src/models/deepseek2ocr.cpp', 55, 67, '只有加载面',
     '文件共 80 行，只做 `load_arch_hparams` + `load_arch_tensors`；'
     '`build_arch_graph` 返回的 `graph` 类定义在别处。图代码里没有 `build_moe_ffn`。'),
    ('src/models/dots1.cpp', 145, 155, '真 MoE',
     '与 deepseek.cpp 同构（dots1 是 DeepSeek 系衍生）。'),
    ('src/models/ernie4-5-moe.cpp', 81, 91, '只构图，不建张量',
     '反向的不符：本文件只有 `build_moe_ffn` 调用，专家张量由 `ernie4-5.cpp` 的 '
     '`arch == LLM_ARCH_ERNIE4_5_MOE` 分支创建。这里传了 `ffn_exp_probs_b`（第 5 个位置参数）。'),
    ('src/models/ernie4-5.cpp', 49, 56, '只有加载面',
     '`create_tensor` 建了 `ffn_gate_inp` 与三组 `*_exps`（`ffn_gate_exps` 标记 '
     '`TENSOR_NOT_REQUIRED`），但本文件的图代码只有稠密 `build_ffn` 分支。'),
    ('src/models/exaone-moe.cpp', 188, 198, '真 MoE',
     '构图前先判断 `model.layers[il].ffn_gate_inp == nullptr` 来决定走稠密还是 MoE。'),
    ('src/models/glm4-moe.cpp', 235, 245, '真 MoE',
     '同一文件里有两处调用：MTP 块（235 行）与主干（390 行）。'),
    ('src/models/granite-moe.cpp', 66, 69, '只有加载面',
     '文件共 84 行，只有 `load_arch_hparams` + `load_arch_tensors` + '
     '`build_arch_graph` 三件事；`graph` 类不在本文件。'),
    ('src/models/granite-swa.cpp', 277, 287, '真 MoE',
     '多传了一个 `gate_up_exps` 实参（融合路）；专家张量由 `create_tensor_gate_up_exps()` 创建。'),
    ('src/models/granite.cpp', 279, 289, '真 MoE',
     '与 granite-swa 同构，区别只在注意力（SWA vs 全注意力）。'),
    ('src/models/grok.cpp', 158, 168, '真 MoE',
     '`ffn_gate_exps` 标记 `TENSOR_NOT_REQUIRED`，`n_ff_exp` 作为专家宽度。'),
    ('src/models/grovemoe.cpp', 133, 148, '真 MoE（形态 B）',
     '★ 唯一在本文件里自己写 `ggml_mul_mat`（经 `build_lora_mm`）算路由打分的；'
     '随后把它作为 `probs_in` 传进 `build_moe_ffn`，`gate_inp` 位置传 `nullptr`。'
     '同一文件下面还有一次针对 `*_chexps`（chunk expert）的调用。'),
    ('src/models/hunyuan-moe.cpp', 147, 157, '真 MoE',
     '结果先存进 `cur_moe`，之后再与别的分支合并。'),
    ('src/models/hy-v3.cpp', 173, 188, '真 MoE（形态 C）',
     '参数最多的一处：`gate_up_exps` + `ffn_exp_probs_b` + 三个 `*_s` 缩放张量；'
     '下面还紧跟一段共享专家的 `build_ffn`，最后 `ggml_add(moe_out, sh_out)`。'),
    ('src/models/laguna.cpp', 277, 287, '真 MoE',
     '上一行注释写明 `routed_scaling_factor (all handled by build_moe_ffn)` —— '
     '缩放是原语的活，不是模型文件的活。'),
    ('src/models/lfm2moe.cpp', 44, 48, '只有加载面',
     '文件共 85 行，只有 `load_arch_hparams` + `load_arch_tensors`；'
     '`build_arch_graph` 返回模板化的 `graph<true/false>`，定义在别处。'),
]

L.section(
    '四、名实核对：24 个文件逐个点名',
    '分组依据是"命中了 MoE 图原语"。下面逐个文件给出**逐字引用**与判定。\n\n'
    '| # | 文件 | 判定 |\n|---|---|---|\n' +
    '\n'.join(f'| {i} | `{rel}` | {v} |' for i, (rel, _a, _b, v, _t) in enumerate(_FILES, 1)) +
    '\n\n统计（24 个）：**17 个**专家张量与 `build_moe_ffn` 构图**都在本文件**；'
    '**1 个**是 `ernie4-5-moe.cpp`（只构图，张量在 `ernie4-5.cpp`）；'
    '**1 个**是 `bert.cpp`（是 MoE，但 `gate_exps` 传 `nullptr`，无 gate 专家）；'
    '**4 个**只建专家张量、本文件不构图（`deepseek2ocr.cpp` / `ernie4-5.cpp` / '
    '`granite-moe.cpp` / `lfm2moe.cpp`）；**1 个**完全不是 MoE（`deci.cpp`）。'
    '17 + 1 + 1 + 4 + 1 = 24。',
    src='src/models/dbrx.cpp', parts=[(115, 125)], lang='c')

for _i, (_rel, _a, _b, _verdict, _txt) in enumerate(_FILES, 1):
    L.section(f'四.{_i} {_rel} —— {_verdict}', _txt,
              src=_rel, parts=[(_a, _b)], lang='c')

L.footnote_add('**覆盖声明**：本课声明的源文件是 **25 个** —— `src/models/` 下计划指派的 24 个 `.cpp`，'
               '外加 `src/llama-graph.cpp`（`build_moe_ffn` 的定义处）。')
L.footnote_add('**`src/llama-graph.cpp` 属于 L2-06**（★ 计算图骨架 llama-graph）。'
               '本课显式引用并声明它，是因为 MoE 的公共原语定义在这里；'
               '覆盖度按并集计算，不会让 L2-06 的声明失配，也不会重复计数。')
L.footnote_add('**不计入覆盖率**：本课还提到了 `src/llama-hparams.h`（`n_expert` / `n_expert_groups` / '
               '`n_group_used` / `n_expert_used()` 的声明处，属 L2-02）、`src/llama-arch.h`'
               '（`enum llm_tensor` 里的 `LLM_TENSOR_FFN_GATE_INP` / `_GATE_EXPS` / `_UP_EXPS` / '
               '`_DOWN_EXPS` / `_EXP_PROBS_B`）与 `ggml/include/ggml.h`（算子声明，属 L1-02）。'
               '这三处只作参数名与算子名的核对，**已在 lesson.spec.py 里通过 grep 确认，但未逐字引用**，'
               '故不计入本课覆盖。')
L.footnote_add('**分组依据**：24 个文件来自 `tools/plan_matrix.py` 的静态扫描规则 '
               '`build_moe_ffn|ffn_gate_exps|ffn_gate_inp`（见 `tools/plan_matrix.py` 的 `MODEL_MARKERS`）。'
               '本课第四节逐个读并如实记录了 6 处名实不符。')
L.footnote_add('**命名事实**（已 grep 确认，可复核）：`build_moe_ffn` 里 top-k 用的是 '
               '`ggml_argsort_top_k`（第 2089 / 2096 / 2109 行），**不是** `ggml_top_k`；'
               '`ggml_top_k` 在整个 `src/llama-graph.cpp` 里出现 0 次。'
               '专家矩阵乘用的是 `ggml_mul_mat_id`（经 `build_lora_mm_id`，第 1550 行）。')

L.prereqs('`L2-06`（★ 计算图骨架 llama-graph：`llm_graph_context` 与 `build_ffn`）')

L.goal(
    '说出 MoE 层在计算图上的**七个阶段**，以及每个阶段对应的真实 `ggml_*` 算子（对应验收点）；',
    '默画出 **top-k 路由 + 专家并行** 子图，并说明 `selected_experts` 在这张图里被用了几次、用在哪；',
    '解释"稀疏"在图上的确切含义：专家张量多一维 + 一次按索引收集，其余与稠密 FFN 同构；',
    '区分 `n_expert` / `n_expert_used`（形参）与 `hparams.n_expert_groups` / `hparams.n_group_used`'
    '（分组路由，来自 hparams）这两组参数；',
    '指出 24 个受检文件里哪几个名实不符，并说出各自的实情。')

L.conclusion(
    'MoE 的"稀疏"只在张量维度上',
    '`build_moe_ffn` 的全部输出就是一个 `[n_embd, n_tokens]` 的稠密张量，'
    '与 `build_ffn` 的返回值形状完全一致，直接加回残差。'
    '"稀疏"只活在图内部：专家权重张量比稠密 FFN 多一维（`n_expert`），'
    '而计算时只沿 `n_expert_used` 这一维展开。')

L.conclusion(
    '★ 真实算子序列（无专用 MoE 算子）',
    '```text\n'
    'ggml_mul_mat(ffn_gate_inp, cur)              -> logits   [n_expert, n_tokens]\n'
    'ggml_soft_max | ggml_sigmoid | sqrt+softplus -> probs\n'
    'ggml_add(probs, exp_probs_b)                 -> selection_probs   (可选)\n'
    'ggml_argsort_top_k(selection_probs, k)       -> selected_experts  [k, n_tokens]\n'
    'ggml_get_rows(probs3, selected_experts)      -> weights [1, k, n_tokens]\n'
    'ggml_mul_mat_id(up_exps,   cur, sel)         -> up    [n_ff, k, n_tokens]\n'
    'ggml_mul_mat_id(gate_exps, cur, sel)         -> gate  [n_ff, k, n_tokens]\n'
    'ggml_swiglu_split(gate, up) | ggml_silu(up)  -> act   [n_ff, k, n_tokens]\n'
    'ggml_mul_mat_id(down_exps, act, sel)         -> experts [n_embd, k, n_tokens]\n'
    'ggml_mul(experts, weights)                   -> 加权\n'
    'ggml_view_2d x k ; ggml_add x (k-1)          -> moe_out [n_embd, n_tokens]\n'
    '```\n\n'
    '**没有一个 `ggml_moe_*` 专用算子。** 这张表可以直接和 L2-06 讲的 `build_ffn` '
    '（`ggml_mul_mat` x3 + 一个激活）逐行对照 —— 多的只有"top-k 选择"与"按索引收集/乘"。')

L.conclusion(
    '★ 24 个文件像，是因为差异都在参数上',
    '家族的差异全部落在 `build_moe_ffn` 的 24 个参数上：有没有 `gate_exps`、'
    '`gating_op` 取哪个枚举、`norm_w` / `w_scale` 传什么、有没有 `gate_up_exps`、'
    '以及 `hparams.n_expert_groups` 是否大于 1。**没有谁另写一张图。**\n\n'
    '这也是 L1-02 "op 的身份由 `enum ggml_op` 决定"的直接推论：'
    '算子身份与模型架构解耦，模型只决定算子的**组合方式与参数**。')

L.conclusion(
    '名实不符必须如实写',
    '24 个受检文件里：`deci.cpp` **完全不是 MoE**（只有一处空指针判断）；'
    '`bert.cpp` 是 MoE 但无 gate 专家；`deepseek2ocr.cpp` / `ernie4-5.cpp` / '
    '`granite-moe.cpp` / `lfm2moe.cpp` **只有加载面**；'
    '`ernie4-5-moe.cpp` 反过来只构图不建张量。\n\n'
    '教训：`src/models/` 里**一个文件 ≠ 一个模型** —— 张量加载与图构建可以拆在不同的 TU。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
