#!/usr/bin/env python3
"""L2-09 · 量化、导出与适配器 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-09-quant-export-adapter/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_Q   = 'src/llama-quant.cpp'
SRC_QH  = 'src/llama-quant.h'
SRC_MS  = 'src/llama-model-saver.cpp'
SRC_MSH = 'src/llama-model-saver.h'
SRC_AD  = 'src/llama-adapter.cpp'
SRC_ADH = 'src/llama-adapter.h'
SRC_GR  = 'src/llama-grammar.cpp'
SRC_GRH = 'src/llama-grammar.h'
SRC_CH  = 'src/llama-chat.cpp'
SRC_CHH = 'src/llama-chat.h'
SRC_G   = 'src/llama-graph.cpp'   # 追加引用：LoRA 的图内插入点，只在这里定义


def nidx(a, line):
    """单段引用里，上游行号 line 对应的 notes 键（= 段内偏移）。"""
    return line - a


def ridx(a, line, notes=None):
    """单段引用里，上游行号 line 渲染后的下标 —— 注解行会把后面的行整体推后。

    只给 visual 里的 U.markLines 动态高亮用；scene 的静态高亮走 mark_src，
    由 lessonkit 换算，不需要手算。
    """
    k = line - a
    return k + sum(1 for x in (notes or {}) if x < k)


def fill(tpl, **kv):
    for k, v in kv.items():
        tpl = tpl.replace('__' + k + '__', str(v))
    return tpl


L = Lesson(
    id='L2-09',
    layer='L2 · 从模型到图',
    title='量化、导出与适配器',
    codecap='src/llama-quant / model-saver / adapter / grammar / chat（逐字引用）',
    nav={'prev': {'href': '../L2-08-sampler-and-vocab/index.html', 'label': 'L2-08 采样器与词表'},
         'next': {'href': '../L2-10-models-multimodal/index.html', 'label': 'L2-10 模型家族（一）多模态'}},
)

# ---------------------------------------------------------------- 覆盖声明
# 计划里分给 L2-09 的 10 个文件，全量声明。
L.cover(
    SRC_Q, SRC_QH,
    SRC_MS, SRC_MSH,
    SRC_AD, SRC_ADH,
    SRC_GR, SRC_GRH,
    SRC_CH, SRC_CHH,
    # 追加一个：LoRA 的图内插入点 build_lora_mm 只定义在 llama-graph.cpp（计划里归 L2-06）。
    # 不引用它，本课的验收点（LoRA 在哪一步进图）就没有源码证据。理由见 README 说明。
    SRC_G,
)

L.note('**一句话**：一个已经训练好的模型，在"进入图"之前会经过三条互相独立的改造路径 ——'
       '**量化**改的是权重的数值与类型，**导出**改的是容器（写回 GGUF），'
       '**适配器**一个字节的权重都不改，它只是在建图时往图里多插几次矩阵乘。')
L.note('三条路各有一个入口函数：`llama_model_quantize_impl()`（`src/llama-quant.cpp:913`）、'
       '`llama_model_saver::save()`（`src/llama-model-saver.cpp:499`）、'
       '`llm_graph_context::build_lora_mm()`（`src/llama-graph.cpp:1514`）。'
       '本课把这 10 个文件读一遍，最后回答一个问题：**LoRA 到底是在哪一步被加进图的。**')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2-09 · 全局',
    title='模型进入图之前，有四件事会发生',
    sub='量化改数值与类型，导出改容器，适配器改图；grammar 与对话模板则完全不碰图。',
    caption='本课逐字引用计划里的 10 个文件；另追加 src/llama-graph.cpp 作为"LoRA 在图里的插入点"的证据（见 README 说明）。',
    src=SRC_Q, parts=[(913, 926)], duration=18000,
    mark_src=[913, 922, 923, 924],
    notes={nidx(913, 913): '量化是"读一个 GGUF、写一个 GGUF"；中间全是逐张量的决策'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">权重文件</span><span class="arrow">-></span>
    <span class="chip b">量化 · 导出</span><span class="arrow">-></span>
    <span class="chip c">GGUF</span><span class="arrow">-></span>
    <span class="chip d">建图</span><span class="arrow">-></span>
    <span class="chip e">适配器 · 约束</span>
  </div>
  <div class="row wrap" id="s1cards" style="gap:8px"></div>
  <div class="formula" id="s1msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#s1cards');
const defs = [
  { c: 'a', t: '量化 · llama-quant.cpp', m: 'llama_model_quantize_impl',
    b: '逐张量选类型，再把浮点权重量化成块。<br>写出的还是 GGUF，但数值和 type 都变了。' },
  { c: 'b', t: '导出 · llama-model-saver', m: 'llama_model_saver::save',
    b: '把内存里的模型登记进一个 gguf_context，<br>只有 save() 才真正落盘。' },
  { c: 'c', t: '适配器 · llama-adapter', m: 'llama_adapter_lora::ab_map',
    b: '权重一个字节都不动。<br>建图时给命中的权重多插一次矩阵乘。' },
  { c: 'd', t: '约束 · grammar / chat', m: 'llama_grammar_apply_impl',
    b: '也不进图：一个把非法 token 的 logit 打成 -INFINITY，<br>一个把对话拼成 prompt 文本。' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#s1msg');
const texts = [
  '本课覆盖 10 个文件（外加 1 个追加引用）。先看四条支路各自动了什么。',
  '<span class="k">量化</span>：入口参数只有一个 <span class="v">ftype</span>，' +
  '但真正被决定的是<b>每一个张量</b>的目标类型 —— 第 2 幕展开。',
  '<span class="k">导出</span>：<span class="v">add_kv</span> 写元数据、' +
  '<span class="v">add_tensor</span> 登记张量指针、<span class="v">save()</span> 落盘，三件事分开做。',
  '<span class="k">适配器</span>：这是本课的核心洞察 —— ' +
  'LoRA 不是"改权重"，是<b>在图里多插一次矩阵乘</b>，所以同一个基座能同时挂多套。',
  '<span class="k">grammar 与对话模板</span>都不在图里：' +
  '前者改 logits（回顾 L2-08），后者决定 prompt 文本。'
];
defs.forEach((_, i) => tl.at(600 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-09 · 量化',
    title='★ <span class="hl-a">一个 ftype</span>不是"所有张量同一个类型"',
    sub='ftype 只换算出一个默认类型；最终类型是逐张量算出来的：类别、层号、形状各有权重。',
    caption='下一幕讲 imatrix：为什么低比特类型必须靠它校准。',
    src=SRC_Q, parts=[(708, 739)], duration=21000,
    mark_src=[708, 711, 714, 730, 731, 735],
    notes={nidx(708, 711): '默认类型之上，还有三种"手工/特例"覆盖',
           nidx(708, 730): '没被手工覆盖、也不是"纯量化"时，才走按类别与层号的混合精度表',
           nidx(708, 735): '形状装不下目标类型时回退到可用类型（tensor_type_fallback）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="s2tbl"></div><div class="formula" id="s2msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['次序', '谁说话', '代码里的名字'],
  [['1 默认',     'ftype 换算出一个默认类型',        'default_type'],
   ['2 词嵌入',   'category 是 TOKEN_EMBD 时覆盖',   'token_embedding_type'],
   ['3 输出层',   'category 是 OUTPUT 时覆盖',       'output_tensor_type'],
   ['4 手工覆盖', '张量名正则，命中即生效',          'tensor_type_patterns'],
   ['5 混合精度', '按张量类别 + 层号决定多给比特',   'llama_tensor_get_type_impl'],
   ['6 形状回退', '目标类型装不下这个形状',          'tensor_type_fallback']],
  { monoCols: [2] });
wrap.querySelector('#s2tbl').appendChild(t.el);

const msg = wrap.querySelector('#s2msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '同一个 <span class="v">ftype</span>（比如一个 4 比特混合精度档），落到不同张量上可以是不同类型。顺序就是这张表。',
  '<span class="v">default_type</span> 只是起点：<span class="k">ftype -> 默认 ggml_type</span> 的一次换算。',
  '<span class="k">词嵌入</span>与<span class="k">输出层</span>是两个可以单独指定的张量：' +
  '它们对质量最敏感，所以给了独立的覆盖入口。',
  '<span class="k">手工覆盖</span>按张量名正则生效；命中后连标准策略都不再走。',
  '<span class="k">混合精度</span>才是"哪里多给比特"的真正算法：它按张量类别（attn_v / ffn_down / …）' +
  '和层号（前 1/8、后 1/8 之类）决定升到 Q5_K、Q6_K 还是保持 Q4_K。',
  '最后一步是<span class="k">形状回退</span>：某些形状无法用目标类型表示，必须换一个能用的类型。',
  '所以"我用了某个量化档"这句话，展开后是<b>几百个张量各自的目标类型</b>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2500 + i * 2600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(18600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-09 · 量化',
    title='★ <span class="hl-c">imatrix</span>：改的是误差往哪儿分，不是算法',
    sub='低比特的 IQ* 家族必须有重要性矩阵，否则直接抛错 —— 因为"哪一列重要"本身是数据。',
    caption='"imatrix" 的形状是 ne[0] x ne[2]（按行 1247 校验）；下一幕看它怎么被按行取用。',
    src=SRC_Q, parts=[(822, 841)], duration=19000,
    mark_src=[822, 823, 826, 834, 837],
    notes={nidx(822, 823): '两个豁免：词嵌入与 output 权重',
           nidx(822, 834): 'k-quant 家族唯一的例外：Q2_K 只在 Q2_K_S 这种档位下才要 imatrix'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="s3cards" style="gap:8px"></div>
  <div class="formula" id="s3msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#s3cards');
const defs = [
  { c: 'c', t: '必须有', b: 'IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS /<br>IQ2_S / IQ3_XXS —— 六个 IQ 类型' },
  { c: 'd', t: '唯一的 k-quant 例外', b: 'Q2_K 只在 Q2_K_S 这种档位下才要；<br>源码注释：k 系量化一般不要求 imatrix' },
  { c: 'b', t: '永远豁免', b: '词嵌入与 output 权重：<br>它们本来就不参与这套判断' },
  { c: 'g', t: '缺了会怎样', b: '收集阶段直接抛 runtime_error；<br>日志写着 The result will be garbage' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#s3msg');
const texts = [
  '<span class="v">tensor_requires_imatrix()</span> 只有二十行，却把"哪些类型非它不可"写死了。',
  '六个 <span class="k">IQ*</span> 类型返回 true：它们的码本/码率太激进，' +
  '必须靠重要性矩阵告诉量化器<b>哪些列更值钱</b>。',
  '<span class="k">Q2_K</span> 是唯一的例外，而且取决于档位：只有 Q2_K_S 才要求。',
  '<span class="k">词嵌入与 output 权重</span>不参与 —— 所以这两类张量即使在 IQ 档位下也不报错。',
  '缺 imatrix 的后果不是"质量差点"：<span class="v">llama_model_quantize_impl</span> 在' +
  '收集阶段就抛错（第 1099-1109 行），日志原话是 The result will be garbage。'
];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(15500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-09 · 量化',
    title='逐行量化：<span class="hl-b">chunk</span> 不跨专家边界',
    sub='一张权重矩阵按行切开，行号全局连续；但每个专家有自己的 imatrix 切片，所以切块要在边界处断开。',
    caption='回顾 L1-04：Q4_0 的"32 个权重 + 1 个 scale"这类块布局，正是在 ggml_quantize_chunk 里落地的。',
    src=SRC_Q, parts=[(746, 768)], duration=20000,
    mark_src=[746, 750, 751, 761],
    notes={nidx(746, 762): '量化结果还要逐块校验：校验不过就整体抛错'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">nrows_total = 24</span><span class="arrow">=</span>
    <span class="chip b">nrows_per_expert = 8</span><span class="arrow">x</span>
    <span class="chip c">3 个专家</span>
  </div>
  <div id="s4ruler" style="display:flex;gap:6px;justify-content:center"></div>
  <div class="formula" id="s4msg"></div>`;
root.appendChild(wrap);

const ruler = wrap.querySelector('#s4ruler');
const cells = [];
for (let e = 0; e < 3; e++) {
  const seg = U.el('div', { style: 'display:flex;gap:2px;padding-right:5px;margin-right:3px;border-right:2px solid var(--c)' });
  for (let r = 0; r < 8; r++) {
    const c = U.el('div', { style: 'width:19px;height:17px;border:1px solid var(--border);border-radius:2px;background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
    c.textContent = String(e * 8 + r);
    seg.appendChild(c); cells.push(c);
  }
  ruler.appendChild(seg);
}

const msg = wrap.querySelector('#s4msg');
const texts = [
  '一张权重矩阵的行是<b>全局连续编号</b>的：专家的行也排在同一条行号轴上。',
  '一次 <span class="v">ggml_quantize_chunk</span> 处理若干行（chunk）；多线程时按 chunk 抢任务，' +
  '所以线程数会被 chunk 数限制。',
  '<span class="k">chunk 在专家边界处强制断开</span>：<span class="v">this_nrow</span> 同时受' +
  '<span class="v">nrows_per_chunk</span> 与 <span class="v">nrows_per_expert</span> 约束。',
  '原因在这一行 lambda：imatrix 是<b>按专家切片</b>存的，' +
  '<span class="v">row_global / nrows_per_expert</span> 就是要落到第几张切片。',
  '<span class="v">ggml_row_size(new_type, n_per_row)</span> 决定每行多少字节 ——' +
  '这正是 L1-04 讲的块布局，也是 L1-01 讲的 nb[0]。'
];
tl.at(700, () => {
  msg.innerHTML = texts[0];
  [0, 1, 2, 3, 4].forEach(i => { cells[i].style.background = 'rgba(88,166,255,.22)'; cells[i].style.borderColor = 'var(--a)'; });
});
tl.at(3700, () => {
  [5, 6, 7].forEach(i => { cells[i].style.background = 'rgba(63,185,80,.22)'; cells[i].style.borderColor = 'var(--b)'; });
  msg.innerHTML = texts[1];
});
tl.at(6700, () => {
  cells[7].style.borderColor = 'var(--c)'; cells[7].style.borderRight = '3px solid var(--c)';
  msg.innerHTML = texts[2];
});
tl.at(9700, () => { msg.innerHTML = texts[3]; });
tl.at(12700, () => { [8, 9, 10, 11, 12].forEach(i => { cells[i].style.background = 'rgba(210,153,34,.22)'; }); msg.innerHTML = texts[3]; });
tl.at(15700, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-09 · 导出',
    title='导出：<span class="hl-d">llama_model_saver</span> 的三分工',
    sub='登记元数据、登记张量、写盘是三个独立动作；结构体自己只持有"一个还没落盘的 GGUF"。',
    caption='回顾 L1-06：GGUF 是"元数据 KV + 张量表 + 数据段"的容器；这一课看它的写出侧。',
    src=SRC_MSH, parts=[(12, 45)], duration=19000,
    mark_src=[13, 15, 22, 37, 39, 41, 43],
    notes={nidx(12, 15): 'gguf_ctx 就是"还没落盘的文件"：所有登记都进这里',
           nidx(12, 37): 'add_tensor 只登记张量指针，不拷贝字节',
           nidx(12, 43): '真正的写盘只有这两行'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">add_kv_from_model()</span>
    <span class="arrow">+</span>
    <span class="chip b">add_tensors_from_model()</span>
    <span class="arrow">-></span>
    <span class="chip c">gguf_context</span>
    <span class="arrow">-></span>
    <span class="chip d">save(path)</span>
    <span class="arrow">-></span>
    <span class="chip e">.gguf</span>
  </div>
  <div class="row wrap" id="s5cards" style="gap:8px"></div>
  <div class="formula" id="s5msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#s5cards');
const defs = [
  { c: 'a', t: 'add_kv 的重载族', b: 'u32 / i32 / u64 / f32 / bool / 字符串 /<br>容器（还可以按层展开）' },
  { c: 'b', t: 'add_tensor', b: '名字已存在就跳过（只有 rope_freqs<br>那三个是例外，见 137 行断言）' },
  { c: 'c', t: 'save', b: 'gguf_write_to_file(gguf_ctx, path, false)<br>—— 到这一步才产生字节' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#s5msg');
const texts = [
  '这个结构体只有 4 个字段：一个 gguf_context、一个"是否自己拥有它"、一个模型指针、一个 KV 名字生成器。',
  '<span class="v">gguf_ctx</span> 承载整个导出过程 —— 它此时还只是内存里的一个 GGUF 描述。',
  '<span class="k">元数据</span>侧：<span class="v">add_kv</span> 按类型重载，容器版本还能按层展开成数组。',
  '<span class="k">张量</span>侧：<span class="v">add_tensor</span> 把已有张量<b>登记</b>进上下文；' +
  '字节仍然在模型自己的 buffer 里，直到 save() 才被写出去。',
  '<span class="k">写盘</span>侧：<span class="v">save()</span> 只有一行调用 —— ' +
  '这也是 L1-06 里那张"容器 -> 文件"的落地处。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

NOTES6 = {nidx(1514, 1518): '基座路径：w 在这里被读一次，之后再也不被写',
          nidx(1514, 1521): 'w_s 是逐张量缩放，和 LoRA 无关，只是搭同一趟车',
          nidx(1514, 1524): '循环：挂在同一基座上的每一套 adapter 都各追加一次',
          nidx(1514, 1535): '内层把 cur 投到 rank 维，外层再投回 n_embd'}

L.scene(
    kicker='L2-09 · 适配器',
    title='★ LoRA 不是改权重，是在图里<span class="hl-a">多插一次矩阵乘</span>',
    sub='基座权重 w 在 ggml_mul_mat 之后原封不动；每个命中的 adapter 再追加 a -> b -> scale -> add。',
    caption='本幕引用的 src/llama-graph.cpp 不在计划给 L2-09 的 10 个文件里 —— 它是"图内插入点"的唯一源码证据，故追加引用。',
    src=SRC_G, parts=[(1514, 1543)], duration=22000,
    mark_src=[1518, 1521, 1524, 1525, 1526, 1531, 1533, 1534, 1535, 1538, 1539],
    notes=NOTES6,
    visual=fill('''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:8px">
    <div class="col grow" style="gap:6px" id="s6base"></div>
    <div class="col grow" style="gap:6px" id="s6lora"></div>
  </div>
  <div class="formula" id="s6msg"></div>`;
root.appendChild(wrap);

function nodeRow(host, items, cls) {
  const r = U.el('div', { class: 'flow', style: 'gap:4px' });
  items.forEach((t, i) => {
    if (i) r.appendChild(U.arrow('->'));
    r.appendChild(U.chip(t, cls));
  });
  host.appendChild(r);
  return r;
}
const base = wrap.querySelector('#s6base');
base.innerHTML = '<div class="cm" style="margin:0 0 2px">没有 adapter</div>';
const b1 = nodeRow(base, ['cur', 'mul_mat(w, cur)', 'res'], 'a');
base.appendChild(U.el('div', { class: 'cb', text: '1 个矩阵乘算子' }));

const lora = wrap.querySelector('#s6lora');
lora.innerHTML = '<div class="cm" style="margin:0 0 2px">挂了一套 adapter</div>';
const l1 = nodeRow(lora, ['cur', 'mul_mat(w, cur)', 'res'], 'a');
const l2 = nodeRow(lora, ['cur', 'mul_mat(a, cur)', 'mul_mat(b, .)', 'scale'], 'e');
const l3 = nodeRow(lora, ['res', 'add', "res'"], 'b');
lora.appendChild(U.el('div', { class: 'cb', text: '多出 5 个节点 + 1 次加法（源码注释算作 6 个）' }));

const msg = wrap.querySelector('#s6msg');
const texts = [
  '先看没有 adapter 的情况：<span class="v">build_lora_mm</span> 就是一次普通的 <span class="k">ggml_mul_mat(ctx0, w, cur)</span>。',
  '这个函数名里的 <span class="v">lora</span> 说明了它的地位：<b>所有走它的权重才可能被 LoRA 作用</b>。',
  '<span class="v">get_weight(w)</span> 拿基座张量的<b>名字</b>去查表；查不到就 <span class="k">continue</span>，' +
  '这套 adapter 与这个权重无关。',
  '命中时追加一条<b>并行支路</b>：<span class="v">mul_mat(a, cur)</span> 降到 rank 维，' +
  '<span class="v">mul_mat(b, .)</span> 再升回输出维，然后乘上 scale。',
  '最后 <span class="v">ggml_add(ctx0, res, ab_cur)</span> 把支路并回主干 —— ' +
  '<b>基座权重 w 从头到尾只被读，没有被改写。</b>',
  '所以同一个基座可以同时挂多套 adapter：<span class="v">for (const auto & lora : *loras)</span> 每套各加一次，' +
  '代价只是图上多出节点。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [__M_BASE__]); });
tl.at(4000, () => { msg.innerHTML = texts[1]; U.markLines(document, [__M_BASE__, __M_WS__]); });
tl.at(7300, () => { msg.innerHTML = texts[2]; U.markLines(document, [__M_LOOP__]); });
tl.at(10600, () => { msg.innerHTML = texts[3]; U.markLines(document, [__M_AB__, __M_AB2__]); });
tl.at(13900, () => { msg.innerHTML = texts[4]; U.markLines(document, [__M_SCALE__, __M_ADD__]); });
tl.at(17200, () => { msg.innerHTML = texts[5]; U.markLines(document, [__M_BASE__, __M_LOOP__, __M_AB__, __M_ADD__]); });
''', M_BASE=ridx(1514, 1518, NOTES6), M_WS=ridx(1514, 1521, NOTES6),
     M_LOOP=ridx(1514, 1524, NOTES6), M_AB=ridx(1514, 1533, NOTES6),
     M_AB2=ridx(1514, 1535, NOTES6), M_SCALE=ridx(1514, 1538, NOTES6),
     M_ADD=ridx(1514, 1539, NOTES6))
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-09 · 适配器',
    title='适配器的全部状态：一张 <span class="hl-e">ab_map</span>',
    sub='一套 adapter = "基座张量名 -> (lora_a, lora_b)" 的映射 + 一个 alpha；作用点由建图时按名字查表决定。',
    caption='回顾 L2-06：llm_graph_context 是 build_* 原语族的宿主；build_lora_mm 就是它的一个原语。',
    src=SRC_ADH, parts=[(44, 88)], duration=20000,
    mark_src=[48, 53, 54, 63, 67, 83, 86],
    notes={nidx(44, 50): 'a、b 两个小矩阵就是 LoRA 的全部数据',
           nidx(44, 67): 'ab_map 是这套 adapter 的全部状态：名字 -> (a, b)',
           nidx(44, 86): '这行注释就是"图会多出多少节点"的账本'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="s7cards" style="gap:8px"></div>
  <div class="formula" id="s7msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#s7cards');
const defs = [
  { c: 'e', t: 'ab_map', m: 'name -> (a, b)', b: '一套 adapter 一张表。<br>每套 adapter 还带一个 alpha。' },
  { c: 'a', t: 'get_weight(w)', m: 'w->name 查表', b: '查不到返回 nullptr ——<br>这套 adapter 就跳过这个权重。' },
  { c: 'c', t: 'get_scale', m: 's * alpha / rank', b: 'rank 取自 b->ne[0]；<br>alpha 为 0 时只用传入的 scale。' },
  { c: 'b', t: 'get_n_nodes()', m: 'ab_map.size() * 6u', b: '命中多少个权重，<br>图上就多出 6 倍个节点。' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#s7msg');
const texts = [
  '适配器的"数据面"只有这两个结构体：一个小矩阵对，加一张名字表。',
  '<span class="v">llama_adapter_lora_weight</span> 就是 <span class="v">a</span> 与 ' +
  '<span class="v">b</span> 两个张量指针 —— 没有别的。',
  '<span class="v">get_scale()</span> 把 rank 从形状里读出来：' +
  '<span class="v">rank = b->ne[0]</span>，再按 alpha 折算实际缩放。',
  '<span class="v">ab_map</span> 的键是<b>基座张量的名字</b>；' +
  '<span class="v">get_weight(w)</span> 就是一次 <span class="v">find</span>。',
  '<span class="v">get_n_nodes()</span> 把上一幕的图变化量化了：' +
  '每命中一个权重 6 个节点（a、b、scale、add、两次 mul_mat）。',
  '所以"多套 adapter"在图上就是<b>多次 add</b>，在权重的字节里什么痕迹都没有。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
tl.at(16500, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-09 · 约束',
    title='grammar 不进图：它在采样链上<span class="hl-c">改 logits</span>',
    sub='采样前把不可能的 token 打成 -INFINITY，采样后再按生成的文本推进规则栈 —— 两步都在图外。',
    caption='回顾 L2-08：采样在 CPU 上做、不进图；grammar 正是这条链上的一环，所以它能直接改 logits。',
    src=SRC_GR, parts=[(1355, 1396)], duration=20000,
    mark_src=[1355, 1358, 1362, 1380, 1382, 1392, 1394],
    notes={nidx(1355, 1358): 'awaiting_trigger：懒 grammar 在触发词出现前完全不约束',
           nidx(1355, 1394): '被拒绝的 token 不是被删掉，而是把 logit 打成 -INFINITY'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">logits</span><span class="arrow">-></span>
    <span class="chip c">grammar 拒绝的置 -INFINITY</span><span class="arrow">-></span>
    <span class="chip b">采样（L2-08）</span><span class="arrow">-></span>
    <span class="chip d">accept_token 推进栈</span>
  </div>
  <div class="row wrap" id="s8cards" style="gap:8px"></div>
  <div class="formula" id="s8msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#s8cards');
const defs = [
  { c: 'a', t: 'stacks', b: '一组下推自动机的栈；<br>空了才表示"这个规则已经满足"。' },
  { c: 'b', t: 'allow_eog', b: '只要有任意一个栈为空，<br>才允许结束符活着。' },
  { c: 'c', t: 'candidates', b: '每个 token 先取出 piece 并解码成<br>code point，再拿去和规则比。' },
  { c: 'd', t: 'accept_token', b: '采样之后按同一个 piece<br>推进栈（1472 行）。' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#s8msg');
const texts = [
  '整个函数没有出现任何 ggml 张量 —— 它操作的是采样器的候选数组。',
  '<span class="v">awaiting_trigger</span> 为真时直接返回：<b>懒 grammar 在触发词出现前一个 token 都不约束</b>。',
  '<span class="k">allow_eog</span> 由栈是否为空算出；不允许结束时，所有结束符的 logit 被置为 -INFINITY。',
  '其余 token 走解码 + 规则匹配，被拒绝的同样置 <span class="v">-INFINITY</span> —— ' +
  '<b>不是删除，软屏蔽</b>，后面的采样器仍看到完整候选表。',
  '采样结束后 <span class="v">llama_grammar_accept_token</span> 用<b>同一段 piece</b> 推进栈：' +
  '生成与约束用的是同一个字符串，所以不会错位。',
  '这也解释了 L2-08 的验收点：<b>约束发生在采样这一步，而采样不在图里</b>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
tl.at(17000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-09 · 对话模板',
    title='对话模板：一张 <span class="hl-f">硬编码名字表</span> + 一条 if-else 链',
    sub='模板不是从文件里解析出来的语法，而是几十个手写分支；每个分支把消息拼成一段 prompt 文本。',
    caption='这段文本随后被分词成 inp_tokens —— 图的第一批输入；模板决定了"图上看到什么 token"。',
    src=SRC_CH, parts=[(244, 258)], duration=19000,
    mark_src=[244, 250, 253, 255, 256],
    notes={nidx(244, 250): '这一支只处理 chatml。往下还有几十个 else if，一个模型一族'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">messages</span><span class="arrow">-></span>
    <span class="chip c">模板分支拼串</span><span class="arrow">-></span>
    <span class="chip b">prompt 文本</span><span class="arrow">-></span>
    <span class="chip d">分词</span><span class="arrow">-></span>
    <span class="chip e">inp_tokens</span>
  </div>
  <div class="row wrap" id="s9cards" style="gap:8px"></div>
  <div class="formula" id="s9msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#s9cards');
const defs = [
  { c: 'a', t: '名字表 54 项', b: 'LLM_CHAT_TEMPLATES（28-83 行）<br>从 chatml 到 solar-open。' },
  { c: 'c', t: 'detect 兜底', b: '名字查不到时，按模板字符串里的<br>子串特征猜（89 行起）。' },
  { c: 'b', t: 'add_ass', b: '为真时补一个 assistant 头，<br>让模型从这里接着生成。' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#s9msg');
const texts = [
  '模板的输入是消息数组，输出是一段字符串；它<b>不产生任何算子</b>。',
  '<span class="v">llm_chat_apply_template</span> 从头到尾就是一条 <span class="k">if / else if</span> 链，' +
  '每个内置模板一个分支。',
  '以 <span class="k">chatml</span> 为例：每条消息拼成 ' +
  '<span class="v">im_start + role + 内容 + im_end</span>，用字符串流一路写下去。',
  '<span class="v">add_ass</span> 为真时再补一个 assistant 头 —— 这就是"让模型接着说"的那一步。',
  '模板决定了图上看到什么 token：这段字符串被分词后就是 ' +
  '<span class="v">inp_tokens</span>（回顾 L2-06/L2-07 的图输入）。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14000, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L2-09 · 收束',
    title='把这一课压成一张表',
    sub='量化与导出改的是"文件里的东西"，适配器与约束改的是"图与采样"——后者一个权重字节都不碰。',
    caption='下一课 L2-10：模型家族（一）多模态 —— 视觉编码器的输出怎么接进同一张图。',
    src=SRC_Q, parts=[(1365, 1385)], duration=20000,
    mark_src=[1368, 1371, 1372, 1376, 1377],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="s10tbl"></div><div id="s10ex"></div><div class="formula" id="s10msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '它动的是什么', '关键函数'],
  [['llama-quant.cpp/.h', '权重的数值与 type', 'llama_model_quantize_impl'],
   ['llama-model-saver.cpp/.h', 'GGUF 容器', 'add_kv / add_tensor / save'],
   ['llama-adapter.cpp/.h', '名字 -> (a, b) 映射', 'get_weight / get_scale'],
   ['llama-graph.cpp（追加）', '图本身：多插矩阵乘', 'build_lora_mm'],
   ['llama-grammar.cpp/.h', '采样前的 logits', 'llama_grammar_apply_impl'],
   ['llama-chat.cpp/.h', 'prompt 文本', 'llm_chat_apply_template']],
  { monoCols: [2] });
wrap.querySelector('#s10tbl').appendChild(t.el);

wrap.querySelector('#s10ex').appendChild(W.exercise(
  'LoRA 是在哪一步被加进图的？基座权重张量有没有被改写？',
  '在<b>建图那一步</b>（llama-context.cpp:1425 的 <span class="mono">model.build_graph(gparams)</span>；' +
  '图能复用时不会重建），由 <span class="mono">llm_graph_context::build_lora_mm()</span> ' +
  '（<span class="mono">src/llama-graph.cpp:1514</span>）完成。<br>' +
  '它先做正常的 <span class="mono">ggml_mul_mat(ctx0, w, cur)</span>，再对 ' +
  '<span class="mono">*loras</span> 里每一个能查到该权重的 adapter 追加 ' +
  '<span class="mono">ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur))</span> -> ' +
  '<span class="mono">ggml_scale</span> -> <span class="mono">ggml_add</span>。<br>' +
  '基座权重 <span class="mono">w</span> 只被读，没有被改写；所以同一个基座能同时挂多套 adapter，' +
  '代价是图上多出节点（源码注释：每命中一个权重 6 个）。'));

const msg = wrap.querySelector('#s10msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '十个文件，按"动什么"分成三组。',
  '<span class="k">量化 + 导出</span>动的是<b>文件</b>：数值、类型、容器。',
  '<span class="k">适配器</span>动的是<b>图</b>：权重不变，节点变多。',
  '<span class="k">grammar + 对话模板</span>动的是<b>图和采样之外</b>的两端：logits 与 prompt 文本。',
  '一句话：<span class="v">LoRA 是图上的加法，不是权重里的改写</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
}));
tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、量化：从一个 GGUF 到另一个 GGUF',
    '量化的入口是 `llama_model_quantize_impl()`。它只接收两个文件名和一个参数结构体，'
    '第一件事就是把 `ftype` 换算成默认类型 —— 类型不合法就直接抛错。\n\n'
    '注意参数结构体本身**不在这个文件里**：`llama-quant.h` 全文只有一行 `#pragma once`（见第十六节），'
    '`llama_model_quantize_params` 定义在 `include/llama.h:434`。',
    src=SRC_Q, parts=[(913, 926)], lang='c')
L.section(
    '二、★ 一个 ftype 不是一种类型',
    '`llama_tensor_get_type()` 的返回路径有六条：默认类型、词嵌入特例、输出层特例、'
    '张量名正则手工覆盖、按类别与层号的混合精度表、以及形状回退。'
    '下面这段是后三条的完整逻辑 —— 也就是说，**"混合精度量化"这个说法在这里才落地**。',
    src=SRC_Q, parts=[(708, 739)], lang='c')
L.section(
    '三、★ imatrix：谁重要，是数据不是算法',
    '哪些目标类型**非有 imatrix 不可**，被写死在这二十行里。'
    'IQ 家族全部要求；k-quant 家族只有 Q2_K 是例外，而且只在 Q2_K_S 档位下要求。\n\n'
    '缺失时的后果不是"质量下降"，而是在收集阶段就抛异常：',
    src=SRC_Q, parts=[(822, 841)], lang='c')
L.section(
    '四、imatrix 是在哪里被用掉的',
    '每个张量化之前，代码会按 `ne[0]*ne[2]` 的形状去 imatrix 表里找对应的向量，'
    '尺寸不符（且不是词嵌入）就直接抛错；找不到而目标类型又必须有它，同样抛错。'
    '这段还顺手挡住了"对已经量化过的张量再量化"这种操作。',
    src=SRC_Q, parts=[(1241, 1270)], lang='c')
L.section(
    '五、逐行量化：chunk 与专家边界',
    '一张权重矩阵被按行切成 chunk 交给线程；行号是跨专家全局连续的，'
    '但每个专家有自己的 imatrix 切片，所以 **chunk 不允许跨越专家边界**。'
    '真正的量化调用是 `ggml_quantize_chunk()`，它的产物还会被逐块校验。',
    src=SRC_Q, parts=[(746, 768)], lang='c')
L.section(
    '六、导出：llama_model_saver 的三分工',
    '导出侧的类很小：登记元数据（`add_kv` 的一族重载）、登记张量（`add_tensor`）、写盘（`save`）。'
    '`gguf_ctx` 就是"还没落盘的文件"，所有登记都进它。',
    src=SRC_MSH, parts=[(12, 45)], lang='c')
L.section(
    '七、add_tensor 与 save 的实现',
    '`add_tensor` 只做两件事：已存在同名张量就返回（只有三个 rope 相关张量允许重名），'
    '否则把指针登记进 `gguf_ctx`。**字节仍然在模型自己的 buffer 里。**\n\n'
    '真正的落盘是 `save()`：一行调用，把上下文写成文件。',
    src=SRC_MS, parts=[(131, 142), (499, 505)], lang='c')
L.section(
    '八、★ LoRA 的图内插入点：build_lora_mm',
    '这一段是本课验收点的直接证据。`build_lora_mm()` 的骨架是：\n\n'
    '```text\n'
    'res = ggml_mul_mat(ctx0, w, cur)        // 基座那一次，w 只读\n'
    'for lora in *loras:                     // 每一套挂着的 adapter\n'
    '    lw = lora.get_weight(w)             // 按 w 的名字查表\n'
    '    if lw == nullptr: continue          // 这套 adapter 不管这个权重\n'
    '    ab = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur))\n'
    '    ab = ggml_scale(ctx0, ab, scale)\n'
    '    res = ggml_add(ctx0, res, ab)       // 并回主干\n'
    '```\n\n'
    '**基座张量 `w` 从头到尾没有被改写。** 这段代码属于 `src/llama-graph.cpp`，'
    '计划里把它归给 L2-06（计算图骨架）；本课追加引用它，理由见文末说明。',
    src=SRC_G, parts=[(1514, 1543)], lang='c')
L.section(
    '九、适配器的数据面',
    '一个 adapter 在内存里就是 `ab_map`（基座张量名 -> 一对小矩阵）加一个 `alpha`。'
    '`get_scale()` 把 rank 从 `b` 的形状里读出来；`get_n_nodes()` 则把"图会多出多少节点"'
    '直接写成 `ab_map.size() * 6u`，注释列出了那 6 个节点。',
    src=SRC_ADH, parts=[(44, 88)], lang='c')
L.section(
    '十、适配器是怎么装进来的：靠名字后缀配对',
    '加载 adapter 文件时，每个张量名必须以 `.lora_a` 或 `.lora_b` 结尾；'
    '去掉后缀剩下的就是**基座里的张量名**。两者在 `ab_map` 里配成一对。'
    '不认识的结尾直接抛错；`_norm.weight` 被显式跳过 —— 源码注释说明多数 adapter 不依赖它。',
    src=SRC_AD, parts=[(268, 296)], lang='c')
L.section(
    '十一、运行时按名字查表',
    '建图时 `build_lora_mm()` 调用的就是这一个函数：拿基座张量的名字去 `ab_map` 里找，'
    '找不到返回 `nullptr`。**"这套 adapter 作用于哪些权重"完全是运行时按名字决定的**，'
    '不需要在加载时改写任何权重。',
    src=SRC_AD, parts=[(140, 149)], lang='c')
L.section(
    '十二、grammar：约束是采样的事，不是图的事',
    '`llama_grammar_apply_impl()` 遍历采样候选，把规则不接受的 token 的 logit 置为 `-INFINITY`；'
    '`llama_grammar_accept_token()` 则在采样之后，用**同一个 piece** 推进规则栈。'
    '两个函数都不碰 ggml 张量。',
    src=SRC_GR, parts=[(1355, 1396), (1472, 1500)], lang='c')
L.section(
    '十三、grammar 的数据结构',
    '规则被编译成一串 `llama_grammar_element`：字符、字符区间、取反、规则引用、'
    '以及按 token id 匹配的元素。`llama_grammar` 本体持有规则表与一组栈，'
    '外加"懒 grammar"的触发词缓冲。',
    src=SRC_GRH, parts=[(11, 31)], lang='c')
L.section(
    '十四、对话模板：一张名字表 + 一条 if-else 链',
    '名字表里每一项把模板名映射到枚举值（共 54 项）。模板字符串本身由 '
    '`llama_model_chat_template()`（声明在 `include/llama.h:647`）从模型里取出；'
    '名字对不上时，`llm_chat_detect_template()` 会退化成对模板字符串做子串特征匹配。\n\n'
    '应用模板就是一条很长的分支链，下面是第一条分支（chatml）的原文。',
    src=SRC_CH, parts=[(28, 45), (244, 258)], lang='c')
L.section(
    '十五、模板枚举',
    '`llm_chat_template` 有 56 个值（含末尾的 UNKNOWN）；名字表覆盖其中 54 个。'
    '枚举第一项就是 chatml —— 也就是上一节引用的那条分支。',
    src=SRC_CHH, parts=[(7, 22)], lang='c')
L.section(
    '十六、llama-quant.h 全文',
    '这个头文件全文只有一行。量化模块的对外接口不在它里面：'
    '参数结构体与入口函数都声明在 `include/llama.h`（`llama_model_quantize_params` 在第 434 行，'
    '`llama_model_quantize_default_params` 在第 477 行）。'
    '把它列进覆盖是因为它确实是这个模块的源文件 —— 引用它也是引用"事实"。',
    src=SRC_QH, parts=[(1, 1)], lang='c')

L.footnote_add('本课按计划覆盖 10 个文件，全部计入覆盖率：'
               '`src/llama-quant.{cpp,h}`、`src/llama-model-saver.{cpp,h}`、'
               '`src/llama-adapter.{cpp,h}`、`src/llama-grammar.{cpp,h}`、`src/llama-chat.{cpp,h}`。')
L.footnote_add('**额外引用 1 个清单外文件**：`src/llama-graph.cpp`（计划里归 L2-06「计算图骨架」）。'
               '理由：本课验收点是"能说出 LoRA 是在哪一步被加进图的"，而图内插入点 '
               '`llm_graph_context::build_lora_mm()` 只在 `src/llama-graph.cpp:1514` 定义。'
               '不引用它，这个结论就只能靠推断 —— 而本仓库的红线是"每个断言都必须能从源码看出来"。'
               '覆盖度门禁取的是并集，追加引用不会让任何文件失配（`llama-graph.cpp` 已由 L2-06 覆盖）。')
L.footnote_add('`src/llama-quant.h` 全文 1 行（`#pragma once`），本课在 source.md 第十六节逐字引用，'
               '不计为"未引用但声明"。')

L.prereqs('`L2-08`（采样器与词表 —— 本课第八幕的 grammar 就挂在它的采样链上）')

L.goal(
    '说出 `llama_model_quantize_impl()` 里"类型"是在哪一步被逐个张量决定的；',
    '解释 imatrix 在量化中的角色，以及哪几类目标类型**必须**有它、缺了会怎样；',
    '说明 `llama_model_saver` 的 `add_kv` / `add_tensor` / `save` 三者的分工；',
    '**说出 LoRA 是在哪一步、由哪个函数加进图的**（本课验收点），'
    '并说明基座权重有没有被改写；',
    '说明 grammar 与对话模板为什么都不在计算图里，它们各自改的是什么。')

L.conclusion(
    '三条改造路径，各自动什么',
    '| 路径 | 入口 | 动的是什么 |\n|---|---|---|\n'
    '| 量化 | `llama_model_quantize_impl()` | 权重的**数值与 type**（还是 GGUF） |\n'
    '| 导出 | `llama_model_saver::save()` | **容器**：把内存里的模型写成 GGUF |\n'
    '| 适配器 | `llm_graph_context::build_lora_mm()` | **图**：多插几次矩阵乘，权重不变 |\n'
    '| 约束 | `llama_grammar_apply_impl()` | 采样前的 **logits**（不进图） |\n'
    '| 模板 | `llm_chat_apply_template()` | **prompt 文本**（决定图上看到什么 token） |')

L.conclusion(
    '★ LoRA 是在哪一步进图的',
    '在 `llama_context` **构建计算图**的那一步（`llama-context.cpp:1425` 的 '
    '`model.build_graph(gparams)`；图能复用时不会重建），由 '
    '`llm_graph_context::build_lora_mm()`（`src/llama-graph.cpp:1514`）完成：\n\n'
    '```text\n'
    'res = ggml_mul_mat(ctx0, w, cur)                       // 基座，w 只读\n'
    'for lora in *loras:                                    // 每套挂着的 adapter\n'
    '    lw = lora.get_weight(w)                            // 按名字查 ab_map\n'
    '    if lw == nullptr: continue\n'
    '    ab = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur))\n'
    '    res = ggml_add(ctx0, res, ggml_scale(ctx0, ab, scale))\n'
    '```\n\n'
    '**它是图上的加法，不是权重里的改写。** 证据有三条：`w` 在函数里只作为 '
    '`ggml_mul_mat` 的输入出现；`*loras` 是一个循环，所以能同时挂多套；'
    '`llama_adapter_lora::get_n_nodes()` 直接把代价写成 `ab_map.size() * 6u`。')

L.conclusion(
    'imatrix 与逐行量化',
    'imatrix 是"每一列有多重要"的**数据**（形状 `ne[0] x ne[2]`，按专家切片）。'
    'IQ 家族必须有它，Q2_K 只在 Q2_K_S 档位下需要，词嵌入与输出层永远豁免。\n\n'
    '量化时一张矩阵按行切成 chunk，行号跨专家全局连续，但 **chunk 不跨专家边界** —— '
    '所以 `imatrix_for_row()` 才能用 `row_global / nrows_per_expert` 直接算出该用哪张切片。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
