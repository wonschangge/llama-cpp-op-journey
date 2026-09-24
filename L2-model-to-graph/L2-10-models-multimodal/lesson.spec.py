#!/usr/bin/env python3
"""L2-10 · 模型家族（一）：多模态与视觉编码 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-10-models-multimodal/lesson.spec.py

本课覆盖的 6 个文件来自【静态扫描】分组（图构建代码里出现 clip_ / vision /
mmproj / image_ 之一），不是"内容判定"。source.md 第一节如实写明分组方法与
逐文件的真实主题；tools/mtmd 下的引用是"视觉塔真正在哪里"的对照，不计入覆盖率。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

# 覆盖域内（src/）：本课负责的 6 个文件
SRC_CLIP   = 'src/models/clip.cpp'
SRC_DSV4   = 'src/models/deepseek4.cpp'
SRC_GEMMA4 = 'src/models/gemma4.cpp'
SRC_PTTS   = 'src/models/pockettts.cpp'
SRC_QWEN4  = 'src/models/qwen4exp.cpp'
SRC_MODELS = 'src/models/models.h'

# 覆盖域外（tools/）：视觉塔的真正实现，仅作对照，不计入覆盖率
SRC_MTMD_CLIP   = 'tools/mtmd/clip.cpp'
SRC_MTMD_GRAPH  = 'tools/mtmd/clip-graph.h'
SRC_MTMD_MAIN   = 'tools/mtmd/mtmd.cpp'
SRC_MTMD_BATCH  = 'tools/mtmd/mtmd-helper-common.h'
SRC_MTMD_GEMMA4 = 'tools/mtmd/models/gemma4v.cpp'

L = Lesson(
    id='L2-10',
    layer='L2 · 从模型到图',
    title='模型家族（一）：多模态与视觉编码',
    codecap='src/models/*（逐字引用）+ tools/mtmd 对照',
    nav={'prev': {'href': '../L2-09-quant-export-adapter/index.html',
                  'label': 'L2-09 量化、导出与适配器'},
         'next': {'href': '../L2-11-models-ssm-linear/index.html',
                  'label': 'L2-11 模型家族（二）状态空间与线性注意力'}},
)

L.note('**一句话**：llama.cpp 里"多模态"不是一种新的图引擎 —— 视觉塔（或音频编码器）'
       '是**另一张 ggml 图**，它算完之后只留下**一个 F32 张量**；这个张量被塞进 '
       '`llama_batch` 的 `embd` 字段，当作"已经算好的 embedding"喂进**普通文本图**的入口。')
L.note('本课覆盖的 6 个文件是**静态扫描**分出来的：图构建代码里出现了 `clip_` / `vision` / '
       '`mmproj` / `image_` 之一。**分组依据是命中了这些图原语，不是对文件内容的断言。**'
       '实测结果（见下方第一节）是：8 处命中全部落在注释或元数据键名上，'
       '`clip_` 在 6 个文件里出现 **0 次**。因此本课既讲"文本图这一侧怎么接多模态"，'
       '也如实说明"谁不是多模态"。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2-10 · 全局',
    title='六个"多模态"文件，只有一个是 <span class="hl-a">clip</span> —— 而它是个存根',
    sub='6 个文件来自静态扫描分组：图构建代码里出现 clip_ / vision / mmproj / image_ 之一。'
        '命中原语不等于"它就是多模态模型"。',
    caption='逐文件的真实主题与实测行数见 source.md 第一节；本幕末尾给出结论。',
    src=SRC_CLIP, parts=[(1, 18)], duration=20000,
    mark_src=[3, 6, 11, 16, 17],
    notes_src={17: '这一行就是本课的第一个事实：CLIP 在 llama 侧没有推理图，运行时在 tools/mtmd/clip.cpp'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'clip.cpp · 18 行', b: '三个 [[noreturn]] 存根函数，<br>没有任何图构建代码', m: 'llama_model_clip' },
  { c: 'c', t: 'models.h · 2662 行', b: '151 个模型结构体的公共声明头，<br>含上面那个存根的类型声明', m: 'struct llama_model_clip' },
  { c: 'd', t: 'deepseek4.cpp · 1502 行', b: 'DeepSeek-V4 文本图（MoE + 压缩注意力）<br>命中：vision variant 的 MoE 路由偏置', m: 'ffn_exp_probs_b_vl' },
  { c: 'b', t: 'gemma4.cpp · 498 行', b: 'Gemma 4 文本图（SWA + 共享 KV）<br>命中：use_bidirectional_attention == "vision"', m: 'non_causal_type' },
  { c: 'e', t: 'qwen4exp.cpp · 1294 行', b: 'Qwen4 文本图（线性注意力 + PLE）<br>命中：一个元数据键 ple_image_token_id', m: 'LLM_KV_PLE_IMAGE_TOKEN_ID' },
  { c: 'f', t: 'pockettts.cpp · 146 行', b: 'PocketTTS 的"文本"主干（无 lm_head）<br>命中：音频 latent 由 mmproj 里的 flow net 产生', m: 'llama_model_pockettts' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');

const msg = wrap.querySelector('#msg');
const texts = [
  '先看结论：<span class="k">这 6 个文件里没有一个是视觉塔</span>。真正的视觉编码器在 tools/mtmd（不在覆盖域）。',
  '<span class="v">clip.cpp</span> 只有 18 行：三处 GGML_ABORT。注释说它是 stub，只为让 llama-quantize 能打开 mmproj GGUF。',
  '<span class="v">models.h</span> 是 6 个文件共用的声明头（6 个文件全都包含它）。存根的类型声明就在这里。',
  '<span class="v">deepseek4.cpp</span> 命中 <span class="m">ffn_exp_probs_b_vl</span>：图像 token 用另一套 MoE 路由偏置。',
  '<span class="v">gemma4.cpp</span> 命中 "<span class="m">vision</span>"：HF 配置里的双向注意力开关，落在 SWA 层上。',
  '<span class="v">qwen4exp.cpp</span> 命中 <span class="m">image</span>：元数据键 <span class="m">ple_image_token_id</span>。',
  '<span class="v">pockettts.cpp</span> 命中 <span class="m">mmproj</span>：音频主干不产 logits，latent 由 mmproj 侧生成。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(900 + i * 2400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15600, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = '6 个文件里 <span class="v">1 个存根 + 1 个声明头 + 4 个文本图</span>。'
    + '多模态的"视觉那一半"要在 <span class="k">tools/mtmd</span> 里看，本课用它做对照。';
});
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-10 · 声明',
    title='<span class="hl-c">models.h</span>：为什么 llama 侧必须留一个空壳',
    sub='架构类都要满足 llama_model_base 的接口；CLIP 也用同一个接口声明，只是三个实现全部 [[noreturn]]。',
    caption='回顾 L2-06：build_arch_graph 返回 llm_graph_context —— 这是所有模型图构建的统一入口。',
    src=SRC_MODELS, parts=[(399, 412)], duration=16000,
    mark_src=[400, 401, 404, 407, 410],
    notes_src={400: '这句注释是本课第一个可核对的事实：这些函数永远不会被调用',
               411: '三个函数都是 [[noreturn]]：CLIP 在 llama 侧没有超参加载、没有张量加载、没有图'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '接口契约', b: '模型基类要求实现三个虚函数：<br>load_arch_hparams / load_arch_tensors / build_arch_graph',
    m: 'struct llama_model_clip : public llama_model_base' },
  { c: 'c', t: '为什么不能干脆删掉', b: 'clip.cpp 的注释写明它存在的唯一目的：<br>让 llama-quantize 能打开 <b>mmproj GGUF</b>',
    m: '// Stub to allow llama-quantize to open mmproj GGUFs' },
  { c: 'e', t: '为什么不能在这里实现', b: '真正的 CLIP 运行时在 tools/mtmd/clip.cpp（不在覆盖域）；<br>在 llama 侧再写一遍就是两套实现',
    m: 'GGML_ABORT' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');

const msg = wrap.querySelector('#msg');
const texts = [
  'models.h 是公共声明头：151 个 <span class="m">struct llama_model_*</span>，每个模型一个。',
  '<span class="v">三个虚函数</span>是所有架构都必须满足的接口：超参、权重张量、图构建。',
  'CLIP 也声明了这个类型，但 <span class="k">三个实现都 [[noreturn]]</span>：它从不被调用。',
  '它存在的唯一理由写在 clip.cpp 的注释里：<span class="k">让 llama-quantize 能打开 mmproj GGUF</span>。',
  '运行时在哪？下一幕看那一行注释指的路。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(900 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  U.markLines(document, [i === 0 ? 3 : (i === 1 ? 1 : 12)]);
  msg.innerHTML = texts[i + 1];
}));
tl.at(13200, () => { els.forEach(e => { e.style.opacity = '1'; }); U.markLines(document, [1, 3, 12, 13]); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-10 · 核心',
    title='★ <span class="hl-a">图像是以"已算好的 embedding"进文本图的</span>',
    sub='源码注释把这句话写死了：an image arrives as an embd batch, so ubatch->token is null。',
    caption='对应 L2-05 的 llama_batch：tokens 与 embd 是二选一的两条输入路径。',
    src=SRC_QWEN4, parts=[(1066, 1077)], duration=18000,
    mark_src=[1069, 1070, 1071, 1076],
    notes_src={1077: 'tok_of() 是本课的关键：没有 token 时用占位 id，说明"这一批不是 token 批"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="card" id="c1" style="width:300px;border-left-color:var(--c)">
      <div class="ct" style="color:var(--c)">token 路径（普通文本）</div>
      <div class="cb">ubatch-&gt;token != nullptr<br>
      每个位置有一个 token id，<br>文本图用 ggml_get_rows 查嵌入表得到输入。</div>
      <div class="cm">ubatch-&gt;token[k]</div>
    </div>
    <div class="card" id="c2" style="width:300px;border-left-color:var(--a)">
      <div class="ct" style="color:var(--a)">embd 路径（图像 / 音频）</div>
      <div class="cb">ubatch-&gt;token == nullptr<br>
      每个位置没有 token id，<br>输入是别人<b>已经算好</b>的一整块 embedding。</div>
      <div class="cm">ubatch-&gt;embd</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const c1 = wrap.querySelector('#c1'), c2 = wrap.querySelector('#c2');
c1.style.opacity = '.30'; c2.style.opacity = '.30';
const msg = wrap.querySelector('#msg');
const texts = [
  'Qwen4 的 PLE 输入要按 n-gram 回溯历史 token；但<b>图像批次里根本没有 token</b>。',
  '所以源码先给出一句判断：<span class="v">an image arrives as an embd batch</span> —— 图像是以 embd 批的形式到达的。',
  '在没有 token id 的情况下，它只能<b>顶替一个 id</b>：元数据里的 ple_image_token_id，缺省用 EOS。',
  '最后一行把这个约定收成一个三元表达式：<br><span class="v">ubatch-&gt;token ? ubatch-&gt;token[k] : img_tok</span>。',
  '结论：<span class="k">文本图并不区分"文字"和"图像"</span> —— 它只区分"有 token id"和"有现成 embedding"。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; });
tl.at(3000, () => { c1.style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [3, 4]); });
tl.at(6600, () => { msg.innerHTML = texts[2]; U.markLines(document, [5, 6]); });
tl.at(10200, () => { c2.style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, [10]); });
tl.at(14000, () => { c1.style.opacity = '1'; msg.innerHTML = texts[4]; U.markLines(document, [3, 4, 10]); });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-10 · 核心',
    title='★ 文本图在入口处分叉：<span class="hl-b">ubatch.token ? sqrtf(n_embd) : 1.0f</span>',
    sub='Gemma 4 用同一个指针判断"这一批是 token 还是图像 embedding"，并据此决定要不要缩放。',
    caption='"encoded image emdeddings" 是源码注释原话（原文拼写如此）。这句判断的位置就在图的入口。',
    src=SRC_GEMMA4, parts=[(161, 165)], duration=15000,
    mark_src=[164],
    notes_src={164: 'ubatch.token == nullptr 就是"这一批是图像 embedding"的判据'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip a">build_inp_embd(tok_embd)</span>
    <span class="arrow">-&gt;</span>
    <span class="chip c">ubatch.token ? sqrtf(n_embd) : 1.0f</span>
    <span class="arrow">-&gt;</span>
    <span class="chip b">inpL</span>
  </div>
  <div class="row" style="gap:9px">
    <div class="card" id="c1" style="width:300px;border-left-color:var(--c)">
      <div class="ct" style="color:var(--c)">token 批：要缩放</div>
      <div class="cb">token 嵌入要乘 sqrt(n_embd) —— 这是 Gemma 的嵌入归一化约定。</div>
      <div class="cm">ubatch.token ? sqrtf(n_embd)</div>
    </div>
    <div class="card" id="c2" style="width:300px;border-left-color:var(--b)">
      <div class="ct" style="color:var(--b)">embd 批：不缩放</div>
      <div class="cb">图像 embedding 已经是"算好的"，<b>不能再乘一次</b>，所以因子取 1.0f。</div>
      <div class="cm">ubatch.token ? sqrtf(n_embd) : 1.0f</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const c1 = wrap.querySelector('#c1'), c2 = wrap.querySelector('#c2');
c1.style.opacity = '.30'; c2.style.opacity = '.30';
const msg = wrap.querySelector('#msg');
const texts = [
  '同一行里出现了两个分支：<span class="m">ubatch.token ? sqrtf(n_embd) : 1.0f</span>。',
  '有 token：按 Gemma 的约定乘 <span class="v">sqrt(n_embd)</span>。',
  '没有 token（图像 embedding）：乘 <span class="v">1.0f</span>，即原样通过。',
  '注释写明理由：<span class="k">raw embeddings input (i.e. encoded image emdeddings)</span> 不做归一化。',
  '两个分支<strong>汇合到同一个 inpL</strong>：后面的层完全不知道输入是文字还是图像。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [3]); });
tl.at(3200, () => { c1.style.opacity = '1'; msg.innerHTML = texts[1]; });
tl.at(6200, () => { c2.style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [2]); });
tl.at(9400, () => { msg.innerHTML = texts[3]; });
tl.at(12200, () => { c1.style.opacity = '1'; c2.style.opacity = '1'; msg.innerHTML = texts[4]; U.markLines(document, [0, 3]); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-10 · 视觉塔',
    title='视觉塔是<b>另一张 ggml 图</b>：先 conv2d 切 patch',
    sub='tools/mtmd 里的视觉塔有自己的 ggml_context 与 cgraph；它与文本图不共享节点，只共享"embedding 张量"这个概念。',
    caption='clip-graph.h 的契约写着：build_inp() 返回 shape [n_embd, n_patches]（见 source.md 第三节）。',
    src=SRC_MTMD_GEMMA4, parts=[(4, 16)], duration=16000,
    mark_src=[12, 13, 14],
    notes_src={14: '转置 + cont 之后 ne = [n_embd, n_patches]，与 clip-graph.h 的契约一致'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow wrap" id="chain" style="gap:6px">
    <span class="chip a">inp_raw 像素</span>
    <span class="arrow">-&gt;</span>
    <span class="chip c">scale_bias x2 -1</span>
    <span class="arrow">-&gt;</span>
    <span class="chip d">conv_2d(patch_size, patch_size, stride 1)</span>
    <span class="arrow">-&gt;</span>
    <span class="chip e">reshape_3d [n_patches, n_embd, n_batch]</span>
    <span class="arrow">-&gt;</span>
    <span class="chip b">transpose + cont [n_embd, n_patches]</span>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  '视觉塔的入口是 <span class="v">build_inp_raw()</span>：一张原始图像张量（HWC 像素）。',
  'gemma4v 的注释给出归一化公式 <span class="m">patches * 2 - 1</span>，用一次 scale_bias 完成。',
  '<span class="v">ggml_conv_2d</span> 用 patch_size 的核、步长 1 切出 patch —— 这就是 patch embedding。',
  '卷积结果的维度顺序是 <span class="m">[n_patches, n_embd, n_batch]</span>。',
  '转置 + cont 之后变成 <span class="k">ne = [n_embd, n_patches]</span>：<br>ne[0] 是特征宽度，ne[1] 是 patch 数 —— 与 L1-01 的 ne 约定一致。',
  '从这里开始，整条塔都在 <span class="m">[n_embd, n_patches]</span> 这个布局上推进。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [1]); });
tl.at(3200, () => { msg.innerHTML = texts[1]; U.markLines(document, [3, 4, 5]); });
tl.at(6400, () => { msg.innerHTML = texts[2]; U.markLines(document, [8]); });
tl.at(9400, () => { msg.innerHTML = texts[3]; U.markLines(document, [9]); });
tl.at(12200, () => { msg.innerHTML = texts[4]; U.markLines(document, [10]); });
tl.at(14600, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-10 · 视觉塔',
    title='★ 形状全链：<span class="hl-a">[n_embd, n_patches]</span> 到 <span class="hl-a">[n_mmproj_embd, n_tokens]</span>',
    sub='每一步的 ne 都来自源码：ViT -> pooler -> projector，最后成为文本图入口的宽度。',
    caption='术语回顾 L1-01：ne[0] 是最内层维度。这里 ne[0] 是特征宽度，ne[1] 是 token 数。',
    src=SRC_MTMD_GEMMA4, parts=[(76, 118)], duration=22000,
    mark_src=[76, 90, 95, 111, 112, 116],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="flow wrap" id="chain" style="gap:5px">
    <span class="chip a">[n_embd, n_patches]</span>
    <span class="arrow">-&gt;</span>
    <span class="chip d">build_vit x n_layer</span>
    <span class="arrow">-&gt;</span>
    <span class="chip c">kernel_size = hparams.n_merge</span>
    <span class="arrow">-&gt;</span>
    <span class="chip e">[n_embd, out_x * out_y]</span>
    <span class="arrow">-&gt;</span>
    <span class="chip f">mm_input_proj_w</span>
    <span class="arrow">-&gt;</span>
    <span class="chip b">[n_mmproj_embd, n_tokens]</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['步骤', '张量 ne', '出处（逐字引用见 source.md）'],
  [['patch 嵌入后', '[n_embd, n_patches]', 'clip-graph.h:98-100 / models/gemma4v.cpp:13-14'],
   ['ViT 输出', '[n_embd, n_patches]（形状不变）', 'models/gemma4v.cpp:76-96'],
   ['pooler 之后', '[n_embd, out_x * out_y]', 'models/gemma4v.cpp:92-96'],
   ['projector 之后（图的最后一个节点）', '[n_mmproj_embd, n_tokens]', 'models/gemma4v.cpp:111-113 / clip.cpp:5771-5782'],
   ['host 侧扁平缓冲', 'n_mmproj_embd * n_tokens 个 float', 'mtmd.cpp:1781-1783'],
   ['文本图入口张量', '[n_embd, n_tokens]', 'src/models/gemma4.cpp:475'],
   ['宽度约束', 'n_mmproj_embd == n_embd_inp', 'mtmd.cpp:603-608']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '从 <span class="v">[n_embd, n_patches]</span> 开始：n_embd 是视觉塔的隐藏宽度，n_patches 是 patch 数。',
  'ViT 层（<span class="m">build_vit</span>）只改数值，<b>不改形状</b>。',
  'pooler 用 <span class="m">n_merge</span> 的核做平均池化：token 数从 n_patches 降到 <span class="v">out_x * out_y</span>。',
  'projector（<span class="m">mm_input_proj_w</span>）把宽度从 n_embd 换成 <span class="v">n_mmproj_embd</span>。',
  '它同时是<b>整张视觉图的最后一个节点</b>：clip.cpp 直接取 gf 的最后一个节点，并用 ne[1] 校验 token 数。',
  'mtmd 把它拷成一条扁平 float 缓冲：<span class="m">n_embd_out * n_tokens_out</span> 个元素。',
  '文本图那侧看到的输入张量是 <span class="v">[n_embd, n_tokens]</span> —— 宽度必须等于 n_mmproj_embd。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [0]); });
tl.at(3000, () => { msg.innerHTML = texts[1]; U.markLines(document, [0]); });
rows.forEach((r, i) => tl.at(5400 + i * 2200, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 2, 6)];
  U.markLines(document, i < 4 ? [0, 14, 19, 35, 36, 40].slice(0, i + 1) : [40]);
}));
tl.at(20800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-10 · 拼接',
    title='拼接点：<span class="hl-e">llama_batch</span> 的 <span class="hl-e">embd</span> 字段',
    sub='视觉塔的输出被拷成一条扁平 float 缓冲，再作为一个"没有 token 的 batch"交给 llama_decode。',
    caption='mtmd 启动时会断言 n_mmproj_embd == 文本模型的 n_embd_inp，否则直接报错（source.md 第五节）。',
    src=SRC_MTMD_BATCH, parts=[(73, 100)], duration=17000,
    mark_src=[92, 93, 94],
    notes_src={94: 'tokens 为 nullptr、embd 指向视觉塔输出 —— "拼接"就发生在这三行'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" style="gap:10px">
    <div class="card" id="c1" style="width:290px;border-left-color:var(--a)">
      <div class="ct" style="color:var(--a)">视觉塔侧</div>
      <div class="cb">out_embd：一条扁平 float 缓冲<br>长度 = n_embd_out * n_tokens_out<br>
      （每 token 一段连续的特征）</div>
      <div class="cm">std::vector&lt;float&gt; out_embd</div>
    </div>
    <span class="arrow">-&gt;</span>
    <div class="card" id="c2" style="width:290px;border-left-color:var(--e)">
      <div class="ct" style="color:var(--e)">文本图侧</div>
      <div class="cb">llama_batch：tokens = nullptr<br>embd = 上面那条缓冲的指针<br>
      n_tokens = 图像 token 数</div>
      <div class="cm">llama_batch batch</div>
    </div>
  </div>
  <div class="flow" style="justify-content:center">
    <span class="chip c">n_mmproj_embd = llama_model_n_embd_inp(model)</span>
    <span class="arrow">-&gt;</span>
    <span class="chip b">llama_decode(lctx, batch_embd_view)</span>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const c1 = wrap.querySelector('#c1'), c2 = wrap.querySelector('#c2');
c1.style.opacity = '.30'; c2.style.opacity = '.30';
const msg = wrap.querySelector('#msg');
const texts = [
  '视觉塔算完，留在 host 内存里的只有一条 float 缓冲。',
  '它被交给一个专门的小结构：<span class="v">decode_embd_batch</span>。',
  '构造函数里三行是本课的拼接点：<br><span class="k">n_tokens / tokens = nullptr / embd = embd</span>。',
  'tokens 为 nullptr 是<b>合法</b>的：文本图的入口会走 embd 分支（第 3、4 幕）。',
  '宽度由 <span class="m">llama_model_n_embd_inp(model)</span> 给出：图像 embedding 的宽度必须与文本模型一致。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; });
tl.at(3000, () => { c1.style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [9]); });
tl.at(6600, () => { c2.style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, [19, 20, 21]); });
tl.at(10200, () => { msg.innerHTML = texts[3]; U.markLines(document, [20]); });
tl.at(13500, () => { msg.innerHTML = texts[4]; U.markLines(document, [11]); });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-10 · 家族差异',
    title='同一个 LLM，四种"多模态"接法',
    sub='判据都不是 token id，而是 ubatch.embd 这个指针 —— 图在构建时就知道自己是不是在处理媒体输入。',
    caption='L2-11~L2-15 继续看其它家族：SSM 与线性注意力、稀疏 MoE、稠密 Transformer。',
    src=SRC_DSV4, parts=[(1280, 1293)], duration=17000,
    mark_src=[1284, 1285, 1288, 1291],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="formula" id="msg"></div>
  <div class="row wrap" id="cards" style="gap:8px"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'd', t: 'deepseek4', b: 'ubatch.embd != nullptr 时换用 <b>ffn_exp_probs_b_vl</b>：<br>图像 token 走另一套 MoE 路由偏置',
    m: 'is_media = ubatch.embd != nullptr' },
  { c: 'b', t: 'gemma4', b: 'HF 配置 use_bidirectional_attention == "vision" 时，<br>SWA 层改双向注意力',
    m: 'non_causal_type' },
  { c: 'e', t: 'qwen4exp', b: '图像批次没有 token id，用 <b>ple_image_token_id</b><br>（缺省 EOS）顶替',
    m: 'ubatch->token ? ubatch->token[k] : img_tok' },
  { c: 'f', t: 'pockettts', b: '音频侧反过来：主干不产 logits，<br>latent 由 mmproj 里的 flow net 生成',
    m: 'no lm_head' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');

const msg = wrap.querySelector('#msg');
const texts = [
  'DeepSeek-V4 的图在 MoE 层前问一句：<span class="v">is_media = ubatch.embd != nullptr</span>。',
  '有媒体输入：用 vision variant 的路由偏置张量；<br>否则：哈希路由层用 token id 查表，其余层用普通偏置。',
  '同一个 LLM 主干，<span class="k">只在少数节点上分叉</span>。',
  '其它家族的"多模态钩子"各不相同：双向注意力开关、占位 token id、音频 latent 生成。',
  '共同点只有一个：<span class="k">媒体输入都被翻译成"已算好的 embedding"</span>，从 embd 路径进图。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; U.markLines(document, [4, 5]); });
tl.at(3400, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [8, 11]); });
tl.at(7000, () => { msg.innerHTML = texts[2]; });
defs.forEach((_, i) => { if (i > 0) tl.at(9200 + (i - 1) * 2400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[3];
}); });
tl.at(15200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-10 · 收束',
    title='把这一课压成一张表',
    sub='验收点：视觉塔的输出以什么形状喂给 LLM 部分。',
    caption='下一课 L2-11 模型家族（二）：状态空间与线性注意力。',
    src=SRC_GEMMA4, parts=[(473, 477)], duration=18000,
    mark_src=[475],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', '答案（每一格都能在源码里指到）'],
  [['视觉塔输出什么形状', 'ne = [n_mmproj_embd, n_tokens]，F32；它就是视觉图 gf 的最后一个节点'],
   ['怎么喂给 LLM 部分', '拷成 host 侧扁平缓冲，填进 llama_batch.embd，且 tokens = nullptr'],
   ['宽度谁保证一致', 'mtmd 断言 mmproj 的 n_mmproj_embd == 文本模型的 n_embd_inp'],
   ['文本图看到什么', '入口张量 [n_embd, n_tokens]，与 token 路径在同一处汇合'],
   ['token 数谁定', 'clip_n_output_tokens()；clip.cpp 用 embeddings->ne[1] 校验']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '一张图像经过 mmproj 之后，喂给文本图的张量是什么形状？它和普通 token 路径在哪一步汇合？',
  '形状是 <span class="mono">ne = [n_mmproj_embd, n_tokens]</span> 的 F32 张量：'
  + '<span class="mono">ne[0]</span> 是投影后的特征宽度，<span class="mono">ne[1]</span> 是这张图产生的 token 数'
  + '（clip.cpp:5776 用 <span class="mono">embeddings-&gt;ne[1]</span> 校验）。<br>'
  + '它<b>不走</b> <span class="mono">ggml_get_rows(tok_embd, tokens)</span>，而是被拷成一条扁平 float 缓冲、'
  + '填进 <span class="mono">llama_batch.embd</span>（此时 <span class="mono">tokens = nullptr</span>）。<br>'
  + '文本图在入口 <span class="mono">build_inp_embd</span> 处得到 <span class="mono">[n_embd, n_tokens]</span> 的输入张量，'
  + 'token 路径与 embd 路径就在这里汇合 —— 后面的层分不清输入是文字还是图像。<br>'
  + '前提是 <span class="mono">n_mmproj_embd == n_embd_inp</span>，否则 mtmd 在启动时直接报错。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '视觉塔的输出不是一个"特别的类型"，就是一个普通张量：<span class="v">ne = [n_mmproj_embd, n_tokens]</span>。',
  '它进文本图的通道是 <span class="k">llama_batch.embd</span> —— L2-05 讲过的那条"现成 embedding"路径。',
  '宽度必须等于文本模型的 <span class="v">n_embd_inp</span>；文本图入口的输入张量写着 <span class="m">[n_embd, n_tokens]</span>。',
  '所以：<span class="k">多模态不是新引擎，而是"多张图接在一起"</span>。',
  '下一课看另一类家族：状态空间与线性注意力 —— 它们的"记忆"在图上长什么样。'
];
tl.at(400, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(14800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、本课的分组方法（先读这一节）',
    '本课覆盖的 6 个文件**不是按内容选的**，而是**静态扫描**的结果：'
    '凡是图构建代码里出现了 `clip_` / `vision` / `mmproj` / `image_` 之一的 `src/models/*.cpp|h`，'
    '就被分到"多模态与视觉编码"这一课。**分组依据是"命中了这些图原语"，'
    '不是"这个文件就是多模态模型"。**\n\n'
    '我把这 6 个文件逐个读了一遍，实测结果如下。行数来自 `wc -l`；README 的文件表由生成器'
    '按另一种口径计数（`split("\\n")` 的元素个数），每个文件会多 1 —— 两处数字对得上，只是口径不同：\n\n'
    '| 文件 | 行数 | 命中处（原文所在行） | 真实主题 | 是多模态吗 |\n'
    '|---|---|---|---|---|\n'
    '| `src/models/clip.cpp` | 18 | 第 3 行注释里的 `mmproj` | CLIP 的**量化存根**：三个 `[[noreturn]]` 函数，没有任何图 | **不是**（真正的运行时在 `tools/mtmd/clip.cpp`） |\n'
    '| `src/models/models.h` | 2662 | 第 399 行注释里的 `mmproj` | **公共声明头**：151 个 `struct llama_model_*` 与图基类声明 | **不是**（是被所有模型共用的头文件） |\n'
    '| `src/models/deepseek4.cpp` | 1502 | 第 162 行注释里的 `vision` / `image` | DeepSeek-V4 文本图（MoE + 压缩注意力 + MTP） | 部分：**有 vision variant 的钩子** |\n'
    '| `src/models/gemma4.cpp` | 498 | 第 23 行注释里的 `"vision"` | Gemma 4 文本图（SWA + 共享 KV + per-layer 嵌入） | 部分：**有图像输入的分支** |\n'
    '| `src/models/pockettts.cpp` | 146 | 第 4 行注释里的 `mmproj` | PocketTTS 的文本主干（无 lm_head） | 部分是：**音频侧**，不是视觉 |\n'
    '| `src/models/qwen4exp.cpp` | 1294 | 第 90、1072、1073 行的 `ple_image_token_id` | Qwen4 文本图（线性注意力 + PLE n-gram 嵌入） | 部分：**有图像批次的分支** |\n\n'
    '两条要如实说清楚的事实：\n\n'
    '1. **8 处命中里，5 处出现在注释中**（`clip.cpp:3`、`models.h:399`、`deepseek4.cpp:162`、'
    '`gemma4.cpp:23`、`pockettts.cpp:4`），**剩下 3 处都在 `qwen4exp.cpp`，且都是同一个元数据键名 '
    '`ple_image_token_id`**（第 90、1072、1073 行）。**没有一处是图原语调用。**\n'
    '2. 扫描模式里的 `clip_` 在这 6 个文件里出现 **0 次**。6 个文件全部只 `#include "models.h"`'
    '（`clip.cpp` / `gemma4.cpp` / `pockettts.cpp` / `qwen4exp.cpp` 在第 1 行，`deepseek4.cpp` 在第 2 行），'
    '**没有一个是视觉编码器**。\n\n'
    '**所以本课讲两件事**：（a）多模态在**文本图这一侧**是怎么被接进来的（这 6 个文件里能读到）；'
    '（b）视觉塔本身长什么样 —— 它在 `tools/mtmd/`，**不在本仓库的覆盖域内**，'
    '本课引用它只作对照，并在本节末尾与脚注里明确标注**不计入覆盖率**。')

L.section(
    '二、clip.cpp：一个 [[noreturn]] 的存根',
    '这是本课最短的文件，也是最能说明问题的一个。它给出三条 `GGML_ABORT`，'
    '并在注释里写明自己的身份（`Stub to allow llama-quantize to open mmproj GGUFs`），'
    '最后一行直接指路：**CLIP 在 llama 分发路径里没有推理图，运行时在 `tools/mtmd/clip.cpp`**。',
    src=SRC_CLIP, parts=[(3, 18)], lang='c')

L.section(
    '三、视觉塔的契约（tools/mtmd/clip-graph.h）',
    '视觉塔的公共基类 `clip_graph` 把两件事写进了注释：ViT 的通用构建函数 `build_vit`，'
    '以及输入构建函数 `build_inp()` 的返回形状 —— **`[n_embd, n_patches]`**。'
    '另外，投影后的宽度 `n_mmproj_embd` 是基类的一个字段（`clip-graph.h:43`）。\n\n'
    '> 这一节的文件在 `tools/`，**不在覆盖域，不计入覆盖率**。',
    src=SRC_MTMD_GRAPH, parts=[(88, 100)], lang='c')

L.section(
    '四、视觉塔图的出口：最后一个节点就是 embedding（tools/mtmd/clip.cpp）',
    '视觉塔算完之后，代码并不去"找名字"，而是**取这张图的最后一个节点**当输出，'
    '并用它的 `ne[1]` 校验 token 数。这是"视觉塔输出一个张量"这句话最直接的证据。\n\n'
    '> 同样在 `tools/`，不计入覆盖率。',
    src=SRC_MTMD_CLIP, parts=[(5771, 5786)], lang='c')

L.section(
    '五、两个 n_embd 必须相等（tools/mtmd/mtmd.cpp）',
    'mtmd 在建上下文时就断言：mmproj 的输出宽度必须等于文本模型的输入宽度。'
    '不相等就直接抛错（提示语就是"你多半用错了 mmproj"）。\n\n'
    '> 同样在 `tools/`，不计入覆盖率。',
    src=SRC_MTMD_MAIN, parts=[(601, 609)], lang='c')

L.section(
    '六、拷成一条扁平缓冲（tools/mtmd/mtmd.cpp）',
    '视觉塔的输出张量被拷成一条扁平 `std::vector<float>`，长度 = 宽度 x token 数。'
    '这条缓冲随后就是 `llama_batch.embd` 指向的内存。\n\n'
    '> 同样在 `tools/`，不计入覆盖率。',
    src=SRC_MTMD_MAIN, parts=[(1774, 1798)], lang='c')

L.section(
    '七、音频侧：pockettts 的文本主干',
    'PocketTTS 走的是同一条思路的另一半：llama 侧只跑"文本"主干，'
    '**音频 latent 由 mmproj 里的 flow net 生成**。主干甚至有输出头也不产 logits —— '
    '源码直接复用了嵌入表，只为让采样器还能跑起来。',
    src=SRC_PTTS, parts=[(3, 24)], lang='c')

L.section(
    '八、gemma4：per-layer 输入的多模态分支',
    'Gemma 4 的 per-layer 嵌入需要按 token 查表。图像批次没有 token id，'
    '于是源码走"多模态 embedding 路径"：用 padding token（ID=0）的嵌入，'
    '并保留了"这未必与 transformers 实现一致"的 TODO。',
    src=SRC_GEMMA4, parts=[(456, 470)], lang='c')

L.section(
    '九、qwen4exp：图像 token 的 id 来自元数据',
    'Qwen4 的图像占位 id 是一个**可选**元数据键：老文件里没有它，'
    '于是回退到 EOS token。这就是第 3 幕那个 `img_tok` 的来历。',
    src=SRC_QWEN4, parts=[(88, 90)], lang='c')

L.section(
    '十、deepseek4：vision variant 的 MoE 路由偏置',
    'DeepSeek-V4 在每个 MoE 层上多声明了一个**可选**张量：vision variant 的路由偏置。'
    '图构建时用 `ubatch.embd` 判断"这一批是不是媒体输入"，是就换用它。',
    src=SRC_DSV4, parts=[(156, 163)], lang='c')

L.footnote_add('**覆盖声明**：本课覆盖域内（`src/`）的文件是 6 个 —— `src/models/clip.cpp`、'
               '`src/models/deepseek4.cpp`、`src/models/gemma4.cpp`、`src/models/pockettts.cpp`、'
               '`src/models/qwen4exp.cpp`、`src/models/models.h`。它们计入覆盖率。')
L.footnote_add('**不计入覆盖率**：本课另外引用了 `tools/mtmd/` 下的 5 个文件'
               '（`clip.cpp`、`clip-graph.h`、`mtmd.cpp`、`mtmd-helper-common.h`、`models/gemma4v.cpp`）'
               '作为"视觉塔真正在哪里"的对照。按 `tools/repo_universe.py` 的 `CORE_PATTERNS`，'
               '`tools/` 不在覆盖域内 —— 这些引用**被引用但不计入覆盖率**，此处明确声明，不虚报。')
L.footnote_add('**分组来源**：本课 6 个文件来自静态扫描（图构建代码里出现 `clip_` / `vision` / '
               '`mmproj` / `image_` 之一）。实测：8 处命中中 5 处在注释里，3 处是 `qwen4exp.cpp` 的'
               '同一个元数据键名，**没有一处是图原语调用**；`clip_` 在 6 个文件里出现 0 次。'
               '因此本课不声称"这 6 个文件都是多模态模型"。')
L.footnote_add('第 5~7 幕的形状链逐字引自 `tools/mtmd` 的代码；本课只解读形状流转，'
               '视觉塔的完整实现（40+ 个模型文件）不在本课范围内。')

L.prereqs('`L2-06`（★ 计算图骨架 llama-graph）')

L.goal(
    '说出本课 6 个文件各自的真实主题，以及它们为什么会落到"多模态"这一课（对应分组说明）；',
    '说出视觉塔的输出以什么形状喂给 LLM 部分（对应验收点）；',
    '解释 `llama_batch` 的 token 路径与 embd 路径在文本图的哪一步汇合；',
    '举出至少两个"文本图在构建时就区分媒体输入"的源码位置。')

L.conclusion(
    '★ 视觉塔的输出形状',
    '视觉塔是一张**独立的 ggml 图**。它的最后一个节点就是输出张量：\n\n'
    '```text\n'
    'ne = [n_mmproj_embd, n_tokens]      // F32\n'
    'ne[0] = n_mmproj_embd  投影后的特征宽度\n'
    'ne[1] = n_tokens       这张图产生的 token 数\n'
    '```\n\n'
    '这个张量被拷成一条扁平 float 缓冲（长度 `n_mmproj_embd * n_tokens`），'
    '填进 `llama_batch.embd`（此时 `tokens = nullptr`），'
    '并在文本图入口变成 `[n_embd, n_tokens]` 的输入张量。'
    '**宽度相等的约束由 mtmd 在启动时断言：`n_mmproj_embd == n_embd_inp`。**')

L.conclusion(
    '★ 多模态不是新引擎，而是"多张图接在一起"',
    '视觉塔有自己的 `ggml_context`、自己的 `ggml_cgraph`、自己的后端调度；'
    '它与文本图**不共享任何节点**，只共享一个交汇点：**一个已算好的 embedding 张量**。'
    '文本图对这件事的感知只有一处 —— 入口是 `ubatch.token` 还是 `ubatch.embd`。'
    '这与 L2-05 的 `llama_batch`（tokens / embd 二选一）是同一件事的两端。')

L.conclusion(
    '本课 6 个文件的真实主题',
    '| 文件 | 行数 | 真实主题 | 与多模态的关系 |\n|---|---|---|---|\n'
    '| `src/models/clip.cpp` | 18 | CLIP 量化存根 | 只是"让 llama-quantize 能打开 mmproj GGUF" |\n'
    '| `src/models/models.h` | 2662 | 151 个模型结构体的公共声明头 | 存根的类型声明在这里 |\n'
    '| `src/models/deepseek4.cpp` | 1502 | DeepSeek-V4 文本图 | vision variant 的 MoE 路由偏置 |\n'
    '| `src/models/gemma4.cpp` | 498 | Gemma 4 文本图 | 图像 embedding 不做嵌入缩放的入口分支 |\n'
    '| `src/models/pockettts.cpp` | 146 | PocketTTS 文本主干 | 音频侧：latent 由 mmproj 的 flow net 产生 |\n'
    '| `src/models/qwen4exp.cpp` | 1294 | Qwen4 文本图 | 图像批次用占位 token id 顶替 |\n\n'
    '**只有 `clip.cpp` 与"视觉"直接同名，而它恰恰是唯一没有图的那个。**')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
