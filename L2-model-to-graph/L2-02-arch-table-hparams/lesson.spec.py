#!/usr/bin/env python3
"""L2-02 · 架构表与超参：模型长什么样的元数据 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-02-arch-table-hparams/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

ARCH_C = 'src/llama-arch.cpp'
ARCH_H = 'src/llama-arch.h'
HP_C   = 'src/llama-hparams.cpp'
HP_H   = 'src/llama-hparams.h'
CP_C   = 'src/llama-cparams.cpp'
CP_H   = 'src/llama-cparams.h'

L = Lesson(
    id='L2-02',
    layer='L2 · 从模型到图',
    title='架构表与超参：模型长什么样的元数据',
    codecap='src/llama-arch.{cpp,h} / llama-hparams.{cpp,h} / llama-cparams.{cpp,h}（逐字引用）',
    nav={'prev': {'href': '../L2-01-llama-h-api/index.html', 'label': 'L2-01 llama.h 公共 API'},
         'next': {'href': '../L2-03-weight-loading-mmap/index.html', 'label': 'L2-03 权重加载与内存映射'}},
)

L.note('**一句话**：在 llama.cpp 里，"这是哪个模型"由 `enum llm_arch` 回答，'
       '"这个模型长什么样"由 `struct llama_hparams` 回答，'
       '"这次运行怎么配置"由 `struct llama_cparams` 回答。三者都在同一批文件里。')
L.note('本课是 L2 层的地基：后面每一课——加载权重（L2-03）、装配模型（L2-05）、构建计算图（L2-06）'
       '——都要先问这三个问题。而且**张量的名字不是拼字符串拼出来的，是查表查出来的**，'
       '这一点决定了"新增一个模型架构"到底要改哪几处。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2 · 从模型到图',
    title='一个模型"长什么样"，写在三张表和两个结构里',
    sub='架构表回答"这是哪个家族"，hparams 回答"模型的形状"，cparams 回答"这次怎么跑"。',
    caption='回顾 L1-06：GGUF 是 hparams 的来源。指路 L2-03（按张量名加载权重）与 L2-06（图构建读 hparams 定形状）。',
    src=ARCH_H, parts=[(13, 21)], duration=17000,
    mark_src=[13, 14, 15],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">GGUF 文件</span><span class="arrow">-></span>
    <span class="chip a">架构表</span><span class="arrow">-></span>
    <span class="chip b">hparams</span><span class="arrow">-></span>
    <span class="chip c">cparams</span><span class="arrow">-></span>
    <span class="chip d">ggml 图</span>
  </div>
  <div class="row center" id="cards" style="gap:11px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: '架构表 · enum llm_arch', b: '模型家族的编号。<br>GGUF 里 general.architecture<br>的字符串对应它。', m: 'src/llama-arch.h:13-167' },
  { c: 'b', t: 'hparams · 模型的形状', b: '层数、宽度、头数、FFN。<br><b>从 GGUF 读来，加载后不可改</b>。', m: 'src/llama-hparams.h:52-510' },
  { c: 'c', t: 'cparams · 这次怎么跑', b: 'n_ctx、n_batch、线程数。<br><b>来自 llama_context_params</b>，<br>每个 context 都能不同。', m: 'src/llama-cparams.h:10-67' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '本课只回答一个问题：<span class="k">一个模型"长什么样"，写在哪儿？</span>',
  '<span class="v">架构表</span>回答"这是哪个家族" —— 152 个家族，每个有一张字符串身份证。',
  '<span class="v">hparams</span>回答"模型的形状" —— 从 GGUF metadata 读出来，<span class="k">加载后不可改</span>。',
  '<span class="v">cparams</span>回答"这次怎么跑" —— 来自 <span class="v">llama_context_params</span>，<span class="k">每个 context 都可以不一样</span>。',
  '回顾 <span class="k">L1-06</span>：GGUF 容器是 hparams 的来源。<br>下一课 <span class="k">L2-03</span> 讲权重如何按张量名加载，<span class="k">L2-06</span> 讲图构建怎么读 hparams 定层数与形状。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3200 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(13400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-02 · 架构表',
    title='<span class="hl-a">153</span> 条名字，一张"身份证"表',
    sub='enum llm_arch 里 152 个真架构 + 1 个哨兵；名字表正好 153 条。',
    caption='枚举在 src/llama-arch.h:13-167；名字表在 src/llama-arch.cpp:8-162；反查函数 llm_arch_from_string 在同文件 1047 行。',
    src=ARCH_C, parts=[(8, 17)], duration=19000,
    mark_src=[8, 9, 10],
    notes={1: '第 1 行的注释点明：clip 是 dummy，只给 llama-quantize 用'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip c">general.architecture</span><span class="arrow">=</span>
    <span class="chip a">"llama"</span><span class="arrow">-></span>
    <span class="chip b">llm_arch_from_string()</span><span class="arrow">-></span>
    <span class="chip d">LLM_ARCH_LLAMA</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: 'enum llm_arch', b: '153 个成员 =<br>152 个真架构 + LLM_ARCH_UNKNOWN', m: 'src/llama-arch.h:13-167' },
  { c: 'b', t: 'LLM_ARCH_NAMES', b: '<b>字符串就是 GGUF 里<br>general.architecture 的值</b>', m: 'src/llama-arch.cpp:8-162' },
  { c: 'c', t: 'llm_arch_from_string()', b: '拿字符串反查枚举；<br>查不到返回 LLM_ARCH_UNKNOWN', m: 'src/llama-arch.cpp:1047' },
  { c: 'd', t: 'llm_arch_is_*()', b: 'recurrent / hybrid / diffusion<br>三种记忆形态的查询', m: 'src/llama-arch.cpp:1061-1108' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '先看表本身：<span class="v">std::map&lt;llm_arch, const char *&gt;</span>，一行一个家族。',
  '枚举本身是 <span class="k">153 个成员</span>：152 个真架构，加上末尾的哨兵 <span class="v">LLM_ARCH_UNKNOWN</span>（src/llama-arch.h:166）。',
  '这张表一共 <span class="k">153 条</span>，与枚举一一对应。<br>第 9 行的注释点出一个特例：<span class="v">clip</span> 是 dummy，只给 llama-quantize 用。',
  '表的字符串<b>直接对应 GGUF 的 <span class="v">general.architecture</span></b>。<br>模型文件里写 "llama"，这里就认领 <span class="v">LLM_ARCH_LLAMA</span>。',
  '<span class="v">llm_arch_from_string()</span>（1047 行）线性扫这张表，查不到就返回哨兵。<br>权重加载器用它把架构字符串变成枚举。',
  '同一张表还喂给 <span class="v">llm_arch_all()</span>（1031 行）和 <span class="v">llm_arch_name()</span>（1040 行）。<br><span class="k">一张表，三个出口。</span>'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3000 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-02 · 核心',
    title='★ <span class="hl-c">%s</span>：GGUF 的 key 是"算"出来的',
    sub='LLM_KV::operator() 把 LLM_KV_NAMES 的模板和 LLM_ARCH_NAMES 的字符串拼成真正的 metadata key。',
    caption='LLM_KV_NAMES 共 243 条（src/llama-arch.cpp:164-431），带 %s 前缀的那些才是架构相关的。',
    src=ARCH_C, parts=[(1000, 1011)], duration=19000,
    mark_src=[1002, 1003],
    notes={3: '::format 把 LLM_KV_NAMES 的模板套上 LLM_ARCH_NAMES 的架构名'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="col" id="steps" style="gap:5px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const steps = [
  { l: 'LLM_KV_NAMES[LLM_KV_CONTEXT_LENGTH]', r: '"%s.context_length"', c: 'c' },
  { l: 'LLM_ARCH_NAMES[LLM_ARCH_LLAMA]', r: '"llama"', c: 'a' },
  { l: 'LLM_KV(LLM_ARCH_LLAMA)(LLM_KV_CONTEXT_LENGTH)', r: '"llama.context_length"', c: 'b' },
  { l: 'src/llama-model.cpp:1254 用它读 GGUF', r: 'hparams.n_ctx_train', c: 'e' }
];
const host = wrap.querySelector('#steps');
const els = steps.map(s => {
  const e = U.el('div', { class: 'formula', style: 'padding:5px 9px;font-size:10px' });
  e.innerHTML = '<span style="color:var(--muted)">' + U.esc(s.l) + '</span>' +
                ' <span class="arrow">-></span> ' +
                '<span style="color:var(--' + s.c + ')">' + U.esc(s.r) + '</span>';
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

const msg = wrap.querySelector('#msg');
const texts = [
  '左边是模板，右边是架构名。一行 <span class="v">::format</span> 就把两者合起来。',
  '<span class="v">LLM_KV_NAMES</span> 里像 <span class="k">%s.context_length</span> 这样的模板，<span class="v">%s</span> 位置就是架构名。',
  '<span class="v">LLM_ARCH_NAMES.at(arch)</span> 取出 <span class="k">"llama"</span>，于是 key 变成 <span class="k">llama.context_length</span>。',
  '<b>这就是 hparams 的来源机制</b>：<br>用这个 key 读 GGUF，写进 <span class="v">hparams.n_ctx_train</span>。',
  '同一套机制读 <span class="v">n_embd</span>（1255 行）和 <span class="v">n_layer_all</span>（1259 行）。<br><span class="k">换架构名，整套 metadata 前缀跟着换。</span>',
  '这就是 hparams 与 cparams 的分工起点：<br>凡是从 GGUF 按这种 key 读出来的 → <span class="v">hparams</span>；凡是从调用者参数来的 → <span class="v">cparams</span>。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
els.forEach((_, i) => tl.at(3200 + i * 3000, () => {
  els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15200, () => { els.forEach(x => x.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-02 · 核心',
    title='★ 三张并行的表，条目数并不相等',
    sub='enum llm_tensor 274 项、LLM_TENSOR_NAMES 273 条、LLM_TENSOR_INFOS 268 条。',
    caption='enum llm_tensor 在 src/llama-arch.h:438-713；LLM_TENSOR_NAMES 在 src/llama-arch.cpp:433-707；LLM_TENSOR_INFOS 在 src/llama-arch.cpp:719-998。',
    src=ARCH_H, parts=[(438, 449)], duration=22000,
    mark_src=[438, 446],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div id="tbl"></div>
  <div class="row center" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['三张并行的表', '条数', '它给出什么'],
  [['enum llm_tensor', '274', '权重张量的"种类"编号'],
   ['LLM_TENSOR_NAMES', '273', '名字模板，%d 是层号 / 专家号'],
   ['LLM_TENSOR_INFOS', '268', 'layer 归属 + ggml_op']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const cdefs = [
  { c: 'e', t: '273 条里有 1 个重复 key', b: '<span style="font-family:var(--mono)">LLM_TENSOR_SSM_NORM</span> 在 493 与 518 行各出现一次，所以<b>不同的名字只有 272 个</b>。' },
  { c: 'd', t: '274 减 268 等于 6 个"孤儿"', b: '<span style="font-family:var(--mono)">ATTN_ROT_EMBD</span>、<span style="font-family:var(--mono)">FFN_GATE/DOWN/UP_EXP</span>、<span style="font-family:var(--mono)">POST_ATTN_NORM</span>、<span style="font-family:var(--mono)">POST_MLP_NORM</span> 有枚举、没有 INFOS 记录。' }
];
const cards = cdefs.map(d => { const e = U.card(d, { style: 'width:330px' }); wrap.querySelector('#cards').appendChild(e); return e; });
cards.forEach(e => e.style.opacity = '.35');

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '权重张量的"种类"是一个枚举：<span class="v">enum llm_tensor</span>，<span class="k">274</span> 项。',
  '<span class="v">LLM_TENSOR_NAMES</span>：<span class="k">273</span> 条名字模板（src/llama-arch.cpp:433-707）。<br>模板里的 <span class="v">%d</span> 就是层号 / 专家号。',
  '细看会发现 273 条里有 <span class="k">1 个重复 key</span>，所以不同名字只有 272 个。',
  '<span class="v">LLM_TENSOR_INFOS</span>：只有 <span class="k">268</span> 条，每条给一个 layer 归属和一个 ggml_op。',
  '<span class="k">6 个枚举成员没有 INFOS 记录</span>（见下方卡片）。它们在别处也确实没被引用 —— 是只留在表里的条目。',
  '查 INFOS 用的是 <span class="v">.at()</span>（src/llama-arch.cpp:1058）。<br><span class="k">key 不存在会抛异常</span>，所以新增张量时这张表不能漏。'
];
tl.at(600, () => { rows[0].className = 'on'; msg.innerHTML = texts[0]; });
tl.at(3400, () => { rows.forEach((x, k) => { x.className = k === 1 ? 'on' : ''; }); msg.innerHTML = texts[1]; });
tl.at(6500, () => { cards[0].style.opacity = '1'; msg.innerHTML = texts[2]; });
tl.at(9600, () => { rows.forEach((x, k) => { x.className = k === 2 ? 'on' : ''; }); msg.innerHTML = texts[3]; });
tl.at(13300, () => { rows.forEach(x => { x.className = ''; }); cards[1].style.opacity = '1'; msg.innerHTML = texts[4]; });
tl.at(17600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-02 · 张量命名',
    title='名字 = <span class="hl-c">format</span>(模板, bid, xid)',
    sub='str() 从 LLM_TENSOR_NAMES 取模板，用层号和专家号填充 %d，再补上调用方给的后缀。',
    caption='LLM_TN 的用法注释在 src/llama-arch.h:731-742；模板查不到时 str() 直接 GGML_ABORT。',
    src=ARCH_C, parts=[(1016, 1028)], duration=19000,
    mark_src=[1016, 1021],
    notes={0: 'str() 是 LLM_TN_IMPL 的唯一出口：张量名在这里才真正生成',
           5: 'bid 是层号，xid 是专家号；模板里有几个 %d，就用几个'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="col" id="steps" style="gap:5px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const steps = [
  { l: 'LLM_TENSOR_NAMES[LLM_TENSOR_TOKEN_EMBD]  (434 行)', r: '"token_embd"', c: 'a' },
  { l: 'LLM_TENSOR_NAMES[LLM_TENSOR_ATTN_NORM]  (440 行)', r: '"blk.%d.attn_norm"', c: 'b' },
  { l: 'tn(LLM_TENSOR_ATTN_NORM, "weight", 3)', r: '"blk.3.attn_norm.weight"', c: 'c' },
  { l: 'LLM_TENSOR_NAMES[LLM_TENSOR_FFN_GATE_EXP]  (451 行)', r: '"blk.%d.ffn_gate.%d"', c: 'd' }
];
const host = wrap.querySelector('#steps');
const els = steps.map(s => {
  const e = U.el('div', { class: 'formula', style: 'padding:5px 9px;font-size:10px' });
  e.innerHTML = '<span style="color:var(--muted)">' + U.esc(s.l) + '</span>' +
                ' <span class="arrow">-></span> ' +
                '<span style="color:var(--' + s.c + ')">' + U.esc(s.r) + '</span>';
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

const msg = wrap.querySelector('#msg');
const texts = [
  '张量名有三种"形状"，取决于模板里有几个 <span class="v">%d</span>。',
  '没有 <span class="v">%d</span>：名字与层号无关，如 <span class="k">token_embd</span>。',
  '一个 <span class="v">%d</span>：填层号。模板 <span class="k">blk.%d.attn_norm</span> 加后缀 <span class="k">weight</span>，' +
  '得到 <span class="k">blk.3.attn_norm.weight</span>（用法注释在 src/llama-arch.h:739）。',
  '两个 <span class="v">%d</span>：层号加专家号，MoE 的逐专家权重用这种模板。',
  '<span class="k">权重加载就是拿这个名字去 GGUF 里查</span> —— 名字模板错了就会报找不到张量（详见 L2-03）。',
  '还有一处硬约束：模板在 <span class="v">LLM_TENSOR_NAMES</span> 里查不到时，<br><span class="v">str()</span> 会直接 <span class="v">GGML_ABORT</span>（1017-1019 行）。'
];
els.forEach((e, i) => tl.at(600 + i * 3000, () => {
  els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(12800, () => { els.forEach(x => { x.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
tl.at(15600, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-02 · hparams',
    title='<span class="hl-d">llama_hparams</span>：模型文件说什么，就是什么',
    sub='n_ctx_train / n_embd / n_layer_all / n_expert：全部由 GGUF metadata 决定。',
    caption='struct llama_hparams 共 459 行（src/llama-hparams.h:52-510），字段越靠后越"家族专属"。',
    src=HP_H, parts=[(64, 72)], duration=20000,
    mark_src=[64, 65, 66, 72],
    notes={0: '注意是 n_ctx_train：训练时的长度，不是这次推理用多长',
           2: 'n_layer_all 含 nextn 预测层；n_layer() 返回的是"实际要跑的层数"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'n_ctx_train', b: '模型<b>训练时</b>的上下文长度。<br>不是这次推理用多长。', m: 'uint32_t n_ctx_train;' },
  { c: 'b', t: 'n_embd', b: '隐藏层宽度。<br>图上几乎所有张量的末维。', m: 'uint32_t n_embd;' },
  { c: 'c', t: 'n_layer_all', b: '层数。加载时断言<br>不超过 LLAMA_MAX_LAYERS。', m: 'uint32_t n_layer_all;' },
  { c: 'd', t: 'n_expert', b: 'MoE 的专家数；<br>稠密模型是 0。', m: 'uint32_t n_expert = 0;' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '最核心的四个字段。前三个几乎每个模型都会用到。',
  '<span class="v">n_ctx_train</span>：模型<b>训练时</b>的上下文长度。<br>这次推理能用多长是 <span class="v">cparams.n_ctx</span> —— 两个不同的数。',
  '<span class="v">n_embd</span>：隐藏层宽度。它决定计算图上几乎所有张量的最后一维，<br>所以 L2-06 建图时每个算子的形状都要读它。',
  '<span class="v">n_layer_all</span>：层数。加载时会断言它大于 0 且不超过 <span class="v">LLAMA_MAX_LAYERS</span>（src/llama-model.cpp:1260）。',
  '<span class="v">n_expert</span>：MoE 专家数，稠密模型为 0。<br>后面还有 400 多行字段，是各家族专属的（SSM、MLA、RWKV、HRM 等）。',
  '这些数字<b>全部来自 GGUF metadata</b>：上一幕的 <span class="v">LLM_KV::operator()</span> 就是读取路径。',
  '<span class="k">契约：hparams 加载完就不再变。</span>想改上下文长度，改的是 cparams。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3000 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15000, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
tl.at(17400, () => { msg.innerHTML = texts[6]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-02 · 核心',
    title='★ 每个形状字段都有 <span class="hl-e">(il)</span> 版本：逐层可以不同',
    sub='n_head_arr / n_head_kv_arr / n_ff_arr 是长度 LLAMA_MAX_LAYERS 的数组，getter 按层号取值。',
    caption='数组声明在 src/llama-hparams.h:98-105；n_head() 在 src/llama-hparams.cpp:50，n_head_kv() 在 58。',
    src=HP_C, parts=[(99, 116)], duration=19000,
    mark_src=[99, 106, 109],
    notes={4: '有些层根本没有 KV（纯 SSM 层），这里返回 0 而不是除零',
           11: '同一个 getter，SWA 层和全注意力层返回不同的 RoPE 维度'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row center" id="cards" style="gap:8px"></div>
  <div class="col" id="eqs" style="gap:5px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'e', t: 'n_head_arr[]', b: '每层的查询头数。', m: 'src/llama-hparams.h:98' },
  { c: 'f', t: 'n_head_kv_arr[]', b: '每层的 KV 头数；GGUF 里<br>没有时直接等于 n_head_arr。', m: 'src/llama-model.cpp:1334' },
  { c: 'd', t: 'n_ff_arr[]', b: '每层的 FFN 中间宽度 ——<br>MoE 层与稠密层可以混排。', m: 'src/llama-model.cpp:1327' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const eqs = [
  { l: 'n_gqa(0)', r: 'n_head(0) / n_head_kv(0)', c: 'e' },
  { l: 'n_gqa(7)', r: 'n_head(7) / n_head_kv(7)', c: 'f' },
  { l: 'n_rot(7)', r: 'is_swa(7) 为真取 n_rot_swa，否则取 n_rot_full', c: 'd' }
];
const eqHost = wrap.querySelector('#eqs');
const eqEls = eqs.map(s => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 9px;font-size:10px' });
  e.innerHTML = '<span style="color:var(--' + s.c + ')">' + U.esc(s.l) + '</span>' +
                ' <span class="arrow">=</span> ' +
                '<span style="color:var(--muted)">' + U.esc(s.r) + '</span>';
  eqHost.appendChild(e);
  return e;
});
eqEls.forEach(e => e.style.opacity = '.28');

const msg = wrap.querySelector('#msg');
const texts = [
  '这三个数组（还有 n_ff_exp_arr、n_expert_used_arr）就是"逐层形状"的存放处，长度都是 <span class="v">LLAMA_MAX_LAYERS = 512</span>。',
  '<span class="v">n_gqa(il)</span>：GQA 比例 = 查询头数 / KV 头数。<br>它读的是 per-layer 数组，不是单个字段。',
  '第 102 行有个防御：<span class="v">n_head_kv == 0</span> 时直接返回 0 —— 混合架构里有纯 SSM 层，没有 KV。',
  '所以"GQA 比例"<b>不是模型级的常数</b>：同一个模型里可以逐层不同。',
  '<span class="v">n_rot(il)</span>（110 行）同理：SWA 层取 <span class="v">n_rot_swa</span>，全注意力层取 <span class="v">n_rot_full</span>。',
  '<span class="k">这就是 hparams 必须用数组的原因</span>：SWA、混合记忆、MoE 都会让每层形状不一样。'
];
tl.at(600, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[0]; });
tl.at(3400, () => { eqEls.forEach((x, k) => { x.style.opacity = k === 0 ? '1' : '.28'; }); msg.innerHTML = texts[1]; });
tl.at(6200, () => { msg.innerHTML = texts[2]; });
tl.at(9200, () => { eqEls.forEach((x, k) => { x.style.opacity = k === 1 ? '1' : '.28'; }); msg.innerHTML = texts[3]; });
tl.at(12200, () => { eqEls.forEach((x, k) => { x.style.opacity = k === 2 ? '1' : '.28'; }); msg.innerHTML = texts[4]; });
tl.at(15200, () => { eqEls.forEach(x => { x.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-02 · cparams',
    title='<span class="hl-b">llama_cparams</span>：一次运行的可变配置',
    sub='n_ctx / n_batch / n_ubatch / n_threads 都来自 llama_context_params —— 同一个模型可以有多个 context。',
    caption='struct llama_cparams：src/llama-cparams.h:10-67；填充发生在 src/llama-context.cpp 的构造函数里（84 行起）。',
    src=CP_H, parts=[(10, 20)], duration=19000,
    mark_src=[10, 11, 13, 14, 19],
    notes={1: '对照 hparams.n_ctx_train：一个来自模型文件，一个来自调用者',
           3: 'n_batch 是一次送进图的 token 上限，n_ubatch 是其中 micro-batch 的大小'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['', 'hparams（模型）', 'cparams（这次运行）'],
  [['上下文长度', 'n_ctx_train', 'n_ctx'],
   ['批量', '（没有）', 'n_batch / n_ubatch'],
   ['线程数', '（没有）', 'n_threads / n_threads_batch'],
   ['来源', 'GGUF metadata', 'llama_context_params'],
   ['能改吗', '加载后固定', '每个 context 各自一套']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '<span class="v">llama_cparams</span> 的第一个字段就叫 <span class="v">n_ctx</span>，注释写着 <b>context size used during inference</b>。',
  '和 hparams 的 <span class="v">n_ctx_train</span> 对照：<br>一个是"训练时多长"，一个是"<span class="k">这次用多长</span>"。',
  '<span class="v">n_batch</span> / <span class="v">n_ubatch</span>：一次送进图多少 token，以及再切成多大的 micro-batch（L2-05 展开）。',
  '<span class="v">n_threads</span> / <span class="v">n_threads_batch</span>：CPU 线程数 —— <span class="k">显然不可能来自模型文件</span>。',
  '填充者在 src/llama-context.cpp:246：<span class="v">cparams.n_batch</span> 直接由 <span class="v">params.n_batch</span> 算出来。',
  '<span class="k">判据：这个数在模型文件里吗？</span>在 → hparams；不在 → cparams。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(15600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-02 · 收束',
    title='新增一个架构，要动哪几张表',
    sub='五张表 + 一组查询函数；漏掉任何一张，模型要么"认不出来"，要么"张量找不到"。',
    caption='下一课 L2-03：权重加载按张量名去 GGUF 里查，名字模板错了就会报找不到张量。',
    src=ARCH_C, parts=[(1061, 1073)], duration=24000,
    mark_src=[1061, 1062],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['要动的地方', '在哪（文件:行）', '它回答什么', '漏了会怎样'],
  [['enum llm_arch', 'llama-arch.h:13-167', '家族编号', '代码里无法引用它'],
   ['LLM_ARCH_NAMES', 'llama-arch.cpp:8-162', 'GGUF 架构字符串', 'general.architecture 反查失败'],
   ['enum llm_tensor', 'llama-arch.h:438-713', '新权重的编号', 'LLM_TN 表达不出来'],
   ['LLM_TENSOR_NAMES', 'llama-arch.cpp:433-707', '张量名模板', 'str() 直接 GGML_ABORT'],
   ['LLM_TENSOR_INFOS', 'llama-arch.cpp:719-998', '层归属 + ggml_op', 'info_for() 的 .at() 抛异常'],
   ['llm_arch_is_* 三个查询', 'llama-arch.cpp:1061-1108', '记忆形态', 'KV cache 建错类型']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '要把一个全新的架构加进 llama.cpp，最少要动哪几处？如果它还带来一个从没见过的权重张量呢？',
  '<b>架构本身（2 处，缺一不可）：</b><br>' +
  '1. <span style="font-family:var(--mono)">enum llm_arch</span> 里加新成员，放在 ' +
  '<span style="font-family:var(--mono)">LLM_ARCH_UNKNOWN</span> 之前（llama-arch.h:166）。<br>' +
  '2. <span style="font-family:var(--mono)">LLM_ARCH_NAMES</span> 里加一条映射（llama-arch.cpp:8-162）。' +
  '这个字符串必须与 GGUF 里 <span style="font-family:var(--mono)">general.architecture</span> 的值一致，' +
  '否则 <span style="font-family:var(--mono)">llm_arch_from_string()</span>（1047）反查不到。<br>' +
  '<b>新张量（3 处）：</b><br>' +
  '3. <span style="font-family:var(--mono)">enum llm_tensor</span>（llama-arch.h:438-713）；<br>' +
  '4. <span style="font-family:var(--mono)">LLM_TENSOR_NAMES</span>（llama-arch.cpp:433-707）—— ' +
  '名字模板必须与 GGUF 里真实的张量名逐字一致；<br>' +
  '5. <span style="font-family:var(--mono)">LLM_TENSOR_INFOS</span>（llama-arch.cpp:719-998）—— 给 layer 与 ggml_op。<br>' +
  '<b>若它还是 recurrent / hybrid / diffusion：</b>在 ' +
  '<span style="font-family:var(--mono)">llm_arch_is_recurrent</span>（1061）、' +
  '<span style="font-family:var(--mono)">llm_arch_is_hybrid</span>（1075）、' +
  '<span style="font-family:var(--mono)">llm_arch_is_diffusion</span>（1100）里各加一个 case。<br>' +
  '<b>另外（本课不覆盖，只指路）：</b>还要在 src/llama-model.cpp:43 的模型分派里把新枚举接到一个模型实现上。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '五张表加一组查询函数。前两张决定"认不认得出来"，后三张决定"找不找得到张量"。',
  '<span class="k">enum llm_arch 加 LLM_ARCH_NAMES</span> 是一对：编号与字符串必须同时存在。',
  '<span class="k">enum llm_tensor 加 LLM_TENSOR_NAMES 加 LLM_TENSOR_INFOS</span> 是三张并行表，条目数不相等（上一幕）。',
  '<span class="v">llm_arch_is_recurrent</span> / <span class="v">is_hybrid</span> / <span class="v">is_diffusion</span> 决定记忆形态；' +
  'L2-04 会看到它如何影响 KV cache 的建法。',
  '一句话：<span class="v">架构表决定"认不认得出"，张量表决定"找不找得到"。</span>'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2500 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(1 + (i > 1 ? 2 : i), 3)];
}));
tl.at(17400, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、enum llm_arch：152 个家族 + 1 个哨兵',
    '枚举从 `LLM_ARCH_CLIP` 开始（第 14 行），到第 166 行的 `LLM_ARCH_UNKNOWN` 结束，'
    '中间共 **153 个成员**。最后那个不是真架构，而是"查不到"的返回值：'
    '`llm_arch_from_string()` 找不到匹配时返回它。',
    src=ARCH_H, parts=[(160, 168)], lang='c')

L.section(
    '二、LLM_ARCH_NAMES：枚举值与 GGUF 字符串的对照表',
    '这张 `std::map<llm_arch, const char *>` 共 **153 条**，与枚举成员一一对应。'
    '映射到右边的字符串**就是** GGUF 文件里 `general.architecture` 的值 —— '
    '所以它同时也是"模型文件 ↔ 代码枚举"的接口。源码注释还标出一个特例：'
    '`clip` 是 dummy，只给 `llama-quantize` 用。',
    src=ARCH_C, parts=[(8, 13)], lang='c')

L.section(
    '三、LLM_KV_NAMES：metadata key 的模板',
    '`LLM_KV_NAMES` 共 **243 条**（第 164-431 行）。前面的 `general.*` 与架构无关，'
    '后面那些带 `%s` 前缀的才是架构相关的模板 —— `%s` 就是架构名字符串的位置。',
    src=ARCH_C, parts=[(164, 170)], lang='c')

L.section(
    '四、enum llm_tensor 与两张派生表',
    '`enum llm_tensor`（`src/llama-arch.h:438-713`）共 **274** 项。'
    '它对应两张表：`LLM_TENSOR_NAMES`（`src/llama-arch.cpp:433-707`）**273** 条名字模板；'
    '`LLM_TENSOR_INFOS`（`src/llama-arch.cpp:719-998`）**268** 条，每条给 '
    '`llm_tensor_layer layer` 与 `ggml_op op` 两个字段。\n\n'
    '三个数字并不相等：273 条初始化项里有 1 个重复 key（`LLM_TENSOR_SSM_NORM` 在 493 与 518 行各一次），'
    '所以不同的名字只有 272 个；另有 6 个枚举成员在 `LLM_TENSOR_INFOS` 里没有记录 —— '
    '`LLM_TENSOR_ATTN_ROT_EMBD`、`LLM_TENSOR_FFN_GATE_EXP`、`LLM_TENSOR_FFN_DOWN_EXP`、'
    '`LLM_TENSOR_FFN_UP_EXP`、`LLM_TENSOR_POST_ATTN_NORM`、`LLM_TENSOR_POST_MLP_NORM`。'
    '后两个（`src/llama-arch.h:496-497`）在整棵源码树里只出现在枚举定义处。',
    src=ARCH_C, parts=[(719, 724)], lang='c')

L.section(
    '五、LLM_TN：张量名的构造入口',
    '模型实现里不写字符串，而是 `LLM_TN(arch)` 加一个 `llm_tensor` 枚举值。'
    '源码把用法写在注释里：模板 + 后缀 + 层号（+ 专家号），'
    '最终的字符串由 `LLM_TN_IMPL::str()`（`src/llama-arch.cpp:1016-1028`）生成。',
    src=ARCH_H, parts=[(731, 742)], lang='c')

L.section(
    '六、llama_hparams：逐层的形状数组',
    '为什么 `n_head` 之类的字段要放数组？因为**每一层的形状可以不一样**。'
    '数组长度是 `LLAMA_MAX_LAYERS = 512`（`src/llama-hparams.h:10`），'
    'getter 用层号索引；`n_head_kv_arr` 在 GGUF 没写时直接等于 `n_head_arr`'
    '（`src/llama-model.cpp:1334`）。',
    src=HP_H, parts=[(98, 105)], lang='c')

L.section(
    '七、llama_cparams：一次运行的配置',
    '`llama_cparams` 里既有 `n_ctx` / `n_batch` 这类调用者可调的字段，'
    '也有一批 `fused_*` / `auto_*` 开关（运行时自动选择融合算子的结果）。'
    '注意第 28-30 行的注释：源码明确说 YaRN 的四个系数**不放进 GGUF**，'
    '因为所有现有模型取值相同 —— 这正好从反面说明"进不进 GGUF"是这两个结构的分界线。',
    src=CP_H, parts=[(35, 55)], lang='c')

L.section(
    '八、llama-cparams.cpp：整个文件只有 5 行',
    '六个文件里最小的一个。它只导出一个函数：并行序列数上限。'
    '上限的值来自头文件里的 `LLAMA_MAX_SEQ`（`src/llama-cparams.h:8`）。',
    src=CP_C, parts=[(1, 5)], lang='c')

L.footnote_add('本课逐字引用 6 个文件：`src/llama-arch.cpp`、`src/llama-arch.h`、'
               '`src/llama-hparams.cpp`、`src/llama-hparams.h`、`src/llama-cparams.cpp`、'
               '`src/llama-cparams.h`，全部计入覆盖率。')
L.footnote_add('场景与正文里用 `文件:行` 形式指路、但**没有逐字引用**的文件有：'
               '`src/llama-model.cpp`（1254/1255/1259/1260/1327/1328/1334/43 行）、'
               '`src/llama-model-loader.cpp`、'
               '`src/llama-context.cpp`（84/246 行）。它们不计入本课覆盖率，'
               '其行号由本课作者用 `grep -n` / `sed -n` 实地核对过。')

L.prereqs('`L2-01`')

L.goal(
    '说出 `enum llm_arch` 有多少个成员、哨兵是谁，以及它与 GGUF `general.architecture` 的对应关系（对应验收点）；',
    '解释 `LLM_KV::operator()` 如何把架构名与 key 模板拼成真正的 metadata key —— 这是 hparams 的来源；',
    '说清 `llama_hparams` 与 `llama_cparams` 的分工判据：这个数在不在模型文件里；',
    '列举新增一个模型架构必须同时修改的几张表，并说出漏掉每一张的后果。')

L.conclusion(
    '架构表：一张表，三个出口',
    '`enum llm_arch` **153** 个成员（152 个真架构 + 哨兵 `LLM_ARCH_UNKNOWN`），'
    '与 `LLM_ARCH_NAMES` 的 **153** 条一一对应。\n\n'
    '| 出口 | 函数 | 位置 |\n|---|---|---|\n'
    '| 枚举 → 字符串 | `llm_arch_name()` | `src/llama-arch.cpp:1040` |\n'
    '| 字符串 → 枚举 | `llm_arch_from_string()` | `src/llama-arch.cpp:1047` |\n'
    '| 枚举 → 全部枚举 | `llm_arch_all()` | `src/llama-arch.cpp:1031` |\n\n'
    '日志里打印的架构名、加载时认领的架构、遍历所有架构都走这一张表。')

L.conclusion(
    '★ hparams 与 cparams 的分工判据',
    '一句话判据：**这个数在模型文件里吗？**\n\n'
    '| | hparams | cparams |\n|---|---|---|\n'
    '| 代表字段 | `n_ctx_train` `n_embd` `n_layer_all` `n_expert` | `n_ctx` `n_batch` `n_ubatch` `n_threads` |\n'
    '| 来源 | GGUF metadata | `llama_context_params` |\n'
    '| 谁读 | 建图（L2-06）、KV cache（L2-04） | 上下文与批处理（L2-05） |\n'
    '| 生命周期 | 加载后固定 | 每个 `llama_context` 一套 |\n\n'
    '分界线在源码里也有反证：`src/llama-cparams.h:28-30` 的注释说 YaRN 的四个系数'
    '**不放进 GGUF** —— 因为它们不属于模型。')

L.conclusion(
    '★ 新增一个架构：五张表 + 一组查询',
    '| 要动的表 | 位置 | 它回答什么 |\n|---|---|---|\n'
    '| `enum llm_arch` | `src/llama-arch.h:13-167` | 家族编号 |\n'
    '| `LLM_ARCH_NAMES` | `src/llama-arch.cpp:8-162` | GGUF 架构字符串 |\n'
    '| `enum llm_tensor` | `src/llama-arch.h:438-713` | 新权重的编号 |\n'
    '| `LLM_TENSOR_NAMES` | `src/llama-arch.cpp:433-707` | 张量名模板 |\n'
    '| `LLM_TENSOR_INFOS` | `src/llama-arch.cpp:719-998` | 层归属 + `ggml_op` |\n\n'
    '若该架构是 recurrent / hybrid / diffusion，还要在 `llm_arch_is_recurrent`（1061）、'
    '`llm_arch_is_hybrid`（1075）、`llm_arch_is_diffusion`（1100）里加 case。'
    '**架构表决定"认不认得出"，张量表决定"找不找得到"。**')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
