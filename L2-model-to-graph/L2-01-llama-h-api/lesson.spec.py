#!/usr/bin/env python3
"""L2-01 · llama.h 公共 API 全景 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-01-llama-h-api/lesson.spec.py

约定：
  - 所有代码引用只写行号（parts / mark_src），由 lessonkit 从上游逐字抽取。
  - mark_src 用【上游真实行号】；图示里用 hi([...]) 按【序号】点亮，
    序号即 mark_src 列表的下标（lessonkit 按行号升序换算）。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_H    = 'include/llama.h'
SRC_CPPH = 'include/llama-cpp.h'
SRC_CPP  = 'src/llama.cpp'
SRC_EXT  = 'src/llama-ext.h'

# 图示里通用的一段：按序号点亮代码区里被标记的行
HI_JS = '''
const ML = document.querySelectorAll('mark.ln-mark');
function hi(ks) {                      // ks = 序号数组；'all' = 全部点亮；[] = 全灭
  const all = (ks === 'all');
  ML.forEach((m, i) => {
    const on = all || ks.indexOf(i) >= 0;
    m.className = on ? 'ln-mark on' : 'ln-mark';
    m.style.display = on ? 'inline-block' : 'none';
  });
}
'''

L = Lesson(
    id='L2-01',
    layer='L2 · 从模型到图',
    title='llama.h 公共 API 全景',
    codecap='include/llama.h 等 4 个文件（逐字引用）',
    nav={'prev': {'href': '../../L1-operator-representation/L1-06-gguf-container/index.html',
                  'label': 'L1-06 GGUF 容器'},
         'next': {'href': '../L2-02-arch-table-hparams/index.html',
                  'label': 'L2-02 架构表与超参'}},
)

L.note('**一句话**：L1-06 结束时，权重还只是一只 GGUF 文件；L2 层要把它变成"待执行的图"。'
       '这一层的第一步，是先认清**谁在对外承诺什么** —— 那份承诺写在 `include/llama.h` 里。')
L.note('本课的视角是"调用方看到的世界"：公共头文件里有哪些不透明类型、哪些参数结构体，'
       '以及一个程序从启动到出第一个 token 必须按什么顺序调用它们。'
       '内部实现散落在哪些文件，本课只点到位置，展开留给 L2-05 与 L2-07。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2 · 从模型到图',
    title='<span class="hl-a">一个头文件</span>，四个文件的分工',
    sub='公共 API 的边界在哪里：契约、实现入口、C++ 包装、未定型的扩展。',
    caption='本课覆盖 include/llama.h、include/llama-cpp.h、src/llama.cpp、src/llama-ext.h 四个文件，全部计入覆盖率。',
    src=SRC_H, parts=[(51, 70)], duration=17000,
    mark_src=[61, 62, 63, 64, 66],
    notes={13: '四个不透明类型的唯一公共声明处：调用方只拿得到指针，看不到任何字段'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">权重文件 GGUF</span><span class="arrow">-></span>
    <span class="chip a">公共 API 契约</span><span class="arrow">-></span>
    <span class="chip b">内部模块实现</span><span class="arrow">-></span>
    <span class="chip d">待执行的图</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
const defs = [
  { c: 'a', t: 'include/llama.h',   m: '1646 行', b: '唯一对外契约<br>所有 llama_* 在这里声明' },
  { c: 'b', t: 'src/llama.cpp',     m: '620 行',  b: '契约的实现入口<br>初始化、加载、装配' },
  { c: 'c', t: 'include/llama-cpp.h', m: '30 行', b: '只给 C++ 用<br>智能指针 RAII 包装' },
  { c: 'd', t: 'src/llama-ext.h',   m: '134 行',  b: 'staging：尚未定型<br>允许破坏性变更' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.card(d, { style: 'width:163px;flex:0 0 auto' });
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.30');
hi([]);

const msg = wrap.querySelector('#msg');
const texts = [
  '从一个 GGUF 文件到一张待执行的图，要穿过 <span class="k">四层</span>。第一层就是公共 API 契约。',
  '<span class="v">include/llama.h</span>：1646 行，241 处 <span class="k">LLAMA_API</span> 声明。<br>它<b>只有声明</b>，一个函数体都没有 —— 这就是"契约"。',
  '<span class="v">src/llama.cpp</span>：只有 620 行，却 include 了全部内部模块。<br>它的角色是"接线"，第 6 幕专门讲。',
  '<span class="v">include/llama-cpp.h</span>：30 行，开头就 <span class="k">#error</span> 拒绝 C。<br>它把 C 的所有权规则翻译成 C++ 的 RAII。',
  '<span class="v">src/llama-ext.h</span>：自称 staging 头，允许破坏性变更。<br>新 API 先在这里长出来，稳定后再搬进 llama.h。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14200, () => {
  els.forEach(e => e.style.opacity = '1');
  hi('all');
  msg.innerHTML = '右边这段就是契约的开头：<span class="k">四个不透明类型</span>，加上一个 memory 句柄。<br>'
    + '有了它们，内部结构体怎么改都不会破坏调用方。';
});
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-01 · 三段式',
    title='★ <span class="hl-a">三个不透明指针</span>，三段生命周期',
    sub='模型 / 上下文 / 采样器各有独立的创建与释放函数；默认值由函数给出，不写在字段里。',
    caption='上下文参数里的 n_ctx / n_batch 只是"请求值"，实际值要用 llama_n_ctx(ctx) 一族查询（llama.h:567-575）。',
    src=SRC_H, parts=[(472, 477)], duration=17000,
    mark_src=[474, 475, 476, 477],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row" id="cards" style="gap:8px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
const defs = [
  { c: 'a', t: 'struct llama_model', b: '权重 + 超参 + 词表<br>只读，可被多个上下文共享',
    m: '建 517 · 释 542' },
  { c: 'c', t: 'struct llama_context', b: '一次推理的运行时状态<br>memory / KV cache / 后端调度器',
    m: '建 544 · 释 554' },
  { c: 'b', t: 'struct llama_sampler', b: '采样链，在 CPU 上跑<br>生命周期不依附上下文',
    m: '建 1355 · 释 1350' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.card(d, { style: 'width:219px;flex:0 0 auto' });
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.30');
hi([]);

const msg = wrap.querySelector('#msg');
const texts = [
  '三段式：<span class="k">模型</span> → <span class="k">上下文</span> → <span class="k">采样器</span>。三者生命周期互相独立。',
  '<span class="v">struct llama_model</span>：从文件读出来的东西，<b>只读</b>。<br>建：<span class="m">llama_model_load_from_file</span>（517）；释：<span class="m">llama_model_free</span>（542）。',
  '<span class="v">struct llama_context</span>：一次推理的全部运行时状态。<br>建：<span class="m">llama_init_from_model</span>（544）；释：<span class="m">llama_free</span>（554）—— 释放函数不叫 llama_context_free。',
  '<span class="v">struct llama_sampler</span>：采样链。<br>建：<span class="m">llama_sampler_chain_init</span>（1355）；释：<span class="m">llama_sampler_free</span>（1350）。',
  '每个类都给一个 <span class="k">default_params</span> 函数（右边四行），而不是在头文件里写字段默认值 ——<br>默认值由实现决定，可以随版本变；字段布局却保持稳定。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  hi([i]);
  msg.innerHTML = texts[i];
}));
tl.at(14300, () => {
  els.forEach(e => e.style.opacity = '1');
  hi('all');
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-01 · 参数族',
    title='参数结构体族：<span class="hl-c">4 个 struct</span> + 4 个默认值函数',
    sub='llama.cpp 不用"一堆散参数"，而是"一个按值传的结构体 + 一个取默认值的函数"。',
    caption='结构体按值传（by value），所以里面有注释专门说明：bool 集中放末尾，避免拷贝时错位。',
    src=SRC_H, parts=[(314, 330)], duration=18000,
    mark_src=[314, 316, 319, 321, 322, 323, 325],
    notes={2: 'devices：NULL 结尾的设备列表；为 NULL 表示用全部可用设备（设备发现见 L3-02）',
           9: 'load_mode：auto / none / mmap / mlock / mmap+mlock / dio —— 决定权重怎么进内存（L2-03）',
           11: 'lazy_mode：配合 mmap 按需读取张量，而不是一次性全部读进来'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
hi('all');

const t = U.table(
  ['参数结构体', '行', '用在', '代表字段'],
  [['llama_model_params',          '314', '加载模型',   'devices / n_gpu_layers / split_mode / load_mode / vocab_only'],
   ['llama_context_params',        '360', '创建上下文', 'n_ctx / n_batch / n_ubatch / n_seq_max / type_k / type_v'],
   ['llama_model_quantize_params', '434', '量化模型',   'ftype / imatrix / kv_overrides / tt_overrides'],
   ['llama_sampler_chain_params',  '457', '创建采样链', 'no_perf（唯一字段，默认 true）']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '四个结构体，四个"入口参数包"。按声明顺序看：模型怎么进来 → 上下文多大 → 怎么量化 → 采样链要不要计时。',
  '<span class="v">llama_model_params</span>（314）：决定模型怎么被读进来 —— 放哪几张卡、怎么切、要不要 mmap。',
  '<span class="v">llama_context_params</span>（360）：决定上下文多大、批多大、KV cache 用什么精度。<br>它同时是 <span class="k">n_ctx</span> 这类"请求值"的载体，真实值要另外查询（570）。',
  '<span class="v">llama_model_quantize_params</span>（434）与 <span class="v">llama_sampler_chain_params</span>（457）：<br>一个管离线量化，一个只管采样链要不要计时。',
  '一句话：<span class="k">结构体是"输入契约"，默认值函数是"实现说了算"</span>。<br>llama.h:472-477 那四行 default_params 就是这条约定的落点。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3300 + i * 2800, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15000, () => {
  rows.forEach(x => { x.className = ''; });
  hi('all');
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-01 · 核心',
    title='★ <span class="hl-a">六步调用顺序</span>：从 backend_init 到 sampler_sample',
    sub='一个用 llama.cpp 的程序，最小骨架就是这六个调用。顺序不能换。',
    caption='这六处声明散布在 1646 行头文件的六个不同段落 —— 代码区用区间分隔行标出各自的真实位置。',
    src=SRC_H,
    parts=[(482, 482), (517, 519), (544, 546), (1180, 1187), (998, 1000), (1548, 1548)],
    duration=26000,
    mark_src=[482, 517, 544, 1180, 998, 1548],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="row" id="r1" style="gap:6px;justify-content:center"></div>
  <div class="row" id="r2" style="gap:6px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
const defs = [
  { c: 'a', t: '1 · llama_backend_init',         b: '初始化时间与后端注册表',     m: 'llama.h:482' },
  { c: 'a', t: '2 · llama_model_load_from_file', b: '读 GGUF，建模型与设备',      m: 'llama.h:517' },
  { c: 'c', t: '3 · llama_init_from_model',      b: '建上下文：memory 与调度器',  m: 'llama.h:544' },
  { c: 'b', t: '4 · llama_tokenize',             b: '文本切成 token id',          m: 'llama.h:1180' },
  { c: 'd', t: '5 · llama_decode',               b: '跑一遍图，产出 logits',      m: 'llama.h:998' },
  { c: 'e', t: '6 · llama_sampler_sample',       b: '从 logits 选出下一个 token', m: 'llama.h:1548' }
];
const r1 = wrap.querySelector('#r1'), r2 = wrap.querySelector('#r2');
const els = [];
defs.forEach((d, i) => {
  const host = i < 3 ? r1 : r2;
  if (i % 3 !== 0) host.appendChild(U.arrow('->'));
  const e = U.card(d, { style: 'width:196px;flex:0 0 auto' });
  host.appendChild(e); els.push(e);
});
els.forEach(e => e.style.opacity = '.28');
hi([]);

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">llama_backend_init</span>：进程级，只调一次。它注册后端、初始化 f16 表（实现在 src/llama.cpp:122）。',
  '<span class="v">llama_model_load_from_file</span>：把 GGUF 读成 <span class="k">llama_model</span>。<br>权重从哪来？回顾 L1-06 —— 就是那一课拆开的容器文件。',
  '<span class="v">llama_init_from_model</span>：模型只读，上下文才是"这次的运行状态"。<br>KV cache（memory）在这一步建出来 —— 展开在 L2-04。',
  '<span class="v">llama_tokenize</span>：第一个参数是 <span class="m">const struct llama_vocab *</span>，不是 model。<br>词表要用 <span class="m">llama_model_get_vocab(model)</span>（588）取。',
  '<span class="v">llama_decode</span>：把 <span class="m">llama_batch</span> 喂进去。<br>它<b>必须有 memory</b>；返回值 1 = 找不到 KV 槽位，-1 = 输入非法（998 上方的注释写全了）。',
  '<span class="v">llama_sampler_sample</span>：第 5 步产出的 logits 在这里被消费。<br>它是 <span class="m">get_logits_ith → apply → accept</span> 的简写（1541-1546）。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  hi([i]);
  msg.innerHTML = texts[i];
}));
tl.at(21500, () => {
  els.forEach(e => e.style.opacity = '1');
  hi('all');
  msg.innerHTML = '六步里 <span class="k">只有第 5 步会碰图</span>；第 6 步在 CPU 上、不进图。<br>'
    + '所以"从模型到图"这条主线，真正的主角是第 2 步与第 5 步 —— 见 L2-05、L2-07。';
});
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-01 · 输入',
    title='<span class="hl-b">llama_batch</span>：喂给 decode 的那一张表',
    sub='七个字段全是并行数组，长度都等于 n_tokens；这是 L2 层"数据进图"的统一入口。',
    caption='decode 的注释明写 Requires the context to have a memory（987）；memory 用 llama_get_memory(ctx)（585）取，再用 llama_memory_seq_* 一族操作。',
    src=SRC_H, parts=[(247, 272), (742, 749)], duration=20000,
    mark_src=[263, 265, 267, 268, 271, 742],
    notes={20: 'embd 与 token 二选一：要么给 token id，要么直接给词向量',
           24: 'logits 为 0 的 token 不输出 logits —— 只要最后一个 token 时，其余全填 0'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:6px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin:0 0 2px">struct llama_batch —— 7 个字段</div>';
const fields = [
  ['n_tokens',  '本批 token 数；其余数组的长度'],
  ['token',     'token id 数组（与 embd 二选一）'],
  ['embd',      '词向量数组（与 token 二选一）'],
  ['pos',       '每个 token 在序列中的位置'],
  ['n_seq_id',  '每个 token 归属几个序列'],
  ['seq_id',    '每个 token 归属的序列 id 列表'],
  ['logits',    '该 token 是否要输出 logits']
];
const rows = fields.map(f => {
  const e = U.el('div', { class: 'formula', style: 'padding:3px 8px;font-size:10px' });
  e.innerHTML = '<span class="m">' + U.esc(f[0]) + '</span> &nbsp;' + U.esc(f[1]);
  left.appendChild(e);
  return e;
});
rows.forEach(e => e.style.opacity = '.30');

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">为什么是"数组的数组"</div>' +
  '<div class="cb">一个 batch 可以同时装多条序列（n_seq_id / seq_id 都是二维），' +
  '所以<b>一次 decode 能并行推进多个请求</b>。切分与 ubatch 见 <b>L2-05</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">decode 需要 memory</div>' +
  '<div class="cb"><span class="cm" style="margin:0">Requires the context to have a memory</span>（987）。' +
  'memory 是 KV cache 之上的抽象：<span class="cm" style="margin:0">llama_get_memory</span>（585）取句柄，' +
  '再用 <span class="cm" style="margin:0">llama_memory_seq_rm</span>（756）按序列增删。展开在 <b>L2-04</b>。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">llama_batch</span> 是一条"记录表"：每列长度都等于 <span class="k">n_tokens</span>。',
  '<span class="v">n_tokens</span> + <span class="v">token</span>：最常见的用法 —— 给一串 token id。',
  '<span class="v">embd</span> 与 <span class="v">token</span> 互斥：要直接喂向量就填 embd，把 token 置 NULL。',
  '<span class="v">pos</span> 为 NULL 时位置自动跟踪；<span class="v">seq_id</span> 为 NULL 时当作序列 0。',
  '<span class="v">n_seq_id</span> / <span class="v">seq_id</span>：把"一个 token 属于哪些序列"表达成二维数组 ——<br>这是多序列并行推理的数据基础（L2-05）。',
  '<span class="v">logits</span> 是唯一的输出开关：<br>为 NULL 且非 embeddings 时，<b>只输出最后一个 token</b>。',
  'batch 与 memory 是 L2 层的两块"输入状态"：一个描述<b>这次算什么</b>，一个描述<b>之前算过什么</b>。'
];
const mk = [[0], [1], [2], [3], [], [], [4]];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(3200 + i * 2400, () => {
  rows.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.30'; });
  hi(mk[i]);
  msg.innerHTML = texts[i + 1];
}));
tl.at(19000, () => {
  rows.forEach(r => { r.style.opacity = '1'; });
  hi('all');
  msg.innerHTML = texts[6];
});
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-01 · 聚合入口',
    title='★ <span class="hl-a">src/llama.cpp</span>：620 行的接线板',
    sub='它 include 了所有内部模块，自己却几乎不算数：只做初始化、加载编排、chat 模板与分片路径。',
    caption='"聚合入口"的含义：它是唯一把全部内部模块拉在一起的翻译单元，加载与初始化在这里被翻译成内部调用；六个主调用里只有两个落在这个文件。',
    src=SRC_CPP,
    parts=[(1, 17), (34, 36), (316, 327), (344, 362), (369, 369)],
    duration=22000,
    mark_src=[1, 6, 35, 319, 325, 348, 356, 369],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">llama_model_loader</span><span class="arrow">-></span>
    <span class="chip c">llama_model_create</span><span class="arrow">-></span>
    <span class="chip d">load_hparams</span><span class="arrow">-></span>
    <span class="chip d">load_vocab</span><span class="arrow">-></span>
    <span class="chip b">load_tensors</span>
  </div>
  <div class="row" id="cards" style="gap:8px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
const defs = [
  { c: 'a', t: '它 include 了一切', b: '9 个内部头 + 4 个 ggml/gguf 头<br>llama-context.h / llama-vocab.h / llama-model.h …',
    m: '第 1-17 行' },
  { c: 'c', t: '它自己只做接线', b: 'backend 初始化、模型加载编排、<br>chat 模板、分片路径',
    m: '// interface implementation' },
  { c: 'b', t: '重活在别的文件', b: 'llama_init_from_model 在 llama-context.cpp:3739<br>llama_tokenize 在 llama-vocab.cpp:4496',
    m: 'llama_sampler_sample 在 llama-sampler.cpp:895' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.card(d, { style: 'width:219px;flex:0 0 auto' });
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.30');
hi([]);

const msg = wrap.querySelector('#msg');
const texts = [
  '这个文件只有 <span class="k">620 行</span>，却 include 了全部内部模块 —— 典型的"聚合入口"。',
  '第 1-17 行是它的身份证明：<span class="v">llama.h</span> + 9 个内部头 + 4 个 ggml/gguf 头。<br>其中 llama-chat.h 与 llama-model-saver.h 在 src/ 下只被各自的实现文件和它 include。',
  '它实现的公共函数只有这几类：<span class="m">llama_backend_init</span>（122）、模型加载（465）、<br><span class="m">llama_chat_apply_template</span>（509）、<span class="m">llama_split_path</span>（544）。',
  '<span class="v">llama_model_load</span>（316）是加载编排：<br>构造 loader → 建模型 → 挑设备 → 读 hparams / vocab / stats → 搬张量。',
  '逐步对照跨课：<span class="k">hparams</span> 是 L2-02 的主题，<span class="k">load_tensors</span> 是 L2-03 的主题。',
  '所以"llama.cpp 是聚合入口"不是修辞：<span class="k">它把 C ABI 的调用翻译成内部模块的调用序列</span>，自己不含任何算子。'
];
const mk = [[0], [1], [2], [3, 4], [5, 6], [7]];
defs.forEach((_, i) => tl.at(700 + i * 3500, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  hi(mk[Math.min(i, 5)]);
  msg.innerHTML = texts[i];
}));
tl.at(18200, () => {
  els.forEach(e => e.style.opacity = '1');
  hi('all');
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-01 · C++ 包装',
    title='<span class="hl-b">llama-cpp.h</span>：把所有权写进类型',
    sub='30 行，四个 deleter，四个 unique_ptr 别名 —— C 的"记得手动释放"变成 C++ 的"不可能忘记"。',
    caption='注意对称性：创建用哪个函数，就决定了释放用哪个函数；而 context 的释放函数叫 llama_free。',
    src=SRC_CPPH, parts=[(1, 30)], duration=17000,
    mark_src=[4, 12, 16, 20, 24, 27, 28, 29, 30],
    notes={11: '四个 deleter 一一对应 llama.h 里的四个释放函数（542 / 554 / 1350 / 712）',
           29: '拿到 llama_model_ptr，就等于"离开作用域自动 llama_model_free"—— 泄漏在类型层面被排除'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row wrap" id="cards" style="gap:8px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
const defs = [
  { c: 'a', t: 'llama_model_ptr',   b: '模型；<br>按值共享最省事', m: 'llama_model_free' },
  { c: 'c', t: 'llama_context_ptr', b: '上下文；<br>名字与释放函数不对称', m: 'llama_free' },
  { c: 'b', t: 'llama_sampler_ptr', b: '采样器 / 采样链；<br>进链后不再自己释放', m: 'llama_sampler_free' },
  { c: 'd', t: 'llama_adapter_lora_ptr', b: 'LoRA 适配器；<br>寿命不得超过模型', m: 'llama_adapter_lora_free' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.card(d, { style: 'width:163px;flex:0 0 auto' });
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.30');
hi([]);

const msg = wrap.querySelector('#msg');
const texts = [
  '第一件事是拒绝 C：<span class="k">#ifndef __cplusplus</span> 紧接一个 <span class="k">#error</span>（3-5）。',
  '<span class="v">llama_model_ptr</span>：deleter 调 <span class="m">llama_model_free</span>（llama.h:542）。',
  '<span class="v">llama_context_ptr</span>：deleter 调 <span class="m">llama_free</span>（554）。<br>创建函数叫 <span class="m">llama_init_from_model</span>，释放函数却叫 <span class="m">llama_free</span> —— 公共 API 的历史命名。',
  '<span class="v">llama_sampler_ptr</span>：deleter 调 <span class="m">llama_sampler_free</span>（1350）。<br>但一旦交给 <span class="m">llama_sampler_chain_add</span>（1358），所有权就归链，不能再自己释放。',
  '<span class="v">llama_adapter_lora_ptr</span>：LoRA 的释放函数（712）。<br>llama.h 里那句注释仍然成立：适配器的寿命不能超过模型。',
  '最后四个 <span class="k">typedef</span> 才是这个头的全部产出：<br>C 的所有权约定是文档，C++ 的所有权约定是类型。'
];
const mk = [[0], [1], [2], [3], [4], [5, 6, 7, 8]];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  hi(mk[Math.min(i, 5)]);
  msg.innerHTML = texts[i];
}));
tl.at(16200, () => {
  els.forEach(e => e.style.opacity = '1');
  hi('all');
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-01 · 扩展头',
    title='<span class="hl-c">llama-ext.h</span>：试验场，与它没做到的自律',
    sub='新 API 先在这里长出来，稳定了再搬进 llama.h；但它开头那句"尽量别被 include"已经被打破了。',
    caption='本课只讲这个文件的角色；它声明的建图与量化 API 分别在 L2-06 / L2-07 与 L2-09 展开。',
    src=SRC_EXT, parts=[(1, 28)], duration=18000,
    mark_src=[3, 4, 5, 13, 20, 22],
    notes={4: '实测：src/ 下已有 4 个文件 include 了它（llama-context.h、llama-model.cpp、llama-context.cpp、llama-quant.cpp）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row wrap" id="cards" style="gap:8px;justify-content:center"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
const defs = [
  { c: 'a', t: 'llama_graph_reserve', b: '预留一张计算图，直到下次调用前有效。<br>建图入口在这里露头 —— 见 L2-06 / L2-07',
    m: 'llama-ext.h:13' },
  { c: 'c', t: 'llama_ftype_get_default_type', b: '给定 ftype 返回默认 ggml_type。<br>量化类型选择 —— 见 L2-09',
    m: 'llama-ext.h:20' },
  { c: 'd', t: 'quantize_state_impl', b: 'C++ 的 struct，没有不透明包装。<br>所以这个头文件天然只能 C++ 用',
    m: 'llama-ext.h:22' },
  { c: 'g', t: '注释与实测不一致', b: '第 5 行写"尽量别 include 这个头"，<br>但 src/ 下已有 4 处 include',
    m: 'grep 实测' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => {
  const e = U.card(d, { style: 'width:163px;flex:0 0 auto' });
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.30');
hi([]);

const msg = wrap.querySelector('#msg');
const texts = [
  '第 3-5 行是这个文件的自我介绍：<span class="k">staging header</span>，允许破坏性变更，一切都算 WIP。',
  '所以读它的方式和读 llama.h <b>不同</b>：<span class="v">llama.h</span> 是承诺，<span class="v">llama-ext.h</span> 是"当前状态"。',
  '<span class="v">llama_graph_reserve</span>（13）是建图入口：给定 n_tokens / n_seqs / n_outputs 预留一张图。<br>这就是"从模型到图"那句主题词在公共 API 上的落点。',
  '<span class="v">quantize_state_impl</span>（22）是个 C++ 类型，连前置声明都直接写着。<br>它和文件后面的 <span class="m">llama_memory_breakdown</span>（84，用 std::map）一起，决定了这个头不能给 C 用。',
  '同一段里还有 <span class="m">llama_memory_breakdown_data</span>（67）：模型 / 上下文 / 计算缓冲各占多少。<br>这是选后端时要看的账（L3-02 / L4-03）。',
  '最后一条要记牢：<span class="k">源码注释不一定是真的</span>。<br>第 5 行的自律，grep 一下就知道没做到 —— 凡结论必实测。'
];
const mk = [[0, 1, 2], [0, 1, 2], [3], [5], [], [2]];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  hi(mk[Math.min(i, 5)]);
  msg.innerHTML = texts[i];
}));
tl.at(17200, () => {
  els.forEach(e => e.style.opacity = '1');
  hi('all');
  msg.innerHTML = texts[5];
});
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-01 · 收束',
    title='把这一课压成一张表',
    sub='头文件自己就写了正确用法：先建链，再循环 decode → sample。',
    caption='下一课 L2-02 进入架构表与超参：llama_model 里那些 hparams 究竟从 GGUF 的哪个键读出来。',
    src=SRC_H, parts=[(1243, 1272)], duration=20000,
    mark_src=[1243, 1248, 1250, 1266, 1269],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);
''' + HI_JS + '''
hi([]);

const t = U.table(
  ['文件', '它是什么', '关键内容', '接着看'],
  [['include/llama.h',     '公共 C ABI 契约', '三段式对象 / 参数结构体族 / 六步调用序',    '全课'],
   ['include/llama-cpp.h', 'C++ RAII 包装',   '4 个 deleter + 4 个 unique_ptr 别名',      'common/ 与 tests/'],
   ['src/llama.cpp',       '实现与装配入口',  'backend 初始化 / 模型加载编排 / chat / split', 'L2-02 · L2-03'],
   ['src/llama-ext.h',     'staging 扩展头',  'llama_graph_reserve / llama_quant_*',      'L2-06 · L2-07']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '不看代码，把"从加载模型到拿到第一个 token"的调用链按顺序写出来；' +
  '并说出哪一步要求上下文里<b>必须有 memory</b>。',
  '依次是 <span class="mono">llama_backend_init</span> → ' +
  '<span class="mono">llama_model_load_from_file</span> → ' +
  '<span class="mono">llama_init_from_model</span> → ' +
  '<span class="mono">llama_tokenize</span> → ' +
  '<span class="mono">llama_decode</span> → ' +
  '<span class="mono">llama_sampler_sample</span>。<br>' +
  '要求有 memory 的是 <span class="mono">llama_decode</span>（llama.h:986-987 的注释：' +
  '<span class="mono">Requires the context to have a memory</span>）；' +
  '<span class="mono">llama_encode</span> 恰好相反，注释里明写它不用 KV cache（977）。<br>' +
  '批次用 <span class="mono">llama_batch_init</span>（968）准备，用完 ' +
  '<span class="mono">llama_batch_free</span>（974）释放。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '四个文件，四种角色 —— 记住分工，比记住函数名有用。',
  '<span class="k">llama.h</span> 是唯一承诺：其它三个都不该被调用方直接依赖（llama-ext.h 除外，它本来就是给外部工具用的）。',
  '<span class="k">llama-cpp.h</span> 只服务 C++；用 C 写程序就得自己管释放。',
  '<span class="k">src/llama.cpp</span> 是入口，<span class="k">src/llama-ext.h</span> 是出口 —— 新 API 从这里长大。',
  '头文件注释里那段示例（1245-1274）已经把顺序写好了：<span class="v">建链 → decode → sample</span>。<br>本课的验收点就是能默画出这六步。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(13500, () => {
  rows.forEach(x => { x.className = ''; });
  hi('all');
  msg.innerHTML = texts[4];
});
tl.at(16800, () => {
  hi([2, 4]);
  msg.innerHTML = '最后一句留给下一课：<span class="k">本课只回答了"谁在承诺"</span>，'
    + '还没回答"模型里到底存了什么"。L2-02 从 hparams 开始拆。';
});
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、契约的边界：llama.h 的分段结构',
    '`llama.h` 用 `extern "C"` 把整个文件包起来（51-53），然后立刻声明四个**不透明类型**。'
    '从这一行开始，调用方与实现之间只剩指针：`llama_model` / `llama_context` / `llama_sampler` 的字段'
    '一个字都不在公共头文件里，**改内部布局不会破坏 ABI**。\n\n'
    '本课覆盖的四个文件，分工是：\n\n'
    '| 文件 | 行数 | 角色 |\n|---|---|---|\n'
    '| `include/llama.h` | 1646 | 唯一对外契约（241 处 `LLAMA_API` 声明） |\n'
    '| `include/llama-cpp.h` | 30 | C++ RAII 包装，开头 `#error` 拒绝 C |\n'
    '| `src/llama.cpp` | 620 | 契约的实现入口与装配处 |\n'
    '| `src/llama-ext.h` | 134 | staging 扩展头，允许破坏性变更 |\n',
    src=SRC_H, parts=[(51, 70)], lang='cpp')

L.section(
    '二、★ 三段式对象：三个不透明指针，三段生命周期',
    '三个类的创建与释放函数**不对称**，这是读公共 API 时最容易踩的地方：\n\n'
    '| 对象 | 创建 | 释放 |\n|---|---|---|\n'
    '| `struct llama_model` | `llama_model_load_from_file`（517） | `llama_model_free`（542） |\n'
    '| `struct llama_context` | `llama_init_from_model`（544） | `llama_free`（554） |\n'
    '| `struct llama_sampler` | `llama_sampler_chain_init`（1355） | `llama_sampler_free`（1350） |\n\n'
    '默认值不写在结构体里，而是由四个 `default_params` 函数给出（474-477）——'
    '这样默认值可以随实现版本变化，而头文件的字段布局保持稳定。',
    src=SRC_H, parts=[(472, 477)], lang='cpp')

L.section(
    '三、参数结构体族',
    '四个参数结构体按声明顺序排开：模型加载（314）、上下文创建（360）、'
    '离线量化（434）、采样链（457）。它们都是**按值传递**的，所以 `llama_context_params` 里'
    '专门写了一句注释：把 bool 集中放在结构体末尾，避免按值拷贝时错位。\n\n'
    '注意 `n_ctx` 这类字段只是"请求值"：创建之后要用 `llama_n_ctx(ctx)`（570）一族查询实际值。',
    src=SRC_H, parts=[(314, 330)], lang='cpp')

L.section(
    '四、★ 六步调用顺序',
    '这是本课的验收点。一个最小可用的 llama.cpp 程序按这个顺序调用：\n\n'
    '```text\n'
    '1  llama_backend_init()            进程级，一次\n'
    '2  llama_model_load_from_file()    读 GGUF -> llama_model\n'
    '3  llama_init_from_model()         建 llama_context（含 memory）\n'
    '4  llama_tokenize()                文本 -> token id\n'
    '5  llama_decode()                  跑图，产出 logits\n'
    '6  llama_sampler_sample()          从 logits 选 token\n'
    '```\n\n'
    '代码区把这六处声明从 1646 行头文件的六个段落里抽出来拼在一起；'
    '区间分隔行标出了各自的真实行号。',
    src=SRC_H,
    parts=[(482, 482), (517, 519), (544, 546), (1180, 1187), (998, 1000), (1548, 1548)],
    lang='cpp')

L.section(
    '五、输入与记忆：llama_batch 与 memory',
    '`llama_batch` 的七个字段都是并行数组，长度统一为 `n_tokens`；'
    '`n_seq_id` 与 `seq_id` 是二维的，所以一个 batch 能同时推进多条序列。\n\n'
    '`llama_decode` 的注释写明它**必须有 memory**（987），而 `llama_encode` 恰好相反（977）。'
    'memory 是 KV cache 之上的抽象：`llama_get_memory(ctx)`（585）取句柄，'
    '`llama_memory_seq_rm` / `seq_cp` / `seq_add`（756 / 765 / 780）按序列增删。',
    src=SRC_H, parts=[(247, 272)], lang='cpp')

L.section(
    '六、★ src/llama.cpp 是聚合入口',
    '它只有 620 行，却 include 了全部内部模块。它的角色是**把公共契约翻译成内部调用序列**：'
    '进程初始化、模型加载编排、chat 模板、分片路径。真正的重活分散在别的文件里 ——'
    '`llama_init_from_model` 在 `src/llama-context.cpp:3739`，`llama_decode` 在 `src/llama-context.cpp:4326`，'
    '`llama_tokenize` 在 `src/llama-vocab.cpp:4496`，`llama_sampler_sample` 在 `src/llama-sampler.cpp:895`。',
    src=SRC_CPP, parts=[(1, 17)], lang='cpp')

L.section(
    '七、C++ RAII 包装 llama-cpp.h',
    '30 行、四个 deleter、四个 `unique_ptr` 别名。它把"记得释放"从文档约定变成类型约束。'
    '注意 `llama_context_deleter` 调的是 `llama_free`（llama.h:554），而不是某个 `llama_context_free`。',
    src=SRC_CPPH, parts=[(1, 30)], lang='cpp')

L.section(
    '八、staging 头 llama-ext.h',
    '开头三行自述：这是 staging header，允许破坏性变更，一切都算 WIP（3-5）。'
    '它声明了建图入口 `llama_graph_reserve`（13）与量化状态 API（22-28）。\n\n'
    '第 5 行那句自律 ——"尽量别在代码库其它地方 include 这个头" —— **实测没做到**：'
    '`src/` 下已有 4 个文件 include 了它。源码注释不一定是真的，凡结论必实测。',
    src=SRC_EXT, parts=[(1, 28)], lang='cpp')

L.footnote_add('本课引用 `include/llama.h`、`include/llama-cpp.h`、`src/llama.cpp`、`src/llama-ext.h` '
               '四个文件，全部计入覆盖率。')
L.footnote_add('散文与图示里提到的内部定义位置（`src/llama-context.cpp:3739`、`src/llama-context.cpp:4326`、'
               '`src/llama-vocab.cpp:4496`、`src/llama-sampler.cpp:895`、`src/llama-quant.cpp:847`）'
               '由 `grep -n` 实测确认，但**不构成本课的代码引用**，故不计入本课覆盖率。')

L.prereqs('`L1-06`')

L.goal(
    '说出 `include/llama.h` / `include/llama-cpp.h` / `src/llama.cpp` / `src/llama-ext.h` 四个文件各自的角色（对应验收点）；',
    '默画出从 `llama_backend_init` 到 `llama_sampler_sample` 的六步调用顺序，并指出哪一步必须有 memory；',
    '说清 `llama_model` / `llama_context` / `llama_sampler` 三个不透明类型各自的创建与释放函数；',
    '解释为什么"默认值"由 `llama_*_default_params()` 函数给出，而不是写在结构体字段里。')

L.conclusion(
    '四个文件，四种角色',
    '| 文件 | 角色 | 关键内容 |\n|---|---|---|\n'
    '| `include/llama.h` | 公共 C ABI 契约 | 三个不透明类型、四个参数结构体、六步调用序 |\n'
    '| `include/llama-cpp.h` | C++ RAII 包装 | 四个 deleter + 四个 `unique_ptr` 别名 |\n'
    '| `src/llama.cpp` | 实现与装配入口 | `llama_backend_init` / 模型加载编排 / chat 模板 / 分片 |\n'
    '| `src/llama-ext.h` | staging 扩展头 | `llama_graph_reserve`、量化状态 API、显存账单 |')

L.conclusion(
    '★ 六步调用顺序',
    '```text\n'
    'llama_backend_init          482    进程级，一次\n'
    'llama_model_load_from_file  517    读 GGUF（容器格式见 L1-06）\n'
    'llama_init_from_model       544    建 memory / 调度器\n'
    'llama_tokenize              1180   第一个参数是 const llama_vocab *\n'
    'llama_decode                998    必须有 memory；这一步才碰图\n'
    'llama_sampler_sample        1548   在 CPU 上消费 logits，不进图\n'
    '```\n\n'
    '**只有 `llama_decode` 会真正跑图**；采样在 CPU 上、图之外完成。')

L.conclusion(
    '★ 聚合入口的地位',
    '`src/llama.cpp` 只有 620 行，却 include 了 `llama-impl.h`、`llama-version.h`、`llama-chat.h`、'
    '`llama-context.h`、`llama-mmap.h`、`llama-vocab.h`、`llama-model-loader.h`、`llama-model-saver.h`、'
    '`llama-model.h` 九个内部头 —— 用 `grep -rl` 逐个核对这些头的 include 清单，**交集只有它一个文件**。'
    '因此模型加载编排（`llama_model_load`，316）与设备准备落在这里。'
    '想找某个公共函数的实现，先看这里，再看它转发给了哪个模块。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
