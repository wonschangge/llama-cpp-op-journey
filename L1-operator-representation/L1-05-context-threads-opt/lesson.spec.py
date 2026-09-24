#!/usr/bin/env python3
"""L1-05 · 上下文、线程与优化器接口 —— 课件 spec。

运行：python3 L1-operator-representation/L1-05-context-threads-opt/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

GC   = 'ggml/src/ggml.c'
GCPP = 'ggml/src/ggml.cpp'
THR  = 'ggml/src/ggml-threading.cpp'
THRH = 'ggml/src/ggml-threading.h'
OPT  = 'ggml/src/ggml-opt.cpp'
OPTH = 'ggml/include/ggml-opt.h'

L = Lesson(
    id='L1-05',
    layer='L1 · 算子的表示',
    title='上下文、线程与优化器接口',
    codecap='ggml.c / ggml-opt.cpp / ggml-opt.h / ggml-threading.* / ggml.cpp（逐字引用）',
    nav={'prev': {'href': '../L1-04-quant-block-layout/index.html', 'label': 'L1-04 量化块结构'},
         'next': {'href': '../L1-06-gguf-container/index.html', 'label': 'L1-06 GGUF 容器'}},
)

L.note('**一句话**：`ggml_context` 不是一个"对象工厂"，而是**一块 arena（预分配的连续内存）+ 一条对象链表**。'
       '张量与计算图这些对象都**顺序摆进这块内存**，于是"建图"退化成指针加法，'
       '`ggml_free(ctx)` 是一次释放；代价是**没有释放单个张量的接口**。')
L.note('本课先把 arena 讲透（它是后面所有内存话题的地基），再看同一层的另外三个文件：'
       '`ggml-threading` 提供了什么并发原语、`ggml-opt` 的训练接口长什么样、'
       '以及 `ggml.cpp` 里到底装了什么。')
L.note('**一处实测纠正（重要）**：计划文档原本把这一课的要点写成"`ggml_context` 的对象分配、**线程池**"。'
       '逐行读完这五个文件后，实情是：`ggml-threading.cpp` **没有线程池**，'
       '它只提供一对全局临界区函数；`ggml_context` / `ggml_object` / `ggml_init` / `ggml_new_object` '
       '的实现也**不在这五个文件里**，而在 `ggml/src/ggml.c`。'
       '这一课按实测来讲，并把这次纠正本身当成一个教学点：**按文件名或计划猜内容，会猜错。**')
L.note('> 覆盖说明：本课需要 `ggml_context` 的实现来支撑"为什么用 arena"这条验收点，'
       '因此额外引用了 `ggml/src/ggml.c`（它本就由 L1-02 / L1-03 / L4-04 共享）。'
       '重叠覆盖不影响覆盖度门禁（并集判定）。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L1 · 算子的表示',
    title='五个文件，一个问题：<span class="hl-a">对象住在哪里</span>',
    sub='context 管内存、threading 管并发、ggml-opt 管训练；ggml.cpp 其实是进程级兜底。',
    caption='回顾 L1-01：ggml_tensor 是定长值类型 —— 正因为定长，它才能被整块塞进 arena。',
    src=OPTH, parts=[(196, 206)], duration=17000,
mark_src=[201, 203],
    notes_src={201: 'no_alloc == true：图里的张量只占"对象头"，数据不从这里出',
           203: '两个 context：一个放参数与输入（静态），一个放其余（自动重分配）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="flow" style="justify-content:center">' +
    '<span class="chip">模型定义</span><span class="arrow">-></span>' +
    '<span class="chip a">ggml_context</span><span class="arrow">-></span>' +
    '<span class="chip b">ggml_cgraph</span><span class="arrow">-></span>' +
    '<span class="chip c">后端执行</span></div>' +
  '<div class="row wrap center" id="faces" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'ggml/src/ggml.c —— 主角', b: 'context 的实现：<br>一块 arena + 一条对象链表', m: 'ggml_init · ggml_new_object' },
  { c: 'b', t: 'ggml-opt.cpp / ggml-opt.h', b: '训练接口：dataset、opt_context、<br>epoch、fit', m: 'ggml_opt_fit' },
  { c: 'f', t: 'ggml-threading.cpp / .h', b: '并发原语：<br>只有一对临界区函数', m: 'ggml_critical_section_start' },
  { c: 'g', t: 'ggml/src/ggml.cpp', b: '与 context 无关：<br>进程级 terminate 兜底', m: 'ggml_uncaught_exception' }
];
const host = wrap.querySelector('#faces');
const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '先看四个角色的分工。第一张卡就提醒一件事：<span class="v">ggml_context</span> 的实现<b>不在</b>本课的五个文件里。',
  'arena 的实现在 <span class="v">ggml/src/ggml.c</span> —— 本课额外引用了它，否则"为什么用 arena"讲不出证据。',
  '<span class="v">ggml-opt</span> 是 context 最大的一位客户：<b>它自己同时管四个 context</b>。',
  '<span class="v">ggml-threading</span> 一共 12 行。本课如实讲它提供了什么，也讲它<b>没有</b>提供什么 —— 它不是线程池。',
  '五个文件，三个问题：<b>对象住在哪里、谁能并发碰它、训练时怎么用它</b>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14500, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L1-05 · context',
    title='★ <span class="hl-a">struct ggml_context</span>：一块内存 + 一条对象链表',
    sub='七个字段就够了：有多大、在哪、是不是自己的、要不要分配数据、几个对象、头、尾。',
    caption='示意图：对象头 + 对齐后的负载，从 mem_buffer 头部起一个挨一个往后摆，没有空闲链表。',
    src=GC, parts=[(958, 985)], duration=25000,
mark_src=[959, 960, 962, 964, 969, 975, 976, 977, 978, 979, 983, 984],
    notes_src={960: 'offs / size：这个对象在 mem_buffer 里的偏移，与对齐后占用的字节数',
           969: 'GGML_OBJECT_SIZE：对象头本身的字节数，后面每个偏移都要加上它',
           976: 'mem_size：这块 arena 一共多少字节，ggml_init 之后不再变',
           979: 'no_alloc：true 时只放对象头，张量数据交给后端 buffer（ggml-opt 全程用它）',
           984: 'objects_begin / objects_end：链表首尾；end 同时就是下一幕要用的 bump 指针'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = '<div class="cm" style="margin:0">mem_buffer —— 一次分配拿到的整块 arena（mem_size 字节）</div>' +
  '<div class="row" id="arena" style="gap:3px;align-items:stretch"></div>' +
  '<div class="row center" id="ptr" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);
const focus = ls => {
  const ms = document.querySelectorAll('mark.ln-mark');
  let first = null;
  for (let i = 0; i < ms.length; i++) {
    const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
    ms[i].className = on ? 'ln-mark on' : 'ln-mark';
    ms[i].style.display = 'inline-block';
    if (on && !first) first = ms[i];
  }
  if (first) first.scrollIntoView({ block: 'nearest' });
};

const segs = [
  { t: 'obj 头', w: 52, c: 'a' },
  { t: 'ggml_tensor', w: 100, c: 'b' },
  { t: 'obj 头', w: 52, c: 'a' },
  { t: 'ggml_tensor', w: 100, c: 'b' },
  { t: 'obj 头', w: 52, c: 'a' },
  { t: 'ggml_cgraph', w: 100, c: 'd' },
  { t: '还没走到', w: 170, c: 'x' }
];
const arena = wrap.querySelector('#arena');
const boxes = segs.map(s => {
  const free = (s.c === 'x');
  const e = U.el('div', { style: 'width:' + s.w + 'px;height:34px;border:1px solid var(--' +
    (free ? 'border' : s.c) + ');border-radius:4px;background:' +
    (free ? '#10151b' : 'rgba(88,166,255,.10)') +
    ';display:flex;align-items:center;justify-content:center;font-family:var(--mono);' +
    'font-size:8.5px;color:var(--muted);text-align:center;padding:2px' });
  e.textContent = s.t;
  arena.appendChild(e);
  return e;
});
const ptr = wrap.querySelector('#ptr');
ptr.innerHTML = '<span class="chip a">objects_begin</span>' +
  '<span class="chip d">objects_end = bump 指针</span>' +
  '<span class="chip">n_objects</span>';

const msg = wrap.querySelector('#msg');
const texts = [
  'arena 里<b>没有空闲链表</b>：对象一个挨一个往后摆，靠 <span class="k">offs + size</span> 找到自己。',
  '<span class="v">struct ggml_object</span> 就是对象头：<span class="v">offs</span>（在 mem_buffer 里的偏移）、' +
    '<span class="v">size</span>（对齐后的字节数）、<span class="v">next</span>（链到下一个对象）、' +
    '<span class="v">type</span>（是张量、图，还是工作缓冲）。',
  '<span class="v">GGML_OBJECT_SIZE = sizeof(struct ggml_object)</span>：每段负载前面都要先让出这么多字节。<br>' +
    '末尾那个 <span class="v">char padding[4]</span> 就是用来把头部长度补齐对齐的。',
  '<span class="v">struct ggml_context</span> 只有 7 个字段：<br>' +
    '有多大（<span class="v">mem_size</span>）、在哪（<span class="v">mem_buffer</span>）、' +
    '是不是自己的（<span class="v">mem_buffer_owned</span>）、要不要分配数据（<span class="v">no_alloc</span>）、' +
    '几个对象（<span class="v">n_objects</span>）、链表头、链表尾。',
  '<span class="v">objects_end</span> 同时就是 <b>bump 指针</b>：下一个对象从它后面开始。<br>' +
    '下一幕的 <span class="v">ggml_new_object</span> 就是照这个算的 —— <b>建图 = 指针加法</b>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; focus([1, 2, 5, 7]); });
tl.at(5200, () => { msg.innerHTML = texts[1]; focus([5, 7]); });
tl.at(9800, () => { msg.innerHTML = texts[2]; focus([12]); });
tl.at(14400, () => { msg.innerHTML = texts[3]; focus([19, 20, 22, 23, 24, 29, 30]); });
tl.at(20200, () => {
  boxes[0].style.background = 'rgba(88,166,255,.30)';
  boxes[2].style.background = 'rgba(88,166,255,.30)';
  boxes[4].style.background = 'rgba(88,166,255,.30)';
  msg.innerHTML = texts[4];
  focus([29, 30]);
});
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L1-05 · context',
    title='<span class="hl-b">ggml_init</span>：整块 arena 只申请一次',
    sub='context 结构自己是一次 malloc；arena 是一次 malloc；两件事都只做一次。',
    caption='注意第一个动作是进临界区 —— 那是本课里 threading 文件的用处，第 8 幕展开。',
    src=GC, parts=[(1611, 1651)], duration=24000,
mark_src=[1614, 1625, 1632, 1636, 1637, 1638, 1641, 1644, 1646],
    notes_src={1628: 'mem_size 为 0 时给一个最小兜底，所以 ggml_init 允许"0 大小"的 context',
           1632: '调用者给了 mem_buffer 就原样用；没给就把 mem_size 补齐对齐后自己分配',
           1636: 'mem_buffer 是 arena 的基址，后面所有对象都在它上面做偏移',
           1637: 'mem_buffer_owned：决定 ggml_free 到底要不要真去 free 这块内存',
           1638: 'no_alloc 原样抄进 context：它控制张量数据是否也从 arena 里出',
           1641: '链表头尾都是 NULL —— 这是一个刚开的、空的对象世界'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
  '<div id="tbl"></div><div class="formula" id="msg"></div>';
root.appendChild(wrap);
const focus = ls => {
  const ms = document.querySelectorAll('mark.ln-mark');
  let first = null;
  for (let i = 0; i < ms.length; i++) {
    const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
    ms[i].className = on ? 'ln-mark on' : 'ln-mark';
    ms[i].style.display = 'inline-block';
    if (on && !first) first = ms[i];
  }
  if (first) first.scrollIntoView({ block: 'nearest' });
};

const defs = [
  { c: 'c', t: '第一步 · 临界区', m: 'ggml_critical_section_start()', b: '只保护"首次调用初始化时间系统"这一段，保护完立刻退出' },
  { c: 'a', t: '第二步 · context 自己', m: 'GGML_MALLOC(sizeof(struct ggml_context))', b: '注意：这一次 malloc 拿到的只是那 7 个字段，不是 arena' },
  { c: 'b', t: '第三步 · arena', m: 'ggml_aligned_malloc(mem_size)', b: '整块内存<b>一次</b>拿到；调用者自带 mem_buffer 时这一步跳过' },
  { c: 'e', t: '第四步 · 变成空世界', m: 'objects_begin = objects_end = NULL', b: '还没有任何对象，链表的头尾都是空' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const t = U.table(
  ['ggml_init_params 字段', '在这里变成什么', '谁说了算'],
  [['mem_size', 'arena 的字节数（0 会被兜底）', '调用者预算是多少'],
   ['mem_buffer', 'arena 的基址（NULL 就自己分配）', '调用者带不带内存来'],
   ['no_alloc', '原样抄进 context', '调用者要不要 arena 存数据']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');
rows.forEach(r => { r.className = ''; });

const msg = wrap.querySelector('#msg');
const texts = [
  'ggml_init 只做四件事。先把"内存"和"对象"分开看：',
  '<span class="v">GGML_MALLOC(sizeof(struct ggml_context))</span>：<b>这一次 malloc 拿的是结构体本身</b>，不是 arena。',
  '<span class="v">mem_size</span> 是调用者给的预算。给 0 也允许 —— 会被兜底成 <span class="v">GGML_MEM_ALIGN</span>。',
  '<span class="v">ggml_aligned_malloc(mem_size)</span>：<b>arena 整块只申请这一次</b>。<br>' +
    '调用者自带 <span class="v">mem_buffer</span> 时就借用，连这一次都省掉（于是 <span class="v">mem_buffer_owned = false</span>）。',
  '收尾：<span class="v">n_objects = 0</span>，链表头尾都是 <span class="v">NULL</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
  if (i === 1) { rows.forEach((r, k) => { r.className = (k === 0) ? 'on' : ''; }); focus([14]); }
  if (i === 2) { rows.forEach((r, k) => { r.className = (k <= 1) ? 'on' : ''; }); focus([22]); }
  if (i === 3) { rows.forEach(r => { r.className = 'on'; }); focus([27, 29, 31, 35, 39, 41]); }
  if (i === 0) { focus([3]); }
}));
tl.at(16600, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  rows.forEach(r => { r.className = ''; });
  msg.innerHTML = texts[4];
  focus([35]);
});
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L1-05 · context',
    title='★ 释放是 <span class="hl-c">O(1)</span>：一次 free，或者一次置空',
    sub='ggml_free 只 free 一次内存块；ggml_reset 连 free 都不做，把两个指针抹掉就整块可再用。',
    caption='对照上一幕：既然只有"一次分配"，释放自然也只需要"一次释放"。',
    src=GC, parts=[(1653, 1677)], duration=19000,
mark_src=[1658, 1659, 1660, 1668, 1669, 1672, 1676],
    notes_src={1660: 'reset：只把首尾指针置 NULL —— 整块内存立刻可再用，一个对象都没 free',
           1670: 'free 块里只有这一次真释放；借来的 buffer（owned = false）连这次都没有',
           1672: 'GGML_FREE(ctx)：连那 7 个字段的结构体自己也一起放掉'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>' +
  '<div class="flow center" id="flow" style="justify-content:center"></div>';
root.appendChild(wrap);
const focus = ls => {
  const ms = document.querySelectorAll('mark.ln-mark');
  let first = null;
  for (let i = 0; i < ms.length; i++) {
    const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
    ms[i].className = on ? 'ln-mark on' : 'ln-mark';
    ms[i].style.display = 'inline-block';
    if (on && !first) first = ms[i];
  }
  if (first) first.scrollIntoView({ block: 'nearest' });
};

const defs = [
  { c: 'c', t: 'ggml_reset(ctx)', m: 'n_objects = 0; begin = end = NULL', b: '整块 arena <b>当场作废、当场可再用</b>。<br>不 free、不清零、不看内容 —— 代价是里面所有旧指针一起失效。' },
  { c: 'b', t: 'ggml_free(ctx)', m: 'ggml_aligned_free(mem_buffer, mem_size)', b: '整块只释放这一次。<br>借来的 buffer（<span class="cm" style="margin:0">owned = false</span>）连这一次都没有。' },
  { c: 'd', t: 'ggml_used_mem(ctx)', m: 'objects_end->offs + objects_end->size', b: '用了多少 = 最后一个对象的末端。<br><b>bump 指针的身高就是用量表。</b>' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const flow = wrap.querySelector('#flow');
flow.innerHTML = '<span class="chip c">ggml_reset</span><span class="arrow">:</span>' +
  '<span class="chip">O(1)</span><span class="arrow">|</span>' +
  '<span class="chip b">ggml_free</span><span class="arrow">:</span>' +
  '<span class="chip">O(1)</span><span class="arrow">|</span>' +
  '<span class="chip d">ggml_used_mem</span><span class="arrow">:</span>' +
  '<span class="chip">O(1)</span>';

const msg = wrap.querySelector('#msg');
const texts = [
  '释放这一侧和分配一样短。三个函数：',
  '<span class="v">ggml_reset</span> 把 <span class="v">n_objects / objects_begin / objects_end</span> 各抹一次 —— ' +
    '<b>整块内存立刻可再用</b>，但一个对象也没被 free。',
  '<span class="v">ggml_free</span> 只在 <span class="v">mem_buffer_owned</span> 为真时 free 一次，' +
    '然后 <span class="v">GGML_FREE(ctx)</span> 放掉结构体自己。',
  '<span class="v">ggml_used_mem</span> 直接返回最后一个对象的末端：' +
    '<span class="v">objects_end->offs + objects_end->size</span>。<b>用量不用记账，指针自己就是账本。</b>',
  '一句话：<span class="k">arena 的分配与释放都是 O(1)，代价是没有"释放其中一个对象"这回事</span>。<br>' +
    '下一幕看 ggml-opt 为这个代价付出了什么。'
];
tl.at(700, () => {
  els.forEach((e, k) => { e.style.opacity = k === 0 ? '1' : '.30'; });
  msg.innerHTML = texts[0];
  focus([5, 6, 7]);
});
tl.at(4600, () => {
  els.forEach((e, k) => { e.style.opacity = k === 1 ? '1' : '.30'; });
  msg.innerHTML = texts[1];
  focus([5, 6, 7]);
});
tl.at(8500, () => {
  els.forEach((e, k) => { e.style.opacity = k === 2 ? '1' : '.30'; });
  msg.innerHTML = texts[2];
  focus([16, 17, 21]);
});
tl.at(12400, () => {
  msg.innerHTML = texts[3];
  focus([26]);
});
tl.at(15800, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
  focus([]);
});
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L1-05 · 核心',
    title='★ <span class="hl-a">ggml_new_object</span>：bump 分配的全过程',
    sub='取当前末端、对齐、算新位置、查预算、写对象头、串到链表尾 —— 没有查找，没有空闲表。',
    caption='这就是"建图 = 指针加法"的字面实现：新对象地址 = mem_buffer + cur_end。',
    src=GC, parts=[(1708, 1759)], duration=27000,
mark_src=[1710, 1712, 1714, 1718, 1720, 1721, 1734, 1738, 1744, 1753, 1756, 1759],
    notes_src={1714: 'cur_end = 上一个对象的 offs + size —— 这就是 bump 指针的当前值',
           1718: '先对齐到 GGML_MEM_ALIGN，再谈放得下放不下',
           1721: '没有任何查找：新对象直接落在 mem_buffer + cur_end',
           1741: '放不下就告警并返回 NULL（debug 版直接 abort）—— arena 不会扩容',
           1744: 'offs 特意跳过对象头本身：负载从对象头之后开始'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);
const focus = ls => {
  const ms = document.querySelectorAll('mark.ln-mark');
  let first = null;
  for (let i = 0; i < ms.length; i++) {
    const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
    ms[i].className = on ? 'ln-mark on' : 'ln-mark';
    ms[i].style.display = 'inline-block';
    if (on && !first) first = ms[i];
  }
  if (first) first.scrollIntoView({ block: 'nearest' });
};

const defs = [
  { c: 'a', t: '1 · 取末端', m: 'cur_end = offs + size', b: '从 <span class="cm" style="margin:0">ctx->objects_end</span> 读。<br>空链表时是 0。' },
  { c: 'b', t: '2 · 对齐', m: 'GGML_PAD(size, GGML_MEM_ALIGN)', b: '先把要放的字节数补齐，<br>后面所有偏移都按它算。' },
  { c: 'c', t: '3 · 算地址', m: 'mem_buffer + cur_end', b: '新对象头就落在这儿。<br><b>没有查找，没有空闲表。</b>' },
  { c: 'f', t: '4 · 查预算', m: 'cur_end + size + OBJECT_SIZE > mem_size', b: '超出就告警并返回 NULL。<br><b>arena 不会扩容。</b>' },
  { c: 'd', t: '5 · 写头', m: 'offs = cur_end + GGML_OBJECT_SIZE', b: 'offs 跳过对象头；<br>负载紧跟在头后面。' },
  { c: 'e', t: '6 · 串链', m: 'obj_cur->next = obj_new', b: '首对象时改写 <span class="cm" style="margin:0">objects_begin</span>；<br>无论如何都改写 <span class="cm" style="margin:0">objects_end</span>。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:214px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.26');

const msg = wrap.querySelector('#msg');
const texts = [
  '整个分配过程六步，全程只有算术和一次比较。',
  '<span class="v">cur_offs / cur_size / cur_end</span>：bump 指针的当前值就是"上一个对象的末端"。',
  '<span class="v">GGML_PAD(size, GGML_MEM_ALIGN)</span>：先对齐，再判断装不装得下。',
  '<span class="v">mem_buffer + cur_end</span>：<b>新对象的位置是一次指针加法算出来的</b>，没有任何查找。',
  '<span class="v">cur_end + size_needed + GGML_OBJECT_SIZE &gt; ctx->mem_size</span> 就告警返回 NULL。' +
    '<br>这就是 arena 的"预算超支"：<b>没有扩容，只有失败</b>。',
  '<span class="v">.offs = cur_end + GGML_OBJECT_SIZE</span>：负载从对象头之后开始，所以 <span class="v">offs</span> 要跳过头部。',
  '最后把它挂到链表尾：<b>对象链表就是 arena 里唯一的索引结构</b>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.26'; });
  msg.innerHTML = texts[i];
  focus([[2, 4, 6], [6], [11], [14, 15], [29, 33], [40], [50, 53, 56]][i]);
}));
tl.at(23000, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[6];
  focus([]);
});
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L1-05 · 代价',
    title='★ arena 的代价：<span class="hl-e">ggml-opt</span> 为什么要四个 context',
    sub='张量只能跟着整块 arena 一起死。要按生命周期分组，唯一的办法是分组开 context。',
    caption='这里的关键选择不是数据结构，而是"哪些张量共用一次生死"。',
    src=OPT, parts=[(30, 37), (584, 594)], duration=21000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = '<div class="row center" id="ctxs" style="gap:7px"></div>' +
  '<div class="row center" id="frees" style="gap:6px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'ctx_static', b: '梯度、优化器动量、loss' },
  { c: 'b', t: 'ctx_cpu', b: '优化器参数（1 个张量）' },
  { c: 'c', t: 'ctx_compute', b: '调用者的临时张量' },
  { c: 'd', t: 'ctx_copy', b: '静态图的副本' }
];
const host = wrap.querySelector('#ctxs');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const frees = wrap.querySelector('#frees');
frees.innerHTML = '<span class="chip">ggml_backend_buffer_free x2</span>' +
  '<span class="arrow">-></span><span class="chip a">ggml_free(ctx_static)</span>' +
  '<span class="chip b">ggml_free(ctx_cpu)</span>' +
  '<span class="chip d">ggml_free(ctx_copy)</span>';

const msg = wrap.querySelector('#msg');
const texts = [
  'ggml_opt_context 里有 <b>四个</b> ggml_context 字段。这不是随手写的。',
  '回顾第 4 幕：能释放的粒度只有"整个 context"。<span class="v">ggml_reset</span>/<span class="v">ggml_free</span> 都不认识单个张量。',
  '所以"想让一批张量一起死"，就得让它们<b>共用一个 context</b>：<br>' +
    '<span class="v">ctx_static</span> 放梯度与动量，<span class="v">ctx_cpu</span> 只放一个优化器参数张量，<span class="v">ctx_copy</span> 放静态图副本。',
  '<span class="v">ggml_opt_free</span> 里就是三次 <span class="v">ggml_free</span> —— ' +
    '<b>释放一个训练上下文 = 释放三块 arena</b>，仍然是 O(1) 次调用。',
  '这就是 arena 的取舍：<span class="k">分配与释放极快，但"按张量释放"这个能力被换掉了</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(4300, () => { msg.innerHTML = texts[1]; });
els.forEach((_, i) => tl.at(8000 + i * 2400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[2];
}));
tl.at(17600, () => { msg.innerHTML = texts[3]; });
tl.at(19000, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L1-05 · 代价',
    title='arena 的另一面：内存要<span class="hl-c">提前算出来</span>',
    sub='mem_size 是 ggml_init 的参数，所以调用者必须先知道"会有几个张量"。',
    caption='呼应 L1-01：张量是定长值类型，在 arena 里的占用不随形状变 —— 预算才能按"个数"算。',
    src=OPT, parts=[(346, 364)], duration=21000,
mark_src=[347, 354, 355, 356, 357, 359, 361, 363],
    notes_src={357: 'size_meta：先数张量个数，再乘每个张量的固定开销 —— arena 的预算是这么来的',
           363: '拿算出来的字节数去 ggml_init：arena 的大小是"算"出来的，不是"长"出来的'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div id="tbl"></div><div class="formula" id="msg"></div>';
root.appendChild(wrap);
const focus = ls => {
  const ms = document.querySelectorAll('mark.ln-mark');
  let first = null;
  for (let i = 0; i < ms.length; i++) {
    const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
    ms[i].className = on ? 'ln-mark on' : 'ln-mark';
    ms[i].style.display = 'inline-block';
    if (on && !first) first = ms[i];
  }
  if (first) first.scrollIntoView({ block: 'nearest' });
};

const t = U.table(
  ['预算项', '数量', '它是什么'],
  [['n_loss', '1', '损失项（每个 loss 一个梯度）'],
   ['tensors_per_param x n_param', '(1 或 0) + (2 或 0)', '每个参数：累积梯度 1 个；AdamW 再要 m / v 两个动量'],
   ['tensors_const', 'static_graphs ? 9 : 0', 'labels / loss / pred / ncorrect 等固定项'],
   ['x ggml_tensor_overhead()', '每个的固定开销', '对象头 + 定长张量体（与形状无关）']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  'arena 要一次 malloc，就必须先知道要多少字节。于是有了这份预算表：',
  '<span class="v">n_param</span> 是遍历 forward 图数出来的 —— ' +
    '带 <span class="v">GGML_TENSOR_FLAG_PARAM</span> 的节点有几个。',
  '<span class="v">tensors_per_param</span> 取决于两个开关：要不要累积梯度、优化器是不是 AdamW（要两个动量）。',
  '<span class="v">tensors_const</span> 是 loss / labels / pred / ncorrect 这一批固定项。',
  '三者相加，<b>再乘每个张量的固定开销</b> <span class="v">ggml_tensor_overhead()</span> —— 就是 <span class="v">mem_size</span>。',
  '关键点：<span class="k">这个乘法成立，是因为一个张量在 arena 里的占用不随形状变</span>（L1-01：ne/nb/src 都是定长数组）。'
];
function show(i) {
  rows.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i];
}
tl.at(700, () => { show(0); rows.forEach(r => { r.className = ''; }); });
tl.at(4300, () => { show(1); rows.forEach((r, k) => { r.className = (k <= 0) ? 'on' : ''; }); });
tl.at(8000, () => { show(2); rows.forEach((r, k) => { r.className = (k === 1) ? 'on' : ''; }); });
tl.at(11700, () => { show(3); rows.forEach((r, k) => { r.className = (k === 2) ? 'on' : ''; }); });
tl.at(15400, () => { show(4); rows.forEach((r, k) => { r.className = (k === 3) ? 'on' : ''; }); focus([8, 9, 10, 11, 14, 16, 18]); });
tl.at(18500, () => { show(5); rows.forEach((r, k) => { r.className = (k === 3) ? 'on' : ''; }); });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L1-05 · 线程',
    title='<span class="hl-f">ggml-threading</span> 到底提供了什么：一把全局互斥锁',
    sub='整个实现文件 12 行、整个头文件 14 行。它没有线程池，也没有任务队列。',
    caption='本课引用到的调用点：ggml/src/ggml.c 的 ggml_init（第 3 幕代码块第 4、13 行）；它不是全仓唯一一处。',
    src=THR, parts=[(1, 12)], duration=17000,
mark_src=[1, 4, 6, 7, 10, 11],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);
const focus = ls => {
  const ms = document.querySelectorAll('mark.ln-mark');
  let first = null;
  for (let i = 0; i < ms.length; i++) {
    const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
    ms[i].className = on ? 'ln-mark on' : 'ln-mark';
    ms[i].style.display = 'inline-block';
    if (on && !first) first = ms[i];
  }
  if (first) first.scrollIntoView({ block: 'nearest' });
};

const defs = [
  { c: 'f', t: '就这些（.cpp，12 行）', b: '一个 <span class="cm" style="margin:0">std::mutex</span>，<br>两个薄封装函数。', m: 'ggml_critical_section_start/end' },
  { c: 'a', t: '就这些（.h，14 行）', b: '头文件里只声明这两个函数，<br>没有别的公开 API。', m: 'GGML_API void ggml_critical_section_start' },
  { c: 'g', t: '它<b>没有</b>提供的', b: '线程池、任务队列、worker、<br>原子计数器 —— 一个都没有。', m: '（实测：grep 与逐行阅读）' },
  { c: 'c', t: '本课看到的调用点', b: 'ggml_init 里保护"首次初始化<br>时间系统"那一段。', m: 'ggml/src/ggml.c:1614, 1623' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '把整个文件读完只需要 12 行。<b>它提供的东西少得让人意外。</b>',
  '<span class="v">std::mutex ggml_critical_section_mutex</span>：一个进程级的全局锁，没有名字空间、没有分层。',
  '<span class="v">start()</span> 就是 <span class="v">lock()</span>，<span class="v">end()</span> 就是 <span class="v">unlock()</span> —— ' +
    '不含 RAII、不含超时、不含重入。<b>调用者必须自己保证配对。</b>',
  '这把锁在仓库里还有别的调用点（不在本课引用范围内），形态都一样：<b>保护"首次调用时初始化一张共享表"</b>。',
  '这一课的教训：<span class="k">计划/目录名会骗人，只有逐行读过的文件不会</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; focus([0, 3]); });
tl.at(4200, () => { msg.innerHTML = texts[1]; focus([3]); });
tl.at(7700, () => { msg.innerHTML = texts[2]; focus([5, 6, 9, 10]); });
tl.at(11200, () => { msg.innerHTML = texts[3]; focus([]); });
tl.at(13800, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L1-05 · 兜底',
    title='<span class="hl-g">ggml.cpp</span>：这个文件其实与 context 无关',
    sub='26 行，只做一件事：没人接住的异常，先打 backtrace，再 abort。',
    caption='本课覆盖的五个文件中，只有它不参与对象分配 —— 如实讲它实际做了什么。',
    src=GCPP, parts=[(6, 26)], duration=17000,
mark_src=[8, 13, 17, 24, 25],
    notes_src={13: '打印调用栈之后仍然 abort —— 这个兜底不吞异常，只让它留下线索',
           17: 'GGML_NO_BACKTRACE 环境变量：设了就完全不装这个 handler'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div class="row wrap" id="cards" style="gap:8px"></div>' +
  '<div class="formula" id="msg"></div>';
root.appendChild(wrap);
const focus = ls => {
  const ms = document.querySelectorAll('mark.ln-mark');
  let first = null;
  for (let i = 0; i < ms.length; i++) {
    const on = ls.indexOf(+ms[i].getAttribute('data-l')) >= 0;
    ms[i].className = on ? 'ln-mark on' : 'ln-mark';
    ms[i].style.display = 'inline-block';
    if (on && !first) first = ms[i];
  }
  if (first) first.scrollIntoView({ block: 'nearest' });
};

const defs = [
  { c: 'g', t: '它做了什么', b: '静态初始化时把 <span class="cm" style="margin:0">std::terminate</span> 换掉；<br>异常没人接住时先 <span class="cm" style="margin:0">ggml_print_backtrace()</span>。', m: 'ggml_uncaught_exception' },
  { c: 'c', t: '它不做什么', b: '不分配、不释放、<br>不碰 context、不碰线程。', m: '（26 行全文）' },
  { c: 'b', t: '开关', b: '<span class="cm" style="margin:0">GGML_NO_BACKTRACE</span> 环境变量存在，<br>就直接不装 handler。', m: 'getenv("GGML_NO_BACKTRACE")' },
  { c: 'd', t: '为什么值得看', b: '读完它才知道：<br><b>文件名不等于内容</b>。', m: 'static bool ... = []{ ... }()' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:300px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '这一课覆盖的五个文件里，ggml.cpp 是最容易被误判的一个。',
  '<span class="v">GGML_NORETURN static void ggml_uncaught_exception()</span>：' +
    '先 <span class="v">ggml_print_backtrace()</span>，再调用上一个 terminate handler，最后 <span class="v">abort()</span>。',
  '<span class="v">previous_terminate_handler</span> 保存原来的处理器，' +
    '<span class="v">std::set_terminate(ggml_uncaught_exception)</span> 把它换掉 —— <b>在静态初始化阶段完成</b>。',
  '注意那个 <span class="v">GGML_ASSERT(prev != ggml_uncaught_exception)</span>：' +
    '<b>防止这个 handler 被装两次造成自递归</b>。',
  '所以它和 context、线程、优化器都无关 —— 它是整个 ggml 的<b>进程级兜底</b>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3700, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
  focus([[2, 7], [], [12], [20, 21]][i]);
}));
tl.at(15200, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[4];
  focus([]);
});
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L1-05 · 收束',
    title='压成一张表：arena 换来了什么、换走了什么',
    sub='一次分配、一次释放、O(1) 的建图；代价是没有按张量释放，以及内存要提前预算。',
    caption='下一课 L1-06：这些张量的权重从 GGUF 文件里来。',
    src=OPTH, parts=[(238, 251)], duration=19000,
mark_src=[239, 241, 249, 250],
    notes_src={241: 'ctx_compute：只放"临时张量"的那个 context；参数与输入的那个在调用者手上',
           250: 'val_split：数据集后段按这个比例做验证，对应 ggml_opt_epoch 的 idata_split'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = '<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>';
root.appendChild(wrap);

const t = U.table(
  ['机制', '在这一课里看到的实际做法', '去哪一课展开'],
  [['arena', '一块内存 + 一条对象链表；ggml_init 只 malloc 一次', '本课 · 第 2/3 幕'],
   ['bump 分配', 'ggml_new_object：mem_buffer + cur_end，顺序摆放', '本课 · 第 5 幕'],
   ['O(1) 释放', 'ggml_free 只 free 一次；ggml_reset 只把两个指针置空', '本课 · 第 4 幕'],
   ['不能单独释放', 'ggml-opt 用四个 context 按生命周期分组', 'L4-01 分配器'],
   ['内存要预算', 'mem_size = 张量个数 x ggml_tensor_overhead()', 'L1-01 定长张量'],
   ['并发原语', '一个全局 std::mutex，两个薄封装函数', '本课 · 第 8 幕']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '<span class="mono">ggml_opt</span> 要给静态 context 预留多少字节？为什么可以这样算？',
  '代码里是 <span class="mono">size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead()</span>：' +
  '<b>先数出会创建多少个张量，再乘每个张量的固定开销</b>。<br>' +
  '能这样算的前提是 <b>L1-01</b>：<span class="mono">ggml_tensor</span> 是<b>定长值类型</b>' +
  '（<span class="mono">ne[4]</span> / <span class="mono">nb[4]</span> / <span class="mono">src[10]</span> 都是定长数组），' +
  '它在 arena 里的占用不随形状变化。<br>' +
  '换句话说：<b>arena 的预算是按"个数"算的，不是按"形状"算的</b>。若张量是变长的，这个乘法就不成立了。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '六行，就是这一课的全部：',
  '<span class="k">一次分配</span> + <span class="k">一次释放</span> 换来 <span class="k">指针加法的建图速度</span>。',
  '换走的是两件事：<b>不能单独释放一个张量</b>，以及 <b>mem_size 必须提前算出来</b>。',
  '所以 L4-01 的 ggml-alloc 要在 arena 之上再做一层：它负责把"同时活着的张量"复用同一段内存。',
  '记忆锚点：<span class="v">ggml_context 是对象世界的边界，ggml_free 是这个世界的唯一出口</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(17600, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、★ struct ggml_context：一块 arena + 一条对象链表',
    '`ggml_context` 的全部状态就是这七个字段。它**不持有**任何"分配器"结构：'
    '没有空闲链表、没有分桶、没有引用计数 —— 只有一块内存（`mem_buffer` / `mem_size`）'
    '和一条把对象串起来的单链表（`objects_begin` / `objects_end`）。\n\n'
    '每个对象前面都压着一个 `struct ggml_object` 头部，它记录两件事：'
    '**这个对象在 mem_buffer 里从哪开始**（`offs`）、**占多少字节**（`size`）。'
    '`GGML_OBJECT_SIZE` 就是头部自身的字节数 —— 后面所有的偏移都要加上它。\n\n'
    '注意 `struct ggml_object` 末尾那个 `char padding[4]`：它不是数据，'
    '是为了让头部长度对齐（`ggml/src/ggml.c:1262` 另有一条 `static_assert`，断言头部大小是 `GGML_MEM_ALIGN` 的整数倍）。',
    src=GC, parts=[(958, 985)], lang='c')

L.section(
    '二、一次 malloc：ggml_init',
    '`ggml_init` 里其实有**两次** malloc，但都不是"每个对象一次"：\n\n'
    '```text\n'
    'GGML_MALLOC(sizeof(struct ggml_context))   <- 结构体自己，一次\n'
    'ggml_aligned_malloc(mem_size)              <- arena 整块，一次\n'
    '```\n\n'
    '第二次可以完全没有：如果调用者自带 `mem_buffer`，就直接借用，'
    '并把 `mem_buffer_owned` 置为 false（于是 `ggml_free` 也不会去 free 它）。\n\n'
    '`mem_size` 为 0 是允许的，会被兜底成 `GGML_MEM_ALIGN`。'
    '函数开头那段临界区保护的是"首次调用时初始化时间系统"，'
    '这也是本课引用到的 `ggml-threading` 用法。',
    src=GC, parts=[(1611, 1651)], lang='c')

L.section(
    '三、O(1) 的释放：ggml_reset / ggml_free / ggml_used_mem',
    '分配是一次，释放自然也是一次。三个函数都很短：\n\n'
    '| 函数 | 做什么 | 复杂度 |\n|---|---|---|\n'
    '| `ggml_reset` | 把 `n_objects` / `objects_begin` / `objects_end` 各抹一次 | O(1)，且不 free |\n'
    '| `ggml_free` | 若 `mem_buffer_owned` 则 free 一次，再放掉结构体自己 | O(1) |\n'
    '| `ggml_used_mem` | 返回最后一个对象的末端 | O(1) |\n\n'
    '`ggml_used_mem` 值得单独看一眼：**用量不需要记账，bump 指针的身高就是账本。**',
    src=GC, parts=[(1653, 1677)], lang='c')

L.section(
    '四、★ bump 分配：ggml_new_object',
    '这是"建图 = 指针加法"的字面实现。整个函数只有六步，没有任何查找：\n\n'
    '```text\n'
    'cur_end  = 上一个对象的 offs + size        （空链表时为 0）\n'
    'size_needed = GGML_PAD(size, GGML_MEM_ALIGN)\n'
    'obj_new  = mem_buffer + cur_end            <- 新对象的位置\n'
    'if (cur_end + size_needed + GGML_OBJECT_SIZE > mem_size) 失败\n'
    'obj_new->offs = cur_end + GGML_OBJECT_SIZE  （跳过对象头）\n'
    'obj_cur->next = obj_new；objects_end = obj_new\n'
    '```\n\n'
    '两处细节值得记住：\n\n'
    '1. **越界只有失败，没有扩容。** 超出 `mem_size` 就告警返回 `NULL`（debug 版直接 `GGML_ABORT`）。\n'
    '2. **`offs` 跳过对象头。** 负载从 `mem_buffer + obj_new->offs` 开始，'
    '而对象头本身在 `obj_new` 处 —— 这正是 `ggml_get_next_tensor` 里'
    '`(char *)tensor - GGML_OBJECT_SIZE` 能反推出头部的原因。',
    src=GC, parts=[(1708, 1759)], lang='c')

L.section(
    '五、★ arena 的两个后果：四个 context 与一份预算',
    '`ggml-opt` 是 context 最重的客户。它的设计几乎全部由 arena 的性质决定：\n\n'
    '**后果一：按生命周期分组。** `ggml_opt_context` 里有四个 `ggml_context` 字段。'
    '因为能释放的最小粒度就是"整个 context"，想让一批张量一起死，'
    '唯一办法是让它们共用一个 context。`ggml_opt_free` 里于是就是三次 `ggml_free`。\n\n'
    '**后果二：内存必须提前预算。** `mem_size` 是 `ggml_init` 的参数，'
    '所以调用者必须先算出"会有几个张量"，再乘每个张量的固定开销 `ggml_tensor_overhead()`。'
    '这个乘法能成立，靠的是 L1-01 的结论：`ggml_tensor` 是定长值类型。',
    src=OPT, parts=[(30, 37), (584, 594), (346, 364)], lang='cpp')

L.section(
    '六、数据集：一个 arena + 一个后端 buffer',
    '`ggml_opt_dataset_init` 把"元数据"和"数据"分开：\n\n'
    '- `mem_size = 2 * ggml_tensor_overhead()`：这个 arena **只放两个张量的对象头**'
    '（data 与 labels），所以 `no_alloc = true`；\n'
    '- 真正的数据由 `ggml_backend_alloc_ctx_tensors_from_buft` 按这两个张量的形状'
    '在后端 buffer 上分配。\n\n'
    '于是 `ggml_opt_dataset_free` 也就是两件事：先还 buffer，再 `ggml_free(ctx)`。',
    src=OPT, parts=[(102, 135)], lang='cpp')

L.section(
    '七、复用：把整个 context 扔掉重建',
    'arena 不能单独回收，但可以**整车换掉**。`ggml_opt_alloc` 在静态图路径下就是这么做的：'
    '先把旧的 `ctx_copy` 整个 `ggml_free`，再用新的预算 `ggml_init` 一个，'
    '然后在新 arena 里 `dup_graph` 出一份图副本。\n\n'
    '这是 arena 世界里的"回收"：**没有 free 单个对象，只有 free 整个世界再开一个新的。**',
    src=OPT, parts=[(762, 771)], lang='cpp')

L.section(
    '八、线程：ggml-threading 的全部内容',
    '**实测纠正**：计划文档把这一课的要点写成"线程池"，但逐行读完这两个文件后可以确认 —— '
    '`ggml-threading.cpp` 一共 12 行，内容是一个全局 `std::mutex` 加两个薄封装；'
    '`ggml-threading.h` 一共 14 行，只声明这两个函数。'
    '**没有线程池、没有任务队列、没有 worker、没有原子计数器。**\n\n'
    '本课引用到的调用点是 `ggml/src/ggml.c` 的 `ggml_init`，保护"首次调用初始化时间系统"那一小段。'
    '（CPU 侧真正的并行度在 L5 层的后端课里讲。）',
    src=THR, parts=[(1, 12)], lang='cpp')

L.section(
    '九、ggml-threading.h：整个头文件只有两个函数',
    '把整个头文件贴在这里，因为"它只有两个函数"这句话只有看全文才能确认。'
    '`extern "C"` 包裹说明它是 C ABI —— 供 C 代码和别的语言绑定调用。',
    src=THRH, parts=[(1, 14)], lang='c')

L.section(
    '十、ggml.cpp：进程级兜底，与 context 无关',
    '本课覆盖的五个文件里，`ggml.cpp` 是最容易被误判的一个。它 26 行，'
    '做的是把一个 `std::terminate` handler 装进进程：异常没人接住时先打 backtrace，再 abort。'
    '`GGML_NO_BACKTRACE` 环境变量存在时，这个 handler 干脆不装。\n\n'
    '这也是"源码/计划不一定是真的"的另一个例子：**文件名暗示的内容与实际内容可以完全无关。**',
    src=GCPP, parts=[(6, 26)], lang='cpp')

L.section(
    '十一、训练接口：一个 epoch 里的三段调用',
    '`ggml_opt_epoch` 是训练循环的骨架：前半段数据集做训练（`backward = true`），'
    '后半段做验证（`backward = false`）。每个 batch 都是同一套三步：'
    '`ggml_opt_alloc` 准备（必要时分配/重建图）-> `ggml_opt_dataset_get_batch` 拷数据 -> '
    '`ggml_opt_eval` 算。\n\n'
    '最高层再包一层 `ggml_opt_fit`：它建 `ggml_opt_context`、按 epoch 循环、'
    '最后 `ggml_opt_free`。**文档里推荐的用法（`ggml-opt.h` 的 Intended Usage）就是"两个 context + `no_alloc`"。**',
    src=OPT, parts=[(881, 924)], lang='cpp')

L.footnote_add('本课声明的源文件共 6 个：计划指派的 5 个（`ggml/src/ggml.cpp`、'
               '`ggml/src/ggml-threading.cpp`、`ggml/src/ggml-threading.h`、'
               '`ggml/src/ggml-opt.cpp`、`ggml/include/ggml-opt.h`），'
               '外加 `ggml/src/ggml.c`。额外引用后者的原因：`ggml_context` / `ggml_object` / '
               '`ggml_init` / `ggml_free` / `ggml_new_object` 的实现都在 `ggml.c` 里，'
               '不在前五个文件中；没有它，验收点"为什么用 arena 而非逐个 malloc"无法由引用代码直接支撑。'
               '`ggml.c` 本就由 L1-02 / L1-03 / L4-04 共享，重叠覆盖不影响覆盖度门禁（并集判定）。')
L.footnote_add('**实测纠正记录**：计划文档原先把本课要点写作"`ggml_context` 的对象分配、**线程池**、'
               '`ggml-opt` 的训练接口"。实测结果：`ggml-threading.cpp` 只有 12 行，'
               '内容是全局 `std::mutex` + `ggml_critical_section_start/end` 两个封装，**没有线程池**；'
               '`ggml.cpp` 26 行，内容是进程级 `std::terminate` handler，**与 context 无关**。'
               '本课按实测内容撰写，并把这次纠正本身作为教学点（第 1、8、9、10 幕与第八、十节）。')
L.footnote_add('本课不含任何命令行参数引用；提到 `GGML_NO_BACKTRACE` 是环境变量，不是 CLI 选项。')

L.prereqs('`L1-01`（ggml 张量：算子的数据面）')

L.goal(
    '画出 `ggml_context` 的 arena 布局：`mem_buffer` / `mem_size` / 对象头 / `offs` / `objects_end`（对应验收点）；',
    '说出"用 arena 而不是逐个 `malloc`"换来了什么、又换走了什么，'
    '并解释为什么这直接导致 `ggml-opt` 要同时持有四个 context；',
    '解释为什么 `mem_size` 可以按"张量个数 x `ggml_tensor_overhead()`"来预算（呼应 L1-01 的定长张量）；',
    '说出 `ggml-threading` 实际提供了什么、没有提供什么，以及它在 `ggml_init` 里的调用点；',
    '说出 `ggml_opt` 的两个训练入口：`ggml_opt_epoch`（一次训练/验证循环）与 `ggml_opt_fit`（最高层封装）。')

L.conclusion(
    '★ 为什么是 arena，而不是逐个 malloc',
    '`ggml_context` 把"对象分配"从"每次调用 malloc"变成"在一块预分配内存上做指针加法"：\n\n'
    '| 维度 | 逐个 malloc | arena（ggml_context 的做法） |\n|---|---|---|\n'
    '| 分配 | 每次都要走分配器、找空闲块 | `mem_buffer + cur_end`，纯算术（`ggml_new_object`） |\n'
    '| 释放 | 每个对象各自 free | 一次 `ggml_free`，或 `ggml_reset` 把指针抹掉 |\n'
    '| 元数据 | 每个对象都可能带堆头 | 只有对象头 `offs` / `size` / `next` / `type` |\n'
    '| 用量查询 | 需要记账 | `objects_end->offs + objects_end->size` |\n'
    '| 按对象释放 | 可以 | **不可以** —— 只能整块来、整块走 |\n'
    '| 内存大小 | 按需增长 | 必须提前预算 `mem_size`，超了就失败 |\n\n'
    '最后两行就是代价。**`ggml-opt` 的四个 context（`ctx_static` / `ctx_cpu` / `ctx_compute` / `ctx_copy`）'
    '正是"不能按对象释放"的直接产物**：它用"分组开 context"来换"分组释放"。')

L.conclusion(
    '三条 O(1)',
    '```text\n'
    'ggml_new_object : mem_buffer + cur_end          <- 分配 O(1)\n'
    'ggml_free       : ggml_aligned_free(mem_buffer)  <- 释放 O(1)\n'
    'ggml_reset      : begin = end = NULL             <- 复用 O(1)\n'
    '```\n\n'
    '**建图快到这个程度，是"把整张图放进一个 context"这个设计换来的。**'
    '但也正因为如此，L4-01 的 `ggml-alloc` 才必须存在：'
    '它要在 arena 之上再算一层"哪些张量可以复用同一段内存"，'
    '因为 arena 自己不会回收任何一个字节。')

L.conclusion(
    '跨课呼应',
    '| 本课的结论 | 依赖/展开于 |\n|---|---|\n'
    '| 张量能整块放进 arena | **L1-01**：`ggml_tensor` 是定长值类型，可 memcpy、可顺序摆放 |\n'
    '| 图节点也住在 context 里 | **L1-03**：计算图与拓扑遍历（`ggml/src/ggml.c:7479` 的 `ggml_new_graph_custom` 同样走 `ggml_new_object`） |\n'
    '| arena 之上还要一层复用 | **L4-01**：`ggml-alloc` 在 arena 之上做张量内存复用 |\n'
    '| `no_alloc = true` 时数据在别处 | **L4-02 / L4-03**：后端 buffer 与调度器 |')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
