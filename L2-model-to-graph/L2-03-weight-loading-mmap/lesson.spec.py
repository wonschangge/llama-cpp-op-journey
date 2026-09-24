#!/usr/bin/env python3
"""L2-03 · 权重加载与内存映射 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-03-weight-loading-mmap/lesson.spec.py

引用一律按行号从上游抽取（tools/lessonkit.py 的 q()），spec 里不出现手打代码。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

H   = 'src/llama-model-loader.h'
C   = 'src/llama-model-loader.cpp'
MH  = 'src/llama-mmap.h'
MC  = 'src/llama-mmap.cpp'
IOH = 'src/llama-io.h'
IOC = 'src/llama-io.cpp'

L = Lesson(
    id='L2-03',
    layer='L2 · 从模型到图',
    title='权重加载与内存映射',
    codecap='llama-model-loader.{h,cpp} / llama-mmap.{h,cpp} / llama-io.{h,cpp}（逐字引用）',
    nav={'prev': {'href': '../L2-02-arch-table-hparams/index.html', 'label': 'L2-02 架构表与超参'},
         'next': {'href': '../L2-04-kv-cache-and-memory/index.html', 'label': 'L2-04 KV cache 与记忆家族'}},
)

L.note('**一句话**：权重加载解决的是"GGUF 文件里的一段字节，怎么变成某个后端 buffer 里的一个张量"。'
       '这条路上有两个独立的问题：**怎么搬**（mmap 直接把文件页映射成张量数据，还是 read 进内存）'
       '和**搬到哪**（哪个 buffer type 持有它）。两者都在建图阶段就定了，'
       '`load_all_data()` 只是执行。')
L.note('本课覆盖 6 个文件：权重侧是 `src/llama-model-loader.{h,cpp}`，文件与映射侧是 `src/llama-mmap.{h,cpp}`；'
       '另外用 `src/llama-io.{h,cpp}` 对照 llama.cpp 里**另一套** I/O 抽象（状态序列化，不是权重）。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2 · 从模型到图',
    title='一个权重 = 一段字节区间 + 一个张量',
    sub='loader 在建图之前就记下每个权重的（文件号，文件内偏移）；落到哪由后面决定。',
    caption='回顾 L1-01：`data` / `buffer` 是 ggml_tensor 的"位置"字段 —— 本课讲它们怎么被填上。',
    src=H, parts=[(34, 50)], duration=14000,
    marks=[1, 2, 5, 13],
    notes={2: '文件号：分片模型（split）由多个 .gguf 组成，idx 指是哪一片',
           12: 'offs 是文件内的【绝对字节偏移】= 数据段基址 + 该张量在数据段内的偏移'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">GGUF 分片文件</span><span class="arrow">-></span>
    <span class="chip a">llama_tensor_weight</span><span class="arrow">-></span>
    <span class="chip b">ggml_tensor</span><span class="arrow">-></span>
    <span class="chip c">后端 buffer</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: '从哪来', b: '文件号 idx + 文件内偏移 offs。<br>构造时就查 GGUF 并做边界检查。', m: 'uint16_t idx;  size_t offs;' },
  { c: 'b', t: '落到谁', b: 'loader 只认张量名字：<br>名字 -> 权重的表就是 weights_map。', m: 'ggml_tensor * tensor;' },
  { c: 'c', t: '落到哪', b: '由建图时选出的 buffer type<br>决定，GGUF 里没有任何设备信息。', m: 'tensor->buffer / tensor->data' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '这一课回答三个问题：<span class="v">从哪来</span>、<span class="v">怎么搬</span>、<span class="v">落到哪</span>。',
  '<span class="k">从哪来</span>：权重 = (文件号 idx, 文件内偏移 offs)，<br>再加一个等着被填充的张量指针。',
  '构造时还有一道 <span class="v">file->size()</span> 边界检查：<br>数据必须真的落在文件里，否则模型被判为损坏。',
  '<span class="k">落到哪</span>没法从文件回答 —— 所以才有第 7 幕的"谁决定 buffer type"。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-03 · 开关',
    title='<span class="hl-c">load_mode</span> 只折成两个布尔',
    sub='要不要 mmap、要不要 direct I/O，在构造 loader 时就定死，之后整条加载路径都按它分支。',
    caption='命令行上的对应开关是 --load-mode（实测 llama-cli --help 里有 -lm, --load-mode MODE）。',
    src=C, parts=[(559, 560)], duration=20000,
    marks=[0, 1],
    notes={1: 'direct I/O 走的是 O_DIRECT 整块读，与 mmap 互斥；两者都不开就是普通 read'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['load_mode', 'use_mmap', 'use_direct_io', '加载手段'],
  [['LLAMA_LOAD_MODE_AUTO  (-1)', 'true', 'false', '默认；平台不支持时自动关掉 mmap'],
   ['LLAMA_LOAD_MODE_NONE  (0)', 'false', 'false', '普通 read'],
   ['LLAMA_LOAD_MODE_MMAP  (1)', 'true', 'false', '映射整个文件'],
   ['LLAMA_LOAD_MODE_MLOCK  (2)', 'false', 'false', '普通 read + 给页加锁'],
   ['LLAMA_LOAD_MODE_MMAP_MLOCK (3)', 'true', 'false', '映射 + 给页加锁'],
   ['LLAMA_LOAD_MODE_DIRECT_IO (4)', 'false', 'true', '绕开页缓存的整块读']],
  { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '六个 API 取值，落到代码里只是这一行折出来的两个布尔：',
  '<span class="v">AUTO / MMAP / MMAP_MLOCK</span> -> <span class="k">use_mmap = true</span>；只有 AUTO 带"自动判定"的语义。',
  '<span class="v">NONE / MLOCK</span> 都是普通 read，区别只在调用方要不要把页锁住（mlock 由调用方传 mlock_mmaps 进来）。',
  '<span class="v">DIRECT_IO</span> 打开 O_DIRECT（llama-mmap.cpp:198-216），读偏移和长度都要按块对齐 —— 第 6 幕会看到它的影响。',
  '平台不支持时（<span class="v">llama_mmap::SUPPORTED == false</span>）构造末尾把 <span class="v">use_mmap</span> 强行置回 false（llama-model-loader.cpp:829-832）。',
  '所以"用不用 mmap"是【构造时】定下的，运行期不再变化。'
];
const rowtext = [1, 2, 1, 2, 1, 3];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[rowtext[i]];
}));
tl.at(16600, () => {
  rows.forEach((x, k) => { x.className = (k === 0 || k === 2 || k === 4) ? 'on' : ''; });
  msg.innerHTML = texts[4];
});
tl.at(18600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-03 · 映射',
    title='llama_mmap 的门面：三个操作',
    sub='建立映射、拿基地址、按页归还。三个平台分支（POSIX / Win32 / 不支持）藏在 pimpl 后面。',
    caption='loader 持有 `llama_mmaps mappings`（llama-model-loader.h:124），构造它的地方是 init_mappings（同文件 1410-1447 行）。',
    src=MH, parts=[(44, 63)], duration=15000,
    marks=[0, 5, 9, 10, 13, 15],
    notes={10: 'addr() 是零拷贝的入口：权重数据地址 = addr() + 文件内偏移',
           14: 'SUPPORTED 是编译期常量；不支持时构造会直接抛 "mmap not supported"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'a', t: '① 建立映射', b: '构造即映射：整个文件一段，<br>prefetch / numa / lazy_ranges 都是提示。', m: 'llama_mmap(file, prefetch, numa, lazy_ranges)' },
  { c: 'b', t: '② 拿基地址', b: '映射的虚拟地址 + size。<br>权重数据地址 = addr() + offs。', m: 'void * addr() const;' },
  { c: 'c', t: '③ 归还页', b: '按页粒度还回去，加载结束后<br>把元数据、未用张量的页释放掉。', m: 'void unmap_fragment(size_t first, size_t last);' },
  { c: 'd', t: '④ 平台门', b: 'POSIX / Win32 / 不支持三分支，<br>对外只暴露 SUPPORTED 一个常量。', m: 'static const bool SUPPORTED;' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '先把接口读成三个动作 + 一个平台门。',
  '<span class="k">构造即映射</span>：没有单独的 open() —— 对象活着的期间映射就有效，析构函数负责 munmap。',
  '<span class="v">addr()</span> 是后面 load_all_data 里 <span class="v">mapping->addr() + weight->offs</span> 的来源。',
  '<span class="v">unmap_fragment</span> 的存在说明映射不是"要么全在要么全不在"：可以只把某段页还回去。',
  '注释写明 ranges 是 <span class="v">[first, last)</span> 区间 —— lazy 读要把"哪些段留随机访问"告诉映射层。'
];
defs.forEach((_, i) => tl.at(700 + i * 3100, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13400, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-03 · 核心',
    title='★ <span class="hl-a">mmap</span>：省掉的是拷贝，不是必然的 I/O',
    sub='映射只改页表；字节什么时候从磁盘进来，由 prefetch / lazy / numa 三个提示决定。',
    caption='回顾 L1-06：映射要求张量数据在文件里按对齐偏移连续排布 —— GGUF 的数据段正好如此（对比 llama-model-loader.cpp:688-693 的对齐检查）。',
    src=MC, parts=[(469, 485)], duration=19000,
    marks=[4, 12, 14],
    notes={0: '映射粒度是整个文件；prefetch / numa / lazy_ranges 只影响"页什么时候进来"',
           13: 'PROT_READ：只读映射。内核知道没人会写，这些页可以与文件页缓存共享'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin:0 0 2px">同样读一遍权重，两条路的差别</div>';
const t = U.table(
  ['', 'mmap', 'read'],
  [['加载时做什么', '建映射（改页表）', '把字节读进用户缓冲'],
   ['谁付磁盘 I/O', '第一次访问该页时', '加载时一次付清'],
   ['页缓存', '直接用文件的页', '页 -> 用户缓冲，多一次拷贝'],
   ['常驻内存', '被访问过的页', '全部权重'],
   ['可调项', 'prefetch / numa / lazy', '基本没有']],
  { monoCols: [0] });
left.appendChild(t.el);

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">MAP_POPULATE（480 行）</div>' +
  '<div class="cb">prefetch 非 0 且没有 lazy 区间时才加：一次把页表建好。' +
  '只要有 lazy 区间就必须关掉 —— 否则会把本该懒读的页也拉进来。</div></div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">madvise（500-507 行）</div>' +
  '<div class="cb">prefetch 区间提示 <span class="cm" style="margin:0">POSIX_MADV_WILLNEED</span>（顺序读），' +
  'lazy 区间提示 <span class="cm" style="margin:0">POSIX_MADV_RANDOM</span>（别看太远）。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)">' +
  '<div class="ct" style="color:var(--d)">numa（473、508 行）</div>' +
  '<div class="cb">NUMA 机器上直接否决预取：<span class="cm" style="margin:0">prefetch = 0</span>，' +
  '并对全文件用 MADV_RANDOM —— 避免把远端内存拉满。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '核心只有一行：<span class="v">mmap(NULL, file->size(), PROT_READ, MAP_SHARED, fd, 0)</span>。',
  '<span class="k">只读映射</span>：内核知道这些页不会被写，可以直接与文件页缓存共享，不需要私有副本。',
  '映射建立是 O(1)，但<span class="k">字节什么时候进来是可调的</span>：MAP_POPULATE 立刻预取，否则留到访问时。',
  '源码注释把边界说清楚了：<span class="v">MAP_POPULATE would fault in the lazy ranges too</span> —— 预取与懒读互斥。',
  '因此准确的说法是：mmap 省掉【用户态缓冲 + 一次拷贝】，并把 I/O 变成按页触发；<br>它并不自动等于"加载时零 I/O"。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [14]); });
tl.at(4000, () => { msg.innerHTML = texts[1]; U.markLines(document, [14]); });
tl.at(7600, () => { msg.innerHTML = texts[2]; U.markLines(document, [12]); });
tl.at(11200, () => { msg.innerHTML = texts[3]; U.markLines(document, [12]); });
tl.at(14800, () => { msg.innerHTML = texts[4]; U.markLines(document, [4, 12, 14]); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-03 · 核心',
    title='★ <span class="hl-a">指过去</span>还是<span class="hl-b">拷过去</span>',
    sub='load_all_data 的第一句判据：数据是"已经有地址"，还是"还得从文件搬"。',
    caption='lazy 读（TENSOR_READ_LAZY）也走这一支：即使整体不用 mmap，被标记的大张量仍会拿到映射指针。',
    src=C, parts=[(1638, 1671)], duration=18000,
    marks=[2, 11, 20, 21, 33, 34],
    notes={2: '一行判据：走映射（use_mmap 或这个张量被标记为懒读），还是走 read',
           20: '把张量直接指到映射里的那个地址 —— 零拷贝落位，页错误推迟到内核读它的时候'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const dia = wrap.querySelector('#dia');

const box = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
box.innerHTML = '<div class="ct" style="color:var(--a)">分支 A · 在映射里</div>' +
  '<div class="cb">data = mapping->addr() + weight->offs<br>' +
  '<b>cur->data == nullptr</b> 时直接把它挂上去：</div>' +
  '<div class="cm" style="margin:4px 0 0">ggml_backend_tensor_alloc(buf_mmap, cur, data);</div>' +
  '<div class="cb" style="margin-top:4px">代价：<b>页错误</b>推迟到后端内核第一次读它。</div>';

const box2 = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
box2.innerHTML = '<div class="ct" style="color:var(--b)">分支 B · 已经有 buffer</div>' +
  '<div class="cb">张量已经分配在自己的 buffer 里，<br>映射里的数据要拷过去：</div>' +
  '<div class="cm" style="margin:4px 0 0">ggml_backend_tensor_set(cur, data, 0, n_size);</div>' +
  '<div class="cb" style="margin-top:4px">代价：一次 memcpy，但页用完就可以放。</div>';

dia.appendChild(box); dia.appendChild(U.arrow('->')); dia.appendChild(box2);

const msg = wrap.querySelector('#msg');
const texts = [
  '映射里的数据要变成张量的 data，有两条出路，判据是"张量是不是还没地方放"。',
  '<span class="k">分支 A</span>：<span class="v">buf_mmap && cur->data == nullptr</span> —— 这个文件已经有对应的后端 buffer，直接把 data 指过去。',
  '指过去之后，源码顺手记下这个文件里【真正用过】的页区间（<span class="v">mmaps_used</span>，1666-1668 行），收尾时按它归还。',
  '<span class="k">分支 B</span>：张量已经有自己的 buffer（例如显存里），那就 <span class="v">ggml_backend_tensor_set</span> 拷一次。',
  '两条路都不做"选后端"这件事 —— buffer 是上一幕之前就按 buffer type 分好的（见第 7 幕）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [2]); });
tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [20, 21]); });
tl.at(7200, () => { msg.innerHTML = texts[2]; U.markLines(document, [11]); });
tl.at(10800, () => { msg.innerHTML = texts[3]; U.markLines(document, [33, 34]); });
tl.at(14400, () => { msg.innerHTML = texts[4]; U.markLines(document, [2, 21, 34]); });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-03 · read 侧',
    title='不走映射时：三条搬运路径',
    sub='host buffer 直接读；设备 buffer 有异步能力就走 pinned host buffer 中转；没有就现开临时缓冲。',
    caption='张量会按"需要中转的排前面、大的排前面"稳定排序（同一文件 1614-1623 行），让暂存峰值最小。',
    src=C, parts=[(1673, 1690), (1737, 1746)], duration=18000,
    marks=[2, 4, 13, 23, 26],
    notes={11: '设备 buffer：后端支持异步时，先把文件块读进 pinned host buffer，再异步上传 —— 这就是 host 中转',
           14: 'direct I/O 模式下 alignment > 1，偏移与长度都要按块对齐（read_alignment() 来自 st_blksize）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#cards');
const defs = [
  { c: 'b', t: '① host buffer：直读', b: '<b>ggml_backend_buffer_is_host</b> 为真时，' +
      'seek 到偏移后直接 read 进张量的 data —— 没有中转缓冲。', m: 'file->read_raw(cur->data, n_size);' },
  { c: 'a', t: '② 设备 buffer + 异步', b: '先建 4 个 pinned host buffer（1MB，' +
      'direct I/O 时 64MB + 对齐），分块读进来再异步上传。', m: 'ggml_backend_tensor_set_async(...)' },
  { c: 'c', t: '③ 设备 buffer + 同步', b: '没有异步能力就现开一个临时缓冲：读一份、set 过去、释放。' +
      '注释说明它按张量作用域分配。', m: 'std::vector<no_init<uint8_t>> read_buf(n_size);' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '非 mmap 路径要先问一句：这个张量住的 buffer 是主存还是设备？',
  '<span class="k">主存（host buffer）</span>：文件字节直接落到张量上，读一次就完事。',
  '<span class="k">设备</span>且后端有 async / host_buffer / events 三项能力（1526-1598 行）：走 pinned host buffer 中转。',
  '中转是为了让 <span class="v">ggml_backend_tensor_set_async</span> 与下一次磁盘读重叠 —— 代价是一份额外的 pinned 内存。',
  '都没有时退化成最朴素的一版：临时 buffer 读一份、<span class="v">ggml_backend_tensor_set</span> 拷到设备。',
  '三条路的共同点：<span class="k">张量早就落好位了</span>，这里只负责把字节送过去。'
];
defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(12800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-03 · 核心',
    title='★ 谁决定权重落到哪个后端',
    sub='答案是一串判据 + 一个函数：按 buft_list 的顺序问"这个 buffer type 能不能跑用到这个权重的算子"。',
    caption='对照 L3-01：`ggml_backend_buffer_type_t` 与 `ggml_backend_dev_supports_op` 是后端契约的一部分；这里的判据完全建立在它之上。',
    src=C, parts=[(1066, 1078)], duration=20000,
    marks=[1, 4, 7, 9],
    notes={1: 'buft_list 是"这一层可以用的 buffer type 列表"，顺序就是优先级（list 由调用方按 offload 参数排好）',
           6: '判据不在这个函数里：weight_buft_supported 给权重造一个假算子，挂 0 字节哑 buffer，再问后端 supports_op'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="flow" style="gap:4px"></div>
    <div class="col grow" id="side" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const steps = [
  ['1', 'a', 'load_mode 折出 use_mmap / use_direct_io（559-560 行）'],
  ['2', 'c', '张量属于哪一层：input / output / repeating -> 取对应的 buft_list（1215-1230 行）'],
  ['3', 'd', 'tensor_buft_overrides 命中？命中就用它；命中 CPU 时再走 select_weight_buft（1235-1260 行）'],
  ['4', 'b', '否则 select_weight_buft：按列表顺序试（1067-1075 行）'],
  ['5', 'b', '判据：假算子 + 0 字节哑 buffer -> ggml_backend_dev_supports_op（1056-1059 行）'],
  ['6', 'c', '修正：use_mmap 且选中的是设备的 host buffer -> 换成 CPU 的 buffer type（1269-1277 行）'],
  ['7', 'd', '产物：按 buft 取（或新建）一个 ggml context，张量就建在那里（1122-1152、1380 行）'],
  ['8', 'a', '执行：load_all_data 指过去 / 拷过去 / 中转（1638-1747 行）']
];
const host = wrap.querySelector('#flow');
host.innerHTML = '<div class="cm" style="margin:0 0 2px">决定链（从上到下）</div>';
const els = steps.map(s => {
  const e = U.el('div', { class: 'formula', style: 'padding:3px 7px;display:flex;gap:6px;align-items:baseline' });
  e.innerHTML = '<span class="chip ' + s[1] + '" style="flex:0 0 auto">' + s[0] + '</span><span>' + s[2] + '</span>';
  host.appendChild(e);
  return e;
});
els.forEach(e => { e.style.opacity = '.32'; });

const side = wrap.querySelector('#side');
side.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">★ 决定的时刻</div>' +
  '<div class="cb">建图阶段：每拿到一个权重就调用一次 <span class="cm" style="margin:0">create_tensor()</span>。' +
  '此时形状已经知道（来自 GGUF 元数据），<b>数据一个字节都还没读</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">为什么不用数据来判断</div>' +
  '<div class="cb">判据是【算子的形状 + 后端能力】，不是数据内容：' +
  'weight_buft_supported 按 op 造一个假算子（MUL_MAT / GET_ROWS / ROPE ...），' +
  '再挂一个 0 字节哑 buffer 让后端自己回答。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">一个都选不上会怎样</div>' +
  '<div class="cb">列表全试完仍无解就抛 "failed to find a compatible buffer type"（1264-1266 行）；' +
  '退到非首选时会记账，收尾时打一条 debug 日志（1280-1287、1403-1407 行）。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '把决定链摊开：这是本课的验收点所在。',
  '★ 答案：决定发生在 <span class="k">create_tensor()</span> 被调用的那一刻 —— 建图阶段，数据还没读。',
  '第 5 步是全部判据的核心：不只是"这个 buffer type 存不存在"，而是"用它跑这个算子行不行"。',
  '第 6 步是 use_mmap 唯一影响选择结果的地方：映射的权重宁可留在 CPU，也不放进设备的 host buffer。',
  '第 8 步只执行落位；拿到手的 <span class="v">bufs</span> 已经按文件号分好，不再挑后端。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
els.forEach((e, i) => tl.at(2600 + i * 2100, () => {
  els.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.32'; });
  msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 4)];
}));
tl.at(19400, () => {
  els.forEach(x => { x.style.opacity = '1'; });
  msg.innerHTML = 'L4-02 的调度器再根据"权重在哪个 buffer 里"决定算子放哪 —— 前提正是这里的落位。';
});
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L2-03 · 对照',
    title='同一个工程里的另一套 I/O',
    sub='llama-io.h 的虚接口不服务权重，它服务"状态"：KV 与序列的存取。',
    caption='实现类 llama_io_write_host / _file / _device / _dummy 在 src/llama-context.cpp（L2-07 覆盖），本课不引用其代码。',
    src=IOH, parts=[(9, 35)], duration=17000,
    marks=[5, 7, 10, 12, 15, 20, 22],
    notes={5: '读侧与写侧对称：只有 5 个纯虚函数 —— 裸字节 + 张量分片 + 计数',
           19: '注意 read_tensor 带 (offset, size)：状态是按张量区间搬的，不是整块'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['', '权重 I/O（本课前 7 幕）', '状态 I/O（llama-io）'],
  [['方向', '只读', '既读又写'],
   ['粒度', '整个张量（或按行懒读）', '按 (offset, size) 分片'],
   ['载体', 'llama_file + llama_mmap', 'llama_io_read_i / llama_io_write_i'],
   ['实现', 'llama-mmap.cpp', 'llama-context.cpp 的 host / file / device 三个实现'],
   ['用途', '加载 GGUF 权重', '保存与恢复 KV cache、序列状态（见 L2-04）']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '为什么权重不走这套虚接口？因为它是【只读、一次、巨大】的 —— 虚函数帮不上忙。',
  '<span class="k">方向</span>：权重只读；状态要能存能取，才需要 read / write 两侧对称的接口。',
  '<span class="k">粒度</span>：<span class="v">read_tensor/write_tensor(tensor, offset, size)</span> —— 状态按张量区间搬，可以只存某几个 cell。',
  '<span class="k">载体与实现</span>：权重用 llama_file + llama_mmap；状态用虚接口。llama-io.cpp 全文只实现带 <span class="v">uint32_t</span> 长度前缀的字符串 —— 自描述字节流。',
  '<span class="k">用途</span>：权重加载 GGUF；状态保存与恢复 KV cache、序列状态（见 L2-04）。',
  '分界很清楚：<span class="k">文件里躺着的</span>走 mmap / read，<span class="k">运行期反复存取的</span>走 llama-io。'
];
const rowtext = [1, 2, 3, 3, 4];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2900 + i * 2700, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[rowtext[i]];
}));
tl.at(16400, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L2-03 · 收束',
    title='把这一课压成一张表',
    sub='两个输入、两个决策、两个执行动作。',
    caption='下一课 L2-04：权重落位之后，运行时那块内存（KV cache 与记忆家族）怎么组织。',
    src=H, parts=[(255, 260)], duration=17000,
    marks=[0, 2],
    notes={2: 'bufs 的键是【文件号】：哪个文件的数据放在哪个后端 buffer 里；mmap 时它还可以是"按映射区建的 buffer"'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', '答案', '源码锚点'],
  [['权重从哪来', 'GGUF 数据段里的字节区间（文件号 + 偏移）', 'llama-model-loader.h:34-50'],
   ['mmap 还是 read', 'load_mode 折成 use_mmap / use_direct_io 两个布尔', 'llama-model-loader.cpp:559-560'],
   ['mmap 怎么变地址', '建只读映射；addr() + offs 就是张量数据地址', 'llama-mmap.cpp:482，llama-model-loader.cpp:1648'],
   ['谁决定落到哪个后端', 'create_tensor -> buft_list -> select_weight_buft -> supports_op', 'llama-model-loader.cpp:1066-1078'],
   ['什么时候决定', '建图阶段；load_all_data 只执行落位', 'llama-model-loader.cpp:1493-1797'],
   ['要不要 host 中转', 'host buffer 直读；否则 pinned / 临时缓冲中转', 'llama-model-loader.cpp:1675-1746']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '一个权重张量在什么时刻决定自己落到哪个后端？把加载模式从 mmap 改成普通 read，这个决定会变吗？',
  '决定发生在<b>建图阶段</b>：模型构建代码每拿到一个权重就调用一次 ' +
  '<span class="mono">create_tensor()</span>，它按张量所属层取出 buft_list（1215-1230 行），' +
  '再用 <span class="mono">select_weight_buft()</span>（1067-1075 行）按顺序询问；' +
  '判据是 <span class="mono">weight_buft_supported()</span>（927-1064 行）：' +
  '给权重造一个假算子、挂一个 0 字节哑 buffer，再问后端 ' +
  '<span class="mono">ggml_backend_dev_supports_op()</span>（1058-1059 行）。此时数据一个字节都没读。<br>' +
  '改加载模式确实会改变这个决定：<span class="mono">use_mmap</span> 为真时，' +
  '若选中的是设备的 host buffer，会被强制换成 CPU 的 buffer type（1269-1277 行）。<br>' +
  '而"落位动作"是稍后的 <span class="mono">load_all_data()</span> 做的：' +
  '映射就直接把 <span class="mono">cur->data</span> 指到映射地址（1658 行），' +
  '否则拷贝或中转（1670 / 1742 行）。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '六行表：前两行是输入，中间两行是决策，最后两行是执行。',
  '<span class="k">输入</span>：文件里的字节区间、以及"要不要 mmap"。',
  '<span class="k">决策</span>：谁决定（create_tensor 里的 select_weight_buft）、什么时候决定（建图阶段）。',
  '<span class="k">执行</span>：load_all_data 只负责把字节送到已经定好的位置上。',
  '记住一句话：<span class="v">mmap 决定"怎么搬"，buft_list 决定"搬到哪"，两者在建图时就都定了</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(Math.floor(i / 2) + 1, 3)];
}));
tl.at(16200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、权重索引：一个权重就是一段字节区间',
    '`llama_model_loader` 在构造时就把 GGUF 里每个张量的数据位置记下来：文件号 `idx`、'
    '文件内绝对偏移 `offs`，再加上一个等着被填充的 `ggml_tensor *`。\n\n'
    '注意 `offs` 的来历：`gguf_get_data_offset()`（数据段基址）+ `gguf_get_tensor_offset()`'
    '（张量在数据段内的偏移）。这正是 mmap 能工作的前提 —— **偏移是文件内的绝对偏移**，'
    '所以映射基址 + 偏移就等于张量数据地址。紧接着的边界检查说明 loader 对"数据必须真的在文件里"'
    '是强校验的。',
    src=H, parts=[(34, 50)], lang='c')

L.section(
    '二、加载模式：两个布尔在构造时定下',
    '`llama_load_mode` 有六个取值，落到 loader 里只是两个布尔。`AUTO` 也归到 `use_mmap = true`，'
    '它的"自动"语义体现在别处：真正分配 buffer 时按设备能力决定能不能把映射交给后端。\n\n'
    '`mlock` 不由这里决定：loader 只在 `load_all_data()` 里按调用方传进来的 `llama_mlocks *` 办事。'
    '如果平台根本不支持 mmap，构造末尾会把 `use_mmap` 关掉并打一条警告 —— '
    '于是"选择 mmap"在运行期永远不会失败，只会静默退化成 read。',
    src=C, parts=[(559, 560), (829, 832)], lang='c')

L.section(
    '三、llama_mmap：三个操作 + 一个平台门',
    '对外接口只有四项：构造（建立映射）、`size()` / `addr()`（拿基地址）、`unmap_fragment()`'
    '（按页归还）、`SUPPORTED`（平台门）。三个平台分支（POSIX、Win32、不支持）都在 `impl` 里，'
    '调用方看不到。\n\n'
    '`ranges` 是 `[first, last)` 的字节区间列表，它的用途是**懒读**：告诉映射层哪些段要留着随机访问，'
    '其余段可以放心顺序预取。',
    src=MH, parts=[(44, 63)], lang='c')

L.section(
    '四、★ 映射的建立：只读、整文件、预取可调',
    '一次 `mmap()` 覆盖整个文件，`PROT_READ` + `MAP_SHARED`。因为声明了只读，内核可以让这些页'
    '直接指向文件的页缓存，不需要给写者准备私有副本 —— 这是 mmap 省掉"用户态缓冲 + 一次拷贝"的根源。\n\n'
    '但**字节什么时候进来是可调的**：`prefetch` 非 0 且没有懒读区间时加 `MAP_POPULATE`（源码注释：'
    '`MAP_POPULATE would fault in the lazy ranges too`）；`posix_madvise` 用 `WILLNEED` / `RANDOM`'
    '给不同区间不同的读法提示；NUMA 机器上直接否决预取。所以"mmap 让加载不花时间"是一个'
    '**带前提的说法**：省掉的是拷贝与用户态缓冲，页什么时候进来另说。',
    src=MC, parts=[(469, 516)], lang='c')

L.section(
    '五、按页归还：unmap_fragment',
    '`unmap_fragment()` 先把区间对齐到页边界，再 `munmap()` —— 所以"还一半页"是不会发生的：'
    '对齐后长度为 0 就直接返回。归还之后它同步维护自己那份"还有哪些段映射着"的账本，'
    '析构函数只对账本里剩下的段收尾。',
    src=MC, parts=[(530, 572)], lang='c')

L.section(
    '六、收尾：把没用的页还回去',
    '加载完成后，`load_all_data()` 用 `mmaps_used` 记下的区间做一次收尾：'
    '**数据段之前**（文件头 / 元数据）与**用到的区间之后**（对齐边角、未用张量）全部 `unmap_fragment`。'
    '映射不是"要么全留要么全放"，只有真正被权重用到的页留下来。',
    src=C, parts=[(1776, 1788)], lang='c')

L.section(
    '七、llama_file：read 侧的底座',
    '不走 mmap 时，字节靠 `llama_file` 搬。它在 Linux 上可以打开 `O_DIRECT` 并把 `st_blksize` 记成'
    '`alignment`；`read_raw()` 因此分两种：direct I/O 走 `read_aligned_chunk()`（对齐 + 临时对齐缓冲 + memcpy），'
    '否则直接 `read_raw_unsafe()`。\n\n'
    '这套 alignment 一路上传：`load_all_data()` 会取 `file->read_alignment()` 来算对齐偏移，'
    '并据此把中转缓冲从 1MB 放大到 64MB + 2 个对齐块。',
    src=MC, parts=[(198, 216), (344, 350), (388, 392)], lang='c')

L.section(
    '八、★ load_all_data：先排序，再逐个落位',
    '`load_all_data()` 是整条加载路径的执行者。开头先处理"没有文件"的虚拟模型，然后建立上传后端、'
    '确定中转缓冲大小；接着把上下文里的张量收集成列。\n\n'
    '不走 mmap 时先做一次稳定排序：**需要中转的排前面、大的排前面** —— 让最大的暂存缓冲'
    '在权重驻留最少的时候被分配，同时保证同一时刻只有一份暂存活着。\n\n'
    '然后是主循环：对每个张量取出它的权重记录，用一个判据 `from_mapping` 分成两支。'
    '注意这一层循环**不做任何后端选择**：张量早就建在按 buffer type 分的 ggml context 里了。',
    src=C, parts=[(1493, 1505), (1607, 1623), (1625, 1671)], lang='c')

L.section(
    '九、非 mmap：三条搬运路径',
    '不是映射的字节只有三种去处：已经住在 host buffer 里就直接读进去；住在设备上且后端支持'
    '异步上传，就先读进 pinned host buffer 再 `set_async`（读下一块与上传上一块重叠）；'
    '两者都不满足就现开一个临时缓冲读完再 `set` 过去。',
    src=C, parts=[(1672, 1747)], lang='c')

L.section(
    '十、★ buffer type 的判据：一个有形状、没有数据的假算子',
    '这是全课最需要看清的一段。`select_weight_buft()` 本身只是"按顺序试"，真正的判据是 '
    '`weight_buft_supported()`：它按权重的算子类型（`MUL_MAT` / `GET_ROWS` / `ROPE` / `SSM_CONV` ...）'
    '**造一个假算子**，给权重挂一个 **0 字节哑 buffer**，然后问后端 `ggml_backend_dev_supports_op()`。\n\n'
    '也就是说：判据是"这个后端用这个 buffer type 跑这个算子行不行"，而不是"数据放在哪"。'
    '正因如此，选择可以在**没有任何数据**的情况下、在建图阶段完成。',
    src=C, parts=[(927, 932), (1056, 1078), (1215, 1230), (1262, 1289)], lang='c')

L.section(
    '十一、llama-io：同一工程里的另一套 I/O',
    '`llama_io_write_i` / `llama_io_read_i` 是**状态序列化**的抽象：5 个纯虚函数（裸字节、'
    '张量区间、计数），读侧写侧对称。`llama-io.cpp` 全文只实现一件事 —— 带 `uint32_t` 长度前缀的字符串，'
    '说明这是一条自描述的字节流。\n\n'
    '它和权重加载没有交集：权重的特点是**只读、一次、巨大**，不需要虚接口；'
    '状态（KV cache、序列状态）的特点是**反复存取、可以只取一部分**，所以才需要抽象。'
    '具体实现类（`llama_io_write_host` / `_file` / `_device` / `_dummy`）在 `src/llama-context.cpp`，'
    '由 L2-07 覆盖。',
    src=IOH, parts=[(9, 35)], lang='c')

L.section(
    '十二、llama-io.cpp：全文只有两个函数',
    '这套接口里唯一的非虚实现就是 `write_string` / `read_string`：先写 `uint32_t` 长度，再写字节；'
    '读侧反过来。派生类只需要实现裸字节的 `write` / `read`，字符串与"张量的某一区间"由基类组合出来。\n\n'
    '这很能说明它的定位：**它假设自己是一条可以自描述的字节流** —— 所以能边写边报长度（`n_bytes()`）、'
    '能按张量区间分片，也就支持"只存某几个序列"这种用法。权重加载完全不需要这些性质。',
    src=IOC, parts=[(5, 20)], lang='c')

L.section(
    '十三、mmap 的延伸：按需读行（结构）',
    '映射打开了一扇门：既然数据已经有地址，就可以**不把整张量读进来**，用到哪几行读哪几行。'
    '`lazy_read` 把要懒读的张量名、以及"每个文件里哪些区间要留随机访问"记下来，'
    '交给映射层的 `ranges`（见第三幕的 `unmap_fragment` 与 `lazy_ranges`）。',
    src=H, parts=[(87, 118)], lang='c')

L.section(
    '十四、懒读的开关与代价',
    '`lazy_read::add()` 有三道门：模式为 `OFF` 直接拒绝；非 `ON` 模式下只对超过 4 GiB 的张量自动开启；'
    '`llama_mmap::SUPPORTED` 为假时明确拒绝并打印警告 —— 因为懒读目前**依赖 mmap**。',
    src=C, parts=[(1086, 1114)], lang='c')

L.footnote_add('本课引用 `src/llama-model-loader.{h,cpp}`、`src/llama-mmap.{h,cpp}`、'
               '`src/llama-io.{h,cpp}` 共 6 个源文件，全部计入覆盖率。')
L.footnote_add('文中提到的调用方（`src/llama-model.cpp`：`init_mappings()` / `load_all_data()` 的调用点、'
               '`use_mlock` 的判定、`mlock_mmaps` 的构造）属于 L2-05 的覆盖范围；'
               '`src/llama-context.cpp`（llama-io 的实现类）属于 L2-07。本课不引用这两个文件的代码，'
               '故不计入本课覆盖率。')
L.footnote_add('文中出现的命令行开关 `--load-mode` 由真实二进制的 `llama-cli --help` 实测确认存在。')

L.prereqs('`L2-02`（架构表与超参：张量名与"它属于哪一层"来自那里）')

L.goal(
    '说出权重张量在**什么时刻**决定自己落到哪个后端，以及这个决定的输入是什么（对应验收点）；',
    '解释 mmap 与 read 两条路径在加载时间、常驻内存、额外拷贝上的差别，以及 `MAP_POPULATE` / '
    '`madvise` / NUMA 三个可调项的作用；',
    '说明 `use_mmap` / `use_direct_io` 怎么从 `load_mode` 得到，平台不支持 mmap 时会怎样；',
    '区分 llama.cpp 的两套 I/O：权重加载（`llama_file` / `llama_mmap`）与状态序列化'
    '（`llama_io_read_i` / `llama_io_write_i`）。')

L.conclusion(
    '★ mmap 省的是拷贝，不是 I/O',
    '| | mmap | read |\n|---|---|---|\n'
    '| 加载时做什么 | 建映射（改页表） | 把字节读进用户缓冲 |\n'
    '| 谁触发磁盘 I/O | 第一次访问该页时 | 加载时一次付清 |\n'
    '| 额外拷贝 | 无（直接用文件页） | 页 -> 用户缓冲 |\n'
    '| 可调项 | `prefetch` / `numa` / `lazy_ranges` | 基本没有 |\n\n'
    '源码注释把这一点写得很直白：`MAP_POPULATE would fault in the lazy ranges too`。'
    '所以"mmap 不占 RAM 也不占加载时间"要加前提 —— **它把 I/O 变成按页触发，并省掉用户态缓冲**；'
    '页什么时候进来由预取策略决定。')

L.conclusion(
    '★ 落位在建图时决定，在加载时执行',
    '决定链：`load_mode` -> `use_mmap` / `use_direct_io`（559-560 行）-> 张量所属层决定用哪张 '
    '`buft_list`（1215-1230 行）-> `select_weight_buft()` 按顺序试（1067-1075 行）-> '
    '`weight_buft_supported()` 造假算子问后端 `supports_op`（1058-1059 行）-> 选定 buffer type，'
    '张量被建在"该 buft 的 ggml context"里。\n\n'
    '`load_all_data()` 只是执行：映射就直接把 `cur->data` 指过去（1658 行），否则拷贝或中转。'
    '**所以"权重在哪个后端"这个问题，在建图阶段就有答案了，而且答案不依赖数据。**')

L.conclusion(
    '一套工程，两套 I/O',
    '| | 权重 I/O | 状态 I/O（llama-io） |\n|---|---|---|\n'
    '| 方向 | 只读 | 读 + 写 |\n'
    '| 粒度 | 整个张量（或按行懒读） | `(offset, size)` 分片 |\n'
    '| 载体 | `llama_file` + `llama_mmap` | `llama_io_read_i` / `llama_io_write_i` |\n'
    '| 用途 | 加载 GGUF 权重 | 存取 KV cache / 序列状态 |\n\n'
    '分界很清楚：**文件里躺着的**走 mmap / read，**运行期反复存取的**走 llama-io。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
