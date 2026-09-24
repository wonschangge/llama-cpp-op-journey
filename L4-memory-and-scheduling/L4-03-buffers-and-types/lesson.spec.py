#!/usr/bin/env python3
"""L4-03 · buffer 与 buffer type —— 课件 spec。

运行：python3 L4-memory-and-scheduling/L4-03-buffers-and-types/lesson.spec.py

本课聚焦 buffer / buffer type 这一面：分配、初始化、属性查询、host 与 device 的分工，
以及 pinned host buffer 的真实使用场景。调度器的切分算法是 L4-02 的主题，本课不重复。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

C = 'ggml/src/ggml-backend.cpp'
H = 'ggml/include/ggml-backend.h'
M = 'src/llama-model-loader.cpp'       # 第 8 幕的使用方证据（逐字引用）

L = Lesson(
    id='L4-03',
    layer='L4 · 内存与调度',
    title='buffer 与 buffer type',
    kicker='L4 · 内存与调度',
    codecap='ggml-backend.cpp · ggml-backend.h · llama-model-loader.cpp（逐字引用）',
    nav={'prev': {'href': '../L4-02-scheduler-split/index.html', 'label': 'L4-02 ★ 调度器'},
         'next': {'href': '../L4-04-graph-compute-entry/index.html', 'label': 'L4-04 图执行入口'}},
)

# ------------------------------------------------------------------ 工具

def notes_for(parts, by_line):
    """把"上游行号 -> 注解"换算成 lessonkit 的 notes（键 = 段内 0-based 下标）。

    ★ lessonkit 的 notes 键是【段内】下标，多段引用时同一个键会在每一段里
      各插一次（注解会串到别的段去，而且不报错）。所以这里直接禁止多段引用
      带注解：多段就只靠 mark_src 与画面文字讲。
    """
    if len(parts) > 1:
        raise SystemExit('多段引用不能带注解（lessonkit 的 notes 键是段内下标，会串段）')
    a, b = parts[0]
    missing = set(by_line) - set(range(a, b + 1))
    if missing:
        raise SystemExit(f'注解行号不在引用区间内: {sorted(missing)}')
    return {ln - a: txt for ln, txt in by_line.items()}


def _rmap(parts, notes):
    """复刻 lessonkit._render_code 的行序；None 表示注解行 / 多段分隔行。"""
    out = []
    for pi, (a, b) in enumerate(parts):
        if pi:
            out.append(None)
        for k, ln in enumerate(range(a, b + 1)):
            out.append(ln)
            if k in notes:
                out.append(None)
    return out


def ridx(parts, notes, lines):
    """上游行号 -> 渲染下标，供 U.markLines 使用。"""
    m = _rmap(parts, notes)
    return '[' + ', '.join(str(m.index(ln)) for ln in lines) + ']'


# ------------------------------------------------------------------ 第 1 幕

P1 = [(33, 43)]

L.scene(
    kicker='L4 · 内存与调度',
    title='张量脚下那块内存：<span class="hl-a">buft</span> 是厂，<span class="hl-b">buffer</span> 是地',
    sub='buffer type 回答"这类内存怎么造、有什么属性"；buffer 回答"就是这一块"。',
    caption='回顾 L1-01：tensor 的 buffer / data 两个字段，指的就是本课这两个对象。',
    src=H, parts=P1, duration=17000,
    mark_src=[37, 38, 39, 40, 41, 42, 43],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:11px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">模型 / 中间结果</span><span class="arrow">-&gt;</span>
    <span class="chip a">buft（类型）</span><span class="arrow">-&gt;</span>
    <span class="chip b">buffer（一块内存）</span><span class="arrow">-&gt;</span>
    <span class="chip c">tensor.buffer / data</span>
  </div>
  <div class="row center" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'buft · buffer type', b: '这类内存<b>怎么造</b>、有什么属性：<br>对齐、上限、是不是 host', m: 'ggml_backend_buffer_type_t' },
  { c: 'b', t: 'buffer · 一块内存', b: '已经分配好的那一块：<br>基址、大小、用途', m: 'ggml_backend_buffer_t' },
  { c: 'c', t: 'tensor 的两个字段', b: '回顾 L1-01：<b>buffer</b> 指向这一块，<br><b>data</b> 是里面的地址', m: 'tensor->buffer / tensor->data' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '先分清两个词：<span class="k">type</span> 是"哪一类内存"，<span class="k">buffer</span> 是"这一块内存"。',
  '回顾 L1-01：每个 <span class="v">ggml_tensor</span> 都有 <span class="v">buffer</span> 与 <span class="v">data</span> 两个字段 —— 它们就是本课的主角。',
  '回顾 L3-01：这七个公共函数就是 <span class="v">buffer_type_i</span> 虚表的镜像。这里看它们的<b>实现与语义</b>。',
  '本课路线：七个函数 → buft 与 buffer 的绑定 → 数据面 → host / device 分工 → <span class="k">pinned host buffer</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
}));
tl.at(700 + 4 * 3400, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  U.markLines(document, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  msg.innerHTML = texts[3];
});
'''
)

# ------------------------------------------------------------------ 第 2 幕

P2 = [(34, 52)]
N2 = {
    37: 'get_name：必需。日志与报错都用它标识这块内存（"CPU" / "CUDA0"）',
    46: 'alloc_buffer：必需，也是这一层唯一的产出 —— 把"字节数"变成一块 buffer',
    51: 'get_alignment：必需，没有默认值 —— 分配器必须知道对齐要求（L4-01）',
}

L.scene(
    kicker='L4-03 · 契约',
    title='buft 的<span class="hl-a">三个必需函数</span>：名字、分配、对齐',
    sub='公共 API 全是虚表转发：断言 buft 非空，然后调 buft->iface —— 一行都没有业务逻辑。',
    caption='七个函数的全景在右边的表里；这一段是它们的实现在 ggml/src/ggml-backend.cpp:34-52。',
    src=C, parts=P2, notes=notes_for(P2, N2), duration=20000,
    mark_src=[34, 39, 41, 43, 48],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['公共函数', '作用', '必需性'],
  [['ggml_backend_buft_name', '这块内存类型叫什么', '必需'],
   ['ggml_backend_buft_alloc_buffer', '把字节数变成一块 buffer', '必需'],
   ['ggml_backend_buft_get_alignment', '这张量数据的对齐要求', '必需'],
   ['ggml_backend_buft_get_max_size', '单块 buffer 的尺寸上限', '可选：默认 SIZE_MAX'],
   ['ggml_backend_buft_get_alloc_size', '一个张量实际要占多少字节', '可选：默认 ggml_nbytes'],
   ['ggml_backend_buft_is_host', '这块内存 CPU 能不能直接读写', '可选：默认 false'],
   ['ggml_backend_buft_get_device', '这个 buft 属于哪个设备', '字段直读']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);

const rows = Array.from(t.body.querySelectorAll('tr'));
const msg = wrap.querySelector('#msg');
const texts = [
  '这是 buft 的全部公共 API —— 与 L3-01 的 <span class="v">buffer_type_i</span> 虚表一一对应。',
  '<span class="v">ggml_backend_buft_name</span>：必需。日志与报错都用它标识这块内存。',
  '<span class="v">ggml_backend_buft_alloc_buffer</span>：必需，也是这一层唯一的产出。第 41-44 行是零尺寸分支：返回一个哑 buffer，不真分配。',
  '<span class="v">ggml_backend_buft_get_alignment</span>：必需，同样没有默认值 —— 对齐是分配器的硬前提（L4-01 用它给每个节点补齐）。',
  '剩下四项：三个可选（下一幕看它们的默认值），<span class="v">get_device</span> 只是一行字段直读。'
];
const steps = [@@S0@@, @@S1@@, @@S2@@];
steps.forEach((st, i) => tl.at(700 + i * 3400, () => {
  rows.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; });
  U.markLines(document, st);
  msg.innerHTML = texts[i];
}));
tl.at(700 + 3 * 3400, () => {
  rows.forEach((r, k) => { r.className = (k >= 3) ? 'on' : ''; });
  U.markLines(document, @@ALL@@);
  msg.innerHTML = texts[4];
});
'''.replace('@@S0@@', ridx(P2, notes_for(P2, N2), [34])) \
   .replace('@@S1@@', ridx(P2, notes_for(P2, N2), [39, 41, 43])) \
   .replace('@@S2@@', ridx(P2, notes_for(P2, N2), [48])) \
   .replace('@@ALL@@', ridx(P2, notes_for(P2, N2), [34, 39, 48]))
)

# ------------------------------------------------------------------ 第 3 幕

P3 = [(53, 87)]
N3 = {
    56: 'get_max_size：可选 —— 先判空再调；iface 没填就直接返回 SIZE_MAX',
    65: 'get_alloc_size：可选 —— iface 没填就用 ggml_nbytes(tensor)',
    71: '断言：返回值不能小于 ggml_nbytes，但【可以更大】',
    74: 'TAG_ALLOC_SIZE_EXPAND：量化类型与"可能扩张"的算子允许分配尺寸更大',
    83: 'is_host：可选 —— 同样先判空；没填一律按 false 处理',
    86: '所以"没实现 is_host"不等于"这是 host 内存"，而是被当成 device buffer',
}

L.scene(
    kicker='L4-03 · 契约',
    title='三个可选函数：<span class="hl-c">默认值也是契约</span>',
    sub='get_max_size 默认 SIZE_MAX、get_alloc_size 默认 ggml_nbytes、is_host 默认 false。',
    caption='这三个默认值决定了后端能少写多少代码；也决定了分配器必须用哪个函数算尺寸。',
    src=C, parts=P3, notes=notes_for(P3, N3), duration=22000,
    mark_src=[53, 55, 56, 57, 59, 62, 65, 66, 67, 71, 73, 74, 78, 81, 83, 86],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="cards" style="gap:11px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'get_max_size', b: '不填 <span class="cm" style="margin:0">SIZE_MAX</span><br>单块 buffer 不限大小', m: 'if (buft->iface.get_max_size)' },
  { c: 'b', t: 'get_alloc_size', b: '不填 <span class="cm" style="margin:0">ggml_nbytes</span><br>但允许返回<b>更大</b>的值', m: 'if (buft->iface.get_alloc_size)' },
  { c: 'c', t: 'is_host', b: '不填 <span class="cm" style="margin:0">false</span><br>缺省 = 当成 device buffer', m: 'if (buft->iface.is_host)' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:218px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '三个可选项各自有安全默认值 —— 这是后端可以"少填"的原因。',
  '<span class="v">get_max_size</span>：先判空再调（第 56-58 行），没填就是 <span class="v">SIZE_MAX</span>，注释把默认值写在旁边。',
  '<span class="v">get_alloc_size</span>：默认 <span class="v">ggml_nbytes</span>，但断言明确允许返回<b>更大</b>的值 —— 量化类型、repack、以及带 <span class="v">TAG_ALLOC_SIZE_EXPAND</span> 标记的算子都要额外空间。',
  '<span class="v">is_host</span>：同样先判空，没填就 <span class="v">return false</span> —— 缺省语义是"这不是 host 内存"。',
  '所以任何分配路径都必须用 <span class="v">get_alloc_size</span>，不能拿 <span class="v">ggml_nbytes</span> 顶替。'
];
const groups = [[53, 55, 56, 57, 59], [62, 65, 66, 67, 71, 73, 74, 78], [81, 83, 86]];
groups.forEach((g, i) => tl.at(700 + i * 4300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  U.markLines(document, i === 0 ? @@A@@ : (i === 1 ? @@B@@ : @@C@@));
  msg.innerHTML = texts[i + 1];
}));
tl.at(700 + 3 * 4300, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  U.markLines(document, @@ALL@@);
  msg.innerHTML = texts[4];
});
'''.replace('@@A@@', ridx(P3, notes_for(P3, N3), [53, 56, 57, 59])) \
   .replace('@@B@@', ridx(P3, notes_for(P3, N3), [62, 65, 71, 73, 74, 78])) \
   .replace('@@C@@', ridx(P3, notes_for(P3, N3), [81, 83, 86])) \
   .replace('@@ALL@@', ridx(P3, notes_for(P3, N3), [56, 65, 71, 74, 83]))
)

# ------------------------------------------------------------------ 第 4 幕

P4 = [(96, 110), (171, 185), (204, 207)]

L.scene(
    kicker='L4-03 · 绑定',
    title='★ <span class="hl-b">buffer</span> 只是 <span class="hl-a">buft</span> 的句柄',
    sub='buffer 里只存了虚表、buft 指针、context、size、usage；它的属性全部要回问 buft。',
    caption='对比 L3-01：那里看虚表长什么样，这里看虚表怎么被一个具体对象持有。',
    src=C, parts=P4, duration=24000,
    mark_src=[97, 98, 103, 104, 105, 106, 172, 176, 180, 184, 206],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:6px"></div>
    <div class="col grow" id="right" style="gap:6px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:3px">struct ggml_backend_buffer 的五个字段</div>';
[['iface', '这块 buffer 的虚表：读写、清空、释放都在这'],
 ['buft', '它属于哪一类内存 —— 唯一记录'],
 ['context', '后端私有数据（例如显存的指针）'],
 ['size', '这块内存多少字节'],
 ['usage', 'ANY / WEIGHTS / COMPUTE']].forEach(f => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
  e.innerHTML = '<span class="m">' + U.esc(f[0]) + '</span> — ' + f[1];
  left.appendChild(e);
});

const right = wrap.querySelector('#right');
const t = U.table(
  ['buffer 侧 API', '转发目标'],
  [['ggml_backend_buffer_get_alignment', 'buft_get_alignment'],
   ['ggml_backend_buffer_get_max_size', 'buft_get_max_size'],
   ['ggml_backend_buffer_get_alloc_size', 'buft_get_alloc_size'],
   ['ggml_backend_buffer_is_host', 'buft_is_host'],
   ['ggml_backend_buffer_get_type', '不转发：返回 buffer->buft']],
  { monoCols: [0] });
right.appendChild(t.el);
const rows = Array.from(t.body.querySelectorAll('tr'));

const msg = wrap.querySelector('#msg');
const texts = [
  '先看 <span class="v">ggml_backend_buffer_init</span>：它接收 buft，把它原样存进 <span class="v">.buft</span>，再记下 size 与 usage。',
  '所以 buffer = 虚表 + <b>buft 指针</b> + context + size + usage。它自己<b>不存</b>对齐、上限、host 属性。',
  '证据在回程：<span class="v">ggml_backend_buffer_get_alignment</span> 只有一行 —— 先取 <span class="v">buffer-&gt;buft</span>，再问它。',
  '四个查询全是这个形状：<span class="v">max_size</span> / <span class="v">alloc_size</span> / <span class="v">is_host</span> 都一样 —— 同一个 buft 造出来的 buffer，答案必然相同。',
  '唯一不转发的是 <span class="v">ggml_backend_buffer_get_type</span>：直接返回 <span class="v">buffer-&gt;buft</span>。有了它，拿到 buffer 的人才能反查 buft。',
  '回顾 L4-02：调度器的 <span class="v">backend_from_buffer()</span> 读的就是这个回程指针 —— 先知道"哪类内存"，再决定"哪个后端能读它"。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, @@A@@); });
tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, @@B@@); });
const pairs = [@@C@@, @@D@@, @@E@@, @@F@@];
pairs.forEach((st, i) => tl.at(8000 + i * 2800, () => {
  rows.forEach((r, k) => { r.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 2];
  U.markLines(document, st);
}));
tl.at(8000 + 4 * 2800 + 900, () => {
  rows.forEach(r => { r.className = ''; });
  msg.innerHTML = texts[5];
  U.markLines(document, @@G@@);
});
'''.replace('@@A@@', ridx(P4, {}, [97, 103, 106])) \
   .replace('@@B@@', ridx(P4, {}, [104, 105, 106])) \
   .replace('@@C@@', ridx(P4, {}, [172])) \
   .replace('@@D@@', ridx(P4, {}, [176])) \
   .replace('@@E@@', ridx(P4, {}, [180, 184])) \
   .replace('@@F@@', ridx(P4, {}, [206])) \
   .replace('@@G@@', ridx(P4, {}, [172, 176, 180, 184, 206]))
)

# ------------------------------------------------------------------ 第 5 幕

P5 = [(335, 348), (350, 363), (409, 423)]

L.scene(
    kicker='L4-03 · 数据面',
    title='数据的进出：<span class="hl-d">set</span> / <span class="hl-e">get</span> / <span class="hl-f">memset</span> 都先解析 <span class="v">tensor-&gt;buffer</span>',
    sub='三个函数的开头一行完全相同：有 view_src 就用源张量的 buffer，否则用自己的。',
    caption='回顾 L1-01：view_src 非空表示"数据在别人那里"—— 这里就是它影响读写的时刻。',
    src=C, parts=P5, duration=24000,
    mark_src=[336, 337, 338, 345, 347, 352, 353, 355, 360, 362, 411, 417, 420, 422],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">tensor</span><span class="arrow">-&gt;</span>
    <span class="chip a">view_src ? view_src-&gt;buffer : tensor-&gt;buffer</span><span class="arrow">-&gt;</span>
    <span class="chip b">buffer-&gt;iface.*</span>
  </div>
  <div class="row center" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'd', t: 'ggml_backend_tensor_set', b: '写入：把 host 内存的数据<br>写进张量所在的 buffer', m: 'iface.set_tensor' },
  { c: 'e', t: 'ggml_backend_tensor_get', b: '读出：从张量所在的 buffer<br>拷回 host 内存', m: 'iface.get_tensor' },
  { c: 'f', t: 'ggml_backend_tensor_memset', b: '填充：要求后端实现<br>memset_tensor，否则断言失败', m: 'iface.memset_tensor' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '三个函数，同一个开头：<span class="v">tensor-&gt;view_src ? tensor-&gt;view_src-&gt;buffer : tensor-&gt;buffer</span>。',
  '回顾 L1-01：<span class="v">view_src</span> 非空 = 这个张量不拥有数据。所以读写要去找<b>源张量</b>的 buffer。',
  '<span class="v">set</span> 还要过越界断言（<span class="v">offset + size</span> 不能超过 <span class="v">ggml_nbytes</span>），然后转发给 buffer 虚表的 <span class="v">set_tensor</span>。',
  '<span class="v">get</span> 是 set 的镜像：同样的解析、同样的越界检查，只是方向相反。',
  '<span class="v">memset</span> 多一条断言：<span class="v">buf-&gt;iface.memset_tensor != NULL</span> —— 填充能力<b>不是每个 buffer 都有</b>。',
  '一句话：<span class="k">张量的读写从来不是直接 memcpy，而是"找到 buffer，再让 buffer 自己搬"</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i];
  U.markLines(document, i === 0 ? @@A@@ : (i === 1 ? @@B@@ : @@C@@));
}));
tl.at(700 + 3 * 3400, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[5];
  U.markLines(document, @@ALL@@);
});
'''.replace('@@A@@', ridx(P5, {}, [336, 337, 338, 347])) \
   .replace('@@B@@', ridx(P5, {}, [352, 353, 360, 362])) \
   .replace('@@C@@', ridx(P5, {}, [411, 417, 420, 422])) \
   .replace('@@ALL@@', ridx(P5, {}, [337, 347, 362, 420]))
)

# ------------------------------------------------------------------ 第 6 幕

P6 = [(216, 222), (488, 509), (2464, 2478)]

L.scene(
    kicker='L4-03 · 分工',
    title='<span class="hl-e">host buffer</span> 与 <span class="hl-f">device buffer</span> 的分工',
    sub='跨 buffer 搬张量时，is_host 决定"谁来搬"：host 侧用 set / get，device 之间才走后端 cpy_tensor。',
    caption='回顾 L2-03：权重从磁盘到显存正是这条路 —— host 侧读出来、set 进去。',
    src=C, parts=P6, duration=28000,
    mark_src=[217, 218, 219, 489, 495, 496, 497, 498, 499, 501, 504, 505, 506, 2464, 2465, 2470, 2478],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="formula" style="text-align:center">
    判据只有一个：<span class="m">ggml_backend_buffer_is_host(buffer)</span>
    <span class="arrow">-&gt;</span> <span class="m">buft-&gt;iface.is_host(buft)</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '① src 在 host', b: '直接 <span class="cm" style="margin:0">set</span> 进 dst：读的一端是 CPU 能碰的内存', m: 'ggml_backend_tensor_set' },
  { c: 'b', t: '② dst 在 host', b: '直接从 src <span class="cm" style="margin:0">get</span> 出来：写的一端是 CPU 能碰的内存', m: 'ggml_backend_tensor_get' },
  { c: 'c', t: '③ 两边都不是 host', b: '先试目的 buffer 的 <span class="cm" style="margin:0">cpy_tensor</span>；没有就 malloc 中转', m: 'ggml_backend_buffer_copy_tensor' },
  { c: 'd', t: 'host buffer 是什么', b: 'CPU 的默认 buft 里 <span class="cm" style="margin:0">is_host</span> 返回 true —— 它就是 system memory', m: 'ggml_backend_cpu_buffer_type_is_host' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">ggml_backend_tensor_copy</span> 只有 20 行，却是 host / device 分工的全部判据。',
  '<span class="v">ggml_backend_buffer_copy_tensor</span> 取的是 <b>dst</b> 的 <span class="v">cpy_tensor</span>（视图则取 view_src 的）—— 能力由目的地提供。',
  '<span class="k">src 在 host</span>：直接 set 到 dst。这正是 L2-03 权重上传的形状。',
  '<span class="k">dst 在 host</span>：直接从 src get 出来 —— 例如把显存里的结果取回 CPU。',
  '两边都不在 host：先试后端的 <span class="v">cpy_tensor</span>；连它都没有就 malloc 一块内存中转，调试版会打印 <span class="v">slow copy</span> 警告。',
  '回到定义：CPU 的 buft 里 <span class="v">is_host</span> 直接 <span class="v">return true</span>（第 2464-2468 行）—— 因为它的 buffer 就是 system memory。',
  '所以 host 与 device 的差别，说到底就是 <span class="k">"CPU 能不能直接解引用这块内存"</span>；也只有 host 侧能当跨设备搬运的跳板。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, @@A@@); });
tl.at(4200, () => { msg.innerHTML = texts[1]; U.markLines(document, @@B@@); });
defs.forEach((_, i) => tl.at(7600 + i * 3600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.32'; });
  msg.innerHTML = texts[i + 2];
  U.markLines(document, i === 0 ? @@C@@ : (i === 1 ? @@D@@ : (i === 2 ? @@E@@ : @@F@@)));
}));
tl.at(7600 + 4 * 3600, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[6];
  U.markLines(document, @@ALL@@);
});
'''.replace('@@A@@', ridx(P6, {}, [217, 218, 489])) \
   .replace('@@B@@', ridx(P6, {}, [217, 219])) \
   .replace('@@C@@', ridx(P6, {}, [495, 496])) \
   .replace('@@D@@', ridx(P6, {}, [497, 498])) \
   .replace('@@E@@', ridx(P6, {}, [219, 499, 501, 504, 505, 506])) \
   .replace('@@F@@', ridx(P6, {}, [2464, 2465, 2470, 2478])) \
   .replace('@@ALL@@', ridx(P6, {}, [217, 219, 495, 497, 499, 504, 2465]))
)

# ------------------------------------------------------------------ 第 7 幕

P7 = [(152, 159), (615, 632), (2122, 2147)]

L.scene(
    kicker='L4-03 · 装配',
    title='<span class="hl-g">手工装配</span>：宿主内存 → buffer → 张量的 data',
    sub='设备能给出三种 buffer；把已经存在的地址装成张量，靠的是 tensor_alloc。',
    caption='回顾 L2-03：mmap 权重落位走的就是 buffer_from_host_ptr + tensor_alloc 这两步。',
    src=C, parts=P7, duration=28000,
    mark_src=[152, 155, 156, 615, 617, 620, 622, 626, 631, 2122, 2129, 2130, 2134, 2139, 2144, 2145, 2146],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">mmap 的地址</span><span class="arrow">-&gt;</span>
    <span class="chip a">buffer_from_host_ptr</span><span class="arrow">-&gt;</span>
    <span class="chip b">buffer</span><span class="arrow">-&gt;</span>
    <span class="chip c">tensor_alloc</span><span class="arrow">-&gt;</span>
    <span class="chip d">tensor-&gt;data</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '设备的默认 buft', b: '设备内存（例如显存）；后端层的分配入口都指向它', m: 'dev_buffer_type' },
  { c: 'c', t: '设备的 host buft', b: '宿主内存；设备可以不提供 —— 没实现就返回 NULL', m: 'dev_host_buffer_type' },
  { c: 'g', t: '宿主指针包成 buffer', b: '地址已经在手上（mmap / 页缓存），直接包一块 buffer', m: 'dev_buffer_from_host_ptr' },
  { c: 'd', t: '挂成张量', b: 'tensor_alloc 把张量钉到 buffer 内的某个地址上', m: 'ggml_backend_tensor_alloc' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '设备的三个 buffer 入口：默认（设备内存）、host（宿主内存，可为 NULL）、把宿主指针包成 buffer。',
  '<span class="v">ggml_backend_dev_host_buffer_type</span> 只有 8 行，关键是那句 <span class="v">if (... == NULL) return NULL</span>：这个能力是<b>可选</b>的。',
  '<span class="v">ggml_backend_dev_buffer_from_host_ptr</span>：宿主内存已经在手 —— 不为它再分配，直接包成 buffer（mmap 走这条）。',
  '<span class="v">ggml_backend_tensor_alloc(buffer, tensor, addr)</span>：把已有地址挂成张量的 data，最后调用 <span class="v">init_tensor</span>。',
  '四条断言把前提钉死：tensor 必须还没有 buffer / data / view_src，且 addr 必须落在 buffer 区间内（meta buffer 除外）。',
  '对照 <span class="v">ggml_backend_view_init</span>：视图不分配 —— 直接借用源张量的 buffer，data = 源 data + view_offs。',
  '回顾 L2-03：mmap 的权重就是这样落位的 —— 先把映射的地址包成 buffer，再把每个张量钉到各自偏移上。'
];
tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; U.markLines(document, @@A@@); });
tl.at(4200, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, @@B@@); });
tl.at(7700, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, @@C@@); });
tl.at(11200, () => { els[3].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, @@D@@); });
tl.at(15200, () => { msg.innerHTML = texts[4]; U.markLines(document, @@E@@); });
tl.at(19200, () => { msg.innerHTML = texts[5]; U.markLines(document, @@F@@); });
tl.at(23200, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[6];
  U.markLines(document, @@ALL@@);
});
'''.replace('@@A@@', ridx(P7, {}, [615, 617])) \
   .replace('@@B@@', ridx(P7, {}, [620, 622, 626])) \
   .replace('@@C@@', ridx(P7, {}, [631])) \
   .replace('@@D@@', ridx(P7, {}, [152, 156, 2134, 2144, 2145, 2146])) \
   .replace('@@E@@', ridx(P7, {}, [2139, 2140])) \
   .replace('@@F@@', ridx(P7, {}, [2122, 2129, 2130])) \
   .replace('@@ALL@@', ridx(P7, {}, [617, 622, 631, 156, 2144, 2129]))
)

# ------------------------------------------------------------------ 第 8 幕

P8 = [(1530, 1578)]
N8 = {
    1530: '前提：use_mmap 与 check_tensors 都为假 —— 否则直接返回 nullptr',
    1538: '从一块已有 buffer 反查 buft、再反查 device —— 第 2/4 幕的 API 在这里被用上',
    1546: '只有"默认 buft"才走异步上传：host buft 自己不算',
    1554: '三个能力位一起查：async + host_buffer + events',
    1560: '向设备要它的 host buft —— 就是第 7 幕的 dev_host_buffer_type',
    1567: '注释直接写明：create pinned memory buffers',
    1569: '用同一个 buft_alloc_buffer 分配 pinned 缓冲（不是 malloc）',
    1577: '缓冲留下来备用：host_buffers',
    1578: 'buffer_get_base 拿到基址：文件内容就读进这块固定内存',
}

L.scene(
    kicker='L4-03 · 验收点',
    title='★ <span class="hl-a">pinned host buffer</span> 到底用在什么场景',
    sub='不用 mmap 加载模型时：为了异步上传权重到显存，先把权重读进设备的 pinned host buffer。',
    caption='这一段在 src/llama-model-loader.cpp:1530-1578（load_all_data 的 upload_backend lambda）；该文件的逐字展开见 L2-03。',
    src=M, parts=P8, notes=notes_for(P8, N8), duration=32000,
    mark_src=[1530, 1531, 1538, 1539, 1546, 1552, 1554, 1560, 1567, 1569, 1577, 1578],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">GGUF 文件</span><span class="arrow">-&gt;</span>
    <span class="chip a">pinned host buffer</span><span class="arrow">-&gt;</span>
    <span class="chip b">显存 buffer</span><span class="arrow">-&gt;</span>
    <span class="chip c">图算起来</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '什么时候走这条路', b: '不用 mmap、也不校验张量时（第 1527 行的门槛）：权重必须从文件读进来', m: 'use_mmap == false' },
  { c: 'c', t: '三个能力位一起查', b: 'async + host_buffer + events，缺一个就退回同步加载', m: 'props.caps.*' },
  { c: 'b', t: '中转站', b: '向设备要 host buft，分配 pinned 内存并配一个事件', m: 'dev_host_buffer_type' },
  { c: 'd', t: '怎么用掉', b: '文件读进 pinned 内存，再 set_async 上传到显存', m: 'ggml_backend_tensor_set_async' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.32');

const msg = wrap.querySelector('#msg');
const texts = [
  '先给答案：<b>不用 mmap 加载模型时</b>，权重要从文件读进显存 —— 中间那块中转内存就是 pinned host buffer。',
  '为什么要 pinned：只有固定（不换页）的宿主内存才能被设备直接 DMA，异步上传才有意义。',
  '判据是三个能力位：<span class="v">caps.async</span> + <span class="v">caps.host_buffer</span> + <span class="v">caps.events</span>（第 1554 行）—— <span class="v">host_buffer</span> 就是 pinned 的开关。',
  '拿到 <span class="v">host_buft</span> 之后，用的是<b>同一个</b> <span class="v">ggml_backend_buft_alloc_buffer</span> —— pinned 内存与显存走同一套 API。',
  '分配完立刻拿基址（第 1578 行）；紧接着还给每块缓冲配一个事件（第 1580-1587 行，见 source.md 第十一节），复用前先等上一次上传结束。',
  '上传在 1727 行：<span class="v">ggml_backend_tensor_set_async</span>。一句话：<span class="k">pinned host buffer = 非 mmap 路径下的异步上传中转站</span>。'
];
tl.at(700, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; U.markLines(document, @@A@@); });
tl.at(5200, () => { msg.innerHTML = texts[1]; U.markLines(document, @@B@@); });
tl.at(9200, () => { els[1].style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, @@C@@); });
tl.at(13400, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[3]; U.markLines(document, @@D@@); });
tl.at(17600, () => { els[3].style.opacity = '1'; msg.innerHTML = texts[4]; U.markLines(document, @@E@@); });
tl.at(21800, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[5];
  U.markLines(document, @@ALL@@);
});
'''.replace('@@A@@', ridx(P8, notes_for(P8, N8), [1530, 1531, 1538, 1539])) \
   .replace('@@B@@', ridx(P8, notes_for(P8, N8), [1554])) \
   .replace('@@C@@', ridx(P8, notes_for(P8, N8), [1552, 1560])) \
   .replace('@@D@@', ridx(P8, notes_for(P8, N8), [1567, 1569, 1577, 1578])) \
   .replace('@@E@@', ridx(P8, notes_for(P8, N8), [1546, 1560, 1569])) \
   .replace('@@ALL@@', ridx(P8, notes_for(P8, N8), [1554, 1560, 1569, 1578]))
)

# ------------------------------------------------------------------ 第 9 幕

P9 = [(248, 263)]
N9 = {
    250: '后端层的入口：默认 buft = 设备的 buft',
    254: 'ggml_backend_alloc_buffer = 默认 buft 的 alloc_buffer',
    258: '对齐与最大尺寸也走默认 buft —— 这两个值正是分配器（L4-01）要的',
    262: '所以"给后端分配内存"这条路，从头到尾不需要知道 buft 是什么',
}

L.scene(
    kicker='L4-03 · 收束',
    title='把这一课压成一张表',
    sub='buft 七个函数、buffer 一套句柄、host / device 一条判据 —— 最后都收在"默认 buft"这一个入口上。',
    caption='下一课 L4-04：图执行入口 ggml_backend_graph_compute 与异步。',
    src=C, parts=P9, notes=notes_for(P9, N9), duration=26000,
    mark_src=[248, 250, 253, 254, 257, 258, 261, 262],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['环节', '走哪个函数', '本课第几幕', '谁在调用'],
  [['造一块内存', 'ggml_backend_buft_alloc_buffer', '2', '模型加载、分配器、调度器'],
   ['问属性', 'get_alignment / get_max_size / get_alloc_size / is_host', '2 / 3', '分配器 L4-01、拷贝路由'],
   ['读写数据', 'tensor_set / tensor_get / tensor_memset', '5', '权重上传、KV cache、输出'],
   ['跨内存搬运', 'is_host + cpy_tensor', '6', '调度器的跨设备副本 L4-02'],
   ['挂上已有内存', 'buffer_from_host_ptr + tensor_alloc', '7', 'mmap 权重落位 L2-03'],
   ['异步中转', 'dev_host_buffer_type（pinned）', '8', '非 mmap 权重上传']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '不看代码回答：<b>pinned host buffer 在什么场景下被用到</b>？为什么必须是 pinned？',
  '场景：<b>不用 mmap 加载权重</b>时，权重需要从文件读进显存 —— 中间那块中转内存就是 pinned host buffer。<br>'
  + '证据链（<span class="mono">src/llama-model-loader.cpp</span> 的 upload_backend lambda）：<br>'
  + '1) 门槛：<span class="mono">use_mmap || check_tensors</span> 为真就直接返回（1527 行）；<br>'
  + '2) 能力检查：<span class="mono">props.caps.async &amp;&amp; props.caps.host_buffer &amp;&amp; props.caps.events</span>（1554 行）；<br>'
  + '3) 取设备的 host buft：<span class="mono">ggml_backend_dev_host_buffer_type(dev)</span>（1560 行）；<br>'
  + '4) 分配 pinned 缓冲：<span class="mono">ggml_backend_buft_alloc_buffer(host_buft, buffer_size)</span>（1569 行），基址来自 <span class="mono">ggml_backend_buffer_get_base</span>（1578 行）；<br>'
  + '5) 配事件：<span class="mono">ggml_backend_event_new</span>（1580 行），复用前先 <span class="mono">ggml_backend_event_synchronize</span>（1706 行）；<br>'
  + '6) 真正的上传：<span class="mono">ggml_backend_tensor_set_async</span>（1727 行）。<br>'
  + '为什么 pinned：只有固定不换页的宿主内存才能被设备直接 DMA，读文件与上传显存才能重叠。<br>'
  + '对照：走 mmap 时不需要它 —— 数据本来就在页缓存里，<span class="mono">buffer_from_host_ptr</span> 直接把映射包成 buffer（L2-03）。'));

const msg = wrap.querySelector('#msg');
const rows = Array.from(t.body.querySelectorAll('tr'));
const texts = [
  '回头看最后一组函数：后端层的四个便捷入口。',
  '<span class="v">ggml_backend_get_default_buffer_type</span> 就是"问设备要它的默认 buft"。',
  '<span class="v">ggml_backend_alloc_buffer</span> = 默认 buft 的 <span class="v">alloc_buffer</span>：调用方连 buft 都不用拿。',
  '<span class="v">get_alignment</span> / <span class="v">get_max_size</span> 也走默认 buft —— 这两个值正是 L4-01 的分配器要的：用 alignment 补齐节点、一旦超过 max_size 就换一块 buffer。',
  '所以整条链是：<span class="v">backend -&gt; device -&gt; 默认 buft -&gt; buffer -&gt; tensor.data</span>。',
  '记住一句话：<span class="k">buft 是"哪类内存"的契约，buffer 是"这一块内存"的句柄；host 与 device 的分工只由一个 is_host 决定</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(2900, () => { msg.innerHTML = texts[1]; U.markLines(document, @@A@@); rows[0].className = 'on'; });
tl.at(6400, () => { msg.innerHTML = texts[2]; U.markLines(document, @@B@@); rows[0].className = ''; rows[1].className = 'on'; });
tl.at(10400, () => { msg.innerHTML = texts[3]; U.markLines(document, @@C@@); rows[1].className = ''; rows[2].className = 'on'; });
tl.at(15400, () => { msg.innerHTML = texts[4]; U.markLines(document, @@D@@); rows[2].className = ''; rows[3].className = 'on'; rows[4].className = 'on'; rows[5].className = 'on'; });
tl.at(21000, () => {
  rows.forEach(r => { r.className = ''; });
  U.markLines(document, @@ALL@@);
  msg.innerHTML = texts[5];
});
'''.replace('@@A@@', ridx(P9, notes_for(P9, N9), [248, 250])) \
   .replace('@@B@@', ridx(P9, notes_for(P9, N9), [253, 254])) \
   .replace('@@C@@', ridx(P9, notes_for(P9, N9), [257, 258])) \
   .replace('@@D@@', ridx(P9, notes_for(P9, N9), [261, 262])) \
   .replace('@@ALL@@', ridx(P9, notes_for(P9, N9), [250, 254, 258, 262]))
)

# ------------------------------------------------------------------ source.md

L.note('**一句话**：`ggml_tensor` 的 `buffer` / `data` 两个字段指向的，就是本课的两个对象 —— '
       '**buffer type**（哪一类内存：怎么造、对齐多少、上限多大、是不是 host）与 '
       '**buffer**（这一块内存本身）。')
L.note('本课与 L4-02 覆盖同样两个文件，但焦点不同：L4-02 讲**调度器怎么按 buffer type 切图**，'
       '本课讲 **buffer type 本身**——分配、初始化、属性查询，以及 host buffer 与 device buffer 的分工。'
       '切分算法不在这里重复。')

L.section(
    '一、两个对象：buffer type 与 buffer',
    '公共头把两者都做成**不透明句柄**（`typedef struct ggml_backend_buffer_type *` 与 '
    '`typedef struct ggml_backend_buffer *`，第 24-25 行）。所谓 "type" 不是 C++ 的类型，'
    '而是"一类内存"：它知道这块内存怎么分配、对齐要求是多少、单块上限多大、CPU 能不能直接读写。\n\n'
    '下面这七个函数就是这一层的全部公共 API —— 与 L3-01 逐字展开过的 `ggml_backend_buffer_type_i` '
    '虚表一一对应（6 个函数指针 + 一个 `get_device`）。',
    src=H, parts=[(33, 43)], lang='c')

L.section(
    '二、★ 七个公共函数：转发、默认值与归属',
    '七个函数的实现有一个共同形状：**断言 buft 非空，然后转发给 `buft->iface`**。'
    '公共 API 里没有一行业务逻辑 —— 这正是 L3-01 说的"公共 API 是虚表的镜像"。\n\n'
    '| 函数 | 必需性 | 缺省行为 |\n|---|---|---|\n'
    '| `ggml_backend_buft_name` | 必需 | 无 —— 直接调 `iface.get_name` |\n'
    '| `ggml_backend_buft_alloc_buffer` | 必需 | 无；`size == 0` 时返回一个哑 buffer |\n'
    '| `ggml_backend_buft_get_alignment` | 必需 | 无 —— 分配器必须知道对齐 |\n'
    '| `ggml_backend_buft_get_max_size` | 可选 | `SIZE_MAX`（等于不限） |\n'
    '| `ggml_backend_buft_get_alloc_size` | 可选 | `ggml_nbytes(tensor)` |\n'
    '| `ggml_backend_buft_is_host` | 可选 | `false` |\n'
    '| `ggml_backend_buft_get_device` | —— | 直接读 `buft->device` 字段，不走虚表 |\n\n'
    '三处"可选"是这一层的设计要点：**后端可以少填，框架给安全默认值**。'
    '其中 `get_alloc_size` 最容易被误解 —— 它的断言（第 69-74 行）明确允许返回值**大于** '
    '`ggml_nbytes`（量化类型、repack、以及源码里那个 `TAG_ALLOC_SIZE_EXPAND` 标记的算子）。'
    '所以任何分配路径都必须用 `get_alloc_size`，不能拿 `ggml_nbytes` 顶替。',
    src=C, parts=[(34, 92)], lang='c')

L.section(
    '三、★ buffer 与 buft 的绑定：一行赋值',
    '`ggml_backend_buffer_init` 是"buffer 从哪来"的答案：它接收 buft，把它原样存进 `.buft` 字段，'
    '再记下 `context`（后端私有数据）、`size` 与 `usage`（初值 `ANY`）。'
    '每个 buft 的 `alloc_buffer` 实现最后都要调它 —— 第六节的 CPU 实现就是活例子。\n\n'
    'buffer 结构里**没有**对齐、上限、host 这三个属性：它们全部属于 buft。'
    '唯一的回程是 `ggml_backend_buffer_get_type`（直接返回 `buffer->buft`），'
    'L4-02 里调度器的 `backend_from_buffer()` 读的就是它。',
    src=C, parts=[(96, 110), (204, 207)], lang='c')

L.section(
    '四、回程：buffer 的属性查询全是转发',
    '`ggml_backend_buffer_get_alignment` / `get_max_size` / `get_alloc_size` / `is_host` 四个函数'
    '各自只有一行：先 `ggml_backend_buffer_get_type(buffer)`，再调对应的 `ggml_backend_buft_*`。\n\n'
    '这解释了一件容易搞混的事：**"这块 buffer 是不是 host"不是 buffer 的性质，而是它那一类内存的性质**。'
    '同一个 buft 造出来的所有 buffer，`is_host` 的答案必然相同。',
    src=C, parts=[(171, 185)], lang='c')

L.section(
    '五、buffer 的生命周期：五个可选回调',
    'buffer 侧的公共函数也是"转发 + 可选"：`free_buffer` 与 `reset` 先判空再调；'
    '`clear` 只在非零尺寸时调（注释写明"零尺寸的 buffer 可以不实现它"）；'
    '`get_base` 对零尺寸 buffer 直接返回 NULL；`init_tensor` 是"张量挂上来"时的回调，'
    '不实现就返回 `GGML_STATUS_SUCCESS`。五个函数都允许后端留空。\n\n'
    '`ggml_backend_buffer_free` 的顺序值得一提：先给后端机会释放自己的资源（`iface.free_buffer`），'
    '最后才 `delete buffer` —— 后端可以只实现"释放自己的部分"。',
    src=C, parts=[(116, 125), (132, 159), (161, 169), (209, 214)], lang='c')

L.section(
    '六、★ 契约落地：CPU 的 buffer type',
    '这是全套课件里最"完整"的一个真实 buft：`ggml_backend_cpu_buffer_type()`。'
    '源码注释写明它定义在 ggml-backend.cpp 里（而不是 CPU 后端目录），'
    '**目的是让所有后端都能拿到它**（第 2437-2439 行）。\n\n'
    '| 虚表项 | 实现 | 说明 |\n|---|---|---|\n'
    '| `get_name` | 返回 `"CPU"` | 必需项，一行 |\n'
    '| `alloc_buffer` | `ggml_aligned_malloc` + `ggml_backend_buffer_init` | 分配 + 装虚表 + 绑 buft |\n'
    '| `get_alignment` | 返回 `TENSOR_ALIGNMENT` | 编译期常量 |\n'
    '| `get_max_size` | `NULL` | 走默认值 `SIZE_MAX` |\n'
    '| `get_alloc_size` | `NULL` | 走默认值 `ggml_nbytes` |\n'
    '| `is_host` | 返回 `true` | CPU 的 buffer 就是 host buffer |\n\n'
    '宿主结构体里 `device` 是 `NULL`（注释留着 FIXME），所以对 CPU 的 buft 调 '
    '`ggml_backend_buft_get_device` 会得到 NULL —— 用这个返回值之前必须判空。',
    src=C, parts=[(2437, 2485)], lang='c')

L.section(
    '七、数据面：set / get / memset / cpy_tensor',
    '张量数据的读写入口都长一个样：**先解析出 buffer，再让 buffer 自己搬**。'
    '`ggml_backend_tensor_set` / `_get` / `_memset` 三个函数的第一行都是\n\n'
    '```text\n'
    'buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;\n'
    '```\n\n'
    '这正是 L1-01 里 `view_src` 的语义在数据面的兑现：视图不拥有数据，读写要走源张量的 buffer。'
    '`_get` / `_set` 有越界断言，`_memset` 还额外要求后端实现了 `memset_tensor` —— '
    '**不是每个 buffer 都能被填充**。\n\n'
    '第 216-222 行的 `ggml_backend_buffer_copy_tensor` 是另一条路：它取 **dst**（视图则取 view_src）'
    '的 `cpy_tensor`，由目的地决定能不能直接搬 —— 这是后端做 DMA 的机会，没有就返回 `false`。',
    src=C, parts=[(216, 222), (335, 348), (350, 363), (409, 423)], lang='c')

L.section(
    '八、★ host 与 device 的分工：一个 is_host 定方向',
    '`ggml_backend_tensor_copy` 只有 20 行，却把 host / device 的分工写完了：\n\n'
    '| 情形 | 走哪条路 |\n|---|---|\n'
    '| `src` 是 host buffer | `ggml_backend_tensor_set(dst, src->data, ...)` |\n'
    '| `dst` 是 host buffer | `ggml_backend_tensor_get(src, dst->data, ...)` |\n'
    '| 两边都不是 host | `ggml_backend_buffer_copy_tensor` → 后端的 `cpy_tensor` |\n'
    '| 连 `cpy_tensor` 都没有 | `malloc` 一块内存中转，调试版打印 `slow copy` 警告 |\n\n'
    '关键在于"host"这个判据来自 buft 的 `is_host`，而它是**可选**的（缺省 `false`）—— '
    '也就是说，一个后端如果不实现 `is_host`，它的 buffer 一律被当成 device buffer，'
    '所有拷贝都要绕道 `cpy_tensor` 或 malloc 中转。'
    'L2-03 的权重上传（host 侧读出来、set 进显存）正是表中的第一行。',
    src=C, parts=[(488, 509)], lang='c')

L.section(
    '九、★ 手工装配：宿主指针 → buffer → 张量的 data',
    '设备侧有三个 buffer 入口：默认 buft（`ggml_backend_dev_buffer_type`）、'
    'host buft（`ggml_backend_dev_host_buffer_type`，**可选，可能返回 NULL**）、'
    '以及把已有宿主指针包成 buffer 的 `ggml_backend_dev_buffer_from_host_ptr`。\n\n'
    '拿到 buffer 之后，把张量钉上去的是 `ggml_backend_tensor_alloc(buffer, tensor, addr)`：'
    '四条断言要求张量还没有 `buffer` / `data` / `view_src`，且 `addr` 必须落在 buffer 区间内'
    '（meta buffer 除外）；然后两步赋值，最后调用 `init_tensor` 回调。'
    '对照 `ggml_backend_view_init`：视图连 addr 都不用给，直接借用源张量的 buffer 与 `view_offs`。\n\n'
    '回顾 L2-03：mmap 权重落位就是 `buffer_from_host_ptr`（把映射包成 buffer）加 '
    '`tensor_alloc`（把每个张量钉到各自的 `weight->offs`）这两步。',
    src=C, parts=[(152, 159), (615, 632), (2122, 2147)], lang='c')

L.section(
    '十、★ pinned host buffer 的声明：设备能力位',
    '`struct ggml_backend_dev_caps` 是设备对外的能力清单，五个布尔位各自带一行注释。'
    '本课关心的两个是：\n\n'
    '| 能力位 | 注释原文 | 含义 |\n|---|---|---|\n'
    '| `host_buffer` | `pinned host buffer` | 设备能提供一块**固定（page-lock）的宿主内存** |\n'
    '| `buffer_from_host_ptr` | `creating buffers from host ptr` | 能把已有宿主指针包成 buffer（mmap） |\n\n'
    '注意 `host_buffer` 与 `buffer_from_host_ptr` 是**两位**：前者是"设备能给你一块 pinned 内存"，'
    '后者是"设备能接管你已经有的内存"。第九节的 `ggml_backend_dev_host_buffer_type` 对应第一位，'
    '`ggml_backend_dev_buffer_from_host_ptr` 对应第二位 —— 也正因为它可选，函数必须能返回 NULL。',
    src=H, parts=[(147, 159)], lang='c')

L.section(
    '十一、★ pinned host buffer 的使用场景（验收点）',
    '答案在模型加载器里：**不用 mmap 时，权重需要从文件读进显存，中间那块中转内存就是 pinned host buffer**。\n\n'
    '`llama_model_loader::load_all_data` 里有一个 `upload_backend` lambda：'
    '`use_mmap || check_tensors` 为真就直接放弃这条路（第 1527 行）；'
    '否则它先确认手上那块 buffer 就是设备的**默认 buft**（第 1546 行），'
    '再检查三个能力位 `caps.async && caps.host_buffer && caps.events`（第 1554 行），'
    '然后 `ggml_backend_dev_host_buffer_type(dev)` 取 host buft（第 1560 行），'
    '用同一个 `ggml_backend_buft_alloc_buffer` 分配 pinned 缓冲（第 1569 行），'
    '并从 `ggml_backend_buffer_get_base` 拿基址（第 1578 行）。'
    '每个缓冲还配一个事件（`ggml_backend_event_new`，第 1580 行），'
    '用来在复用之前等上一次上传完成。\n\n'
    '**为什么必须是 pinned**：只有固定不换页的宿主内存才能被设备直接 DMA，'
    '异步上传才能与读文件重叠。第六节里 CPU 的默认 buffer 也是 host 内存，但它**不是** '
    'pinned —— 所以设备要单独给出一个 `host_buffer_type`。',
    src=M, parts=[(1530, 1588)], lang='c')

L.section(
    '十二、pinned 缓冲怎么被用掉：读文件 + 异步上传',
    '同一文件的 1699-1735 行是消费端：把文件内容读进 pinned 缓冲（`file->read_raw_unsafe`），'
    '再 `ggml_backend_tensor_set_async(upload_backend, cur, ...)` 上传到显存（第 1727 行），'
    '紧接着 `ggml_backend_event_record` 记下这次上传（第 1729 行）。'
    '下一轮复用同一块缓冲之前，先 `ggml_backend_event_synchronize`（第 1706 行）。\n\n'
    '这段代码把"pinned + async + events"三个能力位为什么必须**同时**具备讲清楚了：'
    'pinned 提供 DMA 源，async 提供搬运通道，events 提供"这块缓冲什么时候能再用"的判据。'
    '少了任何一个，加载器就退回同步读加同步 `set` 的路径。',
    src=M, parts=[(1699, 1735)], lang='c')

L.section(
    '十三、后端层的便捷函数：不用知道 buft 也能分配',
    '最后看四个后端级入口：`ggml_backend_get_default_buffer_type`、`ggml_backend_alloc_buffer`、'
    '`ggml_backend_get_alignment`、`ggml_backend_get_max_size`。'
    '它们全部走同一条路：`backend -> device -> 默认 buft -> 对应的 buft 函数`。\n\n'
    '`get_alignment` 与 `get_max_size` 的真正消费者是分配器：L4-01 的 `ggml-alloc.c:1168-1230` 用 '
    '`alignment` 把每个节点的尺寸补齐、用 `max_size` 判断"这一块还装得下吗，装不下就换一块 buffer"。'
    '这两个值正是本课第二节里 `get_alignment`（必需）与 `get_max_size`（可选）的用途。',
    src=C, parts=[(248, 263)], lang='c')

L.footnote_add('本课逐字引用 `ggml/src/ggml-backend.cpp` 与 `ggml/include/ggml-backend.h` 两个文件，'
               '两者都计入覆盖率。')
L.footnote_add('第 8 幕与第十一、十二节另逐字引用 `src/llama-model-loader.cpp`（pinned host buffer 的'
               '使用方证据，1530-1588 与 1699-1735 行），同样计入覆盖率；'
               '该文件的主线（mmap 与权重落位）是 L2-03 的主题，本课不重复。')
L.footnote_add('文中提到的 `ggml-alloc.c` 行号（1168-1230）属于 L4-01 的覆盖范围，'
               '本课不引用其源码，故不计入本课覆盖率。')

L.prereqs('`L4-02`')

L.goal(
    '说出 `ggml_backend_buft_*` 七个公共函数各自的用途，以及哪三个是必需的、'
    '另外三个不实现时框架给什么默认值（对应本课主干）；',
    '解释 `ggml_backend_buft_alloc_buffer` 与 `ggml_backend_buffer_init` 如何把 buft 与 buffer 串起来，'
    '以及为什么 buffer 的属性查询全部要转发给 buft；',
    '说出 host buffer 与 device buffer 的分工：`ggml_backend_tensor_copy` 如何用 `is_host` 决定拷贝方向；',
    '说出 **pinned host buffer 在什么场景下被用到**（对应验收点）：'
    '不用 mmap 加载模型时，权重从文件读进显存的那块中转内存；',
    '说明 `ggml_backend_tensor_alloc` 与 `ggml_backend_dev_buffer_from_host_ptr` 在 mmap 权重落位'
    '（L2-03）里各做什么。')

L.conclusion(
    '★ buft 是契约，buffer 是句柄',
    '| 对象 | 是什么 | 关键函数 |\n|---|---|---|\n'
    '| `ggml_backend_buffer_type_t` | "哪一类内存"：分配 + 属性 | `alloc_buffer` / `get_alignment` / '
    '`get_max_size` / `get_alloc_size` / `is_host` |\n'
    '| `ggml_backend_buffer_t` | "这一块内存"：虚表 + buft 指针 + context + size + usage | '
    '`get_base` / `set_tensor` / `get_tensor` / `cpy_tensor` |\n\n'
    'buffer 不存自己的属性 —— 四个查询函数各一行转发给 buft；'
    '唯一的回程是 `ggml_backend_buffer_get_type`。')

L.conclusion(
    '★ 三个可选项与它们的默认值',
    '`get_max_size` 缺省 `SIZE_MAX`、`get_alloc_size` 缺省 `ggml_nbytes`、`is_host` 缺省 `false`。'
    '第三条最要命：**没实现 `is_host` 不等于"是 host"**，而是被当成 device buffer —— '
    '于是所有跨 buffer 拷贝都会绕开 `set` / `get` 这条快路。'
    '`get_alloc_size` 则可以**大于** `ggml_nbytes`，所以分配路径必须用它。')

L.conclusion(
    '★ host / device 的分工只有一条判据',
    '`ggml_backend_buffer_is_host()`（转发到 buft 的 `is_host`）决定 `ggml_backend_tensor_copy` 的方向：'
    'src 在 host 就 `set`、dst 在 host 就 `get`、两边都不是才轮到后端的 `cpy_tensor`，'
    '再不行就 `malloc` 中转。host buffer 就是"CPU 能直接读写的那一侧"，'
    '这也是 L2-03 权重上传与 L4-02 跨设备副本的共同基础。')

L.conclusion(
    '★ pinned host buffer 用在非 mmap 的权重上传',
    '走 mmap 时数据本来就在页缓存里，用 `buffer_from_host_ptr` 直接包成 buffer 即可；'
    '**不走 mmap 时**，权重必须先落到一块设备认可的宿主内存里再上传 —— '
    '那就是设备的 `host_buffer_type` 分配的 pinned 内存：'
    '`caps.host_buffer` 声明它存在（ggml-backend.h:151-152），'
    '`ggml_backend_dev_host_buffer_type()` 把它取出来，'
    '模型加载器用它做 `ggml_backend_tensor_set_async` 的中转站（llama-model-loader.cpp:1554-1587、1727）。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
