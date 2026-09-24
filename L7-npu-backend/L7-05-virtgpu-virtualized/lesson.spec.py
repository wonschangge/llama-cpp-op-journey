#!/usr/bin/env python3
"""L7-05 · VirtGPU 后端（虚拟化 GPU）—— 课件 spec。

运行：python3 L7-npu-backend/L7-05-virtgpu-virtualized/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_API   = 'ggml/src/ggml-virtgpu/backend/shared/api_remoting.h'
SRC_RPC   = 'ggml/src/ggml-virtgpu/backend/shared/apir_cs_rpc.h'
SRC_FRONT = 'ggml/src/ggml-virtgpu/apir_cs_ggml-rpc-front.cpp'
SRC_BACK  = 'ggml/src/ggml-virtgpu/backend/apir_cs_ggml-rpc-back.cpp'
SRC_CMD   = 'ggml/src/ggml-virtgpu/backend/shared/apir_backend.gen.h'
SRC_DISP  = 'ggml/src/ggml-virtgpu/backend/backend-dispatched.gen.h'
SRC_IMPL  = 'ggml/src/ggml-virtgpu/virtgpu-forward-impl.h'
SRC_VGPU  = 'ggml/src/ggml-virtgpu/virtgpu.cpp'
SRC_VH    = 'ggml/src/ggml-virtgpu/virtgpu.h'
SRC_FWD   = 'ggml/src/ggml-virtgpu/virtgpu-forward-buffer.cpp'
SRC_DBUF  = 'ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer.cpp'
SRC_SHM   = 'ggml/src/ggml-virtgpu/virtgpu-shm.cpp'
SRC_HDR   = 'ggml/include/ggml-virtgpu.h'


# ------------------------------------------------------------------ 工具

def _ridx(parts, notes, line):
    """上游行号 -> 渲染下标（注解行与段间分隔行都占一个下标）。"""
    i = 0
    for pi, (a, b) in enumerate(parts):
        if pi:
            i += 1                              # 段间分隔行 //>> ---- ... ----
        for ln in range(a, b + 1):
            if ln == line:
                return i
            i += 1
            if ln in notes:
                i += 1                          # 注解行本身占位、不占行号
    raise SystemExit('_ridx: %d 不在引用区间 %s 内' % (line, parts))


def marks(vis, parts, notes, **groups):
    """把 visual 里的 @名字@ 占位符换成 U.markLines 的实参。

    ★ 下标一律由行号换算。手数渲染下标是这一课最容易错的地方：注解行会让
      其后所有行的下标整体后移；数错了不会报错，只会高亮错行。"""
    for k, lines in groups.items():
        lit = '[' + ', '.join(str(_ridx(parts, notes, l)) for l in lines) + ']'
        vis = vis.replace('@%s@' % k, lit)
    if '@' in vis:
        raise SystemExit('marks: visual 里还有未替换的占位符：%r' % vis[vis.index('@'):][:40])
    return vis


L = Lesson(
    id='L7-05',
    layer='L7 · NPU 与加速器后端',
    title='VirtGPU 后端：虚拟化 GPU',
    codecap='ggml/src/ggml-virtgpu/ 与 ggml/include/ggml-virtgpu.h（逐字引用）',
    nav={'prev': {'href': '../L7-04-executorch-edge/index.html', 'label': 'L7-04 ExecuTorch 后端'},
         'next': {'href': '../L7-06-small-backends/index.html', 'label': 'L7-06 小后端合集'}},
)

L.note('**一句话**：这是唯一一个**从根上改变数据面**的后端 —— 前面 16 个后端都假设 '
       '`tensor->data` 是一个本机可访问的地址，而这里，程序跑在客户机（VM）里，'
       'GPU 在宿主机上，**那个地址在程序所在的地址空间里根本不存在**。'
       '于是指针退化成"buffer 内偏移"，数据靠一块共享内存窗口搬过去，'
       '一切操作变成一条条命令。')
L.note('本课覆盖 `ggml/src/ggml-virtgpu/` 下 **38 个文件**加公共头 `ggml/include/ggml-virtgpu.h`，'
       '共 **39 个**（`wc -l` 合计 4414 行），按"墙的哪一侧"分四族。'
       '逐字引用 13 个，其余在 `source.md` 第十五节按族说明。')

# 计划里 L7-05 的全部 39 个文件。逐字引用的只是其中的核心（见各幕 src=），
# 其余在 source.md 第十五节按族列出；这里全部计入覆盖。
L.cover(
    'ggml/include/ggml-virtgpu.h',
    'ggml/src/ggml-virtgpu/apir_cs_ggml-rpc-front.cpp',
    'ggml/src/ggml-virtgpu/backend/apir_cs_ggml-rpc-back.cpp',
    'ggml/src/ggml-virtgpu/backend/backend-convert.h',
    'ggml/src/ggml-virtgpu/backend/backend-dispatched-backend.cpp',
    'ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer-type.cpp',
    'ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer.cpp',
    'ggml/src/ggml-virtgpu/backend/backend-dispatched-device.cpp',
    'ggml/src/ggml-virtgpu/backend/backend-dispatched.cpp',
    'ggml/src/ggml-virtgpu/backend/backend-dispatched.gen.h',
    'ggml/src/ggml-virtgpu/backend/backend-dispatched.h',
    'ggml/src/ggml-virtgpu/backend/backend-virgl-apir.h',
    'ggml/src/ggml-virtgpu/backend/backend.cpp',
    'ggml/src/ggml-virtgpu/backend/shared/api_remoting.h',
    'ggml/src/ggml-virtgpu/backend/shared/apir_backend.gen.h',
    'ggml/src/ggml-virtgpu/backend/shared/apir_backend.h',
    'ggml/src/ggml-virtgpu/backend/shared/apir_cs.h',
    'ggml/src/ggml-virtgpu/backend/shared/apir_cs_ggml.h',
    'ggml/src/ggml-virtgpu/backend/shared/apir_cs_rpc.h',
    'ggml/src/ggml-virtgpu/ggml-backend-buffer-type.cpp',
    'ggml/src/ggml-virtgpu/ggml-backend-buffer.cpp',
    'ggml/src/ggml-virtgpu/ggml-backend-device.cpp',
    'ggml/src/ggml-virtgpu/ggml-backend-reg.cpp',
    'ggml/src/ggml-virtgpu/ggml-backend.cpp',
    'ggml/src/ggml-virtgpu/ggml-remoting.h',
    'ggml/src/ggml-virtgpu/include/apir_hw.h',
    'ggml/src/ggml-virtgpu/virtgpu-apir.h',
    'ggml/src/ggml-virtgpu/virtgpu-forward-backend.cpp',
    'ggml/src/ggml-virtgpu/virtgpu-forward-buffer-type.cpp',
    'ggml/src/ggml-virtgpu/virtgpu-forward-buffer.cpp',
    'ggml/src/ggml-virtgpu/virtgpu-forward-device.cpp',
    'ggml/src/ggml-virtgpu/virtgpu-forward-impl.h',
    'ggml/src/ggml-virtgpu/virtgpu-forward.gen.h',
    'ggml/src/ggml-virtgpu/virtgpu-shm.cpp',
    'ggml/src/ggml-virtgpu/virtgpu-shm.h',
    'ggml/src/ggml-virtgpu/virtgpu-utils.cpp',
    'ggml/src/ggml-virtgpu/virtgpu-utils.h',
    'ggml/src/ggml-virtgpu/virtgpu.cpp',
    'ggml/src/ggml-virtgpu/virtgpu.h',
)

# ==================================================================== 第 1 幕

L.scene(
    kicker='L7 · NPU 与加速器后端',
    title='一台 VM 里的 llama.cpp，GPU 在墙的另一边',
    sub='39 个文件分成四族。看源码之前，先记住一句话：数据只能搬过去，不能指过去。',
    caption='这一课的第一手依据是这个协议头：它自己声明"本文件的其余部分必须与 virglrenderer 的 apir-protocol.h 一致"。',
    src=SRC_API, parts=[(3, 20)], duration=18000,
    mark_src=[3, 9, 12, 14, 17],
    notes_src={3: '这一行框定了本课的边界：协议的真相在 virglrenderer 那一侧，这里只是镜像',
               17: '23 条后端命令全部从这一条 FORWARD 走 —— 见第 4 幕'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">客户机 VM：libggml-virtgpu</span><span class="arrow">-></span>
    <span class="chip c">/dev/dri + 共享窗口</span><span class="arrow">-></span>
    <span class="chip b">宿主机：libggml-virtgpu-backend</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['族', '文件', '跑在哪', '本课怎么讲'],
  [['客户机侧 frontend', '18', 'VM 里的 ggml-virtgpu 库', '幕 2/5/6/7/9，第七/十三节'],
   ['宿主机侧 backend', '11', 'virglrenderer 加载的库', '幕 3，第六/十二节'],
   ['两侧共用协议', '6', 'backend/shared/，编译进两边', '幕 1/4，第二节'],
   ['通用工具', '4', '客户机侧：窗口 + 稀疏数组', '幕 8，第十四节']]);
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '本课覆盖 39 个文件：<span class="k">ggml/src/ggml-virtgpu/</span> 下 38 个 + 公共头 <span class="k">ggml/include/ggml-virtgpu.h</span>，wc -l 合计 4414 行。',
  '客户机侧 18 个文件：三个 ggml 虚表（device / buffer_type / buffer）+ 一层转发（virtgpu-forward-*）。',
  '宿主机侧 11 个文件：把同一批调用在真后端上重放（backend-dispatched-*）。',
  '两侧共用 6 个头：命令枚举、编码器、tensor 的线格式都在 <span class="k">backend/shared/</span>。',
  '剩下 4 个是通用工具：共享内存窗口（virtgpu-shm）与稀疏数组（virtgpu-utils）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(14000, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = '一句话：<span class="k">数据只能搬过去，不能指过去。</span> 下一幕看这句话在代码里的第一处痕迹。';
});
'''
)

# ==================================================================== 第 2 幕

S2_PARTS = [(36, 43)]
S2_NOTES = {43: '减掉的是 buffer 的 base。这不是优化：墙那边的地址空间里，这个数只有"相对"才有意义'}
S2_VIS = '''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="two" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const host = wrap.querySelector('#two');
const left = U.card({ c: 'b', t: '其它 16 个后端',
  b: '<b>data 是本机可访问的地址</b>：<br>CPU 的 malloc、CUDA 的显存、Metal 的 buffer，<br>内核拿到它就能解引用。',
  m: 'kernel(gpu_ptr, tensor->data)' }, { style: 'width:330px' });
const right = U.card({ c: 'a', t: 'ggml-virtgpu',
  b: '<b>data 过墙前要变成偏移</b>：<br>客户机持有的这个值属于宿主机的地址空间，<br>在 VM 里解引用没有意义。',
  m: 'result.data -= buffer_base' }, { style: 'width:330px' });
host.appendChild(left); host.appendChild(right);
left.style.opacity = '.35'; right.style.opacity = '.35';

const msg = wrap.querySelector('#msg');
const texts = [
  '一切从 <span class="k">apir_serialize_tensor()</span> 开始：把一个 <span class="v">ggml_tensor</span> 变成一个定长 POD。',
  '左边这条路，整个仓库走了 16 遍：<span class="v">tensor->data</span> 就是设备指针，内核直接读。',
  '<span class="k">到了这里不行。</span> 客户机里的 <span class="v">tensor->data</span> 是宿主机地址空间里的一个数值 —— 它指向的内存，客户机没有映射。',
  '于是最后一行把它减掉 buffer 的 base：<span class="v">data - base = 该 tensor 在 buffer 内的字节偏移</span>。这是唯一能过墙的形式。',
  '宿主侧再把偏移加回自己的 base（下一幕）。<span class="k">跨墙传的是"第几字节"，不是"在哪里"。</span>'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
tl.at(3400, () => { left.style.opacity = '1'; right.style.opacity = '.35'; msg.innerHTML = texts[1]; U.markLines(document, @A@); });
tl.at(7000, () => { left.style.opacity = '.35'; right.style.opacity = '1'; msg.innerHTML = texts[2]; U.markLines(document, @B@); });
tl.at(10500, () => { msg.innerHTML = texts[3]; U.markLines(document, @C@); });
tl.at(13500, () => { msg.innerHTML = texts[4]; U.markLines(document, @C@); });
'''
L.scene(
    kicker='L7-05 · 核心',
    title='★ <span class="hl-a">指针退化成偏移</span>：过墙前先减掉 buffer base',
    sub='其它后端把 tensor->data 直接交给内核去解引用；这里同一个字段必须先变成一个相对量。',
    caption='回顾 L1-01：那一课说"data 就是数据在哪"的答案。这个前提在虚拟化下第一个失效。',
    src=SRC_FRONT, parts=S2_PARTS, duration=16000,
    mark_src=[36, 39, 42], notes_src=S2_NOTES,
    visual=marks(S2_VIS, S2_PARTS, S2_NOTES,
                 A=(36,), B=(36, 39), C=(36, 39, 42))
)

# ==================================================================== 第 3 幕

S3_PARTS = [(42, 53)]
S3_NOTES = {46: '宿主自己的 base：由真后端的 buffer 给出，客户机拿不到它的映射，只能拿到这个数值',
            50: '加回来 —— 这一步之后，tensor->data 才重新是一个可解引用的设备指针',
            53: '同一条断言同时卡住下界与上界：偏移为负、或越过 buffer 末尾，都在这里被拒'}
S3_VIS = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip c">线上：偏移</span><span class="arrow">+</span>
    <span class="chip d">宿主：buffer_start</span><span class="arrow">=</span>
    <span class="chip b">真指针 tensor_data</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['第 52 / 53 行的断言', '挡住什么'],
  [['tensor_data + tensor_size >= tensor_data', '无符号加法回绕（越界的第一种伪装）'],
   ['tensor_data      >= buffer_start', '偏移为负 / 落在 buffer 之前'],
   ['tensor_data + tensor_size <= buffer_start + buffer_size', '越过 buffer 末尾']]);
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '解序列化的第一件事：把线格式里的 <span class="v">uint64_t data</span> 取出来。',
  '<span class="k">tensor_size</span> 来自 <span class="v">ggml_nbytes()</span>，<span class="k">buffer_start / buffer_size</span> 来自宿主自己的真 buffer。',
  '第 52 行先做一次溢出检查：<span class="v">a + b >= a</span> 不成立就说明回绕了。',
  '第 53 行再做一次区间检查：偏移必须落在 <span class="v">[buffer_start, buffer_start + buffer_size)</span> 内。',
  '注意校验的是<b>字节区间</b>，不是指针本身 —— 因为跨墙传过来的从来就不是指针。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, @A@); });
tl.at(3300, () => { msg.innerHTML = texts[1]; U.markLines(document, @B@); });
rows.forEach((r, i) => tl.at(6200 + i * 2900, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 2];
  U.markLines(document, @C@);
}));
tl.at(15000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; U.markLines(document, @C@); });
'''
L.scene(
    kicker='L7-05 · 核心',
    title='墙的另一边：偏移 + 宿主的 <span class="hl-d">buffer_start</span> = 真指针',
    sub='宿主机反解出指针，并且顺手做了隔着一堵墙唯一还能做的地址校验。',
    caption='这两条断言是本课里唯一"客户机说什么都不能全信"的地方：偏移越界会在宿主侧被拦下。',
    src=SRC_BACK, parts=S3_PARTS, duration=17000,
    mark_src=[42, 46, 50, 52, 53], notes_src=S3_NOTES,
    visual=marks(S3_VIS, S3_PARTS, S3_NOTES,
                 A=(42,), B=(42, 46, 50), C=(42, 46, 50, 52, 53))
)

# ==================================================================== 第 4 幕

L.scene(
    kicker='L7-05 · 命令流',
    title='<span class="hl-c">23 条命令</span>：整个后端就是一张枚举表',
    sub='ggml 的 device / buffer_type / buffer / backend 四个面，被压成 0..22 号命令。',
    caption='这个文件是生成的（来源见末幕说明）。它和宿主侧的分发表 backend-dispatched.gen.h 必须逐条对齐。',
    src=SRC_CMD, parts=[(1, 36)], duration=18000,
    mark_src=[1, 4, 10, 16, 24, 32, 35],
    notes_src={1: '协议的另一半在这里：命令号是双方的契约，改一个就要两边一起改',
               35: 'COUNT = 最后一个命令号 + 1；宿主侧用它做下标越界的上界检查'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'device · 0-9', b: '设备名、类型、显存、<br>supports_op、buffer_from_ptr', m: '10 条' },
  { c: 'b', t: 'buffer-type · 10-15', b: '名字、对齐、上限、<br>分配 buffer、算 alloc_size', m: '6 条' },
  { c: 'c', t: 'buffer · 16-21', b: '数据面：get_base、<br>set/get/cpy_tensor、clear', m: '6 条' },
  { c: 'd', t: 'backend · 22', b: '执行：<br>graph_compute', m: '1 条' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const msg = wrap.querySelector('#msg');
const texts = [
  '四个面共 <span class="k">23 条命令</span>（0..22 号）+ 一个计数常量：先按面看它们的分工。',
  '<span class="v">device</span> 组最厚：10 条。设备的每一项能力都要问一遍宿主，答案在墙那边。',
  '<span class="v">buffer-type</span> 组 6 条。第 13 号 <span class="k">is_host</span> 已废弃，但编号保留、槽位保留。',
  '<span class="v">buffer</span> 组 6 条，这一组就是数据面：<span class="k">SET_TENSOR / GET_TENSOR</span> 是全部数据搬运入口。',
  '<span class="v">backend</span> 组只有 1 条：<span class="k">GRAPH_COMPUTE</span>。整张图一次性过去，不是逐算子往返。',
  '所以"命令流"并不细碎：<span class="v">一次建 buffer、一次传数据、一次算整图</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
els.forEach((_, i) => tl.at(2900 + i * 2700, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(14800, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[5]; });
'''
)

# ==================================================================== 第 5 幕

S5_PARTS = [(378, 412)]
S5_NOTES = {383: 'thread_local + 栈数组：每个线程一个 4KiB 编码缓冲，不碰堆',
            403: '过渡期的兼容开关：老 capset 下命令号整体平移 331（见 virtgpu.h 的 VENUS_COMMAND_TYPE_LENGTH）',
            409: '第一条业务参数永远是 reply 窗口的 res_id —— 宿主靠它知道该往哪块内存写回复'}
S5_MARK = [383, 400, 406, 407, 409, 410]
S5_VIS = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="stream"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'int32 cmd_type', b: 'FORWARD / HANDSHAKE /<br>LOADLIBRARY 三选一', m: 'apir_encode_int32_t' },
  { c: 'b', t: 'int32 cmd_flags', b: '转发层的标志位；<br>业务命令这里传 0', m: 'apir_encode_int32_t' },
  { c: 'c', t: 'uint32 reply_res_id', b: '回复窗口的 res_id：<br>宿主往这块内存写结果', m: 'gpu->reply_shmem.res_id' },
  { c: 'd', t: '业务参数…', b: '各命令自己的字段，<br>按 apir_cs.h 的编码器追加', m: 'apir_encode_*' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const stream = wrap.querySelector('#stream');
stream.innerHTML = '<span class="m">字节序：</span>' +
  '<span style="color:var(--a)">[cmd_type]</span>' +
  '<span style="color:var(--b)">[cmd_flags]</span>' +
  '<span style="color:var(--c)">[reply_res_id]</span>' +
  '<span style="color:var(--d)">[arg0][arg1]…</span>';

const msg = wrap.querySelector('#msg');
const texts = [
  '一次远程调用 = 往一个 4KiB 缓冲里按顺序写字段，然后把这块缓冲交给 ioctl。',
  '<span class="v">cmd_type</span>：只有三个值。业务命令全部包在 <span class="k">FORWARD</span> 里，具体是第几号命令放在 <span class="v">cmd_flags</span>。',
  '<span class="v">cmd_flags</span>：转发层的标志位，各业务命令自己传 0。',
  '<span class="v">reply_res_id</span>：<span class="k">这是同步机制的关键</span> —— 它把"回复写哪块内存"写进了命令本身。',
  '此后每个业务参数都由 <span class="v">apir_cs.h</span> 里的编码器逐字段追加；头部固定 12 字节，从不变。',
  '所以一条命令 = <span class="v">12 字节公共头 + 各命令自己的参数</span>。下一幕看这块缓冲怎么交给内核。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
const mLines = [@L0@, @L1@, @L2@, @L3@];
els.forEach((_, i) => tl.at(3000 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
  U.markLines(document, mLines[Math.min(i, 3)]);
}));
tl.at(15200, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  msg.innerHTML = texts[5];
  U.markLines(document, @ALL@);
});
'''
L.scene(
    kicker='L7-05 · 命令流',
    title='一条命令的固定前缀：<span class="hl-b">类型 / 标志 / 回复窗口</span>',
    sub='所有命令共用同一个编码器，前三段是固定的 —— 第三条告诉宿主把回复写到哪里。',
    caption='编码缓冲区是 thread_local 的 4KiB 栈数组：命令流不为此分配堆内存。',
    src=SRC_VGPU, parts=S5_PARTS, duration=18000,
    mark_src=S5_MARK, notes_src=S5_NOTES,
    visual=marks(S5_VIS, S5_PARTS, S5_NOTES,
                 L0=(400, 406), L1=(400, 406, 407), L2=(400, 406, 407, 409, 410),
                 L3=(400, 406, 407, 409, 410), ALL=tuple(S5_MARK))
)

# ==================================================================== 第 6 幕

S6_PARTS = [(444, 470), (487, 510)]
S6_NOTES = {459: 'fence_fd = 0：不注册 fence，也不要求内核回一个 fd',
            498: '不是 CPU 自旋：睡 15 微秒再看一次，把等待让给别的线程'}
S6_MARK = [444, 459, 462, 463, 470, 490, 492, 496, 498, 505]
S6_VIS = '''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="row" id="top" style="gap:9px">
    <div class="col grow" id="fields"></div>
    <div class="col grow" id="poll"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

wrap.querySelector('#fields').innerHTML =
  '<div class="cm" style="margin-bottom:4px">drm_virtgpu_execbuffer 的同步字段</div>' +
  ['fence_fd = 0', 'num_in_syncobjs = 0', 'num_out_syncobjs = 0', 'in_syncobjs = 0', 'out_syncobjs = 0']
    .map(s => '<div class="formula" style="padding:4px 8px;font-size:10px"><span class="v">' + s + '</span></div>')
    .join('');

const poll = wrap.querySelector('#poll');
poll.innerHTML = '<div class="cm" style="margin-bottom:4px">reply 窗口的第一个 4 字节</div>' +
  '<div class="formula" style="padding:5px 8px;font-size:11px">' +
  '<span class="m">notif_value</span> = <span id="cval" style="color:var(--e)">0</span></div>' +
  '<div class="card" style="border-left-color:var(--f);margin-top:5px">' +
  '<div class="cb">读到非 0 就跳出循环。源码注释写明：<b>extract the actual return value from the notif flag</b>，<br>' +
  '<span class="cm" style="margin:0">returned_value = notif_value - 1</span></div></div>';

const cval = poll.querySelector('#cval');
const msg = wrap.querySelector('#msg');
const texts = [
  '客户机把命令准备好，然后 <span class="k">drmIoctl(DRM_IOCTL_VIRTGPU_EXECBUFFER)</span> 把命令交给内核 → hypervisor → 宿主。',
  '同步字段全为 0：<span class="v">fence_fd</span>、<span class="v">num_in_syncobjs</span>、<span class="v">num_out_syncobjs</span> 都不参与。',
  'context init 阶段是同一个选择：<span class="v">VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK</span> 的值写 0，注释写明"不要在 fence 信号时产生 drm_events"。',
  '等待发生在 <span class="k">reply_shmem.mmap_ptr</span> 的第一个 4 字节上：它是一个 <span class="v">atomic_uint</span>，用 acquire 语义加载。',
  '没到就 <span class="v">os_time_sleep(15µs)</span> 再来一次；带超时的调用（如握手）还有一个毫秒级上限。',
  '醒来后：<span class="v">returned_value = notif_value - 1</span>，回复正文紧接着那 4 字节，交给解码器继续读。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, @A@); });
tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, @B@); });
tl.at(6800, () => { msg.innerHTML = texts[2]; U.markLines(document, @B@); });
const seq = [0, 0, 0, 0, 7];
seq.forEach((v, i) => tl.at(9800 + i * 1400, () => {
  cval.textContent = String(v);
  msg.innerHTML = texts[3];
  U.markLines(document, @C@);
}));
tl.at(17200, () => { msg.innerHTML = texts[4]; U.markLines(document, @D@); });
tl.at(19800, () => { msg.innerHTML = texts[5]; U.markLines(document, @ALL@); });
'''
L.scene(
    kicker='L7-05 · 同步',
    title='提交与等待：<span class="hl-e">零个 fence</span>，一个 atomic 计数器',
    sub='execbuffer 的三个同步字段全是 0；回复靠轮询 reply 窗口的第一个 4 字节。',
    caption='这解释了为什么 ggml_backend_i 里 synchronize / event_record / event_wait 三个槽位在这个后端上全是 NULL。',
    src=SRC_VGPU, parts=S6_PARTS, duration=24000,
    mark_src=S6_MARK, notes_src=S6_NOTES,
    visual=marks(S6_VIS, S6_PARTS, S6_NOTES,
                 A=(444, 470), B=(444, 459, 462, 463, 470),
                 C=(444, 459, 462, 463, 470, 490, 492),
                 D=(444, 459, 462, 463, 470, 490, 492, 496, 498),
                 ALL=tuple(S6_MARK))
)

# ==================================================================== 第 7 幕

S7_PARTS = [(32, 57)]
S7_NOTES = {41: '小于 24MiB 走常驻窗口 data_shmem；更大的临时建一个专属 blob',
            43: '常驻窗口是所有线程共用的，用之前必须先上锁',
            53: '数据本身的搬运就在这一行：一次 memcpy，写进客户机自己 mmap 到的窗口',
            54: '命令里带的是窗口的 res_id，不是指针 —— 宿主用同一个 id 换回它那一侧的地址'}
S7_MARK = [32, 41, 43, 47, 53, 54, 57]
S7_VIS = '''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `
  <div class="row" id="steps" style="gap:7px"></div>
  <div class="row" id="side" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const steps = wrap.querySelector('#steps');
const defs = [
  { c: 'a', t: '1 编头部', b: '命令号 + buffer 句柄 + tensor' },
  { c: 'b', t: '2 挑窗口', b: '小数据走常驻窗口，<br>大数据现建 blob' },
  { c: 'c', t: '3 上锁', b: '常驻窗口是共享的' },
  { c: 'd', t: '4 搬运', b: 'memcpy 进客户机的<br> mmap 指针' },
  { c: 'e', t: '5 发命令', b: 'res_id + offset + size' },
  { c: 'f', t: '6 收尾', b: '解锁 / 销毁临时窗口' }
];
const els = defs.map(d => { const e = U.card(d, { style: 'width:109px' }); steps.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');

const side = wrap.querySelector('#side');
const g = U.card({ c: 'c', t: '客户机这一侧写下的是字节', b: '<span class="cm" style="margin:0">memcpy(shmem->mmap_ptr, data, size)</span>', m: 'virtgpu-forward-buffer.cpp:53' }, { style: 'width:336px' });
const h = U.card({ c: 'b', t: '宿主这一侧换回的是指针', b: '<span class="cm" style="margin:0">get_shmem_ptr(ctx_id, shmem_res_id)</span>', m: 'backend-dispatched-buffer.cpp:64' }, { style: 'width:336px' });
side.appendChild(g); side.appendChild(h);
g.style.opacity = '.35'; h.style.opacity = '.35';

const msg = wrap.querySelector('#msg');
const texts = [
  '对照 L4-03 的 buffer 契约：<span class="v">set_tensor(buffer, tensor, data, offset, size)</span> 的五个参数原样保留，只是实现换成了过墙。',
  '第 41 行先算大小：<span class="v">size <= data_shmem.mmap_size</span> 就直接复用常驻窗口。',
  '不复用的情况（第 49 行）现建一个 <span class="v">virtgpu_shmem</span>，用完在收尾阶段销毁。',
  '<span class="k">第 53 行是全部数据搬运</span>：一次 memcpy，目标是自己 mmap 出来的窗口地址 —— 在客户机里它是普通内存。',
  '第 54 行命令里带 <span class="v">res_id</span>；宿主侧第 64 行用同一个 id 拿回它那一侧的指针，第 71 行交给真后端写进设备内存。',
  '所以链路上没有任何一处"把设备内存映射进客户机"。<span class="k">搬的是字节，不是映射。</span>'
];
tl.at(600, () => { msg.innerHTML = texts[0]; U.markLines(document, @A@); });
tl.at(3300, () => { msg.innerHTML = texts[1]; els.forEach((e, k) => { e.style.opacity = k <= 1 ? '1' : '.28'; }); U.markLines(document, @A@); });
tl.at(6300, () => { msg.innerHTML = texts[2]; els.forEach((e, k) => { e.style.opacity = k === 2 ? '1' : '.28'; }); U.markLines(document, @B@); });
tl.at(9600, () => { msg.innerHTML = texts[3]; els.forEach((e, k) => { e.style.opacity = k === 3 ? '1' : '.28'; }); g.style.opacity = '1'; U.markLines(document, @C@); });
tl.at(13200, () => { msg.innerHTML = texts[4]; els.forEach((e, k) => { e.style.opacity = k >= 4 ? '1' : '.28'; }); h.style.opacity = '1'; U.markLines(document, @D@); });
tl.at(17800, () => { msg.innerHTML = texts[5]; els.forEach(e => { e.style.opacity = '1'; }); g.style.opacity = '1'; h.style.opacity = '1'; U.markLines(document, @ALL@); });
'''
L.scene(
    kicker='L7-05 · 数据面',
    title='★ <span class="hl-c">数据只能搬过去</span>：一次 memcpy + 一条命令',
    sub='set_tensor 的全部动作：挑窗口、上锁、把字节拷进客户机自己 mmap 到的内存、把 res_id 放进命令。',
    caption='宿主侧的对应实现是 backend-dispatched-buffer.cpp 的 backend_buffer_set_tensor：解出 res_id → get_shmem_ptr → buffer->iface.set_tensor。',
    src=SRC_FWD, parts=S7_PARTS, duration=22000,
    mark_src=S7_MARK, notes_src=S7_NOTES,
    visual=marks(S7_VIS, S7_PARTS, S7_NOTES,
                 A=(41,), B=(41, 43, 47), C=(41, 43, 47, 53),
                 D=(41, 43, 47, 53, 54), ALL=tuple(S7_MARK))
)

# ==================================================================== 第 8 幕

L.scene(
    kicker='L7-05 · 数据面',
    title='窗口是怎么造出来的：一个 <span class="hl-b">HOST3D blob</span> + 一次 mmap',
    sub='客户机向 DRM 申请一块可映射的 blob，拿到 res_id 与一个本地指针；res_id 就是这块内存在墙那边的名字。',
    caption='宿主侧声明自己需要的能力就是两个回调：get_config 与 get_shmem_ptr —— 后者按 res_id 换回宿主指针。',
    src=SRC_SHM, parts=[(75, 98)], duration=19000,
    mark_src=[76, 79, 80, 86, 93, 96],
    notes_src={79: 'HOST3D + USE_MAPPABLE：内存由宿主的 GPU 栈分配，但客户机可以映射它',
               86: '映射出来的指针落在客户机自己的地址空间里 —— 这就是"窗口"两个字的全部意思'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div id="sz" class="row" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '1 对齐', b: 'size 先对齐到<br>16384 字节', m: 'align64(size, 16384)' },
  { c: 'b', t: '2 建 blob', b: 'RESOURCE_CREATE_BLOB<br>→ res_id + gem_handle', m: 'virtgpu_ioctl_resource_create_blob' },
  { c: 'c', t: '3 导出偏移', b: 'DRM_IOCTL_VIRTGPU_MAP<br>换一个 mmap 用的 offset', m: 'virtgpu_ioctl_map' },
  { c: 'd', t: '4 映射', b: 'mmap(PROT_READ|PROT_WRITE,<br>MAP_SHARED)', m: 'void * ptr = mmap(...)' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

wrap.querySelector('#sz').innerHTML =
  '<div class="card" style="width:336px;border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">常驻数据窗口</div>' +
  '<div class="cb">SHMEM_DATA_SIZE = 0x1830000（源码注释写 24MiB）。' +
  '所有 set/get_tensor 默认都走它，所以每次用之前要上锁。</div></div>' +
  '<div class="card" style="width:336px;border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">常驻回复窗口</div>' +
  '<div class="cb">SHMEM_REPLY_SIZE = 0x4000（16KiB）。' +
  '第一个 4 字节是 atomic 计数器，之后是回复正文。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '四个步骤全部在客户机侧完成：<span class="k">建窗口这件事本身不需要宿主配合</span>。',
  '<span class="v">align64(size, 16384)</span> 先对齐，之后才建 blob。',
  '<span class="k">VIRTGPU_BLOB_MEM_HOST3D</span>：内存归宿主的 GPU 栈管；<span class="k">VIRTGPU_BLOB_FLAG_USE_MAPPABLE</span>：客户机可以映射它。',
  '<span class="v">mmap</span> 出来的指针落在客户机自己的地址空间里 —— 这就是"窗口"两个字的全部意思。',
  '第 96 行把 <span class="v">gem_handle</span> 也存下来：销毁时用 <span class="k">DRM_IOCTL_GEM_CLOSE</span> 归还。',
  '最后回到宿主侧：<span class="v">virgl_apir_callbacks</span> 只有两个函数，其中 <span class="k">get_shmem_ptr(ctx_id, res_id)</span> 就是"按名字换指针"。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
els.forEach((_, i) => tl.at(2900 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[5]; });
'''
)

# ==================================================================== 第 9 幕

L.scene(
    kicker='L7-05 · 收束',
    title='墙的两侧，各管一段',
    sub='客户机侧实现 ggml 的契约并把调用翻译成命令；宿主侧把命令翻译回 ggml 的调用。',
    caption='下一课 L7-06 是小后端合集（RPC 也是"把远端当设备"）。对照 L7-02 Hexagon：计划里那一课的要点是 host 与 DSP 的边界，而这里隔的是虚拟化边界。',
    src=SRC_HDR, parts=[(1, 14)], duration=20000,
    mark_src=[10],
    notes_src={10: '客户机侧对外暴露的全部 API 就这一行：注册一个 backend。其余全在墙后面'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['环节', '客户机侧（VM 里）', '宿主侧（hypervisor）', '源码依据'],
  [['设备能力', '缓存 name / type / memory', '向真后端查询并回传', 'ggml-backend-reg.cpp:34-58'],
   ['buffer 分配', '只发命令，拿回一个句柄', '真后端 alloc_buffer（记入跟踪表）', 'backend-dispatched-buffer-type.cpp:60-79'],
   ['数据写入', 'memcpy 进共享窗口', 'get_shmem_ptr 换指针后写设备', 'virtgpu-forward-buffer.cpp:53 / backend-dispatched-buffer.cpp:64,71'],
   ['图执行', '整图序列化后一次发走', '反序列化后交给真后端 graph_compute', 'virtgpu-forward-backend.cpp:17 / backend-dispatched-backend.cpp:56,93'],
   ['同步', '轮询 reply 窗口的 atomic', '真后端是异步的则补一次 synchronize', 'virtgpu.cpp:490 / backend-dispatched-backend.cpp:95-97'],
   ['地址', '只持有偏移与句柄', '持有真指针与真 buffer', 'apir_cs_ggml-rpc-front.cpp:42 / …-back.cpp:50']],
  { monoCols: [] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '这台 VM 里的 llama.cpp 能不能直接把宿主机 GPU 的显存映射进来，让 <span class="mono">tensor->data</span> 指过去？' +
  '不能的话，一个 tensor 的数据是怎么到宿主 GPU 的？',
  '不能。三条理由，都在代码里：<br>' +
  '① <span class="mono">apir_buffer_get_base()</span> 送回客户机的是一个 <span class="mono">uintptr_t</span>（宿主的地址），' +
  '它在 VM 的地址空间里没有映射，解引用就是野指针；<br>' +
  '② 所以 <span class="mono">tensor->data</span> 过墙前被减去 buffer base，变成 <b>buffer 内偏移</b>' +
  '（<span class="mono">apir_cs_ggml-rpc-front.cpp:42</span>），宿主再加回自己的 <span class="mono">buffer_start</span> 并断言没越界' +
  '（<span class="mono">apir_cs_ggml-rpc-back.cpp:50-53</span>）；<br>' +
  '③ 字节本身走共享内存窗口：客户机建一个 <span class="mono">HOST3D + USE_MAPPABLE</span> 的 blob 并 mmap 它' +
  '（<span class="mono">virtgpu-shm.cpp:79,86</span>），把数据 memcpy 进去（<span class="mono">virtgpu-forward-buffer.cpp:53</span>），' +
  '命令里只带 <span class="mono">res_id + offset + size</span>（<span class="mono">:54-57</span>），' +
  '宿主用 res_id 换回自己那一侧的指针（<span class="mono">backend-dispatched-buffer.cpp:64</span>）再写进真 buffer（<span class="mono">:71</span>）。<br>' +
  '一句话：<b>搬的是字节，不是映射；传的是偏移，不是指针。</b>'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '按"谁做这件事"把六条链路列出来，每一条都能在源码里指到行。',
  '<span class="k">设备能力</span>：客户机缓存一次，之后不再往返 —— 这是它少有的"不每次过墙"的地方。',
  '<span class="k">buffer 分配</span>：客户机只有一个句柄，宿主侧才真正持有 buffer 对象（还在跟踪表里）。',
  '<span class="k">数据写入</span>：客户机做的是 memcpy，宿主做的是"换指针 + 写设备"。',
  '<span class="k">图执行</span>：一次往返算整张图，命令流并不细碎。',
  '<span class="k">同步</span>：客户机这一侧只有一个 atomic 计数器；宿主只在真后端异步时补一次 synchronize。',
  '记住验收点：<span class="v">设备内存在宿主机上，客户机拿不到指针</span>，所以只能"共享内存窗口 + 命令流"。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 6)];
}));
tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、覆盖清单与四个族',
    '本课覆盖计划里 `L7-05` 的全部 **39 个**源文件：`ggml/src/ggml-virtgpu/` 下 38 个 '
    '加公共头 `ggml/include/ggml-virtgpu.h`，`wc -l` 合计 **4414 行**；'
    '后缀分布是 20 个 `.cpp` + 19 个 `.h`，没有 `.c`。按"跑在墙的哪一侧"分四族：\n\n'
    '| 族 | 文件数 | 目录 | 角色 |\n|---|---|---|---|\n'
    '| 客户机侧 frontend | 18 | `ggml/src/ggml-virtgpu/`（顶层）+ `ggml/include/ggml-virtgpu.h` | 实现 ggml 的三张虚表，把调用翻译成命令 |\n'
    '| 宿主机侧 backend | 11 | `ggml/src/ggml-virtgpu/backend/` | 把命令翻译回 ggml 调用，交给真后端 |\n'
    '| 两侧共用协议 | 6 | `ggml/src/ggml-virtgpu/backend/shared/` | 命令枚举、编解码器、tensor 线格式 |\n'
    '| 通用工具 | 4 | `ggml/src/ggml-virtgpu/virtgpu-shm.*` `virtgpu-utils.*` | 共享内存窗口与稀疏数组 |\n\n'
    '客户机侧与宿主机侧被编译成两个不同的库（`ggml-virtgpu` 与 `ggml-virtgpu-backend`）。'
    '下面这个头文件是双方的共同基准 —— 它自己声明要与 virglrenderer 的 `apir-protocol.h` 保持一致：',
    src=SRC_API, parts=[(3, 20)], lang='cpp')

L.section(
    '二、★ 过墙的 tensor：apir_rpc_tensor',
    '一个 `ggml_tensor` 里全是指针（`data` / `buffer` / `src[]` / `view_src` / `extra`），'
    '不能直接 memcpy 过去。这个 POD 就是它的线格式：所有指针字段换成 `uint64_t`，'
    '`ne[]` / `nb[]` / `op_params[]` 原样保留。\n\n'
    '注意 `nb[]` 被完整带过去了：**跨墙的 tensor 依然可以是任意步长的**，'
    '布局信息没有丢，丢的只是"地址"。',
    src=SRC_RPC, parts=[(14, 30)], lang='cpp')

L.section(
    '三、★ tensor->data 是偏移，不是指针',
    '序列化的最后一步：如果 tensor 有数据，就把 `data` 换成它相对 buffer base 的偏移。\n\n'
    '这一行是本课的核心证据。L1-01 说"`data` 回答数据在哪"；在这里，'
    '客户机里的 `tensor->data` 之所以看起来像个地址，只是因为 ggml-alloc 当初是拿'
    '宿主侧返回的 base 算出来的 —— 那个数在客户机的地址空间里没有任何映射。',
    src=SRC_FRONT, parts=[(36, 43)], lang='cpp')

L.section(
    '四、宿主机侧把偏移还原成指针',
    '反序列化时，宿主把自己的 `buffer_start` 加回去，并用两条断言把偏移限制在 buffer 内。'
    '这是隔着特权边界唯一还能做的地址校验：**校验字节区间，而不是校验指针**。',
    src=SRC_BACK, parts=[(42, 53)], lang='cpp')

L.section(
    '五、命令枚举：23 条命令 + 一个计数',
    '整个后端的"接口"就是这张枚举表：device 组 0-9（10 条）、buffer-type 组 10-15（6 条）、'
    'buffer 组 16-21（6 条）、backend 组 22（1 条），共 **23 条命令**。'
    '`APIR_BACKEND_DISPATCH_TABLE_COUNT = 23` 是最后一条命令号加一，宿主侧用它做下标越界检查。\n\n'
    '第 13 号 `BUFFER_TYPE_IS_HOST` 已废弃（宿主的实现恒返回 false），但**编号与槽位都保留**。',
    src=SRC_CMD, parts=[(1, 36)], lang='cpp')

L.section(
    '六、分发表：命令号就是数组下标',
    '宿主机侧把命令号直接当数组下标用（`backend.cpp:137` 的 `apir_backend_dispatch_table[cmd_type]`，'
    '越界由 `backend.cpp:131-135` 拦下）。表的顺序必须与枚举逐条对齐 —— 这是生成的代码，'
    '两边由同一份定义产出。',
    src=SRC_DISP, parts=[(36, 72)], lang='cpp')

L.section(
    '七、客户机侧的转发入口：一条宏',
    '23 个 `virtgpu-forward-*` 函数长得一模一样：准备编码器、写参数、发出去、读回复。'
    '所以它被压成了两个宏 —— `REMOTE_CALL_PREPARE` 与 `REMOTE_CALL`。\n\n'
    '注意 `REMOTE_CALL_PREPARE` 传的命令号是 `APIR_COMMAND_TYPE_FORWARD`，'
    '真正的业务命令号放在 `cmd_flags` 里；`REMOTE_CALL` 的 `max_wait_ms` 传 0，即不带超时。',
    src=SRC_IMPL, parts=[(11, 36)], lang='cpp')

L.section(
    '八、一条命令的固定前缀',
    '所有命令共用同一个编码器，前三段固定：`cmd_type`（int32）、`cmd_flags`（int32）、'
    '`reply_res_id`（uint32）。第三条把"回复写到哪块内存"写进了命令本身 —— '
    '这正是第九、十节讲的同步机制得以成立的原因。\n\n'
    '编码缓冲区是 `thread_local` 的 4KiB 栈数组：命令流不为此分配堆内存。',
    src=SRC_VGPU, parts=[(378, 413)], lang='cpp')

L.section(
    '九、提交：一个 execbuffer，fence 字段全为 0',
    '命令交给内核的方式是一次 `DRM_IOCTL_VIRTGPU_EXECBUFFER`。'
    '注意 `fence_fd`、`num_in_syncobjs`、`num_out_syncobjs` 全部为 0：'
    '**这个后端不使用 DRM 的 fence 或 syncobj 来同步**。'
    'context 初始化时也做了同一个选择（`virtgpu.cpp:337`，'
    '`VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK` 的值写 0，注释写明不要在 fence 信号时产生 drm_events）。',
    src=SRC_VGPU, parts=[(444, 470)], lang='cpp')

L.section(
    '十、等待：一个 atomic 计数器 + 15µs 轮询',
    '回复窗口的第一个 4 字节是一个 `atomic_uint`：宿主执行完把它改成非 0，'
    '客户机用 acquire 语义轮询它，读到非 0 就退出循环，再按源码注释所说'
    '"extract the actual return value from the notif flag"减一得到返回码。'
    '没有到就 `os_time_sleep(15)` 微秒再看。\n\n'
    '这解释了为什么 `ggml_backend_i` 里 `synchronize` / `event_record` / `event_wait` '
    '三个槽位在这个后端上全是 NULL（`ggml-backend.cpp:41,47,48`）：'
    '**每次远程调用本来就是同步的**，等待已经内建在 `remote_call()` 里了。',
    src=SRC_VGPU, parts=[(487, 510)], lang='cpp')

L.section(
    '十一、★ 数据面的全流程：memcpy + 一条命令',
    '回到验收点：一个客户机里的 tensor，数据是怎么到宿主 GPU 的？全部动作就在这个函数里。\n\n'
    '第 53 行是唯一的字节搬运点：`memcpy(shmem->mmap_ptr, data, size)` —— '
    '目标是客户机自己 mmap 出来的窗口地址，在 VM 里它是一段普通内存。'
    '第 54 行把窗口的 `res_id` 放进命令，宿主用同一个 id 换回它那一侧的指针。\n\n'
    '链路上**没有任何一处把设备内存映射进客户机**。这直接回答了本课的验收点。',
    src=SRC_FWD, parts=[(32, 57)], lang='cpp')

L.section(
    '十二、宿主侧接住数据',
    '宿主收到 `APIR_COMMAND_TYPE_BUFFER_SET_TENSOR` 后：解出 buffer 句柄、tensor、'
    '窗口 `res_id`、`offset`、`size`，用 `get_shmem_ptr()` 把 `res_id` 换成宿主机侧的指针，'
    '再调用**真后端**的 `buffer->iface.set_tensor()`。到这一步，数据才真正进入设备内存。',
    src=SRC_DBUF, parts=[(35, 74)], lang='cpp')

L.section(
    '十三、两块常驻窗口与一把锁',
    '客户机侧一共只建两块常驻共享内存：数据窗口与回复窗口。它们的尺寸、'
    '以及保护数据窗口的那把互斥锁，都写在 `struct virtgpu` 里。\n\n'
    '数据窗口 24MiB 是"够用就好"的选择：`set_tensor` 超过它就临时另建一块；'
    '回复窗口只有 16KiB，因为回复里装的是标量、字符串与句柄，不是张量数据。',
    src=SRC_VH, parts=[(50, 51), (75, 80)], lang='cpp')

L.section(
    '十四、窗口是怎么造出来的',
    '客户机侧建窗口只有四个动作：对齐、建 blob、导出 mmap 偏移、mmap。'
    '`VIRTGPU_BLOB_MEM_HOST3D` 说明内存由宿主的 GPU 栈分配；'
    '`VIRTGPU_BLOB_FLAG_USE_MAPPABLE` 说明客户机可以映射它 —— 两个标志缺一不可。\n\n'
    '宿主侧对应的能力声明只有两个回调：`get_config` 与 `get_shmem_ptr`（`backend-virgl-apir.h:16-19`）。',
    src=SRC_SHM, parts=[(75, 98)], lang='cpp')

L.section(
    '十五、其余 26 个文件按族说明',
    '本课逐字引用了上面 13 个文件；其余 26 个按族列出各自的职责。'
    '下表"行数"一列取自 `wc -l`。\n\n'
    '**客户机侧（12 个未逐字引用）**\n\n'
    '| 文件 | 行数 | 职责 |\n|---|---|---|\n'
    '| `ggml-backend-reg.cpp` | 213 | 注册表：`ggml_backend_virtgpu_reg()`；一次性把设备名/类型/显存/buffer type 全部缓存（34-58 行） |\n'
    '| `ggml-backend.cpp` | 72 | `ggml_backend_i` 虚表；`graph_compute` 转发给 `apir_backend_graph_compute`；`synchronize` / `event_*` 全为 NULL |\n'
    '| `ggml-backend-device.cpp` | 160 | `ggml_backend_device_i` 虚表；`buffer_from_host_ptr` 两条路径；`get_props` 末尾把 `caps.buffer_from_host_ptr` 强制改回 false |\n'
    '| `ggml-backend-buffer.cpp` | 123 | `ggml_backend_buffer_i` 两张表（普通 / from_ptr）；`set_tensor` 按 `is_from_ptr` 二选一：要么 memcpy，要么发命令 |\n'
    '| `ggml-backend-buffer-type.cpp` | 81 | `ggml_backend_buffer_type_i` 虚表；`alloc_buffer` 里按宿主的 `buffer_from_host_ptr` 能力选路 |\n'
    '| `ggml-remoting.h` | 71 | 两侧的上下文结构（`shared_memory` 向量、`apir_buffer_context_t`）与两个 `ggml_buffer*_to_apir_handle` 换算 |\n'
    '| `virtgpu-forward-device.cpp` | 194 | 10 条 device 命令的客户机实现；`apir_device_buffer_from_ptr` 是客户机侧另一条建窗口的分配路径 |\n'
    '| `virtgpu-forward-buffer-type.cpp` | 110 | 6 条 buffer-type 命令的客户机实现 |\n'
    '| `virtgpu-forward-backend.cpp` | 58 | 把 cgraph 序列化成一个字节缓冲放进窗口，再发一条 `GRAPH_COMPUTE` |\n'
    '| `virtgpu-forward.gen.h` | 54 | 生成物：客户机侧转发函数的原型 |\n'
    '| `virtgpu-apir.h` | 15 | `apir_buffer_context_t` 的定义（host_handle + shmem + buft_host_handle） |\n'
    '| `include/apir_hw.h` | 9 | `virgl_renderer_capset_apir` 的镜像：协议版本 + `supports_blob_resources`（公共头 `ggml/include/ggml-virtgpu.h` 在末幕逐字引用） |\n\n'
    '**宿主机侧（8 个未逐字引用）**\n\n'
    '| 文件 | 行数 | 职责 |\n|---|---|---|\n'
    '| `backend/backend.cpp` | 144 | `apir_backend_initialize` / `apir_backend_dispatcher` / `apir_backend_deinit`：`dlopen` 真后端库并按命令号查表 |\n'
    '| `backend/backend-dispatched.cpp` | 51 | 全局 `reg` / `dev` / `bck`，以及把真后端的注册函数变成一次初始化 |\n'
    '| `backend/backend-dispatched-device.cpp` | 149 | 10 条 device 命令的宿主实现；`buffer_from_ptr` 把窗口指针交给真后端 |\n'
    '| `backend/backend-dispatched-buffer-type.cpp` | 105 | 6 条 buffer-type 命令的宿主实现；`is_host` 恒返回 false；`alloc_buffer` 成功后记入跟踪表 |\n'
    '| `backend/backend-dispatched-backend.cpp` | 102 | 反序列化图、按需检查 `supports_op`、调真后端 `graph_compute`、异步后端补一次 `synchronize` |\n'
    '| `backend/backend-dispatched.h` | 27 | `virgl_apir_context` 与 `backend_dispatch_t` 函数指针类型 |\n'
    '| `backend/backend-convert.h` | 13 | 宿主机侧的 `ggml_buffer_to_apir_handle`：句柄就是指针本身 |\n'
    '| `backend/backend-virgl-apir.h` | 32 | 宿主库对 virglrenderer 的两个回调声明与三个导出函数 |\n\n'
    '**两侧共用（3 个未逐字引用：`api_remoting.h` / `apir_cs_rpc.h` / `apir_backend.gen.h` 已在前文逐字引用）**\n\n'
    '| 文件 | 行数 | 职责 |\n|---|---|---|\n'
    '| `backend/shared/apir_cs.h` | 378 | 编解码器：encoder / decoder、fatal 标志、按类型读写，全部是 header-only 的 inline 函数 |\n'
    '| `backend/shared/apir_cs_ggml.h` | 232 | ggml 特有的编解码：tensor 的两种编码（线格式 / 内联结构体）、buffer 句柄、cgraph |\n'
    '| `backend/shared/apir_backend.h` | 50 | 初始化返回码与两个句柄类型 `apir_buffer_type_host_handle_t` / `apir_buffer_host_handle_t` |\n\n'
    '**通用工具（3 个未逐字引用：`virtgpu-shm.cpp` 在第十四节逐字引用）**\n\n'
    '| 文件 | 行数 | 职责 |\n|---|---|---|\n'
    '| `virtgpu-shm.h` | 23 | `struct virtgpu_shmem`：res_id / mmap_size / mmap_ptr / gem_handle 四个字段 |\n'
    '| `virtgpu-utils.cpp` | 179 | `util_sparse_array`：可按 id 索引的稀疏数组，节点用 CAS 无锁挂接 |\n'
    '| `virtgpu-utils.h` | 86 | 对齐宏、`os_time_sleep`、计时器 —— 第十节的 15µs 轮询就来自这里 |')

L.footnote_add('本课声明 39 个源文件 = 计划里 L7-05 的全部：`ggml/src/ggml-virtgpu/` 下 38 个 '
               '加 `ggml/include/ggml-virtgpu.h`，**无一遗漏**。其中 13 个被逐字引用：'
               '核心 8 个（`virtgpu.cpp`、`virtgpu.h`、`virtgpu-shm.cpp`、`virtgpu-forward-buffer.cpp`、'
               '`backend/backend-dispatched-buffer.cpp`、`apir_cs_ggml-rpc-front.cpp`、'
               '`backend/apir_cs_ggml-rpc-back.cpp`、`virtgpu-forward-impl.h`），'
               '协议与公共头 5 个（`backend/shared/api_remoting.h`、`backend/shared/apir_cs_rpc.h`、'
               '`backend/shared/apir_backend.gen.h`、`backend/backend-dispatched.gen.h`、'
               '`ggml/include/ggml-virtgpu.h`）；其余 26 个在第十五节按族说明。')
L.footnote_add('**覆盖域外、不计入覆盖率**的文件（本课只在文字里提及，后缀不在本视角的覆盖域内，'
               '故不进覆盖声明）：`ggml/src/ggml-virtgpu/ggmlremoting_functions.yaml`（166 行，'
               '远程函数的 YAML 定义）、`ggml/src/ggml-virtgpu/regenerate_remoting.py`（333 行，'
               '由 YAML 生成三个 `*.gen.h`）、`ggml/src/ggml-virtgpu/CMakeLists.txt` 与 '
               '`ggml/src/ggml-virtgpu/backend/CMakeLists.txt`（两个库的构建定义）。')
L.footnote_add('宿主机侧的库并不在本仓库内被链接成一个可运行程序：它是被 virglrenderer 用 '
               '`dlopen` 加载的（`backend/backend.cpp:76`），加载路径来自 hypervisor 的配置。'
               '本课不实测这条链路，只做源码级断言。')
L.footnote_add('本课不讨论任何命令行参数；参数门禁的真值集来自真实二进制的帮助输出。')

L.prereqs('`L7-04`（ExecuTorch 后端）；建议先看 `L1-01`（张量的数据面）、'
          '`L3-01`（后端契约与虚表）、`L4-03`（buffer 与 buffer type）')

L.goal(
    '说出一台客户机里的 tensor，数据是怎么到宿主 GPU 的（对应验收点）：'
    '哪一行做 memcpy、命令里带的是什么、宿主怎么换回指针；',
    '解释虚拟化下为什么不能直接把设备内存映射进客户机，以及 `tensor->data` 在这里变成了什么；',
    '列出 APIR 的 23 条命令与四个分组，并说出一次远程调用的固定前缀是哪三个字段；',
    '说出这个后端用什么做同步，以及为什么它在 `ggml_backend_i` 上没有 synchronize / event 接口；',
    '指出 39 个文件分别在墙的哪一侧（客户机 18 / 宿主 11 / 共用 6 / 工具 4）。')

L.conclusion(
    '★ 虚拟化拿掉了"指针"这个前提',
    '其它 16 个后端都建立在同一个前提上：`tensor->data` 是一个本机可访问的地址。'
    '在 ggml-virtgpu 里这个前提不成立 —— 设备内存在宿主机上，客户机里没有它的映射。\n\n'
    '| 面 | 其它后端 | ggml-virtgpu |\n|---|---|---|\n'
    '| `tensor->data` | 设备指针，内核直接解引用 | **相对 buffer base 的偏移** |\n'
    '| 数据搬运 | 内核自己读 | 客户机 memcpy 进共享窗口，宿主再写设备 |\n'
    '| 地址校验 | 内核侧按指针校验 | 校验**字节区间**是否落在 buffer 内 |\n\n'
    '这是唯一一个从根上改变数据面的后端：**搬的是字节，不是映射；传的是偏移，不是指针。**'
    '（L1-01 说 `data` 回答"数据在哪"；这里它只能回答"第几字节"。）')

L.conclusion(
    '数据面 = 共享内存窗口 + 命令流',
    '一条 `set_tensor` 的完整路径：\n\n'
    '```text\n'
    '客户机 ggml_backend_remoting_buffer_set_tensor  (ggml-backend-buffer.cpp:16)\n'
    '  -> apir_buffer_set_tensor                     (virtgpu-forward-buffer.cpp:22)\n'
    '       选窗口 -> mtx_lock -> memcpy(mmap_ptr, data, size)   (:41-53)\n'
    '       命令里放 res_id + offset + size                     (:54-57)\n'
    '  -> remote_call -> DRM_IOCTL_VIRTGPU_EXECBUFFER           (virtgpu.cpp:470)\n'
    '     ... hypervisor ...\n'
    '宿主 backend_buffer_set_tensor                  (backend-dispatched-buffer.cpp:35)\n'
    '       get_shmem_ptr(ctx_id, res_id)                       (:64)\n'
    '       buffer->iface.set_tensor(...)  -> 真后端 -> 设备内存  (:71)\n'
    '```\n\n'
    '对照 L3-01：客户机侧实现的仍然是同一套 `ggml_backend_*_i` 契约；'
    '对照 L4-03：`buffer` 这个概念没有变，变的只是"buffer 在哪、谁能碰它"。\n\n'
    '再对照计划里的 L7-02（Hexagon）：那一课的要点是 host 侧与 DSP 侧的边界，'
    '"跨"发生在同一台机器内部；这里跨的是虚拟化特权边界，'
    '所以客户机连设备内存的地址都拿不到，只剩偏移与句柄。')

L.conclusion(
    '同步 = 一个 atomic 计数器',
    '`execbuffer` 的 `fence_fd`、`num_in_syncobjs`、`num_out_syncobjs` 全为 0，'
    'context 初始化也显式关掉了 fence 事件。回复靠轮询共享回复窗口的第一个 4 字节'
    '（`atomic_uint` + acquire 语义），没到就睡 15µs 再看。\n\n'
    '所以这个后端在 `ggml_backend_i` 上不需要 `synchronize` / `event_record` / `event_wait`：'
    '**每次远程调用本来就是同步的**。这与 L4-04 讲的异步后端是两种相反的取舍。')

L.conclusion(
    '可序列化是这一课的隐含约束',
    '跨墙的 tensor 必须能被压成定长 POD `apir_rpc_tensor`；带 `extra` 的 tensor '
    '在 `apir_encode_ggml_tensor_inline()` 里会直接 abort（`apir_cs_ggml.h:177-179`）。'
    '但 `nb[]` 是原样带过去的，所以**步长自由的 tensor 依然可以过墙** —— '
    '被拿掉的是地址，不是布局。整张计算图也走同一条路：序列化成一个字节缓冲，'
    '一次 `GRAPH_COMPUTE` 送过去。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
