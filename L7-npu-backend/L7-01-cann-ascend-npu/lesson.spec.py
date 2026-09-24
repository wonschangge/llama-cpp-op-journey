#!/usr/bin/env python3
"""L7-01 · CANN 后端（Ascend NPU）—— 课件 spec。

运行：python3 L7-npu-backend/L7-01-cann-ascend-npu/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_MAIN = 'ggml/src/ggml-cann/ggml-cann.cpp'
SRC_OPS  = 'ggml/src/ggml-cann/aclnn_ops.cpp'
SRC_OPSH = 'ggml/src/ggml-cann/aclnn_ops.h'
SRC_TEN  = 'ggml/src/ggml-cann/acl_tensor.cpp'
SRC_TENH = 'ggml/src/ggml-cann/acl_tensor.h'
SRC_CMN  = 'ggml/src/ggml-cann/common.h'
SRC_API  = 'ggml/include/ggml-cann.h'

L = Lesson(
    id='L7-01',
    layer='L7 · NPU 与加速器后端',
    title='CANN 后端（Ascend NPU）：算子由厂商库提供',
    codecap='ggml/src/ggml-cann/ 6 文件 + ggml/include/ggml-cann.h（逐字引用）',
    nav={'prev': {'href': '../../L6-gpu-backend/L6-09-metal-kernels/index.html',
                  'label': 'L6-09 Metal 内核'},
         'next': {'href': '../L7-02-hexagon-qualcomm/index.html',
                  'label': 'L7-02 Hexagon 后端'}},
)

L.note('**一句话**：CANN 后端不是一个"自己写 kernel 的后端"，而是一个**翻译层** —— '
       '它把 ggml 图上的每个 op 翻译成华为 CANN 算子库（aclnn）里一个**预编译算子**的调用。'
       '整课 10058 行源码里，没有一行是矩阵乘的内核代码。')
L.note('对比 L6-01：CUDA 后端的 `mul_mat` 是手写的 CUDA kernel（`.cu` 文件里的 `__global__` 函数）；'
       'CANN 后端的 `mul_mat` 是 `aclnnMm` / `aclnnBatchMatMul` / `aclnnMatmul` 三次库调用之一。'
       '**这就是 NPU 后端与 GPU 后端最大的差别，也是本课的核心。**')
L.note('这份"翻译"有代价：厂商库里没有的算子，后端只能**说不**（`supports_op` 返回 false），'
       '那部分图会留在 CPU 上 —— 这正是 L4-02 里调度器切分的输入。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L7 · NPU 与加速器后端',
    title='CANN 后端：<span class="hl-a">7 个文件</span>、10058 行、两个公共入口',
    sub='华为昇腾（Ascend）NPU 的 ggml 后端，行数为 wc -l。七个文件分工明确：一个交契约、一个做分派、一个翻译算子。',
    caption='回顾 L3-01：每个后端都要交三张函数指针表。本课看 CANN 交了什么、又是怎么把一个 op 变成库调用的。',
    src=SRC_API, parts=[(35, 49)], duration=16000,
    mark_src=[35, 37, 49],
    notes_src={35: '最多 16 张卡',
               37: '入口一：注册（L3-02/L3-03 讲注册表怎么发现它）',
               49: '入口二：按设备号创建一个后端实例'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '行数', '里面是什么'],
  [['ggml-cann.cpp', '3070', '三张虚表 + 算子的分派 switch + 内存池实现'],
   ['aclnn_ops.cpp', '4479', '每个 op 的具体实现：调哪个 aclnn 算子'],
   ['aclnn_ops.h', '1191', '上面那些函数的声明 + 调用宏 GGML_CANN_CALL_ACLNN_OP'],
   ['common.h', '651', 'context：内存池与 stream'],
   ['acl_tensor.h', '349', 'ggml_tensor -> aclTensor 的封装与所有权'],
   ['acl_tensor.cpp', '195', '上面的实现 + ggml_type -> aclDataType 映射'],
   ['ggml-cann.h', '123', '公共 API：只有 2 个入口（本幕左侧）']],
  { monoCols: [0, 1] });
wrap.querySelector('#tbl').appendChild(t.el);

const rows = t.body.querySelectorAll('tr');
const msg = wrap.querySelector('#msg');
const texts = [
  '七个文件，按"离 ggml 多近"排：越往上越是 ggml 的约定，越往下越是华为的库。',
  '公共头 <span class="v">ggml-cann.h</span> 只有 <span class="k">123 行</span>：一个注册函数 + 一个创建函数。' +
  'L3-01 说过：<span class="k">公共 API 是给人调的，虚表是给后端实现的。</span>',
  '<span class="v">ggml-cann.cpp</span> 最厚（3070 行）：三张虚表、分派 switch、三种内存池都在这。',
  '<span class="v">aclnn_ops.cpp</span> 更厚（4479 行）：但它几乎没有"算法"，全是"怎么把 ggml 的参数喂给 aclnn"。',
  '记住这条分工，后面八幕都在这张表里转：<span class="k">契约 -> 分派 -> 翻译 -> 落到 ACL 运行时。</span>'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2300 + i * 2500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(14600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L7-01 · 契约',
    title='三张虚表：CANN 交给 ggml 的<span class="hl-b">全部约定</span>',
    sub='L3-01 拆过的三张表，这里是 CANN 的具体填法 —— 每张表就是一个 static const 变量。',
    caption='同一文件里还有 buffer_i（1469 行）与 reg_i（2985 行），属于 L3-02/L4-03 的范围，本幕不展开。',
    src=SRC_MAIN, parts=[(2751, 2768)], duration=18000,
    mark_src=[2751, 2764, 2768],
    notes_src={2751: '计算层：16 个函数指针，本文件 2751-2768',
               2764: 'graph_compute 是"跑一整张图"的唯一入口，下一幕从它进去',
               2768: '三张表都是 static const —— 编译期就有，没有构造逻辑'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '计算层 · ggml_backend_i', b: '16 个函数指针。<br>真正跑图的是 graph_compute。' },
  { c: 'b', t: '内存层 · ggml_backend_buffer_type_i', b: '6 个函数指针。<br>alloc_buffer 决定 NPU 显存怎么要。' },
  { c: 'c', t: '设备层 · ggml_backend_device_i', b: '15 个函数指针。<br>supports_op 决定"这个 op 我接不接"。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const t = U.table(
  ['虚表', 'CANN 的行号', '本课关心哪个指针'],
  [['ggml_backend_i', '2751-2768', 'graph_compute（2764）—— 分派的入口'],
   ['ggml_backend_buffer_type_i', '1597-1604', 'alloc_buffer（1599）—— 内存从哪来'],
   ['ggml_backend_device_i', '2939-2955', 'supports_op（2949）—— 覆盖度的边界']],
  { monoCols: [0, 1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const texts = [
  '三张表，三个问题：<span class="k">怎么算、内存从哪来、这个 op 我接不接。</span>',
  '<span class="v">ggml_backend_i</span>：16 个指针，必需的只有 3 个；CANN 填了 10 个，其余是 NULL（L3-01 讲过）。',
  '<span class="v">ggml_backend_buffer_type_i</span>：只有 6 个指针。<span class="k">alloc_buffer 把所有"要显存"的请求收进一个地方</span> —— 第 7 幕看它。',
  '<span class="v">ggml_backend_device_i</span>：15 个指针。其中 <span class="k">supports_op</span> 是整课后半段的主角：' +
  '它说"不"的 op，调度器就得切给别的后端。',
  '三张表都是 <span class="v">static const</span>，没有虚函数、没有继承 —— ' +
  '这就是 L3-01 说的"<span class="k">C 风格的多态</span>"。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(11200, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
tl.at(14500, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L7-01 · 核心',
    title='★ 算子不是内核，是<span class="hl-a">一次厂商库调用</span>',
    sub='CANN 后端不写 kernel。它只做两件事：问库"要多少 workspace"，然后"下发执行"。',
    caption='对比 L6-01：CUDA 的 mul_mat 是手写 kernel；这里 mul_mat 最终落到 aclnnMm（见第 5 幕的映射表）。',
    src=SRC_OPSH, parts=[(922, 934)], duration=20000,
    mark_src=[922, 927, 930, 933],
    notes_src={922: '这一个宏，就是 CANN 后端与厂商库之间的全部接口',
               927: '第一步：只问"要多少 workspace" —— 还没执行。注意 ## 把算子名拼进函数名',
               930: 'workspace 从 context 的内存池里要（第 7 幕）',
               933: '第二步：真正下发。最后一个参数是 CTX.stream()（第 7 幕）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">ggml op</span><span class="arrow">-></span>
    <span class="chip b">aclnn&lt;名字&gt;GetWorkspaceSize</span><span class="arrow">-></span>
    <span class="chip c">workspace</span><span class="arrow">-></span>
    <span class="chip d">aclnn&lt;名字&gt;(...)</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['你以为后端要做的事', 'CANN 后端实际做的事'],
  [['写矩阵乘的 tiling / 双缓冲', '调 aclnnMm，把两个 aclTensor 和一个 executor 传进去'],
   ['写 softmax 的数值稳定实现', '调 aclnnSoftmax，dim 作为参数传进去'],
   ['为每个 op 申请临时显存', 'workspaceSize 由库算出来，后端照数申请'],
   ['自己排 kernel 的启动顺序', '全部下发到同一条 stream，按队列顺序执行']],
  { monoCols: [] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '整个"后端"被压缩成 <span class="k">两段式</span>：先问尺寸，再执行。' +
  '这是 CANN 算子库（aclnn）的统一调用约定。',
  '<span class="v">aclnn##OP_NAME##GetWorkspaceSize(...)</span>：算子的"试算"阶段，返回 workspace 大小与一个 executor。' +
  '<span class="k">这一步还不算。</span>',
  '<span class="v">ggml_cann_pool_alloc workspace_allocator(CTX.pool(), workspaceSize)</span>：' +
  'workspace 也是从内存池要的，<span class="k">不是 malloc</span>。',
  '<span class="v">aclnn##OP_NAME(workspaceAddr, workspaceSize, executor, CTX.stream())</span>：' +
  '真正下发。<span class="k">异步</span> —— 返回时算子可能还没跑完。',
  '所以"写一个 CANN 算子"= 把 ggml 张量转成 aclTensor + 按库的签名填参数。' +
  '<span class="k">算法的好坏由华为的库决定，不由 llama.cpp 决定。</span>'
];
rows.forEach((r, i) => tl.at(800 + i * 3600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i];
}));
tl.at(15200, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L7-01 · 分派',
    title='从 <span class="hl-c">dst-&gt;op</span> 到分派函数：一个 switch 的三种形态',
    sub='graph_compute 拿到一张图，逐个节点走 compute_forward。整个后端的"路由表"就是这一个 switch。',
    caption='回顾 L1-02：enum ggml_op 是算子的身份。这里正是按身份分派 —— 身份决定走哪条翻译路径。',
    src=SRC_MAIN, parts=[(1773, 1818)], duration=21000,
    mark_src=[1774, 1789, 1798, 1804, 1806, 1814, 1817],
    notes_src={1774: '唯一的判断依据：dst->op',
               1789: '形态一：把 aclnn 函数当模板参数传进去（二元算子）',
               1798: 'GGML_OP_MUL 走的是同一个模板，只换了 aclnn 函数',
               1804: 'UNARY 是二级分派：先看是不是 GGML_OP_UNARY，再看是哪个一元算子',
               1806: '形态二：宏 —— GGML_CANN_CALL_OP_UNARY(Abs) 展开成 aclnnAbs',
               1814: '同一个宏换个名字就是另一个算子：aclnnGelu'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '形态一 · 模板参数', b: '二元算子复用一套代码：<br>只有 aclnn 函数不同。' },
  { c: 'b', t: '形态二 · 宏拼接', b: '一元 / GLU 算子数量多、<br>签名一致，用宏省掉样板。' },
  { c: 'c', t: '形态三 · 专用函数', b: '矩阵乘、注意力、RoPE 这些<br>要改形状的，各写一个包装。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const t = U.table(
  ['形态', '源码里的样子', 'ggml-cann.cpp 行'],
  [['模板参数', 'ggml_cann_binary_op<aclnn_add>(ctx, dst)', '1789'],
   ['模板参数', 'ggml_cann_binary_op<aclnn_mul>(ctx, dst)', '1798'],
   ['宏', 'GGML_CANN_CALL_OP_UNARY(Abs)', '1806'],
   ['宏', 'GGML_CANN_CALL_OP_UNARY(Gelu)', '1814'],
   ['专用函数', 'ggml_cann_mul_mat(ctx, dst)', '1919'],
   ['专用函数', 'ggml_cann_flash_attn_ext(ctx, dst)', '2002']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">switch (dst->op)</span>：70 多个 case，每个 case 只做一件事 —— 把 dst 交给一个分派函数。',
  '<span class="k">形态一</span>：签名同构的二元算子，aclnn 函数名直接当模板参数，' +
  '共用 <span class="v">ggml_cann_binary_op</span>。',
  '<span class="k">形态二</span>：一元算子走两层 switch（先 op 后 unary_op），再由宏把名字拼进 aclnn 调用。',
  '<span class="k">形态三</span>：矩阵乘、注意力这类要重排形状的，各有一个 <span class="v">ggml_cann_*</span> 专用包装。',
  '三种形态的共同点：<span class="k">case 里没有算法，只有路由。</span>' +
  '真正的翻译在 aclnn_ops.cpp —— 下一幕看它的结果。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(11500, () => { els.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = texts[3]; });
tl.at(15000, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L7-01 · 核心',
    title='★ <span class="hl-c">ggml op -&gt; ACL 算子</span>的映射清单',
    sub='下表每一行的行号都能去源码核对：左边是 ggml 的算子身份，右边是华为库里那个算子的名字。',
    caption='分派行在 ggml-cann.cpp 的 ggml_cann_compute_forward；aclnn 行在 aclnn_ops.cpp。两者合起来就是"验收点"的答案。',
    src=SRC_OPS, parts=[(2218, 2248)], duration=30000,
    mark_src=[2226, 2228, 2230, 2236, 2239, 2245],
    notes_src={2226: '权重是否转成 NPU 偏好的 FRACTAL_NZ 排布，由环境变量决定',
               2234: '同一个 GGML_OP_MUL_MAT，按维度数落到三个不同的 aclnn 算子'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['ggml op', '分派行', 'ACL 算子（aclnn*）', 'aclnn_ops.cpp'],
  [['GGML_OP_MUL_MAT', '1918', 'aclnnMm / aclnnBatchMatMul / aclnnMatmul', '2236/2239/2245'],
   ['GGML_OP_MUL_MAT（量化权重）', '1918', 'aclnnWeightQuantBatchMatmulV2', '2357'],
   ['GGML_OP_MUL_MAT_ID', '1921', 'aclnnIndexSelect + aclnnBatchMatMul', '3644 / 3661'],
   ['GGML_OP_FLASH_ATTN_EXT', '2001', 'aclnnFusedInferAttentionScoreV2', '4123'],
   ['GGML_OP_ROPE', '1959', 'aclnnRotaryPositionEmbedding', '3191'],
   ['GGML_OP_SOFT_MAX', '1956', 'aclnnSoftmax', '1932'],
   ['GGML_OP_RMS_NORM', '1915', 'aclnnRmsNorm', '1347'],
   ['GGML_OP_NORM', '1885', 'aclnnLayerNorm', '583'],
   ['GGML_OP_ADD / SUB', '1787 / 1791', 'aclnnAdd / aclnnSub', '364 / 374'],
   ['GGML_OP_MUL / DIV', '1797 / 1800', 'aclnnMul / aclnnDiv', '382 / 390'],
   ['GGML_OP_GET_ROWS', '1778', 'aclnnGatherV2', '2003'],
   ['GGML_OP_CONV_TRANSPOSE_1D', '1986', 'aclnnConvolution', '3442'],
   ['GGML_OP_DIAG_MASK_INF', '1953', 'aclnnInplaceTriu + aclnnTril + aclnnInplaceAdd', '1370-1372']],
  { monoCols: [0, 1, 2, 3] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="k">13 行，覆盖了 llama.cpp 推理热路径上的绝大多数算子。</span>读法：左一列是 ggml 的身份，右一列是库里的名字。',
  '<span class="v">GGML_OP_MUL_MAT</span> 一个 op 对三个 aclnn 算子：<span class="k">按维度数选</span>（2 维用 Mm，3 维用 BatchMatMul，更高用 Matmul）。',
  '量化权重走完全不同的算子：<span class="v">aclnnWeightQuantBatchMatmulV2</span> —— ' +
  '<span class="k">反量化在库里做</span>，后端只负责把 Q4_0/Q8_0 的块喂进去。',
  '<span class="v">GGML_OP_FLASH_ATTN_EXT</span> 映射到一个融合算子 <span class="v">aclnnFusedInferAttentionScoreV2</span>：' +
  '整个注意力在库里一次算完。',
  '<span class="v">GGML_OP_DIAG_MASK_INF</span> 一个 op 拆成三次库调用：InplaceTriu 造 mask、Tril 裁剪、InplaceAdd 加上去。' +
  '<span class="k">一个 ggml op 不一定对应一个 ACL 算子。</span>',
  '这三条合起来就是验收点：<span class="k">CANN 后端把一个 ggml op 映射到 ACL 算子，靠的是 ggml-cann.cpp 的 switch + aclnn_ops.cpp 里的 aclnn 调用。</span>'
];
rows.forEach((r, i) => tl.at(600 + i * 2200, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i, 3)];
}));
tl.at(16800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
tl.at(21000, () => { msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L7-01 · 数据面',
    title='ggml_tensor -&gt; <span class="hl-d">aclTensor</span>：两处约定差异',
    sub='ggml 的 nb 以字节计、ne[0] 在最前；CANN 的 stride 以元素计、最内层维在最后。转换函数把这两处抹平。',
    caption='回想 L1-01：nb[0] = ggml_type_size(type)，而这里要除以 ggml_element_size —— L1-01 的约定在这里第二次被用到。',
    src=SRC_TEN, parts=[(53, 92)], duration=19000,
    mark_src=[53, 63, 65, 67, 80, 86, 87, 89],
    notes_src={67: '差异一：ggml 的 nb 是【字节】，acl 的 stride 是【元素个数】—— 要除',
               80: '差异二：acl 要一个"存储长度"，按 ne 和 stride 自己算出来',
               86: '差异三：ggml 的 ne[0] 在最前，CANN 的最内层维在最后 —— 整个数组要 reverse',
               89: '转换的终点：aclCreateTensor，把 ggml 的 data 指针原样交出去（不拷贝）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '步长单位', b: 'ggml: nb[i] 是<b>字节</b><br>CANN: stride[i] 是<b>元素个数</b>' },
  { c: 'b', t: '维度顺序', b: 'ggml: ne[0] 最内层<br>CANN: 最后一维最内层' },
  { c: 'd', t: '结果 · 零拷贝', b: 'tensor-&gt;data 原样交给<br>aclCreateTensor，<b>不复制数据</b>' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:220px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');

const t = U.table(
  ['转换步骤', '源码', '行'],
  [['取形状与步长', 'acl_ne[i] = tensor->ne[i]', '65'],
   ['字节 -> 元素', 'acl_stride[i] = tensor->nb[i] / ggml_element_size(tensor)', '67'],
   ['算存储长度', 'acl_storage_len += (acl_ne[i] - 1) * acl_stride[i]', '80'],
   ['翻转维度', 'std::reverse(acl_ne, acl_ne + final_dims)', '86'],
   ['交给 ACL', 'aclCreateTensor(..., tensor->data)', '89']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '这个函数是所有 aclnn 调用的<span class="k">前置条件</span>：每个 ggml 张量先变成 aclTensor，才谈得上喂给库。',
  '<span class="v">ne</span> 与 <span class="v">nb</span> 两个参数不是必需的：不传就按张量自己的形状走（第 63 行的分支）。' +
  '<span class="k">传了就是广播用的客户形状</span> —— 广播的活也在这里干。',
  '<span class="v">std::reverse</span> 那两行是整幕的关键：' +
  '<span class="k">ggml 的行主序与 CANN 的维度约定正好相反</span>，不翻转结果就是错的。',
  '<span class="v">acl_storage_len</span> 用 ne 与 stride 现算：CANN 要据此判断这块内存够不够，' +
  '而不是相信调用者。',
  '注意最后一行：<span class="v">tensor-&gt;data</span> 原样传入。<span class="k">转换只造了一个 aclTensor 描述符，没有拷贝数据。</span>'
];
defs.forEach((_, i) => tl.at(700 + i * 3800, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i];
}));
tl.at(15900, () => {
  els.forEach(e => { e.style.opacity = '1'; });
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[4];
});
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L7-01 · 内存与 stream',
    title='context 里的两样东西：<span class="hl-e">内存池</span>与 <span class="hl-f">stream</span>',
    sub='一个 NPU 设备一个 context。8 条 stream 按需创建，一个内存池按设备能力选实现 —— 两者都是懒创建的。',
    caption='对比 L4-01：CPU 侧靠 ggml-alloc 的图分配器；NPU 侧在其之上多一层"按大小复用的设备池"。',
    src=SRC_CMN, parts=[(612, 648)], duration=20000,
    mark_src=[612, 613, 618, 627, 630, 637, 643, 645],
    notes_src={613: '懒创建：streams[] 初始全是 nullptr，第一次用到才建',
               618: '真正的 ACL 调用：aclrtCreateStream',
               627: '不带参数的 stream() 就是 stream(0) —— 默认流',
               630: 'TODO 注释直说：每个 stream 本该有自己的池，现在还是共用一个',
               645: '池也是懒创建的，第一次调用 pool() 才 new'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

wrap.querySelector('#left').innerHTML =
  '<div class="cm" style="margin-bottom:4px">stream：8 条，懒创建</div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">GGML_CANN_MAX_STREAMS = 8</div>' +
  '<div class="cb">common.h 第 48 行。数组本体在 <span class="cm" style="margin:0">streams[8]</span>（第 575 行）。<br>' +
  '<span class="cm" style="margin:0">stream(i)</span> 第一次调用才 aclrtCreateStream。</div></div>' +
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">为什么可以懒创建</div>' +
  '<div class="cb">因为 aclnn 算子是<b>异步下发</b>的：第 3 幕那个宏的最后一个参数就是 ' +
  '<span class="cm" style="margin:0">CTX.stream()</span>。</div></div>';

wrap.querySelector('#right').innerHTML =
  '<div class="cm" style="margin-bottom:4px">pool：三种实现，按设备选</div>' +
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">new_pool_for_device（ggml-cann.cpp:758）</div>' +
  '<div class="cb"><span class="cm" style="margin:0">prio</span> -&gt; ggml_cann_pool_buf_prio（200）<br>' +
  'VMM 设备 -&gt; ggml_cann_pool_vmm（592）<br>否则 -&gt; ggml_cann_pool_buf（390）</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">基类只有两个纯虚函数</div>' +
  '<div class="cb">common.h 第 112-137 行：<span class="cm" style="margin:0">alloc(size, &amp;actual_size)</span> 与 ' +
  '<span class="cm" style="margin:0">free(ptr, size)</span>。RAII 包装是 ggml_cann_pool_alloc。</div></div>';

const t = U.table(
  ['纯虚函数', '谁在调它', '为什么不能直接 aclrtMalloc'],
  [['alloc(size, &actual_size)', '第 3 幕的 workspace、算子内部的临时缓冲', 'NPU 分配一次很贵，按 size 复用能省下反复申请']],
  { monoCols: [0] });
wrap.appendChild(t.el);

const msg = wrap.querySelector('#msg');
const texts = [
  'context 管两样东西：<span class="k">活干在哪条队列上（stream）、中间结果放哪（pool）。</span>',
  '<span class="v">stream()</span> 是懒创建的：<span class="k">第一次调用才 aclrtCreateStream</span>，之后一直复用。',
  '<span class="v">pool()</span> 也是懒创建的，但选哪种实现由设备能力决定 —— VMM 设备用虚拟内存池，否则用固定 256 槽的 buffer 池。',
  '两者共用一个前提：<span class="k">aclnn 算子是异步的</span>。所以 common.h 第 133 行的注释专门警告：' +
  '"释放内存前要确认算子已经跑完"。',
  '一句话：<span class="v">stream 决定顺序，pool 决定地址</span> —— 这正是异步 NPU 后端必须自己管的两件事。'
];
const el = wrap.querySelector('#msg');
tl.at(800, () => { el.innerHTML = texts[0]; });
tl.at(4200, () => { el.innerHTML = texts[1]; });
tl.at(7800, () => { el.innerHTML = texts[2]; });
tl.at(11400, () => { el.innerHTML = texts[3]; });
tl.at(15200, () => { el.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L7-01 · 覆盖度',
    title='★ <span class="hl-a">supports_op</span>：厂商库没有的算子，后端只能说不',
    sub='这张 switch 判断的不是"能不能算"，而是"ACL 里有没有对应的算子、这个数据类型库收不收"。',
    caption='L4-02 讲调度器怎么按 supports_op 把图切给不同后端 —— 这里看到切分的输入从哪来。',
    src=SRC_MAIN, parts=[(2440, 2460)], duration=20000,
    mark_src=[2440, 2442, 2444, 2446, 2449, 2451, 2452, 2456, 2457, 2458],
    notes_src={2440: '第一个问题：这个 op 是什么（对应哪个 aclnn 算子）',
               2442: '第二个问题：权重是什么类型 —— 决定调哪个 aclnn 算子',
               2449: 'Q8_0 / Q4_0 能接，但 310P 上例外，所以整个 case 被 #ifdef 掉',
               2456: '量化类型还有额外条件：必须连续 —— 否则 aclnn 的假设不成立'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:4px">支持的权重类型（GGML_OP_MUL_MAT）</div>';
[
  { l: 'F16 / F32', n: 2, c: 'b', d: 'aclnnMm / BatchMatMul / Matmul' },
  { l: 'BF16', n: 1, c: 'c', d: '非 310P 才有（2443-2445 行）' },
  { l: 'Q8_0 / Q4_0', n: 2, c: 'e', d: '要连续，且 310P 上不支持' }
].forEach(g => {
  const b = U.bar(g.l + '  (' + g.n + ')', g.c);
  left.appendChild(b.el);
  b.fill.style.width = (g.n * 26) + '%';
  b.val.textContent = g.n;
});

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">代价：覆盖度是借来的</div>' +
  '<div class="cb">能支持的 op 集合 = ACL 算子库的子集。' +
  '库升级，后端覆盖面才跟着涨；<b>llama.cpp 自己不改一行也无法多支持一个 op</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--g)">' +
  '<div class="ct" style="color:var(--g)">后果：图会被切开</div>' +
  '<div class="cb">返回 false 的节点不能卸载到 NPU。<br>' +
  '<span class="cm" style="margin:0">ggml_backend_cann_offload_op</span>（2881 行）还加了一条：' +
  '<span class="cm" style="margin:0">ne[1]</span> 小于阈值的小算子也不卸载。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '同一份 <span class="v">dist</span>：<span class="k">支持 = aclnn 有这个算子 + 这个类型库收</span>。',
  '<span class="v">F16 / F32 / BF16</span> 直接 true —— 对应的 aclnn 算子在，且没有额外约束。',
  '<span class="v">Q4_0 / Q8_0</span> 多一条 <span class="v">ggml_is_contiguous</span>：<span class="k">库的假设变成了后端的门槛。</span>',
  '这一点与 L6-01（CUDA 自己写 kernel）正好相反：<span class="k">CUDA 能自己补上一个 op，CANN 不能。</span>',
  '所以 L4-02 的切分必须存在：<span class="v">supports_op</span> 说不的节点留在 CPU，' +
  'NPU 只跑它能跑的那部分 —— <span class="k">这不是设计缺陷，是厂商库后端的必然形态。</span>'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(4300, () => { msg.innerHTML = texts[1]; });
tl.at(7900, () => { msg.innerHTML = texts[2]; });
tl.at(11500, () => { msg.innerHTML = texts[3]; });
tl.at(15100, () => { msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L7-01 · 收束',
    title='把这一课压成一张表',
    sub='一个 op 进 NPU 要走五步：分派 -> 转张量 -> 填参数 -> 问 workspace -> 下发到 stream。',
    caption='下一课 L7-02 是另一种加速器后端：Hexagon，它连 ACL 这样的统一算子库都没有。',
    src=SRC_MAIN, parts=[(2028, 2033)], duration=20000,
    mark_src=[2028, 2029, 2032],
    notes_src={2029: '分派表里没有的 op，直接返回 false（真正的把关在 supports_op）',
               2032: '有对应 aclnn 算子，就算"已下发" —— 注意它是异步的，这里并没有等待'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['步骤', 'CANN 的代码', '位置'],
  [['1 分派', 'switch (dst->op) -> ggml_cann_xxx', 'ggml-cann.cpp:1773'],
   ['2 转张量', 'ggml_cann_create_tensor -> aclCreateTensor', 'acl_tensor.cpp:53'],
   ['3 填参数', 'aclnn 算子的签名（如 dim / eps / alpha）', 'aclnn_ops.cpp:4479 行里'],
   ['4 问 workspace', 'aclnn<名>GetWorkspaceSize + CTX.pool()', 'aclnn_ops.h:927'],
   ['5 下发', 'aclnn<名>(..., CTX.stream())', 'aclnn_ops.h:933'],
   ['把关', 'supports_op 说不的，根本不进这条链', 'ggml-cann.cpp:2949']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '一个 <span class="mono">GGML_OP_SOFT_MAX</span> 节点进了 CANN 后端。' +
  '请说出它从"图上节点"走到"NPU 上执行"的三步，并给出每一步所在的文件与行号。',
  '1. <b>分派</b>：<span class="mono">ggml_cann_compute_forward</span> 的 ' +
  '<span class="mono">case GGML_OP_SOFT_MAX:</span>（ggml-cann.cpp:1956）调到 ' +
  '<span class="mono">ggml_cann_softmax(ctx, dst)</span>（同文件 1957 行）。<br>' +
  '2. <b>转张量</b>：<span class="mono">ggml_cann_softmax</span>（aclnn_ops.cpp:1935）用 ' +
  '<span class="mono">ggml_cann_create_tensor(src0)</span>（1939 行）把 ggml 张量变成 aclTensor，' +
  '转换本体在 acl_tensor.cpp:53。<br>' +
  '3. <b>下发</b>：<span class="mono">aclnn_softmax</span>（aclnn_ops.cpp:1931）里的 ' +
  '<span class="mono">GGML_CANN_CALL_ACLNN_OP(ctx, Softmax, ...)</span>（1932 行）；' +
  '宏展开成 <span class="mono">aclnnSoftmaxGetWorkspaceSize</span>（aclnn_ops.h:927）与 ' +
  '<span class="mono">aclnnSoftmax(..., CTX.stream())</span>（933 行）。<br>' +
  '结论：<b>llama.cpp 里没有 softmax 的 kernel 代码，只有一次 Huawei 库调用。</b>'));

const msg = wrap.querySelector('#msg');
const texts = [
  '六行，五个步骤加一道把关。每个 aclnn 算子都走同一条链。',
  '<span class="k">分派</span>按 L1-02 的 op 身份走；<span class="k">把关</span>按 L4-02 的切分走。',
  '<span class="k">转张量</span>抹平 ggml 与 CANN 的约定差异（步长单位、维度顺序）；这一步零拷贝。',
  '<span class="k">问 workspace + 下发</span>是 aclnn 的两段式约定，最后落到 <span class="v">CTX.stream()</span>。',
  '一句话：<span class="v">CANN 后端 = ggml 身份到华为算子库的翻译层</span>。' +
  '它自己不算，只负责把算得对的事情交给对的人。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2200 + i * 2400, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 4)];
}));
tl.at(17000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、公共 API：两个入口',
    '`ggml-cann.h` 只有 123 行。它对外只暴露两件事：**注册自己**（让 L3-02 的注册表发现它）'
    '和**按设备号创建一个后端实例**。其余全是设备信息查询。',
    src=SRC_API, parts=[(35, 49)], lang='c')

L.section(
    '二、★ aclnn 的两段式调用：整个后端的接口只有这一个宏',
    '这段注释值得逐字读：宏"在指定 stream 上提交一个异步任务"，workspace 来自分配器，'
    '而分配器可以立刻"释放"这块内存 —— 因为**同一条 stream 上的任务按队列顺序执行**，'
    '后面的任务不会提前读到它。这就是 CANN 后端把"内存池 + stream"绑在一起讲的原因。',
    src=SRC_OPSH, parts=[(904, 934)], lang='c')

L.section(
    '三、算子名字来自厂商头文件',
    '`aclnn_ops.cpp` 开头是一长串 `#include <aclnnop/aclnn_*.h>` —— **每个 aclnn 算子一个头文件**。'
    '这份 include 清单就是"这个后端能支持哪些 op"的第一手证据。',
    src=SRC_OPS, parts=[(29, 40)], lang='c')

L.section(
    '四、分派的另外两种形态',
    '第 4 幕引了二元算子的模板形态。一元算子走的是宏形态（两层 switch：先 `GGML_OP_UNARY`，'
    '再 `ggml_get_unary_op`）；矩阵乘、注意力这类大算子走专用函数形态。',
    src=SRC_MAIN, parts=[(1803, 1826)], lang='c')

L.section(
    '五、专用函数形态：MUL_MAT 与 FLASH_ATTN_EXT',
    '第 4 幕的动画代码只引到 `GGML_OP_UNARY` 为止，第三种形态（专用函数）在这里补齐。'
    '`GGML_OP_MUL_MAT` 的 case 只有两行：调用 `ggml_cann_mul_mat`。'
    '所有关于维度、量化类型、内存排布的判断都在 `aclnn_ops.cpp` 里 —— **分派层保持极薄**。',
    src=SRC_MAIN, parts=[(1915, 1923), (2001, 2003)], lang='c')

L.section(
    '六、ggml_cann_create_tensor：两处约定差异 + 存储长度',
    '转换函数要处理两处约定差异：步长单位（字节 -> 元素）与维度顺序（reverse）；'
    '它还有两个可选参数 `ne` / `nb`：不传就按张量自己的形状，传了就是**客户形状**（广播用）。',
    src=SRC_TEN, parts=[(53, 92)], lang='c')

L.section(
    '七、aclTensor 的所有权：acl_deleter',
    '所有 ACL 对象都被 `std::unique_ptr` 包着，删除函数以**模板参数**的形式传进去 —— '
    '所以 `acl_tensor_ptr` / `acl_int_array_ptr` / `acl_scalar_ptr` / `acl_tensor_list_ptr` 四个别名'
    '共用同一个 deleter 模板。这保证了 aclnn 调用里抛异常也不会漏掉 `aclDestroyTensor`。',
    src=SRC_TENH, parts=[(45, 59)], lang='c')

L.section(
    '八、内存池：一个基类，三种实现',
    '基类只有两个纯虚函数；具体实现由 `new_pool_for_device` 按设备能力选：'
    '环境变量指定 `prio` 用优先队列池，设备支持 VMM 用虚拟内存池，否则用固定 256 槽的 buffer 池。'
    '无论哪种，最终都落到 `aclrtMalloc`。',
    src=SRC_MAIN, parts=[(758, 772)], lang='c')

L.section(
    '九、stream：8 条，懒创建',
    '`GGML_CANN_MAX_STREAMS` 是 8，数组初值全是 `nullptr`，第一次用到才 `aclrtCreateStream`。'
    'context 析构时逐条销毁。',
    src=SRC_CMN, parts=[(46, 48)], lang='c')

L.section(
    '十、补充：context 的成员与析构',
    '`streams[]` 数组本体在这里声明；析构函数负责销毁事件与全部 stream。',
    src=SRC_CMN, parts=[(572, 605)], lang='c')

L.section(
    '十一、完整映射清单（含第 5 幕未列出的部分）',
    '下面这张表把 `ggml_cann_compute_forward` 的 case 与 `aclnn_ops.cpp` 里的 '
    '`GGML_CANN_CALL_ACLNN_OP` 调用逐条对上。行号来自本课引用的同一份源码。\n\n'
    '| ggml op | 分派行（ggml-cann.cpp） | ACL 算子 | 调用行（aclnn_ops.cpp） |\n'
    '|---|---|---|---|\n'
    '| `GGML_OP_MUL_MAT` | 1918 | `aclnnMm` | 2236 |\n'
    '| `GGML_OP_MUL_MAT`（3 维） | 1918 | `aclnnBatchMatMul` | 2239 |\n'
    '| `GGML_OP_MUL_MAT`（4 维以上） | 1918 | `aclnnMatmul` | 2245 |\n'
    '| `GGML_OP_MUL_MAT`（Q4_0/Q8_0） | 1918 | `aclnnWeightQuantBatchMatmulV2` | 2357 |\n'
    '| `GGML_OP_MUL_MAT_ID` | 1921 | `aclnnIndexSelect` + `aclnnBatchMatMul` | 3644 / 3661 |\n'
    '| `GGML_OP_FLASH_ATTN_EXT` | 2001 | `aclnnFusedInferAttentionScoreV2` | 4123 |\n'
    '| `GGML_OP_ROPE` | 1959 | `aclnnRotaryPositionEmbedding` | 3191 |\n'
    '| `GGML_OP_SOFT_MAX` | 1956 | `aclnnSoftmax` | 1932 |\n'
    '| `GGML_OP_RMS_NORM` | 1915 | `aclnnRmsNorm` | 1347 |\n'
    '| `GGML_OP_NORM` | 1885 | `aclnnLayerNorm` | 583 |\n'
    '| `GGML_OP_GROUP_NORM` | 1888 | `aclnnGroupNorm` | 721 |\n'
    '| `GGML_OP_L2_NORM` | 1891 | `aclnnNorm` + `aclnnClampMin` + `aclnnDiv` | 614 / 623 / 627 |\n'
    '| `GGML_OP_ADD` | 1787 | `aclnnAdd` / `aclnnInplaceAdd` | 364 / 366 |\n'
    '| `GGML_OP_SUB` | 1791 | `aclnnSub` | 374 |\n'
    '| `GGML_OP_MUL` | 1797 | `aclnnMul` / `aclnnInplaceMul` | 382 / 384 |\n'
    '| `GGML_OP_DIV` | 1800 | `aclnnDiv` | 390 |\n'
    '| `GGML_OP_SCALE` | 1924 | `aclnnMuls` | 554 |\n'
    '| `GGML_OP_CLAMP` | 1935 | `aclnnClamp` | 540 |\n'
    '| `GGML_OP_GET_ROWS` | 1778 | `aclnnGatherV2` | 2003 |\n'
    '| `GGML_OP_SET_ROWS` | 1781 | `aclnnInplaceIndexCopy` / `aclnnRepeatInterleaveIntWithDim` | 2126 / 2186 |\n'
    '| `GGML_OP_REPEAT` | 1775 | `aclnnRepeat` | 324 |\n'
    '| `GGML_OP_CPY` | 1938 | `aclnnInplaceCopy` / `aclnnCast` | 1118 / 344 |\n'
    '| `GGML_OP_CONCAT` | 1897 | `aclnnCat` | 458 |\n'
    '| `GGML_OP_IM2COL` | 1962 | `aclnnIm2col` | 1544 |\n'
    '| `GGML_OP_CONV_TRANSPOSE_1D` | 1986 | `aclnnConvolution` | 3442 |\n'
    '| `GGML_OP_SSM_CONV` | 2010 | `aclnnConvolution` | 4350 |\n'
    '| `GGML_OP_POOL_2D` | 1965 | `aclnnAvgPool2d` / `aclnnMaxPool` | 1024 / 1087 |\n'
    '| `GGML_OP_UPSCALE` | 1900 | `aclnnUpsampleNearest2d` | 930 |\n'
    '| `GGML_OP_PAD` | 1903 | `aclnnConstantPadNd` | 955 |\n'
    '| `GGML_OP_PAD_REFLECT_1D` | 1995 | `aclnnReflectionPad1d` | 3531 |\n'
    '| `GGML_OP_SUM` / `GGML_OP_SUM_ROWS` | 1968 / 1971 | `aclnnReduceSum` | 801 |\n'
    '| `GGML_OP_MEAN` | 1992 | `aclnnMean` | 3508 |\n'
    '| `GGML_OP_CUMSUM` | 2013 | `aclnnCumsum` | 822 |\n'
    '| `GGML_OP_ARGSORT` | 1974 | `aclnnArgsort` + `aclnnCast` | 567 / 569 |\n'
    '| `GGML_OP_ARGMAX` | 1977 | `aclnnArgMax` | 3328 |\n'
    '| `GGML_OP_COUNT_EQUAL` | 1998 | `aclnnEqTensor` + `aclnnReduceSum` | 3553 / 3559 |\n'
    '| `GGML_OP_TRI` | 2016 | `aclnnTril` / `aclnnTriu` | 903 / 907 |\n'
    '| `GGML_OP_DIAG_MASK_INF` | 1953 | `aclnnInplaceTriu` + `aclnnTril` + `aclnnInplaceAdd` | 1370-1372 |\n'
    '| `GGML_OP_SOLVE_TRI` | 2025 | `aclnnTriangularSolve` | 842 |\n'
    '| `GGML_OP_LEAKY_RELU` | 1912 | `aclnnLeakyRelu` | 441 |\n'
    '| `GGML_OP_ELU`（UNARY） | 1845 | `aclnnElu` | 3495 |\n'
    '| `GGML_OP_STEP`（UNARY） | 1851 | `aclnnGtScalar` | 3573 |\n'
    '| `GGML_OP_SOFTPLUS`（UNARY） | 1854 | `aclnnSoftplus` | 3587 |\n'
    '| `GGML_OP_CROSS_ENTROPY_LOSS` | 1894 | `aclnnSoftmaxCrossEntropyWithLogits` + `aclnnReduceSum` + `aclnnMuls` | 665 / 684 / 692 |\n'
    '| `GGML_OP_FILL` | 2019 | `aclnnInplaceFillScalar` | 1250 |\n'
    '| `GGML_OP_ARANGE` | 1906 | `aclnnArange` | 507 |\n'
    '| `GGML_OP_QUANTIZE`（走 CPY） | 1938 | `aclnnCast` | 344 |\n'
    '| `GGML_OP_RESHAPE` / `VIEW` / `PERMUTE` / `TRANSPOSE` | 1948-1951 | 无 —— 只改描述符，不产生计算 | — |\n\n'
    '最后一行是这一课的一个反面注脚：**视图类算子不需要任何 ACL 算子**，'
    '因为它们在 ggml 层面就已经是"换个读法"（见 L1-01）。')

L.section(
    '十二、不支持的 op：supports_op 的兜底',
    '`supports_op` 的最后两行是整个后端覆盖度的边界：switch 没列到的 op，一律 `false`。'
    '返回 false 的节点由 L4-02 的调度器切给别的后端。',
    src=SRC_MAIN, parts=[(2705, 2706)], lang='c')

L.footnote_add('本课引用 7 个文件，全部计入覆盖率：`ggml/include/ggml-cann.h`、'
               '`ggml/src/ggml-cann/ggml-cann.cpp`、`ggml/src/ggml-cann/aclnn_ops.cpp`、'
               '`ggml/src/ggml-cann/aclnn_ops.h`、`ggml/src/ggml-cann/acl_tensor.cpp`、'
               '`ggml/src/ggml-cann/acl_tensor.h`、`ggml/src/ggml-cann/common.h`。')
L.footnote_add('本课**没有**在 Ascend NPU 上实测运行（本机无昇腾设备）。所有断言都来自源码：'
               '函数名、行号、映射关系均可逐字核对；"异步执行""库决定算法"这类结论来自源码注释与调用形态，'
               '未在真机上验证。')
L.footnote_add('第 11 节的大表里，`GGML_OP_CPY` 与 `GGML_OP_QUANTIZE` 两行都指向 `aclnnCast`（344 行）：'
               '`aclnn_cast` 是 `ggml_cann_cpy` 内部的类型转换路径，`GGML_OP_QUANTIZE` 本身没有独立 case。')

L.prereqs('`L6-09`')

L.goal(
    '说出 CANN 后端把一个 ggml op 映射到 ACL 算子的**完整链路**（分派 -> 转张量 -> 填参数 -> '
    '问 workspace -> 下发到 stream）（对应验收点）；',
    '解释 `GGML_CANN_CALL_ACLNN_OP` 宏为什么是"两段式"，以及 workspace 从哪来；',
    '说明 `ggml_cann_create_tensor` 抹平了 ggml 与 CANN 之间的哪两处约定差异，并说出它为什么要重算存储长度；',
    '判断一个 op 会不会被卸载到 NPU，并说出 `supports_op` 返回 false 的后果（呼应 L4-02）。')

L.conclusion(
    '★ 算子由厂商库提供',
    'CANN 后端 10058 行源码里没有矩阵乘、没有 softmax 的 kernel 实现。'
    '每个 op 最终都变成 `aclnn<名字>GetWorkspaceSize` + `aclnn<名字>(..., stream)` 两次调用，'
    '宏在 `aclnn_ops.h:922-934`。\n\n'
    '| 对比项 | L6-01 CUDA 后端 | 本课 CANN 后端 |\n|---|---|---|\n'
    '| 算子实现在哪 | 本仓库的 `.cu` 文件 | 华为 CANN 库（预编译） |\n'
    '| 加一个新算子 | 自己写 kernel | 等库提供，或退回 CPU |\n'
    '| 覆盖度上界 | 作者愿意写多少 | ACL 算子库有多少 |\n'
    '| 后端代码的职责 | 算法 + 调度 | 只有调度与翻译 |')

L.conclusion(
    '五步链路（验收点的答案）',
    '```text\n'
    '1 分派      switch (dst->op) -> ggml_cann_xxx     ggml-cann.cpp:1773\n'
    '2 转张量    ggml_cann_create_tensor -> aclCreateTensor   acl_tensor.cpp:53\n'
    '3 填参数    aclnn 算子签名（dim / eps / alpha / ...）    aclnn_ops.cpp\n'
    '4 问尺寸    aclnn<名>GetWorkspaceSize + CTX.pool()       aclnn_ops.h:927\n'
    '5 下发      aclnn<名>(..., CTX.stream())                aclnn_ops.h:933\n'
    '把关        supports_op == false 的 op 不进这条链        ggml-cann.cpp:2949\n'
    '```\n\n'
    '第 1 步用的是 L1-02 的算子身份；第 6 行用的是 L4-02 的切分判据。')

L.conclusion(
    '覆盖度是借来的',
    '`ggml_backend_cann_supports_op`（ggml-cann.cpp:2403）的 switch 里，'
    '每个 case 都要同时回答"ACL 有没有这个算子"和"这个数据类型库收不收"。'
    '兜底的 `default: return false;`（2705-2706 行）划出了后端的能力边界 —— '
    '**边界不在 llama.cpp 手里，在厂商库手里**。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
