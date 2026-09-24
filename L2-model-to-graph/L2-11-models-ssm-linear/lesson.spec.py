#!/usr/bin/env python3
"""L2-11 · 模型家族（二）：状态空间与线性注意力 —— 课件 spec。

运行：python3 L2-model-to-graph/L2-11-models-ssm-linear/lesson.spec.py

本课覆盖的 30 个文件来自【静态扫描】分组，不是内容判定：
`python3 tools/plan_matrix.py --files | awk -F'\\t' '$1=="L2-11"'` 给出的 30 个
`src/models/*.cpp`，其图构建代码里命中 `build_ssm|ssm_conv|ssm_scan|gated_delta|build_rwkv|wkv`
之一。逐文件核对后：22 个文件名副其实，8 个文件（deepseek2 / deepseek32 / dots3note /
glm-dsa / hy-v4 / minicpm3 / plm / dflash）只是因为 MLA 的低秩 KV 压缩投影叫
`wkv_a_mqa` / `wkv_b` 而被 `wkv` 子串命中 —— 它们走 KV cache，与线性注意力无关。
source.md 第一节逐文件给出真实主题。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

# ---------------------------------------------------------------- 覆盖域内（src/）
# 本课负责的 30 个文件 —— 全部在 src/models/ 下，全部计入覆盖率。
OWNED = [
    'src/models/arwkv7.cpp',
    'src/models/bailingmoe3.cpp',
    'src/models/deepseek2.cpp',
    'src/models/deepseek32.cpp',
    'src/models/delta-net-base.cpp',
    'src/models/dflash.cpp',
    'src/models/dots3note.cpp',
    'src/models/falcon-h1.cpp',
    'src/models/glm-dsa.cpp',
    'src/models/granite-hybrid.cpp',
    'src/models/hy-v4.cpp',
    'src/models/jamba.cpp',
    'src/models/kimi-k3.cpp',
    'src/models/kimi-linear.cpp',
    'src/models/lfm2.cpp',
    'src/models/mamba2.cpp',
    'src/models/mamba-base.cpp',
    'src/models/mamba.cpp',
    'src/models/minicpm3.cpp',
    'src/models/nemotron-h.cpp',
    'src/models/plamo2.cpp',
    'src/models/plm.cpp',
    'src/models/qwen35.cpp',
    'src/models/qwen35moe.cpp',
    'src/models/qwen3next.cpp',
    'src/models/rwkv6-base.cpp',
    'src/models/rwkv6.cpp',
    'src/models/rwkv6qwen2.cpp',
    'src/models/rwkv7-base.cpp',
    'src/models/rwkv7.cpp',
]

# ---------------------------------------------------------------- 跨课引用文件
# 它们的主课在别处，本课引用是为了把"图上的调用"对到"算子契约"和"内核实现"。
SRC_MAMBA_BASE = 'src/models/mamba-base.cpp'     # 本课核心
SRC_MAMBA      = 'src/models/mamba.cpp'
SRC_RWKV6B     = 'src/models/rwkv6-base.cpp'
SRC_RWKV7B     = 'src/models/rwkv7-base.cpp'
SRC_DELTA      = 'src/models/delta-net-base.cpp'
SRC_KIMI_LIN   = 'src/models/kimi-linear.cpp'
SRC_BAILING    = 'src/models/bailingmoe3.cpp'
SRC_PLAMO2     = 'src/models/plamo2.cpp'
SRC_QWEN3NEXT  = 'src/models/qwen3next.cpp'
SRC_GRANITE    = 'src/models/granite-hybrid.cpp'
SRC_LFM2       = 'src/models/lfm2.cpp'
SRC_DS2        = 'src/models/deepseek2.cpp'
SRC_DFLASH     = 'src/models/dflash.cpp'

SRC_GGML_H     = 'ggml/include/ggml.h'           # 跨课：L1-01
SRC_GGML_C     = 'ggml/src/ggml.c'               # 跨课：L1-02 / L1-03 / L1-04 / L1-05 / L4-04
SRC_GGML_OPS   = 'ggml/src/ggml-cpu/ops.cpp'     # 跨课：L5-02
SRC_MEM_REC    = 'src/llama-memory-recurrent.h'  # 跨课：L2-04
SRC_GRAPH_H    = 'src/llama-graph.h'             # 跨课：L2-06
SRC_GRAPH_CPP  = 'src/llama-graph.cpp'           # 跨课：L2-06 / L8-01
SRC_HPARAMS    = 'src/llama-hparams.cpp'         # 跨课：L2-02

L = Lesson(
    id='L2-11',
    layer='L2 · 从模型到图',
    title='模型家族（二）：状态空间与线性注意力',
    codecap='src/models/*（逐字引用）+ ggml / llama-graph 契约对照',
    nav={'prev': {'href': '../L2-10-models-multimodal/index.html',
                  'label': 'L2-10 模型家族（一）多模态'},
         'next': {'href': '../L2-12-models-moe-a/index.html',
                  'label': 'L2-12 模型家族（三）MoE 上'}},
)

L.cover(*OWNED)

L.note('**一句话**：状态空间模型（Mamba）与线性注意力（RWKV / gated delta net）在 llama.cpp 里'
       '不是新的图引擎，而是**换了一种"记忆"**：不用随上下文增长的 KV cache，'
       '改用一份**固定大小的状态张量**，把它交给一两个**融合算子**去读写。'
       '`llm_build_mamba_base` / `llm_build_rwkv6_base` / `llm_build_rwkv7_base` / '
       '`llm_build_delta_net_base` 四个基类，就是这套记忆的图侧接口。')
L.note('回顾 L2-06：模型文件不直接调 `ggml_*`，而是调 `llama-graph` 的 `build_*` 原语。'
       '本课看到的 `build_rs` / `build_inp_mem_hybrid` 正是那一族里的"记忆"分支 —— '
       '它们**不在** `src/llama-graph.cpp` 里另起一套，而是和 KV cache 的 `build_attn` 并列。'
       '回顾 L2-04：`llama_memory_recurrent` 提供 `get_r_l()` / `get_s_l()` 两个张量，'
       '本课讲的就是**这两个张量在图上怎么被读、怎么被写回**。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L2-11 · 全局',
    title='30 个文件里，<span class="hl-a">22 个</span>真是状态空间/线性注意力，'
          '<span class="hl-e">8 个</span>只是名字里有 wkv',
    sub='分组来自静态扫描：图构建代码里命中 build_ssm / ssm_conv / ssm_scan / gated_delta / '
        'build_rwkv / wkv 之一。命中原语不等于"它就是 SSM 模型"。',
    caption='逐文件的真实主题见第 8、9 幕的表格，以及 source.md 第一节。',
    src=SRC_MAMBA, parts=[(83, 107)], duration=20000,
    mark_src=[87, 94, 103, 104, 106],
    notes_src={87: '整个 Mamba 图继承 llm_build_mamba_base —— SSM 的图原语全在那个基类里',
               94: 'build_rs_inp()：recurrent 记忆的输入占位，与 KV cache 的 build_attn_inp_kv 并列（L2-04）',
               104: 'Mamba-2 走 build_mamba2_layer',
               106: 'Mamba-1 走 build_mamba_layer'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip a">30 个 src/models/*.cpp</span><span class="arrow">-&gt;</span>
    <span class="chip b">4 个基类</span><span class="arrow">-&gt;</span>
    <span class="chip c">3 个 ggml 算子</span><span class="arrow">-&gt;</span>
    <span class="chip d">固定大小状态张量</span>
  </div>
  <div class="row" style="gap:8px">
    <div class="card" style="border-left-color:var(--b);flex:1 1 0">
      <div class="ct" style="color:var(--b)">4 个图基类（本课的主角）</div>
      <div class="cb"><span class="cm" style="margin:0">llm_build_mamba_base</span>（Mamba-1/2 及 5 个混合模型）<br>
      <span class="cm" style="margin:0">llm_build_rwkv6_base</span> / <span class="cm" style="margin:0">llm_build_rwkv7_base</span><br>
      <span class="cm" style="margin:0">llm_build_delta_net_base</span>（gated delta net / KDA）</div>
    </div>
    <div class="card" style="border-left-color:var(--c);flex:1 1 0">
      <div class="ct" style="color:var(--c)">3 个融合算子</div>
      <div class="cb"><span class="cm" style="margin:0">ggml_ssm_conv</span> — 短卷积，读写卷积状态<br>
      <span class="cm" style="margin:0">ggml_ssm_scan</span> — 选择性扫描，Mamba 的核心<br>
      <span class="cm" style="margin:0">ggml_rwkv_wkv6 / wkv7 / gated_linear_attn / gated_delta_net</span></div>
    </div>
    <div class="card" style="border-left-color:var(--e);flex:1 1 0">
      <div class="ct" style="color:var(--e)">8 个名实不符</div>
      <div class="cb"><span class="cm" style="margin:0">deepseek2 · deepseek32 · dots3note<br>glm-dsa · hy-v4 · minicpm3 · plm · dflash</span><br>
      命中的是 MLA 的 <span class="cm" style="margin:0">wkv_a_mqa</span> 子串，走 KV cache。</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const msg = wrap.querySelector('#msg');
const texts = [
  '先给全局：本课要回答的是 <span class="k">一个 SSM/线性注意力层，在 ggml 图上长什么样</span>。',
  '入口非常统一：<span class="v">llama_model_mamba::graph</span> 继承 <span class="v">llm_build_mamba_base</span>，'
    + '逐层调 <span class="v">build_mamba_layer</span> 或 <span class="v">build_mamba2_layer</span>。<br>'
    + 'RWKV 与 delta net 各自有同构的基类。',
  '算子只有少数几个：<span class="v">ggml_ssm_conv</span> 管卷积状态，<span class="v">ggml_ssm_scan</span> 管 SSM 状态，'
    + '其它家族各有一个融合算子。',
  '<span class="k">四个基类 + 三个算子 = 22 个模型文件</span>。剩下 8 个是被 <span class="v">wkv</span> 子串扫进来的，'
    + '第 9 幕会逐个说明。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(4200, () => { msg.innerHTML = texts[1]; });
tl.at(9200, () => { msg.innerHTML = texts[2]; });
tl.at(14800, () => { msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L2-11 · 核心',
    title='★ <span class="hl-a">SSM 不需要 KV cache</span>：历史被压进一份固定大小的状态',
    sub='两种记忆各占一个张量：get_r_l() 是卷积状态，get_s_l() 是 SSM 状态。'
        '它们的大小由超参决定，与 n_ctx 无关。',
    caption='对比 L2-04 第 10 幕：那一幕讲 memory 侧的接口，这一幕讲图侧怎么把它取出来用；'
            '两者的接口就是 get_r_l / get_s_l 这两个函数。',
    src=SRC_MAMBA_BASE, parts=[(14, 41)], duration=24000,
    mark_src=[16, 20, 21, 22, 26, 37, 38, 40],
    notes_src={16: 'kv_head：这一批的每个序列映射到状态缓存的哪一行 —— 就是 L2-04 里 cell/slot 的同一套机制',
               26: 'n_seqs 是这一批的序列数，不是上下文长度：state 里没有 token 维',
               37: '卷积状态：形状由 n_embd_r() 决定',
               38: 'SSM 状态：形状由 n_embd_s() 决定',
               40: 'build_rs() 把"状态缓存的若干行"取成"这一批要用的状态张量"（L2-06 的原语）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div>
  <div class="row" style="gap:8px">
    <div class="card" style="border-left-color:var(--g);flex:1 1 0">
      <div class="ct" style="color:var(--g)">KV cache：随上下文线性增长</div>
      <div class="cb">大小 ≈ <span class="cm" style="margin:0">n_ctx x n_layer x n_head_kv x head_dim</span><br>
      每来一个 token 就多一行；prompt 越长、显存越吃紧。</div>
    </div>
    <div class="card" style="border-left-color:var(--b);flex:1 1 0">
      <div class="ct" style="color:var(--b)">recurrent state：与 n_ctx 无关</div>
      <div class="cb">Mamba：<span class="cm" style="margin:0">ssm_d_state x ssm_d_inner</span><br>
      KDA：<span class="cm" style="margin:0">head_dim x head_dim x n_head</span><br>
      跑 10 个 token 和跑 10 万个 token，占的是同一块内存。</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['', '张量', '由谁给', '大小由什么决定'],
  [['卷积状态', 'get_r_l(il)', 'llama_memory_recurrent_context', 'n_embd_r()'],
   ['SSM 状态', 'get_s_l(il)', 'llama_memory_recurrent_context', 'n_embd_s()'],
   ['（对照）K/V', 'mctx->get_k / get_v', 'llama_kv_cache_context', 'n_ctx —— 随上下文增长']],
  { monoCols: [1, 3] });
t.el.style.fontSize = '9.5px';
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '先记住两句话：<span class="k">KV cache 的大小是 O(n_ctx)，recurrent state 的大小是 O(1)</span>。',
  '<span class="v">get_r_l(il)</span> 给的是<b>卷积状态</b>：只有 <span class="v">d_conv - 1</span> 步的窗口。',
  '<span class="v">get_s_l(il)</span> 给的是<b>SSM 状态</b>：整段历史的压缩表示，'
    + '形状 <span class="m">{d_state, head_dim, n_head, n_seqs+}</span> —— <span class="k">没有 token 维</span>。',
  '<span class="v">build_rs()</span> 把"缓存的若干行"变成"这一批要用的状态"。'
    + '它不复制状态本身，只是按 <span class="v">ids</span> 选行 —— 这就是 L2-04 里 slot 机制的图侧用法。',
  '所以 SSM 模型能跑很长的上下文而显存不涨；代价是<span class="k">历史被有损压缩</span>，'
    + '这正是"线性注意力"这个名字的来历。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(4500, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 0 ? 'on' : ''); });
tl.at(9000, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = i === 1 ? 'on' : ''); });
tl.at(14000, () => { msg.innerHTML = texts[3]; rows.forEach(r => r.className = ''); });
tl.at(19000, () => { msg.innerHTML = texts[4]; rows.forEach((r, i) => r.className = i === 2 ? 'on' : ''); });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L2-11 · 图原语 · 卷积',
    title='<span class="hl-b">卷积状态</span>：一个 (d_conv − 1) 长的滑动窗口',
    sub='这是整条链上唯一保留"原始 token 片段"的地方，长度是常数，不是上下文长度。',
    caption='注意 217-220 行：写回缓存用的不是"最后一个位置"，而是 n_seq_tokens - slot 处的快照 —— '
            'K 个槽位分别对应"最终状态"和"回退 s 个 token 的状态"（L2-04）。',
    src=SRC_MAMBA_BASE, parts=[(204, 238)], duration=22000,
    mark_src=[207, 217, 232, 235, 237],
    notes_src={207: 'ggml_concat：把缓存的 d_conv-1 列拼到本轮输入前面 —— 状态与输入在这里拼成一条序列',
               217: '把 conv_x 的尾部快照 cpy 回 conv_states_all（下一幕会看到同样的写法用于 SSM 状态）',
               232: 'ggml_ssm_conv：一个算子完成 1D 卷积（源码 224-231 行解释了它等价于什么）',
               235: '卷积之后接 bias 与 SiLU，与普通卷积层的区别只在状态从哪来'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip d">conv_states_all（缓存）</span><span class="arrow">-&gt;</span>
    <span class="chip a">ggml_concat</span><span class="arrow">-&gt;</span>
    <span class="chip b">ggml_ssm_conv</span><span class="arrow">-&gt;</span>
    <span class="chip c">+bias / SiLU</span><span class="arrow">-&gt;</span>
    <span class="chip d">cpy 回缓存</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'd', t: '读', m: 'build_rs(inp, conv_states_all, n_embd_r(), n_seqs)',
    b: 'build_rs 按 ids 选出这一批序列的状态行；<br>再 reshape 成 {d_conv-1, d_inner, n_seqs}。' },
  { c: 'a', t: '拼', m: 'ggml_concat(conv, transpose(xBC), 0)',
    b: '状态在上面、本轮输入在下面，沿第 0 维拼接 —— <br>于是卷积"看得到"历史窗口。' },
  { c: 'b', t: '算', m: 'ggml_ssm_conv(ctx0, conv_x, ssm_conv1d)',
    b: '一个算子做完整条 1D 卷积。<br>权重形状 [d_conv, d_inner]。' },
  { c: 'c', t: '写', m: 'ggml_cpy(conv_x 的尾部快照, conv_states_all 的槽位)',
    b: '快照偏移是 n_seq_tokens - slot，<br>所以 K 个槽位里放的是不同回退深度的状态。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '卷积状态只有 <span class="v">d_conv - 1</span> 步 —— 它是一个定长移位寄存器，不是 cache。',
  '<span class="v">ggml_concat</span> 这一步很关键：状态与输入被拼成一条连续序列，'
    + '后面的卷积才不需要"跨图状态"。',
  '<span class="v">ggml_ssm_conv</span> 是纯函数：它只吃 conv_x 与权重，'
    + '<span class="k">不碰任何缓存</span>。状态进出全部由显式的读/写节点负责。',
  '写回用 <span class="v">ggml_cpy</span> 而不是 inplace —— 这样 L4-02 的调度器才能看清依赖顺序：'
    + '<span class="k">先读旧状态、再写新状态</span>。',
  '同样的"读-算-写"三拍，下一幕在 SSM 状态上再来一次。'
];
defs.forEach((_, i) => tl.at(700 + i * 3600, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(18400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L2-11 · 核心 · SSM',
    title='★ <span class="hl-c">ggml_ssm_scan</span> 的 7 个输入：形状写在注释里',
    sub='内核开头七行就是权威的输入契约。8 个参数（含 K）分别描述状态、输入、步长、衰减与选择矩阵。',
    caption='输出张量的构造在 ggml/src/ggml.c:5739-5751（逐字见 source.md 第三节）：'
            '长度 = nelements(x) + K * s->ne[0] * s->ne[1] * s->ne[2] * ids->ne[0]。',
    src=SRC_GGML_OPS, parts=[(9773, 9796)], duration=26000,
    mark_src=[9776, 9777, 9779, 9780, 9781, 9782, 9793, 9796],
    notes_src={9776: 's：状态。第 3 维是 n_seqs+ —— 比这一批的序列数多，多出来的行是缓存里别的序列',
               9777: 'x：本轮输入（"SSM 里的 V"）。ne[1]=n_head 在 Mamba-1 里退化成 d_inner',
               9779: 'A：衰减矩阵。[d_state, n_head] 或 [1, n_head]；后者是 Mamba-2 的"每头一个标量衰减"',
               9780: 'B / C：选择矩阵，形状相同（源码 5708 行断言 ggml_are_same_shape(B, C)）',
               9782: 'ids：每个序列读状态缓存里的哪一行 —— 这就是 recurrent cache 的"slot 表"',
               9793: 'K：一个 int32 op_param，不是张量。K>1 时额外返回 K 份回退快照',
               9796: 's_off：状态在输出张量里的起始偏移 —— 输出是"y 拼上状态"的一维张量'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['参数', '形状', '在 Mamba 里是什么'],
  [['s', '{d_state, head_dim, n_head, n_seqs+}', '上一刻的状态（从缓存按 ids 选中）'],
   ['x', '{head_dim, n_head, n_seq_tokens, n_seqs}', '本轮输入，来自 ssm_conv 的输出'],
   ['dt', '{n_head, n_seq_tokens, n_seqs}', '每步的离散化步长，先 softplus 再加 bias'],
   ['A', '{d_state, n_head} 或 {1, n_head}', '衰减；Mamba-2 每头一个标量'],
   ['B', '{d_state, n_group, n_seq_tokens, n_seqs}', '选择矩阵（写）'],
   ['C', '{d_state, n_group, n_seq_tokens, n_seqs}', '选择矩阵（读）；必须与 B 同形'],
   ['ids', '{n_seqs}（I32 向量）', '每个序列对应状态缓存的哪一行'],
   ['K', 'int32 op_param（不是张量）', 'K = n_rs_seq + 1；K>1 时返回回退快照']],
  { monoCols: [0, 1] });
t.el.style.fontSize = '9px';
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '验收点的前半：<span class="k">ssm_scan 有 7 个张量输入 + 1 个整数参数 K</span>。',
  '<span class="v">s</span> 的第 3 维是 <span class="v">n_seqs+</span>，'
    + '也就是说它<b>直接看着整份缓存</b>，靠 <span class="v">ids</span> 选行。',
  '<span class="v">dt / A</span> 一起决定衰减：内核里 <span class="m">dA = expf(softplus(dt) * A)</span>。',
  '<span class="v">B / C</span> 是"选择"的全部：同一份 x，靠 B、C 决定这一刻写进状态多少、读出多少。',
  '<span class="v">K</span> 是唯一的整数参数，用来一次多要 K 份历史快照（供回退用）。',
  '<span class="k">注意：没有任何一个输入带 n_ctx。SSM 的"上下文"就是 s 这张定长张量。</span>'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(4600, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 0 || i === 6 ? 'on' : ''); });
tl.at(9600, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = i === 2 || i === 3 ? 'on' : ''); });
tl.at(14400, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 4 || i === 5 ? 'on' : ''); });
tl.at(19200, () => { msg.innerHTML = texts[4]; rows.forEach((r, i) => r.className = i === 7 ? 'on' : ''); });
tl.at(22800, () => { msg.innerHTML = texts[5]; rows.forEach(r => r.className = ''); });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L2-11 · 核心 · SSM',
    title='★ 状态更新发生在<span class="hl-a">内核的扫描循环里</span>，写回缓存是图上的一个 cpy',
    sub='输入视图 → 交给 ssm_scan（拿 ids 直接读缓存）→ 输出张量里同时含 y 与新状态 → 显式 cpy 回 ssm_states_all。',
    caption='验收点的后半：新状态由内核写在输出张量的后半段（ggml/src/ggml-cpu/ops.cpp:9819-9821 定位读写指针，'
            '9923 / 9976 处 s[i] = state），模型侧再用 274-279 行的 ggml_cpy 搬回持久缓存。',
    src=SRC_MAMBA_BASE, parts=[(240, 282)], duration=28000,
    mark_src=[243, 245, 247, 251, 253, 258, 264, 267, 276, 278],
    notes_src={243: 'x：从卷积输出里切出 {head_dim, n_head, n_seq_tokens, n_seqs}',
               245: 'B、C：同一个张量 xBC 的两个 4 维视图，偏移不同 —— 不复制数据',
               251: 'dt 先 cont 成连续张量，再加 dt_bias（ssm_scan 断言 dt 必须连续）',
               253: 'A 直接就是权重张量 ssm_a，没有额外处理',
               258: '这个 lambda 就是 build_rs 的 get_state_rows 参数 —— 它决定"怎么把状态读出来"',
               264: 'ids 直接交给 ssm_scan：内核自己按 ids 去 s 里选行，省掉一次 get_rows 拷贝',
               267: 'build_rs 返回的就是 ssm_scan 的输出张量（y 与新状态拼在一起）',
               276: '取输出张量后半段：K 份新状态，偏移 state_offset = nelements(x) * element_size',
               278: '目标视图落在 ssm_states_all 的 kv_head 行上 —— 状态就这样回到持久缓存'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" style="justify-content:center">
    <span class="chip a">build_rs 读缓存</span><span class="arrow">-&gt;</span>
    <span class="chip b">ggml_ssm_scan</span><span class="arrow">-&gt;</span>
    <span class="chip c">y ⧺ K 份新状态</span><span class="arrow">-&gt;</span>
    <span class="chip d">ggml_cpy 回 ssm_states_all</span>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['步', '图上发生什么', '行号'],
  [['① 读', 'build_rs → reshape 状态缓存 → ssm_scan 按 ids 选行', '258-267'],
   ['② 算', '内核在 scan 循环里更新 s：s[i] = state（同时写出 y[ii]）', 'ops.cpp 9923 / 9976'],
   ['③ 出', '输出张量 = [ y（前 nelements(x) 个） | K 份新状态 ]', 'ggml.c 5739-5741'],
   ['④ 写回', 'ggml_cpy(y_ssm 的后半段 → ssm_states_all 的 kv_head 行)', '274-279'],
   ['⑤ 用 y', 'y = view_4d(y_ssm, 前半段) → + D*x → swiglu → ssm_out', '281-298']],
  { monoCols: [2] });
t.el.style.fontSize = '9.5px';
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '四拍：<span class="k">读缓存 → 扫描 → 输出含新状态 → cpy 写回</span>。',
  '<span class="v">ggml_ssm_scan</span> 是纯算子：它<b>不</b>直接改 <span class="v">ssm_states_all</span>，'
    + '只把新状态写在<b>自己的输出张量</b>里。',
  '<span class="v">ids</span> 直接传进算子，是这里最巧的一处：内核用 '
    + '<span class="m">src0->data + ids[i3]*src0->nb[3]</span> 定位读地址 —— <span class="k">省掉一次 get_rows 拷贝</span>。',
  '写回用的是 <span class="v">ggml_cpy</span>，所以调度器能排出"先读后写"的顺序；'
    + '<span class="v">kv_head</span> 决定落在哪一行，与 L2-04 的 slot 是同一套编号。',
  '<span class="v">K = n_rs_seq + 1</span> 时输出里会有多份快照，分别落在缓存的相邻槽位 —— 这是投机解码回退的基础。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach((r, i) => r.className = i === 0 ? 'on' : ''); });
tl.at(5200, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 1 ? 'on' : ''); });
tl.at(10500, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
tl.at(16000, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 2 || i === 3 ? 'on' : ''); });
tl.at(22000, () => { msg.innerHTML = texts[4]; rows.forEach((r, i) => r.className = i === 4 ? 'on' : ''); });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L2-11 · 家族 · RWKV',
    title='<span class="hl-d">RWKV</span>：同一套"读-算-写"，换一个融合算子',
    sub='wkv6 与 wkv7 的差别只在算子；状态进出缓存的方式与 Mamba 完全同构。',
    caption='对比第 5 幕：Mamba 把 ids 传进算子（内核自己选行），RWKV 用 build_rs 的默认实现 '
            'ggml_get_rows 先把状态行取出来（llama-graph.h:1333）。两条路都成立，代价不同。',
    src=SRC_RWKV6B, parts=[(133, 147)], duration=22000,
    mark_src=[133, 137, 139, 141, 142, 145],
    notes_src={133: 'build_rs 不带 lambda —— 走默认的 ggml_get_rows，先取出 n_seqs 行状态',
               137: 'is_qrwkv 时用 ggml_gated_linear_attn，否则用 ggml_rwkv_wkv6；两者签名不同但语义同位',
               139: 'wkv_state 是输入，也是输出的一部分',
               141: '前 n_embd * n_tokens 个元素是这一层的输出 y',
               142: '紧随其后的是新状态：n_embd * head_size * n_seqs 个 float',
               145: '与 Mamba 第 5 幕同款的显式 cpy：把新状态搬回 mctx_cur->get_s_l(il)'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div>
  <div class="row" style="gap:8px">
    <div class="card" style="border-left-color:var(--d);flex:1 1 0">
      <div class="ct" style="color:var(--d)">输出张量的切法</div>
      <div class="cb"><span class="cm" style="margin:0">view_1d(out, n_embd*n_tokens, 0)</span> → y<br>
      <span class="cm" style="margin:0">view_1d(out, n_embd*head_size*n_seqs, n_embd*n_tokens*4)</span> → 新状态<br>
      偏移用 <b>字节</b> 表达，不是元素个数 —— 与 Mamba 那一幕的 state_offset 同一个套路。</div>
    </div>
    <div class="card" style="border-left-color:var(--b);flex:1 1 0">
      <div class="ct" style="color:var(--b)">RWKV 还多一份状态：token shift</div>
      <div class="cb">RWKV 每层把"上一个 token 的 norm 输出"也存起来，
      由 <span class="cm" style="margin:0">build_rwkv_token_shift_load / store</span> 读写（L2-06 的原语）。<br>
      它是 <span class="cm" style="margin:0">n_embd_r()</span> 那一份卷积状态在 RWKV 里的对应物。</div>
    </div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '这一层的算子', '状态行号'],
  [['src/models/rwkv6-base.cpp', 'ggml_rwkv_wkv6 或 ggml_gated_linear_attn', '137 / 139'],
   ['src/models/rwkv7-base.cpp', 'ggml_rwkv_wkv7', '107'],
   ['src/models/delta-net-base.cpp', 'ggml_gated_delta_net', '402 / 567']],
  { monoCols: [0, 1, 2] });
t.el.style.fontSize = '9.5px';
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '把 Mamba 那一幕的骨架换成 RWKV，形状完全对得上：<span class="k">读状态 → 一个融合算子 → 输出含新状态 → cpy 写回</span>。',
  '<span class="v">ggml_rwkv_wkv6</span> 与 <span class="v">ggml_rwkv_wkv7</span> 的参数不同'
    + '（wkv7 多了 r/w/k/v/a 五路），但都返回"y 拼新状态"。',
  'RWKV 的 <span class="v">build_rs</span> 没传 lambda，走的是默认 '
    + '<span class="v">ggml_get_rows</span> —— 先把状态行拷出来，再交给算子。<br>'
    + 'Mamba 则把 <span class="v">ids</span> 交给算子省掉这次拷贝。',
  '因此这四个基类在图上长得几乎一样，<span class="k">差异被压缩成"用哪个融合算子 + 状态怎么取"两件事</span>。',
  '这条结论与 L2-06 一致：模型架构的差别，到原语这一层就只剩参数选择。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(4600, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i < 2 ? 'on' : ''); });
tl.at(9500, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
tl.at(14500, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 2 ? 'on' : ''); });
tl.at(18500, () => { msg.innerHTML = texts[4]; rows.forEach(r => r.className = ''); });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L2-11 · 家族 · delta net',
    title='<span class="hl-f">gated delta net</span>：状态是一整块 <span class="hl-f">矩阵</span>，不是向量',
    sub='delta net 的状态形状是 {S_v, S_v, H_v, n_seqs} —— 更像"外积累加器"，比 Mamba 的 {d_state, head_dim} 大得多。',
    caption='Kimi Linear / Kimi K3 / BailingMoE3 / Qwen3-Next / Qwen3.5 的 recurrent 层都走这条路径；'
            '同一份代码用 is_recr(il) 与普通的注意力层交替。',
    src=SRC_DELTA, parts=[(527, 568)], duration=24000,
    mark_src=[537, 543, 546, 548, 555, 563, 567],
    notes_src={537: 'n_seq_tokens > 1 是预填充，== 1 是逐 token 解码 —— 两条路径的算子实现不同',
               543: 'keep 为真时走 K>1 的分支：一次多要 K 份快照，供回退用',
               546: 'cparams.n_rs_seq 决定要不要保留回退快照',
               548: '不保留时 build_delta_net 返回 (output, new_state) 一对张量',
               555: '与 Mamba 同样的写回：ggml_cpy 到 ssm_states_all 的 kv_head 行',
               563: '状态元素个数 D = S_v * S_v * H_v —— 二次方级，比 Mamba 的状态大得多',
               567: 'K>1 分支：状态快照直接由算子写在输出张量里，按 n_written 个槽位切'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:8px">
    <div class="card" style="border-left-color:var(--f);flex:1 1 0">
      <div class="ct" style="color:var(--f)">状态形状对比</div>
      <div class="cb">Mamba：<span class="cm" style="margin:0">{d_state, head_dim, n_head}</span> —— 每头一个向量<br>
      GDN：<span class="cm" style="margin:0">{S_v, S_v, H_v, n_seqs}</span> —— 每头一个方阵<br>
      KDA（Kimi）：<span class="cm" style="margin:0">head_dim x head_dim x n_head</span>，源码注释给了 128x128x32 = 524288。</div>
    </div>
    <div class="card" style="border-left-color:var(--b);flex:1 1 0">
      <div class="ct" style="color:var(--b)">三条实现路径</div>
      <div class="cb"><span class="cm" style="margin:0">build_delta_net_autoregressive</span>（n_seq_tokens == 1）<br>
      <span class="cm" style="margin:0">build_delta_net_chunking</span>（预填充，分块并行）<br>
      <span class="cm" style="margin:0">build_delta_net_fused</span>（后端有融合内核时）<br>
      由 <span class="cm" style="margin:0">cparams.fused_gdn_ar / fused_gdn_ch</span> 两个开关选。</div>
    </div>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['谁用它', '行号', '说明'],
  [['qwen3next.cpp', '543', 'gated delta net 层 + 注意力层 + MoE'],
   ['qwen35.cpp / qwen35moe.cpp', '448 / 472', '同族，MoE 版多一层专家'],
   ['kimi-linear.cpp', '342 起', 'KDA 层（有卷积状态）+ MLA 层'],
   ['bailingmoe3.cpp', '290', 'KDA 层走 build_recurrent_attn，卷积用 ggml_ssm_conv'],
   ['kimi-k3.cpp', '459', 'n_head_kv == 0 标记 recurrent 层']],
  { monoCols: [0, 1] });
t.el.style.fontSize = '9px';
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  'delta net 把"线性注意力"写成了一个递推：<span class="k">状态是一块方阵，每步做一次带门控的外积更新</span>。',
  '<span class="v">build_delta_net</span> 用 <span class="v">n_seq_tokens</span> 与两个 cparams 开关'
    + '在三条实现路径之间选一条 —— <span class="k">同一语义，算子粒度不同</span>（对照 L2-06 第 6 幕的 flash attention 快慢路）。',
  '<span class="v">K = cparams.n_rs_seq + 1</span> 与 Mamba 完全一致：这就是"投机解码要回退几步"的统一表达。',
  '这条路径被五个模型家族共用：<span class="v">qwen3next / qwen35 / qwen35moe / kimi-linear / kimi-k3 / bailingmoe3</span>。'
    + '<span class="k">一个基类撑起半个线性注意力家族。</span>',
  '回顾 L2-14 / L2-15：那些稠密 Transformer 里没有这一段 —— 它们的记忆全部在 KV cache 里。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(4800, () => { msg.innerHTML = texts[1]; rows.forEach(r => r.className = ''); });
tl.at(10200, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
tl.at(15200, () => { msg.innerHTML = texts[3]; rows.forEach(r => r.className = 'on'); });
tl.at(20000, () => { msg.innerHTML = texts[4]; rows.forEach(r => r.className = ''); });
'''
)

# ------------------------------------------------------------------ 第 8 幕

FILES_A = [
    ('arwkv7.cpp', 'RWKV-7 的变体：每层 token-shift + build_rwkv7_time_mix + channel mix', '是'),
    ('bailingmoe3.cpp', 'KDA 层 + MLA 层 + MoE：KDA 走 build_recurrent_attn，卷积用 ggml_ssm_conv', '是（混合）'),
    ('deepseek2.cpp', 'DeepSeek-V2 的 MLA + MoE：wkv_a_mqa / wkv_b 是低秩 KV 压缩投影', '否'),
    ('deepseek32.cpp', 'DeepSeek-V3.2 的 DSA 稀疏注意力 + MLA，走 KV cache', '否'),
    ('delta-net-base.cpp', 'gated delta net 公共基类：三种实现 + 卷积状态 + 状态写回', '是（基类）'),
    ('dflash.cpp', 'DFlash 投机解码草稿图：layer.wkv 只是一个普通 K/V 投影权重名', '否'),
    ('dots3note.cpp', 'MLA + DSA indexer（源码注释写明 adapted from deepseek32）', '否'),
    ('falcon-h1.cpp', '混合：按 hparams.is_recr(il) 在 Mamba2 层与注意力层间切换', '是（混合）'),
    ('glm-dsa.cpp', 'GLM 的 DSA 稀疏注意力 + MLA，走 KV cache', '否'),
    ('granite-hybrid.cpp', '混合：Mamba2 层 + 注意力层，inp->get_recr() 传状态', '是（混合）'),
    ('hy-v4.cpp', 'Hunyuan V4 的 MLA + DSA，走 KV cache', '否'),
    ('jamba.cpp', '混合：Mamba-1 层 + 注意力层 + MoE', '是（混合）'),
    ('kimi-k3.cpp', 'KDA 层（n_head_kv == 0 标记）+ MLA 层', '是（混合）'),
    ('kimi-linear.cpp', 'KDA 层（ggml_ssm_conv + gated delta net）+ MLA 层', '是（混合）'),
    ('lfm2.cpp', '短卷积块：卷积状态存 recurrent cache，用 ggml_ssm_conv；无 ssm_scan', '是（只卷积）'),
]

L.scene(
    kicker='L2-11 · 清点（上）',
    title='15 个文件各自的<span class="hl-a">真实主题</span>（按扫描清单顺序）',
    sub='逐文件读图构建代码后写下，不按文件名猜。第三列是本课的判定。',
    caption='顺序即 plan_matrix.py --files 给出的顺序；后 15 个见下一幕。',
    src=SRC_LFM2, parts=[(191, 226)], duration=26000,
    mark_src=[192, 193, 222, 223],
    notes_src={192: '读卷积状态：注意的是 get_r_l —— recurrent 那一份',
               193: 'build_rs 取行；没有 get_ssm_rows lambda，因为 LFM2 没有 ssm_scan',
               223: 'ggml_ssm_conv：LFM2 复用了 Mamba 的卷积算子，但只用它做短卷积'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '真实主题（读代码得出）', 'SSM/线性注意力？'],
  ''' + repr(FILES_A).replace("'", '"') + '''.map(r => [r[0], r[1], r[2]]),
  { monoCols: [0] });
t.el.style.fontSize = '9px';
t.el.style.lineHeight = '1.15';
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '这 15 个里，<span class="k">8 个名副其实，7 个不是</span>。',
  '<span class="v">delta-net-base.cpp</span> 是基类不是模型：它同时服务 kimi-linear / kimi-k3 / bailingmoe3 / qwen3next / qwen35。',
  '<span class="v">bailingmoe3 / kimi-k3 / kimi-linear</span> 是"KDA + MLA"的混合体：'
    + '<span class="k">同一份模型里既有 recurrent 层也有 KV cache 层</span>，由 is_recr(il) 分开。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(8000, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = i === 4 ? 'on' : ''); });
tl.at(15000, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = (i === 1 || i === 12 || i === 13) ? 'on' : ''); });
'''
)

# ------------------------------------------------------------------ 第 9 幕

FILES_B = [
    ('mamba2.cpp', 'Mamba-2 的超参与权重加载；图直接复用 llama_model_mamba::graph', '是'),
    ('mamba-base.cpp', 'llm_build_mamba_base：build_mamba_layer / build_mamba2_layer，ggml_ssm_scan 的唯一出处', '是（基类）'),
    ('mamba.cpp', 'Mamba-1/2 的图：逐层 build_mamba_layer 或 build_mamba2_layer + 残差', '是'),
    ('minicpm3.cpp', 'MiniCPM3 的 MLA 图，走 build_attn_inp_kv', '否'),
    ('nemotron-h.cpp', '混合：Mamba2 层 + 注意力层 + MoE', '是（混合）'),
    ('plamo2.cpp', '自带一份 Mamba 实现：ggml_ssm_conv + ggml_ssm_scan 内联在模型文件里', '是'),
    ('plm.cpp', 'PLM 的 MLA 图，走 build_attn_inp_kv', '否'),
    ('qwen35.cpp', 'gated delta net 层 + 注意力层混合', '是（混合）'),
    ('qwen35moe.cpp', '同上 + MoE', '是（混合）'),
    ('qwen3next.cpp', 'gated delta net 层 + 注意力层 + MoE', '是（混合）'),
    ('rwkv6-base.cpp', 'llm_build_rwkv6_base：channel mix / time mix，wkv6 或 gated_linear_attn', '是（基类）'),
    ('rwkv6.cpp', 'RWKV-6 的图：token-shift + time mix + channel mix', '是'),
    ('rwkv6qwen2.cpp', 'RWKV-6 的 time mix + Qwen2 式 build_ffn（FFN_SILU / FFN_PAR）', '是（混合）'),
    ('rwkv7-base.cpp', 'llm_build_rwkv7_base：ggml_rwkv_wkv7 + 状态回写', '是（基类）'),
    ('rwkv7.cpp', 'RWKV-7 的图：token-shift + build_rwkv7_time_mix', '是'),
]

L.scene(
    kicker='L2-11 · 清点（下）',
    title='★ 余下 15 个，以及<span class="hl-e">8 个名实不符</span>的全部理由',
    sub='8 个"否"的命中点都只是子串 wkv：MLA 的低秩 KV 压缩投影叫 wkv_a_mqa / wkv_b，'
        'dflash 里干脆只是一个普通 K/V 权重叫 wkv。',
    caption='它们走 build_attn_inp_kv（KV cache），不调 build_rs / build_inp_mem_hybrid，'
            '也不出现任何一个 ggml_ssm_* 或 ggml_*wkv* 算子。',
    src=SRC_DS2, parts=[(112, 120)], duration=26000,
    mark_src=[112, 116, 117, 119],
    notes_src={112: 'wkv_a_mqa：MLA 的"下投影"—— 把 n_embd 压到 kv_lora_rank + rope 维；与 RWKV 无关',
               116: 'wk_b / wv_b 是 MLA 的"上投影"',
               119: '不分片的旧式写法里，这个上投影就叫 wkv_b —— 子串 wkv 就来自这里'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '真实主题（读代码得出）', 'SSM/线性注意力？'],
  ''' + repr(FILES_B).replace("'", '"') + '''.map(r => [r[0], r[1], r[2]]),
  { monoCols: [0] });
t.el.style.fontSize = '9px';
t.el.style.lineHeight = '1.15';
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '后 15 个里，<span class="k">12 个名副其实，3 个不是</span>。加上上一幕：'
    + '<span class="v">22 是 / 8 否</span>。',
  '8 个"否"的公共点：它们都在声明 <span class="v">LLM_TENSOR_ATTN_KV_A_MQA</span> / '
    + '<span class="v">LLM_TENSOR_ATTN_KV_B</span> 这类 MLA 张量，名字里带 wkv。',
  '判据不是名字，而是三件事：<span class="k">有没有 build_rs / build_inp_mem_hybrid、'
    + '有没有 ggml_ssm_* 或 ggml_*wkv* 算子、状态是不是定长张量</span>。这 8 个三条全不满足。',
  '<span class="v">dflash.cpp</span> 更彻底：它的 <span class="v">layer.wkv</span> 就是一次普通 K/V 投影'
    + '（<span class="v">LLM_TENSOR_ATTN_KV</span>），文件里连 MLA 都不是。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(8000, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = (i === 3 || i === 6) ? 'on' : ''); });
tl.at(15000, () => { msg.innerHTML = texts[2]; rows.forEach(r => r.className = ''); });
tl.at(21000, () => { msg.innerHTML = texts[3]; rows.forEach(r => r.className = ''); });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L2-11 · 收束',
    title='★ 一张表收束：<span class="hl-a">记忆</span>决定图的形状',
    sub='同一层里，"怎么记"比"算什么"更能决定图长什么样 —— 这就是本课与稠密家族课的分界。',
    caption='下一课 L2-12：MoE 家族（上），看稀疏专家怎么在图上展开；'
            '本课的混合模型（jamba / granite-hybrid / qwen3next 等）在那一课还会再出现。',
    src=SRC_BAILING, parts=[(255, 262)], duration=20000,
    mark_src=[255, 259, 260, 262],
    notes_src={255: 'is_recr(il)：这一层是 recurrent 还是 attention，由 hparams 预先标好',
               259: 'recurrent 层：卷积状态，n_embd_r()',
               260: 'build_rs 取这一批要用的状态行',
               262: '然后一路 build_recurrent_attn —— 与纯 Mamba 走的是同一条路'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['问题', 'KV cache 路线', 'recurrent 路线（本课）', '在哪一课'],
  [['记忆住哪', 'mctx->get_k / get_v（张量）', 'mctx->get_r_l / get_s_l（张量）', 'L2-04'],
   ['大小', 'O(n_ctx)，随上下文增长', 'O(1)，由结构超参决定', '本课 · 第 2 幕'],
   ['读', 'build_rs → get_rows 或把 ids 交给算子', 'build_rs / build_attn_inp_kv', 'L2-06 / 本课'],
   ['写', 'cpy_k / cpy_v', '模型里的 ggml_cpy 写回缓存行', '本课 · 第 5 幕'],
   ['算', 'mul_mat + soft_max（或 flash_attn）', 'ggml_ssm_scan / wkv6 / wkv7 / gdn', '本课 · 第 4~7 幕'],
   ['层怎么选', '全部层', 'hparams.is_recr(il)', '本课 · 第 10 幕'],
   ['代表模型', 'L2-14 / L2-15 的稠密家族、MLA 家族', 'mamba / rwkv / qwen3next / kimi', '—']],
  { monoCols: [1, 2] });
t.el.style.fontSize = '9px';
t.el.style.lineHeight = '1.2';
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '不看代码，说出 <span class="mono">ggml_ssm_scan</span> 的输入、输出，'
  + '以及<b>状态更新究竟发生在哪一步</b>？',
  '<b>输入</b>：7 个张量 + 1 个整数。<br>'
  + '<span class="mono">s {d_state, head_dim, n_head, n_seqs+}</span>（旧状态，'
  + '用 <span class="mono">ids {n_seqs}</span> 选行）、'
  + '<span class="mono">x {head_dim, n_head, n_seq_tokens, n_seqs}</span>、'
  + '<span class="mono">dt {n_head, n_seq_tokens, n_seqs}</span>、'
  + '<span class="mono">A {1, n_head} 或 {d_state, n_head}</span>、'
  + '<span class="mono">B / C {d_state, n_group, n_seq_tokens, n_seqs}</span>，'
  + '外加整数 <span class="mono">K</span>（op_param，不是张量）。<br>'
  + '<b>输出</b>：一个一维 F32 张量，长度 = '
  + '<span class="mono">nelements(x) + K * s->ne[0] * s->ne[1] * s->ne[2] * ids->ne[0]</span>；'
  + '前半段是 <span class="mono">y</span>，后半段是 <span class="mono">K</span> 份新状态'
  + '（<span class="mono">ggml/src/ggml.c:5739-5741</span>）。<br>'
  + '<b>状态更新发生在内核的扫描循环里</b>：'
  + '<span class="mono">ggml/src/ggml-cpu/ops.cpp:9819-9821</span> 用 '
  + '<span class="mono">ids[i3]</span> 定位旧状态读指针 <span class="mono">s0</span>，'
  + '并把新状态写指针 <span class="mono">s</span> 指向<b>算子自己的输出张量</b>后半段'
  + '（<span class="mono">s_off = nelements(x) * element_size</span>）；'
  + '循环体里 <span class="mono">s[i] = state</span>（9923 / 9976 行）就是更新本身。<br>'
  + '把新状态搬回持久缓存是<b>图上的另一个节点</b>：'
  + '<span class="mono">src/models/mamba-base.cpp:274-279</span> 的 '
  + '<span class="mono">ggml_cpy(view_3d(y_ssm, ..., state_offset), '
  + 'view_3d(ssm_states_all, ..., kv_head * row_size))</span>。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '把整课压成一张表：<span class="k">"怎么记"决定图的形状</span>。',
  '读的入口是 <span class="v">build_rs</span> / <span class="v">build_attn_inp_kv</span>，'
    + '两者都在 L2-06 那一族原语里 —— <span class="k">recurrent 不是旁路，是并列的一条</span>。',
  '写的入口永远是模型文件里一个显式的 <span class="v">ggml_cpy</span>：'
    + '<span class="k">算子不改缓存，只产出新状态</span>。',
  '混合模型靠 <span class="v">hparams.is_recr(il)</span> 在两种层之间切换 —— '
    + 'jamba / granite-hybrid / nemotron-h / falcon-h1 / lfm2 / qwen3next / qwen35 / kimi 全是这个套路。',
  '下一课 L2-12 讲 MoE：那是"怎么算"的另一半，与本课的"怎么记"正交。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; rows.forEach(r => r.className = ''); });
tl.at(4200, () => { msg.innerHTML = texts[1]; rows.forEach((r, i) => r.className = (i === 2 || i === 3) ? 'on' : ''); });
tl.at(8600, () => { msg.innerHTML = texts[2]; rows.forEach((r, i) => r.className = i === 3 ? 'on' : ''); });
tl.at(13000, () => { msg.innerHTML = texts[3]; rows.forEach((r, i) => r.className = i === 5 ? 'on' : ''); });
tl.at(16500, () => { msg.innerHTML = texts[4]; rows.forEach(r => r.className = ''); });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、30 个文件的真实主题（分组依据与判定）',
    '**分组依据是静态扫描，不是内容判定。** 命令：\n\n'
    '```text\n'
    'cd /data/WORKSPACE/llama.cpp-project/curriculum\n'
    'python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L2-11"{print $2}\'\n'
    '```\n\n'
    '这 30 个 `src/models/*.cpp` 的图构建代码里命中 `build_ssm|ssm_conv|ssm_scan|gated_delta|'
    'build_rwkv|wkv` 之一。**逐文件读过之后，22 个名副其实，8 个名实不符。**\n\n'
    '| 文件 | 真实主题（读代码得出） | 判定 | 判断依据 |\n|---|---|---|---|\n'
    '| `arwkv7.cpp` | RWKV-7 变体：token-shift + `build_rwkv7_time_mix` + channel mix | 是 | `llm_build_rwkv7_base` + `ggml_rwkv_wkv7` |\n'
    '| `bailingmoe3.cpp` | KDA 层 + MLA 层 + MoE | 是（混合） | `build_recurrent_attn`(290) + `ggml_ssm_conv`(213) + `get_s_l`(286) |\n'
    '| `deepseek2.cpp` | DeepSeek-V2 的 MLA + MoE | 否 | 只有 `wkv_a_mqa`(112) / `wkv_b`(119) 命中；走 `build_attn_inp_kv` |\n'
    '| `deepseek32.cpp` | DeepSeek-V3.2 的 DSA 稀疏注意力 + MLA | 否 | 3 处命中全是 `wkv_a_mqa`；走 KV cache |\n'
    '| `delta-net-base.cpp` | gated delta net 公共基类 | 是（基类） | `ggml_gated_delta_net`(402/567) + `build_conv_state`(449) |\n'
    '| `dflash.cpp` | DFlash 投机解码草稿图（DSV4 / MLA 双模式） | 否 | 命中的 `layer.wkv`(196) 是 `LLM_TENSOR_ATTN_KV`，普通 K/V 投影 |\n'
    '| `dots3note.cpp` | MLA + DSA indexer | 否 | 源码注释写明 adapted from deepseek32；只有 `wkv_a_mqa`(96/315) |\n'
    '| `falcon-h1.cpp` | 混合：Mamba2 层 + 注意力层 | 是（混合） | `build_inp_mem_hybrid`(125) + `build_mamba2_layer`(161) |\n'
    '| `glm-dsa.cpp` | GLM 的 DSA 稀疏注意力 + MLA | 否 | 3 处命中全是 `wkv_a_mqa` |\n'
    '| `granite-hybrid.cpp` | 混合：Mamba2 层 + 注意力层 | 是（混合） | `is_recr(il)`(162) + `build_mamba2_layer`(164) |\n'
    '| `hy-v4.cpp` | Hunyuan V4 的 MLA + DSA | 否 | 3 处命中全是 `wkv_a_mqa` |\n'
    '| `jamba.cpp` | 混合：Mamba-1 层 + 注意力层 + MoE | 是（混合） | `build_inp_mem_hybrid`(117) + `build_mamba_layer`(128) |\n'
    '| `kimi-k3.cpp` | KDA 层 + MLA 层 | 是（混合） | `n_head_kv == 0` 标记 recurrent 层(30-32) + `build_recurrent_attn`(459) |\n'
    '| `kimi-linear.cpp` | KDA 层 + MLA 层 | 是（混合） | `ggml_ssm_conv`(226) + `build_rs`(297/342) |\n'
    '| `lfm2.cpp` | 短卷积块 + 注意力层 | 是（只卷积） | `get_r_l`(192) + `ggml_ssm_conv`(223)；**无 `ssm_scan`** |\n'
    '| `mamba2.cpp` | Mamba-2 超参与权重加载；图复用 `llama_model_mamba::graph` | 是 | `ssm_conv1d`(69) + `using graph = llama_model_mamba::graph` |\n'
    '| `mamba-base.cpp` | `llm_build_mamba_base`：`build_mamba_layer` / `build_mamba2_layer` | 是（基类） | `ggml_ssm_scan`(125/264) + `ggml_ssm_conv`(77/232) |\n'
    '| `mamba.cpp` | Mamba-1/2 的图 | 是 | `llm_build_mamba_base` + 逐层 `build_mamba*_layer`(104/106) |\n'
    '| `minicpm3.cpp` | MiniCPM3 的 MLA 图 | 否 | 只有 `wkv_a_mqa` / `wkv_b`；走 `build_attn_inp_kv` |\n'
    '| `nemotron-h.cpp` | 混合：Mamba2 层 + 注意力层 + MoE | 是（混合） | `build_inp_mem_hybrid`(199) + `build_mamba2_layer`(215) |\n'
    '| `plamo2.cpp` | 自带一份 Mamba 实现（`ggml_ssm_conv` + `ggml_ssm_scan` 内联） | 是 | `ggml_ssm_scan`(385) + `build_rs`(286/388) |\n'
    '| `plm.cpp` | PLM 的 MLA 图 | 否 | 只有 `wkv_a_mqa` / `wkv_b`；走 `build_attn_inp_kv` |\n'
    '| `qwen35.cpp` | gated delta net 层 + 注意力层 | 是（混合） | `ggml_ssm_conv`(391) + `build_recurrent_attn`(448) |\n'
    '| `qwen35moe.cpp` | 同上 + MoE | 是（混合） | `ggml_ssm_conv`(415) + `build_recurrent_attn`(472) |\n'
    '| `qwen3next.cpp` | gated delta net 层 + 注意力层 + MoE | 是（混合） | `ggml_ssm_conv`(471) + `build_recurrent_attn`(543) |\n'
    '| `rwkv6-base.cpp` | `llm_build_rwkv6_base` | 是（基类） | `ggml_rwkv_wkv6`(139) / `ggml_gated_linear_attn`(137) |\n'
    '| `rwkv6.cpp` | RWKV-6 的图 | 是 | `build_rwkv6_time_mix`(130) + `build_rwkv_token_shift_*`(116/148) |\n'
    '| `rwkv6qwen2.cpp` | RWKV-6 time mix + Qwen2 式 FFN | 是（混合） | `build_rwkv6_time_mix`(116) + `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, ...)`(138) |\n'
    '| `rwkv7-base.cpp` | `llm_build_rwkv7_base` | 是（基类） | `ggml_rwkv_wkv7`(107) + 状态 `ggml_cpy`(111-114) |\n'
    '| `rwkv7.cpp` | RWKV-7 的图 | 是 | `build_rwkv7_time_mix`(161) + `build_rwkv7_channel_mix`(190) |\n\n'
    '**判定用的三条硬指标**（8 个"否"三条全不满足）：\n\n'
    '1. 有没有 `build_rs` / `build_inp_mem_hybrid` 这类 recurrent 记忆入口；\n'
    '2. 有没有 `ggml_ssm_conv` / `ggml_ssm_scan` / `ggml_rwkv_wkv*` / `ggml_gated_*` 算子；\n'
    '3. 状态是不是与 `n_ctx` 无关的定长张量。\n\n'
    '`lfm2.cpp` 只满足第 2 条的 `ggml_ssm_conv` 与第 3 条的卷积状态，**没有 `ggml_ssm_scan`** —— '
    '它是一个"只借用了卷积状态"的混合模型，本课如实标注为「是（只卷积）」。')

L.section(
    '二、★ recurrent 记忆与 KV cache 的分工',
    '`llama_memory_recurrent_context` 给出的两个张量：`get_r_l(il)` 是卷积状态（大小 `n_embd_r()`），'
    '`get_s_l(il)` 是 SSM/WKV 状态（大小 `n_embd_s()`）。两者都不含 token 维。\n\n'
    '`n_embd_s()` 是一个**纯结构常量**：RWKV 走 `n_embd * wkv_head_size`，'
    'KDA 走 `head_dim * head_dim * n_head()`（源码注释给了 128×128×32 = 524288），'
    'Mamba 走 `ssm_d_state * ssm_d_inner`。**没有一项依赖 `n_ctx`。**\n\n'
    '对照 L2-04：那一课讲的是 memory 侧的接口与 cell/slot 机制；'
    '本课讲的是**图侧怎么把这两个张量取出来、算完、再写回去**。',
    src=SRC_HPARAMS, parts=[(236, 258)], lang='c')

L.section(
    '三、memory 侧的接口：get_r_l / get_s_l',
    '本课图侧的所有"读状态"最终都落到这两个函数上。`get_head()` 给出这一批序列在状态缓存里的起始行，'
    '`get_size()` 给出缓存的总行数 —— 与 L2-04 的 KV cache cell/slot 是同一套编号机制。'
    '注意还有第三个张量 `get_p_l(il)`（本课覆盖的 30 个文件都没有用到它）。',
    src=SRC_MEM_REC, parts=[(166, 176)], lang='c')

L.section(
    '四、ggml_ssm_scan 的 C API',
    '8 个形参就是这一课验收点的前半部分。紧挨在上面的是 `ggml_ssm_conv`（只有 2 个输入：'
    '`sx` 是拼好状态的序列、`c` 是卷积权重）—— 卷积算子连状态张量都不接，'
    '状态进出完全由图上的 `ggml_concat` 与 `ggml_cpy` 负责。',
    src=SRC_GGML_H, parts=[(2524, 2538)], lang='c')

L.section(
    '五、ggml_ssm_scan 的算子契约（输入 / 输出 / 状态）',
    '`ggml/include/ggml.h:2529-2538` 给出 C API 的 8 个形参；'
    '`ggml/src/ggml.c:5711-5751` 用一连串 `GGML_ASSERT` 把它们之间的形状关系钉死，'
    '最后构造输出张量 —— **输出是"y 拼上 K 份状态"的一维 F32 张量**。',
    src=SRC_GGML_C, parts=[(5711, 5751)], lang='c')

L.section(
    '六、状态更新在哪一步发生（验收点）',
    '分两步，分别在两个地方：\n\n'
    '**第一步（内核里）**：`ggml/src/ggml-cpu/ops.cpp` 的 `ggml_compute_forward_ssm_scan_f32` '
    '在扫描循环里更新状态。它用 `ids[i3]` 定位旧状态的读指针 `s0`，'
    '把新状态的写指针 `s` 指向**算子自己的输出张量**后半段（`s_off = nelements(x) * element_size`），'
    '循环体里 `s[i] = state` 就是更新本身。注意：**算子不改持久缓存**。\n\n'
    '**第二步（图上）**：模型文件里一个显式的 `ggml_cpy` 把输出张量后半段搬回 `ssm_states_all` 的 '
    '`kv_head` 行。Mamba 在 `src/models/mamba-base.cpp:274-279`，RWKV 在 '
    '`src/models/rwkv6-base.cpp:144-147`，delta net 在 `src/models/delta-net-base.cpp:555-558`。',
    src=SRC_MAMBA_BASE, parts=[(267, 279)], lang='c')

L.section(
    '七、混合模型：is_recr(il) 把两种层拼进一个图',
    '`jamba` / `granite-hybrid` / `nemotron-h` / `falcon-h1` / `qwen3next` / `qwen35` / '
    '`qwen35moe` / `kimi-linear` / `kimi-k3` / `bailingmoe3` 都是混合体：'
    '一部分层走 recurrent 记忆，另一部分层走 KV cache，由 `hparams.is_recr(il)` 在图上分支。'
    '`kimi-k3.cpp:30-32` 的源码注释明说：`n_head_kv == 0` 就标记为 KDA（recurrent）层。',
    src=SRC_GRANITE, parts=[(162, 168)], lang='c')

L.section(
    '八、RWKV 的"读-算-写"与 Mamba 的同构性',
    '`rwkv6-base.cpp:133-147` 与 `mamba-base.cpp:258-279` 是同构的：'
    '都是"从 `get_s_l(il)` 取状态 → 交给一个融合算子 → 输出张量里同时含 y 与新状态 → `ggml_cpy` 写回"。'
    '差别只有两处：用哪个融合算子，以及**状态怎么取** —— '
    'RWKV 走 `build_rs` 的默认实现 `ggml_get_rows`（`llama-graph.h:1333`），'
    'Mamba / plamo2 传入自己的 lambda，把 `ids` 直接交给 `ggml_ssm_scan`。',
    src=SRC_RWKV7B, parts=[(105, 114)], lang='c')

L.section(
    '九、token shift：RWKV 的第二份状态',
    'RWKV 每层除了 wkv 状态，还要存"上一个 token 的 norm 输出"（token shift）。'
    '它由 `build_rwkv_token_shift_load` / `build_rwkv_token_shift_store` 两个原语读写，'
    '占用的正是 `n_embd_r()` 那一份卷积状态。这两个原语的实现在 `src/llama-graph.cpp`（L2-06），'
    '下面是 load 那个（store 在 3579-3597，做法就是一个 `ggml_cpy` 回 `get_r_l(il)`）。',
    src=SRC_GRAPH_CPP, parts=[(3558, 3577)], lang='c')

L.section(
    '十、build_rs：状态行怎么被取出来',
    '`build_rs` 是 `llama-graph` 的"记忆读原语"，与 KV cache 的 `build_attn` 并列（L2-06）。'
    '它做三件事：把状态张量 reshape 成 `{state_size, n_rs}`、把一个槽位清零、'
    '然后按 `ids` 取出这一批要用的行。**默认的行选择就是 `ggml_get_rows`**；'
    '传了 lambda 的调用方（Mamba / plamo2）可以把 `ids` 一路传进融合算子，省掉这次拷贝。',
    src=SRC_GRAPH_H, parts=[(1318, 1342)], lang='c')

L.section(
    '十一、卷积状态：一个定长移位寄存器',
    '`ggml_ssm_conv` 是纯算子，只吃 `conv_x` 与权重；状态与输入的拼接由 `ggml_concat` 完成，'
    '写回由显式 `ggml_cpy` 完成。窗口长度恒为 `d_conv - 1`。'
    '`kimi-linear.cpp:216-225` 的注释额外说明了权重布局约定：`ggml_ssm_conv` 按 '
    '`c[conv_step + channel * d_conv]` 取权重。',
    src=SRC_KIMI_LIN, parts=[(216, 231)], lang='c')

L.footnote_add('**覆盖声明**：本课声明块里的 30 个 `src/models/*.cpp` 是本课负责的文件，全部计入覆盖率，'
               '与 `tools/plan_matrix.py --files` 里 L2-11 的清单逐字一致。')
L.footnote_add('**跨课引用（同样计入覆盖率，但它们的主课在别处）**：'
               '`ggml/include/ggml.h`（L1-01）、`ggml/src/ggml.c`（L1-02/03/04/05、L4-04）、'
               '`ggml/src/ggml-cpu/ops.cpp`（L5-02）、`src/llama-memory-recurrent.h`（L2-04）、'
               '`src/llama-graph.{h,cpp}`（L2-06）、`src/llama-hparams.cpp`（L2-02）。'
               '本课引用它们，是为了把"模型文件里的调用"对到"算子契约"与"内核实现"上 —— '
               '**图原语的出处是 `src/llama-graph.{h,cpp}`（L2-06），不在本课范围**；'
               '本课只讲模型文件怎么调用它们。此处明确声明，不虚报。')
L.footnote_add('**一个需要澄清的措辞**：本课的 `build_ssm_*` / `build_rwkv_*` 原语**不在** '
               '`src/llama-graph.cpp` 里。实测：`grep -rn "build_ssm_scan|build_ssm_conv" src/` 无输出；'
               '它们实际定义在 `src/models/mamba-base.cpp`（`build_mamba_layer` / `build_mamba2_layer`）、'
               '`src/models/rwkv6-base.cpp`（`build_rwkv6_*`）、`src/models/rwkv7-base.cpp`（`build_rwkv7_*`）、'
               '`src/models/delta-net-base.cpp`（`build_delta_net` / `build_recurrent_attn`）。'
               '`llama-graph.cpp` 里与本课相关的是 `build_rs`(3478) 与 '
               '`build_rwkv_token_shift_load/store`(3558/3579)。')
L.footnote_add('**统计口径**：本课的"22 是 / 8 否"由逐文件读图构建代码得出，'
               '判据写在第一节末尾的三条硬指标里；`grep -c` 的命中数只用于定位，不用于判定。')

L.prereqs('`L2-06`（★ 计算图骨架 llama-graph）')

L.goal(
    '说出本课 30 个文件里哪些真正属于状态空间/线性注意力，并给出判定依据（对应第 8、9 幕的表格）；',
    '解释为什么 SSM/线性注意力模型不需要 KV cache，以及 `get_r_l()` / `get_s_l()` 各自是什么；',
    '**说出 `ggml_ssm_scan` 的 7 个张量输入 + 1 个整数参数各自的形状与含义（验收点前半）；**',
    '**说出状态更新发生在哪一步，以及"内核更新"与"图上写回"为什么是两个不同的节点（验收点后半）；**',
    '解释 `hparams.is_recr(il)` 如何在同一个模型里把 recurrent 层与注意力层拼进一张图。')

L.conclusion(
    '★ 状态更新发生在哪一步（验收点）',
    '```text\n'
    '① 读    build_rs → reshape 状态缓存 {d_state, head_dim, n_head, state_slots}\n'
    '            mamba-base.cpp:258-267（把 ids 交给算子；RWKV 则先用 ggml_get_rows 取行）\n'
    '② 算    ggml_ssm_scan 在扫描循环里更新状态：\n'
    '            ops.cpp:9819-9821  用 ids[i3] 定位读指针 s0，写指针 s 指向输出张量后半段\n'
    '            ops.cpp:9923/9976  循环体 s[i] = state  —— 更新就在这里\n'
    '③ 出    输出张量 = [ y | K 份新状态 ]\n'
    '            ggml.c:5739-5741   长度 = nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]\n'
    '④ 写回  ggml_cpy(output 后半段 → ssm_states_all 的 kv_head 行)\n'
    '            mamba-base.cpp:274-279 / rwkv6-base.cpp:144-147 / delta-net-base.cpp:555-558\n'
    '```\n\n'
    '**关键区别：算子不改持久缓存，它只把新状态写在自己的输出张量里；'
    '写回缓存是图上一个独立的 `ggml_cpy` 节点，由调度器保证"先读后写"的次序。**')

L.conclusion(
    '★ SSM 不需要 KV cache',
    '记忆的两种形态：\n\n'
    '| | KV cache | recurrent state（本课） |\n|---|---|---|\n'
    '| 张量 | `get_k` / `get_v` | `get_r_l` / `get_s_l` |\n'
    '| 大小 | `O(n_ctx)`，随上下文增长 | `O(1)`，由结构超参决定 |\n'
    '| 形状 | 带 token 维 | **没有 token 维** |\n'
    '| 读写 | `cpy_k` / `cpy_v` + `build_attn` | 显式 `ggml_cpy` + 融合算子 |\n\n'
    '`llama_hparams::n_embd_s()` 是纯结构常量：RWKV 走 `n_embd * wkv_head_size`，'
    'KDA 走 `head_dim * head_dim * n_head()`，Mamba 走 `ssm_d_state * ssm_d_inner` —— '
    '没有一项依赖 `n_ctx`。代价是历史被有损压缩，这正是"线性注意力"的来历。')

L.conclusion(
    '本课 30 个文件的判定结果',
    '**22 个名副其实**（4 个基类含在内）：`mamba-base` `mamba` `mamba2` `plamo2` '
    '`delta-net-base` `rwkv6-base` `rwkv6` `rwkv6qwen2` `rwkv7-base` `rwkv7` `arwkv7` '
    '`jamba` `granite-hybrid` `nemotron-h` `falcon-h1` `lfm2` `qwen3next` `qwen35` `qwen35moe` '
    '`kimi-linear` `kimi-k3` `bailingmoe3`。\n\n'
    '**8 个名实不符**：`deepseek2` `deepseek32` `dots3note` `glm-dsa` `hy-v4` `minicpm3` `plm` `dflash` —— '
    '它们命中的只是 MLA 低秩 KV 压缩投影的名字 `wkv_a_mqa` / `wkv_b`（`dflash` 里更只是一个普通 K/V 权重 `wkv`），'
    '走的是 `build_attn_inp_kv` 与 KV cache，与线性注意力无关。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
