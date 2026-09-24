#!/usr/bin/env python3
"""L8-01 · 端到端：一个 mul_mat 的完整旅程 —— 课件 spec。

运行：python3 L8-end-to-end/L8-01-journey-of-mul-mat/lesson.spec.py

本课是收束课：把 L1~L7 各层讲的"一跳"串成一条可以复述的链。
所有引用一律按行号从上游抽取（见 plan/AUTHORING.md 第 3.1 节）。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

SRC_GRAPH  = 'src/llama-graph.cpp'                  # 本课主文件（计划内的 1 文件）
SRC_MODEL  = 'src/models/llama.cpp'                 # build_attn 的调用点
SRC_GGML   = 'ggml/src/ggml.c'                      # mul_mat 构造器 + 入图/拓扑序
SRC_CTX    = 'src/llama-context.cpp'                # decode 与 graph_compute
SRC_SCHED  = 'ggml/src/ggml-backend.cpp'            # 切分、归属、执行
SRC_CPUCPP = 'ggml/src/ggml-cpu/ggml-cpu.cpp'       # CPU 后端接口与入口
SRC_CPUC   = 'ggml/src/ggml-cpu/ggml-cpu.c'         # CPU 分派 + vec_dot 选择
SRC_CUDA   = 'ggml/src/ggml-cuda/ggml-cuda.cu'      # 同一跳的 CUDA 实现（对照）
SRC_X86    = 'ggml/src/ggml-cpu/arch/x86/quants.c'  # x86 的 vec_dot 内核

L = Lesson(
    id='L8-01',
    layer='L8 · 端到端',
    title='★ 端到端：一个 mul_mat 的完整旅程',
    codecap='9 个文件 · 逐字引用（每条链路都标了真实行号）',
    nav={'prev': {'href': '../../L7-npu-backend/L7-06-small-backends/index.html',
                  'label': 'L7-06 小后端合集'},
         'next': {'href': '../L8-02-offload-decision/index.html',
                  'label': 'L8-02 ★ offload 决策实战'}},
)

# 计划里本课的主文件 + 链路追加的 3 个文件（显式声明，便于对照覆盖率）
L.cover(SRC_GRAPH, SRC_GGML, SRC_SCHED, SRC_CPUC)

L.note('**一句话**：这一课不讲新机制，只做一件事 —— 挑 `build_attn` 里**一次真实的 '
       '`ggml_mul_mat` 调用**，从它在模型代码里被调用的那一刻起，一路走到 CPU 的 '
       '`vec_dot` 内核，并在**每一跳上指出判据**：谁决定下一步走哪。')
L.note('这条链路过 **9 个文件**：`src/models/llama.cpp` → `src/llama-graph.cpp` → '
       '`ggml/src/ggml.c` → `src/llama-context.cpp` → `ggml/src/ggml-backend.cpp` → '
       '`ggml/src/ggml-cpu/ggml-cpu.cpp` → `ggml/src/ggml-cpu/ggml-cpu.c` → '
       '`ggml/src/ggml-cpu/arch/x86/quants.c`（对照：`ggml/src/ggml-cuda/ggml-cuda.cu`）。')
L.note('主角选的是**输出投影**那一次：`build_attn`（`src/llama-graph.cpp:2766`）在第 2804 行'
       '调用 `build_lora_mm(wo, cur, wo_s)`，后者在第 1518 行调用 `ggml_mul_mat(ctx0, w, cur)`。'
       '选它的原因很实际：`wo` 是**权重**，它的 buffer 决定这次 mul_mat 归哪个后端 —— '
       '这正是 L4-02 讲的那条判据；而同一段 `build_attn_mha` 里的另一次 mul_mat（第 2715 行）'
       '没有权重，归属判据完全不同（第 7 幕会对照）。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L8 · 端到端',
    title='一课走完一条链路：<span class="hl-a">ggml_mul_mat</span> 的完整旅程',
    sub='一次 mul_mat 要穿过 4 个世界、9 个文件；每一跳都有一个明确的判据。',
    caption='前置课 L7-06（小后端合集）；下一课 L8-02 用真实模型参数走一遍 offload 决策。',
    src=SRC_GRAPH, parts=[(1514, 1518)], duration=20000,
    mark_src=[1518],
    notes_src={1518: '本课的主角就是这一行：它在图上创建了一个 GGML_OP_MUL_MAT 节点'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip a">建图 L1/L2</span><span class="arrow">-&gt;</span>
    <span class="chip b">入图 L1</span><span class="arrow">-&gt;</span>
    <span class="chip c">调度 L3/L4</span><span class="arrow">-&gt;</span>
    <span class="chip d">内核 L5/L6/L7</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '第 1 段 · 建图', m: 'ggml_mul_mat',
    b: '模型 build 函数调用 build_attn；<br>一次调用只往图上加一个节点' },
  { c: 'b', t: '第 2 段 · 入图', m: 'ggml_build_forward_expand',
    b: '从根出发 DFS；<br>visited_hash_set 决定节点只进图一次' },
  { c: 'c', t: '第 3 段 · 调度', m: 'ggml_backend_sched_*',
    b: '切分 + 归属 + 分配：<br>决定这个节点跑在哪个后端' },
  { c: 'd', t: '第 4 段 · 内核', m: 'ggml_compute_forward',
    b: '后端自己的 switch 再分派：<br>vec_dot / CUDA kernel / NPU 算子' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '本课只做一件事：把一个 <span class="k">ggml_mul_mat</span> 从"被调用"到"算完"，逐跳走完。',
  '主角：<span class="v">build_lora_mm(wo, cur)</span> 里的 <span class="v">ggml_mul_mat(ctx0, w, cur)</span>（llama-graph.cpp:1518）。',
  '它要穿过 9 个文件：models/llama.cpp、llama-graph.cpp、ggml.c、llama-context.cpp、ggml-backend.cpp、ggml-cpu.cpp/c、arch/x86/quants.c。',
  '记住判据：<span class="k">can_mul_mat</span> → <span class="k">visited_hash_set</span> → <span class="k">weight buffer</span> → <span class="k">iface 槽</span> → <span class="k">tensor-&gt;op</span> → <span class="k">src0-&gt;type</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(17600, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L8-01 · 第 0 跳',
    title='起点：<span class="hl-b">模型 build 函数</span>调用 build_attn',
    sub='第 169 行把这一层的 wo 权重交出去；第 246 行才把整条链拉进图。',
    caption='回顾 L2-06：src/models/llama.cpp 的 graph<>() 是整张图的装配线。',
    src=SRC_MODEL, parts=[(169, 172), (246, 246)], duration=17000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="row" style="gap:9px">
    <div class="col grow" id="stack" style="gap:6px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const stack = wrap.querySelector('#stack');
stack.innerHTML = '<div class="cm" style="margin-bottom:2px">模型侧：llama_model_llama::graph&lt;embed&gt;()</div>';
const frames = [
  { n: 'for (int il = 0; il < n_layer; ++il)', c: 'a', d: '逐层装配（第 132 行起）' },
  { n: 'cur = build_attn(inp_attn, ...)', c: 'b', d: '把 wo / wo_b / wo_s 与 Q/K/V 交出去（169）' },
  { n: 'cur = build_lora_mm(wo, cur, wo_s)', c: 'c', d: 'build_attn 内部：输出投影（2804）' },
  { n: 'ggml_mul_mat(ctx0, w, cur)', c: 'd', d: '图上多一个 MUL_MAT 节点（1518）' }
];
const els = frames.map(f => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
  e.innerHTML = '<span class="cm" style="margin:0">' + U.esc(f.n) + '</span><br>' +
    '<span style="color:var(--' + f.c + ');font-size:9px">' + U.esc(f.d) + '</span>';
  stack.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">wo 是权重，不是激活</div>' +
  '<div class="cb">它在模型加载时就被放进某个后端的 buffer，' +
  'usage = <span class="cm" style="margin:0">GGML_BACKEND_BUFFER_USAGE_WEIGHTS</span>。' +
  '这个事实会在第 7 幕决定这次 mul_mat 归谁。</div></div>' +
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">第 246 行：整条链入图</div>' +
  '<div class="cb">层循环里建出来的节点，最后靠 <span class="cm" style="margin:0">ggml_build_forward_expand(gf, cur)</span> ' +
  '（246）从根反向 DFS 才真正进入 <span class="cm" style="margin:0">cgraph</span>。' +
  '中间还有若干次局部的 expand（如 llama-graph.cpp:2735、2783）。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '起点不在 ggml 里，在<span class="k">模型代码</span>里：一次普通的 C++ 调用。',
  '<span class="v">build_attn</span> 拿到的是本层的 <span class="k">wo</span>（输出投影权重）与 Q/K/V 三个张量。',
  '<span class="v">build_lora_mm</span> 只是个薄封装：它把权重与激活交给 <span class="k">ggml_mul_mat</span>。',
  '注意这一步<span class="k">什么都没算</span>：只是往 <span class="v">ctx0</span> 里加了一个节点。'
];
els.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.28');
  msg.innerHTML = texts[i];
}));
tl.at(14800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L8-01 · ★ 洞察',
    title='★ 图上没有"执行"：<span class="hl-a">一次调用 = 一个节点</span>',
    sub='build_attn 第 2804 行 → build_lora_mm 第 1514 行 → ggml_mul_mat 第 1518 行。',
    caption='对照：同一段 build_attn_mha 里的 kqv = ggml_mul_mat(v, kq)（2715）没有权重，归属判据不同（见第 7 幕与 L4-02）。',
    src=SRC_GRAPH, parts=[(2803, 2805), (1514, 1518)], duration=20000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia" style="gap:11px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const dia = wrap.querySelector('#dia');

const a = U.el('div', { class: 'card', style: 'width:210px;border-left-color:var(--a)' });
a.innerHTML = '<div class="ct" style="color:var(--a)">build_attn(wo, cur)</div>' +
  '<div class="cb">llama-graph.cpp:2804<br>wo 非空 → 做输出投影<br>' +
  '<span class="cm" style="margin:0">cur = build_lora_mm(wo, cur, wo_s)</span></div>';
const b = U.el('div', { class: 'card', style: 'width:210px;border-left-color:var(--c)' });
b.innerHTML = '<div class="ct" style="color:var(--c)">build_lora_mm</div>' +
  '<div class="cb">llama-graph.cpp:1518<br>本体只有一行（外加 LoRA 分支）<br>' +
  '<span class="cm" style="margin:0">ggml_mul_mat(ctx0, w, cur)</span></div>';
const c = U.el('div', { class: 'card', style: 'width:210px;border-left-color:var(--d)' });
c.innerHTML = '<div class="ct" style="color:var(--d)">新节点 kqv_out</div>' +
  '<div class="cb">op = MUL_MAT<br>src[0] = w（权重）<br>src[1] = cur（激活）</div>';
dia.appendChild(a); dia.appendChild(U.arrow('-&gt;')); dia.appendChild(b);
dia.appendChild(U.arrow('-&gt;')); dia.appendChild(c);

const msg = wrap.querySelector('#msg');
const texts = [
  '把这三行连起来看：<span class="k">模型代码 → 薄封装 → 构造器</span>，中间没有任何计算。',
  '<span class="v">ggml_mul_mat</span> 返回的是一个 <span class="k">ggml_tensor *</span> —— 它是一个<em>句柄</em>，不是结果。',
  '节点的全部信息就是：<span class="v">op</span> + <span class="v">src[]</span> + <span class="v">ne[]</span> + <span class="v">type</span>。',
  '所以"什么时候算"这个问题，在图上根本不存在 —— 它由后面第 6 幕的 decode 决定。'
];
[0, 1, 2].forEach(i => tl.at(700 + i * 3600, () => {
  msg.innerHTML = texts[i];
}));
tl.at(11600, () => { msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L8-01 · 第 1 跳',
    title='构造器：<span class="hl-c">三条断言</span> + 一个输出形状',
    sub='ggml_mul_mat 的实现只有 15 行；判据全在 ggml_can_mul_mat 里。',
    caption='回顾 L1-02：op 身份（3351）与输出形状规则（3348）都写在构造器里，没有单独的元数据表。',
    src=SRC_GGML, parts=[(3333, 3356)], duration=22000,
    mark_src=[3333, 3336, 3337, 3338, 3341, 3345, 3346, 3348, 3349, 3351, 3352, 3353],
    notes_src={3333: '判据函数：能不能乘，只看三个 ne 关系',
               3336: '① 内维必须相等：a 的列数 == b 的列数',
               3337: '② a 必须在 dim2 上可广播（b 的 ne[2] 是 a 的整数倍）',
               3338: '③ 同上，dim3',
               3341: '构造器本体从这里开始',
               3345: '不满足就直接断言失败 —— 这是"编译期之外"的图级检查',
               3346: 'a 不能是转置视图：内核按 src0 的行扫描（L5-03 的前提）',
               3348: '★ 输出形状只有一行：{ a->ne[1], b->ne[1], b->ne[2], b->ne[3] }',
               3349: '输出类型固定 F32，与 a/b 的类型无关（量化权重也乘出 F32）',
               3351: '身份面：op 决定后面所有 switch 走哪个 case',
               3352: '输入个数由这里赋了几个 src 决定（L1-02 实测：没有 nargs 表）',
               3353: 'src[1] 是激活 —— 它在第 7 幕决定归属时帮不上忙'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:2px">判据：ggml_can_mul_mat(a, b)</div>';
const checks = [
  { n: 'a-&gt;ne[0] == b-&gt;ne[0]', c: 'a', d: '内维相等（K 维对齐）' },
  { n: 'b-&gt;ne[2] % a-&gt;ne[2] == 0', c: 'b', d: 'a 在 dim2 可广播' },
  { n: 'b-&gt;ne[3] % a-&gt;ne[3] == 0', c: 'c', d: 'a 在 dim3 可广播' },
  { n: '!ggml_is_transposed(a)', c: 'd', d: 'src0 不能是转置视图' }
];
const els = checks.map(k => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
  e.innerHTML = '<span class="cm" style="margin:0;color:var(--' + k.c + ')">' + k.n + '</span>' +
    '<br><span style="font-size:9px">' + U.esc(k.d) + '</span>';
  left.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="formula" id="shape" style="font-size:10px"></div>' +
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">输出为什么是 F32？</div>' +
  '<div class="cb">因为 <span class="cm" style="margin:0">ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)</span> 写死了。' +
  '量化权重负责"省内存"，累加与结果一律走 F32 —— 这是 L1-04 与 L5-03 的共同前提。</div></div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">输入是"借来的"</div>' +
  '<div class="cb">构造器只存指针：<span class="cm" style="margin:0">src[0] = a</span>、' +
  '<span class="cm" style="margin:0">src[1] = b</span>。' +
  '所以这个节点与 <span class="cm" style="margin:0">wo</span>、与激活共享同一块内存。</div></div>';
const shape = right.querySelector('#shape');
shape.innerHTML = '<span class="m">ne</span> = { <span style="color:var(--a)">a-&gt;ne[1]</span>, ' +
  '<span style="color:var(--b)">b-&gt;ne[1]</span>, <span style="color:var(--c)">b-&gt;ne[2]</span>, ' +
  '<span style="color:var(--d)">b-&gt;ne[3]</span> }';

const msg = wrap.querySelector('#msg');
const texts = [
  '构造器只做四件事：<span class="k">断言可乘</span>、<span class="k">算 ne[]</span>、<span class="k">建张量</span>、<span class="k">填 op 与 src</span>。',
  '三条 ne 判据 + 一条"src0 不能转置"，就是这张图能不能建出来的全部条件。',
  '输出形状 = <span class="v">{a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3]}</span>：把 a 的"行数"与 b 的其余维度拼起来。',
  '输出类型与输入类型<span class="k">无关</span>：永远是 F32。这就是"量化权重乘 F32 激活"能成立的原因。',
  '到这里，图上多了一个节点 —— 但还没有任何东西把它<span class="k">串进图</span>。下一幕。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
els.forEach((_, i) => tl.at(3300 + i * 3000, () => {
  els.forEach((e, k) => e.style.opacity = k <= i ? '1' : '.28');
  msg.innerHTML = texts[Math.min(i + 1, 3)];
}));
tl.at(16200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L8-01 · 第 2 跳',
    title='入图：<span class="hl-e">hash set</span> 决定"要不要重算"',
    sub='ggml_build_forward_expand 是入口，真正的拓扑序遍历在 ggml_visit_parents_graph。',
    caption='回顾 L1-03：visited_hash_set 让每个节点只入图一次 —— 这也是"图是 DAG 不是树"的证据。',
    src=SRC_GGML, parts=[(7337, 7339), (7236, 7245)], duration=20000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip e">expand(cur)</span><span class="arrow">-&gt;</span>
    <span class="chip c">visit_parents</span><span class="arrow">-&gt;</span>
    <span class="chip b">hash_find</span><span class="arrow">-&gt;</span>
    <span class="chip a">入 cgraph-&gt;nodes</span>
  </div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:2px">一次 expand 的递归顺序（简化）</div>';
const steps = [
  { n: 'mul_mat 节点（我们这次）', c: 'a' },
  { n: 'src[0] = w（权重，叶子）', c: 'b' },
  { n: 'src[1] = cur（上一层输出）', c: 'c' },
  { n: '…继续向上一层递归', c: 'd' }
];
const els = steps.map(s => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:9.5px' });
  e.innerHTML = '<span style="color:var(--' + s.c + ')">' + U.esc(s.n) + '</span>';
  left.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.28');

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">先问"见过没有"</div>' +
  '<div class="cb"><span class="cm" style="margin:0">ggml_hash_find(&amp;cgraph-&gt;visited_hash_set, node)</span>（7241）。' +
  'used 位已置 → 直接返回，不再往下走（7244）。共用同一个权重的两个节点不会把它算两遍。</div></div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">叶子与节点分开存</div>' +
  '<div class="cb">op == NONE 且不是 PARAM 的进 <span class="cm" style="margin:0">cgraph-&gt;leafs</span>，' +
  '其余进 <span class="cm" style="margin:0">cgraph-&gt;nodes</span>。' +
  '后端只需要遍历 nodes（一次 switch）。</div></div>' +
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">顺序 = 拓扑序</div>' +
  '<div class="cb">先递归父节点再记录自己，所以 nodes[] 里"生产者一定排在消费者前面" —— ' +
  '第 9 幕 CPU 线程按这个顺序扫一遍即可，不需要再做依赖检查。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">ggml_build_forward_expand(gf, cur)</span>（7337）只转发到 <span class="v">ggml_build_forward_impl</span>。',
  'DFS 的每一站都先查 hash set：<span class="k">没见过 → 打标记 + 继续递归；见过 → 立刻返回</span>。',
  '我们的 mul_mat 节点就是这样入图的：src[1] 一路递归回上一层的输出，src[0] 递归到权重叶子。',
  '入图结束后，<span class="v">cgraph-&gt;nodes</span> 里就是一张排好序的待执行清单 —— 还没有后端的概念。'
];
els.forEach((_, i) => tl.at(700 + i * 3600, () => {
  els.forEach((e, k) => e.style.opacity = k <= i ? '1' : '.28');
  msg.innerHTML = texts[Math.min(i, 3)];
}));
tl.at(16800, () => { msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L8-01 · 第 3 跳',
    title='执行入口：<span class="hl-b">decode</span> 的三件事',
    sub='C API 只转发；llama_context::decode 决定"重建图 / 分配 / 计算"。',
    caption='回顾 L2-07：K/V 写缓存、ubatch 准备都在这一层；本幕只看与这条链有关的三行。',
    src=SRC_CTX, parts=[(4326, 4331), (1433, 1454), (2583, 2590)], duration=22000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '① 建图（1425）', m: 'model.build_graph(gparams)',
    b: '图的形状只取决于 ubatch 参数；<br>参数没变就复用上一张图（1415 n_reused++）' },
  { c: 'b', t: '② 分配（1435）', m: 'ggml_backend_sched_alloc_graph',
    b: '对调度器说"这张图要跑了"：<br>切分 + 归属 + buffer 一次做完' },
  { c: 'c', t: '③ 计算（1454 → 2588）', m: 'graph_compute → ..._async',
    b: 'llama_context::graph_compute（2569）设好线程数，<br>交给 ggml_backend_sched_graph_compute_async' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">llama_decode</span>（4326）只有一行转发：<span class="k">ctx-&gt;decode(batch)</span>。',
  '<span class="v">decode</span> 里与这条链有关的就是三个调用点：建图、分配、计算。',
  '注意<span class="k">分配</span>这一步：<span class="v">ggml_backend_sched_alloc_graph</span> 才第一次涉及"后端"。',
  '而 <span class="v">graph_compute</span> 只是设置线程数（2584-2586），真正干活的是下一幕的调度器。'
];
defs.forEach((_, i) => tl.at(700 + i * 4000, () => {
  els.forEach((e, k) => e.style.opacity = k === i ? '1' : '.30');
  msg.innerHTML = texts[i];
}));
tl.at(17400, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L8-01 · ★ 洞察',
    title='★ 判据：<span class="hl-a">权重住在哪个 buffer</span>，op 就归哪个后端',
    sub='调度器按固定优先级给每个节点找归属；我们这次命中的是"输入里有权重"这一条。',
    caption='回顾 L4-02：优先级四层与 copy 节点的产生都在那一课逐字展开；本幕只把它放回链路。',
    src=SRC_SCHED, parts=[(1992, 2001), (921, 948), (961, 969)], duration=24000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:7px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '① 预分配（923）', m: 'backend_from_buffer', b: '张量自己有 buffer：<br>直接用它的后端（权重走这条）' },
  { c: 'b', t: '② 视图（930）', m: 'view_src-&gt;buffer', b: '视图张量看 view_src：<br>不产生数据，归属必然跟着源' },
  { c: 'c', t: '③ 图输入（945）', m: 'GGML_TENSOR_FLAG_INPUT', b: '输入张量固定给<br>最后一个后端（约定是 CPU）' },
  { c: 'd', t: '④ 输入有权重（967）', m: 'BUFFER_USAGE_WEIGHTS', b: '★ 本次命中：src[0] = wo<br>用权重的后端，除非更高优先级后端愿意接' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:163px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.26');
const msg = wrap.querySelector('#msg');
const texts = [
  '切分发生在分配之前：<span class="v">sched_alloc_graph</span>（1992）第一件事就是 <span class="v">split_graph</span>（2000）。',
  '每个节点先问 <span class="v">backend_id_from_cur</span>（921）：按 ①预分配 → ②视图 → ③图输入 → ④权重 的顺序找。',
  '我们这次落在 ④：<span class="k">src[0] = wo</span> 的 buffer usage 是 WEIGHTS → <span class="v">src_backend_id</span> 就是 wo 所在后端的编号（967-968）。',
  '如果那个后端支持这个 op，就<span class="k">一路跳过</span>准备拿图的邻居；否则会被 copy 节点隔开（L4-02 的产物）。',
  '对照：<span class="v">kqv = ggml_mul_mat(v, kq)</span>（llama-graph.cpp:2715）两个输入都不是权重 → 走到 921 的 <span class="k">-1</span>，改由邻居"带走"（1058 set_if_supported）或按"支持输入最多的后端"分配（pass 3，1210-1246）。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
defs.forEach((_, i) => tl.at(3400 + i * 3400, () => {
  els.forEach((e, k) => e.style.opacity = k <= i ? '1' : '.26');
  msg.innerHTML = texts[Math.min(i + 1, 3)];
}));
tl.at(17800, () => { msg.innerHTML = texts[3]; });
tl.at(20200, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L8-01 · 第 4 跳',
    title='从 split 到接口：<span class="hl-c">backend_id → iface.graph_compute</span>',
    sub='切分把图切成连续段；每段拿到一个后端指针，执行就是一次虚表调用。',
    caption='回顾 L3-01（后端契约）与 L4-04（图执行入口）：本幕是两者的交点。',
    src=SRC_SCHED, parts=[(461, 464), (1646, 1660), (1798, 1802)], duration=20000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip c">split[i]</span><span class="arrow">-&gt;</span>
    <span class="chip b">backend_id</span><span class="arrow">-&gt;</span>
    <span class="chip a">sched-&gt;backends[id]</span><span class="arrow">-&gt;</span>
    <span class="chip d">iface.graph_compute</span>
  </div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin-bottom:2px">split 的两句话（L4-02 第 2 幕）</div>' +
  '<div class="card" style="border-left-color:var(--c)">' +
  '<div class="ct" style="color:var(--c)">这一段的图</div>' +
  '<div class="cb">split-&gt;graph：连续的同后端节点<br>' +
  '<span class="cm" style="margin:0">compute_splits</span> 直接把它交给后端（1799）</div></div>' +
  '<div class="card" style="border-left-color:var(--b)">' +
  '<div class="ct" style="color:var(--b)">这一段要吃什么</div>' +
  '<div class="cb">split-&gt;n_inputs / inputs[]：跨后端的张量<br>' +
  '先 copy，再算（1786-1792）</div></div>';
left.appendChild(U.el('div', { class: 'formula', style: 'font-size:9.5px',
  html: '本次的 mul_mat 与 <span class="k">wo</span> 同属一段 —— 无需 copy' }));

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">唯一的"多态"点</div>' +
  '<div class="cb"><span class="cm" style="margin:0">backend-&gt;iface.graph_compute(backend, cgraph)</span>（463）' +
  '是全链路唯一一次"不知道会跳到哪"的调用。CPU 把它填成 ' +
  '<span class="cm" style="margin:0">ggml_backend_cpu_graph_compute</span>（ggml-cpu.cpp:206），' +
  'CUDA 填成 <span class="cm" style="margin:0">ggml_backend_cuda_graph_compute</span>（ggml-cuda.cu:4853）。</div></div>' +
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">同一张图可能被调多次</div>' +
  '<div class="cb">for 循环按 <span class="cm" style="margin:0">sched-&gt;n_splits</span> 逐段执行（1656）；' +
  '段与段之间如果需要，还会插入同步或事件等待。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">compute_splits</span>（1646）按段循环：CPU 段、GPU 段、再回到 CPU 段，各自调一次后端。',
  '每段的执行体是 <span class="v">ggml_backend_graph_compute_async(split_backend, &amp;split-&gt;graph)</span>（1799）。',
  '<span class="v">ggml_backend_graph_compute_async</span>（461）只有一行：<span class="k">backend-&gt;iface.graph_compute(...)</span>。',
  '到这里，"算哪个 op"这件事已经从链路里消失了 —— 后端拿到的是<span class="k">一整段图</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(4600, () => { msg.innerHTML = texts[1]; });
tl.at(9200, () => { msg.innerHTML = texts[2]; });
tl.at(14000, () => { msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L8-01 · ★ 洞察',
    title='CPU 侧：<span class="hl-d">switch (tensor-&gt;op)</span> → mul_mat → vec_dot',
    sub='大 switch 的判据是 op；vec_dot 的判据是 src0->type。两个字段，两次分派。',
    caption='回顾 L5-01（CPU 分派）与 L5-03（向量化内核）：本幕把这两课接回链路，并给出完整调用链表。',
    src=SRC_CPUC, parts=[(1744, 1756), (1869, 1872), (240, 249), (1165, 1183)], duration=26000,
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:7px;width:100%' });
wrap.innerHTML = `<div class="formula" id="msg" style="font-size:9.5px;padding:5px 8px"></div>
  <div id="tbl"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['跳（函数）', '文件:行号', '判据：谁决定下一步'],
  [['模型 build → build_attn', 'src/models/llama.cpp:169', '层里有 wo 权重 → 走注意力块'],
   ['build_attn → build_lora_mm(wo, cur)', 'src/llama-graph.cpp:2804', 'wo 非空 },
   ['build_lora_mm → ggml_mul_mat(w, cur)', 'src/llama-graph.cpp:1518', '这一次调用只是加节点'],
   ['构造器：断言 + ne[] + op/src', 'ggml/src/ggml.c:3333 / 3348', 'can_mul_mat 的三条 ne 判据'],
   ['ggml_build_forward_expand → visit_parents', 'ggml/src/ggml.c:7337 → 7236', 'visited_hash_set 没见过才继续 DFS'],
   ['llama_decode → llama_context::decode', 'src/llama-context.cpp:4326 → 1704', '图没变就复用（1415 n_reused）'],
   ['sched_alloc_graph → split_graph', 'ggml/src/ggml-backend.cpp:1992 → 1066', '切分在分配之前（2000）'],
   ['backend_id_from_cur（归属）', 'ggml/src/ggml-backend.cpp:921', 'src[0] 的 buffer = WEIGHTS（967）'],
   ['compute_splits → iface.graph_compute', 'ggml/src/ggml-backend.cpp:1799 → 461', 'split-&gt;backend_id（1658）'],
   ['cpu_graph_compute → ggml_graph_compute', 'ggml/src/ggml-cpu/ggml-cpu.cpp:170 → ggml-cpu.c:3399', 'CPU 接口表第 206 行填的就是它'],
   ['compute_forward → mul_mat → one_chunk', 'ggml/src/ggml-cpu/ggml-cpu.c:1744 → 1869 → 1255', 'switch (tensor-&gt;op)'],
   ['vec_dot 内核', 'ggml-cpu.c:1182 → arch/x86/quants.c:701', 'type_traits_cpu[src0-&gt;type].vec_dot'],
   ['（对照）CUDA 同一跳', 'ggml/src/ggml-cuda/ggml-cuda.cu:2259 → 1823', '同一个 iface 槽，填的是 CUDA 实现']],
  { monoCols: [1] });
t.el.style.fontSize = '9.5px';
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
function hi(a, b) { rows.forEach(function (r, k) { r.className = (k >= a && k <= b) ? 'on' : ''; }); }
const texts = [
  '<span class="k">第 1 段 · 建图</span>：模型代码 → build_attn → build_lora_mm → ggml_mul_mat。图上多了一个 MUL_MAT 节点。',
  '<span class="k">第 2 段 · 构造与入图</span>：can_mul_mat 决定形状；hash set 决定它进图一次、拓扑序正确。',
  '<span class="k">第 3 段 · 调度</span>：切分 → 归属（权重在哪）→ 分配。这一步之后"归哪个后端"就定了。',
  '<span class="k">第 4 段 · 内核</span>：接口槽 → CPU 入口 → switch(op) → mul_mat → vec_dot(src0->type)。',
  '一句话：这条链上每一个箭头，都是<span class="v">某个字段的一次判断</span>。'
];
tl.at(700, () => { hi(0, 2); msg.innerHTML = texts[0]; });
tl.at(6300, () => { hi(3, 4); msg.innerHTML = texts[1]; });
tl.at(11800, () => { hi(5, 8); msg.innerHTML = texts[2]; });
tl.at(17300, () => { hi(9, 12); msg.innerHTML = texts[3]; });
tl.at(22600, () => { hi(0, 12); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 10 幕

L.scene(
    kicker='L8-01 · 收束',
    title='把这一课压成一张表：<span class="hl-a">每一层负责哪一跳</span>',
    sub='L1~L8 各管一段；回到起点那一行，这条链就闭合了。',
    caption='下一课 L8-02：用真实模型参数走一遍 offload 决策，看这条链在 CPU/GPU 之间怎么切。',
    src=SRC_GRAPH, parts=[(1514, 1518)], duration=24000,
    mark_src=[1518],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['层', '在这条链路上负责哪一跳', '本课引用的位置'],
  [['L1 算子的表示', '节点身份（op）与输出形状（ne[]）的定义', 'ggml/src/ggml.c:3333-3356'],
   ['L2 从模型到图', '把权重接到图上：build_attn → build_lora_mm', 'models/llama.cpp:169 / llama-graph.cpp:2804'],
   ['L3 后端发现与注册', '接口表把 graph_compute 槽绑到具体实现', 'ggml-backend.cpp:461 → ggml-cpu.cpp:206'],
   ['L4 内存与调度', '切分 + 归属 + buffer：决定跑在哪个后端', 'ggml-backend.cpp:1992 / 921'],
   ['L5 CPU 后端执行', '大 switch 分派 + type_traits 选 vec_dot', 'ggml-cpu.c:1744 / 1182'],
   ['L6 GPU 后端执行', '同一跳的 CUDA 实现（契约不变）', 'ggml-cuda.cu:2259 → 1823'],
   ['L7 NPU 与加速器', '同一跳的厂商实现：CANN / Hexagon / OpenVINO …', '不引用源码：见 L7-01 ~ L7-06'],
   ['L8 端到端', '把各跳串成可复述的链，并给出每跳判据', '本课（上表 13 行）']],
  { monoCols: [2] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '不看代码，复述 <span class="mono">build_attn</span> 里输出投影那次 ' +
  '<span class="mono">ggml_mul_mat</span> 的五跳，并说出每一跳的判据。',
  '<b>① 构造</b>（<span class="mono">ggml/src/ggml.c:3341</span>）：' +
  '<span class="mono">ggml_can_mul_mat</span> 判三条 ne 关系（3336-3338）→ 输出 ' +
  '<span class="mono">ne = {a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3]}</span>（3348），' +
  '<span class="mono">op = GGML_OP_MUL_MAT</span>（3351）。<br>' +
  '<b>② 入图</b>（<span class="mono">ggml.c:7337 → 7236</span>）：' +
  '<span class="mono">visited_hash_set</span> 没见过才继续 DFS（7241-7244）。<br>' +
  '<b>③ 调度</b>（<span class="mono">ggml-backend.cpp:2000 → 1066 → 921</span>）：' +
  '<span class="mono">src[0] = wo</span> 的 buffer usage 是 ' +
  '<span class="mono">GGML_BACKEND_BUFFER_USAGE_WEIGHTS</span>（967）→ ' +
  '用 <span class="mono">ggml_backend_sched_backend_from_buffer</span> 找"支持该 buffer 类型且支持该 op"的后端（888-898）。<br>' +
  '<b>④ 执行</b>（<span class="mono">ggml-backend.cpp:1799 → 461</span>）：' +
  '<span class="mono">split-&gt;backend_id</span> 选出后端（1658），调 ' +
  '<span class="mono">backend-&gt;iface.graph_compute</span>（463）。<br>' +
  '<b>⑤ 内核</b>（<span class="mono">ggml-cpu.c:1744 → 1869 → 1255 → 1460 → 1182</span>）：' +
  '<span class="mono">switch (tensor-&gt;op)</span> → <span class="mono">case GGML_OP_MUL_MAT</span> → ' +
  '<span class="mono">type_traits_cpu[src0-&gt;type].vec_dot</span>；Q4_0 绑定到 ' +
  '<span class="mono">ggml_vec_dot_q4_0_q8_0</span>（240-243，x86 实现见 ' +
  '<span class="mono">arch/x86/quants.c:701</span>）。<br>' +
  '<b>反例</b>：<span class="mono">kqv = ggml_mul_mat(v, kq)</span>（llama-graph.cpp:2715）两个输入都不是权重，' +
  '第 ③ 跳走的是"邻居/最多支持输入"路径 —— 判据不同，落点就可能不同。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '八层，各管一段；这一课把它们串成了一条线。',
  '<span class="k">L1/L2</span> 负责"建"：op、形状、src[] 与装配线。',
  '<span class="k">L3/L4</span> 负责"分"：接口契约、切分、归属、buffer。',
  '<span class="k">L5/L6/L7</span> 负责"算"：同一个 op，三套实现。',
  '<span class="k">L8</span> 负责"串"：把判据连起来，才叫读懂了这条链。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2200 + i * 2200, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(1 + Math.floor(i / 2), 3)];
}));
tl.at(20000, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[4] +
    ' ｜ 全套课件门户：<a href="../../index.html">index.html</a>' +
    ' ｜ 下一课：<a href="../L8-02-offload-decision/index.html">L8-02 ★ offload 决策实战</a>';
});
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、起点：build_attn 里的输出投影',
    '模型侧（`src/models/llama.cpp`）在第 169 行把本层的 `wo / wo_b / wo_s` 与 Q/K/V 交给 '
    '`build_attn`；`build_attn`（`src/llama-graph.cpp:2766`）在第 2804 行调用 '
    '`build_lora_mm(wo, cur, wo_s)`；`build_lora_mm`（第 1514 行）的本体就是第 1518 行的 '
    '`ggml_mul_mat(ctx0, w, cur)`。\n\n'
    '**这一串调用里没有任何计算**：它只是往 `ctx0` 里加了一个 `GGML_OP_MUL_MAT` 节点。'
    '下面是这三处的逐字引用（两个文件）。',
    src=SRC_MODEL, parts=[(169, 172), (246, 246)], lang='cpp')

L.section(
    '二、这一次 mul_mat 的构造器',
    '`ggml_mul_mat` 只做四件事：\n\n'
    '1. 断言 `ggml_can_mul_mat(a, b)`（三条 `ne` 关系）与 `!ggml_is_transposed(a)`；\n'
    '2. 算输出形状 `ne = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] }`；\n'
    '3. `ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)` —— **输出恒为 F32**；\n'
    '4. 填 `op = GGML_OP_MUL_MAT` 与 `src[0] = a`、`src[1] = b`。\n\n'
    '第 4 步就是 L1-02 的结论："输入个数由构造器赋了几个 `src[i]` 决定"—— 这里赋了两个。',
    src=SRC_GRAPH, parts=[(2803, 2805), (1514, 1518)], lang='cpp')

L.section(
    '三、入图：拓扑序遍历与 hash set',
    '`ggml_build_forward_expand`（`ggml/src/ggml.c:7337`）只是入口，转手就调 '
    '`ggml_build_forward_impl` → `ggml_visit_parents_graph`（第 7236 行）。\n\n'
    '递归的每一站先做一次 `ggml_hash_find(&cgraph->visited_hash_set, node)`：'
    '`used` 位已置就立即返回（第 7244 行）。这就是 L1-03 里"图是 DAG、公共子图只算一次"的实现。'
    '本篇的 mul_mat 节点正是由 `build_attn` 内部的局部 expand（`llama-graph.cpp:2735`、`2783`）'
    '与模型 build 函数末尾的 `ggml_build_forward_expand(gf, cur)`（`src/models/llama.cpp:246`）'
    '两类调用拉进图的。',
    src=SRC_GGML, parts=[(7337, 7339), (7236, 7245)], lang='c')

L.section(
    '四、进入执行：decode 的三个调用点',
    'C API `llama_decode`（`src/llama-context.cpp:4326`）只有一行转发。'
    '`llama_context::decode`（第 1704 行）里与本链路有关的三个调用点是：'
    '建图（1425 `model.build_graph`）、分配（1435 `ggml_backend_sched_alloc_graph`）、'
    '计算（1454 `graph_compute` → 2588 `ggml_backend_sched_graph_compute_async`）。\n\n'
    '注意第 1415 行的 `n_reused++`：图能复用就不重建，但**分配仍然每次都做**（L2-07 的结论）。',
    src=SRC_CTX, parts=[(4326, 4331), (1433, 1454), (2583, 2590)], lang='cpp')

L.section(
    '五、调度器：归属判据与逐段执行',
    '`ggml_backend_sched_alloc_graph`（`ggml/src/ggml-backend.cpp:1992`）第一件事是 '
    '`ggml_backend_sched_split_graph`（2000）。切分时每个节点问 '
    '`ggml_backend_sched_backend_id_from_cur`（921），判据按固定优先级：'
    '预分配 buffer（923）> 视图的 buffer（930）> 图输入（945）> **输入里有权重**（967）。\n\n'
    '本篇的 mul_mat 命中最后一条：`src[0] = wo` 的 buffer usage 是 `WEIGHTS`，'
    '于是归属就取 `ggml_backend_sched_backend_from_buffer`（888）找到的那个后端。'
    '执行阶段 `ggml_backend_sched_compute_splits`（1646）逐段调用 '
    '`ggml_backend_graph_compute_async`（1799）→ `backend->iface.graph_compute`（463）。\n\n'
    '对照（同一段代码里的第二个 mul_mat）：`kqv = ggml_mul_mat(ctx0, v, kq)`'
    '（`src/llama-graph.cpp:2715`）两个输入都不是权重，`backend_id_from_cur` 对它返回 `-1`，'
    '改由邻居带走（`ggml_backend_sched_set_if_supported`，1058）或在 pass 3（1210-1246）'
    '按"支持输入最多的后端"分配 —— 判据不同，落点就可能不同。',
    src=SRC_SCHED, parts=[(1992, 2001), (921, 948), (961, 969), (1646, 1660), (1798, 1802),
                          (461, 464)], lang='cpp')

L.section(
    '六、CPU 侧：接口表 → 入口 → 大 switch → vec_dot',
    'CPU 后端的接口表把 `graph_compute` 槽填成 `ggml_backend_cpu_graph_compute`'
    '（`ggml/src/ggml-cpu/ggml-cpu.cpp:206`，函数体在 170）：它先 `ggml_graph_plan` 算 `cplan`，'
    '再调 `ggml_graph_compute`（`ggml/src/ggml-cpu/ggml-cpu.c:3399`）。\n\n'
    '线程主体按 `cgraph->nodes[]` 的顺序逐个调用 `ggml_compute_forward`（第 1744 行），'
    '那里是一个 `switch (tensor->op)`；`GGML_OP_MUL_MAT` 的 case 在第 1869 行，'
    '指向 `ggml_compute_forward_mul_mat`（1255）→ `..._one_chunk`（1165）。\n\n'
    '内核的选择在 `..._one_chunk` 里：`type_traits_cpu[type].vec_dot`（1182）。'
    '以 Q4_0 权重为例，表里绑定的是 `ggml_vec_dot_q4_0_q8_0`、`vec_dot_type = GGML_TYPE_Q8_0`'
    '（240-243）；该函数按架构由 CMake 从 `arch/<arch>/quants.c` 选一份编译'
    '（x86 的在 `ggml/src/ggml-cpu/arch/x86/quants.c:701`）。',
    src=SRC_CPUCPP, parts=[(170, 191), (193, 210)], lang='cpp')

L.section(
    '七、CPU 分派与内核选择的逐字引用',
    '第一段是分派入口（`ggml_compute_forward`），第二段是 `GGML_OP_MUL_MAT` 的 case，'
    '第三段是 Q4_0 在 `type_traits_cpu[]` 里的条目，第四段是 `..._one_chunk` 里 '
    '`vec_dot` 的绑定处（真正调用在 1244 行）。',
    src=SRC_CPUC, parts=[(1744, 1756), (1869, 1872), (240, 249), (1165, 1183)], lang='c')

L.section(
    '八、同一跳在 CUDA 侧（对照）',
    '同一个 `iface.graph_compute` 槽，在 CUDA 后端里填的是 '
    '`ggml_backend_cuda_graph_compute`（`ggml/src/ggml-cuda/ggml-cuda.cu:4421`，接口表见 4853）。'
    '它内部的 `ggml_cuda_compute_forward`（2067）同样是一个 `switch (dst->op)`，'
    '`GGML_OP_MUL_MAT` 的 case 在第 2259 行，指向 `ggml_cuda_mul_mat`（1823）。\n\n'
    '**两次分派的形状完全一样**：一次按 `op` 找到函数，再由函数内部按 `src0->type` 与形状'
    '挑具体内核。这就是 L6-01 与本课的交点。',
    src=SRC_CUDA, parts=[(4421, 4428), (2067, 2071), (2256, 2262), (1823, 1830),
                         (4850, 4854)], lang='cpp')

L.section(
    '九、完整调用链表（每一步的判据）',
    '把这一课压成一张表。左列是跳，中间是位置，右列是"谁决定下一步走哪"。\n\n'
    '| # | 跳（函数） | 文件:行号 | 判据 |\n|---|---|---|---|\n'
    '| 0 | 模型 build → `build_attn` | `src/models/llama.cpp:169` | 层里有 `wo` → 走注意力块 |\n'
    '| 1 | `build_attn` → `build_lora_mm(wo, cur)` | `src/llama-graph.cpp:2804` | `wo` 非空 |\n'
    '| 2 | `build_lora_mm` → `ggml_mul_mat(w, cur)` | `src/llama-graph.cpp:1518` | 只是加一个节点 |\n'
    '| 3 | 构造器：断言 + `ne[]` + `op/src` | `ggml/src/ggml.c:3333 / 3348 / 3351` | `ggml_can_mul_mat` 三条 ne 判据 |\n'
    '| 4 | `ggml_build_forward_expand` → `ggml_visit_parents_graph` | `ggml/src/ggml.c:7337 → 7236` | `visited_hash_set` 没见过才继续 |\n'
    '| 5 | `llama_decode` → `llama_context::decode` → `graph_compute` | `src/llama-context.cpp:4326 → 1704 → 2569` | 图没变就复用（1415），但分配照做 |\n'
    '| 6 | `ggml_backend_sched_alloc_graph` → `split_graph` | `ggml/src/ggml-backend.cpp:1992 → 1066` | 切分在分配之前（2000） |\n'
    '| 7 | `ggml_backend_sched_backend_id_from_cur` | `ggml/src/ggml-backend.cpp:921` | `src[0]` 的 buffer usage = WEIGHTS（967） |\n'
    '| 8 | `compute_splits` → `iface.graph_compute` | `ggml/src/ggml-backend.cpp:1799 → 461` | `split->backend_id`（1658） |\n'
    '| 9 | `ggml_backend_cpu_graph_compute` → `ggml_graph_compute` | `ggml-cpu/ggml-cpu.cpp:170` → `ggml-cpu/ggml-cpu.c:3399` | 接口表第 206 行 |\n'
    '| 10 | `ggml_compute_forward` → `..._mul_mat` → `..._one_chunk` | `ggml-cpu/ggml-cpu.c:1744 → 1869 → 1255 → 1165` | `switch (tensor->op)` 与 `case GGML_OP_MUL_MAT`（1869） |\n'
    '| 11 | `vec_dot` 内核 | `ggml-cpu/ggml-cpu.c:1182` → `ggml-cpu/arch/x86/quants.c:701` | `type_traits_cpu[src0->type].vec_dot`（240-243） |\n'
    '| 12 | （对照）CUDA 同一跳 | `ggml-cuda/ggml-cuda.cu:2259 → 1823` | 同一个 iface 槽，另一套实现 |\n\n'
    '表里的每个行号都在本课的逐字引用里出现过（第 0/4/5/9 跳的调用点除外 —— 它们分别是 '
    '`src/models/llama.cpp:169`、`src/models/llama.cpp:246`、`src/llama-context.cpp:1415`、'
    '`ggml/src/ggml-cpu/ggml-cpu.cpp:206`，也都在这几段引用区间内）。')

L.footnote_add(
    '本课覆盖 **9 个源文件**，全部计入覆盖率。其中 `src/llama-graph.cpp` 是计划里本课的主文件；'
    '按"这条链路"追加了 3 个：`ggml/src/ggml.c`（`ggml_mul_mat` 构造器 + '
    '`ggml_build_forward_expand`/拓扑序）、`ggml/src/ggml-backend.cpp`（切分、归属与调度执行）、'
    '`ggml/src/ggml-cpu/ggml-cpu.c`（CPU 分派与 `vec_dot` 选择）。')
L.footnote_add(
    '另外 5 个文件是这条链路的相邻跳，本课只取"这一跳"的几行：'
    '`src/models/llama.cpp`（`build_attn` 的调用点与最终 expand）、'
    '`src/llama-context.cpp`（decode 与 graph_compute）、'
    '`ggml/src/ggml-cpu/ggml-cpu.cpp`（CPU 接口表与入口）、'
    '`ggml/src/ggml-cuda/ggml-cuda.cu`（同一跳的 CUDA 实现，作对照）、'
    '`ggml/src/ggml-cpu/arch/x86/quants.c`（x86 版 `ggml_vec_dot_q4_0_q8_0`）。'
    '这些文件各自的专课会逐字展开：L2-06 / L2-07 / L5-01 / L6-01 / L5-03。')
L.footnote_add(
    '说明：`ggml/src/ggml-cpu/arch/<arch>/quants.c` 的同名函数由 CMake 按架构选一份编译'
    '（`ggml/src/ggml-cpu/CMakeLists.txt` 的 `GGML_CPU_SOURCES`），本课引用的是 x86 那一份；'
    '换架构时 `type_traits_cpu[]` 里的绑定名不变，实现文件会换。')

L.prereqs('`L7-06`')

L.goal(
    '不看代码，复述一次 `ggml_mul_mat` 从 `build_attn` 到 CPU `vec_dot` 的完整链路（对应验收点）；',
    '说出这条链上每一跳的**判据**：`ggml_can_mul_mat` / `visited_hash_set` / '
    '`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` / `iface.graph_compute` / `switch (tensor->op)` / '
    '`type_traits_cpu[src0->type].vec_dot`；',
    '解释为什么同一个 `build_attn` 里的两次 `ggml_mul_mat` 可能落在不同后端（有权重 vs 没有权重）；',
    '拿到任意一个 ggml 算子，能说出它落到 CPU/CUDA 内核要经过哪几个文件、哪几个函数。')

L.conclusion(
    '这条链路的四个世界',
    '一次 `ggml_mul_mat` 穿过四个世界，每个世界只回答一个问题：\n\n'
    '| 世界 | 问题 | 谁回答 |\n|---|---|---|\n'
    '| 建图（L1/L2） | 节点是什么、什么形状 | `ggml_mul_mat` 构造器 |\n'
    '| 入图（L1） | 节点进图了没有、顺序如何 | `ggml_visit_parents_graph` + hash set |\n'
    '| 调度（L3/L4） | 跑在哪个后端、内存谁给 | `ggml_backend_sched_*` |\n'
    '| 内核（L5/L6/L7） | 具体用哪个 kernel | 后端自己的 `switch (op)` |')

L.conclusion(
    '★ 每一跳都有一个判据',
    '把这一课真正要记住的东西压成一行：\n\n'
    '```text\n'
    '能不能乘        -> ggml_can_mul_mat（三条 ne 关系）\n'
    '进图几次        -> visited_hash_set\n'
    '归哪个后端      -> src[0] 的 buffer usage（权重优先）\n'
    '调哪个后端实现  -> backend->iface.graph_compute\n'
    '走哪个 case     -> tensor->op\n'
    '用哪个 kernel   -> type_traits_cpu[src0->type].vec_dot\n'
    '```\n\n'
    '**六个判据，六次"看某个字段"** —— 这就是 llama.cpp 把模型跑到设备上的全部机制骨架。')

L.conclusion(
    '★ 同一个 op，两种归属',
    '`build_attn` 的调用栈里其实有两次 `ggml_mul_mat`：\n\n'
    '- `cur = build_lora_mm(wo, cur, wo_s)`（`llama-graph.cpp:2804` → `1518`）：`src[0] = wo` 是**权重**，'
    '归属由权重所在的后端决定（`ggml-backend.cpp:967`）；\n'
    '- `kqv = ggml_mul_mat(ctx0, v, kq)`（`llama-graph.cpp:2715`）：两个输入都是**激活**，'
    '`backend_id_from_cur` 返回 `-1`，归属改由邻居与"支持输入最多的后端"决定（1058 / 1210-1246）。\n\n'
    '同一个算子、同一张图，判据不同，落点就可能不同 —— 这也解释了 `offload_kqv` 这类开关'
    '为什么能直接把这一段按在 CPU 上（`llama-graph.cpp:2729-2732`）。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
