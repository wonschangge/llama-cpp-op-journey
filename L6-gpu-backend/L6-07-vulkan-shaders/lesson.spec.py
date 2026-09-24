#!/usr/bin/env python3
"""L6-07 · Vulkan 后端：计算着色器 —— 课件 spec。

运行：python3 L6-gpu-backend/L6-07-vulkan-shaders/lesson.spec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

VKSH = 'ggml/src/ggml-vulkan/vulkan-shaders'
GEN  = VKSH + '/vulkan-shaders-gen.cpp'

L = Lesson(
    id='L6-07',
    layer='L6 · GPU 后端执行',
    title='Vulkan 后端：计算着色器',
    codecap='ggml/src/ggml-vulkan/vulkan-shaders/（逐字引用）',
    nav={'prev': {'href': '../L6-06-vulkan-host/index.html', 'label': 'L6-06 Vulkan 主机端'},
         'next': {'href': '../L6-08-metal-host-fusion/index.html', 'label': 'L6-08 Metal 主机端与图融合'}},
)

L.note('**一句话**：Vulkan 后端里，"一个算子"就是一个 `.comp`（GLSL 计算着色器）；'
       '而 `.comp` 本身只是**模板** —— 真正被 `vkCreateComputePipelines` 用掉的是'
       '由 `vulkan-shaders-gen.cpp` 用 `glslc` 针对一组宏定义编译出来的 `.spv`，'
       '再被写成 C 数组编进 `ggml-vulkan` 的静态库里。')
L.note('这一课覆盖 `vulkan-shaders/` 下的**全部 161 个 `.comp`** 加 1 个生成器'
       '（`vulkan-shaders-gen.cpp`），共 162 个源文件。逐字引用的只是少数代表，'
       '其余在 `source.md` 里按族列出。')
L.note('回顾 **L6-06**：那一课讲主机端怎么*使用*这些着色器（pipeline 缓存、descriptor set、'
       'push constants）；本课讲这些着色器**本身**是怎么写的、怎么被造出来的。'
       '回顾 **L1-04**：那一课画的是 CPU 侧量化块的内存布局 —— 本课会看到 GLSL 侧'
       '**另一份、但形状完全相同**的定义。对照 **L6-02**：CUDA 侧同类内核用 `.cu` 模板'
       '实例化，Vulkan 侧用 `.comp` + 宏定义，解决的是同一个问题。')

# --------------------------------------------------------------------- 覆盖声明
# 计划里 L6-07 的全部 162 个文件（`python3 tools/plan_matrix.py --files | awk -F'\t'
# '$1=="L6-07"{print $2}' | wc -l` == 162：161 个 .comp + 1 个 vulkan-shaders-gen.cpp）。
# 逐字引用的只是少数代表（见各幕 src= 与 source.md 各节），这里全部计入覆盖。
L.cover(
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_f32.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq1_m.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq1_s.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_s.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_xs.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_xxs.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq3_s.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq3_xxs.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq4_nl.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq4_xs.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_mxfp4.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_nvfp4.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q1_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_k.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q3_k.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_1.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_k.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_1.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_k.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q6_k.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_q8_0.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq1_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq2_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq1_m.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq1_s.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_s.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xs.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xxs.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_s.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_xxs.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq4_xs.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_nc.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_p021.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q3_k.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q5_k.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq1_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq2_0.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_split_k_reduce.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_cm2.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_1.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_2.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_mask_opt.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/soft_max_back.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large1.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large2.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large3.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/group_norm.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/l2_norm.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/norm.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_back.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_partials.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/rope_multi.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/rope_neox.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/rope_norm.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/rope_vision.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/bfloat16.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat2.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat2_decode_vector.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/float_e2m1.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/float_e4m3.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/acc.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/add.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/add1.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/add_id.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/arange.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/argmax.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/argsort.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/argsort_large.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/col2im_1d.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/concat.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/contig_copy.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/conv2d_dw.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/conv2d_mm.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/conv3d_mm.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/conv_transpose_1d.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/copy.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/copy_from_quant.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose_02.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/count_equal.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/count_experts.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/cross_entropy_loss.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/cross_entropy_loss_back.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/cumsum.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/cumsum_multipass1.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/cumsum_multipass2.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/diag.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/diag_mask_inf.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/div.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_pre.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/fill.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/geglu.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/geglu_erf.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/geglu_quick.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/get_rows.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/get_rows_back.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/get_rows_quant.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/gla.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/im2col.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/im2col_3d.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/log.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/mul.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/multi_add.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/opt_step_adamw.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/opt_step_sgd.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/out_prod.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/pad.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/pad_reflect_1d.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/pool1d.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/pool2d.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/reglu.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/repeat.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/repeat_back.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/roll.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/scale.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/silu_back.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/snake.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/solve_tri.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/ssm_conv.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/ssm_scan.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/sub.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/sum_rows.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/swiglu.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/swiglu_clamp.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/swiglu_oai.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/timestep_embedding.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/topk_argsort.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/topk_moe.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/topk_nary_search.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/topk_radix_select.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/tri.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/unary.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/upscale.comp',
    'ggml/src/ggml-vulkan/vulkan-shaders/wkv6.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/wkv7.comp', 'ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp',
)

L.prereqs('`L6-06`（Vulkan 主机端：pipeline、descriptor set、shader 生成）；'
          '建议对照 `L1-04`（量化块布局）与 `L6-02`（CUDA 侧量化矩阵乘）')

L.goal(
    '说出一个 `.comp` 从生成到被 pipeline 使用经过哪几步（对应验收点）；',
    '解释为什么 `strings_to_spv` 这一层是"变体工厂"——同一个 `.comp` 会被编译成很多个 SPIR-V；',
    '指出 GLSL 侧的 `block_q4_0` 与 CPU 侧 `ggml-common.h` 里的 `block_q4_0` 是两份独立的定义，'
    '并说出这对"新增一个量化类型"意味着什么；',
    '说出 `mul_mm.comp` / `mul_mmq.comp` / `mul_mat_vec_*.comp` 三个族各自负责什么形状的 `GGML_OP_MUL_MAT`；',
    '说明 `dequant_q4_0.comp` 里 64 个线程、32 个元素一块是如何映射到输出的 32 个元素上的。')

# ==================================================================== 第 1 幕

L.scene(
    kicker='L6 · GPU 后端执行',
    title='<span class="hl-a">161 个 .comp</span>：Vulkan 的算子实现全在这里',
    sub='先按名字分族数一遍。族名就是 ggml op 的名字 —— 这是本仓库最一致的命名约定之一。',
    caption='计数命令：ls ggml/src/ggml-vulkan/vulkan-shaders/ 下的 .comp（含 feature-tests/ 子目录）= 161。',
    src=GEN, parts=[(49, 78)], duration=17000,
    mark_src=[49, 54, 58, 61, 73, 77],
    notes_src={49: '类型清单：每个名字都会在每个着色器族里派生出若干变体',
               77: 'bf16 在最后 —— 注释没有写"只能追加"，但这个顺序是稳定约定'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const rows = [
  ['反量化 dequant_*', '26', 'dequant_q4_0.comp', '反量化成 f16（MUL_MAT 的 f16 路径前置）'],
  ['矩阵乘 mul_mat_vec*', '19', 'mul_mat_vec_q4_k.comp', 'GGML_OP_MUL_MAT（小 M）/ MUL_MAT_ID（MoE）'],
  ['矩阵乘 mul_mm* / mul_mmq', '4', 'mul_mm.comp', 'GGML_OP_MUL_MAT（大 M）；split-K 归约'],
  ['注意力 flash_attn*', '8', 'flash_attn.comp', 'GGML_OP_FLASH_ATTN_EXT'],
  ['soft_max*', '5', 'soft_max.comp', 'GGML_OP_SOFT_MAX'],
  ['归一化（rms_norm* / norm / l2_norm / group_norm）', '6', 'rms_norm.comp',
   'RMS_NORM / NORM / L2_NORM / GROUP_NORM'],
  ['rope_*', '4', 'rope_norm.comp', 'GGML_OP_ROPE'],
  ['逐元素 / 激活族', '26', 'unary.comp', 'UNARY / GLU / ADD / MUL / SCALE ...'],
  ['索引 / 排序 / 归约族', '17', 'topk_moe.comp', 'TOP_K / ARGSORT / GET_ROWS / SUM_ROWS / CUMSUM'],
  ['搬运 / 拷贝族', '13', 'copy.comp', 'DUP / CPY / CONT / PAD / REPEAT / ROLL'],
  ['SSM / 线性注意力族', '10', 'ssm_scan.comp', 'SSM_SCAN / SSM_CONV / RWKV_WKV6 / GATED_DELTA_NET'],
  ['卷积 / 池化 / 图像族', '9', 'conv2d_mm.comp', 'CONV_2D / CONV_2D_DW / CONV_3D / IM2COL / POOL_2D'],
  ['feature-tests/*', '7', 'coopmat2.comp', '不是算子：编译期探测设备能力'],
  ['优化器 / 其他', '7', 'opt_step_adamw.comp', 'OPT_STEP_ADAMW / OPT_STEP_SGD / OUT_PROD']
];
const t = U.table(['族（前缀）', '个数', '代表 .comp', '实现哪个 ggml op'], rows,
                  { monoCols: [2] });
wrap.querySelector('#tbl').appendChild(t.el);

const trs = t.body.querySelectorAll('tr');
const msg = wrap.querySelector('#msg');
const texts = [
  '161 个 <span class="v">.comp</span> 按前缀分成 14 个族。' +
  '最大的族是 <span class="k">dequant_*</span>（26）—— 每个量化类型一个。',
  '三个矩阵乘族加起来 23 个：<span class="k">mul_mm*</span> 管大 M、' +
  '<span class="k">mul_mat_vec*</span> 管小 M、<span class="k">mul_mmq</span> 管整数点积。',
  '注意力与归一化：<span class="k">flash_attn*</span>（8）里有 cm1/cm2/decode/split-k 等变体，' +
  '<span class="k">rms_norm</span> 一个文件还要管三种融合。',
  '其余 89 个是逐元素、索引、搬运、SSM、卷积、优化器 —— ' +
  '命名约定一致：<span class="v">族名即 ggml op 名</span>。'
];
const groups = [[0, 1, 2], [3, 4, 5, 6], [7, 8], [9, 10, 11, 12, 13]];
groups.forEach((g, i) => tl.at(700 + i * 3600, () => {
  trs.forEach((r, k) => { r.className = g.indexOf(k) >= 0 ? 'on' : ''; });
  msg.innerHTML = texts[i];
}));
tl.at(15600, () => {
  trs.forEach(r => { r.className = ''; });
  msg.innerHTML = '一句话：<span class="k">一个 ggml op 在 Vulkan 上 = 一个 .comp 族</span>。';
});
'''
)

# ==================================================================== 第 2 幕

L.scene(
    kicker='L6-07 · 核心',
    title='★ 一个 <span class="hl-c">.comp</span> 是<span class="hl-a">模板</span>：<span class="hl-a">string_to_spv</span> 是变体工厂',
    sub='验收点：说出一个 .comp 从生成到被 pipeline 使用经过哪几步。这 16 行是第 2、3 步。',
    caption='注意第 444 行：进程只编译与自己文件名匹配的那个 .comp —— CMake 为每个 .comp 起一个生成器进程。',
    src=GEN, parts=[(436, 451)], duration=20000,
    mark_src=[436, 437, 438, 444, 450],
    notes_src={437: '变体名 = 名字 + 精度后缀（_f16acc / _cm1 / _cm2 / _fp32）+ 自定义 suffix',
               438: '输出路径 = <output_dir>/<变体名>.spv —— 一个变体一个 SPIR-V 文件',
               444: '只编译与输入文件名匹配的变体；实际编译的是 CMake 为每个 .comp 起的那个进程',
               450: '异步扔进线程池：成千上万个变体是并发编译出来的'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '1 · process_shaders()', b: '遍历 (着色器 × 类型 × 变体)：<br>f32/f16 × 量化类型 × coopmat 档', m: 'process_shaders()' },
  { c: 'c', t: '2 · string_to_spv()', b: '拼出<b>变体名</b>与 <b>.spv 输出路径</b>；<br>同一份 .comp 因此有几百个 SPIR-V', m: 'name + ".spv"' },
  { c: 'e', t: '3 · glslc', b: '把每个 <b>-D 宏</b>展开成命令行参数，<br>异步编译出 .spv', m: 'glslc ... -o name.spv' },
  { c: 'd', t: '4 · write_output_files()', b: '读回 .spv 字节，写成<br><b>&lt;name&gt;_data[]</b> 与 <b>&lt;name&gt;_len</b>', m: 'const unsigned char ..._data[]' },
  { c: 'b', t: '5 · 主机端建 pipeline', b: 'createShaderModule(spv_data)<br>→ vkCreateComputePipelines', m: 'ggml_vk_create_pipeline_func()' },
  { c: 'f', t: '6 · 运行时查表 dispatch', b: '按 (权重类型, 列数, workgroup 档)<br>从 pipeline 表里取，然后 dispatch', m: 'ggml_vk_dispatch_pipeline()' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');
const msg = wrap.querySelector('#msg');
const texts = [
  '先记住这六步。左边代码是第 <span class="k">2</span> 步：' +
  '<span class="v">string_to_spv()</span> 把 (着色器, 宏定义) 变成 (变体名, .spv 路径)。',
  '第 <span class="k">1</span> 步在上游：<span class="v">process_shaders()</span> 决定要哪些变体。' +
  '同一份 <span class="v">mul_mm.comp</span> 会被点名几十次。',
  '第 <span class="k">2</span> 步的 <span class="v">name</span> 后缀是真信息：' +
  '<span class="v">_cm1</span>=coopmat、<span class="v">_cm2</span>=coopmat2、' +
  '<span class="v">_f16acc</span>=f16 累加、<span class="v">_fp32</span>=fp32 累加。',
  '第 <span class="k">3</span> 步：<span class="v">defines</span> 里的每一项都变成命令行上的一个宏。' +
  '所以"一个 .comp"其实是一个<b>参数化的家族</b>。',
  '第 <span class="k">4</span> 步：<span class="v">.spv</span> 是二进制，落地成 C 数组才能编进静态库 —— ' +
  '这一步让 Vulkan 后端<b>不需要运行时编译着色器</b>。',
  '第 <span class="k">5、6</span> 步在主机端（<span class="v">ggml-vulkan.cpp</span>，L6-06 的地盘）：' +
  '字节数组 → shader module → compute pipeline → 按类型查表 dispatch。'
];
defs.forEach((_, i) => tl.at(700 + i * 3000, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(18800, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = '六步一句话：<span class="k">.comp --(defines)--> .spv --(C 数组)--> pipeline --(查表)--> dispatch</span>';
});
'''
)

# ==================================================================== 第 3 幕

L.scene(
    kicker='L6-07 · 生成链（第 3 步）',
    title='宏定义变成 <span class="hl-c">glslc</span> 命令行',
    sub='这一步没有任何魔法：一个 define 一个 -D，最后 -o 出 .spv（下面是非 Windows 分支）。',
    caption='上一节 source.md 给出完整的 350-384 行：第 351 行按变体名决定目标环境，第 382 行把每个宏变成一个 -D。',
    src=GEN, parts=[(355, 370)], duration=18000,
    mark_src=[356, 363, 364, 367],
    notes_src={356: '整条命令在这里拼出来：阶段 + 目标环境 + 输入 + 输出',
               363: '四类变体跳过 spirv-opt：注释逐条给了上游 issue 编号',
               364: '-O 只在安全时加 —— 这是一处"优化会改坏着色器"的现场记录',
               367: '同一个函数还顺带产出依赖文件（给构建系统用）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="flow" id="cmd" style="justify-content:center"></div>
  <div class="row" id="cards" style="gap:9px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const cmd = wrap.querySelector('#cmd');
const parts = [
  { t: 'glslc', c: 'a' },
  { t: '-fshader-stage=compute', c: '' },
  { t: 'target-env', c: 'c' },
  { t: '-D DATA_A_Q4_K=1', c: 'e' },
  { t: '-D LOAD_VEC_A=4', c: 'e' },
  { t: '-D FLOAT_TYPE=float16_t', c: 'e' },
  { t: '-D ...', c: 'e' },
  { t: '-o mul_mm_q4_k_f16.spv', c: 'b' }
];
const chips = parts.map(p => { const e = U.chip(p.t, p.c); cmd.appendChild(e); return e; });
chips.forEach(e => e.style.opacity = '.25');

const defs = [
  { c: 'e', t: 'defines 是谁给的？', b: '上游 <span class="cm" style="margin:0">matmul_shaders()</span> 里的 <b>merge_maps(...)</b>：' +
      '基础字典（FLOAT_TYPE 系列）+ 类型键（DATA_A_Q4_K）+ 变体键（MULMAT_QUANT 等）。' },
  { c: 'c', t: '目标环境由变体名决定', b: '名字里含 <b>_cm2</b> 就按 Vulkan 1.3 编译，否则 1.2。' +
      'coopmat2 的着色器用到 1.3 才有的能力。' },
  { c: 'd', t: '依赖文件与调试信息', b: '可选地让 glslc 同时产出 <b>.d</b>（给 CMake 用）与 <b>调试信息</b>；' +
      '都由生成器的编译期开关控制。' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.3');

const msg = wrap.querySelector('#msg');
const texts = [
  '先把命令拼出来：<span class="v">GLSLC</span> 默认就是 <span class="k">glslc</span>（可用命令行覆盖）。',
  '<span class="v">-fshader-stage=compute</span>：所有 .comp 都是 <b>compute</b> 阶段 —— ' +
  '没有顶点/片元着色器。',
  '接下来是 <span class="k">target-env</span>：由变体名里有没有 <span class="v">_cm2</span> 决定。',
  '然后是 <span class="k">一长串 -D</span>：这就是"同一个 .comp 编译出几百个 SPIR-V"的全部机制。',
  '可选开关：<span class="k">依赖文件</span>与<span class="k">调试信息</span>，' +
  '还有去掉 spirv-opt 的那个分支（第 363 行）。',
  '最后 <span class="v">-o &lt;变体名&gt;.spv</span>。命令拼好后交给 <span class="k">execute_command</span> 跑。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
chips.forEach((c, i) => tl.at(2400 + i * 1500, () => {
  chips.forEach((x, k) => { x.style.opacity = k <= i ? '1' : '.25'; });
  msg.innerHTML = texts[Math.min(i, 5)];
}));
tl.at(15600, () => {
  els.forEach(e => e.style.opacity = '1');
  msg.innerHTML = '一句话：<span class="k">变体名 + defines 决定命令行，命令行决定 .spv</span>。';
});
'''
)

# ==================================================================== 第 4 幕

L.scene(
    kicker='L6-07 · 生成链（第 4 步）',
    title='<span class="hl-b">.spv</span> 变成 C 数组：<span class="hl-a">&lt;name&gt;_data[]</span> / <span class="hl-a">&lt;name&gt;_len</span>',
    sub='这一步之后，SPIR-V 就是普通的目标文件内容 —— 跟着 ggml-vulkan 一起被链接。',
    caption='对应的 extern 声明写在 <name>_data[] 头文件里（第 1252-1253 行），主机端 include 它就能用。',
    src=GEN, parts=[(1252, 1270)], duration=17000,
    mark_src=[1252, 1253, 1256, 1261, 1262, 1265],
    notes_src={1252: '头文件里只放 extern 声明：<name>_len 与 <name>_data[]',
               1256: '把上一步 glslc 产出的 .spv 读回内存',
               1261: '字节数写成 <name>_len',
               1262: '整个 SPIR-V 二进制逐字节写成 C 数组 <name>_data[]'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const dia = wrap.querySelector('#dia');

const left = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--c)' });
left.innerHTML = '<div class="ct" style="color:var(--c)">磁盘上的中间产物</div>' +
  '<div class="cb">每个变体一个文件：<br>' +
  '<span class="cm" style="margin:0">&lt;output_dir&gt;/&lt;变体名&gt;.spv</span><br>' +
  '名字里带着类型与精度后缀，例如某个 <b>mul_mat_vec_q4_k</b> 变体。</div>' +
  '<div class="cm" id="spv" style="margin-top:5px">.spv = SPIR-V 二进制</div>';

const right = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--b)' });
right.innerHTML = '<div class="ct" style="color:var(--b)">生成的 C 源文件</div>' +
  '<div class="cb">每个变体两行：<br>' +
  '<span class="cm" style="margin:0">const uint64_t &lt;name&gt;_len = ...;</span><br>' +
  '<span class="cm" style="margin:0">const unsigned char &lt;name&gt;_data[N] = {...};</span><br>' +
  '编进静态库，<b>不需要运行时编译着色器</b>。</div>' +
  '<div class="cm" id="arr" style="margin-top:5px">逐字节写成 0x.. 列表</div>';

dia.appendChild(left); dia.appendChild(U.arrow('->')); dia.appendChild(right);
left.style.opacity = '.35'; right.style.opacity = '.35';

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">write_output_files()</span> 遍历 <span class="k">shader_fnames</span> —— ' +
  '第 2 幕里 <span class="v">string_to_spv()</span> 每登记一个变体，这里就产出一对符号。',
  '第 <span class="k">1256</span> 行把 <span class="v">.spv</span> 读回来；' +
  '<span class="v">read_binary_file()</span> 读的是上一步 glslc 的产物。',
  '第 <span class="k">1261-1262</span> 行写出两个符号：<span class="v">&lt;name&gt;_len</span> 与 ' +
  '<span class="v">&lt;name&gt;_data[]</span>。',
  '头文件（第 1252-1253 行）只放 <span class="k">extern 声明</span>：' +
  '主机端 <span class="v">#include</span> 它就能引用到全部变体。',
  '这就是第 2 幕第 5 步的输入：<span class="v">ggml_vk_create_pipeline_func()</span> 收到 ' +
  '<span class="v">spv_size</span> 与 <span class="v">spv_data</span>，建 shader module。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
tl.at(3600, () => { left.style.opacity = '1'; msg.innerHTML = texts[1]; });
tl.at(6600, () => { msg.innerHTML = texts[2]; });
tl.at(9600, () => { right.style.opacity = '1'; msg.innerHTML = texts[3]; });
tl.at(12600, () => { msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== 第 5 幕

L.scene(
    kicker='L6-07 · 核心',
    title='★ 量化块定义：<span class="hl-a">GLSL</span> 与 <span class="hl-e">C</span> 是<span class="hl-d">两份</span>定义',
    sub='形状完全一样，但写在两个文件里、由两套代码各自维护 —— 这是本课最值得记住的一件事。',
    caption='对照 source.md 第九节：CPU 侧 ggml-common.h 的 block_q4_0 逐字引用。',
    src=VKSH + '/types.glsl', parts=[(56, 77)], duration=19000,
    mark_src=[56, 57, 59, 61, 62, 66, 67, 71, 74, 75],
    notes_src={56: '块大小 32：与 CPU 的 QK4_0 完全一致',
               59: 'GLSL 的块结构体：一个 f16 scale + 16 字节 nibble',
               61: 'float16_t 在这里是"存储类型"，不是计算精度',
               66: 'packed16 视图：同一块内存，按 uint16 读，方便一次取 4 个 nibble',
               70: '这一组宏把"类型号"变成 QUANT_K / A_TYPE，供整个着色器复用'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="dia" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const dia = wrap.querySelector('#dia');

const glsl = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--a)' });
glsl.innerHTML = '<div class="ct" style="color:var(--a)">GLSL 侧 · types.glsl</div>' +
  '<div class="cb">' +
  '<span class="cm" style="margin:0">struct block_q4_0 {</span><br>' +
  '<span class="cm" style="margin:0">&nbsp;&nbsp;float16_t d;</span><br>' +
  '<span class="cm" style="margin:0">&nbsp;&nbsp;uint8_t qs[16];</span><br>' +
  '<span class="cm" style="margin:0">};</span><br>' +
  '<span class="cm" style="margin:0">#define QUANT_K_Q4_0 32</span></div>' +
  '<div class="cb" style="margin-top:5px">被 <b>所有</b> Vulkan 着色器 include。</div>';

const cside = U.el('div', { class: 'card', style: 'width:300px;border-left-color:var(--e)' });
cside.innerHTML = '<div class="ct" style="color:var(--e)">C 侧 · ggml-common.h（L1-04）</div>' +
  '<div class="cb">' +
  '<span class="cm" style="margin:0">typedef struct {</span><br>' +
  '<span class="cm" style="margin:0">&nbsp;&nbsp;ggml_half d;</span><br>' +
  '<span class="cm" style="margin:0">&nbsp;&nbsp;uint8_t qs[QK4_0 / 2];</span><br>' +
  '<span class="cm" style="margin:0">} block_q4_0;</span><br>' +
  '<span class="cm" style="margin:0">#define QK4_0 32</span></div>' +
  '<div class="cb" style="margin-top:5px">CPU 内核、CUDA、Metal……各自还有一份。</div>';

dia.appendChild(glsl); dia.appendChild(U.arrow('=')); dia.appendChild(cside);
glsl.style.opacity = '.35'; cside.style.opacity = '.35';

const msg = wrap.querySelector('#msg');
const texts = [
  '<span class="v">types.glsl</span> 是 GLSL 侧的"量化块字典"：每个类型一个 struct。',
  '<span class="k">Q4_0</span>：32 个元素一块，块内 = 1 个 f16 scale + 16 字节 nibble。' +
  '与左边 C 侧的字段<b>逐个对应</b>。',
  '<span class="k">packed16</span> 视图不是新数据：同一块内存换个读法，' +
  '让 <span class="v">dequantize4()</span> 一次取 4 个 nibble。',
  '第 70-77 行的 <span class="v">#if defined(DATA_A_Q4_0)</span> 是<b>编译期</b>选择：' +
  '一个着色器只认识一种 A 类型。',
  '结论：<span class="k">新增一个量化类型 = 改多处</span> —— ' +
  'CPU 块定义、GLSL 块定义、dequant 着色器、生成器的类型清单，以及每个后端各自的 kernel。'
];
tl.at(600, () => { glsl.style.opacity = '1'; msg.innerHTML = texts[0]; });
tl.at(3400, () => { msg.innerHTML = texts[1]; });
tl.at(6400, () => { msg.innerHTML = texts[2]; });
tl.at(9400, () => { msg.innerHTML = texts[3]; });
tl.at(12400, () => { cside.style.opacity = '1'; msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== 第 6 幕

L.scene(
    kicker='L6-07 · 反量化族（26 个）',
    title='<span class="hl-c">dequant_q4_0.comp</span>：一个反量化内核的全部 30 行',
    sub='最大的族、最短的文件。看懂这 30 行，其余的 dequant_*.comp 都是同一个骨架换个块布局。',
    caption='回顾 L1-04：Q4_0 的 32 个权重 = 1 个 f16 scale + 32 个 4-bit。',
    src=VKSH + '/dequant_q4_0.comp', parts=[(1, 30)], duration=20000,
    mark_src=[5, 7, 8, 11, 13, 14, 16, 17, 24, 27, 28],
    notes_src={5: '256 个线程：每 64 个线程负责一个 block，于是一个 workgroup 处理 4 个 block',
               7: 'binding 0 读量化块数组，binding 1 写 f16 输出',
               11: 'i = 本 workgroup 的第几个 block（每批 4 个）',
               13: 'tid 在 block 内的编号 0..63',
               14: 'il ∈ {0,1}：决定从 qs 的哪 8 个字节取值（0..7 还是 8..15）',
               16: 'ib：全局块号；第 17 行越界就返回',
               24: '每块一个 f16 scale，转成 float 后乘在结果上',
               27: '低 nibble 减 8 —— 这是 Q4_0 的零点（对照 L1-04）',
               28: '高 nibble 同理：同一字节的另一半'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="top" style="gap:9px;align-items:center"></div>
  <div class="col" id="rows" style="gap:6px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const top = wrap.querySelector('#top');
top.innerHTML = '<span class="chip c">1 个 block_q4_0</span>' +
  '<span class="cm" style="margin:0">d: f16</span>' +
  '<span class="cm" style="margin:0">qs[16]</span>' +
  '<span class="arrow">-></span>' +
  '<span class="chip b">32 个 f16</span>' +
  '<span class="cm" style="margin:0">data_b[b_idx + 0..31]</span>';

const rows = wrap.querySelector('#rows');
function mkRow(label, cls) {
  const r = U.el('div', { class: 'row', style: 'gap:4px;align-items:center' });
  r.appendChild(U.el('span', { class: 'cm', style: 'margin:0;width:132px;font-size:9px', text: label }));
  const cells = [];
  for (let i = 0; i < 16; i++) {
    const c = U.el('div', { style: 'width:24px;height:17px;border:1px solid var(--border);border-radius:2px;' +
      'background:#10151b;display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--dim)' });
    c.textContent = '--';
    r.appendChild(c); cells.push(c);
  }
  rows.appendChild(r);
  return { r: r, cells: cells, cls: cls };
}
const lo = mkRow('+l  <- 低 nibble', 'a');
const hi = mkRow('+l+16  <- 高 nibble', 'e');
lo.r.style.opacity = '.35'; hi.r.style.opacity = '.35';

const msg = wrap.querySelector('#msg');
const texts = [
  '输入是一个 <span class="k">block_q4_0</span>：一个 f16 scale + 16 字节（32 个 nibble）。' +
  '输出是 32 个 f16。',
  '第 <span class="v">11</span> 行：一个 workgroup 认领 <b>4 个块</b>，' +
  '<span class="v">tid/64</span> 决定自己管哪一个；第 <span class="v">13-14</span> 行再把 64 个线程对半分。',
  '第 <span class="v">27-28</span> 行每次 unroll 写<b>两个</b>元素：' +
  '<span class="v">+l</span> 取低 4 位、<span class="v">+l+16</span> 取高 4 位 —— ' +
  '所以两行分别是输出的前 16 个与后 16 个。',
  '<span class="v">il=0</span> 的线程读 <span class="v">qs[0..7]</span>（<span class="v">q_idx=0</span>），' +
  '填两行的<b>前 8 格</b>。',
  '<span class="v">il=1</span> 的线程读 <span class="v">qs[8..15]</span>（<span class="v">q_idx=8</span>），' +
  '填两行的<b>后 8 格</b> —— 32 个元素这才齐。',
  '一句话：<span class="k">dequant_*.comp 只有块布局不同</span>，骨架完全一致。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
tl.at(3400, () => { msg.innerHTML = texts[1]; lo.r.style.opacity = '1'; hi.r.style.opacity = '1'; });
tl.at(6400, () => {
  lo.cells.forEach(c => { c.textContent = '?'; });
  hi.cells.forEach(c => { c.textContent = '?'; });
  msg.innerHTML = texts[2];
});
tl.at(9400, () => {
  lo.cells.slice(0, 8).forEach((c, i) => { c.style.background = 'rgba(88,166,255,.20)'; c.style.borderColor = 'var(--a)'; c.textContent = 'q' + i; });
  hi.cells.slice(0, 8).forEach((c, i) => { c.style.background = 'rgba(88,166,255,.20)'; c.style.borderColor = 'var(--a)'; c.textContent = 'q' + i; });
  msg.innerHTML = texts[3];
});
tl.at(12400, () => {
  lo.cells.slice(8).forEach((c, i) => { c.style.background = 'rgba(247,120,186,.20)'; c.style.borderColor = 'var(--e)'; c.textContent = 'q' + (i + 8); });
  hi.cells.slice(8).forEach((c, i) => { c.style.background = 'rgba(247,120,186,.20)'; c.style.borderColor = 'var(--e)'; c.textContent = 'q' + (i + 8); });
  msg.innerHTML = texts[4];
});
tl.at(16400, () => {
  lo.cells.forEach(c => { c.style.background = 'rgba(63,185,80,.20)'; c.style.borderColor = 'var(--b)'; });
  hi.cells.forEach(c => { c.style.background = 'rgba(63,185,80,.20)'; c.style.borderColor = 'var(--b)'; });
  msg.innerHTML = texts[5];
});
'''
)

# ==================================================================== 第 7 幕

L.scene(
    kicker='L6-07 · 矩阵乘族（23 个）',
    title='<span class="hl-b">mul_mmq.comp</span>：B 侧先量化成 <span class="hl-a">q8_1</span>，再整数点积',
    sub='三个矩阵乘族分工不同：大 M 走 mul_mm，小 M 走 mul_mat_vec，整数点积走 mul_mmq。',
    caption='B 侧的 q8_1 由 quantize_q8_1.comp 现场算出，再喂给 mul_mmq —— 权重始终是量化态，只有激活被压成 int8。',
    src=VKSH + '/mul_mmq.comp', parts=[(26, 41)], duration=18000,
    mark_src=[26, 28, 30, 33, 35, 36, 39, 40],
    notes_src={26: 'workgroup 大小也是 spec constant：同一个 SPIR-V 服务多种设备',
               35: 'B 侧不是 f32/f16，而是 4 个 q8_1 块打包成的 128 字节块',
               39: 'MUL_MAT_ID（MoE 的按专家矩阵乘）多两个 binding：专家 id 与计数'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'mul_mm.comp', b: '大 M（prompt / 训练式批量）：<br>block 级分块 + 共享内存，' +
      '非 coopmat 时每种 A 类型<b>单独编译</b>。', m: 'GGML_OP_MUL_MAT（大 M）' },
  { c: 'c', t: 'mul_mat_vec_q4_k.comp', b: '小 M（逐 token 解码，最常见）：<br>' +
      '一个 workgroup 算一行，K-quant 用 super-block。', m: 'GGML_OP_MUL_MAT / MUL_MAT_ID' },
  { c: 'e', t: 'mul_mmq.comp', b: '整数点积路径：<br>B 量化成 q8_1，A 保持量化，' +
      '用 <b>dot</b> 指令累加。', m: 'GGML_OP_MUL_MAT（量化×量化）' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d, { style: 'width:222px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.28');

const msg = wrap.querySelector('#msg');
const texts = [
  '先看左边的 binding：<span class="k">A</span> 是量化权重（有 packed16 / packed32 视图），' +
  '<span class="k">B</span> 是 <span class="v">block_q8_1_x4_packed128</span>，<span class="k">D</span> 是输出。',
  '为什么要 q8_1？因为这条路径用的是 <span class="k">整数点积</span>（第 7 行的扩展）：' +
  'A 的 nibble 与 B 的 int8 直接做 <b>dot</b>，比反量化成浮点再 FMA 更省。',
  '<span class="v">B</span> 不是权重 —— 是激活。权重（A）保持量化态不被展开，' +
  '这是 Vulkan 后端省显存/带宽的关键。',
  '三个族的<b>分工边界</b>不同：<span class="v">mul_mat_vec_*</span> 的 pipeline 表按' +
  '<b>列数</b>索引（source.md 第七节：<span class="v">[dmmv_wg][a_type][num_cols-1]</span>），' +
  '<span class="v">mul_mm*</span> 按 M/N 分块，<span class="v">mul_mmq</span> 走整数点积。',
  '<span class="v">MUL_MAT_ID</span> 分支（第 38-41 行）多两个 binding：' +
  '专家 id 与每专家行数 —— MoE 的 <span class="v">GGML_OP_MUL_MAT_ID</span> 就落在这里。'
];
els.forEach((e, i) => tl.at(700 + i * 3200, () => {
  els.forEach((x, k) => { x.style.opacity = k === i ? '1' : '.28'; });
  msg.innerHTML = texts[i];
}));
tl.at(11500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
tl.at(14800, () => { msg.innerHTML = texts[4]; });
'''
)

# ==================================================================== 第 8 幕

L.scene(
    kicker='L6-07 · K-quant 的写法',
    title='<span class="hl-d">mul_mat_vec_q4_k.comp</span>：6-bit scale 是<span class="hl-a">位运算</span>拼出来的',
    sub='K-quant 的 super-block 有 256 个元素，8 个 6-bit scale 被打包进 12 字节 —— 所以解包就是移位与掩码。',
    caption='对照 L1-04：Q4_K 的块是 dm(2×f16) + scales[12] + qs[128] = 144 字节 / 256 个元素。',
    src=VKSH + '/mul_mat_vec_q4_k.comp', parts=[(11, 35)], duration=19000,
    mark_src=[11, 12, 13, 15, 17, 19, 20, 23, 24, 25, 26],
    notes_src={11: 'calc_superblock：一次算一个 256 元素的 super-block',
               12: 'y1_idx / y2_idx 相差 128 —— 一个 super-block 被劈成前后两半',
               17: 'dm 是两个 f16：d（scale）与 dmin',
               19: 'scales 的 12 个字节里塞了 8 个 6-bit scale + 8 个 6-bit min',
               23: '把相邻两个 16-bit 槽拼成一个 uint32',
               25: 'unpack8 + 掩码 0x3F3F3F3F：从一个 uint32 里取出 4 个 6-bit scale',
               26: '后 4 个 scale 连同刚才溢出的高位一起拼出来'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" id="top" style="gap:6px;align-items:center"></div>
  <div class="row wrap" id="subs" style="gap:5px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const top = wrap.querySelector('#top');
top.innerHTML = '<span class="chip d">1 个 super-block = 256 个元素</span>' +
  '<span class="cm" style="margin:0">= dm(2 x f16) + scales[12] + qs[128] = 144 字节</span>' +
  '<span class="arrow">-></span>' +
  '<span class="chip a">8 个 sub-block x 32 个 4-bit</span>';

const subs = wrap.querySelector('#subs');
const cells = [];
for (let i = 0; i < 8; i++) {
  const c = U.el('div', { style: 'width:78px;height:34px;border:1px solid var(--border);border-radius:3px;' +
    'background:#10151b;display:flex;flex-direction:column;align-items:center;justify-content:center;' +
    'font-size:9px;color:var(--dim)' });
  c.innerHTML = '<b style="color:var(--d)">sub ' + i + '</b><span>6-bit scale</span>';
  subs.appendChild(c); cells.push(c);
}

const msg = wrap.querySelector('#msg');
const texts = [
  'K-quant 的块是 <span class="k">super-block</span>：256 个元素，8 个 sub-block 各 32 个。',
  '每个 sub-block 有自己的 6-bit scale。8 个 x 6 bit = 48 bit，正好塞进 ' +
  '<span class="v">scales[12]</span>（12 字节 = 96 bit，另一半留给 min）。',
  '第 <span class="v">19-21</span> 行一次取 3 个 16-bit 槽' +
  '（<span class="v">scales[]</span> 一共 6 个 uint16 = 12 字节），' +
  '第 <span class="v">23-26</span> 行用移位/掩码把它们摊成 8 个 6-bit scale。',
  '第 <span class="v">12-13</span> 行：一个 super-block 的数据被劈成 ' +
  '<span class="v">y1_idx</span>（前 128 元素）与 <span class="v">y2_idx</span>（后 128）。',
  '第 <span class="v">37-38</span> 行取 qs：<span class="v">q_offset/4</span> 与 <span class="v">+16</span> —— ' +
  '同样是"一次读 4 个字节、拆出 8 个 nibble"的手法。',
  '一句话：<span class="k">K-quant 在着色器里 = 移位 + 掩码 + fma</span>，没有查表魔法。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
tl.at(3400, () => { msg.innerHTML = texts[1]; });
tl.at(6400, () => {
  cells.forEach((c, i) => { c.style.borderColor = 'var(--d)'; c.style.background = 'rgba(188,140,255,.15)'; });
  msg.innerHTML = texts[2];
});
tl.at(9600, () => {
  cells.forEach((c, i) => { c.style.background = i < 4 ? 'rgba(88,166,255,.18)' : 'rgba(247,120,186,.18)'; });
  msg.innerHTML = texts[3];
});
tl.at(12600, () => { msg.innerHTML = texts[4]; });
tl.at(16000, () => {
  cells.forEach(c => { c.style.background = 'rgba(63,185,80,.18)'; c.style.borderColor = 'var(--b)'; });
  msg.innerHTML = texts[5];
});
'''
)

# ==================================================================== 第 9 幕

L.scene(
    kicker='L6-07 · 收束',
    title='把这一课压成一张表',
    sub='六步链 + 三个数字。回到第 2 幕的那 16 行 —— 整条链的枢纽就在那里。',
    caption='下一课 L6-08：Metal 后端怎么把多个 ggml op 融合进一个 kernel。',
    src=GEN, parts=[(436, 451)], duration=20000,
    mark_src=[437, 438, 444, 450],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:8px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['步', '在哪', '产物'],
  [['1', 'process_shaders()', '要编译的 (着色器, 类型, 变体) 清单'],
   ['2', 'string_to_spv()', '变体名 + <output_dir>/<name>.spv 路径'],
   ['3', 'glslc（string_to_spv_func）', '<name>.spv（SPIR-V 二进制）'],
   ['4', 'write_output_files()', '<name>_data[] / <name>_len（C 数组）'],
   ['5', 'ggml_vk_create_pipeline_func()', 'shader module → compute pipeline'],
   ['6', 'ggml_vk_get_dequantize_*', '运行时按类型查表取 pipeline → dispatch']],
  { monoCols: [1] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '不看代码，说出一个 <span class="mono">.comp</span> 从生成到被 pipeline 使用经过哪几步？',
  '<b>六步</b>：<br>' +
  '1. <span class="mono">process_shaders()</span> 决定要哪些变体；<br>' +
  '2. <span class="mono">string_to_spv()</span> 拼变体名与 <span class="mono">.spv</span> 输出路径；<br>' +
  '3. <span class="mono">glslc</span> 带着一串 <span class="mono">-D</span> 宏把 <span class="mono">.comp</span> 编译成 <span class="mono">.spv</span>；<br>' +
  '4. <span class="mono">write_output_files()</span> 把 <span class="mono">.spv</span> 读成 ' +
  '<span class="mono">&lt;name&gt;_data[]</span> 与 <span class="mono">&lt;name&gt;_len</span>；<br>' +
  '5. 主机端 <span class="mono">createShaderModule()</span> → <span class="mono">vkCreateComputePipelines</span>；<br>' +
  '6. 运行时按 (权重类型, 列数, workgroup 档) 查表取 pipeline，再 dispatch。<br>' +
  '关键：<b>第 2 步到第 3 步之间是"一变多"</b> —— 一个 <span class="mono">.comp</span> 变成几百个 SPIR-V。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '六步，每步一个产物。',
  '<span class="k">1-2</span>：同一个 <span class="v">.comp</span> 被点名很多次，' +
  '每次一组不同的宏 —— 这就是"变体"。',
  '<span class="k">3</span>：<span class="v">glslc</span> 是唯一的编译器；' +
  '<span class="v">.spv</span> 是它的产物。',
  '<span class="k">4</span>：<span class="v">.spv</span> 变成 C 数组，' +
  '于是<b>不需要运行时编译着色器</b>。',
  '<span class="k">5-6</span>：主机端建 pipeline，运行时按类型查表 —— ' +
  '这就是 L6-06 讲的那一半。',
  '三个数字：<span class="v">161</span> 个 <span class="v">.comp</span>、' +
  '<span class="v">26</span> 个反量化着色器、<span class="v">4</span> 个矩阵乘/归约着色器文件' +
  '撑起 <span class="v">MUL_MAT</span> 的全部形状。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2600 + i * 2200, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 5)];
}));
tl.at(17500, () => {
  rows.forEach(x => { x.className = ''; });
  msg.innerHTML = texts[5];
});
'''
)

# ================================================================== source.md

L.section(
    '一、161 个 `.comp` 按前缀分族',
    '计数命令（两个数都对过）：\n\n'
    '```text\n'
    '$ ls ggml/src/ggml-vulkan/vulkan-shaders/*.comp | wc -l        # 154\n'
    '$ find ggml/src/ggml-vulkan/vulkan-shaders -name "*.comp" | wc -l  # 161（含 feature-tests/ 下 7 个）\n'
    '```\n\n'
    '按前缀分成 14 个族。`type_names` 是生成器里的**类型清单**：'
    '清单里每个名字都会在 `mul_mat_vec*`、`mul_mm*`、`dequant_*` 等族里派生出变体，'
    '所以"加一个量化类型"在生成器里就是往这个 vector 里加一行。',
    src=GEN, parts=[(49, 78)], lang='cpp')

L.section(
    '二、★ 六步链的枢纽：`string_to_spv()`',
    '生成器的入口是 `process_shaders()`，它按 (着色器文件 × 类型 × 变体) 调用 `string_to_spv()`。'
    '这个函数做三件事：把**变体名**拼出来（精度后缀 + coopmat 档 + 自定义 suffix）、'
    '算出 `<output_dir>/<name>.spv`、然后异步交给 `string_to_spv_func()` 去跑 `glslc`。'
    '第 444 行那个"只编译与自己同名的源文件"的分支解释了 CMake 的调用方式：'
    '**每个 `.comp` 一个生成器进程**，最后再汇总成同一个头文件。',
    src=GEN, parts=[(436, 451)], lang='cpp')

L.section(
    '三、变体并不总是"一类一变"：`MULMAT_QUANT` 用一个 SPIR-V 覆盖全部量化类型',
    '大多数族是"一个类型一个变体"。但 `mul_mm.comp` 有一条例外路径：'
    '源码注释写得很直白 —— *one SPIR-V for all quant types, selected via MmTypeA spec constant*。'
    '它只编译 `_quant_f16` / `_quant_f32` 两个变体，运行时用 specialization constant `MmTypeA` '
    '选择走哪一支。代价是着色器里多一层运行时分支，收益是 pipeline 数量大减。',
    src=GEN, parts=[(656, 672)], lang='cpp')

L.section(
    '四、第 3 步：`glslc` 命令行是怎么拼出来的',
    '命令行只有一个必选阶段参数（compute）、一个由变体名决定的目标环境，'
    '以及一大串 `-D`。`defines` 是 `std::map`，所以顺序稳定、命令行可复现。'
    '注意第 363 行那个分支：coopmat / bf16 / rope / dot2 四类变体**不加** spirv-opt，'
    '源码注释逐条给了上游 issue 编号 —— 这是"优化遍会改坏某些着色器"的现场记录。',
    src=GEN, parts=[(350, 384)], lang='cpp')

L.section(
    '五、第 4 步：`.spv` 变成 C 数组',
    '`write_output_files()` 遍历 `shader_fnames`（第 2 步每登记一个变体就多一项），'
    '把每个 `.spv` 读回来，逐字节写成 C 数组，并在头文件里给出 `extern` 声明。'
    '这一步是"Vulkan 后端不需要运行时编译着色器"的全部原因：'
    'SPIR-V 变成了和 `.o` 一样的目标文件内容。',
    src=GEN, parts=[(1252, 1270)], lang='cpp')

L.section(
    '六、第 5 步：主机端把 `spv_data` 变成 shader module',
    '`ggml_vk_create_pipeline_func()` 的入参就是上一步产出的 `spv_size` / `spv_data`。'
    '它先按设备能力**补丁 SPIR-V**（打开支持的 fp16 float controls），'
    '再建 `vk::ShaderModuleCreateInfo`，随后 `createShaderModule()`，'
    '最后连同 specialization constants 与 push constant range 一起建 compute pipeline。'
    '这一课只看它与上一课的接缝；细节在 L6-06。',
    src='ggml/src/ggml-vulkan/ggml-vulkan.cpp', parts=[(554, 572)], lang='cpp')

L.section(
    '七、第 6 步：运行时按 (类型, 列数, workgroup 档) 查表',
    'pipeline 全部建成后放在 `device->pipeline_dequant_mul_mat_vec_f32_f32[w][a_type][i]` 这类'
    '三维表里。运行时 `ggml_vk_get_dequantize_mul_mat_vec()` 根据 B 的类型（f32/f16/q8_1）、'
    'A 的类型、以及算出的 workgroup 档位取出一个 pipeline，交给 `ggml_vk_dispatch_pipeline()`。'
    '**"一个 ggml op 落在哪个 SPIR-V 上"最终就压缩成这一次数组索引。**',
    src='ggml/src/ggml-vulkan/ggml-vulkan.cpp', parts=[(5377, 5385)], lang='cpp')

L.section(
    '八、★ GLSL 侧的量化块：`types.glsl`',
    '`types.glsl` 是 GLSL 侧的"量化块字典"：每个类型一个 struct，加上块大小宏，'
    '再用 `#if defined(DATA_A_xxx)` 把当前编译的类型绑到 `QUANT_K` / `A_TYPE` 上。'
    '因为这一切发生在**编译期**，一个 SPIR-V 只认识一种 A 类型 —— 这正是变体数量的来源。',
    src=VKSH + '/types.glsl', parts=[(56, 77)], lang='glsl')

L.section(
    '九、★ C 侧的同一个块（L1-04）',
    '同一个 Q4_0 块在 CPU 侧的定义。字段逐个对应：`ggml_half d` ↔ `float16_t d`，'
    '`uint8_t qs[QK4_0 / 2]` ↔ `uint8_t qs[16]`，块大小都是 32。'
    '**两份定义、两个文件、两套 assert/宏** —— 这就是"新增一个量化类型要同时改 CPU 与所有 GPU 后端"'
    '的代码依据。下面这 6 行属于 L1-04 的覆盖范围，本课只为对照引用。',
    src='ggml/src/ggml-common.h', parts=[(194, 199)], lang='c')

L.section(
    '十、反量化片段：`dequant_funcs.glsl` 里的 Q4_0',
    '被 `get_rows_quant.comp` 等着色器 include 的反量化片段。'
    '`dequantize()` 一次出 2 个值、`dequantize4()` 一次出 4 个值，'
    '两者都只是"取 nibble、减 8"，scale 由调用方在外面乘 —— '
    '与 `dequant_q4_0.comp` 把 scale 乘进去的写法相比，分工不同、块布局完全一样。',
    src=VKSH + '/dequant_funcs.glsl', parts=[(64, 73)], lang='glsl')

L.section(
    '十一、`mul_mm.comp` 的两套量化路径',
    '`mul_mm.comp` 是矩阵乘的主文件。它有两种被编译的方式：'
    '打开 `MULMAT_QUANT` 时走**运行时类型开关**（`MmTypeA` 是个 specialization constant，'
    '第 36 行），否则走**编译期类型开关**（`DATA_A_*` 宏）。'
    '第 39 行的 `#include "types.glsl"` 是所有着色器共享量化块定义的入口。',
    src=VKSH + '/mul_mm.comp', parts=[(34, 49)], lang='glsl')

L.section(
    '十二、`flash_attn.comp`：Q 是 f32、K/V 是 f16',
    'flash attention 的着色器把 Q 当 f32、K/V 当 f16 读（binding 0/1/2 的三组别名视图）。'
    '它在第 13-18 行按 `MMQ` 宏分成两个变体：MMQ 那条要求整数点积扩展，'
    '并把 `mul_mmq_shmem_types.glsl` 的共享内存块类型拉进来 —— '
    '也就是**注意力与量化矩阵乘共用同一套 packed 块类型**。',
    src=VKSH + '/flash_attn.comp', parts=[(13, 48)], lang='glsl')

L.section(
    '十三、`rms_norm.comp`：一个文件顶三种融合',
    '`rms_norm.comp` 的 `main()` 不是简单调用一次：它按 `num_blocks` 把 '
    '`rms_norm(num_iters)` **实例化十次**，让每种的循环次数在编译期成为常量。'
    '同一份着色器还通过 `RMS_NORM_ROPE_FUSION` / `RMS_NORM_ADD_FUSION` / '
    '`RMS_NORM_SET_ROWS_FUSION` 承担三条融合路径 —— 这解释了生成器里'
    '`rms_norm_*` 那一长串变体名。',
    src=VKSH + '/rms_norm.comp', parts=[(153, 178)], lang='glsl')

L.section(
    '十四、`soft_max.comp`：push constant 里的算子参数',
    'softmax 的 `parameter` 块把 ggml 算子的标量参数整包搬进 push constant：'
    '形状（`ne00..ne13`）、步长（`nb11..nb13`）、`scale`、ALiBi 的 `max_bias`/`m0`/`m1`、'
    '以及 sink 个数。**这是 L1-01 说的 `op_params` 在 GPU 侧的落地方式**。',
    src=VKSH + '/soft_max.comp', parts=[(41, 56)], lang='glsl')

L.section(
    '十五、`rope_norm.comp`：最小的 `.comp` 骨架',
    '整个文件 17 行：`#version` → include 一个 head（push constant 与类型）→ include 一个 funcs'
    '（真正的算法）→ `void main()` 里把全局线程号拆成 (row, head, batch) 等索引，'
    '然后调用 `rope_norm(i0, i1, i2, i3, pc)`。'
    '**其余 160 个 `.comp` 都是这个骨架加更多分支。**',
    src=VKSH + '/rope_norm.comp', parts=[(1, 17)], lang='glsl')

L.footnote_add('本课声明计划里 L6-07 的**全部 162 个文件**（161 个 `.comp` + `vulkan-shaders-gen.cpp`），'
               '无一遗漏；计数命令见 `source.md` 第一节。')
L.footnote_add('另有 4 个文件被本课引用：`ggml/src/ggml-vulkan/ggml-vulkan.cpp`（归 L6-06，'
               '本课用它的第 554-572 与 5377-5385 行说明"被 pipeline 使用"这一步）、'
               '`ggml/src/ggml-common.h`（归 L1-04，本课用第 194-199 行做 GLSL/C 两份块定义的对照）。'
               '这两个文件在覆盖域内，各自的主覆盖仍在原课。')
L.footnote_add('还有 `types.glsl` 与 `dequant_funcs.glsl`（Vulkan 着色器的 `.glsl` 头文件）。'
               '`.glsl` 后缀**不在覆盖域**内（覆盖域只收 `.comp`），所以它们**不计入覆盖率**，'
               '仅作为论据逐字引用 —— 计入与不计入在这里是分开的账。')
L.footnote_add('本课不讨论任何命令行参数；参数门禁的真值集来自真实二进制与本仓库门禁脚本的帮助输出。')

L.conclusion(
    '★ 一个 `.comp` 是模板，SPIR-V 才是产物',
    '六步链：\n\n'
    '```text\n'
    'process_shaders()  ->  string_to_spv()  ->  glslc  ->  write_output_files()\n'
    '                   ->  createShaderModule()  ->  vkCreateComputePipelines  ->  查表 dispatch\n'
    '```\n\n'
    '第 2 步到第 3 步之间是**一变多**：同一个 `.comp` 配不同的宏，编译出几百个 `.spv`。'
    '所以"Vulkan 后端有多少着色器"这个问题没有单一答案 —— 取决于运行 `vulkan-shaders-gen` 时'
    '打开了哪些编译期开关（coopmat / coopmat2 / 整数点积 / bf16 / fp4 ...）。')

L.conclusion(
    '★ 量化块的定义是复制出来的，不是共享的',
    '同一个 `block_q4_0` 至少存在两份**独立**定义：`ggml-common.h`（C，CPU 内核用）与 '
    '`types.glsl`（GLSL，Vulkan 着色器用），CUDA / Metal / SYCL 各后端还有各自的版本。'
    '形状必须逐个字段一致，但**编译器不会替你检查这件事**。\n\n'
    '因此新增一个量化类型是一串跨语言的改动：C 块定义 → GLSL 块定义 → `dequant_<type>.comp` → '
    '生成器的 `type_names` → 各后端 kernel 的类型分发。这就是"量化类型是全局概念"的具体含义。')

L.conclusion(
    '三个矩阵乘族的分工边界是 M',
    '| 族 | 文件数 | 负责 |\n|---|---|---|\n'
    '| `mul_mat_vec*` | 19 | 小 M（逐 token 解码）与 `MUL_MAT_ID` |\n'
    '| `mul_mm*` / `mul_mmq` | 4 | 大 M；`mul_mmq` 是整数点积（B 侧 q8_1） |\n'
    '| `dequant_*` | 26 | 把量化权重先反量化成 f16，供 f16 路径使用 |\n\n'
    '三者合计 49 个文件，占 161 个 `.comp` 的三成 —— '
    '矩阵乘是 GPU 后端里最贵的算子，也是变体最多的地方。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
