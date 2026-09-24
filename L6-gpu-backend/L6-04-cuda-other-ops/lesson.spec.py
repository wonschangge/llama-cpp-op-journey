#!/usr/bin/env python3
"""L6-04 · CUDA 其余算子与模板实例化 —— 课件 spec。

运行：python3 L6-gpu-backend/L6-04-cuda-other-ops/lesson.spec.py

本课覆盖 plan_matrix 里 L6-04 的**全部** 280 个文件：
  - 160 个手写的 .cu / .cuh / .h（ggml/src/ggml-cuda/ 下的算子实现与辅助头）
  - 120 个 template-instances/*.cu（生成文件）
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402
import plan_matrix as pmat             # noqa: E402

# ------------------------------------------------------------------ 源文件
SRC_CUDA   = 'ggml/src/ggml-cuda/ggml-cuda.cu'
SRC_NORM   = 'ggml/src/ggml-cuda/norm.cu'
SRC_SOFTMAX = 'ggml/src/ggml-cuda/softmax.cu'
SRC_ROPE   = 'ggml/src/ggml-cuda/rope.cu'
SRC_SSM    = 'ggml/src/ggml-cuda/ssm-scan.cu'
SRC_TOPK   = 'ggml/src/ggml-cuda/top-k.cu'
SRC_MOE    = 'ggml/src/ggml-cuda/moe-weighted-reduction.cu'
SRC_QUANT  = 'ggml/src/ggml-cuda/quantize.cu'
SRC_MMQ    = 'ggml/src/ggml-cuda/mmq.cu'
SRC_MMVQ   = 'ggml/src/ggml-cuda/mmvq.cu'
CUH_MMQ    = 'ggml/src/ggml-cuda/mmq.cuh'
CUH_MMF    = 'ggml/src/ggml-cuda/mmf.cuh'
CUH_FVEC   = 'ggml/src/ggml-cuda/fattn-vec.cuh'
CUH_FTILE  = 'ggml/src/ggml-cuda/fattn-tile.cuh'
CUH_FMMA   = 'ggml/src/ggml-cuda/fattn-mma-f16.cuh'
TI_MMQ     = 'ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu'
TI_MMF     = 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_1.cu'
TI_FVEC    = 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu'
TI_FTILE   = 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq128-dv128.cu'
TI_FMMA    = 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu'

# ------------------------------------------------------------------ 计数
# 全部来自命令的真实输出，可复核（见 source.md 第一节）：
#   python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L6-04"{print $2}' | wc -l
#   grep -c 'template-instances/' <那份清单>
N_FILES        = 280
N_HANDWRITTEN  = 160          # 非 template-instances = 68 个 .cu + 89 个 .cuh + 3 个 vendors/*.h
N_TI           = 120          # template-instances/*.cu
N_TI_MMQ       = 22
N_TI_MMF       = 16
N_TI_FATTN     = 82           # = 21 (mma-f16) + 12 (tile) + 49 (vec)
N_TI_FATTN_MMA = 21
N_TI_FATTN_TILE = 12
N_TI_FATTN_VEC = 49

# 计数之间的算术关系也一并断言，避免任何一处笔误悄悄溜过去。
assert N_HANDWRITTEN + N_TI == N_FILES, (N_HANDWRITTEN, N_TI, N_FILES)
assert N_TI_MMQ + N_TI_MMF + N_TI_FATTN == N_TI, (N_TI_MMQ, N_TI_MMF, N_TI_FATTN, N_TI)
assert N_TI_FATTN_MMA + N_TI_FATTN_TILE + N_TI_FATTN_VEC == N_TI_FATTN

# 手写文件（非 template-instances）：160 个
FILES_HANDWRITTEN = [
    'ggml/src/ggml-cuda/acc.cu', 'ggml/src/ggml-cuda/acc.cuh',
    'ggml/src/ggml-cuda/add-id.cu', 'ggml/src/ggml-cuda/add-id.cuh',
    'ggml/src/ggml-cuda/allreduce.cu', 'ggml/src/ggml-cuda/allreduce.cuh',
    'ggml/src/ggml-cuda/arange.cu', 'ggml/src/ggml-cuda/arange.cuh',
    'ggml/src/ggml-cuda/argmax.cu', 'ggml/src/ggml-cuda/argmax.cuh',
    'ggml/src/ggml-cuda/argsort.cu', 'ggml/src/ggml-cuda/argsort.cuh',
    'ggml/src/ggml-cuda/binbcast.cu', 'ggml/src/ggml-cuda/binbcast.cuh',
    'ggml/src/ggml-cuda/clamp.cu', 'ggml/src/ggml-cuda/clamp.cuh',
    'ggml/src/ggml-cuda/col2im-1d.cu', 'ggml/src/ggml-cuda/col2im-1d.cuh',
    'ggml/src/ggml-cuda/common.cuh', 'ggml/src/ggml-cuda/concat.cu',
    'ggml/src/ggml-cuda/concat.cuh', 'ggml/src/ggml-cuda/conv-transpose-1d.cu',
    'ggml/src/ggml-cuda/conv-transpose-1d.cuh', 'ggml/src/ggml-cuda/conv2d-dw.cu',
    'ggml/src/ggml-cuda/conv2d-dw.cuh', 'ggml/src/ggml-cuda/conv2d-transpose.cu',
    'ggml/src/ggml-cuda/conv2d-transpose.cuh', 'ggml/src/ggml-cuda/conv2d.cu',
    'ggml/src/ggml-cuda/conv2d.cuh', 'ggml/src/ggml-cuda/convert.cu',
    'ggml/src/ggml-cuda/convert.cuh', 'ggml/src/ggml-cuda/count-equal.cu',
    'ggml/src/ggml-cuda/count-equal.cuh', 'ggml/src/ggml-cuda/cp-async.cuh',
    'ggml/src/ggml-cuda/cpy-utils.cuh', 'ggml/src/ggml-cuda/cpy.cu',
    'ggml/src/ggml-cuda/cpy.cuh', 'ggml/src/ggml-cuda/cross-entropy-loss.cu',
    'ggml/src/ggml-cuda/cross-entropy-loss.cuh', 'ggml/src/ggml-cuda/cumsum.cu',
    'ggml/src/ggml-cuda/cumsum.cuh', 'ggml/src/ggml-cuda/dequantize.cuh',
    'ggml/src/ggml-cuda/diag.cu', 'ggml/src/ggml-cuda/diag.cuh',
    'ggml/src/ggml-cuda/diagmask.cu', 'ggml/src/ggml-cuda/diagmask.cuh',
    'ggml/src/ggml-cuda/dsv4-hc.cu', 'ggml/src/ggml-cuda/dsv4-hc.cuh',
    'ggml/src/ggml-cuda/fattn-common.cuh', 'ggml/src/ggml-cuda/fattn-mma-f16.cuh',
    'ggml/src/ggml-cuda/fattn-tile.cu', 'ggml/src/ggml-cuda/fattn-tile.cuh',
    'ggml/src/ggml-cuda/fattn-vec.cuh', 'ggml/src/ggml-cuda/fattn.cu',
    'ggml/src/ggml-cuda/fattn.cuh', 'ggml/src/ggml-cuda/fill.cu',
    'ggml/src/ggml-cuda/fill.cuh', 'ggml/src/ggml-cuda/fwht.cu',
    'ggml/src/ggml-cuda/fwht.cuh', 'ggml/src/ggml-cuda/gated_delta_net.cu',
    'ggml/src/ggml-cuda/gated_delta_net.cuh', 'ggml/src/ggml-cuda/getrows.cu',
    'ggml/src/ggml-cuda/getrows.cuh', 'ggml/src/ggml-cuda/ggml-cuda.cu',
    'ggml/src/ggml-cuda/gla.cu', 'ggml/src/ggml-cuda/gla.cuh',
    'ggml/src/ggml-cuda/im2col.cu', 'ggml/src/ggml-cuda/im2col.cuh',
    'ggml/src/ggml-cuda/lightning-indexer.cu', 'ggml/src/ggml-cuda/lightning-indexer.cuh',
    'ggml/src/ggml-cuda/mean.cu', 'ggml/src/ggml-cuda/mean.cuh',
    'ggml/src/ggml-cuda/mma.cuh', 'ggml/src/ggml-cuda/mmf.cu',
    'ggml/src/ggml-cuda/mmf.cuh', 'ggml/src/ggml-cuda/mmid.cu',
    'ggml/src/ggml-cuda/mmid.cuh', 'ggml/src/ggml-cuda/mmq-config-ampere.cuh',
    'ggml/src/ggml-cuda/mmq-config-blackwell.cuh', 'ggml/src/ggml-cuda/mmq-config-cdna.cuh',
    'ggml/src/ggml-cuda/mmq-config-gcn.cuh', 'ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh',
    'ggml/src/ggml-cuda/mmq-config-pascal-older.cuh', 'ggml/src/ggml-cuda/mmq-config-rdna2.cuh',
    'ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh', 'ggml/src/ggml-cuda/mmq-config-rdna3.cuh',
    'ggml/src/ggml-cuda/mmq-config-rdna4.cuh', 'ggml/src/ggml-cuda/mmq-load-tiles.cuh',
    'ggml/src/ggml-cuda/mmq-vec-dot.cuh', 'ggml/src/ggml-cuda/mmq.cu',
    'ggml/src/ggml-cuda/mmq.cuh', 'ggml/src/ggml-cuda/mmvf.cu',
    'ggml/src/ggml-cuda/mmvf.cuh', 'ggml/src/ggml-cuda/mmvq.cu',
    'ggml/src/ggml-cuda/mmvq.cuh', 'ggml/src/ggml-cuda/moe-weighted-reduction.cu',
    'ggml/src/ggml-cuda/moe-weighted-reduction.cuh', 'ggml/src/ggml-cuda/norm.cu',
    'ggml/src/ggml-cuda/norm.cuh', 'ggml/src/ggml-cuda/opt-step-adamw.cu',
    'ggml/src/ggml-cuda/opt-step-adamw.cuh', 'ggml/src/ggml-cuda/opt-step-sgd.cu',
    'ggml/src/ggml-cuda/opt-step-sgd.cuh', 'ggml/src/ggml-cuda/out-prod.cu',
    'ggml/src/ggml-cuda/out-prod.cuh', 'ggml/src/ggml-cuda/pad.cu',
    'ggml/src/ggml-cuda/pad.cuh', 'ggml/src/ggml-cuda/pad_reflect_1d.cu',
    'ggml/src/ggml-cuda/pad_reflect_1d.cuh', 'ggml/src/ggml-cuda/pool1d.cu',
    'ggml/src/ggml-cuda/pool1d.cuh', 'ggml/src/ggml-cuda/pool2d.cu',
    'ggml/src/ggml-cuda/pool2d.cuh', 'ggml/src/ggml-cuda/quantize.cu',
    'ggml/src/ggml-cuda/quantize.cuh', 'ggml/src/ggml-cuda/reduce_rows.cuh',
    'ggml/src/ggml-cuda/roll.cu', 'ggml/src/ggml-cuda/roll.cuh',
    'ggml/src/ggml-cuda/rope.cu', 'ggml/src/ggml-cuda/rope.cuh',
    'ggml/src/ggml-cuda/scale.cu', 'ggml/src/ggml-cuda/scale.cuh',
    'ggml/src/ggml-cuda/set-rows.cu', 'ggml/src/ggml-cuda/set-rows.cuh',
    'ggml/src/ggml-cuda/set.cu', 'ggml/src/ggml-cuda/set.cuh',
    'ggml/src/ggml-cuda/snake.cu', 'ggml/src/ggml-cuda/snake.cuh',
    'ggml/src/ggml-cuda/softcap.cu', 'ggml/src/ggml-cuda/softcap.cuh',
    'ggml/src/ggml-cuda/softmax.cu', 'ggml/src/ggml-cuda/softmax.cuh',
    'ggml/src/ggml-cuda/solve_tri.cu', 'ggml/src/ggml-cuda/solve_tri.cuh',
    'ggml/src/ggml-cuda/ssm-conv.cu', 'ggml/src/ggml-cuda/ssm-conv.cuh',
    'ggml/src/ggml-cuda/ssm-scan.cu', 'ggml/src/ggml-cuda/ssm-scan.cuh',
    'ggml/src/ggml-cuda/sum.cu', 'ggml/src/ggml-cuda/sum.cuh',
    'ggml/src/ggml-cuda/sumrows.cu', 'ggml/src/ggml-cuda/sumrows.cuh',
    'ggml/src/ggml-cuda/top-k.cu', 'ggml/src/ggml-cuda/top-k.cuh',
    'ggml/src/ggml-cuda/topk-moe.cu', 'ggml/src/ggml-cuda/topk-moe.cuh',
    'ggml/src/ggml-cuda/tri.cu', 'ggml/src/ggml-cuda/tri.cuh',
    'ggml/src/ggml-cuda/tsembd.cu', 'ggml/src/ggml-cuda/tsembd.cuh',
    'ggml/src/ggml-cuda/unary.cu', 'ggml/src/ggml-cuda/unary.cuh',
    'ggml/src/ggml-cuda/upscale.cu', 'ggml/src/ggml-cuda/upscale.cuh',
    'ggml/src/ggml-cuda/vecdotq.cuh', 'ggml/src/ggml-cuda/vendors/cuda.h',
    'ggml/src/ggml-cuda/vendors/hip.h', 'ggml/src/ggml-cuda/vendors/musa.h',
    'ggml/src/ggml-cuda/wkv.cu', 'ggml/src/ggml-cuda/wkv.cuh',
]

# template-instances/ 下的生成文件：120 个
FILES_TEMPLATE_INSTANCES = [
    'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_32.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_32.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu', 'ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq112-dv112.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq128-dv128.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq192-dv128.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq256-dv256.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq320-dv256.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq40-dv40.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq512-dv512.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq576-dv512.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq64-dv64.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq72-dv72.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq80-dv80.cu', 'ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq96-dv96.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-f16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q4_0.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q4_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q5_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q5_1.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q8_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-bf16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q5_0.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q5_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q8_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-bf16.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-f16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_1.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q5_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q5_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q8_0.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-bf16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-f16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q4_0.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q4_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q5_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q5_1.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q8_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-bf16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-f16.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q4_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q4_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_0.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q8_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-bf16.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-f16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q4_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q4_1.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q5_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q5_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q8_0.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-bf16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-f16.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q4_0.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q4_1.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q5_0.cu', 'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q5_1.cu',
    'ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_1.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_10.cu',
    'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_11.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_12.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_13.cu',
    'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_14.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_15.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_16.cu',
    'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_2.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_3.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_4.cu',
    'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_5.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_6.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_7.cu',
    'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_8.cu', 'ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_9.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-iq1_s.cu',
    'ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu',
    'ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu',
    'ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu',
    'ggml/src/ggml-cuda/template-instances/mmq-instance-q1_0.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q2_k.cu',
    'ggml/src/ggml-cuda/template-instances/mmq-instance-q3_k.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q4_1.cu',
    'ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.cu',
    'ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu', 'ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu',
]
# ------------------------------------------------------------------ 覆盖声明
# 与计划（plan_matrix 的 L6-04 清单）逐条对齐：不一致就直接报错，不静默通过。
_ASSIGN, _ = pmat.resolve()
_PLANNED = _ASSIGN['L6-04']
FILES = FILES_HANDWRITTEN + FILES_TEMPLATE_INSTANCES
_missing = sorted(set(_PLANNED) - set(FILES))
_extra = sorted(set(FILES) - set(_PLANNED))
if _missing or _extra:
    raise SystemExit(f'L6-04 清单与计划不一致：漏 {len(_missing)}，多 {len(_extra)}')
if len(FILES) != N_FILES:
    raise SystemExit(f'清单文件数 {len(FILES)} != {N_FILES}')

# 幕里用到的几个计数，注入 JS 时保持与上面的常量同源。
_JS_MAP = '[%d, %d, %d]' % (N_FILES, N_HANDWRITTEN, N_TI)
_JS_TI = '[%d, %d, %d, %d, %d]' % (N_TI_MMQ, N_TI_MMF, N_TI_FATTN_VEC,
                                    N_TI_FATTN_TILE, N_TI_FATTN_MMA)

L = Lesson(
    id='L6-04',
    layer='L6 · GPU 后端执行',
    title='CUDA 其余算子与模板实例化',
    kicker='L6 · GPU 后端执行',
    codecap='ggml/src/ggml-cuda/ 与 template-instances/（逐字引用）',
    nav={'prev': {'href': '../L6-03-cuda-flash-attention/index.html',
                  'label': 'L6-03 ★ CUDA FlashAttention'},
         'next': {'href': '../L6-05-sycl-backend/index.html',
                  'label': 'L6-05 SYCL 后端'}},
)

L.cover(*FILES)

L.note('**一句话**：L6-01 讲了 CUDA 后端的骨架与分派、L6-02 讲了量化矩阵乘、L6-03 讲了 '
       'FlashAttention；这一课把 `ggml/src/ggml-cuda/` 剩下的部分全部收掉 —— '
       '**160 个手写文件 + 120 个生成的模板实例**，并回答一个具体问题：'
       '`template-instances/` 这个目录为什么必须存在。')
L.note('这一课是本层唯一"全目录声明"的课：计划里 L6-04 的清单就是 '
       '`ggml/src/ggml-cuda/` 下的全部 %d 个源文件，本课把它们**全部**写进覆盖声明。'
       'L6-01 / L6-02 / L6-03 覆盖的是这个目录的**子集**'
       '（`ggml/include/ggml-cuda.h` 这个公共头则归 L6-01）。' % N_FILES)

# ================================================================== 第 1 幕

L.scene(
    kicker='L6-04 · 全局',
    title='本课的地图：<span class="hl-a">%d</span> 个文件，两类来源' % N_FILES,
    sub='ggml/src/ggml-cuda/ 下的文件分成两类：人写的算子实现，和脚本生成的模板实例。',
    caption='计数命令与逐族清单见 source.md 第一节与第十一节；数字全部来自命令输出。',
    src=SRC_CUDA, parts=[(5, 49)], duration=16000,
    mark_src=[35, 36, 43, 44, 49],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row" id="stats" style="gap:9px"></div>
  <div class="row wrap" id="ops" style="gap:6px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const N = @COUNTS@;

const stats = [
  { n: String(N[0]), t: '本课清单文件数', s: N[1] + ' 手写 + ' + N[2] + ' 生成', c: 'a' },
  { n: String(N[1]), t: '手写 .cu / .cuh / .h', s: '68 .cu + 89 .cuh + 3 vendors/*.h', c: 'b' },
  { n: String(N[2]), t: 'template-instances/*.cu', s: '全部由脚本生成', c: 'c' }
];
const host = wrap.querySelector('#stats');
const els = stats.map(d => {
  const e = U.el('div', { class: 'card', style: 'width:218px' });
  e.style.borderLeftColor = 'var(--' + d.c + ')';
  e.innerHTML = '<div class="ct" style="font-size:22px;color:var(--' + d.c + ')">' + d.n + '</div>' +
    '<div class="cb"><b>' + d.t + '</b><br>' + d.s + '</div>';
  host.appendChild(e);
  return e;
});
els.forEach(e => e.style.opacity = '.35');

const opHost = wrap.querySelector('#ops');
['norm', 'rope', 'softmax', 'ssm-scan', 'top-k', 'argsort', 'quantize',
 'moe-weighted-reduction', 'wkv', 'gla', 'conv2d', 'cumsum'].forEach(n => {
  opHost.appendChild(U.chip(n, 'b'));
});

const msg = wrap.querySelector('#msg');
const texts = [
  '右边是 `ggml-cuda.cu` 的 include 清单（第 5-49 行）：<b>一个 .cuh 对应一组算子</b>。',
  '本课清单 = <span class="v">' + N[0] + '</span> 个文件：<span class="k">' + N[1] +
    '</span> 个人写、<span class="k">' + N[2] + '</span> 个生成。生成的那部分全在 ' +
    '<span class="v">template-instances/</span>。',
  '手写的那些里，绝大多数是一对 <span class="v">xxx.cu</span> + <span class="v">xxx.cuh</span>：' +
    '前者是 kernel 与入口，后者是声明。',
  '下面按"一个算子一幕"走：先看它们怎么被分派，再看 6 个代表性算子的 kernel 结构，' +
    '最后回答 template-instances 为什么必须存在。'
];
tl.at(600, () => { els[0].style.opacity = '1'; msg.innerHTML = texts[0]; });
tl.at(4200, () => { els[1].style.opacity = '1'; U.markLines(document, [0, 1, 2, 3, 4]); msg.innerHTML = texts[1]; });
tl.at(8000, () => { els[2].style.opacity = '1'; msg.innerHTML = texts[2]; });
tl.at(11800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''.replace('@COUNTS@', _JS_MAP)
)

# ================================================================== 第 2 幕

L.scene(
    kicker='L6-04 · 分派',
    title='和 L6-01 同一个 <span class="hl-a">switch</span>：每个 op 一个入口',
    sub='非矩阵乘的算子没有特殊通道 —— 它们和 MUL_MAT 挤在同一张 dst->op 分派表里。',
    caption='回顾 L6-01：ggml_cuda_compute_forward() 是 CUDA 后端唯一的算子入口。',
    src=SRC_CUDA, parts=[(2355, 2366)], duration=15000,
    mark_src=[2355, 2358, 2361, 2364],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['上游行号', 'case（GGML_OP_*）', '入口函数', '本课哪一幕'],
  [['2355', 'SSM_CONV', 'ggml_cuda_op_ssm_conv', 'L2-11 的图原语'],
   ['2358', 'SSM_SCAN', 'ggml_cuda_op_ssm_scan', '第 6 幕'],
   ['2361', 'TOP_K', 'ggml_cuda_op_top_k', '第 7 幕'],
   ['2364', 'ARGSORT', 'ggml_cuda_op_argsort', '第 7 幕'],
   ['2301', 'SOFT_MAX', 'ggml_cuda_op_soft_max', '第 4 幕'],
   ['2307', 'ROPE', 'ggml_cuda_op_rope', '第 5 幕'],
   ['2253', 'RMS_NORM', 'ggml_cuda_op_rms_norm', '第 3 幕']],
  { monoCols: [0, 2] });
wrap.querySelector('#tbl').appendChild(t.el);

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '这 12 行是同一个 switch 里的一段：<span class="k">case -> 一个 ggml_cuda_op_* 入口</span>。',
  '入口函数的命名是统一的：<span class="v">ggml_cuda_op_</span> 加上算子名的小写形式。',
  '<span class="v">SSM_SCAN</span> / <span class="v">TOP_K</span> / <span class="v">ARGSORT</span> ' +
    '都是这一课的内容。<span class="k">"多 kernel 选一"就发生在入口函数内部</span>：' +
    '按形状与类型挑具体 kernel。',
  '表中其它行号（2301 SOFT_MAX / 2307 ROPE / 2253 RMS_NORM）见 source.md 第二节的逐字引用。',
  '跨课呼应：<span class="k">L6-02（mmq/mmvq/mmf）与 L6-03（fattn）也是这张表上的 case</span>，' +
    '它们内部又各有"多 kernel 选一"的判据 —— 本课只看非矩阵乘的。'
];
tl.at(600, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2000 + i * 700, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[1];
}));
tl.at(8000, () => { msg.innerHTML = texts[2]; });
tl.at(11000, () => { msg.innerHTML = texts[3]; });
tl.at(13000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[4]; });
'''
)

# ================================================================== 第 3 幕

L.scene(
    kicker='L6-04 · norm.cu',
    title='norm.cu：<span class="hl-a">一个 block 管一行</span>，规约在 block 内闭合',
    sub='RMSNorm 在 GPU 上是一个"行内规约"问题：一行的平方和算完，才能写回这一行。',
    caption='同一个 kernel 还能顺手融合 mul（和 add）—— 靠的是编译期模板参数，不是运行期分支。',
    src=SRC_NORM, parts=[(131, 155)], duration=17000,
    mark_src=[131, 133, 137, 138, 140, 141, 144, 147, 151, 152],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const left = wrap.querySelector('#left');
const right = wrap.querySelector('#right');

left.innerHTML = '<div class="cm" style="margin:0 0 4px">四步，全在一个 block 内</div>';
const steps = [
  { c: 'a', t: '① 平方和', b: '每个线程按 <span class="cm" style="margin:0">col += block_size</span> 跨步扫自己那几列，累加 xi*xi' },
  { c: 'b', t: '② block 内规约', b: '<span class="cm" style="margin:0">block_reduce&lt;SUM&gt;(tmp, s_sum)</span>：共享内存 + warp 规约' },
  { c: 'c', t: '③ 求 scale', b: '<span class="cm" style="margin:0">rsqrtf(mean + eps)</span>，其中 mean = tmp / ncols' },
  { c: 'd', t: '④ 再扫一遍写回', b: '按编译期分支决定是否顺带乘 mul / 加 add' }
];
const els = steps.map(s => { const e = U.card(s, { style: 'width:100%' }); left.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.3');

right.innerHTML =
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">do_multiply / do_add 是模板参数</div>' +
  '<div class="cb">kernel 模板头是 <span class="cm" style="margin:0">template &lt;int block_size, ' +
  'bool do_multiply = false, bool do_add = false&gt;</span>（见 source.md 第三节）。<br>' +
  '所以 <span class="cm" style="margin:0">if constexpr</span> 的三个分支在编译期就只剩一个 —— ' +
  '<b>融合不是运行期判断</b>。</div></div>' +
  '<div class="card" style="border-left-color:var(--f)">' +
  '<div class="ct" style="color:var(--f)">谁决定 grid？</div>' +
  '<div class="cb"><span class="cm" style="margin:0">blocks_num(nrows, nchannels, nsamples)</span>：' +
  'grid 的三个维度就是张量的三个外维 —— 一行一个 block，不跨行规约。' +
  '线程数按 ncols 是否小于 1024 在 256 / 1024 之间选（见 source.md 第三节）。</div></div>' +
  '<div class="card" style="border-left-color:var(--g)">' +
  '<div class="ct" style="color:var(--g)">跨课呼应</div>' +
  '<div class="cb">L5-01 的 CPU 版 rms_norm 是同样的"两遍扫描"，差别只在规约用 SIMD 而不是 ' +
  'block_reduce。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  'RMSNorm = 在一行里做一次平方和规约，再按 <span class="v">scale</span> 缩放写回。',
  '<span class="v">for (col = tid; col &lt; ncols; col += block_size)</span>：' +
  '线程按块大小跨步 —— 这是所有"行内规约"kernel 的写法。',
  '<span class="v">block_reduce&lt;SUM&gt;</span> 把每线程的部分和压成一个数；' +
  '之后 <span class="v">scale = rsqrtf(mean + eps)</span>。',
  '<span class="v">if constexpr</span> 三个分支：只乘 mul / 乘 mul 再加 add / 什么都不做。' +
  'L2 的稠密模型里 <span class="cm" style="margin:0">RMS_NORM + MUL</span> 就是这么合并的。',
  '一句话：<span class="k">运行期的形状决定 grid，编译期的开关决定融合</span>。'
];
let t = 700;
els.forEach((e, i) => { tl.at(t, () => {
  els.forEach((x, k) => x.style.opacity = k <= i ? '1' : '.3');
  msg.innerHTML = texts[i];
}); t += 3200; });
tl.at(t, () => { msg.innerHTML = texts[4]; U.markLines(document, [13, 14, 15, 16, 17, 20, 21]); });
'''
)

# ================================================================== 第 4 幕

L.scene(
    kicker='L6-04 · softmax.cu',
    title='softmax.cu：把 <span class="hl-c">ncols</span> 折进编译期，但不建目录',
    sub='同一份 kernel 源码，对若干常见列数做编译期特化；选择靠一条折叠表达式。',
    caption='调用点写的是一串常量：launch_soft_max_kernels<32, 64, ..., 4096>（见 source.md 第四节）。',
    src=SRC_SOFTMAX, parts=[(278, 304)], duration=18000,
    mark_src=[278, 285, 286, 287, 289, 299, 303, 304],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:6px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin:0 0 4px">调用点给的候选（source.md 第四节）</div>';
const cand = [32, 64, 128, 256, 512, 1024, 2048, 4096];
const cels = cand.map(n => {
  const e = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
  e.innerHTML = '<span class="m">ncols == ' + n + '</span> -> <span style="color:var(--c)">soft_max_f32&lt;true, ' + n + ', block&gt;</span>';
  left.appendChild(e);
  return e;
});
const dflt = U.el('div', { class: 'formula', style: 'padding:4px 8px;font-size:10px' });
dflt.innerHTML = '<span class="m">其它列数</span> -> <span style="color:var(--d)">soft_max_f32&lt;true, 0, 0&gt;</span>';
left.appendChild(dflt);
const all = cels.concat([dflt]);
all.forEach(e => e.style.opacity = '.28');

const right = wrap.querySelector('#right');
right.innerHTML =
  '<div class="card" style="border-left-color:var(--a)">' +
  '<div class="ct" style="color:var(--a)">选择机制：一条折叠表达式</div>' +
  '<div class="cb">每个候选值包成 <span class="cm" style="margin:0">std::integral_constant&lt;int, Ns&gt;</span>，' +
  '由 <span class="cm" style="margin:0">(launch_kernel(...) || ...)</span> 折叠依次试；' +
  '命中就 <span class="cm" style="margin:0">return</span>。</div></div>' +
  '<div class="card" style="border-left-color:var(--d)">' +
  '<div class="ct" style="color:var(--d)">兜底：ncols_template = 0</div>' +
  '<div class="cb">没命中就用 <span class="cm" style="margin:0">soft_max_f32&lt;true, 0, 0&gt;</span>：' +
  '源码注释写了，此时循环边界未知、不能展开。</div></div>' +
  '<div class="card" style="border-left-color:var(--e)">' +
  '<div class="ct" style="color:var(--e)">和 template-instances/ 的差别</div>' +
  '<div class="cb">这里 8 个特化<b>全在同一个 .cu 里实例化</b>；' +
  'mmq 的 22 个类型则是<b>一个类型一个文件</b>（第 9 幕）。</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  'softmax 的列数是个热点形状：常见值就那几个。',
  '于是源码把它们做成<span class="k">编译期常量</span>：ncols_template 一确定，' +
  '循环就能展开、共享内存大小也能定死。',
  '选择发生在<b>一次折叠</b>里：候选依次尝试，命中即返回。' +
  '这是"多 kernel 选一"的第三种写法（前两种在 L6-02 / L6-03）。',
  '都没命中 -> <span class="v">&lt;true, 0, 0&gt;</span> 通用版兜底。' +
  '<span class="k">特化是加速，兜底是正确性。</span>',
  '注意：这个文件<b>没有</b>对应的 template-instances 目录 —— 因为实例化量小，' +
  '放在同一个 TU 里编译得动。第 9 幕回答"什么时候必须拆文件"。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3600, () => { msg.innerHTML = texts[1]; U.markLines(document, [8, 9]); });
cels.forEach((e, i) => tl.at(6200 + i * 900, () => {
  all.forEach((x, k) => x.style.opacity = k === i ? '1' : '.28');
  msg.innerHTML = texts[2];
}));
tl.at(14200, () => { all.forEach(x => x.style.opacity = '.28'); dflt.style.opacity = '1'; msg.innerHTML = texts[3]; });
tl.at(16200, () => { all.forEach(x => x.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ================================================================== 第 5 幕

L.scene(
    kicker='L6-04 · rope.cu',
    title='rope.cu：一个 <span class="hl-b">mode</span> 整数，四种成对方式',
    sub='RoPE 的变体不是四份图，而是 op_params 里 mode 的几个位 —— 但落地时确实是四个 kernel。',
    caption='每个 kernel 再按 src0/dst 的 dtype 组合分派（F32/F32、F32/F16、F16/F16），见 source.md 第五节。',
    src=SRC_ROPE, parts=[(606, 621)], duration=16000,
    mark_src=[606, 607, 608, 609, 615, 616],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: 'rope_norm', b: '交错式（GPT-J 风格）：<br>相邻两维成对旋转', m: '四路里的 else 分支' },
  { c: 'b', t: 'rope_neox', b: '半区式（GPT-NeoX 风格）：<br>前后半区对应维成对', m: 'mode & GGML_ROPE_TYPE_NEOX' },
  { c: 'c', t: 'rope_multi', b: 'M-RoPE：三个 section<br>各用不同的位置增量', m: 'mode & GGML_ROPE_TYPE_MROPE' },
  { c: 'd', t: 'rope_vision', b: '视觉塔：旋转对跨越整行，<br>不支持 offset', m: 'mode == GGML_ROPE_TYPE_VISION' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => U.card(d, { style: 'width:163px' }));
els.forEach(e => host.appendChild(e));
els.forEach(e => e.style.opacity = '.3');

const msg = wrap.querySelector('#msg');
const texts = [
  '四个 bool 从 <span class="v">mode</span> 里解出来 —— mode 来自 ' +
    '<span class="v">dst->op_params[2]</span>（L1-02 讲的那 16 个 int32）。',
  '<span class="v">is_neox</span>：是不是 NeoX 的成对方式；' +
    '<span class="v">is_mrope</span> / <span class="v">is_imrope</span>：多模态的 section 式位置。',
  '<span class="v">is_vision</span> 带两条例外断言：<span class="k">n_dims 必须是半个头</span>、' +
    '<span class="k">不支持 offset</span> —— 因为它的旋转对跨整个行。',
  '接着是 <span class="v">if (is_neox) ... else if (is_mrope &amp;&amp; !is_vision) ... ' +
    'else if (is_vision) ... else ...</span> 四路，每路再按 dtype 三选一。',
  '跨课呼应：<span class="k">同一个 GGML_OP_ROPE，在图上是同一个节点</span>，' +
    '差别全在 op_params。'
];
defs.forEach((_, i) => tl.at(700 + i * 2900, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.3'; });
  msg.innerHTML = texts[Math.min(i, 2)];
}));
tl.at(12400, () => { msg.innerHTML = texts[3]; U.markLines(document, [0, 1, 2, 3]); });
tl.at(14500, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ================================================================== 第 6 幕

L.scene(
    kicker='L6-04 · ssm-scan.cu',
    title='ssm-scan.cu：57 行 switch —— <span class="hl-a">手写的实例表</span>',
    sub='同一个 kernel 模板，按 token 数 1..8 各实例化一份（循环可完全展开），其它走通用版。',
    caption='这就是 template-instances/ 想自动化掉的东西 —— 只不过这里组合少，源码选择了手写。',
    src=SRC_SSM, parts=[(282, 338)], duration=18000,
    mark_src=[282, 284, 285, 326, 332, 333],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="lane" style="gap:6px"></div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const lane = wrap.querySelector('#lane');
const chips = ['1', '2', '3', '4', '5', '6', '7', '8', '其它'].map((n, i) => {
  const c = U.chip('n_tok = ' + n, i < 8 ? 'a' : 'd');
  lane.appendChild(c); return c;
});

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin:0 0 4px">模板参数（source.md 第六节）</div>' +
  '<div class="card" style="border-left-color:var(--b)"><div class="cb">' +
  '<span class="cm" style="margin:0">template &lt;size_t splitD, size_t N, size_t L_template&gt;</span><br>' +
  '<b>L_template</b> = 序列长度：1..8 时循环长度已知，编译器可展开；<br>' +
  '<b>L_template = 0</b> 时长度走运行期参数 <span class="cm" style="margin:0">L_param</span>。' +
  '</div></div>';
const right = wrap.querySelector('#right');
right.innerHTML = '<div class="cm" style="margin:0 0 4px">同一文件里的第二条路</div>' +
  '<div class="card" style="border-left-color:var(--e)"><div class="cb">' +
  '长序列且是 Mamba-2（<span class="cm" style="margin:0">n_tok &gt; SSM_SSD_MIN_TOKENS</span>，' +
  '阈值 128）时改走 <b>SSD</b> 路径：把 scan 变成 matmul，还要卡设备能力 ' +
  '<span class="cm" style="margin:0">cc &gt;= GGML_CUDA_CC_TURING</span>。' +
  '</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  'SSM 的 scan 是<b>顺序依赖</b>的：第 t 步要第 t-1 步的 state。',
  '源码的应对：把 <span class="k">序列长度做进模板</span>。' +
    '<span class="v">ssm_scan_f32&lt;threads, 16, 1&gt;</span> 到 <span class="v">&lt;threads, 16, 8&gt;</span> 各一份。',
  '<span class="v">case 1..8</span> 分别调用这 8 份；<span class="v">default</span> 用 ' +
    '<span class="v">L_template = 0</span> 的通用版。',
  '这就是<b>手写的实例表</b>：一个模板参数取值 -> 一行 case。' +
    '<span class="k">第 9 幕的 template-instances/ 是同一件事，只是组合太多、写不下，改用脚本生成。</span>',
  '同时它还有第二条路：<span class="v">use_ssd</span> 成立时整个 scan 变成 matmul ' +
    '—— 和 L6-02 的主题接上。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
chips.forEach((c, i) => tl.at(3000 + i * 650, () => c.style.opacity = (i < 8 ? '1' : '.35')));
tl.at(9200, () => { msg.innerHTML = texts[1]; U.markLines(document, [2, 3]); });
tl.at(11500, () => { msg.innerHTML = texts[2]; U.markLines(document, [3, 44, 50]); chips.forEach(c => c.style.opacity = '1'); });
tl.at(14200, () => { msg.innerHTML = texts[3]; });
tl.at(16200, () => { msg.innerHTML = texts[4]; });
'''
)

# ================================================================== 第 7 幕

L.scene(
    kicker='L6-04 · top-k.cu',
    title='top-k.cu：一条 <span class="hl-c">多段式</span> 的 kernel 流水线',
    sub='排序类算子在 GPU 上常常不是"一个 kernel"，而是几个小 kernel 依次跑，中间结果放显存。',
    caption='同一个文件里还有两条备选路径（CUB / 按位排序），由编译期宏决定，见 source.md 第七节。',
    src=SRC_TOPK, parts=[(180, 209)], duration=18000,
    mark_src=[180, 183, 184, 185, 186, 193, 195, 196, 197, 199, 200, 205, 206],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row center" id="flow" style="gap:5px"></div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const flow = wrap.querySelector('#flow');
const stages = ['init', 'histogram', 'select', 'x4 轮', 'reset', 'gather'];
const chips = stages.map((s, i) => {
  const c = U.chip(s, i === 3 ? 'd' : 'c');
  flow.appendChild(c);
  if (i < stages.length - 1) flow.appendChild(U.arrow('->'));
  return c;
});

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin:0 0 4px">基数选择（radix select）</div>' +
  '<div class="card" style="border-left-color:var(--a)"><div class="cb">' +
  '<span class="cm" style="margin:0">RADIX_BITS = 8</span>：每轮看 8 个 bit；' +
  'float 是 32 bit，所以 <span class="cm" style="margin:0">32 / 8 = 4</span> 轮。<br>' +
  '每轮：先按这一段 bit 统计直方图（<b>histogram</b>），再决定第 k 大落在哪个桶（<b>select</b>）。' +
  '</div></div>' +
  '<div class="card" style="border-left-color:var(--b)"><div class="cb">' +
  '中间状态 <span class="cm" style="margin:0">top_k_radix_state</span> 与直方图都从 ' +
  '<span class="cm" style="margin:0">ctx.pool()</span> 分配 —— 复用后端的内存池。' +
  '</div></div>';
const right = wrap.querySelector('#right');
right.innerHTML = '<div class="cm" style="margin:0 0 4px">为什么不是一个 kernel？</div>' +
  '<div class="card" style="border-left-color:var(--e)"><div class="cb">' +
  '每一轮都要<b>全局同步</b>：直方图必须全部写完，select 才能读。' +
  '拆成多个 kernel、放进同一条 stream，用 stream 的顺序性换同步 —— ' +
  '这是 GPU 上做迭代算法的常见办法。' +
  '</div></div>' +
  '<div class="card" style="border-left-color:var(--f)"><div class="cb">' +
  '最后 <span class="cm" style="margin:0">top_k_radix_gather</span> 把选中的 k 个下标写进 dst。' +
  '</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  'top-k 的第一步是<b>找第 k 大的值</b>，第二步才是收集下标。',
  '<span class="v">RADIX_BITS = 8</span>：一次处理 8 个 bit，32 位浮点要 <span class="k">4 轮</span>。',
  '每轮两段：<span class="v">histogram</span>（多 block 统计）-> <span class="v">select</span>（单 block 决策）。',
  '轮次之间需要全局可见性，所以只能是<b>多个 kernel 顺序启动</b>，同一 stream 保证次序。',
  '一句话：<span class="k">GPU 上的"排序算子"通常是一串 kernel，而不是一个大 kernel</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
tl.at(3400, () => { chips[3].style.opacity = '1'; msg.innerHTML = texts[1]; U.markLines(document, [3, 4, 15]); });
tl.at(7000, () => { msg.innerHTML = texts[2]; U.markLines(document, [15, 16, 17, 18, 19]); });
tl.at(11000, () => { msg.innerHTML = texts[3]; U.markLines(document, [0, 1, 2, 3, 4]); });
tl.at(15000, () => { msg.innerHTML = texts[4]; chips.forEach(c => c.style.opacity = '1'); });
'''
)

# ================================================================== 第 8 幕

L.scene(
    kicker='L6-04 · 洞察',
    title='★ <span class="hl-a">不是每个 .cu 都是一个 op</span>：moe 归约是个融合',
    sub='moe-weighted-reduction.cu 里没有 GGML_OP —— 它由图级模式匹配触发，跳过一整段子图。',
    caption='匹配规则在 ggml-cuda.cu 的 ggml_cuda_try_fuse() 里（source.md 第九节逐字引用）。',
    src=SRC_MOE, parts=[(3, 39)], duration=19000,
    mark_src=[3, 9, 10, 15, 16, 17, 19, 22, 24, 27, 35, 36],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="roles" style="gap:8px"></div>
  <div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', t: '① op 实现', b: '被 compute_forward 的 switch 调<br>norm / rope / softmax / ssm-scan / top-k', m: 'ggml_cuda_op_*' },
  { c: 'd', t: '② 别人的助手', b: '不被 switch 调，被别的 kernel 调<br>quantize.cu 给 mmq / mmvq 准备 Q8_1 输入', m: 'quantize_mmq_q8_1_cuda' },
  { c: 'e', t: '③ 图级融合', b: '由模式匹配触发，吃掉一整段子图<br>moe-weighted-reduction.cu 就是这一类', m: 'ggml_cuda_try_fuse' }
];
const host = wrap.querySelector('#roles');
const els = defs.map(d => U.card(d, { style: 'width:218px' }));
els.forEach(e => host.appendChild(e));
els.forEach(e => e.style.opacity = '.3');

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin:0 0 4px">kernel 在算什么</div>' +
  '<div class="card" style="border-left-color:var(--e)"><div class="cb">' +
  '一个线程负责<b>一个输出列</b>（<span class="cm" style="margin:0">col = blockIdx.y*blockDim.x + threadIdx.x</span>），' +
  '然后<b>串行累加 n_expert_used 个专家</b>。' +
  '</div></div>';
const right = wrap.querySelector('#right');
right.innerHTML = '<div class="cm" style="margin:0 0 4px">融合掉了什么</div>' +
  '<div class="card" style="border-left-color:var(--a)"><div class="cb">' +
  '长形式是 <b>2k+1</b> 个节点（k = 专家数）；源码注释写明它从 ' +
  '<span class="cm" style="margin:0">GGML_OP_MUL</span> 起头，上限 ' +
  '<span class="cm" style="margin:0">MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15</span>。' +
  '</div></div>' +
  '<div class="card" style="border-left-color:var(--b)"><div class="cb">' +
  '融合成功则一次 kernel 调用替代 2k+1 次节点计算 —— ' +
  '<b>省的是 kernel 启动与中间张量的读写</b>。' +
  '</div></div>';

const msg = wrap.querySelector('#msg');
const texts = [
  '把 ggml-cuda/ 里的 .cu 按"谁调它"分成三类。',
  '<span class="v">ggml_cuda_op_moe_weighted_reduction</span> 的入口收的是 experts / weights ' +
    '两个张量，不是图节点 —— 它没有 <span class="v">dst->op</span> 可查。',
  '在 <span class="v">compute_forward</span> 里找不到 ' +
    '<span class="v">GGML_OP_MOE_WEIGHTED</span>：<span class="k">它不是算子，是融合规则</span>。',
  '它由 <span class="v">ggml_cuda_try_fuse()</span> 在遍历图时匹配：' +
    '<span class="v">GGML_OP_MUL</span> 起头 + 形状判据（weights 的 ne[0] == 1 且与 MUL 同形）。',
  '所以"读 CUDA 后端"不能只读分派表：<span class="k">还有一条图级的路径</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
els.forEach((e, i) => tl.at(3600 + i * 3200, () => {
  els.forEach((x, k) => x.style.opacity = k === i ? '1' : '.3');
  msg.innerHTML = texts[i + 1];
}));
tl.at(16800, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ================================================================== 第 9 幕

L.scene(
    kicker='L6-04 · 洞察',
    title='★ <span class="hl-a">template-instances/</span>：为什么这个目录必须存在',
    sub='一个实例文件只有 5 行：一句"脚本生成"，一个 include，一行不带 extern 的宏。',
    caption='头文件里是带 extern 的同名宏 —— 声明与定义分家，正是显式实例化的标准写法。',
    src=TI_MMQ, parts=[(1, 5)], duration=20000,
    mark_src=[1, 3, 5],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row" style="gap:9px">
    <div class="col grow" id="left" style="gap:7px"></div>
    <div class="col grow" id="right" style="gap:7px"></div>
  </div>
  <div id="tbl"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const TI = @TI@;

const left = wrap.querySelector('#left');
left.innerHTML = '<div class="cm" style="margin:0 0 4px">头文件那一半（mmq.cuh，见第十节）</div>' +
  '<div class="card" style="border-left-color:var(--a)"><div class="cb">' +
  '<span class="cm" style="margin:0">#define DECL_MMQ_CASE(type) template void mul_mat_q_case&lt;type&gt;(...)</span><br>' +
  '<span class="cm" style="margin:0">extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);</span> ...<br>' +
  '带 <b>extern</b> = <b>显式实例化声明</b>：告诉每个包含 mmq.cuh 的 TU' +
  '"别在这儿实例化，符号在别处"。' +
  '</div></div>';
const right = wrap.querySelector('#right');
right.innerHTML = '<div class="cm" style="margin:0 0 4px">实例文件这一半（左边代码）</div>' +
  '<div class="card" style="border-left-color:var(--c)"><div class="cb">' +
  '<span class="cm" style="margin:0">#include "../mmq.cuh"</span><br>' +
  '<span class="cm" style="margin:0">DECL_MMQ_CASE(GGML_TYPE_Q4_0);</span><br>' +
  '同一个宏，<b>不带 extern</b> = <b>显式实例化定义</b>：这个 TU 负责生成 ' +
  '<span class="cm" style="margin:0">mul_mat_q_case&lt;GGML_TYPE_Q4_0&gt;</span> 的代码。' +
  '</div></div>' +
  '<div class="card" style="border-left-color:var(--d)"><div class="cb">' +
  '删掉目录会怎样：这个名字<b>没有任何 TU 提供定义</b>，而 extern 声明又抑制了隐式实例化 ' +
  '—— 链接期 <span class="cm" style="margin:0">undefined reference</span>。' +
  '</div></div>';

const t = U.table(
  ['族（前缀）', '文件数', '每个文件实例化什么', '模板参数空间'],
  [['mmq-instance-*', String(TI[0]), 'DECL_MMQ_CASE(一个量化类型)', '22 个量化类型'],
   ['mmf-instance-ncols_*', String(TI[1]), 'DECL_MMF_CASE(n)', 'n = 1..16'],
   ['fattn-vec-instance-*', String(TI[2]), '3 行 DECL_FATTN_VEC_CASE(64/128/256)', '7 个 type_K x 7 个 type_V'],
   ['fattn-tile-instance-*', String(TI[3]), '一行 DECL_FATTN_TILE_CASE(DKQ, DV)', '12 组 (DKQ, DV)'],
   ['fattn-mma-f16-instance-*', String(TI[4]), '2-8 行 DECL_FATTN_MMA_F16_CASE(...)', '21 组 (ncols1, ncols2)']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '这个文件一共 5 行：一句"脚本生成、别手改"，一个 include，一行宏。' +
  '<span class="k">它的全部内容就是"我要实例化这一种组合"。</span>',
  '因为 CUDA 模板（含 <span class="v">__global__</span> 模板）只有在被实例化时才生成设备代码，' +
  '而实例化必须发生在某个 TU 里。',
  'mmq.cuh 用 <b>extern 声明</b>把 22 个类型全部"挂起来"，' +
  '于是定义只能来自那 22 个文件 —— 一个文件一个类型。',
  '<span class="k">模板参数空间有多大，文件就有多少</span>：' +
  'fattn 的 82 个文件 = head size x ncols x dtype 的组合数（L6-03）。',
  '一句话：<span class="k">template-instances/ 是"显式实例化的落地点"，不是"重复代码"</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; U.markLines(document, [0, 4]); });
tl.at(4600, () => { msg.innerHTML = texts[1]; });
tl.at(8600, () => { msg.innerHTML = texts[2]; });
rows.forEach((r, i) => tl.at(11000 + i * 1300, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
}));
tl.at(17800, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[3]; });
tl.at(19200, () => { msg.innerHTML = texts[4]; });
'''.replace('@TI@', _JS_TI)
)

# ================================================================== 第 10 幕

L.scene(
    kicker='L6-04 · 收束',
    title='把这一课压成一张表',
    sub='六个算子、三类角色、一个目录。',
    caption='下一课 L6-05 换一个后端：SYCL 如何用同一套算子结构重写一遍。',
    src=TI_FVEC, parts=[(1, 7)], duration=22000,
    mark_src=[1, 5, 6, 7],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['文件', '角色', '这一课看到的关键点'],
  [['ggml-cuda.cu', '分派', 'dst->op 一个大 switch，每个 case 一个 ggml_cuda_op_* 入口'],
   ['norm.cu', 'op 实现', '一行一个 block + block_reduce<SUM>；do_multiply/do_add 是模板开关'],
   ['softmax.cu', 'op 实现', 'integral_constant 折叠挑编译期 ncols；<true,0,0> 兜底'],
   ['rope.cu', 'op 实现', 'mode 的四个位 -> 四种成对方式，再按 dtype 分派'],
   ['ssm-scan.cu', 'op 实现', '57 行 switch = 手写实例表；长序列改走 SSD matmul'],
   ['top-k.cu', 'op 实现', '基数选择拆成多个 kernel，靠 stream 顺序做同步'],
   ['quantize.cu', '助手', '不是 op：给 mmq / mmvq 准备 Q8_1 输入'],
   ['moe-weighted-reduction.cu', '图级融合', '由 ggml_cuda_try_fuse 匹配子图，不在 switch 里'],
   ['template-instances/', '生成', '显式实例化定义，一类组合一个 TU']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

wrap.querySelector('#ex').appendChild(W.exercise(
  '为什么 <span class="mono">ggml/src/ggml-cuda/template-instances/</span> 这个目录必须存在？' +
  '把其中 <span class="mono">mmq-instance-q4_0.cu</span> 删掉会发生什么？',
  '因为"只在头文件里写模板"在这里行不通：<br>' +
  '1. <span class="mono">mmq.cuh</span> 对 22 个量化类型写的是<b>带 extern</b> 的 ' +
  '<span class="mono">DECL_MMQ_CASE(...)</span>（显式实例化<b>声明</b>，mmq.cuh:1571-1580）；<br>' +
  '2. extern 声明会<b>抑制</b>包含它的 TU 里的隐式实例化 —— 符号必须由别处提供；<br>' +
  '3. 提供符号的就是 <span class="mono">template-instances/</span> 里那 22 个文件：' +
  '每个文件一行<b>不带 extern</b> 的同一个宏（显式实例化<b>定义</b>）。<br>' +
  '所以删掉 <span class="mono">mmq-instance-q4_0.cu</span>，' +
  '<span class="mono">mul_mat_q_case&lt;GGML_TYPE_Q4_0&gt;</span> 就没有任何 TU 提供定义，' +
  '链接期报 undefined reference。<br>' +
  '更大的意义：文件数 = 模板参数空间的组合数（mmq 22 / mmf 16 / fattn 82，共 120）。'));

const msg = wrap.querySelector('#msg');
const texts = [
  '最后回到验收点：<span class="k">template-instances/ 是显式实例化的落地点</span>。',
  '六个算子的共同点：<span class="v">运行期形状 -> 选 kernel / 定 grid</span>，' +
    '<span class="v">编译期参数 -> 定能不能展开、要不要融合</span>。',
  '三类角色的共同点：<span class="k">不是所有 .cu 都挂在 dst->op 上</span>。',
  '下一课 L6-05：SYCL 后端怎么用同一套算子结构重写一遍。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2200 + i * 1500, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
}));
tl.at(16000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[1]; });
tl.at(18500, () => { msg.innerHTML = texts[2]; });
tl.at(20500, () => { msg.innerHTML = texts[3]; });
'''
)

# ================================================================== source.md

L.section(
    '一、本课的清单是怎么数出来的',
    '这一课覆盖 `plan_matrix` 里 **L6-04** 的**全部**文件。清单与计数都用命令取，不靠肉眼：\n\n'
    '```text\n'
    'python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L6-04"{print $2}\' | wc -l\n'
    '  -> 280\n'
    'python3 tools/plan_matrix.py --files | awk -F\'\\t\' \'$1=="L6-04"{print $2}\' '
    '| grep -c template-instances/\n'
    '  -> 120\n'
    'ls ggml/src/ggml-cuda/template-instances/*.cu | wc -l\n'
    '  -> 120\n'
    '```\n\n'
    '280 个文件的构成（按后缀实测）：\n\n'
    '| 来源 | 数量 | 说明 |\n|---|---|---|\n'
    '| `template-instances/*.cu` | 120 | 全部由脚本生成 |\n'
    '| 手写 `.cu` | 68 | 算子实现 / 助手 / 融合 |\n'
    '| 手写 `.cuh` | 89 | 头文件：声明、kernel 模板、配置表 |\n'
    '| `vendors/*.h` | 3 | `cuda.h` / `hip.h` / `musa.h` 厂商适配层 |\n\n'
    '> `template-instances/` 目录里还有第 121 个文件 `generate_cu_files.py`（生成脚本本身）。'
    '它是 `.py`，不在本视角的覆盖域（覆盖域后缀见 `tools/repo_universe.py`），**不计入**'
    '本课的 280 个声明。每个生成文件的第一行都写着它的来历：'
    '`This file has been autogenerated by generate_cu_files.py, do not edit manually.`',
    src=SRC_CUDA, parts=[(5, 49)], lang='c')

L.section(
    '二、分派：ggml_cuda_compute_forward 的 switch',
    '回顾 L6-01：CUDA 后端对外的唯一入口是 `ggml_cuda_compute_forward()`，'
    '它以 `dst->op` 做 switch，每个 case 调一个 `ggml_cuda_op_*`。'
    '本课讲的算子都在这张表上 —— 它们和 MUL_MAT 挤在一起，没有特殊通道。\n\n'
    '下面按上游行号拼出与本课直接相关的几段（分隔行标出真实区间）：',
    src=SRC_CUDA, parts=[(2067, 2070), (2253, 2256), (2301, 2308), (2355, 2366)], lang='c')

L.section(
    '三、norm.cu：kernel 模板头与 launch 配置',
    '幕 3 引用的片段在 kernel 中段（平方和规约与写回）。它的模板头与 grid 配置在这里：'
    '`do_multiply` / `do_add` 是**编译期**开关；`blocks_num(nrows, nchannels, nsamples)` '
    '让"一行一个 block"成为结构性事实；线程数按 `ncols` 是否小于 1024 在 256 与 1024 之间选，'
    '实例化出来的是 `rms_norm_f32<256, false>` 与 `rms_norm_f32<1024, false>`。',
    src=SRC_NORM, parts=[(76, 79), (304, 319)], lang='c')

L.section(
    '四、softmax.cu：8 个编译期列数与通用兜底',
    '幕 4 引用了选择机制本身。这里补上**调用点** —— 它把候选列数写成模板实参列表：'
    '`launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>`；'
    '共享内存不够时才走另一条路（跨 SM 协作启动）。',
    src=SRC_SOFTMAX, parts=[(342, 356), (278, 296)], lang='c')

L.section(
    '五、rope.cu：四个 kernel，两种成对方式',
    '幕 5 引用的是 `mode` 的解析与四路分支的入口。这里补上分支本体：'
    '`is_neox` 一路的三种 dtype 组合、`is_mrope` 的 `rope_multi_cuda`、'
    '`is_vision` 的 `rope_vision_cuda`，以及默认一路的 `rope_norm_cuda` —— '
    '可以看到每个 kernel 都有 `<forward, T, T>` 形式的模板参数：**方向**也是编译期的。',
    src=SRC_ROPE, parts=[(631, 650), (662, 679)], lang='c')

L.section(
    '六、ssm-scan.cu：模板参数与"第二条路"的判据',
    '幕 6 引用了 `switch (n_tok)` 的 57 行实例表。这里补三处：'
    '① `SSM_SSD_MIN_TOKENS` 的定义（阈值 128 token）；'
    '② 模板头的三个参数（`splitD` / `N` / `L_template`）与 `L_template == 0` 的含义；'
    '③ 选择走 SSD（把 scan 变成 matmul）的**完整判据** —— 它同时看张量形状、'
    'token 数区间与设备计算能力。',
    src=SRC_SSM, parts=[(13, 46), (824, 843)], lang='c')

L.section(
    '七、top-k.cu：同一个算子，三条实现路径',
    '幕 7 引用了 radix 这条路的多个 kernel。但同一个 `ggml_cuda_op_top_k` 还挂着两条'
    '备选路径，由**编译期宏**决定走哪条：能拿到 CUB 时用 `top_k_cub`；'
    '否则退化成 "argsort + 拷贝"，其中 argsort 又按共享内存是否放得下，'
    '在按位排序与 CUB 之间选。这段也是"多 kernel 选一"的好例子。',
    src=SRC_TOPK, parts=[(229, 270)], lang='c')

L.section(
    '八、quantize.cu：不被 switch 调的"助手"',
    '`quantize.cu` 里没有 `ggml_cuda_op_quantize`：它提供的是 '
    '`quantize_row_q8_1_cuda` 与 `quantize_mmq_q8_1_cuda`，由 **mmq / mmvq** 调用（L6-02）。'
    '注意 `quantize_mmq_q8_1_cuda` 内部的 switch —— 它按 '
    '`mmq_get_q8_1_ds_layout(type_src0)` 选模板实例：'
    '**量化块的内部布局必须与 mmq 的读法一致**。',
    src=SRC_QUANT, parts=[(571, 600)], lang='c')

L.section(
    '八（续）、调用方：mmq.cu 里的那一行',
    '`quantize_mmq_q8_1_cuda` 的调用点在 `mmq.cu` 的 mul_mat_q 路径里：'
    '激活 `src1` 先被量化成 Q8_1，再交给 mmq 的 kernel；'
    'NVFP4 走的是另一个量化入口（`quantize_mmq_fp4_cuda`）。'
    '这就是"助手"的含义 —— 它服务的是 L6-02 的主角。',
    src=SRC_MMQ, parts=[(144, 158)], lang='c')

L.section(
    '八（续）、另一个调用方：mmvq.cu',
    'mmvq 走的是**另一个**助手入口 `quantize_row_q8_1_cuda`（L6-02 的 mmvq 路径），'
    '同样是"先量化激活、再做点积"。两个调用点合起来说明：'
    '`quantize.cu` 不是算子，是矩阵乘路径的前置步骤。',
    src=SRC_MMVQ, parts=[(1500, 1507)], lang='c')

L.section(
    '九、★ moe-weighted-reduction.cu：图级融合，不是算子',
    '幕 8 引用了这个 kernel 与它的 launcher。它由 `ggml_cuda_try_fuse()` 在遍历图时'
    '匹配触发，匹配规则在 `ggml-cuda.cu` 里。判据有两层：入口节点的 op / 类型 / 连续性，'
    '以及"哪个输入是 experts、哪个是 broadcast 的 weights"；'
    '融合前还要过一次内存区间检查。\n\n'
    '为什么说"它不是算子"？因为这份文件里没有 `GGML_OP` 常量可挂：\n\n'
    '```text\n'
    'grep -c \'GGML_OP_MOE_WEIGHTED\' ggml/src/ggml-cuda/ggml-cuda.cu\n'
    '  -> 0\n'
    '```\n\n'
    '触发它的入口在 `ggml_cuda_try_fuse()` 里，以 `GGML_OP_MUL` 为起点。',
    src=SRC_CUDA, parts=[(3035, 3060), (3441, 3449)], lang='c')

L.section(
    '十、★ template-instances/：显式实例化的两半',
    '这一节把"为什么必须存在"拆成可核对的两半：**头文件里的 extern 声明**，与'
    '**实例文件里的定义**。两者用的是同一个宏、同一份模板，区别只在 `extern` 这个词。'
    '先看 mmq 族：`DECL_MMQ_CASE(type)` 展开成一条函数模板的显式实例化；'
    '带 `extern` 的 22 行在头文件里，不带 `extern` 的 22 行在 22 个实例文件里。',
    src=CUH_MMQ, parts=[(1571, 1580)], lang='c')

L.section(
    '十（续）、mmf：声明侧两种宏只差一个 extern',
    '`DECL_MMF_CASE_EXTERN(n)` 与 `DECL_MMF_CASE(n)` 展开出的是同一批 6 个实例'
    '（float / half2 / bfloat162 x 两种 ROWS_PER_BLOCK），区别只在前者多一个 `extern`。'
    '后者被 16 个实例文件调用，前者被头文件调用 16 次。',
    src=CUH_MMF, parts=[(882, 903)], lang='c')

L.section(
    '十（续）、fattn-vec：一个宏展开出 7 种 type_V',
    '`EXTERN_DECL_FATTN_VEC_CASES(D, type_K)` 一次展开出 7 条声明（F16 / Q4_0 / Q4_1 / '
    'Q5_0 / Q5_1 / Q8_0 / BF16）。21 次调用 = 147 条声明，落到 49 个实例文件里'
    '（每个文件含 D = 64 / 128 / 256 三行）。',
    src=CUH_FVEC, parts=[(574, 586)], lang='c')

L.section(
    '十（续）、fattn-tile：12 组 (DKQ, DV)',
    '12 条 `extern` 对应 12 个实例文件，一个文件一组头维度。',
    src=CUH_FTILE, parts=[(1340, 1349)], lang='c')

L.section(
    '十（续）、fattn-mma-f16：一个二元宏展开 5 个 ncols2',
    '`DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)` 把一个 head size 展开成 '
    '5 个 (ncols1, ncols2) 组合；每个实例文件反过来，固定一组 (ncols1, ncols2) '
    '而列出多个 head size。',
    src=CUH_FMMA, parts=[(2121, 2130)], lang='c')

L.section(
    '十（续）、实例侧：一个文件一行（mmq）',
    '五个族的实例文件各看一个。第一行都是同一句"脚本生成、别手改"，'
    '中间的 include 指向本族的头文件，最后一行是**不带 extern** 的宏调用 —— '
    '**这一行就是定义**。先看 mmq：',
    src=TI_MMQ, parts=[(1, 5)], lang='c')

L.section(
    '十（续）、实例侧：mmf',
    '`mmf-instance-ncols_1.cu` 的全部内容就是"实例化 ncols_dst = 1"这一种组合。',
    src=TI_MMF, parts=[(1, 5)], lang='c')

L.section(
    '十（续）、实例侧：fattn-vec',
    '一个文件覆盖 D = 64 / 128 / 256 三个头维度 —— 所以"49 个文件"对应的是 '
    '7 x 7 种 (type_K, type_V) 组合，而不是 147 个文件。',
    src=TI_FVEC, parts=[(1, 7)], lang='c')

L.section(
    '十（续）、实例侧：fattn-tile',
    '`fattn-tile-instance-dkq128-dv128.cu` 只有一行 `DECL_FATTN_TILE_CASE(128, 128)`。',
    src=TI_FTILE, parts=[(1, 5)], lang='c')

L.section(
    '十（续）、实例侧：fattn-mma-f16',
    '`fattn-mma-f16-instance-ncols1_16-ncols2_1.cu` 固定 (ncols1, ncols2) = (16, 1)，'
    '列出 6 个 head size。族内文件的行数并不相同（实测 2 到 8 行），'
    '所以"文件数"与"实例化语句数"是两个不同的数字。',
    src=TI_FMMA, parts=[(1, 10)], lang='c')

L.section(
    '十一、按前缀族的分类与计数（全部实测）',
    '`template-instances/` 的 120 个文件按文件名前缀分成五族。文件数是 `ls` 数出来的，'
    '"每个文件里几行实例化语句"是数 `DECL_` 开头的行得到的：\n\n'
    '| 族（前缀） | 文件数 | 每文件实例化语句数 | 模板参数空间 |\n|---|---|---|---|\n'
    '| `mmq-instance-*` | 22 | 1 | 22 个量化类型（与 `mmq.cuh` 的 22 条 extern 一一对应） |\n'
    '| `mmf-instance-ncols_*` | 16 | 1 | n = 1..16（与 16 条 `DECL_MMF_CASE_EXTERN` 对应） |\n'
    '| `fattn-vec-instance-*` | 49 | 3 | 7 个 type_K x 7 个 type_V = 49 |\n'
    '| `fattn-tile-instance-*` | 12 | 1 | 12 组 (DKQ, DV)（与 12 条 extern 对应） |\n'
    '| `fattn-mma-f16-instance-*` | 21 | 2-8（合计 126 行） | 21 组 (ncols1, ncols2) |\n'
    '| `fattn-*` 三族小计 | **82** | 285 行 | FlashAttention 的三条实现路线（L6-03） |\n'
    '| **合计** | **120** | 323 行 | — |\n\n'
    '```text\n'
    'ls ggml/src/ggml-cuda/template-instances/*.cu | wc -l                       -> 120\n'
    'ls ggml/src/ggml-cuda/template-instances/mmq-instance-*.cu | wc -l         -> 22\n'
    'ls ggml/src/ggml-cuda/template-instances/mmf-instance-*.cu | wc -l         -> 16\n'
    'ls ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*.cu | wc -l   -> 49\n'
    'ls ggml/src/ggml-cuda/template-instances/fattn-tile-instance-*.cu | wc -l  -> 12\n'
    'ls ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-*.cu | wc -l -> 21\n'
    'grep -c \'^extern DECL_MMQ_CASE\' ggml/src/ggml-cuda/mmq.cuh               -> 22\n'
    'grep -c \'^DECL_MMF_CASE_EXTERN\' ggml/src/ggml-cuda/mmf.cuh               -> 16\n'
    'grep -c \'^extern DECL_FATTN_TILE_CASE\' ggml/src/ggml-cuda/fattn-tile.cuh -> 12\n'
    '```\n\n'
    '一句话：**模板参数空间有多大，文件就有多少。**')

L.section(
    '十二、结论：三类角色，一个目录',
    '把这一课压成三句话：\n\n'
    '1. **算子实现是"入口函数 + kernel 模板"**：入口函数从 `dst` 取形状与 `op_params`，'
    'kernel 用模板参数表达编译期选择（可展开的循环、要不要融合、块大小）。\n'
    '2. **不是所有 .cu 都是 op**：`quantize.cu` 是 mmq / mmvq 的助手；'
    '`moe-weighted-reduction.cu` 是图级融合，由 `ggml_cuda_try_fuse()` 触发。\n'
    '3. **`template-instances/` 是显式实例化的落地点**：头文件用 `extern` 声明把它们"挂起来"，'
    '每个实例文件用一个不带 `extern` 的宏提供定义，一个组合一个 TU。'
    '删掉它们，链接期就会 `undefined reference`。')

L.footnote_add('本课覆盖声明 = `tools/plan_matrix.py` 里 L6-04 的**全部** %d 个文件'
               '（160 个手写 + 120 个 `template-instances/*.cu`）。'
               'spec 在 build 时会拿这份清单与计划对账，不一致直接报错。' % N_FILES)
L.footnote_add('`ggml/src/ggml-cuda/template-instances/generate_cu_files.py` 是生成器本身，'
               '后缀 `.py` 不在覆盖域内，**不计入**本课的 280 个声明。')
L.footnote_add('`ggml/include/ggml-cuda.h`（公共后端 API）属于 L6-01 的清单，不在本课目录内，'
               '因此也不在本课声明里。')
L.footnote_add('本课不涉及任何命令行参数；参数门禁的真值集来自真实二进制的帮助输出。')

L.prereqs('`L6-03`（★ CUDA FlashAttention）；本课默认你已经看过 `L6-01`（后端骨架与分派）与 '
          '`L6-02`（量化矩阵乘）。')

L.goal(
    '说出 `ggml/src/ggml-cuda/` 下两类文件的分工，以及本课清单的规模'
    '（280 = 160 手写 + 120 生成）；',
    '解释 `template-instances/` 目录为什么必须存在 —— 头文件的 `extern` 声明与实例文件的'
    '定义各是哪一半，删掉会怎样（对应验收点）；',
    '说出 `norm.cu` / `softmax.cu` / `rope.cu` / `ssm-scan.cu` / `top-k.cu` 各自'
    '"运行期形状 -> kernel" 的判据；',
    '举出至少两个"不是 op 实现"的 `.cu`，并说明它们分别被谁调用；',
    '指出 `ssm-scan.cu` 里"手写实例表"与 `template-instances/` 的关系。')

L.conclusion(
    '本课清单：280 = 160 + 120',
    '`ggml/src/ggml-cuda/` 下属于 L6-04 的 280 个文件：\n\n'
    '| 来源 | 数量 | 是什么 |\n|---|---|---|\n'
    '| 手写 `.cu` | 68 | 算子实现、助手、融合 |\n'
    '| 手写 `.cuh` | 89 | 声明、kernel 模板、配置表 |\n'
    '| `vendors/*.h` | 3 | 厂商适配层 |\n'
    '| `template-instances/*.cu` | 120 | 生成：显式实例化定义 |\n\n'
    'L6-01 / L6-02 / L6-03 覆盖的是这个目录的**子集**；本课是全目录声明。')

L.conclusion(
    '★ template-instances/ 为什么必须存在',
    '两半合起来才成立：\n\n'
    '```text\n'
    'mmq.cuh（声明侧）：  extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);   <- 显式实例化声明\n'
    'template-instances/mmq-instance-q1_0.cu（定义侧）：\n'
    '                     DECL_MMQ_CASE(GGML_TYPE_Q1_0);           <- 显式实例化定义\n'
    '```\n\n'
    '`extern` 抑制本 TU 的隐式实例化，符号必须由别处提供；实例文件就是"别处"。'
    '**模板参数空间有多大，文件就有多少**：'
    'mmq 22 / mmf 16 / fattn-vec 49 / fattn-tile 12 / fattn-mma-f16 21 = 120。')

L.conclusion(
    '★ 三类角色：op / 助手 / 融合',
    '| 角色 | 谁调用 | 例子 |\n|---|---|---|\n'
    '| op 实现 | `ggml_cuda_compute_forward` 的 switch | `norm.cu` `rope.cu` `softmax.cu` '
    '`ssm-scan.cu` `top-k.cu` |\n'
    '| 助手 | 别的 kernel | `quantize.cu`（mmq / mmvq 的 Q8_1 输入） |\n'
    '| 图级融合 | `ggml_cuda_try_fuse()` 的模式匹配 | `moe-weighted-reduction.cu` |\n\n'
    '所以"读完分派表"并不等于"读完后端"。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
    print('covered:', len(FILES), 'files =', len(FILES_HANDWRITTEN), 'handwritten +',
          len(FILES_TEMPLATE_INSTANCES), 'template-instances')
