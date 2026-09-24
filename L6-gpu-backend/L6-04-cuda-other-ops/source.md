<!-- llama-coverage
ggml/src/ggml-cuda/acc.cu
ggml/src/ggml-cuda/acc.cuh
ggml/src/ggml-cuda/add-id.cu
ggml/src/ggml-cuda/add-id.cuh
ggml/src/ggml-cuda/allreduce.cu
ggml/src/ggml-cuda/allreduce.cuh
ggml/src/ggml-cuda/arange.cu
ggml/src/ggml-cuda/arange.cuh
ggml/src/ggml-cuda/argmax.cu
ggml/src/ggml-cuda/argmax.cuh
ggml/src/ggml-cuda/argsort.cu
ggml/src/ggml-cuda/argsort.cuh
ggml/src/ggml-cuda/binbcast.cu
ggml/src/ggml-cuda/binbcast.cuh
ggml/src/ggml-cuda/clamp.cu
ggml/src/ggml-cuda/clamp.cuh
ggml/src/ggml-cuda/col2im-1d.cu
ggml/src/ggml-cuda/col2im-1d.cuh
ggml/src/ggml-cuda/common.cuh
ggml/src/ggml-cuda/concat.cu
ggml/src/ggml-cuda/concat.cuh
ggml/src/ggml-cuda/conv-transpose-1d.cu
ggml/src/ggml-cuda/conv-transpose-1d.cuh
ggml/src/ggml-cuda/conv2d-dw.cu
ggml/src/ggml-cuda/conv2d-dw.cuh
ggml/src/ggml-cuda/conv2d-transpose.cu
ggml/src/ggml-cuda/conv2d-transpose.cuh
ggml/src/ggml-cuda/conv2d.cu
ggml/src/ggml-cuda/conv2d.cuh
ggml/src/ggml-cuda/convert.cu
ggml/src/ggml-cuda/convert.cuh
ggml/src/ggml-cuda/count-equal.cu
ggml/src/ggml-cuda/count-equal.cuh
ggml/src/ggml-cuda/cp-async.cuh
ggml/src/ggml-cuda/cpy-utils.cuh
ggml/src/ggml-cuda/cpy.cu
ggml/src/ggml-cuda/cpy.cuh
ggml/src/ggml-cuda/cross-entropy-loss.cu
ggml/src/ggml-cuda/cross-entropy-loss.cuh
ggml/src/ggml-cuda/cumsum.cu
ggml/src/ggml-cuda/cumsum.cuh
ggml/src/ggml-cuda/dequantize.cuh
ggml/src/ggml-cuda/diag.cu
ggml/src/ggml-cuda/diag.cuh
ggml/src/ggml-cuda/diagmask.cu
ggml/src/ggml-cuda/diagmask.cuh
ggml/src/ggml-cuda/dsv4-hc.cu
ggml/src/ggml-cuda/dsv4-hc.cuh
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/fattn-mma-f16.cuh
ggml/src/ggml-cuda/fattn-tile.cu
ggml/src/ggml-cuda/fattn-tile.cuh
ggml/src/ggml-cuda/fattn-vec.cuh
ggml/src/ggml-cuda/fattn.cu
ggml/src/ggml-cuda/fattn.cuh
ggml/src/ggml-cuda/fill.cu
ggml/src/ggml-cuda/fill.cuh
ggml/src/ggml-cuda/fwht.cu
ggml/src/ggml-cuda/fwht.cuh
ggml/src/ggml-cuda/gated_delta_net.cu
ggml/src/ggml-cuda/gated_delta_net.cuh
ggml/src/ggml-cuda/getrows.cu
ggml/src/ggml-cuda/getrows.cuh
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/gla.cu
ggml/src/ggml-cuda/gla.cuh
ggml/src/ggml-cuda/im2col.cu
ggml/src/ggml-cuda/im2col.cuh
ggml/src/ggml-cuda/lightning-indexer.cu
ggml/src/ggml-cuda/lightning-indexer.cuh
ggml/src/ggml-cuda/mean.cu
ggml/src/ggml-cuda/mean.cuh
ggml/src/ggml-cuda/mma.cuh
ggml/src/ggml-cuda/mmf.cu
ggml/src/ggml-cuda/mmf.cuh
ggml/src/ggml-cuda/mmid.cu
ggml/src/ggml-cuda/mmid.cuh
ggml/src/ggml-cuda/mmq-config-ampere.cuh
ggml/src/ggml-cuda/mmq-config-blackwell.cuh
ggml/src/ggml-cuda/mmq-config-cdna.cuh
ggml/src/ggml-cuda/mmq-config-gcn.cuh
ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh
ggml/src/ggml-cuda/mmq-config-pascal-older.cuh
ggml/src/ggml-cuda/mmq-config-rdna2.cuh
ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh
ggml/src/ggml-cuda/mmq-config-rdna3.cuh
ggml/src/ggml-cuda/mmq-config-rdna4.cuh
ggml/src/ggml-cuda/mmq-load-tiles.cuh
ggml/src/ggml-cuda/mmq-vec-dot.cuh
ggml/src/ggml-cuda/mmq.cu
ggml/src/ggml-cuda/mmq.cuh
ggml/src/ggml-cuda/mmvf.cu
ggml/src/ggml-cuda/mmvf.cuh
ggml/src/ggml-cuda/mmvq.cu
ggml/src/ggml-cuda/mmvq.cuh
ggml/src/ggml-cuda/moe-weighted-reduction.cu
ggml/src/ggml-cuda/moe-weighted-reduction.cuh
ggml/src/ggml-cuda/norm.cu
ggml/src/ggml-cuda/norm.cuh
ggml/src/ggml-cuda/opt-step-adamw.cu
ggml/src/ggml-cuda/opt-step-adamw.cuh
ggml/src/ggml-cuda/opt-step-sgd.cu
ggml/src/ggml-cuda/opt-step-sgd.cuh
ggml/src/ggml-cuda/out-prod.cu
ggml/src/ggml-cuda/out-prod.cuh
ggml/src/ggml-cuda/pad.cu
ggml/src/ggml-cuda/pad.cuh
ggml/src/ggml-cuda/pad_reflect_1d.cu
ggml/src/ggml-cuda/pad_reflect_1d.cuh
ggml/src/ggml-cuda/pool1d.cu
ggml/src/ggml-cuda/pool1d.cuh
ggml/src/ggml-cuda/pool2d.cu
ggml/src/ggml-cuda/pool2d.cuh
ggml/src/ggml-cuda/quantize.cu
ggml/src/ggml-cuda/quantize.cuh
ggml/src/ggml-cuda/reduce_rows.cuh
ggml/src/ggml-cuda/roll.cu
ggml/src/ggml-cuda/roll.cuh
ggml/src/ggml-cuda/rope.cu
ggml/src/ggml-cuda/rope.cuh
ggml/src/ggml-cuda/scale.cu
ggml/src/ggml-cuda/scale.cuh
ggml/src/ggml-cuda/set-rows.cu
ggml/src/ggml-cuda/set-rows.cuh
ggml/src/ggml-cuda/set.cu
ggml/src/ggml-cuda/set.cuh
ggml/src/ggml-cuda/snake.cu
ggml/src/ggml-cuda/snake.cuh
ggml/src/ggml-cuda/softcap.cu
ggml/src/ggml-cuda/softcap.cuh
ggml/src/ggml-cuda/softmax.cu
ggml/src/ggml-cuda/softmax.cuh
ggml/src/ggml-cuda/solve_tri.cu
ggml/src/ggml-cuda/solve_tri.cuh
ggml/src/ggml-cuda/ssm-conv.cu
ggml/src/ggml-cuda/ssm-conv.cuh
ggml/src/ggml-cuda/ssm-scan.cu
ggml/src/ggml-cuda/ssm-scan.cuh
ggml/src/ggml-cuda/sum.cu
ggml/src/ggml-cuda/sum.cuh
ggml/src/ggml-cuda/sumrows.cu
ggml/src/ggml-cuda/sumrows.cuh
ggml/src/ggml-cuda/top-k.cu
ggml/src/ggml-cuda/top-k.cuh
ggml/src/ggml-cuda/topk-moe.cu
ggml/src/ggml-cuda/topk-moe.cuh
ggml/src/ggml-cuda/tri.cu
ggml/src/ggml-cuda/tri.cuh
ggml/src/ggml-cuda/tsembd.cu
ggml/src/ggml-cuda/tsembd.cuh
ggml/src/ggml-cuda/unary.cu
ggml/src/ggml-cuda/unary.cuh
ggml/src/ggml-cuda/upscale.cu
ggml/src/ggml-cuda/upscale.cuh
ggml/src/ggml-cuda/vecdotq.cuh
ggml/src/ggml-cuda/vendors/cuda.h
ggml/src/ggml-cuda/vendors/hip.h
ggml/src/ggml-cuda/vendors/musa.h
ggml/src/ggml-cuda/wkv.cu
ggml/src/ggml-cuda/wkv.cuh
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_32.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_16.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_32.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_16.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu
ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq112-dv112.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq128-dv128.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq192-dv128.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq256-dv256.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq320-dv256.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq40-dv40.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq512-dv512.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq576-dv512.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq64-dv64.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq72-dv72.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq80-dv80.cu
ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq96-dv96.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-f16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q4_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q5_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q5_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q8_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-bf16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q5_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q5_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q8_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-bf16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-f16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q5_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q5_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q8_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-bf16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-f16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q4_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q5_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q5_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q8_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-bf16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-f16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q4_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q8_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-bf16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-f16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q4_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q5_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q5_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q8_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-bf16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-f16.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q4_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q4_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q5_0.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q5_1.cu
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_1.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_10.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_11.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_12.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_13.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_14.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_15.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_16.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_2.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_3.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_4.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_5.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_6.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_7.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_8.cu
ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_9.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq1_s.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q1_0.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q2_k.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q3_k.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q4_1.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu
ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu
-->

# L6-04 · CUDA 其余算子与模板实例化 — 源文件

**一句话**：L6-01 讲了 CUDA 后端的骨架与分派、L6-02 讲了量化矩阵乘、L6-03 讲了 FlashAttention；这一课把 `ggml/src/ggml-cuda/` 剩下的部分全部收掉 —— **160 个手写文件 + 120 个生成的模板实例**，并回答一个具体问题：`template-instances/` 这个目录为什么必须存在。

这一课是本层唯一"全目录声明"的课：计划里 L6-04 的清单就是 `ggml/src/ggml-cuda/` 下的全部 280 个源文件，本课把它们**全部**写进覆盖声明。L6-01 / L6-02 / L6-03 覆盖的是这个目录的**子集**（`ggml/include/ggml-cuda.h` 这个公共头则归 L6-01）。

---

## 一、本课的清单是怎么数出来的

这一课覆盖 `plan_matrix` 里 **L6-04** 的**全部**文件。清单与计数都用命令取，不靠肉眼：

```text
python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L6-04"{print $2}' | wc -l
  -> 280
python3 tools/plan_matrix.py --files | awk -F'\t' '$1=="L6-04"{print $2}' | grep -c template-instances/
  -> 120
ls ggml/src/ggml-cuda/template-instances/*.cu | wc -l
  -> 120
```

280 个文件的构成（按后缀实测）：

| 来源 | 数量 | 说明 |
|---|---|---|
| `template-instances/*.cu` | 120 | 全部由脚本生成 |
| 手写 `.cu` | 68 | 算子实现 / 助手 / 融合 |
| 手写 `.cuh` | 89 | 头文件：声明、kernel 模板、配置表 |
| `vendors/*.h` | 3 | `cuda.h` / `hip.h` / `musa.h` 厂商适配层 |

> `template-instances/` 目录里还有第 121 个文件 `generate_cu_files.py`（生成脚本本身）。它是 `.py`，不在本视角的覆盖域（覆盖域后缀见 `tools/repo_universe.py`），**不计入**本课的 280 个声明。每个生成文件的第一行都写着它的来历：`This file has been autogenerated by generate_cu_files.py, do not edit manually.`

<!-- src: ggml/src/ggml-cuda/ggml-cuda.cu -->
```c
#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
```

## 二、分派：ggml_cuda_compute_forward 的 switch

回顾 L6-01：CUDA 后端对外的唯一入口是 `ggml_cuda_compute_forward()`，它以 `dst->op` 做 switch，每个 case 调一个 `ggml_cuda_op_*`。本课讲的算子都在这张表上 —— 它们和 MUL_MAT 挤在一起，没有特殊通道。

下面按上游行号拼出与本课直接相关的几段（分隔行标出真实区间）：

<!-- src: ggml/src/ggml-cuda/ggml-cuda.cu -->
```c
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:2253-2256 ----
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:2301-2308 ----
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:2355-2366 ----
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
```

## 三、norm.cu：kernel 模板头与 launch 配置

幕 3 引用的片段在 kernel 中段（平方和规约与写回）。它的模板头与 grid 配置在这里：`do_multiply` / `do_add` 是**编译期**开关；`blocks_num(nrows, nchannels, nsamples)` 让"一行一个 block"成为结构性事实；线程数按 `ncols` 是否小于 1024 在 256 与 1024 之间选，实例化出来的是 `rms_norm_f32<256, false>` 与 `rms_norm_f32<1024, false>`。

<!-- src: ggml/src/ggml-cuda/norm.cu -->
```c
template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
//>> ---- ggml/src/ggml-cuda/norm.cu:304-319 ----
static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
```

## 四、softmax.cu：8 个编译期列数与通用兜底

幕 4 引用了选择机制本身。这里补上**调用点** —— 它把候选列数写成模板实参列表：`launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>`；共享内存不够时才走另一条路（跨 SM 协作启动）。

<!-- src: ggml/src/ggml-cuda/softmax.cu -->
```c
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;


    if (nbytes_shared <= smpbo) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(x, mask, sinks, dst, params, stream, block_dims, block_nums, nbytes_shared);
    } else {
        // Parallelize across SMs for top-p/dist-sampling
        // The heuristic for parallelizing rows across SMs vs parallelizing single row & looping over all rows was done on the basis of a B6000 GPU and
        // Can be adapted further for lower-SM-count GPUs, though keeping data in registers should be implemented first as that is the optimal solution.
        if (ggml_cuda_info().devices[id].supports_cooperative_launch &&
            ncols_x / (params.ne01 * params.ne02 * params.ne03) > 8192 && mask == nullptr && sinks == nullptr &&
//>> ---- ggml/src/ggml-cuda/softmax.cu:278-296 ----
template<int... Ns, typename T>
static void launch_soft_max_kernels(const float * x, const T * mask, const float * sinks, float * dst,
                             const soft_max_params & p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, ncols, block, T>), smpbo);
            soft_max_f32<true, ncols, block><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, mask, sinks, dst, p);
            return true;
        }
        return false;
    };
```

## 五、rope.cu：四个 kernel，两种成对方式

幕 5 引用的是 `mode` 的解析与四路分支的入口。这里补上分支本体：`is_neox` 一路的三种 dtype 组合、`is_mrope` 的 `rope_multi_cuda`、`is_vision` 的 `rope_vision_cuda`，以及默认一路的 `rope_norm_cuda` —— 可以看到每个 kernel 都有 `<forward, T, T>` 形式的模板参数：**方向**也是编译期的。

<!-- src: ggml/src/ggml-cuda/rope.cu -->
```c
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
//>> ---- ggml/src/ggml-cuda/rope.cu:662-679 ----
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
```

## 六、ssm-scan.cu：模板参数与"第二条路"的判据

幕 6 引用了 `switch (n_tok)` 的 57 行实例表。这里补三处：① `SSM_SSD_MIN_TOKENS` 的定义（阈值 128 token）；② 模板头的三个参数（`splitD` / `N` / `L_template`）与 `L_template == 0` 的含义；③ 选择走 SSD（把 scan 变成 matmul）的**完整判据** —— 它同时看张量形状、token 数区间与设备计算能力。

<!-- src: ggml/src/ggml-cuda/ssm-scan.cu -->
```c
// Minimum number of tokens to use SSD (State Space Duality) matmul path instead of scan path.
// For n_tok <= this threshold, the scan kernel is used (lower overhead for short sequences).
#define SSM_SSD_MIN_TOKENS 128

// prepare_dt kernel dimensions: one block per (head, seq), each block handles DT_MAX_ITEMS items.
#define SSM_SSD_DT_BLOCK     256
#define SSM_SSD_DT_MAX_ITEMS  32

// Maximum tokens the SSD path supports, derived from the prepare_dt kernel block capacity.
#define SSM_SSD_MAX_TOKENS (SSM_SSD_DT_BLOCK * SSM_SSD_DT_MAX_ITEMS)

// Chunk size for chunked SSD. Caps matmul cost at O(chunk^2) per chunk.
#define SSM_SSD_CHUNK_SIZE 256

// We would like to keep pragma unroll for cases where L_template is not 0,
// so we suppress the clang transformation warning.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <size_t splitD, size_t N, size_t L_template>
__global__ void __launch_bounds__(splitD, 1)
    ssm_scan_f32(const float * src0_ptr, const float * src1_ptr, const float * src2_ptr,
                 const float * src3_ptr, const float * src4_ptr, const float * src5_ptr,
                 const int32_t * src6_ptr, float * dst_ptr,
                 const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
                 const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
                 const int64_t s_off, const int64_t d_inner, const int64_t L_param)
{
    const float   * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float   * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float   * GGML_CUDA_RESTRICT src2 = src2_ptr;
    const float   * GGML_CUDA_RESTRICT src3 = src3_ptr;
//>> ---- ggml/src/ggml-cuda/ssm-scan.cu:824-843 ----
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
    // Requires NVIDIA Turing+ otherwise fallback to scan.
    const bool is_mamba2 = (src3->nb[1] == sizeof(float));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_ssd = is_mamba2 && n_t > SSM_SSD_MIN_TOKENS
                      && K == 1
                      && n_t <= SSM_SSD_MAX_TOKENS
                      && GGML_CUDA_CC_IS_NVIDIA(cc)
                      && cc >= GGML_CUDA_CC_TURING
                      && nr % 8 == 0;  // cuBLAS requires 8-element (16-byte) alignment

    if (use_ssd) {
        // ssm_ssd_init_state_kernel uses flat linear indexing within each sequence,
        // so src0 must be fully contiguous across all inner dimensions.
        // The scan path handles non-contiguous nb[2] via src0_nb2 but does not handle nb[1].
        GGML_ASSERT(src0->nb[1] == nc         * sizeof(float));
        GGML_ASSERT(src0->nb[2] == nc * nr    * sizeof(float));

        ssm_scan_ssd_f32_cuda(ctx,
```

## 七、top-k.cu：同一个算子，三条实现路径

幕 7 引用了 radix 这条路的多个 kernel。但同一个 `ggml_cuda_op_top_k` 还挂着两条备选路径，由**编译期宏**决定走哪条：能拿到 CUB 时用 `top_k_cub`；否则退化成 "argsort + 拷贝"，其中 argsort 又按共享内存是否放得下，在按位排序与 CUB 之间选。这段也是"多 kernel 选一"的好例子。

<!-- src: ggml/src/ggml-cuda/top-k.cu -->
```c
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
```

## 八、quantize.cu：不被 switch 调的"助手"

`quantize.cu` 里没有 `ggml_cuda_op_quantize`：它提供的是 `quantize_row_q8_1_cuda` 与 `quantize_mmq_q8_1_cuda`，由 **mmq / mmvq** 调用（L6-02）。注意 `quantize_mmq_q8_1_cuda` 内部的 switch —— 它按 `mmq_get_q8_1_ds_layout(type_src0)` 选模板实例：**量化块的内部布局必须与 mmq 的读法一致**。

<!-- src: ggml/src/ggml-cuda/quantize.cu -->
```c
    ggml_cuda_kernel_launch(quantize_q8_1, launch_params, x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        default:
            GGML_ABORT("fatal error");
```

## 八（续）、调用方：mmq.cu 里的那一行

`quantize_mmq_q8_1_cuda` 的调用点在 `mmq.cu` 的 mul_mat_q 路径里：激活 `src1` 先被量化成 Q8_1，再交给 mmq 的 kernel；NVFP4 走的是另一个量化入口（`quantize_mmq_fp4_cuda`）。这就是"助手"的含义 —— 它服务的是 L6-02 的主角。

<!-- src: ggml/src/ggml-cuda/mmq.cu -->
```c
        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
```

## 八（续）、另一个调用方：mmvq.cu

mmvq 走的是**另一个**助手入口 `quantize_row_q8_1_cuda`（L6-02 的 mmvq 路径），同样是"先量化激活、再做点积"。两个调用点合起来说明：`quantize.cu` 不是算子，是矩阵乘路径的前置步骤。

<!-- src: ggml/src/ggml-cuda/mmvq.cu -->
```c
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
```

## 九、★ moe-weighted-reduction.cu：图级融合，不是算子

幕 8 引用了这个 kernel 与它的 launcher。它由 `ggml_cuda_try_fuse()` 在遍历图时匹配触发，匹配规则在 `ggml-cuda.cu` 里。判据有两层：入口节点的 op / 类型 / 连续性，以及"哪个输入是 experts、哪个是 broadcast 的 weights"；融合前还要过一次内存区间检查。

为什么说"它不是算子"？因为这份文件里没有 `GGML_OP` 常量可挂：

```text
grep -c 'GGML_OP_MOE_WEIGHTED' ggml/src/ggml-cuda/ggml-cuda.cu
  -> 0
```

触发它的入口在 `ggml_cuda_try_fuse()` 里，以 `GGML_OP_MUL` 为起点。

<!-- src: ggml/src/ggml-cuda/ggml-cuda.cu -->
```c
// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
//>> ---- ggml/src/ggml-cuda/ggml-cuda.cu:3441-3449 ----
    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
```

## 十、★ template-instances/：显式实例化的两半

这一节把"为什么必须存在"拆成可核对的两半：**头文件里的 extern 声明**，与**实例文件里的定义**。两者用的是同一个宏、同一份模板，区别只在 `extern` 这个词。先看 mmq 族：`DECL_MMQ_CASE(type)` 展开成一条函数模板的显式实例化；带 `extern` 的 22 行在头文件里，不带 `extern` 的 22 行在 22 个实例文件里。

<!-- src: ggml/src/ggml-cuda/mmq.cuh -->
```c
#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q2_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
```

## 十（续）、mmf：声明侧两种宏只差一个 extern

`DECL_MMF_CASE_EXTERN(n)` 与 `DECL_MMF_CASE(n)` 展开出的是同一批 6 个实例（float / half2 / bfloat162 x 两种 ROWS_PER_BLOCK），区别只在前者多一个 `extern`。后者被 16 个实例文件调用，前者被头文件调用 16 次。

<!-- src: ggml/src/ggml-cuda/mmf.cuh -->
```c
#define DECL_MMF_CASE_HELPER(T, nrows_dst, ncols_dst) \
    template void mul_mat_f_cuda<T, nrows_dst, ncols_dst>( \
        const T * x, const float * y, const int32_t * ids, float * dst, \
        const int64_t ncols_x, const int64_t nrows_x, int64_t ncols_dst_total, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst, \
        const int64_t stride_col_id, const int64_t stride_row_id, \
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst, \
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,\
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst, \
        cudaStream_t stream, const mmf_ids_data * ids_data);

#if !defined(GGML_USE_MUSA)
#define DECL_MMF_CASE_EXTERN(ncols_dst) \
    extern DECL_MMF_CASE_HELPER(float, MMF_ROWS_PER_BLOCK, ncols_dst) \
    extern DECL_MMF_CASE_HELPER(half2, MMF_ROWS_PER_BLOCK, ncols_dst) \
    extern DECL_MMF_CASE_HELPER(nv_bfloat162, MMF_ROWS_PER_BLOCK, ncols_dst) \
    extern DECL_MMF_CASE_HELPER(float, MMF_ROWS_PER_BLOCK_CDNA, ncols_dst) \
    extern DECL_MMF_CASE_HELPER(half2, MMF_ROWS_PER_BLOCK_CDNA, ncols_dst) \
    extern DECL_MMF_CASE_HELPER(nv_bfloat162, MMF_ROWS_PER_BLOCK_CDNA, ncols_dst)

#define DECL_MMF_CASE(ncols_dst) \
    DECL_MMF_CASE_HELPER(float, MMF_ROWS_PER_BLOCK, ncols_dst) \
    DECL_MMF_CASE_HELPER(half2, MMF_ROWS_PER_BLOCK, ncols_dst) \
```

## 十（续）、fattn-vec：一个宏展开出 7 种 type_V

`EXTERN_DECL_FATTN_VEC_CASES(D, type_K)` 一次展开出 7 条声明（F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / BF16）。21 次调用 = 147 条声明，落到 49 个实例文件里（每个文件含 D = 64 / 128 / 256 三行）。

<!-- src: ggml/src/ggml-cuda/fattn-vec.cuh -->
```c
#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

```

## 十（续）、fattn-tile：12 组 (DKQ, DV)

12 条 `extern` 对应 12 个实例文件，一个文件一组头维度。

<!-- src: ggml/src/ggml-cuda/fattn-tile.cuh -->
```c
#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    template void ggml_cuda_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_TILE_CASE( 40,  40);
extern DECL_FATTN_TILE_CASE( 64,  64);
extern DECL_FATTN_TILE_CASE( 72,  72);
extern DECL_FATTN_TILE_CASE( 80,  80);
extern DECL_FATTN_TILE_CASE( 96,  96);
extern DECL_FATTN_TILE_CASE(112, 112);
```

## 十（续）、fattn-mma-f16：一个二元宏展开 5 个 ncols2

`DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)` 把一个 head size 展开成 5 个 (ncols1, ncols2) 组合；每个实例文件反过来，固定一组 (ncols1, ncols2) 而列出多个 head size。

<!-- src: ggml/src/ggml-cuda/fattn-mma-f16.cuh -->
```c
#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \
```

## 十（续）、实例侧：一个文件一行（mmq）

五个族的实例文件各看一个。第一行都是同一句"脚本生成、别手改"，中间的 include 指向本族的头文件，最后一行是**不带 extern** 的宏调用 —— **这一行就是定义**。先看 mmq：

<!-- src: ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu -->
```c
// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_Q4_0);
```

## 十（续）、实例侧：mmf

`mmf-instance-ncols_1.cu` 的全部内容就是"实例化 ncols_dst = 1"这一种组合。

<!-- src: ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_1.cu -->
```c
// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../mmf.cuh"

DECL_MMF_CASE(1);
```

## 十（续）、实例侧：fattn-vec

一个文件覆盖 D = 64 / 128 / 256 三个头维度 —— 所以"49 个文件"对应的是 7 x 7 种 (type_K, type_V) 组合，而不是 147 个文件。

<!-- src: ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu -->
```c
// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);
```

## 十（续）、实例侧：fattn-tile

`fattn-tile-instance-dkq128-dv128.cu` 只有一行 `DECL_FATTN_TILE_CASE(128, 128)`。

<!-- src: ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq128-dv128.cu -->
```c
// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../fattn-tile.cuh"

DECL_FATTN_TILE_CASE(128, 128);
```

## 十（续）、实例侧：fattn-mma-f16

`fattn-mma-f16-instance-ncols1_16-ncols2_1.cu` 固定 (ncols1, ncols2) = (16, 1)，列出 6 个 head size。族内文件的行数并不相同（实测 2 到 8 行），所以"文件数"与"实例化语句数"是两个不同的数字。

<!-- src: ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu -->
```c
// This file has been autogenerated by generate_cu_files.py, do not edit manually.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(64, 64, 16, 1);
DECL_FATTN_MMA_F16_CASE(80, 80, 16, 1);
DECL_FATTN_MMA_F16_CASE(96, 96, 16, 1);
DECL_FATTN_MMA_F16_CASE(112, 112, 16, 1);
DECL_FATTN_MMA_F16_CASE(128, 128, 16, 1);
DECL_FATTN_MMA_F16_CASE(256, 256, 16, 1);
```

## 十一、按前缀族的分类与计数（全部实测）

`template-instances/` 的 120 个文件按文件名前缀分成五族。文件数是 `ls` 数出来的，"每个文件里几行实例化语句"是数 `DECL_` 开头的行得到的：

| 族（前缀） | 文件数 | 每文件实例化语句数 | 模板参数空间 |
|---|---|---|---|
| `mmq-instance-*` | 22 | 1 | 22 个量化类型（与 `mmq.cuh` 的 22 条 extern 一一对应） |
| `mmf-instance-ncols_*` | 16 | 1 | n = 1..16（与 16 条 `DECL_MMF_CASE_EXTERN` 对应） |
| `fattn-vec-instance-*` | 49 | 3 | 7 个 type_K x 7 个 type_V = 49 |
| `fattn-tile-instance-*` | 12 | 1 | 12 组 (DKQ, DV)（与 12 条 extern 对应） |
| `fattn-mma-f16-instance-*` | 21 | 2-8（合计 126 行） | 21 组 (ncols1, ncols2) |
| `fattn-*` 三族小计 | **82** | 285 行 | FlashAttention 的三条实现路线（L6-03） |
| **合计** | **120** | 323 行 | — |

```text
ls ggml/src/ggml-cuda/template-instances/*.cu | wc -l                       -> 120
ls ggml/src/ggml-cuda/template-instances/mmq-instance-*.cu | wc -l         -> 22
ls ggml/src/ggml-cuda/template-instances/mmf-instance-*.cu | wc -l         -> 16
ls ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*.cu | wc -l   -> 49
ls ggml/src/ggml-cuda/template-instances/fattn-tile-instance-*.cu | wc -l  -> 12
ls ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-*.cu | wc -l -> 21
grep -c '^extern DECL_MMQ_CASE' ggml/src/ggml-cuda/mmq.cuh               -> 22
grep -c '^DECL_MMF_CASE_EXTERN' ggml/src/ggml-cuda/mmf.cuh               -> 16
grep -c '^extern DECL_FATTN_TILE_CASE' ggml/src/ggml-cuda/fattn-tile.cuh -> 12
```

一句话：**模板参数空间有多大，文件就有多少。**

## 十二、结论：三类角色，一个目录

把这一课压成三句话：

1. **算子实现是"入口函数 + kernel 模板"**：入口函数从 `dst` 取形状与 `op_params`，kernel 用模板参数表达编译期选择（可展开的循环、要不要融合、块大小）。
2. **不是所有 .cu 都是 op**：`quantize.cu` 是 mmq / mmvq 的助手；`moe-weighted-reduction.cu` 是图级融合，由 `ggml_cuda_try_fuse()` 触发。
3. **`template-instances/` 是显式实例化的落地点**：头文件用 `extern` 声明把它们"挂起来"，每个实例文件用一个不带 `extern` 的宏提供定义，一个组合一个 TU。删掉它们，链接期就会 `undefined reference`。

---

## 说明

- 本课覆盖声明 = `tools/plan_matrix.py` 里 L6-04 的**全部** 280 个文件（160 个手写 + 120 个 `template-instances/*.cu`）。spec 在 build 时会拿这份清单与计划对账，不一致直接报错。
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` 是生成器本身，后缀 `.py` 不在覆盖域内，**不计入**本课的 280 个声明。
- `ggml/include/ggml-cuda.h`（公共后端 API）属于 L6-01 的清单，不在本课目录内，因此也不在本课声明里。
- 本课不涉及任何命令行参数；参数门禁的真值集来自真实二进制的帮助输出。
