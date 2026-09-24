# L6-04 · CUDA 其余算子与模板实例化 — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-03`（★ CUDA FlashAttention）；本课默认你已经看过 `L6-01`（后端骨架与分派）与 `L6-02`（量化矩阵乘）。

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml/src/ggml-cuda/` 下两类文件的分工，以及本课清单的规模（280 = 160 手写 + 120 生成）；
2. 解释 `template-instances/` 目录为什么必须存在 —— 头文件的 `extern` 声明与实例文件的定义各是哪一半，删掉会怎样（对应验收点）；
3. 说出 `norm.cu` / `softmax.cu` / `rope.cu` / `ssm-scan.cu` / `top-k.cu` 各自"运行期形状 -> kernel" 的判据；
4. 举出至少两个"不是 op 实现"的 `.cu`，并说明它们分别被谁调用；
5. 指出 `ssm-scan.cu` 里"手写实例表"与 `template-instances/` 的关系。

## 覆盖的源文件（280 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-cuda/acc.cu` | 62 |
| `ggml/src/ggml-cuda/acc.cuh` | 6 |
| `ggml/src/ggml-cuda/add-id.cu` | 59 |
| `ggml/src/ggml-cuda/add-id.cuh` | 4 |
| `ggml/src/ggml-cuda/allreduce.cu` | 978 |
| `ggml/src/ggml-cuda/allreduce.cuh` | 30 |
| `ggml/src/ggml-cuda/arange.cu` | 35 |
| `ggml/src/ggml-cuda/arange.cuh` | 6 |
| `ggml/src/ggml-cuda/argmax.cu` | 92 |
| `ggml/src/ggml-cuda/argmax.cuh` | 4 |
| `ggml/src/ggml-cuda/argsort.cu` | 296 |
| `ggml/src/ggml-cuda/argsort.cuh` | 21 |
| `ggml/src/ggml-cuda/binbcast.cu` | 575 |
| `ggml/src/ggml-cuda/binbcast.cuh` | 13 |
| `ggml/src/ggml-cuda/clamp.cu` | 46 |
| `ggml/src/ggml-cuda/clamp.cuh` | 6 |
| `ggml/src/ggml-cuda/col2im-1d.cu` | 82 |
| `ggml/src/ggml-cuda/col2im-1d.cuh` | 4 |
| `ggml/src/ggml-cuda/common.cuh` | 1718 |
| `ggml/src/ggml-cuda/concat.cu` | 243 |
| `ggml/src/ggml-cuda/concat.cuh` | 6 |
| `ggml/src/ggml-cuda/conv-transpose-1d.cu` | 89 |
| `ggml/src/ggml-cuda/conv-transpose-1d.cuh` | 6 |
| `ggml/src/ggml-cuda/conv2d-dw.cu` | 162 |
| `ggml/src/ggml-cuda/conv2d-dw.cuh` | 6 |
| `ggml/src/ggml-cuda/conv2d-transpose.cu` | 116 |
| `ggml/src/ggml-cuda/conv2d-transpose.cuh` | 6 |
| `ggml/src/ggml-cuda/conv2d.cu` | 451 |
| `ggml/src/ggml-cuda/conv2d.cuh` | 6 |
| `ggml/src/ggml-cuda/convert.cu` | 738 |
| `ggml/src/ggml-cuda/convert.cuh` | 67 |
| `ggml/src/ggml-cuda/count-equal.cu` | 65 |
| `ggml/src/ggml-cuda/count-equal.cuh` | 6 |
| `ggml/src/ggml-cuda/cp-async.cuh` | 58 |
| `ggml/src/ggml-cuda/cpy-utils.cuh` | 218 |
| `ggml/src/ggml-cuda/cpy.cu` | 626 |
| `ggml/src/ggml-cuda/cpy.cuh` | 8 |
| `ggml/src/ggml-cuda/cross-entropy-loss.cu` | 178 |
| `ggml/src/ggml-cuda/cross-entropy-loss.cuh` | 8 |
| `ggml/src/ggml-cuda/cumsum.cu` | 308 |
| `ggml/src/ggml-cuda/cumsum.cuh` | 6 |
| `ggml/src/ggml-cuda/dequantize.cuh` | 453 |
| `ggml/src/ggml-cuda/diag.cu` | 78 |
| `ggml/src/ggml-cuda/diag.cuh` | 6 |
| `ggml/src/ggml-cuda/diagmask.cu` | 41 |
| `ggml/src/ggml-cuda/diagmask.cuh` | 6 |
| `ggml/src/ggml-cuda/dsv4-hc.cu` | 317 |
| `ggml/src/ggml-cuda/dsv4-hc.cuh` | 7 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | 1303 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 2188 |
| `ggml/src/ggml-cuda/fattn-tile.cu` | 61 |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | 1356 |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | 610 |
| `ggml/src/ggml-cuda/fattn.cu` | 775 |
| `ggml/src/ggml-cuda/fattn.cuh` | 8 |
| `ggml/src/ggml-cuda/fill.cu` | 38 |
| `ggml/src/ggml-cuda/fill.cuh` | 4 |
| `ggml/src/ggml-cuda/fwht.cu` | 102 |
| `ggml/src/ggml-cuda/fwht.cuh` | 5 |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | 328 |
| `ggml/src/ggml-cuda/gated_delta_net.cuh` | 15 |
| `ggml/src/ggml-cuda/getrows.cu` | 491 |
| `ggml/src/ggml-cuda/getrows.cuh` | 16 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5857 |
| `ggml/src/ggml-cuda/gla.cu` | 94 |
| `ggml/src/ggml-cuda/gla.cuh` | 4 |
| `ggml/src/ggml-cuda/im2col.cu` | 271 |
| `ggml/src/ggml-cuda/im2col.cuh` | 7 |
| `ggml/src/ggml-cuda/lightning-indexer.cu` | 589 |
| `ggml/src/ggml-cuda/lightning-indexer.cuh` | 5 |
| `ggml/src/ggml-cuda/mean.cu` | 85 |
| `ggml/src/ggml-cuda/mean.cuh` | 4 |
| `ggml/src/ggml-cuda/mma.cuh` | 1517 |
| `ggml/src/ggml-cuda/mmf.cu` | 192 |
| `ggml/src/ggml-cuda/mmf.cuh` | 928 |
| `ggml/src/ggml-cuda/mmid.cu` | 182 |
| `ggml/src/ggml-cuda/mmid.cuh` | 6 |
| `ggml/src/ggml-cuda/mmq-config-ampere.cuh` | 384 |
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | 38 |
| `ggml/src/ggml-cuda/mmq-config-cdna.cuh` | 186 |
| `ggml/src/ggml-cuda/mmq-config-gcn.cuh` | 282 |
| `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh` | 274 |
| `ggml/src/ggml-cuda/mmq-config-pascal-older.cuh` | 274 |
| `ggml/src/ggml-cuda/mmq-config-rdna2.cuh` | 274 |
| `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh` | 291 |
| `ggml/src/ggml-cuda/mmq-config-rdna3.cuh` | 275 |
| `ggml/src/ggml-cuda/mmq-config-rdna4.cuh` | 291 |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | 1769 |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | 1241 |
| `ggml/src/ggml-cuda/mmq.cu` | 394 |
| `ggml/src/ggml-cuda/mmq.cuh` | 1606 |
| `ggml/src/ggml-cuda/mmvf.cu` | 876 |
| `ggml/src/ggml-cuda/mmvf.cuh` | 15 |
| `ggml/src/ggml-cuda/mmvq.cu` | 1568 |
| `ggml/src/ggml-cuda/mmvq.cuh` | 19 |
| `ggml/src/ggml-cuda/moe-weighted-reduction.cu` | 66 |
| `ggml/src/ggml-cuda/moe-weighted-reduction.cuh` | 8 |
| `ggml/src/ggml-cuda/norm.cu` | 699 |
| `ggml/src/ggml-cuda/norm.cuh` | 19 |
| `ggml/src/ggml-cuda/opt-step-adamw.cu` | 79 |
| `ggml/src/ggml-cuda/opt-step-adamw.cuh` | 6 |
| `ggml/src/ggml-cuda/opt-step-sgd.cu` | 50 |
| `ggml/src/ggml-cuda/opt-step-sgd.cuh` | 6 |
| `ggml/src/ggml-cuda/out-prod.cu` | 126 |
| `ggml/src/ggml-cuda/out-prod.cuh` | 4 |
| `ggml/src/ggml-cuda/pad.cu` | 107 |
| `ggml/src/ggml-cuda/pad.cuh` | 6 |
| `ggml/src/ggml-cuda/pad_reflect_1d.cu` | 92 |
| `ggml/src/ggml-cuda/pad_reflect_1d.cuh` | 6 |
| `ggml/src/ggml-cuda/pool1d.cu` | 86 |
| `ggml/src/ggml-cuda/pool1d.cuh` | 6 |
| `ggml/src/ggml-cuda/pool2d.cu` | 95 |
| `ggml/src/ggml-cuda/pool2d.cuh` | 6 |
| `ggml/src/ggml-cuda/quantize.cu` | 698 |
| `ggml/src/ggml-cuda/quantize.cuh` | 71 |
| `ggml/src/ggml-cuda/reduce_rows.cuh` | 73 |
| `ggml/src/ggml-cuda/roll.cu` | 68 |
| `ggml/src/ggml-cuda/roll.cuh` | 6 |
| `ggml/src/ggml-cuda/rope.cu` | 942 |
| `ggml/src/ggml-cuda/rope.cuh` | 12 |
| `ggml/src/ggml-cuda/scale.cu` | 38 |
| `ggml/src/ggml-cuda/scale.cuh` | 6 |
| `ggml/src/ggml-cuda/set-rows.cu` | 399 |
| `ggml/src/ggml-cuda/set-rows.cuh` | 8 |
| `ggml/src/ggml-cuda/set.cu` | 40 |
| `ggml/src/ggml-cuda/set.cuh` | 8 |
| `ggml/src/ggml-cuda/snake.cu` | 73 |
| `ggml/src/ggml-cuda/snake.cuh` | 9 |
| `ggml/src/ggml-cuda/softcap.cu` | 38 |
| `ggml/src/ggml-cuda/softcap.cuh` | 6 |
| `ggml/src/ggml-cuda/softmax.cu` | 481 |
| `ggml/src/ggml-cuda/softmax.cuh` | 8 |
| `ggml/src/ggml-cuda/solve_tri.cu` | 274 |
| `ggml/src/ggml-cuda/solve_tri.cuh` | 4 |
| `ggml/src/ggml-cuda/ssm-conv.cu` | 207 |
| `ggml/src/ggml-cuda/ssm-conv.cuh` | 4 |
| `ggml/src/ggml-cuda/ssm-scan.cu` | 860 |
| `ggml/src/ggml-cuda/ssm-scan.cuh` | 4 |
| `ggml/src/ggml-cuda/sum.cu` | 42 |
| `ggml/src/ggml-cuda/sum.cuh` | 6 |
| `ggml/src/ggml-cuda/sumrows.cu` | 54 |
| `ggml/src/ggml-cuda/sumrows.cuh` | 5 |
| `ggml/src/ggml-cuda/top-k.cu` | 276 |
| `ggml/src/ggml-cuda/top-k.cuh` | 4 |
| `ggml/src/ggml-cuda/topk-moe.cu` | 443 |
| `ggml/src/ggml-cuda/topk-moe.cuh` | 32 |
| `ggml/src/ggml-cuda/tri.cu` | 137 |
| `ggml/src/ggml-cuda/tri.cuh` | 6 |
| `ggml/src/ggml-cuda/tsembd.cu` | 48 |
| `ggml/src/ggml-cuda/tsembd.cuh` | 6 |
| `ggml/src/ggml-cuda/unary.cu` | 722 |
| `ggml/src/ggml-cuda/unary.cuh` | 124 |
| `ggml/src/ggml-cuda/upscale.cu` | 294 |
| `ggml/src/ggml-cuda/upscale.cuh` | 6 |
| `ggml/src/ggml-cuda/vecdotq.cuh` | 1381 |
| `ggml/src/ggml-cuda/vendors/cuda.h` | 29 |
| `ggml/src/ggml-cuda/vendors/hip.h` | 314 |
| `ggml/src/ggml-cuda/vendors/musa.h` | 152 |
| `ggml/src/ggml-cuda/wkv.cu` | 254 |
| `ggml/src/ggml-cuda/wkv.cuh` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu` | 7 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_32.cu` | 7 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu` | 11 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu` | 12 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_16.cu` | 7 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_32.cu` | 7 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu` | 11 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu` | 12 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_16.cu` | 7 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu` | 12 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu` | 11 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu` | 11 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu` | 12 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu` | 13 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq112-dv112.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq128-dv128.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq192-dv128.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq256-dv256.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq320-dv256.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq40-dv40.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq512-dv512.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq576-dv512.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq64-dv64.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq72-dv72.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq80-dv80.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq96-dv96.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-f16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q4_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q5_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q5_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-q8_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-bf16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q5_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q5_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q8_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-bf16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-f16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q5_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q5_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q8_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-bf16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-f16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q4_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q5_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q5_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q8_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-bf16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-f16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q4_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q8_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-bf16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-f16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q4_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q5_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q5_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q8_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-bf16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-f16.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q4_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q5_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q5_1.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu` | 8 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_1.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_10.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_11.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_12.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_13.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_14.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_15.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_16.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_2.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_3.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_4.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_5.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_6.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_7.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_8.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_9.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq1_s.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q1_0.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q2_k.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q3_k.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_1.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu` | 6 |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu` | 6 |

> **说明**：本课覆盖声明 = `tools/plan_matrix.py` 里 L6-04 的**全部** 280 个文件（160 个手写 + 120 个 `template-instances/*.cu`）。spec 在 build 时会拿这份清单与计划对账，不一致直接报错。
> **说明**：`ggml/src/ggml-cuda/template-instances/generate_cu_files.py` 是生成器本身，后缀 `.py` 不在覆盖域内，**不计入**本课的 280 个声明。
> **说明**：`ggml/include/ggml-cuda.h`（公共后端 API）属于 L6-01 的清单，不在本课目录内，因此也不在本课声明里。
> **说明**：本课不涉及任何命令行参数；参数门禁的真值集来自真实二进制的帮助输出。

## 场景（10 幕）

1. **本课的地图：280 个文件，两类来源** — ggml/src/ggml-cuda/ 下的文件分成两类：人写的算子实现，和脚本生成的模板实例。
2. **和 L6-01 同一个 switch：每个 op 一个入口** — 非矩阵乘的算子没有特殊通道 —— 它们和 MUL_MAT 挤在同一张 dst->op 分派表里。
3. **norm.cu：一个 block 管一行，规约在 block 内闭合** — RMSNorm 在 GPU 上是一个"行内规约"问题：一行的平方和算完，才能写回这一行。
4. **softmax.cu：把 ncols 折进编译期，但不建目录** — 同一份 kernel 源码，对若干常见列数做编译期特化；选择靠一条折叠表达式。
5. **rope.cu：一个 mode 整数，四种成对方式** — RoPE 的变体不是四份图，而是 op_params 里 mode 的几个位 —— 但落地时确实是四个 kernel。
6. **ssm-scan.cu：57 行 switch —— 手写的实例表** — 同一个 kernel 模板，按 token 数 1..8 各实例化一份（循环可完全展开），其它走通用版。
7. **top-k.cu：一条 多段式 的 kernel 流水线** — 排序类算子在 GPU 上常常不是"一个 kernel"，而是几个小 kernel 依次跑，中间结果放显存。
8. **★ 不是每个 .cu 都是一个 op：moe 归约是个融合** — moe-weighted-reduction.cu 里没有 GGML_OP —— 它由图级模式匹配触发，跳过一整段子图。
9. **★ template-instances/：为什么这个目录必须存在** — 一个实例文件只有 5 行：一句"脚本生成"，一个 include，一行不带 extern 的宏。
10. **把这一课压成一张表** — 六个算子、三类角色、一个目录。

## 核心结论

### 本课清单：280 = 160 + 120

`ggml/src/ggml-cuda/` 下属于 L6-04 的 280 个文件：

| 来源 | 数量 | 是什么 |
|---|---|---|
| 手写 `.cu` | 68 | 算子实现、助手、融合 |
| 手写 `.cuh` | 89 | 声明、kernel 模板、配置表 |
| `vendors/*.h` | 3 | 厂商适配层 |
| `template-instances/*.cu` | 120 | 生成：显式实例化定义 |

L6-01 / L6-02 / L6-03 覆盖的是这个目录的**子集**；本课是全目录声明。

### ★ template-instances/ 为什么必须存在

两半合起来才成立：

```text
mmq.cuh（声明侧）：  extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);   <- 显式实例化声明
template-instances/mmq-instance-q1_0.cu（定义侧）：
                     DECL_MMQ_CASE(GGML_TYPE_Q1_0);           <- 显式实例化定义
```

`extern` 抑制本 TU 的隐式实例化，符号必须由别处提供；实例文件就是"别处"。**模板参数空间有多大，文件就有多少**：mmq 22 / mmf 16 / fattn-vec 49 / fattn-tile 12 / fattn-mma-f16 21 = 120。

### ★ 三类角色：op / 助手 / 融合

| 角色 | 谁调用 | 例子 |
|---|---|---|
| op 实现 | `ggml_cuda_compute_forward` 的 switch | `norm.cu` `rope.cu` `softmax.cu` `ssm-scan.cu` `top-k.cu` |
| 助手 | 别的 kernel | `quantize.cu`（mmq / mmvq 的 Q8_1 输入） |
| 图级融合 | `ggml_cuda_try_fuse()` 的模式匹配 | `moe-weighted-reduction.cu` |

所以"读完分派表"并不等于"读完后端"。

## 验收点

- [x] 保真门禁：31 处引用 —— 31 个引用块 / 39 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 280 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 3 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
