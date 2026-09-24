# L6-05 · SYCL 后端（Intel GPU） — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-04`（CUDA 其余算子）

## 学习目标

看完这一课，你应该能：

1. 给一个算子（如 `GGML_OP_MUL_MAT`），同时说出 SYCL 后端与 CUDA 后端在它上面的**文件名、函数名与行号**（对应本课验收点）；
2. 说出 `ggml_backend_sycl_context` 与 `ggml_backend_cuda_context` 的字段对应关系，以及 `queue_ptr` 与 `cudaStream_t` 的差异；
3. 解释为什么 SYCL 后端能用 173 个文件跟上 CUDA 侧的算子覆盖（结构同构 + 数据面局部替换）；
4. 说出 `template-instances/` 的组织方式，并指出 SYCL 侧与 CUDA 侧在`EXTERN_DECL_FATTN_VEC_CASES` 类型清单上的**真实差异**；
5. 说明 `ggml_backend_sycl_graph_compute` 在什么条件下走"录制重放"、否则怎么退回。

## 覆盖的源文件（179 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-sycl.h` | 60 |
| `ggml/src/ggml-sycl/add-id.cpp` | 82 |
| `ggml/src/ggml-sycl/add-id.hpp` | 9 |
| `ggml/src/ggml-sycl/backend.hpp` | 54 |
| `ggml/src/ggml-sycl/base.hpp` | 44 |
| `ggml/src/ggml-sycl/binbcast.cpp` | 651 |
| `ggml/src/ggml-sycl/binbcast.hpp` | 70 |
| `ggml/src/ggml-sycl/col2im-1d.cpp` | 103 |
| `ggml/src/ggml-sycl/col2im-1d.hpp` | 9 |
| `ggml/src/ggml-sycl/common.cpp` | 163 |
| `ggml/src/ggml-sycl/common.hpp` | 1015 |
| `ggml/src/ggml-sycl/concat.cpp` | 510 |
| `ggml/src/ggml-sycl/concat.hpp` | 21 |
| `ggml/src/ggml-sycl/conv.cpp` | 102 |
| `ggml/src/ggml-sycl/conv.hpp` | 21 |
| `ggml/src/ggml-sycl/conv2d-dw.cpp` | 171 |
| `ggml/src/ggml-sycl/conv2d-dw.hpp` | 11 |
| `ggml/src/ggml-sycl/conv2d-transpose.cpp` | 126 |
| `ggml/src/ggml-sycl/conv2d-transpose.hpp` | 11 |
| `ggml/src/ggml-sycl/conv2d.cpp` | 151 |
| `ggml/src/ggml-sycl/conv2d.hpp` | 11 |
| `ggml/src/ggml-sycl/conv3d.cpp` | 225 |
| `ggml/src/ggml-sycl/conv3d.hpp` | 9 |
| `ggml/src/ggml-sycl/convert.cpp` | 878 |
| `ggml/src/ggml-sycl/convert.hpp` | 65 |
| `ggml/src/ggml-sycl/count-equal.cpp` | 80 |
| `ggml/src/ggml-sycl/count-equal.hpp` | 10 |
| `ggml/src/ggml-sycl/cpy.cpp` | 1434 |
| `ggml/src/ggml-sycl/cpy.hpp` | 538 |
| `ggml/src/ggml-sycl/cross_entropy_loss.cpp` | 256 |
| `ggml/src/ggml-sycl/cross_entropy_loss.hpp` | 8 |
| `ggml/src/ggml-sycl/cumsum.cpp` | 149 |
| `ggml/src/ggml-sycl/cumsum.hpp` | 6 |
| `ggml/src/ggml-sycl/dequantize.hpp` | 1682 |
| `ggml/src/ggml-sycl/diag.cpp` | 68 |
| `ggml/src/ggml-sycl/diag.hpp` | 6 |
| `ggml/src/ggml-sycl/dmmv.cpp` | 2228 |
| `ggml/src/ggml-sycl/dmmv.hpp` | 28 |
| `ggml/src/ggml-sycl/dpct/helper.hpp` | 3783 |
| `ggml/src/ggml-sycl/dsv4-hc.cpp` | 339 |
| `ggml/src/ggml-sycl/dsv4-hc.hpp` | 11 |
| `ggml/src/ggml-sycl/element_wise.cpp` | 1433 |
| `ggml/src/ggml-sycl/element_wise.hpp` | 136 |
| `ggml/src/ggml-sycl/esimd.hpp` | 590 |
| `ggml/src/ggml-sycl/fattn-buffers.cpp` | 61 |
| `ggml/src/ggml-sycl/fattn-buffers.hpp` | 64 |
| `ggml/src/ggml-sycl/fattn-common.hpp` | 1186 |
| `ggml/src/ggml-sycl/fattn-mkl.cpp` | 695 |
| `ggml/src/ggml-sycl/fattn-onednn.cpp` | 451 |
| `ggml/src/ggml-sycl/fattn-onednn.hpp` | 19 |
| `ggml/src/ggml-sycl/fattn-tile.cpp` | 60 |
| `ggml/src/ggml-sycl/fattn-tile.hpp` | 1251 |
| `ggml/src/ggml-sycl/fattn-vec.hpp` | 685 |
| `ggml/src/ggml-sycl/fattn.cpp` | 453 |
| `ggml/src/ggml-sycl/fattn.hpp` | 43 |
| `ggml/src/ggml-sycl/fill.cpp` | 56 |
| `ggml/src/ggml-sycl/fill.hpp` | 6 |
| `ggml/src/ggml-sycl/fusion.cpp` | 281 |
| `ggml/src/ggml-sycl/fusion.hpp` | 18 |
| `ggml/src/ggml-sycl/fwht.cpp` | 292 |
| `ggml/src/ggml-sycl/fwht.hpp` | 13 |
| `ggml/src/ggml-sycl/gated_delta_net.cpp` | 367 |
| `ggml/src/ggml-sycl/gated_delta_net.hpp` | 20 |
| `ggml/src/ggml-sycl/gemm.hpp` | 96 |
| `ggml/src/ggml-sycl/getrows.cpp` | 474 |
| `ggml/src/ggml-sycl/getrows.hpp` | 22 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 7293 |
| `ggml/src/ggml-sycl/gla.cpp` | 107 |
| `ggml/src/ggml-sycl/gla.hpp` | 9 |
| `ggml/src/ggml-sycl/im2col.cpp` | 401 |
| `ggml/src/ggml-sycl/im2col.hpp` | 24 |
| `ggml/src/ggml-sycl/lightning-indexer.cpp` | 198 |
| `ggml/src/ggml-sycl/lightning-indexer.hpp` | 9 |
| `ggml/src/ggml-sycl/mem.cpp` | 152 |
| `ggml/src/ggml-sycl/mem.hpp` | 17 |
| `ggml/src/ggml-sycl/memtrace.cpp` | 195 |
| `ggml/src/ggml-sycl/memtrace.hpp` | 29 |
| `ggml/src/ggml-sycl/mmq.cpp` | 3031 |
| `ggml/src/ggml-sycl/mmq.hpp` | 34 |
| `ggml/src/ggml-sycl/mmvq.cpp` | 3292 |
| `ggml/src/ggml-sycl/mmvq.hpp` | 97 |
| `ggml/src/ggml-sycl/norm.cpp` | 1138 |
| `ggml/src/ggml-sycl/norm.hpp` | 38 |
| `ggml/src/ggml-sycl/opt-step.cpp` | 132 |
| `ggml/src/ggml-sycl/opt-step.hpp` | 7 |
| `ggml/src/ggml-sycl/outprod.cpp` | 84 |
| `ggml/src/ggml-sycl/outprod.hpp` | 11 |
| `ggml/src/ggml-sycl/pad.cpp` | 98 |
| `ggml/src/ggml-sycl/pad.hpp` | 25 |
| `ggml/src/ggml-sycl/pad_reflect_1d.cpp` | 101 |
| `ggml/src/ggml-sycl/pad_reflect_1d.hpp` | 11 |
| `ggml/src/ggml-sycl/pool.cpp` | 186 |
| `ggml/src/ggml-sycl/pool.hpp` | 23 |
| `ggml/src/ggml-sycl/presets.hpp` | 80 |
| `ggml/src/ggml-sycl/quantize.hpp` | 134 |
| `ggml/src/ggml-sycl/quants.hpp` | 205 |
| `ggml/src/ggml-sycl/repeat_back.cpp` | 77 |
| `ggml/src/ggml-sycl/repeat_back.hpp` | 9 |
| `ggml/src/ggml-sycl/roll.cpp` | 123 |
| `ggml/src/ggml-sycl/roll.hpp` | 21 |
| `ggml/src/ggml-sycl/rope.cpp` | 652 |
| `ggml/src/ggml-sycl/rope.hpp` | 27 |
| `ggml/src/ggml-sycl/set.cpp` | 74 |
| `ggml/src/ggml-sycl/set.hpp` | 6 |
| `ggml/src/ggml-sycl/set_rows.cpp` | 578 |
| `ggml/src/ggml-sycl/set_rows.hpp` | 9 |
| `ggml/src/ggml-sycl/softmax.cpp` | 426 |
| `ggml/src/ggml-sycl/softmax.hpp` | 25 |
| `ggml/src/ggml-sycl/solve_tri.cpp` | 173 |
| `ggml/src/ggml-sycl/solve_tri.hpp` | 9 |
| `ggml/src/ggml-sycl/ssm_conv.cpp` | 362 |
| `ggml/src/ggml-sycl/ssm_conv.hpp` | 7 |
| `ggml/src/ggml-sycl/ssm_scan.cpp` | 171 |
| `ggml/src/ggml-sycl/ssm_scan.hpp` | 6 |
| `ggml/src/ggml-sycl/sycl_hw.cpp` | 68 |
| `ggml/src/ggml-sycl/sycl_hw.hpp` | 39 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq112-dv112.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq128-dv128.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq256-dv256.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq40-dv40.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq512-dv512.cpp` | 7 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq576-dv512.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq64-dv64.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq72-dv72.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq80-dv80.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq96-dv96.cpp` | 6 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-f16.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q4_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q4_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q5_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q5_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-f16-q8_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_0-f16.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_0-q4_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_0-q4_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_0-q5_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_0-q5_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_0-q8_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_1-f16.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_1-q4_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_1-q4_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_1-q5_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_1-q5_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q4_1-q8_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_0-f16.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_0-q4_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_0-q4_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_0-q5_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_0-q5_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_0-q8_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_1-f16.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_1-q4_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_1-q4_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_1-q5_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_1-q5_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q5_1-q8_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q8_0-f16.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q8_0-q4_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q8_0-q4_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q8_0-q5_0.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q8_0-q5_1.cpp` | 9 |
| `ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q8_0-q8_0.cpp` | 9 |
| `ggml/src/ggml-sycl/topk-moe.cpp` | 621 |
| `ggml/src/ggml-sycl/topk-moe.hpp` | 13 |
| `ggml/src/ggml-sycl/topk-radix.cpp` | 532 |
| `ggml/src/ggml-sycl/topk-radix.hpp` | 25 |
| `ggml/src/ggml-sycl/tsembd.cpp` | 74 |
| `ggml/src/ggml-sycl/tsembd.hpp` | 21 |
| `ggml/src/ggml-sycl/type.hpp` | 113 |
| `ggml/src/ggml-sycl/upscale.cpp` | 411 |
| `ggml/src/ggml-sycl/upscale.hpp` | 10 |
| `ggml/src/ggml-sycl/vecdotq.hpp` | 1748 |
| `ggml/src/ggml-sycl/wkv.cpp` | 294 |
| `ggml/src/ggml-sycl/wkv.hpp` | 11 |
| `ggml/src/ggml-sycl/CMakeLists.txt` | 225 |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-q4_0.cu` | 8 |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | 610 |
| `ggml/src/ggml-cuda/common.cuh` | 1718 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5857 |

> **说明**：覆盖声明 **179** 项 = 计划指派给 L6-05 的 **174** 个文件（`ggml/include/ggml-sycl.h` + `ggml/src/ggml-sycl/` 下 173 个）+ 本课逐字引用的 **4** 个 CUDA 侧对照文件（`common.cuh`、`fattn-vec.cuh`、`template-instances/fattn-vec-instance-f16-q4_0.cu`、`ggml-cuda.cu`）+ 1 个非覆盖域文件（下面的 `CMakeLists.txt`）。CUDA 这 4 个在计划里归属 L6-01 / L6-03 / L6-04，这里计入是因为「声明 = 本课逐字引用过的文件」：不虚报，也不漏报。
> **说明**：`ggml/src/ggml-sycl/CMakeLists.txt` 被逐字引用，但它不在覆盖域的源文件后缀集合内，按 `check_coverage.py` 的 B3 规则记为「真实存在但不计入覆盖率」，未写进覆盖声明。

## 场景（9 幕）

1. **SYCL 后端：CUDA 的平行实现** — Intel GPU 走 SYCL；它没有另立一套算子体系，而是把 CUDA 侧那一套照搬过来。
2. **174 个文件，七族** — 家族的划分几乎照抄 CUDA：一算子一对文件、模板实例单独成目录、主机框架独立。
3. **★ 同一批 kernel，同一套组织** — SYCL 的模板实例文件与 CUDA 的同名文件逐字相同，只有两处差异：.hpp/.cuh 与多一行 512。
4. **CUDA ↔ SYCL 结构对照表** — 同一个算子，两边落在哪个文件、哪个函数。左边一列看完，对应关系就成立。
5. **★ 差异在数据面，不在算子层** — 同一张算子表，换掉的是"怎么发命令、内存从哪来、谁来算 GEMM"。
6. **L3-01 的虚表在 SYCL 侧逐项填满** — registry、device、buffer type（外加 buffer）四张接口，字段顺序由 ggml-backend-impl.h 钉死。
7. **量化矩阵乘：三条路都搬过来了** — mmq（大 batch）/ mmvq（小 batch）/ dmmv（单向量）；三者共用同一套 q8_1 量化与 vec_dot。
8. **graph_compute：录制重放 vs 逐节点执行** — 和 CUDA 侧同一套判据：先问"这张图能不能录"，能录就用可更新的执行图，不能录就逐节点跑。
9. **把 SYCL 后端压成一张表** — 同一个算子，两边各在哪；记住这张表，L6-06 起的其它 GPU 后端都是同一套问法。

## 核心结论

### ★ SYCL 与 CUDA 结构同构

SYCL 后端不是另起炉灶，而是 CUDA 后端的平行实现。实测的对应关系：

| 环节 | CUDA 侧（file:line） | SYCL 侧（file:line） |
|---|---|---|
| 后端上下文 | `ggml_backend_cuda_context`（common.cuh:1455） | `ggml_backend_sycl_context`（common.hpp:336） |
| 执行流 | `cudaStream_t streams[][]`（common.cuh:1460） | `queue_ptr qptrs[][]`（common.hpp:341） |
| 设备内存池 | `ggml_cuda_pool_alloc`（common.cuh:1215） | `ggml_sycl_pool_alloc`（common.hpp:263） |
| buffer 虚表 | `ggml_backend_cuda_buffer_interface`（ggml-cuda.cu:853） | `ggml_backend_sycl_buffer_interface`（ggml-sycl.cpp:917） |
| device 虚表 | `ggml_backend_cuda_device_interface`（ggml-cuda.cu:5651） | `ggml_backend_sycl_device_interface`（ggml-sycl.cpp:6903） |
| MUL_MAT 分派 | `ggml_cuda_mul_mat`（ggml-cuda.cu:1823） | `ggml_sycl_mul_mat`（ggml-sycl.cpp:4765） |
| MMVQ 回调 | `ggml_cuda_op_mul_mat_vec_q`（mmvq.cuh:14） | `ggml_sycl_op_mul_mat_vec_q`（mmvq.hpp:19） |
| FA 入口 | `ggml_cuda_flash_attn_ext`（fattn.cu:755） | `ggml_sycl_flash_attn_ext`（fattn.cpp:276） |
| 图执行 | `ggml_backend_cuda_graph_compute`（ggml-cuda.cu:4421） | `ggml_backend_sycl_graph_compute`（ggml-sycl.cpp:6202） |

最硬的一条证据是宏：`DECL_FATTN_VEC_CASE` 在两边展开体完全相同，只差 `ggml_cuda_*` / `ggml_sycl_*` 与上下文类型名（fattn-vec.cuh:574 / fattn-vec.hpp:644）。

### ★ 差异在数据面与编译链

算子层可以照搬，数据面必须重写：

| 维度 | CUDA | SYCL |
|---|---|---|
| 执行流 | `cudaStream_t` | `sycl::queue *`（`queue_ptr`） |
| 设备内存 | `cudaMalloc` / `cudaFree` | USM：`sycl::malloc_device` / `free` |
| 事件 | `cudaEvent_t` | `dpct::event_ptr` |
| 数学库 | cuBLAS | oneDNN（`dnnl::matmul`）/ oneMKL |
| 编译链 | nvcc | oneAPI DPC++（`icpx -fsycl`） |

缓冲区对齐两边**都是 128 字节**（`SYCL_BUFFER_ALIGNMENT`，common.hpp:253；`ggml_backend_cuda_buffer_type_get_alignment` 直接 `return 128`，ggml-cuda.cu:903）。这解释了为什么 SYCL 的算子层能直接沿用 CUDA 侧对 tile / padding 的假设。

### SYCL 独有的分支

不是完全复制，SYCL 侧多出三条 CUDA 侧没有的路：

1. **厂商库 FlashAttention**：`BEST_FATTN_KERNEL_ONEDNN = 150` 与 `BEST_FATTN_KERNEL_MKL = 300`（fattn.cpp:97-103），对应 `fattn-onednn.cpp` 与 `fattn-mkl.cpp`；
2. **独立的 dmmv 路径**：`dmmv.cpp`（2227 行）+ reorder 变体，CUDA 侧 v0.5.0 已无同名文件；
3. **SYCL command graph**：需要 `ext_oneapi_limited_graph` / `ext_oneapi_graph` 两个设备特性，不像 CUDA 那样有 warmup 步骤。

反向的差异也有一处：`EXTERN_DECL_FATTN_VEC_CASES` 里 CUDA 侧有 `GGML_TYPE_BF16`，SYCL 侧没有（fattn-vec.cuh:585 vs fattn-vec.hpp:648-654）。

## 验收点

- [x] 保真门禁：21 处引用 —— 21 个引用块 / 21 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 179 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 3 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
