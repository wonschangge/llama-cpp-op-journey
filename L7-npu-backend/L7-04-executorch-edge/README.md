# L7-04 · ExecuTorch 后端（边缘/移动端）：离线编译的设备内核与运行期派发 — 课件说明

> 层：**L7 · NPU 与加速器后端** ｜ 前置课：`L7-03`

## 学习目标

看完这一课，你应该能：

1. 说出这个后端的"编译"发生在哪个阶段、运行期 `ggml_backend_et_graph_compute` 只做哪几件事；
2. 说出 `rt::IRuntime` / `dev::IDeviceLayer` 各自负责什么，以及"加载内核"这条路径上的 4 个步骤；
3. 说清 ET 与 GPU 后端在数据面抽象上的差别（跨边界传什么、设备侧能看到什么元数据、内存怎么来）；
4. 指出 ET 的 buffer 有哪些硬性假设（cache line 对齐、`nb[0]` 连续、`is_host = false`、`cpy_tensor = false`）；
5. 解释 uberkernel 与宿主侧算子融合的区别，以及它为什么要靠设备侧全局屏障。

## 覆盖的源文件（68 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-et.h` | 29 |
| `ggml/src/ggml-et/ggml-et-common.h` | 87 |
| `ggml/src/ggml-et/ggml-et-cpu-compare.cpp` | 503 |
| `ggml/src/ggml-et/ggml-et-cpu-compare.h` | 55 |
| `ggml/src/ggml-et/ggml-et-kernels.cpp` | 509 |
| `ggml/src/ggml-et/ggml-et-kernels.h` | 49 |
| `ggml/src/ggml-et/ggml-et-memops.cpp` | 37 |
| `ggml/src/ggml-et/ggml-et-memops.h` | 19 |
| `ggml/src/ggml-et/ggml-et-ops.cpp` | 2585 |
| `ggml/src/ggml-et/ggml-et-ops.h` | 394 |
| `ggml/src/ggml-et/ggml-et-uberkernel-common.h` | 18 |
| `ggml/src/ggml-et/ggml-et.cpp` | 1881 |
| `ggml/src/ggml-et/et-kernels/src/block_ops.h` | 998 |
| `ggml/src/ggml-et/et-kernels/src/ggml_tensor.h` | 45 |
| `ggml/src/ggml-et/et-kernels/src/math_fp.h` | 300 |
| `ggml/src/ggml-et/et-kernels/src/platform.h` | 546 |
| `ggml/src/ggml-et/et-kernels/src/quants.h` | 73 |
| `ggml/src/ggml-et/et-kernels/src/tensor.h` | 898 |
| `ggml/src/ggml-et/et-kernels/src/clamp_f32.c` | 121 |
| `ggml/src/ggml-et/et-kernels/src/concat_f32.c` | 176 |
| `ggml/src/ggml-et/et-kernels/src/cont_f16.c` | 108 |
| `ggml/src/ggml-et/et-kernels/src/cont_f32.c` | 249 |
| `ggml/src/ggml-et/et-kernels/src/conv_2d_f32_me.c` | 808 |
| `ggml/src/ggml-et/et-kernels/src/cpy_f32_f16.c` | 111 |
| `ggml/src/ggml-et/et-kernels/src/cumsum_f32.c` | 97 |
| `ggml/src/ggml-et/et-kernels/src/diag_f32.c` | 91 |
| `ggml/src/ggml-et/et-kernels/src/el_map_f32.c` | 378 |
| `ggml/src/ggml-et/et-kernels/src/fill_f32.c` | 88 |
| `ggml/src/ggml-et/et-kernels/src/flash_attn_ext_f16_me.c` | 1001 |
| `ggml/src/ggml-et/et-kernels/src/flash_attn_ext_f32.c` | 218 |
| `ggml/src/ggml-et/et-kernels/src/gated_delta_net_f32.c` | 347 |
| `ggml/src/ggml-et/et-kernels/src/get_rows_f32.c` | 613 |
| `ggml/src/ggml-et/et-kernels/src/glu_f32.c` | 607 |
| `ggml/src/ggml-et/et-kernels/src/group_norm_f32.c` | 172 |
| `ggml/src/ggml-et/et-kernels/src/im2col.c` | 131 |
| `ggml/src/ggml-et/et-kernels/src/l2_norm_f32.c` | 238 |
| `ggml/src/ggml-et/et-kernels/src/mean_f32.c` | 221 |
| `ggml/src/ggml-et/et-kernels/src/memops.c` | 182 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_Q4_0.c` | 359 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_Q4_0_matrix_engine.c` | 369 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_Q8_0.c` | 414 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_f16.c` | 143 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_f16_matrix_engine.c` | 330 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_f32.c` | 138 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_f32_matrix_engine.c` | 156 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_id_Q4_0.c` | 170 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_id_Q8_0.c` | 161 |
| `ggml/src/ggml-et/et-kernels/src/mul_mat_id_f32.c` | 289 |
| `ggml/src/ggml-et/et-kernels/src/norm_f32.c` | 329 |
| `ggml/src/ggml-et/et-kernels/src/pad_f32.c` | 166 |
| `ggml/src/ggml-et/et-kernels/src/repeat_f32.c` | 119 |
| `ggml/src/ggml-et/et-kernels/src/rms_norm_f32.c` | 271 |
| `ggml/src/ggml-et/et-kernels/src/rms_norm_mul_f32.c` | 291 |
| `ggml/src/ggml-et/et-kernels/src/rope_f32.c` | 657 |
| `ggml/src/ggml-et/et-kernels/src/rwkv_wkv6_f32.c` | 185 |
| `ggml/src/ggml-et/et-kernels/src/rwkv_wkv7_f32.c` | 273 |
| `ggml/src/ggml-et/et-kernels/src/scale_f32.c` | 95 |
| `ggml/src/ggml-et/et-kernels/src/set_f32.c` | 102 |
| `ggml/src/ggml-et/et-kernels/src/set_rows_f32.c` | 395 |
| `ggml/src/ggml-et/et-kernels/src/softmax_f32.c` | 699 |
| `ggml/src/ggml-et/et-kernels/src/solve_tri_f32.c` | 110 |
| `ggml/src/ggml-et/et-kernels/src/sqr_f32.c` | 89 |
| `ggml/src/ggml-et/et-kernels/src/ssm_conv_f32.c` | 130 |
| `ggml/src/ggml-et/et-kernels/src/ssm_scan_f32.c` | 283 |
| `ggml/src/ggml-et/et-kernels/src/sum_rows_f32.c` | 104 |
| `ggml/src/ggml-et/et-kernels/src/tri_f32.c` | 245 |
| `ggml/src/ggml-et/et-kernels/src/uberkernel.c` | 498 |
| `ggml/src/ggml-et/et-kernels/src/unary_f32.c` | 706 |

> **说明**：本课覆盖声明是 `tools/plan_matrix.py --files` 里 L7-04 的**全部 68 个文件**：`ggml/include/ggml-et.h`（1）+ `ggml/src/ggml-et/*.{cpp,h}`（11）+ `ggml/src/ggml-et/et-kernels/src/*.h`（6）+ `et-kernels/src/*.c`（50）。合计 23191 行。
> **说明**：**正文引用但不计入覆盖率的文件**（不在本视角的源文件后缀/路径里，故未进覆盖声明）：`ggml/src/ggml-et/CMakeLists.txt`（内核清单与嵌入）、`ggml/src/ggml-et/et-kernels/CMakeLists.txt`（RISC-V 交叉编译）、`ggml/src/ggml-et/et-kernels/scripts/check_unimplemented_instructions.sh`、`docs/backend/ET.md`（上游文档）、`et-kernels/src/RunBackend.sh`、以及同目录下不在覆盖域的 `crt.S` / `linker.ld`。另有若干对照引用属于别的课：`ggml/src/ggml-cuda/scale.cu`（L6-01）、`ggml/src/ggml-cuda/ggml-cuda.cu`（L6-01）、`ggml/src/ggml-cann/acl_tensor.cpp`（L7-01）、`ggml/src/ggml-openvino/ggml-decoder.cpp`（L7-03）、`ggml/src/ggml-common.h`（L1-04）。
> **说明**：**关于课名与计划口径的修正**：计划文档把这一课记为"ExecuTorch 后端"，讲解要点是"ExecuTorch 的委托机制、图导出与运行时"。实测（`grep -rni` 全目录）`ggml/src/ggml-et/` 里没有 `executorch` / `torch` / `delegate` / `.pte` 任何符号；实际机制是"ET 平台 SDK（`dev::IDeviceLayer` + `rt::IRuntime`）+ 构建期交叉编译的内核 ELF"。本课按代码实际机制讲解，并在第 3、8 幕显式对照这两种口径。
> **说明**：**未实测的部分**（明确边界）：ET-SoC 是专用硬件，本机没有 `/opt/et` 平台与 RISC-V 工具链，因此本课**没有**真的构建或运行过这个后端。所有结论都来自**逐字引用的源码**与**行号可核对的注释**；凡是"注释声称"的地方（如硬件未实现指令、L2/L1 不连贯）本课都标明出处，未当作已实测的行为。

## 场景（9 幕）

1. **ET 后端：68 个文件分成两个世界** — 宿主侧 11 个文件说"怎么派发"，设备侧 56 个文件说"内核怎么写、怎么被加载"。
2. **运行期没有编译：一张图 = 一个 for + 一个 switch** — graph_compute 逐节点分派，只在两处做图级融合（RMS_NORM+MUL、MUL_MAT+ADD）。
3. **★ "编译"发生在图之外：内核是预编译的 ELF，运行期按名字取** — 这段代码里没有任何编译器：只有"名字 -> 字节 -> runtime->loadCode() -> KernelId"。
4. **真正的"运行时"是 SDK 的两个对象，不是 ggml** — ggml 侧只持有两个 shared_ptr：设备层 + 运行时；流、内核句柄、事件都挂在设备上下文里。
5. **★ 跨边界的不是裸指针，是整个 struct ggml_tensor** — 宿主侧把张量结构体按值拷进参数块；设备侧于是拿到同一套 ne/nb/type/data 词汇表。
6. **设备内核：同一个结构体，裸机环境里再声明一遍** — entry_point(params, env)：参数是宿主打包好的那块内存，env 告诉它有多少个 hart 可用。
7. **内存契约：设备内存、cache 对齐、没有 host 映射** — buffer_type 虚表把"这块内存是怎么来的"讲清楚了；supports_op 再把内核的假设写成前置条件。
8. **与 GPU 后端比：数据面抽象差在哪** — 同一套 ggml_backend_i 契约之下，两边"过边界的东西"完全不同。
9. **把这一课压成一张表** — 运行期只有执行与失败两种结局；所有"翻译"都发生在构建期。

## 核心结论

### 一句话

**ET 后端的"编译"在图之外**：50 个裸机内核在构建期交叉编译成 ELF 并嵌进宿主库，运行期只有"按名字取代码 -> `loadCode` -> `kernelLaunch`"；`graph_compute` 是一个 `for` 套一个 `switch`（39 个 case），没有任何图级编译或代码生成。

### 与 CANN / OpenVINO 的路线差别

L7-01 的 CANN 与 L7-03 的 OpenVINO 都是**运行期映射**：前者把 ggml op 变成 ACL 描述符（`aclCreateTensor`，`ggml-cann/acl_tensor.cpp:89`），后者把 ggml 子图翻成 `ov::Tensor` / `ov::op` 组成的模型（`ggml-openvino/ggml-decoder.cpp:1392`）再交给 OpenVINO 运行时（L7-03 展开）。ET 把"翻译"提前到构建期，运行期只做执行 —— 代价是**算子集固定**：没有内核的 op 只能在 `supports_op` 阶段被拒（44 个 case），回退由调度器（L4-02）完成。

### 数据面抽象：共享表示 + 独立内存

**共享表示**：跨边界的是整个 `struct ggml_tensor`（`ggml-et-ops.cpp:282-283`），设备内核直接读 `ne[]` / `nb[]` / `type` / `data`（`et-kernels/src/scale_f32.c:32-40`），连量化块的 `block_q8_0` 都复用 `ggml-common.h`（`et-kernels/src/quants.h:10-11`）。

**独立内存**：设备内存只能由 `runtime->mallocDevice` 给（`ggml-et.cpp:404`），`is_host = false`（439）、`host_buffer = false`（1649）、`buffer_from_host_ptr = NULL`（1681）、`cpy_tensor` 恒 false（333-340）。这与 CUDA 的 pinned host buffer（`ggml-cuda.cu:1309-1317`）和 `cpy_tensor_async`（2485）正好相反。

## 验收点

- [x] 保真门禁：27 处引用 —— 27 个引用块 / 62 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 68 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 5 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
