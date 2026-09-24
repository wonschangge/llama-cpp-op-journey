# L6-07 · Vulkan 后端：计算着色器 — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-06`（Vulkan 主机端：pipeline、descriptor set、shader 生成）；建议对照 `L1-04`（量化块布局）与 `L6-02`（CUDA 侧量化矩阵乘）

## 学习目标

看完这一课，你应该能：

1. 说出一个 `.comp` 从生成到被 pipeline 使用经过哪几步（对应验收点）；
2. 解释为什么 `strings_to_spv` 这一层是"变体工厂"——同一个 `.comp` 会被编译成很多个 SPIR-V；
3. 指出 GLSL 侧的 `block_q4_0` 与 CPU 侧 `ggml-common.h` 里的 `block_q4_0` 是两份独立的定义，并说出这对"新增一个量化类型"意味着什么；
4. 说出 `mul_mm.comp` / `mul_mmq.comp` / `mul_mat_vec_*.comp` 三个族各自负责什么形状的 `GGML_OP_MUL_MAT`；
5. 说明 `dequant_q4_0.comp` 里 64 个线程、32 个元素一块是如何映射到输出的 32 个元素上的。

## 覆盖的源文件（166 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_f32.comp` | 21 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq1_m.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq1_s.comp` | 36 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_s.comp` | 45 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_xs.comp` | 44 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_xxs.comp` | 50 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq3_s.comp` | 41 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq3_xxs.comp` | 52 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq4_nl.comp` | 33 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq4_xs.comp` | 35 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_mxfp4.comp` | 33 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_nvfp4.comp` | 33 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q1_0.comp` | 30 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_0.comp` | 30 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_k.comp` | 35 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q3_k.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_0.comp` | 31 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_1.comp` | 33 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_k.comp` | 69 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_0.comp` | 35 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_1.comp` | 36 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_k.comp` | 71 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q6_k.comp` | 34 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q8_0.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq1_0.comp` | 29 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq2_0.comp` | 32 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp` | 265 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq1_m.comp` | 133 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq1_s.comp` | 96 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_s.comp` | 91 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xs.comp` | 106 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xxs.comp` | 88 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_s.comp` | 94 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_xxs.comp` | 89 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq4_xs.comp` | 98 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_nc.comp` | 125 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_p021.comp` | 157 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k.comp` | 129 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q3_k.comp` | 133 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp` | 135 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q5_k.comp` | 166 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp` | 131 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq1_0.comp` | 86 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq2_0.comp` | 103 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp` | 143 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_split_k_reduce.comp` | 49 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp` | 560 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_cm2.comp` | 780 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp` | 320 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` | 771 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp` | 664 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp` | 569 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_1.comp` | 264 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_2.comp` | 409 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_mask_opt.comp` | 163 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.comp` | 103 |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp` | 122 |
| `ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp` | 196 |
| `ggml/src/ggml-vulkan/vulkan-shaders/soft_max_back.comp` | 55 |
| `ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large1.comp` | 63 |
| `ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large2.comp` | 80 |
| `ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large3.comp` | 66 |
| `ggml/src/ggml-vulkan/vulkan-shaders/group_norm.comp` | 67 |
| `ggml/src/ggml-vulkan/vulkan-shaders/l2_norm.comp` | 42 |
| `ggml/src/ggml-vulkan/vulkan-shaders/norm.comp` | 45 |
| `ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp` | 179 |
| `ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_back.comp` | 56 |
| `ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_partials.comp` | 88 |
| `ggml/src/ggml-vulkan/vulkan-shaders/rope_multi.comp` | 18 |
| `ggml/src/ggml-vulkan/vulkan-shaders/rope_neox.comp` | 18 |
| `ggml/src/ggml-vulkan/vulkan-shaders/rope_norm.comp` | 18 |
| `ggml/src/ggml-vulkan/vulkan-shaders/rope_vision.comp` | 18 |
| `ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/bfloat16.comp` | 8 |
| `ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat.comp` | 8 |
| `ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat2.comp` | 8 |
| `ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat2_decode_vector.comp` | 8 |
| `ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/float_e2m1.comp` | 8 |
| `ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/float_e4m3.comp` | 8 |
| `ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp` | 8 |
| `ggml/src/ggml-vulkan/vulkan-shaders/acc.comp` | 38 |
| `ggml/src/ggml-vulkan/vulkan-shaders/add.comp` | 70 |
| `ggml/src/ggml-vulkan/vulkan-shaders/add1.comp` | 29 |
| `ggml/src/ggml-vulkan/vulkan-shaders/add_id.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/arange.comp` | 21 |
| `ggml/src/ggml-vulkan/vulkan-shaders/argmax.comp` | 61 |
| `ggml/src/ggml-vulkan/vulkan-shaders/argsort.comp` | 93 |
| `ggml/src/ggml-vulkan/vulkan-shaders/argsort_large.comp` | 126 |
| `ggml/src/ggml-vulkan/vulkan-shaders/col2im_1d.comp` | 62 |
| `ggml/src/ggml-vulkan/vulkan-shaders/concat.comp` | 42 |
| `ggml/src/ggml-vulkan/vulkan-shaders/contig_copy.comp` | 54 |
| `ggml/src/ggml-vulkan/vulkan-shaders/conv2d_dw.comp` | 106 |
| `ggml/src/ggml-vulkan/vulkan-shaders/conv2d_mm.comp` | 474 |
| `ggml/src/ggml-vulkan/vulkan-shaders/conv3d_mm.comp` | 425 |
| `ggml/src/ggml-vulkan/vulkan-shaders/conv_transpose_1d.comp` | 99 |
| `ggml/src/ggml-vulkan/vulkan-shaders/copy.comp` | 26 |
| `ggml/src/ggml-vulkan/vulkan-shaders/copy_from_quant.comp` | 52 |
| `ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp` | 351 |
| `ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose.comp` | 68 |
| `ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose_02.comp` | 62 |
| `ggml/src/ggml-vulkan/vulkan-shaders/count_equal.comp` | 32 |
| `ggml/src/ggml-vulkan/vulkan-shaders/count_experts.comp` | 141 |
| `ggml/src/ggml-vulkan/vulkan-shaders/cross_entropy_loss.comp` | 79 |
| `ggml/src/ggml-vulkan/vulkan-shaders/cross_entropy_loss_back.comp` | 76 |
| `ggml/src/ggml-vulkan/vulkan-shaders/cumsum.comp` | 84 |
| `ggml/src/ggml-vulkan/vulkan-shaders/cumsum_multipass1.comp` | 61 |
| `ggml/src/ggml-vulkan/vulkan-shaders/cumsum_multipass2.comp` | 67 |
| `ggml/src/ggml-vulkan/vulkan-shaders/diag.comp` | 29 |
| `ggml/src/ggml-vulkan/vulkan-shaders/diag_mask_inf.comp` | 35 |
| `ggml/src/ggml-vulkan/vulkan-shaders/div.comp` | 28 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp` | 91 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post.comp` | 93 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_pre.comp` | 76 |
| `ggml/src/ggml-vulkan/vulkan-shaders/fill.comp` | 22 |
| `ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp` | 116 |
| `ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp` | 190 |
| `ggml/src/ggml-vulkan/vulkan-shaders/geglu.comp` | 14 |
| `ggml/src/ggml-vulkan/vulkan-shaders/geglu_erf.comp` | 28 |
| `ggml/src/ggml-vulkan/vulkan-shaders/geglu_quick.comp` | 12 |
| `ggml/src/ggml-vulkan/vulkan-shaders/get_rows.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/get_rows_back.comp` | 26 |
| `ggml/src/ggml-vulkan/vulkan-shaders/get_rows_quant.comp` | 52 |
| `ggml/src/ggml-vulkan/vulkan-shaders/gla.comp` | 83 |
| `ggml/src/ggml-vulkan/vulkan-shaders/im2col.comp` | 139 |
| `ggml/src/ggml-vulkan/vulkan-shaders/im2col_3d.comp` | 125 |
| `ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp` | 152 |
| `ggml/src/ggml-vulkan/vulkan-shaders/log.comp` | 18 |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul.comp` | 28 |
| `ggml/src/ggml-vulkan/vulkan-shaders/multi_add.comp` | 195 |
| `ggml/src/ggml-vulkan/vulkan-shaders/opt_step_adamw.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/opt_step_sgd.comp` | 23 |
| `ggml/src/ggml-vulkan/vulkan-shaders/out_prod.comp` | 60 |
| `ggml/src/ggml-vulkan/vulkan-shaders/pad.comp` | 65 |
| `ggml/src/ggml-vulkan/vulkan-shaders/pad_reflect_1d.comp` | 44 |
| `ggml/src/ggml-vulkan/vulkan-shaders/pool1d.comp` | 66 |
| `ggml/src/ggml-vulkan/vulkan-shaders/pool2d.comp` | 75 |
| `ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp` | 128 |
| `ggml/src/ggml-vulkan/vulkan-shaders/reglu.comp` | 10 |
| `ggml/src/ggml-vulkan/vulkan-shaders/repeat.comp` | 27 |
| `ggml/src/ggml-vulkan/vulkan-shaders/repeat_back.comp` | 38 |
| `ggml/src/ggml-vulkan/vulkan-shaders/roll.comp` | 47 |
| `ggml/src/ggml-vulkan/vulkan-shaders/scale.comp` | 25 |
| `ggml/src/ggml-vulkan/vulkan-shaders/silu_back.comp` | 27 |
| `ggml/src/ggml-vulkan/vulkan-shaders/snake.comp` | 50 |
| `ggml/src/ggml-vulkan/vulkan-shaders/solve_tri.comp` | 82 |
| `ggml/src/ggml-vulkan/vulkan-shaders/ssm_conv.comp` | 61 |
| `ggml/src/ggml-vulkan/vulkan-shaders/ssm_scan.comp` | 135 |
| `ggml/src/ggml-vulkan/vulkan-shaders/sub.comp` | 30 |
| `ggml/src/ggml-vulkan/vulkan-shaders/sum_rows.comp` | 48 |
| `ggml/src/ggml-vulkan/vulkan-shaders/swiglu.comp` | 10 |
| `ggml/src/ggml-vulkan/vulkan-shaders/swiglu_clamp.comp` | 13 |
| `ggml/src/ggml-vulkan/vulkan-shaders/swiglu_oai.comp` | 15 |
| `ggml/src/ggml-vulkan/vulkan-shaders/timestep_embedding.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/topk_argsort.comp` | 119 |
| `ggml/src/ggml-vulkan/vulkan-shaders/topk_moe.comp` | 222 |
| `ggml/src/ggml-vulkan/vulkan-shaders/topk_nary_search.comp` | 247 |
| `ggml/src/ggml-vulkan/vulkan-shaders/topk_radix_select.comp` | 145 |
| `ggml/src/ggml-vulkan/vulkan-shaders/tri.comp` | 43 |
| `ggml/src/ggml-vulkan/vulkan-shaders/unary.comp` | 206 |
| `ggml/src/ggml-vulkan/vulkan-shaders/upscale.comp` | 179 |
| `ggml/src/ggml-vulkan/vulkan-shaders/wkv6.comp` | 88 |
| `ggml/src/ggml-vulkan/vulkan-shaders/wkv7.comp` | 92 |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | 1432 |
| `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` | 2025 |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 16275 |
| `ggml/src/ggml-common.h` | 1912 |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` | 757 |

> **说明**：本课声明计划里 L6-07 的**全部 162 个文件**（161 个 `.comp` + `vulkan-shaders-gen.cpp`），无一遗漏；计数命令见 `source.md` 第一节。
> **说明**：另有 4 个文件被本课引用：`ggml/src/ggml-vulkan/ggml-vulkan.cpp`（归 L6-06，本课用它的第 554-572 与 5377-5385 行说明"被 pipeline 使用"这一步）、`ggml/src/ggml-common.h`（归 L1-04，本课用第 194-199 行做 GLSL/C 两份块定义的对照）。这两个文件在覆盖域内，各自的主覆盖仍在原课。
> **说明**：还有 `types.glsl` 与 `dequant_funcs.glsl`（Vulkan 着色器的 `.glsl` 头文件）。`.glsl` 后缀**不在覆盖域**内（覆盖域只收 `.comp`），所以它们**不计入覆盖率**，仅作为论据逐字引用 —— 计入与不计入在这里是分开的账。
> **说明**：本课不讨论任何命令行参数；参数门禁的真值集来自真实二进制与本仓库门禁脚本的帮助输出。

## 场景（9 幕）

1. **161 个 .comp：Vulkan 的算子实现全在这里** — 先按名字分族数一遍。族名就是 ggml op 的名字 —— 这是本仓库最一致的命名约定之一。
2. **★ 一个 .comp 是模板：string_to_spv 是变体工厂** — 验收点：说出一个 .comp 从生成到被 pipeline 使用经过哪几步。这 16 行是第 2、3 步。
3. **宏定义变成 glslc 命令行** — 这一步没有任何魔法：一个 define 一个 -D，最后 -o 出 .spv（下面是非 Windows 分支）。
4. **.spv 变成 C 数组：&lt;name&gt;_data[] / &lt;name&gt;_len** — 这一步之后，SPIR-V 就是普通的目标文件内容 —— 跟着 ggml-vulkan 一起被链接。
5. **★ 量化块定义：GLSL 与 C 是两份定义** — 形状完全一样，但写在两个文件里、由两套代码各自维护 —— 这是本课最值得记住的一件事。
6. **dequant_q4_0.comp：一个反量化内核的全部 30 行** — 最大的族、最短的文件。看懂这 30 行，其余的 dequant_*.comp 都是同一个骨架换个块布局。
7. **mul_mmq.comp：B 侧先量化成 q8_1，再整数点积** — 三个矩阵乘族分工不同：大 M 走 mul_mm，小 M 走 mul_mat_vec，整数点积走 mul_mmq。
8. **mul_mat_vec_q4_k.comp：6-bit scale 是位运算拼出来的** — K-quant 的 super-block 有 256 个元素，8 个 6-bit scale 被打包进 12 字节 —— 所以解包就是移位与掩码。
9. **把这一课压成一张表** — 六步链 + 三个数字。回到第 2 幕的那 16 行 —— 整条链的枢纽就在那里。

## 核心结论

### ★ 一个 `.comp` 是模板，SPIR-V 才是产物

六步链：

```text
process_shaders()  ->  string_to_spv()  ->  glslc  ->  write_output_files()
                   ->  createShaderModule()  ->  vkCreateComputePipelines  ->  查表 dispatch
```

第 2 步到第 3 步之间是**一变多**：同一个 `.comp` 配不同的宏，编译出几百个 `.spv`。所以"Vulkan 后端有多少着色器"这个问题没有单一答案 —— 取决于运行 `vulkan-shaders-gen` 时打开了哪些编译期开关（coopmat / coopmat2 / 整数点积 / bf16 / fp4 ...）。

### ★ 量化块的定义是复制出来的，不是共享的

同一个 `block_q4_0` 至少存在两份**独立**定义：`ggml-common.h`（C，CPU 内核用）与 `types.glsl`（GLSL，Vulkan 着色器用），CUDA / Metal / SYCL 各后端还有各自的版本。形状必须逐个字段一致，但**编译器不会替你检查这件事**。

因此新增一个量化类型是一串跨语言的改动：C 块定义 → GLSL 块定义 → `dequant_<type>.comp` → 生成器的 `type_names` → 各后端 kernel 的类型分发。这就是"量化类型是全局概念"的具体含义。

### 三个矩阵乘族的分工边界是 M

| 族 | 文件数 | 负责 |
|---|---|---|
| `mul_mat_vec*` | 19 | 小 M（逐 token 解码）与 `MUL_MAT_ID` |
| `mul_mm*` / `mul_mmq` | 4 | 大 M；`mul_mmq` 是整数点积（B 侧 q8_1） |
| `dequant_*` | 26 | 把量化权重先反量化成 f16，供 f16 路径使用 |

三者合计 49 个文件，占 161 个 `.comp` 的三成 —— 矩阵乘是 GPU 后端里最贵的算子，也是变体最多的地方。

## 验收点

- [x] 保真门禁：24 处引用 —— 24 个引用块 / 62 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 166 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
