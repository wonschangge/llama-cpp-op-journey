<!-- llama-coverage
ggml/src/ggml-vulkan/vulkan-shaders/dequant_f32.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq1_m.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq1_s.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_s.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_xs.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_xxs.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq3_s.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq3_xxs.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq4_nl.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq4_xs.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_mxfp4.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_nvfp4.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q1_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q3_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_1.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q4_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_1.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q5_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q6_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q8_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq1_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq2_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq1_m.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq1_s.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_s.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xs.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xxs.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_s.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_xxs.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq4_xs.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_nc.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_p021.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q3_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q5_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq1_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq2_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_split_k_reduce.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_cm2.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_1.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_2.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_mask_opt.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp
ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp
ggml/src/ggml-vulkan/vulkan-shaders/soft_max_back.comp
ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large1.comp
ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large2.comp
ggml/src/ggml-vulkan/vulkan-shaders/soft_max_large3.comp
ggml/src/ggml-vulkan/vulkan-shaders/group_norm.comp
ggml/src/ggml-vulkan/vulkan-shaders/l2_norm.comp
ggml/src/ggml-vulkan/vulkan-shaders/norm.comp
ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp
ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_back.comp
ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_partials.comp
ggml/src/ggml-vulkan/vulkan-shaders/rope_multi.comp
ggml/src/ggml-vulkan/vulkan-shaders/rope_neox.comp
ggml/src/ggml-vulkan/vulkan-shaders/rope_norm.comp
ggml/src/ggml-vulkan/vulkan-shaders/rope_vision.comp
ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/bfloat16.comp
ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat.comp
ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat2.comp
ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat2_decode_vector.comp
ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/float_e2m1.comp
ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/float_e4m3.comp
ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp
ggml/src/ggml-vulkan/vulkan-shaders/acc.comp
ggml/src/ggml-vulkan/vulkan-shaders/add.comp
ggml/src/ggml-vulkan/vulkan-shaders/add1.comp
ggml/src/ggml-vulkan/vulkan-shaders/add_id.comp
ggml/src/ggml-vulkan/vulkan-shaders/arange.comp
ggml/src/ggml-vulkan/vulkan-shaders/argmax.comp
ggml/src/ggml-vulkan/vulkan-shaders/argsort.comp
ggml/src/ggml-vulkan/vulkan-shaders/argsort_large.comp
ggml/src/ggml-vulkan/vulkan-shaders/col2im_1d.comp
ggml/src/ggml-vulkan/vulkan-shaders/concat.comp
ggml/src/ggml-vulkan/vulkan-shaders/contig_copy.comp
ggml/src/ggml-vulkan/vulkan-shaders/conv2d_dw.comp
ggml/src/ggml-vulkan/vulkan-shaders/conv2d_mm.comp
ggml/src/ggml-vulkan/vulkan-shaders/conv3d_mm.comp
ggml/src/ggml-vulkan/vulkan-shaders/conv_transpose_1d.comp
ggml/src/ggml-vulkan/vulkan-shaders/copy.comp
ggml/src/ggml-vulkan/vulkan-shaders/copy_from_quant.comp
ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp
ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose.comp
ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose_02.comp
ggml/src/ggml-vulkan/vulkan-shaders/count_equal.comp
ggml/src/ggml-vulkan/vulkan-shaders/count_experts.comp
ggml/src/ggml-vulkan/vulkan-shaders/cross_entropy_loss.comp
ggml/src/ggml-vulkan/vulkan-shaders/cross_entropy_loss_back.comp
ggml/src/ggml-vulkan/vulkan-shaders/cumsum.comp
ggml/src/ggml-vulkan/vulkan-shaders/cumsum_multipass1.comp
ggml/src/ggml-vulkan/vulkan-shaders/cumsum_multipass2.comp
ggml/src/ggml-vulkan/vulkan-shaders/diag.comp
ggml/src/ggml-vulkan/vulkan-shaders/diag_mask_inf.comp
ggml/src/ggml-vulkan/vulkan-shaders/div.comp
ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp
ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post.comp
ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_pre.comp
ggml/src/ggml-vulkan/vulkan-shaders/fill.comp
ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp
ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp
ggml/src/ggml-vulkan/vulkan-shaders/geglu.comp
ggml/src/ggml-vulkan/vulkan-shaders/geglu_erf.comp
ggml/src/ggml-vulkan/vulkan-shaders/geglu_quick.comp
ggml/src/ggml-vulkan/vulkan-shaders/get_rows.comp
ggml/src/ggml-vulkan/vulkan-shaders/get_rows_back.comp
ggml/src/ggml-vulkan/vulkan-shaders/get_rows_quant.comp
ggml/src/ggml-vulkan/vulkan-shaders/gla.comp
ggml/src/ggml-vulkan/vulkan-shaders/im2col.comp
ggml/src/ggml-vulkan/vulkan-shaders/im2col_3d.comp
ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp
ggml/src/ggml-vulkan/vulkan-shaders/log.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul.comp
ggml/src/ggml-vulkan/vulkan-shaders/multi_add.comp
ggml/src/ggml-vulkan/vulkan-shaders/opt_step_adamw.comp
ggml/src/ggml-vulkan/vulkan-shaders/opt_step_sgd.comp
ggml/src/ggml-vulkan/vulkan-shaders/out_prod.comp
ggml/src/ggml-vulkan/vulkan-shaders/pad.comp
ggml/src/ggml-vulkan/vulkan-shaders/pad_reflect_1d.comp
ggml/src/ggml-vulkan/vulkan-shaders/pool1d.comp
ggml/src/ggml-vulkan/vulkan-shaders/pool2d.comp
ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp
ggml/src/ggml-vulkan/vulkan-shaders/reglu.comp
ggml/src/ggml-vulkan/vulkan-shaders/repeat.comp
ggml/src/ggml-vulkan/vulkan-shaders/repeat_back.comp
ggml/src/ggml-vulkan/vulkan-shaders/roll.comp
ggml/src/ggml-vulkan/vulkan-shaders/scale.comp
ggml/src/ggml-vulkan/vulkan-shaders/silu_back.comp
ggml/src/ggml-vulkan/vulkan-shaders/snake.comp
ggml/src/ggml-vulkan/vulkan-shaders/solve_tri.comp
ggml/src/ggml-vulkan/vulkan-shaders/ssm_conv.comp
ggml/src/ggml-vulkan/vulkan-shaders/ssm_scan.comp
ggml/src/ggml-vulkan/vulkan-shaders/sub.comp
ggml/src/ggml-vulkan/vulkan-shaders/sum_rows.comp
ggml/src/ggml-vulkan/vulkan-shaders/swiglu.comp
ggml/src/ggml-vulkan/vulkan-shaders/swiglu_clamp.comp
ggml/src/ggml-vulkan/vulkan-shaders/swiglu_oai.comp
ggml/src/ggml-vulkan/vulkan-shaders/timestep_embedding.comp
ggml/src/ggml-vulkan/vulkan-shaders/topk_argsort.comp
ggml/src/ggml-vulkan/vulkan-shaders/topk_moe.comp
ggml/src/ggml-vulkan/vulkan-shaders/topk_nary_search.comp
ggml/src/ggml-vulkan/vulkan-shaders/topk_radix_select.comp
ggml/src/ggml-vulkan/vulkan-shaders/tri.comp
ggml/src/ggml-vulkan/vulkan-shaders/unary.comp
ggml/src/ggml-vulkan/vulkan-shaders/upscale.comp
ggml/src/ggml-vulkan/vulkan-shaders/wkv6.comp
ggml/src/ggml-vulkan/vulkan-shaders/wkv7.comp
ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
ggml/src/ggml-vulkan/vulkan-shaders/types.glsl
ggml/src/ggml-vulkan/ggml-vulkan.cpp
ggml/src/ggml-common.h
ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl
-->

# L6-07 · Vulkan 后端：计算着色器 — 源文件

**一句话**：Vulkan 后端里，"一个算子"就是一个 `.comp`（GLSL 计算着色器）；而 `.comp` 本身只是**模板** —— 真正被 `vkCreateComputePipelines` 用掉的是由 `vulkan-shaders-gen.cpp` 用 `glslc` 针对一组宏定义编译出来的 `.spv`，再被写成 C 数组编进 `ggml-vulkan` 的静态库里。

这一课覆盖 `vulkan-shaders/` 下的**全部 161 个 `.comp`** 加 1 个生成器（`vulkan-shaders-gen.cpp`），共 162 个源文件。逐字引用的只是少数代表，其余在 `source.md` 里按族列出。

回顾 **L6-06**：那一课讲主机端怎么*使用*这些着色器（pipeline 缓存、descriptor set、push constants）；本课讲这些着色器**本身**是怎么写的、怎么被造出来的。回顾 **L1-04**：那一课画的是 CPU 侧量化块的内存布局 —— 本课会看到 GLSL 侧**另一份、但形状完全相同**的定义。对照 **L6-02**：CUDA 侧同类内核用 `.cu` 模板实例化，Vulkan 侧用 `.comp` + 宏定义，解决的是同一个问题。

---

## 一、161 个 `.comp` 按前缀分族

计数命令（两个数都对过）：

```text
$ ls ggml/src/ggml-vulkan/vulkan-shaders/*.comp | wc -l        # 154
$ find ggml/src/ggml-vulkan/vulkan-shaders -name "*.comp" | wc -l  # 161（含 feature-tests/ 下 7 个）
```

按前缀分成 14 个族。`type_names` 是生成器里的**类型清单**：清单里每个名字都会在 `mul_mat_vec*`、`mul_mm*`、`dequant_*` 等族里派生出变体，所以"加一个量化类型"在生成器里就是往这个 vector 里加一行。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp -->
```cpp
const std::vector<std::string> type_names = {
    "f32",
    "f16",
    "q1_0",
    "q2_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "nvfp4",
    "tq1_0",
    "tq2_0",
    "bf16",
};
```

## 二、★ 六步链的枢纽：`string_to_spv()`

生成器的入口是 `process_shaders()`，它按 (着色器文件 × 类型 × 变体) 调用 `string_to_spv()`。这个函数做三件事：把**变体名**拼出来（精度后缀 + coopmat 档 + 自定义 suffix）、算出 `<output_dir>/<name>.spv`、然后异步交给 `string_to_spv_func()` 去跑 `glslc`。第 444 行那个"只编译与自己同名的源文件"的分支解释了 CMake 的调用方式：**每个 `.comp` 一个生成器进程**，最后再汇总成同一个头文件。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp -->
```cpp
void string_to_spv(std::string name, const std::string& source, const std::map<std::string, std::string>& defines, bool fp16 = true, bool coopmat = false, bool coopmat2 = false, bool f16acc = false, const std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string out_path = join_paths(output_dir, name + ".spv");

    if (input_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, out_path));
        return;
    } else if (basename(input_filepath) != source) {
        // Only compile shader variants matching the input filename
        return;
    }

    compile_count_guard slot = acquire_compile_slot();
    compiles.push_back(std::async(
        string_to_spv_func, name, input_filepath, out_path, defines, coopmat, generate_dep_file, std::move(slot)));
```

## 三、变体并不总是"一类一变"：`MULMAT_QUANT` 用一个 SPIR-V 覆盖全部量化类型

大多数族是"一个类型一个变体"。但 `mul_mm.comp` 有一条例外路径：源码注释写得很直白 —— *one SPIR-V for all quant types, selected via MmTypeA spec constant*。它只编译 `_quant_f16` / `_quant_f32` 两个变体，运行时用 specialization constant `MmTypeA` 选择走哪一支。代价是着色器里多一层运行时分支，收益是 pipeline 数量大减。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp -->
```cpp
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        const std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_to_spv(shader_name + "_quant_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        if (!coopmat2) {
            string_to_spv(shader_name + "_quant_f32" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
```

## 四、第 3 步：`glslc` 命令行是怎么拼出来的

命令行只有一个必选阶段参数（compute）、一个由变体名决定的目标环境，以及一大串 `-D`。`defines` 是 `std::map`，所以顺序稳定、命令行可复现。注意第 363 行那个分支：coopmat / bf16 / rope / dot2 四类变体**不加** spirv-opt，源码注释逐条给了上游 issue 编号 —— 这是"优化遍会改坏某些着色器"的现场记录。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp -->
```cpp
void string_to_spv_func(std::string name, std::string in_path, std::string out_path, std::map<std::string, std::string> defines, bool coopmat, bool dep_file, compile_count_guard slot) {
    std::string target_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #ifdef _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, "\"" + in_path + "\"", "-o", "\"" + out_path + "\""};
    #else
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, in_path, "-o", out_path};
    #endif

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    if (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    if (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#ifdef _WIN32
        cmd.push_back("\"" + target_cpp + ".d\"");
#else
        cmd.push_back(target_cpp + ".d");
#endif
    }

    #ifdef GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #endif

    for (const auto& define : defines) {
        cmd.push_back("-D" + define.first + "=" + define.second);
    }

```

## 五、第 4 步：`.spv` 变成 C 数组

`write_output_files()` 遍历 `shader_fnames`（第 2 步每登记一个变体就多一项），把每个 `.spv` 读回来，逐字节写成 C 数组，并在头文件里给出 `extern` 声明。这一步是"Vulkan 后端不需要运行时编译着色器"的全部原因：SPIR-V 变成了和 `.o` 一样的目标文件内容。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp -->
```cpp
        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        if (input_filepath != "") {
            std::string data = read_binary_file(path);
            if (data.empty()) {
                continue;
            }

            src << "const uint64_t " << name << "_len = " << data.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << data.size() << "] = {\n" << std::hex;
            auto bytes = reinterpret_cast<const uint8_t*>(data.data());
            for (size_t i = 0; i < data.size(); ++i) {
                src << "0x" << static_cast<int>(bytes[i]) << ",";
                if ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }
```

## 六、第 5 步：主机端把 `spv_data` 变成 shader module

`ggml_vk_create_pipeline_func()` 的入参就是上一步产出的 `spv_size` / `spv_data`。它先按设备能力**补丁 SPIR-V**（打开支持的 fp16 float controls），再建 `vk::ShaderModuleCreateInfo`，随后 `createShaderModule()`，最后连同 specialization constants 与 push constant range 一起建 compute pipeline。这一课只看它与上一课的接缝；细节在 L6-06。

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan.cpp -->
```cpp
static void ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_size, const void* spv_data, const std::string entrypoint,
                                         uint32_t parameter_count, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness, bool require_full_subgroups, uint32_t required_subgroup_size) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_count <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(parameter_count <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    if (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        const uint32_t* spv_words = reinterpret_cast<const uint32_t *>(spv_data);
        size_t word_count = spv_size / sizeof(uint32_t);
        spirv.assign(spv_words, spv_words + word_count);
```

## 七、第 6 步：运行时按 (类型, 列数, workgroup 档) 查表

pipeline 全部建成后放在 `device->pipeline_dequant_mul_mat_vec_f32_f32[w][a_type][i]` 这类三维表里。运行时 `ggml_vk_get_dequantize_mul_mat_vec()` 根据 B 的类型（f32/f16/q8_1）、A 的类型、以及算出的 workgroup 档位取出一个 pipeline，交给 `ggml_vk_dispatch_pipeline()`。**"一个 ggml op 落在哪个 SPIR-V 上"最终就压缩成这一次数组索引。**

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan.cpp -->
```cpp
    if (b_type == GGML_TYPE_Q8_1) {
        if (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
        }
        return ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][num_cols-1];
    }

    return b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][num_cols-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][num_cols-1];
}
```

## 八、★ GLSL 侧的量化块：`types.glsl`

`types.glsl` 是 GLSL 侧的"量化块字典"：每个类型一个 struct，加上块大小宏，再用 `#if defined(DATA_A_xxx)` 把当前编译的类型绑到 `QUANT_K` / `A_TYPE` 上。因为这一切发生在**编译期**，一个 SPIR-V 只认识一种 A 类型 —— 这正是变体数量的来源。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/types.glsl -->
```glsl
#define QUANT_K_Q4_0 32
#define QUANT_R_Q4_0 2

struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};
struct block_q4_0_packed16
{
    float16_t d;
    uint16_t qs[16/2];
};

#if defined(DATA_A_Q4_0)
#define QUANT_K QUANT_K_Q4_0
#define QUANT_R QUANT_R_Q4_0
#define QUANT_AUXF 1
#define A_TYPE block_q4_0
#define A_TYPE_PACKED16 block_q4_0_packed16
#define DATA_A_QUANT_LEGACY
#endif
```

## 九、★ C 侧的同一个块（L1-04）

同一个 Q4_0 块在 CPU 侧的定义。字段逐个对应：`ggml_half d` ↔ `float16_t d`，`uint8_t qs[QK4_0 / 2]` ↔ `uint8_t qs[16]`，块大小都是 32。**两份定义、两个文件、两套 assert/宏** —— 这就是"新增一个量化类型要同时改 CPU 与所有 GPU 后端"的代码依据。下面这 6 行属于 L1-04 的覆盖范围，本课只为对照引用。

<!-- src: ggml/src/ggml-common.h -->
```c
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");
```

## 十、反量化片段：`dequant_funcs.glsl` 里的 Q4_0

被 `get_rows_quant.comp` 等着色器 include 的反量化片段。`dequantize()` 一次出 2 个值、`dequantize4()` 一次出 4 个值，两者都只是"取 nibble、减 8"，scale 由调用方在外面乘 —— 与 `dequant_q4_0.comp` 把 scale 乘进去的写法相比，分工不同、块布局完全一样。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl -->
```glsl
#if defined(DATA_A_Q4_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2(vui & 0xF, vui >> 4) - 8.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f);
}
#endif
```

## 十一、`mul_mm.comp` 的两套量化路径

`mul_mm.comp` 是矩阵乘的主文件。它有两种被编译的方式：打开 `MULMAT_QUANT` 时走**运行时类型开关**（`MmTypeA` 是个 specialization constant，第 36 行），否则走**编译期类型开关**（`DATA_A_*` 宏）。第 39 行的 `#include "types.glsl"` 是所有着色器共享量化块定义的入口。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp -->
```glsl
#ifdef MULMAT_QUANT
#include "ggml_type_ids.glsl"
layout (constant_id = 12) const uint MmTypeA = 0;
#endif

#include "types.glsl"
#include "dot_product_funcs.glsl"

#ifndef MULMAT_QUANT
#ifndef LOAD_VEC_A
#define LOAD_VEC_A 1
#endif
#endif
#ifndef LOAD_VEC_B
#define LOAD_VEC_B 1
#endif
```

## 十二、`flash_attn.comp`：Q 是 f32、K/V 是 f16

flash attention 的着色器把 Q 当 f32、K/V 当 f16 读（binding 0/1/2 的三组别名视图）。它在第 13-18 行按 `MMQ` 宏分成两个变体：MMQ 那条要求整数点积扩展，并把 `mul_mmq_shmem_types.glsl` 的共享内存块类型拉进来 —— 也就是**注意力与量化矩阵乘共用同一套 packed 块类型**。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp -->
```glsl
#ifdef MMQ
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_clustered : require

#include "mul_mmq_shmem_types.glsl"
#endif

#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_vote : enable

#include "types.glsl"
#include "dot_product_funcs.glsl"
#include "flash_attn_base.glsl"
#include "flash_attn_dequant.glsl"

const uint32_t HSK_per_thread = HSK / D_split;
const uint32_t HSV_per_thread = HSV / D_split;

const uint32_t rows_per_thread = Br / row_split;
const uint32_t cols_per_iter = WorkGroupSize / D_split / row_split;
const uint32_t cols_per_thread = Bc / cols_per_iter;
const uint32_t num_subgroups = SubGroupSize == 0 ? 0 : WorkGroupSize / SubGroupSize;


layout (binding = 0) readonly buffer Q {float data_q[];};
layout (binding = 0) readonly buffer QV4 {vec4 data_qv4[];};
layout (binding = 1) readonly buffer K {float16_t data_k[];};
layout (binding = 1) readonly buffer KV4 {f16vec4 data_kv4[];};
layout (binding = 2) readonly buffer V {float16_t data_v[];};
layout (binding = 2) readonly buffer VV4 {f16vec4 data_vv4[];};
layout (binding = 3) readonly buffer M {float16_t data_m[];};

// If SubGroupSize is set to 0 then only use shmem reductions
const uint32_t tmpsh_size = (SubGroupSize > 0) ? (row_split == 1 ? num_subgroups * D_split : num_subgroups) : WorkGroupSize;
shared float tmpsh[tmpsh_size];
shared FLOAT_TYPEV4 tmpshv4[tmpsh_size];
```

## 十三、`rms_norm.comp`：一个文件顶三种融合

`rms_norm.comp` 的 `main()` 不是简单调用一次：它按 `num_blocks` 把 `rms_norm(num_iters)` **实例化十次**，让每种的循环次数在编译期成为常量。同一份着色器还通过 `RMS_NORM_ROPE_FUSION` / `RMS_NORM_ADD_FUSION` / `RMS_NORM_SET_ROWS_FUSION` 承担三条融合路径 —— 这解释了生成器里`rms_norm_*` 那一长串变体名。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp -->
```glsl
void main() {
    // instantiate the rms_norm function for several different
    // dimensions, to allow loop unrolling
    uint num_blocks = (p.ne00 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 32) {
        rms_norm(num_blocks);
    } else if (num_blocks > 16) {
        rms_norm(32);
    } else if (num_blocks > 12) {
        rms_norm(16);
    } else if (num_blocks > 10) {
        rms_norm(12);
    } else if (num_blocks > 8) {
        rms_norm(10);
    } else if (num_blocks > 4) {
        rms_norm(8);
    } else if (num_blocks == 4) {
        rms_norm(4);
    } else if (num_blocks == 3) {
        rms_norm(3);
    } else if (num_blocks == 2) {
        rms_norm(2);
    } else if (num_blocks == 1) {
        rms_norm(1);
    }
}
```

## 十四、`soft_max.comp`：push constant 里的算子参数

softmax 的 `parameter` 块把 ggml 算子的标量参数整包搬进 push constant：形状（`ne00..ne13`）、步长（`nb11..nb13`）、`scale`、ALiBi 的 `max_bias`/`m0`/`m1`、以及 sink 个数。**这是 L1-01 说的 `op_params` 在 GPU 侧的落地方式**。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp -->
```glsl
void soft_max(uint num_iters) {
    const uint tid = gl_LocalInvocationID.x;
    const uint rowx = gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x;

    const uint32_t i03 = rowx / (p.ne01 * p.ne02);
    const uint32_t i02 = (rowx - i03 * p.ne01 * p.ne02) / p.ne01;
    const uint32_t i01 = rowx % p.ne01;

    uint rowy_start = 0;
    if (p.KY > 0) {
        rowy_start = i01 * p.nb11 + (i02 % p.ne12) * p.nb12 + (i03 % p.ne13) * p.nb13;
    }

    if (rowx >= p.nrows_x) {
        return;
    }
```

## 十五、`rope_norm.comp`：最小的 `.comp` 骨架

整个文件 17 行：`#version` → include 一个 head（push constant 与类型）→ include 一个 funcs（真正的算法）→ `void main()` 里把全局线程号拆成 (row, head, batch) 等索引，然后调用 `rope_norm(i0, i1, i2, i3, pc)`。**其余 160 个 `.comp` 都是这个骨架加更多分支。**

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/rope_norm.comp -->
```glsl
#version 450

#include "rope_head.glsl"
#include "rope_funcs.glsl"

void main() {
    const uint i0 = 2*gl_GlobalInvocationID.y;
    const uint row = gl_GlobalInvocationID.x + 32768 * gl_GlobalInvocationID.z;
    if (row >= pc.nrows) {
        return;
    }
    const uint i3 = row / (pc.ne01*pc.ne02);
    const uint i2 = (row - i3 * pc.ne01*pc.ne02) / pc.ne01;
    const uint i1 = (row - i3 * pc.ne01*pc.ne02 - i2 * pc.ne01);

    rope_norm(i0, i1, i2, i3, pc);
}
```

---

## 说明

- 本课声明计划里 L6-07 的**全部 162 个文件**（161 个 `.comp` + `vulkan-shaders-gen.cpp`），无一遗漏；计数命令见 `source.md` 第一节。
- 另有 4 个文件被本课引用：`ggml/src/ggml-vulkan/ggml-vulkan.cpp`（归 L6-06，本课用它的第 554-572 与 5377-5385 行说明"被 pipeline 使用"这一步）、`ggml/src/ggml-common.h`（归 L1-04，本课用第 194-199 行做 GLSL/C 两份块定义的对照）。这两个文件在覆盖域内，各自的主覆盖仍在原课。
- 还有 `types.glsl` 与 `dequant_funcs.glsl`（Vulkan 着色器的 `.glsl` 头文件）。`.glsl` 后缀**不在覆盖域**内（覆盖域只收 `.comp`），所以它们**不计入覆盖率**，仅作为论据逐字引用 —— 计入与不计入在这里是分开的账。
- 本课不讨论任何命令行参数；参数门禁的真值集来自真实二进制与本仓库门禁脚本的帮助输出。
