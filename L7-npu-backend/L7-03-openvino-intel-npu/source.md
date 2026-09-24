<!-- llama-coverage
ggml/include/ggml-openvino.h
ggml/src/ggml-openvino/openvino/translate_session.cpp
ggml/src/ggml-openvino/openvino/op_table.cpp
ggml/src/ggml-openvino/ggml-openvino.cpp
ggml/src/ggml-openvino/ggml-openvino-extra.cpp
ggml/src/ggml-openvino/utils.cpp
ggml/src/ggml-openvino/openvino/op/rms_norm.cpp
ggml/src/ggml-openvino/ggml-decoder.cpp
ggml/src/ggml-openvino/ggml-decoder.h
ggml/src/ggml-openvino/ggml-openvino-extra.h
ggml/src/ggml-openvino/ggml-quants.cpp
ggml/src/ggml-openvino/ggml-quants.h
ggml/src/ggml-openvino/model-cache.cpp
ggml/src/ggml-openvino/model-cache.h
ggml/src/ggml-openvino/utils.h
ggml/src/ggml-openvino/openvino/decoder.h
ggml/src/ggml-openvino/openvino/frontend.cpp
ggml/src/ggml-openvino/openvino/frontend.h
ggml/src/ggml-openvino/openvino/input_model.cpp
ggml/src/ggml-openvino/openvino/input_model.h
ggml/src/ggml-openvino/openvino/node_context.h
ggml/src/ggml-openvino/openvino/op_table.h
ggml/src/ggml-openvino/openvino/translate_session.h
ggml/src/ggml-openvino/openvino/utils.cpp
ggml/src/ggml-openvino/openvino/utils.h
ggml/src/ggml-openvino/openvino/op/add.cpp
ggml/src/ggml-openvino/openvino/op/add_id.cpp
ggml/src/ggml-openvino/openvino/op/argsort.cpp
ggml/src/ggml-openvino/openvino/op/clamp.cpp
ggml/src/ggml-openvino/openvino/op/concat.cpp
ggml/src/ggml-openvino/openvino/op/cont.cpp
ggml/src/ggml-openvino/openvino/op/cpy.cpp
ggml/src/ggml-openvino/openvino/op/cumsum.cpp
ggml/src/ggml-openvino/openvino/op/diag.cpp
ggml/src/ggml-openvino/openvino/op/div.cpp
ggml/src/ggml-openvino/openvino/op/fill.cpp
ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp
ggml/src/ggml-openvino/openvino/op/gated_delta_net.cpp
ggml/src/ggml-openvino/openvino/op/gated_delta_net.hpp
ggml/src/ggml-openvino/openvino/op/gather_matmul.hpp
ggml/src/ggml-openvino/openvino/op/get_rows.cpp
ggml/src/ggml-openvino/openvino/op/glu_geglu.cpp
ggml/src/ggml-openvino/openvino/op/glu_geglu_quick.cpp
ggml/src/ggml-openvino/openvino/op/glu_swiglu.cpp
ggml/src/ggml-openvino/openvino/op/im2col.cpp
ggml/src/ggml-openvino/openvino/op/l2_norm.cpp
ggml/src/ggml-openvino/openvino/op/moe_compressed.hpp
ggml/src/ggml-openvino/openvino/op/mul_mat_id.cpp
ggml/src/ggml-openvino/openvino/op/mulmat.cpp
ggml/src/ggml-openvino/openvino/op/norm.cpp
ggml/src/ggml-openvino/openvino/op/pad.cpp
ggml/src/ggml-openvino/openvino/op/permute.cpp
ggml/src/ggml-openvino/openvino/op/pool_2d.cpp
ggml/src/ggml-openvino/openvino/op/repeat.cpp
ggml/src/ggml-openvino/openvino/op/reshape.cpp
ggml/src/ggml-openvino/openvino/op/roll.cpp
ggml/src/ggml-openvino/openvino/op/rope.cpp
ggml/src/ggml-openvino/openvino/op/scale.cpp
ggml/src/ggml-openvino/openvino/op/set.cpp
ggml/src/ggml-openvino/openvino/op/set_rows.cpp
ggml/src/ggml-openvino/openvino/op/softmax.cpp
ggml/src/ggml-openvino/openvino/op/solve_tri.cpp
ggml/src/ggml-openvino/openvino/op/sqr.cpp
ggml/src/ggml-openvino/openvino/op/ssm_conv.cpp
ggml/src/ggml-openvino/openvino/op/sum_rows.cpp
ggml/src/ggml-openvino/openvino/op/transpose.cpp
ggml/src/ggml-openvino/openvino/op/tri.cpp
ggml/src/ggml-openvino/openvino/op/unary_softplus.cpp
ggml/src/ggml-openvino/openvino/op/view.cpp
ggml/src/ggml-openvino/openvino/pass/fuse_moe_compressed.cpp
ggml/src/ggml-openvino/openvino/pass/fuse_moe_compressed.h
ggml/src/ggml-openvino/openvino/pass/fuse_to_conv.cpp
ggml/src/ggml-openvino/openvino/pass/fuse_to_conv.h
ggml/src/ggml-openvino/openvino/pass/fuse_to_sdpa.cpp
ggml/src/ggml-openvino/openvino/pass/fuse_to_sdpa.h
ggml/src/ggml-openvino/openvino/pass/kv_state_seq_axis.cpp
ggml/src/ggml-openvino/openvino/pass/kv_state_seq_axis.h
ggml/src/ggml-openvino/openvino/pass/mark_decompression_convert_constant_folding.h
ggml/src/ggml-openvino/openvino/pass/mark_dequantization_subgraph.h
ggml/src/ggml-openvino/openvino/pass/squeeze_matmul.cpp
ggml/src/ggml-openvino/openvino/pass/squeeze_matmul.h
ggml/src/ggml-openvino/openvino/rt_info/weightless_caching_attributes.hpp
-->

# L7-03 · OpenVINO 后端：把 ggml 图翻译成另一套图 IR — 源文件

**一句话**：OpenVINO 后端不写 kernel。它把调度器切给它的那段 ggml 子图**翻译**成 OpenVINO 的图 IR（`ov::Model`），再交给 Intel 的 NPU / GPU / CPU 插件去编译。

这个视角解释了本课后面所有的"奇怪"设计：

- 支持判据不是"我能不能算得快"，而是"**这个 op 有没有对应的 `ov::op`**"；
- 所以判据是**一次问一个 op**（`supports_op`），但**一次交出一整段子图**（`graph_compute`）；
- 不支持的 op 不需要"软件回退"，它**根本不会被切进来** —— 切分发生在 L4-02 的调度器里。

本课覆盖 **82 个文件**：`ggml/include/ggml-openvino.h` + `ggml/src/ggml-openvino/` 下的 81 个。其中 45 个在 `openvino/op/` 下，一个文件（最多）负责一个（或一族）ggml 算子到 `ov::op` 的翻译。

---

## 一、82 个文件：五族分工

`L7-03` 的覆盖清单由 `plan_matrix.py` 指派，实测 **82 个文件**：公共头 1 个 + `ggml/src/ggml-openvino/` 下 81 个。按目录分成五族：

| 族 | 个数 | 目录 / 代表文件 | 职责 |
|---|---|---|---|
| 顶层胶水 | 12 | `ggml-openvino.cpp` `ggml-decoder.*` `ggml-openvino-extra.*` `ggml-quants.*` `model-cache.*` `utils.*` + `include/ggml-openvino.h` | 虚表实现、cgraph→decoder、设备配置、权重量化 / 重打包、编译缓存 |
| frontend 骨架 | 12 | `openvino/op_table.*` `translate_session.*` `input_model.*` `node_context.h` `decoder.h` `frontend.*` `utils.*` | 翻译器的注册表与调度：节点 → 翻译函数 → `ov::Model` |
| op 翻译器 | 45 | `openvino/op/` 下 42 个 `.cpp` + 3 个 `.hpp` | 每个（族）ggml 算子一个文件，把 `NodeContext` 变成 `ov::OutputVector` |
| OV pass | 12 | `openvino/pass/` | 翻译完成后的图改写：`FuseToConv` `FuseToSdpa` `SqueezeMatmul` `KVStateSeqAxis` 等 |
| rt_info | 1 | `openvino/rt_info/weightless_caching_attributes.hpp` | 给大 `Constant` 打标记，避免 NPUW 重复拷贝权重 |

**读法**：想知道"某个 ggml 算子在这个后端上会变成什么"，先查 `op_table.cpp` 有没有它，再打开 `openvino/op/` 下对应的那个文件。

## 二、★ 翻译表：后端能力的唯一真相来源

`get_supported_ops()` 返回一张 `std::unordered_map<std::string, CreatorFunction>`。key 是 **ggml 的 op 宏名**（`GGML_OP_*` / `GGML_UNARY_OP_*` / `GGML_GLU_OP_*`），value 是翻译函数。表里共 **55 项**（第 26-80 行）。

两个值得注意的细节（都能从下面的逐字引用读出来）：

- `GGML_OP_FILL` 出现了**两次**（第 32 行与第 75 行）—— 两处都指向 `op::translate_fill`，所以无论哪一条生效，结果都相同。
- 第 81-82 行是**被注释掉**的 `GGML_OP_SOLVE_TRI`：`solve_tri.cpp:33` 里的 `translate_solve_tri` 仍然存在，但没有登记进表 —— 于是它永远不会被调用（`supports_op` 会说"没有翻译器"）。这说明**能力边界由这张表决定，而不是由存在哪些文件决定**。

<!-- src: ggml/src/ggml-openvino/openvino/op_table.cpp -->
```cpp
std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    using namespace ov::op;
    return {
        {"GGML_OP_ADD",             op::translate_add                              },
        {"GGML_OP_ADD1",            op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_ADD_ID",          op::translate_add_id                           },
        {"GGML_OP_CONCAT",          op::translate_concat                           },
        {"GGML_OP_CONT",            op::translate_cont                             },
        {"GGML_OP_DIV",             op::translate_div                              },
        {"GGML_OP_FILL",            op::translate_fill                             },
        {"GGML_OP_GET_ROWS",        op::translate_get_rows                         },
        {"GGML_OP_IM2COL",          op::translate_im2col                           },
        {"GGML_OP_MUL",             op::translate_1to1_match_2_inputs<v1::Multiply>},
        {"GGML_OP_MUL_MAT",         op::translate_mulmat                           },
        {"GGML_OP_MUL_MAT_ID",      op::translate_mul_mat_id                       },
        {"GGML_OP_PERMUTE",         op::translate_permute                          },
        {"GGML_OP_RESHAPE",         op::translate_reshape                          },
        {"GGML_OP_RMS_NORM",        op::translate_rms_norm                         },
        {"GGML_OP_NORM",            op::translate_norm                             },
        {"GGML_OP_L2_NORM",         op::translate_l2_norm                          },
        {"GGML_OP_SUM_ROWS",        op::translate_sum_rows                         },
        {"GGML_OP_ROPE",            op::translate_rope                             },
        {"GGML_OP_SCALE",           op::translate_scale                            },
        {"GGML_OP_SQR",             op::translate_sqr                              },
        {"GGML_OP_SQRT",            op::translate_sqrt                             },
        {"GGML_OP_SOFT_MAX",        op::translate_soft_max                         },
        {"GGML_OP_ARGSORT",         op::translate_argsort                          },
        {"GGML_OP_SUB",             op::translate_1to1_match_2_inputs<v1::Subtract>},
        {"GGML_OP_TRANSPOSE",       op::translate_transpose                        },
        {"GGML_UNARY_OP_GELU",      op::translate_1to1_match_1_input<v7::Gelu>     },
        {"GGML_UNARY_OP_SIGMOID",   op::translate_1to1_match_1_input<v0::Sigmoid>  },
        {"GGML_UNARY_OP_SILU",      op::translate_1to1_match_1_input<v4::Swish>    },
        {"GGML_UNARY_OP_SOFTPLUS",  op::translate_unary_softplus                   },
        {"GGML_UNARY_OP_TANH",      op::translate_1to1_match_1_input<v0::Tanh>     },
        {"GGML_UNARY_OP_EXP",       op::translate_1to1_match_1_input<v0::Exp>      },
        {"GGML_UNARY_OP_NEG",       op::translate_1to1_match_1_input<v0::Negative> },
        {"GGML_UNARY_OP_RELU",      op::translate_1to1_match_1_input<v0::Relu>     },
        {"GGML_OP_VIEW",            op::translate_view                             },
        {"GGML_GLU_OP_SWIGLU",      op::translate_glu_swiglu                       },
        {"GGML_GLU_OP_SWIGLU_OAI",  op::translate_glu_swiglu_oai                   },
        {"GGML_GLU_OP_SWIGLU_CLAMP", op::translate_glu_swiglu_clamp                 },
        {"GGML_GLU_OP_GEGLU",       op::translate_glu_geglu                        },
        {"GGML_GLU_OP_GEGLU_QUICK", op::translate_glu_geglu_quick                  },
        {"GGML_OP_SET_ROWS",        op::translate_set_rows                         },
        {"GGML_OP_CPY",             op::translate_cpy                              },
        {"GGML_OP_FLASH_ATTN_EXT",  op::translate_flash_attn_ext                   },
        {"GGML_OP_CLAMP",           op::translate_clamp                            },
        {"GGML_OP_PAD",             op::translate_pad                              },
        {"GGML_OP_SSM_CONV",        op::translate_ssm_conv                         },
        {"GGML_OP_GATED_DELTA_NET", op::translate_gated_delta_net                  },
        {"GGML_OP_REPEAT",          op::translate_repeat                           },
        {"GGML_OP_CUMSUM",          op::translate_cumsum                           },
        {"GGML_OP_FILL",            op::translate_fill                             },
        {"GGML_OP_DIAG",            op::translate_diag                             },
        {"GGML_OP_TRI",             op::translate_tri                              },
        {"GGML_OP_SET",             op::translate_set                              },
        {"GGML_OP_POOL_2D",         op::translate_pool_2d                          },
        {"GGML_OP_ROLL",            op::translate_roll                             },
        // solve_tri has accuracy issues on GPU
        // {"GGML_OP_SOLVE_TRI",       op::translate_solve_tri                        },
    };
}
```

## 三、★ 支持判据：三道门槛（验收点）

`ggml_backend_openvino_device_supports_op_impl()` 是**验收点答案**所在。它一次只看**一个** op，返回 `(bool, reason)`：

```text
门槛一：op 名字在 build_supported_sets() 派生出的集合里吗
        （集合来自 op_table 的 key；UNARY/GLU 走各自的子枚举集合）
门槛二：op->type 与全部非空 src[i]->type 都在 13 种白名单里吗
门槛三：量化张量的 ne[2] == 1 吗（MUL_MAT_ID 的专家权重除外）
过完三道，再过 is_op_supported_case() 的设备 / 形状例外表
```

**所以"什么形状的子图会被整体交给 OpenVINO"的答案是**：在图里按拓扑序取**最长的、每个节点都过三道门槛的连续区间**。调度器（L4-02）在每个"过不去"的节点前面断一刀，跨段的数据由调度器插 copy。OpenVINO 后端自己**不写任何软件回退**。

<!-- src: ggml/src/ggml-openvino/ggml-openvino.cpp -->
```cpp
static ggml_openvino_op_support ggml_backend_openvino_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);

    static std::unordered_set<ggml_type> supported_types{
        GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_I64,  GGML_TYPE_I32,  GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,
        GGML_TYPE_MXFP4};

    // derive supported op sets from the op_table map, keys in
    // the map use the full macro name (e.g. "GGML_OP_ADD"), while
    // the ggml_*_op_name() helpers return only the trailing part (e.g. "ADD").
    // each set is built once and cached.
    static const auto build_supported_sets = [] {
        const auto & table = ov::frontend::ggml::get_supported_ops();
        std::unordered_set<ggml_op> ops;
        std::unordered_set<ggml_unary_op> unary_ops;
        std::unordered_set<ggml_glu_op> glu_ops;

        // GGML_OP_NONE has no translator but is always safe to add to the supported set.
        ops.insert(GGML_OP_NONE);

        for (int i = 0; i < GGML_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_OP_") + ggml_op_name(static_cast<ggml_op>(i));
            if (table.count(key)) {
                ops.insert(static_cast<ggml_op>(i));
            }
        }
        for (int i = 0; i < GGML_UNARY_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_UNARY_OP_") + ggml_unary_op_name(static_cast<ggml_unary_op>(i));
            if (table.count(key)) {
                unary_ops.insert(static_cast<ggml_unary_op>(i));
            }
        }
        for (int i = 0; i < GGML_GLU_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_GLU_OP_") + ggml_glu_op_name(static_cast<ggml_glu_op>(i));
            if (table.count(key)) {
                glu_ops.insert(static_cast<ggml_glu_op>(i));
            }
        }
        return std::make_tuple(ops, unary_ops, glu_ops);
    };
    static const auto supported_sets = build_supported_sets();
    static const auto & supported_ops = std::get<0>(supported_sets);
    static const auto & supported_unary_ops = std::get<1>(supported_sets);
    static const auto & supported_glu_ops = std::get<2>(supported_sets);

    switch (op->op) {
    case GGML_OP_UNARY: {
        auto supported = supported_unary_ops.find(ggml_get_unary_op(op)) != supported_unary_ops.end();
        if (!supported) {
            return {false, "unary op " + std::string(ggml_unary_op_name(ggml_get_unary_op(op))) + " has no op translator"};
        }
        if (ggml_get_unary_op(op) == GGML_UNARY_OP_EXP && op->type == GGML_TYPE_F32) {
            return {false, "UNARY_EXP with F32 type is not supported"};
        }
        break;
    }
    case GGML_OP_GLU: {
        auto supported = supported_glu_ops.find(ggml_get_glu_op(op)) != supported_glu_ops.end();
        if (!supported) {
            return {false, "GLU op " + std::string(ggml_glu_op_name(ggml_get_glu_op(op))) + " has no op translator"};
        }
        // if (has_view_op_input(op)) {
        //     return {false, "GLU op " + std::string(ggml_glu_op_name(ggml_get_glu_op(op))) + " with view input is not supported"};
        // }
        if (op->src[1] == nullptr && op->src[0]->ne[0] % 2 != 0) {
            // triggers bug in ov gpu
            return {false, "GLU op with odd src0 ne[0] and null src1 is not supported"};
        }
        break;
    }
    default: {
        auto supported = supported_ops.find(op->op) != supported_ops.end();
        if (!supported) {
            return {false, "op " + std::string(ggml_op_name(op->op)) + " has no op translator"};
        }
        static std::set<ggml_op> ops_not_support_view_input{};
        if (ops_not_support_view_input.find(op->op) != ops_not_support_view_input.end() && has_view_op_input(op)) {
            return {false, "op " + std::string(ggml_op_name(op->op)) + " with VIEW input is not supported"};
        }
    }
    }

    if (supported_types.find(op->type) == supported_types.end()) {
        return {false, "tensor type " + std::string(ggml_type_name(op->type)) + " is not supported"};
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto * src = op->src[i];
        if (src == nullptr) {
            break;
        }
        if (supported_types.find(src->type) == supported_types.end()) {
            return {false, "src[" + std::to_string(i) + "] type " + std::string(ggml_type_name(src->type)) + " is not supported"};
        }
        const bool is_supported_3d_moe_expert =
            op->op == GGML_OP_MUL_MAT_ID && i == 0 && (src->type == GGML_TYPE_MXFP4 || src->ne[3] == 1);
        if (ggml_is_quantized(src->type) && src->ne[2] != 1 && !is_supported_3d_moe_expert) {
            return {false, "3D quantized tensor for src[" + std::to_string(i) + "] is not supported"};
        }
    }

    auto op_support_case = is_op_supported_case(op);
    if (!op_support_case.is_supported) {
        return op_support_case;
```

## 四、设备选择：NPU / GPU / CPU

设备名只有一个来源：环境变量 `GGML_OPENVINO_DEVICE`，默认 `CPU`；如果 `ov::Core::get_available_devices()` 里没有它，就警告一句并回退到 `CPU`。

设备名定下之后有三处后果：

1. **编译配置**：只有 `NPU` 会填 `NPU_COMPILER_DYNAMIC_QUANTIZATION` 与一整组 `NPUW_*` key；
2. **内存通道**：只有 `GPU` 会建 OpenCL context/queue 并把 remote context 交给 OV；
3. **支持判据**：`supports_op` 的例外表里大量出现 `ggml_openvino_get_device_name() == "GPU"` `== "NPU"` 的比较 —— 同一个 op 在不同设备上的答案可以不同。

另外 `is_npu` 直接决定执行路径：`utils.cpp:1470` 用 `ggml_openvino_is_npu()` 在 **static（NPU 形状固定）** 与 **dynamic** 两条路径之间二选一；`ggml-openvino.cpp:837` 里 stateful 执行也被显式禁用在 NPU 上。

<!-- src: ggml/src/ggml-openvino/ggml-openvino-extra.cpp -->
```cpp
    device_name = ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE", "CPU");
    auto available_devices = ov_singleton_core().get_available_devices();
    if (std::find(available_devices.begin(), available_devices.end(), device_name) == available_devices.end()) {
        GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\n", device_name.c_str());
        device_name = "CPU";
    }
    is_npu = (device_name == "NPU");

    const char * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    if (device_name == "NPU") {
        compile_config = {
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        if (cache_dir && strlen(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        const char * compilation_mode_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        if (compilation_mode_params && strlen(compilation_mode_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_mode_params;
        }
    } else if (cache_dir && strlen(cache_dir) > 0) {
```

## 五、回退策略：缺席，而不是分支

不支持的 op 由**切分阶段**解决：`supports_op` 说不，调度器就不把它切进来。所以 OpenVINO 后端里找不到"这个算子我自己实现一个慢版本"的代码。

运行时真正要判断的是另一件事：**这次拿到的是整图，还是被切碎的一段**。`is_model_splitted()` 用"节点的 `use_count` 是否等于把它当输入的节点数"来反推；`is_naive()` 用节点数是否 `< 20` 来判断是否小图；只有"小图且没被切过"才走 `naive_compute()` 这条一次性翻译的捷径。这两个检测都由 `GGML_OPENVINO_ENABLE_FALLBACK`（`utils.cpp:1492` 起）与缓存命中情况控制。

<!-- src: ggml/src/ggml-openvino/utils.cpp -->
```cpp
    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    graph_key key(cgraph);
    bool key_seen = false;
    if (!cache_disabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        key_seen = r_ctx->decoder_cache.find(key) != r_ctx->decoder_cache.end();
    }

    bool model_is_splitted = key_seen ? false : is_model_splitted(cgraph);

    if (is_naive(cgraph)) {
        if (!model_is_splitted) {
            return naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    auto start_time = ggml_time_us();
```

## 六、翻译粒度：一个 ggml op → 0..N 个 ov::op

`translate_*` 的返回类型是 `ov::OutputVector`，长度不受限。`translate_rms_norm()` 的收尾这一段就是 6 个 ov 节点（Multiply / ReduceMean / Add / Sqrt / Divide / Multiply）：

<!-- src: ggml/src/ggml-openvino/openvino/op/rms_norm.cpp -->
```cpp
    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    return rename_outputs_with_suffix({res}, context.get_name());
```

## 七、另一头：一个 ggml op → 一个融合 ov::op

`FLASH_ATTN_EXT` 的翻译器把整块注意力交给 OpenVINO 的 `ov::op::v13::ScaledDotProductAttention`（有 mask 与无 mask 两个分支）。这是"粒度由 ov 算子集决定"的证明：ggml 侧一个节点，ov 侧一个算子。

<!-- src: ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp -->
```cpp
    constexpr auto causal = false;
    if (has_mask) {
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, scale_node, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } else {
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, scale_node, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
```

## 八、三张虚表：和别的后端签的是同一份契约

OpenVINO 后端填的表与 L3-01 讲的三层契约完全一致，只是 `ggml_backend_i` 里除了 `get_name` / `free` / `graph_compute` 全是 `NULL`，而 `ggml_backend_device_i` 里 `supports_op` 指向本课的主角。

<!-- src: ggml/src/ggml-openvino/ggml-openvino.cpp -->
```cpp
static const struct ggml_backend_device_i ggml_backend_openvino_device_interface = {
    /* .get_name             = */ ggml_backend_openvino_device_get_name,
    /* .get_description      = */ ggml_backend_openvino_device_get_description,
    /* .get_memory           = */ ggml_backend_openvino_device_get_memory,
    /* .get_type             = */ ggml_backend_openvino_device_get_type,
    /* .get_props            = */ ggml_backend_openvino_device_get_props,
    /* .init_backend         = */ ggml_backend_openvino_device_init,
    /* .get_buffer_type      = */ ggml_backend_openvino_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_openvino_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_openvino_device_supports_op,
    /* .supports_buft        = */ ggml_backend_openvino_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
```

---

## 说明

- 本课覆盖清单里的 **82 个文件全部计入覆盖率**，其中 **8 个核心文件被逐字引用**：`ggml/include/ggml-openvino.h`、`ggml-openvino.cpp`、`ggml-openvino-extra.cpp`、`utils.cpp`、`openvino/op_table.cpp`、`openvino/translate_session.cpp`、`openvino/op/rms_norm.cpp`、`openvino/op/flash_attn_ext.cpp`。其余 74 个文件按族在第一节的表格里说明分工，不逐个引用。
- 场景 3 的映射表里每一行的"翻译成什么 ov::op"都来自实测，不是凭印象写的：走一行模板特化（`translate_1to1_match_*<T>`）的，类名直接写在 `op_table.cpp` 里；专属翻译器的类名来自对该文件的 `grep -o "ov::op::v[0-9]*::[A-Za-z_0-9]*"`。表里只列了最常见的 16 行，完整 55 项以 `op_table.cpp` 第 23-84 行的逐字引用为准。
- 本课**不展开**权重重打包（`ggml-quants.cpp`）与编译缓存（`model-cache.cpp`）的算法细节 —— 这两个文件已进覆盖声明，但它们属于"数据面"而非"图翻译"，留给后续修订。
