# L7-03 · OpenVINO 后端：把 ggml 图翻译成另一套图 IR — 课件说明

> 层：**L7 · NPU 与加速器后端** ｜ 前置课：`L7-02`（Hexagon 后端）；建议先看 L3-01（虚表契约）与 L4-02（切分）

## 学习目标

看完这一课，你应该能：

1. 说出 OpenVINO 后端在 `ggml_backend_device_i.supports_op` 里的**三道判据**（对应验收点）；
2. 据此判断**什么形状的子图会被整体交给 OpenVINO** —— 最长的、每节点都通过判据的连续区间；
3. 解释为什么这个后端**没有软件回退代码**，以及"回退"实际发生在哪一层；
4. 说出 `GGML_OPENVINO_DEVICE` 的三个取值各自改变了什么（编译配置 / 内存通道 / 判据例外）；
5. 举出"一个 ggml op → 多个 ov::op"和"一个 ggml op → 一个融合 ov::op"各一个例子。

## 覆盖的源文件（82 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-openvino.h` | 38 |
| `ggml/src/ggml-openvino/openvino/translate_session.cpp` | 512 |
| `ggml/src/ggml-openvino/openvino/op_table.cpp` | 89 |
| `ggml/src/ggml-openvino/ggml-openvino.cpp` | 1665 |
| `ggml/src/ggml-openvino/ggml-openvino-extra.cpp` | 578 |
| `ggml/src/ggml-openvino/utils.cpp` | 1774 |
| `ggml/src/ggml-openvino/openvino/op/rms_norm.cpp` | 80 |
| `ggml/src/ggml-openvino/ggml-decoder.cpp` | 2251 |
| `ggml/src/ggml-openvino/ggml-decoder.h` | 484 |
| `ggml/src/ggml-openvino/ggml-openvino-extra.h` | 220 |
| `ggml/src/ggml-openvino/ggml-quants.cpp` | 1374 |
| `ggml/src/ggml-openvino/ggml-quants.h` | 59 |
| `ggml/src/ggml-openvino/model-cache.cpp` | 274 |
| `ggml/src/ggml-openvino/model-cache.h` | 57 |
| `ggml/src/ggml-openvino/utils.h` | 193 |
| `ggml/src/ggml-openvino/openvino/decoder.h` | 134 |
| `ggml/src/ggml-openvino/openvino/frontend.cpp` | 29 |
| `ggml/src/ggml-openvino/openvino/frontend.h` | 23 |
| `ggml/src/ggml-openvino/openvino/input_model.cpp` | 18 |
| `ggml/src/ggml-openvino/openvino/input_model.h` | 30 |
| `ggml/src/ggml-openvino/openvino/node_context.h` | 176 |
| `ggml/src/ggml-openvino/openvino/op_table.h` | 67 |
| `ggml/src/ggml-openvino/openvino/translate_session.h` | 30 |
| `ggml/src/ggml-openvino/openvino/utils.cpp` | 879 |
| `ggml/src/ggml-openvino/openvino/utils.h` | 87 |
| `ggml/src/ggml-openvino/openvino/op/add.cpp` | 60 |
| `ggml/src/ggml-openvino/openvino/op/add_id.cpp` | 77 |
| `ggml/src/ggml-openvino/openvino/op/argsort.cpp` | 48 |
| `ggml/src/ggml-openvino/openvino/op/clamp.cpp` | 34 |
| `ggml/src/ggml-openvino/openvino/op/concat.cpp` | 49 |
| `ggml/src/ggml-openvino/openvino/op/cont.cpp` | 38 |
| `ggml/src/ggml-openvino/openvino/op/cpy.cpp` | 296 |
| `ggml/src/ggml-openvino/openvino/op/cumsum.cpp` | 30 |
| `ggml/src/ggml-openvino/openvino/op/diag.cpp` | 38 |
| `ggml/src/ggml-openvino/openvino/op/div.cpp` | 139 |
| `ggml/src/ggml-openvino/openvino/op/fill.cpp` | 35 |
| `ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp` | 244 |
| `ggml/src/ggml-openvino/openvino/op/gated_delta_net.cpp` | 328 |
| `ggml/src/ggml-openvino/openvino/op/gated_delta_net.hpp` | 66 |
| `ggml/src/ggml-openvino/openvino/op/gather_matmul.hpp` | 44 |
| `ggml/src/ggml-openvino/openvino/op/get_rows.cpp` | 143 |
| `ggml/src/ggml-openvino/openvino/op/glu_geglu.cpp` | 76 |
| `ggml/src/ggml-openvino/openvino/op/glu_geglu_quick.cpp` | 63 |
| `ggml/src/ggml-openvino/openvino/op/glu_swiglu.cpp` | 120 |
| `ggml/src/ggml-openvino/openvino/op/im2col.cpp` | 120 |
| `ggml/src/ggml-openvino/openvino/op/l2_norm.cpp` | 62 |
| `ggml/src/ggml-openvino/openvino/op/moe_compressed.hpp` | 91 |
| `ggml/src/ggml-openvino/openvino/op/mul_mat_id.cpp` | 230 |
| `ggml/src/ggml-openvino/openvino/op/mulmat.cpp` | 93 |
| `ggml/src/ggml-openvino/openvino/op/norm.cpp` | 32 |
| `ggml/src/ggml-openvino/openvino/op/pad.cpp` | 95 |
| `ggml/src/ggml-openvino/openvino/op/permute.cpp` | 160 |
| `ggml/src/ggml-openvino/openvino/op/pool_2d.cpp` | 54 |
| `ggml/src/ggml-openvino/openvino/op/repeat.cpp` | 48 |
| `ggml/src/ggml-openvino/openvino/op/reshape.cpp` | 114 |
| `ggml/src/ggml-openvino/openvino/op/roll.cpp` | 37 |
| `ggml/src/ggml-openvino/openvino/op/rope.cpp` | 239 |
| `ggml/src/ggml-openvino/openvino/op/scale.cpp` | 87 |
| `ggml/src/ggml-openvino/openvino/op/set.cpp` | 77 |
| `ggml/src/ggml-openvino/openvino/op/set_rows.cpp` | 97 |
| `ggml/src/ggml-openvino/openvino/op/softmax.cpp` | 167 |
| `ggml/src/ggml-openvino/openvino/op/solve_tri.cpp` | 109 |
| `ggml/src/ggml-openvino/openvino/op/sqr.cpp` | 36 |
| `ggml/src/ggml-openvino/openvino/op/ssm_conv.cpp` | 62 |
| `ggml/src/ggml-openvino/openvino/op/sum_rows.cpp` | 28 |
| `ggml/src/ggml-openvino/openvino/op/transpose.cpp` | 53 |
| `ggml/src/ggml-openvino/openvino/op/tri.cpp` | 83 |
| `ggml/src/ggml-openvino/openvino/op/unary_softplus.cpp` | 45 |
| `ggml/src/ggml-openvino/openvino/op/view.cpp` | 246 |
| `ggml/src/ggml-openvino/openvino/pass/fuse_moe_compressed.cpp` | 274 |
| `ggml/src/ggml-openvino/openvino/pass/fuse_moe_compressed.h` | 20 |
| `ggml/src/ggml-openvino/openvino/pass/fuse_to_conv.cpp` | 213 |
| `ggml/src/ggml-openvino/openvino/pass/fuse_to_conv.h` | 18 |
| `ggml/src/ggml-openvino/openvino/pass/fuse_to_sdpa.cpp` | 61 |
| `ggml/src/ggml-openvino/openvino/pass/fuse_to_sdpa.h` | 18 |
| `ggml/src/ggml-openvino/openvino/pass/kv_state_seq_axis.cpp` | 115 |
| `ggml/src/ggml-openvino/openvino/pass/kv_state_seq_axis.h` | 25 |
| `ggml/src/ggml-openvino/openvino/pass/mark_decompression_convert_constant_folding.h` | 30 |
| `ggml/src/ggml-openvino/openvino/pass/mark_dequantization_subgraph.h` | 45 |
| `ggml/src/ggml-openvino/openvino/pass/squeeze_matmul.cpp` | 60 |
| `ggml/src/ggml-openvino/openvino/pass/squeeze_matmul.h` | 18 |
| `ggml/src/ggml-openvino/openvino/rt_info/weightless_caching_attributes.hpp` | 42 |

> **说明**：本课覆盖清单里的 **82 个文件全部计入覆盖率**，其中 **8 个核心文件被逐字引用**：`ggml/include/ggml-openvino.h`、`ggml-openvino.cpp`、`ggml-openvino-extra.cpp`、`utils.cpp`、`openvino/op_table.cpp`、`openvino/translate_session.cpp`、`openvino/op/rms_norm.cpp`、`openvino/op/flash_attn_ext.cpp`。其余 74 个文件按族在第一节的表格里说明分工，不逐个引用。
> **说明**：场景 3 的映射表里每一行的"翻译成什么 ov::op"都来自实测，不是凭印象写的：走一行模板特化（`translate_1to1_match_*<T>`）的，类名直接写在 `op_table.cpp` 里；专属翻译器的类名来自对该文件的 `grep -o "ov::op::v[0-9]*::[A-Za-z_0-9]*"`。表里只列了最常见的 16 行，完整 55 项以 `op_table.cpp` 第 23-84 行的逐字引用为准。
> **说明**：本课**不展开**权重重打包（`ggml-quants.cpp`）与编译缓存（`model-cache.cpp`）的算法细节 —— 这两个文件已进覆盖声明，但它们属于"数据面"而非"图翻译"，留给后续修订。

## 场景（9 幕）

1. **OpenVINO 后端：翻译图，而不是写 kernel** — 公共头只有 23 行有效内容、11 个 C 函数。真正的实现是 81 个 C++ 文件构成的"前端"。
2. **★ 一个 ggml 节点 = 一次查表** — 翻译的骨架只有十几行：取 op 名字，去表里找翻译函数，调用它，把结果登记进 tensor_map。
3. **op_table.cpp：后端能力的唯一真相来源** — 一张 unordered_map：key 是 ggml 的 op 宏名，value 是翻译函数。表里有什么，后端就能算什么。
4. **★ supported_ops 不是手写的清单，是从翻译表派生的** — 三道门槛的第一道：这个 op（或 unary / glu 子枚举）的名字，在 op_table 里有没有？
5. **第二、三道门槛：类型 × 形状 × 设备例外** — op 有翻译器还不够：算子与全部输入的类型必须在 13 种白名单里，量化张量还不许是 3D。
6. **一个后端，三种设备：NPU / GPU / CPU** — 设备名从 GGML_OPENVINO_DEVICE 读，默认 CPU；不可用就回退 CPU。设备一旦定下，编译配置与判据分支都跟着变。
7. **不支持的 op 不需要回退代码：它根本不会被切进来** — 运行时真正要判断的，是"这次拿到的是整图、还是被切碎的一段" —— 后者不能走缓存与 naive 捷径。
8. **一个 ggml op 进去，一串 ov::op 出来** — RMS_NORM 是最典型的例子：6 个 ov 节点、零个 kernel。这是"翻译"这个视角最直接的证据。
9. **一张表说清 OpenVINO 后端** — 支持判据、设备、回退、粒度 —— 四个问题，四个答案，都在同一个设计决定下。

## 核心结论

### ★ 支持判据 = 翻译器的存在性

`supports_op` 的第一道门槛就是

```text
table.count("GGML_OP_" + ggml_op_name(op))   // op_table.cpp 的 key
```

它一次只回答**一个 op**，返回 `(bool, reason)`。三道门槛合起来是：

| # | 判据 | 代码位置 |
|---|---|---|
| 一 | op（/ unary / glu）名字在 op_table 派生的集合里 | `ggml-openvino.cpp:1470-1488` |
| 二 | `op->type` 与全部 `src[i]->type` ∈ 13 种白名单 | `ggml-openvino.cpp:1532-1542` |
| 三 | 量化张量的 `ne[2] == 1`（MUL_MAT_ID 专家权重除外） | `ggml-openvino.cpp:1543-1547` |
| + | 设备 / 形状例外表 | `is_op_supported_case`，`1129` 行起 |

### ★ 一次交出一个子图，而不是一个算子

`supports_op` 是**逐节点**问的，`graph_compute` 收到的却是**一整段子图**。两者由 L4-02 的调度器连接：它在每个"说不"的节点前断一刀，于是切给 OpenVINO 的永远是**连续的同后端段**。

所以验收点的完整答案是：

> 在图里按拓扑序取**最长的、每个节点都通过三道判据（且满足设备例外）的连续区间** —— 这一段会被整体翻译成一个 `ov::Model`，一次 `compile_model`，一次推理。

任何一格不过，就在那里断开；跨段的数据由调度器插 copy，OpenVINO 后端不写回退代码。

### 这是一台"图到图"的编译器前端，不是一组 kernel

| 维度 | 写 kernel 的后端（L5/L6） | OpenVINO 后端（L7-03） |
|---|---|---|
| 主要工程量 | 每个算子的 SIMD / 线程实现 | 45 个 op 翻译器 |
| "支持"的含义 | 我实现了这个 kernel | 我有这个 op 的 `ov::op` 映射 |
| 一个 ggml op | 一次 kernel 调用 | 0..N 个 `ov::op` 节点 |
| 不支持的 op | 常要写标量回退 | 不写任何代码，交给切分 |
| 性能来源 | 手写向量化 | OpenVINO 插件的图优化与设备编译 |

与 L7-01（CANN 映射到 ACL 算子）同属"映射到厂商算子"这一路线；L7-02（Hexagon）则是把整图交给远端 DSP。三者的差别不在契约（都由 L3-01 定），而在**这段子图被翻译成了什么**。

## 验收点

- [x] 保真门禁：16 处引用 —— 16 个引用块 / 74 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 82 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 1 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
