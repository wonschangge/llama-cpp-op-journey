# L6-08 · Metal 后端：主机端与图融合 — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-07`

## 学习目标

看完这一课，你应该能：

1. 说出融合在图上被触发的**两个时刻**、各自用的 `ggml_metal_fusion_mode`，以及为什么两个时刻的检查强度不同（对应验收点）；
2. 说清"由谁决定"：后端维护的 `ggml_metal_fusions` 表（26 条表项 / 9 种融合）、模式自有的 `check` 回调，以及来自 ggml 层的通用可消除判据 `ggml_can_fuse_subgraph_ext`；
3. 举出至少四种可融合 op 组合，并说出它们各自落到哪个 kernel（如 `NORM + MUL + ADD` 落到 `kernel_norm_mul_add_f32`）；
4. 解释为什么融合对前端与其它后端不可见（图从未被修改，优化阶段的打包会立刻解包），并说出 CUDA 融合路径与本课的差别。

## 覆盖的源文件（16 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-metal/ggml-metal-fusion.h` | 99 |
| `ggml/src/ggml-metal/ggml-metal.cpp` | 1054 |
| `ggml/src/ggml-metal/ggml-metal-common.cpp` | 501 |
| `ggml/src/ggml-metal/ggml-metal-context.m` | 788 |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 5758 |
| `ggml/src/ggml-metal/ggml-metal-fusion.cpp` | 1120 |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | 2577 |
| `ggml/src/ggml-metal/ggml-metal-common.h` | 58 |
| `ggml/src/ggml-metal/ggml-metal-context.h` | 43 |
| `ggml/src/ggml-metal/ggml-metal-ops.h` | 111 |
| `ggml/src/ggml-metal/ggml-metal-device.h` | 367 |
| `ggml/src/ggml-metal/ggml-metal-device.m` | 2505 |
| `ggml/src/ggml-metal/ggml-metal-impl.h` | 1383 |
| `ggml/src/ggml-metal/ggml-metal-tuning.h` | 76 |
| `ggml/src/ggml-metal/ggml-metal-tuning.cpp` | 1173 |
| `ggml/include/ggml-metal.h` | 62 |

> **说明**：本课覆盖 `ggml/src/ggml-metal/` 下的 15 个主机端源文件与 `ggml/include/ggml-metal.h`，共 16 个，全部计入覆盖率。
> **说明**：本课**不引用**任何 `.metal` 内核源文件（`ggml/src/ggml-metal/kernels/` 目录）—— 它们属于 L6-09 的覆盖域。课件里出现的 kernel 名（如 `kernel_norm_mul_add_f32`）全部来自本课引用的 `ggml-metal-device.cpp`，逐字可查。
> **说明**：`GGML_METAL_FUSION_DISABLE` 与 `GGML_METAL_FUSION_DEBUG` 是**环境变量**，不是命令行参数；出处是 `ggml-metal-device.m:1281-1286` 的 `getenv()` 调用。
> **说明**：源码注释不总是准确的：`ggml-metal-common.cpp:456` 的注释把文件名写成 `ggml-metal-fuse.cpp`，而仓库里实际的文件是 `ggml-metal-fusion.cpp`。本课第 4 幕按行号逐字引用了这一行（保留原样），此处特别说明。
> **说明**：与 CUDA 的对照只给行号指路（`ggml-cuda.cu:3180` 的 `ggml_cuda_can_fuse`、`:4505` 的 `ggml_backend_cuda_graph_optimize`），未逐字引用 CUDA 源码，因此 `ggml-cuda` 目录不计入本课覆盖率，留给 L6-01。

## 场景（9 幕）

1. **Metal 的融合：一张表，两个阶段** — 开头的这 6 行注释就是本课的全部内容：融合模式只声明一次，优化阶段和编码阶段读同一张表。
2. **★ 融合只存在于 "图已建好、还没编码" 的窗口里** — 后端拿到 ggml_cgraph 之后、把节点编码成 Metal 命令之前 —— 这就是融合能存在的全部空间。
3. **同一张表，被问两次：STRUCTURAL 与 FULL** — 模式枚举解释了"何时"：优化阶段张量还没分配，只能做结构判定；编码阶段才能做完整判定。
4. **先按表打包，再重排 —— 否则重排会把融合打断** — 调度器切分完图、做 graph_copy 之前，会对每个后端调用一次 graph_optimize 钩子。
5. **编码循环：idx += res - 1 就是"吞掉"发生的地方** — ggml_metal_op_encode() 返回它消费了几个节点；调用者据此跳过被融合的节点。
6. **先过并发判定，才有资格谈融合** — 只有在"这个节点能安全地和前面已编码的节点并发"时，编码器才会去查融合表。
7. **决定权在这张 26 条表项 / 9 种融合 的表里** — ggml_metal_fusions 就是"什么能融合"的全部答案。数一数：26 条表项，归到 9 个融合 id。
8. **★ 命中的真判据：连续、同形、可消除，再加后端私有约束** — 表只声明"模式长什么样"；能不能真的融合，由这几道检查依次裁决。
9. **融合的最后一公里：fusion id 决定取哪个 pipeline** — 命中之后，编码器告诉 device "我要融合 N 跳"，device 按名字取（或编译）对应的 kernel。

## 核心结论

### 融合的两个触发点

| 阶段 | 调用链 | 模式 | 此时张量 |
|---|---|---|---|
| 优化阶段 | 调度器 -> `.graph_optimize` -> `ggml_metal_graph_optimize()` -> `ggml_graph_optimize()` -> `ggml_metal_fusion_max()` | `GGML_METAL_FUSION_STRUCTURAL` | 未分配 |
| 编码阶段 | `ggml_metal_graph_compute()` -> `ggml_metal_op_encode()` -> `encode_impl()` -> `ctx->can_fuse()` | `GGML_METAL_FUSION_FULL` | 已分配 |

编码阶段的多余检查是"外部源与融合输出是否内存重叠"（`ggml_metal_fusion_check_memory_ranges`），以及各模式 `check` 里的 `->data` 判空。

### ★ 决定权在一张声明式的表

`ggml_metal_fusions` 声明了 26 条表项，归到 9 个 `ggml_metal_fusion_id`。表项结构是 `{ id, ops_all, outs, unsafe, check }`；`unsafe` 表示跳过通用判据、由 `check` 独家把关。两个阶段读同一张表，所以注释敢写"the two phases can never disagree"。

**通用判据在 ggml（`ggml_can_fuse_subgraph_ext`，检查中间节点能否被消除），模式与决策在后端（Metal 的表）。**

### 可融合组合与落点（9 种）

| 融合 id | op 序列 | kernel |
|---|---|---|
| `NORM_MUL` | NORM / RMS_NORM + MUL | `kernel_norm_mul_f32` / `kernel_rms_norm_mul_f32` |
| `NORM_MUL_ADD` | NORM / RMS_NORM + MUL + ADD | `kernel_norm_mul_add_f32` / `kernel_rms_norm_mul_add_f32` |
| `NORM_SCALE` | NORM / RMS_NORM + SCALE | `kernel_*_norm_mul_f32_use_scale` |
| `ADD_CHAIN` | ADD x N，N 在 [2, 7] | `kernel_bin_fuse_<t0>_<t1>_<t>`，名字带 `nf=N` |
| `SNAKE` | MUL + SIN + SQR + MUL + ADD | `kernel_snake_<type>` |
| `GDN_CACHE` | GATED_DELTA_NET + CPY（写穿） | 复用 `kernel_gated_delta_net_<type>_<nsg>` |
| `TOPK_MOE` | SOFT_MAX + ARGSORT + GET_ROWS（+ norm / + SCALE） | `kernel_topk_moe_f32` |
| `MOE_REDUCE` | MUL + VIEW x k + ADD x k，2 到 8 专家 | `kernel_moe_reduce_f32` |
| `SSM_CONV_SILU` | SSM_CONV + UNARY(silu) | `kernel_ssm_conv_<t0>_<t1>`，名字带 `silu=1` |

NORM 与 `bin` 两类在 `ne[0] % 4 == 0` 时会加 `_4` 后缀（float4 版本）。

### 融合不改图，也不产生新的 ggml op

优化阶段把可融合节点临时打包，重排一结束就立刻解包（`// unfuse`）。编码阶段只是"少走几步编码"，图上的节点一个都没少。

特别注意：**本版本里没有 `GGML_OP_MUL_ADD` 这样的新枚举**。融合的"名字"只出现在三处：Metal 私有的 `ggml_metal_fusion_id`、调试统计用的标签（由 `ggml_op_name()` 用 `+` 拼出来）、以及最终那个 kernel 字符串。

## 验收点

- [x] 保真门禁：21 处引用 —— 21 个引用块 / 39 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 16 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
