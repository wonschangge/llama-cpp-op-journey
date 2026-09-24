# L2-06 · ★ 计算图骨架 llama-graph — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-05`

## 学习目标

看完这一课，你应该能：

1. 说出 `llm_graph_context` 的成员各自由 `hparams` / `cparams` / `ubatch` 中哪一类决定；
2. 解释 `build_norm` 如何用同一个入口分派 RMSNorm 与 LayerNorm，以及两者的 eps 来自不同字段；
3. 说明 `build_attn` 里 KV cache 的写路径与读路径分别由谁提供；
4. **完整说出 `build_moe_ffn` 把一个 MoE 层展开成了哪些 ggml 算子**（本课验收点）；
5. 解释为什么"模型架构差异"在这套原语下只剩"调哪些原语 + 传什么超参"两个自由度。

## 覆盖的源文件（2 个）

| 文件 | 行数 |
|---|---|
| `src/llama-graph.cpp` | 3916 |
| `src/llama-graph.h` | 1391 |

> **说明**：本课只引用 `src/llama-graph.cpp` 与 `src/llama-graph.h`，两者均计入覆盖率。
> **说明**：文中提到的 `ggml_*` 算子名（`ggml_mul_mat_id` / `ggml_argsort_top_k` / `ggml_swiglu_split` 等）来自 `ggml/include/ggml.h`；该文件不在本课覆盖声明内，故不计入本课覆盖率。
> **说明**：`src/models/` 的统计数字（156 个文件 / 137 个出现原语调用 / 57 个调用 build_moe_ffn）由 grep 实测得到，不是估计值。余下 19 个文件（视觉塔 clip.cpp、BERT 系、RWKV/Mamba 系等）在自己的 build() 里直接调 ggml，不经过本课的原语。

## 场景（10 幕）

1. **llama-graph：模型的最后一公里** — 左边三个只读输入，右边两样产物：一处张量 arena（ctx0）和一张计算图（gf）。
2. **★ llm_graph_context：把超参摊平成常量的工作台** — 构造函数一行一个字段 —— 左边是成员，右边是来源。看右列就知道"谁决定了图的形状"。
3. **build_norm：一个 switch 分派三种归一化** — 选哪个算子由 llm_norm_type 决定；eps 从 hparams 来；权重与偏置是可选的第二步。
4. **build_ffn：三个矩阵乘 + 一个激活** — up / gate / down 三次矩阵乘都由 build_lora_mm 发出；激活由 type_op 分派，GLU 变体由 type_gate 决定。
5. **build_attn：先写进 KV cache，再整段读回来** — Q/K/V 投影不在这一层（由 build_qkv 做）；这一层负责记账：把本轮的 K/V 写进缓存，再连历史一起读出来。
6. **build_attn_mha：快路与慢路** — cparams.flash_attn 有值且没有 KQ bias 时，整个注意力塌缩成一个 ggml_flash_attn_ext 节点。
7. **build_moe_ffn 第一步：路由打分与门控** — MoE 的第一件事不是算专家，而是"每个 token 该去哪些专家" —— 一次矩阵乘 + 一个门控函数。
8. **build_moe_ffn 第二步：top-k 与权重归一** — 选出 n_expert_used 个专家，取出它们的权重，再决定要不要重新归一化。
9. **★ build_moe_ffn 展开成的 ggml 算子清单** — 这就是本课的验收点：一个 MoE 层 = 路由 1 次 + 门控 1 个 + top-k 1 个 + 三次 mul_mat_id + 聚合若干。
10. **★ 模型架构的差异 = 原语选择 + 超参** — 同一个 build_* 原语，靠枚举值与 hparams 分支出几十种架构 —— 这就是 156 个模型文件长得像的原因。

## 核心结论

### 六个原语族 = Transformer 的全部结构

| 原语 | 展开成什么 |
|---|---|
| `build_lora_mm` / `build_lora_mm_id` | `ggml_mul_mat` / `ggml_mul_mat_id` |
| `build_norm` | `ggml_norm` / `ggml_rms_norm` / `ggml_group_norm`，再 `ggml_mul`（+`ggml_add`） |
| `build_ffn` | up / gate / down 三次矩阵乘 + 激活（`ggml_swiglu_split` 等） |
| `build_moe_ffn` | 路由 + top-k + 三次 `ggml_mul_mat_id` + 加权聚合 |
| `build_attn` / `build_attn_mha` | KV cache 读写 + `ggml_flash_attn_ext`（或 4 节点慢路） |
| `build_inp_*` | `ggml_new_tensor` + `ggml_set_input` 的占位输入 |

### ★ build_moe_ffn 展开出的 ggml 算子（验收点）

```text
路由      ggml_mul_mat          (2025 调用点 / build_lora_mm 内 1518)
门控      ggml_soft_max (2043) | ggml_sigmoid (2047) | ggml_softplus (2055)
选专家    ggml_argsort_top_k                        (2109)
取权重    ggml_reshape_3d (2120) + ggml_get_rows    (2123)
归一化    ggml_sum_rows (2137) + ggml_clamp (2141) + ggml_div (2144)
专家并行  ggml_mul_mat_id x3   (up 2190 / gate 2203 / down 2304；在 build_lora_mm_id 内 1552)
激活      ggml_swiglu_split                         (2246)
加权      ggml_mul                                 (2321)
聚合      ggml_view_2d (2337) + ggml_add (2346)   [k = 1 时再加 ggml_cont (2353)]
```

### ★ 模型差异被压缩成"原语选择 + 超参"

实测：`src/models/` 下 156 个文件里，137 个出现 `build_norm` / `build_ffn` / `build_attn` / `build_moe_ffn` 的调用，57 个模型文件调用 `build_moe_ffn`。

原因在构造函数里就写明了：图的一切形状都由 `hparams`（结构）、`cparams`（运行参数）、`ubatch`（这一批的 token 数）三者决定，而原语把这些值翻译成算子。于是"新架构"往往只是换几个枚举值加几个超参，不需要新算子 —— 这正是 L2-10~L2-15 六课能覆盖全部模型家族的前提。

## 验收点

- [x] 保真门禁：18 处引用 —— 18 个引用块 / 67 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 2 项，无空课、无幻影；全局覆盖 844/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
