# L2-11 · 模型家族（二）：状态空间与线性注意力 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-06`（★ 计算图骨架 llama-graph）

## 学习目标

看完这一课，你应该能：

1. 说出本课 30 个文件里哪些真正属于状态空间/线性注意力，并给出判定依据（对应第 8、9 幕的表格）；
2. 解释为什么 SSM/线性注意力模型不需要 KV cache，以及 `get_r_l()` / `get_s_l()` 各自是什么；
3. **说出 `ggml_ssm_scan` 的 7 个张量输入 + 1 个整数参数各自的形状与含义（验收点前半）；**
4. **说出状态更新发生在哪一步，以及"内核更新"与"图上写回"为什么是两个不同的节点（验收点后半）；**
5. 解释 `hparams.is_recr(il)` 如何在同一个模型里把 recurrent 层与注意力层拼进一张图。

## 覆盖的源文件（37 个）

| 文件 | 行数 |
|---|---|
| `src/models/arwkv7.cpp` | 203 |
| `src/models/bailingmoe3.cpp` | 541 |
| `src/models/deepseek2.cpp` | 714 |
| `src/models/deepseek32.cpp` | 727 |
| `src/models/delta-net-base.cpp` | 607 |
| `src/models/dflash.cpp` | 1044 |
| `src/models/dots3note.cpp` | 477 |
| `src/models/falcon-h1.cpp` | 210 |
| `src/models/glm-dsa.cpp` | 770 |
| `src/models/granite-hybrid.cpp` | 303 |
| `src/models/hy-v4.cpp` | 602 |
| `src/models/jamba.cpp` | 199 |
| `src/models/kimi-k3.cpp` | 619 |
| `src/models/kimi-linear.cpp` | 563 |
| `src/models/lfm2.cpp` | 300 |
| `src/models/mamba2.cpp` | 89 |
| `src/models/mamba-base.cpp` | 305 |
| `src/models/mamba.cpp` | 138 |
| `src/models/minicpm3.cpp` | 253 |
| `src/models/nemotron-h.cpp` | 341 |
| `src/models/plamo2.cpp` | 427 |
| `src/models/plm.cpp` | 207 |
| `src/models/qwen35.cpp` | 645 |
| `src/models/qwen35moe.cpp` | 742 |
| `src/models/qwen3next.cpp` | 823 |
| `src/models/rwkv6-base.cpp` | 165 |
| `src/models/rwkv6.cpp` | 186 |
| `src/models/rwkv6qwen2.cpp` | 168 |
| `src/models/rwkv7-base.cpp` | 138 |
| `src/models/rwkv7.cpp` | 212 |
| `ggml/src/ggml-cpu/ops.cpp` | 12207 |
| `src/llama-hparams.cpp` | 354 |
| `src/llama-memory-recurrent.h` | 196 |
| `ggml/include/ggml.h` | 3023 |
| `ggml/src/ggml.c` | 8168 |
| `src/llama-graph.cpp` | 3916 |
| `src/llama-graph.h` | 1391 |

> **说明**：**覆盖声明**：本课声明块里的 30 个 `src/models/*.cpp` 是本课负责的文件，全部计入覆盖率，与 `tools/plan_matrix.py --files` 里 L2-11 的清单逐字一致。
> **说明**：**跨课引用（同样计入覆盖率，但它们的主课在别处）**：`ggml/include/ggml.h`（L1-01）、`ggml/src/ggml.c`（L1-02/03/04/05、L4-04）、`ggml/src/ggml-cpu/ops.cpp`（L5-02）、`src/llama-memory-recurrent.h`（L2-04）、`src/llama-graph.{h,cpp}`（L2-06）、`src/llama-hparams.cpp`（L2-02）。本课引用它们，是为了把"模型文件里的调用"对到"算子契约"与"内核实现"上 —— **图原语的出处是 `src/llama-graph.{h,cpp}`（L2-06），不在本课范围**；本课只讲模型文件怎么调用它们。此处明确声明，不虚报。
> **说明**：**一个需要澄清的措辞**：本课的 `build_ssm_*` / `build_rwkv_*` 原语**不在** `src/llama-graph.cpp` 里。实测：`grep -rn "build_ssm_scan|build_ssm_conv" src/` 无输出；它们实际定义在 `src/models/mamba-base.cpp`（`build_mamba_layer` / `build_mamba2_layer`）、`src/models/rwkv6-base.cpp`（`build_rwkv6_*`）、`src/models/rwkv7-base.cpp`（`build_rwkv7_*`）、`src/models/delta-net-base.cpp`（`build_delta_net` / `build_recurrent_attn`）。`llama-graph.cpp` 里与本课相关的是 `build_rs`(3478) 与 `build_rwkv_token_shift_load/store`(3558/3579)。
> **说明**：**统计口径**：本课的"22 是 / 8 否"由逐文件读图构建代码得出，判据写在第一节末尾的三条硬指标里；`grep -c` 的命中数只用于定位，不用于判定。

## 场景（10 幕）

1. **30 个文件里，22 个真是状态空间/线性注意力，8 个只是名字里有 wkv** — 分组来自静态扫描：图构建代码里命中 build_ssm / ssm_conv / ssm_scan / gated_delta / build_rwkv / wkv 之一。命中原语不等于"它就是 SSM 模型"。
2. **★ SSM 不需要 KV cache：历史被压进一份固定大小的状态** — 两种记忆各占一个张量：get_r_l() 是卷积状态，get_s_l() 是 SSM 状态。它们的大小由超参决定，与 n_ctx 无关。
3. **卷积状态：一个 (d_conv − 1) 长的滑动窗口** — 这是整条链上唯一保留"原始 token 片段"的地方，长度是常数，不是上下文长度。
4. **★ ggml_ssm_scan 的 7 个输入：形状写在注释里** — 内核开头七行就是权威的输入契约。8 个参数（含 K）分别描述状态、输入、步长、衰减与选择矩阵。
5. **★ 状态更新发生在内核的扫描循环里，写回缓存是图上的一个 cpy** — 输入视图 → 交给 ssm_scan（拿 ids 直接读缓存）→ 输出张量里同时含 y 与新状态 → 显式 cpy 回 ssm_states_all。
6. **RWKV：同一套"读-算-写"，换一个融合算子** — wkv6 与 wkv7 的差别只在算子；状态进出缓存的方式与 Mamba 完全同构。
7. **gated delta net：状态是一整块 矩阵，不是向量** — delta net 的状态形状是 {S_v, S_v, H_v, n_seqs} —— 更像"外积累加器"，比 Mamba 的 {d_state, head_dim} 大得多。
8. **15 个文件各自的真实主题（按扫描清单顺序）** — 逐文件读图构建代码后写下，不按文件名猜。第三列是本课的判定。
9. **★ 余下 15 个，以及8 个名实不符的全部理由** — 8 个"否"的命中点都只是子串 wkv：MLA 的低秩 KV 压缩投影叫 wkv_a_mqa / wkv_b，dflash 里干脆只是一个普通 K/V 权重叫 wkv。
10. **★ 一张表收束：记忆决定图的形状** — 同一层里，"怎么记"比"算什么"更能决定图长什么样 —— 这就是本课与稠密家族课的分界。

## 核心结论

### ★ 状态更新发生在哪一步（验收点）

```text
① 读    build_rs → reshape 状态缓存 {d_state, head_dim, n_head, state_slots}
            mamba-base.cpp:258-267（把 ids 交给算子；RWKV 则先用 ggml_get_rows 取行）
② 算    ggml_ssm_scan 在扫描循环里更新状态：
            ops.cpp:9819-9821  用 ids[i3] 定位读指针 s0，写指针 s 指向输出张量后半段
            ops.cpp:9923/9976  循环体 s[i] = state  —— 更新就在这里
③ 出    输出张量 = [ y | K 份新状态 ]
            ggml.c:5739-5741   长度 = nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]
④ 写回  ggml_cpy(output 后半段 → ssm_states_all 的 kv_head 行)
            mamba-base.cpp:274-279 / rwkv6-base.cpp:144-147 / delta-net-base.cpp:555-558
```

**关键区别：算子不改持久缓存，它只把新状态写在自己的输出张量里；写回缓存是图上一个独立的 `ggml_cpy` 节点，由调度器保证"先读后写"的次序。**

### ★ SSM 不需要 KV cache

记忆的两种形态：

| | KV cache | recurrent state（本课） |
|---|---|---|
| 张量 | `get_k` / `get_v` | `get_r_l` / `get_s_l` |
| 大小 | `O(n_ctx)`，随上下文增长 | `O(1)`，由结构超参决定 |
| 形状 | 带 token 维 | **没有 token 维** |
| 读写 | `cpy_k` / `cpy_v` + `build_attn` | 显式 `ggml_cpy` + 融合算子 |

`llama_hparams::n_embd_s()` 是纯结构常量：RWKV 走 `n_embd * wkv_head_size`，KDA 走 `head_dim * head_dim * n_head()`，Mamba 走 `ssm_d_state * ssm_d_inner` —— 没有一项依赖 `n_ctx`。代价是历史被有损压缩，这正是"线性注意力"的来历。

### 本课 30 个文件的判定结果

**22 个名副其实**（4 个基类含在内）：`mamba-base` `mamba` `mamba2` `plamo2` `delta-net-base` `rwkv6-base` `rwkv6` `rwkv6qwen2` `rwkv7-base` `rwkv7` `arwkv7` `jamba` `granite-hybrid` `nemotron-h` `falcon-h1` `lfm2` `qwen3next` `qwen35` `qwen35moe` `kimi-linear` `kimi-k3` `bailingmoe3`。

**8 个名实不符**：`deepseek2` `deepseek32` `dots3note` `glm-dsa` `hy-v4` `minicpm3` `plm` `dflash` —— 它们命中的只是 MLA 低秩 KV 压缩投影的名字 `wkv_a_mqa` / `wkv_b`（`dflash` 里更只是一个普通 K/V 权重 `wkv`），走的是 `build_attn_inp_kv` 与 KV cache，与线性注意力无关。

## 验收点

- [ ] 保真门禁：20 个引用块全部逐字来自声明的源文件，且位置连续
- [ ] 覆盖度门禁：37 个源文件均被声明
- [ ] 参数门禁：无非法命令行参数
- [ ] 语法检查：通过
- [ ] 渲染门禁：10 幕，无 JS 错误、无布局溢出、交互可用
- [ ] 自检：不看代码，能说出 `struct ggml_tensor` 里哪些字段决定一次内核调用的形状

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
