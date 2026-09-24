# L2-02 · 架构表与超参：模型长什么样的元数据 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-01`

## 学习目标

看完这一课，你应该能：

1. 说出 `enum llm_arch` 有多少个成员、哨兵是谁，以及它与 GGUF `general.architecture` 的对应关系（对应验收点）；
2. 解释 `LLM_KV::operator()` 如何把架构名与 key 模板拼成真正的 metadata key —— 这是 hparams 的来源；
3. 说清 `llama_hparams` 与 `llama_cparams` 的分工判据：这个数在不在模型文件里；
4. 列举新增一个模型架构必须同时修改的几张表，并说出漏掉每一张的后果。

## 覆盖的源文件（6 个）

| 文件 | 行数 |
|---|---|
| `src/llama-arch.h` | 798 |
| `src/llama-arch.cpp` | 1167 |
| `src/llama-hparams.h` | 513 |
| `src/llama-hparams.cpp` | 354 |
| `src/llama-cparams.h` | 68 |
| `src/llama-cparams.cpp` | 6 |

> **说明**：本课逐字引用 6 个文件：`src/llama-arch.cpp`、`src/llama-arch.h`、`src/llama-hparams.cpp`、`src/llama-hparams.h`、`src/llama-cparams.cpp`、`src/llama-cparams.h`，全部计入覆盖率。
> **说明**：场景与正文里用 `文件:行` 形式指路、但**没有逐字引用**的文件有：`src/llama-model.cpp`（1254/1255/1259/1260/1327/1328/1334/43 行）、`src/llama-model-loader.cpp`、`src/llama-context.cpp`（84/246 行）。它们不计入本课覆盖率，其行号由本课作者用 `grep -n` / `sed -n` 实地核对过。

## 场景（9 幕）

1. **一个模型"长什么样"，写在三张表和两个结构里** — 架构表回答"这是哪个家族"，hparams 回答"模型的形状"，cparams 回答"这次怎么跑"。
2. **153 条名字，一张"身份证"表** — enum llm_arch 里 152 个真架构 + 1 个哨兵；名字表正好 153 条。
3. **★ %s：GGUF 的 key 是"算"出来的** — LLM_KV::operator() 把 LLM_KV_NAMES 的模板和 LLM_ARCH_NAMES 的字符串拼成真正的 metadata key。
4. **★ 三张并行的表，条目数并不相等** — enum llm_tensor 274 项、LLM_TENSOR_NAMES 273 条、LLM_TENSOR_INFOS 268 条。
5. **名字 = format(模板, bid, xid)** — str() 从 LLM_TENSOR_NAMES 取模板，用层号和专家号填充 %d，再补上调用方给的后缀。
6. **llama_hparams：模型文件说什么，就是什么** — n_ctx_train / n_embd / n_layer_all / n_expert：全部由 GGUF metadata 决定。
7. **★ 每个形状字段都有 (il) 版本：逐层可以不同** — n_head_arr / n_head_kv_arr / n_ff_arr 是长度 LLAMA_MAX_LAYERS 的数组，getter 按层号取值。
8. **llama_cparams：一次运行的可变配置** — n_ctx / n_batch / n_ubatch / n_threads 都来自 llama_context_params —— 同一个模型可以有多个 context。
9. **新增一个架构，要动哪几张表** — 五张表 + 一组查询函数；漏掉任何一张，模型要么"认不出来"，要么"张量找不到"。

## 核心结论

### 架构表：一张表，三个出口

`enum llm_arch` **153** 个成员（152 个真架构 + 哨兵 `LLM_ARCH_UNKNOWN`），与 `LLM_ARCH_NAMES` 的 **153** 条一一对应。

| 出口 | 函数 | 位置 |
|---|---|---|
| 枚举 → 字符串 | `llm_arch_name()` | `src/llama-arch.cpp:1040` |
| 字符串 → 枚举 | `llm_arch_from_string()` | `src/llama-arch.cpp:1047` |
| 枚举 → 全部枚举 | `llm_arch_all()` | `src/llama-arch.cpp:1031` |

日志里打印的架构名、加载时认领的架构、遍历所有架构都走这一张表。

### ★ hparams 与 cparams 的分工判据

一句话判据：**这个数在模型文件里吗？**

| | hparams | cparams |
|---|---|---|
| 代表字段 | `n_ctx_train` `n_embd` `n_layer_all` `n_expert` | `n_ctx` `n_batch` `n_ubatch` `n_threads` |
| 来源 | GGUF metadata | `llama_context_params` |
| 谁读 | 建图（L2-06）、KV cache（L2-04） | 上下文与批处理（L2-05） |
| 生命周期 | 加载后固定 | 每个 `llama_context` 一套 |

分界线在源码里也有反证：`src/llama-cparams.h:28-30` 的注释说 YaRN 的四个系数**不放进 GGUF** —— 因为它们不属于模型。

### ★ 新增一个架构：五张表 + 一组查询

| 要动的表 | 位置 | 它回答什么 |
|---|---|---|
| `enum llm_arch` | `src/llama-arch.h:13-167` | 家族编号 |
| `LLM_ARCH_NAMES` | `src/llama-arch.cpp:8-162` | GGUF 架构字符串 |
| `enum llm_tensor` | `src/llama-arch.h:438-713` | 新权重的编号 |
| `LLM_TENSOR_NAMES` | `src/llama-arch.cpp:433-707` | 张量名模板 |
| `LLM_TENSOR_INFOS` | `src/llama-arch.cpp:719-998` | 层归属 + `ggml_op` |

若该架构是 recurrent / hybrid / diffusion，还要在 `llm_arch_is_recurrent`（1061）、`llm_arch_is_hybrid`（1075）、`llm_arch_is_diffusion`（1100）里加 case。**架构表决定"认不认得出"，张量表决定"找不找得到"。**

## 验收点

- [x] 保真门禁：17 处引用 —— 17 个引用块 / 27 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 6 项，无空课、无幻影；全局覆盖 59/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 33 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
