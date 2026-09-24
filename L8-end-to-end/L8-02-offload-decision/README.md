# L8-02 · ★ 端到端：offload 决策实战 — 课件说明

> 层：**L8 · 端到端** ｜ 前置课：`L8-01`

## 学习目标

看完这一课，你应该能：

1. 说出 `n_gpu_layers` 的**真实语义**，并解释为什么"把 27 层模型全放 GPU"要写 28（对应验收点）；
2. 给定 `n_layer_all` 与 `n_gpu_layers`，算出 `i_gpu_start` 并列出哪些层、哪些槽位在 GPU 上（对应验收点）；
3. 说明"load_tensors 按**层**分配设备"与"调度器按**节点**切图"这两次决策的接口是什么；
4. 解释为什么 `n_gpu_layers` 不能推出 `ggml_backend_sched_get_n_splits()` 的值；
5. 指出输入层与输出层在 offload 决策里的不同待遇，并各给出行号。

## 覆盖的源文件（3 个）

| 文件 | 行数 |
|---|---|
| `src/llama-model.cpp` | 3360 |
| `ggml/src/ggml-backend.cpp` | 2514 |
| `src/llama-context.cpp` | 4413 |

> **说明**：本课覆盖 3 个源文件，均计入覆盖率：`src/llama-model.cpp`、`ggml/src/ggml-backend.cpp`、`src/llama-context.cpp`。
> **说明**：算例用的 27 层取自 `src/models/deepseek2.cpp:7-8`（源码注释点名 DeepSeek-V2-Lite，判据 `hparams.n_layer() == 27`）。该文件不在本课声明里，**不计入本课覆盖率**。
> **说明**：`LLAMA_MAX_LAYERS = 512`（`src/llama-hparams.h:11`）是 `n_layer_all` 的上界，`llama-model.cpp:1260` 用它做断言。该头文件不在本课声明里，不计入本课覆盖率。
> **说明**：本课的所有数字都是**源码算术**的结果（`i_gpu_start` 公式 + 槽位判据），不是运行时实测：本机的 llama.cpp 只构建了 CPU 后端（`llama_supports_gpu_offload()` 为假），`load_tensors()` 的 offload 日志分支因此不会执行。要实测请在带 GPU 后端的机器上跑并打开调试日志。

## 场景（9 幕）

1. **n_gpu_layers 到 split：一条五站的决策链** — 用户只给了一个整数，图却被改了形状。这条链上有两次完全不同的决策。
2. **★ n_gpu_layers = 最后 N 个槽位，不是前 N 层** — 槽位从 0 数到 n_layer_all，最后那一个槽位是输出层 —— 它也算在 N 里面。
3. **层号 -> 设备 -> 权重的 buffer：dev_layer[] 是唯一的路由表** — 每个重复层拿到一个 (设备, 候选 buffer 类型表)；输入层永远在 CPU，输出层由同一个函数算出。
4. **调度器只认 buffer：usage == WEIGHTS 就是"层"信息的载体** — "这一层的权重在 GPU 上"这句话，到调度器这里被翻译成"这个 src 的 buffer 用了 WEIGHTS"。
5. **★ offload 的粒度是层，切分的粒度是节点** — 段是"图上连续的一段节点"，边界落在 node 下标 i 上 —— 不是落在层号上。
6. **没有权重的节点由扩张决定归属，而 CPU 是一堵墙** — pass 2 把 GPU 归属向上下两个方向推开，碰到 CPU 归属的节点就把"当前后端"清空。
7. **边界不一定落在层边界上：源码自己打了一个补丁** — norm / l_last 可能被扩张轮次判给"下一层"的后端 —— 于是 llama_context 手工把它按回去。
8. **把一个 27 层模型从 n_gpu_layers = 0 走到 28** — 算例模型取自源码里点名的真实架构：DeepSeek-V2-Lite，n_layer_all = 27。
9. **把决策链压成一张表** — 五个决策点、各自的判据与产出；最后用一道题自测。

## 核心结论

### ★ n_gpu_layers 的真实语义：末尾 N 个槽位

`i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`（`src/llama-model.cpp:1521`），槽位号 `>= i_gpu_start` 的才上 GPU。槽位总数是 `n_layer_all + 1`：重复层 `0..n_layer_all-1` 加上一个**输出层**槽位。

| 说法 | 对错 |
|---|---|
| "把前 N 层放 GPU" | 错。是**末尾** N 个槽位 |
| "`--n-gpu-layers 27` 让 27 层全上 GPU" | 错。`i_gpu_start = 1`，第 0 层仍在 CPU |
| "要全上 GPU 得给 `n_layer_all + 1`" | 对。参数取 `-1`（默认）时 `n_gpu_layers()` 返回的正是这个值（1926 行） |

### ★ 两次决策，两种粒度

| | 第 1 次（加载期） | 第 2 次（建图期） |
|---|---|---|
| 决策者 | `llama_model_base::load_tensors()` | `ggml_backend_sched_split_graph()` |
| 粒度 | **层**（`dev_layer[il]`） | **节点**（`graph->nodes[i]`） |
| 判据 | `i_gpu_start` 与 `splits[]` | `src->buffer->usage == WEIGHTS` + 扩张 |
| 产出 | 权重落在哪块 buffer | `splits[]`（节点区间）+ 边界 copy |

接口只有一处：**权重 buffer 上的 `WEIGHTS` 标记**。所以"给定 `n_gpu_layers` 预测切分结果"的正确做法是：先按 1521 行算出哪些权重在哪，再看调度器按它的规则读出什么 —— 而不是拿层数直接当段数。

### 边界会被"修补"，不是算法的性质

`llama_context::graph_get_cb` 只对 `norm` / `l_last` 做手工钉死，且只在 `ubatch.n_tokens < 32 || full_offload` 时执行。也就是说：**预填充（`n_tokens >= 32`）且非完全 offload 时，段边界可以劈开一层**。这一段的存在本身就是"切分粒度是节点、不是层"的最好证据。

## 验收点

- [x] 保真门禁：19 处引用 —— 19 个引用块 / 29 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 3 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 5 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
