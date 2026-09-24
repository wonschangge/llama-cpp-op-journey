# L4-04 · 图执行入口：graph_compute 与异步 — 课件说明

> 层：**L4 · 内存与调度** ｜ 前置课：`L4-03`

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml_backend_graph_compute()` 与 `_async()` 的唯一差别，以及那一行调到哪里（对应验收点）；
2. 解释 `ggml_backend_synchronize()` 为什么可能"什么都不做"，并说出 CPU 与 CUDA 两种后端各自的行为；
3. 说出 `event_record` / `event_wait` / `event_synchronize` 的等待主体分别是谁；
4. 说出 graph reuse 的三层缓存，以及调度器判断"切分是否还算数"的判据；
5. 说明 `ggml.c` 在 v0.5.0 里还剩什么（容器、视图、状态字符串），执行入口搬到了哪里。

## 覆盖的源文件（2 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-backend.cpp` | 2514 |
| `ggml/src/ggml.c` | 8168 |

> **说明**：本课逐字引用 `ggml/src/ggml-backend.cpp` 与 `ggml/src/ggml.c` 两个文件，计入覆盖率。
> **说明**：文中按行号提到的 `ggml/src/ggml-cpu/ggml-cpu.cpp`、`ggml/src/ggml-cpu/ggml-cpu.c`、`ggml/src/ggml-cuda/ggml-cuda.cu`、`src/llama-context.cpp` 分别是 L5-01、L6-01、L2-07 的覆盖文件，本课只指路、不引用其源码，**不计入本课覆盖率**。

## 场景（9 幕）

1. **一个入口，两条路径：提交 与 完成** — 两个函数体只差一行 —— 那一行就是"等待"。
2. **等待发生在 iface.synchronize，不在这里** — 这一层的全部内容是"断言 + 判空 + 转发"：后端没实现 synchronize，它就立刻返回。
3. **事件：把"完成"变成可等待的句柄** — record 在一次提交之后盖章；wait 让另一个后端等，synchronize 让 host 等。
4. **调度器把同一对入口又写了一遍** — ggml_backend_sched_graph_compute() = 异步版 + 对所有后端各等一次。
5. **★ graph reuse：第二次跑同一张图，不重切、不重分配** — 判据是"每个节点的后端归属有没有变"；没变就落回上一次的分配结果。
6. **同步路径的最后一步：对每个后端都等一次** — ggml_backend_sched_synchronize() 循环所有后端，顺带把 next_copy 归零。
7. **★ 异步把"提交"与"完成"分开：等待只剩这几处** — 每个 split 用 _async 提交（第 1799 行），提交后立刻盖章（第 1839 行）；host 只在必要时才等。
8. **执行入口已经不在 ggml.c 里了** — v0.5.0 实测：ggml.c 里 graph_compute 出现 0 次；ggml_graph_plan 搬到了 ggml-cpu.c。
9. **把"在哪一步等"压成一张表** — 同步路径在 ggml_backend_synchronize 里等；异步路径把等待推迟到跨后端交接与收尾。

## 核心结论

### 同步 = 异步 + 一次等待

`ggml_backend_graph_compute()`（第 455-459 行）的三行函数体就是全部答案：

```text
err = ggml_backend_graph_compute_async(backend, cgraph);   // 第 456 行：提交
ggml_backend_synchronize(backend);                          // 第 457 行：等待
return err;
```

**同步路径在 `ggml_backend_synchronize()` 里等**（第 431 行转到 `iface.synchronize`）；异步路径不等，它把"完成"变成事件，等的时候再等。

### ★ 等待点清单

| 路径 | 在哪一步等待 |
|---|---|
| 单后端同步 | `ggml_backend_graph_compute` 第 457 行 → 第 431 行 `iface.synchronize` |
| 调度器同步 | `ggml_backend_sched_graph_compute` 第 2013 行 → 第 2034-2036 行循环所有后端 |
| 调度器异步 | 只在跨后端交接（1663-1669）、覆盖用户输入（1679-1683）、收尾（2032-2043）时等 |

CPU 后端 `iface.synchronize == NULL`，第 427-429 行空返回 —— 它的等待发生在 `graph_compute` 内部（`ggml_graph_compute` 返回即算完，L5-01）；CUDA 后端在第 4477 行提交完就返回，等的是 `cudaStreamSynchronize`（`ggml-cuda.cu:2544`）。

### ★ graph reuse 是三层短路

① 调度器层：`is_alloc` 为真则跳过 reset 与 alloc（第 2019-2027 行）；
② 切分层：`node_backend_ids` / `leaf_backend_ids` 与上一次比较，"后端变了且 buffer type 变了"才算变（第 1594-1595、1600-1608 行）；
③ 显存层：`ggml_gallocr_alloc_graph()` 在预留块里重新指派（第 1611 行，细节见 L4-01）。

`ggml_backend_sched_synchronize()` 之后还会把 `next_copy` 固定回 0（第 2037-2041 行），注释写明原因：副本轮转会改变图结构，可能让 CUDA 等后端的图缓存失效。

## 验收点

- [x] 保真门禁：17 处引用 —— 17 个引用块 / 34 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 2 项，无空课、无幻影；全局覆盖 257/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 87 文件 15 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
