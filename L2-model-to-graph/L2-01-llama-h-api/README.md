# L2-01 · llama.h 公共 API 全景 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L1-06`

## 学习目标

看完这一课，你应该能：

1. 说出 `include/llama.h` / `include/llama-cpp.h` / `src/llama.cpp` / `src/llama-ext.h` 四个文件各自的角色（对应验收点）；
2. 默画出从 `llama_backend_init` 到 `llama_sampler_sample` 的六步调用顺序，并指出哪一步必须有 memory；
3. 说清 `llama_model` / `llama_context` / `llama_sampler` 三个不透明类型各自的创建与释放函数；
4. 解释为什么"默认值"由 `llama_*_default_params()` 函数给出，而不是写在结构体字段里。

## 覆盖的源文件（4 个）

| 文件 | 行数 |
|---|---|
| `include/llama.h` | 1647 |
| `src/llama.cpp` | 621 |
| `include/llama-cpp.h` | 31 |
| `src/llama-ext.h` | 135 |

> **说明**：本课引用 `include/llama.h`、`include/llama-cpp.h`、`src/llama.cpp`、`src/llama-ext.h` 四个文件，全部计入覆盖率。
> **说明**：散文与图示里提到的内部定义位置（`src/llama-context.cpp:3739`、`src/llama-context.cpp:4326`、`src/llama-vocab.cpp:4496`、`src/llama-sampler.cpp:895`、`src/llama-quant.cpp:847`）由 `grep -n` 实测确认，但**不构成本课的代码引用**，故不计入本课覆盖率。

## 场景（9 幕）

1. **一个头文件，四个文件的分工** — 公共 API 的边界在哪里：契约、实现入口、C++ 包装、未定型的扩展。
2. **★ 三个不透明指针，三段生命周期** — 模型 / 上下文 / 采样器各有独立的创建与释放函数；默认值由函数给出，不写在字段里。
3. **参数结构体族：4 个 struct + 4 个默认值函数** — llama.cpp 不用"一堆散参数"，而是"一个按值传的结构体 + 一个取默认值的函数"。
4. **★ 六步调用顺序：从 backend_init 到 sampler_sample** — 一个用 llama.cpp 的程序，最小骨架就是这六个调用。顺序不能换。
5. **llama_batch：喂给 decode 的那一张表** — 七个字段全是并行数组，长度都等于 n_tokens；这是 L2 层"数据进图"的统一入口。
6. **★ src/llama.cpp：620 行的接线板** — 它 include 了所有内部模块，自己却几乎不算数：只做初始化、加载编排、chat 模板与分片路径。
7. **llama-cpp.h：把所有权写进类型** — 30 行，四个 deleter，四个 unique_ptr 别名 —— C 的"记得手动释放"变成 C++ 的"不可能忘记"。
8. **llama-ext.h：试验场，与它没做到的自律** — 新 API 先在这里长出来，稳定了再搬进 llama.h；但它开头那句"尽量别被 include"已经被打破了。
9. **把这一课压成一张表** — 头文件自己就写了正确用法：先建链，再循环 decode → sample。

## 核心结论

### 四个文件，四种角色

| 文件 | 角色 | 关键内容 |
|---|---|---|
| `include/llama.h` | 公共 C ABI 契约 | 三个不透明类型、四个参数结构体、六步调用序 |
| `include/llama-cpp.h` | C++ RAII 包装 | 四个 deleter + 四个 `unique_ptr` 别名 |
| `src/llama.cpp` | 实现与装配入口 | `llama_backend_init` / 模型加载编排 / chat 模板 / 分片 |
| `src/llama-ext.h` | staging 扩展头 | `llama_graph_reserve`、量化状态 API、显存账单 |

### ★ 六步调用顺序

```text
llama_backend_init          482    进程级，一次
llama_model_load_from_file  517    读 GGUF（容器格式见 L1-06）
llama_init_from_model       544    建 memory / 调度器
llama_tokenize              1180   第一个参数是 const llama_vocab *
llama_decode                998    必须有 memory；这一步才碰图
llama_sampler_sample        1548   在 CPU 上消费 logits，不进图
```

**只有 `llama_decode` 会真正跑图**；采样在 CPU 上、图之外完成。

### ★ 聚合入口的地位

`src/llama.cpp` 只有 620 行，却 include 了 `llama-impl.h`、`llama-version.h`、`llama-chat.h`、`llama-context.h`、`llama-mmap.h`、`llama-vocab.h`、`llama-model-loader.h`、`llama-model-saver.h`、`llama-model.h` 九个内部头 —— 用 `grep -rl` 逐个核对这些头的 include 清单，**交集只有它一个文件**。因此模型加载编排（`llama_model_load`，316）与设备准备落在这里。想找某个公共函数的实现，先看这里，再看它转发给了哪个模块。

## 验收点

- [x] 保真门禁：17 处引用 —— 17 个引用块 / 40 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 4 项，无空课、无幻影；全局覆盖 844/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
