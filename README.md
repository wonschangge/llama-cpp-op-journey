# llama.cpp 算子之旅

> **一个视角**：一个算子，从模型定义出发，如何落到 CPU / GPU / NPU 上真正执行。

这是一套**可离线双击打开**的动画课件，逐层拆解 [llama.cpp](https://github.com/ggml-org/llama.cpp)
里"算子"的一生：它在 `ggml` 里怎么被表示、怎么从模型定义被建造成一张图、
怎么被调度器切给不同后端、最后在 CPU / CUDA / Vulkan / Metal / SYCL / CANN /
Hexagon / OpenVINO 等后端上变成真实的 kernel。

上游锚定版本：**`v0.5.0`**（commit `7fe450e19305b828c199d602c23a8337aaa1f03b`）。

---

## 为什么值得看

llama.cpp 有 17 个后端、1290 个"算子执行路径"上的源文件。文档大多按目录讲，
学完还是不知道**一个 `ggml_mul_mat` 到底走了哪些代码**。

这套课件只走一条线：**算子的旅程**。每一课都是"看源码 → 看动画 → 做练习"，
每一段代码引用都**逐字**来自上游源文件，并且经过**四道门禁**验证。

## 四道门禁

课件里"看起来对"不算数，以下四条都由脚本判定，退出码即结论：

| 门禁 | 断言 | 脚本 |
|---|---|---|
| **覆盖度** | 覆盖域中 1290 个源文件，每个都至少被一课引用；无空课、无幻影引用 | `tools/check_coverage.py` |
| **保真** | 课件里每一段代码引用都**逐字**来自源文件，且在源文件中**位置连续** | `tools/check_ir_fidelity.py` |
| **参数有效性** | 课件提到的每个命令行参数都真实存在（跑真实二进制的 `--help` 取真值） | `tools/check_flags.py` |
| **渲染** | 每课在无头浏览器里 0 JS 错误、0 布局溢出、交互可用（两种分辨率） | `tools/check_render.py` |

## 分层课程（52 课）

| 层 | 主题 | 课数 | 文件 |
|---|---|---|---|
| **L1** | 算子的表示 —— ggml 用什么统一表示一个算子 | 6 | 17 |
| **L2** | 从模型到图 —— 权重与超参如何变成待执行的图 | 15 | 224 |
| **L3** | 后端发现与注册 —— 机器上有哪些后端 | 4 | 7 |
| **L4** | 内存与调度 —— tensor 住在哪、图怎么切给后端 | 4 | 9 |
| **L5** | CPU 后端执行 —— 量化点积与 SIMD 多架构 | 5 | 68 |
| **L6** | GPU 后端执行 —— CUDA / SYCL / Vulkan / Metal | 9 | 690 |
| **L7** | NPU 与加速器后端 —— CANN / Hexagon / OpenVINO / ExecuTorch ... | 6 | 312 |
| **L8** | 端到端 —— 一个 `mul_mat` 的完整旅程 | 3 | 4 |

完整清单见 [`plan/TODOLIST.md`](plan/TODOLIST.md)，
文件到课的映射见 [`plan/COVERAGE.md`](plan/COVERAGE.md)。

## 怎么用

- **在线看**：打开 <https://wonschangge.github.io/llama-cpp-op-journey/>
- **离线看**：clone 后直接双击任意一课的 `index.html`。
  零 CDN、零构建、零网络请求 —— `shared/` 是唯一的引擎实现。

## 目录结构

```
.
├── index.html              门户（过滤 / 搜索）
├── shared/                 动画引擎（唯一一份，各课引用，不内联复制）
├── tools/                  四道门禁 + 计划矩阵 + 拆块工具
├── plan/                   TODOLIST.md（计划）· COVERAGE.md（覆盖矩阵）
└── L1-.../L8-.../          各层课件，每课四个文件：
                            index.html / lesson.js / source.md / README.md
```

## 重新验证

```bash
export LLAMA_UPSTREAM=/path/to/llama.cpp      # 上游 checkout（v0.5.0）
python3 tools/lint_lessons.py                 # 静态语法检查（秒级）
python3 tools/check_ir_fidelity.py --all      # 保真
python3 tools/check_flags.py                  # 参数有效性
python3 tools/check_coverage.py               # 覆盖度
python3 tools/check_render.py --all           # 渲染（慢，建议后台）
```

## 许可与归属

课件正文与 `shared/`、`tools/` 代码以 MIT 许可发布。

课件中**逐字引用**的源码片段来自 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，
同样以 MIT 许可发布，版权归其贡献者所有。详见 [`NOTICE`](NOTICE)。
