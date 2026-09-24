# L8-03 · 端到端：新后端要做什么 — 课件说明

> 层：**L8 · 端到端** ｜ 前置课：`L8-02`

## 学习目标

看完这一课，你应该能：

1. 列出新增一个后端**必须实现**的接口函数（按 L3-01 的三张主虚表分组，共 15 个必需项），并说出哪些可以直接留空（对应验收点）；
2. 说出 BLAS 后端"少写了什么"：内存层的 `buffer_type_i` + `buffer_i` 共 17 个指针一行未写，靠 `ggml_backend_cpu_buffer_type()` 借来（对应验收点）；
3. 解释 `supports_op` 与 `graph_compute` 的契约关系：前者是承诺（决定切分），后者是兑现（`default` 是 `GGML_ABORT`），因此二者必须一致；
4. 说出必须注册 / 导出的符号（`reg()`、`api_version`、`ggml_backend_init`，可选 `ggml_backend_score`）与必须跑通的四条验证步骤（对应验收点）。

## 覆盖的源文件（5 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-backend-impl.h` | 292 |
| `ggml/src/ggml-blas/ggml-blas.cpp` | 531 |
| `ggml/include/ggml-blas.h` | 26 |
| `ggml/src/ggml-backend-reg.cpp` | 606 |
| `ggml/src/ggml-backend-dl.cpp` | 49 |

> **说明**：本课逐字引用 **5 个文件**，全部计入覆盖率：`ggml/src/ggml-backend-impl.h`（计划指定）、`ggml/src/ggml-blas/ggml-blas.cpp`、`ggml/include/ggml-blas.h`、`ggml/src/ggml-backend-reg.cpp`、`ggml/src/ggml-backend-dl.cpp`（后四个为本课追加，用来把"清单"落到真实代码上：BLAS 是最小的真实后端，reg/dl 是注册与加载的实现）。
> **说明**：文中提到但**不计入本课覆盖率**的文件：`ggml/src/CMakeLists.txt`（第 589 行 `ggml_add_backend(BLAS)`）、`ggml/CMakeLists.txt`（第 86 行 `GGML_BACKEND_DL`、第 194 行 `GGML_BLAS`）、`ggml/src/ggml-blas/CMakeLists.txt`、`ggml/src/ggml-backend-dl.h`、`tests/test-backend-ops.cpp`、`common/arg.cpp` —— 它们是构建接线与测试工具，不属于本课覆盖域。
> **说明**：本课引用的数字来自实测命令：`wc -l ggml/src/ggml-blas/ggml-blas.cpp` = 530；`wc -l ggml/src/ggml-zendnn/ggml-zendnn.cpp` = 838；`find ggml/src/ggml-cuda -type f \( -name "*.cu" -o -name "*.cuh" \)` = 277 个文件、合计 45718 行；`grep -c "ggml_backend_buffer_i\|ggml_backend_buffer_type_i" ggml/src/ggml-blas/ggml-blas.cpp` = 0；三张虚表的非 NULL 指针按 `= */` 行统计为 3（backend_i）+ 10（device_i）+ 4（reg_i）= 17。
> **说明**：**边界说明**：本机构建为 `GGML_BLAS=OFF`、`GGML_BACKEND_DL=OFF`，且 `build/bin` 下没有 `test-backend-ops`，因此第十二节的四条验证步骤**来自源码**（`tests/test-backend-ops.cpp` 的 `usage()` 第 12003-12021 行、`main()` 第 12136-12149 行；`common/arg.cpp` 第 1138-1160 行），不是本课的运行输出。本课实测过的命令只有 `llama-cli --list-devices`，输出是 `Available devices:` 与 `  (none)`（因为没有启用任何非 CPU 后端）。

## 场景（9 幕）

1. **最后一课：把七层压成一份清单** — 新增一个后端要回答四个问题：接口函数写哪些、哪些可以留空、必须导出什么符号、必须跑通哪些测试。
2. **★ 不是从零开始：BLAS 后端就是活证据** — 530 行里只有三张虚表被填；内存层的 17 个指针一行都没写 —— 它直接借了 CPU 的 buffer type。
3. **三张虚表逐张看：device_i 是最大的一张** — 设备层 10 个、计算层 3 个、注册层 4 个；内存层与数据面直接沿用 CPU 的实现。
4. **★ supports_op：新后端的第一等公民** — 切分器只问它一个问题：这个算子你能跑吗？说 false 的算子会被留给别的后端 —— 这就是"回退"。
5. **graph_compute 的 default 是 GGML_ABORT** — 后端自己不做回退：说 false 的算子根本不会进来；进来的算子如果没实现，就直接 abort。
6. **一个工厂函数 + 一个导出宏，就是注册的全部** — 静态构建时 reg.cpp 直接调用 reg()；动态构建时 GGML_BACKEND_DL_IMPL 把同一个 reg() 导出成 ggml_backend_init。
7. **清单（a）必须实现 ·（b）可以留空** — 按 L3-01 的三张主虚表分组：必需 23 项、可留空 29 项；BLAS 的最小做法是"借掉 8 个必需项"。
8. **清单（c）必须导出的符号 ·（d）必须跑通的测试** — 符号面只有 4 个名字；测试面从"设备出现"到"端到端推理"共 4 步，全部用仓库现成工具。
9. **从模型到后端：整套课件的一张总图** — 一个算子从模型定义出发，经过建图、切分、后端契约、内存、内核，最后落到某个设备上 —— 这条链就是全套课件。

## 核心结论

### ★ 写一个新后端不是从零开始

BLAS 是仓库里最小的真实后端：**1 个文件、530 行**，只定义三张虚表。

| 虚表 | 槽位 | BLAS 填了 | 说明 |
|---|---|---|---|
| `ggml_backend_device_i` | 15 | 10 | 自我介绍 5 + 创建/内存 3 + 能力 2 |
| `ggml_backend_i` | 16 | 3 | `get_name` / `free` / `graph_compute` |
| `ggml_backend_reg_i` | 4 | 4 | 名字 / 设备数 / 取设备 / 扩展入口 |
| `buffer_type_i` + `buffer_i` | 17 | **0** | 直接返回 `ggml_backend_cpu_buffer_type()` |

合计 52 个函数指针里只碰了 **17** 个；必需项也因此从 23 降到 15（内存层 8 个借出去了）。对照 ggml-cuda 的 277 个文件 / 45718 行 —— **BLAS 的规模就是"最少要写多少"的答案**。

### ★ supports_op 是承诺，graph_compute 是兑现

切分器（L4-02）只按 `supports_op` 决定归属：说 `false` 的算子自动回退给别的后端，**不需要后端自己写任何转发代码**。而 `graph_compute` 的 `default` 分支是 `GGML_ABORT`（第 253 行）—— 进来的算子如果是没实现的，进程直接终止。

所以新后端最容易踩的坑是：**`supports_op` 说了 true，`graph_compute` 没实现。**

### 注册与导出：两条路，同一个入口

静态构建（`GGML_USE_BLAS` + `ggml_add_backend(BLAS)`）：注册表构造函数直接调用 `reg()`（`reg.cpp` 第 160 行）。动态构建（`GGML_BACKEND_DL`）：`GGML_BACKEND_DL_IMPL` 展开出 `extern "C"` 的 `ggml_backend_init`，加载器 `dlopen` + `dlsym` 找到它（`reg.cpp` 第 237 行），再校验 `api_version`（第 246 行）。两条路的产物都是同一个 `ggml_backend_reg`。

### 新后端清单（本课验收点）

**(a) 必须实现**：`device_i` 9 + `backend_i` 3 + `reg_i` 3 = 15 个（内存层与数据面可借 CPU，全量则是 23 个）。

**(b) 可留空**：所有带 `(optional)` 注释的 29 个槽位；最小实现里 `backend_i` 只填 3 个、`device_i` 只填 9 个。

**(c) 必须注册 / 导出**：`<name>_reg()`、`reg->api_version = GGML_BACKEND_API_VERSION`、`ggml_backend_init`（DL 宏生成），可选 `ggml_backend_score`；扩展函数走 `get_proc_address`。

**(d) 必须通过**：`llama-cli --list-devices`（设备出现）-> `test-backend-ops support -b <NAME>`（能力探测）-> `test-backend-ops test -b <NAME>`（与 CPU 比误差）-> 真实模型端到端推理。

## 验收点

- [x] 保真门禁：19 处引用 —— 19 个引用块 / 54 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 5 项，无空课、无幻影；全局覆盖 1048/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 5 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
