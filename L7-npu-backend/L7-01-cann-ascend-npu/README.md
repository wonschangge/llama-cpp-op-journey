# L7-01 · CANN 后端（Ascend NPU）：算子由厂商库提供 — 课件说明

> 层：**L7 · NPU 与加速器后端** ｜ 前置课：`L6-09`

## 学习目标

看完这一课，你应该能：

1. 说出 CANN 后端把一个 ggml op 映射到 ACL 算子的**完整链路**（分派 -> 转张量 -> 填参数 -> 问 workspace -> 下发到 stream）（对应验收点）；
2. 解释 `GGML_CANN_CALL_ACLNN_OP` 宏为什么是"两段式"，以及 workspace 从哪来；
3. 说明 `ggml_cann_create_tensor` 抹平了 ggml 与 CANN 之间的哪三处约定差异；
4. 判断一个 op 会不会被卸载到 NPU，并说出 `supports_op` 返回 false 的后果（呼应 L4-02）。

## 覆盖的源文件（7 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-cann.h` | 124 |
| `ggml/src/ggml-cann/ggml-cann.cpp` | 3071 |
| `ggml/src/ggml-cann/aclnn_ops.h` | 1192 |
| `ggml/src/ggml-cann/aclnn_ops.cpp` | 4480 |
| `ggml/src/ggml-cann/acl_tensor.cpp` | 196 |
| `ggml/src/ggml-cann/common.h` | 652 |
| `ggml/src/ggml-cann/acl_tensor.h` | 350 |

> **说明**：本课引用 7 个文件，全部计入覆盖率：`ggml/include/ggml-cann.h`、`ggml/src/ggml-cann/ggml-cann.cpp`、`ggml/src/ggml-cann/aclnn_ops.cpp`、`ggml/src/ggml-cann/aclnn_ops.h`、`ggml/src/ggml-cann/acl_tensor.cpp`、`ggml/src/ggml-cann/acl_tensor.h`、`ggml/src/ggml-cann/common.h`。
> **说明**：本课**没有**在 Ascend NPU 上实测运行（本机无昇腾设备）。所有断言都来自源码：函数名、行号、映射关系均可逐字核对；"异步执行""库决定算法"这类结论来自源码注释与调用形态，未在真机上验证。
> **说明**：第 11 节的大表里，`GGML_OP_CPY` 与 `GGML_OP_QUANTIZE` 两行都指向 `aclnnCast`（344 行）：`aclnn_cast` 是 `ggml_cann_cpy` 内部的类型转换路径，`GGML_OP_QUANTIZE` 本身没有独立 case。

## 场景（9 幕）

1. **CANN 后端：7 个文件、10058 行、两个公共入口** — 华为昇腾（Ascend）NPU 的 ggml 后端。七个文件分工明确：一个交契约、一个做分派、一个翻译算子。
2. **三张虚表：CANN 交给 ggml 的全部约定** — L3-01 拆过的三张表，这里是 CANN 的具体填法 —— 每张表就是一个 static const 变量。
3. **★ 算子不是内核，是一次厂商库调用** — CANN 后端不写 kernel。它只做两件事：问库"要多少 workspace"，然后"下发执行"。
4. **从 dst-&gt;op 到分派函数：一个 switch 的三种形态** — graph_compute 拿到一张图，逐个节点走 compute_forward。整个后端的"路由表"就是这一个 switch。
5. **★ ggml op -&gt; ACL 算子的映射清单** — 下表每一行的行号都能去源码核对：左边是 ggml 的算子身份，右边是华为库里那个算子的名字。
6. **ggml_tensor -&gt; aclTensor：两处约定差异** — ggml 的 ne[0] 是最内层、nb 以字节计；CANN 的最后一维是最内层、stride 以元素计。转换函数把三件事抹平。
7. **context 里的两样东西：内存池与 stream** — 一个 NPU 设备一个 context。8 条 stream 按需创建，一个内存池按设备能力选实现 —— 两者都是懒创建的。
8. **★ supports_op：厂商库没有的算子，后端只能说不** — 这张 switch 判断的不是"能不能算"，而是"ACL 里有没有对应的算子、这个数据类型库收不收"。
9. **把这一课压成一张表** — 一个 op 进 NPU 要走五步：分派 -> 转张量 -> 填参数 -> 问 workspace -> 下发到 stream。

## 核心结论

### ★ 算子由厂商库提供

CANN 后端 10058 行源码里没有矩阵乘、没有 softmax 的 kernel 实现。每个 op 最终都变成 `aclnn<名字>GetWorkspaceSize` + `aclnn<名字>(..., stream)` 两次调用，宏在 `aclnn_ops.h:922-934`。

| 对比项 | L6-01 CUDA 后端 | 本课 CANN 后端 |
|---|---|---|
| 算子实现在哪 | 本仓库的 `.cu` 文件 | 华为 CANN 库（预编译） |
| 加一个新算子 | 自己写 kernel | 等库提供，或退回 CPU |
| 覆盖度上界 | 作者愿意写多少 | ACL 算子库有多少 |
| 后端代码的职责 | 算法 + 调度 | 只有调度与翻译 |

### 五步链路（验收点的答案）

```text
1 分派      switch (dst->op) -> ggml_cann_xxx     ggml-cann.cpp:1773
2 转张量    ggml_cann_create_tensor -> aclCreateTensor   acl_tensor.cpp:53
3 填参数    aclnn 算子签名（dim / eps / alpha / ...）    aclnn_ops.cpp
4 问尺寸    aclnn<名>GetWorkspaceSize + CTX.pool()       aclnn_ops.h:927
5 下发      aclnn<名>(..., CTX.stream())                aclnn_ops.h:933
把关        supports_op == false 的 op 不进这条链        ggml-cann.cpp:2949
```

第 1 步用的是 L1-02 的算子身份；第 6 行用的是 L4-02 的切分判据。

### 覆盖度是借来的

`ggml_backend_cann_supports_op`（ggml-cann.cpp:2403）的 switch 里，每个 case 都要同时回答"ACL 有没有这个算子"和"这个数据类型库收不收"。兜底的 `default: return false;`（2705-2706 行）划出了后端的能力边界 —— **边界不在 llama.cpp 手里，在厂商库手里**。

## 验收点

- [ ] 保真门禁：20 个引用块全部逐字来自声明的源文件，且位置连续
- [ ] 覆盖度门禁：7 个源文件均被声明
- [ ] 参数门禁：无非法命令行参数
- [ ] 语法检查：通过
- [ ] 渲染门禁：9 幕，无 JS 错误、无布局溢出、交互可用
- [ ] 自检：不看代码，能说出 `struct ggml_tensor` 里哪些字段决定一次内核调用的形状

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
