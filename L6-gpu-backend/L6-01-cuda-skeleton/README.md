# L6-01 · CUDA 后端骨架：一个 switch + 一个 stream — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L5-05`

## 学习目标

看完这一课，你应该能：

1. 说出 CUDA 后端的三张虚表分别在哪个文件、哪一行，以及 `graph_plan_*` 四个槽为什么是 NULL；
2. 解释 `stream(device, stream)` 为什么是懒创建，以及 `cudaStreamNonBlocking` 对并发分支的意义；
3. 说明 CUDA graph 在什么条件下才被启用（预热、兼容性、架构门槛）；
4. **（验收点）**说出 `ggml_backend_cuda_device_supports_op()` 决定一个 op 支持的判据，并指出它与图切分（L4-02）的关系；
5. 说明 `ggml_cuda_compute_forward()` 的 `default: return false` 与 4359 处断言的因果关系。

## 覆盖的源文件（3 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-cuda.h` | 48 |
| `ggml/src/ggml-cuda/common.cuh` | 1718 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5857 |

> **说明**：本课覆盖三个文件：`ggml/include/ggml-cuda.h`、`ggml/src/ggml-cuda/common.cuh`、`ggml/src/ggml-cuda/ggml-cuda.cu`，均计入覆盖率。
> **说明**：第 7 幕引用的 CPU 侧对照文件 `ggml/src/ggml-cpu/ggml-cpu.c` 只在 caption 里指路，本课不引用其源码，故不计入本课覆盖率。
> **说明**："88 个 case"的统计口径：`ggml-cuda.cu` 中 `ggml_cuda_compute_forward()` 内（2067-2432 行）缩进为 8 空格的 `case GGML_OP_*:` 标签，`grep -c` 得 88，去重后仍是 88。

## 场景（10 幕）

1. **CUDA 后端 = 四个 C 入口 + 三张虚表** — 公共头只声明入口；真正的约定在 common.cuh 的上下文和 ggml-cuda.cu 的三张虚表里。
2. **后端对象：一个 device，一组 stream，一组池** — struct ggml_backend_cuda_context 就是"CUDA 后端"这个对象的全部状态。
3. **stream 是懒创建的，而且不是 legacy 默认流** — 没有"新建 stream"这种后端接口 —— stream 属于上下文，随用随建，随上下文一起销毁。
4. **三张虚表里的第一张：backend 虚表** — 16 个槽里有 4 个是 NULL —— CUDA 不用 ggml 的图计划接口，它走自己的 capture。
5. **图不是第一次就 capture：先预热，再录，之后只 launch** — 同一张图连续两次"属性没变"，才值得录成 CUDA graph —— 因为图的地址和形状都被录死了。
6. **一次遍历：跳过、融合、然后交给 compute_forward** — 遍历 cgraph->n_nodes 的下标 i —— CUDA 后端的执行就是一个线性遍历。
7. **★ 一个 switch：88 个 case 就是 CUDA 的 op 全表** — ggml_cuda_compute_forward() 的全部内容：switch (dst->op)，每个 case 调一个 ggml_cuda_* 实现。
8. **设备属性：cc 是一个整数，能力判断就是比大小** — ggml_cuda_device_info 把每个设备的能力缓存下来；后面所有"这个卡行不行"都是读它。
9. **★ supports_op：能力号 + dtype 决定切不切给 CUDA** — 它只回 true / false，不加解释 —— 但 L4-02 的图切分完全建立在这一个函数上。
10. **把 CUDA 后端压成一张表** — 接口 → 上下文 → 图 → 分派 → 判据 → 显存：六件事，一条链。

## 核心结论

### ★ 骨架 = 一个 switch + 一个 stream

CUDA 后端的执行面只有两件事：

| 事 | 在哪 | 依据什么 |
|---|---|---|
| op 怎么跑 | `ggml_cuda_compute_forward()`（2067-2432，88 个 case） | `dst->op` 一个字段 |
| 活儿发到哪条流 | `ggml_backend_cuda_context::stream()`（common.cuh:1528） | `curr_stream_no` |

其余全是为这两件事服务的：虚表定契约、设备属性定能力、pool 定临时显存、capture 定图怎么重放。

### ★ 支持与否由 supports_op 决定，不由执行路径决定

`ggml_backend_cuda_device_supports_op()`（5132-5591）按顺序判五件事：

| 判据 | 位置 | 典型出局条件 |
|---|---|---|
| 数据位置 | 5136-5143 | src 在别的设备的 CUDA buffer |
| op 白名单 | 5145 起的 switch | 名单外落到 5588 `default: return false` |
| dtype 白名单 | 5191 起 `switch (a->type)` | 不在列表的权重类型 |
| 布局 / 参数 | 5196 / 5202 | `nb[0] != element_size`、MUL_MAT_ID + F32 精度 |
| 设备能力号 | 5206 / 5574 | cc 不满足该算子的 `*_supported()` |

返回 false 的节点会被 L4-02 的调度器切给别的后端 —— **这就是"支持与否"直接决定图切分结果**。

### 执行路径没有兜底

`ggml_cuda_compute_forward()` 的 `default` 只做一件事：`return false`（2415-2416）。调用方（4355-4359）把它转成 `GGML_ASSERT(ok)`。所以能在执行期到达的 op，一定是 supports_op 已经放行过的 op。

## 验收点

- [x] 保真门禁：16 处引用 —— 16 个引用块 / 59 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 3 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
