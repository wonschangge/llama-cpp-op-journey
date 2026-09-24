# L7-06 · 小后端合集：OpenCL / WebGPU / MUSA / RPC / BLAS / ZenDNN / zDNN — 课件说明

> 层：**L7 · NPU 与加速器后端** ｜ 前置课：`L7-05`

## 学习目标

看完这一课，你应该能：

1. 说出这 7 个后端各自"最少要写什么"，并给出实测的文件数与行数。
2. 解释同一份 `ggml_backend_i` 契约如何容纳差别极大的数据面（宿主内存 / 设备显存 / 硬件专用布局 / 远端机器）。
3. 复述 RPC 后端把一个 tensor 送到远端的完整路径：序列化 → 打包 → socket → 服务端还原并写入远端内存。
4. 指出 RPC 设备在调度器眼里就是一块 GPU（`supports_op` 目前恒真），因而是 L4-02 图切分的普通参与者。

## 覆盖的源文件（29 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-blas.h` | 26 |
| `ggml/include/ggml-opencl.h` | 27 |
| `ggml/include/ggml-rpc.h` | 36 |
| `ggml/include/ggml-webgpu.h` | 20 |
| `ggml/include/ggml-zdnn.h` | 18 |
| `ggml/include/ggml-zendnn.h` | 23 |
| `ggml/src/ggml-blas/ggml-blas.cpp` | 531 |
| `ggml/src/ggml-musa/mudnn.cu` | 113 |
| `ggml/src/ggml-musa/mudnn.cuh` | 13 |
| `ggml/src/ggml-opencl/cl-program-cache.cpp` | 454 |
| `ggml/src/ggml-opencl/cl-program-cache.h` | 76 |
| `ggml/src/ggml-opencl/fa_tune.h` | 93 |
| `ggml/src/ggml-opencl/ggml-opencl.cpp` | 29503 |
| `ggml/src/ggml-opencl/libdl.h` | 80 |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 2375 |
| `ggml/src/ggml-rpc/transport-apple.cpp` | 482 |
| `ggml/src/ggml-rpc/transport-apple.h` | 28 |
| `ggml/src/ggml-rpc/transport.cpp` | 740 |
| `ggml/src/ggml-rpc/transport.h` | 39 |
| `ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp` | 3472 |
| `ggml/src/ggml-webgpu/ggml-webgpu.cpp` | 4888 |
| `ggml/src/ggml-webgpu/pre_wgsl.hpp` | 809 |
| `ggml/src/ggml-zdnn/common.hpp` | 60 |
| `ggml/src/ggml-zdnn/ggml-zdnn.cpp` | 639 |
| `ggml/src/ggml-zdnn/mmf.cpp` | 81 |
| `ggml/src/ggml-zdnn/mmf.hpp` | 13 |
| `ggml/src/ggml-zdnn/utils.cpp` | 80 |
| `ggml/src/ggml-zdnn/utils.hpp` | 20 |
| `ggml/src/ggml-zendnn/ggml-zendnn.cpp` | 839 |

> **说明**：正文表格里的行数按 `wc -l` 实测（例如 `ggml-blas.cpp` = 530 行）；README 的「覆盖的源文件」表由 `lessonkit` 生成，按换行符切分计数，每个文件会比 `wc -l` 多 1（文件末尾的换行也算一行）。
> **说明**：WebGPU 的 WGSL 内核文本来自构建期生成的 `ggml-wgsl-shaders.hpp`（源文件在 `ggml/src/ggml-webgpu/wgsl-shaders/*.wgsl`，由 `embed_wgsl.py` 嵌入），该生成头不在本课覆盖域内；本课引用的是使用它的 `ggml-webgpu-shader-lib.hpp`。
> **说明**：OpenCL 本身是构建期依赖（`ggml/src/ggml-opencl/CMakeLists.txt` 里 `find_package(OpenCL REQUIRED)`）；`libdl.h` 动态加载的是 Adreno 预编译内核库，不是 OpenCL 运行时。构建脚本不计入本课覆盖声明。
> **说明**：本课覆盖 29 个文件，全部进覆盖声明；`ggml/src/ggml-musa/CMakeLists.txt` 只在第一段讲解里被引用来说明"MUSA 为什么只有 2 个自有源文件"，不计入覆盖声明。
> **说明**：所有行数来自 `wc -l`，所有行号来自上游 `v0.5.0`（commit `7fe450e19305`）；引用一律由 `lessonkit` 按行号抽取，未手抄。

## 场景（9 幕）

1. **七个后端，一个契约** — 回顾 L3-01：ggml 不认识任何硬件，只认识 ggml_backend_i 这一堆函数指针。
2. **一张表量完七个后端** — 行数、文件数都来自 wc -l 实测；"要写什么"一句话来自逐文件阅读。
3. **BLAS：整个后端就是一个 switch** — 不新增数据面、不新增 buffer type —— 只是把两个 op 换成厂商 gemm。
4. **★ rpc_tensor：一个 tensor 在网线上的样子** — ggml_tensor 里有指针，不能直接过网；于是有一个逐字段对应的 POD 版本。
5. **把 tensor 送过去：打包 + 一条 socket** — 元数据 + 一个标志位 + 偏移 + 裸数据 = 一次 SET_TENSOR。整张图也走同一条路。
6. **WebGPU：内核是 WGSL 源码字符串** — 同一张虚表，但一半的槽是 NULL，且默认异步 —— 因为它跑在浏览器里。
7. **OpenCL：运行期加载、编译、缓存** — 29502 行里绝大部分是内核与派发；真正"后端骨架"的部分只有几千行。
8. **★ MUSA / ZenDNN / zDNN：翻译深度决定数据面** — 三个后端都在"把 ggml 的布局翻译成厂商库要的布局"，但翻译到哪一层，差别巨大。
9. **把这一课压成一张表** — 四种数据面，一套契约。下一课 L8-01 让一个 mul_mat 从模型一路走到内核。

## 核心结论

### 四种数据面，一套契约

把这一课的 7 个后端按"数据在哪、谁搬"归类，只有四种答案：

1. **零改动**：BLAS / ZenDNN —— buffer type 直接复用 CPU 的，只是把个别算子换成厂商库；
2. **设备内存**：WebGPU / OpenCL —— 数据在驱动管理的缓冲里，写入要过 `WriteBuffer` / `clEnqueueWriteBuffer`；
3. **硬件专用布局**：zDNN —— 每个 tensor 有自己的 `zdnn_ztensor`，写入时做 `zdnn_transform_ztensor`，`is_host = false`；
4. **远端机器**：RPC —— 数据在另一台机器上，tensor 与整张图都要序列化过网。

四种答案，填的是**同一张** `ggml_backend_i` / `ggml_backend_buffer_type_i` / `ggml_backend_device_i`。这正是 L3-01 那三张表的抽象力：契约不描述"数据在哪"，只描述"怎么问、怎么给、怎么算"。

### RPC：tensor 如何过网（验收点）

1. 客户端 `serialize_tensor`（`ggml-rpc.cpp:628`）把 `ggml_tensor` 填进定长的 `rpc_tensor`：形状/步长/op/参数原样复制，`buffer` 写远端句柄 `remote_ptr`（`:642`），`data` 写**本机指针值**（`:643`）；
2. `serialize_set_tensor`（`:701`）把它和 `cache_flag`、`offset`、裸数据拼成一块连续内存；
3. `dispatcher->send(RPC_CMD_SET_TENSOR, ...)`（`:740`）经 `socket_t::send_data`（`transport.cpp:603`）发出；大权重先用 FNV 哈希问远端是否已有（`:724-737`）；
4. 服务端 `recv_msg` 收下整块（`:1977`），`deserialize_tensor` 按元数据重建 tensor 并把 `data` 还原成远端地址（`:1415`），最后 `ggml_backend_tensor_set` 写入远端 buffer（`:1472`）。

计算走同一条路：`serialize_graph`（`:1003`）把节点与全部 tensor 发过去，服务端重建图并调用自己的 `ggml_backend_graph_compute`（`:1772`）。

### RPC 设备就是一块 GPU

`ggml_backend_rpc_device_get_type()` 返回 `GGML_BACKEND_DEVICE_TYPE_GPU`（`ggml-rpc.cpp:2174`），`supports_op` 目前**恒真**（`:2209-2213`，源码 TODO 写着应当去问远端）。所以 L4-02 的调度器会像对待普通 GPU 一样把子图切给它 —— `ggml_backend_rpc_add_server`（`ggml-rpc.h:31`）让一台远端服务器变成一个可注册的设备。

## 验收点

- [x] 保真门禁：34 处引用 —— 34 个引用块 / 46 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 29 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 3 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
