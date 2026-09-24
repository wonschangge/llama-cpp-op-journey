# L7-05 · VirtGPU 后端：虚拟化 GPU — 课件说明

> 层：**L7 · NPU 与加速器后端** ｜ 前置课：`L7-04`（ExecuTorch 后端）；建议先看 `L1-01`（张量的数据面）、`L3-01`（后端契约与虚表）、`L4-03`（buffer 与 buffer type）

## 学习目标

看完这一课，你应该能：

1. 说出一台客户机里的 tensor，数据是怎么到宿主 GPU 的（对应验收点）：哪一行做 memcpy、命令里带的是什么、宿主怎么换回指针；
2. 解释虚拟化下为什么不能直接把设备内存映射进客户机，以及 `tensor->data` 在这里变成了什么；
3. 列出 APIR 的 23 条命令与四个分组，并说出一次远程调用的固定前缀是哪三个字段；
4. 说出这个后端用什么做同步，以及为什么它在 `ggml_backend_i` 上没有 synchronize / event 接口；
5. 指出 39 个文件分别在墙的哪一侧（客户机 18 / 宿主 11 / 共用 6 / 工具 4）。

## 覆盖的源文件（39 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-virtgpu.h` | 15 |
| `ggml/src/ggml-virtgpu/apir_cs_ggml-rpc-front.cpp` | 88 |
| `ggml/src/ggml-virtgpu/backend/apir_cs_ggml-rpc-back.cpp` | 116 |
| `ggml/src/ggml-virtgpu/backend/backend-convert.h` | 14 |
| `ggml/src/ggml-virtgpu/backend/backend-dispatched-backend.cpp` | 103 |
| `ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer-type.cpp` | 106 |
| `ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer.cpp` | 180 |
| `ggml/src/ggml-virtgpu/backend/backend-dispatched-device.cpp` | 150 |
| `ggml/src/ggml-virtgpu/backend/backend-dispatched.cpp` | 52 |
| `ggml/src/ggml-virtgpu/backend/backend-dispatched.gen.h` | 74 |
| `ggml/src/ggml-virtgpu/backend/backend-dispatched.h` | 28 |
| `ggml/src/ggml-virtgpu/backend/backend-virgl-apir.h` | 33 |
| `ggml/src/ggml-virtgpu/backend/backend.cpp` | 145 |
| `ggml/src/ggml-virtgpu/backend/shared/api_remoting.h` | 96 |
| `ggml/src/ggml-virtgpu/backend/shared/apir_backend.gen.h` | 95 |
| `ggml/src/ggml-virtgpu/backend/shared/apir_backend.h` | 51 |
| `ggml/src/ggml-virtgpu/backend/shared/apir_cs.h` | 379 |
| `ggml/src/ggml-virtgpu/backend/shared/apir_cs_ggml.h` | 233 |
| `ggml/src/ggml-virtgpu/backend/shared/apir_cs_rpc.h` | 59 |
| `ggml/src/ggml-virtgpu/ggml-backend-buffer-type.cpp` | 82 |
| `ggml/src/ggml-virtgpu/ggml-backend-buffer.cpp` | 124 |
| `ggml/src/ggml-virtgpu/ggml-backend-device.cpp` | 161 |
| `ggml/src/ggml-virtgpu/ggml-backend-reg.cpp` | 214 |
| `ggml/src/ggml-virtgpu/ggml-backend.cpp` | 73 |
| `ggml/src/ggml-virtgpu/ggml-remoting.h` | 72 |
| `ggml/src/ggml-virtgpu/include/apir_hw.h` | 10 |
| `ggml/src/ggml-virtgpu/virtgpu-apir.h` | 16 |
| `ggml/src/ggml-virtgpu/virtgpu-forward-backend.cpp` | 59 |
| `ggml/src/ggml-virtgpu/virtgpu-forward-buffer-type.cpp` | 111 |
| `ggml/src/ggml-virtgpu/virtgpu-forward-buffer.cpp` | 174 |
| `ggml/src/ggml-virtgpu/virtgpu-forward-device.cpp` | 195 |
| `ggml/src/ggml-virtgpu/virtgpu-forward-impl.h` | 37 |
| `ggml/src/ggml-virtgpu/virtgpu-forward.gen.h` | 55 |
| `ggml/src/ggml-virtgpu/virtgpu-shm.cpp` | 100 |
| `ggml/src/ggml-virtgpu/virtgpu-shm.h` | 24 |
| `ggml/src/ggml-virtgpu/virtgpu-utils.cpp` | 180 |
| `ggml/src/ggml-virtgpu/virtgpu-utils.h` | 87 |
| `ggml/src/ggml-virtgpu/virtgpu.cpp` | 546 |
| `ggml/src/ggml-virtgpu/virtgpu.h` | 116 |

> **说明**：本课声明 39 个源文件 = 计划里 L7-05 的全部：`ggml/src/ggml-virtgpu/` 下 38 个 加 `ggml/include/ggml-virtgpu.h`，**无一遗漏**。其中 13 个被逐字引用：核心 8 个（`virtgpu.cpp`、`virtgpu.h`、`virtgpu-shm.cpp`、`virtgpu-forward-buffer.cpp`、`backend/backend-dispatched-buffer.cpp`、`apir_cs_ggml-rpc-front.cpp`、`backend/apir_cs_ggml-rpc-back.cpp`、`virtgpu-forward-impl.h`），协议与公共头 5 个（`backend/shared/api_remoting.h`、`backend/shared/apir_cs_rpc.h`、`backend/shared/apir_backend.gen.h`、`backend/backend-dispatched.gen.h`、`ggml/include/ggml-virtgpu.h`）；其余 26 个在第十五节按族说明。
> **说明**：**覆盖域外、不计入覆盖率**的文件（本课只在文字里提及，后缀不在本视角的覆盖域内，故不进覆盖声明）：`ggml/src/ggml-virtgpu/ggmlremoting_functions.yaml`（166 行，远程函数的 YAML 定义）、`ggml/src/ggml-virtgpu/regenerate_remoting.py`（333 行，由 YAML 生成三个 `*.gen.h`）、`ggml/src/ggml-virtgpu/CMakeLists.txt` 与 `ggml/src/ggml-virtgpu/backend/CMakeLists.txt`（两个库的构建定义）。
> **说明**：宿主机侧的库并不在本仓库内被链接成一个可运行程序：它是被 virglrenderer 用 `dlopen` 加载的（`backend/backend.cpp:76`），加载路径来自 hypervisor 的配置。本课不实测这条链路，只做源码级断言。
> **说明**：本课不讨论任何命令行参数；参数门禁的真值集来自真实二进制的帮助输出。

## 场景（9 幕）

1. **一台 VM 里的 llama.cpp，GPU 在墙的另一边** — 39 个文件分成四族。看源码之前，先记住一句话：数据只能搬过去，不能指过去。
2. **★ 指针退化成偏移：过墙前先减掉 buffer base** — 其它后端把 tensor->data 直接交给内核去解引用；这里同一个字段必须先变成一个相对量。
3. **墙的另一边：偏移 + 宿主的 buffer_start = 真指针** — 宿主机反解出指针，并且顺手做了隔着一堵墙唯一还能做的地址校验。
4. **23 条命令：整个后端就是一张枚举表** — ggml 的 device / buffer_type / buffer / backend 四个面，被压成 0..22 号命令。
5. **一条命令的固定前缀：类型 / 标志 / 回复窗口** — 所有命令共用同一个编码器，前三段是固定的 —— 第三条告诉宿主把回复写到哪里。
6. **提交与等待：零个 fence，一个 atomic 计数器** — execbuffer 的三个同步字段全是 0；回复靠轮询 reply 窗口的第一个 4 字节。
7. **★ 数据只能搬过去：一次 memcpy + 一条命令** — set_tensor 的全部动作：挑窗口、上锁、把字节拷进客户机自己 mmap 到的内存、把 res_id 放进命令。
8. **窗口是怎么造出来的：一个 HOST3D blob + 一次 mmap** — 客户机向 DRM 申请一块可映射的 blob，拿到 res_id 与一个本地指针；res_id 就是这块内存在墙那边的名字。
9. **墙的两侧，各管一段** — 客户机侧实现 ggml 的契约并把调用翻译成命令；宿主侧把命令翻译回 ggml 的调用。

## 核心结论

### ★ 虚拟化拿掉了"指针"这个前提

其它 16 个后端都建立在同一个前提上：`tensor->data` 是一个本机可访问的地址。在 ggml-virtgpu 里这个前提不成立 —— 设备内存在宿主机上，客户机里没有它的映射。

| 面 | 其它后端 | ggml-virtgpu |
|---|---|---|
| `tensor->data` | 设备指针，内核直接解引用 | **相对 buffer base 的偏移** |
| 数据搬运 | 内核自己读 | 客户机 memcpy 进共享窗口，宿主再写设备 |
| 地址校验 | 内核侧按指针校验 | 校验**字节区间**是否落在 buffer 内 |

这是唯一一个从根上改变数据面的后端：**搬的是字节，不是映射；传的是偏移，不是指针。**（L1-01 说 `data` 回答"数据在哪"；这里它只能回答"第几字节"。）

### 数据面 = 共享内存窗口 + 命令流

一条 `set_tensor` 的完整路径：

```text
客户机 ggml_backend_remoting_buffer_set_tensor  (ggml-backend-buffer.cpp:16)
  -> apir_buffer_set_tensor                     (virtgpu-forward-buffer.cpp:22)
       选窗口 -> mtx_lock -> memcpy(mmap_ptr, data, size)   (:41-53)
       命令里放 res_id + offset + size                     (:54-57)
  -> remote_call -> DRM_IOCTL_VIRTGPU_EXECBUFFER           (virtgpu.cpp:470)
     ... hypervisor ...
宿主 backend_buffer_set_tensor                  (backend-dispatched-buffer.cpp:35)
       get_shmem_ptr(ctx_id, res_id)                       (:64)
       buffer->iface.set_tensor(...)  -> 真后端 -> 设备内存  (:71)
```

对照 L3-01：客户机侧实现的仍然是同一套 `ggml_backend_*_i` 契约；对照 L4-03：`buffer` 这个概念没有变，变的只是"buffer 在哪、谁能碰它"。

再对照计划里的 L7-02（Hexagon）：那一课的要点是 host 侧与 DSP 侧的边界，"跨"发生在同一台机器内部；这里跨的是虚拟化特权边界，所以客户机连设备内存的地址都拿不到，只剩偏移与句柄。

### 同步 = 一个 atomic 计数器

`execbuffer` 的 `fence_fd`、`num_in_syncobjs`、`num_out_syncobjs` 全为 0，context 初始化也显式关掉了 fence 事件。回复靠轮询共享回复窗口的第一个 4 字节（`atomic_uint` + acquire 语义），没到就睡 15µs 再看。

所以这个后端在 `ggml_backend_i` 上不需要 `synchronize` / `event_record` / `event_wait`：**每次远程调用本来就是同步的**。这与 L4-04 讲的异步后端是两种相反的取舍。

### 可序列化是这一课的隐含约束

跨墙的 tensor 必须能被压成定长 POD `apir_rpc_tensor`；带 `extra` 的 tensor 在 `apir_encode_ggml_tensor_inline()` 里会直接 abort（`apir_cs_ggml.h:177-179`）。但 `nb[]` 是原样带过去的，所以**步长自由的 tensor 依然可以过墙** —— 被拿掉的是地址，不是布局。整张计算图也走同一条路：序列化成一个字节缓冲，一次 `GRAPH_COMPUTE` 送过去。

## 验收点

- [x] 保真门禁：23 处引用 —— 23 个引用块 / 43 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 39 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
