# L7-02 · Hexagon 后端（Qualcomm NPU/DSP）：host 与 DSP 的边界 — 课件说明

> 层：**L7 · NPU 与加速器后端** ｜ 前置课：`L7-01`（CANN 后端）

## 学习目标

看完这一课，你应该能：

1. 说出 **host 侧与 DSP 侧各自负责什么**（本课验收点）：各自读哪几个文件、做哪几件事；
2. 解释为什么这个后端的接口里必须有 `rpcmem_alloc2` / `rpcmem_to_fd` / `fastrpc_mmap` 这三步，而同机的 GPU 后端不需要；
3. 说明 `dspqueue` 在 host 与 DSP 两侧各是什么形态（`dspqueue_t` 与 `queue_id`），以及一次 `dspqueue_write` 携带了什么；
4. 描述一次算子在 DSP 上的完整路径：解包 → cache 维护 → 地址重定位 → `execute_op` → 多线程 → 栅栏 → 回写响应。

## 覆盖的源文件（87 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | 8071 |
| `ggml/src/ggml-hexagon/htp-drv.cpp` | 426 |
| `ggml/src/ggml-hexagon/htp-drv.h` | 124 |
| `ggml/src/ggml-hexagon/htp-opnode.h` | 396 |
| `ggml/src/ggml-hexagon/libdl.h` | 80 |
| `ggml/include/ggml-hexagon.h` | 20 |
| `ggml/src/ggml-hexagon/htp/main.c` | 1361 |
| `ggml/src/ggml-hexagon/htp/act-ops.c` | 667 |
| `ggml/src/ggml-hexagon/htp/allreduce-ops.c` | 458 |
| `ggml/src/ggml-hexagon/htp/allreduce-ops.h` | 52 |
| `ggml/src/ggml-hexagon/htp/argsort-ops.c` | 1096 |
| `ggml/src/ggml-hexagon/htp/binary-ops.c` | 1186 |
| `ggml/src/ggml-hexagon/htp/binary-ops.h` | 112 |
| `ggml/src/ggml-hexagon/htp/concat-ops.c` | 333 |
| `ggml/src/ggml-hexagon/htp/cpy-ops.c` | 522 |
| `ggml/src/ggml-hexagon/htp/cumsum-ops.c` | 284 |
| `ggml/src/ggml-hexagon/htp/diag-ops.c` | 247 |
| `ggml/src/ggml-hexagon/htp/fill-ops.c` | 147 |
| `ggml/src/ggml-hexagon/htp/flash-attn-ops.c` | 2566 |
| `ggml/src/ggml-hexagon/htp/flash-attn-ops.h` | 394 |
| `ggml/src/ggml-hexagon/htp/gated-delta-net-ops.c` | 2451 |
| `ggml/src/ggml-hexagon/htp/gated-delta-net-ops.h` | 302 |
| `ggml/src/ggml-hexagon/htp/get-rows-ops.c` | 296 |
| `ggml/src/ggml-hexagon/htp/get-rows-ops.h` | 78 |
| `ggml/src/ggml-hexagon/htp/im2col-ops.c` | 587 |
| `ggml/src/ggml-hexagon/htp/matmul-ops.c` | 4476 |
| `ggml/src/ggml-hexagon/htp/matmul-ops.h` | 831 |
| `ggml/src/ggml-hexagon/htp/pad-ops.c` | 576 |
| `ggml/src/ggml-hexagon/htp/repeat-ops.c` | 170 |
| `ggml/src/ggml-hexagon/htp/roll-ops.c` | 317 |
| `ggml/src/ggml-hexagon/htp/rope-ops.c` | 820 |
| `ggml/src/ggml-hexagon/htp/rope-ops.h` | 63 |
| `ggml/src/ggml-hexagon/htp/set-rows-ops.c` | 261 |
| `ggml/src/ggml-hexagon/htp/set-rows-ops.h` | 75 |
| `ggml/src/ggml-hexagon/htp/softmax-ops.c` | 615 |
| `ggml/src/ggml-hexagon/htp/softmax-ops.h` | 107 |
| `ggml/src/ggml-hexagon/htp/solve-tri-ops.c` | 300 |
| `ggml/src/ggml-hexagon/htp/ssm-conv.c` | 497 |
| `ggml/src/ggml-hexagon/htp/ssm-conv.h` | 41 |
| `ggml/src/ggml-hexagon/htp/sum-rows-ops.c` | 155 |
| `ggml/src/ggml-hexagon/htp/unary-ops.c` | 1711 |
| `ggml/src/ggml-hexagon/htp/unary-ops.h` | 179 |
| `ggml/src/ggml-hexagon/htp/htp-ctx.h` | 178 |
| `ggml/src/ggml-hexagon/htp/htp-fence.h` | 90 |
| `ggml/src/ggml-hexagon/htp/htp-ops.h` | 260 |
| `ggml/src/ggml-hexagon/htp/htp-tensor.c` | 361 |
| `ggml/src/ggml-hexagon/htp/htp-tensor.h` | 144 |
| `ggml/src/ggml-hexagon/htp/htp-vtcm.h` | 20 |
| `ggml/src/ggml-hexagon/htp/hvx-arith.h` | 726 |
| `ggml/src/ggml-hexagon/htp/hvx-base.h` | 293 |
| `ggml/src/ggml-hexagon/htp/hvx-copy.h` | 263 |
| `ggml/src/ggml-hexagon/htp/hvx-div.h` | 292 |
| `ggml/src/ggml-hexagon/htp/hvx-dump.h` | 130 |
| `ggml/src/ggml-hexagon/htp/hvx-exp.h` | 256 |
| `ggml/src/ggml-hexagon/htp/hvx-fa-kernels.h` | 304 |
| `ggml/src/ggml-hexagon/htp/hvx-flash-attn.h` | 48 |
| `ggml/src/ggml-hexagon/htp/hvx-floor.h` | 101 |
| `ggml/src/ggml-hexagon/htp/hvx-inverse.h` | 211 |
| `ggml/src/ggml-hexagon/htp/hvx-log.h` | 119 |
| `ggml/src/ggml-hexagon/htp/hvx-mm-kernels-float.h` | 383 |
| `ggml/src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h` | 1056 |
| `ggml/src/ggml-hexagon/htp/hvx-norm.h` | 455 |
| `ggml/src/ggml-hexagon/htp/hvx-pow.h` | 43 |
| `ggml/src/ggml-hexagon/htp/hvx-quant.h` | 166 |
| `ggml/src/ggml-hexagon/htp/hvx-reduce.h` | 337 |
| `ggml/src/ggml-hexagon/htp/hvx-repl.h` | 75 |
| `ggml/src/ggml-hexagon/htp/hvx-scale.h` | 200 |
| `ggml/src/ggml-hexagon/htp/hvx-sigmoid.h` | 182 |
| `ggml/src/ggml-hexagon/htp/hvx-sin-cos.h` | 79 |
| `ggml/src/ggml-hexagon/htp/hvx-sqrt.h` | 190 |
| `ggml/src/ggml-hexagon/htp/hvx-types.h` | 37 |
| `ggml/src/ggml-hexagon/htp/hvx-utils.h` | 25 |
| `ggml/src/ggml-hexagon/htp/hmx-fa-kernels.h` | 706 |
| `ggml/src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h` | 1408 |
| `ggml/src/ggml-hexagon/htp/hmx-queue.c` | 171 |
| `ggml/src/ggml-hexagon/htp/hmx-queue.h` | 161 |
| `ggml/src/ggml-hexagon/htp/hmx-utils.h` | 223 |
| `ggml/src/ggml-hexagon/htp/dma-queue.c` | 213 |
| `ggml/src/ggml-hexagon/htp/dma-queue.h` | 516 |
| `ggml/src/ggml-hexagon/htp/work-queue.c` | 245 |
| `ggml/src/ggml-hexagon/htp/work-queue.h` | 39 |
| `ggml/src/ggml-hexagon/htp/hex-bitmap.h` | 25 |
| `ggml/src/ggml-hexagon/htp/hex-common.h` | 90 |
| `ggml/src/ggml-hexagon/htp/hex-dump.h` | 87 |
| `ggml/src/ggml-hexagon/htp/hex-fastdiv.h` | 38 |
| `ggml/src/ggml-hexagon/htp/hex-profile.h` | 65 |
| `ggml/src/ggml-hexagon/htp/hex-utils.h` | 65 |

> **说明**：本课覆盖域 = `ggml/include/ggml-hexagon.h`（1）+ `ggml/src/ggml-hexagon/` 下全部源文件（86）= **87**，与 `python3 tools/plan_matrix.py --files` 给出的 L7-02 清单逐条一致，全部计入覆盖率。
> **说明**：`ggml/src/ggml-hexagon/htp/htp_iface.idl`、`htp/CMakeLists.txt`、`htp/cmake-toolchain.cmake` **不在覆盖域内**（后缀不在 `SOURCE_SUFFIXES` 里），本课只在散文里提到它们的作用，不计入覆盖率，也没有为它们做 `cover` 声明。
> **说明**：上游位置：`ggml/src/ggml-hexagon/`、`ggml/include/ggml-hexagon.h`，v0.5.0（commit `7fe450e19305`）。`ggml-hexagon.cpp` 8070 行、`htp/main.c` 1360 行（`wc -l` 计数；README 上方表格里的行数由 lessonkit 按换行切分统计，比 `wc -l` 多 1）。`htp/` 目录共 84 个文件，其中 29 个 `.c` + 52 个 `.h` 计入覆盖域，另有 `CMakeLists.txt`、`cmake-toolchain.cmake`、`htp_iface.idl` 三个非源文件不计入。本课不试图覆盖这些文件的全部内容。

## 场景（9 幕）

1. **一个后端，两个处理器** — Hexagon 后端横跨 host（ARM/x86）与 DSP（Hexagon HTP）两颗核；本课覆盖的 87 个文件分属两边。
2. **注册：HTP 就是一个普通 ggml 后端** — 名字叫 "HTP"，虚表结构和别的后端一模一样 —— 特殊之处全在背后。
3. **★ host 侧的会话：句柄、队列、队列号 —— 接口不止是"提交 kernel"** — 一个 session 里同时住着"远端句柄"和"命令队列"两个东西，这在同一颗芯片的 GPU 后端里是看不到的。
4. **远端内存：rpcmem → fd → fastrpc_mmap** — DSP 看不到 host 的 malloc 堆，host 也递不过去一个指针 —— 唯一的通行证是「文件描述符」。
5. **对岸：dspqueue_import 与一个连续块** — DSP 侧入口 htp_iface_start() 收到队列号，导入同一条队列，然后把线程池、DMA 队列、VTCM 一次性摆好。
6. **一次 dspqueue_write 里装了什么** — 请求信封是一个定长头 + 紧跟其后的三个数组；DSP 侧用同一个公式把它切回来。
7. **HTP 图 = ggml 图的翻译结果，不是另一张图** — 每个 ggml 节点被包成一个 htp_opnode：换成 HTP opcode、挂上预计算的 kernel 参数、接上融合链。
8. **★ DSP 内部：线程池 + 栅栏** — 一个 batch 在 DSP 上被切成 n_threads 份并行跑；跑完靠一个"序号 + 状态"的栅栏对齐。
9. **把边界画成一张表：host 管什么、DSP 管什么** — 验收点：看完这一课，应该能不看代码就把这张表说出来。

## 核心结论

### ★ 跨两个处理器，所以接口不止是"提交 kernel"

Hexagon 后端的算力在**另一颗处理器**（DSP/HTP）上，host 与 DSP 不共享地址空间、也不共享 cache。因此它的接口必须自己搭出三样东西：

| 需要什么 | 这个后端怎么做 | 源码依据 |
|---|---|---|
| 跨域内存 | `rpcmem_alloc2` → `rpcmem_to_fd` → `fastrpc_mmap` | `ggml-hexagon.cpp:586` / `:627` |
| 命令队列 | `dspqueue_create` → `dspqueue_export` → DSP `dspqueue_import` | `ggml-hexagon.cpp:3964` / `main.c:387` |
| 显式同步 | `htp_fence_write` / `htp_fence_read` + `syncht` + cache clean | `htp-fence.h:17` |

同机的 GPU 后端（L6 系）把这三件事都交给了驱动与统一虚拟地址，所以它的接口看上去只是"提交 kernel"。**这就是本课与 L6-06 的分水岭。**

### host 侧与 DSP 侧的职责划分

| 侧 | 负责 | 不负责 | 代表源码 |
|---|---|---|---|
| host | 注册与驱动加载、建会话、分/映射内存、把 ggml 图翻成 htp_opnode、预计算 kernel_params、repack 权重、攒批并 `dspqueue_write` | 不做任何数值计算 | `ggml-hexagon.cpp:8045` / `:3964` / `:6400` / `:3434` |
| DSP | 实现 `htp_iface_*`、导入队列、申请 VTCM、建线程池与 DMA 队列、解包 batch、cache 维护、地址重定位、`execute_op` 分发、栅栏、回写响应 | 不做内存分配策略、不做图级优化 | `main.c:366` / `:387` / `:1149` / `:810` / `htp-fence.h:17` |

### 一次算子在 Hexagon 上的完整路径

```text
host: ggml 图 -> op_is_compute 过滤 -> htp_opnode(opcode, kernel_params, fused)
    -> mmap_tensor -> op_batch.add_op -> 满了 -> opqueue.push(三个数组)
    -> dspqueue_write(queue, req, buf)

DSP : dspqueue_read_noblock -> process_opbatch
    -> 整批 qurt_mem_cache_clean + hex_l2fetch_block
    -> prep_op_bufs(复用/新建 mmap) -> prep_tensors(偏移 -> 虚拟地址)
    -> execute_op 大 switch -> op_*() -> work_queue_run(n_threads)
    -> work_queue_suspend -> 刷 cache -> htp_mdev_group_barrier
    -> dspqueue_write(响应)

host: opqueue.pop -> 校验 seq / status -> 下一个 batch
```

注意这条链上**没有一步是"把数据拷贝过去"**：内存一开始就是共享的（`rpcmem`），过边界的只有描述符。这是这个后端能在 DSP 上跑大模型的前提。

## 验收点

- [x] 保真门禁：25 处引用 —— 25 个引用块 / 56 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 87 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 4 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
