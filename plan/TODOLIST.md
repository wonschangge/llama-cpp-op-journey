# 分层课程计划 —— llama.cpp 算子之旅

> **视角**：一个算子如何从模型定义出发，落到 CPU / GPU / NPU 等后端上真正执行。
> **上游**：`ggml-org/llama.cpp` @ `v0.5.0` (`7fe450e19305`)

**总课数 52　覆盖域 1290 个源文件**（核心层 330 + 后端层 960）

勾选状态由 `tools/plan_matrix.py --plan` 依据磁盘上四个文件是否齐备自动生成，不靠肉眼。

---

## L1 · 算子的表示

> ggml 用什么统一表示一个算子：张量、op 枚举、图
>
> 6 课，覆盖 20 个源文件。

- [x] **`L1-01`** ggml 张量：算子的数据面　`P0`　1 文件
  - 讲解要点：张量结构体字段、数据类型枚举、维度与步长约定；为什么 ggml 用 type 而非 dtype 表达量化
  - 验收点：能说出 ggml_tensor 里哪些字段决定一次内核调用的形状
  - 目录：`L1-operator-representation/L1-01-tensor-data-plane/`
- [x] **`L1-02`** 算子枚举与元数据：算子的身份　`P0`　3 文件
  - 讲解要点：GGML_OP_* 枚举、GGML_OP_NAME / GGML_OP_SYMBOL 两张身份表、输入个数由构造器赋值决定（实测无 nargs 表）、输出形状规则在构造器内、op_params 槽位随 op 变、ggml_op_is_empty 的分类用法
  - 验收点：给出任意一个 GGML_OP_*，能说出它的输入个数与输出形状规则在哪定义
  - 目录：`L1-operator-representation/L1-02-op-enum-and-nargs/`
- [x] **`L1-03`** 计算图与拓扑遍历　`P0`　3 文件
  - 讲解要点：ggml_cgraph、view_src/view_offs、ggml_build_forward_expand 的拓扑排序、visited_hash_set
  - 验收点：能解释为什么拓扑排序用 hash set 而不是标记位
  - 目录：`L1-operator-representation/L1-03-graph-and-toposort/`
- [x] **`L1-04`** 量化块结构：算子内层的压缩数据　`P0`　5 文件
  - 讲解要点：block_q4_0 等块布局（实测：Q4_0=18B / Q4_1=20B / Q8_0=34B / Q4_K=144B / Q6_K=210B）、scale/min 的表示、反量化函数族、ggml.c 里的 ggml_type_traits 表如何绑定类型与块大小
  - 验收点：能画出 Q4_0 与 Q4_K 的块内存布局差异，并说出对内核的影响
  - 目录：`L1-operator-representation/L1-04-quant-block-layout/`
- [x] **`L1-05`** 上下文、线程与优化器接口　`P0`　6 文件
  - 讲解要点：ggml_context 的 arena 对象分配（实测在 ggml.c）、全局临界区、ggml-opt 的训练接口
  - 验收点：能说明 ggml_context 为什么用 arena 而非逐个 malloc
  - 目录：`L1-operator-representation/L1-05-context-threads-opt/`
- [x] **`L1-06`** GGUF：算子的权重从文件来　`P0`　2 文件
  - 讲解要点：GGUF 容器格式、KV 段、张量信息段、对齐、读写 API 与错误处理
  - 验收点：能按字节说出 GGUF 文件头到张量数据之间的布局
  - 目录：`L1-operator-representation/L1-06-gguf-container/`

## L2 · 从模型到图

> 权重与超参如何被翻译成待执行的 ggml 图
>
> 15 课，覆盖 224 个源文件。

- [x] **`L2-01`** llama.h 公共 API 全景　`P0`　4 文件
  - 讲解要点：模型/上下文/采样三段式 API、参数结构体族、llama.cpp 作为聚合入口的地位
  - 验收点：能画出从 llama_model_load 到 llama_decode 的 API 调用顺序图
  - 目录：`L2-model-to-graph/L2-01-llama-h-api/`
- [x] **`L2-02`** 架构表与超参：模型长什么样的元数据　`P0`　6 文件
  - 讲解要点：LLM_ARCH_* 枚举、张量命名表、hparams 与 cparams 的分工
  - 验收点：能解释新增一个模型架构需要在架构表里加哪几项
  - 目录：`L2-model-to-graph/L2-02-arch-table-hparams/`
- [x] **`L2-03`** 权重加载与内存映射　`P0`　6 文件
  - 讲解要点：mmap vs read、后端 buffer type 的选择时机、张量落位（哪个后端持有权重）
  - 验收点：能说出权重张量在什么时刻决定自己被分配到哪个后端
  - 目录：`L2-model-to-graph/L2-03-weight-loading-mmap/`
- [x] **`L2-04`** KV cache 与记忆家族　`P0`　23 文件
  - 讲解要点：KV cache 的 cell 抽象、iswa/dsa/msa/dsv4 变体、recurrent 与 hybrid 记忆
  - 验收点：能解释为什么需要 memory 抽象而不是一个统一的 KV cache
  - 目录：`L2-model-to-graph/L2-04-kv-cache-and-memory/`
- [x] **`L2-05`** 批、解码参数与模型装配　`P0`　6 文件
  - 讲解要点：llama_batch 的 token 序列结构、ubatch 切分、模型构造与设备分配
  - 验收点：能说明一个 batch 是如何被切成 ubatch 并影响图形态的
  - 目录：`L2-model-to-graph/L2-05-batch-and-model-build/`
- [x] **`L2-06`** ★ 计算图骨架 llama-graph　`P0`　2 文件
  - 讲解要点：llm_graph_context、build_* 原语族（build_norm/build_attn/build_ffn/build_moe_ffn）
  - 验收点：能说出 build_moe_ffn 把一个 MoE 层展开成哪些 ggml 算子
  - 目录：`L2-model-to-graph/L2-06-graph-skeleton/`
- [x] **`L2-07`** ★ 上下文与解码 llama-context　`P0`　2 文件
  - 讲解要点：decode 主流程、图构建→分配→计算的调用链、graph reuse
  - 验收点：能按顺序列出 llama_decode 内部从建图到 ggml_backend_sched_graph_compute 的每一步
  - 目录：`L2-model-to-graph/L2-07-context-and-decode/`
- [x] **`L2-08`** 采样器与词表　`P1`　8 文件
  - 讲解要点：采样链、grammar 约束、BPE 分词与 unicode 规范化
  - 验收点：能说明采样为何在 CPU 上做而不进图
  - 目录：`L2-model-to-graph/L2-08-sampler-and-vocab/`
- [x] **`L2-09`** 量化、导出与适配器　`P1`　10 文件
  - 讲解要点：量化流水线、imatrix、LoRA 适配器如何改变图、对话模板
  - 验收点：能说出 LoRA 是在哪一步被加进图的
  - 目录：`L2-model-to-graph/L2-09-quant-export-adapter/`
- [x] **`L2-10`** 模型家族（一）：多模态与视觉编码　`P0`　6 文件
  - 讲解要点：视觉/音频编码器如何进入同一个图；mmproj 的算子与 LLM 部分如何拼接
  - 验收点：能说出视觉塔的输出以什么形状喂给 LLM 部分
  - 目录：`L2-model-to-graph/L2-10-models-multimodal/`
- [ ] **`L2-11`** 模型家族（二）：状态空间与线性注意力　`P0`　30 文件
  - 讲解要点：SSM/卷积状态如何在图上表达；recurrent 状态与 KV cache 的区别
  - 验收点：能说明 ssm_scan 的输入输出与状态更新在哪一步发生
  - 目录：`L2-model-to-graph/L2-11-models-ssm-linear/`
- [ ] **`L2-12`** 模型家族（三）：稀疏专家 MoE（上）　`P0`　24 文件
  - 讲解要点：ffn_gate_inp / ffn_gate_exps 的组合方式、专家路由算子在图上的形态
  - 验收点：能画出 top-k 路由 + 专家并行的子图
  - 目录：`L2-model-to-graph/L2-12-models-moe-a/`
- [ ] **`L2-13`** 模型家族（四）：稀疏专家 MoE（下）　`P0`　24 文件
  - 讲解要点：共享专家、专家并行、MoE 与注意力变体的组合
  - 验收点：能说出 build_moe_ffn 的参数里哪些控制专家选择
  - 目录：`L2-model-to-graph/L2-13-models-moe-b/`
- [ ] **`L2-14`** 模型家族（五）：稠密 Transformer（上）　`P0`　37 文件
  - 讲解要点：标准 transformer 的图构建：embedding → 注意力 → FFN → norm → lm_head
  - 验收点：能默写出 llm_build_llama 的主干图
  - 目录：`L2-model-to-graph/L2-14-models-dense-a/`
- [ ] **`L2-15`** 模型家族（六）：稠密 Transformer（下）与投机解码草稿模型　`P0`　36 文件
  - 讲解要点：注意力变体（GQA/SWA/MLA）、投机解码草稿模型的图裁剪
  - 验收点：能说出草稿模型与主模型共享 KV 的图差异
  - 目录：`L2-model-to-graph/L2-15-models-dense-b/`

## L3 · 后端发现与注册

> 机器上有哪些后端、它们如何被注册与创建设备
>
> 4 课，覆盖 7 个源文件。

- [ ] **`L3-01`** 后端接口 ggml-backend.h：后端的契约　`P0`　2 文件
  - 讲解要点：ggml_backend_i / ggml_backend_device_i / ggml_backend_buffer_type_i 三张虚表
  - 验收点：能说出实现一个新后端要实现哪几个接口函数
  - 目录：`L3-backend-registry/L3-01-backend-contract/`
- [ ] **`L3-02`** ★ 后端注册表与设备发现　`P0`　2 文件
  - 讲解要点：静态注册 vs 动态加载、设备枚举顺序、多后端共存时的设备命名
  - 验收点：能解释为什么设备顺序会影响 offload 结果
  - 目录：`L3-backend-registry/L3-02-registry-and-devices/`
- [ ] **`L3-03`** 动态加载后端 ggml-backend-dl　`P1`　2 文件
  - 讲解要点：运行时 dlopen 后端动态库、符号解析、版本校验
  - 验收点：能说出后端动态库必须导出哪些符号
  - 目录：`L3-backend-registry/L3-03-dynamic-loading/`
- [ ] **`L3-04`** ggml-feats：后端能力探测　`P1`　1 文件
  - 讲解要点：编译期/运行期特性开关如何影响后端行为
  - 验收点：能说出一个特性开关从定义到被后端查询的路径
  - 目录：`L3-backend-registry/L3-04-backend-features/`

## L4 · 内存与调度

> tensor 住在哪个 buffer、图如何被切给不同后端
>
> 4 课，覆盖 9 个源文件。

- [ ] **`L4-01`** 分配器 ggml-alloc：算子的内存从哪来　`P0`　2 文件
  - 讲解要点：arena 分配、tensor 复用、view 的处理、图重放时的内存规划
  - 验收点：能说出为什么 ggml 需要自己管理内存而不是逐个 ggml_new_tensor
  - 目录：`L4-memory-and-scheduling/L4-01-ggml-alloc/`
- [ ] **`L4-02`** ★ 调度器：把图切给不同后端　`P0`　2 文件
  - 讲解要点：ggml_backend_sched 的 split、offload 决策、跨后端 copy 节点、权重缓冲
  - 验收点：能解释图切分的边界为什么必须插 copy 节点
  - 目录：`L4-memory-and-scheduling/L4-02-scheduler-split/`
- [ ] **`L4-03`** buffer 与 buffer type　`P1`　2 文件
  - 讲解要点：buffer type 的分配/初始化/属性查询、host buffer 与 device buffer 的分工
  - 验收点：能说出 pinned host buffer 在什么场景下被用到
  - 目录：`L4-memory-and-scheduling/L4-03-buffers-and-types/`
- [ ] **`L4-04`** 图执行入口：graph_compute 与异步　`P1`　3 文件
  - 讲解要点：ggml_backend_graph_compute / _async、事件同步、graph reuse 的缓存
  - 验收点：能说出同步与异步路径分别在哪一步等待完成
  - 目录：`L4-memory-and-scheduling/L4-04-graph-compute-entry/`

## L5 · CPU 后端执行

> 算子落到 CPU 的真实内核：量化点积与 SIMD 多架构
>
> 5 课，覆盖 68 个源文件。

- [ ] **`L5-01`** CPU 后端骨架：从 graph_compute 到算子分派　`P0`　3 文件
  - 讲解要点：ggml_compute_forward 的大 switch、工作线程切分、nth 与任务划分
  - 验收点：能说出一个 op 从进入 CPU 后端到调用具体内核经过哪几层
  - 目录：`L5-cpu-backend/L5-01-cpu-backend-skeleton/`
- [ ] **`L5-02`** ★ 一元与二元算子内核　`P0`　6 文件
  - 讲解要点：逐元素算子的向量化、类型组合的模板展开、sliding window 变体
  - 验收点：能说出 unary op 的 SIMD 分发是怎么做的
  - 目录：`L5-cpu-backend/L5-02-unary-binary-ops/`
- [ ] **`L5-03`** ★ 向量化基础设施　`P0`　4 文件
  - 讲解要点：ggml_vec_dot_* 家族、SIMD 抽象层、不同 ISA 的映射
  - 验收点：能说出 ggml_vec_dot_q4_0_q8_0 在 AVX2 与 NEON 上的实现差异
  - 目录：`L5-cpu-backend/L5-03-simd-infrastructure/`
- [ ] **`L5-04`** ★ 量化与 repack　`P0`　6 文件
  - 讲解要点：repack 把量化块重排成 SIMD 友好的布局；traits 如何选择 kernel 变体
  - 验收点：能解释 repack 之后的 layout 为什么能加速点积
  - 目录：`L5-cpu-backend/L5-04-quants-and-repack/`
- [ ] **`L5-05`** ★ 多架构 SIMD 与厂商加速　`P0`　49 文件
  - 讲解要点：x86/ARM/RISC-V/s390/PowerPC/LoongArch/WASM 各自的量化内核；AMX 与 KleidiAI
  - 验收点：能说出同一算子在不同架构上选择了哪个 kernel
  - 目录：`L5-cpu-backend/L5-05-multiarch-and-vendor/`

## L6 · GPU 后端执行

> 算子落到 NVIDIA/AMD/Intel/Apple/浏览器 GPU 的形式
>
> 9 课，覆盖 690 个源文件。

- [ ] **`L6-01`** CUDA 后端骨架　`P0`　3 文件
  - 讲解要点：后端接口实现、stream 管理、graph capture、设备属性与能力判断
  - 验收点：能说出 CUDA 后端如何决定一个 op 是否支持
  - 目录：`L6-gpu-backend/L6-01-cuda-skeleton/`
- [ ] **`L6-02`** ★ CUDA 量化矩阵乘：mmq / mmvq / mmf　`P0`　10 文件
  - 讲解要点：prefetch/tile 加载、量化点积在 GPU 上的展开、batch 大小决定的 kernel 选择
  - 验收点：能说出 mmq 与 mmvq 分别在什么形状下被选中
  - 目录：`L6-gpu-backend/L6-02-cuda-quant-matmul/`
- [ ] **`L6-03`** ★ CUDA FlashAttention　`P0`　15 文件
  - 讲解要点：vec/tile/mma 三条实现路线、模板实例化矩阵、mask 与 alibi 的处理
  - 验收点：能说出 fattn-vec 与 fattn-mma 各自适用的 head size 与 dtype
  - 目录：`L6-gpu-backend/L6-03-cuda-flash-attention/`
- [ ] **`L6-04`** CUDA 其余算子与模板实例化　`P1`　280 文件
  - 讲解要点：归一化、rope、ssm、排序、MoE 归约等算子的 GPU 实现；template-instances 的作用
  - 验收点：能说出 template-instances 目录为什么必须存在
  - 目录：`L6-gpu-backend/L6-04-cuda-other-ops/`
- [ ] **`L6-05`** SYCL 后端（Intel GPU）　`P1`　174 文件
  - 讲解要点：SYCL 与 CUDA 的对照、DPC++ 队列、量化矩阵乘的移植
  - 验收点：能说出 SYCL 后端与 CUDA 后端在同一算子上的结构对应关系
  - 目录：`L6-gpu-backend/L6-05-sycl-backend/`
- [ ] **`L6-06`** Vulkan 后端：主机端　`P1`　7 文件
  - 讲解要点：pipeline 缓存、descriptor set、push constants、shader 生成（.comp -> SPIR-V）
  - 验收点：能说出 Vulkan 后端为什么需要预生成 shader
  - 目录：`L6-gpu-backend/L6-06-vulkan-host/`
- [ ] **`L6-07`** Vulkan 后端：计算着色器　`P1`　162 文件
  - 讲解要点：GLSL 计算着色器如何实现 mul_mm/mul_mmq/flash_attn 与各量化类型的反量化
  - 验收点：能说出一个 .comp 从生成到被 pipeline 使用经过哪几步
  - 目录：`L6-gpu-backend/L6-07-vulkan-shaders/`
- [ ] **`L6-08`** Metal 后端：主机端与图融合　`P1`　16 文件
  - 讲解要点：Metal 的算子融合（fusion）机制：把多个 ggml op 合成一个 kernel
  - 验收点：能说出 fusion 在图上何时被触发、由谁决定
  - 目录：`L6-gpu-backend/L6-08-metal-host-fusion/`
- [ ] **`L6-09`** Metal 后端：Metal 内核　`P1`　23 文件
  - 讲解要点：mul_mm/mul_mv/flash-attention/rope 等 .metal 内核的写法与量化类型支持
  - 验收点：能说出 mul_mv 相比 mul_mm 在什么形状下更优
  - 目录：`L6-gpu-backend/L6-09-metal-kernels/`

## L7 · NPU 与加速器后端

> 算子落到 NPU/DSP/专用加速器的形式
>
> 6 课，覆盖 312 个源文件。

- [ ] **`L7-01`** CANN 后端（Ascend NPU）　`P1`　7 文件
  - 讲解要点：Ascend 的 ACL 接口、算子映射、内存与 stream
  - 验收点：能说出 CANN 后端如何把一个 ggml op 映射到 ACL 算子
  - 目录：`L7-npu-backend/L7-01-cann-ascend-npu/`
- [ ] **`L7-02`** Hexagon 后端（Qualcomm NPU/DSP）（上）　`P2`　87 文件
  - 讲解要点：Hexagon DSP 的远端执行、HTP 图、host 与 DSP 的边界
  - 验收点：能说出 host 侧与 DSP 侧各自负责什么
  - 目录：`L7-npu-backend/L7-02-hexagon-qualcomm/`
- [ ] **`L7-03`** OpenVINO 后端（Intel NPU/GPU）　`P2`　82 文件
  - 讲解要点：把 ggml 子图翻译成 OpenVINO 图、设备选择、回退策略
  - 验收点：能说出什么形状的子图会被整体交给 OpenVINO
  - 目录：`L7-npu-backend/L7-03-openvino-intel-npu/`
- [ ] **`L7-04`** ExecuTorch 后端（边缘/移动端）　`P2`　68 文件
  - 讲解要点：ExecuTorch 的委托机制、图导出与运行时
  - 验收点：能说出这个后端与其它后端在数据面抽象上的不同
  - 目录：`L7-npu-backend/L7-04-executorch-edge/`
- [ ] **`L7-05`** VirtGPU 后端（虚拟化 GPU）　`P2`　39 文件
  - 讲解要点：客户机与宿主机之间的命令流、共享内存与同步
  - 验收点：能说出虚拟化场景下为什么不能直接映射设备内存
  - 目录：`L7-npu-backend/L7-05-virtgpu-virtualized/`
- [ ] **`L7-06`** 小后端合集：OpenCL / WebGPU / MUSA / RPC / BLAS / ZenDNN / zDNN　`P2`　29 文件
  - 讲解要点：这些后端共享同一套 ggml_backend_i 契约但数据面差别极大；RPC 把远端当设备
  - 验收点：能说出 RPC 后端如何把一个 tensor 送到远端
  - 目录：`L7-npu-backend/L7-06-small-backends/`

## L8 · 端到端

> 一次真实推理里，一个算子走过的完整路径
>
> 3 课，覆盖 4 个源文件。

- [ ] **`L8-01`** ★ 端到端：一个 mul_mat 的完整旅程　`P1`　1 文件
  - 讲解要点：从 build_attn 里的一次 ggml_mul_mat 调用，一路走到 CPU/CUDA 内核的完整调用链
  - 验收点：能不看代码复述这条链路并指出每一步的判据
  - 目录：`L8-end-to-end/L8-01-journey-of-mul-mat/`
- [ ] **`L8-02`** ★ 端到端：offload 决策实战　`P1`　2 文件
  - 讲解要点：用真实模型参数走一遍图切分，看哪些层被放到 GPU、哪些留在 CPU
  - 验收点：能预测给定 n_gpu_layers 下的图切分结果
  - 目录：`L8-end-to-end/L8-02-offload-decision/`
- [ ] **`L8-03`** 端到端：新后端要做什么　`P2`　1 文件
  - 讲解要点：综合前面所有课，给出新增一个后端的清单与最小实现
  - 验收点：能列出一个新后端必须实现的函数与必须通过的测试
  - 目录：`L8-end-to-end/L8-03-writing-a-new-backend/`

