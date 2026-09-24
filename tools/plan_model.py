#!/usr/bin/env python3
"""课程计划模型 —— 把覆盖域里的每个源文件指派到一课。

这是 plan/COVERAGE.md 的唯一数据源，也是"覆盖率"这个概念的基准。
运行时断言：覆盖域中不存在未指派文件。

视角：算子从模型定义到 CPU/GPU/NPU 等后端执行。
"""

# 每个 lesson 的 `paths` 是「前缀」或「精确路径」的列表。
# 允许同一文件被多课声明（门禁只要求"至少被一课声明"）。

LAYERS = [
    ("L1", "算子的表示", "ggml 用什么统一表示一个算子：张量、op 枚举、图"),
    ("L2", "从模型到图", "权重与超参如何被翻译成待执行的 ggml 图"),
    ("L3", "后端发现与注册", "机器上有哪些后端、它们如何被注册与创建设备"),
    ("L4", "内存与调度", "tensor 住在哪个 buffer、图如何被切给不同后端"),
    ("L5", "CPU 后端执行", "算子落到 CPU 的真实内核：量化点积与 SIMD 多架构"),
    ("L6", "GPU 后端执行", "算子落到 NVIDIA/AMD/Intel/Apple/浏览器 GPU 的形式"),
    ("L7", "NPU 与加速器后端", "算子落到 NPU/DSP/专用加速器的形式"),
    ("L8", "端到端", "一次真实推理里，一个算子走过的完整路径"),
]

# ---------------------------------------------------------------- 计划

LESSONS = [

# ============================== L1 算子的表示 ==============================
dict(id="L1-01", layer="L1", prio="P0",
     title="ggml 张量：算子的数据面",
     paths=["ggml/include/ggml.h"],
     ideas="张量结构体字段、数据类型枚举、维度与步长约定；为什么 ggml 用 type 而非 dtype 表达量化",
     accept="能说出 ggml_tensor 里哪些字段决定一次内核调用的形状"),

dict(id="L1-02", layer="L1", prio="P0",
     title="算子枚举与元数据：算子的身份",
     paths=["ggml/src/ggml.c", "ggml/src/ggml-impl.h"],
     ideas="GGML_OP_* 枚举、ggml_op_name、参数个数表 nargs、算子如何在图上成为一个节点",
     accept="给出任意一个 GGML_OP_*，能说出它的输入个数与输出形状规则在哪定义"),

dict(id="L1-03", layer="L1", prio="P0",
     title="计算图与拓扑遍历",
     paths=["ggml/src/ggml.c", "ggml/include/ggml-cpp.h"],
     ideas="ggml_cgraph、view_src/view_offs、ggml_build_forward_expand 的拓扑排序、visited_hash_set",
     accept="能解释为什么拓扑排序用 hash set 而不是标记位"),

dict(id="L1-04", layer="L1", prio="P0",
     title="量化块结构：算子内层的压缩数据",
     paths=["ggml/src/ggml-quants.c", "ggml/src/ggml-quants.h", "ggml/src/ggml-common.h",
            # 实测回改：QK4_0 等块大小宏【不在】 ggml-quants.h，而在 ggml-common.h
            # （ggml-quants.h 第 4 行 #include "ggml-common.h"）；
            # ggml_type_traits 表实测只在 ggml.c（起于 632 行），是"类型号->块大小"
            # 绑定的唯一出处，故追加引用 ggml.c。原计划措辞有误，已回改。
            "ggml/src/ggml.c"],
     ideas="block_q4_0 等块布局（实测：Q4_0=18B / Q4_1=20B / Q8_0=34B / Q4_K=144B / Q6_K=210B）、"
           "scale/min 的表示、反量化函数族、ggml.c 里的 ggml_type_traits 表如何绑定类型与块大小",
     accept="能画出 Q4_0 与 Q4_K 的块内存布局差异，并说出对内核的影响"),

dict(id="L1-05", layer="L1", prio="P0",
     title="上下文、线程与优化器接口",
     paths=["ggml/src/ggml.cpp", "ggml/src/ggml-threading.cpp", "ggml/src/ggml-threading.h",
            "ggml/src/ggml-opt.cpp", "ggml/include/ggml-opt.h",
            # 实测回改：arena 的真实实现在 ggml.c（958-985 / 1611-1677 / 1708-1759），
            # 不在本课原清单里。ggml.cpp 实测只有 26 行、装一个 std::terminate handler；
            # ggml-threading.* 实测只有全局互斥锁、没有线程池。原计划措辞有误，已回改。
            "ggml/src/ggml.c"],
     ideas="ggml_context 的 arena 对象分配（实测在 ggml.c）、全局临界区、ggml-opt 的训练接口",
     accept="能说明 ggml_context 为什么用 arena 而非逐个 malloc"),

dict(id="L1-06", layer="L1", prio="P0",
     title="GGUF：算子的权重从文件来",
     paths=["ggml/src/gguf.cpp", "ggml/include/gguf.h"],
     ideas="GGUF 容器格式、KV 段、张量信息段、对齐、读写 API 与错误处理",
     accept="能按字节说出 GGUF 文件头到张量数据之间的布局"),

# ============================== L2 从模型到图 ==============================
dict(id="L2-01", layer="L2", prio="P0",
     title="llama.h 公共 API 全景",
     paths=["include/llama.h", "include/llama-cpp.h", "src/llama.cpp", "src/llama-ext.h"],
     ideas="模型/上下文/采样三段式 API、参数结构体族、llama.cpp 作为聚合入口的地位",
     accept="能画出从 llama_model_load 到 llama_decode 的 API 调用顺序图"),

dict(id="L2-02", layer="L2", prio="P0",
     title="架构表与超参：模型长什么样的元数据",
     paths=["src/llama-arch.cpp", "src/llama-arch.h", "src/llama-hparams.cpp",
            "src/llama-hparams.h", "src/llama-cparams.cpp", "src/llama-cparams.h"],
     ideas="LLM_ARCH_* 枚举、张量命名表、hparams 与 cparams 的分工",
     accept="能解释新增一个模型架构需要在架构表里加哪几项"),

dict(id="L2-03", layer="L2", prio="P0",
     title="权重加载与内存映射",
     paths=["src/llama-model-loader.cpp", "src/llama-model-loader.h", "src/llama-mmap.cpp",
            "src/llama-mmap.h", "src/llama-io.cpp", "src/llama-io.h"],
     ideas="mmap vs read、后端 buffer type 的选择时机、张量落位（哪个后端持有权重）",
     accept="能说出权重张量在什么时刻决定自己被分配到哪个后端"),

dict(id="L2-04", layer="L2", prio="P0",
     title="KV cache 与记忆家族",
     paths=["src/llama-kv-cache", "src/llama-kv-cells.h", "src/llama-memory"],
     ideas="KV cache 的 cell 抽象、iswa/dsa/msa/dsv4 变体、recurrent 与 hybrid 记忆",
     accept="能解释为什么需要 memory 抽象而不是一个统一的 KV cache"),

dict(id="L2-05", layer="L2", prio="P0",
     title="批、解码参数与模型装配",
     paths=["src/llama-batch.cpp", "src/llama-batch.h", "src/llama-model.cpp",
            "src/llama-model.h", "src/llama-impl.cpp", "src/llama-impl.h"],
     ideas="llama_batch 的 token 序列结构、ubatch 切分、模型构造与设备分配",
     accept="能说明一个 batch 是如何被切成 ubatch 并影响图形态的"),

dict(id="L2-06", layer="L2", prio="P0",
     title="★ 计算图骨架 llama-graph",
     paths=["src/llama-graph.cpp", "src/llama-graph.h"],
     ideas="llm_graph_context、build_* 原语族（build_norm/build_attn/build_ffn/build_moe_ffn）",
     accept="能说出 build_moe_ffn 把一个 MoE 层展开成哪些 ggml 算子"),

dict(id="L2-07", layer="L2", prio="P0",
     title="★ 上下文与解码 llama-context",
     paths=["src/llama-context.cpp", "src/llama-context.h"],
     ideas="decode 主流程、图构建→分配→计算的调用链、graph reuse",
     accept="能按顺序列出 llama_decode 内部从建图到 ggml_backend_sched_graph_compute 的每一步"),

dict(id="L2-08", layer="L2", prio="P1",
     title="采样器与词表",
     paths=["src/llama-sampler.cpp", "src/llama-sampler.h", "src/llama-vocab.cpp",
            "src/llama-vocab.h", "src/unicode.cpp", "src/unicode.h", "src/unicode-data.cpp",
            "src/unicode-data.h"],
     ideas="采样链、grammar 约束、BPE 分词与 unicode 规范化",
     accept="能说明采样为何在 CPU 上做而不进图"),

dict(id="L2-09", layer="L2", prio="P1",
     title="量化、导出与适配器",
     paths=["src/llama-quant.cpp", "src/llama-quant.h", "src/llama-model-saver.cpp",
            "src/llama-model-saver.h", "src/llama-adapter.cpp", "src/llama-adapter.h",
            "src/llama-grammar.cpp", "src/llama-grammar.h", "src/llama-chat.cpp",
            "src/llama-chat.h"],
     ideas="量化流水线、imatrix、LoRA 适配器如何改变图、对话模板",
     accept="能说出 LoRA 是在哪一步被加进图的"),

# ---- 模型家族：按图构建原语静态分类（见 plan/COVERAGE.md 的分组依据）----
dict(id="L2-10", layer="L2", prio="P0", group="models",
     title="模型家族（一）：多模态与视觉编码",
     paths=["src/models/clip.cpp", "src/models/deepseek4.cpp", "src/models/gemma4.cpp",
            "src/models/pockettts.cpp", "src/models/qwen4exp.cpp", "src/models/models.h"],
     ideas="视觉/音频编码器如何进入同一个图；mmproj 的算子与 LLM 部分如何拼接",
     accept="能说出视觉塔的输出以什么形状喂给 LLM 部分"),

dict(id="L2-11", layer="L2", prio="P0", group="models",
     title="模型家族（二）：状态空间与线性注意力",
     paths=["src/models/"],  # 由 plan_matrix 按 ssm 分组填充，见 GROUP_OVERRIDES
     ideas="SSM/卷积状态如何在图上表达；recurrent 状态与 KV cache 的区别",
     accept="能说明 ssm_scan 的输入输出与状态更新在哪一步发生"),

dict(id="L2-12", layer="L2", prio="P0", group="models",
     title="模型家族（三）：稀疏专家 MoE（上）",
     paths=["src/models/"],
     ideas="ffn_gate_inp / ffn_gate_exps 的组合方式、专家路由算子在图上的形态",
     accept="能画出 top-k 路由 + 专家并行的子图"),

dict(id="L2-13", layer="L2", prio="P0", group="models",
     title="模型家族（四）：稀疏专家 MoE（下）",
     paths=["src/models/"],
     ideas="共享专家、专家并行、MoE 与注意力变体的组合",
     accept="能说出 build_moe_ffn 的参数里哪些控制专家选择"),

dict(id="L2-14", layer="L2", prio="P0", group="models",
     title="模型家族（五）：稠密 Transformer（上）",
     paths=["src/models/"],
     ideas="标准 transformer 的图构建：embedding → 注意力 → FFN → norm → lm_head",
     accept="能默写出 llm_build_llama 的主干图"),

dict(id="L2-15", layer="L2", prio="P0", group="models",
     title="模型家族（六）：稠密 Transformer（下）与投机解码草稿模型",
     paths=["src/models/"],
     ideas="注意力变体（GQA/SWA/MLA）、投机解码草稿模型的图裁剪",
     accept="能说出草稿模型与主模型共享 KV 的图差异"),

# ============================== L3 后端发现与注册 ==============================
dict(id="L3-01", layer="L3", prio="P0",
     title="后端接口 ggml-backend.h：后端的契约",
     paths=["ggml/include/ggml-backend.h", "ggml/src/ggml-backend-impl.h"],
     ideas="ggml_backend_i / ggml_backend_device_i / ggml_backend_buffer_type_i 三张虚表",
     accept="能说出实现一个新后端要实现哪几个接口函数"),

dict(id="L3-02", layer="L3", prio="P0",
     title="★ 后端注册表与设备发现",
     paths=["ggml/src/ggml-backend-reg.cpp", "ggml/src/ggml-backend-meta.cpp"],
     ideas="静态注册 vs 动态加载、设备枚举顺序、多后端共存时的设备命名",
     accept="能解释为什么设备顺序会影响 offload 结果"),

dict(id="L3-03", layer="L3", prio="P1",
     title="动态加载后端 ggml-backend-dl",
     paths=["ggml/src/ggml-backend-dl.cpp", "ggml/src/ggml-backend-dl.h"],
     ideas="运行时 dlopen 后端动态库、符号解析、版本校验",
     accept="能说出后端动态库必须导出哪些符号"),

dict(id="L3-04", layer="L3", prio="P1",
     title="ggml-feats：后端能力探测",
     paths=["ggml/src/ggml-feats.h"],
     ideas="编译期/运行期特性开关如何影响后端行为",
     accept="能说出一个特性开关从定义到被后端查询的路径"),

# ============================== L4 内存与调度 ==============================
dict(id="L4-01", layer="L4", prio="P0",
     title="分配器 ggml-alloc：算子的内存从哪来",
     paths=["ggml/src/ggml-alloc.c", "ggml/include/ggml-alloc.h"],
     ideas="arena 分配、tensor 复用、view 的处理、图重放时的内存规划",
     accept="能说出为什么 ggml 需要自己管理内存而不是逐个 ggml_new_tensor"),

dict(id="L4-02", layer="L4", prio="P0",
     title="★ 调度器：把图切给不同后端",
     paths=["ggml/src/ggml-backend.cpp", "ggml/include/ggml-backend.h"],
     ideas="ggml_backend_sched 的 split、offload 决策、跨后端 copy 节点、权重缓冲",
     accept="能解释图切分的边界为什么必须插 copy 节点"),

dict(id="L4-03", layer="L4", prio="P1",
     title="buffer 与 buffer type",
     paths=["ggml/src/ggml-backend.cpp", "ggml/include/ggml-backend.h"],
     ideas="buffer type 的分配/初始化/属性查询、host buffer 与 device buffer 的分工",
     accept="能说出 pinned host buffer 在什么场景下被用到"),

dict(id="L4-04", layer="L4", prio="P1",
     title="图执行入口：graph_compute 与异步",
     paths=["ggml/src/ggml-backend.cpp", "ggml/src/ggml.c"],
     ideas="ggml_backend_graph_compute / _async、事件同步、graph reuse 的缓存",
     accept="能说出同步与异步路径分别在哪一步等待完成"),

# ============================== L5 CPU 后端执行 ==============================
dict(id="L5-01", layer="L5", prio="P0",
     title="CPU 后端骨架：从 graph_compute 到算子分派",
     paths=["ggml/src/ggml-cpu/ggml-cpu.c", "ggml/src/ggml-cpu/ggml-cpu.cpp",
            "ggml/include/ggml-cpu.h"],
     ideas="ggml_compute_forward 的大 switch、工作线程切分、nth 与任务划分",
     accept="能说出一个 op 从进入 CPU 后端到调用具体内核经过哪几层"),

dict(id="L5-02", layer="L5", prio="P0",
     title="★ 一元与二元算子内核",
     paths=["ggml/src/ggml-cpu/unary-ops.cpp", "ggml/src/ggml-cpu/unary-ops.h",
            "ggml/src/ggml-cpu/binary-ops.cpp", "ggml/src/ggml-cpu/binary-ops.h",
            "ggml/src/ggml-cpu/ops.cpp", "ggml/src/ggml-cpu/ops.h"],
     ideas="逐元素算子的向量化、类型组合的模板展开、sliding window 变体",
     accept="能说出 unary op 的 SIMD 分发是怎么做的"),

dict(id="L5-03", layer="L5", prio="P0",
     title="★ 向量化基础设施",
     paths=["ggml/src/ggml-cpu/vec.cpp", "ggml/src/ggml-cpu/vec.h",
            "ggml/src/ggml-cpu/simd-mappings.h", "ggml/src/ggml-cpu/simd-gemm.h"],
     ideas="ggml_vec_dot_* 家族、SIMD 抽象层、不同 ISA 的映射",
     accept="能说出 ggml_vec_dot_q4_0_q8_0 在 AVX2 与 NEON 上的实现差异"),

dict(id="L5-04", layer="L5", prio="P0",
     title="★ 量化与 repack",
     paths=["ggml/src/ggml-cpu/quants.c", "ggml/src/ggml-cpu/quants.h",
            "ggml/src/ggml-cpu/repack.cpp", "ggml/src/ggml-cpu/repack.h",
            "ggml/src/ggml-cpu/traits.cpp", "ggml/src/ggml-cpu/traits.h"],
     ideas="repack 把量化块重排成 SIMD 友好的布局；traits 如何选择 kernel 变体",
     accept="能解释 repack 之后的 layout 为什么能加速点积"),

dict(id="L5-05", layer="L5", prio="P0",
     title="★ 多架构 SIMD 与厂商加速",
     paths=["ggml/src/ggml-cpu/arch/", "ggml/src/ggml-cpu/arch-fallback.h",
            "ggml/src/ggml-cpu/amx/", "ggml/src/ggml-cpu/kleidiai/",
            "ggml/src/ggml-cpu/spacemit/", "ggml/src/ggml-cpu/llamafile/",
            "ggml/src/ggml-cpu/hbm.cpp", "ggml/src/ggml-cpu/hbm.h",
            "ggml/src/ggml-cpu/iqp.cpp", "ggml/src/ggml-cpu/iqp.h",
            "ggml/src/ggml-cpu/common.h", "ggml/src/ggml-cpu/ggml-cpu-impl.h"],
     ideas="x86/ARM/RISC-V/s390/PowerPC/LoongArch/WASM 各自的量化内核；AMX 与 KleidiAI",
     accept="能说出同一算子在不同架构上选择了哪个 kernel"),

# ============================== L6 GPU 后端执行 ==============================
dict(id="L6-01", layer="L6", prio="P0",
     title="CUDA 后端骨架",
     paths=["ggml/src/ggml-cuda/ggml-cuda.cu", "ggml/src/ggml-cuda/common.cuh",
            "ggml/include/ggml-cuda.h"],
     ideas="后端接口实现、stream 管理、graph capture、设备属性与能力判断",
     accept="能说出 CUDA 后端如何决定一个 op 是否支持"),

dict(id="L6-02", layer="L6", prio="P0",
     title="★ CUDA 量化矩阵乘：mmq / mmvq / mmf",
     paths=["ggml/src/ggml-cuda/mmq.cuh", "ggml/src/ggml-cuda/mmq.cu",
            "ggml/src/ggml-cuda/mmq-load-tiles.cuh", "ggml/src/ggml-cuda/mmq-vec-dot.cuh",
            "ggml/src/ggml-cuda/mmvq.cu", "ggml/src/ggml-cuda/mmvq.cuh",
            "ggml/src/ggml-cuda/mmvf.cu", "ggml/src/ggml-cuda/mmvf.cuh",
            "ggml/src/ggml-cuda/vecdotq.cuh", "ggml/src/ggml-cuda/mma.cuh",
            "ggml/src/ggml-cuda/mmq-instance-"],
     ideas="prefetch/tile 加载、量化点积在 GPU 上的展开、batch 大小决定的 kernel 选择",
     accept="能说出 mmq 与 mmvq 分别在什么形状下被选中"),

dict(id="L6-03", layer="L6", prio="P0",
     title="★ CUDA FlashAttention",
     paths=["ggml/src/ggml-cuda/fattn", "ggml/src/ggml-cuda/fattn-",
            "ggml/src/ggml-cuda/pad", "ggml/src/ggml-cuda/unary", "ggml/src/ggml-cuda/softcap"],
     ideas="vec/tile/mma 三条实现路线、模板实例化矩阵、mask 与 alibi 的处理",
     accept="能说出 fattn-vec 与 fattn-mma 各自适用的 head size 与 dtype"),

dict(id="L6-04", layer="L6", prio="P1",
     title="CUDA 其余算子与模板实例化",
     paths=["ggml/src/ggml-cuda/"],
     ideas="归一化、rope、ssm、排序、MoE 归约等算子的 GPU 实现；template-instances 的作用",
     accept="能说出 template-instances 目录为什么必须存在"),

dict(id="L6-05", layer="L6", prio="P1",
     title="SYCL 后端（Intel GPU）",
     paths=["ggml/src/ggml-sycl/", "ggml/include/ggml-sycl.h"],
     ideas="SYCL 与 CUDA 的对照、DPC++ 队列、量化矩阵乘的移植",
     accept="能说出 SYCL 后端与 CUDA 后端在同一算子上的结构对应关系"),

dict(id="L6-06", layer="L6", prio="P1",
     title="Vulkan 后端：主机端",
     paths=["ggml/src/ggml-vulkan/ggml-vulkan.cpp", "ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp",
            "ggml/src/ggml-vulkan/ggml-vulkan-common.h", "ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp",
            "ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h",
            "ggml/src/ggml-vulkan/ggml-vulkan-types.h", "ggml/include/ggml-vulkan.h"],
     ideas="pipeline 缓存、descriptor set、push constants、shader 生成（.comp -> SPIR-V）",
     accept="能说出 Vulkan 后端为什么需要预生成 shader"),

dict(id="L6-07", layer="L6", prio="P1",
     title="Vulkan 后端：计算着色器",
     paths=["ggml/src/ggml-vulkan/vulkan-shaders/"],
     ideas="GLSL 计算着色器如何实现 mul_mm/mul_mmq/flash_attn 与各量化类型的反量化",
     accept="能说出一个 .comp 从生成到被 pipeline 使用经过哪几步"),

dict(id="L6-08", layer="L6", prio="P1",
     title="Metal 后端：主机端与图融合",
     paths=["ggml/src/ggml-metal/ggml-metal.cpp", "ggml/src/ggml-metal/ggml-metal-common.cpp",
            "ggml/src/ggml-metal/ggml-metal-common.h", "ggml/src/ggml-metal/ggml-metal-context.h",
            "ggml/src/ggml-metal/ggml-metal-context.m", "ggml/src/ggml-metal/ggml-metal-device.cpp",
            "ggml/src/ggml-metal/ggml-metal-device.h", "ggml/src/ggml-metal/ggml-metal-device.m",
            "ggml/src/ggml-metal/ggml-metal-fusion.cpp", "ggml/src/ggml-metal/ggml-metal-fusion.h",
            "ggml/src/ggml-metal/ggml-metal-impl.h", "ggml/src/ggml-metal/ggml-metal-ops.cpp",
            "ggml/src/ggml-metal/ggml-metal-ops.h", "ggml/src/ggml-metal/ggml-metal-tuning.cpp",
            "ggml/src/ggml-metal/ggml-metal-tuning.h", "ggml/include/ggml-metal.h"],
     ideas="Metal 的算子融合（fusion）机制：把多个 ggml op 合成一个 kernel",
     accept="能说出 fusion 在图上何时被触发、由谁决定"),

dict(id="L6-09", layer="L6", prio="P1",
     title="Metal 后端：Metal 内核",
     paths=["ggml/src/ggml-metal/kernels/"],
     ideas="mul_mm/mul_mv/flash-attention/rope 等 .metal 内核的写法与量化类型支持",
     accept="能说出 mul_mv 相比 mul_mm 在什么形状下更优"),

# ============================== L7 NPU 与加速器后端 ==============================
dict(id="L7-01", layer="L7", prio="P1",
     title="CANN 后端（Ascend NPU）",
     paths=["ggml/src/ggml-cann/", "ggml/include/ggml-cann.h"],
     ideas="Ascend 的 ACL 接口、算子映射、内存与 stream",
     accept="能说出 CANN 后端如何把一个 ggml op 映射到 ACL 算子"),

dict(id="L7-02", layer="L7", prio="P2",
     title="Hexagon 后端（Qualcomm NPU/DSP）（上）",
     paths=["ggml/src/ggml-hexagon/", "ggml/include/ggml-hexagon.h"],
     ideas="Hexagon DSP 的远端执行、HTP 图、host 与 DSP 的边界",
     accept="能说出 host 侧与 DSP 侧各自负责什么"),

dict(id="L7-03", layer="L7", prio="P2",
     title="OpenVINO 后端（Intel NPU/GPU）",
     paths=["ggml/src/ggml-openvino/", "ggml/include/ggml-openvino.h"],
     ideas="把 ggml 子图翻译成 OpenVINO 图、设备选择、回退策略",
     accept="能说出什么形状的子图会被整体交给 OpenVINO"),

dict(id="L7-04", layer="L7", prio="P2",
     title="ExecuTorch 后端（边缘/移动端）",
     paths=["ggml/src/ggml-et/", "ggml/include/ggml-et.h"],
     ideas="ExecuTorch 的委托机制、图导出与运行时",
     accept="能说出这个后端与其它后端在数据面抽象上的不同"),

dict(id="L7-05", layer="L7", prio="P2",
     title="VirtGPU 后端（虚拟化 GPU）",
     paths=["ggml/src/ggml-virtgpu/", "ggml/include/ggml-virtgpu.h"],
     ideas="客户机与宿主机之间的命令流、共享内存与同步",
     accept="能说出虚拟化场景下为什么不能直接映射设备内存"),

dict(id="L7-06", layer="L7", prio="P2",
     title="小后端合集：OpenCL / WebGPU / MUSA / RPC / BLAS / ZenDNN / zDNN",
     paths=["ggml/src/ggml-opencl/", "ggml/include/ggml-opencl.h",
            "ggml/src/ggml-webgpu/", "ggml/include/ggml-webgpu.h",
            "ggml/src/ggml-musa/", "ggml/src/ggml-rpc/", "ggml/include/ggml-rpc.h",
            "ggml/src/ggml-blas/", "ggml/include/ggml-blas.h",
            "ggml/src/ggml-zendnn/", "ggml/include/ggml-zendnn.h",
            "ggml/src/ggml-zdnn/", "ggml/include/ggml-zdnn.h"],
     ideas="这些后端共享同一套 ggml_backend_i 契约但数据面差别极大；RPC 把远端当设备",
     accept="能说出 RPC 后端如何把一个 tensor 送到远端"),

# ============================== L8 端到端 ==============================
dict(id="L8-01", layer="L8", prio="P1",
     title="★ 端到端：一个 mul_mat 的完整旅程",
     paths=["src/llama-graph.cpp"],
     ideas="从 build_attn 里的一次 ggml_mul_mat 调用，一路走到 CPU/CUDA 内核的完整调用链",
     accept="能不看代码复述这条链路并指出每一步的判据"),

dict(id="L8-02", layer="L8", prio="P1",
     title="★ 端到端：offload 决策实战",
     paths=["ggml/src/ggml-backend.cpp", "src/llama-model.cpp"],
     ideas="用真实模型参数走一遍图切分，看哪些层被放到 GPU、哪些留在 CPU",
     accept="能预测给定 n_gpu_layers 下的图切分结果"),

dict(id="L8-03", layer="L8", prio="P2",
     title="端到端：新后端要做什么",
     paths=["ggml/src/ggml-backend-impl.h"],
     ideas="综合前面所有课，给出新增一个后端的清单与最小实现",
     accept="能列出一个新后端必须实现的函数与必须通过的测试"),
]
