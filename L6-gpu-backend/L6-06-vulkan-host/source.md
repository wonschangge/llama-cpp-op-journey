<!-- llama-coverage
ggml/include/ggml-vulkan.h
ggml/src/ggml-vulkan/ggml-vulkan.cpp
ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp
ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp
ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h
ggml/src/ggml-vulkan/ggml-vulkan-types.h
ggml/src/ggml-vulkan/ggml-vulkan-common.h
ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
ggml/src/ggml-vulkan/CMakeLists.txt
-->

# L6-06 · Vulkan 后端：主机端 — 源文件

**一句话**：Vulkan 后端把 ggml 图上的一个节点翻译成三样 Vulkan 对象 —— **计算 pipeline**、**descriptor set**、**push constants**；而这三样东西的"内核一侧"是构建期就编好的 SPIR-V 字节数组，运行期只做对象创建与绑定。

本课只看**主机端**（C++ 那一侧）。计算着色器本身的 GLSL 源码是下一课 L6-07 的主题；但"为什么 Vulkan 必须有独立的 shader 生成步骤"这个问题，只有在主机端这一侧才看得清 —— 因为运行期能拿到的只有 `spv_data` / `spv_size` 两个值。

---

## 一、三张虚表：ggml_backend_i 的 Vulkan 实现

回顾 L3-01：一个 ggml 后端就是三张函数指针表（`ggml_backend_i` / `ggml_backend_buffer_i` / `ggml_backend_buffer_type_i`）。Vulkan 把 `backend` 那张放在 `ggml-vulkan.cpp` 末尾，两个 buffer 面的表放在 `ggml-vulkan-buffers.cpp`。

值得注意的不是填了什么，而是**哪些槽位被显式置 `NULL`**：四个 `graph_plan_*` 全为空，说明 Vulkan 不走"图计划缓存"那条路径；`event_record` / `event_wait` 有实现，用来和其它后端做跨后端的流水线同步。

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan.cpp -->
```c
static ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .event_record            = */ ggml_backend_vk_event_record,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_graph_optimize,
};
```

## 二、生成器把 .spv 嵌成 C 数组

这是"预生成"这条链路的最后一环：生成器把刚编译出来的 `.spv` 文件读成二进制，先声明 `_len` 与 `_data`（写进 `.hpp`），再把每个字节以十六进制字面量写进 `_data[]`（写进 `.cpp`）。注意数组的字节数**就是文件长度**：SPIR-V 是数据，没有别的包装。

头文件那一遍（`source` 为空）只走到第 1253 行的声明，因为没有 `input_filepath` 就没有可读的 `.spv`。

<!-- src: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp -->
```c
        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";
//>> ---- ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp:1261-1268 ----
            src << "const uint64_t " << name << "_len = " << data.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << data.size() << "] = {\n" << std::hex;
            auto bytes = reinterpret_cast<const uint8_t*>(data.data());
            for (size_t i = 0; i < data.size(); ++i) {
                src << "0x" << static_cast<int>(bytes[i]) << ",";
                if ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
```

## 三、构建规则：每个 .comp 一条自定义命令

这条命令把上一节与下一节接起来：四个参数分别给出外部编译器（glslc）、这一个 `.comp`、"给它专用的"嵌入文件（所以 `add_custom_command` 的 `OUTPUT` 是逐文件生成的名字），以及所有 `.comp` 共享的同一个头文件。

`DEPFILE` 让 CMake 知道这个 `.cpp` 依赖哪些被 `#include` 的 GLSL 片段；`DEPENDS` 里同时列了 `.comp` 本身、生成器源码和生成器目标 —— 三者任一变化都会触发重编。

<!-- src: ggml/src/ggml-vulkan/CMakeLists.txt -->
```cmake
        add_custom_command(
            OUTPUT  ${_ggml_vk_target_cpp}
            DEPFILE ${_ggml_vk_target_cpp}.d
            COMMAND ${_ggml_vk_genshaders_cmd}
                --glslc      ${Vulkan_GLSLC_EXECUTABLE}
                --source     ${file_full}
                --output-dir ${_ggml_vk_output_dir}
                --target-hpp ${_ggml_vk_header}
                --target-cpp ${_ggml_vk_target_cpp}
            DEPENDS ${file_full}
                    ${_ggml_vk_shaders_gen_sources}
                    vulkan-shaders-gen
            COMMENT "Generate vulkan shaders for ${file}"
        )
        target_sources(ggml-vulkan PRIVATE ${_ggml_vk_target_cpp})
```

## 四、生成的头文件怎么进入编译单元

这三行是整条构建期链路的落点：`ggml-vulkan-shaders.hpp` **不在源码仓库里**，它是 `vulkan-shaders-gen` 在构建目录里生成的，内容只有每个 shader 变体的 `extern const uint64_t <name>_len;` 与 `extern const unsigned char <name>_data[];` 声明。

`ggml-vulkan-types.h` 是 Vulkan 源文件的第一层公共头（`ggml-vulkan-common.h` 又引入它），所以每个编译单元都能看到这些数组 —— 于是 `ggml-vulkan.cpp` 里可以直接写 `matmul_f16_cm2_len, matmul_f16_cm2_data` 这样的标识符。

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan-types.h -->
```c
#include "ggml-backend-impl.h"

#include "ggml-vulkan-shaders.hpp"
```

## 五、pipeline 的缓存与懒编译

设备初始化时会调一次 `ggml_vk_load_shaders(device)`，把整套 pipeline 建出来存进 `device->pipeline_*`。运行期第一次用到某个 pipeline 时，`ggml_pipeline_request_descriptor_sets()` 先看它的 `compiled`：没编完，就把这一个 pipeline 再交给 `ggml_vk_load_shaders(device, pipeline)`（只处理它），然后由条件变量等待编译完成。

`ggml_vk_create_pipeline` 里的 `initialized` 判断保证：重复调用不会覆盖已经填好的字段，只会走"要不要编译"的那一半逻辑。

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan.cpp -->
```c
void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_set_requirements += n;
    if (!pipeline->compiled) {
        ggml_vk_load_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan.cpp:1841-1854 ----
            if (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            if (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_count = parameter_count;
                pipeline->push_constant_size = push_constant_size;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = true;
#if defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#endif
            }
```

## 六、buffer 的内存类型选择

Vulkan 的内存类型是**设备暴露出来的**（`memoryTypes` 加上每个资源的 `memoryTypeBits` 掩码），所以分配必须"按属性挑 + 逐个试"。`ggml_vk_find_memory_properties()` 做第一层过滤：资源允许这个类型、属性位满足要求、堆还放得下。

`ggml_vk_create_buffer_device()` 则是策略表：优先主机内存、UMA 设备、禁用 host-visible 显存、能否用 rebar —— 每一档都给出"首选属性 + 备选属性"，`ggml_vk_create_buffer()` 会按顺序试，全失败才抛 `OutOfDeviceMemoryError`。这是"同一份 ggml 代码在不同 GPU 上拿到不同内存"的原因。

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp -->
```c
static std::vector<uint32_t> ggml_vk_find_memory_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    return indices;
}
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp:193-210 ----
vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } else if (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } else if (device->disable_host_visible_vidmem) {
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
```

## 七、一次 dispatch 的组装：descriptor set + push constants

`ggml_vk_dispatch_pipeline()` 是主机端所有算子的公共出口。它把本课的三种零件按固定顺序装上：取一个 descriptor set 并写入 `parameter_count` 个 storage buffer，推送 push constants，绑定 pipeline 与 descriptor set，最后 dispatch。

四个断言就是"契约检查"：buffer 数不能超过 12、实际给的 buffer 数必须等于建表时的 `parameter_count`、push constant 的字节数必须等于建表时的 `push_constant_size`、workgroup 数不能超过设备上限。这正是第 4 幕与第 8 幕在编译期/建表期就要把这两个数定下来的原因。

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan-common.h -->
```c
template <typename T>
inline void ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos, const T &push_constants, std::array<uint32_t, 3> elements) {
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    for (auto& buffer : descriptor_buffer_infos) {
        std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.maxComputeWorkGroupCount[0] &&
                wg1 <= ctx->device->properties.limits.maxComputeWorkGroupCount[1] &&
                wg2 <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
    GGML_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size());
    GGML_ASSERT(pipeline->push_constant_size == push_constant_size(push_constants));

    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
    vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_count, vk::DescriptorType::eStorageBuffer, nullptr, descriptor_buffer_infos.begin() };
    ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});

    subctx->s->buffer->buf.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, push_constant_size(push_constants), push_constant_data(push_constants));
    subctx->s->buffer->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_set },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->buffer->buf.dispatch(wg0, wg1, wg2);
    }
}
```

## 八、主机端的可观测性：内存日志与结果校验

`ggml-vulkan-debug.cpp` 装的是三个 logger（内存 / 性能 / 同步）与结果校验路径。内存 logger 用内存属性位把每次分配分成 `device` 与 `host` 两类累加，这正是第四节那张"策略表"的可观测侧面。

`ggml_vk_get_op_batch_size()` 给每个算子一个"批量"估计：`GET_ROWS` 返回 0（不值得卸载）、`MUL_MAT` 返回 `ne[1]`、`MUL_MAT_ID` / `ROPE` 返回 `ne[2]`、其余返回 `ggml_nrows()`。它是设备侧卸载判据的输入（见 `ggml-vulkan.cpp:15693` 的 `ggml_backend_vk_device_offload_op`）。

<!-- src: ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp -->
```c
void vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
    if (!vk_memory_logger_enabled) {
        return;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    const std::string type = device ? "device" : "host";
    allocations[buf->buffer] = size;
    total_device += device ? size : 0;
    total_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
}
//>> ---- ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp:766-779 ----
int64_t ggml_vk_get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}
```

---

## 说明

- 本课覆盖计划清单里的 7 个文件（`ggml/include/ggml-vulkan.h` + `ggml/src/ggml-vulkan/` 下 6 个），全部计入覆盖率。
- 本课另逐字引用了两个**构建期**文件：`ggml/src/ggml-vulkan/CMakeLists.txt` 与 `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`。后者按计划归属 L6-07（整个 `vulkan-shaders/` 目录都在那一课）；本课只在"生成期与主机端如何使用它"这一侧引用它 —— 因为"为什么需要预生成 shader"这条验收点，只有把两处放在一起才看得出来。
- 第 6 幕对照表里的 CUDA / SYCL 行号来自本仓库直接检索（`ggml-cuda.cu` / `ggml-sycl.cpp`），本课不引用它们的源码，故不计入本课覆盖率 —— 它们是 L6-01 / L6-05 的覆盖范围。
