# L6-06 · Vulkan 后端：主机端 — 课件说明

> 层：**L6 · GPU 后端执行** ｜ 前置课：`L6-05`（SYCL 后端：对照另一种"主机端 + 内核"的分工）

## 学习目标

看完这一课，你应该能：

1. 说出 Vulkan 后端为什么需要**预生成的 shader**（本课验收点），并背出 `.comp` 到 pipeline 的步骤链；
2. 说出一个 ggml 节点从"进后端"到"发 dispatch"要经过哪三样 Vulkan 对象，各自在哪个文件里创建；
3. 对照 `L3-01` 的三张虚表契约，指出 Vulkan 的三个实现分别在哪个文件、哪些槽位被显式置 `NULL`；
4. 解释 `descriptor set` 与 `push constants` 的分工：为什么前者可以整套复用，后者每次都重写。

## 覆盖的源文件（9 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-vulkan.h` | 30 |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 16275 |
| `ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp` | 784 |
| `ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp` | 1562 |
| `ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h` | 1111 |
| `ggml/src/ggml-vulkan/ggml-vulkan-types.h` | 1434 |
| `ggml/src/ggml-vulkan/ggml-vulkan-common.h` | 283 |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | 1432 |
| `ggml/src/ggml-vulkan/CMakeLists.txt` | 265 |

> **说明**：本课覆盖计划清单里的 7 个文件（`ggml/include/ggml-vulkan.h` + `ggml/src/ggml-vulkan/` 下 6 个），全部计入覆盖率。
> **说明**：本课另逐字引用了两个**构建期**文件：`ggml/src/ggml-vulkan/CMakeLists.txt` 与 `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`。后者按计划归属 L6-07（整个 `vulkan-shaders/` 目录都在那一课）；本课只在"生成期与主机端如何使用它"这一侧引用它 —— 因为"为什么需要预生成 shader"这条验收点，只有把两处放在一起才看得出来。
> **说明**：第 6 幕对照表里的 CUDA / SYCL 行号来自本仓库直接检索（`ggml-cuda.cu` / `ggml-sycl.cpp`），本课不引用它们的源码，故不计入本课覆盖率 —— 它们是 L6-01 / L6-05 的覆盖范围。

## 场景（9 幕）

1. **Vulkan 后端：主机端在做什么** — 后端对外只有三张虚表；表后面是一条"构建期编 shader、运行期建对象"的流水线。
2. **★ Vulkan 只吃 SPIR-V，所以 shader 必须在构建期编好** — 主机端能拿到的只有一段字节数组；GLSL 前端是一个外部工具，只存在于构建期。
3. **★ 构建系统把 glslc 变成一条编译规则** — 每个 .comp 一条自定义命令；生成的 .cpp 进库，生成的 .hpp 被主机端头文件 include。
4. **运行期只做三件事：module / layout / pipeline** — SPIR-V 是数据，不是代码：主机端把它交给驱动去创建对象，自己不编译。
5. **pipeline 是对象，也是缓存项** — 设备启动时一次建齐，运行期只查表；没编完的才补编，并用条件变量等它。
6. **三张虚表，与 CUDA / SYCL 同构** — L3-01 定义契约，各后端填表。Vulkan 的两个 buffer 面表就在 buffers.cpp 里。
7. **一套布局，一个池，按需增长** — 所有 pipeline 共用同一个 descriptor set layout：12 个 storage buffer 槽位。
8. **push constants：每个 op 一张 POD 结构** — 形状、步长、标量参数都塞在这里；上限由源码的 static_assert 钉死。
9. **一次 dispatch 的全部零件** — 这个模板函数就是本课的落点：pipeline + descriptor set + push constants -> 一次分派。

## 核心结论

### ★ 为什么 Vulkan 必须预生成 shader

主机端拿到的是**字节**，不是源码：`ggml_vk_create_pipeline_func()` 的入参就是 `spv_size` / `spv_data`，第 564 行直接把这段 uint32 数组交给 `ShaderModuleCreateInfo`。

完整的步骤链（每一步都有逐字引用）：

```text
vulkan-shaders/<name>.comp            GLSL 计算着色器源码（L6-07）
  -> process_shaders() 登记变体       vulkan-shaders-gen.cpp:675 起
  -> string_to_spv_func() 调 glslc    vulkan-shaders-gen.cpp:350-357
  -> <name>.spv（SPIR-V 字节码）      构建目录，不在源码仓库
  -> write_output_files() 嵌成数组    vulkan-shaders-gen.cpp:1261-1268
  -> <name>.comp.cpp + .hpp           由 CMake 自定义命令产出
  -> #include "ggml-vulkan-shaders.hpp"   ggml-vulkan-types.h:114
  -> ggml_vk_load_shaders() 取用       ggml-vulkan.cpp:1592 起
  -> createShaderModule / Pipeline     ggml-vulkan.cpp:669 / :744
```

**推论的边界**：运行期能选的只是"已经生成好的变体"——`ggml_vk_op_get_pipeline()` 那张大 switch 和 `ggml_vk_load_shaders()` 里成排的 `_len` / `_data` 就是可选集合的全部。想要一个新内核，必须改 `.comp`、在生成器里登记变体、再重新构建。

### 三张虚表 + 三种零件

| 面 | Vulkan 实现 | 关键槽位 |
|---|---|---|
| `ggml_backend_buffer_type_i` | `ggml-vulkan-buffers.cpp:3` | `is_host = NULL` |
| `ggml_backend_buffer_i` | `ggml-vulkan-buffers.cpp:745` | `reset = NULL` |
| `ggml_backend_i` | `ggml-vulkan.cpp:14830` | 四个 `graph_plan_* = NULL` |

一次 dispatch 的三种零件：**pipeline**（不变）、**descriptor set**（每次变，装数据缓冲）、**push constants**（每次变，装标量参数）。三者由 `ggml_vk_dispatch_pipeline()` 按固定顺序装上。

### 跨课位置

- **L6-05（SYCL）**：同样是"主机端 + 内核"，但内核与主机端是同一门语言（C++），没有独立的着色器编译步骤。
- **L6-07（Vulkan 计算着色器）**：接手本课留下的 `.comp`，讲 `mul_mm` / `mul_mmq` / `flash_attn` 怎么写。
- **L3-01（后端契约）**：本课第 6 幕的三张表就是那份契约的 Vulkan 实现。

## 验收点

- [ ] 保真门禁：17 个引用块全部逐字来自声明的源文件，且位置连续
- [ ] 覆盖度门禁：9 个源文件均被声明
- [ ] 参数门禁：无非法命令行参数
- [ ] 语法检查：通过
- [ ] 渲染门禁：9 幕，无 JS 错误、无布局溢出、交互可用
- [ ] 自检：不看代码，能说出 `struct ggml_tensor` 里哪些字段决定一次内核调用的形状

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
