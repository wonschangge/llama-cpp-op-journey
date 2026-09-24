<!-- llama-coverage
ggml/include/ggml-backend.h
ggml/src/ggml-backend-impl.h
-->

# L3-01 · 后端接口 ggml-backend.h：后端的契约 — 源文件

**一句话**：ggml 不认识任何一块硬件。它只认识一份**契约** —— 一堆函数指针。谁把这份契约填完，谁就是一个后端；CPU、CUDA、Metal、Vulkan 之间的差别，在 ggml 这一侧全部消失了。

本课覆盖两个文件：公共头 `ggml/include/ggml-backend.h`（模型代码能看到的）与内部头 `ggml/src/ggml-backend-impl.h`（**后端实现者**才需要看的）。前者的 115 处 `GGML_API` 只是转发，真正的约定是后者里的五张虚表；这一课重点拆三张：`ggml_backend_device_i`（发现与创建）、`ggml_backend_buffer_type_i`（内存怎么分配）、`ggml_backend_i`（计算怎么执行）。

---

## 一、为什么需要契约：公共头只有不透明句柄

模型代码从头到尾只拿到指针，拿不到结构体。这不是封装洁癖，而是**二进制兼容**的代价控制：后端的结构体可以随便改，只要公共头里的 7 个 typedef 和函数签名不变。

注意 `ggml_backend_graph_plan_t` 是个例外 —— 它是 `void *`，因为"执行计划"的形状完全由后端自己决定。

<!-- src: ggml/include/ggml-backend.h -->
```c
    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
    typedef struct ggml_backend_event * ggml_backend_event_t;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;
    typedef struct ggml_backend_reg * ggml_backend_reg_t;
    typedef struct ggml_backend_device * ggml_backend_dev_t;
```

## 二、★ 三张虚表 = 三个层次

虚表的形状是固定的：一个只装函数指针的 `struct`，名字以 `_i`（interface）结尾；同名的宿主结构体把这张表放在**第一个字段** `iface`，再挂上归属关系和私有 `context`。

| 虚表 | 层次 | 回答的问题 | 函数指针 |
|---|---|---|---|
| `ggml_backend_device_i` | 设备 | 我是谁、能不能跑、怎么变成后端 | 15 |
| `ggml_backend_buffer_type_i` | 内存 | 一块张量内存怎么分配 | 6 |
| `ggml_backend_i` | 计算 | 一张图怎么算出来 | 16 |

文件第一行就声明自己是 internal header；第 11 行的 `GGML_BACKEND_API_VERSION` 是这份契约的版本号，动态加载后端时要用（L3-03）。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    #define GGML_BACKEND_API_VERSION 2

    //
    // Backend buffer type
    //

    struct ggml_backend_buffer_type_i {
        const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        void * context;
    };
```

## 三、★ 设备层：ggml_backend_device_i 的 15 个函数指针

把 15 个指针按职责分四组：自我介绍 5（`get_name` `get_description` `get_memory` `get_type` `get_props`）、创建与内存 4（`init_backend` `get_buffer_type` `get_host_buffer_type` `buffer_from_host_ptr`）、能力判据 3（`supports_op` `supports_buft` `offload_op`）、事件 3（`event_new` `event_free` `event_synchronize`）。

源码注释给 6 个标了 `(optional)`：`get_host_buffer_type`、`buffer_from_host_ptr`、`offload_op` 和事件三件套。剩下的 9 个必须填。

`buffer_from_host_ptr` 的注释写明用途是 "memory mapped models" —— 这正是 L2-03 里权重直接从磁盘映射进内存的那条路。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
    struct ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        const char * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        const char * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        void         (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);

        // device type
        enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        void (*get_props)(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, const char * params);

        // preferred buffer type
        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };
```

## 四、★ 计算层：ggml_backend_i 的 16 个函数指针，必需只有 3 个

16 个指针里，没有落在 `(optional)` 注释组里的只有 `get_name`、`free`、`graph_compute`：

| 组 | 个数 | 说明 |
|---|---|---|
| 必需 | 3 | `get_name` / `free` / `graph_compute` |
| 异步与同步 | 6 | 五个 async 数据访问 + `synchronize`（支持异步就必须填） |
| 预留与可选 | 7 | `graph_plan` 四件套（注释写 "not used currently"）+ 事件两件 + `graph_optimize` |

值得注意：这张表里**没有** `supports_op`。能力判据在设备虚表里 —— 见下一节。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params);
    };
```

## 五、内存层（分配前）：ggml_backend_buffer_type_i

6 个函数指针，3 个必需：`get_name`、`alloc_buffer`、`get_alignment`。三个可选的注释都给出了默认行为：`get_max_size` 默认 `SIZE_MAX`、`get_alloc_size` 默认 `ggml_nbytes`、`is_host` 默认 `false`。

`alloc_buffer` 是这一层的产出：它把字节数变成一块 buffer。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
    struct ggml_backend_buffer_type_i {
        const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
    };
```

## 六、内存层（分配后）：ggml_backend_buffer_i

11 个函数指针，5 个必需：`get_base`、`memset_tensor`、`set_tensor`、`get_tensor`、`clear`。可选的 6 个是 `free_buffer`、`init_tensor`、两个 2d 拷贝、`cpy_tensor`、`reset`。

`init_tensor` 的注释写着 "eg. add tensor extras" —— 它填的正是 L1-01 里 `ggml_tensor` 的 `extra` 字段。回顾 L1-01：张量的 `buffer` 字段指向的就是这里的 `ggml_backend_buffer`。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
    //
    // Backend buffer
    //

    struct ggml_backend_buffer_i {
        // (optional) free the buffer
        void         (*free_buffer)  (ggml_backend_buffer_t buffer);
        // base address of the buffer
        void *       (*get_base)     (ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        // tensor data access
        void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) 2d data copies
        void         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
        // clear the entire buffer
        void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        void         (*reset)        (ggml_backend_buffer_t buffer);
    };
```

## 七、公共 API 是虚表的镜像

这一段把"哪一层负责能力判据"讲透了：`ggml_backend_supports_op` / `ggml_backend_supports_buft` / `ggml_backend_offload_op` 三个后端级公共 API 上面有一行注释 —— "will be removed, use device version instead"。翻回计算层虚表（第 121-156 行）可以验证：那里确实一个 `supports_*` 都没有。

最后一行 `ggml_backend_get_device` 是反向指针：从后端找回它的设备，对应内部头里 `struct ggml_backend` 的 `device` 字段。

<!-- src: ggml/include/ggml-backend.h -->
```c
    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);

    // NOTE: will be removed, use device version instead
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
    GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);
```

## 八、注册面与动态加载入口

`ggml_backend_reg_i` 只有 4 个函数指针：3 个必需（`get_name`、`get_device_count`、`get_device`）加 1 个可选（`get_proc_address`）。宿主结构体 `struct ggml_backend_reg` 的第一个字段是 `api_version`，注释要求初始化成 `GGML_BACKEND_API_VERSION`。

这一段末尾还给出了动态加载的握手符号：后端动态库必须导出 `ggml_backend_init`（返回 `ggml_backend_reg_t`），可选导出 `ggml_backend_score`（分数高的后端优先，0 表示不支持）。注册流程与设备枚举顺序是 L3-02 的主题，动态库的符号解析是 L3-03 的主题。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
    //
    // Backend (reg)
    //

    struct ggml_backend_reg_i {
        const char * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_count)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
    };

    struct ggml_backend_reg {
        int api_version; // initialize to GGML_BACKEND_API_VERSION
        struct ggml_backend_reg_i iface;
        void * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    typedef int                (*ggml_backend_score_t)(void);
```

## 九、把"实现一个新后端"压成一张表

把五个虚表的函数指针个数与必需项数一遍，就是本课验收点的答案：

| 虚表（层次） | 指针数 | 必需 | 可留空 |
|---|---|---|---|
| `ggml_backend_device_i`（发现与创建） | 15 | 9 | 6 |
| `ggml_backend_i`（计算执行） | 16 | 3 | 13 |
| `ggml_backend_buffer_type_i`（内存分配） | 6 | 3 | 3 |
| `ggml_backend_buffer_i`（内存读写） | 11 | 5 | 6 |
| `ggml_backend_reg_i`（注册） | 4 | 3 | 1 |
| **合计** | **52** | **23** | **29** |

判据只有一条：**源码注释标了 `(optional)` 的可以留空，其余必须实现**。三张主虚表（前三行）合计 15 个必需项 —— 这就是"最少要写多少函数"的答案；要真的搬张量、要能被发现，就再加数据面 5 个与注册面 3 个。

```text
最小落地顺序（L8-03 会逐项对照）：
  1. buffer_type_i: get_name / alloc_buffer / get_alignment
  2. buffer_i:      get_base / memset_tensor / set_tensor / get_tensor / clear
  3. backend_i:     get_name / free / graph_compute
  4. device_i:      自我介绍 5 + init_backend / get_buffer_type / supports_op / supports_buft
  5. reg_i:         get_name / get_device_count / get_device
```

---

## 说明

- 本课逐字引用 `ggml/include/ggml-backend.h` 与 `ggml/src/ggml-backend-impl.h` 两个文件，两者都计入覆盖率。
- 文中的函数指针个数、必需/可留空个数，均由脚本按 iface 结构体的起止行统计 `(*name)` 形态的成员，并以 "(optional)" 注释组为判据数出，不是凭印象写的。
- 跨课指路只给课号与主题（L1-01 / L2-03 / L3-02 / L3-03 / L3-04 / L4-03 / L5-01 / L8-03），不引用它们的源码，故不计入本课覆盖率。
