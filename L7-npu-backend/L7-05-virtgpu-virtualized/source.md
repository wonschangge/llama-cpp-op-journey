<!-- llama-coverage
ggml/include/ggml-virtgpu.h
ggml/src/ggml-virtgpu/apir_cs_ggml-rpc-front.cpp
ggml/src/ggml-virtgpu/backend/apir_cs_ggml-rpc-back.cpp
ggml/src/ggml-virtgpu/backend/backend-convert.h
ggml/src/ggml-virtgpu/backend/backend-dispatched-backend.cpp
ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer-type.cpp
ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer.cpp
ggml/src/ggml-virtgpu/backend/backend-dispatched-device.cpp
ggml/src/ggml-virtgpu/backend/backend-dispatched.cpp
ggml/src/ggml-virtgpu/backend/backend-dispatched.gen.h
ggml/src/ggml-virtgpu/backend/backend-dispatched.h
ggml/src/ggml-virtgpu/backend/backend-virgl-apir.h
ggml/src/ggml-virtgpu/backend/backend.cpp
ggml/src/ggml-virtgpu/backend/shared/api_remoting.h
ggml/src/ggml-virtgpu/backend/shared/apir_backend.gen.h
ggml/src/ggml-virtgpu/backend/shared/apir_backend.h
ggml/src/ggml-virtgpu/backend/shared/apir_cs.h
ggml/src/ggml-virtgpu/backend/shared/apir_cs_ggml.h
ggml/src/ggml-virtgpu/backend/shared/apir_cs_rpc.h
ggml/src/ggml-virtgpu/ggml-backend-buffer-type.cpp
ggml/src/ggml-virtgpu/ggml-backend-buffer.cpp
ggml/src/ggml-virtgpu/ggml-backend-device.cpp
ggml/src/ggml-virtgpu/ggml-backend-reg.cpp
ggml/src/ggml-virtgpu/ggml-backend.cpp
ggml/src/ggml-virtgpu/ggml-remoting.h
ggml/src/ggml-virtgpu/include/apir_hw.h
ggml/src/ggml-virtgpu/virtgpu-apir.h
ggml/src/ggml-virtgpu/virtgpu-forward-backend.cpp
ggml/src/ggml-virtgpu/virtgpu-forward-buffer-type.cpp
ggml/src/ggml-virtgpu/virtgpu-forward-buffer.cpp
ggml/src/ggml-virtgpu/virtgpu-forward-device.cpp
ggml/src/ggml-virtgpu/virtgpu-forward-impl.h
ggml/src/ggml-virtgpu/virtgpu-forward.gen.h
ggml/src/ggml-virtgpu/virtgpu-shm.cpp
ggml/src/ggml-virtgpu/virtgpu-shm.h
ggml/src/ggml-virtgpu/virtgpu-utils.cpp
ggml/src/ggml-virtgpu/virtgpu-utils.h
ggml/src/ggml-virtgpu/virtgpu.cpp
ggml/src/ggml-virtgpu/virtgpu.h
-->

# L7-05 · VirtGPU 后端：虚拟化 GPU — 源文件

**一句话**：这是唯一一个**从根上改变数据面**的后端 —— 前面 16 个后端都假设 `tensor->data` 是一个本机可访问的地址，而这里，程序跑在客户机（VM）里，GPU 在宿主机上，**那个地址在程序所在的地址空间里根本不存在**。于是指针退化成"buffer 内偏移"，数据靠一块共享内存窗口搬过去，一切操作变成一条条命令。

本课覆盖 `ggml/src/ggml-virtgpu/` 下 **38 个文件**加公共头 `ggml/include/ggml-virtgpu.h`，共 **39 个**（`wc -l` 合计 4414 行），按"墙的哪一侧"分四族。逐字引用 13 个，其余在 `source.md` 第十五节按族说明。

---

## 一、覆盖清单与四个族

本课覆盖计划里 `L7-05` 的全部 **39 个**源文件：`ggml/src/ggml-virtgpu/` 下 38 个 加公共头 `ggml/include/ggml-virtgpu.h`，`wc -l` 合计 **4414 行**；后缀分布是 20 个 `.cpp` + 19 个 `.h`，没有 `.c`。按"跑在墙的哪一侧"分四族：

| 族 | 文件数 | 目录 | 角色 |
|---|---|---|---|
| 客户机侧 frontend | 18 | `ggml/src/ggml-virtgpu/`（顶层）+ `ggml/include/ggml-virtgpu.h` | 实现 ggml 的三张虚表，把调用翻译成命令 |
| 宿主机侧 backend | 11 | `ggml/src/ggml-virtgpu/backend/` | 把命令翻译回 ggml 调用，交给真后端 |
| 两侧共用协议 | 6 | `ggml/src/ggml-virtgpu/backend/shared/` | 命令枚举、编解码器、tensor 线格式 |
| 通用工具 | 4 | `ggml/src/ggml-virtgpu/virtgpu-shm.*` `virtgpu-utils.*` | 共享内存窗口与稀疏数组 |

客户机侧与宿主机侧被编译成两个不同的库（`ggml-virtgpu` 与 `ggml-virtgpu-backend`）。下面这个头文件是双方的共同基准 —— 它自己声明要与 virglrenderer 的 `apir-protocol.h` 保持一致：

<!-- src: ggml/src/ggml-virtgpu/backend/shared/api_remoting.h -->
```cpp
/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#include <unistd.h>

#include <cstdint>

#define APIR_PROTOCOL_MAJOR 0
#define APIR_PROTOCOL_MINOR 2

#define APIR_HANDSHAKE_MAGIC 0xab1e

enum ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_LENGTH = 3,
};
```

## 二、★ 过墙的 tensor：apir_rpc_tensor

一个 `ggml_tensor` 里全是指针（`data` / `buffer` / `src[]` / `view_src` / `extra`），不能直接 memcpy 过去。这个 POD 就是它的线格式：所有指针字段换成 `uint64_t`，`ne[]` / `nb[]` / `op_params[]` 原样保留。

注意 `nb[]` 被完整带过去了：**跨墙的 tensor 依然可以是任意步长的**，布局信息没有丢，丢的只是"地址"。

<!-- src: ggml/src/ggml-virtgpu/backend/shared/apir_cs_rpc.h -->
```cpp
struct apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];

    char padding[4];
};
```

## 三、★ tensor->data 是偏移，不是指针

序列化的最后一步：如果 tensor 有数据，就把 `data` 换成它相对 buffer base 的偏移。

这一行是本课的核心证据。L1-01 说"`data` 回答数据在哪"；在这里，客户机里的 `tensor->data` 之所以看起来像个地址，只是因为 ggml-alloc 当初是拿宿主侧返回的 base 算出来的 —— 那个数在客户机的地址空间里没有任何映射。

<!-- src: ggml/src/ggml-virtgpu/apir_cs_ggml-rpc-front.cpp -->
```cpp
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    if (tensor->data) {
        if (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
```

## 四、宿主机侧把偏移还原成指针

反序列化时，宿主把自己的 `buffer_start` 加回去，并用两条断言把偏移限制在 buffer 内。这是隔着特权边界唯一还能做的地址校验：**校验字节区间，而不是校验指针**。

<!-- src: ggml/src/ggml-virtgpu/backend/apir_cs_ggml-rpc-back.cpp -->
```cpp
    uint64_t tensor_data = tensor->data;
    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size  = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size  = (uint64_t) ggml_backend_buffer_get_size(result->buffer);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_data += buffer_start;

        GGML_ASSERT(tensor_data + tensor_size >= tensor_data);  // check for overflow
        GGML_ASSERT(tensor_data >= buffer_start && tensor_data + tensor_size <= buffer_start + buffer_size);
```

## 五、命令枚举：23 条命令 + 一个计数

整个后端的"接口"就是这张枚举表：device 组 0-9（10 条）、buffer-type 组 10-15（6 条）、buffer 组 16-21（6 条）、backend 组 22（1 条），共 **23 条命令**。`APIR_BACKEND_DISPATCH_TABLE_COUNT = 23` 是最后一条命令号加一，宿主侧用它做下标越界检查。

第 13 号 `BUFFER_TYPE_IS_HOST` 已废弃（宿主的实现恒返回 false），但**编号与槽位都保留**。

<!-- src: ggml/src/ggml-virtgpu/backend/shared/apir_backend.gen.h -->
```cpp
typedef enum ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_COUNT        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_MEMORY       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE   = 12,
    APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER   = 14,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE = 15,

    /* buffer */
    APIR_COMMAND_TYPE_BUFFER_GET_BASE    = 16,
    APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = 17,
    APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_BUFFER_CLEAR       = 20,
    APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_TABLE_COUNT = 23,
} ApirBackendCommandType;
```

## 六、分发表：命令号就是数组下标

宿主机侧把命令号直接当数组下标用（`backend.cpp:137` 的 `apir_backend_dispatch_table[cmd_type]`，越界由 `backend.cpp:131-135` 拦下）。表的顺序必须与枚举逐条对齐 —— 这是生成的代码，两边由同一份定义产出。

<!-- src: ggml/src/ggml-virtgpu/backend/backend-dispatched.gen.h -->
```cpp
static const backend_dispatch_t apir_backend_dispatch_table[APIR_BACKEND_DISPATCH_TABLE_COUNT] = {

    /* device */

    /* APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT  = */ backend_device_get_device_count,
    /* APIR_COMMAND_TYPE_DEVICE_GET_COUNT  = */ backend_device_get_count,
    /* APIR_COMMAND_TYPE_DEVICE_GET_NAME  = */ backend_device_get_name,
    /* APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = */ backend_device_get_description,
    /* APIR_COMMAND_TYPE_DEVICE_GET_TYPE  = */ backend_device_get_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_MEMORY  = */ backend_device_get_memory,
    /* APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP  = */ backend_device_supports_op,
    /* APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = */ backend_device_get_buffer_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_PROPS  = */ backend_device_get_props,
    /* APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = */ backend_device_buffer_from_ptr,

    /* buffer-type */

    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME  = */ backend_buffer_type_get_name,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = */ backend_buffer_type_get_alignment,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE  = */ backend_buffer_type_get_max_size,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST  = */ backend_buffer_type_is_host /* DEPRECATED */,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER  = */ backend_buffer_type_alloc_buffer,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE  = */ backend_buffer_type_get_alloc_size,

    /* buffer */

    /* APIR_COMMAND_TYPE_BUFFER_GET_BASE  = */ backend_buffer_get_base,
    /* APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = */ backend_buffer_set_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = */ backend_buffer_get_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = */ backend_buffer_cpy_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CLEAR  = */ backend_buffer_clear,
    /* APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER  = */ backend_buffer_free_buffer,

    /* backend */

    /* APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE  = */ backend_backend_graph_compute,
};
```

## 七、客户机侧的转发入口：一条宏

23 个 `virtgpu-forward-*` 函数长得一模一样：准备编码器、写参数、发出去、读回复。所以它被压成了两个宏 —— `REMOTE_CALL_PREPARE` 与 `REMOTE_CALL`。

注意 `REMOTE_CALL_PREPARE` 传的命令号是 `APIR_COMMAND_TYPE_FORWARD`，真正的业务命令号放在 `cmd_flags` 里；`REMOTE_CALL` 的 `max_wait_ms` 传 0，即不带超时。

<!-- src: ggml/src/ggml-virtgpu/virtgpu-forward-impl.h -->
```cpp
#define REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    const char * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    do {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        if (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } while (0)

#define REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    do {                                                                                                    \
        ret_name = (ApirForwardReturnCode) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, NULL); \
        if (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        if (ret_name < APIR_FORWARD_BASE_INDEX) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (ApirForwardReturnCode) (ret_name - APIR_FORWARD_BASE_INDEX);                            \
        if (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } while (0)
```

## 八、一条命令的固定前缀

所有命令共用同一个编码器，前三段固定：`cmd_type`（int32）、`cmd_flags`（int32）、`reply_res_id`（uint32）。第三条把"回复写到哪块内存"写进了命令本身 —— 这正是第九、十节讲的同步机制得以成立的原因。

编码缓冲区是 `thread_local` 的 4KiB 栈数组：命令流不为此分配堆内存。

<!-- src: ggml/src/ggml-virtgpu/virtgpu.cpp -->
```cpp
apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local char encoder_buffer[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_buffer,
        .start = encoder_buffer,
        .end   = encoder_buffer + sizeof(encoder_buffer),
        .fatal = false,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    if (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_LENGTH;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t reply_res_id = gpu->reply_shmem.res_id;
    apir_encode_uint32_t(&enc, &reply_res_id);

    return &enc;
}
```

## 九、提交：一个 execbuffer，fence 字段全为 0

命令交给内核的方式是一次 `DRM_IOCTL_VIRTGPU_EXECBUFFER`。注意 `fence_fd`、`num_in_syncobjs`、`num_out_syncobjs` 全部为 0：**这个后端不使用 DRM 的 fence 或 syncobj 来同步**。context 初始化时也做了同一个选择（`virtgpu.cpp:337`，`VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK` 的值写 0，注释写明不要在 fence 信号时产生 drm_events）。

<!-- src: ggml/src/ggml-virtgpu/virtgpu.cpp -->
```cpp
    volatile std::atomic_uint * atomic_reply_notif = (volatile std::atomic_uint *) gpu->reply_shmem.mmap_ptr;
    *atomic_reply_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer args = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .num_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .num_in_syncobjs  = 0,
        .num_out_syncobjs = 0,
        .in_syncobjs      = 0,
        .out_syncobjs     = 0,
    };

    *decoder = NULL;

    int ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &args);
```

## 十、等待：一个 atomic 计数器 + 15µs 轮询

回复窗口的第一个 4 字节是一个 `atomic_uint`：宿主执行完把它改成非 0，客户机用 acquire 语义轮询它，读到非 0 就退出循环，再按源码注释所说"extract the actual return value from the notif flag"减一得到返回码。没有到就 `os_time_sleep(15)` 微秒再看。

这解释了为什么 `ggml_backend_i` 里 `synchronize` / `event_record` / `event_wait` 三个槽位在这个后端上全是 NULL（`ggml-backend.cpp:41,47,48`）：**每次远程调用本来就是同步的**，等待已经内建在 `remote_call()` 里了。

<!-- src: ggml/src/ggml-virtgpu/virtgpu.cpp -->
```cpp
    bool     timedout    = false;
    uint32_t notif_value = 0;
    while (true) {
        notif_value = std::atomic_load_explicit(atomic_reply_notif, std::memory_order_acquire);

        if (notif_value != 0) {
            break;
        }

        int64_t base_sleep_us = 15;

        os_time_sleep(base_sleep_us);

        if (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            long long end_time    = (long long) ts_end.tv_sec * 1000000000LL + ts_end.tv_nsec;
            float     duration_ms = (end_time - start_time) / 1000000;

            if (duration_ms > max_wait_ms) {
                timedout = true;
                break;
            }
        }
    }
```

## 十一、★ 数据面的全流程：memcpy + 一条命令

回到验收点：一个客户机里的 tensor，数据是怎么到宿主 GPU 的？全部动作就在这个函数里。

第 53 行是唯一的字节搬运点：`memcpy(shmem->mmap_ptr, data, size)` —— 目标是客户机自己 mmap 出来的窗口地址，在 VM 里它是一段普通内存。第 54 行把窗口的 `res_id` 放进命令，宿主用同一个 id 换回它那一侧的指针。

链路上**没有任何一处把设备内存映射进客户机**。这直接回答了本课的验收点。

<!-- src: ggml/src/ggml-virtgpu/virtgpu-forward-buffer.cpp -->
```cpp
    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_SET_TENSOR);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   temp_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &temp_shmem;
    bool            using_shared_shmem = false;

    if (size <= gpu->data_shmem.mmap_size) {
        // Lock mutex before using shared data_shmem buffer
        if (mtx_lock(&gpu->data_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = true;
        shmem              = &gpu->data_shmem;

    } else if (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    memcpy(shmem->mmap_ptr, data, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_size_t(encoder, &offset);
    apir_encode_size_t(encoder, &size);
```

## 十二、宿主侧接住数据

宿主收到 `APIR_COMMAND_TYPE_BUFFER_SET_TENSOR` 后：解出 buffer 句柄、tensor、窗口 `res_id`、`offset`、`size`，用 `get_shmem_ptr()` 把 `res_id` 换成宿主机侧的指针，再调用**真后端**的 `buffer->iface.set_tensor()`。到这一步，数据才真正进入设备内存。

<!-- src: ggml/src/ggml-virtgpu/backend/backend-dispatched-buffer.cpp -->
```cpp
uint32_t backend_buffer_set_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!buffer || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        return 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t offset;
    apir_decode_size_t(dec, &offset);

    size_t size;
    apir_decode_size_t(dec, &size);

    if (validate_buffer_operation(offset, size, __func__) != 0) {
        return 1;
    }

    void * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    if (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        return 1;
    }

    buffer->iface.set_tensor(buffer, tensor, shmem_data, offset, size);

    return 0;
}
```

## 十三、两块常驻窗口与一把锁

客户机侧一共只建两块常驻共享内存：数据窗口与回复窗口。它们的尺寸、以及保护数据窗口的那把互斥锁，都写在 `struct virtgpu` 里。

数据窗口 24MiB 是"够用就好"的选择：`set_tensor` 超过它就临时另建一块；回复窗口只有 16KiB，因为回复里装的是标量、字符串与句柄，不是张量数据。

<!-- src: ggml/src/ggml-virtgpu/virtgpu.h -->
```cpp
#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000
//>> ---- ggml/src/ggml-virtgpu/virtgpu.h:75-80 ----
    /* APIR communication pages */
    virtgpu_shmem reply_shmem;
    virtgpu_shmem data_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t data_shmem_mutex;
```

## 十四、窗口是怎么造出来的

客户机侧建窗口只有四个动作：对齐、建 blob、导出 mmap 偏移、mmap。`VIRTGPU_BLOB_MEM_HOST3D` 说明内存由宿主的 GPU 栈分配；`VIRTGPU_BLOB_FLAG_USE_MAPPABLE` 说明客户机可以映射它 —— 两个标志缺一不可。

宿主侧对应的能力声明只有两个回调：`get_config` 与 `get_shmem_ptr`（`backend-virgl-apir.h:16-19`）。

<!-- src: ggml/src/ggml-virtgpu/virtgpu-shm.cpp -->
```cpp
int virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    if (!gem_handle) {
        return 1;
    }

    void * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    if (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        return 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_size  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    return 0;
```

## 十五、其余 26 个文件按族说明

本课逐字引用了上面 13 个文件；其余 26 个按族列出各自的职责。下表"行数"一列取自 `wc -l`。

**客户机侧（12 个未逐字引用）**

| 文件 | 行数 | 职责 |
|---|---|---|
| `ggml-backend-reg.cpp` | 213 | 注册表：`ggml_backend_virtgpu_reg()`；一次性把设备名/类型/显存/buffer type 全部缓存（34-58 行） |
| `ggml-backend.cpp` | 72 | `ggml_backend_i` 虚表；`graph_compute` 转发给 `apir_backend_graph_compute`；`synchronize` / `event_*` 全为 NULL |
| `ggml-backend-device.cpp` | 160 | `ggml_backend_device_i` 虚表；`buffer_from_host_ptr` 两条路径；`get_props` 末尾把 `caps.buffer_from_host_ptr` 强制改回 false |
| `ggml-backend-buffer.cpp` | 123 | `ggml_backend_buffer_i` 两张表（普通 / from_ptr）；`set_tensor` 按 `is_from_ptr` 二选一：要么 memcpy，要么发命令 |
| `ggml-backend-buffer-type.cpp` | 81 | `ggml_backend_buffer_type_i` 虚表；`alloc_buffer` 里按宿主的 `buffer_from_host_ptr` 能力选路 |
| `ggml-remoting.h` | 71 | 两侧的上下文结构（`shared_memory` 向量、`apir_buffer_context_t`）与两个 `ggml_buffer*_to_apir_handle` 换算 |
| `virtgpu-forward-device.cpp` | 194 | 10 条 device 命令的客户机实现；`apir_device_buffer_from_ptr` 是客户机侧另一条建窗口的分配路径 |
| `virtgpu-forward-buffer-type.cpp` | 110 | 6 条 buffer-type 命令的客户机实现 |
| `virtgpu-forward-backend.cpp` | 58 | 把 cgraph 序列化成一个字节缓冲放进窗口，再发一条 `GRAPH_COMPUTE` |
| `virtgpu-forward.gen.h` | 54 | 生成物：客户机侧转发函数的原型 |
| `virtgpu-apir.h` | 15 | `apir_buffer_context_t` 的定义（host_handle + shmem + buft_host_handle） |
| `include/apir_hw.h` | 9 | `virgl_renderer_capset_apir` 的镜像：协议版本 + `supports_blob_resources`（公共头 `ggml/include/ggml-virtgpu.h` 在末幕逐字引用） |

**宿主机侧（8 个未逐字引用）**

| 文件 | 行数 | 职责 |
|---|---|---|
| `backend/backend.cpp` | 144 | `apir_backend_initialize` / `apir_backend_dispatcher` / `apir_backend_deinit`：`dlopen` 真后端库并按命令号查表 |
| `backend/backend-dispatched.cpp` | 51 | 全局 `reg` / `dev` / `bck`，以及把真后端的注册函数变成一次初始化 |
| `backend/backend-dispatched-device.cpp` | 149 | 10 条 device 命令的宿主实现；`buffer_from_ptr` 把窗口指针交给真后端 |
| `backend/backend-dispatched-buffer-type.cpp` | 105 | 6 条 buffer-type 命令的宿主实现；`is_host` 恒返回 false；`alloc_buffer` 成功后记入跟踪表 |
| `backend/backend-dispatched-backend.cpp` | 102 | 反序列化图、按需检查 `supports_op`、调真后端 `graph_compute`、异步后端补一次 `synchronize` |
| `backend/backend-dispatched.h` | 27 | `virgl_apir_context` 与 `backend_dispatch_t` 函数指针类型 |
| `backend/backend-convert.h` | 13 | 宿主机侧的 `ggml_buffer_to_apir_handle`：句柄就是指针本身 |
| `backend/backend-virgl-apir.h` | 32 | 宿主库对 virglrenderer 的两个回调声明与三个导出函数 |

**两侧共用（3 个未逐字引用：`api_remoting.h` / `apir_cs_rpc.h` / `apir_backend.gen.h` 已在前文逐字引用）**

| 文件 | 行数 | 职责 |
|---|---|---|
| `backend/shared/apir_cs.h` | 378 | 编解码器：encoder / decoder、fatal 标志、按类型读写，全部是 header-only 的 inline 函数 |
| `backend/shared/apir_cs_ggml.h` | 232 | ggml 特有的编解码：tensor 的两种编码（线格式 / 内联结构体）、buffer 句柄、cgraph |
| `backend/shared/apir_backend.h` | 50 | 初始化返回码与两个句柄类型 `apir_buffer_type_host_handle_t` / `apir_buffer_host_handle_t` |

**通用工具（3 个未逐字引用：`virtgpu-shm.cpp` 在第十四节逐字引用）**

| 文件 | 行数 | 职责 |
|---|---|---|
| `virtgpu-shm.h` | 23 | `struct virtgpu_shmem`：res_id / mmap_size / mmap_ptr / gem_handle 四个字段 |
| `virtgpu-utils.cpp` | 179 | `util_sparse_array`：可按 id 索引的稀疏数组，节点用 CAS 无锁挂接 |
| `virtgpu-utils.h` | 86 | 对齐宏、`os_time_sleep`、计时器 —— 第十节的 15µs 轮询就来自这里 |

---

## 说明

- 本课声明 39 个源文件 = 计划里 L7-05 的全部：`ggml/src/ggml-virtgpu/` 下 38 个 加 `ggml/include/ggml-virtgpu.h`，**无一遗漏**。其中 13 个被逐字引用：核心 8 个（`virtgpu.cpp`、`virtgpu.h`、`virtgpu-shm.cpp`、`virtgpu-forward-buffer.cpp`、`backend/backend-dispatched-buffer.cpp`、`apir_cs_ggml-rpc-front.cpp`、`backend/apir_cs_ggml-rpc-back.cpp`、`virtgpu-forward-impl.h`），协议与公共头 5 个（`backend/shared/api_remoting.h`、`backend/shared/apir_cs_rpc.h`、`backend/shared/apir_backend.gen.h`、`backend/backend-dispatched.gen.h`、`ggml/include/ggml-virtgpu.h`）；其余 26 个在第十五节按族说明。
- **覆盖域外、不计入覆盖率**的文件（本课只在文字里提及，后缀不在本视角的覆盖域内，故不进覆盖声明）：`ggml/src/ggml-virtgpu/ggmlremoting_functions.yaml`（166 行，远程函数的 YAML 定义）、`ggml/src/ggml-virtgpu/regenerate_remoting.py`（333 行，由 YAML 生成三个 `*.gen.h`）、`ggml/src/ggml-virtgpu/CMakeLists.txt` 与 `ggml/src/ggml-virtgpu/backend/CMakeLists.txt`（两个库的构建定义）。
- 宿主机侧的库并不在本仓库内被链接成一个可运行程序：它是被 virglrenderer 用 `dlopen` 加载的（`backend/backend.cpp:76`），加载路径来自 hypervisor 的配置。本课不实测这条链路，只做源码级断言。
- 本课不讨论任何命令行参数；参数门禁的真值集来自真实二进制的帮助输出。
