<!-- llama-coverage
ggml/include/ggml-backend.h
ggml/src/ggml-backend.cpp
src/llama-model-loader.cpp
-->

# L4-03 · buffer 与 buffer type — 源文件

**一句话**：`ggml_tensor` 的 `buffer` / `data` 两个字段指向的，就是本课的两个对象 —— **buffer type**（哪一类内存：怎么造、对齐多少、上限多大、是不是 host）与 **buffer**（这一块内存本身）。

本课与 L4-02 的主覆盖文件相同（就是上面这两个 ggml 文件），但焦点不同：L4-02 讲**调度器怎么按 buffer type 切图**，本课讲 **buffer type 本身**——分配、初始化、属性查询，以及 host buffer 与 device buffer 的分工。切分算法不在这里重复。

---

## 一、两个对象：buffer type 与 buffer

公共头把两者都做成**不透明句柄**（`typedef struct ggml_backend_buffer_type *` 与 `typedef struct ggml_backend_buffer *`，第 24-25 行）。所谓 "type" 不是 C++ 的类型，而是"一类内存"：它知道这块内存怎么分配、对齐要求是多少、单块上限多大、CPU 能不能直接读写。

下面这七个函数就是这一层的全部公共 API —— 与 L3-01 逐字展开过的 `ggml_backend_buffer_type_i` 虚表一一对应（6 个函数指针 + 一个 `get_device`）。

<!-- src: ggml/include/ggml-backend.h -->
```c
    //
    // Backend buffer type
    //

    GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer  (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API size_t                ggml_backend_buft_get_alignment (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_max_size  (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    GGML_API bool                  ggml_backend_buft_is_host       (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);
```

## 二、★ 七个公共函数：转发、默认值与归属

七个函数的实现有一个共同形状：**断言 buft 非空，然后转发给 `buft->iface`**。公共 API 里没有一行业务逻辑 —— 这正是 L3-01 说的"公共 API 是虚表的镜像"。

| 函数 | 必需性 | 缺省行为 |
|---|---|---|
| `ggml_backend_buft_name` | 必需 | 无 —— 直接调 `iface.get_name` |
| `ggml_backend_buft_alloc_buffer` | 必需 | 无；`size == 0` 时返回一个哑 buffer |
| `ggml_backend_buft_get_alignment` | 必需 | 无 —— 分配器必须知道对齐 |
| `ggml_backend_buft_get_max_size` | 可选 | `SIZE_MAX`（等于不限） |
| `ggml_backend_buft_get_alloc_size` | 可选 | `ggml_nbytes(tensor)` |
| `ggml_backend_buft_is_host` | 可选 | `false` |
| `ggml_backend_buft_get_device` | —— | 直接读 `buft->device` 字段，不走虚表 |

三处"可选"是这一层的设计要点：**后端可以少填，框架给安全默认值**。其中 `get_alloc_size` 最容易被误解 —— 它的断言（第 69-74 行）明确允许返回值**大于** `ggml_nbytes`（量化类型、repack、以及源码里那个 `TAG_ALLOC_SIZE_EXPAND` 标记的算子）。所以任何分配路径都必须用 `get_alloc_size`，不能拿 `ggml_nbytes` 顶替。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}
```

## 三、★ buffer 与 buft 的绑定：一行赋值

`ggml_backend_buffer_init` 是"buffer 从哪来"的答案：它接收 buft，把它原样存进 `.buft` 字段，再记下 `context`（后端私有数据）、`size` 与 `usage`（初值 `ANY`）。每个 buft 的 `alloc_buffer` 实现最后都要调它 —— 第六节的 CPU 实现就是活例子。

buffer 结构里**没有**对齐、上限、host 这三个属性：它们全部属于 buft。唯一的回程是 `ggml_backend_buffer_get_type`（直接返回 `buffer->buft`），L4-02 里调度器的 `backend_from_buffer()` 读的就是它。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}
//>> ---- ggml/src/ggml-backend.cpp:204-207 ----
ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}
```

## 四、回程：buffer 的属性查询全是转发

`ggml_backend_buffer_get_alignment` / `get_max_size` / `get_alloc_size` / `is_host` 四个函数各自只有一行：先 `ggml_backend_buffer_get_type(buffer)`，再调对应的 `ggml_backend_buft_*`。

这解释了一件容易搞混的事：**"这块 buffer 是不是 host"不是 buffer 的性质，而是它那一类内存的性质**。同一个 buft 造出来的所有 buffer，`is_host` 的答案必然相同。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}
```

## 五、buffer 的生命周期：五个可选回调

buffer 侧的公共函数也是"转发 + 可选"：`free_buffer` 与 `reset` 先判空再调；`clear` 只在非零尺寸时调（注释写明"零尺寸的 buffer 可以不实现它"）；`get_base` 对零尺寸 buffer 直接返回 NULL；`init_tensor` 是"张量挂上来"时的回调，不实现就返回 `GGML_STATUS_SUCCESS`。五个函数都允许后端留空。

`ggml_backend_buffer_free` 的顺序值得一提：先给后端机会释放自己的资源（`iface.free_buffer`），最后才 `delete buffer` —— 后端可以只实现"释放自己的部分"。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}
//>> ---- ggml/src/ggml-backend.cpp:132-159 ----
void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}
//>> ---- ggml/src/ggml-backend.cpp:161-169 ----
void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}
//>> ---- ggml/src/ggml-backend.cpp:209-214 ----
void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}
```

## 六、★ 契约落地：CPU 的 buffer type

这是全套课件里最"完整"的一个真实 buft：`ggml_backend_cpu_buffer_type()`。源码注释写明它定义在 ggml-backend.cpp 里（而不是 CPU 后端目录），**目的是让所有后端都能拿到它**（第 2437-2439 行）。

| 虚表项 | 实现 | 说明 |
|---|---|---|
| `get_name` | 返回 `"CPU"` | 必需项，一行 |
| `alloc_buffer` | `ggml_aligned_malloc` + `ggml_backend_buffer_init` | 分配 + 装虚表 + 绑 buft |
| `get_alignment` | 返回 `TENSOR_ALIGNMENT` | 编译期常量 |
| `get_max_size` | `NULL` | 走默认值 `SIZE_MAX` |
| `get_alloc_size` | `NULL` | 走默认值 `ggml_nbytes` |
| `is_host` | 返回 `true` | CPU 的 buffer 就是 host buffer |

宿主结构体里 `device` 是 `NULL`（注释留着 FIXME），所以对 CPU 的 buft 调 `ggml_backend_buft_get_device` 会得到 NULL —— 用这个返回值之前必须判空。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}
```

## 七、数据面：set / get / memset / cpy_tensor

张量数据的读写入口都长一个样：**先解析出 buffer，再让 buffer 自己搬**。`ggml_backend_tensor_set` / `_get` / `_memset` 三个函数的第一行都是

```text
buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
```

这正是 L1-01 里 `view_src` 的语义在数据面的兑现：视图不拥有数据，读写要走源张量的 buffer。`_get` / `_set` 有越界断言，`_memset` 还额外要求后端实现了 `memset_tensor` —— **不是每个 buffer 都能被填充**。

第 216-222 行的 `ggml_backend_buffer_copy_tensor` 是另一条路：它取 **dst**（视图则取 view_src）的 `cpy_tensor`，由目的地决定能不能直接搬 —— 这是后端做 DMA 的机会，没有就返回 `false`。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}
//>> ---- ggml/src/ggml-backend.cpp:335-348 ----
void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}
//>> ---- ggml/src/ggml-backend.cpp:350-363 ----
void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}
//>> ---- ggml/src/ggml-backend.cpp:409-423 ----
void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}
```

## 八、★ host 与 device 的分工：一个 is_host 定方向

`ggml_backend_tensor_copy` 只有 20 行，却把 host / device 的分工写完了：

| 情形 | 走哪条路 |
|---|---|
| `src` 是 host buffer | `ggml_backend_tensor_set(dst, src->data, ...)` |
| `dst` 是 host buffer | `ggml_backend_tensor_get(src, dst->data, ...)` |
| 两边都不是 host | `ggml_backend_buffer_copy_tensor` → 后端的 `cpy_tensor` |
| 连 `cpy_tensor` 都没有 | `malloc` 一块内存中转，调试版打印 `slow copy` 警告 |

关键在于"host"这个判据来自 buft 的 `is_host`，而它是**可选**的（缺省 `false`）—— 也就是说，一个后端如果不实现 `is_host`，它的 buffer 一律被当成 device buffer，所有拷贝都要绕道 `cpy_tensor` 或 malloc 中转。L2-03 的权重上传（host 侧读出来、set 进显存）正是表中的第一行。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}
```

## 九、★ 手工装配：宿主指针 → buffer → 张量的 data

设备侧有三个 buffer 入口：默认 buft（`ggml_backend_dev_buffer_type`）、host buft（`ggml_backend_dev_host_buffer_type`，**可选，可能返回 NULL**）、以及把已有宿主指针包成 buffer 的 `ggml_backend_dev_buffer_from_host_ptr`。

拿到 buffer 之后，把张量钉上去的是 `ggml_backend_tensor_alloc(buffer, tensor, addr)`：四条断言要求张量还没有 `buffer` / `data` / `view_src`，且 `addr` 必须落在 buffer 区间内（meta buffer 除外）；然后两步赋值，最后调用 `init_tensor` 回调。对照 `ggml_backend_view_init`：视图连 addr 都不用给，直接借用源张量的 buffer 与 `view_offs`。

回顾 L2-03：mmap 权重落位就是 `buffer_from_host_ptr`（把映射包成 buffer）加 `tensor_alloc`（把每个张量钉到各自的 `weight->offs`）这两步。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}
//>> ---- ggml/src/ggml-backend.cpp:615-632 ----
ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}
//>> ---- ggml/src/ggml-backend.cpp:2122-2147 ----
enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}
```

## 十、★ pinned host buffer 的声明：设备能力位

`struct ggml_backend_dev_caps` 是设备对外的能力清单，五个布尔位各自带一行注释。本课关心的两个是：

| 能力位 | 注释原文 | 含义 |
|---|---|---|
| `host_buffer` | `pinned host buffer` | 设备能提供一块**固定（page-lock）的宿主内存** |
| `buffer_from_host_ptr` | `creating buffers from host ptr` | 能把已有宿主指针包成 buffer（mmap） |

注意 `host_buffer` 与 `buffer_from_host_ptr` 是**两位**：前者是"设备能给你一块 pinned 内存"，后者是"设备能接管你已经有的内存"。第九节的 `ggml_backend_dev_host_buffer_type` 对应第一位，`ggml_backend_dev_buffer_from_host_ptr` 对应第二位 —— 也正因为它可选，函数必须能返回 NULL。

<!-- src: ggml/include/ggml-backend.h -->
```c
    // functionality supported by the device
    struct ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
        // mmap is supported for loading
        bool mmap_support;
    };
```

## 十一、★ pinned host buffer 的使用场景（验收点）

答案在模型加载器里：**不用 mmap 时，权重需要从文件读进显存，中间那块中转内存就是 pinned host buffer**。

`llama_model_loader::load_all_data` 里有一个 `upload_backend` lambda：`use_mmap || check_tensors` 为真就直接放弃这条路（第 1527 行）；否则它先确认手上那块 buffer 就是设备的**默认 buft**（第 1546 行），再检查三个能力位 `caps.async && caps.host_buffer && caps.events`（第 1554 行），然后 `ggml_backend_dev_host_buffer_type(dev)` 取 host buft（第 1560 行），用同一个 `ggml_backend_buft_alloc_buffer` 分配 pinned 缓冲（第 1569 行），并从 `ggml_backend_buffer_get_base` 拿基址（第 1578 行）。每个缓冲还配一个事件（`ggml_backend_event_new`，第 1580 行），用来在复用之前等上一次上传完成。

**为什么必须是 pinned**：只有固定不换页的宿主内存才能被设备直接 DMA，异步上传才能与读文件重叠。第六节里 CPU 的默认 buffer 也是 host 内存，但它**不是** pinned —— 所以设备要单独给出一个 `host_buffer_type`。

<!-- src: src/llama-model-loader.cpp -->
```c
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
        auto * dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
            LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);

            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

            auto * event = ggml_backend_event_new(dev);
            if (!event) {
                LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            events.emplace_back(event);
        }
```

## 十二、pinned 缓冲怎么被用掉：读文件 + 异步上传

同一文件的 1699-1735 行是消费端：把文件内容读进 pinned 缓冲（`file->read_raw_unsafe`），再 `ggml_backend_tensor_set_async(upload_backend, cur, ...)` 上传到显存（第 1727 行），紧接着 `ggml_backend_event_record` 记下这次上传（第 1729 行）。下一轮复用同一块缓冲之前，先 `ggml_backend_event_synchronize`（第 1706 行）。

这段代码把"pinned + async + events"三个能力位为什么必须**同时**具备讲清楚了：pinned 提供 DMA 源，async 提供搬运通道，events 提供"这块缓冲什么时候能再用"的判据。少了任何一个，加载器就退回同步读加同步 `set` 的路径。

<!-- src: src/llama-model-loader.cpp -->
```c
                    while (bytes_read < read_end - read_start) {
                        size_t read_size = std::min<size_t>(buffer_size, read_end - read_start - bytes_read);

                        // Align the destination pointer within the pinned buffer
                        uintptr_t ptr_dest_aligned = (reinterpret_cast<uintptr_t>(host_ptrs[buffer_idx]) + alignment - 1) & ~(alignment - 1);

                        // Wait for previous upload to complete before reusing buffer
                        ggml_backend_event_synchronize(events[buffer_idx]);

                        // Read aligned chunk from file
                        file->read_raw_unsafe(reinterpret_cast<void *>(ptr_dest_aligned), read_size);

                        // Calculate actual data portion (excluding alignment padding)
                        uintptr_t ptr_data = ptr_dest_aligned;
                        size_t data_to_copy = read_size;

                        // Skip alignment padding at start of first chunk
                        if (bytes_read == 0) {
                            ptr_data += offset_from_alignment;
                            data_to_copy -= offset_from_alignment;
                        }

                        // Trim alignment padding at end of last chunk
                        if (aligned_offset + bytes_read + read_size > offset + n_size) {
                            data_to_copy -= (read_end - (offset + n_size));
                        }

                        // Async upload actual data to GPU
                        ggml_backend_tensor_set_async(upload_backend, cur,
                                                      reinterpret_cast<void *>(ptr_data), data_read, data_to_copy);
                        ggml_backend_event_record(events[buffer_idx], upload_backend);

                        data_read += data_to_copy;
                        bytes_read += read_size;

                        ++buffer_idx;
                        buffer_idx %= n_buffers;
```

## 十三、后端层的便捷函数：不用知道 buft 也能分配

最后看四个后端级入口：`ggml_backend_get_default_buffer_type`、`ggml_backend_alloc_buffer`、`ggml_backend_get_alignment`、`ggml_backend_get_max_size`。它们全部走同一条路：`backend -> device -> 默认 buft -> 对应的 buft 函数`。

`get_alignment` 与 `get_max_size` 的真正消费者是分配器：L4-01 的 `ggml-alloc.c:1168-1230` 用 `alignment` 把每个节点的尺寸补齐、用 `max_size` 判断"这一块还装得下吗，装不下就换一块 buffer"。这两个值正是本课第二节里 `get_alignment`（必需）与 `get_max_size`（可选）的用途。

<!-- src: ggml/src/ggml-backend.cpp -->
```c
ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}
```

---

## 说明

- 本课逐字引用 `ggml/src/ggml-backend.cpp` 与 `ggml/include/ggml-backend.h` 两个文件，两者都计入覆盖率。
- 第 8 幕与第十一、十二节另逐字引用 `src/llama-model-loader.cpp`（pinned host buffer 的使用方证据，1530-1588 与 1699-1735 行），同样计入覆盖率；该文件的主线（mmap 与权重落位）是 L2-03 的主题，本课不重复。
- 文中提到的 `ggml-alloc.c` 行号（1168-1230）属于 L4-01 的覆盖范围，本课不引用其源码，故不计入本课覆盖率。
