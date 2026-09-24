<!-- llama-coverage
ggml/include/ggml-cuda.h
ggml/src/ggml-cuda/common.cuh
ggml/src/ggml-cuda/ggml-cuda.cu
-->

# L6-01 · CUDA 后端骨架：一个 switch + 一个 stream — 源文件

**一句话**：CUDA 后端看着有 5856 行，骨架其实只有两件东西 —— **一张 `switch (dst->op)` 的表**（决定 op 怎么跑）和**一组 stream**（决定活儿发到哪条流上）；而"这个 op 到底支不支持"不在执行路径里，在 `ggml_backend_cuda_device_supports_op()` 里。

回顾 L3-01：后端契约是三张虚表（backend / device / buffer_type）。这一课把 CUDA 的三张实现找出来，再顺着 `graph_compute` 走一遍：图怎么被遍历、节点怎么被分派、图什么时候被 capture 成 CUDA graph。

---

## 一、设备属性：ggml_cuda_device_info

后端初始化时把每个设备探测一遍，结果缓存在这个结构里；之后所有"这台卡行不行"的判断都是读 `ggml_cuda_info()`（common.cuh:1202）。注意 `devices[]` 是按设备号索引的定长数组 —— 长度就是公共头里的 `GGML_CUDA_MAX_DEVICES`。

<!-- src: ggml/src/ggml-cuda/common.cuh -->
```c
struct ggml_cuda_device_info {
    int device_count;           // number of (possibly virtual) devices exposed to the rest of ggml
    int physical_device_count;  // number of physical CUDA devices actually present

    struct cuda_device_info {
        int     cc;                             // compute capability
        int     nsm;                            // number of streaming multiprocessors
        size_t  smpb;                           // max. shared memory per block
        size_t  smpbo;                          // max. shared memory per block (with opt-in)
        bool    integrated;                     // Device is integrated as opposed to discrete
        bool    vmm;                            // virtual memory support
        size_t  vmm_granularity;                // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;                      // Number of threads in a dispatch
        bool    supports_cooperative_launch;    // whether cooperative launch is supported
        int     physical_device;                // backing physical CUDA device for this (virtual) device
        int     physical_share_count;           // number of (virtual) devices sharing this device's physical GPU
        int     virtual_index;                  // index of this (virtual) device among those sharing its physical GPU
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};
```

## 二、能力号：cc 的编码与 GGML_CUDA_CC_* 阶梯

CUDA 侧的 cc 是 `100 * major + minor`（ggml-cuda.cu:349），所以 7.5 写成 `750`。源码为每一代架构留了一个常量，能力判断就是拿 cc 和这些常量比大小 —— 例如图 capture 的门槛是 `cc < GGML_CUDA_CC_VOLTA`（ggml-cuda.cu:4409）。AMD / MUSA 的 cc 各自加一个偏移量，因此判断"是不是 NVIDIA"用 `GGML_CUDA_CC_IS_NVIDIA(cc)`。

<!-- src: ggml/src/ggml-cuda/common.cuh -->
```c
#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_DP4A            610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_ORIN            870
#define GGML_CUDA_CC_ADA_LOVELACE    890
#define GGML_CUDA_CC_HOPPER          900
// While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core instructions available to 1200 family, see
// https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms
#define GGML_CUDA_CC_BLACKWELL       1200
#define GGML_CUDA_CC_DGX_SPARK       1210
#define GGML_CUDA_CC_RUBIN           1300
#define GGML_CUDA_CC_OFFSET_AMD      0x1000000
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_MTHREADS)
```

## 三、显存：池接口与 RAII 包装

算子的临时缓冲不直接 cudaMalloc，而是走池：`ggml_cuda_pool` 只有 `alloc` / `free` 两个纯虚函数，`ggml_cuda_pool_alloc<T>` 是模板 RAII 包装 —— 构造时分配、析构时归还，所以算子代码里看不到显式的 free。池本身按 (设备, stream 槽) 分开：`pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]`（common.cuh:1559）。

<!-- src: ggml/src/ggml-cuda/common.cuh -->
```cpp
struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};
```

## 四、池的选型：VMM 还是 legacy

池有两种实现：`ggml_cuda_pool_vmm`（虚拟内存预留）与 `ggml_cuda_pool_leg`（传统缓冲池，`MAX_BUFFERS = 256`，见 ggml-cuda.cu:418-428）。选哪个由设备属性 `vmm` 决定 —— 这也是"设备属性会影响执行策略"的一个具体例子。

<!-- src: ggml/src/ggml-cuda/ggml-cuda.cu -->
```cpp
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}
```

## 五、buffer_type 虚表：显存怎么分配

第三张虚表只关心"显存"这件事：名字、分配、对齐（128 字节）、一个张量要多少字节。注意 `get_alloc_size` 对量化类型会补齐到 `MATRIX_ROW_PADDING` 的整数倍 —— 这不是浪费，是给 kernel 的行访问留的对齐（见 L6-02）。

<!-- src: ggml/src/ggml-cuda/ggml-cuda.cu -->
```c
static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};
```

## 六、supports_op 的兜底也是 return false

函数末尾把若干算子直接放行（`return true`），最后用 `default: return false` 收尾。也就是说：**"不支持"是默认答案，"支持"必须被显式列出**。这份名单必须与 `ggml_cuda_compute_forward()` 的 88 个 case 保持同步，否则执行时会在 ggml-cuda.cu:4359 的断言上炸掉。

<!-- src: ggml/src/ggml-cuda/ggml-cuda.cu -->
```c
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}
```

---

## 说明

- 本课覆盖三个文件：`ggml/include/ggml-cuda.h`、`ggml/src/ggml-cuda/common.cuh`、`ggml/src/ggml-cuda/ggml-cuda.cu`，均计入覆盖率。
- 第 7 幕引用的 CPU 侧对照文件 `ggml/src/ggml-cpu/ggml-cpu.c` 只在 caption 里指路，本课不引用其源码，故不计入本课覆盖率。
- "88 个 case"的统计口径：`ggml-cuda.cu` 中 `ggml_cuda_compute_forward()` 内（2067-2432 行）缩进为 8 空格的 `case GGML_OP_*:` 标签，`grep -c` 得 88，去重后仍是 88。
