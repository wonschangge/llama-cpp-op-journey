<!-- llama-coverage
ggml/include/ggml-cann.h
ggml/src/ggml-cann/ggml-cann.cpp
ggml/src/ggml-cann/aclnn_ops.h
ggml/src/ggml-cann/aclnn_ops.cpp
ggml/src/ggml-cann/acl_tensor.cpp
ggml/src/ggml-cann/common.h
ggml/src/ggml-cann/acl_tensor.h
-->

# L7-01 · CANN 后端（Ascend NPU）：算子由厂商库提供 — 源文件

**一句话**：CANN 后端不是一个"自己写 kernel 的后端"，而是一个**翻译层** —— 它把 ggml 图上的每个 op 翻译成华为 CANN 算子库（aclnn）里一个**预编译算子**的调用。整课 10058 行源码里，没有一行是矩阵乘的内核代码。

对比 L6-01：CUDA 后端的 `mul_mat` 是手写的 CUDA kernel（`.cu` 文件里的 `__global__` 函数）；CANN 后端的 `mul_mat` 是 `aclnnMm` / `aclnnBatchMatMul` / `aclnnMatmul` 三次库调用之一。**这就是 NPU 后端与 GPU 后端最大的差别，也是本课的核心。**

这份"翻译"有代价：厂商库里没有的算子，后端只能**说不**（`supports_op` 返回 false），那部分图会留在 CPU 上 —— 这正是 L4-02 里调度器切分的输入。

---

## 一、公共 API：两个入口

`ggml-cann.h` 只有 123 行。它对外只暴露两件事：**注册自己**（让 L3-02 的注册表发现它）和**按设备号创建一个后端实例**。其余全是设备信息查询。

<!-- src: ggml/include/ggml-cann.h -->
```c
#define GGML_CANN_MAX_DEVICES 16

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cann_reg(void);

/**
 * @brief Initializes the CANN backend for a specified device.
 *
 * This function initializes the CANN backend for the given device.
 * It verifies the device index, allocates a context, and creates a backend
 * instance.
 *
 * @param device The index of the device to initialize.
 * @return A pointer to the initialized backend instance, or nullptr on failure.
 */
GGML_BACKEND_API ggml_backend_t ggml_backend_cann_init(int32_t device);
```

## 二、★ aclnn 的两段式调用：整个后端的接口只有这一个宏

这段注释值得逐字读：宏"在指定 stream 上提交一个异步任务"，workspace 来自分配器，而分配器可以立刻"释放"这块内存 —— 因为**同一条 stream 上的任务按队列顺序执行**，后面的任务不会提前读到它。这就是 CANN 后端把"内存池 + stream"绑在一起讲的原因。

<!-- src: ggml/src/ggml-cann/aclnn_ops.h -->
```c
/**
 * @brief Launches an asynchronous task using the memory allocator.
 *
 * This macro submit an asynchronous task on the specified stream.
 * The task uses memory allocated by the allocator. It is guaranteed
 * that the memory will not be accessed by other tasks until this task
 * completes, due to the sequential execution order within the same stream.
 *
 * @param OP_NAME aclnn operator name.
 * @param args Additional arguments required by the task.
 *
 * @note
 * Memory from the allocator will be "freed" immediately and can be
 * reallocated to other pointers. However, it won't be accessed by any
 * other task before this asynchronous task ends, because all tasks in the
 * same stream are executed in queue order.
 */

#    define GGML_CANN_CALL_ACLNN_OP(CTX, OP_NAME, ...)                                           \
        do {                                                                                     \
            uint64_t        workspaceSize = 0;                                                   \
            aclOpExecutor * executor;                                                            \
            void *          workspaceAddr = nullptr;                                             \
            ACL_CHECK(aclnn##OP_NAME##GetWorkspaceSize(__VA_ARGS__, &workspaceSize, &executor)); \
            /* workspace should alloced in main thread to keep malloc order when using vmm. */   \
            if (workspaceSize > 0) {                                                             \
                ggml_cann_pool_alloc workspace_allocator(CTX.pool(), workspaceSize);             \
                workspaceAddr = workspace_allocator.get();                                       \
            }                                                                                    \
            ACL_CHECK(aclnn##OP_NAME(workspaceAddr, workspaceSize, executor, CTX.stream()));     \
        } while (0)
```

## 三、算子名字来自厂商头文件

`aclnn_ops.cpp` 开头是一长串 `#include <aclnnop/aclnn_*.h>` —— **每个 aclnn 算子一个头文件**。这份 include 清单就是"这个后端能支持哪些 op"的第一手证据。

<!-- src: ggml/src/ggml-cann/aclnn_ops.cpp -->
```c
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_add_rms_norm.h>
#include <aclnnop/aclnn_addcdiv.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_avgpool2d.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_div.h>
```

## 四、分派的另外两种形态

第 4 幕引了二元算子的模板形态。一元算子走的是宏形态（两层 switch：先 `GGML_OP_UNARY`，再 `ggml_get_unary_op`）；矩阵乘、注意力这类大算子走专用函数形态。

<!-- src: ggml/src/ggml-cann/ggml-cann.cpp -->
```c
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    break;
                case GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    break;
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    break;
                case GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    {
                        auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    break;
```

## 五、专用函数形态：MUL_MAT 与 FLASH_ATTN_EXT

第 4 幕的动画代码只引到 `GGML_OP_UNARY` 为止，第三种形态（专用函数）在这里补齐。`GGML_OP_MUL_MAT` 的 case 只有两行：调用 `ggml_cann_mul_mat`。所有关于维度、量化类型、内存排布的判断都在 `aclnn_ops.cpp` 里 —— **分派层保持极薄**。

<!-- src: ggml/src/ggml-cann/ggml-cann.cpp -->
```c
        case GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            break;
//>> ---- ggml/src/ggml-cann/ggml-cann.cpp:2001-2003 ----
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            break;
```

## 六、ggml_cann_create_tensor：两处约定差异 + 存储长度

转换函数要处理两处约定差异：步长单位（字节 -> 元素）与维度顺序（reverse）；它还有两个可选参数 `ne` / `nb`：不传就按张量自己的形状，传了就是**客户形状**（广播用）。

<!-- src: ggml/src/ggml-cann/acl_tensor.cpp -->
```c
acl_tensor_ptr ggml_cann_create_tensor(const ggml_tensor * tensor,
                                       int64_t *           ne,
                                       size_t *            nb,
                                       int64_t             dims,
                                       aclFormat           format,
                                       size_t              offset) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    if (ne == nullptr) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i]     = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_size(tensor);
        }
    } else {
        // With bcast
        for (int i = 0; i < dims; i++) {
            acl_ne[i]     = ne[i];
            acl_stride[i] = nb[i] / ggml_element_size(tensor);
        }
    }

    int64_t final_dims      = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    for (int i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }
    size_t elem_offset = offset / ggml_element_size(tensor);
    acl_storage_len += elem_offset;

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor * raw = aclCreateTensor(acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride, elem_offset,
                                      format, &acl_storage_len, 1, tensor->data);

    return acl_tensor_ptr(raw);
```

## 七、aclTensor 的所有权：acl_deleter

所有 ACL 对象都被 `std::unique_ptr` 包着，删除函数以**模板参数**的形式传进去 —— 所以 `acl_tensor_ptr` / `acl_int_array_ptr` / `acl_scalar_ptr` / `acl_tensor_list_ptr` 四个别名共用同一个 deleter 模板。这保证了 aclnn 调用里抛异常也不会漏掉 `aclDestroyTensor`。

<!-- src: ggml/src/ggml-cann/acl_tensor.h -->
```c
aclDataType ggml_cann_type_mapping(ggml_type type);

// Deleter for acl objects.
template <typename T, aclError (*DestroyFunc)(const T *)> struct acl_deleter {
    void operator()(T * ptr) const noexcept {
        if (ptr) {
            ACL_CHECK(DestroyFunc(ptr));
        }
    }
};

using acl_tensor_ptr      = std::unique_ptr<aclTensor, acl_deleter<aclTensor, aclDestroyTensor>>;
using acl_int_array_ptr   = std::unique_ptr<aclIntArray, acl_deleter<aclIntArray, aclDestroyIntArray>>;
using acl_scalar_ptr      = std::unique_ptr<aclScalar, acl_deleter<aclScalar, aclDestroyScalar>>;
using acl_tensor_list_ptr = std::unique_ptr<aclTensorList, acl_deleter<aclTensorList, aclDestroyTensorList>>;
```

## 八、内存池：一个基类，三种实现

基类只有两个纯虚函数；具体实现由 `new_pool_for_device` 按设备能力选：环境变量指定 `prio` 用优先队列池，设备支持 VMM 用虚拟内存池，否则用固定 256 槽的 buffer 池。无论哪种，最终都落到 `aclrtMalloc`。

<!-- src: ggml/src/ggml-cann/ggml-cann.cpp -->
```c
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(int device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    if (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_buf_prio(device));
    }

    if (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_buf(device));
```

## 九、stream：8 条，懒创建

`GGML_CANN_MAX_STREAMS` 是 8，数组初值全是 `nullptr`，第一次用到才 `aclrtCreateStream`。context 析构时逐条销毁。

<!-- src: ggml/src/ggml-cann/common.h -->
```c

#define MATRIX_ROW_PADDING    512
#define GGML_CANN_MAX_STREAMS 8
```

## 十、补充：context 的成员与析构

`streams[]` 数组本体在这里声明；析构函数负责销毁事件与全部 stream。

<!-- src: ggml/src/ggml-cann/common.h -->
```c
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrtStream streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(int device) : device(device), name("CANN" + std::to_string(device)) {
        ggml_cann_set_device(device);
        description = aclrtGetSocName();

#ifdef USE_ACL_GRAPH
        acl_graph_mode = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").value_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_graph_mode ? "GRAPH" : "EAGER",
                      acl_graph_mode ? "acl graph enabled" : "acl graph disabled");
#endif
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_set_device(device);
        if (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        for (int i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }
```

## 十一、完整映射清单（含第 5 幕未列出的部分）

下面这张表把 `ggml_cann_compute_forward` 的 case 与 `aclnn_ops.cpp` 里的 `GGML_CANN_CALL_ACLNN_OP` 调用逐条对上。行号来自本课引用的同一份源码。

| ggml op | 分派行（ggml-cann.cpp） | ACL 算子 | 调用行（aclnn_ops.cpp） |
|---|---|---|---|
| `GGML_OP_MUL_MAT` | 1918 | `aclnnMm` | 2236 |
| `GGML_OP_MUL_MAT`（3 维） | 1918 | `aclnnBatchMatMul` | 2239 |
| `GGML_OP_MUL_MAT`（4 维以上） | 1918 | `aclnnMatmul` | 2245 |
| `GGML_OP_MUL_MAT`（Q4_0/Q8_0） | 1918 | `aclnnWeightQuantBatchMatmulV2` | 2357 |
| `GGML_OP_MUL_MAT_ID` | 1921 | `aclnnIndexSelect` + `aclnnBatchMatMul` | 3644 / 3661 |
| `GGML_OP_FLASH_ATTN_EXT` | 2001 | `aclnnFusedInferAttentionScoreV2` | 4123 |
| `GGML_OP_ROPE` | 1959 | `aclnnRotaryPositionEmbedding` | 3191 |
| `GGML_OP_SOFT_MAX` | 1956 | `aclnnSoftmax` | 1932 |
| `GGML_OP_RMS_NORM` | 1915 | `aclnnRmsNorm` | 1347 |
| `GGML_OP_NORM` | 1885 | `aclnnLayerNorm` | 583 |
| `GGML_OP_GROUP_NORM` | 1888 | `aclnnGroupNorm` | 721 |
| `GGML_OP_L2_NORM` | 1891 | `aclnnNorm` + `aclnnClampMin` + `aclnnDiv` | 614 / 623 / 627 |
| `GGML_OP_ADD` | 1787 | `aclnnAdd` / `aclnnInplaceAdd` | 364 / 366 |
| `GGML_OP_SUB` | 1791 | `aclnnSub` | 374 |
| `GGML_OP_MUL` | 1797 | `aclnnMul` / `aclnnInplaceMul` | 382 / 384 |
| `GGML_OP_DIV` | 1800 | `aclnnDiv` | 390 |
| `GGML_OP_SCALE` | 1924 | `aclnnMuls` | 554 |
| `GGML_OP_CLAMP` | 1935 | `aclnnClamp` | 540 |
| `GGML_OP_GET_ROWS` | 1778 | `aclnnGatherV2` | 2003 |
| `GGML_OP_SET_ROWS` | 1781 | `aclnnInplaceIndexCopy` / `aclnnRepeatInterleaveIntWithDim` | 2126 / 2186 |
| `GGML_OP_REPEAT` | 1775 | `aclnnRepeat` | 324 |
| `GGML_OP_CPY` | 1938 | `aclnnInplaceCopy` / `aclnnCast` | 1118 / 344 |
| `GGML_OP_CONCAT` | 1897 | `aclnnCat` | 458 |
| `GGML_OP_IM2COL` | 1962 | `aclnnIm2col` | 1544 |
| `GGML_OP_CONV_TRANSPOSE_1D` | 1986 | `aclnnConvolution` | 3442 |
| `GGML_OP_SSM_CONV` | 2010 | `aclnnConvolution` | 4350 |
| `GGML_OP_POOL_2D` | 1965 | `aclnnAvgPool2d` / `aclnnMaxPool` | 1024 / 1087 |
| `GGML_OP_UPSCALE` | 1900 | `aclnnUpsampleNearest2d` | 930 |
| `GGML_OP_PAD` | 1903 | `aclnnConstantPadNd` | 955 |
| `GGML_OP_PAD_REFLECT_1D` | 1995 | `aclnnReflectionPad1d` | 3531 |
| `GGML_OP_SUM` / `GGML_OP_SUM_ROWS` | 1968 / 1971 | `aclnnReduceSum` | 801 |
| `GGML_OP_MEAN` | 1992 | `aclnnMean` | 3508 |
| `GGML_OP_CUMSUM` | 2013 | `aclnnCumsum` | 822 |
| `GGML_OP_ARGSORT` | 1974 | `aclnnArgsort` + `aclnnCast` | 567 / 569 |
| `GGML_OP_ARGMAX` | 1977 | `aclnnArgMax` | 3328 |
| `GGML_OP_COUNT_EQUAL` | 1998 | `aclnnEqTensor` + `aclnnReduceSum` | 3553 / 3559 |
| `GGML_OP_TRI` | 2016 | `aclnnTril` / `aclnnTriu` | 903 / 907 |
| `GGML_OP_DIAG_MASK_INF` | 1953 | `aclnnInplaceTriu` + `aclnnTril` + `aclnnInplaceAdd` | 1370-1372 |
| `GGML_OP_SOLVE_TRI` | 2025 | `aclnnTriangularSolve` | 842 |
| `GGML_OP_LEAKY_RELU` | 1912 | `aclnnLeakyRelu` | 441 |
| `GGML_OP_ELU`（UNARY） | 1845 | `aclnnElu` | 3495 |
| `GGML_OP_STEP`（UNARY） | 1851 | `aclnnGtScalar` | 3573 |
| `GGML_OP_SOFTPLUS`（UNARY） | 1854 | `aclnnSoftplus` | 3587 |
| `GGML_OP_CROSS_ENTROPY_LOSS` | 1894 | `aclnnSoftmaxCrossEntropyWithLogits` + `aclnnReduceSum` + `aclnnMuls` | 665 / 684 / 692 |
| `GGML_OP_FILL` | 2019 | `aclnnInplaceFillScalar` | 1250 |
| `GGML_OP_ARANGE` | 1906 | `aclnnArange` | 507 |
| `GGML_OP_QUANTIZE`（走 CPY） | 1938 | `aclnnCast` | 344 |
| `GGML_OP_RESHAPE` / `VIEW` / `PERMUTE` / `TRANSPOSE` | 1948-1951 | 无 —— 只改描述符，不产生计算 | — |

最后一行是这一课的一个反面注脚：**视图类算子不需要任何 ACL 算子**，因为它们在 ggml 层面就已经是"换个读法"（见 L1-01）。

## 十二、不支持的 op：supports_op 的兜底

`supports_op` 的最后两行是整个后端覆盖度的边界：switch 没列到的 op，一律 `false`。返回 false 的节点由 L4-02 的调度器切给别的后端。

<!-- src: ggml/src/ggml-cann/ggml-cann.cpp -->
```c
        default:
            return false;
```

---

## 说明

- 本课引用 7 个文件，全部计入覆盖率：`ggml/include/ggml-cann.h`、`ggml/src/ggml-cann/ggml-cann.cpp`、`ggml/src/ggml-cann/aclnn_ops.cpp`、`ggml/src/ggml-cann/aclnn_ops.h`、`ggml/src/ggml-cann/acl_tensor.cpp`、`ggml/src/ggml-cann/acl_tensor.h`、`ggml/src/ggml-cann/common.h`。
- 本课**没有**在 Ascend NPU 上实测运行（本机无昇腾设备）。所有断言都来自源码：函数名、行号、映射关系均可逐字核对；"异步执行""库决定算法"这类结论来自源码注释与调用形态，未在真机上验证。
- 第 11 节的大表里，`GGML_OP_CPY` 与 `GGML_OP_QUANTIZE` 两行都指向 `aclnnCast`（344 行）：`aclnn_cast` 是 `ggml_cann_cpy` 内部的类型转换路径，`GGML_OP_QUANTIZE` 本身没有独立 case。
