<!-- llama-coverage
ggml/src/ggml-backend-impl.h
ggml/src/ggml-blas/ggml-blas.cpp
ggml/include/ggml-blas.h
ggml/src/ggml-backend-reg.cpp
ggml/src/ggml-backend-dl.cpp
-->

# L8-03 · 端到端：新后端要做什么 — 源文件

**一句话**：写一个新后端**不是从零开始**。你只需要按 `ggml-backend-impl.h` 的三张虚表填少数几个必需函数、写一个说真话的 `supports_op`，再把 `ggml_backend_init` 导出，剩下的（内存分配、算子回退、图切分）全部由现成机制接住。

证据就是 BLAS 后端：`ggml/src/ggml-blas/ggml-blas.cpp` 一个文件、530 行，是仓库里**最小的真实后端**。它的规模就是"最少要写多少"的答案 —— 而对照的 ggml-cuda 是 277 个文件、45718 行。这一课把前七层压成一份可勾选的清单。

---

## 一、一个后端的第一屏：四个 include

写一个新后端，第一步不是写函数，而是**确认自己站在哪份契约上**。BLAS 的前 20 行把依赖交代得很清楚：`ggml-backend-impl.h`（虚表定义，L3-01 的主角）、`ggml-impl.h`（`GGML_ASSERT` / `GGML_LOG_*`）、自己的公共头 `ggml-blas.h`，以及一个编译期选定的 BLAS 库。

注意最后这一组 `#if defined(GGML_BLAS_USE_*)`：**后端可以有自己的编译期配置**，ggml 侧完全不需要知道。

<!-- src: ggml/src/ggml-blas/ggml-blas.cpp -->
```cpp
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-blas.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <cstring>

#if defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif
```

## 二、★ 530 行里只有三张虚表

BLAS 后端只用 1 个文件 530 行就实现了 `MUL_MAT` 与 `OUT_PROD`。它的"契约面"只有三处：`ggml_backend_i`（第 262-279 行）、`ggml_backend_device_i`（第 456-472 行）、`ggml_backend_reg_i`（第 513-518 行）。

`ggml_backend_i` 的 16 个槽位里只有 **3 个**非 NULL：`get_name`、`free`、`graph_compute`。其余 13 个（异步读写、`synchronize`、四个 `graph_plan_*`、两个 event、`graph_optimize`）全部留空 —— 源码注释把它们标成了 `(optional)`，L3-01 已经数过。

<!-- src: ggml/src/ggml-blas/ggml-blas.cpp -->
```cpp
static struct ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_blas_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}
```

## 三、设备层 10 个指针，内存层 0 个

`ggml_backend_device_i` 有 15 个槽位，BLAS 填了 10 个：自我介绍 5（`get_name` / `get_description` / `get_memory` / `get_type` / `get_props`）、`init_backend`、`get_buffer_type`、`buffer_from_host_ptr`（可选）、`supports_op`、`supports_buft`。

关键在 `get_buffer_type` 这一行：它直接返回 `ggml_backend_cpu_buffer_type()`。于是 `buffer_type_i`（6 个指针）与 `buffer_i`（11 个指针）**一个都不用写** —— BLAS 全文出现这两个结构体名字的次数是 **0**。这正是 L4-03 讲的"buffer 家族"在实战里的用法。

<!-- src: ggml/src/ggml-blas/ggml-blas.cpp -->
```cpp
static const struct ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
```

## 四、★ supports_op：切分的唯一依据

L4-02 讲过调度器的归属优先级：对每个节点依次问各后端的 `supports_op` 与 `offload_op`，第一个说"能"的后端拿走这个节点。所以 `supports_op` 是**新后端的第一等公民**：它决定了这个后端会拿到哪些算子，也决定了其余算子如何"回退"给别人。

BLAS 的策略很清楚：视图类算子（`NONE` / `RESHAPE` / `VIEW` / `PERMUTE` / `TRANSPOSE`）无条件 true；`MUL_MAT` 要求两个输入连续、`src1` 是 F32、三个维度都不小于 `min_batch = 32`、`src0` 是 F32 或可反量化，并且**没有** `GGML_HINT_SRC0_IS_HADAMARD` 提示；`OUT_PROD` 另有一组判据；其余一律 `default: return false`。

最后那行 `return false` 就是"回退开关"：它不需要写任何搬运或转发代码。

<!-- src: ggml/src/ggml-blas/ggml-blas.cpp -->
```cpp
static bool ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }

        case GGML_OP_OUT_PROD:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

        default:
            return false;

    }

    GGML_UNUSED(dev);
}
```

## 五、graph_compute：default 是 GGML_ABORT

`graph_compute` 是 `ggml_backend_i` 里唯一的执行入口。BLAS 的实现是一个 switch：`MUL_MAT` -> `ggml_backend_blas_mul_mat`，`OUT_PROD` -> `ggml_backend_blas_out_prod`，五个视图算子显式 `break`，`default` 分支是 `GGML_ABORT`。

这条 `GGML_ABORT` 说明了一件重要的事：**后端自己不做回退**。不支持的算子根本不会走到这里 —— 切分阶段（L4-02）已经按 `supports_op` 把它分给了别的后端。反过来说：`supports_op` 说 true 而 `graph_compute` 没实现，第一次推理就会 abort。

<!-- src: ggml/src/ggml-blas/ggml-blas.cpp -->
```cpp
static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, node);
                break;

            case GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}
```

## 六、注册：一个工厂函数 + 一个导出宏

后端必须能回答"你是谁、你有哪些设备"。BLAS 的答案是 `ggml_backend_blas_reg()`：返回一个静态的 `ggml_backend_reg`，里面只有 `api_version`、`iface`、`context` 三个字段。

最后一行的 `GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)` 是 L3-03 的主角：开启动态构建时，它展开出一个 `extern "C"` 的 `ggml_backend_init()`，返回值就是同一个 `reg()`。也就是说**静态构建与动态构建共用同一个注册结构**，区别只在"谁来调用它"。

<!-- src: ggml/src/ggml-blas/ggml-blas.cpp -->
```cpp
ggml_backend_reg_t ggml_backend_blas_reg(void) {
    static struct ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
```

## 七、注册表怎么接住它（静态路径）

`ggml/src/ggml-backend-reg.cpp` 是注册表的实现。静态构建时，它在 `GGML_USE_BLAS` 打开的情况下 include 后端头，并在注册表构造函数里调用 `reg()`；`register_backend` 负责去重（同一个 reg 只登记一次）并把该后端的**所有设备**登记进 `devices` 表。

下一节的动态路径复用同一个 `register_backend`，只是 reg 来自 `dlopen` 出来的库。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```cpp
    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        if (!reg) {
            return;
        }

        for (auto & entry : backends) {
            if (entry.reg == reg) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }
```

## 八、注册表怎么接住它（动态路径与版本门槛）

动态加载路径在 `load_backend` 里，只有四步：`dl_load_library` 打开库、`dl_get_sym` 找可选的 `ggml_backend_score`、找必需的 `ggml_backend_init`、然后校验 `reg->api_version != GGML_BACKEND_API_VERSION`。

L3-03 已经拆过这段；这里补一句与新后端作者有关的结论：**版本号写错的后端不会崩，只会被静默丢弃并打一条日志**（第 252-253 行）。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```cpp
    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            return nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } else {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            return nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        return reg;
    }
```

## 九、动态加载只做两件事：dlopen 与 dlsym

`ggml/src/ggml-backend-dl.cpp` 是加载器的全部平台差异所在：POSIX 侧就是 `dlopen(path, RTLD_NOW | RTLD_LOCAL)` 与 `dlsym(handle, name)`，Windows 侧换成 `LoadLibraryW` 与 `GetProcAddress`。

这就是"动态加载约定"的全部内容：**后端不需要写任何加载代码**，它只需要按约定把 `ggml_backend_init`（以及可选的 `ggml_backend_score`）导出成 C 符号。

<!-- src: ggml/src/ggml-backend-dl.cpp -->
```cpp
dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}
```

## 十、公共头：一个后端对外的四个声明

后端还要有一份自己的公共头，让 ggml 侧能声明式地引用它。`ggml/include/ggml-blas.h` 只有四个函数：创建 backend、判断"是不是我"、设置线程数、取注册结构。

注意 `ggml_backend_blas_set_n_threads` 并不在虚表里 —— 它是后端私有的扩展函数，通过 `ggml_backend_reg_i::get_proc_address` 按名字暴露（第 503-511 行），测试工具 `test-backend-ops` 正是这样拿到它并设置线程数的（第 12145-12147 行）。

<!-- src: ggml/include/ggml-blas.h -->
```cpp
// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_blas_init(void);

GGML_BACKEND_API bool ggml_backend_is_blas(ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_BACKEND_API void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_blas_reg(void);
```

## 十一、清单（a）必须实现的接口函数 ·（b）可以留空的

按 L3-01 的分组数一遍（判据：源码注释里没有 `(optional)`）：

| 虚表 | (a) 必须实现 | (b) 可留空 |
|---|---|---|
| `ggml_backend_device_i`（impl.h 176-218） | 9：`get_name` `get_description` `get_memory` `get_type` `get_props` `init_backend` `get_buffer_type` `supports_op` `supports_buft` | 6：`get_host_buffer_type` `buffer_from_host_ptr` `offload_op` `event_new` `event_free` `event_synchronize` |
| `ggml_backend_i`（impl.h 121-156） | 3：`get_name` `free` `graph_compute` | 13：异步 5、`synchronize`、`graph_plan_*` 4、event 2、`graph_optimize` |
| `ggml_backend_buffer_type_i`（impl.h 17-29） | 3：`get_name` `alloc_buffer` `get_alignment` | 3：`get_max_size` `get_alloc_size` `is_host` |
| `ggml_backend_buffer_i`（impl.h 46-67） | 5：`get_base` `memset_tensor` `set_tensor` `get_tensor` `clear` | 6：`free_buffer` `init_tensor` `set_tensor_2d` `get_tensor_2d` `cpy_tensor` `reset` |
| `ggml_backend_reg_i`（impl.h 230-240） | 3：`get_name` `get_device_count` `get_device` | 1：`get_proc_address` |
| 合计 | **23 个必需项**（三张主虚表 15 + 数据面 5 + 注册面 3） | 29 个 |

**最小实现可以更少**：把 `get_buffer_type` 接到 `ggml_backend_cpu_buffer_type()`，内存层那 8 个必需项（`buffer_type_i` 3 + `buffer_i` 5）就不用写了 —— BLAS 正是如此。

## 十二、清单（c）必须注册 / 导出的符号 ·（d）必须通过的测试

**(c) 符号。** 静态构建：`ggml_backend_reg_t <name>_reg(void)`，由注册表构造函数直接调用（`reg.cpp` 第 160 行是 BLAS 的例子）；`reg->api_version` 必须等于 `GGML_BACKEND_API_VERSION`（`impl.h` 第 11 行），否则加载时被丢弃（`reg.cpp` 第 246 行）。动态构建：`GGML_BACKEND_DL_IMPL` 生成的 `extern "C" ggml_backend_init`，加载器用 `dl_get_sym` 按名查找（`reg.cpp` 第 237 行）；可选再加 `GGML_BACKEND_DL_SCORE_IMPL` 生成的 `ggml_backend_score`（`reg.cpp` 第 229、534 行）。

**(d) 测试。** 全部用仓库现成工具，四条：

```text
1 设备出现    llama-cli --list-devices            -> 打印 name: description（只列非 CPU 设备）
2 能力探测    test-backend-ops support -b <NAME>  -> 逐算子打印 supports_op 的 yes / no
3 正确性      test-backend-ops test -b <NAME>     -> 与 CPU 后端逐元素比误差（硬门禁）
4 端到端      llama-cli -m model.gguf -ngl N      -> 真实模型跑通
```

第 3 步的判据在 `tests/test-backend-ops.cpp`：先把 `supports_op` 为 false 的用例标成 "not supported"，再对支持的用例调 `ggml_backend_compare_graph_backend`，逐元素算误差并与该类型的 `max_err` 比较（第 1530-1549 行）。测试工具还会通过 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")` 来找后端的线程设置函数（第 12145 行）—— 这正是 BLAS 实现 `get_proc_address` 的原因。

---

## 说明

- 本课逐字引用 **5 个文件**，全部计入覆盖率：`ggml/src/ggml-backend-impl.h`（计划指定）、`ggml/src/ggml-blas/ggml-blas.cpp`、`ggml/include/ggml-blas.h`、`ggml/src/ggml-backend-reg.cpp`、`ggml/src/ggml-backend-dl.cpp`（后四个为本课追加，用来把"清单"落到真实代码上：BLAS 是最小的真实后端，reg/dl 是注册与加载的实现）。
- 文中提到但**不计入本课覆盖率**的文件：`ggml/src/CMakeLists.txt`（第 589 行 `ggml_add_backend(BLAS)`）、`ggml/CMakeLists.txt`（第 86 行 `GGML_BACKEND_DL`、第 194 行 `GGML_BLAS`）、`ggml/src/ggml-blas/CMakeLists.txt`、`ggml/src/ggml-backend-dl.h`、`tests/test-backend-ops.cpp`、`common/arg.cpp` —— 它们是构建接线与测试工具，不属于本课覆盖域。
- 本课引用的数字来自实测命令：`wc -l ggml/src/ggml-blas/ggml-blas.cpp` = 530；`wc -l ggml/src/ggml-zendnn/ggml-zendnn.cpp` = 838；`find ggml/src/ggml-cuda -type f \( -name "*.cu" -o -name "*.cuh" \)` = 277 个文件、合计 45718 行；`grep -c "ggml_backend_buffer_i\|ggml_backend_buffer_type_i" ggml/src/ggml-blas/ggml-blas.cpp` = 0；三张虚表的非 NULL 指针按 `= */` 行统计为 3（backend_i）+ 10（device_i）+ 4（reg_i）= 17。
- **边界说明**：本机构建为 `GGML_BLAS=OFF`、`GGML_BACKEND_DL=OFF`，且 `build/bin` 下没有 `test-backend-ops`，因此第十二节的四条验证步骤**来自源码**（`tests/test-backend-ops.cpp` 的 `usage()` 第 12003-12021 行、`main()` 第 12136-12149 行；`common/arg.cpp` 第 1138-1160 行），不是本课的运行输出。本课实测过的命令只有 `llama-cli --list-devices`，输出是 `Available devices:` 与 `  (none)`（因为没有启用任何非 CPU 后端）。
