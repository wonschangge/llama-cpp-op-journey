<!-- llama-coverage
ggml/src/ggml-backend-reg.cpp
src/llama.cpp
ggml/src/ggml-backend-meta.cpp
-->

# L3-02 · ★ 后端注册表与设备发现 — 源文件

**一句话**：`ggml_backend_registry` 就是**两个 `std::vector`** —— 一个装后端，一个装设备。这一课讲它们**被谁、按什么顺序**填满：**有哪些后端是编译期 `#ifdef` 决定的**，运行期只做动态加载补充；而**设备在数组里的下标，直接决定 offload 时“第一个 GPU”是谁**。

阅读顺序：先看注册表结构（第 1 幕），再看静态注册的 16 个开关（第 2 幕）与注册动作（第 3 幕），然后是注册时机（第 4 幕）与设备枚举顺序（第 5 幕）—— 第 6 幕把“顺序”接到 offload 上，第 7 幕讲动态加载为什么只能追加，第 8、9 幕讲 Meta 设备与设备命名。

---

## 一、注册表：两个 vector

`ggml_backend_registry` 的全部状态就是两个 `std::vector`：一个装后端注册项，一个装设备。后端项多带一个动态库句柄 —— 这是第 7 幕要用的东西。

注意这里**没有**任何排序字段、优先级或哈希：`devices` 的顺序完全由“谁先被 push_back”决定，而这正是第 5、6 幕的主题。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

```

## 二、★ 静态注册：构造函数里的 16 个开关

注册表的构造函数是一串 `#ifdef GGML_USE_*`（120-173 行）。开关一打开，对应的后端就在**第一次触碰注册表时**被注册进来。

开关个数（命令口径）：

```text
$ grep -c "GGML_USE_" ggml/src/ggml-backend-reg.cpp
32
$ grep -o "GGML_USE_[A-Z0-9_]*" ggml/src/ggml-backend-reg.cpp | sort -u | wc -l
16
```

32 = 16 处 include 守卫（29-91 行）+ 16 处构造函数守卫（120-173 行）。完整清单与行号见第 2 幕的表格。

这些宏是 CMake 加的：`ggml/src/CMakeLists.txt:428-439` 的 `ggml_add_backend()` 里，只有 `GGML_BACKEND_DL=OFF` 时才执行 `target_compile_definitions(ggml PUBLIC GGML_USE_<BACKEND>)`。**打开 `GGML_BACKEND_DL`，这一段 `#ifdef` 就全空了** —— 后端改由运行期加载（第 7 幕、L3-03）。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
    // Add runtime disable check
    if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } else {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#endif
```

## 三、注册动作：追加 + 去重

`register_backend()` 只做两件事：把后端 `push_back` 到 `backends`，再按这个后端自报的设备数，把设备逐个交给 `register_device()`。

两个循环都**只有 `push_back`**，两侧都有“已经存在就直接 return”的去重（191-195、207-212）。去重按指针比较，不做任何排序或合并 —— 所以两个不同后端各自报出的设备会依次排在数组里。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
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

## 四、单例与内部 API：注册发生在什么时候

`get_reg()` 是一个函数内 `static` 单例：**注册不是启动时的一次全局初始化，而是第一次触碰注册表时发生的**。`ggml_backend_load_all()` 与所有 `ggml_backend_*` 查询最终都会先走到这里。

实测（本机 CPU-only 构建，`-DGGML_USE_CPU`）：

```text
reg_count before load_all = 1     # 静态注册已经完成
reg_count after  load_all = 1     # 没有可加载的动态后端
reg[0] = CPU (devs=1)
dev[0] = CPU | AMD Ryzen 7 5700X 8-Core Processor | type=0
```

`ggml_backend_register()` / `ggml_backend_device_register()` 是给“树外”调用者用的内部 API：例如 RPC 客户端在连上服务器后，把新后端注册进同一个注册表（`common/arg.cpp:1180`）。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}
```

## 五、★ 设备枚举顺序：下标 = 注册顺序

设备枚举的全部实现就是这两行转发。没有排序、没有过滤、没有重算 —— `devices[index]` 的下标是**唯一**的“设备号”。

按名字（345-353）与按类型（355-363）查找也都是**从头扫、返回第一个匹配**：当同类型有多个设备时，返回谁仍然由顺序决定。名字比较用 `striequals`（307-314），大小写不敏感。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
size_t ggml_backend_dev_count() {
    return get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    return get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (striequals(ggml_backend_dev_name(dev), name)) {
            return dev;
        }
    }
    return nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}
```

## 六、动态加载：准入校验与两张写死的顺序表

`load_backend()` 是动态加载的**准入闸门**：先 `dl_load_library()`，再要求可选的 `ggml_backend_score()` 非 0，然后必须找到 `ggml_backend_init`，最后校验 `reg->api_version` **精确等于** `GGML_BACKEND_API_VERSION`。任何一步不过就返回 `nullptr`，注册表不受影响。

（这些原语的实现见 L3-03；本课只关心“它什么时候、往哪儿追加”。）

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
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

## 七、load_all 的候选名字表

`ggml_backend_load_all_from_path()` 依次尝试 **15 个**候选名字（命令口径：`grep -c 'ggml_backend_load_best("' ggml/src/ggml-backend-reg.cpp` = 15）：

```text
blas, zendnn, cann, cuda, hip, metal, rpc, sycl,
vulkan, virtgpu, opencl, hexagon, musa, openvino, cpu
```

★ 这张表与构造函数里的 `#ifdef` 顺序**不是同一张表**。例如 `CANN`：静态顺序排在 CUDA 之后（156 vs 120 行），动态顺序排在 CUDA 之前（587 vs 588 行）。由于注册一律是 `push_back`，“同一个后端是编译进来的还是加载进来的”，会改变它在 `devices[]` 里的下标。

最后一段是环境变量 `GGML_BACKEND_PATH`：这是唯一由用户指定路径的加载入口（601-604）。与之相对，`GGML_DISABLE_VULKAN`（131 行）是唯一能在运行期关掉一个**已编译**后端的开关。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("hexagon", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
```

## 八、★ Meta backend：合成的设备

`ggml_backend_meta_device()` 把一组“简单设备”打包成一个设备：内存相加（99-110）、能力取交集（154-159）、类型固定为 `GGML_BACKEND_DEVICE_TYPE_META`（112-116）。

它**不是注册表成员**：全文件没有任何 `ggml_backend_register()` 调用，造出来的 `ggml_backend_device` 里 `reg = nullptr`（237-241）。所以 `ggml_backend_dev_count()` 数不到它，llama.cpp 的设备选择循环里遇到 META 型设备会直接 `GGML_ABORT`（`src/llama.cpp:269-270`）——它由调用方按需创建：张量并行时把枚举到的设备打包（`src/llama.cpp:193-220`）。

设备名是拼出来的：`"Meta(" + 各简单设备名 + ")"`（65-77），buffer type 同理（256-268）。普通后端的名字则完全由后端自己给：CUDA 用 `GGML_CUDA_NAME + 序号`（`ggml-cuda.cu:5798`），CPU 设备固定 `"CPU"`（`ggml-cpu.cpp:353-357`）。

<!-- src: ggml/src/ggml-backend-meta.cpp -->
```c
ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}
```

## 九、消费侧：注册顺序怎么变成 offload

注册表本身不 offload。真正的消费者是 `llama.cpp` 的默认设备选择：它按注册顺序遍历设备，把 GPU 型设备收进 `gpus`（254 行 `push_back`），拼成 `model->devices`（RPC 前置 275-277、GPU 追加 278-279）。

之后：`--main-gpu N` 取 `model->devices[N]`（297-299）；layer/row 模式下 `tensor_split` 的第 j 项对应 `model->devices[j]`（`llama-model.cpp:1488-1508`），归一化后由 `upper_bound(splits, ...)` 决定每层交给哪块卡（`llama-model.cpp:1529`），写进 `dev_layer[il]`（1540-1542），权重张量再按它选 buffer 类型完成落位（L2-03）。

<!-- src: src/llama.cpp -->
```cpp
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
```

---

## 说明

- 本课逐字引用 3 个文件：`ggml/src/ggml-backend-reg.cpp`、`ggml/src/ggml-backend-meta.cpp`（计划里本课的 2 个），外加 `src/llama.cpp` —— 只为一处：设备顺序的消费侧（第 6 幕 / 第九节）。该文件本就由 L2-01 覆盖，本课是重复声明（门禁口径：至少被一课声明）。
- 按行号指路、未引用原文因而**不计入本课覆盖率**的文件：`ggml/src/CMakeLists.txt:428-439`（谁定义 GGML_USE_*）、`common/arg.cpp:1116-1180`（设备名解析与 RPC 注册）、`ggml/src/ggml-cuda/ggml-cuda.cu:5798` 与 `ggml/include/ggml-cuda.h:11/14/17`（CUDA/ROCm/MUSA 设备名）、`ggml/src/ggml-cpu/ggml-cpu.cpp:353-357`（CPU 设备名）、`ggml/src/ggml-metal/ggml-metal.cpp:307`、`ggml/src/ggml-rpc/ggml-rpc.cpp:2273-2276`（RPC 设备数来自 context）、`ggml/src/ggml-cann/ggml-cann.cpp:2807-2810`（CANN 设备是 GPU 型）、`src/llama-model.cpp:1488-1542`（splits 与 dev_layer，L2-05 已逐字引用）。
- 第 3、5 幕的“本机实测”来自作者自建的临时 harness（直接调用 `ggml_backend_register()` 注册两个假后端，再读 `ggml_backend_dev_count()` / `ggml_backend_dev_get(i)`），跑在本仓库 `build/` 下已有的 CPU-only 构建上；该 harness 不属于本课件仓库，也不计入覆盖率。
