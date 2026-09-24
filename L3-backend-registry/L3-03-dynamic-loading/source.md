<!-- llama-coverage
ggml/src/ggml-backend-dl.h
ggml/src/ggml-backend-dl.cpp
ggml/src/ggml-backend-reg.cpp
ggml/src/ggml-backend-impl.h
ggml/src/ggml-cpu/ggml-cpu.cpp
-->

# L3-03 · 动态加载后端 ggml-backend-dl — 源文件

**一句话**：动态加载只做两件事 —— **打开动态库**、**按名字取符号**；而"必须导出哪些符号、版本要多新"这套规矩，写在注册表那一侧。

上游这两个文件一共 93 行（`wc -l` 口径：`ggml-backend-dl.h` 45 行、`ggml-backend-dl.cpp` 48 行），本课**全文引用**它们；再把注册表里**用到这两个原语的那几段**（`ggml-backend-reg.cpp`）一并逐字引出来 —— 否则"后端动态库必须导出哪些符号"这个问题，在本课的两个文件里根本找不到答案。

**核心洞察**：动态加载把"支持哪些后端"从**编译期**推迟到**运行期**；代价是接口必须冻结、必须版本校验 —— 这正是 L3-01 那三张虚表不能随便改的原因。

---

## 一、加载原语：dl_load_library / dl_get_sym / dl_error

POSIX 分支的全部实现。注意两件事：

1. 三个函数的签名里**没有任何后端相关的类型** —— 返回值是 `dl_handle *` 与 `void *`，所以调用方必须自己转型（注册表那边转成 `ggml_backend_init_t` / `ggml_backend_score_t`）。
2. `RTLD_NOW | RTLD_LOCAL` 这两个 flag 是这段代码里唯一的"策略"：前者要求**打开时就解析全部符号**（缺符号等于打开失败），后者要求**符号不进全局命名空间**（多个后端导出同名符号也不会互相顶掉）。

`dl_error()` 把 `dlerror()` 的 NULL 折成空串 —— 于是调用方（`GGML_LOG_ERROR` 那一行）可以无条件地打印它。

<!-- src: ggml/src/ggml-backend-dl.cpp -->
```c
#else

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

#endif
```

## 二、同一抽象的另一端：Windows

Windows 分支用 `LoadLibraryW` / `GetProcAddress`，并在两处用 `SetErrorMode` 压掉系统错误弹窗 （注释写明理由：suppress error dialogs for missing DLLs）。

对照上一节可以看到一处**能力不对等**：POSIX 的 `dl_error()` 会返回 `dlerror()` 的文本，而 Windows 分支的 `dl_error()` 直接 `return "";`。所以 Windows 上加载失败的日志里，错误描述永远是空的 —— 这是实现现状，不是笔误。

关闭动作不在这里：它由头文件里的删除器负责（下一节）。

<!-- src: ggml/src/ggml-backend-dl.cpp -->
```c
#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

const char * dl_error() {
    return "";
}
```

## 三、句柄：dl_handle / dl_handle_deleter / dl_handle_ptr

两端要暴露同一个类型 `dl_handle *`：Windows 的 `HMODULE` 本身就是指针，所以用 `std::remove_pointer_t<HMODULE>` 取出"被指向类型"；POSIX 侧直接把 `dl_handle` 定义成 `void`。两种写法都让 `dl_handle *` 合法。

`dl_handle_deleter` 把关闭动作（`FreeLibrary` / `dlclose`）绑到析构函数上，`dl_handle_ptr` 则是 `std::unique_ptr<dl_handle, dl_handle_deleter>`：**只可 move，不可拷贝**。这条约束的意义在下一节会看到 —— 句柄被 move 进注册表，库就在整个进程里保持加载。

<!-- src: ggml/src/ggml-backend-dl.h -->
```c
#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;
```

## 四、★ 动态库必须导出的符号

这是本课验收点的**直接证据**。注册表加载一个动态库的完整流程，以及它按名字取的两个符号：

| 符号（dlsym 的字符串） | 必需性 | 行为 |
|---|---|---|
| `"ggml_backend_init"` | **必需** | 找不到就打印 `failed to find ggml_backend_init` 并放弃这个库 |
| `"ggml_backend_score"` | 可选 | 取不到也继续；但返回 0 表示"本机不支持"，同样放弃 |

三条 `return nullptr`（第 226 / 234 / 242 行）说明**任何一种失败都不会注册任何东西**：加载器要么交出一个可用的 reg，要么什么都不留。

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
```

## 五、★ 版本校验：api_version 必须精确相等

`ggml_backend_init` 返回的 reg 要过两道检查，写在同一句 `if` 里：`!reg || reg->api_version != GGML_BACKEND_API_VERSION`。注意是 `!=`（精确相等），不是"大于等于"或兼容区间 —— 虚表布局变了，旧库就不可用，宁可拒绝加载，也不要等到调用虚表时崩在别处。

两种失败各有一条日志；版本不符的那条会把 backend 版本与当前版本都打出来。

最后一行 `register_backend(reg, std::move(handle))`：**句柄被 move 进注册表**，于是库不会被卸载（句柄怎么存，见第八节）。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
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

## 六、符号从哪来：GGML_BACKEND_DL_IMPL

后端不手写符号名：它写一行宏，宏负责生成 `extern "C"` 的导出函数。`extern "C"` 是必需的 —— `dlsym` 按**字面名字**查符号，而 C++ 会对函数名做名字改编（mangling）。

整个宏块被 `#ifdef GGML_BACKEND_DL` 包住，最后四行是没有定义该宏时的分支：两个宏展开成**空**。也就是说，"库里有没有这两个符号"由**构建开关**决定，不是库里天然就有。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            extern "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(void) {                 \
                return reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            extern "C" {                                   \
            GGML_BACKEND_API int ggml_backend_score(void); \
            }                                              \
            int ggml_backend_score(void) {                 \
                return score_fn();                         \
            }
#    else
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);  \
            ggml_backend_reg_t                  ggml_backend_init(void) { \
                return reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API int ggml_backend_score(void);  \
            int                  ggml_backend_score(void) { \
                return score_fn();                          \
            }
#    endif
#else
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
```

## 七、后端侧只写一行

真实后端怎么用这个宏：CPU 后端在自己的注册函数下面写一行 `GGML_BACKEND_DL_IMPL(...)`，把 `ggml_backend_cpu_reg` 交给宏。展开后，这个库就导出了 `ggml_backend_init`。

（`ggml-cpu.cpp` 是 CPU 后端课的主文件，本课只为"宏的真实调用点"引用这一行。）

<!-- src: ggml/src/ggml-cpu/ggml-cpu.cpp -->
```c
GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
```

## 八、句柄交到注册表手上

前半是注册表的 entry 类型：**一个 reg + 一个句柄**。后半是 `register_backend()`：它把 entry 推进 `backends` 向量（句柄随之被 move 进去），并为这个 reg 登记全部设备。

把这两段与本课第三节的 `dl_handle_ptr` 连起来看：**"后端还加载着吗"完全由注册表持不持有这个 unique_ptr 决定**。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};
//>> ---- ggml/src/ggml-backend-reg.cpp:186-205 ----
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

## 九、择优路径上 score 是必需的

`ggml_backend_load_best()` 在目录里枚举候选库时，走的是另一条路径：取 `"ggml_backend_score"`、调用它、保留分数最高的那个文件；**取不到 score 的库会被跳过**（日志：`failed to find ggml_backend_score`）。

所以"可选符号"这个说法要加限定：对 `load_backend(path)`（直接点名加载）它是可选的；对"按名字自动择优"它是必需的。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } else {
                            if (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
```

## 十、加载入口与候选名字

启动时的入口：`ggml_backend_load_all()` 转发到这里。它按写死的名字列表逐个尝试（blas / zendnn / cann / cuda / hip / metal / rpc / sycl / vulkan / virtgpu / opencl / hexagon / musa / openvino / cpu），最后再看环境变量 `GGML_BACKEND_PATH`，用 `ggml_backend_load()` 加载一个"树外"后端。

`silent` 随构建变化：Release（`NDEBUG`）下为 true —— 候选库不存在属于常态，不必刷屏。

<!-- src: ggml/src/ggml-backend-reg.cpp -->
```c
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

## 十一、对照：编译期注册的静态后端

同一个注册表的另一条入口。构造函数里是一串 `#ifdef GGML_USE_*`：**编译期**就决定了哪些后端存在，运行期只是把它们登记进 `backends`。

把这一节与第十节并排看，就是本课的核心对比：静态注册的"支持哪些后端"是编译期常量；动态加载把它变成了运行期才回答的问题 —— 代价是符号名与版本号必须成为稳定契约。

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

## 十二、版本号的当前值

契约的版本戳就这一行。它与第五节的 `!=` 校验配对：**只要这个数字变了，所有已编译的动态后端都会被拒绝加载**，必须与主库一起重编。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
    #define GGML_BACKEND_API_VERSION 2
```

---

## 说明

- 本课为回答"后端动态库必须导出哪些符号"，额外引用了 `ggml/src/ggml-backend-reg.cpp`、`ggml/src/ggml-backend-impl.h` 与 `ggml/src/ggml-cpu/ggml-cpu.cpp` 的相应片段。前两者分别是 L3-02 / L3-01 的主文件，第三者属于 CPU 后端课；三者在覆盖度门禁里都算作被本课引用（`llama-coverage` 块里已一并声明）。
- **实测说明**（避免只从注释推断）：本机 `build/` 的 `CMakeCache.txt` 里写着 `GGML_BACKEND_DL:BOOL=OFF`，因此 `nm -D build/bin/libggml-cpu.so` 的输出里既没有 `ggml_backend_init` 也没有 `ggml_backend_score` —— `GGML_BACKEND_DL_IMPL` 展开为空（impl.h:285-286）。符号是否存在与构建选项绑定，不是"动态库里天然就有"。
