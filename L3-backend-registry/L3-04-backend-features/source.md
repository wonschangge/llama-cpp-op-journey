<!-- llama-coverage
ggml/src/ggml-feats.h
ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp
ggml/src/ggml-backend-reg.cpp
ggml/src/ggml-cpu/ggml-cpu.cpp
ggml/src/ggml-cpu/CMakeLists.txt
ggml/src/ggml-backend-impl.h
ggml/include/ggml-backend.h
src/llama.cpp
ggml/src/ggml-cpu/ggml-cpu.c
ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
-->

# L3-04 · ggml-feats：后端能力探测 — 源文件

**一句话**：`ggml-feats.h` 是"运行期"那一半 —— 它回答"**这台机器的 CPU 支持什么**"，而 CMake 里的 `GGML_USE_*` 是"编译期"那一半，回答"**这份二进制是按什么指令集编出来的**"。后端能不能被加载，取决于这两个答案的**乘积**。

这个头文件只有 166 行，而且**整个文件被一道 `#if defined(__aarch64__) || defined(_M_ARM64)` 包住** —— 在 x86 上它是一个空文件。它小，是因为它只做一件事：把平台 API 问出来的能力位，装进一个 8 字段的小结构体。

---

## 一、两道门：架构门与平台门

整个文件被一道 `#if defined(__aarch64__) || defined(_M_ARM64)` 包住（第 3 行），最后一行（第 166 行）是它的 `#endif`。这意味着：**在 x86 / RISC-V 上，这个文件编译出来是空的**，不会有任何函数被定义。门里面又按平台分成三支：`__linux__` / `__APPLE__` / `_WIN32`。

<!-- src: ggml/src/ggml-feats.h -->
```c
#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)

#if defined(__linux__)
#include <sys/auxv.h>
#include <sys/prctl.h>

#if !defined(HWCAP2_SVE2)
#define HWCAP2_SVE2 (1ULL << 1)
#endif
```

## 二、Linux 侧的常量补丁：为什么要自己写 9 个 `#define`

这些常量本该由 `<sys/auxv.h>` / `<sys/prctl.h>` 提供，但每个外面都套着 `#if !defined(...)`：**系统头里已经有名字就跳过，没有才补一份**。这个写法本身说明了它要解决的问题 —— 编译这套代码的机器，其系统头不一定认识这些位（`HWCAP2_SME2 = 1ULL << 37` 是相当新的位）。

<!-- src: ggml/src/ggml-feats.h -->
```c
#if !defined(HWCAP_FPHP)
#define HWCAP_FPHP (1 << 9)
#endif

#if !defined(HWCAP_ASIMDHP)
#define HWCAP_ASIMDHP (1 << 10)
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1ULL << 13)
#endif

#if !defined(HWCAP_ASIMDDP)
#define HWCAP_ASIMDDP (1 << 20)
#endif

#if !defined(HWCAP_SVE)
#define HWCAP_SVE (1 << 22)
#endif

#if !defined(HWCAP2_SME)
#define HWCAP2_SME (1ULL << 23)
#endif

#if !defined(HWCAP2_SME2)
#define HWCAP2_SME2 (1ULL << 37)
#endif

#if !defined(PR_SVE_GET_VL)
#define PR_SVE_GET_VL 51
#endif

#if !defined(PR_SVE_VL_LEN_MASK)
#define PR_SVE_VL_LEN_MASK 0xffff
#endif
```

## 三、macOS / Windows 侧的常量补丁

另外两个平台分支只 include 平台头，然后补 `PF_ARM_*` 常量（Windows 用的处理器特性编号）。注意 Apple 分支不需要补任何常量 —— 它按**字符串名字**问 sysctl。

<!-- src: ggml/src/ggml-feats.h -->
```c
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>

#if !defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif

#if !defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE_INSTRUCTIONS_AVAILABLE 46
#endif

#if !defined(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE 47
#endif

#if !defined(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE 66
#endif

#if !defined(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE 67
#endif

#if !defined(PF_ARM_SME_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME_INSTRUCTIONS_AVAILABLE 70
#endif

#if !defined(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME2_INSTRUCTIONS_AVAILABLE 71
#endif

#endif
```

## 四、探测结果的形状

7 个布尔 + 1 个 `int`。这是"运行期探测"的全部产出，**按值返回、`static inline`** —— 谁 include 谁就编译出自己的一份，不需要 ggml 导出任何符号。这正是 L3-03 里被 `dlopen` 进来的后端 `.so` 能自己给自己打分的前提。

★ 注意 `has_sme2`：它被探测了，但本课引用的两个使用点（`cpu-feats.cpp` 的 score 函数、`kleidiai.cpp` 的 `ctx.features`）都没有读它。

<!-- src: ggml/src/ggml-feats.h -->
```c
typedef struct ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_sme;
    bool has_sme2;
    int sve_cnt;
} ggml_feats_arch64_runtime_t;
```

## 五、★ Linux 路径：getauxval + prctl

两次 `getauxval` 把内核填好的位图取回，然后逐个位映射成字段；唯一需要"再问一次"的是 SVE 向量长度（位图里没有这个数）。

<!-- src: ggml/src/ggml-feats.h -->
```c
#if defined(__linux__)
    const unsigned long hwcap  = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    if (runtime_feat.has_sve) {
        const int vl = prctl(PR_SVE_GET_VL);
        if (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_MASK;
        }
    }
```

## 六、macOS / Windows 路径

同一个结构体，三种问法：Linux 读位图、macOS 按名字问 sysctl、Windows 按编号问 `IsProcessorFeaturePresent`。两处 `sve_cnt = 0` 都配了源码注释说明理由（**注释声称**：Apple 不支持用户态 non-streaming SVE；Windows 这边不暴露运行期向量长度）。

<!-- src: ggml/src/ggml-feats.h -->
```c
#elif defined(__APPLE__)
    int oldp = 0;
    size_t size = sizeof(oldp);

    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#elif defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#endif
```

## 七、编译期开关是从哪来的

`GGML_USE_*` 不是源文件里定义的，而是 CMake 在该变体的编译定义里追加的。下面的片段来自 `GGML_CPU_ALL_VARIANTS` 分支：每开一个 ARM 特性，就同时改 `-march` 目标、加一个 `ARCH_TAGS` 标签、追加一条 `ARCH_DEFINITIONS`。**这三件事必须一起发生**：指令集、库名后缀、以及源文件里那个 `#ifdef`。

<!-- src: ggml/src/ggml-cpu/CMakeLists.txt -->
```cmake
                    if (GGML_INTERNAL_DOTPROD)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+dotprod")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_DOTPROD)
                    endif()
                    if (GGML_INTERNAL_FP16_VECTOR_ARITHMETIC)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+fp16")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_FP16_VECTOR_ARITHMETIC)
                    endif()
                    if (GGML_INTERNAL_SVE)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE)
                    endif()
```

## 八、★ 路径 A 的第一步：编译期与运行期相乘

`ggml_backend_cpu_aarch64_score()` 是本课的枢纽：`#ifdef GGML_USE_X` 表示"这份二进制是按 X 编的，它**要求**运行环境有 X"；`if (!af.has_x) { return 0; }` 表示"要求落空就出局"。满足则按位加分（基准分 1）。

★ `0` 不是"分数低"，而是**约定的"本机不支持"**（见下一节 `ggml_backend_score_t` 的注释）。

<!-- src: ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp -->
```c
#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#if defined(__aarch64__) || defined(_M_ARM64)

static int ggml_backend_cpu_aarch64_score() {
    int score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    if (!af.has_dotprod) { return 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    if (!af.has_fp16) { return 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    if (!af.has_sve) { return 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    if (!af.has_i8mm) { return 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    if (!af.has_sve2) { return 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    if (!af.has_sme) { return 0; }
    score += 1<<6;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)
```

## 九、路径 A 的第二步：score 怎么变成一个可被查询的符号

`GGML_BACKEND_DL_SCORE_IMPL(score_fn)` 展开出 `extern "C"` 的 `ggml_backend_score()`，它只是把上面那个静态函数转发出去。注意 `#ifdef GGML_BACKEND_DL`：**只有在动态加载构建下这个符号才存在**；静态构建时宏展开为空。

<!-- src: ggml/src/ggml-backend-impl.h -->
```c
    // Initialize the backend
    typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    typedef int                (*ggml_backend_score_t)(void);
```

## 十、路径 A 的第三步：加载器按 score 择库

`load_backend()` 里：取不到 `ggml_backend_score` 就照常加载（静态后端没有这个符号）；取得到且返回 0，就**放弃这个 .so**。

另一处是目录扫描：对每个候选 `.so` 求分，`if (s > best_score)` 留下最高的一个；`best_score` 仍为 0 时，才回退去加载基础版后端。

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
```

## 十一、★ 路径 B：能力被"报"出去

第二条路径不参与"选库"，只回答"你支持什么"。契约在公共头文件里：一个 `name`/`value` 的两字符串结构体，外加一个返回**以 `nullptr` 结尾的数组**的函数指针类型。注意取这个函数的方式是 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")` —— **按符号名字符串查询**，而不是直接链接。

<!-- src: ggml/include/ggml-backend.h -->
```c
    GGML_API void *             ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);
//>> ---- ggml/include/ggml-backend.h:219-225 ----
    typedef void                         (*ggml_backend_set_abort_callback_t)(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data);
    // Get a list of feature flags supported by the backend (returns a NULL-terminated array)
    struct ggml_backend_feature {
        const char * name;
        const char * value;
    };
    typedef struct ggml_backend_feature * (*ggml_backend_get_features_t)(ggml_backend_reg_t reg);
```

## 十二、路径 B 的实现：表里混着两种来源

`ggml_backend_cpu_get_features()` 里，绝大多数项来自 `ggml_cpu_has_*()`（编译期宏）；表尾几项则是直接 `#ifdef GGML_USE_*` 追加的；`SVE_CNT` 是唯一带运行期数值的项。整张表用 `static` + lambda 只构建一次，最后以 `{ nullptr, nullptr }` 结尾。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.cpp -->
```cpp
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
```

## 十三、路径 B 的表尾：编译期追加项

这几个 `#ifdef` 项说明：**后端特性表不只是"硬件能力表"**，它同时描述了这份构建打开了哪些可选组件（OpenMP、KleidiAI、repack 等）。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.cpp -->
```cpp
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();
```

## 十四、谁在读这张表

`llama_print_system_info()` 遍历注册表里的每个后端，用同一个符号名取出特性函数，再把 `name = value` 拼成一行字符串。这就是启动时打印的 system_info。

<!-- src: src/llama.cpp -->
```cpp
const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

```

## 十五、对照：`ggml_cpu_has_*` 是编译期宏，不是运行期探测

同一个头文件名叫"feats"，但 `ggml_cpu_has_dotprod()` 这类函数查的是 **编译期** 的 `__ARM_FEATURE_DOTPROD`，不是运行期位图。它回答的是"我被编成了什么样"。

运行期那一侧另有实现：SVE 向量长度由 `svcntb()` 在初始化时问出来，同样只在 `__ARM_FEATURE_SVE` 编进来时才存在。

<!-- src: ggml/src/ggml-cpu/ggml-cpu.c -->
```c
int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}
```

## 十六、运行期特性如何决定内核

第二个使用点：KleidiAI 后端把运行期探测结果压成一个位掩码 `ctx.features`，内核实现按这个掩码选择。这是本课与 L5/L6（后端如何用特性决定 kernel）的接口。

<!-- src: ggml/src/ggml-cpu/kleidiai/kleidiai.cpp -->
```cpp
        const auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);
```

---

## 说明

- 本课的主文件是 `ggml/src/ggml-feats.h`（166 行）。它只有两个使用点，但"一个特性开关从定义到被后端查询的路径"要跨 5 个文件才能走完，因此本课**额外声明覆盖**：`ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`（唯一把探测结果变成后端分数的文件）、`ggml/src/ggml-backend-impl.h`（`ggml_backend_score_t` 与导出宏）、`ggml/src/ggml-backend-reg.cpp`（加载器读 score）、`ggml/include/ggml-backend.h`（特性表契约）、`ggml/src/ggml-cpu/ggml-cpu.cpp`（特性表实现）、`src/llama.cpp`（特性表的消费者）、`ggml/src/ggml-cpu/ggml-cpu.c`（编译期查询的对照）、`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`（运行期特性 -> 内核）。追加引用不会让任何原文件失配：覆盖度门禁取的是并集。
- `ggml/src/ggml-cpu/CMakeLists.txt` 也出现在本课引用里，用于说明 `GGML_USE_*` 的来源。它不在本视角的覆盖域后缀内（存在，但不计入覆盖率），属于"诚实的不计入"。
- 本课没有引用 `ggml_backend_dev_get_features` 这个符号：它在本版本里不存在。真实的查询方式是 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")`，返回类型是 `ggml_backend_feature *`。
