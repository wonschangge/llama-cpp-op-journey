# L3-04 · ggml-feats：后端能力探测 — 课件说明

> 层：**L3 · 后端发现与注册** ｜ 前置课：`L3-02`（注册表与设备）、`L3-03`（动态加载后端）

## 学习目标

看完这一课，你应该能：

1. 说出一个特性开关的完整路径：`GGML_USE_*`（CMake 编译定义）-> `#ifdef` + 运行期 `has_*` 相与 -> `ggml_backend_score()` -> 加载器按分数选库（对应验收点）；
2. 解释 `ggml-feats.h` 为什么整份内容被一道架构门包住，以及它在非 ARM64 平台上的形态；
3. 说明同一份探测结果的两条去处：后端自评（决定是否加载）与能力上报（`ggml_backend_feature` 数组）；
4. 区分编译期查询（`ggml_cpu_has_dotprod()` 查 `__ARM_FEATURE_DOTPROD`）与运行期探测（`ggml_feats_get_arch64_runtime()` 查 `AT_HWCAP`/`AT_HWCAP2`/`sysctl`/`IsProcessorFeaturePresent`）。

## 覆盖的源文件（10 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-feats.h` | 167 |
| `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp` | 42 |
| `ggml/src/ggml-backend-reg.cpp` | 606 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 717 |
| `ggml/src/ggml-cpu/CMakeLists.txt` | 688 |
| `ggml/src/ggml-backend-impl.h` | 292 |
| `ggml/include/ggml-backend.h` | 438 |
| `src/llama.cpp` | 621 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 3945 |
| `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` | 1920 |

> **说明**：本课的主文件是 `ggml/src/ggml-feats.h`（166 行）。它只有两个使用点，但"一个特性开关从定义到被后端查询的路径"要跨 5 个文件才能走完，因此本课**额外声明覆盖**：`ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`（唯一把探测结果变成后端分数的文件）、`ggml/src/ggml-backend-impl.h`（`ggml_backend_score_t` 与导出宏）、`ggml/src/ggml-backend-reg.cpp`（加载器读 score）、`ggml/include/ggml-backend.h`（特性表契约）、`ggml/src/ggml-cpu/ggml-cpu.cpp`（特性表实现）、`src/llama.cpp`（特性表的消费者）、`ggml/src/ggml-cpu/ggml-cpu.c`（编译期查询的对照）、`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`（运行期特性 -> 内核）。追加引用不会让任何原文件失配：覆盖度门禁取的是并集。
> **说明**：`ggml/src/ggml-cpu/CMakeLists.txt` 也出现在本课引用里，用于说明 `GGML_USE_*` 的来源。它不在本视角的覆盖域后缀内（存在，但不计入覆盖率），属于"诚实的不计入"。
> **说明**：本课说 `ggml-feats.h` "166 行"，用的是 `wc -l` 的口径（末行有换行符）。README 顶部的覆盖表由 lessonkit 按 `split("\n")` 计数，会把末尾换行多算一行（显示 167）—— 两个数字指的是同一个文件，不是矛盾。
> **说明**：本课没有引用 `ggml_backend_dev_get_features` 这个符号：它在本版本里不存在。真实的查询方式是 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")`，返回类型是 `ggml_backend_feature *`。

## 场景（9 幕）

1. **166 行，只回答一个问题：这台机器支持什么** — 探测结果为真时加分、为假时出局；编译期宏决定"要问哪些问题"，运行期探测决定"答案是什么"。
2. **自带一份 ABI 常量补丁：先问系统头有没有** — Linux 侧要用的 9 个常量，每一个都写成 #if !defined(...) 才定义。
3. **探测结果的形状：7 个布尔 + 1 个长度** — 值语义、按值返回、static inline —— 这三个性质决定了它不需要 ggml 导出任何符号。
4. **★ Linux：一次 getauxval，读两个位图** — 能力位是内核填好的；唯一需要"再问一次"的是 SVE 的向量长度。
5. **同一份结构体，三种问法** — macOS 按名字问 sysctl，Windows 按编号问 IsProcessorFeaturePresent；两侧都把 sve_cnt 记为 0。
6. **★ 特性开关的路径：编译期 #ifdef x 运行期探测** — cpu-feats.cpp 是唯二使用点之一：它把探测结果翻译成"这个后端变体值多少分"。
7. **路径的终点：加载器在后端被加载之前就问它** — score 为 0 的 .so 直接 return nullptr；不是崩溃，是安静地跳过。
8. **★ 第二条路径：把能力报出去，而不是用来打分** — ggml_backend_cpu_get_features() 返回一张以 { nullptr, nullptr } 结尾的特性表。
9. **把这一课压成一张"谁定义 / 谁读"的表** — 一条路径决定"要不要加载你"，另一条路径决定"怎么描述你"。

## 核心结论

### `ggml-feats.h` = 运行期那一半

它只有 166 行，整份内容被 `#if defined(__aarch64__) || defined(_M_ARM64)` 包住，按平台分成 Linux（`getauxval` + `prctl`）、macOS（`sysctlbyname`）、Windows（`IsProcessorFeaturePresent`）三支，产出一个 8 字段的值语义结构体。**它不是"后端特性表"，而是"CPU 能力探测器"** —— 后端特性表是它的下游消费者之一。

### ★ 一个特性开关的完整路径

```text
CMake: ARCH_DEFINITIONS += GGML_USE_DOTPROD      (ggml/src/ggml-cpu/CMakeLists.txt)
  -> #ifdef GGML_USE_DOTPROD                     (cpu-feats.cpp)
     + af.has_dotprod                            (ggml-feats.h: hwcap & HWCAP_ASIMDDP)
     -> 相与：不满足 return 0，满足 score += 1<<k
  -> ggml_backend_score()                        (GGML_BACKEND_DL_SCORE_IMPL 导出)
  -> dl_get_sym + score_fn() == 0 ? 放弃 : 加载  (ggml-backend-reg.cpp)
```

编译期开关决定"这份二进制的前提"，运行期探测决定"这台机器满不满足前提"，两者都不满足时的表现是**安静地不被加载**。

### 两条路径，同一份探测结果

| 路径 | 入口 | 终点 | 回答的问题 |
|---|---|---|---|
| A 后端自评 | `ggml_backend_cpu_aarch64_score()` | `ggml-backend-reg.cpp` 的加载器 | 这个后端要不要加载 |
| B 能力上报 | `ggml_backend_cpu_get_features()` | `llama_print_system_info()` | 这个后端支持什么 |

B 路径的取值方式是 `ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")`，返回以 `nullptr` 结尾的 `ggml_backend_feature[]`。

## 验收点

- [x] 保真门禁：25 处引用 —— 25 个引用块 / 39 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 10 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
