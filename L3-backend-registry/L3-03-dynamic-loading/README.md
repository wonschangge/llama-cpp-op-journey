# L3-03 · 动态加载后端 ggml-backend-dl — 课件说明

> 层：**L3 · 后端发现与注册** ｜ 前置课：`L3-02` 后端注册表与设备发现

## 学习目标

看完这一课，你应该能：

1. 说出后端动态库必须导出的符号名，并区分哪个是必需的、哪个是"自动加载时必需"的（对应验收点）；
2. 解释 `dl_load_library` / `dl_get_sym` / `dl_error` 三个原语在 POSIX 与 Windows 上各是什么；
3. 说明 `RTLD_NOW` 与 `RTLD_LOCAL` 各自解决什么问题；
4. 解释注册表为什么要求 `reg->api_version` 与 `GGML_BACKEND_API_VERSION` 精确相等；
5. 说明 `GGML_BACKEND_DL_IMPL` 宏与 `GGML_BACKEND_DL` 构建开关如何决定符号是否存在。

## 覆盖的源文件（5 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-backend-dl.h` | 46 |
| `ggml/src/ggml-backend-dl.cpp` | 49 |
| `ggml/src/ggml-backend-reg.cpp` | 606 |
| `ggml/src/ggml-backend-impl.h` | 292 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 717 |

> **说明**：本课为回答"后端动态库必须导出哪些符号"，额外引用了 `ggml/src/ggml-backend-reg.cpp`、`ggml/src/ggml-backend-impl.h` 与 `ggml/src/ggml-cpu/ggml-cpu.cpp` 的相应片段。前两者分别是 L3-02 / L3-01 的主文件，第三者属于 CPU 后端课；三者在覆盖度门禁里都算作被本课引用（`llama-coverage` 块里已一并声明）。
> **说明**：**实测说明**（避免只从注释推断）：本机 `build/` 的 `CMakeCache.txt` 里写着 `GGML_BACKEND_DL:BOOL=OFF`，因此 `nm -D build/bin/libggml-cpu.so` 的输出里既没有 `ggml_backend_init` 也没有 `ggml_backend_score` —— `GGML_BACKEND_DL_IMPL` 展开为空（impl.h:285-286）。符号是否存在与构建选项绑定，不是"动态库里天然就有"。

## 场景（9 幕）

1. **动态加载：把"支持哪些后端"推迟到 运行期** — 上游只用 93 行做这件事：一个"打开库"的原语、一个"取符号"的原语，剩下全交给注册表。
2. **POSIX 侧就三行：dlopen / dlsym / dlerror** — 两个 flag 决定了"什么时候发现缺符号"和"符号会不会进全局命名空间"。
3. **同一抽象的另一端：LoadLibraryW / GetProcAddress** — Windows 分支还要压掉系统错误弹窗；代价是 dl_error() 在 Windows 上恒为空串。
4. **dl_handle_ptr：库的寿命交给"谁持有"** — 两端都把句柄包成 unique_ptr + 自定义删除器；移动语义保证"谁持有谁负责关闭"。
5. **★ 动态库必须导出哪些符号？** — 两个字符串：必需的是 "ggml_backend_init"；可选的是 "ggml_backend_score" —— 缺了它就没法参与自动加载的择优。
6. **★ 版本校验：api_version 必须精确相等** — 不是"大于等于"，也不是兼容区间：reg->api_version 与当前版本不等就直接拒绝加载。
7. **符号不是手写的：GGML_BACKEND_DL_IMPL 宏** — 后端只提供自己的注册函数；符号名与 extern "C" 由这个宏负责，而且只在 GGML_BACKEND_DL 打开时存在。
8. **谁在调用加载器：候选名字 + GGML_BACKEND_PATH** — 启动时按写死的名字列表找库；每个名字走一次"枚举 → 打分 → 取最高分"。
9. **把这一课压成一张表** — 两个符号、一个版本号、一个句柄 —— 动态加载的全部约定。

## 核心结论

### 动态加载 = 两个原语 + 三条约定

| 项 | 内容 | 出处 |
|---|---|---|
| 打开库 | `dl_load_library()`：POSIX 用 `dlopen(path, RTLD_NOW | RTLD_LOCAL)` | `ggml-backend-dl.cpp:35` |
| 取符号 | `dl_get_sym()`：POSIX 用 `dlsym`，Windows 用 `GetProcAddress` | `ggml-backend-dl.cpp:40` / `:21` |
| 必需符号 | `"ggml_backend_init"`，返回 `ggml_backend_reg_t` | `ggml-backend-reg.cpp:237` |
| 可选符号 | `"ggml_backend_score"`，返回 `int`；自动择优路径上必需 | `ggml-backend-reg.cpp:229` / `:534-548` |
| 版本 | `reg->api_version != GGML_BACKEND_API_VERSION` 就拒绝加载 | `ggml-backend-reg.cpp:246` |

句柄的生命周期由 `dl_handle_ptr`（`std::unique_ptr` + 自定义删除器）表达；加载成功后它被 move 进注册表的 entry，库在整个进程里保持加载。

### ★ 编译期决定 -> 运行期决定

静态后端在注册表构造函数里用 `#ifdef GGML_USE_*` 登记（`ggml-backend-reg.cpp:119-136`）；动态后端则是在运行期按候选名字找库、`dlopen`、取符号、校验版本。

**这正是动态加载的代价**：一旦"支持哪些后端"推迟到运行期，主库与后端库之间就不能再靠编译器对齐 —— 所以符号名必须冻结（`ggml_backend_init` / `ggml_backend_score`）、结构体布局必须冻结（L3-01 的三张虚表），并配一个精确相等的版本号 `GGML_BACKEND_API_VERSION`。

### 失败一律不注册

加载路径上的每一种失败（打不开库、`init` 符号缺失、reg 为 NULL、版本不符、score 为 0）都只做一件事：`return nullptr`。注册表不会留下半成品后端 —— 这也是为什么"某个后端没出现"往往要在日志里找原因，而不是在设备列表里。

## 验收点

- [x] 保真门禁：21 处引用 —— 21 个引用块 / 49 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 5 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
