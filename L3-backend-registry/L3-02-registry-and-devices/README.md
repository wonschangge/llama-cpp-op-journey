# L3-02 · ★ 后端注册表与设备发现 — 课件说明

> 层：**L3 · 后端发现与注册** ｜ 前置课：`L3-01`

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml_backend_registry` 的两个成员，以及“设备 0”这个编号是在哪里被定下来的；
2. 列出至少 6 个编译期后端开关，并说明它们由谁（哪个 CMake 函数的哪一段）决定；
3. 解释为什么“同一份 `--tensor-split` 参数在两次运行里可能指向不同的卡”，以及为什么用 `--device` 更稳；
4. 说明 Meta 设备与注册表里普通设备的区别（谁创建它、它的 `reg` 字段是什么、为什么 `dev_count()` 数不到它）。

## 覆盖的源文件（3 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-backend-reg.cpp` | 606 |
| `src/llama.cpp` | 621 |
| `ggml/src/ggml-backend-meta.cpp` | 2518 |

> **说明**：本课逐字引用 3 个文件：`ggml/src/ggml-backend-reg.cpp`、`ggml/src/ggml-backend-meta.cpp`（计划里本课的 2 个），外加 `src/llama.cpp` —— 只为一处：设备顺序的消费侧（第 6 幕 / 第九节）。该文件本就由 L2-01 覆盖，本课是重复声明（门禁口径：至少被一课声明）。
> **说明**：按行号指路、未引用原文因而**不计入本课覆盖率**的文件：`ggml/src/CMakeLists.txt:428-439`（谁定义 GGML_USE_*）、`common/arg.cpp:1116-1180`（设备名解析与 RPC 注册）、`ggml/src/ggml-cuda/ggml-cuda.cu:5798` 与 `ggml/include/ggml-cuda.h:11/14/17`（CUDA/ROCm/MUSA 设备名）、`ggml/src/ggml-cpu/ggml-cpu.cpp:353-357`（CPU 设备名）、`ggml/src/ggml-metal/ggml-metal.cpp:307`、`ggml/src/ggml-rpc/ggml-rpc.cpp:2273-2276`（RPC 设备数来自 context）、`ggml/src/ggml-cann/ggml-cann.cpp:2807-2810`（CANN 设备是 GPU 型）、`src/llama-model.cpp:1488-1542`（splits 与 dev_layer，L2-05 已逐字引用）。
> **说明**：第 3、5 幕的“本机实测”来自作者自建的临时 harness（直接调用 `ggml_backend_register()` 注册两个假后端，再读 `ggml_backend_dev_count()` / `ggml_backend_dev_get(i)`），跑在本仓库 `build/` 下已有的 CPU-only 构建上；该 harness 不属于本课件仓库，也不计入覆盖率。

## 场景（10 幕）

1. **注册表：这台机器上有什么** — 结构体里只有两个 vector：后端列表，和它们贡献的设备列表。
2. **★ 有哪些后端，是编译期定下的** — 注册表构造函数里 16 个 #ifdef；开关打开一个，就多注册一个后端。
3. **注册 = 追加到末尾 + 去重** — register_backend() 把后端 push_back，再按它报的设备数逐个 push_back 设备。
4. **get_reg()：注册发生在第一次触碰注册表时** — 函数内 static 单例；静态后端在构造函数里就位，动态加载只是随后补票。
5. **设备枚举：下标就是设备号** — ggml_backend_dev_get(i) 返回 devices[i]；顺序由注册顺序决定，与名字无关。
6. **★ 注册顺序怎么变成 offload 结果** — llama.cpp 按注册顺序把 GPU 型设备收进 model->devices；tensor-split / main-gpu 的下标就是它。
7. **★ 动态加载只能追加到静态后端之后** — load_all_from_path() 按写死的顺序找 15 个名字的动态库；找到的都 push_back 到数组末尾。
8. **Meta backend：把 N 张卡合成一个设备** — ggml_backend_meta_device() 造出来的设备不在注册表里：reg = nullptr。
9. **多后端共存时，设备名是谁起的** — 注册表不发明名字：它只转手后端给的 get_name；Meta 设备的名字是拼出来的。
10. **把这一课压成一张表** — 三句话：有哪些后端看编译期；谁是设备 0 看注册顺序；Meta 设备不在注册表里。

## 核心结论

### ★ 有哪些后端 = 编译期开关

`ggml_backend_registry` 的构造函数里有 **16 个** `#ifdef GGML_USE_*`（120-173 行），每个开关注册一个后端家族。这些宏由 `ggml/src/CMakeLists.txt:428-439` 在 `GGML_BACKEND_DL=OFF` 时加给 `ggml` 目标。

运行期的 `ggml_backend_load_all_from_path()`（574-604）尝试 15 个候选名字的动态库，是在这个静态集合**之外**做补充 —— 而且它排在同一批数组的**后面**。

### ★ 设备顺序 = 注册顺序，且无人重排

`register_backend()` 只做 `push_back`（201），`register_device()` 也只做 `push_back`（217）；`ggml_backend_dev_get(i)` 直接返回 `devices[i]`（340-343）。没有任何一步按名字、显存或性能重排。

实测（本机）：先注册 `ZZZ` 再注册 `AAA`，得到 `dev[0]=CPU, dev[1]=ZZZ0, dev[2]=AAA0` —— **顺序 = 注册顺序，不是字典序**；重复注册同一个 `reg`，设备数不变。

### ★ 顺序决定 offload：谁是“第一个 GPU”

`llama.cpp` 按 `ggml_backend_dev_get(i)` 的顺序收集 GPU 型设备（`src/llama.cpp:222-231`），拼成 `model->devices`。`--main-gpu N` 是它的下标（297-299），`--tensor-split` 的第 j 项对应 `devices[j]`（`llama-model.cpp:1488-1508`），最终由 `upper_bound(splits, ...)` 决定每层落在哪块卡（1529），写进 `dev_layer[il]`（1540-1542）。

静态注册表与动态加载表是**两张不同的写死顺序表**（`CANN` 与 `CUDA` 的先后正好相反），所以“某个后端是编译进来的还是加载进来的”会改变设备下标 —— 这也是 `--device`（走名字，`by_name`）比依赖下标更稳的原因（L2-03 权重落位、L2-05 dev_layer、L4-02 调度器都建立在这份顺序上）。

### Meta 设备：合出来的设备，不在注册表里

`ggml_backend_meta_device()`（215-245）把 N 个简单设备合成一个设备：内存相加、`supports_op` 取交集、类型为 `GGML_BACKEND_DEVICE_TYPE_META`；它造的 `ggml_backend_device` 里 `reg = nullptr`，全文件没有注册调用，因此不出现在 `ggml_backend_dev_count()` 的结果里，而由调用方（张量并行的 llama.cpp）按需创建。名字是拼出来的：`Meta(CUDA0,CUDA1)`。

## 验收点

- [x] 保真门禁：19 处引用 —— 19 个引用块 / 27 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 3 项，无空课、无幻影；全局覆盖 1048/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 9 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：10 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
