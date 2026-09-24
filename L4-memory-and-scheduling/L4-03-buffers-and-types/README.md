# L4-03 · buffer 与 buffer type — 课件说明

> 层：**L4 · 内存与调度** ｜ 前置课：`L4-02`

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml_backend_buft_*` 七个公共函数各自的用途，以及哪三个是必需的、另外三个不实现时框架给什么默认值（对应本课主干）；
2. 解释 `ggml_backend_buft_alloc_buffer` 与 `ggml_backend_buffer_init` 如何把 buft 与 buffer 串起来，以及为什么 buffer 的属性查询全部要转发给 buft；
3. 说出 host buffer 与 device buffer 的分工：`ggml_backend_tensor_copy` 如何用 `is_host` 决定拷贝方向；
4. 说出 **pinned host buffer 在什么场景下被用到**（对应验收点）：不用 mmap 加载模型时，权重从文件读进显存的那块中转内存；
5. 说明 `ggml_backend_tensor_alloc` 与 `ggml_backend_dev_buffer_from_host_ptr` 在 mmap 权重落位（L2-03）里各做什么。

## 覆盖的源文件（3 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/ggml-backend.h` | 438 |
| `ggml/src/ggml-backend.cpp` | 2514 |
| `src/llama-model-loader.cpp` | 1812 |

> **说明**：本课逐字引用 `ggml/src/ggml-backend.cpp` 与 `ggml/include/ggml-backend.h` 两个文件，两者都计入覆盖率。
> **说明**：第 8 幕与第十一、十二节另逐字引用 `src/llama-model-loader.cpp`（pinned host buffer 的使用方证据，1530-1588 与 1699-1735 行），同样计入覆盖率；该文件的主线（mmap 与权重落位）是 L2-03 的主题，本课不重复。
> **说明**：文中提到的 `ggml-alloc.c` 行号（1168-1230）属于 L4-01 的覆盖范围，本课不引用其源码，故不计入本课覆盖率。

## 场景（9 幕）

1. **张量脚下那块内存：buft 是厂，buffer 是地** — buffer type 回答"这类内存怎么造、有什么属性"；buffer 回答"就是这一块"。
2. **buft 的三个必需函数：名字、分配、对齐** — 公共 API 全是虚表转发：断言 buft 非空，然后调 buft->iface —— 一行都没有业务逻辑。
3. **三个可选函数：默认值也是契约** — get_max_size 默认 SIZE_MAX、get_alloc_size 默认 ggml_nbytes、is_host 默认 false。
4. **★ buffer 只是 buft 的句柄** — buffer 里只存了虚表、buft 指针、context、size、usage；它的属性全部要回问 buft。
5. **数据的进出：set / get / memset 都先解析 tensor-&gt;buffer** — 三个函数的开头一行完全相同：有 view_src 就用源张量的 buffer，否则用自己的。
6. **host buffer 与 device buffer 的分工** — 跨 buffer 搬张量时，is_host 决定"谁来搬"：host 侧用 set / get，device 之间才走后端 cpy_tensor。
7. **手工装配：宿主内存 → buffer → 张量的 data** — 设备能给出三种 buffer；把已经存在的地址装成张量，靠的是 tensor_alloc。
8. **★ pinned host buffer 到底用在什么场景** — 不用 mmap 加载模型时：为了异步上传权重到显存，先把权重读进设备的 pinned host buffer。
9. **把这一课压成一张表** — buft 七个函数、buffer 一套句柄、host / device 一条判据 —— 最后都收在"默认 buft"这一个入口上。

## 核心结论

### ★ buft 是契约，buffer 是句柄

| 对象 | 是什么 | 关键函数 |
|---|---|---|
| `ggml_backend_buffer_type_t` | "哪一类内存"：分配 + 属性 | `alloc_buffer` / `get_alignment` / `get_max_size` / `get_alloc_size` / `is_host` |
| `ggml_backend_buffer_t` | "这一块内存"：虚表 + buft 指针 + context + size + usage | `get_base` / `set_tensor` / `get_tensor` / `cpy_tensor` |

buffer 不存自己的属性 —— 四个查询函数各一行转发给 buft；唯一的回程是 `ggml_backend_buffer_get_type`。

### ★ 三个可选项与它们的默认值

`get_max_size` 缺省 `SIZE_MAX`、`get_alloc_size` 缺省 `ggml_nbytes`、`is_host` 缺省 `false`。第三条最要命：**没实现 `is_host` 不等于"是 host"**，而是被当成 device buffer —— 于是所有跨 buffer 拷贝都会绕开 `set` / `get` 这条快路。`get_alloc_size` 则可以**大于** `ggml_nbytes`，所以分配路径必须用它。

### ★ host / device 的分工只有一条判据

`ggml_backend_buffer_is_host()`（转发到 buft 的 `is_host`）决定 `ggml_backend_tensor_copy` 的方向：src 在 host 就 `set`、dst 在 host 就 `get`、两边都不是才轮到后端的 `cpy_tensor`，再不行就 `malloc` 中转。host buffer 就是"CPU 能直接读写的那一侧"，这也是 L2-03 权重上传与 L4-02 跨设备副本的共同基础。

### ★ pinned host buffer 用在非 mmap 的权重上传

走 mmap 时数据本来就在页缓存里，用 `buffer_from_host_ptr` 直接包成 buffer 即可；**不走 mmap 时**，权重必须先落到一块设备认可的宿主内存里再上传 —— 那就是设备的 `host_buffer_type` 分配的 pinned 内存：`caps.host_buffer` 声明它存在（ggml-backend.h:151-152），`ggml_backend_dev_host_buffer_type()` 把它取出来，模型加载器用它做 `ggml_backend_tensor_set_async` 的中转站（llama-model-loader.cpp:1554-1587、1727）。

## 验收点

- [x] 保真门禁：22 处引用 —— 22 个引用块 / 59 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 3 项，无空课、无幻影；全局覆盖 1290/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
