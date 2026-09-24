# L2-03 · 权重加载与内存映射 — 课件说明

> 层：**L2 · 从模型到图** ｜ 前置课：`L2-02`（架构表与超参：张量名与"它属于哪一层"来自那里）

## 学习目标

看完这一课，你应该能：

1. 说出权重张量在**什么时刻**决定自己落到哪个后端，以及这个决定的输入是什么（对应验收点）；
2. 解释 mmap 与 read 两条路径在加载时间、常驻内存、额外拷贝上的差别，以及 `MAP_POPULATE` / `madvise` / NUMA 三个可调项的作用；
3. 说明 `use_mmap` / `use_direct_io` 怎么从 `load_mode` 得到，平台不支持 mmap 时会怎样；
4. 区分 llama.cpp 的两套 I/O：权重加载（`llama_file` / `llama_mmap`）与状态序列化（`llama_io_read_i` / `llama_io_write_i`）。

## 覆盖的源文件（6 个）

| 文件 | 行数 |
|---|---|
| `src/llama-model-loader.h` | 266 |
| `src/llama-model-loader.cpp` | 1812 |
| `src/llama-mmap.h` | 80 |
| `src/llama-mmap.cpp` | 827 |
| `src/llama-io.h` | 36 |
| `src/llama-io.cpp` | 21 |

> **说明**：本课引用 `src/llama-model-loader.{h,cpp}`、`src/llama-mmap.{h,cpp}`、`src/llama-io.{h,cpp}` 共 6 个源文件，全部计入覆盖率。
> **说明**：文中提到的调用方（`src/llama-model.cpp`：`init_mappings()` / `load_all_data()` 的调用点、`use_mlock` 的判定、`mlock_mmaps` 的构造）属于 L2-05 的覆盖范围；`src/llama-context.cpp`（llama-io 的实现类）属于 L2-07。本课不引用这两个文件的代码，故不计入本课覆盖率。
> **说明**：文中出现的命令行开关 `--load-mode` 由真实二进制的 `llama-cli --help` 实测确认存在。

## 场景（9 幕）

1. **一个权重 = 一段字节区间 + 一个张量** — loader 在建图之前就记下每个权重的（文件号，文件内偏移）；落到哪由后面决定。
2. **load_mode 只折成两个布尔** — 要不要 mmap、要不要 direct I/O，在构造 loader 时就定死，之后整条加载路径都按它分支。
3. **llama_mmap 的门面：三个操作** — 建立映射、拿基地址、按页归还。三个平台分支（POSIX / Win32 / 不支持）藏在 pimpl 后面。
4. **★ mmap：省掉的是拷贝，不是必然的 I/O** — 映射只改页表；字节什么时候从磁盘进来，由 prefetch / lazy / numa 三个提示决定。
5. **★ 指过去还是拷过去** — load_all_data 的第一句判据：数据是"已经有地址"，还是"还得从文件搬"。
6. **不走映射时：三条搬运路径** — host buffer 直接读；设备 buffer 有异步能力就走 pinned host buffer 中转；没有就现开临时缓冲。
7. **★ 谁决定权重落到哪个后端** — 答案是一串判据 + 一个函数：按 buft_list 的顺序问"这个 buffer type 能不能跑用到这个权重的算子"。
8. **同一个工程里的另一套 I/O** — llama-io.h 的虚接口不服务权重，它服务"状态"：KV 与序列的存取。
9. **把这一课压成一张表** — 两个输入、两个决策、两个执行动作。

## 核心结论

### ★ mmap 省的是拷贝，不是 I/O

| | mmap | read |
|---|---|---|
| 加载时做什么 | 建映射（改页表） | 把字节读进用户缓冲 |
| 谁触发磁盘 I/O | 第一次访问该页时 | 加载时一次付清 |
| 额外拷贝 | 无（直接用文件页） | 页 -> 用户缓冲 |
| 可调项 | `prefetch` / `numa` / `lazy_ranges` | 基本没有 |

源码注释把这一点写得很直白：`MAP_POPULATE would fault in the lazy ranges too`。所以"mmap 不占 RAM 也不占加载时间"要加前提 —— **它把 I/O 变成按页触发，并省掉用户态缓冲**；页什么时候进来由预取策略决定。

### ★ 落位在建图时决定，在加载时执行

决定链：`load_mode` -> `use_mmap` / `use_direct_io`（559-560 行）-> 张量所属层决定用哪张 `buft_list`（1215-1230 行）-> `select_weight_buft()` 按顺序试（1067-1075 行）-> `weight_buft_supported()` 造假算子问后端 `supports_op`（1058-1059 行）-> 选定 buffer type，张量被建在"该 buft 的 ggml context"里。

`load_all_data()` 只是执行：映射就直接把 `cur->data` 指过去（1658 行），否则拷贝或中转。**所以"权重在哪个后端"这个问题，在建图阶段就有答案了，而且答案不依赖数据。**

### 一套工程，两套 I/O

| | 权重 I/O | 状态 I/O（llama-io） |
|---|---|---|
| 方向 | 只读 | 读 + 写 |
| 粒度 | 整个张量（或按行懒读） | `(offset, size)` 分片 |
| 载体 | `llama_file` + `llama_mmap` | `llama_io_read_i` / `llama_io_write_i` |
| 用途 | 加载 GGUF 权重 | 存取 KV cache / 序列状态 |

分界很清楚：**文件里躺着的**走 mmap / read，**运行期反复存取的**走 llama-io。

## 验收点

- [x] 保真门禁：23 处引用 —— 23 个引用块 / 47 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 6 项，无空课、无幻影；全局覆盖 59/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 33 文件 8 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
