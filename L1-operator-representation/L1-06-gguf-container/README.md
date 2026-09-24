# L1-06 · GGUF：算子的权重从文件来 — 课件说明

> 层：**L1 · 算子的表示** ｜ 前置课：`L1-01`

## 学习目标

看完这一课，你应该能：

1. 按字节顺序说出 GGUF 从文件头到张量数据区的完整布局（对应验收点）；
2. 解释为什么说 GGUF 是**自描述**格式，并举出一个只能靠 KV 段承载的元数据（如 chat template）；
3. 说出张量数据区的起始偏移是如何被 `GGML_PAD` 对齐到 `alignment` 的，以及 alignment 的默认值与覆盖方式；
4. 说明 `ti.offset == ctx->size` 这条断言为什么等价于"数据区连续"，以及它对 L2-03 mmap 的意义。

## 覆盖的源文件（2 个）

| 文件 | 行数 |
|---|---|
| `ggml/include/gguf.h` | 212 |
| `ggml/src/gguf.cpp` | 1715 |

> **说明**：本课引用 `ggml/include/gguf.h` 与 `ggml/src/gguf.cpp` 两个文件，均计入覆盖率。
> **说明**：第 6 / 8 幕的对齐算例（1140 -> 1152、93 -> 96）是用 `GGML_PAD` 的定义手算的演示数字，不是从某个真实模型文件读出来的；`GGML_PAD` 的定义位于 `ggml/include/ggml.h`（属 L1-01 覆盖，本课不重复引用）。

## 场景（8 幕）

1. **算子的权重，从文件来** — GGUF 是"自描述"的容器：元数据与张量信息写在文件里，读的人不需要任何额外约定。
2. **文件头：magic · version · n_tensors · n_kv** — 4 + 4 + 8 + 8 = 24 字节。按顺序读这四样，任何一样不对就直接返回 NULL。
3. **★ KV 段：GGUF 为什么是自描述的** — 键是"长度 + 字节"，类型是 int32 枚举，"元素个数"只在数组时出现。任何元数据都能塞进来。
4. **★ 文件布局：头 -> KV 段 -> 张量信息段 -> padding -> 张量数据** — 上游把整个布局按 1..7 条写成了注释。本课其余各幕，就是把这张图逐格讲完。
5. **张量信息段：名字 + 形状 + 类型 + offset** — 读进来的是一个 gguf_tensor_info：里面已经有一个 ggml_tensor（装信息），外加一个数据区偏移。
6. **★ 对齐：GGML_PAD 把数据区推到 alignment 的整数倍** — 读的时候靠 seek 把文件指针推到对齐位置；写的时候靠 pad 补 0 —— 两边算出同一个数。
7. **写的一侧：gguf_set_tensor_data 只记指针，offset 由前一个张量决定** — 写入顺序与读取顺序严格对称：头 -> KV -> 张量信息 -> pad -> 数据。
8. **把这一课压成一张字节表** — 从文件第 0 字节到张量数据第一字节，逐段过一遍；表里的每一项都能在源码里找到出处。

## 核心结论

### ★ 文件布局（验收点）

```text
偏移 0    : magic   "GGUF"          4 字节
偏移 4    : version  uint32         4 字节
偏移 8    : n_tensors int64         8 字节
偏移 16   : n_kv      int64         8 字节
偏移 24   : KV 段（n_kv 条，变长）
之后      : 张量信息段（n_tensors 条，变长）
之后      : padding 补 0，把数据区起点推到 alignment 的整数倍
对齐后    : 张量数据区（连续，每个张量按 alignment 补齐）
```

编码规则：字符串 = `uint64` 长度 + 无 `\0` 的字节；枚举 = `int32_t`；bool = `int8_t`。常量：`GGUF_MAGIC = "GGUF"`、`GGUF_VERSION = 3`、`GGUF_DEFAULT_ALIGNMENT = 32`。

### ★ GGUF 是自描述的

KV 段里每条都是"键 + 类型编号 + 值字节"。类型编号（`gguf_type`）写在文件里，所以读的人不需要预先知道有哪些元数据、也不需要为每种元数据改格式。超参、词表、chat template 走的是**同一个**机制。

一个特例：`general.alignment` 本身是普通 KV，但读完后会被提升成 `ctx->alignment`，直接决定数据区的字节布局。

### ★ 对齐 + 连续 = mmap 的前提

读：`seek(start + GGML_PAD(tell() - start, alignment))` 把文件指针推到对齐位置，再记下 `ctx->offset`。写：`pad(alignment)` 补 0 直到 `written_bytes % alignment == 0`。两侧算出同一个数。

连续性由 `ti.offset == ctx->size` 强制：每个张量的 offset 必须等于前面所有张量补齐后的累计值，所以数据区是**一块无洞、无重叠的连续区间**。L2-03 正是靠这两点，才能把文件 mmap 进内存、把每个 `ggml_tensor::data` 直接指向 `blob + offset`，一次拷贝都不做。

### 对称的读写 API

| 方向 | 入口 | 关键行为 |
|---|---|---|
| 读 | `gguf_init_from_file` | 读头 -> 读 KV -> 读张量信息 -> 对齐 -> 记 `ctx->offset` |
| 读 | `gguf_init_from_buffer` / `_from_callback` | 同一套 `gguf_init_from_reader`，只换数据来源 |
| 写 | `gguf_add_tensor` | `offset` 由前一个张量推导，保证连续 |
| 写 | `gguf_set_tensor_data` | 只记录指针，不拷贝 |
| 写 | `gguf_write_to_file` | 头 -> KV -> 张量信息 -> pad -> 数据，与读严格对称 |

错误处理统一为"打日志 + 释放 + 返回 `nullptr`（写侧返回 `false`）"：magic 不符、版本非法、KV 读失败、张量信息读失败、数据区 seek 失败、offset 不连续，每一条都有独立的错误信息。

## 验收点

- [x] 保真门禁：15 处引用 —— 15 个引用块 / 46 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 2 项，无空课、无幻影；全局覆盖 34/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 18 文件 2 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：8 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
