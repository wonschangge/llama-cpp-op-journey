<!-- llama-coverage
ggml/include/gguf.h
ggml/src/gguf.cpp
-->

# L1-06 · GGUF：算子的权重从文件来 — 源文件

**一句话**：前面五课讲的都是"内存里的算子"——张量、op 枚举、图、量化块、context。而真实推理时，这些算子的**权重**来自磁盘上的一个文件，这个文件就是 **GGUF**。GGUF 是**自描述**的容器：元数据（KV 段）在前，张量信息（张量信息段）居中，权重数据（张量数据区）在最后，且数据区的起始偏移被**对齐到 alignment**。

把这一课和 L2-03 连起来看，就是完整的一条线：L1-06 讲文件怎么摆，L2-03 讲 mmap 怎么把文件里的字节直接变成 `ggml_tensor::data` 指向的内存 —— 中间**一次拷贝都不做**。这条路能成立，靠的正是本课最后两幕讲的"数据区连续 + 按 alignment 对齐"。

---

## 一、文件格式：上游的七条注释就是验收点

`gguf.h` 开头用七条注释写死了整个文件布局。**本课的验收点就是复述这张图**：magic -> version -> n_tensors -> n_kv -> KV 段 -> 张量信息段 -> 张量数据区（可选、对齐）。注释还补了三条编码规则：字符串 = `uint64` 长度 + 无 `\0` 的字节；所有枚举是 `int32_t`；所有 bool 是 `int8_t`。

<!-- src: ggml/include/gguf.h -->
```c
// GGUF files have the following structure:
//
// 1. File magic "GGUF" (4 bytes).
// 2. File version (uint32_t).
// 3. Number of ggml tensors in file (int64_t).
// 4. Number of key-value-pairs in file (int64_t).
// 5. For each KV pair:
//   1. The key (string).
//   2. The value type (gguf_type).
//   3a. If the value type is GGUF_TYPE_ARRAY:
//     1. The type of the array (gguf_type).
//     2. The number of elements in the array (uint64_t).
//     3. The binary representation of each element in the array.
//   3b. Otherwise:
//     1. The binary representation of the value.
// 6. For each ggml tensor:
//   1. The tensor name (string).
//   2. The number of dimensions of the tensor (uint32_t).
//   3. For each dimension:
//     1. The size of the tensor in the dimension (int64_t).
//   4. The tensor data type (ggml_type).
//   5. The tensor data offset in the tensor data binary blob (uint64_t).
// 7. The tensor data binary blob (optional, aligned).
```

## 二、常量：magic、版本、对齐

`GGUF_MAGIC` 是字符串常量 `"GGUF"`，`GGUF_VERSION` 是 3，`GGUF_DEFAULT_ALIGNMENT` 是 32。对齐可以被文件里的 KV `general.alignment` 覆盖 —— 读的时候会检查它必须是 `uint32`、并且是 2 的幂。

<!-- src: ggml/include/gguf.h -->
```c
#define GGUF_MAGIC   "GGUF"
#define GGUF_VERSION 3

#define GGUF_KEY_GENERAL_ALIGNMENT "general.alignment"

#define GGUF_DEFAULT_ALIGNMENT 32

```

## 三、★ KV 段：自描述的核心

KV 段让 GGUF 成为**自描述**格式：键、值的类型、值的字节都在文件里。类型编号（`gguf_type`）是稳定的数字 —— 读的人据此决定后面读几个字节。因为是"任意键值"，所以超参、词表、甚至整段 chat template 都能塞进同一个机制里，不需要为每种元数据单独设计字段。

<!-- src: ggml/src/gguf.cpp -->
```cpp
    // KV pairs
    {
        for (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type     = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;

            try {
                ok = ok && gr.read(key);
            } catch (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            if (ok && key.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = false;
            }
            for (size_t j = 0; ok && j < ctx->kv.size(); ++j) {
                if (key == ctx->kv[j].key) {
                    GGML_LOG_ERROR("%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, key.c_str(), j, i);
                    ok = false;
                }
            }
            if (!ok) {
                break;
            }

            ok = ok && gr.read(type);
            if (type == GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.read(type);
                ok = ok && gr.read(n);
            }
            if (!ok) {
                break;
            }

```

## 四、读流程：从文件头到数据区

读流程的顺序与文件布局一一对应。四个头字段读完后依次是版本检查（0 / 字节序 / v1 / 过新）、KV 段、张量信息段；最后把文件指针推到对齐位置，并记下数据区起点 `ctx->offset`。其中张量信息的读取还带一串校验：重名、维数越界、负数维、类型越界、`ne[0]` 不是块大小整数倍 —— 每一条都是"读到坏文件不要把它当合法数据用"。

<!-- src: ggml/src/gguf.cpp -->
```cpp
                ok = false;
            }
            if (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = false;
                break;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            for (int64_t j = 0; ok && j < i; ++j) {
                if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                    GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\n", __func__, info.t.name, j, i);
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.read(n_dims);
            if (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = false;
                break;
            }
            for (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                if (j < n_dims) {
                    ok = ok && gr.read(info.t.ne[j]);
                }

                // check that all ne are non-negative
                if (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    break;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable; the guard also avoids a division by zero below)
            if (ok && ggml_nelements(&info.t) > 0 &&
                ((INT64_MAX/info.t.ne[1] <= info.t.ne[0]) ||
                 (INT64_MAX/info.t.ne[2] <= info.t.ne[0]*info.t.ne[1]) ||
                 (INT64_MAX/info.t.ne[3] <= info.t.ne[0]*info.t.ne[1]*info.t.ne[2]))) {

                GGML_LOG_ERROR("%s: total number of elements in tensor '%s' with shape "
                    "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_COUNT);
                ok = false;
                break;
            }
            const size_t  type_size = ggml_type_size(info.t.type);
            const int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (int) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                break;
            }

            // check that the size of the tensor in bytes is representable
            if (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = false;
                break;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        if (!ok) {
            break;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx->info.push_back(info);
    }

    if (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    if (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            size_t padded_size = GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
            if (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                return nullptr;
            }
```

## 五、★ 对齐与连续性

`GGML_PAD(x, n) = ((x) + (n) - 1) & ~((n) - 1)`（定义在 `ggml/include/ggml.h`）。读的时候用 `seek` 把文件指针推到"相对起点"的对齐位置；写的时候用 `pad` 补 0 字节。两侧算出同一个数，文件才能被对方读回。

紧接着的循环断言 `ti.offset == ctx->size`：每个张量的 offset 必须等于前面所有张量**补齐后的累计值**。这条断言把"数据区是一整块连续区间"从约定变成了强制 —— 而连续 + 对齐，正是 L2-03 的 mmap 能零拷贝把张量指向文件的原因。

<!-- src: ggml/src/gguf.cpp -->
```cpp
    if (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            size_t padded_size = GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
            if (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
```

## 六、写流程

写入顺序与读取顺序对称：magic 的四个字节分别写、version、n_tensors、n_kv，然后是每条 KV、每条张量信息、`gw.pad(ctx->alignment)`，最后是各张量数据。张量的 `offset` 由 `gguf_add_tensor` 推导（前一个的 offset + 前一个补齐后的字节数），`gguf_set_tensor_data` **只记录指针、不拷贝数据**。

<!-- src: ggml/src/gguf.cpp -->
```cpp
    // write header
    gw.write(GGUF_MAGIC[0]);
    gw.write(GGUF_MAGIC[1]);
    gw.write(GGUF_MAGIC[2]);
    gw.write(GGUF_MAGIC[3]);
    gw.write(ctx->version);
    gw.write(n_tensors);
    gw.write(n_kv);

    // write key-value pairs
    for (int64_t i = 0; i < n_kv; ++i) {
        gw.write(ctx->kv[i]);
    }

    // write tensor info
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);
//>> ---- ggml/src/gguf.cpp:1660-1665 ----
    const size_t offset_data = gw.written_bytes;

    // write tensor data
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
```

## 七、写侧的 pad：与读侧的 GGML_PAD 是同一个算法

写侧的对齐就是一个 while 循环：每补一个 0 字节，`written_bytes` 加一，直到整除 `alignment`。读侧用 `GGML_PAD` 算出同一个目标位置再 seek 过去。**两侧必须算出同一个数**，否则写出来的文件自己都读不回来。

<!-- src: ggml/src/gguf.cpp -->
```cpp
    void pad(const size_t alignment) {
        while (written_bytes % alignment != 0) {
            const int8_t zero = 0;
            write(zero);
        }
    }
```

---

## 说明

- 本课引用 `ggml/include/gguf.h` 与 `ggml/src/gguf.cpp` 两个文件，均计入覆盖率。
- 第 6 / 8 幕的对齐算例（1140 -> 1152、93 -> 96）是用 `GGML_PAD` 的定义手算的演示数字，不是从某个真实模型文件读出来的；`GGML_PAD` 的定义位于 `ggml/include/ggml.h`（属 L1-01 覆盖，本课不重复引用）。
