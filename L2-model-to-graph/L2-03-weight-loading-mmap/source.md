<!-- llama-coverage
src/llama-model-loader.h
src/llama-model-loader.cpp
src/llama-mmap.h
src/llama-mmap.cpp
src/llama-io.h
src/llama-io.cpp
-->

# L2-03 · 权重加载与内存映射 — 源文件

**一句话**：权重加载解决的是"GGUF 文件里的一段字节，怎么变成某个后端 buffer 里的一个张量"。这条路上有两个独立的问题：**怎么搬**（mmap 直接把文件页映射成张量数据，还是 read 进内存）和**搬到哪**（哪个 buffer type 持有它）。两者都在建图阶段就定了，`load_all_data()` 只是执行。

本课覆盖 6 个文件：权重侧是 `src/llama-model-loader.{h,cpp}`，文件与映射侧是 `src/llama-mmap.{h,cpp}`；另外用 `src/llama-io.{h,cpp}` 对照 llama.cpp 里**另一套** I/O 抽象（状态序列化，不是权重）。

---

## 一、权重索引：一个权重就是一段字节区间

`llama_model_loader` 在构造时就把 GGUF 里每个张量的数据位置记下来：文件号 `idx`、文件内绝对偏移 `offs`，再加上一个等着被填充的 `ggml_tensor *`。

注意 `offs` 的来历：`gguf_get_data_offset()`（数据段基址）+ `gguf_get_tensor_offset()`（张量在数据段内的偏移）。这正是 mmap 能工作的前提 —— **偏移是文件内的绝对偏移**，所以映射基址 + 偏移就等于张量数据地址。紧接着的边界检查说明 loader 对"数据必须真的在文件里"是强校验的。

<!-- src: src/llama-model-loader.h -->
```c
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
```

## 二、加载模式：两个布尔在构造时定下

`llama_load_mode` 有六个取值，落到 loader 里只是两个布尔。`AUTO` 也归到 `use_mmap = true`，它的"自动"语义体现在别处：真正分配 buffer 时按设备能力决定能不能把映射交给后端。

`mlock` 不由这里决定：loader 只在 `load_all_data()` 里按调用方传进来的 `llama_mlocks *` 办事。如果平台根本不支持 mmap，构造末尾会把 `use_mmap` 关掉并打一条警告 —— 于是"选择 mmap"在运行期永远不会失败，只会静默退化成 read。

<!-- src: src/llama-model-loader.cpp -->
```c
    this->use_mmap      = load_mode == LLAMA_LOAD_MODE_MMAP || load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK || load_mode == LLAMA_LOAD_MODE_AUTO;
    this->use_direct_io = load_mode == LLAMA_LOAD_MODE_DIRECT_IO;
//>> ---- src/llama-model-loader.cpp:829-832 ----
    if (this->use_mmap && !llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        this->use_mmap = false;
    }
```

## 三、llama_mmap：三个操作 + 一个平台门

对外接口只有四项：构造（建立映射）、`size()` / `addr()`（拿基地址）、`unmap_fragment()`（按页归还）、`SUPPORTED`（平台门）。三个平台分支（POSIX、Win32、不支持）都在 `impl` 里，调用方看不到。

`ranges` 是 `[first, last)` 的字节区间列表，它的用途是**懒读**：告诉映射层哪些段要留着随机访问，其余段可以放心顺序预取。

<!-- src: src/llama-mmap.h -->
```c
struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
```

## 四、★ 映射的建立：只读、整文件、预取可调

一次 `mmap()` 覆盖整个文件，`PROT_READ` + `MAP_SHARED`。因为声明了只读，内核可以让这些页直接指向文件的页缓存，不需要给写者准备私有副本 —— 这是 mmap 省掉"用户态缓冲 + 一次拷贝"的根源。

但**字节什么时候进来是可调的**：`prefetch` 非 0 且没有懒读区间时加 `MAP_POPULATE`（源码注释：`MAP_POPULATE would fault in the lazy ranges too`）；`posix_madvise` 用 `WILLNEED` / `RANDOM`给不同区间不同的读法提示；NUMA 机器上直接否决预取。所以"mmap 让加载不花时间"是一个**带前提的说法**：省掉的是拷贝与用户态缓冲，页什么时候进来另说。

<!-- src: src/llama-mmap.cpp -->
```c
    impl(struct llama_file * file, size_t prefetch, bool numa, const llama_mmap::ranges & lazy_ranges) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        // MAP_POPULATE would fault in the lazy ranges too
        if (prefetch && lazy_ranges.empty()) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        // page-aligned madvise over [beg, end), clamped to the file
        auto advise = [&](size_t beg, size_t end, int advice, const char * name) {
            const size_t page_size = sysconf(_SC_PAGESIZE);
            beg = beg & ~(page_size - 1);
            end = std::min((end + page_size - 1) & ~(page_size - 1), file->size());
            if (beg >= end) {
                return;
            }
            if (posix_madvise((char *) addr + beg, end - beg, advice)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., %s) failed: %s\n", name, strerror(errno));
            }
        };

        if (prefetch > 0) {
            for (const auto & range : ranges_complement(lazy_ranges, std::min(file->size(), prefetch))) {
                advise(range.first, range.second, POSIX_MADV_WILLNEED, "POSIX_MADV_WILLNEED");
            }
        }
        for (const auto & range : lazy_ranges) {
            advise(range.first, range.second, POSIX_MADV_RANDOM, "POSIX_MADV_RANDOM");
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());
    }
```

## 五、按页归还：unmap_fragment

`unmap_fragment()` 先把区间对齐到页边界，再 `munmap()` —— 所以"还一半页"是不会发生的：对齐后长度为 0 就直接返回。归还之后它同步维护自己那份"还有哪些段映射着"的账本，析构函数只对账本里剩下的段收尾。

<!-- src: src/llama-mmap.cpp -->
```c
    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
```

## 六、收尾：把没用的页还回去

加载完成后，`load_all_data()` 用 `mmaps_used` 记下的区间做一次收尾：**数据段之前**（文件头 / 元数据）与**用到的区间之后**（对齐边角、未用张量）全部 `unmap_fragment`。映射不是"要么全留要么全放"，只有真正被权重用到的页留下来。

<!-- src: src/llama-model-loader.cpp -->
```c
    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
```

## 七、llama_file：read 侧的底座

不走 mmap 时，字节靠 `llama_file` 搬。它在 Linux 上可以打开 `O_DIRECT` 并把 `st_blksize` 记成`alignment`；`read_raw()` 因此分两种：direct I/O 走 `read_aligned_chunk()`（对齐 + 临时对齐缓冲 + memcpy），否则直接 `read_raw_unsafe()`。

这套 alignment 一路上传：`load_all_data()` 会取 `file->read_alignment()` 来算对齐偏移，并据此把中转缓冲从 1MB 放大到 64MB + 2 个对齐块。

<!-- src: src/llama-mmap.cpp -->
```c
#ifdef __linux__
    bool init_fd() {
        fd = open(fname.c_str(), O_RDONLY | O_DIRECT);

        if (fd != -1) {
            struct stat file_stats{};
            fstat(fd, &file_stats);

            size = file_stats.st_size;
            alignment = file_stats.st_blksize;

            off_t ret = lseek(fd, 0, SEEK_SET);
            if (ret == -1) {
                throw std::runtime_error(format("seek error: %s", strerror(errno)));
            }
            return true;
        }
        return false;
    }
//>> ---- src/llama-mmap.cpp:344-350 ----
    void read_raw(void * ptr, size_t len) {
        if (has_direct_io()) {
            read_aligned_chunk(ptr, len);
        } else {
            read_raw_unsafe(ptr, len);
        }
    }
//>> ---- src/llama-mmap.cpp:388-392 ----
    size_t read_alignment() const {
        return alignment;
    }

    size_t alignment = 1;
```

## 八、★ load_all_data：先排序，再逐个落位

`load_all_data()` 是整条加载路径的执行者。开头先处理"没有文件"的虚拟模型，然后建立上传后端、确定中转缓冲大小；接着把上下文里的张量收集成列。

不走 mmap 时先做一次稳定排序：**需要中转的排前面、大的排前面** —— 让最大的暂存缓冲在权重驻留最少的时候被分配，同时保证同一时刻只有一份暂存活着。

然后是主循环：对每个张量取出它的权重记录，用一个判据 `from_mapping` 分成两支。注意这一层循环**不做任何后端选择**：张量早就建在按 buffer type 分的 ggml context 里了。

<!-- src: src/llama-model-loader.cpp -->
```c
bool llama_model_loader::load_all_data(
        struct ggml_context * ctx,
        llama_buf_map & bufs,
        llama_mlocks * lmlocks,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    if (files.empty()) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            set_tensor_data(t, set_tensor_data_ud);
        }
        return true;
    }
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");
//>> ---- src/llama-model-loader.cpp:1607-1623 ----
    std::vector<ggml_tensor *> tensors;
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        tensors.push_back(cur);
    }

    // without mmap, tensors in non-host buffers are staged through a temporary buffer sized like the tensor
    // load them biggest-first so the largest staging buffer is allocated while the fewest weights are resident
    if (!use_mmap) {
        std::stable_sort(tensors.begin(), tensors.end(), [](const ggml_tensor * a, const ggml_tensor * b) {
            const bool staged_a = a->buffer && !ggml_backend_buffer_is_host(a->buffer);
            const bool staged_b = b->buffer && !ggml_backend_buffer_is_host(b->buffer);
            if (staged_a != staged_b) {
                return staged_a;
            }
            return staged_a && ggml_nbytes(a) > ggml_nbytes(b);
        });
    }
//>> ---- src/llama-model-loader.cpp:1625-1671 ----
    for (struct ggml_tensor * cur : tensors) {
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        const bool from_mapping = use_mmap || lazy.has(cur);

        if (from_mapping) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs.count(weight->idx)) {
                buf_mmap = bufs.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);

                // locking a lazy tensor would fault all of it in, which is what lazy avoids
                if (lmlocks && !lazy.has(cur)) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
```

## 九、非 mmap：三条搬运路径

不是映射的字节只有三种去处：已经住在 host buffer 里就直接读进去；住在设备上且后端支持异步上传，就先读进 pinned host buffer 再 `set_async`（读下一块与上传上一块重叠）；两者都不满足就现开一个临时缓冲读完再 `set` 过去。

<!-- src: src/llama-model-loader.cpp -->
```c
        } else {
            const auto & file = files.at(weight->idx);

            if (ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (upload_backend) {
                    size_t offset = weight->offs;
                    alignment = file->read_alignment();
                    size_t aligned_offset = offset & ~(alignment - 1);
                    size_t offset_from_alignment = offset - aligned_offset;
                    file->seek(aligned_offset, SEEK_SET);

                    // Calculate aligned read boundaries
                    size_t read_start = aligned_offset;
                    size_t read_end = (offset + n_size + alignment - 1) & ~(alignment - 1);

                    size_t bytes_read = 0;
                    size_t data_read = 0;  // Actual tensor data copied (excluding padding)

                    while (bytes_read < read_end - read_start) {
                        size_t read_size = std::min<size_t>(buffer_size, read_end - read_start - bytes_read);

                        // Align the destination pointer within the pinned buffer
                        uintptr_t ptr_dest_aligned = (reinterpret_cast<uintptr_t>(host_ptrs[buffer_idx]) + alignment - 1) & ~(alignment - 1);

                        // Wait for previous upload to complete before reusing buffer
                        ggml_backend_event_synchronize(events[buffer_idx]);

                        // Read aligned chunk from file
                        file->read_raw_unsafe(reinterpret_cast<void *>(ptr_dest_aligned), read_size);

                        // Calculate actual data portion (excluding alignment padding)
                        uintptr_t ptr_data = ptr_dest_aligned;
                        size_t data_to_copy = read_size;

                        // Skip alignment padding at start of first chunk
                        if (bytes_read == 0) {
                            ptr_data += offset_from_alignment;
                            data_to_copy -= offset_from_alignment;
                        }

                        // Trim alignment padding at end of last chunk
                        if (aligned_offset + bytes_read + read_size > offset + n_size) {
                            data_to_copy -= (read_end - (offset + n_size));
                        }

                        // Async upload actual data to GPU
                        ggml_backend_tensor_set_async(upload_backend, cur,
                                                      reinterpret_cast<void *>(ptr_data), data_read, data_to_copy);
                        ggml_backend_event_record(events[buffer_idx], upload_backend);

                        data_read += data_to_copy;
                        bytes_read += read_size;

                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    // scoped to one tensor so only one staging buffer is alive at a time
                    std::vector<no_init<uint8_t>> read_buf(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }
            }
```

## 十、★ buffer type 的判据：一个有形状、没有数据的假算子

这是全课最需要看清的一段。`select_weight_buft()` 本身只是"按顺序试"，真正的判据是 `weight_buft_supported()`：它按权重的算子类型（`MUL_MAT` / `GET_ROWS` / `ROPE` / `SSM_CONV` ...）**造一个假算子**，给权重挂一个 **0 字节哑 buffer**，然后问后端 `ggml_backend_dev_supports_op()`。

也就是说：判据是"这个后端用这个 buffer type 跑这个算子行不行"，而不是"数据放在哪"。正因如此，选择可以在**没有任何数据**的情况下、在建图阶段完成。

<!-- src: src/llama-model-loader.cpp -->
```c
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }
//>> ---- src/llama-model-loader.cpp:1056-1078 ----
    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t * buft_list) {
    GGML_ASSERT(!buft_list->empty());
    for (const auto & cur : *buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}
//>> ---- src/llama-model-loader.cpp:1215-1230 ----
        // select the buffer type for this tensor
        const buft_list_t * buft_list;
        switch (info.layer) {
            case LLM_TENSOR_LAYER_INPUT:
                buft_list = buft_list_input;
                break;
            case LLM_TENSOR_LAYER_OUTPUT:
                buft_list = buft_list_output;
                break;
            case LLM_TENSOR_LAYER_REPEATING:
                GGML_ASSERT(buft_list_layer != nullptr);
                buft_list = buft_list_layer;
                break;
            default:
                GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
        }
//>> ---- src/llama-model-loader.cpp:1262-1289 ----
        if (!buft) {
            buft = select_weight_buft(hparams, t_meta, op, buft_list);
            if (!buft) {
                throw std::runtime_error(format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
            }
        }

        // avoid using a host buffer when using mmap
        auto * buft_dev = ggml_backend_buft_get_device(buft);
        if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
            auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!cpu_dev) {
                throw std::runtime_error("no CPU backend found");
            }
            buft = ggml_backend_dev_buffer_type(cpu_dev);
        }

        if (buft != buft_list->front().second) {
            if (n_tensors_moved == 0) {
                first_tensor_moved_name = t_meta->name;
                first_tensor_moved_type_name = ggml_type_name(t_meta->type);
                first_moved_from_buft = buft_list->front().second;
                first_moved_to_buft   = buft;
            }
            n_tensors_moved++;
        }

        return buft;
```

## 十一、llama-io：同一工程里的另一套 I/O

`llama_io_write_i` / `llama_io_read_i` 是**状态序列化**的抽象：5 个纯虚函数（裸字节、张量区间、计数），读侧写侧对称。`llama-io.cpp` 全文只实现一件事 —— 带 `uint32_t` 长度前缀的字符串，说明这是一条自描述的字节流。

它和权重加载没有交集：权重的特点是**只读、一次、巨大**，不需要虚接口；状态（KV cache、序列状态）的特点是**反复存取、可以只取一部分**，所以才需要抽象。具体实现类（`llama_io_write_host` / `_file` / `_device` / `_dummy`）在 `src/llama-context.cpp`，由 L2-07 覆盖。

<!-- src: src/llama-io.h -->
```c
class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
```

## 十二、llama-io.cpp：全文只有两个函数

这套接口里唯一的非虚实现就是 `write_string` / `read_string`：先写 `uint32_t` 长度，再写字节；读侧反过来。派生类只需要实现裸字节的 `write` / `read`，字符串与"张量的某一区间"由基类组合出来。

这很能说明它的定位：**它假设自己是一条可以自描述的字节流** —— 所以能边写边报长度（`n_bytes()`）、能按张量区间分片，也就支持"只存某几个序列"这种用法。权重加载完全不需要这些性质。

<!-- src: src/llama-io.cpp -->
```c
void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}
```

## 十三、mmap 的延伸：按需读行（结构）

映射打开了一扇门：既然数据已经有地址，就可以**不把整张量读进来**，用到哪几行读哪几行。`lazy_read` 把要懒读的张量名、以及"每个文件里哪些区间要留随机访问"记下来，交给映射层的 `ranges`（见第三幕的 `unmap_fragment` 与 `lazy_ranges`）。

<!-- src: src/llama-model-loader.h -->
```c
    // handle TENSOR_READ_LAZY
    // use case: keep PLE / engrams embd tensors on disk, read them on demand
    struct lazy_read {
        // set by the caller before the create_tensor() calls
        enum llama_lazy_mode mode = LLAMA_LAZY_MODE_OFF;

        // decide whether this tensor is read lazily
        // pass w to also record it, or nullptr to only ask
        bool add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w);

        bool any() const {
            return !ranges.empty();
        }

        bool has(const ggml_tensor * t) const {
            return tensors.count(ggml_get_name(t)) > 0;
        }

        const llama_mmap::ranges & for_file(uint32_t idx) const {
            static const llama_mmap::ranges none;

            const auto it = ranges.find(idx);
            return it == ranges.end() ? none : it->second;
        }

        // lazy tensors are gathered on the host, so no offload setting applies to them
        static ggml_backend_buffer_type_t buft();

    private:
        std::map<uint32_t, llama_mmap::ranges> ranges;
        std::set<std::string>                  tensors;
    } lazy;
```

## 十四、懒读的开关与代价

`lazy_read::add()` 有三道门：模式为 `OFF` 直接拒绝；非 `ON` 模式下只对超过 4 GiB 的张量自动开启；`llama_mmap::SUPPORTED` 为假时明确拒绝并打印警告 —— 因为懒读目前**依赖 mmap**。

<!-- src: src/llama-model-loader.cpp -->
```c
}

bool llama_model_loader::lazy_read::add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w) {
    if (mode == LLAMA_LAZY_MODE_OFF) {
        return false;
    }

    // do not lazy-read small tensors, it has significant overhead and is not worth it
    constexpr size_t auto_min_size = 4ull * 1024 * 1024 * 1024;
    if (mode != LLAMA_LAZY_MODE_ON && ggml_nbytes(t) <= auto_min_size) {
        return false;
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not available, so tensor %s (size = %zu MiB) is loaded into RAM in full\n",
                __func__, name.c_str(), ggml_nbytes(t)/1024/1024);
        return false;
    }

    if (w) {
        ranges[w->idx].emplace_back(w->offs, w->offs + ggml_nbytes(t));
        tensors.insert(name);

        LLAMA_LOG_INFO("%s: tensor %s (size = %zu MiB) lazy read enabled\n",
                __func__, name.c_str(), ggml_nbytes(t)/1024/1024);
    }

    return true;
}
```

---

## 说明

- 本课引用 `src/llama-model-loader.{h,cpp}`、`src/llama-mmap.{h,cpp}`、`src/llama-io.{h,cpp}` 共 6 个源文件，全部计入覆盖率。
- 文中提到的调用方（`src/llama-model.cpp`：`init_mappings()` / `load_all_data()` 的调用点、`use_mlock` 的判定、`mlock_mmaps` 的构造）属于 L2-05 的覆盖范围；`src/llama-context.cpp`（llama-io 的实现类）属于 L2-07。本课不引用这两个文件的代码，故不计入本课覆盖率。
- 文中出现的命令行开关 `--load-mode` 由真实二进制的 `llama-cli --help` 实测确认存在。
