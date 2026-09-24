<!-- llama-coverage
ggml/src/ggml-metal/ggml-metal-fusion.h
ggml/src/ggml-metal/ggml-metal.cpp
ggml/src/ggml-metal/ggml-metal-common.cpp
ggml/src/ggml-metal/ggml-metal-context.m
ggml/src/ggml-metal/ggml-metal-ops.cpp
ggml/src/ggml-metal/ggml-metal-fusion.cpp
ggml/src/ggml-metal/ggml-metal-device.cpp
ggml/src/ggml-metal/ggml-metal-common.h
ggml/src/ggml-metal/ggml-metal-context.h
ggml/src/ggml-metal/ggml-metal-ops.h
ggml/src/ggml-metal/ggml-metal-device.h
ggml/src/ggml-metal/ggml-metal-device.m
ggml/src/ggml-metal/ggml-metal-impl.h
ggml/src/ggml-metal/ggml-metal-tuning.h
ggml/src/ggml-metal/ggml-metal-tuning.cpp
ggml/include/ggml-metal.h
-->

# L6-08 · Metal 后端：主机端与图融合 — 源文件

**一句话**：Metal 后端不是"一个 op 一个 kernel"。它在**图已经建好、但还没有被编码成 Metal 命令**的那个窗口里，把连续几个 ggml 节点合并成**一次** kernel 调用 —— 这件事完全发生在后端内部，前端不感知，图上的节点也一个都不少。

本课覆盖 `ggml/src/ggml-metal/` 下 **15 个主机端源文件**（`.cpp` / `.h` / `.m`）加上对外头 `ggml/include/ggml-metal.h`，共 **16 个**。`.metal` 内核源码（`kernels/` 目录）属于 L6-09 的覆盖域，本课不引用，只在指路时提到内核名。

---

## 一、课程地图：16 个文件各管一段

本课只覆盖 **主机端**（host side）—— 也就是跑在 CPU 上、负责"决定调哪个内核、怎么调"的那部分代码。GPU 上跑的 `.metal` 内核源码属于 L6-09。

| 文件 | 行数 | 在本课里的角色 |
|---|---|---|
| `ggml-metal-fusion.h` | 98 | 融合的模式枚举、9 个 fusion id、两期共用的函数声明 |
| `ggml-metal-fusion.cpp` | 1119 | 26 条表项的表、匹配算法、模式 check、内存别名检查 |
| `ggml-metal-common.h` | 57 | `ggml_graph_optimize()` 与内存范围对象的声明 |
| `ggml-metal-common.cpp` | 500 | 为融合而打包、重排提并发、解包 |
| `ggml-metal-context.h` | 42 | 后端上下文：两个图入口（compute / optimize） |
| `ggml-metal-context.m` | 787 | 命令缓冲、编码线程划分、逐节点编码循环 |
| `ggml-metal-ops.h` | 110 | 编码器接口：`ggml_metal_op_encode()` 返回消费的节点数 |
| `ggml-metal-ops.cpp` | 5757 | 每个 op 的编码器；命中融合后改参数、跳节点 |
| `ggml-metal-device.h` | 366 | 设备 / 库 / pipeline 抽象 |
| `ggml-metal-device.cpp` | 2576 | pipeline 名拼接与缓存（融合后取哪个 kernel） |
| `ggml-metal-device.m` | 2504 | Objective-C 侧：MTLDevice、pipeline 编译、融合开关 |
| `ggml-metal-impl.h` | 1382 | kernel 参数 ABI；`nef1[3]` 等是融合专用槽位 |
| `ggml-metal-tuning.h` | 75 | 调参表结构（FA vec 的 Q / NE 配置） |
| `ggml-metal-tuning.cpp` | 1172 | 调参表数据与查表顺序 —— 与融合正交的另一类主机端决策 |
| `ggml-metal.cpp` | 1053 | 后端注册与接口实现：融合的两个触发入口 |
| `ggml/include/ggml-metal.h` | 61 | 对外 API —— 里面**没有**任何融合声明 |

读法建议：先看 `ggml-metal-fusion.h` 的 6 行头注释（第 1 幕），那就已经是全课的提纲；再看 `ggml-metal-fusion.cpp` 里那张表（第 7 幕）；最后回到两个触发点（第 4、5 幕）。

## 二、`ggml_graph_optimize()` 是通用的，融合只是它的第一个"用户"

真正被调度器调用的钩子是后端的 `.graph_optimize`。Metal 侧把它接到 `ggml_graph_optimize()`，而这个函数自己声明是**通用实现**：注释说"this implementation is generic and not specific to metal"。它必须尊重融合 —— 因为融合要求节点相邻，而重排会把节点搬走。

<!-- src: ggml/src/ggml-metal/ggml-metal-common.h -->
```c
// reorder the nodes in the graph to improve concurrency, while respecting fusion
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf);
```

## 三、★ 重排之后一定要解包：融合是临时的，不是永久的

这是本课最容易被误解的一处。优化阶段确实"把节点打包了"，但打包只是一次**局部的重排视图**；重排一结束，代码立刻 `// unfuse`，把节点按新顺序**原样摊回去**。

所以：**图从头到尾没有被修改过**。真正"融合"的是编码阶段的动作，而不是图的结构。这也解释了为什么前端、调度器、其它后端都不需要知道融合的存在。

<!-- src: ggml/src/ggml-metal/ggml-metal-common.cpp -->
```c
    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
```

## 四、表项长什么样：`{ id, ops_all, outs, unsafe, check }`

`ops` 是**过滤掉空节点**（`NONE` / `RESHAPE` / `TRANSPOSE` / `VIEW` / `PERMUTE`）之后的序列，`ops_all` 是含空节点的原始序列 —— 两者都要，因为匹配走 `ops`、打包和分配依赖走 `ops_all`。

`unsafe` 的含义写在字段注释里：为真时，通用的"链式 + 同形 + `ggml_can_fuse_subgraph`"检查全部跳过，`check` 回调成为**唯一**的裁判。`GDN_CACHE` 就是这种情况 —— 它不是"消除中间节点"式的融合，而是"把快照直接写进循环缓存、顺手跳过那次 CPY"的**写穿**式融合。

<!-- src: ggml/src/ggml-metal/ggml-metal-fusion.cpp -->
```c
struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                  const struct ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct ggml_metal_fusion   * fusion,
                          const struct ggml_tensor * const * nodes,
                          const struct ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};
```

## 五、后端上下文只暴露两个图入口

`ggml_metal_context` 是"这个后端的一个会话"。它对外只有两个与图有关的函数：执行与优化。融合的两个触发点，正好一一对应这两个入口。

<!-- src: ggml/src/ggml-metal/ggml-metal-context.h -->
```c
//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);
```

## 六、编码器接口：返回值就是"我消费了几个节点"

`ggml_metal_op_encode()` 的返回值不是成功/失败，而是**消费掉的节点数**。这个设计让融合可以完全藏在编码器内部：调用者只需要 `idx += res - 1`。文件顶部还前向声明了 `struct ggml_metal_fusion`，并且上下文里带着一个 `ggml_metal_fusion_info *`（共享的融合统计/开关）。

<!-- src: ggml/src/ggml-metal/ggml-metal-ops.h -->
```c
typedef struct ggml_metal_op * ggml_metal_op_t;

struct ggml_metal_fusion; // forward decl (ggml-metal-device.h)

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        struct ggml_cgraph * gf,
        struct ggml_metal_fusion_info * finfo,
        int  idx_start,
        int  idx_end,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph);

void ggml_metal_op_free(ggml_metal_op_t ctx);

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx);

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx);
```

## 七、设备 / 库 / pipeline：融合后的 kernel 从哪来

`ggml_metal_library_*` 是 MTLLibrary 的包装：先按名字查缓存，查不到才编译。所有 `ggml_metal_library_get_pipeline_*` 都是"拼名字 -> 查缓存 -> 必要时编译"这一个套路；融合 kernel 不特殊，只是名字里多带了融合参数（下面第九节）。

<!-- src: ggml/src/ggml-metal/ggml-metal-device.h -->
```c
//
// MTLLibrary wrapper
//

typedef struct ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose);

void ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, const char * name);
struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, enum ggml_op op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, enum ggml_type tsrc, enum ggml_type tdst);
```

## 八、谁有权关掉融合：设备初始化时的两个环境变量

`ggml_metal_fusion_info` 由**设备**持有（不是每个上下文一份），这样同一设备上的多个后端上下文共享同一份开关与计数，统计才不会打架。开关来自 `GGML_METAL_FUSION_DISABLE`，调试级别来自 `GGML_METAL_FUSION_DEBUG`（级别大于 0 时才会收集统计）。

注意这是**环境变量**，不是命令行参数 —— 本课不涉及任何 CLI 开关。

<!-- src: ggml/src/ggml-metal/ggml-metal-device.m -->
```c
                {
                    const char * val = getenv("GGML_METAL_FUSION_DEBUG");
                    dev->finfo = ggml_metal_fusion_info_init(
                            getenv("GGML_METAL_FUSION_DISABLE") == nil,
                            val ? atoi(val) : 0);
                }
```

## 九、融合 kernel 的 ABI：多出来的那两组槽位

对比一个普通 `RMS_NORM` 与融合三跳的版本：参数结构体里除了 `ne00` / `nb1..nb3` / `eps`，还多了 `nef1[3]`…`nbf3[3]` 六组槽位（每组 3 个，对应第 1、2 跳的额外输入在各维上的元素数与字节步长），以及一个 `scale`。第 1 跳填 `*f1`，第 2 跳填 `*f2`，以此类推 —— 这就是"一个 kernel 干 N 跳"在参数层面的代价。

<!-- src: ggml/src/ggml-metal/ggml-metal-impl.h -->
```c
typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
    float    scale;
} ggml_metal_kargs_norm;
```

## 十、`ADD` 链：同一个 kernel，靠 pipeline 名里的 `nf=` 区分跳数

ADD 链是"参数化融合"的最好例子：只有**一个** `kernel_bin_fuse_*` 内核，它把 `src1` 当成一排首尾相接的加数，用 `args.o1[i]` 给出第 i 跳的偏移。跳数 `n_fuse` 不进 kernel 参数，而是进 **pipeline 名**（`nf=%d`），因此不同跳数会编译出不同的特化版本。名字里还带 `op=`（0=ADD / 1=SUB / 2=MUL / 3=DIV）、`rb=`（行广播）与 `cb=`（列广播）三个开关。

<!-- src: ggml/src/ggml-metal/ggml-metal-device.cpp -->
```c
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, const ggml_tensor * op, int32_t n_fuse) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t1_str = ggml_type_name(op->src[1]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    const bool is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    const bool is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", base, op_num, n_fuse, is_rb, is_cb);
```

## 十一、与融合正交的另一类主机端决策：调参表

Metal 主机端不只有"融合"一种决策。`ggml_metal_tuning` 维护的是另一类：给定 GPU 家族、数据类型、头维度 `dk`/`dv` 与序列长度，该选哪个 **flash-attention vec 内核配置**（`Q` 与 `NE`）。它和融合毫无关系 —— 融合决定"几个 op 合成一个",调参决定"这一个 kernel 用哪套展开参数"。

<!-- src: ggml/src/ggml-metal/ggml-metal-tuning.h -->
```c
#include <cstdint>
#include <vector>

namespace ggml_metal_tuning {

// FA vec selection buckets. ne01 (query rows) splits decode (==1) from batch (>=2), the
// batch side refined into {2,3,4,5}: Q>1 reuses one K/V load across rows, so it only pays
// off once ne01 aligns with Q. ne11 (KV length) is bucketed too, as the Q>1 crossover is
// head-size dependent (small dk crosses late, large dk wins even at short KV).
constexpr int FA_VEC_NE11_BUCKETS[] = { 1024, 4096, 16384 };
constexpr int FA_VEC_NE01_BUCKETS[] = { 2, 3, 4, 5 };

int fa_vec_ne11_bucket(int64_t ne11);
int fa_vec_ne01_bucket(int64_t ne01);

```

## 十二、调参表的查法：精确桶 -> 域默认 -> 基线

查表顺序写得很清楚：先按 `(family, dtype, dk, dv, ne11_b, ne01_b)` 找精确行；找不到就把 `ne11` 折叠成"域默认"再找一次；再找不到就退回按 `(dk, dv)` 硬编码的基线。短 KV（`ne11_b == 0`）直接走基线 —— 注释说明此时注意力只占一步里很小的一块，不值得特化。

<!-- src: ggml/src/ggml-metal/ggml-metal-tuning.cpp -->
```c

    // exact bucket, then the ne01 domain default (ne11 collapsed)
    if (auto * c = find_cfg(fa_vec_tuned_table, std::size(fa_vec_tuned_table), k)) {
        return *c;
    }
    k.ne11_b = FA_VEC_NE11_DEFAULT;
    k.ne01_b = (ne01_b == 0) ? FA_VEC_DOMAIN_DECODE : FA_VEC_DOMAIN_BATCH;
    if (auto * c = find_cfg(fa_vec_tuned_table, std::size(fa_vec_tuned_table), k)) {
        return *c;
    }
```

## 十三、对外 API 里没有融合

`ggml/include/ggml-metal.h` 里只有初始化、判断、abort 回调、feature family 检查、抓取命令缓冲、以及注册后端。**没有任何 fusion 相关的东西**。这是本课结论最直接的证据：融合是后端私有的实现细节，用这个后端的人（包括 llama.cpp 的模型代码）完全看不见它。

<!-- src: ggml/include/ggml-metal.h -->
```c
//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);
```

---

## 说明

- 本课覆盖 `ggml/src/ggml-metal/` 下的 15 个主机端源文件与 `ggml/include/ggml-metal.h`，共 16 个，全部计入覆盖率。
- 本课**不引用**任何 `.metal` 内核源文件（`ggml/src/ggml-metal/kernels/` 目录）—— 它们属于 L6-09 的覆盖域。课件里出现的 kernel 名（如 `kernel_norm_mul_add_f32`）全部来自本课引用的 `ggml-metal-device.cpp`，逐字可查。
- `GGML_METAL_FUSION_DISABLE` 与 `GGML_METAL_FUSION_DEBUG` 是**环境变量**，不是命令行参数；出处是 `ggml-metal-device.m:1281-1286` 的 `getenv()` 调用。
- 源码注释不总是准确的：`ggml-metal-common.cpp:456` 的注释把文件名写成 `ggml-metal-fuse.cpp`，而仓库里实际的文件是 `ggml-metal-fusion.cpp`。本课第 4 幕按行号逐字引用了这一行（保留原样），此处特别说明。
- 与 CUDA 的对照只给行号指路（`ggml-cuda.cu:3180` 的 `ggml_cuda_can_fuse`、`:4505` 的 `ggml_backend_cuda_graph_optimize`），未逐字引用 CUDA 源码，因此 `ggml-cuda` 目录不计入本课覆盖率，留给 L6-01。
