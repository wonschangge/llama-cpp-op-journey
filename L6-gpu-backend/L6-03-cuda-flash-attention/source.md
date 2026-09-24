<!-- llama-coverage
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/fattn-mma-f16.cuh
ggml/src/ggml-cuda/fattn-tile.cu
ggml/src/ggml-cuda/fattn-tile.cuh
ggml/src/ggml-cuda/fattn-vec.cuh
ggml/src/ggml-cuda/fattn.cu
ggml/src/ggml-cuda/fattn.cuh
ggml/src/ggml-cuda/pad.cu
ggml/src/ggml-cuda/pad.cuh
ggml/src/ggml-cuda/pad_reflect_1d.cu
ggml/src/ggml-cuda/pad_reflect_1d.cuh
ggml/src/ggml-cuda/softcap.cu
ggml/src/ggml-cuda/softcap.cuh
ggml/src/ggml-cuda/unary.cu
ggml/src/ggml-cuda/unary.cuh
-->

# L6-03 · ★ CUDA FlashAttention：三条路线与一张实例矩阵 — 源文件

**一句话**：CUDA 后端的 FlashAttention **不是一个 kernel**，而是**三条实现路线**（`fattn-vec` / `fattn-tile` / `fattn-mma-f16`）**加一张模板实例矩阵**。路线由 **head size** 与 **K/V 的 dtype** 决定；具体 shape 组合则由**编译期实例化**覆盖 ——没被实例化的组合在运行期**根本不存在**。

本课覆盖 plan 矩阵分配给 `L6-03` 的 **15 个文件**（`fattn*` / `pad*` / `unary*` / `softcap*`），全部计入覆盖率。所有引用按行号从上游 v0.5.0 抽取，行号只对该版本有效。

`template-instances/` 下的 82 个 `fattn-*.cu` 实例文件（49 vec + 21 mma + 12 tile）按 plan 矩阵归属 **L6-04**（那一课的验收点正是“template-instances 为什么必须存在”）。本课解释这张矩阵的结构与它被查表的方式，但**不把这些文件写进覆盖声明**，它们不计入本课覆盖率。

---

## 一、接口：三个函数

`fattn.cuh` 只有三行声明，但它们就是 CUDA 后端 FlashAttention 的全部对外接口：执行（`ggml_cuda_flash_attn_ext`）、能力查询（`ggml_cuda_flash_attn_ext_supported`，被 `ggml-cuda.cu` 的 `supports_op` 调用）、以及额外显存大小（`ggml_cuda_flash_attn_ext_get_alloc_size`，被显存分配器调用）。

<!-- src: ggml/src/ggml-cuda/fattn.cuh -->
```c
#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
```

## 二、入口：三条路线，三个 launcher

`ggml_cuda_flash_attn_ext()` 里没有算法，只有一个 `switch`：先问 `ggml_cuda_get_best_fattn_kernel()` 该走哪条路线，再调用对应的 launcher。枚举值 `NONE = 0 / VEC = 100 / TILE = 200 / MMA_F16 = 400` 只是标签 —— **没有任何地方比较这几个数字的大小**，路线是下面那些 `if` 分支显式挑出来的。

<!-- src: ggml/src/ggml-cuda/fattn.cu -->
```c
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}
```

## 三、第一步：head size 白名单

选择逻辑的第一段。8 个对称 head size（40 / 64 / 72 / 80 / 96 / 112 / 128 / 256）只要求 `V->ne[0] == K->ne[0]`；192 / 320 / 512 / 576 这四个（前三者 V 比 K 窄）还要求 `gqa_opt_applies`。白名单之外直接 `return BEST_FATTN_KERNEL_NONE`，于是 `ggml_cuda_flash_attn_ext_supported()` 返回 false，调度器会把节点切给别的后端（见 L4-02）。

`gqa_opt_applies` 的定义在 559-572 行：`gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0`，并且要求 Q/K/V/mask 的高维步长都是 16 的倍数。**max_bias 就是 ALiBi** —— 用了 ALiBi 就没有这个优化，不对称 head size 也就不可用。

<!-- src: ggml/src/ggml-cuda/fattn.cu -->
```c
    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }
```

## 四、★ 路线判据：can_use_vector_kernel × 设备能力

这一段是“vec 还是 mma”的答案：

| 判据 | 内容 | 出处 |
|---|---|---|
| 形状 | `Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0` | 635 行 |
| 设备 | `turing_mma_available(cc)` | 638 行 |
| batch | 未量化 K/V：Ada 上 `Q->ne[1] == 1`；量化 K/V：Ada 上 `<= 2`，更老的卡 `== 1` | 639-659 行 |

`Q->ne[0] != 192` 的原因写在上一行注释里：192 满足 “% 64 == 0”，但**没有 vec 实例**（vec 的 DKQ 与 DV 必须相同，而 192 的 V 是 128），所以强制它走 mma。

excerpt 之外还有三个分支：`volta_mma_available`（673-681 行，中等 batch 用 tile）、`amd_mfma_available` / `amd_wmma_available`（684-700 行，batch 够大才用 mma）、以及没有任何 Tensor Core 时的兜底（702-716 行：小 batch 用 vec，否则 tile）。

<!-- src: ggml/src/ggml-cuda/fattn.cu -->
```c
    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }
```

## 五、vec：形状门槛与 cols_per_block

vec 的 case 函数只做三件事：把 `Q->ne[1]` 映射成 `cols_per_block`（1 或 2）、按 `logit_softcap` 是否为 0 选两个编译期版本之一、然后 `launch_fattn<D, cols_per_block, 1>`。注意最后两个模板参数：**ncols2 = 1、stream_k = false** —— vec 既不做 GQA 方向的分块，也不沿 K/V 方向切分，所以它只适合 Q 列很少（解码）的场景。

另外 `need_f16_K / need_f16_V` 是 `type_K == GGML_TYPE_F16` 这种**逐类型判断**：只有非 f16 的 K/V 才会在 `launch_fattn` 里被转换 —— 量化类型如果编译了实例，就能直接进内核。

<!-- src: ggml/src/ggml-cuda/fattn-vec.cuh -->
```c
template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}
```

## 六、vec：49 个实例文件 × 3 个 head size

`DECL_FATTN_VEC_CASE(D, type_K, type_V)` 是显式实例化；`EXTERN_DECL_FATTN_VEC_CASES(D, type_K)` 把它对 7 种 V 类型展开一次。下面 21 行 extern 就是 3 个 head size × 7 种 K 类型，每种再展开 7 种 V 类型 —— 对应 `template-instances/` 里的 **49 个 `fattn-vec-instance-<type_K>-<type_V>.cu`**，每个文件里 3 行 `DECL_FATTN_VEC_CASE`，合计 147 份实例化。

<!-- src: ggml/src/ggml-cuda/fattn-vec.cuh -->
```c
#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)
```

## 七、tile：launcher 的 switch 覆盖全部 head size

`ggml_cuda_flash_attn_ext_tile()` 是一张 (DKQ, DV) 的查表：12 个 case，其中 192/128、320/256、576/512 是 K 宽 V 窄的不对称组合（分别对应不同模型的 head 布局）。**40 与 72 只有这条路线支持** —— mma 的 switch 里没有它们。

<!-- src: ggml/src/ggml-cuda/fattn-tile.cu -->
```c
void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
```

## 八、tile：12 组 (DKQ, DV) 实例 + 每组的调参表

`DECL_FATTN_TILE_CASE(DKQ, DV)` 声明显式实例化，12 行 extern 对应 12 个实例文件（`fattn-tile-instance-dkq192-dv128.cu` 之类）。tile 的模板参数只有 DKQ / DV 与一个 `use_logit_softcap` bool：**K/V 的 dtype 不参与模板**，非 f16 的 K/V 进来前已经被转成 f16。

<!-- src: ggml/src/ggml-cuda/fattn-tile.cuh -->
```c
#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    template void ggml_cuda_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_TILE_CASE( 40,  40);
extern DECL_FATTN_TILE_CASE( 64,  64);
extern DECL_FATTN_TILE_CASE( 72,  72);
extern DECL_FATTN_TILE_CASE( 80,  80);
extern DECL_FATTN_TILE_CASE( 96,  96);
extern DECL_FATTN_TILE_CASE(112, 112);
extern DECL_FATTN_TILE_CASE(128, 128);
extern DECL_FATTN_TILE_CASE(192, 128);
extern DECL_FATTN_TILE_CASE(256, 256);
extern DECL_FATTN_TILE_CASE(320, 256);
extern DECL_FATTN_TILE_CASE(512, 512);
extern DECL_FATTN_TILE_CASE(576, 512);
```

## 九、tile 的参数表：nthreads / occupancy / nbatch_fa / nbatch_K

tile 内核的启动参数不是运行期调出来的，而是一张 `constexpr` 表：每个 (DKQ, DV, ncols) 组合对应一组 `(nthreads, occupancy, nbatch_fa, nbatch_K)`，打包进一个 `uint32_t`（ROCm 编译器不支持在 `__launch_bounds__` 里用模板，所以用打包宏绕开）。`nbatch_fa` 是每次迭代处理多少行 KQ，`nbatch_K` 是并行加载多少列 K。

<!-- src: ggml/src/ggml-cuda/fattn-tile.cuh -->
```c
static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp16(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  64,  40)

```

## 十、mma：实例键是四元组 (DKQ, DV, ncols1, ncols2)

`DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)` 声明显式实例化；`DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)` 一次展开 5 个 ncols2 （1 / 2 / 4 / 8 / 16，ncols1 = ncols / ncols2）。`launch_fattn<DV, ncols1, ncols2>(..., true, true, true)` 的三个 true 是 `need_f16_K / need_f16_V / stream_k`：**mma 路线只吃 f16**，且大 batch 时会沿 KV 方向再切分（stream-k）。

<!-- src: ggml/src/ggml-cuda/fattn-mma-f16.cuh -->
```c
#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

```

## 十一、mma 的 sparse 变体只有 4 组

mask 在 mma 路线上还能再优化一层：把 mask 压实成“每行有哪些 K 位置要算”的索引，只对有效位置做 gather。但 `may_use_sparse` 是编译期判据，只有下面这 4 个 (DKQ, DV, ncols1, ncols2) 组合有 sparse 内核。要不要用则由 `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse()`（在 `fattn.cu` 128-152 行）在运行期按 mask/softcap/KV 长度再判断一次。

<!-- src: ggml/src/ggml-cuda/fattn-mma-f16.cuh -->
```c
static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(
        const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return (DKQ == 512 && DV == 512 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 576 && DV == 512 && ncols1 == 1 && ncols2 == 16) ||
           (DKQ == 256 && DV == 256 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 256 && DV == 256 && ncols1 == 8 && ncols2 == 8);
}

```

## 十二、实例为什么必须存在：查不到就退化成 f16

`ggml_cuda_get_fattn_vec_case()`（436-494 行）是一张运行期查表：按 `(head_size, type_K, type_V)` 依次试那些 `if constexpr (GGML_CUDA_FA_<K>_<V>)` 宏，命中就返回函数指针，全不命中返回 `nullptr`。

`ggml_cuda_flash_attn_ext_vec()` 拿到 `nullptr` 时不是报错，而是**回退到 f16-f16 实例**并打印一条只出现一次的警告。这就是 `template-instances/` 存在的理由：C++ 模板要有实例才会变成机器码，没写出来的组合在二进制里根本不存在。

构建期由 CMake 决定编哪些组合：`GGML_CUDA_FA_QUANTS` 的默认值是 `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`（`ggml/CMakeLists.txt` 207 行），`ggml/cmake/common.cmake` 的 `ggml_cuda_fattn_vec_instances()` 据此把源文件加进构建、并把 `GGML_CUDA_FA_<TYPE_K>_<TYPE_V>` 宏定义成 0 或 1。`all` 才会展开全部 7×7 = 49 个组合。

<!-- src: ggml/src/ggml-cuda/fattn.cu -->
```c
static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}
```

## 十三、mask / ALiBi / logit_softcap：launch_fattn 里的公共准备

三条路线的 launcher 最后都会调 `launch_fattn`（同文件 975 行起）。三个修饰参数就打包在节点的 `op_params` 里：scale、max_bias（ALiBi）、logit_softcap。回顾 L2-06：`build_attn()` 在 `src/llama-graph.cpp` 2643-2644 行调 `ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias, ...)`，把这三个值交给图节点 —— **契约在图上，实现在这段 C++ 里**。

另外两条相关约束：`GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16)`（1001 行），以及 `#define FATTN_KQ_STRIDE 256`（9 行）—— 后者既是 KV 方向分块的粒度，也是 635 行那个 `K->ne[1] % FATTN_KQ_STRIDE == 0` 门槛的来历。

<!-- src: ggml/src/ggml-cuda/fattn-common.cuh -->
```c
    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
```

## 十四、邻接算子：softcap 是 scale + tanh + scale 的融合

plan 矩阵按路径前缀把 `softcap*` 也分给了本课。这一族算子在 CUDA 后端里不是走常规算子分派，而是由**图融合**创建的：`ggml-cuda.cu` 的融合匹配器看到相邻的 `{GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE}`（UNARY 是 TANH）就调用 `ggml_cuda_op_softcap`。这正好对应 `build_attn()` 的非 FlashAttention 分支（`src/llama-graph.cpp` 2685-2696 行）为 logit softcap 手工发出的 scale -> tanh -> scale 三个节点 —— 有 FlashAttention 时这一步被折进内核，没有时由这个融合算子补上。

<!-- src: ggml/src/ggml-cuda/softcap.cu -->
```c
// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, ggml_nelements(src0), stream);
}
```

## 十五、softcap.cuh：块大小与声明

四行头文件：块大小宏 + 一个 op 入口。注意签名比常见的单目算子多一个 `src` 参数 —— 因为它是**三个节点融合**出来的，scale 来自前一个节点的 op_params，softcap 来自后一个节点的 op_params。

<!-- src: ggml/src/ggml-cuda/softcap.cuh -->
```c
#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);
```

## 十六、邻接算子：unary 一个模板带二十多个单目算子

`unary.cu` 把每个单目算子写成一个 `float op_xxx(float)` 的自由函数，再让 `ggml_cuda_op_unary<op>` 这一个模板负责取指针、断言类型、按 F16/F32 分派。新增一个单目算子只需要写一个 op 函数加一行包装（157 行起是那些包装）。

<!-- src: ggml/src/ggml-cuda/unary.cu -->
```c
template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}
```

## 十七、unary.cuh：块大小宏与算子清单

所有单目算子共用 256 线程的块大小（`CUDA_NEG_BLOCK_SIZE` 等一长串宏都等于 256，这是历史遗留的命名），下面逐个声明 `ggml_cuda_op_*`。这份声明清单就是“CUDA 后端支持哪些单目算子”的真值。

<!-- src: ggml/src/ggml-cuda/unary.cuh -->
```c
#pragma once
#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
```

## 十八、邻接算子：pad

`GGML_OP_PAD` 的 CUDA 实现只有 F32 一种类型：从 `op_params` 里读出 4 个维度的左右填充量 `lp0..lp3 / rp0..rp3` 与一个 `circular` 标志，然后按元素写出去。（`GGML_OP_PAD` / `GGML_OP_PAD_REFLECT_1D` 在 `ggml/include/ggml.h` 560-561 行。）

<!-- src: ggml/src/ggml-cuda/pad.cu -->
```c
void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t lp0      = ((const int32_t *) (dst->op_params))[0];
    const int32_t rp0      = ((const int32_t *) (dst->op_params))[1];
    const int32_t lp1      = ((const int32_t *) (dst->op_params))[2];
    const int32_t rp1      = ((const int32_t *) (dst->op_params))[3];
    const int32_t lp2      = ((const int32_t *) (dst->op_params))[4];
    const int32_t rp2      = ((const int32_t *) (dst->op_params))[5];
    const int32_t lp3      = ((const int32_t *) (dst->op_params))[6];
    const int32_t rp3      = ((const int32_t *) (dst->op_params))[7];
    const int32_t circular = ((const int32_t *) (dst->op_params))[8];
```

## 十九、pad.cuh：块大小与声明

五行头文件。`CUDA_PAD_BLOCK_SIZE = 256` 被下面的 kernel 启动配置使用。

<!-- src: ggml/src/ggml-cuda/pad.cuh -->
```c
#include "common.cuh"

#define CUDA_PAD_BLOCK_SIZE 256

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

## 二十、邻接算子：pad_reflect_1d

一维反射填充：只对最内层维度补 `p0` / `p1`，并且只接受 F32。源码里有一条直白的自检 —— `GGML_ASSERT(ne0 == ne00 + p0 + p1)`：填充后的长度必须等于原长加两侧填充量。

<!-- src: ggml/src/ggml-cuda/pad_reflect_1d.cu -->
```c
void ggml_cuda_op_pad_reflect_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int       p0   = opts[0];
    const int       p1   = opts[1];

    const int64_t ne00        = src0->ne[0];
    const int64_t ne01        = src0->ne[1];
    const uint3   ne01_packed = init_fastdiv_values(ne01);
    const int64_t ne02        = src0->ne[2];
    const int64_t ne03        = src0->ne[3];

    const int64_t ne0 = dst->ne[0];

    // sanity: padded length matches
    GGML_ASSERT(ne0 == ne00 + p0 + p1);
```

## 二十一、pad_reflect_1d.cuh：块大小与声明

同样五行：块大小宏 + 一个 op 入口。

<!-- src: ggml/src/ggml-cuda/pad_reflect_1d.cuh -->
```c
#include "common.cuh"

#define CUDA_PAD_REFLECT_1D_BLOCK_SIZE 256

void ggml_cuda_op_pad_reflect_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

## 二十二、实例文件长什么样（不计入本课覆盖率）

`template-instances/` 下的 fattn 实例文件一共 **82 个**：49 个 vec + 21 个 mma + 12 个 tile。它们都由 `generate_cu_files.py` 生成（文件头写明 “autogenerated, do not edit manually”），内容极短 —— 一个 `#include "../fattn-xxx.cuh"` 加上若干行 `DECL_FATTN_*`：

| 家族 | 文件数 | 文件名编码的模板参数 | 每个文件里的实例行数 |
|---|---|---|---|
| vec | 49 | `<type_K>-<type_V>`（7×7） | 3（D = 64/128/256），合计 147 |
| mma | 21 | `ncols1_<n>-ncols2_<m>` | 每个文件展开若干 (DKQ, DV)，合计 126 |
| tile | 12 | `dkq<DKQ>-dv<DV>` | 1 |

按 plan 矩阵，这些文件归属 **L6-04**（它的验收点正是“template-instances 为什么必须存在”），因此**本课不把它们写进覆盖声明，也不计入本课覆盖率**；本课只解释这张矩阵被谁查表、为什么查不到就会退化（第 8 幕 / 第十二节）。

---

## 说明

- 本课覆盖 plan 矩阵分配给 `L6-03` 的 15 个文件（`fattn*` / `pad*` / `unary*` / `softcap*`），全部计入覆盖率。
- `template-instances/` 下的 82 个 fattn 实例文件按 plan 矩阵归属 L6-04，**不计入本课覆盖率**（正文中有说明与统计表）。
- 场景与正文里引用的 `ggml/src/ggml-cuda/ggml-cuda.cu`（算子分派与 softcap 融合）、`ggml/cmake/common.cmake`、`ggml/CMakeLists.txt`、`src/llama-graph.cpp`（L2-06 的 build_attn）都只作**跨课指路**，不作覆盖声明。
