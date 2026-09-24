<!-- llama-coverage
ggml/include/ggml.h
-->

# L1-01 · ggml 张量：算子的数据面 — 源文件

**一句话**：在 llama.cpp 里，一个"算子"不是一个函数调用，而是**计算图上的一个节点**；而这个节点同时有**数据面**（`struct ggml_tensor`：形状、类型、数据在哪）和**身份面**（`enum ggml_op`：它是什么运算）。这一课只讲数据面。

读懂 `ggml_tensor` 是本视角的起点：后面每一课——图构建、后端切分、CPU/GPU 内核——都在读写这个结构体里的同样几个字段。

---

## 一、常量：结构体的形状被宏钉死

`ggml_tensor` 的所有数组字段都是**定长**的，长度由这几个宏给出。这不是风格问题：定长意味着结构体可以 `memcpy`、可以整体放进 arena，也意味着**任何后端内核都可以假设秩不超过 4**，不必处理任意维。

<!-- src: ggml/include/ggml.h -->
```c
#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         2048
#define GGML_MAX_SRC            10
#define GGML_MAX_N_THREADS      512
#define GGML_MAX_OP_PARAMS      64
```

## 二、★ struct ggml_tensor 逐字段

这是"算子的数据面"的全部定义。按用途分成 7 组读：

| 组 | 字段 | 含义 |
|---|---|---|
| 身份/精度 | `type` `op` `op_params` `flags` | 什么类型、什么运算、标量参数 |
| 形状 | `ne[GGML_MAX_DIMS]` | 每维元素个数 |
| 步长 | `nb[GGML_MAX_DIMS]` | 每维跨多少字节 |
| 位置 | `buffer` `data` `extra` | 数据在哪个后端缓冲 |
| 连接 | `src[GGML_MAX_SRC]` | 输入张量，即图的边 |
| 视图 | `view_src` `view_offs` | 不拥有数据时的来源与偏移 |
| 调试 | `name[GGML_MAX_NAME]` | 报错与图打印 |

注意 `op_params` 的声明方式：源码注释写明是 "allocated as int32_t for alignment"，即 16 个 `int32_t`（`GGML_MAX_OP_PARAMS / sizeof(int32_t)`）。算子的标量参数（轴号、eps、scale）都按位塞进这 64 字节，详见下面的逐字引用。

<!-- src: ggml/include/ggml.h -->
```c
    struct ggml_tensor {
        enum ggml_type type;

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };
```

## 三、★ ne[] / nb[]：三条不变量

源码把约定直接写在注释里。三条规则合起来定义了 ggml 的内存布局：

<!-- src: ggml/include/ggml.h -->
```c
        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]
```

## 四、enum ggml_type：编号即文件格式

类型枚举的**数值**会写进 GGUF 文件，所以源码给了一条硬约束：只能往末尾追加。枚举里那些被注释掉的编号（如 `GGML_TYPE_Q4_2 = 4`）说明：支持可以移除，**编号不能回收**。

<!-- src: ggml/include/ggml.h -->
```c
    // NOTE: always add types at the end of the enum to keep backward compatibility
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_Q2_0    = 42,
        GGML_TYPE_COUNT   = 43,
    };
```

## 五、三个度量函数

内核和分配器都不直接读 `type_size` 表，而是走这三个函数。注意 `ggml_type_sizef` 已经被标记 `GGML_DEPRECATED`，注释要求改用 `ggml_row_size()`。

<!-- src: ggml/include/ggml.h -->
```c
    GGML_API int64_t ggml_blck_size(enum ggml_type type);
    GGML_API size_t  ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
    GGML_API size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row
```

## 六、view_src / view_offs：视图算子

`RESHAPE` / `VIEW` / `TRANSPOSE` / `PERMUTE` 这类算子**不产生新数据**，只改变"怎么读"。源码在结构体里用两个字段表达这一点。

<!-- src: ggml/include/ggml.h -->
```c
        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;
```

---

## 说明

- 本课只引用 `ggml/include/ggml.h` 一个文件，计入覆盖率。
- 场景 4 的数值算例（Q4_0 / Q4_K 的 block size 与 type size）来自 L1-04 将要逐字展开的 `ggml/src/ggml-common.h`；本课不引用其源码，故不计入本课覆盖率。
