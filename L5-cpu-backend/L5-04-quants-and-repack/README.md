# L5-04 · ★ 量化与 repack — 课件说明

> 层：**L5 · CPU 后端执行** ｜ 前置课：`L5-03`（★ 向量化基础设施）

## 学习目标

看完这一课，你应该能：

1. 说清 repack 与 `GGML_TYPE` 的分工：类型号管"块多大、怎么反量化"，后端管"块与块之间怎么摆"（对应验收点）；
2. 解释交错布局为什么能加速点积：偏移 `k·N·G + j·G + i` 让"行"变成向量载入的一个维度；
3. 说出 `block_q4_0x4` 与 `block_q4_0` 的字节数关系（72 = 4 x 18），以及 `xor 0x88` 在重排时顺手做掉了什么；
4. 说明 traits 的两层钩子（`extra_buffer_type` 认领 / `tensor_traits` 执行）在一个 op 上如何先后生效；
5. 按 ISA 与 `ne[1]` 门槛，判断一个 Q4_0 张量会选中 8x8 / 4x8 / 4x4 / 16x1 中的哪一个。

## 覆盖的源文件（6 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml-cpu/quants.h` | 107 |
| `ggml/src/ggml-cpu/quants.c` | 1340 |
| `ggml/src/ggml-cpu/repack.h` | 259 |
| `ggml/src/ggml-cpu/repack.cpp` | 5254 |
| `ggml/src/ggml-cpu/traits.h` | 39 |
| `ggml/src/ggml-cpu/traits.cpp` | 37 |

> **说明**：本课只引用 `ggml/src/ggml-cpu/` 下的 6 个文件（quants.c / quants.h / repack.cpp / repack.h / traits.cpp / traits.h），全部计入覆盖率。
> **说明**：文中为了定位而提到的 `ggml-common.h`（L1-04）、`ggml-cpu.c`（L5-01）、`arch/x86/repack.cpp` 与 `arch-fallback.h`（L5-05）、`src/llama-model.cpp`（L2 系列）、`common/arg.cpp` 均**不引用其源码**，故不计入本课覆盖率。
> **说明**：`QK4_0 = 32`、`QK8_0 = 32` 的定义在 `ggml-common.h:194,251`；本课引用的 `repack.cpp:136` 有 `assert(QK8_0 == 32)` 可交叉验证。
> **说明**：第 3 幕的字节条按 `repack.h:26-46` 的结构体字段与 `static_assert` 画出：`d[4]` = 4 x 2 字节，`qs` = `QK8_0 * 2` = 64 字节，合计 72 = 4 x 18。

## 场景（9 幕）

1. **同一份权重，两种排法：类型号固定，布局可换** — 一个 Q4_0 张量既可以"每行每 32 个权重一块"顺序存，也可以"4 行交错"存；内核跟着换。
2. **基线：标量点积要先拆 nibble** — 原始 Q4_0 里一个字节管相隔 16 的两个元素 —— 载入之后还得 mask / shift / 减 8 才能变成 int8。
3. **★ repack：交错布局从类型号里搬进后端** — block：一个块里放 N 行的 scale 和 N 行的 quants —— 字节总数不变，只是换了顺序。
4. **make_block_q4_0x4：4 行轮流搬，顺手异或掉偏置** — src_id = i % 4、src_offset = (i/4) x 4、dst_offset = i x 4 —— 每次只搬 4 字节，轮到下一行。
5. **交错是双边的：激活也要按 4 行一组量化** — 权重在加载时重排一次；激活每次前向都要重排 —— 两侧必须用同一个交错粒度才配得上。
6. **★ 交错块怎么读：偏移 = k·N·G + j·G + i** — k 走列、j 走行、i 走行内字节（本例 N = G = 4）—— 一条向量载入就同时拿到多行的同一批列。
7. **traits：把 op 从默认路径接过来的钩子** — tensor_traits 只回答两件事：要多大 work 空间、这个 op 我来算；extra_buffer_type 负责认领。
8. **变体选择：ISA + 形状 决定 4x4 / 4x8 / 8x8 / 16x1** — AVX2/SVE -> 8x8，NEON+i8mm -> 4x8，NEON+dotprod -> 4x4，VXE -> 4x4，RVV 256-bit -> 16x1；都不满足返回 nullptr。
9. **谁先认领这个 op —— 与"变体 -> SIMD 宽度"总表** — 每个 op 进入 CPU 后端，先遍历 extra buffer types；认领成功就走 repack 内核，否则回到默认实现。

## 核心结论

### ★ repack = 把交错布局从类型号里搬进后端

`block<K,N>` 是编译期模板，`block_q4_0x4 / x8 / x16` 只是 `N = 4 / 8 / 16` 的别名。7 条 `static_assert` 钉死一件事：**重排前后字节总数完全相等**（`4 x 18 = 72`）。

这与 L1-01 里提过的旧式 `GGML_TYPE_Q4_0_4_4` 形成对照：交错曾经是类型号的一部分（`ggml.h:421` 注明 support has been removed），现在它是后端缓冲类型的属性 —— 类型号保持"纯数据格式"，性能布局交给后端。

### ★ 为什么交错能加速点积

原布局里一个字节管相隔 16 的两个元素，载入后必须 `& 0x0F` / `>> 4` / `- 8`；交错布局把同一批列在 N 行上的数据排成连续地址：

```text
权重的第 j 行第 i 个字节：偏移 = k * N * G + j * G + i
（k 走列、j 走行、G = 交错粒度；4x4 变体里 N = G = 4）
```

于是**一条向量载入同时覆盖多行的同一批列**，`j` 每加 1 只走 G 字节；4x4 的 gemm 一次算出 `sumf[4][4]` 共 16 个部分和。重排时顺手 `^= 0x88`，把"减 8"也提前做掉了，内核只需 `<< 4` / `& 0xF0` 与一次 `>> 4` 收尾。

### 交错是双边契约，粒度必须一致

权重在**加载时**重排一次（repack 缓冲的 set_tensor），激活在**每次前向**重排（`ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>`，4 行一组写进 `params->wdata`）。只有两侧的 `blck_size_interleave` 相同，gemm 才能用同一条乘加同时消费 4 行权重与 4 个 token；`ne11` 不是 4 的倍数时，余下的行退回 `from_float` + gemv。

### 变体选择 = 类型号 + ISA + 形状

`ggml_repack_get_optimal_repack_type` 对每个类型号按顺序试分支，`return` 出去的那个实例就是本次运行的 kernel 变体；都不满足则 `return nullptr`，张量留在普通缓冲走默认 `MUL_MAT`。

| 类型 | 变体（触发 ISA） | 依据行 |
|---|---|---|
| Q4_0 | 8x8（AVX2/SVE+i8mm）· 4x8（NEON+i8mm）· 4x4（NEON+dotprod / VXE）· 16x1（RVV 256）| 4973-5005 |
| Q4_K | 8x8（AVX2 或 NEON+i8mm）· 8x4（NEON+dotprod）· 16x1（RVV）| 5006-5032 |
| Q2_K | 8x8（AVX512）· 16x1（RVV）| 5033-5049 |
| Q5_K / Q6_K | 8x8 或 8x4（只有 NEON 两条路）| 5050-5071 |
| IQ4_NL / MXFP4 | 8x8（AVX2）· 4x4（NEON+dotprod）· 16x1（仅 IQ4_NL，RVV）| 5072-5104 |
| Q8_0 / Q1_0 | 4x8（NEON+i8mm）· 4x4（NEON+dotprod）· 16x1（仅 Q8_0，RVV）| 5105-5139 |

如果 ISA 与形状都不满足，第 8 幕那句 `return nullptr` 就是答案 —— **repack 是可选项，不是必需项**。

## 验收点

- [x] 保真门禁：18 处引用 —— 18 个引用块 / 26 个连续段逐字命中
- [x] 覆盖度门禁：本课声明 6 项，无空课、无幻影；全局覆盖 1116/1290
- [x] 参数门禁：真值集 381 长 / 76 短选项，扫描 3 文件 3 处引用，0 处非法
- [x] 语法检查：0 错误 / 0 警告
- [x] 渲染门禁：9 幕 —— 1 页面 x 2 分辨率，0 错误 / 0 溢出

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
