# L1-03 · 计算图与拓扑遍历：节点怎么排成一队 — 课件说明

> 层：**L1 · 算子的表示** ｜ 前置课：`L1-01`、`L1-02`

## 学习目标

看完这一课，你应该能：

1. 说出 `ggml_build_forward_expand()` 展开出的 `nodes[]` 为什么天然是拓扑序（后序 DFS + 断言）；
2. 解释 `visited_hash_set` 如何用 `keys[] + used[]` 同时解决"去重"和"图内编号"两件事（对应验收点）；
3. 论证为什么 visited 标记**不能**打在 `ggml_tensor` 上（张量跨图复用、清标记、拿不到稳定槽位）；
4. 区分视图的两条链：`src[0]`（计算依赖，遍历会走）与 `view_src`（数据归属，遍历不走）；
5. 说出 `ggml_graph_clear()` 为什么只要重置计数器和 `used` 位图。

## 覆盖的源文件（3 个）

| 文件 | 行数 |
|---|---|
| `ggml/src/ggml.c` | 8168 |
| `ggml/src/ggml-impl.h` | 796 |
| `ggml/include/ggml-cpp.h` | 40 |

> **说明**：本课声明 3 个源文件：`ggml/src/ggml.c`（主）、`ggml/src/ggml-impl.h`（`struct ggml_cgraph` 与 `struct ggml_hash_set` 的定义与实现所在）、`ggml/include/ggml-cpp.h`（C++ RAII 包装层）。三者都在覆盖域内，均计入覆盖率。
> **说明**：`ggml_graph_plan()` / `ggml_graph_compute()` 这一版不在 `ggml.c` 里（它们属于 CPU 后端的 `ggml-cpu`）。本课只说"顺序被后端按 nodes[] 消费"，不引用其源码，具体展开见 L4-04。
> **说明**：第 7 幕提到的 GGML_DEFAULT_GRAPH_SIZE（默认图容量）定义在 `ggml/include/ggml.h`，该文件由 L1-01 覆盖；本课不引用其行号，故不计入本课声明。
> **说明**：工程备注：`tools/check_flags.py` 的 `strip_verbatim()` 会把 lesson.js 的**内容**当作路径传给 `dump_scenes.js`，node 因此把整份内容回显到 stderr，而该回显在 65536 字节处被截断（实测 `stderr 长度 = 2 x 文件字节 + 51`）。截断点落在多字节字符中间时，Python 的 `text=True` 解码会抛 `UnicodeDecodeError`，整个参数门禁失败。本课的 `visual` 因此写得紧凑，把 lesson.js 控制在 32742 字节以内。

## 场景（9 幕）

1. **从张量到图：这一课看的是遍历** — L1-01 给了节点（tensor），L1-02 给了身份（op）；本课看 ggml 如何把它们串成一张有序、可执行的图。
2. **★ 后序 DFS：nodes[] 出来就是拓扑序** — 父节点在它的所有 src[] 之后入列；源码用一条断言把这个不变量钉死。
3. **★ 每个张量只入列一次：先查表，再插入** — 同一张量被多个下游节点引用时，第二次到达必须立刻返回 —— 否则它会在 nodes[]/leafs[] 里出现两次。
4. **src[] 就是边：递归只走这一个数组** — 沿着 src[0..9] 递归；等所有输入都处理完，才决定这个张量进 leafs[] 还是 nodes[]。
5. **★ 标记位属于图，不属于张量** — struct ggml_hash_set 只有三个成员：容量、指针数组、位图。去重所需的状态全在这里。
6. **ggml_hash_find / ggml_hash_insert：线性探测** — 哈希函数只是把指针右移 4 位（对齐保证低 4 位为 0）；冲突就 +1 找下一格，用 used[] 判断占用。
7. **建图：哈希表容量翻倍，整张图是一个 arena 对象** — 源码注释直接写了为什么翻倍：it needs to hold both nodes and leafs。
8. **view_src 不是图的边：遍历不走它** — 一次 ggml_view_impl 调用同时建了两条链：src[0] 是计算依赖，view_src 是数据归属。
9. **收束：这张图归谁所有** — 左表把本课压成六行；右边源码说明这些 C 结构在 C++ 侧由 unique_ptr 托管（RAII）。

## 核心结论

### ★ 顺序：后序 DFS 让 nodes[] 直接可用

`ggml_visit_parents_graph()` 是后序遍历：先递归所有 `src[]`，再把当前张量追加进 `leafs[]` 或 `nodes[]`。所以"输入永远排在使用它的节点之前"是构造性事实，不需要事后再排序。源码用一条断言把它钉死：

```text
本次最后入列的节点 == 本次的起点张量
```

后端（L4-04）按 `nodes[0], nodes[1], ...` 顺序派发内核即可。

### ★ 去重：先查表，再插入

每个张量在图上只占一个槽位。第二次到达同一个张量时，`used` 位已经是 1，遍历走 "already visited" 分支直接返回 —— 不再入列，只在调用点把 `use_counts` 加一。但这一支并非完全空转：`compute` 为真时它仍会递归给 `src[]` 补上 `GGML_TENSOR_FLAG_COMPUTE`，因为"标记为要算"和"入列一次"是两件事。如果不去重：同一张量会进 `nodes[]`/`leafs[]` 两次、被执行两遍，`use_counts` 与梯度累加的账也会算错。

### ★ 为什么是 hash set 而不是标记位

答案不是"hash 更快"，而是**状态归谁**：

| 维度 | 张量上的标记位 | 图上的 hash set |
|---|---|---|
| 状态归属 | 属于张量，被所有图共享 | 属于 `cgraph`（`visited_hash_set`） |
| 多图复用 | 两张图互相污染 | 每张图各一份；`graph_view` 可显式共享 |
| 清空 | 必须逐个张量清 | `ggml_graph_clear()` 整块 `memset` |
| 图内编号 | 给不出 | 槽位就是 `use_counts` / `grads` / `grad_accs` 的下标 |

同一批 `ggml_tensor *` 会同时出现在多张图里（`ggml_graph_view` / `ggml_graph_dup`），而 `ggml_hash_set` 只有 3 个成员、容量取质数、冲突用线性探测 —— 结构简单，却把 visited 状态放在了正确的容器里。

### 视图：src[] 是计算链，view_src 是数据链

`ggml_view_impl()`（`ggml_view_1d/2d/3d/4d` 的共同实现）在一次调用里建两条链：

```text
ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset)   -> view_src = a, view_offs = offset
result->src[0] = a                                          -> 计算依赖
```

拓扑遍历只跟 `src[]`，所以视图在图上是个普通节点；需要"视图的数据从哪来"的地方（例如子图融合判定）必须自己沿 `view_src` 链上溯，并要求图外的来源是常量权重（见第 8 幕引用的源码）。

## 验收点

- [ ] 保真门禁：15 个引用块全部逐字来自声明的源文件，且位置连续
- [ ] 覆盖度门禁：3 个源文件均被声明
- [ ] 参数门禁：无非法命令行参数
- [ ] 语法检查：通过
- [ ] 渲染门禁：9 幕，无 JS 错误、无布局溢出、交互可用
- [ ] 自检：不看代码，能说出 `struct ggml_tensor` 里哪些字段决定一次内核调用的形状

> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，不靠肉眼。
