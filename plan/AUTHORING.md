# 课件撰写规范（作者必读）

本仓库是 llama.cpp 的动画课件，视角是**一个算子从模型定义到 CPU/GPU/NPU 后端执行**。
每课由**一个 spec 文件**生成四个交付文件。

> **最重要的一条**：代码引用**一律按行号从上游抽取**，绝不手抄。
> 手抄 5000 行 C/C++ 必然出错，而且这类错误靠肉眼看不出。

---

## 1. 工作目录与上游

| 项 | 值 |
|---|---|
| 课件仓库根 | `/data/WORKSPACE/llama.cpp-project/curriculum` |
| 上游源码根 | `/data/WORKSPACE/llama.cpp-project/llama.cpp`（`v0.5.0`，commit `7fe450e19305`） |
| 上游路径环境变量 | `LLAMA_UPSTREAM`（默认即上表） |

**所有引用的路径都相对上游根**，例如 `ggml/src/ggml-cpu/ops.cpp`。

---

## 2. 一课 = 一个 spec

在课件目录下写 `lesson.spec.py`，路径见 `plan/TODOLIST.md` 里该课的「目录」字段。

```
python3 <课目录>/lesson.spec.py          # 生成四个文件
python3 tools/verify_lesson.py <课目录>   # 跑全部检查并改写 README 验收点
```

`lesson.spec.py` 的骨架：

```python
#!/usr/bin/env python3
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson

SRC = 'ggml/src/ggml.c'          # 本课主要覆盖的源文件（可多个）

L = Lesson(
    id='L1-02', layer='L1 · 算子的表示',
    title='算子枚举与元数据：算子的身份',
    codecap='ggml/src/ggml.c（逐字引用）',
    nav={'prev': {'href': '../L1-01-tensor-data-plane/index.html', 'label': 'L1-01 ggml 张量'},
         'next': {'href': '../L1-03-graph-and-toposort/index.html', 'label': 'L1-03 计算图与拓扑遍历'}},
)

L.note('**一句话**：……')          # source.md 开头的引言，可多条
L.prereqs('`L1-01`')              # README 里的前置课
L.goal('……', '……')               # README 学习目标
L.conclusion('小节标题', '结论正文（可含表格）')

L.scene(kicker='L1 · 算子的表示', title='幕标题', sub='一句话说明本幕讲什么',
        caption='补充说明，常用来指路到别的课',
        src=SRC, parts=[(492, 520)],            # ★ 行号区间（1-based，含端点）
        marks=[0, 2],                            # 高亮 code 的第几行（0-based）
        notes={3: '这一行是……'},                  # 在第 3 行后插入注解行
        duration=15000,
        visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);
const host = wrap.querySelector('#cards');
const defs = [ { c: 'a', t: '标题', b: '正文', m: 'ggml_xxx()' } ];
const els = defs.map(d => { const e = U.card(d, { style: 'width:224px' }); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = ['第一步……', '第二步……', '收束……'];
defs.forEach((_, i) => tl.at(700 + i * 3200, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
''')
# …更多 scene …
# L.section('一、标题', '正文……', src=SRC, parts=[(222, 226)], lang='c')

if __name__ == '__main__':
    L.build()
```

完整可运行范例：`L1-operator-representation/L1-01-tensor-data-plane/lesson.spec.py`。

---

## 3. 硬性规则

### 3.1 代码引用

| 规则 | 说明 |
|---|---|
| **只用行号** | `parts=[(a, b)]`，由 `lessonkit` 从上游抽取。**禁止在 spec 里手打代码文本。** |
| **区间必须真实存在** | 越界会直接报错。写完先跑一遍 spec。 |
| **单段优先** | 多段拼接会把行号槽关掉（行号无法用单一数字表达），非必要不用。 |
| **注解用 `notes`** | 渲染成 `//>>` 开头的注解行，不计入逐字校验，且会把引用切成多个连续段。 |
| **示意性代码用 `text` 块** | 在 `source.md` 的散文里，如果要写"伪代码"，用 ```` ```text ```` 围栏，**不要**用 `c`/`cpp` —— 否则保真门禁会当成真实引用去校验。 |

### 3.2 绝不编造（这条最贵）

已实际发生过的编造：占位符 `(...)` 伪装成真代码、发明不存在的命令行参数、
"凭印象"写出源文件里没有的片段、打错标识符。

- **提到的每个命令行参数都必须真实存在。** 真值集来自真实二进制的 `--help`。
  要确认就跑：`cd /data/WORKSPACE/llama.cpp-project/llama.cpp && ./build/bin/llama-cli --help`
- **凡"文件里写着但没见过实际输出"的说法，必须实测确认。**
- 不能确认的数字/行为，就不要写。

### 3.3 源码注释不一定是真的

有些注释是**过时的**。凡从注释推断出的结论，要么实测，要么明确写成"源码注释声称"。

### 3.4 排版（舞台几何）

`#sceneBody` 可视区（1280x720 下实测）= **697 x 499** 像素。**所有布局按它设计**。

| 规则 | 说明 |
|---|---|
| 全局已有 `min-width: 0` | 但卡片仍要显式给宽度 |
| 一行 N 张卡 | `N x 卡宽 + (N-1) x 间距 <= 697`。常用：3 张 x 220px；4 张 x 163px；2 张 x 300px |
| 长标识符 | 交给 CSS 换行（`overflow-wrap:anywhere`）。**禁止手工 `<br>` 拆行**，会被参数门禁误判 |
| 字号 | 标题 11-15px / 正文 10px / 说明 9.5px。**克制** |
| 时间轴 | `tl.at(ms, fn)`；时间点递增，间隔 2500-3500ms；最后一个时间点 < `duration` |
| 一幕只讲一件事 | 宁可 8 幕，不要 4 幕塞满 |

### 3.5 每课结构

- **首幕给全局**：本课覆盖什么、为什么重要、与哪些课相关。
- **末幕做收束**：压成一张表 + 一句话 + 下一课指路；可放 `W.exercise(问题, 答案)`。
- 中间每幕一个概念，用 `★` 标出 1-3 个真正的洞察（写在幕标题里，用 `<span class="hl-a">` 高亮）。
- **跨课呼应要显式**："回顾 L1-01：……" / "这与 L5-04 的判据一致"。

### 3.6 可用的引擎接口

| 接口 | 用途 |
|---|---|
| `U.el(tag, attrs, ...children)` | 建元素。`attrs` 支持 `class` / `style`（**CSS 字符串**，如 `'gap:9px;width:100%'`）/ `html` / `text` / `onclick` |
| `U.card({c,t,b,m}, {style})` | 卡片。`c` ∈ `a b c d e f g`（配色），`t` 标题，`b` 正文（可含 HTML），`m` 等宽行 |
| `U.chip(text, cls)` | 小标签 |
| `U.arrow('->')` / `U.formula(html)` | 箭头 / 说明条 |
| `U.bar(label, color)` | 条形图，返回 `{el, fill, val}`；用 `fill.style.width = '60%'` 驱动 |
| `U.table(headers, rows, {monoCols:[1]})` | 表格，返回 `{el, body}`；`body.querySelectorAll('tr')` 用于逐行高亮 |
| `U.esc(s)` | HTML 转义 |
| `U.markLines(document, [3,4])` | 打开代码区第 3、4 行的行标记 |
| `W.exercise(q, a)` | 折叠式练习 |
| `tl.at(ms, fn)` | 时间轴 |

**不要**用 ES module、不要引任何外部资源、不要用 `fetch`。

---

## 4. 交付前自检

```bash
python3 <课目录>/lesson.spec.py            # 必须先跑这个
python3 tools/verify_lesson.py <课目录>     # 五项检查全绿才算完
```

`verify_lesson.py` 会把真实命令输出写进该课 `README.md` 的「验收点」。
**只要有一项不是 `[x]`，这一课就没写完。**

常见失败与对策：

| 症状 | 对策 |
|---|---|
| `保真门禁` 报某段只有前 N 行命中 | 区间里夹了不属于该处的行 —— 缩小 `parts`，或把中间行也纳入区间 |
| `参数门禁` 报某 flag 不存在 | 该参数是你编的。删掉，或跑 `--help` 确认正确写法 |
| `语法检查` 报 SCENES 求值失败 | `visual` 里有 JS 语法错误（常见：`style: 'gap':13px` 引号位置写错） |
| `渲染门禁` 报溢出 | 卡片太宽/太多。按 §3.4 重算宽度，或减卡片 |
