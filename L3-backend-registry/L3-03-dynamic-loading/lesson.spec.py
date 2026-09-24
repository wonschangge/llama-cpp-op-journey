#!/usr/bin/env python3
"""L3-03 · 动态加载后端 ggml-backend-dl —— 课件 spec。

运行：python3 L3-backend-registry/L3-03-dynamic-loading/lesson.spec.py

覆盖：
  ggml/src/ggml-backend-dl.cpp   48 行（POSIX dlopen/dlsym/dlerror + Windows LoadLibraryW）
  ggml/src/ggml-backend-dl.h     45 行（dl_handle / dl_handle_deleter / dl_handle_ptr）
  ggml/src/ggml-backend-reg.cpp  —— 只引"用到这两个原语"的几段（符号名、版本校验、注册）
  ggml/src/ggml-backend-impl.h   —— 只引"定义导出符号"的宏与 API 版本号
  ggml/src/ggml-cpu/ggml-cpu.cpp —— 只引真实后端"写一行宏"的调用点

本课的全部断言都能在前四节的逐字引用里看出来；序号引用（如 reg.cpp:237）指向的
代码块都在本课 source.md 里逐字出现过。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from lessonkit import Lesson           # noqa: E402

DL_H = 'ggml/src/ggml-backend-dl.h'
DL_C = 'ggml/src/ggml-backend-dl.cpp'
REG = 'ggml/src/ggml-backend-reg.cpp'
IMPL = 'ggml/src/ggml-backend-impl.h'
CPU = 'ggml/src/ggml-cpu/ggml-cpu.cpp'

L = Lesson(
    id='L3-03',
    layer='L3 · 后端发现与注册',
    title='动态加载后端 ggml-backend-dl',
    codecap='ggml/src/ggml-backend-dl.{h,cpp} 等（逐字引用）',
    nav={'prev': {'href': '../L3-02-registry-and-devices/index.html',
                  'label': 'L3-02 ★ 后端注册表与设备发现'},
         'next': {'href': '../L3-04-backend-features/index.html',
                  'label': 'L3-04 后端能力探测'}},
)

L.note('**一句话**：动态加载只做两件事 —— **打开动态库**、**按名字取符号**；'
       '而"必须导出哪些符号、版本要多新"这套规矩，写在注册表那一侧。')
L.note('上游这两个文件一共 93 行（`wc -l` 口径：`ggml-backend-dl.h` 45 行、'
       '`ggml-backend-dl.cpp` 48 行），本课**全文引用**它们；'
       '再把注册表里**用到这两个原语的那几段**（`ggml-backend-reg.cpp`）一并逐字引出来 —— '
       '否则"后端动态库必须导出哪些符号"这个问题，在本课的两个文件里根本找不到答案。')
L.note('**核心洞察**：动态加载把"支持哪些后端"从**编译期**推迟到**运行期**；'
       '代价是接口必须冻结、必须版本校验 —— 这正是 L3-01 那三张虚表不能随便改的原因。')

# ------------------------------------------------------------------ 第 1 幕

L.scene(
    kicker='L3 · 后端发现与注册',
    title='动态加载：把"支持哪些后端"推迟到 <span class="hl-a">运行期</span>',
    sub='上游只用 93 行做这件事：一个"打开库"的原语、一个"取符号"的原语，剩下全交给注册表。',
    caption='回顾 L3-02：静态后端在 ggml_backend_registry 的构造函数里用 #ifdef 登记（见 source.md 第十一节）；'
            '动态后端走本课这条入口，两条路最终汇入同一个注册表。',
    src=DL_H, parts=[(42, 44)], duration=16000,
    mark_src=[42, 43, 44],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">libggml-cpu.so</span><span class="arrow">-></span>
    <span class="chip a">dl_load_library</span><span class="arrow">-></span>
    <span class="chip c">dl_get_sym</span><span class="arrow">-></span>
    <span class="chip b">api_version 校验</span><span class="arrow">-></span>
    <span class="chip d">register_backend</span>
  </div>
  <div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '214px', t: '本课的两个文件', b: '两个文件一共 93 行：<br>两个原语 + 一个句柄类型。',
    m: 'dl_load_library · dl_get_sym · dl_error' },
  { c: 'c', w: '214px', t: '它们不认识"后端"', b: '文件里没有一处提到<br>CPU / CUDA / Vulkan 之类的名字。',
    m: 'void * dl_get_sym(dl_handle * handle, const char * name)' },
  { c: 'b', w: '214px', t: '规矩在注册表那一侧', b: '取哪个符号、版本要对得上、<br>失败怎么办，全在 reg.cpp 里。',
    m: '"ggml_backend_init"' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '动态加载把"支持哪些后端"从<span class="k">编译期</span>挪到<span class="k">运行期</span>：同一份二进制，配上不同目录里的库就能多出不同的后端。',
  '<span class="v">ggml-backend-dl</span> 只做两件事：<span class="k">打开库</span>、<span class="k">按名字取符号</span>。<br>它连"后端"这个概念都不认识。',
  '但"取哪个符号"不是它定的：约定写在 <span class="v">ggml-backend-reg.cpp</span> 里，两个字符串是 <span class="v">ggml_backend_init</span> 与 <span class="v">ggml_backend_score</span>。',
  '本课顺序：先看两个原语（POSIX / Windows），再看句柄归谁，最后逐字读注册表那一侧的符号名与版本校验。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(13900, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 2 幕

L.scene(
    kicker='L3-03 · 加载原语',
    title='POSIX 侧就三行：<span class="hl-a">dlopen</span> / <span class="hl-c">dlsym</span> / <span class="hl-c">dlerror</span>',
    sub='两个 flag 决定了"什么时候发现缺符号"和"符号会不会进全局命名空间"。',
    caption='RTLD_NOW 让缺失符号在打开时就暴露，这正好是注册表"加载即校验"的前提；'
            'RTLD_LOCAL 让多个后端共存时彼此不可见（对应 L3-02 的多后端设备顺序）。',
    src=DL_C, parts=[(32, 48)], duration=17000,
    mark_src=[32, 34, 35, 39, 40, 43, 44, 45],
    notes_src={35: 'RTLD_NOW | RTLD_LOCAL：前者"现在就把符号全部解析"，后者"符号不进全局命名空间"',
               40: '名字由调用方给的字符串决定 —— 拼错了只会得到 NULL，不会有编译错误',
               44: 'dlerror() 可能返回 NULL（当前没有错误），这里折成空串，调用方不必判空'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row center" id="cards" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '300px', t: 'RTLD_NOW', b: '打开时就把符号全部解析。<br>缺符号 = dlopen 直接返回 NULL，<br>而不是拖到第一次调用才失败。',
    m: 'RTLD_NOW' },
  { c: 'c', w: '300px', t: 'RTLD_LOCAL', b: '库的符号不进全局命名空间。<br>两个后端导出同名符号，<br>也不会互相顶掉。',
    m: 'RTLD_LOCAL' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  'POSIX 分支的全部实现就是三件事：<span class="v">dlopen</span> 打开、<span class="v">dlsym</span> 取符号、<span class="v">dlerror</span> 取错误文本。',
  '<span class="v">RTLD_NOW</span>：<span class="k">现在就解析</span>。缺符号等于打开失败 —— 这正是"加载即校验"的前提。',
  '<span class="v">RTLD_LOCAL</span>：<span class="k">符号不进全局命名空间</span>。多后端共存时彼此不可见，不会互相顶掉。',
  '注意返回值全是"无类型指针"：<span class="v">dl_handle *</span> 与 <span class="v">void *</span> —— 所以调用方必须按约定自己转型。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14300, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 3 幕

L.scene(
    kicker='L3-03 · 加载原语',
    title='同一抽象的另一端：<span class="hl-b">LoadLibraryW</span> / <span class="hl-c">GetProcAddress</span>',
    sub='Windows 分支还要压掉系统错误弹窗；代价是 dl_error() 在 Windows 上恒为空串。',
    caption='这正是"接口只有三个函数"的好处：平台差异被关在这 48 行里，注册表那边只看见 '
            'dl_load_library / dl_get_sym / dl_error，看不见任何 #ifdef。',
    src=DL_C, parts=[(3, 30)], duration=19000,
    mark_src=[3, 5, 7, 10, 17, 21, 28, 29],
    notes_src={6: '注释写明目的：缺 DLL 时不要弹系统对话框 —— 否则命令行工具会卡在错误弹窗上',
               10: 'LoadLibraryW 取宽字符路径：fs::path 在 Windows 上是 UTF-16',
               18: '同样的两行再出现一次：取符号这一步也做了同样的保护',
               21: 'GetProcAddress 就是 Windows 版的 dlsym',
               29: 'Windows 分支的 dl_error() 直接返回空串：调用方拿不到系统错误描述（实现现状，不是笔误）'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['动作', 'Windows 分支', 'POSIX 分支'],
  [['打开库', 'LoadLibraryW(path.wstring().c_str())', 'dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL)'],
   ['取符号', 'GetProcAddress(handle, name)', 'dlsym(handle, name)'],
   ['关闭库', 'FreeLibrary(handle)', 'dlclose(handle)'],
   ['错误文本', 'dl_error() 恒返回 ""', 'dlerror()'],
   ['额外动作', 'SetErrorMode 压掉错误弹窗', '无']],
  { monoCols: [1, 2] });
wrap.querySelector('#tbl').appendChild(t.el);
const rows = t.body.querySelectorAll('tr');

const msg = wrap.querySelector('#msg');
const texts = [
  '同一个头文件、同一组签名，实现分两份：Windows 与 POSIX。',
  '<span class="v">打开库</span>：Windows 走 <span class="k">LoadLibraryW</span>（宽字符路径）；两行 SetErrorMode 先把系统错误弹窗压掉。',
  '<span class="v">取符号</span>：<span class="k">GetProcAddress</span>；取完再把 SetErrorMode 恢复成原来的模式。',
  '<span class="v">关闭库</span>：Windows 用 FreeLibrary、POSIX 用 dlclose —— 但这两处都不在这里，而在头文件的删除器里（下一幕）。',
  '<span class="v">错误文本</span>：POSIX 有 dlerror()，Windows 分支直接返回空串。',
  '平台差异被关在这两个函数里：注册表那边完全看不见 <span class="k">#ifdef</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2200 + i * 2600, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[i + 1];
}));
tl.at(15600, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[5]; });
'''
)

# ------------------------------------------------------------------ 第 4 幕

L.scene(
    kicker='L3-03 · 句柄',
    title='<span class="hl-a">dl_handle_ptr</span>：库的寿命交给"谁持有"',
    sub='两端都把句柄包成 unique_ptr + 自定义删除器；移动语义保证"谁持有谁负责关闭"。',
    caption='回顾 L3-01：句柄与虚表是两件事 —— 句柄管"库还在不在"，虚表管"怎么调用"。',
    src=DL_H, parts=[(18, 40)], duration=17000,
    mark_src=[18, 20, 22, 23, 24, 28, 30, 32, 33, 34, 38, 40],
    notes_src={20: 'HMODULE 本身是指针类型；取出"被指向类型"，好让 dl_handle * 在两端写法一致',
               23: 'Windows 侧的关闭动作写在这个删除器里：下一行调用 FreeLibrary',
               30: 'POSIX 侧 dl_handle 就是 void —— 于是 dl_handle * 就是 void *',
               33: 'POSIX 侧的关闭动作：下一行调用 dlclose',
               40: 'unique_ptr：拷贝被删除，只能 move —— 一个库不会被关闭两次'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '214px', t: 'Windows 侧', b: 'HMODULE 本身是指针；<br>取出被指向类型，让 dl_handle *<br>在两端写法一致。',
    m: 'using dl_handle = std::remove_pointer_t<HMODULE>;' },
  { c: 'c', w: '214px', t: 'POSIX 侧', b: 'dl_handle 就是 void，<br>于是 dl_handle * 就是 void *。',
    m: 'using dl_handle = void;' },
  { c: 'b', w: '214px', t: '统一的所有权', b: 'unique_ptr 只可 move：<br>库不会被关闭两次，<br>"交给注册表"是一次明确的 move。',
    m: 'using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '两端都要有同一个类型名 <span class="v">dl_handle *</span>：Windows 用 <span class="k">remove_pointer_t</span> 把 HMODULE 的"指针性"抹平。',
  '<span class="v">dl_handle_deleter</span> 把关闭动作绑到析构：<span class="k">FreeLibrary</span> 与 <span class="k">dlclose</span> 各一处。',
  '<span class="v">unique_ptr</span> 只可 move 不可拷贝 —— 库不会被关闭两次；这也让"把库交给注册表"变成一次明确的 move。',
  '于是"库还开着吗"就变成了<span class="k">"还有谁持有这个 unique_ptr"</span>。'
];
defs.forEach((_, i) => tl.at(700 + i * 3400, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(14300, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 5 幕

L.scene(
    kicker='L3-03 · 核心',
    title='★ 动态库必须导出哪些符号？',
    sub='两个字符串：必需的是 "ggml_backend_init"；可选的是 "ggml_backend_score" —— '
        '缺了它就没法参与自动加载的择优。',
    caption='这两个字符串出现在 ggml-backend-reg.cpp:229 与 :237 —— 本课逐字引出来，不做转述。',
    src=REG, parts=[(220, 243)], duration=20000,
    mark_src=[220, 221, 222, 226, 229, 230, 234, 237, 238, 240, 242],
    notes_src={221: '第一步：打开库。失败就 return nullptr —— 不会注册任何东西',
               229: '符号一："ggml_backend_score"（可选）：返回 0 表示"本机不支持"，直接放弃这个库',
               237: '符号二："ggml_backend_init"（必需）：找不到就报错返回',
               240: '错误文本里把符号名写死了 —— 这是"必须导出哪些符号"最直接的证据'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'c', w: '214px', t: '必需：ggml_backend_init', b: '返回 ggml_backend_reg_t<br>（虚表 + api_version）。<br>找不到就记日志并放弃这个库。',
    m: '"ggml_backend_init"   reg.cpp:237' },
  { c: 'b', w: '214px', t: '可选：ggml_backend_score', b: '返回 int；0 = 本机不支持。<br>缺了它，库只能被直接点名加载，<br>无法参与"择优"。',
    m: '"ggml_backend_score"   reg.cpp:229' },
  { c: 'a', w: '214px', t: '为什么必须 extern "C"', b: 'dlsym 按"字面名字"查符号，<br>而 C++ 会对函数名做名字改编。',
    m: 'GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '加载成功后，注册表按名字取符号：<span class="k">先 score，后 init</span>。',
  '<span class="v">"ggml_backend_score"</span>：找不到也继续；但返回 0 就<span class="k">直接放弃这个库</span>（本机不支持）。',
  '<span class="v">"ggml_backend_init"</span>：<span class="k">没有它就没有后端</span> —— 报错文本里直接写着这个名字。',
  '验收点就在这里：<span class="v">必需 ggml_backend_init</span>；<span class="v">可选 ggml_backend_score</span>（自动加载的择优路径上要用它）。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(16600, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 6 幕

L.scene(
    kicker='L3-03 · 核心',
    title='★ 版本校验：<span class="hl-a">api_version</span> 必须精确相等',
    sub='不是"大于等于"，也不是兼容区间：reg->api_version 与当前版本不等就直接拒绝加载。',
    caption='回顾 L3-01：那三张虚表是 ABI 契约，api_version 就是契约的版本号 —— 虚表布局一变就必须改它。',
    src=REG, parts=[(245, 264)], duration=20000,
    mark_src=[245, 246, 249, 252, 253, 256, 259, 261, 263],
    notes_src={245: '调用刚才取到的函数指针 —— 后端库在这里交出自己的 reg',
               246: '两道检查合并成一句：reg 非 NULL，且 api_version 精确等于当前版本',
               252: '日志把两个版本号都打出来（backend: %d, current: %d）—— 排查"装的是旧库"看这一行就够',
               261: '通过校验才注册；句柄被 move 进注册表，库从此不会被卸载'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'a', w: '214px', t: '通过条件（两道）', b: 'reg 非 NULL；<br>api_version 与当前版本精确相等。',
    m: 'if (!reg || reg->api_version != GGML_BACKEND_API_VERSION)' },
  { c: 'c', w: '214px', t: '两种失败，两条日志', b: 'reg == NULL → "returned NULL"<br>版本不符 → 打出两个版本号。',
    m: 'incompatible API version (backend: %d, current: %d)' },
  { c: 'd', w: '214px', t: '为什么必须精确相等', b: 'L3-01 的三张虚表是 ABI 契约：<br>字段顺序或个数一变，<br>旧库的调用约定就错了。',
    m: '#define GGML_BACKEND_API_VERSION 2' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '拿到 <span class="v">ggml_backend_init</span> 还不够：它返回的东西要过版本关。',
  '<span class="v">reg == NULL</span> 与 <span class="v">api_version 不等</span> 被合并成同一个 if —— 两种失败都不注册。',
  '版本不符的日志会把 <span class="k">backend 的版本</span>与<span class="k">当前版本</span>都打出来，定位"装的是旧库"只需看这一行。',
  '这正是 L3-01 说的"虚表必须冻结"：<span class="v">api_version</span> 是契约的版本戳，只认精确相等。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(16600, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 7 幕

L.scene(
    kicker='L3-03 · 符号从哪来',
    title='符号不是手写的：<span class="hl-b">GGML_BACKEND_DL_IMPL</span> 宏',
    sub='后端只提供自己的注册函数；符号名与 extern "C" 由这个宏负责，而且只在 GGML_BACKEND_DL 打开时存在。',
    caption='构建开关决定符号是否存在 —— 这也是"动态加载是可选构建形态"的原因。',
    src=IMPL, parts=[(256, 286)], duration=20000,
    mark_src=[256, 258, 260, 262, 265, 267, 269, 284, 285, 286],
    notes_src={256: '整个宏块被 GGML_BACKEND_DL 包住：只有动态库构建才会展开',
               260: 'extern "C" + GGML_BACKEND_API：符号名不做名字改编，并按平台规则导出',
               265: 'score 宏与 init 宏结构相同，只是返回 int',
               285: '没有 GGML_BACKEND_DL 时宏展开为空 —— 库里不会有这两个符号'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `
  <div class="flow" style="justify-content:center">
    <span class="chip">后端源码</span><span class="arrow">-></span>
    <span class="chip b">GGML_BACKEND_DL_IMPL(reg_fn)</span><span class="arrow">-></span>
    <span class="chip c">ggml_backend_init</span>
  </div>
  <div class="row center" id="cards" style="gap:10px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'b', w: '300px', t: '宏替你写符号', b: '后端只提供自己的注册函数，<br>宏把它包成 extern "C" 的导出符号。',
    m: 'GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)' },
  { c: 'a', w: '300px', t: '不开开关就没有符号', b: 'GGML_BACKEND_DL 未定义时两个宏展开为空 ——<br>库里没有这两个符号，<br>注册表也就不会去加载它。',
    m: '#    define GGML_BACKEND_DL_IMPL(reg_fn)' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '符号不是手写的：后端写一行宏，宏把它变成 <span class="v">extern "C"</span> 的导出符号。',
  '<span class="v">GGML_BACKEND_DL_IMPL(reg_fn)</span> 生成 <span class="v">ggml_backend_init</span>，函数体就是 <span class="k">return reg_fn();</span>。',
  '<span class="v">GGML_BACKEND_DL_SCORE_IMPL(score_fn)</span> 生成 <span class="v">ggml_backend_score</span>，返回 int；L3-04 会讲它的打分公司从哪来。',
  '关键在最后四行：<span class="k">不开 GGML_BACKEND_DL，两个宏展开为空</span> —— 库里根本不存在这两个符号。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(16600, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[3]; });
'''
)

# ------------------------------------------------------------------ 第 8 幕

L.scene(
    kicker='L3-03 · 加载入口',
    title='谁在调用加载器：候选名字 + <span class="hl-c">GGML_BACKEND_PATH</span>',
    sub='启动时按写死的名字列表找库；每个名字走一次"枚举 → 打分 → 取最高分"。',
    caption='回顾 L3-02：构造函数里的 #ifdef 静态注册与本幕的候选列表，是同一个注册表的两个入口（见 source.md 第十一节）。',
    src=REG, parts=[(578, 604)], duration=19000,
    mark_src=[578, 579, 580, 585, 599, 600, 601, 602, 603],
    notes_src={579: 'Release（NDEBUG）下 silent = true：候选库不存在是常态，不打日志',
               585: '候选名字逐个尝试 —— 同一名字（如 cpu）可能有多个变体库，靠 score 择优',
               600: '最后再看环境变量 GGML_BACKEND_PATH：树外后端不必改 llama.cpp 的构建'},
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:10px;width:100%' });
wrap.innerHTML = `<div class="row wrap" id="cards" style="gap:8px"></div>
  <div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const defs = [
  { c: 'c', w: '214px', t: '候选名字写死在代码里', b: '本幕引用的 15 行 = 15 个名字：<br>blas / zendnn / cann / cuda / hip /<br>metal / rpc / sycl / vulkan / virtgpu /<br>opencl / hexagon / musa / openvino / cpu',
    m: 'ggml_backend_load_best("cpu", silent, dir_path);' },
  { c: 'a', w: '214px', t: 'silent 随构建变化', b: 'Release 下 silent = true：<br>候选库不存在属于常态，<br>不必刷屏。',
    m: '#ifdef NDEBUG' },
  { c: 'b', w: '214px', t: '树外后端的入口', b: '环境变量指向的库也会被加载：<br>第三方后端不必改<br>llama.cpp 的构建。',
    m: 'std::getenv("GGML_BACKEND_PATH")' }
];
const host = wrap.querySelector('#cards');
const els = defs.map(d => { const e = U.card(d); host.appendChild(e); return e; });
els.forEach(e => e.style.opacity = '.30');
const msg = wrap.querySelector('#msg');
const texts = [
  '启动时按写死的候选名字逐个尝试：同一名字可能有多个变体库。',
  '每个候选走一次<span class="k">枚举 → 打分 → 取最高分</span>，最后只加载最高分的那个（打分循环见 source.md 第九节）。',
  '<span class="v">silent</span> 在 Release 下为 true：找不到候选库属于正常情况，不打日志。',
  '最后再看环境变量 <span class="v">GGML_BACKEND_PATH</span>：树外后端不必改 llama.cpp 的构建就能被加载。',
  '对照 L3-02：静态后端走构造函数里的 <span class="k">#ifdef 注册</span>，动态后端走这条候选列表 —— 两条入口汇入同一个注册表。'
];
defs.forEach((_, i) => tl.at(700 + i * 3300, () => {
  els.forEach((e, k) => { e.style.opacity = k === i ? '1' : '.30'; });
  msg.innerHTML = texts[i];
}));
tl.at(16600, () => { els.forEach(e => e.style.opacity = '1'); msg.innerHTML = texts[4]; });
'''
)

# ------------------------------------------------------------------ 第 9 幕

L.scene(
    kicker='L3-03 · 收束',
    title='把这一课压成一张表',
    sub='两个符号、一个版本号、一个句柄 —— 动态加载的全部约定。',
    caption='下一课 L3-04 讲后端能力探测：score 是怎么打出来的、特性开关怎么被后端查询。',
    src=IMPL, parts=[(251, 254)], duration=18000,
    mark_src=[251, 252, 253, 254],
    visual='''
const wrap = U.el('div', { class: 'col', style: 'gap:9px;width:100%' });
wrap.innerHTML = `<div id="tbl"></div><div id="ex"></div><div class="formula" id="msg"></div>`;
root.appendChild(wrap);

const t = U.table(
  ['约定', '性质', '内容', '出处'],
  [['ggml_backend_init', '必需', 'dlsym 取出的注册函数，返回 reg 虚表', 'reg.cpp:237'],
   ['ggml_backend_score', '可选（择优时需要）', '返回 int，0 = 本机不支持', 'reg.cpp:229'],
   ['api_version', '必须精确相等', '与 GGML_BACKEND_API_VERSION 不等就拒绝', 'reg.cpp:246 / impl.h:11'],
   ['dl_handle_ptr', '库的寿命', '句柄被 move 进注册表，库一直保持加载', 'dl.h:40 / reg.cpp:261'],
   ['RTLD_NOW | RTLD_LOCAL', '打开方式', '立即解析符号 + 符号不进全局命名空间', 'dl.cpp:35'],
   ['GGML_BACKEND_PATH', '树外后端入口', '环境变量指向的库也会被加载', 'reg.cpp:600-602']],
  { monoCols: [0] });
wrap.querySelector('#tbl').appendChild(t.el);

wrap.querySelector('#ex').appendChild(W.exercise(
  '你要为一个新加速器写 ggml 后端动态库（构建时打开 GGML_BACKEND_DL）。' +
  '最少必须导出哪个符号？它返回什么？注册表还会检查这个返回值的哪个字段？',
  '最少必须导出 <span class="mono">ggml_backend_init</span>（<span class="mono">extern "C"</span>，' +
  '返回 <span class="mono">ggml_backend_reg_t</span>）。<br>' +
  '建议同时导出 <span class="mono">ggml_backend_score</span>：它在"按名字择优"的路径上是必需的 —— ' +
  '没有它，库会被跳过（见 source.md 第九节）。<br>' +
  '注册表检查返回值的 <span class="mono">api_version</span> 必须精确等于 ' +
  '<span class="mono">GGML_BACKEND_API_VERSION</span>（当前为 2），' +
  '否则报 "incompatible API version" 并放弃这个库。<br>' +
  '别忘了句柄的所有权：加载成功后注册表会持有它，库在整个进程里保持加载。'));

const msg = wrap.querySelector('#msg');
const rows = t.body.querySelectorAll('tr');
const texts = [
  '六条约定，前三条是"必须"，后三条是"怎么加载、怎么活"。',
  '<span class="v">ggml_backend_init</span>：唯一的必需符号。',
  '<span class="v">ggml_backend_score</span>：可选，但自动加载的择优路径上缺它不行。',
  '<span class="v">api_version</span>：只认精确相等 —— 这是 L3-01 的虚表契约的版本戳。',
  '<span class="v">dl_handle_ptr</span>：句柄归注册表，库就不会被卸载。',
  '<span class="v">RTLD_NOW | RTLD_LOCAL</span> 与 <span class="v">GGML_BACKEND_PATH</span>：打开方式与加载入口。',
  '一句话：<span class="k">动态加载 = 两个原语 + 三条约定（init 必需、版本精确相等、句柄交给注册表）</span>。'
];
tl.at(700, () => { msg.innerHTML = texts[0]; });
rows.forEach((r, i) => tl.at(2400 + i * 2200, () => {
  rows.forEach((x, k) => { x.className = (k === i) ? 'on' : ''; });
  msg.innerHTML = texts[Math.min(i + 1, 6)];
}));
tl.at(16000, () => { rows.forEach(x => { x.className = ''; }); msg.innerHTML = texts[6]; });
'''
)

# ------------------------------------------------------------------ source.md

L.section(
    '一、加载原语：dl_load_library / dl_get_sym / dl_error',
    'POSIX 分支的全部实现。注意两件事：\n\n'
    '1. 三个函数的签名里**没有任何后端相关的类型** —— 返回值是 `dl_handle *` 与 `void *`，'
    '所以调用方必须自己转型（注册表那边转成 `ggml_backend_init_t` / `ggml_backend_score_t`）。\n'
    '2. `RTLD_NOW | RTLD_LOCAL` 这两个 flag 是这段代码里唯一的"策略"：'
    '前者要求**打开时就解析全部符号**（缺符号等于打开失败），后者要求**符号不进全局命名空间**'
    '（多个后端导出同名符号也不会互相顶掉）。\n\n'
    '`dl_error()` 把 `dlerror()` 的 NULL 折成空串 —— 于是调用方（`GGML_LOG_ERROR` 那一行）'
    '可以无条件地打印它。',
    src=DL_C, parts=[(32, 48)], lang='c')

L.section(
    '二、同一抽象的另一端：Windows',
    'Windows 分支用 `LoadLibraryW` / `GetProcAddress`，并在两处用 `SetErrorMode` 压掉系统错误弹窗'
    '（注释写明理由：suppress error dialogs for missing DLLs）。\n\n'
    '对照上一节可以看到一处**能力不对等**：POSIX 的 `dl_error()` 会返回 `dlerror()` 的文本，'
    '而 Windows 分支的 `dl_error()` 直接 `return "";`。'
    '所以 Windows 上加载失败的日志里，错误描述永远是空的 —— 这是实现现状，不是笔误。\n\n'
    '关闭动作不在这里：它由头文件里的删除器负责（下一节）。',
    src=DL_C, parts=[(3, 30)], lang='c')

L.section(
    '三、句柄：dl_handle / dl_handle_deleter / dl_handle_ptr',
    '两端要暴露同一个类型 `dl_handle *`：Windows 的 `HMODULE` 本身就是指针，'
    '所以用 `std::remove_pointer_t<HMODULE>` 取出"被指向类型"；POSIX 侧直接把 `dl_handle` 定义成 `void`。'
    '两种写法都让 `dl_handle *` 合法。\n\n'
    '`dl_handle_deleter` 把关闭动作（`FreeLibrary` / `dlclose`）绑到析构函数上，'
    '`dl_handle_ptr` 则是 `std::unique_ptr<dl_handle, dl_handle_deleter>`：**只可 move，不可拷贝**。'
    '这条约束的意义在下一节会看到 —— 句柄被 move 进注册表，库就在整个进程里保持加载。',
    src=DL_H, parts=[(18, 40)], lang='c')

L.section(
    '四、★ 动态库必须导出的符号',
    '这是本课验收点的**直接证据**。注册表加载一个动态库的完整流程，以及它按名字取的两个符号：\n\n'
    '| 符号（dlsym 的字符串） | 必需性 | 行为 |\n|---|---|---|\n'
    '| `"ggml_backend_init"` | **必需** | 找不到就打印 `failed to find ggml_backend_init` 并放弃这个库 |\n'
    '| `"ggml_backend_score"` | 可选 | 取不到也继续；但返回 0 表示"本机不支持"，同样放弃 |\n\n'
    '三条 `return nullptr`（第 226 / 234 / 242 行）说明**任何一种失败都不会注册任何东西**：'
    '加载器要么交出一个可用的 reg，要么什么都不留。',
    src=REG, parts=[(220, 243)], lang='c')

L.section(
    '五、★ 版本校验：api_version 必须精确相等',
    '`ggml_backend_init` 返回的 reg 要过两道检查，写在同一句 `if` 里：'
    '`!reg || reg->api_version != GGML_BACKEND_API_VERSION`。'
    '注意是 `!=`（精确相等），不是"大于等于"或兼容区间 —— 虚表布局变了，旧库就不可用，'
    '宁可拒绝加载，也不要等到调用虚表时崩在别处。\n\n'
    '两种失败各有一条日志；版本不符的那条会把 backend 版本与当前版本都打出来。\n\n'
    '最后一行 `register_backend(reg, std::move(handle))`：**句柄被 move 进注册表**，'
    '于是库不会被卸载（句柄怎么存，见第八节）。',
    src=REG, parts=[(245, 264)], lang='c')

L.section(
    '六、符号从哪来：GGML_BACKEND_DL_IMPL',
    '后端不手写符号名：它写一行宏，宏负责生成 `extern "C"` 的导出函数。'
    '`extern "C"` 是必需的 —— `dlsym` 按**字面名字**查符号，而 C++ 会对函数名做名字改编（mangling）。\n\n'
    '整个宏块被 `#ifdef GGML_BACKEND_DL` 包住，最后四行是没有定义该宏时的分支：'
    '两个宏展开成**空**。也就是说，"库里有没有这两个符号"由**构建开关**决定，'
    '不是库里天然就有。',
    src=IMPL, parts=[(256, 286)], lang='c')

L.section(
    '七、后端侧只写一行',
    '真实后端怎么用这个宏：CPU 后端在自己的注册函数下面写一行 `GGML_BACKEND_DL_IMPL(...)`，'
    '把 `ggml_backend_cpu_reg` 交给宏。展开后，这个库就导出了 `ggml_backend_init`。\n\n'
    '（`ggml-cpu.cpp` 是 CPU 后端课的主文件，本课只为"宏的真实调用点"引用这一行。）',
    src=CPU, parts=[(716, 716)], lang='c')

L.section(
    '八、句柄交到注册表手上',
    '前半是注册表的 entry 类型：**一个 reg + 一个句柄**。后半是 `register_backend()`：'
    '它把 entry 推进 `backends` 向量（句柄随之被 move 进去），并为这个 reg 登记全部设备。\n\n'
    '把这两段与本课第三节的 `dl_handle_ptr` 连起来看：'
    '**"后端还加载着吗"完全由注册表持不持有这个 unique_ptr 决定**。',
    src=REG, parts=[(110, 113), (186, 205)], lang='c')

L.section(
    '九、择优路径上 score 是必需的',
    '`ggml_backend_load_best()` 在目录里枚举候选库时，走的是另一条路径：'
    '取 `"ggml_backend_score"`、调用它、保留分数最高的那个文件；'
    '**取不到 score 的库会被跳过**（日志：`failed to find ggml_backend_score`）。\n\n'
    '所以"可选符号"这个说法要加限定：对 `load_backend(path)`（直接点名加载）它是可选的；'
    '对"按名字自动择优"它是必需的。',
    src=REG, parts=[(534, 548)], lang='c')

L.section(
    '十、加载入口与候选名字',
    '启动时的入口：`ggml_backend_load_all()` 转发到这里。它按写死的名字列表逐个尝试'
    '（blas / zendnn / cann / cuda / hip / metal / rpc / sycl / vulkan / virtgpu / opencl / '
    'hexagon / musa / openvino / cpu），最后再看环境变量 `GGML_BACKEND_PATH`，'
    '用 `ggml_backend_load()` 加载一个"树外"后端。\n\n'
    '`silent` 随构建变化：Release（`NDEBUG`）下为 true —— 候选库不存在属于常态，不必刷屏。',
    src=REG, parts=[(578, 604)], lang='c')

L.section(
    '十一、对照：编译期注册的静态后端',
    '同一个注册表的另一条入口。构造函数里是一串 `#ifdef GGML_USE_*`：'
    '**编译期**就决定了哪些后端存在，运行期只是把它们登记进 `backends`。\n\n'
    '把这一节与第十节并排看，就是本课的核心对比：'
    '静态注册的"支持哪些后端"是编译期常量；动态加载把它变成了运行期才回答的问题 —— '
    '代价是符号名与版本号必须成为稳定契约。',
    src=REG, parts=[(119, 136)], lang='c')

L.section(
    '十二、版本号的当前值',
    '契约的版本戳就这一行。它与第五节的 `!=` 校验配对：'
    '**只要这个数字变了，所有已编译的动态后端都会被拒绝加载**，必须与主库一起重编。',
    src=IMPL, parts=[(11, 11)], lang='c')

L.footnote_add('本课为回答"后端动态库必须导出哪些符号"，额外引用了 `ggml/src/ggml-backend-reg.cpp`、'
               '`ggml/src/ggml-backend-impl.h` 与 `ggml/src/ggml-cpu/ggml-cpu.cpp` 的相应片段。'
               '前两者分别是 L3-02 / L3-01 的主文件，第三者属于 CPU 后端课；'
               '三者在覆盖度门禁里都算作被本课引用（`llama-coverage` 块里已一并声明）。')
L.footnote_add('**实测说明**（避免只从注释推断）：本机 `build/` 的 `CMakeCache.txt` 里写着 '
               '`GGML_BACKEND_DL:BOOL=OFF`，因此 `nm -D build/bin/libggml-cpu.so` 的输出里'
               '既没有 `ggml_backend_init` 也没有 `ggml_backend_score` —— '
               '`GGML_BACKEND_DL_IMPL` 展开为空（impl.h:285-286）。'
               '符号是否存在与构建选项绑定，不是"动态库里天然就有"。')

L.prereqs('`L3-02` 后端注册表与设备发现')

L.goal(
    '说出后端动态库必须导出的符号名，并区分哪个是必需的、哪个是"自动加载时必需"的（对应验收点）；',
    '解释 `dl_load_library` / `dl_get_sym` / `dl_error` 三个原语在 POSIX 与 Windows 上各是什么；',
    '说明 `RTLD_NOW` 与 `RTLD_LOCAL` 各自解决什么问题；',
    '解释注册表为什么要求 `reg->api_version` 与 `GGML_BACKEND_API_VERSION` 精确相等；',
    '说明 `GGML_BACKEND_DL_IMPL` 宏与 `GGML_BACKEND_DL` 构建开关如何决定符号是否存在。')

L.conclusion(
    '动态加载 = 两个原语 + 三条约定',
    '| 项 | 内容 | 出处 |\n|---|---|---|\n'
    '| 打开库 | `dl_load_library()`：POSIX 用 `dlopen(path, RTLD_NOW | RTLD_LOCAL)` | `ggml-backend-dl.cpp:35` |\n'
    '| 取符号 | `dl_get_sym()`：POSIX 用 `dlsym`，Windows 用 `GetProcAddress` | `ggml-backend-dl.cpp:40` / `:21` |\n'
    '| 必需符号 | `"ggml_backend_init"`，返回 `ggml_backend_reg_t` | `ggml-backend-reg.cpp:237` |\n'
    '| 可选符号 | `"ggml_backend_score"`，返回 `int`；自动择优路径上必需 | `ggml-backend-reg.cpp:229` / `:534-548` |\n'
    '| 版本 | `reg->api_version != GGML_BACKEND_API_VERSION` 就拒绝加载 | `ggml-backend-reg.cpp:246` |\n\n'
    '句柄的生命周期由 `dl_handle_ptr`（`std::unique_ptr` + 自定义删除器）表达；'
    '加载成功后它被 move 进注册表的 entry，库在整个进程里保持加载。')

L.conclusion(
    '★ 编译期决定 -> 运行期决定',
    '静态后端在注册表构造函数里用 `#ifdef GGML_USE_*` 登记（`ggml-backend-reg.cpp:119-136`）；'
    '动态后端则是在运行期按候选名字找库、`dlopen`、取符号、校验版本。\n\n'
    '**这正是动态加载的代价**：一旦"支持哪些后端"推迟到运行期，主库与后端库之间就不能再靠编译器对齐 —— '
    '所以符号名必须冻结（`ggml_backend_init` / `ggml_backend_score`）、'
    '结构体布局必须冻结（L3-01 的三张虚表），并配一个精确相等的版本号 `GGML_BACKEND_API_VERSION`。')

L.conclusion(
    '失败一律不注册',
    '加载路径上的每一种失败（打不开库、`init` 符号缺失、reg 为 NULL、版本不符、score 为 0）'
    '都只做一件事：`return nullptr`。注册表不会留下半成品后端 —— '
    '这也是为什么"某个后端没出现"往往要在日志里找原因，而不是在设备列表里。')

if __name__ == '__main__':
    d = L.build()
    print('built:', os.path.relpath(d, _ROOT))
