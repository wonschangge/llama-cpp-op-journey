#!/usr/bin/env python3
"""lessonkit —— 课件生成器。

为什么需要它：52 课、约 6000 行 C/C++ 引用。逐字保真不能靠手抄（必然出错）。
本模块把"引用哪一段"变成【行号】，从上游文件直接抽取文本，逐字保真由构造保证；
保真门禁则作为独立回归继续校验最终产物。

一课 = 一个 spec（Python），build() 产出 lesson.js 与 source.md 两个交付文件。

spec 用法（写在该课目录下的 lesson.spec.py）：

    from lessonkit import Lesson

    L = Lesson(id='L1-01', layer='L1', title='...', kicker='...', codecap='...')

    L.scene(kicker='...', title='...', sub='...', caption='...',
            src='ggml/include/ggml.h',
            parts=[(684, 717)],            # 逐字引用的行区间（1-based，含端点）
            notes={3: '这一行是……'},        # 在 code 的第 3 行（0-based）后插入注解
            marks=[3, 4],                  # 高亮 code 的第几行
            duration=15000,
            visual='''...JS...''')         # root/tl 可用

    L.section('一、标题', '正文……', src='ggml/include/ggml.h', parts=[(222, 226)])
    L.build()
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
import repo_universe as ru          # noqa: E402

NOTE_PREFIX = '//>>'


def q(rel, a, b):
    """取上游文件 rel 的第 a..b 行（1-based，含端点）原文。"""
    path = os.path.join(ru.UPSTREAM, rel)
    with open(path, encoding='utf-8', errors='replace') as fh:
        lines = fh.read().split('\n')
    if a < 1 or b > len(lines) or a > b:
        raise SystemExit(f'行号越界: {rel}:{a}-{b}（文件共 {len(lines)} 行）')
    return lines[a - 1:b]


def js_escape(s):
    """把文本安全地放进 JS 模板字符串。"""
    return (s.replace('\\', '\\\\')
             .replace('`', '\\`')
             .replace('${', '\\${'))


class Lesson:
    def __init__(self, id, layer, title, kicker=None, codecap='源码（逐字引用）',
                 nav=None, autoadvance=True):
        self.id = id
        self.layer = layer
        self.title = title
        self.kicker = kicker or ''
        self.codecap = codecap
        self.nav = nav or {'prev': None, 'next': None}
        self.autoadvance = autoadvance
        self.scenes = []
        self.sections = []
        self.files = []          # 覆盖声明（去重）
        self.preamble = []
        self.footnote = []
        self.goals = []
        self.conclusions = []
        self.prereq = '无'

    def goal(self, *items):
        for it in items:
            self.goals.append(it)
        return self

    def conclusion(self, heading, text):
        self.conclusions.append((heading, text))
        return self

    def prereqs(self, text):
        self.prereq = text
        return self

    # ---------------------------------------------------------------- 录入

    def cover(self, *rels):
        for r in rels:
            if r not in self.files:
                self.files.append(r)
        return self

    def scene(self, kicker, title, sub, caption, src, parts, visual,
              notes=None, marks=None, duration=15000, linebase=None):
        code = self._render_code(src, parts, notes or {}, linebase)
        self.cover(src)
        self.scenes.append(dict(
            kicker=kicker, title=title, sub=sub, caption=caption,
            src=src, code=code, marks=marks or [], duration=duration,
            visual=visual, lineno=self._last_linebase))
        return self

    def note(self, text):
        """散文段落（放在 source.md 开头）。"""
        self.preamble.append(text)
        return self

    def section(self, heading, prose, src=None, parts=None, notes=None, lang='c'):
        block = None
        if src and parts:
            self.cover(src)
            block = dict(src=src, code=self._render_code(src, parts, notes or {}),
                         lang=lang)
        self.sections.append(dict(heading=heading, prose=prose, block=block))
        return self

    def footnote_add(self, text):
        self.footnote.append(text)
        return self

    # ------------------------------------------------------------ 渲染代码

    def _render_code(self, src, parts, notes, linebase=None):
        out = []
        for pi, (a, b) in enumerate(parts):
            chunk = q(src, a, b)
            if pi:
                out.append(f'{NOTE_PREFIX} ---- {src}:{a}-{b} ----')
            for k, ln in enumerate(chunk):
                out.append(ln)
                if k in notes:
                    out.append(f'{NOTE_PREFIX} {notes[k]}')
        # 行号槽：单段时显示【上游真实行号】（避免误导）；多段时禁用（0 = 不显示），
        # 因为拼接后的行号无法用单一数字表达，改由 //>> ---- 分隔行标注真实区间。
        if linebase:
            self._last_linebase = linebase
        elif len(parts) == 1:
            self._last_linebase = parts[0][0]
        else:
            self._last_linebase = 0
        return '\n'.join(out)

    # ---------------------------------------------------------------- 产出

    def build(self, outdir=None):
        d = outdir or os.path.join(HERE, self._dir())
        os.makedirs(d, exist_ok=True)
        self._write_lesson_js(os.path.join(d, 'lesson.js'))
        self._write_source_md(os.path.join(d, 'source.md'))
        self._write_index_html(os.path.join(d, 'index.html'))
        self._write_readme(os.path.join(d, 'README.md'))
        return d

    def _strip_html(self, s):
        import re as _re
        return _re.sub(r'<[^>]+>', '', s or '')

    def _nblocks(self):
        n = 0
        for s in self.scenes:
            if s['code']:
                n += 1
        for sec in self.sections:
            if sec['block']:
                n += 1
        return n

    def _write_readme(self, path):
        import re as _re
        L = []
        L.append(f"# {self.id} · {self.title} — 课件说明")
        L.append('')
        L.append(f"> 层：**{self.layer}** ｜ 前置课：{self.prereq}")
        L.append('')
        L.append('## 学习目标')
        L.append('')
        L.append('看完这一课，你应该能：')
        L.append('')
        for i, g in enumerate(self.goals, 1):
            L.append(f'{i}. {g}')
        L.append('')
        L.append(f'## 覆盖的源文件（{len(self.files)} 个）')
        L.append('')
        L.append('| 文件 | 行数 |')
        L.append('|---|---|')
        for f in self.files:
            p = os.path.join(ru.UPSTREAM, f)
            n = len(open(p, encoding='utf-8', errors='replace').read().split('\n')) if os.path.isfile(p) else 0
            L.append(f'| `{f}` | {n} |')
        L.append('')
        if self.footnote:
            for f in self.footnote:
                L.append(f'> **说明**：{f}')
            L.append('')
        L.append(f'## 场景（{len(self.scenes)} 幕）')
        L.append('')
        for i, s in enumerate(self.scenes, 1):
            L.append(f"{i}. **{self._strip_html(s['title'])}** — {self._strip_html(s['sub'])}")
        L.append('')
        L.append('## 核心结论')
        L.append('')
        for h, t in self.conclusions:
            L.append(f'### {h}')
            L.append('')
            L.append(t)
            L.append('')
        L.append('## 验收点')
        L.append('')
        L.append(f'- [ ] 保真门禁：{self._nblocks()} 个引用块全部逐字来自声明的源文件，且位置连续')
        L.append(f'- [ ] 覆盖度门禁：{len(self.files)} 个源文件均被声明')
        L.append('- [ ] 参数门禁：无非法命令行参数')
        L.append('- [ ] 语法检查：通过')
        L.append(f"- [ ] 渲染门禁：{len(self.scenes)} 幕，无 JS 错误、无布局溢出、交互可用")
        L.append('- [ ] 自检：不看代码，能说出 `struct ggml_tensor` 里哪些字段决定一次内核调用的形状')
        L.append('')
        L.append('> 勾选状态由 `python3 tools/verify_lesson.py <课目录>` 依据实际命令输出改写，'
                 '不靠肉眼。')
        L.append('')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(L))

    def _dir(self):
        sys.path.insert(0, os.path.join(HERE, 'tools'))
        import plan_matrix as pmat
        for les in pmat.pm.LESSONS:
            if les['id'] == self.id:
                return pmat.lesson_dir(les)
        raise SystemExit(f'计划里没有这一课: {self.id}')

    def _write_lesson_js(self, path):
        L = []
        L.append('/* ' + '=' * 74)
        L.append(f"   {self.id} · {self.title}")
        L.append('   ' + '-' * 74)
        L.append('   本文件由 lesson.spec.py 经 tools/lessonkit.py 生成。')
        L.append('   代码块按行号从上游源文件直接抽取，逐字保真由构造保证；')
        L.append('   最终产物仍由 tools/check_ir_fidelity.py 独立校验。')
        L.append('   排版基准：#sceneBody 可视区 = 697 x 499 舞台像素（1280x720 下实测）。')
        L.append('   ' + '=' * 74 + ' */')
        L.append("'use strict';")
        L.append('')
        L.append('const SCENES = [')
        for i, s in enumerate(self.scenes):
            L.append(f'/* {"-" * 54} {i + 1} {s["title"]} */')
            L.append('{')
            L.append(f'  kicker: {json.dumps(s["kicker"], ensure_ascii=False)},')
            L.append(f'  title: {json.dumps(s["title"], ensure_ascii=False)},')
            L.append(f'  sub: {json.dumps(s["sub"], ensure_ascii=False)},')
            L.append(f'  caption: {json.dumps(s["caption"], ensure_ascii=False)},')
            L.append(f'  src: {json.dumps(s["src"])},')
            L.append(f'  mark: {json.dumps(s["marks"])},')
            L.append(f'  lineNo: {s["lineno"]},')
            L.append('  code: `' + js_escape(s['code']) + '`,')
            L.append(f'  duration: {s["duration"]},')
            L.append('  build(root, tl) {')
            for line in s['visual'].strip('\n').split('\n'):
                L.append('    ' + line if line.strip() else '')
            L.append('  }')
            L.append('},')
            L.append('')
        L.append('];')
        L.append('')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(L))

    def _write_source_md(self, path):
        L = []
        L.append('<!-- llama-coverage')
        for f in self.files:
            L.append(f)
        L.append('-->')
        L.append('')
        L.append(f'# {self.id} · {self.title} — 源文件')
        L.append('')
        for p in self.preamble:
            L.append(p)
            L.append('')
        L.append('---')
        L.append('')
        for sec in self.sections:
            L.append(f'## {sec["heading"]}')
            L.append('')
            L.append(sec['prose'])
            L.append('')
            if sec['block']:
                b = sec['block']
                L.append(f'<!-- src: {b["src"]} -->')
                L.append(f'```{b["lang"]}')
                L.append(b['code'])
                L.append('```')
                L.append('')
        if self.footnote:
            L.append('---')
            L.append('')
            L.append('## 说明')
            L.append('')
            for f in self.footnote:
                L.append(f'- {f}')
            L.append('')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(L))

    def _write_index_html(self, path):
        up = self._shared_prefix()
        p = self.nav.get('prev')
        n = self.nav.get('next')
        def link(x, fallback):
            if x and x.get('href'):
                return "{ href: '%s', label: '%s' }" % (x['href'], x.get('label', fallback))
            return 'null'
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{self.id} {self.title} · llama.cpp 算子之旅</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230d1117'/%3E%3Cpath d='M7 22 L13 10 L19 22' stroke='%2358a6ff' stroke-width='2.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='24' cy='12' r='3' fill='%233fb950'/%3E%3C/svg%3E">
<link rel="stylesheet" href="{up}shared/theme.css">
</head>
<body>
<script src="{up}shared/engine.js"></script>
<script src="{up}shared/widgets.js"></script>
<script src="{up}shared/lesson-shell.js"></script>
<script src="lesson.js"></script>
<script>
SHELL.boot(SCENES, {{
  kicker: {json.dumps(self.kicker or self.layer, ensure_ascii=False)},
  title: {json.dumps(self.id + ' · ' + self.title, ensure_ascii=False)},
  codeCap: {json.dumps(self.codecap, ensure_ascii=False)},
  autoAdvance: {'true' if self.autoadvance else 'false'},
  nav: {{ prev: {link(p, '上一课')}, next: {link(n, '下一课')} }}
}});
</script>
</body>
</html>
'''
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(html)

    def _shared_prefix(self):
        depth = len(self._dir().split('/'))
        return '../' * depth


def run(spec_path):
    """执行一个 lesson.spec.py。"""
    import runpy
    g = {'__name__': '__main__', '__file__': spec_path}
    runpy.run_path(spec_path, init_globals=g, run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python3 tools/lessonkit.py <lesson.spec.py> [...]')
        sys.exit(2)
    for p in sys.argv[1:]:
        run(os.path.abspath(p))
        print('built:', p)
