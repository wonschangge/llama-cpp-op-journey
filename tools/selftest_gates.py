#!/usr/bin/env python3
"""门禁自检 —— 证明门禁不是空转的。

一个只会说"通过"的门禁等于没有门禁。本脚本构造【故意出错】的样本，
断言每一道门禁都能把它抓出来。全部通过才算门禁可信。

用法:
    python3 tools/selftest_gates.py
"""

import os
import re
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 样本必须放在【层目录下的课目录】里 —— 门禁只扫描这一层，放别处会变成
# "找不到课件" 从而假通过（第一次自检就踩到了这个坑）。
TMP = os.path.join(HERE, 'L1-operator-representation')
TOOLS = os.path.join(HERE, 'tools')

USAGE = """usage: selftest_gates.py

门禁自检：构造故意出错的样本，断言每道门禁都能抓出来。

options:
  -h, --help  显示本帮助
"""


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, script)] + list(args),
                       capture_output=True, text=True)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)


def case_phantom():
    """B3：引用一个不存在的文件，覆盖度门禁必须报幻影。"""
    d = os.path.join(TMP, 'L1-99-phantom')
    w(os.path.join(d, 'source.md'),
      '<!-- llama-coverage\nggml/include/ggml.h\nggml/src/does-not-exist.cpp\n-->\n\n# x\n')
    code, out = run('check_coverage.py', '--lesson',
                    'L1-operator-representation/L1-99-phantom')
    return code != 0 and '幻影' in out, out.strip().split('\n')[-1]


def case_fabricated():
    """A1：引用块里混入一行源文件里没有的代码，保真门禁必须报出来。"""
    d = os.path.join(TMP, 'L1-99-fabricated')
    w(os.path.join(d, 'source.md'),
      '<!-- llama-coverage\nggml/include/ggml.h\n-->\n\n'
      '# x\n\n<!-- src: ggml/include/ggml.h -->\n```c\n'
      '    struct ggml_tensor {\n'
      '        enum ggml_type type;\n'
      '        int this_line_does_not_exist_in_upstream = 42;\n'
      '        struct ggml_backend_buffer * buffer;\n'
      '```\n')
    code, out = run('check_ir_fidelity.py', '--lesson',
                    'L1-operator-representation/L1-99-fabricated', '--quiet')
    return code != 0, out.strip().split('\n')[-1]


def case_gap():
    """A3：引用块中间跳过若干行（不连续），保真门禁必须报出来。"""
    d = os.path.join(TMP, 'L1-99-gap')
    w(os.path.join(d, 'source.md'),
      '<!-- llama-coverage\nggml/include/ggml.h\n-->\n\n'
      '# x\n\n<!-- src: ggml/include/ggml.h -->\n```c\n'
      '    struct ggml_tensor {\n'
      '        enum ggml_type type;\n'
      # 故意跳过【非空行】688（buffer 行），直接接 ne[] 行 —— 位置不连续。
      # 注意不能用空行来构造"不连续"：空行在本门禁里是透明化的（设计如此）。
      '        int64_t ne[GGML_MAX_DIMS]; // number of elements\n'
      '```\n')
    code, out = run('check_ir_fidelity.py', '--lesson',
                    'L1-operator-representation/L1-99-gap', '--quiet')
    return code != 0, out.strip().split('\n')[-1]


def case_flag():
    """A2：发明一个不存在的命令行参数，参数门禁必须报出来。"""
    d = os.path.join(TMP, 'L1-99-flag')
    w(os.path.join(d, 'README.md'),
      '# x\n\n用法：`llama-cli --totally-made-up-flag 3`\n')
    code, out = run('check_flags.py', '--quiet')
    ok = code != 0 and 'totally-made-up-flag' in out
    return ok, [l for l in out.strip().split('\n') if 'totally' in l][:1] or out.strip().split('\n')[-1:]


def case_syntax():
    """lint：故意写坏 JS 语法，静态检查必须报出来。"""
    d = os.path.join(TMP, 'L1-99-syntax')
    w(os.path.join(d, 'lesson.js'),
      "const SCENES = [{ kicker: 'a', title: 't', sub:'s', caption:'c',\n"
      "  src: 'ggml/include/ggml.h', code: `x`, duration: 9000,\n"
      "  build(root, tl) { const w = U.el('div', { style: 'gap':13px }); } }];\n")
    w(os.path.join(d, 'index.html'), '<html><body></body></html>')
    code, out = run('lint_lessons.py', '--lesson',
                    'L1-operator-representation/L1-99-syntax', '--quiet')
    return code != 0, out.strip().split('\n')[-1]


def case_render_overflow():
    """C：故意让卡片宽度超出舞台，渲染门禁必须报溢出。"""
    d = os.path.join(TMP, 'L1-99-overflow')
    up = '../../'   # 课件目录深度为 2，所以 shared/ 在两级之上
    w(os.path.join(d, 'lesson.js'),
      "const SCENES = [{ kicker:'k', title:'t', sub:'s', caption:'c',\n"
      "  src: 'ggml/include/ggml.h', code: 'int x;', duration: 9000,\n"
      "  build(root, tl) { root.innerHTML = '<div style=\"width:3000px;height:40px;"
      "background:#333\"></div>'; } }];\n")
    w(os.path.join(d, 'index.html'),
      '<!DOCTYPE html><html><head><meta charset="utf-8">'
      '<link rel="icon" href="data:image/svg+xml,%3Csvg/%3E">'
      f'<link rel="stylesheet" href="{up}shared/theme.css"></head><body>'
      f'<script src="{up}shared/engine.js"></script>'
      f'<script src="{up}shared/widgets.js"></script>'
      f'<script src="{up}shared/lesson-shell.js"></script>'
      '<script src="lesson.js"></script><script>SHELL.boot(SCENES,{nav:{}});</script>'
      '</body></html>')
    code, out = run('check_render.py', '--lesson',
                    'L1-operator-representation/L1-99-overflow', '--quiet')
    return code != 0 and '溢出' in out, out.strip().split('\n')[-1]


CASES = [
    ('B3 幻影引用', case_phantom),
    ('A1 编造代码行', case_fabricated),
    ('A3 引用不连续', case_gap),
    ('A2 发明参数', case_flag),
    ('lint 语法错误', case_syntax),
    ('C 布局溢出', case_render_overflow),
]


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    if os.path.isdir(TMP):
        shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    bad = 0
    print('门禁自检：每一条都构造一个【应当被抓到】的样本\n')
    for name, fn in CASES:
        try:
            caught, detail = fn()
        except Exception as e:
            caught, detail = False, f'自检本身抛异常: {e}'
        mark = '✓' if caught else '✗'
        if not caught:
            bad += 1
        d = detail if isinstance(detail, str) else ' '.join(detail)
        print(f'  {mark} {name:<16} {"已抓到" if caught else "【漏报！】"}  {d.strip()[:96]}')
    for name in os.listdir(TMP):
        if name.startswith('L1-99-'):
            shutil.rmtree(os.path.join(TMP, name), ignore_errors=True)
    print()
    if bad:
        print(f'✗ 门禁自检失败：{bad}/{len(CASES)} 个样本没被抓住 —— 门禁不可信')
        return 1
    print(f'✓ 门禁自检通过：{len(CASES)}/{len(CASES)} 个故意出错的样本全部被抓到')
    return 0


if __name__ == '__main__':
    sys.exit(main())
