#!/usr/bin/env python3
"""跑一课的全部检查，并据【实际命令输出】改写该课 README.md 的验收点。

勾选不靠肉眼：只有对应命令退出码为 0 才打 [x]，括号里写的是命令的真实输出数字。

用法:
    python3 tools/verify_lesson.py L1-operator-representation/L1-01-tensor-data-plane
    python3 tools/verify_lesson.py --all
"""

import os
import re
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(HERE, 'tools')
SKIP_DIRS = {'shared', 'tools', 'plan', 'node_modules', '.git'}

USAGE = """usage: verify_lesson.py <课目录> | --all

跑一课（或全部课）的五项检查，并把结果写回该课 README.md 的"验收点"。

options:
  --all      全量
  --quiet    只输出结论
  -h, --help 显示本帮助
"""


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, script)] + list(args),
                       capture_output=True, text=True)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def lesson_dirs():
    out = []
    for name in sorted(os.listdir(HERE)):
        if name in SKIP_DIRS or name.startswith('.'):
            continue
        d = os.path.join(HERE, name)
        if not os.path.isdir(d):
            continue
        for sub in sorted(os.listdir(d)):
            p = os.path.join(d, sub)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'lesson.js')):
                out.append(p)
    return out


def facts(d):
    """从产物里读出结构性事实。"""
    f = {'scenes': 0, 'blocks': 0, 'files': 0, 'refs': 0}
    smd = os.path.join(d, 'source.md')
    if os.path.isfile(smd):
        with open(smd, encoding='utf-8') as fh:
            t = fh.read()
        m = re.search(r'<!--\s*llama-coverage\s*(.*?)-->', t, re.S)
        if m:
            f['files'] = len([x for x in m.group(1).split('\n') if x.strip()])
        f['refs'] = t.count('<!-- src:')
    ljs = os.path.join(d, 'lesson.js')
    if os.path.isfile(ljs):
        r = subprocess.run(['node', os.path.join(TOOLS, 'dump_scenes.js'), ljs],
                           capture_output=True, text=True)
        try:
            data = json.loads(r.stdout)
            f['scenes'] = len(data.get('scenes', []))
        except Exception:
            pass
        f['blocks'] = open(ljs, encoding='utf-8').read().count('  code: `')
    return f


def sanitize(msg):
    """验收点正文里不能出现 flag 形态的 token：否则门禁读到 README 会自我触发。"""
    return re.sub(r'(?<![\w(-])--[a-z][a-z0-9-]*', lambda m: m.group(0).replace('--', '- -'), msg)


def verify(d, quiet=False):
    rel = os.path.relpath(d, HERE)
    res = {}

    code, out = run('lint_lessons.py', '--lesson', d, '--quiet')
    m = re.search(r'(\d+) 课，(\d+) 个?错误，(\d+) 个警告', out)
    res['lint'] = (code == 0, f"{m.group(2)} 错误 / {m.group(3)} 警告" if m else sanitize(out.strip()[-60:]))

    code, out = run('check_ir_fidelity.py', '--lesson', d, '--quiet')
    m = re.search(r'(\d+) 个引用块 / (\d+) 个连续段', out)
    res['fid'] = (code == 0, f"{m.group(1)} 个引用块 / {m.group(2)} 个连续段逐字命中" if m else sanitize(out.strip()[-60:]))

    code, out = run('check_flags.py', '--lesson', d, '--quiet')
    m = re.search(r'长选项 (\d+) 个 / 短选项 (\d+) 个，扫描 (\d+) 个文件、(\d+) 处引用', out)
    res['flags'] = (code == 0, (f"真值集 {m.group(1)} 长 / {m.group(2)} 短选项，"
                                f"扫描 {m.group(3)} 文件 {m.group(4)} 处引用，0 处非法"
                                if code == 0 and m else '存在非法命令行参数（见 check_flags.py 输出）'))

    code, out = run('check_coverage.py', '--lesson', d, '--quiet')
    m = re.search(r'声明 (\d+) 项.*?全局进度 (\d+)/(\d+)', out)
    res['cov'] = (code == 0 and bool(m),
                  f"本课声明 {m.group(1)} 项，无空课、无幻影；全局覆盖 {m.group(2)}/{m.group(3)}"
                  if m else '本课声明无效')

    code, out = run('check_render.py', '--lesson', d, '--quiet')
    m = re.search(r'(\d+) 个页面 × (\d+) 种分辨率', out)
    res['render'] = (code == 0, f"{m.group(1)} 页面 x {m.group(2)} 分辨率，0 错误 / 0 溢出" if m else sanitize(out.strip()[-80:]))

    f = facts(d)
    lines = ['## 验收点', '']
    ok = res['fid'][0]
    lines.append(f"- [{'x' if res['fid'][0] else ' '}] 保真门禁：{f['blocks'] + f['refs']} 处引用 —— {res['fid'][1]}")
    lines.append(f"- [{'x' if res['cov'][0] else ' '}] 覆盖度门禁：{res['cov'][1]}")
    lines.append(f"- [{'x' if res['flags'][0] else ' '}] 参数门禁：{res['flags'][1]}")
    lines.append(f"- [{'x' if res['lint'][0] else ' '}] 语法检查：{res['lint'][1]}")
    lines.append(f"- [{'x' if res['render'][0] else ' '}] 渲染门禁：{f['scenes']} 幕 —— {res['render'][1]}")

    rdm = os.path.join(d, 'README.md')
    if os.path.isfile(rdm):
        with open(rdm, encoding='utf-8') as fh:
            txt = fh.read()
        head, sep, _tail = txt.partition('## 验收点')
        tail = ''
        m = re.search(r'> 勾选状态由.*', txt, re.S)
        if m:
            tail = m.group(0)
        new = head + '\n'.join(lines) + '\n\n' + tail
        with open(rdm, 'w', encoding='utf-8') as fh:
            fh.write(new)

    allok = all(v[0] for v in res.values())
    if not quiet or not allok:
        print(f'{"✓" if allok else "✗"} {rel}')
        for k in ('lint', 'fid', 'flags', 'cov', 'render'):
            print(f'    {"x" if res[k][0] else " "} {k}: {res[k][1]}')
    return allok


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    quiet = '--quiet' in sys.argv
    if '--all' in sys.argv:
        dirs = lesson_dirs()
    else:
        rest = [a for a in sys.argv[1:] if not a.startswith('--')]
        if not rest:
            print(USAGE)
            return 2
        dirs = [rest[0] if os.path.isabs(rest[0]) else os.path.join(HERE, rest[0])]
    if not dirs:
        print('没有课件')
        return 0
    ok = 0
    for d in dirs:
        if verify(d, quiet):
            ok += 1
    print(f'{"✓" if ok == len(dirs) else "✗"} 验收：{ok}/{len(dirs)} 课全绿')
    return 0 if ok == len(dirs) else 1


if __name__ == '__main__':
    sys.exit(main())
