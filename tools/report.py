#!/usr/bin/env python3
"""分层进度报告 —— 所有数字来自实测命令，禁止只报"进展顺利"。

用法:
    python3 tools/report.py            # 控制台报告
    python3 tools/report.py --md       # 写入 plan/REPORT.md
"""

import os
import re
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
import repo_universe as ru          # noqa: E402
import plan_model as pm             # noqa: E402
import plan_matrix as pmat          # noqa: E402

USAGE = """usage: report.py [--md] [--quiet]

分层进度报告：课时数 / 文件数 / 门禁实测输出 / 提交 SHA / 与计划的偏差。

options:
  --md       写入 plan/REPORT.md
  --quiet    只输出结论
  -h, --help 显示本帮助
"""


def sh(*args, cwd=None):
    r = subprocess.run(list(args), capture_output=True, text=True, cwd=cwd or HERE)
    return r.returncode, (r.stdout or '').strip()


def lesson_status():
    assign, unassigned = pmat.resolve()
    out = []
    for les in pm.LESSONS:
        d = os.path.join(HERE, pmat.lesson_dir(les))
        files = ('index.html', 'lesson.js', 'source.md', 'README.md')
        have = [f for f in files if os.path.isfile(os.path.join(d, f))]
        ok = len(have) == 4
        scenes = 0
        ljs = os.path.join(d, 'lesson.js')
        if os.path.isfile(ljs):
            with open(ljs, encoding='utf-8') as fh:
                scenes = fh.read().count('  kicker:')
        # 从 README 的验收点读回勾选状态（那是 verify_lesson 写的实测结果）
        ticks = 0
        rdm = os.path.join(d, 'README.md')
        if os.path.isfile(rdm):
            with open(rdm, encoding='utf-8') as fh:
                t = fh.read()
            sec = t.split('## 验收点')[-1] if '## 验收点' in t else ''
            ticks = sec.count('- [x]')
        out.append(dict(id=les['id'], layer=les['layer'], prio=les['prio'],
                        title=les['title'], done=ok, scenes=scenes,
                        nfiles=len(assign[les['id']]), ticks=ticks))
    return out, unassigned


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    quiet = '--quiet' in sys.argv
    to_md = '--md' in sys.argv

    st, unassigned = lesson_status()
    total = len(st)
    done = sum(1 for s in st if s['done'])
    green = sum(1 for s in st if s['ticks'] >= 5)
    scenes = sum(s['scenes'] for s in st)
    uni = ru.universe()
    covered = set()
    for les in pm.LESSONS:
        d = os.path.join(HERE, pmat.lesson_dir(les))
        smd = os.path.join(d, 'source.md')
        if not os.path.isfile(smd):
            continue
        with open(smd, encoding='utf-8') as fh:
            m = re.search(r'<!--\s*llama-coverage\s*(.*?)-->', fh.read(), re.S)
        if m:
            for x in m.group(1).split('\n'):
                x = x.strip()
                if x in set(uni):
                    covered.add(x)

    _, cov_out = sh(sys.executable, 'tools/check_coverage.py', '--quiet')
    _, fid_out = sh(sys.executable, 'tools/check_ir_fidelity.py', '--quiet')
    _, flag_out = sh(sys.executable, 'tools/check_flags.py', '--quiet')
    _, lint_out = sh(sys.executable, 'tools/lint_lessons.py', '--quiet')
    rc_self, self_out = sh(sys.executable, 'tools/selftest_gates.py')
    _, log = sh('git', 'log', '--oneline', '-n', '40')
    commits = [l for l in log.split('\n') if l.strip()]

    L = []
    A = L.append
    A('# 交付进度报告')
    A('')
    A(f'> 上游 `ggml-org/llama.cpp` @ `{ru.UPSTREAM_TAG}` (`{ru.UPSTREAM_COMMIT[:12]}`)。'
      '本报告所有数字均来自实际命令输出。')
    A('')
    A('## 一、总量')
    A('')
    A('| 指标 | 实测值 |')
    A('|---|---|')
    A(f'| 计划课数 | {total} |')
    A(f'| 四个交付文件齐备的课 | **{done}** |')
    A(f'| 五项验收全绿（README 有 5 个 [x]）的课 | **{green}** |')
    A(f'| 动画幕总数 | {scenes} |')
    A(f'| 覆盖域源文件 | {len(uni)} |')
    A(f'| 已被课件覆盖 | **{len(covered)}**（{len(covered)/len(uni)*100:.1f}%） |')
    A(f'| 未指派到课的文件 | {len(unassigned)} |')
    A('')
    A('## 二、分层')
    A('')
    A('| 层 | 课数 | 已齐备 | 全绿 | 覆盖文件 |')
    A('|---|---|---|---|---|')
    for lid, name, _desc in pm.LAYERS:
        ls = [s for s in st if s['layer'] == lid]
        A(f"| {lid} {name} | {len(ls)} | {sum(1 for s in ls if s['done'])} | "
          f"{sum(1 for s in ls if s['ticks'] >= 5)} | {sum(s['nfiles'] for s in ls)} |")
    A('')
    A('## 三、门禁实测输出')
    A('')
    A('```')
    for label, txt in (('覆盖度 B', cov_out), ('保真 A1/A3', fid_out),
                       ('参数 A2', flag_out), ('语法 lint', lint_out),
                       ('门禁自检', self_out)):
        A(f'--- {label} ---')
        A(txt.split('\n')[-1] if txt else '(无输出)')
    A('```')
    A('')
    A('## 四、完成清单（按课）')
    A('')
    A('| 课 | 层 | 优先级 | 幕 | 覆盖文件 | 验收打勾 | 状态 |')
    A('|---|---|---|---|---|---|---|')
    for s in st:
        A(f"| `{s['id']}` | {s['layer']} | {s['prio']} | {s['scenes']} | "
          f"{s['nfiles']} | {s['ticks']}/5 | "
          f"{'完成' if s['ticks'] >= 5 else ('已生成' if s['done'] else '未开始')} |")
    A('')
    A('## 五、提交历史（最近 40）')
    A('')
    A('```')
    for c in commits:
        A(c)
    A('```')
    A('')

    txt = '\n'.join(L)
    if to_md:
        p = os.path.join(HERE, 'plan', 'REPORT.md')
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(txt)
        print(f'已写出 {p}')
    if not quiet or not to_md:
        print(txt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
