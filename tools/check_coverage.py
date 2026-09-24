#!/usr/bin/env python3
"""覆盖度门禁（B 组）—— 覆盖域中每个源文件都被至少一课引用；无空课、无幻影引用。

  B1 全覆盖：universe - covered 必须为空
  B2 无空课：每课 source.md 的 coverage 块至少 1 项
  B3 无幻影：declared 里每一项必须真实存在

★ B3 的正确实现（踩过的坑）：
  不能用 `d in universe` 判断"是否存在"。universe 只含本视角的后缀，
  引用 .txt / .md / CMakeLists.txt 会被【误报】为不存在。
  正确做法：先查 universe，不在其中时再查文件系统真实存在性 —— 真实存在但非目标
  后缀的引用记为"被引用但不计入覆盖率"（诚实的不计入，既不误报也不虚报）。

用法:
    python3 tools/check_coverage.py
    python3 tools/check_coverage.py --quiet
    python3 tools/check_coverage.py --missing     # 列出未覆盖文件
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repo_universe as ru          # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'shared', 'tools', 'plan', 'node_modules', '.git'}
BLOCK_RE = re.compile(r'<!--\s*llama-coverage\s*(.*?)-->', re.S)

USAGE = """usage: check_coverage.py [--all] [--quiet] [--missing]

覆盖度门禁：覆盖域中每个源文件都被至少一课引用；无空课、无幻影引用。

options:
  --all      全量检查（默认行为，显式开关便于脚本化）
  --quiet    只输出结论
  --missing  列出未覆盖文件清单
  -h, --help 显示本帮助
"""


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
            if os.path.isdir(p):
                out.append(p)
    return out


def parse_declared(path):
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    m = BLOCK_RE.search(text)
    if not m:
        return None
    return [ln.strip() for ln in m.group(1).split('\n')
            if ln.strip() and not ln.strip().startswith('#')]


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    quiet = '--quiet' in sys.argv
    show_missing = '--missing' in sys.argv
    _all = '--all' in sys.argv          # 显式全量开关；默认已是全量
    only = sys.argv[sys.argv.index('--lesson') + 1] if '--lesson' in sys.argv else None

    universe = set(ru.universe())
    covered = set()
    non_counted = set()          # 真实存在但不计入覆盖率
    phantom = []
    empty_lessons = []
    declared_total = 0
    dirs = lesson_dirs()

    for d in dirs:
        rel = os.path.relpath(d, HERE)
        smd = os.path.join(d, 'source.md')
        if not os.path.isfile(smd):
            empty_lessons.append(f'{rel}（缺 source.md）')
            continue
        declared = parse_declared(smd)
        if declared is None:
            empty_lessons.append(f'{rel}（缺 llama-coverage 声明块）')
            continue
        if len(declared) == 0:
            empty_lessons.append(f'{rel}（声明块为空）')
            continue
        declared_total += len(declared)
        for dep in declared:
            dep = dep.lstrip('./')
            if dep in universe:
                covered.add(dep)
            elif os.path.isfile(os.path.join(ru.UPSTREAM, dep)):
                non_counted.add(dep)
            else:
                phantom.append((rel, dep))

    missing = sorted(universe - covered)
    total = len(universe)
    pct = (len(covered) / total * 100) if total else 100.0

    # --lesson 模式：只断言【本课】的声明有效（B2 无空课 / B3 无幻影）。
    # 全局 B1 覆盖率在所有课写完之前不可能满足，所以单课循环里不把它当阻塞项，
    # 只作为进度报告 —— 最终交付检查会跑全量 B1。
    if only:
        rel = os.path.normpath(only)
        mine = [e for e in empty_lessons if rel in e]
        my_phantom = [(l, d) for l, d in phantom if rel in l]
        smd = os.path.join(HERE if not os.path.isabs(only) else '', only, 'source.md')
        smd = smd if os.path.isfile(smd) else os.path.join(HERE, only, 'source.md')
        n_decl = 0
        if os.path.isfile(smd):
            d = parse_declared(smd)
            n_decl = len(d) if d else 0
        okd = (not mine) and (not my_phantom) and n_decl > 0
        if not okd:
            print(f'✗ B 组失败（本课）：声明 {n_decl} 项，'
                  f'空课 {len(mine)}，幻影 {len(my_phantom)}')
            for l, dd in my_phantom[:10]:
                print(f'   - {l}: {dd}')
            return 1
        print(f'✓ B 组通过（本课）：声明 {n_decl} 项，无空课，无幻影引用；'
              f'全局进度 {len(covered)}/{total}（{pct:.1f}%）')
        return 0

    ok = True
    if phantom:
        ok = False
    if empty_lessons:
        ok = False
    if missing:
        ok = False

    if not quiet:
        print(f'覆盖域源文件总数 : {total}')
        print(f'已被课件覆盖     : {len(covered)}  ({pct:.1f}%)')
        print(f'课件 source.md 数: {len(dirs)}')
        print(f'声明条目总数     : {declared_total}')
        if non_counted:
            print(f'真实存在但不计入覆盖率（非本视角后缀）: {len(non_counted)}')
        print()

    if empty_lessons:
        print(f'✗ B2 失败：{len(empty_lessons)} 课没有有效声明')
        for e in empty_lessons[:20]:
            print('   -', e)
    if phantom:
        print(f'✗ B3 失败：{len(phantom)} 处幻影引用（文件不存在）')
        for les, dep in phantom[:30]:
            print(f'   - {les}: {dep}')
    if missing:
        print(f'✗ B1 失败：{len(missing)} 个文件未被任何一课覆盖')
        if show_missing:
            for m in missing[:80]:
                print('   -', m)
            if len(missing) > 80:
                print(f'   ... 另有 {len(missing) - 80} 个')
        else:
            print('   （加 --missing 查看清单）')

    if not ok:
        return 1

    if dirs:
        print(f'✓ B 组通过：{len(covered)}/{total} 全覆盖（{pct:.1f}%），'
              f'无空课，无幻影引用')
    else:
        print(f'✓ B 组通过：空课件集（0/{total}，覆盖率 0.0%）—— 门禁不误报')
    return 0


if __name__ == '__main__':
    sys.exit(main())
