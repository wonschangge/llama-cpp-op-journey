#!/usr/bin/env python3
"""自动拆块 —— 写完 source.md 后先跑它，再跑保真门禁。

问题：把"签名 + 函数体"写进同一个围栏块时，源文件里夹在中间的注释行没被引用，
整块就不再【连续】，保真门禁会报错。

两种合法处理（本工具做第 2 种，因为它不掩盖任何东西）：
  1. 把夹在中间的注释行原样保留在引用里（最忠实）
  2. 在断点处把围栏块【拆成两个块】，各自重复 <!-- src: ... --> 声明

★ 不做的事：插入假的"注解"去盖住断点。那会把真实的不连续藏起来，
   让门禁失去意义。

用法:
    python3 tools/split_blocks.py --all            # 报告哪些块需要拆
    python3 tools/split_blocks.py --all --write    # 就地拆块
    python3 tools/split_blocks.py --lesson DIR --write
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repo_universe as ru                                  # noqa: E402
from check_ir_fidelity import (normalize, compact_source,  # noqa: E402
                               locate, SENTINEL, SRC_RE, FENCE_RE)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'shared', 'tools', 'plan', 'node_modules', '.git'}

USAGE = """usage: split_blocks.py [--all] [--lesson DIR] [--write] [--quiet]

自动拆块：把因"中间夹了未引用的行"而不连续的围栏块，在断点处拆成多个块。

options:
  --all        全量处理（默认行为，显式开关便于脚本化）
  --lesson DIR 只处理一课
  --write      就地写回（默认只报告）
  --quiet      只输出结论
  -h, --help   显示本帮助
"""


def lesson_dirs(only=None):
    if only:
        p = only if os.path.isabs(only) else os.path.join(HERE, only)
        return [p] if os.path.isdir(p) else []
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


def match_run(run, src_texts):
    n, _a, _b = locate(run, src_texts)
    return n == len(run)


def split_code(code_lines, src_texts):
    """把 code_lines 切成若干"各自连续命中"的段。返回 (runs, unmatched)。"""
    runs, cur, unmatched = [], [], []
    for line in code_lines:
        if not line.strip():
            cur.append(line)
            continue
        trial = [x for x in cur if x.strip()] + [line]
        if match_run(trial, src_texts):
            cur.append(line)
        else:
            if [x for x in cur if x.strip()]:
                runs.append(cur)
            cur = []
            if match_run([line], src_texts):
                cur = [line]
            else:
                unmatched.append(line.strip()[:100])
    if [x for x in cur if x.strip()]:
        runs.append(cur)
    return runs, unmatched


def process(path, write):
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    out, i, changed, nblocks, nsplit = [], 0, False, 0, 0
    pending = None
    while i < len(lines):
        m = SRC_RE.search(lines[i])
        if m and not FENCE_RE.match(lines[i]):
            pending = m.group(1)
            out.append(lines[i])
            i += 1
            continue
        f = FENCE_RE.match(lines[i])
        if f and pending:
            lang = f.group(1)
            j = i + 1
            body = []
            while j < len(lines) and not FENCE_RE.match(lines[j]):
                body.append(lines[j])
                j += 1
            sl = ru.UPSTREAM and os.path.join(ru.UPSTREAM, pending)
            if lang.lower() in ('text', 'txt', 'plain') or not os.path.isfile(sl or ''):
                out.extend(lines[i:j + 1])
                pending = None
                i = j + 1
                continue
            with open(sl, encoding='utf-8', errors='replace') as fh2:
                src_lines = fh2.read().split('\n')
            _nos, texts = compact_source(src_lines)
            nblocks += 1
            # 先按注解行切，再对每段做连续性拆分
            segs, cur = [], []
            for ln in body:
                if ln.lstrip().startswith(SENTINEL):
                    if cur:
                        segs.append((cur, False))
                        cur = []
                    segs.append(([ln], True))
                else:
                    cur.append(ln)
            if cur:
                segs.append((cur, False))

            pieces = []
            for seg, is_note in segs:
                if is_note:
                    pieces.append(seg)
                    continue
                runs, unmatched = split_code(seg, texts)
                flat = [x for r in runs for x in r]
                if len(flat) < len([x for x in seg if x.strip()]):
                    unmatched = unmatched or ['<未知>']
                if unmatched:
                    print(f'   ! {os.path.relpath(path, HERE)}: 以下行在 {pending} 里找不到，'
                          f'拆块救不了（疑似打错/编造）:')
                    for u in unmatched[:5]:
                        print(f'      {u}')
                for ri, r in enumerate(runs):
                    if ri:
                        nsplit += 1
                    pieces.append(r)

            if len(pieces) == 1:
                out.extend(lines[i:j + 1])
            else:
                changed = True
                for k, p in enumerate(pieces):
                    if k:
                        out.append(f'<!-- src: {pending} -->')
                        out.append(lines[i])
                    out.extend(p)
                out.append(lines[j])
            pending = None
            i = j + 1
            continue
        out.append(lines[i])
        i += 1

    if changed and write:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(out))
    return nblocks, nsplit, changed


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    write = '--write' in sys.argv
    quiet = '--quiet' in sys.argv
    only = sys.argv[sys.argv.index('--lesson') + 1] if '--lesson' in sys.argv else None
    dirs = lesson_dirs(only)
    tb = ts = tc = 0
    for d in dirs:
        p = os.path.join(d, 'source.md')
        if not os.path.isfile(p):
            continue
        b, s, c = process(p, write)
        tb += b
        ts += s
        tc += bool(c)
        if c and not quiet and not write:
            print(f'   需要拆块: {os.path.relpath(p, HERE)}（{s} 个断点）')
    if not dirs:
        print('没有课件（空集，无需拆块）')
        return 0
    verb = '已拆' if write else '待拆'
    print(f'{"✓" if not tc else "!"} split_blocks: {tb} 个引用块，'
          f'{verb} {ts} 处断点，涉及 {tc} 个 source.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
