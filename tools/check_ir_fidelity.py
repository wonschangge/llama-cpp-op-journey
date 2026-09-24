#!/usr/bin/env python3
"""保真门禁（A1/A3）—— 课件里每一段代码引用必须逐字来自声明的源文件，且位置连续。

检查两个来源：
  1. <课>/source.md  里的围栏代码块，其上方用 <!-- src: 相对路径 --> 声明出处
  2. <课>/lesson.js  里每个 scene 的 code 字段，其 scene 用 src: '相对路径' 声明出处

注解行：以 `//>>` 开头的行是【作者注解】，不计入逐字校验，并作为连续段的切分点。
        它渲染出来是注解样式，不会被误当成上游代码。
        切分后每一段仍必须与源文件位置连续 —— 这是本门禁的核心断言。

用法:
    python3 tools/check_ir_fidelity.py               # 全量
    python3 tools/check_ir_fidelity.py --lesson L1-operator-representation/L1-01-tensor-data-plane
    python3 tools/check_ir_fidelity.py --quiet
    python3 tools/check_ir_fidelity.py --ranges      # 打印每段命中的源文件行号
"""

import os
import re
import sys
import json
import subprocess
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repo_universe as ru          # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTINEL = '//>>'
SRC_RE = re.compile(r'<!--\s*src:\s*([^\s>]+?)(?:\s*:\s*(\S+))?\s*-->')
SKIP_DIRS = {'shared', 'tools', 'plan', 'node_modules', '.git'}
FENCE_RE = re.compile(r'^\s*```\s*([A-Za-z0-9_+-]*)\s*$')

USAGE = """usage: check_ir_fidelity.py [--all] [--lesson DIR] [--quiet] [--ranges]

保真门禁：课件里的代码引用必须逐字来自声明的源文件，且位置连续。

options:
  --all            全量检查（默认行为，显式开关便于脚本化）
  --lesson DIR     只检查一课（相对仓库根或绝对路径）
  --quiet          只输出结论
  --ranges         打印每段命中的源文件行号区间
  -h, --help       显示本帮助
"""

_cache = {}


def read_source(rel):
    if rel not in _cache:
        path = os.path.join(ru.UPSTREAM, rel)
        if not os.path.isfile(path):
            _cache[rel] = None
        else:
            with open(path, encoding='utf-8', errors='replace') as fh:
                _cache[rel] = fh.read().split('\n')
    return _cache[rel]


def normalize(line):
    s = line.rstrip()
    s = re.sub(r'\s*//\s*expected-.*$', '', s)
    return s.strip()


def compact(lines):
    """去掉空行与纯注解行，返回 [(原文, 源文件行号或 None)]。

    空行透明化：白空格不承载语义，且上游常有空行穿插。
    """
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        if ln.lstrip().startswith(SENTINEL):
            out.append((ln, None))          # 注解：切分点
            continue
        out.append((ln, i))
    return out


def build_index(src_lines):
    idx = collections.defaultdict(list)
    for i, ln in enumerate(src_lines):
        if not ln.strip():
            continue
        idx[normalize(ln)].append(i)
    return idx


def compact_source(src_lines):
    """源文件去掉空行后的 (行号列表, 规范化文本列表)。"""
    nos, texts = [], []
    for i, ln in enumerate(src_lines):
        if not ln.strip():
            continue
        nos.append(i)
        texts.append(normalize(ln))
    return nos, texts


def runs_of(items):
    """按注解行把 items 切成若干连续段。items = [(原文, 行号)]"""
    out, cur = [], []
    for text, _no in items:
        if text.lstrip().startswith(SENTINEL):
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(text)
    if cur:
        out.append(cur)
    return out


def locate(run, src_texts):
    """在源文件里找与 run 匹配最长的一段连续区间。

    返回 (匹配行数, 起始下标, 结束下标)。要求【位置连续】，不是"每行都出现过"。
    """
    if not run:
        return 0, -1, -1
    first = normalize(run[0])
    best = (0, -1, -1)
    for start, t in enumerate(src_texts):
        if t != first:
            continue
        k = 0
        while k < len(run) and start + k < len(src_texts) and src_texts[start + k] == normalize(run[k]):
            k += 1
        if k > best[0]:
            best = (k, start, start + k - 1)
        if k == len(run):
            break
    return best


def check_block(code, srcs, label, problems, ranges, show_ranges):
    """srcs 可以是单个路径或路径列表（择一命中即可）。"""
    if isinstance(srcs, str):
        srcs = [srcs]
    items = compact(code.split('\n'))
    if not items:
        return 0, 0
    rs = runs_of(items)
    ok_runs = 0
    for ri, run in enumerate(rs):
        best = (0, -1, -1)
        hit_file = None
        for rel in srcs:
            sl = read_source(rel)
            if sl is None:
                problems.append(f'{label}: 声明的源文件不存在: {rel}')
                continue
            _nos, texts = compact_source(sl)
            cand = locate(run, texts)
            if cand[0] > best[0]:
                best = cand
                hit_file = rel
        if best[0] == len(run):
            ok_runs += 1
            if show_ranges and hit_file:
                nos, _t = compact_source(read_source(hit_file))
                a = nos[best[1]] + 1
                b = nos[best[2]] + 1
                ranges.append(f'{label} run{ri + 1}: {hit_file}:{a}-{b} ({len(run)} 行)')
        else:
            bad = run[best[0]] if best[0] < len(run) else run[0]
            got = best[0]
            detail = {
                'label': label, 'run': ri + 1, 'n': len(run),
                'matched': got, 'file': hit_file or srcs[0],
                'first_bad': bad.strip()[:120],
            }
            problems.append(
                f"{label} 第 {ri + 1} 段（{len(run)} 行）只有前 {got} 行能在 "
                f"{hit_file or srcs[0]} 里连续命中；断点: {detail['first_bad']}")
    return ok_runs, len(rs)


def parse_source_md(path):
    """返回 [(srcs, code, label)]"""
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    out = []
    pending = None
    i = 0
    while i < len(lines):
        m = SRC_RE.search(lines[i])
        if m:
            pending = m.group(1)
            i += 1
            continue
        f = FENCE_RE.match(lines[i])
        if f:
            lang = f.group(1)
            j = i + 1
            body = []
            while j < len(lines) and not FENCE_RE.match(lines[j]):
                body.append(lines[j])
                j += 1
            if lang and lang.lower() not in ('text', 'txt', 'plain'):
                if pending is None:
                    out.append((None, '\n'.join(body), f'{os.path.relpath(path, HERE)} 代码块@{i + 1}'))
                else:
                    out.append((pending, '\n'.join(body), f'{os.path.relpath(path, HERE)} 代码块@{i + 1}'))
            pending = None
            i = j + 1
            continue
        i += 1
    return out


def parse_lesson_js(path):
    r = subprocess.run(['node', os.path.join(HERE, 'tools', 'dump_scenes.js'), path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip() or 'node 执行失败'
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        return None, f'JSON 解析失败: {e}'
    if 'parseError' in data:
        return None, data['parseError']
    return data['scenes'], None


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


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    quiet = '--quiet' in sys.argv
    show_ranges = '--ranges' in sys.argv
    # --all 是显式的"全量"开关；默认行为已经是全量
    _all = '--all' in sys.argv
    only = None
    if '--lesson' in sys.argv:
        only = sys.argv[sys.argv.index('--lesson') + 1]

    dirs = lesson_dirs(only)
    if not dirs:
        print('没有找到任何课件目录（空课件集：0 个引用块，门禁不误报）')
        return 0

    problems, ranges = [], []
    n_blocks = n_runs = n_ok = 0
    n_lessons = 0
    for d in dirs:
        rel = os.path.relpath(d, HERE)
        smd = os.path.join(d, 'source.md')
        ljs = os.path.join(d, 'lesson.js')
        if not os.path.isfile(smd) and not os.path.isfile(ljs):
            continue
        n_lessons += 1
        if os.path.isfile(smd):
            for srcs, code, label in parse_source_md(smd):
                if srcs is None:
                    problems.append(f'{label}: 缺少 <!-- src: 路径 --> 声明')
                    continue
                n_blocks += 1
                ok, tot = check_block(code, srcs, label, problems, ranges, show_ranges)
                n_ok += ok
                n_runs += tot
        if os.path.isfile(ljs):
            scenes, err = parse_lesson_js(ljs)
            if err:
                problems.append(f'{rel}/lesson.js: {err}')
                continue
            for s in scenes:
                if s['code'] is None:
                    continue
                label = f"{rel}/lesson.js 第 {s['i'] + 1} 幕"
                if not s['src']:
                    problems.append(f'{label}: 有 code 但缺少 src 声明')
                    continue
                n_blocks += 1
                ok, tot = check_block(s['code'], s['src'], label, problems, ranges, show_ranges)
                n_ok += ok
                n_runs += tot

    if show_ranges:
        print('== 命中区间 ==')
        for r in ranges:
            print('  ', r)
        print()

    if problems:
        print(f'✗ A 组失败：{len(problems)} 处问题 / {n_blocks} 个引用块')
        for p in problems[:60]:
            print('   -', p)
        if len(problems) > 60:
            print(f'   ... 另有 {len(problems) - 60} 处')
        return 1

    if not quiet:
        print(f'课件数       : {n_lessons}')
        print(f'引用块数     : {n_blocks}')
        print(f'连续段数     : {n_runs}（注解行切分后）')
        print(f'逐字命中段数 : {n_ok}')
    print(f'✓ A 组通过：{n_blocks} 个引用块 / {n_runs} 个连续段全部逐字且位置连续命中')
    return 0


if __name__ == '__main__':
    sys.exit(main())
