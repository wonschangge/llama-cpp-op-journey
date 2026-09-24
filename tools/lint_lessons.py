#!/usr/bin/env python3
"""静态语法检查 —— 秒级，放在整个流程最前面。

为什么需要：`style: 'gap':13px` 这类引号位置笔误会造成语法错误、课件白屏。
渲染门禁虽能发现，但要启动浏览器（慢）。本检查不需要浏览器。

检查项：
  L1  每个课件目录四个文件齐备（index.html / lesson.js / source.md / README.md）
  L2  lesson.js 通过 node --check（语法）
  L3  SCENES 能求值且非空；每个有 code 的 scene 必须声明 src
  L4  index.html 引入了四个 shared 脚本与主题
  L5  favicon 是内联 data URI（避免离线 404）
  L6  零外链：没有 http(s) 的 script/link/img 引用
  L7  导航链指向的目录真实存在
  L8  README.md 声明前置课，且编号严格小于本课

用法:
    python3 tools/lint_lessons.py [--all] [--lesson DIR] [--quiet]
"""

import os
import re
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'shared', 'tools', 'plan', 'node_modules', '.git'}
FILES = ('index.html', 'lesson.js', 'source.md', 'README.md')

USAGE = """usage: lint_lessons.py [--all] [--lesson DIR] [--quiet]

静态语法检查（秒级）：文件齐备、JS 语法、引擎引用、零外链、导航链、前置课。

options:
  --all        全量检查（默认行为，显式开关便于脚本化）
  --lesson DIR 只检查一课
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
            if os.path.isdir(p) and any(os.path.isfile(os.path.join(p, f)) for f in FILES):
                out.append(p)
    return out


def _planned_dirs():
    """计划里的全部课件目录（相对仓库根）。用于区分「还没写」与「写错了」。"""
    try:
        sys.path.insert(0, os.path.join(HERE, 'tools'))
        import plan_model as pm
        import plan_matrix as pmat
        return {pmat.lesson_dir(l) for l in pm.LESSONS}
    except Exception:
        return set()


PLANNED = _planned_dirs()


def lint_one(d):
    """返回 (lesson_id, [错误], [警告])"""
    rel = os.path.relpath(d, HERE)
    lid = os.path.basename(d).split('-')[0] + '-' + os.path.basename(d).split('-')[1] \
        if '-' in os.path.basename(d) else os.path.basename(d)
    errs, warns = [], []

    for f in FILES:
        if not os.path.isfile(os.path.join(d, f)):
            errs.append(f'缺文件 {f}')

    ljs = os.path.join(d, 'lesson.js')
    if os.path.isfile(ljs):
        r = subprocess.run(['node', '--check', ljs], capture_output=True, text=True)
        if r.returncode != 0:
            errs.append('JS 语法错误: ' + (r.stderr.strip().split('\n')[0:3] and
                                           ' | '.join(r.stderr.strip().split('\n')[:3])))
        else:
            r2 = subprocess.run(['node', os.path.join(HERE, 'tools', 'dump_scenes.js'), ljs],
                                capture_output=True, text=True)
            try:
                data = json.loads(r2.stdout)
            except Exception:
                data = {'parseError': 'dump_scenes 输出无法解析'}
            if 'parseError' in data:
                errs.append('SCENES 求值失败: ' + data['parseError'])
            else:
                scenes = data.get('scenes', [])
                if not scenes:
                    errs.append('SCENES 为空')
                if len(scenes) < 4:
                    warns.append(f'只有 {len(scenes)} 幕（建议 >= 4：首幕给全局、末幕做收束）')
                for s in scenes:
                    if s['code'] and not s['src']:
                        errs.append(f"第 {s['i'] + 1} 幕有 code 但没有 src 声明")
                    if s['code'] and s['code'].count('\n') > 60:
                        warns.append(f"第 {s['i'] + 1} 幕代码 {s['len']} 行，偏长（建议 <= 60）")

    idx = os.path.join(d, 'index.html')
    if os.path.isfile(idx):
        with open(idx, encoding='utf-8') as fh:
            html = fh.read()
        for need in ('shared/engine.js', 'shared/widgets.js', 'shared/lesson-shell.js',
                     'lesson.js', 'shared/theme.css'):
            if need not in html:
                errs.append(f'index.html 缺少引用: {need}')
        if 'SHELL.boot' not in html:
            errs.append('index.html 没有调用 SHELL.boot')
        if not re.search(r'rel=["\']icon["\'][^>]*href=["\']data:', html):
            errs.append('favicon 不是内联 data URI（离线会 404）')
        for m in re.finditer(r'(?:src|href)=["\'](https?://[^"\']+)', html):
            errs.append(f'存在外链（违反离线优先）: {m.group(1)}')
        for m in re.finditer(r'(?:id=["\']nav(?:Prev|Next)["\'][^>]*href|href)=["\']([^"\']+)',
                             html):
            href = m.group(1)
            if href.startswith(('http', 'data:', '#')):
                continue
            target = os.path.normpath(os.path.join(d, href))
            if os.path.exists(target):
                continue
            # 目标是「计划中但尚未撰写」的课 -> 只是警告，不是错误。
            # 否则链接指向的就是真错（或最终交付时仍未补齐）。
            tdir = os.path.dirname(target)
            trel = os.path.relpath(tdir, HERE)
            if trel in PLANNED or trel.rstrip('/') in PLANNED:
                warns.append(f'导航指向计划中但尚未撰写的课: {href}')
            else:
                errs.append(f'导航链接指向不存在且不在计划内: {href}')

    rdm = os.path.join(d, 'README.md')
    if os.path.isfile(rdm):
        with open(rdm, encoding='utf-8') as fh:
            txt = fh.read()
        if '前置课' not in txt:
            warns.append('README.md 未声明前置课')
        else:
            m = re.search(r'前置课[^\n]*', txt)
            my = re.search(r'L(\d+)-(\d+)', os.path.basename(d))
            if m and my:
                for pm in re.finditer(r'L(\d+)-(\d+)', m.group(0)):
                    if (int(pm.group(1)), int(pm.group(2))) >= (int(my.group(1)), int(my.group(2))):
                        errs.append(f'前置课编号不小于本课: {pm.group(0)}')

    return lid, errs, warns


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    quiet = '--quiet' in sys.argv
    only = sys.argv[sys.argv.index('--lesson') + 1] if '--lesson' in sys.argv else None
    dirs = lesson_dirs(only)
    if not dirs:
        print('没有课件（空集，lint 通过）')
        return 0

    total_err = 0
    total_warn = 0
    for d in dirs:
        lid, errs, warns = lint_one(d)
        total_err += len(errs)
        total_warn += len(warns)
        if errs or (warns and not quiet):
            print(f'--- {os.path.relpath(d, HERE)}')
            for e in errs:
                print(f'   ✗ {e}')
            for w in warns:
                print(f'   ! {w}')
    if total_err:
        print(f'✗ lint 失败：{len(dirs)} 课，{total_err} 个错误，{total_warn} 个警告')
        return 1
    print(f'✓ lint 通过：{len(dirs)} 课，0 错误，{total_warn} 个警告')
    return 0


if __name__ == '__main__':
    sys.exit(main())
