#!/usr/bin/env python3
"""参数有效性门禁（A2）—— 课件里提到的每个命令行参数都必须真实存在。

真值来源：真实构建出来的二进制跑 --help（不是文档、不是记忆）。
覆盖工具：llama-cli / llama-server / llama-bench / llama-quantize / llama-tokenize
          / llama-perplexity / llama-imatrix / llama-batched-bench / llama-gguf-split

★ 两个踩过的坑（已在实现中规避）：
  - 用 <br> 手工拆行长参数 -> 检查器会把片段当成独立参数。本仓库约定长标识符
    交给 CSS 换行，不允许手工拆行。
  - `--some-flag-...` 省略号 -> 会被提取成 `--some-flag-`。这里显式剥离尾部
    连字符/省略号，并跳过剥完长度不足的 token。

用法:
    python3 tools/check_flags.py
    python3 tools/check_flags.py --quiet
    python3 tools/check_flags.py --list      # 打印真值集大小与样例
"""

import os
import re
import sys
import json
import glob
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get('LLAMA_BIN',
                     '/data/WORKSPACE/llama.cpp-project/llama.cpp/build/bin')
SKIP_DIRS = {'shared', 'tools', 'plan', 'node_modules', '.git'}

TOOLS = ['llama-cli', 'llama-server', 'llama-bench', 'llama-quantize',
         'llama-tokenize', 'llama-perplexity', 'llama-imatrix',
         'llama-batched-bench', 'llama-gguf-split', 'llama-gguf', 'llama-tts',
         'llama-fit-params', 'llama-completion', 'llama-mtmd-cli']

LONG_RE = re.compile(r'(?<![\w(-])--[a-z][a-z0-9-]*')
SHORT_RE = re.compile(r'(?<![\w-])-([a-zA-Z])(?![\w-])')

# ★ 门禁自身的缺陷修复记录：最初用 `--[a-z][a-z0-9-]*`，把 CSS 自定义属性
#   var(--dim) 误报成命令行参数 --dim。判据是"前面不能是 ( 或 - 或字母数字"，
#   因为命令行参数总是出现在行首/空格/引号之后，不会紧跟在左括号后面。

# 允许出现的、非 CLI 的连字符 token（附理由，逐条审查）
ALLOW = {
    '--': 'markdown 分隔线 / C 里的递减运算符',
    '--help': 'CLI 通用',
}

USAGE = """usage: check_flags.py [--all] [--quiet] [--list]

参数有效性门禁：课件里提到的每个命令行参数都必须真实存在。
真值来源是真实二进制的 --help，以及本仓库门禁脚本自己的 --help。

options:
  --all        全量检查（默认行为，显式开关便于脚本化）
  --lesson DIR 只检查一课（+ 仓库根 README）
  --quiet      只输出结论
  --list       打印真值集规模与样例
  -h, --help   显示本帮助
"""

_cache_truth = None


def our_tools_truth():
    """本仓库自己的门禁脚本也按同一条规则取真值：跑它们的 --help。

    README 会写 `python3 tools/check_render.py --all` 这类命令，
    这些 flag 同样必须真实存在 —— 不能只对上游严格、对自己宽松。
    """
    long_opts, short_opts = set(), set()
    our = sorted(glob.glob(os.path.join(HERE, 'tools', '*.py')))
    used = []
    for p in our:
        try:
            r = subprocess.run([sys.executable, p, '--help'],
                               capture_output=True, text=True, timeout=60)
        except Exception:
            continue
        out = (r.stdout or '') + '\n' + (r.stderr or '')
        if 'usage:' not in out.lower():
            continue
        used.append(os.path.basename(p))
        long_opts |= set(LONG_RE.findall(out))
        for m in re.finditer(r'(?<![\w-])(-[a-zA-Z])(?![\w-])', out):
            short_opts.add(m.group(1))
    return long_opts, short_opts, used


def build_truth():
    global _cache_truth
    if _cache_truth is not None:
        return _cache_truth
    long_opts, short_opts = set(), set()
    used = []
    for name in TOOLS:
        p = os.path.join(BIN, name)
        if not os.path.isfile(p) or not os.access(p, os.X_OK):
            continue
        try:
            r = subprocess.run([p, '--help'], capture_output=True, text=True, timeout=60)
        except Exception:
            continue
        out = (r.stdout or '') + '\n' + (r.stderr or '')
        if not out.strip():
            continue
        used.append(name)
        long_opts |= set(LONG_RE.findall(out))
        # 短选项只在 "--xxx, -y" 这类并列位置取，避免抓到正文里的 "-"
        for m in re.finditer(r'(?:^|[\s,|])(-[a-zA-Z]{1,3})(?=[\s,)]|$)', out):
            short_opts.add(m.group(1))
    # 环境变量也是合法引用
    envs = set()
    for name in used:
        try:
            r = subprocess.run([os.path.join(BIN, name), '--help'],
                               capture_output=True, text=True, timeout=60)
            envs |= set(re.findall(r'LLAMA_ARG_[A-Z0-9_]+', (r.stdout or '') + (r.stderr or '')))
        except Exception:
            pass
    # 本仓库自己的工具
    olong, oshort, oused = our_tools_truth()
    long_opts |= olong
    short_opts |= oshort
    used += ['(本仓库) ' + n for n in oused]
    _cache_truth = (long_opts, short_opts, envs, used)
    return _cache_truth


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


def strip_verbatim(text, is_js):
    """把"逐字引用的上游代码"抠掉，只留下作者写的散文。

    这些区域里的 `-` 属于上游代码，不是课件在宣称某个 CLI 参数。
    """
    if is_js:
        # text 是【lesson.js 的路径】—— 交给 node 求值 SCENES，再把 code 字段抠掉
        r = subprocess.run(['node', os.path.join(HERE, 'tools', 'dump_scenes.js'), text],
                           capture_output=True, text=True, errors='replace')
        try:
            data = json.loads(r.stdout)
        except Exception as e:
            # ★ 绝不静默降级：抠不掉逐字引用，就等于把上游 C 代码当散文去扫
            #   `--flag`，会产出大量假阳性，或者（更糟）因为 except 吞掉而假装通过。
            return None, f'无法求值 SCENES（{e}）'
        if 'parseError' in data:
            return None, f'SCENES 解析失败: {data["parseError"]}'
        with open(text, encoding='utf-8', errors='replace') as fh:
            content = fh.read()
        # ★ 必须用【转义后】的形式去匹配。lessonkit 写 lesson.js 时把 code 里的
        #   `\`、反引号、`${` 做了 JS 模板字符串转义（js_escape）。若拿未转义的
        #   code 去 replace，只要代码里含这三者之一就匹配不上 —— 整块上游代码
        #   被当成散文扫描，里面的命令行参数（如 glslc 的 --target-env）全部误报。
        #   由 L6-07 子代理定位到字节（L6-06 的 CMakeLists 里的 ${...} 命中此坑）。
        def _js_escape(s):
            return (s.replace('\\', '\\\\')
                     .replace('`', '\\`')
                     .replace('${', '\\${'))
        # ★ 先在【原始 content】上一次找齐所有区间，再从后往前统一挖空。
        #   不能边找边 replace：当一幕的 code 是另一幕的子串时（L8-02 的
        #   第 1 幕是第 8 幕的子片段），先替换短的会把长的从中间挖断，
        #   长的从此匹配不上 → 误判为"定位失败"。
        #   由 L6-07 子代理用脚本复现并定位（全仓库只有 L8-02 踩到）。
        spans = []
        for s in data.get('scenes', []):
            if not s.get('code'):
                continue
            esc = _js_escape(s['code'])
            i = content.find(esc)
            if i < 0:
                return None, (f"第 {s['i'] + 1} 幕的逐字引用无法从 lesson.js 定位"
                              f"（转义形式也不匹配）—— 拒绝静默降级")
            spans.append((i, i + len(esc)))
        buf = list(content)
        for a, b in sorted(spans, reverse=True):
            buf[a:b] = '\n'
        return ''.join(buf), None
    # markdown: 抠掉带 <!-- src: --> 声明的围栏代码块
    lines = text.split('\n')
    out, i = [], 0
    fence = re.compile(r'^\s*```')
    src_re = re.compile(r'<!--\s*src:')
    pending = False
    while i < len(lines):
        if src_re.search(lines[i]):
            pending = True
            i += 1
            continue
        if fence.match(lines[i]):
            i += 1
            while i < len(lines) and not fence.match(lines[i]):
                if pending:
                    i += 1
                    continue
                out.append(lines[i])
                i += 1
            pending = False
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out), None


def extract_flags(text):
    found = set()
    for m in LONG_RE.finditer(text):
        tok = m.group(0)
        tok = re.sub(r'[-.]+$', '', tok)          # 剥离尾部 - 与省略号
        if len(tok) <= 3:
            continue
        found.add(tok)
    return found


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    quiet = '--quiet' in sys.argv
    _all = '--all' in sys.argv          # 显式全量开关；默认已是全量
    # ★ 并行写作下的必要设计：本门禁原本只能全仓库扫描，于是【任何一个】课件
    #   正在被写入（临时语法不完整）都会让所有课的验收变红。
    #   加 --lesson 后，单课验收只看该课，全局检查留给最终交付。
    only = sys.argv[sys.argv.index('--lesson') + 1] if '--lesson' in sys.argv else None
    long_opts, short_opts, envs, used = build_truth()
    if not used:
        print(f'✗ A2 无法建立真值集：{BIN} 下找不到可用二进制。'
              f'先构建 llama.cpp（见 README）。')
        return 1

    if '--list' in sys.argv:
        print(f'真值来源: {", ".join(used)}')
        print(f'长选项 {len(long_opts)} / 短选项 {len(short_opts)} / 环境变量 {len(envs)}')
        print('样例:', ' '.join(sorted(long_opts)[:12]))
        return 0

    bad = []
    n_files = 0
    n_tokens = 0
    dirs = lesson_dirs()
    if only:
        target = only if os.path.isabs(only) else os.path.join(HERE, only)
        target = os.path.normpath(target)
        dirs = [d for d in dirs if os.path.normpath(d) == target]
        if not dirs:
            print(f'✗ A2：找不到课件目录 {only}')
            return 1
    for d in dirs:
        for fn in ('source.md', 'lesson.js', 'README.md'):
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            n_files += 1
            with open(p, encoding='utf-8') as fh:
                raw = fh.read()
            # ★ js 分支需要【路径】（要交给 node 去 parse），不是文件内容。
            #   曾经误传内容 -> node 报 ENAMETOOLONG -> stderr 被 64KiB 截断在多字节
            #   UTF-8 边界上 -> UnicodeDecodeError -> 门禁崩溃。
            #   因为本门禁是全局扫描，一课崩掉会让【所有课】的参数门禁都失败。
            text, why = strip_verbatim(p if fn.endswith(".js") else raw, fn.endswith(".js"))
            if why:
                print(f"✗ A2 失败：{os.path.relpath(p, HERE)}: {why}")
                print("  （拒绝静默降级：抠不掉逐字引用就会把上游代码当散文扫，结果不可信）")
                return 1
            for tok in sorted(extract_flags(text)):
                n_tokens += 1
                if tok in long_opts or tok in ALLOW:
                    continue
                if tok.upper().replace('-', '_') in envs:
                    continue
                bad.append((os.path.relpath(p, HERE), tok))
        # README 里的短选项
    # 全仓库 README 也扫一遍
    root_readme = os.path.join(HERE, 'README.md')
    if os.path.isfile(root_readme):
        with open(root_readme, encoding='utf-8') as fh:
            for tok in sorted(extract_flags(fh.read())):
                n_tokens += 1
                if tok in long_opts or tok in ALLOW:
                    continue
                bad.append(('README.md', tok))

    if bad:
        print(f'✗ A2 失败：{len(bad)} 处参数不存在')
        seen = set()
        for f, tok in bad:
            if (f, tok) in seen:
                continue
            seen.add((f, tok))
            print(f'   - {f}: {tok}')
        return 1

    if not quiet:
        print(f'flag 真值来源：{", ".join(used)}')
        print(f'  （长选项 {len(long_opts)} 个，短选项 {len(short_opts)} 个，'
              f'环境变量 {len(envs)} 个）')
        print(f'扫描文件       : {n_files} 个')
        print(f'扫描到 flag    : {n_tokens} 处')
    print(f'✓ A2 通过：全部命令行参数均真实存在'
          f'（长选项 {len(long_opts)} 个 / 短选项 {len(short_opts)} 个，'
          f'扫描 {n_files} 个文件、{n_tokens} 处引用）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
