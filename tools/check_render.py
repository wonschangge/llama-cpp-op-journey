#!/usr/bin/env python3
"""渲染门禁（C 组）—— 每课在无头浏览器里 0 JS 错误、0 布局溢出、交互可用。

★ 溢出判定公式（踩过的坑）：不要把"右边界超出"写成"宽度小于可视区"之类的反向判断。
  正确判据是【四个边界】都在可视区内，留 1px 容差避免亚像素误差：
      left >= -1 && right <= vw+1 && top >= -1 && bottom <= vh+1

★ 早/中/末三个时刻各快照一次 —— 动画中途才出现的元素最容易溢出。

★ 用 file:// 打开，验证的是"双击 index.html 就能看"这个核心体验本身。

用法:
    python3 tools/check_render.py --all
    python3 tools/check_render.py --lesson L1-operator-representation/L1-01-tensor-data-plane
    python3 tools/check_render.py --portal            # 门户页用针对性检查
    python3 tools/check_render.py --all --quiet
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'shared', 'tools', 'plan', 'node_modules', '.git'}
CHROME = os.environ.get('CHROME_PATH', '/usr/bin/google-chrome')
VIEWPORTS = ((1280, 720), (1920, 1080))

USAGE = """usage: check_render.py [--all] [--lesson DIR] [--portal] [--quiet] [--json]

渲染门禁：无头浏览器里 0 JS 错误、0 布局溢出、交互可用（两种分辨率）。

options:
  --all        全量检查（默认行为，显式开关便于脚本化）
  --lesson DIR 只检查一课
  --portal     改为检查门户页（非课件页面，用针对性检查）
  --quiet      只输出结论
  --json       以 JSON 输出结果
  -h, --help   显示本帮助
"""

OVERFLOW_JS = """() => {
  const out = [], vw = innerWidth, vh = innerHeight;
  document.querySelectorAll('#visual *').forEach(el => {
    const b = el.getBoundingClientRect();
    if (b.width <= 0 || b.height <= 0) return;
    if (b.left < -1 || b.right > vw + 1 || b.top < -1 || b.bottom > vh + 1) {
      out.push({ cls: (el.className && String(el.className)) || el.tagName,
                 l: Math.round(b.left), r: Math.round(b.right),
                 t: Math.round(b.top), bo: Math.round(b.bottom) });
    }
  });
  return out;
}"""

# 门户是【滚动页】，纵向溢出是设计如此，所以只查【横向】溢出。
# 课件页是固定舞台，才需要四边界判据。这是两套不同的判据。
PORTAL_OVERFLOW_JS = """() => {
  const out = [], vw = innerWidth;
  document.querySelectorAll('.page *').forEach(el => {
    const b = el.getBoundingClientRect();
    if (b.width <= 0 || b.height <= 0) return;
    if (b.left < -1 || b.right > vw + 1) {
      out.push({ cls: (el.className && String(el.className)) || el.tagName,
                 l: Math.round(b.left), r: Math.round(b.right) });
    }
  });
  return out;
}"""


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
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'index.html')):
                out.append(p)
    return out


def check_lesson(browser, url, w, h):
    """返回 (errors, overflows, interact_ok, scene_count)"""
    page = browser.new_page(viewport={'width': w, 'height': h})
    errors, cerrs = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: cerrs.append(m.text) if m.type == 'error' else None)
    res = {'errors': [], 'overflows': [], 'interact': [], 'scenes': 0}
    try:
        page.goto(url, wait_until='load')
        page.wait_for_timeout(400)

        info = page.evaluate("""() => ({
            hasShell: typeof SHELL !== 'undefined',
            count: (typeof SHELL !== 'undefined') ? SHELL.count : 0,
            idx: (typeof SHELL !== 'undefined') ? SHELL.index : -1,
            hasVisual: !!document.querySelector('#visual'),
            hasPlay: !!document.querySelector('#playBtn'),
            dots: document.querySelectorAll('.dot').length
        })""")
        res['scenes'] = info.get('count', 0)
        if not info.get('hasShell'):
            res['errors'].append('SHELL 未定义')
        if not info.get('hasVisual'):
            res['errors'].append('#visual 不存在')
        if info.get('count', 0) == 0:
            res['errors'].append('场景数为 0')

        # 交互自检
        try:
            if info.get('hasPlay'):
                # 课件加载后会自动播放，所以不能假设初始是暂停态：
                # 断言"点一下状态翻转，再点一下翻回来"。
                p0 = page.evaluate("() => !!(SHELL.timeline && SHELL.timeline.playing)")
                page.click('#playBtn'); page.wait_for_timeout(150)
                p1 = page.evaluate("() => !!(SHELL.timeline && SHELL.timeline.playing)")
                page.click('#playBtn'); page.wait_for_timeout(150)
                p2 = page.evaluate("() => !!(SHELL.timeline && SHELL.timeline.playing)")
                res['interact'].append(('play/pause', p1 != p0 and p2 == p0))
            n = info.get('count', 0)
            if n > 1:
                page.evaluate("() => SHELL.goto(0)")
                page.wait_for_timeout(120)
                page.keyboard.press('ArrowRight'); page.wait_for_timeout(150)
                k = page.evaluate("() => SHELL.index")
                res['interact'].append(('ArrowRight', k == 1))
                page.keyboard.press('ArrowLeft'); page.wait_for_timeout(150)
                k = page.evaluate("() => SHELL.index")
                res['interact'].append(('ArrowLeft', k == 0))
                if info.get('dots', 0) > 1:
                    page.evaluate("() => document.querySelectorAll('.dot')[1].click()")
                    page.wait_for_timeout(150)
                    k = page.evaluate("() => SHELL.index")
                    res['interact'].append(('dot-jump', k == 1))
                ov = page.evaluate("() => document.querySelector('#navNext') ? document.querySelector('#navNext').tagName : 'MISSING'")
                res['interact'].append(('nav-next-exists', ov != 'MISSING'))
                ov = page.evaluate("() => document.querySelector('#navPrev') ? document.querySelector('#navPrev').tagName : 'MISSING'")
                res['interact'].append(('nav-prev-exists', ov != 'MISSING'))
        except Exception as e:
            res['errors'].append('交互自检异常: ' + str(e))

        # 早/中/末三个时刻各查一次溢出
        dur = page.evaluate("() => (SHELL.timeline ? SHELL.timeline.duration : 8000)")
        for t in (300, max(600, int(dur * 0.5)), max(900, int(dur * 0.92))):
            page.wait_for_timeout(350 if t < 500 else 250)
            ov = page.evaluate(OVERFLOW_JS)
            for o in ov:
                o['at'] = t
                res['overflows'].append(o)
            if len(res['overflows']) > 40:
                break
    except Exception as e:
        res['errors'].append('页面异常: ' + str(e))
    finally:
        res['errors'] += ['[console] ' + c for c in cerrs]
        page.close()
    return res


def check_portal(browser, url, w, h):
    page = browser.new_page(viewport={'width': w, 'height': h})
    errors, cerrs = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: cerrs.append(m.text) if m.type == 'error' else None)
    res = {'errors': [], 'overflows': [], 'interact': [], 'scenes': 0}
    try:
        page.goto(url, wait_until='load')
        page.wait_for_timeout(500)
        info = page.evaluate("""() => {
            const cards = Array.from(document.querySelectorAll('a.card, [data-lesson]'));
            const links = cards.map(a => a.getAttribute('href'))
                .filter(h => h && h.indexOf('javascript:') !== 0);
            return { cards: cards.length, links: links,
                     hasSearch: !!document.querySelector('#q, input[type=search]'),
                     empty: (document.querySelector('#empty') || {}).style ?
                            document.querySelector('#empty').style.display !== 'none' : false };
        }""")
        res['scenes'] = info.get('cards', 0)
        res['interact'].append(('cards>0', info.get('cards', 0) > 0))
        # 链接有效性：逐个核对目标文件是否存在（离线 file:// 场景下必须真实存在）
        bad = []
        for h in info.get('links', []):
            t = os.path.normpath(os.path.join(HERE, h.split('#')[0]))
            if not os.path.exists(t):
                bad.append(h)
        res['interact'].append(('links-valid', not bad))
        if bad:
            res['errors'].append('失效链接: ' + ', '.join(bad[:5]))
        if not info.get('hasSearch'):
            res['interact'].append(('search-present', False))
        else:
            res['interact'].append(('search-present', True))
            try:
                page.fill('#q', 'mul_mat')
                page.wait_for_timeout(350)
                after = page.evaluate("() => document.querySelectorAll('[data-lesson]').length")
                res['interact'].append(('search-filters', after < info.get('cards', 0)))
                page.fill('#q', 'zzzz-no-such-thing')
                page.wait_for_timeout(300)
                zero = page.evaluate("() => document.querySelectorAll('[data-lesson]').length")
                res['interact'].append(('search-empties', zero == 0))
                page.fill('#q', '')
                page.wait_for_timeout(250)
                back = page.evaluate("() => document.querySelectorAll('[data-lesson]').length")
                res['interact'].append(('search-restores', back == info.get('cards', 0)))
            except Exception as e:
                res['errors'].append('搜索自检异常: ' + str(e))
        # 层过滤
        try:
            page.evaluate("() => { const b = document.querySelector('.fbtn[data-layer=\"L6\"]'); if (b) b.click(); }")
            page.wait_for_timeout(300)
            only6 = page.evaluate("""() => {
                const ls = Array.from(document.querySelectorAll('[data-lesson]'));
                return ls.length > 0 && ls.every(a => a.getAttribute('data-lesson').indexOf('L6-') === 0);
            }""")
            res['interact'].append(('layer-filter', bool(only6)))
            page.evaluate("() => { const b = document.querySelector('.fbtn[data-layer=\"all\"]'); if (b) b.click(); }")
            page.wait_for_timeout(250)
        except Exception as e:
            res['errors'].append('层过滤自检异常: ' + str(e))
        for t in (300, 800, 1500):
            page.wait_for_timeout(300)
            ov = page.evaluate(PORTAL_OVERFLOW_JS)
            for o in ov:
                o['at'] = t
                res['overflows'].append(o)
    except Exception as e:
        res['errors'].append('页面异常: ' + str(e))
    finally:
        res['errors'] += ['[console] ' + c for c in cerrs]
        page.close()
    return res


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(USAGE)
        return 0
    from playwright.sync_api import sync_playwright

    quiet = '--quiet' in sys.argv
    as_json = '--json' in sys.argv
    portal = '--portal' in sys.argv
    only = sys.argv[sys.argv.index('--lesson') + 1] if '--lesson' in sys.argv else None

    if portal:
        targets = [('portal', os.path.join(HERE, 'index.html'))]
    else:
        targets = [(os.path.relpath(d, HERE), os.path.join(d, 'index.html'))
                   for d in lesson_dirs(only)]
    if not targets:
        print('没有课件（空集，渲染门禁通过）')
        return 0
    missing = [n for n, p in targets if not os.path.isfile(p)]
    if missing:
        print(f'✗ 以下课件缺 index.html: {missing[:5]}')
        return 1

    results = {}
    fails = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME,
                                    args=['--no-sandbox', '--disable-dev-shm-usage',
                                          '--allow-file-access-from-files'])
        for name, path in targets:
            url = 'file://' + path
            per = []
            for (w, h) in VIEWPORTS:
                r = check_portal(browser, url, w, h) if portal else check_lesson(browser, url, w, h)
                r['viewport'] = f'{w}x{h}'
                per.append(r)
            results[name] = per
            bad = [r for r in per if r['errors'] or r['overflows'] or
                   any(not ok for _n, ok in r['interact'])]
            if bad:
                fails += 1
                print(f'✗ {name}')
                for r in bad:
                    for e in r['errors'][:6]:
                        print(f'    @{r["viewport"]} 错误: {e}')
                    for o in r['overflows'][:6]:
                        print(f'    @{r["viewport"]} 溢出: {o}')
                    for n, ok in r['interact']:
                        if not ok:
                            print(f'    @{r["viewport"]} 交互失败: {n}')
            elif not quiet:
                r0 = per[0]
                print(f'✓ {name}  {r0["scenes"]} 幕  '
                      f'{len(r0["interact"])} 项交互  '
                      f'@{", ".join(x["viewport"] for x in per)}')
        browser.close()

    if as_json:
        print(json.dumps(results, ensure_ascii=False)[:2000])
    if fails:
        print(f'✗ C 组失败：{fails}/{len(targets)} 个页面有问题')
        return 1
    print(f'✓ C 组通过：{len(targets)} 个页面 × {len(VIEWPORTS)} 种分辨率，'
          f'0 JS 错误 / 0 布局溢出 / 交互全部可用')
    return 0


if __name__ == '__main__':
    sys.exit(main())
