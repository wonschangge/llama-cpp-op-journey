#!/usr/bin/env python3
"""生成门户 index.html —— 静态烘焙，零 fetch、零外链。

数据（课程清单 + 完成状态）在生成时直接写进 HTML，所以双击可开、离线可用。
支持按层过滤 + 关键词搜索，纯 DOM 操作，无网络请求。

用法:
    python3 tools/build_portal.py
"""

import os
import re
import sys
import json
import html

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
import repo_universe as ru          # noqa: E402
import plan_model as pm             # noqa: E402
import plan_matrix as pmat          # noqa: E402


def read_title(d):
    """从 README.md 或 lesson.js 取标题（若课件已生成）。"""
    r = os.path.join(d, 'README.md')
    if os.path.isfile(r):
        with open(r, encoding='utf-8') as fh:
            m = re.search(r'^#\s+(.+?)\s+—', fh.read(), re.M)
            if m:
                return m.group(1)
    return None


def scenes_of(d):
    ljs = os.path.join(d, 'lesson.js')
    if not os.path.isfile(ljs):
        return 0
    with open(ljs, encoding='utf-8') as fh:
        t = fh.read()
    return t.count('  kicker:')


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""usage: build_portal.py

生成门户 index.html（静态烘焙课程清单与完成状态，零 fetch、零外链）。

options:
  -h, --help  显示本帮助
""")
        return 0

    assign, unassigned = pmat.resolve()
    items = []
    done = 0
    for les in pm.LESSONS:
        d = os.path.join(HERE, pmat.lesson_dir(les))
        files = ('index.html', 'lesson.js', 'source.md', 'README.md')
        ok = all(os.path.isfile(os.path.join(d, f)) for f in files)
        nfiles = len(assign[les['id']])
        # 覆盖面按文件名摘要（前 3 个 + 计数）
        names = [p.split('/')[-1] for p in assign[les['id']][:3]]
        items.append(dict(
            id=les['id'], layer=les['layer'], prio=les['prio'],
            title=read_title(d) or les['title'],
            ideas=les['ideas'], accept=les['accept'],
            dir=pmat.lesson_dir(les), nfiles=nfiles, scenes=scenes_of(d),
            done=ok, names=names,
        ))
        done += ok

    layers = []
    for lid, name, desc in pm.LAYERS:
        n = sum(1 for i in items if i['layer'] == lid)
        dn = sum(1 for i in items if i['layer'] == lid and i['done'])
        nf = sum(i['nfiles'] for i in items if i['layer'] == lid)
        layers.append(dict(id=lid, name=name, desc=desc, n=n, done=dn, files=nf))

    data = json.dumps(dict(items=items, layers=layers,
                           total=len(items), done=done,
                           universe=len(ru.universe()),
                           tag=ru.UPSTREAM_TAG, commit=ru.UPSTREAM_COMMIT[:12]),
                      ensure_ascii=False)

    out = PORTAL_HTML.replace('/*__DATA__*/', 'var DATA = ' + data + ';')
    path = os.path.join(HERE, 'index.html')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(out)
    print(f'已写出 {path}：{done}/{len(items)} 课已发布，'
          f'{len(layers)} 层，覆盖域 {len(ru.universe())} 文件')
    return 0


PORTAL_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama.cpp 算子之旅 · 从模型定义到 CPU/GPU/NPU 后端执行</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230d1117'/%3E%3Cpath d='M7 22 L13 10 L19 22' stroke='%2358a6ff' stroke-width='2.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='24' cy='12' r='3' fill='%233fb950'/%3E%3C/svg%3E">
<link rel="stylesheet" href="shared/theme.css">
<style>
  html, body { overflow: auto; }
  .page { max-width: 1180px; margin: 0 auto; padding: 26px 20px 60px; }
  .hero h1 { font-size: 25px; margin: 0 0 8px; line-height: 1.3; }
  .hero .pitch { font-size: 13px; color: var(--muted); line-height: 1.65; max-width: 830px; }
  .hero .pitch b { color: var(--text); }
  .stats { display: flex; gap: 22px; margin: 20px 0 6px; flex-wrap: wrap; }
  .stat .v { font-family: var(--mono); font-size: 21px; color: var(--a); }
  .stat .l { font-size: 10px; color: var(--dim); letter-spacing: .06em; text-transform: uppercase; }
  .bar-wrap { margin: 18px 0 24px; }
  .bar-outer { height: 7px; background: #10151b; border: 1px solid var(--border);
               border-radius: 5px; overflow: hidden; }
  .bar-inner { height: 100%; width: 0; background: linear-gradient(90deg, var(--a), var(--f));
               transition: width .5s; }
  .bar-cap { font-size: 10px; color: var(--dim); margin-top: 5px; }
  .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
             margin: 20px 0 14px; position: sticky; top: 0; background: var(--bg);
             padding: 10px 0; z-index: 5; border-bottom: 1px solid var(--border); }
  .fbtn { font-size: 11px; padding: 5px 11px; border-radius: 999px;
          border: 1px solid var(--border); background: var(--panel);
          color: var(--muted); cursor: pointer; font-family: inherit; white-space: nowrap; }
  .fbtn:hover { border-color: var(--border-hi); color: var(--text); }
  .fbtn.on { background: rgba(88,166,255,.14); border-color: var(--a); color: var(--a); }
  #q { flex: 1 1 220px; min-width: 160px; font-size: 12px; padding: 6px 11px;
       border-radius: 7px; border: 1px solid var(--border); background: #10151b;
       color: var(--text); font-family: inherit; }
  #q:focus { outline: none; border-color: var(--a); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: 11px; }
  a.card { display: block; text-decoration: none; color: inherit;
           border-left-width: 3px; }
  a.card:hover { background: #1f2630; border-color: var(--border-hi); }
  a.card.todo { opacity: .48; cursor: default; }
  a.card .top { display: flex; align-items: center; gap: 7px; margin-bottom: 4px; }
  a.card .lid { font-family: var(--mono); font-size: 10px; color: var(--a); }
  a.card .ct { flex: 1 1 auto; font-size: 12px; }
  a.card .cb { font-size: 10px; color: var(--muted); line-height: 1.5; }
  a.card .meta { font-family: var(--mono); font-size: 9px; color: var(--dim); margin-top: 5px; }
  .badge { font-size: 9px; padding: 1px 6px; border-radius: 999px;
           border: 1px solid var(--border-hi); color: var(--dim); white-space: nowrap; }
  .badge.ok { color: var(--b); border-color: rgba(63,185,80,.5); }
  .empty { color: var(--dim); font-size: 12px; padding: 30px 0; text-align: center; }
  footer { margin-top: 40px; font-size: 10.5px; color: var(--dim); line-height: 1.7;
           border-top: 1px solid var(--border); padding-top: 16px; }
  footer a { color: var(--muted); }
</style>
</head>
<body>
<div class="page">
  <div class="hero">
    <h1>llama.cpp 算子之旅</h1>
    <p class="pitch">
      <b>一个视角</b>：一个算子，从模型定义出发，如何落到 CPU / GPU / NPU 上真正执行。
      这套课件逐层拆解它在 <b>ggml</b> 里怎么被表示、怎么从模型定义被建造成一张图、
      怎么被调度器切给不同后端、最后在 CUDA / Vulkan / Metal / SYCL / CANN /
      Hexagon / OpenVINO 等后端上变成真实的 kernel。
    </p>
    <div class="stats" id="stats"></div>
    <div class="bar-wrap">
      <div class="bar-outer"><div class="bar-inner" id="pbar"></div></div>
      <div class="bar-cap" id="pcap"></div>
    </div>
  </div>

  <div class="toolbar">
    <button class="fbtn on" data-layer="all">全部</button>
    <span id="layerbtns"></span>
    <input id="q" type="search" placeholder="搜索：课号 / 主题 / 文件名（如 mul_mat、kv-cache、backend）">
  </div>

  <div class="grid" id="grid"></div>
  <div class="empty" id="empty" style="display:none">没有匹配的课</div>

  <footer>
    <div>上游：<a href="https://github.com/ggml-org/llama.cpp">ggml-org/llama.cpp</a>
      @ <span id="tag"></span>。课件中的代码引用逐字来自上游源文件，由四道门禁脚本校验
      （覆盖度 / 保真 / 参数有效性 / 渲染）。</div>
    <div>许可：课件正文与引擎以 MIT 发布；引用的上游代码片段版权归 llama.cpp 贡献者所有，
      见 <a href="https://github.com/wonschangge/llama-cpp-op-journey/blob/main/NOTICE">NOTICE</a>。</div>
    <div>离线可用：零 CDN、零构建、零网络请求。clone 后直接双击任意一课 <code>index.html</code> 即可。</div>
  </footer>
</div>

<script>
/*__DATA__*/
(function () {
  'use strict';
  var cur = 'all', kw = '';
  var LAYER_COLOR = { L1:'a', L2:'b', L3:'c', L4:'d', L5:'e', L6:'f', L7:'g', L8:'a' };

  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  function renderStats() {
    var s = document.getElementById('stats');
    var f = [
      [DATA.done + '/' + DATA.total, '已发布课件'],
      [String(DATA.layers.length), '分层'],
      [String(DATA.universe), '覆盖域源文件'],
      ['4', '道门禁']
    ];
    f.forEach(function (x) {
      var d = el('div', 'stat');
      d.appendChild(el('div', 'v', x[0]));
      d.appendChild(el('div', 'l', x[1]));
      s.appendChild(d);
    });
    var pct = DATA.total ? Math.round(DATA.done / DATA.total * 100) : 0;
    document.getElementById('pbar').style.width = pct + '%';
    document.getElementById('pcap').textContent =
      '进度 ' + DATA.done + ' / ' + DATA.total + ' 课（' + pct + '%）';
    document.getElementById('tag').textContent = DATA.tag + ' (' + DATA.commit + ')';
  }

  function renderLayerBtns() {
    var host = document.getElementById('layerbtns');
    DATA.layers.forEach(function (l) {
      var b = el('button', 'fbtn');
      b.setAttribute('data-layer', l.id);
      b.textContent = l.id + ' ' + l.name + ' (' + l.done + '/' + l.n + ')';
      b.title = l.desc + ' —— ' + l.files + ' 个源文件';
      host.appendChild(b);
    });
    host.querySelectorAll('.fbtn').forEach(function (b) {
      b.addEventListener('click', function () { setLayer(b.getAttribute('data-layer')); });
    });
  }

  function setLayer(l) {
    cur = l;
    document.querySelectorAll('.fbtn').forEach(function (b) {
      b.classList.toggle('on', b.getAttribute('data-layer') === l);
    });
    render();
  }

  function match(it) {
    if (cur !== 'all' && it.layer !== cur) return false;
    if (!kw) return true;
    var hay = (it.id + ' ' + it.title + ' ' + it.ideas + ' ' + it.accept + ' ' +
               it.names.join(' ') + ' ' + it.dir).toLowerCase();
    return hay.indexOf(kw) >= 0;
  }

  function render() {
    var g = document.getElementById('grid');
    while (g.firstChild) g.removeChild(g.firstChild);
    var n = 0;
    DATA.items.forEach(function (it) {
      if (!match(it)) return;
      n++;
      var a = el('a', 'card' + (it.done ? '' : ' todo'));
      a.setAttribute('data-lesson', it.id);
      a.href = it.done ? it.dir + '/index.html' : 'javascript:void(0)';
      a.style.borderLeftColor = 'var(--' + (LAYER_COLOR[it.layer] || 'a') + ')';

      var top = el('div', 'top');
      top.appendChild(el('span', 'lid', it.id));
      top.appendChild(el('span', 'ct', it.title));
      var bd = el('span', 'badge' + (it.done ? ' ok' : ''),
                  it.done ? ('已发布 ' + it.scenes + ' 幕') : '计划中');
      top.appendChild(bd);
      a.appendChild(top);

      a.appendChild(el('div', 'cb', it.ideas));
      a.appendChild(el('div', 'meta',
        it.layer + ' · ' + it.prio + ' · 覆盖 ' + it.nfiles + ' 个源文件' +
        (it.names.length ? '  [' + it.names.join(', ') + (it.nfiles > 3 ? ', ...' : '') + ']' : '')));
      g.appendChild(a);
    });
    document.getElementById('empty').style.display = n ? 'none' : '';
  }

  document.getElementById('q').addEventListener('input', function (e) {
    kw = e.target.value.trim().toLowerCase();
    render();
  });
  document.querySelectorAll('.fbtn[data-layer]').forEach(function (b) {
    b.addEventListener('click', function () { setLayer(b.getAttribute('data-layer')); });
  });

  renderStats();
  renderLayerBtns();
  render();
})();
</script>
</body>
</html>
'''

if __name__ == '__main__':
    sys.exit(main())
