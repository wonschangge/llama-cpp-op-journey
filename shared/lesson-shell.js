/* ==========================================================================
   llama.cpp 算子之旅 · 课件外壳
   --------------------------------------------------------------------------
   SHELL.boot(SCENES, options)
     options = {
       kicker, codeCap,
       nav:   { prev: {href, label} | null, next: {href, label} | null },
       autoAdvance: true|false
     }
   提供：上一课/下一课、播放/暂停、重播、进度圆点跳转、方向键切幕。
   ========================================================================== */
'use strict';

var SHELL = (function () {

  var scenes = [], opts = {}, cur = -1, tl = null;
  var els = {};

  function boot(list, options) {
    scenes = (typeof list !== 'undefined' && list) ? list : (window.SCENES || []);
    opts = options || {};
    document.title = (opts.title || document.title);

    if (!scenes.length) {
      document.body.appendChild(W.errScenes());
      return;
    }
    try { build(); } catch (e) {
      document.body.appendChild(W.errLayout(e && e.message));
      return;
    }
    goto(0);
  }

  function build() {
    var head = U.el('div', { class: 'shell-head' },
      U.el('span', { class: 'kicker', text: opts.kicker || '' }),
      U.el('span', { class: 'ltitle', text: opts.title || document.title }));

    var nav = U.el('div', { class: 'navlinks' });
    var p = opts.nav && opts.nav.prev, nx = opts.nav && opts.nav.next;
    if (p && p.href) nav.appendChild(U.el('a', { id: 'navPrev', href: p.href, text: '← ' + (p.label || '上一课') }));
    else             nav.appendChild(U.el('span', { id: 'navPrev', class: 'disabled', text: '← 上一课' }));
    if (nx && nx.href) nav.appendChild(U.el('a', { id: 'navNext', href: nx.href, text: (nx.label || '下一课') + ' →' }));
    else               nav.appendChild(U.el('span', { id: 'navNext', class: 'disabled', text: '下一课 →' }));
    head.appendChild(nav);

    els.code = U.el('div', { class: 'code', id: 'code' });

    els.kicker  = U.el('div', { class: 'scene-kicker' });
    els.title   = U.el('div', { class: 'scene-title' });
    els.sub     = U.el('div', { class: 'scene-sub' });
    els.body    = U.el('div', { class: 'grow col', id: 'sceneBody' });
    els.caption = U.el('div', { class: 'scene-caption' });
    els.visual  = U.el('div', { id: 'visual' },
      els.kicker, els.title, els.sub, els.body, els.caption);

    els.play = U.el('button', { class: 'ctl', id: 'playBtn', text: '暂停' });
    els.play.addEventListener('click', toggle);
    var replay = U.el('button', { class: 'ctl', id: 'replayBtn', text: '重播' });
    replay.addEventListener('click', function () { goto(cur); });
    els.dots = U.el('div', { class: 'dots', id: 'dots' });
    els.tinfo = U.el('span', { class: 'tinfo', id: 'tinfo', text: '' });

    var shell = U.el('div', { class: 'shell' },
      head,
      U.el('div', { class: 'main' },
        U.el('div', { class: 'pane' },
          U.el('div', { class: 'pane-head', text: opts.codeCap || '源码（逐字引用）' }),
          els.code),
        U.el('div', { class: 'pane', style: 'flex:1 1 auto' }, els.visual)),
      U.el('div', { class: 'shell-foot' }, els.play, replay, els.dots, els.tinfo));

    document.body.appendChild(shell);

    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', function () { fitCode(); });
  }

  function onKey(ev) {
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') { goto(cur + 1); ev.preventDefault(); }
    else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') { goto(cur - 1); ev.preventDefault(); }
    else if (ev.key === ' ' || ev.code === 'Space') { toggle(); ev.preventDefault(); }
    else if (ev.key === 'Home') { goto(0); ev.preventDefault(); }
    else if (ev.key === 'End') { goto(scenes.length - 1); ev.preventDefault(); }
  }

  function toggle() {
    if (!tl) return;
    if (tl.playing) { tl.pause(); els.play.textContent = '播放'; }
    else {
      if (tl.t >= tl.duration) { tl.seek(0); }
      tl.play(); els.play.textContent = '暂停';
    }
  }

  function buildDots() {
    U.clear(els.dots);
    scenes.forEach(function (s, i) {
      var d = U.el('div', { class: 'dot', title: (i + 1) + '. ' + (s.title || '') });
      d.addEventListener('click', function () { goto(i); });
      els.dots.appendChild(d);
    });
  }

  function paintDots() {
    var ds = els.dots.querySelectorAll('.dot');
    for (var i = 0; i < ds.length; i++) {
      ds[i].className = 'dot' + (i === cur ? ' on' : (i < cur ? ' done' : ''));
    }
  }

  function goto(i) {
    if (i < 0 || i >= scenes.length) return;
    cur = i;
    var s = scenes[i];

    els.kicker.textContent = s.kicker || opts.kicker || '';
    els.title.innerHTML = s.title || '';
    els.sub.textContent = s.sub || '';
    els.caption.textContent = s.caption || '';

    /* 代码区：逐字引用（行号槽用上游真实行号，见 lessonkit 的 lineNo） */
    if (s.code) {
      els.code.innerHTML = U.codeBlock(s.code, s.mark || null, s.lineNo);
      els.code.style.display = '';
    } else {
      els.code.innerHTML = '';
      els.code.style.display = 'none';
    }

    U.clear(els.body);
    if (tl) { tl.pause(); tl = null; }

    tl = new U.Timeline();
    tl.duration = Math.max(s.duration || 12000, 1200);
    tl.onTick = function (t) {
      var pct = Math.min(100, Math.round(t / tl.duration * 100));
      els.tinfo.textContent = (cur + 1) + '/' + scenes.length + '  ' + pct + '%';
    };
    tl.onEnd = function () {
      els.play.textContent = '播放';
      if (opts.autoAdvance !== false && cur + 1 < scenes.length) {
        setTimeout(function () { goto(cur + 1); }, 600);
      }
    };
    var built = true;
    try { if (typeof s.build === 'function') s.build(els.body, tl); }
    catch (e) { built = false; U.clear(els.body); els.body.appendChild(W.errScene(i, e)); }

    paintDots();
    fitCode();
    if (built) { els.play.textContent = '暂停'; tl.play(); }
    else { els.play.textContent = '播放'; }
  }

  /* 代码区按最长行做最小自适应（不做手工 <br>，靠 CSS 换行/滚动） */
  function fitCode() {
    var longest = 0;
    var lines = els.code.textContent.split('\n');
    for (var i = 0; i < lines.length; i++) if (lines[i].length > longest) longest = lines[i].length;
    var fs = 10.5;
    if (longest > 96) fs = 8.4;
    else if (longest > 84) fs = 9.0;
    else if (longest > 74) fs = 9.6;
    els.code.style.fontSize = fs + 'px';
  }

  return {
    boot: boot,
    goto: goto,
    toggle: toggle,
    get index() { return cur; },
    get count() { return scenes.length; },
    get timeline() { return tl; }
  };
})();
