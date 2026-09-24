/* ==========================================================================
   llama.cpp 算子之旅 · 可复用组件
   --------------------------------------------------------------------------
   各课件通过 W.* 使用，不要内联复制。
   ========================================================================== */
'use strict';

var W = (function () {

  /* 折叠式练习：q 问题，a 答案（可含 HTML）。点击展开/收起。 */
  function exercise(q, a) {
    var box = U.el('div', { class: 'ex' });
    var qq = U.el('div', { class: 'ex-q', html: '<span class="chip c">练习</span> ' + q });
    var aa = U.el('div', { class: 'ex-a', html: a });
    qq.addEventListener('click', function () { box.classList.toggle('open'); });
    box.appendChild(qq); box.appendChild(aa);
    return box;
  }

  /* 名词解释表：pairs = [[键, 值], ...] */
  function defs(pairs, opts) {
    var rows = pairs.map(function (p) { return [p[0], p[1]]; });
    return U.table((opts && opts.head) || ['符号 / 名字', '含义'], rows,
      { monoCols: [0] }).el;
  }

  /* 键值条：把一组 chip 排成一行 */
  function kv(pairs) {
    var row = U.el('div', { class: 'flow' });
    pairs.forEach(function (p, i) {
      if (i) row.appendChild(U.arrow(':'));
      row.appendChild(U.chip(p[0], p[1]));
    });
    return row;
  }

  /* 无场景错误页（引擎/课件自身问题要优雅报告，而不是白屏） */
  function errScenes() {
    var e = U.el('div', { class: 'err' });
    e.appendChild(U.el('h3', { text: '没有加载到任何场景' }));
    e.appendChild(U.el('p', { text: 'SCENES 未定义或为空。常见原因：' }));
    var ul = U.el('ul');
    ['lesson.js 未加载或路径不对',
     'lesson.js 里有语法错误（用 tools/lint_lessons.py 秒级定位）',
     'SCENES 数组为空'].forEach(function (t) {
      ul.appendChild(U.el('li', { text: t }));
    });
    e.appendChild(ul);
    return e;
  }

  function errLayout(msg) {
    var e = U.el('div', { class: 'err' });
    e.appendChild(U.el('h3', { text: '课件外壳初始化失败' }));
    e.appendChild(U.el('p', { text: msg || '未知错误' }));
    return e;
  }

  /* 一幕没有 build()，或 build 抛错时的占位 */
  function errScene(i, err) {
    var e = U.el('div', { class: 'err' });
    e.appendChild(U.el('h3', { text: '第 ' + (i + 1) + ' 幕渲染失败' }));
    e.appendChild(U.el('p', { text: String(err && err.message || err) }));
    return e;
  }

  return {
    exercise: exercise, defs: defs, kv: kv,
    errScenes: errScenes, errLayout: errLayout, errScene: errScene
  };
})();
