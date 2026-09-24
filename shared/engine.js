/* ==========================================================================
   llama.cpp 算子之旅 · 动画引擎
   --------------------------------------------------------------------------
   硬性约束（踩过坑的结论，不是偏好）：
     - 经典 <script> 全局，不用 ES module（file:// 双击可开；module 会被 CORS 拦）
     - 零 CDN、零构建、零网络请求
     - U.hl 必须是【单遍 tokenizer】。多次 str.replace() 会把前面插入的标签
       再匹配一次，产出 "nm">@mesh 这类破碎输出
     - 各课件只引用本文件，不得内联复制引擎代码
   ========================================================================== */
'use strict';

var U = (function () {

  /* ------------------------------------------------------------ DOM 工具 */

  function el(tag, attrs) {
    var e = document.createElement(tag), k, i, ch;
    if (attrs) {
      for (k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v === null || v === undefined) continue;
        if (k === 'class')      e.className = v;
        else if (k === 'style') e.style.cssText = v;
        else if (k === 'html')  e.innerHTML = v;
        else if (k === 'text')  e.textContent = v;
        else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), v);
        else e.setAttribute(k, v);
      }
    }
    for (i = 2; i < arguments.length; i++) {
      ch = arguments[i];
      if (ch === null || ch === undefined) continue;
      e.appendChild(typeof ch === 'string' ? document.createTextNode(ch) : ch);
    }
    return e;
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  /* ------------------------------------------------------- 单遍 tokenizer */

  var KW = {
    'alignas':1,'alignof':1,'asm':1,'auto':1,'bool':1,'break':1,'case':1,'catch':1,
    'char':1,'class':1,'const':1,'constexpr':1,'continue':1,'default':1,'delete':1,
    'do':1,'double':1,'else':1,'enum':1,'explicit':1,'extern':1,'false':1,'float':1,
    'for':1,'friend':1,'goto':1,'if':1,'inline':1,'int':1,'long':1,'mutable':1,
    'namespace':1,'new':1,'noexcept':1,'nullptr':1,'operator':1,'override':1,
    'private':1,'protected':1,'public':1,'register':1,'return':1,'short':1,
    'signed':1,'sizeof':1,'static':1,'static_cast':1,'reinterpret_cast':1,
    'const_cast':1,'dynamic_cast':1,'struct':1,'switch':1,'template':1,'this':1,
    'throw':1,'true':1,'try':1,'typedef':1,'typename':1,'union':1,'unsigned':1,
    'using':1,'virtual':1,'void':1,'volatile':1,'while':1,'restrict':1,
    'NULL':1,'__global__':1,'__device__':1,'__host__':1,'__shared__':1,
    '__constant__':1,'__forceinline__':1,'__restrict__':1,'float4':1,'half':1,
    'kernel':1,'constant':1,'device':1,'threadgroup':1,'vec4':1,'layout':1,
    'shared':1,'uniform':1,'void_':1
  };

  var PUNCT = '{}()[];,.<>=+-*/%&|^!~?:#';

  function isIdStart(c) { return /[A-Za-z_$]/.test(c); }
  function isIdChar(c)  { return /[A-Za-z0-9_$]/.test(c); }
  function isDigit(c)   { return c >= '0' && c <= '9'; }

  function span(cls, text) {
    return cls ? '<span class="' + cls + '">' + esc(text) + '</span>' : esc(text);
  }

  /* 单遍扫描：从左到右，遇到 token 就吐对应标签，否则原样吐一个字符。
     绝不回头再扫已产出的 HTML。 */
  function hl(code) {
    var out = '', i = 0, n = code.length, lineStart = true;
    while (i < n) {
      var c = code[i], c2 = code[i + 1];

      /* 行注释 */
      if (c === '/' && c2 === '/') {
        var j = code.indexOf('\n', i);
        if (j < 0) j = n;
        out += span('tk-c', code.slice(i, j));
        i = j; lineStart = false; continue;
      }
      /* 块注释 */
      if (c === '/' && c2 === '*') {
        var k = code.indexOf('*/', i + 2);
        k = (k < 0) ? n : k + 2;
        out += span('tk-c', code.slice(i, k));
        lineStart = false;
        i = k; continue;
      }
      /* 预处理指令（仅在行首） */
      if (c === '#' && lineStart) {
        var e2 = code.indexOf('\n', i);
        if (e2 < 0) e2 = n;
        var line = code.slice(i, e2);
        var m = /^(#\s*\w+)([\s\S]*)$/.exec(line);
        if (m) out += span('tk-p', m[1]) + span('tk-pa', m[2]);
        else   out += span('tk-p', line);
        i = e2; lineStart = false; continue;
      }
      /* 字符串 / 字符字面量 */
      if (c === '"' || c === "'") {
        var q = c, p = i + 1;
        while (p < n) {
          if (code[p] === '\\') { p += 2; continue; }
          if (code[p] === q) { p++; break; }
          if (code[p] === '\n') break;
          p++;
        }
        out += span('tk-s', code.slice(i, p));
        i = p; lineStart = false; continue;
      }
      /* 数字 */
      if (isDigit(c) || (c === '.' && isDigit(c2))) {
        var s = i;
        while (i < n && /[0-9a-fA-FxX._']/.test(code[i])) i++;
        /* 后缀 u U l L f F */
        while (i < n && /[uUlLfF]/.test(code[i])) i++;
        out += span('tk-n', code.slice(s, i));
        lineStart = false; continue;
      }
      /* 标识符 / 关键字 */
      if (isIdStart(c)) {
        var t = i;
        while (i < n && isIdChar(code[i])) i++;
        var word = code.slice(t, i);
        var rest = code.slice(i).replace(/^\s+/, '');
        var cls;
        if (KW[word]) cls = 'tk-k';
        else if (/^[A-Z][A-Z0-9_]{2,}$/.test(word)) cls = 'tk-m';        /* 宏/枚举 */
        else if (/^(ggml|llama|gguf|GGML|LLAMA)_/.test(word)) cls = 'tk-t';
        else if (/_t$/.test(word) || /^[A-Z][a-z]/.test(word)) cls = 'tk-t';
        else if (rest.charAt(0) === '(') cls = 'tk-f';
        out += span(cls, word);
        lineStart = false; continue;
      }
      /* 换行 */
      if (c === '\n') { out += '\n'; i++; lineStart = true; continue; }
      /* 空白 */
      if (c === ' ' || c === '\t' || c === '\r') {
        out += c; i++; continue;
      }
      /* 标点 */
      if (PUNCT.indexOf(c) >= 0) {
        out += span('tk-o', c); i++; lineStart = false; continue;
      }
      out += esc(c); i++; lineStart = false;
    }
    return out;
  }

  /* 带行号的代码渲染。startNo <= 0 表示不显示行号槽（多段拼接时用）。
     ★ 注解行（//>> 开头）【不占用行号】——否则其后所有真实行号会整体错位。
     行号槽是"上游真实行号"，错位就等于误导学员。 */
  function codeBlock(code, markLines, startNo) {
    var lines = code.split('\n');
    var marks = {};
    (markLines || []).forEach(function (x) { marks[x] = 1; });
    var showNo = !(startNo === 0 || startNo === null || startNo === undefined);
    var n = showNo ? (startNo || 1) : 1;
    var html = '';
    for (var i = 0; i < lines.length; i++) {
      var isNote = /^\s*\/\/>>/.test(lines[i]);
      var body = hl(lines[i]);
      if (marks[i]) body = '<mark class="ln-mark" data-l="' + i + '">' + body + '</mark>';
      var gut = '';
      if (showNo) {
        if (isNote) {
          gut = '<span class="ln ln-note">  >>  </span>';
        } else {
          gut = '<span class="ln">' + pad(n, 4) + '</span>  ';
          n++;
        }
      }
      html += gut + body + '\n';
    }
    return html;
  }

  function pad(n, w) {
    var s = String(n);
    while (s.length < w) s = ' ' + s;
    return s;
  }

  /* ------------------------------------------------------------- 时间轴 */

  function Timeline() {
    this.events = [];
    this.duration = 0;
    this.t = 0;
    this.playing = false;
    this._fired = [];
    this._raf = null;
    this._last = 0;
    this.onEnd = null;
    this.onTick = null;
  }
  Timeline.prototype.at = function (ms, fn) {
    this.events.push({ t: ms, fn: fn, done: false });
    if (ms > this.duration) this.duration = ms;
    return this;
  };
  Timeline.prototype.reset = function () {
    this.t = 0; this._fired = [];
    for (var i = 0; i < this.events.length; i++) this.events[i].done = false;
  };
  Timeline.prototype._run = function (now) {
    if (!this.playing) return;
    var dt = now - this._last;
    this._last = now;
    this.t += dt;
    this._fire();
    if (this.onTick) this.onTick(this.t);
    if (this.t >= this.duration) {
      this.playing = false;
      if (this.onEnd) this.onEnd();
      return;
    }
    var self = this;
    this._raf = requestAnimationFrame(function (x) { self._run(x); });
  };
  Timeline.prototype._fire = function () {
    for (var i = 0; i < this.events.length; i++) {
      var e = this.events[i];
      if (!e.done && this.t >= e.t) { e.done = true; safe(e.fn); }
    }
  };
  Timeline.prototype.play = function () {
    if (this.playing) return;
    this.playing = true;
    this._last = performance.now();
    var self = this;
    this._raf = requestAnimationFrame(function (x) { self._run(x); });
  };
  Timeline.prototype.pause = function () {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  };
  /* 跳到某时刻并把该时刻之前的事件全部补跑 */
  Timeline.prototype.seek = function (ms) {
    this.pause();
    this.reset();
    this.t = ms;
    this._fire();
    if (this.onTick) this.onTick(this.t);
  };

  function safe(fn) {
    try { fn(); }
    catch (err) {
      if (window.console) console.error('[engine] scene step failed:', err);
    }
  }

  /* ------------------------------------------------------------ 卡片组件 */

  /* def = {c:'a'|'b'|..., t:'标题', b:'正文', m:'mono 行', w:'宽度'} */
  function card(def, extra) {
    var cls = 'card' + (def.c ? ' c-' + def.c : '');
    var e = el('div', { class: cls, style: (def.w ? 'width:' + def.w + ';' : '') + 'flex:0 0 auto' });
    e.style.borderLeftColor = 'var(--' + (def.c || 'border') + ')';
    if (def.t) e.appendChild(el('div', { class: 'ct', html: esc(def.t), style: 'color:var(--' + (def.c || 'text') + ')' }));
    if (def.b) e.appendChild(el('div', { class: 'cb', html: def.b }));
    if (def.m) e.appendChild(el('div', { class: 'cm', text: def.m }));
    if (extra) for (var k in extra) e.setAttribute(k, extra[k]);
    return e;
  }

  function chip(text, cls) {
    return el('span', { class: 'chip' + (cls ? ' ' + cls : ''), text: text });
  }

  function arrow(text) {
    return el('span', { class: 'arrow', text: text || '->' });
  }

  function formula(html) {
    return el('div', { class: 'formula', html: html });
  }

  function bar(label, color) {
    var f = el('div', { class: 'bf', style: 'background:var(--' + color + ')' });
    var v = el('div', { class: 'bv', text: '' });
    var e = el('div', { class: 'bar' },
      el('div', { class: 'bl', text: label }),
      el('div', { class: 'bt' }, f),
      v);
    return { el: e, fill: f, val: v };
  }

  function table(headers, rows, opts) {
    var t = el('table', { class: 'grid' });
    var thead = el('thead'), tr = el('tr');
    headers.forEach(function (h) { tr.appendChild(el('th', { text: h })); });
    thead.appendChild(tr); t.appendChild(thead);
    var tb = el('tbody');
    rows.forEach(function (r) {
      var row = el('tr');
      r.forEach(function (cell, i) {
        var td = el('td', { class: (opts && opts.monoCols && opts.monoCols.indexOf(i) >= 0) ? 'mono' : '' });
        if (cell && typeof cell === 'object' && cell.html) td.innerHTML = cell.html;
        else td.textContent = cell == null ? '' : String(cell);
        row.appendChild(td);
      });
      tb.appendChild(row);
    });
    t.appendChild(tb);
    return { el: t, body: tb };
  }

  /* 高亮代码区里指定的行（配合 codeBlock 的 mark 元素） */
  function markLines(root, lines) {
    var marks = root.querySelectorAll('mark.ln-mark');
    var set = {};
    (lines || []).forEach(function (x) { set[x] = 1; });
    for (var i = 0; i < marks.length; i++) {
      var on = !!set[marks[i].getAttribute('data-l')];
      marks[i].className = on ? 'ln-mark on' : 'ln-mark';
      marks[i].style.display = on ? 'inline-block' : 'none';
    }
  }

  return {
    el: el, esc: esc, clear: clear, hl: hl, codeBlock: codeBlock,
    Timeline: Timeline, card: card, chip: chip, arrow: arrow,
    formula: formula, bar: bar, table: table, markLines: markLines,
    FAVICON: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230d1117'/%3E%3Cpath d='M7 22 L13 10 L19 22' stroke='%2358a6ff' stroke-width='2.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='24' cy='12' r='3' fill='%233fb950'/%3E%3C/svg%3E"
  };
})();
