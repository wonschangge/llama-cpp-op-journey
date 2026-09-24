#!/usr/bin/env node
/* ==========================================================================
   把一课 lesson.js 里的 SCENES 导出为 JSON，供保真门禁逐字校验。

   为什么用 vm 而不是正则：lesson.js 里 build() 引用的 U/W/tl 在【加载期】不会被
   调用（只有 SHELL.boot 之后才会执行 build），所以给几个桩就能安全求值。
   正则抓 code 字段会被模板字符串里的反引号/转义坑死。
   ========================================================================== */
'use strict';

const fs = require('fs');
const vm = require('vm');

const file = process.argv[2];
if (!file) { process.stderr.write('usage: dump_scenes.js <lesson.js>\n'); process.exit(2); }

let text;
try { text = fs.readFileSync(file, 'utf8'); }
catch (e) { process.stderr.write('cannot read ' + file + ': ' + e.message + '\n'); process.exit(2); }

const noop = function () {};
const stubEl = function () { return { appendChild: noop, style: {}, setAttribute: noop,
  addEventListener: noop, classList: { toggle: noop, add: noop, remove: noop }, querySelector: function () { return null; },
  querySelectorAll: function () { return []; } }; };

const sandbox = {
  console: console,
  window: { console: console, addEventListener: noop, requestAnimationFrame: noop },
  document: { createElement: stubEl, createTextNode: noop, addEventListener: noop,
              querySelector: function () { return null; } },
  requestAnimationFrame: noop,
  cancelAnimationFrame: noop,
  performance: { now: function () { return 0; } },
};
sandbox.U = { el: function () { return {}; }, esc: function (s) { return s; }, clear: noop,
              hl: function (s) { return s; }, codeBlock: function () { return ''; },
              Timeline: function () { this.at = function () { return this; }; },
              card: function () { return {}; }, chip: function () { return {}; },
              arrow: function () { return {}; }, formula: function () { return {}; },
              bar: function () { return { el: {}, fill: {}, val: {} }; },
              table: function () { return { el: {}, body: {} }; }, markLines: noop,
              FAVICON: '' };
sandbox.W = { exercise: function () { return {}; }, defs: function () { return {}; },
              kv: function () { return {}; }, errScenes: function () { return {}; },
              errLayout: function () { return {}; }, errScene: function () { return {}; } };
sandbox.SHELL = { boot: noop, goto: noop, toggle: noop };

vm.createContext(sandbox);
try { vm.runInContext(text, sandbox, { filename: file }); }
catch (e) {
  process.stdout.write(JSON.stringify({ parseError: String(e && e.message || e) }));
  process.exit(0);
}

const scenes = (function () {
  // 注意：顶层 `const SCENES` 不会成为 sandbox 全局对象的属性（只有 var 会）。
  // 所以必须在 context 内部求值，而不是读 sandbox.SCENES。
  try { return vm.runInContext('typeof SCENES !== "undefined" ? SCENES : undefined', sandbox); }
  catch (e) { return undefined; }
})();
if (!Array.isArray(scenes)) {
  process.stdout.write(JSON.stringify({ parseError: 'SCENES is not an array' }));
  process.exit(0);
}

const out = scenes.map(function (s, i) {
  return {
    i: i,
    title: s && s.title ? String(s.title) : '',
    src: (s && typeof s.src === 'string') ? s.src : null,
    code: (s && typeof s.code === 'string') ? s.code : null,
    len: (s && typeof s.code === 'string') ? s.code.split('\n').length : 0,
  };
});
process.stdout.write(JSON.stringify({ scenes: out }));
