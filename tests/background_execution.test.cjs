// Lifecycle regression coverage for the production scheduler. This fixture
// uses no browser, WASM instance, GPU, socket or native power process.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { pathToFileURL } = require('node:url');
const path = require('node:path');

function fixture(visibility = 'visible') {
  const listeners = new Map();
  const native = new Map();
  const workers = [];
  const timers = new Map();
  let timerId = 0;
  const add = (name, fn) => listeners.set(name, fn);
  let nativeId = 0;
  const window = {
    Worker: true,
    requestAnimationFrame(fn) { native.set(++nativeId, fn); return nativeId; },
    cancelAnimationFrame(id) { native.delete(id); },
    addEventListener: add,
  };
  const document = { visibilityState: visibility, getElementById: () => null, addEventListener: add };
  class Worker {
    constructor() { this.messages = []; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    tick(id = this.messages.filter((m) => m.type === 'tick').at(-1)?.id) {
      this.onmessage({ data: { type: 'tick', id } });
    }
  }
  const modulePath = path.resolve(__dirname, '../src/html/background.js');
  const source = fs.readFileSync(modulePath, 'utf8')
    .replace('export function installBackgroundExecution', 'function installBackgroundExecution')
    .replaceAll('import.meta.url', 'moduleUrl');
  vm.runInNewContext(`${source}\ninstallBackgroundExecution();`, {
    window, document, Worker, URL, console, performance: { now: () => 1234 },
    location: { hostname: 'localhost', href: 'http://localhost:3000/', protocol: 'http:' },
    moduleUrl: pathToFileURL(modulePath).href,
    setTimeout(fn) { timers.set(++timerId, fn); return timerId; },
    clearTimeout(id) { timers.delete(id); },
  });
  return { window, document, native, workers, event: (name) => listeners.get(name)?.() };
}

test('visible frames use native RAF; hiding moves the pending frame without losing it', () => {
  const f = fixture();
  let calls = 0;
  f.window.requestAnimationFrame(() => calls++);
  assert.equal(f.native.size, 1);
  f.document.visibilityState = 'hidden';
  f.event('visibilitychange');
  assert.equal(f.native.size, 0);
  f.workers[0].tick();
  assert.equal(calls, 1);
});

test('background cancellation and nested callbacks preserve one outstanding tick', () => {
  const f = fixture('hidden');
  const calls = [];
  let second;
  f.window.requestAnimationFrame(() => {
    calls.push('first');
    f.window.cancelAnimationFrame(second);
    f.window.requestAnimationFrame(() => calls.push('next'));
  });
  second = f.window.requestAnimationFrame(() => calls.push('cancelled'));
  const worker = f.workers[0];
  const tickMessages = () => worker.messages.filter((m) => m.type === 'tick');
  assert.equal(tickMessages().length, 1);
  const previous = tickMessages()[0].id;
  worker.tick(previous);
  assert.deepEqual(calls, ['first']);
  assert.equal(tickMessages().length, 2);
  worker.tick(previous); // a late/duplicate packet must not run the next frame
  assert.deepEqual(calls, ['first']);
  worker.tick();
  assert.deepEqual(calls, ['first', 'next']);
});

test('returning visible ignores stale worker ticks; disabling closes the worker', () => {
  const f = fixture('hidden');
  let calls = 0;
  f.window.requestAnimationFrame(() => calls++);
  const worker = f.workers[0];
  f.document.visibilityState = 'visible';
  f.event('visibilitychange');
  worker.tick();
  assert.equal(calls, 0);
  assert.equal(f.native.size, 1);
  [...f.native.values()][0](1234);
  assert.equal(calls, 1);
  f.window.ryuguBackground.setEnabled(false);
  assert.equal(worker.terminated, true);
  assert.equal(f.window.ryuguBackground.powerActive, false);
});

test('pagehide stops power/scheduling; BFCache pageshow creates one fresh worker', () => {
  const f = fixture('hidden');
  let calls = 0;
  f.window.requestAnimationFrame(() => calls++);
  f.event('pagehide');
  assert.equal(f.workers[0].terminated, true);
  f.workers[0].tick();
  assert.equal(calls, 0);
  f.event('pageshow');
  assert.equal(f.workers.length, 2);
  f.workers[0].onerror({ message: 'delayed error from the terminated worker' });
  assert.notEqual(f.workers[1].terminated, true);
  f.workers[1].tick();
  assert.equal(calls, 1);
});
