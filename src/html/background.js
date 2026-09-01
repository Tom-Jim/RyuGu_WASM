// Bevy/winit owns a DOM window and requests updates through window RAF. Keep
// native RAF in the foreground, but supply its pending callbacks from a paced
// worker while hidden. This decouples scheduling from painting, NOT WASM from
// the main thread, and cannot override browser freeze/discard or OS sleep.
export function installBackgroundExecution() {
  if (window.ryuguBackground) return window.ryuguBackground;
  const nativeRequest = window.requestAnimationFrame.bind(window);
  const nativeCancel = window.cancelAnimationFrame.bind(window);
  const callbacks = new Map();
  let nextId = 0x40000000;
  let sequence = 0;
  let pendingTick = null;
  let flushing = false;
  let worker = null;
  let enabled = true;
  let suspended = false;
  let frozen = false;
  let powerActive = false;
  let restartTimer = null;
  let lastMaintenance = 0;
  let detail = 'Connecting local sleep prevention…';
  const control = document.getElementById('background-toggle');
  const status = document.getElementById('background-status');
  const hiddenMode = () => enabled && !suspended && !frozen && worker && document.visibilityState === 'hidden';

  function maintenance() {
    if (performance.now() - lastMaintenance < 5000) return;
    lastMaintenance = performance.now();
    window.ryuguCapture?.retry?.();
  }
  function restartWorker() {
    if (!enabled || suspended || restartTimer !== null) return;
    restartTimer = setTimeout(() => {
      restartTimer = null;
      startWorker();
      rearm();
    }, 5000);
  }

  function publishStatus() {
    if (control) {
      control.checked = enabled;
      control.disabled = !('Worker' in window);
    }
    if (status) {
      const mode = !enabled ? 'Background execution off.' : suspended ? 'Page suspended.'
        : frozen ? 'Browser froze this page; computation cannot run until it resumes.'
        : !worker ? 'Background scheduler unavailable.' : 'Worker background scheduling enabled.';
      status.textContent = `${mode} ${detail}`;
      status.dataset.powerActive = String(powerActive);
    }
  }
  function requestTick() {
    if (!hiddenMode() || pendingTick !== null || flushing || callbacks.size === 0) return;
    pendingTick = ++sequence;
    worker.postMessage({ type: 'tick', id: pendingTick });
  }
  function invoke(id, timestamp) {
    const entry = callbacks.get(id);
    if (!entry) return;
    callbacks.delete(id);
    try { entry.callback(timestamp); }
    catch (error) {
      if (globalThis.reportError) globalThis.reportError(error);
      else console.error(error);
    }
  }
  function arm(id, entry) {
    if (hiddenMode()) requestTick();
    else if (!suspended && !frozen && entry.nativeId === null) {
      entry.nativeId = nativeRequest((timestamp) => invoke(id, timestamp));
    }
  }
  function rearm() {
    pendingTick = null;
    worker?.postMessage({ type: 'cancel-tick' });
    for (const [id, entry] of callbacks) {
      if (entry.nativeId !== null) nativeCancel(entry.nativeId);
      entry.nativeId = null;
      arm(id, entry);
    }
    publishStatus();
  }
  function stopWorker() {
    const current = worker;
    worker = null;
    pendingTick = null;
    current?.postMessage({ type: 'stop' });
    // Termination closes the worker-owned WebSocket even on navigation.
    current?.terminate();
    powerActive = false;
  }
  function startWorker() {
    if (!enabled || suspended || worker) return;
    try {
      worker = new Worker(new URL('./background-worker.js', import.meta.url), { type: 'module' });
    } catch (error) {
      detail = `Worker unavailable: ${error.message}`;
      publishStatus();
      restartWorker();
      return;
    }
    const current = worker;
    current.onmessage = ({ data }) => {
      if (worker !== current) return;
      if (data.type === 'power-status') {
        powerActive = data.active === true;
        detail = data.detail;
        publishStatus();
      } else if (data.type === 'maintenance') {
        maintenance();
      } else if (data.type === 'tick' && data.id === pendingTick) {
        pendingTick = null;
        if (!hiddenMode()) return;
        // Snapshot the ids, not the callbacks: cancellation by an earlier
        // callback must still suppress a later callback in this same frame.
        flushing = true;
        const timestamp = performance.now();
        try { for (const id of [...callbacks.keys()]) invoke(id, timestamp); }
        finally { flushing = false; requestTick(); maintenance(); }
      } else if (data.type === 'scheduler-error') {
        detail = `Background scheduling failed: ${data.detail}`;
        stopWorker();
        rearm();
        restartWorker();
      }
    };
    current.onerror = (event) => {
      if (worker !== current) return;
      detail = `Background worker failed: ${event.message || 'unknown error'}`;
      stopWorker();
      rearm();
      restartWorker();
    };
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
    const power = local ? new URL('/__ryugu/power', location.href) : null;
    if (power) power.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    current.postMessage({ type: 'power', url: power?.href ?? null });
  }

  // Install before importing WASM so winit uses the same request/cancel pair.
  window.requestAnimationFrame = (callback) => {
    if (typeof callback !== 'function') throw new TypeError('RAF callback must be a function');
    const id = nextId++;
    const entry = { callback, nativeId: null };
    callbacks.set(id, entry);
    arm(id, entry);
    return id;
  };
  window.cancelAnimationFrame = (id) => {
    const entry = callbacks.get(id);
    if (!entry) { nativeCancel(id); return; }
    if (entry.nativeId !== null) nativeCancel(entry.nativeId);
    callbacks.delete(id);
  };
  const api = {
    get enabled() { return enabled; },
    get powerActive() { return powerActive; },
    setEnabled(value) {
      enabled = Boolean(value);
      if (enabled) { detail = 'Connecting local sleep prevention…'; startWorker(); }
      else {
        clearTimeout(restartTimer);
        restartTimer = null;
        stopWorker(); detail = 'Local sleep prevention released.';
      }
      rearm();
    },
  };
  window.ryuguBackground = api;
  control?.addEventListener('change', () => api.setEnabled(control.checked));
  document.addEventListener('visibilitychange', () => { startWorker(); rearm(); maintenance(); });
  document.addEventListener('freeze', () => {
    frozen = true;
    rearm();
  });
  document.addEventListener('resume', () => {
    frozen = false;
    startWorker();
    worker?.postMessage({ type: 'reconnect' });
    rearm();
    maintenance();
  });
  window.addEventListener('online', () => { worker?.postMessage({ type: 'reconnect' }); maintenance(); });
  window.addEventListener('pagehide', () => {
    suspended = true;
    clearTimeout(restartTimer);
    restartTimer = null;
    stopWorker(); rearm();
  });
  window.addEventListener('pageshow', () => { suspended = false; frozen = false; startWorker(); rearm(); });
  startWorker();
  publishStatus();
  return api;
}
