// Durable, permission-free exports. PNGs are rendered from the same SVG/data
// as the chart, not screen pixels: hidden tabs do not need a paint or RAF.
// This archive preserves completed results, NOT live WASM/GPU state. A reload
// retries saved exports but cannot resume an interrupted numerical repetition.
(() => {
  const sources = [32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000];
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
  const pending = new Map();
  const currentFiles = new Map();
  const links = new Map();
  let run = null;
  let nextSelection = null;
  let busy = false;
  let storageError = '';
  let uploadError = '';
  let storageChain = Promise.resolve();
  let lastRetry = 0;
  const db = new Promise((resolve, reject) => {
    let abandoned = false;
    const timeout = setTimeout(() => {
      abandoned = true;
      reject(new Error('Browser archive did not open within 10 seconds'));
    }, 10_000);
    const request = indexedDB.open('ryugu-benchmark-exports', 1);
    request.onupgradeneeded = () => {
      const store = request.result.createObjectStore('files', { keyPath: 'id' });
      store.createIndex('pending', 'pending');
    };
    request.onsuccess = () => {
      clearTimeout(timeout);
      if (abandoned) { request.result.close(); return; }
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
    request.onerror = () => { clearTimeout(timeout); reject(request.error); };
    request.onblocked = () => {
      abandoned = true;
      clearTimeout(timeout);
      reject(new Error('Close older Ryugu tabs to unlock local export storage'));
    };
  }).catch((error) => {
    storageError = `Browser archive unavailable: ${error.message}. Keep this tab open until disk saves finish.`;
    publish();
    return null;
  });

  function persist(record) {
    // Order replacements of results.json and PNG conversion/upload receipts.
    storageChain = storageChain.then(async () => {
      const database = await db;
      if (!database) return;
      await new Promise((resolve, reject) => {
        const transaction = database.transaction('files', 'readwrite');
        const timeout = setTimeout(() => {
          try { transaction.abort(); } catch { /* already completed */ }
          reject(new Error('Archive write timed out'));
        }, 10_000);
        transaction.objectStore('files').put(record);
        transaction.oncomplete = () => { clearTimeout(timeout); resolve(); };
        transaction.onerror = () => { clearTimeout(timeout); reject(transaction.error); };
        transaction.onabort = () => { clearTimeout(timeout); reject(transaction.error ?? new Error('Archive transaction aborted')); };
      });
    }).catch((error) => {
      storageError = `Browser archive failed: ${error.message}. Keep this tab open until disk saves finish.`;
      publish();
    });
    return storageChain;
  }

  function publish() {
    const status = document.getElementById('quadrature-capture-status');
    if (!status) return;
    const pngs = [...currentFiles.values()].filter((file) => file.name.endsWith('.png'));
    const saved = pngs.filter((file) => file.saved).length;
    status.textContent = run
      ? `Auto screenshots: ${pngs.length}/9 captured · ${saved}/9 saved to disk · Kρ=${run.selection.densityModels}, Nt=${run.selection.targets}, ${run.selection.accuracyProfile}, ${run.selection.timingKey}. `
        + (local ? `Folder: benchmark-captures/${run.id}/. ` : 'Hosted page: use the download links below; local disk service unavailable. ')
      : 'Auto screenshots: 32K–8192K, 9 PNGs, one after each source finishes 7 repetitions. Parameters are fixed when Run is pressed. ';
    status.textContent += [storageError, uploadError].filter(Boolean).join(' ');
    const host = document.getElementById('quadrature-capture-files');
    if (!host) return;
    for (const file of currentFiles.values()) {
      if (file.svg) continue; // PNG has not been encoded yet.
      const old = links.get(file.id);
      if (old?.body === file.body) continue;
      if (old) URL.revokeObjectURL(old.url);
      const url = URL.createObjectURL(file.body);
      const anchor = old?.anchor ?? document.createElement('a');
      anchor.href = url;
      anchor.download = file.name;
      anchor.textContent = file.name;
      if (!old) host.append(anchor, document.createTextNode(' · '));
      links.set(file.id, { body: file.body, url, anchor });
    }
  }

  function enqueue(name, body, svg = false) {
    const record = { id: `${run.id}/${name}`, run: run.id, name, body, svg, pending: 1, saved: false };
    pending.set(record.id, record);
    currentFiles.set(record.id, record);
    void persist(record).then(() => flush());
  }

  async function pngFromSvg(blob) {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    let timeout;
    try {
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error('SVG screenshot could not be decoded'));
        timeout = setTimeout(() => reject(new Error('Screenshot decode timed out; will retry')), 30_000);
        image.src = url;
      });
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('PNG export requires a 2D canvas');
      context.drawImage(image, 0, 0);
      clearTimeout(timeout);
      return await new Promise((resolve, reject) => {
        timeout = setTimeout(() => reject(new Error('PNG encoding timed out; will retry')), 30_000);
        canvas.toBlob((png) => png ? resolve(png) : reject(new Error('PNG encoding failed')), 'image/png');
      });
    } finally {
      clearTimeout(timeout);
      image.onload = null;
      image.onerror = null;
      URL.revokeObjectURL(url);
    }
  }

  async function flush() {
    if (busy) return;
    busy = true;
    try {
      // Snapshot the queue; newly completed repetitions can replace results
      // while this upload is in flight. Never acknowledge a newer replacement.
      for (const original of [...pending.values()]) {
        if (pending.get(original.id) !== original) continue;
        let record = original;
        try {
          if (record.svg) {
            const png = await pngFromSvg(record.body);
            record = { ...record, body: png, svg: false };
            pending.set(record.id, record);
            if (currentFiles.has(record.id)) currentFiles.set(record.id, record);
            await persist(record);
            publish();
          }
          if (!local) {
            // Keep downloadable files archived without retrying an absent
            // local endpoint on a static hosted deployment.
            pending.delete(record.id);
            await persist({ ...record, pending: 0 });
            continue;
          }
          const response = await fetch(`/__ryugu/exports/${record.run}/${record.name}`, {
            method: 'PUT', body: record.body,
            headers: { 'Content-Type': record.body.type },
            signal: AbortSignal.timeout(30_000),
          });
          if (!response.ok) throw new Error(`Local save returned HTTP ${response.status}`);
          const receipt = await response.json();
          if (typeof receipt.saved !== 'string') throw new Error('Missing disk-save receipt');
          if (pending.get(record.id) === record) {
            pending.delete(record.id);
            const saved = { ...record, pending: 0, saved: true };
            if (currentFiles.has(record.id)) currentFiles.set(record.id, saved);
            await persist(saved);
          }
          uploadError = '';
        } catch (error) {
          uploadError = `${error.message}. Queued exports will retry automatically; numerical work continues.`;
          // Disk/service failures should not spawn nine identical retries.
          break;
        }
      }
    } finally { busy = false; publish(); }
  }

  function selection() {
    return {
      ...window.ryuguUi.curveSelection(),
      timingKey: document.getElementById('quadrature-timing').value,
      accuracyProfile: document.querySelector('#quadrature-modal [data-planning-accuracy]').value,
    };
  }

  function observe(planning) {
    if (planning.workload !== 'quadrature') return;
    if (!run || run.backendRunId !== planning.runId) {
      if (!planning.running && !planning.curve?.length) return;
      for (const item of links.values()) URL.revokeObjectURL(item.url);
      links.clear();
      currentFiles.clear();
      document.getElementById('quadrature-capture-files')?.replaceChildren();
      run = {
        id: `${new Date().toISOString().replace(/[:.]/g, '-')}-${crypto.randomUUID()}`,
        backendRunId: planning.runId,
        selection: nextSelection ?? selection(),
        captured: new Set(),
        resultKey: null,
      };
      nextSelection = null;
      // Persistent storage is best effort and does not open a file picker.
      try { navigator.storage?.persist?.().catch(() => {}); } catch { /* optional */ }
    }
    const resultKey = `${planning.curve?.length ?? 0}/${planning.running}/${planning.completed}/${planning.accuracyProfile}`;
    if (resultKey !== run.resultKey) {
      enqueue('results.json', new Blob([JSON.stringify({
        version: 1, savedAt: new Date().toISOString(), run: run.id,
        screenshotSelection: run.selection, sourceCounts: sources, planning,
        recovery: 'Completed results only. Live WASM/GPU state is not a restart checkpoint.',
      })], { type: 'application/json' }));
      run.resultKey = resultKey;
    }
    const required = Math.max(7, planning.requiredRepeats ?? 7);
    const rows = (planning.curve ?? []).filter((row) => row.densityModels === run.selection.densityModels
      && row.targets === run.selection.targets);
    sources.forEach((source, index) => {
      if (run.captured.has(source)) return;
      const repeats = new Set(rows.filter((row) => row.sources === source).map((row) => row.repeat));
      if (!Array.from({ length: required }, (_, i) => i + 1).every((repeat) => repeats.has(repeat))) return;
      // A failed numerical gate is still a finished source, and is captured
      // honestly as a gap/FAIL rather than waiting forever for a passing point.
      const svg = window.ryuguUi.exportQuadrature(planning, run.selection, source);
      enqueue(`${String(index + 1).padStart(2, '0')}-${source / 1000}K.png`,
        new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), true);
      run.captured.add(source);
    });
    publish();
    retry();
  }

  function retry() {
    if (performance.now() - lastRetry < 5000) return;
    lastRetry = performance.now();
    void flush();
  }

  window.ryuguCapture = {
    begin() { nextSelection = selection(); },
    observe,
    retry,
    get pendingCount() { return pending.size; },
  };
  // Drain interrupted exports after reload; never start an unrequested new
  // calculation or silently combine measurements from different captures.
  void db.then(async (database) => {
    if (!database) return;
    const records = await new Promise((resolve, reject) => {
      const request = database.transaction('files').objectStore('files').index('pending').getAll(1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    for (const record of records) if (!pending.has(record.id)) pending.set(record.id, record);
    await flush();
  }).catch((error) => { storageError = `Archive recovery failed: ${error.message}`; publish(); });
  window.addEventListener('online', retry);
  window.addEventListener('pageshow', retry);
  document.addEventListener('visibilitychange', retry);
  setInterval(retry, 5000);
})();
