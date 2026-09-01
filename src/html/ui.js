// Shared by both chart renderers. A timing cell is published only after all
// seven distinct repetitions pass; never take a median of just the survivors.
window.ryuguCurveStatistics = (samples, densityModels, targets, requiredRepeats = 7, timingKey = 'times') => {
  const groups = new Map();
  for (const sample of samples ?? []) {
    if (sample.densityModels !== densityModels || sample.targets !== targets
      || !Number.isInteger(sample.repeat) || sample.repeat < 1) continue;
    if (!groups.has(sample.sources)) groups.set(sample.sources, new Map());
    groups.get(sample.sources).set(sample.repeat, sample);
  }
  return [...groups].sort((a, b) => a[0] - b[0]).map(([sources, repeats]) => {
    const rows = [...repeats.values()];
    const complete = Array.from({ length: Math.max(7, requiredRepeats) }, (_, i) => i + 1)
      .every((repeat) => repeats.has(repeat));
    const methods = Array.from({ length: 6 }, (_, index) => {
      const valid = rows.filter((sample) => sample.eligible?.[index] === true
        && Number.isFinite(sample.times?.[index]) && sample.times[index] > 0);
      const rejected = rows.length - valid.length;
      const qualified = complete && rejected === 0;
      const measured = valid.map((sample) => sample[timingKey]?.[index]);
      const timingAvailable = qualified && measured.every((time) => Number.isFinite(time) && time >= 0);
      const sorted = timingAvailable ? measured.sort((a, b) => a - b) : [];
      const median = sorted.length ? (sorted[Math.floor(sorted.length / 2)] + sorted[Math.ceil(sorted.length / 2) - 1]) / 2 : null;
      const maximumError = (key) => rows.every((sample) => Number.isFinite(sample[key]?.[index]))
        ? Math.max(...rows.map((sample) => sample[key][index])) : null;
      return {
        value: timingAvailable && median > 0 ? median : null,
        low: timingAvailable ? sorted[0] : null,
        high: timingAvailable ? sorted[sorted.length - 1] : null,
        timingAvailable,
        belowResolution: timingAvailable && median === 0,
        count: rows.length,
        rejected,
        gravityError: maximumError('gravityErrors'),
        gradientError: maximumError('gradientErrors'),
        reasons: [...new Set(rows.flatMap((sample) => sample.failureReasons?.[index] ?? []))],
        strictPassed: rows.filter((sample) => sample.strictEligible?.[index] === true).length,
        status: rejected ? 'FAIL' : qualified ? 'PASS' : 'PENDING',
      };
    });
    return { sources, methods };
  });
};

// A full source axis is shared by every series. Missing/failed cells remain
// explicit gaps; completed cells do not wait for the rest of the sweep.
window.ryuguCurvePlotData = (groups, sourceCounts) => {
  const bySource = new Map(groups.map((group) => [group.sources, group]));
  return Array.from({ length: 6 }, (_, methodIndex) => ({
    points: sourceCounts.map((source) => {
      const method = bySource.get(source)?.methods[methodIndex];
      return [source, method?.status === 'PASS' ? method.value : null, method];
    }),
    ranges: sourceCounts.flatMap((source) => {
      const method = bySource.get(source)?.methods[methodIndex];
      return method?.status === 'PASS' && method.low > 0 && Number.isFinite(method.high)
        ? [[source, method.low, method.high]] : [];
    }),
  }));
};

// Backend completion is authoritative. Rounding 99.5% to 100% or keeping a
// previous run's high-water mark can falsely announce completion.
window.ryuguPlanningProgress = (planning) => ({
  runId: planning.runId,
  progress: planning.completed === true ? 100
    : Math.floor(Math.min(99.9, Math.max(0, Number(planning.progress) || 0)) * 10) / 10,
  accuracy: Math.min(100, Math.max(0, Number(planning.accuracy) || 0)),
  running: Boolean(planning.running),
  completed: planning.completed === true,
  workCompleted: Number(planning.workCompleted) || 0,
  workTotal: Number(planning.workTotal) || 0,
});

(() => {
  const queue = [];
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const push = (type, value, extra = {}) => queue.push(JSON.stringify({ type, value, ...extra }));
  const methodKeys = ['radial', 'werner', 'eq106', 'fft', 'fmm'];
  const methodLabels = ['Radial', 'Werner', 'Eq.106', 'Packed FFT', 'FMM'];
  const methodColors = ['#58c8ff', '#ff7d89', '#36e7f2', '#ffb23d', '#42dc77'];
  const curveColors = ['#36e7f2', '#9af8ff', '#ffb23d', '#ffe071', '#42dc77', '#a8f7bd'];
  const curveLabels = ['Eq.106 raw total', 'Eq.106 checked total', 'FFT raw total', 'FFT checked total', 'FMM raw total', 'FMM checked total'];
  const quadratureSourceCounts = [32_000, 64_000, 128_000, 256_000, 512_000, 1_024_000, 2_048_000, 4_096_000, 8_192_000];
  const metricFields = {
    density: ['density', ''],
    'inversion-time': ['timeMs', 'ms'],
    'gravity-error': ['gravityError', 'relative error'],
    'gradient-error': ['gradientError', 'relative error'],
    pericenter: ['pericenterError', 'm'],
    altitude: ['minimumAltitude', 'm'],
    separation: ['separation', 'score'],
    objective: ['objective', 'score'],
    segments: ['segments', 'segments'],
    speedup: ['totalMs', 'ms'],
    cold: ['coldCandidates', 'candidates'],
  };
  let editingProbe = false;
  let editingTrajectory = false;
  let lastSnapshot = null;
  let lastCurveRenderKey = null;
  let lastCurveDataKey = null;
  let lastCurveGroups = [];
  let openDialog = null;
  let returnFocus = null;

  // Panels start in a non-overlapping grid, then can be repositioned by their
  // heading without changing the DOM order or blocking their internal scroll.
  document.addEventListener('pointerdown', (event) => {
    const handle = event.target.closest?.('.drag-handle');
    const panel = handle?.closest?.('[data-float-panel]');
    if (!panel || event.button !== 0) return;
    const start = { x: event.clientX, y: event.clientY };
    const origin = panel._ryuguOffset ?? { x: 0, y: 0 };
    panel.classList.add('is-dragging');
    handle.setPointerCapture?.(event.pointerId);
    const move = (moveEvent) => {
      const x = origin.x + moveEvent.clientX - start.x;
      const y = origin.y + moveEvent.clientY - start.y;
      panel._ryuguOffset = { x, y };
      panel.style.transform = `translate(${x}px, ${y}px)`;
    };
    const end = () => {
      panel.classList.remove('is-dragging');
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', end);
      handle.removeEventListener('pointercancel', end);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', end);
    handle.addEventListener('pointercancel', end);
    event.stopPropagation();
    event.preventDefault();
  });

  // Closing the quadrature page is a cancellation, not merely a visual hide.
  // The Rust action drops its job and prevents any later GPU dispatch.
  document.addEventListener('click', (event) => {
    const dialog = event.target.closest?.('.modal');
    if (dialog && event.target === dialog && dialog.id === 'quadrature-modal') {
      push('quadrature-cancel', null);
    }
  });

  document.addEventListener('input', (event) => {
    if (event.target.id === 'acceleration') push('acceleration', Number(event.target.value));
    if (event.target.dataset.probe) {
      editingProbe = true;
      const parameter = event.target.dataset.probe;
      const value = Number(event.target.value);
      $('#probe-' + parameter + '-out').textContent = parameter === 'speed' ? value.toFixed(3) : value.toFixed(0);
    }
  });
  document.addEventListener('change', (event) => {
    if (event.target.dataset.probe) {
      push('probe', Number(event.target.value), { parameter: event.target.dataset.probe });
      editingProbe = false;
    }
  });
  document.addEventListener('focusin', (event) => {
    if (event.target.matches?.('[data-trajectory-field]')) {
      editingTrajectory = true;
      event.target.dataset.initialValue = event.target.value;
    }
  });
  function submitTrajectoryField(input) {
    const values = input.value.split(/[\s,]+/).filter(Boolean).map(Number);
    if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) {
      input.setCustomValidity('Enter exactly three finite numbers.');
      input.reportValidity();
      return false;
    }
    input.setCustomValidity('');
    push('trajectory-knot', values, {
      index: Number(input.dataset.trajectoryIndex),
      field: input.dataset.trajectoryField,
    });
    input.dataset.initialValue = input.value;
    return true;
  }
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.target.matches?.('[data-trajectory-field]')) {
      event.preventDefault();
      if (submitTrajectoryField(event.target)) event.target.blur();
    }
  });
  document.addEventListener('focusout', (event) => {
    if (!event.target.matches?.('[data-trajectory-field]')) return;
    if (event.target.value !== event.target.dataset.initialValue) submitTrajectoryField(event.target);
    editingTrajectory = false;
  });
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !openDialog) return;
    push(openDialog === $('#performance-page') ? 'performance-close' : 'quadrature-cancel', null);
  });

  const svgNode = (name, attrs = {}, text = '') => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text) element.textContent = text;
    return element;
  };
  const formatAxis = (value) => {
    if (!Number.isFinite(value)) return '--';
    const normalized = Object.is(value, -0) ? 0 : value;
    return normalized
      .toExponential(3)
      .replace(/\.?(?:0+)e/, 'e')
      .replace('e+', 'e');
  };
  function drawChart(svg, series, { fitViewport = false, yLog = false, xLabel = '', yLabel = '', xDomain = null, xCategories = null, minimumYDomain = null, empty = 'Waiting for samples…' } = {}) {
    // Fallback callers may supply a host div; SVG shapes require an actual
    // SVG viewport, not a div carrying a meaningless viewBox attribute.
    if (svg.namespaceURI !== 'http://www.w3.org/2000/svg') {
      const viewport = svgNode('svg', { role: 'img' });
      svg.replaceChildren(viewport);
      svg = viewport;
    }
    svg.replaceChildren();
    const width = fitViewport ? Math.max(svg.clientWidth, 640) : 900;
    const height = fitViewport ? Math.max(svg.clientHeight, 240) : 430;
    const margin = { l: 104, r: 28, t: 22, b: 70 };
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const points = series
      .flatMap((item) => item.points)
      .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]) && (!yLog || point[1] > 0));
    const categoryIndex = xCategories ? new Map(xCategories.map((value, index) => [value, index])) : null;
    const transformX = (value) => categoryIndex ? categoryIndex.get(value) : value;
    const transformY = (value) => yLog ? Math.log10(value) : value;
    const xs = points.map((point) => transformX(point[0]));
    const ys = points.map((point) => transformY(point[1])).concat(series
      .flatMap((item) => item.ranges ?? []).flatMap((range) => range.slice(1))
      .filter((value) => Number.isFinite(value) && (!yLog || value > 0)).map(transformY));
    let xMin = xs.length ? Math.min(...xs) : 0;
    let xMax = xs.length ? Math.max(...xs) : 1;
    let yMin = ys.length ? Math.min(...ys) : 0;
    let yMax = ys.length ? Math.max(...ys) : 1;
    if (categoryIndex) {
      xMin = 0;
      xMax = Math.max(xCategories.length - 1, 1);
    } else if (xDomain?.length === 2 && xDomain.every(Number.isFinite)) {
      xMin = transformX(xDomain[0]);
      xMax = transformX(xDomain[1]);
    }
    if (minimumYDomain?.length === 2 && minimumYDomain.every((value) => Number.isFinite(value) && (!yLog || value > 0))) {
      const domainMin = transformY(minimumYDomain[0]);
      const domainMax = transformY(minimumYDomain[1]);
      yMin = points.length ? Math.min(yMin, domainMin) : domainMin;
      yMax = points.length ? Math.max(yMax, domainMax) : domainMax;
    }
    if (xMin === xMax) {
      const pad = Math.max(Math.abs(xMin) * 0.05, 1);
      xMin -= pad;
      xMax += pad;
    }
    if (yMin === yMax) {
      const pad = Math.max(Math.abs(yMin) * 0.05, yLog ? 0.5 : 1);
      yMin -= pad;
      yMax += pad;
    } else if (points.length) {
      const pad = (yMax - yMin) * 0.06;
      yMin -= pad;
      yMax += pad;
    }
    const pixelX = (value) => margin.l + (transformX(value) - xMin) / (xMax - xMin) * (width - margin.l - margin.r);
    const pixelY = (value) => height - margin.b - (transformY(value) - yMin) / (yMax - yMin) * (height - margin.t - margin.b);
    for (let index = 0; index <= 5; index += 1) {
      const y = margin.t + index / 5 * (height - margin.t - margin.b);
      const raw = yMax - index / 5 * (yMax - yMin);
      const actual = yLog ? 10 ** raw : raw;
      svg.append(
        svgNode('line', { x1: margin.l, x2: width - margin.r, y1: y, y2: y, class: 'grid-line' }),
        svgNode('text', { x: margin.l - 9, y: y + 3, 'text-anchor': 'end', class: 'axis-label' }, formatAxis(actual)),
      );
    }
    const xTicks = xCategories ?? Array.from({ length: 5 }, (_, index) => {
      return xMin + index / 4 * (xMax - xMin);
    });
    xTicks.forEach((value) => {
      const raw = transformX(value);
      const x = margin.l + (raw - xMin) / (xMax - xMin) * (width - margin.l - margin.r);
      const label = categoryIndex ? `${value / 1000}K` : formatAxis(value);
      svg.append(
        svgNode('line', { x1: x, x2: x, y1: margin.t, y2: height - margin.b, class: 'grid-line' }),
        svgNode('text', { x, y: height - margin.b + 18, 'text-anchor': 'middle', class: 'axis-label' }, label),
      );
    });
    svg.append(
      svgNode('line', { x1: margin.l, x2: width - margin.r, y1: height - margin.b, y2: height - margin.b, class: 'axis-line' }),
      svgNode('line', { x1: margin.l, x2: margin.l, y1: margin.t, y2: height - margin.b, class: 'axis-line' }),
      svgNode('text', { x: (margin.l + width - margin.r) / 2, y: height - 8, 'text-anchor': 'middle', class: 'axis-label' }, xLabel),
      svgNode('text', { x: 13, y: height / 2, transform: `rotate(-90 13 ${height / 2})`, 'text-anchor': 'middle', class: 'axis-label' }, yLabel),
    );
    if (!points.length) {
      svg.append(svgNode('text', { x: (margin.l + width - margin.r) / 2, y: height / 2, 'text-anchor': 'middle', class: 'empty-label' }, empty));
      return;
    }
    series.forEach((item) => {
      const isValid = (point) => Number.isFinite(point[0]) && Number.isFinite(point[1]) && (!yLog || point[1] > 0);
      const valid = item.points.filter(isValid);
      if (valid.length) {
        let connected = false;
        const path = item.points.map((point) => {
          if (!isValid(point)) { connected = false; return ''; }
          const command = connected ? 'L' : 'M';
          connected = true;
          return `${command}${pixelX(point[0]).toFixed(2)},${pixelY(point[1]).toFixed(2)}`;
        }).join(' ');
        const pathAttributes = { d: path, stroke: item.color, class: 'chart-line' };
        if (item.dashed) {
          pathAttributes['stroke-dasharray'] = '8 5';
        }
        svg.append(svgNode('path', pathAttributes));
        for (const [source, low, high] of item.ranges ?? []) {
          const x = pixelX(source);
          svg.append(svgNode('path', {
            d: `M${x},${pixelY(low)}V${pixelY(high)} M${x - 3},${pixelY(low)}H${x + 3} M${x - 3},${pixelY(high)}H${x + 3}`,
            stroke: item.color, fill: 'none',
          }));
        }
        valid.forEach((point) => {
          const marker = svgNode('circle', {
            cx: pixelX(point[0]).toFixed(2),
            cy: pixelY(point[1]).toFixed(2),
            r: 4,
            fill: item.color,
            class: 'chart-point',
          });
          const sourceLabel = categoryIndex ? `${point[0] / 1000}K` : formatAxis(point[0]);
          const stats = point[2];
          const detail = stats
            ? `; ${stats.count} repetitions; min–max ${formatAxis(stats.low)}–${formatAxis(stats.high)} ms; εg=${formatAxis(stats.gravityError)}, ε∇g=${formatAxis(stats.gradientError)}; strict ${stats.strictPassed}/${stats.count}` : '';
          marker.append(svgNode('title', {}, `${item.label} at ${sourceLabel}: ${formatAxis(point[1])} ms${detail}`));
          svg.append(marker);
        });
      }
    });
  }

  function curveSeries(groups, timingKey) {
    const plot = window.ryuguCurvePlotData(groups, quadratureSourceCounts);
    return curveLabels.map((label, index) => ({
      label: timingKey === 'times' ? label : label.replace(' total', ' kernels'),
      color: curveColors[index],
      dashed: index % 2 === 1,
      ...plot[index],
    }));
  }

  function exportQuadrature(planning, selection, source) {
    // Build a detached, self-contained SVG at fixed resolution. Do not depend
    // on layout, the visible dropdowns, screenshots of the desktop, or RAF.
    const timingKey = selection.timingKey;
    const strict = selection.accuracyProfile === 'strict';
    const rows = (planning.curve ?? []).filter((row) => row.sources <= source).map((row) => {
      const masks = strict ? row.strictFailures : row.screeningFailures;
      return {
        ...row,
        times: row.rawTimes ?? row.times,
        kernelTimes: row.rawKernelTimes ?? row.kernelTimes,
        evaluationKernelTimes: row.rawEvaluationKernelTimes ?? row.evaluationKernelTimes,
        eligible: masks ? masks.map((mask) => mask === 0) : row.eligible,
        failureReasons: strict ? row.strictFailureReasons ?? row.failureReasons
          : row.screeningFailureReasons ?? row.failureReasons,
      };
    });
    const groups = window.ryuguCurveStatistics(rows, selection.densityModels,
      selection.targets, planning.requiredRepeats, timingKey);
    const series = curveSeries(groups, timingKey);
    // XMLSerializer supplies the namespace for this createElementNS root.
    const root = svgNode('svg', { width: 1800, height: 1500, viewBox: '0 0 1800 1500' });
    root.append(svgNode('style', {}, `
      text { font-family: monospace; fill: #d9edf0; font-size: 20px; }
      .axis-label { fill: #a3bcc1; font-size: 11px; }
      .grid-line { stroke: #193439; stroke-width: 0.7; }
      .axis-line { stroke: #50737b; }
      .chart-line { fill: none; stroke-width: 2; }
      .chart-point { stroke: #071215; stroke-width: 1; }
      .empty-label { fill: #98b4bc; font-size: 12px; }
    `));
    root.append(svgNode('rect', { width: 1800, height: 1500, fill: '#061013' }));
    const text = (x, y, value, attrs = {}) => root.append(svgNode('text', { x, y, ...attrs }, value));
    text(32, 48, 'Quadrature-source crossover', { style: 'font-size:32px;font-weight:bold' });
    text(32, 88, `Kρ=${selection.densityModels} · Nt=${selection.targets} · ${selection.accuracyProfile} · ${timingKey} · 7 repetitions / median / min–max`);
    const completed = groups.filter((group) => group.methods.every((method) => method.count >= (planning.requiredRepeats ?? 7))).length;
    text(32, 122, `Milestone ${source / 1000}K · ${completed}/9 source sizes complete · ${new Date().toISOString()}`);
    text(32, 155, 'Finished FAIL cells remain gaps. Screenshots keep the parameters selected at task launch.', { fill: '#98b4bc' });
    const chart = svgNode('svg', { x: 20, y: 174, width: 1760, height: 840 });
    drawChart(chart, series, {
      yLog: true, xCategories: quadratureSourceCounts, xLabel: 'source points',
      yLabel: `${timingKey === 'times' ? 'pipeline total' : 'GPU kernels'} median / min–max (ms)`,
      empty: 'Completed cells have no qualified positive timings; see the accuracy results below.',
    });
    root.append(chart);
    series.forEach((item, index) => {
      const x = 32 + (index % 3) * 580;
      const y = 1040 + Math.floor(index / 3) * 38;
      root.append(svgNode('line', { x1: x, x2: x + 42, y1: y - 7, y2: y - 7,
        stroke: item.color, 'stroke-width': 3, ...(item.dashed ? { 'stroke-dasharray': '8 5' } : {}) }));
      text(x + 52, y, item.label);
    });
    text(32, 1130, `${source / 1000}K accuracy / timings (ms)`, { style: 'font-weight:bold' });
    text(680, 1130, 'Status / median');
    text(1030, 1130, 'min–max');
    text(1370, 1130, 'RMS εg / ε∇g');
    const cell = groups.find((group) => group.sources === source);
    cell?.methods.forEach((method, index) => {
      const y = 1172 + index * 42;
      text(32, y, series[index].label);
      text(680, y, `${method.status} ${method.count}/7 · ${formatAxis(method.value)}`);
      text(1030, y, `${formatAxis(method.low)}–${formatAxis(method.high)}`);
      text(1370, y, `${formatAxis(method.gravityError)} / ${formatAxis(method.gradientError)}`);
    });
    text(32, 1440, 'Raw + checked GPU methods · independent f64 validation · full result/gate details in results.json');
    text(32, 1476, 'Capture continues while hidden; OS sleep/browser discard can stop the engine.', { fill: '#98b4bc' });
    return new XMLSerializer().serializeToString(root);
  }

  function renderQuadrature(planning, selection) {
    const timingKey = $('#quadrature-timing').value;
    const timingTitle = timingKey === 'times' ? 'pipeline total' : timingKey === 'kernelTimes'
      ? 'GPU method kernels' : 'GPU target kernels';
    // Numerical rows are append-only within a run. Do not reaggregate the
    // whole sweep or rewrite accuracy text on every status/progress snapshot.
    const dataKey = JSON.stringify([planning.runId, planning.accuracyProfile,
      selection.densityModels, selection.targets, planning.requiredRepeats, timingKey, planning.curve?.length ?? 0]);
    if (dataKey !== lastCurveDataKey) {
      lastCurveGroups = window.ryuguCurveStatistics(planning.curve,
        selection.densityModels, selection.targets, planning.requiredRepeats, timingKey);
      lastCurveDataKey = dataKey;
      lastCurveRenderKey = null;
    }
    if (lastCurveRenderKey !== null) return;
    const groups = lastCurveGroups;
    $('#quadrature-timing-policy').textContent = timingKey === 'times' ? planning.timingDefinition ?? ''
      : `${timingTitle}: wgpu pass-boundary timestamps only; checked is raw + the additional checked pass. Excludes CPU preparation, copies, queue waits and metrics reduction. FFT source deposition, 56 basis convolutions and density combinations run on GPU. FMM moment construction, tree traversal, near field and 56-basis density mixing also run on GPU; target-only FMM mixes cached responses. These kernels do different work, so this is not hardware FLOP throughput or an end-to-end algorithm speedup. Missing/zero-resolution timestamps are not replaced by wall time.`;
    const required = Math.max(7, planning.requiredRepeats ?? 7);
    const complete = groups.filter((group) => group.methods.every((method) => method.count >= required));
    const plotted = groups.reduce((total, group) => total
      + group.methods.filter((method) => method.status === 'PASS' && Number.isFinite(method.value)).length, 0);
    const failed = groups.reduce((total, group) => total
      + group.methods.filter((method) => method.status === 'FAIL').length, 0);
    const unavailable = groups.reduce((total, group) => total
      + group.methods.filter((method) => method.status === 'PASS' && !Number.isFinite(method.value)).length, 0);
    const pending = groups.find((group) => group.methods.some((method) => method.count < required));
    const status = $('#quadrature-plot-status');
    status.dataset.accuracyState = failed ? 'fail' : plotted ? 'pass' : 'pending';
    const progress = pending
      ? ` · ${pending.sources / 1000}K accumulating ${pending.methods[0].count}/${required} repetitions`
      : '';
    status.textContent = `${complete.length}/${quadratureSourceCounts.length} source sizes complete · ${plotted} qualified method points plotted${failed ? ` · ${failed} failed method cells excluded` : ''}${unavailable ? ` · ${unavailable} timestamp cells unavailable/below resolution` : ''}${progress}. Each completed source size adds its points immediately.`;
    const errorText = (value) => Number.isFinite(value) ? value.toExponential(2) : 'unavailable';
    $('#quadrature-accuracy').textContent = groups.length ? groups.map((group) =>
      `${group.sources / 1000}K: ` + group.methods.map((method, index) =>
        `${curveLabels[index]} ${planning.accuracyProfile} ${method.status} (${method.count}/${required}${method.rejected ? `; ${method.rejected} failed` : ''}; strict ${method.strictPassed}/${method.count}; εg=${errorText(method.gravityError)}, ε∇g=${errorText(method.gradientError)}${method.reasons.length ? '; reasons: ' + method.reasons.join(', ') : ''}${method.status === 'PASS' && !Number.isFinite(method.value) ? '; timestamp unavailable/below resolution' : ''})`
      ).join(' · ')).join('\n') : 'No completed repetitions for this Kρ × Nt cell.';
    // Preserve existing point nodes between new results (and their tooltips).
    // A seventh repetition updates immediately; task completion is not a gate.
    const key = JSON.stringify([planning.runId, planning.accuracyProfile,
      selection.densityModels, selection.targets, timingKey, groups]);
    if (key !== lastCurveRenderKey) {
      const series = curveSeries(groups, timingKey);
      drawChart($('#quadrature-chart'), series, {
        fitViewport: true,
        yLog: true,
        xCategories: quadratureSourceCounts,
        xLabel: 'source points',
        yLabel: `${timingTitle} median / min–max (ms)`,
        empty: unavailable ? 'GPU timestamps unavailable or below timer resolution; pipeline totals remain available.'
          : failed ? 'Completed repetitions failed accuracy; see details below.'
          : 'Waiting for the first source size to complete 7 passing repetitions…',
      });
      makeLegend($('#curve-legend'), series);
      lastCurveRenderKey = key;
    }
  }
  const bytes = (values) => {
    const total = (values ?? []).reduce((sum, value) => sum + value, 0);
    return total < 1048576 ? (total / 1024).toFixed(1) + ' KiB' : (total / 1048576).toFixed(1) + ' MiB';
  };
  const pressed = (selector, test) => $$(selector).forEach((button) => button.setAttribute('aria-pressed', String(test(button))));
  const makeLegend = (container, series) => {
    const entries = series.map((item) => {
      const span = document.createElement('span');
      span.style.setProperty('--series', item.color);
      span.dataset.lineStyle = item.dashed ? 'certified' : 'raw';
      span.textContent = item.label;
      return span;
    });
    container.replaceChildren(...entries);
  };
  function toggleDialog(dialog, visible) {
    if (visible && openDialog !== dialog) {
      returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      openDialog = dialog;
      dialog.inert = false;
      dialog.classList.add('open');
      dialog.setAttribute('aria-hidden', 'false');
      requestAnimationFrame(() => dialog.querySelector('button')?.focus());
    } else if (!visible && openDialog === dialog) {
      if (dialog.contains(document.activeElement)) returnFocus?.focus?.();
      openDialog = null;
      returnFocus = null;
      dialog.inert = true;
      dialog.classList.remove('open');
      dialog.setAttribute('aria-hidden', 'true');
    } else if (!visible) {
      dialog.inert = true;
      dialog.classList.remove('open');
      dialog.setAttribute('aria-hidden', 'true');
    }
  }
  const finiteText = (value, suffix = '') => Number.isFinite(value) ? `${formatAxis(value)} ${suffix}`.trim() : '--';
  const densityText = (row) => {
    const density = finiteText(row.density);
    const scale = Number.isFinite(row.densityScale) ? ` (${finiteText(row.densityScale, 'x')})` : '';
    const fit = Number.isFinite(row.fit) ? finiteText(row.fit * 100, '%') : '--';
    const holdout = finiteText(row.holdoutRmse);
    return `density ${density}${scale} · fit ${fit} · holdout RMSE ${holdout}`;
  };
  function renderPlanning(planning) {
    $$('[data-planning-accuracy]').forEach((select) => {
      select.value = planning.accuracyProfile ?? 'strict';
    });
    const limits = planning.accuracyLimits;
    const accuracyLabel = planning.accuracyProfile === 'screening'
      ? 'Screening comparison thresholds (relaxed; strict verdict shown separately)'
      : 'Strict comparison thresholds';
    const thresholdText = limits
      ? `RMS εg ≤ ${limits.gravity.toExponential(1)}, ε∇g ≤ ${limits.gradient.toExponential(1)}; p99 ≤ ${limits.gravityP99.toExponential(1)} / ${limits.gradientP99.toExponential(1)}; max ≤ ${limits.gravityMax.toExponential(1)} / ${limits.gradientMax.toExponential(1)}; drift ≤ ${limits.pericenterM} m`
      : 'Thresholds pending';
    $('#planning-accuracy-note').dataset.profile = planning.accuracyProfile;
    $('#quadrature-accuracy-policy').dataset.profile = planning.accuracyProfile;
    $('#planning-accuracy-note').textContent = `${accuracyLabel}. ${thresholdText}. ${planning.implementation ?? ''}`;
    $('#quadrature-accuracy-policy').textContent = `${accuracyLabel}. ${thresholdText}. Each source size is plotted after 7 passing repetitions.`;
    $('#quadrature-accuracy-summary').textContent = `Accuracy details — ${planning.accuracyProfile ?? 'strict'} profile (worst repetition RMS)`;
    pressed('[data-action="planning-metric"]', (button) => button.dataset.value === planning.metric);
    pressed('[data-action="planning-workload"]', (button) => button.dataset.value === planning.workload);
    const inversionRows = (lastSnapshot?.inversion?.results ?? []).filter(Boolean);
    const rows = planning.metric === 'density' || planning.metric === 'inversion-time'
      ? inversionRows
      : (planning.results ?? []).filter(Boolean);
    const [field, unit] = metricFields[planning.metric] ?? ['totalMs', 'ms'];
    if (planning.metric === 'density') {
      const displayed = lastSnapshot?.inversion?.displayed;
      $('#planning-result').textContent = rows.length
        ? rows.map((row) => `${row.method}: ${densityText(row)}`).join(' · ')
        : displayed
          ? `${displayed.method}: ${densityText(displayed)}`
          : 'Waiting for density inversion result.';
      return;
    }
    if (planning.metric === 'inversion-time') {
      $('#planning-result').textContent = rows.length
        ? rows.map((row) => `${row.method}: inversion ${finiteText(row.timeMs, unit)}`).join(' · ')
        : 'Waiting for density inversion result.';
      return;
    }
    $('#planning-result').dataset.accuracyState = planning.metric !== 'speedup' || !rows.length
      ? 'pending' : rows.some((row) => row.eligible !== true) ? 'fail' : 'pass';
    $('#planning-result').textContent = rows.length
      ? rows.map((row) => planning.metric === 'speedup' && row.eligible !== true
        ? `${row.method}: FAIL (${row.failureReasons?.join(', ') || 'accuracy'})`
        : `${row.method}: ${finiteText(row[field], unit)}${planning.metric === 'speedup' ? ` [${planning.accuracyProfile}; strict ${row.strictEligible ? 'PASS' : 'FAIL'}]` : ''}`).join(' · ')
      : 'No comparison result yet.';
  }
  function renderInversion(inversion, method) {
    const status = $('#inversion-status');
    const inversionButton = $('[data-action="inversion-start"]');
    const results = (inversion.results ?? []).filter(Boolean);
    const inversionSupported = method !== 'radial' && method !== 'werner';
    inversionButton.hidden = !inversionSupported;
    if (!inversionSupported) {
      status.dataset.state = 'forward-only';
      status.textContent = 'Forward-only method. Switch to Eq.106, Packed FFT, or FMM to invert the shared Radial trajectory.';
    } else if (inversion.error) {
      status.dataset.state = 'error';
      status.textContent = inversion.error;
    } else if (inversion.running) {
      status.dataset.state = 'running';
      status.textContent = 'Convex density inversion running…';
    } else if (results.length) {
      status.dataset.state = 'ready';
      status.textContent = results.map((row) => `${row.method}: ρ ${finiteText(row.density)} · ${finiteText(row.timeMs, 'ms')}`).join(' | ');
    } else {
      status.dataset.state = inversion.ready ? 'ready' : 'capturing';
      status.textContent = inversion.ready ? 'Trajectory captured. Inversion is ready.' : 'Capturing the common trajectory for inversion…';
    }
    // Queue the request even while the five-second capture is warming up.
    // Rust keeps it in the inversion state and starts it once the frozen
    // trajectory is valid; disabling here made the action appear broken.
    inversionButton.disabled = !inversionSupported || inversion.running;
  }
  const vectorText = (values) => (values ?? []).map((value) => Number(value).toFixed(3)).join(', ');
  function renderTrajectoryControls(inversion, method) {
    const panel = $('#trajectory-controls');
    const visible = inversion.ready && method !== 'radial' && method !== 'werner' && (inversion.trajectory?.length ?? 0) >= 2;
    panel.hidden = !visible;
    if (!visible || editingTrajectory) return;
    const fields = [];
    inversion.trajectory.forEach((knot, index) => {
      for (const [field, label] of [['position', 'P'], ['velocity', 'V']]) {
        const input = document.createElement('input');
        input.type = 'text';
        input.inputMode = 'decimal';
        input.autocomplete = 'off';
        input.spellcheck = false;
        input.value = vectorText(knot[field]);
        input.dataset.trajectoryField = field;
        input.dataset.trajectoryIndex = String(index);
        const row = document.createElement('label');
        row.className = 'trajectory-field';
        row.append(Object.assign(document.createElement('span'), { textContent: `${label}${index + 1}` }), input);
        fields.push(row);
      }
    });
    $('#trajectory-fields').replaceChildren(...fields);
  }
  function renderResidual(residual) {
    const card = $('#residual-card');
    card.hidden = !residual.visible;
    if (!residual.visible) return;
    $('#residual-order').textContent = `ORDER ${residual.order} · ${residual.mode}`;
    const remainder = Number.isFinite(residual.remainder) ? residual.remainder.toExponential(2) : '--';
    const relative = Number.isFinite(residual.relativeResidual) ? residual.relativeResidual.toExponential(2) : '--';
    $('#residual-diagnostics').textContent = `segments ${residual.segments} · accepted/rejected ${residual.accepted}/${residual.rejected} · Picard ${residual.picardIterations ?? '--'} · endpoint ${residual.endpointIterations ?? '--'} · remainder ${remainder} · relative ${relative}`;
  }
  // Vue/ECharts is the primary telemetry renderer. This lightweight SVG path is
  // deliberately kept as a resilience fallback: mobile port forwarding can
  // occasionally lose the deferred module request after the page and WASM have
  // already loaded. The fallback consumes the same snapshot and keeps the
  // coordinate system adaptive until the module becomes available again.
  const telemetryWindow = (samples, mapper, positiveOnly = false) => (samples ?? [])
    .map(mapper)
    .filter(([time, value]) => Number.isFinite(time) && Number.isFinite(value) && (!positiveOnly || value > 0))
    .slice(-96);
  const telemetryDomain = (points) => {
    if (!points.length) return null;
    const values = points.map(([time]) => time);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const padding = high === low ? Math.max(Math.abs(high) * .01, 1) : (high - low) * .04;
    return [low - padding, high + padding];
  };
  function fallbackTelemetrySvg(host, key) {
    let svg = host.querySelector(`svg[data-telemetry-fallback="${key}"]`);
    if (!svg) {
      svg = svgNode('svg', { 'data-telemetry-fallback': key, role: 'img' });
      host.replaceChildren(svg);
    }
    return svg;
  }
  function renderTelemetryFallback(snapshot) {
    if (window.ryuguTelemetryReady) return;
    const residual = telemetryWindow(snapshot.eq106Residual?.samples, (sample) => [Number(sample.time), Number(sample.epsilon)], true);
    const jacobi = telemetryWindow(snapshot.jacobi, (sample) => [Number(sample[0]), Number(sample[1])]);
    drawChart(fallbackTelemetrySvg($('#residual-chart'), 'residual'), [{ label: 'Eq.106 residual', color: '#36e7f2', points: residual }], {
      yLog: true,
      xLabel: 't (s)',
      yLabel: 'ε max',
      xDomain: telemetryDomain(residual),
      empty: 'Waiting for residual samples…',
    });
    drawChart(fallbackTelemetrySvg($('#jacobi-chart'), 'jacobi'), [{ label: 'Jacobi constant', color: '#43df81', points: jacobi }], {
      xLabel: 't (s)',
      yLabel: 'Cⱼ',
      xDomain: telemetryDomain(jacobi),
      empty: 'Waiting for Jacobi samples…',
    });
  }
  function renderPerformance(performance) {
    toggleDialog($('#performance-page'), performance.active);
    if (!performance.active) return;
    document.title = performance.measuring ? 'Benchmark running · Ryugu Dynamics' : 'Benchmark results · Ryugu Dynamics';
    pressed('[data-action="performance-method"]', (button) => Boolean(performance.enabled[Number(button.dataset.value)]));
    const fpsSeries = performance.fpsHistory.map((history, index) => ({
      color: methodColors[index],
      points: history.map((value, sample) => [sample, value]),
    }));
    const jacobiSeries = performance.jacobiHistory.map((history, index) => {
      const baseline = history[0]?.[1];
      const denominator = Math.max(Math.abs(baseline ?? 0), 1e-12);
      return {
        color: methodColors[index],
        points: history.map((sample, progress) => [progress, Math.abs((sample[1] - baseline) / denominator)]),
      };
    });
    if (!window.ryuguBenchmarkChartsReady) {
      drawChart($('#performance-fps-chart'), fpsSeries, {
        xLabel: 'measurement sample',
        yLabel: 'frames per second',
        minimumYDomain: [0, 60],
      });
      drawChart($('#performance-jacobi-chart'), jacobiSeries, { yLog: true, xLabel: 'measurement sample', yLabel: '|ΔCⱼ/Cⱼ₀| (log₁₀)' });
    }
    $('#performance-status').textContent = performance.measuring ? `Measuring ${methodLabels[performance.phase] ?? 'method'}…` : 'Benchmark complete. Repeat uses the same enabled methods.';
    const summaries = methodLabels.map((label, index) => {
      const span = document.createElement('span');
      span.textContent = `${label} ${performance.fps[index] > 0 ? performance.fps[index].toFixed(1) + ' FPS' : '--'}`;
      return span;
    });
    $('#performance-summary').replaceChildren(...summaries);
  }

  window.ryuguUi = {
    exportQuadrature,
    curveSelection: () => ({
      densityModels: Number($('#quadrature-density').value),
      targets: Number($('#quadrature-targets').value),
      scope: $('#quadrature-scope').value,
    }),
    activate(button) {
      if (!button?.dataset.action || button.disabled) return;
      returnFocus = button;
      const action = button.dataset.action;
      if (action === 'quadrature-start') window.ryuguCapture?.begin();
      const value = action === 'normals'
        ? button.getAttribute('aria-pressed') !== 'true'
        : action === 'section'
          ? button.getAttribute('aria-pressed') !== 'true'
          : button.dataset.value ?? null;
      push(action, value, action === 'quadrature-start' ? window.ryuguUi.curveSelection() : {});
    },
    takeAction: () => queue.shift() ?? '',
    resumeQuadrature(selection) {
      const densityModels = Number(selection?.densityModels);
      const targets = Number(selection?.targets);
      const scope = selection?.scope === 'all' ? 'all' : 'selected';
      if (![1, 4, 16, 64, 256, 512, 1024].includes(densityModels)
        || ![8, 64, 241, 1024, 8192].includes(targets)) return false;
      queue.push(JSON.stringify({
        type: 'quadrature-start',
        value: null,
        densityModels,
        targets,
        scope,
      }));
      return true;
    },
    render(snapshot) {
      if (snapshot.planning.curve == null) {
        snapshot.planning.curve = snapshot.planning.runId === lastSnapshot?.planning.runId
          ? lastSnapshot.planning.curve : [];
      }
      lastSnapshot = snapshot;
      // Do not consume the restart marker here. The bootstrap script reads it
      // before the first WASM snapshot is published; consuming it in render
      // would erase a queued recovery before the action system can see it.
      // Archiving is independent of repainting and must never throw through
      // the WASM UI bridge and interrupt the numerical engine.
      try { window.ryuguCapture?.observe(snapshot.planning); }
      catch (error) {
        $('#quadrature-capture-status').textContent = `Screenshot export error: ${error.message}; calculation continues and export will retry.`;
      }
      if (document.visibilityState === 'hidden') return;
      window.dispatchEvent(new CustomEvent('ryugu-snapshot', { detail: snapshot }));
      if (!snapshot.performance.active) document.title = 'Ryugu Dynamics Laboratory';
      $('#fps').textContent = 'FPS ' + snapshot.fps.toFixed(0);
      $('#health-dot').style.background = snapshot.runtimeError ? '#ff6262' : '#43e58a';
      $('#method-label').textContent = snapshot.methodLabel;
      const activeMemoryIndex = methodKeys.indexOf(snapshot.method);
      const activeMemory = Number.isFinite(snapshot.activeVramBytes)
        ? snapshot.activeVramBytes
        : activeMemoryIndex >= 0 ? snapshot.memoryBytes[activeMemoryIndex] : 0;
      $('#vram').textContent = bytes([activeMemory]);
      $('#acceleration').value = snapshot.acceleration;
      $('#acceleration-out').textContent = snapshot.acceleration + '×';
      pressed('[data-action="method"]', (button) => button.dataset.value === snapshot.method);
      pressed('[data-action="camera"]', (button) => button.dataset.value === snapshot.camera);
      $('[data-action="normals"]').setAttribute('aria-pressed', String(snapshot.normals));
      $('[data-action="section"]').setAttribute('aria-pressed', String(snapshot.section));
      if (!editingProbe) {
        Object.entries(snapshot.probe).forEach(([key, value]) => {
          const input = $('#probe-' + key);
          const output = $('#probe-' + key + '-out');
          if (input) input.value = value;
          if (output) output.textContent = key === 'speed' ? value.toFixed(3) : value.toFixed(0);
        });
      }
      $('#planning-status').textContent = snapshot.planning.status;
      $('#modal-status').textContent = snapshot.planning.workload === 'quadrature'
        ? snapshot.planning.status : 'Choose parameters, then press Run to start the quadrature task.';
      $('#quadrature-state').textContent = snapshot.planning.running ? Math.round(snapshot.planning.sourceCount / 1000) + 'K · R' + snapshot.planning.repeat : 'IDLE';
      $$('[data-action="quadrature-start"]').forEach((button) => { button.disabled = snapshot.planning.running; });
      const plan = snapshot.planning;
      const cells = plan.scope === 'all' ? 'all 35 Kρ × Nt combinations' : `Kρ=${plan.densityModels}, Nt=${plan.targets}`;
      $('#quadrature-run-scope').textContent = plan.workload === 'quadrature'
        ? `Task: ${cells} · 9 source sizes × 7 repeats · randomized order · median / min–max`
        : 'Choose Kρ, Nt and scope, then Run · 9 source sizes × 7 repeats · randomized order';
      $('#quadrature-implementation').textContent = plan.implementation ?? '';
      $('#quadrature-work').textContent = plan.workload === 'quadrature'
        ? `${Math.floor(plan.workCompleted ?? 0).toLocaleString()} / ${Math.floor(plan.workTotal ?? 0).toLocaleString()} estimated operation units · source/basis/FFT/RHS/target/reference work. Geometry-dependent work is estimated, not measured FLOPs or elapsed time. 100% requires all computation and validation to finish.`
        : 'No quadrature work is launched by opening this window.';
      $('#repeat-benchmark').disabled = !snapshot.performance.active || snapshot.performance.measuring;
      toggleDialog($('#quadrature-modal'), snapshot.planning.visible);
      const selection = window.ryuguUi.curveSelection();
      if (snapshot.planning.visible) renderQuadrature(snapshot.planning, selection);
      // The fullscreen benchmark covers these charts. Keep state snapshots,
      // but do not rebuild invisible SVGs/tables while it is open.
      if (!snapshot.planning.visible) {
        renderResidual(snapshot.eq106Residual);
        renderTelemetryFallback(snapshot);
        renderPlanning(snapshot.planning);
        renderInversion(snapshot.inversion, snapshot.method);
        renderTrajectoryControls(snapshot.inversion, snapshot.method);
        renderPerformance(snapshot.performance);
      }
      $('#runtime-message').textContent = snapshot.runtimeError ?? '';
      if (snapshot.planning.running) {
        try {
          sessionStorage.removeItem('ryugu-device-lost-attempts');
          sessionStorage.removeItem('ryugu-device-lost-reload-pending');
        } catch { /* optional */ }
      }
      toggleDialog($('#runtime-modal'), Boolean(snapshot.runtimeError));
    },
    get snapshot() { return lastSnapshot; },
  };
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && lastSnapshot) window.ryuguUi.render({ ...lastSnapshot });
  });
  // Opening the modal or expanding accuracy details changes the viewport,
  // even if no new numerical result arrives. SVG needs only a new viewBox;
  // it has no hidden-container renderer initialization to get stranded in.
  if (typeof ResizeObserver !== 'undefined') {
    let sizeKey = '';
    let resizeFrame = 0;
    const curveResize = new ResizeObserver(([entry]) => {
      if (!lastSnapshot?.planning.visible || entry.contentRect.width <= 0 || entry.contentRect.height <= 0) return;
      const nextSize = `${Math.round(entry.contentRect.width)}x${Math.round(entry.contentRect.height)}`;
      if (nextSize === sizeKey) return;
      sizeKey = nextSize;
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
        lastCurveRenderKey = null;
        if (lastSnapshot?.planning.visible) renderQuadrature(lastSnapshot.planning, window.ryuguUi.curveSelection());
      });
    });
    curveResize.observe($('#quadrature-chart'));
  }
  for (const selector of ['#quadrature-density', '#quadrature-targets', '#quadrature-timing']) {
    $(selector).addEventListener('change', () => {
      if (lastSnapshot) window.ryuguUi.render({ ...lastSnapshot });
    });
  }
  $$('[data-planning-accuracy]').forEach((select) => {
    select.addEventListener('change', () => push('planning-accuracy', select.value));
  });
})();
