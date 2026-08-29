(() => {
  const queue = [];
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const push = (type, value, extra = {}) => queue.push(JSON.stringify({ type, value, ...extra }));
  const methodKeys = ['radial', 'werner', 'eq106', 'fft', 'fmm'];
  const methodLabels = ['Radial', 'Werner', 'Eq.106', 'Packed FFT', 'FMM'];
  const methodColors = ['#58c8ff', '#ff7d89', '#36e7f2', '#ffb23d', '#42dc77'];
  const curveColors = ['#36e7f2', '#9af8ff', '#ffb23d', '#ffe071', '#42dc77', '#a8f7bd'];
  const curveLabels = ['Eq.106 raw', 'Eq.106 certified', 'Packed FFT raw', 'Packed FFT certified', 'FMM raw', 'FMM certified'];
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
  function drawChart(svg, series, { yLog = false, xLabel = '', yLabel = '', xDomain = null, xCategories = null, minimumYDomain = null, empty = 'Waiting for samples…' } = {}) {
    svg.replaceChildren();
    const width = 900;
    const height = 430;
    const margin = { l: 104, r: 28, t: 22, b: 70 };
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const points = series
      .flatMap((item) => item.points)
      .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]) && (!yLog || point[1] > 0));
    const categoryIndex = xCategories ? new Map(xCategories.map((value, index) => [value, index])) : null;
    const transformX = (value) => categoryIndex ? categoryIndex.get(value) : value;
    const transformY = (value) => yLog ? Math.log10(value) : value;
    const xs = points.map((point) => transformX(point[0]));
    const ys = points.map((point) => transformY(point[1]));
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
      const valid = item.points.filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]) && (!yLog || point[1] > 0));
      if (valid.length) {
        const path = valid.map((point, index) => `${index ? 'L' : 'M'}${pixelX(point[0]).toFixed(2)},${pixelY(point[1]).toFixed(2)}`).join(' ');
        const pathAttributes = { d: path, stroke: item.color, class: 'chart-line' };
        if (item.dashed) {
          pathAttributes['stroke-dasharray'] = '8 5';
        }
        svg.append(svgNode('path', pathAttributes));
        valid.forEach((point) => {
          const marker = svgNode('circle', {
            cx: pixelX(point[0]).toFixed(2),
            cy: pixelY(point[1]).toFixed(2),
            r: 4,
            fill: item.color,
            class: 'chart-point',
          });
          const sourceLabel = categoryIndex ? `${point[0] / 1000}K` : formatAxis(point[0]);
          marker.append(svgNode('title', {}, `${item.label} at ${sourceLabel}: ${formatAxis(point[1])} ms`));
          svg.append(marker);
        });
      }
    });
  }

  const median = (values) => {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    return sorted.length ? sorted[Math.floor(sorted.length / 2)] : NaN;
  };
  function curveSeries(samples) {
    const groups = new Map();
    samples.forEach((sample) => {
      if (!groups.has(sample.sources)) groups.set(sample.sources, Array.from({ length: 6 }, () => []));
      sample.times?.forEach((time, index) => {
        if (Number.isFinite(time) && time > 0) groups.get(sample.sources)[index].push(time);
      });
    });
    return curveLabels.map((label, index) => ({
      label,
      color: curveColors[index],
      dashed: index % 2 === 1,
      points: [...groups]
        .sort((a, b) => a[0] - b[0])
        .map(([sources, values]) => [sources, median(values[index])])
        .filter((point) => Number.isFinite(point[1])),
    }));
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
    $('#planning-result').textContent = rows.length
      ? rows.map((row) => `${row.method}: ${finiteText(row[field], unit)}`).join(' · ')
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
    const points = (residual.samples ?? []).map((sample) => [sample.time, sample.epsilon]);
    drawChart($('#residual-chart'), [{ color: '#36e7f2', points }], {
      yLog: true,
      xLabel: 'simulation time (s)',
      yLabel: 'max residual ε (log₁₀)',
      empty: 'Waiting for Eq.106 certified residuals…',
    });
    $('#residual-order').textContent = `ORDER ${residual.order} · ${residual.mode}`;
    const remainder = Number.isFinite(residual.remainder) ? residual.remainder.toExponential(2) : '--';
    const relative = Number.isFinite(residual.relativeResidual) ? residual.relativeResidual.toExponential(2) : '--';
    $('#residual-diagnostics').textContent = `segments ${residual.segments} · accepted/rejected ${residual.accepted}/${residual.rejected} · Picard ${residual.picardIterations ?? '--'} · endpoint ${residual.endpointIterations ?? '--'} · remainder ${remainder} · relative ${relative}`;
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
    drawChart($('#performance-fps-chart'), fpsSeries, {
      xLabel: 'measurement sample',
      yLabel: 'frames per second',
      minimumYDomain: [0, 60],
    });
    drawChart($('#performance-jacobi-chart'), jacobiSeries, { yLog: true, xLabel: 'measurement sample', yLabel: '|ΔCⱼ/Cⱼ₀| (log₁₀)' });
    $('#performance-status').textContent = performance.measuring ? `Measuring ${methodLabels[performance.phase] ?? 'method'}…` : 'Benchmark complete. Repeat uses the same enabled methods.';
    const summaries = methodLabels.map((label, index) => {
      const span = document.createElement('span');
      span.textContent = `${label} ${performance.fps[index] > 0 ? performance.fps[index].toFixed(1) + ' FPS' : '--'}`;
      return span;
    });
    $('#performance-summary').replaceChildren(...summaries);
  }

  window.ryuguUi = {
    activate(button) {
      if (!button?.dataset.action || button.disabled) return;
      returnFocus = button;
      const action = button.dataset.action;
      const value = action === 'normals'
        ? button.getAttribute('aria-pressed') !== 'true'
        : action === 'section'
          ? button.getAttribute('aria-pressed') !== 'true'
          : button.dataset.value ?? null;
      push(action, value);
    },
    takeAction: () => queue.shift() ?? '',
    render(snapshot) {
      lastSnapshot = snapshot;
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
      $('#modal-status').textContent = snapshot.planning.status;
      $('#quadrature-state').textContent = snapshot.planning.running ? Math.round(snapshot.planning.sourceCount / 1000) + 'K · R' + snapshot.planning.repeat : 'IDLE';
      $('#quadrature-start').disabled = snapshot.planning.running;
      $('#repeat-benchmark').disabled = !snapshot.performance.active || snapshot.performance.measuring;
      toggleDialog($('#quadrature-modal'), snapshot.planning.visible);
      const benchmarkSeries = curveSeries(snapshot.planning.curve);
      drawChart($('#quadrature-chart'), benchmarkSeries, {
        yLog: true,
        xCategories: quadratureSourceCounts,
        xLabel: 'distinct quadrature points (32K–8192K)',
        yLabel: 'measured total time (ms, log₁₀ scale)',
        empty: 'Waiting for the first completed 32K source point…',
      });
      makeLegend($('#curve-legend'), benchmarkSeries);
      drawChart($('#jacobi-chart'), [{ color: '#43df81', points: snapshot.jacobi.filter((point) => Number.isFinite(point[1])) }], { xLabel: 'simulation time (s)', yLabel: 'Cⱼ' });
      renderResidual(snapshot.eq106Residual);
      renderPlanning(snapshot.planning);
      renderInversion(snapshot.inversion, snapshot.method);
      renderTrajectoryControls(snapshot.inversion, snapshot.method);
      renderPerformance(snapshot.performance);
      $('#runtime-message').textContent = snapshot.runtimeError ?? '';
      toggleDialog($('#runtime-modal'), Boolean(snapshot.runtimeError));
    },
    get snapshot() { return lastSnapshot; },
  };
})();
