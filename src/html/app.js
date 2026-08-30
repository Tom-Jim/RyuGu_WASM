import { createApp, computed, ref, onMounted, onBeforeUnmount } from 'vue';
import VChart from 'vue-echarts';
import { use } from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { SVGRenderer } from 'echarts/renderers';

use([LineChart, GridComponent, TooltipComponent, SVGRenderer]);

const clampPercent = (value) => Math.min(100, Math.max(0, Number(value) || 0));
const emptyProgress = () => ({ runId: null, progress: 0, accuracy: 0, running: false });
const mergeProgress = (previous, planning) => {
  const sameRun = previous.runId === planning.runId;
  return {
    runId: planning.runId,
    progress: sameRun
      ? Math.max(previous.progress, clampPercent(planning.progress))
      : clampPercent(planning.progress),
    accuracy: sameRun
      ? Math.max(previous.accuracy, clampPercent(planning.accuracy))
      : clampPercent(planning.accuracy),
    running: Boolean(planning.running),
  };
};

const useViewportNavigation = () => {
  const view = ref({ zoom: 1, x: 0, y: 0 });
  const minZoom = 0.75;
  const maxZoom = 2.5;
  const touchPointers = new Map();
  const consumedTouches = new Set();
  let canvas = null;
  let viewportFrame = null;
  let rightPan = null;
  let pinch = null;

  const isCanvasTarget = (target) => target === canvas;
  const midpoint = (first, second) => ({
    x: (first.x + second.x) / 2,
    y: (first.y + second.y) / 2,
  });
  const distance = (first, second) => Math.hypot(first.x - second.x, first.y - second.y);
  const displayCenter = () => ({ x: innerWidth / 2, y: innerHeight / 2 });
  const applyView = () => {
    viewportFrame?.style.setProperty('--user-zoom', String(view.value.zoom));
    viewportFrame?.style.setProperty('--view-offset-x', `${view.value.x}px`);
    viewportFrame?.style.setProperty('--view-offset-y', `${view.value.y}px`);
  };
  const stopEvent = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  const zoomAt = (previousAnchor, nextAnchor, nextZoom) => {
    const oldZoom = view.value.zoom;
    const zoom = Math.min(maxZoom, Math.max(minZoom, nextZoom));
    const ratio = zoom / oldZoom;
    const center = displayCenter();
    view.value.x = nextAnchor.x - center.x - ratio * (previousAnchor.x - center.x - view.value.x);
    view.value.y = nextAnchor.y - center.y - ratio * (previousAnchor.y - center.y - view.value.y);
    view.value.zoom = zoom;
    applyView();
  };
  const updatePinch = () => {
    if (touchPointers.size !== 2) return;
    const [first, second] = [...touchPointers.values()];
    const nextMidpoint = midpoint(first, second);
    const nextDistance = Math.max(distance(first, second), 1);
    if (!pinch) {
      pinch = { midpoint: nextMidpoint, distance: nextDistance };
      return;
    }
    zoomAt(pinch.midpoint, nextMidpoint, view.value.zoom * nextDistance / pinch.distance);
    pinch = { midpoint: nextMidpoint, distance: nextDistance };
  };

  const onPointerDown = (event) => {
    if (!isCanvasTarget(event.target)) return;
    if (event.pointerType === 'mouse' && event.button === 2) {
      rightPan = { id: event.pointerId, x: event.clientX, y: event.clientY };
      canvas.setPointerCapture?.(event.pointerId);
      document.documentElement.classList.add('is-view-panning');
      stopEvent(event);
      return;
    }
    if (event.pointerType !== 'touch') return;
    touchPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (touchPointers.size === 2) {
      touchPointers.forEach((_, pointerId) => consumedTouches.add(pointerId));
      canvas.setPointerCapture?.(event.pointerId);
      updatePinch();
      stopEvent(event);
    }
  };
  const onPointerMove = (event) => {
    if (rightPan?.id === event.pointerId) {
      view.value.x += event.clientX - rightPan.x;
      view.value.y += event.clientY - rightPan.y;
      rightPan = { id: event.pointerId, x: event.clientX, y: event.clientY };
      applyView();
      stopEvent(event);
      return;
    }
    if (!touchPointers.has(event.pointerId)) return;
    touchPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (consumedTouches.has(event.pointerId)) {
      if (touchPointers.size === 2) updatePinch();
      stopEvent(event);
    }
  };
  const onPointerEnd = (event) => {
    if (rightPan?.id === event.pointerId) {
      rightPan = null;
      document.documentElement.classList.remove('is-view-panning');
    }
    if (touchPointers.delete(event.pointerId)) {
      consumedTouches.delete(event.pointerId);
      pinch = null;
    }
  };
  const onWheel = (event) => {
    if (!isCanvasTarget(event.target)) return;
    const pixels = event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? event.deltaY * 16
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? event.deltaY * innerHeight
        : event.deltaY;
    zoomAt({ x: event.clientX, y: event.clientY }, { x: event.clientX, y: event.clientY }, view.value.zoom * Math.exp(-pixels * 0.0015));
    stopEvent(event);
  };
  const onContextMenu = (event) => {
    if (!isCanvasTarget(event.target)) return;
    stopEvent(event);
  };

  onMounted(() => {
    viewportFrame = document.getElementById('viewport-frame');
    canvas = document.getElementById('bevy');
    applyView();
    document.addEventListener('pointerdown', onPointerDown, { capture: true, passive: false });
    document.addEventListener('pointermove', onPointerMove, { capture: true, passive: false });
    document.addEventListener('pointerup', onPointerEnd, { capture: true, passive: true });
    document.addEventListener('pointercancel', onPointerEnd, { capture: true, passive: true });
    document.addEventListener('wheel', onWheel, { capture: true, passive: false });
    document.addEventListener('contextmenu', onContextMenu, { capture: true, passive: false });
  });
  onBeforeUnmount(() => {
    document.removeEventListener('pointerdown', onPointerDown, true);
    document.removeEventListener('pointermove', onPointerMove, true);
    document.removeEventListener('pointerup', onPointerEnd, true);
    document.removeEventListener('pointercancel', onPointerEnd, true);
    document.removeEventListener('wheel', onWheel, true);
    document.removeEventListener('contextmenu', onContextMenu, true);
  });
};

const app = createApp({
  setup() {
    useViewportNavigation();
    const tracked = ref({ first: emptyProgress(), stress: emptyProgress() });
    const labels = { first: 'First', stress: 'Stress', quadrature: 'Quadrature' };
    const update = (event) => {
      const next = event.detail?.planning;
      if (!next || !(next.workload in tracked.value)) return;
      tracked.value[next.workload] = mergeProgress(tracked.value[next.workload], next);
    };
    onMounted(() => window.addEventListener('ryugu-snapshot', update));
    onBeforeUnmount(() => window.removeEventListener('ryugu-snapshot', update));
    const workloads = ['first', 'stress'];
    return { tracked, labels, workloads };
  },
  template: `
    <section class="mt-2 grid gap-2" aria-live="polite" aria-label="Calculation progress">
      <div v-for="kind in workloads" :key="kind" class="rounded-md border border-cyan-200/15 bg-black/20 px-2 py-1.5">
        <div class="flex items-center justify-between gap-2 font-mono text-[10px] text-slate-300">
          <span>{{ labels[kind] }} calculation</span>
          <span>{{ Math.round(tracked[kind].progress) }}%</span>
        </div>
        <div class="mt-1 h-1.5 overflow-hidden rounded bg-cyan-950/80" role="progressbar" :aria-label="labels[kind] + ' calculation progress'" :aria-valuenow="Math.round(tracked[kind].progress)" aria-valuemin="0" aria-valuemax="100">
          <div class="h-full rounded bg-cyan-300" :style="{ width: tracked[kind].progress + '%' }"></div>
        </div>
        <div class="mt-1 font-mono text-[9px] text-slate-500">{{ tracked[kind].running ? Math.round(tracked[kind].progress) + '% complete' : tracked[kind].progress >= 100 ? 'Complete' : tracked[kind].progress > 0 ? 'Stopped' : 'Ready' }}</div>
      </div>
    </section>
  `,
});

app.mount('#planning-progress');

createApp({
  setup() {
    const snapshot = ref(null);
    const planning = computed(() => snapshot.value?.planning ?? { progress: 0, accuracy: 0, workload: 'quadrature', running: false });
    const tracked = ref(emptyProgress());
    const update = (event) => {
      snapshot.value = event.detail;
      const next = event.detail?.planning;
      if (next?.workload === 'quadrature') tracked.value = mergeProgress(tracked.value, next);
    };
    onMounted(() => window.addEventListener('ryugu-snapshot', update));
    onBeforeUnmount(() => window.removeEventListener('ryugu-snapshot', update));
    return { planning, tracked };
  },
  template: `<div v-if="planning.workload === 'quadrature'" class="mt-1 min-w-64"><div class="flex justify-between font-mono text-[10px] text-slate-300"><span>Quadrature calculation</span><span>{{ Math.round(tracked.progress) }}% complete</span></div><div class="mt-1 h-1.5 overflow-hidden rounded bg-cyan-950/80" role="progressbar" aria-label="Quadrature calculation progress" :aria-valuenow="Math.round(tracked.progress)" aria-valuemin="0" aria-valuemax="100"><div class="h-full rounded bg-cyan-300" :style="{ width: tracked.progress + '%' }"></div></div></div>`,
}).mount('#quadrature-progress');

const LIVE_SAMPLE_WINDOW = 96;
const chartNumber = (value) => Number(value).toExponential(3).replace(/\.?(?:0+)e/, 'e').replace('e+', 'e');
const chartAxisNumber = (value) => Number(value).toExponential(5).replace(/\.?(?:0+)e/, 'e').replace('e+', 'e');

function recentTelemetryPoints(snapshot, kind) {
  const source = kind === 'jacobi'
    ? snapshot?.jacobi ?? []
    : snapshot?.eq106Residual?.samples ?? [];
  const points = source
    .map((sample) => kind === 'jacobi' ? [Number(sample[0]), Number(sample[1])] : [Number(sample.time), Number(sample.epsilon)])
    .filter(([time, value]) => Number.isFinite(time) && Number.isFinite(value) && (kind !== 'residual' || value > 0));
  return points.slice(-LIVE_SAMPLE_WINDOW);
}

function paddedDomain(points, logarithmic) {
  const values = points.map(([, value]) => value);
  if (!values.length) return null;
  const low = Math.min(...values);
  const high = Math.max(...values);
  if (logarithmic) {
    const safeLow = Math.max(low, Number.MIN_VALUE);
    if (safeLow === high) return [safeLow / 2, high * 2];
    return [safeLow / 1.35, high * 1.35];
  }
  const span = high - low;
  const pad = span > 0 ? span * 0.13 : Math.max(Math.abs(high) * 0.035, 1e-9);
  return [low - pad, high + pad];
}

function paddedTimeDomain(points) {
  if (!points.length) return null;
  const times = points.map(([time]) => time);
  const low = Math.min(...times);
  const high = Math.max(...times);
  const span = high - low;
  const pad = span > 0 ? span * 0.04 : Math.max(Math.abs(high) * 0.01, 1);
  return [low - pad, high + pad];
}

function telemetryOption(points, kind) {
  const isResidual = kind === 'residual';
  const domain = paddedDomain(points, isResidual);
  const timeDomain = paddedTimeDomain(points);
  const color = isResidual ? '#36e7f2' : '#43df81';
  const last = points.at(-1);
  return {
    animation: false,
    backgroundColor: 'transparent',
    grid: { left: 48, right: 12, top: 10, bottom: 28, containLabel: false },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(2, 12, 16, .96)',
      borderColor: 'rgba(102, 232, 235, .55)',
      borderWidth: 1,
      padding: [6, 8],
      textStyle: { color: '#ddf5f6', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 11 },
      formatter: (items) => {
        const item = items?.[0];
        if (!item) return 'Waiting for samples';
        return `t ${chartNumber(item.value[0])} s<br/>${isResidual ? 'ε max' : 'Cⱼ'} ${chartNumber(item.value[1])}`;
      },
    },
    xAxis: {
      type: 'value',
      min: timeDomain?.[0],
      max: timeDomain?.[1],
      scale: true,
      name: 't (s)',
      nameLocation: 'middle',
      nameGap: 20,
      nameTextStyle: { color: '#789097', fontSize: 9, fontFamily: 'ui-monospace, monospace' },
      axisLine: { lineStyle: { color: 'rgba(120, 208, 213, .36)' } },
      axisTick: { show: false },
      splitLine: { show: true, lineStyle: { color: 'rgba(103, 193, 198, .10)' } },
      axisLabel: { color: '#7e9aa0', fontSize: 9, fontFamily: 'ui-monospace, monospace', formatter: chartNumber, hideOverlap: true },
    },
    yAxis: {
      type: isResidual ? 'log' : 'value',
      logBase: 10,
      name: isResidual ? 'ε max' : 'Cⱼ',
      nameLocation: 'middle',
      nameGap: 37,
      nameTextStyle: { color: '#789097', fontSize: 9, fontFamily: 'ui-monospace, monospace' },
      min: domain?.[0],
      max: domain?.[1],
      scale: true,
      axisLine: { lineStyle: { color: 'rgba(120, 208, 213, .36)' } },
      axisTick: { show: false },
      splitLine: { show: true, lineStyle: { color: 'rgba(103, 193, 198, .10)' } },
      axisLabel: { color: '#7e9aa0', fontSize: 9, fontFamily: 'ui-monospace, monospace', formatter: chartAxisNumber, hideOverlap: true },
    },
    series: [{
      type: 'line',
      name: isResidual ? 'Eq.106 residual' : 'Jacobi constant',
      data: points,
      showSymbol: false,
      clip: true,
      sampling: 'lttb',
      lineStyle: { color, width: 2 },
      itemStyle: { color },
      areaStyle: { color: isResidual ? 'rgba(54, 231, 242, .08)' : 'rgba(67, 223, 129, .07)' },
      markPoint: last ? { symbol: 'circle', symbolSize: 7, itemStyle: { color, borderColor: '#02090b', borderWidth: 1 }, label: { show: false }, data: [{ coord: last }] } : undefined,
    }],
  };
}

function mountTelemetryChart(target, kind) {
  createApp({
    components: { VChart },
    setup() {
      const snapshot = ref(null);
      const update = (event) => { snapshot.value = event.detail ?? null; };
      const points = computed(() => recentTelemetryPoints(snapshot.value, kind));
      const option = computed(() => telemetryOption(points.value, kind));
      const windowLabel = computed(() => {
        if (!points.value.length) return 'WAITING FOR SAMPLES';
        const [start] = points.value[0];
        const [end] = points.value.at(-1);
        return `${points.value.length} SAMPLES · ${chartNumber(start)}–${chartNumber(end)} s`;
      });
      onMounted(() => window.addEventListener('ryugu-snapshot', update));
      onBeforeUnmount(() => window.removeEventListener('ryugu-snapshot', update));
      return { option, windowLabel };
    },
    template: '<v-chart class="live-echart" :option="option" :autoresize="{ throttle: 80 }" renderer="svg" role="img" :aria-label="windowLabel" />',
  }).mount(target);
}

mountTelemetryChart('#residual-chart', 'residual');
mountTelemetryChart('#jacobi-chart', 'jacobi');
