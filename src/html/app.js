import { createApp, computed, ref, shallowRef, onMounted, onBeforeUnmount } from './vue-compiler.js';
import VChart from 'vue-echarts';
import { use } from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { SVGRenderer } from 'echarts/renderers';

use([LineChart, GridComponent, TooltipComponent, SVGRenderer]);

const emptyProgress = () => ({ runId: null, progress: 0, accuracy: 0, running: false, completed: false });
const mergeProgress = (_previous, planning) => window.ryuguPlanningProgress(planning);

const app = createApp({
  setup() {
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
          <span>{{ tracked[kind].progress }}%</span>
        </div>
        <div class="mt-1 h-1.5 overflow-hidden rounded bg-cyan-950/80" role="progressbar" :aria-label="labels[kind] + ' calculation progress'" :aria-valuenow="tracked[kind].progress" aria-valuemin="0" aria-valuemax="100">
          <div class="h-full rounded bg-cyan-300" :style="{ width: tracked[kind].progress + '%' }"></div>
        </div>
        <div class="mt-1 font-mono text-[9px] text-slate-500">{{ tracked[kind].running ? 'Running · ' + tracked[kind].progress + '%' : tracked[kind].completed ? 'Complete' : tracked[kind].progress > 0 ? 'Stopped · ' + tracked[kind].progress + '%' : 'Ready' }}</div>
      </div>
    </section>
  `,
});

app.mount('#planning-progress');

createApp({
  setup() {
    const snapshot = shallowRef(null);
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
  template: `<div v-if="planning.workload === 'quadrature'" class="mt-1 min-w-64"><div class="flex justify-between font-mono text-[10px] text-slate-300"><span>Quadrature calculation</span><span>{{ tracked.progress }}% complete</span></div><div class="mt-1 h-1.5 overflow-hidden rounded bg-cyan-950/80" role="progressbar" aria-label="Quadrature calculation progress" :aria-valuenow="tracked.progress" aria-valuemin="0" aria-valuemax="100"><div class="h-full rounded bg-cyan-300" :style="{ width: tracked.progress + '%' }"></div></div></div>`,
}).mount('#quadrature-progress');

const LIVE_SAMPLE_WINDOW = 96;
const chartNumber = (value) => Number(value).toExponential(3).replace(/\.?(?:0+)e/, 'e').replace('e+', 'e');
const chartAxisNumber = (value) => Number(value).toExponential(5).replace(/\.?(?:0+)e/, 'e').replace('e+', 'e');

function frequencyDisplayScale(points) {
  const maximum = Math.max(0, ...points.map(([, value]) => Math.abs(value)));
  if (!Number.isFinite(maximum) || maximum === 0) return { factor: 1, exponent: 0 };
  const exponent = Math.floor(Math.log10(maximum));
  return { factor: 10 ** exponent, exponent };
}

function recentTelemetryPoints(snapshot) {
  const transform = snapshot?.method === 'frequency_domain';
  const points = (transform ? snapshot?.frequencyDomain ?? [] : snapshot?.jacobi ?? [])
    .map((sample) => [Number(sample[0]), Number(sample[1])])
    .filter(([time, value]) => Number.isFinite(time) && Number.isFinite(value));
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

function telemetryOption(points, transform) {
  const frequencyScale = transform ? frequencyDisplayScale(points) : { factor: 1, exponent: 0 };
  const plottedPoints = transform
    ? points.map(([frequency, magnitude]) => [frequency, magnitude / frequencyScale.factor])
    : points;
  const domain = paddedDomain(plottedPoints, false);
  const timeDomain = paddedTimeDomain(points);
  const color = transform ? '#36e7f2' : '#43df81';
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
        return transform
          ? `σ ${chartNumber(item.value[0])} s⁻¹<br/>‖g̃γ(σ)‖ ${chartNumber(item.value[1] * frequencyScale.factor)}`
          : `t ${chartNumber(item.value[0])} s<br/>Cⱼ ${chartNumber(item.value[1])}`;
      },
    },
    xAxis: {
      type: 'value',
      min: timeDomain?.[0],
      max: timeDomain?.[1],
      scale: true,
      name: transform ? 'σ (s⁻¹)' : 't (s)',
      nameLocation: 'middle',
      nameGap: 20,
      nameTextStyle: { color: '#789097', fontSize: 9, fontFamily: 'ui-monospace, monospace' },
      axisLine: { lineStyle: { color: 'rgba(120, 208, 213, .36)' } },
      axisTick: { show: false },
      splitLine: { show: true, lineStyle: { color: 'rgba(103, 193, 198, .10)' } },
      axisLabel: { color: '#7e9aa0', fontSize: 9, fontFamily: 'ui-monospace, monospace', formatter: chartNumber, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      logBase: 10,
      name: transform ? `‖g̃γ(σ)‖ × 10^${frequencyScale.exponent}` : 'Cⱼ',
      nameLocation: 'middle',
      nameGap: 37,
      nameTextStyle: { color: '#789097', fontSize: 9, fontFamily: 'ui-monospace, monospace' },
      min: domain?.[0],
      max: domain?.[1],
      scale: true,
      axisLine: { lineStyle: { color: 'rgba(120, 208, 213, .36)' } },
      axisTick: { show: false },
      splitLine: { show: true, lineStyle: { color: 'rgba(103, 193, 198, .10)' } },
      axisLabel: {
        color: '#7e9aa0',
        fontSize: 9,
        fontFamily: 'ui-monospace, monospace',
        formatter: transform ? (value) => Number(value).toFixed(3) : chartAxisNumber,
        hideOverlap: true,
      },
    },
    series: [{
      type: 'line',
      name: transform ? 'Frequency-domain algorithm norm' : 'Jacobi constant',
      data: plottedPoints,
      showSymbol: transform,
      symbol: 'circle',
      symbolSize: transform ? 4 : 0,
      clip: true,
      sampling: transform ? undefined : 'lttb',
      smooth: false,
      lineStyle: { color, width: 2 },
      itemStyle: { color },
      areaStyle: { color: transform ? 'rgba(54, 231, 242, .08)' : 'rgba(67, 223, 129, .07)' },
      markPoint: last ? { symbol: 'circle', symbolSize: 7, itemStyle: { color, borderColor: '#02090b', borderWidth: 1 }, label: { show: false }, data: [{ coord: last }] } : undefined,
    }],
  };
}

function mountTelemetryChart(target) {
  createApp({
    components: { VChart },
    setup() {
      // A phone can finish the WASM boot before its delayed module fetch. Use
      // the most recent UI snapshot immediately instead of waiting for the
      // next render tick.
      const snapshot = shallowRef(window.ryuguUi?.snapshot ?? null);
      const update = (event) => {
        const next = event.detail;
        if (!next) return;
        snapshot.value = next;
      };
      const points = computed(() => recentTelemetryPoints(snapshot.value));
      const transform = computed(() => snapshot.value?.method === 'frequency_domain');
      const option = computed(() => telemetryOption(points.value, transform.value));
      const windowLabel = computed(() => {
        if (!points.value.length) return 'WAITING FOR SAMPLES';
        const [start] = points.value[0];
        const [end] = points.value.at(-1);
        const unit = transform.value ? 's⁻¹' : 's';
        return `${points.value.length} SAMPLES · ${chartNumber(start)}–${chartNumber(end)} ${unit}`;
      });
      onMounted(() => window.addEventListener('ryugu-snapshot', update));
      onBeforeUnmount(() => window.removeEventListener('ryugu-snapshot', update));
      return { option, windowLabel };
    },
    template: '<v-chart class="live-echart" :option="option" :autoresize="{ throttle: 80 }" :init-options="{ renderer: \'svg\' }" role="img" :aria-label="windowLabel" />',
  }).mount(target);
}

mountTelemetryChart('#jacobi-chart');
window.ryuguTelemetryReady = true;

const benchmarkColors = ['#42dc77', '#ffb23d', '#ff7d89', '#58c8ff', '#36e7f2', '#a8f7bd'];
const benchmarkLabels = ['FMM', 'FFT', 'Werner', 'Radial', 'Frequency-domain'];

const modalAxisStyle = {
  axisLine: { lineStyle: { color: 'rgba(120, 208, 213, .38)' } },
  axisTick: { show: false },
  splitLine: { show: true, lineStyle: { color: 'rgba(103, 193, 198, .12)' } },
  axisLabel: { color: '#8fa9ae', fontSize: 10, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', hideOverlap: true },
  nameTextStyle: { color: '#8fa9ae', fontSize: 10, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' },
};
const modalChartFrame = () => ({
  animation: false,
  backgroundColor: 'transparent',
  grid: { left: 62, right: 22, top: 14, bottom: 48 },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(2, 12, 16, .96)',
    borderColor: 'rgba(102, 232, 235, .55)',
    textStyle: { color: '#ddf5f6', fontFamily: 'ui-monospace, monospace', fontSize: 11 },
  },
});

function performanceOption(snapshot, kind) {
  const histories = kind === 'fps'
    ? snapshot?.performance?.fpsHistory ?? []
    : snapshot?.performance?.diagnosticHistory ?? [];
  const series = histories.map((history, index) => {
    const baseline = Number(history?.[0]?.[1] ?? 0);
    const points = (history ?? []).map((sample, pointIndex) => {
      const value = kind === 'fps'
        ? Number(sample)
        : Math.max(Math.abs((Number(sample?.[1]) - baseline) / Math.max(Math.abs(baseline), 1e-12)), 1e-16);
      return [pointIndex, value];
    }).filter(([, value]) => Number.isFinite(value));
    return {
      type: 'line',
      name: benchmarkLabels[index] ?? `Method ${index + 1}`,
      data: points,
      showSymbol: false,
      sampling: 'lttb',
      lineStyle: { width: 2, color: benchmarkColors[index] ?? '#d9eef0' },
      itemStyle: { color: benchmarkColors[index] ?? '#d9eef0' },
    };
  });
  return {
    ...modalChartFrame(),
    xAxis: { ...modalAxisStyle, type: 'value', name: 'measurement sample', minInterval: 1 },
    yAxis: {
      ...modalAxisStyle,
      type: kind === 'fps' ? 'value' : 'log',
      logBase: 10,
      name: kind === 'fps' ? 'frames / second' : '|ΔD / D₀|',
      min: kind === 'fps' ? 0 : undefined,
      scale: true,
      axisLabel: { ...modalAxisStyle.axisLabel, formatter: chartAxisNumber },
    },
    series,
  };
}

// The quadrature curve is a native SVG owned by ui.js. It updates as each
// source cell completes, including while the modal is hidden. Do not mount a
// second renderer into that node or gate it on ECharts/module readiness.

function mountBenchmarkChart(target, makeOption) {
  createApp({
    components: { VChart },
    setup() {
      const snapshot = shallowRef(window.ryuguUi?.snapshot ?? null);
      const update = (event) => { snapshot.value = event.detail ?? null; };
      onMounted(() => window.addEventListener('ryugu-snapshot', update));
      onBeforeUnmount(() => window.removeEventListener('ryugu-snapshot', update));
      return { option: computed(() => makeOption(snapshot.value)) };
    },
    template: '<v-chart class="modal-echart" :option="option" :autoresize="{ throttle: 80 }" :init-options="{ renderer: \'svg\' }" />',
  }).mount(target);
}

mountBenchmarkChart('#performance-fps-chart', (snapshot) => performanceOption(snapshot, 'fps'));
mountBenchmarkChart('#performance-jacobi-chart', (snapshot) => performanceOption(snapshot, 'jacobi'));
window.ryuguBenchmarkChartsReady = true;
