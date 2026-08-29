import { createApp, computed, ref, onMounted, onBeforeUnmount } from 'vue';

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
