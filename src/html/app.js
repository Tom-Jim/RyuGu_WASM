import { createApp, computed, ref, onMounted, onBeforeUnmount } from 'vue';

const app = createApp({
  setup() {
    const snapshot = ref(null);
    const planning = computed(() => snapshot.value?.planning ?? { progress: 0, accuracy: 0, workload: 'first', running: false });
    const labels = { first: 'First', stress: 'Stress', quadrature: 'Quadrature' };
    const update = (event) => { snapshot.value = event.detail; };
    onMounted(() => window.addEventListener('ryugu-snapshot', update));
    onBeforeUnmount(() => window.removeEventListener('ryugu-snapshot', update));
    const workloads = ['first', 'stress'];
    return { planning, labels, workloads };
  },
  template: `
    <section class="mt-2 grid gap-2" aria-live="polite" aria-label="Calculation progress">
      <div v-for="kind in workloads" :key="kind" class="rounded-md border border-cyan-200/15 bg-black/20 px-2 py-1.5">
        <div class="flex items-center justify-between gap-2 font-mono text-[10px] text-slate-300">
          <span>{{ labels[kind] }} accuracy</span>
          <span>{{ kind === planning.workload ? Math.round(planning.accuracy) + '%' : 'waiting' }}</span>
        </div>
        <div class="mt-1 h-1.5 overflow-hidden rounded bg-cyan-950/80" role="progressbar" :aria-label="labels[kind] + ' calculation progress'" :aria-valuenow="kind === planning.workload ? Math.round(planning.progress) : 0" aria-valuemin="0" aria-valuemax="100">
          <div class="h-full rounded bg-cyan-300 transition-[width] duration-200" :style="{ width: (kind === planning.workload ? planning.progress : 0) + '%' }"></div>
        </div>
        <div class="mt-1 font-mono text-[9px] text-slate-500">{{ kind === planning.workload && planning.running ? Math.round(planning.progress) + '% complete' : 'Ready' }}</div>
      </div>
    </section>
  `,
});

app.mount('#planning-progress');

createApp({
  setup() {
    const snapshot = ref(null);
    const planning = computed(() => snapshot.value?.planning ?? { progress: 0, accuracy: 0, workload: 'quadrature', running: false });
    const update = (event) => { snapshot.value = event.detail; };
    onMounted(() => window.addEventListener('ryugu-snapshot', update));
    onBeforeUnmount(() => window.removeEventListener('ryugu-snapshot', update));
    return { planning };
  },
  template: `<div v-if="planning.workload === 'quadrature'" class="mt-1 min-w-64"><div class="flex justify-between font-mono text-[10px] text-slate-300"><span>Quadrature verification</span><span>{{ Math.round(planning.accuracy) }}% · {{ Math.round(planning.progress) }}% complete</span></div><div class="mt-1 h-1.5 overflow-hidden rounded bg-cyan-950/80"><div class="h-full rounded bg-cyan-300 transition-[width] duration-200" :style="{ width: planning.progress + '%' }"></div></div></div>`,
}).mount('#quadrature-progress');
