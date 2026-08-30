import { mkdir, readFile, writeFile } from 'node:fs/promises';

const pageUrl = 'https://tom-jim.github.io/RyuGu_WASM/';
const badgeUrl = 'https://hits.sh/tom-jim.github.io/RyuGu_WASM.svg?view=today-total&label=site%20visits&color=48e7e2&labelColor=071418';
const historyPath = 'data/site-traffic.json';
const chartPath = 'assets/site-traffic.svg';
const readmePath = 'README.md';
const checkOnly = process.argv.includes('--check');
const today = process.env.HITS_DATE ?? new Date().toISOString().slice(0, 10);

function parseBadge(svg) {
  const title = svg.match(/<title>([^<]+)<\/title>/i)?.[1] ?? '';
  const match = title.match(/:\s*([\d,]+)\s*\/\s*([\d,]+)/);
  if (!match) throw new Error(`Could not parse today/total values from HITS SVG title: ${title || '(missing)'}`);
  return {
    today: Number(match[1].replaceAll(',', '')),
    total: Number(match[2].replaceAll(',', '')),
  };
}

const compact = (value) => new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
const xml = (value) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

function renderChart(history) {
  const samples = history.slice(-90);
  const width = 920;
  const height = 300;
  const plot = { left: 70, right: 28, top: 62, bottom: 48 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const timestamps = samples.map((sample) => Date.parse(`${sample.date}T00:00:00Z`));
  const values = samples.map((sample) => sample.total);
  const xMin = Math.min(...timestamps);
  const xMax = Math.max(...timestamps);
  const yDataMin = Math.min(...values);
  const yDataMax = Math.max(...values);
  const ySpan = yDataMax - yDataMin;
  const yMin = samples.length === 1 ? 0 : Math.max(0, yDataMin - Math.max(ySpan * 0.15, 1));
  const yMax = samples.length === 1 ? Math.max(5, yDataMax * 1.25) : yDataMax + Math.max(ySpan * 0.15, 1);
  const x = (timestamp, index) => xMax === xMin ? plot.left + plotWidth / 2 : plot.left + (timestamp - xMin) / (xMax - xMin) * plotWidth;
  const y = (value) => plot.top + (yMax - value) / Math.max(yMax - yMin, 1) * plotHeight;
  const points = samples.map((sample, index) => `${x(timestamps[index], index).toFixed(1)},${y(sample.total).toFixed(1)}`);
  const areaPoints = [`${points[0]?.split(',')[0] ?? plot.left},${plot.top + plotHeight}`, ...points, `${points.at(-1)?.split(',')[0] ?? plot.left},${plot.top + plotHeight}`].join(' ');
  const yGrid = Array.from({ length: 5 }, (_, index) => {
    const value = yMin + (yMax - yMin) * index / 4;
    const py = y(value);
    return `<line x1="${plot.left}" y1="${py.toFixed(1)}" x2="${width - plot.right}" y2="${py.toFixed(1)}"/><text x="${plot.left - 12}" y="${(py + 4).toFixed(1)}" text-anchor="end">${xml(compact(Math.round(value)))}</text>`;
  }).join('');
  const tickIndexes = [...new Set(Array.from({ length: Math.min(5, samples.length) }, (_, index) => Math.round(index * (samples.length - 1) / Math.max(Math.min(5, samples.length) - 1, 1))))];
  const xGrid = tickIndexes.map((sampleIndex) => {
    const px = x(timestamps[sampleIndex], sampleIndex);
    return `<line x1="${px.toFixed(1)}" y1="${plot.top}" x2="${px.toFixed(1)}" y2="${plot.top + plotHeight}"/><text x="${px.toFixed(1)}" y="${height - 20}" text-anchor="middle">${xml(samples[sampleIndex].date.slice(5))}</text>`;
  }).join('');
  const latest = samples.at(-1);
  const latestX = x(timestamps.at(-1), samples.length - 1);
  const latestY = y(latest.total);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
<title id="title">RyuGu WASM live site traffic</title><desc id="desc">Daily HITS snapshots for ${xml(pageUrl)}. Latest total ${latest.total}.</desc>
<defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#48e7e2" stop-opacity=".28"/><stop offset="1" stop-color="#48e7e2" stop-opacity="0"/></linearGradient><filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#86a3a8;font-size:12px}.grid line{stroke:#5fc7cc;stroke-opacity:.13}.trace{fill:none;stroke:#48e7e2;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1.4s ease-out forwards}.pulse{animation:pulse 1.8s ease-in-out infinite}@keyframes draw{to{stroke-dashoffset:0}}@keyframes pulse{50%{r:7;opacity:.55}}</style>
<rect width="100%" height="100%" rx="14" fill="#050f13"/><rect x="1" y="1" width="918" height="298" rx="13" fill="none" stroke="#48e7e2" stroke-opacity=".24"/>
<text x="28" y="31" fill="#d9eef0" font-size="17" font-weight="700">RyuGu WASM · live site traffic</text><text x="28" y="50">hits.sh daily snapshot · today ${latest.today} · total ${latest.total}</text>
<g class="grid">${yGrid}${xGrid}</g><line x1="${plot.left}" y1="${plot.top}" x2="${plot.left}" y2="${plot.top + plotHeight}" stroke="#8be9e6" stroke-opacity=".42"/><line x1="${plot.left}" y1="${plot.top + plotHeight}" x2="${width - plot.right}" y2="${plot.top + plotHeight}" stroke="#8be9e6" stroke-opacity=".42"/>
<polygon points="${areaPoints}" fill="url(#area)"/><polyline class="trace" pathLength="1" points="${points.join(' ')}"/><circle class="pulse" cx="${latestX.toFixed(1)}" cy="${latestY.toFixed(1)}" r="4.5" fill="#43df81" filter="url(#glow)"/>
<text x="${width - 28}" y="31" text-anchor="end" fill="#48e7e2">90-day rolling history</text><text x="${width / 2}" y="${height - 4}" text-anchor="middle">UTC date</text><text x="16" y="${plot.top + plotHeight / 2}" text-anchor="middle" transform="rotate(-90 16 ${plot.top + plotHeight / 2})">total visits</text></svg>`;
}

async function loadBadge() {
  if (process.env.HITS_BADGE_SVG) return process.env.HITS_BADGE_SVG;
  const response = await fetch(badgeUrl, { headers: { Accept: 'image/svg+xml', 'User-Agent': 'RyuGu-WASM-traffic-chart' } });
  if (!response.ok) throw new Error(`HITS returned HTTP ${response.status}`);
  return response.text();
}

const badge = parseBadge(await loadBadge());
let history = [];
try { history = JSON.parse(await readFile(historyPath, 'utf8')); } catch (error) { if (error.code !== 'ENOENT') throw error; }
const previous = history.at(-1);
const sample = { date: today, today: badge.today, total: Math.max(badge.total, previous?.total ?? 0) };
if (previous?.date === today) history[history.length - 1] = sample; else history.push(sample);
history = history.slice(-180);
const chart = renderChart(history);
if (checkOnly) {
  if (!chart.includes('<polyline') || !chart.includes('total visits')) throw new Error('Generated chart is incomplete');
  console.log(`Traffic SVG check passed: ${sample.today} today / ${sample.total} total`);
} else {
  await mkdir('data', { recursive: true });
  await mkdir('assets', { recursive: true });
  await writeFile(historyPath, `${JSON.stringify(history, null, 2)}\n`);
  await writeFile(chartPath, `${chart}\n`);
  const readme = await readFile(readmePath, 'utf8');
  await writeFile(readmePath, readme.replace(/site-traffic\.svg\?v=[0-9-]+/, `site-traffic.svg?v=${today}`));
  console.log(`Updated ${chartPath}: ${sample.today} today / ${sample.total} total`);
}
