// Pure data-gating regressions; no DOM, browser, GPU, or WASM required.
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const context = { window: {} };
const source = readFileSync(`${__dirname}/../src/html/ui.js`, 'utf8');
runInNewContext(source.slice(0, source.indexOf('\n(() => {')), context);
const statistics = context.window.ryuguCurveStatistics;
const samples = () => Array.from({ length: 7 }, (_, i) => ({
  sources: 32000, densityModels: 512, targets: 241, repeat: i + 1,
  times: Array(6).fill(10 + i), eligible: Array(6).fill(true),
  gravityErrors: Array(6).fill(1e-4), gradientErrors: Array(6).fill(1e-3),
}));

test('median and min/max require seven distinct qualified repetitions', () => {
  const rows = samples();
  const method = statistics(rows, 512, 241)[0].methods[0];
  assert.equal(method.value, 13);
  assert.equal(method.low, 10);
  assert.equal(method.high, 16);
  assert.equal(method.status, 'PASS');
  assert.equal(statistics(rows.slice(0, 6), 512, 241)[0].methods[0].value, null);
  assert.equal(statistics([...rows.slice(0, 6), rows[0]], 512, 241)[0].methods[0].value, null);
});

test('one failure suppresses the whole cell, not just the failed repetition', () => {
  const rows = samples();
  rows[0].eligible[0] = false;
  const methods = statistics(rows, 512, 241)[0].methods;
  assert.equal(methods[0].value, null);
  assert.equal(methods[0].status, 'FAIL');
  assert.equal(methods[0].rejected, 1);
  assert.equal(methods[1].value, 13);
});

test('missing eligibility and nonfinite/null/nonpositive times fail closed', () => {
  for (const time of [null, NaN, Infinity, 0, -1]) {
    const rows = samples();
    rows[0].times[0] = time;
    assert.equal(statistics(rows, 512, 241)[0].methods[0].value, null);
  }
  const rows = samples();
  delete rows[0].eligible;
  assert.equal(statistics(rows, 512, 241)[0].methods[0].value, null);
});

test('source, density and target dimensions never share a median', () => {
  const rows = samples();
  const other = rows.map((row) => ({ ...row, densityModels: 1, times: Array(6).fill(100) }));
  const targets = rows.map((row) => ({ ...row, targets: 8, times: Array(6).fill(200) }));
  const sources = rows.map((row) => ({ ...row, sources: 64000, times: Array(6).fill(300) }));
  const all = [...rows, ...other, ...targets, ...sources];
  assert.equal(statistics(all, 512, 241)[0].methods[0].value, 13);
  assert.equal(statistics(all, 512, 241)[1].methods[0].value, 300);
  assert.equal(statistics(all, 1, 241)[0].methods[0].value, 100);
  assert.equal(statistics(all, 512, 8)[0].methods[0].value, 200);
});

test('screening pass retains strict failure and exposes gate reasons', () => {
  const rows = samples().map((row) => ({ ...row,
    strictEligible: Array(6).fill(false),
    failureReasons: Array.from({ length: 6 }, () => []),
    accuracyProfile: 'screening',
  }));
  const screening = statistics(rows, 512, 241)[0].methods[0];
  assert.equal(screening.status, 'PASS');
  assert.equal(screening.strictPassed, 0);
  rows[0].eligible[0] = false;
  rows[0].failureReasons[0] = ['gradient p99/max'];
  const failed = statistics(rows, 512, 241)[0].methods[0];
  assert.equal(failed.value, null);
  assert.equal(failed.reasons[0], 'gradient p99/max');
});


test('100 percent requires explicit final completion, never rounding', () => {
  const progress = context.window.ryuguPlanningProgress;
  for (const value of [99.49, 99.99, 100, 110]) {
    assert.ok(progress({ progress: value, running: true }).progress < 100);
    assert.ok(progress({ progress: value, running: false }).progress < 100);
  }
  assert.equal(progress({ progress: 100, completed: true }).progress, 100);
  assert.equal(progress({ progress: 0, runId: 2, running: true }).progress, 0);
  assert.equal(progress({ progress: NaN }).progress, 0);
});


test('source cells publish incrementally, retaining the previous point while the next accumulates', () => {
  const axis = [32000, 64000, 128000, 256000];
  const first = samples();
  const next = samples().map((row) => ({ ...row, sources: 64000, times: Array(6).fill(30) }));
  const plot = (rows) => context.window.ryuguCurvePlotData(statistics(rows, 512, 241), axis);
  // A whole sweep is not required: the first seven repetitions produce a point.
  let data = plot(first);
  assert.equal(data[0].points[0][1], 13);
  assert.equal(data[0].points[1][1], null);
  data = plot([...first, ...next.slice(0, 6)]);
  assert.equal(data[0].points[0][1], 13);
  assert.equal(data[0].points[1][1], null);
  // The seventh result adds the neighboring point; both share the fixed axis.
  data = plot([...first, ...next]);
  assert.equal(data[0].points[0][1], 13);
  assert.equal(data[0].points[1][1], 30);
  assert.equal(data[0].points[2][1], null);
  assert.equal(data[0].ranges.length, 2);
});

test('a failed intervening source remains a line break rather than a fabricated timing', () => {
  const first = samples();
  const failed = samples().map((row) => ({ ...row, sources: 64000, eligible: Array(6).fill(false) }));
  const last = samples().map((row) => ({ ...row, sources: 128000 }));
  const series = context.window.ryuguCurvePlotData(statistics([...first, ...failed, ...last], 512, 241),
    [32000, 64000, 128000])[0];
  assert.equal(series.points[0][1], 13);
  assert.equal(series.points[1][1], null);
  assert.equal(series.points[2][1], 13);
  assert.equal(series.ranges.length, 2);
});


test('timestamp views never replace a missing GPU duration with pipeline wall time', () => {
  const rows = samples().map((row) => ({ ...row, kernelTimes: Array(6).fill(0.25) }));
  let method = statistics(rows, 512, 241, 7, 'kernelTimes')[0].methods[0];
  assert.equal(method.value, 0.25);
  rows[0].kernelTimes[0] = null;
  method = statistics(rows, 512, 241, 7, 'kernelTimes')[0].methods[0];
  assert.equal(method.status, 'PASS'); // numerical qualification is independent
  assert.equal(method.value, null);
  assert.equal(method.timingAvailable, false);
  assert.equal(statistics(rows, 512, 241)[0].methods[0].value, 13);
});

test('quantized zero timestamps are not fabricated into positive log-chart points', () => {
  const rows = samples().map((row) => ({ ...row, kernelTimes: Array(6).fill(0) }));
  const groups = statistics(rows, 512, 241, 7, 'kernelTimes');
  assert.equal(groups[0].methods[0].status, 'PASS');
  assert.equal(groups[0].methods[0].belowResolution, true);
  const plot = context.window.ryuguCurvePlotData(groups, [32000]);
  assert.equal(plot[0].points[0][1], null);
  assert.equal(plot[0].ranges.length, 0);
});
