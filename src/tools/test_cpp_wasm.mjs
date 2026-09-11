import { instantiateBackend, createFieldBackend } from "../backend/host/cpp_backend.mjs";

function equal(actual, expected, message) {
  if (actual !== expected) throw new Error(message ?? `${actual} !== ${expected}`);
}
function ok(condition, message) {
  if (!condition) throw new Error(message);
}

const backend = await instantiateBackend(
  await Bun.file(new URL("../backend/zig/zig-out/bin/ryugu_backend.wasm", import.meta.url)).bytes(),
);
const { api } = backend;
const put = (data) => backend.put(new Float64Array(data));
const position = put([10, 0, 0]);
const acceleration = put([0, 0, 0]);
const potential = put([0]);
const G = 6.67430e-11;
const close = (actual, expected, tolerance = 1e-10) => {
  ok(Math.abs(actual - expected) <= tolerance * Math.abs(expected) + 1e-24,
    `${actual} != ${expected}`);
};
try {
  equal(api.ryugu_basilisk_protocol_version(), 1);
  equal(api.ryugu_direct_sum_eval(put([0, 0, 0]), put([2]), 1n,
    position, acceleration, potential), 0);
  close(backend.doubles(acceleration, 3)[0], -2 * G / 100);
  close(backend.doubles(potential, 1)[0], 2 * G / 10);

  // A radial ray along +x has an independently integrable potential outside it.
  const cells = put([1, 0, 0, 0.25, 1, 2, 1000, 0]);
  equal(api.ryugu_radial_boost_eval(cells, 1n, position,
    acceleration, potential, 1e-16, 1e-10), 0);
  const primitive = (r) => -r * r / 2 - 10 * r - 100 * Math.log(10 - r);
  close(backend.doubles(potential, 1)[0], G * 250 * (primitive(2) - primitive(1)), 1e-10);

  // Regular tetrahedron, outward triangles, far-field monopole limit.
  const vertices = put([1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1]);
  const facets = backend.put(new Uint32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]));
  const far = put([100, 0, 0]);
  equal(api.ryugu_werner_eval(vertices, 4n, facets, 4n, 1,
    far, acceleration, potential), 0);
  close(backend.doubles(acceleration, 3)[0], -1e-4, 1e-6);
  close(backend.doubles(potential, 1)[0], 0.01, 1e-6);
  console.log("PASS: C++ WASM direct sum, Boost radial integral, Basilisk polyhedral gravity");
  if (Bun.argv.includes("--fmm")) {
    const xyz = [], masses = [], targets = [];
    for (let i = 0; i < 128; i++) {
      xyz.push(Math.sin(i * 1.1), Math.cos(i * 2.3), Math.sin(i * 3.7));
      masses.push(1 + i / 128);
    }
    for (let i = 0; i < 16; i++) targets.push(3 + Math.sin(i), Math.cos(i), Math.sin(i * 2));
    const sourcePtr = put(xyz), massPtr = put(masses);
    const potentials = put(new Array(16).fill(0)), gradients = put(new Array(48).fill(0));
    equal(api.ryugu_exafmm_eval(sourcePtr, massPtr, 128n, put(targets), 16n,
      4, 8, potentials, gradients), 0);
    const values = backend.doubles(potentials, 16), forces = backend.doubles(gradients, 48);
    let maxRelativeForceError = 0;
    for (let i = 0; i < 16; i++) {
      equal(api.ryugu_direct_sum_eval(sourcePtr, massPtr, 128n,
        put(targets.slice(3 * i, 3 * i + 3)), acceleration, potential), 0);
      close(values[i], backend.doubles(potential, 1)[0], 0.005);
      const reference = backend.doubles(acceleration, 3);
      const error = Math.hypot(...reference.map((v, j) => v - forces[3 * i + j])) / Math.hypot(...reference);
      maxRelativeForceError = Math.max(maxRelativeForceError, error);
      ok(error < 0.01, `FMM relative acceleration error ${error}`);
    }
    console.log(`PASS: ExaFMM WASM vs direct sum, max relative acceleration error ${maxRelativeForceError}`);
  }
  if (Bun.argv.includes("--flups")) {
    const n = 16, count = n ** 3;
    const density = new Float64Array(count);
    const center = (8 * n + 8) * n + 8;
    density[center] = 1;
    const phi = put(new Array(count).fill(0));
    const field = put(new Array(3 * count).fill(0));
    equal(api.ryugu_flups_free_space_eval(backend.put(density), n, n, n,
      put([1, 1, 1]), put([n, n, n]), 1, phi, field), 0);
    const p = backend.doubles(phi, count), a = backend.doubles(field, 3 * count);
    ok(p.every(Number.isFinite) && a.every(Number.isFinite));
    console.log("FLUPS probe:", p[center + 4], a[3 * (center + 4)]);
    ok(a[3 * (center + 4)] < 0, "gravity must point towards the positive mass");
    close(Math.abs(p[center + 4]), 0.25, 0.1);
    console.log("PASS: FLUPS WASM free-space point-source potential and gravity direction");

    {
      // Live 32³ box must resolve the probe, not a ±4096 m desert of 256 m cells.
      const live = createFieldBackend(backend);
      const bodyRadius = 450;
      const mass = 4.5e11;
      const rho = mass / ((4 / 3) * Math.PI * bodyRadius ** 3);
      const dirs = 32;
      const solid = 4 * Math.PI / dirs;
      const cells = [];
      for (let i = 0; i < dirs; i++) {
        const z = 1 - 2 * (i + 0.5) / dirs;
        const rxy = Math.sqrt(Math.max(0, 1 - z * z));
        const angle = 2.399963229728653 * i;
        cells.push(rxy * Math.cos(angle), rxy * Math.sin(angle), z, solid, 0, bodyRadius, rho, 0);
      }
      live.configure(
        new Float64Array(cells),
        new Float64Array([1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1]),
        new Uint32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]),
        mass,
      );
      const ic = live.evaluate("fft", [-617, 0, -65]);
      const fmm = live.evaluate("fmm", [-617, 0, -65]);
      const mag = (g) => Math.hypot(g[0], g[1], g[2]);
      const relative = mag([ic[0] - fmm[0], ic[1] - fmm[1], ic[2] - fmm[2]]) / mag(fmm);
      ok(relative < 0.05, `live FLUPS vs FMM at the probe IC is ${relative}`);
      const far = live.evaluate("fft", [20000, 0, 0]);
      ok(far.every(Number.isFinite), "far FFT target must stay finite");
      console.log(`PASS: live FLUPS probe field vs FMM (relative ${relative})`);
    }
  }
} finally {
  backend.release();
}
