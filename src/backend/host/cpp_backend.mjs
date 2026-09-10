// Minimal browser host for the WASI imports used by the C++ numerical module.
// Filesystem operations return WASI errno values, never fabricated success.
export async function instantiateBackend(bytes) {
  let instance;
  const view = () => new DataView(instance.exports.memory.buffer);
  const wasi = {
    args_sizes_get(argc, size) {
      view().setUint32(argc, 0, true);
      view().setUint32(size, 0, true);
      return 0;
    },
    args_get() { return 0; },
    environ_sizes_get(count, size) {
      view().setUint32(count, 0, true);
      view().setUint32(size, 0, true);
      return 0;
    },
    environ_get() { return 0; },
    clock_time_get(clock, precision, output) {
      if (clock !== 0 && clock !== 1) return 28;
      const milliseconds = clock === 0 ? Date.now() : performance.now();
      view().setBigUint64(output, BigInt(Math.floor(milliseconds * 1e6)), true);
      return 0;
    },
    fd_fdstat_get(fd, output) {
      if (fd !== 1 && fd !== 2) return 8;
      new Uint8Array(instance.exports.memory.buffer, output, 24).fill(0);
      view().setUint8(output, 2);
      view().setBigUint64(output + 8, 64n, true);
      return 0;
    },
    fd_read() { return 8; },
    fd_fdstat_set_flags() { return 58; },
    fd_prestat_get() { return 8; },
    fd_prestat_dir_name() { return 8; },
    path_create_directory() { return 76; },
    path_filestat_get() { return 76; },
    path_open() { return 76; },
    fd_close() { return 8; },
    fd_seek() { return 70; },
    fd_write(fd, iovs, count, written) {
      if (fd !== 1 && fd !== 2) return 8;
      let size = 0;
      const decoder = new TextDecoder();
      for (let i = 0; i < count; i++) {
        const pointer = view().getUint32(iovs + i * 8, true);
        const length = view().getUint32(iovs + i * 8 + 4, true);
        const message = decoder.decode(new Uint8Array(instance.exports.memory.buffer, pointer, length));
        (fd === 2 ? console.error : console.log)(message);
        size += length;
      }
      view().setUint32(written, size, true);
      return 0;
    },
    proc_exit(code) { throw new Error(`C++ backend exited with code ${code}`); },
  };
  ({ instance } = await WebAssembly.instantiate(bytes, {
    wasi_snapshot_preview1: wasi,
    env: { ryugu_rust_tick(time, output) {
      if (!globalThis.ryuguRustBackend) throw new Error("Rust backend is not loaded");
      const state = globalThis.ryuguRustBackend.tick(time);
      new Float64Array(instance.exports.memory.buffer, output, 6).set(state);
    } },
  }));
  const api = instance.exports;
  api.__wasm_call_ctors();
  const allocations = new Set();
  return {
    api,
    put(array) {
      const pointer = api.ryugu_alloc(array.byteLength);
      if (!pointer) throw new Error("C++ backend allocation failed");
      allocations.add(pointer);
      new Uint8Array(api.memory.buffer, pointer, array.byteLength).set(
        new Uint8Array(array.buffer, array.byteOffset, array.byteLength),
      );
      return pointer;
    },
    doubles(pointer, count) {
      return Array.from(new Float64Array(api.memory.buffer, pointer, count));
    },
    free(pointer) {
      if (allocations.delete(pointer)) api.ryugu_free(pointer);
    },
    release() {
      for (const pointer of allocations) api.ryugu_free(pointer);
      allocations.clear();
    },
  };
}

// All field results use body-frame SI coordinates and positive potential U.
// FLUPS solves for negative Phi, so only its potential is sign-converted.
export function createFieldBackend(backend) {
  const { api } = backend;
  const G = 6.67430e-11;
  let geometry;
  let fft;
  const own = [];
  function put(array) {
    const pointer = backend.put(array);
    own.push(pointer);
    return pointer;
  }
  function checked(status, method) {
    if (status !== 0) throw new Error(`${method} failed with C ABI status ${status}`);
  }
  function temporary(callback) {
    const pointers = [];
    const alloc = (array) => {
      const pointer = backend.put(array);
      pointers.push(pointer);
      return pointer;
    };
    try { return callback(alloc); }
    finally { for (const pointer of pointers) backend.free(pointer); }
  }
  function buildWernerPreviewMesh(vertices) {
    // Basilisk preprocesses every facet before its first field evaluation.
    // A 200k-facet mesh blocks the browser event loop, so live propagation
    // uses a closed icosahedral proxy at the measured body scale. The C++
    // Werner implementation remains the evaluator; high-resolution work uses
    // the source-based C++ batch routes rather than this live preview path.
    const phi = (1 + Math.sqrt(5)) / 2;
    const unit = [
      -1, phi, 0, 1, phi, 0, -1, -phi, 0, 1, -phi, 0,
      0, -1, phi, 0, 1, phi, 0, -1, -phi, 0, 1, -phi,
      phi, 0, -1, phi, 0, 1, -phi, 0, -1, -phi, 0, 1,
    ];
    const faces = new Uint32Array([
      0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
      1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
      3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
      4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
    ]);
    const center = [0, 0, 0];
    for (let i = 0; i < vertices.length; i += 3) {
      for (let axis = 0; axis < 3; axis++) center[axis] += vertices[i + axis];
    }
    for (let axis = 0; axis < 3; axis++) center[axis] /= vertices.length / 3;
    let radius = 0;
    for (let i = 0; i < vertices.length; i += 3) {
      radius += Math.hypot(vertices[i] - center[0], vertices[i + 1] - center[1], vertices[i + 2] - center[2]);
    }
    radius = Math.max(radius / (vertices.length / 3), 1);
    const scale = radius / Math.hypot(1, phi);
    const points = new Float64Array(unit.length);
    for (let i = 0; i < unit.length; i += 3) {
      points[i] = center[0] + unit[i] * scale;
      points[i + 1] = center[1] + unit[i + 1] * scale;
      points[i + 2] = center[2] + unit[i + 2] * scale;
    }
    return { vertices: points, facets: faces };
  }
  function configure(cells, vertices, facets, mass) {
    if (cells.length === 0 || cells.length % 8 || vertices.length === 0 || vertices.length % 3 || facets.length === 0 || facets.length % 3
        || !Number.isFinite(mass) || mass <= 0) throw new Error("Invalid gravity geometry");
    if (!cells.every(Number.isFinite) || !vertices.every(Number.isFinite)
        || !facets.every((i) => Number.isInteger(i) && i >= 0 && i < vertices.length / 3)) {
      throw new Error("Invalid gravity geometry values");
    }
    const werner = buildWernerPreviewMesh(vertices);
    const nodes = [-0.9602898564975363, -0.7966664774136267, -0.525532409916329,
      -0.1834346424956498, 0.1834346424956498, 0.525532409916329,
      0.7966664774136267, 0.9602898564975363];
    const weights = [0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
      0.362683783378362, 0.362683783378362, 0.3137066458778873,
      0.2223810344533745, 0.1012285362903763];
    const xyz = new Float64Array(cells.length * 3);
    const masses = new Float64Array(cells.length);
    for (let cell = 0; cell < cells.length / 8; cell++) {
      const offset = cell * 8;
      const half = (cells[offset + 5] - cells[offset + 4]) / 2;
      const center = (cells[offset + 5] + cells[offset + 4]) / 2;
      for (let node = 0; node < 8; node++) {
        const i = cell * 8 + node;
        const radius = center + half * nodes[node];
        for (let axis = 0; axis < 3; axis++) xyz[3 * i + axis] = cells[offset + axis] * radius;
        masses[i] = weights[node] * half * radius * radius * cells[offset + 3] * cells[offset + 6];
      }
    }
    api.ryugu_werner_reset_cache();
    for (const pointer of own.splice(0)) backend.free(pointer);
    fft = undefined;
    geometry = undefined;
    geometry = { cells: put(new Float64Array(cells)), cellCount: BigInt(cells.length / 8),
      wernerVertices: put(werner.vertices), vertexCount: BigInt(werner.vertices.length / 3),
      wernerFacets: put(werner.facets), facetCount: BigInt(werner.facets.length / 3),
      xyz: put(xyz), masses: put(masses), sourceCount: BigInt(masses.length),
      xyzValues: xyz, massValues: masses, mu: G * mass };
  }
  function buildFft() {
    // A 32 cubed live grid keeps the complete FLUPS free-space solve below a
    // browser frame-budget scale. Scientific comparison jobs choose their own
    // source and target batches through the C++ API.
    const n = 32;
    const half = 4096;
    const spacing = 2 * half / n;
    const density = new Float64Array(n ** 3);
    for (let i = 0; i < geometry.massValues.length; i++) {
      const coordinate = [0, 1, 2].map((axis) => (geometry.xyzValues[3 * i + axis] + half) / spacing - 0.5);
      const base = coordinate.map(Math.floor);
      if (base.some((v) => v < 0 || v + 1 >= n)) throw new Error("Mass source outside FLUPS grid");
      for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
        const weight = [x, y, z].reduce((value, bit, axis) =>
          value * (bit ? coordinate[axis] - base[axis] : 1 - coordinate[axis] + base[axis]), 1);
        density[((base[2] + z) * n + base[1] + y) * n + base[0] + x] +=
          weight * geometry.massValues[i] / spacing ** 3;
      }
    }
    fft = temporary((alloc) => {
      const potential = alloc(new Float64Array(n ** 3));
      const acceleration = alloc(new Float64Array(3 * n ** 3));
      checked(api.ryugu_flups_free_space_eval(alloc(density), n, n, n,
        alloc(new Float64Array([spacing, spacing, spacing])),
        alloc(new Float64Array([2 * half, 2 * half, 2 * half])), G, potential, acceleration), "FLUPS");
      return { n, half, spacing, potential: backend.doubles(potential, n ** 3),
        acceleration: backend.doubles(acceleration, 3 * n ** 3) };
    });
  }
  function evaluate(method, position) {
    if (!geometry) throw new Error("C++ gravity geometry is not configured");
    if (position.length !== 3 || !position.every(Number.isFinite)) throw new Error("Invalid gravity target");
    if (method === "fft") {
      if (!fft) buildFft();
      const coordinate = position.map((v) => (v + fft.half) / fft.spacing - 0.5);
      const base = coordinate.map(Math.floor);
      if (base.some((v) => v < 0 || v + 1 >= fft.n)) throw new Error("Gravity target outside FLUPS grid");
      const value = [0, 0, 0, 0];
      for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
        const weight = [x, y, z].reduce((v, bit, axis) =>
          v * (bit ? coordinate[axis] - base[axis] : 1 - coordinate[axis] + base[axis]), 1);
        const i = ((base[2] + z) * fft.n + base[1] + y) * fft.n + base[0] + x;
        for (let axis = 0; axis < 3; axis++) value[axis] += weight * fft.acceleration[3 * i + axis];
        value[3] -= weight * fft.potential[i];
      }
      return value;
    }
    return temporary((alloc) => {
      const target = alloc(new Float64Array(position));
      const acceleration = alloc(new Float64Array(3));
      const potential = alloc(new Float64Array(1));
      let status;
      if (method === "radial") status = api.ryugu_radial_boost_eval(geometry.cells, geometry.cellCount,
        target, acceleration, potential, 1e-12, 1e-8);
      else if (method === "direct") status = api.ryugu_direct_sum_eval(geometry.xyz, geometry.masses,
        geometry.sourceCount, target, acceleration, potential);
      else if (method === "werner") status = api.ryugu_werner_eval(geometry.wernerVertices, geometry.vertexCount,
        geometry.wernerFacets, geometry.facetCount, geometry.mu, target, acceleration, potential);
      else if (method === "fmm") status = api.ryugu_exafmm_eval(geometry.xyz, geometry.masses,
        geometry.sourceCount, target, 1n, 4, 64, potential, acceleration);
      else throw new Error(`Unknown C++ gravity method: ${method}`);
      checked(status, method);
      const result = [...backend.doubles(acceleration, 3), ...backend.doubles(potential, 1)];
      if (!result.every(Number.isFinite)) throw new Error(`${method} returned a non-finite field`);
      return result;
    });
  }
  function evaluateSources(method, xyz, masses, targets) {
    if (xyz.length !== masses.length * 3 || targets.length % 3
        || !xyz.every(Number.isFinite) || !masses.every(Number.isFinite)
        || !targets.every(Number.isFinite)) throw new Error("Invalid source batch");
    if (masses.length === 0) return new Float64Array(targets.length / 3 * 4);
    return temporary((alloc) => {
      const source = alloc(new Float64Array(xyz));
      const mass = alloc(new Float64Array(masses));
      const count = targets.length / 3;
      if (method === "fmm") {
        const potential = alloc(new Float64Array(count));
        const field = alloc(new Float64Array(count * 3));
        checked(api.ryugu_exafmm_eval(source, mass, BigInt(masses.length),
          alloc(new Float64Array(targets)), BigInt(count), 4, 64, potential, field), method);
        // Create views only after the C++ call: memory.grow can detach earlier
        // views. Copy into JS-owned storage before temporary allocations free.
        const a = new Float64Array(api.memory.buffer, field, count * 3);
        const u = new Float64Array(api.memory.buffer, potential, count);
        const result = new Float64Array(count * 4);
        for (let i = 0; i < count; i++) {
          result[4 * i] = a[3 * i];
          result[4 * i + 1] = a[3 * i + 1];
          result[4 * i + 2] = a[3 * i + 2];
          result[4 * i + 3] = u[i];
        }
        return result;
      }
      const previousGeometry = geometry, previousFft = fft;
      try {
        geometry = { xyz: source, masses: mass, sourceCount: BigInt(masses.length),
          xyzValues: xyz, massValues: masses };
        fft = undefined;
        const result = new Float64Array(count * 4);
        for (let i = 0; i < count; i++) result.set(evaluate(method, Array.from(targets.slice(i * 3, i * 3 + 3))), i * 4);
        return result;
      } finally { geometry = previousGeometry; fft = previousFft; }
    });
  }
  return { configure, evaluate, evaluateSources };
}
