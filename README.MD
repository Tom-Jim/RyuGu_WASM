# Ryugu WASM — Asteroid Gravitational Dynamics Simulator

[![Live Demo](https://img.shields.io/badge/Live_Demo-WebGPU-success?style=for-the-badge)](https://tom-jim.github.io/RyuGu_WASM/)
![Bevy](https://img.shields.io/badge/Bevy-0.19.0-purple.svg)
![Rust](https://img.shields.io/badge/Rust-Edition_2024-orange.svg)
![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)

Real-time orbital simulator of asteroid (162173) Ryugu, compiled to WASM via `wasm-pack`. A Cassini-class probe orbits Ryugu under gravity computed each frame by a WebGPU compute shader running a Gaver-Stehfest Numerical Inverse Laplace Transform (NILT) against a precomputed Struve-Neumann function lookup table. The Bevy 0.19 engine handles rendering, ECS scheduling, and the render-world/main-world GPU readback pattern.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f6fe44af-676f-4436-87c1-21b7f67a8e3a" width="100%" alt="Ryugu WebGPU Simulation"/>
</p>

| Orbital Trajectory | Precession |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/cdca307b-7b0b-4eca-a28b-96cc822f8069" width="100%"/> | <img src="https://github.com/user-attachments/assets/3c152cf3-071a-4807-8fdd-6c981ec3f4d4" width="100%"/> |
| **Probe view** | **Surface normals** |
| <img src="https://github.com/user-attachments/assets/977a59b3-a0b1-44b4-8426-2237dbcac1fe" width="100%"/> | <img src="https://github.com/user-attachments/assets/d914c5f5-9a68-44ee-8ed6-0b75625bd312" width="100%"/> |

---

## Quick Start

```sh
# Primes Bun's module cache; no runtime JS dependencies
bun install

# Dev build (debug, opt-level 1) + dev server at http://localhost:3000
bun run dev

# Production build (release: opt-level "z", LTO) -> pkg/
bun run build

# Start dev server only (requires pkg/ to already exist)
bun run serve

# Lint / format (native host only -- not the wasm32 target)
cargo clippy
cargo fmt
```

> `cargo run` / `cargo build` without `--target wasm32-unknown-unknown` are only useful
> for syntax checking. The runnable artifact is the WASM package produced by `wasm-pack`.

**Browser:** WebGPU required. Chrome/Chromium 113+, Edge 113+, Safari 18+.
Firefox has WebGPU behind a flag but is not supported in production.

The dev server (`server.ts`) sets `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` on every response, required for
`SharedArrayBuffer` / Bevy WASM thread support.

**Before running,** `lut_struve.bin` must exist in the project root.
If the file is missing, `build_gravity_voxels_system` calls
`std::fs::read("lut_struve.bin").unwrap()` and panics immediately.

Regenerate with:

```sh
uv run scripts/gen_lut.py
```

---

## Directory Structure and File Responsibilities

```
Ryugu_wasm/
├── src/
│   ├── lib.rs               Bevy App entry point (#[wasm_bindgen(start)] on pub fn main()).
│   │                        Configures BROWSER_WEBGPU backend. Raises limits:
│   │                          max_storage_buffers_per_shader_stage = 8
│   │                          max_compute_workgroups_per_dimension = 65535
│   │                        Registers all plugins. Chains all 14 Update systems with
│   │                        .chain() for strict sequential ordering every frame.
│   ├── components.rs        All global physical constants (G, RYUGU_MASS, TIME_SCALE,
│   │                        rotation period, spin axis, epsilons, orbit parameters).
│   │                        All ECS Component and Resource type definitions.
│   ├── topology.rs          Builds a CSR (Compressed Sparse Row) adjacency list from
│   │                        the welded mesh. For each vertex, stores sorted neighbor
│   │                        indices in a flat array with an offsets vec for O(1) slice.
│   ├── welding.rs           Vertex deduplication via spatial hashing with configurable
│   │                        epsilon (1e-4 is used in build_topology_system). Quantizes
│   │                        each vertex to grid cells; merges vertices in the same cell.
│   └── systems/
│       ├── scale.rs         normalize_model_scale_system: traverses entity+children
│       │                    AABB tree, finds max dimension, applies uniform scale to
│       │                    match TargetSize. Marks entity ScaleNormalized (runs once).
│       │                    build_topology_system: welds mesh (eps=1e-4), builds CSR
│       │                    topology, then solves DensityC via 4-point Gaussian
│       │                    quadrature over all signed tetrahedra anchored at origin.
│       ├── gravity_pipeline.rs  StruveLutSource resource defined here (not components.rs).
│       │                    build_gravity_voxels_system: loads lut_struve.bin (panics
│       │                    if missing). Assigns per-voxel mass proportional to
│       │                    1/(||p*scale|| + DENSITY_EPSILON), normalized to conserve
│       │                    total RYUGU_MASS. Packs voxels as [x,y,z,mass] f32x4.
│       │                    extract_grav_input_system: transforms Cassini position
│       │                    from world frame into Ryugu body-fixed frame via inv_rot.
│       │                    Packs 80-byte uniform buffer: [probe_xyz, G, voxel_count,
│       │                    M=6, pad, pad, V[0..12] as 3x vec4<f32>].
│       │                    dispatch_grav_system: dispatches n_workgroups=ceil(N/64)
│       │                    workgroups. Initiates async staging buffer map_async.
│       │                    poll_gravity_readback: folds all workgroup partial vec4
│       │                    results into GravityAcceleration resource.
│       ├── physics.rs       physics_system: semi-implicit (symplectic) Euler.
│       │                    Rotates GPU result from Ryugu body frame to world frame.
│       │                    Falls back to softened Newtonian point-mass until GPU
│       │                    pipeline warms up (grav_voxels is None or result is zero).
│       │                    ryugu_rotation_system: incremental quaternion rotation
│       │                    applied each frame using delta_angle = omega * dt.
│       ├── render.rs        Scene setup, lighting, camera, Cassini initial conditions.
│       │                    Orbit history ring buffer (capacity ORBIT_HISTORY_LEN=27500).
│       │                    Gizmo rendering for trajectory, normals, section plane.
│       ├── compute_pipeline.rs  NormalsComputePlugin: one-shot GPU dispatch (DispatchState
│       │                    goes Done after first fire). Writes result to staging buffer,
│       │                    maps async, pushes to NormalsReadbackChannel.
│       ├── compute_normals.rs   CPU normal computation. Accumulates area-weighted face
│       │                    normals per vertex synchronously. Inserts AsteroidNormalsGpuData
│       │                    on the same frame topology becomes available. Because this
│       │                    runs before the GPU readback arrives, it short-circuits the
│       │                    GPU normals path in practice (poll_normals_readback returns
│       │                    early if AsteroidNormalsGpuData already exists).
│       └── ui.rs            FPS display, keyboard toggle handlers for normals/section.
├── assets/shaders/
│   ├── gravity.wgsl         Gaver-Stehfest NILT loop over 12 terms (k=1..12).
│   │                        Per-thread voxel contribution written to workgroup shared
│   │                        memory (shared_acc[64]). Parallel stride reduction:
│   │                        stride starts at 32, halved each iteration via stride>>=1u.
│   │                        Each workgroup writes one vec4 partial sum to output_acc.
│   └── normals.wgsl         Per-vertex normal from CSR neighbor ring. Iterates
│                            consecutive neighbor pairs, accumulates cross(a-p, b-p).
│                            Normalizes; defaults to (0,1,0) for degenerate vertices.
└── scripts/
    └── gen_lut.py           Computes S_v(z) = (pi/2)[H_v(z) - Y_v(z)] for v=0,1
                             over z in [0,50] at N=4096 uniform samples using
                             scipy.special.struve and scipy.special.yv. Exports
                             interleaved f32 binary [S0[0],S1[0],S0[1],S1[1],...].
                             Output: lut_struve.bin (32768 bytes), lut_struve.npy,
                             lut_config.json.
```

---

## Physical Constants

All values sourced directly from `src/components.rs`:

| Constant | Value | Source |
|---|---|---|
| `G` | 6.6743e-11 m^3 kg^-1 s^-2 | Gravitational constant |
| `RYUGU_MASS` | 4.5e11 kg | Hayabusa2 estimate |
| `CASSINI_MASS` | 2500.0 kg | Probe model mass |
| `GRAVITY_EPSILON` | 1.0 m | Softening for Newtonian fallback only |
| `DENSITY_EPSILON` | 10.0 m | Softening in 1/(r+e) kernel |
| `TIME_SCALE` | 20000.0 | Simulation speed multiplier |
| `RYUGU_ROTATION_PERIOD_SECS` | 7.63 * 3600 = 27468 s | Physical spin period |
| `RYUGU_SPIN_AXIS` | Vec3(-0.043, -0.914, 0.405) | Unnormalized; normalized in code |
| `ORBIT_HISTORY_LEN` | 27500 | Trail point capacity |
| `NORMAL_ARROW_LENGTH` | 35.0 m (scene units) | Gizmo arrow length |
| `VISIBILITY_THRESHOLD` | 250.0 m (scene units) | Normal visibility cutoff |
| `SECTION_CLIP_RADIUS` | 450.0 m (scene units) | Section plane clip circle |

---

## Density Constant C: Derivation and Kernel

`DensityC` is solved once in `build_topology_system` (`src/systems/scale.rs`) after the mesh
is welded. It normalizes the gravitational kernel so that the total integrated mass equals
`RYUGU_MASS`.

**Kernel used in the code** (lines 116-119 of `scale.rs`):

```rust
let f = |p: DVec3| {
    1.0 / (p.length() + eps)  // eps = DENSITY_EPSILON = 10.0
};
```

**Integration method:** 4-point Gaussian quadrature (Keast rule) over each signed tetrahedron
formed by a mesh triangle and the origin. The barycentric coordinates are:

```
a = (5 + 3*sqrt(5)) / 20  ≈ 0.5854102
b = (5 - sqrt(5))   / 20  ≈ 0.1381966
```

Quadrature points:
```
p1 = va*a + vb*b + vc*b
p2 = va*b + vb*a + vc*b
p3 = va*b + vb*b + vc*a
p4 = va*b + vb*b + vc*b
```

All evaluated in f64 precision. Tetrahedron signed volume: `vol = va·(vb×vc) / 6`.

```
total_integral = sum over all triangles of: vol * 0.25 * (f(p1)+f(p2)+f(p3)+f(p4))
DensityC = RYUGU_MASS / total_integral
```

The same kernel `1/(||p||*scale + DENSITY_EPSILON)` is used in `build_gravity_voxels_system`
to assign per-voxel masses, ensuring the two are consistent: the density model used to
solve C is the same model used at runtime.

---

## Gravity Compute Pipeline: Step-by-Step

The pipeline spans main world, render world, and async GPU callbacks.

**Step 1 — Voxel data construction** (`build_gravity_voxels_system`, runs once)

For each vertex position `p` in the welded mesh (N vertices total):

```
r_v    = max(||p * scale||, 1e-3)
weight = 1.0 / (r_v + DENSITY_EPSILON)
```

Weights are summed, then normalized: `mass_i = weight_i * (RYUGU_MASS / sum_weights)`.

Each voxel is packed as `[x, y, z, mass]` (4 x f32, 16 bytes). The voxel buffer and the
LUT buffer are uploaded to GPU VRAM once and reused every frame.

**Step 2 — Frame-level uniform packing** (`dispatch_grav_system`)

Every frame, Cassini's position is transformed from world space into Ryugu's body-fixed frame:

```rust
let inv_rot  = ryugu_transform.rotation.inverse();
let local_pos = inv_rot * (cassini_pos - ryugu_pos);
```

An 80-byte uniform buffer is written with strict WGSL alignment:

```
bytes 0-15:  probe_x, probe_y, probe_z, G          (4x f32)
bytes 16-31: voxel_count, stehfest_M=6, pad, pad   (2x u32, 2x f32)
bytes 32-47: V[0..3]   (vec4<f32>)
bytes 48-63: V[4..7]   (vec4<f32>)
bytes 64-79: V[8..11]  (vec4<f32>)
```

The 12 Gaver-Stehfest coefficients for M=6 hardcoded in `gravity_pipeline.rs`:

```
k:   1       2        3           4           5            6
V: -17.0  1450.0  -27244.0  196885.333  -696515.5  1354060.167

k:   7          8           9          10        11       12
V: -1533036.0  1018861.833  -387807.667  79427.5  -7846.5  301.833
```

Maximum absolute value: |V_6| = 1,354,060.17 (approximately 1.35e6). These coefficients
alternate in sign and grow to magnitudes that nearly exhaust the 23-bit mantissa of fp32.

**Step 3 — GPU dispatch**

`ceil(N / 64)` workgroups are dispatched. Each thread processes one voxel.
After the compute pass completes, the output buffer is copied to a staging buffer.

**Step 4 — Async readback**

`staging.slice(..).map_async(MapMode::Read, callback)` is called. The callback fires on the
next frame when the GPU has finished. It writes the partial sum array to `GravityReadbackChannel`
(an `Arc<Mutex<Option<Vec<[f32;4]>>>>`).

**Step 5 — Main-world reduction** (`poll_gravity_readback`)

```rust
let total = partial_sums.iter().fold(Vec3::ZERO, |acc, v| acc + Vec3::new(v[0],v[1],v[2]));
grav_acc.0 = total;
```

**Step 6 — Frame-to-world rotation** (`physics_system`)

The GPU result is in Ryugu's body-fixed frame. It is rotated to world space:

```rust
let acceleration = r_transform.rotation * grav_acc.0;
```

---

## LUT and Special Functions

The Struve-Neumann decay modes are defined as:

```
S_v(z) = (pi/2) * [H_v(z) - Y_v(z)]
```

where `H_v` is the Struve function and `Y_v` is the Bessel function of the second kind
(Neumann function). These satisfy the derivative identity `S0'(z) = 1 - S1(z)`.

`scripts/gen_lut.py` computes S0 and S1 over z in [0, 50] at N=4096 uniform samples
using `scipy.special.struve` and `scipy.special.yv` in float64. The z=0 singularity is
handled by evaluating at z=1e-12 then overwriting [0] = 0.0 for both functions.

Output format: interleaved f32 binary `[S0[0], S1[0], S0[1], S1[1], ...]` — 4096 pairs,
8192 f32 values, 32768 bytes (32 KB). The WGSL shader reads this as `array<vec2<f32>>`.

**GPU-side lookup** (`get_S_modes` in `gravity.wgsl`):

```wgsl
let idx_f = clamp((z / 50.0) * 4095.0, 0.0, 4095.0);
let idx0  = u32(idx_f);
let idx1  = min(idx0 + 1u, 4095u);
let fract = idx_f - f32(idx0);
return mix(lut[idx0], lut[idx1], fract);   // hardware linear interp, O(1)
```

---

## Gaver-Stehfest + fp32: The Numerical Tightrope

### The Stehfest NILT formula

The Gaver-Stehfest algorithm recovers a time-domain function f(t) from its Laplace
transform F(s) evaluated at M=6 pairs of real probe frequencies:

```
f(h) ~= (ln2 / h) * sum_{k=1}^{12} V_k * F(s_k),   s_k = k * ln2 / h
```

where h is the probe distance (treated as the "time" variable) and V_k are the
precomputed alternating-sign Stehfest coefficients listed above.

### The as_val cancellation: a critical shader observation

Inside `compute_voxel_acc` (gravity.wgsl), the LUT argument is computed as:

```wgsl
let a     = h;         // approximation: set 'a' equal to probe distance h
let as_val = a * s_k;  // = h * (k * ln2 / h) = k * ln2
```

The h cancels exactly. `as_val` is always `k * ln2`, a fixed constant for each k,
completely independent of the actual probe-voxel distance or asteroid geometry.
The LUT is sampled at 12 fixed points: {0.693, 1.386, 2.079, ..., 8.318}.

This is a deliberate engineering simplification. In the full analytical kernel derived as
$$I_R = \int_0^\infty \frac{e^{-sw}}{\sqrt{w^2 + a^2}} dw = \mathcal{S}_0(as) = \frac{\pi}{2} \Big[ \mathbf{H}_0(as) - Y_0(as) \Big]$$,
the argument `a*s` would depend on the geometric parameter
`a = sqrt(u(r)^2 + c^2)` — a function of the actual source-observer configuration.
By setting `a = h`, the per-voxel kernel reduces to a purely distance-dependent form:

```
a_i = -(G * mass_i * ln2 / h^3) * C_stehfest * unit_dir
```

where `C_stehfest = sum_k V_k * (S0(k*ln2) + S1(k*ln2))` is a fixed scalar constant
evaluated at shader compile-time arguments. This makes the gravity model equivalent to
a weighted point-mass law with a distance decay encoded in the Stehfest-LUT weighting.

### fp32 catastrophic cancellation mitigation

The Stehfest coefficients alternate in sign with magnitudes up to ~1.35e6. If the sum
`sum_k V_k * G_total(s_k)` were computed by first accumulating G_total then scaling,
the intermediate values would overflow fp32 dynamic range against the small per-voxel
mass terms (~1e-11 range). The shader avoids this by multiplying V_k directly inside
the innermost voxel loop at thread level, before reduction:

```wgsl
let s_domain_acc = -mass_term * ((S0 + S1) / (h * h)) * unit_dir;
g_total += v_k * s_domain_acc;
```

Each thread accumulates the V_k-weighted per-voxel contribution locally. This keeps
intermediate products in the fp32 comfortable range before the workgroup reduction.

---

## WebGPU Parallel Reduction Architecture

The compute shader uses `@compute @workgroup_size(64, 1, 1)`. Each workgroup processes
64 voxels. The reduction pattern in WGSL:

```wgsl
var<workgroup> shared_acc: array<vec4<f32>, 64>;

// Thread stores its contribution
shared_acc[lid] = contrib;
workgroupBarrier();

// Parallel stride reduction: O(log2(64)) = 6 barrier synchronizations
var stride = 32u;
loop {
    if stride == 0u { break; }
    if lid < stride {
        shared_acc[lid] += shared_acc[lid + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
}

// Thread 0 writes the workgroup partial sum
if lid == 0u {
    output_acc[workgroup_id.x] = shared_acc[0];
}
```

Each workgroup writes one vec4 partial sum. The main world reduces these on CPU in
`poll_gravity_readback`. For a 200k-vertex mesh, this means approximately 3125 workgroups,
each doing a 6-stage tree reduction — far fewer total additions than a naive serial sum.

---

## Surface Normals Pipeline: Dual Path

There are two independent code paths for computing surface normals. They race:

**Path A — CPU** (`compute_normals.rs`, runs in `Update`):

Accumulates area-weighted face normal contributions per vertex synchronously,
then immediately inserts `AsteroidNormalsGpuData`. This runs on the same frame
that `build_topology_system` completes.

**Path B — GPU** (`compute_pipeline.rs` + `normals.wgsl`, fires once in `Render`):

Dispatches a one-shot compute pass over the CSR neighbor ring. The result travels
through a staging buffer async readback. `poll_normals_readback` in `Update` tries to
insert `AsteroidNormalsGpuData` from the channel — but only if it does not already exist:

```rust
fn poll_normals_readback(...) {
    if existing.is_some() { return; }   // short-circuits if CPU path already ran
    ...
}
```

**In practice, Path A always wins.** Path A runs synchronously in `Update` the same frame
topology is built. Path B requires at least one full frame roundtrip for the GPU readback.
Since `existing.is_some()` is true by the time the readback fires, Path B's result is
silently discarded. The GPU normals shader exists and dispatches correctly, but its output
is never used at runtime.

---

## Orbit Integrator

`physics_system` (`src/systems/physics.rs`) implements semi-implicit (symplectic) Euler:

```rust
let dt = time.delta_secs() * TIME_SCALE;   // TIME_SCALE = 20000.0

// Use GPU gravity if available; else Newtonian point-mass fallback
let acceleration = if grav_voxels.is_some() && grav_acc.0 != Vec3::ZERO {
    r_transform.rotation * grav_acc.0   // rotate body-frame result to world
} else {
    // Softened Newtonian: a = G*M*r / (|r|^2 + eps^2)^(3/2)
    let diff    = ryugu_pos - cassini_pos;
    let dist_sq = diff.length_squared() + GRAVITY_EPSILON * GRAVITY_EPSILON;
    (diff / dist_sq.sqrt()) * (G * RYUGU_MASS / dist_sq)
};

c_velocity.0    += acceleration * dt;      // velocity update first
c_transform.translation += c_velocity.0 * dt;  // position uses updated velocity
```

Velocity is updated before position. This is symplectic Euler (Euler-Cromer), which is
first-order but has better long-term energy behavior than explicit Euler.

**Initial orbit** (set in `render.rs`):

```rust
let r0     = Vec3::new(-650.0, 380.0, 700.0);  // |r0| ≈ 967 m from Ryugu center
let v_circ = (G * RYUGU_MASS / r0.length()).sqrt();
let v_init = r_hat.cross(Vec3::Y).normalize() * (1.11 * v_circ);
```

The 1.11 factor places Cassini at 11% above circular velocity, producing an eccentric
rather than circular initial orbit.

---

## Ryugu Rotation

`ryugu_rotation_system` applies a fixed angular increment each frame:

```rust
let angular_speed = TAU / RYUGU_ROTATION_PERIOD_SECS;   // rad/s (real time)
let delta_angle   = angular_speed * dt;                  // dt = delta_secs * 20000
let axis          = RYUGU_SPIN_AXIS.normalize();         // (-0.043,-0.914,0.405) normalized
let rotation      = Quat::from_axis_angle(axis, delta_angle);
transform.rotation = rotation * transform.rotation;
```

Because `dt` includes `TIME_SCALE = 20000`, Ryugu completes one full rotation in
7.63 / 20000 hours of wall-clock time ≈ 1.37 seconds of real time.

The body-fixed frame rotates with Ryugu. Cassini's position is expressed in this frame
each frame for the GPU dispatch, and the result is rotated back to world space afterward.

---

## Why Energy is NOT Conserved

There are three independent reasons energy is not conserved in this simulation:

**1. Time-varying potential field (physical, non-negotiable)**

Ryugu rotates with a physical period of 7.63 hours. Because the mass distribution
is not spherically symmetric (Ryugu has an equatorial ridge and irregular shape),
the gravitational potential U(r, t) is explicitly time-dependent. In a time-varying
potential, total mechanical energy E = KE + U is not a conserved quantity — the
potential does work on the orbiting probe as the field rotates. This is a real physical
effect, not a numerical artifact.

**2. Numerical integrator energy drift**

Symplectic Euler conserves a modified Hamiltonian exactly (not the true Hamiltonian),
which means it has bounded but nonzero energy oscillation. With a fixed `TIME_SCALE = 20000`
multiplier, the effective time step per frame is `delta_secs * 20000`. At 60 FPS this is
approximately 333 simulation seconds per rendered frame. Over long runs, the first-order
integrator introduces baseline secular energy drift proportional to the step size squared.

**3. fp32 precision and Stehfest truncation**

The Gaver-Stehfest series is truncated at M=6 (12 terms). The residual from truncation
introduces a systematic bias in the computed gravity vector. Combined with fp32 arithmetic
(~7 decimal digits of precision) and the alternating-sign cancellation of coefficients
up to 1.35e6 in magnitude, the computed acceleration field has numerical noise of order
fp32 rounding error times the coefficient magnitude. A non-conservative numerical force
field does not maintain curl-free conditions (∇×g_approx ≠ 0), which means work done
along closed paths is nonzero, directly violating energy conservation.

---

## System Ordering

All Update systems in `lib.rs` are chained with `.chain()`, enforcing strict sequential
execution every frame:

```
normalize_model_scale_system
  → build_topology_system
  → build_gravity_voxels_system
  → compute_asteroid_normals_system
  → physics_system
  → ryugu_rotation_system
  → camera_switch_system
  → normal_toggle_system
  → section_toggle_system
  → update_hint_on_mode_change
  → camera_follow_system
  → fps_update_system
  → section_alpha_system
  → render_gizmos_system
  → render_section_system
```

The first three systems are guarded by marker components (`Without<ScaleNormalized>`,
`Without<TopologyBuilt>`) and run only once. The GPU compute dispatches happen in the
`Render` schedule (render world), separately from this Update chain.

---

## Interactive Controls

| Key | Action |
|---|---|
| `S` | Toggle camera mode: Overview / Follow Cassini |
| `F` | Toggle surface normal arrow gizmos |
| `D` | Toggle cross-section plane view |
| Mouse drag | Pan-orbit camera rotation |
| Scroll wheel | Zoom |

---

## References

- Watanabe S. et al. (2019). Hayabusa2 arrives at the carbonaceous asteroid 162173 Ryugu. *Science* 364, 268-272.
- JAXA Hayabusa2 Mission: https://www.hayabusa2.jaxa.jp/en/
- Stehfest H. (1970). Algorithm 368: Numerical inversion of Laplace transforms. *CACM* 13(1), 47-49.
- Bevy Engine: https://bevyengine.org/
- W3C WebGPU Specification: https://www.w3.org/TR/webgpu/
- Mathematical derivation of the Laplace-domain gravity sensitivity kernel: `mathstrict.md` (in-repo)

---

## License

Copyright (C) 2026 Ryugu WASM Contributors

This program is free software: you can redistribute it and/or modify it under the terms of the
**GNU General Public License v3.0** as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

