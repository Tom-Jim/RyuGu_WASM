# Ryugu WASM — Asteroid Gravitational Dynamics Simulator

[![Live Demo](https://img.shields.io/badge/Live_Demo-WebGPU-success?style=for-the-badge)](https://tom-jim.github.io/RyuGu_WASM/)
![Bevy](https://img.shields.io/badge/Bevy-0.19.0-purple.svg)
![Rust](https://img.shields.io/badge/Rust-Edition_2024-orange.svg)
![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)

Real-time orbital simulator of asteroid (162173) Ryugu, compiled to WASM via `wasm-pack`. A Cassini-class probe orbits Ryugu under gravity computed each frame by a WebGPU compute shader running a Gaver-Stehfest Numerical Inverse Laplace Transform (NILT) against a precomputed Struve-Neumann function lookup table. The Bevy 0.19 engine handles rendering, ECS scheduling, and the render-world/main-world GPU readback pattern.

https://github.com/user-attachments/assets/a78d663a-cb98-47b3-91c3-7303a12346cd

| Orbital Trajectory | Precession |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/ddc359c3-5b81-47ee-95fd-a9a1ccf20107" width="100%"/> | <img src="https://github.com/user-attachments/assets/d9482e7a-2c14-46b4-91a8-71a884567236" width="100%"/> |
| **Mass distribution** | **Probe view** |
| <img src="https://github.com/user-attachments/assets/9959363f-ab28-45e5-b43d-98e076b1791b" width="100%"/> | <img src="https://github.com/user-attachments/assets/548a0e90-2b29-484b-a974-a4f28d4bfe17" width="100%"/> |
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

## 🌐 Browser Support & WebGPU Setup

> ⚠️ **Crucial Warning**  
> This project relies on **WebGPU** and **WASM multi-threading**. **Your browser must have WebGPU properly enabled to run the simulation, otherwise the page will render a black screen.** Please verify your browser environment and configuration before launching.

---

### 1. Browser Compatibility

| Browser | Support Status | Notes |
| :--- | :--- | :--- |
| **Google Chrome** | 🟢 `113+` | Recommended / Native Support |
| **Microsoft Edge** | 🟢 `113+` | Recommended / Native Support |
| **Apple Safari** | 🟢 `18+` | macOS / iOS Native Support |
| **Mozilla Firefox** | 🟡 Experimental | Requires flag enabled; **Not supported in production** |

---

### 2. Cross-Origin Isolation (COOP / COEP Headers)

To enable **Bevy WASM multi-threading** and `SharedArrayBuffer` support, the dev server (`server.ts`) sets the following headers on every response:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp

```

> 💡 **Deployment Note:** If you deploy the built static assets to production environments (e.g., Nginx, Vercel, Cloudflare Pages), make sure to configure these HTTP headers on your web server; otherwise, WASM thread support will fail to initialize.

---

### 3. How to Check WebGPU Status

1. Open Chrome or Edge and navigate to `chrome://gpu`.
2. Look for **WebGPU** under the **Graphics Feature Status** list:
* 🟢 **Hardware accelerated**: WebGPU is active and working.
* 🔴 **Disabled** or **Unavailable**: WebGPU is inactive. Running the project will result in a **black screen**. Refer to the setup guide below.



---

### 4. Step-by-Step WebGPU Setup Guide

#### 🔹 Chrome / Edge

WebGPU is enabled by default in Chrome/Edge 113+. If you encounter a black screen, follow these troubleshooting steps:

1. **Verify Hardware Acceleration (Required):**
* Go to **Settings** -> **System**.
* Turn on **"Use graphics acceleration when available"**.
* **Relaunch the browser**.


2. **Force Enable WebGPU (If black screen persists):**
* Navigate to `chrome://flags/#enable-unsafe-webgpu`.
* Set **Unsafe WebGPU** to `Enabled`.
* Click **Relaunch** at the bottom right.



#### 🔹 Firefox (Testing / Experimental Only)

Firefox WebGPU support is experimental and not supported in production. To test:

1. Navigate to `about:config` and accept the warning prompt.
2. Search for `dom.webgpu.enabled` and set it to `true`.
3. (On some platforms) Search for `gfx.webgpu.force-enabled` and set it to `true`.
4. **Restart Firefox**.

#### 🔹 Safari (macOS / iOS)

* Safari 18+ natively supports WebGPU with no additional setup required.
* If using an older version, update your OS or switch to Chrome/Edge 113+.

---

### 5. Troubleshooting

* **Still getting a black screen after enabling WebGPU?**
1. **Update GPU Drivers:** Ensure your graphics drivers (NVIDIA / AMD / Intel) are fully updated.
2. **Switch to Discrete GPU:** If using a dual-GPU laptop, configure system settings to force your browser to use the high-performance / dedicated GPU.
3. **Inspect DevTools Logs:** Press `F12` to open Developer Tools and check the Console tab for errors like `WebGPU is not supported` or `Adapter not found`.


**Before running,** `lut_struve.bin` must exist in the project root.
If the file is missing, `build_gravity_voxels_system` calls
`std::fs::read("lut_struve.bin").unwrap()` and panics immediately.

Regenerate with:

```sh
uv run scripts/gen_lut.py
```

---

## 🦀 Why Rust? (Tech Stack Justification)

A common question is whether this real-time asteroid gravity simulation could be implemented in **Python** or **C++**. Below is the architectural justification for selecting **Rust + WebGPU + WASM**:

### Performance & Architecture Comparison

| Feature / Metric | 🦀 **Rust (Selected)** | 🐍 **Python** | ⚡ **C++** |
| :--- | :--- | :--- | :--- |
| **In-Browser Execution (WASM)** | **Native (60+ FPS)** | Python Extremely Slow (Pyodide / PyScript overhead) | **Native (60+ FPS)** |
| **WebGPU Ecosystem** | 🟢 **First-Class** (`wgpu` powers Firefox/Deno native WebGPU) | 🟡 Desktop-only (`wgpu-py`), poor web support | 🟡 Complex setup (Dawn / Emscripten C bindings) |
| **Async GPU Safety** | 🛡️ **Guaranteed at compile-time** (`Arc<Mutex<T>>`, Ownership) | High runtime overhead & GIL bottlenecks | ⚠️ Prone to race conditions & dangling pointers |
| **Build & Toolchain** | 🟢 Single `cargo build --target wasm32-unknown-unknown` | 🔴 Multi-megabyte interpreter bundle | 🔴 Complex Emscripten / CMake toolchain difficulty |

### Key Technical Advantages of Rust

* **Zero-Cost WebAssembly Integration:** Rust compiles directly to lean WASM without requiring a Garbage Collector (GC) or a heavy runtime engine. This guarantees deterministic 60 FPS physics integration and minimal frame variance.
* **Memory Safety in Async Readbacks:** WebGPU mapped staging buffers require asynchronous memory synchronization between GPU and CPU worlds. Rust’s strict ownership model eliminates data races and use-after-free bugs when lock-free channels poll GPU staging buffers.
* **Ecosystem Synergy with `wgpu` & `Bevy`:** The core Rust `wgpu` crate is the industry reference implementation for WebGPU. Combined with Bevy's ECS (Entity Component System), the simulation decouples physics integrators, GPU dispatch loops, and UI gizmos cleanly.
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
│       │                    M=3, pad, pad, V[0..5] padded to 3x vec4<f32>].
│       │                    dispatch_grav_system: dispatches n_workgroups=ceil(N/64)
│       │                    workgroups. Initiates async staging buffer map_async.
│       │                    poll_gravity_readback: folds all workgroup partial vec4
│       │                    results into GravityAcceleration resource.
│       ├── physics.rs       physics_system: Euler-Cromer (semi-implicit/symplectic Euler).
│       │                    Rotates GPU result from Ryugu body frame to world frame.
│       │                    Blends Newtonian and GPU gravity over GRAVITY_BLEND_FRAMES=60
│       │                    frames. Clamps GPU acceleration to MAX_ACC=1.5e-3 m/s².
│       │                    ryugu_rotation_system: incremental quaternion rotation
│       │                    applied each frame using delta_angle = omega * dt.
│       ├── render.rs        Scene setup, lighting, camera, Cassini initial conditions.
│       │                    Orbit history ring buffer (capacity ORBIT_HISTORY_LEN=27500).
│       │                    Gizmo rendering for trajectory, normals, section plane.
│       ├── compute_pipeline.rs  NormalsComputePlugin: one-shot GPU dispatch (DispatchState
│       │                    goes Done after first fire). Writes result to staging buffer,
│       │                    maps async, pushes to NormalsReadbackChannel.
│       └── ui.rs            FPS display, keyboard toggle handlers for normals/section.
├── assets/shaders/
│   ├── gravity.wgsl         Gaver-Stehfest NILT loop over 6 terms (k=1..6, M=3).
│   │                        Per-thread voxel contribution written to workgroup shared
│   │                        memory (shared_acc[64]). Parallel stride reduction:
│   │                        stride starts at 32, halved each iteration via stride>>=1u.
│   │                        Each workgroup writes one vec4 partial sum to output_acc.
│   └── normals.wgsl         Per-vertex normal from CSR neighbor ring. Iterates
│                            consecutive neighbor pairs, accumulates cross(a-p, b-p).
│                            Normalizes; defaults to (0,1,0) for degenerate vertices.
└── scripts/
    └── gen_lut.py           Computes S_v(z) = (pi/2)[H_v(z) - Y_v(z)] for v=0,1
                             over z in [0,200] at N=4096 uniform samples using
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
| `GRAVITY_EPSILON` | 1.0 | Softening for Newtonian fallback only |
| `DENSITY_EPSILON` | 10.0 | Softening in 1/(r+e) kernel |
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
bytes 16-31: voxel_count, stehfest_M=3, pad, pad   (2x u32, 2x f32)
bytes 32-47: V[0..3]   (vec4<f32>)
bytes 48-63: V[4..5], 0, 0   (vec4<f32>, last 2 slots zero-padded)
bytes 64-79: 0, 0, 0, 0      (vec4<f32>, unused)
```

The 6 Gaver-Stehfest coefficients for M=3 hardcoded in `gravity_pipeline.rs`:

```
k:  1      2      3       4       5       6
V:  1.0  -49.0  366.0  -858.0  810.0  -270.0
```

Maximum absolute value: |V_4| = 858. These coefficients alternate in sign and sum to zero.

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

`scripts/gen_lut.py` computes S0 and S1 over z in [0, 200] at N=4096 uniform samples
using `scipy.special.struve` and `scipy.special.yv` in float64. The z=0 singularity is
handled by evaluating at z=1e-12 then overwriting [0] = 0.0 for both functions.

Output format: interleaved f32 binary `[S0[0], S1[0], S0[1], S1[1], ...]` — 4096 pairs,
8192 f32 values, 32768 bytes (32 KB). The WGSL shader reads this as `array<vec2<f32>>`.

**GPU-side lookup** (`get_S_modes` in `gravity.wgsl`):

```wgsl
let idx_f = clamp((z / 200.0) * 4095.0, 0.0, 4095.0);
let idx0  = u32(idx_f);
let idx1  = min(idx0 + 1u, 4095u);
let fract = idx_f - f32(idx0);
return mix(lut[idx0], lut[idx1], fract);   // hardware linear interp, O(1)
```

---

## Gaver-Stehfest + fp32: The Numerical Tightrope

### The Stehfest NILT formula

The Gaver-Stehfest algorithm recovers a time-domain function f(t) from its Laplace
transform F(s) evaluated at M=3 pairs of real probe frequencies:

```
f(h) ~= (ln2 / h) * sum_{k=1}^{6} V_k * F(s_k),   s_k = k * ln2 / h
```

where h is the probe-voxel distance (treated as the "time" variable) and V_k are the
precomputed alternating-sign Stehfest coefficients listed above.

### The as_val argument: actual shader behavior

Inside `compute_voxel_acc` (gravity.wgsl), the LUT argument is computed as:

```wgsl
let r_probe   = length(probe);                   // probe distance from Ryugu center
let safe_max_a = 200.0 * h / (6.0 * ln2);       // clamp so as_val <= 200 for all k
let a          = min(r_probe, safe_max_a);
let as_val     = a * s_k;                        // = r_probe * k * ln2 / h
```

`a` is set to `r_probe` (probe's distance from the asteroid center), **not** the
probe-voxel distance `h`. The LUT sampling point therefore varies with geometry:
`as_val = r_probe * k * ln2 / h`. The clamp ensures `as_val <= 200` (the LUT range)
for all k up to 6, which activates only for extremely close voxels (h < r_probe/48).

### fp32 catastrophic cancellation mitigation

The Stehfest coefficients alternate in sign with magnitudes up to ~858. The shader
multiplies V_k directly inside the innermost voxel loop at thread level, before reduction:

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

## Surface Normals Pipeline

Surface normals are computed once via a one-shot GPU compute pass (`NormalsComputePlugin`
in `compute_pipeline.rs` + `normals.wgsl`).

`build_topology_system` builds the CSR adjacency data. On the next frame, the GPU compute
pass dispatches over all vertices. The result travels through an async staging buffer
readback. `poll_normals_readback` in `Update` inserts `AsteroidNormalsGpuData` from the
channel when data is ready:

```rust
fn poll_normals_readback(...) {
    if existing.is_some() { return; }   // already populated, skip
    ...
    commands.insert_resource(AsteroidNormalsGpuData(normals));
}
```

The WGSL shader (`normals.wgsl`) computes each vertex normal by iterating the sorted CSR
neighbor ring: for each consecutive pair of neighbors (a, b), it accumulates
`cross(a-p, b-p)`, then normalizes. Vertices with fewer than 2 neighbors default to (0,1,0).

---

## Orbit Integrator

`physics_system` (`src/systems/physics.rs`) implements Euler-Cromer (semi-implicit/symplectic Euler):

```rust
let dt = time.delta_secs() * TIME_SCALE;   // TIME_SCALE = 20000.0

// Newtonian point-mass fallback (always computed as blend anchor)
let diff          = ryugu_pos - cassini_pos;
let dist_sq       = diff.length_squared() + GRAVITY_EPSILON * GRAVITY_EPSILON;
let newtonian_acc = (diff / dist_sq.sqrt()) * (G * RYUGU_MASS / dist_sq);

// GPU gravity with blend warm-up (0→1 over GRAVITY_BLEND_FRAMES=60 frames)
let acceleration = if grav_voxels.is_some() {
    let gpu_acc = r_transform.rotation * grav_acc.0;  // body→world frame
    if !gpu_acc.is_finite() || gpu_acc == Vec3::ZERO {
        newtonian_acc
    } else {
        let mag = gpu_acc.length();
        let safe_gpu = if mag > 1.5e-3 { gpu_acc * (1.5e-3 / mag) } else { gpu_acc };
        blend.0 = (blend.0 + 1.0 / 60.0).min(1.0);
        newtonian_acc.lerp(safe_gpu, blend.0)
    }
} else {
    newtonian_acc
};

c_velocity.0 += acceleration * dt;         // velocity update first (symplectic)
c_transform.translation += c_velocity.0 * dt;
```

Velocity is updated before position. This is symplectic Euler (Euler-Cromer), which is
first-order but has better long-term energy behavior than explicit Euler.

**Initial orbit** (set in `render.rs`):

```rust
let r0     = Vec3::new(-1000.0, 200.0, 100.0);  // |r0| ≈ 1025 m from Ryugu center
let v_circ = (G * RYUGU_MASS / r0.length()).sqrt();
let v_init = r_hat.cross(Vec3::Y).normalize() * (1.253 * v_circ);
```

The 1.253 factor places Cassini at 25.3% above circular velocity, producing an eccentric
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

The Gaver-Stehfest series is truncated at M=3 (6 terms). The residual from truncation
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
- Mathematical derivation of the Laplace-domain gravity sensitivity kernel: `mathpub.md` (in-repo)

---

## License

Copyright (C) 2026 Ryugu WASM Contributors

This program is free software: you can redistribute it and/or modify it under the terms of the
**GNU General Public License v3.0** as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

