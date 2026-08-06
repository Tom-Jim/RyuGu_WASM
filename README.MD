# Ryugu WASM

[![Live Demo](https://img.shields.io/badge/Live_Demo-WebGPU-success?style=for-the-badge)](https://tom-jim.github.io/RyuGu_WASM/)
![Bevy](https://img.shields.io/badge/Bevy-0.19.0-purple.svg)
![Rust](https://img.shields.io/badge/Rust-Edition_2024-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Ryugu WASM is a real-time gravitational-dynamics simulator for asteroid (162173) Ryugu. It is written in Rust with Bevy 0.19, compiled to WebAssembly, and uses WebGPU compute shaders for forward gravity evaluation.

The simulator provides two GPU gravity modes:

- **Radial Analytic (Equation):** a non-uniform radial density model evaluated as angular cells and mass-preserving radial layers.
- **Werner Polyhedron:** a corrected closed-polyhedron Werner--Scheeres implementation for a homogeneous body with the same shape and total mass.

These modes intentionally use different density assumptions. They are two forward models, not two numerical solvers for an identical density field.

## Features

- Equation spatial-domain GPU forward evaluation.
- Four equal-volume radial layers per angular cell.
- Mass-preserving discretization of $\rho(r)=C/(r+10)$.
- Homogeneous Werner polyhedron reference using shared-edge dyads and signed face solid angles.
- One-shot GPU surface-normal computation from welded-mesh CSR topology.
- Asynchronous GPU readback with reusable staging buffers and in-flight dispatch guards.
- 60 Hz fixed-step physics and Ryugu rotation.
- 60 FPS frame pacing, VSync, and a 30 Hz unfocused mode.
- Method-aware density cross-section visualization.
- Newtonian point-mass fallback while a GPU result is unavailable or invalid.

## Requirements

- Rust toolchain with the `wasm32-unknown-unknown` target.
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/).
- [Bun](https://bun.sh/) for the local server and package scripts.
- A browser with WebGPU support.

Install the Rust target if necessary:

```sh
rustup target add wasm32-unknown-unknown
```

## Quick start

```sh
# Prime Bun's package state.
bun install

# Debug WASM build, then serve at http://localhost:3000.
bun run dev

# Release WASM build only; output goes to pkg/.
bun run build

# Release build followed by the local server.
bun run preview

# Serve an existing pkg/ build.
bun run serve
```

Useful validation commands:

```sh
cargo fmt --all
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown
wasm-pack build --target web --dev
```

`cargo build` is a native syntax and type check. The browser artifact is produced by `wasm-pack`.

The current gravity implementation does not use `lut_struve.bin`; no LUT generation step is required before running the application.

## Browser and server behavior

The page checks `navigator.gpu` and requests an adapter before loading the WASM bundle. Chrome and Edge with WebGPU enabled are the recommended development browsers.

The Bun server returns these headers on every response:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The WebGPU render backend is configured with:

- `max_storage_buffers_per_shader_stage = 8`;
- `max_compute_workgroups_per_dimension = 65535`;
- canvas selector `#bevy`;
- `AssetMetaCheck::Never` for browser assets.

## Gravity models

### Radial Analytic: Equation

The default model uses

$$
\rho(r)=\frac{C}{r+\varepsilon},
\qquad \varepsilon=10\ \mathrm m.
$$

The surface mesh is converted into angular cells. Each cell stores a representative direction, a solid-angle weight, and a surface radius. It is then split into four equal-volume radial layers.

Each layer stores the volume-weighted mean density of the continuous model. This preserves the layer mass, and \(C\) is normalized so the complete discretization has mass `RYUGU_MASS`.

At runtime, one GPU invocation evaluates one angular-layer contribution. Workgroups contain 64 invocations and reduce their results to one partial acceleration. See [mathpub.md](mathpub.md) for the public GPU evaluation form of Equation.

The implementation assumes Ryugu is star-shaped relative to the model origin. A body with multiple disjoint radial intervals in one direction would require an extended source layout.

### Homogeneous Werner polyhedron

The Werner mode uses constant density

$$
\rho_{\mathrm W}=\frac{M_{\mathrm{Ryugu}}}{V_{\mathrm{mesh}}}.
$$

Mesh faces are oriented outward. Every watertight shared edge is combined with its two adjacent face normals to build one edge dyad. The GPU sums the shared-edge logarithmic terms and signed face-solid-angle terms, then multiplies the result by $G\rho_{\mathrm W}$.

This mode does not use `DensityC` or the radial layers. It is a homogeneous reference model.

### Density section view

Press `D` to show a camera-facing density section:

- Equation mode shows the continuous radial color field \(C/(r+10)\).
- Werner mode shows one uniform color throughout the interior because its density is constant.

The section is a visualization of the selected model. Equation itself uses four piecewise-constant, mass-preserving layers per angular cell.

## GPU execution and readback

Both gravity pipelines follow the same render-world pattern:

```text
main-world source data
  -> ExtractSchedule
  -> immutable GPU storage buffers
  -> per-frame body-fixed probe uniform
  -> WebGPU compute dispatch
  -> 64-thread workgroup reduction
  -> reusable MAP_READ staging buffer
  -> asynchronous channel
  -> main-world acceleration resource
```

An atomic `in_flight` flag prevents overlapping maps of the same staging buffer. Buffers and bind groups are reused instead of being allocated every frame.

The GPU result is computed in Ryugu's body-fixed frame and rotated back into world space before integration.

## Physics and frame pacing

`physics_system` and `ryugu_rotation_system` run in `FixedUpdate` at 60 Hz. Simulation time is accelerated by `TIME_SCALE = 20000`.

The probe uses Euler--Cromer integration:

```text
velocity += acceleration * dt
position += velocity * dt
```

Before the first valid GPU sample, the integrator uses a softened Newtonian point-mass acceleration. A valid GPU result is clamped to `1.5e-3 m/s²` and blended in over 60 fixed updates.

Frame pacing uses:

- `PresentMode::AutoVsync`;
- a 60 Hz focused Winit update interval;
- `bevy_framepace` at 60 FPS for native builds;
- a 30 Hz low-power unfocused interval.

## Surface topology and normals

After the GLTF scene loads:

1. The Ryugu root is uniformly scaled to a 900-unit maximum dimension.
2. Mesh vertices are welded with a quantization tolerance of `1e-4`.
3. A CSR adjacency list is constructed from the welded triangle mesh.
4. `NormalsComputePlugin` dispatches a one-shot compute shader.
5. The normal result is read back and displayed when `F` is enabled.

One-shot initialization is guarded by `ScaleNormalized` and `TopologyBuilt` marker components.

## Runtime controls

| Input | Action |
|---|---|
| `S` | Switch between overview and probe-follow camera modes. |
| `F` | Toggle GPU-computed surface-normal gizmos. |
| `D` | Toggle the density section for the active gravity model. |
| `G` | Switch between Equation and homogeneous Werner gravity. |
| Mouse drag | Orbit the camera. |
| Scroll wheel | Zoom. |

Switching with `G` clears the old trajectory, restores the initial probe state, resets Ryugu's rotation, and warms the selected GPU result from the Newtonian fallback.

The orbit line is cyan in Equation mode and red in Werner mode.

## Physical constants

| Constant | Value | Purpose |
|---|---:|---|
| `G` | `6.6743e-11` | Gravitational constant. |
| `RYUGU_MASS` | `4.5e11 kg` | Total asteroid mass. |
| `CASSINI_MASS` | `2500 kg` | Probe mass. |
| `DENSITY_EPSILON` | `10 m` | Radial-density regularization. |
| `GRAVITY_EPSILON` | `1 m` | Point-mass fallback softening. |
| `RYUGU_ROTATION_PERIOD_SECS` | `7.63 h` | Physical rotation period. |
| `TIME_SCALE` | `20000` | Simulation speed multiplier. |
| `ORBIT_HISTORY_LEN` | `27500` | Maximum stored trail points. |

The current initial probe state is:

```text
position = (-1000, 1200, 100)
speed    = 1.053 * sqrt(G * RYUGU_MASS / |position|)
```

## Scheduling

### Startup

```text
setup_scene, setup_ui, setup_fps_ui
```

### Main-world update chain

```text
normalize model scale
  -> build welded topology
  -> camera and keyboard controls
  -> UI updates
  -> section material update
  -> gizmo and density-section rendering
```

The gravity plugins independently build their immutable source resources and poll completed readbacks in `Update`.

### Fixed update

```text
physics_system -> ryugu_rotation_system
```

### Render world

```text
ExtractSchedule -> compute dispatch -> async buffer mapping
```

## Project structure

```text
Ryugu_wasm/
├── Cargo.toml
├── package.json
├── index.html
├── server.ts
├── mathpub.md
├── src/
│   ├── lib.rs
│   ├── components.rs
│   ├── topology.rs
│   ├── welding.rs
│   └── systems/
│       ├── mod.rs
│       ├── scale.rs
│       ├── compute_pipeline.rs
│       ├── gravity_pipeline.rs
│       ├── werner_pipeline.rs
│       ├── physics.rs
│       ├── render.rs
│       └── ui.rs
├── assets/
│   ├── models/
│   │   ├── ryugu.glb
│   │   ├── cassini.gltf
│   │   └── cassini.bin
│   └── shaders/
│       ├── gravity.wgsl
│       ├── werner_gravity.wgsl
│       └── normals.wgsl
├── scripts/
│   └── gen_lut.py
└── .github/workflows/
    └── deploy.yml
```

### File responsibilities

| File | Responsibility |
|---|---|
| `src/lib.rs` | Bevy app construction, WebGPU limits, plugins, schedules, fixed timestep, and frame pacing. |
| `src/components.rs` | Physical constants, ECS components, gravity mode, shared sources, and readback resources. |
| `src/topology.rs` | CSR adjacency construction from the welded mesh. |
| `src/welding.rs` | Quantized vertex deduplication. |
| `src/systems/scale.rs` | One-shot model normalization and topology creation. |
| `src/systems/compute_pipeline.rs` | One-shot GPU surface-normal computation and readback. |
| `src/systems/gravity_pipeline.rs` | Equation angular/radial preprocessing, GPU dispatch, and readback. |
| `src/systems/werner_pipeline.rs` | Homogeneous closed-polyhedron preprocessing, Werner GPU dispatch, and readback. |
| `src/systems/physics.rs` | Fixed-step fallback selection, acceleration blending, orbit integration, and asteroid rotation. |
| `src/systems/render.rs` | Scene, camera, orbit gizmos, normals, and method-aware density section. |
| `src/systems/ui.rs` | FPS display and keyboard controls. |
| `assets/shaders/gravity.wgsl` | Equation layer evaluation and workgroup reduction. |
| `assets/shaders/werner_gravity.wgsl` | Shared-edge and signed-face Werner evaluation. |
| `assets/shaders/normals.wgsl` | CSR-neighbor surface normals. |
| `server.ts` | Static Bun server with COOP/COEP headers. |
| `index.html` | WebGPU preflight and WASM bootstrap. |
| `scripts/gen_lut.py` | Legacy LUT generator; retained for research history and unused by the current runtime. |

## Testing

The Rust tests cover:

- the Equation radial primitive against direct numerical integration;
- mass-density radial integration;
- solid-angle construction;
- Rust/WGSL uniform-buffer layouts;
- Werner far-field behavior for a closed tetrahedron;
- WGSL parsing and semantic validation with Naga.

## Known limitations

- Equation currently assumes a star-shaped body and one radial interval per direction.
- Four radial layers approximate the continuous density pointwise, although each layer mass is preserved.
- Werner mode is homogeneous and is not a non-uniform-density Werner extension.
- GPU readback is asynchronous, so physics consumes the latest completed sample.
- Euler--Cromer is first-order accurate and the large time scale favors visualization over precision orbit determination.
- The browser page requires WebGPU; it does not load the full simulation after the preflight fails.

## Deployment

`.github/workflows/deploy.yml` builds the release WASM package, assembles `index.html`, `assets/`, and `pkg/` into a Pages artifact, and deploys it to GitHub Pages.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
