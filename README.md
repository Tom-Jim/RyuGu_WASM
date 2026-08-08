# Ryugu WASM

[![Live Demo](https://img.shields.io/badge/Live_Demo-WebGPU-success?style=for-the-badge)](https://tom-jim.github.io/RyuGu_WASM/)
![Bevy](https://img.shields.io/badge/Bevy-0.19.0-purple.svg)
![Rust](https://img.shields.io/badge/Rust-Edition_2024-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

<img src=https://github.com/user-attachments/assets/2ce9f064-98bd-4658-8c0e-999abf0d0297 width="100%" alt="trajectory demonstration" />

| **Orbital Trajectory** | **ProbeView** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/ac05a5ac-e6e2-4f44-b0ca-8a447ba30b7f" width="100%"/> | <img src="https://github.com/user-attachments/assets/f58a18c3-161b-4945-9078-bcfa835c2ed4" width="100%"/> |
| **Change Orbit** | **Change Algorithm** |
| <img src="https://github.com/user-attachments/assets/7921e222-c7c4-4758-8dfe-82575efdeeb5" width="100%"/> | <img src="https://github.com/user-attachments/assets/6ac1d9aa-2a2f-4744-b8f0-47dd4f11b352" width="100%"/> |
---

Ryugu WASM is a real-time gravitational-dynamics simulator for asteroid (162173) Ryugu. It is written in Rust with Bevy 0.19, compiled to WebAssembly, and uses WebGPU compute shaders for forward gravity evaluation.

The simulator provides two GPU forward gravity modes plus an adaptive curved-arc solver:

- **Radial Analytic (Equation):** a non-uniform radial density model evaluated as angular cells and mass-preserving radial layers.
- **Werner Polyhedron:** a corrected closed-polyhedron Werner--Scheeres implementation for a homogeneous body with the same shape and total mass.
- **Eq.106 Adaptive Curved-Arc:** the non-periodic 70-to-106 Taylor transport is used by default; after ten stable orbit closures it promotes to the periodic branch. Segments are split until the documented Taylor convergence bound is safe.

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
- Four on-screen sliders for the probe's initial position and velocity multiplier.
- A `1x`--`8x` simulation-acceleration slider that skips presentation of fully integrated intermediate frames without enlarging the stable physics step.
- A scrolling rotating-frame Jacobi-constant chart with automatic vertical scaling.
- An Eq.157 dual-representation residual chart while the curved-arc solver is active.
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

WebGPU is available only in a [secure context](https://developer.mozilla.org/en-US/docs/Web/API/GPU): use HTTPS in deployment. `http://localhost` is accepted for local development.

### Enabling WebGPU by browser

#### Chrome and Chromium

1. Update to a current browser release. WebGPU has been enabled by default on supported desktop platforms since Chrome 113.
2. Open `chrome://settings/system` and enable **Use graphics acceleration when available**, then restart the browser.
3. Open `chrome://gpu` and confirm that WebGPU is hardware accelerated.
4. If a supported test machine is still blocked, open `chrome://flags/#enable-unsafe-webgpu`, enable **Unsafe WebGPU**, and restart. On Linux, `chrome://flags/#enable-vulkan` may also be needed for a Vulkan-backed configuration.

See the official [Chrome WebGPU troubleshooting guide](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips) for current platform restrictions. Flags are a development fallback and should not be required on an ordinarily supported configuration.

#### Microsoft Edge

1. Open `edge://settings/system` and enable **Use graphics acceleration when available**, then restart Edge.
2. Check the adapter and feature status at `edge://gpu`.
3. If WebGPU is unavailable on an otherwise supported development machine, enable `edge://flags/#enable-unsafe-webgpu` and restart.

Edge uses Chromium's WebGPU implementation, so Chrome's platform notes and driver requirements generally apply.

#### Firefox

WebGPU availability depends on Firefox version, operating system, and GPU driver. Try a current stable release first. If `navigator.gpu` remains unavailable:

1. Open `about:config` and accept the warning.
2. Search for `dom.webgpu.enabled` and set it to `true`.
3. Restart Firefox and test again. Firefox Nightly is the preferred fallback on a platform where stable Firefox has not enabled WebGPU by default.

Mozilla tracks the current platform rollout and preference in [Firefox experimental features](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features#webgpu_api).

#### Safari

Safari 26 ships WebGPU on supported Apple devices and normally requires no flag. Enable **Settings → Advanced → Show features for web developers**, then use **Develop → Feature Flags** only when testing an older Safari Technology Preview build where WebGPU is still exposed as an experimental switch. Apple's implementation maps WebGPU to Metal; see the [Safari 26 WebGPU release notes](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/) and the older [Technology Preview setup instructions](https://webkit.org/blog/14879/webgpu-now-available-for-testing-in-safari-technology-preview/).

After changing any browser setting, open the developer console on the served page and verify:

```js
Boolean(navigator.gpu && await navigator.gpu.requestAdapter())
```

A `true` result confirms that the page can obtain an adapter. It does not guarantee that every requested limit is supported; the application still reports adapter or device creation failures in the console.

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

## Why Rust, Bevy, WebAssembly, and WebGPU

This stack is selected for the simulator's specific workload rather than for rendering alone:

| Technology | Reason for this project |
|---|---|
| Rust | Strong ownership and type checks make asynchronous GPU buffers, readback channels, and ECS state easier to keep valid without a garbage collector in the simulation loop. |
| Bevy | Its ECS and separate main/render worlds provide an explicit place for fixed-step physics, one-shot mesh preprocessing, render extraction, compute dispatch, and asynchronous readback. |
| WebAssembly | The same Rust gravity and physics code is distributed as a browser artifact with near-native numeric execution and no local application installation. |
| WebGPU | Compute shaders, storage buffers, workgroups, and WGSL validation allow thousands of independent gravity contributions to be evaluated and reduced in parallel. WebGL does not provide the general compute pipeline used here. |

Together, Rust and Bevy keep the CPU-side data flow structured, WebAssembly makes that code portable to the browser, and WebGPU moves the dominant forward-gravity summation to the user's GPU through modern Metal, Direct3D 12, or Vulkan-class backends.

### Why Rust instead of C++ or Python

- **Rust:** provides native-level numerical performance while checking ownership, lifetimes, thread safety, and many buffer-layout mistakes at compile time. Its WebAssembly toolchain integrates directly with `wasm-bindgen`, and Bevy supplies one consistent ECS architecture for native and browser builds. This is the best fit for the project's asynchronous WebGPU readback and long-running simulation state.
- **C++:** can deliver comparable native and WebAssembly performance and has mature scientific libraries. It was not selected here because manual memory ownership, pointer lifetime management, and JavaScript/WASM binding maintenance would increase the risk and maintenance cost around asynchronous mapped GPU buffers. C++ remains a reasonable option where an existing C++ physics engine or library ecosystem is the primary requirement.
- **Python:** is excellent for deriving formulas, preprocessing data, validating results, and rapidly prototyping numerical experiments; the project still uses Python for research scripts where appropriate. It was not selected for the interactive runtime because ordinary Python execution cannot directly provide the same browser-native WebGPU/Bevy pipeline, and per-frame Python or Python-to-browser bridging would add overhead around the 60 Hz simulation loop.

The choice is therefore workload-specific: Rust is used for the real-time browser runtime, while Python remains useful for offline analysis and validation. The decision does not imply that Rust is universally faster or more suitable than C++ for every numerical project.

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

Each dispatch carries a monotonically increasing request ID, simulation epoch, simulation time, and a snapshot of the probe and Ryugu transforms. The asynchronous readback returns that snapshot together with its acceleration and potential, so an old body-frame result is never rotated with a newer asteroid attitude. Completed samples are kept in a bounded eight-entry history instead of being reduced to an unlabelled "latest value".

Main-world polling runs in `PreUpdate`, before the following `FixedUpdate`. Partial workgroup results are accumulated in CPU `f64`, while the GPU remains responsible for the parallel field evaluation.

## Physics and frame pacing

`physics_system` and `ryugu_rotation_system` run in `FixedUpdate` at 60 Hz. Simulation time is accelerated by `TIME_SCALE = 500`; one fixed update therefore advances approximately `8.33 s` of simulation time.

Each fixed update is divided into 12 kick--drift--kick leapfrog substeps:

```text
velocity += 0.5 * acceleration(position, t) * substep_dt
position += velocity * substep_dt
velocity += 0.5 * acceleration(position, t + substep_dt) * substep_dt
```

Before the first valid GPU sample, the integrator uses a softened Newtonian point-mass acceleration. Once samples arrive, the integrator predicts only the non-spherical residual relative to a point mass: it evaluates the point-mass anchor at the current substep position and adds a history-derived residual. This avoids extrapolating the dominant radial field from a stale probe position.

Within the known sample interval, residual acceleration uses cubic Hermite interpolation. Beyond the newest sample, radial mode permits at most two sample intervals of slope-limited extrapolation; Werner mode holds the newest residual because browser measurements showed that extrapolating its more cancellation-sensitive samples increased drift. Ryugu's known rotation is evaluated analytically at every substep boundary. A valid combined acceleration is clamped to `1.5e-3 m/s²` and blended in over 60 fixed updates.

The upper-right simulation-acceleration control selects `1x` through `8x`. At `Nx`, one displayed fixed update completes `N` full stable physics frames, each retaining the same `8.33 s` frame interval and 12 leapfrog substeps. Intermediate states are added to the orbit trail but are not presented individually. The multiplier therefore increases simulated time per displayed frame without multiplying `dt`; it does not make a single integration step coarser. GPU readback remains once per displayed frame, so the range is deliberately bounded.

Frame pacing uses:

- `PresentMode::AutoVsync`;
- a 60 Hz focused Winit update interval;
- `bevy_framepace` at 60 FPS for native builds;
- a 30 Hz low-power unfocused interval.

## Rotating-frame Jacobi-constant chart

The lower-right display plots the specific Jacobi constant in Ryugu's body-fixed rotating frame. Let \(R\) be Ryugu's world rotation, \(\boldsymbol\omega\) its angular velocity, and \(U>0\) the positive gravitational potential returned by the active GPU model. The body-frame position and velocity are

$$
\mathbf r_b=R^{-1}(\mathbf r_p-\mathbf r_R),
\qquad
\mathbf v_b=R^{-1}\mathbf v_p-\boldsymbol\omega_b\times\mathbf r_b,
\qquad
\boldsymbol\omega_b=R^{-1}\boldsymbol\omega.
$$

The displayed diagnostic is

$$
C_J=2U(\mathbf r_b)
+\lVert\boldsymbol\omega_b\times\mathbf r_b\rVert^2
-\lVert\mathbf v_b\rVert^2,
$$

with units of $\mathrm{m^2/s^2}$. This is a specific invariant, so probe mass does not appear.

Both compute shaders reduce acceleration in `xyz` and the matching positive potential in `w`. The radial shader evaluates both quantities at the same eight Gauss--Legendre nodes, avoiding an f32 central difference of nearly equal potentials. The Werner shader uses the standard edge-minus-face polyhedron field and a compensated double-single sum for its strongly cancelling scalar potential. The chart selects the potential belonging to the active gravity method and waits until the 60-step GPU warm-up blend is complete before recording samples.

Every chart sample is evaluated from the position, velocity, Ryugu attitude, and potential carried by the same completed GPU request. It never combines the current probe state with a delayed potential. For exact continuous dynamics in a steadily rotating, time-independent body-fixed potential, $C_J$ should remain constant. Visible drift therefore helps expose remaining integration, interpolation, clamping, or f32 field error; it is a diagnostic rather than a proof of exact conservation.

The chart retains 256 timestamped GPU samples. Before the time window fills, the newest point moves from left to right. After it fills, old samples leave from the left, the newest yellow point remains on the right edge, and the previous values continue as a scrolling line. The visible minimum and maximum receive an 8% margin, so the vertical scale automatically expands or contracts while keeping the full visible trajectory inside the plot. The label `dC/|C0|` reports the relative change across the currently visible history, making small variations distinguishable from a large physical drift.

In the final browser validation at the default initial conditions, the visible-window relative change was approximately `-2.4e-4%` in radial mode and `+2.3e-2%` in Werner mode while rendering at 59--60 FPS. These are observed regression values, not universal error bounds for every orbit or slider setting.

Changing a probe slider or switching gravity methods resets the trajectory, GPU-potential warm-up, and Jacobi history so unrelated runs are not joined into one curve.

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
| `G` | Cycle through Equation, homogeneous Werner, and Eq.106 adaptive curved-arc gravity. |
| `X`, `Y`, `Z` sliders | Set the three components of the initial probe position from `-2000` to `2000` in 100 intervals (`40` per step). |
| `Speed` slider | Set the circular-speed multiplier from `0` to `2` in 100 intervals (`0.02` per step). |
| Upper-right acceleration slider | Advance `1`--`8` complete stable physics frames per displayed frame. |
| Mouse drag | Orbit the camera. |
| Scroll wheel | Zoom. |

Moving a probe slider immediately clears the old trajectory, applies the new position and tangent velocity, resets Ryugu's rotation, and warms the active GPU result from the Newtonian fallback. Switching with `G` performs the same reset using the current slider values.

The orbit line is cyan in Equation mode, red in Werner mode, and purple in Eq.106 curved-arc mode.

## Physical constants

| Constant | Value | Purpose |
|---|---:|---|
| `G` | `6.6743e-11` | Gravitational constant. |
| `RYUGU_MASS` | `4.5e11 kg` | Total asteroid mass. |
| `CASSINI_MASS` | `2500 kg` | Probe mass. |
| `DENSITY_EPSILON` | `10 m` | Radial-density regularization. |
| `GRAVITY_EPSILON` | `1 m` | Point-mass fallback softening. |
| `RYUGU_ROTATION_PERIOD_SECS` | `7.63 h` | Physical rotation period. |
| `TIME_SCALE` | `500` | Simulation speed multiplier chosen to keep asynchronous interpolation and integration error controlled. |
| `PHYSICS_SUBSTEPS` | `12` | Leapfrog substeps per 60 Hz fixed update. |
| `MIN_SIMULATION_ACCELERATION` | `1` | Minimum stable physics frames advanced per displayed frame. |
| `MAX_SIMULATION_ACCELERATION` | `8` | Maximum stable physics frames advanced per displayed frame. |
| `GRAVITY_SAMPLE_HISTORY_CAPACITY` | `8` | Timestamped GPU field samples retained for interpolation and matching diagnostics. |
| `ORBIT_HISTORY_LEN` | `27500` | Maximum stored trail points. |
| `PROBE_SPEED_FACTOR` | `1.053` | Default multiplier applied to the local circular-orbit speed. |
| `JACOBI_HISTORY_CAPACITY` | `256` | Number of Jacobi-constant samples retained by the scrolling chart. |

The current initial probe state is:

```text
position = (-1000, 1200, 100)
speed    = 1.053 * sqrt(G * RYUGU_MASS / |position|)
```

The on-screen sliders replace these defaults at runtime. A zero position produces a zero initial velocity instead of an undefined normalized direction.

## Scheduling

### Startup

```text
setup_scene, setup_ui, setup_fps_ui, setup_probe_controls,
setup_simulation_acceleration_control, setup_jacobi_chart
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

The gravity plugins independently build their immutable source resources in `Update` and poll completed readbacks in `PreUpdate`, so newly mapped samples are available before fixed-step physics.

### Fixed update

```text
physics_system -> ryugu_rotation_system -> record_probe_jacobi_system
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
│       ├── energy.rs
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
| `src/components.rs` | Physical constants, runtime probe initial conditions, ECS components, gravity mode, shared sources, and readback resources. |
| `src/topology.rs` | CSR adjacency construction from the welded mesh. |
| `src/welding.rs` | Quantized vertex deduplication. |
| `src/systems/scale.rs` | One-shot model normalization and topology creation. |
| `src/systems/energy.rs` | Snapshot-aligned body-frame Jacobi evaluation, relative-drift reporting, rolling history, automatic chart scaling, and right-edge point rendering. |
| `src/systems/compute_pipeline.rs` | One-shot GPU surface-normal computation and readback. |
| `src/systems/gravity_pipeline.rs` | Radial angular/layer preprocessing, snapshot-tagged GPU dispatch, f64 partial reduction, and history insertion. |
| `src/systems/werner_pipeline.rs` | Homogeneous closed-polyhedron preprocessing, snapshot-tagged Werner dispatch, f64 partial reduction, and history insertion. |
| `src/systems/physics.rs` | Point-mass residual interpolation, bounded async prediction, 12-substep leapfrog integration, and asteroid rotation. |
| `src/systems/render.rs` | Scene, camera, orbit gizmos, normals, and method-aware density section. |
| `src/systems/ui.rs` | FPS display, keyboard controls, four probe sliders, and the upper-right simulation-acceleration slider. |
| `assets/shaders/gravity.wgsl` | Joint eight-node radial acceleration/potential quadrature and workgroup reduction. |
| `assets/shaders/werner_gravity.wgsl` | Shared-edge and signed-face Werner field evaluation with compensated potential summation. |
| `assets/shaders/normals.wgsl` | CSR-neighbor surface normals. |
| `server.ts` | Static Bun server with COOP/COEP headers. |
| `index.html` | WebGPU preflight and WASM bootstrap. |
| `scripts/gen_lut.py` | Legacy LUT generator; retained for research history and unused by the current runtime. |

## Testing

The Rust tests cover:

- the Equation radial primitive against direct numerical integration;
- the radial potential primitive against direct numerical integration;
- 1000 deterministic f32 radial rays against high-resolution direct integration, including acceleration/potential gradient consistency;
- Werner acceleration and potential far-field limits;
- Werner potential finite-difference gradient consistency for a closed tetrahedron;
- interpolation and bounded-extrapolation behavior for asynchronous field samples;
- the rotating-frame Jacobi definition and invalid-input handling;
- mass-density radial integration;
- solid-angle construction;
- Rust/WGSL uniform-buffer layouts;
- Werner far-field behavior for a closed tetrahedron;
- WGSL parsing and semantic validation with Naga.

## Known limitations

- Equation currently assumes a star-shaped body and one radial interval per direction.
- Four radial layers approximate the continuous density pointwise, although each layer mass is preserved.
- Werner mode is homogeneous and is not a non-uniform-density Werner extension.
- GPU readback remains asynchronous. Snapshot tags prevent state mismatch, but force prediction between completed samples is still a numerical approximation.
- At acceleration above `1x`, the integration step remains unchanged but GPU field samples are farther apart in simulation time; the `8x` cap limits this interpolation tradeoff.
- Leapfrog substeps greatly reduce secular integration drift, but this remains an interactive f32 visualization rather than a precision orbit-determination tool.
- The current `TIME_SCALE = 500` is a deliberate fidelity/performance compromise; increasing it without increasing the GPU sampling rate or changing the predictor can reintroduce drift.
- The browser page requires WebGPU; it does not load the full simulation after the preflight fails.

## Deployment

`.github/workflows/deploy.yml` builds the release WASM package, assembles `index.html`, `assets/`, and `pkg/` into a Pages artifact, and deploys it to GitHub Pages.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
