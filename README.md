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

Ryugu WASM is an interactive gravitational-dynamics simulator for asteroid (162173) Ryugu. The runtime is implemented in Rust with Bevy 0.19, compiled to WebAssembly, and uses WebGPU compute shaders for selected forward-field evaluations. The project is intended for numerical experimentation and software validation; it is not an orbit-determination or flight-certification tool.

The simulator exposes five complementary gravity algorithm slots through the `G` key:

- **Werner--Scheeres homogeneous polyhedron:** the classical reference algorithm. It reduces the exterior field of a closed, consistently oriented, constant-density polyhedron to edge and face sums.
- **Radial-analytic GPU method:** a project-specific discretization for a star-shaped body with radially structured density. It uses angular cells, mass-preserving radial layers, analytic radial primitives, and GPU reduction.
- **Equation (106) adaptive curved-trajectory method:** a project-specific trajectory representation. It transports the Equation (70) straight-line operator to curved arcs with an adaptive spatial Taylor expansion.
- **MMFFT + GPU-memory compression:** a dedicated 16-byte compressed source buffer, decode shader, tiled workgroup reduction, snapshot-tagged readback, Jacobi series, and performance slot.
- **FMM:** an octree-compressed inverse-density source representation with GPU traversal, multipole moments, and snapshot-tagged readback.

The five slots use different source representations and have different validity conditions. MMFFT and FMM use compressed inverse-density source data; their compression and truncation errors should still be characterized against an independent high-precision reference before scientific use.

## Features

- Equation spatial-domain GPU forward evaluation.
- Four equal-volume radial layers per angular cell, shared by the radial, Eq.(106), MMFFT, and FMM source paths.
- Mass-preserving discretization of $\rho(r)=C/(r+10)$.
- Homogeneous Werner polyhedron reference using shared-edge dyads and signed face solid angles.
- One-shot GPU surface-normal computation from welded-mesh CSR topology.
- Asynchronous GPU readback with reusable staging buffers and in-flight dispatch guards.
- 60 Hz fixed-step physics and Ryugu rotation.
- 60 FPS frame pacing, VSync, and a 30 Hz unfocused mode.
- Method-aware density cross-section visualization.
- A centered 16:9 presentation frame with letterboxing on non-16:9 browser viewports.
- A top-center `Rotate 90 deg` control that rotates the complete presentation frame by one quarter turn per press.
- Four on-screen sliders for the probe's initial position and velocity multiplier.
- A `1x`--`8x` simulation-acceleration slider that skips presentation of fully integrated intermediate frames without enlarging the stable physics step.
- A scrolling rotating-frame Jacobi-constant chart with automatic vertical scaling.
- An Eq.157 dual-representation residual chart while the curved-arc solver is active.
- A five-method performance workspace with per-method checkboxes, repeat testing, FPS curves, and Jacobi curves.
- A live `VRAM estimate` readout below the simulation-acceleration control. It reports the current algorithm's allocated GPU-buffer estimate, its share of the five-method total, and all five per-method estimates.
- Blocking numerical-error overlay; integration pauses while a selected evaluator is warming up and stops if certification fails.

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

Python research and validation tools are managed with `uv`. The repository
pins the stable interpreter used by this project in `.python-version` and
`pyproject.toml` (`Python 3.14.6`):

```sh
uv sync
uv run pytest -q
uv run python scripts/wasmtime_benchmark.py --wasm pkg/Ryugu_wasm_bg.wasm
```

The Wasmtime report measures wasm compilation and instantiation separately.
Because the generated Bevy artifact contains browser/WebGPU imports, a
headless run may report those imports and omit the numeric export call while
still providing a reproducible compile benchmark.

The top-center **Performance comparison** button opens an opaque performance
workspace that hides the 3D scene. The application cycles through the five
algorithms in repeated 120-frame measurement windows and continues until the
top-center **3D display** button is selected. The workspace shows a rolling
FPS plot and one Jacobi series for each selected method. Every method has a
checkbox; unchecking it removes its curves and excludes it from the queue.
The **Repeat benchmark** button starts a fresh pass without changing the
current selection. These measurements describe the current browser, GPU,
driver, window size, and selected probe state; they are not portable hardware
benchmarks or accuracy guarantees.

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
3. Restart Firefox and test again. Firefox Nightly can also be used for testing on a platform where stable Firefox has not enabled WebGPU by default.

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

This stack was selected for the simulator's current workload rather than as a general recommendation:

| Technology | Reason for this project |
|---|---|
| Rust | Ownership and type checking help organize asynchronous GPU buffers, readback channels, and ECS state. This reduces some classes of implementation error but does not remove numerical or browser-specific failure modes. |
| Bevy | Its ECS and separate main/render worlds provide an explicit place for fixed-step physics, one-shot mesh preprocessing, render extraction, compute dispatch, and asynchronous readback. |
| WebAssembly | The Rust gravity and physics code can be distributed as a browser artifact without a native installation. Runtime performance remains dependent on the browser and WebGPU implementation. |
| WebGPU | Compute shaders, storage buffers, workgroups, and WGSL validation allow thousands of independent gravity contributions to be evaluated and reduced in parallel. WebGL does not provide the general compute pipeline used here. |

Together, Rust and Bevy keep the CPU-side data flow structured, WebAssembly makes that code portable to the browser, and WebGPU moves the dominant forward-gravity summation to the user's GPU through modern Metal, Direct3D 12, or Vulkan-class backends.

### Why Rust instead of C++ or Python

- **Rust:** provides compile-time ownership and type checks and integrates with `wasm-bindgen`. These properties are useful for the asynchronous ECS and GPU readback code, but they do not by themselves establish numerical accuracy.
- **C++:** would also be a reasonable implementation language and may be preferable when an established native physics or mesh-processing library is required. The present project uses Rust to keep one implementation shared between native tests and the browser target.
- **Python:** is used for derivations, validation, tests, and Wasmtime benchmarking. It is not part of the interactive browser loop because the current runtime is organized around Bevy and WebGPU rather than a Python-to-browser bridge.

The choice is therefore workload-specific. It should not be interpreted as evidence that Rust is generally faster or more suitable than C++, Python, or other scientific-computing environments.

## Gravity models

>The project-specific radial-analytic, Eq.(106), MMFFT, and FMM constructions are experimental methods. Their mathematical scope and convergence conditions are documented in [`docs/mathtidy.md`](docs/mathtidy.md) and [`docs/mathtidy_EN.md`](docs/mathtidy_EN.md). Werner--Scheeres is the classical homogeneous reference.

### Radial Analytic: Equation

The default model uses

$$
\rho(r)=\frac{C}{r+\varepsilon},
\qquad \varepsilon=10\ \mathrm m.
$$

The surface mesh is converted into angular cells. Each cell stores a representative direction, a solid-angle weight, and a surface radius. It is then split into four equal-volume radial layers.

Each layer stores the volume-weighted mean density of the continuous model. This preserves the layer mass, and $C$ is normalized so the complete discretization has mass `RYUGU_MASS`.

At runtime, one GPU invocation evaluates one angular-layer contribution. Workgroups contain 64 invocations and reduce their results to one partial acceleration. Acceleration and positive potential use the same quadrature nodes.

The implementation assumes Ryugu is star-shaped relative to the model origin. A body with multiple disjoint radial intervals in one direction would require an extended source layout.

### Homogeneous Werner polyhedron

The Werner mode uses constant density

$$
\rho_{\mathrm W}=\frac{M_{\mathrm{Ryugu}}}{V_{\mathrm{mesh}}}.
$$

Mesh faces are oriented outward. Every watertight shared edge is combined with its two adjacent face normals to build one edge dyad. The GPU sums the shared-edge logarithmic terms and signed face-solid-angle terms, then multiplies the result by $G\rho_{\mathrm W}$.

This mode does not use `DensityC` or the radial layers. It is a homogeneous reference model.

### Equation (106) adaptive curved trajectory

Equation (106) begins with the Equation (70) complex-frequency field on a straight reference line and writes a curved trajectory as

$$
\mathbf q(t)=\overline{\mathbf q}(t)+\delta\mathbf q(t).
$$

The production browser path uses a bounded point-source representation of
the radial density. Its CPU reference directly evaluates the complex Laplace
kernel and derives the horizontal and vertical kernels. WGSL assembles the
same discrete source set on a fixed 257-frequency grid with fixed half-line
quadrature. A segmented Chebyshev table for $Q_{m-1/2}(\chi)$ is uploaded to
WGSL and used for an independent Fourier-kernel cross-check.

This matches the *discretized* Eq.(106) operator. The full continuous-density
mode coefficients of Eqs.(81)--(86), explicit Type-2/Type-3 NUFFT matrices of
Eqs.(89)--(95), and a matrix-valued GPU Padé sideband evaluator are not
claimed to be implemented.

In a fixed Cartesian basis, the mathematical transport operator is

$$
\mathbf g(\overline{\mathbf q}+\delta\mathbf q)
=\exp(\delta\mathbf q\cdot\nabla_{\mathbf q})
\mathbf g(\overline{\mathbf q}).
$$

Finite non-periodic arcs use local polynomial displacement models and mixed
spatial/complex-frequency derivatives by default. The periodic planner is
enabled only after at least ten consecutive orbit closures satisfy the
configured position, velocity, and period tolerances. CPU Taylor and Padé
certificates are diagnostic gates; the current WGSL force path uses analytic
point-field translation correction rather than a full matrix-valued Padé jet.

Each candidate arc is divided into as many straight reference segments as required by the convergence guard

$$
\varepsilon_{\max}
=\sup_h\frac{\|\delta\mathbf q(h)\|}{d(h)}<1,
$$

where $d(h)$ is a conservative distance from the reference line to the density support. The solver selects the lowest permitted Taylor order whose remainder estimate is below tolerance and bisects a segment when that order is insufficient.

While this algorithm is active, the lower-right chart displays an Eq.(157)-inspired dual residual: accumulated curved-path acceleration work minus the independently read-back potential difference. It is a consistency diagnostic, not a proof that the two continuous operators are independently implemented.

### Density section view

Press `D` to show a camera-facing density section:

- Radial-analytic, Equation (106), MMFFT, and FMM modes show the inverse-density color field $C/(r+10)$.
- Werner mode shows one uniform color throughout the interior because its density is constant.

The section is a visualization of the selected model. Eq.(106) aggregates
the radial layers into certified point sources, MMFFT quantizes those layers,
and FMM builds an octree from them; Werner remains homogeneous.

## Comparative scope

The methods should be selected according to the source model and trajectory,
not according to a single aggregate performance number.

| Method | Useful when | Main limitations in this implementation |
|---|---|---|
| Werner--Scheeres | A closed, consistently oriented, homogeneous polyhedron is an adequate model or an independent reference is required. | It does not represent heterogeneous density without decomposing the body into multiple polyhedra. Near-degenerate mesh geometry and cancellation in the scalar potential still require care. |
| Radial-analytic GPU | The body is approximately star-shaped and a radial density law such as $\rho(r)=C/(r+\varepsilon)$ is acceptable. Repeated pointwise evaluations can reuse the angular/layer source. | The star-shaped assumption excludes general multi-interval radial geometry. Angular and radial discretization introduce approximation error, and near-surface cells require conservative handling. |
| Equation (106) curved trajectory | A trajectory can be divided into segments satisfying the Taylor convergence guard and the same complex-frequency representation is reused along structured arcs. | The CPU reference uses mass/moment-compressed density cells and a certified frequency window. Strong curvature, near-surface motion, or failed residual checks stop the simulation instead of selecting another force model. |
| MMFFT + compression | A compact quantized source buffer and tiled GPU reduction are useful when memory bandwidth is the limiting resource. | Quantization and source discretization are approximate; the runtime path is not a continuous FFT/NUFFT proof. |
| FMM | Hierarchical inverse-density sources reduce far-field work for repeated evaluations. | The current tree/multipole order is truncated and requires an independent error study near the body surface. |

The current implementation is therefore a comparative research prototype. A
method should be considered acceptable only after comparison with an
independent reference for the geometry, density, and trajectory under study.

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

The FPS counter is followed by a `VRAM estimate` readout. Because WebGPU does
not expose a portable driver-level VRAM-usage counter, the displayed values are
calculated from the actual source, uniform, output, staging, spectrum, LUT,
and operator-table buffer sizes used by the five render-world pipelines. The
readout is positioned below the simulation-acceleration panel and updates as
source data or the active gravity method changes.

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

Before the selected evaluator has a valid snapshot or Equation (106) certificate, the integrator pauses. It never advances a trajectory using another force model. Radial mode interpolates completed body-frame samples and permits only bounded, slope-limited extrapolation; Werner, MMFFT, and FMM hold the newest valid sample.

Within the known sample interval, residual acceleration uses cubic Hermite interpolation. Beyond the newest sample, radial mode permits at most two sample intervals of slope-limited extrapolation; Werner mode holds the newest residual because browser measurements showed that extrapolating its more cancellation-sensitive samples increased drift. Ryugu's known rotation is evaluated analytically at every substep boundary. A valid combined acceleration is clamped to `1.5e-3 m/s²` and blended in over 60 fixed updates.

The upper-right simulation-acceleration control selects `1x` through `8x`. At `Nx`, one displayed fixed update completes `N` full stable physics frames, each retaining the same `8.33 s` frame interval and 12 leapfrog substeps. Intermediate states are added to the orbit trail but are not presented individually. The multiplier therefore increases simulated time per displayed frame without multiplying `dt`; it does not make a single integration step coarser. GPU readback remains once per displayed frame, so the range is deliberately bounded.

Frame pacing uses:

- `PresentMode::AutoVsync`;
- a 60 Hz focused Winit update interval;
- `bevy_framepace` at 60 FPS for native builds;
- a 30 Hz low-power unfocused interval.

## Rotating-frame Jacobi-constant chart

The lower-right display plots the specific Jacobi constant in Ryugu's body-fixed rotating frame. Let $R$ be Ryugu's world rotation, $\boldsymbol\omega$ its angular velocity, and $U>0$ the positive gravitational potential returned by the active GPU model. The body-frame position and velocity are

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

## Performance comparison

The browser performance workspace is opened with the top-center
**Performance comparison** button. It places an opaque UI layer over the 3D
scene so that the displayed frame rate is not visually confused with the
ordinary 3D view. The corresponding **3D display** button returns to the
simulation.

The workspace cycles through radial-analytic, Werner--Scheeres, Equation
(106), MMFFT + GPU-memory-compression, and FMM. Each selected mode is sampled in a
120-frame window, after which the next
mode is selected. The cycle repeats until the user returns to the 3D view. The
reported value is the observed browser frame rate during the current window;
it includes UI, rendering, WebGPU dispatch, readback scheduling, and the
current browser's presentation behavior. It is not a solver-only throughput
measurement and should not be compared across machines without recording the
browser version, GPU, driver, window size, and probe settings.

The workspace contains two rolling plots. The first stores one FPS series per
gravity method. The second stores one Jacobi series per selected method. Eq.
(106) keeps its near-straight and residual diagnostics in the 3D view's
dedicated history; MMFFT and FMM use their own compressed GPU source and
readback channels.

For a headless compilation and instantiation measurement, use the Python
Wasmtime script after building the browser artifact:

```sh
uv run python scripts/wasmtime_benchmark.py --wasm pkg/Ryugu_wasm_bg.wasm \
  --calls 8 --iterations 100000 --json
```

Wasmtime can compile the generated module without a browser, but the Bevy
artifact contains browser and WebGPU imports. Consequently, instantiation or
the exported numeric call may be unavailable in a headless environment. Such
a result is still useful for reporting module size, compilation time, and the
specific missing import; it is not a substitute for a browser benchmark.

## Surface topology and normals

After the GLTF scene loads:

1. The Ryugu root is uniformly scaled to a maximum dimension of 900 scene units.
2. Mesh vertices are welded with a quantization tolerance of `1e-4`.
3. A CSR adjacency list is constructed from the welded triangle mesh.
4. `NormalsComputePlugin` dispatches a one-shot compute shader.
5. The normal result is read back and displayed when `F` is enabled.

One-shot initialization is guarded by the `ScaleNormalized` and
`TopologyBuilt` marker components.

## Runtime controls

| Input | Action |
|---|---|
| `S` | Switch between overview and probe-follow camera modes. |
| `F` | Toggle GPU-computed surface-normal gizmos. |
| `D` | Toggle the density section for the active gravity model. |
| `G` | Cycle through radial-analytic, homogeneous Werner, Eq.106 adaptive curved-arc, MMFFT + memory-compression, and FMM gravity. |
| `X`, `Y`, `Z` sliders | Set the three components of the initial probe position from `-2000` to `2000` in 100 intervals (`40` per step). |
| `Speed` slider | Set the circular-speed multiplier from `0` to `2` in 100 intervals (`0.02` per step). |
| Upper-right acceleration slider | Advance `1`--`8` complete stable physics frames per displayed frame. |
| Mouse drag | Orbit the camera. |
| Scroll wheel | Zoom. |

Moving a probe slider immediately clears the old trajectory, applies the new position and tangent velocity, resets Ryugu's rotation, and warms the selected evaluator. Switching with `G` performs the same reset using the current slider values. Physics remains paused until that evaluator has a valid sample or certificate.

The orbit line is cyan in radial-analytic mode, red in Werner mode, and purple in Eq.106 curved-arc mode.

## Physical constants

| Constant | Value | Purpose |
|---|---:|---|
| `G` | `6.6743e-11` | Gravitational constant. |
| `RYUGU_MASS` | `4.5e11 kg` | Total asteroid mass. |
| `CASSINI_MASS` | `2500 kg` | Probe mass. |
| `DENSITY_EPSILON` | `10 m` | Radial-density regularization. |
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
setup_simulation_acceleration_control, setup_performance_controls,
setup_performance_chart_segments, setup_jacobi_chart
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
├── docs/
│   ├── mathtidy.md
│   ├── mathtidy_EN.md
│   └── mathstrict_EN.md
├── src/
│   ├── lib.rs
│   ├── components.rs
│   ├── topology.rs
│   ├── welding.rs
│   └── systems/
│       ├── mod.rs
│       ├── gravity/        # radial, Werner, Eq.(106), MMFFT, FMM
│       ├── gpu/            # shared render-world compute helpers
│       ├── model/          # scale and topology preparation
│       ├── presentation/   # camera, section view, controls, charts
│       └── simulation/     # physics and Jacobi diagnostics
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
│   └── wasmtime_benchmark.py
├── tests/
│   └── test_gravity_models.py
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
| `src/systems/gravity/eq106_reference.rs` | Independent f64 Eq.(106) kernels, common-frequency inversion, direct residuals, Taylor jets, and Padé certificates. |
| `src/systems/gravity/eq106_operator.rs` | Segmented Chebyshev coefficients for $Q_{m-1/2}(\chi)$, CPU certification, and GPU buffer serialization. |
| `src/systems/gravity/eq106_gpu.rs` | Render-world Eq.(106) spectrum assembly, Bromwich evaluation, and toroidal-harmonic cross-check. |
| `src/systems/gravity/radial.rs` | Radial angular/layer preprocessing, snapshot-tagged GPU dispatch, f64 partial reduction, and history insertion. |
| `src/systems/gravity/werner.rs` | Homogeneous closed-polyhedron preprocessing and Werner GPU dispatch/readback. |
| `src/systems/gravity/mmfft.rs` | Quantized inverse-density source packing, tiled GPU reduction, and readback. |
| `src/systems/gravity/fmm.rs` | Octree/multipole source construction, GPU traversal, and readback. |
| `src/systems/gpu/normals.rs` | One-shot GPU surface-normal computation and readback. |
| `src/systems/model/scale.rs` | One-shot model normalization and topology creation. |
| `src/systems/presentation/render.rs` | Scene, camera, orbit gizmos, normals, and method-aware density section. |
| `src/systems/presentation/ui.rs` | FPS display, keyboard controls, probe sliders, simulation acceleration, and performance workspace. |
| `src/systems/simulation/physics.rs` | Point-mass residual interpolation, bounded async prediction, fixed-step integration, and asteroid rotation. |
| `src/systems/simulation/energy.rs` | Snapshot-aligned rotating-frame Jacobi evaluation and residual charts. |
| `assets/shaders/gravity.wgsl` | Joint eight-node radial acceleration/potential quadrature and workgroup reduction. |
| `assets/shaders/werner_gravity.wgsl` | Shared-edge and signed-face Werner field evaluation with compensated potential summation. |
| `assets/shaders/eq106_complex.wgsl` | Fixed-frequency Eq.(106) spectrum, Bromwich inversion, and branch-free Chebyshev toroidal cross-check. |
| `assets/shaders/mmfft_compressed.wgsl` | Quantized inverse-density decode and tiled reduction. |
| `assets/shaders/fmm_gravity.wgsl` | Octree/multipole GPU traversal. |
| `assets/shaders/normals.wgsl` | CSR-neighbor surface normals. |
| `server.ts` | Static Bun server with COOP/COEP headers. |
| `index.html` | WebGPU preflight and WASM bootstrap. |
| `scripts/wasmtime_benchmark.py` | Headless Wasmtime compilation and instantiation benchmark for the generated browser module. |
| `tests/test_gravity_models.py` | Python checks for the inverse-density law and Equation (106) convergence guards. |
| `docs/mathtidy.md` | Chinese derivation and implementation conditions for the near-straight and Fourier-Chebyshev formulations. |
| `docs/mathtidy_EN.md` | English derivation of Equations (70), (79)--(110), (155)--(158), including convergence and residual analysis. |

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
- segmented $Q_{m-1/2}(\chi)$ Chebyshev tensor certification and truncated Fourier reconstruction;
- Rust/WGSL uniform-buffer layouts;
- Werner far-field behavior for a closed tetrahedron;
- WGSL parsing and semantic validation with Naga.

## Known limitations

- The radial-analytic method assumes a star-shaped body and one radial interval per direction.
- Four radial layers approximate the continuous density pointwise, although each layer mass is preserved.
- Werner mode is homogeneous and is not a non-uniform-density Werner extension.
- MMFFT and FMM use quantized or truncated hierarchical source representations; neither is a zero-error continuous-density solver.
- Eq.(106) requires every accepted Taylor segment to remain inside the convergence radius relative to the density support. CPU Taylor/Padé constructors and pole checks are present, but the full matrix-valued Padé sideband evaluator is not in WGSL; unresolved segments stop with an error.
- The GPU Eq.(106) path uses eight-point-source mass/moment compression, 257 common frequencies, fixed half-line quadrature, f32 spectrum storage, and analytic point-field translation correction. It does not yet implement the document's full continuous-density Fourier-Chebyshev coefficient assembly or explicit Type-2/Type-3 NUFFT matrix.
- The segmented toroidal-harmonic table is an independently certified Q-function cross-check. Mode truncation, interval coverage, f32 coefficient storage, and near-field fallback remain explicit approximation sources.
- The Eq.(157)-inspired chart residual compares curved-path work with read-back potential; it is not an independent continuous-density proof because both paths use the same discrete source model.
- The periodic Equation (106) branch is a promoted optimization after ten stable closures, not an assumption applied to a newly observed orbit.
- GPU readback remains asynchronous. Snapshot tags prevent state mismatch, but force prediction between completed samples is still a numerical approximation.
- At acceleration above `1x`, the integration step remains unchanged but GPU field samples are farther apart in simulation time; the `8x` cap limits this interpolation tradeoff.
- Leapfrog substeps greatly reduce secular integration drift, but this remains an interactive f32 visualization rather than a precision orbit-determination tool.
- The current `TIME_SCALE = 500` is a deliberate fidelity/performance compromise; increasing it without increasing the GPU sampling rate or changing the predictor can reintroduce drift.
- The browser page requires WebGPU; it does not load the full simulation after the preflight fails.

## Mathematical coverage audit

The implementation was checked against [`docs/mathtidy.md`](docs/mathtidy.md)
and [`docs/mathtidy_EN.md`](docs/mathtidy_EN.md):

| Document result | Status in this project |
|---|---|
| Eq.(79) toroidal-harmonic identity | CPU adaptive quadrature plus segmented Chebyshev $Q_{m-1/2}$ table; WGSL cross-check with explicit mode truncation. |
| Eqs.(81)--(86) density Fourier/Chebyshev separation | Not the production source assembly; the runtime uses eight mass/moment point sources. |
| Eqs.(89)--(95) explicit trajectory transform / NUFFT | Common-frequency Bromwich summation is present; explicit Type-2/Type-3 NUFFT matrices are not. |
| Eq.(106) straight-line complex-frequency field | Implemented for the discrete point-source representation on CPU and WGSL, including the vertical boundary identity in the CPU reference. |
| Eqs.(109)--(110) Bromwich inversion | Implemented on the shared 257-frequency grid with finite frequency and half-line truncation. |
| Eqs.(155)--(158) dual residual and convergence guard | CPU Taylor/Padé certificates and runtime residual gate are present; the residual is a discrete consistency diagnostic. |

Accordingly, the project is a validated GPU/CPU implementation of a
**discretized and truncated** Eq.(106) operator. It is not a proof of the
untruncated continuous-density formula, machine-precision equality, or a
guaranteed frame rate for every browser and trajectory.

## Deployment

`.github/workflows/deploy.yml` builds the release WASM package, assembles `index.html`, `assets/`, and `pkg/` into a Pages artifact, and deploys it to GitHub Pages.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
