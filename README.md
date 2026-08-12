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

Interactive Ryugu gravity and trajectory simulator written in Rust/Bevy, compiled to WebAssembly, and accelerated with WebGPU. It provides five gravity evaluators:

- radial analytic GPU quadrature;
- homogeneous Werner–Scheeres polyhedron;
- Equation (106) adaptive curved-trajectory evaluation;
- spherical-ring MMFFT with radix-2 FFT/IFFT;
- order-two FMM with P2M, M2M, M2L, L2L, and leaf near field.

The four heterogeneous methods use the mass-normalized density

$$
\rho(r)=C\ln\left(1+\frac r{10\,\mathrm m}\right).
$$

Werner remains a homogeneous reference. This project is intended for numerical experimentation, not flight certification.

## Features

- Five switchable gravity methods with independent GPU sources and readback histories.
- Fixed-step leapfrog integration with `1x`–`8x` presentation acceleration.
- Adaptive Equation (106) segmentation using curvature, distance, and Taylor-remainder bounds.
- Rotating-frame Jacobi and Equation (157) residual charts.
- Method-aware density section, orbit trail, probe view, surface normals, and performance comparison.
- Snapshot-tagged asynchronous WebGPU readback and numerical-error blocking.

## Requirements

- Rust with target `wasm32-unknown-unknown`.
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/).
- [Bun](https://bun.sh/).
- A WebGPU-capable browser.

```sh
rustup target add wasm32-unknown-unknown
```

## Quick start

```sh
bun install
bun run dev       # debug build and http://localhost:3000
bun run build     # release WASM in pkg/
bun run preview   # release build and server
bun run serve     # serve an existing build
```

Validation:

```sh
cargo fmt --all
cargo test
cargo clippy --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown
wasm-pack build --target web
python3 tests/test_gravity_models.py
```

Optional Python tooling uses `uv`:

```sh
uv sync
uv run pytest -q
uv run python scripts/wasmtime_benchmark.py --wasm pkg/ryugu_wasm_bg.wasm
```

## Browser and server behavior

The page checks `navigator.gpu` before loading WASM. Development uses `http://localhost`; deployments require HTTPS. The Bun server sends:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Enabling WebGPU by browser

Use a current browser and enable hardware acceleration.

#### Chrome and Chromium

Check `chrome://gpu`. For unsupported development configurations only, try `chrome://flags/#enable-unsafe-webgpu`.

#### Microsoft Edge

Check `edge://gpu`; Edge uses Chromium's WebGPU backend and flags.

#### Firefox

Use a current release. If necessary for testing, enable `dom.webgpu.enabled` in `about:config`.

#### Safari

Current Safari versions use Metal-backed WebGPU on supported Apple hardware. Older Technology Preview builds may require the WebGPU feature flag.

Adapter smoke test:

```js
Boolean(navigator.gpu && await navigator.gpu.requestAdapter())
```

## Why Rust, Bevy, WebAssembly, and WebGPU

| Technology | Role |
|---|---|
| Rust | Shared native/WASM implementation with strong ownership and type checking. |
| Bevy | ECS scheduling, fixed updates, render extraction, and GPU resource management. |
| WebAssembly | Browser delivery without a native installation. |
| WebGPU | Parallel field evaluation, reductions, and asynchronous readback. |

### Why Rust instead of C++ or Python

Rust fits the current Bevy/WASM/WebGPU pipeline. C++ is equally viable for native scientific libraries; Python remains useful for derivations and validation but is not in the interactive render loop.

## Gravity models

Detailed derivations and convergence conditions are in [`docs/mathtidy.md`](docs/mathtidy.md) and [`docs/mathtidy_EN.md`](docs/mathtidy_EN.md).

### Radial Analytic: Equation

The star-shaped surface mesh is converted to angular cells and four equal-volume radial layers. Each layer stores the exact mean of the logarithmic density, preserving total layer mass. GPU invocations evaluate acceleration and positive potential with the same eight-node radial quadrature.

The model assumes one radial interval per direction.

### Homogeneous Werner polyhedron

Werner uses

$$
\rho_W=M_{\mathrm{Ryugu}}/V_{\mathrm{mesh}}.
$$

The GPU evaluates shared-edge logarithmic terms and signed face-solid-angle terms for the closed, outward-oriented mesh. It does not use the heterogeneous radial density.

### Equation (106) adaptive curved trajectory

The trajectory is represented as

$$
\mathbf q(t)=\overline{\mathbf q}(t)+\delta\mathbf q(t),
\qquad
\mathbf g(\overline{\mathbf q}+\delta\mathbf q)
=\exp(\delta\mathbf q\!\cdot\!\nabla)\mathbf g(\overline{\mathbf q}).
$$

The GPU uses a mass-preserving `4×8×16` radial/polar/azimuth source quadrature and a 129-frequency Bromwich grid. Curved arcs are planned in the body frame. A segment is accepted only when

$$
\varepsilon_{\max}=\sup_h\frac{\|\delta\mathbf q(h)\|}{d(h)}<1
$$

and the geometric Taylor remainder is below tolerance. The curvature bound

$$
\frac{\kappa\ell^2}{2d_{\min}}<1
$$

forces shorter segments at higher curvature. Failed segments are bisected; an unresolved minimum segment stops integration instead of extrapolating outside the Taylor disk. Periodic mode requires ten stable closures.

### Density section view

Press `D` to display the selected density:

- radial, Equation (106), MMFFT, and FMM: outward-increasing logarithmic density;
- Werner: uniform density.

## Comparative scope

| Method | Strength | Main approximation |
|---|---|---|
| Werner | Closed homogeneous polyhedron reference. | No heterogeneous density. |
| Radial | Reusable star-shaped angular/radial source. | Four radial layers and angular discretization. |
| Equation (106) | Structured curved-trajectory spectral reuse. | Finite source/frequency quadrature and Taylor segmentation. |
| MMFFT | True periodic FFT/kernel/IFFT ring convolution. | Finite spherical-ring deposition. |
| FMM | Hierarchical P2M/M2M/M2L/L2L evaluation. | Order-two local expansion and leaf discretization. |

Compare methods against a suitable independent reference for the selected density and trajectory. Werner and the other four methods intentionally model different densities.

## GPU execution and readback

```text
main-world source
  -> ExtractSchedule
  -> GPU buffers and body-frame request uniform
  -> compute dispatch and workgroup reduction
  -> staging-buffer mapping
  -> snapshot-tagged gravity history
  -> fixed-step physics
```

Readback never blocks the render thread. Snapshot epoch, request ID, time, transform, position, and velocity prevent mixing fields with the wrong simulation state.

## Physics and frame pacing

Physics runs at 60 fixed updates per second. Each displayed update contains 12 leapfrog substeps; `Nx` acceleration advances `N` complete fixed updates without enlarging the integration step.

Radial and Equation (106) histories provide bounded interpolation/prediction. Werner, MMFFT, and FMM hold the newest completed field. Integration pauses during warm-up or after a numerical/certification failure.

## Rotating-frame Jacobi-constant chart

The chart displays

$$
C_J=2U(\mathbf r_b)
+\|\boldsymbol\omega_b\times\mathbf r_b\|^2
-\|\mathbf v_b\|^2.
$$

Every sample uses one completed request's position, velocity, attitude, and potential. The chart holds 256 samples, reports `dC/|C0|`, and adds an 8% vertical margin around visible extrema.

## Performance comparison

The top-center button opens a five-method workspace. Each enabled method receives a 120-frame measurement window with FPS and Jacobi histories. Results include the browser, driver, rendering, dispatch, and readback overhead; they are not solver-only or cross-machine benchmarks.

Headless WASM compilation timing:

```sh
uv run python scripts/wasmtime_benchmark.py --wasm pkg/ryugu_wasm_bg.wasm \
  --calls 8 --iterations 100000 --json
```

## Surface topology and normals

After loading the GLTF model, the application normalizes scale, welds vertices at `1e-4`, builds CSR adjacency, and runs a one-shot WebGPU normal pass. Press `F` to display the result.

## Runtime controls

| Input | Action |
|---|---|
| `S` | Toggle overview/probe camera. |
| `F` | Toggle surface normals. |
| `D` | Toggle density section. |
| `G` | Cycle the five gravity methods. |
| `X`, `Y`, `Z` | Set initial probe position. |
| `Speed` | Set circular-speed multiplier. |
| Acceleration | Select `1x`–`8x` complete fixed updates per frame. |
| Mouse drag / wheel | Orbit / zoom camera. |

Changing initial conditions or gravity method resets the trajectory and waits for a valid field sample.

## Physical constants

| Constant | Value |
|---|---:|
| `G` | `6.6743e-11` |
| `RYUGU_MASS` | `4.5e11 kg` |
| `CASSINI_MASS` | `2500 kg` |
| `DENSITY_EPSILON` | `10 m` |
| `RYUGU_ROTATION_PERIOD_SECS` | `7.63 h` |
| `TIME_SCALE` | `500` |
| `PHYSICS_SUBSTEPS` | `12` |
| `MAX_SIMULATION_ACCELERATION` | `8` |
| `PROBE_SPEED_FACTOR` | `1.053` |

Default probe state:

```text
position = (-1000, 1200, 100)
speed    = 1.053 * sqrt(G * RYUGU_MASS / |position|)
```

## Scheduling

### Startup

```text
scene and UI setup -> model load -> source initialization
```

### Main-world update chain

```text
model normalization -> topology -> controls/UI -> visualization
```

GPU results are polled in `PreUpdate`; immutable sources are built in `Update`.

### Fixed update

```text
physics -> Ryugu rotation -> Jacobi diagnostics
```

### Render world

```text
extraction -> compute dispatch -> asynchronous mapping
```

## Project structure

```text
Ryugu_wasm/
├── src/
│   ├── lib.rs
│   ├── components.rs
│   └── systems/
│       ├── gravity/
│       ├── gpu/
│       ├── model/
│       ├── presentation/
│       └── simulation/
├── assets/{models,shaders}/
├── docs/
├── scripts/
├── tests/
├── index.html
├── server.ts
└── Cargo.toml
```

### File responsibilities

| Path | Responsibility |
|---|---|
| `src/components.rs` | Constants, ECS state, gravity sources, and histories. |
| `src/systems/gravity/` | Radial, Werner, Equation (106), MMFFT, and FMM implementations. |
| `src/systems/presentation/` | Scene, controls, density section, and charts. |
| `src/systems/simulation/` | Fixed-step physics and diagnostics. |
| `assets/shaders/` | WebGPU compute shaders. |
| `tests/` | Independent numerical checks. |
| `docs/` | Mathematical derivations and convergence conditions. |

## Testing

Rust tests cover density integration, acceleration/potential consistency, Werner and multipole far fields, M2L acceptance, Equation (106) transforms and convergence, interpolation, Jacobi evaluation, buffer layouts, and WGSL validation. Python tests independently check the logarithmic density and convergence guards.

## Known limitations

- Radial sources require a star-shaped body.
- Werner is homogeneous; the other methods use logarithmic density.
- Equation (106), MMFFT, and FMM use finite discretizations.
- FMM local expansions are order two; M2L requires a full source-plus-target radius below `0.20×distance` and a `0.5%` node-field certificate.
- GPU readback and prediction remain asynchronous numerical approximations.
- The simulator uses f32 GPU arithmetic and is not an orbit-determination tool.
- WebGPU is required.

## Mathematical coverage audit

| Document result | Runtime status |
|---|---|
| Eq. (79) toroidal identity | Certified segmented Chebyshev table used as a cross-check. |
| Eqs. (81)–(86) density separation | Approximated by the mass-preserving `4×8×16` source tensor. |
| Eqs. (89)–(95) NUFFT | Common-frequency Bromwich summation; no explicit general NUFFT matrix. |
| Eq. (106) straight-line field | Implemented for the discrete density representation on CPU and GPU. |
| Eqs. (109)–(110) inversion | Finite 129-frequency and half-line quadrature. |
| Eqs. (155)–(158) | Adaptive Taylor guard and dual residual implemented. |

The implementation is a tested, discretized Equation (106) evaluator, not an exact untruncated continuous-density identity.

## Deployment

`.github/workflows/deploy.yml` builds the release WASM package and deploys `index.html`, `assets/`, and `pkg/` to GitHub Pages.

## License

MIT; see [LICENSE](LICENSE).
