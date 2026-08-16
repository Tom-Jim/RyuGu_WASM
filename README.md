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
- two-level 3D MMFFT with zero-padded FFT/kernel/IFFT convolution;
- order-two FMM with P2M, M2M, M2L, L2L, and exact leaf P2P.

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
- Trajectory-to-density inversion from 16 editable position/velocity controls, a continuous Quintic Hermite track, and simulated annealing over the occupied 3D density voxels.
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

Detailed derivations and convergence conditions are in [`mathpub.pdf`](mathpub.pdf).

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

The GPU uses a mass-preserving `4×8×32` radial/polar/azimuth source quadrature, density modes `m=0..16`, and a 129-frequency Bromwich grid. Curved arcs are planned in the body frame. A segment is accepted only when

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

## Trajectory Density Inversion

After five seconds of simulation, the 3D view exposes two editable columns of
16 uniformly resampled detector states: position on the left and velocity on
the right. The `Invert trajectory` button uses those states as **Quintic
Hermite control nodes**, not as only 16 independent gravity observations.

For the default 16 nodes, the 15 Hermite intervals are sampled at 16 points
per interval, with shared endpoints, giving

$$
15\times16+1=241
$$

trajectory positions and inertial-acceleration observations, or 723 scalar
acceleration components. The displayed magenta inversion trajectory and the
inverse solver use the same Hermite position/acceleration evaluator. Knot
accelerations come from non-uniform three-point quadratic differentiation of
the editable velocity controls, including second-order endpoint stencils.

The asteroid is voxelized into a `4×4×4` Cartesian grid; only cells that
intersect the radial source are retained (56 cells for the bundled Ryugu
model). Every retained voxel has an independent positive density. Simulated
annealing starts from a total-mass-preserving uniform density and minimizes a
trajectory-data term together with total-mass, neighbor-smoothness, and weak
uniform-prior terms. Proposals include smooth zero-mass radial modes for fast
large-scale convergence and mass-conserving exchanges across adjacent voxel
faces for three-dimensional structure.

The original forward density is never used as the annealing initial state or
as an inverse observation. It is retained only after optimization for the
displayed validation metrics:

- Werner is evaluated against its uniform-density reference.
- Radial, Equation (106), MMFFT, and FMM are evaluated against the
  volume-averaged outward-increasing logarithmic reference density.

The upper-right inversion panel reports fit (`1 -` volume-weighted relative
density RMSE), density RMSE, annealing objective improvement, the number of
Quintic track samples, voxel count, density range, density spread, mass scale,
objective value, and annealing iterations. A zero objective improvement is
reported explicitly as remaining at the uniform start; it is not displayed as
a successful non-uniform recovery.

When inversion is active, the section visualization follows the rotating
asteroid and shows the recovered density field rather than the forward-model
field. Dynamic contours are drawn over each section: the exterior outline is
always present, while interior contours appear for recoverable density levels.

## Comparative scope

| Method | Strength | Main approximation |
|---|---|---|
| Werner | Closed homogeneous polyhedron reference. | No heterogeneous density. |
| Radial | Reusable star-shaped angular/radial source. | Four radial layers and angular discretization. |
| Equation (106) | Structured curved-trajectory spectral reuse. | Finite source/frequency quadrature and Taylor segmentation. |
| MMFFT | Two-level 3D zero-padded FFT/kernel/IFFT convolution. | Finite Cartesian mesh spacing and interpolation. |
| FMM | Hierarchical P2M/M2M/M2L/L2L plus exact leaf P2P. | Order-two source/local expansions. |

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

Every sample uses the CPU-integrated position and velocity. Eq.106 evaluates the matching conservative local potential
`U_loc = U0 + g0·dx + 1/2 dx^T H dx`, so its displayed force and potential obey `g = grad(U)` within each cached segment. The chart holds 256 samples, reports `dC/|C0|`, and adds an 8% vertical margin around visible extrema.

## Performance comparison

The top-center button opens a five-method workspace. Each enabled method receives a 120-frame measurement window normalized to 1x simulation acceleration; the previous acceleration is restored on exit. Jacobi curves plot per-method relative drift against each sample's actual simulation time, without visual offsets or a synthetic 100-hour axis. Eq.106 samples retain segment id, line origin, local `(h,u,v)`, and all four runtime certificates. Results include the browser, driver, rendering, dispatch, and readback overhead; they are not solver-only or cross-machine benchmarks.

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
| `Invert trajectory` | Start density inversion from the 16 displayed position/velocity controls after their five-second capture is complete. |
| Position / Velocity rows | Click a row, enter `x, y, z`, and press Enter to replace one Hermite control value. |
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
| `src/systems/simulation/` | Fixed-step physics, diagnostics, and continuous-trajectory 3D density inversion. |
| `assets/shaders/` | WebGPU compute shaders. |
| `tests/` | Independent numerical checks. |
| `docs/` | Mathematical derivations and convergence conditions. |

## Testing

Rust tests cover density integration, acceleration/potential consistency, Werner and multipole far fields, M2L acceptance, Equation (106) transforms and convergence, interpolation, Jacobi evaluation, buffer layouts, WGSL validation, and the inversion contract: the original density cannot seed annealing, dense Quintic sampling is used instead of cached gravity, and voxel sensitivities remain spatially distinct. Python tests independently check the logarithmic density and convergence guards.

## Known limitations

- Radial sources require a star-shaped body.
- Werner is homogeneous; the other methods use logarithmic density.
- Equation (106), MMFFT, and FMM use finite discretizations.
- FMM local expansions are order two; M2L requires the full source-plus-target radius below `0.10×distance`, while non-separated leaves use exact P2P.
- Density inversion is a regularized, trajectory-constrained reconstruction. A single external track cannot uniquely resolve every unconstrained interior mode; the mass and smoothness terms select a stable 3D solution in the remaining null space.
- The inversion sensitivity operator uses the voxelized Newton kernel, with MMFFT's finite-grid softening represented in its inverse operator. It is a reconstruction benchmark for comparing the runtime methods, not a calibrated spacecraft-navigation estimator.
- GPU readback and prediction remain asynchronous numerical approximations.
- The simulator uses f32 GPU arithmetic and is not an orbit-determination tool.
- WebGPU is required.

## Mathematical coverage audit

| Document result | Runtime status |
|---|---|
| Eq. (79) toroidal identity | Certified segmented Chebyshev table used by the independent potential representation. |
| Eqs. (81)–(86) density separation | Density Fourier modes `m=0..16` from the mass-preserving `4×8×32` tensor. |
| Eqs. (89)–(95) NUFFT | Common-frequency Bromwich summation; no explicit general NUFFT matrix. |
| Eq. (106) straight-line field | Implemented for the discrete density representation on CPU and GPU. |
| Eqs. (109)–(110) inversion | Finite 129-frequency and half-line quadrature. |
| Eq. (118) curved translation | Planner-selected directional Taylor jet through order eight. |
| Eqs. (155)–(158) | Adaptive Taylor guard and dual residual implemented. |

The implementation is a tested, discretized Equation (106) evaluator, not an exact untruncated continuous-density identity.

## Deployment

`.github/workflows/deploy.yml` builds the release WASM package and deploys `index.html`, `assets/`, and `pkg/` to GitHub Pages.

## License

MIT; see [LICENSE](LICENSE).
