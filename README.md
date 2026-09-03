# RyuGu WASM

[![Live demo](https://img.shields.io/badge/Live_demo-WebGPU-success)](https://tom-jim.github.io/RyuGu_WASM/)
[![Bevy](https://img.shields.io/badge/Bevy-0.19.1-purple)](https://bevy.org/)
[![Rust](https://img.shields.io/badge/Rust-2024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

RyuGu WASM is a research prototype for evaluating gravity around asteroid
(162173) Ryugu and propagating a probe trajectory in the browser. Rust and
Bevy own the simulation; WebGPU/WGSL owns the parallel field kernels; the
browser layer provides controls, charts, and WebAssembly startup.

This is a numerical demonstrator, not flight software or evidence of mission
suitability. Results depend on discretization, GPU precision, interpolation,
truncation, and the selected validation gates.

## Methods

| Method | Implementation boundary |
| --- | --- |
| Radial analytic | Mass-preserving radial layers and GPU Gauss–Legendre evaluation. |
| Werner polyhedron | Homogeneous closed polyhedron evaluation from oriented mesh topology. |
| Frequency-domain algorithm | Finite reciprocal-space quadrature, spectral evaluation, and asynchronous GPU readback implementing the Eq. (184) path. |
| MMFFT compressed | CPU zero-padded FFT preparation followed by GPU packed-f16 potential interpolation. |
| FMM | CPU source/tree preparation and GPU target-cell expansion plus exact near-field P2P. |

Eq. (184) is implemented as a finite numerical realization, not as an
unbounded exact transform. CPU f64 reference calculations are separate from
GPU f32 results so the comparison does not validate a method against its own
approximation.

## Code layout

```text
src/
├── lib.rs                    Bevy app, startup, schedules, WASM exports
├── bevy/                     ECS adapters, scene setup, rendering, UI snapshots
├── interface/                Shared resources, requests, histories, contracts
├── cpu/                      Source preparation, physics, planning, inversion,
│                             f64 reference and benchmark helpers
├── gpu/                      Render-world compute pipelines and readback paths
└── wgsl/                     WebGPU shaders and runtime shader validation

assets/
├── models/                   Ryugu and probe geometry
├── operators/                Transform/operator lookup tables
└── shaders/                  Auxiliary presentation shaders
```

The dependency direction is intentionally one-way:

```text
mesh → shared source contract → CPU preparation / GPU extraction
     → WGSL dispatch → async readback → Bevy state → JSON snapshot → UI
```

`src/interface/` is the boundary between numerical code and presentation. It
contains snapshot identities, capture IDs, workload identities, metric rows,
and history buffers, preventing stale GPU packets or mismatched source meshes
from being silently compared.

`src/bevy/` schedules ECS systems and owns the visible scene. Gizmos are
budgeted presentation primitives: live trajectories are capped to a fixed
point count, normals are sampled, and density overlays use line segments
instead of one high-resolution sphere entity per sample. Radial keeps the
yellow detector marker, its live trajectory, and section view; only the
non-yellow initial knot markers are omitted.

The frequency-domain algorithm animates the captured known trajectory while
its whole-trajectory transform is evaluated. It is a prescribed-path
visualization, not an instantaneous-force substitute for the pointwise
integrators.

`src/cpu/` prepares mass-preserving sources, integrates the probe, assembles
planning references, and performs independent f64 checks. It may prepare FFT,
FMM, and inversion data, but it does not replace a failed GPU method with a
different physical model.

`src/gpu/` contains Bevy RenderApp extraction, bind-group layouts, dispatch,
timestamp collection, and readback decoding. Frequency-domain nodes use an
explicit `vec4<f32>(kx, ky, kz, weight)` storage layout shared by Rust and
WGSL. Shader variants are checked by the runtime shader frontend tests.

`src/wgsl/` contains the parallel kernels. Storage-array parameter blocks are
used for batched work; output packets carry snapshot identity and diagnostic
fields so the main world can reject stale results.

Arrow Up/Right zoom the Bevy camera in; Arrow Down/Left zoom it out. These
keys never resize or translate the HTML overlay. Pointer wheel, right-drag,
and pinch gestures remain available for the display surface.

## Fair comparison rules

The comparison UI reports preparation, warm evaluation, readback, accuracy,
and certified timing separately. Methods use common source/target workloads
where their mathematical scope permits it. Werner remains homogeneous while
the heterogeneous methods use the shared logarithmic source profile.

Frequency-domain, MMFFT, and FMM planning rows share target counts, density
model counts, repeats, source-size sweeps, and f64 reference observations.
Failed accuracy or incomplete workload gates remain visible as failures; they
are not converted into attractive timing numbers.

The benchmark is an end-to-end browser workload, not a claim about asymptotic
complexity. GPU preprocessing, cache rebuilds, readback, and CPU integration
are part of the reported path. A warm cached result is meaningful only when
the same source identity, trajectory, and target workload are retained.

## Build and deployment

```sh
bun install
bun run build
bun run server.ts       # optional local static server
```

GitHub Pages deployment is defined in `.github/workflows/deploy.yml`. The
generated `pkg/` directory is loaded by `src/html/index.html`; deployment
injects a cache-busting version so HTML, JS, and WASM stay aligned.

The repository does not require a browser or preview server for source-level
shader and contract tests. WebGPU behavior still depends on the browser
adapter and device limits available at runtime.

## Page-open telemetry

`src/html/visit.js` sends one event per tab session only when the exact
canonical URL `https://tom-jim.github.io/RyuGu_WASM/` is opened. It does not
run for localhost, previews, README images, asset requests, or query-string
variants.

The event contains a browser-generated random visitor ID and an ISO-8601 UTC
timestamp. This is an anonymous pseudonymous identifier, not a name, email,
IP address, or claim of real-world identity. The server may record request
time and user-agent according to its privacy policy.

GitHub Pages is static and cannot persist `/api/visit` by itself. Production
counting therefore requires a privacy-reviewed endpoint at
`/RyuGu_WASM/api/visit` (for example, a separately managed Worker or
serverless function). Until that endpoint is configured, the hook fails
silently and no counter is claimed. The old HITS image-request workflow was
removed.

## Current limitations

- GPU arithmetic is primarily f32 and readback is asynchronous.
- Frequency-domain evaluation is finite-band and finite-quadrature; validity
  depends on segment guards and reference checks.
- MMFFT includes CPU FFT preprocessing, finite grids, interpolation, and
  packed-f16 quantization.
- FMM uses a fixed order/depth configuration and needs external convergence
  studies for broader claims.
- Density inversion is regularized and non-unique.
- The radial source model assumes a star-shaped body representation.

## License

MIT. See [LICENSE](LICENSE).
