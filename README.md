# Ryugu WASM

[![Live Demo](https://img.shields.io/badge/Live_Demo-WebGPU-success?style=for-the-badge)](https://tom-jim.github.io/RyuGu_WASM/)
![Bevy](https://img.shields.io/badge/Bevy-0.19.1-purple.svg)
![Rust](https://img.shields.io/badge/Rust-Edition_2024-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An experimental browser-based simulator for gravitational-field evaluation and spacecraft trajectories around asteroid (162173) Ryugu. The project is written in Rust with Bevy, compiled to WebAssembly, and uses WebGPU for field evaluation.

The repository compares five forward models and contains a research implementation of the near-straight-trajectory formulation called **Equation (106)** in [`mathpub.pdf`](mathpub.pdf). Equation (106) is still an exploratory numerical method: the current implementation is discretized, finite-band, and validated only within the tests and diagnostics described below.

> **Scope:** research prototype and synthetic-data demonstrator. It is not flight software, an orbit-determination product, or evidence of a proven performance advantage over established solvers.

[Open the live WebGPU demo](https://tom-jim.github.io/RyuGu_WASM/)

<img src="https://github.com/user-attachments/assets/e6f0435d-14c5-4218-aa1d-cd0ebd26684c" width="100%" alt="algorithm comparison" />

| **Orbital Trajectory** | **ProbeView** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/ac05a5ac-e6e2-4f44-b0ca-8a447ba30b7f" width="100%" alt="orbital trajectory" /> | <img src="https://github.com/user-attachments/assets/f58a18c3-161b-4945-9078-bcfa835c2ed4" width="100%" alt="probe view" /> |
| **Change Orbit** | **Change Algorithm** |
| <img src="https://github.com/user-attachments/assets/7921e222-c7c4-4758-8dfe-82575efdeeb5" width="100%" alt="change orbit" /> | <img src="https://github.com/user-attachments/assets/6ac1d9aa-2a2f-4744-b8f0-47dd4f11b352" width="100%" alt="change gravity algorithm" /> |

## What is implemented

The simulator exposes five switchable gravity methods. Four share the mass-normalized heterogeneous profile
$\rho(r)=C\ln(1+r/10\,\mathrm m)$; Werner-Scheeres is the homogeneous reference.

```mermaid
flowchart LR
    Mesh["Ryugu mesh"] --> Sources["Mass-preserving sources"]
    Sources --> Radial["GPU Radial Analytic"]
    Mesh --> Werner["GPU Werner Polyhedron"]
    Sources --> Eq106["Eq.106 Adaptive Curved-Arc"]
    Sources --> FFT["CPU MMFFT + packed-f16 GPU interpolation"]
    Sources --> Tree["GPU order-2 target-cell FMM"]
    Radial --> Compare["Common snapshots / error checks"]
    Werner --> Compare
    Eq106 --> Compare
    FFT --> Compare
    Tree --> Compare
```

| Method | Preparation | Runtime kernel | Scope |
|---|---|---|---|
| GPU Radial Analytic | Four equal-volume layers; analytic layer mass | 8-node Gauss-Legendre quadrature | Heterogeneous direct reference |
| GPU Werner Polyhedron | Oriented faces, edges, dyads | Closed homogeneous polyhedron formula | Homogeneous only; invalid topology records are reported/skipped |
| Eq.106 Adaptive Curved-Arc | Shared `4 × 8 × 32 = 1024` source tensor, tables, segments | Cached spectra, Bromwich inversion, acceleration/potential/Jacobian | Experimental near-straight segment reuse |
| Packed MMFFT | CPU zero-padded Newton convolution on `64³` + `16³` grids | GPU tricubic sampling/differentiation of scale-normalized packed-f16 potential | 8x potential-buffer compression; CPU preprocessing + GPU interpolation |
| Order-2 target-cell FMM | Source P2M/M2M; cached target-leaf M2L coefficients | Reused L2P local expansion plus exact leaf P2P | Shared target boxes remove the former per-target all-node scan |

| UI method | Source preparation | Runtime evaluation | Main qualification |
|---|---|---|---|
| **GPU Radial Analytic** | The star-shaped mesh is divided into four equal-volume radial layers per angular cell; layer masses are integrated analytically. | WebGPU evaluates the field with eight-node Gauss-Legendre radial quadrature. | A direct heterogeneous reference. The mass integration is analytic, but the field evaluation is quadrature rather than a closed-form solver. |
| **GPU Werner Polyhedron** | CPU constructs oriented faces, shared edges, and geometric dyads. | WebGPU evaluates the homogeneous closed-polyhedron formula. | Homogeneous only; unusable boundary or non-manifold edge records are skipped and reported during preprocessing. |
| **Eq.106 Adaptive Curved-Arc** | The shared `4 x 8 x 32 = 1024` source aggregation, special-function tables, and trajectory segments are prepared. | WebGPU builds and caches transformed line spectra, then evaluates acceleration, potential, and a local Jacobian. | Experimental hybrid realization of Eq.106; most useful when many samples reuse a geometrically guarded near-straight segment. |
| **Common source discretization** | Runtime uses the shared `1024` aggregation; planning refines it mass- and centroid-preservingly from `32K` through `8192K`. | Radial, Eq.106, packed MMFFT, and FMM consume the same point set at each comparison size. | Werner remains a separate homogeneous closed-polyhedron reference. |
| **Packed MMFFT + GPU Interpolation** | CPU performs a zero-padded Newton-kernel FFT convolution on two grids (`64^3` and `16^3`). | WebGPU unpacks two normalized binary16 potentials per word, samples them tricubically, and differentiates the same interpolant. | The GPU hierarchy uses 2 rather than 16 bytes per grid point. This remains CPU FFT preprocessing plus GPU interpolation, not an end-to-end GPU FFT. |
| **GPU Order-2 Target-Cell FMM** | CPU performs source P2M/M2M and translates accepted source moments into one analytic local potential/gradient/Hessian per occupied target leaf. | States in one target leaf reuse that expansion through L2P; only non-separated leaves execute exact P2P. | This is a source/target-cell FMM hot path; its fixed order/depth and strict acceptance still require external error certification. |

## Mathematical core of Equation (106)

```mermaid
flowchart TD
    Newton["Newton volume integral"] --> Line["Straight reference line\nF(s;a,z') and endpoint term"]
    Line --> Kernels["Dimensionless kernels\nK_V, K_H"]
    Kernels --> Coeff["Transverse coefficients"]
    Coeff --> Spectrum["129 signed complex-frequency bins"]
    Spectrum --> Invert["Finite-band Bromwich sum"]
    Invert --> Taylor["Adaptive transverse Taylor\norder 1..8 (3..45 coeffs)"]
    Taylor --> Field["g, U, Jacobian, certificates"]
```

### 1. Newtonian starting point

For source point $\mathbf p$, observation point $\mathbf q$, and $R=\lVert\mathbf p-\mathbf q\rVert$, the positive gravitational potential and acceleration convention used by the project is

$$
U(\mathbf q)=G\iiint_V\frac{\rho(\mathbf p)}{R}\,dV',
\qquad
\mathbf g(\mathbf q)=G\iiint_V\rho(\mathbf p)
\frac{\mathbf p-\mathbf q}{R^3}\,dV'.
$$

The volume density is represented in source-centered spherical coordinates, so

$$
dV'=\lambda^2\sin\theta'\,d\lambda\,d\theta'\,d\phi'.
$$

### 2. Straight-reference-line transform

Around a chosen reference line, write the transverse source-line distance as $a$ and the signed longitudinal source coordinate as $z'$. For $\mathrm{Re}(s)>0$, define

$$
F(s;a,z')=\int_0^\infty
\frac{e^{-sh}}{\sqrt{a^2+(h-z')^2}}\,dh.
$$

The boundary identity

$$
\frac{\partial F}{\partial z'}=-sF+\frac{1}{R_0},
\qquad R_0=\sqrt{a^2+z'^2},
$$

is essential: the $1/R_0$ term is the endpoint contribution from the half-line and must not be discarded.

Introduce

$$
x=as,\qquad \eta=\frac{z'}a,\qquad \Psi(x,\eta)=F(s;a,z'),
$$

and the two dimensionless kernels

$$
K_V=x\Psi-\frac{1}{\sqrt{1+\eta^2}},
$$

$$
K_H=x\frac{\partial\Psi}{\partial x}
+\eta x\Psi-\frac{\eta}{\sqrt{1+\eta^2}}.
$$

Express the source point in the local frame of the reference line as $(r'_\perp,\phi',z')$. With $\Delta\phi=\phi-\phi'$, its transverse separation from the line is

$$
a^2=\varrho^2+r_\perp'^2-2\varrho r'_\perp\cos\Delta\phi
$$

and the transformed straight-line acceleration used as the Eq.106 reference coefficient is

$$
\widetilde{\mathbf g}(s)=G\iiint
\rho(\lambda,\theta',\phi')\lambda^2\sin\theta'
\left[
\mathbf e_\varrho
\frac{\varrho-r'_\perp\cos\Delta\phi}{a^2}K_H
+\mathbf e_\phi
\frac{r'_\perp\sin\Delta\phi}{a^2}K_H
+\mathbf e_z\frac{1}{a}K_V
\right]
\,d\lambda\,d\theta'\,d\phi'.
$$

In the mathematical note this expression first appears as Eq.70 and is later carried into the formulation labelled Eq.106. Apparent singular factors at $a\to0$ require the dedicated axis continuation implemented in the shader; arbitrary softening is not part of this derivation.

### 3. Curved-trajectory continuation

For a trajectory written as a reference-line point plus a transverse offset,

$$
\mathbf q(t)=\bar{\mathbf q}(h(t))+\delta\mathbf q(t),
$$

the field is locally continued with the translation operator

$$
\mathbf g(\bar{\mathbf q}+\delta\mathbf q)=\sum_{n=0}^{A}\frac{1}{n!}\left(\delta\mathbf q\cdot\nabla\right)^n\mathbf g(\bar{\mathbf q})+\mathbf R_{A+1}.
$$

The production path uses a bivariate transverse Taylor polynomial with an adaptive order from **1 through 8** (`3` through `45` coefficients). The planner monitors a conservative distance ratio $\varepsilon$ and a geometric-series remainder estimate, selecting the lowest order that satisfies the configured remainder target. A segment must be rebuilt or rejected when its curvature or source proximity exceeds the configured guard.

### 4. Code realization and reuse contract

| Stage | Operation | Reuse boundary |
|---|---|---|
| Sample | `64` half-line nodes traverse the discrete source tensor | Rebuilt only for a new segment |
| Transform | Newton Taylor jet → `129` signed complex bins | Cached per spectral element |
| Analytic correction | Replace zeroth coefficient with Eq.70/Eq.106 kernel from $\Psi,\partial_x\Psi$ tables | Shared tables |
| Inversion | Finite-band Bromwich reconstruction | Reused for all targets on the segment |
| Curved arc | Order `1..8` transverse polynomial | Guarded by curvature/distance certificates |
| Output | Acceleration, potential, Jacobian, diagnostics | Async snapshot-tagged readback |

This is a hybrid numerical realization, not an exact closed form for every curved-trajectory coefficient. The intended advantage is repeated target evaluation on a valid cached segment; a new segment still pays spectrum construction.

## Runtime architecture

### Software architecture and dependency direction

The Rust code is organized around one-way dependencies. Bevy owns scheduling,
render extraction, assets, and presentation; numerical backends do not reach
back into Bevy systems. Both CPU and GPU paths exchange the same snapshot and
field-sample contract through `src/interface/`.

```mermaid
flowchart LR
    Bevy["Bevy adapter\nsrc/bevy/\nECS + RenderApp"] --> Interface["Unified interface\nsrc/interface/\nrequest / response / history"]
    Interface --> CPU["CPU backends\nsrc/cpu/\nplanning, source prep, integration"]
    Interface --> GPU["GPU backends\nsrc/gpu/\nrender-world compute plugins"]
    CPU --> CpuBenchmark["CPU benchmark\nsrc/cpu/benchmark.rs\ndeterministic scalar kernels"]
    GPU --> GpuTests["GPU tests\nsrc/gpu/tests.rs\nWGSL validation"]
    GPU --> GpuBenchmark["GPU performance metrics\nshared interface resources\nportable timing fields"]
```

The four top-level layers are intentionally directional:

```text
src/bevy/      Bevy scheduling, render extraction, scene, UI, and diagnostics
src/interface/ shared resources, snapshots, histories, and method selection
src/cpu/       trajectory planning, source preparation, integration, inversion,
               Eq.106 reference operators, and CPU benchmark entry points
src/gpu/       WebGPU compute backends, shader validation tests, and portable
               performance metric collection
```

There are no compatibility redirect modules in the runtime. Imports point
directly from Bevy adapters to the shared interface and then to CPU or GPU
implementations. The Eq.106 dispatch path is split into named preparation,
trajectory-batch, sensitivity-matrix, single-target, and layout stages; no
scheduling order, buffer contract, or physics fallback behavior changes.

The benchmark entry point is exported from `src/cpu/benchmark.rs` rather than
the Bevy application entry point. This keeps host/WASM benchmark tooling
independent from window creation while preserving the public
`benchmark_gravity_algorithms(iterations)` export.

```mermaid
flowchart LR
    Mesh["Ryugu GLB mesh"] --> Topology["Normalize mesh<br/>weld vertices<br/>build topology"]
    Topology --> RadialSource["Mass-preserving<br/>radial layers"]
    Topology --> WernerSource["Werner faces<br/>and shared edges"]

    RadialSource --> RadialGPU["Radial quadrature<br/>WebGPU"]
    RadialSource --> EqSource["Eq.106<br/>4 x 8 x 32 tensor"]
    RadialSource --> MMFFTBuild["Two-level FFT grids<br/>CPU preprocessing"]
    RadialSource --> FMMBuild["Octree + multipoles<br/>CPU preprocessing"]

    EqSource --> EqGPU["Spectrum build/cache<br/>Bromwich evaluation"]
    MMFFTBuild --> MMFFTGPU["Tricubic field sampling<br/>WebGPU"]
    FMMBuild --> FMMGPU["Tree traversal<br/>WebGPU"]
    WernerSource --> WernerGPU["Polyhedron field<br/>WebGPU"]

    Clock["Simulation clock<br/>probe request"] --> Select["Selected gravity method"]
    RadialGPU --> Select
    WernerGPU --> Select
    EqGPU --> Select
    MMFFTGPU --> Select
    FMMGPU --> Select

    Select --> Readback["Snapshot-tagged<br/>asynchronous readback"]
    Readback --> Volterra["Eq.106: h-domain<br/>Volterra/Picard waveform"]
    Readback --> Verlet["Other methods: 100-step<br/>velocity-Verlet integration"]
    Volterra --> State["Probe transform<br/>trajectory history"]
    Verlet --> State
    State --> UI["3D view, charts,<br/>Jacobi and inversion UI"]
    State --> Clock
```

The browser has no automatic CPU fallback for the real-time gravity path. If a valid WebGPU evaluator or matching field sample is unavailable, trajectory advancement pauses rather than silently switching algorithms. The Eq.106 path closes position, local field, longitudinal speed, time, and transverse motion with damped Picard sweeps. It rejects longitudinal turning points, non-convergence, and exits from the certified Taylor tube; the other four field models retain the established velocity-Verlet path.

## Equation (106) segment pipeline

```mermaid
flowchart TD
    History["Recent trajectory history"] --> Planner["Adaptive line/arc planner<br/>curvature + distance guards"]
    Source["Mass-preserving source tensor"] --> Samples["64 half-line samples<br/>direct Newton Taylor jet"]
    Planner --> Samples
    Samples --> Numerical["Numerical Laplace spectra<br/>for transverse coefficients"]
    Tables["Precomputed Psi and Psix tables"] --> Analytic["Analytical Eq.70/Eq.106<br/>reference-line coefficient"]
    Numerical --> Merge["Cached 129-bin<br/>complex spectrum"]
    Analytic --> Merge
    Merge --> Inverse["Finite-band<br/>Bromwich inversion"]
    Planner --> Offset["Transverse offset<br/>order <= 4"]
    Offset --> Taylor["Bivariate Taylor correction"]
    Inverse --> Taylor
    Taylor --> Output["Acceleration + potential<br/>Jacobian + certificates"]
    Source --> Toroidal["Fourier-toroidal potential<br/>m = 0...16"]
    Toroidal --> Residual["Eq.157 dual-representation<br/>residual diagnostic"]
    Output --> Residual
    Output --> Readback["GPU readback -> CPU physics"]
```

## Diagnostics and density inversion

### Rotating-frame Jacobi diagnostic

For body-frame position $\mathbf r$, velocity $\mathbf v_{\rm rot}$, spin $\boldsymbol\omega$, and positive potential $U$, the displayed quantity is

$$
C_J=2U+\lVert\boldsymbol\omega\times\mathbf r\rVert^2
-\lVert\mathbf v_{\rm rot}\rVert^2.
$$

Its relative drift is a consistency diagnostic for the coupled force, potential, frame transformation, readback, and time integrator. A flat curve does not by itself prove that a gravity model is physically correct; a drift can come from field approximation, stale GPU data, interpolation, or integration error.

### Dual-representation residual

```mermaid
flowchart LR
    EqField["Eq.106 curve-integrated potential"] --> Residual["Residual"]
    Toroidal["Fourier-toroidal potential\nm = 0..16"] --> Residual
    Residual --> Meaning["Numerical disagreement diagnostic\nnot an independent physical measurement"]
```

### Synthetic density inversion

```mermaid
flowchart TD
    Knots["16 editable knots"] --> Track["Quintic-Hermite track\n241 shared samples + capture_id"]
    Source["786432 radial records"] --> Truth["Independent f64 order-2 tree\nopening 0.025"]
    Track --> Truth
    Sources1024["1024 aggregated sources"] --> Voxels["56 occupied voxels\n4³ grid, unit-density bases"]
    Voxels --> Matrix["Common response matrix A\n723 acceleration components"]
    Track --> Matrix
    Truth --> Obs["Frozen observations\ncache by capture + source hash"]
    Obs --> QP["Clarabel 56×56 QP\nmass equality + bounds + smoothness"]
    Matrix --> QP
    QP --> Holdout["Offset holdout validation"]
```

| Contract | Implementation |
|---|---|
| Shared input | Same capture, source geometry, voxel ownership, target array, and sample count for Eq.106/MMFFT/FMM |
| Eq.106 matrix | 56 columns in one command encoder; acceleration-only readback; diagnostics disabled |
| MMFFT matrix | Real CIC deposition, `64³`/`16³` zero-padded Newton FFT, cached plans/spectra/workspaces |
| Tree matrix | CPU `f64` quadrupole treecode with the distributed shared basis (distinct from runtime target-cell FMM) |
| Noise model | Diagonal covariance: `0.1%` relative noise plus absolute floor; three seeded Gaussian solves averaged |
| History/cache | Latest and best fit plus cold/warm matrix, Clarabel, verification, and total times; keys include capture/source/basis/config hashes |
| Scope | Radial and Werner are forward-only; fit is a regularized synthetic score, not a posterior or uniqueness claim |

The frozen capture, source hash, voxel basis, and target array are shared by every inverse backend; changing a knot or source invalidates all rows. Eq.106 builds 56 acceleration-only columns in one encoder, MMFFT uses real CIC deposition with cached `64³`/`16³` FFT plans and spectra, and the third row is a CPU `f64` quadrupole treecode. Clarabel solves the shared `56 × 56` constrained QP (mass equality, bounds, smoothness, weak prior) under diagonal `0.1%` noise; three seeded solves are averaged and checked on a separate offset holdout. Radial and Werner remain forward-only. History separates cold/warm matrix, solve, verification, and total times; cache keys include capture/source/basis/config hashes. Release builds use `opt-level = 3`, LTO, and one codegen unit. GPU hot paths use 64-lane reductions, and status telemetry reports preparation, paced GPU wall/copy/map, assembly, solve, verification, dispatches, rebuilds, and cache hits. Dawn/Metal timestamp queries remain disabled.

The reported `fit` is a synthetic volume-weighted density score against the known reference model. It is not a posterior probability, confidence level, or real-observation accuracy. One external trajectory cannot uniquely recover an arbitrary three-dimensional density field without assumptions and regularization.

### Near-synchronous robust pericenter planning

The probe controls retain the existing 620 m benchmark orbit and add a `Near-sync ellipse` preset. The preset starts at apocenter with position `(-1097.269, 51.622, 0) m`, uses Ryugu's spin axis as the orbit normal, and derives velocity from the existing circular-speed multiplier with `speed_factor = 0.82408`; the displayed velocity is therefore not an independently hard-coded state. The nominal two-body orbit has period `27495.468 s`, semimajor axis `831.624 m`, eccentricity `0.320889`, pericenter radius `564.765 m`, and apocenter radius `1098.483 m`.

```mermaid
flowchart LR
    Capture["Immutable quintic capture"] --> Jets["Nominal-density f64\nfield + symmetric gradient"]
    Capture --> Initial["Perturbed initial states\ntrust limit 15 m"]
    Jets --> Volterra["Damped Volterra/Picard\ntrajectory propagation"]
    Initial --> Volterra
    Volterra --> First["First\n32 × 4 × 241"]
    Volterra --> Stress["Interactive Stress\n2048 × 32 × 512"]
    First --> Validate["f64 direct validation\nall candidates"]
    Stress --> Stratified["Deterministic stratified validation"]
    Validate --> Compare["Shared hashes + coverage"]
    Stratified --> Compare
    Compare --> Methods["Eq.106 | packed MMFFT | target-cell FMM"]
```

The planning contract is separate from inversion. Every point stores body-fixed state, time, rotation, candidate/sample identity, and transverse distance. Candidate curves are no longer drawn independently at every sample: 70% of the 15 m tube defines perturbed initial positions, reserving the remaining radius for differential-force drift, and each complete curve is propagated under the nominal density model's `f64` reference field plus symmetric position Jacobian. The propagator uses the same coupled `h`-domain Volterra/Picard equations as the live Eq.106 path and rejects curves that leave the full trust tube. Eq.106 then selects Taylor order `1..8`, caps elements at `300 s`, and runs line sampling, spectrum assembly, analytic correction, and target evaluation as four batched stages.

These are dynamically consistent nominal-model benchmark curves, not certified flyable trajectories: thrust, delta-v, navigation uncertainty, and model-specific repropagation for every density row are not enforced. All methods deliberately evaluate the same propagated states so the forward-kernel timing comparison remains paired. The selectable workloads are:

| Profile | Candidate trajectories | Density models | Samples per candidate | Evaluations |
|---|---:|---:|---:|---:|
| First | 32 | 4 | 241 | 30,848 |
| Interactive Stress | 2,048 | 32 | 512 | 33,554,432 |

```mermaid
xychart-beta
    title "Planning workload (evaluation count)"
    x-axis [First, Interactive-Stress]
    y-axis "Evaluations" 0 --> 33554432
    bar [30848, 33554432]
```

Source-resolution experiments use the same mass-preserving spatial refinement for every method:

```mermaid
flowchart LR
    S1["32K"] --> S2["64K"] --> S3["128K"] --> S4["256K"] --> S5["512K"] --> S6["1024K"] --> S7["2048K"] --> S8["4096K"] --> S9["8192K"]
    S9 --> Plot["Raw + certified time\nEq.106 | packed MMFFT | FMM"]
```

The upper-right comparison panel keeps `Density fit` and `Inversion time` and adds gravity error, gradient error, propagated pericenter error, minimum altitude, reference-model separation, the planning objective, Eq.106 segment count, speedup versus the FMM baseline, and cold-start amortization. The quadrature curve reports raw and independently certified series for all three methods; certification repeats the complete hot workload and checks the common deterministic strata against the same direct `f64` source sum. The `K x 56` density rows contain genuinely different center, shell, lobe, quadrupole, and rubble patterns and are normalized to the same total mass. Planning results are accepted only when Eq.106, packed MMFFT, and FMM carry the same nonzero capture, source geometry, voxel basis, candidate-state, density-model, and sample-array hashes; use the selected `B x K x H` workload; and identify their actual backends.

| Fairness gate | Required condition |
|---|---|
| Coverage | Same nonzero fully covered candidates; contiguous Eq.106 segments |
| Accuracy | Gravity `≤ 1e-3`; gradient `≤ 1e-2`; pericenter error `≤ 1 m` |
| Segments | Target `≤ 10`; hard maximum `16`; each element `≤ 300 s` |
| Verdict | Eq.106 advantage `≥3×` faster than both baselines; strong at `≥5×` |
| Timing | Include preprocessing; report cold-start amortization separately |

The comparison covers this repository's fixed configurations only; it is not a claim against a tuned FMM or fully GPU-resident MMFFT. Inversion, fit history, and density visualization remain independent.

`First` and `Stress` are two sizes of the same planning structure and the same three forward algorithms. First uses fixed candidate tiles for repeatable comparisons; Stress adapts its tile budget and yields when a rendered frame exceeds the interactive budget. Dynamically propagated candidate positions and all structured density rows are uploaded once as immutable shared GPU buffers. Eq.106 runs its line-sample, spectrum-assembly, and field-evaluation passes; packed MMFFT samples its real two-level convolution hierarchy; FMM evaluates cached target-cell local expansions and exact near interactions. A common GPU reduction emits one compact metric row per candidate, while only deterministic verification rows are copied to mapped staging.

A probe collision still resets the live flight scene after the three-second crash notice, but it does not reset the independent frozen planning job, its selected metric, or completed First/Interactive-Stress rows. Those rows remain tied to their original workload and capture hashes; starting a later planning run creates a new identity normally.

Common trajectory/density preparation is reported separately and excluded from every method total. Method CPU payload preprocessing, command submission, paced GPU wall/copy/map, CPU reduction, request counts, tile range, and kernel-evaluation count are reported separately. Independent verification accumulates field and gradient in `f64`; it is timed separately. Eq.106 reuses its spectral partition, packed MMFFT reuses RustFFT plans/kernel spectra/workspaces without caching per-source heap stencils, and FMM reuses source/target topology across density rows. Accuracy is checked against the exact source count selected by the curve, not a hidden 1024-point surrogate.

## Runtime controls

| Input | Action |
|---|---|
| `S` | Toggle overview and probe cameras. |
| `F` | Toggle computed surface normals. |
| `D` | Toggle the density section view. |
| `G` | Cycle through the five gravity methods. |
| `Invert trajectory` | Run the synthetic inversion after trajectory capture is complete. |
| Trajectory point panel | Scroll through the 16 point/velocity pairs; replace one vector with `x, y, z`. |
| `X`, `Y`, `Z`, `Speed` | Change the initial probe state. |
| `First` / `Stress` / `Quadrature` | Run the compact, extended, or source-crossover planning workload; the UI reports truthful completion and verification progress. |
| Simulation acceleration | Execute `1x`-`8x` complete fixed updates per rendered frame. |
| Panel heading drag / lower-right resize | Reposition or resize a work panel; the initial layout never overlaps panels. |
| Left-drag / wheel on uncovered scene | Orbit and zoom the Bevy camera. HTML controls only capture input over their own visible bounds. |

Changing the initial conditions or gravity method resets the trajectory and waits for a field sample associated with the new request.

Closing the Quadrature page, selecting a density/inversion metric, or starting trajectory inversion cancels the active planning generation. No later planning GPU dispatch is submitted; an already submitted WebGPU command cannot be interrupted by the browser and is only allowed to drain. Native CPU candidate propagation uses a bounded `crossbeam-channel` worker queue and commits results in deterministic candidate order. The default WASM target keeps that same preparation phase cooperative, one bounded slice per frame, because browser worker/atomics support is deployment-specific.

## Build and run

### Requirements

- Rust with the `wasm32-unknown-unknown` target
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/)
- [Bun](https://bun.sh/)
- a current WebGPU-capable browser with hardware acceleration

```sh
rustup target add wasm32-unknown-unknown
bun install
bun run dev       # Tailwind + debug WASM build, then http://localhost:3000
bun run build     # regenerate Tailwind and create the release WASM package in pkg/
bun run styles    # regenerate src/html/tailwind.generated.css
bun run preview   # Tailwind + release build, then local server
bun run serve     # serve an existing build
```

The development server supplies the cross-origin isolation headers used by the application. GitHub Pages deployment is defined in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).

### Validation commands

```sh
cargo fmt --all --check
cargo test
cargo clippy --target wasm32-unknown-unknown --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown
node --check src/html/ui.js
node --check src/html/app.js
bun run styles
wasm-pack build --target web --out-name ryugu_wasm
```

Optional Python checks and the Wasmtime benchmark use `uv`:

```sh
uv sync
uv run pytest -q
uv run python scripts/wasmtime_benchmark.py --wasm pkg/ryugu_wasm_bg.wasm
```

The Rust tests cover density integration, shader parsing, operator tables, transform/inverse consistency, planner guards, radial/Werner/MMFFT/FMM reference cases, physics sampling, Jacobi evaluation, and inversion invariants. Passing unit tests establishes consistency with the tested discretization; it does not establish Eq.106 convergence over every body, trajectory, or parameter range.

## Repository map

```mermaid
flowchart TB
    Root["RyuGu_WASM/"] --> Lib["src/lib.rs<br/>application setup and schedules"]
    Root --> Bevy["src/bevy/<br/>scene, UI, charts, scaling"]
    Root --> Interface["src/interface/<br/>shared data and contracts"]
    Root --> CPU["src/cpu/<br/>planning, integration, inversion"]
    Root --> GPU["src/gpu/<br/>WebGPU evaluators and tests"]
    CPU --> CpuBench["benchmark.rs"]
    GPU --> GpuTests["tests.rs<br/>WGSL validation"]
    GPU --> GpuBench["benchmark.rs<br/>timestamp metadata"]
    CPU --> Eq106Cpu["eq106_reference.rs<br/>eq106_operator.rs<br/>curved_arc.rs"]
    GPU --> Eq106Gpu["eq106.rs"]
    GPU --> OtherGpu["radial.rs<br/>werner.rs<br/>mmfft.rs<br/>fmm.rs<br/>normals.rs"]
    Root --> Shaders["src/wgsl/"]
    Shaders --> EqShader["eq106_complex.wgsl"]
    Shaders --> OtherShaders["gravity.wgsl<br/>werner_gravity.wgsl<br/>mmfft_compressed.wgsl<br/>fmm_gravity.wgsl"]
    Root --> Operators["assets/operators/<br/>Eq.106 special-function tables"]
    Root --> Models["assets/models/<br/>bundled Ryugu and probe meshes"]
    Root --> Math["mathpub.pdf<br/>mathematical derivation"]
    Root --> Tests["tests/<br/>Python numerical checks"]
    Root --> Workflow[".github/workflows/deploy.yml<br/>GitHub Pages deployment"]
```

Key implementation entry points:

| File | Responsibility |
|---|---|
| [`src/lib.rs`](src/lib.rs) | Bevy plugins, startup systems, update ordering, and WebGPU availability checks. |
| [`src/bevy/`](src/bevy/) | Bevy scheduling, scene setup, presentation, UI, and diagnostic charts. |
| [`src/interface/`](src/interface/) | Shared snapshots, histories, resources, and method-selection helpers. |
| [`src/cpu/curved_arc.rs`](src/cpu/curved_arc.rs) | Eq.106 source discretization, trajectory planner, Fourier modes, and geometric guards. |
| [`src/cpu/volterra.rs`](src/cpu/volterra.rs) | `f64` reference-line Volterra/Picard propagation, endpoint matching, prefix integration, and convergence guards. |
| [`src/gpu/eq106.rs`](src/gpu/eq106.rs) | Eq.106 buffers, render-world dispatch, readback, and history management. |
| [`src/wgsl/eq106_complex.wgsl`](src/wgsl/eq106_complex.wgsl) | Half-line sampling, complex spectrum assembly, analytical reference coefficient, Bromwich reconstruction, Taylor correction, and diagnostics. |
| [`src/cpu/physics.rs`](src/cpu/physics.rs) | CPU trajectory integration using snapshot-matched GPU field results. |
| [`src/cpu/inversion.rs`](src/cpu/inversion.rs) | Quintic track construction and regularized synthetic voxel inversion. |
| [`mathpub.pdf`](mathpub.pdf) | Full exploratory derivation and stated convergence conditions. |

## Benchmark interpretation

The in-app comparison reports end-to-end rendered frame throughput while each method participates in its normal preprocessing, dispatch, readback, physics, and presentation path. It is useful for interactive regression testing, but it is not a solver-only benchmark and should not be used to claim asymptotic superiority.

A publishable comparison should separately measure:

- preprocessing and cache-build time;
- warm cached-query latency and throughput;
- GPU readback and CPU integration overhead;
- acceleration and potential error against a high-accuracy common reference;
- Jacobi drift at matched time step and precision;
- memory use and rebuild frequency as curvature, altitude, and source resolution vary.

The expected Eq.106 advantage, if confirmed, is narrow but testable: a fixed body and density discretization, a long near-straight exterior track, and enough repeated samples per valid segment to amortize spectrum construction. FMM or grid methods may be preferable for arbitrary three-dimensional queries, rapidly changing trajectories, or broad field-volume evaluation.

## Known limitations

- The radial source model assumes the body is star-shaped with respect to its chosen center.
- The Werner solver is homogeneous, whereas the other displayed methods use the logarithmic heterogeneous profile.
- Eq.106 uses finite source, half-line, frequency, Fourier, and Taylor truncations; the $a\to0$ axis branch and special-function table domain require separate guards.
- Strongly curved or near-surface trajectories can force frequent Eq.106 segment rebuilds and remove its reuse advantage.
- Stress requests batch 8-16 candidate curves and one density model at a time. Eq.106 still encodes the four mathematical stages once per spectral element inside that request; a future two-dimensional element dispatch and multi-density special-function reuse require dedicated numerical and browser validation before they can replace this conservative path.
- Planning candidates are self-consistent under the nominal density's linearized reference field, but every randomized density row still shares that trajectory. A true density-orbit joint planning solve requires model-specific GPU-resident Picard propagation or an adjoint formulation.
- The code does not implement the general Type-3/Type-2 NUFFT construction discussed in the mathematical note.
- MMFFT accuracy is limited by finite grids, binary16 quantization, interpolation, and CPU-side FFT preprocessing; the certified curve exposes those errors.
- The target-cell FMM uses fixed depth, strict acceptance, and order-two local expansions. It is a complete source-to-local/P2P pipeline for this fixed configuration, not a claim of parity with a tuned high-order adaptive FMM library.
- GPU arithmetic is primarily `f32`, and GPU readback is asynchronous.
- The density inversion is regularized and non-unique. Eq.106 and MMFFT use method-consistent unit-voxel forward responses, while the shared frozen trajectory and `capture_id` contract prevent cross-capture comparisons.
- Numerical agreement in this repository does not establish novelty, general convergence, or mission readiness. Those require literature review, independent derivation review, convergence studies, and reproducible external benchmarks.

Eq.106 implementation note: Taylor order is selected per certified segment from 1 through 8. Planning line samples, spectrum assembly, analytic correction, and target evaluation use one batched dispatch per stage with a storage-array parameter block and isolated per-segment spectral slices; this removes the former per-element bind-group and dispatch serialization while preserving the existing output and direct-verification contract.

## License

MIT. See [`LICENSE`](LICENSE).
