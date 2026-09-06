# Ryugu Dynamics Laboratory

[![CI/CD](https://github.com/Tom-jim/RyuGu_WASM/actions/workflows/deploy.yml/badge.svg)](https://github.com/Tom-jim/RyuGu_WASM/actions/workflows/deploy.yml)
[![Live demo](https://img.shields.io/badge/Live_demo-WebGPU-success)](https://tom-jim.github.io/RyuGu_WASM/)
[![Bevy](https://img.shields.io/badge/Bevy-0.19.1-purple)](https://bevy.org/)
[![Rust](https://img.shields.io/badge/Rust-2024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**A browser-based planetary-science workbench for investigating how an irregular asteroid's shape and internal density affect spacecraft motion, surface gravity, and the information recoverable from an observation arc.**

Ryugu Dynamics Laboratory combines a rotating Ryugu shape model, heterogeneous gravity models, live probe propagation, surface diagnostics, and constrained density inversion. Its central research direction is a **frequency-domain compression algorithm** that separates the asteroid's density representation from the spacecraft's trajectory representation, allowing repeated forward calculations and inverse problems to reuse numerical work.

Rust and Bevy manage physical state and visualization; WebGPU/WGSL performs parallel numerical evaluation; an HTML interface with Vue/ECharts exposes the experiments. The **Frequency-domain** control selects the frequency-domain compression algorithm. The **Radial** and **FFT** controls select separate solvers.

The public mathematical companion is [mathpub.md](mathpub.md). It is the sole linked mathematical reference in this README. The formulas below are self-contained and use descriptive names rather than document-specific equation numbers.

## Planetary-science questions

The project is organized around four connected questions:

| Scientific question | What the workbench provides | Interpretation |
| --- | --- | --- |
| How does interior heterogeneity change motion near an irregular asteroid? | Method selection, adjustable probe initial conditions, rotating-body gravity, live orbit history, and numerical diagnostics | Controlled synthetic experiments with a specified shape and density model |
| Where do rotation and local shape change the surface environment? | Effective gravity, gravity-gradient magnitude, effective slope, triangle normals, and density sections | Diagnostics relevant to proximity operations and surface-force interpretation |
| Which observation arcs distinguish competing interior models? | Trajectory editing and planning comparisons, including altitude, pericenter, model separation, and gradient-related metrics | Experimental tools for studying geometry and information content, rather than an operational mission optimizer |
| What density structure can an arc constrain? | Frozen-trajectory forward operators, sensitivity calculations, finite-dimensional density fitting, mass constraints, and regularization | Conditional inference within the selected basis; a good fit does not establish a unique physical interior |

Ryugu supplies the shape and physical scale for these experiments. The configured mass is $4.5\times10^{11}\,\mathrm{kg}$ and the rotation period is $7.63\,\mathrm{h}$. The spacecraft visualization uses a Cassini asset, while its motion is generated from the experiment's initial conditions. It is not a reconstruction of a historical Cassini encounter with Ryugu. The default density profile is a synthetic hypothesis, not a measured map of Ryugu's interior.

## Research contribution and positioning

Frequency-domain computation has a long-established role in terrestrial gravity and magnetic exploration. Parker-style potential-field calculations, interface inversion, and spectral or low-rank separation exploit repeated structure in sources and observation geometry. Their usefulness in geophysics does not imply that the same compression transfers unchanged to a spacecraft moving around a small, rotating, irregular body.

In small-body applications, polyhedral, mascon, and harmonic representations are more established reference approaches. **Trajectory-aware frequency-domain compression is a less conventional direction in that setting.** This is a qualitative research positioning, not a claim that spectral asteroid models are absent or that this project is the first to use them.

The project's distinctive aim is to connect three operations within one inspectable experiment:

1. **Generate a physically coupled observation arc.** Evaluate the extended asteroid's gravity with an independent propagation path, advance the probe, and retain its positions, velocities, times, and body attitudes.
2. **Separate source structure from trajectory sampling.** Combine a reusable density spectrum with a characteristic of the captured trajectory to produce whole-arc observations.
3. **Reuse that operator for interior inference.** Freeze the observed arc and apply the same discretized observation definition to density hypotheses and sensitivity columns.

The intended advantage is amortization across long arcs, many density models, and repeated inverse iterations. Neither the use of Fourier transforms nor moving a kernel to the GPU alone establishes a new algorithm or a speed advantage. Accuracy, memory, preprocessing cost, and reuse count must be measured together.

## Physical model

For asteroid volume $V$, density $\rho(\mathbf p)$, and an exterior field point $\mathbf q$, the model uses the positive Newtonian potential convention:

$$
U(\mathbf q)=G_N\int_V\frac{\rho(\mathbf p)}{\lVert\mathbf q-\mathbf p\rVert}\,dV_{\mathbf p},
\qquad
\mathbf g(\mathbf q)=\nabla_{\mathbf q}U(\mathbf q)
=G_N\int_V\rho(\mathbf p)\frac{\mathbf p-\mathbf q}{\lVert\mathbf p-\mathbf q\rVert^3}\,dV_{\mathbf p}.
$$

Here $G_N$ is Newton's gravitational constant. The source is an extended, shaped body. Density is stationary in the body frame during a forward calculation; body attitude connects body-frame fields and observations to inertial spacecraft motion.

Radial, FFT, FMM, and the frequency-domain compression algorithm default to the mass-normalized logarithmic radial profile

$$
\rho(r)=C\ln\left(1+\frac{r}{\varepsilon}\right),
\qquad
C=\frac{M}{\displaystyle\int_V\ln\left(1+\frac{\lVert\mathbf p\rVert}{\varepsilon}\right)dV_{\mathbf p}},
$$

where $M$ is the configured asteroid mass and $\varepsilon>0$ is the density-profile length scale. Werner uses a homogeneous closed polyhedron normalized to the same mass. Equal total mass does not make these density distributions equivalent: a heterogeneous-versus-Werner difference includes a physical-model difference as well as discretization effects.

The current heterogeneous geometry is a finite star-shaped angular-cell and radial-shell representation. More general geometries supported by the mathematical formulation require additional source representations; they are not automatically supported by the current shell implementation.

## The frequency-domain compression algorithm

The algorithm has two complementary mathematical forms: a **reference-line gravity response** and a **whole-trajectory observation response**. They share the same Newtonian density model, but their outputs have different meanings.

### Reference-line gravity response

Choose a body-frame reference half-line

$$
\overline{\mathbf q}(h)=\mathbf q_0+h\mathbf e_z,
\qquad h\ge0,
\qquad
\inf_{h\ge0}\mathrm{dist}(\overline{\mathbf q}(h),V)>0.
$$

For the cylindrical component form, take $\mathbf q_0=\varrho\mathbf e_\varrho$ at azimuth $\phi$. Write a source point in spherical coordinates $(\lambda,\theta',\phi')$ and define

$$
\begin{aligned}
r'_\perp&=\lambda\sin\theta', & z'&=\lambda\cos\theta', & \Delta\phi&=\phi-\phi',\\
a^2&=\varrho^2+(r'_\perp)^2-2\varrho r'_\perp\cos\Delta\phi,
& x&=a s_h, & \eta&=z'/a.
\end{aligned}
$$

The spatial Laplace parameter satisfies $\mathrm{Re}\,s_h>0.$ Define the scalar kernel and its derivatives by

$$
\begin{aligned}
\Psi(x,\eta)&=\int_0^\infty\frac{e^{-xu}}{\sqrt{1+(u-\eta)^2}}\,du,\\
K_H(x,\eta)&=x\,\partial_x\Psi-\eta\,\partial_\eta\Psi,\\
K_V(x,\eta)&=-\partial_\eta\Psi
=x\Psi-\frac{1}{\sqrt{1+\eta^2}},\\
A_H&=\frac{\varrho-r'_\perp\cos\Delta\phi}{a^2},\qquad
B_H=\frac{r'_\perp\sin\Delta\phi}{a^2},\qquad
A_V=\frac1a.
\end{aligned}
$$

The complete three-component reference-line response is

$$
\boxed{
\begin{aligned}
\widetilde{\mathbf g}_{\mathrm{line}}(s_h)
&:=\int_0^\infty e^{-s_hh}\mathbf g(\overline{\mathbf q}(h))\,dh\\
&=G_N\iiint_V\rho(\lambda,\theta',\phi')\lambda^2\sin\theta'\\
&\quad\times\left[
\mathbf e_\varrho A_HK_H(x,\eta)
+\mathbf e_\phi B_HK_H(x,\eta)
+\mathbf e_z A_VK_V(x,\eta)
\right]\,d\lambda\,d\theta'\,d\phi'.
\end{aligned}
}
$$

This follows by taking a one-sided Laplace transform of the Newton kernel along the reference line and differentiating its scalar transform. The $a\to0$ case requires the appropriate analytic limit; the apparent singularities in the component representation are not a license to alter the physical kernel with arbitrary smoothing.

Pointwise gravity is recovered through an inverse transform:

$$
\mathbf g(\overline{\mathbf q}(h))
=\frac{e^{\sigma_h h}}{2\pi}
\int_{-\infty}^{\infty}
\widetilde{\mathbf g}_{\mathrm{line}}(\sigma_h+i\omega_h)e^{i\omega_h h}\,d\omega_h,
\qquad \sigma_h>0.
$$

**Why this form helps.** It turns a succession of source-to-target evaluations along a reference line into a family of density-linear frequency responses. For long, smooth flyby segments and repeated density hypotheses, a sufficiently compact set of common responses could replace a much larger stored pointwise operator. Recovering gravity is still necessary before that response can drive an ordinary orbit integrator. A curved orbit needs updated geometry, controlled local corrections, or a general trajectory representation; it is not obtained by replaying a straight reference line.

### Whole-trajectory observation response

Use the spatial Fourier convention

$$
\widehat\rho(\boldsymbol\kappa)
=\int_V\rho(\mathbf p)e^{-i\boldsymbol\kappa\cdot\mathbf p}\,dV_{\mathbf p},
\qquad \kappa^2=\boldsymbol\kappa\cdot\boldsymbol\kappa.
$$

For a body-frame observation arc $\boldsymbol\gamma(t)$, $0\le t\le T$, define

$$
\mathcal T_\gamma(s_t,\boldsymbol\kappa)
=\int_0^T e^{-s_t t+i\boldsymbol\kappa\cdot\boldsymbol\gamma(t)}\,dt.
$$

Here $s_t$ is a temporal Laplace parameter, whereas $s_h$ above is conjugate to distance. Their units differ. The finite observation window is part of the operator definition.

To display the full density-dependent formula, partition the body into cells $K_a$, with local origins $\mathbf p_a$ and piecewise polynomial density

$$
P_a(\mathbf x)=\sum_{|\boldsymbol\alpha|\le p}c_{a\boldsymbol\alpha}\mathbf x^{\boldsymbol\alpha},
\qquad \mathbf x=\mathbf p-\mathbf p_a.
$$

Define the cell's exponential generating function and coefficient-extraction form:

$$
\begin{aligned}
E_a(\mathbf z)&=\int_{K_a-\mathbf p_a}e^{\mathbf z\cdot\mathbf x}\,dV_{\mathbf x},\\
\Omega_a(\boldsymbol\zeta)&=
\sum_{|\boldsymbol\alpha|\le p}c_{a\boldsymbol\alpha}\boldsymbol\alpha!
\frac{d\zeta_1\wedge d\zeta_2\wedge d\zeta_3}
{\zeta_1^{\alpha_1+1}\zeta_2^{\alpha_2+1}\zeta_3^{\alpha_3+1}}.
\end{aligned}
$$

The multi-index factorial is $\boldsymbol\alpha!=\alpha_1!\alpha_2!\alpha_3!$. The residue extracts the polynomial-weighted cell spectrum exactly for that finite density model:

$$
\mathop{\mathrm{Res}}\limits_{\boldsymbol\zeta=\mathbf0}
\left[E_a(-i\boldsymbol\kappa+\boldsymbol\zeta)\Omega_a(\boldsymbol\zeta)\right]
=\int_{K_a-\mathbf p_a}P_a(\mathbf x)e^{-i\boldsymbol\kappa\cdot\mathbf x}\,dV_{\mathbf x}.
$$

The complete whole-trajectory response is then

$$
\boxed{
\begin{aligned}
\widetilde{\mathbf g}_\gamma(s_t)
&:=\int_0^T e^{-s_t t}\mathbf g(\boldsymbol\gamma(t))\,dt\\
&=\frac{G_N}{(2\pi)^3}\int_{\mathbb R^3}
\frac{4\pi i\boldsymbol\kappa}{\kappa^2}
\mathcal T_\gamma(s_t,\boldsymbol\kappa)\\
&\quad\times\sum_{a=1}^{N_K}e^{-i\boldsymbol\kappa\cdot\mathbf p_a}
\mathop{\mathrm{Res}}\limits_{\boldsymbol\zeta=\mathbf0}
\left[E_a(-i\boldsymbol\kappa+\boldsymbol\zeta)\Omega_a(\boldsymbol\zeta)\right]
\,d^3\boldsymbol\kappa.
\end{aligned}
}
$$

Equivalently, the cell sum is $\widehat\rho(\boldsymbol\kappa)$: the integrand is the **Newton multiplier × density spectrum × trajectory characteristic**. The Fourier integral is understood with the convergence or regularization needed for the continuous potential; the implementation uses a finite quadrature whose truncation must be assessed separately.

**Why this form helps.** A fixed asteroid model can reuse its density spectrum across arcs. A fixed observed arc can reuse its trajectory characteristic across density hypotheses. Density-basis spectra can also be reused in repeated sensitivity and inverse calculations. General curved trajectories enter through $\mathcal T_\gamma$, without requiring one global straight-line approximation.

This output is a transformed observation of the entire arc, with units of acceleration multiplied by time. It is not instantaneous acceleration. A known-trajectory density operator is linear in density; simultaneously solving for an unknown trajectory and density remains nonlinear because the trajectory itself depends on gravity. Body-frame vector observations also require the appropriate attitude transformation when compared with inertial or instrument-frame data.

### How the derivation creates reusable structure

The reference-line and whole-trajectory forms belong to the same frequency-domain compression algorithm. Their connection follows from the Fourier representation of the Newton kernel:

$$
\widetilde{\mathbf g}_{\mathrm{line}}(s_h)
=\frac{G_N}{(2\pi)^3}\int_{\mathbb R^3}
\frac{4\pi i\boldsymbol\kappa}{\kappa^2}
\frac{\widehat\rho(\boldsymbol\kappa)e^{i\boldsymbol\kappa\cdot\mathbf q_0}}
{s_h-i\boldsymbol\kappa\cdot\mathbf e_z}\,d^3\boldsymbol\kappa.
$$

The reference line has a rational propagation factor. Replacing that factor with the characteristic of a finite, curved arc gives the general observation form. For a constant-velocity segment, the characteristic can itself be evaluated explicitly:

$$
\mathcal T_\gamma(s_t,\boldsymbol\kappa)
=e^{i\boldsymbol\kappa\cdot\mathbf q_0}
\frac{1-e^{-[s_t-i\boldsymbol\kappa\cdot\mathbf v]T}}
{s_t-i\boldsymbol\kappa\cdot\mathbf v}.
$$

Several additional reductions motivate the research roadmap:

| Derivation step | Computational opportunity | Condition or limitation |
| --- | --- | --- |
| Integration by parts and Gauss' theorem | Replace piecewise constant volume contributions with density-jump interface integrals | Variable density retains a volume density-gradient term; the outer surface alone is insufficient |
| Polynomial cell generating functions and multivariate residues | Compute density moments and spectra without repeatedly performing the same volume quadrature | Exactness applies to the chosen polynomial approximation; stable confluent limits are needed near coincident poles |
| Residue integration over a longitudinal wave number | Reduce a three-dimensional spectral integral to two dimensions | Requires a suitable one-sided source/target geometry; global orbit coverage needs local charts or the general three-dimensional form |
| Local Parker-style terrain expansions | Reuse FFTs for repeated interface or terrain calculations | A global asteroid is not a single height map; near-surface, high-wave-number terms can converge slowly |
| Angular Fourier and trajectory Chebyshev representations | Reduce the number of independent modes along a smooth arc | Compression weakens near sources, across sharp turns, and at thrust discontinuities; segmenting the arc is necessary |

These are mathematical opportunities, not a statement that all branches already run in the browser.

### Current implementation and suitable workloads

The implemented data flow is:

**Mass-normalized volume cells → independent WGSL propagation field → numerical orbit integration → captured body-frame arc → frequency-domain compression algorithm observations and sensitivities → constrained density fit.**

- **Propagation:** the dedicated WGSL field kernel evaluates the spatial Newton field equivalent to the inverse-transformed reference-line response. It uses four-point radial Gauss quadrature within angular-shell cells. It does not numerically invert a stored Laplace response at every frame, and it does not use the truncated observation spectrum as the orbit force. Its acceleration history is separate from Radial, FFT, FMM, and Werner.
- **Time integration:** the shared leapfrog-style substeps consume the selected evaluator's timestamped acceleration history, with bounded interpolation/extrapolation and pauses when usable readbacks are unavailable. Force latency remains a numerical error source; substepping alone does not guarantee the accuracy of freshly evaluated forces at every substep.
- **Observation:** the captured trajectory supplies positions, physical times, and attitudes. The current transform uses 64 reciprocal wave-number nodes, positive real Laplace samples, and composite-trapezoid time weights. Source spectra and time contributions are reduced in GPU workgroups.
- **Inversion:** the current finite model uses 56 occupied density voxels. Forward observations and unit-density sensitivity columns are evaluated in the same discrete observation space. The runtime volume-cell spectrum and the planning/inversion basis representations have distinct layouts; agreement still requires discretization checks.
- **Reuse:** density, trajectory-capture, and basis identifiers govern caches and asynchronous results. Changed inputs must invalidate the corresponding products.

The analytic polynomial-cell residue expansion, adaptive two-dimensional reductions, and general low-rank trajectory representation are extension paths. The current finite quadrature is an experimental realization of the separated operator, not a complete implementation of every analytic reduction above.

For $N_t$ time samples, $N_\rho$ density coefficients, $B$ arc segments, and $K_s$ retained responses per segment, a dense three-component pointwise operator needs $O(3N_tN_\rho)$ real entries. A reference-response representation would need $O(3BK_sN_\rho)$ complex entries, plus reconstruction data. The memory opportunity is substantial only when $BK_s\ll N_t$ at the required scientific accuracy; complex storage and preprocessing must be counted. This comparison is against a stored dense operator, not the memory cost of a matrix-free FMM.

The strongest candidate applications are repeated interior-model screening, long smooth flyby arcs, repeated forward/transpose operations in inversion, and observation-design sweeps over related trajectories. A single field point, a short arbitrary arc, or demanding near-contact evaluation may favor a simpler direct, polyhedral, FFT, or hierarchical method. No universal speedup is claimed.

## Interior inference and observation design

Freezing the observed trajectory separates a linear density-estimation subproblem from the nonlinear task of generating that trajectory. For coefficients $\mathbf a$, observation vector $\mathbf y$, and a fixed-arc operator $A_\gamma$, a representative objective is

$$
\min_{\mathbf a}\;
\frac12\lVert A_\gamma\mathbf a-\mathbf y\rVert_W^2
+\frac\lambda2\lVert L(\mathbf a-\mathbf a_0)\rVert_2^2,
\qquad
C\mathbf a=\mathbf d,\quad
\rho_{\min}\le\rho(\mathbf a)\le\rho_{\max}.
$$

Here $W$ weights observations, $L$ regularizes the estimate toward a prior $\mathbf a_0$, and $C\mathbf a=\mathbf d$ represents constraints such as known total mass. The present solver applies its configured mass and density constraints; additional moment constraints are extensions of this formulation.

A unique regularized finite-dimensional solution requires suitable positive curvature on feasible directions. It does not imply that one external orbit uniquely determines an arbitrary three-dimensional density. For planetary interpretation, future comparisons should ask which density contrasts are resolvable under realistic noise, how results change with the basis and prior, and whether additional altitudes or viewing geometries reduce ambiguity. Synthetic fitting is currently the principal use case; operational radio-science interpretation also needs tracking observables, instrument effects, and nongravitational-force models.

## Solvers and surface products

| UI selection | Representation and role | Main accuracy questions |
| --- | --- | --- |
| **Frequency-domain** | Independent volume-field propagation followed by the frequency-domain compression algorithm for whole-arc observations and sensitivities | Volume quadrature, spectral bandwidth, temporal sampling, and forward/basis consistency |
| **Radial** | Angular cells with analytic radial endpoint contributions and local numerical stabilization | Angular resolution, shell approximation, and near-alignment conditioning |
| **Werner** | Homogeneous closed-polyhedron edge and face evaluation | Mesh closure, orientation, near-surface conditioning, and the uniform-density assumption |
| **FFT** | Cartesian source deposition, spectral convolution, and field interpolation | Grid resolution, domain extent, kernel discretization, and interpolation |
| **FMM** | Hierarchical source aggregation with near/far evaluation | Multipole truncation, opening criteria, tree resolution, and near-field treatment |

Select a method and press **Calculate field** for effective gravity, gradient magnitude, or effective slope. Surface products are computed in bounded CPU batches and uploaded to a display overlay; they are separate from the live GPU orbit evaluator. In particular, the frequency-domain surface product uses a finite spectral field approximation, so its convergence must be checked separately from the propagation field.

**Section** shows the selected method's default density distribution without changing the density model or launching a surface calculation. **Normals** shows one outward normal per triangle. The orbit trail retains up to 100,000 integrated history positions; it displays the path already travelled.

**Relative error** compares effective-gravity magnitudes on shared surface patches, using comparison minus baseline divided by the baseline magnitude, with a small denominator floor. A positive result means a larger value than the selected baseline, not necessarily overestimation of physical truth. Homogeneous/heterogeneous comparisons are model contrasts unless the physical inputs are matched. Effective slope includes centrifugal acceleration and local normals; interpreting it as regolith stability also requires material strength and contact physics.

## Why Rust, WebAssembly, and WGSL?

The stack is chosen to keep a scientific experiment interactive while making its numerical data flow explicit.

| Layer | Why it fits this project | Tradeoff |
| --- | --- | --- |
| **Rust** | Typed physical state, ownership of buffers and caches, reusable CPU reference operators, and explicit handling of asynchronous GPU results | Memory safety does not prove numerical correctness; generations, frames, units, and GPU layouts still need validation |
| **WebAssembly** | Distributes the compiled Rust model through a web page, reducing setup for teaching, collaboration, and exploratory experiments | Browser memory limits, suspension, startup cost, and readback latency differ from a native scientific application |
| **WebGPU / WGSL** | Exposes parallel source, wave-number, time, target, face, and density-basis work; shared-memory reductions and reusable buffers suit repeated linear operators | GPU f32 arithmetic, adapter limits, synchronization, and transfer costs constrain accuracy and workload size |
| **Bevy** | Keeps body attitude, probe state, scene geometry, camera, and diagnostics in one scheduled application | Rendering competes with computation for frame time and GPU resources |
| **HTML / Vue / ECharts** | Provides accessible controls, resizable panels, scientific charts, and mobile interaction around the simulation | UI scheduling and scientific timing must remain separate |

GPU acceleration is concentrated where contributions can be evaluated independently and reduced. The CPU retains orchestration, constrained optimization, reference calculations, and current surface-product evaluation. The goal is to minimize repeated work and transfers, not to move every small operation onto the GPU. Matching Rust buffer layouts and WGSL storage layouts is part of correctness, as is rejecting stale readbacks.

## Using the workbench

1. Choose a gravity method and set probe position and speed. Inspect the live orbit and method-specific diagnostics.
2. Use **Section**, **Normals**, and the surface controls to relate the field to the body geometry.
3. In frequency-domain mode, allow an observation arc to be captured before using **Invert density**. The trajectory editor and planning controls support further experiments with the sampled arc.
4. Use the planning metrics and **First**, **Stress**, or **Quadrature** workloads to examine accuracy, density-model counts, target counts, and preprocessing amortization.
5. Inspect the performance page for frame-rate and diagnostic histories, keeping numerical accuracy separate from display speed.

The left rail holds initial conditions, surface analysis, and trajectory editing; the center shows the scene; the right rail holds inversion, planning, and long-run controls. Panels can be dragged, resized, and scrolled. **Reset UI view** restores panel positions and overall UI scale.

### Mobile interaction

The default **Bevy camera** gesture mode rotates with one finger and pans or pinch-zooms the camera with two fingers. **Whole UI** mode scales and pans the complete workbench. Switching waits for the active touch sequence to finish. **Rotate 90°**, safe-area handling, large touch targets, and a vertical narrow-screen layout support mobile use.

Mobile rendering uses simplified materials, disabled scene MSAA, and a capped canvas pixel ratio. These reduce display cost; they do not change the requested physical density model.

### Background execution and keep-alive

Long experiments use foreground animation scheduling, a background Worker, and, when served locally, a WebSocket clock and macOS sleep-prevention lease. The scheduler retains one pending tick rather than starting another WASM engine. **Continue scheduling when this tab is hidden** controls the background path; **Keep the screen awake while visible** requests Screen Wake Lock and reacquires it when appropriate.

The static-hosted page has no access to the local server's sleep-prevention process. Browser freeze/discard, forced system sleep, and lid closure can still interrupt work. Completed results, pending screenshot exports, and supported recovery paths help preserve experiment outputs, but reloading does not restore arbitrary live WASM/GPU state. Local captures are stored in `benchmark-captures/`; static hosting uses downloads.

## Performance evidence and fair comparisons

<img src="https://github.com/user-attachments/assets/84ac0b30-d669-4a40-99a5-31ef39b3f8c0" width="100%" alt="algorithm comparison" />

This figure is retained from the original performance record. It was not remeasured for the current implementation or this README revision and is not evidence that all proposed compression techniques are implemented or faster.

The workbench retains randomized method order, repeated measurements, medians and ranges, selectable source sizes, density-model counts, target counts, and accuracy thresholds. GPU timing uses GPU timestamps when available; missing timestamp measurements remain missing. CPU preparation, command encoding, transfers, and readback waits belong in end-to-end timing, not in GPU kernel time.

A scientifically useful comparison must distinguish:

- **Physical inputs:** shape, density, total mass, body rotation, and observation geometry.
- **Output space:** point acceleration, potential, gradient, integrated orbit, or whole-trajectory transform. These errors are not interchangeable.
- **Accuracy:** source resolution, near-surface behavior, time integration, spectral truncation, and held-out observations.
- **Workload:** cold preparation versus cached reuse, memory consumption, forward and sensitivity costs, and total completion time.

Agreement with an f64 reference using the same finite quadrature checks implementation consistency. It does not establish convergence to the continuous field. Likewise, a low transformed residual can coexist with unresolved high-frequency field structure. Comparisons against independently refined spatial references are needed before making proximity-navigation claims.

## Expected improvements

The next priorities are scientific validation and useful compression at a stated error tolerance:

| Direction | Planned work | Planetary-science value and acceptance evidence |
| --- | --- | --- |
| **Convergence and error budgets** | Refine angular/radial cells, wave-number coverage, observation sampling, and force-update timing independently | Separate physical density effects from numerical orbit drift; report surface, field, gradient, and arc errors |
| **Adaptive spectral compression** | Select modes and arc segments from error estimates; investigate Chebyshev/NUFFT and low-rank representations | Preserve small density-anomaly signatures while reducing storage and repeated inference cost |
| **Analytic source representations** | Implement stable polynomial-cell spectra, interface reductions, and appropriate local two-dimensional charts | Support richer heterogeneity and non-star-shaped bodies; validate against independent volume and polyhedral calculations |
| **Near-surface accuracy** | Combine controlled local source integration with a reusable spectral far field | Make low-altitude diagnostics credible where global spectral convergence becomes expensive |
| **Inverse-problem diagnostics** | Add noise-aware weighting, resolution and conditioning studies, prior sensitivity, multiple arcs, and uncertainty reporting | Distinguish a regularized numerical fit from recoverable interior structure |
| **Mission observables and forces** | Add instrument-frame/radio-science observations, ephemerides, solar radiation pressure, third-body perturbations, and controlled maneuvers | Move from synthetic gravity experiments toward mission-relevant inference and trajectory design |
| **GPU execution and reproducibility** | Reduce readbacks, batch forward/transpose work, evaluate GPU-side integration, and export complete experiment settings | Measure accuracy-qualified end-to-end gains across desktop and mobile adapters |
| **Surface dynamics** | Investigate contact, friction, and constrained motion with explicit event and error control | Extend gravity/slope inspection toward lander, hopper, and regolith-motion studies |

These are development directions, not delivered capability claims. Compression should be retained only where it preserves the observation information required by the scientific question and improves a measured cost at matched accuracy.

## Code map

| Path | Responsibility |
| --- | --- |
| `src/lib.rs` | Application resources, plugins, scheduling, and runtime setup |
| `src/cpu/density.rs` | Shared density geometry and mass-preserving source representation |
| `src/cpu/physics.rs` | Orbit integration, timestamped force use, and observation-arc capture |
| `src/cpu/frequency_domain.rs` | Frequency-domain source preparation and discrete transform definitions |
| `src/gpu/frequency_domain_pipeline/` | Whole-trajectory transforms, spectra, sensitivities, planning, and readback |
| `src/gpu/fmm_pipeline/` | Hierarchical gravity preparation and evaluation |
| `src/gpu/mmfft_pipeline/` | Grid, spectrum, and interpolated field evaluation |
| `src/gpu/radial.rs` | Independent radial analytic field dispatch and reductions |
| `src/cpu/inversion_components/` | Density basis, reference operators, caches, and constrained optimization |
| `src/bevy/surface_field.rs` | Chunked surface-field evaluation and display |
| `src/wgsl/` | Unified numerical and rendering shaders |
| `src/html/navigation.js` | Camera/whole-UI gestures, panel transforms, and mobile interaction |
| `src/html/background*.js` | Background scheduling and keep-alive coordination |
| `src/html/ui.js` / `src/html/app.js` | Controls, snapshots, and charts |
| `mathpub.md` | Public mathematical companion |

## Build and checks

A local build needs Rust with the `wasm32-unknown-unknown` target, `wasm-pack`, and Bun. The interactive numerical application requires an available WebGPU adapter in a supporting browser; local serving uses `localhost`, and deployment uses HTTPS.

```sh
rustup target add wasm32-unknown-unknown
bun install
bun run build
bun run serve
```

The local server runs at `http://localhost:3000`. `bun run dev` builds the development WASM package and starts the server; `bun run preview` performs a release build before serving. Keep the supplied model and operator assets with the project.

The Rust checks used by CI include:

```sh
cargo fmt --all -- --check
RUSTC_WRAPPER= cargo clippy --locked --target wasm32-unknown-unknown --lib -- -D warnings
RUSTC_WRAPPER= cargo check --locked --target wasm32-unknown-unknown --lib
```

The GitHub Pages workflow builds the release WASM package and Tailwind stylesheet, bundles the Vue telemetry interface, checks JavaScript syntax and deployment asset paths, and assembles the static site. Passing those checks establishes build and packaging consistency; scientific convergence and device-specific GPU behavior require separate validation.

## License

MIT; see [LICENSE](LICENSE).
