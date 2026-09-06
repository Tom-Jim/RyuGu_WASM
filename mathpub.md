# Ryugu Dynamics Laboratory: Public Mathematical Description

This document is the public mathematical companion to the Ryugu Dynamics Laboratory README. It describes the physical model, the four forward methods, the frequency-domain data flow, density inversion, GPU organization, and the validation limits of the current implementation. Internal derivation notebooks are not part of the public interface.

## 1. Common gravity model

Let $V$ be the asteroid volume, let $\rho(\mathbf p)$ be its mass density, and let $\mathbf q$ be an exterior observation point. The positive Newtonian potential and acceleration are

$$
U(\mathbf q)=G\int_V\frac{\rho(\mathbf p)}{\lVert\mathbf p-\mathbf q\rVert}\,dV,
\qquad
\mathbf g(\mathbf q)=\nabla_{\mathbf q}U(\mathbf q)
=G\int_V\rho(\mathbf p)\frac{\mathbf p-\mathbf q}{\lVert\mathbf p-\mathbf q\rVert^3}\,dV.
$$

The sign convention is fixed throughout the application: potential is positive and its gradient points toward the source. Geometry, density, and total mass are kept consistent across the forward methods so that differences measure numerical method behavior rather than different physical inputs.

## 2. Source representation and density modes

The shared source representation is a star-shaped angular mesh with radial shells. Each angular cell stores a representative direction, its solid angle, the surface radius, and shell density values. Shell boundaries are chosen by equal volume, so the source mass is preserved when the number of radial layers changes.

The **constant** mode assigns one homogeneous density to all occupied cells and normalizes it to the requested total mass. The **variable** mode evaluates the configured radial density profile and preserves each shell's integrated mass. The UI uses these same representations for field display, surface sampling, orbit propagation, and inversion; switching modes invalidates cached field products and schedules a bounded recomputation.

The Werner method is the homogeneous closed-polyhedron reference. It is mathematically independent of the shell quadrature. A heterogeneous surface request uses the heterogeneous source representation of the selected method rather than silently displaying a homogeneous reference result.

## 3. Forward methods

### 3.1 Werner polyhedron reference

For a consistently oriented closed triangular mesh with uniform density, the volume integral is reduced to finite edge and face sums. Edge terms contain logarithms of endpoint distances; face terms contain signed solid angles. The acceleration, potential, and gravity-gradient tensor are assembled from the same edge and face dyads, so the mesh orientation and sign convention are shared by every output.

Edge and face contributions are independent for one field point. GPU implementations can therefore reduce edge terms and face terms in separate workgroups, then combine the partial sums. The method is a homogeneous benchmark and becomes a heterogeneous solver only if the body is explicitly decomposed into multiple constant-density polyhedra.

### 3.2 Radial analytic solver

For a star-shaped body write $\mathbf p=r\mathbf n'$ with $0\le r\le a(\mathbf n')$. The volume integral separates into an angular integral and a radial primitive. For one angular direction define

$$
b=\mathbf q\cdot\mathbf n',\qquad
c^2=\lVert\mathbf q\rVert^2-b^2,qquad
D(r)=\sqrt{(r-b)^2+c^2}.
$$

The integrand

$$
r^2\frac{r\mathbf n'-\mathbf q}{D(r)^3}
$$

has a closed endpoint primitive. A shell contribution is the primitive at its outer radius minus the primitive at its inner radius. Summing these endpoint differences over angular cells and multiplying by solid angle gives the field. The scalar primitive supplies the matching potential, which provides a direct gradient consistency check.

Near the narrow alignment in which the endpoint expression loses conditioning, only the affected shell uses a small local Gauss rule. This is a stabilization of a local degeneracy; it does not replace the radial analytic method globally.

### 3.3 FFT reference

The FFT path samples the source on a regular Cartesian grid, transforms the density, applies the discretized Newton kernel in Fourier space, and interpolates the resulting field back to requested points. Its strengths are regular-grid throughput and predictable batching. Grid spacing, finite precision, interpolation, and kernel truncation are separate error sources.

### 3.4 FMM reference

The FMM path builds a hierarchy over source samples, stores multipole moments, and evaluates near and far contributions with different expansions. Source construction, tree traversal, and target evaluation are independently batched. The method is useful for large point sets and provides a field reference that is independent of both the regular FFT grid and the polyhedron formulas.

## 4. Frequency-domain propagation and observation

Frequency-domain mode has two physically separate stages.

### 4.1 Propagation stage

The spacecraft state $(\mathbf q,\mathbf v)$ is advanced from the current state using the continuous Ryugu density field. The spatial field is evaluated by the frequency-domain propagation kernel and returned through its own asynchronous GPU readback. A causal leapfrog step consumes only a readback belonging to the current density and state generation. If a readback is not ready, the integrator waits; it does not substitute a Radial, FFT, FMM, Werner, or prerecorded-track value.

This produces the actual observation arc, including the captured positions, velocities, times, and body attitudes. The arc is generated at runtime from the selected initial conditions and current density model.

### 4.2 Whole-trajectory transform

After the propagation arc has been captured, the transform stage evaluates the complete trajectory rather than a single fixed input track. For Laplace parameter $s$ and wave vector $\mathbf k$, the trajectory characteristic is

$$
\mathcal T_\gamma(s,\mathbf k)
=\int_0^T e^{-st+i\mathbf k\cdot\mathbf\gamma(t)}\,dt.
$$

The frequency-domain observation is formed by combining this characteristic with the density spectrum and the Newton kernel. The implementation uses composite-trapezoid weights over the captured samples and positive real Laplace frequencies. The observation is therefore a whole-trajectory quantity; it is not a point acceleration and cannot be compared to a point-field error without changing the metric.

The trajectory record is immutable for one inversion pass. If the trajectory, density generation, or sensitivity column changes, the corresponding spectrum and readback caches are invalidated explicitly.

## 5. Density inversion

Inversion freezes the observed trajectory and represents density with a finite basis over occupied source cells. Forward observations and sensitivity columns use the same discrete transform operator. The solver minimizes a constrained objective of the form

$$
\frac12\lVert A\mathbf x-\mathbf y\rVert^2
+\frac{\lambda}{2}\lVert L\mathbf x\rVert^2
$$

subject to mass, positivity, and configured density bounds. Here $\mathbf x$ contains basis coefficients, $A$ is the trajectory-observation operator, $\mathbf y$ is the measured or synthetic observation, and $L$ supplies the selected regularization.

The implementation reports residuals in the observation space and keeps the forward model, sensitivity model, and CPU reference operator aligned. This separation prevents a density update from reusing a stale forward spectrum or a point-field cache from another method.

## 6. GPU organization

WGSL shaders are kept under `src/wgsl/` and loaded through the Rust shader module. Work is divided into independent source, wave-number, time, edge, face, target, or shell records wherever the operator permits it. Typical reductions are staged as follows:

1. each invocation loads one independent record;
2. local contributions are evaluated in f32;
3. workgroup memory reduces vector, scalar, and Jacobian terms;
4. one partial record is written per workgroup;
5. a second dispatch or asynchronous readback completes the reduction.

The trajectory transform uses paired reciprocal wave-number nodes and multiple time lanes per workgroup. Density spectra are reused while the density generation is unchanged. Sensitivity updates invalidate only the affected spectrum, and results are published only when their trajectory and generation identifiers still match the current request.

The CPU retains f64 reference paths for operator checks, planning, and recovery. These references validate implementation consistency for the chosen finite discretization; they do not prove convergence of an infinite integral or of a continuous inverse problem.

## 7. Surface fields and comparisons

Surface products sample a bounded set of shared triangle targets and map the results onto the coverage grid in chunks. The same target positions and normals are used when comparing methods. Effective slope combines the local field with the selected body-frame normal. Error maps report comparison minus baseline, and missing or stale asynchronous results are not presented as valid samples.

Chunking is part of the numerical interface: field calculation yields between batches so FMM, Werner, and inversion requests do not monopolize the browser main loop. Cached products are keyed by method, density generation, target generation, and requested quantity.

## 8. Mobile and background execution

Mobile adaptation has two independent scales. In camera mode, gestures affect the Bevy camera while HTML controls retain their dimensions. In whole-UI mode, pinch, wheel, and drag affect the complete UI transform. Switching modes waits for the active touch sequence to finish so a gesture cannot be split between the two coordinate systems.

Background execution uses a foreground animation clock, a Worker scheduler, and the local-server keep-alive path. Only one pending tick is retained, and restoration or device loss recreates the existing engine rather than starting a second one. Screen Wake Lock is an optional foreground aid; it does not replace Worker scheduling or the local keep-alive lease.

## 9. Validation and limits

The meaningful validation matrix includes homogeneous spheres and closed polyhedra, heterogeneous star-shaped bodies, near-surface and far-field targets, straight and curved arcs, and repeated density updates. Report relative field, potential, gradient, trajectory, and wall-clock errors separately. GPU timestamps exclude CPU encoding and readback waits.

The current source geometry is a finite angular-cell and shell approximation. Finite wave-number and radial quadrature, f32 arithmetic, FFT grid quantization, FMM order, near-field treatment, and asynchronous extrapolation each have their own convergence behavior. A single external trajectory cannot identify an arbitrary three-dimensional density without a finite parameterization, priors, or additional observations.

The public claims of this project are therefore deliberately scoped: Werner supplies a homogeneous polyhedron reference; Radial, FFT, FMM, and frequency-domain methods are separate discretized solvers; and inversion results are meaningful only with the stated geometry, density basis, constraints, and validation metrics.
