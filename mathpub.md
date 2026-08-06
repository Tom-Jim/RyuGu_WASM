# Equation: GPU Evaluation Form

## Scope

This document states the forward model used by the application and explains only how it is converted into a GPU-computable form. It does not include the derivation of Equation, an inverse problem, or any reconstruction procedure.

## Forward model

Let the observation point be $\mathbf q$, let $\mathbf n'_k$ denote a source direction, and let $[a_{k,j},a_{k,j+1}]$ be radial layer $j$ in that direction. The forward gravitational acceleration is evaluated with Equation:

$$
\boxed{
\mathbf g(\mathbf q)
\approx
G\int_{\mathbb S^2}
\sum_{j=0}^{N_r-1}
\rho_j(\mathbf n')
\left[
\mathbf K(a_{j+1};\mathbf q,\mathbf n')
-
\mathbf K(a_j;\mathbf q,\mathbf n')
\right]
\,d\Omega'
}
$$

Here, $\mathbf K$ is the closed-form radial primitive evaluated at a layer boundary. The GPU evaluates $\mathbf K$ directly at the two endpoints; it does not perform a numerical integral over the regular radial interval.

## Angular discretization

The asteroid surface mesh supplies a finite set of angular cells. For cell $k$, the preprocessing stage stores:

- a representative unit direction $\mathbf n'_k$;
- the cell solid angle $\Delta\Omega_k$;
- the surface radius $a_k$ along that direction.

For a surface triangle with normalized vertex directions $\mathbf u_0$, $\mathbf u_1$, and $\mathbf u_2$, its solid angle is evaluated as

$$
\Delta\Omega_k
=
2\operatorname{atan2}\!\left(
\left|\mathbf u_0\cdot(\mathbf u_1\times\mathbf u_2)\right|,
1+\mathbf u_0\cdot\mathbf u_1
+\mathbf u_1\cdot\mathbf u_2
+\mathbf u_2\cdot\mathbf u_0
\right).
$$

The angular integral in Equation then becomes a weighted sum.

## Radial layers

Each angular cell is divided into \(N_r=4\) equal-volume radial layers:

$$
a_{k,j}=a_k\left(\frac{j}{N_r}\right)^{1/3},
\qquad j=0,\ldots,N_r.
$$

The current non-uniform density model is

$$
\rho(r)=\frac{C}{r+\varepsilon},
\qquad \varepsilon=10\ \mathrm m.
$$

One constant density is stored per layer. It is the volume-weighted mean of the continuous density:

$$
\rho_{k,j}
=
C\,
\frac{
\displaystyle\int_{a_{k,j}}^{a_{k,j+1}}\frac{r^2}{r+\varepsilon}\,dr
}{
\displaystyle\frac{a_{k,j+1}^3-a_{k,j}^3}{3}
}.
$$

This choice preserves the mass of every discretized radial layer. The normalization constant \(C\) is selected so that the sum of all angular-layer masses equals the configured asteroid mass.

## Fully discrete GPU expression

After angular and radial discretization, Equation becomes

$$
\boxed{
\mathbf g(\mathbf q)
\approx
G\sum_{k=0}^{N_\Omega-1}
\sum_{j=0}^{N_r-1}
\Delta\Omega_k\,\rho_{k,j}
\left[
\mathbf K(a_{k,j+1};\mathbf q,\mathbf n'_k)
-
\mathbf K(a_{k,j};\mathbf q,\mathbf n'_k)
\right]
}
$$

Every pair \((k,j)\) is independent for a fixed observation point. This independence is the basis of the GPU implementation.

## GPU record layout

One angular-layer pair occupies 32 bytes:

```text
vec4<f32> direction_solid_angle = [n'_x, n'_y, n'_z, DeltaOmega]
vec4<f32> radii_density         = [a_inner, a_outer, rho_layer, 0]
```

The per-dispatch uniform occupies 32 bytes:

```text
[q_x, q_y, q_z, G, layer_count, padding...]
```

The observation point is transformed into the asteroid body-fixed frame before this uniform is uploaded.

## Parallel evaluation

The compute shader uses workgroups of 64 invocations:

1. Invocation $i$ reads one angular-layer record.
2. It evaluates $\mathbf K$ at the outer and inner layer boundaries.
3. It multiplies the difference by $G\Delta\Omega_k\rho_{k,j}$.
4. The 64 contributions are reduced in workgroup memory with a binary tree reduction.
5. One `vec4<f32>` partial sum is written per workgroup.
6. The main world adds the workgroup partial sums and obtains $\mathbf g(\mathbf q)$.

The reduction needs six synchronization stages because $\log_2 64=6$.

## Degenerate angular configuration

The closed-form primitive becomes numerically ill-conditioned when the source direction is nearly collinear with the observation direction. For that narrow case, the shader evaluates only the affected radial layer with an eight-point Gauss--Legendre rule. Regular directions continue to use the closed-form endpoint difference.

## Runtime data flow

```text
surface mesh
  -> angular cells and radial layers
  -> immutable GPU storage buffer
  -> body-fixed observation point
  -> Equation compute dispatch
  -> workgroup partial sums
  -> asynchronous staging-buffer readback
  -> body-frame acceleration
  -> world-frame rotation
  -> fixed-step orbit integration
```

GPU buffers and bind groups are created once and reused. An atomic in-flight flag prevents the same staging buffer from being mapped by overlapping frames.

## Applicability

The current discretization assumes that the body is star-shaped relative to the model origin, so one surface radius is sufficient for each angular direction. A non-star-shaped body requires multiple disjoint radial intervals in the same angular cell.
