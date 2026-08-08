# Three Complementary Gravity Algorithms for Irregular Small Bodies

## Abstract

This document presents three gravitational-field algorithms implemented or investigated in the Ryugu WASM project. They are deliberately separated by their mathematical assumptions, source representations, numerical structures, and intended applications.

1. **The Werner-Scheeres polyhedron algorithm** is a classical reference method for the exact exterior potential of a homogeneous, closed polyhedron.
2. **The radial-analytic GPU algorithm** is the second, original algorithm of this project. It represents a star-shaped body by angular cells and mass-preserving radial layers, evaluates each radial integral analytically, and parallelizes only the remaining angular-layer sum.
3. **The Equation (106) adaptive curved-trajectory algorithm** is the third, original algorithm. It begins with the complex-frequency straight-line operator derived from Equation (70), transports that operator to a curved trajectory by a convergent spatial Taylor expansion, and switches from a finite non-periodic representation to a periodic sideband representation only after repeated stable orbital closure has been established.

The three methods are complementary rather than interchangeable. The Werner model is the most appropriate homogeneous polyhedral benchmark; the radial-analytic method is designed for repeated evaluation of radially structured heterogeneous density fields; and the Equation (106) method is intended for long, structured trajectory arcs for which a compressed complex-frequency representation can be reused.

---

## 1. Common Physical Model and Notation

Let the field point be $\mathbf q\in\mathbb R^3$, the source point be $\mathbf p\in V$, and the mass density be $\rho(\mathbf p)$. The positive Newtonian potential is

$$
U(\mathbf q)
=
G_N\iiint_V
\frac{\rho(\mathbf p)}{\|\mathbf p-\mathbf q\|}
\,dV',
$$

and the gravitational acceleration is

$$
\boxed{
\mathbf g(\mathbf q)
=
\nabla_{\mathbf q}U(\mathbf q)
=
G_N\iiint_V
\rho(\mathbf p)
\frac{\mathbf p-\mathbf q}
{\|\mathbf p-\mathbf q\|^3}
\,dV'.
}
$$

The sign convention is therefore fixed: $(U>0)$, and its gradient points toward the source mass. Throughout the document, $(G_N)$ denotes the Newtonian gravitational constant, whereas $(G)$ may be used in implementation-oriented formulas for the same constant.

The three algorithms differ only in how the volume integral is represented and evaluated.

---

# Part I. Classical Reference Algorithm

## 2. Werner-Scheeres Homogeneous Polyhedron Method

### 2.1 Scope and assumptions

The Werner-Scheeres method evaluates the gravitational potential and acceleration of a body whose boundary is represented by a consistently oriented, closed triangular polyhedron and whose density is spatially uniform:

$$
\rho(\mathbf p)=\rho_0.
$$

The method converts the volume integral into finite sums over the edges and faces of the boundary mesh. For an exterior field point and an exact closed polyhedron, these sums are analytic up to elementary logarithms and solid-angle functions. No voxelization or volumetric quadrature is required.

The method is not a heterogeneous-density algorithm unless the body is further decomposed into multiple constant-density polyhedra. In this project it is used as a homogeneous reference model and as an independent benchmark for the other forward solvers.

### 2.2 Geometric quantities

Let the polyhedron have vertex set $\{\mathbf v_i\}$, edge set $\mathcal E$, and triangular face set $\mathcal F$. Relative to the field point $\mathbf q$, define

$$
\mathbf r_i=\mathbf v_i-\mathbf q.
$$

For a face $f=(i,j,k)$, let $\mathbf n_f$ be the outward unit normal and define the face dyad

$$
\boxed{
\mathbf F_f=\mathbf n_f\mathbf n_f^{\mathsf T}.
}
$$

For an edge $e=(i,j)$ shared by two adjacent faces, the edge dyad $\mathbf E_e$ is assembled from the outward face normals and the corresponding in-plane edge normals. Its exact sign depends on the globally consistent orientation convention; once that convention is fixed, the same $\mathbf E_e$ must be used in the potential, acceleration, and Hessian formulas.

Let

$$
\ell_e=\|\mathbf v_j-\mathbf v_i\|,
\qquad
r_i=\|\mathbf r_i\|,
\qquad
r_j=\|\mathbf r_j\|.
$$

The edge logarithm is

$$
\boxed{
L_e
=
\ln
\frac{r_i+r_j+\ell_e}
{r_i+r_j-\ell_e}.
}
$$

For a triangular face $f=(i,j,k)$, the signed solid angle subtended at $\mathbf q$ is evaluated robustly as

$$
\boxed{
\omega_f
=
2\operatorname{atan2}
\left(
\mathbf r_i\cdot(\mathbf r_j\times\mathbf r_k),
r_ir_jr_k
+r_i\mathbf r_j\cdot\mathbf r_k
+r_j\mathbf r_k\cdot\mathbf r_i
+r_k\mathbf r_i\cdot\mathbf r_j
\right).
}
$$

The signed formulation is essential. Replacing $\omega_f$ by its absolute value destroys the orientation-dependent cancellation required by the divergence theorem.

### 2.3 Potential, acceleration, and gravity gradient

Choose any point on edge $e$ to define $\mathbf r_e$, and any point on face $f$ to define $\mathbf r_f$. With a consistent Werner-Scheeres orientation convention, the positive potential can be written as

$$
\boxed{
U_{\rm W}(\mathbf q)
=
\frac{G_N\rho_0}{2}
\left[
\sum_{e\in\mathcal E}
\mathbf r_e^{\mathsf T}\mathbf E_e\mathbf r_e\,L_e
-
\sum_{f\in\mathcal F}
\mathbf r_f^{\mathsf T}\mathbf F_f\mathbf r_f\,\omega_f
\right].
}
$$

Differentiation gives the gravitational acceleration

$$
\boxed{
\mathbf g_{\rm W}(\mathbf q)
=
-G_N\rho_0
\left[
\sum_{e\in\mathcal E}
\mathbf E_e\mathbf r_e\,L_e
-
\sum_{f\in\mathcal F}
\mathbf F_f\mathbf r_f\,\omega_f
\right].
}
$$

The gravity-gradient tensor has the corresponding finite-sum structure

$$
\boxed{
\nabla\mathbf g_{\rm W}
=
G_N\rho_0
\left[
\sum_{e\in\mathcal E}\mathbf E_eL_e
-
\sum_{f\in\mathcal F}\mathbf F_f\omega_f
\right],
}
$$

subject to the same sign and dyad convention as the preceding two equations.

### 2.4 Numerical properties

The method has several important properties.

- Its exterior solution is analytic for the adopted polyhedral geometry.
- Its far field converges to the point-mass field of total mass $M=\rho_0V$.
- A consistently oriented closed mesh produces the correct signed-angle identities.
- The method remains sensitive to mesh defects, non-manifold edges, duplicated faces, and inconsistent face winding.
- Near an edge, face, or vertex, logarithmic and solid-angle terms can become individually large even though their assembled physical result remains finite away from the boundary. Stable evaluation therefore requires robust `atan2`, logarithm-domain checks, and compensated accumulation.

### 2.5 GPU organization

The edge and face contributions are independent for a fixed field point. A GPU implementation therefore uses two parallel reductions:

1. one reduction over all edge terms;
2. one reduction over all face terms.

The reduced edge and face sums are then combined to obtain $U_{\rm W}$ and $\mathbf g_{\rm W}$. Geometry-dependent dyads are precomputed once. Only the relative vectors, edge logarithms, and face solid angles vary with the observation point.

### 2.6 Role in the present project

The Werner-Scheeres solver is the **first algorithmic category** in this document. It is classical rather than original. Its purpose is to provide

- a homogeneous-density physical model;
- an independent benchmark for acceleration and potential;
- a near-surface reference for a triangulated body;
- and a validation path that does not share the radial discretization or complex-frequency assumptions of the two original methods.

---

# Part II. Second Algorithm: Original Radial-Analytic GPU Method

## 3. Mathematical Principle

### 3.1 Star-shaped source representation

Assume that the body is star-shaped with respect to a chosen origin. Each source direction $\mathbf n'\in\mathbb S^2$ intersects the surface once, at radius $a(\mathbf n')$. Write

$$
\mathbf p=r\mathbf n',
\qquad
0\le r\le a(\mathbf n'),
$$

with volume element

$$
dV'=r^2\,dr\,d\Omega'.
$$

The gravitational acceleration becomes

$$
\mathbf g(\mathbf q)
=
G_N\int_{\mathbb S^2}
\int_0^{a(\mathbf n')}
\rho(r,\mathbf n')
\frac{r\mathbf n'-\mathbf q}
{\|r\mathbf n'-\mathbf q\|^3}
r^2\,dr\,d\Omega'.
$$

The central observation is that the radial integral can be evaluated analytically on every interval on which the density is represented by a constant or a basis function with a known primitive.

### 3.2 Radial primitive

Let

$$
b=\mathbf q\cdot\mathbf n',
\qquad
c^2=\|\mathbf q\|^2-b^2,
\qquad
D(r)=\sqrt{(r-b)^2+c^2}.
$$

For a piecewise-constant radial density $\rho_j(\mathbf n')$ on $[a_j,a_{j+1}]$, define the vector primitive $\mathbf K$ by

$$
\frac{d\mathbf K}{dr}
=
r^2
\frac{r\mathbf n'-\mathbf q}{D(r)^3}.
$$

Then the layer contribution is exactly

$$
\int_{a_j}^{a_{j+1}}
r^2
\frac{r\mathbf n'-\mathbf q}{D(r)^3}
\,dr
=
\mathbf K(a_{j+1};\mathbf q,\mathbf n')
-
\mathbf K(a_j;\mathbf q,\mathbf n').
$$

Consequently,

$$
\boxed{
\mathbf g(\mathbf q)
\approx
G_N\int_{\mathbb S^2}
\sum_{j=0}^{N_r-1}
\rho_j(\mathbf n')
\left[
\mathbf K(a_{j+1};\mathbf q,\mathbf n')
-
\mathbf K(a_j;\mathbf q,\mathbf n')
\right]
\,d\Omega'.
}
$$

The approximation is introduced by the angular and density discretizations, not by numerical integration along a regular radial layer.

### 3.3 Angular discretization

The surface mesh induces angular cells. For cell $k$, preprocessing stores

- a representative unit direction $\mathbf n'_k$;
- its solid angle $\Delta\Omega_k$;
- and the surface radius $a_k$.

For normalized triangle directions $\mathbf u_0,\mathbf u_1,\mathbf u_2$, the cell solid angle is

$$
\boxed{
\Delta\Omega_k
=
2\operatorname{atan2}
\left(
\left|\mathbf u_0\cdot(\mathbf u_1\times\mathbf u_2)\right|,
1+\mathbf u_0\cdot\mathbf u_1
+\mathbf u_1\cdot\mathbf u_2
+\mathbf u_2\cdot\mathbf u_0
\right).
}
$$

The remaining angular integral becomes a weighted finite sum.

### 3.4 Mass-preserving radial layers

The implementation uses $N_r=4$ equal-volume radial layers:

$$
a_{k,j}
=
a_k\left(\frac{j}{N_r}\right)^{1/3},
\qquad
j=0,\ldots,N_r.
$$

For the current heterogeneous radial model,

$$
\rho(r)=\frac{C}{r+\varepsilon},
\qquad
\varepsilon=10\ {\rm m}.
$$

The density stored in layer $(k,j)$ is its volume-weighted mean:

$$
\boxed{
\rho_{k,j}
=
C\,
\frac{
\displaystyle
\int_{a_{k,j}}^{a_{k,j+1}}
\frac{r^2}{r+\varepsilon}\,dr
}{
\displaystyle
\frac{a_{k,j+1}^3-a_{k,j}^3}{3}
}.
}
$$

This definition preserves the mass of every radial layer. The normalization constant $C$ is selected so that the sum of all angular-layer masses equals the prescribed asteroid mass.

### 3.5 Fully discrete forward operator

After both discretizations,

$$
\boxed{
\mathbf g(\mathbf q)
\approx
G_N
\sum_{k=0}^{N_\Omega-1}
\sum_{j=0}^{N_r-1}
\Delta\Omega_k\rho_{k,j}
\left[
\mathbf K(a_{k,j+1};\mathbf q,\mathbf n'_k)
-
\mathbf K(a_{k,j};\mathbf q,\mathbf n'_k)
\right].
}
$$

The matching positive potential is evaluated from the corresponding scalar radial primitive. Computing potential and acceleration from analytically related primitives provides a direct check of

$$
\mathbf g=\nabla U.
$$

### 3.6 GPU data structure and reduction

One angular-layer record occupies 32 bytes:

```text
vec4<f32> direction_solid_angle = [n'_x, n'_y, n'_z, DeltaOmega]
vec4<f32> radii_density         = [a_inner, a_outer, rho_layer, 0]
```

For a fixed observation point, every pair $(k,j)$ is independent. A workgroup therefore

1. loads one angular-layer record per invocation;
2. evaluates the primitive at both layer endpoints;
3. multiplies their difference by $(G_N\Delta\Omega_k\rho_{k,j})$;
4. reduces acceleration and potential in shared memory;
5. writes one partial sum per workgroup;
6. and completes the final reduction after asynchronous readback.

### 3.7 Degenerate and near-singular configurations

When $\mathbf n'\) is nearly collinear with $\mathbf q$, the closed-form primitive can become ill-conditioned because algebraically cancelling terms are individually large. The implementation detects this narrow configuration and evaluates only the affected layer with an eight-point Gauss-Legendre rule. Regular layers continue to use the endpoint primitive.

This fallback is not a global replacement of the analytic method. It is a local stabilization of a removable or nearly removable degeneracy.

### 3.8 Original contribution and limitations

The original contribution is the computational organization

$$
\boxed{
\text{star-shaped angular geometry}
+
\text{mass-preserving radial basis}
+
\text{analytic radial primitive}
+
\text{GPU angular reduction}.
}
$$

Its principal advantages are reduced radial discretization error, compact source storage, and efficient repeated evaluation for multiple density models that share the same angular geometry.

Its principal limitations are

- the star-shaped-body assumption;
- piecewise radial density approximation;
- reduced advantage for arbitrary non-radial heterogeneity;
- and the need for special treatment near degenerate source-field alignment.

---

# Part III. Third Algorithm: Equation (106) Adaptive Curved-Trajectory Method

## 4. Straight-Line Complex-Frequency Operator

### 4.1 Equation (70)

Let the source point use spherical coordinates $(\lambda,\theta',\phi')$, and let the observation line use cylindrical coordinates $(\varrho,\phi,h)$. Define

$$
r'_{\perp}=\lambda\sin\theta',
\qquad
z'=\lambda\cos\theta',
\qquad
\Delta\phi=\phi-\phi',
$$

$$
a
=
\sqrt{
\varrho^2+r_{\perp}'^2
-2\varrho r'_{\perp}\cos\Delta\phi
},
\qquad
x=as,
\qquad
\eta=\frac{z'}a.
$$

Let $\Psi(x,\eta)$ denote the dimensionless scalar Laplace kernel. Define

$$
K_H
=
x\Psi_x+\eta x\Psi
-\frac{\eta}{\sqrt{1+\eta^2}},
$$

$$
K_V
=
x\Psi
-\frac1{\sqrt{1+\eta^2}}.
$$

Then the straight-line complex-frequency gravitational operator is

$$
\boxed{
\begin{aligned}
\widetilde{\mathbf g}_{70}(s)
=
G_N\iiint
&\rho(\lambda,\theta',\phi')
\lambda^2\sin\theta'
\Bigg[
\mathbf e_\varrho
\frac{\varrho-r'_{\perp}\cos\Delta\phi}{a^2}K_H
\\
&+
\mathbf e_\phi
\frac{r'_{\perp}\sin\Delta\phi}{a^2}K_H
+
\mathbf e_z\frac1aK_V
\Bigg]
d\lambda\,d\theta'\,d\phi'.
\end{aligned}
}
\tag{70}
$$

The apparent factors $(1/a)$ and $1/a^2$ require explicit limiting analysis as $a\to0$. Some singularities are coordinate artifacts; others correspond to a genuine near-field or self-cell singularity. An arbitrary numerical regularizer cannot replace the analytic limit or a self-cell integral.

### 4.2 Equation (106)

For the reference line

$$
\overline{\mathbf q}(h)
=
\mathbf q_0+h\mathbf e_z,
$$

the operator is

$$
\boxed{
\widetilde{\mathbf g}_{70}(s)
=
G_N\iiint
\rho(\lambda,\theta',\phi')
\lambda^2\sin\theta'
\left[
\mathbf e_\varrho A_HK_H
+
\mathbf e_\phi B_HK_H
+
\mathbf e_zA_VK_V
\right]
d\lambda\,d\theta'\,d\phi'.
}
\tag{106}
$$

It represents

$$
\widetilde{\mathbf g}_{70}(s)
=
\int_0^\infty
\mathbf g(\overline{\mathbf q}(h))e^{-sh}\,dh.
$$

For $(s=\sigma+i\omega)$, $\sigma>0$, the spatial field is recovered by the Bromwich inversion

$$
\boxed{
\mathbf g(h)
=
\frac{e^{\sigma h}}{2\pi}
\int_{-\infty}^{\infty}
\widetilde{\mathbf g}_{70}(\sigma+i\omega)
e^{i\omega h}\,d\omega.
}
$$

With common frequencies $(\omega_k)$,

$$
\boxed{
\mathbf g(h_j)
\approx
\frac{e^{\sigma h_j}\Delta\omega}{2\pi}
\sum_{k=-K}^{K}
\widetilde{\mathbf g}_{70}(\sigma+i\omega_k)
e^{i\omega_kh_j}.
}
$$

Unlike pointwise Gaver-Stehfest inversion, all observation points share the same frequency set. This shared spectrum is the source of the possible trajectory-level compression.

## 5. Curved-Trajectory Transport

### 5.1 Reference-line decomposition

Let

$$
\mathbf q(t)
=
\overline{\mathbf q}(t)
+
\delta\mathbf q(t),
\qquad
\overline{\mathbf q}(t)
=
\mathbf q_0+vt\mathbf e_z.
$$

After transforming the straight-line field to a fixed Cartesian basis, the exact translation operator is

$$
\mathbf g(\overline{\mathbf q}+\delta\mathbf q)
=
\exp(\delta\mathbf q\cdot\nabla_{\mathbf q})
\mathbf g(\overline{\mathbf q}).
$$

Therefore,

$$
\boxed{
\mathbf g_\gamma(t)
=
\sum_{n=0}^{\infty}
\frac1{n!}
\left[
\delta\mathbf q(t)\cdot\nabla_{\mathbf q}
\right]^n
\mathbf g_{70}(t).
}
\tag{118}
$$

The first two corrections are

$$
\mathbf g_\gamma
\approx
\mathbf g_{70}
+
\delta q_i\partial_i\mathbf g_{70}
+
\frac12\delta q_i\delta q_j
\partial_i\partial_j\mathbf g_{70}.
$$

### 5.2 Finite non-periodic trajectory

For a finite non-periodic arc, approximate each displacement component by a local polynomial,

$$
\delta q(t)=\sum_{k=0}^{P}d_kt^k.
$$

The identity

$$
\mathcal L[t^kf(t)](s)
=
(-1)^k\partial_s^k\widetilde f(s)
$$

converts the trajectory correction into mixed spatial and complex-frequency derivatives:

$$
\boxed{
\widetilde{\mathbf g}_\gamma(s)
\approx
\sum_{\alpha,k}
D_{\alpha k}
\partial_s^k
\partial_{\mathbf q}^{\alpha}
\widetilde{\mathbf g}_{70}(s).
}
\tag{126}
$$

This non-periodic representation is the default because a newly observed trajectory cannot be assumed periodic.

### 5.3 General periodic curved trajectory

Only after stable closure has been demonstrated may the displacement be represented by a Fourier series,

$$
\delta\mathbf q(t)
=
\sum_{\ell=-L}^{L}
\mathbf d_\ell e^{i\ell\Omega t}.
$$

Truncation of the translation operator at order $A$ produces spatial derivatives and frequency sidebands:

$$
\boxed{
\widetilde{\mathbf g}_\gamma(s)
\approx
\sum_{|\alpha|\le A}
\sum_{\nu=-AL}^{AL}
c_{\alpha,\nu}
\partial^\alpha
\widetilde{\mathbf g}_{70}(s-i\nu\Omega).
}
\tag{122}
$$

In the implemented planner, promotion to this periodic category requires at least ten consecutive orbit closures satisfying position, velocity, and period tolerances. A failed closure resets the consecutive-closure counter.

## 6. Adaptive Segmentation and Taylor Convergence

### 6.1 Convergence radius

For a source point $(\mathbf p)$, the Newton kernel is singular at $\mathbf q=\mathbf p$. A Taylor expansion about the reference line can converge only while the trajectory displacement remains inside the nearest-source distance. Define

$$
\boxed{
\varepsilon_{\max}
=
\sup_h
\frac{\|\delta\mathbf q(h)\|}{d(h)}
<1,
}
$$

where $d(h)$ is a conservative lower bound on the distance from the reference line to the density support.

For a local arc with curvature $(\kappa)$ and half-length $(\ell)$,

$$
\|\delta\mathbf q\|
\approx
\frac{\kappa\ell^2}{2},
$$

so a practical local condition is

$$
\boxed{
\frac{\kappa\ell^2}{2d_{\min}}<1.
}
$$

### 6.2 Remainder bound

For truncation at Taylor order $(A)$, the geometric form of the remainder satisfies

$$
\boxed{
\|\mathbf R_A\|
\lesssim
C\frac{G_NM}{d_{\min}^2}
\frac{\varepsilon_{\max}^{A+1}}
{1-\varepsilon_{\max}},
}
$$

where $C$ depends on directional derivatives and source geometry.

The implementation uses the dimensionless guard

$$
R_A^{\rm rel}
=
\frac{\varepsilon_{\max}^{A+1}}
{1-\varepsilon_{\max}}
$$

and selects the smallest admissible order whose bound is below the prescribed tolerance. If no allowed order is sufficient, the arc is bisected. A segment that remains non-convergent at the minimum sample count is rejected and the dynamics falls back to a bounded Newtonian anchor.

### 6.3 Number of Taylor terms

A full three-dimensional Taylor expansion through degree $A$ contains

$$
B_A=\binom{A+3}{3}
$$

multi-index terms. Thus the method is competitive only when the trajectory displacement has a sparse directional or spectral structure, or when adaptive segmentation keeps $A$ small.

## 7. Potential, Path Work, and Dual Residual

### 7.1 Straight-line potential difference

Along a straight reference line,

$$
\frac{dU}{dh}=g_z(h),
$$

and therefore

$$
\widetilde{\Delta U}(s)
=
\frac{\widetilde g_z(s)}s.
$$

### 7.2 Curved-path work identity

For a general cylindrical trajectory $\mathbf q(h)=(\varrho(h),\phi(h),z(h))$,

$$
\boxed{
\frac{dU}{dh}
=
g_\varrho\varrho'
+
g_\phi\varrho\phi'
+
g_zz'.
}
\tag{146}
$$

Hence

$$
\boxed{
\Delta U_\gamma(h)
=
\int_0^h
\left[
g_\varrho\varrho'
+
g_\phi\varrho\phi'
+
g_zz'
\right]d\xi.
}
\tag{147}
$$

The corresponding Taylor-transport expression is

$$
\boxed{
\widetilde{\Delta U}_\gamma(s)
=
\frac1s
\mathcal L
\left[
(\mathbf e_z+\delta\mathbf q')\cdot
e^{\delta\mathbf q\cdot\nabla}
\mathbf g_{70}
\right](s).
}
\tag{151}
$$

### 7.3 Dual-representation residual

Let $P_{70}^{(A)}\rho$ be the Equation (106) potential-difference operator truncated at Taylor order $A$, and let $P_{\rm spec}^{(H,P)}\rho$ be an independent spectral or direct potential representation. In the exact limit they represent the same Newtonian operator. At finite resolution,

$$
\boxed{
\mathbf r_{\rm dual}
=
\left[
P_{70}^{(A)}
-
P_{\rm spec}^{(H,P)}
\right]\rho.
}
\tag{157}
$$

The runtime monitor evaluates a discrete version of this identity by comparing

1. the trapezoidal curved-path work integral of the three-axis acceleration; and
2. the independently accumulated positive gravitational-potential difference returned by the GPU kernel.

Thus the displayed quantity has units of specific potential,

$$
|r_{\rm dual}|\quad [{\rm m^2\,s^{-2}}].
$$

It detects inconsistent acceleration and potential, insufficient path resolution, Taylor-transport error, floating-point cancellation, and implementation mistakes in either representation.

## 8. Complexity

Let

- $(K_\omega)$ be the number of common complex frequencies;
- $N_\rho$ be the number of density degrees of freedom;
- $B$ be the retained number of spatial-derivative and sideband terms;
- $N_t$ be the number of trajectory samples.

The curved complex-frequency construction costs approximately

$$
O(BK_\omega N_\rho),
$$

and the inverse transform or NUFFT costs

$$
O\!\left(B(K_\omega+N_t)\log K_\omega\right).
$$

The total is

$$
\boxed{
O\!\left[
BK_\omega N_\rho
+
B(K_\omega+N_t)\log K_\omega
\right].
}
$$

The necessary, but not sufficient, compression criterion is

$$
\boxed{
BK_\omega\ll N_t.
}
$$

If strong curvature forces a large $B$, or if only a few field points are required, the method loses its advantage.

## 9. Intended Contribution and Limitations

The Equation (106) method should not be claimed as a universal replacement for the fast multipole method, polyhedron gravity, or direct mascon evaluation. Its specific hypothesis is

$$
\boxed{
\text{operator-level complex-frequency compression is advantageous}
}
$$

for long, structured, repeatedly evaluated flyby or orbital arcs.

The method is most appropriate when

- the shape and rotation model are already known;
- a long arc contains many samples;
- the density model is changed repeatedly while the trajectory is approximately fixed;
- the arc can be partitioned into low-curvature segments;
- and the requested output is naturally expressed in the frequency domain or by a reduced set of trajectory coefficients.

It is poorly suited when

- the trajectory is strongly curved over long intervals;
- only a few field evaluations are needed;
- the body is adequately modeled as a homogeneous polyhedron;
- the trajectory changes substantially after every density update;
- or the field point approaches the density support so closely that the Taylor convergence radius is exhausted.

Finally, a single flyby does not uniquely determine an arbitrary three-dimensional density field. Without additional trajectories, priors, gradient measurements, or long-term tracking, the identifiable parameters are generally restricted to total mass, center-of-mass displacement, low-order moments, or a small prescribed set of anomalies.

---

# Part IV. Comparative Taxonomy

## 10. Algorithm Selection

| Property | Werner-Scheeres polyhedron | Original radial-analytic GPU method | Original Equation (106) curved-trajectory method |
|---|---|---|---|
| Density assumption | Uniform | Radially structured, piecewise basis | General discretized density in the formal operator |
| Geometry assumption | Closed oriented polyhedron | Star-shaped body | Convergent segmented trajectory relative to reference lines |
| Primary analytic reduction | Volume to edge/face sums | Radial integral to endpoint primitives | Trajectory to common complex-frequency coefficients |
| Natural GPU parallelism | Edge and face reductions | Angular-layer reductions | Frequency-density-derivative blocks |
| Near-surface behavior | Strong reference method for homogeneous polyhedron | Requires degenerate/self-cell treatment | Taylor convergence deteriorates near density support |
| Repeated density models | Limited unless multiple polyhedra are used | Strong | Potentially strong when the compressed operator is reusable |
| Arbitrary trajectory | Yes, pointwise | Yes, pointwise | Only through convergent segmentation or periodic sidebands |
| Role in this project | Classical benchmark | Second algorithm; original | Third algorithm; original |

## 11. Cross-Validation Strategy

A credible numerical study should compare the three methods and direct reference solutions on

- a point mass;
- a homogeneous sphere;
- a closed tetrahedron or irregular polyhedron;
- a heterogeneous star-shaped body;
- straight, sinusoidally curved, hyperbolic, and periodic trajectories;
- near-surface and far-field points;
- and both single-point and long-trajectory workloads.

At minimum, the following quantities should be reported:

$$
\frac{\|\mathbf g_{\rm method}-\mathbf g_{\rm ref}\|}
{\|\mathbf g_{\rm ref}\|},
\qquad
\frac{|U_{\rm method}-U_{\rm ref}|}
{|U_{\rm ref}|},
$$

$$
\|\nabla U-\mathbf g\|,
\qquad
|r_{\rm dual}|,
$$

and wall-clock time, memory traffic, preprocessing cost, and amortized cost per repeated density evaluation.

## 12. Publication-Level Claims

The academically defensible claims are as follows.

1. The Werner-Scheeres method is a classical homogeneous-polyhedron reference algorithm and is not claimed as original.
2. The radial-analytic GPU method is original in its combined source representation, mass-preserving radial discretization, analytic radial endpoint evaluation, and GPU reduction architecture.
3. The Equation (106) method is original in its construction of a segmented curved-trajectory operator from the Equation (70) complex-frequency straight-line kernel, with explicit non-periodic and periodic branches, a Taylor convergence guard, and a dual potential-work residual.
4. Neither original method is universally superior. Their value must be demonstrated within their stated structural regimes and against strong baselines such as polyhedron gravity, mascons, FFT methods, and FMM-based reduced-order models.