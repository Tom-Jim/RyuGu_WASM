## Core Mathematical Framework

### 1. Newtonian Gravitational Field and Sensitivity Kernel (Fréchet Derivative)

$$\mathbf{g}(\mathbf{q}) = G \iiint_V \rho(\mathbf{p}) \frac{\mathbf{p} - \mathbf{q}}{R^3} \, dV' \implies \frac{\delta \mathbf{g}(\mathbf{q})}{\delta \rho(\mathbf{p})} = G \nabla_{\mathbf{q}} \left( \frac{1}{R} \right)$$

---

### 2. Closed-Form Analytical Kernel of the Variational Tensor Definite Integral

$$\begin{aligned} \int_0^{r_i} \frac{\delta \mathbf{G}}{\delta \rho} \, dr = G \int_0^\infty e^{-s\zeta} & \left[ \left( \frac{1}{D_\zeta(r)} + \cot\theta \frac{(\zeta-z')\bigl(D_\zeta(r)-u(r)\bigr)}{k^2 D_\zeta(r)} \right) \mathbf{e}_r \right. \\ & \left. + \left( \frac{\cot\theta}{D_\zeta(r)} - \frac{(\zeta-z')\bigl(D_\zeta(r)-u(r)\bigr)}{k^2 D_\zeta(r)} \right) \mathbf{e}_\theta \right. \\ & \left. - \frac{r'\sin\theta'\sin(\phi-\phi')}{\sin\theta} \frac{u(r)}{k^2 D_\zeta(r)} \mathbf{e}_\phi \right]_{r=0}^{r=r_i} d\zeta \end{aligned}$$

---

### 3. Analytical Decay Kernel: Struve–Neumann Modes ($\mathcal{S}_0$, $\mathcal{S}_1$)

$$I_R = \int_0^\infty \frac{e^{-sw}}{\sqrt{w^2 + a^2}} \, dw = \mathcal{S}_0(as) \equiv \frac{\pi}{2} \left[ \mathbf{H}_0(as) - Y_0(as) \right]$$

$$\int_0^\infty \frac{e^{-sw}}{(w^2+a^2)^{3/2}} \, dw = \frac{s}{a} \left[ \mathcal{S}_1(as) - 1 \right]$$

---

### 4. Spatial-Domain Inversion: Gaver–Stehfest Numerical Inverse Laplace Transform

$$\mathbf{g}(h) \approx \frac{\ln 2}{h} \sum_{k=1}^{2M} V_k \cdot \mathbf{G}_{\text{total}}\!\left( s = k \frac{\ln 2}{h} \right)$$

---
