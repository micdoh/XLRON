# Differentiable DRA Pipeline

This page explains how XLRON's Distributed Raman Amplification (DRA) model is made *end-to-end differentiable* — that is, how a single call of `jax.value_and_grad` can compute the gradient of total network throughput with respect to the Raman pump powers, propagating cleanly through ordinary differential equation (ODE) integration, a two-point boundary-value problem (BVP), a nonlinear least-squares fit, and the closed-form Gaussian Noise (GN) model integrals (which include inter-channel stimulated Raman scattering, abbreviated ISRS).

The page is intended for readers who know optical communications and the GN model but may not have a deep background in automatic differentiation (AD). It starts from first principles, names every concept it uses, and then walks through the two non-trivial differentiability tricks (an *implicit-function-theorem* gradient for the BVP solve, and a *surrogate gradient* for the profile fit) that the implementation in `xlron/environments/gn_model/isrs_gn_model_dra.py` relies on.

If you only want to *use* DRA in a simulation, the [GN Model page](gn_model.md#distributed-raman-amplification-dra) is enough. This page is for understanding why pump-power optimisation works and what is approximated in the gradient.

!!! note "Acronyms used on this page"
    **DRA** — Distributed Raman Amplification.
    **GN** — Gaussian Noise (model). **ISRS** — Inter-channel Stimulated Raman Scattering.
    **ODE** — Ordinary Differential Equation. **BVP** — Boundary-Value Problem.
    **WDM** — Wavelength Division Multiplexing.
    **NLI** — Nonlinear Interference. **ASE** — Amplified Spontaneous Emission.
    **SNR** — Signal-to-Noise Ratio. **FEC** — Forward Error Correction.
    **AD** — Automatic Differentiation. **JVP** — Jacobian-Vector Product (forward-mode AD).
    **VJP** — Vector-Jacobian Product (reverse-mode AD). **IFT** — Implicit Function Theorem.
    **TNC** — Truncated Newton Constrained (a scipy optimiser).
    **LM** — Levenberg–Marquardt (a nonlinear least-squares solver).

---

## Why we want differentiability

In a Raman-amplified network, the operator chooses backward pump powers \(\mathbf{P}_p \in \mathbb{R}^{M}\) (and optionally pump frequencies and per-channel launch powers). These choices propagate through complicated physics — pump–signal coupling, ISRS tilt, nonlinear interference — and ultimately determine the achievable Shannon-Hartley throughput \(T(\mathbf{P}_p)\) summed over all lightpaths.

To find the pump configuration that maximises throughput we have two options:

- **Black-box search** (grid scan, Bayesian optimisation, evolutionary methods). Each trial costs a full forward simulation; convergence is slow and scales poorly with \(M\).
- **Gradient ascent** (Adam, L-BFGS, etc.). Each step costs roughly two forward simulations (forward + reverse pass) and converges in tens of iterations even for \(M \sim 10\)–\(20\) pumps with bounds and budget constraints.

Gradient ascent requires \(\nabla_{\mathbf{P}_p} T\). If the simulation pipeline is built entirely from JAX primitives, this gradient is free — you write the forward pass and JAX's autodiff machinery gives you the backward pass automatically. The DRA pipeline, however, contains two iterative solvers that JAX cannot differentiate natively. The rest of this page explains how we work around that.

---

## A brief detour: how automatic differentiation works

This subsection is short and pragmatic — just enough to understand what "differentiable" means in this context.

### The chain rule, applied automatically

When you write a JAX function `f(x)`, the library traces every elementary operation (`+`, `*`, `exp`, matrix multiplications, `where`, …) and builds a **computational graph**. Each elementary operation has a known local derivative. To compute \(\partial f / \partial x\), JAX walks the graph and applies the chain rule:

\[
\frac{\partial f}{\partial x} \;=\; \frac{\partial f}{\partial u_n} \cdot \frac{\partial u_n}{\partial u_{n-1}} \cdots \frac{\partial u_1}{\partial x}.
\]

This works in two modes:

- **Forward mode** (in JAX: `jvp`, the Jacobian-vector product): pushes a tangent \(\dot{x}\) forward through the graph. Cheap when the input is low-dimensional and the output is high-dimensional.
- **Reverse mode** (in JAX: `vjp`, the vector-Jacobian product, also wrapped by `jax.grad`): pulls a cotangent \(\bar{f}\) backward through the graph. Cheap when the input is high-dimensional and the output is a scalar — exactly our case (many pump powers, one throughput number).

### Where it breaks down

Three things will block automatic differentiation:

1. **Non-JAX code.** A call to `scipy.optimize.minimize`, a NumPy array operation, or any Python control flow that depends on a traced value's *content* — JAX cannot see inside these and has no derivative rule.
2. **Iterative solvers without convergence rules.** Even if you implement an iterative method in pure JAX, naively differentiating through every iteration ("unrolling") is memory-hungry and numerically fragile. There are better ways (next section).
3. **Non-smooth or singular operations.** Things like `argmax`, `where(cond, …)` with traced `cond`, or matrix solves where the matrix becomes singular at the operating point.

When we hit one of these, we have to provide a custom derivative rule using `jax.custom_vjp` (or `jax.custom_jvp`). The rule tells JAX: "Trust me, when you need the gradient through this block, here's the formula." JAX then uses our formula instead of trying to differentiate the block automatically.

### The implicit function theorem

Many of the things we want gradients through are defined *implicitly*: \(\mathbf{x}^\star\) is whatever value satisfies an equation \(\mathbf{F}(\mathbf{x}^\star, \mathbf{y}) = \mathbf{0}\), parameterised by some upstream input \(\mathbf{y}\). The implicit function theorem (IFT) says that if \(\partial \mathbf{F}/\partial \mathbf{x}\) is invertible at the solution, then

\[
\frac{\partial \mathbf{x}^\star}{\partial \mathbf{y}} \;=\; -\!\left(\frac{\partial \mathbf{F}}{\partial \mathbf{x}}\right)^{-1}\! \frac{\partial \mathbf{F}}{\partial \mathbf{y}}.
\]

The point is: you can compute \(\partial \mathbf{x}^\star / \partial \mathbf{y}\) using only *one* Jacobian of the equation \(\mathbf{F}\), evaluated at the converged point — without ever differentiating through the iterations of the solver that found \(\mathbf{x}^\star\). This is what makes implicit-function-style gradients efficient and stable.

We use this trick once, explicitly, for the boundary-value solve in stage 2 of the pipeline.

---

## The pipeline at a glance

The DRA simulation maps pump powers to per-channel SNR (and ultimately throughput) in four stages:

| Stage | What it does | Implemented by |
|------:|---|---|
| 1 | Solve coupled Raman ODE → numerical signal-power profile \(P_i(z)\) | `jax.experimental.ode.odeint` |
| 2 | Find \(z=0\) backward-pump power matching the prescribed \(z=L\) target | scipy TNC + `custom_vjp` (IFT backward) |
| 3 | Fit an analytical 5-parameter ansatz to \(P_i(z)\) → fit parameters \(\theta_i\) | `jaxopt.LevenbergMarquardt` (forward only) |
| 4 | Substitute fit parameters into closed-form ISRS GN integrals → NLI, ASE, SNR, throughput | Pure JAX |

The remainder of this page works through each stage.

---

## Stage 1: The Raman ODE

A Raman-amplified fibre carries \(N\) wavelength-division-multiplexed (WDM) signal channels and \(M\) backward-propagating pump lasers. Each carrier's power along the fibre is governed by a coupled ODE:

\[
\frac{dP_i}{dz} \;=\; \Bigl(\sum_j g_R(f_i, f_j)\,P_j \;-\; \alpha_i\Bigr)\, P_i,
\]

where:

- \(\alpha_i\) is the attenuation coefficient. For forward-propagating signals the sign is positive (signals are attenuated as \(z\) increases). For *backward*-propagating pumps we flip the sign — they propagate against \(z\), so in the forward integration they appear to *gain* energy.
- \(g_R(f_i, f_j)\) is the Raman gain coefficient between carriers at frequencies \(f_j\) (donor) and \(f_i\) (receiver). XLRON uses the **triangular Raman approximation** \(g_R = C_r |\Delta f|\) with a 15 THz cutoff, valid across the C+L band.
- The signal–signal block of \(g_R\) is **zeroed out** when integrating the ODE. Signal–signal Raman tilt is handled separately by the GN model's perturbative ISRS formula; including it here would double-count.

The ODE is integrated by `jax.experimental.ode.odeint` (Dormand–Prince adaptive method). `odeint` ships with its own `custom_vjp` rule based on the **adjoint method** — this means JAX *can* take derivatives of the ODE solution with respect to its inputs (initial conditions, ODE parameters) automatically. Stage 1 is therefore differentiable out of the box.

---

## Stage 2: The boundary-value problem

The pumps are *backward-propagating*, which means we know the pump power at the **far end** of the span (\(z=L\)) — that's the value we set at the launch — but `odeint` integrates forward from \(z=0\). To use a single forward-integration ODE solver, we must first answer: *"What pump power at \(z=0\) would, after the coupled propagation, produce the prescribed power at \(z=L\)?"*

This is the classical two-point BVP we mentioned earlier. We solve it as an inner optimisation:

\[
\mathbf{x}^\star \;=\; \arg\min_{\mathbf{x}}\; \big\| \mathbf{P}_p(L; \mathbf{x}) - \mathbf{P}_p^{\text{target}} \big\|^2, \qquad \mathbf{x} \in [10^{-7}, 10^3]^M\, \mathrm{W},
\]

where \(\mathbf{P}_p(L; \mathbf{x})\) is the pump power at \(z=L\) obtained by running the forward ODE with starting pump power \(\mathbf{x}\) at \(z=0\). The constraint \(\mathbf{x} \ge 10^{-7}\) prevents zero pump power (which would cause a singularity), and \(\mathbf{x} \le 10^{3}\) guards against runaway iterations.

### TNC: the solver

We minimise this with `scipy.optimize.minimize(method="TNC")`, a truncated Newton method with constraints — a quasi-Newton optimiser with native support for box constraints. It uses gradient information (which JAX provides by autodiff through `odeint`), approximates Hessian-vector products, and "truncates" its inner conjugate-gradient solve for each Newton step. It is an excellent fit for smooth nonlinear problems with simple bounds.

The catch: scipy lives outside JAX. JAX has no derivative rule for "the output of a scipy optimisation as a function of its target". So if we naively call scipy inside a JAX-traced function, the gradient through this block will fail.

### IFT to the rescue

We don't actually need to differentiate through TNC's iterations — we only need the derivative of the *converged* solution \(\mathbf{x}^\star\) with respect to the prescribed target. At the optimum the residual is zero:

\[
\mathbf{P}_p\bigl(L;\,\mathbf{x}^\star\bigr) \;=\; \mathbf{P}_p^{\text{target}}.
\]

This is an implicit equation defining \(\mathbf{x}^\star\) as a function of the target. Differentiating both sides with respect to the target and solving:

\[
\frac{\partial \mathbf{x}^\star}{\partial \mathbf{P}_p^{\text{target}}} \;=\; J^{-1}, \qquad J \;=\; \frac{\partial \mathbf{P}_p(L)}{\partial \mathbf{x}}\bigg|_{\mathbf{x}^\star}.
\]

\(J\) is the Jacobian of a *forward* ODE evaluation, which JAX can compute directly via autodiff through `odeint`. So the cost of the gradient is one forward ODE run (already done) plus one Jacobian call plus a single linear solve.

We package this as a `jax.custom_vjp`:

```python
@jax.custom_vjp
def _solve_bw_boundary(target_bw_pow):
    # forward: call scipy TNC, return converged pump-at-z=0
    ...

def _solve_bw_fwd(target_bw_pow):
    bw_z0 = _solve_bw_boundary(target_bw_pow)
    return bw_z0, (bw_z0, target_bw_pow)

def _solve_bw_bwd(res, g_bw_z0):
    bw_z0, target_bw_pow = res
    J = jax.jacobian(_bw_at_L)(bw_z0)            # forward AD through odeint
    g_target = jnp.linalg.solve(J.T, g_bw_z0)    # IFT-derived VJP
    return (g_target,)

_solve_bw_boundary.defvjp(_solve_bw_fwd, _solve_bw_bwd)
```

To the rest of the pipeline, the BVP solve looks like a single differentiable operator. Inside, scipy does the heavy lifting on the forward pass and IFT supplies the backward pass.

---

## Stage 3: Fitting an analytical profile

### What is an "ansatz"?

An **ansatz** (German for "approach" or "starting point") is a *guessed* parametric form for an unknown function. You don't derive it from first principles — you propose it because it has the right qualitative shape and is mathematically convenient, then fit its free parameters to data. Common examples: variational ansätze in quantum mechanics, exponential decay ansätze for transient signals, polynomial ansätze for boundary-layer profiles.

Why an ansatz here? The ODE in stage 1 gives us the signal power profile \(P_i(z)\) as a numerical curve sampled at, say, 200 \(z\)-points per channel per span. The downstream GN-model NLI integrals require the profile in a *closed-form analytical* shape so the integrals can be evaluated symbolically. So we need a smooth analytical formula whose shape matches the numerical \(P_i(z)\) closely enough — that's the ansatz.

### The Semrau profile

XLRON uses the five-parameter ansatz of Semrau *et al.*:

\[
\rho_{\text{semi}}(z;\, C_f, a_f, C_b, a_b, a) \;=\; e^{-az}\bigl[1 - x_i(z)\,\Delta f_i\bigr],
\]

\[
x_i(z) \;=\; C_f\, P_f\, L_{\text{eff}}^{f}(z, a_f) \;+\; C_b\, L_{\text{eff}}^{b}(z, a_b),
\]

with effective lengths

\[
L_{\text{eff}}^{f}(z, a_f) \;=\; \frac{1 - e^{-a_f z}}{a_f}, \qquad
L_{\text{eff}}^{b}(z, a_b) \;=\; \frac{e^{-a_b(L-z)} - e^{-a_b L}}{a_b}.
\]

Interpretation of the five parameters per channel:

| Parameter | Captures |
|---|---|
| \(C_f\) | Forward-pump Raman coupling strength |
| \(a_f\) | Forward-pump effective decay rate |
| \(C_b\) | Backward-pump Raman coupling strength |
| \(a_b\) | Backward-pump effective decay rate |
| \(a\) | Net effective attenuation along the fibre |

We fit these parameters by minimising the squared residual between \(\rho_{\text{semi}}\) and the normalised ODE profile \(\rho_i(z) = P_i(z) / P_i(0)\) at all sampled \(z\). This is a standard nonlinear least-squares problem — one fit per channel, performed in parallel by `jax.vmap`.

### LM: the right tool for nonlinear least-squares

We use `jaxopt.LevenbergMarquardt` (LM), a Gauss–Newton-style optimiser specialised for nonlinear least-squares. LM exploits the residual structure: it approximates the Hessian by \(J^\top J\) plus a damping term \(\lambda\,\mathrm{diag}(J^\top J)\). This makes it converge quadratically near the optimum on well-conditioned problems while remaining robust when the initial guess is far from the solution.

In principle, jaxopt can give you smooth gradients through the LM optimum via implicit differentiation (the same IFT trick we used for the BVP — at a least-squares optimum, \(J^\top r = 0\) is the implicit equation). So why do we *not* rely on this?

### Why we stop the gradient through the fit

The Semrau ansatz contains the factor \((1 - x_i(z)\,\Delta f_i)\). For channels far from the average pump frequency — i.e. large \(|\Delta f_i|\) — three things go wrong simultaneously:

1. The factor can drift toward zero or even negative, leaving the regime where the linearised tilt approximation is physically meaningful.
2. The fit Jacobian becomes ill-conditioned (parameter sensitivity scales with \(\Delta f_i\)).
3. The minimum becomes shallow or non-unique. At a near-degenerate optimum, \((J^\top J)^{-1}\) blows up and the IFT-style gradient produces NaNs.

Switching to a different optimiser (trust-region reflective, L-BFGS, etc.) does not fix this — the issue is the *model*, not the solver.

The pragmatic fix is to use the LM fit *only in the forward pass* and let the gradient flow through a different route. In code:

```python
ch_norm_no_grad = jax.lax.stop_gradient(ch_norm)        # detach LM input
multipliers = lm_solver.run(jnp.ones(5), …, ch_norm_no_grad, …).params
fit_5 = jax.lax.stop_gradient(init_params * multipliers)  # detach LM output
```

`jax.lax.stop_gradient` is the JAX primitive that says "in the backward pass, treat this value as a constant". It does not affect the forward pass at all — the LM fit still happens, the fitted parameters still feed into the NLI integrals downstream — but JAX's autodiff sees a zero gradient through this block.

### The surrogate gradient: \(G_i\)

If the gradient does not flow through the LM fit, how does any pump-power sensitivity reach the throughput? Through a separate, well-conditioned quantity computed alongside the profile: the **per-channel Raman gain** at the span endpoint,

\[
G_i \;=\; \frac{P_i(L)}{P_i(0)\, e^{-\alpha L}}.
\]

\(G_i\) is a single scalar per channel, not a separate model — it is just a ratio of two ODE outputs. \(G_i = 1\) means "the pumps did nothing"; \(G_i = 10\) means "the pumps gave 10 dB of net Raman gain on top of passive attenuation". It is differentiable with respect to pump powers because both numerator and denominator are differentiable through `odeint`, and the ratio cannot become singular as long as \(P_i(0) > 0\).

\(G_i\) feeds directly into the **ASE noise** calculation via the Friis hybrid noise figure (see [GN Model § DRA](gn_model.md#ase-noise-with-dra)):

\[
\mathrm{NF}_{\text{DRA}} = \frac{1}{G_i} + 2 n_{sp}\!\left(1 - \frac{1}{G_i}\right), \qquad
\mathrm{NF}_{\text{hybrid}} = \mathrm{NF}_{\text{DRA}} + \frac{\mathrm{NF}_{\text{EDFA}} - 1}{G_i}.
\]

So in the gradient graph, the path from pump powers to ASE noise is fully differentiable; the path from pump powers to NLI is forward-only.

The fit parameter array stored in `EnvParams.raman_fit_params` has shape `(6, num_channels, max_spans)`. Rows 0–4 are the LM-fitted \([C_f, a_f, C_b, a_b, a]\); row 5 is \(G_i\). Both are derived from the same ODE solution but enter the SNR calculation through different routes.

---

## Stage 4: NLI, ASE, SNR, throughput

The remaining stages are pure JAX and fully differentiable. Per channel \(i\), nonlinear interference (NLI) and amplified spontaneous emission (ASE) noise are computed as

\[
P_{\text{NLI},i} = P_i^3 \cdot \eta_n(C_f, a_f, C_b, a_b, a),
\qquad
P_{\text{ASE},i} = N_{\text{spans}} \cdot \mathrm{NF}_{\text{hybrid}}(G_i) \cdot G_{\text{total}} \cdot h f_i B_i,
\]

and combined into the per-channel signal-to-noise ratio (SNR) and the network total throughput \(T\):

\[
\mathrm{SNR}_i \;=\; \frac{P_i}{P_{\text{ASE},i} + P_{\text{NLI},i}}, \qquad
T \;=\; \sum_{\text{lightpaths}\,p} 2 B_p \log_2\!\left(1 + \mathrm{SNR}_p\right) \cdot (1 - \mathrm{FEC}_{\text{overhead}}),
\]

where \(\mathrm{FEC}_{\text{overhead}}\) is the forward error correction (FEC) overhead fraction.

Both \(P_{\text{NLI}}\) and \(P_{\text{ASE}}\) appear in the SNR denominator. In the *forward* pass, both depend on pump powers (through the LM fit and through \(G_i\) respectively). In the *backward* pass, only the \(G_i\) path survives because the LM fit is stop-gradiented.

---

## Putting it all together: how the gradient actually flows

```
                       FORWARD PASS                                  BACKWARD PASS
                       ────────────                                  ─────────────

    pump_pow_bw                                            ∂T/∂pump_pow_bw
        │                                                          ▲
        ▼                                                          │
  custom_vjp(scipy TNC)  ◄── BVP solve ──►   IFT: J⁻ᵀ ∇            │
        │                                                          │
        ▼                                                          │
      odeint  ◄── Raman ODE ──►  adjoint method (JAX built-in)     │
        │                                                          │
        ├──────────────────► raman_gain_linear (G_i, row 5)  ──────┤  carries gradient
        │                              │                            │
        │                              ▼                            │
        │                      Friis hybrid NF                      │
        │                              │                            │
        │                              ▼                            │
        │                         P_ASE noise ─────────┐            │
        │                                              │            │
        └──────────► LM fit (rows 0–4)                 │            │
                          │                            │            │
                  STOP_GRADIENT                        │            │
                          │                            ▼            │
                          ▼                      SNR = P/(ASE+NLI) ─┘
                    Semrau ansatz                     │
                          │                           │
                          ▼                           ▼
                      P_NLI ◄──────── feeds ────► throughput T
```

In the forward pass, both NLI and ASE are computed exactly — the SNR you log is honest. In the backward pass, only the \(G_i\) path between pumps and ASE is alive; the NLI sensitivity is set to zero by `stop_gradient`.

This makes the gradient a **surrogate**: the optimiser behaves as if it were maximising "throughput, holding NLI fixed at its current value". After each optimiser step, NLI is re-evaluated at the new pump configuration, so the *forward objective* always reflects the true SNR. Only the *direction* of each step is approximate.

---

## Why this approximation is acceptable

Two reasons:

1. **ASE is the dominant pump-sensitive term.** Distributed Raman amplification exists precisely to lower ASE — that's its physical purpose. \(\partial P_{\text{ASE}}/\partial \mathbf{P}_p\) is large; \(\partial P_{\text{NLI}}/\partial \mathbf{P}_p\) is comparatively small (it acts second-order through profile-shape changes). Capturing the ASE sensitivity alone gives a gradient direction that is well-aligned with the true gradient.
2. **The forward objective is exact.** At every optimiser step we evaluate the *full* ISRS GN-model SNR including correct NLI. So if the optimiser converges, the converged pumps maximise the *true* throughput. The surrogate only affects the path taken there, not the destination.

This is the same kind of trade-off used in policy gradient methods (the REINFORCE trick), in straight-through estimators for discrete operations, and in many physics-informed learning settings. It is honest and reviewable: the simulator tells you the right number, and the optimiser climbs in a direction that may not be the steepest but is empirically good enough.

---

## Suggested framing for papers

> The full pipeline — ODE solve, BVP boundary condition, profile fit, GN integrals, SNR, throughput — is end-to-end differentiable. Two steps require care:
>
> *(i)* The two-point boundary-value problem for the backward pump's \(z=0\) initial condition is solved by scipy's TNC with bounds. A custom backward pass derived from the implicit function theorem provides the gradient analytically:
> \[
> \frac{\partial \mathbf{x}^\star}{\partial \mathbf{P}_p^{\text{target}}} = J^{-1}, \qquad J = \tfrac{\partial \mathbf{P}_p(L)}{\partial \mathbf{x}}\big|_{\mathbf{x}^\star},
> \]
> avoiding the need to unroll TNC iterations.
>
> *(ii)* The Levenberg–Marquardt fit of the five-parameter Semrau profile is treated as a forward-only block. Implicit-differentiation gradients through the LM optimum are numerically unstable for channels far from the average pump frequency, where the linearised tilt ansatz becomes ill-conditioned. We therefore route the pump-power gradient through a separate ODE-derived per-channel Raman gain factor \(G_i = P_i(L)/[P_i(0)\,e^{-\alpha L}]\) which carries the dominant sensitivity of throughput to pump configuration via the ASE noise figure. The fit parameters still enter the *forward* NLI calculation; the *backward* pass treats them as constants.
>
> Together, these two devices yield a tractable surrogate gradient with an exact forward objective. To our knowledge, this is the first end-to-end differentiable optimisation of Raman pump powers driven by a wideband ISRS GN model.

---

## Code references

| Component | File | Function |
|---|---|---|
| Non-differentiable fit (uses scipy directly) | `xlron/environments/gn_model/isrs_gn_model_dra.py` | `fit_dra_params_triangular` |
| Differentiable fit (used by pump optimisation) | `xlron/environments/gn_model/isrs_gn_model_dra.py` | `fit_dra_params_jax` |
| BVP `custom_vjp` definition | `xlron/environments/gn_model/isrs_gn_model_dra.py` | `_solve_bw_boundary` |
| LM fit (with `stop_gradient` on inputs and outputs) | `xlron/environments/gn_model/isrs_gn_model_dra.py` | inside `fit_dra_params_jax` |
| NLI integral consumer of fit parameters | `xlron/environments/gn_model/isrs_gn_model_dra.py` | `gn_model_dra` |
| SNR consumer of \(G_i\) (Friis hybrid NF) | `xlron/environments/gn_model/isrs_gn_model_dra.py` | `get_snr_dra` |
| End-to-end throughput objective | `experimental/validation/pump_optimization.py` | `make_throughput_objective` |

---

## See Also

- [GN Model Physical Layer](gn_model.md) — full description of XLRON's ISRS GN model and the DRA configuration parameters.
- [GN Model § Distributed Raman Amplification](gn_model.md#distributed-raman-amplification-dra) — physical layer setup for DRA, including how the fit parameters are stored and consumed.
