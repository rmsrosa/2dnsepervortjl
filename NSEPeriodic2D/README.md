# NSEPeriodic2D

Pseudo-spectral simulation of the two-dimensional incompressible Navier-Stokes equations in vorticity form,

$$
\partial_t\omega + \mathbf{u}\cdot\nabla\omega = \nu\Delta\omega + g,
$$

on the periodic domain $[0, L_x) \times [0, L_y)$, with a steady forcing whose curl is $g$. It collects, in reusable form, the methods developed and compared in the notes of this repository ([website](https://rmsrosa.github.io/2dnsepervortjl/)):

* the vorticity is represented by its real Fourier transform, dealiased by the 2/3 rule, and the nonlinear term is computed in Basdevant's form, with four FFTs per evaluation, without allocations;
* problems are built for the solvers of [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl), in three formulations: for explicit methods, for exponential and IMEX methods (split problem, with the viscous term as a diagonal operator) and for implicit methods (with the analytic Jacobian-vector product, matrix-free);
* `simulate` integrates and samples the diagnostics (energy, enstrophy, palinstrophy, the Taylor-like wavenumber $\kappa_\sigma$, the energy spectrum, the velocity maxima and the CFL number) without storing the states.

The package is not registered. The solvers are not a dependency: load the OrdinaryDiffEq packages you need.

The section "The NSEPeriodic2D package" of the [website](https://rmsrosa.github.io/2dnsepervortjl/) explains and uses the package, with plots.

## Installation

From Julia, either from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/rmsrosa/2dnsepervortjl", subdir = "NSEPeriodic2D")
```

or from a local clone of the repository:

```julia
using Pkg
Pkg.develop(path = "path/to/2dnsepervortjl/NSEPeriodic2D")
```

Then add the solver packages, e.g. `Pkg.add(["OrdinaryDiffEq", "OrdinaryDiffEqExponentialRK"])`. It requires Julia 1.13 or later.

## Usage

```julia
using NSEPeriodic2D
using OrdinaryDiffEq: Tsit5

# Domain and discretization: periods Lx, Ly and Nx × Ny points (2/3 dealiasing by default)
grid = Grid(Lx = 2π, Ly = 2π, Nx = 128, Ny = 128)

# Viscosity and curl of the forcing: a function g(x, y), a physical or spectral array,
# or a band of modes with random phases
g_hat, modes = band_forcing(grid, 2, 4; rms = 1.0)
model = NSEModel(grid; ν = 4e-3, forcing = g_hat)

# Initial vorticity: a function ω(x, y), a physical or spectral array, e.g.
# stokes_state(model) = (1/ν)A⁻¹g, random_vorticity(grid, κmin, κmax; energy),
# or vorticity_from_velocity(grid, u, v)
prob = NSEProblem(model, stokes_state(model), (0.0, 100.0))

# Integrate, sampling the diagnostics every save_every time units
res = simulate(prob, Tsit5(); reltol = 1e-2, abstol = 1e-5, save_every = 0.5)

res.t, res.energy, res.enstrophy, res.palinstrophy    # time series
kappa_sigma(res.enstrophy, res.palinstrophy)          # √(⟨P⟩/⟨Z⟩)
batch_mean_ci(res.energy)                             # mean and 95% confidence half-width
res.κ, res.spectrum                                   # time-averaged energy spectrum
ω = vorticity(model, res.w_hat_final)                 # physical vorticity at the end
u, v = velocity(model, res.w_hat_final)
```

`solve(prob, alg; kwargs...)` also works, returning the usual solution of DifferentialEquations.jl; convert its states with `state_to_vorticity_hat(prob, sol.u[end])`.

### Choosing the formulation and the method

Following the conclusions of the notes ("Stiff solvers, Jacobians and automatic differentiation" and "Time stepping for weakly turbulent flows"):

| Regime | Formulation | Methods |
| :--- | :--- | :--- |
| Advection-dominated (small viscosity, turbulent or chaotic flows) | `NSEProblem(model, ω0, tspan)` (`kind = :full`) | `Tsit5()`, which follows the bursts automatically; or a fixed step with `SSPRK33()` or `RK4()` (`dt`, `adaptive = false`) |
| Viscous term stiff (large viscosity, high resolution, steady states) | `kind = :split` | `ETDRK4()` with a fixed `dt`; IMEX methods such as `CNAB2(linsolve = DiagonalFactorization())` |
| Viscous, converging to steady states, large steps | `kind = :implicit` | `QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false)` and other implicit methods, matrix-free with the analytic Jacobian-vector product |

With `Tsit5()`, a loose tolerance (`reltol = 1e-2`) is cheap and fine for a warm-up, but, in a test with the forced flow of the docs, it biased the time-averaged energy by about 5%, while `reltol = 1e-4` agreed with fixed-step `SSPRK33()` runs; check the statistics against a tighter tolerance.

For a fixed step, `suggest_dt(model, ω̂; σ, umax, vmax)` gives `dt = σ / (κx_max umax + κy_max vmax)`. In the notes, RK4, ETDRK4 and SSPRK33 were stable up to $\sigma \approx 4$ and blew up at $\sigma = 6$, with the velocity maxima over a long run (bursts included), and CNAB2 needs a step about five times smaller. With `simulate`, `dt` must divide `save_every`. ETDRK4 in OrdinaryDiffEq cannot change its step during a run.

Notes:

* `CNAB2` must be given `linsolve = DiagonalFactorization()` (from LinearSolve); with the default linear solver it builds and factorizes a dense matrix at every step.
* A model holds preallocated work arrays and FFTW plans, so it must not be shared by computations running concurrently; build one per thread or task.
* The forcing is steady.

## Stochastic and random forcing

The package also solves

$$
\mathrm{d}\mathbf{u} + (\mathbf{u}\cdot\nabla\mathbf{u} + \nabla p)\,\mathrm{d}t = \nu\Delta\mathbf{u}\,\mathrm{d}t + \mathbf{f}\,\mathrm{d}t + N_t\,\mathrm{d}t + \sigma\Phi\,\mathrm{d}W_t,
$$

with a random forcing $N_t$ (an arbitrary process) and an additive noise $\sigma\Phi\,\mathrm{d}W_t$ on a finite set of modes:

```julia
modes = fourier_modes(grid, 3, 5)                  # divergence-free Fourier modes, unit mean square
noise = AdditiveNoise(modes; σ = 0.06)             # σ Φ dWₜ; energy_injection_rate(noise) = ½σ² Σ aⱼ²
path  = ou_path(4, (0.0, 100.0); θ = 0.5, s = 0.3, dt = 0.01)   # or compound_poisson_path, lorenz_path, any η(t)
model = NSEModel(grid; ν = 1e-2, noise, random_forcing = RandomForcing(fourier_modes(grid, [(1, 2), (2, 1)]), path))
prob  = NSEProblem(model, zeros(grid.Nx, grid.Ny), (0.0, 100.0); kind = :sde)
res   = simulate(prob, StochasticETDRK4(); dt = 0.01, save_every = 0.5, seed = 1)
```

* `kind = :sde` builds an `SDEProblem` (real state, sparse noise matrix) for the solvers of StochasticDiffEq, e.g. `EM()`, `SRA1()`, `SOSRA()`.
* The package has its own fixed-step schemes, used with `simulate`: `StochasticETDRK4()`, `StochasticExpEuler()` (both integrate the viscous term and the stochastic convolution exactly) and `SplitStochasticRK(:rk4)` or `SplitStochasticRK(:ssprk3)` (Strang splitting with the noise).
* A random forcing alone is a time-dependent forcing, so the deterministic solvers and formulations also apply. For a compound Poisson path, pass `tstops = jump_times(path)` to adaptive solvers.

In a comparison against the exact budgets $2\nu\langle Z\rangle = \varepsilon$ and $2\nu\langle P\rangle = \eta$ (website section "Stochastic and random forcing"):
* `StochasticETDRK4()` and `SplitStochasticRK(:rk4)` were unbiased within about 1% up to a CFL number of about 4.4, and were the cheapest.
* `EM()` and `StochasticExpEuler()` were biased by a few percent even with much smaller steps.
* `SRA1()` and `SOSRA()` are practical only with very loose tolerances (`reltol = abstol = 1`); with usual tolerances their steps collapse.

## Examples

The folder [`examples/`](examples/) has two scripts, with their own environment (run them from that folder with `julia --project=. script.jl`):

* `steady_state.jl`: large viscosity and single-mode forcing, converging to the steady state, with ETDRK4 and with an implicit method (about half a minute);
* `forced_flow.jl`: band forcing from the Stokes state, warm-up with `Tsit5()`, then statistics with a fixed-step `SSPRK33()` (a few minutes).

## Tests

```julia
using Pkg
Pkg.activate("path/to/2dnsepervortjl/NSEPeriodic2D")
Pkg.test()
```

The tests check:
* the spectral operators and the diagnostics against exact formulas, on a rectangular domain;
* that the Stokes state is a steady state for a forcing on a single shell;
* that the Jacobian-vector product equals the centered difference quotient (exactly, the equation being quadratic);
* that energy and enstrophy are conserved without viscosity and forcing;
* that the three formulations give the same solution;
* the behaviour of `simulate`, including the detection of blow-ups.
