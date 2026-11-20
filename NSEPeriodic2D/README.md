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
