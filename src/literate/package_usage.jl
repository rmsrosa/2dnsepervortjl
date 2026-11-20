# # Using the NSEPeriodic2D package

# The methods developed in the previous sections are collected, in reusable form, in the package [NSEPeriodic2D](https://github.com/rmsrosa/2dnsepervortjl/tree/main/NSEPeriodic2D), in the folder `NSEPeriodic2D/` of the repository. The package is not registered, but it can be installed directly from GitHub with
#
# ```julia
# using Pkg
# Pkg.add(url = "https://github.com/rmsrosa/2dnsepervortjl", subdir = "NSEPeriodic2D")
# ```
#
# It solves the vorticity equation
#
# $$
# \partial_t\omega + \mathbf{u}\cdot\nabla\omega = \nu\Delta\omega + g
# $$
#
# on a domain periodic in $x$ and in $y$, with periods $L_x$ and $L_y$, where $g$ is the curl of a steady forcing. The vorticity is represented by its real Fourier transform, dealiased with the 2/3 rule, and the nonlinear term is computed in Basdevant's form, with four FFTs per evaluation and no allocations, as in [Approximations of the bilinear term](/pages/literated/tests_bilinear).
#
# The package builds the problems for the solvers of [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl), which are not a dependency: we load the solvers we need. The problems come in three formulations, following the conclusions of the previous sections:
#
# * `kind = :full`, for explicit methods, adaptive (`Tsit5()`) or with a fixed step (`SSPRK33()`, `RK4()`): the choice for advection-dominated flows (see [Time stepping for weakly turbulent flows](/pages/turbulence_timestepping));
# * `kind = :split`, with the viscous term as a diagonal operator, for exponential integrators (`ETDRK4()`) and IMEX methods: the choice when the viscous term is stiff (see [Exponential time differencing with ETDRK4](/pages/literated/evolution_etdrk4));
# * `kind = :implicit`, with the analytic Jacobian-vector product, for implicit methods used matrix-free (see [Implicit solvers with the analytic Jacobian-vector product](/pages/literated/evolution_jvp)).
#
# The function `simulate` integrates a problem and samples a few diagnostics along the way, without storing the states. This page shows the building blocks; the [next page](/pages/literated/package_forced_flow) uses them for a forced flow.

# ## Packages

using NSEPeriodic2D
using OrdinaryDiffEq: Tsit5
using OrdinaryDiffEqExponentialRK: ETDRK4
using OrdinaryDiffEqBDF: QNDF
using LinearSolve: KrylovJL_GMRES
using Random: Xoshiro
using Plots

# Physical fields are arrays `f[ix, iy]`, with `x` along the first dimension, so we transpose them for `heatmap`:

plotfield(grid, f; kwargs...) = heatmap(grid.x, grid.y, permutedims(f);
    aspect_ratio = 1, xlims = (0, grid.Lx), ylims = (0, grid.Ly), xlabel = "x", ylabel = "y",
    color = :balance, clims = (-1, 1) .* maximum(abs, f), titlefont = 10, kwargs...)
nothing

# ## The building blocks

# The domain and its discretization: the periods `Lx` and `Ly` and the number of points `Nx × Ny`.

grid = Grid(Lx = 2π, Ly = 2π, Nx = 64)

# The model: the viscosity and the curl of the forcing, which can be given as a function `g(x, y)`, as an array of values on the grid, or as a spectral array. Here, a single mode:

model = NSEModel(grid; ν = 1.0, forcing = (x, y) -> 20cos(x + 2y))

# The initial condition can also be given as a function, a physical or a spectral array, or built with the helpers `random_vorticity`, `stokes_state` and `vorticity_from_velocity`. Here, a random field with modes in the band $1 \leq |\kappa| \leq 8$ and kinetic energy $5$:

ω0 = random_vorticity(grid, 1, 8; energy = 5.0, rng = Xoshiro(1))
nothing

plotfield(grid, vorticity(model, ω0); title = "initial vorticity")

# ## A viscous flow converging to the steady state

# Since the forcing has a single mode, the Stokes state $\omega_\infty = (1/\nu)A^{-1}g$ is a steady state of the full equation, and, at this large viscosity, it attracts all the solutions:

ω∞ = stokes_state(model)
nothing

# The viscous term is stiff here, so we use the split formulation and ETDRK4, with a fixed step. The diagnostics are sampled every `save_every` time units, and we also keep snapshots of the vorticity:

prob = NSEProblem(model, ω0, (0.0, 8.0); kind = :split)

res = simulate(prob, ETDRK4(); dt = 0.01, save_every = 0.1, snapshot_every = 0.1)

# The result holds the time series of the energy $E = \tfrac12\langle|\mathbf{u}|^2\rangle$, the enstrophy $Z = \tfrac12\langle\omega^2\rangle$, the palinstrophy $P = \tfrac12\langle|\nabla\omega|^2\rangle$, the velocity maxima and the CFL number, the time-averaged energy spectrum, the snapshots and the final state:

plot(res.t, [res.energy res.enstrophy res.palinstrophy], yscale = :log10, xlabel = "t",
    label = ["E" "Z" "P"], title = "Energy, enstrophy and palinstrophy", titlefont = 10)

# The distance to the steady state, measured by the $L^2$ norm of the velocity, eventually decays at the rate $e^{-\nu\kappa_{\min}^2 t} = e^{-t}$ of the slowest modes, those with $|\kappa| = 1$:

dist = [sqrt(2energy(model, w .- ω∞)) for (t, w) in res.snapshots]
nothing

plot(res.t, dist, yscale = :log10, xlabel = "t", label = "‖u(t) - u∞‖",
    title = "Distance to the steady state", titlefont = 10)
plot!(res.t, dist[1] .* exp.(-(res.t .- res.t[1])), linestyle = :dash, label = "e^{-t}")

#

plotfield(grid, vorticity(model, res.w_hat_final); title = "vorticity at t = 8")

# The same problem with an implicit method, used matrix-free with the analytic Jacobian-vector product, is set up with `kind = :implicit`:

probi = NSEProblem(model, ω0, (0.0, 8.0); kind = :implicit)

resi = simulate(probi, QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false); save_every = 0.1)

# The two final states agree within the tolerance of the implicit method:

sqrt(energy(model, resi.w_hat_final .- res.w_hat_final) / energy(model, res.w_hat_final))

# The usual `solve` of DifferentialEquations.jl also works with these problems, returning the usual solution; its states are converted back to the spectral vorticity with `state_to_vorticity_hat`:

sol = solve(prob, ETDRK4(); dt = 0.01, saveat = 2.0)

sol.t

#

w_end = state_to_vorticity_hat(prob, sol.u[end])
sqrt(energy(model, w_end .- res.w_hat_final) / energy(model, res.w_hat_final))

# ## A decaying flow in a rectangular domain

# The periods in each direction may differ. Here, a domain twice as long as it is wide, with no forcing and a small viscosity, starting from a random field with modes in the band $4 \leq |\kappa| \leq 8$:

grid2 = Grid(Lx = 4π, Ly = 2π, Nx = 128, Ny = 64)

model2 = NSEModel(grid2; ν = 1e-3)

ω0 = random_vorticity(grid2, 4, 8; energy = 0.5, rng = Xoshiro(2))
nothing

prob2 = NSEProblem(model2, ω0, (0.0, 60.0))
nothing

# The viscous term is not stiff, and advection dominates, so we use an adaptive explicit method:

res2 = simulate(prob2, Tsit5(); reltol = 1e-4, abstol = 1e-7, save_every = 0.5, snapshot_every = 20.0)

# The vortices merge into larger ones, while the energy is nearly conserved and the enstrophy decays, as expected in two-dimensional flows:

plot(
    plotfield(grid2, vorticity(model2, ω0); title = "t = 0"),
    [plotfield(grid2, vorticity(model2, w); title = "t = $(round(Int, t))") for (t, w) in res2.snapshots]...,
    layout = (2, 2), size = (800, 450), colorbar = false
)

#

plot(res2.t, [res2.energy ./ res2.energy[1] res2.enstrophy ./ res2.enstrophy[1]], xlabel = "t",
    label = ["E(t)/E(0)" "Z(t)/Z(0)"], title = "Decaying flow", titlefont = 10)

# The energy spectrum moves to larger scales; `energy_spectrum` gives the spectrum of a single state, and `res2.spectrum` the time average:

κ, E0 = energy_spectrum(model2, ω0)
κ, E1 = energy_spectrum(model2, res2.w_hat_final)
plot(κ, [E0 E1] .+ eps(), xscale = :log10, yscale = :log10, ylims = (1e-8, 1), xlabel = "κ", ylabel = "E(κ)",
    label = ["t = 0" "t = 60"], title = "Energy spectrum", titlefont = 10)
