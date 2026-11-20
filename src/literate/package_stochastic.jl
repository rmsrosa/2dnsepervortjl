# # Stochastic and random forcing

# The package also handles a forcing with a random component, for the equation
#
# $$
# \mathrm{d}\mathbf{u} + \left(\mathbf{u}\cdot\nabla\mathbf{u} + \nabla p\right)\mathrm{d}t = \nu\Delta\mathbf{u}\,\mathrm{d}t + \mathbf{f}\,\mathrm{d}t + N_t\,\mathrm{d}t + \sigma\Phi\,\mathrm{d}W_t, \qquad \nabla\cdot\mathbf{u} = 0,
# $$
#
# where $\mathbf{f}$ is a steady forcing, as before, $W_t$ is an $n$-dimensional Wiener process, $\Phi : \mathbb{R}^n \to H$ maps it onto divergence-free velocity fields (the modes forced stochastically), and $N_t$ is an arbitrary stochastic process with values in $H$, such as an Ornstein-Uhlenbeck process, a compound Poisson process, or the output of an embedded low-dimensional chaotic system. Each term may be absent. In vorticity form, the curls of these terms are added to the equation.
#
# * The forcing modes are given by `fourier_modes(grid, κmin, κmax)` (or by a list of wavenumbers, or by arbitrary fields with `ForcingModes`): divergence-free Fourier modes $\sqrt{2}\,\nabla^\perp(\cos(\boldsymbol{\kappa}\cdot\mathbf{x})/|\boldsymbol{\kappa}|)$ and $\sqrt{2}\,\nabla^\perp(\sin(\boldsymbol{\kappa}\cdot\mathbf{x})/|\boldsymbol{\kappa}|)$, normalized to unit mean square.
# * The stochastic forcing $\sigma\Phi\,\mathrm{d}W_t$ is an `AdditiveNoise(modes; σ, amplitudes)`, with $\Phi e_j$ = `amplitudes[j]` times the $j$-th mode. Its mean injection rates of energy, $\varepsilon = \tfrac12\sigma^2\sum_j a_j^2\langle|\boldsymbol{\phi}_j|^2\rangle$, and of enstrophy, $\eta = \tfrac12\sigma^2\sum_j a_j^2\langle|\nabla\times\boldsymbol{\phi}_j|^2\rangle$, do not depend on the flow (Itô), and are given by `energy_injection_rate` and `enstrophy_injection_rate`.
# * The random forcing $N_t = \sum_j \eta_j(t)\boldsymbol{\phi}_j$ is a `RandomForcing(modes, η)`, for any function `η(t)` returning the coefficients: sample paths from `ou_path`, `compound_poisson_path` and `lorenz_path`, or any other.
#
# With a stochastic forcing, problems are built with `kind = :sde`, for the solvers of [StochasticDiffEq.jl](https://github.com/SciML/StochasticDiffEq.jl) (e.g. `EM()`, `SRA1()`, `SOSRA()`), or integrated with the fixed-step schemes of the package: `StochasticETDRK4()`, `StochasticExpEuler()` and `SplitStochasticRK(:rk4)` or `SplitStochasticRK(:ssprk3)`. A random forcing alone is a time-dependent forcing, and the problem can be solved with any of the deterministic methods of the previous pages.

# ## Packages

using NSEPeriodic2D
using OrdinaryDiffEq: Tsit5
using Plots
using Printf
using Random: Xoshiro

plotfield(grid, f; kwargs...) = heatmap(grid.x, grid.y, permutedims(f);
    aspect_ratio = 1, xlims = (0, grid.Lx), ylims = (0, grid.Ly), xlabel = "x", ylabel = "y",
    color = :balance, clims = (-1, 1) .* maximum(abs, f), titlefont = 10, kwargs...)
nothing

# ## A flow driven by an additive noise

# A $64 \times 64$ grid, viscosity $\nu = 10^{-2}$, and a white-in-time forcing on all the modes with $3 \leq |\boldsymbol{\kappa}| \leq 5$, with energy injection rate $\varepsilon = 0.1$ and no deterministic forcing:

grid = Grid(Lx = 2π, Nx = 64)
modes = fourier_modes(grid, 3, 5)
ε = 0.1
noise = AdditiveNoise(modes; σ = sqrt(2ε / length(modes)))

#

model = NSEModel(grid; ν = 1e-2, noise)

# Starting from rest, we integrate with the stochastic ETDRK4 scheme, with a fixed step:

prob = NSEProblem(model, zeros(grid.Nx, grid.Ny), (0.0, 400.0); kind = :sde)
res = simulate(prob, StochasticETDRK4(); dt = 0.01, save_every = 0.5, seed = 2026)

#

plot(res.t, [res.energy res.enstrophy], xlabel = "t", label = ["E" "Z"],
    title = "Energy and enstrophy, driven by the noise", titlefont = 10)

# In a statistically steady state, the mean energy and enstrophy budgets close exactly: $2\nu\langle Z\rangle = \varepsilon$ and $2\nu\langle P\rangle = \eta$, with $Z$ the enstrophy and $P$ the palinstrophy. This holds for any flow, chaotic or not, so it is a sharp check of a numerical method, independent of the trajectory. After the transient:

s = res.t .> 150
@printf("2ν⟨Z⟩/ε = %.3f,   2ν⟨P⟩/η = %.3f\n", 2 * model.ν * sum(res.enstrophy[s]) / count(s) / ε,
    2 * model.ν * sum(res.palinstrophy[s]) / count(s) / enstrophy_injection_rate(noise))

#

plotfield(grid, vorticity(model, res.w_hat_final); title = "vorticity at t = 400")

# ## Random forcing: Ornstein-Uhlenbeck, compound Poisson, and a chaotic process

# Now a random forcing on the four modes with wavenumbers $(1, 2)$ and $(2, 1)$, driven by three kinds of processes, with no noise:

rmodes = fourier_modes(grid, [(1, 2), (2, 1)])
tspan = (0.0, 200.0)

ou = ou_path(4, tspan; θ = 0.5, s = 0.3, dt = 0.01, rng = Xoshiro(1))
cp = compound_poisson_path(4, tspan; rate = 0.2, θ = 0.2, jump = rng -> 0.3randn(rng), rng = Xoshiro(2))
lz = lorenz_path(tspan; timescale = 5.0, scale = 0.3)
lz4(t) = (x = lz(t); [x[1], x[2], x[3], 0.0])
nothing

#

ts = 0:0.1:200
plot(
    plot(ts, reduce(hcat, [copy(ou(t)) for t in ts])', label = false, title = "Ornstein-Uhlenbeck", titlefont = 10),
    plot(ts, reduce(hcat, [copy(cp(t)) for t in ts])', label = false, title = "compound Poisson, with decay", titlefont = 10),
    plot(ts, reduce(hcat, [lz4(t) for t in ts])', label = false, title = "Lorenz-63", titlefont = 10, xlabel = "t"),
    layout = (3, 1), size = (700, 600)
)

# A random forcing alone is a time-dependent forcing, so the deterministic solvers apply, path by path; for the compound Poisson process, with jumps, we give the jump times to the adaptive solver as `tstops`:

runs = map([("Ornstein-Uhlenbeck", ou, (;)), ("compound Poisson", cp, (; tstops = jump_times(cp))), ("Lorenz-63", lz4, (;))]) do (name, η, kw)
    m = NSEModel(grid; ν = 1e-2, random_forcing = RandomForcing(rmodes, η))
    name => simulate(NSEProblem(m, zeros(grid.Nx, grid.Ny), tspan), Tsit5(); reltol = 1e-6, abstol = 1e-9,
        save_every = 0.5, spectrum = false, kw...)
end
plot([r.t for (_, r) in runs], [r.energy for (_, r) in runs], label = permutedims(first.(runs)),
    xlabel = "t", ylabel = "E", title = "Energy of flows driven by random forcings", titlefont = 10)

# Both terms can be combined, with the stochastic schemes: here, the noise of the first example together with the Ornstein-Uhlenbeck forcing:

model2 = NSEModel(grid; ν = 1e-2, noise, random_forcing = RandomForcing(rmodes, ou))
res2 = simulate(NSEProblem(model2, zeros(grid.Nx, grid.Ny), tspan; kind = :sde), StochasticETDRK4();
    dt = 0.01, save_every = 0.5, seed = 7)
plot(res2.t, res2.energy, xlabel = "t", label = "E", title = "Noise and Ornstein-Uhlenbeck forcing", titlefont = 10)

# ## Comparison of methods

# RESULTS_PLACEHOLDER
