# # A forced flow with the package

# Here we use the package to reproduce, on a smaller scale, the workflow of [Time stepping for weakly turbulent flows](/pages/turbulence_timestepping): a steady forcing on a band of modes, an initial condition given by the Stokes state $(1/\nu)A^{-1}g$, a warm-up until the statistics settle, and then the time averages, with confidence intervals, computed with two different methods.
#
# To keep the building of this page short, we use a coarser grid, $64 \times 64$, and a larger viscosity, $\nu = 10^{-2}$, than in the notes ($128 \times 128$ and $\nu = 4 \times 10^{-3}$). The flow is in a similar regime, irregular and, seemingly, weakly turbulent, with the Taylor-like wavenumber $\kappa_\sigma$ inside the forcing band.

# ## Packages

using NSEPeriodic2D
using OrdinaryDiffEq: Tsit5
using OrdinaryDiffEqSSPRK: SSPRK33
using Plots
using Printf

plotfield(grid, f; kwargs...) = heatmap(grid.x, grid.y, permutedims(f);
    aspect_ratio = 1, xlims = (0, grid.Lx), ylims = (0, grid.Ly), xlabel = "x", ylabel = "y",
    color = :balance, clims = (-1, 1) .* maximum(abs, f), titlefont = 10, kwargs...)
nothing

# ## The model

grid = Grid(Lx = 2π, Nx = 64)

# The curl of the forcing has equal amplitudes and random phases on all the modes with $2 \leq |\kappa| \leq 4$, and root-mean-square value 1:

g_hat, modes = band_forcing(grid, 2, 4; rms = 1.0)
length(modes)

#

model = NSEModel(grid; ν = 1e-2, forcing = g_hat)

plotfield(grid, vorticity(model, g_hat); title = "curl of the forcing")

# ## Warm-up

# We start from the Stokes state $(1/\nu)A^{-1}g$, which is not a steady state of the full equation for this forcing and has a lot of energy, and integrate with an adaptive explicit method, which follows the changes in the velocity automatically. A loose tolerance suffices, since the step is limited by stability rather than accuracy:

t0 = 300.0
warmup = simulate(NSEProblem(model, stokes_state(model), (0.0, t0)), Tsit5();
    reltol = 1e-2, abstol = 1e-5, save_every = 1.0, spectrum = false, snapshot_every = 100.0)

#

plot(warmup.t, [warmup.energy warmup.enstrophy], yscale = :log10, xlabel = "t",
    label = ["E" "Z"], title = "Warm-up from the Stokes state", titlefont = 10)

# The statistics settle after about 200 time units. Snapshots of the vorticity during the warm-up:

plot(
    plotfield(grid, vorticity(model, stokes_state(model)); title = "t = 0 (Stokes state)"),
    [plotfield(grid, vorticity(model, w); title = "t = $(round(Int, t))") for (t, w) in warmup.snapshots]...,
    layout = (2, 2), size = (700, 650), colorbar = false
)

# ## Choosing a fixed time step

# For a fixed-step method, the step is limited by the advective CFL condition, `dt ≤ σ / (κx_max umax + κy_max vmax)`. The velocity fluctuates, so we take the maxima over the last 100 time units of the warm-up, with a margin, and `σ = 2.5`, below the stability limit of about 4 found in the notes for `SSPRK33()`. The step must divide the sampling interval:

recent = warmup.t .> t0 - 100
umax, vmax = 1.2 .* (maximum(warmup.umax[recent]), maximum(warmup.vmax[recent]))
dt = suggest_dt(model, warmup.w_hat_final; σ = 2.5, umax, vmax)
save_every = 0.5
dt = save_every / ceil(save_every / dt)

# ## Statistics with two methods

# Starting from the state at $t_0$, we integrate for 1000 time units with `SSPRK33()`, with this fixed step, and, for comparison, with `Tsit5()` again, now with a tighter tolerance. In longer test runs of this flow (3000 time units), `Tsit5()` with `reltol = 1e-2` gave a time-averaged energy about 5% larger than `SSPRK33()` with this step or half of it, and than `Tsit5()` with `reltol = 1e-4`, which all agreed with each other: a loose tolerance is fine for the warm-up, but may bias the statistics slightly.

T = 1000.0
prob = NSEProblem(model, warmup.w_hat_final, (t0, t0 + T))

res_ssp = simulate(prob, SSPRK33(); dt, save_every)

#

res_tsit = simulate(prob, Tsit5(); reltol = 1e-4, abstol = 1e-7, save_every)

# The CFL number of the fixed-step run, which should stay below the chosen `σ` (a stronger burst than those of the warm-up could exceed it, and, if much stronger, make the run blow up, which `simulate` would detect and report in its `retcode`):

plot(res_ssp.t, res_ssp.cfl, xlabel = "t", label = "CFL number (SSPRK33)", ylims = (0, 3),
    title = "CFL number", titlefont = 10)
hline!([2.5], linestyle = :dash, label = "σ")

# The two runs start from the same state, but the flow is chaotic, so the small differences between the methods grow, and, after some time, the two trajectories are unrelated:

plot(res_ssp.t, [res_ssp.energy res_tsit.energy], xlabel = "t", label = ["SSPRK33" "Tsit5"],
    xlims = (t0, t0 + 200), title = "Energy", titlefont = 10)

# Their statistics, on the other hand, should agree within the sampling errors. Here are the time averages, with 95% confidence intervals from 10 batch means (`batch_mean_ci`), and the Taylor-like wavenumber $\kappa_\sigma = \sqrt{\langle P\rangle/\langle Z\rangle}$:

for (name, r) in (("SSPRK33", res_ssp), ("Tsit5", res_tsit))
    E, hE = batch_mean_ci(r.energy)
    Z, hZ = batch_mean_ci(r.enstrophy)
    P, hP = batch_mean_ci(r.palinstrophy)
    @printf("%-8s ⟨E⟩ = %.3f ± %.3f   ⟨Z⟩ = %.3f ± %.3f   ⟨P⟩ = %.2f ± %.2f   κσ = %.3f\n",
        name, E, hE, Z, hZ, P, hP, kappa_sigma(r.enstrophy, r.palinstrophy))
end

# The numbers above are computed when this page is built, and, the flow being chaotic, they change with the round-off errors of the machine that builds it. As discussed in [Time stepping for weakly turbulent flows](/pages/turbulence_timestepping), the intervals from batch means tend to be too narrow for flows like this one, with intermittent bursts correlated over long times, so the averages of the two runs may differ by somewhat more than their intervals suggest; longer runs, or several runs from different initial conditions, reduce the uncertainty. The ratio $\kappa_\sigma$ is usually much more robust than the averages themselves.

# The time-averaged energy spectra:

plot(res_ssp.κ, [res_ssp.spectrum res_tsit.spectrum] .+ eps(), xscale = :log10, yscale = :log10,
    xlabel = "κ", ylabel = "E(κ)", label = ["SSPRK33" "Tsit5"], ylims = (1e-10, 10),
    title = "Time-averaged energy spectrum", titlefont = 10)

# The cost of each run, in evaluations of the right-hand side and wall time:

for (name, r) in (("SSPRK33", res_ssp), ("Tsit5", res_tsit))
    @printf("%-8s steps: %6d   evaluations: %7d   wall time: %.2f s\n", name, r.stats.naccept, r.stats.nf, r.stats.wall)
end

# The final vorticity:

plotfield(grid, vorticity(model, res_ssp.w_hat_final); title = "vorticity at t = $(round(Int, t0 + T)) (SSPRK33)")
