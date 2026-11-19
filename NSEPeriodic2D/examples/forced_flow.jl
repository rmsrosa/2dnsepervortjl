# A forced, (seemingly weakly) turbulent and chaotic flow, as in the notes "Time stepping for
# weakly turbulent flows": forcing on all the modes with 2 ≤ |κ| ≤ 4, starting from the
# Stokes state (1/ν)A⁻¹g. The times here are short, for a quick demonstration; in the notes,
# the statistics settle after about 1000 time units and are averaged over 1000.
#
# Run from the folder `examples/`:  julia --project=. forced_flow.jl
using NSEPeriodic2D
using OrdinaryDiffEq: Tsit5
using OrdinaryDiffEqSSPRK: SSPRK33

grid = Grid(Lx = 2π, Nx = 128)
g_hat, modes = band_forcing(grid, 2, 4; rms = 1.0)
model = NSEModel(grid; ν = 4e-3, forcing = g_hat)
println(model, ", ", length(modes), " forced modes")

# Warm-up with an adaptive explicit method, which follows the bursts automatically
t0 = 200.0
warmup = simulate(NSEProblem(model, stokes_state(model), (0.0, t0)), Tsit5();
    reltol = 1e-2, abstol = 1e-5, save_every = 1.0, spectrum = false, progress = true)
println(warmup)

# Statistics with a fixed-step method: the step from the velocity maxima of the warm-up
# (with a margin), with σ = 2.5 (stable up to about 4 in the notes)
umax, vmax = 1.2 * maximum(warmup.umax), 1.2 * maximum(warmup.vmax)
dt = suggest_dt(model, warmup.w_hat_final; σ = 2.5, umax, vmax)
dt = 0.5 / ceil(0.5 / dt)                    # so that dt divides save_every
T = 200.0
res = simulate(NSEProblem(model, warmup.w_hat_final, (t0, t0 + T)), SSPRK33(); dt, save_every = 0.5)
println(res, ", dt = ", dt, ", maximum CFL = ", maximum(res.cfl))

for (name, x) in (("⟨E⟩", res.energy), ("⟨Z⟩", res.enstrophy), ("⟨P⟩", res.palinstrophy))
    μ, h = batch_mean_ci(x)
    println(rpad(name, 5), round(μ, sigdigits = 4), " ± ", round(h, sigdigits = 2))
end
println("κσ = ", round(kappa_sigma(res.enstrophy, res.palinstrophy), sigdigits = 4))
println("Time-averaged energy spectrum, first modes: ", round.(res.spectrum[1:6], sigdigits = 3))
