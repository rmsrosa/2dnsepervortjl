# Tests of the stochastic and random forcing, included from runtests.jl
using StochasticDiffEq: EM, SRA1, SOSRA
using Statistics: mean, var
using SparseArrays: nnz

@testset "Forcing modes and additive noise" begin
    g = Grid(Lx = 2π, Ly = 4π, Nx = 32, Ny = 48)
    fm = fourier_modes(g, [(1, 2), (0, 3), (2, 0)])
    @test length(fm) == 6
    @test fm.velocity_ms ≈ ones(6)
    κ2 = [1 + 1, 1 + 1, 2.25, 2.25, 4, 4]                  # |κ|² with κ = (jx, jy/2)
    @test fm.vorticity_ms ≈ κ2
    m = NSEModel(g; ν = 0.1)
    # the modes are the vorticities √2|κ|cos(κ⋅x) and √2|κ|sin(κ⋅x)
    @test vorticity(m, reshape(fm.Q[:, 1], g.Mx, g.Ny)) ≈ [√2 * sqrt(2) * cos(x + y) for x in g.x, y in g.y]
    @test vorticity(m, reshape(fm.Q[:, 4], g.Mx, g.Ny)) ≈ [√2 * 1.5 * sin(1.5y) for x in g.x, y in g.y]
    a = AdditiveNoise(fm; σ = 0.5, amplitudes = 1:6)
    @test energy_injection_rate(a) ≈ 0.125 * sum((1:6) .^ 2)
    @test enstrophy_injection_rate(a) ≈ 0.125 * sum((1:6) .^ 2 .* κ2)
    @test length(fourier_modes(Grid(Nx = 32), 2, 4)) == 40
    # the noise matrix is sparse: one or two coefficients per mode
    p = NSEProblem(NSEModel(g; ν = 0.1, noise = a), zeros(g.Nx, g.Ny), (0.0, 1.0); kind = :sde)
    @test nnz(p.prob.noise_rate_prototype) ≤ 4 * length(fm)
    @test_throws ArgumentError NSEProblem(NSEModel(g; ν = 0.1, noise = a), zeros(g.Nx, g.Ny), (0.0, 1.0))
    @test_throws ArgumentError NSEProblem(m, zeros(g.Nx, g.Ny), (0.0, 1.0); kind = :sde)
end

@testset "Sample paths" begin
    p = ou_path(2, (0.0, 20000.0); θ = 0.5, s = 2.0, dt = 0.1, rng = Xoshiro(1))
    x = reduce(hcat, [copy(p(t)) for t in 0.0:1.0:20000.0])
    @test abs(mean(x)) < 0.15
    @test var(x) ≈ 4 rtol = 0.1
    @test p(0.05) ≈ (p.X[:, 1] .+ p.X[:, 2]) ./ 2               # linear interpolation
    c = compound_poisson_path(3, (0.0, 100.0); rate = 2.0, rng = Xoshiro(2))
    τ = jump_times(c)
    @test issorted(τ) && length(τ) > 400
    @test c(0.0) == zeros(3)
    # without decay, the value is the sum of the jumps so far
    @test c(50.0)[1] ≈ sum(diff(c.values[1])[c.times[1][2:end] .≤ 50.0])
    d = compound_poisson_path(1, (0.0, 10.0); rate = 1.0, θ = 2.0, rng = Xoshiro(3))
    t1, t2 = d.times[1][2], d.times[1][3]
    @test d((t1 + t2) / 2)[1] ≈ d(t1)[1] * exp(-(t2 - t1))     # decay at rate θ = 2 between jumps
    l = lorenz_path((0.0, 50.0); timescale = 2.0)
    y = reduce(hcat, [copy(l(t)) for t in 0.0:0.01:50.0])
    @test all(isfinite, y) && maximum(abs, y) < 5
    @test maximum(abs, diff(y; dims = 2)) < 0.5                 # continuous
    @test maximum(abs, diff(diff(y; dims = 2); dims = 2)) < 0.05 # and smooth
end

@testset "Random forcing" begin
    g = Grid(Nx = 32)
    fm = fourier_modes(g, 1, 3)
    c = randn(Xoshiro(4), length(fm))
    # a constant random forcing is the same as the corresponding steady forcing
    m1 = NSEModel(g; ν = 0.05, random_forcing = RandomForcing(fm, t -> c))
    m2 = NSEModel(g; ν = 0.05, forcing = reshape(fm.Q * c, g.Mx, g.Ny))
    w0 = random_vorticity(g, 1, 6; rng = Xoshiro(5)); w = vec(copy(w0))
    d1 = similar(w); d2 = similar(w)
    NSEPeriodic2D.rhs_vec!(d1, w, m1, 0.3); NSEPeriodic2D.rhs_vec!(d2, w, m2, 0.3)
    @test d1 ≈ d2
    # a time-dependent forcing, with the deterministic solvers and the package schemes
    path = ou_path(length(fm), (0.0, 2.0); θ = 1.0, s = 1.0, dt = 0.01, rng = Xoshiro(6))
    m = NSEModel(g; ν = 0.5, random_forcing = RandomForcing(fm, path))
    p = NSEProblem(m, w0, (0.0, 2.0))
    r1 = simulate(p, Tsit5(); reltol = 1e-8, abstol = 1e-10, save_every = 0.5, spectrum = false)
    r2 = simulate(p, SplitStochasticRK(:rk4); dt = 0.005, save_every = 0.5, spectrum = false)
    r3 = simulate(p, StochasticETDRK4(); dt = 0.005, save_every = 0.5, spectrum = false)
    @test r1.retcode == r2.retcode == r3.retcode == :Success
    @test relerr(r2.w_hat_final, r1.w_hat_final) < 1e-5
    @test relerr(r3.w_hat_final, r1.w_hat_final) < 1e-5
end

# A single wavevector: the flow stays on one shell, where the nonlinear term vanishes, and the
# coefficients are Ornstein-Uhlenbeck processes with stationary energy ε / (2ν|κ|²).
@testset "Stochastic schemes on an exactly solvable case" begin
    g = Grid(Nx = 32)
    a = AdditiveNoise(fourier_modes(g, [(1, 2)]); σ = 1.0)       # ε = 1, |κ|² = 5
    m = NSEModel(g; ν = 0.1, noise = a)
    Eexact = energy_injection_rate(a) / (2 * 0.1 * 5)
    @test Eexact ≈ 1
    w0 = zeros(g.Nx, g.Ny)
    check(r) = (μ = mean(r.energy); @test r.retcode == :Success; @test μ ≈ Eexact rtol = 0.12; μ)
    # exact for the linear part; the step is limited by the stability of the explicit treatment
    # of the (round-off) perturbations off the shell
    check(simulate(NSEProblem(m, w0, (0.0, 2000.0); kind = :sde), StochasticExpEuler(); dt = 0.05, save_every = 0.5, seed = 1, spectrum = false))
    check(simulate(NSEProblem(m, w0, (0.0, 2000.0); kind = :sde), StochasticETDRK4(); dt = 0.05, save_every = 0.5, seed = 8, spectrum = false))
    check(simulate(NSEProblem(m, w0, (0.0, 2000.0); kind = :sde), SplitStochasticRK(:rk4); dt = 0.05, save_every = 0.5, seed = 2, spectrum = false))
    check(simulate(NSEProblem(m, w0, (0.0, 2000.0); kind = :sde), SplitStochasticRK(:ssprk3); dt = 0.05, save_every = 0.5, seed = 3, spectrum = false))
    check(simulate(NSEProblem(m, w0, (0.0, 1000.0); kind = :sde), EM(); dt = 0.01, save_every = 0.5, seed = 4, spectrum = false))
    check(simulate(NSEProblem(m, w0, (0.0, 1000.0); kind = :sde), SRA1(); save_every = 0.5, seed = 5, spectrum = false))
    check(simulate(NSEProblem(m, w0, (0.0, 1000.0); kind = :sde), SOSRA(); save_every = 0.5, seed = 6, spectrum = false))
    # the flow stays on the shell
    r = simulate(NSEProblem(m, w0, (0.0, 10.0); kind = :sde), StochasticExpEuler(); dt = 0.1, save_every = 0.5, seed = 7)
    @test r.palinstrophy ≈ 5 .* r.enstrophy
    @test r.enstrophy ≈ 5 .* r.energy
    # reproducible with a seed
    r2 = simulate(NSEProblem(m, w0, (0.0, 10.0); kind = :sde), StochasticExpEuler(); dt = 0.1, save_every = 0.5, seed = 7)
    @test r2.energy == r.energy
end
