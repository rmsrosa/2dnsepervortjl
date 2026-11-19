using NSEPeriodic2D
using NSEPeriodic2D: nonlinear!, jvp!, rhs_vec!
using Test
using FFTW: rfft
using LinearAlgebra: norm
using Random: Xoshiro
using OrdinaryDiffEq: Tsit5, Vern9
using OrdinaryDiffEqExponentialRK: ETDRK4
using OrdinaryDiffEqBDF: QNDF
using LinearSolve: KrylovJL_GMRES

relerr(a, b) = norm(a .- b) / norm(b)

@testset "NSEPeriodic2D" begin

@testset "Grid" begin
    g = Grid(Nx = 64)
    @test (g.Nx, g.Ny, g.Mx) == (64, 64, 33)
    @test g.Lx ≈ 2π && g.Ly ≈ 2π
    @test g.κx_max ≈ 21 && g.κy_max ≈ 21
    @test count(g.mask) == 22 * 43
    r = Grid(Lx = 2π, Ly = 4π, Nx = 32, Ny = 48)
    @test r.κy[2] ≈ 0.5
    @test size(r.κ2) == (17, 48)
    @test r.y[end] ≈ 4π - 4π / 48
end

# Exact fields on a rectangular domain: ψ = sin(2x) cos(3y/2) + cos(x + y), with Ly = 4π
g = Grid(Lx = 2π, Ly = 4π, Nx = 64, Ny = 96)
ψf(x, y) = sin(2x) * cos(1.5y) + cos(x + 0.5y)
uf(x, y) = -1.5 * sin(2x) * sin(1.5y) - 0.5 * sin(x + 0.5y)        # ∂ᵧψ
vf(x, y) = -2cos(2x) * cos(1.5y) + sin(x + 0.5y)                   # -∂ₓψ
ωf(x, y) = 6.25 * sin(2x) * cos(1.5y) + 1.25 * cos(x + 0.5y)       # -Δψ
physical(f) = [f(x, y) for x in g.x, y in g.y]

@testset "Fields and spectral operators" begin
    m = NSEModel(g; ν = 0.1)
    w = vorticity_hat(g, ωf)
    @test vorticity(m, w) ≈ physical(ωf)
    u, v = velocity(m, w)
    @test u ≈ physical(uf) atol = 1e-12
    @test v ≈ physical(vf) atol = 1e-12
    @test streamfunction(m, w) ≈ physical(ψf) atol = 1e-12
    @test vorticity_from_velocity(g, uf, vf) ≈ w
    @test vorticity_hat(g, physical(ωf)) ≈ w
    @test vorticity_hat(g, w) ≈ w
    @test_throws DimensionMismatch vorticity_hat(g, zeros(3, 3))
end

@testset "Diagnostics" begin
    m = NSEModel(g; ν = 0.1)
    w = vorticity_hat(g, ωf)
    u, v = physical(uf), physical(vf)
    @test energy(m, w) ≈ sum(u .^ 2 .+ v .^ 2) / 2 / length(u)
    @test enstrophy(m, w) ≈ sum(physical(ωf) .^ 2) / 2 / length(u)
    ωx(x, y) = 12.5 * cos(2x) * cos(1.5y) - 1.25 * sin(x + 0.5y)
    ωy(x, y) = -9.375 * sin(2x) * sin(1.5y) - 0.625 * sin(x + 0.5y)
    @test palinstrophy(m, w) ≈ sum(physical(ωx) .^ 2 .+ physical(ωy) .^ 2) / 2 / length(u)
    κ, E = energy_spectrum(m, w)
    @test sum(E) ≈ energy(m, w)
    @test all(max_velocity(m, w) .≈ (maximum(abs, u), maximum(abs, v)))
    @test cfl_number(m, w, 0.01) ≈ 0.01 * (g.κx_max * maximum(abs, u) + g.κy_max * maximum(abs, v))
    @test cfl_number(m, w, suggest_dt(m, w; σ = 2.0)) ≈ 2.0
    @test kappa_sigma(enstrophy(m, w), palinstrophy(m, w)) ≈ sqrt(palinstrophy(m, w) / enstrophy(m, w))
    μ, h = batch_mean_ci(collect(1.0:100.0))
    @test μ ≈ 50.5 && h > 0
end

@testset "Forcing and initial conditions" begin
    gs = Grid(Nx = 64)
    f, modes = band_forcing(gs, 2, 4; rms = 0.7)
    @test length(modes) == 20
    @test sqrt(sum(abs2, NSEModel(gs; ν = 1.0).iplan * copy(f)) / 64^2) ≈ 0.7
    w = random_vorticity(gs, 3, 8; energy = 2.5, rng = Xoshiro(1))
    @test energy(NSEModel(gs; ν = 1.0), w) ≈ 2.5
    # a field built from a real physical field is unchanged by a round trip
    m = NSEModel(gs; ν = 1.0)
    @test vorticity_hat(gs, vorticity(m, w)) ≈ w
end

@testset "Steady state, Jacobian-vector product" begin
    # forcing on the single shell |κ|² = 25 (and the Stokes state is a steady state)
    gs = Grid(Nx = 64)
    gf(x, y) = cos(3x + 4y) + 2sin(5x) - cos(4x - 3y + 1)
    m = NSEModel(gs; ν = 0.05, forcing = gf)
    w = stokes_state(m)
    dw = similar(w); nonlinear!(dw, w, m)
    @test norm(dw .- 0.05 .* gs.κ2 .* w) / norm(m.g_hat) < 1e-12
    # the jvp equals the centered difference quotient exactly, the rhs being quadratic
    m = NSEModel(gs; ν = 0.05, forcing = band_forcing(gs, 2, 4)[1])
    w = random_vorticity(gs, 1, 10; energy = 1.0, rng = Xoshiro(2))
    h = random_vorticity(gs, 1, 10; energy = 1.0, rng = Xoshiro(3))
    Jh = similar(w); jvp!(Jh, h, w, m)
    F(x) = (u = vec(copy(x)); du = similar(u); rhs_vec!(du, u, m, 0.0); reshape(du, size(x)))
    ε = 0.1
    @test relerr(Jh, (F(w .+ ε .* h) .- F(w .- ε .* h)) ./ 2ε) < 1e-12
end

@testset "Inviscid, unforced conservation of energy and enstrophy" begin
    gs = Grid(Nx = 32)
    m = NSEModel(gs; ν = 0.0)
    w0 = random_vorticity(gs, 1, 8; energy = 1.0, rng = Xoshiro(4))
    sol = solve(NSEProblem(m, w0, (0.0, 2.0)), Vern9(); abstol = 1e-12, reltol = 1e-12, save_everystep = false)
    w1 = reshape(sol.u[end], gs.Mx, gs.Ny)
    @test relerr(w1, w0) > 1e-2                       # the flow does evolve
    @test energy(m, w1) ≈ energy(m, w0) rtol = 1e-9
    @test enstrophy(m, w1) ≈ enstrophy(m, w0) rtol = 1e-9
end

@testset "The three problem kinds agree" begin
    gs = Grid(Nx = 32)
    m = NSEModel(gs; ν = 0.2, forcing = band_forcing(gs, 1, 3)[1])
    w0 = random_vorticity(gs, 1, 6; energy = 1.0, rng = Xoshiro(5))
    tspan = (0.0, 0.5)
    pf = NSEProblem(m, w0, tspan)
    ref = state_to_vorticity_hat(pf, solve(pf, Vern9(); abstol = 1e-12, reltol = 1e-12, save_everystep = false).u[end])
    ps = NSEProblem(m, w0, tspan; kind = :split)
    ws = state_to_vorticity_hat(ps, solve(ps, ETDRK4(); dt = 1e-3, save_everystep = false).u[end])
    @test relerr(ws, ref) < 1e-8
    pim = NSEProblem(m, w0, tspan; kind = :implicit)
    @test pim.prob.u0 isa Vector{Float64}
    wi = state_to_vorticity_hat(pim, solve(pim, QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false);
        abstol = 1e-10, reltol = 1e-10, save_everystep = false).u[end])
    @test relerr(wi, ref) < 1e-6
    @test_throws ArgumentError NSEProblem(m, w0, tspan; kind = :other)
    p2 = remake_problem(pf; tspan = (0.0, 1.0))
    @test p2.prob.tspan == (0.0, 1.0) && p2.prob.u0 == pf.prob.u0
end

@testset "simulate" begin
    # Single-mode forcing at large viscosity: convergence to the Stokes steady state
    gs = Grid(Nx = 32)
    m = NSEModel(gs; ν = 1.0, forcing = (x, y) -> 2cos(x + 2y))
    ws = stokes_state(m)
    w0 = random_vorticity(gs, 1, 5; energy = 0.5, rng = Xoshiro(6))
    p = NSEProblem(m, w0, (0.0, 20.0); kind = :split)
    r = simulate(p, ETDRK4(); dt = 0.01, save_every = 0.5, snapshot_every = 10.0)
    @test r.retcode == :Success
    @test length(r.t) == 40 && r.t[end] ≈ 20.0
    @test length(r.snapshots) == 2 && r.snapshots[end][1] ≈ 20.0
    @test relerr(r.w_hat_final, ws) < 1e-6
    @test r.energy[end] ≈ energy(m, ws) rtol = 1e-6
    @test sum(r.spectrum) ≈ sum(r.energy) / length(r.energy) rtol = 1e-10
    @test all(isfinite, r.cfl)
    # adaptive method: samples at the requested times, no CFL recorded
    r2 = simulate(NSEProblem(m, w0, (0.0, 2.0)), Tsit5(); save_every = 0.25, spectrum = false)
    @test r2.t ≈ 0.25:0.25:2.0
    @test all(isnan, r2.cfl) && isempty(r2.spectrum)
    # argument checks
    @test_throws ArgumentError simulate(p, ETDRK4(); save_every = 0.5)            # no dt
    @test_throws ArgumentError simulate(p, ETDRK4(); dt = 0.3, save_every = 0.5)  # dt does not divide
    # blow-up detection with an unstable time step
    mb = NSEModel(gs; ν = 1e-3)
    pb = NSEProblem(mb, random_vorticity(gs, 1, 8; energy = 10.0, rng = Xoshiro(7)), (0.0, 50.0); kind = :split)
    rb = simulate(pb, ETDRK4(); dt = 0.5, save_every = 0.5)
    @test rb.retcode == :BlowUp
end

end
