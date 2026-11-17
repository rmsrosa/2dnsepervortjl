# # Benchmark of stiff and exponential solvers

# This script runs the benchmark reported in the page "Stiff solvers, Jacobians and automatic differentiation" (`src/literate/stiff_solvers_report.jl`), on the machine where it is run. The code is a compact version of that in `src/literate/evolution_jvp.jl` and `src/literate/evolution_etdrk4.jl`, to which we refer for the details. To reduce the noise a bit, each method is run once to compile it, and then three more times, and the minimum of the three elapsed times is reported.

# ### Packages and environment

using FFTW
using OrdinaryDiffEq
using OrdinaryDiffEqBDF: QNDF, FBDF
using OrdinaryDiffEqSDIRK: KenCarp47, TRBDF2
using OrdinaryDiffEqExponentialRK: ETDRK4
using SciMLOperators: DiagonalOperator
using LinearSolve: KrylovJL_GMRES
using LinearAlgebra: norm, mul!
using Random
import LinearSolve, SciMLOperators

println("Julia ", VERSION, " on ", Sys.MACHINE)
println("CPU: ", Sys.cpu_info()[1].model, " (", Sys.CPU_THREADS, " logical cores)")
println("Julia threads: ", Threads.nthreads(), ", FFTW threads: ", FFTW.nthreads())
for pkg in (OrdinaryDiffEq, FFTW, LinearSolve, SciMLOperators)
    println(rpad(string(pkg), 16), pkgversion(pkg))
end

# ### Setup

function get_operators(N, κ₀)
    Dx_hat = im * κ₀ * [
        ifelse(k1 ≤ div(N, 2) + 1, k1 - 1, k1 - 1 - N) for k2 in 1:div(N, 2)+1, k1 in 1:N
    ]
    Dy_hat = im * κ₀ * [k2 - 1 for k2 in 1:div(N, 2)+1, k1 in 1:N]
    Delta_hat = - κ₀^2 * [
        ifelse(k1 ≤ div(N, 2) + 1, (k1 - 1)^2 + (k2 - 1)^2, (k1 - 1 - N)^2 + (k2 - 1)^2)
        for k2 in 1:div(N, 2)+1, k1 in 1:N
    ]
    DxsqDysq_hat = Dx_hat.^2 .- Dy_hat.^2
    Dxy_hat = Dx_hat .* Dy_hat
    Hu_hat = - Dy_hat ./ Delta_hat
    Hu_hat[1, 1] = 0.0
    Hv_hat = Dx_hat ./ Delta_hat
    Hv_hat[1, 1] = 0.0
    return Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat
end

function field_from_modes(rng::AbstractRNG, L, N, num_modes::Int)
    κ₀ = 2π/L
    x = y = (L/N):(L/N):L
    modes = rand(rng, 1:div(N,10), num_modes, 2)
    amps = rand(rng, num_modes, 2)
    return sum(
        [
            2κ₀^2 * (k[1]^2 + k[2]^2) * (
                a[1] * cos.(κ₀ * (k[1] * one.(y) * x' + k[2] * y * one.(x)'))
                - a[2] * sin.(κ₀ * (k[1] * one.(y) * x' + k[2] * y * one.(x)'))
            )
            for (k, a) in zip(eachrow(modes), eachrow(amps))
        ]
    )
end

function dealias!(w_hat, N, Nsub)
    w_hat[div(Nsub,2) + 1:end, :] .= 0
    w_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0
    return w_hat
end

function get_params(N, Nsub, ν, g_hat, operators)
    M = div(N, 2) + 1
    phys() = zeros(N, N)
    spec() = zeros(ComplexF64, M, N)
    plan = plan_rfft(phys(), flags=FFTW.MEASURE)
    plan_inv = plan_irfft(spec(), N, flags=FFTW.MEASURE)
    return (;
        operators, ν, g_hat, N, Nsub, M, plan, plan_inv,
        uw = phys(), vw = phys(), uh = phys(), vh = phys(), a = phys(), b = phys(),
        sc = spec(), a_hat = spec(), b_hat = spec()
    )
end

# The nonlinear part, $\hat g - \hat B(\hat\omega, \hat\omega)$, acting on the complex half-spectrum:

function nonlinear!(dw_hat, w_hat, params)
    (; operators, g_hat, N, Nsub, plan, plan_inv, uw, vw, a, b, sc, a_hat, b_hat) = params
    _, _, _, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = operators
    sc .= Hu_hat .* w_hat
    mul!(uw, plan_inv, sc)
    sc .= Hv_hat .* w_hat
    mul!(vw, plan_inv, sc)
    @. a = uw * vw
    @. b = vw^2 - uw^2
    mul!(a_hat, plan, a)
    mul!(b_hat, plan, b)
    @. dw_hat = g_hat - DxsqDysq_hat * a_hat - Dxy_hat * b_hat
    dealias!(dw_hat, N, Nsub)
    return dw_hat
end

# The full right-hand side and its Jacobian-vector product, acting on the real state vector, for the explicit and implicit methods:

function rhs!(du, u, params, t)
    w_hat = reinterpret(reshape, ComplexF64, u)
    dw_hat = reinterpret(reshape, ComplexF64, du)
    nonlinear!(dw_hat, w_hat, params)
    _, _, Delta_hat = params.operators
    @. dw_hat += params.ν * Delta_hat * w_hat
    return du
end

function jvp!(Jv, v, u, params, t)
    (; operators, ν, N, Nsub, plan, plan_inv, uw, vw, uh, vh, a, b, sc, a_hat, b_hat) = params
    _, _, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = operators
    v = reshape(v, size(u))
    Jv = reshape(Jv, size(u))
    w_hat = reinterpret(reshape, ComplexF64, u)
    h_hat = reinterpret(reshape, ComplexF64, v)
    Jh_hat = reinterpret(reshape, ComplexF64, Jv)
    sc .= Hu_hat .* w_hat
    mul!(uw, plan_inv, sc)
    sc .= Hv_hat .* w_hat
    mul!(vw, plan_inv, sc)
    sc .= Hu_hat .* h_hat
    mul!(uh, plan_inv, sc)
    sc .= Hv_hat .* h_hat
    mul!(vh, plan_inv, sc)
    @. a = uw * vh + uh * vw
    @. b = 2 * (vw * vh - uw * uh)
    mul!(a_hat, plan, a)
    mul!(b_hat, plan, b)
    @. Jh_hat = ν * Delta_hat * h_hat - DxsqDysq_hat * a_hat - Dxy_hat * b_hat
    dealias!(Jh_hat, N, Nsub)
    return Jv
end

# The nonlinear part acting on the vectorized complex state, for ETDRK4:

nonlinear_vec!(dw, w, params, t) = (nonlinear!(reshape(dw, params.M, :), reshape(w, params.M, :), params); dw)

# The same parameters as in the static tables:

L = 2π
κ₀ = 2π/L
N = 128
Nsub = 84
ν = 1.0
tspan = (0.0, 1.0)

operators = get_operators(N, κ₀)
rng = Xoshiro(123)
vort_init_hat = dealias!(rfft(field_from_modes(rng, L, N, 4)), N, Nsub)
g_hat = dealias!(rfft(field_from_modes(rng, L, N, 1)), N, Nsub)
params = get_params(N, Nsub, ν, g_hat, operators)

u0 = collect(reinterpret(reshape, Float64, vort_init_hat))
w0 = vec(copy(vort_init_hat))

prob_real = ODEProblem(ODEFunction(rhs!; jvp = jvp!), u0, tspan, params)
prob_split = SplitODEProblem(DiagonalOperator(vec(complex.(ν .* operators[3]))), nonlinear_vec!, w0, tspan, params)

u_ref = solve(prob_real, Vern9(), abstol = 1e-12, reltol = 1e-12, save_everystep = false).u[end]
w_ref = vec(reinterpret(reshape, ComplexF64, u_ref))
nothing

# ### Benchmark

function bench(prob, alg, uref; kwargs...)
    sol = solve(prob, alg; save_everystep = false, kwargs...)
    elapsed = minimum(@elapsed(solve(prob, alg; save_everystep = false, kwargs...)) for _ in 1:3)
    relerr = norm(sol.u[end] .- uref) / norm(uref)
    return sol.retcode, elapsed, relerr, sol.stats.naccept
end

benchmarks = [
    "Tsit5 (explicit)" => () -> bench(prob_real, Tsit5(), u_ref),
    "Vern9 (explicit)" => () -> bench(prob_real, Vern9(), u_ref),
    "QNDF + jvp + GMRES" => () -> bench(prob_real, QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false), u_ref),
    "FBDF + jvp + GMRES" => () -> bench(prob_real, FBDF(linsolve = KrylovJL_GMRES(), concrete_jac = false), u_ref),
    "KenCarp47 + jvp + GMRES" => () -> bench(prob_real, KenCarp47(linsolve = KrylovJL_GMRES(), concrete_jac = false), u_ref),
    "TRBDF2 + jvp + GMRES" => () -> bench(prob_real, TRBDF2(linsolve = KrylovJL_GMRES(), concrete_jac = false), u_ref),
    "ETDRK4, dt = 1e-2" => () -> bench(prob_split, ETDRK4(), w_ref; dt = 1e-2),
    "ETDRK4, dt = 5e-3" => () -> bench(prob_split, ETDRK4(), w_ref; dt = 5e-3),
    "ETDRK4, dt = 2.5e-3" => () -> bench(prob_split, ETDRK4(), w_ref; dt = 2.5e-3),
    "ETDRK4, dt = 1.25e-3" => () -> bench(prob_split, ETDRK4(), w_ref; dt = 1.25e-3),
]

println(rpad("Method", 26), rpad("Return code", 13), lpad("Time (s)", 10), lpad("Rel. error", 12), lpad("Steps", 8))
println("-"^69)
for (name, run) in benchmarks
    retcode, elapsed, relerr, steps = run()
    println(
        rpad(name, 26), rpad(string(retcode), 13),
        lpad(round(elapsed, digits = 3), 10),
        lpad(round(relerr, sigdigits = 2), 12),
        lpad(steps, 8)
    )
end
