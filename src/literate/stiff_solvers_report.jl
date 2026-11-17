# # Stiff solvers, Jacobians and automatic differentiation

# This page reports a set of experiments on how to use the stiff (implicit) solvers of [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) with the pseudo-spectral right-hand side of [Evolution via method of lines](/pages/literated/tests_mol), and on the current state of automatic differentiation (AD) in Julia for this kind of code. The two approaches that came out of it are implemented in [Implicit solvers with the analytic Jacobian-vector product](/pages/literated/evolution_jvp) and in [Exponential time differencing with ETDRK4](/pages/literated/evolution_etdrk4).

# Most of the experiments described here are not executed when this page is built: their results were obtained once, on the machine described in the [Environment](#environment) section, and are reported as static tables. The timings, in particular, are indicative only: they are single runs, after a warm-up call that compiles each solver, with no attempt at statistics, and they depend on the hardware, on the number of threads and on the load of the machine. Only the relative comparison between the methods is meaningful, and even that may change from one machine to another. For comparison, the main benchmark is run again when this page is built, and its results are shown at the [end of the page](#timings_on_the_machine_that_built_this_page). The timings printed in the other two pages are also obtained when those pages are built, so they may differ from the static tables below.

# ## The problem

# Stiff solvers need the Jacobian of the right-hand side, or at least its action on vectors, and by default it is computed with forward-mode AD via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). The right-hand side `nsepervort_hat_rhs!` of [Evolution via method of lines](/pages/literated/tests_mol) has three features that get in the way:
#
# * the state is complex (the `rfft` half-spectrum);
# * it calls FFTW, a C library, through precomputed plans, which only accept `Float32`/`Float64` arrays;
# * it is in-place and writes into preallocated `Float64`/`ComplexF64` work arrays.
#
# It also has a fourth feature which is not related to AD, but which is unsafe with any adaptive or implicit solver: it mutates its *input*, by zeroing the dealiased modes of `vort_hat`. The new pages dealias the initial condition once instead; since the output of the right-hand side is dealiased, the state stays dealiased.

# ## Environment

# * Julia 1.13.3, on a 4-core Intel Xeon @ 2.10GHz (Linux), with 4 Julia threads; November 2026.
# * OrdinaryDiffEq v7.8.1 (OrdinaryDiffEqBDF v2.4.14, OrdinaryDiffEqSDIRK v2.9.8, OrdinaryDiffEqRosenbrock v2.7.5, OrdinaryDiffEqExponentialRK v2.4.0, OrdinaryDiffEqIMEXMultistep v2.3.0, OrdinaryDiffEqDifferentiation v3.12.5), SciMLBase v3.58.0, SciMLOperators v1.30.4, LinearSolve v5.18.3 (Krylov v0.10.12).
# * FFTW v1.10.0, AbstractFFTs v1.5.0.
# * ForwardDiff v1.5.0, Enzyme v0.13.220, Mooncake v0.5.70, DifferentiationInterface v0.7.22, ADTypes v1.24.0, PreallocationTools v1.7.1.
#
# Note that in OrdinaryDiffEq v7 only a few solvers are exported by `OrdinaryDiffEq` itself; `QNDF`, `FBDF`, `KenCarp47`, `TRBDF2`, `ETDRK4` and others must be loaded from their sub-packages (`OrdinaryDiffEqBDF`, `OrdinaryDiffEqSDIRK`, `OrdinaryDiffEqExponentialRK`, ...). Also, the `autodiff` keyword of the solvers takes an [ADTypes.jl](https://github.com/SciML/ADTypes.jl) object, such as `AutoForwardDiff()`, `AutoFiniteDiff()` or `AutoEnzyme()`, instead of a boolean.

# ## The current state of automatic differentiation

# The AD ecosystem in Julia is now organized around two packages: [ADTypes.jl](https://github.com/SciML/ADTypes.jl), which defines the types used to select a backend (`AutoForwardDiff()`, `AutoEnzyme()`, `AutoMooncake()`, `AutoFiniteDiff()`, ...), and [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl), a common interface to all the backends, which is what OrdinaryDiffEq uses internally to compute Jacobians and Jacobian-vector products. The main backends are:
#
# * [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl): forward mode, via dual numbers; mature and still the default in the SciML ecosystem.
# * [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl): forward and reverse modes, working at the level of the LLVM code; supports mutation and complex numbers.
# * [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl): reverse mode (and forward mode), source-to-source, supports mutation.
# * [Zygote.jl](https://github.com/FluxML/Zygote.jl): reverse mode, does not support mutation, so it is not suitable for in-place right-hand sides.
# * [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl): not AD, but available through the same interface as `AutoFiniteDiff()`.
#
# For differentiating *through* the solver, e.g. with respect to parameters, the relevant package is [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl). Sparse Jacobians are handled by `AutoSparse` together with [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl) and [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl), but they do not help here, since our Jacobian is dense.

# Here is what happened with the *original* right-hand side, unchanged:
#
# | Backend / test | Result |
# | :--- | :--- |
# | `Rodas5P()` with the default ForwardDiff | fails: `ForwardDiff.DerivativeConfig` has no method for a `ComplexF64` state |
# | same, with `ForwardDiff.can_dual(::Type{ComplexF64}) = true` (the old workaround in [Evolution via method of lines](/pages/literated/tests_mol)) | fails in the same way; the workaround no longer has any effect |
# | `rfft` of an `Array{<:ForwardDiff.Dual}` | fails: `type ForwardDiff.Dual{...} not supported` |
# | FFTW plan applied to an `Array{<:ForwardDiff.Dual}` | fails: no conversion from `Dual` to `Float64` |
# | `Rodas5P(autodiff = AutoEnzyme())` | fails: Enzyme has no derivative rule for the FFTW C library calls (`No forward mode derivative found for ... fftw_alignment_of`) |
# | Mooncake | not tested; it would also need rules for the FFTW calls |
# | `Rodas5P(autodiff = AutoFiniteDiff())`, $N = 32$ | works, building a dense Jacobian by finite differences |
#
# So, the issues from a few years ago are still there: ForwardDiff does not handle complex states in OrdinaryDiffEq, and neither ForwardDiff nor Enzyme can go through FFTW.

# ForwardDiff *can* be made to work, and we checked that it does (with `Rodas5P`, `QNDF` and `KenCarp4` at $N = 32$, and with Jacobian-free Newton-Krylov at $N = 128$), but this requires four changes to the right-hand side:
#
# 1. storing the state as a real array, `reinterpret(reshape, Float64, vort_hat)`, and viewing it as complex inside the right-hand side;
# 2. wrapping the work arrays with `PreallocationTools.DiffCache`, so they can hold dual numbers;
# 3. writing an FFT wrapper for arrays of dual numbers, which, using the linearity of the transform, applies the plan to the value and to each partial derivative separately (about twenty lines);
# 4. not mutating the input.
#
# The resulting Jacobian agreed with the one from finite differences to within $5 \times 10^{-5}$, relative, which is the accuracy of the finite differences. However, the right-hand side is *bilinear*, so the Jacobian is easy to write down by hand, and that is both simpler and faster. We did not pursue the AD version any further.

# ## The analytic Jacobian

# With $F(\omega) = g + \nu\Delta\omega - B(\omega, \omega)$, with $B$ bilinear, the Jacobian is $J(\omega)h = \nu\Delta h - B(\omega, h) - B(h, \omega)$. Each product $J(\omega)h$ costs six FFTs, against four for $F$. Since $F$ is quadratic, the centered difference $(F(\omega + \varepsilon h) - F(\omega - \varepsilon h))/2\varepsilon$ equals $J(\omega)h$ exactly, which gives a sharp test: the two agree to about $4 \times 10^{-14}$, relative.
#
# Three caveats:
#
# * **The Jacobian is dense.** The bilinear term is a convolution in spectral space. At $N = 32$, 29% of the entries of the Jacobian are nonzero. At $N = 128$, with 16640 real unknowns, the dense matrix would take about 2.2 GB, before any factorization. An attempt to use `KenCarp4` with a dense finite-difference Jacobian at $N = 128$ was still running after twenty minutes, using more than 5 GB, and was stopped. Hence the Jacobian should be given as a Jacobian-vector product, `ODEFunction(f; jvp = jvp!)`, and combined with a Krylov linear solver, `linsolve = KrylovJL_GMRES(), concrete_jac = false`.
# * **The state should be real.** The `rfft` half-spectrum represents a real field, so the maps involved are only real-linear: $J(\mathrm{i}h) \neq \mathrm{i}J(h)$, while Newton and GMRES iterations on a complex vector assume complex linearity. Reinterpreting the complex array as a real array of size $(2, M, N)$ costs nothing.
# * **Rosenbrock methods still need $\partial F/\partial t$.** Even with the Jacobian given via `jac`, `Rodas5P` calls AD for the time derivative and fails. Since the forcing is steady, we pass `tgrad = (dT, u, p, t) -> fill!(dT, 0)`.

# ## Timings

# All runs are for $N = 128$ (`Nsub = 84`), $\nu = 1$, $t \in [0, 1]$, starting from a random four-mode vorticity field and with a single-mode steady forcing, as in [Evolution via method of lines](/pages/literated/tests_mol). The relative error is measured at $t = 1$ against a reference solution computed with `Vern9()` and `abstol = reltol = 1e-12`. Adaptive methods use the default tolerances.

# These tables were obtained on the machine described in the [Environment](#environment) section and are kept here as a record; see the [end of the page](#timings_on_the_machine_that_built_this_page) for the same benchmark run on the machine that built this page.

# ### Matrix-free implicit methods with the analytic Jacobian-vector product
#
# The number of evaluations is the solver's own count, `sol.stats.nf`, which includes the Jacobian-vector products; most of the evaluations are, in fact, Jacobian-vector products, one per GMRES iteration (in a separate run with counters, for instance, `QNDF` used 217 evaluations of $F$ and 1475 of $J(\omega)h$; the counts vary slightly from run to run, since multithreaded FFTs change the round-off errors and, with them, the number of GMRES iterations).
#
# | Method | Time (s) | Relative error | Evaluations ($F$ + jvp) | Steps |
# | :--- | ---: | ---: | ---: | ---: |
# | `Tsit5()` (explicit) | 1.54 | 1.6e-9 | 5871 | 978 |
# | `Vern9()` (explicit) | 3.05 | 2.5e-11 | 12402 | 775 |
# | `QNDF` + jvp + GMRES | 0.89 | 3.0e-7 | 1817 | 186 |
# | `FBDF` + jvp + GMRES | 1.60 | 8.2e-6 | 2633 | 224 |
# | `KenCarp47` + jvp + GMRES | 1.54 | 2.9e-7 | 3303 | 19 |
# | `TRBDF2` + jvp + GMRES | 1.06 | 1.9e-5 | 2429 | 28 |
#
# On $t \in [0, 10]$, converging to the steady state, `QNDF` + jvp + GMRES took 220 steps and reached a distance of about $8 \times 10^{-7}$ from the steady state.

# ### Exponential time differencing
#
# Here the stiff linear part $\nu\Delta$, diagonal in spectral space, is passed as a `DiagonalOperator` in a `SplitODEProblem` and integrated exactly, and no Jacobian is needed at all. `ETDRK4` is used with a fixed time step.
#
# | Method | Time (s) | Relative error | Steps |
# | :--- | ---: | ---: | ---: |
# | `Tsit5()` (explicit) | 1.42 | 4.2e-10 | 979 |
# | `Vern9()` (explicit) | 2.88 | 1.5e-11 | 775 |
# | `ETDRK4()`, `dt = 1e-2` | 0.17 | 8.2e-5 | 100 |
# | `ETDRK4()`, `dt = 5e-3` | 0.26 | 5.5e-6 | 200 |
# | `ETDRK4()`, `dt = 2.5e-3` | 0.38 | 4.0e-7 | 401 |
# | `ETDRK4()`, `dt = 1.25e-3` | 0.82 | 2.8e-8 | 801 |
#
# The errors decrease at the expected fourth order. On $t \in [0, 10]$, with `dt = 1e-2`, `ETDRK4` reached a distance of about $8 \times 10^{-7}$ from the steady state.
#
# With $\nu = 0.1$ (in an earlier run, with a different random initial condition), on the other hand, `dt = 1e-2` is too large and the solution blows up (`ReturnCode.Unstable`), while `dt = 1e-3` gives a relative error of $3.7 \times 10^{-5}$ in about 1.6 s, against 0.35 s for `Tsit5()` with an error of $6.7 \times 10^{-6}$: with less viscosity the problem is less stiff and the time step is limited by the nonlinear term, which favours the explicit, adaptive methods.
#
# The IMEX multistep method `CNAB2()`, with `dt = 1e-3`, worked but was very slow in this setup (about 87 s for $t \in [0, 1]$, with a relative error of $7 \times 10^{-4}$); we did not investigate why.

# ### Small grids, dense Jacobian
#
# At $N = 32$, with the dense Jacobian assembled column by column from the analytic jvp, `QNDF`, `KenCarp4` and `FBDF` work, as does `Rodas5P`, once `tgrad` is given.

# ### A turbulent case
#
# With $\nu = 0.01$, the initial vorticity used here is large enough for the flow to be turbulent. All methods, including `Vern9()`, then give relative errors of order one at $t = 1$, at the default tolerances, compared with the tight-tolerance reference. This reflects the sensitivity to the initial conditions of the flow, not a failure of the methods, and such cases are not useful for comparing accuracy.

# ## Conclusions

# * No AD is needed: since the nonlinear term is bilinear, the Jacobian-vector product is a few lines of code and costs about one and a half evaluations of the right-hand side.
# * The Jacobian is dense, so it should be used matrix-free, with a Krylov linear solver, except on small grids.
# * The state should be stored as a real array for the implicit methods.
# * For the viscous, moderately stiff case $\nu = 1$, $N = 128$, `QNDF` with the analytic jvp and GMRES was the fastest of the implicit methods, somewhat faster than `Tsit5()`, and `ETDRK4` was faster still, for comparable accuracy. The advantage of the stiff and exponential methods grows with $N$ and $\nu$ (the stiffness scales with $\nu N^2$) and in long runs towards a steady state; for less viscous, more nonlinear regimes, explicit adaptive methods remain competitive.
# * If AD becomes necessary, for instance for sensitivities with respect to parameters, the obstacle is still the FFT: neither ForwardDiff nor Enzyme can differentiate through FFTW plans out of the box.

# ## Timings on the machine that built this page

# Here, the main benchmark is run again, when this page is built. The code is a compact version of that in [Implicit solvers with the analytic Jacobian-vector product](/pages/literated/evolution_jvp) and [Exponential time differencing with ETDRK4](/pages/literated/evolution_etdrk4), to which we refer for the details. To reduce the noise a bit, each method is run once to compile it, and then three more times, and the minimum of the three elapsed times is reported.

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
