# # Implicit solvers with the analytic Jacobian-vector product

# Stiff solvers in [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl), such as BDF, SDIRK and Rosenbrock methods, need the Jacobian of the right-hand side of the equation, or at least its action on vectors. By default the Jacobian is computed with automatic differentiation, which does not go through the complex arithmetic and the FFTW calls of our right-hand side (see the [report on stiff solvers and automatic differentiation](/pages/literated/stiff_solvers_report)).

# We do not need automatic differentiation, though. The vorticity equation in spectral form reads
#
# $$
# \frac{\mathrm{d}\hat\omega}{\mathrm{d}t} = F(\hat\omega) = \hat g + \nu\hat\Delta\hat\omega - \hat B(\hat\omega, \hat\omega),
# $$
#
# where $B$ is bilinear. In Basdevant's formulation,
#
# $$
# B(\omega, \omega) = \left(\partial_x^2 - \partial_y^2\right)(uv) + \partial_{xy}\left(v^2 - u^2\right),
# $$
#
# with $(u, v)$ the velocity field recovered from $\omega$. Since $B$ is bilinear, the Jacobian of $F$ at $\omega$ applied to a direction $h$ is simply
#
# $$
# J(\omega) h = \nu\Delta h - B(\omega, h) - B(h, \omega),
# $$
#
# and, in Basdevant's form,
#
# $$
# B(\omega, h) + B(h, \omega) = \left(\partial_x^2 - \partial_y^2\right)\left(u_\omega v_h + u_h v_\omega\right) + \partial_{xy}\left(2v_\omega v_h - 2u_\omega u_h\right),
# $$
#
# where $(u_\omega, v_\omega)$ and $(u_h, v_h)$ are the velocity fields associated with $\omega$ and $h$. Computing $J(\omega)h$ costs about one and a half evaluations of $F$: four inverse and two direct FFTs, against two inverse and two direct FFTs for $F$.

# Two remarks guide the implementation:
#
# * **The Jacobian is a dense matrix.** The bilinear term is a convolution in spectral space, so every mode interacts with every other mode. For $N = 128$, the state has $65 \times 128$ complex entries, i.e. $16\,640$ real unknowns, and the dense Jacobian would take about 2.2 GB, before any LU factorization. So we do not assemble it. Instead, we give the solver the Jacobian-vector product (`jvp`) and use a Krylov method (GMRES) for the linear systems in the Newton iterations (a *Jacobian-free Newton-Krylov* method).
#
# * **The state should be real.** The `rfft` half-spectrum represents a *real* field, so the hidden half of the spectrum holds complex conjugates and the maps involved are only real-linear: $J(\mathrm{i}h) \neq \mathrm{i}J(h)$. Newton and GMRES iterations on a complex vector assume complex linearity. Hence, we store the state as a real array of size $(2, M, N)$, via `reinterpret(reshape, Float64, vort_hat)`, and view it as complex inside the right-hand side. This costs nothing, since it is the same memory.

# ## Packages

using FFTW
using Plots
using OrdinaryDiffEq
using OrdinaryDiffEqBDF: QNDF, FBDF
using OrdinaryDiffEqSDIRK: KenCarp47, TRBDF2
using LinearSolve: KrylovJL_GMRES
using LinearAlgebra: norm, mul!
using Random

@info "Threads: $(FFTW.nthreads())"

# ## Operators

function get_operators(N, κ₀)
    ## Differentiation in spectral space
    Dx_hat = im * κ₀ * [
        ifelse(k1 ≤ div(N, 2) + 1, k1 - 1, k1 - 1 - N) for k2 in 1:div(N, 2)+1, k1 in 1:N
    ]
    Dy_hat = im * κ₀ * [k2 - 1 for k2 in 1:div(N, 2)+1, k1 in 1:N]

    Delta_hat = - κ₀^2 * [
        ifelse(k1 ≤ div(N, 2) + 1, (k1 - 1)^2 + (k2 - 1)^2, (k1 - 1 - N)^2 + (k2 - 1)^2)
        for k2 in 1:div(N, 2)+1, k1 in 1:N
    ]

    ## For the Basdevant formulation
    DxsqDysq_hat = Dx_hat.^2 .- Dy_hat.^2
    Dxy_hat = Dx_hat .* Dy_hat

    ## Recovering of the velocity field from the vorticity
    Hu_hat = - Dy_hat ./ Delta_hat
    Hu_hat[1, 1] = 0.0
    Hv_hat = Dx_hat ./ Delta_hat
    Hv_hat[1, 1] = 0.0

    return Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat
end

# Methods to generate a scalar field from a list of wavenumbers and amplitudes

function field_from_modes(L, N, modes::Matrix{<:Integer}, amps::Matrix{<:Real})
    κ₀ = 2π/L
    x = y = (L/N):(L/N):L
    field = sum(
        [
            2κ₀^2 * (k[1]^2 + k[2]^2) * (
                a[1] * cos.(κ₀ * (k[1] * one.(y) * x' + k[2] * y * one.(x)'))
                - a[2] * sin.(κ₀ * (k[1] * one.(y) * x' + k[2] * y * one.(x)'))
            )
            for (k, a) in zip(eachrow(modes), eachrow(amps))
        ]
    )
    return field
end

function field_from_modes(rng::AbstractRNG, L, N, num_modes::Int)
    modes = rand(rng, 1:div(N,10), num_modes, 2)
    amps = rand(rng, num_modes, 2)
    field = field_from_modes(L, N, modes, amps)
    return field
end

field_from_modes(L, N, num_modes::Int) = field_from_modes(Xoshiro(), L, N, num_modes)

# Dealiasing, by zeroing the modes outside the retained range

function dealias!(w_hat, N, Nsub)
    w_hat[div(Nsub,2) + 1:end, :] .= 0
    w_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0
    return w_hat
end

# Conversion between the complex half-spectrum and the real state vector, which share the same memory layout:

to_real(w_hat) = collect(reinterpret(reshape, Float64, w_hat))
to_complex(u) = reinterpret(reshape, ComplexF64, u)
nothing

# ## Parameters and work arrays

function get_params(N, Nsub, ν, g_hat, operators)
    M = div(N, 2) + 1
    phys() = zeros(N, N)
    spec() = zeros(ComplexF64, M, N)
    plan = plan_rfft(phys(), flags=FFTW.MEASURE)
    plan_inv = plan_irfft(spec(), N, flags=FFTW.MEASURE)
    return (;
        operators, ν, g_hat, N, Nsub, plan, plan_inv,
        uw = phys(), vw = phys(), uh = phys(), vh = phys(), a = phys(), b = phys(),
        sc = spec(), a_hat = spec(), b_hat = spec()
    )
end

# ## The right-hand side and its Jacobian-vector product

# The right-hand side is the same as in [Evolution via method of lines](/pages/literated/tests_mol), except that it acts on the real state vector and that it does not modify its input. The input is assumed to be already dealiased, which is preserved by the evolution since the output is dealiased.

function nsepervort_rhs!(du, u, params, t)
    (; operators, ν, g_hat, N, Nsub, plan, plan_inv, uw, vw, a, b, sc, a_hat, b_hat) = params
    _, _, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = operators
    w_hat = reinterpret(reshape, ComplexF64, u)
    dw_hat = reinterpret(reshape, ComplexF64, du)

    sc .= Hu_hat .* w_hat
    mul!(uw, plan_inv, sc)
    sc .= Hv_hat .* w_hat
    mul!(vw, plan_inv, sc)

    @. a = uw * vw
    @. b = vw^2 - uw^2
    mul!(a_hat, plan, a)
    mul!(b_hat, plan, b)

    @. dw_hat = g_hat + ν * Delta_hat * w_hat - DxsqDysq_hat * a_hat - Dxy_hat * b_hat
    dealias!(dw_hat, N, Nsub)
    return du
end

# The Jacobian-vector product follows the formula above. The signature `(Jv, v, u, p, t)` is the one expected by the `jvp` field of an `ODEFunction`. The solver may pass `v` and `Jv` as flat vectors, so we reshape them first.

function nsepervort_jvp!(Jv, v, u, params, t)
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

# ## The spatial domain and its discretization

L = 2π
κ₀ = 2π/L
N = 128
Nsub = 84
x = y = (L/N):(L/N):L

ν = 1.0e-0 # viscosity

operators = get_operators(N, κ₀)
Delta_hat = operators[3]

rng = Xoshiro(123)

vort_init_hat = dealias!(rfft(field_from_modes(rng, L, N, 4)), N, Nsub)

g_steady_hat = dealias!(rfft(field_from_modes(rng, L, N, 1)), N, Nsub)

vort_steady_hat = - g_steady_hat ./ Delta_hat
vort_steady_hat[1, 1] = 0.0im
vort_steady = irfft(vort_steady_hat, N)

params = get_params(N, Nsub, ν, g_steady_hat, operators)

u0 = to_real(vort_init_hat)
nothing

# ## Checking the Jacobian-vector product

# Since $F$ is quadratic, the centered difference quotient $(F(\omega + \varepsilon h) - F(\omega - \varepsilon h))/(2\varepsilon)$ is *exactly* $J(\omega)h$, for any $\varepsilon$, up to round-off errors. This gives a sharp test of the implementation.

h = to_real(dealias!(rfft(field_from_modes(rng, L, N, 3)), N, Nsub))
Jh = similar(u0)
nsepervort_jvp!(Jh, h, u0, params, 0.0)

ε = 1.0e-2
Fp = nsepervort_rhs!(similar(u0), u0 .+ ε .* h, params, 0.0)
Fm = nsepervort_rhs!(similar(u0), u0 .- ε .* h, params, 0.0)

norm(Jh .- (Fp .- Fm) ./ 2ε) / norm(Jh)

# ## Solving with matrix-free implicit methods

# We pass the `jvp` to the `ODEFunction` and choose GMRES, from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl), as the linear solver, with `concrete_jac = false`, so that no Jacobian matrix is ever built.

tspan = (0.0, 1.0)

fun = ODEFunction(nsepervort_rhs!; jvp = nsepervort_jvp!)
prob = ODEProblem(fun, u0, tspan, params)
nothing

# For comparison, we compute a reference solution with an explicit high-order method and tight tolerances.

sol_ref = solve(prob, Vern9(), abstol = 1e-12, reltol = 1e-12, save_everystep = false)
u_ref = sol_ref.u[end]
nothing

# Now we solve with a few methods, at the default tolerances, and record the elapsed time (the first call to each solver compiles it, so we time the second call), the relative error at the final time and the number of evaluations counted by the solver, which, for the implicit methods, includes the Jacobian-vector products (most of them, in fact, since they are used in every GMRES iteration). The timings are measured when this page is built, so they depend on the machine that builds it; see the [report on stiff solvers and automatic differentiation](/pages/literated/stiff_solvers_report) for a discussion.

methods = [
    "Tsit5 (explicit)" => Tsit5(),
    "Vern9 (explicit)" => Vern9(),
    "QNDF + jvp + GMRES" => QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false),
    "FBDF + jvp + GMRES" => FBDF(linsolve = KrylovJL_GMRES(), concrete_jac = false),
    "KenCarp47 + jvp + GMRES" => KenCarp47(linsolve = KrylovJL_GMRES(), concrete_jac = false),
    "TRBDF2 + jvp + GMRES" => TRBDF2(linsolve = KrylovJL_GMRES(), concrete_jac = false),
]

for (name, alg) in methods
    solve(prob, alg, save_everystep = false)
    elapsed = @elapsed sol = solve(prob, alg, save_everystep = false)
    relerr = norm(sol.u[end] .- u_ref) / norm(u_ref)
    println(
        rpad(name, 26),
        rpad(string(sol.retcode), 10),
        "time: ", rpad(round(elapsed, digits = 3), 7), "s   ",
        "relerr: ", rpad(round(relerr, sigdigits = 2), 9),
        "f + jvp evals: ", rpad(sol.stats.nf, 7),
        "steps: ", sol.stats.naccept
    )
end

# ## Convergence to the steady state

# The forcing has a single mode and the viscosity is large, so the solution converges to the steady state $\omega_\infty = -\Delta^{-1}g$. With the implicit method, we can take large time steps once the transient has decayed.

prob_long = ODEProblem(fun, u0, (0.0, 10.0), params)
sol = solve(prob_long, QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false))

sol.retcode

# Number of time steps and of evaluations of the right-hand side and of the Jacobian-vector product:

sol.stats.naccept, sol.stats.nf

# Distance to the steady state at the final time, in spectral space and in physical space:

norm(to_complex(sol.u[end]) .- vort_steady_hat) * L / N / N

#

vort = irfft(to_complex(sol.u[end]), N)
norm(vort .- vort_steady) * L / N

#

dist = [norm(to_complex(u) .- vort_steady_hat) * L / N / N for u in sol.u]

plot(sol.t, dist, yscale = :log10, marker = :circle, markersize = 2, xlabel = "time", ylabel = "L² distance", title = "Convergence to the steady state (QNDF + jvp + GMRES)", titlefont = 10, label = false)

#

heatmap(x, y, vort, xlabel="x", ylabel="y", title="vorticity", titlefont=12)

# ## A dense Jacobian for small grids

# For small grids, the dense Jacobian is affordable and can be assembled column by column from the `jvp`. This is useful for methods that require a Jacobian matrix, such as the Rosenbrock methods. Rosenbrock methods also need the time derivative $\partial F/\partial t$, which would otherwise be computed via automatic differentiation; here the forcing is steady, so it vanishes and we pass it explicitly with `tgrad`.

function nsepervort_jac!(J, u, params, t)
    e = zero(u)
    for k in eachindex(u)
        e[k] = 1
        nsepervort_jvp!(view(J, :, k), e, u, params, t)
        e[k] = 0
    end
    return J
end

nsepervort_tgrad!(dT, u, params, t) = fill!(dT, 0)

N_small = 32
Nsub_small = 20
operators_small = get_operators(N_small, κ₀)
vort_init_hat_small = dealias!(rfft(field_from_modes(rng, L, N_small, 4)), N_small, Nsub_small)
g_hat_small = dealias!(rfft(field_from_modes(rng, L, N_small, 1)), N_small, Nsub_small)
params_small = get_params(N_small, Nsub_small, ν, g_hat_small, operators_small)
u0_small = to_real(vort_init_hat_small)

fun_small = ODEFunction(nsepervort_rhs!; jac = nsepervort_jac!, tgrad = nsepervort_tgrad!)
prob_small = ODEProblem(fun_small, u0_small, (0.0, 1.0), params_small)

sol_small = solve(prob_small, Rodas5P())

sol_small.retcode, sol_small.stats.njacs

# The Jacobian has $(2 \times 17 \times 32)^2 = 1088^2$ entries, a good fraction of which are nonzero, as expected from the convolution structure:

J_small = zeros(length(u0_small), length(u0_small))
nsepervort_jac!(J_small, u0_small, params_small, 0.0)
count(!iszero, J_small) / length(J_small)
