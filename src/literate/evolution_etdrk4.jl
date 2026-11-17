# # Exponential time differencing with ETDRK4

# In the vorticity equation in spectral form,
#
# $$
# \frac{\mathrm{d}\hat\omega}{\mathrm{d}t} = \nu\hat\Delta\hat\omega + \hat g - \hat B(\hat\omega, \hat\omega),
# $$
#
# the stiffness comes from the viscous term $\nu\hat\Delta\hat\omega$, whose eigenvalues $-\nu\kappa_0^2|\mathbf{k}|^2$ grow quadratically with the wavenumber. This term, however, is linear and *diagonal* in spectral space, so it can be integrated exactly. Exponential time differencing (ETD) schemes treat the linear part exactly, via the variation of constants formula, and approximate the remaining, nonlinear and non-stiff part, $N(\hat\omega) = \hat g - \hat B(\hat\omega, \hat\omega)$:
#
# $$
# \hat\omega(t + \tau) = e^{\tau\nu\hat\Delta}\hat\omega(t) + \int_0^\tau e^{(\tau - s)\nu\hat\Delta} N(\hat\omega(t + s))\;\mathrm{d}s.
# $$
#
# The fourth-order Runge-Kutta version of this approach, ETDRK4, from [Cox & Matthews (2002)](https://doi.org/10.1006/jcph.2002.6995), in the stable form of [Kassam & Trefethen (2005)](https://doi.org/10.1137/S1064827502410633), is a classical choice for pseudo-spectral simulations. It is implemented in [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) as `ETDRK4()`, for a `SplitODEProblem` with the linear part given as an operator.

# This approach needs neither the Jacobian nor automatic differentiation. Since there are no Newton iterations, the state can be kept as the complex half-spectrum.

# ## Packages

using FFTW
using Plots
using OrdinaryDiffEq
using OrdinaryDiffEqExponentialRK: ETDRK4
using SciMLOperators: DiagonalOperator
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

# ## Parameters and work arrays

function get_params(N, Nsub, g_hat, operators)
    M = div(N, 2) + 1
    phys() = zeros(N, N)
    spec() = zeros(ComplexF64, M, N)
    plan = plan_rfft(phys(), flags=FFTW.MEASURE)
    plan_inv = plan_irfft(spec(), N, flags=FFTW.MEASURE)
    return (;
        operators, g_hat, N, Nsub, M, plan, plan_inv,
        u = phys(), v = phys(), a = phys(), b = phys(),
        sc = spec(), a_hat = spec(), b_hat = spec()
    )
end

# ## The nonlinear part of the right-hand side

# The linear operator in DifferentialEquations.jl acts on vectors, so the state is the vectorized half-spectrum `vec(vort_hat)`, which we reshape back to a matrix inside the right-hand side (no copy is involved). The initial condition is dealiased and both the linear and the nonlinear parts preserve that.

function nsepervort_nonlinear!(dw, w, params, t)
    (; operators, g_hat, N, Nsub, M, plan, plan_inv, u, v, a, b, sc, a_hat, b_hat) = params
    _, _, _, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = operators
    w_hat = reshape(w, M, N)
    dw_hat = reshape(dw, M, N)

    sc .= Hu_hat .* w_hat
    mul!(u, plan_inv, sc)
    sc .= Hv_hat .* w_hat
    mul!(v, plan_inv, sc)

    @. a = u * v
    @. b = v^2 - u^2
    mul!(a_hat, plan, a)
    mul!(b_hat, plan, b)

    @. dw_hat = g_hat - DxsqDysq_hat * a_hat - Dxy_hat * b_hat
    dealias!(dw_hat, N, Nsub)
    return dw
end

# For comparison with explicit methods, we also define the full right-hand side, linear plus nonlinear parts.

function nsepervort_full!(dw, w, params, t)
    nsepervort_nonlinear!(dw, w, params, t)
    dw .+= params.nu_Delta .* w
    return dw
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

nu_Delta = vec(complex.(ν .* Delta_hat))

params = (; get_params(N, Nsub, g_steady_hat, operators)..., nu_Delta)

w0 = vec(copy(vort_init_hat))
nothing

# ## The split problem

# The linear part is the diagonal operator $\nu\hat\Delta$:

A = DiagonalOperator(nu_Delta)

tspan = (0.0, 1.0)

prob = SplitODEProblem(A, nsepervort_nonlinear!, w0, tspan, params)
nothing

# Reference solution, with an explicit high-order method and tight tolerances:

prob_full = ODEProblem(nsepervort_full!, w0, tspan, params)
sol_ref = solve(prob_full, Vern9(), abstol = 1e-12, reltol = 1e-12, save_everystep = false)
w_ref = sol_ref.u[end]
nothing

# ## Accuracy and timings

# ETDRK4 is used here with a fixed time step. We compare it with adaptive explicit methods at their default tolerances. We time the second call to each solver, so compilation is not included. Each ETDRK4 step costs four evaluations of the nonlinear part, plus products with the precomputed diagonal exponentials, while each Tsit5 step costs six evaluations of the full right-hand side and each Vern9 step sixteen. The timings are measured when this page is built, so they depend on the machine that builds it; see the [report on stiff solvers and automatic differentiation](/pages/literated/stiff_solvers_report) for a discussion.

function run(prob, alg; kwargs...)
    solve(prob, alg; save_everystep = false, kwargs...)
    elapsed = @elapsed sol = solve(prob, alg; save_everystep = false, kwargs...)
    relerr = norm(sol.u[end] .- w_ref) / norm(w_ref)
    return sol, elapsed, relerr
end

function report(name, sol, elapsed, relerr)
    println(
        rpad(name, 22),
        rpad(string(sol.retcode), 10),
        "time: ", rpad(round(elapsed, digits = 3), 7), "s   ",
        "relerr: ", rpad(round(relerr, sigdigits = 2), 9),
        "steps: ", sol.stats.naccept
    )
end

for (name, alg) in ("Tsit5 (explicit)" => Tsit5(), "Vern9 (explicit)" => Vern9())
    report(name, run(prob_full, alg)...)
end

dts = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
relerrs = Float64[]
for dt in dts
    sol, elapsed, relerr = run(prob, ETDRK4(); dt)
    report("ETDRK4 dt=$dt", sol, elapsed, relerr)
    push!(relerrs, relerr)
end

# The observed order of convergence, which should approach 4 as the time step decreases:

log2.(relerrs[1:end-1] ./ relerrs[2:end])

#

plot(dts, relerrs, xscale = :log10, yscale = :log10, marker = :circle, xlabel = "time step", ylabel = "relative error at t = 1", label = "ETDRK4", legend = :topleft, title = "Convergence of ETDRK4", titlefont = 10)
plot!(dts, relerrs[end] .* (dts ./ dts[end]).^4, linestyle = :dash, label = "slope 4")

# ## Convergence to the steady state

# Over a longer time interval, the solution converges to the steady state $\omega_\infty = -\Delta^{-1}g$. A fixed time step of $10^{-2}$ is enough to follow it, with a thousand steps up to $t = 10$.

prob_long = SplitODEProblem(A, nsepervort_nonlinear!, w0, (0.0, 10.0), params)
sol = solve(prob_long, ETDRK4(), dt = 1e-2, saveat = 0.05)

sol.retcode

# Distance to the steady state at the final time, in spectral space and in physical space:

norm(reshape(sol.u[end], :, N) .- vort_steady_hat) * L / N / N

#

vort = irfft(reshape(sol.u[end], :, N), N)
norm(vort .- vort_steady) * L / N

#

dist = [norm(reshape(w, :, N) .- vort_steady_hat) * L / N / N for w in sol.u]

plot(sol.t, dist, yscale = :log10, xlabel = "time", ylabel = "L² distance", title = "Convergence to the steady state (ETDRK4)", titlefont = 10, label = false)

#

heatmap(x, y, vort, xlabel="x", ylabel="y", title="vorticity", titlefont=12)
