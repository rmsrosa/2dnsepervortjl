# # Evolution via method of lines

using FFTW
using Plots
using OrdinaryDiffEq
using LinearAlgebra: norm, mul!
using Test
using BenchmarkTools
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

# Differential equations

function nsepervort_hat_rhs!(dvorhattdt, vort_hat, params, t)
    operators, vars, auxs, plans = params
    
    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    ν, g_hat, N, Nsub = vars
    u_hat, v_hat, u, v, uv, v2u2, uv_hat, v2u2_hat = auxs
    plan, plan_inv = plans

    vort_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    vort_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im

    u_hat .= Hu_hat .* vort_hat
    v_hat .= Hv_hat .* vort_hat

    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)

    uv .= u .* v
    v2u2 .= v.^2 .- u.^2
    mul!(uv_hat, plan, uv)
    mul!(v2u2_hat, plan, v2u2)

    dvorhattdt .= g_hat .+ ν .* Delta_hat .* vort_hat .- DxsqDysq_hat .* uv_hat .- Dxy_hat .* v2u2_hat

    ## dealiasing
    dvorhattdt[div(Nsub,2) + 1:end, :] .= 0.0im
    dvorhattdt[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im

    return dvorhattdt
end

# ## The spatial domain and its discretization

L = 2π
κ₀ = 2π/L
N = 128
Nsub = 84
x = y = (L/N):(L/N):L

# ## Test convergence to one-mode steady state

ν = 1.0e-0 # viscosity

Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = get_operators(N, κ₀)

vort_init = field_from_modes(L, N, 4)
vort_init_hat = rfft(vort_init)

g_steady = field_from_modes(L, N, 1)
g_steady_hat = rfft(g_steady)

vort_steady_hat = - g_steady_hat ./ Delta_hat
vort_steady_hat[1, 1] = 0.0im
vort_steady = irfft(vort_steady_hat, N)

vort_hat = copy(vort_init_hat)
vort = irfft(vort_hat, N)

u_hat = similar(vort_hat)
v_hat = similar(vort_hat)
u = similar(vort)
v = similar(vort)
uv = similar(vort)
v2u2 = similar(vort)
uv_hat = similar(vort_hat)
v2u2_hat = similar(vort_hat)
plan = plan_rfft(vort, flags=FFTW.MEASURE)
plan_inv = plan_irfft(vort_hat, N, flags=FFTW.MEASURE)

operators = Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat
vars = ν, g_steady_hat, N, Nsub
auxs = u_hat, v_hat, u, v, uv, v2u2, uv_hat, v2u2_hat
plans = plan, plan_inv

params = (
    operators,
    vars,
    auxs,
    plans
)

nothing
#

tspan = (0.0, 10.0)

prob = ODEProblem(nsepervort_hat_rhs!, vort_init_hat, tspan, params)
nothing

#

OrdinaryDiffEq.ForwardDiff.can_dual(::Type{ComplexF64}) = true

sol = solve(prob, Vern9()) # QNDF() etc.

sol.retcode

# Distância entre a solução estacionária e a solução ao final do intervalo de tempo.

norm(sol.u[end] - vort_steady_hat) * L/N /N

#

vort = irfft(sol.u[end], N)
nothing

# No espaço físico:

norm(vort .- vort_steady) * L / N

# Na norma do máximo:

maximum(abs, vort .- vort_steady)

#

heatmap(x, y, vort, xlabel="x", ylabel="y", title="vorticity", titlefont=12)

#

surface(x, y, vort, xlabel="x", ylabel="y", title="vorticity", titlefont=12)

#

surface(x, y, vort .- vort_steady, xlabel="x", ylabel="y", title="vorticity", titlefont=12)

# We can check how the $L^2$ distance of the vorticity to the steady state evolves.

error = [norm(sol.u[n] .- vort_steady_hat) * L/N / N for n in eachindex(sol.u)]
nothing

#

plt1 = plot(sol.t, error, title="Enstrophy convergence", titlefont = 10, xaxis="time", yaxis = "enstrophy", label=false)

plt2 = plot(sol.t[1:div(end, 100)], error[1:div(end, 100)], title="Enstrophy convergence", titlefont = 10, xaxis="time", yaxis = "enstrophy", label=false)

plt3 = plot(sol.t[div(end,100):div(end, 50)], error[div(end,100):div(end, 50)], title="Enstrophy convergence", titlefont = 10, xaxis="time", yaxis = "enstrophy", label=false)

plt4 = plot(sol.t[end-1000:end], error[end-1000:end], title="Enstrophy convergence", titlefont = 10, xaxis="time", yaxis = "enstrophy", label=false)

plot(plt1, plt2, plt3, plt4, layout = 4)
#