# # Testing FFTW for fully periodic fluid flows

# Here we test the convergence in the case of a known globally stable steady state.

# Here are the packages we are gonna need.

using FFTW
using Plots
using Test
using Random

# ## The spatial domain

L = 2π
κ₀ = 2π/L
N = 128
Nsub = 84
x = y = (L/N):(L/N):L

# Operators

Dx_hat = im * κ₀ * [ifelse(k1 ≤ N/2 + 1, k1 - 1, k1 - 1 - N) for k2 in 1:N/2+1, k1 in 1:N]
Dy_hat = im * κ₀ * [k2 - 1 for k2 in 1:N/2+1, k1 in 1:N]

Delta_hat = - κ₀^2 * [
    ifelse(k1 ≤ N/2 + 1, (k1 - 1)^2 + (k2 - 1)^2, (k1 - 1 - N)^2 + (k2 - 1)^2)
    for k2 in 1:N/2+1, k1 in 1:N
]

Hu_hat = - Dy_hat ./ Delta_hat
Hu_hat[1, 1] = 0.0
Hv_hat = Dx_hat ./ Delta_hat
Hv_hat[1, 1] = 0.0

# One-mode steady state

ν = 1.0e-0 # viscosity
κ = (x = 1, y = 2) # forced mode
α = (re = 0.1, im = 0.05) # strength

vort_steady = 2κ₀^2 * sum(abs2, κ) * (
    α.re * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - α.im * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)
vort_steady_hat = rfft(vort_steady)

display(heatmap(x, y, vort_steady, xlabel="x", ylabel="y", title="steady state vorticity", titlefont=12))

g_steady = ν * (
    2α.re * κ₀^4 * sum(abs2, κ)^2 * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - 2α.im * κ₀^4 * sum(abs2, κ)^2 * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)
g_steady_hat = rfft(g_steady)

display(heatmap(x, y, g_steady, xlabel="x", ylabel="y", title="forcing term", titlefont=12))

Gr = sqrt(
    2 * sum(abs2, α) / κ₀^4 / ν^4
)

@info "Grashof number: $(round(Gr, digits=4))"

# evolution step

function step!(vort_hat, dt, params)
    Exp_nu_dt_Delta_hat, Dx_hat, Dy_hat, Hu_hat, Hv_hat, g_hat, N, Nsub = params
    u_hat = Hu_hat .* vort_hat
    v_hat = Hv_hat .* vort_hat
    u = irfft(u_hat, N)
    v = irfft(v_hat, N)
    vort = irfft(vort_hat, N)
    wu_hat = rfft(vort .* u)
    wv_hat = rfft(vort .* v)
    vort_hat .= Exp_nu_dt_Delta_hat .* ( 
        vort_hat .+ dt * ( 
            g_hat .- Dx_hat .* wu_hat .- Dy_hat .* wv_hat
        )
    )
    # dealiasing
    vort_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    vort_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im
    return vort_hat
end

# Initial vorticity

rng = Xoshiro(123)
num_modes = 4
vort_init = sum(
    [
        2κ₀^2 * (kx^2 + ky^2) * (
            ar * cos.(κ₀ * (kx * one.(y) * x' + ky * y * one.(x)'))
            - ai * sin.(κ₀ * (kx * one.(y) * x' + ky * y * one.(x)'))
        )
        for (kx, ky, ar, ai) in zip(
            rand(rng, 1:div(N,10), num_modes),
            rand(rng, 1:div(N,10), num_modes),
            rand(rng, num_modes),
            rand(rng, num_modes)
        )
    ]
)

vort_init_hat = rfft(vort_init)

display(heatmap(x, y, vort_init, xlabel="x", ylabel="y", title="initial vorticity", titlefont=12))

# Time evolution setup

dt = 1.0e-5 # 1.0e-3 a 1.0e-5
t_final = 3.0
Exp_nu_dt_Delta_hat = exp.(ν * dt * Delta_hat)
params = Exp_nu_dt_Delta_hat, Dx_hat, Dy_hat, Hu_hat, Hv_hat, g_steady_hat, N, Nsub
num_steps = Int(round(t_final / dt))

vort_hat = copy(vort_init_hat)

println("Enstrophy convergence:")
println(sum(abs2, vort_hat - vort_steady_hat) * (1/N)^4)
for n in 1:num_steps
    step!(vort_hat, dt, params)
    if rem(n, 1000) == 0
        println(sum(abs2, vort_hat - vort_steady_hat) * (1/N)^4)
    end
end

vort = irfft(vort_hat, N)
display(heatmap(x, y, vort, xlabel="x", ylabel="y", title="vorticity", titlefont=12))

display(surface(x, y, vort - vort_steady, xlabel="x", ylabel="y", zlabel="error", title="difference `vort_final .- vort_steady`", titlefont=12))

# The difference from the actual fixed point to the limit value of the discretized method decreases with dt
## Eg. think of `̇ẋ = - a (x - xₑ)`, where `a>0` and `xₑ` is the actual equilibrium.
## Discretizing it with `x^{n+1}exp(at_{n+1}) - x^n exp(at_n) = dt a exp(at_n)` yields the fixed point
## `x̄ = (exp(-a dt) * dt * a) / (1 - exp(-a * dt))`.
## We have `x̄ - xₑ → 0` as `dt → 0`.
##
## `f(a, k) = k * a * exp(-a * k) / (1 - exp(-a * k))`

