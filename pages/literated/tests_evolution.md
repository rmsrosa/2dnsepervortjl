@def title = "Testing different approximations for the rate-of-change law"

# {{ get_title }}

Ultimately we want to evolve, in time, an equation of the form $u_t = F(u)$. For the 2D NSE, the most computationally demanding term is the bilinear one: $B(\omega) = \left(\partial_x^2 - \partial_y^2\right)(uv) + \partial_{xy}\left(v^2 - u^2\right)$, where $\mathbf{u} = (u(x, y), v(x, y))$ is the velocity field, which, in the two-dimensional periodic case, can easily be obtained from the vorticity $\omega = \boldsymbol{\nabla} \times \mathbf{u}$.

Here, we test different ways of approximating $B(u)$

Here are the packages we are gonna need.

````julia
using FFTW
using Plots
using LinearAlgebra: mul!
using Test
using BenchmarkTools
using Random

@info "Threads: $(FFTW.nthreads())"
````

````
[ Info: Threads: 8

````

## Operators

We first define a method to build the various operators acting in spectral space:

````julia
function get_operators(N, κ₀)
    # Differentiation in spectral space
    Dx_hat = im * κ₀ * [
        ifelse(k1 ≤ div(N, 2) + 1, k1 - 1, k1 - 1 - N) for k2 in 1:div(N, 2)+1, k1 in 1:N
    ]
    Dy_hat = im * κ₀ * [k2 - 1 for k2 in 1:div(N, 2)+1, k1 in 1:N]

    Delta_hat = - κ₀^2 * [
        ifelse(k1 ≤ div(N, 2) + 1, (k1 - 1)^2 + (k2 - 1)^2, (k1 - 1 - N)^2 + (k2 - 1)^2)
        for k2 in 1:div(N, 2)+1, k1 in 1:N
    ]

    # Recovering of the velocity field from the vorticity
    Hu_hat = - Dy_hat ./ Delta_hat
    Hu_hat[1, 1] = 0.0
    Hv_hat = Dx_hat ./ Delta_hat
    Hv_hat[1, 1] = 0.0

    # For the Basdevant formulation
    DxsqDysq_hat = Dx_hat.^2 .- Dy_hat.^2
    Dxy_hat = Dx_hat .* Dy_hat

    return Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat
end
````

````
get_operators (generic function with 1 method)
````

Methods to generate a scalar field from a certain list of excitable wavenumbers.

````julia
function field_from_spectrum(L, N, modes::Matrix{<:Integer}, amps::Matrix{<:Real})
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

function field_from_spectrum(rng::AbstractRNG, L, N, num_modes::Int)

    modes = rand(rng, 1:div(N,10), num_modes, 2)
    amps = rand(rng, num_modes, 2)

    field = field_from_spectrum(L, N, modes, amps)

    return field
end

field_from_spectrum(L, N, num_modes::Int) = field_from_spectrum(Xoshiro(), L, N, num_modes)
````

````
field_from_spectrum (generic function with 3 methods)
````

Evolution steps

````julia
function step_naive!(vort_hat, dt, params)
    operators, vars = params
    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat, Exp_nu_dt_Delta_hat = operators
    g_hat, N, Nsub = vars
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

function step_Basdevant!(vort_hat, dt, params)
    operators, vars = params
    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat, Exp_nu_dt_Delta_hat = operators
    g_hat, N, Nsub = vars
    u_hat = Hu_hat .* vort_hat
    v_hat = Hv_hat .* vort_hat
    u = irfft(u_hat, N)
    v = irfft(v_hat, N)
    uv_hat = rfft(u .* v)
    v2u2_hat = rfft(v.^2 - u.^2)
    vort_hat .= Exp_nu_dt_Delta_hat .* (
        vort_hat .+ dt * (
            g_hat .- DxsqDysq_hat .* uv_hat .- Dxy_hat .* v2u2_hat
        )
    )
    # dealiasing
    vort_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    vort_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im
    return vort_hat
end

function step_Basdevant_plan!(vort_hat, dt, params)
    operators, vars, auxs, plans = params

    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat, Exp_nu_dt_Delta_hat = operators
    g_hat, N, Nsub = vars
    u_hat, v_hat, u, v, uv, v2u2, uv_hat, v2u2_hat = auxs
    plan, plan_inv = plans

    u_hat .= Hu_hat .* vort_hat
    v_hat .= Hv_hat .* vort_hat

    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)

    uv .= u .* v
    v2u2 .= v.^2 .- u.^2
    mul!(uv_hat, plan, uv)
    mul!(v2u2_hat, plan, v2u2)

    @. vort_hat = Exp_nu_dt_Delta_hat * (
        vort_hat + dt * (
            g_hat - DxsqDysq_hat * uv_hat - Dxy_hat * v2u2_hat
        )
    )

#=     vort_hat .= Exp_nu_dt_Delta_hat .* (
        vort_hat .+ dt .* (
            g_hat .- DxsqDysq_hat .* uv_hat .- Dxy_hat .* v2u2_hat
        )
    ) =#
    # dealiasing
    vort_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    vort_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im
    return vort_hat
end

function step_Basdevant_plan_loop!(vort_hat, dt, params)
    operators, vars, auxs, plans = params

    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat, Exp_nu_dt_Delta_hat = operators
    g_hat, N, Nsub = vars
    u_hat, v_hat, u, v, uv, v2u2, uv_hat, v2u2_hat = auxs
    plan, plan_inv = plans

    @inbounds for i in eachindex(u_hat)
        u_hat[i] = Hu_hat[i] * vort_hat[i]
        v_hat[i] = Hv_hat[i] .* vort_hat[i]
    end

    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)

    @inbounds for i in eachindex(u)
        uv[i] = u[i] * v[i]
        v2u2[i] = v[i]^2 - u[i]^2
    end

    mul!(uv_hat, plan, uv)
    mul!(v2u2_hat, plan, v2u2)

    for i in eachindex(vort_hat)
        vort_hat[i] = Exp_nu_dt_Delta_hat[i] * (
            vort_hat[i] + dt * (
                g_hat[i] - DxsqDysq_hat[i] * uv_hat[i] - Dxy_hat[i] * v2u2_hat[i]
            )
        )
    end
    # dealiasing
    @inbounds for j in 1:N, i in div(Nsub,2) + 1:div(N, 2) + 1
        vort_hat[i, j] = 0.0im
    end
    @inbounds for j in div(Nsub,2) + 1:div(N,2) + div(Nsub,2), i in 1:div(N, 2) + 1
        vort_hat[i, j] = 0.0im
    end
    return vort_hat
end

function step_Basdevant_plan_doubleloop!(vort_hat, dt, params)
    operators, vars, auxs, plans = params

    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat, Exp_nu_dt_Delta_hat = operators
    g_hat, N, Nsub = vars
    u_hat, v_hat, u, v, uv, v2u2, uv_hat, v2u2_hat = auxs
    plan, plan_inv = plans

    M, N = size(vort_hat)

    @inbounds for j in 1:N, i in 1:M
        u_hat[i, j] = Hu_hat[i, j] * vort_hat[i, j]
        v_hat[i, j] = Hv_hat[i, j] .* vort_hat[i, j]
    end

    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)

    @inbounds for j in 1:N, i in 1:N
        uv[i, j] = u[i, j] * v[i, j]
        v2u2[i, j] = v[i, j]^2 - u[i, j]^2
    end

    mul!(uv_hat, plan, uv)
    mul!(v2u2_hat, plan, v2u2)

    for j in 1:N, i in 1:M
        vort_hat[i, j] = Exp_nu_dt_Delta_hat[i, j] * (
            vort_hat[i, j] + dt * (
                g_hat[i, j] - DxsqDysq_hat[i, j] * uv_hat[i, j] - Dxy_hat[i, j] * v2u2_hat[i, j]
            )
        )
    end
    # dealiasing
    @inbounds for j in 1:N, i in div(Nsub,2) + 1:M
        vort_hat[i, j] = 0.0im
    end
    @inbounds for j in div(Nsub,2) + 1:div(N,2) + div(Nsub,2), i in 1:M
        vort_hat[i, j] = 0.0im
    end
    return vort_hat
end
````

````
step_Basdevant_plan_doubleloop! (generic function with 1 method)
````

## The spatial domain

````julia
L = 2π
κ₀ = 2π/L
N = 128
Nsub = 84
x = y = (L/N):(L/N):L

ν = 1.0

dt = 1.0e-5 # 1.0e-3 a 1.0e-5
t_final = 1.0

num_steps = Int(round(t_final / dt))

Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = get_operators(N, κ₀)
Exp_nu_dt_Delta_hat = exp.(ν * dt * Delta_hat)

vort_init = field_from_spectrum(L, N, 4)
vort_init_hat = rfft(vort_init)

g_steady = field_from_spectrum(L, N, 1)
g_steady_hat = rfft(g_steady)

vort_steady_hat = - g_steady_hat ./ Delta_hat
vort_steady_hat[1, 1] = 0.0im
vort_steady = irfft(vort_steady_hat, N)

vort_hat = copy(vort_init_hat)
vort = irfft(vort_hat, N)

u_hat = zero(vort_hat)
v_hat = zero(vort_hat)
u = zeros(N, N)
v = zeros(N, N)
uv = zeros(N, N)
v2u2 = zeros(N, N)
uv_hat = zero(vort_hat)
v2u2_hat = zero(vort_hat)
plan = plan_rfft(vort, flags=FFTW.MEASURE)
plan_inv = plan_irfft(vort_hat, N, flags=FFTW.MEASURE)

operators =
    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat, Exp_nu_dt_Delta_hat
vars = g_steady_hat, N, Nsub
auxs = u_hat, v_hat, u, v, uv, v2u2, uv_hat, v2u2_hat
plans = plan, plan_inv

params = (
    operators,
    vars,
    auxs,
    plans
)

vort_hat = copy(vort_init_hat)
vort = irfft(vort_hat, N)

println("Enstrophy convergence:")
println(sum(abs2, vort_hat - vort_steady_hat) * (1/N)^4)
for n in 1:num_steps
    # step_naive!(vort_hat, dt, (operators, vars))
    # step_Basdevant!(vort_hat, dt, (operators, vars))
    # step_Basdevant_plan!(vort_hat, dt, (operators, vars, auxs, plans))
    step_Basdevant_plan_doubleloop!(vort_hat, dt, (operators, vars, auxs, plans))
    if rem(n, 1000) == 0
        println(sum(abs2, vort_hat - vort_steady_hat) * (1/N)^4)
    end
end

vort = irfft(vort_hat, N)
display(heatmap(x, y, vort, xlabel="x", ylabel="y", title="vorticity", titlefont=12))

display(surface(x, y, vort - vort_steady, xlabel="x", ylabel="y", zlabel="error", title="difference `vort_final .- vort_steady`", titlefont=12))

@info "step_naive!"
@btime step_naive!(vh, τ, p) setup = (vh = copy(vort_hat); τ = $dt; p = $(operators, vars));
@info "step_Basdevant!"
@btime step_Basdevant!(vh, τ, p) setup = (vh = copy(vort_hat); τ = $dt; p = $(operators, vars));
@info "step_Basdevant_plan!"
@btime step_Basdevant_plan!(vh, τ, p) setup = (vh = copy(vort_hat); τ = $dt; p = $(operators, vars, auxs, plans));
@info "step_Basdevant_plan_loop!"
@btime step_Basdevant_plan_loop!(vh, τ, p) setup = (vh = copy(vort_hat); τ = $dt; p = $(operators, vars, auxs, plans));
@info "step_Basdevant_plan_doubleloop!"
@btime step_Basdevant_plan_doubleloop!(vh, τ, p) setup = (vh = copy(vort_hat); τ = $dt; p = $(operators, vars, auxs, plans));

#=
[ Info: step_naive!
  261.500 μs (211 allocations: 2.16 MiB)
[ Info: step_Basdevant!
  222.791 μs (174 allocations: 2.03 MiB)
[ Info: step_Basdevant_plan!
  136.834 μs (0 allocations: 0 bytes)
[ Info: step_Basdevant_plan_loop!
  131.042 μs (0 allocations: 0 bytes)
[ Info: step_Basdevant_plan_doubleloop!
  137.000 μs (0 allocations: 0 bytes)
=#
````

````
Enstrophy convergence:
45769.07527692402
1682.0109178224593
92.56640651951113
20.82659804740208
15.075064949968283
13.015325120499924
11.417457454560534
10.049091037017913
8.863795234009723
7.83349679955828
6.935609024892442
6.15121794702641
5.464369373098197
4.8615680680529705
4.331373750791764
3.8640665744667873
3.4513688629483763
3.086213527040771
2.762551339102698
2.4751905168006787
2.219663124709171
1.9921137061745977
1.789206328077821
1.608046872527756
1.446117956289445
1.3012243148546252
1.1714468666951623
1.05510398640547
0.9507187737550964
0.8569913183337483
0.7727751343212599
0.6970570835697061
0.6289402231753729
0.5676291106355554
0.5124171793266524
0.46267586252840953
0.41784519811966153
0.3774256904835555
0.3409712428030012
0.3080830032039092
0.2784039932607906
0.2516144081528566
0.22742749501304316
0.20558593037230266
0.18585862957711938
0.16803793106735843
0.151937106786084
0.13738815703298188
0.12423985399852298
0.11235600321576472
0.10161389639614761
0.09190293270268658
0.08312338856428193
0.07518531873557793
0.0680075735302151
0.06151691906120483
0.05564724896030896
0.0503388774600379
0.045537904941747746
0.04119564811009055
0.03726812787191372
0.03371560879685993
0.030502184734415173
0.027595405772347932
0.02496594225649205
0.022587282061966404
0.020435457719869144
0.01848880036870609
0.01672771782262538
0.015134494334380227
0.013693109884467721
0.012389077053110827
0.011209293732117591
0.010141910112146381
0.00917620854012427
0.008302494983759819
0.007512000967216404
0.006796794955783223
0.006149702269301356
0.005564232695475939
0.005034515056201537
0.004555238053649847
0.0041215967890302465
0.0037292444064149256
0.0033742483675359546
0.0030530509116280676
0.002762433297768564
0.0024994834662438394
0.002261566790692854
0.0020462996245326982
0.001851525373811395
0.0016752928544706788
0.001515836715317604
0.0013715597290502814
0.001241016772683997
0.0011229003358828708
0.0010160274111992002
0.0009193276342224482
0.0008318325542881731
0.0007526659278260432
0.0006810349367549992
[ Info: step_naive!
  445.500 μs (195 allocations: 2.16 MiB)
[ Info: step_Basdevant!
  368.208 μs (162 allocations: 2.03 MiB)
[ Info: step_Basdevant_plan!
  176.458 μs (0 allocations: 0 bytes)
[ Info: step_Basdevant_plan_loop!
  170.291 μs (0 allocations: 0 bytes)
[ Info: step_Basdevant_plan_doubleloop!
  169.666 μs (0 allocations: 0 bytes)

````

