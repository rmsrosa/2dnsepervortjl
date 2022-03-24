# # Approximations of the bilinear term

# Ultimately we want to evolve, in time, an equation of the form # $$
#   \omega_t = G(\omega).
# $$
# For the 2D NSE, $G$ is of the form
# $$
#  G(\omega) = g + \nu \Delta \omega + B(\omega),
# $$
# where
# $$
#   B(\omega) = \left(\partial_x^2 - \partial_y^2\right)(uv) + \partial_{xy}\left(v^2 - u^2\right),
# $$
# with $\mathbf{u} = (u(x, y), v(x, y))$ being the velocity field associated with the vorticity, i.e.  $\omega = \boldsymbol{\nabla} \times \mathbf{u} = v_x - u_y$.

# In spectral space, $\Delta \omega$ is just a diagonal operator, while $g$ is a given scalar field, which are both easy to compute. The most computationally demanding term is $B(\omega)$.

# Here, we test different ways of approximating $B(\omega)$.

# The packages we are gonna need.

using FFTW
using Plots
using LinearAlgebra: mul!
using Test
using BenchmarkTools
using Random

@info "Threads: $(FFTW.nthreads())"

# ## Operators

# We first define a method to build the various operators acting in spectral space:

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

# ## Generation of a scalar field

# Method to generate a scalar field from a list of wavenumbers and amplitudes:

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

# Method to randomly generate a scalar field from a given random number generator and a given number wavenumbers to be excited:

function field_from_modes(rng::AbstractRNG, L, N, num_modes::Int)

    modes = rand(rng, 1:div(N,10), num_modes, 2)
    amps = rand(rng, num_modes, 2)

    field = field_from_modes(L, N, modes, amps)

    return field
end

# Method to randomly generate a scalar field from a given number wavenumbers to be excited:

field_from_modes(L, N, num_modes::Int) = field_from_modes(Xoshiro(), L, N, num_modes)

# ## Methods for computing the bilinear term

function bilinear_naive(vort_hat, params)
    operators, vars = params

    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    N, Nsub = vars

    u_hat = Hu_hat .* vort_hat
    v_hat = Hv_hat .* vort_hat
    u = irfft(u_hat, N)
    v = irfft(v_hat, N)
    vort = irfft(vort_hat, N)
    wu_hat = rfft(vort .* u)
    wv_hat = rfft(vort .* v)

    bilin_hat = Dx_hat .* wu_hat .+ Dy_hat .* wv_hat
    ## dealiasing
    bilin_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    bilin_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im
    return bilin_hat
end

#

function bilinear_auxs!(bilin_hat, vort_hat, params)
    operators, vars, auxs = params

    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    N, Nsub = vars
    u_hat, v_hat, u, v, w, wu, wv, wu_hat, wv_hat = auxs

    u_hat .= Hu_hat .* vort_hat
    v_hat .= Hv_hat .* vort_hat
    u .= irfft(u_hat, N)
    v .= irfft(v_hat, N)
    w .= irfft(vort_hat, N)
    wu .= w .* u
    wv .= w .* v
    wu_hat .= rfft(wu)
    wv_hat .= rfft(wv)

    bilin_hat .= Dx_hat .* wu_hat .+ Dy_hat .* wv_hat
    ## dealiasing
    bilin_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    bilin_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im

    return bilin_hat
end

#

function bilinear_plan_brdcst!(bilin_hat, vort_hat, params)
    operators, vars, auxs, plans = params

    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    N, Nsub = vars
    u_hat, v_hat, u, v, w, wu, wv, wu_hat, wv_hat = auxs
    plan, plan_inv = plans

    u_hat .= Hu_hat .* vort_hat
    v_hat .= Hv_hat .* vort_hat
    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)
    ## mul! on inverse plan mutates last argument, but we need to preserve vort_hat
    ## so, reuse v_hat for that:
    v_hat .= vort_hat
    mul!(w, plan_inv, v_hat) 
    wu .= w .* u
    wv .= w .* v
    
    mul!(wu_hat, plan, wu)
    mul!(wv_hat, plan, wv)
 
    bilin_hat .= Dx_hat .* wu_hat .+ Dy_hat .* wv_hat
    ## dealiasing
    bilin_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    bilin_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im
 
    return bilin_hat
end

#

function bilinear_Basdevant_brdcst(vort_hat, params)
    operators, vars = params

    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    N, Nsub = vars

    u_hat = Hu_hat .* vort_hat
    v_hat = Hv_hat .* vort_hat
    u = irfft(u_hat, N)
    v = irfft(v_hat, N)
    uv_hat = rfft(u .* v)
    v2u2_hat = rfft(v.^2 - u.^2)

    bilin_hat = DxsqDysq_hat .* uv_hat .+ Dxy_hat .* v2u2_hat
    ## dealiasing
    bilin_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    bilin_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im

    return bilin_hat
end

#

function bilinear_Basdevant_plan_brdcst!(bilin_hat, vort_hat, params)
    operators, vars, auxs, plans = params
    
    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    N, Nsub = vars
    u_hat, v_hat, u, v, w, uv, v2u2, uv_hat, v2u2_hat = auxs
    plan, plan_inv = plans

    u_hat .= Hu_hat .* vort_hat
    v_hat .= Hv_hat .* vort_hat

    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)

    uv .= u .* v
    v2u2 .= v.^2 .- u.^2
    mul!(uv_hat, plan, uv)
    mul!(v2u2_hat, plan, v2u2)

    bilin_hat .= DxsqDysq_hat .* uv_hat .+ Dxy_hat .* v2u2_hat

    ## dealiasing
    bilin_hat[div(Nsub,2) + 1:end, :] .= 0.0im
    bilin_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0.0im

    return bilin_hat
end

#

function bilinear_Basdevant_plan_loop!(bilin_hat, vort_hat, params)
    operators, vars, auxs, plans = params
    
    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    N, Nsub = vars
    u_hat, v_hat, u, v, w, uv, v2u2, uv_hat, v2u2_hat = auxs
    plan, plan_inv = plans

    @inbounds for k in eachindex(u_hat)
        u_hat[k] = Hu_hat[k] * vort_hat[k]
        v_hat[k] = Hv_hat[k] * vort_hat[k]
    end

    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)

    @inbounds for k in eachindex(u)
        uv[k] = u[k] * v[k]
        v2u2[k] = v[k]^2 - u[k]^2
    end

    mul!(uv_hat, plan, uv)
    mul!(v2u2_hat, plan, v2u2)

    @inbounds for k in eachindex(vort_hat)
        bilin_hat[k] = DxsqDysq_hat[k] * uv_hat[k] + Dxy_hat[k] * v2u2_hat[k]
    end
    ## dealiasing
    @inbounds for j in 1:N, i in div(Nsub,2) + 1:div(N, 2) + 1
        bilin_hat[i, j] = 0.0im
    end
    @inbounds for j in div(Nsub,2) + 1:div(N,2) + div(Nsub,2), i in 1:div(N, 2) + 1
        bilin_hat[i, j] = 0.0im
    end

    return bilin_hat
end

#

function bilinear_Basdevant_plan_doubleloop!(bilin_hat, vort_hat, params)
    operators, vars, auxs, plans = params
    
    Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat,
        DxsqDysq_hat, Dxy_hat = operators
    N, Nsub = vars
    u_hat, v_hat, u, v, w, uv, v2u2, uv_hat, v2u2_hat = auxs
    plan, plan_inv = plans

    M, N = size(vort_hat)

    @inbounds for j in 1:N, i in 1:M
        u_hat[i, j] = Hu_hat[i, j] * vort_hat[i, j]
        v_hat[i, j] = Hv_hat[i, j] * vort_hat[i, j]
    end

    mul!(u, plan_inv, u_hat)
    mul!(v, plan_inv, v_hat)

    @inbounds for j in 1:N, i in 1:N
        uv[i, j] = u[i, j] * v[i, j]
        v2u2[i, j] = v[i, j]^2 - u[i, j]^2
    end

    mul!(uv_hat, plan, uv)
    mul!(v2u2_hat, plan, v2u2)

    @inbounds for j in 1:N, i in 1:M
        bilin_hat[i, j] = DxsqDysq_hat[i, j] * uv_hat[i, j] + Dxy_hat[i, j] * v2u2_hat[i, j]
    end
    ## dealiasing
    @inbounds for j in 1:N, i in div(Nsub,2) + 1:M
        bilin_hat[i, j] = 0.0im
    end
    @inbounds for j in div(Nsub,2) + 1:div(N,2) + div(Nsub,2), i in 1:M
        bilin_hat[i, j] = 0.0im
    end

    return bilin_hat
end

# ## The spatial domain and its discretization

L = 2π
κ₀ = 2π/L
N = 256
Nsub = 162
x = y = (L/N):(L/N):L

# ## Preparation for tests

Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = get_operators(N, κ₀)

vort = field_from_modes(L, N, 48)
vort_hat = rfft(vort)

u_hat = similar(vort_hat)
v_hat = similar(vort_hat)
u = similar(vort)
v = similar(vort)
w = similar(vort)
uv = similar(vort)
v2u2 = similar(vort)
uv_hat = similar(vort_hat)
v2u2_hat = similar(vort_hat)
bilin_hat = similar(vort_hat)

plan = plan_rfft(vort, flags=FFTW.MEASURE)
plan_inv = plan_irfft(vort_hat, N, flags=FFTW.MEASURE)

operators = Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat
vars = N, Nsub
auxs = u_hat, v_hat, u, v, w, uv, v2u2, uv_hat, v2u2_hat
plans = plan, plan_inv

# ## Check results

# As a first test, we check that all implementations return about the same vector

@testset "same result bilinear terms" begin

    bilin_hat_naive = bilinear_naive(
        vort_hat,
        (operators, vars)
    )

    bilinear_hat_brdcst = similar(vort_hat)
    bilinear_auxs!(bilinear_hat_brdcst, vort_hat, (operators, vars, auxs))

    bilin_hat_plan_brdcst = similar(vort_hat)
    bilinear_plan_brdcst!(
        bilin_hat_plan_brdcst,
        vort_hat,
        (operators, vars, auxs, plans)
    )

    bilin_hat_Basdevant_brdcst = bilinear_Basdevant_brdcst(
        vort_hat,
        (operators, vars)
    )

    bilin_hat_Basdevant_plan_brdcst = similar(vort_hat)
    bilinear_Basdevant_plan_brdcst!(
        bilin_hat_Basdevant_plan_brdcst,
        vort_hat,
        (operators, vars, auxs, plans)
    )

    bilin_hat_Basdevant_plan_loop = similar(vort_hat)
    bilinear_Basdevant_plan_loop!(
        bilin_hat_Basdevant_plan_loop,
        vort_hat,
        (operators, vars, auxs, plans)
    )

    bilin_hat_Basdevant_plan_doubleloop = similar(vort_hat)
    bilinear_Basdevant_plan_doubleloop!(
        bilin_hat_Basdevant_plan_doubleloop,
        vort_hat,
        (operators, vars, auxs, plans)
    )

    @test bilin_hat_naive ≈ bilinear_hat_brdcst
    @test bilinear_hat_brdcst ≈ bilin_hat_plan_brdcst
    @test bilin_hat_plan_brdcst ≈ bilin_hat_Basdevant_brdcst
    @test bilin_hat_Basdevant_brdcst ≈ bilin_hat_Basdevant_plan_brdcst
    @test bilin_hat_Basdevant_plan_brdcst ≈ bilin_hat_Basdevant_plan_loop
    @test bilin_hat_Basdevant_plan_loop ≈ bilin_hat_Basdevant_plan_doubleloop
end
nothing

# ## Orthogonality

# As a second test, we check the orthogonality condition of the bilinear term.

# Due to the periodicity and the divergence-free conditions, we should have $(B(\omega), \omega)_{L^2} = 0$. We compute $B(\omega)$ in spectral space and transform it to physical space in order to check that.

bilin_hat_naive = bilinear_naive(vort_hat, (operators, vars))
bilin_naive = irfft(bilin_hat_naive, N)
nothing

# We compute the inner product by summing over the nodal values and multiplying that by the area element $\Delta x \Delta y = (L/N)^2$. Notice computing the inner product as such has an inherent numerical error, which is also of the order of the area element, so we don't expect the output to be of machine precision.

sum(bilin_naive .* vort) * (L / N)^2

# In spectral space, due to Parseval's identity, this translates to the following (except we need to correct for some indices...)

M = iseven(N) ? div(N, 2) : div(N, 2) + 1

(
    2 * sum(bilin_hat_naive[2:M,:] .* conj.(vort_hat[2:M,:])) 
    + sum(bilin_hat_naive[1,:] .* conj.(vort_hat[1,:]))
    + sum(bilin_hat_naive[M+1:end,:] .* conj.(vort_hat[M+1:end,:]))
) / N^2 * (L / N)^2

# Hmm, something is wrong...

# ## Benchmark

@info "bilinear_naive"
@btime bilinear_naive(vh, p) setup = (vh = copy($vort_hat); p = $(operators, vars));

@info "bilinear_auxs!"
@btime bilinear_auxs!(bh, vh, p) setup = (bh = similar($vort_hat); vh = copy($vort_hat); p = $(operators, vars, auxs, plans));

@info "bilinear_Basdevant_brdcst"
@btime bilinear_Basdevant_brdcst(vh, p) setup = (vh = copy($vort_hat); p = $(operators, vars));

@info "bilinear_plan_brdcst!"
@btime bilinear_plan_brdcst!(bh, vh, p) setup = (bh = similar($vort_hat); vh = copy($vort_hat); p = $(operators, vars, auxs, plans));

@info "bilin_hat_Basdevant_plan_brdcst!"
@btime bilinear_Basdevant_plan_brdcst!(bh, vh, p) setup = (bh = similar($vort_hat); vh = copy($vort_hat); p = $(operators, vars, auxs, plans));

@info "bilinear_Basdevant_plan_loop!"
@btime bilinear_Basdevant_plan_loop!(bh, vh, p) setup = (bh = similar($vort_hat); vh = copy($vort_hat); p = $(operators, vars, auxs, plans));

@info "bilinear_Basdevant_plan_doubleloop!"
@btime bilinear_Basdevant_plan_doubleloop!(bh, vh, p) setup = (bh = similar($vort_hat); vh = copy($vort_hat); p = $(operators, vars, auxs, plans));

# ## Conclusions

# Clearly, `bilinear_naive` is pretty bad, in great part because of the use of the allocating transforms `rfft` and `irfft`.

# The method `bilinear_auxs!` saves some allocations but is pretty slow as well, due to also using `rfft` and `irfft`.

# The method `bilinear_Basdevant_brdcst` saves some time by using the Basdevant formulation, but is still quite slow because of `rfft` and `irfft`.

# All the others are much faster, due to using the plans for the transforms and auxiliary functions to avoid allocation of temporary variable.

# Of the nonallocating ones, the ones that use the Basdevant formulation have a clear advantage. Indeed, computing the direct and inverse Fourier transform are the most costly operation here, so reducing it from five to four makes a significant difference.

# Of these nonallocating ones that use the Basdevant formulation, the two that use explicit loops perform better than the broadcast one, with a slight advantage for the one that uses a single loop with `eachindex` instead of a double loop.

# If one does [@code_warntype](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#man-code-warntype-1) on them, one sees that most are clean, but there are some yellow warnings with type `Union{Nothing, Tuple{Int64, Int64}}` precisely on the two fastest ones. Yellow warnings, fortunately, tend to be harmless, as is the case here. In this case, the warning is due to the way the [iterator interface](https://docs.julialang.org/en/v1/manual/interfaces/#Interfaces) is implemented, which returns `nothing` when the end of the iteration is reached. One can get rid of it with `while` loops, but it won't improve the performance.
