# Fixed-step schemes for the stochastic equation, written for this problem (diagonal viscous
# operator, additive noise on a few modes), used through `simulate`.

"""
    AbstractFixedStepper

Fixed-step schemes of this package, used with [`simulate`](@ref): [`StochasticExpEuler`](@ref)
and [`SplitStochasticRK`](@ref). They work with or without a stochastic forcing in the model.
"""
abstract type AbstractFixedStepper end

"""
    StochasticExpEuler()

Stochastic exponential Euler scheme (exponential Euler-Maruyama). With `L = νΔ` (diagonal),
`N` the rest of the drift and `G = σΦ`,

    ωₙ₊₁ = e^{LΔt} ωₙ + Δt φ₁(LΔt) N(ωₙ, tₙ) + ∫ e^{L(tₙ₊₁ - s)} G dWₛ,

with `φ₁(z) = (eᶻ - 1)/z`, where the stochastic convolution is sampled exactly, with variance
`(e^{2LΔt} - 1)/(2L)` (exact when each noise mode lives on a single shell `|κ| = const`, as the
Fourier modes of [`fourier_modes`](@ref) do). The viscous term and the linear part of the noise
are integrated exactly, so the step is limited only by the nonlinear term. Order one, strong
and weak. One evaluation of the nonlinear term per step.
"""
struct StochasticExpEuler <: AbstractFixedStepper end

"""
    StochasticETDRK4()

Exponential time differencing with the fourth-order Runge-Kutta scheme of Cox and Matthews for
the drift (`L = νΔ` integrated exactly, the rest with ETDRK4, with the coefficients computed as
in Kassam and Trefethen), plus the stochastic convolution `∫ e^{L(tₙ₊₁ - s)} G dWₛ`, sampled
exactly as in [`StochasticExpEuler`](@ref). Exact for the linear part, like the exponential
Euler scheme, but, unlike it, stable for advection up to the CFL limit of RK4, since ETDRK4
reduces to the classical Runge-Kutta method as `ν → 0`. Four evaluations of the nonlinear term
per step. Without noise, it is the deterministic ETDRK4.

Note: the explicit treatment of the nonlinear term in [`StochasticExpEuler`](@ref) (forward
Euler) is unstable for advection unless viscous damping compensates, so, for
advection-dominated flows, it needs much smaller steps than this scheme.
"""
struct StochasticETDRK4 <: AbstractFixedStepper end

"""
    SplitStochasticRK(scheme = :rk4)

Splitting of a deterministic Runge-Kutta step and the noise increment, in symmetric (Strang)
form: half of the Wiener increment, a step of the deterministic equation with the classical
fourth-order Runge-Kutta method (`scheme = :rk4`) or the three-stage strong-stability-preserving
method (`scheme = :ssprk3`), and the other half of the increment. Without noise, it is the
deterministic method. A common choice in practice; its weak order, for the statistics, is
limited by the splitting.
"""
struct SplitStochasticRK <: AbstractFixedStepper
    scheme::Symbol
    function SplitStochasticRK(scheme::Symbol = :rk4)
        scheme in (:rk4, :ssprk3) || throw(ArgumentError("scheme must be :rk4 or :ssprk3"))
        return new(scheme)
    end
end

# Drift on the complex vectorized state.
_drift!(dw, w, m, t) = rhs_vec!(dw, w, m, t)

struct _ExpEulerCache
    E::Vector{Float64}       # e^{LΔt}
    Φ1::Vector{Float64}      # Δt φ₁(LΔt)
    S::Vector{Float64}       # standard deviation factor of the stochastic convolution
    N::Vector{ComplexF64}
    nz::Vector{ComplexF64}
    ξ::Vector{Float64}
end

function _cache(::StochasticExpEuler, m::NSEModel, dt)
    L = real.(m.L)
    E = exp.(L .* dt)
    Φ1 = [l == 0 ? dt : (e - 1) / l for (l, e) in zip(L, E)]
    S = [l == 0 ? sqrt(dt) : sqrt((e^2 - 1) / (2l)) for (l, e) in zip(L, E)]
    n = m.noise === nothing ? 0 : length(m.noise)
    return _ExpEulerCache(E, Φ1, S, zeros(ComplexF64, length(L)), zeros(ComplexF64, length(L)), zeros(n))
end

function _step!(w, c::_ExpEulerCache, ::StochasticExpEuler, m::NSEModel, t, dt, rng)
    nonlinear_vec!(c.N, w, m, t)
    @. w = c.E * w + c.Φ1 * c.N
    if m.noise !== nothing
        randn!(rng, c.ξ)
        mul!(c.nz, m.noise.G, c.ξ)
        @. w += c.S * c.nz
    end
    return w
end

struct _RKCache
    k1::Vector{ComplexF64}; k2::Vector{ComplexF64}; k3::Vector{ComplexF64}; k4::Vector{ComplexF64}
    tmp::Vector{ComplexF64}; nz::Vector{ComplexF64}; ξ::Vector{Float64}
end

function _cache(::SplitStochasticRK, m::NSEModel, dt)
    n = length(m.L); z() = zeros(ComplexF64, n)
    return _RKCache(z(), z(), z(), z(), z(), z(), zeros(m.noise === nothing ? 0 : length(m.noise)))
end

function _half_noise!(w, c::_RKCache, m, dt, rng)
    m.noise === nothing && return w
    randn!(rng, c.ξ)
    mul!(c.nz, m.noise.G, c.ξ)
    @. w += sqrt(dt / 2) * c.nz
    return w
end

function _step!(w, c::_RKCache, s::SplitStochasticRK, m::NSEModel, t, dt, rng)
    _half_noise!(w, c, m, dt, rng)
    if s.scheme === :rk4
        _drift!(c.k1, w, m, t)
        @. c.tmp = w + dt / 2 * c.k1; _drift!(c.k2, c.tmp, m, t + dt / 2)
        @. c.tmp = w + dt / 2 * c.k2; _drift!(c.k3, c.tmp, m, t + dt / 2)
        @. c.tmp = w + dt * c.k3;     _drift!(c.k4, c.tmp, m, t + dt)
        @. w += dt / 6 * (c.k1 + 2 * c.k2 + 2 * c.k3 + c.k4)
    else  # SSPRK3 (Shu-Osher)
        _drift!(c.k1, w, m, t)
        @. c.tmp = w + dt * c.k1
        _drift!(c.k2, c.tmp, m, t + dt)
        @. c.tmp = 3 / 4 * w + 1 / 4 * (c.tmp + dt * c.k2)
        _drift!(c.k3, c.tmp, m, t + dt / 2)
        @. w = 1 / 3 * w + 2 / 3 * (c.tmp + dt * c.k3)
    end
    _half_noise!(w, c, m, dt, rng)
    return w
end

struct _ETDRK4Cache
    E::Vector{Float64}; E2::Vector{Float64}
    Q::Vector{Float64}; f1::Vector{Float64}; f2::Vector{Float64}; f3::Vector{Float64}
    S::Vector{Float64}
    Nu::Vector{ComplexF64}; Na::Vector{ComplexF64}; Nb::Vector{ComplexF64}; Nc::Vector{ComplexF64}
    a::Vector{ComplexF64}; b::Vector{ComplexF64}; c::Vector{ComplexF64}
    nz::Vector{ComplexF64}; ξ::Vector{Float64}
end

# ETDRK4 coefficients by contour integrals (Kassam & Trefethen, 2005), for real L ≤ 0.
function _cache(::StochasticETDRK4, m::NSEModel, dt)
    L = real.(m.L)
    M = 32
    r = [exp(im * π * (j - 0.5) / M) for j in 1:M]
    n = length(L)
    E = exp.(dt .* L); E2 = exp.(dt .* L ./ 2)
    Q = zeros(n); f1 = zeros(n); f2 = zeros(n); f3 = zeros(n)
    for i in 1:n
        z = dt * L[i] .+ r
        Q[i] = dt * real(sum((exp.(z ./ 2) .- 1) ./ z) / M)
        f1[i] = dt * real(sum((-4 .- z .+ exp.(z) .* (4 .- 3z .+ z .^ 2)) ./ z .^ 3) / M)
        f2[i] = dt * real(sum((2 .+ z .+ exp.(z) .* (z .- 2)) ./ z .^ 3) / M)
        f3[i] = dt * real(sum((-4 .- 3z .- z .^ 2 .+ exp.(z) .* (4 .- z)) ./ z .^ 3) / M)
    end
    S = [l == 0 ? sqrt(dt) : sqrt((e^2 - 1) / (2l)) for (l, e) in zip(L, E)]
    z() = zeros(ComplexF64, n)
    return _ETDRK4Cache(E, E2, Q, f1, f2, f3, S, z(), z(), z(), z(), z(), z(), z(), z(),
        zeros(m.noise === nothing ? 0 : length(m.noise)))
end

function _step!(w, c::_ETDRK4Cache, ::StochasticETDRK4, m::NSEModel, t, dt, rng)
    nonlinear_vec!(c.Nu, w, m, t)
    @. c.a = c.E2 * w + c.Q * c.Nu
    nonlinear_vec!(c.Na, c.a, m, t + dt / 2)
    @. c.b = c.E2 * w + c.Q * c.Na
    nonlinear_vec!(c.Nb, c.b, m, t + dt / 2)
    @. c.c = c.E2 * c.a + c.Q * (2 * c.Nb - c.Nu)
    nonlinear_vec!(c.Nc, c.c, m, t + dt)
    @. w = c.E * w + c.f1 * c.Nu + 2 * c.f2 * (c.Na + c.Nb) + c.f3 * c.Nc
    if m.noise !== nothing
        randn!(rng, c.ξ)
        mul!(c.nz, m.noise.G, c.ξ)
        @. w += c.S * c.nz
    end
    return w
end

_evals_per_step(::StochasticETDRK4) = 4
_evals_per_step(::StochasticExpEuler) = 1
_evals_per_step(s::SplitStochasticRK) = s.scheme === :rk4 ? 4 : 3
