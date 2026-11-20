# Stochastic (white-in-time) and random (arbitrary process) forcing terms.
#
# In velocity form, the equation is
#     du + (u⋅∇u + ∇p) dt = νΔu dt + Nₜ dt + σ Φ dWₜ,   ∇⋅u = 0,
# where Φ : ℝⁿ → H maps the n components of the Wiener process Wₜ onto divergence-free
# velocity fields, and Nₜ is an arbitrary stochastic process with values in H. Taking the curl,
# the vorticity equation gets the terms curl(Nₜ) dt + σ curl(Φ dWₜ). Both are expressed on a
# finite set of forcing modes, given by their vorticities.

"""
    ForcingModes

A finite set of forcing modes, given by the spectral vorticities (curls) of the velocity
fields `φ₁, …, φₙ`, stored as the columns of `Q`, of size `(Mx * Ny, n)`, with a label for
each mode and the mean squares `⟨|φⱼ|²⟩` of the velocity fields and `⟨(curl φⱼ)²⟩` of their
vorticities.

Build it with [`fourier_modes`](@ref), or from arbitrary vorticity fields with
`ForcingModes(grid, fields)`.
"""
struct ForcingModes
    Q::Matrix{ComplexF64}
    labels::Vector{String}
    velocity_ms::Vector{Float64}
    vorticity_ms::Vector{Float64}
end

Base.length(f::ForcingModes) = size(f.Q, 2)
Base.show(io::IO, f::ForcingModes) = print(io, "ForcingModes($(length(f)) modes)")

"""
    ForcingModes(grid, fields; labels)

Forcing modes from a vector of vorticity fields (each in any form accepted by
[`vorticity_hat`](@ref)), the curls of the velocity modes `φⱼ`.
"""
function ForcingModes(grid::Grid, fields::AbstractVector; labels = ["mode $j" for j in eachindex(fields)])
    cols = [vec(vorticity_hat(grid, f)) for f in fields]
    any(c -> abs(c[1]) > 1e-12 * max(1.0, maximum(abs, c)), cols) &&
        @warn "some forcing modes have a nonzero mean vorticity, which a periodic velocity field cannot have; it is ignored"
    for c in cols
        c[1] = 0
        m = maximum(abs, c)
        c[abs.(c) .< 1e-12 * m] .= 0     # drop the round-off of the transforms, keeping Q sparse
    end
    Q = reduce(hcat, cols)
    W = vec(spectral_weights(grid)); k2 = vec(grid.κ2)
    vel = [sum(W[i] * abs2(Q[i, j]) / k2[i] for i in axes(Q, 1) if k2[i] > 0) for j in axes(Q, 2)]
    vor = [sum(W[i] * abs2(Q[i, j]) for i in axes(Q, 1)) for j in axes(Q, 2)]
    return ForcingModes(Q, collect(String, labels), vel, vor)
end

"""
    fourier_modes(grid, κmin, κmax)
    fourier_modes(grid, wavenumbers)

Divergence-free Fourier velocity modes `φ = √2 ∇⊥(cos(κ⋅x)/|κ|)` and `√2 ∇⊥(sin(κ⋅x)/|κ|)`, two
for each wavevector `κ` (one of each pair `±κ`), either with `κmin ≤ |κ| ≤ κmax` or given by
the integer wavenumber indices `(jx, jy)` in `wavenumbers`. They are normalized to unit mean
square, `⟨|φ|²⟩ = 1`, and their vorticities are `√2 |κ| cos(κ⋅x)` and `√2 |κ| sin(κ⋅x)`.
"""
function fourier_modes(grid::Grid, κmin::Real, κmax::Real)
    ws = Tuple{Int, Int}[]
    for j in 1:grid.Ny, i in 1:grid.Mx
        κ = sqrt(grid.κ2[i, j])
        (grid.mask[i, j] && κ > 0 && κmin ≤ κ ≤ κmax) || continue
        jx = i - 1
        jy = j - 1 < (grid.Ny + 1) ÷ 2 ? j - 1 : j - 1 - grid.Ny
        jx == 0 && jy < 0 && continue
        push!(ws, (jx, jy))
    end
    isempty(ws) && throw(ArgumentError("no retained modes in the band $κmin ≤ |κ| ≤ $κmax"))
    return fourier_modes(grid, ws)
end

function fourier_modes(grid::Grid, wavenumbers::AbstractVector{<:Tuple{Integer, Integer}})
    fields = Function[]; labels = String[]
    for (jx, jy) in wavenumbers
        kx = 2π * jx / grid.Lx; ky = 2π * jy / grid.Ly
        κ = sqrt(kx^2 + ky^2)
        κ > 0 || throw(ArgumentError("the wavenumber (0, 0) cannot be forced"))
        push!(fields, (x, y) -> √2 * κ * cos(kx * x + ky * y)); push!(labels, "cos($jx, $jy)")
        push!(fields, (x, y) -> √2 * κ * sin(kx * x + ky * y)); push!(labels, "sin($jx, $jy)")
    end
    return ForcingModes(grid, fields; labels)
end

"""
    AdditiveNoise(modes; σ = 1.0, amplitudes = ones(length(modes)))

The stochastic forcing `σ Φ dWₜ`, with `Φ eⱼ = amplitudes[j] φⱼ` for the forcing modes `φⱼ` in
`modes` (a [`ForcingModes`](@ref)), driven by a Wiener process with `length(modes)`
independent components. Its mean rate of injection of energy is
`ε = ½ σ² Σⱼ amplitudes[j]² ⟨|φⱼ|²⟩`, and of enstrophy, `η = ½ σ² Σⱼ amplitudes[j]² ⟨(curl φⱼ)²⟩`,
independently of the flow (Itô).
"""
struct AdditiveNoise
    modes::ForcingModes
    σ::Float64
    amplitudes::Vector{Float64}
    G::Matrix{ComplexF64}          # σ Φ in spectral vorticity, columns of size Mx * Ny
end

function AdditiveNoise(modes::ForcingModes; σ::Real = 1.0, amplitudes = ones(length(modes)))
    length(amplitudes) == length(modes) || throw(DimensionMismatch("one amplitude per mode is needed"))
    G = modes.Q .* (σ .* transpose(collect(Float64, amplitudes)))
    return AdditiveNoise(modes, Float64(σ), collect(Float64, amplitudes), G)
end

Base.length(a::AdditiveNoise) = length(a.modes)
Base.show(io::IO, a::AdditiveNoise) = print(io, "AdditiveNoise(σ = $(a.σ), $(length(a)) modes, ε = ",
    round(energy_injection_rate(a), sigdigits = 4), ")")

"Mean rate of injection of energy `½⟨|u|²⟩` by the additive noise."
energy_injection_rate(a::AdditiveNoise) = a.σ^2 / 2 * sum(a.amplitudes .^ 2 .* a.modes.velocity_ms)
"Mean rate of injection of enstrophy `½⟨ω²⟩` by the additive noise."
enstrophy_injection_rate(a::AdditiveNoise) = a.σ^2 / 2 * sum(a.amplitudes .^ 2 .* a.modes.vorticity_ms)

"""
    RandomForcing(modes, η)

The random forcing `Nₜ = Σⱼ ηⱼ(t) φⱼ` on the forcing modes `φⱼ` of `modes`, where `η(t)` returns a
vector with `length(modes)` components: a sample path of any process, e.g. from
[`ou_path`](@ref), [`compound_poisson_path`](@ref) or [`lorenz_path`](@ref), or any function of
`t`. For paths with jumps, pass `tstops = jump_times(path)` to adaptive solvers.
"""
struct RandomForcing{F}
    modes::ForcingModes
    η::F
    buf::Vector{ComplexF64}
end

RandomForcing(modes::ForcingModes, η) = RandomForcing(modes, η, zeros(ComplexF64, size(modes.Q, 1)))
Base.show(io::IO, r::RandomForcing) = print(io, "RandomForcing($(length(r.modes)) modes, ", typeof(r.η).name.name, ")")

# Adds Σⱼ ηⱼ(t) curl φⱼ to the spectral vorticity vector `dw` (of length Mx * Ny).
function add_random_forcing!(dw::AbstractVector, r::RandomForcing, t)
    mul!(r.buf, r.modes.Q, r.η(t))
    dw .+= r.buf
    return dw
end

# ---------------------------------------------------------------------------
# Sample paths

"""
    SamplePath

A sample path of a vector-valued process, callable as `path(t)`, which returns the value at
time `t` (a vector). Built by [`ou_path`](@ref), [`compound_poisson_path`](@ref) and
[`lorenz_path`](@ref).
"""
abstract type SamplePath end

"""
    ou_path(n, tspan; θ, s, dt, rng = Xoshiro(), x0 = nothing)

A sample path of `n` independent Ornstein-Uhlenbeck processes `dXₜ = -θ Xₜ dt + s √(2θ) dBₜ`,
with stationary standard deviation `s` and correlation time `1/θ`, sampled exactly at the
times `tspan[1]:dt:tspan[2]` and interpolated linearly in between. It starts from the
stationary distribution unless `x0` is given.
"""
struct OUPath <: SamplePath
    t0::Float64
    dt::Float64
    X::Matrix{Float64}          # (n, ntimes)
    out::Vector{Float64}
end

function ou_path(n::Integer, tspan; θ::Real, s::Real, dt::Real, rng::AbstractRNG = Xoshiro(), x0 = nothing)
    t0, tf = Float64.(tspan)
    m = ceil(Int, (tf - t0) / dt) + 1
    X = zeros(n, m)
    X[:, 1] = x0 === nothing ? s .* randn(rng, n) : x0
    a = exp(-θ * dt); b = s * sqrt(1 - a^2)
    for k in 2:m
        X[:, k] .= a .* X[:, k-1] .+ b .* randn(rng, n)
    end
    return OUPath(t0, Float64(dt), X, zeros(n))
end

function (p::OUPath)(t)
    τ = (t - p.t0) / p.dt
    k = clamp(floor(Int, τ), 0, size(p.X, 2) - 2)
    λ = clamp(τ - k, 0.0, 1.0)
    @views p.out .= (1 - λ) .* p.X[:, k+1] .+ λ .* p.X[:, k+2]
    return p.out
end

"""
    compound_poisson_path(n, tspan; rate, jump = rng -> randn(rng), θ = 0.0, rng = Xoshiro())

A sample path of `n` independent compound Poisson processes, each with jumps at the times of
a Poisson process with the given `rate` and jump sizes drawn by `jump(rng)`, and, if `θ > 0`,
exponential decay between jumps (`dXₜ = -θ Xₜ dt + dJₜ`, a Lévy-driven Ornstein-Uhlenbeck
process, or shot noise, which is stationary). Exact at all times. The jump times, all
components together, are given by `jump_times(path)`.
"""
struct CompoundPoissonPath <: SamplePath
    θ::Float64
    times::Vector{Vector{Float64}}   # jump times of each component (with t0 first)
    values::Vector{Vector{Float64}}  # value right after each jump
    out::Vector{Float64}
end

function compound_poisson_path(n::Integer, tspan; rate::Real, jump = rng -> randn(rng), θ::Real = 0.0,
        rng::AbstractRNG = Xoshiro())
    t0, tf = Float64.(tspan)
    times = Vector{Vector{Float64}}(undef, n); values = similar(times)
    for i in 1:n
        ts = [t0]; vs = [0.0]
        t = t0
        while true
            t += -log(rand(rng)) / rate
            t > tf && break
            push!(vs, vs[end] * exp(-θ * (t - ts[end])) + jump(rng))
            push!(ts, t)
        end
        times[i] = ts; values[i] = vs
    end
    return CompoundPoissonPath(Float64(θ), times, values, zeros(n))
end

function (p::CompoundPoissonPath)(t)
    for i in eachindex(p.out)
        k = max(searchsortedlast(p.times[i], t), 1)
        p.out[i] = p.values[i][k] * exp(-p.θ * (t - p.times[i][k]))
    end
    return p.out
end

"Jump times of a compound Poisson path, sorted, all components together."
jump_times(p::CompoundPoissonPath) = sort!(reduce(vcat, [ts[2:end] for ts in p.times]))

"""
    lorenz_path(tspan; timescale = 1.0, scale = 1.0, σ = 10.0, ρ = 28.0, β = 8/3,
        x0 = [1.0, 1.0, 20.0], spinup = 50.0, dt = 0.002)

A sample path of the Lorenz-63 system, a low-dimensional chaotic process with three
components, centered and scaled: returns `scale .* (x, y, z - ρ + 1) ./ 8`, evaluated at the
Lorenz time `(t - tspan[1]) / timescale`, after a spin-up of `spinup` Lorenz time units onto
the attractor. Integrated with the classical fourth-order Runge-Kutta method with step `dt`
(in Lorenz time) and interpolated with cubic Hermite polynomials, so it is smooth.
"""
struct LorenzPath <: SamplePath
    t0::Float64
    timescale::Float64
    scale::Float64
    dt::Float64
    X::Matrix{Float64}      # (3, ntimes)
    dX::Matrix{Float64}     # derivatives
    shift::Vector{Float64}
    out::Vector{Float64}
end

function lorenz_path(tspan; timescale::Real = 1.0, scale::Real = 1.0, σ::Real = 10.0, ρ::Real = 28.0,
        β::Real = 8/3, x0 = [1.0, 1.0, 20.0], spinup::Real = 50.0, dt::Real = 0.002)
    f(x) = [σ * (x[2] - x[1]), x[1] * (ρ - x[3]) - x[2], x[1] * x[2] - β * x[3]]
    rk4(x) = (k1 = f(x); k2 = f(x .+ dt / 2 .* k1); k3 = f(x .+ dt / 2 .* k2); k4 = f(x .+ dt .* k3);
        x .+ dt / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4))
    x = collect(Float64, x0)
    for _ in 1:round(Int, spinup / dt)
        x = rk4(x)
    end
    t0, tf = Float64.(tspan)
    m = ceil(Int, (tf - t0) / timescale / dt) + 2
    X = zeros(3, m); dX = zeros(3, m)
    for k in 1:m
        X[:, k] = x; dX[:, k] = f(x)
        x = rk4(x)
    end
    return LorenzPath(t0, Float64(timescale), Float64(scale), Float64(dt), X, dX, [0.0, 0.0, ρ - 1], zeros(3))
end

function (p::LorenzPath)(t)
    τ = (t - p.t0) / p.timescale / p.dt
    k = clamp(floor(Int, τ), 0, size(p.X, 2) - 2)
    s = clamp(τ - k, 0.0, 1.0)
    h00 = 2s^3 - 3s^2 + 1; h10 = s^3 - 2s^2 + s; h01 = -2s^3 + 3s^2; h11 = s^3 - s^2
    @inbounds for i in 1:3
        x = h00 * p.X[i, k+1] + h10 * p.dt * p.dX[i, k+1] + h01 * p.X[i, k+2] + h11 * p.dt * p.dX[i, k+2]
        p.out[i] = p.scale * (x - p.shift[i]) / 8
    end
    return p.out
end
