# Physical fields and diagnostics, all taking the spectral vorticity `w_hat` of size (Mx, Ny).
# Means are spatial means over the domain.

"Physical vorticity, of size `(Nx, Ny)`."
vorticity(m::NSEModel, w_hat) = (m.sc .= w_hat; m.iplan * m.sc)

"Physical velocity field `(u, v)`, each of size `(Nx, Ny)`."
function velocity(m::NSEModel, w_hat)
    u = zeros(m.grid.Nx, m.grid.Ny); v = similar(u)
    return _velocity!(u, v, w_hat, m)
end

"Physical stream function `ψ`, with `-Δψ = ω`, `u = ∂ᵧψ`, `v = -∂ₓψ`, and zero mean."
function streamfunction(m::NSEModel, w_hat)
    k2 = m.grid.κ2
    @. m.sc = ifelse(k2 > 0, w_hat / k2, zero(ComplexF64))
    return m.iplan * m.sc
end

function _quadratic(m::NSEModel, w_hat, p)
    W = spectral_weights(m.grid); k2 = m.grid.κ2
    s = 0.0
    @inbounds for i in eachindex(w_hat)
        k2[i] > 0 || continue
        s += W[i] * abs2(w_hat[i]) * k2[i]^p
    end
    return s / 2
end

"Kinetic energy `E = ½⟨|u|²⟩`."
energy(m::NSEModel, w_hat) = _quadratic(m, w_hat, -1)
"Enstrophy `Z = ½⟨|∇u|²⟩ = ½⟨ω²⟩`."
enstrophy(m::NSEModel, w_hat) = _quadratic(m, w_hat, 0)
"Palinstrophy `P = ½⟨|Δu|²⟩ = ½⟨|∇ω|²⟩`."
palinstrophy(m::NSEModel, w_hat) = _quadratic(m, w_hat, 1)

"""
    kappa_sigma(Z, P)

The Taylor-like wavenumber `κσ = √(⟨P⟩/⟨Z⟩) = √(⟨‖Δu‖²⟩/⟨‖∇u‖²⟩)`, from time averages (or
instantaneous values) of the enstrophy `Z` and palinstrophy `P`; vectors are averaged first.
"""
kappa_sigma(Z::Real, P::Real) = sqrt(P / Z)
kappa_sigma(Z::AbstractVector, P::AbstractVector) = sqrt(mean(P) / mean(Z))

"""
    energy_spectrum(m, w_hat)

Shell-summed energy spectrum: returns `(κ, E)`, where `E[n]` is the energy in the modes with
`(n - ½)Δκ ≤ |κ| < (n + ½)Δκ`, with `Δκ = 2π / max(Lx, Ly)`, so that `sum(E)` is the energy.
"""
function energy_spectrum(m::NSEModel, w_hat)
    g = m.grid
    Δκ = 2π / max(g.Lx, g.Ly)
    nmax = ceil(Int, sqrt(g.κx_max^2 + g.κy_max^2) / Δκ) + 1
    E = zeros(nmax)
    W = spectral_weights(g)
    @inbounds for i in eachindex(w_hat)
        g.κ2[i] > 0 || continue
        n = round(Int, sqrt(g.κ2[i]) / Δκ)
        n < 1 && continue
        E[n] += W[i] * abs2(w_hat[i]) / g.κ2[i] / 2
    end
    return Δκ .* (1:nmax), E
end

"Maximum of `|u|` and of `|v|` over the grid, `(umax, vmax)`."
function max_velocity(m::NSEModel, w_hat)
    _velocity!(m.u, m.v, w_hat, m)
    return maximum(abs, m.u), maximum(abs, m.v)
end

"""
    cfl_number(m, w_hat, dt)

The advective CFL number `dt (κx_max |u|max + κy_max |v|max)`, the bound on the size of the
advection eigenvalues times the time step. The viscous counterpart is `dt ν (κx_max² + κy_max²)`.
"""
function cfl_number(m::NSEModel, w_hat, dt)
    umax, vmax = max_velocity(m, w_hat)
    return dt * (m.grid.κx_max * umax + m.grid.κy_max * vmax)
end

"""
    suggest_dt(m, w_hat; σ = 2.0, umax = nothing, vmax = nothing)

A fixed time step for explicit or exponential methods, `dt = σ / (κx_max umax + κy_max vmax)`,
with the maximum velocities of `w_hat` unless given. As a reference, from the experiments in
the notes, RK4, ETDRK4 and SSPRK33 were stable with `σ` up to about 4 and blew up at `σ = 6`
(with the velocity maxima over a long run, including bursts), and CNAB2 needs `σ` about five
times smaller. For flows with intermittent bursts, take `umax` and `vmax` from a long warm-up,
not from a single state. Also bounded by `σ / (ν (κx_max² + κy_max²))` for explicit methods
when the viscous term dominates (use a split problem and ETDRK4 in that case).
"""
function suggest_dt(m::NSEModel, w_hat; σ::Real = 2.0, umax = nothing, vmax = nothing)
    if umax === nothing || vmax === nothing
        u0, v0 = max_velocity(m, w_hat)
        umax = umax === nothing ? u0 : umax
        vmax = vmax === nothing ? v0 : vmax
    end
    rate = m.grid.κx_max * umax + m.grid.κy_max * vmax
    rate > 0 || return Inf
    return σ / rate
end

"""
    batch_mean_ci(x; nbatches = 10)

Mean of the time series `x` and the half-width of its 95% confidence interval, by batch means
(the batches should be longer than the correlation time of `x`, otherwise the interval is
underestimated). Returns `(mean, halfwidth)`.
"""
function batch_mean_ci(x::AbstractVector; nbatches::Integer = 10)
    n = length(x) ÷ nbatches
    n ≥ 1 || throw(ArgumentError("need at least $nbatches samples"))
    bm = [mean(@view x[(b-1)*n+1:b*n]) for b in 1:nbatches]
    return mean(bm), _tquantile975(nbatches - 1) * std(bm) / sqrt(nbatches)
end

# Two-sided 95% quantiles of Student's t distribution, for the batch-mean intervals.
function _tquantile975(dof::Integer)
    table = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086)
    dof ≤ 20 && return table[dof]
    dof ≤ 30 && return 2.042
    dof ≤ 60 && return 2.000
    return 1.960
end
