"""
    SimulationResult

Output of [`simulate`](@ref). Fields:
* `t`: sampling times;
* `energy, enstrophy, palinstrophy`: time series of `E`, `Z` and `P`;
* `umax, vmax`: time series of the maxima of `|u|` and `|v|`;
* `cfl`: time series of the advective CFL number (`NaN` for adaptive methods);
* `κ, spectrum`: wavenumbers and time-averaged energy spectrum (empty if not requested);
* `snapshots`: vector of `(t, ω̂)` pairs, if requested;
* `w_hat_final`: spectral vorticity at the last sampling time;
* `retcode`: `:Success`, `:BlowUp` (non-finite or exploding energy) or the solver's return code;
* `stats`: solver statistics (`nf`, `naccept`, `nreject`) and wall time in seconds.
"""
struct SimulationResult
    t::Vector{Float64}
    energy::Vector{Float64}
    enstrophy::Vector{Float64}
    palinstrophy::Vector{Float64}
    umax::Vector{Float64}
    vmax::Vector{Float64}
    cfl::Vector{Float64}
    κ::Vector{Float64}
    spectrum::Vector{Float64}
    snapshots::Vector{Tuple{Float64, Matrix{ComplexF64}}}
    w_hat_final::Matrix{ComplexF64}
    retcode::Symbol
    stats::NamedTuple
end

function Base.show(io::IO, r::SimulationResult)
    print(io, "SimulationResult($(r.retcode), $(length(r.t)) samples")
    isempty(r.t) || print(io, " in [$(r.t[1]), $(r.t[end])]")
    print(io, ", wall time $(round(r.stats.wall, sigdigits = 3)) s)")
end

"""
    simulate(p::NSEProblem, alg; save_every = 0.5, dt = nothing, spectrum = true,
        snapshot_every = nothing, blowup_energy = 1e8, progress = false, kwargs...)

Integrates `p` with the solver `alg` over `p.prob.tspan`, sampling the diagnostics every
`save_every` time units without storing the states, and returns a [`SimulationResult`](@ref).
Other keyword arguments (`reltol`, `abstol`, `adaptive`, `callback`, ...) are passed to the solver.

* With an adaptive method, the integration stops at each sampling time.
* With a fixed step (methods that are not adaptive, such as `ETDRK4()`, or `adaptive = false`),
  `dt` must divide `save_every`, and the solver takes exactly `save_every / dt` steps between
  samples, so the step is never changed (ETDRK4 in OrdinaryDiffEq does not allow that).
* `spectrum = true` accumulates the time-averaged energy spectrum.
* `snapshot_every`, if given, a multiple of `save_every`, keeps copies of the spectral
  vorticity at those intervals.
* The integration stops with `retcode = :BlowUp` if the energy becomes non-finite or larger
  than `blowup_energy`.
* `progress = true` prints a line every 10% of the time interval.
"""
# Checks the sampling parameters and returns (nsamples, nsub, snap_stride); nsub = 0 for
# adaptive integration.
function _sampling(p, save_every, dt, fixed, snapshot_every)
    t0, tf = p.prob.tspan
    nsamples = round(Int, (tf - t0) / save_every)
    abs(nsamples * save_every - (tf - t0)) ≤ 1e-9 * max(1.0, abs(tf - t0)) ||
        throw(ArgumentError("save_every = $save_every must divide the time interval $(tf - t0)"))
    nsub = 0
    if fixed
        dt === nothing && throw(ArgumentError("a fixed-step method requires `dt`"))
        nsub = round(Int, save_every / dt)
        (nsub ≥ 1 && abs(nsub * dt - save_every) ≤ 1e-9 * save_every) ||
            throw(ArgumentError("dt = $dt must divide save_every = $save_every"))
    end
    snap_stride = 0
    if snapshot_every !== nothing
        snap_stride = round(Int, snapshot_every / save_every)
        (snap_stride ≥ 1 && abs(snap_stride * save_every - snapshot_every) ≤ 1e-9 * snapshot_every) ||
            throw(ArgumentError("snapshot_every must be a multiple of save_every"))
    end
    return nsamples, nsub, snap_stride
end

# Accumulates the diagnostics along a run.
mutable struct _Recorder
    ts::Vector{Float64}; Es::Vector{Float64}; Zs::Vector{Float64}; Ps::Vector{Float64}
    us::Vector{Float64}; vs::Vector{Float64}; cfls::Vector{Float64}
    κ::Vector{Float64}; spec::Vector{Float64}; spectrum::Bool
    snaps::Vector{Tuple{Float64, Matrix{ComplexF64}}}; snap_stride::Int
    nsamples::Int; progress::Bool; blowup_energy::Float64
end

function _Recorder(m, w0, spectrum, snap_stride, nsamples, progress, blowup_energy)
    κ, spec = spectrum ? energy_spectrum(m, w0) : (Float64[], Float64[])
    return _Recorder(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
        collect(κ), zero(spec), spectrum, Tuple{Float64, Matrix{ComplexF64}}[], snap_stride,
        nsamples, progress, Float64(blowup_energy))
end

# Records the sample n at time t of the spectral vorticity w (a matrix); returns false on blow-up.
function _record!(r::_Recorder, m, n, t, w, dt)
    E = energy(m, w)
    (!isfinite(E) || E > r.blowup_energy) && return false
    umax, vmax = max_velocity(m, w)
    push!(r.ts, t); push!(r.Es, E); push!(r.Zs, enstrophy(m, w)); push!(r.Ps, palinstrophy(m, w))
    push!(r.us, umax); push!(r.vs, vmax)
    push!(r.cfls, dt === nothing ? NaN : dt * (m.grid.κx_max * umax + m.grid.κy_max * vmax))
    r.spectrum && (r.spec .+= energy_spectrum(m, w)[2])
    r.snap_stride > 0 && n % r.snap_stride == 0 && push!(r.snaps, (t, copy(w)))
    if r.progress && n % max(1, r.nsamples ÷ 10) == 0
        println("t = $(round(t, sigdigits = 6)): E = $(round(E, sigdigits = 4)), ",
            "Z = $(round(r.Zs[end], sigdigits = 4)), umax = $(round(umax, sigdigits = 4)), vmax = $(round(vmax, sigdigits = 4))")
    end
    return true
end

function _result(r::_Recorder, w_final, retcode, stats)
    r.spectrum && !isempty(r.ts) && (r.spec ./= length(r.ts))
    return SimulationResult(r.ts, r.Es, r.Zs, r.Ps, r.us, r.vs, r.cfls, r.κ, r.spec, r.snaps,
        copy(w_final), retcode, stats)
end

function simulate(p::NSEProblem, alg; save_every::Real = 0.5, dt = nothing, spectrum::Bool = true,
        snapshot_every = nothing, blowup_energy::Real = 1e8, progress::Bool = false, kwargs...)
    m = p.model
    t0 = p.prob.tspan[1]
    fixed = !SciMLBase.isadaptive(alg) || get(kwargs, :adaptive, true) == false
    nsamples, nsub, snap_stride = _sampling(p, save_every, dt, fixed, snapshot_every)
    solver_kwargs = dt === nothing ? kwargs : (; dt, kwargs...)
    integ = init(p.prob, alg; save_everystep = false, save_start = false, save_end = false,
        maxiters = typemax(Int), solver_kwargs...)
    rec = _Recorder(m, state_to_vorticity_hat(p, p.prob.u0), spectrum, snap_stride, nsamples, progress, blowup_energy)
    retcode = :Success
    wall = @elapsed for n in 1:nsamples
        if fixed
            for _ in 1:nsub
                step!(integ)
            end
        else
            step!(integ, t0 + n * save_every - integ.t, true)
        end
        if !(integ.sol.retcode in (ReturnCode.Default, ReturnCode.Success))
            retcode = Symbol(integ.sol.retcode); break
        end
        t = fixed ? t0 + n * save_every : integ.t
        _record!(rec, m, n, t, state_to_vorticity_hat(p, integ.u), fixed ? dt : nothing) || (retcode = :BlowUp; break)
    end
    stats = (; nf = integ.stats.nf + integ.stats.nf2, naccept = integ.stats.naccept,
        nreject = integ.stats.nreject, wall)
    return _result(rec, state_to_vorticity_hat(p, integ.u), retcode, stats)
end

"""
    simulate(p::NSEProblem, scheme::AbstractFixedStepper; dt, save_every = 0.5, seed = nothing,
        rng = Xoshiro(seed), spectrum = true, snapshot_every = nothing, blowup_energy = 1e8,
        progress = false)

Integrates `p` with one of the fixed-step schemes of this package, [`StochasticExpEuler`](@ref)
or [`SplitStochasticRK`](@ref), with time step `dt` (which must divide `save_every`), sampling
the diagnostics as the other method of `simulate`. The Wiener increments of the stochastic
forcing, if any, are drawn from `rng`. Works with any `kind` of problem (for a model with
a stochastic forcing, the problem has `kind = :sde`).
"""
function simulate(p::NSEProblem, scheme::AbstractFixedStepper; dt::Real, save_every::Real = 0.5,
        seed = nothing, rng::AbstractRNG = seed === nothing ? Xoshiro() : Xoshiro(seed), spectrum::Bool = true,
        snapshot_every = nothing, blowup_energy::Real = 1e8, progress::Bool = false)
    m = p.model
    t0 = p.prob.tspan[1]
    nsamples, nsub, snap_stride = _sampling(p, save_every, dt, true, snapshot_every)
    w = Vector{ComplexF64}(vec(copy(state_to_vorticity_hat(p, p.prob.u0))))
    cache = _cache(scheme, m, dt)
    rec = _Recorder(m, _mat(m, w), spectrum, snap_stride, nsamples, progress, blowup_energy)
    retcode = :Success
    steps = 0
    wall = @elapsed for n in 1:nsamples
        for k in 1:nsub
            _step!(w, cache, scheme, m, t0 + ((n - 1) * nsub + k - 1) * dt, dt, rng)
        end
        steps += nsub
        _record!(rec, m, n, t0 + n * save_every, _mat(m, w), dt) || (retcode = :BlowUp; break)
    end
    stats = (; nf = steps * _evals_per_step(scheme), naccept = steps, nreject = 0, wall)
    return _result(rec, _mat(m, w), retcode, stats)
end
