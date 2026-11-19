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
function simulate(p::NSEProblem, alg; save_every::Real = 0.5, dt = nothing, spectrum::Bool = true,
        snapshot_every = nothing, blowup_energy::Real = 1e8, progress::Bool = false, kwargs...)
    m = p.model
    t0, tf = p.prob.tspan
    nsamples = round(Int, (tf - t0) / save_every)
    abs(nsamples * save_every - (tf - t0)) ≤ 1e-9 * max(1.0, abs(tf - t0)) ||
        throw(ArgumentError("save_every = $save_every must divide the time interval $(tf - t0)"))

    fixed = !SciMLBase.isadaptive(alg) || get(kwargs, :adaptive, true) == false
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

    solver_kwargs = dt === nothing ? kwargs : (; dt, kwargs...)
    integ = init(p.prob, alg; save_everystep = false, save_start = false, save_end = false,
        maxiters = typemax(Int), solver_kwargs...)

    ts = Float64[]; Es = Float64[]; Zs = Float64[]; Ps = Float64[]
    us = Float64[]; vs = Float64[]; cfls = Float64[]
    κ, spec = spectrum ? energy_spectrum(m, state_to_vorticity_hat(p, p.prob.u0)) : (Float64[], Float64[])
    spec = zero(spec)
    snaps = Tuple{Float64, Matrix{ComplexF64}}[]
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
        w = state_to_vorticity_hat(p, integ.u)
        E = energy(m, w)
        if !isfinite(E) || E > blowup_energy
            retcode = :BlowUp; break
        end
        umax, vmax = max_velocity(m, w)
        push!(ts, fixed ? t0 + n * save_every : integ.t); push!(Es, E); push!(Zs, enstrophy(m, w)); push!(Ps, palinstrophy(m, w))
        push!(us, umax); push!(vs, vmax)
        push!(cfls, fixed ? dt * (m.grid.κx_max * umax + m.grid.κy_max * vmax) : NaN)
        spectrum && (spec .+= energy_spectrum(m, w)[2])
        snap_stride > 0 && n % snap_stride == 0 && push!(snaps, (integ.t, copy(w)))
        if progress && n % max(1, nsamples ÷ 10) == 0
            println("t = $(round(integ.t, sigdigits = 6)): E = $(round(E, sigdigits = 4)), ",
                "Z = $(round(Zs[end], sigdigits = 4)), umax = $(round(umax, sigdigits = 4)), vmax = $(round(vmax, sigdigits = 4))")
        end
    end
    spectrum && !isempty(ts) && (spec ./= length(ts))
    stats = (; nf = integ.stats.nf + integ.stats.nf2, naccept = integ.stats.naccept,
        nreject = integ.stats.nreject, wall)
    return SimulationResult(ts, Es, Zs, Ps, us, vs, cfls, collect(κ), spec, snaps,
        copy(state_to_vorticity_hat(p, integ.u)), retcode, stats)
end
