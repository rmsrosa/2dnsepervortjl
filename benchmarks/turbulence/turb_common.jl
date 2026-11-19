# Shared code for the chaotic-flow experiment: forced 2D NSE in vorticity form, N = 128.
using FFTW, LinearAlgebra, Random, Printf, Serialization, Statistics
using OrdinaryDiffEq
using OrdinaryDiffEqExponentialRK: ETDRK4
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqSSPRK: SSPRK33
using OrdinaryDiffEqIMEXMultistep: CNAB2
using SciMLOperators: DiagonalOperator
using LinearSolve: DiagonalFactorization

function get_operators(N, κ₀)
    Dx_hat = im * κ₀ * [ifelse(k1 ≤ div(N, 2) + 1, k1 - 1, k1 - 1 - N) for k2 in 1:div(N, 2)+1, k1 in 1:N]
    Dy_hat = im * κ₀ * [k2 - 1 for k2 in 1:div(N, 2)+1, k1 in 1:N]
    Delta_hat = - κ₀^2 * [ifelse(k1 ≤ div(N, 2) + 1, (k1 - 1)^2 + (k2 - 1)^2, (k1 - 1 - N)^2 + (k2 - 1)^2)
        for k2 in 1:div(N, 2)+1, k1 in 1:N]
    DxsqDysq_hat = Dx_hat.^2 .- Dy_hat.^2
    Dxy_hat = Dx_hat .* Dy_hat
    Hu_hat = - Dy_hat ./ Delta_hat; Hu_hat[1, 1] = 0.0
    Hv_hat = Dx_hat ./ Delta_hat; Hv_hat[1, 1] = 0.0
    return Dx_hat, Dy_hat, Delta_hat, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat
end

function dealias!(w_hat, N, Nsub)
    w_hat[div(Nsub,2) + 1:end, :] .= 0
    w_hat[:, div(Nsub,2) + 1:div(N,2) + div(Nsub,2)] .= 0
    return w_hat
end

# Curl of the forcing: all integer modes with kmin ≤ |k| ≤ kmax, with random phases and equal
# amplitudes, scaled so that the curl has root-mean-square value `grms` in physical space.
function band_forcing(N, kmin, kmax, grms; seed = 2026)
    rng = Xoshiro(seed)
    L = 2π; x = y = (L/N):(L/N):L
    g = zeros(N, N)
    modes = Tuple{Int,Int}[]
    for kx in -kmax:kmax, ky in 0:kmax
        kmin^2 ≤ kx^2 + ky^2 ≤ kmax^2 || continue
        ky == 0 && kx < 0 && continue          # (kx, 0) and (-kx, 0) are the same real mode
        push!(modes, (kx, ky))
        ϕ = 2π * rand(rng)
        g .+= cos.(kx .* one.(y) .* x' .+ ky .* y .* one.(x)' .+ ϕ)
    end
    g .*= grms / sqrt(mean(abs2, g))
    return g, modes
end

struct Params{O, G, P1, P2, R, C}
    ops::O; g_hat::G; N::Int; Nsub::Int; M::Int; ν::Float64
    plan::P1; plan_inv::P2
    u::R; v::R; a::R; b::R
    sc::C; a_hat::C; b_hat::C
    nu_Delta::Vector{ComplexF64}
end

function make_params(N, Nsub, ν, g_hat; nthreads = 1)
    FFTW.set_num_threads(nthreads)
    ops = get_operators(N, 1.0)
    M = div(N, 2) + 1
    r() = zeros(N, N); c() = zeros(ComplexF64, M, N)
    plan = plan_rfft(r(), flags = FFTW.MEASURE); plan_inv = plan_irfft(c(), N, flags = FFTW.MEASURE)
    return Params(ops, g_hat, N, Nsub, M, ν, plan, plan_inv, r(), r(), r(), r(), c(), c(), c(),
        vec(complex.(ν .* ops[3])))
end

function nonlinear!(dw_hat, w_hat, p::Params)
    _, _, _, Hu_hat, Hv_hat, DxsqDysq_hat, Dxy_hat = p.ops
    p.sc .= Hu_hat .* w_hat; mul!(p.u, p.plan_inv, p.sc)
    p.sc .= Hv_hat .* w_hat; mul!(p.v, p.plan_inv, p.sc)
    @. p.a = p.u * p.v
    @. p.b = p.v^2 - p.u^2
    mul!(p.a_hat, p.plan, p.a); mul!(p.b_hat, p.plan, p.b)
    @. dw_hat = p.g_hat - DxsqDysq_hat * p.a_hat - Dxy_hat * p.b_hat
    dealias!(dw_hat, p.N, p.Nsub)
    return dw_hat
end

# Vectorized complex state (length M*N), as required by the DiagonalOperator of the split problem
nonlinear_vec!(dw, w, p, t) = (nonlinear!(reshape(dw, p.M, :), reshape(w, p.M, :), p); dw)
full_vec!(dw, w, p, t) = (nonlinear_vec!(dw, w, p, t); dw .+= p.nu_Delta .* w; dw)

full_problem(w0, tspan, p) = ODEProblem(full_vec!, w0, tspan, p)
split_problem(w0, tspan, p) = SplitODEProblem(DiagonalOperator(p.nu_Delta), nonlinear_vec!, w0, tspan, p)

# ---------------------------------------------------------------------------
# Diagnostics. With the rfft half-spectrum, the modes with ky > 0 count twice.
struct Diag
    wgt::Matrix{Float64}; k2::Matrix{Float64}; shell::Matrix{Int}; nshell::Int; kmax::Float64
end
function Diag(p::Params)
    k2 = -real.(p.ops[3])
    wgt = [i == 1 ? 1.0 : 2.0 for i in 1:p.M, j in 1:p.N] ./ Float64(p.N)^4
    shell = round.(Int, sqrt.(k2))
    return Diag(wgt, k2, shell, div(p.Nsub, 2), div(p.Nsub, 2) - 1)
end

# Returns E = ½⟨|u|²⟩, Z = ½⟨|∇u|²⟩ = ½⟨ω²⟩, P = ½⟨|Δu|²⟩ = ½⟨|∇ω|²⟩ (spatial means)
function quadratic(w_hat, d::Diag)
    E = Z = P = 0.0
    @inbounds for i in eachindex(w_hat)
        a = d.wgt[i] * abs2(w_hat[i]) / 2
        d.k2[i] == 0 && continue
        E += a / d.k2[i]; Z += a; P += a * d.k2[i]
    end
    return E, Z, P
end

function add_spectrum!(spec, w_hat, d::Diag)
    @inbounds for i in eachindex(w_hat)
        s = d.shell[i]
        (d.k2[i] == 0 || s > d.nshell) && continue
        spec[s] += d.wgt[i] * abs2(w_hat[i]) / 2 / d.k2[i]
    end
    return spec
end

function max_velocity(w_hat, p::Params)
    p.sc .= p.ops[4] .* w_hat; mul!(p.u, p.plan_inv, p.sc)
    p.sc .= p.ops[5] .* w_hat; mul!(p.v, p.plan_inv, p.sc)
    return maximum(abs, p.u) + maximum(abs, p.v)
end

# ---------------------------------------------------------------------------
# Runs `alg` from w0 for a time T, sampling the diagnostics every Δs, and returns the time
# series, the time-averaged spectrum and the cost. Stops if the solution blows up.
function run_stats(prob, alg, p, T, Δs; nsub = nothing, solve_kwargs...)
    d = Diag(p)
    integ = init(prob, alg; save_everystep = false, save_start = false, save_end = false,
        maxiters = typemax(Int), solve_kwargs...)
    nsamp = round(Int, T / Δs)
    ts = Float64[]; Es = Float64[]; Zs = Float64[]; Ps = Float64[]; Us = Float64[]
    spec = zeros(d.nshell)
    blowup = false
    t0 = integ.t
    wall = @elapsed for n in 1:nsamp
        if nsub === nothing
            step!(integ, t0 + n * Δs - integ.t, true)    # adaptive: land on the sample time
        else
            for _ in 1:nsub; step!(integ); end         # fixed step: exactly nsub steps
        end
        w = reshape(integ.u, p.M, :)
        E, Z, P = quadratic(w, d)
        if !isfinite(E) || E > 1e8 || integ.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
            blowup = true; break
        end
        push!(ts, integ.t); push!(Es, E); push!(Zs, Z); push!(Ps, P); push!(Us, max_velocity(w, p))
        add_spectrum!(spec, w, d)
    end
    spec ./= max(length(ts), 1)
    return (; ts, Es, Zs, Ps, Us, spec, wall, blowup, nf = integ.stats.nf + integ.stats.nf2,
        naccept = integ.stats.naccept, nreject = integ.stats.nreject, u_end = copy(integ.u))
end

# Mean and 95% confidence half-width by batch means
const TQ = Dict(10 => 2.262, 20 => 2.093)   # t-quantiles 0.975 with nb - 1 dof
function batch_means(x; nb = 10)
    n = length(x) ÷ nb
    return [mean(@view x[(b-1)*n+1:b*n]) for b in 1:nb]
end
ci(bm) = (mean(bm), TQ[length(bm)] * std(bm) / sqrt(length(bm)))

# Hand-written CNAB2 (Crank-Nicolson for νΔ, Adams-Bashforth 2 for the rest), one evaluation of
# the nonlinear term per step, with the same sampling as run_stats. First step: CN/forward Euler.
function run_cnab2_hand(w0, p, dt, nsub, T, Δs, t0)
    d = Diag(p)
    w = copy(w0); Nn = similar(w); Nm = similar(w)
    L = p.nu_Delta
    num = @. 1 + dt / 2 * L; den = @. 1 - dt / 2 * L
    ts = Float64[]; Es = Float64[]; Zs = Float64[]; Ps = Float64[]; Us = Float64[]
    spec = zeros(d.nshell); blowup = false; nsteps = 0
    wall = @elapsed begin
        nonlinear_vec!(Nm, w, p, 0.0)
        @. w = (num * w + dt * Nm) / den
        nsteps = 1
        for n in 1:round(Int, T / Δs)
            for _ in (n == 1 ? 2 : 1):nsub
                nonlinear_vec!(Nn, w, p, 0.0)
                @. w = (num * w + dt * (1.5 * Nn - 0.5 * Nm)) / den
                Nm, Nn = Nn, Nm
                nsteps += 1
            end
            wm = reshape(w, p.M, :)
            E, Z, P = quadratic(wm, d)
            if !isfinite(E) || E > 1e8
                blowup = true; break
            end
            push!(ts, t0 + n * Δs); push!(Es, E); push!(Zs, Z); push!(Ps, P); push!(Us, max_velocity(wm, p))
            add_spectrum!(spec, wm, d)
        end
    end
    spec ./= max(length(ts), 1)
    return (; ts, Es, Zs, Ps, Us, spec, wall, blowup, nf = nsteps, naccept = nsteps, nreject = 0, u_end = w)
end
