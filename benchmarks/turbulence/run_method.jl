# julia run_method.jl ν grms t0 Ustar method param T
#   method ∈ ETDRK4, RK4, SSPRK33, CNAB2 (param = target CFL number σ = dt kmax Ustar)
#   method = Tsit5 (param = reltol; abstol = 1e-3 * reltol)
include("turb_common.jl")
ν = parse(Float64, ARGS[1]); grms = parse(Float64, ARGS[2]); t0 = parse(Float64, ARGS[3])
Ustar = parse(Float64, ARGS[4]); method = ARGS[5]; param = parse(Float64, ARGS[6]); T = parse(Float64, ARGS[7])
N = 128; Nsub = 84; Δs = 0.5
g, _ = band_forcing(N, 2, 4, grms)
g_hat = dealias!(rfft(g), N, Nsub)
p = make_params(N, Nsub, ν, g_hat; nthreads = 1)
tag = "warm_$(ν)_$(grms)"
w0 = deserialize(joinpath(tag, "state_$(Int(t0)).jls"))
kmax = Diag(p).kmax
tspan = (t0, t0 + T)

if method == "Tsit5"
    r = run_stats(full_problem(w0, tspan, p), Tsit5(), p, T, Δs; reltol = param, abstol = 1e-3 * param)
    dt = NaN
else
    nsub = ceil(Int, Δs * kmax * Ustar / param)        # steps per sampling interval
    dt = Δs / nsub
    if method == "CNAB2hand"
        r = run_cnab2_hand(w0, p, dt, nsub, T, Δs, t0)
    else
    prob, alg = method == "ETDRK4"  ? (split_problem(w0, tspan, p), ETDRK4()) :
                method == "CNAB2"   ? (split_problem(w0, tspan, p), CNAB2(linsolve = DiagonalFactorization())) :
                method == "RK4"     ? (full_problem(w0, tspan, p), RK4()) :
                method == "SSPRK33" ? (full_problem(w0, tspan, p), SSPRK33()) : error("unknown method")
    r = run_stats(prob, alg, p, T, Δs; nsub, dt, adaptive = false)
    end
end
σ_actual = isnan(dt) ? NaN : dt * kmax * Ustar
mkpath("results")
out = (; method, param, dt, σ = σ_actual, ν, grms, t0, T, kmax, r...)
serialize(joinpath("results", "$(method)_$(param).jls"), out)
@printf("%-8s param=%-8g dt=%-10.4g σ=%-6.3g blowup=%-5s tend=%-8.1f wall=%7.1fs steps=%8d nf=%9d  ⟨E⟩=%.4g ⟨Z⟩=%.4g ⟨P⟩=%.4g max CFL=%.3g\n",
    method, param, dt, σ_actual, r.blowup, isempty(r.ts) ? t0 : r.ts[end], r.wall, r.naccept, r.nf,
    mean(r.Es), mean(r.Zs), mean(r.Ps), isnan(dt) ? NaN : dt * kmax * maximum(r.Us; init = 0.0))
