include("turb_common.jl")
ν = parse(Float64, ARGS[1]); grms = parse(Float64, ARGS[2]); T = parse(Float64, ARGS[3])
N = 128; Nsub = 84; chunk = 100.0
g, modes = band_forcing(N, 2, 4, grms)
g_hat = dealias!(rfft(g), N, Nsub)
p = make_params(N, Nsub, ν, g_hat; nthreads = 1)
w = vec(- g_hat ./ (ν .* p.ops[3])); w[1] = 0          # (1/ν) A⁻¹ g, in vorticity
tag = "warm_$(ν)_$(grms)"; mkpath(tag)
t = 0.0; series = NamedTuple[]
while t < T
    r = run_stats(full_problem(w, (t, t + chunk), p), Tsit5(), p, chunk, 0.5)
    global w = r.u_end; global t += chunk
    push!(series, (; r.ts, r.Es, r.Zs, r.Ps, r.Us))
    serialize(joinpath(tag, "state_$(Int(t)).jls"), w)
    @printf("t=%6.0f ⟨E⟩=%9.4g ⟨Z⟩=%9.4g ⟨P⟩=%9.4g κσ=√(⟨P⟩/⟨Z⟩)=%6.3g max(|u|+|v|)=%7.3g  E(1)/E=%5.2f steps=%7d wall=%6.1fs\n",
        t, mean(r.Es), mean(r.Zs), mean(r.Ps), sqrt(mean(r.Ps)/mean(r.Zs)), maximum(r.Us), r.spec[1]/sum(r.spec), r.naccept, r.wall)
    flush(stdout)
end
serialize(joinpath(tag, "series.jls"), series)
