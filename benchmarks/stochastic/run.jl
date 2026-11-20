# julia --project=. run.jl method param T   (param: dt for fixed-step schemes, reltol = abstol for adaptive)
include("common.jl")
method, param, T = ARGS[1], parse(Float64, ARGS[2]), parse(Float64, ARGS[3])
warm = deserialize("warm.jls")
p = NSEProblem(model, warm.w, (warm.t0, warm.t0 + T); kind = :sde)
seed = 1000 + hash(ARGS) % 1000
r = if method == "ETDRK4"
    simulate(p, StochasticETDRK4(); dt = param, save_every = 0.5, seed)
elseif method == "ExpEuler"
    simulate(p, StochasticExpEuler(); dt = param, save_every = 0.5, seed)
elseif method == "SplitRK4"
    simulate(p, SplitStochasticRK(:rk4); dt = param, save_every = 0.5, seed)
elseif method == "SplitSSPRK3"
    simulate(p, SplitStochasticRK(:ssprk3); dt = param, save_every = 0.5, seed)
elseif method == "EM"
    simulate(p, EM(); dt = param, save_every = 0.5, seed)
elseif method == "SRA1"
    simulate(p, SRA1(); reltol = param, abstol = param, save_every = 0.5, seed)
elseif method == "SOSRA"
    simulate(p, SOSRA(); reltol = param, abstol = param, save_every = 0.5, seed)
end
mkpath("results")
serialize(joinpath("results", "$(method)_$(param).jls"), (; method, param, T, r))
if r.retcode == :Success
    bE, hE, bZ, hZ = balances(r)
    @printf("%-12s %-8g %-8s wall %7.1f s  steps %8d  nf %9d  energy balance %+.4f ± %.4f  enstrophy balance %+.4f ± %.4f  ⟨E⟩ %.3f  max CFL %.2f\n",
        method, param, r.retcode, r.stats.wall, r.stats.naccept, r.stats.nf, bE, hE, bZ, hZ, mean(r.energy), maximum(r.cfl))
else
    @printf("%-12s %-8g %-8s at t = %.1f  wall %7.1f s\n", method, param, r.retcode, isempty(r.t) ? NaN : r.t[end], r.stats.wall)
end
