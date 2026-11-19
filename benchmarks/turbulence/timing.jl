# Cost per step of each method, single-threaded, run alone (no other processes competing),
# plus a hand-written CNAB2 with one evaluation of the nonlinear term per step.
include("turb_common.jl")
ν = parse(Float64, ARGS[1]); grms = parse(Float64, ARGS[2]); t0 = parse(Float64, ARGS[3])
N = 128; Nsub = 84
g, _ = band_forcing(N, 2, 4, grms)
g_hat = dealias!(rfft(g), N, Nsub)
p = make_params(N, Nsub, ν, g_hat; nthreads = 1)
w0 = deserialize(joinpath("warm_$(ν)_$(grms)", "state_$(Int(t0)).jls"))
dt = 2e-4; nsteps = 2000; tspan = (t0, t0 + nsteps * dt)

# Crank-Nicolson for νΔ, Adams-Bashforth 2 for the nonlinear term; first step by
# Crank-Nicolson/forward Euler.
function cnab2_hand(w0, p, dt, nsteps)
    w = copy(w0); Nn = similar(w); Nm = similar(w)
    L = p.nu_Delta
    num = @. 1 + dt / 2 * L; den = @. 1 - dt / 2 * L
    nonlinear_vec!(Nm, w, p, 0.0)
    @. w = (num * w + dt * Nm) / den
    for _ in 2:nsteps
        nonlinear_vec!(Nn, w, p, 0.0)
        @. w = (num * w + dt * (1.5 * Nn - 0.5 * Nm)) / den
        Nm, Nn = Nn, Nm
    end
    return w
end

cnab2_hand(w0, p, dt, 10)
t_hand = @elapsed w_hand = cnab2_hand(w0, p, dt, nsteps)
println(@sprintf("%-22s %8.3f ms/step", "CNAB2 (hand-written)", 1e3 * t_hand / nsteps))

for (name, prob, alg) in [
        ("ETDRK4", split_problem(w0, tspan, p), ETDRK4()),
        ("RK4", full_problem(w0, tspan, p), RK4()),
        ("SSPRK33", full_problem(w0, tspan, p), SSPRK33()),
        ("CNAB2 (OrdinaryDiffEq)", split_problem(w0, tspan, p), CNAB2(linsolve = DiagonalFactorization())),
    ]
    solve(prob, alg; dt, adaptive = false, save_everystep = false)
    t = @elapsed sol = solve(prob, alg; dt, adaptive = false, save_everystep = false)
    println(@sprintf("%-22s %8.3f ms/step", name, 1e3 * t / nsteps))
    if name == "CNAB2 (OrdinaryDiffEq)"
        println(@sprintf("  relative difference hand-written vs OrdinaryDiffEq CNAB2 after %d steps: %.2e",
            nsteps, norm(sol.u[end] - w_hand) / norm(w_hand)))
    end
end
w = similar(w0); prob = full_problem(w0, tspan, p)
solve(prob, Tsit5(); save_everystep = false)
t = @elapsed sol = solve(prob, Tsit5(); dt, adaptive = false, save_everystep = false)
println(@sprintf("%-22s %8.3f ms/step (fixed step, for reference)", "Tsit5", 1e3 * t / nsteps))
dw = similar(w0); nonlinear_vec!(dw, w0, p, 0.0)
t = @elapsed for _ in 1:nsteps; nonlinear_vec!(dw, w0, p, 0.0); end
println(@sprintf("%-22s %8.3f ms", "one nonlinear eval", 1e3 * t / nsteps))
