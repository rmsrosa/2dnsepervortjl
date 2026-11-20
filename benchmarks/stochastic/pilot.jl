include("common.jl")
println(model, "  η = ", η, "  modes: ", length(modes))
r = simulate(NSEProblem(model, zeros(N, N), (0.0, 600.0); kind = :sde), StochasticETDRK4(); dt = 0.01, save_every = 0.5, seed = 1, progress = true)
println(r)
for (a, b) in ((50, 150), (150, 300), (300, 450), (450, 600))
    s = a .< r.t .≤ b
    @printf("t ∈ (%d, %d]: ⟨E⟩ = %.3f ⟨Z⟩ = %.3f 2ν⟨Z⟩/ε = %.3f 2ν⟨P⟩/η = %.3f max(|u|+|v|) = %.2f κσ = %.2f\n", a, b,
        mean(r.energy[s]), mean(r.enstrophy[s]), 2ν * mean(r.enstrophy[s]) / ε, 2ν * mean(r.palinstrophy[s]) / η,
        maximum(r.umax[s] .+ r.vmax[s]), kappa_sigma(r.enstrophy[s], r.palinstrophy[s]))
end
@printf("spectrum E(kmax)/max = %.2g, max CFL %.2f\n", r.spectrum[findlast(>(0), r.spectrum)] / maximum(r.spectrum), maximum(r.cfl))
serialize("warm.jls", (; t0 = r.t[end], w = r.w_hat_final))
