# Table and figures from results/*.jls:  julia --project=. analyze.jl [outdir]
include("common.jl")
using Plots
outdir = get(ARGS, 1, "figs"); mkpath(outdir)
label = Dict("ETDRK4" => "Stochastic ETDRK4", "ExpEuler" => "Stochastic exponential Euler",
    "SplitRK4" => "Split RK4", "SplitSSPRK3" => "Split SSPRK3", "EM" => "Euler-Maruyama",
    "SRA1" => "SRA1", "SOSRA" => "SOSRA")
order = ["ETDRK4", "SplitRK4", "SplitSSPRK3", "ExpEuler", "EM", "SRA1", "SOSRA"]
runs = [deserialize(joinpath("results", f)) for f in readdir("results") if endswith(f, ".jls")]
sort!(runs, by = x -> (findfirst(==(x.method), order), x.method in ("SRA1", "SOSRA") ? -x.param : -x.param))
io = IOBuffer()
println(io, "| Method | dt or reltol | Status | Max CFL | Energy balance 2ν⟨Z⟩/ε - 1 | Enstrophy balance 2ν⟨P⟩/η - 1 | ⟨E⟩ | Nonlinear evaluations per time unit | Wall time per time unit (s) |")
println(io, "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
rows = []
for x in runs
    r = x.r
    ok = r.retcode == :Success
    par = x.method in ("SRA1", "SOSRA") ? "reltol = $(x.param)" : "dt = $(x.param)"
    T = isempty(r.t) ? NaN : r.t[end] - r.t[1] + 0.5
    if ok
        bE, hE, bZ, hZ = balances(r)
        push!(rows, (; x.method, x.param, bE, hE, bZ, hZ, evals = r.stats.nf / x.T, wall = r.stats.wall / x.T))
        @printf(io, "| %s | %s | stable | %s | %+.3f ± %.3f | %+.3f ± %.3f | %.3f | %.0f | %.4f |\n", label[x.method], par,
            all(isnan, r.cfl) ? "–" : @sprintf("%.2f", maximum(r.cfl)), bE, hE, bZ, hZ, mean(r.energy), r.stats.nf / x.T, r.stats.wall / x.T)
    else
        @printf(io, "| %s | %s | blew up at t = %.0f | – | – | – | – | – | – |\n", label[x.method], par, isempty(r.t) ? NaN : r.t[end])
    end
end
table = String(take!(io)); write(joinpath(outdir, "table.md"), table); println(table)
colors = Dict("ETDRK4" => :green, "SplitRK4" => :blue, "SplitSSPRK3" => :orange, "ExpEuler" => :purple,
    "EM" => :red, "SRA1" => :black, "SOSRA" => :gray)
for (key, h, name, file) in [(:bE, :hE, "energy balance 2ν⟨Z⟩/ε - 1", "energy_balance"), (:bZ, :hZ, "enstrophy balance 2ν⟨P⟩/η - 1", "enstrophy_balance")]
    plt = plot(xlabel = "wall time per time unit (s)", ylabel = name, xscale = :log10, legend = :outerright,
        size = (850, 420), title = "Relative imbalance, with 95% confidence intervals", titlefont = 10)
    hline!(plt, [0.0], color = :gray, linestyle = :dash, label = false)
    for m in order
        xs = [x for x in rows if x.method == m]; isempty(xs) && continue
        scatter!(plt, [x.wall for x in xs], [getfield(x, key) for x in xs], yerror = [getfield(x, h) for x in xs],
            label = label[m], color = colors[m], markersize = 5)
    end
    savefig(plt, joinpath(outdir, "$file.png"))
end
println("figures in ", outdir)
