# Analysis of the chaotic-flow runs: statistics with confidence intervals, CFL numbers, cost,
# figures. Usage: julia analyze.jl [outdir]
include("turb_common.jl")
using Plots
outdir = get(ARGS, 1, "figs"); mkpath(outdir)

evals_per_step = Dict("RK4" => 4, "ETDRK4" => 4, "SSPRK33" => 3, "CNAB2hand" => 1, "CNAB2" => 1)
order = Dict("Tsit5" => 1, "RK4" => 2, "ETDRK4" => 3, "SSPRK33" => 4, "CNAB2hand" => 5)
label(m) = m == "CNAB2hand" ? "CNAB2" : m

runs = [deserialize(joinpath("results", f)) for f in readdir("results") if endswith(f, ".jls")]
sort!(runs, by = r -> (order[r.method], r.method == "Tsit5" ? -r.param : r.param))

rows = []
for r in runs
    stable = !r.blowup && length(r.ts) == round(Int, r.T / 0.5)
    cflt = r.method == "Tsit5" ? NaN : r.dt * r.kmax * median(r.Us)
    cflm = r.method == "Tsit5" ? NaN : r.dt * r.kmax * maximum(r.Us; init = 0.0)
    evals = r.method == "Tsit5" ? r.nf / (isempty(r.ts) ? NaN : r.ts[end] - r.t0) :
        evals_per_step[r.method] / r.dt
    if stable
        bE, bZ, bP = batch_means(r.Es), batch_means(r.Zs), batch_means(r.Ps)
        E = ci(bE); Z = ci(bZ); P = ci(bP); K2 = ci(bP ./ bZ)
    else
        E = Z = P = K2 = (NaN, NaN)
    end
    push!(rows, (; r.method, r.param, r.dt, stable, tend = isempty(r.ts) ? r.t0 : r.ts[end], cflt, cflm,
        evals, E, Z, P, K2, r.spec, r.wall))
end

fmt(x) = isnan(x) ? "–" : @sprintf("%.3g", x)
fmtci(c) = isnan(c[1]) ? "–" : @sprintf("%.3f ± %.3f", c[1], c[2])
io = IOBuffer()
println(io, "| Method | Parameter | dt | Typical CFL | Max CFL | Stable | ⟨E⟩ | ⟨Z⟩ | ⟨P⟩ | κσ² = ⟨P⟩/⟨Z⟩ | Nonlinear evals per time unit |")
println(io, "| :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |")
for x in rows
    par = x.method == "Tsit5" ? @sprintf("reltol = %g", x.param) : @sprintf("σ = %g", x.param)
    st = x.stable ? "yes" : @sprintf("blew up at t = %.0f", x.tend)
    println(io, "| ", label(x.method), " | ", par, " | ", fmt(x.dt), " | ", fmt(x.cflt), " | ", fmt(x.cflm), " | ", st,
        " | ", fmtci(x.E), " | ", fmtci(x.Z), " | ", fmtci(x.P), " | ", fmtci(x.K2), " | ", @sprintf("%.0f", x.evals), " |")
end
table = String(take!(io))
write(joinpath(outdir, "table.md"), table)
println(table)

# Figures
st = [x for x in rows if x.stable]
colors = Dict("Tsit5" => :black, "RK4" => :blue, "ETDRK4" => :green, "SSPRK33" => :orange, "CNAB2hand" => :red)
for (key, name, file) in [(:E, "⟨E⟩", "energy"), (:Z, "⟨Z⟩", "enstrophy"), (:K2, "κσ² = ⟨P⟩/⟨Z⟩", "kappa_sigma")]
    plt = plot(xlabel = "nonlinear evaluations per time unit", ylabel = name, xscale = :log10,
        legend = :outerright, size = (800, 400), title = "$name with 95% confidence intervals", titlefont = 10)
    for m in ["Tsit5", "RK4", "ETDRK4", "SSPRK33", "CNAB2hand"]
        xs = [x for x in st if x.method == m]; isempty(xs) && continue
        scatter!(plt, [x.evals for x in xs], [getfield(x, key)[1] for x in xs],
            yerror = [getfield(x, key)[2] for x in xs], label = label(m), color = colors[m], markersize = 5)
    end
    savefig(plt, joinpath(outdir, "$file.png"))
end

plt = plot(xlabel = "k", ylabel = "E(k)", xscale = :log10, yscale = :log10, legend = :bottomleft,
    size = (800, 500), title = "Time-averaged energy spectra (all stable runs)", titlefont = 10)
for x in st
    plot!(plt, 1:length(x.spec), x.spec, color = colors[x.method], alpha = 0.6,
        label = label(x.method) * (x.method == "Tsit5" ? " reltol=$(x.param)" : " σ=$(x.param)"))
end
savefig(plt, joinpath(outdir, "spectra.png"))

# Warm-up: time series of the energy and enstrophy from the initial condition (1/ν)A⁻¹g
series = deserialize(joinpath("warm_0.004_1.0", "series.jls"))
ts = reduce(vcat, [s.ts for s in series]); Es = reduce(vcat, [s.Es for s in series]); Zs = reduce(vcat, [s.Zs for s in series])
plt = plot(ts, [Es Zs], label = ["E" "Z"], yscale = :log10, xlabel = "t", size = (800, 350),
    title = "Warm-up with Tsit5 from (1/ν)A⁻¹g", titlefont = 10)
vline!(plt, [1200], color = :gray, linestyle = :dash, label = "t₀ = 1200")
savefig(plt, joinpath(outdir, "warmup.png"))
println("figures written to ", outdir)
