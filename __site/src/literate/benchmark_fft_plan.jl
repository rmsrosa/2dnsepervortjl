# # Benchmarking flags for FFTW plans

# We test planning the FFTW with different flags.

# Here are the packages we need.

using FFTW
using Plots
using LinearAlgebra: mul!
using Test
using BenchmarkTools
using Random
using BenchmarkPlots
using StatsPlots

@info "Threads: $(FFTW.nthreads())"

# ## Preparing the benchmarks

# We prepare a suite of benchmarks, using [BenchmarkTools.BenchmarkGroup()](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/#The-BenchmarkGroup-type).

# The best plan usually depends on the size of the array, which in this case is $N \times N$. In particular, it depens on the factorization of $N$. So, we benchmark different values of $N$.

# We choose the following values:

Ns = [2^6, 3^4, 2 * 3 * 5 * 7, 2^8, 2^2 * 3^2 * 5^2, 2^10, 2^11, 2^2 * 3 * 5^2 * 7]

# We must also prepare some other variables.

# List of flags for planning the FFTs:

flags = ["ESTIMATE", "MEASURE", "PATIENT", "EXHAUSTIVE"]
ext_flags = ["NO PLAN", flags...]
nothing

# Physical space:
L = 2π
κ₀ = 2π/L
nothing

# Excited modes for testing field:
rng = Xoshiro(123)
num_modes = 24
max_mode = 16

kxs = rand(rng, 1:max_mode, num_modes)
kys = rand(rng, 1:max_mode, num_modes)
ars = 10*randn(rng, num_modes)
ais = 10*randn(rng, num_modes)
nothing

# The suite of benchmarks:

suite = BenchmarkGroup()
for flag in ext_flags
    suite[flag] = BenchmarkGroup()
end

plan_stats = Dict{String, Dict{Int, Float64}}(flag => Dict() for flag in flags)

# Now we are ready to prepare the suite of benchmarks. Keep in mind that preparing the suite includes planning the transforms, and plans with a `PATIENT` or `EXAUSTIVE` flag take some time for large $N$.

for N in Ns
    x = y = (L/N):(L/N):L
    vort = sum(
        [
            2κ₀^2 * (kx^2 + ky^2) * (
                ar * cos.(κ₀ * (kx * one.(y) * x' + ky * y * one.(x)'))
                - ai * sin.(κ₀ * (kx * one.(y) * x' + ky * y * one.(x)'))
            )
            for (kx, ky, ar, ai) in zip(kxs, kys, ars, ais)
        ]
    )

    vort_hat = rfft(vort)

    flag = "NO PLAN"
    @info "N = $N; flag: $flag"
    suite[flag][N] = @benchmarkable rfft(w) setup = (w = copy($vort));

    for flag in flags
        @info "N = $N; flag: $flag"
        planed, pstats... = @timed plan_rfft(vort, flags = eval(Meta.parse("FFTW.$flag")))
        plan_stats[flag][N] = pstats.time

        suite[flag][N] = @benchmarkable mul!(w_hat, p, w) setup = (
            w_hat = copy($vort_hat);
            p = $planed;
            w = copy($vort)
        );
    end
end

suite

# Looking at `plan_stats`, we can see the time spent in planning the transforms.

for N in Ns
    @info "N = $N"
    for flag in flags
        @info "$flag: \t$(BenchmarkTools.prettytime(plan_stats[flag][N] * 1.0e+9))"
    end
end

# ## Running the benchmark

# This should take some time as well.

results, stats... = @timed run(suite)
nothing

# Here are the stats of the run:

stats

# Let's take a look at the results. The time shown is the minimum time of the trial runs for each benchmark.

results

# ## Analysis of the benchmark

# We start plotting the minimum and median times of the benchmark trials.

# First with the minimum time for low values of `N`.

plt = plot(
    title="Minimum times for different plans with vector fields of different sizes",
    xlabel = "N",
    ylabel = "time (ns)",
    xticks = Ns[1:4],
    rotation = 90,
    titlefont=10,
    legend=:topleft
)

for flag in ext_flags
    plot!(plt, Ns[1:4], N -> minimum(values(results[flag][N]).times),
    linestyle = :dash,
    markershape = :rect,
    label="$flag"
    )
end

plt

# Now, the minimum time for all values of `N`:

plt = plot(
    title="Minimum times for different plans with vector fields of different sizes",
    xlabel = "N",
    ylabel = "time (ns)",
    xticks = Ns,
    rotation = 90,
    titlefont=10,
    legend=:topleft
)

for flag in ext_flags
    plot!(plt, Ns, N -> minimum(values(results[flag][N]).times),
    linestyle = :dash,
    markershape = :rect,
    label="$flag"
    )
end

plt

# Next with the median time, starting with low values of `N`:

plt = plot(
    title="Median times for different plans with vector fields of different sizes",
    xlabel = "N",
    ylabel = "time (ms)",
    xticks = Ns[1:4],
    rotation = 90,
    titlefont=10,
    legend=:topleft
)

for flag in ext_flags
    plot!(plt, Ns[1:4], N -> median(values(results[flag][N]).times),
    linestyle = :solid,
    markershape = :circle,
    label="$flag"
    )
end
plt

# Now with all values of `N`:

plt = plot(
    title="Median times for different plans with vector fields of different sizes",
    xlabel = "N",
    ylabel = "time (ms)",
    xticks = Ns,
    rotation = 90,
    titlefont=10,
    legend=:topleft
)

for flag in ext_flags
    plot!(plt, Ns, N -> median(values(results[flag][N]).times)./ 1.0e+6,
    linestyle = :solid,
    markershape = :circle,
    label="$flag"
    )
end

plt

# Next we have a look at the set of trials, with violin plots.

plts = []

for N in Ns
    push!(
        plts,
        violin(
            [results[flag][N].times for flag in flags],
            title = "Trials with N = $N",
            titlefont = 12,
            xticks = (1:length(flags), string.(flags)),
            yaxis = "time (ns)",
            legend = nothing
        )
    )
end

if isodd(length(Ns))
    push!(plts, plot(border = :none))
end

plt = plot(plts..., layout = (div(length(plts), 2), 2), size = (800, 1200))

# We may look at the plot recipe build for the results of running a benchmark suite. But need to work on sorting the values of `N`.

# Let us take a final closer look at the median values.

for N in Ns
    @info "N = $N"
    for flag in flags
        @info "median time for flag $flag: $(BenchmarkTools.prettytime(median(values(results[flag][N]).times)))"
    end
end

# **Notice that `N` with powers of 2 are a little better.** The timings for the non-powers-of-2 are skewed up.

# We can have a better look at that by fitting the expected order of complexity. The discrete Fourier transform is of order $K\log(K)$, but this is a two-dimensional problem, so $K = N^2$.

# Let us fit a single plan, say `PATIENT`, with the median times.

flag = "PATIENT"

# We only fit the data for the powers of two.

twos = filter(N -> isinteger(log2(N)), Ns)

# Here is the Vandermonde matrix for the fit.

mat = [ones(length(twos)) [N^2 * log(N^2) for N in twos]]

# We get
a, b = mat \ [median(values(results[flag][N].times)) for N in twos]

# Now we plot the resulting fit agains the data

plt = plot(
    title="Median times for plan $flag with vector fields of different sizes",
    xlabel = "N (stretched out as N²)",
    ylabel = "time (ms)",
    xticks = (Ns.^2, string.(Ns)),
    rotation = 90,
    titlefont=10,
    legend=:topleft
)

plot!(plt, Ns.^2, map(N -> median(values(results[flag][isqrt(N)]).times)./ 1.0e+6, Ns.^2),
linestyle = :solid,
markershape = :circle,
label="$flag"
)

plot!(plt, Ns.^2, K -> (a + b * K * log(K) )./ 1.0e+6, linestyle = :dash, markershape = :square, label="N²log(N²) fit")


plt

# Now we take a closer look at the results for low `N`. Because the values are so discrepant for low and high `Ns` (they scale with $N^2$), instead of zooming into the result for low `N`, we do another fit, for these values. One option is to do a weighted least square fit, but that didn't turn out to be the best, so we we just restrict the fit to the low `N`.

# We look at the powers of two that have low `N`:

lowtwos = filter(N -> isinteger(log2(N)), Ns[1:4])

# And fit them with the Vandermonde matrix

lowmat = [ones(length(lowtwos)) [N^2 * log(N^2) for N in lowtwos]]

# We get
c, d = lowmat \ [median(values(results[flag][N].times)) for N in lowtwos]

# Now we can visualize the result.

plt = plot(
    title="Median times for plan $flag with vector fields of different sizes",
    xlabel = "N (stretched out as N²)",
    ylabel = "time (ms)",
    xticks = (Ns[1:4].^2, string.(Ns[1:4])),
    rotation = 90,
    titlefont=10,
    legend=:topleft
)

plot!(plt, Ns[1:4].^2, map(N -> median(values(results[flag][isqrt(N)]).times)./ 1.0e+6, Ns[1:4].^2),
linestyle = :solid,
markershape = :circle,
label="$flag"
)

plot!(plt, Ns[1:4].^2, K -> (c + d * K * log(K) )./ 1.0e+6, linestyle = :dash, markershape = :square, label="N²log(N²) fit")


plt

# Hmm, notice there are just two powers of two in this low range of `N`, so we could have just draw a straight line joint these two, but we leave it here in this generic way in case we change the values in the vector `Ns` of values of `N`.

# ## Conclusions

# Well, no plan is consistently worse, with all the others being closer together. In general, "PATIENT" and "EXHAUSTIVE" perform better, but planning them is costly, especially "EXHAUSTIVE", which doesn't seem to be worth it.

# Hence, for a quick plan, "MEASURE" seems to be a good choice. Otherwise, "PATIENT" seems like the best option.

# Moreover, powers of single primes, and in particular powers of 2, tend also to be faster, relatively speaking.

# Hence, prefer `N` with powers of 2, plan with `PATIENT`, and apply the plan with `mul!`.
