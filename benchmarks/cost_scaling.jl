# Cost of one evaluation of the nonlinear term and of one StochasticETDRK4 step for several grid
# sizes, to estimate the cost of a simulation on a given machine. Run from the repository root:
#     julia --project=. benchmarks/cost_scaling.jl [threads] [Nmax]
using NSEPeriodic2D, Printf, Random
threads = parse(Int, get(ARGS, 1, "1"))
Nmax = parse(Int, get(ARGS, 2, "1024"))
println("Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", FFTW threads: ", threads)
@printf("%6s %14s %16s %18s\n", "N", "nonlinear (ms)", "ETDRK4 step (ms)", "ns / (N² log₂N)")
for N in filter(≤(Nmax), [64, 128, 256, 512, 1024, 2048, 4096])
    grid = Grid(Nx = N)
    model = NSEModel(grid; ν = 1e-3, fftw_threads = threads)
    w = vec(random_vorticity(grid, 1, N ÷ 8; rng = Xoshiro(1)))
    dw = similar(w)
    NSEPeriodic2D.nonlinear_vec!(dw, w, model, 0.0)
    n = max(3, round(Int, 2e7 / (N^2 * log2(N))))
    t = @elapsed for _ in 1:n
        NSEPeriodic2D.nonlinear_vec!(dw, w, model, 0.0)
    end
    t /= n
    s = StochasticETDRK4()
    c = NSEPeriodic2D._cache(s, model, 1e-3)
    NSEPeriodic2D._step!(w, c, s, model, 0.0, 1e-3, Xoshiro(1))
    ts = @elapsed for _ in 1:n
        NSEPeriodic2D._step!(w, c, s, model, 0.0, 1e-3, Xoshiro(1))
    end
    ts /= n
    @printf("%6d %14.3f %16.3f %18.2f\n", N, 1e3t, 1e3ts, 1e9t / (N^2 * log2(N)))
end
