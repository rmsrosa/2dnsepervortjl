# Benchmarks

These scripts are not part of the website, since they take long to run and their timings depend on the machine. They are kept here as a reference, to be run separately.

* `benchmark_fft_plan.jl`: benchmarks of the FFTW planning flags (`ESTIMATE`, `MEASURE`, `PATIENT` and `EXHAUSTIVE`) for several grid sizes, up to $N = 2100$. It takes about half an hour, mostly planning with `PATIENT` and `EXHAUSTIVE` for the larger grids.
* `turbulence/`: the comparison of time-stepping methods for a weakly turbulent flow reported in the page "Time stepping for weakly turbulent flows" (warm-up, runs of all the methods, timings and analysis). Run `bash run_all.sh` from that folder; it takes about 1.5 hours on 4 cores.
* `benchmark_stiff_solvers.jl`: the benchmark reported in the page "Stiff solvers, Jacobians and automatic differentiation", comparing explicit, implicit (with the analytic Jacobian-vector product) and exponential (ETDRK4) solvers. It takes about a minute.

They are written in the [Literate.jl](https://github.com/fredrikekre/Literate.jl) format, like the pages of the site, and use the environment of the repository. They can be run as plain scripts (in which case the plots are not shown), from the root of the repository:

```bash
julia --project=. -t auto benchmarks/benchmark_stiff_solvers.jl
```

or, to keep the output and the plots, converted to an executed markdown page or notebook, written to `benchmarks/output/` (which is not tracked by git):

```bash
julia --project=. -t auto -e 'using Literate; Literate.markdown("benchmarks/benchmark_fft_plan.jl", "benchmarks/output"; execute = true, credit = false)'
```

```bash
julia --project=. -t auto -e 'using Literate; Literate.notebook("benchmarks/benchmark_fft_plan.jl", "benchmarks/output"; execute = true, credit = false)'
```
