# # Testing FFTW for fully periodic fluid flows

# Here we test planning the FFTW.

# Here are the packages we are gonna need.

using FFTW
using Plots
using LinearAlgebra: mul!
using Test
using BenchmarkTools
using Random

@info "Threads: $(FFTW.nthreads())"

# ## The spatial domain

L = 2π
κ₀ = 2π/L
N = 256
x = y = (L/N):(L/N):L

# Vorticity for the tests

rng = Xoshiro(123)
num_modes = 8
vort = sum(
    [
        2κ₀^2 * (kx^2 + ky^2) * (
            ar * cos.(κ₀ * (kx * one.(y) * x' + ky * y * one.(x)'))
            - ai * sin.(κ₀ * (kx * one.(y) * x' + ky * y * one.(x)'))
        )
        for (kx, ky, ar, ai) in zip(
            rand(rng, 1:div(N,4), num_modes),
            rand(rng, 1:div(N,4), num_modes),
            10*rand(rng, num_modes),
            10*rand(rng, num_modes)
        )
    ]
)

vort_hat = rfft(vort)

# ## Planning the FFTWs

@time plan_estimate = plan_rfft(vort) # default is FFTW.ESTIMATE
@time plan_inv_estimate = plan_irfft(vort_hat, N)
@time plan_measure = plan_rfft(vort, flags=FFTW.MEASURE)
@time plan_inv_measure = plan_irfft(vort_hat, N, flags=FFTW.MEASURE)
@time plan_patient = plan_rfft(vort, flags=FFTW.PATIENT)
@time plan_inv_patient = plan_irfft(vort_hat, N, flags=FFTW.PATIENT)
@time plan_exhaustive = plan_rfft(vort, flags=FFTW.EXHAUSTIVE)
@time plan_inv_exhaustive = plan_irfft(vort_hat, N, flags=FFTW.EXHAUSTIVE)

@testset "Planned FFTW with ESTIMATE" begin
    v_hat = plan_estimate * vort
    @test v_hat ≈ rfft(vort)
    @test plan_inv_estimate * v_hat ≈ vort
    @test plan_inv_estimate * v_hat ≈ irfft(rfft(vort), N)
    v_hat_bak = copy(v_hat)
    mul!(v_hat, plan_estimate, vort)
    @test v_hat ≈ v_hat_bak
end

@testset "Planned FFTW with MEASURE" begin
    v_hat = plan_measure * vort
    @test v_hat ≈ rfft(vort)
    @test plan_inv_measure * v_hat ≈ vort
    @test plan_inv_measure * v_hat ≈ irfft(rfft(vort), N)
    v_hat_bak = copy(v_hat)
    mul!(v_hat, plan_measure, vort)
    @test v_hat ≈ v_hat_bak
end

@testset "Planned FFTW with PATIENT" begin
    v_hat = plan_patient * vort
    @test v_hat ≈ rfft(vort)
    @test plan_inv_patient * v_hat ≈ vort
    @test plan_inv_patient * v_hat ≈ irfft(rfft(vort), N)
    v_hat_bak = copy(v_hat)
    mul!(v_hat, plan_patient, vort)
    @test v_hat ≈ v_hat_bak
end

@testset "Planned FFTW with EXHAUSTIVE" begin
    v_hat = plan_exhaustive * vort
    @test v_hat ≈ rfft(vort)
    @test plan_inv_exhaustive * v_hat ≈ vort
    @test plan_inv_exhaustive * v_hat ≈ irfft(rfft(vort), N)
    v_hat_bak = copy(v_hat)
    mul!(v_hat, plan_exhaustive, vort)
    @test v_hat ≈ v_hat_bak
end

@info "N = $N"

@info "FFTW no plan"
@btime rfft(v) setup = (v = copy(vort));

@info "FFTW plan with ESTIMATE"
@btime p * v setup = (p = $plan_estimate; v = copy($vort));
@btime mul!(v_hat, p, v) setup = (
    v_hat = copy($vort_hat);
    p = $plan_estimate;
    v = copy($vort)
);

@info "FFTW plan with MEASURE"
@btime p * v setup = (p = $plan_measure; v = copy($vort));
@btime mul!(v_hat, p, v) setup = (
    v_hat = copy($vort_hat);
    p = $plan_measure;
    v = copy($vort)
);

@info "FFTW plan with PATIENT"
@btime p * v setup = (p = $plan_patient; v = copy($vort));
@btime mul!(v_hat, p, v) setup = (
    v_hat = copy($vort_hat);
    p = $plan_patient;
    v = copy($vort)
);

@info "FFTW plan with EXHAUSTIVE"
@btime p * v setup = (p = $plan_exhaustive; v = copy($vort));
@btime mul!(v_hat, p, v) setup = (
    v_hat = copy($vort_hat);
    p = $plan_exhaustive;
    v = copy($vort)
);

# Well, no plan is consistently worse.

# With N not too large, all the others are similar, with ESTIMATE and MEASURE slightly better and with ESTIMATE leading it.

# For N larger, MEASURE got the first place, with ESTIMATE a little behind and with PATIENT and EXHAUSTIVE near no-plan.
#'
# Tested with N = 32, 64, 128, 256, 512, 1024, and 2048.

# In summary, just use the default ESTIMATE for ≤ 128 and MEASURE for ≥ 256

#=
```julia
[ Info: Threads: 8

...

[ Info: N = 64
[ Info: FFTW no plan
  26.375 μs (38 allocations: 35.45 KiB)
[ Info: FFTW plan with ESTIMATE
  4.771 μs (2 allocations: 33.05 KiB)
  4.565 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with MEASURE
  5.347 μs (2 allocations: 33.05 KiB)
  5.125 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with PATIENT
  5.292 μs (2 allocations: 33.05 KiB)
  4.774 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with EXHAUSTIVE
  6.550 μs (2 allocations: 33.05 KiB)
  5.410 μs (0 allocations: 0 bytes)

[ Info: N = 128
[ Info: FFTW no plan
  49.541 μs (38 allocations: 132.45 KiB)
[ Info: FFTW plan with ESTIMATE
  27.209 μs (2 allocations: 130.05 KiB)
  26.625 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with MEASURE
  29.042 μs (2 allocations: 130.05 KiB)
  27.125 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with PATIENT
  29.166 μs (2 allocations: 130.05 KiB)
  27.416 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with EXHAUSTIVE
  29.417 μs (2 allocations: 130.05 KiB)
  27.333 μs (0 allocations: 0 bytes)

[ Info: N = 256
[ Info: FFTW no plan
  167.000 μs (38 allocations: 518.45 KiB)
[ Info: FFTW plan with ESTIMATE
  138.042 μs (2 allocations: 516.05 KiB)
  137.375 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with MEASURE
  119.000 μs (2 allocations: 516.05 KiB)
  118.708 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with PATIENT
  135.542 μs (2 allocations: 516.05 KiB)
  133.875 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with EXHAUSTIVE
  137.125 μs (2 allocations: 516.05 KiB)
  131.500 μs (0 allocations: 0 bytes)

[ Info: N = 512
[ Info: FFTW no plan
  639.416 μs (38 allocations: 2.01 MiB)
[ Info: FFTW plan with ESTIMATE
  583.917 μs (2 allocations: 2.01 MiB)
  582.333 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with MEASURE
  524.583 μs (2 allocations: 2.01 MiB)
  515.458 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with PATIENT
  687.834 μs (2 allocations: 2.01 MiB)
  578.833 μs (0 allocations: 0 bytes)
[ Info: FFTW plan with EXHAUSTIVE
  623.042 μs (2 allocations: 2.01 MiB)
  594.375 μs (0 allocations: 0 bytes)

[ Info: N = 1024
[ Info: FFTW no plan
  4.040 ms (38 allocations: 8.02 MiB)
[ Info: FFTW plan with ESTIMATE
  2.902 ms (2 allocations: 8.02 MiB)
  2.879 ms (0 allocations: 0 bytes)
[ Info: FFTW plan with MEASURE
  2.856 ms (2 allocations: 8.02 MiB)
  2.826 ms (0 allocations: 0 bytes)
[ Info: FFTW plan with PATIENT
  3.399 ms (2 allocations: 8.02 MiB)
  3.169 ms (0 allocations: 0 bytes)
[ Info: FFTW plan with EXHAUSTIVE
  4.037 ms (2 allocations: 8.02 MiB)
  3.560 ms (0 allocations: 0 bytes)
```


=#