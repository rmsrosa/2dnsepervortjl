# # Tests for FFTW plans

# Here we test planning the FFTW and run some simple benchmarks.

# Here are the packages we need.

using FFTW
using Plots
using LinearAlgebra: mul!
using Test
using Random
using BenchmarkTools

@info "Threads: $(FFTW.nthreads())"

# ## Testing setup

# ### The spatial domain and discretization

L = 2π
κ₀ = 2π/L
N = 144 # 2^4 * 3^2
x = y = (L/N):(L/N):L
nothing

# ### Vorticity field for the tests

# We randomly excite a certain number of modes.
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
nothing

# Visualizing the vorticity field

heatmap(x, y, vort, xlabel="x", ylabel="y", title="Vorticity field", titlefont=12)

# ### Planning the FFTWs

#
plan_estimate = plan_rfft(vort); # default is FFTW.ESTIMATE
plan_inv_estimate = plan_irfft(vort_hat, N);
plan_measure = plan_rfft(vort, flags=FFTW.MEASURE);
plan_inv_measure = plan_irfft(vort_hat, N, flags=FFTW.MEASURE);
plan_patient = plan_rfft(vort, flags=FFTW.PATIENT);
plan_inv_patient = plan_irfft(vort_hat, N, flags=FFTW.PATIENT);
plan_exhaustive = plan_rfft(vort, flags=FFTW.EXHAUSTIVE);
plan_inv_exhaustive = plan_irfft(vort_hat, N, flags=FFTW.EXHAUSTIVE);

# ## Sanity tests of the different plans

@testset "Planned FFTW with ESTIMATE" begin
    w_hat = plan_estimate * vort
    w_hat_org = copy(w_hat)
    @test w_hat ≈ rfft(vort)
    @test plan_inv_estimate * w_hat ≈ vort
    @test plan_inv_estimate * w_hat ≈ irfft(rfft(vort), N)
    w_hat_mul = similar(w_hat)
    mul!(w_hat_mul, plan_estimate, vort)
    @test w_hat ≈ w_hat_mul
    vort_back = similar(vort)
    mul!(vort_back, plan_inv_estimate, w_hat) # careful, inverse with mul! may mutate w_hat as well
    @test vort_back ≈ vort
end
nothing

#

@testset "Planned FFTW with MEASURE" begin
    w_hat = plan_measure * vort
    @test w_hat ≈ rfft(vort)
    @test plan_inv_measure * w_hat ≈ vort
    @test plan_inv_measure * w_hat ≈ irfft(rfft(vort), N)
    w_hat_back = similar(w_hat)
    mul!(w_hat_back, plan_measure, vort)
    @test w_hat ≈ w_hat_back
    vort_back = similar(vort)
    mul!(vort_back, plan_inv_measure, w_hat)  # careful, inverse with mul! may mutate w_hat as well
    @test vort_back ≈ vort
end
nothing

#

@testset "Planned FFTW with PATIENT" begin
    w_hat = plan_patient * vort
    @test w_hat ≈ rfft(vort)
    @test plan_inv_patient * w_hat ≈ vort
    @test plan_inv_patient * w_hat ≈ irfft(rfft(vort), N)
    w_hat_back = similar(w_hat)
    mul!(w_hat_back, plan_patient, vort)
    @test w_hat ≈ w_hat_back
    vort_back = similar(vort)
    mul!(vort_back, plan_inv_patient, w_hat)  # careful, inverse with mul! may mutate w_hat as well
    @test vort_back ≈ vort
end
nothing

#

@testset "Planned FFTW with EXHAUSTIVE" begin
    w_hat = plan_exhaustive * vort
    @test w_hat ≈ rfft(vort)
    @test plan_inv_exhaustive * w_hat ≈ vort
    @test plan_inv_exhaustive * w_hat ≈ irfft(rfft(vort), N)
    w_hat_back = similar(w_hat)
    mul!(w_hat_back, plan_exhaustive, vort)
    @test w_hat ≈ w_hat_back
    vort_back = similar(vort)
    mul!(vort_back, plan_inv_exhaustive, w_hat) # careful, inverse with mul! may mutate w_hat as well
    @test vort_back ≈ vort
end
nothing

# ## Timings and allocations

# Let us look at it with `@btime`. **Notice that `mul!` does not allocate and is slightly faster.**

@info "FFTW no plan"
@btime rfft(w) setup = (w = copy(vort));

@info "FFTW plan with ESTIMATE"
@btime p * w setup = (p = $plan_estimate; w = copy($vort));
@info "FFTW plan with ESTIMATE and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = copy($vort_hat);
    p = $plan_estimate;
    w = copy($vort)
);

@info "FFTW plan with MEASURE"
@btime p * w setup = (p = $plan_measure; w = copy($vort));
@info "FFTW plan with MEASURE and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = copy($vort_hat);
    p = $plan_measure;
    w = copy($vort)
);

@info "FFTW plan with PATIENT"
@btime p * w setup = (p = $plan_patient; w = copy($vort));
@info "FFTW plan with PATIENT and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = copy($vort_hat);
    p = $plan_patient;
    w = copy($vort)
);

@info "FFTW plan with EXHAUSTIVE"
@btime p * w setup = (p = $plan_exhaustive; w = copy($vort));
@info "FFTW plan with EXHAUSTIVE and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = copy($vort_hat);
    p = $plan_exhaustive;
    w = copy($vort)
);

# ## Direct and inverse transforms

# Here we compare the timings between direct and inverse transforms. We restrict this to using `mul!`, since this is what we will use at the end. **Notice the direct transform is slightly faster than the inverse transform.**

# Only one note of **warning:** when using `mul!` with an inverse transform plan, the last argument may mutate. Type `@edit mul!(vort, plan_inv_estimate.p, vort_hat)` to see where this happens.

@info "Direct and inverse FFTW plan with ESTIMATE and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = similar($vort_hat);
    p = $plan_estimate;
    w = copy($vort)
);
@btime mul!(w, p, w_hat) setup = (
    w = similar($vort);
    p = $plan_inv_estimate;
    w_hat = copy($vort_hat)
);

@info "Direct and inverse FFTW plan with MEASURE and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = similar($vort_hat);
    p = $plan_measure;
    w = copy($vort)
);
@btime mul!(w, p, w_hat) setup = (
    w = similar($vort);
    p = $plan_inv_measure;
    w_hat = copy($vort_hat)
);

@info "Direct and inverse FFTW plan with PATIENT and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = similar($vort_hat);
    p = $plan_patient;
    w = copy($vort)
);
@btime mul!(w, p, w_hat) setup = (
    w = similar($vort);
    p = $plan_inv_patient;
    w_hat = copy($vort_hat)
);

@info "Direct and inverse FFTW plan with EXHAUSTIVE and mul!"
@btime mul!(w_hat, p, w) setup = (
    w_hat = similar($vort_hat);
    p = $plan_exhaustive;
    w = copy($vort)
);
@btime mul!(w, p, w_hat) setup = (
    w = similar($vort);
    p = $plan_inv_exhaustive;
    w_hat = copy($vort_hat)
);
