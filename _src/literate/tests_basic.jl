# # Testing FFTW for fully periodic fluid flows

# Here we test the methods, in FFTW, to be used in the pseudo-spectral code described previously.

# Here are the packages we are gonna need.

using FFTW
using Plots
using Test

# ## The spatial domain

L = 2π
κ₀ = 2π/L
N = 128
x = y = (L/N):(L/N):L

# ## Direct and inverse discrete Fourier transform

vort = sin.(one.(y) * x') .* cos.(3y * one.(x)')
dx_vort = cos.(one.(y) * x') .* cos.(3y * one.(x)')
dy_vort = - 3 * sin.(one.(y) * x') .* sin.(3y * one.(x)')
dd_vort = - sin.(one.(y) * x') .* cos.(3y * one.(x)') - 9 * sin.(one.(y) * x') .* cos.(3y * one.(x)')

display(surface(x, y, vort, xlabel="x", ylabel="y", zlabel="vorticity"))

display(heatmap(x, y, vort, xlabel="x", ylabel="y"))

@testset "check composition `irfft ∘ rfft`" begin
    vort_hat = rfft(vort)
    vortback = irfft(vort_hat, N)
    @test vort ≈ vortback
end

# ## Operators

Dx_hat = im * κ₀ * [ifelse(k1 ≤ N/2 + 1, k1 - 1, k1 - 1 - N) for k2 in 1:N/2+1, k1 in 1:N]
Dy_hat = im * κ₀ * [k2 - 1 for k2 in 1:N/2+1, k1 in 1:N]

Delta_hat = - κ₀^2 * [
    ifelse(k1 ≤ N/2 + 1, (k1 - 1)^2 + (k2 - 1)^2, (k1 - 1 - N)^2 + (k2 - 1)^2)
    for k2 in 1:N/2+1, k1 in 1:N
]

Hu_hat = - Dy_hat ./ Delta_hat
Hu_hat[1, 1] = 0.0
Hv_hat = Dx_hat ./ Delta_hat
Hv_hat[1, 1] = 0.0

# Tests

@testset "Check operators" begin
    vort_hat = rfft(vort)
    u_hat = Hu_hat .* vort_hat
    v_hat = Hv_hat .* vort_hat
    @test irfft(Dx_hat .* vort_hat, N) ≈ dx_vort # check derivative dx in spectral space
    @test irfft(Dy_hat .* vort_hat, N) ≈ dy_vort # check derivative dy in spectral space
    @test Delta_hat ≈ Dx_hat.^2 .+ Dy_hat.^2 # check Laplacian in spectral space
    @test irfft(Delta_hat .* vort_hat, N) ≈ dd_vort
    @test Dx_hat .* v_hat - Dy_hat .* u_hat ≈ vort_hat # check recovering velocity field from vorticity
end

# check "steadyness" of one-mode steady state

ν = 1.0e-0 # viscosity
κ = (x = 1, y = 2) # forced mode
α = (re = 0.1, im = 0.05) # strength

vort_steady = 2κ₀^2 * sum(abs2, κ) * (
    α.re * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - α.im * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)
vort_steady_hat = rfft(vort_steady)

display(heatmap(x, y, vort_steady, xlabel="x", ylabel="y", title="steady state vorticity", titlefont=12))

g_steady = ν * (
    2α.re * κ₀^4 * sum(abs2, κ)^2 * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - 2α.im * κ₀^4 * sum(abs2, κ)^2 * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)
g_steady_hat = rfft(g_steady)

display(heatmap(x, y, g_steady, xlabel="x", ylabel="y", title="forcing term", titlefont=12))

@testset "Check single-mode stable steady state" begin
    u_steady_hat = Hu_hat .* vort_steady_hat
    v_steady_hat = Hv_hat .* vort_steady_hat

    u_steady = irfft(u_steady_hat, N)
    v_steady = irfft(v_steady_hat, N)

    vort_steady_back = irfft(vort_steady_hat, N)

    wu_steady_hat = rfft(vort_steady_back .* u_steady)
    wv_steady_hat = rfft(vort_steady_back .* v_steady)

    rhs_steady = g_steady_hat .+ Delta_hat .* vort_steady_hat .- Dx_hat .* wu_steady_hat .- Dy_hat .* wv_steady_hat

    vort_steady_sol_hat = - g_steady_hat ./ Delta_hat
    vort_steady_sol_hat[1, 1] = 0.0im
    @test maximum(abs.(Dx_hat .* wu_steady_hat .+ Dy_hat .* wv_steady_hat)) ≤ √eps() # Vanishing bilinear term on one-mode steady state
    @test maximum(abs.(rhs_steady)) ≤ √eps() # Vanishing RHS on steady state
    @test maximum(abs.(g_steady_hat .+ Delta_hat .* vort_steady_hat)) ≤ √eps() # Vanishing linear Stokes on steady state

    @test g_steady_hat ≈ - Delta_hat .* vort_steady_hat ≈ - Delta_hat .* vort_steady_hat .- Dx_hat .* wu_steady_hat .- Dy_hat .* wv_steady_hat # Steady state

    @test vort_steady_sol_hat ≈ vort_steady_hat # Steady state solution
end
