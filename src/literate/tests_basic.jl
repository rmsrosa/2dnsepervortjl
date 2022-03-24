# # Testing direct and inverse FFTW

# Here we test the methods, in FFTW, to be used in our pseudo-spectral code. It is mostly a sanity check.

# We start by loading the packages we need:

using FFTW
using WGLMakie
using Test

# ## The spatial domain and its discretization

# We consider a square domain of sides $L = 2\pi$, for which the smallest wavenumber is $\kappa_0 = 2\pi/L$.

L = 2π
κ₀ = 2π/L
nothing

# We set the number $N$ of points for the mesh in each direction, yielding a mesh $(x_i, y_j)_{i, j = 1,\ldots ,N}$, with $x_N = y_N = L$, and steps $x_{i+1} - x_i = L/N$, and $y_{j+1} - y_j = L/N$. Due to the periodicity, we don't need to store the values corresponding to $i = j = 0$.

N = 128
x = y = (L/N):(L/N):L
nothing

# We may visualize the grid with a scatter plot, although if the mesh is too thin, we won't quite see the details. If using GLMakie or WGLMakie, one can zoom in for a detailed view.

fig, ax, plt = scatter(vec(x .* one.(y)'), vec(one.(x) .* y'))

fig

# ## A vorticity function for testing

# In order to test the methods from [FFTW.jl](http://www.fftw.org), we define a certain vorticity function and its derivatives

vort = sin.(one.(y) * x') .* cos.(3y * one.(x)')
dx_vort = cos.(one.(y) * x') .* cos.(3y * one.(x)')
dy_vort = - 3 * sin.(one.(y) * x') .* sin.(3y * one.(x)')
dd_vort = - sin.(one.(y) * x') .* cos.(3y * one.(x)') - 9 * sin.(one.(y) * x') .* cos.(3y * one.(x)')

nothing

# We may visualize the vorticity as a surface:

fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98))

ax = Axis3(fig[1, 1], xlabel = "x", ylabel = "y", zlabel = "vorticity level", title = "Vorticity graph")

surface!(ax, x, y, vort, colormap = :berlin)

fig

# Or, better yet, as a heatmap:

fig, ax, plt = heatmap(x, y, vort, colormap = :berlin)

ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Vorticity heatmap")

fig

# ## Testing the direct and inverse discrete Fourier transform

# We check going back and forth with the real fft and inverse real fft:

@testset "check composition `irfft ∘ rfft`" begin
    vort_hat = rfft(vort)
    vortback = irfft(vort_hat, N)
    @test vort ≈ vortback
end
nothing

# We can also visualize the excited modes:

κ_x = κ₀ * (0:1/N:1/2)
κ_y = κ₀ * (0:1/N:1-1/N)

vort_hat = rfft(vort)

fig, ax, plt = heatmap(κ_x, κ_y, abs.(vort_hat).^2, colormap = :berlin)

ax = Axis(fig[1, 1], xlabel = "k_x", ylabel = "k_y", title = "Enstrophy spectrum heatmap")

fig

# We have excited modes $(\pm 1, \pm 3)$. Due to the reality condition of the vorticity, the Fourier spectrum has an Hermitian symmetry around the origin. Using the real Fourier transform, only half of the modes need to be stored. In this case, only modes $(1, \pm 3)$ are retained. Moreover, the negative modes are shifted above. Due to the one-indexing of Julia, this means that modes $(\pm a, \pm b)$ are represented by indices $[a + 1, b + 1]$ and $[a + 1, N + 1 - b]$. In our case, the excited modes are associated with

vort_hat[4, 2]

# and

vort_hat[4, 128]

# These are the only excited modes, as we can check:

for i in 1:div(N, 2) + 1, j in 1:N
    if abs(vort_hat[i, j])^2 > eps()
        println("vort_hat[$i, $j] = $(vort_hat[i, j])")
    end
end

# ## Differential operators

# In order to check the derivatives in spectral space, we define the following operators. Actually, they are just vectors, since derivatives in spectral space act as diagonal operators. Hence, a straighforward Hadamard product suffices.

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

nothing

# ## Testing the spectral operators

# Now we are ready to test the differentiation in spectral space, with the operators just defined.

@testset "Check operators" begin
    vort_hat = rfft(vort)
    u_hat = Hu_hat .* vort_hat
    v_hat = Hv_hat .* vort_hat
    ## check derivative dx in spectral space:
    @test irfft(Dx_hat .* vort_hat, N) ≈ dx_vort 
    ## check derivative dy in spectral space:
    @test irfft(Dy_hat .* vort_hat, N) ≈ dy_vort 
    ## check Laplacian in spectral space:
    @test Delta_hat ≈ Dx_hat.^2 .+ Dy_hat.^2
    @test irfft(Delta_hat .* vort_hat, N) ≈ dd_vort
    ## check recovering velocity field from vorticity
    @test Dx_hat .* v_hat - Dy_hat .* u_hat ≈ vort_hat 
end
nothing

# ## One-mode steady state

# For forced-periodic flows, when the forcing function contains a single Fourier mode, there is a corresponding steady state vorticity function corresponding also to that single-mode in Fourier space, only with a different amplitude.

# In order to construct such pairs of forcing mode / steady state, we set the viscosity of the flow, choose the mode to be forced and define the strength of that force:

ν = 1.0e-0 # viscosity
κ = (x = 1, y = 2) # forced mode
α = (re = 0.1, im = 0.05) # strength

nothing

# With these parameters, we find the curl `g_steady` of the forcing term and the vorticity `vort_steady` of the corresponding steady-state:

g_steady = ν * (
    2α.re * κ₀^4 * sum(abs2, κ)^2 * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - 2α.im * κ₀^4 * sum(abs2, κ)^2 * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)

vort_steady = 2κ₀^2 * sum(abs2, κ) * (
    α.re * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - α.im * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)
nothing

# Here are some visualizations of the curl of the forcing term and of the vorticity

heatmap(x, y, g_steady, xlabel="x", ylabel="y", title="Curl of the forcing term", titlefont=12)

heatmap(x, y, vort_steady, xlabel="x", ylabel="y", title="Steady state vorticity", titlefont=12)

nothing

# The discrete Fourier transform of this vector fields are given as

g_steady_hat = rfft(g_steady)

vort_steady_hat = rfft(vort_steady)

nothing

# ## Testing the steady state

# We are now ready to test the steadyness of this vorticity field:

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
    ## Vanishing bilinear term on one-mode steady state
    @test maximum(abs.(Dx_hat .* wu_steady_hat .+ Dy_hat .* wv_steady_hat)) ≤ √eps()
    ## Vanishing RHS on steady state
    @test maximum(abs.(rhs_steady)) ≤ √eps()
    ## Vanishing linear Stokes on steady state
    @test maximum(abs.(g_steady_hat .+ Delta_hat .* vort_steady_hat)) ≤ √eps()

    ## Steady state equation
    @test g_steady_hat ≈ - Delta_hat .* vort_steady_hat ≈ - Delta_hat .* vort_steady_hat .- Dx_hat .* wu_steady_hat .- Dy_hat .* wv_steady_hat

    ## Steady state solution
    @test vort_steady_sol_hat ≈ vort_steady_hat
end
nothing

# All seems good.
