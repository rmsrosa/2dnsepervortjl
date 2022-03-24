@def title = "Testing direct and inverse FFTW"

# {{ get_title }}

Here we test the methods, in FFTW, to be used in our pseudo-spectral code. It is mostly a sanity check.

We start by loading the packages we need:

````julia
using FFTW
using WGLMakie
using Test
````

## The spatial domain and its discretization

We consider a square domain of sides $L = 2\pi$, for which the smallest wavenumber is $\kappa_0 = 2\pi/L$.

````julia
L = 2π
κ₀ = 2π/L
nothing
````

We set the number $N$ of points for the mesh in each direction, yielding a mesh $(x_i, y_j)_{i, j = 1,\ldots ,N}$, with $x_N = y_N = L$, and steps $x_{i+1} - x_i = L/N$, and $y_{j+1} - y_j = L/N$. Due to the periodicity, we don't need to store the values corresponding to $i = j = 0$.

````julia
N = 128
x = y = (L/N):(L/N):L
nothing
````

We may visualize the grid with a scatter plot, although if the mesh is too thin, we won't quite see the details. If using GLMakie or WGLMakie, one can zoom in for a detailed view.

````julia
fig, ax, plt = scatter(vec(x .* one.(y)'), vec(one.(x) .* y'))

fig
````

~~~
<div data-jscall-id="2"><script data-jscall-id="3" type="text/javascript">
    function register_resize_handler(remote_origin) {
        function resize_callback(event) {
            if (event.origin !== remote_origin) {
                return;
            }
            const uuid = event.data[0];
            const width = event.data[1];
            const height = event.data[2];
            const iframe = document.getElementById('8297ebbb-f3ea-4cfb-b5f9-14d97240a8ac');
            if (iframe) {
                iframe.style.width = width + "px";
                iframe.style.height = height + "px";
            }
        }
        if (window.addEventListener) {
            window.addEventListener("message", resize_callback, false);
        } else if (window.attachEvent) {
            window.attachEvent("onmessage", resize_callback);
        }
    }
    register_resize_handler('http://127.0.0.1:9284')

</script><iframe scrolling="no" id="8297ebbb-f3ea-4cfb-b5f9-14d97240a8ac" data-jscall-id="1" src="http://127.0.0.1:9284/8297ebbb-f3ea-4cfb-b5f9-14d97240a8ac" style="position: relative; display: block; width: 100%; height: 100%; padding: 0; overflow: hidden; border: none"></iframe></div>

~~~

## A vorticity function for testing

In order to test the methods from [FFTW.jl](http://www.fftw.org), we define a certain vorticity function and its derivatives

````julia
vort = sin.(one.(y) * x') .* cos.(3y * one.(x)')
dx_vort = cos.(one.(y) * x') .* cos.(3y * one.(x)')
dy_vort = - 3 * sin.(one.(y) * x') .* sin.(3y * one.(x)')
dd_vort = - sin.(one.(y) * x') .* cos.(3y * one.(x)') - 9 * sin.(one.(y) * x') .* cos.(3y * one.(x)')

nothing
````

We may visualize the vorticity as a surface:

````julia
fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98))

ax = Axis3(fig[1, 1], xlabel = "x", ylabel = "y", zlabel = "vorticity level", title = "Vorticity graph")

surface!(ax, x, y, vort, colormap = :berlin)

fig
````

~~~
<div data-jscall-id="5"><script data-jscall-id="6" type="text/javascript">
    function register_resize_handler(remote_origin) {
        function resize_callback(event) {
            if (event.origin !== remote_origin) {
                return;
            }
            const uuid = event.data[0];
            const width = event.data[1];
            const height = event.data[2];
            const iframe = document.getElementById('6bc3c87b-25f9-4b19-8ea2-33c47338eece');
            if (iframe) {
                iframe.style.width = width + "px";
                iframe.style.height = height + "px";
            }
        }
        if (window.addEventListener) {
            window.addEventListener("message", resize_callback, false);
        } else if (window.attachEvent) {
            window.attachEvent("onmessage", resize_callback);
        }
    }
    register_resize_handler('http://127.0.0.1:9284')

</script><iframe scrolling="no" id="6bc3c87b-25f9-4b19-8ea2-33c47338eece" data-jscall-id="4" src="http://127.0.0.1:9284/6bc3c87b-25f9-4b19-8ea2-33c47338eece" style="position: relative; display: block; width: 100%; height: 100%; padding: 0; overflow: hidden; border: none"></iframe></div>

~~~

Or, better yet, as a heatmap:

````julia
fig, ax, plt = heatmap(x, y, vort, colormap = :berlin)

ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Vorticity heatmap")

fig
````

~~~
<div data-jscall-id="8"><script data-jscall-id="9" type="text/javascript">
    function register_resize_handler(remote_origin) {
        function resize_callback(event) {
            if (event.origin !== remote_origin) {
                return;
            }
            const uuid = event.data[0];
            const width = event.data[1];
            const height = event.data[2];
            const iframe = document.getElementById('24d36028-93a3-48fd-8c3a-723743ab44c0');
            if (iframe) {
                iframe.style.width = width + "px";
                iframe.style.height = height + "px";
            }
        }
        if (window.addEventListener) {
            window.addEventListener("message", resize_callback, false);
        } else if (window.attachEvent) {
            window.attachEvent("onmessage", resize_callback);
        }
    }
    register_resize_handler('http://127.0.0.1:9284')

</script><iframe scrolling="no" id="24d36028-93a3-48fd-8c3a-723743ab44c0" data-jscall-id="7" src="http://127.0.0.1:9284/24d36028-93a3-48fd-8c3a-723743ab44c0" style="position: relative; display: block; width: 100%; height: 100%; padding: 0; overflow: hidden; border: none"></iframe></div>

~~~

## Testing the direct and inverse discrete Fourier transform

We check going back and forth with the real fft and inverse real fft:

````julia
@testset "check composition `irfft ∘ rfft`" begin
    vort_hat = rfft(vort)
    vortback = irfft(vort_hat, N)
    @test vort ≈ vortback
end
nothing
````

````
Test Summary:                    | Pass  Total
check composition `irfft ∘ rfft` |    1      1

````

We can also visualize the excited modes:

````julia
κ_x = κ₀ * (0:1/N:1/2)
κ_y = κ₀ * (0:1/N:1-1/N)

vort_hat = rfft(vort)

fig, ax, plt = heatmap(κ_x, κ_y, abs.(vort_hat).^2, colormap = :berlin)

ax = Axis(fig[1, 1], xlabel = "k_x", ylabel = "k_y", title = "Enstrophy spectrum heatmap")

fig
````

~~~
<div data-jscall-id="11"><script data-jscall-id="12" type="text/javascript">
    function register_resize_handler(remote_origin) {
        function resize_callback(event) {
            if (event.origin !== remote_origin) {
                return;
            }
            const uuid = event.data[0];
            const width = event.data[1];
            const height = event.data[2];
            const iframe = document.getElementById('f67f7111-40d0-46b2-a96c-9a55f7aec38b');
            if (iframe) {
                iframe.style.width = width + "px";
                iframe.style.height = height + "px";
            }
        }
        if (window.addEventListener) {
            window.addEventListener("message", resize_callback, false);
        } else if (window.attachEvent) {
            window.attachEvent("onmessage", resize_callback);
        }
    }
    register_resize_handler('http://127.0.0.1:9284')

</script><iframe scrolling="no" id="f67f7111-40d0-46b2-a96c-9a55f7aec38b" data-jscall-id="10" src="http://127.0.0.1:9284/f67f7111-40d0-46b2-a96c-9a55f7aec38b" style="position: relative; display: block; width: 100%; height: 100%; padding: 0; overflow: hidden; border: none"></iframe></div>

~~~

We have excited modes $(\pm 1, \pm 3)$. Due to the reality condition of the vorticity, the Fourier spectrum has an Hermitian symmetry around the origin. Using the real Fourier transform, only half of the modes need to be stored. In this case, only modes $(1, \pm 3)$ are retained. Moreover, the negative modes are shifted above. Due to the one-indexing of Julia, this means that modes $(\pm a, \pm b)$ are represented by indices $[a + 1, b + 1]$ and $[a + 1, N + 1 - b]$. In our case, the excited modes are associated with

````julia
vort_hat[4, 2]
````

````
799.089958978059 - 4017.2965085316328im
````

and

````julia
vort_hat[4, 128]
````

````
-401.4782067898798 + 4076.2766404493186im
````

These are the only excited modes, as we can check:

````julia
for i in 1:div(N, 2) + 1, j in 1:N
    if abs(vort_hat[i, j])^2 > eps()
        println("vort_hat[$i, $j] = $(vort_hat[i, j])")
    end
end
````

````
vort_hat[4, 2] = 799.089958978059 - 4017.2965085316328im
vort_hat[4, 128] = -401.4782067898798 + 4076.2766404493186im

````

## Differential operators

In order to check the derivatives in spectral space, we define the following operators. Actually, they are just vectors, since derivatives in spectral space act as diagonal operators. Hence, a straighforward Hadamard product suffices.

````julia
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
````

## Testing the spectral operators

Now we are ready to test the differentiation in spectral space, with the operators just defined.

````julia
@testset "Check operators" begin
    vort_hat = rfft(vort)
    u_hat = Hu_hat .* vort_hat
    v_hat = Hv_hat .* vort_hat
    # check derivative dx in spectral space:
    @test irfft(Dx_hat .* vort_hat, N) ≈ dx_vort
    # check derivative dy in spectral space:
    @test irfft(Dy_hat .* vort_hat, N) ≈ dy_vort
    # check Laplacian in spectral space:
    @test Delta_hat ≈ Dx_hat.^2 .+ Dy_hat.^2
    @test irfft(Delta_hat .* vort_hat, N) ≈ dd_vort
    # check recovering velocity field from vorticity
    @test Dx_hat .* v_hat - Dy_hat .* u_hat ≈ vort_hat
end
nothing
````

````
Test Summary:   | Pass  Total
Check operators |    5      5

````

## One-mode steady state

For forced-periodic flows, when the forcing function contains a single Fourier mode, there is a corresponding steady state vorticity function corresponding also to that single-mode in Fourier space, only with a different amplitude.

In order to construct such pairs of forcing mode / steady state, we set the viscosity of the flow, choose the mode to be forced and define the strength of that force:

````julia
ν = 1.0e-0 # viscosity
κ = (x = 1, y = 2) # forced mode
α = (re = 0.1, im = 0.05) # strength

nothing
````

With these parameters, we find the curl `g_steady` of the forcing term and the vorticity `vort_steady` of the corresponding steady-state:

````julia
g_steady = ν * (
    2α.re * κ₀^4 * sum(abs2, κ)^2 * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - 2α.im * κ₀^4 * sum(abs2, κ)^2 * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)

vort_steady = 2κ₀^2 * sum(abs2, κ) * (
    α.re * cos.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
    - α.im * sin.(κ₀ * (κ.x * one.(y) * x' + κ.y * y * one.(x)'))
)
nothing
````

Here are some visualizations of the curl of the forcing term and of the vorticity

````julia
heatmap(x, y, g_steady, xlabel="x", ylabel="y", title="Curl of the forcing term", titlefont=12)

heatmap(x, y, vort_steady, xlabel="x", ylabel="y", title="Steady state vorticity", titlefont=12)

nothing
````

The discrete Fourier transform of this vector fields are given as

````julia
g_steady_hat = rfft(g_steady)

vort_steady_hat = rfft(vort_steady)

nothing
````

## Testing the steady state

We are now ready to test the steadyness of this vorticity field:

````julia
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
    # Vanishing bilinear term on one-mode steady state
    @test maximum(abs.(Dx_hat .* wu_steady_hat .+ Dy_hat .* wv_steady_hat)) ≤ √eps()
    # Vanishing RHS on steady state
    @test maximum(abs.(rhs_steady)) ≤ √eps()
    # Vanishing linear Stokes on steady state
    @test maximum(abs.(g_steady_hat .+ Delta_hat .* vort_steady_hat)) ≤ √eps()

    # Steady state equation
    @test g_steady_hat ≈ - Delta_hat .* vort_steady_hat ≈ - Delta_hat .* vort_steady_hat .- Dx_hat .* wu_steady_hat .- Dy_hat .* wv_steady_hat

    # Steady state solution
    @test vort_steady_sol_hat ≈ vort_steady_hat
end
nothing
````

````
Test Summary:                         | Pass  Total
Check single-mode stable steady state |    5      5

````

All seems good.

