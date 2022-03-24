@def title = "Introduction"

# {{ get_title }}

We want to simulate the flow of a homogeneous, incompressible Newtonian viscous fluid in a fully-periodic two-dimensional domain under a given steady volume force.

The aim is to find steady forces able to sustain turbulence, at least in the sense of having a well-defined enstrophy-cascade range. It is known that a single-mode forcing is never sufficient for that [Constantin, Foias & Manley (1994)](/pages/references/#cfm94) (see also [Foias, Manley, Jolly & Rosa (2002)](/pages/references/#fjmr2002)). We intend to look for a two-mode forcing for that.

We use the vorticity formulation, with a pseudo-spectral discretization of the space variables, and the method of lines for the time integration. We use [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) for the discrete Fourier transforms and [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) for the time evolution.
