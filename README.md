# Pseudo-spectral simulation of fully-periodic two-dimensional incompressible Navier-Stokes flows in Julia

[docs-dev-img]: https://img.shields.io/badge/docs-dev-green.svg
[docs-dev-url]: https://rmsrosa.github.io/2dnsepervortjl/


[![][docs-dev-img]][docs-dev-url] [![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE) ![GitHub repo size](https://img.shields.io/github/repo-size/rmsrosa/2dnsepervortjl) ![Workflow Status](https://github.com/rmsrosa/2dnsepervortjl/actions/workflows/Deploy.yml/badge.svg)

Simulation of homogeneous, incompressible Newtonian viscous flows in a fully-periodic two-dimensional domain under a given steady volume force, via [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) and [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).
