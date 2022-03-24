# Pseudo-spectral simulation of fully-periodic two-dimensional incompressible Navier-Stokes flows in Julia

[docs-dev-img]: https://img.shields.io/badge/docs-dev-green.svg
[docs-dev-url]: https://rmsrosa.github.io/2dnsepervortjl/


[![][docs-dev-img]][docs-dev-url] [![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE) ![GitHub repo size](https://img.shields.io/github/repo-size/rmsrosa/2dnsepervortjl) ![Workflow Status](https://github.com/rmsrosa/2dnsepervortjl/actions/workflows/Deploy.yml/badge.svg)

Simulation of homogeneous, incompressible Newtonian viscous flows in a fully-periodic two-dimensional domain under a given steady volume force, via [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) and [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).

## Choice of github actions

There are two workflows ready to be used, one that rebuilds the whole site during the github action (`.github/Deploy_build.yml`) and the other that simply uses the site build locally (`.github/Deploy_nobuild.yml`):

* If you want the github workflow to (re)build the whole site when deploying it, you should copy `.github/Deploy_build.yml` to `.github/workflows/Deploy.yml`, so it runs when new commits are pushed to `main`. It is also recommended to have `__site/` included in the `.gitignore` file, so it is not duplicated in the repo.

* If, however, you don't want to rebuild the site and simply use the site that was built locally, then you should copy `.github/Deploy_nobuild.yml` to `.github/workflows/Deploy.yml`, instead. This action simply moves the contents of `__site/` in the main branch to the root of the `gh-pages` branch, to serve the website. In this case, you should **not** have `__site/` included in the `.gitignore` file.
