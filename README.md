# Pseudo-spectral simulation of fully-periodic two-dimensional incompressible Navier-Stokes flows in Julia

[docs-dev-img]: https://img.shields.io/badge/docs-dev-green.svg
[docs-dev-url]: https://rmsrosa.github.io/2dnsepervortjl/

[![][docs-dev-img]][docs-dev-url] [![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE) ![GitHub repo size](https://img.shields.io/github/repo-size/rmsrosa/2dnsepervortjl) ![Workflow Status](https://github.com/rmsrosa/2dnsepervortjl/actions/workflows/Deploy.yml/badge.svg)

Simulation of homogeneous, incompressible Newtonian viscous flows in a fully-periodic two-dimensional domain under a given steady volume force, via [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) and [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).

## Building the site

The site is built with [Franklin.jl](https://github.com/tlienart/Franklin.jl), and the pages in `src/literate/` are executed with [Literate.jl](https://github.com/fredrikekre/Literate.jl) during the build. It requires Julia 1.13 or later.

The GitHub workflow `.github/workflows/Deploy.yml` builds the whole site, executing all the pages, on every push to `main`, and deploys the contents of `__site/` to the `gh-pages` branch. It also builds the site, without deploying it, on pull requests, so that errors in the pages show up before merging. It can also be triggered by hand, from the Actions tab.

The generated files (`__site/`, `pages/literated/`, `generated/` and `_assets/pages/`) are not tracked in git.

To preview the site locally:

```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()

julia> using Franklin

julia> serve()
```

A page in `src/literate/` is executed again only when its source is newer than the generated markdown file in `pages/literated/`; delete the latter to force it.
