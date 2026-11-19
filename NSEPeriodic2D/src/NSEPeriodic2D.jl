"""
    NSEPeriodic2D

Pseudo-spectral simulation of the two-dimensional incompressible Navier-Stokes equations in
vorticity form,

    ∂ₜω + u ⋅ ∇ω = νΔω + g,

on the periodic domain [0, Lx) × [0, Ly), where ω = ∂ₓv - ∂ᵧu is the vorticity of the velocity
field u = (u, v) and g is the curl of the (steady) forcing.

The vorticity is represented by its real Fourier transform (`rfft`), dealiased by truncation,
and the nonlinear term is computed in Basdevant's form, with four FFTs per evaluation.

The package builds the problems (`NSEProblem`) to be solved with the solvers of
DifferentialEquations.jl, which are not a dependency: load the solver package of your choice
(e.g. `OrdinaryDiffEq`, `OrdinaryDiffEqExponentialRK`, ...). See the README for an overview.
"""
module NSEPeriodic2D

using FFTW
using LinearAlgebra: mul!, norm
using Random: AbstractRNG, Xoshiro
using Statistics: mean, std
using SciMLBase: SciMLBase, ODEProblem, SplitODEProblem, ODEFunction, ReturnCode
using SciMLOperators: DiagonalOperator
using CommonSolve: CommonSolve, solve, init, step!

include("grid.jl")
include("fields.jl")
include("model.jl")
include("problems.jl")
include("diagnostics.jl")
include("simulate.jl")

export Grid, NSEModel, NSEProblem, remake_problem
export vorticity_hat, vorticity_from_velocity, forcing_hat
export band_forcing, random_vorticity, stokes_state
export vorticity, velocity, streamfunction, state_to_vorticity_hat
export energy, enstrophy, palinstrophy, kappa_sigma, energy_spectrum, max_velocity, cfl_number, suggest_dt
export simulate, SimulationResult, batch_mean_ci
export solve, init, step!

end
