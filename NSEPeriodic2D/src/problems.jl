"""
    NSEProblem(model, ω0, tspan; kind = :full)

The initial-value problem for the vorticity equation of `model`, with initial vorticity `ω0`
(in any form accepted by [`vorticity_hat`](@ref)) on the time interval `tspan`, wrapping a
problem of DifferentialEquations.jl. Solve it with `solve(prob, alg; kwargs...)` or
[`simulate`](@ref), with a solver `alg` from the OrdinaryDiffEq packages. The `kind` selects
the formulation, which must match the solver:

* `:full` (default): `ODEProblem` for the whole right-hand side, with the vectorized complex
  half-spectrum as state. For explicit methods, adaptive (`Tsit5()`, `Vern9()`, ...) or with a
  fixed step (`RK4()`, `SSPRK33()`, ... with `dt` and `adaptive = false`). This is the choice
  for advection-dominated (e.g. turbulent) flows.
* `:split`: `SplitODEProblem` with the viscous term `νΔ` as a `DiagonalOperator` and the rest
  as the nonlinear part. For exponential integrators (`ETDRK4()`, with a fixed `dt`) and IMEX
  methods (e.g. `CNAB2(linsolve = DiagonalFactorization())`; without the diagonal
  factorization, the default linear solver is about 1000 times slower). This is the choice
  when the viscous term is stiff.
* `:implicit`: `ODEProblem` with the real reinterpretation of the state and the analytic
  Jacobian-vector product (`jvp`), for implicit methods used matrix-free, e.g.
  `QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false)` (with `KrylovJL_GMRES` from
  LinearSolve). The Jacobian is dense, so it should not be assembled except on small grids.
  For viscous flows converging to steady states.

The state of the solution can be converted back to the spectral vorticity with
[`state_to_vorticity_hat`](@ref).
"""
struct NSEProblem{M <: NSEModel, P}
    model::M
    prob::P
    kind::Symbol
end

function NSEProblem(model::NSEModel, ω0, tspan; kind::Symbol = :full)
    w0 = vec(copy(vorticity_hat(model.grid, ω0)))
    tspan = (Float64(tspan[1]), Float64(tspan[2]))
    prob = if kind === :full
        ODEProblem(rhs_vec!, w0, tspan, model)
    elseif kind === :split
        SplitODEProblem(DiagonalOperator(model.L), nonlinear_vec!, w0, tspan, model)
    elseif kind === :implicit
        ODEProblem(ODEFunction(rhs_vec!; jvp = jvp_vec!), collect(reinterpret(Float64, w0)), tspan, model)
    else
        throw(ArgumentError("kind must be :full, :split or :implicit, got :$kind"))
    end
    return NSEProblem(model, prob, kind)
end

Base.show(io::IO, p::NSEProblem) = print(io, "NSEProblem(:$(p.kind), tspan = $(p.prob.tspan), ", p.model, ")")

CommonSolve.solve(p::NSEProblem, args...; kwargs...) = solve(p.prob, args...; kwargs...)
CommonSolve.init(p::NSEProblem, args...; kwargs...) = init(p.prob, args...; kwargs...)

"""
    remake_problem(p::NSEProblem; ω0 = nothing, tspan = nothing)

A copy of `p` with a new initial vorticity and/or time interval, sharing the model.
"""
function remake_problem(p::NSEProblem; ω0 = nothing, tspan = nothing)
    ω0 = ω0 === nothing ? state_to_vorticity_hat(p, p.prob.u0) : ω0
    tspan = tspan === nothing ? p.prob.tspan : tspan
    return NSEProblem(p.model, ω0, tspan; kind = p.kind)
end

"""
    state_to_vorticity_hat(p::NSEProblem, u)

The spectral vorticity, of size `(Mx, Ny)`, corresponding to the state `u` of a solution of
`p` (e.g. `sol.u[end]` or `integrator.u`). Shares memory with `u`.
"""
state_to_vorticity_hat(p::NSEProblem, u) = _mat(p.model, u)
