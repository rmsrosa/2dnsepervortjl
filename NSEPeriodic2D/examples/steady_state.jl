# Large viscosity and single-mode forcing: the flow converges to the steady state
# ω∞ = (1/ν)A⁻¹g. The viscous term is stiff, so we use ETDRK4 on the split problem, and an
# implicit method with the analytic Jacobian-vector product.
#
# Run from the folder `examples/`:  julia --project=. steady_state.jl
using NSEPeriodic2D
using OrdinaryDiffEqExponentialRK: ETDRK4
using OrdinaryDiffEqBDF: QNDF
using LinearSolve: KrylovJL_GMRES
using Random: Xoshiro

grid = Grid(Lx = 2π, Nx = 128)
model = NSEModel(grid; ν = 1.0, forcing = (x, y) -> 50cos(2x + 3y))
ω∞ = stokes_state(model)
ω0 = random_vorticity(grid, 1, 10; energy = 10.0, rng = Xoshiro(1))

# Exponential integrator with a fixed step
prob = NSEProblem(model, ω0, (0.0, 10.0); kind = :split)
res = simulate(prob, ETDRK4(); dt = 1e-2, save_every = 0.5)
println(res)
println("ETDRK4: relative distance to the steady state at t = 10: ",
    sqrt(energy(model, res.w_hat_final .- ω∞) / energy(model, ω∞)))

# Implicit method, matrix-free, with the analytic Jacobian-vector product
prob = NSEProblem(model, ω0, (0.0, 10.0); kind = :implicit)
res = simulate(prob, QNDF(linsolve = KrylovJL_GMRES(), concrete_jac = false); save_every = 0.5)
println(res)
println("QNDF:   relative distance to the steady state at t = 10: ",
    sqrt(energy(model, res.w_hat_final .- ω∞) / energy(model, ω∞)))
