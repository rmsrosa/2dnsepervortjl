# Stochastically forced flow: no deterministic forcing, additive noise on the modes with
# 3 ≤ |κ| ≤ 5, with energy injection rate ε = 0.1, on a 64 × 64 grid, with ν = 0.01.
using NSEPeriodic2D, Printf, Serialization, Statistics
using StochasticDiffEq: EM, SRA1, SOSRA

const N = 64; const ν = 1e-2; const ε = 0.1
grid = Grid(Nx = N)
modes = fourier_modes(grid, 3, 5)
noise = AdditiveNoise(modes; σ = sqrt(2ε / length(modes)))
model = NSEModel(grid; ν, noise)
const η = enstrophy_injection_rate(noise)

# Relative imbalances of the energy and enstrophy budgets over a run, corrected for the change of
# E and Z over the run, with 95% confidence half-widths from batch means of the dissipation.
function balances(r; nb = 10)
    T = r.t[end] - r.t[1]
    bZ, hZ = batch_mean_ci(2ν .* r.enstrophy; nbatches = nb)
    bP, hP = batch_mean_ci(2ν .* r.palinstrophy; nbatches = nb)
    dE = (r.energy[end] - r.energy[1]) / T
    dZ = (r.enstrophy[end] - r.enstrophy[1]) / T
    return (bZ + dE) / ε - 1, hZ / ε, (bP + dZ) / η - 1, hP / η
end
