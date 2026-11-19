"""
    Grid(; Lx = 2π, Ly = Lx, Nx, Ny = Nx, dealias = 2/3)

Uniform grid on the periodic domain [0, Lx) × [0, Ly), with `Nx × Ny` points.

Physical fields are arrays of size `(Nx, Ny)`, indexed as `f[ix, iy]` ↔ `f(x[ix], y[iy])`.
Spectral fields are the real Fourier transforms of those, `rfft(f)`, of size `(Mx, Ny)`, with
`Mx = Nx ÷ 2 + 1` (only the nonnegative wavenumbers in `x` are stored).

Modes with integer wavenumber indices `|jx| > dealias * Nx / 2` or `|jy| > dealias * Ny / 2`
are set to zero (the 2/3 rule, by default), as are the Nyquist modes. Use `dealias = 1` to keep
all modes but the Nyquist ones (not recommended for nonlinear simulations).

Fields:
* `Lx, Ly, Nx, Ny, Mx`;
* `x, y`: grid coordinates, starting at zero;
* `κx` (length `Mx`) and `κy` (length `Ny`): physical wavenumbers, `2π jx / Lx` and `2π jy / Ly`;
* `κ2`: matrix of `κx² + κy²`, of size `(Mx, Ny)`;
* `mask`: `BitMatrix` of the retained modes;
* `κx_max, κy_max`: largest retained physical wavenumbers.
"""
struct Grid
    Lx::Float64
    Ly::Float64
    Nx::Int
    Ny::Int
    Mx::Int
    x::Vector{Float64}
    y::Vector{Float64}
    κx::Vector{Float64}
    κy::Vector{Float64}
    κ2::Matrix{Float64}
    mask::BitMatrix
    κx_max::Float64
    κy_max::Float64
end

function Grid(; Lx::Real = 2π, Ly::Real = Lx, Nx::Integer, Ny::Integer = Nx, dealias::Real = 2/3)
    Nx ≥ 4 && Ny ≥ 4 || throw(ArgumentError("Nx and Ny must be at least 4"))
    0 < dealias ≤ 1 || throw(ArgumentError("dealias must be in (0, 1]"))
    Mx = Nx ÷ 2 + 1
    jx = collect(0:Mx-1)
    jy = [j < (Ny + 1) ÷ 2 ? j : j - Ny for j in 0:Ny-1]
    κx = 2π / Lx .* jx
    κy = 2π / Ly .* jy
    κ2 = [a^2 + b^2 for a in κx, b in κy]
    Kx = floor(Int, dealias * Nx / 2)
    Ky = floor(Int, dealias * Ny / 2)
    nyqx = iseven(Nx) ? Nx ÷ 2 : typemax(Int)
    nyqy = iseven(Ny) ? Ny ÷ 2 : typemax(Int)
    mask = BitMatrix([abs(a) ≤ Kx && abs(b) ≤ Ky && abs(a) != nyqx && abs(b) != nyqy for a in jx, b in jy])
    x = collect((0:Nx-1) .* (Lx / Nx))
    y = collect((0:Ny-1) .* (Ly / Ny))
    κx_max = 2π / Lx * min(Kx, iseven(Nx) ? Nx ÷ 2 - 1 : Nx ÷ 2)
    κy_max = 2π / Ly * min(Ky, iseven(Ny) ? Ny ÷ 2 - 1 : Ny ÷ 2)
    return Grid(Float64(Lx), Float64(Ly), Nx, Ny, Mx, x, y, κx, κy, κ2, mask, κx_max, κy_max)
end

Base.show(io::IO, g::Grid) = print(io, "Grid(Lx = $(g.Lx), Ly = $(g.Ly), Nx = $(g.Nx), Ny = $(g.Ny), ",
    "retained |κx| ≤ $(round(g.κx_max, sigdigits = 4)), |κy| ≤ $(round(g.κy_max, sigdigits = 4)))")

"Applies the dealiasing mask of `grid` to the spectral field `w_hat`, in place."
dealias!(w_hat::AbstractMatrix, grid::Grid) = (w_hat .*= grid.mask; w_hat)

# Weights for sums over the half spectrum: the modes with 0 < jx < Nx/2 stand for two modes
# each (jx and -jx); also includes the normalization of the unnormalized `rfft`, so that
# sum(weights .* abs2.(f_hat)) is the spatial mean of f².
function spectral_weights(grid::Grid)
    w = [(i == 1 || (iseven(grid.Nx) && i == grid.Mx)) ? 1.0 : 2.0 for i in 1:grid.Mx, j in 1:grid.Ny]
    return w ./ (Float64(grid.Nx) * grid.Ny)^2
end
