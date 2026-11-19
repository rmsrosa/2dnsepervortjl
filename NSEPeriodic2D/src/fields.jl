# Conversion of user input into dealiased spectral fields, and helpers to build initial
# conditions and forcing terms.

"""
    vorticity_hat(grid, ω)

Dealiased spectral representation, of size `(Mx, Ny)`, of the scalar field `ω`, given as
* a function `ω(x, y)`;
* a physical field, a real array of size `(Nx, Ny)`;
* a spectral field, a complex array of size `(Mx, Ny)` (copied and dealiased).

The mean (the `(0, 0)` mode) is kept; for the vorticity of a periodic velocity field it should
vanish.
"""
function vorticity_hat(grid::Grid, ω::Function)
    return vorticity_hat(grid, [Float64(ω(x, y)) for x in grid.x, y in grid.y])
end

function vorticity_hat(grid::Grid, ω::AbstractMatrix{<:Real})
    size(ω) == (grid.Nx, grid.Ny) || throw(DimensionMismatch(
        "physical field must have size $((grid.Nx, grid.Ny)), got $(size(ω))"))
    return dealias!(rfft(Float64.(ω)), grid)
end

function vorticity_hat(grid::Grid, ω::AbstractMatrix{<:Complex})
    size(ω) == (grid.Mx, grid.Ny) || throw(DimensionMismatch(
        "spectral field must have size $((grid.Mx, grid.Ny)), got $(size(ω))"))
    return dealias!(ComplexF64.(ω), grid)
end

"""
    forcing_hat(grid, g)

Dealiased spectral representation of the curl `g` of the forcing, given as for
[`vorticity_hat`](@ref), or `nothing` for no forcing. To give the forcing as a vector field
`(fx, fy)` instead, use `vorticity_from_velocity(grid, fx, fy)`.
"""
forcing_hat(grid::Grid, ::Nothing) = zeros(ComplexF64, grid.Mx, grid.Ny)
forcing_hat(grid::Grid, g) = vorticity_hat(grid, g)

"""
    vorticity_from_velocity(grid, u, v)

Dealiased spectral representation of the curl `∂ₓv - ∂ᵧu` of the vector field `(u, v)`, each
component given as a function `(x, y) -> value` or as a physical array of size `(Nx, Ny)`.
"""
function vorticity_from_velocity(grid::Grid, u, v)
    u_hat = vorticity_hat(grid, u)
    v_hat = vorticity_hat(grid, v)
    return dealias!(im .* grid.κx .* v_hat .- im .* transpose(grid.κy) .* u_hat, grid)
end

"""
    band_forcing(grid, κmin, κmax; rms = 1.0, rng = Xoshiro(2026))

Curl of a steady forcing, in spectral form, with equal amplitudes and random phases on all the
modes with physical wavenumber `κmin ≤ |κ| ≤ κmax`, scaled so that the curl has spatial
root-mean-square value `rms`. Returns the spectral field and the list of integer wavenumber
indices `(jx, jy)` of the forced modes (one of each pair `±(jx, jy)`).
"""
function band_forcing(grid::Grid, κmin::Real, κmax::Real; rms::Real = 1.0, rng::AbstractRNG = Xoshiro(2026))
    g_hat = zeros(ComplexF64, grid.Mx, grid.Ny)
    modes = Tuple{Int, Int}[]
    for j in 1:grid.Ny, i in 1:grid.Mx
        grid.mask[i, j] || continue
        κ = sqrt(grid.κ2[i, j])
        κmin ≤ κ ≤ κmax && κ > 0 || continue
        jx = i - 1
        jy = j - 1 < (grid.Ny + 1) ÷ 2 ? j - 1 : j - 1 - grid.Ny
        jx == 0 && jy < 0 && continue        # (0, -jy) is the conjugate of (0, jy)
        g_hat[i, j] = cis(2π * rand(rng))
        if jx == 0                           # keep the Hermitian symmetry on the jx = 0 column
            g_hat[1, mod(-jy, grid.Ny) + 1] = conj(g_hat[i, j])
        end
        push!(modes, (jx, jy))
    end
    isempty(modes) && throw(ArgumentError("no retained modes in the band $κmin ≤ |κ| ≤ $κmax"))
    return _scale_rms(grid, g_hat, rms), modes
end

"""
    random_vorticity(grid, κmin, κmax; energy = 1.0, slope = 0.0, rng = Xoshiro())

Random vorticity field, in spectral form, with random phases on all the modes with physical
wavenumber `κmin ≤ |κ| ≤ κmax`, amplitudes proportional to `|κ|^slope`, scaled to have kinetic
energy `½⟨|u|²⟩ = energy`.
"""
function random_vorticity(grid::Grid, κmin::Real, κmax::Real; energy::Real = 1.0, slope::Real = 0.0,
        rng::AbstractRNG = Xoshiro())
    w_hat = zeros(ComplexF64, grid.Mx, grid.Ny)
    for j in 1:grid.Ny, i in 1:grid.Mx
        κ = sqrt(grid.κ2[i, j])
        (grid.mask[i, j] && κ > 0 && κmin ≤ κ ≤ κmax) || continue
        w_hat[i, j] = κ^slope * cis(2π * rand(rng))
    end
    _hermitian_first_column!(w_hat, grid)
    E = _energy(grid, w_hat)
    E > 0 || throw(ArgumentError("no retained modes in the band $κmin ≤ |κ| ≤ $κmax"))
    return w_hat .* sqrt(energy / E)
end

"""
    stokes_state(model)

The vorticity `(1/ν)A⁻¹g`, i.e. the steady state of the Stokes problem `νΔω + g = 0`, in
spectral form. It is a steady state of the full equation only in special cases (e.g. when
the forcing lives on a single shell `|κ| = const`), but it is a convenient initial condition.
"""
stokes_state(model) = [model.grid.κ2[i] > 0 ? model.g_hat[i] / (model.ν * model.grid.κ2[i]) : zero(ComplexF64)
    for i in CartesianIndices(model.g_hat)]

# Enforce the Hermitian symmetry w(0, -jy) = conj(w(0, jy)) on the jx = 0 column, so that the
# field is the transform of a real field.
function _hermitian_first_column!(w_hat, grid::Grid)
    Ny = grid.Ny
    for j in 2:Ny
        jc = mod(-(j - 1), Ny) + 1
        if j < jc
            w_hat[1, jc] = conj(w_hat[1, j])
        elseif j == jc
            w_hat[1, j] = real(w_hat[1, j])
        end
    end
    w_hat[1, 1] = real(w_hat[1, 1])
    return w_hat
end

function _scale_rms(grid::Grid, f_hat, rms)
    m = sqrt(sum(spectral_weights(grid) .* abs2.(f_hat)))
    return f_hat .* (rms / m)
end

function _energy(grid::Grid, w_hat)
    W = spectral_weights(grid)
    return sum(W[i] * abs2(w_hat[i]) / grid.κ2[i] for i in eachindex(w_hat) if grid.κ2[i] > 0) / 2
end
