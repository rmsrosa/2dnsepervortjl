"""
    NSEModel(grid; ν, forcing = nothing, fftw_flags = FFTW.MEASURE, fftw_threads = 1)

The vorticity equation `∂ₜω = νΔω + g - (∂ₓ² - ∂ᵧ²)(uv) - ∂ₓ∂ᵧ(v² - u²)` (Basdevant's form of
`-u ⋅ ∇ω`) on `grid`, with viscosity `ν` and curl of the forcing `g` given by `forcing` in any
form accepted by [`forcing_hat`](@ref) (a function `g(x, y)`, a physical or a spectral array,
or `nothing`). The forcing is steady.

The model holds the spectral operators, the FFTW plans and preallocated work arrays used by
the right-hand side, so the right-hand side does not allocate. Because of the work arrays, a
model must not be shared by computations running concurrently in different threads; build
one model per thread or task instead.
"""
struct NSEModel{P, IP}
    grid::Grid
    ν::Float64
    g_hat::Matrix{ComplexF64}
    Hu::Matrix{ComplexF64}           # û = Hu ω̂
    Hv::Matrix{ComplexF64}           # v̂ = Hv ω̂
    DxxDyy::Matrix{ComplexF64}       # symbol of ∂ₓ² - ∂ᵧ²
    Dxy::Matrix{ComplexF64}          # symbol of ∂ₓ∂ᵧ
    L::Vector{ComplexF64}            # diagonal of the linear operator νΔ, vectorized
    plan::P
    iplan::IP
    # work arrays
    u::Matrix{Float64}
    v::Matrix{Float64}
    uh::Matrix{Float64}
    vh::Matrix{Float64}
    a::Matrix{Float64}
    b::Matrix{Float64}
    sc::Matrix{ComplexF64}
    a_hat::Matrix{ComplexF64}
    b_hat::Matrix{ComplexF64}
end

function NSEModel(grid::Grid; ν::Real, forcing = nothing, fftw_flags = FFTW.MEASURE, fftw_threads::Integer = 1)
    ν ≥ 0 || throw(ArgumentError("the viscosity must be nonnegative"))
    FFTW.set_num_threads(fftw_threads)
    Dx = [im * a for a in grid.κx, _ in grid.κy]
    Dy = [im * b for _ in grid.κx, b in grid.κy]
    k2 = grid.κ2
    Hu = [k2[i] > 0 ? Dy[i] / k2[i] : zero(ComplexF64) for i in eachindex(k2)]
    Hv = [k2[i] > 0 ? -Dx[i] / k2[i] : zero(ComplexF64) for i in eachindex(k2)]
    Hu = reshape(Hu, size(k2)); Hv = reshape(Hv, size(k2))
    phys() = zeros(grid.Nx, grid.Ny)
    spec() = zeros(ComplexF64, grid.Mx, grid.Ny)
    plan = plan_rfft(phys(); flags = fftw_flags)
    iplan = plan_irfft(spec(), grid.Nx; flags = fftw_flags)
    g_hat = forcing_hat(grid, forcing)
    return NSEModel(grid, Float64(ν), g_hat, Hu, Hv, Dx .^ 2 .- Dy .^ 2, Dx .* Dy,
        vec(complex.(-ν .* k2)), plan, iplan,
        phys(), phys(), phys(), phys(), phys(), phys(), spec(), spec(), spec())
end

Base.show(io::IO, m::NSEModel) = print(io, "NSEModel(ν = $(m.ν), ", m.grid, ")")

# The irfft plans may overwrite their input, so they are always applied to the work array `sc`.
function _velocity!(u, v, w_hat, m::NSEModel)
    m.sc .= m.Hu .* w_hat
    mul!(u, m.iplan, m.sc)
    m.sc .= m.Hv .* w_hat
    mul!(v, m.iplan, m.sc)
    return u, v
end

"""
    nonlinear!(dw_hat, w_hat, model)

Writes `g - (∂ₓ² - ∂ᵧ²)(uv) - ∂ₓ∂ᵧ(v² - u²)`, i.e. the forcing minus the advection term, for the
spectral vorticity `w_hat`, into `dw_hat` (both of size `(Mx, Ny)`), dealiased.
"""
function nonlinear!(dw_hat, w_hat, m::NSEModel)
    _velocity!(m.u, m.v, w_hat, m)
    @. m.a = m.u * m.v
    @. m.b = m.v^2 - m.u^2
    mul!(m.a_hat, m.plan, m.a)
    mul!(m.b_hat, m.plan, m.b)
    @. dw_hat = m.g_hat - m.DxxDyy * m.a_hat - m.Dxy * m.b_hat
    return dealias!(dw_hat, m.grid)
end

"""
    jvp!(Jh_hat, h_hat, w_hat, model)

Writes the Jacobian of the right-hand side at `w_hat`, applied to `h_hat`, into `Jh_hat`. Since
the advection term `B(ω, ω)` is bilinear, `J(ω)h = νΔh - B(ω, h) - B(h, ω)`, which costs six
FFTs.
"""
function jvp!(Jh_hat, h_hat, w_hat, m::NSEModel)
    _velocity!(m.u, m.v, w_hat, m)
    _velocity!(m.uh, m.vh, h_hat, m)
    @. m.a = m.u * m.vh + m.uh * m.v
    @. m.b = 2 * (m.v * m.vh - m.u * m.uh)
    mul!(m.a_hat, m.plan, m.a)
    mul!(m.b_hat, m.plan, m.b)
    @. Jh_hat = -m.ν * m.grid.κ2 * h_hat - m.DxxDyy * m.a_hat - m.Dxy * m.b_hat
    return dealias!(Jh_hat, m.grid)
end

# Right-hand sides on the vectorized complex state (length Mx * Ny), the form required by the
# DiagonalOperator of the split problem, and on its real reinterpretation (length 2Mx * Ny),
# for the Newton-Krylov iterations of implicit methods.
_mat(m::NSEModel, u::AbstractVector{<:Complex}) = reshape(u, m.grid.Mx, m.grid.Ny)
_mat(m::NSEModel, u::AbstractVector{<:Real}) = reshape(reinterpret(ComplexF64, u), m.grid.Mx, m.grid.Ny)

nonlinear_vec!(du, u, m::NSEModel, t) = (nonlinear!(_mat(m, du), _mat(m, u), m); du)

function rhs_vec!(du, u, m::NSEModel, t)
    nonlinear!(_mat(m, du), _mat(m, u), m)
    w = _mat(m, u); dw = _mat(m, du)
    @. dw -= m.ν * m.grid.κ2 * w
    return du
end

function jvp_vec!(Jv, v, u, m::NSEModel, t)
    jvp!(_mat(m, vec(Jv)), _mat(m, vec(v)), _mat(m, u), m)
    return Jv
end
