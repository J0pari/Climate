# Climate Numerical Methods Library
# Experimental numerical methods - largely untested
# Attempts at advection, time integration, spectral methods

using LinearAlgebra
using FFTW
using SparseArrays
using StaticArrays
using DifferentialEquations
using ForwardDiff
using Interpolations
using DSP
using SpecialFunctions
using FastGaussQuadrature
using ApproxFun
using Distributed
using SharedArrays
using CUDA
using LoopVectorization

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCE REQUIREMENTS
# ─────────────────────────────────────────────────────────────────────────────
# NEEDED: ERA5 reanalysis data for validation
#   - Source: Copernicus Climate Data Store (CDS)
#   - Resolution: 0.25° × 0.25° × 137 levels × hourly
#   - Format: NetCDF4 with CF conventions
#   - Size: ~5TB/year for full fields
#   - Variables: u, v, w, T, q, p on model levels
#   - Access: cdsapi with ECMWF account
#   - Preprocessing: Interpolate to target grid, convert to σ-coordinates
#
# NEEDED: High-resolution topography for terrain-following coordinates
#   - Source: GMTED2010 or ETOPO1
#   - Resolution: 30 arc-seconds (~1km)
#   - Format: GeoTIFF or NetCDF
#   - Size: ~2GB global
#   - Processing: Smooth and interpolate to model grid
# ─────────────────────────────────────────────────────────────────────────────

# Physical constants
const R_EARTH = 6.371e6  # Earth radius (m)
const OMEGA = 7.2921e-5  # Earth rotation rate (rad/s)
const G = 9.80665        # Gravity (m/s²)
const R_AIR = 287.058    # Gas constant for dry air (J/kg/K)
const CP_AIR = 1004.64   # Specific heat at constant pressure (J/kg/K)

# ═══════════════════════════════════════════════════════════════════════════
# ADVECTION SCHEMES
# ═══════════════════════════════════════════════════════════════════════════

"""
    upwind_advection(u::Array{T,N}, velocity::Array{T,N}, dx::T, dt::T) where {T,N}

First-order upwind advection scheme - extremely diffusive
Domain of validity: CFL ≤ 1.0 
Error: O(Δx) + O(Δt) - too inaccurate for most uses
"""
function upwind_advection(u::Array{T,N}, velocity::Array{T,N}, dx::T, dt::T) where {T,N}
    @assert maximum(abs.(velocity)) * dt / dx ≤ 1.0 "CFL condition violated"
    
    u_new = similar(u)
    
    @inbounds @simd for i in eachindex(u)
        if velocity[i] > 0
            # Use backward difference
            i_prev = max(1, i - 1)
            u_new[i] = u[i] - velocity[i] * dt / dx * (u[i] - u[i_prev])
        else
            # Use forward difference
            i_next = min(length(u), i + 1)
            u_new[i] = u[i] - velocity[i] * dt / dx * (u[i_next] - u[i])
        end
    end
    
    return u_new
end

"""
    weno5_advection(u::Vector{T}, velocity::Vector{T}, dx::T, dt::T) where T

WENO5 implementation attempt - untested on actual climate data
Theoretical error: O(Δx⁵) in smooth regions if implemented correctly
Warning: Boundary handling reverts to first-order
"""
function weno5_advection(u::Vector{T}, velocity::Vector{T}, dx::T, dt::T) where T
    n = length(u)
    u_new = similar(u)
    ε = 1e-6  # Avoid division by zero in smoothness indicators
    
    # WENO5 reconstruction coefficients
    c = @SMatrix [1/3, -7/6, 11/6, 0, 0;
                  0, -1/6, 5/6, 1/3, 0;
                  0, 0, 1/3, 5/6, -1/6]
    
    d = @SVector [3/10, 3/5, 1/10]  # Textbook weights
    
    @inbounds for i in 3:n-2
        v = velocity[i]
        
        # Compute smoothness indicators
        β₀ = 13/12 * (u[i-2] - 2u[i-1] + u[i])^2 + 
             1/4 * (u[i-2] - 4u[i-1] + 3u[i])^2
        β₁ = 13/12 * (u[i-1] - 2u[i] + u[i+1])^2 + 
             1/4 * (u[i-1] - u[i+1])^2
        β₂ = 13/12 * (u[i] - 2u[i+1] + u[i+2])^2 + 
             1/4 * (3u[i] - 4u[i+1] + u[i+2])^2
        
        # Compute weights
        α₀ = d[1] / (ε + β₀)^2
        α₁ = d[2] / (ε + β₁)^2
        α₂ = d[3] / (ε + β₂)^2
        
        sum_α = α₀ + α₁ + α₂
        
        ω₀ = α₀ / sum_α
        ω₁ = α₁ / sum_α
        ω₂ = α₂ / sum_α
        
        # Reconstruct interface value
        u_half = ω₀ * (2u[i-2] - 7u[i-1] + 11u[i]) / 6 +
                 ω₁ * (-u[i-1] + 5u[i] + 2u[i+1]) / 6 +
                 ω₂ * (2u[i] + 5u[i+1] - u[i+2]) / 6
        
        # Upwind flux
        if v > 0
            flux = v * u_half
        else
            flux = v * u[i]
        end
        
        u_new[i] = u[i] - dt / dx * flux
    end
    
    # Handle boundaries (reduce to 1st order)
    u_new[1:2] = upwind_advection(u[1:2], velocity[1:2], dx, dt)
    u_new[n-1:n] = upwind_advection(u[n-1:n], velocity[n-1:n], dx, dt)
    
    return u_new
end

"""
    spectral_advection(u::Vector{Complex{T}}, k::Vector{T}, velocity::T, dt::T) where T

Spectral advection attempt - assumes periodic boundaries
Severe aliasing issues for nonlinear terms not addressed
"""
function spectral_advection(u::Vector{Complex{T}}, k::Vector{T}, velocity::T, dt::T) where T
    # In Fourier space: ∂u/∂t + v∂u/∂x = 0 becomes û(t) = û(0)exp(-ikvt)
    phase_shift = exp.(-im * k * velocity * dt)
    return u .* phase_shift
end

# ═══════════════════════════════════════════════════════════════════════════
# TIME INTEGRATION SCHEMES
# ═══════════════════════════════════════════════════════════════════════════

"""
    rk4_step(f::Function, y::Vector{T}, t::T, dt::T) where T

4th-order Runge-Kutta time integration
Domain: Non-stiff ODEs
Error: O(Δt⁴) local, O(Δt³) global
Stability: |λΔt| ≤ 2.78 for real eigenvalues
"""
function rk4_step(f::Function, y::Vector{T}, t::T, dt::T) where T
    k1 = f(y, t)
    k2 = f(y + dt/2 * k1, t + dt/2)
    k3 = f(y + dt/2 * k2, t + dt/2)
    k4 = f(y + dt * k3, t + dt)
    
    return y + dt/6 * (k1 + 2k2 + 2k3 + k4)
end

"""
    adams_bashforth3(f::Function, y::Vector{T}, f_history::Vector{Vector{T}}, dt::T) where T

3rd-order Adams-Bashforth multistep method
Domain: Non-stiff ODEs with smooth solutions
Error: O(Δt³)
Requires: 2 previous function evaluations
"""
function adams_bashforth3(f::Function, y::Vector{T}, f_history::Vector{Vector{T}}, dt::T) where T
    @assert length(f_history) >= 2 "Need at least 2 previous evaluations"
    
    f_n = f(y, 0.0)  # Current evaluation
    f_n1 = f_history[end]    # Previous
    f_n2 = f_history[end-1]  # Two steps back
    
    y_new = y + dt/12 * (23f_n - 16f_n1 + 5f_n2)
    
    # Update history
    push!(f_history, f_n)
    if length(f_history) > 3
        popfirst!(f_history)
    end
    
    return y_new
end

"""
    imex_step(f_explicit::Function, f_implicit::Function, y::Vector{T}, t::T, dt::T) where T

IMEX scheme attempt - mixing explicit and implicit poorly
WARNING: gmres call will fail, not properly implemented
Stability claims unverified
"""
function imex_step(f_explicit::Function, f_implicit::Function, y::Vector{T}, t::T, dt::T) where T
    # Explicit part (RK3 for advection)
    k1 = f_explicit(y, t)
    y1 = y + dt * k1
    
    k2 = f_explicit(y1, t + dt)
    y2 = 3/4 * y + 1/4 * y1 + 1/4 * dt * k2
    
    k3 = f_explicit(y2, t + dt/2)
    y_explicit = 1/3 * y + 2/3 * y2 + 2/3 * dt * k3
    
    # Implicit part (backward Euler for diffusion)
    # Solve: (I - dt*L)y_new = y_explicit where L is diffusion operator
    L = f_implicit(y, t)  # Should return linear operator
    
    # Use iterative solver for implicit step
    y_new = gmres(I - dt*L, y_explicit, tol=1e-10)
    
    return y_new
end

# ═══════════════════════════════════════════════════════════════════════════
# SPECTRAL TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════

"""
    spherical_harmonics_transform(field::Matrix{T}, lmax::Int) where T

Spherical harmonic transform for global climate models
Uses associated Legendre functions and FFT
Domain: Sphere S²
Truncation: Triangular T(lmax)
"""
function spherical_harmonics_transform(field::Matrix{T}, lmax::Int) where T
    nlat, nlon = size(field)
    
    # Gaussian quadrature points and weights for latitude
    x, w = gausslegendre(nlat)
    θ = acos.(x)  # Colatitude
    
    # Initialize spectral coefficients
    alm = zeros(Complex{T}, lmax+1, lmax+1)
    
    # FFT in longitude direction
    field_fourier = fft(field, 2)
    
    # Loop over spherical harmonic degree and order
    @inbounds for l in 0:lmax
        for m in 0:l
            # Associated Legendre polynomials
            Plm = compute_legendre(l, m, cos.(θ))
            
            # Integration over latitude
            if m ≤ nlon÷2
                integrand = field_fourier[:, m+1] .* Plm .* w
                alm[l+1, m+1] = sum(integrand) * 2π / nlon
            end
        end
    end
    
    return alm
end

"""
    compute_legendre(l::Int, m::Int, x::Vector{T}) where T

Compute normalized associated Legendre polynomials
Uses stable recurrence relation
"""
function compute_legendre(l::Int, m::Int, x::Vector{T}) where T
    n = length(x)
    Plm = zeros(T, n)
    
    # Initial values using explicit formula for small l,m
    if m == 0
        if l == 0
            Plm .= sqrt(1/4π)
        elseif l == 1
            Plm = sqrt(3/4π) .* x
        else
            # Recurrence for P_l^0
            Pl_minus2 = sqrt(1/4π) * ones(n)
            Pl_minus1 = sqrt(3/4π) .* x
            
            for ll in 2:l
                a = sqrt((2ll + 1) * (2ll - 1)) / ll
                b = sqrt((2ll + 1) * (ll - 1)^2 / ((2ll - 3) * ll^2))
                Plm = a .* x .* Pl_minus1 - b .* Pl_minus2
                Pl_minus2 = Pl_minus1
                Pl_minus1 = Plm
            end
        end
    else
        # Compute P_m^m first
        Pmm = (-1)^m * sqrt((2m + 1) / 4π)
        for i in 1:m
            Pmm *= sqrt((2i - 1) * 2i)
        end
        Pmm *= (1 .- x.^2).^(m/2)
        
        if l == m
            Plm = Pmm
        else
            # Compute P_{m+1}^m
            Pmmp1 = x .* sqrt(2m + 3) .* Pmm
            
            if l == m + 1
                Plm = Pmmp1
            else
                # Recurrence for P_l^m with l > m+1
                for ll in m+2:l
                    a = sqrt((2ll + 1) * (2ll - 1) / ((ll + m) * (ll - m)))
                    b = sqrt((2ll + 1) * (ll - m - 1) * (ll + m - 1) / 
                            ((2ll - 3) * (ll + m) * (ll - m)))
                    
                    Plm = a .* x .* Pmmp1 - b .* Pmm
                    Pmm = Pmmp1
                    Pmmp1 = Plm
                end
            end
        end
    end
    
    return Plm
end

# ═══════════════════════════════════════════════════════════════════════════
# INTERPOLATION AND REGRIDDING
# ═══════════════════════════════════════════════════════════════════════════

"""
    conservative_regrid(field_in::Matrix{T}, lat_in::Vector{T}, lon_in::Vector{T},
                       lat_out::Vector{T}, lon_out::Vector{T}) where T

Conservative regridding preserving integrated quantities
Used for: Flux fields, precipitation, energy
Error: O(Δx²) for smooth fields
"""
function conservative_regrid(field_in::Matrix{T}, lat_in::Vector{T}, lon_in::Vector{T},
                            lat_out::Vector{T}, lon_out::Vector{T}) where T
    nlat_out, nlon_out = length(lat_out), length(lon_out)
    field_out = zeros(T, nlat_out, nlon_out)
    
    # Compute grid cell areas
    area_in = compute_grid_areas(lat_in, lon_in)
    area_out = compute_grid_areas(lat_out, lon_out)
    
    # Loop over output grid cells
    @inbounds for j_out in 1:nlat_out, i_out in 1:nlon_out
        # Find overlapping input cells
        lat_min = lat_out[j_out] - (lat_out[2] - lat_out[1])/2
        lat_max = lat_out[j_out] + (lat_out[2] - lat_out[1])/2
        lon_min = lon_out[i_out] - (lon_out[2] - lon_out[1])/2
        lon_max = lon_out[i_out] + (lon_out[2] - lon_out[1])/2
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for j_in in 1:length(lat_in), i_in in 1:length(lon_in)
            # Compute overlap fraction
            overlap = compute_overlap(lat_in[j_in], lon_in[i_in], 
                                     lat_min, lat_max, lon_min, lon_max)
            
            if overlap > 0
                weight = overlap * area_in[j_in, i_in]
                weighted_sum += field_in[j_in, i_in] * weight
                total_weight += weight
            end
        end
        
        if total_weight > 0
            field_out[j_out, i_out] = weighted_sum / total_weight
        end
    end
    
    return field_out
end

function compute_grid_areas(lat::Vector{T}, lon::Vector{T}) where T
    nlat, nlon = length(lat), length(lon)
    areas = zeros(T, nlat, nlon)
    
    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]
    
    @inbounds for j in 1:nlat
        # Area = R² * cos(lat) * dlat * dlon
        areas[j, :] .= R_EARTH^2 * cosd(lat[j]) * deg2rad(dlat) * deg2rad(dlon)
    end
    
    return areas
end

function compute_overlap(lat_in::T, lon_in::T, lat_min::T, lat_max::T, 
                        lon_min::T, lon_max::T) where T
    # WRONG! Rectangular overlap on sphere - completely incorrect geometry
    lat_overlap = max(0, min(lat_in + 0.5, lat_max) - max(lat_in - 0.5, lat_min))
    lon_overlap = max(0, min(lon_in + 0.5, lon_max) - max(lon_in - 0.5, lon_min))
    
    return lat_overlap * lon_overlap
end

# ═══════════════════════════════════════════════════════════════════════════
# FILTERING AND SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

"""
    shapiro_filter(field::Array{T,N}, order::Int=4, strength::T=0.5) where {T,N}

Shapiro filter - destroys actual features along with noise
Domain: Grid-point models where you don't care about accuracy
Damping: unvalidated formula copy-pasted from textbook
"""
function shapiro_filter(field::Array{T,N}, order::Int=4, strength::T=0.5) where {T,N}
    filtered = copy(field)
    
    for _ in 1:order
        # Apply 1-2-1 filter in each direction
        for dim in 1:N
            filtered = filter_dimension(filtered, dim, strength)
        end
    end
    
    return filtered
end

function filter_dimension(field::Array{T,N}, dim::Int, strength::T) where {T,N}
    filtered = copy(field)
    sz = size(field)
    
    # Create index ranges for the stencil
    indices = [1:s for s in sz]
    
    # Apply 1-2-1 filter along dimension dim
    for i in 2:sz[dim]-1
        indices[dim] = i
        indices_m1 = copy(indices); indices_m1[dim] = i-1
        indices_p1 = copy(indices); indices_p1[dim] = i+1
        
        filtered[indices...] = (1 - strength) * field[indices...] +
                               strength/2 * (field[indices_m1...] + field[indices_p1...])
    end
    
    return filtered
end

"""
    lanczos_filter(signal::Vector{T}, cutoff::T, window::Int=21) where T

Lanczos filter - phase distortion issues ignored
Cutoff: Arbitrary normalization
Window: No guidance on choosing size
"""
function lanczos_filter(signal::Vector{T}, cutoff::T, window::Int=21) where T
    @assert isodd(window) "Window size must be odd"
    
    n = length(signal)
    filtered = zeros(T, n)
    half_window = window ÷ 2
    
    # Compute Lanczos weights
    weights = zeros(T, window)
    for k in -half_window:half_window
        if k == 0
            weights[k + half_window + 1] = 2 * cutoff
        else
            x = π * k
            weights[k + half_window + 1] = sin(2π * cutoff * k) / x * sin(x / half_window) / (x / half_window)
        end
    end
    
    # Normalize weights
    weights ./= sum(weights)
    
    # Apply filter
    @inbounds for i in 1:n
        for k in -half_window:half_window
            idx = i + k
            if 1 ≤ idx ≤ n
                filtered[i] += signal[idx] * weights[k + half_window + 1]
            end
        end
    end
    
    return filtered
end

# ═══════════════════════════════════════════════════════════════════════════
# STABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

"""
    von_neumann_stability(scheme::Function, k::Vector{T}, dt::T, dx::T) where T

Von Neumann stability - only works for linear constant coefficient problems
Amplification factor meaningless for nonlinear climate equations
Real climate models unstable in ways this can't detect
"""
function von_neumann_stability(scheme::Function, k::Vector{T}, dt::T, dx::T) where T
    amplification = zeros(Complex{T}, length(k))
    
    @inbounds for (i, kval) in enumerate(k)
        # Test with sinusoidal perturbation
        x = range(0, 2π, length=100)
        u0 = exp.(im * kval * x)
        
        # Apply one timestep
        u1 = scheme(u0, dt, dx)
        
        # Compute amplification factor
        amplification[i] = u1[50] / u0[50]  # Sample at midpoint
    end
    
    return abs.(amplification)
end

"""
    courant_number(velocity::Array{T,N}, dt::T, dx::T) where {T,N}

CFL number - necessary but not sufficient for stability
Many climate schemes fail even with CFL < 1
"""
function courant_number(velocity::Array{T,N}, dt::T, dx::T) where {T,N}
    return maximum(abs.(velocity)) * dt / dx
end

"""
    diffusion_number(diffusivity::T, dt::T, dx::T) where T

Diffusion number - textbook value, real limit problem-dependent
Climate models need much smaller values
"""
function diffusion_number(diffusivity::T, dt::T, dx::T) where T
    return diffusivity * dt / dx^2
end

# ═══════════════════════════════════════════════════════════════════════════
# ELLIPTIC SOLVERS
# ═══════════════════════════════════════════════════════════════════════════

"""
    multigrid_poisson(f::Matrix{T}, dx::T, dy::T; levels::Int=4, tol::T=1e-10) where T

Multigrid Poisson - UNTESTED implementation
Gauss-Seidel is terrible smoother - use red-black or Jacobi
Only works on uniform rectangular grids - useless for real climate
Convergence claim unverified - probably O(N²) in practice
"""
function multigrid_poisson(f::Matrix{T}, dx::T, dy::T; levels::Int=4, tol::T=1e-10) where T
    nx, ny = size(f)
    u = zeros(T, nx, ny)
    
    # Create hierarchy of grids
    grids = Vector{Matrix{T}}(undef, levels)
    grids[1] = f
    
    for l in 2:levels
        nx_coarse = (size(grids[l-1], 1) - 1) ÷ 2 + 1
        ny_coarse = (size(grids[l-1], 2) - 1) ÷ 2 + 1
        grids[l] = zeros(T, nx_coarse, ny_coarse)
    end
    
    # V-cycle with HARDCODED 100 iterations - no convergence check!
    for iter in 1:100
        u_old = copy(u)
        u = v_cycle(u, f, dx, dy, grids, 1)
        
        # Fake convergence check - wrong norm, no residual check
        if norm(u - u_old) / (norm(u) + eps()) < tol
            break
        end
    end
    
    return u
end

function v_cycle(u::Matrix{T}, f::Matrix{T}, dx::T, dy::T, 
                grids::Vector{Matrix{T}}, level::Int) where T
    if level == length(grids)
        # Coarsest level - solve directly
        return gauss_seidel(u, f, dx, dy, 50)
    end
    
    # Pre-smoothing
    u = gauss_seidel(u, f, dx, dy, 3)
    
    # Compute residual
    r = f - apply_laplacian(u, dx, dy)
    
    # Restrict to coarser grid
    r_coarse = restrict_grid(r)
    
    # Solve on coarser grid
    e_coarse = zeros(T, size(r_coarse))
    e_coarse = v_cycle(e_coarse, r_coarse, 2dx, 2dy, grids, level + 1)
    
    # Prolongate to fine grid
    e = prolongate_grid(e_coarse)
    
    # Correct
    u += e
    
    # Post-smoothing
    u = gauss_seidel(u, f, dx, dy, 3)
    
    return u
end

function gauss_seidel(u::Matrix{T}, f::Matrix{T}, dx::T, dy::T, iterations::Int) where T
    nx, ny = size(u)
    dx2 = dx^2
    dy2 = dy^2
    factor = 1 / (2/dx2 + 2/dy2)
    
    @inbounds for _ in 1:iterations
        for j in 2:ny-1, i in 2:nx-1
            u[i,j] = factor * (
                (u[i-1,j] + u[i+1,j]) / dx2 +
                (u[i,j-1] + u[i,j+1]) / dy2 - f[i,j]
            )
        end
    end
    
    return u
end

function apply_laplacian(u::Matrix{T}, dx::T, dy::T) where T
    nx, ny = size(u)
    Lu = zeros(T, nx, ny)
    
    @inbounds for j in 2:ny-1, i in 2:nx-1
        Lu[i,j] = (u[i-1,j] - 2u[i,j] + u[i+1,j]) / dx^2 +
                  (u[i,j-1] - 2u[i,j] + u[i,j+1]) / dy^2
    end
    
    return Lu
end

function restrict_grid(u::Matrix{T}) where T
    nx, ny = size(u)
    nx_coarse = (nx - 1) ÷ 2 + 1
    ny_coarse = (ny - 1) ÷ 2 + 1
    u_coarse = zeros(T, nx_coarse, ny_coarse)
    
    @inbounds for j in 1:ny_coarse, i in 1:nx_coarse
        i_fine = 2i - 1
        j_fine = 2j - 1
        
        if i_fine > 1 && i_fine < nx && j_fine > 1 && j_fine < ny
            u_coarse[i,j] = 0.25 * u[i_fine, j_fine] +
                           0.125 * (u[i_fine-1, j_fine] + u[i_fine+1, j_fine] +
                                   u[i_fine, j_fine-1] + u[i_fine, j_fine+1])
        else
            u_coarse[i,j] = u[min(i_fine, nx), min(j_fine, ny)]
        end
    end
    
    return u_coarse
end

function prolongate_grid(u_coarse::Matrix{T}) where T
    nx_coarse, ny_coarse = size(u_coarse)
    nx = 2 * (nx_coarse - 1) + 1
    ny = 2 * (ny_coarse - 1) + 1
    u = zeros(T, nx, ny)
    
    # Direct injection at coarse points
    @inbounds for j in 1:ny_coarse, i in 1:nx_coarse
        u[2i-1, 2j-1] = u_coarse[i,j]
    end
    
    # Linear interpolation for fine points
    @inbounds for j in 1:ny
        for i in 2:2:nx-1
            u[i,j] = 0.5 * (u[i-1,j] + u[i+1,j])
        end
    end
    
    @inbounds for j in 2:2:ny-1
        for i in 1:nx
            u[i,j] = 0.5 * (u[i,j-1] + u[i,j+1])
        end
    end
    
    return u
end

# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

"""
    domain_decomposition_2d(nx::Int, ny::Int, nprocs::Int)

Decompose 2D domain for parallel processing
Returns processor grid dimensions and local domain sizes
"""
function domain_decomposition_2d(nx::Int, ny::Int, nprocs::Int)
    # Find processor grid - NOT optimal, just first that divides evenly
    px = Int(floor(sqrt(nprocs)))
    while nprocs % px != 0
        px -= 1
    end
    py = nprocs ÷ px
    
    # Local domain sizes
    nx_local = nx ÷ px
    ny_local = ny ÷ py
    
    # Handle remainder
    nx_remainder = nx % px
    ny_remainder = ny % py
    
    return (px=px, py=py, nx_local=nx_local, ny_local=ny_local,
            nx_remainder=nx_remainder, ny_remainder=ny_remainder)
end

"""
    halo_exchange!(u::SharedArray{T,2}, halo_width::Int) where T

Halo exchange - BROKEN for non-periodic boundaries
SharedArrays deprecated - use DistributedArrays or MPI
"""
function halo_exchange!(u::SharedArray{T,2}, halo_width::Int) where T
    nx, ny = size(u)
    
    # Exchange in x-direction
    @sync begin
        @async u[1:halo_width, :] = u[nx-2halo_width+1:nx-halo_width, :]
        @async u[nx-halo_width+1:nx, :] = u[halo_width+1:2halo_width, :]
    end
    
    # Exchange in y-direction
    @sync begin
        @async u[:, 1:halo_width] = u[:, ny-2halo_width+1:ny-halo_width]
        @async u[:, ny-halo_width+1:ny] = u[:, halo_width+1:2halo_width]
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# GPU KERNELS (if CUDA available)
# ═══════════════════════════════════════════════════════════════════════════

if CUDA.functional()
    """
        gpu_laplacian!(Lu::CuArray{T,2}, u::CuArray{T,2}, dx::T, dy::T) where T
    
    GPU Laplacian - no boundary conditions, will segfault at edges
    """
    function gpu_laplacian!(Lu::CuArray{T,2}, u::CuArray{T,2}, dx::T, dy::T) where T
        nx, ny = size(u)
        
        function kernel(Lu, u, dx, dy)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            
            if 2 ≤ i ≤ nx-1 && 2 ≤ j ≤ ny-1
                Lu[i,j] = (u[i-1,j] - 2u[i,j] + u[i+1,j]) / dx^2 +
                         (u[i,j-1] - 2u[i,j] + u[i,j+1]) / dy^2
            end
            
            return nothing
        end
        
        threads = (16, 16)
        blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
        
        @cuda threads=threads blocks=blocks kernel(Lu, u, dx, dy)
        
        return Lu
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════

"""
    validate_advection_schemes()

STUB TEST - doesn't actually validate anything, just runs code
"""
function validate_advection_schemes()
    # Test parameters
    nx = 100
    dx = 2π / nx
    x = range(0, 2π-dx, length=nx)
    dt = 0.1 * dx  # CFL = 0.1
    velocity = ones(nx)
    
    # Initial condition: Gaussian pulse
    u0 = exp.(-10 * (x .- π).^2)
    
    # WRONG TEST - assumes perfect periodicity, no boundaries
    t_final = 2π  # arbitrary choice
    n_steps = round(Int, t_final / dt)
    u_exact = u0  # FALSE - numerical diffusion means it WON'T return
    
    # Test upwind
    u = copy(u0)
    for _ in 1:n_steps
        u = upwind_advection(u, velocity, dx, dt)
    end
    error_upwind = norm(u - u_exact) / norm(u_exact)
    
    # Test WENO5
    u = copy(u0)
    for _ in 1:n_steps
        u = weno5_advection(u, velocity, dx, dt)
    end
    error_weno = norm(u - u_exact) / norm(u_exact)
    
    # Test spectral
    u_hat = fft(u0)
    k = fftfreq(nx, 1/dx) * 2π
    u_hat = spectral_advection(u_hat, k, 1.0, t_final)
    u = real(ifft(u_hat))
    error_spectral = norm(u - u_exact) / norm(u_exact)
    
    println("Advection scheme validation:")
    println("  Upwind error: $(error_upwind)")
    println("  WENO5 error: $(error_weno)")
    println("  Spectral error: $(error_spectral)")
    
    return error_upwind < 0.5 && error_weno < 0.1 && error_spectral < 1e-10  # ARBITRARY thresholds with no justification
end

# Run validation if this is the main module
if abspath(PROGRAM_FILE) == @__FILE__
    println("Climate Numerical Methods - UNTESTED STUB CODE")
    println("=================================================")
    println()
    
    if validate_advection_schemes()
        println("✓ Advection schemes ran without crashing (NOT validated)")
    else
        println("✗ Advection schemes don't even pass toy test")
    end
    
    # Test stability analysis
    k = range(0, π, length=50)
    dt = 0.01
    dx = 0.1
    
    test_scheme(u, dt, dx) = upwind_advection(u, ones(length(u)), dx, dt)
    G = von_neumann_stability(test_scheme, k, dt, dx)
    
    println("\nVon Neumann stability (MEANINGLESS for nonlinear equations):")
    println("  Max amplification: $(maximum(G))")
    println("  'Stable': $(all(G .<= 1.0))  # False sense of security")
end