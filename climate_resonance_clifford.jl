# Climate oscillation coupling using Clifford algebra Cl(n)
# Dimension emerges from observed oscillation modes
# ENSO, NAO, PDO, AMO, IOD, AAO, QBO, MJO
#
# DATA SOURCE REQUIREMENTS:
#
# 1. CLIMATE OSCILLATION INDICES:
#    - Source: NOAA PSL Climate Indices
#    - URL: https://psl.noaa.gov/data/climateindices/list/
#    - Indices needed:
#      * Niño3.4 (ENSO): Monthly 1870-present, 5°N-5°S, 170°W-120°W SST
#      * NAO: Station-based or PC-based, monthly 1865-present
#      * PDO: Leading PC of North Pacific SST, 1900-present
#      * AMO: Atlantic SST 0-70°N, detrended, 1856-present
#      * IOD/DMI: West-East Indian Ocean SST gradient, 1870-present
#      * AAO/SAM: 700hPa height 20°S-90°S, 1979-present
#      * QBO: 30hPa equatorial zonal wind, 1953-present
#      * MJO: Wheeler-Hendon RMM indices, daily 1974-present
#    - Format: ASCII time series or NetCDF4
#    - Size: <100MB for all indices
#    - Preprocessing: Standardize, remove trend, apply 3-month running mean
#    - Missing: Pre-satellite AAO, consistent QBO before 1953
#
# 2. CROSS-SPECTRAL ANALYSIS DATA:
#    - Source: Extended Reconstructed SST (ERSSTv5)
#    - Resolution: 2° x 2° monthly since 1854
#    - Format: NetCDF4
#    - Size: ~1GB
#    - API: https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html
#    - Preprocessing: Compute coherence spectra between index regions
#    - Missing: Subsurface ocean data for complete coupling
#
# 3. WAVELET DECOMPOSITION DATA:
#    - Source: Same as indices above
#    - Method: Morlet wavelet transform
#    - Output: Time-frequency power and phase
#    - Size: ~100MB per index (time × frequency)
#    - Missing: Uncertainty in wavelet transforms
#
# 4. REANALYSIS FOR SPATIAL PATTERNS:
#    - Source: ERA5 or NCEP/NCAR Reanalysis
#    - Variables: SLP, 500hPa height, SST, precipitation
#    - Resolution: 2.5° x 2.5° monthly
#    - Temporal: 1948-present (NCEP) or 1940-present (ERA5)
#    - Format: NetCDF4
#    - Size: ~10GB for required variables
#    - API: PSL or Copernicus CDS
#    - Preprocessing: Regress against oscillation indices
#    - Missing: Reliable data before 1948
#
# 5. RESONANCE DETECTION:
#    - Source: Computed from index time series
#    - Method: Hilbert transform for instantaneous phase
#    - Output: Phase locking episodes
#    - Missing: Nonlinear coupling strength metrics

using LinearAlgebra
using StaticArrays
using FFTW

# Find minimal dimension for oscillation representation

"""
Find minimal dimension to represent oscillation interactions
"""
function find_natural_dimension(oscillation_periods::Vector{Float64})
    # Each oscillation needs 2 dimensions (amplitude + phase)
    # Their interactions need additional dimensions
    n_modes = length(oscillation_periods)
    
    # Minimal dimension for independent oscillations
    min_dim = 2 * n_modes
    
    # Check which interactions are resonant (period ratios near integers)
    resonances = 0
    for i in 1:n_modes
        for j in i+1:n_modes
            ratio = oscillation_periods[i] / oscillation_periods[j]
            # Is this ratio close to a small integer ratio?
            for p in 1:5, q in 1:5
                if abs(ratio - p/q) < 0.1
                    resonances += 1
                    break
                end
            end
        end
    end
    
    # Resonant modes need extra dimensions for their coupling
    natural_dim = min_dim + resonances
    
    # Find the next power of 2 for Clifford algebra
    # Clifford algebras Cl(n) have dimension 2^n
    clifford_dim = Int(ceil(log2(natural_dim)))
    
    return natural_dim, clifford_dim
end

"""
Climate oscillation algebra from observational data
"""
struct ClimateAlgebra
    dimension::Int
    clifford_dim::Int
    gamma_matrices::Union{Nothing, Vector{Matrix{ComplexF64}}}
    
    # The actual climate oscillations that DETERMINE the structure
    oscillation_names::Vector{String}
    oscillation_periods::Vector{Float64}  # in years
    coupling_matrix::Matrix{Float64}      # observed coupling strengths
    
    function ClimateAlgebra(observations::Dict{String, Float64})
        names = collect(keys(observations))
        periods = collect(values(observations))
        
        # Calculate dimension from observed oscillations
        natural_dim, cliff_dim = find_natural_dimension(periods)
        
        # Build coupling matrix from observed correlations
        # Build from observed correlations
        n = length(names)
        coupling = zeros(n, n)
        
        # We would fill this from actual climate data
        # For now, using period ratios as a proxy
        for i in 1:n
            for j in 1:n
                if i != j
                    # Coupling strength inversely proportional to period difference
                    coupling[i,j] = 1.0 / (1.0 + abs(periods[i] - periods[j]))
                end
            end
        end
        
        # Only build Clifford structure if dimension is reasonable
        gamma = nothing
        if cliff_dim <= 10  # Beyond this, computation becomes impractical
            gamma = build_minimal_clifford(cliff_dim)
        end
        
        new(natural_dim, cliff_dim, gamma, names, periods, coupling)
    end
end

"""
Build Clifford algebra Cl(n) with gamma matrices
"""
function build_minimal_clifford(n::Int)
    dim = 2^(n÷2)
    
    # Pauli matrices - the fundamental 2D building blocks
    σ₁ = ComplexF64[0 1; 1 0]
    σ₂ = ComplexF64[0 -im; im 0]
    σ₃ = ComplexF64[1 0; 0 -1]
    
    gamma = Vector{Matrix{ComplexF64}}(undef, n)
    
    # Build using tensor products of Pauli matrices
    if n == 1
        gamma[1] = σ₁
    elseif n == 2
        gamma[1] = σ₁
        gamma[2] = σ₂
    else
        # Recursive construction via tensor products
        for i in 1:n
            if i <= n÷2
                gamma[i] = kron(I(2^((i-1)÷2)), kron(σ₁, I(2^((n-i-1)÷2))))
            else
                gamma[i] = kron(I(2^((i-n÷2-1))), kron(σ₃, I(2^(n-i))))
            end
        end
    end
    
    return gamma
end

"""
Compute climate resonance pattern at location and time
"""
function compute_resonance_pattern(algebra::ClimateAlgebra, 
                                  lat::Float64, lon::Float64, t::Float64)
    pattern = 0.0
    
    # Sum contributions from each oscillation mode
    for (i, name) in enumerate(algebra.oscillation_names)
        period = algebra.oscillation_periods[i]
        
        # Spatial pattern from EOF analysis
        spatial_pattern = get_spatial_pattern(name, lat, lon)
        
        # Temporal oscillation
        temporal = sin(2π * t / period)
        
        # Contribution weighted by coupling to other modes
        weight = sum(algebra.coupling_matrix[i, :])
        
        pattern += weight * spatial_pattern * temporal
    end
    
    return pattern
end

"""
Spatial patterns from EOF analysis
"""
function get_spatial_pattern(mode_name::String, lat::Float64, lon::Float64)
    if mode_name == "ENSO"
        # Tropical Pacific pattern
        return exp(-(lat^2/400)) * cos(π * (lon - 180) / 180)
        
    elseif mode_name == "NAO"
        # North Atlantic dipole pattern
        north_center = exp(-((lat - 65)^2 + (lon + 20)^2) / 200)
        south_center = exp(-((lat - 35)^2 + (lon + 30)^2) / 200)
        return north_center - south_center
        
    elseif mode_name == "PDO"
        # North Pacific pattern
        return exp(-((lat - 45)^2 / 300)) * cos(π * (lon - 200) / 120)
        
    elseif mode_name == "AMO"
        # Atlantic multidecadal pattern
        return exp(-((lat - 30)^2 / 500)) * sin(π * (lon + 50) / 100)
        
    else
        # Unknown mode - no pattern
        return 0.0
    end
end

"""
Calculate coupling matrix from time series correlations
"""
function discover_coupling_structure(time_series::Dict{String, Vector{Float64}})
    modes = collect(keys(time_series))
    n = length(modes)
    coupling = zeros(n, n)
    
    for i in 1:n
        for j in i+1:n
            # Compute lagged correlations
            series1 = time_series[modes[i]]
            series2 = time_series[modes[j]]
            
            # Find maximum correlation at any lag
            max_corr = 0.0
            for lag in -24:24  # months
                if lag >= 0
                    s1 = series1[1:end-lag]
                    s2 = series2[1+lag:end]
                else
                    s1 = series1[1-lag:end]
                    s2 = series2[1:end+lag]
                end
                
                if length(s1) > 0 && length(s2) > 0
                    corr = cor(s1, s2)
                    max_corr = max(abs(corr), max_corr)
                end
            end
            
            coupling[i,j] = max_corr
            coupling[j,i] = max_corr
        end
    end
    
    return coupling
end

# Example usage

# Climate oscillation periods in years
climate_observations = Dict(
    "ENSO" => 3.5,      # 2-7 year period
    "NAO" => 0.2,       # Intraseasonal to interannual  
    "PDO" => 20.0,      # 20-30 year period
    "AMO" => 65.0,      # 50-80 year period
    "IOD" => 2.0,       # Similar to ENSO
    "AAO" => 0.1,       # Weekly to monthly
    "QBO" => 2.3,       # ~28 months
    "MJO" => 0.15       # 30-60 days
)

# Create algebra with dimension from oscillation data
algebra = ClimateAlgebra(climate_observations)

println("Natural dimension from climate data: $(algebra.dimension)")
println("Clifford algebra dimension: 2^$(algebra.clifford_dim) = $(2^algebra.clifford_dim)")
println("Dimension calculated from $(length(climate_observations)) oscillation modes")

# Display strongest mode couplings
println("\nStrongest couplings (from period analysis):")
for i in 1:length(algebra.oscillation_names)
    for j in i+1:length(algebra.oscillation_names)
        if algebra.coupling_matrix[i,j] > 0.5
            println("  $(algebra.oscillation_names[i]) <-> $(algebra.oscillation_names[j]): $(round(algebra.coupling_matrix[i,j], digits=3))")
        end
    end
end

# Compute a resonance pattern
lat, lon, t = 15.0, -150.0, 100.0  # Tropical Pacific
resonance = compute_resonance_pattern(algebra, lat, lon, t)
println("\nResonance at ($(lat)°, $(lon)°) at t=$(t): $(resonance)")

# Notes:
# - Spatial patterns simplified (full EOF requires gridded data)
# - Coupling matrix needs long observational time series
# - Experimental Clifford algebra framework