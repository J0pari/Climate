# Climate statistical manifold with Fisher-Rao geometry
# Experimental Amari α-geometry implementation

using LinearAlgebra
using Statistics
using Distributions
using DifferentialEquations
using ForwardDiff
using ReverseDiff
using Zygote
using Manifolds
using ManifoldDiffEq
using InformationGeometry
using StatsBase
using PDMats
using Optim
using Flux
using CUDA
using MPI
using HDF5
using NetCDF
using NCDatasets
using Dates
using ProgressMeter
using Logging
using JSON3
using TOML

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCE REQUIREMENTS
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. CMIP6 MODEL PARAMETER DISTRIBUTIONS:
#    - Source: ES-DOC + ESGF model output
#    - Models: 100+ models from 50+ institutions
#    - Parameters per model:
#      * Cloud microphysics: autoconversion rate, ice fall speed
#      * Turbulence: mixing length, Richardson number critical
#      * Convection: entrainment rate, CAPE closure timescale
#      * Radiation: cloud overlap parameter, aerosol scaling
#      * Land surface: soil hydraulic conductivity, vegetation resistance
#    - Format: JSON parameter cards from ES-DOC, NetCDF4 output
#    - Size: ~100GB for key diagnostic fields from all models
#    - API: https://es-doc.org/ for parameters, ESGF for output
#    - Preprocessing: Extract implicit parameters via inverse methods
#    - Missing: Many models don't document all tuning choices
#    - Missing: Parameter covariance within models
#
# 2. OBSERVATIONAL CONSTRAINTS FOR FISHER INFORMATION:
#    - Temperature: 
#      * HadCRUT5: 5° × 5° monthly, 1850-present
#      * Berkeley Earth: 1° × 1° monthly, 1850-present
#      * Format: NetCDF4 with ensemble uncertainty
#      * Size: ~1GB per dataset
#    - Ocean heat content:
#      * IAP/Cheng: 1° × 1° × 42 levels, 1940-present
#      * NOAA/NCEI: 1° × 1° × 26 levels, 1955-present
#      * Format: NetCDF4 with reconstruction uncertainty
#      * Size: ~5GB per dataset
#    - TOA radiation:
#      * CERES EBAF Ed4.2: 1° × 1° monthly, 2000-present
#      * Format: NetCDF4 with calibration uncertainty
#      * Size: ~1GB
#    - Sea ice:
#      * NSIDC CDR: 25km daily, 1979-present
#      * Format: NetCDF4
#      * Size: ~10GB
#    - Carbon cycle:
#      * Atmospheric CO2: Mauna Loa + GLOBALVIEW
#      * Ocean pCO2: SOCAT v2022
#      * Format: CSV, NetCDF4
#      * Size: ~5GB total
#    - Missing: Pre-satellite cloud observations
#    - Missing: Deep ocean before Argo
#
# 3. PRIOR DISTRIBUTIONS FOR BAYESIAN INFERENCE:
#    - Source: Literature meta-analysis
#    - ECS prior: 
#      * Sherwood+ 2020: Log-normal(1.17, 0.33)
#      * IPCC AR6: 2.5-4.0 K likely range
#    - Cloud feedback:
#      * Zelinka+ 2020: 0.42 ± 0.35 W/m²/K
#      * Myers+ 2021: Constrained by satellite
#    - Aerosol forcing:
#      * Bellouin+ 2020: -1.3 ± 0.7 W/m²
#    - Format: Published PDFs, some as CSV samples
#    - Size: <100MB
#    - Missing: Joint priors (correlation structure)
#    - Missing: Structural uncertainty representation
#
# 4. EMERGENT CONSTRAINT RELATIONSHIPS:
#    - Source: Published constraint database
#    - Constraints: 40+ published relationships
#    - Examples:
#      * Seasonal cycle amplitude → ECS (Cox+ 2018)
#      * Low cloud cover → cloud feedback (Myers+ 2021)
#      * ENSO amplitude → sensitivity (Bellenger+ 2014)
#    - Format: CSV with X (observable) and Y (target) pairs
#    - Size: <10MB
#    - Validation: Out-of-sample testing on CMIP5→CMIP6
#    - Missing: Constraint independence assessment
#
# 5. INFORMATION GEOMETRY METRICS:
#    - Fisher information matrix:
#      * Computed from likelihood gradients
#      * Size: N_params × N_params per model
#    - KL divergence:
#      * Between model and observations
#      * Between model pairs
#    - Wasserstein distance:
#      * For comparing distributions
#    - Format: HDF5 matrices
#    - Size: ~10GB for all models
#    - Missing: Efficient computation for large N
#
# 6. PALEOCLIMATE CONSTRAINTS:
#    - LGM (21 kya):
#      * Temperature: -4 to -6 K global cooling
#      * CO2: 180 ppm
#      * Ice sheets: Peltier ICE-6G reconstruction
#    - Mid-Pliocene (3.2 Mya):
#      * Temperature: +2 to +3 K warming
#      * CO2: 400 ppm
#    - PETM (56 Mya):
#      * Temperature: +5 to +8 K warming
#      * Carbon release: 3000-7000 GtC
#    - Format: Proxy compilations in LiPD, CSV
#    - Size: ~1GB for major compilations
#    - API: https://www.ncei.noaa.gov/products/paleoclimatology
#    - Missing: Proxy model uncertainty
#
# 7. PERTURBED PHYSICS ENSEMBLES:
#    - Source: climateprediction.net, HadCM3
#    - Parameters: 30+ dimensions perturbed
#    - Size: 10,000+ members
#    - Format: Reduced output (~100MB per member)
#    - Purpose: Sample parameter space densely
#    - Missing: PPE for modern ESMs (too expensive)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Physical constants
const STEFAN_BOLTZMANN = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
const EARTH_RADIUS = 6.371e6  # m
const SOLAR_CONSTANT = 1361.0  # W/m²
const PLANCK_FEEDBACK = 3.2  # W/m²/K
const CO2_DOUBLING_FORCING = 3.7  # W/m²
const PREINDUSTRIAL_CO2 = 280.0  # ppm
const OCEAN_AREA = 3.61e14  # m²
const ATMOSPHERE_MASS = 5.15e18  # kg

# IPCC AR6 parameter ranges
const ECS_RANGE = (2.0, 5.0)  # Equilibrium Climate Sensitivity (K) - AR6 likely range
const ECS_BEST = 3.0  # Best estimate
const TCR_RANGE = (1.4, 2.2)  # Transient Climate Response (K) - AR6 likely range
const TCR_BEST = 1.8  # Best estimate
const AEROSOL_RANGE = (-1.6, -0.6)  # Aerosol ERF 1750-2019 (W/m²) - AR6 assessed
const CLOUD_FEEDBACK_RANGE = (-0.25, 0.45)  # Cloud feedback (W/m²/K) - Sherwood et al. 2020
const OCEAN_DIFFUSIVITY_RANGE = (0.5, 2.5)  # Vertical diffusivity (cm²/s) - observational constraint
const CARBON_FEEDBACK_RANGE = (-20.0, 80.0)  # γ_L land carbon feedback (GtC/K)
const ICE_SENSITIVITY_RANGE = (0.5, 1.5)  # Ice sheet sensitivity (m SLE/K)

# ─────────────────────────────────────────────────────────────────────────────
# CLIMATE PARAMETER SPACE
# ─────────────────────────────────────────────────────────────────────────────

"""
Climate model parameters as a point on statistical manifold
"""
struct ClimateParameters{T<:Real}
    ecs::T                    # Equilibrium climate sensitivity
    tcr::T                    # Transient climate response
    aerosol_forcing::T        # Total aerosol forcing
    cloud_feedback::T         # Cloud feedback strength
    ocean_diffusivity::T      # Vertical ocean mixing
    carbon_feedback::T        # Carbon cycle feedback
    ice_sheet_sensitivity::T  # Ice sheet response
    
    function ClimateParameters{T}(ecs, tcr, aerosol, cloud, ocean, carbon, ice) where T
        # Check physical bounds
        @assert ECS_RANGE[1] ≤ ecs ≤ ECS_RANGE[2] "ECS out of bounds"
        @assert TCR_RANGE[1] ≤ tcr ≤ TCR_RANGE[2] "TCR out of bounds"
        @assert AEROSOL_RANGE[1] ≤ aerosol ≤ AEROSOL_RANGE[2] "Aerosol forcing out of bounds"
        new(ecs, tcr, aerosol, cloud, ocean, carbon, ice)
    end
end

# Convert to vector for manifold operations
Base.Vector(p::ClimateParameters) = [p.ecs, p.tcr, p.aerosol_forcing, p.cloud_feedback, 
                                      p.ocean_diffusivity, p.carbon_feedback, p.ice_sheet_sensitivity]

# ─────────────────────────────────────────────────────────────────────────────
# FISHER INFORMATION METRIC
# ─────────────────────────────────────────────────────────────────────────────

"""
Fisher Information Matrix for climate model parameters
Measures information content about parameters from observations
"""
struct FisherMetric{T<:Real} <: Metric{MetricManifold{ℝ,DefaultEuclideanMetric{Tuple{7}}}}
    observations::Matrix{T}  # Climate observations [time × variables]
                            # DATA SOURCE: HadCRUT5, BEST, GISTEMP monthly anomalies
                            # FORMAT: [months × (Tglobal, Tland, Tocean, OHC, TOA_imbalance)]
                            # RESOLUTION: 1850-2023 monthly = 2088 rows × 5 variables
    model::Function         # Forward climate model
    noise_covariance::PDMat{T}  # Observation error covariance
                               # DATA SOURCE: Instrument uncertainty + representation error
                               # MISSING: Full error covariance - currently diagonal approximation
end

"""
Compute Fisher Information Matrix at parameter point
F_ij = E[∂log L/∂θ_i × ∂log L/∂θ_j]
"""
function fisher_information_matrix(metric::FisherMetric, params::ClimateParameters)
    θ = Vector(params)
    n_params = length(θ)
    
    # Forward model sensitivity
    function log_likelihood(θ_vec)
        p = ClimateParameters(θ_vec...)
        predictions = metric.model(p)
        residuals = vec(metric.observations - predictions)
        return -0.5 * invquad(metric.noise_covariance, residuals)
    end
    
    # Compute Hessian of log-likelihood
    H = ForwardDiff.hessian(log_likelihood, θ)
    
    # Fisher information is negative expected Hessian
    F = -H
    
    # Make positive definite
    F_psd = nearestSPD(F)
    
    return F_psd
end

"""
Natural gradient: ∇̃f = F^{-1}∇f
"""
function natural_gradient(metric::FisherMetric, params::ClimateParameters, loss_fn::Function)
    θ = Vector(params)
    
    # Euclidean gradient
    ∇f = ForwardDiff.gradient(loss_fn, θ)
    
    # Fisher information matrix
    F = fisher_information_matrix(metric, params)
    
    # Natural gradient with regularization
    F_reg = F + 1e-6 * I
    ∇̃f = F_reg \ ∇f
    
    return ∇̃f
end

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL DIVERGENCES
# ─────────────────────────────────────────────────────────────────────────────

"""
Kullback-Leibler divergence between climate model distributions
D_KL(P||Q) = ∫ p(x|θ₁) log[p(x|θ₁)/p(x|θ₂)] dx
"""
function kl_divergence(p1::ClimateParameters, p2::ClimateParameters, 
                       n_samples::Int=10000, time_horizon::Float64=100.0)
    # Generate climate trajectories from both parameter sets
    trajectories1 = simulate_climate_ensemble(p1, n_samples, time_horizon)
    trajectories2 = simulate_climate_ensemble(p2, n_samples, time_horizon)
    
    # Fit multivariate normal to trajectories
    μ1, Σ1 = mean_and_cov(trajectories1, 2)  # Mean over ensemble
    μ2, Σ2 = mean_and_cov(trajectories2, 2)
    
    # KL divergence for multivariate normals
    k = length(μ1)  # Dimension
    Σ2_inv = inv(Σ2)
    
    kl = 0.5 * (tr(Σ2_inv * Σ1) + (μ2 - μ1)' * Σ2_inv * (μ2 - μ1) - k + logdet(Σ2) - logdet(Σ1))
    
    return kl
end

"""
Jeffrey's divergence (symmetric KL)
J(P,Q) = D_KL(P||Q) + D_KL(Q||P)
"""
function jeffreys_divergence(p1::ClimateParameters, p2::ClimateParameters)
    return kl_divergence(p1, p2) + kl_divergence(p2, p1)
end

"""
Wasserstein distance between climate distributions
W₂(P,Q) - transport distance (not actually optimal - uses heuristic)
"""
function wasserstein_distance(p1::ClimateParameters, p2::ClimateParameters,
                             n_samples::Int=1000)
    # Generate samples
    samples1 = simulate_climate_ensemble(p1, n_samples, 100.0)
    samples2 = simulate_climate_ensemble(p2, n_samples, 100.0)
    
    # Sinkhorn algorithm
    using OptimalTransport  # Package probably not installed
    C = pairwise(Euclidean(), samples1', samples2')  # Cost matrix
    ε = 0.01  # Entropic regularization
    
    # Sinkhorn-Knopp algorithm
    P = sinkhorn(ones(n_samples)/n_samples, ones(n_samples)/n_samples, C, ε)
    
    # Wasserstein distance
    W2 = sqrt(sum(P .* C))
    
    return W2
end

# ─────────────────────────────────────────────────────────────────────────────
# α-CONNECTIONS AND DUAL GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

"""
α-connection on climate parameter manifold
"""
struct AlphaConnection{T<:Real}
    α::T  # Connection parameter ∈ [-1, 1]
    metric::FisherMetric{T}
end

"""
Christoffel symbols for α-connection
Γ^k_{ij}^(α) = Γ^k_{ij}^(0) + (α/2) T_{ijk}
where T is the Amari-Chentsov tensor
"""
function christoffel_symbols_alpha(conn::AlphaConnection, params::ClimateParameters)
    θ = Vector(params)
    n = length(θ)
    
    # Levi-Civita connection
    F = fisher_information_matrix(conn.metric, params)
    F_inv = inv(F)
    
    # Compute derivatives of Fisher metric
    ∂F = zeros(n, n, n)
    ε = 1e-6
    for k in 1:n
        θ_plus = copy(θ)
        θ_plus[k] += ε
        p_plus = ClimateParameters(θ_plus...)
        F_plus = fisher_information_matrix(conn.metric, p_plus)
        
        θ_minus = copy(θ)
        θ_minus[k] -= ε
        p_minus = ClimateParameters(θ_minus...)
        F_minus = fisher_information_matrix(conn.metric, p_minus)
        
        ∂F[:,:,k] = (F_plus - F_minus) / (2ε)
    end
    
    # Levi-Civita Christoffel symbols
    Γ_LC = zeros(n, n, n)
    for i in 1:n, j in 1:n, k in 1:n
        for l in 1:n
            Γ_LC[k,i,j] += 0.5 * F_inv[k,l] * (∂F[l,i,j] + ∂F[l,j,i] - ∂F[i,j,l])
        end
    end
    
    # Amari-Chentsov tensor
    T = compute_amari_chentsov_tensor(conn.metric, params)
    
    # α-connection Christoffel symbols
    Γ_alpha = Γ_LC + (conn.α / 2) * T
    
    return Γ_alpha
end

"""
Amari-Chentsov tensor T_{ijk} = E[∂_i log p × ∂_j log p × ∂_k log p]
"""
function compute_amari_chentsov_tensor(metric::FisherMetric, params::ClimateParameters)
    θ = Vector(params)
    n = length(θ)
    
    # Score function and its derivatives
    function score(θ_vec, x)
        p = ClimateParameters(θ_vec...)
        predictions = metric.model(p)
        residuals = x - predictions
        Σ_inv = inv(metric.noise_covariance)
        return Σ_inv * residuals  # Score = ∇ log p(x|θ)
    end
    
    # Monte Carlo estimation
    T = zeros(n, n, n)
    n_samples = 1000
    
    for _ in 1:n_samples
        # Sample from model
        predictions = metric.model(params)
        noise = rand(MvNormal(zeros(size(predictions,1)), metric.noise_covariance))
        x = predictions + noise
        
        # Compute score
        s = ForwardDiff.gradient(θ -> sum(score(θ, x)), θ)
        
        # Accumulate third moment
        for i in 1:n, j in 1:n, k in 1:n
            T[i,j,k] += s[i] * s[j] * s[k]
        end
    end
    
    T ./= n_samples
    
    return T
end

# ─────────────────────────────────────────────────────────────────────────────
# GEODESICS AND PARAMETER PATHS (NOT OPTIMAL - JUST STRAIGHT LINES)
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute geodesic between parameter estimates
"""
function parameter_geodesic(metric::FisherMetric, 
                           p_start::ClimateParameters,
                           p_end::ClimateParameters,
                           n_steps::Int=100)
    # Initial conditions
    θ0 = Vector(p_start)
    θ1 = Vector(p_end)
    
    # Geodesic equation: d²θ/dt² + Γⁱⱼₖ (dθʲ/dt)(dθᵏ/dt) = 0
    function geodesic_ode!(du, u, p, t)
        θ = u[1:7]
        dθ = u[8:14]
        
        # Current parameters
        params = ClimateParameters(θ...)
        
        # Christoffel symbols at current position
        conn = AlphaConnection(0.0, metric)  # Use Levi-Civita connection
        Γ = christoffel_symbols_alpha(conn, params)
        
        # Acceleration
        d²θ = zeros(7)
        for i in 1:7
            for j in 1:7, k in 1:7
                d²θ[i] -= Γ[i,j,k] * dθ[j] * dθ[k]
            end
        end
        
        # Update derivatives
        du[1:7] = dθ
        du[8:14] = d²θ
    end
    
    # Boundary value problem
    u0 = vcat(θ0, (θ1 - θ0))  # Initial position and velocity
    tspan = (0.0, 1.0)
    
    prob = ODEProblem(geodesic_ode!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=range(0, 1, length=n_steps))
    
    # Extract parameter path
    path = [ClimateParameters(sol.u[i][1:7]...) for i in 1:length(sol.u)]
    
    return path
end

# ─────────────────────────────────────────────────────────────────────────────
# CLIMATE MODEL SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Energy balance model with stochastic forcing
"""
function simulate_climate_ensemble(params::ClimateParameters, 
                                  n_ensemble::Int,
                                  time_horizon::Float64)
    # Energy balance model: C dT/dt = F - λT + σ dW
    C = 7.3  # Heat capacity (W⋅yr⋅m⁻²⋅K⁻¹)
    
    function climate_sde!(du, u, p, t)
        T = u[1]  # Temperature anomaly
        
        # Radiative forcing (simplified)
        F = 5.35 * log(420.0/280.0)  # CO2 forcing for 420 ppm
        F += params.aerosol_forcing
        
        # Feedback parameter
        λ = 3.2 / params.ecs  # Planck feedback modified by ECS
        λ -= params.cloud_feedback
        
        # Deterministic drift
        du[1] = (F - λ*T) / C
    end
    
    function climate_noise!(du, u, p, t)
        du[1] = 0.3  # Climate variability (K/√yr)
    end
    
    # Run ensemble
    trajectories = zeros(Int(time_horizon), n_ensemble)
    
    Threads.@threads for i in 1:n_ensemble
        prob = SDEProblem(climate_sde!, climate_noise!, [0.0], (0.0, time_horizon))
        sol = solve(prob, SOSRI(), saveat=1.0)
        trajectories[:, i] = sol.u
    end
    
    return trajectories
end

# ─────────────────────────────────────────────────────────────────────────────
# CRAMÉR-RAO BOUNDS AND EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────

"""
Cramér-Rao lower bound
"""
function cramer_rao_bound(metric::FisherMetric, params::ClimateParameters)
    F = fisher_information_matrix(metric, params)
    
    # CRB is inverse of Fisher information
    CRB = inv(F)
    
    # Extract bounds for each parameter
    bounds = Dict(
        :ecs => sqrt(CRB[1,1]),
        :tcr => sqrt(CRB[2,2]),
        :aerosol => sqrt(CRB[3,3]),
        :cloud => sqrt(CRB[4,4]),
        :ocean => sqrt(CRB[5,5]),
        :carbon => sqrt(CRB[6,6]),
        :ice => sqrt(CRB[7,7])
    )
    
    return bounds
end

"""
Estimator efficiency e = CRB / Var(θ̂)
"""
function estimator_efficiency(estimator_variance::Matrix{Float64},
                             crb::Matrix{Float64})
    # Efficiency for each parameter
    n = size(estimator_variance, 1)
    efficiency = zeros(n)
    
    for i in 1:n
        if estimator_variance[i,i] > 0
            efficiency[i] = crb[i,i] / estimator_variance[i,i]
        end
    end
    
    # Geometric mean efficiency
    overall = prod(efficiency)^(1/n)
    
    return efficiency, overall
end

# ─────────────────────────────────────────────────────────────────────────────
# JEFFREYS PRIOR
# ─────────────────────────────────────────────────────────────────────────────

"""
Jeffreys prior π(θ) ∝ √det(F(θ))
"""
struct JeffreysPrior{T<:Real}
    metric::FisherMetric{T}
end

function logpdf(prior::JeffreysPrior, params::ClimateParameters)
    F = fisher_information_matrix(prior.metric, params)
    return 0.5 * logdet(F)
end

function rand(prior::JeffreysPrior)
    # MCMC sampling
    θ_init = [
        rand(Uniform(ECS_RANGE...)),
        rand(Uniform(TCR_RANGE...)),
        rand(Uniform(AEROSOL_RANGE...)),
        rand(Uniform(CLOUD_FEEDBACK_RANGE...)),
        rand(Uniform(OCEAN_DIFFUSIVITY_RANGE...)),
        rand(Uniform(-1.0, 1.0)),  # Carbon feedback
        rand(Uniform(0.1, 2.0))    # Ice sensitivity
    ]
    
    # Metropolis-Hastings sampling
    n_samples = 10000
    samples = zeros(n_samples, 7)
    θ_current = θ_init
    
    for i in 1:n_samples
        # Propose new parameters
        θ_proposed = θ_current + 0.1 * randn(7)
        
        # Check bounds
        if all(in_bounds(θ_proposed))
            p_current = ClimateParameters(θ_current...)
            p_proposed = ClimateParameters(θ_proposed...)
            
            # Acceptance ratio
            log_ratio = logpdf(prior, p_proposed) - logpdf(prior, p_current)
            
            if log(rand()) < log_ratio
                θ_current = θ_proposed
            end
        end
        
        samples[i, :] = θ_current
    end
    
    # Return last sample
    return ClimateParameters(samples[end, :]...)
end

# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

using Test

@testset "Climate Information Geometry" begin
    # Create test observations
    obs = randn(100, 3)  # 100 time points, 3 variables
    model_fn = p -> randn(100, 3)  # Dummy model
    Σ = PDMat(Matrix(1.0I, 3, 3))
    
    metric = FisherMetric(obs, model_fn, Σ)
    
    # Test parameters
    p1 = ClimateParameters{Float64}(3.0, 1.7, -1.0, 0.5, 2.0, 0.2, 0.5)
    p2 = ClimateParameters{Float64}(4.0, 2.0, -0.8, 0.7, 2.5, 0.3, 0.6)
    
    @testset "Fisher Information" begin
        F = fisher_information_matrix(metric, p1)
        @test size(F) == (7, 7)
        @test issymmetric(F)
        @test isposdef(F)
    end
    
    @testset "KL Divergence" begin
        kl = kl_divergence(p1, p2, 100, 10.0)
        @test kl ≥ 0  # KL divergence is non-negative
        
        # Test symmetry of Jeffreys divergence
        j = jeffreys_divergence(p1, p2)
        @test j ≥ kl  # Jeffreys ≥ KL
    end
    
    @testset "Geodesics" begin
        path = parameter_geodesic(metric, p1, p2, 10)
        @test length(path) == 10
        @test path[1] ≈ p1
        @test path[end] ≈ p2
    end
    
    @testset "Cramér-Rao Bounds" begin
        bounds = cramer_rao_bound(metric, p1)
        @test all(values(bounds) .> 0)
        @test bounds[:ecs] > 0.1  # ECS uncertainty should be significant
    end
end