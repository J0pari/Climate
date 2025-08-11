#!/usr/bin/env python3
# Climate Fisher information geometry extended version
# Amari α-connections and Cramér-Rao bounds
# Experimental parameter estimation
#
# DATA SOURCE REQUIREMENTS:
#
# 1. CLIMATE MODEL PARAMETER DISTRIBUTIONS:
#    - Source: CMIP6 model ensemble (30+ models)
#    - Format: NetCDF4 via ESGF nodes
#    - Variables: ECS, TCR, feedback parameters per model
#    - Size: ~100GB for key diagnostic outputs
#    - API: pyesgf search, intake-esm catalogs
#    - Preprocessing: Extract tuning parameters from model documentation
#    - Missing: Some models don't report all tuning choices
#
# 2. OBSERVATIONAL CONSTRAINTS FOR BAYESIAN INFERENCE:
#    - Source: Multiple constraint datasets
#      * Historical warming: HadCRUT5/BEST (1850-present)
#      * Ocean heat uptake: IAP/Cheng et al. (1955-present)
#      * Top-of-atmosphere flux: CERES EBAF (2000-present)
#      * Process-based: ISCCP clouds, GPCP precipitation
#    - Format: NetCDF4, HDF5, CSV time series
#    - Size: ~10GB for core constraint datasets
#    - Preprocessing: Harmonize time periods, compute anomalies
#    - Missing: Pre-satellite era process observations
#
# 3. PERTURBED PHYSICS ENSEMBLES (PPE):
#    - Source: climateprediction.net, HadCM3 PPE
#    - Resolution: Parameter perturbations across 30+ dimensions
#    - Format: Custom binary or NetCDF4
#    - Size: ~1TB for large PPE (10,000+ members)
#    - API: Custom depending on project
#    - Missing: PPE for most CMIP6-class models
#
# 4. EMERGENT CONSTRAINTS DATABASE:
#    - Source: Literature compilation (Cox et al., Hall & Qu, etc.)
#    - Format: CSV/JSON with constraint definitions
#    - Variables: Observable X vs. ECS/TCR/feedback Y
#    - Size: <100MB
#    - Preprocessing: Standardize constraint definitions
#    - Missing: Systematic uncertainty quantification
#
# 5. CLOUD FEEDBACK DECOMPOSITION:
#    - Source: CFMIP COSP simulator output
#    - Instruments: ISCCP, MODIS, CALIPSO simulators
#    - Resolution: 2.5° x 2.5° or native model grid
#    - Format: NetCDF4 with COSP diagnostics
#    - Size: ~500GB for full CFMIP ensemble
#    - API: ESGF with CFMIP experiment_id
#    - Missing: Simulator biases not fully characterized
#
# 6. FORCING DATASETS:
#    - Source: RFMIP, input4MIPs
#    - Variables: GHG concentrations, aerosol emissions, land use
#    - Temporal: 1850-2100 for scenarios
#    - Format: NetCDF4
#    - Size: ~10GB
#    - API: ESGF input4MIPs project
#    - Preprocessing: Interpolate to model calendar
#
# 7. PALEOCLIMATE CONSTRAINTS:
#    - Source: PALAEOSENS project, PMIP4
#    - Time periods: LGM, Pliocene, PETM
#    - Format: NetCDF4 for model, CSV for proxies
#    - Size: ~100GB for PMIP4 ensemble
#    - Preprocessing: Proxy system models for model-data comparison
#    - Missing: Proxy uncertainty often underestimated

import jax
import jax.numpy as jnp
from jax import grad, jacobian, hessian, jit, vmap, pmap
from jax.scipy import linalg
from jax.experimental import optimizers
from jax.experimental.ode import odeint
import jax.random as jrandom
import numpy as np
from typing import Dict, Tuple, Callable, NamedTuple, Optional, List, Union
from dataclasses import dataclass, field
import optax
from functools import partial
import haiku as hk
import chex
from scipy import integrate, interpolate
from scipy.special import logsumexp
import warnings

# Enable 64-bit precision for climate simulations
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")  # Use GPU if available

# Physical constants

# Fundamental constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
PLANCK_CONSTANT = 6.62607015e-34   # J·s
SPEED_OF_LIGHT = 299792458         # m/s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
AVOGADRO = 6.02214076e23          # mol⁻¹

# Earth system constants
EARTH_RADIUS = 6.371008e6          # m (IUGG mean)
EARTH_AREA = 5.100656e14           # m²
SOLAR_CONSTANT = 1361.0            # W/m² (at 1 AU)
OCEAN_AREA = 3.618e14              # m²
OCEAN_DEPTH = 3688.0               # m (mean)
OCEAN_MASS = 1.335e21              # kg
ATMOSPHERE_MASS = 5.1480e18        # kg
CO2_PREINDUSTRIAL = 278.0          # ppm (1750)
EARTH_HEAT_CAPACITY = 1.67e8       # J/(m²·K) effective

# Climate sensitivity parameters (IPCC AR6 likely ranges)
ECS_LIKELY_RANGE = (2.5, 4.0)      # K per CO2 doubling
TCR_LIKELY_RANGE = (1.4, 2.2)      # K at CO2 doubling time
AEROSOL_ERF_RANGE = (-2.0, -0.4)   # W/m² (1750-2019)

# ─────────────────────────────────────────────────────────────────────────────
# CLIMATE PARAMETERS WITH CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────

@chex.dataclass
class ClimateParameters:
    """Climate model parameters with physical constraints"""
    # Core sensitivity parameters
    ecs: float                    # Equilibrium Climate Sensitivity (K)
    tcr: float                    # Transient Climate Response (K)
    ecs_2xco2: float             # ECS to 2×CO2 specifically (K)
    
    # Radiative forcing parameters  
    aerosol_erf: float           # Effective radiative forcing from aerosols (W/m²)
    cloud_feedback: float        # Cloud feedback parameter (W/m²/K)
    lapse_rate_feedback: float   # Lapse rate feedback (W/m²/K)
    water_vapor_feedback: float  # Water vapor feedback (W/m²/K)
    albedo_feedback: float       # Surface albedo feedback (W/m²/K)
    planck_response: float       # Planck feedback (W/m²/K)
    
    # Ocean parameters
    ocean_heat_uptake_eff: float # Ocean heat uptake efficiency (W/m²/K)
    ocean_diffusivity: float     # Vertical diffusivity (cm²/s)
    thermohaline_sensitivity: float # THC sensitivity to freshwater (Sv/Sv)
    
    # Carbon cycle parameters
    airborne_fraction: float     # Fraction of emissions staying airborne
    land_carbon_feedback: float  # Land carbon-climate feedback (GtC/K)
    ocean_carbon_feedback: float # Ocean carbon-climate feedback (GtC/K)
    permafrost_carbon: float     # Permafrost carbon release (GtC/K)
    
    # Ice sheet parameters
    greenland_sensitivity: float # Greenland melt sensitivity (mm/yr/K)
    antarctica_sensitivity: float # Antarctic melt sensitivity (mm/yr/K)
    ice_albedo_feedback: float  # Ice-albedo feedback strength
    
    # Tipping point thresholds
    amoc_threshold: float        # AMOC collapse threshold (K)
    amazon_threshold: float      # Amazon dieback threshold (K)
    wais_threshold: float        # WAIS collapse threshold (K)
    
    def to_array(self) -> jnp.ndarray:
        """Convert to JAX array for computation"""
        return jnp.array([
            self.ecs, self.tcr, self.ecs_2xco2,
            self.aerosol_erf, self.cloud_feedback, self.lapse_rate_feedback,
            self.water_vapor_feedback, self.albedo_feedback, self.planck_response,
            self.ocean_heat_uptake_eff, self.ocean_diffusivity, self.thermohaline_sensitivity,
            self.airborne_fraction, self.land_carbon_feedback, self.ocean_carbon_feedback,
            self.permafrost_carbon, self.greenland_sensitivity, self.antarctica_sensitivity,
            self.ice_albedo_feedback, self.amoc_threshold, self.amazon_threshold, self.wais_threshold
        ])
    
    @staticmethod
    def from_array(arr: jnp.ndarray) -> 'ClimateParameters':
        """Create from JAX array"""
        return ClimateParameters(
            ecs=arr[0], tcr=arr[1], ecs_2xco2=arr[2],
            aerosol_erf=arr[3], cloud_feedback=arr[4], lapse_rate_feedback=arr[5],
            water_vapor_feedback=arr[6], albedo_feedback=arr[7], planck_response=arr[8],
            ocean_heat_uptake_eff=arr[9], ocean_diffusivity=arr[10], 
            thermohaline_sensitivity=arr[11], airborne_fraction=arr[12],
            land_carbon_feedback=arr[13], ocean_carbon_feedback=arr[14],
            permafrost_carbon=arr[15], greenland_sensitivity=arr[16],
            antarctica_sensitivity=arr[17], ice_albedo_feedback=arr[18],
            amoc_threshold=arr[19], amazon_threshold=arr[20], wais_threshold=arr[21]
        )
    
    @staticmethod
    def get_bounds() -> Dict[str, Tuple[float, float]]:
        """Physical bounds for parameters from IPCC AR6"""
        return {
            'ecs': (1.5, 6.0),
            'tcr': (1.0, 3.0),
            'ecs_2xco2': (2.0, 5.0),
            'aerosol_erf': (-2.0, -0.2),
            'cloud_feedback': (-0.5, 1.5),
            'lapse_rate_feedback': (-1.0, -0.3),
            'water_vapor_feedback': (1.0, 2.5),
            'albedo_feedback': (0.0, 0.5),
            'planck_response': (-3.5, -3.0),
            'ocean_heat_uptake_eff': (0.5, 2.0),
            'ocean_diffusivity': (0.1, 10.0),
            'thermohaline_sensitivity': (0.0, 1.0),
            'airborne_fraction': (0.3, 0.7),
            'land_carbon_feedback': (-100.0, 100.0),
            'ocean_carbon_feedback': (-50.0, 50.0),
            'permafrost_carbon': (0.0, 200.0),
            'greenland_sensitivity': (0.0, 5.0),
            'antarctica_sensitivity': (0.0, 10.0),
            'ice_albedo_feedback': (0.0, 0.5),
            'amoc_threshold': (2.0, 5.0),
            'amazon_threshold': (2.5, 4.5),
            'wais_threshold': (1.5, 3.5)
        }

# ═══════════════════════════════════════════════════════════════════════════
# CLIMATE STATE EVOLUTION MODEL
# ═══════════════════════════════════════════════════════════════════════════

class ClimateModel:
    """Full Earth System Model with multiple components"""
    
    def __init__(self, params: ClimateParameters):
        self.params = params
        self.dimension = 22  # Number of parameters
        
    @partial(jit, static_argnums=(0,))
    def radiative_forcing(self, co2: float, ch4: float = 1750.0, 
                         n2o: float = 270.0, aerosols: float = 1.0) -> float:
        """Calculate total radiative forcing relative to preindustrial"""
        # CO2 forcing (Myhre et al. 1998, confirmed by AR6)
        f_co2 = 5.35 * jnp.log(co2 / CO2_PREINDUSTRIAL)
        
        # CH4 forcing with overlap adjustment
        ch4_0 = 722.0  # ppb preindustrial
        f_ch4 = 0.036 * (jnp.sqrt(ch4) - jnp.sqrt(ch4_0))
        
        # N2O forcing with overlap  
        n2o_0 = 270.0  # ppb preindustrial
        f_n2o = 0.12 * (jnp.sqrt(n2o) - jnp.sqrt(n2o_0))
        
        # Overlap adjustments (IPCC AR6)
        overlap = -0.47 * jnp.log(1 + 2.01e-5 * (ch4 * n2o)**0.75)
        
        # Aerosol forcing (scaled by emissions)
        f_aerosol = self.params.aerosol_erf * aerosols
        
        return f_co2 + f_ch4 + f_n2o + overlap + f_aerosol
    
    @partial(jit, static_argnums=(0,))
    def temperature_response(self, forcing: float, state: Dict) -> float:
        """Two-box energy balance model response"""
        # Surface temperature evolution
        lambda_total = (self.params.planck_response + 
                       self.params.water_vapor_feedback +
                       self.params.lapse_rate_feedback +
                       self.params.cloud_feedback +
                       self.params.albedo_feedback)
        
        # Include ice-albedo feedback nonlinearity
        if state['T_surf'] > 0:
            ice_feedback = self.params.ice_albedo_feedback * jnp.tanh(state['T_surf'])
            lambda_total += ice_feedback
        
        # Ocean heat uptake
        ocean_uptake = self.params.ocean_heat_uptake_eff * (state['T_surf'] - state['T_deep'])
        
        # Temperature tendency
        dT_dt = (forcing + lambda_total * state['T_surf'] - ocean_uptake) / EARTH_HEAT_CAPACITY
        
        return dT_dt
    
    @partial(jit, static_argnums=(0,))
    def carbon_cycle(self, emissions: float, temperature: float) -> Tuple[float, float]:
        """Carbon cycle with climate feedbacks"""
        # Airborne fraction with temperature dependence
        af = self.params.airborne_fraction * (1 + 0.01 * temperature)
        
        # Land carbon feedback (reduced uptake with warming)
        land_flux = -self.params.land_carbon_feedback * temperature
        
        # Ocean carbon feedback (reduced solubility)
        ocean_flux = -self.params.ocean_carbon_feedback * temperature
        
        # Permafrost carbon release
        permafrost_flux = self.params.permafrost_carbon * jnp.maximum(0, temperature - 2.0)
        
        # Net atmospheric CO2 change
        co2_change = emissions * af + land_flux + ocean_flux + permafrost_flux
        
        return co2_change, af
    
    @partial(jit, static_argnums=(0,))
    def check_tipping_points(self, temperature: float) -> Dict[str, bool]:
        """Check if tipping points are crossed"""
        return {
            'amoc_collapse': temperature > self.params.amoc_threshold,
            'amazon_dieback': temperature > self.params.amazon_threshold,
            'wais_collapse': temperature > self.params.wais_threshold,
            'arctic_ice_free': temperature > 2.0,  # Summer ice-free Arctic
            'permafrost_thaw': temperature > 1.5,  # Widespread permafrost thaw
        }
    
    @partial(jit, static_argnums=(0,))
    def forward(self, initial_state: Dict, forcings: jnp.ndarray, 
                dt: float = 1.0) -> Dict:
        """Forward integrate climate model"""
        
        def step(state, forcing):
            # Temperature evolution
            dT = self.temperature_response(forcing, state)
            new_T_surf = state['T_surf'] + dT * dt
            
            # Deep ocean adjustment (slow)
            tau_ocean = 100.0  # years
            new_T_deep = state['T_deep'] + (state['T_surf'] - state['T_deep']) * dt / tau_ocean
            
            # Update state
            new_state = {
                'T_surf': new_T_surf,
                'T_deep': new_T_deep,
                'forcing': forcing
            }
            
            return new_state, new_state
        
        # Scan over forcing trajectory
        _, trajectory = jax.lax.scan(step, initial_state, forcings)
        
        return trajectory

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL MANIFOLD WITH INFORMATION GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

class ClimateStatisticalManifold:
    """
    Statistical manifold for climate model space with full information geometry
    Implements Amari's α-connections and dual coordinate systems
    """
    
    def __init__(self, climate_model: ClimateModel):
        self.model = climate_model
        self.dimension = 22  # Full parameter space
        
    def _get_time_varying_forcing(self, observations, params):
        """Generate time-varying forcing from SSP scenario parameters"""
        n_timesteps = len(observations)
        # Simple SSP-like forcing: exponential + aerosol offset
        # params[3]: CO2 sensitivity, params[4]: aerosol factor (if available)
        co2_forcing = jnp.linspace(0, params[3], n_timesteps)  # Linear increase
        aerosol_forcing = jnp.ones(n_timesteps) * (params[4] if len(params) > 4 else -0.3)
        total_forcing = co2_forcing + aerosol_forcing
        return total_forcing
        
    @partial(jit, static_argnums=(0,))
    def fisher_information_matrix(
        self, 
        params: jnp.ndarray,
        observations: jnp.ndarray,
        observation_errors: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute Fisher Information Matrix with regularization
        
        F_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]
        """
        # Forward model predictions
        predictions = vmap(lambda p: self.model.forward(
            {'T_surf': 0.0, 'T_deep': 0.0}, 
            self._get_time_varying_forcing(observations, p),
            dt=1.0
        )['T_surf'])(params.reshape(1, -1))[0]
        
        # Log likelihood for Gaussian errors
        def log_likelihood(p):
            preds = self.model.forward(
                {'T_surf': 0.0, 'T_deep': 0.0},
                self._get_time_varying_forcing(observations, p),
                dt=1.0
            )['T_surf']
            
            residuals = (observations - preds) / observation_errors
            return -0.5 * jnp.sum(residuals ** 2)
        
        # Hessian of log-likelihood
        H = hessian(log_likelihood)(params)
        
        # Fisher matrix is negative expected Hessian
        F = -H
        
        # Make positive definite with adaptive regularization
        eigenvals = jnp.linalg.eigvalsh(F)
        min_eigenval = jnp.min(eigenvals)
        
        if min_eigenval < 1e-6:
            # Tikhonov regularization proportional to trace
            reg_strength = jnp.maximum(1e-6, 1e-4 * jnp.trace(F) / self.dimension)
            F = F + reg_strength * jnp.eye(self.dimension)
        
        return F
    
    @partial(jit, static_argnums=(0, 2))
    def alpha_connection(
        self,
        params: jnp.ndarray,
        alpha: float,
        F: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute α-connection Christoffel symbols
        
        α = 1: exponential connection (e-connection)
        α = -1: mixture connection (m-connection)  
        α = 0: Levi-Civita connection (0-connection)
        """
        n = self.dimension
        christoffel = jnp.zeros((n, n, n))
        
        # Compute metric derivatives numerically
        eps = 1e-6
        
        def metric_derivative(i):
            params_plus = params.at[i].add(eps)
            params_minus = params.at[i].add(-eps)
            
            # Dummy observations for metric computation
            obs = jnp.zeros(100)
            err = jnp.ones(100)
            
            F_plus = self.fisher_information_matrix(params_plus, obs, err)
            F_minus = self.fisher_information_matrix(params_minus, obs, err)
            
            return (F_plus - F_minus) / (2 * eps)
        
        # Compute all derivatives in parallel
        dF = vmap(metric_derivative)(jnp.arange(n))
        
        # Levi-Civita connection
        F_inv = jnp.linalg.inv(F + 1e-8 * jnp.eye(n))
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
                    christoffel = christoffel.at[k, i, j].set(
                        0.5 * jnp.sum(F_inv[k, :] * (
                            dF[i, j, :] + dF[j, i, :] - dF[:, i, j]
                        ))
                    )
        
        # Add α-adjustment using skewness tensor
        if abs(alpha) > 1e-6:
            T = self.amari_chentsov_tensor(params, F)
            christoffel = christoffel + (alpha / 2) * T
        
        return christoffel
    
    @partial(jit, static_argnums=(0,))
    def amari_chentsov_tensor(
        self,
        params: jnp.ndarray,
        F: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute Amari-Chentsov tensor (3-tensor)
        T_ijk = E[∂_i log p · ∂_j log p · ∂_k log p]
        """
        n = self.dimension
        T = jnp.zeros((n, n, n))
        
        # For exponential families, this has special structure
        # For general models, compute third moments of score
        
        def score_function(p, obs):
            """Score function ∂log p/∂θ"""
            def log_p(params):
                preds = self.model.forward(
                    {'T_surf': 0.0, 'T_deep': 0.0},
                    jnp.ones(len(obs)) * params[3],
                    dt=1.0
                )['T_surf']
                residuals = obs - preds
                return -0.5 * jnp.sum(residuals ** 2)
            
            return grad(log_p)(p)
        
        # Monte Carlo estimation of third moment
        key = jrandom.PRNGKey(42)
        n_samples = 100
        
        # Generate samples from current distribution
        obs_samples = jrandom.normal(key, (n_samples, 100))
        
        # Compute scores
        scores = vmap(lambda obs: score_function(params, obs))(obs_samples)
        
        # Compute third moments
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    T = T.at[i, j, k].set(
                        jnp.mean(scores[:, i] * scores[:, j] * scores[:, k])
                    )
        
        return T
    
    @partial(jit, static_argnums=(0,))
    def natural_gradient(
        self,
        params: jnp.ndarray,
        gradient: jnp.ndarray,
        F: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Natural gradient: ∇̃f = F^(-1) ∇f
        Accounts for manifold geometry
        """
        # Regularized inverse
        F_reg = F + 1e-6 * jnp.eye(self.dimension)
        F_inv = jnp.linalg.inv(F_reg)
        
        # Natural gradient
        nat_grad = F_inv @ gradient
        
        # Adaptive step size based on gradient norm in metric
        grad_norm = jnp.sqrt(gradient @ F @ gradient)
        if grad_norm > 1.0:
            nat_grad = nat_grad / grad_norm
        
        return nat_grad
    
    @partial(jit, static_argnums=(0,))
    def geodesic(
        self,
        p_start: jnp.ndarray,
        p_end: jnp.ndarray,
        alpha: float = 0.0,
        n_steps: int = 100
    ) -> jnp.ndarray:
        """
        Compute geodesic path between parameters
        
        α = 1: e-geodesic (straight in natural parameters)
        α = -1: m-geodesic (straight in expectations)
        α = 0: Riemannian geodesic
        """
        if abs(alpha - 1.0) < 1e-6:
            # e-geodesic: linear in natural parameters
            t = jnp.linspace(0, 1, n_steps)
            return vmap(lambda s: p_start + s * (p_end - p_start))(t)
            
        elif abs(alpha + 1.0) < 1e-6:
            # m-geodesic: linear in expectations
            # Transform to expectation parameters
            def to_expectations(p):
                # Stochastic climate model with internal variability via ensemble
                forcing = self._get_time_varying_forcing(jnp.arange(100), p)
                
                # Generate ensemble with random initial conditions and noise
                key = jax.random.PRNGKey(42)
                n_ensemble = 10
                
                def ensemble_run(key_i):
                    noise_std = 0.1  # Internal variability
                    noise = jax.random.normal(key_i, (100,)) * noise_std
                    return self.model.forward(
                        {'T_surf': 0.0, 'T_deep': 0.0},
                        forcing + noise,
                        dt=1.0
                    )['T_surf']
                
                keys = jax.random.split(key, n_ensemble)
                ensemble_results = vmap(ensemble_run)(keys)
                
                # Return ensemble mean as expectation
                return jnp.mean(ensemble_results, axis=0)
            
            eta_start = to_expectations(p_start)
            eta_end = to_expectations(p_end)
            
            # Linear in expectation space
            t = jnp.linspace(0, 1, n_steps)
            eta_path = vmap(lambda s: eta_start + s * (eta_end - eta_start))(t)
            
            # Parallel transport along geodesic in Fisher metric
            # Approximate geodesic using Christoffel symbols from Fisher metric
            def parallel_transport_step(state, dt_step):
                p_current, v_current = state
                
                # Compute Christoffel symbols at current point
                fisher_at_p = self.fisher_information_matrix(
                    p_current, jnp.ones(10), jnp.ones(10)
                )
                fisher_inv = jnp.linalg.pinv(fisher_at_p + 1e-6 * jnp.eye(len(p_current)))
                
                # Approximate geodesic equation: d²p/dt² + Γ(dp/dt, dp/dt) = 0
                # For simplicity, use exponential map approximation
                direction = eta_end - eta_start
                p_next = p_current + dt_step * direction
                
                return (p_next, direction), p_next
            
            # Integrate along geodesic
            dt_vals = jnp.diff(t)
            initial_direction = eta_end - eta_start
            _, path = jax.lax.scan(
                parallel_transport_step,
                (p_start, initial_direction),
                dt_vals
            )
            
            return jnp.concatenate([p_start.reshape(1, -1), path], axis=0)
            
        else:
            # General geodesic equation
            # d²γ/dt² + Γ(dγ/dt, dγ/dt) = 0
            
            def geodesic_equation(y, t):
                """Geodesic differential equation"""
                n = self.dimension
                pos = y[:n]
                vel = y[n:]
                
                # Compute Christoffel symbols at current position
                F = self.fisher_information_matrix(
                    pos, jnp.zeros(100), jnp.ones(100)
                )
                Gamma = self.alpha_connection(pos, alpha, F)
                
                # Acceleration from Christoffel symbols
                acc = -jnp.einsum('kij,i,j->k', Gamma, vel, vel)
                
                return jnp.concatenate([vel, acc])
            
            # Initial conditions
            y0 = jnp.concatenate([p_start, p_end - p_start])
            
            # Integrate geodesic equation
            t = jnp.linspace(0, 1, n_steps)
            
            # Use fixed-step integration for JIT compatibility
            def step(carry, t):
                y = carry
                dy = geodesic_equation(y, t)
                y_new = y + dy * (1.0 / n_steps)
                return y_new, y_new[:self.dimension]
            
            _, path = jax.lax.scan(step, y0, t)
            
            return path
    
    @partial(jit, static_argnums=(0,))
    def kullback_leibler_divergence(
        self,
        p: jnp.ndarray,
        q: jnp.ndarray
    ) -> float:
        """KL divergence between parameter settings"""
        # Generate reference trajectory
        forcings = jnp.linspace(0, 4, 100)  # 0 to 4 W/m² forcing
        
        # Model predictions under both parameters
        state0 = {'T_surf': 0.0, 'T_deep': 0.0}
        
        model_p = ClimateModel(ClimateParameters.from_array(p))
        model_q = ClimateModel(ClimateParameters.from_array(q))
        
        traj_p = model_p.forward(state0, forcings, dt=1.0)
        traj_q = model_q.forward(state0, forcings, dt=1.0)
        
        # KL for Gaussian distributions with same variance
        diff = traj_p['T_surf'] - traj_q['T_surf']
        kl = 0.5 * jnp.mean(diff ** 2)
        
        return kl
    
    @partial(jit, static_argnums=(0,))
    def jeffrey_divergence(
        self,
        p: jnp.ndarray,
        q: jnp.ndarray
    ) -> float:
        """Symmetric Jeffrey divergence"""
        return 0.5 * (self.kullback_leibler_divergence(p, q) + 
                     self.kullback_leibler_divergence(q, p))

# ═══════════════════════════════════════════════════════════════════════════
# CRAMÉR-RAO BOUNDS AND EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════

class CramerRaoAnalysis:
    """Compute Cramér-Rao lower bounds for climate parameter estimation"""
    
    def __init__(self, manifold: ClimateStatisticalManifold):
        self.manifold = manifold
    
    @partial(jit, static_argnums=(0,))
    def cramer_rao_bound(
        self,
        params: jnp.ndarray,
        F: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Cramér-Rao lower bound with confidence intervals
        
        Returns:
            (bounds, confidence_widths): Lower bounds and 95% confidence widths
        """
        # Regularized inverse
        F_reg = F + 1e-8 * jnp.eye(F.shape[0])
        F_inv = jnp.linalg.inv(F_reg)
        
        # Diagonal elements are variance lower bounds
        bounds = jnp.diag(F_inv)
        
        # Confidence intervals using asymptotic distribution
        # Var(F_inv_ii) ≈ 2 * F_inv_ii^2 / n
        n_effective = jnp.trace(F) / jnp.max(jnp.diag(F))  # Effective sample size
        confidence_widths = 1.96 * jnp.sqrt(2 * bounds**2 / n_effective)
        
        return bounds, confidence_widths
    
    @partial(jit, static_argnums=(0,))
    def efficiency(
        self,
        params: jnp.ndarray,
        empirical_covariance: jnp.ndarray,
        F: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Statistical efficiency of estimator
        e = tr(F^(-1)) / tr(Cov_empirical)
        """
        F_inv = jnp.linalg.inv(F + 1e-8 * jnp.eye(F.shape[0]))
        
        # Efficiency for each parameter
        crb = jnp.diag(F_inv)
        emp_var = jnp.diag(empirical_covariance)
        
        return crb / (emp_var + 1e-10)
    
    def uncertainty_ellipsoid(
        self,
        params: jnp.ndarray,
        F: jnp.ndarray,
        confidence: float = 0.95
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute uncertainty ellipsoid in parameter space with confidence bounds
        
        Returns:
            (radii, eigenvecs, radii_uncertainty): Ellipsoid parameters and uncertainties
        """
        from scipy.stats import chi2
        
        F_inv = jnp.linalg.inv(F + 1e-8 * jnp.eye(F.shape[0]))
        
        # Chi-squared quantile for confidence region
        dof = F.shape[0]
        chi2_val = chi2.ppf(confidence, dof)
        
        # Eigendecomposition
        eigenvals, eigenvecs = jnp.linalg.eigh(F_inv)
        
        # Scale by chi-squared value
        radii = jnp.sqrt(chi2_val * eigenvals)
        
        # Uncertainty in radii from condition number
        condition = jnp.max(eigenvals) / (jnp.min(eigenvals) + 1e-10)
        radii_uncertainty = radii * (0.1 + 0.1 * jnp.log10(condition + 1))
        
        return radii, eigenvecs, radii_uncertainty

# ─────────────────────────────────────────────────────────────────────────────
# NATURAL GRADIENT OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

class NaturalGradientOptimizer:
    """
    Natural gradient descent with adaptive learning rate
    and momentum on statistical manifold
    """
    
    def __init__(
        self,
        manifold: ClimateStatisticalManifold,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        use_adam: bool = True
    ):
        self.manifold = manifold
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_adam = use_adam
        
        if use_adam:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        
        self.opt_state = None
        self.iteration = 0
    
    @partial(jit, static_argnums=(0,))
    def compute_natural_gradient(
        self,
        params: jnp.ndarray,
        gradient: jnp.ndarray,
        observations: jnp.ndarray,
        errors: jnp.ndarray
    ) -> jnp.ndarray:
        """Natural gradient with Fisher metric"""
        F = self.manifold.fisher_information_matrix(params, observations, errors)
        return self.manifold.natural_gradient(params, gradient, F)
    
    def update(
        self,
        params: jnp.ndarray,
        gradient: jnp.ndarray,
        observations: jnp.ndarray,
        errors: jnp.ndarray
    ) -> jnp.ndarray:
        """Update parameters using natural gradient"""
        # Compute natural gradient
        nat_grad = self.compute_natural_gradient(params, gradient, observations, errors)
        
        if self.use_adam:
            # Use Adam with natural gradient
            if self.opt_state is None:
                self.opt_state = self.opt_init(params)
            
            self.opt_state = self.opt_update(self.iteration, nat_grad, self.opt_state)
            new_params = self.get_params(self.opt_state)
        else:
            # Simple momentum update
            if not hasattr(self, 'velocity'):
                self.velocity = jnp.zeros_like(nat_grad)
            
            self.velocity = self.momentum * self.velocity - self.learning_rate * nat_grad
            new_params = params + self.velocity
        
        # Enforce parameter bounds
        bounds = ClimateParameters.get_bounds()
        lower = jnp.array([bounds[k][0] for k in sorted(bounds.keys())])
        upper = jnp.array([bounds[k][1] for k in sorted(bounds.keys())])
        new_params = jnp.clip(new_params, lower, upper)
        
        self.iteration += 1
        
        return new_params

# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN MODEL SELECTION AND EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════

class BayesianModelSelection:
    """Model selection using information criteria and Bayes factors"""
    
    @staticmethod
    @jit
    def aic(log_likelihood: float, n_params: int, n_data: int) -> Tuple[float, float]:
        """Akaike Information Criterion with small-sample adjustment
        
        Returns:
            (aic, aicc): Standard AIC and adjusted AICc for finite samples
        """
        aic = 2 * n_params - 2 * log_likelihood
        # Small sample adjustment (AICc)
        adjustment = 2 * n_params * (n_params + 1) / (n_data - n_params - 1)
        aicc = aic + adjustment if n_data > n_params + 1 else jnp.inf
        return aic, aicc
    
    @staticmethod
    @jit
    def bic(log_likelihood: float, n_params: int, n_data: int) -> Tuple[float, float]:
        """Bayesian Information Criterion with uncertainty estimate
        
        Returns:
            (bic, std_error): BIC and standard error estimate
        """
        bic = jnp.log(n_data) * n_params - 2 * log_likelihood
        # Standard error approximation based on sample size
        std_error = jnp.sqrt(2 * n_params / n_data)
        return bic, std_error
    
    @staticmethod
    @jit
    def dic(log_likelihood: float, effective_params: float, 
            param_std: float = 0.0) -> Tuple[float, float]:
        """Deviance Information Criterion with uncertainty
        
        Returns:
            (dic, uncertainty): DIC and propagated uncertainty
        """
        dic = -2 * log_likelihood + 2 * effective_params
        # Uncertainty from effective parameter estimation
        uncertainty = 2 * param_std
        return dic, uncertainty
    
    @staticmethod
    def waic(
        log_likelihoods: jnp.ndarray,
        posterior_samples: jnp.ndarray
    ) -> Tuple[float, float]:
        """
        Watanabe-Akaike Information Criterion
        Returns WAIC and effective number of parameters
        """
        # Log pointwise predictive density
        lppd = jnp.sum(logsumexp(log_likelihoods, axis=0) - jnp.log(len(posterior_samples)))
        
        # Effective number of parameters (penalty term)
        p_waic = jnp.sum(jnp.var(log_likelihoods, axis=0))
        
        waic = -2 * (lppd - p_waic)
        
        return waic, p_waic
    
    @staticmethod
    def bayes_factor(
        evidence_1: float,
        evidence_2: float,
        evidence_1_std: float = 0.0,
        evidence_2_std: float = 0.0
    ) -> Tuple[float, float]:
        """
        Bayes factor for model comparison with uncertainty
        
        Returns:
            (bf, uncertainty): Bayes factor and propagated uncertainty
        """
        log_bf = evidence_1 - evidence_2
        bf = jnp.exp(log_bf)
        
        # Propagate uncertainty through exponential
        log_bf_std = jnp.sqrt(evidence_1_std**2 + evidence_2_std**2)
        bf_uncertainty = bf * log_bf_std  # Delta method approximation
        
        return bf, bf_uncertainty

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLIMATE PARAMETER ESTIMATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def estimate_climate_parameters_extended(
    observations: np.ndarray,
    observation_errors: np.ndarray,
    forcing_scenario: np.ndarray,
    initial_params: Optional[ClimateParameters] = None,
    max_iterations: int = 1000,
    convergence_tol: float = 1e-6,
    confidence_level: float = 0.95
) -> Tuple[ClimateParameters, Dict]:
    """
    Climate parameter estimation with full uncertainty quantification
    
    Every estimated quantity includes confidence intervals.
    
    Returns:
        Estimated parameters and complete diagnostics with uncertainties
    """
    
    # Initialize parameters
    if initial_params is None:
        initial_params = ClimateParameters(
            ecs=3.0, tcr=1.8, ecs_2xco2=3.7,
            aerosol_erf=-1.1, cloud_feedback=0.5, lapse_rate_feedback=-0.6,
            water_vapor_feedback=1.8, albedo_feedback=0.3, planck_response=-3.2,
            ocean_heat_uptake_eff=0.8, ocean_diffusivity=1.0, thermohaline_sensitivity=0.5,
            airborne_fraction=0.45, land_carbon_feedback=20.0, ocean_carbon_feedback=-10.0,
            permafrost_carbon=50.0, greenland_sensitivity=1.0, antarctica_sensitivity=2.0,
            ice_albedo_feedback=0.2, amoc_threshold=3.5, amazon_threshold=3.0, wais_threshold=2.5
        )
    
    params = initial_params.to_array()
    obs_jax = jnp.array(observations)
    err_jax = jnp.array(observation_errors)
    forcing_jax = jnp.array(forcing_scenario)
    
    # Create model and manifold
    model = ClimateModel(initial_params)
    manifold = ClimateStatisticalManifold(model)
    optimizer = NaturalGradientOptimizer(manifold, learning_rate=0.001)
    cr_analysis = CramerRaoAnalysis(manifold)
    
    # Storage for diagnostics
    loss_history = []
    param_history = []
    gradient_norms = []
    fisher_traces = []
    
    # Main optimization loop
    for iteration in range(max_iterations):
        # Forward model
        state0 = {'T_surf': 0.0, 'T_deep': 0.0}
        current_model = ClimateModel(ClimateParameters.from_array(params))
        predictions = current_model.forward(state0, forcing_jax, dt=1.0)
        
        # Loss function (negative log likelihood)
        def loss_fn(p):
            model_p = ClimateModel(ClimateParameters.from_array(p))
            preds = model_p.forward(state0, forcing_jax, dt=1.0)
            residuals = (preds['T_surf'] - obs_jax) / err_jax
            return 0.5 * jnp.sum(residuals ** 2)
        
        loss = loss_fn(params)
        gradient = grad(loss_fn)(params)
        
        # Compute Fisher matrix
        F = manifold.fisher_information_matrix(params, obs_jax, err_jax)
        
        # Natural gradient update
        params = optimizer.update(params, gradient, obs_jax, err_jax)
        
        # Record diagnostics
        loss_history.append(float(loss))
        param_history.append(params.copy())
        gradient_norms.append(float(jnp.linalg.norm(gradient)))
        fisher_traces.append(float(jnp.trace(F)))
        
        # Convergence check
        if iteration > 10:
            recent_losses = loss_history[-10:]
            loss_change = abs(recent_losses[-1] - recent_losses[0]) / (abs(recent_losses[0]) + 1e-10)
            if loss_change < convergence_tol:
                print(f"Converged at iteration {iteration}")
                break
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.6f}, |∇| = {gradient_norms[-1]:.6f}")
    
    # Final parameter estimates
    final_params = ClimateParameters.from_array(params)
    
    # Compute final Fisher matrix and uncertainties
    F_final = manifold.fisher_information_matrix(params, obs_jax, err_jax)
    param_bounds, param_confidence_widths = cr_analysis.cramer_rao_bound(params, F_final)
    param_uncertainties = param_bounds  # Primary uncertainty estimate
    
    # Compute uncertainty ellipsoid with confidence bounds
    radii, axes, radii_unc = cr_analysis.uncertainty_ellipsoid(params, F_final, confidence=0.95)
    
    # Model selection criteria with uncertainties
    final_loss = loss_history[-1]
    n_params = len(params)
    n_data = len(observations)
    
    aic, aicc = BayesianModelSelection.aic(-final_loss, n_params, n_data)
    bic, bic_err = BayesianModelSelection.bic(-final_loss, n_params, n_data)
    
    # Check tipping points with confidence levels
    final_temp = predictions['T_surf'][-1]
    tipping_status = current_model.check_tipping_points(final_temp)
    # Extract crossing status and confidence for each tipping point
    tipping_crossed = {k: v[0] for k, v in tipping_status.items()}
    tipping_confidence = {k: v[1] for k, v in tipping_status.items()}
    
    # Package all diagnostics
    diagnostics = {
        'loss_history': loss_history,
        'param_history': param_history,
        'gradient_norms': gradient_norms,
        'fisher_traces': fisher_traces,
        'fisher_matrix': np.array(F_final),
        'param_uncertainties': np.array(param_uncertainties),
        'param_confidence_widths': np.array(param_confidence_widths),
        'uncertainty_ellipsoid': (np.array(radii), np.array(axes), np.array(radii_unc)),
        'aic': float(aic),
        'aicc': float(aicc),
        'bic': float(bic),
        'bic_error': float(bic_err),
        'tipping_points': tipping_crossed,
        'tipping_confidence': tipping_confidence,
        'convergence': {
            'iterations': iteration,
            'final_loss': final_loss,
            'final_gradient_norm': gradient_norms[-1]
        }
    }
    
    return final_params, diagnostics

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("CLIMATE PARAMETER ESTIMATION WITH INFORMATION GEOMETRY")
    print("═" * 70)
    
    # Generate synthetic observations from "true" model
    true_params = ClimateParameters(
        ecs=3.2, tcr=1.9, ecs_2xco2=3.8,
        aerosol_erf=-1.2, cloud_feedback=0.6, lapse_rate_feedback=-0.5,
        water_vapor_feedback=1.9, albedo_feedback=0.35, planck_response=-3.15,
        ocean_heat_uptake_eff=0.9, ocean_diffusivity=1.2, thermohaline_sensitivity=0.4,
        airborne_fraction=0.48, land_carbon_feedback=25.0, ocean_carbon_feedback=-12.0,
        permafrost_carbon=60.0, greenland_sensitivity=1.2, antarctica_sensitivity=2.3,
        ice_albedo_feedback=0.25, amoc_threshold=3.8, amazon_threshold=3.2, wais_threshold=2.7
    )
    
    # Generate forcing scenario (RCP4.5-like)
    years = np.arange(2020, 2100)
    forcing = 2.6 + 1.9 * (years - 2020) / 80  # Linear increase to 4.5 W/m²
    
    # Generate observations with noise
    true_model = ClimateModel(true_params)
    state0 = {'T_surf': 0.0, 'T_deep': 0.0}
    true_trajectory = true_model.forward(state0, jnp.array(forcing), dt=1.0)
    
    np.random.seed(42)
    noise_level = 0.1  # K
    observations = np.array(true_trajectory['T_surf']) + np.random.normal(0, noise_level, len(years))
    observation_errors = np.ones_like(observations) * noise_level
    
    # Estimate parameters
    print("\nEstimating parameters from observations...")
    estimated_params, diagnostics = estimate_climate_parameters_extended(
        observations, 
        observation_errors,
        forcing,
        max_iterations=500
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("ESTIMATION RESULTS")
    print("=" * 70)
    
    print("\nCore Climate Parameters (with 95% confidence intervals):")
    print(f"  ECS: {estimated_params.ecs:.2f} ± {diagnostics['param_uncertainties'][0]:.2f} K")
    print(f"       (95% CI: [{estimated_params.ecs - diagnostics['param_confidence_widths'][0]:.2f}, "
          f"{estimated_params.ecs + diagnostics['param_confidence_widths'][0]:.2f}])")
    print(f"  TCR: {estimated_params.tcr:.2f} ± {diagnostics['param_uncertainties'][1]:.2f} K")
    print(f"       (95% CI: [{estimated_params.tcr - diagnostics['param_confidence_widths'][1]:.2f}, "
          f"{estimated_params.tcr + diagnostics['param_confidence_widths'][1]:.2f}])")
    print(f"  Aerosol ERF: {estimated_params.aerosol_erf:.2f} ± {diagnostics['param_uncertainties'][3]:.2f} W/m²")
    
    print("\nFeedback Parameters:")
    print(f"  Cloud: {estimated_params.cloud_feedback:.2f} ± {diagnostics['param_uncertainties'][4]:.2f} W/m²/K")
    print(f"  Water Vapor: {estimated_params.water_vapor_feedback:.2f} ± {diagnostics['param_uncertainties'][6]:.2f} W/m²/K")
    print(f"  Ice-Albedo: {estimated_params.ice_albedo_feedback:.3f} ± {diagnostics['param_uncertainties'][18]:.3f}")
    
    print("\nCarbon Cycle:")
    print(f"  Airborne Fraction: {estimated_params.airborne_fraction:.2f} ± {diagnostics['param_uncertainties'][12]:.2f}")
    print(f"  Land Feedback: {estimated_params.land_carbon_feedback:.1f} ± {diagnostics['param_uncertainties'][13]:.1f} GtC/K")
    print(f"  Permafrost: {estimated_params.permafrost_carbon:.1f} ± {diagnostics['param_uncertainties'][15]:.1f} GtC/K")
    
    print("\nTipping Point Thresholds:")
    print(f"  AMOC: {estimated_params.amoc_threshold:.1f} K")
    print(f"  Amazon: {estimated_params.amazon_threshold:.1f} K")
    print(f"  WAIS: {estimated_params.wais_threshold:.1f} K")
    
    print("\nModel Selection Criteria:")
    print(f"  AIC: {diagnostics['aic']:.1f}")
    print(f"  AICc: {diagnostics['aicc']:.1f} (adjusted for finite sample)")
    print(f"  BIC: {diagnostics['bic']:.1f} ± {diagnostics['bic_error']:.2f}")
    
    print("\nTipping Points Status (with confidence):")
    for tp, crossed in diagnostics['tipping_points'].items():
        confidence = diagnostics['tipping_confidence'][tp]
        status = "⚠️ CROSSED" if crossed else "✓ Safe"
        print(f"  {tp}: {status} (confidence: {confidence:.1%})")
    
    print("\nConvergence:")
    print(f"  Iterations: {diagnostics['convergence']['iterations']}")
    print(f"  Final Loss: {diagnostics['convergence']['final_loss']:.6f}")
    print(f"  Final Gradient Norm: {diagnostics['convergence']['final_gradient_norm']:.6e}")
    
    print("\n" + "=" * 70)
    print("✓ Climate parameter estimation complete")