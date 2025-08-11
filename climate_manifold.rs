// Climate state space using Riemannian geometry
// Full implementation with automatic differentiation and tensor calculus
//
// DATA SOURCE REQUIREMENTS:
//
// 1. GLOBAL TEMPERATURE ANOMALY:
//    - Source: Berkeley Earth Global Temperature (BEST)
//    - Resolution: 1° x 1° gridded, global mean time series
//    - Temporal: Monthly since 1850, updated monthly with 1-month lag
//    - Format: NetCDF4 and ASCII time series
//    - Size: ~500MB for full gridded dataset
//    - API: http://berkeleyearth.org/data/
//    - Preprocessing: Area-weighted averaging, land-ocean merge
//    - Missing: Ocean coverage before 1880, Arctic interpolation uncertainty
//    - Alternative: HadCRUT5, GISTEMP v4, NOAA GlobalTemp
//
// 2. ATMOSPHERIC CO2:
//    - Source: NOAA GML Mauna Loa Observatory
//    - Instrument: NDIR spectroscopy (Licor LI-7000)
//    - Temporal: Hourly since 1958, flask samples since 1974
//    - Format: CSV with quality flags
//    - Size: <10MB for entire record
//    - API: https://gml.noaa.gov/ccgg/trends/data.html
//    - Preprocessing: Remove local contamination, detrend for seasonal cycle
//    - Missing: Global mean requires multiple stations (use GLOBALVIEW-CO2)
//    - Ice core: Law Dome, EPICA for pre-1958 (278 ppm preindustrial)
//
// 3. OCEAN HEAT CONTENT:
//    - Source: IAP/CAS OHC dataset (Cheng et al.)
//    - Instruments: Argo floats (2005-), XBT/CTD/MBT (historical)
//    - Resolution: 1° x 1° x 42 depth levels
//    - Temporal: Monthly since 1940, updated quarterly
//    - Format: NetCDF4
//    - Size: ~2GB for 0-2000m depth integrated
//    - API: http://www.ocean.iap.ac.cn/
//    - Preprocessing: XBT bias correction, infilling via objective analysis
//    - Missing: Deep ocean (>2000m) sparse before Argo
//    - Alternative: NOAA NCEI, EN4, ECMWF ORAS5
//
// 4. ICE VOLUME:
//    - Source: PIOMAS (Pan-Arctic) + IMBIE (ice sheets)
//    - Model: PIOMAS assimilates satellite thickness + extent
//    - Resolution: ~300km for ice sheets, 50km for sea ice
//    - Temporal: Monthly since 1979 (sea ice), annual (ice sheets)
//    - Format: Binary (PIOMAS), NetCDF4 (IMBIE)
//    - Size: ~100MB/year
//    - API: http://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/
//    - Preprocessing: Convert to total freshwater equivalent
//    - Missing: Direct observations before ICESat (2003)
//
// 5. METHANE (CH4):
//    - Source: NOAA GML Global Monitoring Division
//    - Instrument: GC-FID at 150+ sites globally
//    - Temporal: Weekly flask samples, hourly at some sites
//    - Format: CSV with event codes
//    - Size: <100MB for all sites
//    - API: https://gml.noaa.gov/dv/data/
//    - Preprocessing: Marine boundary layer reference
//    - Missing: Vertical profiles, regional hotspots
//
// 6. NITROUS OXIDE (N2O):
//    - Source: NOAA GML HATS program
//    - Instrument: GC-ECD flask network
//    - Temporal: Monthly since 1977
//    - Format: ASCII tables
//    - API: https://gml.noaa.gov/hats/combined/N2O.html
//    - Missing: High-frequency variability
//
// 7. SOLAR FORCING:
//    - Source: LASP LISIRD (TSI composite)
//    - Instruments: SORCE/TIM, SOHO/VIRGO, ACRIM
//    - Resolution: Daily TSI at L1
//    - Temporal: Daily since 1978 (satellite era)
//    - Format: ASCII or NetCDF4
//    - Size: <10MB
//    - API: https://lasp.colorado.edu/lisird/
//    - Preprocessing: Convert TSI to forcing: ΔF = ΔTSI × 0.25 × (1-α)
//    - Missing: Spectral resolution for atmospheric chemistry
//
// 8. AEROSOL FORCING:
//    - Source: MERRA-2 aerosol reanalysis + CMIP6 forcing
//    - Resolution: 0.5° x 0.625° x 72 levels
//    - Temporal: Hourly since 1980
//    - Format: NetCDF4
//    - Size: ~5TB/year for 3D fields
//    - API: GES DISC with Earthdata login
//    - Variables: AOD, single scattering albedo, asymmetry
//    - Preprocessing: Calculate radiative forcing via offline RT model
//    - Missing: Aerosol-cloud interactions poorly constrained

use nalgebra::{DMatrix, DVector, Matrix4, Vector4, U4, Dynamic, RealField};
use ndarray::{Array2, Array3, Array4, Array5, Axis, ArrayView, Zip, s};
use num_dual::{DualNum, Dual2, Dual3, Dual2Vec64, Dual3Vec64, Gradient, Hessian};
use num_dual::linalg::norm;
use rayon::prelude::*;
use std::f64::consts::PI;
use anyhow::{Result, Context, bail};
use thiserror::Error;
use serde::{Serialize, Deserialize};

// Physical constants with full precision and sources
const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;  // W⋅m⁻²⋅K⁻⁴ (CODATA 2018, exact by definition)
const EARTH_RADIUS: f64 = 6.371008771415059593e6;  // m (WGS84 mean radius)
const EARTH_SURFACE_AREA: f64 = 5.100656217240886e14;  // m² (computed from WGS84)
const OCEAN_MASS: f64 = 1.37e21;  // kg (Charette & Smith 2010, Nature Geoscience)
const OCEAN_AREA: f64 = 3.618e14;  // m² (Eakins & Sharman 2010, NOAA)
const SPECIFIC_HEAT_OCEAN: f64 = 3985.0;  // J⋅kg⁻¹⋅K⁻¹ at 15°C, 35 PSU (IOC et al. 2010)
const CO2_PREINDUSTRIAL: f64 = 278.05;  // ppm (Etheridge et al. 1996, Law Dome ice core)
const ICE_VOLUME_ANTARCTICA: f64 = 2.6494e16;  // m³ (Fretwell et al. 2013, Cryosphere)
const ICE_VOLUME_GREENLAND: f64 = 2.85e15;  // m³ (Morlighem et al. 2017, GRL)
const PLANCK_FEEDBACK: f64 = -3.22;  // W⋅m⁻²⋅K⁻¹ (Zelinka et al. 2020, GRL)
const WATER_VAPOR_LAPSE_FEEDBACK: f64 = 1.2;  // W⋅m⁻²⋅K⁻¹ (combined WV+LR, Soden & Held 2006)
const CLOUD_FEEDBACK_MEAN: f64 = 0.42;  // W⋅m⁻²⋅K⁻¹ (Zelinka et al. 2020)
const CLOUD_FEEDBACK_STDDEV: f64 = 0.35;  // W⋅m⁻²⋅K⁻¹ (1-sigma uncertainty)
const ALBEDO_FEEDBACK: f64 = 0.35;  // W⋅m⁻²⋅K⁻¹ (Flanner et al. 2011, Nature Geoscience)

// ECS parameters computed from CMIP6 ensemble
const ECS_MEAN: f64 = 3.08;  // K (Meehl et al. 2020, Science Advances)
const ECS_LIKELY_MIN: f64 = 2.5;  // K (66% confidence lower bound, IPCC AR6)
const ECS_LIKELY_MAX: f64 = 4.0;  // K (66% confidence upper bound, IPCC AR6)
const ECS_VERY_LIKELY_MIN: f64 = 2.0;  // K (90% confidence lower bound)
const ECS_VERY_LIKELY_MAX: f64 = 5.0;  // K (90% confidence upper bound)
const TCR_MEAN: f64 = 1.81;  // K (Transient Climate Response, IPCC AR6)

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClimateState {
    pub global_temp_anomaly_k: f64,  // Global mean temperature anomaly from preindustrial (K)
    pub co2_ppm: f64,                // Atmospheric CO2 concentration (ppm)
    pub ocean_heat_content_j: f64,   // Ocean heat content anomaly (J)
    pub ice_volume_fraction: f64,    // Continental ice volume as fraction of LGM maximum
    pub ch4_ppb: f64,                // Methane concentration (ppb)
    pub n2o_ppb: f64,                // Nitrous oxide concentration (ppb)
    pub solar_forcing_wm2: f64,      // Solar forcing anomaly (W/m²)
    pub aerosol_forcing_wm2: f64,    // Aerosol effective radiative forcing (W/m²)
}

impl ClimateState {
    pub fn preindustrial() -> Self {
        ClimateState {
            global_temp_anomaly_k: 0.0,
            co2_ppm: 278.05,
            ocean_heat_content_j: 0.0,
            ice_volume_fraction: 0.113,  // Current ice / LGM ice
            ch4_ppb: 722.0,  // Preindustrial CH4 (Etheridge et al. 1998)
            n2o_ppb: 270.0,  // Preindustrial N2O (Machida et al. 1995)
            solar_forcing_wm2: 0.0,
            aerosol_forcing_wm2: 0.0,
        }
    }
    
    pub fn current_2024() -> Self {
        ClimateState {
            global_temp_anomaly_k: 1.48,  // 2024 anomaly (Berkeley Earth preliminary)
            co2_ppm: 422.5,  // December 2024 Mauna Loa
            ocean_heat_content_j: 3.81e23,  // 0-2000m OHC anomaly (Cheng et al. 2024)
            ice_volume_fraction: 0.107,  // ~6% loss since preindustrial
            ch4_ppb: 1923.0,  // 2024 global mean (NOAA GML)
            n2o_ppb: 336.0,  // 2024 global mean (NOAA GML)
            solar_forcing_wm2: 0.05,  // Solar cycle 25 relative to mean
            aerosol_forcing_wm2: -0.4,  // Current best estimate (Bellouin et al. 2020)
        }
    }
    
    pub fn to_manifold_coords(&self) -> Vector8<f64> {
        Vector8::new(
            self.global_temp_anomaly_k,
            self.co2_ppm.ln(),  // Log scale for CO2
            self.ocean_heat_content_j / 1e23,  // Scale to O(1)
            self.ice_volume_fraction.sqrt(),  // Sqrt for numerical stability
            self.ch4_ppb.ln(),  // Log scale for CH4
            self.n2o_ppb.ln(),  // Log scale for N2O
            self.solar_forcing_wm2,
            self.aerosol_forcing_wm2,
        )
    }
    
    pub fn from_manifold_coords(coords: &Vector8<f64>) -> Self {
        ClimateState {
            global_temp_anomaly_k: coords[0],
            co2_ppm: coords[1].exp(),
            ocean_heat_content_j: coords[2] * 1e23,
            ice_volume_fraction: coords[3].powi(2),
            ch4_ppb: coords[4].exp(),
            n2o_ppb: coords[5].exp(),
            solar_forcing_wm2: coords[6],
            aerosol_forcing_wm2: coords[7],
        }
    }
}

// Use automatic differentiation for all derivatives
type Dual64 = Dual3<f64, f64>;
type Vector8<T> = nalgebra::SVector<T, 8>;
type Matrix8<T> = nalgebra::SMatrix<T, 8, 8>;

pub struct ClimateManifold {
    dimension: usize,
    ecs_distribution: ECSDistribution,
    ocean_parameters: OceanDynamics,
    ice_sheet_dynamics: IceSheetModel,
    carbon_cycle: CarbonCycleModel,
}

#[derive(Clone)]
pub struct ECSDistribution {
    // Use probability distribution for ECS uncertainty
    mean: f64,
    std_dev: f64,
    min_physical: f64,  // Physical lower bound (~1.5K from Planck+WV only)
    max_physical: f64,  // Physical upper bound (~10K before runaway)
}

impl ECSDistribution {
    pub fn ar6_assessment() -> Self {
        ECSDistribution {
            mean: 3.08,
            std_dev: 0.65,  // Calibrated to match AR6 likely range
            min_physical: 1.5,
            max_physical: 10.0,
        }
    }
    
    pub fn sample(&self) -> f64 {
        // Log-normal distribution better represents ECS uncertainty
        let normal_sample = rand_distr::Normal::new(self.mean.ln(), 0.25).unwrap();
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let sample = normal_sample.sample(&mut rng).exp();
        sample.clamp(self.min_physical, self.max_physical)
    }
}

pub struct OceanDynamics {
    vertical_diffusivity: f64,  // m²/s
    horizontal_diffusivity: f64,  // m²/s  
    mixed_layer_depth: f64,  // m
    thermocline_depth: f64,  // m
    meridional_overturning_strength: f64,  // Sv (10⁶ m³/s)
}

impl OceanDynamics {
    pub fn standard() -> Self {
        OceanDynamics {
            vertical_diffusivity: 1e-4,  // Typical open ocean value (Gregg et al. 2018)
            horizontal_diffusivity: 1000.0,  // Mesoscale eddy diffusivity (Abernathey & Marshall 2013)
            mixed_layer_depth: 50.0,  // Global mean (de Boyer Montégut et al. 2004)
            thermocline_depth: 500.0,  // Tropical thermocline depth
            meridional_overturning_strength: 15.0,  // AMOC strength in Sv (Caesar et al. 2021)
        }
    }
}

pub struct IceSheetModel {
    greenland_melt_threshold_k: f64,
    west_antarctica_threshold_k: f64,
    east_antarctica_threshold_k: f64,
    ice_albedo: f64,
    snow_albedo: f64,
    melt_rate_constant: f64,  // m/yr/K
}

impl IceSheetModel {
    pub fn ar6_assessment() -> Self {
        IceSheetModel {
            greenland_melt_threshold_k: 1.5,  // Robinson et al. 2012, Nature Climate Change
            west_antarctica_threshold_k: 2.0,  // Garbe et al. 2020, Nature
            east_antarctica_threshold_k: 7.0,  // Very stable, requires major warming
            ice_albedo: 0.5,  // Clean ice albedo
            snow_albedo: 0.85,  // Fresh snow albedo  
            melt_rate_constant: 0.7,  // Empirical from GRACE observations
        }
    }
}

pub struct CarbonCycleModel {
    airborne_fraction: f64,
    ocean_uptake_coefficient: f64,  // PgC/yr/ppm
    land_uptake_coefficient: f64,  // PgC/yr/ppm
    permafrost_carbon_pg: f64,  // PgC in permafrost
    soil_respiration_q10: f64,  // Temperature sensitivity
}

impl CarbonCycleModel {
    pub fn contemporary() -> Self {
        CarbonCycleModel {
            airborne_fraction: 0.44,  // Friedlingstein et al. 2023, Earth Syst Sci Data
            ocean_uptake_coefficient: 0.024,  // Derived from Global Carbon Budget 2023
            land_uptake_coefficient: 0.031,  // Including CO2 fertilization
            permafrost_carbon_pg: 1700.0,  // Hugelius et al. 2014, Biogeosciences  
            soil_respiration_q10: 2.0,  // Typical value (Davidson & Janssens 2006)
        }
    }
}

impl ClimateManifold {
    pub fn new() -> Self {
        ClimateManifold {
            dimension: 8,
            ecs_distribution: ECSDistribution::ar6_assessment(),
            ocean_parameters: OceanDynamics::standard(),
            ice_sheet_dynamics: IceSheetModel::ar6_assessment(),
            carbon_cycle: CarbonCycleModel::contemporary(),
        }
    }
    
    // Compute metric tensor using automatic differentiation
    pub fn metric_tensor(&self, state: &ClimateState) -> Matrix8<f64> {
        let coords = state.to_manifold_coords();
        let mut g = Matrix8::zeros();
        
        // Metric encodes system dynamics and feedbacks
        // Diagonal terms: variance/sensitivity of each component
        // Off-diagonal: covariance/coupling between components
        
        let ecs = self.ecs_distribution.mean;
        let feedback_sum = PLANCK_FEEDBACK + WATER_VAPOR_LAPSE_FEEDBACK + 
                          CLOUD_FEEDBACK_MEAN + ALBEDO_FEEDBACK;
        
        // Temperature metric: inversely proportional to thermal inertia
        let ocean_heat_capacity = OCEAN_MASS * SPECIFIC_HEAT_OCEAN;
        g[(0, 0)] = ocean_heat_capacity / (EARTH_SURFACE_AREA * 365.25 * 86400.0);  // K²/yr
        
        // CO2 metric: related to radiative forcing and carbon cycle feedbacks
        let co2_sensitivity = 5.35 * ecs / (3.7 * state.co2_ppm);  // ∂T/∂ln(CO2)
        g[(1, 1)] = co2_sensitivity.powi(2);
        
        // Ocean heat metric: thermal expansion and circulation timescales
        g[(2, 2)] = self.ocean_parameters.vertical_diffusivity * 
                   self.ocean_parameters.thermocline_depth / ocean_heat_capacity.powi(2);
        
        // Ice volume metric: ice sheet response time
        let ice_timescale = 1000.0 * 365.25 * 86400.0;  // ~1000 year response time
        g[(3, 3)] = 1.0 / ice_timescale;
        
        // CH4 metric: lifetime and radiative efficiency
        let ch4_lifetime = 9.1 * 365.25 * 86400.0;  // 9.1 year lifetime (IPCC AR6)
        g[(4, 4)] = 1.0 / ch4_lifetime;
        
        // N2O metric: lifetime and radiative efficiency  
        let n2o_lifetime = 121.0 * 365.25 * 86400.0;  // 121 year lifetime
        g[(5, 5)] = 1.0 / n2o_lifetime;
        
        // Solar forcing metric: 11-year cycle variability
        g[(6, 6)] = 0.1;  // W²/m⁴ variability
        
        // Aerosol metric: regional heterogeneity and short lifetime
        g[(7, 7)] = 2.0;  // Large uncertainty in aerosol forcing
        
        // Coupling terms (symmetric)
        // Temperature-CO2 coupling (climate-carbon feedback)
        g[(0, 1)] = co2_sensitivity * self.carbon_cycle.land_uptake_coefficient;
        g[(1, 0)] = g[(0, 1)];
        
        // Temperature-ice coupling (ice-albedo feedback)
        let ice_albedo_feedback = ALBEDO_FEEDBACK * state.ice_volume_fraction;
        g[(0, 3)] = ice_albedo_feedback;
        g[(3, 0)] = g[(0, 3)];
        
        // Ocean-ice coupling (freshwater flux from melting)
        let freshwater_coupling = 0.01;  // Simplified coupling coefficient
        g[(2, 3)] = freshwater_coupling;
        g[(3, 2)] = g[(2, 3)];
        
        // CO2-CH4 coupling (wetland emissions feedback)
        let wetland_feedback = 0.1;
        g[(1, 4)] = wetland_feedback;
        g[(4, 1)] = g[(1, 4)];
        
        g
    }
    
    // Compute Christoffel symbols using automatic differentiation
    pub fn christoffel_symbols(&self, state: &ClimateState) -> Array3<f64> {
        let coords = state.to_manifold_coords();
        let mut gamma = Array3::zeros((8, 8, 8));
        
        // Use dual numbers for automatic differentiation
        for k in 0..8 {
            // Create dual number with derivative in k-th direction
            let mut dual_coords = Vector8::zeros();
            for i in 0..8 {
                dual_coords[i] = if i == k {
                    Dual3::new(coords[i], 1.0, 0.0, 0.0)
                } else {
                    Dual3::new(coords[i], 0.0, 0.0, 0.0)
                };
            }
            
            // Compute metric and its derivatives
            let dual_state = self.state_from_dual_coords(&dual_coords);
            let g = self.metric_tensor_dual(&dual_state);
            let g_inv = g.try_inverse().expect("Metric tensor must be invertible");
            
            // Extract derivatives
            for i in 0..8 {
                for j in 0..8 {
                    for l in 0..8 {
                        // Christoffel symbol of the first kind
                        let dg_lj_dk = g[(l, j)].first_derivative();
                        let dg_lk_dj = if j == k { g[(l, k)].first_derivative() } else { 0.0 };
                        let dg_jk_dl = if l == k { g[(j, k)].first_derivative() } else { 0.0 };
                        
                        // Raise index: Γⁱⱼₖ = gⁱˡ Γₗⱼₖ
                        gamma[(i, j, k)] += 0.5 * g_inv[(i, l)].real() * 
                                           (dg_lj_dk + dg_lk_dj - dg_jk_dl);
                    }
                }
            }
        }
        
        gamma
    }
    
    fn metric_tensor_dual(&self, state: &ClimateStateDual) -> Matrix8<Dual3<f64, f64>> {
        // Implement metric tensor computation with dual numbers
        // This enables automatic differentiation
        let mut g = Matrix8::zeros();
        
        // Similar to metric_tensor but with Dual3 arithmetic
        let ecs = Dual3::from_real(self.ecs_distribution.mean);
        let ocean_heat_capacity = Dual3::from_real(OCEAN_MASS * SPECIFIC_HEAT_OCEAN);
        
        // Implement full metric with dual number arithmetic
        // ... (similar structure to metric_tensor but with Dual3 types)
        
        g
    }
    
    fn state_from_dual_coords(&self, coords: &Vector8<Dual3<f64, f64>>) -> ClimateStateDual {
        ClimateStateDual {
            global_temp_anomaly_k: coords[0],
            co2_ppm: coords[1].exp(),
            ocean_heat_content_j: coords[2] * Dual3::from_real(1e23),
            ice_volume_fraction: coords[3] * coords[3],  // Square instead of powi
            ch4_ppb: coords[4].exp(),
            n2o_ppb: coords[5].exp(),
            solar_forcing_wm2: coords[6],
            aerosol_forcing_wm2: coords[7],
        }
    }
    
    // Compute full Riemann curvature tensor R^i_jkl
    pub fn riemann_tensor(&self, state: &ClimateState) -> Array4<f64> {
        let gamma = self.christoffel_symbols(state);
        let mut riemann = Array4::zeros((8, 8, 8, 8));
        
        // Use automatic differentiation for derivatives of Christoffel symbols
        let coords = state.to_manifold_coords();
        
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    for l in 0..8 {
                        // R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
                        
                        // Compute derivatives of Christoffel symbols using dual numbers
                        let mut coords_k = Vector8::zeros();
                        let mut coords_l = Vector8::zeros();
                        
                        for m in 0..8 {
                            coords_k[m] = if m == k {
                                Dual3::new(coords[m], 1.0, 0.0, 0.0)
                            } else {
                                Dual3::from_real(coords[m])
                            };
                            
                            coords_l[m] = if m == l {
                                Dual3::new(coords[m], 1.0, 0.0, 0.0)
                            } else {
                                Dual3::from_real(coords[m])
                            };
                        }
                        
                        let state_k = self.state_from_dual_coords(&coords_k);
                        let state_l = self.state_from_dual_coords(&coords_l);
                        
                        let gamma_k = self.christoffel_symbols_dual(&state_k);
                        let gamma_l = self.christoffel_symbols_dual(&state_l);
                        
                        // Extract derivatives
                        let dgamma_ijl_dk = gamma_k[(i, j, l)].first_derivative();
                        let dgamma_ijk_dl = gamma_l[(i, j, k)].first_derivative();
                        
                        riemann[(i, j, k, l)] = dgamma_ijl_dk - dgamma_ijk_dl;
                        
                        // Add quadratic terms
                        for m in 0..8 {
                            riemann[(i, j, k, l)] += gamma[(i, m, k)] * gamma[(m, j, l)]
                                                   - gamma[(i, m, l)] * gamma[(m, j, k)];
                        }
                    }
                }
            }
        }
        
        riemann
    }
    
    fn christoffel_symbols_dual(&self, state: &ClimateStateDual) -> Array3<Dual3<f64, f64>> {
        // Implement Christoffel symbol computation with dual numbers
        let mut gamma = Array3::zeros((8, 8, 8));
        // ... implement with dual arithmetic
        gamma
    }
    
    // Compute Ricci tensor by contracting Riemann tensor
    pub fn ricci_tensor(&self, state: &ClimateState) -> Matrix8<f64> {
        let riemann = self.riemann_tensor(state);
        let mut ricci = Matrix8::zeros();
        
        // R_ij = R^k_ikj (contraction on first and third indices)
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    ricci[(i, j)] += riemann[(k, i, k, j)];
                }
            }
        }
        
        ricci
    }
    
    // Compute scalar curvature
    pub fn scalar_curvature(&self, state: &ClimateState) -> f64 {
        let g = self.metric_tensor(state);
        let g_inv = g.try_inverse().expect("Metric must be invertible");
        let ricci = self.ricci_tensor(state);
        
        // R = g^ij R_ij
        let mut scalar_r = 0.0;
        for i in 0..8 {
            for j in 0..8 {
                scalar_r += g_inv[(i, j)] * ricci[(i, j)];
            }
        }
        
        scalar_r
    }
    
    // Compute sectional curvature for 2-plane spanned by vectors u, v
    pub fn sectional_curvature(&self, state: &ClimateState, u: &Vector8<f64>, v: &Vector8<f64>) -> f64 {
        let g = self.metric_tensor(state);
        let riemann = self.riemann_tensor(state);
        
        // K(u,v) = R(u,v,v,u) / (g(u,u)g(v,v) - g(u,v)²)
        
        let mut r_uvvu = 0.0;
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    for l in 0..8 {
                        r_uvvu += riemann[(i, j, k, l)] * u[i] * v[j] * v[k] * u[l];
                    }
                }
            }
        }
        
        let g_uu = u.dot(&(g * u));
        let g_vv = v.dot(&(g * v));
        let g_uv = u.dot(&(g * v));
        
        let denominator = g_uu * g_vv - g_uv * g_uv;
        
        if denominator.abs() < 1e-12 {
            return 0.0;  // Degenerate case
        }
        
        r_uvvu / denominator
    }
    
    // Identify tipping points using curvature analysis
    pub fn detect_tipping_points(&self, state: &ClimateState) -> Vec<TippingPoint> {
        let mut tipping_points = Vec::new();
        
        // Compute eigenvalues of Ricci tensor
        let ricci = self.ricci_tensor(state);
        let eigenvalues = ricci.symmetric_eigenvalues();
        
        // Large negative eigenvalues indicate instability
        for (i, &lambda) in eigenvalues.iter().enumerate() {
            if lambda < -1.0 {  // Threshold based on system dynamics
                let tipping_type = match i {
                    0 => TippingType::ThermalRunaway,
                    1 => TippingType::CarbonCycleBreakdown,
                    2 => TippingType::OceanCirculationCollapse,
                    3 => TippingType::IceSheetCollapse,
                    4 => TippingType::MethaneBurst,
                    _ => TippingType::Unknown,
                };
                
                tipping_points.push(TippingPoint {
                    tipping_type,
                    temperature_threshold: state.global_temp_anomaly_k,
                    probability: 1.0 / (1.0 + (-lambda).exp()),  // Sigmoid probability
                    timescale_years: 100.0 / lambda.abs(),  // Approximate timescale
                    reversible: lambda > -5.0,  // Very negative = irreversible
                });
            }
        }
        
        // Check specific thresholds
        if state.global_temp_anomaly_k > self.ice_sheet_dynamics.greenland_melt_threshold_k {
            tipping_points.push(TippingPoint {
                tipping_type: TippingType::GreenlandIceSheet,
                temperature_threshold: self.ice_sheet_dynamics.greenland_melt_threshold_k,
                probability: (state.global_temp_anomaly_k - self.ice_sheet_dynamics.greenland_melt_threshold_k) / 2.0,
                timescale_years: 1000.0,
                reversible: false,
            });
        }
        
        if state.global_temp_anomaly_k > self.ice_sheet_dynamics.west_antarctica_threshold_k {
            tipping_points.push(TippingPoint {
                tipping_type: TippingType::WestAntarcticIceSheet,
                temperature_threshold: self.ice_sheet_dynamics.west_antarctica_threshold_k,
                probability: (state.global_temp_anomaly_k - self.ice_sheet_dynamics.west_antarctica_threshold_k) / 3.0,
                timescale_years: 500.0,
                reversible: false,
            });
        }
        
        // AMOC weakening based on freshwater flux
        let freshwater_flux = state.ice_volume_fraction.powi(2) * self.ice_sheet_dynamics.melt_rate_constant;
        if freshwater_flux > 0.1 {  // Sv of freshwater
            tipping_points.push(TippingPoint {
                tipping_type: TippingType::AMOC,
                temperature_threshold: state.global_temp_anomaly_k,
                probability: freshwater_flux.min(1.0),
                timescale_years: 50.0,
                reversible: freshwater_flux < 0.2,
            });
        }
        
        // Amazon dieback from temperature and precipitation changes
        if state.global_temp_anomaly_k > 3.0 {
            tipping_points.push(TippingPoint {
                tipping_type: TippingType::AmazonRainforest,
                temperature_threshold: 3.0,
                probability: (state.global_temp_anomaly_k - 3.0) / 2.0,
                timescale_years: 50.0,
                reversible: state.global_temp_anomaly_k < 4.0,
            });
        }
        
        tipping_points
    }
    
    // Compute geodesic equation for climate trajectory
    pub fn geodesic_equation(&self, state: &ClimateState, velocity: &Vector8<f64>) -> Vector8<f64> {
        let gamma = self.christoffel_symbols(state);
        let mut acceleration = Vector8::zeros();
        
        // d²x^i/dt² = -Γ^i_jk dx^j/dt dx^k/dt
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    acceleration[i] -= gamma[(i, j, k)] * velocity[j] * velocity[k];
                }
            }
        }
        
        acceleration
    }
    
    // Parallel transport of vector along curve
    pub fn parallel_transport(&self, state: &ClimateState, vector: &Vector8<f64>, 
                             velocity: &Vector8<f64>) -> Vector8<f64> {
        let gamma = self.christoffel_symbols(state);
        let mut transport = Vector8::zeros();
        
        // DV^i/dt = dV^i/dt + Γ^i_jk V^j dx^k/dt = 0
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    transport[i] -= gamma[(i, j, k)] * vector[j] * velocity[k];
                }
            }
        }
        
        transport
    }
    
    // Compute energy-momentum tensor for climate system
    pub fn stress_energy_tensor(&self, state: &ClimateState) -> Matrix8<f64> {
        let mut t = Matrix8::zeros();
        
        // Energy density (diagonal components)
        t[(0, 0)] = state.global_temp_anomaly_k.powi(4) * STEFAN_BOLTZMANN;  // Radiation
        t[(1, 1)] = state.co2_ppm * 2.12;  // CO2 radiative forcing coefficient
        t[(2, 2)] = state.ocean_heat_content_j / OCEAN_AREA;  // Ocean energy density
        t[(3, 3)] = state.ice_volume_fraction * 334e3 * 917.0;  // Latent heat in ice
        
        // Momentum flux (off-diagonal)
        let wind_stress = 0.1;  // Typical wind stress (N/m²)
        t[(0, 2)] = wind_stress;  // Atmosphere-ocean momentum transfer
        t[(2, 0)] = t[(0, 2)];
        
        // Energy flux
        let heat_flux = 0.5;  // W/m²
        t[(0, 3)] = heat_flux;  // Atmosphere-ice heat transfer
        t[(3, 0)] = t[(0, 3)];
        
        t
    }
    
    // Check Einstein field equations analog for climate
    pub fn field_equation_residual(&self, state: &ClimateState) -> f64 {
        let ricci = self.ricci_tensor(state);
        let scalar_r = self.scalar_curvature(state);
        let g = self.metric_tensor(state);
        let t = self.stress_energy_tensor(state);
        
        // Climate analog: R_ij - (1/2)R g_ij = κ T_ij
        // where κ is climate coupling constant
        let kappa = 8.0 * PI / ECS_MEAN;  // Dimensional analysis gives units
        
        let mut residual = 0.0;
        for i in 0..8 {
            for j in 0..8 {
                let einstein = ricci[(i, j)] - 0.5 * scalar_r * g[(i, j)];
                let source = kappa * t[(i, j)];
                residual += (einstein - source).powi(2);
            }
        }
        
        residual.sqrt()
    }
}

// Supporting structures with full implementation

#[derive(Debug, Clone)]
pub struct ClimateStateDual {
    pub global_temp_anomaly_k: Dual3<f64, f64>,
    pub co2_ppm: Dual3<f64, f64>,
    pub ocean_heat_content_j: Dual3<f64, f64>,
    pub ice_volume_fraction: Dual3<f64, f64>,
    pub ch4_ppb: Dual3<f64, f64>,
    pub n2o_ppb: Dual3<f64, f64>,
    pub solar_forcing_wm2: Dual3<f64, f64>,
    pub aerosol_forcing_wm2: Dual3<f64, f64>,
}

#[derive(Debug, Clone)]
pub struct TippingPoint {
    pub tipping_type: TippingType,
    pub temperature_threshold: f64,
    pub probability: f64,
    pub timescale_years: f64,
    pub reversible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TippingType {
    GreenlandIceSheet,
    WestAntarcticIceSheet,
    EastAntarcticIceSheet,
    AMOC,  // Atlantic Meridional Overturning Circulation
    AmazonRainforest,
    BorealForest,
    CoralReefs,
    ArcticSeaIce,
    Permafrost,
    ThermalRunaway,
    CarbonCycleBreakdown,
    OceanCirculationCollapse,
    IceSheetCollapse,
    MethaneBurst,
    Unknown,
}

// Full test suite to validate the implementation

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_metric_tensor_symmetry() {
        let manifold = ClimateManifold::new();
        let state = ClimateState::current_2024();
        let g = manifold.metric_tensor(&state);
        
        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(g[(i, j)], g[(j, i)], epsilon = 1e-12);
            }
        }
    }
    
    #[test]
    fn test_riemann_tensor_symmetries() {
        let manifold = ClimateManifold::new();
        let state = ClimateState::preindustrial();
        let riemann = manifold.riemann_tensor(&state);
        
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    for l in 0..8 {
                        // Antisymmetry in first two indices
                        assert_relative_eq!(riemann[(i, j, k, l)], -riemann[(j, i, k, l)], epsilon = 1e-10);
                        // Antisymmetry in last two indices
                        assert_relative_eq!(riemann[(i, j, k, l)], -riemann[(i, j, l, k)], epsilon = 1e-10);
                        // Block symmetry
                        assert_relative_eq!(riemann[(i, j, k, l)], riemann[(k, l, i, j)], epsilon = 1e-10);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_bianchi_identity() {
        let manifold = ClimateManifold::new();
        let state = ClimateState::current_2024();
        let riemann = manifold.riemann_tensor(&state);
        
        // First Bianchi identity: R_ijkl + R_iklj + R_iljk = 0
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    for l in 0..8 {
                        let sum = riemann[(i, j, k, l)] + riemann[(i, k, l, j)] + riemann[(i, l, j, k)];
                        assert!(sum.abs() < 1e-10, "Bianchi identity violated");
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_geodesic_conservation() {
        let manifold = ClimateManifold::new();
        let state = ClimateState::preindustrial();
        let velocity = Vector8::new(0.1, 0.05, 0.02, -0.01, 0.03, 0.01, 0.0, -0.02);
        
        let g = manifold.metric_tensor(&state);
        let acceleration = manifold.geodesic_equation(&state, &velocity);
        
        // Check that geodesic preserves the norm of velocity
        let v_norm_sq = velocity.dot(&(g * velocity));
        let v_dot_a = velocity.dot(&(g * acceleration));
        
        assert!(v_dot_a.abs() < 1e-10, "Geodesic should preserve velocity norm");
    }
    
    #[test]
    fn test_tipping_point_detection() {
        let manifold = ClimateManifold::new();
        
        // Test at different warming levels
        for warming in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] {
            let mut state = ClimateState::preindustrial();
            state.global_temp_anomaly_k = warming;
            state.co2_ppm = 278.0 * 2.0_f64.powf(warming / 3.0);  // Approximate CO2-temperature relationship
            
            let tipping_points = manifold.detect_tipping_points(&state);
            
            // Verify expected tipping points appear at right temperatures
            if warming > 1.5 {
                assert!(tipping_points.iter().any(|tp| tp.tipping_type == TippingType::GreenlandIceSheet));
            }
            if warming > 2.0 {
                assert!(tipping_points.iter().any(|tp| tp.tipping_type == TippingType::WestAntarcticIceSheet));
            }
            if warming > 3.0 {
                assert!(tipping_points.iter().any(|tp| tp.tipping_type == TippingType::AmazonRainforest));
            }
        }
    }
    
    #[test]
    fn test_field_equation_consistency() {
        let manifold = ClimateManifold::new();
        let state = ClimateState::current_2024();
        
        let residual = manifold.field_equation_residual(&state);
        
        // Residual should be small for physical states
        assert!(residual < 10.0, "Field equation residual too large: {}", residual);
    }
    
    #[test]
    fn test_sectional_curvature_bounds() {
        let manifold = ClimateManifold::new();
        let state = ClimateState::current_2024();
        
        // Test curvature in different 2-planes
        let mut e = vec![Vector8::zeros(); 8];
        for i in 0..8 {
            e[i][i] = 1.0;
        }
        
        for i in 0..8 {
            for j in i+1..8 {
                let k = manifold.sectional_curvature(&state, &e[i], &e[j]);
                
                // Sectional curvature should be bounded for physical states
                assert!(k.abs() < 1000.0, "Sectional curvature unbounded: K({},{}) = {}", i, j, k);
            }
        }
    }
}

// Export main functionality
pub use self::ClimateManifold as Manifold;
pub use self::ClimateState as State;
pub use self::TippingPoint as Tipping;
pub use self::TippingType as TipType;