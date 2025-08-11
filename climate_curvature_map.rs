// Climate Riemann curvature tensor field
// Experimental curvature tracking

// ============================================================================
// DATA SOURCE REQUIREMENTS - CURVATURE TENSOR ANALYSIS
// ============================================================================
//
// CORE PHYSICAL OBSERVATIONS:
// Source: ERA5 Reanalysis, CMIP6 Ensemble, Real-time weather stations
// Instrument: ECMWF Model, Weather station networks, Satellite radiometers
// Spatiotemporal Resolution: 
//   - ERA5: 0.25° × 0.25°, hourly, 1979-present
//   - CMIP6: Variable (typically 1-2°), monthly/daily, 1850-2300
//   - Stations: Point measurements, hourly/daily
// File Format: NetCDF4, GRIB2
// Data Size: ~2TB/year ERA5, ~50GB/year CMIP6 subset
// API Access: Copernicus CDS API, ESGF THREDDS servers
// Variables: 
//   - Global mean temperature (K) - derived from 2m temperature
//   - Atmospheric CO2 (ppm) - from NOAA/ESRL measurements  
//   - Ocean heat content (ZJ) - from Argo float profiles
//   - Global ice volume (km³) - from GRACE satellites, ice sheet models
//   - AMOC strength (Sv) - from ocean reanalysis, RAPID array
//   - Methane (ppb) - from NOAA/ESRL networks
//   - Cloud fraction (0-1) - from MODIS, ISCCP cloud products
//   - Soil carbon (GtC) - from land surface models, observations
//
// GEOMETRIC CURVATURE COMPUTATION:
// Preprocessing Required:
//   1. Spatiotemporal interpolation to common 8D state space grid
//   2. Calculate Fisher information metric from observation covariances
//   3. Compute Christoffel symbols via automatic differentiation
//   4. Calculate Riemann tensor components from connection derivatives
//   5. Quality control: Remove singular points, smooth noise
//
// MISSING REQUIREMENTS MAKING THIS A STUB:
// - High-frequency ocean observations for AMOC fluctuations
// - Real-time soil carbon measurements with global coverage
// - Consistent cloud property retrievals across satellite missions
// - Automatic differentiation framework for metric tensor derivatives
// - Scalable tensor computation library for 8D+ manifolds
//
// IMPLEMENTATION GAPS:
// - Currently uses synthetic test data instead of real observations
// - Simplified metric assumes Euclidean base, needs Fisher information
// - Christoffel symbol computation lacks numerical stability checks
// - Missing parallel processing for high-dimensional tensor operations
// - No validation against known climate attractor properties

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use nalgebra::{DMatrix, DVector, Matrix4, Vector4, Matrix, U8};
type Matrix8<T> = Matrix<T, U8, U8, nalgebra::ArrayStorage<T, 8, 8>>;
use num_dual::{DualNum, Dual2, Dual3, Dual2Vec64, Dual3Vec64, Gradient, Hessian};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use dashmap::DashMap;
use crossbeam_channel::{bounded, Sender, Receiver};
use parking_lot::Mutex;
use ahash::AHasher;
use std::hash::{Hash, Hasher};

// Climate manifold coordinates

/// Climate state in thermodynamic phase space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimatePoint {
    pub temp_k: f64,           // Global mean temperature (K)
    pub co2_ppm: f64,          // Atmospheric CO2 (ppm)
    pub ocean_heat_zj: f64,    // Ocean heat content (ZJ)
    pub ice_volume_km3: f64,   // Global ice volume (km³)
    pub amoc_sv: f64,          // AMOC strength (Sverdrups)
    pub ch4_ppb: f64,          // Methane concentration (ppb)
    pub cloud_fraction: f64,   // Global cloud cover fraction
    pub soil_carbon_gtc: f64,  // Soil carbon reservoir (GtC)
}

impl ClimatePoint {
    /// Convert to 8D vector for manifold computations
    pub fn to_vector(&self) -> Vector8 {
        Vector8::new(
            self.temp_k,
            self.co2_ppm,
            self.ocean_heat_zj,
            self.ice_volume_km3,
            self.amoc_sv,
            self.ch4_ppb,
            self.cloud_fraction,
            self.soil_carbon_gtc,
        )
    }
    
    /// Compute local metric tensor g_ij at this point
    pub fn metric_tensor(&self) -> Matrix8 {
        let mut g = Matrix8::zeros();
        
        // Diagonal terms (variances in each dimension)
        g[(0, 0)] = 1.0;  // Temperature variance ~1K
        g[(1, 1)] = 100.0;  // CO2 variance ~10ppm -> scale by 100
        g[(2, 2)] = 10.0;   // Ocean heat ~3ZJ variance
        g[(3, 3)] = 1e6;    // Ice volume ~1000km³ variance
        g[(4, 4)] = 4.0;    // AMOC ~2Sv variance
        g[(5, 5)] = 400.0;  // CH4 ~20ppb variance
        g[(6, 6)] = 0.01;   // Cloud fraction ~0.1 variance
        g[(7, 7)] = 100.0;  // Soil carbon ~10GtC variance
        
        // Off-diagonal correlations (climate feedbacks)
        // Temperature-CO2 coupling (positive feedback)
        g[(0, 1)] = g[(1, 0)] = 0.3 * (g[(0, 0)] * g[(1, 1)]).sqrt();
        
        // Temperature-ice coupling (ice-albedo feedback)
        g[(0, 3)] = g[(3, 0)] = -0.7 * (g[(0, 0)] * g[(3, 3)]).sqrt();
        
        // CO2-ocean heat coupling (ocean CO2 solubility)
        g[(1, 2)] = g[(2, 1)] = -0.2 * (g[(1, 1)] * g[(2, 2)]).sqrt();
        
        // AMOC-temperature coupling (thermohaline circulation)
        g[(0, 4)] = g[(4, 0)] = -0.5 * (g[(0, 0)] * g[(4, 4)]).sqrt();
        
        // Methane-temperature coupling (permafrost feedback)
        g[(0, 5)] = g[(5, 0)] = 0.4 * (g[(0, 0)] * g[(5, 5)]).sqrt();
        
        // Cloud-temperature coupling (cloud feedback uncertainty)
        g[(0, 6)] = g[(6, 0)] = 0.1 * (g[(0, 0)] * g[(6, 6)]).sqrt();
        
        // Soil carbon-temperature (soil respiration feedback)
        g[(0, 7)] = g[(7, 0)] = -0.3 * (g[(0, 0)] * g[(7, 7)]).sqrt();
        
        g
    }
}

type Vector8 = nalgebra::Vector8<f64>;
type Matrix8 = nalgebra::Matrix8<f64>;

// ---
// CLIMATE STATE TRANSITIONS WITH RIEMANN CURVATURE
// ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateTransition {
    pub from_state: ClimateState,
    pub to_state: ClimateState,
    pub riemann_tensor: RiemannTensor,
    pub ricci_scalar: f64,
    pub sectional_curvatures: HashMap<String, f64>,
    pub holonomy_group: HolonomyGroup,
    pub lyapunov_exponents: Vec<f64>,
    pub kolmogorov_entropy: f64,
    pub fisher_information: Matrix8,
    pub transition_probability: f64,
    pub irreversibility_measure: f64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum ClimateState {
    // Stable states
    Holocene,
    Preindustrial,
    Industrial,
    Modern,
    
    // Temperature regimes
    OnePointFiveC,
    TwoC,
    ThreeC,
    FourC,
    FiveC,
    
    // Tipping elements
    ArcticIceFree,
    AMOCCollapsed,
    AmazonDieback,
    WAISCollapsed,
    GreenlandCollapsed,
    PermafrostThawed,
    CoralReefBleached,
    
    // Scenarios
    RCP26,
    RCP45,
    RCP60,
    RCP85,
    SSP119,
    SSP126,
    SSP245,
    SSP370,
    SSP585,
    
    // Extreme states
    HothouseEarth,
    SnowballEarth,
    RunawayGreenhouse,
}

impl ClimateState {
    /// Get characteristic point in phase space
    pub fn reference_point(&self) -> ClimatePoint {
        match self {
            Self::Holocene => ClimatePoint {
                temp_k: 287.0,
                co2_ppm: 280.0,
                ocean_heat_zj: 0.0,
                ice_volume_km3: 2.6e7,
                amoc_sv: 15.0,
                ch4_ppb: 700.0,
                cloud_fraction: 0.67,
                soil_carbon_gtc: 1600.0,
            },
            Self::TwoC => ClimatePoint {
                temp_k: 289.0,
                co2_ppm: 450.0,
                ocean_heat_zj: 300.0,
                ice_volume_km3: 2.3e7,
                amoc_sv: 12.0,
                ch4_ppb: 1900.0,
                cloud_fraction: 0.65,
                soil_carbon_gtc: 1500.0,
            },
            Self::HothouseEarth => ClimatePoint {
                temp_k: 293.0,
                co2_ppm: 1200.0,
                ocean_heat_zj: 1000.0,
                ice_volume_km3: 0.0,
                amoc_sv: 0.0,
                ch4_ppb: 3500.0,
                cloud_fraction: 0.55,
                soil_carbon_gtc: 800.0,
            },
            Self::AMOCCollapsed => ClimatePoint {
                temp_k: 288.0,
                co2_ppm: 500.0,
                ocean_heat_zj: 400.0,
                ice_volume_km3: 2.0e7,
                amoc_sv: 0.0,  // Collapsed!
                ch4_ppb: 2000.0,
                cloud_fraction: 0.70,
                soil_carbon_gtc: 1400.0,
            },
            _ => Self::Holocene.reference_point(),  // Default
        }
    }
}

// ---
// RIEMANN CURVATURE TENSOR R^i_jkl
// ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiemannTensor {
    /// R^i_jkl components in local coordinates
    pub components: Vec<f64>,  // 8^4 = 4096 components (but symmetric)
    pub dimension: usize,
}

impl RiemannTensor {
    pub fn compute(point: &ClimatePoint, epsilon: f64) -> Self {
        let dim = 8;
        let mut components = vec![0.0; dim * dim * dim * dim];
        
        // Compute Christoffel symbols first
        let christoffel = Self::compute_christoffel(point, epsilon);
        
        // R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let mut r_ijkl = 0.0;
                        
                        // Partial derivatives of Christoffel symbols
                        let dg_k = Self::partial_christoffel(point, &christoffel, k, epsilon);
                        let dg_l = Self::partial_christoffel(point, &christoffel, l, epsilon);
                        
                        r_ijkl += dg_k[i * dim * dim + j * dim + l];
                        r_ijkl -= dg_l[i * dim * dim + j * dim + k];
                        
                        // Quadratic terms
                        for m in 0..dim {
                            r_ijkl += christoffel[i * dim * dim + m * dim + k] * 
                                     christoffel[m * dim * dim + j * dim + l];
                            r_ijkl -= christoffel[i * dim * dim + m * dim + l] * 
                                     christoffel[m * dim * dim + j * dim + k];
                        }
                        
                        components[i * dim * dim * dim + j * dim * dim + k * dim + l] = r_ijkl;
                    }
                }
            }
        }
        
        RiemannTensor { components, dimension: dim }
    }
    
    fn compute_christoffel(point: &ClimatePoint, _epsilon: f64) -> Vec<f64> {
        let dim = 8;
        let mut christoffel = vec![0.0; dim * dim * dim];
        
        // Convert climate point to dual numbers for automatic differentiation
        let coords = vec![
            point.temp_k, point.co2_ppm, point.ocean_heat_zj, point.ice_volume_km3,
            point.amoc_sv, point.ch4_ppb, point.cloud_fraction, point.soil_carbon_gtc
        ];
        
        // Γ^i_jk = (1/2) g^il (∂_j g_lk + ∂_k g_jl - ∂_l g_jk)
        // Using automatic differentiation via dual numbers
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    // Create dual number coordinates with derivatives w.r.t. j and k
                    let mut dual_coords_j = Dual2Vec64::new(coords.clone(), coords.clone());
                    let mut dual_coords_k = Dual2Vec64::new(coords.clone(), coords.clone());
                    
                    // Set unit perturbations for automatic differentiation
                    for idx in 0..dim {
                        dual_coords_j[idx] = if idx == j {
                            Dual2::new(coords[idx], 1.0, 0.0)
                        } else {
                            Dual2::new(coords[idx], 0.0, 0.0)
                        };
                        
                        dual_coords_k[idx] = if idx == k {
                            Dual2::new(coords[idx], 0.0, 1.0)
                        } else {
                            Dual2::new(coords[idx], 0.0, 0.0)
                        };
                    }
                    
                    // Compute metric tensor with dual numbers to get derivatives
                    let g_dual = Self::metric_tensor_dual(&dual_coords_j);
                    let g_inv = point.metric_tensor().try_inverse().unwrap_or(Matrix8::identity());
                    
                    let mut gamma = 0.0;
                    for l in 0..dim {
                        // Extract derivatives from dual numbers
                        // This gives us ∂g_lk/∂x_j automatically without finite differences
                        let dg_lk_dj = g_dual[(l, k)].first_derivative();
                        let dg_jl_dk = g_dual[(j, l)].second_derivative(); 
                        let dg_jk_dl = if j == l { g_dual[(j, k)].first_derivative() } else { 0.0 };
                        
                        gamma += 0.5 * g_inv[(i, l)] * (dg_lk_dj + dg_jl_dk - dg_jk_dl);
                    }
                    
                    christoffel[i * dim * dim + j * dim + k] = gamma;
                }
            }
        }
        
        christoffel
    }
    
    fn metric_tensor_dual(coords: &Dual2Vec64) -> Matrix8<Dual2<f64, f64>> {
        // Compute metric tensor with dual number coordinates
        // This automatically computes derivatives via dual arithmetic
        let mut g = Matrix8::zeros();
        
        // Fisher-Rao metric components with automatic differentiation
        // Temperature-temperature component
        g[(0, 0)] = Dual2::new(1.0, 0.0, 0.0) / (coords[0] * coords[0]);
        
        // CO2-CO2 component 
        g[(1, 1)] = Dual2::new(1.0, 0.0, 0.0) / (coords[1] * coords[1]);
        
        // Ocean heat content component
        g[(2, 2)] = Dual2::new(1.0, 0.0, 0.0) / (coords[2].abs() + Dual2::new(1e-10, 0.0, 0.0));
        
        // Ice volume component with singularity near zero
        let ice_reg = coords[3].abs() + Dual2::new(1e-6, 0.0, 0.0);
        g[(3, 3)] = Dual2::new(1.0, 0.0, 0.0) / (ice_reg * ice_reg);
        
        // AMOC component
        g[(4, 4)] = Dual2::new(1.0, 0.0, 0.0) / (coords[4].abs() + Dual2::new(0.1, 0.0, 0.0));
        
        // Methane component
        g[(5, 5)] = Dual2::new(1.0, 0.0, 0.0) / (coords[5] * coords[5]);
        
        // Cloud fraction component
        let cloud_reg = coords[6] * (Dual2::new(1.0, 0.0, 0.0) - coords[6]) + Dual2::new(1e-6, 0.0, 0.0);
        g[(6, 6)] = Dual2::new(1.0, 0.0, 0.0) / cloud_reg;
        
        // Soil carbon component
        g[(7, 7)] = Dual2::new(1.0, 0.0, 0.0) / (coords[7].abs() + Dual2::new(1.0, 0.0, 0.0));
        
        // Off-diagonal coupling terms
        g[(0, 1)] = coords[0] * coords[1] / Dual2::new(1000.0, 0.0, 0.0); // Temp-CO2 coupling
        g[(1, 0)] = g[(0, 1)];
        
        g[(0, 3)] = -coords[0] / (ice_reg * Dual2::new(100.0, 0.0, 0.0)); // Temp-ice feedback
        g[(3, 0)] = g[(0, 3)];
        
        g
    }
    
    fn perturb_coordinate(point: &mut ClimatePoint, coord: usize, delta: f64) {
        match coord {
            0 => point.temp_k += delta,
            1 => point.co2_ppm += delta,
            2 => point.ocean_heat_zj += delta,
            3 => point.ice_volume_km3 += delta,
            4 => point.amoc_sv += delta,
            5 => point.ch4_ppb += delta,
            6 => point.cloud_fraction += delta,
            7 => point.soil_carbon_gtc += delta,
            _ => {}
        }
    }
    
    fn partial_christoffel(point: &ClimatePoint, _christoffel: &[f64], index: usize, _epsilon: f64) -> Vec<f64> {
        // Use third-order dual numbers to compute derivatives of Christoffel symbols
        let dim = 8;
        let coords = vec![
            point.temp_k, point.co2_ppm, point.ocean_heat_zj, point.ice_volume_km3,
            point.amoc_sv, point.ch4_ppb, point.cloud_fraction, point.soil_carbon_gtc
        ];
        
        // Create dual number coordinates with third-order derivatives
        let mut dual_coords = Dual3Vec64::new(coords.clone(), coords.clone());
        for idx in 0..dim {
            dual_coords[idx] = if idx == index {
                Dual3::new(coords[idx], 1.0, 0.0, 0.0) // Derivative w.r.t. index
            } else {
                Dual3::new(coords[idx], 0.0, 0.0, 0.0)
            };
        }
        
        // Compute Christoffel symbols with automatic differentiation
        let mut partial_gamma = vec![0.0; dim * dim * dim];
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    // This automatically computes ∂Γ^i_jk/∂x_index via dual arithmetic
                    // No finite differences needed!
                    let gamma_dual = Self::compute_christoffel_dual(&dual_coords, i, j, k);
                    partial_gamma[i * dim * dim + j * dim + k] = gamma_dual.first_derivative();
                }
            }
        }
        
        partial_gamma
    }
    
    fn compute_christoffel_dual(dual_coords: &Dual3Vec64, i: usize, j: usize, k: usize) -> Dual3<f64, f64> {
        // Compute single Christoffel symbol with dual numbers
        let g_dual = Self::metric_tensor_dual3(dual_coords);
        let g_regular = Self::coords_to_metric(&dual_coords.iter().map(|d| d.value()).collect::<Vec<_>>());
        let g_inv = g_regular.try_inverse().unwrap_or(Matrix8::identity());
        
        let mut gamma = Dual3::new(0.0, 0.0, 0.0, 0.0);
        for l in 0..8 {
            // Automatic differentiation handles all the derivatives
            gamma += Dual3::new(0.5 * g_inv[(i, l)], 0.0, 0.0, 0.0) * 
                     (g_dual[(l, k)].derivative(j) + g_dual[(j, l)].derivative(k) - g_dual[(j, k)].derivative(l));
        }
        gamma
    }
    
    fn metric_tensor_dual3(coords: &Dual3Vec64) -> Vec<Vec<Dual3<f64, f64>>> {
        // Similar to metric_tensor_dual but with third-order dual numbers
        let mut g = vec![vec![Dual3::new(0.0, 0.0, 0.0, 0.0); 8]; 8];
        
        // Diagonal components
        g[0][0] = Dual3::new(1.0, 0.0, 0.0, 0.0) / (coords[0] * coords[0]);
        g[1][1] = Dual3::new(1.0, 0.0, 0.0, 0.0) / (coords[1] * coords[1]);
        g[2][2] = Dual3::new(1.0, 0.0, 0.0, 0.0) / (coords[2].abs() + Dual3::new(1e-10, 0.0, 0.0, 0.0));
        
        let ice_reg = coords[3].abs() + Dual3::new(1e-6, 0.0, 0.0, 0.0);
        g[3][3] = Dual3::new(1.0, 0.0, 0.0, 0.0) / (ice_reg * ice_reg);
        
        g[4][4] = Dual3::new(1.0, 0.0, 0.0, 0.0) / (coords[4].abs() + Dual3::new(0.1, 0.0, 0.0, 0.0));
        g[5][5] = Dual3::new(1.0, 0.0, 0.0, 0.0) / (coords[5] * coords[5]);
        
        let cloud_reg = coords[6] * (Dual3::new(1.0, 0.0, 0.0, 0.0) - coords[6]) + Dual3::new(1e-6, 0.0, 0.0, 0.0);
        g[6][6] = Dual3::new(1.0, 0.0, 0.0, 0.0) / cloud_reg;
        
        g[7][7] = Dual3::new(1.0, 0.0, 0.0, 0.0) / (coords[7].abs() + Dual3::new(1.0, 0.0, 0.0, 0.0));
        
        // Off-diagonal couplings
        g[0][1] = coords[0] * coords[1] / Dual3::new(1000.0, 0.0, 0.0, 0.0);
        g[1][0] = g[0][1];
        
        g[0][3] = -coords[0] / (ice_reg * Dual3::new(100.0, 0.0, 0.0, 0.0));
        g[3][0] = g[0][3];
        
        g
    }
    
    fn coords_to_metric(coords: &[f64]) -> Matrix8<f64> {
        // Convert coordinates to regular metric tensor
        let mut g = Matrix8::zeros();
        
        g[(0, 0)] = 1.0 / (coords[0] * coords[0]);
        g[(1, 1)] = 1.0 / (coords[1] * coords[1]);
        g[(2, 2)] = 1.0 / (coords[2].abs() + 1e-10);
        g[(3, 3)] = 1.0 / ((coords[3].abs() + 1e-6).powi(2));
        g[(4, 4)] = 1.0 / (coords[4].abs() + 0.1);
        g[(5, 5)] = 1.0 / (coords[5] * coords[5]);
        g[(6, 6)] = 1.0 / (coords[6] * (1.0 - coords[6]) + 1e-6);
        g[(7, 7)] = 1.0 / (coords[7].abs() + 1.0);
        
        g[(0, 1)] = coords[0] * coords[1] / 1000.0;
        g[(1, 0)] = g[(0, 1)];
        
        g[(0, 3)] = -coords[0] / ((coords[3].abs() + 1e-6) * 100.0);
        g[(3, 0)] = g[(0, 3)];
        
        g
    }
    
    /// Compute Ricci scalar R = g^ij R_ij
    pub fn ricci_scalar(&self, g_inv: &Matrix8) -> f64 {
        let dim = self.dimension;
        let mut ricci_scalar = 0.0;
        
        // Contract twice: R = g^ij R_ij = g^ij R^k_ikj
        for i in 0..dim {
            for j in 0..dim {
                let mut r_ij = 0.0;
                for k in 0..dim {
                    r_ij += self.components[k * dim * dim * dim + i * dim * dim + k * dim + j];
                }
                ricci_scalar += g_inv[(i, j)] * r_ij;
            }
        }
        
        ricci_scalar
    }
    
    /// Compute sectional curvature K(v, w) for plane spanned by v, w
    pub fn sectional_curvature(&self, v: &Vector8, w: &Vector8, g: &Matrix8) -> f64 {
        // K(v,w) = R(v,w,w,v) / (|v|²|w|² - <v,w>²)
        let r_vwwv = self.evaluate(v, w, w, v);
        let v_norm_sq = v.dot(&(g * v));
        let w_norm_sq = w.dot(&(g * w));
        let vw_inner = v.dot(&(g * w));
        
        let denominator = v_norm_sq * w_norm_sq - vw_inner * vw_inner;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            r_vwwv / denominator
        }
    }
    
    fn evaluate(&self, v1: &Vector8, v2: &Vector8, v3: &Vector8, v4: &Vector8) -> f64 {
        let dim = self.dimension;
        let mut result = 0.0;
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let idx = i * dim * dim * dim + j * dim * dim + k * dim + l;
                        result += self.components[idx] * v1[i] * v2[j] * v3[k] * v4[l];
                    }
                }
            }
        }
        
        result
    }
}

// ---
// HOLONOMY GROUP AND PARALLEL TRANSPORT
// ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyGroup {
    pub generators: Vec<Matrix8>,
    pub structure_constants: Vec<f64>,
    pub representation_dim: usize,
    pub is_irreducible: bool,
    pub berger_type: BergerClassification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BergerClassification {
    SO8,           // Generic Riemannian
    U4,            // Kähler (complex structure)
    Sp2,           // Hyperkähler
    G2,            // Exceptional holonomy
    Spin7,         // Exceptional holonomy
    Reducible,     // Product manifold
    Symmetric,     // Symmetric space
}

impl HolonomyGroup {
    pub fn compute_from_curvature(riemann: &RiemannTensor, point: &ClimatePoint) -> Self {
        // Compute holonomy by parallel transport around infinitesimal loops
        let dim = 8;
        let mut generators = Vec::new();
        
        // For each pair of coordinate directions, compute holonomy
        for i in 0..dim {
            for j in i+1..dim {
                let mut hol_matrix = Matrix8::identity();
                
                // Holonomy ≈ I + R_ij dx^i ∧ dx^j for small loop
                for k in 0..dim {
                    for l in 0..dim {
                        let idx = k * dim * dim * dim + l * dim * dim + i * dim + j;
                        hol_matrix[(k, l)] += 0.01 * riemann.components[idx];
                    }
                }
                
                generators.push(hol_matrix - Matrix8::identity());
            }
        }
        
        // Classify based on preserved structures
        let berger_type = Self::classify_holonomy(&generators, point);
        
        HolonomyGroup {
            generators,
            structure_constants: vec![0.0; dim * dim * dim],  // Would compute Lie algebra structure
            representation_dim: dim,
            is_irreducible: !matches!(berger_type, BergerClassification::Reducible),
            berger_type,
        }
    }
    
    fn classify_holonomy(generators: &[Matrix8], point: &ClimatePoint) -> BergerClassification {
        // Check for special structures preserved by holonomy
        
        // Check if metric is Kähler (has complex structure)
        if Self::has_complex_structure(generators) {
            return BergerClassification::U4;
        }
        
        // Check for symmetric space (constant curvature)
        if Self::is_symmetric_space(generators) {
            return BergerClassification::Symmetric;
        }
        
        // Default to generic SO(8)
        BergerClassification::SO8
    }
    
    fn has_complex_structure(generators: &[Matrix8]) -> bool {
        // Check if there exists J with J² = -I preserved by holonomy
        // Look for generators that could represent complex multiplication
        for gen in generators {
            // Check if generator squared equals negative identity
            let gen_squared = gen * gen;
            let neg_identity = -Matrix8::identity();
            
            // Check if gen² ≈ -I (allowing for numerical tolerance)
            let diff = gen_squared - neg_identity;
            let frobenius_norm = diff.iter().fold(0.0, |acc, &x| acc + x * x).sqrt();
            
            if frobenius_norm < 1e-10 {
                // Found a complex structure candidate
                // REQUIRES: Full implementation should check integrability via Nijenhuis tensor N_J = 0
                // For now, accept any generator with J² = -I property
                return true;
            }
        }
        false
    }
    
    fn is_symmetric_space(generators: &[Matrix8]) -> bool {
        // Check if ∇R = 0 (covariant derivative of curvature vanishes)
        // For symmetric spaces, holonomy generators commute in specific pattern
        
        if generators.len() < 2 {
            return true; // Trivial case
        }
        
        // Check if all generators pairwise commute or follow symmetric space pattern
        for i in 0..generators.len() {
            for j in (i + 1)..generators.len() {
                let gen_i = &generators[i];
                let gen_j = &generators[j];
                
                // Compute commutator [A_i, A_j] = A_i * A_j - A_j * A_i
                let commutator = gen_i * gen_j - gen_j * gen_i;
                
                // Check if commutator is small (indicating symmetric structure)
                let comm_norm = commutator.iter().fold(0.0, |acc, &x| acc + x * x).sqrt();
                
                if comm_norm > 1e-8 {
                    // Non-zero commutator suggests non-symmetric space
                    // REQUIRES: Full implementation should compute ∇_X R(Y,Z)W = 0
                    // This is a simplified heuristic based on holonomy structure
                    return false;
                }
            }
        }
        true // All generators approximately commute
    }
}

// ---
// Curvature map with streaming
// ---

pub struct ClimateCurvatureField {
    /// Transition curvatures indexed by state pair hash
    transitions: Arc<DashMap<u64, TransitionCurvature>>,
    
    /// Ring buffer for recent high-curvature events
    high_curvature_events: Arc<Mutex<VecDeque<CurvatureEvent>>>,
    
    /// Real-time streaming channels
    event_sender: Sender<CurvatureEvent>,
    event_receiver: Receiver<CurvatureEvent>,
    
    /// Tipping point detector
    tipping_detector: TippingPointDetector,
    
    /// Early warning system
    early_warning: EarlyWarningSystem,
    
    /// Statistics aggregator
    stats: Arc<RwLock<CurvatureStatistics>>,
    
    /// Configuration
    config: CurvatureConfig,
}

#[derive(Debug, Clone)]
struct TransitionCurvature {
    from: ClimateState,
    to: ClimateState,
    samples: VecDeque<CurvatureSample>,
    mean_ricci_scalar: f64,
    variance_ricci_scalar: f64,
    max_sectional_curvature: f64,
    lyapunov_spectrum: Vec<f64>,
    fisher_info_determinant: f64,
    last_updated_ns: u64,
    
    // Tipping point indicators
    critical_slowing_down: f64,
    variance_inflation: f64,
    lag1_autocorrelation: f64,
    skewness_change: f64,
    spatial_correlation_range: f64,
    
    // Irreversibility measures
    hysteresis_width: f64,
    basin_stability: f64,
    committal_probability: f64,
}

#[derive(Debug, Clone)]
struct CurvatureSample {
    timestamp_ns: u64,
    ricci_scalar: f64,
    max_sectional: f64,
    min_sectional: f64,
    mean_sectional: f64,
    gauss_curvature: f64,
    scalar_curvature: f64,
    weyl_tensor_norm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureEvent {
    pub timestamp_ns: u64,
    pub event_type: EventType,
    pub severity: Severity,
    pub from_state: ClimateState,
    pub to_state: ClimateState,
    pub curvature_value: f64,
    pub confidence: f64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    TippingPointApproaching,
    TippingPointCrossed,
    IrreversibleTransition,
    CurvatureSingularity,
    BifurcationDetected,
    CriticalSlowingDown,
    VarianceInflation,
    HysteresisDetected,
    RegimeShift,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialOrd, PartialEq)]
pub enum Severity {
    Info,
    Warning,
    Critical,
    Emergency,
}

struct TippingPointDetector {
    thresholds: HashMap<String, f64>,
    detection_window: usize,
    sensitivity: f64,
}

impl TippingPointDetector {
    fn detect(&self, transition: &TransitionCurvature) -> Option<CurvatureEvent> {
        // Multi-indicator tipping point detection
        let mut indicators = 0;
        let mut confidence = 0.0;
        
        // Check critical slowing down
        if transition.critical_slowing_down > 0.8 {
            indicators += 1;
            confidence += 0.2;
        }
        
        // Check variance inflation
        if transition.variance_inflation > 2.0 {
            indicators += 1;
            confidence += 0.2;
        }
        
        // Check lag-1 autocorrelation approaching 1
        if transition.lag1_autocorrelation > 0.9 {
            indicators += 1;
            confidence += 0.3;
        }
        
        // Check curvature singularity (very high Ricci scalar)
        if transition.mean_ricci_scalar.abs() > 100.0 {
            indicators += 1;
            confidence += 0.3;
        }
        
        if indicators >= 3 {
            Some(CurvatureEvent {
                timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
                event_type: EventType::TippingPointApproaching,
                severity: if confidence > 0.7 { Severity::Emergency } else { Severity::Critical },
                from_state: transition.from.clone(),
                to_state: transition.to.clone(),
                curvature_value: transition.mean_ricci_scalar,
                confidence: confidence.min(1.0),
                message: format!("Tipping point detected: {} indicators triggered", indicators),
            })
        } else {
            None
        }
    }
}

struct EarlyWarningSystem {
    detrending_window: usize,
    kendall_tau_threshold: f64,
    warning_lead_time_steps: usize,
}

impl EarlyWarningSystem {
    fn analyze(&self, samples: &VecDeque<CurvatureSample>) -> Vec<EarlyWarningSignal> {
        let mut signals = Vec::new();
        
        if samples.len() < self.detrending_window {
            return signals;
        }
        
        // Compute rolling statistics
        let recent: Vec<f64> = samples.iter()
            .rev()
            .take(self.detrending_window)
            .map(|s| s.ricci_scalar)
            .collect();
        
        // Detrend using linear regression
        let detrended = self.detrend(&recent);
        
        // Check for increasing variance
        let var_first_half = self.variance(&detrended[..detrended.len()/2]);
        let var_second_half = self.variance(&detrended[detrended.len()/2..]);
        
        if var_second_half > var_first_half * 1.5 {
            signals.push(EarlyWarningSignal::IncreasingVariance(var_second_half / var_first_half));
        }
        
        // Check for increasing autocorrelation
        let ac = self.autocorrelation(&detrended, 1);
        if ac > 0.8 {
            signals.push(EarlyWarningSignal::CriticalSlowingDown(ac));
        }
        
        // Check for flickering (increased switching between states)
        let switches = self.count_sign_changes(&detrended);
        if switches > detrended.len() / 4 {
            signals.push(EarlyWarningSignal::Flickering(switches as f64 / detrended.len() as f64));
        }
        
        signals
    }
    
    fn detrend(&self, data: &[f64]) -> Vec<f64> {
        // Simple linear detrending
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = data.iter().sum::<f64>() / n;
        
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean) * (x - x_mean);
        }
        
        let slope = num / den;
        let intercept = y_mean - slope * x_mean;
        
        data.iter().enumerate()
            .map(|(i, &y)| y - (slope * i as f64 + intercept))
            .collect()
    }
    
    fn variance(&self, data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
    }
    
    fn autocorrelation(&self, data: &[f64], lag: usize) -> f64 {
        let n = data.len();
        if lag >= n { return 0.0; }
        
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = self.variance(data);
        
        let mut cov = 0.0;
        for i in 0..n-lag {
            cov += (data[i] - mean) * (data[i + lag] - mean);
        }
        cov /= (n - lag) as f64;
        
        cov / var
    }
    
    fn count_sign_changes(&self, data: &[f64]) -> usize {
        let mut changes = 0;
        for i in 1..data.len() {
            if data[i].signum() != data[i-1].signum() {
                changes += 1;
            }
        }
        changes
    }
}

#[derive(Debug, Clone)]
enum EarlyWarningSignal {
    CriticalSlowingDown(f64),
    IncreasingVariance(f64),
    Flickering(f64),
    SkewnessChange(f64),
    SpatialCorrelationIncrease(f64),
}

#[derive(Debug, Clone, Default)]
struct CurvatureStatistics {
    total_samples: u64,
    mean_global_curvature: f64,
    max_observed_curvature: f64,
    min_observed_curvature: f64,
    tipping_events_detected: u64,
    irreversible_transitions: u64,
    current_regime: ClimateState,
    regime_duration_ns: u64,
}

#[derive(Debug, Clone)]
struct CurvatureConfig {
    max_samples_per_transition: usize,
    high_curvature_threshold: f64,
    tipping_point_threshold: f64,
    irreversibility_threshold: f64,
    event_buffer_size: usize,
    streaming_batch_size: usize,
}

impl Default for CurvatureConfig {
    fn default() -> Self {
        Self {
            max_samples_per_transition: 10000,
            high_curvature_threshold: 10.0,
            tipping_point_threshold: 50.0,
            irreversibility_threshold: 100.0,
            event_buffer_size: 100000,
            streaming_batch_size: 1000,
        }
    }
}

impl ClimateCurvatureField {
    pub fn new(config: CurvatureConfig) -> Self {
        let (tx, rx) = bounded(config.event_buffer_size);
        
        Self {
            transitions: Arc::new(DashMap::new()),
            high_curvature_events: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            event_sender: tx,
            event_receiver: rx,
            tipping_detector: TippingPointDetector {
                thresholds: HashMap::new(),
                detection_window: 100,
                sensitivity: 0.8,
            },
            early_warning: EarlyWarningSystem {
                detrending_window: 50,
                kendall_tau_threshold: 0.7,
                warning_lead_time_steps: 20,
            },
            stats: Arc::new(RwLock::new(CurvatureStatistics::default())),
            config,
        }
    }
    
    pub fn process_transition(&self, from: ClimateState, to: ClimateState, point: ClimatePoint) {
        let transition_hash = self.hash_transition(&from, &to);
        
        // Compute curvature at this point
        let riemann = RiemannTensor::compute(&point, 1e-6);
        let g_inv = point.metric_tensor().try_inverse().unwrap_or(Matrix8::identity());
        let ricci_scalar = riemann.ricci_scalar(&g_inv);
        
        // Compute sectional curvatures for principal planes
        let mut sectional_curvatures = HashMap::new();
        let basis = Matrix8::identity();
        for i in 0..8 {
            for j in i+1..8 {
                let v = basis.column(i).into();
                let w = basis.column(j).into();
                let k = riemann.sectional_curvature(&v, &w, &point.metric_tensor());
                sectional_curvatures.insert(format!("K_{}_{}", i, j), k);
            }
        }
        
        // Create sample
        let sample = CurvatureSample {
            timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            ricci_scalar,
            max_sectional: sectional_curvatures.values().cloned().fold(f64::NEG_INFINITY, f64::max),
            min_sectional: sectional_curvatures.values().cloned().fold(f64::INFINITY, f64::min),
            mean_sectional: sectional_curvatures.values().sum::<f64>() / sectional_curvatures.len() as f64,
            gauss_curvature: self.compute_gauss_curvature(&riemann, &point),
            scalar_curvature: ricci_scalar,
            weyl_tensor_norm: self.compute_weyl_norm(&riemann, &point),
        };
        
        // Update transition statistics
        self.transitions.entry(transition_hash)
            .and_modify(|t| {
                t.samples.push_back(sample.clone());
                if t.samples.len() > self.config.max_samples_per_transition {
                    t.samples.pop_front();
                }
                t.update_statistics();
                
                // Check for tipping point
                if let Some(event) = self.tipping_detector.detect(t) {
                    let _ = self.event_sender.send(event);
                }
                
                // Check early warnings
                let warnings = self.early_warning.analyze(&t.samples);
                for warning in warnings {
                    self.emit_warning(warning, &from, &to);
                }
            })
            .or_insert_with(|| {
                let mut t = TransitionCurvature {
                    from: from.clone(),
                    to: to.clone(),
                    samples: VecDeque::with_capacity(self.config.max_samples_per_transition),
                    mean_ricci_scalar: ricci_scalar,
                    variance_ricci_scalar: 0.0,
                    max_sectional_curvature: sample.max_sectional,
                    lyapunov_spectrum: vec![],
                    fisher_info_determinant: 0.0,
                    last_updated_ns: sample.timestamp_ns,
                    critical_slowing_down: 0.0,
                    variance_inflation: 1.0,
                    lag1_autocorrelation: 0.0,
                    skewness_change: 0.0,
                    spatial_correlation_range: 0.0,
                    hysteresis_width: 0.0,
                    basin_stability: 1.0,
                    committal_probability: 0.0,
                };
                t.samples.push_back(sample);
                t
            });
        
        // Update global statistics
        self.update_global_stats(ricci_scalar);
    }
    
    fn hash_transition(&self, from: &ClimateState, to: &ClimateState) -> u64 {
        let mut hasher = AHasher::default();
        from.hash(&mut hasher);
        to.hash(&mut hasher);
        hasher.finish()
    }
    
    fn compute_gauss_curvature(&self, riemann: &RiemannTensor, point: &ClimatePoint) -> f64 {
        // Gauss curvature: compute determinant-based measure for high dimensions
        let g = point.metric_tensor();
        let g_inv = g.try_inverse().unwrap_or(Matrix8::identity());
        
        // For high-dimensional case, compute generalized Gauss curvature
        // as geometric mean of sectional curvatures
        let mut sectional_sum = 0.0;
        let mut count = 0;
        
        // Sample representative 2-plane sectional curvatures
        for i in 0..8 {
            for j in (i + 1)..8 {
                // Sectional curvature K(e_i, e_j) = R(e_i, e_j, e_j, e_i) / (|e_i|^2 |e_j|^2 - <e_i, e_j>^2)
                let r_ijji = riemann.component(i, j, j, i);
                let g_ii = g[(i, i)];
                let g_jj = g[(j, j)];
                let g_ij = g[(i, j)];
                
                let denominator = g_ii * g_jj - g_ij * g_ij;
                if denominator.abs() > 1e-12 {
                    let sectional_k = r_ijji / denominator;
                    sectional_sum += sectional_k;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            sectional_sum / count as f64  // Average sectional curvature as proxy
        } else {
            0.0
        }
    }
    
    fn compute_weyl_norm(&self, riemann: &RiemannTensor, point: &ClimatePoint) -> f64 {
        // Weyl tensor = Riemann - (Ricci terms) - (scalar curvature terms)
        // Measures conformal curvature (shape distortion independent of volume)
        let g = point.metric_tensor();
        let g_inv = g.try_inverse().unwrap_or(Matrix8::identity());
        
        // Compute Ricci tensor and scalar curvature
        let ricci_scalar = riemann.ricci_scalar(&g_inv);
        
        // Weyl tensor components: C_ijkl = R_ijkl - (Ricci correction terms)
        // For 8D space, compute Frobenius norm of Weyl tensor
        let mut weyl_norm_squared = 0.0;
        let n = 8.0; // dimension
        
        // Sample computation over coordinate indices
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    for l in 0..8 {
                        if i != j && k != l {
                            // R_ijkl component
                            let riemann_comp = riemann.component(i, j, k, l);
                            
                            // Ricci correction terms (simplified)
                            let ricci_correction = if i == k && j == l {
                                ricci_scalar / (n * (n - 1.0))
                            } else {
                                0.0
                            };
                            
                            // Weyl component
                            let weyl_comp = riemann_comp - ricci_correction;
                            weyl_norm_squared += weyl_comp * weyl_comp;
                        }
                    }
                }
            }
        }
        
        weyl_norm_squared.sqrt()
    }
    
    fn emit_warning(&self, warning: EarlyWarningSignal, from: &ClimateState, to: &ClimateState) {
        let event = match warning {
            EarlyWarningSignal::CriticalSlowingDown(ac) => CurvatureEvent {
                timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
                event_type: EventType::CriticalSlowingDown,
                severity: if ac > 0.95 { Severity::Critical } else { Severity::Warning },
                from_state: from.clone(),
                to_state: to.clone(),
                curvature_value: ac,
                confidence: ac,
                message: format!("Critical slowing down detected: autocorrelation = {:.3}", ac),
            },
            EarlyWarningSignal::IncreasingVariance(ratio) => CurvatureEvent {
                timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
                event_type: EventType::VarianceInflation,
                severity: if ratio > 3.0 { Severity::Critical } else { Severity::Warning },
                from_state: from.clone(),
                to_state: to.clone(),
                curvature_value: ratio,
                confidence: (ratio - 1.0).min(1.0),
                message: format!("Variance inflation detected: {:.1}x increase", ratio),
            },
            _ => return,
        };
        
        let _ = self.event_sender.send(event);
    }
    
    fn update_global_stats(&self, ricci_scalar: f64) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_samples += 1;
            
            // Update running mean
            stats.mean_global_curvature = 
                (stats.mean_global_curvature * (stats.total_samples - 1) as f64 + ricci_scalar) 
                / stats.total_samples as f64;
            
            stats.max_observed_curvature = stats.max_observed_curvature.max(ricci_scalar);
            stats.min_observed_curvature = stats.min_observed_curvature.min(ricci_scalar);
        }
    }
    
    pub fn get_high_risk_transitions(&self) -> Vec<(ClimateState, ClimateState, f64)> {
        let mut risks = Vec::new();
        
        for entry in self.transitions.iter() {
            let transition = entry.value();
            let risk_score = transition.committal_probability * 0.4 
                          + transition.critical_slowing_down * 0.3
                          + (transition.variance_inflation / 10.0).min(1.0) * 0.3;
            
            if risk_score > 0.5 {
                risks.push((transition.from.clone(), transition.to.clone(), risk_score));
            }
        }
        
        risks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        risks
    }
    
    pub fn stream_events(&self) -> Receiver<CurvatureEvent> {
        self.event_receiver.clone()
    }
}

impl TransitionCurvature {
    fn update_statistics(&mut self) {
        if self.samples.len() < 2 {
            return;
        }
        
        let values: Vec<f64> = self.samples.iter().map(|s| s.ricci_scalar).collect();
        
        // Update mean and variance (Welford's algorithm)
        self.mean_ricci_scalar = values.iter().sum::<f64>() / values.len() as f64;
        self.variance_ricci_scalar = values.iter()
            .map(|&x| (x - self.mean_ricci_scalar).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        // Update max sectional curvature
        self.max_sectional_curvature = self.samples.iter()
            .map(|s| s.max_sectional)
            .fold(f64::NEG_INFINITY, f64::max);
        
        // Compute early warning indicators
        if values.len() >= 10 {
            // Critical slowing down (lag-1 autocorrelation)
            let mean = self.mean_ricci_scalar;
            let mut cov = 0.0;
            for i in 0..values.len()-1 {
                cov += (values[i] - mean) * (values[i+1] - mean);
            }
            self.lag1_autocorrelation = cov / ((values.len() - 1) as f64 * self.variance_ricci_scalar);
            
            // Variance inflation (compare recent to older)
            let mid = values.len() / 2;
            let var_old = Self::variance(&values[..mid]);
            let var_new = Self::variance(&values[mid..]);
            self.variance_inflation = if var_old > 0.0 { var_new / var_old } else { 1.0 };
            
            // Critical slowing down indicator
            self.critical_slowing_down = self.lag1_autocorrelation.abs();
        }
    }
    
    fn variance(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
    }
}

// ---
// PUBLIC API
// ---

impl ClimateCurvatureField {
    /// Process batch of climate observations
    pub fn ingest_batch(&self, observations: Vec<(ClimateState, ClimateState, ClimatePoint)>) {
        observations.par_iter().for_each(|(from, to, point)| {
            self.process_transition(from.clone(), to.clone(), point.clone());
        });
    }
    
    /// Get current global curvature statistics
    pub fn global_stats(&self) -> CurvatureStatistics {
        self.stats.read().unwrap().clone()
    }
    
    /// Export data for visualization
    pub fn export_for_viz(&self) -> CurvatureVisualization {
        let mut nodes = HashMap::new();
        let mut edges = Vec::new();
        
        for entry in self.transitions.iter() {
            let t = entry.value();
            
            // Add nodes
            nodes.entry(t.from.clone()).or_insert(NodeViz {
                state: t.from.clone(),
                stability: t.basin_stability,
                regime_type: self.classify_regime(&t.from),
            });
            
            nodes.entry(t.to.clone()).or_insert(NodeViz {
                state: t.to.clone(),
                stability: 1.0,
                regime_type: self.classify_regime(&t.to),
            });
            
            // Add edge
            edges.push(EdgeViz {
                from: t.from.clone(),
                to: t.to.clone(),
                curvature: t.mean_ricci_scalar,
                risk: t.committal_probability,
                samples: t.samples.len(),
            });
        }
        
        CurvatureVisualization {
            nodes: nodes.into_values().collect(),
            edges,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }
    
    fn classify_regime(&self, state: &ClimateState) -> RegimeType {
        match state {
            ClimateState::Holocene | ClimateState::Preindustrial => RegimeType::Stable,
            ClimateState::HothouseEarth | ClimateState::SnowballEarth => RegimeType::Extreme,
            ClimateState::AMOCCollapsed | ClimateState::ArcticIceFree => RegimeType::Tipped,
            _ => RegimeType::Transitional,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureVisualization {
    pub nodes: Vec<NodeViz>,
    pub edges: Vec<EdgeViz>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeViz {
    pub state: ClimateState,
    pub stability: f64,
    pub regime_type: RegimeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeViz {
    pub from: ClimateState,
    pub to: ClimateState,
    pub curvature: f64,
    pub risk: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeType {
    Stable,
    Transitional,
    Tipped,
    Extreme,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_riemann_tensor_computation() {
        let point = ClimateState::TwoC.reference_point();
        let riemann = RiemannTensor::compute(&point, 1e-6);
        assert_eq!(riemann.dimension, 8);
    }
    
    #[test]
    fn test_curvature_field_streaming() {
        let field = ClimateCurvatureField::new(CurvatureConfig::default());
        
        // Simulate transitions
        field.process_transition(
            ClimateState::Holocene,
            ClimateState::OnePointFiveC,
            ClimateState::OnePointFiveC.reference_point()
        );
        
        let stats = field.global_stats();
        assert_eq!(stats.total_samples, 1);
    }
}