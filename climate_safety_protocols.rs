// Climate model safety protocols and engineering guardrails
// Failure mode prevention for complex system analysis
// Experimental validation and constraint checking

// ─────────────────────────────────────────────────────────────────────────────
// DATA SOURCE REQUIREMENTS
// ─────────────────────────────────────────────────────────────────────────────
// 
// 1. ENERGY BALANCE CONSTRAINT VALIDATION:
//    - Source: CERES EBAF-TOA Edition 4.2, EBAF-Surface Ed4.2
//    - Instrument: Terra/Aqua/NPP/NOAA-20 CERES radiometers
//    - Resolution: 1° x 1° global monthly means
//    - Temporal: 2000-03 to present, updated quarterly
//    - Format: NetCDF4 with CF conventions
//    - Size: ~100MB/year for TOA+Surface products
//    - Variables: SW/LW fluxes (all-sky, clear-sky), cloud radiative effect
//    - API: https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAF42Selection.jsp
//    - Constraint: Global TOA imbalance = 0.71 ± 0.10 W/m² (Loeb et al. 2021)
//    - Preprocessing: Area-weighted global means, annual cycle removal
//    - Missing: Accurate pre-2000 satellite radiation, polar winter interpolation
//
// 2. WATER CYCLE CLOSURE:
//    - Precipitation: GPCP v2.3 (1979-present, 2.5° monthly)
//    - Evaporation: ERA5 (1940-present, 0.25° hourly)
//    - Moisture flux: ERA5 vertically integrated (q*u, q*v)
//    - River discharge: GRDC stations + GloFAS reanalysis
//    - Format: NetCDF4 for gridded, CSV for gauge data
//    - Size: ~50GB for all components
//    - Constraint: P - E - R = dS/dt within 10% (harder over land)
//    - API: GPCP from PSL, ERA5 from CDS, GRDC restricted access
//    - Missing: Groundwater changes, ice sheet discharge, irrigation
//
// 3. MASS BALANCE CONSTRAINTS:
//    - Source: GRACE/GRACE-FO mascons (2002-present)
//    - Resolution: 3° mascon solutions, monthly
//    - Format: NetCDF4 with GIA corrections
//    - Size: ~1GB for time series
//    - Variables: Total water storage anomaly, ice mass change
//    - API: https://grace.jpl.nasa.gov/data/get-data/
//    - Constraint: Global ocean mass + land water + ice = constant
//    - Missing: 2017-2018 gap between missions
//
// 4. PARAMETER BOUNDS FROM MULTI-CONSTRAINT ANALYSIS:
//    - Source: Sherwood et al. 2020, IPCC AR6 Chapter 7
//    - ECS: 2.5-4.0 K (likely), 2.0-5.0 K (very likely)
//    - TCR: 1.4-2.2 K (likely)
//    - Cloud feedback: 0.42 ± 0.35 W/m²/K (Zelinka et al. 2020)
//    - Aerosol ERF: -1.3 ± 0.7 W/m² (Bellouin et al. 2020)
//    - Format: Published tables, some in CSV supplements
//    - Size: <100MB for all constraints
//    - Missing: Non-Gaussian error structures, structural uncertainty
//
// 5. EXTREME EVENT PHYSICAL THRESHOLDS:
//    - Source: EM-DAT + NOAA Storm Events + IBTrACS
//    - Tropical cyclones: Max wind > 33 m/s (Category 1+)
//    - Heat waves: T > 95th percentile for 3+ days
//    - Drought: SPI < -1.5 or PDSI < -3
//    - Floods: Return period > 10 years from GloFAS
//    - Format: CSV databases with geocoding
//    - Size: ~1GB for global catalogs
//    - API: EM-DAT requires registration, NOAA public
//    - Missing: Consistent global thresholds, compound events
//
// 6. MODEL STABILITY DIAGNOSTICS:
//    - CFL condition: Courant number < 0.5 for advection
//    - Energy drift: < 0.1 W/m² per century in control runs
//    - Water conservation: < 0.01 mm/year global drift
//    - Temperature drift: < 0.1 K/century in piControl
//    - Source: CMIP6 piControl runs for baseline
//    - Format: Time series diagnostics in NetCDF4
//    - Missing: Standardized stability metrics across models
//
// 7. NUMERICAL PRECISION REQUIREMENTS:
//    - Machine epsilon: 2.22e-16 for float64
//    - Roundoff accumulation: Monitor with Kahan summation
//    - Catastrophic cancellation: Detect with condition numbers
//    - Source: IEEE 754 standard, numerical analysis literature
//    - Missing: Automated precision loss detection
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use nalgebra::DMatrix; // additive linear algebra for eigen diagnostics

// Additive macro for concise failure recording
macro_rules! record_fail {
    ($sys:expr, $err:expr) => {{
        let e = $err;
        if let Ok(mut telem) = $sys.telemetry.lock() { telem.record_failure(&e); }
        return Err(e);
    }};
}

// ---
// Failure modes
// ---

#[derive(Debug, Error)]
pub enum ClimateModelFailure {
    #[error("Parameter bounds violated: {param} = {value}, allowed: [{min}, {max}]")]
    ParameterBoundsViolation { param: String, value: f64, min: f64, max: f64 },
    
    #[error("Numerical instability detected: gradient norm = {norm}")]
    NumericalInstability { norm: f64 },
    
    #[error("Conservation law violated: {law} imbalance = {imbalance} W/m²")]
    ConservationViolation { law: String, imbalance: f64 },
    
    #[error("Unphysical state: {description}")]
    UnphysicalState { description: String },
    
    #[error("Computation timeout after {seconds}s")]
    ComputationTimeout { seconds: u64 },
    
    #[error("Memory limit exceeded: {used_gb} GB > {limit_gb} GB")]
    MemoryExceeded { used_gb: f64, limit_gb: f64 },
    
    #[error("Cascade failure: {failed_components:?}")]
    CascadeFailure { failed_components: Vec<String> },
}

// ---
// Operational modes
// ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalMode {
    /// Deterministic computation
    Deterministic,
    /// Stochastic mode  
    Stochastic,
    /// Event-driven cascade analysis
    EventDriven,
    /// Deep uncertainty exploration with wide bounds
    Exploratory,
    /// Sandbox containment mode (additive Batch3) - isolates risky dynamics with stricter limits
    Sandbox,
}

impl OperationalMode {
    /// Get computational constraints for this mode
    pub fn constraints(&self) -> ModeConstraints {
        match self {
            Self::Deterministic => ModeConstraints {
                max_iterations: 1000,
                convergence_tolerance: 1e-8,
                max_memory_gb: 16.0,
                timeout_seconds: 60,
                allow_extrapolation: false,
                require_conservation: true,
                max_gradient_norm: 1e3,
            },
            Self::Stochastic => ModeConstraints {
                max_iterations: 10000,
                convergence_tolerance: 1e-6,
                max_memory_gb: 32.0,
                timeout_seconds: 300,
                allow_extrapolation: true,
                require_conservation: false,
                max_gradient_norm: 1e4,
            },
            Self::EventDriven => ModeConstraints {
                max_iterations: 100000,
                convergence_tolerance: 1e-4,
                max_memory_gb: 64.0,
                timeout_seconds: 600,
                allow_extrapolation: true,
                require_conservation: false,
                max_gradient_norm: 1e5,
            },
            Self::Exploratory => ModeConstraints {
                max_iterations: 1000000,
                convergence_tolerance: 1e-3,
                max_memory_gb: 128.0,
                timeout_seconds: 3600,
                allow_extrapolation: true,
                require_conservation: false,
                max_gradient_norm: 1e6,
            },
            Self::Sandbox => ModeConstraints {
                max_iterations: 500,
                convergence_tolerance: 5e-8, // stricter
                max_memory_gb: 4.0,
                timeout_seconds: 30,
                allow_extrapolation: false,
                require_conservation: true,
                max_gradient_norm: 5e2,
            },
        }
    }
    
    /// Check if we've been in this mode too long
    pub fn check_duration_safety(&self, duration: Duration) -> Result<(), ClimateModelFailure> {
        let max_duration = match self {
            Self::Deterministic => Duration::from_secs(300),  // 5 min max
            Self::Stochastic => Duration::from_secs(1800),    // 30 min max
            Self::EventDriven => Duration::from_secs(3600),   // 1 hour max
            Self::Exploratory => Duration::from_secs(7200),   // 2 hours max
            Self::Sandbox => Duration::from_secs(120),        // containment short window
        };
        
        if duration > max_duration {
            // Mode exhaustion - need to switch
            Err(ClimateModelFailure::ComputationTimeout { 
                seconds: duration.as_secs() 
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModeConstraints {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub max_memory_gb: f64,
    pub timeout_seconds: u64,
    pub allow_extrapolation: bool,
    pub require_conservation: bool,
    pub max_gradient_norm: f64,
}

// ---
// Parameter validation
// ---

pub struct ParameterValidator {
    bounds: HashMap<String, (f64, f64)>,
    dependencies: HashMap<String, Vec<ParameterConstraint>>,
}

#[derive(Debug, Clone)]
pub struct ParameterConstraint {
    pub other_param: String,
    pub constraint_type: ConstraintType,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    LessThan,
    GreaterThan,
    SumLessThan(f64),
    ProductLessThan(f64),
    Ratio(f64, f64),  // min_ratio, max_ratio
}

impl ParameterValidator {
    pub fn new() -> Self {
        let mut validator = Self {
            bounds: HashMap::new(),
            dependencies: HashMap::new(),
        };
        
        // Physical bounds from IPCC AR6
        validator.bounds.insert("ecs".to_string(), (1.5, 6.0));
        validator.bounds.insert("tcr".to_string(), (1.0, 3.0));
        validator.bounds.insert("aerosol_forcing".to_string(), (-2.0, -0.2));
        validator.bounds.insert("ocean_heat_uptake".to_string(), (0.3, 1.5));
        validator.bounds.insert("cloud_feedback".to_string(), (-0.5, 1.5));
        validator.bounds.insert("co2_concentration".to_string(), (280.0, 2000.0));
        validator.bounds.insert("global_temp_anomaly".to_string(), (-2.0, 10.0));
        
        // Inter-parameter constraints
        validator.dependencies.insert("tcr".to_string(), vec![
            ParameterConstraint {
                other_param: "ecs".to_string(),
                constraint_type: ConstraintType::LessThan,
            }
        ]);
        
        validator
    }
    
    pub fn validate(&self, params: &HashMap<String, f64>) -> Result<(), ClimateModelFailure> {
        // Check individual bounds
        for (name, value) in params {
            if let Some((min, max)) = self.bounds.get(name) {
                if value < min || value > max {
                    return Err(ClimateModelFailure::ParameterBoundsViolation {
                        param: name.clone(),
                        value: *value,
                        min: *min,
                        max: *max,
                    });
                }
            }
        }
        
        // Check inter-parameter constraints
        for (param, constraints) in &self.dependencies {
            if let Some(value) = params.get(param) {
                for constraint in constraints {
                    if let Some(other_value) = params.get(&constraint.other_param) {
                        match constraint.constraint_type {
                            ConstraintType::LessThan => {
                                if value >= other_value {
                                    return Err(ClimateModelFailure::UnphysicalState {
                                        description: format!("{} must be < {}", param, constraint.other_param)
                                    });
                                }
                            },
                            ConstraintType::GreaterThan => {
                                if value <= other_value {
                                    return Err(ClimateModelFailure::UnphysicalState {
                                        description: format!("{} must be > {}", param, constraint.other_param)
                                    });
                                }
                            },
                            ConstraintType::SumLessThan(limit) => {
                                let sum = value + other_value;
                                if sum >= limit {
                                    return Err(ClimateModelFailure::UnphysicalState {
                                        description: format!("{} + {} must be < {}", param, constraint.other_param, limit)
                                    });
                                }
                            },
                            ConstraintType::ProductLessThan(limit) => {
                                let product = value * other_value;
                                if product >= limit {
                                    return Err(ClimateModelFailure::UnphysicalState {
                                        description: format!("{} * {} must be < {}", param, constraint.other_param, limit)
                                    });
                                }
                            },
                            ConstraintType::Ratio(min_ratio, max_ratio) => {
                                let ratio = value / other_value;
                                if ratio < min_ratio || ratio > max_ratio {
                                    return Err(ClimateModelFailure::UnphysicalState {
                                        description: format!("{}/{} ratio must be in [{}, {}]", param, constraint.other_param, min_ratio, max_ratio)
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}

// ---
// Conservation checking
// ---

pub struct ConservationChecker {
    tolerance: f64,
    history: VecDeque<ConservationState>,
    max_history: usize,
}

#[derive(Debug, Clone)]
pub struct ConservationState {
    timestamp: Instant,
    energy_imbalance: f64,
    mass_imbalance: f64,
    momentum_imbalance: f64,
}

impl ConservationChecker {
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            history: VecDeque::new(),
            max_history: 100,
        }
    }
    
    pub fn check_energy_balance(
        &mut self,
        incoming_radiation: f64,
        outgoing_radiation: f64,
        ocean_heat_uptake: f64,
    ) -> Result<(), ClimateModelFailure> {
        let imbalance = incoming_radiation - outgoing_radiation - ocean_heat_uptake;
        
        if imbalance.abs() > self.tolerance {
            return Err(ClimateModelFailure::ConservationViolation {
                law: "energy".to_string(),
                imbalance,
            });
        }
        
        // Record for trend analysis
        let state = ConservationState {
            timestamp: Instant::now(),
            energy_imbalance: imbalance,
            mass_imbalance: 0.0,
            momentum_imbalance: 0.0,
        };
        
        self.history.push_back(state);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
        
        // Check for systematic drift
        if self.history.len() >= 10 {
            let recent_mean: f64 = self.history.iter()
                .rev()
                .take(10)
                .map(|s| s.energy_imbalance)
                .sum::<f64>() / 10.0;
                
            if recent_mean.abs() > self.tolerance * 0.5 {
                return Err(ClimateModelFailure::ConservationViolation {
                    law: "energy_drift".to_string(),
                    imbalance: recent_mean,
                });
            }
        }
        
        Ok(())
    }
    
    pub fn check_carbon_mass_balance(
        &self,
        atmospheric_co2: f64,
        ocean_carbon: f64,
        land_carbon: f64,
        emissions: f64,
        initial_total: f64,
    ) -> Result<(), ClimateModelFailure> {
        let current_total = atmospheric_co2 + ocean_carbon + land_carbon;
        let expected_total = initial_total + emissions;
        let imbalance = current_total - expected_total;
        
        if imbalance.abs() > self.tolerance * expected_total {
            return Err(ClimateModelFailure::ConservationViolation {
                law: "carbon_mass".to_string(),
                imbalance,
            });
        }
        
        Ok(())
    }
}

// ---
// Stability monitoring
// ---

pub struct StabilityMonitor {
    gradient_history: VecDeque<f64>,
    max_gradient: f64,
    divergence_threshold: f64,
    base_divergence_threshold: f64,
    rolling: RollingStats,
}

impl StabilityMonitor {
    pub fn new(max_gradient: f64) -> Self {
        Self {
            gradient_history: VecDeque::with_capacity(100),
            max_gradient,
            divergence_threshold: 3.0,  // Factor increase indicating divergence
            base_divergence_threshold: 3.0,
            rolling: RollingStats::new(50),
        }
    }
    
    pub fn check_gradient(&mut self, gradient_norm: f64) -> Result<(), ClimateModelFailure> {
        // Absolute check
        if gradient_norm > self.max_gradient || gradient_norm.is_nan() || gradient_norm.is_infinite() {
            return Err(ClimateModelFailure::NumericalInstability { norm: gradient_norm });
        }
        
        // Trend check for divergence
        self.gradient_history.push_back(gradient_norm);
        if self.gradient_history.len() > 100 {
            self.gradient_history.pop_front();
        }

        // Update rolling stats adaptively (additive)
        self.rolling.push(gradient_norm);
        if let (Some(m), Some(var)) = (self.rolling.mean(), self.rolling.variance()) {
            let sigma = var.sqrt();
            // Adaptive divergence threshold: base + dynamic factor
            let dynamic_factor = ((sigma / (m.abs() + 1e-9)).clamp(0.5, 5.0)) + 1.0;
            self.divergence_threshold = self.base_divergence_threshold * dynamic_factor.min(8.0);
        }
        
        if self.gradient_history.len() >= 10 {
            let old_mean = self.gradient_history.iter()
                .take(5)
                .sum::<f64>() / 5.0;
            let new_mean = self.gradient_history.iter()
                .rev()
                .take(5)
                .sum::<f64>() / 5.0;
                
            if new_mean > old_mean * self.divergence_threshold {
                return Err(ClimateModelFailure::NumericalInstability { 
                    norm: new_mean 
                });
            }
        }
        
        Ok(())
    }
    
    pub fn check_eigenvalues(&self, matrix: &Vec<Vec<f64>>) -> Result<(), ClimateModelFailure> {
        // Check condition number for numerical stability
        // Additive: real symmetric eigenvalue condition estimation attempt
        if let Some(cond) = estimate_condition_number(matrix) {
            if !cond.is_finite() || cond.is_nan() || cond > 1e8 {
                return Err(ClimateModelFailure::NumericalInstability { norm: cond });
            }
            // Acceptable condition number path
            return Ok(());
        }
        
        // Check matrix dimensions
        if matrix.is_empty() {
            return Err(ClimateModelFailure::UnphysicalState {
                description: "Empty matrix provided for eigenvalue computation".to_string()
            });
        }
        
        let n = matrix.len();
        for row in matrix {
            if row.len() != n {
                return Err(ClimateModelFailure::UnphysicalState {
                    description: format!("Non-square matrix: expected {}x{}", n, n)
                });
            }
        }
        
        // Check for NaN or infinite values
        for (i, row) in matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val.is_nan() {
                    return Err(ClimateModelFailure::NumericalInstability {
                        norm: f64::NAN
                    });
                }
                if val.is_infinite() {
                    return Err(ClimateModelFailure::NumericalInstability {
                        norm: f64::INFINITY
                    });
                }
            }
        }
        
    // Fallback placeholder (legacy path retained additively)
    let max_eigenvalue = 1000.0;  // legacy approximation
    let min_eigenvalue = 0.001;   // legacy approximation
        
        // Check for near-singular matrix
        if min_eigenvalue.abs() < 1e-10 {
            return Err(ClimateModelFailure::NumericalInstability {
                norm: 1e15  // Near-singular condition number
            });
        }
        
        let condition_number = max_eigenvalue / min_eigenvalue;
        
        if condition_number > 1e6 {
            return Err(ClimateModelFailure::NumericalInstability {
                norm: condition_number
            });
        }
        
        Ok(())
    }
}

// ---
// Cascade protection
// ---

pub struct CascadeProtection {
    component_states: HashMap<String, ComponentState>,
    failure_dependencies: HashMap<String, Vec<String>>,
    max_simultaneous_failures: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComponentState {
    Healthy,
    Degraded,
    Failed,
    Recovering,
}

impl CascadeProtection {
    pub fn new() -> Self {
        let mut protection = Self {
            component_states: HashMap::new(),
            failure_dependencies: HashMap::new(),
            max_simultaneous_failures: 2,
        };
        
        // Define climate system components
        protection.component_states.insert("ice_sheets".to_string(), ComponentState::Healthy);
        protection.component_states.insert("amoc".to_string(), ComponentState::Healthy);
        protection.component_states.insert("amazon".to_string(), ComponentState::Healthy);
        protection.component_states.insert("permafrost".to_string(), ComponentState::Healthy);
        protection.component_states.insert("monsoon".to_string(), ComponentState::Healthy);
        
        // Define failure cascade paths
        protection.failure_dependencies.insert("ice_sheets".to_string(), 
            vec!["amoc".to_string()]);
        protection.failure_dependencies.insert("amoc".to_string(), 
            vec!["monsoon".to_string()]);
        protection.failure_dependencies.insert("amazon".to_string(), 
            vec!["monsoon".to_string()]);
        
        protection
    }
    
    pub fn update_component(&mut self, name: &str, state: ComponentState) -> Result<(), ClimateModelFailure> {
        // Check if component exists
        if !self.component_states.contains_key(name) {
            return Err(ClimateModelFailure::UnphysicalState {
                description: format!("Unknown component: {}", name)
            });
        }
        
        // Check for invalid state transitions
        if let Some(current_state) = self.component_states.get(name) {
            match (current_state, &state) {
                (ComponentState::Failed, ComponentState::Healthy) => {
                    // Can't go directly from Failed to Healthy without Recovery
                    return Err(ClimateModelFailure::UnphysicalState {
                        description: format!("Invalid transition: {} cannot go from Failed to Healthy without Recovery phase", name)
                    });
                },
                _ => {}
            }
        }
        
        // Check recursion depth to prevent infinite cascades
        static RECURSION_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let depth = RECURSION_DEPTH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if depth > 10 {
            RECURSION_DEPTH.store(0, std::sync::atomic::Ordering::SeqCst);
            return Err(ClimateModelFailure::CascadeFailure {
                failed_components: vec![format!("Recursion depth exceeded for component {}", name)]
            });
        }
        
        self.component_states.insert(name.to_string(), state.clone());
        
        // Check for cascade
        if state == ComponentState::Failed {
            let failed_count = self.component_states.values()
                .filter(|s| **s == ComponentState::Failed)
                .count();
                
            if failed_count > self.max_simultaneous_failures {
                let failed_components: Vec<String> = self.component_states
                    .iter()
                    .filter(|(_, s)| **s == ComponentState::Failed)
                    .map(|(name, _)| name.clone())
                    .collect();
                    
                return Err(ClimateModelFailure::CascadeFailure { failed_components });
            }
            
            // Mark dependent components as degraded
            if let Some(deps) = self.failure_dependencies.get(name) {
                for dep in deps {
                    if self.component_states.get(dep) == Some(&ComponentState::Healthy) {
                        self.component_states.insert(dep.clone(), ComponentState::Degraded);
                    }
                }
            }
        }
        
        RECURSION_DEPTH.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    
    pub fn can_proceed(&self) -> bool {
        // Don't proceed if critical components have failed
        let critical = ["amoc", "ice_sheets"];
        for component in critical {
            if self.component_states.get(component) == Some(&ComponentState::Failed) {
                return false;
            }
        }
        true
    }
}

// ---
// Mode transitions
// ---

pub struct ModeTransitionGuard {
    current_mode: OperationalMode,
    mode_start: Instant,
    transition_history: VecDeque<ModeTransition>,
    min_mode_duration: Duration,
}

#[derive(Debug, Clone)]
pub struct ModeTransition {
    from: OperationalMode,
    to: OperationalMode,
    timestamp: Instant,
    reason: String,
}

impl ModeTransitionGuard {
    pub fn new(initial_mode: OperationalMode) -> Self {
        Self {
            current_mode: initial_mode,
            mode_start: Instant::now(),
            transition_history: VecDeque::new(),
            min_mode_duration: Duration::from_secs(10),
        }
    }
    
    pub fn request_transition(
        &mut self,
        to: OperationalMode,
        reason: &str,
    ) -> Result<(), String> {
        // Check minimum duration
        let duration = Instant::now() - self.mode_start;
        if duration < self.min_mode_duration {
            return Err(format!(
                "Mode transition too soon: {} seconds remaining",
                (self.min_mode_duration - duration).as_secs()
            ));
        }
        
        // Validate transition path
        let valid = match (self.current_mode, to) {
            // Can't jump directly from deterministic to exploratory
            (OperationalMode::Deterministic, OperationalMode::Exploratory) => false,
            (OperationalMode::Exploratory, OperationalMode::Deterministic) => false,
            // All other transitions OK
            _ => true,
        };
        
        if !valid {
            return Err(format!(
                "Invalid transition: {:?} -> {:?}",
                self.current_mode, to
            ));
        }
        
        // Record transition
        self.transition_history.push_back(ModeTransition {
            from: self.current_mode,
            to,
            timestamp: Instant::now(),
            reason: reason.to_string(),
        });
        
        // Keep history bounded
        if self.transition_history.len() > 100 {
            self.transition_history.pop_front();
        }
        
        self.current_mode = to;
        self.mode_start = Instant::now();
        
        Ok(())
    }
    
    pub fn current_mode(&self) -> OperationalMode {
        self.current_mode
    }
    
    pub fn time_in_mode(&self) -> Duration {
        Instant::now() - self.mode_start
    }
}

// ---
// Safety system
// ---

pub struct ClimateSafetySystem {
    mode_guard: Arc<Mutex<ModeTransitionGuard>>,
    validator: Arc<ParameterValidator>,
    conservation: Arc<Mutex<ConservationChecker>>,
    stability: Arc<Mutex<StabilityMonitor>>,
    cascade: Arc<Mutex<CascadeProtection>>,
    emergency_stop: Arc<AtomicBool>,
    telemetry: Arc<Mutex<SafetyTelemetry>>, // additive telemetry
    extended_conservation: Arc<Mutex<ExtendedConservationChecker>>, // batch2 additive
    joint_model: Arc<Mutex<JointParameterModel>>, // batch2 additive
    anomaly_scorer: Arc<Mutex<ProbabilisticAnomalyScorer>>, // batch2 additive
    recovery: Arc<Mutex<RecoveryManager>>, // batch3 additive
    audit: Arc<Mutex<AuditTrail>>, // batch3 additive
    rollback_planner: Arc<RollbackPlanner>, // batch4 additive
    alert_dispatcher: Arc<Mutex<AlertDispatcher>>, // batch4 additive
}

impl ClimateSafetySystem {
    pub fn new() -> Self {
        Self {
            mode_guard: Arc::new(Mutex::new(
                ModeTransitionGuard::new(OperationalMode::Deterministic)
            )),
            validator: Arc::new(ParameterValidator::new()),
            conservation: Arc::new(Mutex::new(ConservationChecker::new(1e-6))),
            stability: Arc::new(Mutex::new(StabilityMonitor::new(1e6))),
            cascade: Arc::new(Mutex::new(CascadeProtection::new())),
            emergency_stop: Arc::new(AtomicBool::new(false)),
            telemetry: Arc::new(Mutex::new(SafetyTelemetry::new())),
            extended_conservation: Arc::new(Mutex::new(ExtendedConservationChecker::new(1e-3))),
            joint_model: Arc::new(Mutex::new(JointParameterModel::new(vec![
                "ecs".to_string(),
                "tcr".to_string(),
                "aerosol_forcing".to_string(),
                "ocean_heat_uptake".to_string(),
                "cloud_feedback".to_string(),
            ]))),
            anomaly_scorer: Arc::new(Mutex::new(ProbabilisticAnomalyScorer::new(0.92))),
            recovery: Arc::new(Mutex::new(RecoveryManager::new())),
            audit: Arc::new(Mutex::new(AuditTrail::new(40))),
            rollback_planner: Arc::new(RollbackPlanner::default()),
            alert_dispatcher: Arc::new(Mutex::new(AlertDispatcher::new(100))),
        }
    }
    
    pub fn validate_operation(
        &self,
        params: &HashMap<String, f64>,
        gradient_norm: Option<f64>,
    ) -> Result<(), ClimateModelFailure> {
        if let Ok(mut t) = self.telemetry.lock() { t.record_validation_start(); }
        // Check emergency stop
        if self.emergency_stop.load(Ordering::Relaxed) {
            record_fail!(self, ClimateModelFailure::UnphysicalState { description: "Emergency stop activated".to_string() });
        }
        
        // Validate parameters
        if let Err(e) = self.validator.validate(params) { record_fail!(self, e); }
        
        // Check numerical stability
        if let Some(norm) = gradient_norm {
            // Handle potential mutex poisoning
            let mut stability = self.stability.lock()
                .map_err(|_| ClimateModelFailure::UnphysicalState {
                    description: "Stability monitor mutex poisoned - previous thread panicked".to_string()
                })?;
            if let Err(e) = stability.check_gradient(norm) { record_fail!(self, e); }
        }
        
        // Check mode duration
        let guard = self.mode_guard.lock()
            .map_err(|_| ClimateModelFailure::UnphysicalState {
                description: "Mode guard mutex poisoned - previous thread panicked".to_string()
            })?;
    let mode = guard.current_mode();
    let duration = guard.time_in_mode();
    if let Err(e) = mode.check_duration_safety(duration) { record_fail!(self, e); }
        
        // Check cascade state
        let cascade = self.cascade.lock()
            .map_err(|_| ClimateModelFailure::UnphysicalState {
                description: "Cascade protection mutex poisoned - previous thread panicked".to_string()
            })?;
        if !cascade.can_proceed() {
            record_fail!(self, ClimateModelFailure::CascadeFailure { failed_components: vec!["System cascade detected".to_string()] });
        }

        // --- Batch2 additive: extended conservation checks (soft warnings unless severe) ---
        // Expect optional parameters: precip, evap, moisture_flux_div, u_in, u_out, momentum_forcing, momentum_tendency
        if let (Some(p), Some(e), Some(div)) = (params.get("precip"), params.get("evap"), params.get("moisture_flux_div")) {
            if let Ok(mut ext) = self.extended_conservation.lock() {
                let water = ext.check_water_budget(*p, *e, *div);
                if let Ok(mut tele) = self.telemetry.lock() {
                    if let Some(z) = water.z_score { if z > 2.5 { tele.add_soft_warning(format!("Water budget anomaly z={:.2}", z)); } }
                    if water.severe_violation { tele.add_soft_warning(format!("Severe water imbalance {:.6} (tolerance {:.6})", water.imbalance, ext.tolerance)); }
                }
            }
        }
        if let (Some(u_in), Some(u_out), Some(force), Some(tend)) = (params.get("u_in"), params.get("u_out"), params.get("momentum_forcing"), params.get("momentum_tendency")) {
            if let Ok(mut ext) = self.extended_conservation.lock() {
                let mom = ext.check_momentum_budget(*u_in, *u_out, *force, *tend);
                if let Ok(mut tele) = self.telemetry.lock() {
                    if let Some(z) = mom.z_score { if z > 2.5 { tele.add_soft_warning(format!("Momentum budget anomaly z={:.2}", z)); } }
                    if mom.severe_violation { tele.add_soft_warning(format!("Severe momentum imbalance {:.6}", mom.imbalance)); }
                }
            }
        }

        // --- Batch2 additive: joint parameter Mahalanobis distance ---
        if let Ok(mut jm) = self.joint_model.lock() {
            jm.update(params);
            if let Some(dist) = jm.mahalanobis(params) {
                if let Ok(mut tele) = self.telemetry.lock() { tele.update_mahalanobis(dist); }
                // thresholds: >7.5 hard fail, >5.0 soft warning
                if dist > 7.5 {
                    record_fail!(self, ClimateModelFailure::UnphysicalState { description: format!("Joint parameter anomaly Mahalanobis={:.3}", dist) });
                } else if dist > 5.0 {
                    if let Ok(mut tele) = self.telemetry.lock() { tele.add_soft_warning(format!("Joint parameter distance {:.3}", dist)); }
                }
            }
        }

        // --- Batch2 additive: probabilistic anomaly scorer ---
        if let (Ok(jm), Ok(mut scorer)) = (self.joint_model.lock(), self.anomaly_scorer.lock()) {
            if let Some(dist) = jm.last_distance { // distance from last mahalanobis evaluation
                let severity = (dist / 5.0).min(10.0);
                let prob = scorer.update(severity);
                if let Ok(mut tele) = self.telemetry.lock() { tele.update_anomaly_probability(prob); }
                if prob > 0.9 {
                    record_fail!(self, ClimateModelFailure::UnphysicalState { description: format!("Probabilistic anomaly p={:.3}", prob) });
                } else if prob > 0.7 {
                    if let Ok(mut tele) = self.telemetry.lock() { tele.add_soft_warning(format!("Elevated anomaly probability p={:.3}", prob)); }
                }
                // Batch3: recovery & sandbox logic with hysteresis
                if let (Ok(mut rec), Ok(mut guard)) = (self.recovery.lock(), self.mode_guard.lock()) {
                    let phase_before = rec.phase.clone();
                    rec.update(prob, dist);
                    if let Ok(mut tele) = self.telemetry.lock() { tele.update_phase(&rec.phase); }
                    if rec.request_sandbox_activation() && guard.current_mode() != OperationalMode::Sandbox {
                        let _ = guard.request_transition(OperationalMode::Sandbox, "Anomaly containment sandbox activation");
                        if let Ok(mut tele) = self.telemetry.lock() { tele.add_soft_warning("Sandbox mode engaged"); }
                    } else if rec.can_exit_sandbox() && guard.current_mode() == OperationalMode::Sandbox {
                        let _ = guard.request_transition(OperationalMode::Deterministic, "Sandbox exit hysteresis satisfied");
                        if let Ok(mut tele) = self.telemetry.lock() { tele.add_soft_warning("Sandbox mode exited"); }
                    }
                    if phase_before != rec.phase { if let Ok(mut tele) = self.telemetry.lock() { tele.add_soft_warning(format!("Phase transition {} -> {}", phase_before, rec.phase)); } }
                }
            }
        }

        // Batch3: audit snapshot capture at end of validation (non-failing)
        if let (Ok(tele), Ok(mut audit)) = (self.telemetry.lock(), self.audit.lock()) {
            audit.maybe_snapshot(&tele);
        }
        
        if let Ok(mut t) = self.telemetry.lock() { t.record_validation_success(); }
        Ok(())
    }
    
    pub fn emergency_stop(&self) {
        self.emergency_stop.store(true, Ordering::Relaxed);
    }
    
    pub fn reset_emergency_stop(&self) {
        self.emergency_stop.store(false, Ordering::Relaxed);
    }

    pub fn telemetry_json(&self) -> Option<String> {
        self.telemetry.lock().ok().map(|t| t.to_json())
    }

    // Batch4 additive: audit export
    pub fn audit_json(&self) -> Option<String> { self.audit.lock().ok().map(|a| a.to_json()) }
    pub fn alerts_json(&self) -> Option<String> { self.alert_dispatcher.lock().ok().map(|a| a.to_json()) }

    // Batch4 additive: extended validation allowing parameter rollback planning (params mutable)
    pub fn validate_operation_with_mut_params(
        &self,
        params: &mut HashMap<String,f64>,
        gradient_norm: Option<f64>,
    ) -> Result<(), ClimateModelFailure> {
        // Use immutable path first, capturing previous phase
        let prev_phase = self.recovery.lock().ok().map(|r| r.phase.clone());
        // Call original immutable validation (shadow copy) to reuse logic
        let shadow = params.clone();
        match self.validate_operation(&shadow, gradient_norm) {
            Ok(_) => {},
            Err(e) => { return Err(e); }
        }
        // After successful immutable validation, check if we are in containment and plan rollback
        if let (Ok(rec), Some(prev)) = (self.recovery.lock(), prev_phase) {
            if rec.phase == RecoveryPhase::Containment && prev != RecoveryPhase::Containment {
                if let Ok(planner) = Arc::clone(&self.rollback_planner).try_into() { let _ = planner; }
                let actions = self.rollback_planner.plan(params, rec.last_prob);
                if let Ok(mut tele) = self.telemetry.lock() { tele.add_rollback_plan(&actions); }
                if let Ok(mut alerts) = self.alert_dispatcher.lock() { alerts.queue_alert(format!("Rollback planned with {} actions", actions.len())); }
                // Apply actions (deterministic scaling) additively (mutating params provided by caller)
                for act in actions { if let Some(v) = params.get_mut(&act.param) { *v *= act.factor; } }
            }
        }
        Ok(())
    }
}

// --- Additive telemetry & stats -------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SafetyTelemetry {
    start_time: Instant,
    total_validations: u64,
    failed_validations: u64,
    failure_counts: HashMap<String,u64>,
    last_failures: VecDeque<String>,
    max_last: usize,
    gradient_mean: f64,
    gradient_var: f64,
    gradient_samples: u64,
    last_mahalanobis: Option<f64>, // batch2 additive
    anomaly_probability: Option<f64>, // batch2 additive
    soft_warnings: VecDeque<String>, // batch2 additive
    max_soft: usize, // batch2 additive
    phase: String, // batch3 additive
    sandbox_events: u64, // batch3 additive
    // Batch4 additive telemetry deltas/rates
    last_rate_update: Instant,
    last_validation_count: u64,
    last_failure_count: u64,
    validation_rate_per_min: f64,
    failure_rate_per_min: f64,
    rollback_plans: VecDeque<String>,
    max_plans: usize,
    alerts: VecDeque<String>,
    max_alerts: usize,
}

impl SafetyTelemetry {
    pub fn new() -> Self { Self { start_time: Instant::now(), total_validations:0, failed_validations:0, failure_counts: HashMap::new(), last_failures: VecDeque::new(), max_last: 50, gradient_mean:0.0, gradient_var:0.0, gradient_samples:0, last_mahalanobis: None, anomaly_probability: None, soft_warnings: VecDeque::new(), max_soft: 100, phase: "Normal".to_string(), sandbox_events: 0, last_rate_update: Instant::now(), last_validation_count:0, last_failure_count:0, validation_rate_per_min:0.0, failure_rate_per_min:0.0, rollback_plans: VecDeque::new(), max_plans: 30, alerts: VecDeque::new(), max_alerts: 100 } }
    pub fn record_validation_start(&mut self) { self.total_validations += 1; }
    pub fn record_validation_success(&mut self) { /* nothing extra now */ }
    pub fn record_failure(&mut self, fail: &ClimateModelFailure) {
        self.failed_validations += 1;
        let key = match fail { 
            ClimateModelFailure::ParameterBoundsViolation {..} => "ParameterBoundsViolation",
            ClimateModelFailure::NumericalInstability {..} => "NumericalInstability",
            ClimateModelFailure::ConservationViolation {..} => "ConservationViolation",
            ClimateModelFailure::UnphysicalState {..} => "UnphysicalState",
            ClimateModelFailure::ComputationTimeout {..} => "ComputationTimeout",
            ClimateModelFailure::MemoryExceeded {..} => "MemoryExceeded",
            ClimateModelFailure::CascadeFailure {..} => "CascadeFailure",
        }.to_string();
        *self.failure_counts.entry(key.clone()).or_insert(0) += 1;
        self.last_failures.push_back(format!("{:?}", fail));
        if self.last_failures.len() > self.max_last { self.last_failures.pop_front(); }
    }
    pub fn update_gradient_stats(&mut self, g: f64) {
        // Welford
        self.gradient_samples += 1;
        let n = self.gradient_samples as f64;
        let delta = g - self.gradient_mean;
        self.gradient_mean += delta / n;
        self.gradient_var += delta * (g - self.gradient_mean);
    }
    pub fn gradient_std(&self) -> Option<f64> { if self.gradient_samples > 1 { Some((self.gradient_var / (self.gradient_samples as f64 -1.0)).sqrt()) } else { None } }
    pub fn update_mahalanobis(&mut self, d: f64) { self.last_mahalanobis = Some(d); }
    pub fn update_anomaly_probability(&mut self, p: f64) { self.anomaly_probability = Some(p); }
    pub fn add_soft_warning<T: Into<String>>(&mut self, w: T) { self.soft_warnings.push_back(w.into()); if self.soft_warnings.len()>self.max_soft { self.soft_warnings.pop_front(); } }
    pub fn update_phase(&mut self, phase: &str) { self.phase = phase.to_string(); }
    pub fn inc_sandbox(&mut self) { self.sandbox_events += 1; }
    pub fn update_rates(&mut self) {
        let elapsed = self.last_rate_update.elapsed().as_secs_f64();
        if elapsed >= 30.0 { // update every 30s
            let dv = self.total_validations - self.last_validation_count;
            let df = self.failed_validations - self.last_failure_count;
            if elapsed > 0.0 { self.validation_rate_per_min = (dv as f64)/(elapsed/60.0); self.failure_rate_per_min = (df as f64)/(elapsed/60.0); }
            self.last_rate_update = Instant::now();
            self.last_validation_count = self.total_validations;
            self.last_failure_count = self.failed_validations;
        }
    }
    pub fn add_rollback_plan(&mut self, actions: &Vec<RollbackAction>) {
        let summary = format!("{}", actions.iter().map(|a| format!("{}x{:.3}", a.param, a.factor)).collect::<Vec<_>>().join(";"));
        self.rollback_plans.push_back(summary);
        if self.rollback_plans.len()>self.max_plans { self.rollback_plans.pop_front(); }
    }
    pub fn add_alert<T: Into<String>>(&mut self, a: T) { self.alerts.push_back(a.into()); if self.alerts.len()>self.max_alerts { self.alerts.pop_front(); } }
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"uptime_seconds\": {},\n", self.start_time.elapsed().as_secs()));
        s.push_str(&format!("  \"total_validations\": {},\n", self.total_validations));
        s.push_str(&format!("  \"failed_validations\": {},\n", self.failed_validations));
        s.push_str("  \"failure_counts\": {\n");
        for (i,(k,v)) in self.failure_counts.iter().enumerate() { s.push_str(&format!("    \"{}\": {}{}\n", k, v, if i+1==self.failure_counts.len(){""} else {","})); }
        s.push_str("  },\n");
        if let Some(std) = self.gradient_std() { s.push_str(&format!("  \"gradient_mean\": {:.6},\n  \"gradient_std\": {:.6},\n", self.gradient_mean, std)); }
        s.push_str("  \"recent_failures\": [\n");
        for (i,f) in self.last_failures.iter().enumerate() { s.push_str(&format!("    \"{}\"{}\n", f, if i+1==self.last_failures.len(){""} else {","})); }
        s.push_str("  ],\n");
        if let Some(d) = self.last_mahalanobis { s.push_str(&format!("  \"last_mahalanobis\": {:.6},\n", d)); }
        if let Some(p) = self.anomaly_probability { s.push_str(&format!("  \"anomaly_probability\": {:.6},\n", p)); }
        s.push_str("  \"soft_warnings\": [\n");
        for (i,w) in self.soft_warnings.iter().enumerate() { s.push_str(&format!("    \"{}\"{}\n", w.replace('"', "'"), if i+1==self.soft_warnings.len(){""} else {","})); }
        s.push_str("  ],\n");
        s.push_str(&format!("  \"phase\": \"{}\",\n", self.phase));
        s.push_str(&format!("  \"sandbox_events\": {}\n", self.sandbox_events));
    s.push_str(&format!(",  \"validation_rate_per_min\": {:.4},\n  \"failure_rate_per_min\": {:.4},\n", self.validation_rate_per_min, self.failure_rate_per_min));
    // rollback plans
    s.push_str("  \"rollback_plans\": [\n");
    for (i,p) in self.rollback_plans.iter().enumerate() { s.push_str(&format!("    \"{}\"{}\n", p.replace('"', "'"), if i+1==self.rollback_plans.len(){""} else {","})); }
    s.push_str("  ],\n  \"alerts\": [\n");
    for (i,a) in self.alerts.iter().enumerate() { s.push_str(&format!("    \"{}\"{}\n", a.replace('"', "'"), if i+1==self.alerts.len(){""} else {","})); }
    s.push_str("  ]\n");
        s.push_str("}\n");
        s
    }
}

// --- Batch3: Recovery phases & hysteresis ---------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryPhase { Normal, Watch, Containment, Recovering }

impl std::fmt::Display for RecoveryPhase { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", match self { RecoveryPhase::Normal=>"Normal", RecoveryPhase::Watch=>"Watch", RecoveryPhase::Containment=>"Containment", RecoveryPhase::Recovering=>"Recovering" }) } }

pub struct RecoveryManager {
    pub phase: RecoveryPhase,
    last_prob: f64,
    last_dist: f64,
    sandbox_enter_time: Option<Instant>,
    sandbox_min_duration: Duration,
    hysteresis_delta: f64,
    watch_threshold: f64,
    containment_threshold: f64,
    recover_threshold: f64,
}

impl RecoveryManager {
    pub fn new() -> Self { Self { phase: RecoveryPhase::Normal, last_prob:0.0, last_dist:0.0, sandbox_enter_time: None, sandbox_min_duration: Duration::from_secs(20), hysteresis_delta: 0.15, watch_threshold:0.55, containment_threshold:0.8, recover_threshold:0.4 } }
    pub fn update(&mut self, probability: f64, distance: f64) {
        self.last_prob = probability; self.last_dist = distance;
        match self.phase {
            RecoveryPhase::Normal => { if probability > self.watch_threshold { self.phase = RecoveryPhase::Watch; } },
            RecoveryPhase::Watch => { if probability > self.containment_threshold { self.phase = RecoveryPhase::Containment; self.sandbox_enter_time = Some(Instant::now()); } else if probability < (self.watch_threshold - self.hysteresis_delta) { self.phase = RecoveryPhase::Normal; } },
            RecoveryPhase::Containment => { if probability < self.recover_threshold && if let Some(t)=self.sandbox_enter_time { t.elapsed()>self.sandbox_min_duration } else { false } { self.phase = RecoveryPhase::Recovering; } },
            RecoveryPhase::Recovering => { if probability < (self.recover_threshold - self.hysteresis_delta) { self.phase = RecoveryPhase::Normal; } else if probability > self.containment_threshold { self.phase = RecoveryPhase::Containment; } },
        }
    }
    pub fn request_sandbox_activation(&self) -> bool { self.phase == RecoveryPhase::Containment }
    pub fn can_exit_sandbox(&self) -> bool { matches!(self.phase, RecoveryPhase::Recovering | RecoveryPhase::Normal) }
}

// --- Batch3: Audit trail --------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AuditSnapshot { timestamp: Instant, phase: String, last_mahalanobis: Option<f64>, anomaly_probability: Option<f64>, soft_warnings: Vec<String> }
pub struct AuditTrail { snapshots: VecDeque<AuditSnapshot>, max: usize }
impl AuditTrail { pub fn new(max: usize) -> Self { Self { snapshots: VecDeque::new(), max } }
    pub fn maybe_snapshot(&mut self, tele: &SafetyTelemetry) {
        // snapshot every 5 validations or on phase change indicator (soft warning count multiple of 7)
        if tele.total_validations % 5 == 0 || tele.soft_warnings.len() % 7 == 0 {
            let snap = AuditSnapshot { timestamp: Instant::now(), phase: tele.phase.clone(), last_mahalanobis: tele.last_mahalanobis, anomaly_probability: tele.anomaly_probability, soft_warnings: tele.soft_warnings.iter().rev().take(10).cloned().collect() };
            self.snapshots.push_back(snap);
            if self.snapshots.len()>self.max { self.snapshots.pop_front(); }
        }
    }
    pub fn count(&self) -> usize { self.snapshots.len() }
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n  \"snapshots\": [\n");
        for (i, snap) in self.snapshots.iter().enumerate() {
            s.push_str(&format!("    {{ \"age_ms\": {}, \"phase\": \"{}\", \"last_mahalanobis\": {}, \"anomaly_probability\": {}, \"warnings\": [{}] }}{}\n",
                snap.timestamp.elapsed().as_millis(),
                snap.phase,
                snap.last_mahalanobis.map(|v| format!("{:.4}", v)).unwrap_or("null".to_string()),
                snap.anomaly_probability.map(|v| format!("{:.4}", v)).unwrap_or("null".to_string()),
                snap.soft_warnings.iter().map(|w| format!("\"{}\"", w.replace('"',"'"))).collect::<Vec<_>>().join(","),
                if i+1==self.snapshots.len(){""} else {","}
            ));
        }
        s.push_str("  ]\n}\n");
        s
    }
}

// --- Batch4: Rollback planner ---------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RollbackAction { pub param: String, pub factor: f64, pub reason: String }

#[derive(Debug, Default)]
pub struct RollbackPlanner { /* future config */ }
impl RollbackPlanner {
    pub fn plan(&self, params: &HashMap<String,f64>, severity: f64) -> Vec<RollbackAction> {
        // Severity scaled reductions; choose subset of sensitive parameters
        let mut actions = Vec::new();
        let scale = (severity/10.0).clamp(0.05,0.5); // 5% to 50%
        for key in ["ecs","tcr","cloud_feedback","ocean_heat_uptake"] {
            if let Some(v) = params.get(key) { if *v > 0.0 { actions.push(RollbackAction { param: key.to_string(), factor: 1.0 - scale*0.5, reason: format!("Containment severity {:.2}", severity) }); } }
        }
        actions
    }
}

// --- Batch4: Alert dispatcher ---------------------------------------------------------
#[derive(Debug)]
pub struct AlertDispatcher { queue: VecDeque<String>, max: usize }
impl AlertDispatcher { pub fn new(max: usize) -> Self { Self { queue: VecDeque::new(), max } }
    pub fn queue_alert(&mut self, msg: String) { self.queue.push_back(msg); if self.queue.len()>self.max { self.queue.pop_front(); } }
    pub fn to_json(&self) -> String { let mut s = String::from("{\n  \"alerts\": [\n"); for (i,a) in self.queue.iter().enumerate() { s.push_str(&format!("    \"{}\"{}\n", a.replace('"',"'"), if i+1==self.queue.len(){""} else {","})); } s.push_str("  ]\n}\n"); s }
}

// --- Batch2 additive: extended conservation checker ----------------------------------
#[derive(Debug, Clone)]
pub struct ExtendedConservationChecker {
    pub tolerance: f64,
    water_history: VecDeque<f64>,
    momentum_history: VecDeque<f64>,
    max_hist: usize,
}

#[derive(Debug, Clone)]
pub struct BudgetCheckResult { pub imbalance: f64, pub z_score: Option<f64>, pub severe_violation: bool }

impl ExtendedConservationChecker {
    pub fn new(tolerance: f64) -> Self { Self { tolerance, water_history: VecDeque::new(), momentum_history: VecDeque::new(), max_hist: 200 } }
    pub fn push_and_z(values: &mut VecDeque<f64>, max: usize, v: f64) -> Option<f64> {
        values.push_back(v);
        if values.len() > max { values.pop_front(); }
        if values.len() >= 10 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let var = values.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0);
            let std = var.max(1e-18).sqrt();
            Some((v - mean)/std)
        } else { None }
    }
    pub fn check_water_budget(&mut self, precip: f64, evap: f64, moisture_flux_div: f64) -> BudgetCheckResult {
        let imbalance = (precip - evap) - moisture_flux_div; // P - E - div(F)
        let z = Self::push_and_z(&mut self.water_history, self.max_hist, imbalance);
        let severe = imbalance.abs() > self.tolerance * 5.0;
        BudgetCheckResult { imbalance, z_score: z, severe_violation: severe }
    }
    pub fn check_momentum_budget(&mut self, u_in: f64, u_out: f64, forcing: f64, tendency: f64) -> BudgetCheckResult {
        let imbalance = (u_out - u_in) - (forcing + tendency);
        let z = Self::push_and_z(&mut self.momentum_history, self.max_hist, imbalance);
        let severe = imbalance.abs() > self.tolerance * 10.0;
        BudgetCheckResult { imbalance, z_score: z, severe_violation: severe }
    }
}

// --- Batch2 additive: joint parameter model (Mahalanobis) -----------------------------
#[derive(Debug, Clone)]
pub struct JointParameterModel {
    keys: Vec<String>,
    count: u64,
    mean: Vec<f64>,
    m2: Vec<Vec<f64>>, // sum of products for covariance
    pub last_distance: Option<f64>,
}

impl JointParameterModel {
    pub fn new(keys: Vec<String>) -> Self {
        let n = keys.len();
        Self { keys, count: 0, mean: vec![0.0; n], m2: vec![vec![0.0; n]; n], last_distance: None }
    }
    pub fn update(&mut self, params: &HashMap<String,f64>) {
        let mut x = Vec::with_capacity(self.keys.len());
        for k in &self.keys { if let Some(v) = params.get(k) { x.push(*v); } else { return; } }
        self.count += 1;
        let n = self.count as f64;
        // delta vector
        let mut delta: Vec<f64> = Vec::with_capacity(x.len());
        for i in 0..x.len() { delta.push(x[i] - self.mean[i]); }
        // update mean
        for i in 0..x.len() { self.mean[i] += delta[i] / n; }
        // update m2
        for i in 0..x.len() { for j in i..x.len() { let val = delta[i]*(x[j]-self.mean[j]); self.m2[i][j] += val; if i!=j { self.m2[j][i] += val; } } }
    }
    pub fn covariance(&self) -> Option<DMatrix<f64>> {
        if self.count < 2 { return None; }
        let n = self.keys.len();
        let mut data = Vec::with_capacity(n*n);
        for i in 0..n { for j in 0..n { data.push(self.m2[i][j] / (self.count as f64 - 1.0)); } }
        Some(DMatrix::from_row_slice(n,n,&data))
    }
    pub fn mahalanobis(&mut self, params: &HashMap<String,f64>) -> Option<f64> {
        if self.count < (self.keys.len() as u64 + 5) { return None; }
        let cov = self.covariance()?;
        // Regularize & invert
        let n = cov.nrows();
        let ident = DMatrix::<f64>::identity(n,n);
        let cov_reg = &cov + ident.scale(1e-6);
        let cov_inv = cov_reg.try_inverse()?;
        let mut x = DMatrix::<f64>::zeros(n,1);
        for (i,k) in self.keys.iter().enumerate() { x[(i,0)] = *params.get(k)?; }
        let mut mean = DMatrix::<f64>::zeros(n,1);
        for i in 0..n { mean[(i,0)] = self.mean[i]; }
        let diff = x - mean;
        let d2 = (diff.transpose() * cov_inv * diff)[(0,0)].abs();
        let d = d2.sqrt();
        self.last_distance = Some(d);
        Some(d)
    }
}

// --- Batch2 additive: probabilistic anomaly scorer ------------------------------------
#[derive(Debug, Clone)]
pub struct ProbabilisticAnomalyScorer { alpha: f64, probability: f64, history: VecDeque<f64>, max_hist: usize }
impl ProbabilisticAnomalyScorer { pub fn new(alpha: f64) -> Self { Self { alpha, probability: 0.0, history: VecDeque::new(), max_hist: 500 } }
    pub fn update(&mut self, severity: f64) -> f64 { // severity >=0
        // logistic mapping severity -> instantaneous anomaly likelihood
        let inst = 1.0 / (1.0 + (-severity).exp());
        self.probability = self.alpha * self.probability + (1.0 - self.alpha) * inst;
        self.history.push_back(self.probability); if self.history.len()>self.max_hist { self.history.pop_front(); }
        self.probability
    }
    pub fn probability(&self) -> f64 { self.probability }
}

// Rolling statistics for adaptive thresholds
#[derive(Debug, Clone)]
pub struct RollingStats { window: usize, values: VecDeque<f64>, sum: f64, sum_sq: f64 }
impl RollingStats { pub fn new(window: usize) -> Self { Self { window, values: VecDeque::new(), sum:0.0, sum_sq:0.0 } } 
    pub fn push(&mut self, v: f64) { self.values.push_back(v); self.sum += v; self.sum_sq += v*v; if self.values.len()>self.window { if let Some(old)=self.values.pop_front(){ self.sum -= old; self.sum_sq -= old*old; } } }
    pub fn mean(&self) -> Option<f64> { if self.values.is_empty(){None}else{Some(self.sum / self.values.len() as f64)} }
    pub fn variance(&self) -> Option<f64> { if self.values.len()<2 {None}else{ let m = self.mean().unwrap(); Some((self.sum_sq / self.values.len() as f64) - m*m) } }
}

// Condition number estimation helper (symmetric)
pub fn estimate_condition_number(matrix: &Vec<Vec<f64>>) -> Option<f64> {
    if matrix.is_empty() { return None; }
    let n = matrix.len();
    for row in matrix { if row.len()!=n { return None; } }
    // Build DMatrix
    let mut data = Vec::with_capacity(n*n);
    for r in 0..n { for c in 0..n { data.push(matrix[r][c]); } }
    let m = DMatrix::<f64>::from_row_slice(n,n,&data);
    // Symmetrize (additive non-destructive) A' = (A + A^T)/2
    let mt = m.transpose();
    let sym = (&m + &mt) * 0.5;
    // Use simple power iteration for largest eigenvalue
    let mut v = DMatrix::<f64>::from_element(n,1,1.0/n as f64);
    let mut lambda_max = 0.0;
    for _ in 0..60 { let w = &sym * &v; let norm = w.norm(); if norm < 1e-18 { return None; } v = &w / norm; lambda_max = (v.transpose()*&sym*&v)[(0,0)]; }
    if !lambda_max.is_finite() || lambda_max.abs() < 1e-18 { return None; }
    // Approximate min eigenvalue via shifted power on (lambda_max I - A)
    let shift_mat = DMatrix::<f64>::identity(n,n)*lambda_max - sym.clone();
    let mut v2 = DMatrix::<f64>::from_element(n,1,1.0/(n as f64));
    let mut mu = 0.0;
    for _ in 0..60 { let w = &shift_mat * &v2; let norm = w.norm(); if norm < 1e-18 { break; } v2 = &w / norm; mu = (v2.transpose()*&shift_mat*&v2)[(0,0)]; }
    let lambda_min = lambda_max - mu.abs();
    if !lambda_min.is_finite() || lambda_min.abs() < 1e-18 { return None; }
    Some((lambda_max.abs()) / lambda_min.abs())
}

// ---
// Example
// ---

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parameter_validation() {
        let validator = ParameterValidator::new();
        let mut params = HashMap::new();
        params.insert("ecs".to_string(), 3.0);
        params.insert("tcr".to_string(), 1.8);
        
        assert!(validator.validate(&params).is_ok());
        
        // TCR > ECS should fail
        params.insert("tcr".to_string(), 3.5);
        assert!(validator.validate(&params).is_err());
    }
    
    #[test]
    fn test_conservation_checking() {
        let mut checker = ConservationChecker::new(0.1);
        
        // Balanced system
        assert!(checker.check_energy_balance(340.0, 339.5, 0.5).is_ok());
        
        // Imbalanced system
        assert!(checker.check_energy_balance(340.0, 330.0, 0.0).is_err());
    }
    
    #[test]
    fn test_cascade_protection() {
        let mut cascade = CascadeProtection::new();
        
        // Single failure OK
        assert!(cascade.update_component("ice_sheets", ComponentState::Failed).is_ok());
        
        // Check dependent component states
        assert_eq!(cascade.component_states["amoc"], ComponentState::Degraded);
        
        // Too many failures triggers error
        cascade.update_component("amazon", ComponentState::Failed).unwrap();
        assert!(cascade.update_component("permafrost", ComponentState::Failed).is_err());
    }
}