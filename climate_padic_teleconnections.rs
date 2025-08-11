// Climate teleconnection analysis with p-adic distance metrics
// Experimental hierarchical relationship mapping
// Atmospheric and oceanic bridge detection

// ─────────────────────────────────────────────────────────────────────────────
// DATA SOURCE REQUIREMENTS
// ─────────────────────────────────────────────────────────────────────────────
//
// 1. HIGH-RESOLUTION ATMOSPHERIC REANALYSIS FOR TELECONNECTIONS:
//    - Source: ERA5 from ECMWF Copernicus Climate Data Store
//    - Resolution: 0.25° x 0.25° x 137 model levels + 37 pressure levels
//    - Temporal: Hourly from 1940-01-01 to present (5-day lag)
//    - Format: GRIB2 native, NetCDF4 converted
//    - Size: ~5TB/year for full 3D fields, ~500GB/year for key levels
//    - Variables REQUIRED for teleconnections:
//      * Geopotential height (Z) at 500, 300, 200 hPa
//      * Temperature (T) at all levels for baroclinicity
//      * U, V wind components for jet stream analysis
//      * Specific humidity (Q) for moisture transport
//      * Vorticity and divergence (VO, D) for Rossby waves
//      * Sea level pressure (SLP) for surface patterns
//    - API: cdsapi Python client with CDS account
//    - Preprocessing: Anomaly calculation, seasonal detrending, EOF analysis
//    - Missing: Consistent quality before 1958 (pre-IGY)
//    - Missing: Upper stratosphere reliability before 1979
//
// 2. COMPREHENSIVE CLIMATE INDEX DATABASE:
//    - Source: NOAA PSL + CPC + JMA + BOM + Met Office
//    - Indices with EXACT definitions:
//      * Niño3.4: SST anomaly 5°N-5°S, 170°W-120°W (ERSSTv5)
//      * NAO: SLP Lisbon - SLP Reykjavik normalized (station-based)
//      * PDO: Leading PC of North Pacific SST 20°N-70°N (detrended)
//      * AMO: Atlantic SST 0°-60°N minus global mean (unfiltered)
//      * IOD/DMI: Western box (50°-70°E, 10°S-10°N) - Eastern (90°-110°E)
//      * SAM/AAO: Zonal mean SLP difference 40°S - 65°S
//      * QBO: Zonal wind at 30 hPa equator (Singapore radiosonde)
//      * MJO: RMM1/RMM2 from Wheeler-Hendon (2004) EOF method
//    - Temporal: Monthly 1850-present (varies by index)
//    - Format: ASCII columns, CSV, some NetCDF4
//    - Size: <100MB for all indices combined
//    - API: https://psl.noaa.gov/data/climateindices/list/
//    - Preprocessing: 3-month running mean for some, standardization
//    - Missing: Uncertainty estimates for indices
//    - Missing: Consistent pre-1950 calculations
//
// 3. TELECONNECTION SPATIAL PATTERNS AND LOADINGS:
//    - Source: CPC rotated PCA patterns + EOF analysis
//    - Patterns with correlation maps:
//      * PNA: Pacific-North American (4 centers)
//      * EA: East Atlantic
//      * WP: West Pacific
//      * EP/NP: East Pacific/North Pacific
//      * Scandinavia (SCAND)
//      * Polar/Eurasia
//      * Tropical/Northern Hemisphere (TNH)
//    - Resolution: 2.5° x 2.5° global grids
//    - Format: NetCDF4 with correlation coefficients
//    - Size: ~100MB for all patterns
//    - API: https://www.cpc.ncep.noaa.gov/data/teledoc/
//    - Method: Rotated PCA on 500 hPa height anomalies
//    - Missing: Time-varying EOFs (currently fixed 1981-2010 base)
//    - Missing: Vertical structure of patterns
//
// 4. WAVE ACTIVITY FLUX FOR TELECONNECTION PATHWAYS:
//    - Source: Computed from ERA5 6-hourly data
//    - Method: Takaya & Nakamura (2001) for stationary waves
//    - Variables needed:
//      * Geopotential height perturbations
//      * Horizontal wind (U, V) at 300 hPa
//      * Temperature for baroclinic conversion
//    - Resolution: 2.5° x 2.5° (smoothed from 0.25°)
//    - Format: Computed fields in NetCDF4
//    - Size: ~10GB/year for daily fluxes
//    - Preprocessing: 10-day low-pass filter, remove zonal mean
//    - Missing: Real-time operational wave flux product
//    - Missing: Transient eddy flux separation
//
// 5. CROSS-SPECTRAL COHERENCE FOR TELECONNECTIONS:
//    - Source: Paired station/grid point time series
//    - Station pairs: 1000+ key teleconnection pairs
//    - Method: Multitaper spectral analysis (5 tapers)
//    - Frequency range: 10 days to 100 years
//    - Format: Coherence + phase as function of frequency
//    - Size: ~1GB for global coherence maps
//    - Significance: Monte Carlo with 1000 AR(1) surrogates
//    - Missing: Nonlinear coherence metrics
//    - Missing: Time-varying coherence
//
// 6. P-ADIC VALIDATION DATA (CRITICAL FOR THIS MODULE):
//    - Source: Constructed test cases with known hierarchies
//    - Validation approach:
//      * 1000 pairs of climate events with known teleconnection strength
//      * Compute p-adic distance for various primes p
//      * Compute Euclidean distance in state space
//      * Correlate both with observed teleconnection amplitude
//    - Format: HDF5 with distance matrices
//    - Size: ~10GB for validation suite
//    - Ground truth: Lag correlation from observations
//    - Success metric: p-adic must beat Euclidean by >10% correlation
//    - Missing: NO VALIDATION EXISTS - P-ADIC APPROACH UNPROVEN
//    - CRITICAL: If validation fails, DELETE ENTIRE P-ADIC FRAMEWORK
//
// 7. EXTREME EVENT TELECONNECTION CATALOG:
//    - Source: IBTrACS, EM-DAT, GHCN-Daily extremes
//    - Events: Heat waves, cold spells, floods, droughts
//    - Teleconnection attribution: Lag correlation analysis
//    - Format: Event database with teleconnection loadings
//    - Size: ~1GB
//    - Missing: Mechanistic attribution beyond correlation
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::f64::consts::{PI, E};
use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array3, Array4, ArrayView2, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use dashmap::DashMap;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::{dijkstra, connected_components};
use statistical::{mean, standard_deviation};

// Physical constants - mix of measured values and approximations
const EARTH_RADIUS_M: f64 = 6_371_008.8;  // Mean Earth radius (SOURCE: IUGG 2015 standard)
// Latitude-dependent functions for realistic climate parameters
fn rossby_radius_km(lat_deg: f64) -> f64 {
    // Rossby radius varies with Coriolis parameter: Ld = sqrt(gH)/f
    let lat_rad = lat_deg.to_radians();
    let coriolis = 2.0 * 7.2921e-5 * lat_rad.sin().abs(); // Earth rotation rate
    if coriolis > 1e-10 {
        (9.81 * 1000.0).sqrt() / coriolis / 1000.0 // Convert to km
    } else {
        5000.0 // Large value near equator
    }
}

fn kelvin_wave_speed_ms(lat_deg: f64) -> f64 {
    // Kelvin wave speed: c = sqrt(gH) where H varies with thermocline depth
    let thermocline_depth = 200.0 + 100.0 * (lat_deg.abs() / 90.0); // Deeper at poles
    (9.81 * thermocline_depth).sqrt()
}

fn walker_cell_width_km(enso_state: f64) -> f64 {
    // Walker circulation width varies with ENSO: wider during La Niña
    12000.0 + 3000.0 * enso_state // ENSO index: positive = El Niño, negative = La Niña
}

fn hadley_cell_width_deg(season: f64) -> f64 {
    // Hadley cell width varies seasonally between ~23-35 degrees
    27.0 + 8.0 * (season * 2.0 * std::f64::consts::PI).sin()
}

fn jet_stream_core_ms(lat_deg: f64, season: f64) -> f64 {
    // Jet stream strength varies with latitude and season
    let base_speed = if lat_deg.abs() > 25.0 && lat_deg.abs() < 65.0 {
        50.0 + 30.0 * ((lat_deg.abs() - 45.0) / 20.0).cos()
    } else {
        20.0
    };
    base_speed + 15.0 * (season * 2.0 * std::f64::consts::PI).cos() // Winter stronger
}

fn ferrel_cell_lat(season: f64) -> f64 {
    // Ferrel cell latitude varies seasonally
    58.0 + 5.0 * (season * 2.0 * std::f64::consts::PI).sin() // ~53-63°
}

// P-adic climate state representation

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadicClimateState {
    // Location coordinates
    pub lat: f64,
    pub lon: f64,
    pub pressure_level_hpa: f64,
    
    // P-adic representation (base p expansion)
    pub prime: u32,
    pub digits: Vec<u32>,  // p-adic digits [a₀, a₁, a₂, ...]
    pub precision: usize,  // Number of p-adic digits
    
    // Climate observables
    pub temperature_k: f64,
    pub geopotential_m: f64,
    pub u_wind_ms: f64,
    pub v_wind_ms: f64,
    pub specific_humidity: f64,
    pub vorticity: f64,
    pub divergence: f64,
    
    // Teleconnection indices
    pub enso_index: f64,      // El Niño-Southern Oscillation
    pub nao_index: f64,       // North Atlantic Oscillation
    pub pdo_index: f64,       // Pacific Decadal Oscillation
    pub amo_index: f64,       // Atlantic Multidecadal Oscillation
    pub iod_index: f64,       // Indian Ocean Dipole
    pub sam_index: f64,       // Southern Annular Mode
    pub mjo_phase: u8,        // Madden-Julian Oscillation phase (1-8)
    pub qbo_phase: f64,       // Quasi-Biennial Oscillation
}

impl PadicClimateState {
    /// Create from atmospheric observations
    pub fn from_observations(
        lat: f64, 
        lon: f64,
        pressure_hpa: f64,
        obs: &HashMap<String, f64>
    ) -> Self {
        // Determine prime based on dominant oscillation
        let prime = Self::select_prime_for_region(lat, lon);
        
        // Encode state into p-adic representation
        let digits = Self::encode_climate_state(obs, prime);
        
        Self {
            lat,
            lon,
            pressure_level_hpa: pressure_hpa,
            prime,
            digits,
            precision: Self::determine_precision(&obs, lat, lon),
            temperature_k: obs.get("T").copied().unwrap_or(288.0), // 288K = 15°C global mean
            geopotential_m: obs.get("Z").copied().unwrap_or(0.0),
            u_wind_ms: obs.get("U").copied().unwrap_or(0.0),
            v_wind_ms: obs.get("V").copied().unwrap_or(0.0),
            specific_humidity: obs.get("Q").copied().unwrap_or(0.01), // 0.01 kg/kg ~ 60% RH at surface
            vorticity: obs.get("VO").copied().unwrap_or(0.0),
            divergence: obs.get("D").copied().unwrap_or(0.0),
            enso_index: obs.get("ENSO").copied().unwrap_or(0.0),
            nao_index: obs.get("NAO").copied().unwrap_or(0.0),
            pdo_index: obs.get("PDO").copied().unwrap_or(0.0),
            amo_index: obs.get("AMO").copied().unwrap_or(0.0),
            iod_index: obs.get("IOD").copied().unwrap_or(0.0),
            sam_index: obs.get("SAM").copied().unwrap_or(0.0),
            mjo_phase: obs.get("MJO").copied().unwrap_or(0.0) as u8,
            qbo_phase: obs.get("QBO").copied().unwrap_or(0.0),
        }
    }
    
    fn select_prime_for_region(lat: f64, lon: f64) -> u32 {
        match (lat, lon) {
            (lat, lon) if lat.abs() < 10.0 && lon > 120.0 && lon < 280.0 => 2,
            (lat, lon) if lat > 40.0 && (lon > 280.0 || lon < 30.0) => 3,
            (lat, lon) if lat.abs() < 30.0 && lon > 40.0 && lon < 120.0 => 5,
            (lat, _) if lat < -40.0 => 7,
            (lat, _) if lat > 70.0 => 11,
            _ => 13,
        }
    }
    
    fn determine_precision(obs: &HashMap<String, f64>, lat: f64, lon: f64) -> usize {
        // Data-driven precision based on observation quality and regional characteristics
        let mut precision = 5; // Base precision
        
        // Increase precision for high-quality data regions
        if obs.len() > 15 { // Rich observation set
            precision += 2;
        }
        
        // Increase precision in teleconnection-sensitive regions
        if lat.abs() < 30.0 { // Tropical regions (ENSO, MJO)
            precision += 1;
        }
        if lat.abs() > 60.0 { // Polar regions (NAO, AO)
            precision += 1;
        }
        
        // Reduce precision for sparse data regions
        if obs.len() < 8 {
            precision = (precision - 1).max(3);
        }
        
        // Adaptive precision based on data variability
        let temp_var = obs.get("T_anom").map(|t| t.abs()).unwrap_or(0.0);
        if temp_var > 3.0 { // High variability requires more precision
            precision += 1;
        }
        
        precision.min(15).max(3)
    }
    
    fn encode_climate_state(obs: &HashMap<String, f64>, prime: u32) -> Vec<u32> {
        let mut digits = Vec::with_capacity(10);
        
        let t_anom = obs.get("T_anom").copied().unwrap_or(0.0);
        let t_class = ((t_anom + 5.0).max(0.0).min(10.0) * (prime - 1) as f64 / 10.0) as u32;
        digits.push(t_class % prime);
        
        let z_anom = obs.get("Z_anom").copied().unwrap_or(0.0);
        let z_class = ((z_anom + 100.0).max(0.0).min(200.0) * (prime - 1) as f64 / 200.0) as u32;
        digits.push(z_class % prime);
        
        let shear = obs.get("shear").copied().unwrap_or(0.0);
        let shear_class = (shear.min(50.0) * (prime - 1) as f64 / 50.0) as u32;
        digits.push(shear_class % prime);
        
        let mfc = obs.get("MFC").copied().unwrap_or(0.0);
        let mfc_class = ((mfc + 10.0).max(0.0).min(20.0) * (prime - 1) as f64 / 20.0) as u32;
        digits.push(mfc_class % prime);
        
        let vadv = obs.get("VADV").copied().unwrap_or(0.0);
        let vadv_class = ((vadv + 5.0).max(0.0).min(10.0) * (prime - 1) as f64 / 10.0) as u32;
        digits.push(vadv_class % prime);
        
        for _ in digits.len()..10 {
            digits.push(0);
        }
        
        digits
    }
}

// ---
// P-ADIC DISTANCE METRICS
// ---

pub struct PadicTeleconnectionMetric {
    /// Cache of computed distances
    distance_cache: Arc<DashMap<(u64, u64), f64>>,
    
    /// Teleconnection graph
    graph: Arc<RwLock<Graph<PadicClimateState, f64>>>,
    
    /// Node index mapping
    location_to_node: Arc<DashMap<(i32, i32), NodeIndex>>,
    
    /// Waveguide paths (e.g., jet stream)
    waveguides: Arc<RwLock<Vec<WaveguidePath>>>,
}

#[derive(Debug, Clone)]
struct WaveguidePath {
    nodes: Vec<(f64, f64)>,  // (lat, lon) waypoints
    strength: f64,            // Waveguide strength (m/s)
    wavenumber: i32,          // Dominant wavenumber
    phase_speed: f64,         // Phase speed (m/s)
    group_velocity: f64,      // Group velocity (m/s)
}

impl PadicTeleconnectionMetric {
    pub fn new() -> Self {
        Self {
            distance_cache: Arc::new(DashMap::new()),
            graph: Arc::new(RwLock::new(Graph::new())),
            location_to_node: Arc::new(DashMap::new()),
            waveguides: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Compute p-adic distance between two climate states
    pub fn padic_distance(&self, state1: &PadicClimateState, state2: &PadicClimateState) -> f64 {
        // Check cache first
        let key = (self.state_hash(state1), self.state_hash(state2));
        if let Some(dist) = self.distance_cache.get(&key) {
            return *dist;
        }
        
        // Check same prime base
        if state1.prime != state2.prime {
            // Convert to common prime (LCM approach)
            return self.cross_prime_distance(state1, state2);
        }
        
        let p = state1.prime as f64;
        
        // Find first differing p-adic digit
        let mut valuation = 0;
        for i in 0..state1.digits.len().min(state2.digits.len()) {
            if state1.digits[i] != state2.digits[i] {
                valuation = i;
                break;
            }
        }
        
        // P-adic norm: |x|_p = p^(-v_p(x))
        let padic_norm = p.powi(-(valuation as i32));
        
        // Weight by physical distance for teleconnection strength
        let physical_dist = self.haversine_distance(
            state1.lat, state1.lon,
            state2.lat, state2.lon
        );
        
        // Atmospheric bridge decay with distance
        let decay_length = ROSSBY_RADIUS_KM * 1000.0;  // Convert to meters
        let decay_factor = (-physical_dist / decay_length).exp();
        
        // Combined metric
        let distance = padic_norm * (1.0 + physical_dist / decay_length) * decay_factor;
        
        // Cache result
        self.distance_cache.insert(key, distance);
        
        distance
    }
    
    /// Handle distance between states with different primes
    fn cross_prime_distance(&self, state1: &PadicClimateState, state2: &PadicClimateState) -> f64 {
        // Use product formula: ∏_p |x|_p = 1
        // Weight by relative importance of each prime's teleconnection
        
        let p1 = state1.prime as f64;
        let p2 = state2.prime as f64;
        
        // Convert to mixed radix representation
        let lcm = self.lcm(state1.prime, state2.prime);
        
        let norm1 = self.padic_self_norm(state1);
        let norm2 = self.padic_self_norm(state2);
        
        2.0 * norm1 * norm2 / (norm1 + norm2)
    }
    
    /// Self p-adic norm of a state
    fn padic_self_norm(&self, state: &PadicClimateState) -> f64 {
        let p = state.prime as f64;
        
        // Find highest non-zero digit
        let mut max_val = 0;
        for (i, &digit) in state.digits.iter().enumerate() {
            if digit != 0 {
                max_val = i;
            }
        }
        
        p.powi(-(max_val as i32))
    }
    
    /// Haversine distance between two points on Earth
    fn haversine_distance(&self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let dlat = (lat2 - lat1).to_radians();
        let dlon = (lon2 - lon1).to_radians();
        let lat1 = lat1.to_radians();
        let lat2 = lat2.to_radians();
        
        let a = (dlat / 2.0).sin().powi(2) + 
                lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        
        EARTH_RADIUS_M * c
    }
    
    /// Generate hash for state caching
    fn state_hash(&self, state: &PadicClimateState) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash location and prime
        ((state.lat * 100.0) as i32).hash(&mut hasher);
        ((state.lon * 100.0) as i32).hash(&mut hasher);
        state.prime.hash(&mut hasher);
        
        // Hash first few p-adic digits
        for &digit in state.digits.iter().take(5) {
            digit.hash(&mut hasher);
        }
        
        hasher.finish()
    }
    
    fn lcm(&self, a: u32, b: u32) -> u32 {
        a * b / self.gcd(a, b)
    }
    
    fn gcd(&self, mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
}

// ---
// TELECONNECTION PATTERN DETECTION
// ---

pub struct TeleconnectionAnalyzer {
    metric: Arc<PadicTeleconnectionMetric>,
    patterns: Arc<RwLock<HashMap<String, TeleconnectionPattern>>>,
    correlation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleconnectionPattern {
    pub name: String,
    pub source_region: BoundingBox,
    pub target_regions: Vec<BoundingBox>,
    pub correlation: f64,
    pub lag_days: f64,
    pub mechanism: TeleconnectionMechanism,
    pub strength_index: f64,
    pub waveguide_path: Option<Vec<(f64, f64)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TeleconnectionMechanism {
    RossbyWaveTrain,           // Stationary Rossby waves
    KelvinWave,                // Equatorial Kelvin waves
    WalkerCirculation,         // Zonal overturning
    HadleyCirculation,         // Meridional overturning
    JetStreamWaveguide,        // Trapped waves in jet
    StratosphericBridge,       // Stratosphere-troposphere coupling
    OceanicBridge,            // SST-mediated connection
    MixedMode,                // Multiple mechanisms
}

impl TeleconnectionAnalyzer {
    pub fn new(correlation_threshold: f64) -> Self {
        Self {
            metric: Arc::new(PadicTeleconnectionMetric::new()),
            patterns: Arc::new(RwLock::new(HashMap::new())),
            correlation_threshold,
        }
    }
    
    /// Detect teleconnection between Beijing and Miami (or any two regions)
    pub fn detect_teleconnection(
        &self,
        source: &PadicClimateState,
        target: &PadicClimateState,
        historical_data: &Array3<f64>,  // (time, lat, lon)
    ) -> Option<TeleconnectionPattern> {
        
        // Compute p-adic distance
        let padic_dist = self.metric.padic_distance(source, target);
        
        if padic_dist > 1.0 {
            return None;  // Too distant in p-adic metric
        }
        
        // Extract time series for both locations
        let source_series = self.extract_time_series(
            historical_data, 
            source.lat, 
            source.lon
        );
        let target_series = self.extract_time_series(
            historical_data,
            target.lat,
            target.lon
        );
        
        // Compute lagged correlations
        let (max_corr, arbitrary_lag) = self.lagged_correlation(
            &source_series,
            &target_series,
            60
        );
        
        if max_corr.abs() < self.correlation_threshold {
            return None;
        }
        
        // Identify mechanism based on characteristics
        let mechanism = self.identify_mechanism(
            source,
            target,
            arbitrary_lag,
            padic_dist
        );
        
        // Trace waveguide path if applicable
        let waveguide_path = self.trace_waveguide(source, target, &mechanism);
        
        Some(TeleconnectionPattern {
            name: format!("{:.0},{:.0}->{:.0},{:.0}", 
                         source.lat, source.lon, target.lat, target.lon),
            source_region: BoundingBox {
                min_lat: source.lat - 5.0,
                max_lat: source.lat + 5.0,
                min_lon: source.lon - 5.0,
                max_lon: source.lon + 5.0,
            },
            target_regions: vec![BoundingBox {
                min_lat: target.lat - 5.0,
                max_lat: target.lat + 5.0,
                min_lon: target.lon - 5.0,
                max_lon: target.lon + 5.0,
            }],
            correlation: max_corr,
            lag_days: arbitrary_lag as f64,
            mechanism,
            strength_index: (max_corr.abs() / padic_dist).sqrt(),
            waveguide_path,
        })
    }
    
    /// Extract time series for a location
    fn extract_time_series(
        &self,
        data: &Array3<f64>,
        lat: f64,
        lon: f64
    ) -> Vec<f64> {
        // Find nearest grid point
        let lat_idx = ((lat + 90.0) * data.shape()[1] as f64 / 180.0) as usize;
        let lon_idx = ((lon + 180.0) * data.shape()[2] as f64 / 360.0) as usize;
        
        let lat_idx = lat_idx.min(data.shape()[1] - 1);
        let lon_idx = lon_idx.min(data.shape()[2] - 1);
        
        // Extract time series
        data.slice(s![.., lat_idx, lon_idx])
            .iter()
            .copied()
            .collect()
    }
    
    /// Compute lagged correlation
    fn lagged_correlation(
        &self,
        x: &[f64],
        y: &[f64],
        max_lag: usize
    ) -> (f64, usize) {
        let mut max_corr = 0.0;
        let mut arbitrary_lag = 0;  // NOT optimal - just highest correlation
        
        for lag in 0..=max_lag {
            let n = x.len().min(y.len() - lag);
            if n < 30 {
                continue;  // Need minimum samples
            }
            
            // Compute correlation at this lag
            let x_slice = &x[..n];
            let y_slice = &y[lag..lag+n];
            
            let x_mean = x_slice.iter().sum::<f64>() / n as f64;
            let y_mean = y_slice.iter().sum::<f64>() / n as f64;
            
            let mut cov = 0.0;
            let mut x_var = 0.0;
            let mut y_var = 0.0;
            
            for i in 0..n {
                let dx = x_slice[i] - x_mean;
                let dy = y_slice[i] - y_mean;
                cov += dx * dy;
                x_var += dx * dx;
                y_var += dy * dy;
            }
            
            let corr = cov / (x_var * y_var).sqrt();
            
            if corr.abs() > max_corr.abs() {
                max_corr = corr;
                arbitrary_lag = lag;
            }
        }
        
        (max_corr, arbitrary_lag)
    }
    
    /// Identify teleconnection mechanism
    fn identify_mechanism(
        &self,
        source: &PadicClimateState,
        target: &PadicClimateState,
        lag_days: usize,
        padic_dist: f64
    ) -> TeleconnectionMechanism {
        
        // Rossby wave speed estimate
        // TODO: Beta-plane approximation breaks down near equator
        // SOURCE: Pedlosky 1987, β = 2Ωcos(φ)/R where Ω = 7.27×10⁻⁵ rad/s
        let beta = 2.0 * 7.27e-5 * source.lat.to_radians().cos() / EARTH_RADIUS_M;
        let rossby_speed = source.u_wind_ms - beta * ROSSBY_RADIUS_KM * ROSSBY_RADIUS_KM * 1000.0;
        
        // Expected lag for Rossby wave propagation
        let distance = self.metric.haversine_distance(
            source.lat, source.lon,
            target.lat, target.lon
        );
        let expected_rossby_lag = (distance / rossby_speed / 86400.0) as usize;
        
        if (lag_days as i32 - expected_rossby_lag as i32).abs() < 5 {
            // Consistent with Rossby wave propagation
            if source.lat.abs() > 30.0 && target.lat.abs() > 30.0 {
                return TeleconnectionMechanism::JetStreamWaveguide;
            } else {
                return TeleconnectionMechanism::RossbyWaveTrain;
            }
        }
        
        // Kelvin waves (equatorial)
        if source.lat.abs() < 15.0 && target.lat.abs() < 15.0 {
            let expected_kelvin_lag = (distance / KELVIN_WAVE_SPEED_MS / 86400.0) as usize;
            if (lag_days as i32 - expected_kelvin_lag as i32).abs() < 3 {
                return TeleconnectionMechanism::KelvinWave;
            }
        }
        
        // Walker circulation (zonal tropical)
        if source.lat.abs() < 20.0 && target.lat.abs() < 20.0 
           && (source.lon - target.lon).abs() > 90.0 {
            return TeleconnectionMechanism::WalkerCirculation;
        }
        
        // Hadley circulation (meridional tropical-extratropical)
        if (source.lat.abs() < 30.0) != (target.lat.abs() < 30.0) {
            return TeleconnectionMechanism::HadleyCirculation;
        }
        
        // Stratospheric bridge (high altitude, slow)
        if source.pressure_level_hpa < 100.0 && lag_days > 10 {
            return TeleconnectionMechanism::StratosphericBridge;
        }
        
        // Oceanic bridge (very slow)
        if lag_days > 30 {
            return TeleconnectionMechanism::OceanicBridge;
        }
        
        TeleconnectionMechanism::MixedMode
    }
    
    /// Trace waveguide path for teleconnection
    fn trace_waveguide(
        &self,
        source: &PadicClimateState,
        target: &PadicClimateState,
        mechanism: &TeleconnectionMechanism
    ) -> Option<Vec<(f64, f64)>> {
        
        match mechanism {
            TeleconnectionMechanism::JetStreamWaveguide => {
                // Follow jet stream path
                self.trace_jet_stream_path(source.lat, source.lon, target.lat, target.lon)
            },
            TeleconnectionMechanism::RossbyWaveTrain => {
                // Great circle with Rossby wave dispersion
                self.trace_rossby_path(source.lat, source.lon, target.lat, target.lon)
            },
            TeleconnectionMechanism::KelvinWave => {
                // Equatorial waveguide
                Some(self.trace_equatorial_path(source.lon, target.lon))
            },
            _ => None
        }
    }
    
    /// Trace jet stream waveguide
    fn trace_jet_stream_path(
        &self,
        lat1: f64, lon1: f64,
        lat2: f64, lon2: f64
    ) -> Option<Vec<(f64, f64)>> {
        let mut path = Vec::new();
        
        let jet_lat = if lat1 > 0.0 { 40.0 } else { -40.0 };
        
        // Entry point to jet
        path.push((lat1, lon1));
        path.push((jet_lat, lon1));
        
        // Track actual jet stream from wind field data
        let n_steps = 20;
        let dlon = (lon2 - lon1) / n_steps as f64;
        let mut current_lat = jet_lat;
        
        for i in 1..=n_steps {
            let lon = lon1 + i as f64 * dlon;
            
            // Find jet core by tracking maximum wind speed
            let mut max_wind_speed = 0.0;
            let mut best_lat = current_lat;
            
            // Search latitudes around current position
            for lat_offset in -10..=10 {
                let test_lat = current_lat + lat_offset as f64;
                if test_lat.abs() <= 90.0 {
                    let wind_speed = self.estimate_jet_wind_speed(test_lat, lon);
                    if wind_speed > max_wind_speed {
                        max_wind_speed = wind_speed;
                        best_lat = test_lat;
                    }
                }
            }
            
            if max_wind_speed > 25.0 { // Minimum jet threshold
                current_lat = best_lat;
                path.push((current_lat, lon));
            }
        }
        
        // Exit to target
        path.push((lat2, lon2));
        
        if path.len() > 2 { Some(path) } else { None }
    }
    
    /// Estimate jet stream wind speed from temperature gradients and Coriolis force
    fn estimate_jet_wind_speed(&self, lat: f64, lon: f64) -> f64 {
        let lat_rad = lat.to_radians();
        let coriolis = 2.0 * 7.2921e-5 * lat_rad.sin(); // Coriolis parameter
        
        if coriolis.abs() < 1e-10 {
            return 0.0; // No geostrophic wind near equator
        }
        
        // Estimate temperature gradient (simplified using seasonal and latitudinal patterns)
        let season = 0.0; // Would come from actual data
        let temp_gradient = self.estimate_temperature_gradient(lat, lon, season);
        
        // Thermal wind relationship: du/dz = -(g/fT) * dT/dy
        // Integrated over typical jet altitude (~250 hPa)
        let altitude_range = 5000.0; // meters (surface to ~250 hPa)
        let temperature = 250.0; // K at jet level
        let gravity = 9.81;
        
        let geostrophic_wind = (gravity * temp_gradient * altitude_range) / (coriolis.abs() * temperature);
        
        // Add climatological jet strength modulation
        let jet_strength_factor = if lat.abs() > 25.0 && lat.abs() < 65.0 {
            1.0 + 0.3 * ((lat.abs() - 45.0) / 20.0).cos() // Peak around 45°
        } else {
            0.3 // Weaker jets outside main belt
        };
        
        (geostrophic_wind * jet_strength_factor).abs().min(100.0) // Cap at 100 m/s
    }
    
    /// Estimate meridional temperature gradient
    fn estimate_temperature_gradient(&self, lat: f64, _lon: f64, season: f64) -> f64 {
        // Simplified temperature gradient model
        // Real implementation would use actual temperature data
        
        let base_gradient = -0.006; // K/km (typical tropospheric lapse rate)
        
        // Seasonal modulation
        let seasonal_factor = 1.0 + 0.3 * (season * 2.0 * std::f64::consts::PI).cos();
        
        // Latitudinal variation - stronger gradients at mid-latitudes
        let lat_factor = if lat.abs() > 30.0 && lat.abs() < 60.0 {
            1.5 * (1.0 - ((lat.abs() - 45.0) / 15.0).powi(2))
        } else {
            0.5
        };
        
        base_gradient * seasonal_factor * lat_factor
    }
    
    /// Trace Rossby wave path
    fn trace_rossby_path(
        &self,
        lat1: f64, lon1: f64,
        lat2: f64, lon2: f64
    ) -> Option<Vec<(f64, f64)>> {
        let mut path = Vec::new();
        
        let n_points = 20;
        for i in 0..=n_points {
            let t = i as f64 / n_points as f64;
            
            // Great circle interpolation
            let lat = lat1 * (1.0 - t) + lat2 * t;
            let lon = lon1 * (1.0 - t) + lon2 * t;
            
            let wavelength = 60.0;
            let amplitude = 10.0;
            let phase = 2.0 * PI * lon / wavelength;
            let wave_lat = lat + amplitude * phase.sin();
            
            path.push((wave_lat, lon));
        }
        
        Some(path)
    }
    
    /// Trace equatorial waveguide
    fn trace_equatorial_path(&self, lon1: f64, lon2: f64) -> Vec<(f64, f64)> {
        let mut path = Vec::new();
        
        let n_points = 10;
        for i in 0..=n_points {
            let t = i as f64 / n_points as f64;
            let lon = lon1 * (1.0 - t) + lon2 * t;
            path.push((0.0, lon));  // Along equator
        }
        
        path
    }
}

// ---
// BEIJING-MIAMI TELECONNECTION EXAMPLE
// ---

pub struct BeijingMiamiTeleconnection {
    analyzer: Arc<TeleconnectionAnalyzer>,
    beijing_state: PadicClimateState,
    miami_state: PadicClimateState,
}

impl BeijingMiamiTeleconnection {
    pub fn new() -> Self {
        // Beijing: 39.9°N, 116.4°E (SOURCE: Google Maps coordinates)
        let mut beijing_obs = HashMap::new();
        beijing_obs.insert("T".to_string(), 285.0);  // 285K = 12°C typical Beijing annual mean
        beijing_obs.insert("Z".to_string(), 5500.0);  // 5500m typical 500hPa height
        beijing_obs.insert("U".to_string(), 15.0);  // GUESS: 15 m/s westerly
        beijing_obs.insert("V".to_string(), -5.0);   // GUESS: -5 m/s northerly
        beijing_obs.insert("ENSO".to_string(), -0.5);
        beijing_obs.insert("PDO".to_string(), 0.3);
        
        let beijing_state = PadicClimateState::from_observations(
            39.9, 116.4, 500.0, &beijing_obs
        );
        
        // Miami: 25.8°N, 80.2°W = 279.8°E (SOURCE: Google Maps coordinates)
        let mut miami_obs = HashMap::new();
        miami_obs.insert("T".to_string(), 298.0);  // 298K = 25°C typical Miami annual mean
        miami_obs.insert("Z".to_string(), 5850.0);  // 5850m subtropical high influence
        miami_obs.insert("U".to_string(), -8.0);   // GUESS: -8 m/s easterly trade winds
        miami_obs.insert("V".to_string(), 3.0);    // GUESS: 3 m/s southerly
        miami_obs.insert("ENSO".to_string(), -0.5);
        miami_obs.insert("NAO".to_string(), 0.7);
        
        let miami_state = PadicClimateState::from_observations(
            25.8, 279.8, 500.0, &miami_obs
        );
        
        Self {
            analyzer: Arc::new(TeleconnectionAnalyzer::new(0.3)),
            beijing_state,
            miami_state,
        }
    }
    
    /// Compute teleconnection strength
    pub fn compute_teleconnection(&self) -> TeleconnectionResult {
        let metric = PadicTeleconnectionMetric::new();
        
        // P-adic distance
        let padic_dist = metric.padic_distance(&self.beijing_state, &self.miami_state);
        
        // Physical distance
        let physical_dist = metric.haversine_distance(
            self.beijing_state.lat, self.beijing_state.lon,
            self.miami_state.lat, self.miami_state.lon
        );
        
        // Teleconnection strength (inverse of p-adic distance)
        let strength = 1.0 / (1.0 + padic_dist);
        
        // Identify likely mechanism
        let mechanism = if self.beijing_state.lat > 30.0 && self.miami_state.lat > 20.0 {
            // Both in westerlies/subtropics
            TeleconnectionMechanism::JetStreamWaveguide
        } else {
            TeleconnectionMechanism::MixedMode
        };
        
        // Expected lag
        let lag_days = match mechanism {
            TeleconnectionMechanism::JetStreamWaveguide => {
                    physical_dist / (30.0 * 86400.0)
            },
            _ => physical_dist / (10.0 * 86400.0)
        };
        
        TeleconnectionResult {
            source: "Beijing".to_string(),
            target: "Miami".to_string(),
            padic_distance: padic_dist,
            physical_distance_km: physical_dist / 1000.0,
            strength_index: strength,
            mechanism,
            expected_lag_days: lag_days,
            confidence: strength * 0.8,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TeleconnectionResult {
    pub source: String,
    pub target: String,
    pub padic_distance: f64,
    pub physical_distance_km: f64,
    pub strength_index: f64,
    pub mechanism: TeleconnectionMechanism,
    pub expected_lag_days: f64,
    pub confidence: f64,
}

// ---
// TESTS
// ---

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_padic_distance() {
        let metric = PadicTeleconnectionMetric::new();
        
        // Create two similar states
        let mut obs1 = HashMap::new();
        obs1.insert("T".to_string(), 288.0);
        let state1 = PadicClimateState::from_observations(40.0, 116.0, 500.0, &obs1);
        
        let mut obs2 = HashMap::new();
        obs2.insert("T".to_string(), 288.5);
        let state2 = PadicClimateState::from_observations(40.0, 117.0, 500.0, &obs2);
        
        let dist = metric.padic_distance(&state1, &state2);
        assert!(dist > 0.0 && dist < 1.0);
    }
    
    #[test]
    fn test_beijing_miami() {
        let bm = BeijingMiamiTeleconnection::new();
        let result = bm.compute_teleconnection();
        
        println!("Beijing-Miami Teleconnection: {:?}", result);
        assert!(result.physical_distance_km > 10000.0);
        assert!(result.expected_lag_days > 0.0);
    }
}

// ---
// MAIN ENTRY POINT
// ---

fn main() {
    println!("---");
    println!("CLIMATE P-ADIC TELECONNECTION ANALYSIS");
    println!("---");
    
    let bm = BeijingMiamiTeleconnection::new();
    let result = bm.compute_teleconnection();
    
    println!("\nBeijing → Miami Teleconnection:");
    println!("  P-adic distance: {:.4}", result.padic_distance);
    println!("  Physical distance: {:.0} km", result.physical_distance_km);
    println!("  Strength index: {:.3}", result.strength_index);
    println!("  Mechanism: {:?}", result.mechanism);
    println!("  Expected lag: {:.1} days", result.expected_lag_days);
    println!("  Confidence: {:.1}%", result.confidence * 100.0);
    
    println!("\nTeleconnection analysis attempt finished (results unvalidated)");
}