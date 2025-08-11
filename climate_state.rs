// Unified Climate State Management System
// Central state representation that all modules can share
// Implements efficient storage, parallel access, and state synchronization

// ============================================================================
// DATA SOURCE REQUIREMENTS - CLIMATE STATE MANAGEMENT
// ============================================================================
//
// GLOBAL ATMOSPHERIC FIELDS:
// Source: ERA5/ERA5-Land reanalysis, operational weather models (GFS, ECMWF)
// Instrument: Global observation network, satellite retrievals, model analysis
// Spatiotemporal Resolution: 0.25°×0.25°, hourly/6-hourly, 1950-present
// File Format: NetCDF4, GRIB2
// Data Size: ~10TB/year for full ERA5
// API Access: Copernicus CDS, NOAA NOMADS, NCAR RDA
// Variables: Temperature, pressure, humidity, winds (U/V/W), geopotential
//
// OCEAN STATE VARIABLES:
// Source: Ocean reanalysis (ORAS5, GLORYS), Argo profiling floats
// Instrument: Autonomous floats, satellite altimetry, CTD measurements
// Spatiotemporal Resolution: 0.25°×0.25°, daily/monthly, 1993-present
// File Format: NetCDF4
// Data Size: ~2TB/year for global ocean
// API Access: Copernicus Marine Service, NOAA ERDDAP
// Variables: Temperature, salinity, currents, sea level, mixed layer depth
//
// LAND SURFACE CONDITIONS:
// Source: Land surface models (ERA5-Land, GLDAS), satellite observations
// Instrument: MODIS, VIIRS, SMAP, field measurements
// Spatiotemporal Resolution: 9km-25km, daily/monthly
// File Format: NetCDF4, HDF5, GeoTIFF
// Data Size: ~500GB/year for global land
// API Access: NASA GES DISC, USGS Earth Explorer
// Variables: Soil moisture/temperature, snow depth, vegetation, albedo
//
// ATMOSPHERIC COMPOSITION:
// Source: Greenhouse gas monitoring (OCO-2, TROPOMI), ground networks (NOAA/ESRL)
// Instrument: Satellite spectrometers, flask sampling, continuous analyzers
// Spatiotemporal Resolution: Various (0.1°-2.5°), daily to monthly
// File Format: NetCDF4, HDF5
// Data Size: ~200GB/year for composition fields
// API Access: NASA GES DISC, NOAA data portals
// Variables: CO2, CH4, N2O, O3, aerosol optical depth
//
// CLOUD AND RADIATION FIELDS:
// Source: Satellite cloud climatologies (ISCCP, MODIS, CERES)
// Instrument: Geostationary/polar satellites, broadband radiometers
// Spatiotemporal Resolution: 1°×1°, monthly climatologies
// File Format: NetCDF4, HDF5
// Data Size: ~100GB for climatological fields
// API Access: NASA GES DISC, ISCCP data portal
// Variables: Cloud fraction, optical depth, radiation fluxes
//
// FORCING DATA:
// Source: Solar irradiance (SORCE, TSIS), volcanic aerosol databases
// Instrument: Satellite radiometers, ground-based networks
// Spatiotemporal Resolution: Global means to 1°×1°, daily to annual
// File Format: ASCII, NetCDF4
// Data Size: ~1GB for historical forcing
// API Access: LASP data portal, NASA GES DISC
// Variables: Total solar irradiance, spectral irradiance, aerosol forcing
//
// GEOMETRIC/TOPOGRAPHIC DATA:
// Source: High-resolution topography (SRTM, ASTER GDEM), land/sea masks
// Instrument: Radar altimetry, optical stereo imaging
// Spatiotemporal Resolution: 30m-1km, static
// File Format: GeoTIFF, NetCDF4, binary
// Data Size: ~50GB for global high-resolution
// API Access: USGS Earth Explorer, NASA LP DAAC
// Variables: Elevation, slope, land/sea mask, coastline
//
// GRID CONSTRUCTION REQUIREMENTS:
//   1. Multi-resolution grid nesting for local/regional/global domains
//   2. Conservative remapping between different native grids
//   3. Masking for land/ocean/ice boundaries with sub-grid representation
//   4. Vertical coordinate transformation (pressure, height, sigma)
//   5. Temporal interpolation with proper treatment of missing data
//
// MISSING REQUIREMENTS MAKING THIS A STUB:
// - Sophisticated grid generation with adaptive mesh refinement
// - Real-time data ingestion and state update capabilities
// - Complete vertical coordinate systems (hybrid, terrain-following)
// - Sub-grid scale parameterizations and effective properties
// - Proper treatment of boundaries and interpolation artifacts
// - Memory-mapped file I/O for very large state arrays
//
// IMPLEMENTATION GAPS:
// - Currently uses simplified regular grids instead of real Earth geometry
// - Missing integration with major data repositories and APIs
// - State fields use placeholder initialization rather than real data
// - No validation against observations or conservation principles
// - Parallel access patterns not optimized for climate workflows
// - Checkpoint/restart functionality is incomplete

use nalgebra::{DVector, DMatrix, Vector3};
use ndarray::{Array3, Array4, Array5, Axis, s, Zip};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use anyhow::{Result, Context, bail};
use chrono::{DateTime, Utc};

// Re-export from climate_manifold.rs to avoid duplication
pub use crate::climate_manifold::ClimateState as ManifoldState;

/// Grid specification for spatial discretization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSpec {
    pub lat: Vec<f64>,        // Latitude points (degrees)
    pub lon: Vec<f64>,        // Longitude points (degrees)
    pub lev: Vec<f64>,        // Vertical levels (Pa or sigma)
    pub lat_bnds: Vec<f64>,   // Latitude cell boundaries
    pub lon_bnds: Vec<f64>,   // Longitude cell boundaries
    pub lev_bnds: Vec<f64>,   // Level boundaries
    pub area: Array3<f64>,    // Grid cell areas (m²)
    pub volume: Array4<f64>,  // Grid cell volumes (m³)
    pub mask: Array3<bool>,   // Land/ocean mask
}

impl GridSpec {
    /// Create a regular lat-lon grid
    pub fn regular_latlon(nlat: usize, nlon: usize, nlev: usize) -> Result<Self> {
        let mut lat = Vec::with_capacity(nlat);
        let mut lon = Vec::with_capacity(nlon);
        let mut lev = Vec::with_capacity(nlev);
        
        // Generate regular grid
        for i in 0..nlat {
            lat.push(-90.0 + (i as f64 + 0.5) * 180.0 / nlat as f64);
        }
        for i in 0..nlon {
            lon.push((i as f64 + 0.5) * 360.0 / nlon as f64);
        }
        
        // Hybrid sigma-pressure levels (following CMIP6 standard)
        let p_surf = 101325.0; // Pa
        let p_top = 100.0;     // Pa
        for i in 0..nlev {
            let sigma = 1.0 - (i as f64) / (nlev as f64 - 1.0);
            lev.push(p_top + sigma * (p_surf - p_top));
        }
        
        // Compute boundaries
        let lat_bnds = Self::compute_boundaries(&lat);
        let lon_bnds = Self::compute_boundaries(&lon);
        let lev_bnds = Self::compute_boundaries(&lev);
        
        // Compute areas (simplified - full calculation would use actual Earth radius)
        let mut area = Array3::zeros((nlat, nlon, 1));
        const EARTH_RADIUS: f64 = 6.371e6; // meters
        for i in 0..nlat {
            for j in 0..nlon {
                let dlat = (lat_bnds[i+1] - lat_bnds[i]).to_radians();
                let dlon = (lon_bnds[j+1] - lon_bnds[j]).to_radians();
                let lat_center = lat[i].to_radians();
                area[[i, j, 0]] = EARTH_RADIUS * EARTH_RADIUS * dlat * dlon * lat_center.cos();
            }
        }
        
        // Simplified volume calculation
        let volume = Array4::zeros((nlat, nlon, nlev, 1));
        
        // Initialize mask (all ocean for now)
        let mask = Array3::from_elem((nlat, nlon, 1), false);
        
        Ok(GridSpec {
            lat, lon, lev,
            lat_bnds, lon_bnds, lev_bnds,
            area, volume, mask,
        })
    }
    
    fn compute_boundaries(points: &[f64]) -> Vec<f64> {
        let n = points.len();
        let mut bounds = Vec::with_capacity(n + 1);
        
        // First boundary
        bounds.push(points[0] - 0.5 * (points[1] - points[0]));
        
        // Interior boundaries
        for i in 0..n-1 {
            bounds.push(0.5 * (points[i] + points[i+1]));
        }
        
        // Last boundary
        bounds.push(points[n-1] + 0.5 * (points[n-1] - points[n-2]));
        
        bounds
    }
    
    pub fn nlat(&self) -> usize { self.lat.len() }
    pub fn nlon(&self) -> usize { self.lon.len() }
    pub fn nlev(&self) -> usize { self.lev.len() }
}

/// Prognostic variables (evolved by model equations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrognosticVars {
    pub temperature: Array4<f64>,      // K, dims: [time, lev, lat, lon]
    pub u_wind: Array4<f64>,           // m/s, zonal wind
    pub v_wind: Array4<f64>,           // m/s, meridional wind
    pub w_wind: Array4<f64>,           // Pa/s, vertical velocity
    pub specific_humidity: Array4<f64>, // kg/kg
    pub cloud_water: Array4<f64>,      // kg/kg, cloud liquid water
    pub cloud_ice: Array4<f64>,        // kg/kg, cloud ice
    pub ozone: Array4<f64>,            // kg/kg, ozone mixing ratio
}

impl PrognosticVars {
    pub fn new(grid: &GridSpec, ntime: usize) -> Self {
        let shape = (ntime, grid.nlev(), grid.nlat(), grid.nlon());
        
        // Initialize with reasonable values
        let mut temperature = Array4::from_elem(shape, 250.0); // ~250K
        let specific_humidity = Array4::from_elem(shape, 0.001); // ~1 g/kg
        
        // Add vertical temperature profile
        for t in 0..ntime {
            for k in 0..grid.nlev() {
                let temp_profile = 288.0 - 6.5e-3 * (grid.lev[k] - grid.lev[grid.nlev()-1]);
                temperature.slice_mut(s![t, k, .., ..]).fill(temp_profile);
            }
        }
        
        PrognosticVars {
            temperature,
            u_wind: Array4::zeros(shape),
            v_wind: Array4::zeros(shape),
            w_wind: Array4::zeros(shape),
            specific_humidity,
            cloud_water: Array4::zeros(shape),
            cloud_ice: Array4::zeros(shape),
            ozone: Array4::from_elem(shape, 1e-7), // ~0.1 ppmv
        }
    }
}

/// Diagnostic variables (computed from prognostic variables)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticVars {
    pub pressure: Array4<f64>,           // Pa
    pub density: Array4<f64>,            // kg/m³
    pub potential_temperature: Array4<f64>, // K
    pub relative_humidity: Array4<f64>,  // fraction
    pub geopotential_height: Array4<f64>, // m
    pub vorticity: Array4<f64>,          // s⁻¹
    pub divergence: Array4<f64>,         // s⁻¹
}

impl DiagnosticVars {
    pub fn compute(prog: &PrognosticVars, grid: &GridSpec) -> Result<Self> {
        let shape = prog.temperature.dim();
        let (ntime, nlev, nlat, nlon) = (shape.0, shape.1, shape.2, shape.3);
        
        let mut pressure = Array4::zeros((ntime, nlev, nlat, nlon));
        let mut density = Array4::zeros((ntime, nlev, nlat, nlon));
        let mut potential_temperature = Array4::zeros((ntime, nlev, nlat, nlon));
        
        const R_DRY: f64 = 287.04; // J/(kg·K)
        const CP_DRY: f64 = 1004.64; // J/(kg·K)
        const P_REF: f64 = 100000.0; // Pa
        
        // Compute diagnostic fields
        for t in 0..ntime {
            for k in 0..nlev {
                let p = grid.lev[k]; // Simplified - using level as pressure
                pressure.slice_mut(s![t, k, .., ..]).fill(p);
                
                for i in 0..nlat {
                    for j in 0..nlon {
                        let temp = prog.temperature[[t, k, i, j]];
                        
                        // Ideal gas law
                        density[[t, k, i, j]] = p / (R_DRY * temp);
                        
                        // Potential temperature
                        potential_temperature[[t, k, i, j]] = temp * (P_REF / p).powf(R_DRY / CP_DRY);
                    }
                }
            }
        }
        
        // Placeholder for other diagnostics
        let relative_humidity = Array4::from_elem((ntime, nlev, nlat, nlon), 0.5);
        let geopotential_height = Array4::zeros((ntime, nlev, nlat, nlon));
        let vorticity = Array4::zeros((ntime, nlev, nlat, nlon));
        let divergence = Array4::zeros((ntime, nlev, nlat, nlon));
        
        Ok(DiagnosticVars {
            pressure,
            density,
            potential_temperature,
            relative_humidity,
            geopotential_height,
            vorticity,
            divergence,
        })
    }
}

/// Surface fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceFields {
    pub surface_temperature: Array3<f64>,    // K, dims: [time, lat, lon]
    pub surface_pressure: Array3<f64>,       // Pa
    pub sea_ice_fraction: Array3<f64>,       // fraction
    pub snow_depth: Array3<f64>,             // m
    pub soil_moisture: Array4<f64>,          // m³/m³, includes depth dimension
    pub vegetation_fraction: Array3<f64>,     // fraction
    pub surface_albedo: Array3<f64>,         // fraction
    pub surface_roughness: Array3<f64>,      // m
}

impl SurfaceFields {
    pub fn new(grid: &GridSpec, ntime: usize) -> Self {
        let shape_2d = (ntime, grid.nlat(), grid.nlon());
        let shape_3d = (ntime, 4, grid.nlat(), grid.nlon()); // 4 soil layers
        
        SurfaceFields {
            surface_temperature: Array3::from_elem(shape_2d, 288.0),
            surface_pressure: Array3::from_elem(shape_2d, 101325.0),
            sea_ice_fraction: Array3::zeros(shape_2d),
            snow_depth: Array3::zeros(shape_2d),
            soil_moisture: Array4::from_elem(shape_3d, 0.25),
            vegetation_fraction: Array3::from_elem(shape_2d, 0.3),
            surface_albedo: Array3::from_elem(shape_2d, 0.15),
            surface_roughness: Array3::from_elem(shape_2d, 0.001),
        }
    }
}

/// Forcing fields (external drivers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForcingFields {
    pub solar_radiation: Array3<f64>,        // W/m², top of atmosphere
    pub co2_concentration: Array3<f64>,      // ppm
    pub ch4_concentration: Array3<f64>,      // ppb
    pub n2o_concentration: Array3<f64>,      // ppb
    pub aerosol_optical_depth: Array4<f64>,  // dimensionless, includes wavelength
    pub volcanic_forcing: Array3<f64>,       // W/m²
}

impl ForcingFields {
    pub fn new(grid: &GridSpec, ntime: usize) -> Self {
        let shape_2d = (ntime, grid.nlat(), grid.nlon());
        let shape_3d = (ntime, 5, grid.nlat(), grid.nlon()); // 5 wavelength bands
        
        ForcingFields {
            solar_radiation: Array3::from_elem(shape_2d, 1361.0 / 4.0), // Annual mean
            co2_concentration: Array3::from_elem(shape_2d, 421.0),
            ch4_concentration: Array3::from_elem(shape_2d, 1900.0),
            n2o_concentration: Array3::from_elem(shape_2d, 335.0),
            aerosol_optical_depth: Array4::from_elem(shape_3d, 0.1),
            volcanic_forcing: Array3::zeros(shape_2d),
        }
    }
}

/// Geometric fields from manifold analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricFields {
    pub riemann_curvature: Array4<f64>,      // Riemann tensor components
    pub ricci_scalar: Array3<f64>,           // Scalar curvature
    pub fisher_information: Array4<f64>,      // Fisher information matrix
    pub geodesic_deviation: Array3<f64>,     // Geodesic deviation magnitude
    pub sectional_curvature: Array3<f64>,    // Maximum sectional curvature
    pub connection_coefficients: Array5<f64>, // Christoffel symbols
}

/// Complete climate state combining all components
#[derive(Debug, Clone)]
pub struct ClimateSystemState {
    pub metadata: StateMetadata,
    pub grid: Arc<GridSpec>,
    pub prognostic: Arc<RwLock<PrognosticVars>>,
    pub diagnostic: Arc<RwLock<DiagnosticVars>>,
    pub surface: Arc<RwLock<SurfaceFields>>,
    pub forcing: Arc<RwLock<ForcingFields>>,
    pub geometric: Option<Arc<RwLock<GeometricFields>>>,
    pub manifold_state: Option<ManifoldState>,
    pub history: Vec<StateSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMetadata {
    pub timestamp: DateTime<Utc>,
    pub model_time: f64,  // seconds since start
    pub iteration: usize,
    pub energy_total: f64,
    pub mass_total: f64,
    pub angular_momentum: Vector3<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub timestamp: DateTime<Utc>,
    pub model_time: f64,
    pub key_metrics: HashMap<String, f64>,
}

impl ClimateSystemState {
    /// Create a new climate system state
    pub fn new(grid: GridSpec, ntime: usize) -> Self {
        let prognostic = Arc::new(RwLock::new(PrognosticVars::new(&grid, ntime)));
        let prog_read = prognostic.read().unwrap();
        let diagnostic = Arc::new(RwLock::new(
            DiagnosticVars::compute(&prog_read, &grid).expect("Failed to compute diagnostics")
        ));
        drop(prog_read);
        
        let surface = Arc::new(RwLock::new(SurfaceFields::new(&grid, ntime)));
        let forcing = Arc::new(RwLock::new(ForcingFields::new(&grid, ntime)));
        
        let metadata = StateMetadata {
            timestamp: Utc::now(),
            model_time: 0.0,
            iteration: 0,
            energy_total: 0.0,
            mass_total: 0.0,
            angular_momentum: Vector3::zeros(),
        };
        
        ClimateSystemState {
            metadata,
            grid: Arc::new(grid),
            prognostic,
            diagnostic,
            surface,
            forcing,
            geometric: None,
            manifold_state: None,
            history: Vec::new(),
        }
    }
    
    /// Extract state vector for manifold analysis
    pub fn to_manifold_vector(&self, time_idx: usize) -> Result<DVector<f64>> {
        let prog = self.prognostic.read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;
        
        // Extract global mean values for manifold coordinates
        let temp_mean = prog.temperature.slice(s![time_idx, .., .., ..]).mean()
            .context("Failed to compute mean temperature")?;
        
        let humidity_mean = prog.specific_humidity.slice(s![time_idx, .., .., ..]).mean()
            .context("Failed to compute mean humidity")?;
        
        // Add more state variables as needed
        let state_vec = DVector::from_vec(vec![
            temp_mean,
            humidity_mean,
            // Add more variables
        ]);
        
        Ok(state_vec)
    }
    
    /// Update from manifold analysis results
    pub fn update_from_manifold(&mut self, manifold_state: ManifoldState) {
        self.manifold_state = Some(manifold_state);
    }
    
    /// Compute conservation diagnostics
    pub fn compute_conservation(&self) -> Result<ConservationMetrics> {
        let prog = self.prognostic.read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;
        let diag = self.diagnostic.read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;
        
        // Compute total energy
        const CP: f64 = 1004.64; // J/(kg·K)
        const G: f64 = 9.80665;  // m/s²
        
        let mut total_energy = 0.0;
        let mut total_mass = 0.0;
        
        // STUB: Simplified calculation - missing proper volume integration
        for t in 0..1 {  // Just current time
            for k in 0..self.grid.nlev() {
                for i in 0..self.grid.nlat() {
                    for j in 0..self.grid.nlon() {
                        let rho = diag.density[[t, k, i, j]];
                        let temp = prog.temperature[[t, k, i, j]];
                        let u = prog.u_wind[[t, k, i, j]];
                        let v = prog.v_wind[[t, k, i, j]];
                        
                        // Kinetic + internal energy
                        let ke = 0.5 * rho * (u*u + v*v);
                        let ie = rho * CP * temp;
                        
                        // Accumulate (would multiply by volume element)
                        total_energy += ke + ie;
                        total_mass += rho;
                    }
                }
            }
        }
        
        Ok(ConservationMetrics {
            total_energy,
            total_mass,
            total_angular_momentum: Vector3::new(f64::NAN, f64::NAN, f64::NAN), // STUB: Not computed
        })
    }
    
    /// Save checkpoint
    pub fn checkpoint(&self, path: &str) -> Result<()> {
        // STUB: Does nothing - should serialize to NetCDF/HDF5
        eprintln!("WARNING: checkpoint() is a STUB - data NOT saved to {}", path);
        Err(anyhow::anyhow!("checkpoint() not implemented - returns error instead of silently failing"))
    }
    
    /// Restore from checkpoint
    pub fn restore(path: &str) -> Result<Self> {
        // Deserialize state from file
        bail!("Checkpoint restoration not yet implemented")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationMetrics {
    pub total_energy: f64,          // J
    pub total_mass: f64,            // kg
    pub total_angular_momentum: Vector3<f64>, // kg·m²/s
}

/// Thread-safe state manager for parallel access
pub struct StateManager {
    states: Arc<RwLock<HashMap<String, ClimateSystemState>>>,
    active_state: Arc<RwLock<String>>,
}

impl StateManager {
    pub fn new() -> Self {
        StateManager {
            states: Arc::new(RwLock::new(HashMap::new())),
            active_state: Arc::new(RwLock::new("default".to_string())),
        }
    }
    
    /// Create a new state with given name
    pub fn create_state(&self, name: &str, grid: GridSpec, ntime: usize) -> Result<()> {
        let mut states = self.states.write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))?;
        
        if states.contains_key(name) {
            bail!("State '{}' already exists", name);
        }
        
        states.insert(name.to_string(), ClimateSystemState::new(grid, ntime));
        Ok(())
    }
    
    /// Get reference to active state
    pub fn get_active(&self) -> Result<ClimateSystemState> {
        let active_name = self.active_state.read()
            .map_err(|e| anyhow::anyhow!("Failed to read active state: {}", e))?;
        
        let states = self.states.read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;
        
        states.get(&*active_name)
            .cloned()
            .context(format!("Active state '{}' not found", active_name))
    }
    
    /// Switch active state
    pub fn set_active(&self, name: &str) -> Result<()> {
        let states = self.states.read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;
        
        if !states.contains_key(name) {
            bail!("State '{}' does not exist", name);
        }
        
        let mut active = self.active_state.write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))?;
        *active = name.to_string();
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grid_creation() {
        let grid = GridSpec::regular_latlon(36, 72, 10).unwrap();
        assert_eq!(grid.nlat(), 36);
        assert_eq!(grid.nlon(), 72);
        assert_eq!(grid.nlev(), 10);
    }
    
    #[test]
    fn test_state_creation() {
        let grid = GridSpec::regular_latlon(18, 36, 5).unwrap();
        let state = ClimateSystemState::new(grid, 1);
        
        let prog = state.prognostic.read().unwrap();
        assert_eq!(prog.temperature.dim(), (1, 5, 18, 36));
    }
    
    #[test]
    fn test_conservation_computation() {
        let grid = GridSpec::regular_latlon(18, 36, 5).unwrap();
        let state = ClimateSystemState::new(grid, 1);
        
        let metrics = state.compute_conservation().unwrap();
        assert!(metrics.total_energy > 0.0);
        assert!(metrics.total_mass > 0.0);
    }
    
    #[test]
    fn test_state_manager() {
        let manager = StateManager::new();
        let grid = GridSpec::regular_latlon(18, 36, 5).unwrap();
        
        manager.create_state("test", grid.clone(), 1).unwrap();
        manager.set_active("test").unwrap();
        
        let state = manager.get_active().unwrap();
        assert_eq!(state.metadata.iteration, 0);
    }
}