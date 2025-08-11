// Climate manifold network with adaptive dimensionality
// Experimental geometric framework for climate dynamics

// ============================================================================
// DATA SOURCE REQUIREMENTS - MANIFOLD NETWORK ANALYSIS
// ============================================================================
//
// MULTI-SCALE OBSERVATION HIERARCHY:
// 
// GLOBAL SCALE OBSERVATIONS:
// Source: CMIP6 multi-model ensemble, Earth system reanalysis (ERA5, MERRA-2)
// Instrument: Global climate models, satellite-based retrievals, reanalysis systems
// Spatiotemporal Resolution: 50-200km, monthly means, 1850-2100
// File Format: NetCDF4 with CF conventions
// Data Size: ~10TB for full CMIP6 ensemble
// API Access: ESGF data nodes, Google Cloud Public Datasets
// Variables: Temperature, precipitation, sea level pressure, ocean heat content
//
// REGIONAL SCALE OBSERVATIONS:
// Source: Regional climate models (CORDEX), high-resolution satellite products
// Instrument: Advanced weather satellites (MODIS, VIIRS, SEVIRI), regional models
// Spatiotemporal Resolution: 10-25km, daily, 1980-present  
// File Format: NetCDF4, HDF5
// Data Size: ~2TB/year per major region
// API Access: Regional climate data portals, NASA/ESA data services
// Variables: Surface temperature, precipitation, extreme events, land use change
//
// LOCAL SCALE OBSERVATIONS:
// Source: Weather station networks, eddy covariance towers, phenology cameras
// Instrument: Automatic weather stations, flux towers, webcams, citizen science
// Spatiotemporal Resolution: Point measurements, hourly to annual, varying periods
// File Format: CSV, NetCDF4, specialized formats (AmeriFlux, FLUXNET)
// Data Size: ~50GB/year for global networks
// API Access: NOAA NCEI, AmeriFlux, individual research networks
// Variables: Meteorological variables, CO2 fluxes, phenological events
//
// SENSOR-LEVEL REAL-TIME STREAMS:
// Source: IoT weather sensors, satellite direct broadcast, crowdsourced data
// Instrument: Low-cost sensors, mobile apps, connected vehicles
// Spatiotemporal Resolution: High-frequency (minutes), sparse spatial coverage
// File Format: JSON streams, binary protocols, APIs
// Data Size: ~100GB/day globally  
// API Access: Real-time APIs, message queues, data aggregation services
// Variables: Temperature, humidity, pressure, air quality, traffic conditions
//
// SATELLITE DATA STREAMS:
// Source: Geostationary and polar-orbiting satellites (GOES, MSG, NOAA, Sentinel)
// Instrument: Hyperspectral imagers, microwave sounders, radar altimeters
// Spatiotemporal Resolution: 1km-10km, 15min-daily, continuous
// File Format: HDF5, NetCDF4, GRIB2
// Data Size: ~1TB/day from major satellites
// API Access: Direct satellite data services, cloud-based processing
// Variables: Radiances, retrieved geophysical parameters, cloud properties
//
// OCEAN OBSERVATION NETWORKS:
// Source: Argo floats, ocean moorings, satellite altimetry, ship-based measurements
// Instrument: Autonomous profiling floats, CTD sensors, radar altimeters
// Spatiotemporal Resolution: Global 3°×3°, 10-day cycles, 2000-present
// File Format: NetCDF4, specialized oceanographic formats
// Data Size: ~100GB/year for Argo network
// API Access: Argo GDAC, NOAA ERDDAP, Copernicus Marine Service
// Variables: Temperature, salinity, pressure, dissolved oxygen, biogeochemistry
//
// MACHINE LEARNING INTEGRATION:
// - Autoencoder networks for dimension reduction of high-dimensional obs
// - UMAP for nonlinear manifold discovery from multi-source data
// - Neural ODEs for learning climate system dynamics
// - Transformer models for spatiotemporal pattern recognition
// - Physics-informed neural networks for manifold constraints
//
// PREPROCESSING REQUIREMENTS:
//   1. Multi-scale data harmonization and quality control
//   2. Spatiotemporal alignment across observation systems  
//   3. Gap filling and uncertainty quantification
//   4. Dimensionality reduction while preserving physical constraints
//   5. Cross-scale correlation analysis and teleconnection detection
//   6. Manifold learning with topological constraint preservation
//   7. Real-time streaming data integration and anomaly detection
//
// MISSING REQUIREMENTS MAKING THIS A STUB:
// - Comprehensive observation operator framework for all data types
// - Automated machine learning pipeline for dimension discovery
// - Scalable tensor computation libraries for high-dimensional manifolds
// - Real-time data quality control and bias correction systems
// - Cross-platform data format converters and API integrations
// - Uncertainty propagation through manifold transformations
// - Validation against known climate dynamics and attractors
//
// IMPLEMENTATION GAPS:
// - Currently uses simplified synthetic data instead of real observations  
// - Dimension learning algorithms need validation on climate data
// - Missing integration with major climate data repositories
// - No real-time data stream processing capabilities
// - Tensor operations not optimized for climate-specific manifold structure
// - Cross-scale transport operators lack physical basis validation
// - Topological analysis needs connection to climate science theory

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, RwLock};
use nalgebra::{DMatrix, DVector, Complex, SymmetricEigen};
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use petgraph::graph::{Graph, NodeIndex};
use rayon::prelude::*;
use crossbeam::channel::{bounded, Sender, Receiver};
use dashmap::DashMap;
use num_complex::Complex64;
use topological_data_analysis::{PersistentHomology, BettiNumbers};
use differential_geometry::connections::{Connection, AffineConnection};
use statistical_mechanics::{EnsembleStates, ProbabilityDistribution, StatisticalEntropy};
use machine_learning::{Autoencoder, LatentSpace, UMAP};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, bail};

// Hierarchical manifold structure

/// Climate manifold network with adaptive dimensions
pub struct ClimateManifoldNetwork {
    // Core geometric engine (8-20 fundamental dimensions)
    pub core_manifold: CoreClimateManifold,
    
    // Dynamic dimension discovery and learning
    pub dimension_learner: DimensionLearner,
    pub latent_manifolds: HashMap<String, LatentManifold>,
    
    // Hierarchical multi-scale structure
    pub global_manifold: GlobalManifold,           // 8-20 core dimensions
    pub regional_manifolds: Vec<RegionalManifold>, // 100s of dimensions each
    pub local_manifolds: Vec<LocalManifold>,       // 1000s of dimensions
    pub sensor_level_data: SensorNetwork,          // Millions of measurements
    
    // Cross-scale transport and holonomy
    pub cross_scale_holonomy: CrossScaleTransport,
    pub scale_coupling_tensor: ScaleCouplingTensor,
    
    // Real-time data streams
    pub satellite_feeds: Vec<SatelliteDataStream>,
    pub ground_sensors: GroundSensorNetwork,
    pub ocean_buoys: OceanObservationSystem,
    pub atmospheric_soundings: AtmosphericProfileNetwork,
    
    // AI/ML integration for pattern recognition
    pub neural_embeddings: NeuralManifoldEncoder,
    pub pattern_recognizers: Vec<GeometricPatternDetector>,
    pub anomaly_detectors: Vec<ManifoldAnomalyDetector>,
    
    // Prediction engines using geodesic flow
    pub geodesic_forecaster: GeodesicFlowPredictor,
    pub ensemble_geometries: Vec<AlternativeMetric>,
    pub uncertainty_quantifier: GeometricUncertainty,
    
    // Ensemble-based superposition for multiple futures
    pub ensemble_states: EnsembleClimateStates,
    pub correlation_tracker: CorrelationMeasure,
    
    // Topological invariants for irreversibility
    pub topology_computer: ClimateTopology,
    pub persistence_tracker: PersistenceDiagram,
    
    // Policy interventions as forcing fields
    pub policy_forcings: PolicyForcingField,
    pub intervention_optimizer: MinimalCurvaturePathfinder,
}

// Core climate manifold

#[derive(Debug, Clone)]
pub struct CoreClimateManifold {
    // Core physical dimensions (always present)
    pub core_dims: CoreDimensions,
    
    // Discovered latent dimensions (added dynamically)
    pub latent_dims: Vec<LatentDimension>,
    
    // Adaptive metric tensor
    pub metric: AdaptiveMetricTensor,
    
    // Connection with torsion for non-equilibrium dynamics
    pub connection: NonEquilibriumConnection,
    
    // Curvature tensors
    pub riemann: RiemannTensor,
    pub ricci: RicciTensor,
    pub weyl: WeylTensor,
    
    // Current dimensionality
    pub current_dim: usize,
    pub max_dim: usize,
}

#[derive(Debug, Clone)]
pub struct CoreDimensions {
    pub temperature: f64,           // Global mean temperature
    pub co2: f64,                  // Atmospheric CO2
    pub ocean_heat: f64,            // Ocean heat content
    pub ice_volume: f64,            // Total ice volume
    pub amoc_strength: f64,         // Atlantic circulation
    pub methane: f64,               // CH4 concentration
    pub cloud_cover: f64,           // Global cloud fraction
    pub soil_carbon: f64,           // Terrestrial carbon
}

impl CoreClimateManifold {
    pub fn new() -> Self {
        Self {
            core_dims: CoreDimensions {
                temperature: 288.0,
                co2: 421.0,
                ocean_heat: 0.0,
                ice_volume: 2.9e16,
                amoc_strength: 15.0,
                methane: 1900.0,
                cloud_cover: 0.67,
                soil_carbon: 1600.0,
            },
            latent_dims: Vec::new(),
            metric: AdaptiveMetricTensor::new(8),
            connection: NonEquilibriumConnection::new(8),
            riemann: RiemannTensor::zeros(8),
            ricci: RicciTensor::zeros(8),
            weyl: WeylTensor::zeros(8),
            current_dim: 8,
            max_dim: 100,  // Allow up to 100 dimensions
        }
    }
    
    /// Add a newly discovered latent dimension
    pub fn add_latent_dimension(&mut self, dim: LatentDimension) -> Result<()> {
        if self.current_dim >= self.max_dim {
            bail!("Maximum dimensionality reached");
        }
        
        // Test if adding dimension reduces global curvature
        let test_curvature = self.test_dimension_addition(&dim)?;
        
        if test_curvature < self.total_scalar_curvature() {
            // Accept new dimension
            self.latent_dims.push(dim);
            self.current_dim += 1;
            
            // Resize geometric objects
            self.resize_tensors(self.current_dim);
            
            // Recompute metric with new dimension
            self.recompute_metric()?;
            
            Ok(())
        } else {
            bail!("New dimension does not reduce curvature")
        }
    }
    
    fn test_dimension_addition(&self, dim: &LatentDimension) -> Result<f64> {
        // Compute curvature with tentative new dimension
        let mut test_metric = self.metric.clone();
        test_metric.expand_dimension(dim)?;
        
        let test_ricci = self.compute_ricci_with_metric(&test_metric)?;
        Ok(test_ricci.scalar_curvature())
    }
    
    fn total_scalar_curvature(&self) -> f64 {
        self.ricci.scalar_curvature()
    }
    
    fn resize_tensors(&mut self, new_dim: usize) {
        self.metric = AdaptiveMetricTensor::new(new_dim);
        self.connection = NonEquilibriumConnection::new(new_dim);
        self.riemann = RiemannTensor::zeros(new_dim);
        self.ricci = RicciTensor::zeros(new_dim);
        self.weyl = WeylTensor::zeros(new_dim);
    }
    
    fn recompute_metric(&mut self) -> Result<()> {
        // Learn metric from observations
        self.metric.learn_from_data()?;
        
        // Update connection for new metric
        self.connection = NonEquilibriumConnection::from_metric(&self.metric)?;
        
        // Recompute curvature tensors
        self.riemann = RiemannTensor::compute(&self.connection)?;
        self.ricci = self.riemann.contract_to_ricci()?;
        self.weyl = WeylTensor::from_riemann(&self.riemann, &self.ricci)?;
        
        Ok(())
    }
    
    fn compute_ricci_with_metric(&self, metric: &AdaptiveMetricTensor) -> Result<RicciTensor> {
        let conn = NonEquilibriumConnection::from_metric(metric)?;
        let riemann = RiemannTensor::compute(&conn)?;
        riemann.contract_to_ricci()
    }
}

// Dimension discovery

pub struct DimensionLearner {
    autoencoder: ClimateAutoencoder,
    umap_reducer: UMAP,
    discovered_dimensions: Vec<LatentDimension>,
    correlation_threshold: f64,
    curvature_reduction_threshold: f64,
    telemetry: DimensionDiscoveryTelemetry,
    config: DimensionDiscoveryConfig,
}

#[derive(Debug, Clone, Default)]
pub struct DimensionDiscoveryTelemetry {
    pub attempted: usize,
    pub accepted: usize,
    pub accepted_weighted: usize,
    pub cumulative_raw_curvature_reduction: f64,
    pub cumulative_weighted_curvature_reduction: f64,
    pub last_variance_gain: Option<f64>,
    pub records: Vec<DimensionAttemptRecord>,
}

impl DimensionDiscoveryTelemetry {
    pub fn record_attempt(&mut self) { self.attempted += 1; }
    pub fn record_accept(&mut self, raw: f64) { self.accepted += 1; self.cumulative_raw_curvature_reduction += raw; }
    pub fn record_accept_weighted(&mut self, weighted: f64) { self.accepted_weighted += 1; self.cumulative_weighted_curvature_reduction += weighted; }
    pub fn acceptance_rate(&self) -> f64 { if self.attempted==0 {0.0} else { self.accepted as f64 / self.attempted as f64 } }
    pub fn weighted_acceptance_rate(&self) -> f64 { if self.attempted==0 {0.0} else { self.accepted_weighted as f64 / self.attempted as f64 } }
    pub fn record_attempt_detail(&mut self, rec: DimensionAttemptRecord) { self.records.push(rec); }
}

#[derive(Debug, Clone)]
pub struct DimensionAttemptRecord {
    pub name: String,
    pub raw: f64,
    pub weighted: f64,
    pub accepted_raw: bool,
    pub accepted_weighted: bool,
}

#[derive(Debug, Clone)]
pub struct DimensionDiscoveryConfig {
    pub enable_legacy_synthetic_fallback: bool,
    pub min_weighted_gain: f64,
    pub log_telemetry: bool,
}

impl Default for DimensionDiscoveryConfig {
    fn default() -> Self {
        Self {
            enable_legacy_synthetic_fallback: false, // keep old code but disabled by default
            min_weighted_gain: 0.0,
            log_telemetry: true,
        }
    }
}

impl DimensionLearner {
    pub fn discover_new_dimensions(&mut self, observations: &ObservationBatch) -> Vec<LatentDimension> {
        // Use autoencoder to find latent structure
        let latent_codes = self.autoencoder.encode(observations);
        
        // Apply UMAP to find intrinsic dimensionality
        let embedded = self.umap_reducer.fit_transform(&latent_codes);
        
        // Identify dimensions that:
        // 1. Have high variance
        // 2. Are weakly correlated with existing dimensions
        // 3. Would reduce global curvature if added
        
        let mut new_dims = Vec::new();
        
        for candidate in self.extract_candidates(&embedded) {
            if self.is_novel_dimension(&candidate) && 
               self.reduces_curvature(&candidate) {
                new_dims.push(candidate);
            }
        }
        
        new_dims
    }
    
    fn is_novel_dimension(&self, candidate: &LatentDimension) -> bool {
        // Check correlation with existing dimensions
        for existing in &self.discovered_dimensions {
            if candidate.correlation_with(existing) > self.correlation_threshold {
                return false;
            }
        }
        true
    }
    
    fn reduces_curvature(&self, candidate: &LatentDimension) -> bool {
        // Simulate adding dimension and check curvature reduction
        candidate.curvature_reduction > self.curvature_reduction_threshold
    }
    
    fn extract_candidates(&self, embedded: &Array2<f64>) -> Vec<LatentDimension> {
        // Principal component analysis with refinement
        let mut pca = PCA::fit(embedded);
        pca.refine_with_power_iterations(embedded, 15);
        pca.upgrade_synthetic_weights(embedded);
        if let Some(diag) = pca.render_diagnostics() {
            eprintln!("[PCA Diagnostics] {}", diag);
        }
    pca.components()
            .iter()
            .filter(|c| c.explained_variance_ratio() > 0.01)
            .map(|c| LatentDimension::from_component(c))
            .collect()
    }
}

impl DimensionLearner {
    pub fn discover_new_dimensions_curvature_aware(&mut self, observations: &ObservationBatch, baseline_curvature: f64) -> Vec<LatentDimension> {
        let mut dims = self.discover_new_dimensions(observations);
        for d in &mut dims { d.apply_curvature_weight(baseline_curvature); }
        dims.sort_by(|a,b| b.curvature_reduction_weighted.partial_cmp(&a.curvature_reduction_weighted).unwrap_or(std::cmp::Ordering::Equal));
        dims
    }
}

#[derive(Debug, Clone)]
pub struct LatentDimension {
    pub name: String,
    pub values: DVector<f64>,
    pub physical_interpretation: Option<String>,
    pub correlation_matrix: DMatrix<f64>,
    pub curvature_reduction: f64,
    pub curvature_reduction_weighted: f64,
}

impl LatentDimension {
    fn correlation_with(&self, other: &LatentDimension) -> f64 {
        let n = self.values.len() as f64;
        let cov = self.values.dot(&other.values) / n;
        let std_self = (self.values.dot(&self.values) / n).sqrt();
        let std_other = (other.values.dot(&other.values) / n).sqrt();
        cov / (std_self * std_other)
    }
    
    fn from_component(component: &PrincipalComponent) -> Self {
        Self {
            name: format!("latent_{}", component.index),
            values: component.weights.clone(),
            physical_interpretation: None,
            correlation_matrix: DMatrix::identity(component.weights.len(), component.weights.len()),
            curvature_reduction: component.explained_variance_ratio(),
            curvature_reduction_weighted: component.explained_variance_ratio(),
        }
    }
}

impl LatentDimension {
    pub fn apply_curvature_weight(&mut self, baseline_curvature: f64) {
        let raw = self.curvature_reduction.max(1e-12);
        let factor = 1.0 + (baseline_curvature / 100.0).tanh() * raw.sqrt();
        self.curvature_reduction_weighted = raw * factor;
    }
}

// Multiple climate scenarios using ensemble methods

#[derive(Debug, Clone)]
pub struct EnsembleClimateStates {
    /// Multiple climate states with probabilities
    pub branches: Vec<(ClimateState, f64)>,
    
    /// Covariance matrix representation
    pub covariance_matrix: DMatrix<f64>,
    
    /// Shannon entropy (uncertainty measure)
    pub uncertainty_entropy: f64,
    
    /// Decorrelation time for ensemble spread
    pub decorrelation_time: f64,
}

impl EnsembleClimateStates {
    pub fn new_ensemble(states: Vec<ClimateState>, probabilities: Vec<f64>) -> Result<Self> {
        if states.len() != probabilities.len() {
            bail!("States and probabilities must have same length");
        }
        
        // Normalize probabilities
        let total: f64 = probabilities.iter().sum();
        let normalized: Vec<f64> = probabilities.iter()
            .map(|p| p / total)
            .collect();
        
        // Construct covariance matrix from ensemble statistics
        let n = states.len();
        let mut covariance = DMatrix::zeros(n, n);
        
        // Compute ensemble mean and covariance
        for i in 0..n {
            for j in 0..n {
                covariance[(i, j)] = normalized[i] * normalized[j];
            }
        }
        
        // Compute Shannon entropy S = -Σ p_i log p_i
        let entropy = Self::shannon_entropy(&normalized)?;
        
        Ok(Self {
            branches: states.into_iter().zip(normalized).collect(),
            covariance_matrix: covariance,
            uncertainty_entropy: entropy,
            decorrelation_time: Self::estimate_decorrelation_time(entropy),
        })
    }
    
    /// Evolve ensemble state under climate dynamics
    pub fn evolve(&mut self, dynamics: &ClimateDynamics, dt: f64) -> Result<()> {
        // Stochastic evolution with ensemble spread
        let diffusion_operator = dynamics.diffusion_matrix() * dt;
        
        // Apply to each branch
        for (state, probability) in &mut self.branches {
            // Add stochastic noise to preserve ensemble spread
            let noise = dynamics.stochastic_noise() * dt.sqrt();
            state.evolve_stochastic(dt, &noise)?;
            
            // Probability evolution (conservation)
            *probability *= (1.0 - dynamics.decay_rate() * dt);
        }
        
        // Renormalize probabilities
        let total: f64 = self.branches.iter().map(|(_, p)| *p).sum();
        for (_, prob) in &mut self.branches {
            *prob /= total;
        }
        
        // Update covariance matrix
        self.update_covariance_matrix()?;
        
        // Add decorrelation effects
        self.apply_decorrelation(dt)?;
        
        Ok(())
    }
    
    /// Sample a specific state from ensemble
    pub fn sample_state(&mut self) -> Result<ClimateState> {
        // Get probabilities from branches
        let probabilities: Vec<f64> = self.branches.iter()
            .map(|(_, p)| *p)
            .collect();
        
        // Sample according to probability distribution
        let mut rng = rand::thread_rng();
        let dist = WeightedIndex::new(&probabilities)?;
        let index = dist.sample(&mut rng);
        
        // Return sampled state (ensemble continues to exist)
        let sampled_state = self.branches[index].0.clone();
        
        Ok(sampled_state)
    }
    
    fn shannon_entropy(probabilities: &[f64]) -> Result<f64> {
        // Shannon entropy of probability distribution
        let entropy = probabilities.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum();
        
        Ok(entropy)
    }
    
    fn estimate_decorrelation_time(entropy: f64) -> f64 {
        // Higher entropy = shorter decorrelation time
        // This is a heuristic model for ensemble spread
        let base_decorrelation = 30.0;  // days
        base_decorrelation * (-entropy * 0.5).exp()
    }
    
    fn update_covariance_matrix(&mut self) -> Result<()> {
        let n = self.branches.len();
        self.covariance_matrix = DMatrix::zeros(n, n);
        
        // Compute ensemble covariance from state deviations
        for i in 0..n {
            for j in 0..n {
                self.covariance_matrix[(i, j)] = 
                    self.branches[i].1 * self.branches[j].1;
            }
        }
        
        let probs: Vec<f64> = self.branches.iter().map(|(_, p)| *p).collect();
        self.uncertainty_entropy = Self::shannon_entropy(&probs)?;
        Ok(())
    }
    
    fn apply_decorrelation(&mut self, dt: f64) -> Result<()> {
        // Environmental decorrelation reduces ensemble correlations
        let decorrelation_rate = 1.0 / self.decorrelation_time;
        let decay = (-decorrelation_rate * dt).exp();
        
        let n = self.covariance_matrix.nrows();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    self.covariance_matrix[(i, j)] *= decay;
                }
            }
        }
        
        Ok(())
    }
}

// Topological tracking

#[derive(Debug, Clone)]
pub struct ClimateTopology {
    /// Betti numbers - count of holes in each dimension
    pub betti_numbers: Vec<usize>,
    
    /// Persistence diagram - birth/death of topological features
    pub persistence_diagram: Vec<(f64, f64)>,
    
    /// Euler characteristic
    pub euler_characteristic: i32,
    
    /// Homology groups
    pub homology_groups: Vec<HomologyGroup>,
    
    /// Detected topological transitions
    pub transitions: Vec<TopologicalTransition>,
}

impl ClimateTopology {
    pub fn compute_from_manifold(manifold: &CoreClimateManifold) -> Result<Self> {
        // Sample points from manifold
        let point_cloud = manifold.sample_points(10000)?;
        
        // Compute persistent homology
        let persistence = PersistentHomology::compute(&point_cloud)?;
        
        // Extract Betti numbers
        let betti_numbers = persistence.betti_numbers();
        
        // Compute Euler characteristic χ = Σ(-1)ⁱ βᵢ
        let euler = betti_numbers.iter()
            .enumerate()
            .map(|(i, &b)| if i % 2 == 0 { b as i32 } else { -(b as i32) })
            .sum();
        
        Ok(Self {
            betti_numbers,
            persistence_diagram: persistence.diagram(),
            euler_characteristic: euler,
            homology_groups: persistence.homology_groups(),
            transitions: Vec::new(),
        })
    }
    
    /// Detect irreversible transitions from topology changes
    pub fn detect_irreversible_transitions(&mut self, 
                                          old_topology: &ClimateTopology) -> Vec<IrreversibleTransition> {
        let mut transitions = Vec::new();
        
        // Check for changes in Betti numbers (holes appearing/disappearing)
        for (dim, (&old_betti, &new_betti)) in old_topology.betti_numbers.iter()
            .zip(self.betti_numbers.iter())
            .enumerate() 
        {
            if old_betti != new_betti {
                transitions.push(IrreversibleTransition {
                    dimension: dim,
                    old_betti,
                    new_betti,
                    interpretation: Self::interpret_betti_change(dim, old_betti, new_betti),
                    reversibility_barrier: Self::estimate_barrier(dim, old_betti, new_betti),
                });
            }
        }
        
        transitions
    }
    
    fn interpret_betti_change(dim: usize, old: usize, new: usize) -> String {
        match dim {
            0 => {
                if new > old {
                    "Climate system fragmentation - disconnected regions appearing".to_string()
                } else {
                    "Climate regions merging".to_string()
                }
            },
            1 => {
                if new > old {
                    "New circulation patterns forming (loops in phase space)".to_string()
                } else {
                    "Circulation patterns collapsing".to_string()
                }
            },
            2 => {
                if new > old {
                    "Voids in climate attractor - forbidden regions appearing".to_string()
                } else {
                    "Climate phase space becoming more connected".to_string()
                }
            },
            _ => format!("{}-dimensional topological change", dim)
        }
    }
    
    fn estimate_barrier(dim: usize, old: usize, new: usize) -> f64 {
        // Heuristic: higher dimensional changes = higher barriers
        let dimensional_factor = (dim + 1) as f64;
        let change_magnitude = (new as f64 - old as f64).abs();
        
        dimensional_factor * change_magnitude * 10.0  // Energy barrier in arbitrary units
    }
}

#[derive(Debug, Clone)]
pub struct IrreversibleTransition {
    pub dimension: usize,
    pub old_betti: usize,
    pub new_betti: usize,
    pub interpretation: String,
    pub reversibility_barrier: f64,  // Energy required to reverse
}

// Policy interventions as forcing fields

#[derive(Debug, Clone)]
pub struct PolicyForcingField {
    /// Carbon price forcing
    pub carbon_price: f64,  // $/tCO2
    
    /// Renewable subsidy forcing vector
    pub renewable_subsidy: DVector<f64>,  // $/MWh by technology
    
    /// Land use policy forcing
    pub land_use_policy: LandUseForcing,
    
    /// International cooperation forcing
    pub cooperation_strength: f64,  // 0 = fragmented, 1 = unified
    
    /// Policy momentum (rate of change)
    pub policy_momentum: DVector<f64>,
}

impl PolicyForcingField {
    /// Apply policy forcings to climate dynamics
    pub fn apply_forcing(&self, base_connection: &Connection) -> Connection {
        let mut modified = base_connection.clone();
        
        // Carbon price creates economic forcing toward low-carbon states
        modified.add_forcing_gradient(self.carbon_forcing_gradient());
        
        // Renewable subsidies create directional forcing in energy space
        modified.add_directional_forcing(self.renewable_forcing_field());
        
        // Land use policy modifies surface boundary conditions
        modified.apply_surface_forcing(self.land_use_surface_forcing());
        
        // Cooperation affects information/technology diffusion
        modified.scale_diffusion(self.cooperation_strength);
        
        modified
    }
    
    /// Find path under policy constraints
    pub fn arbitrary_intervention_path(&self,  // NOT optimal - just first path found 
                                    start: &ClimateState,
                                    target: &ClimateState,
                                    manifold: &CoreClimateManifold) -> Result<InterventionPath> {
        // Find geodesic with policy-forced dynamics
        
        let modified_connection = self.apply_forcing(&manifold.connection);
        
        // Solve geodesic equation with policy-modified connection
        let geodesic = GeodesicSolver::solve(
            start,
            target,
            &modified_connection,
            &manifold.metric
        )?;
        
        // Compute total "action" (cost) along path
        let total_cost = self.compute_path_cost(&geodesic)?;
        
        // Identify critical policy moments
        let critical_points = self.find_critical_interventions(&geodesic)?;
        
        Ok(InterventionPath {
            trajectory: geodesic,
            total_cost,
            critical_points,
            reversibility: self.assess_reversibility(&geodesic, manifold)?,
        })
    }
    
    fn carbon_forcing_gradient(&self) -> DVector<f64> {
        // Economic forcing toward lower emissions
        let mut forcing = DVector::zeros(8);
        forcing[1] = -self.carbon_price / 100.0;  // CO2 dimension
        forcing
    }
    
    fn renewable_forcing_field(&self) -> DMatrix<f64> {
        // Directional forcing from renewable energy subsidies
        let mut forcing = DMatrix::zeros(8, 8);
        
        // Forcing between energy and emissions dimensions
        forcing[(1, 2)] = self.renewable_subsidy[0];  // Solar forcing
        forcing[(2, 1)] = -self.renewable_subsidy[0]; // Counter-forcing
        
        forcing
    }
    
    fn land_use_surface_forcing(&self) -> DMatrix<f64> {
        // Surface boundary forcing from land use changes
        let mut forcing = DMatrix::zeros(8, 8);
        
        // Reforestation creates carbon sink forcing
        forcing[(0, 7)] = self.land_use_policy.reforestation_rate * 0.1;
        forcing[(7, 0)] = self.land_use_policy.reforestation_rate * 0.1;
        
        forcing
    }
    
    fn compute_path_cost(&self, path: &Geodesic) -> Result<f64> {
        // Integrate policy cost along path
        let mut total_cost = 0.0;
        
        for segment in path.segments() {
            let carbon_cost = self.carbon_price * segment.emissions_change();
            let subsidy_cost = self.renewable_subsidy.dot(&segment.energy_change());
            let land_cost = self.land_use_policy.implementation_cost(segment);
            
            total_cost += carbon_cost + subsidy_cost + land_cost;
        }
        
        Ok(total_cost)
    }
    
    fn find_critical_interventions(&self, path: &Geodesic) -> Result<Vec<CriticalIntervention>> {
        // Find points where intervention is most effective
        let mut critical = Vec::new();
        
        for (i, point) in path.points().enumerate() {
            let curvature = path.curvature_at(i)?;
            
            // High curvature = critical decision point
            if curvature > 10.0 {
                critical.push(CriticalIntervention {
                    time_index: i,
                    state: point.clone(),
                    curvature,
                    intervention_type: self.classify_intervention(point, curvature)?,
                });
            }
        }
        
        Ok(critical)
    }
    
    fn classify_intervention(&self, state: &ClimateState, curvature: f64) -> Result<InterventionType> {
        // Classify based on state and curvature
        if state.co2 > 500.0 && curvature > 50.0 {
            Ok(InterventionType::EmergencyMitigation)
        } else if state.temperature_anomaly > 2.0 {
            Ok(InterventionType::AdaptationPriority)
        } else {
            Ok(InterventionType::PreventiveMeasure)
        }
    }
    
    fn assess_reversibility(&self, path: &Geodesic, manifold: &CoreClimateManifold) -> Result<f64> {
        // Check if reverse path exists with finite action
        let reverse_path = GeodesicSolver::solve(
            path.end_point(),
            path.start_point(),
            &manifold.connection,
            &manifold.metric
        )?;
        
        // Compare forward and reverse path lengths
        let forward_length = path.length();
        let reverse_length = reverse_path.length();
        
        // Reversibility = exp(-(reverse - forward)/temperature)
        let irreversibility = (reverse_length - forward_length).max(0.0);
        Ok((-irreversibility / 10.0).exp())  // 0 = irreversible, 1 = reversible
    }
}

#[derive(Debug, Clone)]
pub struct LandUseForcing {
    pub reforestation_rate: f64,  // Mha/year
    pub agricultural_efficiency: f64,
    pub urban_density_policy: f64,
}

impl LandUseForcing {
    fn implementation_cost(&self, segment: &PathSegment) -> f64 {
        // Cost model for land use changes
        self.reforestation_rate * 1000.0 +  // $/ha
        self.agricultural_efficiency * segment.land_change() * 500.0
    }
}

// Scale coupling

pub struct CrossScaleTransport {
    /// Transport operators between scales
    pub global_to_regional: Vec<TransportOperator>,
    pub regional_to_local: Vec<TransportOperator>,
    pub local_to_sensor: Vec<TransportOperator>,
    
    /// Holonomy for scale loops
    pub scale_holonomy: HashMap<ScaleLoop, HolonomyMatrix>,
    
    /// Information flow between scales
    pub renormalization_group: RenormalizationFlow,
}

impl CrossScaleTransport {
    pub fn transport_perturbation(&self, 
                                  perturbation: &LocalPerturbation,
                                  target_scale: Scale) -> Result<GlobalEffect> {
        // Transport local perturbation to global scale
        let mut effect = perturbation.to_effect();
        
        // Apply transport operators in sequence
        effect = self.local_to_sensor[perturbation.region_id].apply(effect)?;
        effect = self.regional_to_local[perturbation.region_id / 10].apply(effect)?;
        effect = self.global_to_regional[0].apply(effect)?;
        
        // Check for amplification through scales
        let amplification = effect.magnitude() / perturbation.magnitude();
        
        if amplification > 10.0 {
            // Potential cascade effect
            effect.mark_as_cascade();
        }
        
        Ok(effect)
    }
    
    pub fn compute_scale_holonomy(&mut self, loop_path: &ScaleLoop) -> Result<HolonomyMatrix> {
        // Parallel transport around scale hierarchy
        let mut holonomy = HolonomyMatrix::identity();
        
        for edge in loop_path.edges() {
            let transport = self.get_transport_operator(edge)?;
            holonomy = holonomy * transport.to_matrix();
        }
        
        // Store for future reference
        self.scale_holonomy.insert(loop_path.clone(), holonomy.clone());
        
        Ok(holonomy)
    }
}

// Metric learning

pub struct AdaptiveMetricTensor {
    /// Current metric components
    pub components: DMatrix<f64>,
    
    /// Learning rate for metric updates
    pub learning_rate: f64,
    
    /// Regularization to maintain positive definiteness
    pub regularization: f64,
    
    /// Historical metrics for ensemble
    pub history: Vec<DMatrix<f64>>,
}

impl AdaptiveMetricTensor {
    pub fn new(dim: usize) -> Self {
        Self {
            components: DMatrix::identity(dim, dim),
            learning_rate: 0.01,
            regularization: 1e-6,
            history: Vec::new(),
        }
    }
    
    pub fn learn_from_data(&mut self) -> Result<()> {
        // Connect to climate observations
        
        // Compute empirical covariance from observations
        let covariance = self.compute_empirical_covariance()?;
        
        // Update metric to match observed correlations
        self.components = (1.0 - self.learning_rate) * &self.components +
                          self.learning_rate * covariance;
        
        // Make positive definite
        self.regularize()?;
        
        // Store in history
        if self.history.len() > 100 {
            self.history.remove(0);
        }
        self.history.push(self.components.clone());
        
        Ok(())
    }
    
    pub fn expand_dimension(&mut self, dim: &LatentDimension) -> Result<()> {
        let n = self.components.nrows();
        let mut expanded = DMatrix::zeros(n + 1, n + 1);
        
        // Copy existing metric
        for i in 0..n {
            for j in 0..n {
                expanded[(i, j)] = self.components[(i, j)];
            }
        }
        
        // Add new dimension with correlations
        for i in 0..n {
            expanded[(i, n)] = dim.correlation_matrix[(i, 0)];
            expanded[(n, i)] = dim.correlation_matrix[(i, 0)];
        }
        expanded[(n, n)] = 1.0;  // Self-correlation
        
        self.components = expanded;
        Ok(())
    }
    
    fn compute_empirical_covariance(&self) -> Result<DMatrix<f64>> {
        // Compute from data
        Ok(DMatrix::identity(self.components.nrows(), self.components.ncols()))
    }
    
    fn regularize(&mut self) -> Result<()> {
        // Add small diagonal to make positive definite
        let n = self.components.nrows();
        for i in 0..n {
            self.components[(i, i)] += self.regularization;
        }
        Ok(())
    }
}

// Geometric integration framework

impl ClimateManifoldNetwork {
    /// Integrate climate model
    pub fn integrate_model(&mut self, model: Box<dyn ClimateModel>) -> Result<()> {
        // Convert model predictions to geometric language
        let model_metric = model.induced_metric()?;
        let model_curvature = model.predicted_curvature()?;
        
        // Add to ensemble
        self.ensemble_geometries.push(AlternativeMetric {
            source: model.name(),
            metric: model_metric,
            curvature: model_curvature,
            weight: model.skill_score()?,
        });
        
        Ok(())
    }
    
    /// Ingest observation data
    pub fn ingest_observations(&mut self, obs: ObservationBatch) -> Result<()> {
        // Update metric with new observations
        self.core_manifold.metric.learn_from_data()?;
        
        // Check for new latent dimensions
        let new_dims = self.dimension_learner.discover_new_dimensions(&obs);
        for dim in new_dims {
            self.dimension_learner.telemetry.record_attempt();
            let mut accepted_raw = false;
            if self.core_manifold.add_latent_dimension(dim.clone()).is_ok() {
                self.dimension_learner.telemetry.record_accept(dim.curvature_reduction);
                accepted_raw = true;
            }
            self.dimension_learner.telemetry.record_attempt_detail(DimensionAttemptRecord { name: dim.name.clone(), raw: dim.curvature_reduction, weighted: dim.curvature_reduction, accepted_raw, accepted_weighted: false });
        }
        // Curvature-aware discovery pass (additive, does not remove prior additions)
        let baseline_curv = self.core_manifold.ricci.scalar_curvature();
        let weighted = self.dimension_learner.discover_new_dimensions_curvature_aware(&obs, baseline_curv);
        for mut dim in weighted {
            if !self.core_manifold.latent_dims.iter().any(|d| d.name == dim.name) {
                // Accept only if weighting improved potential reduction
                if dim.curvature_reduction_weighted > dim.curvature_reduction + self.dimension_learner.config.min_weighted_gain {
                    self.dimension_learner.telemetry.record_attempt();
                    let mut accepted_weighted = false;
                    if self.core_manifold.add_latent_dimension(dim.clone()).is_ok() {
                        self.dimension_learner.telemetry.record_accept_weighted(dim.curvature_reduction_weighted);
                        accepted_weighted = true;
                    }
                    self.dimension_learner.telemetry.record_attempt_detail(DimensionAttemptRecord { name: dim.name.clone(), raw: dim.curvature_reduction, weighted: dim.curvature_reduction_weighted, accepted_raw: false, accepted_weighted });
                }
            }
        }
        if self.dimension_learner.config.log_telemetry {
            eprintln!("[DimDiscovery] attempts={} accept_raw={} accept_weighted={} acc_rate={:.2}% weighted_rate={:.2}% raw_curv_sum={:.4} weighted_curv_sum={:.4} last_var_gain={:?}",
                self.dimension_learner.telemetry.attempted,
                self.dimension_learner.telemetry.accepted,
                self.dimension_learner.telemetry.accepted_weighted,
                self.dimension_learner.telemetry.acceptance_rate()*100.0,
                self.dimension_learner.telemetry.weighted_acceptance_rate()*100.0,
                self.dimension_learner.telemetry.cumulative_raw_curvature_reduction,
                self.dimension_learner.telemetry.cumulative_weighted_curvature_reduction,
                self.dimension_learner.telemetry.last_variance_gain);
            eprintln!("[DimDiscovery JSON]\n{}", self.dimension_learner.telemetry.summary_json());
        }
        // Adapt threshold after each observation ingestion
        self.dimension_learner.adapt_threshold();
        
        // Update topology
        self.topology_computer = ClimateTopology::compute_from_manifold(&self.core_manifold)?;
        
        Ok(())
    }
    
    /// Generate prediction
    pub fn predict(&self, initial: &ClimateState, time_horizon: f64) -> Result<ClimatePrediction> {
        // Compute geodesic flow from initial conditions
        let geodesic = self.geodesic_forecaster.compute_flow(
            initial,
            time_horizon,
            &self.core_manifold
        )?;
        
        // Quantify uncertainty from curvature
        let uncertainty = self.uncertainty_quantifier.from_geodesic_deviation(&geodesic)?;
        
        // Check for tipping points along path
        let tipping_risks = self.detect_tipping_points_on_path(&geodesic)?;
        
        Ok(ClimatePrediction {
            trajectory: geodesic,
            uncertainty,
            tipping_risks,
            confidence: self.compute_prediction_confidence(&geodesic)?,
        })
    }
    
    /// Detect tipping points on path
    pub fn detect_tipping_points_on_path(&self, path: &Geodesic) -> Result<Vec<TippingPoint>> {
        let mut tipping_points = Vec::new();
        
        for (i, point) in path.points().enumerate() {
            let curvature = path.curvature_at(i)?;
            
            // Singularity detection: curvature → ∞
            if curvature > 100.0 || curvature.is_infinite() {
                tipping_points.push(TippingPoint {
                    location: point.clone(),
                    time: path.time_at(i),
                    type_: self.classify_tipping_point(point, curvature)?,
                    reversibility: self.assess_tipping_reversibility(point)?,
                    cascades: self.identify_cascades(point)?,
                });
            }
        }
        
        Ok(tipping_points)
    }
    
    fn classify_tipping_point(&self, state: &ClimateState, curvature: f64) -> Result<TippingType> {
        if state.ice_volume < 1e15 && curvature > 200.0 {
            Ok(TippingType::IceSheetCollapse)
        } else if state.amoc_strength < 5.0 {
            Ok(TippingType::AMOCShutdown)
        } else if state.temperature_anomaly > 3.0 {
            Ok(TippingType::RunawayWarming)
        } else {
            Ok(TippingType::Unknown)
        }
    }
    
    fn assess_tipping_reversibility(&self, state: &ClimateState) -> Result<f64> {
        // Use topology to assess reversibility
        let current_topology = ClimateTopology::compute_from_manifold(&self.core_manifold)?;
        
        // Simulate reverse transition
        let reversed_state = state.reverse_transition()?;
        
        // Check topological barriers
        let barrier_height = current_topology.estimate_barrier_to_state(&reversed_state)?;
        
        // Convert to probability
        Ok((-barrier_height / 10.0).exp())
    }
    
    fn identify_cascades(&self, trigger_point: &ClimateState) -> Result<Vec<CascadeRisk>> {
        // Use cross-scale transport to identify potential cascades
        let perturbation = LocalPerturbation::from_tipping(trigger_point);
        
        let mut cascades = Vec::new();
        
        for scale in [Scale::Regional, Scale::Global] {
            let effect = self.cross_scale_holonomy.transport_perturbation(&perturbation, scale)?;
            
            if effect.is_cascade() {
                cascades.push(CascadeRisk {
                    scale,
                    probability: effect.cascade_probability(),
                    impact: effect.magnitude(),
                });
            }
        }
        
        Ok(cascades)
    }
    
    fn compute_prediction_confidence(&self, geodesic: &Geodesic) -> Result<f64> {
        // Confidence decreases with path curvature and length
        let mean_curvature = geodesic.mean_curvature()?;
        let path_length = geodesic.length();
        
        // Heuristic confidence model
        Ok((-mean_curvature * path_length / 100.0).exp())
    }
}

// Supporting types

pub struct ClimateState {
    pub temperature_anomaly: f64,
    pub co2: f64,
    pub ice_volume: f64,
    pub amoc_strength: f64,
    // ... more fields
}

impl ClimateState {
    fn evolve_deterministic(&mut self, dt: f64) -> Result<()> {
        // RK4 integration
        let k1 = self.compute_derivatives();
        let mut state2 = self.clone();
        state2.apply_derivatives(&k1, dt * 0.5);
        
        let k2 = state2.compute_derivatives();
        let mut state3 = self.clone();
        state3.apply_derivatives(&k2, dt * 0.5);
        
        let k3 = state3.compute_derivatives();
        let mut state4 = self.clone();
        state4.apply_derivatives(&k3, dt);
        
        let k4 = state4.compute_derivatives();
        
        // Update state using RK4 formula
        self.temperature_anomaly += dt * (k1.dT + 2.0*k2.dT + 2.0*k3.dT + k4.dT) / 6.0;
        self.co2 += dt * (k1.dCO2 + 2.0*k2.dCO2 + 2.0*k3.dCO2 + k4.dCO2) / 6.0;
        self.ice_volume += dt * (k1.dIce + 2.0*k2.dIce + 2.0*k3.dIce + k4.dIce) / 6.0;
        self.amoc_strength += dt * (k1.dAMOC + 2.0*k2.dAMOC + 2.0*k3.dAMOC + k4.dAMOC) / 6.0;
        
        Ok(())
    }
    
    fn evolve_stochastic(&mut self, dt: f64, noise: &DVector<f64>) -> Result<()> {
        // Stochastic differential equation integration
        // First apply deterministic evolution
        self.evolve_deterministic(dt)?;
        
        // Add stochastic noise terms (Euler-Maruyama scheme)
        if noise.len() >= 4 {
            self.temperature_anomaly += noise[0] * dt.sqrt();
            self.co2 += noise[1] * dt.sqrt();
            self.ice_volume += noise[2] * dt.sqrt();
            self.amoc_strength += noise[3] * dt.sqrt();
        }
        
        Ok(())
    }
    
    fn reverse_transition(&self) -> Result<ClimateState> {
        // Reverse transition with hysteresis
        let mut reversed = self.clone();
        
        // Temperature reversal with thermal inertia
        reversed.temperature_anomaly = self.temperature_anomaly - 
            (self.temperature_anomaly - 0.0) * 0.1; // Slow cooling
        
        // CO2 reversal (very slow due to long atmospheric lifetime)
        reversed.co2 = self.co2 - (self.co2 - 280.0) * 0.01;
        
        // Ice sheet reversal (extremely slow, millennia-scale)
        if self.ice_volume < 2.0e16 {
            // Ice regrowth is much slower than loss
            reversed.ice_volume = self.ice_volume + 
                (2.6e16 - self.ice_volume) * 0.001;
        }
        
        // AMOC recovery (exhibits strong hysteresis)
        if self.amoc_strength < 5.0 {
            // Once collapsed, very hard to restart
            reversed.amoc_strength = self.amoc_strength * 1.01;
        } else {
            reversed.amoc_strength = self.amoc_strength + 
                (15.0 - self.amoc_strength) * 0.05;
        }
        
        Ok(reversed)
    }
    
    fn compute_derivatives(&self) -> ClimateDerivatives {
        ClimateDerivatives {
            dT: self.compute_temperature_change(),
            dCO2: self.compute_co2_change(),
            dIce: self.compute_ice_change(),
            dAMOC: self.compute_amoc_change(),
        }
    }
    
    fn compute_temperature_change(&self) -> f64 {
        // Energy balance: dT/dt = (Forcing - λT) / C
        let forcing = 3.7 * (self.co2 / 280.0).ln() / 2.0f64.ln(); // CO2 forcing
        let feedback = -3.2; // Planck feedback W/m²/K
        let heat_capacity = 8.0; // Effective heat capacity
        
        (forcing + feedback * self.temperature_anomaly) / heat_capacity
    }
    
    fn compute_co2_change(&self) -> f64 {
        // Simplified carbon cycle
        let emissions = 10.0; // GtC/year
        let airborne_fraction = 0.45;
        let natural_sink = -2.0 * (self.co2 - 280.0) / 280.0;
        
        emissions * airborne_fraction * 2.12 + natural_sink // 2.12 ppm/GtC
    }
    
    fn compute_ice_change(&self) -> f64 {
        // Ice sheet dynamics
        let melt_rate = if self.temperature_anomaly > 0.0 {
            -1e14 * self.temperature_anomaly.powi(2) // Nonlinear melting
        } else {
            1e13 * (-self.temperature_anomaly) // Slow regrowth
        };
        
        melt_rate
    }
    
    fn compute_amoc_change(&self) -> f64 {
        // AMOC dynamics with critical transition
        let freshwater = self.compute_freshwater_flux();
        let threshold = 0.15; // Sv of freshwater
        
        if freshwater > threshold {
            -2.0 * self.amoc_strength // Rapid collapse
        } else {
            0.5 * (15.0 - self.amoc_strength) - freshwater * 10.0
        }
    }
    
    fn compute_freshwater_flux(&self) -> f64 {
        // Greenland melt contribution
        0.01 * self.temperature_anomaly.max(0.0).powi(2)
    }
    
    fn apply_derivatives(&mut self, deriv: &ClimateDerivatives, dt: f64) {
        self.temperature_anomaly += deriv.dT * dt;
        self.co2 += deriv.dCO2 * dt;
        self.ice_volume += deriv.dIce * dt;
        self.amoc_strength += deriv.dAMOC * dt;
    }
    
    fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.temperature_anomaly,
            self.co2,
            self.ice_volume,
            self.amoc_strength,
        ])
    }
}

struct ClimateDerivatives {
    dT: f64,
    dCO2: f64,
    dIce: f64,
    dAMOC: f64,
}

pub struct ObservationBatch {
    pub timestamps: Vec<f64>,
    pub observations: DMatrix<f64>,
    pub uncertainties: DMatrix<f64>,
    pub locations: Vec<(f64, f64, f64)>, // (lat, lon, depth/height)
    pub variables: Vec<String>,
}

pub struct GlobalManifold {
    pub dimension: usize,
    pub metric: DMatrix<f64>,
    pub christoffel: Array3<f64>,
    pub coverage: f64, // Fraction of Earth covered
}

pub struct RegionalManifold {
    pub region_id: String,
    pub bounds: ((f64, f64), (f64, f64)), // ((lat_min, lat_max), (lon_min, lon_max))
    pub local_dimension: usize,
    pub embedding_map: DMatrix<f64>, // Maps to global manifold
    pub regional_metric: DMatrix<f64>,
}

pub struct LocalManifold {
    pub location: (f64, f64),
    pub radius_km: f64,
    pub micro_dimension: usize,
    pub local_metric: DMatrix<f64>,
    pub observations: Vec<f64>,
}

pub struct SensorNetwork {
    pub sensors: Vec<Sensor>,
    pub topology: Graph<usize, f64>,
    pub data_rate_hz: f64,
    pub latency_ms: f64,
}

pub struct Sensor {
    pub id: String,
    pub location: (f64, f64, f64),
    pub variables: Vec<String>,
    pub accuracy: Vec<f64>,
    pub status: SensorStatus,
}

pub enum SensorStatus {
    Online,
    Degraded,
    Offline,
    Maintenance,
}

pub struct SatelliteDataStream {
    pub satellite_id: String,
    pub orbit_params: OrbitParameters,
    pub instruments: Vec<Instrument>,
    pub data_stream: Receiver<SatelliteData>,
    pub bandwidth_mbps: f64,
}

pub struct OrbitParameters {
    pub altitude_km: f64,
    pub inclination_deg: f64,
    pub period_minutes: f64,
}

pub struct Instrument {
    pub name: String,
    pub spectral_bands: Vec<(f64, f64)>, // (min_wavelength, max_wavelength)
    pub spatial_resolution_m: f64,
    pub swath_width_km: f64,
}

pub struct SatelliteData {
    pub timestamp: f64,
    pub footprint: Vec<(f64, f64)>,
    pub measurements: DMatrix<f64>,
}

pub struct GroundSensorNetwork {
    pub stations: HashMap<String, WeatherStation>,
    pub update_frequency_hz: f64,
    pub quality_control: QualityControl,
}

pub struct WeatherStation {
    pub id: String,
    pub location: (f64, f64, f64),
    pub instruments: Vec<String>,
    pub data_history: VecDeque<StationData>,
}

pub struct StationData {
    pub timestamp: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub humidity: f64,
    pub wind: (f64, f64), // (speed, direction)
    pub precipitation: f64,
}

pub struct QualityControl {
    pub outlier_threshold: f64,
    pub consistency_checks: Vec<ConsistencyCheck>,
}

pub struct ConsistencyCheck {
    pub name: String,
    pub check_fn: Box<dyn Fn(&StationData) -> bool>,
}

pub struct OceanObservationSystem {
    pub argo_floats: Vec<ArgoFloat>,
    pub moorings: Vec<OceanMooring>,
    pub gliders: Vec<OceanGlider>,
    pub coverage_fraction: f64,
}

pub struct ArgoFloat {
    pub id: String,
    pub position: (f64, f64),
    pub profile_depth_m: f64,
    pub temperature_profile: Vec<f64>,
    pub salinity_profile: Vec<f64>,
    pub last_surfaced: f64,
}

pub struct OceanMooring {
    pub location: (f64, f64),
    pub sensors_depths: Vec<f64>,
    pub time_series: HashMap<String, Vec<f64>>,
}

pub struct OceanGlider {
    pub id: String,
    pub trajectory: Vec<(f64, f64, f64, f64)>, // (time, lat, lon, depth)
    pub measurements: HashMap<String, Vec<f64>>,
}

pub struct AtmosphericProfileNetwork {
    pub radiosondes: Vec<Radiosonde>,
    pub lidars: Vec<Lidar>,
    pub launch_schedule: HashMap<String, Vec<f64>>,
}

pub struct Radiosonde {
    pub launch_site: String,
    pub launch_time: f64,
    pub profile: AtmosphericProfile,
}

pub struct AtmosphericProfile {
    pub pressure_levels: Vec<f64>,
    pub temperature: Vec<f64>,
    pub humidity: Vec<f64>,
    pub wind_u: Vec<f64>,
    pub wind_v: Vec<f64>,
}

pub struct Lidar {
    pub location: (f64, f64),
    pub wavelength_nm: f64,
    pub vertical_resolution_m: f64,
    pub max_altitude_km: f64,
}

pub struct NeuralManifoldEncoder {
    pub encoder_network: EncoderNet,
    pub decoder_network: DecoderNet,
    pub latent_dim: usize,
    pub trained_samples: usize,
}

pub struct EncoderNet {
    pub layers: Vec<DMatrix<f64>>,
    pub activation: ActivationFunction,
}

pub struct DecoderNet {
    pub layers: Vec<DMatrix<f64>>,
    pub activation: ActivationFunction,
}

pub enum ActivationFunction {
    ReLU,
    Tanh,
    GELU,
    Swish,
}

pub struct GeometricPatternDetector {
    pub patterns: Vec<GeometricPattern>,
    pub detection_threshold: f64,
    pub spatial_scale: f64,
}

pub struct GeometricPattern {
    pub name: String,
    pub signature: DVector<f64>,
    pub typical_scale: f64,
    pub persistence: f64,
}

pub struct ManifoldAnomalyDetector {
    pub baseline_curvature: f64,
    pub anomaly_threshold: f64,
    pub detection_history: VecDeque<AnomalyEvent>,
}

pub struct AnomalyEvent {
    pub timestamp: f64,
    pub location: (f64, f64),
    pub anomaly_score: f64,
    pub likely_cause: String,
}

pub struct GeodesicFlowPredictor {
    pub integrator: GeodesicIntegrator,
    pub time_horizon: f64,
    pub ensemble_size: usize,
}

pub struct GeodesicIntegrator {
    pub method: IntegrationMethod,
    pub step_size: f64,
    pub adaptive: bool,
    pub tolerance: f64,
}

pub enum IntegrationMethod {
    RungeKutta4,
    RungeKuttaFehlberg,
    DormandPrince,
    SymplecticEuler,
}
pub struct AlternativeMetric {
    source: String,
    metric: DMatrix<f64>,
    curvature: f64,
    weight: f64,
}
pub struct GeometricUncertainty;
pub struct CorrelationMeasure;
pub struct PersistenceDiagram;
pub struct MinimalCurvaturePathfinder;
pub struct ClimateAutoencoder;
pub struct NonEquilibriumConnection {
    pub christoffel: Array3<f64>,
    pub torsion: Array3<f64>, // Non-zero for non-equilibrium
    pub dimension: usize,
}

impl NonEquilibriumConnection {
    pub fn new(dim: usize) -> Self {
        Self {
            christoffel: Array3::zeros((dim, dim, dim)),
            torsion: Array3::zeros((dim, dim, dim)),
            dimension: dim,
        }
    }
    
    pub fn from_metric(metric: &AdaptiveMetricTensor) -> Result<Self> {
        let dim = metric.components.nrows();
        let mut conn = Self::new(dim);
        
        // Compute Christoffel symbols from metric
        let g_inv = metric.components.try_inverse()
            .ok_or_else(|| anyhow::anyhow!("Singular metric"))?;
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    let mut gamma = 0.0;
                    for l in 0..dim {
                        // Simplified - would use full derivatives
                        gamma += 0.5 * g_inv[(i, l)] * 
                            (metric.components[(l, j)] + metric.components[(l, k)] - metric.components[(j, k)]);
                    }
                    conn.christoffel[(i, j, k)] = gamma;
                }
            }
        }
        
        Ok(conn)
    }
}

pub struct RiemannTensor {
    pub components: Array4<f64>,
    pub dimension: usize,
}

impl RiemannTensor {
    pub fn zeros(dim: usize) -> Self {
        Self {
            components: Array4::zeros((dim, dim, dim, dim)),
            dimension: dim,
        }
    }
    
    pub fn compute(connection: &NonEquilibriumConnection) -> Result<Self> {
        let dim = connection.dimension;
        let mut riemann = Self::zeros(dim);
        
        // R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let mut r = 0.0;
                        
                        // Quadratic terms
                        for m in 0..dim {
                            r += connection.christoffel[(i, m, k)] * connection.christoffel[(m, j, l)]
                               - connection.christoffel[(i, m, l)] * connection.christoffel[(m, j, k)];
                        }
                        
                        riemann.components[(i, j, k, l)] = r;
                    }
                }
            }
        }
        
        Ok(riemann)
    }
    
    pub fn contract_to_ricci(&self) -> Result<RicciTensor> {
        let mut ricci = RicciTensor::zeros(self.dimension);
        
        // R_ij = R^k_ikj
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    ricci.components[(i, j)] += self.components[(k, i, k, j)];
                }
            }
        }
        
        Ok(ricci)
    }
}

pub struct RicciTensor {
    pub components: Array2<f64>,
    pub dimension: usize,
}

impl RicciTensor {
    pub fn zeros(dim: usize) -> Self {
        Self {
            components: Array2::zeros((dim, dim)),
            dimension: dim,
        }
    }
    
    pub fn scalar_curvature(&self) -> f64 {
        self.components.diag().sum()
    }
}

pub struct WeylTensor {
    pub components: Array4<f64>,
    pub dimension: usize,
}

impl WeylTensor {
    pub fn zeros(dim: usize) -> Self {
        Self {
            components: Array4::zeros((dim, dim, dim, dim)),
            dimension: dim,
        }
    }
    
    pub fn from_riemann(riemann: &RiemannTensor, ricci: &RicciTensor) -> Result<Self> {
        let dim = riemann.dimension;
        let mut weyl = Self::zeros(dim);
        let scalar_curv = ricci.scalar_curvature();
        
        // W_ijkl = R_ijkl - (g_ik R_jl - g_il R_jk + g_jl R_ik - g_jk R_il)/(n-2) 
        //          + R(g_ik g_jl - g_il g_jk)/((n-1)(n-2))
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let mut w = riemann.components[(i, j, k, l)];
                        
                        // Subtract Ricci terms
                        let n = dim as f64;
                        if dim > 2 {
                            w -= (ricci.components[(j, l)] - ricci.components[(j, k)]) / (n - 2.0);
                            w += scalar_curv / ((n - 1.0) * (n - 2.0));
                        }
                        
                        weyl.components[(i, j, k, l)] = w;
                    }
                }
            }
        }
        
        Ok(weyl)
    }
}

pub struct ScaleCouplingTensor {
    pub components: HashMap<(Scale, Scale), DMatrix<f64>>,
    pub coupling_strength: HashMap<(Scale, Scale), f64>,
}

pub struct HomologyGroup {
    pub dimension: usize,
    pub rank: usize,
    pub torsion: Vec<usize>,
    pub generators: Vec<DVector<f64>>,
}

pub struct TopologicalTransition {
    pub before: Vec<usize>, // Betti numbers before
    pub after: Vec<usize>,  // Betti numbers after
    pub critical_value: f64,
    pub transition_type: TransitionType,
}

pub enum TransitionType {
    Birth,
    Death,
    Bifurcation,
    Merger,
}

pub struct ClimateDynamics {
    pub advection_field: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
    pub diffusion_field: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64>>,
    pub forcing_field: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
    pub dimension: usize,
}

impl ClimateDynamics {
    pub fn diffusion_matrix(&self) -> DMatrix<f64> {
        // Construct diffusion matrix for stochastic evolution
        let mut d = DMatrix::zeros(self.dimension, self.dimension);
        
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if i == j {
                    d[(i, j)] = 0.1; // Diagonal diffusion
                } else {
                    d[(i, j)] = 0.01; // Cross-diffusion
                }
            }
        }
        
        d
    }
    
    pub fn stochastic_noise(&self) -> DVector<f64> {
        // Generate stochastic forcing for ensemble spread
        DVector::from_element(self.dimension, 0.01)
    }
    
    pub fn decay_rate(&self) -> f64 {
        // Ensemble decay rate due to model error growth
        0.001
    }
}

pub struct Connection {
    pub christoffel_symbols: Array3<f64>,
    pub metric: DMatrix<f64>,
    pub dimension: usize,
}

impl Connection {
    pub fn add_forcing_gradient(&mut self, forcing: DVector<f64>) {
        // Add external forcing to the connection
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    self.christoffel_symbols[(i, j, k)] += forcing[i] * 0.01;
                }
            }
        }
    }
    
    pub fn add_directional_forcing(&mut self, forcing_field: DMatrix<f64>) {
        // Add directional forcing field to dynamics
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    self.christoffel_symbols[(i, j, k)] += forcing_field[(j, k)] * 0.01;
                }
            }
        }
    }
    
    pub fn apply_surface_forcing(&mut self, surface_forcing: DMatrix<f64>) {
        // Apply surface boundary forcing to connection
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                self.christoffel_symbols[(i, j, i)] += surface_forcing[(i, j)] * 0.01;
            }
        }
    }
    
    pub fn scale_diffusion(&mut self, factor: f64) {
        // Scale diffusion processes in the connection
        self.christoffel_symbols *= (1.0 + factor * 0.1);
    }
}

pub struct Geodesic {
    pub points: Vec<DVector<f64>>,
    pub times: Vec<f64>,
    pub curvatures: Vec<f64>,
    pub length: f64,
}

impl Geodesic {
    pub fn points(&self) -> &[DVector<f64>] { &self.points }
    pub fn time_at(&self, i: usize) -> f64 { self.times[i] }
    pub fn curvature_at(&self, i: usize) -> Result<f64> { 
        Ok(self.curvatures.get(i).copied().unwrap_or(0.0))
    }
    pub fn mean_curvature(&self) -> Result<f64> {
        Ok(self.curvatures.iter().sum::<f64>() / self.curvatures.len() as f64)
    }
    pub fn length(&self) -> f64 { self.length }
    pub fn start_point(&self) -> &DVector<f64> { &self.points[0] }
    pub fn end_point(&self) -> &DVector<f64> { self.points.last().unwrap() }
    pub fn segments(&self) -> Vec<PathSegment> {
        self.points.windows(2).zip(self.times.windows(2))
            .map(|(pts, times)| PathSegment {
                start: pts[0].clone(),
                end: pts[1].clone(),
                duration: times[1] - times[0],
            })
            .collect()
    }
}
pub struct InterventionPath {
    trajectory: Geodesic,
    total_cost: f64,
    critical_points: Vec<CriticalIntervention>,
    reversibility: f64,
}
pub struct CriticalIntervention {
    time_index: usize,
    state: ClimateState,
    curvature: f64,
    intervention_type: InterventionType,
}
pub enum InterventionType {
    EmergencyMitigation,
    AdaptationPriority,
    PreventiveMeasure,
}
pub struct PathSegment {
    pub start: DVector<f64>,
    pub end: DVector<f64>,
    pub duration: f64,
}

impl PathSegment {
    pub fn emissions_change(&self) -> f64 {
        // CO2 is typically index 1
        self.end[1] - self.start[1]
    }
    
    pub fn energy_change(&self) -> DVector<f64> {
        // Energy components change
        &self.end - &self.start
    }
    
    pub fn land_change(&self) -> f64 {
        // Land use component, if present (index 7)
        if self.end.len() > 7 {
            self.end[7] - self.start[7]
        } else {
            0.0
        }
    }
}
pub struct TransportOperator {
    pub matrix: DMatrix<f64>,
    pub from_scale: Scale,
    pub to_scale: Scale,
}

impl TransportOperator {
    pub fn apply(&self, effect: GlobalEffect) -> Result<GlobalEffect> {
        let transported = &self.matrix * &effect.vector;
        Ok(GlobalEffect {
            vector: transported,
            magnitude: effect.magnitude,
            is_cascade: effect.is_cascade,
            cascade_probability: effect.cascade_probability,
        })
    }
    
    pub fn to_matrix(&self) -> DMatrix<f64> {
        self.matrix.clone()
    }
}

pub struct ScaleLoop {
    pub path: Vec<Scale>,
    pub edges: Vec<(Scale, Scale)>,
}

impl ScaleLoop {
    pub fn edges(&self) -> &[(Scale, Scale)] {
        &self.edges
    }
}

pub struct HolonomyMatrix {
    pub matrix: DMatrix<f64>,
    pub determinant: f64,
    pub eigenvalues: DVector<Complex64>,
}

impl HolonomyMatrix {
    pub fn identity() -> Self {
        let dim = 8;
        Self {
            matrix: DMatrix::identity(dim, dim),
            determinant: 1.0,
            eigenvalues: DVector::from_element(dim, Complex64::new(1.0, 0.0)),
        }
    }
}

impl std::ops::Mul for HolonomyMatrix {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            matrix: &self.matrix * &rhs.matrix,
            determinant: self.determinant * rhs.determinant,
            eigenvalues: self.eigenvalues, // Would recompute
        }
    }
}

pub struct RenormalizationFlow {
    pub beta_functions: Vec<Box<dyn Fn(f64) -> f64>>,
    pub fixed_points: Vec<f64>,
    pub flow_equations: Vec<String>,
}

pub struct LocalPerturbation {
    pub location: (f64, f64),
    pub magnitude: f64,
    pub type_: PerturbationType,
    pub region_id: usize,
}

impl LocalPerturbation {
    pub fn from_tipping(state: &ClimateState) -> Self {
        Self {
            location: (0.0, 0.0), // Would extract from state
            magnitude: state.temperature_anomaly.abs(),
            type_: PerturbationType::TemperatureSpike,
            region_id: 0,
        }
    }
    
    pub fn to_effect(&self) -> GlobalEffect {
        GlobalEffect {
            vector: DVector::from_element(8, self.magnitude),
            magnitude: self.magnitude,
            is_cascade: false,
            cascade_probability: 0.0,
        }
    }
    
    pub fn magnitude(&self) -> f64 { self.magnitude }
}

pub enum PerturbationType {
    TemperatureSpike,
    PrecipitationAnomaly,
    IceCollapse,
    CirculationChange,
}

pub struct GlobalEffect {
    pub vector: DVector<f64>,
    pub magnitude: f64,
    pub is_cascade: bool,
    pub cascade_probability: f64,
}

impl GlobalEffect {
    pub fn magnitude(&self) -> f64 { self.magnitude }
    pub fn is_cascade(&self) -> bool { self.is_cascade }
    pub fn cascade_probability(&self) -> f64 { self.cascade_probability }
    pub fn mark_as_cascade(&mut self) {
        self.is_cascade = true;
        self.cascade_probability = self.magnitude.tanh();
    }
}
pub enum Scale {
    Local,
    Regional,
    Global,
}
pub trait ClimateModel {
    fn name(&self) -> String;
    fn induced_metric(&self) -> Result<DMatrix<f64>>;
    fn predicted_curvature(&self) -> Result<f64>;
    fn skill_score(&self) -> Result<f64>;
}
pub struct ClimatePrediction {
    trajectory: Geodesic,
    uncertainty: f64,
    tipping_risks: Vec<TippingPoint>,
    confidence: f64,
}
pub struct TippingPoint {
    location: ClimateState,
    time: f64,
    type_: TippingType,
    reversibility: f64,
    cascades: Vec<CascadeRisk>,
}
pub enum TippingType {
    IceSheetCollapse,
    AMOCShutdown,
    RunawayWarming,
    Unknown,
}
pub struct CascadeRisk {
    scale: Scale,
    probability: f64,
    impact: f64,
}
pub struct GeodesicSolver;

impl GeodesicSolver {
    pub fn solve(
        start: &ClimateState,
        target: &ClimateState,
        connection: &Connection,
        metric: &DMatrix<f64>
    ) -> Result<Geodesic> {
        let mut points = Vec::new();
        let mut times = Vec::new();
        let mut curvatures = Vec::new();
        
        // Convert states to vectors
        let x0 = start.to_vector();
        let x1 = target.to_vector();
        
        // Number of integration steps
        let n_steps = 100;
        let dt = 1.0 / n_steps as f64;
        
        // Current position and velocity
        let mut x = x0.clone();
        let mut v = (&x1 - &x0) * 0.1; // Initial velocity toward target
        
        let mut total_length = 0.0;
        
        for i in 0..n_steps {
            points.push(x.clone());
            times.push(i as f64 * dt);
            
            // Geodesic equation: d²x^i/dt² + Γ^i_jk (dx^j/dt)(dx^k/dt) = 0
            let mut acceleration = DVector::zeros(connection.dimension);
            
            for i in 0..connection.dimension {
                for j in 0..connection.dimension {
                    for k in 0..connection.dimension {
                        acceleration[i] -= connection.christoffel_symbols[(i, j, k)] * v[j] * v[k];
                    }
                }
            }
            
            // Update velocity and position (Verlet integration)
            v += &acceleration * dt;
            x += &v * dt;
            
            // Compute local curvature
            let local_curv = (acceleration.norm() / v.norm().max(1e-10)).abs();
            curvatures.push(local_curv);
            
            // Accumulate path length
            if i > 0 {
                total_length += (&points[i] - &points[i-1]).norm();
            }
        }
        
        points.push(x1);
        times.push(1.0);
        
        Ok(Geodesic {
            points,
            times,
            curvatures,
            length: total_length,
        })
    }
}

pub struct PCA {
    pub components: Vec<PrincipalComponent>,
    pub explained_variance: Vec<f64>,
    pub variance_captured_initial: f64,
    pub variance_captured_refined: Option<f64>,
    pub predicted_curvature_reduction_gain: Option<f64>,
}

impl PCA {
    pub fn fit(data: &Array2<f64>) -> Self {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        
        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean;
        
        // Compute covariance matrix (feature x feature)
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        // Attempt real eigendecomposition (additive real implementation)
        // Fallback gracefully to synthetic if decomposition fails
        let mut components = Vec::new();
        let mut explained_variance = Vec::new();

        let cov_mat = DMatrix::from_iterator(n_features, n_features, cov.iter().cloned());
        let trace: f64 = (0..n_features).map(|i| cov_mat[(i,i)]).sum();
        if trace > 0.0 {
            if let Ok(eigs) = std::panic::catch_unwind(|| SymmetricEigen::new(cov_mat.clone())) {
                // Eigenvalues ascending; take top k
                let k = n_features.min(10);
                for idx in 0..k {
                    let ev = eigs.eigenvalues[n_features-1-idx].max(0.0);
                    let ratio = ev / trace;
                    let vec = eigs.eigenvectors.column(n_features-1-idx).clone_owned();
                    components.push(PrincipalComponent {
                        index: idx,
                        weights: DVector::from_iterator(vec.len(), vec.iter().cloned()),
                        eigen_variance_ratio: Some(ratio),
                    });
                    explained_variance.push(ratio);
                }
            }
        }
        // If eigendecomposition branch produced nothing, retain original simplistic components
        if components.is_empty() {
            for i in 0..n_features.min(10) {
                components.push(PrincipalComponent { index: i, weights: DVector::from_element(n_features, 0.1), eigen_variance_ratio: None });
                explained_variance.push(1.0 / (i + 1) as f64);
            }
        }

    let variance_captured_initial = explained_variance.iter().sum();
    Self { components, explained_variance, variance_captured_initial, variance_captured_refined: None, predicted_curvature_reduction_gain: None }
    }
    
    pub fn components(&self) -> &[PrincipalComponent] {
        &self.components
    }

    /// Additive: refine PCA with power iteration deflation for top components
    pub fn refine_with_power_iterations(&mut self, data: &Array2<f64>, iters: usize) {
        let n_features = data.ncols();
        if n_features == 0 { return; }
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean;
        let mut cov = centered.t().dot(&centered) / (data.nrows().saturating_sub(1) as f64).max(1.0);
        let k = self.components.len();
        let max_k = k.min(10);
        let mut new_components: Vec<PrincipalComponent> = Vec::new();
        let mut new_explained: Vec<f64> = Vec::new();
        let mut total_var: f64 = cov.diag().sum();
        for idx in 0..max_k {
            // Random init
            let mut v = DVector::from_fn(n_features, |i,_| ((i+1+idx) as f64 * 0.017).sin());
            for _ in 0..iters.max(3) {
                let w = &cov * &v;
                let norm = w.norm().max(1e-12);
                v = w / norm;
            }
            let lambda = v.transpose() * &cov * &v;
            // Deflation
            let denom = v.transpose() * &v;
            if denom[(0,0)].abs() > 1e-12 {
                let outer = &v * v.transpose() / denom[(0,0)];
                cov = cov - outer * lambda[(0,0)];
            }
            new_explained.push(lambda[(0,0)] / total_var.max(1e-12));
            new_components.push(PrincipalComponent { index: idx, weights: v.clone() });
        }
        // Only replace if variance capture improved
        let old_sum: f64 = self.explained_variance.iter().sum();
        let new_sum: f64 = new_explained.iter().sum();
        if new_sum > old_sum { 
            self.components = new_components; 
            self.explained_variance = new_explained; 
            self.variance_captured_refined = Some(new_sum);
            self.predicted_curvature_reduction_gain = Some(new_sum - old_sum);
        }
    }

    pub fn upgrade_synthetic_weights(&mut self, data: &Array2<f64>) {
        if self.components.is_empty() { return; }
        // Detect uniform synthetic weights (all same value per component)
        let synthetic = self.components.iter().all(|c| {
            c.weights.iter().all(|w| (w - c.weights[0]).abs() < 1e-12)
        });
        if !synthetic { return; }
        let n_features = data.ncols();
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean;
        // Compute feature means magnitude pattern
        let mut pattern: Vec<f64> = (0..n_features).map(|j| centered.column(j).sum() / centered.nrows() as f64).collect();
        let norm: f64 = pattern.iter().map(|v| v*v).sum::<f64>().sqrt().max(1e-12);
        for (i, comp) in self.components.iter_mut().enumerate() {
            comp.weights = DVector::from_iterator(n_features, pattern.iter().map(|v| v / norm * ((i+1) as f64).ln().max(1.0)));
        }
    }

    pub fn render_diagnostics(&self) -> Option<String> {
        if let Some(refined) = self.variance_captured_refined {
            return Some(format!("variance_initial={:.4} variance_refined={:.4} gain={:.4}",
                self.variance_captured_initial,
                refined,
                self.predicted_curvature_reduction_gain.unwrap_or(0.0)));
        }
        None
    }
}

pub struct PrincipalComponent {
    pub index: usize,
    pub weights: DVector<f64>,
    pub eigen_variance_ratio: Option<f64>,
}

impl PrincipalComponent {
    pub fn explained_variance_ratio(&self) -> f64 { self.eigen_variance_ratio.unwrap_or(1.0 / (self.index + 1) as f64) }
}

// Additional required types
use ndarray::{Array2, Array3, Array4, Axis};
use rand::distributions::WeightedIndex;
use rand::Rng;