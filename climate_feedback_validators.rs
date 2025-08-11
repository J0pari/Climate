// Climate feedback detection using IPCC AR6 thresholds
//
// DATA SOURCE REQUIREMENTS:
// 
// 1. SEA ICE EXTENT:
//    - Source: NSIDC CHARCTIC (https://nsidc.org/arcticseaicenews/charctic-interactive-sea-ice-graph/)
//    - Instrument: SSMR/SSMI/SSMIS passive microwave
//    - Resolution: 25km x 25km polar stereographic grid
//    - Temporal: Daily since 1979-10-26
//    - Format: NetCDF4, also available as CSV
//    - Size: ~200MB/year for both poles
//    - API: ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/
//    - Preprocessing: Apply NASA Team algorithm for concentration > 15%
//    - Missing: Real-time data has 2-day lag, needs quality control
//
// 2. SURFACE ALBEDO:
//    - Source: MODIS MCD43C3 Version 6.1
//    - Instrument: Terra+Aqua MODIS
//    - Resolution: 0.05° global CMG (Climate Modeling Grid)
//    - Temporal: 16-day composite, daily since 2000-02-24
//    - Format: HDF-EOS2
//    - Size: ~50GB/year for black-sky and white-sky albedo
//    - API: https://ladsweb.modaps.eosdis.nasa.gov/
//    - License: NASA Earthdata account required (free)
//    - Preprocessing: Gap-fill using BRDF model, quality flag filtering
//    - Missing: Cloud contamination in polar regions, no data before 2000
//
// 3. SURFACE TEMPERATURE:
//    - Source: ERA5 2m temperature
//    - Provider: ECMWF Copernicus Climate Data Store
//    - Resolution: 0.25° x 0.25° global
//    - Temporal: Hourly since 1940, reanalysis updated monthly
//    - Format: GRIB2 or NetCDF4
//    - Size: ~24GB/year for hourly, ~1GB/year for daily means
//    - API: cdsapi Python client with CDS account
//    - Preprocessing: Convert from Kelvin, apply land-sea mask
//    - Missing: 5-day lag for quality-controlled data
//
// 4. CLOUD RADIATIVE EFFECT (for feedback decomposition):
//    - Source: CERES EBAF-TOA Edition 4.2
//    - Instrument: Terra/Aqua/NPP CERES
//    - Resolution: 1° x 1° global
//    - Temporal: Monthly since 2000-03
//    - Format: NetCDF4
//    - Size: ~100MB/year
//    - API: https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAF42Selection.jsp
//    - Preprocessing: Calculate CRE = all-sky - clear-sky fluxes
//    - Missing: Diurnal sampling bias, polar night interpolation

use std::collections::VecDeque;
use std::f64::consts::E;
use serde::{Serialize, Deserialize};

// Ice-albedo feedback

pub struct IceAlbedoFeedback {
    /// Historical ice extent (million km²)
    ice_extent_history: VecDeque<f64>,
    /// Historical albedo values
    albedo_history: VecDeque<f64>,
    /// Temperature history (K)
    temperature_history: VecDeque<f64>,
    /// Maximum allowed positive feedback strength
    max_feedback_strength: f64,
    /// Runaway threshold (derivative of feedback)
    runaway_threshold: f64,
}

impl IceAlbedoFeedback {
    pub fn new() -> Self {
        Self {
            ice_extent_history: VecDeque::with_capacity(100),
            albedo_history: VecDeque::with_capacity(100),
            temperature_history: VecDeque::with_capacity(100),
            max_feedback_strength: 0.5,  // W/m²/K
            runaway_threshold: 0.1,      // W/m²/K²
        }
    }
    
    pub fn validate(
        &mut self,
        ice_extent_km2: f64,
        albedo: f64,
        temperature_k: f64,
        timestep: f64,
    ) -> Result<FeedbackState, FeedbackFailure> {
        // Record current state
        self.ice_extent_history.push_back(ice_extent_km2);
        self.albedo_history.push_back(albedo);
        self.temperature_history.push_back(temperature_k);
        
        // Keep history bounded
        if self.ice_extent_history.len() > 100 {
            self.ice_extent_history.pop_front();
            self.albedo_history.pop_front();
            self.temperature_history.pop_front();
        }
        
        // Need sufficient history
        if self.ice_extent_history.len() < 10 {
            return Ok(FeedbackState::Insufficient);
        }
        
        // Calculate feedback strength: ∂α/∂T where α is albedo
        let feedback_strength = self.calculate_feedback_strength(timestep)?;
        
        // Check against maximum
        if feedback_strength > self.max_feedback_strength {
            return Err(FeedbackFailure::PositiveFeedbackExcessive {
                feedback_type: "ice-albedo".to_string(),
                strength: feedback_strength,
                threshold: self.max_feedback_strength,
            });
        }
        
        // Calculate acceleration (second derivative)
        let acceleration = self.calculate_feedback_acceleration(timestep)?;
        
        // Check for runaway
        if acceleration > self.runaway_threshold {
            return Err(FeedbackFailure::RunawayDetected {
                feedback_type: "ice-albedo".to_string(),
                acceleration,
                ice_extent_remaining: ice_extent_km2,
            });
        }
        
        // Classify state
        let state = if feedback_strength < 0.1 {
            FeedbackState::Stable
        } else if feedback_strength < 0.3 {
            FeedbackState::Active { strength: feedback_strength }
        } else {
            FeedbackState::Critical { 
                strength: feedback_strength,
                time_to_runaway: self.estimate_time_to_runaway(feedback_strength, acceleration),
            }
        };
        
        Ok(state)
    }
    
    fn calculate_feedback_strength(&self, timestep: f64) -> Result<f64, FeedbackFailure> {
        let n = self.temperature_history.len();
        if n < 2 {
            return Ok(0.0);
        }
        
        // Calculate ∂α/∂T using linear regression
        let temps: Vec<f64> = self.temperature_history.iter().copied().collect();
        let albedos: Vec<f64> = self.albedo_history.iter().copied().collect();
        
        // Simple finite difference for now
        let dt = temps[n-1] - temps[n-2];
        let da = albedos[n-1] - albedos[n-2];
        
        if dt.abs() < 1e-10 {
            return Ok(0.0);
        }
        
        // Convert to radiative feedback (W/m²/K)
        // Incoming solar = 340 W/m², so change in absorbed = 340 * Δα
        let solar_constant = 340.0;  // W/m² global average
        let feedback = -solar_constant * (da / dt);  // Negative because less ice = lower albedo = more absorption
        
        Ok(feedback)
    }
    
    fn calculate_feedback_acceleration(&self, timestep: f64) -> Result<f64, FeedbackFailure> {
        // Need at least 3 points for acceleration
        if self.temperature_history.len() < 3 {
            return Ok(0.0);
        }
        
        let n = self.temperature_history.len();
        
        // Calculate feedback at two points
        let dt1 = self.temperature_history[n-2] - self.temperature_history[n-3];
        let da1 = self.albedo_history[n-2] - self.albedo_history[n-3];
        let feedback1 = if dt1.abs() > 1e-10 { -340.0 * (da1 / dt1) } else { 0.0 };
        
        let dt2 = self.temperature_history[n-1] - self.temperature_history[n-2];
        let da2 = self.albedo_history[n-1] - self.albedo_history[n-2];
        let feedback2 = if dt2.abs() > 1e-10 { -340.0 * (da2 / dt2) } else { 0.0 };
        
        // Acceleration is change in feedback strength
        let acceleration = (feedback2 - feedback1) / timestep;
        
        Ok(acceleration)
    }
    
    fn estimate_time_to_runaway(&self, strength: f64, acceleration: f64) -> f64 {
        if acceleration <= 0.0 {
            return f64::INFINITY;
        }
        
        // Time until feedback strength doubles
        let doubling_time = strength / acceleration;
        
        // Time until we hit runaway threshold
        let time_to_threshold = (self.max_feedback_strength - strength) / acceleration;
        
        time_to_threshold.min(doubling_time)
    }
}

// Atlantic Meridional Overturning Circulation monitoring

pub struct AMOCMonitor {
    /// AMOC strength history (Sverdrups)
    flow_history: VecDeque<f64>,
    /// North Atlantic salinity gradient
    salinity_gradient_history: VecDeque<f64>,
    /// Temperature gradient history
    temp_gradient_history: VecDeque<f64>,
    /// Critical thresholds
    min_flow_sv: f64,
    min_salinity_gradient: f64,
    freshwater_threshold: f64,
}

impl AMOCMonitor {
    pub fn new() -> Self {
        Self {
            flow_history: VecDeque::with_capacity(100),
            salinity_gradient_history: VecDeque::with_capacity(100),
            temp_gradient_history: VecDeque::with_capacity(100),
            min_flow_sv: 5.0,           // Sverdrups (10^6 m³/s)
            min_salinity_gradient: 0.5,  // PSU difference
            freshwater_threshold: 0.3,   // Sv of freshwater input
        }
    }
    
    pub fn validate(
        &mut self,
        amoc_strength_sv: f64,
        north_salinity_psu: f64,
        south_salinity_psu: f64,
        north_temp_k: f64,
        south_temp_k: f64,
        greenland_melt_sv: f64,
    ) -> Result<AMOCState, FeedbackFailure> {
        // Calculate gradients
        let salinity_gradient = north_salinity_psu - south_salinity_psu;
        let temp_gradient = north_temp_k - south_temp_k;
        
        // Record history
        self.flow_history.push_back(amoc_strength_sv);
        self.salinity_gradient_history.push_back(salinity_gradient);
        self.temp_gradient_history.push_back(temp_gradient);
        
        // Bound history
        if self.flow_history.len() > 100 {
            self.flow_history.pop_front();
            self.salinity_gradient_history.pop_front();
            self.temp_gradient_history.pop_front();
        }
        
        // Check critical thresholds
        if amoc_strength_sv < self.min_flow_sv {
            return Err(FeedbackFailure::AMOCCollapse {
                current_flow: amoc_strength_sv,
                critical_flow: self.min_flow_sv,
            });
        }
        
        // Check salinity gradient (drives circulation)
        if salinity_gradient < self.min_salinity_gradient {
            return Err(FeedbackFailure::AMOCWeakening {
                salinity_gradient,
                critical_gradient: self.min_salinity_gradient,
            });
        }
        
        // Check freshwater forcing
        if greenland_melt_sv > self.freshwater_threshold {
            return Err(FeedbackFailure::FreshwaterForcing {
                melt_rate: greenland_melt_sv,
                threshold: self.freshwater_threshold,
            });
        }
        
        // Calculate trend
        let flow_trend = self.calculate_flow_trend();
        let stability_index = self.calculate_stability_index(
            amoc_strength_sv,
            salinity_gradient,
            temp_gradient
        );
        
        // Classify state
        let state = if stability_index > 0.7 {
            AMOCState::Stable { 
                flow: amoc_strength_sv,
                trend: flow_trend,
            }
        } else if stability_index > 0.4 {
            AMOCState::Weakening {
                flow: amoc_strength_sv,
                stability_index,
                years_to_collapse: self.estimate_collapse_time(flow_trend),
            }
        } else {
            AMOCState::Critical {
                flow: amoc_strength_sv,
                stability_index,
                recovery_possible: salinity_gradient > 0.3,
            }
        };
        
        Ok(state)
    }
    
    fn calculate_flow_trend(&self) -> f64 {
        if self.flow_history.len() < 10 {
            return 0.0;
        }
        
        // Simple linear trend
        let n = self.flow_history.len();
        let recent: Vec<f64> = self.flow_history.iter()
            .skip(n - 10)
            .copied()
            .collect();
        
        // Calculate slope
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        
        for (i, &y) in recent.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        let n = recent.len() as f64;
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        slope
    }
    
    fn calculate_stability_index(
        &self,
        flow: f64,
        salinity_grad: f64,
        temp_grad: f64,
    ) -> f64 {
        // Normalized stability metric
        let flow_factor = (flow / 15.0).min(1.0);  // 15 Sv is strong AMOC
        let salinity_factor = (salinity_grad / 2.0).min(1.0);  // 2 PSU is healthy gradient
        let temp_factor = (temp_grad.abs() / 5.0).min(1.0);  // 5K gradient
        
        // Weighted combination
        0.5 * flow_factor + 0.3 * salinity_factor + 0.2 * temp_factor
    }
    
    fn estimate_collapse_time(&self, trend: f64) -> f64 {
        if trend >= 0.0 {
            return f64::INFINITY;
        }
        
        // Current flow
        let current = self.flow_history.back().copied().unwrap_or(15.0);
        
        // Time to reach critical threshold
        let time_to_critical = (current - self.min_flow_sv) / (-trend);
        
        time_to_critical
    }
}

// Permafrost carbon release tracking

pub struct PermafrostTracker {
    /// Permafrost extent history (million km²)
    extent_history: VecDeque<f64>,
    /// Carbon release rate history (GtC/yr)
    carbon_release_history: VecDeque<f64>,
    /// Methane release rate history (Mt CH4/yr)
    methane_release_history: VecDeque<f64>,
    /// Temperature at permafrost boundary
    boundary_temp_history: VecDeque<f64>,
    /// Thresholds
    max_carbon_release_rate: f64,
    max_methane_release_rate: f64,
    cascade_acceleration_threshold: f64,
}

impl PermafrostTracker {
    pub fn new() -> Self {
        Self {
            extent_history: VecDeque::with_capacity(100),
            carbon_release_history: VecDeque::with_capacity(100),
            methane_release_history: VecDeque::with_capacity(100),
            boundary_temp_history: VecDeque::with_capacity(100),
            max_carbon_release_rate: 2.0,     // GtC/yr
            max_methane_release_rate: 100.0,  // Mt CH4/yr
            cascade_acceleration_threshold: 0.1,  // GtC/yr²
        }
    }
    
    pub fn validate(
        &mut self,
        permafrost_extent_km2: f64,
        carbon_release_gtc_yr: f64,
        methane_release_mt_yr: f64,
        boundary_temp_k: f64,
        active_layer_depth_m: f64,
    ) -> Result<PermafrostState, FeedbackFailure> {
        // Record history
        self.extent_history.push_back(permafrost_extent_km2);
        self.carbon_release_history.push_back(carbon_release_gtc_yr);
        self.methane_release_history.push_back(methane_release_mt_yr);
        self.boundary_temp_history.push_back(boundary_temp_k);
        
        // Bound history
        if self.extent_history.len() > 100 {
            self.extent_history.pop_front();
            self.carbon_release_history.pop_front();
            self.methane_release_history.pop_front();
            self.boundary_temp_history.pop_front();
        }
        
        // Check release rates
        if carbon_release_gtc_yr > self.max_carbon_release_rate {
            return Err(FeedbackFailure::PermafrostCarbon {
                release_rate: carbon_release_gtc_yr,
                threshold: self.max_carbon_release_rate,
            });
        }
        
        if methane_release_mt_yr > self.max_methane_release_rate {
            return Err(FeedbackFailure::PermafrostMethane {
                release_rate: methane_release_mt_yr,
                threshold: self.max_methane_release_rate,
            });
        }
        
        // Calculate acceleration
        let carbon_acceleration = self.calculate_carbon_acceleration();
        
        if carbon_acceleration > self.cascade_acceleration_threshold {
            return Err(FeedbackFailure::PermafrostCascade {
                acceleration: carbon_acceleration,
                total_carbon_released: self.calculate_cumulative_carbon(),
            });
        }
        
        // Calculate vulnerability index
        let vulnerability = self.calculate_vulnerability(
            boundary_temp_k,
            active_layer_depth_m,
            carbon_release_gtc_yr
        );
        
        // Classify state
        let state = if vulnerability < 0.3 {
            PermafrostState::Stable {
                extent: permafrost_extent_km2,
                carbon_flux: carbon_release_gtc_yr,
            }
        } else if vulnerability < 0.7 {
            PermafrostState::Thawing {
                extent: permafrost_extent_km2,
                carbon_flux: carbon_release_gtc_yr,
                methane_flux: methane_release_mt_yr,
                vulnerability_index: vulnerability,
            }
        } else {
            PermafrostState::Cascade {
                extent: permafrost_extent_km2,
                carbon_acceleration,
                years_to_depletion: self.estimate_depletion_time(),
            }
        };
        
        Ok(state)
    }
    
    fn calculate_carbon_acceleration(&self) -> f64 {
        if self.carbon_release_history.len() < 3 {
            return 0.0;
        }
        
        let n = self.carbon_release_history.len();
        let rate1 = self.carbon_release_history[n-2];
        let rate2 = self.carbon_release_history[n-1];
        
        // Simple difference
        rate2 - rate1  // GtC/yr per year
    }
    
    fn calculate_cumulative_carbon(&self) -> f64 {
        self.carbon_release_history.iter().sum()
    }
    
    fn calculate_vulnerability(
        &self,
        temp: f64,
        active_layer: f64,
        carbon_flux: f64,
    ) -> f64 {
        // Temperature factor (273K = 0°C threshold)
        let temp_factor = ((temp - 273.0) / 5.0).max(0.0).min(1.0);
        
        // Active layer factor (deeper = more vulnerable)
        let layer_factor = (active_layer / 3.0).min(1.0);
        
        // Flux factor
        let flux_factor = (carbon_flux / self.max_carbon_release_rate).min(1.0);
        
        // Weighted combination
        0.4 * temp_factor + 0.3 * layer_factor + 0.3 * flux_factor
    }
    
    fn estimate_depletion_time(&self) -> f64 {
        // Total permafrost carbon ~1700 GtC
        let total_carbon = 1700.0;
        
        // Current release rate
        let current_rate = self.carbon_release_history.back()
            .copied()
            .unwrap_or(0.5);
        
        // Acceleration
        let acceleration = self.calculate_carbon_acceleration();
        
        if acceleration <= 0.0 {
            // Constant rate
            total_carbon / current_rate
        } else {
            // Accelerating release - solve quadratic
            // C(t) = r₀*t + 0.5*a*t²
            // When C(t) = total_carbon
            let discriminant = current_rate * current_rate + 2.0 * acceleration * total_carbon;
            if discriminant < 0.0 {
                f64::INFINITY
            } else {
                (-current_rate + discriminant.sqrt()) / acceleration
            }
        }
    }
}

// State enums

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackState {
    Insufficient,
    Stable,
    Active { strength: f64 },
    Critical { strength: f64, time_to_runaway: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AMOCState {
    Stable { 
        flow: f64,
        trend: f64,
    },
    Weakening {
        flow: f64,
        stability_index: f64,
        years_to_collapse: f64,
    },
    Critical {
        flow: f64,
        stability_index: f64,
        recovery_possible: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PermafrostState {
    Stable {
        extent: f64,
        carbon_flux: f64,
    },
    Thawing {
        extent: f64,
        carbon_flux: f64,
        methane_flux: f64,
        vulnerability_index: f64,
    },
    Cascade {
        extent: f64,
        carbon_acceleration: f64,
        years_to_depletion: f64,
    },
}

// Feedback failure conditions

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackFailure {
    PositiveFeedbackExcessive {
        feedback_type: String,
        strength: f64,
        threshold: f64,
    },
    RunawayDetected {
        feedback_type: String,
        acceleration: f64,
        ice_extent_remaining: f64,
    },
    AMOCCollapse {
        current_flow: f64,
        critical_flow: f64,
    },
    AMOCWeakening {
        salinity_gradient: f64,
        critical_gradient: f64,
    },
    FreshwaterForcing {
        melt_rate: f64,
        threshold: f64,
    },
    PermafrostCarbon {
        release_rate: f64,
        threshold: f64,
    },
    PermafrostMethane {
        release_rate: f64,
        threshold: f64,
    },
    PermafrostCascade {
        acceleration: f64,
        total_carbon_released: f64,
    },
}

// Combined feedback monitoring

pub struct FeedbackMonitor {
    ice_albedo: IceAlbedoFeedback,
    amoc: AMOCMonitor,
    permafrost: PermafrostTracker,
}

impl FeedbackMonitor {
    pub fn new() -> Self {
        Self {
            ice_albedo: IceAlbedoFeedback::new(),
            amoc: AMOCMonitor::new(),
            permafrost: PermafrostTracker::new(),
        }
    }
    
    pub fn validate_all(&mut self, state: &ClimateSystemState) -> Vec<Result<String, FeedbackFailure>> {
        let mut results = Vec::new();
        
        // Check ice-albedo
        match self.ice_albedo.validate(
            state.arctic_ice_extent,
            state.planetary_albedo,
            state.global_temp,
            state.timestep,
        ) {
            Ok(s) => results.push(Ok(format!("Ice-albedo: {:?}", s))),
            Err(e) => results.push(Err(e)),
        }
        
        // Check AMOC
        match self.amoc.validate(
            state.amoc_strength,
            state.north_atlantic_salinity,
            state.south_atlantic_salinity,
            state.north_atlantic_temp,
            state.south_atlantic_temp,
            state.greenland_melt_rate,
        ) {
            Ok(s) => results.push(Ok(format!("AMOC: {:?}", s))),
            Err(e) => results.push(Err(e)),
        }
        
        // Check permafrost
        match self.permafrost.validate(
            state.permafrost_extent,
            state.permafrost_carbon_release,
            state.permafrost_methane_release,
            state.arctic_soil_temp,
            state.active_layer_depth,
        ) {
            Ok(s) => results.push(Ok(format!("Permafrost: {:?}", s))),
            Err(e) => results.push(Err(e)),
        }
        
        results
    }
}

#[derive(Debug, Clone)]
pub struct ClimateSystemState {
    pub timestep: f64,
    pub global_temp: f64,
    pub planetary_albedo: f64,
    pub arctic_ice_extent: f64,
    pub amoc_strength: f64,
    pub north_atlantic_salinity: f64,
    pub south_atlantic_salinity: f64,
    pub north_atlantic_temp: f64,
    pub south_atlantic_temp: f64,
    pub greenland_melt_rate: f64,
    pub permafrost_extent: f64,
    pub permafrost_carbon_release: f64,
    pub permafrost_methane_release: f64,
    pub arctic_soil_temp: f64,
    pub active_layer_depth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ice_albedo_runaway() {
        let mut validator = IceAlbedoFeedback::new();
        
        // Simulate warming with ice loss
        for i in 0..20 {
            let temp = 273.0 + i as f64 * 0.5;  // Warming
            let ice = 15.0 - i as f64 * 0.5;    // Ice loss
            let albedo = 0.3 - i as f64 * 0.01;  // Albedo decrease
            
            let result = validator.validate(ice, albedo, temp, 1.0);
            
            if i < 10 {
                assert!(result.is_ok());
            }
        }
    }
    
    #[test]
    fn test_amoc_weakening() {
        let mut indicator = AMOCMonitor::new();
        
        // Simulate AMOC weakening
        for i in 0..15 {
            let flow = 15.0 - i as f64;  // Weakening flow
            let salinity_grad = 2.0 - i as f64 * 0.1;
            
            let result = indicator.validate(
                flow,
                35.5 - i as f64 * 0.1,  // North salinity
                34.0,                     // South salinity
                278.0,                    // North temp
                283.0,                    // South temp
                0.1 * i as f64,          // Increasing melt
            );
            
            if flow < 5.0 {
                assert!(result.is_err());
            }
        }
    }
}