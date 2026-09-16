//! Canonical ice–albedo diagnostic primitives.
//!
//! This module deliberately separates observation, numerical estimation, and
//! policy interpretation.  The legacy implementation mixed all three and
//! converted an undefined temperature derivative into `0.0`; the canonical
//! surface preserves the useful finite-difference machinery while making
//! degeneracy and threshold semantics explicit.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IceAlbedoSample {
    /// Arctic sea-ice extent in million km².  Retained as scientific context;
    /// the current finite-difference estimator does not yet use it directly.
    pub ice_extent_million_km2: f64,
    /// Broadband/aggregate albedo represented as a fraction in [0, 1].
    pub albedo_fraction: f64,
    /// Surface temperature in kelvin.
    pub temperature_k: f64,
    /// Duration of the latest step in caller-declared time units.
    ///
    /// The legacy code did not declare whether this meant days, months, or
    /// years.  The canonical estimator therefore does not invent a unit: it
    /// reports acceleration per step and requires experiment metadata to bind
    /// the time unit before physical interpretation.
    pub step_duration: f64,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum IceAlbedoInputError {
    #[error("{field} must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("albedo_fraction must be in [0, 1], got {0}")]
    AlbedoOutOfRange(f64),
    #[error("ice_extent_million_km2 must be non-negative, got {0}")]
    NegativeIceExtent(f64),
    #[error("temperature_k must be positive, got {0}")]
    NonPositiveTemperature(f64),
    #[error("step_duration must be positive, got {0}")]
    NonPositiveStepDuration(f64),
}

impl IceAlbedoSample {
    fn validate(self) -> Result<Self, IceAlbedoInputError> {
        for (field, value) in [
            ("ice_extent_million_km2", self.ice_extent_million_km2),
            ("albedo_fraction", self.albedo_fraction),
            ("temperature_k", self.temperature_k),
            ("step_duration", self.step_duration),
        ] {
            if !value.is_finite() {
                return Err(IceAlbedoInputError::NonFinite { field, value });
            }
        }
        if self.ice_extent_million_km2 < 0.0 {
            return Err(IceAlbedoInputError::NegativeIceExtent(
                self.ice_extent_million_km2,
            ));
        }
        if !(0.0..=1.0).contains(&self.albedo_fraction) {
            return Err(IceAlbedoInputError::AlbedoOutOfRange(self.albedo_fraction));
        }
        if self.temperature_k <= 0.0 {
            return Err(IceAlbedoInputError::NonPositiveTemperature(
                self.temperature_k,
            ));
        }
        if self.step_duration <= 0.0 {
            return Err(IceAlbedoInputError::NonPositiveStepDuration(
                self.step_duration,
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IceAlbedoEstimatorConfig {
    pub history_capacity: usize,
    pub minimum_samples: usize,
    /// Legacy global-average incoming shortwave reference used to map
    /// d(albedo)/dT into W m⁻² K⁻¹.  This is an estimator assumption, not a
    /// validated climate constant for every regime.
    pub absorbed_shortwave_reference_w_m2: f64,
    pub temperature_increment_tolerance_k: f64,
}

impl IceAlbedoEstimatorConfig {
    /// Preserve the numerical assumptions of the legacy estimator while
    /// exposing them as configuration instead of hidden constants.
    pub const fn legacy_reference() -> Self {
        Self {
            history_capacity: 100,
            minimum_samples: 10,
            absorbed_shortwave_reference_w_m2: 340.0,
            temperature_increment_tolerance_k: 1.0e-10,
        }
    }
}

impl Default for IceAlbedoEstimatorConfig {
    fn default() -> Self {
        Self::legacy_reference()
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum IceAlbedoConfigError {
    #[error("history_capacity ({capacity}) must be >= minimum_samples ({minimum})")]
    CapacityBelowMinimum { capacity: usize, minimum: usize },
    #[error("minimum_samples must be at least 3 to estimate acceleration")]
    TooFewMinimumSamples,
    #[error("absorbed_shortwave_reference_w_m2 must be finite and positive")]
    InvalidShortwaveReference,
    #[error("temperature_increment_tolerance_k must be finite and positive")]
    InvalidTemperatureTolerance,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EstimateDegeneracy {
    TemperatureIncrementBelowTolerance {
        delta_temperature_k: f64,
        tolerance_k: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantityEstimate {
    Available { value: f64 },
    InsufficientHistory { required: usize, actual: usize },
    Degenerate { reason: EstimateDegeneracy },
}

impl QuantityEstimate {
    pub fn value(self) -> Option<f64> {
        match self {
            Self::Available { value } => Some(value),
            Self::InsufficientHistory { .. } | Self::Degenerate { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IceAlbedoDiagnostics {
    pub sample_count: usize,
    pub latest_ice_extent_million_km2: f64,
    pub feedback_strength_w_m2_k: QuantityEstimate,
    /// Change in estimated feedback strength per caller-declared time step.
    pub feedback_acceleration_w_m2_k_per_step: QuantityEstimate,
}

#[derive(Debug, Clone)]
pub struct IceAlbedoTracker {
    config: IceAlbedoEstimatorConfig,
    history: VecDeque<IceAlbedoSample>,
}

impl IceAlbedoTracker {
    pub fn new(config: IceAlbedoEstimatorConfig) -> Result<Self, IceAlbedoConfigError> {
        if config.minimum_samples < 3 {
            return Err(IceAlbedoConfigError::TooFewMinimumSamples);
        }
        if config.history_capacity < config.minimum_samples {
            return Err(IceAlbedoConfigError::CapacityBelowMinimum {
                capacity: config.history_capacity,
                minimum: config.minimum_samples,
            });
        }
        if !config.absorbed_shortwave_reference_w_m2.is_finite()
            || config.absorbed_shortwave_reference_w_m2 <= 0.0
        {
            return Err(IceAlbedoConfigError::InvalidShortwaveReference);
        }
        if !config.temperature_increment_tolerance_k.is_finite()
            || config.temperature_increment_tolerance_k <= 0.0
        {
            return Err(IceAlbedoConfigError::InvalidTemperatureTolerance);
        }

        Ok(Self {
            config,
            history: VecDeque::with_capacity(config.history_capacity),
        })
    }

    pub fn legacy_reference() -> Self {
        Self::new(IceAlbedoEstimatorConfig::legacy_reference())
            .expect("legacy reference configuration is internally valid")
    }

    pub fn observe(
        &mut self,
        sample: IceAlbedoSample,
    ) -> Result<IceAlbedoDiagnostics, IceAlbedoInputError> {
        let sample = sample.validate()?;
        self.history.push_back(sample);
        while self.history.len() > self.config.history_capacity {
            self.history.pop_front();
        }
        Ok(self.diagnostics())
    }

    pub fn diagnostics(&self) -> IceAlbedoDiagnostics {
        let sample_count = self.history.len();
        let latest_ice_extent_million_km2 = self
            .history
            .back()
            .map(|sample| sample.ice_extent_million_km2)
            .unwrap_or(f64::NAN);

        if sample_count < self.config.minimum_samples {
            let estimate = QuantityEstimate::InsufficientHistory {
                required: self.config.minimum_samples,
                actual: sample_count,
            };
            return IceAlbedoDiagnostics {
                sample_count,
                latest_ice_extent_million_km2,
                feedback_strength_w_m2_k: estimate,
                feedback_acceleration_w_m2_k_per_step: estimate,
            };
        }

        IceAlbedoDiagnostics {
            sample_count,
            latest_ice_extent_million_km2,
            feedback_strength_w_m2_k: self.feedback_between(sample_count - 2, sample_count - 1),
            feedback_acceleration_w_m2_k_per_step: self.feedback_acceleration(),
        }
    }

    fn feedback_between(&self, left: usize, right: usize) -> QuantityEstimate {
        let earlier = self.history[left];
        let later = self.history[right];
        let delta_temperature_k = later.temperature_k - earlier.temperature_k;
        if delta_temperature_k.abs() < self.config.temperature_increment_tolerance_k {
            return QuantityEstimate::Degenerate {
                reason: EstimateDegeneracy::TemperatureIncrementBelowTolerance {
                    delta_temperature_k,
                    tolerance_k: self.config.temperature_increment_tolerance_k,
                },
            };
        }

        let delta_albedo = later.albedo_fraction - earlier.albedo_fraction;
        QuantityEstimate::Available {
            value: -self.config.absorbed_shortwave_reference_w_m2
                * (delta_albedo / delta_temperature_k),
        }
    }

    fn feedback_acceleration(&self) -> QuantityEstimate {
        let n = self.history.len();
        let previous = self.feedback_between(n - 3, n - 2);
        let current = self.feedback_between(n - 2, n - 1);
        match (previous, current) {
            (
                QuantityEstimate::Available { value: previous },
                QuantityEstimate::Available { value: current },
            ) => QuantityEstimate::Available {
                value: (current - previous) / self.history[n - 1].step_duration,
            },
            (QuantityEstimate::Degenerate { reason }, _)
            | (_, QuantityEstimate::Degenerate { reason }) => {
                QuantityEstimate::Degenerate { reason }
            }
            // `diagnostics` only calls this after minimum history has been met,
            // so this arm is defensive rather than a reachable normal state.
            (QuantityEstimate::InsufficientHistory { required, actual }, _)
            | (_, QuantityEstimate::InsufficientHistory { required, actual }) => {
                QuantityEstimate::InsufficientHistory { required, actual }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IceAlbedoPolicy {
    pub stable_below_w_m2_k: f64,
    pub active_below_w_m2_k: f64,
    pub excessive_feedback_w_m2_k: f64,
    pub acceleration_limit_w_m2_k_per_step: f64,
}

impl IceAlbedoPolicy {
    /// Legacy thresholds, preserved as an explicit *policy* rather than being
    /// embedded in the estimator or presented as empirically validated tipping
    /// thresholds.
    pub const fn legacy_reference() -> Self {
        Self {
            stable_below_w_m2_k: 0.1,
            active_below_w_m2_k: 0.3,
            excessive_feedback_w_m2_k: 0.5,
            acceleration_limit_w_m2_k_per_step: 0.1,
        }
    }

    pub fn assess(self, diagnostics: IceAlbedoDiagnostics) -> IceAlbedoAssessment {
        let Some(strength) = diagnostics.feedback_strength_w_m2_k.value() else {
            return IceAlbedoAssessment::NotAssessable { diagnostics };
        };
        let Some(acceleration) = diagnostics
            .feedback_acceleration_w_m2_k_per_step
            .value()
        else {
            return IceAlbedoAssessment::NotAssessable { diagnostics };
        };

        let mut breaches = Vec::new();
        if strength > self.excessive_feedback_w_m2_k {
            breaches.push(IceAlbedoPolicyBreach::FeedbackLimitExceeded {
                measured_w_m2_k: strength,
                limit_w_m2_k: self.excessive_feedback_w_m2_k,
            });
        }
        if acceleration > self.acceleration_limit_w_m2_k_per_step {
            breaches.push(IceAlbedoPolicyBreach::AccelerationLimitExceeded {
                measured_w_m2_k_per_step: acceleration,
                limit_w_m2_k_per_step: self.acceleration_limit_w_m2_k_per_step,
                latest_ice_extent_million_km2: diagnostics.latest_ice_extent_million_km2,
            });
        }
        if !breaches.is_empty() {
            return IceAlbedoAssessment::PolicyExceeded {
                diagnostics,
                breaches,
            };
        }

        if strength < self.stable_below_w_m2_k {
            IceAlbedoAssessment::Stable { diagnostics }
        } else if strength < self.active_below_w_m2_k {
            IceAlbedoAssessment::Active { diagnostics }
        } else {
            let projected_steps_to_policy_limit = if acceleration > 0.0 {
                let to_limit = (self.excessive_feedback_w_m2_k - strength) / acceleration;
                let doubling = if strength > 0.0 {
                    strength / acceleration
                } else {
                    f64::INFINITY
                };
                Some(to_limit.min(doubling).max(0.0))
            } else {
                None
            };
            IceAlbedoAssessment::Critical {
                diagnostics,
                projected_steps_to_policy_limit,
            }
        }
    }
}

impl Default for IceAlbedoPolicy {
    fn default() -> Self {
        Self::legacy_reference()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IceAlbedoPolicyBreach {
    FeedbackLimitExceeded {
        measured_w_m2_k: f64,
        limit_w_m2_k: f64,
    },
    AccelerationLimitExceeded {
        measured_w_m2_k_per_step: f64,
        limit_w_m2_k_per_step: f64,
        latest_ice_extent_million_km2: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IceAlbedoAssessment {
    NotAssessable {
        diagnostics: IceAlbedoDiagnostics,
    },
    Stable {
        diagnostics: IceAlbedoDiagnostics,
    },
    Active {
        diagnostics: IceAlbedoDiagnostics,
    },
    Critical {
        diagnostics: IceAlbedoDiagnostics,
        /// This is a projection to a configured policy limit in caller-declared
        /// time steps, not a physical prediction of a climate tipping time.
        projected_steps_to_policy_limit: Option<f64>,
    },
    PolicyExceeded {
        diagnostics: IceAlbedoDiagnostics,
        breaches: Vec<IceAlbedoPolicyBreach>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(index: usize) -> IceAlbedoSample {
        IceAlbedoSample {
            temperature_k: 273.0 + index as f64 * 0.5,
            ice_extent_million_km2: 15.0 - index as f64 * 0.5,
            albedo_fraction: 0.3 - index as f64 * 0.01,
            step_duration: 1.0,
        }
    }

    #[test]
    fn legacy_numerical_assumptions_are_explicit_configuration() {
        let config = IceAlbedoEstimatorConfig::legacy_reference();
        assert_eq!(config.history_capacity, 100);
        assert_eq!(config.minimum_samples, 10);
        assert_eq!(config.absorbed_shortwave_reference_w_m2, 340.0);
        assert_eq!(config.temperature_increment_tolerance_k, 1.0e-10);

        let policy = IceAlbedoPolicy::legacy_reference();
        assert_eq!(policy.stable_below_w_m2_k, 0.1);
        assert_eq!(policy.active_below_w_m2_k, 0.3);
        assert_eq!(policy.excessive_feedback_w_m2_k, 0.5);
        assert_eq!(policy.acceleration_limit_w_m2_k_per_step, 0.1);
    }

    #[test]
    fn readiness_boundary_is_sample_count_not_loop_index() {
        let mut tracker = IceAlbedoTracker::legacy_reference();
        for i in 0..9 {
            let diagnostics = tracker.observe(sample(i)).unwrap();
            assert_eq!(
                diagnostics.feedback_strength_w_m2_k,
                QuantityEstimate::InsufficientHistory {
                    required: 10,
                    actual: i + 1,
                }
            );
        }

        let diagnostics = tracker.observe(sample(9)).unwrap();
        let strength = diagnostics.feedback_strength_w_m2_k.value().unwrap();
        assert!((strength - 6.8).abs() < 1.0e-12);
        let acceleration = diagnostics
            .feedback_acceleration_w_m2_k_per_step
            .value()
            .unwrap();
        assert!(acceleration.abs() < 1.0e-12);
    }

    #[test]
    fn undefined_temperature_derivative_is_not_smoothed_to_zero() {
        let mut tracker = IceAlbedoTracker::legacy_reference();
        for i in 0..9 {
            tracker.observe(sample(i)).unwrap();
        }
        let mut tenth = sample(9);
        tenth.temperature_k = sample(8).temperature_k;
        let diagnostics = tracker.observe(tenth).unwrap();

        assert!(matches!(
            diagnostics.feedback_strength_w_m2_k,
            QuantityEstimate::Degenerate {
                reason: EstimateDegeneracy::TemperatureIncrementBelowTolerance { .. }
            }
        ));
        assert!(matches!(
            IceAlbedoPolicy::legacy_reference().assess(diagnostics),
            IceAlbedoAssessment::NotAssessable { .. }
        ));
    }

    #[test]
    fn policy_collects_distinct_breaches_instead_of_short_circuiting() {
        let diagnostics = IceAlbedoDiagnostics {
            sample_count: 10,
            latest_ice_extent_million_km2: 8.0,
            feedback_strength_w_m2_k: QuantityEstimate::Available { value: 0.6 },
            feedback_acceleration_w_m2_k_per_step: QuantityEstimate::Available { value: 0.2 },
        };

        let assessment = IceAlbedoPolicy::legacy_reference().assess(diagnostics);
        match assessment {
            IceAlbedoAssessment::PolicyExceeded { breaches, .. } => {
                assert_eq!(breaches.len(), 2);
                assert!(matches!(
                    breaches[0],
                    IceAlbedoPolicyBreach::FeedbackLimitExceeded { .. }
                ));
                assert!(matches!(
                    breaches[1],
                    IceAlbedoPolicyBreach::AccelerationLimitExceeded { .. }
                ));
            }
            other => panic!("expected both policy breaches, got {other:?}"),
        }
    }

    #[test]
    fn critical_projection_is_to_policy_limit_not_physical_runaway() {
        let diagnostics = IceAlbedoDiagnostics {
            sample_count: 10,
            latest_ice_extent_million_km2: 9.0,
            feedback_strength_w_m2_k: QuantityEstimate::Available { value: 0.4 },
            feedback_acceleration_w_m2_k_per_step: QuantityEstimate::Available { value: 0.05 },
        };

        match IceAlbedoPolicy::legacy_reference().assess(diagnostics) {
            IceAlbedoAssessment::Critical {
                projected_steps_to_policy_limit: Some(steps),
                ..
            } => assert!((steps - 2.0).abs() < 1.0e-12),
            other => panic!("expected critical assessment, got {other:?}"),
        }
    }

    #[test]
    fn malformed_observations_fail_closed() {
        let mut tracker = IceAlbedoTracker::legacy_reference();
        let mut bad = sample(0);
        bad.albedo_fraction = 1.1;
        assert!(matches!(
            tracker.observe(bad),
            Err(IceAlbedoInputError::AlbedoOutOfRange(_))
        ));

        let mut bad = sample(0);
        bad.step_duration = 0.0;
        assert!(matches!(
            tracker.observe(bad),
            Err(IceAlbedoInputError::NonPositiveStepDuration(_))
        ));
    }
}
