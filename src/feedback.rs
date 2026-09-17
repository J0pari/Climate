//! Canonical numerical primitives for ice–albedo sensitivity diagnostics.
//!
//! This module estimates local changes in albedo with respect to temperature
//! from caller-supplied numeric samples. It does not supply observational data,
//! radiative forcing, empirical calibration, or policy thresholds. Empirical use
//! requires a separate provenance-preserving observation adapter; conversion of
//! d(albedo)/dT to a radiative feedback requires an explicitly sourced and
//! definition-matched shortwave field outside this module.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IceAlbedoNumericalSample {
    /// Sea-ice extent in million km². The estimator reports this as context but
    /// does not use it in the sensitivity calculation.
    pub ice_extent_million_km2: f64,
    /// Broadband/aggregate albedo represented as a fraction in [0, 1].
    pub albedo_fraction: f64,
    /// Temperature in kelvin.
    pub temperature_k: f64,
    /// Duration of the latest step in caller-declared time units.
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

impl IceAlbedoNumericalSample {
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
pub struct IceAlbedoNumericalPolicy {
    pub history_capacity: usize,
    pub minimum_samples: usize,
    pub temperature_increment_tolerance_k: f64,
}

impl IceAlbedoNumericalPolicy {
    pub fn new(
        history_capacity: usize,
        minimum_samples: usize,
        temperature_increment_tolerance_k: f64,
    ) -> Result<Self, IceAlbedoConfigError> {
        let policy = Self {
            history_capacity,
            minimum_samples,
            temperature_increment_tolerance_k,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(self) -> Result<Self, IceAlbedoConfigError> {
        if self.minimum_samples < 3 {
            return Err(IceAlbedoConfigError::TooFewMinimumSamples);
        }
        if self.history_capacity < self.minimum_samples {
            return Err(IceAlbedoConfigError::CapacityBelowMinimum {
                capacity: self.history_capacity,
                minimum: self.minimum_samples,
            });
        }
        if !self.temperature_increment_tolerance_k.is_finite()
            || self.temperature_increment_tolerance_k <= 0.0
        {
            return Err(IceAlbedoConfigError::InvalidTemperatureTolerance);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum IceAlbedoConfigError {
    #[error("history_capacity ({capacity}) must be >= minimum_samples ({minimum})")]
    CapacityBelowMinimum { capacity: usize, minimum: usize },
    #[error("minimum_samples must be at least 3 to estimate sensitivity acceleration")]
    TooFewMinimumSamples,
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
pub struct IceAlbedoSensitivityDiagnostics {
    pub sample_count: usize,
    pub latest_ice_extent_million_km2: f64,
    /// Local finite-difference estimate d(albedo)/dT in K^-1.
    pub albedo_temperature_sensitivity_per_k: QuantityEstimate,
    /// Change in d(albedo)/dT per caller-declared time step.
    pub sensitivity_acceleration_per_k_per_step: QuantityEstimate,
}

#[derive(Debug, Clone)]
pub struct IceAlbedoSensitivityTracker {
    policy: IceAlbedoNumericalPolicy,
    history: VecDeque<IceAlbedoNumericalSample>,
}

impl IceAlbedoSensitivityTracker {
    pub fn new(policy: IceAlbedoNumericalPolicy) -> Result<Self, IceAlbedoConfigError> {
        let policy = policy.validate()?;
        Ok(Self {
            policy,
            history: VecDeque::with_capacity(policy.history_capacity),
        })
    }

    pub fn observe(
        &mut self,
        sample: IceAlbedoNumericalSample,
    ) -> Result<IceAlbedoSensitivityDiagnostics, IceAlbedoInputError> {
        let sample = sample.validate()?;
        self.history.push_back(sample);
        while self.history.len() > self.policy.history_capacity {
            self.history.pop_front();
        }
        Ok(self.diagnostics())
    }

    pub fn diagnostics(&self) -> IceAlbedoSensitivityDiagnostics {
        let sample_count = self.history.len();
        let latest_ice_extent_million_km2 = self
            .history
            .back()
            .map(|sample| sample.ice_extent_million_km2)
            .unwrap_or(f64::NAN);

        if sample_count < self.policy.minimum_samples {
            let estimate = QuantityEstimate::InsufficientHistory {
                required: self.policy.minimum_samples,
                actual: sample_count,
            };
            return IceAlbedoSensitivityDiagnostics {
                sample_count,
                latest_ice_extent_million_km2,
                albedo_temperature_sensitivity_per_k: estimate,
                sensitivity_acceleration_per_k_per_step: estimate,
            };
        }

        IceAlbedoSensitivityDiagnostics {
            sample_count,
            latest_ice_extent_million_km2,
            albedo_temperature_sensitivity_per_k: self.sensitivity_between(
                sample_count - 2,
                sample_count - 1,
            ),
            sensitivity_acceleration_per_k_per_step: self.sensitivity_acceleration(),
        }
    }

    fn sensitivity_between(&self, left: usize, right: usize) -> QuantityEstimate {
        let earlier = self.history[left];
        let later = self.history[right];
        let delta_temperature_k = later.temperature_k - earlier.temperature_k;
        if delta_temperature_k.abs() < self.policy.temperature_increment_tolerance_k {
            return QuantityEstimate::Degenerate {
                reason: EstimateDegeneracy::TemperatureIncrementBelowTolerance {
                    delta_temperature_k,
                    tolerance_k: self.policy.temperature_increment_tolerance_k,
                },
            };
        }

        QuantityEstimate::Available {
            value: (later.albedo_fraction - earlier.albedo_fraction) / delta_temperature_k,
        }
    }

    fn sensitivity_acceleration(&self) -> QuantityEstimate {
        let n = self.history.len();
        let previous = self.sensitivity_between(n - 3, n - 2);
        let current = self.sensitivity_between(n - 2, n - 1);
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
            (QuantityEstimate::InsufficientHistory { required, actual }, _)
            | (_, QuantityEstimate::InsufficientHistory { required, actual }) => {
                QuantityEstimate::InsufficientHistory { required, actual }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> IceAlbedoNumericalPolicy {
        IceAlbedoNumericalPolicy::new(100, 10, 1.0e-10).unwrap()
    }

    fn sample(index: usize) -> IceAlbedoNumericalSample {
        IceAlbedoNumericalSample {
            temperature_k: 273.0 + index as f64 * 0.5,
            ice_extent_million_km2: 15.0 - index as f64 * 0.5,
            albedo_fraction: 0.3 - index as f64 * 0.01,
            step_duration: 1.0,
        }
    }

    #[test]
    fn numerical_policy_is_explicit_and_validated() {
        assert!(IceAlbedoNumericalPolicy::new(2, 3, 1.0e-10).is_err());
        assert!(IceAlbedoNumericalPolicy::new(10, 2, 1.0e-10).is_err());
        assert!(IceAlbedoNumericalPolicy::new(10, 3, 0.0).is_err());
        assert_eq!(policy().minimum_samples, 10);
    }

    #[test]
    fn readiness_boundary_is_sample_count() {
        let mut tracker = IceAlbedoSensitivityTracker::new(policy()).unwrap();
        for i in 0..9 {
            let diagnostics = tracker.observe(sample(i)).unwrap();
            assert_eq!(
                diagnostics.albedo_temperature_sensitivity_per_k,
                QuantityEstimate::InsufficientHistory {
                    required: 10,
                    actual: i + 1,
                }
            );
        }

        let diagnostics = tracker.observe(sample(9)).unwrap();
        let sensitivity = diagnostics
            .albedo_temperature_sensitivity_per_k
            .value()
            .unwrap();
        assert!((sensitivity + 0.02).abs() < 1.0e-12);
        let acceleration = diagnostics
            .sensitivity_acceleration_per_k_per_step
            .value()
            .unwrap();
        assert!(acceleration.abs() < 1.0e-12);
    }

    #[test]
    fn undefined_temperature_derivative_remains_degenerate() {
        let mut tracker = IceAlbedoSensitivityTracker::new(policy()).unwrap();
        for i in 0..9 {
            tracker.observe(sample(i)).unwrap();
        }
        let mut tenth = sample(9);
        tenth.temperature_k = sample(8).temperature_k;
        let diagnostics = tracker.observe(tenth).unwrap();

        assert!(matches!(
            diagnostics.albedo_temperature_sensitivity_per_k,
            QuantityEstimate::Degenerate {
                reason: EstimateDegeneracy::TemperatureIncrementBelowTolerance { .. }
            }
        ));
    }

    #[test]
    fn malformed_numeric_samples_fail_closed() {
        let mut tracker = IceAlbedoSensitivityTracker::new(policy()).unwrap();

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
