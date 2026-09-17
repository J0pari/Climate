//! Provenance-neutral numerical primitives for local albedo-temperature sensitivity.
//!
//! This module computes finite-difference changes in albedo with respect to
//! temperature from explicit numeric samples. Its inputs are numerical values,
//! not observations, and its outputs are not empirical climate-feedback
//! estimates. It intentionally owns no sea-ice extent, radiative forcing,
//! observational source, empirical calibration, process threshold, or fallback.
//!
//! An empirical adapter must bind source-backed values through the repository's
//! existing `crate::physical_state::ObservedField` provenance boundary, preserve
//! artifact identities and missingness, and define aggregation/time alignment
//! before constructing samples for this kernel. Radiative conversion requires a
//! separately sourced and definition-matched shortwave field.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const SENSITIVITY_RATE_HISTORY: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlbedoTemperatureSample {
    /// Broadband/aggregate albedo represented as a fraction in [0, 1].
    pub albedo_fraction: f64,
    /// Temperature in kelvin.
    pub temperature_k: f64,
    /// Elapsed time since the preceding sample, in seconds.
    pub step_duration_s: f64,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AlbedoTemperatureInputError {
    #[error("{field} must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("albedo_fraction must be in [0, 1], got {0}")]
    AlbedoOutOfRange(f64),
    #[error("temperature_k must be positive, got {0}")]
    NonPositiveTemperature(f64),
    #[error("step_duration_s must be positive, got {0}")]
    NonPositiveStepDuration(f64),
}

impl AlbedoTemperatureSample {
    fn validate(self) -> Result<Self, AlbedoTemperatureInputError> {
        for (field, value) in [
            ("albedo_fraction", self.albedo_fraction),
            ("temperature_k", self.temperature_k),
            ("step_duration_s", self.step_duration_s),
        ] {
            if !value.is_finite() {
                return Err(AlbedoTemperatureInputError::NonFinite { field, value });
            }
        }
        if !(0.0..=1.0).contains(&self.albedo_fraction) {
            return Err(AlbedoTemperatureInputError::AlbedoOutOfRange(
                self.albedo_fraction,
            ));
        }
        if self.temperature_k <= 0.0 {
            return Err(AlbedoTemperatureInputError::NonPositiveTemperature(
                self.temperature_k,
            ));
        }
        if self.step_duration_s <= 0.0 {
            return Err(AlbedoTemperatureInputError::NonPositiveStepDuration(
                self.step_duration_s,
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlbedoTemperatureNumericalPolicy {
    /// Positive numerical threshold below which d(albedo)/dT is undefined.
    pub temperature_increment_tolerance_k: f64,
}

impl AlbedoTemperatureNumericalPolicy {
    pub fn new(
        temperature_increment_tolerance_k: f64,
    ) -> Result<Self, AlbedoTemperatureConfigError> {
        let policy = Self {
            temperature_increment_tolerance_k,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(self) -> Result<Self, AlbedoTemperatureConfigError> {
        if !self.temperature_increment_tolerance_k.is_finite()
            || self.temperature_increment_tolerance_k <= 0.0
        {
            return Err(AlbedoTemperatureConfigError::InvalidTemperatureTolerance);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AlbedoTemperatureConfigError {
    #[error("temperature_increment_tolerance_k must be finite and positive")]
    InvalidTemperatureTolerance,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SensitivityDegeneracy {
    TemperatureIncrementBelowTolerance {
        delta_temperature_k: f64,
        tolerance_k: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantityEstimate {
    Available { value: f64 },
    InsufficientHistory { required: usize, actual: usize },
    Degenerate { reason: SensitivityDegeneracy },
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
pub struct AlbedoTemperatureDiagnostics {
    /// Number of samples retained by the local finite-difference stencil.
    pub retained_sample_count: usize,
    /// Local finite-difference estimate d(albedo)/dT in K^-1.
    pub albedo_temperature_sensitivity_per_k: QuantityEstimate,
    /// Change in d(albedo)/dT per second.
    pub sensitivity_rate_per_k_per_s: QuantityEstimate,
}

#[derive(Debug, Clone)]
pub struct AlbedoTemperatureTracker {
    policy: AlbedoTemperatureNumericalPolicy,
    history: VecDeque<AlbedoTemperatureSample>,
}

impl AlbedoTemperatureTracker {
    pub fn new(
        policy: AlbedoTemperatureNumericalPolicy,
    ) -> Result<Self, AlbedoTemperatureConfigError> {
        let policy = policy.validate()?;
        Ok(Self {
            policy,
            history: VecDeque::with_capacity(SENSITIVITY_RATE_HISTORY),
        })
    }

    pub fn observe(
        &mut self,
        sample: AlbedoTemperatureSample,
    ) -> Result<AlbedoTemperatureDiagnostics, AlbedoTemperatureInputError> {
        let sample = sample.validate()?;
        self.history.push_back(sample);
        while self.history.len() > SENSITIVITY_RATE_HISTORY {
            self.history.pop_front();
        }
        Ok(self.diagnostics())
    }

    pub fn diagnostics(&self) -> AlbedoTemperatureDiagnostics {
        let retained_sample_count = self.history.len();
        AlbedoTemperatureDiagnostics {
            retained_sample_count,
            albedo_temperature_sensitivity_per_k: if retained_sample_count >= 2 {
                self.sensitivity_between(retained_sample_count - 2, retained_sample_count - 1)
            } else {
                QuantityEstimate::InsufficientHistory {
                    required: 2,
                    actual: retained_sample_count,
                }
            },
            sensitivity_rate_per_k_per_s: if retained_sample_count
                >= SENSITIVITY_RATE_HISTORY
            {
                self.sensitivity_rate()
            } else {
                QuantityEstimate::InsufficientHistory {
                    required: SENSITIVITY_RATE_HISTORY,
                    actual: retained_sample_count,
                }
            },
        }
    }

    fn sensitivity_between(&self, left: usize, right: usize) -> QuantityEstimate {
        let earlier = self.history[left];
        let later = self.history[right];
        let delta_temperature_k = later.temperature_k - earlier.temperature_k;
        if delta_temperature_k.abs() < self.policy.temperature_increment_tolerance_k {
            return QuantityEstimate::Degenerate {
                reason: SensitivityDegeneracy::TemperatureIncrementBelowTolerance {
                    delta_temperature_k,
                    tolerance_k: self.policy.temperature_increment_tolerance_k,
                },
            };
        }

        QuantityEstimate::Available {
            value: (later.albedo_fraction - earlier.albedo_fraction) / delta_temperature_k,
        }
    }

    fn sensitivity_rate(&self) -> QuantityEstimate {
        let n = self.history.len();
        let previous = self.sensitivity_between(n - 3, n - 2);
        let current = self.sensitivity_between(n - 2, n - 1);
        match (previous, current) {
            (
                QuantityEstimate::Available { value: previous },
                QuantityEstimate::Available { value: current },
            ) => QuantityEstimate::Available {
                value: (current - previous) / self.history[n - 1].step_duration_s,
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

    fn policy() -> AlbedoTemperatureNumericalPolicy {
        AlbedoTemperatureNumericalPolicy::new(1.0e-10).unwrap()
    }

    fn synthetic_sample(index: usize) -> AlbedoTemperatureSample {
        AlbedoTemperatureSample {
            temperature_k: 273.0 + index as f64 * 0.5,
            albedo_fraction: 0.3 - index as f64 * 0.01,
            step_duration_s: 86_400.0,
        }
    }

    #[test]
    fn numerical_policy_is_explicit_and_validated() {
        assert!(AlbedoTemperatureNumericalPolicy::new(0.0).is_err());
        assert!(AlbedoTemperatureNumericalPolicy::new(f64::NAN).is_err());
        assert_eq!(policy().temperature_increment_tolerance_k, 1.0e-10);
    }

    #[test]
    fn derivative_and_rate_require_only_their_mathematical_stencils() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();

        let first = tracker.observe(synthetic_sample(0)).unwrap();
        assert_eq!(
            first.albedo_temperature_sensitivity_per_k,
            QuantityEstimate::InsufficientHistory {
                required: 2,
                actual: 1,
            }
        );
        assert_eq!(
            first.sensitivity_rate_per_k_per_s,
            QuantityEstimate::InsufficientHistory {
                required: 3,
                actual: 1,
            }
        );

        let second = tracker.observe(synthetic_sample(1)).unwrap();
        assert!((second.albedo_temperature_sensitivity_per_k.value().unwrap() + 0.02).abs() < 1.0e-12);
        assert_eq!(
            second.sensitivity_rate_per_k_per_s,
            QuantityEstimate::InsufficientHistory {
                required: 3,
                actual: 2,
            }
        );

        let third = tracker.observe(synthetic_sample(2)).unwrap();
        assert!(third.sensitivity_rate_per_k_per_s.value().unwrap().abs() < 1.0e-18);
    }

    #[test]
    fn history_is_bounded_by_the_required_stencil() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();
        for index in 0..20 {
            tracker.observe(synthetic_sample(index)).unwrap();
        }
        assert_eq!(tracker.diagnostics().retained_sample_count, SENSITIVITY_RATE_HISTORY);
    }

    #[test]
    fn undefined_temperature_derivative_remains_degenerate() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();
        tracker.observe(synthetic_sample(0)).unwrap();
        tracker.observe(synthetic_sample(1)).unwrap();
        let mut third = synthetic_sample(2);
        third.temperature_k = synthetic_sample(1).temperature_k;
        let diagnostics = tracker.observe(third).unwrap();

        assert!(matches!(
            diagnostics.albedo_temperature_sensitivity_per_k,
            QuantityEstimate::Degenerate {
                reason: SensitivityDegeneracy::TemperatureIncrementBelowTolerance { .. }
            }
        ));
        assert!(matches!(
            diagnostics.sensitivity_rate_per_k_per_s,
            QuantityEstimate::Degenerate { .. }
        ));
    }

    #[test]
    fn malformed_numeric_samples_fail_closed() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();

        let mut bad = synthetic_sample(0);
        bad.albedo_fraction = 1.1;
        assert!(matches!(
            tracker.observe(bad),
            Err(AlbedoTemperatureInputError::AlbedoOutOfRange(_))
        ));

        let mut bad = synthetic_sample(0);
        bad.step_duration_s = 0.0;
        assert!(matches!(
            tracker.observe(bad),
            Err(AlbedoTemperatureInputError::NonPositiveStepDuration(_))
        ));
    }
}
