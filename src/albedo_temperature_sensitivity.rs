//! Provenance-neutral numerical primitives for local albedo-temperature trajectory secants.
//!
//! This module computes finite-difference secant slopes along an observed or
//! simulated trajectory in (temperature, albedo) space. Its inputs are numerical
//! values, not observations, and its outputs are not partial derivatives, causal
//! effects, or empirical climate-feedback estimates. It intentionally owns no sea-ice extent, radiative forcing,
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

const SECANT_RATE_HISTORY: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlbedoTemperatureSample {
    /// Broadband/aggregate albedo represented as a fraction in [0, 1].
    pub albedo_fraction: f64,
    /// Temperature in kelvin.
    pub temperature_k: f64,
    /// Sample time in seconds relative to a caller-declared origin.
    pub time_s: f64,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AlbedoTemperatureInputError {
    #[error("{field} must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("albedo_fraction must be in [0, 1], got {0}")]
    AlbedoOutOfRange(f64),
    #[error("temperature_k must be positive, got {0}")]
    NonPositiveTemperature(f64),
    #[error("sample time must increase strictly: previous={previous_s}, current={current_s}")]
    NonIncreasingTime { previous_s: f64, current_s: f64 },
}

impl AlbedoTemperatureSample {
    fn validate(self) -> Result<Self, AlbedoTemperatureInputError> {
        for (field, value) in [
            ("albedo_fraction", self.albedo_fraction),
            ("temperature_k", self.temperature_k),
            ("time_s", self.time_s),
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
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlbedoTemperatureNumericalPolicy {
    /// Positive numerical threshold below which the trajectory secant in temperature is undefined.
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
pub enum SecantDegeneracy {
    TemperatureIncrementBelowTolerance {
        delta_temperature_k: f64,
        tolerance_k: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantityEstimate {
    Available { value: f64 },
    InsufficientHistory { required: usize, actual: usize },
    Degenerate { reason: SecantDegeneracy },
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
    /// Secant slope Δ(albedo)/ΔT along the latest trajectory interval, in K^-1.
    /// This is not a causal or partial feedback derivative.
    pub albedo_temperature_trajectory_secant_per_k: QuantityEstimate,
    /// Change in successive trajectory secant slopes per second.
    pub trajectory_secant_rate_per_k_per_s: QuantityEstimate,
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
            history: VecDeque::with_capacity(SECANT_RATE_HISTORY),
        })
    }

    pub fn observe(
        &mut self,
        sample: AlbedoTemperatureSample,
    ) -> Result<AlbedoTemperatureDiagnostics, AlbedoTemperatureInputError> {
        let sample = sample.validate()?;
        if let Some(previous) = self.history.back() {
            if sample.time_s <= previous.time_s {
                return Err(AlbedoTemperatureInputError::NonIncreasingTime {
                    previous_s: previous.time_s,
                    current_s: sample.time_s,
                });
            }
        }
        self.history.push_back(sample);
        while self.history.len() > SECANT_RATE_HISTORY {
            self.history.pop_front();
        }
        Ok(self.diagnostics())
    }

    pub fn diagnostics(&self) -> AlbedoTemperatureDiagnostics {
        let retained_sample_count = self.history.len();
        AlbedoTemperatureDiagnostics {
            retained_sample_count,
            albedo_temperature_trajectory_secant_per_k: if retained_sample_count >= 2 {
                self.secant_between(retained_sample_count - 2, retained_sample_count - 1)
            } else {
                QuantityEstimate::InsufficientHistory {
                    required: 2,
                    actual: retained_sample_count,
                }
            },
            trajectory_secant_rate_per_k_per_s: if retained_sample_count
                >= SECANT_RATE_HISTORY
            {
                self.secant_rate()
            } else {
                QuantityEstimate::InsufficientHistory {
                    required: SECANT_RATE_HISTORY,
                    actual: retained_sample_count,
                }
            },
        }
    }

    fn secant_between(&self, left: usize, right: usize) -> QuantityEstimate {
        let earlier = self.history[left];
        let later = self.history[right];
        let delta_temperature_k = later.temperature_k - earlier.temperature_k;
        if delta_temperature_k.abs() < self.policy.temperature_increment_tolerance_k {
            return QuantityEstimate::Degenerate {
                reason: SecantDegeneracy::TemperatureIncrementBelowTolerance {
                    delta_temperature_k,
                    tolerance_k: self.policy.temperature_increment_tolerance_k,
                },
            };
        }

        QuantityEstimate::Available {
            value: (later.albedo_fraction - earlier.albedo_fraction) / delta_temperature_k,
        }
    }

    fn secant_rate(&self) -> QuantityEstimate {
        let n = self.history.len();
        let previous = self.secant_between(n - 3, n - 2);
        let current = self.secant_between(n - 2, n - 1);
        match (previous, current) {
            (
                QuantityEstimate::Available { value: previous },
                QuantityEstimate::Available { value: current },
            ) => {
                let midpoint_separation_s =
                    0.5 * (self.history[n - 1].time_s - self.history[n - 3].time_s);
                QuantityEstimate::Available {
                    value: (current - previous) / midpoint_separation_s,
                }
            }
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
            time_s: index as f64 * 86_400.0,
        }
    }

    #[test]
    fn numerical_policy_is_explicit_and_validated() {
        assert!(AlbedoTemperatureNumericalPolicy::new(0.0).is_err());
        assert!(AlbedoTemperatureNumericalPolicy::new(f64::NAN).is_err());
        assert_eq!(policy().temperature_increment_tolerance_k, 1.0e-10);
    }

    #[test]
    fn trajectory_secant_and_rate_require_only_their_mathematical_stencils() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();

        let first = tracker.observe(synthetic_sample(0)).unwrap();
        assert_eq!(
            first.albedo_temperature_trajectory_secant_per_k,
            QuantityEstimate::InsufficientHistory {
                required: 2,
                actual: 1,
            }
        );
        assert_eq!(
            first.trajectory_secant_rate_per_k_per_s,
            QuantityEstimate::InsufficientHistory {
                required: 3,
                actual: 1,
            }
        );

        let second = tracker.observe(synthetic_sample(1)).unwrap();
        assert!((second.albedo_temperature_trajectory_secant_per_k.value().unwrap() + 0.02).abs() < 1.0e-12);
        assert_eq!(
            second.trajectory_secant_rate_per_k_per_s,
            QuantityEstimate::InsufficientHistory {
                required: 3,
                actual: 2,
            }
        );

        let third = tracker.observe(synthetic_sample(2)).unwrap();
        assert!(third.trajectory_secant_rate_per_k_per_s.value().unwrap().abs() < 1.0e-18);
    }

    #[test]
    fn irregular_time_spacing_uses_interval_midpoint_separation() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();
        tracker
            .observe(AlbedoTemperatureSample {
                time_s: 0.0,
                temperature_k: 273.0,
                albedo_fraction: 0.30,
            })
            .unwrap();
        tracker
            .observe(AlbedoTemperatureSample {
                time_s: 10.0,
                temperature_k: 274.0,
                albedo_fraction: 0.28,
            })
            .unwrap();
        let diagnostics = tracker
            .observe(AlbedoTemperatureSample {
                time_s: 40.0,
                temperature_k: 275.0,
                albedo_fraction: 0.25,
            })
            .unwrap();

        let rate = diagnostics.trajectory_secant_rate_per_k_per_s.value().unwrap();
        // Slopes -0.02 and -0.03 live at t=5 s and t=25 s.
        assert!((rate + 0.0005).abs() < 1.0e-15);
    }

    #[test]
    fn history_is_bounded_by_the_required_stencil() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();
        for index in 0..20 {
            tracker.observe(synthetic_sample(index)).unwrap();
        }
        assert_eq!(tracker.diagnostics().retained_sample_count, SECANT_RATE_HISTORY);
    }

    #[test]
    fn undefined_temperature_secant_remains_degenerate() {
        let mut tracker = AlbedoTemperatureTracker::new(policy()).unwrap();
        tracker.observe(synthetic_sample(0)).unwrap();
        tracker.observe(synthetic_sample(1)).unwrap();
        let mut third = synthetic_sample(2);
        third.temperature_k = synthetic_sample(1).temperature_k;
        let diagnostics = tracker.observe(third).unwrap();

        assert!(matches!(
            diagnostics.albedo_temperature_trajectory_secant_per_k,
            QuantityEstimate::Degenerate {
                reason: SecantDegeneracy::TemperatureIncrementBelowTolerance { .. }
            }
        ));
        assert!(matches!(
            diagnostics.trajectory_secant_rate_per_k_per_s,
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

        tracker.observe(synthetic_sample(0)).unwrap();
        let mut bad = synthetic_sample(1);
        bad.time_s = synthetic_sample(0).time_s;
        assert!(matches!(
            tracker.observe(bad),
            Err(AlbedoTemperatureInputError::NonIncreasingTime { .. })
        ));
    }
}
