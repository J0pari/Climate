//! Observation-induced Fisher geometry for the two-layer energy-balance state.
//!
//! This module owns only the climate-specific observation semantics. Generic
//! Gaussian Fisher construction, rank revelation, and conditioning remain in
//! `crate::inference`.
//!
//! The state is `(T_surface, T_deep)`. For the standard linear two-layer model:
//!
//! - surface temperature has Jacobian `(1, 0)`;
//! - top-of-atmosphere imbalance `F - lambda T_surface` has Jacobian
//!   `(-lambda, 0)`;
//! - ocean heat uptake `gamma (T_surface - T_deep)` has Jacobian
//!   `(gamma, -gamma)`.
//!
//! After each observation row is whitened by its declared standard deviation,
//! the state Fisher metric is `G = H^T H`. Collinear observation rows may
//! increase information along an already observed direction without increasing
//! identifiable state dimension.

use nalgebra::DMatrix;
use thiserror::Error;

use crate::inference::{
    gaussian_mean_fisher, FisherInformationError, FisherInformationReport,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoLayerObservation {
    SurfaceTemperature,
    ToaImbalance,
    OceanHeatUptake,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoLayerObservationChannel {
    pub observation: TwoLayerObservation,
    pub standard_deviation: f64,
}

impl TwoLayerObservationChannel {
    pub const fn new(observation: TwoLayerObservation, standard_deviation: f64) -> Self {
        Self {
            observation,
            standard_deviation,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum TwoLayerObservationGeometryError {
    #[error("at least one two-layer observation channel is required")]
    EmptyObservationSet,
    #[error("climate feedback lambda must be finite and > 0; got {0}")]
    InvalidClimateFeedback(f64),
    #[error("ocean heat-exchange coefficient gamma must be finite and > 0; got {0}")]
    InvalidOceanHeatExchange(f64),
    #[error(transparent)]
    Fisher(#[from] FisherInformationError),
}

fn validate_coefficients(
    climate_feedback: f64,
    ocean_heat_exchange: f64,
) -> Result<(), TwoLayerObservationGeometryError> {
    if !climate_feedback.is_finite() || climate_feedback <= 0.0 {
        return Err(TwoLayerObservationGeometryError::InvalidClimateFeedback(
            climate_feedback,
        ));
    }
    if !ocean_heat_exchange.is_finite() || ocean_heat_exchange <= 0.0 {
        return Err(TwoLayerObservationGeometryError::InvalidOceanHeatExchange(
            ocean_heat_exchange,
        ));
    }
    Ok(())
}

/// Build the unwhitened observation-mean Jacobian with respect to
/// `(T_surface, T_deep)`.
pub fn two_layer_observation_jacobian(
    climate_feedback: f64,
    ocean_heat_exchange: f64,
    observations: &[TwoLayerObservation],
) -> Result<DMatrix<f64>, TwoLayerObservationGeometryError> {
    validate_coefficients(climate_feedback, ocean_heat_exchange)?;
    if observations.is_empty() {
        return Err(TwoLayerObservationGeometryError::EmptyObservationSet);
    }

    let mut rows = Vec::with_capacity(2 * observations.len());
    for observation in observations {
        match observation {
            TwoLayerObservation::SurfaceTemperature => {
                rows.extend_from_slice(&[1.0, 0.0]);
            }
            TwoLayerObservation::ToaImbalance => {
                rows.extend_from_slice(&[-climate_feedback, 0.0]);
            }
            TwoLayerObservation::OceanHeatUptake => {
                rows.extend_from_slice(&[
                    ocean_heat_exchange,
                    -ocean_heat_exchange,
                ]);
            }
        }
    }

    Ok(DMatrix::from_row_slice(observations.len(), 2, &rows))
}

/// Pull independent Gaussian observation noise back to the two-dimensional
/// EBM state through the declared observation maps.
pub fn two_layer_state_fisher(
    climate_feedback: f64,
    ocean_heat_exchange: f64,
    channels: &[TwoLayerObservationChannel],
) -> Result<FisherInformationReport, TwoLayerObservationGeometryError> {
    if channels.is_empty() {
        return Err(TwoLayerObservationGeometryError::EmptyObservationSet);
    }
    let observations = channels
        .iter()
        .map(|channel| channel.observation)
        .collect::<Vec<_>>();
    let noise = channels
        .iter()
        .map(|channel| channel.standard_deviation)
        .collect::<Vec<_>>();
    let jacobian = two_layer_observation_jacobian(
        climate_feedback,
        ocean_heat_exchange,
        &observations,
    )?;
    Ok(gaussian_mean_fisher(&jacobian, &noise)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAMBDA: f64 = 1.13;
    const GAMMA: f64 = 0.73;

    fn channel(
        observation: TwoLayerObservation,
        standard_deviation: f64,
    ) -> TwoLayerObservationChannel {
        TwoLayerObservationChannel::new(observation, standard_deviation)
    }

    fn approx_eq(left: f64, right: f64, relative: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= relative * scale,
            "left={left:e}, right={right:e}, relative={relative:e}"
        );
    }

    #[test]
    fn observation_rows_match_two_layer_physics() {
        let jacobian = two_layer_observation_jacobian(
            LAMBDA,
            GAMMA,
            &[
                TwoLayerObservation::SurfaceTemperature,
                TwoLayerObservation::ToaImbalance,
                TwoLayerObservation::OceanHeatUptake,
            ],
        )
        .unwrap();

        approx_eq(jacobian[(0, 0)], 1.0, 1e-14);
        approx_eq(jacobian[(0, 1)], 0.0, 1e-14);
        approx_eq(jacobian[(1, 0)], -LAMBDA, 1e-14);
        approx_eq(jacobian[(1, 1)], 0.0, 1e-14);
        approx_eq(jacobian[(2, 0)], GAMMA, 1e-14);
        approx_eq(jacobian[(2, 1)], -GAMMA, 1e-14);
    }

    #[test]
    fn surface_and_toa_refine_one_direction_without_identifying_deep_state() {
        let surface = two_layer_state_fisher(
            LAMBDA,
            GAMMA,
            &[channel(TwoLayerObservation::SurfaceTemperature, 0.1)],
        )
        .unwrap();
        let surface_and_toa = two_layer_state_fisher(
            LAMBDA,
            GAMMA,
            &[
                channel(TwoLayerObservation::SurfaceTemperature, 0.1),
                channel(TwoLayerObservation::ToaImbalance, 0.2),
            ],
        )
        .unwrap();

        assert_eq!(surface.parameter_nullity, 1);
        assert_eq!(surface_and_toa.parameter_nullity, 1);
        assert!(surface.fisher_condition_number_2.is_infinite());
        assert!(surface_and_toa.fisher_condition_number_2.is_infinite());
        assert!(surface_and_toa.fisher[(0, 0)] > surface.fisher[(0, 0)]);
        approx_eq(surface_and_toa.fisher[(1, 1)], 0.0, 1e-14);
    }

    #[test]
    fn ocean_heat_uptake_closes_the_local_two_state_geometry() {
        let report = two_layer_state_fisher(
            LAMBDA,
            GAMMA,
            &[
                channel(TwoLayerObservation::SurfaceTemperature, 0.1),
                channel(TwoLayerObservation::OceanHeatUptake, 0.2),
            ],
        )
        .unwrap();

        assert_eq!(report.parameter_nullity, 0);
        assert!(report.is_locally_identifiable());
        assert!(report.fisher_condition_number_2.is_finite());
        assert!(report.fisher.determinant() > 0.0);
        assert!(report.fisher[(0, 1)] < 0.0);
    }

    #[test]
    fn toa_plus_ocean_heat_uptake_is_also_full_rank() {
        let report = two_layer_state_fisher(
            LAMBDA,
            GAMMA,
            &[
                channel(TwoLayerObservation::ToaImbalance, 0.2),
                channel(TwoLayerObservation::OceanHeatUptake, 0.2),
            ],
        )
        .unwrap();

        assert_eq!(report.parameter_nullity, 0);
        assert!(report.is_locally_identifiable());
    }

    #[test]
    fn observation_order_does_not_change_the_fisher_metric() {
        let first = two_layer_state_fisher(
            LAMBDA,
            GAMMA,
            &[
                channel(TwoLayerObservation::SurfaceTemperature, 0.1),
                channel(TwoLayerObservation::ToaImbalance, 0.2),
                channel(TwoLayerObservation::OceanHeatUptake, 0.2),
            ],
        )
        .unwrap();
        let permuted = two_layer_state_fisher(
            LAMBDA,
            GAMMA,
            &[
                channel(TwoLayerObservation::OceanHeatUptake, 0.2),
                channel(TwoLayerObservation::SurfaceTemperature, 0.1),
                channel(TwoLayerObservation::ToaImbalance, 0.2),
            ],
        )
        .unwrap();

        for row in 0..2 {
            for col in 0..2 {
                approx_eq(first.fisher[(row, col)], permuted.fisher[(row, col)], 1e-13);
            }
        }
    }

    #[test]
    fn invalid_physics_and_noise_fail_closed() {
        assert!(matches!(
            two_layer_observation_jacobian(
                0.0,
                GAMMA,
                &[TwoLayerObservation::SurfaceTemperature]
            ),
            Err(TwoLayerObservationGeometryError::InvalidClimateFeedback(_))
        ));
        assert!(matches!(
            two_layer_observation_jacobian(
                LAMBDA,
                f64::NAN,
                &[TwoLayerObservation::SurfaceTemperature]
            ),
            Err(TwoLayerObservationGeometryError::InvalidOceanHeatExchange(_))
        ));
        assert!(matches!(
            two_layer_state_fisher(
                LAMBDA,
                GAMMA,
                &[channel(TwoLayerObservation::SurfaceTemperature, 0.0)]
            ),
            Err(TwoLayerObservationGeometryError::Fisher(
                FisherInformationError::InvalidObservationStdDev { .. }
            ))
        ));
    }
}
