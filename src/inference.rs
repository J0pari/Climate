//! Canonical Fisher-information primitives for Gaussian observation models.
//!
//! The mathematical layer accepts the local Jacobian of a model's predicted
//! observation mean with respect to its parameters. How that Jacobian is
//! obtained (analytic sensitivities, AD, or an independently verified numerical
//! derivative) belongs to the model layer.
//!
//! For independent Gaussian observation errors with declared standard deviations
//! `sigma_r`, the exact expected Fisher information for mean parameters is
//!
//!     I(theta) = J(theta)^T diag(1 / sigma_r^2) J(theta).
//!
//! No regularization is inserted here. Rank deficiency is scientific/numerical
//! information about local identifiability, not a condition to hide with a
//! successful-looking inverse.

use nalgebra::DMatrix;
use thiserror::Error;

use crate::numerics::{conditioning_report, ConditioningError, ConditioningReport};

#[derive(Debug, Clone, PartialEq)]
pub struct FisherInformationReport {
    pub fisher: DMatrix<f64>,
    pub whitened_mean_jacobian: DMatrix<f64>,
    pub jacobian_conditioning: ConditioningReport,
    pub observation_count: usize,
    pub parameter_count: usize,
    pub parameter_nullity: usize,
    pub fisher_condition_number_2: f64,
}

impl FisherInformationReport {
    /// Whether the supplied local observation model identifies every parameter
    /// direction at the numerical-rank tolerance used by the canonical SVD
    /// measurement layer.
    pub fn is_locally_identifiable(&self) -> bool {
        self.parameter_nullity == 0
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum FisherInformationError {
    #[error("mean-model Jacobian must contain at least one observation row and one parameter column")]
    EmptyJacobian,
    #[error(
        "observation-noise vector has length {actual}; expected one standard deviation for each of {expected} Jacobian rows"
    )]
    NoiseDimensionMismatch { expected: usize, actual: usize },
    #[error("observation standard deviation at row {row} must be finite and > 0; got {value}")]
    InvalidObservationStdDev { row: usize, value: f64 },
    #[error("mean-model Jacobian contains a non-finite value at row {row}, column {col}: {value}")]
    NonFiniteJacobian {
        row: usize,
        col: usize,
        value: f64,
    },
    #[error(transparent)]
    Conditioning(#[from] ConditioningError),
}

/// Construct exact expected Fisher information for a local Gaussian mean model.
///
/// `mean_jacobian[(r, p)] = d mean_r / d theta_p` and `observation_stddev[r]`
/// is the known/declared standard deviation for observation `r`.
pub fn gaussian_mean_fisher(
    mean_jacobian: &DMatrix<f64>,
    observation_stddev: &[f64],
) -> Result<FisherInformationReport, FisherInformationError> {
    let observations = mean_jacobian.nrows();
    let parameters = mean_jacobian.ncols();
    if observations == 0 || parameters == 0 {
        return Err(FisherInformationError::EmptyJacobian);
    }
    if observation_stddev.len() != observations {
        return Err(FisherInformationError::NoiseDimensionMismatch {
            expected: observations,
            actual: observation_stddev.len(),
        });
    }

    let mut whitened = mean_jacobian.clone();
    for row in 0..observations {
        let sigma = observation_stddev[row];
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(FisherInformationError::InvalidObservationStdDev {
                row,
                value: sigma,
            });
        }
        for col in 0..parameters {
            let value = mean_jacobian[(row, col)];
            if !value.is_finite() {
                return Err(FisherInformationError::NonFiniteJacobian {
                    row,
                    col,
                    value,
                });
            }
            whitened[(row, col)] = value / sigma;
        }
    }

    let fisher = whitened.transpose() * &whitened;
    let rows = (0..observations)
        .map(|row| (0..parameters).map(|col| whitened[(row, col)]).collect())
        .collect::<Vec<Vec<f64>>>();
    let jacobian_conditioning = conditioning_report(&rows)?;
    let parameter_nullity = parameters.saturating_sub(jacobian_conditioning.numerical_rank);

    // When parameter directions lie in the observation-model nullspace, the
    // Fisher matrix has exact/numerical zero eigenvalues even if the rectangular
    // Jacobian's returned SVD only lists min(m,n) positive singular values.
    let fisher_condition_number_2 = if parameter_nullity > 0 {
        f64::INFINITY
    } else {
        jacobian_conditioning.condition_number_2.powi(2)
    };

    Ok(FisherInformationReport {
        fisher,
        whitened_mean_jacobian: whitened,
        jacobian_conditioning,
        observation_count: observations,
        parameter_count: parameters,
        parameter_nullity,
        fisher_condition_number_2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64, relative: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= relative * scale,
            "left={left:e}, right={right:e}, relative={relative:e}"
        );
    }

    #[test]
    fn fisher_is_whitened_jacobian_gram_matrix() {
        let jacobian = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let report = gaussian_mean_fisher(&jacobian, &[2.0, 1.0]).unwrap();

        approx_eq(report.fisher[(0, 0)], 0.25, 1e-12);
        approx_eq(report.fisher[(1, 1)], 1.0, 1e-12);
        approx_eq(report.fisher[(0, 1)], 0.0, 1e-12);
        assert_eq!(report.parameter_nullity, 0);
        assert!(report.is_locally_identifiable());
        approx_eq(report.fisher_condition_number_2, 4.0, 1e-12);
    }

    #[test]
    fn duplicated_parameter_directions_remain_identifiably_singular() {
        let jacobian = DMatrix::from_row_slice(
            3,
            2,
            &[
                1.0, 1.0,
                2.0, 2.0,
                -1.0, -1.0,
            ],
        );
        let report = gaussian_mean_fisher(&jacobian, &[1.0, 1.0, 1.0]).unwrap();

        assert_eq!(report.jacobian_conditioning.numerical_rank, 1);
        assert_eq!(report.parameter_nullity, 1);
        assert!(!report.is_locally_identifiable());
        assert!(report.fisher_condition_number_2.is_infinite());
    }

    #[test]
    fn too_few_observations_expose_parameter_nullspace() {
        let jacobian = DMatrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        let report = gaussian_mean_fisher(&jacobian, &[1.0]).unwrap();

        assert_eq!(report.jacobian_conditioning.numerical_rank, 1);
        assert_eq!(report.parameter_nullity, 2);
        assert!(report.fisher_condition_number_2.is_infinite());
    }

    #[test]
    fn increasing_observation_noise_reduces_information_quadratically() {
        let jacobian = DMatrix::from_row_slice(2, 1, &[1.0, 2.0]);
        let base = gaussian_mean_fisher(&jacobian, &[1.0, 1.0]).unwrap();
        let noisy = gaussian_mean_fisher(&jacobian, &[2.0, 2.0]).unwrap();

        approx_eq(noisy.fisher[(0, 0)], base.fisher[(0, 0)] / 4.0, 1e-12);
    }

    #[test]
    fn invalid_noise_and_non_finite_jacobians_fail_closed() {
        let jacobian = DMatrix::from_row_slice(1, 1, &[1.0]);
        assert!(matches!(
            gaussian_mean_fisher(&jacobian, &[]),
            Err(FisherInformationError::NoiseDimensionMismatch { .. })
        ));
        assert!(matches!(
            gaussian_mean_fisher(&jacobian, &[0.0]),
            Err(FisherInformationError::InvalidObservationStdDev { .. })
        ));
        assert!(matches!(
            gaussian_mean_fisher(&DMatrix::from_row_slice(1, 1, &[f64::NAN]), &[1.0]),
            Err(FisherInformationError::NonFiniteJacobian { .. })
        ));
    }
}
