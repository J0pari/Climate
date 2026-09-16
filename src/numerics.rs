//! Numerical verification primitives used by Climate's canonical Rust surface.
//!
//! This module deliberately separates measured numerical properties from policy.
//! A condition number is computed from the matrix supplied by the caller; policy
//! decides whether the measured value is acceptable. No fabricated fallback
//! values are permitted.

use nalgebra::DMatrix;
use thiserror::Error;

/// Measured 2-norm conditioning information for a matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditioningReport {
    pub rows: usize,
    pub cols: usize,
    pub sigma_max: f64,
    pub sigma_min: f64,
    pub numerical_rank: usize,
    pub numerical_rank_tolerance: f64,
    pub condition_number_2: f64,
}

impl ConditioningReport {
    pub fn is_rank_deficient(&self) -> bool {
        self.numerical_rank < self.rows.min(self.cols)
    }
}

/// Explicit caller-owned policy for accepting a measured condition number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditioningPolicy {
    pub max_condition_number_2: f64,
}

impl ConditioningPolicy {
    pub fn new(max_condition_number_2: f64) -> Result<Self, ConditioningError> {
        if !max_condition_number_2.is_finite() || max_condition_number_2 < 1.0 {
            return Err(ConditioningError::InvalidPolicy {
                max_condition_number_2,
            });
        }
        Ok(Self {
            max_condition_number_2,
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ConditioningError {
    #[error("matrix must contain at least one row and one column")]
    EmptyMatrix,
    #[error("matrix row {row} has {actual} columns; expected {expected}")]
    RaggedMatrix {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("matrix contains a non-finite value at row {row}, column {col}: {value}")]
    NonFiniteValue {
        row: usize,
        col: usize,
        value: f64,
    },
    #[error("maximum permitted 2-norm condition number must be finite and >= 1; got {max_condition_number_2}")]
    InvalidPolicy { max_condition_number_2: f64 },
    #[error(
        "matrix is numerically rank deficient: rank {numerical_rank}/{full_rank}, sigma_min={sigma_min:e}, tolerance={tolerance:e}"
    )]
    RankDeficient {
        numerical_rank: usize,
        full_rank: usize,
        sigma_min: f64,
        tolerance: f64,
    },
    #[error(
        "measured 2-norm condition number {measured:e} exceeds policy maximum {maximum:e}"
    )]
    ConditionNumberExceeded { measured: f64, maximum: f64 },
}

/// Compute an SVD-based 2-norm conditioning report from the supplied matrix.
///
/// The numerical-rank tolerance follows the standard scale-sensitive form
/// `eps * max(m, n) * sigma_max`, so scaling the matrix scales the tolerance
/// rather than introducing an arbitrary absolute singular-value cutoff.
pub fn conditioning_report(matrix: &[Vec<f64>]) -> Result<ConditioningReport, ConditioningError> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(ConditioningError::EmptyMatrix);
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut data = Vec::with_capacity(rows * cols);

    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != cols {
            return Err(ConditioningError::RaggedMatrix {
                row: row_index,
                expected: cols,
                actual: row.len(),
            });
        }
        for (col_index, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ConditioningError::NonFiniteValue {
                    row: row_index,
                    col: col_index,
                    value,
                });
            }
            data.push(value);
        }
    }

    let singular_values = DMatrix::<f64>::from_row_slice(rows, cols, &data)
        .svd(false, false)
        .singular_values;

    // Non-empty input guarantees at least one singular value.
    let sigma_max = singular_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let sigma_min = singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let numerical_rank_tolerance = f64::EPSILON * rows.max(cols) as f64 * sigma_max;
    let numerical_rank = singular_values
        .iter()
        .filter(|&&sigma| sigma > numerical_rank_tolerance)
        .count();

    let condition_number_2 = if sigma_min <= numerical_rank_tolerance {
        f64::INFINITY
    } else {
        sigma_max / sigma_min
    };

    Ok(ConditioningReport {
        rows,
        cols,
        sigma_max,
        sigma_min,
        numerical_rank,
        numerical_rank_tolerance,
        condition_number_2,
    })
}

/// Measure conditioning and enforce an explicit caller-supplied policy.
pub fn require_well_conditioned(
    matrix: &[Vec<f64>],
    policy: ConditioningPolicy,
) -> Result<ConditioningReport, ConditioningError> {
    // Re-validate the public struct in case a caller constructed it directly.
    let policy = ConditioningPolicy::new(policy.max_condition_number_2)?;
    let report = conditioning_report(matrix)?;
    let full_rank = report.rows.min(report.cols);

    if report.is_rank_deficient() {
        return Err(ConditioningError::RankDeficient {
            numerical_rank: report.numerical_rank,
            full_rank,
            sigma_min: report.sigma_min,
            tolerance: report.numerical_rank_tolerance,
        });
    }

    if report.condition_number_2 > policy.max_condition_number_2 {
        return Err(ConditioningError::ConditionNumberExceeded {
            measured: report.condition_number_2,
            maximum: policy.max_condition_number_2,
        });
    }

    Ok(report)
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
    fn identity_is_measured_as_unit_conditioned() {
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let report = conditioning_report(&matrix).unwrap();
        assert_eq!(report.numerical_rank, 3);
        approx_eq(report.sigma_max, 1.0, 1e-12);
        approx_eq(report.sigma_min, 1.0, 1e-12);
        approx_eq(report.condition_number_2, 1.0, 1e-12);
    }

    #[test]
    fn measured_condition_number_changes_with_matrix_data() {
        let well = vec![vec![1.0, 0.0], vec![0.0, 0.5]];
        let ill = vec![vec![1.0, 0.0], vec![0.0, 1e-6]];

        let well_report = conditioning_report(&well).unwrap();
        let ill_report = conditioning_report(&ill).unwrap();

        approx_eq(well_report.condition_number_2, 2.0, 1e-12);
        approx_eq(ill_report.condition_number_2, 1e6, 1e-9);
        assert!(ill_report.condition_number_2 > well_report.condition_number_2);
    }

    #[test]
    fn condition_number_is_scale_invariant() {
        let base = vec![vec![4.0, 0.0], vec![0.0, 2.0]];
        let scaled = vec![vec![4.0e120, 0.0], vec![0.0, 2.0e120]];
        let base_report = conditioning_report(&base).unwrap();
        let scaled_report = conditioning_report(&scaled).unwrap();
        approx_eq(
            base_report.condition_number_2,
            scaled_report.condition_number_2,
            1e-12,
        );
    }

    #[test]
    fn rank_deficiency_is_derived_from_singular_values() {
        let singular = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let report = conditioning_report(&singular).unwrap();
        assert_eq!(report.numerical_rank, 1);
        assert!(report.condition_number_2.is_infinite());

        let error = require_well_conditioned(
            &singular,
            ConditioningPolicy::new(1e8).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(error, ConditioningError::RankDeficient { .. }));
    }

    #[test]
    fn policy_is_separate_from_measurement() {
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 1e-4]];
        let report = conditioning_report(&matrix).unwrap();
        approx_eq(report.condition_number_2, 1e4, 1e-10);

        assert!(require_well_conditioned(
            &matrix,
            ConditioningPolicy::new(1e5).unwrap(),
        )
        .is_ok());
        assert!(matches!(
            require_well_conditioned(
                &matrix,
                ConditioningPolicy::new(1e3).unwrap(),
            ),
            Err(ConditioningError::ConditionNumberExceeded { .. })
        ));
    }

    #[test]
    fn malformed_and_non_finite_inputs_fail_closed() {
        assert!(matches!(
            conditioning_report(&[]),
            Err(ConditioningError::EmptyMatrix)
        ));
        assert!(matches!(
            conditioning_report(&[vec![1.0, 2.0], vec![3.0]]),
            Err(ConditioningError::RaggedMatrix { .. })
        ));
        assert!(matches!(
            conditioning_report(&[vec![1.0, f64::NAN]]),
            Err(ConditioningError::NonFiniteValue { .. })
        ));
    }
}
