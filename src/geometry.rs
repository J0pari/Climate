//! Canonical local Levi-Civita geometry from explicit metric jets.
//!
//! A caller supplies a symmetric positive-definite metric and its first/second
//! coordinate derivatives at one point. This module derives the connection and
//! curvature tensors only. It does not construct a climate metric and does not
//! attach physical, probabilistic, or tipping-point meaning to curvature.

use crate::numerics::{conditioning_report, ConditioningError, ConditioningPolicy, ConditioningReport};
use nalgebra::DMatrix;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum GeometryError {
    #[error("metric dimension must be positive")]
    EmptyMetric,
    #[error("metric must be square; got {rows}x{cols}")]
    NonSquareMetric { rows: usize, cols: usize },
    #[error("expected {expected} {order}-derivative entries; got {actual}")]
    DerivativeCount {
        order: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{order} derivative ({a},{b:?}) has shape {rows}x{cols}; expected {expected}x{expected}")]
    DerivativeShape {
        order: &'static str,
        a: usize,
        b: Option<usize>,
        rows: usize,
        cols: usize,
        expected: usize,
    },
    #[error("{component} contains non-finite value at ({row},{col}): {value}")]
    NonFinite {
        component: String,
        row: usize,
        col: usize,
        value: f64,
    },
    #[error("metric is not symmetric at ({row},{col}): {left} vs {right} (tolerance {tolerance})")]
    NonSymmetricMetric {
        row: usize,
        col: usize,
        left: f64,
        right: f64,
        tolerance: f64,
    },
    #[error("{order} metric derivative ({coordinate_a},{coordinate_b:?}) is not symmetric at ({row},{col}): {left} vs {right} (tolerance {tolerance})")]
    NonSymmetricDerivative {
        order: &'static str,
        coordinate_a: usize,
        coordinate_b: Option<usize>,
        row: usize,
        col: usize,
        left: f64,
        right: f64,
        tolerance: f64,
    },
    #[error("mixed second metric derivatives do not commute for coordinates ({coordinate_a},{coordinate_b}) at ({row},{col}): {forward} vs {reverse} (tolerance {tolerance})")]
    NonCommutingMixedDerivative {
        coordinate_a: usize,
        coordinate_b: usize,
        row: usize,
        col: usize,
        forward: f64,
        reverse: f64,
        tolerance: f64,
    },
    #[error("metric is not positive definite and therefore is not Riemannian")]
    NotPositiveDefiniteMetric,
    #[error(
        "metric is numerically rank deficient: rank {numerical_rank}/{dimension}, sigma_min={sigma_min:e}, tolerance={tolerance:e}"
    )]
    NumericallyRankDeficientMetric {
        numerical_rank: usize,
        dimension: usize,
        sigma_min: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Conditioning(#[from] ConditioningError),
    #[error("accepted Riemannian metric unexpectedly became non-invertible")]
    SingularMetric,
}

#[derive(Debug, Clone)]
pub struct MetricJet {
    metric: DMatrix<f64>,
    conditioning: ConditioningReport,
    /// `first[k][(i,j)] = ∂_k g_ij`.
    first: Vec<DMatrix<f64>>,
    /// `second[k][l][(i,j)] = ∂_k ∂_l g_ij`.
    second: Vec<Vec<DMatrix<f64>>>,
}

fn structural_tolerance(scale: f64, n: usize) -> f64 {
    f64::EPSILON * n as f64 * scale.max(1.0)
}

fn matrix_scale(matrix: &DMatrix<f64>) -> f64 {
    matrix.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn validate_symmetric_derivative(
    order: &'static str,
    coordinate_a: usize,
    coordinate_b: Option<usize>,
    derivative: &DMatrix<f64>,
) -> Result<(), GeometryError> {
    let n = derivative.nrows();
    let tolerance = structural_tolerance(matrix_scale(derivative), n);
    for row in 0..n {
        for col in (row + 1)..n {
            let left = derivative[(row, col)];
            let right = derivative[(col, row)];
            if (left - right).abs() > tolerance {
                return Err(GeometryError::NonSymmetricDerivative {
                    order,
                    coordinate_a,
                    coordinate_b,
                    row,
                    col,
                    left,
                    right,
                    tolerance,
                });
            }
        }
    }
    Ok(())
}

impl MetricJet {
    pub fn new(
        metric: DMatrix<f64>,
        first: Vec<DMatrix<f64>>,
        second: Vec<Vec<DMatrix<f64>>>,
    ) -> Result<Self, GeometryError> {
        let n = metric.nrows();
        if n == 0 {
            return Err(GeometryError::EmptyMetric);
        }
        if metric.ncols() != n {
            return Err(GeometryError::NonSquareMetric {
                rows: n,
                cols: metric.ncols(),
            });
        }
        validate_matrix("metric", &metric, n, None)?;

        let symmetry_tolerance = structural_tolerance(matrix_scale(&metric), n);
        for i in 0..n {
            for j in (i + 1)..n {
                let left = metric[(i, j)];
                let right = metric[(j, i)];
                if (left - right).abs() > symmetry_tolerance {
                    return Err(GeometryError::NonSymmetricMetric {
                        row: i,
                        col: j,
                        left,
                        right,
                        tolerance: symmetry_tolerance,
                    });
                }
            }
        }

        if metric.clone().cholesky().is_none() {
            return Err(GeometryError::NotPositiveDefiniteMetric);
        }
        let conditioning = conditioning_report(&matrix_rows(&metric))?;
        if conditioning.is_rank_deficient() {
            return Err(GeometryError::NumericallyRankDeficientMetric {
                numerical_rank: conditioning.numerical_rank,
                dimension: n,
                sigma_min: conditioning.sigma_min,
                tolerance: conditioning.numerical_rank_tolerance,
            });
        }

        if first.len() != n {
            return Err(GeometryError::DerivativeCount {
                order: "first",
                expected: n,
                actual: first.len(),
            });
        }
        for (k, derivative) in first.iter().enumerate() {
            validate_matrix("first derivative", derivative, n, Some((k, None)))?;
            validate_symmetric_derivative("first", k, None, derivative)?;
        }

        if second.len() != n {
            return Err(GeometryError::DerivativeCount {
                order: "second outer",
                expected: n,
                actual: second.len(),
            });
        }
        for (k, row) in second.iter().enumerate() {
            if row.len() != n {
                return Err(GeometryError::DerivativeCount {
                    order: "second inner",
                    expected: n,
                    actual: row.len(),
                });
            }
            for (l, derivative) in row.iter().enumerate() {
                validate_matrix("second derivative", derivative, n, Some((k, Some(l))))?;
                validate_symmetric_derivative("second", k, Some(l), derivative)?;
            }
        }

        for coordinate_a in 0..n {
            for coordinate_b in (coordinate_a + 1)..n {
                let forward = &second[coordinate_a][coordinate_b];
                let reverse = &second[coordinate_b][coordinate_a];
                let scale = matrix_scale(forward).max(matrix_scale(reverse));
                let tolerance = structural_tolerance(scale, n);
                for row in 0..n {
                    for col in 0..n {
                        let forward_value = forward[(row, col)];
                        let reverse_value = reverse[(row, col)];
                        if (forward_value - reverse_value).abs() > tolerance {
                            return Err(GeometryError::NonCommutingMixedDerivative {
                                coordinate_a,
                                coordinate_b,
                                row,
                                col,
                                forward: forward_value,
                                reverse: reverse_value,
                                tolerance,
                            });
                        }
                    }
                }
            }
        }

        Ok(Self {
            metric,
            conditioning,
            first,
            second,
        })
    }

    pub fn dimension(&self) -> usize {
        self.metric.nrows()
    }

    pub fn metric(&self) -> &DMatrix<f64> {
        &self.metric
    }

    pub fn conditioning_report(&self) -> &ConditioningReport {
        &self.conditioning
    }

    pub fn new_with_conditioning_policy(
        metric: DMatrix<f64>,
        first: Vec<DMatrix<f64>>,
        second: Vec<Vec<DMatrix<f64>>>,
        policy: ConditioningPolicy,
    ) -> Result<Self, GeometryError> {
        let policy = ConditioningPolicy::new(policy.max_condition_number_2)?;
        let jet = Self::new(metric, first, second)?;
        if jet.conditioning.condition_number_2 > policy.max_condition_number_2 {
            return Err(GeometryError::Conditioning(
                ConditioningError::ConditionNumberExceeded {
                    measured: jet.conditioning.condition_number_2,
                    maximum: policy.max_condition_number_2,
                },
            ));
        }
        Ok(jet)
    }

    pub fn first_derivative(&self, coordinate: usize) -> &DMatrix<f64> {
        &self.first[coordinate]
    }
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn validate_matrix(
    component: &'static str,
    matrix: &DMatrix<f64>,
    n: usize,
    derivative: Option<(usize, Option<usize>)>,
) -> Result<(), GeometryError> {
    if matrix.shape() != (n, n) {
        let (a, b) = derivative.unwrap_or((0, None));
        return Err(GeometryError::DerivativeShape {
            order: if derivative.is_some() { component } else { "metric" },
            a,
            b,
            rows: matrix.nrows(),
            cols: matrix.ncols(),
            expected: n,
        });
    }
    for row in 0..n {
        for col in 0..n {
            let value = matrix[(row, col)];
            if !value.is_finite() {
                return Err(GeometryError::NonFinite {
                    component: component.to_owned(),
                    row,
                    col,
                    value,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct GeometryAtPoint {
    dimension: usize,
    pub inverse_metric: DMatrix<f64>,
    christoffel: Vec<f64>,
    riemann: Vec<f64>,
    pub ricci: DMatrix<f64>,
    pub scalar_curvature: f64,
}

impl GeometryAtPoint {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn christoffel(&self, upper: usize, lower_a: usize, lower_b: usize) -> f64 {
        self.christoffel[idx3(self.dimension, upper, lower_a, lower_b)]
    }

    /// Riemann convention: `R^i_{jkl}` where `k,l` are derivative indices.
    pub fn riemann(&self, upper: usize, lower: usize, deriv_a: usize, deriv_b: usize) -> f64 {
        self.riemann[idx4(self.dimension, upper, lower, deriv_a, deriv_b)]
    }
}

fn idx3(n: usize, a: usize, b: usize, c: usize) -> usize {
    (a * n + b) * n + c
}

fn idx4(n: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * n + b) * n + c) * n + d
}

/// Compute the Levi-Civita connection and curvature from an explicit metric jet.
pub fn levi_civita_from_jet(jet: &MetricJet) -> Result<GeometryAtPoint, GeometryError> {
    let n = jet.dimension();
    let inverse = jet
        .metric
        .clone()
        .cholesky()
        .ok_or(GeometryError::SingularMetric)?
        .inverse();

    let mut christoffel = vec![0.0; n * n * n];
    for upper in 0..n {
        for lower_a in 0..n {
            for lower_b in 0..n {
                let mut value = 0.0;
                for contracted in 0..n {
                    let derivative = jet.first[lower_a][(contracted, lower_b)]
                        + jet.first[lower_b][(contracted, lower_a)]
                        - jet.first[contracted][(lower_a, lower_b)];
                    value += 0.5 * inverse[(upper, contracted)] * derivative;
                }
                christoffel[idx3(n, upper, lower_a, lower_b)] = value;
            }
        }
    }

    let inverse_derivatives: Vec<DMatrix<f64>> = jet
        .first
        .iter()
        .map(|dg| -(&inverse * dg * &inverse))
        .collect();

    let mut d_christoffel = vec![0.0; n * n * n * n];
    for deriv in 0..n {
        for upper in 0..n {
            for lower_a in 0..n {
                for lower_b in 0..n {
                    let mut value = 0.0;
                    for contracted in 0..n {
                        let first_combination = jet.first[lower_a][(contracted, lower_b)]
                            + jet.first[lower_b][(contracted, lower_a)]
                            - jet.first[contracted][(lower_a, lower_b)];
                        let second_combination = jet.second[deriv][lower_a]
                            [(contracted, lower_b)]
                            + jet.second[deriv][lower_b][(contracted, lower_a)]
                            - jet.second[deriv][contracted][(lower_a, lower_b)];
                        value += 0.5
                            * (inverse_derivatives[deriv][(upper, contracted)]
                                * first_combination
                                + inverse[(upper, contracted)] * second_combination);
                    }
                    d_christoffel[idx4(n, deriv, upper, lower_a, lower_b)] = value;
                }
            }
        }
    }

    let mut riemann = vec![0.0; n * n * n * n];
    for upper in 0..n {
        for lower in 0..n {
            for deriv_a in 0..n {
                for deriv_b in 0..n {
                    let mut value = d_christoffel[idx4(n, deriv_a, upper, deriv_b, lower)]
                        - d_christoffel[idx4(n, deriv_b, upper, deriv_a, lower)];
                    for contracted in 0..n {
                        value += christoffel[idx3(n, upper, deriv_a, contracted)]
                            * christoffel[idx3(n, contracted, deriv_b, lower)]
                            - christoffel[idx3(n, upper, deriv_b, contracted)]
                                * christoffel[idx3(n, contracted, deriv_a, lower)];
                    }
                    riemann[idx4(n, upper, lower, deriv_a, deriv_b)] = value;
                }
            }
        }
    }

    let mut ricci = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut value = 0.0;
            for contracted in 0..n {
                value += riemann[idx4(n, contracted, i, contracted, j)];
            }
            ricci[(i, j)] = value;
        }
    }

    let mut scalar_curvature = 0.0;
    for i in 0..n {
        for j in 0..n {
            scalar_curvature += inverse[(i, j)] * ricci[(i, j)];
        }
    }

    Ok(GeometryAtPoint {
        dimension: n,
        inverse_metric: inverse,
        christoffel,
        riemann,
        ricci,
        scalar_curvature,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn zeros(n: usize) -> DMatrix<f64> {
        DMatrix::zeros(n, n)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tol {tolerance})"
        );
    }

    #[test]
    fn cartesian_plane_is_flat() {
        let jet = MetricJet::new(
            DMatrix::identity(2, 2),
            vec![zeros(2), zeros(2)],
            vec![vec![zeros(2), zeros(2)], vec![zeros(2), zeros(2)]],
        )
        .unwrap();
        let geometry = levi_civita_from_jet(&jet).unwrap();
        assert!(geometry.christoffel.iter().all(|value| value.abs() < 1e-14));
        assert!(geometry.riemann.iter().all(|value| value.abs() < 1e-14));
        assert_close(geometry.scalar_curvature, 0.0, 1e-14);
    }

    #[test]
    fn polar_coordinates_have_connection_but_zero_curvature() {
        let r = 2.0;
        let metric = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, r * r]);
        let dg_r = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 2.0 * r]);
        let ddg_rr = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 2.0]);
        let jet = MetricJet::new(
            metric,
            vec![dg_r, zeros(2)],
            vec![vec![ddg_rr, zeros(2)], vec![zeros(2), zeros(2)]],
        )
        .unwrap();
        let geometry = levi_civita_from_jet(&jet).unwrap();

        assert_close(geometry.christoffel(0, 1, 1), -r, 1e-12);
        assert_close(geometry.christoffel(1, 0, 1), 1.0 / r, 1e-12);
        assert_close(geometry.christoffel(1, 1, 0), 1.0 / r, 1e-12);
        assert!(geometry.riemann.iter().all(|value| value.abs() < 1e-11));
        assert_close(geometry.scalar_curvature, 0.0, 1e-11);
    }

    #[test]
    fn nonlinear_coordinate_map_requires_inhomogeneous_connection_term() {
        let u = 0.4_f64;
        let metric = DMatrix::from_row_slice(2, 2, &[1.0 + u * u, u, u, 1.0]);
        let dg_u = DMatrix::from_row_slice(2, 2, &[2.0 * u, 1.0, 1.0, 0.0]);
        let ddg_uu = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 0.0]);
        let scale = matrix_scale(&metric)
            .max(matrix_scale(&dg_u))
            .max(matrix_scale(&ddg_uu));
        let tolerance = 128.0 * f64::EPSILON * scale.max(1.0);
        let jet = MetricJet::new(
            metric,
            vec![dg_u, zeros(2)],
            vec![vec![ddg_uu, zeros(2)], vec![zeros(2), zeros(2)]],
        )
        .unwrap();
        let geometry = levi_civita_from_jet(&jet).unwrap();

        for upper in 0..2 {
            for lower_a in 0..2 {
                for lower_b in 0..2 {
                    let expected = if (upper, lower_a, lower_b) == (1, 0, 0) {
                        1.0
                    } else {
                        0.0
                    };
                    assert_close(
                        geometry.christoffel(upper, lower_a, lower_b),
                        expected,
                        tolerance,
                    );
                }
            }
        }

        let homogeneous_only_prediction = 0.0_f64;
        assert!(
            (geometry.christoffel(1, 0, 0) - homogeneous_only_prediction).abs()
                > 1_000_000.0 * tolerance,
            "omitting the inhomogeneous connection term was not detected"
        );
        for upper in 0..2 {
            for lower in 0..2 {
                for deriv_a in 0..2 {
                    for deriv_b in 0..2 {
                        assert_close(
                            geometry.riemann(upper, lower, deriv_a, deriv_b),
                            0.0,
                            tolerance,
                        );
                    }
                }
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert_close(geometry.ricci[(i, j)], 0.0, tolerance);
            }
        }
        assert_close(geometry.scalar_curvature, 0.0, tolerance);
    }

    #[test]
    fn radius_two_sphere_has_known_curvature_and_structural_identities() {
        let radius = 2.0;
        let theta = PI / 3.0;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let r2 = radius * radius;
        let metric = DMatrix::from_row_slice(
            2,
            2,
            &[r2, 0.0, 0.0, r2 * sin_theta * sin_theta],
        );
        let dg_theta = DMatrix::from_row_slice(
            2,
            2,
            &[0.0, 0.0, 0.0, 2.0 * r2 * sin_theta * cos_theta],
        );
        let ddg_theta_theta = DMatrix::from_row_slice(
            2,
            2,
            &[0.0, 0.0, 0.0, 2.0 * r2 * (cos_theta * cos_theta - sin_theta * sin_theta)],
        );
        let jet = MetricJet::new(
            metric.clone(),
            vec![dg_theta, zeros(2)],
            vec![
                vec![ddg_theta_theta, zeros(2)],
                vec![zeros(2), zeros(2)],
            ],
        )
        .unwrap();
        let geometry = levi_civita_from_jet(&jet).unwrap();

        assert_close(geometry.scalar_curvature, 0.5, 1e-10);
        for i in 0..2 {
            for j in 0..2 {
                assert_close(geometry.ricci[(i, j)], 0.25 * metric[(i, j)], 1e-10);
            }
        }

        for deriv in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    let mut residual = jet.first_derivative(deriv)[(i, j)];
                    for contracted in 0..2 {
                        residual -= geometry.christoffel(contracted, deriv, i)
                            * metric[(contracted, j)];
                        residual -= geometry.christoffel(contracted, deriv, j)
                            * metric[(i, contracted)];
                    }
                    assert_close(residual, 0.0, 1e-10);
                }
            }
        }

        for upper in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        let cyclic = geometry.riemann(upper, j, k, l)
                            + geometry.riemann(upper, k, l, j)
                            + geometry.riemann(upper, l, j, k);
                        assert_close(cyclic, 0.0, 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_metric_fails_closed() {
        let nonsymmetric = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
        assert!(matches!(
            MetricJet::new(
                nonsymmetric,
                vec![zeros(2), zeros(2)],
                vec![vec![zeros(2), zeros(2)], vec![zeros(2), zeros(2)]],
            ),
            Err(GeometryError::NonSymmetricMetric { .. })
        ));

        assert!(matches!(
            MetricJet::new(
                DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]),
                vec![zeros(2), zeros(2)],
                vec![vec![zeros(2), zeros(2)], vec![zeros(2), zeros(2)]],
            ),
            Err(GeometryError::NotPositiveDefiniteMetric)
        ));

        assert!(matches!(
            MetricJet::new(
                DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]),
                vec![zeros(2), zeros(2)],
                vec![vec![zeros(2), zeros(2)], vec![zeros(2), zeros(2)]],
            ),
            Err(GeometryError::NotPositiveDefiniteMetric)
        ));
    }

    #[test]
    fn conditioning_is_measured_and_policy_is_explicit() {
        let metric = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0e-8]);
        let first = vec![zeros(2), zeros(2)];
        let second = vec![vec![zeros(2), zeros(2)], vec![zeros(2), zeros(2)]];

        let jet = MetricJet::new(metric.clone(), first.clone(), second.clone()).unwrap();
        assert!(jet.conditioning_report().condition_number_2 > 9.0e7);

        let strict = ConditioningPolicy::new(1.0e6).unwrap();
        assert!(matches!(
            MetricJet::new_with_conditioning_policy(metric, first, second, strict),
            Err(GeometryError::Conditioning(
                ConditioningError::ConditionNumberExceeded { .. }
            ))
        ));
    }

    #[test]
    fn impossible_metric_derivative_jets_fail_closed() {
        let asymmetric_first = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        assert!(matches!(
            MetricJet::new(
                DMatrix::identity(2, 2),
                vec![asymmetric_first, zeros(2)],
                vec![vec![zeros(2), zeros(2)], vec![zeros(2), zeros(2)]],
            ),
            Err(GeometryError::NonSymmetricDerivative { order: "first", .. })
        ));

        let asymmetric_second = DMatrix::from_row_slice(2, 2, &[0.0, 2.0, 0.0, 0.0]);
        assert!(matches!(
            MetricJet::new(
                DMatrix::identity(2, 2),
                vec![zeros(2), zeros(2)],
                vec![
                    vec![asymmetric_second, zeros(2)],
                    vec![zeros(2), zeros(2)],
                ],
            ),
            Err(GeometryError::NonSymmetricDerivative { order: "second", .. })
        ));

        let mixed_forward = DMatrix::from_diagonal_element(2, 2, 1.0);
        assert!(matches!(
            MetricJet::new(
                DMatrix::identity(2, 2),
                vec![zeros(2), zeros(2)],
                vec![
                    vec![zeros(2), mixed_forward],
                    vec![zeros(2), zeros(2)],
                ],
            ),
            Err(GeometryError::NonCommutingMixedDerivative { .. })
        ));
    }
}
