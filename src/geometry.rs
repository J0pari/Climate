//! Canonical local Levi-Civita geometry from explicit metric jets.
//!
//! This module computes geometric tensors from a symmetric nondegenerate metric
//! and its first/second coordinate derivatives at one point. It deliberately
//! does not construct a climate metric and does not assign physical meaning to
//! curvature. Those are separate scientific hypotheses.

use nalgebra::DMatrix;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum GeometryError {
    #[error("metric dimension must be positive")]
    EmptyMetric,
    #[error("metric must be square; got {rows}x{cols}")]
    NonSquareMetric { rows: usize, cols: usize },
    #[error("expected {expected} first-derivative matrices; got {actual}")]
    FirstDerivativeCount { expected: usize, actual: usize },
    #[error("expected {expected} second-derivative rows; got {actual}")]
    SecondDerivativeOuterCount { expected: usize, actual: usize },
    #[error("second-derivative row {deriv_a} expected {expected} matrices; got {actual}")]
    SecondDerivativeInnerCount {
        deriv_a: usize,
        expected: usize,
        actual: usize,
    },
    #[error("{kind} derivative ({deriv_a},{deriv_b:?}) has shape {rows}x{cols}; expected {expected}x{expected}")]
    DerivativeShape {
        kind: &'static str,
        deriv_a: usize,
        deriv_b: Option<usize>,
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
    #[error("metric is singular or numerically non-invertible")]
    SingularMetric,
}

#[derive(Debug, Clone)]
pub struct MetricJet {
    metric: DMatrix<f64>,
    /// `first[k][(i,j)] = ∂_k g_ij`.
    first: Vec<DMatrix<f64>>,
    /// `second[k][l][(i,j)] = ∂_k ∂_l g_ij`.
    second: Vec<Vec<DMatrix<f64>>>,
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
                rows: metric.nrows(),
                cols: metric.ncols(),
            });
        }
        validate_finite("metric", &metric)?;

        let scale = metric.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let symmetry_tolerance = f64::EPSILON * n as f64 * scale.max(1.0);
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

        if first.len() != n {
            return Err(GeometryError::FirstDerivativeCount {
                expected: n,
                actual: first.len(),
            });
        }
        for (k, derivative) in first.iter().enumerate() {
            validate_shape("first", k, None, derivative, n)?;
            validate_finite(&format!("first derivative {k}"), derivative)?;
        }

        if second.len() != n {
            return Err(GeometryError::SecondDerivativeOuterCount {
                expected: n,
                actual: second.len(),
            });
        }
        for (k, row) in second.iter().enumerate() {
            if row.len() != n {
                return Err(GeometryError::SecondDerivativeInnerCount {
                    deriv_a: k,
                    expected: n,
                    actual: row.len(),
                });
            }
            for (l, derivative) in row.iter().enumerate() {
                validate_shape("second", k, Some(l), derivative, n)?;
                validate_finite(&format!("second derivative ({k},{l})"), derivative)?;
            }
        }

        Ok(Self {
            metric,
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

    pub fn first_derivative(&self, coordinate: usize) -> &DMatrix<f64> {
        &self.first[coordinate]
    }
}

fn validate_shape(
    kind: &'static str,
    deriv_a: usize,
    deriv_b: Option<usize>,
    matrix: &DMatrix<f64>,
    expected: usize,
) -> Result<(), GeometryError> {
    if matrix.shape() != (expected, expected) {
        return Err(GeometryError::DerivativeShape {
            kind,
            deriv_a,
            deriv_b,
            rows: matrix.nrows(),
            cols: matrix.ncols(),
            expected,
        });
    }
    Ok(())
}

fn validate_finite(component: &str, matrix: &DMatrix<f64>) -> Result<(), GeometryError> {
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
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
        .try_inverse()
        .ok_or(GeometryError::SingularMetric)?;

    let mut christoffel = vec![0.0; n * n * n];
    for upper in 0..n {
        for lower_a in 0..n {
            for lower_b in 0..n {
                let mut value = 0.0;
                for contracted in 0..n {
                    let metric_derivative = jet.first[lower_a][(contracted, lower_b)]
                        + jet.first[lower_b][(contracted, lower_a)]
                        - jet.first[contracted][(lower_a, lower_b)];
                    value += 0.5 * inverse[(upper, contracted)] * metric_derivative;
                }
                christoffel[idx3(n, upper, lower_a, lower_b)] = value;
            }
        }
    }

    let inverse_derivatives: Vec<DMatrix<f64>> = jet
        .first
        .iter()
        .map(|derivative| -(&inverse * derivative * &inverse))
        .collect();

    // d_christoffel[p,i,j,k] = ∂_p Γ^i_jk.
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
    for lower_a in 0..n {
        for lower_b in 0..n {
            ricci[(lower_a, lower_b)] = (0..n)
                .map(|contracted| riemann[idx4(n, contracted, lower_a, contracted, lower_b)])
                .sum();
        }
    }

    let scalar_curvature = (0..n)
        .flat_map(|i| (0..n).map(move |j| inverse[(i, j)] * ricci[(i, j)]))
        .sum();

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
        let metric = DMatrix::identity(2, 2);
        let jet = MetricJet::new(
            metric,
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

        // Metric compatibility ∇_k g_ij = 0.
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

        // First Bianchi identity R^i_jkl + R^i_klj + R^i_ljk = 0.
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

        let singular = MetricJet::new(
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]),
            vec![zeros(2), zeros(2)],
            vec![vec![zeros(2), zeros(2)], vec![zeros(2), zeros(2)]],
        )
        .unwrap();
        assert!(matches!(
            levi_civita_from_jet(&singular),
            Err(GeometryError::SingularMetric)
        ));
    }
}
