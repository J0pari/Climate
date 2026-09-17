use climate_geometric_framework::geometry::{levi_civita_from_jet, MetricJet};
use nalgebra::DMatrix;
use std::f64::consts::PI;

type JetData = (
    DMatrix<f64>,
    Vec<DMatrix<f64>>,
    Vec<Vec<DMatrix<f64>>>,
);

fn zeros(n: usize) -> DMatrix<f64> {
    DMatrix::zeros(n, n)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tol {tolerance})"
    );
}

fn sphere_jet_data(radius: f64, theta: f64) -> JetData {
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
        &[
            0.0,
            0.0,
            0.0,
            2.0 * r2 * (cos_theta * cos_theta - sin_theta * sin_theta),
        ],
    );
    (
        metric,
        vec![dg_theta, zeros(2)],
        vec![
            vec![ddg_theta_theta, zeros(2)],
            vec![zeros(2), zeros(2)],
        ],
    )
}

/// Transform a metric jet under a constant linear coordinate map x = A y.
fn linear_transform_jet(
    metric: &DMatrix<f64>,
    first: &[DMatrix<f64>],
    second: &[Vec<DMatrix<f64>>],
    a: &DMatrix<f64>,
) -> JetData {
    let n = metric.nrows();
    assert_eq!(a.shape(), (n, n));
    let at = a.transpose();
    let transformed_metric = &at * metric * a;

    let mut transformed_first = Vec::with_capacity(n);
    for alpha in 0..n {
        let mut derivative = DMatrix::<f64>::zeros(n, n);
        for k in 0..n {
            derivative += &first[k] * a[(k, alpha)];
        }
        transformed_first.push(&at * derivative * a);
    }

    let mut transformed_second = vec![vec![DMatrix::<f64>::zeros(n, n); n]; n];
    for alpha in 0..n {
        for beta in 0..n {
            let mut derivative = DMatrix::<f64>::zeros(n, n);
            for k in 0..n {
                for l in 0..n {
                    derivative += &second[k][l] * (a[(k, alpha)] * a[(l, beta)]);
                }
            }
            transformed_second[alpha][beta] = &at * derivative * a;
        }
    }

    (
        transformed_metric,
        transformed_first,
        transformed_second,
    )
}

fn assert_linear_coordinate_covariance(a: DMatrix<f64>) {
    let (metric, first, second) = sphere_jet_data(2.0, PI / 3.0);
    let source = MetricJet::new(metric, first.clone(), second.clone()).unwrap();
    let source_geometry = levi_civita_from_jet(&source).unwrap();

    let (metric_y, first_y, second_y) =
        linear_transform_jet(source.metric(), &first, &second, &a);
    let transformed = MetricJet::new(metric_y, first_y, second_y).unwrap();
    let transformed_geometry = levi_civita_from_jet(&transformed).unwrap();
    let inverse_a = a.clone().try_inverse().unwrap();

    assert_close(
        transformed_geometry.scalar_curvature,
        source_geometry.scalar_curvature,
        2e-10,
    );

    let expected_ricci = a.transpose() * &source_geometry.ricci * &a;
    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                transformed_geometry.ricci[(i, j)],
                expected_ricci[(i, j)],
                2e-10,
            );
        }
    }

    // For a constant linear coordinate change, Christoffel symbols transform
    // without the inhomogeneous second-coordinate-derivative term.
    for upper in 0..2 {
        for lower_a in 0..2 {
            for lower_b in 0..2 {
                let mut expected = 0.0;
                for i in 0..2 {
                    for j in 0..2 {
                        for k in 0..2 {
                            expected += inverse_a[(upper, i)]
                                * a[(j, lower_a)]
                                * a[(k, lower_b)]
                                * source_geometry.christoffel(i, j, k);
                        }
                    }
                }
                assert_close(
                    transformed_geometry.christoffel(upper, lower_a, lower_b),
                    expected,
                    3e-10,
                );
            }
        }
    }

    for upper in 0..2 {
        for lower in 0..2 {
            for deriv_a in 0..2 {
                for deriv_b in 0..2 {
                    let mut expected = 0.0;
                    for i in 0..2 {
                        for j in 0..2 {
                            for k in 0..2 {
                                for l in 0..2 {
                                    expected += inverse_a[(upper, i)]
                                        * a[(j, lower)]
                                        * a[(k, deriv_a)]
                                        * a[(l, deriv_b)]
                                        * source_geometry.riemann(i, j, k, l);
                                }
                            }
                        }
                    }
                    assert_close(
                        transformed_geometry.riemann(upper, lower, deriv_a, deriv_b),
                        expected,
                        4e-10,
                    );
                }
            }
        }
    }
}

#[test]
fn sphere_geometry_is_covariant_under_axis_permutation() {
    assert_linear_coordinate_covariance(DMatrix::from_row_slice(
        2,
        2,
        &[0.0, 1.0, 1.0, 0.0],
    ));
}

#[test]
fn sphere_geometry_is_covariant_under_anisotropic_coordinate_rescaling() {
    assert_linear_coordinate_covariance(DMatrix::from_row_slice(
        2,
        2,
        &[0.1, 0.0, 0.0, 3.0],
    ));
}

fn conformal_metric(point: [f64; 2]) -> DMatrix<f64> {
    let x = point[0];
    let y = point[1];
    let phi = 0.15 * x * x + x * y + 0.25 * y * y;
    let scale = (2.0 * phi).exp();
    DMatrix::from_row_slice(2, 2, &[scale, 0.0, 0.0, scale])
}

fn shifted(mut point: [f64; 2], coordinate: usize, amount: f64) -> [f64; 2] {
    point[coordinate] += amount;
    point
}

fn finite_difference_jet(point: [f64; 2], h: f64) -> JetData {
    let metric = conformal_metric(point);
    let mut first = Vec::with_capacity(2);
    for coordinate in 0..2 {
        let m_m2 = conformal_metric(shifted(point, coordinate, -2.0 * h));
        let m_m1 = conformal_metric(shifted(point, coordinate, -h));
        let m_p1 = conformal_metric(shifted(point, coordinate, h));
        let m_p2 = conformal_metric(shifted(point, coordinate, 2.0 * h));
        first.push((m_m2 - m_m1 * 8.0 + m_p1 * 8.0 - m_p2) * (1.0 / (12.0 * h)));
    }

    let mut second = vec![vec![zeros(2); 2]; 2];
    for coordinate in 0..2 {
        let m_m2 = conformal_metric(shifted(point, coordinate, -2.0 * h));
        let m_m1 = conformal_metric(shifted(point, coordinate, -h));
        let m_p1 = conformal_metric(shifted(point, coordinate, h));
        let m_p2 = conformal_metric(shifted(point, coordinate, 2.0 * h));
        second[coordinate][coordinate] =
            (-m_p2 + m_p1 * 16.0 - metric.clone() * 30.0 + m_m1 * 16.0 - m_m2)
                * (1.0 / (12.0 * h * h));
    }

    let f_pp = conformal_metric([point[0] + h, point[1] + h]);
    let f_pm = conformal_metric([point[0] + h, point[1] - h]);
    let f_mp = conformal_metric([point[0] - h, point[1] + h]);
    let f_mm = conformal_metric([point[0] - h, point[1] - h]);
    let mixed = (f_pp - f_pm - f_mp + f_mm) * (1.0 / (4.0 * h * h));
    second[0][1] = mixed.clone();
    second[1][0] = mixed;

    (metric, first, second)
}

#[test]
fn finite_difference_metric_jet_recovers_known_conformal_curvature() {
    let point = [0.31, -0.27];
    let h = 2.0e-4;
    let (metric, first, second) = finite_difference_jet(point, h);
    let jet = MetricJet::new(metric, first, second).unwrap();
    let geometry = levi_civita_from_jet(&jet).unwrap();

    // For g = exp(2 phi)(dx^2 + dy^2), scalar curvature is
    // R = -2 exp(-2 phi) Delta(phi). Here Delta(phi) = 0.3 + 0.5 = 0.8.
    let x = point[0];
    let y = point[1];
    let phi = 0.15 * x * x + x * y + 0.25 * y * y;
    let expected_scalar_curvature = -1.6 * (-2.0 * phi).exp();

    assert_close(
        geometry.scalar_curvature,
        expected_scalar_curvature,
        3e-6,
    );
}
