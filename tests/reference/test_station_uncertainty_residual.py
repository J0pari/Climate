from __future__ import annotations

import unittest

import numpy as np

from reference.station_uncertainty_residual import (
    CORRELATED_COVARIANCE_ASSUMPTION,
    COVARIANCE_ASSUMPTION,
    CorrelatedObservationUncertainty,
    DiagonalObservationUncertainty,
    correlated_uncertainty_residual,
    diagonal_uncertainty_residual,
)
from src.station_sheaf import (
    StationEdgeChunk,
    StationIdentitySheaf,
    StationSection,
    StationVariable,
)


VARIABLE = StationVariable("temperature", "degree_Celsius")
EDGES = StationEdgeChunk(
    tail=np.asarray([0, 0, 1], dtype=np.int64),
    head=np.asarray([1, 2, 2], dtype=np.int64),
    distance_m=np.asarray([10_000.0, 15_000.0, 12_000.0]),
)
SHEAF = StationIdentitySheaf(station_count=3, variables=(VARIABLE,))


def section(values, observed=None) -> StationSection:
    values = np.asarray(values, dtype=np.float64).reshape(3, 1)
    if observed is None:
        observed = np.ones((3, 1), dtype=bool)
    return StationSection(variables=(VARIABLE,), values=values, observed=observed)


def uncertainty(sigmas, observed=None) -> DiagonalObservationUncertainty:
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(3, 1)
    if observed is None:
        observed = np.ones((3, 1), dtype=bool)
    return DiagonalObservationUncertainty(
        variables=(VARIABLE,),
        standard_deviation=sigmas,
        observed=observed,
    )


class StationUncertaintyResidualTests(unittest.TestCase):
    def test_exact_residual_remains_the_algebraic_reference(self) -> None:
        values = section([10.0, 11.0, 9.0])
        exact = SHEAF.residual(EDGES, values)
        report = diagonal_uncertainty_residual(
            SHEAF, EDGES, values, uncertainty([2.0, 2.0, 2.0])
        )
        self.assertEqual(report.covariance_assumption, COVARIANCE_ASSUMPTION)
        self.assertEqual(report.residual_unit, "standard_deviation")
        self.assertEqual(report.variable_units, ("degree_Celsius",))
        self.assertEqual(report.degrees_of_freedom, exact.values.size)
        self.assertAlmostEqual(report.raw_residual_energy, exact.energy())
        np.testing.assert_array_equal(report.row_indices, exact.row_indices)

    def test_uncertainty_missingness_masks_without_imputation(self) -> None:
        values = section([10.0, 11.0, 9.0])
        observed = np.asarray([[True], [True], [False]], dtype=bool)
        report = diagonal_uncertainty_residual(
            SHEAF,
            EDGES,
            values,
            uncertainty([1.0, 1.0, 0.0], observed=observed),
        )
        self.assertEqual(report.degrees_of_freedom, 1)
        np.testing.assert_array_equal(report.row_indices, np.asarray([0]))
        self.assertEqual(SHEAF.residual(EDGES, values).values.size, 3)

    def test_known_noise_is_calibrated_without_a_significance_threshold(self) -> None:
        rng = np.random.default_rng(20260922)
        sigma = 2.0
        reduced = []
        declared = uncertainty([sigma, sigma, sigma])
        for _ in range(1200):
            noisy = section(10.0 + rng.normal(0.0, sigma, size=3))
            report = diagonal_uncertainty_residual(SHEAF, EDGES, noisy, declared)
            reduced.append(report.reduced_chi_square())
        mean_reduced = float(np.mean(reduced))
        self.assertGreater(mean_reduced, 0.9)
        self.assertLess(mean_reduced, 1.1)

    def test_injected_fault_increases_standardized_and_raw_residuals(self) -> None:
        declared = uncertainty([1.0, 1.0, 1.0])
        clean = diagonal_uncertainty_residual(
            SHEAF, EDGES, section([10.0, 10.2, 9.8]), declared
        )
        fault = diagonal_uncertainty_residual(
            SHEAF, EDGES, section([16.0, 10.2, 9.8]), declared
        )
        self.assertGreater(fault.chi_square, clean.chi_square)
        self.assertGreater(fault.raw_residual_energy, clean.raw_residual_energy)

    def test_uncertainty_scale_changes_standardized_not_raw_residual(self) -> None:
        values = section([10.0, 12.0, 9.0])
        tight = diagonal_uncertainty_residual(
            SHEAF, EDGES, values, uncertainty([0.5, 0.5, 0.5])
        )
        loose = diagonal_uncertainty_residual(
            SHEAF, EDGES, values, uncertainty([2.0, 2.0, 2.0])
        )
        self.assertEqual(tight.raw_residual_energy, loose.raw_residual_energy)
        self.assertGreater(tight.chi_square, loose.chi_square)


    def test_correlated_covariance_accounts_for_cycle_rank(self) -> None:
        covariance = np.eye(3, dtype=np.float64)
        declared = CorrelatedObservationUncertainty(
            variables=(VARIABLE,),
            covariance=covariance,
            observed=np.ones((3, 1), dtype=bool),
        )
        report = correlated_uncertainty_residual(
            SHEAF, EDGES, section([10.0, 11.0, 9.0]), declared
        )
        self.assertEqual(report.covariance_assumption, CORRELATED_COVARIANCE_ASSUMPTION)
        self.assertEqual(report.degrees_of_freedom, 2)
        self.assertEqual(report.row_units, ("degree_Celsius",) * 3)
        self.assertLess(report.degrees_of_freedom, report.row_indices.size)
        self.assertGreaterEqual(report.numerical_rank_tolerance, 0.0)

    def test_correlated_known_noise_reduced_chi_square_is_calibrated(self) -> None:
        rng = np.random.default_rng(20260924)
        sigma = 2.0
        correlation = 0.4
        covariance = sigma**2 * np.asarray(
            [
                [1.0, correlation, correlation],
                [correlation, 1.0, correlation],
                [correlation, correlation, 1.0],
            ]
        )
        declared = CorrelatedObservationUncertainty(
            variables=(VARIABLE,),
            covariance=covariance,
            observed=np.ones((3, 1), dtype=bool),
        )
        reduced = []
        for values in rng.multivariate_normal(
            mean=np.full(3, 10.0), cov=covariance, size=1600
        ):
            report = correlated_uncertainty_residual(
                SHEAF, EDGES, section(values), declared
            )
            reduced.append(report.reduced_chi_square())
        mean_reduced = float(np.mean(reduced))
        self.assertGreater(mean_reduced, 0.93)
        self.assertLess(mean_reduced, 1.07)

    def test_correlated_missingness_masks_residual_rows_without_imputation(self) -> None:
        observed = np.asarray([[True], [True], [False]], dtype=bool)
        declared = CorrelatedObservationUncertainty(
            variables=(VARIABLE,),
            covariance=np.eye(3, dtype=np.float64),
            observed=observed,
        )
        report = correlated_uncertainty_residual(
            SHEAF, EDGES, section([10.0, 11.0, 9.0]), declared
        )
        np.testing.assert_array_equal(report.row_indices, np.asarray([0]))
        self.assertEqual(report.degrees_of_freedom, 1)

    def test_correlated_covariance_rejects_non_psd_input(self) -> None:
        covariance = np.asarray(
            [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            CorrelatedObservationUncertainty(
                variables=(VARIABLE,),
                covariance=covariance,
                observed=np.ones((3, 1), dtype=bool),
            )

    def test_common_mode_only_covariance_has_no_residual_variance(self) -> None:
        declared = CorrelatedObservationUncertainty(
            variables=(VARIABLE,),
            covariance=np.ones((3, 3), dtype=np.float64),
            observed=np.ones((3, 1), dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "no positive-variance mode"):
            correlated_uncertainty_residual(
                SHEAF, EDGES, section([10.0, 10.0, 10.0]), declared
            )

    def test_invalid_observed_uncertainty_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            uncertainty([1.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            uncertainty([1.0, np.nan, 1.0])


if __name__ == "__main__":
    unittest.main()
