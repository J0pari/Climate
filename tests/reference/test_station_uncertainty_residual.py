from __future__ import annotations

import unittest

import numpy as np

from reference.station_uncertainty_residual import (
    COVARIANCE_ASSUMPTION,
    DiagonalObservationUncertainty,
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

    def test_invalid_observed_uncertainty_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            uncertainty([1.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            uncertainty([1.0, np.nan, 1.0])


if __name__ == "__main__":
    unittest.main()
