from __future__ import annotations

import json
import unittest

import numpy as np

from reference.two_layer_energy_balance import load_fixture
from reference.two_layer_stochastic_representation_statistics import (
    analyze_stochastic_representation_statistics,
)
from reference.two_layer_stochastic_variability import (
    generate_stochastic_trajectory,
    load_stochastic_fixture,
    summarize_stochastic_variability,
)


class TwoLayerStochasticStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm = load_fixture()
        cls.fixture = load_stochastic_fixture()
        cls.physical = summarize_stochastic_variability(cls.ebm, cls.fixture)
        cls.result = analyze_stochastic_representation_statistics(
            cls.ebm,
            cls.fixture,
        )

    def test_internal_exchange_has_no_direct_total_storage_source(self) -> None:
        self.assertLess(
            self.physical["max_abs_direct_internal_storage_w_m2"],
            1e-14,
        )
        self.assertLess(
            self.physical["max_abs_total_budget_residual_w_m2"],
            2e-15,
        )

    def test_seeded_trajectory_is_reproducible_and_excites_both_reservoirs(self) -> None:
        first = generate_stochastic_trajectory(self.ebm, self.fixture)
        second = generate_stochastic_trajectory(self.ebm, self.fixture)
        self.assertEqual(first["trajectory_digest"], second["trajectory_digest"])
        self.assertEqual(
            self.result["trajectory_digest"],
            self.physical["trajectory_digest"],
        )
        covariance = np.asarray(
            self.result["reference_statistics"]["state_covariance_k2"],
            dtype=float,
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        self.assertTrue(np.all(eigenvalues > 0.0))

    def test_full_state_preserves_declared_climate_statistics(self) -> None:
        full = self.result["representations"]["temperature_state"]
        self.assertEqual(full["observation_dimension"], 2)
        self.assertEqual(full["structural_observation_rank"], 2)
        self.assertLess(full["mean_state_l2_error_k"], 1e-14)
        self.assertLess(
            full["state_covariance_relative_frobenius_error"],
            1e-14,
        )
        self.assertLess(
            full["lag_covariance_relative_frobenius_error"],
            1e-14,
        )
        self.assertLess(full["surface_variance_relative_error"], 1e-14)
        self.assertLess(full["deep_variance_relative_error"], 1e-14)

    def test_surface_scalar_preserves_surface_variance_but_loses_deep_statistics(self) -> None:
        surface = self.result["representations"]["surface_temperature_scalar"]
        self.assertEqual(surface["observation_dimension"], 1)
        self.assertEqual(surface["structural_observation_rank"], 1)
        self.assertLess(surface["surface_variance_relative_error"], 1e-14)
        self.assertAlmostEqual(surface["deep_variance_relative_error"], 1.0, places=14)
        self.assertGreater(
            surface["state_covariance_relative_frobenius_error"],
            0.1,
        )
        self.assertGreater(
            surface["lag_covariance_relative_frobenius_error"],
            0.1,
        )

    def test_redundant_surface_pair_adds_coordinate_not_statistics(self) -> None:
        surface = self.result["representations"]["surface_temperature_scalar"]
        redundant = self.result["representations"]["redundant_surface_pair"]
        self.assertEqual(redundant["observation_dimension"], 2)
        self.assertEqual(redundant["structural_observation_rank"], 1)
        for key in (
            "mean_state_l2_error_k",
            "state_covariance_relative_frobenius_error",
            "lag_covariance_relative_frobenius_error",
            "surface_variance_relative_error",
            "deep_variance_relative_error",
        ):
            self.assertAlmostEqual(redundant[key], surface[key], places=12, msg=key)
        self.assertAlmostEqual(
            self.result["redundant_vs_surface_covariance_error_delta"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            self.result["redundant_vs_surface_lag_covariance_error_delta"],
            0.0,
            places=12,
        )

    def test_result_is_json_portable(self) -> None:
        encoded = json.dumps(self.result, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
