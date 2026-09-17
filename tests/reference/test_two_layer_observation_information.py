from __future__ import annotations

import math
import unittest

import numpy as np

from reference.two_layer_energy_balance import load_fixture, mode_basis, parameters_from_fixture
from reference.two_layer_observation_information import (
    OBSERVATION_SUBSETS,
    analyze_observation_geometry,
    gaussian_mean_fisher_reference,
    load_observation_fixture,
    observation_jacobian,
    observation_noise_stddev,
)


class TwoLayerObservationInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm = load_fixture()
        cls.observations = load_observation_fixture()
        cls.result = analyze_observation_geometry(cls.ebm, cls.observations)

    def test_fixture_is_linked_and_explicitly_synthetic(self) -> None:
        self.assertEqual(
            self.observations["ebm_fixture_id"],
            self.ebm["fixture_id"],
        )
        self.assertEqual(
            self.observations["provenance"]["kind"],
            "synthetic_structural_control",
        )

    def test_surface_and_toa_add_precision_without_state_rank(self) -> None:
        surface = self.result["subsets"]["surface_only"]
        surface_toa = self.result["subsets"]["surface_plus_toa"]

        self.assertEqual(surface["state_rank"], 1)
        self.assertEqual(surface["state_nullity"], 1)
        self.assertEqual(surface_toa["state_rank"], 1)
        self.assertEqual(surface_toa["state_nullity"], 1)
        self.assertGreater(
            surface_toa["state_fisher"][0][0],
            surface["state_fisher"][0][0],
        )
        self.assertEqual(surface_toa["state_fisher"][1][1], 0.0)
        self.assertTrue(math.isinf(surface_toa["state_condition_number_2"]))

    def test_ocean_heat_uptake_supplies_the_missing_state_direction(self) -> None:
        for name in ("surface_plus_ocean", "toa_plus_ocean", "all_channels"):
            diagnostics = self.result["subsets"][name]
            self.assertEqual(diagnostics["state_rank"], 2, msg=name)
            self.assertEqual(diagnostics["state_nullity"], 0, msg=name)
            self.assertTrue(
                math.isfinite(diagnostics["state_condition_number_2"]),
                msg=name,
            )

    def test_all_channel_state_fisher_matches_closed_form(self) -> None:
        parameters = parameters_from_fixture(self.ebm)
        channels = self.observations["channels"]
        sigma_surface = channels["surface_temperature"]["standard_deviation"]
        sigma_toa = channels["toa_imbalance"]["standard_deviation"]
        sigma_ocean = channels["ocean_heat_uptake"]["standard_deviation"]

        surface_precision = 1.0 / sigma_surface**2
        toa_precision = parameters.climate_feedback_w_m2_k**2 / sigma_toa**2
        ocean_precision = parameters.ocean_heat_exchange_w_m2_k**2 / sigma_ocean**2
        expected = np.array(
            [
                [
                    surface_precision + toa_precision + ocean_precision,
                    -ocean_precision,
                ],
                [-ocean_precision, ocean_precision],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            self.result["subsets"]["all_channels"]["state_fisher"],
            expected,
            rtol=2e-14,
            atol=2e-14,
        )

    def test_modal_metric_is_the_state_metric_pullback(self) -> None:
        parameters = parameters_from_fixture(self.ebm)
        _, basis = mode_basis(parameters)
        for name in OBSERVATION_SUBSETS:
            diagnostics = self.result["subsets"][name]
            state_fisher = np.asarray(diagnostics["state_fisher"], dtype=float)
            expected_modal = basis.T @ state_fisher @ basis
            np.testing.assert_allclose(
                diagnostics["modal_fisher"],
                expected_modal,
                rtol=2e-14,
                atol=2e-14,
                err_msg=name,
            )

    def test_rank_one_observation_geometry_couples_modal_directions_maximally(self) -> None:
        for name in ("surface_only", "toa_only", "ocean_only", "surface_plus_toa"):
            diagnostics = self.result["subsets"][name]
            self.assertEqual(diagnostics["state_rank"], 1, msg=name)
            self.assertAlmostEqual(
                abs(diagnostics["normalized_cross_mode_coupling"]),
                1.0,
                places=12,
                msg=name,
            )

        for name in ("surface_plus_ocean", "toa_plus_ocean", "all_channels"):
            diagnostics = self.result["subsets"][name]
            self.assertLess(
                abs(diagnostics["normalized_cross_mode_coupling"]),
                1.0,
                msg=name,
            )

    def test_information_distance_is_coordinate_covariant(self) -> None:
        covariance = self.result["coordinate_covariance"]
        self.assertLess(
            covariance["max_fisher_squared_length_disagreement"],
            2e-14,
        )
        for probe in covariance["probes"]:
            values = list(probe["fisher_squared_length"].values())
            self.assertAlmostEqual(values[0], values[1], places=12)
            self.assertAlmostEqual(values[0], values[2], places=12)

    def test_raw_euclidean_distance_is_not_a_shared_representation_metric(self) -> None:
        covariance = self.result["coordinate_covariance"]
        self.assertGreater(covariance["min_raw_euclidean_squared_norm_spread"], 1e-4)

        unit = covariance["unit_rescaling"]
        self.assertAlmostEqual(
            unit["fisher_squared_length_kelvin"],
            unit["fisher_squared_length_millikelvin"],
            places=12,
        )
        ratio = (
            unit["raw_euclidean_squared_norm_millikelvin"]
            / unit["raw_euclidean_squared_norm_kelvin"]
        )
        self.assertAlmostEqual(ratio, 1_000_000.0, places=6)

    def test_uniform_noise_scaling_changes_information_not_geometry_rank_or_angles(self) -> None:
        base = self.result
        doubled = analyze_observation_geometry(
            self.ebm,
            self.observations,
            noise_multiplier=2.0,
        )
        for name in OBSERVATION_SUBSETS:
            base_diag = base["subsets"][name]
            doubled_diag = doubled["subsets"][name]
            self.assertEqual(base_diag["state_rank"], doubled_diag["state_rank"], msg=name)
            self.assertEqual(
                base_diag["state_nullity"], doubled_diag["state_nullity"], msg=name
            )
            np.testing.assert_allclose(
                doubled_diag["state_fisher"],
                np.asarray(base_diag["state_fisher"]) / 4.0,
                rtol=2e-14,
                atol=2e-14,
                err_msg=name,
            )
            np.testing.assert_allclose(
                doubled_diag["modal_fisher"],
                np.asarray(base_diag["modal_fisher"]) / 4.0,
                rtol=2e-14,
                atol=2e-14,
                err_msg=name,
            )
            self.assertAlmostEqual(
                doubled_diag["normalized_cross_mode_coupling"],
                base_diag["normalized_cross_mode_coupling"],
                places=12,
                msg=name,
            )

    def test_reference_fisher_fails_closed_on_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise_multiplier"):
            observation_noise_stddev(
                self.observations,
                ("surface_temperature",),
                noise_multiplier=0.0,
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            observation_jacobian(self.ebm, ("unknown",))
        with self.assertRaisesRegex(ValueError, "match Jacobian"):
            gaussian_mean_fisher_reference(np.eye(2), np.ones(1))
        with self.assertRaisesRegex(ValueError, "positive"):
            gaussian_mean_fisher_reference(np.eye(2), np.array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
