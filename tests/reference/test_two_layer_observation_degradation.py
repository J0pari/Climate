from __future__ import annotations

import json
import unittest

import numpy as np

from reference.two_layer_energy_balance import load_fixture
from reference.two_layer_forced_representation import load_training_fixture
from reference.two_layer_forcing_protocols import load_protocol_fixture
from reference.two_layer_observation_degradation import (
    analyze_observation_degradation,
    load_observation_fixture,
)


class TwoLayerObservationDegradationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = analyze_observation_degradation(
            load_fixture(),
            load_protocol_fixture(),
            load_training_fixture(),
            load_observation_fixture(),
        )

    def test_clean_full_state_remains_exact_control(self) -> None:
        clean = self.result["representations"]["clean_temperature_state"]
        self.assertEqual(clean["observation_dimension"], 2)
        self.assertEqual(clean["structural_observation_rank"], 2)
        self.assertEqual(clean["redundant_dimension_count"], 0)
        self.assertLess(
            clean["confirmation_state_reconstruction_relative_error"], 1e-14
        )
        self.assertLess(
            clean["confirmation_forced_state_prediction_relative_error"], 1e-12
        )

    def test_noise_degrades_full_state_without_changing_structural_rank(self) -> None:
        clean = self.result["representations"]["clean_temperature_state"]
        noisy = self.result["representations"]["noisy_temperature_state"]
        self.assertEqual(noisy["observation_dimension"], 2)
        self.assertEqual(noisy["structural_observation_rank"], 2)
        self.assertGreater(
            noisy["confirmation_state_reconstruction_relative_error"],
            clean["confirmation_state_reconstruction_relative_error"],
        )
        self.assertGreater(
            noisy["confirmation_forced_state_prediction_relative_error"],
            clean["confirmation_forced_state_prediction_relative_error"],
        )

    def test_sparse_surface_view_loses_instantaneous_state_information(self) -> None:
        full = self.result["representations"]["noisy_temperature_state"]
        surface = self.result["representations"]["noisy_surface_scalar"]
        self.assertEqual(surface["observation_dimension"], 1)
        self.assertEqual(surface["structural_observation_rank"], 1)
        self.assertGreater(
            surface["confirmation_state_reconstruction_relative_error"],
            full["confirmation_state_reconstruction_relative_error"],
        )
        self.assertGreater(
            surface["confirmation_forced_state_prediction_relative_error"],
            full["confirmation_forced_state_prediction_relative_error"],
        )
        self.assertGreater(
            self.result["noisy_full_vs_surface_forced_prediction_error_gap"],
            0.0,
        )

    def test_redundant_surface_pair_adds_dimension_not_structural_rank(self) -> None:
        surface = self.result["representations"]["noisy_surface_scalar"]
        redundant = self.result["representations"][
            "redundant_noisy_surface_pair"
        ]
        self.assertEqual(surface["structural_observation_rank"], 1)
        self.assertEqual(redundant["observation_dimension"], 2)
        self.assertEqual(redundant["structural_observation_rank"], 1)
        self.assertEqual(redundant["redundant_dimension_count"], 1)
        self.assertEqual(self.result["redundant_structural_rank_gain"], 0)
        self.assertAlmostEqual(
            redundant["confirmation_state_reconstruction_relative_error"],
            surface["confirmation_state_reconstruction_relative_error"],
            places=12,
        )
        self.assertAlmostEqual(
            redundant["confirmation_forced_state_prediction_relative_error"],
            surface["confirmation_forced_state_prediction_relative_error"],
            places=12,
        )

    def test_confirmation_protocols_remain_the_existing_held_out_set(self) -> None:
        expected = {
            "held_out_step",
            "held_out_ramp",
            "overshoot_reversal",
            "sign_reversal",
        }
        for diagnostics in self.result["representations"].values():
            self.assertEqual(set(diagnostics["protocols"]), expected)

    def test_result_is_json_portable(self) -> None:
        encoded = json.dumps(self.result, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertTrue(np.isfinite(self.result["noise_std_k"]))


if __name__ == "__main__":
    unittest.main()
