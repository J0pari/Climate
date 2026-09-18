from __future__ import annotations

import json
import unittest

from reference.two_layer_energy_balance import load_fixture
from reference.two_layer_regime_feedback import (
    generate_regime_feedback_samples,
    load_regime_fixture,
    summarize_regime_feedback,
)
from reference.two_layer_regime_representation import (
    analyze_regime_representations,
)


class TwoLayerRegimeRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm = load_fixture()
        cls.fixture = load_regime_fixture()
        cls.physical = summarize_regime_feedback(cls.ebm, cls.fixture)
        cls.result = analyze_regime_representations(cls.ebm, cls.fixture)

    def test_all_samples_remain_inside_declared_feedback_regime(self) -> None:
        self.assertGreaterEqual(
            self.physical["observed_minimum_regime_margin_k"],
            self.fixture["minimum_regime_margin_k"],
        )
        self.assertEqual(
            self.result["sample_digest"],
            self.physical["sample_digest"],
        )

    def test_raw_global_map_exposes_regime_model_mismatch(self) -> None:
        raw = self.result["raw_global"]
        self.assertEqual(raw["design_dimension"], 3)
        self.assertEqual(raw["design_rank"], 3)
        self.assertEqual(raw["condition_number_status"], "finite")
        self.assertGreater(raw["training_relative_error"], 0.003)
        self.assertGreater(raw["confirmation_relative_error"], 0.004)

    def test_regime_gated_representation_is_exact_for_local_piecewise_dynamics(self) -> None:
        gated = self.result["regime_gated"]
        self.assertEqual(gated["design_dimension"], 6)
        self.assertEqual(gated["design_rank"], 6)
        self.assertEqual(gated["condition_number_status"], "finite")
        self.assertLess(gated["training_relative_error"], 1e-12)
        self.assertLess(gated["confirmation_relative_error"], 1e-12)
        self.assertGreater(self.result["confirmation_error_reduction"], 0.004)

    def test_swapped_regime_labels_destroy_the_gated_advantage(self) -> None:
        self.assertGreater(
            self.result["swapped_regime_confirmation_relative_error"],
            0.02,
        )
        self.assertGreater(
            self.result["swapped_regime_confirmation_relative_error"],
            self.result["raw_global"]["confirmation_relative_error"],
        )

    def test_sample_generation_is_deterministic(self) -> None:
        first = generate_regime_feedback_samples(self.ebm, self.fixture)
        second = generate_regime_feedback_samples(self.ebm, self.fixture)
        self.assertEqual(first["sample_digest"], second["sample_digest"])

    def test_result_is_json_portable(self) -> None:
        encoded = json.dumps(self.result, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
