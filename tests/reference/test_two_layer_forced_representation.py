from __future__ import annotations

import unittest

import numpy as np

from reference.two_layer_energy_balance import load_fixture
from reference.two_layer_forced_representation import (
    analyze_forced_representations,
    discovery_transition_samples,
    load_training_fixture,
)
from reference.two_layer_forcing_protocols import load_protocol_fixture


class TwoLayerForcedRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm_fixture = load_fixture()
        cls.protocol_fixture = load_protocol_fixture()
        cls.training_fixture = load_training_fixture()
        cls.result = analyze_forced_representations(
            cls.ebm_fixture,
            cls.protocol_fixture,
            cls.training_fixture,
        )

    def test_discovery_sampler_never_uses_confirmation_only_forcing(self) -> None:
        samples = discovery_transition_samples(
            self.ebm_fixture,
            self.protocol_fixture,
            self.training_fixture,
        )
        lower, upper = self.protocol_fixture["discovery_forcing_domain_w_m2"]
        starts = samples["forcing_start_w_m2"]
        ends = starts + (
            samples["forcing_rate_w_m2_per_year"]
            * self.training_fixture["transition_dt_years"]
        )
        self.assertGreaterEqual(float(np.min(starts)), lower)
        self.assertLessEqual(float(np.max(starts)), upper)
        self.assertGreaterEqual(float(np.min(ends)), lower)
        self.assertLessEqual(float(np.max(ends)), upper)

    def test_full_state_identifies_exact_controlled_transition_and_extrapolates(self) -> None:
        full = self.result["representations"]["temperature_state"]
        self.assertEqual(full["representation_dimension"], 2)
        self.assertEqual(full["design_dimension"], 4)
        self.assertEqual(full["design_rank"], 4)
        self.assertEqual(full["condition_number_status"], "finite")
        self.assertLess(full["training_relative_error"], 2e-14)
        self.assertLess(full["confirmation_relative_error"], 2e-13)
        self.assertLess(full["confirmation_max_abs_error"], 2e-12)
        self.assertEqual(
            set(full["protocols"]),
            {
                "held_out_step",
                "held_out_ramp",
                "overshoot_reversal",
                "sign_reversal",
            },
        )

    def test_surface_scalar_fails_markov_closure_under_same_controls(self) -> None:
        scalar = self.result["representations"]["surface_temperature_scalar"]
        self.assertEqual(scalar["representation_dimension"], 1)
        self.assertEqual(scalar["design_dimension"], 3)
        self.assertEqual(scalar["design_rank"], 3)
        self.assertGreater(scalar["training_relative_error"], 0.05)
        self.assertGreater(scalar["confirmation_relative_error"], 0.02)
        self.assertGreater(
            self.result["closure_gap_confirmation_relative_error"], 0.02
        )

    def test_confirmation_is_genuinely_outside_discovery_forcing_range(self) -> None:
        self.assertGreater(
            self.result["held_out_absolute_forcing_margin_w_m2"], 0.0
        )
        full = self.result["representations"]["temperature_state"]
        discovery_max = max(
            abs(value)
            for value in self.protocol_fixture["discovery_forcing_domain_w_m2"]
        )
        self.assertTrue(
            any(
                max(
                    abs(protocol["forcing_min_w_m2"]),
                    abs(protocol["forcing_max_w_m2"]),
                )
                > discovery_max
                for protocol in full["protocols"].values()
            )
        )

    def test_result_is_json_portable(self) -> None:
        import json

        encoded = json.dumps(self.result, allow_nan=False, sort_keys=True)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
