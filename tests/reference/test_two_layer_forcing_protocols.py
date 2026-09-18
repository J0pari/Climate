from __future__ import annotations

import unittest

import numpy as np

from reference.two_layer_energy_balance import (
    advance_constant_forcing,
    equilibrium_state_k,
    load_fixture,
    parameters_from_fixture,
)
from reference.two_layer_forcing_protocols import (
    advance_affine_forcing,
    analyze_protocols,
    load_protocol_fixture,
    simulate_protocol,
)


class TwoLayerForcingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm_fixture = load_fixture()
        cls.protocol_fixture = load_protocol_fixture()
        cls.parameters = parameters_from_fixture(cls.ebm_fixture)

    def test_affine_propagator_reduces_to_constant_forcing_reference(self) -> None:
        initial = np.array([0.17, -0.03])
        for forcing in (-1.0, 0.0, 1.0, 4.0):
            for dt in (0.25, 1.0, 10.0, 75.0):
                affine = advance_affine_forcing(
                    initial, self.parameters, forcing, forcing, dt
                )
                constant = advance_constant_forcing(
                    initial, self.parameters, forcing, dt
                )
                np.testing.assert_allclose(
                    affine, constant, rtol=3e-13, atol=3e-13
                )

    def test_ramp_composition_is_exact_under_segment_subdivision(self) -> None:
        initial = equilibrium_state_k(self.parameters, 0.0)
        direct = advance_affine_forcing(initial, self.parameters, 0.0, 4.0, 50.0)
        first = advance_affine_forcing(initial, self.parameters, 0.0, 2.0, 25.0)
        composed = advance_affine_forcing(first, self.parameters, 2.0, 4.0, 25.0)
        np.testing.assert_allclose(direct, composed, rtol=4e-13, atol=4e-13)

    def test_protocol_suite_closes_budget_and_contains_held_out_forcing(self) -> None:
        result = analyze_protocols(self.ebm_fixture, self.protocol_fixture)
        self.assertLess(
            result["constant_forcing_equivalence_max_abs_state_error_k"], 2e-13
        )
        self.assertLess(result["max_abs_energy_budget_residual_w_m2"], 2e-14)
        self.assertGreater(result["held_out_absolute_forcing_margin_w_m2"], 0.0)
        self.assertEqual(
            set(result["protocols"]),
            {
                "constant_control",
                "held_out_step",
                "held_out_ramp",
                "overshoot_reversal",
                "sign_reversal",
            },
        )
        self.assertGreater(
            max(result["protocols"]["overshoot_reversal"]["forcing_w_m2"]),
            max(result["discovery_forcing_domain_w_m2"]),
        )

    def test_forcing_jump_changes_tendency_not_state(self) -> None:
        protocol = {
            "protocol_id": "jump",
            "initial_equilibrium_forcing_w_m2": 0.0,
            "segments": [
                {
                    "duration_years": 1.0,
                    "forcing_start_w_m2": 3.0,
                    "forcing_end_w_m2": 3.0,
                }
            ],
        }
        result = simulate_protocol(protocol, self.parameters, samples_per_segment=4)
        np.testing.assert_allclose(result["state_k"][0], [0.0, 0.0], rtol=0.0, atol=0.0)
        self.assertEqual(result["forcing_w_m2"][0], 3.0)
        self.assertGreater(result["state_k"][1][0], 0.0)

    def test_malformed_protocols_fail_closed(self) -> None:
        initial = np.zeros(2)
        with self.assertRaisesRegex(ValueError, "positive"):
            advance_affine_forcing(initial, self.parameters, 0.0, 1.0, 0.0)

        bad = {
            "protocol_id": "bad",
            "initial_equilibrium_forcing_w_m2": 0.0,
            "segments": [
                {
                    "duration_years": -1.0,
                    "forcing_start_w_m2": 0.0,
                    "forcing_end_w_m2": 1.0,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "duration"):
            simulate_protocol(bad, self.parameters, samples_per_segment=8)


if __name__ == "__main__":
    unittest.main()
