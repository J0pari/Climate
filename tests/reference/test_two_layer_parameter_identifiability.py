from __future__ import annotations

import json
import unittest

import numpy as np

from reference.two_layer_energy_balance import load_fixture, parameters_from_fixture
from reference.two_layer_forcing_protocols import load_protocol_fixture
from reference.two_layer_parameter_identifiability import (
    PARAMETER_IDS,
    analyze_parameter_identifiability,
    centered_log_parameter_jacobian,
    load_parameter_fixture,
)


class TwoLayerParameterIdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm = load_fixture()
        cls.protocols = load_protocol_fixture()
        cls.fixture = load_parameter_fixture()
        cls.result = analyze_parameter_identifiability(
            cls.ebm,
            cls.protocols,
            cls.fixture,
        )

    def test_fixture_pins_parameter_and_protocol_identity(self) -> None:
        self.assertEqual(self.fixture["parameter_ids"], list(PARAMETER_IDS))
        self.assertEqual(
            self.fixture["transient_protocol_ids"],
            [
                "held_out_step",
                "held_out_ramp",
                "overshoot_reversal",
                "sign_reversal",
            ],
        )

    def test_equilibrium_is_structurally_rank_one(self) -> None:
        equilibrium = self.result["equilibrium"]
        self.assertEqual(equilibrium["parameter_dimension"], 4)
        self.assertEqual(equilibrium["rank"], 1)
        self.assertEqual(equilibrium["nullity"], 3)
        self.assertEqual(
            equilibrium["exactly_zero_sensitivity_parameters"],
            [
                "surface_heat_capacity_w_yr_m2_k",
                "deep_heat_capacity_w_yr_m2_k",
                "ocean_heat_exchange_w_m2_k",
            ],
        )
        self.assertGreater(
            equilibrium["column_norms"]["climate_feedback_w_m2_k"],
            0.0,
        )
        self.assertEqual(self.result["equilibrium_invisible_parameter_count"], 3)

    def test_held_out_transients_locally_span_all_four_parameter_directions(self) -> None:
        transient = self.result["transient"]
        self.assertEqual(transient["parameter_dimension"], 4)
        self.assertEqual(transient["rank"], 4)
        self.assertEqual(transient["nullity"], 0)
        self.assertEqual(transient["condition_number_status"], "finite")
        self.assertTrue(np.isfinite(transient["condition_number"]))
        self.assertGreater(transient["singular_values"][-1], transient["rank_tolerance"])
        self.assertEqual(self.result["local_rank_gain"], 3)

    def test_centered_log_derivatives_are_step_stable(self) -> None:
        self.assertLess(
            self.result["equilibrium_step_consistency_relative_frobenius"],
            1e-7,
        )
        self.assertLess(
            self.result["transient_step_consistency_relative_frobenius"],
            1e-7,
        )

    def test_sensitivity_rank_does_not_require_parameter_fitting(self) -> None:
        parameters = parameters_from_fixture(self.ebm)

        def output(candidate):
            return np.array(
                [
                    candidate.surface_heat_capacity_w_yr_m2_k,
                    candidate.deep_heat_capacity_w_yr_m2_k,
                    candidate.ocean_heat_exchange_w_m2_k,
                    candidate.climate_feedback_w_m2_k,
                ],
                dtype=float,
            )

        jacobian = centered_log_parameter_jacobian(
            parameters,
            output,
            log_step=self.fixture["log_parameter_step"],
        )
        self.assertEqual(jacobian.shape, (4, 4))
        self.assertTrue(np.all(np.diag(jacobian) > 0.0))
        self.assertLess(
            float(np.max(np.abs(jacobian - np.diag(np.diag(jacobian))))),
            1e-12,
        )

    def test_invalid_parameter_identity_fails_closed(self) -> None:
        malformed = dict(self.fixture)
        malformed["parameter_ids"] = list(reversed(malformed["parameter_ids"]))
        with self.assertRaisesRegex(ValueError, "parameter identities changed"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "fixture.json"
                path.write_text(json.dumps(malformed), encoding="utf-8")
                load_parameter_fixture(path)

    def test_result_is_json_portable(self) -> None:
        encoded = json.dumps(self.result, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
