from __future__ import annotations

import unittest

from reference.ebm_recovery_conditioning_guardrails import (
    analyze_conditioning_guardrails,
    load_conditioning_fixture,
)
from reference.information_geometry_ebm_recovery import load_recovery_fixture
from reference.two_layer_energy_balance import load_fixture as load_ebm_fixture
from reference.two_layer_forcing_protocols import load_protocol_fixture
from reference.two_layer_parameter_identifiability import load_parameter_fixture


class EbmRecoveryConditioningGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conditioning = load_conditioning_fixture()
        cls.result = analyze_conditioning_guardrails(
            load_ebm_fixture(),
            load_protocol_fixture(),
            load_recovery_fixture(),
            load_parameter_fixture(),
            cls.conditioning,
        )

    def test_all_preregistered_checks_pass(self) -> None:
        self.assertTrue(self.result["all_checks_passed"])
        self.assertTrue(all(self.result["pass_flags"].values()))

    def test_real_recovery_points_are_full_rank_and_accepted(self) -> None:
        self.assertEqual(len(self.result["recovery_points"]), 5)
        for item in self.result["recovery_points"]:
            self.assertFalse(item["report"]["rank_deficient"])
            self.assertEqual(item["report"]["numerical_rank"], 4)
            self.assertIsNotNone(item["report"]["condition_number_2"])
            self.assertEqual(item["verification_policy_status"], "accepted")

    def test_equilibrium_structural_degeneracy_is_unavailable_not_infinite_sentinel(self) -> None:
        item = self.result["equilibrium_only"]
        self.assertTrue(item["report"]["rank_deficient"])
        self.assertEqual(item["report"]["numerical_rank"], 1)
        self.assertIsNone(item["report"]["condition_number_2"])
        self.assertEqual(item["verification_policy_status"], "rank_deficient")

    def test_near_collinear_control_is_full_rank_but_policy_rejected(self) -> None:
        item = self.result["near_collinear_control"]
        self.assertFalse(item["report"]["rank_deficient"])
        self.assertGreater(
            item["report"]["condition_number_2"],
            self.conditioning["verification_policy"]["max_condition_number_2"],
        )
        self.assertEqual(
            item["verification_policy_status"], "condition_number_exceeded"
        )

    def test_condition_number_is_invariant_to_uniform_noise_and_matrix_scale(self) -> None:
        checks = self.result["actual_checks"]
        limits = self.conditioning["confirmation_checks"]
        self.assertLessEqual(
            checks["uniform_noise_scaling_condition_number_relative_spread"],
            limits["uniform_noise_scaling_condition_number_relative_tolerance"],
        )
        self.assertLessEqual(
            checks["matrix_scaling_condition_number_relative_spread"],
            limits["matrix_scaling_condition_number_relative_tolerance"],
        )

    def test_fisher_condition_matches_squared_jacobian_condition(self) -> None:
        self.assertLessEqual(
            self.result["actual_checks"][
                "fisher_condition_equals_jacobian_condition_squared_max_relative_error"
            ],
            self.conditioning["confirmation_checks"][
                "fisher_condition_equals_jacobian_condition_squared_relative_tolerance"
            ],
        )


if __name__ == "__main__":
    unittest.main()
