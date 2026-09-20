from __future__ import annotations

import unittest

import numpy as np

from reference.information_geometry_ebm_recovery import (
    _gauss_newton_equivalence,
    load_recovery_fixture,
)
from reference.two_layer_ebm_recovery_objective import (
    parameter_vector,
    protocol_temperature_outputs,
)
from reference.two_layer_energy_balance import (
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
)
from reference.two_layer_forcing_protocols import load_protocol_fixture


class InformationGeometryEbmRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ebm = load_ebm_fixture()
        self.protocols = load_protocol_fixture()
        self.recovery = load_recovery_fixture()

    def test_preregistered_trial_grid_is_exact_cartesian_product(self) -> None:
        self.assertEqual(
            self.recovery["trial_pairing"],
            "cartesian_product_dataset_seeds_x_start_log_parameter_offsets",
        )
        self.assertEqual(len(self.recovery["dataset_seeds"]), 4)
        self.assertEqual(len(self.recovery["start_log_parameter_offsets"]), 4)
        self.assertEqual(
            len(self.recovery["dataset_seeds"])
            * len(self.recovery["start_log_parameter_offsets"]),
            16,
        )

    def test_confirmation_protocols_are_disjoint_from_discovery_protocols(self) -> None:
        discovery = set(self.recovery["discovery_protocol_ids"])
        confirmation = set(self.recovery["confirmation_protocol_ids"])
        self.assertTrue(discovery)
        self.assertTrue(confirmation)
        self.assertFalse(discovery & confirmation)

    def test_local_fisher_step_matches_gauss_newton_at_frozen_starts(self) -> None:
        truth = parameters_from_fixture(self.ebm)
        truth_log = np.log(parameter_vector(truth))
        starts = [
            truth_log + np.asarray(offset, dtype=float)
            for offset in self.recovery["start_log_parameter_offsets"]
        ]
        observed = protocol_temperature_outputs(
            truth,
            self.protocols,
            list(self.recovery["discovery_protocol_ids"]),
            samples_per_segment=int(self.recovery["samples_per_segment"]),
        )
        witness = _gauss_newton_equivalence(
            observed,
            self.protocols,
            self.recovery,
            starts,
        )
        self.assertEqual(witness["sample_count"], 4)
        self.assertLessEqual(
            witness["max_relative_step_difference"],
            self.recovery["interpretation_thresholds"][
                "gauss_newton_step_max_relative_difference"
            ],
        )


if __name__ == "__main__":
    unittest.main()
