from __future__ import annotations

import json
import unittest

import numpy as np

from reference.multirepresentation_state_space import (
    analyze_state_space_baselines,
)
from reference.multirepresentation_structure_evaluation import (
    load_evaluation_fixture,
)
from reference.multirepresentation_worlds import (
    load_dynamics_fixture,
    load_fixture,
)


class MultirepresentationStateSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world_fixture = load_fixture()
        cls.dynamics_fixture = load_dynamics_fixture()
        cls.evaluation_fixture = load_evaluation_fixture()
        cls.result = analyze_state_space_baselines(
            cls.world_fixture,
            cls.dynamics_fixture,
            cls.evaluation_fixture,
        )

    def test_state_space_uses_distinct_discovery_and_confirmation_trajectories(self) -> None:
        self.assertNotEqual(
            self.result["discovery_trajectory_digest"],
            self.result["confirmation_trajectory_digest"],
        )

    def test_same_registered_representation_family_is_used_in_every_world(self) -> None:
        required = {
            "raw_concat",
            "concat_pca",
            "linear_cca",
            "factor_analysis",
        }
        for world_id, item in self.result["worlds"].items():
            self.assertEqual(set(item["representations"]), required, msg=world_id)
            for representation_id, diagnostics in item["representations"].items():
                self.assertGreater(
                    diagnostics["representation_dimension"],
                    0,
                    msg=(world_id, representation_id),
                )
                self.assertGreater(
                    diagnostics["discovery_state_rank"],
                    0,
                    msg=(world_id, representation_id),
                )
                self.assertEqual(
                    diagnostics["eigenvalue_count"],
                    diagnostics["discovery_state_rank"],
                )
                self.assertTrue(
                    np.isfinite(diagnostics["training_relative_error"]),
                    msg=(world_id, representation_id),
                )
                self.assertTrue(
                    np.isfinite(diagnostics["confirmation_relative_error"]),
                    msg=(world_id, representation_id),
                )

    def test_shared_target_forecast_is_only_scored_when_authority_declares_target(self) -> None:
        for world_id, item in self.result["worlds"].items():
            target_name = item["shared_evaluation_target_name"]
            for representation_id, diagnostics in item["representations"].items():
                if target_name is None:
                    self.assertNotIn(
                        "confirmation_target_normalized_rmse",
                        diagnostics,
                        msg=(world_id, representation_id),
                    )
                    self.assertNotIn(
                        "confirmation_target_abs_spearman",
                        diagnostics,
                        msg=(world_id, representation_id),
                    )
                else:
                    self.assertTrue(
                        np.isfinite(
                            diagnostics["confirmation_target_normalized_rmse"]
                        ),
                        msg=(world_id, representation_id),
                    )
                    self.assertTrue(
                        np.isfinite(
                            diagnostics["confirmation_target_abs_spearman"]
                        ),
                        msg=(world_id, representation_id),
                    )
                    self.assertGreaterEqual(
                        diagnostics["confirmation_target_normalized_rmse"],
                        0.0,
                    )
                    self.assertGreaterEqual(
                        diagnostics["confirmation_target_abs_spearman"],
                        0.0,
                    )
                    self.assertLessEqual(
                        diagnostics["confirmation_target_abs_spearman"],
                        1.0,
                    )

    def test_result_is_json_portable(self) -> None:
        encoded = json.dumps(self.result, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
