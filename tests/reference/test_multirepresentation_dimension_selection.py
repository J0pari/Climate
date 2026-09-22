from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from reference.multirepresentation_structure_evaluation import (
    load_evaluation_fixture,
    select_world_dimensions,
)
from reference.multirepresentation_worlds import (
    StructuralWorld,
    generate_worlds,
    load_fixture,
)


class MultirepresentationDimensionSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world_fixture = load_fixture()
        cls.evaluation_fixture = load_evaluation_fixture()
        cls.policy = cls.evaluation_fixture["dimension_selector"]
        cls.discovery = generate_worlds(cls.world_fixture)
        cls.confirmation = generate_worlds(
            cls.world_fixture,
            seed_offset=cls.evaluation_fixture["confirmation_seed_offset"],
        )

    def test_policy_uses_two_nearest_neighbor_integer_likelihood(self) -> None:
        self.assertEqual(
            self.policy["family"],
            "two_nearest_neighbor_integer_likelihood",
        )
        self.assertEqual(self.policy["neighbor_search"], "brute_euclidean")
        self.assertEqual(
            self.policy["shared_dimension_rule"],
            "view_a_plus_view_b_minus_joint",
        )

    def test_discovery_and_confirmation_recover_authoritative_dimensions(self) -> None:
        for world_id in sorted(self.discovery):
            expected = self.discovery[world_id].ground_truth
            expected_private = [int(value) for value in expected["private_dimensions"]]
            for role, worlds in (
                ("discovery", self.discovery),
                ("confirmation", self.confirmation),
            ):
                selected = select_world_dimensions(worlds[world_id], self.policy)
                self.assertEqual(selected["status"], "selected", msg=(world_id, role))
                self.assertEqual(
                    selected["selected_shared_dimension"],
                    int(expected["shared_dimension"]),
                    msg=(world_id, role, selected),
                )
                self.assertEqual(
                    selected["selected_private_dimensions"],
                    expected_private,
                    msg=(world_id, role, selected),
                )

    def test_selector_does_not_read_ground_truth_dimensions(self) -> None:
        world = self.discovery["fibered"]
        baseline = select_world_dimensions(world, self.policy)
        altered = copy.copy(world)
        object.__setattr__(
            altered,
            "ground_truth",
            {
                **world.ground_truth,
                "shared_dimension": 99,
                "private_dimensions": [98, 97],
            },
        )
        self.assertEqual(select_world_dimensions(altered, self.policy), baseline)

    def test_likelihood_candidates_are_finite_and_bounded_by_ambient_dimension(self) -> None:
        for world_id, world in self.discovery.items():
            selected = select_world_dimensions(world, self.policy)
            for scope in ("view_a", "view_b", "joint"):
                diagnostics = selected[scope]
                self.assertGreaterEqual(diagnostics["selected_dimension"], 1)
                self.assertLessEqual(
                    diagnostics["selected_dimension"],
                    diagnostics["ambient_dimension"],
                    msg=(world_id, scope),
                )
                self.assertGreater(diagnostics["continuous_mle"], 0.0)
                self.assertEqual(
                    len(diagnostics["candidate_log_likelihoods"]),
                    diagnostics["ambient_dimension"],
                )

    def test_duplicate_observations_fail_closed(self) -> None:
        world = StructuralWorld(
            world_id="duplicate_control",
            relationship="control",
            static_identifiability="control",
            view_a=np.ones((8, 2), dtype=float),
            view_b=np.ones((8, 2), dtype=float),
            targets={},
            ground_truth={},
        )
        with self.assertRaisesRegex(ValueError, "distances are invalid"):
            select_world_dimensions(world, self.policy)

    def test_result_is_json_portable(self) -> None:
        result = {
            world_id: select_world_dimensions(world, self.policy)
            for world_id, world in self.discovery.items()
        }
        encoded = json.dumps(result, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
