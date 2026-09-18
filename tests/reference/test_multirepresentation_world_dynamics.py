from __future__ import annotations

import json
import unittest

import numpy as np

from reference.multirepresentation_worlds import (
    DEFAULT_FIXTURE,
    bind_world_fixture_authority,
    generate_world_trajectories,
    generate_worlds,
    git_blob_sha,
    load_dynamics_fixture,
    load_fixture,
    world_set_digest,
)


class MultirepresentationWorldDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world_fixture = load_fixture()
        cls.dynamics_fixture = load_dynamics_fixture()
        cls.static = generate_worlds(cls.world_fixture)
        cls.discovery = generate_world_trajectories(
            cls.world_fixture,
            cls.dynamics_fixture,
        )
        cls.confirmation = generate_world_trajectories(
            cls.world_fixture,
            cls.dynamics_fixture,
            confirmation=True,
        )

    def test_dynamics_policy_references_world_authority_without_world_table(self) -> None:
        self.assertEqual(
            self.dynamics_fixture["world_fixture_id"],
            self.world_fixture["fixture_id"],
        )
        self.assertEqual(
            self.dynamics_fixture["world_fixture_git_blob_sha"],
            git_blob_sha(DEFAULT_FIXTURE),
        )
        self.assertNotIn("worlds", self.dynamics_fixture)
        self.assertEqual(
            set(self.dynamics_fixture["role_processes"]),
            {"shared", "private", "nuisance"},
        )
        bind_world_fixture_authority(
            self.world_fixture,
            self.dynamics_fixture,
        )

    def test_static_and_temporal_samplers_share_world_semantics(self) -> None:
        self.assertEqual(set(self.static), set(self.discovery))
        for world_id in self.static:
            static = self.static[world_id]
            dynamic = self.discovery[world_id]
            self.assertEqual(static.relationship, dynamic.relationship)
            self.assertEqual(
                static.static_identifiability,
                dynamic.static_identifiability,
            )
            self.assertEqual(static.ground_truth, dynamic.ground_truth)
            self.assertEqual(
                dynamic.view_a.shape[0],
                self.dynamics_fixture["trajectory_steps"],
            )
            self.assertEqual(
                dynamic.view_b.shape[0],
                self.dynamics_fixture["trajectory_steps"],
            )

    def test_discovery_and_confirmation_trajectories_are_isolated(self) -> None:
        self.assertNotEqual(
            world_set_digest(self.discovery),
            world_set_digest(self.confirmation),
        )
        for world_id in self.discovery:
            self.assertEqual(
                self.discovery[world_id].ground_truth,
                self.confirmation[world_id].ground_truth,
            )

    def test_shared_role_has_declared_slow_temporal_memory(self) -> None:
        expected_rho = float(
            self.dynamics_fixture["role_processes"]["shared"]["rho"]
        )
        self.assertGreater(expected_rho, 0.9)
        for world in self.discovery.values():
            target_name = world.ground_truth["shared_evaluation_target_name"]
            if target_name is None:
                continue
            target = np.asarray(world.targets[target_name], dtype=float)
            correlation = float(np.corrcoef(target[:-1], target[1:])[0, 1])
            self.assertGreater(correlation, 0.5, msg=world.world_id)

    def test_static_generation_remains_deterministic_after_renderer_refactor(self) -> None:
        first = generate_worlds(self.world_fixture)
        second = generate_worlds(self.world_fixture)
        self.assertEqual(world_set_digest(first), world_set_digest(second))
        for world_id in first:
            np.testing.assert_array_equal(
                first[world_id].view_a,
                second[world_id].view_a,
            )
            np.testing.assert_array_equal(
                first[world_id].view_b,
                second[world_id].view_b,
            )

    def test_dynamics_fixture_is_json_portable(self) -> None:
        encoded = json.dumps(self.dynamics_fixture, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
