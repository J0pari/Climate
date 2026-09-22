from __future__ import annotations

import unittest

import numpy as np

from reference.multirepresentation_worlds import generate_worlds, load_fixture


class MultirepresentationStructuralWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture()
        cls.worlds = generate_worlds(cls.fixture)

    def test_suite_is_deterministic_and_complete(self) -> None:
        again = generate_worlds(self.fixture)
        expected = {
            "shared_manifold",
            "product",
            "fibered",
            "quotient_noninjective",
            "stratified_regime",
            "nuisance_dominated",
            "independent_null",
        }
        self.assertEqual(set(self.worlds), expected)
        for name in expected:
            np.testing.assert_array_equal(self.worlds[name].view_a, again[name].view_a)
            np.testing.assert_array_equal(self.worlds[name].view_b, again[name].view_b)

    def test_shared_manifold_has_one_recoverable_coordinate(self) -> None:
        world = self.worlds["shared_manifold"]
        shared = world.targets["shared"]
        np.testing.assert_allclose(world.view_a[:, 0], shared)
        np.testing.assert_allclose(np.arctan2(world.view_b[:, 0], world.view_b[:, 1]), shared)
        self.assertEqual(world.ground_truth["shared_dimension"], 1)

    def test_fibered_world_has_shared_base_and_private_fibers(self) -> None:
        world = self.worlds["fibered"]
        base = world.targets["shared_base"]
        np.testing.assert_allclose(world.view_a[:, 0], base)
        np.testing.assert_allclose(np.arctan2(world.view_b[:, 0], world.view_b[:, 1]), base)
        self.assertEqual(world.ground_truth["private_dimensions"], [1, 1])

    def test_quotient_world_loses_sign_in_one_view(self) -> None:
        world = self.worlds["quotient_noninjective"]
        signed = world.targets["signed_coordinate"]
        quotient = world.targets["shared_quotient"]
        np.testing.assert_allclose(quotient, signed**2)
        np.testing.assert_allclose(world.view_b[:, 0], quotient)
        self.assertEqual(world.ground_truth["lost_in_view_b"], "sign_of_signed_coordinate")

    def test_stratified_world_exercises_both_regimes(self) -> None:
        world = self.worlds["stratified_regime"]
        regime = world.targets["regime"]
        self.assertEqual(set(np.unique(regime).tolist()), {0, 1})
        self.assertEqual(world.ground_truth["stratum_count"], 2)

    def test_exact_support_topology_oracle_marks_only_stratified_view_b_disconnected(self) -> None:
        for world_id, world in self.worlds.items():
            expected_components = [1, 2] if world_id == "stratified_regime" else [1, 1]
            self.assertEqual(
                world.ground_truth["view_support_connected_components"],
                expected_components,
                msg=world_id,
            )
            self.assertEqual(
                world.ground_truth["view_support_first_betti_numbers"],
                [0, 0],
                msg=world_id,
            )
            self.assertEqual(
                world.ground_truth["topology_oracle_semantics"],
                "exact noiseless generator-support invariants before finite-sample estimation",
            )

    def test_stratified_view_b_has_an_exact_support_gap(self) -> None:
        world = self.worlds["stratified_regime"]
        regime = world.targets["regime"].astype(bool)
        lower = world.view_b[~regime, 0]
        upper = world.view_b[regime, 0]
        self.assertLess(float(np.max(lower)), 0.0)
        self.assertGreaterEqual(float(np.min(upper)), 1.0)

    def test_nuisance_dominated_world_makes_variance_an_adversary(self) -> None:
        world = self.worlds["nuisance_dominated"]
        shared_variance = float(np.var(world.targets["shared"]))
        nuisance_variance = float(np.var(world.targets["nuisance_a_0"]))
        self.assertGreater(nuisance_variance, 20.0 * shared_variance)

    def test_product_and_null_require_semantic_abstention(self) -> None:
        product = self.worlds["product"]
        null = self.worlds["independent_null"]
        self.assertFalse(product.ground_truth["static_dependence_can_prove_product_semantics"])
        self.assertFalse(null.ground_truth["method_should_invent_shared_structure"])
        self.assertEqual(product.ground_truth["shared_dimension"], 0)
        self.assertEqual(null.ground_truth["shared_dimension"], 0)

    def test_too_small_suite_fails_closed(self) -> None:
        fixture = load_fixture()
        fixture["sample_count"] = 16
        with self.assertRaisesRegex(ValueError, "at least"):
            generate_worlds(fixture)


if __name__ == "__main__":
    unittest.main()
