from __future__ import annotations

import unittest

import numpy as np

from reference.multirepresentation_common import generate_views, load_fixture, run_experiment


class MultirepresentationCommonReferenceTests(unittest.TestCase):
    def test_fixture_generation_is_deterministic_and_semantically_exact(self) -> None:
        fixture = load_fixture()
        first = generate_views(fixture)
        second = generate_views(fixture)

        np.testing.assert_array_equal(first.view_a, second.view_a)
        np.testing.assert_array_equal(first.view_b, second.view_b)
        np.testing.assert_array_equal(first.common_coordinate, second.common_coordinate)
        np.testing.assert_array_equal(first.observation_nuisance, second.observation_nuisance)

        expected_common = first.view_a[:, 0] + first.view_a[:, 1] ** 2
        np.testing.assert_allclose(first.common_coordinate, expected_common, rtol=0.0, atol=0.0)
        self.assertEqual(first.view_a.shape, (fixture["sample_count"], 2))
        self.assertEqual(first.view_b.shape, (fixture["sample_count"], 2))

    def test_common_manifold_candidate_passes_predeclared_controls(self) -> None:
        result = run_experiment(load_fixture())

        self.assertTrue(result["passed"], msg=result)
        self.assertTrue(all(result["checks"].values()), msg=result)
        self.assertGreater(
            result["metrics"]["jointly_smooth"]["shared_abs_spearman"],
            result["controls"]["best_baseline_shared_abs_spearman"],
        )
        self.assertGreater(
            result["metrics"]["jointly_smooth"]["shared_abs_spearman"],
            result["metrics"]["jointly_smooth"]["nuisance_abs_spearman"],
        )

    def test_invalid_sample_count_fails_closed(self) -> None:
        fixture = load_fixture()
        fixture["sample_count"] = 8
        with self.assertRaisesRegex(ValueError, "at least 64 samples"):
            generate_views(fixture)


if __name__ == "__main__":
    unittest.main()
