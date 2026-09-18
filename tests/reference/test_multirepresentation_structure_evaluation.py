from __future__ import annotations

import json
import unittest

import numpy as np

from reference.multirepresentation_structure_evaluation import (
    DEFAULT_WORLD_FIXTURE,
    _git_blob_sha,
    evaluate_structural_worlds,
    load_evaluation_fixture,
)
from reference.multirepresentation_worlds import generate_worlds, load_fixture


class MultirepresentationStructureEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world_fixture = load_fixture()
        cls.evaluation_fixture = load_evaluation_fixture()
        cls.discovery = generate_worlds(cls.world_fixture)
        cls.confirmation = generate_worlds(
            cls.world_fixture,
            seed_offset=cls.evaluation_fixture["confirmation_seed_offset"],
        )
        cls.result = evaluate_structural_worlds(
            cls.world_fixture,
            cls.evaluation_fixture,
        )

    def test_evaluation_policy_binds_exact_authoritative_world_fixture(self) -> None:
        self.assertEqual(
            self.evaluation_fixture["world_fixture_id"],
            self.world_fixture["fixture_id"],
        )
        self.assertEqual(
            self.evaluation_fixture["world_fixture_git_blob_sha"],
            _git_blob_sha(DEFAULT_WORLD_FIXTURE),
        )

    def test_world_authority_owns_shared_evaluation_target_semantics(self) -> None:
        for world in self.discovery.values():
            target_name = world.ground_truth["shared_evaluation_target_name"]
            if target_name is None:
                self.assertEqual(world.ground_truth["shared_dimension"], 0)
            else:
                self.assertGreater(world.ground_truth["shared_dimension"], 0)
                self.assertIn(target_name, world.targets)

    def test_confirmation_changes_samples_without_changing_world_semantics(self) -> None:
        self.assertEqual(set(self.discovery), set(self.confirmation))
        changed = 0
        for world_id in self.discovery:
            first = self.discovery[world_id]
            second = self.confirmation[world_id]
            self.assertEqual(first.relationship, second.relationship)
            self.assertEqual(first.static_identifiability, second.static_identifiability)
            self.assertEqual(first.ground_truth, second.ground_truth)
            if not np.array_equal(first.view_a, second.view_a):
                changed += 1
            if not np.array_equal(first.view_b, second.view_b):
                changed += 1
        self.assertGreater(changed, 0)

    def test_selector_recovers_or_abstains_from_authoritative_target_presence(self) -> None:
        self.assertTrue(
            self.result["selector_summary"]["all_discovery_decisions_calibrated"],
            msg=self.result["worlds"],
        )
        self.assertTrue(
            self.result["selector_summary"]["all_confirmation_decisions_calibrated"],
            msg=self.result["worlds"],
        )
        for world_id, item in self.result["worlds"].items():
            if item["shared_evaluation_target_name"] is None:
                self.assertEqual(
                    item["confirmation_selector"]["decision"],
                    "abstain_no_cross_view_evidence",
                    msg=world_id,
                )
            else:
                self.assertEqual(
                    item["confirmation_selector"]["decision"],
                    "shared_coordinate_supported",
                    msg=world_id,
                )

    def test_coordinate_baselines_do_not_own_truth_or_abstention_policy(self) -> None:
        required = {
            "concat_pca",
            "linear_cca",
            "concat_spectral",
            "factor_analysis",
            "jointly_smooth",
        }
        for world_id, item in self.result["worlds"].items():
            baselines = item["confirmation_coordinate_baselines"]
            self.assertEqual(set(baselines), required)
            for method_id in (
                "concat_pca",
                "linear_cca",
                "concat_spectral",
                "factor_analysis",
            ):
                self.assertEqual(
                    baselines[method_id]["status"],
                    "ok",
                    msg=(world_id, method_id, baselines[method_id]),
                )
            if item["shared_evaluation_target_name"] is not None:
                for method_id, diagnostics in baselines.items():
                    if diagnostics["status"] == "ok":
                        self.assertTrue(
                            np.isfinite(
                                diagnostics["shared_target_abs_spearman"]
                            ),
                            msg=(world_id, method_id),
                        )
            else:
                for diagnostics in baselines.values():
                    self.assertNotIn("shared_target_abs_spearman", diagnostics)

    def test_matched_information_probe_uses_one_policy_across_raw_and_simple_latents(self) -> None:
        required = [
            "raw_concat",
            "concat_pca",
            "linear_cca",
            "factor_analysis",
        ]
        self.assertEqual(
            self.evaluation_fixture["matched_information_probe"]["representations"],
            required,
        )
        for world_id, item in self.result["worlds"].items():
            target_name = item["shared_evaluation_target_name"]
            probe = item["matched_information_probe"]
            if target_name is None:
                self.assertIsNone(probe, msg=world_id)
                continue
            self.assertEqual(list(probe), required)
            self.assertGreater(probe["raw_concat"]["representation_dimension"], 1)
            for representation, diagnostics in probe.items():
                self.assertTrue(
                    np.isfinite(diagnostics["confirmation_normalized_rmse"]),
                    msg=(world_id, representation),
                )
                self.assertTrue(
                    np.isfinite(diagnostics["confirmation_abs_spearman"]),
                    msg=(world_id, representation),
                )
                self.assertGreaterEqual(
                    diagnostics["confirmation_normalized_rmse"],
                    0.0,
                )
                self.assertGreaterEqual(
                    diagnostics["confirmation_abs_spearman"],
                    0.0,
                )
                self.assertLessEqual(
                    diagnostics["confirmation_abs_spearman"],
                    1.0,
                )

    def test_discovery_and_confirmation_samples_have_distinct_digests(self) -> None:
        self.assertNotEqual(
            self.result["discovery_sample_digest"],
            self.result["confirmation_sample_digest"],
        )

    def test_result_is_json_portable(self) -> None:
        encoded = json.dumps(self.result, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", encoded)


if __name__ == "__main__":
    unittest.main()
