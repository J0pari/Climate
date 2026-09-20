from __future__ import annotations

import unittest
from pathlib import Path

from architecture import check_repository_state, state_authorities

ROOT = Path(__file__).resolve().parents[2]


class RepositoryStateAuthorityTests(unittest.TestCase):
    def test_repository_state_contract_is_clean(self) -> None:
        self.assertEqual(check_repository_state.check(ROOT), [])

    def test_freshness_is_commit_scoped_and_cache_invalidating(self) -> None:
        manifest = state_authorities.load_manifest(
            ROOT / "architecture" / "state_authorities.json"
        )
        freshness = manifest["freshness_semantics"]
        self.assertIs(freshness["commit_scoped"], True)
        self.assertIs(freshness["head_movement_invalidates_cached_state"], True)
        sequence = " ".join(manifest["reorientation_sequence"]).lower()
        self.assertIn("exact current main commit sha", sequence)
        self.assertIn("inspect the intervening commits", sequence)
        self.assertIn("github actions for the exact sha", sequence)

    def test_generated_projections_have_explicitly_limited_scope(self) -> None:
        manifest = state_authorities.load_manifest(
            ROOT / "architecture" / "state_authorities.json"
        )
        planning = state_authorities.surface_by_id(manifest, "planning")
        structural = state_authorities.surface_by_id(
            manifest, "structural_realization"
        )
        self.assertEqual(planning["projection"], "docs/ROADMAP.md")
        self.assertEqual(structural["projection"], "docs/generated/STATUS.md")
        self.assertIn("exact-head CI state", planning["excludes"])
        self.assertIn("scientific evaluation outcomes", planning["excludes"])
        self.assertIn("repository evaluation records not promoted through evidence/claim authorities", structural["excludes"])
        self.assertIn("commit history", structural["excludes"])

    def test_projection_authority_inputs_are_nonempty_and_fingerprintable(self) -> None:
        manifest = state_authorities.load_manifest(
            ROOT / "architecture" / "state_authorities.json"
        )
        for surface_id in ("planning", "structural_realization"):
            paths = state_authorities.projection_authority_paths(
                ROOT, manifest, surface_id
            )
            self.assertTrue(paths)
            fingerprint = state_authorities.projection_fingerprint(
                ROOT, manifest, surface_id
            )
            self.assertRegex(fingerprint, r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
