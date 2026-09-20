from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_repository_state, state_authorities

ROOT = Path(__file__).resolve().parents[2]


class RepositoryStateAuthorityTests(unittest.TestCase):
    def test_repository_state_contract_is_clean(self) -> None:
        self.assertEqual(check_repository_state.check(ROOT), [])

    def test_state_surfaces_are_minimal(self) -> None:
        manifest = state_authorities.load_manifest(
            ROOT / "architecture" / "state_authorities.json"
        )
        self.assertEqual(
            {surface["id"] for surface in manifest["surfaces"]},
            {
                "planning",
                "structural_realization",
                "experiment_definitions",
                "evaluation_records",
                "claim_evidence",
                "execution",
                "history",
            },
        )
        self.assertEqual(
            set(manifest["orientation_view"]["includes"]),
            {
                "planning",
                "structural_realization",
                "experiment_definitions",
                "evaluation_records",
                "claim_evidence",
            },
        )
        self.assertEqual(
            set(manifest["orientation_view"]["excludes"]),
            {"execution", "history"},
        )
        self.assertEqual(manifest["orientation_view"]["projection"], "docs/generated/STATE.md")
        self.assertEqual(manifest["orientation_view"]["renderer"], "architecture/render_state.py")

    def test_repository_state_includes_planning_evaluations_and_method_authorities(self) -> None:
        manifest = state_authorities.load_manifest(
            ROOT / "architecture" / "state_authorities.json"
        )
        typed = state_authorities.orientation_authority_paths(ROOT, manifest)
        paths = {
            path.relative_to(ROOT).as_posix()
            for surface_paths in typed.values()
            for path in surface_paths
        }
        self.assertIn("architecture/planning_graph.json", paths)
        self.assertIn("methods/registry.json", paths)
        self.assertIn("claims/registry.json", paths)
        self.assertIn("evidence/registry.json", paths)
        self.assertIn(
            "evaluations/information-geometry/two-layer-ebm-recovery-confirmation-v1/result.json",
            paths,
        )

    def test_orientation_composition_does_not_promote_evaluations(self) -> None:
        manifest = state_authorities.load_manifest(
            ROOT / "architecture" / "state_authorities.json"
        )
        rule = manifest["orientation_view"]["composition_rule"].lower()
        self.assertIn("evaluation record", rule)
        self.assertIn("does not become scientific evidence", rule)

    def test_worker_orientation_requires_current_main_and_exact_head_actions(self) -> None:
        manifest = state_authorities.load_manifest(
            ROOT / "architecture" / "state_authorities.json"
        )
        orientation = " ".join(manifest["worker_orientation"]).lower()
        self.assertIn("current main commit", orientation)
        self.assertIn("intervening commits", orientation)
        self.assertIn("exact-head github actions", orientation)


    def test_hosted_ci_directory_is_absent(self) -> None:
        self.assertFalse((ROOT / ".github" / "workflows").exists())


if __name__ == "__main__":
    unittest.main()
