from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RenderStateTests(unittest.TestCase):
    def test_stale_state_reports_unified_diff(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".render-state-test-", dir=ROOT) as tmpdir:
            output = Path(tmpdir) / "STATE.md"
            output.write_text("# stale state\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "architecture" / "render_state.py"),
                    "--check",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("-# stale state", result.stdout)
        self.assertIn("+# Generated research state", result.stdout)

    def test_state_unifies_planning_and_committed_evaluations(self) -> None:
        state = (ROOT / "docs" / "generated" / "STATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("## Planning frontier", state)
        self.assertIn("## Committed evaluations", state)
        self.assertIn("architecture/planning_graph.json", state)
        self.assertIn(
            "evaluations/information-geometry/two-layer-ebm-recovery-confirmation-v1/result.json",
            state,
        )

    def test_state_does_not_claim_exact_head_execution(self) -> None:
        state = (ROOT / "docs" / "generated" / "STATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Exact-head build, test, and workflow outcomes live in GitHub Actions", state)
        self.assertIn("does not promote evidence or infer CI outcomes", state)


if __name__ == "__main__":
    unittest.main()
