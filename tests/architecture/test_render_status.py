from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RenderStatusTests(unittest.TestCase):
    def test_stale_projection_reports_unified_diff(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".render-status-test-", dir=ROOT) as tmpdir:
            output = Path(tmpdir) / "STATUS.md"
            output.write_text("# stale projection\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "architecture" / "render_status.py"),
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
        self.assertIn("generated status stale:", result.stdout)
        self.assertIn("--- ", result.stdout)
        self.assertIn("+++ rendered status", result.stdout)
        self.assertIn("-# stale projection", result.stdout)
        self.assertIn("+# Generated repository status", result.stdout)


if __name__ == "__main__":
    unittest.main()
