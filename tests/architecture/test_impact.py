from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from architecture.impact import inspect_impact, references_for_target


class ImpactInspectorTests(unittest.TestCase):
    def test_reverse_references_are_classified_without_new_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = "src/alpha_signal.py"
            (root / "src").mkdir()
            (root / target).write_text("VALUE = 1\n", encoding="utf-8")

            fixtures = {
                "architecture/modules/canonical.json": {
                    "schema_version": 2,
                    "modules": [{"path": target}],
                },
                "architecture/planning_graph.json": {
                    "schema_version": 1,
                    "nodes": [{"evidence": [target]}],
                },
                "methods/example.v1.json": {
                    "obligations": [{"witnesses": [target]}],
                },
            }
            for relative, payload in fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            text_fixtures = {
                ".github/workflows/example.yml": f"- '{target}'\n",
                "tests/test_alpha.py": f"# witness: {target}\n",
                "docs/ARCHITECTURE.md": f"Canonical implementation: `{target}`\n",
                "docs/generated/STATUS.md": f"| `{target}` | runnable |\n",
            }
            for relative, content in text_fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            refs = references_for_target(root, target)
            kinds = {item.kind for item in refs}
            self.assertEqual(
                kinds,
                {
                    "module_authority",
                    "planning_authority",
                    "scientific_authority",
                    "workflow",
                    "test",
                    "durable_contract",
                    "generated_projection",
                },
            )

            report = inspect_impact(root, [target])
            self.assertEqual(report["authority"], "none")
            self.assertTrue(report["targets"][0]["exists"])
            self.assertEqual(report["targets"][0]["reference_count"], 7)

    def test_target_file_does_not_count_as_its_own_reverse_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = "src/self_named.py"
            path = root / target
            path.parent.mkdir(parents=True)
            path.write_text(f"# {target}\n", encoding="utf-8")
            self.assertEqual(references_for_target(root, target), [])

    def test_unsafe_target_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                inspect_impact(root, ["/tmp/outside.py"])
            with self.assertRaisesRegex(ValueError, "must not contain"):
                inspect_impact(root, ["src/../outside.py"])


if __name__ == "__main__":
    unittest.main()
