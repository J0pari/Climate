from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_planning, render_roadmap


class PlanningGraphTests(unittest.TestCase):
    def _graph(self):
        return {
            "schema_version": 1,
            "authority": "sole planning authority",
            "nodes": [
                {
                    "id": "a.done",
                    "title": "Done node",
                    "status": "done",
                    "priority": "P0",
                    "resource_class": "R2_toolchain_ci",
                    "summary": "A completed prerequisite.",
                    "depends_on": [],
                    "completion": ["Witness exists."],
                    "evidence": ["witness.txt"],
                },
                {
                    "id": "b.ready",
                    "title": "Ready node",
                    "status": "ready",
                    "priority": "P1",
                    "resource_class": "R2_toolchain_ci",
                    "summary": "A dependent obligation.",
                    "depends_on": ["a.done"],
                    "completion": ["Contract is satisfied."],
                },
            ],
        }

    def test_valid_graph_has_no_findings_and_renders_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "witness.txt").write_text("evidence\n", encoding="utf-8")
            graph = self._graph()
            self.assertEqual(check_planning.check(root, graph), [])
            rendered = render_roadmap.render(graph)
            self.assertIn("`a.done`", rendered)
            self.assertIn("`b.ready`", rendered)
            self.assertLess(rendered.index("`b.ready`"), rendered.index("`a.done`"))

    def test_missing_dependency_and_cycle_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "witness.txt").write_text("evidence\n", encoding="utf-8")
            graph = self._graph()
            graph["nodes"][0]["depends_on"] = ["b.ready"]
            graph["nodes"][1]["depends_on"] = ["a.done", "missing.node"]
            findings = check_planning.check(root, graph)
            codes = {finding.code for finding in findings}
            self.assertIn("planning.dependency_missing", codes)
            self.assertIn("planning.dependency_cycle", codes)

    def test_unresolved_dependency_requires_blocked_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "witness.txt").write_text("evidence\n", encoding="utf-8")
            graph = self._graph()
            graph["nodes"][0]["status"] = "active"
            findings = check_planning.check(root, graph)
            self.assertIn(
                "planning.status_ignores_dependencies",
                {finding.code for finding in findings},
            )

            graph["nodes"][1]["status"] = "blocked"
            findings = check_planning.check(root, graph)
            self.assertNotIn(
                "planning.blocked_without_cause",
                {finding.code for finding in findings},
            )

    def test_external_blocked_node_requires_explicit_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "witness.txt").write_text("evidence\n", encoding="utf-8")
            graph = self._graph()
            graph["nodes"][1]["depends_on"] = []
            graph["nodes"][1]["status"] = "blocked"
            findings = check_planning.check(root, graph)
            self.assertIn(
                "planning.blocked_without_cause",
                {finding.code for finding in findings},
            )

            graph["nodes"][1]["blockers"] = ["Requires external resource."]
            findings = check_planning.check(root, graph)
            self.assertNotIn(
                "planning.blocked_without_cause",
                {finding.code for finding in findings},
            )

    def test_done_nodes_require_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = self._graph()
            findings = check_planning.check(root, graph)
            codes = {finding.code for finding in findings}
            self.assertIn("planning.evidence_missing", codes)


if __name__ == "__main__":
    unittest.main()
