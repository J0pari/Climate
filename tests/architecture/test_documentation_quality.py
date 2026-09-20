from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_documentation_quality


class DocumentationQualityTests(unittest.TestCase):
    def test_empty_quality_self_certification_is_rejected(self) -> None:
        examples = {
            "README.md": "We study the system rigorously.\n",
            "docs/a.md": "This is a technically serious framework.\n",
            "docs/b.md": "A sophisticated method handles the problem.\n",
            "docs/c.md": "The pipeline is trustworthy.\n",
            "docs/d.md": "This is a scientifically meaningful task.\n",
        }
        rules = {
            finding.rule
            for path, text in examples.items()
            for finding in check_documentation_quality.findings_for_text(path, text)
        }
        self.assertEqual(
            rules,
            {
                "quality.rigor_self_certification",
                "quality.prestige_adjective",
                "quality.vague_scientific_meaning",
            },
        )

    def test_temporal_repository_snapshot_language_is_rejected(self) -> None:
        text = (
            "The repository currently supports a solver.\n"
            "At present the benchmark uses one model.\n"
            "The repository already contains the witness.\n"
            "The repository now provides the runtime.\n"
        )
        rules = {
            finding.rule
            for finding in check_documentation_quality.findings_for_text(
                "docs/example.md", text
            )
        }
        self.assertEqual(
            rules,
            {
                "quality.temporal_currently",
                "quality.temporal_at_present",
                "quality.temporal_already_contains",
                "quality.temporal_now_capability",
            },
        )

    def test_concrete_criteria_are_allowed(self) -> None:
        text = (
            "The candidate is retained only when the preregistered held-out metric "
            "improves over the declared baseline.\n"
            "The operator is verified against a manufactured solution and a discrete "
            "budget residual tolerance.\n"
            "The local verification receipt names the commit SHA it executed.\n"
        )
        self.assertEqual(
            check_documentation_quality.findings_for_text("docs/example.md", text),
            [],
        )

    def test_repository_documentation_is_clean(self) -> None:
        self.assertEqual(
            check_documentation_quality.check_repository(),
            [],
        )

    def test_archives_are_not_active_documentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs" / "archive").mkdir(parents=True)
            (root / "docs").mkdir(exist_ok=True)
            (root / "README.md").write_text("Concrete repository purpose.\n")
            (root / "docs" / "active.md").write_text("Declared metric contract.\n")
            (root / "docs" / "archive" / "old.md").write_text(
                "A rigorous sophisticated framework.\n"
            )
            paths = {
                path.relative_to(root).as_posix()
                for path in check_documentation_quality.documentation_paths(root)
            }
            self.assertEqual(paths, {"README.md", "docs/active.md"})


if __name__ == "__main__":
    unittest.main()
