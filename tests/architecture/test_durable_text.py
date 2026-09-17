from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_durable_text


class DurableTextTests(unittest.TestCase):
    def test_edit_history_markers_are_rejected(self):
        examples = {
            "docs/example.md": "Recently added a new coupling layer.\n",
            "src/example.rs": "// Additive: helper introduced during cleanup\n",
            "src/example.cu": "// Batch2 additive: occupancy diagnostic\n",
            "reference/example.py": "# replaces previously inlined logic\n",
            "README.md": "The repository now has a canonical solver.\n",
            "src/provisional.rs": "// Use timeout-based detection for now.\n",
        }
        observed = {
            finding.rule
            for path, text in examples.items()
            for finding in check_durable_text.findings_for_text(path, text)
        }
        self.assertEqual(
            observed,
            {
                "edit_batch_annotation",
                "edit_batch_label",
                "replacement_history",
                "recent_change_narration",
                "repository_temporal_snapshot",
                "provisional_for_now",
            },
        )

    def test_scientific_additive_language_is_allowed(self):
        text = (
            "The additive source is evaluated by the caller.\n"
            "A previous state may be used as an initial guess.\n"
            "The current implementation rejects singular metrics.\n"
        )
        self.assertEqual(
            check_durable_text.findings_for_text("src/example.f90", text),
            [],
        )

    def test_active_docs_and_source_are_scanned_but_archives_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs" / "archive").mkdir(parents=True)
            (root / "docs" / "generated").mkdir(parents=True)
            (root / "src").mkdir()
            (root / "reference").mkdir()

            (root / "README.md").write_text("Stable description.\n", encoding="utf-8")
            (root / "docs" / "active.md").write_text(
                "Recently removed a stale table.\n", encoding="utf-8"
            )
            (root / "docs" / "archive" / "history.md").write_text(
                "Recently added an experiment.\n", encoding="utf-8"
            )
            (root / "docs" / "generated" / "STATUS.md").write_text(
                "The repository now has ten modules.\n", encoding="utf-8"
            )
            (root / "src" / "lib.rs").write_text(
                "// Additive: helper\n", encoding="utf-8"
            )
            (root / "reference" / "oracle.py").write_text(
                "# stable mathematical oracle\n", encoding="utf-8"
            )

            findings = check_durable_text.check_repository(root)
            self.assertEqual(
                {(finding.path, finding.rule) for finding in findings},
                {
                    ("docs/active.md", "recent_change_narration"),
                    ("src/lib.rs", "edit_batch_annotation"),
                },
            )


if __name__ == "__main__":
    unittest.main()
