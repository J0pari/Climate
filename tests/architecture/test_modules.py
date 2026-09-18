"""Negative witnesses for Climate module inventory integrity."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from architecture import check_modules


def claims(*ids: str):
    return {"claims": [{"claim_id": value} for value in ids]}


def module(path: str, **overrides):
    base = {
        "path": path,
        "family": "fixture",
        "language": check_modules.SOURCE_SUFFIXES.get(Path(path).suffix, "unknown"),
        "maturity": "prototype",
        "authority_kind": "canonical_implementation",
        "scientific_evidence_eligible": False,
        "intended_role": "fixture",
        "known_gaps": ["not verified"],
    }
    base.update(overrides)
    return base


class ModuleRegistryLoadingTests(unittest.TestCase):
    def write_fragment(self, directory: Path, name: str, version: int, modules):
        (directory / name).write_text(
            json.dumps({"schema_version": version, "modules": modules}),
            encoding="utf-8",
        )

    def test_fragment_directory_merges_in_filename_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_fragment(directory, "b.json", 1, [module("b.rs")])
            self.write_fragment(directory, "a.json", 1, [module("a.rs")])
            registry = check_modules.load_module_registry(directory)
            self.assertEqual([item["path"] for item in registry["modules"]], ["a.rs", "b.rs"])

    def test_fragment_schema_mismatch_fails_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_fragment(directory, "a.json", 1, [])
            self.write_fragment(directory, "b.json", 2, [])
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                check_modules.load_module_registry(directory)

    def test_duplicate_paths_across_fragments_reach_existing_integrity_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            fragments = root / "registry"
            fragments.mkdir()
            self.write_fragment(fragments, "a.json", 1, [module("x.rs")])
            self.write_fragment(fragments, "b.json", 1, [module("x.rs")])
            registry = check_modules.load_module_registry(fragments)
            findings = check_modules.check(root, registry, claims())
            self.assertIn("modules.path_duplicate", {f.code for f in findings})


class ModuleIntegrityTests(unittest.TestCase):
    def test_unregistered_top_level_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "new_method.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(root, {"modules": []}, claims())
            self.assertIn("modules.source_unregistered", {f.code for f in findings})

    def test_unregistered_nested_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "future_package" / "solver.rs"
            nested.parent.mkdir(parents=True)
            nested.write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(root, {"modules": []}, claims())
            matching = [f for f in findings if f.code == "modules.source_unregistered"]
            self.assertEqual([f.path for f in matching], ["future_package/solver.rs"])

    def test_reference_source_is_part_of_module_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference" / "oracle.py"
            reference.parent.mkdir(parents=True)
            reference.write_text("pass\n", encoding="utf-8")
            findings = check_modules.check(root, {"modules": []}, claims())
            matching = [f for f in findings if f.code == "modules.source_unregistered"]
            self.assertEqual([f.path for f in matching], ["reference/oracle.py"])

    def test_architecture_file_cannot_be_registered_as_scientific_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "architecture" / "gate.py"
            path.parent.mkdir(parents=True)
            path.write_text("pass\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("architecture/gate.py")]},
                claims(),
            )
            self.assertIn("modules.path_not_module_surface", {f.code for f in findings})

    def test_duplicate_module_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs"), module("x.rs")]},
                claims(),
            )
            self.assertIn("modules.path_duplicate", {f.code for f in findings})

    def test_language_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", language="python")]},
                claims(),
            )
            self.assertIn("modules.language_mismatch", {f.code for f in findings})

    def test_missing_authority_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            record = module("x.rs")
            del record["authority_kind"]
            findings = check_modules.check(root, {"modules": [record]}, claims())
            self.assertIn("modules.authority_kind_missing", {f.code for f in findings})

    def test_reference_authority_kind_must_match_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "reference" / "x.py"
            path.parent.mkdir(parents=True)
            path.write_text("pass\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module(
                    "reference/x.py",
                    authority_kind="canonical_implementation",
                )]},
                claims(),
            )
            self.assertIn("modules.authority_kind_mismatch", {f.code for f in findings})

    def test_missing_claim_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", claim_ids=["missing"])]},
                claims("known"),
            )
            self.assertIn("modules.claim_missing", {f.code for f in findings})

    def test_prototype_requires_known_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", known_gaps=[])]},
                claims(),
            )
            self.assertIn("modules.known_gaps_missing", {f.code for f in findings})

    def test_unverified_module_cannot_be_evidence_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module(
                    "x.rs",
                    scientific_evidence_eligible=True,
                    claim_ids=["c"],
                )]},
                claims("c"),
            )
            self.assertIn("modules.evidence_eligibility_too_early", {f.code for f in findings})

    def test_evidence_eligible_module_requires_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module(
                    "x.rs",
                    maturity="verified",
                    scientific_evidence_eligible=True,
                    known_gaps=[],
                )]},
                claims(),
            )
            self.assertIn("modules.evidence_eligibility_without_claim", {f.code for f in findings})

    def test_registered_prototype_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", claim_ids=["c"])]},
                claims("c"),
            )
            self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
