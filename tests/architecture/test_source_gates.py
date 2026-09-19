"""Negative witnesses for Climate static source gates.

Each gate must prove that it rejects/catches the failure class it exists to
prevent. These tests use tiny in-memory source maps or temporary repository
surfaces; they do not assert that the current legacy repository is already clean.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import source_gates as gates


class SourceGateTests(unittest.TestCase):
    def test_shared_surface_includes_nested_science_and_excludes_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            science = root / "future_package" / "solver.rs"
            science.parent.mkdir(parents=True)
            science.write_text("fn x() {}\n", encoding="utf-8")
            reference = root / "reference" / "oracle.py"
            reference.parent.mkdir(parents=True)
            reference.write_text("pass\n", encoding="utf-8")
            control = root / "architecture" / "gate.py"
            control.parent.mkdir(parents=True)
            control.write_text("pass\n", encoding="utf-8")
            test = root / "tests" / "test_gate.py"
            test.parent.mkdir(parents=True)
            test.write_text("pass\n", encoding="utf-8")

            observed = {
                path.relative_to(root.resolve()).as_posix()
                for path in gates.production_files(root)
            }
            self.assertEqual(observed, {"future_package/solver.rs", "reference/oracle.py"})

    def test_managed_memory_is_detected(self):
        findings = gates.gate_managed_memory({
            "gpu/example.cu": ["void* p; cudaMallocManaged(&p, 4096);"],
        })
        self.assertEqual([f.gate for f in findings], ["managed_memory"])

    def test_checked_cuda_call_is_not_flagged_but_raw_call_is(self):
        checked = gates.gate_unchecked_cuda_calls({
            "gpu/example.cu": ["CUDA_CHECK(cudaMalloc(&p, bytes));"],
        })
        raw = gates.gate_unchecked_cuda_calls({
            "gpu/example.cu": ["cudaMalloc(&p, bytes);"],
        })
        self.assertEqual(checked, [])
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0].gate, "unchecked_cuda_call")

    def test_fallback_marker_is_supplemental_inventory(self):
        files = {
            "gpu/example.cu": ["// CPU fallback path"],
        }
        findings = gates.gate_fallback_marker_inventory(files)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].gate, "fallback_marker_inventory")
        self.assertEqual(
            [item.gate for item in gates.run(files)],
            ["fallback_marker_inventory"],
        )
        self.assertEqual(gates.run_strict(files), [])

    def test_placeholder_marker_is_inventoried(self):
        findings = gates.gate_placeholder_inventory({
            "physics/example.f90": ["! TODO: placeholder tendency"],
        })
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].gate, "placeholder_inventory")

    def test_ambient_rng_is_detected(self):
        findings = gates.gate_ambient_rng({
            "methods/example.rs": ["let mut rng = rand::thread_rng();"],
        })
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].gate, "ambient_rng")

    def test_explicit_seeded_rng_example_is_not_detected(self):
        findings = gates.gate_ambient_rng({
            "methods/example.py": ["rng = np.random.default_rng(seed)"],
        })
        self.assertEqual(findings, [])

    def test_heuristic_probability_mapping_is_detected(self):
        findings = gates.gate_interpretive_probability({
            "methods/example.rs": [
                "probability: 1.0 / (1.0 + (-lambda).exp()),"
            ],
        })
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].gate, "interpretive_probability")

    def test_plain_diagnostic_is_not_mislabeled_probability(self):
        findings = gates.gate_interpretive_probability({
            "methods/example.rs": ["curvature_score = lambda.abs();"],
        })
        self.assertEqual(findings, [])

    def test_change_narration_is_inventoried(self):
        findings = gates.gate_change_narration({
            "legacy/example.rs": ["// Batch2 additive: recovery path"],
        })
        self.assertGreaterEqual(len(findings), 1)
        self.assertEqual({finding.gate for finding in findings}, {"change_narration"})
        self.assertEqual({finding.path for finding in findings}, {"legacy/example.rs"})

    def test_scientific_additive_language_is_not_change_narration(self):
        findings = gates.gate_change_narration({
            "physics/example.f90": ["! The additive source is caller-owned."],
        })
        self.assertEqual(findings, [])

    def test_combined_runner_preserves_binding_and_supplemental_classes(self):
        files = {
            "gpu/example.cu": [
                "cudaMalloc(&p, bytes);",
                "cudaMallocManaged(&q, bytes);",
                "// CPU fallback path",
                "probability = score / 2.0;",
                "// Additive: compatibility helper",
            ]
        }
        findings = gates.run(files)
        names = {f.gate for f in findings}
        self.assertIn("unchecked_cuda_call", names)
        self.assertIn("managed_memory", names)
        self.assertIn("fallback_marker_inventory", names)
        self.assertIn("interpretive_probability", names)
        self.assertIn("change_narration", names)

        strict_names = {f.gate for f in gates.run_strict(files)}
        self.assertNotIn("fallback_marker_inventory", strict_names)
        self.assertIn("unchecked_cuda_call", strict_names)


if __name__ == "__main__":
    unittest.main()
