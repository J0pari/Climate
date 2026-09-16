"""Negative witnesses for Climate static source gates.

Each gate must prove that it rejects/catches the failure class it exists to
prevent. These tests use tiny in-memory source maps; they do not assert that the
current legacy repository is already clean.
"""
from __future__ import annotations

import unittest

from architecture import source_gates as gates


class SourceGateTests(unittest.TestCase):
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

    def test_capability_stub_is_inventoried(self):
        findings = gates.gate_silent_capability_fallback({
            "gpu/example.cu": ["// NCCL stub used for single-device parsing"],
        })
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].gate, "capability_fallback")

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

    def test_combined_runner_preserves_multiple_failure_classes(self):
        findings = gates.run({
            "gpu/example.cu": [
                "cudaMalloc(&p, bytes);",
                "cudaMallocManaged(&q, bytes);",
                "// CPU fallback path",
                "probability = score / 2.0;",
            ]
        })
        names = {f.gate for f in findings}
        self.assertIn("unchecked_cuda_call", names)
        self.assertIn("managed_memory", names)
        self.assertIn("capability_fallback", names)
        self.assertIn("interpretive_probability", names)


if __name__ == "__main__":
    unittest.main()
