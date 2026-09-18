"""Reference witnesses for external contract-reasoning scoring."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reference import external_contract_reasoning as scorer


class ExternalContractReasoningTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(scorer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        task_path, self.tasks = scorer._resolve_task_set(self.spec, scorer.ROOT)
        self.task_digest = scorer._sha256(task_path)
        self.subject = "a" * 64

    def predictions(self, *, perfect: bool = True) -> dict:
        responses = []
        for task in self.tasks["tasks"]:
            choice = task["correct_choice"]
            if not perfect:
                choice = scorer._ROTATED[choice]
            responses.append({"task_id": task["task_id"], "choice": choice})
        return {
            "schema": self.spec["prediction_contract"],
            "evaluation_id": self.spec["evaluation_id"],
            "subject_digest": self.subject,
            "runtime": {
                "interface": self.spec["adapter"]["interface"],
                "implementation": "fixture-runtime",
                "subject_digest": self.subject,
            },
            "responses": responses,
        }

    def test_perfect_predictions_score_against_negative_control(self):
        result = scorer.score_predictions(
            self.predictions(),
            spec=self.spec,
            task_set=self.tasks,
            task_set_digest=self.task_digest,
        )
        self.assertEqual(result["metrics"]["contract_choice_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["unsafe_semantic_upgrade_rate"], 0.0)
        self.assertEqual(
            result["negative_controls"]["permuted_answer_key"]["accuracy"],
            0.0,
        )
        self.assertEqual(result["subject_digest"], self.subject)
        self.assertTrue(result["prediction_digest"].startswith("sha256:"))

    def test_runtime_subject_digest_mismatch_refuses(self):
        predictions = self.predictions()
        predictions["runtime"]["subject_digest"] = "b" * 64
        with self.assertRaisesRegex(
            scorer.ContractReasoningError, "runtime subject digest"
        ):
            scorer.score_predictions(
                predictions,
                spec=self.spec,
                task_set=self.tasks,
                task_set_digest=self.task_digest,
            )

    def test_missing_task_refuses_instead_of_scoring_partial_surface(self):
        predictions = self.predictions()
        predictions["responses"].pop()
        with self.assertRaisesRegex(scorer.ContractReasoningError, "incomplete"):
            scorer.score_predictions(
                predictions,
                spec=self.spec,
                task_set=self.tasks,
                task_set_digest=self.task_digest,
            )

    def test_permuted_predictions_do_not_look_like_observed_skill(self):
        result = scorer.score_predictions(
            self.predictions(perfect=False),
            spec=self.spec,
            task_set=self.tasks,
            task_set_digest=self.task_digest,
        )
        self.assertEqual(result["metrics"]["contract_choice_accuracy"], 0.0)
        self.assertEqual(
            result["negative_controls"]["permuted_answer_key"]["accuracy"],
            1.0,
        )
        self.assertLess(
            result["negative_controls"]["permuted_answer_key"]["delta_vs_observed"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
