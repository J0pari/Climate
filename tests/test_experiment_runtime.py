from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.experiment_runtime import DEFAULT_EXPERIMENT, run_experiment


class ExperimentRuntimeTests(unittest.TestCase):
    def test_ebm_experiment_emits_common_contract_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            outcome = run_experiment(
                experiment_path=DEFAULT_EXPERIMENT,
                output_dir=output,
                repository_revision="a" * 40,
                run_scope="fixture-run",
            )
            self.assertEqual(outcome["experiment_id"], "multirepresentation.ebm_dynamics.v1")
            self.assertEqual(len(outcome["runs"]), 2)
            self.assertEqual(outcome["evidence"], [])
            self.assertTrue(all(run["scientific_output_eligible"] for run in outcome["runs"]))
            self.assertTrue(all(run["execution"]["status"] == "eligible" for run in outcome["runs"]))

            condition = [
                metric
                for metric in outcome["metrics"]
                if metric["metric"]["metric_id"]
                == "multirepresentation.ebm.concat.condition_number"
            ]
            self.assertEqual(len(condition), 2)
            self.assertTrue(all(item["status"] == "rank_deficient" for item in condition))
            self.assertTrue(all("value" not in item for item in condition))

            for artifact in outcome["artifacts"]:
                data = (output / artifact["uri"]).read_bytes()
                self.assertEqual(
                    artifact["digest"],
                    "sha256:" + hashlib.sha256(data).hexdigest(),
                )

            for filename in (
                "baseline-exact-modes.json",
                "candidate-representation-dynamics.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                text = (output / filename).read_text(encoding="utf-8")
                self.assertNotIn("Infinity", text)
                self.assertNotIn("NaN", text)
                json.loads(text)


if __name__ == "__main__":
    unittest.main()
