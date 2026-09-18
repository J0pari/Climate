from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from src.experiment_runtime import DEFAULT_EXPERIMENT, ROOT, run_experiment


FORCING_EXPERIMENT = ROOT / "experiments" / "two-layer-ebm-forcing-protocols.v1.json"
FORCED_OOD_EXPERIMENT = ROOT / "experiments" / "multirepresentation-ebm-forced-ood.v1.json"


class ExperimentRuntimeTests(unittest.TestCase):
    def assert_portable_json_tree(self, path: Path) -> None:
        def assert_finite_numbers(value):
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))
            elif isinstance(value, dict):
                for nested in value.values():
                    assert_finite_numbers(nested)
            elif isinstance(value, list):
                for nested in value:
                    assert_finite_numbers(nested)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert_finite_numbers(payload)

    def assert_artifact_digests(self, outcome: dict, output: Path) -> None:
        for artifact in outcome["artifacts"]:
            data = (output / artifact["uri"]).read_bytes()
            self.assertEqual(
                artifact["digest"],
                "sha256:" + hashlib.sha256(data).hexdigest(),
            )

    def test_ebm_dynamics_experiment_emits_common_contract_surface(self) -> None:
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

            self.assert_artifact_digests(outcome, output)
            for filename in (
                "baseline-exact-modes.json",
                "candidate-representation-dynamics.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                self.assert_portable_json_tree(output / filename)

    def test_ebm_forcing_experiment_uses_same_runtime_spine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            outcome = run_experiment(
                experiment_path=FORCING_EXPERIMENT,
                output_dir=output,
                repository_revision="b" * 40,
                run_scope="forcing-fixture-run",
            )
            self.assertEqual(
                outcome["experiment_id"], "physics.two_layer_ebm.forcing_protocols.v1"
            )
            self.assertEqual(outcome["evidence"], [])
            self.assertEqual(len(outcome["runs"]), 2)

            metric_values = {
                metric["metric"]["metric_id"]: metric["value"]
                for metric in outcome["metrics"]
            }
            self.assertLess(
                metric_values[
                    "physics.two_layer_ebm.affine_forcing.constant_equivalence_max_abs_state_error_k"
                ],
                5e-13,
            )
            self.assertLess(
                metric_values[
                    "physics.two_layer_ebm.affine_forcing.max_abs_energy_budget_residual_w_m2"
                ],
                1e-12,
            )
            self.assertGreater(
                metric_values[
                    "physics.two_layer_ebm.ood.held_out_absolute_forcing_margin_w_m2"
                ],
                0.0,
            )

            baseline, candidate = outcome["runs"]
            self.assertEqual(len(baseline["resolved_dataset_digests"]), 1)
            self.assertEqual(len(candidate["resolved_dataset_digests"]), 2)
            self.assertEqual(
                candidate["execution"]["resolved"]["method_id"],
                "physics.two_layer_ebm.affine_forcing_v1",
            )

            self.assert_artifact_digests(outcome, output)
            for filename in (
                "baseline-exact-modes.json",
                "candidate-forcing-protocols.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                self.assert_portable_json_tree(output / filename)


    def test_ebm_forced_ood_experiment_uses_same_runtime_spine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            outcome = run_experiment(
                experiment_path=FORCED_OOD_EXPERIMENT,
                output_dir=output,
                repository_revision="c" * 40,
                run_scope="forced-ood-fixture-run",
            )
            self.assertEqual(
                outcome["experiment_id"], "multirepresentation.ebm_forced_ood.v1"
            )
            self.assertEqual(outcome["evidence"], [])
            self.assertEqual(len(outcome["runs"]), 2)

            metric_values = {
                metric["metric"]["metric_id"]: metric["value"]
                for metric in outcome["metrics"]
            }
            self.assertLess(
                metric_values[
                    "multirepresentation.ebm.ood.full_state_confirmation_relative_error"
                ],
                1e-12,
            )
            self.assertGreater(
                metric_values[
                    "multirepresentation.ebm.ood.surface_scalar_confirmation_relative_error"
                ],
                0.02,
            )
            self.assertGreater(
                metric_values[
                    "multirepresentation.ebm.ood.closure_gap_confirmation_relative_error"
                ],
                0.02,
            )

            baseline, candidate = outcome["runs"]
            self.assertEqual(len(baseline["resolved_dataset_digests"]), 2)
            self.assertEqual(len(candidate["resolved_dataset_digests"]), 3)
            self.assertEqual(
                candidate["execution"]["resolved"]["method_id"],
                "dynamics.affine_control_lstsq.numpy_v1",
            )

            self.assert_artifact_digests(outcome, output)
            for filename in (
                "baseline-forcing-protocols.json",
                "candidate-forced-ood.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                self.assert_portable_json_tree(output / filename)

if __name__ == "__main__":
    unittest.main()
