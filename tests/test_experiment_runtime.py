from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.experiment_runtime import (
    DEFAULT_EXPERIMENT,
    ROOT,
    MethodProcessFailure,
    _experiment_adapters,
    _require_method_runtimes,
    _resolve_experiment_context,
    _run_process,
    run_experiment,
)


FORCING_EXPERIMENT = ROOT / "experiments" / "two-layer-ebm-forcing-protocols.v1.json"
FORCED_OOD_EXPERIMENT = ROOT / "experiments" / "multirepresentation-ebm-forced-ood.v1.json"
OBSERVATION_DEGRADATION_EXPERIMENT = (
    ROOT / "experiments" / "multirepresentation-ebm-observation-degradation.v1.json"
)
PARAMETER_IDENTIFIABILITY_EXPERIMENT = (
    ROOT / "experiments" / "two-layer-ebm-parameter-identifiability.v1.json"
)
STOCHASTIC_STATISTICS_EXPERIMENT = (
    ROOT / "experiments" / "multirepresentation-ebm-stochastic-statistics.v1.json"
)
REGIME_FEEDBACK_EXPERIMENT = (
    ROOT / "experiments" / "multirepresentation-ebm-regime-feedback.v1.json"
)


class ExperimentRuntimeTests(unittest.TestCase):
    def test_method_runtime_resolution_binds_build_and_backend_before_execution(self) -> None:
        context = _resolve_experiment_context(
            experiment_path=DEFAULT_EXPERIMENT,
            output_dir=Path("unused"),
            repository_revision="8" * 40,
            run_scope="method-runtime",
        )
        baseline, candidate = _require_method_runtimes(
            context.experiment,
            context.methods,
            baseline_method="physics.two_layer_ebm.exact_modes_v1",
            baseline_backend="scipy",
            candidate_method="dynamics.dmd.pydmd_v1",
            candidate_backend="pydmd",
        )
        self.assertTrue(baseline.implementation_build.startswith("source-sha256:"))
        self.assertEqual(baseline.execution_identity["backend_id"], "scipy")
        self.assertTrue(candidate.implementation_build.startswith("source-sha256:"))
        self.assertEqual(candidate.execution_identity["backend_id"], "pydmd")

    def test_adapter_registry_is_explicit_and_complete_for_current_runtime(self) -> None:
        self.assertEqual(
            set(_experiment_adapters()),
            {
                "multirepresentation.ebm_dynamics.v1",
                "physics.two_layer_ebm.forcing_protocols.v1",
                "multirepresentation.ebm_forced_ood.v1",
                "multirepresentation.ebm_observation_degradation.v1",
                "physics.two_layer_ebm.parameter_identifiability.v1",
                "multirepresentation.ebm_stochastic_statistics.v1",
                "multirepresentation.ebm_regime_feedback.v1",
            },
        )

    def test_resolution_context_binds_inputs_before_family_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = _resolve_experiment_context(
                experiment_path=DEFAULT_EXPERIMENT,
                output_dir=Path(tmp),
                repository_revision="9" * 40,
                run_scope="resolution-witness",
            )
            self.assertEqual(
                context.experiment["experiment_id"],
                "multirepresentation.ebm_dynamics.v1",
            )
            self.assertTrue(
                context.experiment["_runtime_spec_digest"].startswith("sha256:")
            )
            self.assertEqual(context.repository_revision, "9" * 40)
            self.assertEqual(context.run_scope, "resolution-witness")
            self.assertIn(
                "physics.two_layer_ebm.exact_modes_v1",
                context.methods,
            )

    def test_resolution_context_rejects_implicit_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "repository_revision"):
                _resolve_experiment_context(
                    experiment_path=DEFAULT_EXPERIMENT,
                    output_dir=Path(tmp),
                    repository_revision="",
                    run_scope="scope",
                )
            with self.assertRaisesRegex(ValueError, "run_scope"):
                _resolve_experiment_context(
                    experiment_path=DEFAULT_EXPERIMENT,
                    output_dir=Path(tmp),
                    repository_revision="9" * 40,
                    run_scope="",
                )

    def assert_failed_run(
        self,
        output: Path,
        *,
        failure_class: str,
        failure_stage: str,
        exit_code: int | None,
    ) -> dict:
        payload = json.loads((output / "run-baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["execution"]["status"], "ineligible")
        self.assertEqual(payload["execution"]["failure"], failure_class)
        self.assertFalse(payload["scientific_output_eligible"])
        self.assertEqual(payload["failure_stage"], failure_stage)
        self.assertTrue(payload["failure_detail"])
        self.assertEqual(len(payload["commands"]), 1)
        self.assertTrue(payload["stdout_digest"].startswith("sha256:"))
        self.assertTrue(payload["stderr_digest"].startswith("sha256:"))
        self.assertEqual(payload.get("exit_code"), exit_code)
        self.assertEqual(payload["artifacts"], [])
        return payload

    def test_nonzero_process_exit_persists_ineligible_run_receipt(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["fixture"],
            returncode=17,
            stdout=b"",
            stderr=b"dependency import failed\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch(
                "src.experiment_runtime.subprocess.run",
                return_value=completed,
            ):
                with self.assertRaises(MethodProcessFailure):
                    run_experiment(
                        experiment_path=DEFAULT_EXPERIMENT,
                        output_dir=output,
                        repository_revision="1" * 40,
                        run_scope="failed-exit",
                    )
            manifest = self.assert_failed_run(
                output,
                failure_class="process_failed",
                failure_stage="process_exit",
                exit_code=17,
            )
            self.assertEqual(
                manifest["execution"]["requested"]["method_id"],
                "physics.two_layer_ebm.exact_modes_v1",
            )
            self.assertEqual(len(manifest["resolved_dataset_digests"]), 1)

    def test_invalid_json_with_zero_exit_persists_truthful_failure_receipt(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["fixture"],
            returncode=0,
            stdout=b"not-json\n",
            stderr=b"",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch(
                "src.experiment_runtime.subprocess.run",
                return_value=completed,
            ):
                with self.assertRaises(MethodProcessFailure):
                    run_experiment(
                        experiment_path=DEFAULT_EXPERIMENT,
                        output_dir=output,
                        repository_revision="2" * 40,
                        run_scope="invalid-output",
                    )
            self.assert_failed_run(
                output,
                failure_class="output_invalid",
                failure_stage="output_decode",
                exit_code=0,
            )

    def test_process_launch_failure_has_no_invented_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch(
                "src.experiment_runtime.subprocess.run",
                side_effect=FileNotFoundError("missing executable"),
            ):
                with self.assertRaises(MethodProcessFailure):
                    run_experiment(
                        experiment_path=DEFAULT_EXPERIMENT,
                        output_dir=output,
                        repository_revision="3" * 40,
                        run_scope="missing-implementation",
                    )
            self.assert_failed_run(
                output,
                failure_class="implementation_unavailable",
                failure_stage="process_launch",
                exit_code=None,
            )

    def test_process_receipt_classifies_invalid_json_without_losing_digests(self) -> None:
        with self.assertRaises(MethodProcessFailure) as caught:
            _run_process(
                (
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.write('not-json')",
                ),
                method_id="fixture.method",
                backend_id="python",
            )
        receipt = caught.exception.receipt
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["failure_class"], "output_invalid")
        self.assertEqual(receipt["failure_stage"], "output_decode")
        self.assertTrue(receipt["stdout_digest"].startswith("sha256:"))
        self.assertTrue(receipt["stderr_digest"].startswith("sha256:"))

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

    def assert_experiment_spec_digest(self, run: dict, experiment_path: Path) -> None:
        expected = "sha256:" + hashlib.sha256(experiment_path.read_bytes()).hexdigest()
        self.assertEqual(run["experiment_spec_digest"], expected)

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
            self.assertTrue(
                all(
                    "multirepresentation.ebm_dynamics.v1" in run["run_id"]
                    for run in outcome["runs"]
                )
            )
            self.assertTrue(all(run["scientific_output_eligible"] for run in outcome["runs"]))
            self.assertTrue(all(run["execution"]["status"] == "eligible" for run in outcome["runs"]))
            for run in outcome["runs"]:
                self.assert_experiment_spec_digest(run, DEFAULT_EXPERIMENT)

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

    def test_experiment_spec_digest_changes_for_same_id_with_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant_payload = json.loads(DEFAULT_EXPERIMENT.read_text(encoding="utf-8"))
            variant_payload["question"] = variant_payload["question"] + " "
            variant = root / "variant.json"
            variant.write_text(json.dumps(variant_payload, indent=2) + "\n", encoding="utf-8")

            original = run_experiment(
                experiment_path=DEFAULT_EXPERIMENT,
                output_dir=root / "original",
                repository_revision="a" * 40,
                run_scope="digest-original",
            )
            changed = run_experiment(
                experiment_path=variant,
                output_dir=root / "changed",
                repository_revision="a" * 40,
                run_scope="digest-changed",
            )
            self.assertEqual(original["experiment_id"], changed["experiment_id"])
            self.assertNotEqual(
                original["runs"][0]["experiment_spec_digest"],
                changed["runs"][0]["experiment_spec_digest"],
            )

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
            self.assertTrue(
                all(
                    "physics.two_layer_ebm.forcing_protocols.v1" in run["run_id"]
                    for run in outcome["runs"]
                )
            )

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
            self.assertTrue(
                all(
                    "multirepresentation.ebm_forced_ood.v1" in run["run_id"]
                    for run in outcome["runs"]
                )
            )

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

    def test_observation_degradation_experiment_uses_same_runtime_spine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            outcome = run_experiment(
                experiment_path=OBSERVATION_DEGRADATION_EXPERIMENT,
                output_dir=output,
                repository_revision="d" * 40,
                run_scope="shared-ebm-scope",
            )
            self.assertEqual(
                outcome["experiment_id"],
                "multirepresentation.ebm_observation_degradation.v1",
            )
            self.assertEqual(outcome["evidence"], [])
            self.assertEqual(len(outcome["runs"]), 2)
            self.assertTrue(
                all(
                    "multirepresentation.ebm_observation_degradation.v1"
                    in run["run_id"]
                    for run in outcome["runs"]
                )
            )

            def by_population(metric_id: str) -> dict[str, float]:
                return {
                    item["reference_population"]: item["value"]
                    for item in outcome["metrics"]
                    if item["metric"]["metric_id"] == metric_id
                }

            reconstruction = by_population(
                "multirepresentation.ebm.observation.state_reconstruction_relative_error"
            )
            forced = by_population(
                "multirepresentation.ebm.observation.ood_forced_state_prediction_relative_error"
            )
            ranks = by_population(
                "multirepresentation.ebm.observation.structural_observation_rank"
            )
            dimensions = by_population(
                "multirepresentation.ebm.observation.observation_dimension"
            )

            self.assertLess(reconstruction["clean_temperature_state"], 1e-14)
            self.assertLess(forced["clean_temperature_state"], 1e-12)
            self.assertLess(
                reconstruction["noisy_temperature_state"],
                reconstruction["noisy_surface_scalar"],
            )
            self.assertLess(
                forced["noisy_temperature_state"],
                forced["noisy_surface_scalar"],
            )
            self.assertEqual(ranks["noisy_surface_scalar"], 1)
            self.assertEqual(ranks["redundant_noisy_surface_pair"], 1)
            self.assertEqual(dimensions["redundant_noisy_surface_pair"], 2)
            self.assertAlmostEqual(
                reconstruction["redundant_noisy_surface_pair"],
                reconstruction["noisy_surface_scalar"],
                places=12,
            )
            self.assertAlmostEqual(
                forced["redundant_noisy_surface_pair"],
                forced["noisy_surface_scalar"],
                places=12,
            )

            baseline, candidate = outcome["runs"]
            self.assertEqual(len(baseline["resolved_dataset_digests"]), 2)
            self.assertEqual(len(candidate["resolved_dataset_digests"]), 4)
            self.assertEqual(
                candidate["execution"]["resolved"]["method_id"],
                "dynamics.affine_control_lstsq.numpy_v1",
            )

            self.assert_artifact_digests(outcome, output)
            for filename in (
                "baseline-forcing-protocols.json",
                "candidate-observation-degradation.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                self.assert_portable_json_tree(output / filename)

    def test_parameter_identifiability_experiment_uses_same_runtime_spine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            outcome = run_experiment(
                experiment_path=PARAMETER_IDENTIFIABILITY_EXPERIMENT,
                output_dir=output,
                repository_revision="f" * 40,
                run_scope="parameter-identifiability-fixture-run",
            )
            self.assertEqual(
                outcome["experiment_id"],
                "physics.two_layer_ebm.parameter_identifiability.v1",
            )
            self.assertEqual(outcome["evidence"], [])
            self.assertEqual(len(outcome["runs"]), 2)
            self.assertTrue(
                all(
                    "physics.two_layer_ebm.parameter_identifiability.v1"
                    in run["run_id"]
                    for run in outcome["runs"]
                )
            )

            metric_values = {
                item["metric"]["metric_id"]: item["value"]
                for item in outcome["metrics"]
            }
            self.assertEqual(
                metric_values[
                    "physics.two_layer_ebm.parameter.equilibrium_local_sensitivity_rank"
                ],
                1,
            )
            self.assertEqual(
                metric_values[
                    "physics.two_layer_ebm.parameter.transient_local_sensitivity_rank"
                ],
                4,
            )
            self.assertEqual(
                metric_values["physics.two_layer_ebm.parameter.local_rank_gain"],
                3,
            )
            self.assertEqual(
                metric_values[
                    "physics.two_layer_ebm.parameter.equilibrium_invisible_parameter_count"
                ],
                3,
            )
            self.assertLess(
                metric_values[
                    "physics.two_layer_ebm.parameter.equilibrium_step_consistency_relative_frobenius"
                ],
                1e-7,
            )
            self.assertLess(
                metric_values[
                    "physics.two_layer_ebm.parameter.transient_step_consistency_relative_frobenius"
                ],
                1e-7,
            )
            self.assertTrue(
                math.isfinite(
                    metric_values[
                        "physics.two_layer_ebm.parameter.transient_condition_number"
                    ]
                )
            )

            baseline, candidate = outcome["runs"]
            self.assertEqual(len(baseline["resolved_dataset_digests"]), 2)
            self.assertEqual(len(candidate["resolved_dataset_digests"]), 3)
            self.assertEqual(
                candidate["execution"]["resolved"]["method_id"],
                "physics.two_layer_ebm.log_parameter_sensitivity_v1",
            )

            self.assert_artifact_digests(outcome, output)
            for filename in (
                "baseline-forcing-protocols.json",
                "candidate-parameter-identifiability.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                self.assert_portable_json_tree(output / filename)

    def test_stochastic_statistics_experiment_uses_same_runtime_spine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            outcome = run_experiment(
                experiment_path=STOCHASTIC_STATISTICS_EXPERIMENT,
                output_dir=output,
                repository_revision="g" * 40,
                run_scope="stochastic-statistics-fixture-run",
            )
            self.assertEqual(
                outcome["experiment_id"],
                "multirepresentation.ebm_stochastic_statistics.v1",
            )
            self.assertEqual(outcome["evidence"], [])
            self.assertEqual(len(outcome["runs"]), 2)
            self.assertTrue(
                all(
                    "multirepresentation.ebm_stochastic_statistics.v1"
                    in run["run_id"]
                    for run in outcome["runs"]
                )
            )

            def by_population(metric_id: str) -> dict[str, float]:
                return {
                    item["reference_population"]: item["value"]
                    for item in outcome["metrics"]
                    if item["metric"]["metric_id"] == metric_id
                }

            covariance = by_population(
                "multirepresentation.ebm.stochastic.state_covariance_relative_frobenius_error"
            )
            lag = by_population(
                "multirepresentation.ebm.stochastic.lag_covariance_relative_frobenius_error"
            )
            deep_variance = by_population(
                "multirepresentation.ebm.stochastic.deep_variance_relative_error"
            )
            ranks = by_population(
                "multirepresentation.ebm.stochastic.structural_observation_rank"
            )
            self.assertLess(covariance["temperature_state"], 1e-12)
            self.assertLess(lag["temperature_state"], 1e-12)
            self.assertGreater(covariance["surface_temperature_scalar"], 0.1)
            self.assertGreater(lag["surface_temperature_scalar"], 0.1)
            self.assertAlmostEqual(
                deep_variance["surface_temperature_scalar"], 1.0, places=12
            )
            self.assertEqual(ranks["temperature_state"], 2)
            self.assertEqual(ranks["surface_temperature_scalar"], 1)
            self.assertEqual(ranks["redundant_surface_pair"], 1)
            self.assertAlmostEqual(
                covariance["redundant_surface_pair"],
                covariance["surface_temperature_scalar"],
                places=12,
            )
            self.assertAlmostEqual(
                lag["redundant_surface_pair"],
                lag["surface_temperature_scalar"],
                places=12,
            )

            metric_values = {
                item["metric"]["metric_id"]: item["value"]
                for item in outcome["metrics"]
                if "reference_population" not in item
            }
            self.assertLess(
                metric_values[
                    "physics.two_layer_ebm.stochastic.max_abs_total_budget_residual_w_m2"
                ],
                1e-12,
            )
            self.assertLess(
                metric_values[
                    "physics.two_layer_ebm.stochastic.max_abs_direct_internal_storage_w_m2"
                ],
                1e-14,
            )

            baseline, candidate = outcome["runs"]
            self.assertEqual(len(baseline["resolved_dataset_digests"]), 2)
            self.assertEqual(len(candidate["resolved_dataset_digests"]), 2)
            self.assertEqual(
                baseline["execution"]["resolved"]["method_id"],
                "physics.two_layer_ebm.stochastic_internal_exchange_v1",
            )
            self.assertEqual(
                candidate["execution"]["resolved"]["method_id"],
                "multirepresentation.fixed_decode_statistics.numpy_v1",
            )

            self.assert_artifact_digests(outcome, output)
            for filename in (
                "baseline-stochastic-variability.json",
                "candidate-stochastic-statistics.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                self.assert_portable_json_tree(output / filename)

    def test_regime_feedback_experiment_uses_same_runtime_spine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            outcome = run_experiment(
                experiment_path=REGIME_FEEDBACK_EXPERIMENT,
                output_dir=output,
                repository_revision="h" * 40,
                run_scope="regime-feedback-fixture-run",
            )
            self.assertEqual(
                outcome["experiment_id"],
                "multirepresentation.ebm_regime_feedback.v1",
            )
            self.assertEqual(outcome["evidence"], [])
            self.assertEqual(len(outcome["runs"]), 2)
            self.assertTrue(
                all(
                    "multirepresentation.ebm_regime_feedback.v1"
                    in run["run_id"]
                    for run in outcome["runs"]
                )
            )

            metric_values = {
                item["metric"]["metric_id"]: item["value"]
                for item in outcome["metrics"]
            }
            self.assertGreater(
                metric_values[
                    "multirepresentation.ebm.regime.raw_global_confirmation_relative_error"
                ],
                0.004,
            )
            self.assertLess(
                metric_values[
                    "multirepresentation.ebm.regime.gated_confirmation_relative_error"
                ],
                1e-12,
            )
            self.assertGreater(
                metric_values[
                    "multirepresentation.ebm.regime.confirmation_error_reduction"
                ],
                0.004,
            )
            self.assertGreater(
                metric_values[
                    "multirepresentation.ebm.regime.raw_global_training_relative_error"
                ],
                0.003,
            )
            self.assertLess(
                metric_values[
                    "multirepresentation.ebm.regime.gated_training_relative_error"
                ],
                1e-12,
            )
            self.assertEqual(
                metric_values["multirepresentation.ebm.regime.raw_design_rank"],
                3,
            )
            self.assertEqual(
                metric_values["multirepresentation.ebm.regime.gated_design_rank"],
                6,
            )
            self.assertGreater(
                metric_values[
                    "multirepresentation.ebm.regime.swapped_label_confirmation_relative_error"
                ],
                0.02,
            )
            self.assertGreaterEqual(
                metric_values[
                    "physics.two_layer_ebm.regime.observed_minimum_regime_margin_k"
                ],
                0.5,
            )

            baseline, candidate = outcome["runs"]
            self.assertEqual(len(baseline["resolved_dataset_digests"]), 2)
            self.assertEqual(len(candidate["resolved_dataset_digests"]), 2)
            self.assertEqual(
                baseline["execution"]["resolved"]["method_id"],
                "physics.two_layer_ebm.regime_feedback_local_v1",
            )
            self.assertEqual(
                candidate["execution"]["resolved"]["method_id"],
                "dynamics.regime_gated_lstsq.numpy_v1",
            )

            self.assert_artifact_digests(outcome, output)
            for filename in (
                "baseline-regime-feedback.json",
                "candidate-regime-representation.json",
                "metric-results.json",
                "run-baseline.json",
                "run-candidate.json",
                "outcome.json",
            ):
                self.assert_portable_json_tree(output / filename)

    def test_shared_methods_do_not_collide_across_experiment_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forced = run_experiment(
                experiment_path=FORCED_OOD_EXPERIMENT,
                output_dir=root / "forced",
                repository_revision="e" * 40,
                run_scope="same-scope",
            )
            degraded = run_experiment(
                experiment_path=OBSERVATION_DEGRADATION_EXPERIMENT,
                output_dir=root / "degraded",
                repository_revision="e" * 40,
                run_scope="same-scope",
            )
            forced_ids = {run["run_id"] for run in forced["runs"]}
            degraded_ids = {run["run_id"] for run in degraded["runs"]}
            self.assertTrue(forced_ids.isdisjoint(degraded_ids))


if __name__ == "__main__":
    unittest.main()
