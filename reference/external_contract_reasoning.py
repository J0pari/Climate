"""Reference scorer for Climate's external contract-reasoning evaluation.

This module scores already-produced structured choices. It does not load a
Training artifact, invoke a model runtime, issue a Commons attestation, or make
a Training promotion decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = (
    ROOT / "evaluations" /
    "external-training-artifact-climate-contract-reasoning.v1.json"
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_CHOICES = ("A", "B", "C")
_ROTATED = {"A": "B", "B": "C", "C": "A"}


class ContractReasoningError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_digest(predictions: dict[str, Any]) -> str:
    canonical = json.dumps(
        predictions, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_task_set(spec: dict[str, Any], root: Path) -> tuple[Path, dict[str, Any]]:
    ref = spec.get("task_set")
    if not isinstance(ref, dict):
        raise ContractReasoningError("evaluation spec has no task_set artifact")
    uri = ref.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ContractReasoningError("evaluation task_set.uri is missing")
    path = (root / uri).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractReasoningError("evaluation task set escapes repository root") from error
    if not path.is_file():
        raise ContractReasoningError(f"evaluation task set is missing: {uri}")
    actual = _sha256(path)
    if ref.get("digest") != actual:
        raise ContractReasoningError(
            f"evaluation task-set digest drift: {ref.get('digest')!r} != {actual!r}"
        )
    return path, _load(path)


def score_predictions(
    predictions: dict[str, Any],
    *,
    spec: dict[str, Any],
    task_set: dict[str, Any],
    task_set_digest: str,
) -> dict[str, Any]:
    if predictions.get("schema") != spec.get("prediction_contract"):
        raise ContractReasoningError("prediction schema does not match evaluator contract")
    if predictions.get("evaluation_id") != spec.get("evaluation_id"):
        raise ContractReasoningError("prediction evaluation_id does not match evaluator")
    subject_digest = predictions.get("subject_digest")
    if not isinstance(subject_digest, str) or not _SHA256_HEX.fullmatch(subject_digest):
        raise ContractReasoningError("prediction subject_digest must be full lowercase SHA-256")
    runtime = predictions.get("runtime")
    if not isinstance(runtime, dict):
        raise ContractReasoningError("prediction runtime identity is missing")
    if runtime.get("interface") != spec.get("adapter", {}).get("interface"):
        raise ContractReasoningError("prediction runtime interface does not match evaluator")
    if runtime.get("subject_digest") != subject_digest:
        raise ContractReasoningError("runtime subject digest does not match prediction subject")
    implementation = runtime.get("implementation")
    if not isinstance(implementation, str) or not implementation.strip():
        raise ContractReasoningError("prediction runtime implementation is missing")
    implementation_version = runtime.get("implementation_version")
    if not isinstance(implementation_version, str) or not implementation_version.strip():
        raise ContractReasoningError("prediction runtime implementation_version is missing")
    configuration_digest = runtime.get("configuration_digest")
    if not isinstance(configuration_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", configuration_digest):
        raise ContractReasoningError("prediction runtime configuration_digest is invalid")

    tasks = task_set.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ContractReasoningError("task set contains no tasks")
    task_map = {}
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("task_id"), str):
            raise ContractReasoningError("task set contains an invalid task")
        task_id = task["task_id"]
        if task_id in task_map:
            raise ContractReasoningError(f"duplicate task_id in task set: {task_id}")
        if task.get("correct_choice") not in _CHOICES:
            raise ContractReasoningError(f"task {task_id} has invalid correct_choice")
        task_map[task_id] = task

    responses = predictions.get("responses")
    if not isinstance(responses, list):
        raise ContractReasoningError("prediction responses must be a list")
    response_map: dict[str, str] = {}
    for response in responses:
        if not isinstance(response, dict):
            raise ContractReasoningError("prediction response must be an object")
        task_id = response.get("task_id")
        choice = response.get("choice")
        if task_id not in task_map:
            raise ContractReasoningError(f"prediction names unknown task: {task_id!r}")
        if task_id in response_map:
            raise ContractReasoningError(f"duplicate prediction for task: {task_id}")
        if choice not in _CHOICES:
            raise ContractReasoningError(f"prediction for {task_id} has invalid choice")
        response_map[task_id] = choice

    missing = sorted(set(task_map) - set(response_map))
    if missing:
        raise ContractReasoningError(
            "prediction set is incomplete: " + ", ".join(missing)
        )

    total = len(task_map)
    correct = sum(
        response_map[task_id] == task["correct_choice"]
        for task_id, task in task_map.items()
    )
    control_correct = sum(
        response_map[task_id] == _ROTATED[task["correct_choice"]]
        for task_id, task in task_map.items()
    )
    accuracy = correct / total
    control_accuracy = control_correct / total

    return {
        "schema": spec["result_contract"],
        "evaluation_id": spec["evaluation_id"],
        "subject_digest": subject_digest,
        "task_set_digest": task_set_digest,
        "prediction_digest": _prediction_digest(predictions),
        "metrics": {
            "contract_choice_accuracy": accuracy,
            "unsafe_semantic_upgrade_rate": 1.0 - accuracy,
        },
        "negative_controls": {
            "permuted_answer_key": {
                "accuracy": control_accuracy,
                "delta_vs_observed": accuracy - control_accuracy,
            }
        },
        "evidence": dict(spec["evidence_policy"]),
    }


def score_file(
    predictions_path: Path,
    *,
    spec_path: Path = DEFAULT_SPEC,
    root: Path = ROOT,
) -> dict[str, Any]:
    spec = _load(spec_path)
    task_path, task_set = _resolve_task_set(spec, root)
    return score_predictions(
        _load(predictions_path),
        spec=spec,
        task_set=task_set,
        task_set_digest=_sha256(task_path),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="score a digest-bound external model prediction artifact"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = score_file(args.predictions)
    except (OSError, json.JSONDecodeError, ContractReasoningError) as error:
        print(str(error))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
