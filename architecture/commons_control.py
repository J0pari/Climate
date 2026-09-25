"""Fail-closed Climate client for the Commons control API (control-api/v1).

Climate owns the experiment and scientific semantics. Commons receives only a
declared job plus machine-resource requirements over its loopback control API.
The adapter permits CPU experiment execution into run-artifacts; it does not
grant Commons repository source writes or scientific evidence-promotion
authority. The Commons client module (`control/client.py`) is imported from the
resolved checkout and discovers the daemon's published address, so no port is
hardcoded here and scheduler-state files are never parsed. Pinned Commons
contracts are verified from the checkout's committed HEAD, so an unrelated
session's in-flight worktree edits cannot silently move Climate's interface.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "contracts" / "work-scheduler-pin.json"
CONTROL_API_PIN = ROOT / "contracts" / "control-api-pin.json"
EVALUATION_PIN = ROOT / "contracts" / "evaluation-exchange-pin.json"
EVALUATIONS_DIR = ROOT / "evaluations"
ABI_KEYS = (
    "schema", "contractVersion", "compatibility", "public", "types",
    "endpoints", "resource_semantics",
)
CONTROL_API_ABI_KEYS = (
    "schema", "contractVersion", "compatibility", "public", "semantics",
    "types", "endpoints",
)
_RUN_SCOPE = re.compile(r"^[A-Za-z0-9._-]+$")


class CommonsControlError(RuntimeError):
    pass


class ExternalEvaluationUnavailable(CommonsControlError):
    pass


def load_pin() -> dict[str, Any]:
    return json.loads(PIN.read_text(encoding="utf-8"))


def load_control_api_pin() -> dict[str, Any]:
    return json.loads(CONTROL_API_PIN.read_text(encoding="utf-8"))


def _fingerprint(contract: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    abi = {key: contract.get(key) for key in keys if key in contract}
    canonical = json.dumps(abi, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def abi_fingerprint(contract: Mapping[str, Any]) -> str:
    return _fingerprint(contract, ABI_KEYS)


def control_api_fingerprint(contract: Mapping[str, Any]) -> str:
    return _fingerprint(contract, CONTROL_API_ABI_KEYS)


def commons_root(env: Mapping[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    configured = values.get("COMMONS_ROOT")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_dir():
            return path
        raise CommonsControlError(
            f"COMMONS_ROOT does not name a directory: {path}")
    for sibling in (ROOT.parent / "Commons", ROOT.parent / "commons"):
        if sibling.is_dir():
            return sibling.resolve()
    raise CommonsControlError(
        "Commons checkout unavailable; set COMMONS_ROOT explicitly")


def control_client(env: Mapping[str, str] | None = None):
    """Import the pinned Commons client; it resolves the published address."""
    root = commons_root(env)
    client_path = root / "control" / "client.py"
    if not client_path.is_file():
        raise CommonsControlError(
            f"Commons control client is unavailable: {client_path}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        return importlib.import_module("control.client")
    except Exception as exc:
        raise CommonsControlError(
            f"Commons control client failed to import: {exc}") from exc


def _committed_contract(root: Path, relative: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CommonsControlError(
            f"Commons committed contract is unavailable: {relative}") from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CommonsControlError(
            f"Commons committed contract is not readable JSON: {relative}") from exc


def _require_pinned_contract(
    contract: Mapping[str, Any],
    pin: Mapping[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    if contract.get("schema") != pin["schema"]:
        raise CommonsControlError(
            f"Commons contract schema {contract.get('schema')!r} "
            f"!= pinned {pin['schema']!r}")
    if contract.get("owner") != pin["owner"]:
        raise CommonsControlError(
            f"Commons contract owner {contract.get('owner')!r} "
            f"!= pinned {pin['owner']!r}")
    actual = _fingerprint(contract, keys)
    if actual != pin["fingerprint"]:
        raise CommonsControlError(
            f"Commons contract fingerprint drift for {pin['schema']}: "
            f"{actual} != {pin['fingerprint']}")
    return contract


def verify_contracts(
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = commons_root(env)
    return {
        "control_api": _require_pinned_contract(
            _committed_contract(root, "contracts/control-api-v1.json"),
            load_control_api_pin(),
            CONTROL_API_ABI_KEYS,
        ),
        "work_scheduler": _require_pinned_contract(
            _committed_contract(root, "contracts/work-scheduler-v1.json"),
            load_pin(),
            ABI_KEYS,
        ),
    }


def _api(operation: str, call):
    try:
        return call()
    except CommonsControlError:
        raise
    except Exception as exc:
        raise CommonsControlError(
            f"Commons control API {operation} failed: {exc}") from exc


def scheduler_status(
    env: Mapping[str, str] | None = None,
    etag: str | None = None,
) -> dict[str, Any]:
    verify_contracts(env)
    client = control_client(env)
    return _api("status", lambda: client.status(etag=etag))


def list_jobs(
    repo: str | None = None,
    status: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_contracts(env)
    client = control_client(env)
    return _api("jobs", lambda: client.list_jobs(repo=repo, status=status))


def inspect_job(
    job_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_contracts(env)
    client = control_client(env)
    return _api(f"inspect {job_id}", lambda: client.inspect(job_id=job_id))


def cancel_job(
    job_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_contracts(env)
    client = control_client(env)
    return _api(f"cancel {job_id}", lambda: client.cancel(job_id=job_id))


def inbox(
    repo: str,
    *,
    wait_seconds: float | None = None,
    limit: int | None = None,
    after: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_contracts(env)
    client = control_client(env)
    return _api("inbox", lambda: client.inbox(
        repo=repo, wait_seconds=wait_seconds, limit=limit, after=after))


def ack_message(
    message_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_contracts(env)
    client = control_client(env)
    return _api(f"ack {message_id}", lambda: client.ack(message_id=message_id))


def send_message(
    *,
    repo: str,
    payload: Any,
    name: str | None = None,
    idempotency_key: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_contracts(env)
    client = control_client(env)
    return _api("send", lambda: client.send(
        repo=repo, payload=payload, name=name,
        idempotency_key=idempotency_key))


def submit_cpu_experiment(
    *,
    experiment_path: Path,
    repository_revision: str,
    run_scope: str,
    ram_mib: int,
    max_minutes: float = 30.0,
    priority: int = 0,
    idempotency_key: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Enqueue one existing Climate CPU experiment without changing its semantics."""
    if not repository_revision.strip():
        raise CommonsControlError("repository_revision must be explicit")
    if not _RUN_SCOPE.fullmatch(run_scope):
        raise CommonsControlError(
            "run_scope must contain only letters, digits, dot, underscore, or hyphen")
    if ram_mib <= 0:
        raise CommonsControlError("ram_mib must be positive")

    experiment = experiment_path.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    try:
        experiment.relative_to(experiments_root)
    except ValueError as exc:
        raise CommonsControlError(
            "experiment_path must resolve under Climate experiments/") from exc
    if not experiment.is_file():
        raise CommonsControlError(f"experiment does not exist: {experiment}")

    output_dir = ROOT / "run-artifacts" / "commons" / run_scope
    command = [
        sys.executable,
        str(ROOT / "src" / "experiment_runtime.py"),
        "--experiment", str(experiment),
        "--output-dir", str(output_dir),
        "--repository-revision", repository_revision,
        "--run-scope", run_scope,
    ]
    verify_contracts(env)
    client = control_client(env)
    key = idempotency_key or f"climate-{run_scope}"
    ack = _api("submit", lambda: client.submit(
        name=f"climate-{run_scope}",
        repo="climate",
        command=command,
        cwd=str(ROOT),
        resourceClass="cpu",
        priority=priority,
        maxMinutes=max_minutes,
        ramMib=ram_mib,
        idempotency_key=key,
    ))
    return {
        **ack,
        "experiment": experiment.relative_to(ROOT).as_posix(),
        "outputDir": output_dir.relative_to(ROOT).as_posix(),
        "repositoryRevision": repository_revision,
        "runScope": run_scope,
        "idempotencyKey": key,
    }


_EVAL_ABI_KEYS = (
    "schema", "contractVersion", "compatibility", "types", "invariants",
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def load_evaluation_pin() -> dict[str, Any]:
    return json.loads(EVALUATION_PIN.read_text(encoding="utf-8"))


def evaluation_exchange_fingerprint(contract: Mapping[str, Any]) -> str:
    abi = {key: contract.get(key) for key in _EVAL_ABI_KEYS if key in contract}
    canonical = json.dumps(abi, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_external_artifact_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exchange identity without pretending the artifact is evaluable."""
    required = (
        "producer_repository", "local_artifact_id", "digest",
        "artifact_contract",
    )
    missing = [key for key in required if not ref.get(key)]
    if missing:
        raise CommonsControlError(
            "external artifact ref missing: " + ", ".join(missing))
    if not _REPOSITORY.fullmatch(str(ref["producer_repository"])):
        raise CommonsControlError("producer_repository must be owner/name")
    if not _SHA256_HEX.fullmatch(str(ref["digest"])):
        raise CommonsControlError(
            "external artifact digest must be full lowercase SHA-256")
    return dict(ref)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_ref(path: Path, *, artifact_id: str, schema: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "digest": _sha256_file(path),
        "media_type": "application/json",
        "schema": schema,
        "bytes": path.stat().st_size,
        "uri": str(path),
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommonsControlError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CommonsControlError(f"{label} must contain a JSON object")
    return value


def _validate_transformations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CommonsControlError("native runtime transformations must be a list")
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise CommonsControlError("native runtime transformation must be an object")
        for field in ("transform_id", "implementation", "implementation_version"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise CommonsControlError(f"native runtime transformation {field} is missing")
        digest = item.get("configuration_digest")
        if digest is not None and not _SHA256_REF.fullmatch(str(digest)):
            raise CommonsControlError("native runtime transformation configuration_digest must be sha256:<64 hex>")
        normalized.append(dict(item))
    return normalized


def validate_external_output_import(
    *,
    subject: Mapping[str, Any],
    evaluation_id: str,
    predictions_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Validate an externally executed model output without impersonating its runtime."""
    validated = validate_external_artifact_ref(subject)
    spec = load_external_evaluation_spec(evaluation_id)
    if validated["artifact_contract"] != spec.get("subject_contract"):
        raise CommonsControlError(
            f"evaluator {evaluation_id!r} expects subject contract "
            f"{spec.get('subject_contract')!r}, got {validated['artifact_contract']!r}")

    adapter = spec.get("adapter") or {}
    if adapter.get("status") not in {"native_output_import", "available"}:
        raise ExternalEvaluationUnavailable(
            f"Climate evaluator {evaluation_id!r} has no available native import/runtime path")
    receipt_contract = adapter.get("receipt_contract")
    if receipt_contract != "climate.external-model-runtime-receipt/v1":
        raise CommonsControlError("external evaluator does not declare the native runtime receipt contract")

    predictions_path = predictions_path.resolve()
    receipt_path = receipt_path.resolve()
    predictions = _load_json_object(predictions_path, label="prediction artifact")
    receipt = _load_json_object(receipt_path, label="native runtime receipt")

    if receipt.get("schema") != receipt_contract:
        raise CommonsControlError("native runtime receipt schema does not match evaluator")
    if receipt.get("interface") != adapter.get("interface"):
        raise CommonsControlError("native runtime receipt interface does not match evaluator")
    if receipt.get("execution_mode") != "native_output_import":
        raise CommonsControlError("import path requires execution_mode=native_output_import")
    if receipt.get("status") != "succeeded":
        failure_class = receipt.get("failure_class", "unknown")
        failure_detail = receipt.get("failure_detail", "no failure detail")
        raise CommonsControlError(
            f"native runtime receipt records failed execution: {failure_class}: {failure_detail}")
    if receipt.get("exit_code") != 0:
        raise CommonsControlError("successful native runtime receipt must have exit_code=0")
    for field in ("producer_system", "implementation", "implementation_version"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise CommonsControlError(f"native runtime receipt {field} is missing")
    configuration_digest = receipt.get("configuration_digest")
    if not isinstance(configuration_digest, str) or not _SHA256_REF.fullmatch(configuration_digest):
        raise CommonsControlError("native runtime configuration_digest must be sha256:<64 hex>")
    _validate_transformations(receipt.get("transformations"))

    receipt_subject = receipt.get("subject")
    if not isinstance(receipt_subject, dict):
        raise CommonsControlError("native runtime receipt subject is missing")
    for field in ("producer_repository", "artifact_contract", "digest"):
        if receipt_subject.get(field) != validated.get(field):
            raise CommonsControlError(f"native runtime receipt subject {field} mismatch")

    actual_prediction_digest = _sha256_file(predictions_path)
    if receipt.get("prediction_digest") != actual_prediction_digest:
        raise CommonsControlError("native runtime receipt prediction_digest mismatch")
    if predictions.get("schema") != spec.get("prediction_contract"):
        raise CommonsControlError("prediction artifact schema does not match evaluator")
    if predictions.get("evaluation_id") != evaluation_id:
        raise CommonsControlError("prediction artifact evaluation_id does not match evaluator")
    if predictions.get("subject_digest") != validated["digest"]:
        raise CommonsControlError("prediction artifact subject digest mismatch")

    runtime = predictions.get("runtime")
    if not isinstance(runtime, dict):
        raise CommonsControlError("prediction artifact runtime identity is missing")
    expected_runtime = {
        "interface": adapter.get("interface"),
        "implementation": receipt.get("implementation"),
        "implementation_version": receipt.get("implementation_version"),
        "configuration_digest": configuration_digest,
        "subject_digest": validated["digest"],
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise CommonsControlError(f"prediction runtime {field} mismatch")

    prediction_artifact = _artifact_ref(
        predictions_path,
        artifact_id=f"{evaluation_id}.predictions",
        schema=spec["prediction_contract"],
    )
    receipt_artifact = _artifact_ref(
        receipt_path,
        artifact_id=f"{evaluation_id}.runtime_receipt",
        schema=receipt_contract,
    )
    return {
        "evaluation_id": evaluation_id,
        "subject": validated,
        "runtime": expected_runtime,
        "prediction_digest": actual_prediction_digest,
        "receipt_digest": receipt_artifact["digest"],
        "artifacts": [prediction_artifact, receipt_artifact],
        "transformations": list(receipt["transformations"]),
    }


def load_external_evaluation_spec(evaluation_id: str) -> dict[str, Any]:
    matches = []
    if EVALUATIONS_DIR.is_dir():
        for path in sorted(EVALUATIONS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("evaluation_id") == evaluation_id:
                matches.append((path, data))
    if not matches:
        raise ExternalEvaluationUnavailable(
            f"Climate has no registered external-artifact evaluator "
            f"{evaluation_id!r}; no attestation was issued")
    if len(matches) != 1:
        raise CommonsControlError(
            f"external evaluator {evaluation_id!r} is declared more than once")
    return matches[0][1]


def require_external_evaluator(
    subject: Mapping[str, Any],
    evaluation_id: str,
) -> dict[str, Any]:
    """Resolve a registered evaluator only when a native execution/import path exists."""
    validated = validate_external_artifact_ref(subject)
    spec = load_external_evaluation_spec(evaluation_id)
    if validated["artifact_contract"] != spec.get("subject_contract"):
        raise CommonsControlError(
            f"evaluator {evaluation_id!r} expects subject contract "
            f"{spec.get('subject_contract')!r}, got "
            f"{validated['artifact_contract']!r}")
    adapter = spec.get("adapter") or {}
    if adapter.get("status") not in {"available", "native_output_import"}:
        raise ExternalEvaluationUnavailable(
            f"Climate evaluator {evaluation_id!r} is registered but "
            f"adapter {adapter.get('interface')!r} is unavailable; "
            "no attestation was issued")
    return spec



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Climate client for the Commons control API (control-api/v1)")
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--etag", default=None)

    jobs_parser = sub.add_parser("jobs")
    jobs_parser.add_argument("--repo", default=None)
    jobs_parser.add_argument("--status", default=None)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--job", required=True)

    cancel_parser = sub.add_parser("cancel")
    cancel_parser.add_argument("--job", required=True)

    inbox_parser = sub.add_parser("inbox")
    inbox_parser.add_argument("--repo", required=True)
    inbox_parser.add_argument("--wait", type=float, default=None)
    inbox_parser.add_argument("--limit", type=int, default=None)
    inbox_parser.add_argument("--after", default=None)

    ack_parser = sub.add_parser("ack")
    ack_parser.add_argument("--message", required=True)

    send_parser = sub.add_parser("send")
    send_parser.add_argument("--repo", required=True)
    send_parser.add_argument("--payload", required=True)
    send_parser.add_argument("--name", default=None)
    send_parser.add_argument("--key", default=None)

    submit_parser = sub.add_parser("submit-cpu")
    submit_parser.add_argument("--experiment", type=Path, required=True)
    submit_parser.add_argument("--repository-revision", required=True)
    submit_parser.add_argument("--run-scope", required=True)
    submit_parser.add_argument("--ram", type=int, required=True, dest="ram_mib")
    submit_parser.add_argument("--max-minutes", type=float, default=30.0)
    submit_parser.add_argument("--priority", type=int, default=0)
    submit_parser.add_argument("--idempotency-key", default=None)

    import_parser = sub.add_parser("validate-external-import")
    import_parser.add_argument("--evaluation", required=True)
    import_parser.add_argument("--subject-ref", type=Path, required=True)
    import_parser.add_argument("--predictions", type=Path, required=True)
    import_parser.add_argument("--receipt", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = scheduler_status(etag=args.etag)
        elif args.command == "jobs":
            result = list_jobs(repo=args.repo, status=args.status)
        elif args.command == "inspect":
            result = inspect_job(args.job)
        elif args.command == "cancel":
            result = cancel_job(args.job)
        elif args.command == "inbox":
            result = inbox(
                args.repo,
                wait_seconds=args.wait,
                limit=args.limit,
                after=args.after,
            )
        elif args.command == "ack":
            result = ack_message(args.message)
        elif args.command == "send":
            result = send_message(
                repo=args.repo,
                payload=json.loads(args.payload),
                name=args.name,
                idempotency_key=args.key,
            )
        elif args.command == "submit-cpu":
            result = submit_cpu_experiment(
                experiment_path=args.experiment,
                repository_revision=args.repository_revision,
                run_scope=args.run_scope,
                ram_mib=args.ram_mib,
                max_minutes=args.max_minutes,
                priority=args.priority,
                idempotency_key=args.idempotency_key,
            )
        else:
            subject = _load_json_object(args.subject_ref, label="external subject reference")
            result = validate_external_output_import(
                subject=subject,
                evaluation_id=args.evaluation,
                predictions_path=args.predictions,
                receipt_path=args.receipt,
            )
    except (CommonsControlError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
