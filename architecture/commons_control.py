"""Fail-closed Climate client for Commons work-scheduler/v1.

Climate owns the experiment and scientific semantics. Commons receives only a
declared command plus machine-resource requirements. The adapter permits CPU
experiment execution into run-artifacts; it does not grant Commons repository
source writes or scientific evidence-promotion authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "contracts" / "work-scheduler-pin.json"
EVALUATION_PIN = ROOT / "contracts" / "evaluation-exchange-pin.json"
EVALUATIONS_DIR = ROOT / "evaluations"
ABI_KEYS = (
    "schema", "contractVersion", "compatibility", "public", "types",
    "endpoints", "resource_semantics",
)
_RUN_SCOPE = re.compile(r"^[A-Za-z0-9._-]+$")


class CommonsControlError(RuntimeError):
    pass


class ExternalEvaluationUnavailable(CommonsControlError):
    pass


def load_pin() -> dict[str, Any]:
    return json.loads(PIN.read_text(encoding="utf-8"))


def abi_fingerprint(contract: Mapping[str, Any]) -> str:
    abi = {key: contract.get(key) for key in ABI_KEYS if key in contract}
    canonical = json.dumps(abi, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def scheduler_path(env: Mapping[str, str] | None = None) -> Path:
    path = commons_root(env) / "control" / "work_scheduler.py"
    if not path.is_file():
        raise CommonsControlError(
            f"Commons work scheduler entrypoint is unavailable: {path}")
    return path


def _invoke(args: list[str], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(scheduler_path(env)), *args],
        cwd=str(commons_root(env)),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise CommonsControlError(
            f"Commons scheduler refused with exit {process.returncode}: {detail}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise CommonsControlError(
            "Commons scheduler returned non-JSON output") from exc


def verify_scheduler_contract(
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    contract = _invoke(["contract", "--json"], env)
    pin = load_pin()
    if contract.get("schema") != pin["schema"]:
        raise CommonsControlError(
            f"scheduler schema {contract.get('schema')!r} != pinned {pin['schema']!r}")
    if contract.get("owner") != pin["owner"]:
        raise CommonsControlError(
            f"scheduler owner {contract.get('owner')!r} != pinned {pin['owner']!r}")
    actual = abi_fingerprint(contract)
    if actual != pin["fingerprint"]:
        raise CommonsControlError(
            f"scheduler fingerprint drift: {actual} != {pin['fingerprint']}")
    return contract


def scheduler_status(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    verify_scheduler_contract(env)
    return _invoke(["status"], env)


def inspect_job(
    job_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_scheduler_contract(env)
    return _invoke(["inspect", "--job", job_id], env)


def submit_cpu_experiment(
    *,
    experiment_path: Path,
    repository_revision: str,
    run_scope: str,
    ram_mib: int,
    max_minutes: float = 30.0,
    priority: int = 0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Enqueue one existing Climate CPU experiment without changing its semantics."""
    verify_scheduler_contract(env)
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
    ack = _invoke([
        "submit",
        "--name", f"climate-{run_scope}",
        "--repo", "climate",
        "--resource-class", "cpu",
        "--priority", str(priority),
        "--max-minutes", str(max_minutes),
        "--ram", str(ram_mib),
        "--cwd", str(ROOT),
        "--cmd", *command,
    ], env)
    return {
        **ack,
        "experiment": str(experiment.relative_to(ROOT)),
        "outputDir": str(output_dir.relative_to(ROOT)),
        "repositoryRevision": repository_revision,
        "runScope": run_scope,
    }


_EVAL_ABI_KEYS = (
    "schema", "contractVersion", "compatibility", "types", "invariants",
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
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
    """Resolve a registered evaluator and fail closed when its runtime is absent."""
    validated = validate_external_artifact_ref(subject)
    spec = load_external_evaluation_spec(evaluation_id)
    if validated["artifact_contract"] != spec.get("subject_contract"):
        raise CommonsControlError(
            f"evaluator {evaluation_id!r} expects subject contract "
            f"{spec.get('subject_contract')!r}, got "
            f"{validated['artifact_contract']!r}")
    adapter = spec.get("adapter") or {}
    if adapter.get("status") != "available":
        raise ExternalEvaluationUnavailable(
            f"Climate evaluator {evaluation_id!r} is registered but "
            f"adapter {adapter.get('interface')!r} is unavailable; "
            "no attestation was issued")
    return spec



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Climate client for the Commons work scheduler")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--job", required=True)

    submit_parser = sub.add_parser("submit-cpu")
    submit_parser.add_argument("--experiment", type=Path, required=True)
    submit_parser.add_argument("--repository-revision", required=True)
    submit_parser.add_argument("--run-scope", required=True)
    submit_parser.add_argument("--ram", type=int, required=True, dest="ram_mib")
    submit_parser.add_argument("--max-minutes", type=float, default=30.0)
    submit_parser.add_argument("--priority", type=int, default=0)

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = scheduler_status()
        elif args.command == "inspect":
            result = inspect_job(args.job)
        else:
            result = submit_cpu_experiment(
                experiment_path=args.experiment,
                repository_revision=args.repository_revision,
                run_scope=args.run_scope,
                ram_mib=args.ram_mib,
                max_minutes=args.max_minutes,
                priority=args.priority,
            )
    except CommonsControlError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
