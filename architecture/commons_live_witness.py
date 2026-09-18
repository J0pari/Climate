#!/usr/bin/env python3
"""Real-machine witness for Climate read execution through Commons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from architecture import commons_control

ROOT = Path(__file__).resolve().parents[1]
TERMINAL_STATUSES = {"done", "failed", "cancelled"}
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 1800.0


class LiveWitnessError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_outcome(
    *,
    experiment_path: Path,
    output_dir: Path,
    repository_revision: str,
) -> dict[str, Any]:
    experiment = _load_json(experiment_path)
    outcome_path = output_dir / "outcome.json"
    if not outcome_path.is_file():
        raise LiveWitnessError(f"Climate outcome is missing: {outcome_path}")
    outcome = _load_json(outcome_path)
    if outcome.get("experiment_id") != experiment.get("experiment_id"):
        raise LiveWitnessError(
            "Climate outcome experiment_id does not match the submitted spec")
    runs = outcome.get("runs")
    if not isinstance(runs, list) or not runs:
        raise LiveWitnessError("Climate outcome contains no run manifests")
    mismatched = [
        run.get("run_id")
        for run in runs
        if run.get("repository_revision") != repository_revision
    ]
    if mismatched:
        raise LiveWitnessError(
            "Climate outcome contains run(s) from a different repository revision: "
            + ", ".join(str(item) for item in mismatched))
    return outcome


def run_live_witness(
    *,
    experiment_path: Path,
    repository_revision: str,
    run_scope: str,
    ram_mib: int,
    max_minutes: float,
    priority: int,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise LiveWitnessError("poll and timeout intervals must be positive")
    ack = commons_control.submit_cpu_experiment(
        experiment_path=experiment_path,
        repository_revision=repository_revision,
        run_scope=run_scope,
        ram_mib=ram_mib,
        max_minutes=max_minutes,
        priority=priority,
    )
    job_id = ack.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise LiveWitnessError("Commons submission returned no jobId")

    deadline = time.monotonic() + timeout_seconds
    job: dict[str, Any] | None = None
    while True:
        job = commons_control.inspect_job(job_id)
        status = job.get("status")
        if status in TERMINAL_STATUSES:
            break
        if time.monotonic() >= deadline:
            raise LiveWitnessError(
                f"Commons job {job_id} did not reach a terminal state")
        time.sleep(poll_seconds)

    if job.get("status") != "done" or job.get("exitCode") != 0:
        raise LiveWitnessError(
            f"Commons job {job_id} ended as {job.get('status')!r} "
            f"with exitCode={job.get('exitCode')!r}")

    output_rel = ack.get("outputDir")
    if not isinstance(output_rel, str) or not output_rel:
        raise LiveWitnessError("Climate submission did not return outputDir")
    output_dir = (ROOT / output_rel).resolve()
    allowed_root = (ROOT / "run-artifacts" / "commons").resolve()
    try:
        output_dir.relative_to(allowed_root)
    except ValueError as error:
        raise LiveWitnessError(
            f"Climate outputDir escaped the Commons artifact root: {output_rel}") from error

    outcome = validate_outcome(
        experiment_path=experiment_path.resolve(),
        output_dir=output_dir,
        repository_revision=repository_revision,
    )
    return {
        "jobId": job_id,
        "jobStatus": job["status"],
        "outputDir": output_rel,
        "experimentId": outcome["experiment_id"],
        "repositoryRevision": repository_revision,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="run the real Commons -> Climate read-execution witness")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--run-scope", required=True)
    parser.add_argument("--ram", type=int, required=True, dest="ram_mib")
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    try:
        result = run_live_witness(
            experiment_path=args.experiment,
            repository_revision=args.repository_revision,
            run_scope=args.run_scope,
            ram_mib=args.ram_mib,
            max_minutes=args.max_minutes,
            priority=args.priority,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except (commons_control.CommonsControlError, LiveWitnessError, OSError, json.JSONDecodeError) as error:
        print(str(error))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
