#!/usr/bin/env python3
"""Run the canonical local CPU experiment frontier as one receipted campaign.

The campaign runner is intentionally thin. It discovers experiment identities from
Climate's existing experiment specifications and the canonical runtime's explicit
adapter registry; it does not own a second experiment list or redefine scientific
semantics. The primary target is GitHub Codespaces, but the same runner can be used
in any compatible R1/R2 environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
DEFAULT_ARTIFACT_ROOT = ROOT / "run-artifacts" / "codespaces"
CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git(*args: str) -> str:
    result = _run(["git", *args], check=True)
    return result.stdout.strip()


def repository_state() -> dict[str, Any]:
    revision = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(
        _git(
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude)run-artifacts/**",
        )
    )
    return {
        "repository_revision": revision,
        "branch": branch,
        "dirty": dirty,
        "dirty_check_excludes": ["run-artifacts/"],
    }


def _runtime_supported_ids() -> set[str]:
    sys.path.insert(0, str(ROOT))
    try:
        from src import experiment_runtime
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)
    return set(experiment_runtime._experiment_adapters())


def discover_supported_experiments() -> dict[str, Path]:
    supported = _runtime_supported_ids()
    discovered: dict[str, Path] = {}
    for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot load experiment specification {path}: {error}") from error
        if not isinstance(spec, dict):
            continue
        experiment_id = spec.get("experiment_id")
        if experiment_id in supported:
            if experiment_id in discovered:
                raise RuntimeError(f"duplicate experiment_id {experiment_id!r}")
            discovered[str(experiment_id)] = path

    missing = sorted(supported - set(discovered))
    if missing:
        raise RuntimeError(
            "canonical runtime adapter has no matching experiment specification: "
            + ", ".join(missing)
        )
    return dict(sorted(discovered.items()))


def _safe_component(experiment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_id).strip("-")


def _cue_path() -> str:
    cue = shutil.which("cue")
    if cue:
        return cue
    raise RuntimeError(
        "cue is required for campaign contract vetting; rebuild/bootstrap the "
        "Codespace or install the repository-pinned CUE tool"
    )


def _environment_receipt(
    campaign_dir: Path,
    repository: dict[str, Any],
    cue: str,
) -> dict[str, Any]:
    environment_dir = campaign_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)

    freeze = _run([sys.executable, "-m", "pip", "freeze"], check=True)
    freeze_path = environment_dir / "pip-freeze.txt"
    freeze_path.write_text(freeze.stdout, encoding="utf-8")

    cue_version = _run([cue, "version"], check=True)
    (environment_dir / "cue-version.txt").write_text(
        cue_version.stdout + cue_version.stderr,
        encoding="utf-8",
    )

    return {
        "venue": "github_codespaces"
        if os.environ.get("CODESPACES", "").lower() == "true"
        else "compatible_interactive_environment",
        "codespace_name": os.environ.get("CODESPACE_NAME"),
        "codespace_repository": os.environ.get("GITHUB_REPOSITORY"),
        "machine_name": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        **repository,
        "pip_freeze": {
            "path": freeze_path.relative_to(ROOT).as_posix(),
            "digest": _sha256(freeze_path),
        },
        "cue_version": {
            "path": (environment_dir / "cue-version.txt").relative_to(ROOT).as_posix(),
            "digest": _sha256(environment_dir / "cue-version.txt"),
        },
    }


def _vet_experiment_outputs(cue: str, output_dir: Path) -> tuple[bool, str]:
    targets: list[tuple[str, Path]] = [("#ExperimentOutcome", output_dir / "outcome.json")]
    targets.extend(("#RunManifest", path) for path in sorted(output_dir.glob("run-*.json")))

    transcript: list[str] = []
    ok = True
    for definition, path in targets:
        if not path.is_file():
            transcript.append(f"missing {definition} target: {path}\n")
            ok = False
            continue
        command = [cue, "vet", "-d", definition, "contracts/climate.cue", str(path)]
        result = _run(command)
        transcript.append("$ " + " ".join(command) + "\n")
        transcript.append(result.stdout)
        transcript.append(result.stderr)
        if result.returncode != 0:
            ok = False
    return ok, "".join(transcript)


def run_campaign(
    experiment_ids: list[str],
    *,
    campaign_id: str,
    artifact_root: Path,
    allow_dirty: bool,
) -> tuple[dict[str, Any], int]:
    if not CAMPAIGN_ID_RE.fullmatch(campaign_id):
        raise ValueError("campaign id must contain only letters, digits, '.', '_' or '-'")

    available = discover_supported_experiments()
    unknown = sorted(set(experiment_ids) - set(available))
    if unknown:
        raise ValueError("unsupported experiment id(s): " + ", ".join(unknown))

    repository = repository_state()
    if repository["dirty"] and not allow_dirty:
        raise RuntimeError(
            "working tree is dirty; commit the exact experiment implementation first "
            "or pass --allow-dirty for non-evidence exploration"
        )

    cue = _cue_path()
    campaign_dir = artifact_root / campaign_id
    if campaign_dir.exists():
        raise RuntimeError(f"campaign output already exists: {campaign_dir}")
    campaign_dir.mkdir(parents=True)

    started = datetime.now(timezone.utc).isoformat()
    environment = _environment_receipt(campaign_dir, repository, cue)
    records: list[dict[str, Any]] = []
    overall_ok = True

    for experiment_id in experiment_ids:
        spec_path = available[experiment_id]
        output_dir = campaign_dir / "experiments" / _safe_component(experiment_id)
        output_dir.mkdir(parents=True)
        run_scope = f"codespaces.{campaign_id}.{_safe_component(experiment_id)}"
        command = [
            sys.executable,
            "src/experiment_runtime.py",
            "--experiment",
            spec_path.relative_to(ROOT).as_posix(),
            "--output-dir",
            output_dir.relative_to(ROOT).as_posix(),
            "--repository-revision",
            str(repository["repository_revision"]),
            "--run-scope",
            run_scope,
        ]
        result = _run(command)
        (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

        contract_ok = False
        vet_log = ""
        if result.returncode == 0:
            contract_ok, vet_log = _vet_experiment_outputs(cue, output_dir)
        (output_dir / "contract-vet.txt").write_text(vet_log, encoding="utf-8")

        outcome = output_dir / "outcome.json"
        passed = result.returncode == 0 and contract_ok and outcome.is_file()
        overall_ok = overall_ok and passed
        records.append(
            {
                "experiment_id": experiment_id,
                "experiment_spec": spec_path.relative_to(ROOT).as_posix(),
                "experiment_spec_digest": _sha256(spec_path),
                "command": command,
                "run_scope": run_scope,
                "exit_code": result.returncode,
                "contract_valid": contract_ok,
                "status": "passed" if passed else "failed",
                "output_dir": output_dir.relative_to(ROOT).as_posix(),
                "outcome_digest": _sha256(outcome) if outcome.is_file() else None,
            }
        )

    finished = datetime.now(timezone.utc).isoformat()
    campaign = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "purpose": "execute canonical registered local CPU experiments without redefining their scientific semantics",
        "started_at": started,
        "finished_at": finished,
        "environment": environment,
        "experiments": records,
        "status": "passed" if overall_ok else "failed",
    }
    _write_json(campaign_dir / "campaign.json", campaign)

    summary = [
        f"# Codespaces experiment campaign {campaign_id}",
        "",
        f"- Revision: `{repository['repository_revision']}`",
        f"- Venue: `{environment['venue']}`",
        f"- Status: **{campaign['status']}**",
        "",
        "| Experiment | Status | Contract | Outcome digest |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        summary.append(
            f"| `{record['experiment_id']}` | `{record['status']}` | "
            f"`{'valid' if record['contract_valid'] else 'invalid/unavailable'}` | "
            f"`{record['outcome_digest'] or '—'}` |"
        )
    summary.extend(
        [
            "",
            "This summary is an execution artifact, not scientific promotion. "
            "Evidence enters repository authority only through the normal reviewed evidence path.",
        ]
    )
    (campaign_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return campaign, 0 if overall_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="run canonical local CPU experiments as a receipted Codespaces campaign"
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="experiment_id to run; repeat to select several (default: all supported)",
    )
    parser.add_argument("--list", action="store_true", help="list supported experiment ids")
    parser.add_argument("--campaign-id")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow exploratory execution from a dirty checkout; not suitable as durable evidence",
    )
    args = parser.parse_args()

    available = discover_supported_experiments()
    if args.list:
        for experiment_id, path in available.items():
            print(f"{experiment_id}\t{path.relative_to(ROOT).as_posix()}")
        return 0

    selected = args.experiments or list(available)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    revision = _git("rev-parse", "HEAD")
    campaign_id = args.campaign_id or f"{now}-{revision[:12]}"
    campaign, exit_code = run_campaign(
        selected,
        campaign_id=campaign_id,
        artifact_root=args.artifact_root.resolve(),
        allow_dirty=args.allow_dirty,
    )
    print((args.artifact_root.resolve() / campaign["campaign_id"] / "SUMMARY.md"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
