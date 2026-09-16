#!/usr/bin/env python3
"""Static architecture/scientific source audits for Climate.

The current repository intentionally contains known implementation debt, so the
default command is an audit: findings are printed but do not fail the process.
`--strict` turns findings into a failing gate. The roadmap should only promote a
rule to required CI after either the current debt is fixed or an explicit,
reviewed allowlist with an expiry exists.

Every rule in this module must have a planted negative test. A source gate that
has never demonstrated it can catch its target failure is not evidence.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from architecture import source_surface


@dataclass(frozen=True)
class Finding:
    gate: str
    path: str
    line: int
    message: str
    text: str

    def render(self) -> str:
        return f"{self.gate}: {self.path}:{self.line}: {self.message}\n    {self.text.strip()}"


def production_files(root: Path = ROOT) -> Iterable[Path]:
    """Yield the shared scientific/reference/experiment audit surface."""
    for item in source_surface.iter_source_files(
        root, roles=source_surface.AUDITED_ROLES
    ):
        yield item.path


def read_sources(root: Path = ROOT) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in production_files(root):
        rel = path.relative_to(root.resolve()).as_posix()
        try:
            result[rel] = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
    return result


def _code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return not (
        stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
        or stripped.startswith("!")
        or stripped.startswith("# ")
        or stripped.startswith("--")
    )


def gate_managed_memory(files: dict[str, list[str]]) -> list[Finding]:
    findings: list[Finding] = []
    for path, lines in files.items():
        if not path.endswith((".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp")):
            continue
        for line_no, line in enumerate(lines, 1):
            if _code_line(line) and "cudaMallocManaged" in line:
                findings.append(Finding(
                    "managed_memory", path, line_no,
                    "managed memory is not allowed in evidence-critical GPU paths without an explicit contract exception",
                    line,
                ))
    return findings


CUDA_CALL = re.compile(
    r"\b(cudaMalloc|cudaMallocHost|cudaFree|cudaFreeHost|cudaMemcpy|cudaMemcpyAsync|"
    r"cudaMemset|cudaMemsetAsync|cudaStreamCreate|cudaStreamDestroy|"
    r"cudaStreamSynchronize|cudaDeviceSynchronize)\s*\("
)
CHECK_CONTEXT = re.compile(
    r"CUDA_(?:CHECK|ABORT|WARN)|cudaSuccess|cudaError_t|\b(?:err|error|status)\b",
    re.IGNORECASE,
)


def gate_unchecked_cuda_calls(files: dict[str, list[str]]) -> list[Finding]:
    """Flag obvious raw CUDA calls with no local checked context.

    This is intentionally conservative and line-oriented. It is an audit aid,
    not a parser. Production activation should move calls behind a small set of
    checked wrappers, at which point the rule can become stricter.
    """
    findings: list[Finding] = []
    for path, lines in files.items():
        if not path.endswith((".cu", ".cuh", ".cpp", ".cc", ".cxx")):
            continue
        for line_no, line in enumerate(lines, 1):
            if not _code_line(line) or not CUDA_CALL.search(line):
                continue
            neighborhood = " ".join(lines[max(0, line_no - 2): min(len(lines), line_no + 1)])
            if CHECK_CONTEXT.search(neighborhood):
                continue
            findings.append(Finding(
                "unchecked_cuda_call", path, line_no,
                "CUDA runtime call has no obvious checked error context",
                line,
            ))
    return findings


FALLBACK_MARKERS = (
    "compatibility shim",
    "non-functional mpi",
    "nccl stub",
    "wmma fallback",
    "cpu fallback",
    "fallback path",
)


def gate_silent_capability_fallback(files: dict[str, list[str]]) -> list[Finding]:
    """Inventory capability fallbacks that must never masquerade as validated execution."""
    findings: list[Finding] = []
    for path, lines in files.items():
        for line_no, line in enumerate(lines, 1):
            lower = line.lower()
            if any(marker in lower for marker in FALLBACK_MARKERS):
                findings.append(Finding(
                    "capability_fallback", path, line_no,
                    "capability fallback/shim must resolve to a distinct implementation identity in evidence runs",
                    line,
                ))
    return findings


PLACEHOLDER_MARKERS = re.compile(
    r"\b(TODO|FIXME|placeholder|stub|unimplemented|not implemented|unvalidated)\b",
    re.IGNORECASE,
)


def gate_placeholder_inventory(files: dict[str, list[str]]) -> list[Finding]:
    """Inventory source that cannot silently be promoted to verified status."""
    findings: list[Finding] = []
    for path, lines in files.items():
        for line_no, line in enumerate(lines, 1):
            if PLACEHOLDER_MARKERS.search(line):
                findings.append(Finding(
                    "placeholder_inventory", path, line_no,
                    "placeholder/unvalidated marker requires explicit method maturity and blocks verified promotion",
                    line,
                ))
    return findings


# Patterns below target process-global/default RNG state or constructors with
# no explicit seed. Explicit generators such as np.random.default_rng(seed)
# are intentionally allowed; their seed still has to be captured by RunManifest.
AMBIENT_RNG_PATTERNS = (
    re.compile(r"\brand\s*\("),
    re.compile(r"\bsrand\s*\("),
    re.compile(r"\bthread_rng\s*\("),
    re.compile(r"\b(?:np|numpy)\.random\.(?:rand|randn|random|random_sample|choice|normal|uniform|shuffle|permutation)\s*\("),
    re.compile(r"\b(?:np|numpy)\.random\.default_rng\s*\(\s*\)"),
    re.compile(r"\bRandom\.default_rng\s*\(\s*\)"),
)


def gate_ambient_rng(files: dict[str, list[str]]) -> list[Finding]:
    findings: list[Finding] = []
    for path, lines in files.items():
        for line_no, line in enumerate(lines, 1):
            if not _code_line(line):
                continue
            if any(pattern.search(line) for pattern in AMBIENT_RNG_PATTERNS):
                findings.append(Finding(
                    "ambient_rng", path, line_no,
                    "evidence-producing stochastic code must use an explicit recorded RNG/seed",
                    line,
                ))
    return findings


PROBABILITY_ASSIGNMENT = re.compile(
    r"\b(probability|confidence)\s*[:=].*(sigmoid|exp\s*\(|min\s*\(|max\s*\(|/\s*[0-9])",
    re.IGNORECASE,
)


def gate_interpretive_probability(files: dict[str, list[str]]) -> list[Finding]:
    """Flag heuristic probability/confidence mappings for human review.

    This intentionally errs toward false positives: it detects the dangerous
    class where an arbitrary numerical indicator is transformed into a field
    named probability/confidence without a visible calibration contract.
    """
    findings: list[Finding] = []
    for path, lines in files.items():
        for line_no, line in enumerate(lines, 1):
            if _code_line(line) and PROBABILITY_ASSIGNMENT.search(line):
                findings.append(Finding(
                    "interpretive_probability", path, line_no,
                    "probability/confidence appears derived by heuristic transform; require a versioned validated mapping",
                    line,
                ))
    return findings


Gate = Callable[[dict[str, list[str]]], list[Finding]]
GATES: tuple[Gate, ...] = (
    gate_managed_memory,
    gate_unchecked_cuda_calls,
    gate_silent_capability_fallback,
    gate_placeholder_inventory,
    gate_ambient_rng,
    gate_interpretive_probability,
)


def run(files: dict[str, list[str]]) -> list[Finding]:
    findings: list[Finding] = []
    for gate in GATES:
        findings.extend(gate(files))
    return sorted(findings, key=lambda f: (f.path, f.line, f.gate))


def main() -> int:
    parser = argparse.ArgumentParser(description="audit Climate source for architecture/scientific hazards")
    parser.add_argument("--strict", action="store_true", help="return non-zero when any finding exists")
    parser.add_argument("--summary", action="store_true", help="show counts only")
    args = parser.parse_args()

    files = read_sources()
    findings = run(files)
    if args.summary:
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.gate] = counts.get(finding.gate, 0) + 1
        for gate, count in sorted(counts.items()):
            print(f"{gate}: {count}")
        print(f"total: {len(findings)}")
    else:
        for finding in findings:
            print(finding.render())
        print(f"\n{len(findings)} finding(s) across {len(files)} source file(s)")

    return 1 if args.strict and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
