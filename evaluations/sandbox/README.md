# Sandbox evaluations

This directory stores exploratory computations executed in bounded assistant or other ephemeral CPU environments when they are useful for falsification, replication, robustness analysis, or blocker diagnosis but do not satisfy Climate's evidence-eligibility contract.

A sandbox evaluation must:

- bind the exact repository revision whose semantics it exercised;
- record the actual execution venue and relevant library/toolchain identities;
- state unavailable capabilities explicitly and never substitute another implementation under the same method identity;
- distinguish preregistered experiment replication from post-hoc robustness exploration;
- remain non-promotional unless a later canonical run independently satisfies the repository's execution-resolution, contract-validation, and evidence requirements.

Files here are evaluation records, not entries in `evidence/registry.json`. Canonical evidence belongs under `evidence/runs/` and must retain the exact run, execution, artifact, metric, and verification semantics required by `docs/VALIDATION-AND-EVIDENCE.md`.
