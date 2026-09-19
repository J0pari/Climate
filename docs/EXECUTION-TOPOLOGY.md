# Execution topology and resource-class semantics

Status: **binding execution semantics specification**.

This document defines what Climate resource classes mean, which environments can legitimately answer which kinds of questions, and what evidence those environments can produce. It does not choose, prioritize, or schedule repository work.

`architecture/planning_graph.json` is the sole authority for planned work, priority, dependencies, blockers, completion criteria, and each obligation's `resource_class`. `docs/ROADMAP.md` is only its generated projection. Objective realized repository state is rendered separately in `docs/generated/STATUS.md`.

## 1. Resource classes

A resource class states the minimum execution authority needed to answer a question. A result from a weaker class cannot substitute for a result whose declared class is unavailable.

### `R0_static`

No project-code execution is required.

Examples include architecture/contracts, source/module/claim audits, dependency graphs, semantic sanitation, experiment preregistration, static API inspection, and expected-output fixture design.

### `R1_portable_cpu`

Requires common CPU tooling and is suitable for ordinary hosted CI.

Examples include canonical Rust tests, portable Fortran/C/C++, Python/SymPy references, small numerical fixtures, property/metamorphic tests, deterministic serialization/fingerprint checks, and small baseline statistics.

### `R2_toolchain_ci`

Requires a specific portable toolchain that can be provisioned reproducibly in CI.

Examples include Lean/Mathlib kernel checks, CUE vetting, selected language-specific jobs, unusual compiler versions, and cross-language differential checks.

`R2` is not a lower-quality substitute for `R1`; it distinguishes questions whose independent authority depends on a specialized toolchain or checking kernel.

### `R3_cuda_device`

Requires an actual NVIDIA CUDA-capable device and compatible driver/toolkit.

Examples include kernel numerical correctness, CPU/GPU differential tests, race/synchronization behavior, deterministic reductions, tensor-core or mixed-precision behavior, actual cuBLAS/cuSOLVER/cuFFT behavior, occupancy/transfer/VRAM measurements, and device profiling.

No result in this class may be inferred from source inspection, CUDA compilation alone, compatibility shims, or CPU execution.

### `R4_integrated_system`

Requires Commons or another explicitly named integrated runtime component.

Examples include cross-repository run/trace/causation propagation, sandbox execution, global resource leases, cancellation/restart lineage, shared-device contention/cleanup, and contract verification across repository boundaries.

This class also covers experiments whose correctness depends on an actual external model/evaluation runtime rather than a local imitation: for example a ClimSim-Online E3SM-MMF run, an Anemoi/ACE-family inference run, a Climate-REF execution, or another named integrated system. A locally reproduced command shape is not equivalent to the native runtime receipt.

### `R5_large_data`

Requires substantial external data, model ensembles, or protected confirmation splits.

Examples include ERA5/CMIP/ESGF-scale validation, teleconnection confirmation, observational early-warning validation, station-network reconstruction, and process/model evaluation.

Data identity, licensing, provenance, contamination controls, and split policy are part of the resource requirement.

When the external data system already supports data-local lazy/scalable evaluation, as with AQUA in its intended environment, `R5` does not imply copying the corpus into Climate storage. The authoritative execution may occur where the data live; Climate must still bind the exact dataset/version, preprocessing/evaluation configuration, upstream receipt, and resulting immutable artifacts.

## 2. Execution-environment semantics

### GitHub Actions

Hosted CI is the durable execution authority for work whose declared resource class and dependencies can be reproduced there. It is appropriate for portable compilation/tests, symbolic or formal references, contract vetting, architecture integrity, and deterministic synthetic or analytic experiments.

A dedicated workflow or job is justified by a distinct authority/kernel, dependency profile, failure-attribution boundary, hardware class, or meaningful caching boundary—not merely by the existence of another test.

### Interactive development environments

An interactive environment is useful for diagnosis, proof construction, or integration work that materially benefits from interactivity. Results that are portable should be reduced to a durable repository witness in CI or another declared evidence-producing environment.

Repository identity must be explicit before an interactive result is treated as evidence. A stale checkout is a different implementation identity, not an approximation of `main`.

### GitHub Codespaces

GitHub Codespaces is an execution venue for work that already qualifies as `R1_portable_cpu` or `R2_toolchain_ci`; it is not a new resource class and does not weaken a requirement for `R3_cuda_device`, `R4_integrated_system`, or `R5_large_data`. The repository Codespaces environment should therefore reproduce the same experiment identity, method/configuration resolution, failure semantics, and artifact contracts used elsewhere rather than define Codespaces-specific science.

The repository supplies a dev-container bootstrap and a Codespaces campaign runner for the local CPU experiment runtime. Their purpose is to turn otherwise idle included personal-account compute into falsifiable registered runs, not to manufacture activity. A campaign records the exact checkout revision and environment receipt, runs only experiment identities supported by the canonical runtime, writes results under `run-artifacts/`, and leaves durable scientific promotion to the normal evidence registry/review path.

Personal Codespaces included usage, machine availability, billing ownership, and quotas are account-level GitHub policy and may change independently of Climate. Do not hardcode a monthly hour allowance into an experiment or claim. Use included/no-cost capacity when available, configure account spending controls so exhaustion cannot silently become paid execution when that is not intended, stop compute when a campaign is complete, and delete unneeded environments to avoid ongoing storage consumption. Repetition merely to consume quota is not an experiment; repeated runs require a declared stochastic, robustness, scaling, or replication purpose.

A Codespaces result becomes evidence only to the extent its declared resource class is satisfied and its exact environment/revision/artifacts are bound. Network access from a Codespace does not turn an emulated or locally reimplemented external model into an `R4` native-runtime witness, and a small downloaded sample does not turn an `R1`/`R2` run into `R5` large-data evidence.

### CUDA-equipped device

A CUDA-equipped machine is authoritative only for questions that require real hardware: execution correctness, differential behavior, race/determinism, precision ladders, VRAM/transfers, profiling, and integrated device-resource behavior.

Reference fixtures, tolerances, precision semantics, and expected failure cases should be established independently of the device so the hardware result answers the declared question rather than repairing an underspecified test.

### Integrated systems

Cross-repository or orchestration evidence requires the actual declared integrated components. Matching schemas or locally simulated identifiers do not establish end-to-end scheduling, causation, lease, cancellation, cleanup, or lineage behavior.

### Large-data environments

Large-data evidence requires immutable dataset identity, declared preprocessing, split/withholding policy, provenance, missingness semantics, and the compute/storage environment needed to reproduce the result. A small fixture may verify code paths and schema behavior but cannot substitute for an `R5` validation claim.

## 3. Authority and DRY rule

DRY is primarily semantic, not stylistic. Prefer one authored authority for:

- method identity and maturity;
- claim identity and maturity;
- experiment definition;
- native external capability identity and receipt when an experiment crosses an external system boundary;
- realization obligations;
- dataset/run/artifact/evidence identity;
- requested and resolved execution identity;
- resource and accelerator implementation identity;
- planned work, priorities, dependencies, blockers, and completion criteria.

Generated documentation may project existing authorities. Do not invent a second registry or prose queue solely to restate them.

The generated status projection is deliberately local/offline; it does not scrape transient GitHub Actions pass/fail state into committed documentation. CI state is execution evidence, not durable repository status prose.

## 4. Scarce-resource handoff contract

A handoff to a device or integrated-system executor carries enough information to make the requested execution identity unambiguous:

- exact repository revision and canonical branch;
- method/implementation identity;
- fixed input fixtures or dataset digests;
- independent reference outputs where applicable;
- tolerances and normalization;
- precision and determinism requirements;
- expected failure cases;
- resource/layout contract;
- exact evidence-producing commands;
- the question that genuinely requires that resource class.

The receiving environment resolves that request exactly or reports it unavailable. It does not silently substitute a lesser backend, data source, precision, implementation, or static default.

## 5. Claims that remain resource-gated

The following implications are invalid regardless of how plausible they appear:

- CUDA correctness from successful compilation;
- GPU determinism from source inspection;
- performance from operation counts;
- resource envelopes from guessed VRAM arithmetic;
- multi-GPU behavior from MPI/NCCL shims;
- scientific validation from analytic or synthetic fixtures;
- real-data generalization from discovery data;
- cross-repository integration from matching schema names;
- implementation equivalence from a formal proof about only the abstract mathematics.

Unavailable evidence remains unavailable. The planning graph may record the corresponding obligation as blocked by its required resource class; execution code and durable documentation must not replace that absence with a fallback result or optimistic status.
