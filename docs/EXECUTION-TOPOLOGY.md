# Execution topology and resource-aware research plan

Status: **binding planning model** for choosing Climate work while capabilities differ across environments.

Climate is a resource/dependency graph, not a serial build. Some questions can be settled by static analysis or ordinary CPU CI, some need language-specific toolchains, some need a real CUDA device, some need the surrounding Commons system, and some need substantial protected climate data.

Objective repository state is rendered from local authorities in `docs/generated/STATUS.md`; this document defines **where a question should execute and what that execution can legitimately prove**.

## 1. Planning rule

A work item is eligible when:

1. its semantic prerequisites are satisfied;
2. its required resources are actually available;
3. its result removes meaningful uncertainty or creates a reusable boundary;
4. running it now will not manufacture false evidence about an unavailable resource.

Among eligible work, prefer high information gain, broad reuse, independent witnesses, and bug-prevention leverage. A blocked node does not block independent siblings.

## 2. Resource classes

### `R0_static`

No project-code execution is required.

Examples: architecture/contracts, source/module/claim audits, dependency graphs, semantic sanitation, experiment preregistration, static CUDA/API inspection, expected-output fixture design.

### `R1_portable_cpu`

Requires common CPU tooling and should normally run in hosted CI.

Examples: canonical Rust tests, portable Fortran/C/C++, Python/SymPy references, small numerical fixtures, property/metamorphic tests, deterministic serialization/fingerprint checks, small baseline statistics.

### `R2_toolchain_ci`

Requires a specific portable toolchain that can be provisioned reproducibly in CI.

Examples: Lean/Mathlib kernel checks, CUE vetting, selected Julia/Haskell jobs, unusual compiler versions, cross-language differential checks.

`R2` is not a lower-quality substitute for `R1`; it distinguishes questions whose independent authority comes from a specialized toolchain/kernel.

### `R3_cuda_device`

Requires an actual NVIDIA CUDA-capable device and compatible driver/toolkit.

Examples: kernel numerical correctness, CPU/GPU differential tests, race/synchronization behavior, deterministic reductions, tensor-core/mixed-precision behavior, actual cuBLAS/cuSOLVER/cuFFT behavior, occupancy/transfer/VRAM measurements, Nsight profiling.

No result in this class may be inferred from source inspection, CUDA compilation alone, compatibility shims, or CPU execution.

### `R4_integrated_system`

Requires Commons/other repositories or equivalent runtime components.

Examples: cross-repository run/trace/causation propagation, sandbox execution, global resource leases, cancellation/restart lineage, shared GPU contention/cleanup, contract pressure-testing across repositories.

### `R5_large_data`

Requires substantial external data, model ensembles, or protected confirmation splits.

Examples: ERA5/CMIP/ESGF-scale validation, teleconnection confirmation, observational early-warning validation, station-network reconstruction, process/model evaluation.

Data identity, licensing, provenance, contamination controls, and split policy are part of the resource requirement.

## 3. Execution-environment policy

### GitHub Actions: default executable authority for portable work

Use hosted CI first when the question is reproducible on ordinary CPU/toolchain infrastructure. Current examples include:

- canonical Rust compilation/tests;
- canonical Fortran configuration/build/tests;
- symbolic geometry/information-geometry/Noether/sheaf/Lie references;
- Lean/Mathlib kernel checks;
- CUE and architecture integrity;
- deterministic synthetic/analytic experiments.

CI is preferred because it is independently reproducible, leaves durable logs, and does not consume scarce interactive Codespace/device time.

Do not create a separate workflow per tiny test if it adds no isolation value. A dedicated workflow/job is justified when it gives a distinct authority/kernel, dependency profile, failure attribution, or caching boundary.

### Codespace: scarce interactive parity/integration environment

Use Codespace only when interactivity or environment composition materially helps, for example:

- diagnosing a toolchain problem that CI logs do not localize efficiently;
- proving a new Lean statement interactively before moving the stable proof into CI;
- multi-toolchain parity/integration that is awkward to express in hosted jobs;
- networked build/package debugging;
- preparing a later hardware handoff.

A Codespace result should migrate into CI or another durable witness whenever the question is portable.

Before every Codespace task, explicitly switch/update the canonical branch rather than assuming the existing checkout is current:

```bash
git switch main
git pull --ff-only
git rev-parse HEAD
git status --short
```

Do not spend Codespace credits repeatedly verifying facts CI can already establish.

### CUDA-equipped local device: scarce hardware witness

Reserve the CUDA-equipped machine for questions only real hardware can answer: execution correctness, differential behavior, race/determinism, precision ladders, VRAM/transfers, profiling, and integrated scheduling/resource behavior.

Prepare fixtures/reference outputs/tolerances before using that device so hardware time is spent on empirical uncertainty rather than discovering missing specifications.

## 4. Parallel frontiers

### Contract/evidence frontier (`R0`/`R1`/`R2`)

- stabilize Dataset/Method/Experiment/Run/Artifact/Evidence/Claim contracts;
- enforce graph integrity and fingerprint/causation semantics;
- keep maturity/evidence mechanically constrained;
- align Commons-facing identity without giving Commons scientific authority.

### Mathematical/reference frontier (`R1`/`R2`)

Use several kinds of independent authority where they add real leverage:

- symbolic references for differential/variational/statistical identities;
- Lean for small stable propositions where kernel checking reduces ambiguity;
- canonical Rust/Fortran implementations for executable portable algorithms;
- differential/metamorphic tests between independent realizations.

Formal proof establishes consequences of a mathematical statement. It does not prove that a climate representation, likelihood, metric, or interpretation is scientifically appropriate.

### Scientific-method frontier (`R0`/`R1`, later `R3`/`R5`)

Geometry, information geometry, ultrametric/p-adic, sheaf/cohomology, Clifford, symmetry, modal/scenario, spectral, feedback, and early-warning methods may progress at different speeds through:

```text
definition → exact/reference witness → canonical implementation
→ baseline/ablation → acceleration → synthetic validation → real-data validation
```

Do not force all method families through one serial milestone.

### GPU frontier (`R0`/`R1` now, `R3` later)

Before hardware:

- define layouts/buffer ownership;
- precision/conditioning/determinism contracts;
- immutable CPU/reference fixtures;
- failure semantics;
- candidate-batch isolation tests;
- stage-level differential expectations.

On hardware:

- execute and compare stage by stage;
- measure resource/performance behavior;
- reject accelerator changes that alter declared numerical semantics outside tolerance.

### Integration frontier (`R0` now, `R4` later)

Now: stable inspection, experiment/resource records, Commons-compatible identities, sandbox/write boundaries.

Later: run one real experiment through Commons and verify end-to-end causation, artifacts, evidence, leases, cancellation, and cleanup.

## 5. Authority and DRY rule

DRY is primarily semantic, not stylistic. Prefer one authored authority for:

- method identity/maturity;
- claim identity/maturity;
- experiment definition;
- realization obligations;
- dataset/run/artifact/evidence identity;
- resource and accelerator implementation identity.

Generated documentation may project existing authorities. Do **not** invent a new registry solely to generate prose.

The generated status projection is deliberately local/offline; it does not scrape GitHub Actions pass/fail state into committed documentation. CI state is execution evidence and changes too quickly to become a checked-in factual table.

## 6. Handoff contract for scarce-resource agents

Before asking a device/local agent to work on a scientific kernel, provide as much as possible of:

- exact repository revision and canonical branch;
- method/implementation identity;
- fixed input fixtures/digests;
- independent reference outputs;
- tolerances/normalization;
- precision/determinism requirements;
- expected failure cases;
- resource/layout contract;
- exact evidence-producing commands;
- unresolved questions that genuinely require that environment.

The handoff should ask the scarce resource to resolve uncertainty, not perform routine repository archaeology.

## 7. What must remain gated

Do not promote or assert:

- CUDA correctness from successful compilation;
- GPU determinism from source inspection;
- performance from operation counts;
- resource envelopes from guessed VRAM arithmetic;
- multi-GPU behavior from MPI/NCCL shims;
- scientific validation from analytic/synthetic fixtures;
- real-data generalization from discovery data;
- cross-repository integration from matching schema names;
- implementation equivalence from a formal proof about only the abstract mathematics.

Blocked evidence is represented explicitly rather than replaced by optimistic prose.

## 8. Operating rule while scarce hardware/integration is unavailable

Prioritize:

1. exact/reference/formal witnesses with reusable leverage;
2. canonical portable implementations;
3. semantic sanitation and monolith decomposition;
4. conventional baselines and falsifiable synthetic experiments;
5. data/provenance contracts and small immutable fixtures;
6. resource-gated handoff packages;
7. Commons-facing static/read-only interfaces.

When scarce hardware/system access returns, spend it on execution, differential correctness, profiling, resource behavior, large-data confirmation, and integrated scheduling/lineage—the questions ordinary CI cannot answer.
