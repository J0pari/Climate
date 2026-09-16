# Execution topology and resource-aware research plan

Status: **binding planning model** for choosing Climate work while capabilities differ across environments.

The numbered roadmap in `docs/ROADMAP.md` describes dependency structure and intended maturity, but it must not be interpreted as a strictly linear schedule. Climate is a research system with heterogeneous dependencies: some questions can be settled by static analysis or small CPU references, some need language-specific CI, some need an actual CUDA device, some need the surrounding Commons/Slime/training/KanForge/LLM-Trader system, and some need large protected climate datasets.

Work should therefore be selected from a dependency/resource graph, not by asking only "what is the next phase?"

## 1. Planning rule

A work item is eligible when:

1. its semantic prerequisites are satisfied;
2. its required resources are actually available;
3. its result can remove meaningful uncertainty or create a reusable boundary;
4. executing it now will not manufacture false evidence about an unavailable resource.

Among eligible items, prefer work with high information gain, broad reuse, or strong bug-prevention value.

A blocked node does not block independent siblings.

## 2. Resource classes

Every substantial task or experiment should eventually declare one of these execution classes, plus a concrete `ResourceEnvelope` where executable.

### `R0_static`

No project code execution is required.

Examples:

- architectural specifications;
- contract/schema design;
- repository/module/claim inventory;
- source audits;
- dependency graph construction;
- literature/method review;
- static CUDA/API inspection;
- experiment preregistration;
- expected-output fixture design.

This environment is currently strong here because it has direct GitHub integration and can commit, inspect CI, and maintain a coherent review history.

### `R1_portable_cpu`

Requires only common CPU tooling available in ordinary CI or the present execution environment.

Examples:

- Python reference mathematics;
- portable C/C++/Fortran kernels;
- analytic and manufactured-solution fixtures;
- property/metamorphic tests;
- deterministic serialization/fingerprint tests;
- small baseline statistics;
- repository tooling.

These tasks should be developed deeply now where they establish independent scientific or numerical oracles.

### `R2_toolchain_ci`

Requires a language/toolchain absent from the current local runtime but reasonably provisioned in GitHub Actions.

Current examples include:

- CUE contract vetting;
- isolated Rust checks;
- selected Julia verification jobs;
- selected Haskell algebra/type tests;
- compile-only CUDA checks where a suitable runner/toolkit exists.

CI is an acceptable source of executable evidence for these questions, but iteration latency and runner differences must be recorded.

### `R3_cuda_device`

Requires an actual NVIDIA CUDA-capable device and compatible driver/toolkit.

Examples:

- CUDA kernel numerical correctness;
- CPU/GPU differential tests;
- race and synchronization behavior;
- deterministic reduction validation;
- tensor-core/mixed-precision experiments;
- actual cuBLAS/cuSOLVER/cuFFT behavior;
- occupancy and launch-geometry profiling;
- host/device transfer measurements;
- VRAM pressure/OOM characterization;
- Nsight profiling;
- device-specific performance envelopes.

No result in this class may be inferred from static source inspection or compile success.

### `R4_integrated_system`

Requires the surrounding multi-repository system or equivalent runtime components.

Examples:

- Commons scheduling and leases;
- cross-repository run/trace/causation propagation;
- shared GPU contention and cleanup;
- sandbox execution;
- cancellation/restart lineage;
- Climate artifact consumption by another repository;
- contract pressure testing across Commons, Slime, training-architecture, KanForge, and LLM-Trader.

Until that system is available, Climate should prepare narrow adapters and executable contract fixtures rather than inventing simulated integration results.

### `R5_large_data`

Requires substantial external climate data, protected confirmation splits, model ensembles, or observational archives.

Examples:

- ERA5/CMIP/ESGF-scale validation;
- teleconnection confirmation tests;
- observational early-warning validation;
- large station-network fault/reconstruction experiments;
- process-oriented model evaluation;
- confirmation of claims discovered on synthetic/development fixtures.

Data availability, licensing, provenance, and contamination controls are part of this resource class.

## 3. Current capability map

The present execution environment has useful local CPU tooling including Python, GCC/G++, gfortran, Node/npm, Go, CMake, and Ninja. It currently lacks a local CUDA device/toolkit (`nvidia-smi`/`nvcc`), Rust/Cargo, Julia, Haskell/GHC, and CUE.

This should shape work selection but not be encoded as a permanent property of Climate. Missing portable toolchains can often be supplied by CI. Missing CUDA hardware is a genuine execution boundary.

The expected CUDA-equipped gaming laptop, when available, should be treated as an `R3_cuda_device` worker and later potentially an `R4_integrated_system` worker. Local agents such as OpenCode/Cursor on that device can run the actual shell/toolchains and are therefore especially valuable for hardware-coupled implementation and profiling loops.

## 4. Parallel frontiers

Several frontiers should remain active concurrently.

### Contract/evidence frontier (`R0`/`R1`/`R2`)

- stabilize `DatasetRef`, `MethodDescriptor`, `ExperimentSpec`, `RunManifest`, `ArtifactRef`, `MetricResult`, `EvidenceRecord`, and `Claim`;
- enforce graph integrity across claims/methods/modules/evidence;
- make maturity and evidence eligibility mechanically constrained;
- align Commons-facing identity/fingerprint/causation fields without giving Commons scientific authority.

### Mathematical reference frontier (`R1`)

- implement deliberately boring reference geometry;
- build analytic/manufactured fixtures;
- implement strong conventional baselines;
- make unit/coordinate/conditioning semantics explicit;
- create independent witnesses for later optimized implementations.

### GPU-engineering frontier (`R0` now, `R3` later)

Now:

- define layouts, buffer ownership, transfer boundaries, precision classes, determinism contracts, capability identities, and failure semantics;
- statically audit legacy CUDA;
- preregister CPU/GPU differential experiments;
- prepare fixed fixtures and expected outputs.

Later on the CUDA device:

- compile and execute kernels;
- verify device results against references;
- measure precision sensitivity, occupancy, transfers, VRAM, and throughput;
- use profiling evidence to decide fusion/layout/tensor-core changes.

Performance claims remain blocked until the latter occurs.

### Scientific-method frontier (`R0`/`R1`, later `R3`/`R5`)

- geometry/curvature;
- information geometry;
- ultrametric/p-adic teleconnections;
- sheaf/cohomological consistency;
- Clifford/resonance representations;
- symmetry diagnostics;
- early-warning indicators.

Each method may progress through formal definition, synthetic verification, baseline comparison, accelerated implementation, and real-data validation at different speeds. Do not force all method families through one shared serial milestone.

### Integration frontier (`R0` now, `R4` later)

Now:

- emit stable static inspection output;
- define execution/resource envelopes;
- preserve Commons-compatible run/trace/causation/fingerprint fields;
- specify sandbox/write boundaries.

Later:

- run one real Climate experiment through Commons;
- propagate IDs end to end;
- exercise GPU leases/cancellation/cleanup;
- test whether shared abstractions survive pressure from other repositories.

## 5. DRY as authority control

DRY is primarily semantic here, not stylistic.

Prefer one canonical source for:

- method identity and maturity;
- claim identity and maturity;
- dataset identity/provenance;
- experiment definition;
- run identity and resolved environment;
- artifact identity;
- evidence relation;
- resource requirements;
- accelerator implementation identity.

README tables, dashboards, Commons projections, CI reports, and experiment launchers should derive from these records rather than maintain parallel meanings.

Duplicated authority is a scientific correctness risk because two descriptions can drift while both remain syntactically valid.

## 6. Layer/partition/interface framing

Climate should preserve these boundaries:

```text
scientific semantics
        ↓
experiment + method contracts
        ↓
reference implementations
        ↓
optimized/accelerated implementations
        ↓
execution + resource control
        ↓
run/artifact provenance
        ↓
evidence
        ↓
claims
```

Authority flows downward only where declared. In particular:

- an optimized implementation does not redefine the mathematical method;
- Commons does not redefine Climate evidence semantics;
- a successful run does not promote a claim;
- a GPU speedup does not validate a climate interpretation;
- a scientific method does not own global scheduling/resource policy.

Cross-layer communication should occur through narrow versioned records, not shared mutable global state.

## 7. Graph framing

Represent work as a DAG where nodes are obligations/capabilities and edges are real prerequisites.

Example geometry path:

```text
geometry definitions
   ├─> analytic fixtures
   ├─> coordinate/unit conventions
   └─> conditioning policy
          ↓
CPU reference implementation
          ↓
reference evidence
          ├───────────────┐
          ↓               ↓
GPU implementation   climate candidate metric design
          ↓               ↓
CPU/GPU differential     candidate meta-experiment
          ↓               ↓
GPU verification      synthetic discrimination
          └──────┬────────┘
                 ↓
          real-data validation
                 ↓
       interpretation mapping
```

The unavailable CUDA node does not block analytic fixtures, CPU references, candidate specification, or baseline design.

## 8. Handoff contract for CUDA-equipped agents

Before asking a device-local agent to optimize a scientific kernel, Climate should provide as much as possible of:

- exact method/implementation identity;
- fixed input fixtures and digests;
- independent reference outputs;
- tolerances and normalization;
- precision/determinism requirements;
- expected failure cases;
- layout/resource contract;
- benchmark metric definitions;
- artifact/run manifest expectations;
- commands that produce evidence;
- unresolved hardware questions.

The device-local task should then be narrow and empirical: make the accelerated implementation satisfy the declared numerical contract and measure its actual resource/performance behavior.

## 9. What must remain gated

Do not promote or assert these before the necessary resource class executes:

- CUDA correctness from successful compilation;
- GPU determinism from code inspection;
- performance from operation counts alone;
- resource envelopes from guessed VRAM arithmetic alone;
- multi-GPU behavior from MPI/NCCL compatibility shims;
- scientific validation from synthetic fixtures;
- real-data generalization from discovery datasets;
- cross-repo integration from matching schema names alone.

Blocked evidence should be represented explicitly rather than replaced with optimistic prose.

## 10. Weekly operating rule while the integrated CUDA system is unavailable

Until the CUDA-equipped integrated system is available again, prioritize:

1. independent mathematical/reference witnesses;
2. contract and graph integrity;
3. synthetic/analytic fixtures and conventional baselines;
4. static source/module audits;
5. toolchain-specific CI only where it answers a concrete question;
6. preregistered GPU experiments and handoff packages;
7. Commons-facing static/read-only interfaces.

Avoid speculative large GPU rewrites or broad cross-repository adapters whose correctness cannot be exercised yet.

When the device/system returns, spend its scarce attention on the questions only it can answer: execution, differential correctness, profiling, resource behavior, and integrated scheduling/lineage.
