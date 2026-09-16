# Climate staged roadmap

Status: **execution order**, optimized to reduce rework and prevent scientific/architectural overclaiming.

This roadmap starts from the repository as it exists today: a flat multi-language tree with stale build metadata, incomplete implementations, useful conventional climate components, and several exploratory mathematical methods.

The sequence is deliberately not "clean up everything, then do science." Each phase should produce a useful, testable capability and evidence about what to do next.

## Phase 0 — freeze current truth

Goal: preserve what exists before structural migration.

Deliverables:

- inventory every source file, language, declared role, obvious placeholder/stub status, and current build reference;
- capture current manifest/build drift (`CORE/`, tests, generated config, etc.);
- classify existing README claims into software/numerical/physical/statistical/predictive/interpretive claim types;
- record which claims currently have no executable evidence;
- create tiny immutable fixtures for a few representative data/algorithm paths;
- do **not** move files yet.

Exit criterion: a machine-readable/static report can describe current repository structure and known inconsistencies without executing scientific workloads.

## Phase 1 — contract and evidence primitives

Goal: make later work describable before making it powerful.

Implement initial language-neutral schemas/records for:

- `DatasetRef`;
- `MethodDescriptor`;
- `ExperimentSpec`;
- `RunManifest`;
- `ArtifactRef`;
- `MetricResult`;
- `EvidenceRecord`;
- `Claim`.

Add schema fixtures with positive and negative controls.

Do not overdesign distributed storage or generated language clients yet. JSON + a constraint/schema layer is enough initially.

Exit criterion: a small example experiment can be represented end to end without prose-only assumptions.

## Phase 2 — structural verifier and repository inspector

Goal: convert today's obvious drift into machine findings.

Implement a stable `inspect` command or equivalent that checks:

- declared source/layout paths;
- manifest/workspace membership;
- build-script references;
- required toolchains;
- missing tests/config templates;
- placeholder/fallback declarations where mechanically detectable;
- external-contract files;
- schema compatibility.

Output machine-readable findings with severity and codes.

Integrate this with Commons `observe` mode first.

Exit criterion: Commons can inspect Climate and report structural readiness without executing repository code.

## Phase 3 — choose and establish the migration skeleton

Goal: create the target directory/package boundaries under tests.

Before moving code, decide:

- whether the Rust crate remains one package or becomes a workspace;
- which Fortran/C++/CUDA kernels are one native library versus several;
- whether Julia/Haskell methods remain first-class build products or research-tool entrypoints;
- canonical CLI location;
- schema/contracts location;
- fixture/test layout.

Move one vertical slice at a time with compatibility tests. Do not recreate `CORE/` merely because stale manifests name it.

Exit criterion: top-level build/config metadata describes real paths, and the repository has one documented minimal build/test path that works without optional HPC dependencies.

## Phase 4 — verification harness before model expansion

Goal: establish trustworthy small numerical tests.

Prioritize infrastructure for:

- unit/dimension checks;
- manufactured/analytic solutions;
- convergence studies;
- conservation diagnostics;
- finite-difference/complex-step gradient verification;
- CPU/GPU differential tests where applicable;
- property/metamorphic tests;
- stochastic equivalence tests.

Use small deterministic fixtures suitable for CI.

Exit criterion: at least one physical/numerical kernel and one statistical method progress from `prototype` to `verified` under explicit criteria.

## Phase 5 — data contract and provenance spine

Goal: make data identity and preprocessing impossible to hand-wave.

Implement:

- CF-aware dataset validation;
- immutable source/projection identity;
- transformation DAG records;
- unit/calendar/grid/anomaly-baseline metadata;
- checks against silent default/imputation in validation paths;
- compact acquisition fixtures separated from large external datasets.

Do not require network access for core CI.

Exit criterion: a small observational/reanalysis fixture can be traced from source identity through preprocessing to experiment input digest.

## Phase 6 — established baseline suite

Goal: create scientifically meaningful comparators before evaluating novel methods.

Implement/reference stable baselines for selected tasks:

- EOF/PCA;
- spectral/coherence analysis;
- conventional early-warning statistics;
- simple graph/hierarchical clustering where relevant;
- standard optimizers/parameter-estimation baselines;
- ordinary station QC/reconstruction baselines;
- simple energy-balance/transport benchmarks.

Exit criterion: experimental methods can enter fair comparisons without each inventing its own baseline code.

## Phase 7 — first five meta-experiments

Goal: evaluate high-information uncertain choices independently.

### 7.1 Geometry correctness

Known Euclidean/spherical/constant-curvature fixtures, coordinate transformations, scaling sensitivity, AD derivative checks.

Block climate tipping interpretation until this passes.

### 7.2 Early-warning discrimination

Conventional critical-slowing indicators versus geometric candidates over tipping and non-tipping synthetic systems, then protected real/model cases.

### 7.3 Teleconnection representation

p-adic/ultrametric versus correlation, spectral/coherence, graph, learned-metric, and generic ultrametric baselines. Include random hierarchy/prime controls.

### 7.4 Station consistency

Actual sheaf/cohomology implementation versus graph/residual/QC baselines with injected faults and coverage gaps.

### 7.5 Information geometry

Explicit small likelihood, synthetic parameter recovery, natural gradient versus standard optimizers/preconditioners, held-out prediction/likelihood.

Exit criterion: each method family has positive controls, negative controls, a primary metric, uncertainty, and a retain/revise/reject outcome.

## Phase 8 — scientific claim registry and documentation rendering

Goal: stop README prose from outrunning evidence.

Implement a claim registry and generate or lint documentation against it.

Examples:

- README can say "investigates curvature as a candidate regime indicator" while claim maturity is `prototype`;
- it cannot say "curvature detects tipping points" as fact unless a validated claim supports that wording.

Add checks for forbidden maturity inflation in generated summaries where feasible.

Exit criterion: important scientific assertions in top-level docs resolve to explicit claim/evidence state.

## Phase 9 — Commons sandbox-read integration

Goal: permit controlled execution without write authority.

Implement a small gate set:

- repository inspect;
- fixture validation;
- method verification by ID;
- experiment execution from spec;
- run summarization.

Requirements:

- structured gate descriptors;
- read-only checkout;
- isolated scratch/artifacts;
- no inherited secrets;
- network denied by default;
- resource/time bounds;
- Commons `RunId`/causation IDs propagated;
- immutable receipts.

Exit criterion: Commons can execute one baseline-vs-candidate fixture experiment and recover its full evidence lineage.

## Phase 10 — observational/model benchmark jobs

Goal: move from toy verification into scoped scientific validation.

Add scheduled, versioned benchmark recipes using selected real datasets/model ensembles. They should not run on every commit.

Key requirements:

- immutable dataset projections;
- protected confirmation splits;
- dependence-aware uncertainty;
- provenance and licensing/citations;
- resource estimates;
- retained failed runs;
- comparison against Phase 6 baselines.

Exit criterion: at least one claim reaches `validated` with explicit scope without upgrading the whole repository's maturity.

## Phase 11 — resource-managed HPC/GPU execution

Goal: integrate expensive workloads without duplicating Commons scheduling.

Clarify what remains local numerical orchestration versus global resource control.

Add:

- explicit CPU/memory/GPU requests;
- transactional Commons leases;
- cancellation and cleanup;
- checkpoint/restart contracts where scientifically valid;
- hardware/numerical equivalence criteria.

Exit criterion: worker failure/cancellation cannot silently corrupt evidence or strand leases, and restarted runs remain distinguishable in lineage.

## Phase 12 — controlled write/PR capability

Goal: let Commons propose Climate changes safely.

Only after previous phases:

- branch/PR-only writes initially;
- work order/issue/experiment causation required;
- protected scientific contracts/golden datasets policy;
- post-write static and fixture gates;
- rollback/revert path;
- human review for changes affecting evidence semantics, validation policy, or scientific interpretation.

Exit criterion: Commons can make a bounded mechanical change and prove exactly what changed, why, what gates ran, and what evidence was affected.

## Phase 13 — broader coupled-system decisions

Only now revisit questions such as:

- should the primitive-equation core become a real coupled climate model or remain a benchmark/reference kernel?
- should experimental methods feed online state transitions or remain diagnostics?
- should a shared state representation exist across languages?
- which language implementations survive based on measured utility/maintainability?
- which methods graduate, remain sandbox research, or are retired?

These are evidence-dependent architecture decisions. Delaying them is intentional.

## Priority rule

When choosing between:

- adding another sophisticated climate method, or
- making one existing claim reproducible, falsifiable, and correctly labeled,

prefer the second until the evidence spine is mature enough that adding methods does not increase ambiguity faster than knowledge.

## Near-term implementation order

The next concrete coding tranche should be:

1. inventory/static findings generator;
2. initial contracts/schemas;
3. placeholder/heuristic annotation mechanism;
4. tiny verification fixtures;
5. real minimal build path;
6. baseline benchmark harness;
7. first geometry correctness meta-experiment.

This ordering attacks the current highest-risk failure mode: the repository can currently express more scientific confidence and architectural completeness than its executable evidence supports.
