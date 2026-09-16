# Climate target architecture

Status: **architectural specification / migration target**. This document describes the shape that is justified before the repository has enough empirical evidence to settle every modeling choice. It is deliberately more stable than any one current implementation.

## 1. Architectural objective

Climate should become a research system in which conventional climate methods and unusual mathematical hypotheses can be implemented, verified, compared, falsified, and reproduced without confusing any of those steps with scientific validation.

The architecture should optimize for four properties:

1. **scientific separability** — changing one hypothesis should not silently alter unrelated physics, data handling, or evaluation;
2. **evidence traceability** — every result that matters can be reconstructed from immutable inputs, code, configuration, and environment;
3. **fair comparison** — novel methods and established baselines enter the same experiment/evaluation machinery;
4. **control-plane compatibility** — Commons can inspect, schedule, identify, trace, and compare work without owning Climate's scientific semantics.

This repository should not attempt to become a monolithic all-purpose Earth system model before its components have earned that role.

## 2. The stable waist

The core architecture should converge on a small set of language-neutral records. Exact serialization is not fixed here; CUE/JSON is a reasonable first contract layer, with generated language bindings later if useful.

### `DatasetRef`

Identifies an input dataset or immutable projection of one.

Required conceptual fields:

```text
id / digest
source family
source version or retrieval identity
variable/field identities
spatial domain/grid
vertical coordinates
calendar/time domain
temporal resolution
units/CF standard names where applicable
quality-control policy
preprocessing graph
license/citation
materialized artifact digest(s)
```

For climate data, CF metadata should be preferred over project-local reinterpretations where CF can express the concept. CF 1.12 is a useful current baseline: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html

### `MethodDescriptor`

Describes one executable scientific or numerical method independently of an experiment.

```text
method_id
semantic_version
implementation_build
maturity
hypothesis_family
input_contracts
output_contracts
assumptions
known_limitations
resource_requirements
reference/baseline family
validation status by claim
```

A method descriptor does **not** state that the method is scientifically correct. It states what implementation and interpretation are being evaluated.

### `ExperimentSpec`

Defines a falsifiable comparison rather than just a command.

```text
experiment_id
question / hypothesis
intervention or candidate method
control / baseline methods
dataset_refs
split / withholding policy
preprocessing
seeds
resource envelope
primary metrics
secondary diagnostics
negative controls
falsifiers
stopping rule
multiple-comparison policy
expected artifacts
```

The same specification should be executable locally and schedulable by Commons without changing scientific meaning.

### `RunManifest`

Records what actually happened.

```text
run_id
experiment_id
repository revision
method implementation build(s)
contract fingerprints
resolved input digests
resolved configuration
random seeds
host/container/environment identity
hardware identity where numerically relevant
start/end times
exact commands / entrypoints
exit status
stdout/stderr digests
produced artifact refs
parent/causation ids
```

### `ArtifactRef`

Large arrays, checkpoints, NetCDF/Zarr outputs, logs, figures, and reports should live outside event envelopes and be referenced by digest plus media/schema metadata.

### `EvidenceRecord`

Links a claim to observations, simulations, tests, or other witnesses.

```text
claim_id
evidence_class
verification_status
supports | attacks | depends_on
run_id
artifact_refs
metric results
scope/domain of applicability
known threats to validity
```

### `MetricResult`

A metric must carry semantics, not just a number.

```text
metric_id + version
value
units
aggregation domain
sampling distribution / interval if applicable
reference population/dataset
higher/lower/is-target semantics
missing-data policy
uncertainty method
```

A value such as `0.84` without this context is not a stable scientific interface.

## 3. Layered system shape

### Layer A — acquisition and normalization

Responsibilities:

- retrieve data from ERA5/ESGF/CMIP, GHCN, Argo, satellite archives, etc.;
- preserve original source identity and license/citation;
- decode calendars, units, masks, quality flags, vertical coordinates, and grids;
- make every transformation explicit;
- produce immutable, analysis-ready dataset projections.

Preferred tools can remain Python/xarray/Dask/Zarr/NetCDF where they fit. Xarray's CF-aware decoding and non-standard calendar support are useful patterns rather than reasons to invent a new climate file abstraction.

Hard rule: **download/cache success is not data validity**. Schema checks, coordinate sanity, units, temporal coverage, missingness, and source-specific QC must be separate gates.

### Layer B — reference physical and numerical kernels

This layer owns equations and discretizations intended to model physical climate dynamics.

Examples:

- primitive-equation tendencies;
- thermodynamic/moisture equations;
- radiation/forcing approximations;
- ocean/land/ice component kernels;
- transport/advection/diffusion;
- coupling/remapping;
- spectral transforms;
- time integration.

The target is not one giant mutable `ClimateState` object shared by all experimental code. Kernels should expose typed state/field contracts and pure or tightly bounded transitions where practical.

Numerical schemes must declare conservation properties, stability/CFL assumptions, discretization order, grid requirements, and supported boundary conditions. Conservative remapping should use established methods/libraries when possible; ESMF's conservative regridding support is a relevant reference: https://earthsystemmodeling.org/docs/release/ESMF_8_9_0/ESMF_refdoc/node5.html

### Layer C — established diagnostics and inference

This contains methods with recognizable climate/statistical precedent, for example:

- climatologies/anomalies;
- spectra, cross-spectra, coherence, wavelets;
- EOF/PCA and related modal decompositions;
- feedback regression/decomposition;
- energy-budget and Gregory-style diagnostics;
- ensemble statistics;
- likelihood/Bayesian parameter estimation;
- Fisher information/identifiability when based on a well-defined statistical model;
- standard early-warning statistics;
- model-performance diagnostics.

These methods still require local verification and validation. "Established" means the method family has precedent, not that this implementation is correct.

### Layer D — experimental mathematical methods

This is a first-class research layer, not a junk drawer.

Current candidates include:

- Riemannian/geometric state-space methods;
- curvature-derived regime indicators;
- p-adic/ultrametric teleconnection representations;
- Clifford/geometric-algebra resonance representations;
- sheaf/cohomological consistency methods;
- symmetry/Noether-inspired diagnostics;
- modal-logic scenario constraints;
- learned latent manifold/dimension discovery.

Every method in this layer should use the same `MethodDescriptor -> ExperimentSpec -> RunManifest -> EvidenceRecord` pipeline as conventional baselines.

Experimental methods must not directly assert authoritative physical events such as "AMOC collapse probability = X" unless a separately validated mapping exists from the mathematical quantity to that physical interpretation.

### Layer E — validation and benchmarking

This layer is intentionally independent from the implementation under test.

Responsibilities include:

- analytic/manufactured-solution verification;
- convergence studies;
- conservation/budget closure;
- differential/cross-language tests;
- benchmark suites;
- observational/reanalysis comparison;
- CMIP-context comparison;
- out-of-sample validation;
- robustness/sensitivity analysis;
- uncertainty/calibration analysis;
- negative controls and adversarial cases.

Climate should learn from tools such as ESMValTool without reimplementing an entire evaluation ecosystem. ESMValTool's recipe-based model/observation comparisons and provenance approach are useful external reference points: https://docs.esmvaltool.org/en/latest/

### Layer F — orchestration and provenance

Climate should expose work; it should not become its own distributed control plane if Commons owns that role.

Local orchestration may still be needed for:

- a single experiment process;
- intra-process parallelism;
- GPU kernels;
- model component coupling;
- local checkpoint/restart.

Workspace-level scheduling, cross-repository dependencies, global GPU leases, experiment DAGs, run identity, and cross-repo lineage should align with Commons rather than duplicate it.

## 4. Physical state versus research artifacts

A central architectural correction is to stop treating every mathematical representation as another field of one universal physical state.

Use three categories:

1. **prognostic/diagnostic physical state** — quantities with explicit units, grids, equations, and physical meaning;
2. **derived scientific diagnostics** — EOF coefficients, feedback parameters, spectra, estimated sensitivities, etc.;
3. **research representations** — embeddings, curvature tensors, p-adic encodings, cohomology classes, Clifford multivectors, latent features.

A research representation may influence a physical model only through an explicit, versioned adapter whose scientific interpretation is tested. This prevents an attractive mathematical object from silently becoming physics.

## 5. Package/layout target

Exact names may change, but responsibilities should converge roughly toward:

```text
Climate/
  AGENTS.md
  docs/
  contracts/                 # language-neutral records/schemas
  data/                      # acquisition + preprocessing code, not large data
  physics/                   # reference physical kernels
  numerics/                  # grids, solvers, timestepping, remapping
  diagnostics/               # established diagnostic/statistical methods
  methods/
    geometric/
    information_geometry/
    ultrametric/
    sheaf/
    symmetry/
    scenario_logic/
  experiments/               # declarative experiment specs and fixtures
  validation/                # independent evaluators and benchmark suites
  adapters/                  # language/FFI and Commons-facing adapters
  cli/                       # stable human/automation entrypoints
  tests/
    unit/
    property/
    convergence/
    differential/
    integration/
    regression/
  fixtures/                  # tiny immutable test datasets
```

Do not perform a bulk move into this shape until imports/builds/tests can protect the migration. The current stale `CORE/` references demonstrate why layout should be changed under executable checks.

## 6. Language policy

Language diversity is acceptable when each language has a clear reason to exist.

- **Fortran/C++/CUDA**: performance-sensitive physical/numerical kernels where mature ecosystems or GPU execution justify them.
- **Rust**: safe orchestration adapters, typed scientific infrastructure, selected numerical/research implementations, FFI boundaries.
- **Python**: data access, experiment assembly, analysis, statistics, plotting, interoperability.
- **Julia**: numerical/statistical experiments where Julia's ecosystem materially helps.
- **Haskell**: only where strong algebraic/type abstractions are part of the research question; not as a mandatory runtime dependency for unrelated Climate workflows.

No language should become authoritative merely because an early prototype was written there.

## 7. Interface principles

- Prefer explicit schema/version fields to inferred semantics.
- Units are part of types/contracts at important boundaries.
- Grids/calendars/time bases are part of dataset identity.
- Missingness is data, not a silent default.
- Randomness is seeded and recorded.
- Approximation level is explicit.
- An experiment consumes immutable references and emits immutable artifacts/evidence.
- Domain-specific code never invents its own run identity when Commons supplies one.
- Cross-language FFI stays narrow; complex climate arrays should use documented memory/layout contracts rather than ad-hoc pointer conventions.

## 8. What remains deliberately unresolved

These questions require experiments rather than architecture fiat:

- whether a geometric metric should be physically constructed, statistically learned, or both;
- which geometric invariants, if any, predict regime transitions beyond standard baselines;
- whether an ultrametric/p-adic representation improves teleconnection prediction;
- whether Clifford representations add useful information over spectral/phase baselines;
- whether sheaf-theoretic obstruction measures improve data-quality/reconstruction tasks;
- which early-warning indicators are robust under nonstationary forcing/noise;
- which model complexity is warranted for each target question;
- whether a single coupled physical core should ultimately exist in this repository at all.

The architecture's job is to make these questions cheap to test and hard to misreport.

## 9. Definition of architectural success

The target architecture is doing its job when a new method can be added by:

1. declaring a method contract;
2. selecting immutable datasets;
3. declaring baselines/falsifiers/metrics in an experiment spec;
4. running through common orchestration;
5. emitting reproducible artifacts and evidence;
6. being compared without custom result semantics;
7. remaining unable to overstate its scientific maturity merely because it executed successfully.
