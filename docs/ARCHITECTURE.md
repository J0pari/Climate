# Climate target architecture

Status: **architectural specification / migration target**.

Climate is a heterogeneous research system spanning physical dynamics, numerical methods, mathematical references, statistical inference, experimental representations, data pipelines, and validation. The architecture exists to keep those responsibilities separable so that execution, mathematical correctness, numerical fidelity, empirical validation, and scientific interpretation cannot be confused.

## 1. Architectural objectives

The repository should optimize for:

1. **scientific separability** — changing one hypothesis does not silently alter unrelated physics, data handling, or evaluation;
2. **physical accountability** — every prognostic update can be attributed to a governing operator, source/sink, exchange, solver effect, or numerical correction;
3. **evidence traceability** — important results can be reconstructed from immutable inputs, code, configuration, environment, and runtime identity;
4. **fair comparison** — experimental methods and conventional baselines enter the same experiment/evaluation machinery;
5. **independent verification** — mathematical definitions and physical laws have references separate from optimized implementations;
6. **control-plane compatibility** — Commons may schedule, identify, trace, and compare work without owning Climate's scientific semantics.

This repository should grow by realizing narrow, testable obligations rather than by expanding a monolithic Earth-system implementation.

## 2. Authority ladder

The stable authority ordering is:

```text
mathematical / physical definition
        ↓
independent reference or formal witness
        ↓
canonical portable implementation
        ↓
optimized / accelerated implementation
        ↓
experiment result
        ↓
empirical validation
        ↓
scientific interpretation
```

Passing one layer never promotes the next automatically.

For physical dynamics, the parallel law ladder is:

```text
continuum model + assumptions
        ↓
reference derivation
        ↓
semi-discrete operator identity
        ↓
fully discrete time/solver behavior
        ↓
runtime budget closure
        ↓
adversarial verification
        ↓
benchmark / observational validation
```

[`PHYSICAL-INVARIANTS-REALIZATION.md`](PHYSICAL-INVARIANTS-REALIZATION.md) defines this program in detail.

## 3. Stable language-neutral records

Exact serialization may evolve, but the conceptual records should remain small and explicit.

### `DatasetRef`

Identifies an immutable dataset or projection.

Required concepts:

```text
id / digest
source family and version/retrieval identity
field identities
spatial domain/grid
vertical coordinates
calendar/time domain
temporal resolution
units / CF standard names
quality-control policy
missingness policy
preprocessing graph
license/citation
artifact digests
```

### `MethodDescriptor`

Describes one executable scientific or numerical method.

```text
method_id
semantic_version
implementation_build
maturity
method_family
input_contracts
output_contracts
assumptions
known_limitations
resource_requirements
reference/baseline family
validation status by claim
```

### `PhysicalLawDescriptor`

Describes one invariant, balance law, entropy law, material law, Casimir, or other physical-law obligation.

```text
law_id
model_id
law_class
quantity + units
continuum statement
assumptions
boundary requirements
spatial discretization contract
integrator/solver contract
physical source/sink terms
exchange terms
numerical correction channels
reference witnesses
adversarial witnesses
acceptance tolerances
validation status
```

This record should be introduced when the first canonical fluid laboratory needs it rather than speculatively creating a large schema in advance.

### `ExperimentSpec`

Defines a falsifiable comparison.

```text
experiment_id
question / hypothesis
candidate method
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

### `RunManifest`

Records what actually happened.

```text
run_id
experiment_id
repository revision
implementation builds
contract fingerprints
resolved input digests
resolved configuration
random seeds
host/container/environment identity
hardware identity where numerically relevant
timestep/substep/solver identity where physically relevant
start/end times
exact commands / entrypoints
exit status
stdout/stderr digests
artifact refs
parent/causation ids
```

### `ArtifactRef`

Large arrays, checkpoints, NetCDF/Zarr outputs, logs, figures, and reports live outside event envelopes and are referenced by digest plus media/schema metadata.

### `EvidenceRecord`

Links a claim to tests, simulations, observations, or other witnesses.

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

A metric carries semantics rather than only a scalar.

```text
metric_id + version
value
units
aggregation domain
sampling distribution / interval
reference population/dataset
higher/lower/is-target semantics
missing-data policy
uncertainty method
```

### `BudgetRecord`

Physical kernels should eventually emit a compact ledger record for important quantities.

```text
quantity
units
state_delta
boundary_flux
physical_sources
physical_sinks
internal_exchange
coupling_exchange
numerical_correction
solver_effect
roundoff_estimate
unexplained_residual
process identities
time interval
spatial domain
```

## 4. Layered system shape

### Layer A — acquisition and normalization

Responsibilities:

- retrieve and identify source data;
- preserve source/version/license identity;
- decode calendars, units, masks, quality flags, vertical coordinates, and grids;
- make every transformation explicit;
- produce immutable analysis-ready projections.

Download/cache success is not data validity. Schema, units, coverage, missingness, and source-specific QC are separate gates.

### Layer B — physical and numerical kernels

This layer owns equations and discretizations intended to model climate-system dynamics.

Examples:

- mass and momentum dynamics;
- pressure-coordinate and geometric vertical motion;
- thermodynamics and moisture;
- conservative transport;
- diffusion and dissipation;
- radiation;
- ocean/land/ice component kernels;
- coupling/remapping;
- spectral transforms;
- time integration and implicit solves.

Kernels should expose narrow typed contracts rather than depend on one giant mutable climate-state object.

Every physical kernel declares:

- units and state assumptions;
- grid/coordinate requirements;
- supported boundary conditions;
- stability/CFL assumptions;
- intended conservation/balance properties;
- source/sink/exchange semantics;
- numerical corrections;
- failure conditions.

### Layer C — physical-law and mathematical references

This layer contains independent authorities used to check definitions and derived identities.

Current and planned families include:

- differential geometry;
- information geometry;
- finite-dimensional Lie algebra and variational mechanics;
- finite-complex/cellular-sheaf mathematics;
- barotropic-vorticity invariant references;
- rotating shallow-water Hamiltonian/PV/energy references;
- compatible discrete operator identities;
- manufactured solutions and analytic benchmark fixtures.

A reference has no automatic climate interpretation beyond the model it explicitly represents.

### Layer D — established diagnostics and inference

Examples include:

- climatologies/anomalies;
- spectra, cross-spectra, coherence, wavelets;
- EOF/PCA and related modal decompositions;
- feedback regression/decomposition;
- energy and constituent budgets;
- ensemble statistics;
- likelihood/Bayesian parameter estimation;
- Fisher information/identifiability for explicit statistical models;
- standard early-warning statistics;
- model-performance diagnostics.

Established method families still require local implementation verification.

### Layer E — experimental mathematical representations

This is a first-class research layer with explicit separation from physical state.

Candidate families include:

- Riemannian/geometric state-space methods;
- information-geometric optimization and representation;
- p-adic/ultrametric teleconnection representations;
- Clifford/geometric-algebra representations;
- sheaf/cohomological consistency methods;
- modal-logic scenario constraints;
- learned latent manifolds and representations.

Every experimental method uses the same `MethodDescriptor -> ExperimentSpec -> RunManifest -> EvidenceRecord` path as conventional baselines.

### Layer F — validation and benchmarking

Responsibilities include:

- analytic/manufactured verification;
- convergence studies;
- conservation/balance closure;
- differential and cross-language tests;
- metamorphic coordinate/unit/orientation tests;
- deliberately corrupted negative controls;
- benchmark suites;
- restart/replay tests;
- long-time drift tests;
- observational/reanalysis comparison;
- out-of-sample validation;
- robustness/sensitivity analysis;
- uncertainty/calibration analysis.

### Layer G — orchestration and provenance

Climate exposes scientific work but does not become a duplicate distributed control plane.

Local orchestration may own:

- one experiment process;
- intra-process parallelism;
- GPU kernels;
- model component coupling;
- local checkpoint/restart.

Workspace-level scheduling, cross-repository dependencies, global resource leases, run identity, and cross-repository lineage should align with Commons.

## 5. Physical state versus derived and research state

Use three categories:

1. **prognostic/diagnostic physical state** — quantities with explicit units, grids, equations, and physical meaning;
2. **derived scientific diagnostics** — budgets, spectra, EOF coefficients, feedback parameters, sensitivities, etc.;
3. **research representations** — embeddings, curvature tensors, ultrametric encodings, cohomology classes, Clifford multivectors, latent features.

A research representation may influence physical evolution only through an explicit, versioned adapter with separately tested scientific semantics.

## 6. Package/layout target

Exact names may evolve, but responsibilities should converge toward:

```text
Climate/
  AGENTS.md
  docs/
  contracts/                 # language-neutral records/schemas
  data/                      # acquisition + preprocessing code
  physics/                   # physical kernels and model composition
  numerics/                  # grids, operators, solvers, timestepping, remapping
  diagnostics/               # established diagnostics/statistics
  reference/                 # independent mathematical/physical authorities
  methods/
    geometric/
    information_geometry/
    ultrametric/
    sheaf/
    algebraic/
    scenario_logic/
    latent/
  experiments/               # declarative experiment specs and fixtures
  validation/                # independent evaluators and benchmarks
  adapters/                  # language/FFI and Commons-facing adapters
  cli/                       # stable entrypoints
  tests/
    unit/
    property/
    convergence/
    differential/
    adversarial/
    integration/
    regression/
  fixtures/                  # tiny immutable test datasets
```

Bulk movement into this shape should occur only under executable import/build/test protection.

## 7. Language policy

Language diversity is acceptable when each language has a specific role.

- **Fortran/C++/CUDA** — performance-sensitive physical/numerical kernels where mature ecosystems or accelerator execution justify them.
- **Rust** — typed scientific infrastructure, selected numerical/research implementations, safe orchestration adapters, FFI boundaries.
- **Python** — data access, experiment assembly, independent symbolic/reference work, analysis, statistics, plotting, interoperability.
- **Julia** — numerical/statistical experiments where its ecosystem materially helps.
- **Haskell** — only when algebraic/type abstractions themselves are part of the research problem.

No language or historical implementation becomes authoritative merely through age or breadth.

## 8. Interface principles

- Units are part of important contracts.
- Grids, calendars, coordinates, and orientation are part of state/data identity.
- Missingness is explicit data.
- Randomness is seeded and recorded.
- Approximation level is explicit.
- Timestep, substep, solver tolerance, and splitting policy are part of physical implementation identity.
- Numerical corrections have named budget channels.
- Experiments consume immutable references and emit immutable artifacts/evidence.
- Domain code does not invent a separate global run identity when Commons supplies one.
- Cross-language FFI remains narrow and layout/version contracts are explicit.

## 9. Definition of architectural success

The architecture is doing its job when a new physical or mathematical method can be added by:

1. declaring its model/method contract;
2. identifying an independent definition or reference;
3. selecting immutable fixtures/data;
4. defining baselines, adversarial cases, and metrics;
5. running through common execution/provenance machinery;
6. emitting reproducible artifacts, budgets, and evidence;
7. exposing failure rather than returning plausible output outside its contract;
8. remaining unable to overstate scientific maturity merely because it executed successfully.
