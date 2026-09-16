# Climate obligation roadmap

Status: **binding dependency and priority map**. This is not a serial phase plan.

Climate is a heterogeneous research system: formal mathematics, portable numerical kernels, legacy scientific prototypes, statistical methods, GPU code, external climate data, and Commons integration do not share one global maturity level. Work advances along independent obligation paths when prerequisites and execution resources are available.

Objective repository state is generated at [`docs/generated/STATUS.md`](generated/STATUS.md). Do not duplicate module counts, claim maturity, experiment lists, or realization-ledger counts here.

`docs/EXECUTION-TOPOLOGY.md` defines resource classes. `AGENTS.md` defines correctness and evidence rules. [`docs/PHYSICAL-INVARIANTS-REALIZATION.md`](PHYSICAL-INVARIANTS-REALIZATION.md) defines the physical-law verification program.

## 1. Work-selection rule

Represent work as an obligation graph. A node is ready when:

1. semantic prerequisites are satisfied;
2. required execution resources are available;
3. the result removes material uncertainty or creates a reusable correctness boundary;
4. it does not substitute a weaker witness for a missing stronger one;
5. it can be falsified, differentially checked, or fail closed.

Among ready nodes prefer work that:

- removes a false authority or plausible-output semantic hazard;
- creates an independent reference reused by several implementations;
- turns an important prototype into a canonical executable slice;
- exposes a physical source/sink/exchange/correction that was previously hidden;
- adds an adversarial witness that catches a plausible wrong implementation;
- creates a fair baseline or ablation for an unusual method;
- removes a monolith dependency by exposing scientifically meaningful seams;
- prepares a resource-gated experiment so scarce hardware/data answers one precise question.

## 2. Authority ladders

General research authority:

```text
mathematical/statistical definition
        ↓
formal or independently checkable reference
        ↓
canonical portable implementation
        ↓
optimized / accelerated implementation
        ↓
experiment result
        ↓
empirical validation
        ↓
interpretive or decision mapping
```

Physical-law authority:

```text
continuum model + assumptions
        ↓
independent derivation/reference
        ↓
semi-discrete operator law
        ↓
fully discrete time/solver behavior
        ↓
runtime budget closure
        ↓
adversarial verification
        ↓
benchmark / observational validation
```

Passing one layer never promotes the next automatically.

## 3. Active correctness frontiers

These are independent fronts rather than one global sequence.

### 3.1 Physical dynamics and invariant realization

The highest-leverage physical objective is to build a structure-preserving fluid laboratory before broadening the atmospheric core.

Near-term obligations:

- complete fixed-pressure-coordinate continuity and boundary semantics;
- build a barotropic-vorticity invariant laboratory with energy/enstrophy witnesses and a deliberately ordinary comparison operator;
- build rotating shallow water with explicit mass, energy, PV, potential-enstrophy, balanced-state, wave, forcing, and dissipation contracts;
- introduce runtime quantity/budget records when the first canonical fluid laboratory needs them;
- add compatible divergence/gradient/curl and mass-flux identities as executable tests;
- add correction-ledger channels for filtering, limiting, positivity repair, remapping, projections, and solver inexactness;
- apply the resulting operator discipline to pressure-gradient and momentum dynamics;
- then construct dry/moist total-energy, water, and angular-momentum budgets.

Every physical feature requires adversarial timestep, unit, orientation, boundary, solver, restart, and deliberately corrupted-operator tests appropriate to its law class.

### 3.2 Mathematical-name fidelity and exact authorities

Continue sweeping places where the mathematical name outruns realized structure. Prefer constructive implementation of the named object when it is useful and tractable.

Current high-leverage edges:

- derive and verify Amari α-connections from explicit likelihood/Fisher families;
- extend model-specific fluid references only when a concrete continuum model requires them;
- continue differential checks between symbolic geometry and canonical Rust geometry;
- formalize small immutable algebraic identities in Lean only when they reduce ambiguity in a stable statement.

### 3.3 Geometry correctness → candidate evaluation

The repository has independent symbolic geometry and a canonical CPU Levi-Civita kernel over explicit metric jets. Remaining obligations include:

- independent derivative-generation witnesses;
- coordinate reparameterization, scale/unit, and permutation metamorphics;
- conditioning/refusal behavior near singular and indefinite metrics;
- explicit candidate metric construction separated from generic tensor machinery;
- synthetic regime-indicator experiments against conventional early-warning/state-space baselines;
- CPU/GPU stage-by-stage differential verification on actual CUDA hardware.

No curvature-to-climate-event mapping is eligible merely because tensor calculations are correct.

### 3.4 Sheaf/descent/cohomology realization

The machine ledger `methods/sheaf-realization.v1.json` is authoritative for the exact frontier.

The next scientifically meaningful layer is a defensible climate-data sheaf:

- station/coverage rule → nerve;
- typed stalk contents with units/missingness;
- explicit restrictions;
- identity/composition witnesses;
- climate-data coboundary;
- compatibility/global-section semantics;
- injected-fault and withheld-data experiments;
- graph/residual/QC/interpolation baselines.

The global aggregation mechanism must demonstrate incremental value over baselines with the same restriction information.

### 3.5 Legacy physics decomposition

`climate_physics_core.f90` remains a failing compile probe and mixed-concern monolith. Repair strategy:

```text
preserve → slice → type seams → recompose → verify
```

Scientifically coherent extraction targets include:

- thermodynamics/radiation;
- transport/dynamics;
- boundary-layer/convection;
- microphysics;
- land/ocean/ice coupling;
- budgets/diagnostics.

Each extracted kernel receives units, state assumptions, analytic/manufactured witnesses, and applicable physical-law budgets before becoming canonical.

### 3.6 Data and provenance spine

Build toward:

- immutable `DatasetRef` projections;
- CF-aware units/coordinates/calendars;
- transformation DAGs;
- explicit QC/missingness/imputation;
- compact network-free fixtures;
- source and preprocessing digests in run/evidence records.

Prefer one narrow end-to-end observational/reanalysis fixture over a broad downloader that cannot reconstruct transformations.

### 3.7 Baselines and falsification experiments

Novel methods compete against strong alternatives with the same information access.

Priority comparison families include:

- geometry vs conventional early-warning/Jacobian/state-space indicators;
- sheaf spectral/global diagnostics vs same-restriction local residuals, graph methods, QC, and interpolation;
- p-adic/ultrametric encodings vs generic hierarchical, graph, spectral/coherence, Euclidean, learned, and randomized controls;
- Clifford representations vs complex spectra, Hilbert phase, wavelets, bispectra, DMD/Koopman;
- natural gradients vs standard optimizers and equally informed preconditioners.

Every experiment specifies positive controls, negative controls, ablations, primary metrics, uncertainty treatment, and retain/revise/reject criteria before results are inspected.

## 4. Resource-gated frontiers

### CUDA / accelerator (`R3`)

Do now in CPU/CI:

- immutable fixtures and reference outputs;
- explicit layouts/precision/determinism contracts;
- stage-level differential tolerances;
- candidate-batch isolation tests;
- resource-envelope hypotheses labeled as estimates.

Do only on a real CUDA device:

- kernel execution correctness;
- race/synchronization behavior;
- deterministic reductions;
- mixed-precision/tensor-core behavior;
- VRAM/transfer/occupancy/profiling;
- end-to-end performance claims.

### Integrated Commons (`R4`)

Prepare narrow repository-inspection, experiment, run, artifact, and evidence interfaces. Do not duplicate Commons scheduling or resource leasing inside Climate.

The next meaningful integrated milestone is one real read-only/sandboxed Climate experiment with run/trace/causation/fingerprint lineage preserved end to end.

### Large data (`R5`)

Real-data claims require immutable projections, protected confirmation splits, dependence-aware uncertainty, provenance/licensing, and retained negative/failed results. Synthetic fixtures verify implementations but do not substitute for this layer.

## 5. Documentation maintenance model

**Generated / drift-checked:** objective facts present in registries, including module lifecycle/maturity, claim maturity, experiment registration, and realization-ledger status.

Update the source authority and run:

```bash
python architecture/render_status.py --write
```

**Human-maintained:** priority judgments, scientific rationale, interpretation boundaries, architecture tradeoffs, physical-law assumptions, and resource strategy.

Prefer adding another generated field only when an authoritative machine source already exists.

## 6. Near-term high-leverage edges

Subject to failures discovered by CI, the current short list is:

1. land fixed-pressure-coordinate continuity and then use the resulting seam to define the pressure-gradient/momentum contract;
2. create the barotropic-vorticity invariant laboratory and use it to establish the adversarial operator-testing pattern;
3. begin rotating shallow water as the first complete mass/energy/PV/potential-enstrophy laboratory;
4. turn geometry reference/canonical agreement into derivative and coordinate metamorphic tests;
5. continue decomposition of the legacy Fortran physics monolith along durable physical interfaces;
6. progress climate-data sheaf semantics into explicit station cover/stalk/restriction behavior;
7. strengthen experiment baselines/ablations so unusual methods can be rejected cleanly when they add no value;
8. prepare GPU differential fixtures without consuming accelerator resources until hardware-specific execution is needed.

This list changes when evidence changes. Generated status changes mechanically from machine authorities.
