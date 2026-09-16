# Climate obligation roadmap

Status: **binding dependency and priority map**. This is not a serial phase plan.

Climate is a heterogeneous research system: formal mathematics, portable numerical kernels, legacy scientific prototypes, statistical methods, GPU code, external climate data, and Commons integration do not share one global maturity level. Work should advance along independent obligation paths whenever their prerequisites and required resources are available.

Objective repository state is generated from the live registries at [`docs/generated/STATUS.md`](generated/STATUS.md). Do not duplicate module counts, claim maturity tables, experiment lists, or sheaf-obligation counts here. `python architecture/render_status.py --check` makes that projection a drift-checked CI surface.

`docs/EXECUTION-TOPOLOGY.md` defines resource classes. `AGENTS.md` defines binding correctness and evidence rules. If narrative planning conflicts with executable checks or machine-readable authorities, the executable/machine-readable state wins and this document must be updated.

## 1. Work-selection rule

Represent work as an obligation graph. A node is ready when:

1. its semantic prerequisites are satisfied;
2. the required execution resource is available;
3. the result can remove material uncertainty or create a reusable correctness boundary;
4. it does not substitute a weaker witness for a missing stronger one;
5. it can be falsified, differentially checked, or fail closed.

Among ready nodes prefer work that:

- removes a false authority or plausible-output semantic hazard;
- creates an independent reference/formal witness reused by several implementations;
- turns an important prototype into a canonical executable slice;
- creates a fair baseline or ablation required to evaluate an unusual method;
- removes a monolith/God-object dependency by exposing scientifically meaningful seams;
- prepares a resource-gated experiment so scarce hardware/data answers one precise question.

Do not add sophisticated method families merely because they are interesting. Every addition must earn its place through a falsifiable question, independent comparison, or architectural leverage.

## 2. Authority ladder

A recurring obligation is to keep these layers distinct:

```text
mathematical/statistical definition
        ↓
formal or independently checkable reference
        ↓
canonical portable implementation
        ↓
optimized / language-specific / accelerated implementation
        ↓
experiment result
        ↓
empirical validation
        ↓
interpretive or decision mapping
```

Passing one layer never promotes the next automatically.

Examples:

- Lean proving a Gram-matrix identity does not formally verify the Rust sheaf implementation as a whole;
- a correct Levi-Civita kernel does not establish a scientifically meaningful climate metric;
- an exact p-adic metric does not establish that a climate-to-p-adic encoding is useful;
- a correct Clifford algebra does not establish that Clifford representations improve oscillation analysis;
- a GPU differential match does not establish a climate tipping interpretation.

## 3. Active correctness frontiers

These are independent fronts, not a global sequence.

### 3.1 Mathematical-name fidelity and exact authorities

Continue sweeping places where the mathematical name outruns the realized structure. The default repair is constructive: if the stronger mathematical object is useful and tractable, implement it correctly rather than renaming downward.

Current high-leverage edges:

- derive and verify Amari α-connections from an explicit likelihood/Fisher family rather than dummy observations or regularized matrices;
- use the exact Lie-bracket and Noether references to separate infinitesimal/variational structure from useful but different black-box diagnostics;
- continue differential checks between symbolic geometry and canonical Rust geometry;
- formalize additional small immutable algebraic invariants in Lean only when the statement is stable and the proof reduces ambiguity.

A weaker abstraction is retained only when it is independently the better tool for a real task, not because it is easier to implement.

### 3.2 Geometry correctness → candidate evaluation

The repository now has independent symbolic geometry and a canonical CPU Levi-Civita kernel over explicit metric jets. Remaining obligations include:

- independent derivative-generation witnesses (analytic/AD/finite difference/complex step where applicable);
- coordinate reparameterization, scale/unit, and permutation metamorphics on the canonical implementation;
- conditioning/refusal behavior across near-singular and indefinite metrics;
- explicit candidate metric construction separated from generic tensor machinery;
- synthetic regime-indicator experiments against conventional early-warning/state-space baselines;
- CPU/GPU stage-by-stage differential verification once an actual CUDA device is available.

No curvature-to-tipping probability/timescale mapping is eligible merely because tensor calculations are correct.

### 3.3 Sheaf/descent/cohomology realization

The machine ledger `methods/sheaf-realization.v1.json` is authoritative for the exact open/realized frontier; the generated status document renders it automatically.

The next scientifically meaningful layer is not more abstract topology. It is a defensible climate-data sheaf:

- station/coverage rule → nerve;
- typed stalk contents with units/missingness;
- explicit restrictions;
- identity/composition witnesses;
- climate-data coboundary;
- compatibility/global-section semantics;
- injected-fault and withheld-data experiments;
- graph/residual/QC/interpolation baselines.

The real-valued sheaf Laplacian/singular spectrum remains a candidate global aggregation mechanism. It must demonstrate incremental value over the same-restriction local residual baseline; withholding restriction information from the baseline is not allowed.

### 3.4 Legacy physics decomposition

`climate_physics_core.f90` remains a failing compile probe and a major mixed-concern monolith. The repair strategy is **preserve → slice → type seams → recompose → verify**, not wholesale quarantine and not patch-until-green.

Extract scientifically coherent kernels when their interfaces can be made explicit, for example:

- thermodynamics/radiation;
- transport/dynamics;
- boundary-layer/convection;
- microphysics;
- land/ocean/ice coupling;
- budgets/diagnostics.

Each extracted kernel should gain units/state assumptions plus analytic/manufactured/budget witnesses before becoming canonical. Embedded demonstration programs and placeholder physical closures must not define the final package shape.

### 3.5 Data and provenance spine

The strongest mathematical methods remain scientifically weak without trustworthy data identity. Build toward:

- immutable `DatasetRef` projections;
- CF-aware units/coordinates/calendars;
- transformation DAGs;
- explicit QC/missingness/imputation;
- compact network-free fixtures;
- source and preprocessing digests in run/evidence records.

Prefer one narrow end-to-end observational/reanalysis fixture over a broad downloader that cannot reconstruct its transformations.

### 3.6 Baselines and falsification experiments

Novel methods must compete against strong alternatives with the same information access.

Priority comparison families include:

- geometry vs conventional early-warning/Jacobian/state-space indicators;
- sheaf spectral/global diagnostics vs same-restriction local residuals, graph methods, QC, and interpolation;
- p-adic/ultrametric encodings vs generic hierarchical, graph, spectral/coherence, Euclidean, and learned representations plus randomized controls;
- Clifford representations vs complex spectra, Hilbert phase, wavelets, bispectra, DMD/Koopman;
- natural gradients vs standard optimizers and equally informed preconditioners.

Every experiment needs positive controls, negative controls, ablations, a primary metric, uncertainty treatment, and retain/revise/reject criteria before results are inspected.

## 4. Resource-gated frontiers

### CUDA / accelerator (`R3`)

Do now in CPU/CI:

- immutable fixtures and reference outputs;
- explicit layouts/precision/determinism contracts;
- stage-level differential tolerances;
- candidate-batch isolation tests;
- resource-envelope hypotheses clearly labeled as estimates.

Do only on a real CUDA device:

- kernel execution correctness;
- race/synchronization behavior;
- deterministic reductions;
- mixed-precision/tensor-core behavior;
- VRAM/transfer/occupancy/profiling;
- end-to-end performance claims.

### Integrated Commons (`R4`)

Current Climate/Commons relationship remains experimental and observe-first. Prepare narrow repository-inspection, experiment, run, artifact, and evidence interfaces. Do not duplicate Commons scheduling or resource leasing inside Climate.

The next meaningful integrated milestone is one real read-only/sandboxed Climate experiment with run/trace/causation/fingerprint lineage preserved end to end.

### Large data (`R5`)

Real-data claims require immutable projections, protected confirmation splits, dependence-aware uncertainty, provenance/licensing, and retained negative/failed results. Synthetic fixtures may verify implementations but cannot substitute for this layer.

## 5. Documentation maintenance model

Documentation has two deliberately different maintenance modes.

**Generated / drift-checked:** objective facts already present in registries, such as module lifecycle/maturity, claim maturity, experiment registration, and realization-ledger status. Update the source authority, run:

```bash
python architecture/render_status.py --write
```

and let CI enforce that the generated projection matches.

**Human-maintained:** priority judgments, scientific rationale, interpretation boundaries, architecture tradeoffs, and resource strategy. These should be updated after meaningful changes in capability or evidence; attempting to infer them automatically would hide judgment rather than remove maintenance.

Prefer adding another generated field only when there is an existing authoritative machine source. Do not create a registry merely so prose can be generated.

## 6. Near-term high-leverage edges

Subject to failures discovered by CI, the current high-leverage set is:

1. finish the explicit-likelihood Amari α-connection reference and use it to pressure-test legacy Python/Julia information geometry;
2. turn geometry reference/canonical agreement into derivative and coordinate metamorphic differential tests;
3. decompose the legacy Fortran physics compile failure along durable physical interfaces rather than repairing the monolith in place;
4. progress the climate-data sheaf from generic mathematics into explicit station cover/stalk/restriction semantics;
5. strengthen experiment baselines/ablations so unusual methods can be rejected cleanly when they add no value;
6. prepare GPU differential fixtures without consuming GPU/Codespace resources until hardware-specific execution is actually needed.

This list is intentionally short and manually curated. It should change when evidence changes, while the generated status beneath it updates mechanically.
