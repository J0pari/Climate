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

### 3.2 Multirepresentation climate geometry

The repository now has independent symbolic geometry, a canonical CPU Levi-Civita kernel over explicit metric jets, and canonical Fisher-information primitives for explicit Gaussian observation models. These should support a broader research question than selecting one hand-built climate metric.

The candidate unifying program is defined in `docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md`:

- construct multiple scientifically interpretable representation maps of the same physical climate state rather than one concatenated feature vector;
- evaluate whether their relationships are best described by common-manifold, product, fibered, quotient, local-atlas, or deliberately non-Riemannian structures;
- use information geometry where an explicit observation/likelihood model supplies a statistically grounded local metric;
- let dynamical, thermodynamic, subsystem, spectral/Koopman, observational, and uncertainty representations contribute according to their actual semantics;
- measure cross-representation coupling instead of declaring weakly correlated views to be physically independent;
- treat curvature, geodesics, holonomy, topology, and intrinsic dimension as diagnostics rather than universal optimization objectives;
- test whether candidate geometry preserves/reveals physical dynamics, identifiable information, balances, subsystem coupling, and useful predictive structure beyond simpler representations.

This also changes the interpretation of several experimental mathematics families. Sheaf structure may help with local data/chart consistency, ultrametric methods may contribute an alternative relational geometry or kernel, Clifford structure may represent local oriented/multicomponent dynamics, and Noether/Hamiltonian structure may constrain physical evolution without each becoming a coordinate axis of one universal manifold.

Remaining numerical obligations still include:

- independent derivative-generation witnesses (analytic/AD/finite difference/complex step where applicable);
- coordinate reparameterization, scale/unit, and permutation metamorphics on the canonical geometry implementation;
- conditioning/refusal behavior across near-singular and indefinite metrics;
- explicit candidate metric construction separated from generic tensor machinery;
- reference experiments for common/product/fiber geometry on systems with known latent structure;
- dynamical tests of candidate coordinates using known evolution laws before climate interpretation;
- CPU/GPU stage-by-stage differential verification once an actual CUDA device is available.

No curvature-to-tipping probability/timescale mapping is eligible merely because tensor calculations are correct, and reduced scalar curvature is not a sufficient criterion for accepting a representation or latent dimension.

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

- multirepresentation geometry vs raw concatenation, PCA/CCA, standard latent encoders, diffusion/common-manifold methods, and conventional state-space/Jacobian indicators;
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

1. define and run the first small multirepresentation reference experiment on a dynamical system with known latent factor/common structure, comparing concatenation, linear fusion, and established nonlinear common-manifold methods before inventing a custom learner;
2. connect the canonical Fisher-information primitive and explicit-likelihood Amari reference to a candidate observation-induced metric, keeping state, parameter, and uncertainty geometry distinct;
3. turn geometry reference/canonical agreement into derivative and coordinate metamorphic differential tests so candidate representation work rests on trustworthy geometry;
4. continue decomposing the legacy Fortran physics core toward energy/momentum/continuity-consistent dynamics whose known structure can constrain and test manifold coordinates;
5. progress the climate-data sheaf into explicit station cover/stalk/restriction semantics that can later participate in local chart/data-consistency experiments;
6. prepare GPU differential fixtures only after a candidate manifold computation is useful enough that acceleration removes a demonstrated experiment bottleneck.

This list is intentionally short and manually curated. It should change when evidence changes, while the generated status beneath it updates mechanically.
