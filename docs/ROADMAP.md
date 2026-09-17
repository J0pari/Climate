# Climate obligation roadmap

Status: **binding dependency and priority map**. This is not a serial phase plan.

Climate is a heterogeneous research system: formal mathematics, portable numerical kernels, experimental scientific methods, statistical methods, GPU code, external climate data, and Commons integration do not share one global maturity level. Work should advance along independent obligation paths whenever their prerequisites and required resources are available.

Objective repository state is generated from the live registries at [`docs/generated/STATUS.md`](generated/STATUS.md). Do not duplicate module counts, claim maturity tables, experiment lists, or sheaf-obligation counts here. `python architecture/render_status.py --check` makes that projection a drift-checked CI surface.

`docs/EXECUTION-TOPOLOGY.md` defines resource classes. `AGENTS.md` defines binding correctness and evidence rules. Machine-readable authorities and executable checks are authoritative for the objective present-state facts they own; architectural intent, scientific interpretation, and priority remain curated in their designated documents. A mismatch across those scopes is a defect to reconcile, not permission for one class of authority to silently overwrite another.

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

- the explicit normal-family Fisher/Amari reference now realizes α-connections, duality, and cubic-tensor structure; remaining work is differential migration of legacy information-geometry implementations only where those implementations still earn a role;
- use the exact Lie-bracket and Noether references to separate infinitesimal/variational structure from useful but different black-box diagnostics;
- extend symbolic/canonical geometry agreement from the realized constant-linear coordinate and finite-difference jet witnesses to nonlinear coordinate changes and additional independent derivative backends;
- formalize additional small immutable algebraic invariants in Lean only when the statement is stable and the proof reduces ambiguity.

A weaker abstraction is retained only when it is independently the better tool for a real task, not because it is easier to implement.

### 3.2 Multirepresentation climate geometry

Independent symbolic geometry, a canonical CPU Levi-Civita kernel over explicit metric jets, and canonical Fisher-information primitives for explicit Gaussian observation models provide foundations for a broader research question than selecting one hand-built climate metric.

The candidate unifying program is defined in `docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md`:

- construct multiple scientifically interpretable representation maps of the same physical climate state rather than one concatenated feature vector;
- evaluate whether their relationships are best described by common-manifold, product, fibered, quotient, local-atlas, or deliberately non-Riemannian structures;
- use information geometry where an explicit observation/likelihood model supplies a statistically grounded local metric;
- let dynamical, thermodynamic, subsystem, spectral/Koopman, observational, and uncertainty representations contribute according to their actual semantics;
- measure cross-representation coupling instead of declaring weakly correlated views to be physically independent;
- treat curvature, geodesics, holonomy, topology, and intrinsic dimension as diagnostics rather than universal optimization objectives;
- test whether candidate geometry preserves/reveals physical dynamics, identifiable information, balances, subsystem coupling, and useful predictive structure beyond simpler representations.

Within this program, sheaf structure may help with local data/chart consistency, ultrametric methods may contribute an alternative relational geometry or kernel, Clifford structure may represent local oriented/multicomponent dynamics, and Noether/Hamiltonian structure may constrain physical evolution without each becoming a coordinate axis of one universal manifold.

Realized structural controls now include:

- coordinate-covariant Fisher squared length for the two-layer EBM across temperature-state, heat-flux, exact thermal-mode, and unit-rescaled charts, with raw Euclidean distance retained as an explicit representation/unit-dependent negative control;
- exact separation between one-coordinate unaugmented DMD Markov closure and dynamical observability of the same known two-state system;
- equal-cost synthetic Pareto observation design over the declared two-layer EBM channels, separating cost authority from Fisher rank and fast/slow modal information rather than selecting an arbitrary weighted winner;
- constant-linear coordinate metamorphics for the canonical Levi-Civita kernel, including permutation and anisotropic scaling, plus an independent finite-difference metric-jet witness on a known conformal metric;
- `MetricJet` rejection of derivative data that violate metric-index symmetry or mixed-partial commutation at scale-derived floating-point roundoff.

Numerical obligations still include:

- nonlinear coordinate-change witnesses, including the inhomogeneous Christoffel transformation term;
- additional independent derivative-generation witnesses such as AD or complex-step where the metric construction supports them;
- conditioning/refusal behavior across near-singular metrics and explicit separation between invertibility, positive-definiteness, and scientific metric admissibility;
- explicit candidate metric construction separated from generic tensor machinery;
- reference experiments for common/product/fiber geometry on systems with known latent structure;
- dynamical tests of candidate coordinates using known evolution laws before climate interpretation;
- CPU/GPU stage-by-stage differential verification once an actual CUDA device is available and an accelerated geometry workload has been justified.

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

### 3.4 Physical-kernel decomposition and coupled recomposition

The atmospheric physical core should be composed from scientifically coherent kernels with explicit interfaces rather than one mixed-concern state transition.

Required responsibility boundaries include:

- thermodynamics/radiation;
- transport/dynamics;
- boundary-layer/convection;
- microphysics;
- land/ocean/ice coupling;
- budgets/diagnostics.

Each kernel should carry explicit units, state assumptions, conservation/exchange semantics, and analytic/manufactured/budget witnesses before becoming canonical. Recomposition must preserve the physical transfers between kernels rather than validating each piece in isolation and assuming the coupled system closes automatically.

The most important coupled obligations are continuity-consistent mass fluxes, pressure-gradient/pressure-work coupling, kinetic/internal/potential/latent energy exchange, conservative tracer transport, physically signed dissipation, and explicit boundary/surface/radiative exchanges.

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

Do not select an accelerator subsystem merely because a method is mathematically elaborate. First establish a scientifically useful workload and a demonstrated computational bottleneck.

Once an accelerated workload is justified, prepare on CPU/CI:

- immutable fixtures and reference outputs;
- explicit layouts/precision/determinism contracts;
- stage-level differential tolerances;
- isolation tests for batching or shared workspaces where those semantics exist;
- resource-envelope hypotheses clearly labeled as estimates.

Do only on a real CUDA device:

- kernel execution correctness;
- race/synchronization behavior;
- deterministic reductions;
- mixed-precision/tensor-core behavior;
- VRAM/transfer/occupancy/profiling;
- end-to-end performance claims.

### Integrated Commons (`R4`)

Climate/Commons integration is experimental and observe-first. Prepare narrow repository-inspection, experiment, run, artifact, and evidence interfaces. Do not duplicate Commons scheduling or resource leasing inside Climate.

The next meaningful integrated milestone is one real read-only/sandboxed Climate experiment with run/trace/causation/fingerprint lineage preserved end to end.

### Large data (`R5`)

Real-data claims require immutable projections, protected confirmation splits, dependence-aware uncertainty, provenance/licensing, and retained negative/failed results. Synthetic fixtures may verify implementations but cannot substitute for this layer.

## 5. Documentation maintenance model

Documentation has two deliberately different maintenance modes.

**Generated / drift-checked:** objective facts already present in registries, such as module lifecycle/maturity, claim maturity, experiment registration, and realization-ledger status. Update the source authority, run:

```bash
python architecture/render_status.py --write
```

and let CI enforce that the generated projection matches. On mismatch the checker emits a unified diff so projection drift is diagnosable rather than a generic stale-file failure.

**Human-maintained:** priority judgments, scientific rationale, interpretation boundaries, architecture tradeoffs, and resource strategy. These should be updated after meaningful changes in capability or evidence; attempting to infer them automatically would hide judgment rather than remove maintenance.

Prefer adding another generated field only when there is an existing authoritative machine source. Do not create a registry merely so prose can be generated.

## 6. Near-term high-leverage edges

Subject to failures discovered by CI, the current high-leverage set is:

1. extend geometry differential verification to nonlinear coordinate changes and a second independent derivative-generation route, while keeping derivative-estimation acceptance policy outside the mathematical `MetricJet` contract;
2. define the next coupled atmospheric dynamics slice around pressure-gradient/pressure-work, kinetic/internal/potential-energy exchange, and continuity-consistent interfaces rather than adding another isolated tendency kernel;
3. progress the climate-data sheaf into explicit station cover/stalk/restriction semantics that can later participate in local chart/data-consistency experiments;
4. establish one narrow immutable observational/reanalysis data projection with CF-aware coordinates/units, explicit QC/missingness, transformation lineage, and digests before expanding the data surface;
5. move beyond the equal-cost synthetic observation-design control only when a real cost/noise/provenance authority exists; do not tune representation weights to downstream scores in its absence;
6. prepare GPU differential fixtures only after a candidate manifold computation is useful enough that acceleration removes a demonstrated experiment bottleneck.

Recently retired from this list because their structural controls now execute are coordinate-covariant two-layer EBM Fisher distance, the DMD-versus-observability semantic separation, and equal-cost Pareto observation design. These remain structural controls rather than empirical observing-system or climate-manifold validation.

This list is intentionally short and manually curated. It should change when evidence changes, while the generated status beneath it updates mechanically.
