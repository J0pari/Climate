# Sheaf / cohomology realization ladder

Status: architectural and mathematical correction for `climate.sheaf.consistency`.

## Why this document exists

The legacy `climate_multiscale_sheaf.hs` uses sheaf/cohomology terminology for computations that do not yet satisfy the defining algebraic laws. The goal is not to discard the useful station-network intuition. The goal is to pin the intended mathematics, make every missing obligation explicit, and replace the misleading pieces incrementally with realizations that can be independently checked.

This follows a KanForge-style rule: incomplete intermediate structure is allowed; promotion is not. A stage may contain open obligations analogous to `sorry`, but no artifact may be described as realized cohomology, a Betti number, an adjunction, or a genuine sheaf gluing result until the relevant obligations are discharged by executable witnesses.

## Exact delta from the legacy implementation

### 1. The current `C0 -> C1` map is not a cochain coboundary

`coboundary0` computes an absolute, normalized discrepancy and multiplies it by a scalar overlap weight. Absolute value makes the map nonlinear. A cochain differential over a vector-space coefficient system must be linear. Therefore the current map cannot serve as the differential of the claimed cochain complex.

### 2. The current `C1 -> C2` map is not defined on a demonstrated nerve complex

`coboundary1` iterates over ordered station triples whether or not they form a 2-simplex of an actual cover nerve. Missing pair values are silently replaced by zero. This is a residual heuristic, not a coboundary on a declared simplicial complex.

### 3. `d^2 = 0` is not established

A genuine cochain complex requires `d_(k+1) o d_k = 0`. Because the first map is nonlinear and the second is not constructed from compatible incidence/restriction maps, the legacy code provides no such identity.

### 4. The reported "Betti numbers" are not Betti numbers

The legacy code counts edge residuals above `0.3` and triple residuals above `0.4`, with `b0 = 1` hard-coded. Betti numbers are dimensions of cohomology groups:

`b_k = dim ker(d_k) - dim im(d_(k-1))`.

Threshold counts may remain useful diagnostics, but they must be named threshold counts and must not inherit topological interpretation automatically.

### 5. The object called a sheaf has not satisfied the sheaf structure

The legacy record stores pairwise functions called `restrictions`, but the calculations do not use them to define the cochain maps, and there are no identity/composition witnesses. A realizable finite cellular/sheaf model needs explicit stalks and restriction maps satisfying the appropriate functorial laws on the chosen base complex/poset.

### 6. The current gluing routine is not sheaf gluing

`glueLocalSections` checks a heuristic discrepancy threshold and then applies `Map.unions`. This neither proves compatibility under restriction maps nor establishes existence and uniqueness of a global section. It should be treated as a merge heuristic until those laws exist.

### 7. The current "adjunction" is only an analogy

Two data records plus scalar retention/reconstruction scores do not define categories, functors, natural transformations, or the triangle identities. The terminology should remain quarantined from scientific evidence until those structures are stated and witnessed.

## Realization ladder

The order below is a dependency graph, not a demand to finish the entire theory before useful experiments begin.

### R0 — truthful heuristic surface

Retain useful station overlap, discrepancy, and merge heuristics, but name them as heuristics. No Betti/cohomology/adjunction claims are emitted from this layer.

### R1 — exact finite-complex algebra

Construct a finite abstract simplicial complex, exact cochain groups over a declared coefficient field, linear coboundary matrices, executable `d^2 = 0` checks, and rank-defined cohomology dimensions on known fixtures.

Current realization: `reference/sheaf_cohomology.py` implements the constant rank-one cellular sheaf over `GF(2)`. This is genuine but intentionally narrow. It establishes the algebraic kernel; it does **not** establish a climate-data sheaf.

Required fixtures include at least: interval, disconnected points, circle/triangle boundary, filled triangle, and a 2-sphere triangulation.

### R2 — actual station-cover nerve

Define station coverage sets or another scientifically defensible cover. Construct simplices only from non-empty intersections (or an explicitly justified approximation with error semantics). Coverage-radius choices become experiment inputs, not hidden constants. Compare the constructed nerve against simpler graph representations.

### R3 — nontrivial climate-data sheaf

Define stalk vector spaces and restriction maps for a concrete climate-data task. State units and missing-data semantics. Verify restriction identities/composition and construct the cellular/sheaf coboundary from those restrictions. Re-run `d^2 = 0` as a hard witness.

A scalar residual diagnostic may be derived from a cochain, but the residual itself is not automatically a cohomology class.

### R4 — global-section / obstruction semantics

Define exactly what compatible local sections and a global section mean for the chosen data model. Implement compatibility and gluing against the actual restriction maps. If an obstruction score is used, derive it from the sheaf model and distinguish exact obstruction/nonexistence from a normed approximate inconsistency.

### R5 — synthetic falsification

Use known-topology fixtures, injected station faults, withheld observations, disconnected coverage, and perturbations that should *not* create a topological signal. Compare against graph residuals, ordinary QC, interpolation/kriging, and non-topological learned baselines.

### R6 — climate-data evaluation

Only after R1-R5 are independently witnessed should the method be evaluated on declared real station networks. Any claim of incremental value is empirical and task-scoped; mathematical correctness alone cannot promote it.

## Promotion law

The machine-readable ledger at `methods/sheaf-realization.v1.json` is the source of truth for which obligations are open or realized. `architecture/check_sheaf_realization.py` pins that ledger to the claim statement and requires every realized obligation to cite executable witness files.

A downstream report may say, for example, "constant-sheaf cohomology reference realized; station-data sheaf open." It may not compress that into "sheaf cohomology implemented."

## What the first reference does and does not prove

The first canonical reference proves that the repository can compute genuine finite-complex cohomology for a constant rank-one sheaf over `GF(2)`, with exact arithmetic and known-topology witnesses. This is enough to retire the specific fiction that threshold counts are Betti numbers.

It does **not** prove that the station network is a valid cover nerve, that climate measurements form the chosen sheaf, that a nonzero cohomology class corresponds to a data defect, that a coverage gap is detected, or that the method adds value over standard QC. Those remain explicit open obligations rather than implicit promises.
