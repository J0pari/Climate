# Sheaf / cohomology realization ladder

Status: architectural and mathematical specification for `climate.sheaf.consistency`.

## Mathematical contract

Sheaf/cohomology terminology is reserved for structures that satisfy the defining algebraic laws. Incomplete intermediate structure is allowed, but promotion is not: no artifact may be described as realized cohomology, a Betti number, an adjunction, or a genuine sheaf gluing result until the relevant obligations are discharged by executable witnesses.

The minimum mathematical requirements are:

1. **Linear cochain maps.** For vector-space coefficients, each coboundary must be a linear map between declared cochain groups.
2. **Declared base complex.** Cochains and coboundaries are defined on an explicit simplicial/cellular complex or poset. Higher simplices may not be invented by iterating arbitrary tuples or by silently filling missing faces with zeros.
3. **Cochain-complex identity.** Consecutive coboundaries must satisfy `d_(k+1) o d_k = 0` under the declared coefficient system.
4. **Cohomology-derived Betti numbers.** When the term Betti number is used, it means the dimension of the corresponding cohomology group, e.g. `b_k = dim ker(d_k) - dim im(d_(k-1))` over the declared field. Thresholded residual counts are diagnostics, not Betti numbers.
5. **Explicit stalks and restrictions.** A climate-data sheaf requires declared stalk contents, units, missing-data semantics, and restriction maps satisfying identity/composition laws on the chosen base.
6. **Restriction-derived coboundary.** The data sheaf cochain differential must be constructed from those restriction maps rather than from an unrelated discrepancy heuristic.
7. **Gluing semantics.** Compatibility of local sections and existence/uniqueness of a global section must be defined through the restrictions. Approximate merging or averaging is a separate reconstruction rule, not sheaf gluing by default.
8. **Adjunction terminology.** An adjunction requires actual categories, functors, unit/counit natural transformations, and triangle identities. Scalar reconstruction or retention scores do not by themselves establish one.

## Realization ladder

The order below is a dependency graph, not a demand to finish the entire theory before useful experiments begin.

### R0 — truthful local diagnostics

Station overlap, discrepancy, residual, and reconstruction diagnostics may be used when they are named according to the quantities they actually compute. Topological or categorical interpretation is not inferred from a diagnostic label.

### R1 — exact finite-complex algebra

Construct a finite abstract simplicial complex, exact cochain groups over a declared coefficient field, linear coboundary matrices, executable `d^2 = 0` checks, and rank-defined cohomology dimensions on known fixtures.

`reference/sheaf_cohomology.py` provides an exact constant rank-one cellular-sheaf reference over `GF(2)`. Its scope is deliberately narrow: it establishes the algebraic kernel, not a climate-data sheaf.

Required fixtures include at least: interval, disconnected points, circle/triangle boundary, filled triangle, and a 2-sphere triangulation.

### R2 — station-cover nerve

Define station coverage sets or another scientifically defensible cover. Construct simplices only from non-empty intersections, or from an explicitly justified approximation with error semantics. Coverage-radius choices are experiment inputs, not hidden constants. Compare the constructed nerve against simpler graph representations.

### R3 — climate-data sheaf

Define stalk vector spaces and restriction maps for a concrete climate-data task. State units and missing-data semantics. Verify restriction identities/composition and construct the cellular/sheaf coboundary from those restrictions. Re-run `d^2 = 0` as a hard witness.

A scalar residual diagnostic may be derived from a cochain, but the residual itself is not automatically a cohomology class.

### R4 — global-section / obstruction semantics

Define exactly what compatible local sections and a global section mean for the chosen data model. Implement compatibility and gluing against the actual restriction maps. If an obstruction score is used, derive it from the sheaf model and distinguish exact obstruction/nonexistence from a normed approximate inconsistency.

### R5 — synthetic falsification

Use known-topology fixtures, injected station faults, withheld observations, disconnected coverage, and perturbations that should *not* create a topological signal. Compare against graph residuals, ordinary QC, interpolation/kriging, and non-topological learned baselines.

### R6 — climate-data evaluation

After R1-R5 are independently witnessed, evaluate the method on declared real station networks. Any claim of incremental value is empirical and task-scoped; mathematical correctness alone cannot promote it.

## Promotion law

The machine-readable ledger at `methods/sheaf-realization.v1.json` is the source of truth for which obligations are open or realized. `architecture/check_sheaf_realization.py` pins that ledger to the claim statement and requires every realized obligation to cite executable witness files.

A downstream report must state the realized scope precisely. For example, exact constant-sheaf cohomology and a station-data sheaf are different claims and may mature independently.

## Reference scope

The exact finite-complex reference establishes genuine cohomology for a constant rank-one sheaf over `GF(2)`, with exact arithmetic and known-topology witnesses.

That reference does not establish that a station network is a valid cover nerve, that climate measurements form a particular sheaf, that a nonzero cohomology class corresponds to a data defect, that a coverage gap is detected, or that the method adds value over standard QC. Those are separate scientific and empirical obligations.
