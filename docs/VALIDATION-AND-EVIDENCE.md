# Validation, verification, and evidence architecture

Status: **binding scientific-quality specification**.

The repository must distinguish several failure classes that are easy to conflate:

- code that does not run;
- code that runs but implements the wrong equations;
- numerically unstable or non-convergent algorithms;
- physically inconsistent parameterizations;
- statistically invalid inference;
- data leakage or provenance loss;
- an indicator that correlates in-sample but fails out of sample;
- a mathematically valid quantity with an unjustified climate interpretation;
- a simulation result mislabeled as observation;
- a placeholder or heuristic presented as a validated result;
- a valid result generalized beyond its tested domain.

The quality system must distinguish them rather than compressing them into one green/red test status.

## 1. Five verification/validation planes

### Plane 1 — structural/software integrity

Questions:

- Does the declared source/build layout exist?
- Do manifests and dependency graphs resolve?
- Are generated interfaces synchronized?
- Are required entrypoints present?
- Do serialization round-trips preserve meaning?
- Are schema versions explicit and supported?
- Are placeholders/stubs discoverable?

Examples of checks:

- manifest/path verifier;
- compile/type checking;
- schema validation;
- static checks for unimplemented branches in decision-eligible paths;
- dependency lock/fingerprint checks;
- serialization compatibility fixtures.

Passing this plane earns at most `runnable`.

### Plane 2 — numerical verification

Question: **Did we solve/compute the declared mathematical problem correctly?**

Appropriate techniques depend on the method:

- exact identities and analytic solutions;
- method of manufactured solutions for PDE kernels;
- grid/time-step refinement and observed order of convergence;
- invariant/property testing;
- conservation/budget closure;
- symmetry tests where the discretization claims symmetry;
- dimension/unit checks;
- adjoint/automatic-differentiation gradient checks against finite differences or complex-step derivatives where appropriate;
- CPU/GPU and cross-language differential tests;
- reproducibility tests across thread counts/hardware within declared tolerances;
- conditioning/sensitivity diagnostics.

Every numerical check must record normalization and tolerance. Avoid naked checks like `abs(error) < 1e-6` without scale/meaning.

### Plane 3 — physical/process validation

Question: **Does the model/method reproduce the physical behavior it claims to represent?**

Examples:

- radiative forcing response against accepted benchmark formulas/ranges;
- energy/water/carbon budget behavior;
- idealized circulation/transport benchmarks;
- known response signs and limits;
- feedback decomposition benchmark cases;
- observed/reanalysis climatologies and variability patterns;
- process-oriented diagnostics.

A parameter value being "within an IPCC range" is not by itself validation of the model using it.

### Plane 4 — empirical predictive/generalization validation

Question: **Does the method improve a predeclared target on independent data or held-out regimes?**

Required ingredients:

- train/tune/test or discovery/confirmation separation;
- time-aware and spatially aware splits where dependence makes IID splitting invalid;
- baseline methods;
- uncertainty/confidence intervals;
- multiple-comparison controls when many methods/metrics are searched;
- sensitivity to preprocessing and dataset choice;
- explicit failure analysis.

Climate data are autocorrelated, spatially correlated, nonstationary, and often reused across products. Split design is part of the scientific method.

### Plane 5 — interpretation/decision validation

Question: **Is the mapping from the computed quantity to the stated scientific or operational interpretation justified?**

Examples requiring a separate mapping test:

- curvature -> tipping probability;
- Ricci eigenvalue index -> named tipping element;
- p-adic distance -> teleconnection strength;
- cohomological obstruction -> sensor fault;
- symmetry-breaking score -> regime transition;
- anomaly score -> physical instability;
- likelihood/model score -> decision authority.

This plane exists specifically to prevent category errors where impressive mathematics is assigned climate meaning by analogy alone.

## 2. Claim registry

`claims/registry.json` is the machine-readable authority for registered scientific/software claims. A claim record carries the scoped statement, claim type, subject, maturity, dependencies, required evidence, supporting/attacking evidence, unresolved falsifiers, and owner.

The registry does not make a claim true by registration. It makes the claim and its unmet obligations explicit enough for evidence tooling and generated status to track them without relying on prose.

Useful `claim_type` values include:

```text
software
numerical
physical
statistical
predictive
causal
interpretive
performance
resource
interoperability
```

Scientific prose that asserts or materially strengthens a registered claim should be consistent with the claim's recorded maturity and evidence. If the required evidence is absent, documentation must use hypothesis/prototype language rather than established-fact language.

## 3. Placeholder and heuristic controls

Important implementation paths and parameters must distinguish:

```text
placeholder      = intentionally incomplete implementation
fallback         = alternate behavior used when preferred data/method unavailable
heuristic        = deliberately approximate rule without derived/estimated guarantee
calibrated       = parameter estimated against named data/procedure
literature_fixed = value adopted from a cited source under a declared interpretation
learned          = fit by a recorded training/calibration procedure
```

Important parameters should carry provenance of this sort whenever their origin affects scientific interpretation or reproducibility.

CI and evidence tooling should be able to answer:

- Does a decision-eligible path contain a placeholder?
- Did a fallback activate during a validation run?
- Are default values standing in for missing observations?
- Did an allegedly calibrated result actually use the declared calibration artifact?

Fallback activation must be visible in run evidence.

## 4. Units, dimensions, coordinates, and calendars

A large class of climate bugs are semantically valid arrays with wrong units, grids, vertical coordinates, calendars, orientation, or anomaly baselines.

Boundary contracts should include:

- physical units;
- dimensionality and named dimensions;
- coordinate names and direction;
- calendar/time units;
- horizontal grid/mesh identity;
- vertical coordinate definition;
- masks/land-sea conventions;
- anomaly reference period;
- sign conventions;
- extensive vs intensive quantity semantics where remapping matters.

Where possible, use CF conventions rather than local synonyms. Validate that files claiming CF compliance actually pass appropriate checks.

## 5. Data provenance and contamination

Every derived dataset should form a transformation DAG:

```text
source artifact
  -> QC
  -> unit conversion
  -> temporal subset/aggregation
  -> regrid
  -> anomaly/detrend
  -> feature/diagnostic projection
  -> experiment input
```

Each edge should be versioned and fingerprinted.

The infrastructure must detect or make auditable:

- train/test overlap through shared source periods;
- duplicated observations in different derived products;
- leakage from future periods into normalization/detrending;
- use of reanalysis variables that assimilate target observations;
- tuning on the eventual evaluation dataset;
- silent source-version drift;
- changes in climatological baseline;
- data revisions that invalidate cached results.

## 6. Numerical regression strategy

Golden numeric arrays alone are brittle and can freeze bugs. Prefer a hierarchy:

1. invariant/property tests;
2. analytic/manufactured cases;
3. convergence behavior;
4. cross-implementation comparisons;
5. compact golden fixtures for stable interface behavior;
6. statistical tolerances for stochastic algorithms.

When exact reproducibility is not realistic, define the acceptable equivalence relation explicitly (absolute/relative error, ULPs, distributional test, ensemble statistic, conservation residual, etc.).

## 7. Scientific benchmark suites

Benchmark suites should be versioned data products, not loose scripts. A benchmark definition should include:

```text
benchmark_id
question
input dataset refs
preprocessing
methods/baselines
metrics
expected qualitative invariants
acceptance criteria if appropriate
known limitations
license/citation
```

Candidate benchmark families:

- forcing/energy-balance sanity;
- advection/diffusion transport;
- spectral/wave analysis;
- climate-index reconstruction;
- feedback estimation;
- parameter identifiability;
- tipping/early-warning synthetic systems;
- teleconnection prediction;
- station/data consistency;
- ensemble calibration;
- regime-shift detection.

## 8. Early-warning/tipping benchmark design

This deserves its own benchmark family because false positives are easy.

Include:

- canonical bifurcation systems with known transition points;
- red-noise/null processes;
- smoothly varying non-tipping systems;
- abrupt exogenous shifts without critical slowing down;
- changing noise variance/autocorrelation;
- missing/irregular observations;
- finite-window sensitivity;
- selected model and observational cases.

A new indicator should report discrimination/calibration against these controls, not only examples where the indicator looks visually compelling.

## 9. Experimental geometry validation

### State/manifold geometry

Before interpreting curvature:

- verify metric positive definiteness or deliberately document pseudo-Riemannian cases;
- check units/scaling and coordinate-reparameterization effects;
- verify Christoffel/Riemann calculations on manifolds with known curvature;
- test automatic derivatives independently;
- check sensitivity to metric construction and normalization;
- compare against learned metric/diffusion-map/kernel/PCA baselines;
- evaluate held-out predictive utility.

No curvature value is a tipping probability unless a calibrated mapping experiment establishes that relation.

### Information geometry

Require:

- an explicit likelihood/probabilistic model;
- identifiability/rank/conditioning diagnostics;
- Fisher computation checks against score/Hessian identities where assumptions permit;
- synthetic parameter-recovery experiments;
- comparison to standard optimizers/preconditioners;
- held-out likelihood/predictive checks.

### Ultrametric/p-adic methods

Require:

- hierarchy construction learned or predeclared without target leakage;
- Euclidean/correlation/spectral/graph/hierarchical baselines;
- held-out teleconnection prediction or retrieval target;
- sensitivity to prime/encoding choices;
- permutation/random-hierarchy negative controls;
- comparison to generic ultrametrics so any gain can be attributed specifically.

### Sheaf/topological methods

Do not label thresholded discrepancy counts as Betti numbers. Actual topological claims require a defined complex, boundary/coboundary maps over an appropriate coefficient structure, and rank/kernel/image calculations.

Validation targets can include injected sensor faults, coverage holes, withheld reconstruction, and known metadata discontinuities.

### Symmetry/Clifford/modal-logic methods

Each should have a conventional representation baseline and a task where the proposed structure could add measurable value. Algebraic elegance alone is not the evaluation metric.

## 10. Statistical safeguards

Experiments should declare when relevant:

- primary endpoint before running;
- family of hypotheses;
- correction/control for multiplicity;
- confidence/credible interval procedure;
- block/bootstrap method respecting dependence;
- effective sample-size assumptions;
- stopping rule;
- treatment of failed/invalid runs;
- robustness analysis across preprocessing and datasets.

Exploratory analysis is allowed, but must be labeled exploratory; confirmatory evidence requires a fresh or appropriately held-out evaluation.

## 11. Reproducibility levels

Execution-resource classes use the `R0_static` … `R5_large_data` namespace in `docs/EXECUTION-TOPOLOGY.md`. Reproducibility therefore uses a separate `REP*` namespace:

- **REP0 described** — prose sufficient to understand intent.
- **REP1 replayable** — exact command/config/seed/revision/data refs recorded.
- **REP2 artifact-reproducible** — same environment can recreate outputs within declared equivalence.
- **REP3 environment-reproducible** — environment/container/lockfiles allow recreation elsewhere.
- **REP4 independently replicated** — independent path reproduces the scientific result.

Do not use "reproducible" without indicating which sense when the distinction matters.

## 12. CI versus scientific evaluation

CI is for bounded, fast evidence:

- structural checks;
- tiny fixtures;
- unit/property tests;
- selected manufactured/convergence smoke tests;
- schema/provenance checks;
- deterministic linting of claim/maturity labels.

Large observational/model evaluations belong in scheduled benchmark jobs with immutable receipts. They can gate promotion without running on every commit.

## 13. Failure reporting

A failed experiment is evidence and should be retained. Report at least:

```text
failure_class
stage
expected condition
observed condition
whether outputs are scientifically usable
whether retry is appropriate
whether failure attacks a claim or only infrastructure
```

Do not silently drop numerical failures or only retain successful seeds/members.

## 14. Definition of validated

A claim is `validated` only when:

1. its scope is explicit;
2. the implementation is numerically verified enough for that claim;
3. the evaluation protocol was declared and versioned;
4. input provenance is complete;
5. appropriate baselines/controls were run;
6. uncertainty and dependence were treated appropriately;
7. independent held-out evidence supports the claim;
8. known failures and counterevidence are retained;
9. the evidence record points to immutable run/artifact identities.

Validation never means universally true. It means supported for the stated domain under the stated protocol.