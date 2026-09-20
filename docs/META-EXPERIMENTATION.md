# Meta-experimentation protocol

Status: **research-process specification**.

Climate contains many choices that cannot be settled correctly by architecture review alone: metric construction, numerical schemes, representations, calibration methods, early-warning indicators, data products, interpolation/regridding choices, and experimental mathematical frameworks. The repository therefore needs infrastructure for experiments **about the methods themselves**, not only experiments using those methods.

The goal is to make methodological choices testable, comparable, revisable, and difficult to overfit by narrative. This document defines experiment design; `architecture/planning_graph.json` owns cross-family planning priority and `docs/ROADMAP.md` is only its projection. Actual work selection must first reconcile exact-head planning, realization, evaluation/evidence, and execution state using `docs/REPOSITORY-STATE.md`.

## 1. Unit of research: a contestable method choice

Represent a methodological question as a set of alternatives under one evaluation contract.

Examples:

```text
Question: How should teleconnection similarity be represented?
Candidates:
  standardized Euclidean state distance
  correlation distance
  spectral/coherence distance
  graph embedding
  learned metric
  generic ultrametric
  p-adic encoding
```

```text
Question: How should state-space geometry be constructed?
Candidates:
  physically hand-specified metric
  covariance/Mahalanobis metric
  Fisher metric
  diffusion-map geometry
  learned contrastive metric
  hybrid physically constrained learned metric
```

```text
Question: How should multiple grounded climate representations be integrated?
Candidates:
  standardized raw concatenation
  linear multiview fusion / CCA-family baseline
  shared/common-manifold construction
  product geometry with explicit factor metrics
  state-dependent fibered geometry
  graph/kernel fusion preserving non-Euclidean views
  learned joint representation constrained by physical and information-geometric structure
```

The last question is not equivalent to choosing one winning representation. Different views may remain useful as factors, charts, fibers, observation maps, constraints, or comparison geometries. Experiments should therefore test both *which representation is useful* and *which relationship among representations is scientifically warranted*.

Architecture may define the interface and admissibility constraints; experiment evidence chooses among alternatives for a declared task/domain.

## 2. `ExperimentSpec` principles

A meta-experiment should predeclare:

- research question;
- candidate methods and versions;
- baseline(s), including a deliberately simple baseline;
- target task and domain;
- immutable datasets and split policy;
- hyperparameter/tuning budget per candidate;
- preprocessing shared across methods and method-specific transformations;
- primary endpoint;
- secondary endpoints;
- computational/resource budget;
- uncertainty estimation;
- negative controls;
- ablations;
- falsifiers;
- failure handling;
- stopping rule;
- retain/revise/reject criteria.

The protocol should prevent one method from receiving dramatically more tuning or privileged preprocessing without that difference being part of the stated question.

## 3. Discovery versus confirmation

Separate:

### Discovery

Allowed:

- exploratory plots;
- feature searches;
- trying alternative metrics;
- inspecting failures;
- modifying hypotheses;
- searching hyperparameters;
- selecting promising representations.

Discovery output is **Heuristic/Proposed**, even if impressive.

### Confirmation

Requires:

- frozen candidate implementation/config or bounded tuning protocol;
- fresh/held-out data or an explicitly protected evaluation regime;
- predeclared primary endpoints;
- retained failures;
- uncertainty analysis;
- no post-hoc endpoint substitution.

A discovery dataset should not quietly become confirmatory because acquiring a new climate dataset is inconvenient.

## 4. Split strategies for dependent climate data

Random row splitting is often wrong.

Support explicit strategies such as:

- forward-chaining temporal splits;
- blocked time splits;
- spatial holdout regions;
- leave-one-event/regime-out;
- leave-one-model-out for multimodel ensembles;
- leave-one-dataset/product-out;
- historical-versus-modern period tests;
- synthetic-to-observational transfer;
- model-to-observation transfer.

If training/tuning uses derived products sharing upstream observations with evaluation data, record that dependence.

## 5. Benchmark matrix rather than one leaderboard

Climate methods have different scientific purposes. Avoid a single scalar "best climate method" score.

Use a matrix of tasks such as:

```text
                    reconstruction  prediction  calibration  regime detection  robustness  cost
PCA/EOF
DMD
Fisher method
geometric metric
multirepresentation geometry
ultrametric
sheaf method
...
```

A candidate may be useful because it dominates on one scientifically meaningful axis without being globally superior.

## 6. Ablation design

Novel mathematical frameworks must identify where gains come from.

Examples:

### p-adic/ultrametric

Compare:

- full p-adic encoding;
- same discretization with random digit hierarchy;
- same hierarchy with generic tree distance;
- learned ultrametric without prime arithmetic;
- shuffled regional prime assignment;
- conventional metric baselines.

If all hierarchy-based variants work equally well, the evidence supports hierarchical structure, not specifically p-adic arithmetic.

### geometric climate state

Compare:

- full metric;
- diagonal-only metric;
- no physical off-diagonal terms;
- standardized Euclidean/Mahalanobis;
- learned metric;
- random positive-definite metric matched for scale.

If curvature signals disappear under small arbitrary rescalings, that is evidence against robust interpretation.

### multirepresentation climate state

Compare:

- the full set of declared representation maps;
- each representation alone;
- leave-one-representation-out variants;
- raw concatenation with matched total dimension;
- linear shared-subspace baselines;
- product geometry with cross terms removed;
- shuffled or mismatched cross-view correspondences;
- the same learner with physical/information-geometric constraints removed.

A gain that survives only because one high-capacity view dominates does not establish that the joint manifold structure is useful. Cross-representation terms should earn their place by improving dynamics, information preservation, physical consistency, or another declared scientific objective.

### sheaf consistency

Compare:

- actual cohomological obstruction;
- pairwise residual thresholds;
- graph-based anomaly detection;
- kriging residuals;
- metadata-aware QC baselines.

### Clifford resonance

Compare against complex-valued spectral/phase representations with matched degrees of freedom.

## 7. Negative controls

Every ambitious method family should have controls that should fail or lose structure.

Examples:

- time-shuffled series;
- spatially permuted fields preserving marginal distributions;
- phase-randomized surrogates;
- random hierarchy/tree;
- randomized labels/events;
- synthetic systems without the claimed phenomenon;
- physically impossible perturbations expected to trigger guards;
- dummy features with matched dimension/noise.

A method that scores highly on both real data and its negative control likely measures an artifact of the pipeline or evaluation.

## 8. Positive controls

Use synthetic/idealized systems where the target structure is deliberately present.

Examples:

- known bifurcation with critical slowing down;
- known hierarchical dependency graph;
- manifold with known curvature;
- multiview dynamical system with known shared, product, or fibered latent structure;
- station network with injected overlap inconsistency;
- oscillator system with controlled resonance;
- parameterized model with known Fisher information/identifiability.

Failure on positive controls blocks strong scientific interpretation even if observational plots look plausible.

## 9. Sequential experimentation and researcher degrees of freedom

Climate research can generate many candidate metrics/representations. Repeatedly testing on one benchmark produces selection bias.

The experiment system should record:

- every candidate evaluated, not only the winner;
- number of hyperparameter configurations;
- number of metrics inspected;
- number of dataset/split variants inspected;
- whether the benchmark influenced method design;
- protected confirmation sets and access history where feasible.

When many alternatives are explored, confirmation should use fresh evidence.

## 10. Statistical comparison

Do not rely on "candidate metric is 3% higher" without dependence-aware uncertainty.

Depending on the task, use:

- block/bootstrap confidence intervals;
- paired comparisons over models/events/regions;
- permutation tests respecting exchangeability assumptions;
- Bayesian hierarchical comparison across datasets/models;
- calibration curves and proper scoring rules;
- effect sizes, not only p-values;
- equivalence/noninferiority tests where the goal is cheaper computation at similar quality.

For multiple candidate methods, report the family and correction/model used rather than cherry-picking individual significance.

## 11. Robustness surfaces

A method should be evaluated over perturbations, not only one canonical preprocessing pipeline.

Potential axes:

- dataset/product;
- spatial resolution;
- temporal resolution;
- anomaly baseline;
- detrending method;
- missing-data fraction;
- noise intensity;
- regridding method;
- window length;
- regularization strength;
- physical parameter uncertainty;
- random seed;
- hardware/numerical precision when relevant.

Store the response surface where feasible. A method that is useful only at one tuned coordinate should be described accordingly.

## 12. Scientific promotion

Promotion is claim-specific.

Example path:

```text
concept
 -> prototype
 -> runnable
 -> verified on positive/negative controls
 -> empirically competitive on discovery benchmarks
 -> frozen confirmation evaluation
 -> validated for a scoped task/domain
 -> replicated
 -> eligible for defined downstream use
```

Promotion should cite evidence records. Demotion is allowed when new evidence attacks a claim.

## 13. Method rejection is a successful research outcome

A candidate should be retired or narrowed when:

- it fails its stated positive control;
- it performs no better than simpler baselines under fair tuning;
- its apparent gains vanish under leakage correction;
- its interpretation is not invariant to arbitrary representation choices;
- computational cost overwhelms any measured benefit;
- results fail independent confirmation;
- it cannot distinguish real structure from negative controls.

Keep rejected-result evidence in the experiment/evidence system where it remains relevant to preventing repeated work. Repository source history remains Git's responsibility; this document does not create a separate historical archive.

## 14. Experiment lineage and Commons

A Commons-scheduled Climate experiment should preserve Climate's scientific specification rather than translating it into an opaque command.

Conceptual causal lineage:

```text
ResearchQuestion
 -> ExperimentSpec
 -> DatasetProjection(s)
 -> CandidateRun(s) + BaselineRun(s)
 -> MetricResult(s)
 -> EvidenceRecord(s)
 -> ClaimUpdate / no update
```

Commons can own run identity, scheduling, resources, causation/correlation, and durable cross-repository lineage. Climate owns the meaning of its methods, climate datasets, scientific metrics, and validation policy.

## 15. Cross-model and multi-fidelity experimentation

A scientific question may be realized at several fidelity levels: an analytic/idealized model, a learned emulator, a parent physical GCM, a multimodel ensemble, reanalysis, and direct observations. These are not interchangeable replicates. Each run keeps its native model/runtime identity and evidence class; cross-fidelity agreement or disagreement is itself an experimental result.

Do not create a parallel orchestration or model-serving contract for this purpose. Use the existing `MethodDescriptor`, `DatasetRef`, `ExperimentSpec`, `RunManifest`, `ArtifactRef`, `MetricResult`, and `EvidenceRecord` waist. A thin external adapter should preserve the native runtime receipt and exact intervention mapping. Add new stable contract fields only when repeated executable experiments demonstrate a semantic that cannot be expressed without ambiguity.

Useful multi-fidelity experiments include:

- emulator-to-parent response-operator comparison under matched interventions;
- estimation of the discrepancy `delta(theta) = f_high(theta) - f_low(theta)` and whether that discrepancy is smooth, state-dependent, regime-dependent, or concentrated in identifiable directions;
- emulator-guided selection of a fixed expensive-run budget, compared against grid/random/simple sensitivity designs;
- offline-to-online prediction of coupled instability or drift for learned parameterizations using ClimSim-Online or an equivalent native online workflow;
- comparison of conventional benchmark metrics with invariant, response-structure, or representation-mismatch diagnostics over an AIMIP/CMIP-like model population.

Evidence transport must be explicit. Emulator evidence may motivate or select a parent-model experiment but does not automatically support the parent-model claim; multimodel agreement is not observational evidence; reanalysis is not an intervention oracle. A claim update must state which source produced the evidence and which bridge experiment, if any, justifies carrying information across sources.
## 16. Reference meta-experiment families

The following are useful experiment designs when their prerequisites are ready. Their ordering here is not priority; the planning graph supplies priority, while executable work selection also requires the commit-scoped state reconciliation in `docs/REPOSITORY-STATE.md`.

- **Geometry correctness** — known-curvature manifolds plus coordinate/scaling invariance tests before climate interpretation.
- **Multirepresentation geometry** — synthetic coupled dynamical systems with known shared/product/fibered latent structure, observed through several nonlinear views; compare raw concatenation, linear multiview baselines, established common-manifold methods, and physically constrained factorizations before inventing a bespoke learner.
- **Early-warning discrimination** — conventional indicators versus geometric candidates over tipping/non-tipping synthetic controls.
- **Teleconnection representation** — p-adic/ultrametric candidates versus correlation/spectral/graph baselines on held-out teleconnection targets.
- **Station consistency** — actual sheaf/cohomology implementation versus ordinary QC/graph residual baselines with injected faults.
- **Information-geometry optimization** — natural gradient versus standard optimizers on a small, explicit climate likelihood with known synthetic parameters and held-out observations.
- **Response-structure transfer** — apply matched perturbations to a cheap emulator and a higher-fidelity parent model; compare response subspaces/operators and use frozen selection rules to choose discriminating high-fidelity runs.
- **Offline/online structural failure** — test whether invariant residuals, Jacobian/spectral diagnostics, state-dependent error, or representation mismatch predict online coupled failure beyond ordinary offline loss.

Local tests isolate mathematical and numerical failure modes before broad integration. They are not a reason to keep scientifically related representations permanently isolated: once constituent maps are trustworthy enough, integrated experiments should test whether their joint geometry captures climate dynamics, information, and physical structure that separate views or simpler fusion baselines miss.