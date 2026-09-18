# Multirepresentation climate-manifold research program

Status: **scientific research direction / candidate unifying construct**, not a claim that the climate system is already known to possess one privileged smooth manifold.

This document defines a program for evaluating whether several mathematically different representations of climate state can be assembled into a useful, grounded geometric object. It is intentionally more specific than a generic manifold-learning proposal and less dogmatic than declaring one representation canonical in advance.

The central hypothesis is:

> Climate state may admit a useful **atlas of complementary representations** whose shared and state-dependent geometry can be learned or constructed under physical, statistical, and dynamical constraints. A joint manifold, product/fiber geometry, or related structure should be accepted only if it improves scientifically meaningful tasks while retaining interpretable links to the representations from which it was built.

The point is not to put every mathematical idea in the repository on equal footing. Some representations may form the primary geometry; others may contribute a local metric, a constraint, a kernel, a tangent-space decomposition, or an alternative geometry to compare and reject.

## 1. Scientific premise and guardrails

Several choices that can appear under the single label “climate manifold” are scientifically distinct and must remain independently inspectable:

- climate-state variable selection;
- data-source and observation fusion;
- dimensionality discovery;
- metric construction;
- connection and curvature calculation;
- cross-scale transport and coupling;
- dynamical closure and prediction;
- topology;
- physical interpretation of geometric quantities.

No single geometric scalar determines which representation is scientifically useful. In particular, reduced scalar curvature is not a model-selection law for accepting a latent dimension. A representation can flatten a geometry while discarding climate dynamics, identifiable information, physical balances, or useful subsystem structure.

Curvature, geodesics, holonomy, topology, intrinsic dimension, and related geometric quantities are therefore **consequences to measure**, not universal optimization targets.

## 2. The object being sought

Let `X` denote a physical climate-state space at a declared level of description. `X` need not initially be a manifold, and its full gridded state need not be the coordinate system used for research.

Define multiple representation maps

```text
phi_a : X -> M_a
```

where each `M_a` represents one scientifically interpretable view of the same underlying state or trajectory.

Examples of candidate views include:

- prognostic physical fields and conservative/balanced combinations;
- thermodynamic and compositional state;
- ocean, atmosphere, cryosphere, land, and biogeochemical subsystem summaries;
- multiscale spatial or spectral modes;
- Koopman/DMD or other dynamically coherent observables;
- teleconnection/network representations;
- ensemble and uncertainty representations;
- statistical-model or parameter likelihoods;
- local observational representations from station, satellite, ocean, or reanalysis products;
- learned latent coordinates constrained by the above rather than treated as self-justifying.

The research question is not merely whether these views can be concatenated. It is whether their geometry admits a useful **common, factorized, fibered, quotient, or charted representation**.

## 3. Several geometries may be appropriate

The default should not be one dense Euclidean latent vector.

### 3.1 Product structure

If two representation families describe genuinely separable local degrees of freedom, a product manifold

```text
M = M_1 x ... x M_k
```

is a meaningful hypothesis. Product structure is stronger than low correlation. It predicts a tangent-space decomposition and constrained forms for the metric and Laplacian.

This is a candidate for factors that are approximately independent over a declared regime, not a universal statement that atmosphere, ocean, ice, and land are independent.

### 3.2 Fibered structure

Many climate degrees of freedom are conditional on others. Fast atmospheric variability, cloud-state structure, local circulation modes, or uncertainty geometry may depend strongly on a slower background state.

A fiber-bundle-style hypothesis can represent this more naturally:

```text
fiber / local degrees of freedom
          |
          v
     slow/background state
```

The fiber need not be globally identical over the base. Changes in local effective dimension or coupling can then be represented without pretending that one global Cartesian latent space is appropriate.

### 3.3 Common-manifold structure across observation modalities

Different observing systems may measure different functions of the same latent climate variables while also containing modality-specific nuisance variability.

Methods such as alternating diffusion and jointly smooth features provide concrete non-linear models of this situation: retain variability shared by aligned views while suppressing view-specific nuisance structure. These are relevant candidates for combining, for example, reanalysis, satellite, station, and ocean observations without assuming that raw concatenation defines a sensible metric.

They are not automatically the correct climate solution. A genuinely subsystem-specific climate signal must not be discarded merely because only one modality can observe it.

### 3.4 Quotient or nuisance-removal structure

Some representation directions may correspond to choices that should not alter the scientific state being compared: coordinate parameterization, selected normalization conventions, or other declared nuisance transformations.

Where a true symmetry/equivalence relation exists, quotient geometry may be preferable to asking a learner to rediscover invariance from examples.

Physical variability such as seasonal phase must not be quotiented away merely because it is inconvenient.

### 3.5 Atlas or stratified structure

There is no requirement that the useful climate representation have one global chart or fixed intrinsic dimension. Distinct regimes may need different local coordinates, with explicit overlap/transition maps.

If dimension or smooth structure genuinely changes across regimes, a stratified or piecewise-smooth representation may be more truthful than forcing a globally smooth Riemannian model.

## Structural-world benchmark as the primary falsification program

The multirepresentation program must not be organized around one successful common-coordinate fixture. Its primary synthetic laboratory is a battery of worlds whose true relationships are deliberately different:

- **common manifold** — both views are injective functions of one shared latent coordinate;
- **product** — the paired realization contains view-specific factors but no shared coordinate, and static cross-view dependence alone cannot prove the generative product semantics;
- **fibered** — a shared base is accompanied by private view-specific fibers;
- **quotient/noninjective** — one view discards information, so only a quotient coordinate is common and the lost direction cannot be reconstructed by a joint method;
- **stratified/regime** — the observation relationship changes across declared strata rather than admitting one globally smooth chart;
- **nuisance-dominated** — a real shared coordinate exists but high-variance private nuisance is an adversary to variance-based fusion;
- **independent null** — there is no cross-view shared state to recover.

`fixtures/multirepresentation/structural-worlds-v1.json` and `reference/multirepresentation_worlds.py` are ground-truth authorities for these worlds. They are not a structure-selection algorithm.

A valid method is therefore allowed to **abstain**. In some worlds the requested generative distinction is not identifiable from the supplied static paired observations. The benchmark must reward calibrated refusal in those cases rather than forcing every method to return a manifold label, embedding, or score. A claimed structural identification needs an observation model that makes that distinction identifiable.

The existing jointly-smooth common-coordinate fixture remains useful as one positive control, but it no longer defines the research program. A method that succeeds only on the shared-manifold case and hallucinates shared structure in product or null worlds has failed the larger experiment.

## 4. What “orthogonal representations” should mean

`Orthogonal` is useful only after the inner product or other separation criterion is declared.

Possible meanings include:

- **metric orthogonality:** tangent subspaces have zero cross-inner-product under a candidate metric;
- **statistical orthogonality:** score functions or estimated directions are orthogonal in a Fisher metric;
- **decorrelation:** empirical coordinates have zero covariance under a declared distribution;
- **dynamical decoupling:** linearized or transfer-operator dynamics have weak cross-coupling over a declared regime;
- **information separation:** one view contributes information about a target not already contained in another;
- **algebraic independence:** factors arise from a product or symmetry decomposition.

These are not interchangeable. In particular, decorrelation does not establish dynamical or geometric independence.

The program should therefore **measure cross-representation coupling** rather than force every representation to be orthogonal by construction.

## 5. Metric construction as a scientific hypothesis

Suppose each representation `M_a` has a locally meaningful metric `g_a`. A candidate metric on climate state can be constructed through pullback:

```text
G_a = phi_a^* g_a
```

A composite candidate may then have the schematic form

```text
G = sum_a lambda_a(x) G_a + cross_representation_terms(x)
```

subject to explicit requirements:

- positive definiteness or an explicitly different geometric interpretation;
- unit/nondimensionalization semantics;
- conditioning and rank diagnostics;
- invariance under transformations that should not change the physical conclusion;
- sparse or structured cross-couplings unless evidence supports dense coupling;
- no silent regularization that hides unidentifiable directions;
- locally meaningful behavior when one modality/view is missing.

The weights `lambda_a` and cross terms are not generic hyperparameters to tune until a score improves. They express how much a local displacement in each representation contributes to distinguishable climate change and therefore need physical/statistical interpretation or an explicit learned-candidate status.

## 6. Information geometry has a central, but specific, role

Information geometry is unusually well suited to this program because it supplies a metric with a direct statistical meaning when an explicit probability model exists.

For a parameterized observational or predictive model

```text
p(y | theta)
```

the Fisher information defines local distinguishability of parameter perturbations. The canonical Gaussian-mean Fisher primitive preserves null directions instead of hiding them with a regularized inverse.

There are at least three distinct uses:

1. **parameter geometry** — identify stiff, sloppy, degenerate, or well-observed parameter combinations;
2. **state geometry through an observation model** — pull back a Fisher metric from predicted observation distributions to climate-state or latent coordinates;
3. **uncertainty geometry** — represent ensemble/predictive distributions as statistical states rather than appending moments to a deterministic state vector.

These spaces must not be conflated. A Fisher metric on model parameters is not automatically the physical metric of climate state. Sensitivity maps can connect them, and those maps themselves become scientifically meaningful objects.

The pullback perspective is especially useful because it asks a grounded question:

> Which perturbations of the candidate climate representation are actually distinguishable through the declared observations or predictive distribution?

This can expose directions that a purely geometric latent learner would treat as important even though they are observationally unidentifiable.

## 7. Dynamics should participate in the geometry

A climate representation that reconstructs snapshots well but destroys the evolution law is of limited value.

For physical dynamics

```text
dx/dt = F(x)
```

and a representation `phi`, the local pushforward

```text
d(phi(x))/dt = D phi_x F(x)
```

provides a direct way to test whether a chart or latent coordinate system preserves dynamically meaningful directions.

Candidate geometry should therefore be evaluated using quantities such as:

- predictability of represented tendencies;
- smoothness/closure of the vector field in the learned coordinates;
- separation of slow and fast modes where real timescale separation exists;
- preservation of balanced/conservative subspaces;
- transport of perturbations and uncertainty;
- stability of local tangent estimates;
- ability to reconstruct physically important observables and fluxes.

Diffusion-map/Koopman work is relevant here because it shows that data-driven coordinates can be chosen for dynamical smoothness and timescale structure rather than only snapshot variance.

A learned manifold should not become the authoritative physical time integrator merely because its latent dynamics are compact. Reduced/manifold dynamics must first demonstrate closure, reconstruction, conservation/balance behavior, and out-of-sample trajectory skill.

## 8. Relationship to the structure-preserving physical core

The physical kernels and the multirepresentation manifold program should reinforce one another without collapsing into one layer.

The physical core supplies hard structure such as:

- extensive mass/tracer budgets;
- continuity;
- hydrostatics;
- no-work Coriolis rotation;
- dissipative diffusion;
- thermodynamic identities;
- energy, momentum, phase-change, and coupling balances.

Those structures can constrain or evaluate a candidate representation. Examples:

- tangent directions that violate exact algebraic state constraints can be excluded or penalized;
- metric candidates can be tested for whether physically small conservative perturbations remain small;
- balanced and unbalanced perturbations can be distinguished;
- reduced dynamics can be rejected if they create numerical leakage between physical reservoirs;
- structure-preserving discretizations can provide trajectories whose geometric signatures are not dominated by known numerical artifacts.

The reverse direction is also possible: a well-supported representation may reveal low-dimensional coordinates, slow variables, conditioning structure, or coupling patterns useful for solver partitioning and reduced models.

## 9. Roles of the repository’s unusual mathematical methods

The mathematical families should not be declared equally central merely to make the architecture symmetrical.

### Geometry and manifold learning — primary candidate framework

These define the common representation problem itself: coordinates, charts, local metrics, connections, curvature, geodesics, and representation comparison.

### Information geometry — primary grounding mechanism where probability models exist

Fisher/Amari structure gives statistical meaning to local distance, identifiability, and parameter/observation sensitivity.

### Dynamical/Koopman/spectral representations — primary candidate views

These can contribute coordinates organized by timescale, coherent evolution, oscillatory structure, or predictable observables rather than static variance.

### Sheaf/cohomological methods — local consistency and gluing structure

A climate-data sheaf is more naturally a way to represent **local observation compatibility and transition/restriction structure** than another coordinate axis in a Riemannian vector.

It may help determine whether local charts/data products can be consistently combined and where incompatibilities live.

### Ultrametric/p-adic methods — alternative relational geometry

Hierarchical teleconnection structure may be useful even if it is not smoothly embeddable into the main Riemannian manifold. It can contribute a kernel, neighborhood relation, hierarchy, or competing metric family.

Forcing p-adic distance into a smooth tangent metric would erase the very structure being tested.

### Clifford/geometric algebra — local multicomponent/tangent representation

Clifford structure may be useful for oriented mode interactions, phase relationships, and multivector representations of local tangent dynamics. It does not need to define the global climate manifold to be useful.

### Noether/variational and Hamiltonian structure — physical dynamical constraints

Where conservative subdynamics genuinely possess these structures, they constrain admissible dynamics and numerical evolution. They are not generic latent coordinates.

## 10. Tuning the representation

The central engineering/scientific task is not to optimize one geometric scalar. It is to compare candidate representation systems under a **multi-objective scientific loss surface**.

Candidate objectives include:

### Physical reconstruction

Can the representation reconstruct declared physical observables, budgets, fluxes, and subsystem state without systematic bias?

### Dynamical sufficiency

Can it represent short- and medium-horizon evolution, tendencies, transfer-operator action, regime residence, and transitions better than simpler representations?

### Information content

Does it preserve identifiable directions and uncertainty structure under declared observation/likelihood models?

### Cross-view agreement

Do independent modalities map nearby realizations of the same physical state to compatible local coordinates while preserving valid modality-specific information?

### Representation robustness

Are conclusions stable to units, coordinate reorderings, admissible reparameterizations, data products, and modest preprocessing changes?

### Structural fidelity

Does the representation respect known physical constraints, symmetries, conservation/balance relations, and known subsystem couplings?

### Complexity and conditioning

Does added dimension/coupling materially improve the scientific objectives enough to justify conditioning, sample complexity, computational cost, and interpretability loss?

Curvature, geodesic distortion, holonomy, intrinsic dimension, spectral gap, and topology belong in the diagnostic set, but none is by itself the universal target.

## 11. A staged scientific sandbox

The first experiments should make the representation question answerable before full Earth-system complexity is introduced.

### M0 — known factor/manifold synthetic systems

Construct systems with known latent product, coupled, and fiber-like structure. Observe them through multiple nonlinear modalities with controlled nuisance variables.

Compare:

- raw concatenation;
- PCA/CCA-style linear fusion;
- separate diffusion maps;
- alternating/common-manifold diffusion;
- jointly smooth features;
- product/factor manifold candidates;
- learned latent encoders with matched information access.

Primary outputs are recovery of known latent structure, nuisance rejection, cross-view generalization, and dynamical sufficiency.

### M1 — physically structured idealized dynamics

Use small climate-relevant dynamical systems where physical invariants/balances are known. Candidate examples include rotating shallow-water, moist-column, coupled atmosphere-ocean oscillator, or other independently specified canonical problems as those kernels become available.

Ask whether candidate coordinates separate balanced/unbalanced, slow/fast, conservative/dissipative, or subsystem directions without losing the governing balances.

### M2 — multimodal climate observations

Use aligned immutable projections from complementary data products rather than one concatenated table.

Possible views include atmospheric reanalysis, satellite radiances/retrievals, ocean observations, station networks, cryosphere fields, and model ensembles.

The experiment must distinguish:

- common physical variability;
- legitimately view-specific climate information;
- sensor/product nuisance;
- preprocessing-induced common structure.

### M3 — information-geometric coupling

For a small explicit likelihood or forecast model, compare candidate state-space metrics against the Fisher geometry induced by observations.

Study where physical/dynamical and information-geometric notions of important direction agree, disagree, or become rank-deficient.

### M4 — regime and transition utility

Only after the representation itself is credible, ask whether its geometry improves regime identification, transition forecasting, sensitivity analysis, rare-state detection, or uncertainty propagation over strong dynamical/statistical baselines.

## 12. Engineering boundary

This program needs substantial engineering, but its novelty budget belongs in the representation semantics and their climate integration.

Prefer established implementations for:

- linear algebra and eigensolvers;
- automatic differentiation;
- standard optimization;
- nearest-neighbor/kernel machinery;
- ordinary statistical estimators;
- generic ODE/PDE time integration;
- distributed arrays and accelerators.

Small reference implementations remain appropriate when they expose the mathematics needed for independent differential tests.

Custom Climate code should concentrate on:

- representation definitions and physical units;
- cross-view alignment semantics;
- state/parameter/observation maps;
- physically meaningful factor/fiber hypotheses;
- metric pullbacks and constrained metric composition;
- dynamical pushforwards and reconstruction maps;
- climate-specific constraints and balance diagnostics;
- experiment definitions that compare genuinely different geometries.

Do not introduce a universal `ManifoldFramework` abstraction before the first experiments demonstrate which structures actually recur.

## 13. Relevant external method families

The following are research references, not automatic dependency choices:

- Coifman and Lafon, diffusion maps: multiscale geometry from diffusion operators;
- Lederman and Talmon, alternating diffusion: extraction of geometry common to multiple observation modalities;
- Talmon and Wu / Katz et al., latent common-manifold and multimodal alternating-diffusion methods;
- Dietrich et al., jointly smooth features: orthogonal functions jointly smooth across multiple observed manifolds;
- product-manifold learning/factorization methods for independent continuous factors;
- Giannakis et al., diffusion/Koopman approaches to dynamically meaningful spatiotemporal coordinates;
- Fisher-Rao pullback geometry for probabilistic decoders/observation models.

The purpose of surveying these methods is to avoid inventing generic manifold-learning algorithms while still allowing Climate to contribute the physically grounded multi-view construction they do not decide for us.

## 14. Promotion boundary

This document does **not** assert that there is one correct low-dimensional climate manifold.

Useful outcomes include:

- a stable joint geometry exists over a declared regime;
- only local charts are defensible;
- product/fiber structure is useful for some subsystem split but not others;
- information geometry reveals directions absent from state geometry;
- a supposedly novel representation adds nothing beyond simpler views;
- an ultrametric/sheaf/algebraic method is useful as an auxiliary structure but not as part of the main manifold;
- no low-dimensional joint representation is sufficiently faithful for the tested task.

All of those outcomes advance the research program.

The unifying commitment is therefore not to a particular manifold. It is to **making the relationships among physical, dynamical, statistical, observational, and experimental representations explicit enough that a useful geometry can be discovered, compared, or rejected without erasing what each representation means.**

## Benchmark ladder beyond static structural worlds

The structural-world zoo answers what relationship is present in deliberately known paired views. A complementary climate-relevant ladder uses the exact two-layer EBM to ask what a representation preserves under intervention and observation change.

The first forcing rung uses one exact piecewise-affine propagation path for held forcing, ramps, overshoot/reversal, and sign reversal. The fixture declares a narrow discovery forcing domain and a wider confirmation domain. This split is immutable benchmark semantics: fitted candidates may learn only from discovery inputs, while confirmation trajectories remain held out. The exact reference itself is not trained, so its successful execution is a numerical control rather than evidence of OOD generalization.

Later rungs should add sparse/noisy observations, redundant channels, parameter degeneracy, stochastic variability, and simple regime-dependent feedback as new versioned fixtures. They must not be introduced by silently mutating the original linear control. State reconstruction, forced response, identifiability, climate-statistics preservation, and OOD response remain separate tasks and metrics.
