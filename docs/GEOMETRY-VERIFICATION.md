# Geometry verification program

Status: **living verification contract with partially realized reference/canonical CPU coverage** for the geometric/manifold family.

This document defines the minimum mathematical/numerical evidence required before Climate may interpret curvature, connections, geodesics, or related geometric quantities as climate indicators. Geometry machinery is tested independently of climate data first.

The controlling claim is `climate.geometry.regime_indicator`. Passing mathematical or numerical geometry tests does **not** validate that claim; it only establishes correctness of a declared geometric computation under the tested scope.

## Verification authorities and scope

The geometry program uses complementary portable authorities and executable witnesses:

- `reference/geometry_sympy.py` plus `tests/reference/test_geometry_sympy.py`: symbolic known-geometry oracle covering flat Cartesian, flat polar/nonzero-connection, positive-curvature sphere, negative-curvature Poincaré disk, torsion freedom, metric compatibility, Riemann antisymmetry, and first Bianchi witnesses;
- `src/geometry.rs`: canonical Rust Levi-Civita kernel from explicit local metric jets, with executable Cartesian/polar/sphere fixtures, metric-compatibility and Bianchi witnesses, and fail-closed nonsymmetric/singular-metric behavior;
- `tests/geometry_coordinate_metamorphics.rs`: constant-linear coordinate covariance under permutation and anisotropic rescaling, plus finite-difference metric-jet construction for a nonconstant conformal metric.

These authorities deliberately stop before choosing a climate metric or mapping curvature to tipping/risk/timescale. Remaining obligations include nonlinear coordinate-change witnesses with the inhomogeneous Christoffel term, an additional derivative-generation route independent of finite differences where applicable, conditioning/refusal characterization, geodesic verification, and real CUDA differential comparison.

Module and claim lifecycle state is rendered from machine authorities in `docs/generated/STATUS.md`; this document owns verification semantics and obligations rather than duplicating generated status.

## 1. Separation of questions

The program keeps four questions distinct:

1. **Geometry correctness:** does the implementation compute the declared geometric object correctly?
2. **Representation robustness:** is the quantity stable/invariant under transformations that should not change the geometry?
3. **Numerical robustness:** do precision, discretization, conditioning, and acceleration choices preserve the declared answer?
4. **Climate usefulness:** does a verified geometric quantity add out-of-sample information about climate behavior beyond appropriate baselines?

Questions 1–3 must be answered before question 4 can support a physical/predictive interpretation.

## 2. Reference implementation first

Maintain small, readable reference implementations before optimizing the CUDA path.

Reference priorities:

- explicit metric input rather than a climate-specific hand-built metric;
- analytic metric functions for fixtures;
- derivatives computed by at least one independently checkable method;
- FP64 default;
- clear tensor index conventions;
- no climate tipping/risk mapping;
- outputs limited to geometric/numerical diagnostics.

The reference may use Python/SymPy, Rust, Julia, or another suitable independent mechanism, but the oracle must be independent enough that optimized code does not merely reproduce the same implementation bug.

## 3. Convention contract

Before comparing any tensor values, pin these conventions in code and fixtures:

```text
metric signature
coordinate ordering
Riemann sign convention
R^i_{jkl} index order
Ricci contraction convention
scalar-curvature contraction
sectional-curvature convention
Christoffel index order
finite-difference boundary handling
units/scaling of coordinates
```

A convention mismatch is not a failed physical hypothesis; it is an interface error.

## 4. Fixture A — Euclidean Cartesian space

For dimension 2, 3, and at least one dimension matching the intended climate geometry implementation where practical:

```text
g_ij = delta_ij
Gamma^i_jk = 0
R^i_jkl = 0
Ric_ij = 0
R = 0
```

Tests:

- all Christoffel components within declared FP64 tolerance;
- all Riemann components within tolerance;
- Ricci/scalar curvature within tolerance;
- geodesics are straight lines with affine parameterization;
- parallel transport preserves a constant vector;
- CPU/reference and accelerated paths agree.

This fixture catches incorrect contractions, uninitialized memory, wrong derivative indexing, and basic tensor-layout errors.

## 5. Fixture B — flat Euclidean space in nonlinear coordinates

Use at least polar/cylindrical coordinates and one additional smooth nonlinear coordinate transformation.

Expected behavior:

- Christoffel symbols are generally nonzero;
- Riemann/Ricci/scalar curvature remain zero;
- geodesic coordinate components may look curved while representing straight geometric paths.

This fixture is critical. A method that treats large connection coefficients or coordinate derivatives as physical curvature will fail here.

Primary invariant:

```text
coordinate transformation changes representation, not intrinsic curvature
```

## 6. Fixture C — 2-sphere

For a sphere of radius `r` with standard spherical coordinates:

```text
sectional/Gaussian curvature = 1/r^2
scalar curvature (2D) = 2/r^2
```

Test several radii to ensure the implementation reproduces the expected scaling rather than one memorized magnitude.

Also test away from and near coordinate singularities separately; a chart singularity must not be mislabeled a physical/geometric singularity.

## 7. Fixture D — hyperbolic space

Use a simple 2D hyperbolic metric with known constant negative curvature.

Expected behavior:

- correct sign;
- correct magnitude scaling;
- tensor symmetries hold;
- implementation distinguishes genuine negative curvature from numerical conditioning artifacts.

Together sphere + hyperbolic fixtures prevent a zero-only implementation from passing.

## 8. Fixture E — conformally flat metric with analytic curvature

Choose a smooth conformal factor for which curvature can be derived analytically or checked symbolically.

Purpose:

- exercise nonconstant metric derivatives;
- exercise second-derivative sensitivity;
- test spatially varying curvature rather than constant-curvature memorization;
- expose finite-difference/AD errors.

## 9. Tensor identity witnesses

Where applicable, check geometric identities independently of fixture-specific expected values:

- metric symmetry `g_ij = g_ji`;
- inverse residual `||g g^-1 - I||`;
- Levi-Civita torsion-free symmetry `Gamma^i_jk = Gamma^i_kj`;
- metric compatibility `nabla g ~= 0`;
- Riemann antisymmetries under the chosen index convention;
- pair symmetry after lowering indices;
- first Bianchi identity;
- Ricci symmetry for Levi-Civita connection.

Tolerance must scale with the numerical method and fixture magnitude.

A tensor can have a plausible scalar contraction while violating these identities; scalar-only tests are insufficient.

## 10. Derivative verification

For metric and connection derivatives, compare at least two independent mechanisms on small fixtures:

- analytic derivative where available;
- automatic differentiation;
- central finite difference;
- complex-step derivative where the implementation/function permits it.

Run step-size sweeps for finite differences to demonstrate the expected truncation/roundoff tradeoff.

Do not accept one hard-coded epsilon as derivative validation.

## 11. Coordinate and scaling metamorphic tests

For every candidate geometry implementation add metamorphic tests:

### 11.1 Coordinate reparameterization

Transform a known geometry into another smooth chart and verify coordinate-invariant quantities agree after transformation.

### 11.2 Unit/scale transformation

Where coordinates represent dimensionful climate quantities, changing units (for example K versus scaled K, ppm versus a documented transformed coordinate) must not accidentally create a different scientific conclusion unless the metric definition explicitly depends on that representation.

This test is particularly important for legacy Climate metrics combining transformed CO2/CH4/N2O, scaled ocean heat, ice fraction, and forcing variables.

### 11.3 Permutation

Permute coordinate ordering and verify tensors transform consistently. This catches hard-coded "index 2 means ocean" assumptions inside generic geometric code.

### 11.4 Representation maps, pullbacks, and composition

The multirepresentation program introduces maps `phi_a: X -> M_a` between a declared physical/latent state and representation spaces. Those maps need independent mathematical witnesses before their induced geometry is interpreted.

For small analytic fixtures verify, where applicable:

- a pullback metric agrees with `D phi^T g D phi` under the pinned coordinate convention;
- composition of known maps gives the same pullback as the corresponding staged composition;
- a declared product geometry reduces to the expected block structure when cross terms are absent;
- explicitly declared cross-representation terms transform consistently rather than depending on coordinate ordering;
- two observation views of a known shared latent system recover the expected common directions without inventing information in null directions;
- nuisance transformations declared as quotiented symmetries do not change the intended intrinsic quantities.

Do not call statistically uncorrelated, low-covariance, or low-mutual-information views "orthogonal" unless the exact notion of orthogonality is declared. Metric orthogonality, Fisher orthogonality, covariance decorrelation, dynamical decoupling, and information complementarity are different statements.

## 12. Positive-definiteness and conditioning

A Riemannian metric must be positive definite in the domain where that interpretation is claimed.

For every evaluated point record:

- symmetry residual;
- minimum eigenvalue or factorization success;
- condition estimate;
- regularization applied;
- inverse residual.

Adversarial fixtures include:

- nearly singular positive-definite matrices;
- deliberately indefinite matrices;
- extreme coordinate scales;
- highly anisotropic but valid metrics.

Required behavior:

- invalid metric -> explicit refusal/diagnostic;
- ill-conditioned metric -> diagnostic + declared refinement/regularization behavior;
- no silent replacement by identity/zeros;
- no physical probability emitted from a failed geometry calculation.

This requirement is scoped to objects claimed to be Riemannian metrics. Fisher information and observation-induced pullbacks may be positive **semidefinite** when parameters or state directions are locally unidentifiable. Such null directions are scientific/statistical information and must not be silently regularized into a fictitious positive-definite geometry. If a later algorithm requires a nonsingular metric, the restriction, quotient, prior, or regularization that makes it nonsingular is a separate declared operation.

## 13. Precision ladder

Run the same fixtures through declared precision modes:

```text
reference FP64 (or higher where useful)
FP64 accelerated
FP32 accelerated
mixed precision candidate
FP16/BF16 screening if proposed
```

Measure:

- tensor component error;
- invariant error;
- identity residuals;
- condition dependence;
- categorical stability of any downstream *numerical* classification.

A precision mode that is adequate for ranking easy candidates may still be inadequate for final evidence. Keep those implementation identities distinct.

## 14. CPU/GPU differential testing

The first GPU acceptance path should use the same immutable fixture inputs as the reference implementation.

Compare stage by stage where feasible:

```text
metric
metric derivatives
factorization/inverse diagnostics
Christoffel symbols
selected Riemann components / contractions
Ricci
scalar/sectional curvature
summary metrics
```

Stage-level comparison localizes errors that a final scalar metric could hide.

The first accepted GPU implementation should be single-GPU. Multi-GPU equivalence is a later contract.

## 15. Batched-candidate isolation test

Because a proposed GPU engine evaluates many geometry candidates together, prove candidates cannot contaminate one another.

Required metamorphic witness:

```text
result(candidate A alone)
 == result(candidate A at batch slot 0)
 == result(candidate A at another batch slot)
 == result(candidate A beside adversarial candidate B)
```

within the declared determinism/tolerance class.

Also randomize candidate order and batch size.

## 16. Memory/index stress tests

Use dimensions/grid sizes that exercise:

- non-multiples of warp/block tile sizes;
- minimum legal sizes;
- maximum CI-sized fixture;
- boundary cells;
- odd candidate counts;
- empty/disabled optional outputs;
- large-but-valid tensor strides.

Where tooling is available, run CUDA memory/race checking on the small suite.

## 17. Geodesic verification

Before interpreting geodesics as "optimal climate transitions," verify the numerical geodesic solver separately.

Tests:

- Euclidean straight-line solution;
- great-circle behavior on sphere;
- reversal/parameterization properties appropriate to the solver;
- convergence under step-size refinement;
- conservation of geodesic speed for affine Levi-Civita geodesics within tolerance;
- endpoint/boundary-value solver residual where applicable.

A correct geodesic of an arbitrary hand-built metric is still not automatically a physically optimal climate pathway.

## 18. Prohibited interpretation before verification

Until the relevant mathematical, numerical, and empirical programs are satisfied, outputs such as these remain diagnostic/prototype quantities only:

```text
"tipping probability"
"irreversible"
"AMOC collapse risk"
"thermal runaway"
"timescale to collapse"
```

when they are derived only from curvature/eigenvalue thresholds or analogous heuristic transforms.

Curvature, geodesic length, holonomy, topology, and intrinsic dimension are also not universal optimization objectives. A representation can make one of those quantities smaller or simpler while discarding dynamics, information, or physical meaning. Their role is as properties/diagnostics of a declared representation unless a separate task-specific objective is justified.

## 19. First climate-facing meta-experiments after numerical verification

Before named Earth-system tipping interpretation, use synthetic dynamical systems whose latent geometry and dynamics are deliberately known.

The first multirepresentation experiment should expose the same latent dynamics through several nonlinear views with known shared, product, nuisance, or fibered structure. Compare raw concatenation, linear multiview baselines, established common-manifold methods, and explicitly factorized candidate geometry. Test whether the recovered representation preserves known latent neighborhoods, dynamical evolution, identifiable/null directions, and declared physical invariants. This experiment evaluates representation construction; it does not require a tipping-point story.

A separate regime-indicator experiment should evaluate whether geometry adds information on synthetic dynamical systems with known labels:

- systems approaching a known bifurcation;
- systems with no bifurcation but changing variance/noise;
- nonstationary forcing without tipping;
- regime switches generated by mechanisms not represented by critical slowing down;
- coordinate-transformed copies of the same systems.

Baselines:

- variance;
- lag-1 autocorrelation;
- restoring-rate estimates where appropriate;
- simple state-space/Jacobian indicators;
- learned classifier only as a secondary high-capacity comparator.

Primary regime-indicator question:

```text
Does a verified geometric candidate improve held-out discrimination or calibrated forecasting beyond strong conventional baselines without exploiting coordinate/scaling artifacts?
```

This is evidence for predictive utility, not yet evidence for a named Earth-system tipping element.

## 20. Promotion criteria

A geometric implementation may progress to `verified` only when its declared scope has appropriate evidence for:

- pinned conventions;
- known-geometry fixtures;
- tensor identities;
- derivative witnesses;
- conditioning/refusal behavior;
- coordinate/scaling metamorphic tests;
- representation-map/pullback witnesses when those operations are in scope;
- precision characterization;
- independent implementation agreement where applicable;
- absence of placeholders on the verified computation path.

Verification may be scoped to a subset of these when the module advertises a narrower capability; the scope must be explicit rather than implying the entire geometric stack is verified.

The claim `climate.geometry.regime_indicator` remains `concept` until separate predictive/generalization evidence is attached. Verification of the math implementation must not auto-promote the scientific claim.
