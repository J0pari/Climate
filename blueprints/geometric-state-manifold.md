# Geometric climate-state representation blueprint

Status: **research specification**.

Riemannian geometry is one candidate language for representing climate state and its local structure. It is not assumed to be the unique geometry of the climate system, and geometric quantities do not acquire physical interpretation merely because they can be computed.

This blueprint defines the Riemannian component of the broader multirepresentation program in `docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md`.

## Capability layers

Keep the following responsibilities separate:

```text
geometry/reference
  generic metric -> connection -> curvature machinery

geometry/representation_maps
  physical/statistical state -> declared geometric representation

geometry/metric_candidates
  explicit candidate metric constructions

geometry/diagnostics
  coordinate-invariant geometric and numerical quantities

geometry/dynamics
  pushforward/pullback of declared evolution and tangent structure

geometry/meta_experiments
  comparisons against simpler and established representations

geometry/interpretation
  optional climate-facing mappings supported by separate empirical evidence
```

A representation map and a metric are distinct scientific choices. A correct Levi-Civita implementation does not determine which climate variables belong in a chart, which directions should be identified or quotiented, or what notion of distance is scientifically useful.

## Metric candidates are hypotheses

Each metric candidate must have an explicit identity, domain, assumptions, units/scaling semantics, and construction rule. Candidate families may include:

- physically derived nondimensional metrics;
- Fisher/information metrics from an explicit likelihood;
- covariance/Mahalanobis metrics;
- diffusion or graph-induced local geometry;
- learned local metrics with explicit constraints;
- product metrics assembled from independently meaningful factors;
- pullback metrics induced by observation or representation maps;
- carefully documented hand-designed metrics used as experimental hypotheses.

No metric becomes authoritative because it is convenient, numerically smooth, or produces visually interesting curvature.

## Representation-map semantics

For a representation map

\[
\phi : X \rightarrow M,
\]

where `X` is a declared physical, statistical, or latent state and `M` is a geometric representation, the implementation should make the following inspectable where applicable:

- the coordinates and their units or nondimensionalization;
- nuisance transformations or quotient symmetries;
- the Jacobian `D phi` and its numerical conditioning;
- pullback or pushforward operations;
- null directions and information loss;
- whether the map is local, global, many-to-one, or chart-dependent;
- how the physical evolution field transforms under the map.

When a Fisher or observation metric is pulled back to state coordinates, semidefinite null directions represent local non-identifiability and must not be silently regularized away.

## Geometric diagnostics

Useful geometric outputs may include:

- metric conditioning and rank;
- Christoffel symbols and covariant derivatives;
- Riemann, Ricci, scalar, and sectional curvature;
- geodesic distance and geodesic deviation;
- parallel transport and holonomy;
- volume elements;
- local intrinsic dimension or tangent structure when well defined.

These are descriptors of a declared geometry. Curvature, geodesic length, holonomy, or intrinsic dimension are not universal optimization objectives for selecting the representation.

## Dynamics and physical structure

A geometric representation should be evaluated against the dynamics it is meant to organize. For a state evolution

\[
\dot{x}=F(x),
\]

the representation induces

\[
\frac{d}{dt}\phi(x)=D\phi_x F(x).
\]

Candidate coordinates should therefore be tested for whether they preserve or clarify scientifically relevant structure such as:

- slow/fast directions;
- balanced and unbalanced modes;
- conservative and dissipative subdynamics;
- subsystem coupling;
- identifiable and weakly observed directions;
- invariant or approximately invariant sets;
- trajectory neighborhoods and predictability.

Physical balance laws and structure-preserving dynamical kernels may constrain admissible representations rather than merely serving as downstream diagnostics.

## Verification requirements

Before climate-facing interpretation, require:

1. generic reference geometry passes analytic known-geometry fixtures;
2. derivative mechanisms have independent witnesses;
3. conditioning, rank, positive-definiteness or semidefiniteness, and refusal semantics are explicit;
4. unit, coordinate, and permutation metamorphic tests exist;
5. representation-map and pullback/pushforward operations have analytic or differential witnesses;
6. metric candidates have separate identities and assumptions;
7. geometric outputs remain geometric outputs until a climate interpretation has separate support;
8. optimized/GPU implementations pass CPU/reference differential tests on the same immutable fixtures;
9. climate-usefulness experiments include strong simpler representations and protected evaluation data.

The detailed numerical contract is `docs/GEOMETRY-VERIFICATION.md`.

## Interpretation boundary

The geometry layer does not directly emit:

- named Earth-system tipping events from curvature alone;
- physical tipping probabilities from arbitrary transforms of curvature or eigenvalues;
- physical time-to-collapse from inverse geometric scales without a derived model;
- irreversibility from an arbitrary geometric threshold;
- gravitational or Einstein-equation analogies as physical climate dynamics.

Any such mapping is a separate scientific hypothesis with its own model, assumptions, calibration, and evidence.
