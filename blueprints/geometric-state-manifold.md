# Geometric climate-state manifold blueprint

Status: **capability unavailable / legacy manifold implementation scheduled for removal**.

The legacy `climate_manifold.rs` combined several distinct questions into one authoritative-looking implementation:

- choice of climate-state coordinates;
- hand-constructed Riemannian metric;
- automatic/dual differentiation;
- Christoffel/Riemann/Ricci calculations;
- geodesics;
- curvature-derived regime indicators;
- mappings from eigenvalues/curvature to named tipping elements, probabilities, timescales, and reversibility;
- an Einstein-equation-style analogy.

The dual-number metric/Christoffel paths were incomplete while higher derivatives depended on them, and the climate-risk mappings were heuristic. That makes the live implementation unsafe as a foundation for further work.

## Replacement architecture

Reintroduce as separate layers:

```text
geometry/reference
  generic metric -> connection -> curvature machinery

geometry/metric_candidates
  explicit candidate metric constructions

geometry/representation
  climate variables, units, transforms, coordinate maps

geometry/diagnostics
  coordinate-invariant numerical quantities only

geometry/meta_experiments
  comparisons against conventional baselines

geometry/interpretation
  optional mappings to climate statements, introduced only after separate validation
```

The current verification contract is `docs/GEOMETRY-VERIFICATION.md`, with the preregistered experiment `experiments/geometry-correctness.v1.json` and analytic fixtures under `fixtures/geometry/`.

## Metric candidates are hypotheses

A metric must have an explicit `MethodDescriptor`/candidate identity. Candidate families may include:

- physically derived nondimensional metrics;
- Fisher/information metrics from an explicit likelihood;
- covariance/Mahalanobis metrics;
- learned local metrics;
- diffusion/graph-derived metrics;
- carefully documented hand-designed metrics used only as experimental hypotheses.

No one metric becomes canonical merely because the first implementation used it.

## Reintroduction requirements

Before a climate-specific manifold implementation becomes executable again:

1. the generic reference geometry passes the analytic fixture battery;
2. derivative mechanisms have independent witnesses;
3. conditioning/positive-definiteness/refusal semantics are explicit;
4. unit/coordinate/permutation metamorphic tests exist;
5. the metric candidate has its own identity and assumptions;
6. geometric outputs remain geometric outputs—no probability/tipping labels are emitted by the geometry layer;
7. any GPU implementation passes CPU/reference differential tests on the same immutable fixtures;
8. any climate-usefulness claim has a preregistered baseline comparison and protected evaluation data.

## Removed interpretations

The following are not part of the replacement core unless independently validated later:

- curvature eigenvalue index -> named Earth-system tipping element;
- sigmoid(curvature/eigenvalue) -> physical tipping probability;
- inverse eigenvalue -> physical time-to-collapse;
- arbitrary threshold -> reversibility;
- Einstein-field-equation analogy -> physical climate dynamics.

These ideas may return only as explicitly named experimental hypotheses with evidence contracts, not as default model semantics.
