# Sheaf / cohomology realization contract

Status: architectural and mathematical specification for `climate.sheaf.consistency`.

## Mathematical contract

Sheaf/cohomology terminology is reserved for structures satisfying the defining algebraic laws. Mathematical realization and empirical climate usefulness remain separate claims.

The binding structural requirements are:

1. **Linear cochain maps.** Coboundaries are linear maps between declared cochain groups.
2. **Declared base complex.** Simplices arise from an explicit simplicial/cellular complex or poset rather than arbitrary tuple enumeration.
3. **Cochain-complex identity.** Consecutive coboundaries satisfy `d_(k+1) o d_k = 0` under the declared coefficient system.
4. **Cohomology-derived Betti numbers.** A Betti number is the dimension of a cohomology group over a declared field; thresholded residual counts are not Betti numbers.
5. **Explicit stalks and restrictions.** Climate-data stalk contents, units, missingness, and restriction maps are explicit and restrictions satisfy identity/composition.
6. **Restriction-derived coboundary.** Climate-data cochain differentials are assembled from those restrictions.
7. **Gluing semantics.** Compatible local sections and the global-section space are defined by the restrictions. Averaging or interpolation is a separate reconstruction operation.
8. **Adjunction terminology.** An adjunction requires categories, functors, unit/counit natural transformations, and triangle identities.

## Realized reference scope

### Exact finite-complex and generic sheaf reference

`reference/sheaf_cohomology.py` realizes finite abstract simplicial complexes, exact nerves of declared finite covers, finite-dimensional GF(2) cellular sheaves, functorial restriction maps, block coboundaries, `d² = 0`, global-section compatibility through `ker(d⁰)`, and cohomology dimensions from exact rank arithmetic.

`src/sheaf.rs` independently supplies real-valued degree-zero cellular-sheaf operators on graphs: explicit stalks/restrictions, `d⁰`, compatibility residuals, `L₀ = d⁰ᵀd⁰`, and SVD-derived structural diagnostics. Its scalar identity-restriction special case is also kernel-checked in Lean.

### Provider-backed climate-data reference

`reference/station_temperature_sheaf.py` composes the NCEI GHCN-Daily adapter, the exact finite-cover nerve machinery, pyproj coordinate transformation, NumPy cochain algebra, and a SciPy nearest-neighbor baseline.

The checked-in source fixture is the exact NCEI response for Central Park (`USW00094728`), LaGuardia (`USW00014732`), and JFK (`USW00094789`) for 2024-01-01 through 2024-01-03, identified by:

```text
source_id = ncei.ghcnd.v3
sha256 = 45f8e9a08d61939ca639e599250bdf4fbc8a41986f52d7c7e023f1a947c8bd42
bytes = 6957
records = 9
```

The reference experiment declares a 20 km station-support radius and EPSG:32618 projected coordinates. These are fixture policy, not universal claims about station representativeness. The finite cover is constructed from provider station coordinates under that rule; the resulting three-station fixture contains a 2-simplex.

Vertex and higher-simplex stalk bases contain only daily `TMAX` and `TMIN` variables actually present at every station participating in the simplex. Units are degrees Celsius under the captured metric-unit provider request. Provider quality flags remain explicit. A quality-flagged value can be marked unavailable by declared policy; an absent value is never filled by climatology, interpolation, or a static default.

Restriction maps select the variables shared by face and coface stalks. Their identity/composition semantics are executable, and the resulting oriented real cochains satisfy `d¹d⁰ = 0` on the climate-data fixture.

## Compatibility, gluing, and falsification

The global-section space is `ker(d⁰)`; incompatibility is retained as a residual rather than merged away. On the complete two-variable three-station fixture the global-section dimension is two, corresponding to constant `TMAX` and constant `TMIN` directions. The observed station assignment is not silently promoted to a compatible global section.

Synthetic fault and withholding transforms are explicit derived test operations over the captured provider artifact. They do not replace unavailable source data:

- an injected +10 °C `TMAX` perturbation is recorded in transformation lineage;
- a withheld `TMAX` becomes unavailable and is removed from affected stalk bases rather than imputed;
- provider-QC counts remain independently visible;
- ordinary centered residuals and graph-edge residuals are computed from the same information;
- SciPy nearest-neighbor interpolation is a separate baseline rather than a sheaf restriction or fallback value.

For complete shared temperature stalks, the sheaf compatibility residual equals the ordinary graph residual exactly. That equality is an important negative control: this fixture establishes climate-data sheaf semantics and falsification behavior, not an incremental advantage for sheaf machinery.

## Empirical interpretation boundary

The realized reference does not establish that 20 km is a generally valid station-support scale, that GHCN-D observations are homogenized for every climate task, that a nonzero cohomology class identifies a physical data defect, or that sheaf diagnostics outperform standard QC, residual, graph, or interpolation methods on broader real station networks.

`baseline_and_real_data_incremental_value` remains an open empirical realization obligation in `methods/sheaf-realization.v1.json`. The planning graph is the sole authority for work selection and priority; this document defines only the current mathematical and scientific scope.
