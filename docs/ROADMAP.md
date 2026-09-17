# Climate obligation roadmap

> Generated from `architecture/planning_graph.json`. Do not hand-edit this file.
> The graph is the sole authority for planned work, priority, dependencies, blockers, and completion criteria.

Objective realized state is owned by the module, claim, experiment, hazard, and realization authorities and is rendered separately in `docs/generated/STATUS.md`.

## Planning summary

- Active: 1
- Ready: 15
- Blocked: 3
- Done: 1
- Dropped: 0

## Graph projection

| Obligation | Status | Priority | Resource | Dependencies |
| --- | --- | --- | --- | --- |
| `architecture.legacy_source_depletion` — Deplete and delete legacy root monoliths | `active` | `P0` | `R2_toolchain_ci` | `architecture.reference_values` |
| `data.immutable_observational_projection` — Create one immutable observational or reanalysis projection | `ready` | `P0` | `R5_large_data` | `state.typed_physical_state` |
| `geometry.nonlinear_coordinate_witness` — Verify nonlinear coordinate covariance | `ready` | `P0` | `R2_toolchain_ci` | — |
| `physics.authoritative_flux_recomposition` — Recompose pressure-coordinate physics around authoritative fluxes | `ready` | `P0` | `R2_toolchain_ci` | `architecture.reference_values` |
| `sheaf.climate_data_semantics` — Realize station-cover sheaf semantics | `ready` | `P0` | `R2_toolchain_ci` | — |
| `diagnostics.spectral_oscillation_contracts` — Rebuild spectral and oscillation diagnostics from narrow contracts | `ready` | `P1` | `R2_toolchain_ci` | — |
| `geometry.independent_derivative_route` — Add a second independent metric-derivative route | `ready` | `P1` | `R2_toolchain_ci` | — |
| `information_geometry.recovery_validation` — Validate information-geometry primitives on explicit statistical models | `ready` | `P1` | `R2_toolchain_ci` | — |
| `physics.parameterization_boundaries` — Establish physical parameterization seams and library boundaries | `ready` | `P1` | `R2_toolchain_ci` | `physics.authoritative_flux_recomposition` |
| `state.typed_physical_state` — Define typed physical-state boundaries | `ready` | `P1` | `R2_toolchain_ci` | `physics.authoritative_flux_recomposition` |
| `adapters.versioned_array_ffi` — Define narrow versioned array FFI contracts | `ready` | `P2` | `R2_toolchain_ci` | `state.typed_physical_state` |
| `experimental.clifford_task` — Give Clifford representations a falsifiable climate task | `ready` | `P2` | `R2_toolchain_ci` | `diagnostics.spectral_oscillation_contracts` |
| `experimental.scenario_feasibility_adapter` — Separate climate feasibility from modal truth | `ready` | `P2` | `R2_toolchain_ci` | `data.immutable_observational_projection` |
| `experimental.symmetry_diagnostics` — Separate approximate climate symmetry diagnostics from exact Noether structure | `ready` | `P2` | `R2_toolchain_ci` | — |
| `experimental.teleconnection_encoding` — Test climate encodings for exact ultrametric primitives | `ready` | `P2` | `R2_toolchain_ci` | — |
| `optimization.climate_objective_contracts` — Isolate climate-specific optimization objectives | `ready` | `P2` | `R2_toolchain_ci` | `information_geometry.recovery_validation` |
| `multirepresentation.cost_noise_authority` — Ground multirepresentation weighting in real cost and noise authority | `blocked` | `P1` | `R5_large_data` | `data.immutable_observational_projection` |
| `orchestration.commons_read_execution` — Exercise read-only Commons execution lineage | `blocked` | `P1` | `R4_integrated_system` | — |
| `gpu.justified_geometry_acceleration` — Prepare GPU differential execution only for a justified workload | `blocked` | `P2` | `R3_cuda_device` | `geometry.nonlinear_coordinate_witness`, `geometry.independent_derivative_route` |
| `architecture.reference_values` — Focused atmospheric reference-value authority | `done` | `P0` | `R2_toolchain_ci` | — |

Node summaries, blockers, completion criteria, and evidence paths live only in `architecture/planning_graph.json` so this projection cannot become a second planning surface.
