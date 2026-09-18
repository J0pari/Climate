# Climate obligation roadmap

> Generated from `architecture/planning_graph.json`. Do not hand-edit this file.
> The graph is the sole authority for planned work, priority, dependencies, blockers, and completion criteria.

Objective realized state is owned by the module, claim, experiment, hazard, and realization authorities and is rendered separately in `docs/generated/STATUS.md`.

## Planning summary

- Active: 4
- Ready: 14
- Blocked: 6
- Done: 9
- Dropped: 0

## Graph projection

| Obligation | Status | Priority | Resource | Dependencies |
| --- | --- | --- | --- | --- |
| `experimentation.climate_representation_ladder` — Build a climate-representation benchmark ladder | `active` | `P0` | `R2_toolchain_ci` | `experimentation.common_cpu_runtime` |
| `physics.streaming_budget_accounting` — Compose physical kernels through a streaming budget ledger | `active` | `P0` | `R2_toolchain_ci` | `physics.authoritative_flux_recomposition` |
| `sheaf.climate_data_semantics` — Realize scale-native climate-data sheaf semantics | `active` | `P0` | `R2_toolchain_ci` | `data.external_authority_audit`, `data.immutable_observational_projection` |
| `multirepresentation.structure_baselines` — Test multirepresentation structure against simpler baselines | `active` | `P1` | `R2_toolchain_ci` | — |
| `data.global_free_station_federation` — Federate freely accessible worldwide station observations | `ready` | `P0` | `R5_large_data` | `data.external_authority_audit` |
| `diagnostics.feedback_observation_benchmarks` — Ground feedback diagnostics in explicit observations and process definitions | `ready` | `P1` | `R5_large_data` | `data.immutable_observational_projection` |
| `diagnostics.spectral_oscillation_contracts` — Rebuild spectral and oscillation diagnostics from narrow contracts | `ready` | `P1` | `R2_toolchain_ci` | — |
| `geometry.independent_derivative_route` — Add a second independent metric-derivative route | `ready` | `P1` | `R2_toolchain_ci` | — |
| `information_geometry.recovery_validation` — Validate information-geometry primitives on explicit statistical models | `ready` | `P1` | `R2_toolchain_ci` | — |
| `numerics.conditioning_guardrails` — Define library-backed numerical conditioning guardrails | `ready` | `P1` | `R2_toolchain_ci` | — |
| `physics.parameterization_boundaries` — Establish physical parameterization seams and library boundaries | `ready` | `P1` | `R2_toolchain_ci` | `physics.authoritative_flux_recomposition` |
| `physics.vertical_coordinate_contract` — Define pressure, sigma, and hybrid vertical-coordinate semantics | `ready` | `P1` | `R2_toolchain_ci` | `architecture.reference_values` |
| `simulation.reproducible_restart` — Define reproducible integration and restart state | `ready` | `P1` | `R2_toolchain_ci` | `physics.authoritative_flux_recomposition`, `state.typed_physical_state` |
| `adapters.versioned_array_ffi` — Define narrow versioned array FFI contracts | `ready` | `P2` | `R2_toolchain_ci` | `state.typed_physical_state` |
| `experimental.scenario_feasibility_adapter` — Separate climate feasibility from modal truth | `ready` | `P2` | `R2_toolchain_ci` | `data.immutable_observational_projection` |
| `experimental.symmetry_diagnostics` — Separate approximate climate symmetry diagnostics from exact Noether structure | `ready` | `P2` | `R2_toolchain_ci` | — |
| `experimental.teleconnection_encoding` — Test climate encodings for exact ultrametric primitives | `ready` | `P2` | `R2_toolchain_ci` | — |
| `statistics.model_selection_dependence` — Verify model-selection criteria under climate dependence | `ready` | `P2` | `R2_toolchain_ci` | — |
| `sheaf.global_station_scale` — Exercise sheaf substrate at worldwide station scale | `blocked` | `P0` | `R5_large_data` | `sheaf.climate_data_semantics`, `data.global_free_station_federation` |
| `multirepresentation.cost_noise_authority` — Ground multirepresentation weighting in real cost and noise authority | `blocked` | `P1` | `R5_large_data` | `data.immutable_observational_projection` |
| `orchestration.commons_read_execution` — Exercise read-only Commons execution lineage | `blocked` | `P1` | `R4_integrated_system` | — |
| `experimental.clifford_task` — Give Clifford representations a falsifiable climate task | `blocked` | `P2` | `R2_toolchain_ci` | `diagnostics.spectral_oscillation_contracts` |
| `gpu.justified_geometry_acceleration` — Prepare GPU differential execution only for a justified workload | `blocked` | `P2` | `R3_cuda_device` | `geometry.nonlinear_coordinate_witness`, `geometry.independent_derivative_route` |
| `optimization.climate_objective_contracts` — Isolate climate-specific optimization objectives | `blocked` | `P2` | `R2_toolchain_ci` | `information_geometry.recovery_validation` |
| `architecture.layered_configuration` — Define layered runtime configuration contracts | `done` | `P0` | `R2_toolchain_ci` | `architecture.reference_values` |
| `architecture.legacy_source_depletion` — Deplete and delete legacy root monoliths | `done` | `P0` | `R2_toolchain_ci` | `architecture.reference_values` |
| `architecture.reference_values` — Focused atmospheric reference-value authority | `done` | `P0` | `R2_toolchain_ci` | — |
| `data.external_authority_audit` — Externalize hardcoded climate data and assumptions | `done` | `P0` | `R1_portable_cpu` | — |
| `data.immutable_observational_projection` — Create one immutable observational or reanalysis projection | `done` | `P0` | `R2_toolchain_ci` | `data.external_authority_audit`, `state.typed_physical_state` |
| `experimentation.common_cpu_runtime` — Run one contract-valid CPU experiment end to end | `done` | `P0` | `R1_portable_cpu` | — |
| `geometry.nonlinear_coordinate_witness` — Verify nonlinear coordinate covariance | `done` | `P0` | `R2_toolchain_ci` | — |
| `physics.authoritative_flux_recomposition` — Recompose pressure-coordinate physics around authoritative fluxes | `done` | `P0` | `R2_toolchain_ci` | `architecture.reference_values` |
| `state.typed_physical_state` — Define typed physical-state boundaries | `done` | `P1` | `R2_toolchain_ci` | `physics.authoritative_flux_recomposition` |

Node summaries, blockers, completion criteria, and evidence paths live only in `architecture/planning_graph.json` so this projection cannot become a second planning surface.
