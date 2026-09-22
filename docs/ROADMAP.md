# Climate obligation roadmap

> Generated planning projection. Do not hand-edit.
> Declared planning authority: `architecture/planning_graph.json`.
> Planning-authority fingerprint: `git-tree-sha1:f8abcba3c978bcb915444f0500316fc86a89c2bd`.

Freshness means this projection matches its declared planning authority inputs in the checkout being inspected. Repository state is commit-scoped: movement of `main` invalidates cached conclusions until the current commit is re-oriented and the relevant verification is explicitly rerun.

Repository-local research state is summarized in `docs/generated/STATE.md`.

External-integration obligations bind native external capabilities while keeping Climate-specific scientific semantics local; completion cannot be satisfied by a local shadow implementation of the upstream capability.

## Planning summary

- Active: 4
- Ready: 15
- Blocked: 13
- Done: 18
- Dropped: 0

## Graph projection

| Obligation | Status | Priority | Resource | Dependencies |
| --- | --- | --- | --- | --- |
| `data.global_free_station_federation` — Federate freely accessible worldwide station observations | `active` | `P0` | `R5_large_data` | `data.external_authority_audit` |
| `integration.native_capability_boundaries` — Bind external native capabilities without shadow implementations | `active` | `P0` | `R2_portable_toolchain` | — |
| `multirepresentation.structure_baselines` — Test multirepresentation structure against simpler baselines | `active` | `P1` | `R2_portable_toolchain` | — |
| `numerics.conditioning_guardrails` — Define library-backed numerical conditioning guardrails | `active` | `P1` | `R2_portable_toolchain` | — |
| `integration.aimip_production_archive` — Use the public AIMIP production archive | `ready` | `P0` | `R5_large_data` | — |
| `integration.climate_ref_production_cmip` — Use Climate-REF on production CMIP data | `ready` | `P0` | `R4_integrated_system` | — |
| `architecture.reproducible_verification_environment` — Resolve a complete repository verification environment | `ready` | `P1` | `R2_portable_toolchain` | — |
| `diagnostics.feedback_observation_benchmarks` — Ground feedback diagnostics in explicit observations and process definitions | `ready` | `P1` | `R5_large_data` | `data.immutable_observational_projection` |
| `diagnostics.spectral_oscillation_contracts` — Rebuild spectral and oscillation diagnostics from narrow contracts | `ready` | `P1` | `R2_portable_toolchain` | — |
| `physics.parameterization_boundaries` — Establish physical parameterization seams and library boundaries | `ready` | `P1` | `R2_portable_toolchain` | `physics.authoritative_flux_recomposition` |
| `physics.vertical_coordinate_contract` — Define pressure, sigma, and hybrid vertical-coordinate semantics | `ready` | `P1` | `R2_portable_toolchain` | `architecture.reference_values` |
| `sheaf.observation_uncertainty_compatibility` — Define uncertainty-aware sheaf observation compatibility | `ready` | `P1` | `R2_portable_toolchain` | `sheaf.climate_data_semantics`, `data.immutable_observational_projection` |
| `simulation.reproducible_restart` — Define reproducible integration and restart state | `ready` | `P1` | `R2_portable_toolchain` | `physics.authoritative_flux_recomposition`, `state.typed_physical_state` |
| `validation.cross_model_evidence_transport` — Make cross-model evidence transport explicit and testable | `ready` | `P1` | `R2_portable_toolchain` | `experimentation.claim_evidence_vertical_slice` |
| `adapters.versioned_array_ffi` — Define narrow versioned array FFI contracts | `ready` | `P2` | `R2_portable_toolchain` | `state.typed_physical_state` |
| `experimental.scenario_feasibility_adapter` — Separate climate feasibility from modal truth | `ready` | `P2` | `R2_portable_toolchain` | `data.immutable_observational_projection` |
| `experimental.symmetry_diagnostics` — Separate approximate climate symmetry diagnostics from exact Noether structure | `ready` | `P2` | `R2_portable_toolchain` | — |
| `experimental.teleconnection_encoding` — Test climate encodings for exact ultrametric primitives | `ready` | `P2` | `R2_portable_toolchain` | — |
| `statistics.model_selection_dependence` — Verify model-selection criteria under climate dependence | `ready` | `P2` | `R2_portable_toolchain` | — |
| `integration.aqua_production_reader` — Use AQUA production data access | `blocked` | `P0` | `R4_integrated_system` | `integration.climate_ref_production_cmip` |
| `integration.esmvalcore_production_preprocessing` — Use ESMValCore production preprocessing | `blocked` | `P0` | `R4_integrated_system` | `integration.climate_ref_production_cmip` |
| `orchestration.codespaces_cpu_experiment_campaign` — Turn included Codespaces CPU into receipted experiments | `blocked` | `P0` | `R2_portable_toolchain` | — |
| `sheaf.global_station_scale` — Exercise sheaf substrate at worldwide station scale | `blocked` | `P0` | `R5_large_data` | `sheaf.climate_data_semantics`, `data.global_free_station_federation` |
| `experimentation.multifidelity_response_interrogation` — Use cheap model exploration to choose discriminating high-fidelity experiments | `blocked` | `P1` | `R5_large_data` | `integration.native_capability_boundaries`, `orchestration.external_artifact_evaluation`, `validation.cross_model_evidence_transport` |
| `integration.commons_storage_distribution` — Resolve Commons storage distribution for public Climate consumption | `blocked` | `P1` | `R4_integrated_system` | — |
| `multirepresentation.cost_noise_authority` — Ground multirepresentation weighting in real cost and noise authority | `blocked` | `P1` | `R5_large_data` | `data.immutable_observational_projection` |
| `orchestration.commons_read_execution` — Exercise read-only Commons execution lineage | `blocked` | `P1` | `R4_integrated_system` | — |
| `orchestration.external_artifact_evaluation` — Evaluate external model artifacts through Climate-owned tasks | `blocked` | `P1` | `R4_integrated_system` | `orchestration.commons_read_execution` |
| `data.logical_materialization_identity` — Separate logical observation identity from storage realization | `blocked` | `P2` | `R2_portable_toolchain` | `data.global_free_station_federation` |
| `experimental.clifford_task` — Give Clifford representations a falsifiable climate task | `blocked` | `P2` | `R2_portable_toolchain` | `diagnostics.spectral_oscillation_contracts` |
| `gpu.justified_geometry_acceleration` — Prepare GPU differential execution only for a justified workload | `blocked` | `P2` | `R3_cuda_device` | `geometry.nonlinear_coordinate_witness`, `geometry.independent_derivative_route` |
| `multirepresentation.neighborhood_geometry` — Make neighborhood geometry a first-class representation evaluator | `blocked` | `P2` | `R2_portable_toolchain` | `multirepresentation.structure_baselines`, `experimentation.claim_evidence_vertical_slice` |
| `architecture.layered_configuration` — Define layered runtime configuration contracts | `done` | `P0` | `R2_portable_toolchain` | `architecture.reference_values` |
| `architecture.legacy_source_depletion` — Deplete and delete legacy root monoliths | `done` | `P0` | `R2_portable_toolchain` | `architecture.reference_values` |
| `architecture.reference_values` — Focused atmospheric reference-value authority | `done` | `P0` | `R2_portable_toolchain` | — |
| `data.external_authority_audit` — Externalize hardcoded climate data and assumptions | `done` | `P0` | `R1_portable_cpu` | — |
| `data.immutable_observational_projection` — Create one immutable observational or reanalysis projection | `done` | `P0` | `R2_portable_toolchain` | `data.external_authority_audit`, `state.typed_physical_state` |
| `experimentation.claim_evidence_vertical_slice` — Close one claim-to-evidence scientific loop | `done` | `P0` | `R2_portable_toolchain` | `experimentation.failure_provenance` |
| `experimentation.climate_representation_ladder` — Build a climate-representation benchmark ladder | `done` | `P0` | `R2_portable_toolchain` | `experimentation.common_cpu_runtime` |
| `experimentation.common_cpu_runtime` — Run one contract-valid CPU experiment end to end | `done` | `P0` | `R1_portable_cpu` | — |
| `experimentation.failure_provenance` — Persist structured provenance for failed experiment execution | `done` | `P0` | `R2_portable_toolchain` | `experimentation.common_cpu_runtime` |
| `geometry.nonlinear_coordinate_witness` — Verify nonlinear coordinate covariance | `done` | `P0` | `R2_portable_toolchain` | — |
| `physics.authoritative_flux_recomposition` — Recompose pressure-coordinate physics around authoritative fluxes | `done` | `P0` | `R2_portable_toolchain` | `architecture.reference_values` |
| `physics.streaming_budget_accounting` — Compose physical kernels through a streaming budget ledger | `done` | `P0` | `R2_portable_toolchain` | `physics.authoritative_flux_recomposition` |
| `sheaf.climate_data_semantics` — Realize scale-native climate-data sheaf semantics | `done` | `P0` | `R2_portable_toolchain` | `data.external_authority_audit`, `data.immutable_observational_projection` |
| `experimentation.runtime_protocol_decomposition` — Decompose experiment runtime into composable execution protocols | `done` | `P1` | `R2_portable_toolchain` | `experimentation.claim_evidence_vertical_slice` |
| `geometry.independent_derivative_route` — Add a second independent metric-derivative route | `done` | `P1` | `R2_portable_toolchain` | — |
| `information_geometry.recovery_validation` — Validate information-geometry primitives on explicit statistical models | `done` | `P1` | `R2_portable_toolchain` | — |
| `state.typed_physical_state` — Define typed physical-state boundaries | `done` | `P1` | `R2_portable_toolchain` | `physics.authoritative_flux_recomposition` |
| `optimization.climate_objective_contracts` — Isolate climate-specific optimization objectives | `done` | `P2` | `R2_portable_toolchain` | `information_geometry.recovery_validation` |

Summaries, blockers, completion criteria, and evidence paths remain in `architecture/planning_graph.json`.
