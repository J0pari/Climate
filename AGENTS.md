# Climate repository contract

This repository is an **experimental climate-methods research workspace**. It contains conventional climate-science components, incomplete numerical infrastructure, and deliberately unusual mathematical hypotheses. The repository must preserve that distinction rather than presenting all code as one validated climate model.

This file is the binding contributor/agent contract. `README.md` is descriptive. `docs/ARCHITECTURE.md` and method-specific specifications define intended semantic structure; `architecture/planning_graph.json` is the sole authority for planned work, priorities, dependencies, blockers, and completion criteria; `docs/ROADMAP.md` is its generated human-readable projection; `docs/generated/STATUS.md` projects objective present-state facts from machine-readable realization authorities. Non-binding surveys and research notes do not override those documents. Implementation may be incomplete relative to intended architecture without changing the intended contract.

## 1. Non-negotiable rules

1. **Registration is not validation.** A method may exist in the repository without being scientifically supported.
2. **Running is not verification.** A smoke test only proves that a path executes under the tested environment.
3. **Verification is not validation.** Numerical correctness against equations/test problems does not establish that the equations or interpretation represent climate reality.
4. **Validation is scoped.** Evidence for one variable, region, timescale, dataset, model family, or regime must not be generalized without a declared test.
5. **Novelty is neither a defect nor evidence.** Unusual methods must receive fair baselines and falsifiable tests; they receive no exemption from them.
6. **Physical, statistical, and interpretive claims are distinct.** Do not turn a numerical indicator into a physical probability, tipping threshold, causal mechanism, irreversibility claim, or confidence score without an explicitly validated mapping.
7. **No silent placeholders.** Stubs, heuristic constants, synthetic data, fallback values, incomplete derivatives, approximate solvers, and unimplemented branches must be machine-discoverable or clearly labeled at the point of use.
8. **No architecture-by-prose.** Claimed paths, commands, interfaces, datasets, and generated artifacts require machine checks. Structural drift is a defect, not precedent.
9. **Reproducibility is part of correctness.** Results intended as evidence must bind code revision, method/config versions, dataset identities, preprocessing, random seeds, environment, command, artifacts, and metric definitions.
10. **Commons is a control/evidence boundary, not a scientific authority.** Commons may schedule, fingerprint, trace, and compare Climate experiments. It must not convert experimental Climate output into stronger evidence classes than Climate earned.
11. **Performance is not scientific evidence by itself.** A faster GPU path does not strengthen a scientific claim unless it preserves the declared numerical semantics and passes the same validation contract.
12. **Execution identity is fail-closed.** CPU, fake/compatibility MPI or NCCL, reduced precision, tensor-core alternatives, approximate solvers, and alternate algorithms are distinct implementations. They may execute only when explicitly requested under their own identity. If the requested implementation/backend/precision/resource class is unavailable, the execution is ineligible; it must not substitute another implementation and retain success semantics.
13. **Unavailable resources remain explicit gates.** Compile success, static inspection, mocks, shims, or prose must not stand in for execution on a resource that the claim actually depends on. CUDA correctness needs a CUDA device; integrated scheduling needs the integrated system; real-data generalization needs the declared data.
14. **Preserve semantic information, not incidental syntax.** Refactors may substantially change layout, control flow, abstractions, or language boundaries when that increases technical accuracy, composability, falsifiability, or scientific meaning. Do not preserve brittle structure merely because it is old.
15. **No semantic smoothing.** A migration may simplify syntax, but it must not collapse scientifically meaningful distinctions such as missing vs zero, undefined vs stable, unavailable vs failed, heuristic vs measured, proposal vs evidence, or policy threshold vs physical tipping claim. If compatibility code conflates states, canonical code should split them and document the compatibility difference where that difference remains operationally relevant.
16. **Noncanonical source is temporary mining material, not a permanent architecture tier.** Mine unique domain semantics into canonical code, contracts, references, tests, or planning obligations. Do not migrate generic machinery that belongs in maintained libraries. Once useful semantics are represented elsewhere and no live dependency remains, delete the depleted source; Git history is the archive. `docs/LEGACY-DELETION.md` defines the binding depletion criteria.
17. **Abstraction must buy epistemic or engineering leverage.** Do not add wrapper types, indirection, or framework ceremony solely for stylistic purity. Use meaningfully different constructs—typed states, algebraic variants, declarative registries, generated views, staged pipelines, independent references, or property tests—when they make invalid states harder to represent, expose uncertainty, deepen the model, or enable stronger witnesses.
18. **Incomplete mathematics must expose obligations, not borrow finished names.** It is acceptable for an experimental method to have open realization obligations analogous to proof `sorry`s. The statement and definitions must still be correct. Promotion terms such as cohomology, Betti number, adjunction, exact conservation law, or verified solver require the defining laws to have executable witnesses. Pin the intended claim so progress cannot be faked by weakening the statement; reduce the open obligations over time instead.
19. **Do not weaken a useful mathematical contract merely because the implementation is weak.** If the stronger mathematical object is scientifically/computationally useful and tractable, implement it correctly. Rename/retype downward only when the narrower object is itself the superior reusable abstraction for a real task.
20. **Separate realized-state authority from planning authority.** Facts represented in module/claim/experiment/hazard/realization registries flow into generated status. Human planning judgment is explicit data in `architecture/planning_graph.json`, not an inferred status field or a parallel prose backlog. Do not commit hand-authored continuation snapshots or one-time agent handoffs as repository documents; transient continuation context belongs in the conversation/system context that needs it.
21. **Commit only to `main`.** Repository edits are made directly on `main` in small coherent commits. Do not create or stage work on feature branches, and do not rewrite or force-update history. Before each write, re-read the current `main` version of every file being edited so concurrent or newly learned design intent is preserved rather than overwritten.
22. **Commit history owns change narration.** Durable documentation and source comments describe the current contract, rationale, assumptions, limitations, and invariants. Do not annotate repository text with edit batches, feature-addition notes, replacement history, or temporal status narration that belongs to version history. Put what changed and why in the commit message.
23. **The planning graph owns TODO concerns.** Durable source and documentation must not maintain TODO/FIXME action queues, priority lists, future-work checklists, or dependency plans. Explain a current limitation at the point where it matters, and represent the work needed to remove that limitation in `architecture/planning_graph.json`. Generated roadmap prose may project graph facts but must not add planning content of its own.

## 2. Claim maturity vocabulary

Use these terms consistently in code, docs, reports, and machine-readable records.

- **concept** — mathematical/scientific idea stated precisely enough to discuss.
- **prototype** — implementation exists but may contain placeholders or lack numerical verification.
- **runnable** — executes reproducibly on a declared fixture/environment.
- **verified** — implementation has passed appropriate numerical/software verification for its declared equations/algorithm (analytic cases, manufactured solutions, invariants, convergence, differential tests, etc.).
- **validated** — output has passed predeclared empirical/process/model validation against independent evidence for a stated domain of applicability.
- **replicated** — a materially independent implementation/dataset/team or pipeline reproduces the relevant result within declared tolerances.
- **decision-eligible** — an explicitly defined downstream policy permits this evidence to affect a decision. This is never implied by validation alone.

Scientific maturity is claim- and experiment-scoped. Module registries may carry a coarse implementation lifecycle or readiness label for structural/build/status purposes, but that label must not be interpreted as scientific validation of every path in the file or as a substitute for claim-specific maturity.

## 3. Evidence classes

When Climate integrates with Commons, map evidence without inflation:

- exact proof or independently checkable identity -> `Formal` only when it actually meets that bar;
- measured real-world/reanalysis observation -> `Observed`;
- controlled numerical model output -> `Simulated`;
- historical replay under an alternative intervention/policy -> `Counterfactual`;
- behavior/performance evaluation -> `Behavioral`;
- hypothesis, proxy, analogy, learned or hand-built indicator without empirical validation -> `Heuristic`.

Verification status is orthogonal to evidence class. A simulated result can be verified as correctly computed while remaining simulated rather than observed.

## 4. Method boundaries

The target architecture separates:

- data acquisition and normalization;
- reference physical/numerical kernels;
- diagnostics and established analysis methods;
- experimental mathematical methods;
- validation/benchmarking;
- orchestration, resource control, provenance, and Commons integration.

Experimental modules must not be wired directly into authoritative physical state transitions merely because they share a state struct. Prefer typed artifacts and explicit adapters.

## 5. Required design for new methods

A new research method should declare, before promotion beyond prototype:

- hypothesis/question;
- mathematical object or algorithm;
- expected domain of applicability;
- required inputs and preprocessing;
- outputs and units/meaning;
- assumptions;
- conventional and strong alternative baselines;
- primary metric(s) and uncertainty treatment;
- negative controls and falsifiers;
- leakage/confounding risks;
- computational resource envelope;
- reproducibility inputs;
- criteria for retain/revise/reject.

Do not add a bespoke orchestration path for each mathematical idea. Methods use the common experiment interfaces in `docs/ARCHITECTURE.md` unless a method-specific contract requires a narrower boundary.

## 6. Numerical and scientific correctness

Numerical kernels should grow tests in roughly this order where applicable:

1. shape/unit/domain checks;
2. exact/analytic identities;
3. manufactured or synthetic known-solution tests;
4. convergence-order tests;
5. conservation/budget closure with explicit tolerances;
6. cross-language or independent differential tests;
7. property/metamorphic tests;
8. benchmark problems;
9. observational/model validation.

Do not claim "machine precision" or exact conservation unless a test records the quantity, normalization, precision, horizon, tolerance, and platform sensitivity.

## 7. GPU and accelerator contract

All evidence-producing CUDA/accelerator work must follow `docs/GPU-ENGINEERING.md`.

Before a new accelerated method is treated as more than an exploratory prototype, declare:

- device-resident buffers and lifetimes;
- expected transfer schedule;
- precision/accumulator/refinement policy;
- conditioning/failure behavior where linear algebra is involved;
- determinism class (`D0`, `D1`, or `D2` as defined in the GPU spec);
- RNG identity and seed derivation if stochastic;
- resource envelope/VRAM estimate;
- reference or differential witness;
- whether setup/transfers are included in performance claims;
- requested execution identity and fail-closed behavior when the requested capability is unavailable.

Host code owns scientific interpretation and claim/evidence state. Device kernels execute bounded numerical transformations; they do not promote outputs into physical probabilities, causal claims, or validated conclusions.

Do not optimize away the reference implementation before the accelerated path has independent witnesses.

## 8. Data rules

Climate data artifacts should move toward CF-compliant metadata and explicit provenance. Never silently substitute missing observations with climatological/default values in a path used for validation. Missingness, imputation, regridding, temporal aggregation, unit conversion, detrending, anomaly baselines, and quality-control exclusions are part of the experiment definition.

## 9. Resource-aware execution planning

`docs/EXECUTION-TOPOLOGY.md` defines the binding resource classes. `architecture/planning_graph.json` owns the dependency/priority graph and completion criteria; `docs/ROADMAP.md` is generated from it and is not an independent planning surface.

Use the execution classes defined in the topology document:

```text
R0_static
R1_portable_cpu
R2_toolchain_ci
R3_cuda_device
R4_integrated_system
R5_large_data
```

A planning node is executable when its semantic prerequisites are satisfied, its required resource is available, and the work can produce a falsifiable result, reusable correctness boundary, or explicit failure. Prefer graph-ready work that removes false authority or semantic hazards, creates reusable independent witnesses, establishes canonical semantic seams, supplies fair baselines, depletes monoliths, or prepares a precise resource-gated experiment.

Do not simulate evidence for a missing resource. Instead preregister the experiment, create independent reference outputs, define tolerances/failure semantics, and leave the resource-dependent graph node explicitly blocked.

## 10. Architecture/source audit discipline

The repository contains static/control checks under `architecture/` plus planted negative witnesses in `tests/architecture/`.

Binding architecture checks are executable repository interfaces rather than prose promises. The canonical invocation surface includes:

```text
python -m unittest discover -s tests/architecture -v
python architecture/check_claims.py
python architecture/check_modules.py
python architecture/check_planning.py
python architecture/render_roadmap.py --check
python architecture/render_status.py --check
python architecture/check_hazards.py
python architecture/check_experiments.py
python architecture/check_sheaf_realization.py
python architecture/check_durable_text.py
python architecture/check_root_layout.py
python architecture/source_gates.py --summary
python architecture/inspect_repository.py --json
```

The integrity checks are binding. A source-gate summary may be used as an audit surface without implying that every compatibility/source-debt class is clean. `python architecture/source_gates.py --strict` becomes binding only for defect classes explicitly adopted into the strict policy. Do not make a broad allowlist permanent merely to turn CI green; either repair the source, narrow the gate to the intended semantic boundary, or record a temporary exception with rationale and expiry.

Objective status derived from module/claim/experiment/realization registries belongs in `docs/generated/STATUS.md`. Planning projection belongs in `docs/ROADMAP.md`. When their authorities change, update the generated projections with:

```text
python architecture/render_status.py --write
python architecture/render_roadmap.py --write
```

Do not hand-edit either generated projection.

## 11. Repository-status authority

`main` is the canonical development line.

Objective repository facts that are already represented by machine-readable authorities belong in `docs/generated/STATUS.md`, not in hand-maintained snapshots elsewhere. Fast-changing execution facts such as exact-head CI outcomes remain execution evidence in GitHub Actions rather than committed status prose.

Architecture documents may describe intended contracts that are not yet fully realized. The planning graph reports intended work; generated status and executable checks report realized structure. None silently redefines another authority. When implementation state and intended design disagree, repair the appropriate layer rather than collapsing one into the other.

## 12. Commons-facing behavior

Climate's Commons posture is **experimental / read execution** through the pinned `work-scheduler/v1` boundary. Commons may enqueue explicitly declared Climate work into Climate-owned artifact output paths, but it does not gain source-write or scientific-promotion authority.

Repository write authority is not granted by the scheduler or evaluation-exchange contracts. If Commons is granted source-write authority, that authority must obey this contract's direct-`main`, small-coherent-commit, current-file-reread, and no-history-rewrite rules; Commons must not introduce a branch/PR workflow that conflicts with them.

For GPU work, Commons owns cross-repository resource leases and run identity at the shared execution boundary; Climate owns device-local numerical execution, streams, layouts, kernels, numerical checkpoint semantics, and hardware-specific correctness tests.
