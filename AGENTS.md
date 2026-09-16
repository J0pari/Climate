# Climate repository contract

This repository is an **experimental climate-methods research workspace**. It contains conventional climate-science components, incomplete numerical infrastructure, and deliberately unusual mathematical hypotheses. The repository must preserve that distinction rather than presenting all code as one validated climate model.

This file is the binding contributor/agent contract. `README.md` is descriptive and may lag it. The architectural documents under `docs/` define the intended direction; implementation may be incomplete until the dependency/resource gates in `docs/ROADMAP.md` and `docs/EXECUTION-TOPOLOGY.md` are met.

## 1. Non-negotiable rules

1. **Registration is not validation.** A method may exist in the repository without being scientifically supported.
2. **Running is not verification.** A smoke test only proves that a path executes under the tested environment.
3. **Verification is not validation.** Numerical correctness against equations/test problems does not establish that the equations or interpretation represent climate reality.
4. **Validation is scoped.** Evidence for one variable, region, timescale, dataset, model family, or regime must not be generalized without a declared test.
5. **Novelty is neither a defect nor evidence.** Unusual methods must receive fair baselines and falsifiable tests; they receive no exemption from them.
6. **Physical, statistical, and interpretive claims are distinct.** Do not turn a numerical indicator into a physical probability, tipping threshold, causal mechanism, irreversibility claim, or confidence score without an explicitly validated mapping.
7. **No silent placeholders.** Stubs, heuristic constants, synthetic data, fallback values, incomplete derivatives, approximate solvers, and unimplemented branches must be machine-discoverable or clearly labeled at the point of use.
8. **No architecture-by-prose.** Claimed paths, commands, interfaces, datasets, and generated artifacts must eventually be machine checked. Current structural drift is a known defect, not precedent.
9. **Reproducibility is part of correctness.** Results intended as evidence must bind code revision, method/config versions, dataset identities, preprocessing, random seeds, environment, command, artifacts, and metric definitions.
10. **Commons is a control/evidence boundary, not a scientific authority.** Commons may schedule, fingerprint, trace, and compare Climate experiments. It must not convert experimental Climate output into stronger evidence classes than Climate earned.
11. **Performance is not scientific evidence by itself.** A faster GPU path does not strengthen a scientific claim unless it preserves the declared numerical semantics and passes the same validation contract.
12. **Fallbacks are distinct implementations.** CPU, fake/compatibility MPI or NCCL, reduced precision, tensor-core fallbacks, approximate solvers, and alternate algorithms must not masquerade as the requested implementation in evidence-producing runs.
13. **Unavailable resources remain explicit gates.** Compile success, static inspection, mocks, shims, or prose must not stand in for execution on a resource that the claim actually depends on. CUDA correctness needs a CUDA device; integrated scheduling needs the integrated system; real-data generalization needs the declared data.

## 2. Claim maturity vocabulary

Use these terms consistently in code, docs, reports, and machine-readable records.

- **concept** — mathematical/scientific idea stated precisely enough to discuss.
- **prototype** — implementation exists but may contain placeholders or lack numerical verification.
- **runnable** — executes reproducibly on a declared fixture/environment.
- **verified** — implementation has passed appropriate numerical/software verification for its declared equations/algorithm (analytic cases, manufactured solutions, invariants, convergence, differential tests, etc.).
- **validated** — output has passed predeclared empirical/process/model validation against independent evidence for a stated domain of applicability.
- **replicated** — a materially independent implementation/dataset/team or pipeline reproduces the relevant result within declared tolerances.
- **decision-eligible** — an explicitly defined downstream policy permits this evidence to affect a decision. This is never implied by validation alone.

A module may hold different maturity states for different claims. Store maturity on claims/experiments, not as one flattering label for an entire file.

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

Do not add a bespoke orchestration path for each mathematical idea. Methods should eventually conform to the common experiment interfaces in `docs/ARCHITECTURE.md`.

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
- actual fallback identity if the preferred capability is absent.

Host code owns scientific interpretation and claim/evidence state. Device kernels execute bounded numerical transformations; they do not promote outputs into physical probabilities, causal claims, or validated conclusions.

Do not optimize away the reference implementation before the accelerated path has independent witnesses.

## 8. Data rules

Climate data artifacts should move toward CF-compliant metadata and explicit provenance. Never silently substitute missing observations with climatological/default values in a path used for validation. Missingness, imputation, regridding, temporal aggregation, unit conversion, detrending, anomaly baselines, and quality-control exclusions are part of the experiment definition.

## 9. Resource-aware execution planning

`docs/EXECUTION-TOPOLOGY.md` is binding for work selection.

The numbered phases in `docs/ROADMAP.md` describe dependencies and maturity targets; they are **not** a requirement to finish one phase globally before beginning all work in the next.

Treat the project as a dependency/resource graph. A blocked CUDA, integrated-system, or large-data node must not block independent static, portable-CPU, reference-math, contract, baseline, or CI-toolchain work.

Use the execution classes defined in the topology document:

```text
R0_static
R1_portable_cpu
R2_toolchain_ci
R3_cuda_device
R4_integrated_system
R5_large_data
```

Prefer ready work that removes substantial uncertainty, creates a reusable boundary, or manufactures an executable question for a currently unavailable resource.

Do not simulate evidence for a missing resource. Instead preregister the experiment, create independent reference outputs, define tolerances/failure semantics, and leave the resource-dependent edge explicitly unsatisfied.

## 10. Architecture/source audit discipline

The repository contains static/control checks under `architecture/` plus planted negative witnesses in `tests/architecture/`.

During the current migration stage:

```text
python -m unittest discover -s tests/architecture -v
python architecture/check_claims.py
python architecture/check_modules.py
python architecture/check_experiments.py
python architecture/source_gates.py --summary
python architecture/inspect_repository.py --json
```

The integrity checks are binding. The source-gate summary is currently an **audit**, not a cleanliness assertion, because legacy source intentionally contains known debt.

`python architecture/source_gates.py --strict` becomes binding only as individual debt classes are retired under the roadmap. Do not make a broad allowlist permanent merely to turn CI green; either repair the source, narrow the gate to the intended semantic boundary, or record a temporary exception with rationale and expiry.

## 11. Current repository status

The current `main` tree is not a coherent build workspace. In particular, build metadata refers to a `CORE/` hierarchy and test/config paths that do not exist in the current flat tree. Do not "fix" that by fabricating empty directories or moving files before the target package/layout plan is agreed and migration tests exist.

Several modules already label themselves unvalidated or contain explicit placeholders. Preserve those warnings until evidence justifies changing them.

## 12. Commons-facing behavior

Until the gates in `docs/COMMONS-INTEGRATION.md` are satisfied, Climate should be treated by Commons as **experimental / observe-only**.

The first integration milestone is read-only inspection and evidence capture. Execution comes later behind a sandbox. Write/PR authority comes after binding contracts, reproducible gates, and immutable receipts exist.

For GPU work, Commons should eventually own cross-repository resource leases and run identity; Climate owns device-local numerical execution, streams, layouts, kernels, numerical checkpoint semantics, and hardware-specific correctness tests.

## 13. Read before substantive changes

- `docs/ARCHITECTURE.md`
- `docs/EXECUTION-TOPOLOGY.md`
- `docs/GPU-ENGINEERING.md`
- `docs/VALIDATION-AND-EVIDENCE.md`
- `docs/META-EXPERIMENTATION.md`
- `docs/COMMONS-INTEGRATION.md`
- `docs/ROADMAP.md`
- `docs/GEOMETRY-VERIFICATION.md` for geometric/manifold work

When these documents conflict, prefer the more specific scientific/validation constraint over implementation convenience. Amend the spec deliberately rather than bypassing it in code.
