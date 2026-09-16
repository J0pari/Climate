# Climate repository contract

This repository is an **experimental climate-methods research workspace**. It contains conventional climate-science components, incomplete numerical infrastructure, and deliberately unusual mathematical hypotheses. The repository must preserve that distinction rather than presenting all code as one validated climate model.

This file is the binding contributor/agent contract. `README.md` is descriptive and may lag it. The architectural documents under `docs/` define the intended direction; implementation may be incomplete until the staged gates in `docs/ROADMAP.md` are met.

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

## 7. Data rules

Climate data artifacts should move toward CF-compliant metadata and explicit provenance. Never silently substitute missing observations with climatological/default values in a path used for validation. Missingness, imputation, regridding, temporal aggregation, unit conversion, detrending, anomaly baselines, and quality-control exclusions are part of the experiment definition.

## 8. Current repository status

The current `main` tree is not a coherent build workspace. In particular, build metadata refers to a `CORE/` hierarchy and test/config paths that do not exist in the current flat tree. Do not "fix" that by fabricating empty directories or moving files before the target package/layout plan is agreed and migration tests exist.

Several modules already label themselves unvalidated or contain explicit placeholders. Preserve those warnings until evidence justifies changing them.

## 9. Commons-facing behavior

Until the gates in `docs/COMMONS-INTEGRATION.md` are satisfied, Climate should be treated by Commons as **experimental / observe-only**.

The first integration milestone is read-only inspection and evidence capture. Execution comes later behind a sandbox. Write/PR authority comes after binding contracts, reproducible gates, and immutable receipts exist.

## 10. Read before substantive changes

- `docs/ARCHITECTURE.md`
- `docs/VALIDATION-AND-EVIDENCE.md`
- `docs/META-EXPERIMENTATION.md`
- `docs/COMMONS-INTEGRATION.md`
- `docs/ROADMAP.md`

When these documents conflict, prefer the more specific scientific/validation constraint over implementation convenience. Amend the spec deliberately rather than bypassing it in code.
