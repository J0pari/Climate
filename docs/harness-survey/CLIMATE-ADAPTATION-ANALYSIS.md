# Climate harness adaptation analysis

Status: **non-binding adaptation proposal**.

This document translates the preceding KanForge/LLM-Trader survey into Climate-specific failure surfaces. It does **not** yet change the repository contract. Its purpose is to identify where generic mechanisms fit Climate, where they do not, and which existing Climate controls are currently too weak or too coarse.

The source surveys remain the evidence record:

- `docs/CROSS-REPO-HARNESS-SURVEY.md`
- `docs/harness-survey/DEEP-DIVE-NOTES-01.md`
- `docs/harness-survey/DEEP-DIVE-NOTES-02.md`

## 1. Immediate conclusion

Climate should not copy either source repository's domain architecture.

The transferable mechanism is narrower:

> creative, heuristic, cached, accelerated, or agent-authored paths may propose work; only explicitly designated authority paths may change scientific/evidentiary state.

For Climate, the highest-value adaptation is therefore not a new agent framework. It is to make **authority, state distinctions, reachability, and evidence promotion executable** so that an agent cannot obtain an apparently successful result by taking a semantically weaker route than the one the claim requires.

## 2. First defect to fix: the architecture inventory is already stale

The broad destructive sanitation experiment was reverted, restoring:

- `climate_oscillation_monitor.f90`
- `climate_spectral_analysis.f90`
- `climate_manifold.rs`

but the machine-readable control surfaces were not fully repaired afterward.

Current examples:

- `architecture/modules.json` still describes `climate_oscillation_monitor.f90` as an "explicit unavailable stub" whose legacy implementation was removed;
- `architecture/semantic_hazards.json` still marks the oscillation hazard as `resolved` / `stubbed`;
- the spectral hazard is likewise still marked `resolved` / `stubbed` even though the source implementation is present again;
- the geometry hazard still proposes `replace_with_stub`, the coarse disposition already rejected as inappropriate for a large mixed-quality source file.

This is not just documentation debt. These files are intended to guide later agents and evidence tooling, so stale machine-readable state is itself a harness defect.

### Required direction

A registry should not be trusted merely because its JSON parses and referenced file exists. The registry needs evidence that its **semantic assertions still match the source state**.

Potential mechanisms, to evaluate before implementation:

- source-review fingerprint on manually asserted implementation state;
- symbol-level or region-level hazard scopes rather than file-only scopes;
- executable assertions for states such as `stub`, `unavailable`, or `resolved`;
- resolution witness IDs rather than manually setting `status: resolved`;
- generated projections for facts that can be derived from source/tests instead of restating them manually.

The right answer may combine these rather than rely on a source hash alone.

## 3. File-level maturity is too coarse

`AGENTS.md` says:

> A module may hold different maturity states for different claims. Store maturity on claims/experiments, not as one flattering label for an entire file.

Yet `architecture/modules.json` currently stores `maturity` and `evidence_eligible` on every source file, and `architecture/check_modules.py` enforces promotion rules on those file-level fields.

This creates two failure modes:

1. one safe/verified function could accidentally confer status on unrelated prototype paths in the same large file;
2. one bad path could force an entire scientifically useful file to remain globally unusable, encouraging coarse deletion or rewrites.

Both are undesirable.

### Proposed partition

Keep the module inventory for structural ownership and coarse risk discovery, but move scientific authority to narrower records:

```text
source module
    -> contains implementation units

implementation unit
    -> implements one or more method versions

method version
    -> participates in experiment specs

experiment/run
    -> emits evidence records

evidence records
    -> bear on claim versions

claim promotion
    -> depends on required evidence/witness policy
```

A module-level flag may remain as a conservative summary, but it should be **derived from reachable implementation/method state**, not be the authority itself.

## 4. Hazard records need scope and reachability

The current semantic hazard ledger is path-oriented. That is better than prose-only debt, but too coarse for Climate's large mixed-purpose files.

A hazard should move toward fields such as:

```text
hazard_id
source_path
scope_kind             # symbol | procedure | type | branch | file | generated surface
scope_anchor
hazard_class
state
activation_condition
blocks_method_ids
blocks_claim_ids
blocks_evidence_classes
resolution_witness_ids
reviewed_source_identity
notes
```

The important addition is **reachability**.

An S4 placeholder in an unused prototype procedure should not automatically invalidate an independently verified procedure elsewhere in the same file. Conversely, an S4 path that is reachable from an evidence-producing method must block that method even if the placeholder is hidden behind a fallback branch.

This permits surgical quarantine rather than file deletion.

## 5. `resolved` should be earned, not manually asserted

The survey repeatedly found that strong harnesses make the authority transition harder than the proposal transition.

Climate's hazard ledger currently permits a human/agent to type:

```json
"status": "resolved"
```

without the ledger itself naming the witness that makes that true.

That is too easy for an agent to satisfy cosmetically.

A future `resolved`/`mitigated` state should require a resolution record that points to appropriate witnesses, for example:

- exact source transformation or removed branch;
- negative test proving the old failure mode is now rejected;
- positive test proving the intended supported path still works;
- independent numerical/reference witness where relevant;
- explicit implementation identity if a fallback was separated rather than removed.

The checker, not prose, should decide whether the transition is admissible.

## 6. Agent-authored proposals must not directly promote authority state

Climate already has claim, method, experiment, module, and hazard registries. Those are useful places for an agent to **propose** changes, but not sufficient places for it to assert its own success.

A robust authority graph would distinguish:

```text
proposal:
  "this method may now be verified"

from

promotion receipt:
  "the required independent witnesses resolved and the promotion policy accepted them"
```

An agent may write code, experiments, candidate claim text, and proposed registry deltas. It should not be able to make a claim stronger merely by editing the maturity field in the same change.

This is analogous to KanForge allowing broad proposal/search behavior while reserving commit truth for a separate verifier, and to Trader testing consumer obligations at explicit gates.

## 7. Structural source selection must have one shared authority

There is already a concrete selector mismatch:

- `architecture/source_gates.py` recursively discovers production source files outside excluded directories;
- `architecture/check_modules.py` currently discovers only scientific source files located directly at repository root.

The latter explicitly anticipates a future package migration, but until the selector changes a new nested scientific source file can evade the module registry while still being seen by other scans.

This is exactly the kind of seam an agent can accidentally exploit while reorganizing the repository.

### Direction

Define one reusable source-surface authority, or one generated source manifest, consumed by:

- module registration checks;
- source gates;
- semantic hazard coverage;
- architecture reports;
- potentially CI path selection.

Tests should plant:

- a root scientific file;
- a nested scientific file;
- an intentionally excluded test/doc/generated file;

and prove all consumers agree on classification.

## 8. Structural guards need non-vacuity witnesses

Climate already follows part of this rule: `architecture/source_gates.py` requires planted negative tests for each gate.

The next step is to verify not just rejection logic but **target coverage**.

Examples:

- the CUDA scan should assert that it still sees at least one known CUDA production path;
- module discovery should assert that it sees both current flat sources and future nested fixtures;
- fallback scanning should assert that the production selector has not silently excluded the relevant language/path;
- generated/derived registry checks should contain at least one planted stale projection that must be rejected.

A gate that returns zero findings because it stopped seeing its target should not be indistinguishable from a clean repository.

## 9. Missing, unavailable, blocked, invalid, stale, and empty-valid need distinct states

This is one of the strongest cross-repo lessons and directly addresses Climate's current placeholder/fallback problem.

Climate should avoid APIs where all of these collapse to `None`, `null`, empty arrays, zeros, NaNs, or a generic `success=false`:

```text
not_requested
not_available
resource_unavailable
not_implemented
unsupported_domain
invalid_input
numerical_failure
stale_dependency
corrupt_artifact
empty_valid_result
heuristic_result
verified_result
```

Not every language needs one giant enum, but public/evidence-bearing boundaries need enough information that downstream code cannot reinterpret failure as data.

This is especially important for:

- FFI backend discovery;
- spectral decompositions;
- conditioning/eigen diagnostics;
- observational fields and missingness;
- data acquisition/cache layers;
- CUDA/NCCL/MPI capability paths;
- checkpoint/restart;
- experiment execution under unavailable toolchains/hardware.

## 10. Fallbacks should be implementation identities, not branches hidden under one method name

The current Climate contract already says fallbacks are distinct implementations. The survey suggests making that executable.

A requested implementation such as:

```text
geometry.cuda.fp64.nccl
```

must not silently produce evidence under that identity if runtime actually used:

```text
geometry.cpu.fp64
geometry.cuda.fp32
geometry.cuda.single_device
geometry.cuda.compat_mpi
geometry.cuda.wmma_emulation
```

A fallback may be useful. The dangerous behavior is identity preservation across a semantic change.

The run manifest should record **resolved implementation identity**, and experiment policies can decide whether that identity is admissible for the requested comparison.

## 11. Fast/approximate paths should be allowed to advise without owning truth

The KanForge warm/fresh history suggests a useful pattern for Climate, particularly in ambitious GPU methods.

Examples where a fast path may be useful without being authoritative:

- low-precision curvature candidate screening;
- approximate eigensolvers before FP64/refined verification;
- cached preprocessing before provenance/identity verification;
- learned teleconnection candidate retrieval before baseline evaluation;
- approximate geometric diagnostics before known-manifold/reference checks;
- GPU performance screens before numerical equivalence is established.

A method contract should therefore be able to say whether an implementation is:

```text
proposal_only
screening_only
verification_capable
validation_capable
```

rather than infer authority from successful execution.

## 12. Configurability must have causal reach

Climate has a large configuration surface and many numerical/scientific constants embedded in source.

The Trader survey shows why merely centralizing values is insufficient: a registered knob that does not affect the intended computation creates a fake experimental degree of freedom.

For high-impact Climate parameters, future tests should establish a causal chain such as:

```text
registry/config field
  -> resolved method/experiment config
  -> invocation/kernel input
  -> controlled change in an observable witness
```

This is particularly important for:

- precision/determinism settings;
- physical parameter choices;
- metric construction choices;
- optimizer tolerances;
- decomposition/window parameters;
- stochastic seeds/noise models;
- candidate-batching controls;
- data preprocessing choices.

The test need not assert that changing a parameter improves science. It should prove that the parameter actually reaches the claimed mechanism.

## 13. Single homes should replace duplicated defaults and duplicated semantics

The cross-repo survey repeatedly found value in deriving projections from one semantic authority.

Potential Climate examples:

- method maturity should not be independently restated in README, module inventory, method registry, and claim docs;
- GPU precision/determinism classes should have one canonical definition;
- dataset identity should not be separately reconstructed by each language;
- experiment resource requirements should have one canonical record consumed by local and Commons execution;
- fallback implementation identities should be registered once;
- generated user-facing status tables should come from registries where possible.

DRY here means **one authority**, not merely fewer repeated lines of code.

## 14. Cache and artifact identity must cover semantics, not only bytes

A content digest answers "are these bytes the same?" It does not always answer "are these bytes valid for the current computation?"

Climate will likely need semantic identity for derived artifacts involving:

- dataset source/version and preprocessing graph;
- coordinate/calendar/unit transformations;
- method semantic version;
- code/build fingerprint;
- precision/numerical mode;
- random seed/ensemble identity;
- reference-model/likelihood assumptions;
- relevant experiment split/withholding policy.

This should be especially explicit for expensive reusable artifacts such as:

- normalized climate datasets;
- regridded fields;
- reference geometry tensors;
- Fisher/Jacobian caches;
- learned embeddings;
- GPU-precomputed candidate features;
- checkpoints.

## 15. Development benchmarks and confirmatory evidence should be different surfaces

Climate's `META-EXPERIMENTATION.md` already calls for baselines, falsifiers, and held-out evaluation. The KanForge adaptive checkpoint benchmark and Trader locked-final-test design sharpen the distinction.

Climate likely needs at least three categories:

1. **development fixtures** — tiny, inspectable, reusable, used constantly;
2. **adaptive benchmark suites** — may evolve as failure modes are discovered and can guide engineering choices;
3. **protected confirmatory surfaces** — not repeatedly inspected/tuned against and explicitly spent/versioned when used for stronger claims.

This is particularly relevant to novel methods, where repeated benchmark tuning can otherwise make an unusual representation appear stronger than it generalizes.

A protected surface does not necessarily mean secrecy. It means its use is governed so the development loop cannot repeatedly optimize against the same confirmation signal without that reuse being visible.

## 16. Evidence finalization should be transactional

If a run produces a result that changes evidence/claim state, the repository should avoid ambiguous partial finalization such as:

```text
result written
process crashes
promotion/spent-state receipt not written
```

A future evidence transaction could bind:

- run manifest;
- artifact digests;
- metric results;
- evidence record;
- protected-surface consumption state if applicable;
- claim/promotion decision.

The exact storage mechanism can wait for the operational architecture, but the semantic transaction should be designed before protected evaluation becomes important.

## 17. Replay/conformance should be used where Climate crosses semantic layers

Trader's event-spine replay suggests a generic testing strategy rather than a domain mechanism.

Candidate Climate conformance laws include:

```text
DatasetRef + preprocessing graph
    == materialized normalized dataset semantics

ExperimentSpec
    == locally resolved invocation
    == Commons-resolved invocation

CPU reference geometry
    == GPU geometry within declared numerical contract

RunManifest + artifacts
    == recomputed MetricResult projection

language A contract serialization
    == language B interpretation
```

These tests are valuable because they target translation seams where an agent can accidentally preserve type shape while changing meaning.

## 18. Destructive maintenance needs liveness witnesses

The earlier broad sanitation deletion should become a direct regression lesson.

Before deleting or replacing a substantial scientific surface, require evidence for the deletion scope:

- what exact hazardous path is being removed?
- what useful paths share the file/module?
- is the hazardous path reachable from an evidence-producing method?
- is there a narrower fail-closed/retype/quarantine option?
- what artifacts/tests/history preserve the useful design information?
- which callers become intentionally unavailable?
- is any resume/reproduction path destroyed?

Deletion can still be the correct result, but it should be the output of a liveness/reachability analysis rather than the default S3/S4 response.

This is a necessary amendment to the current sanitation policy's stronger preference for removing substantial prototypes wholesale.

## 19. Failed experiments and recurring friction should be retained as structured information

Two useful source-repo mechanisms should be considered later:

### Failed experiments

A failed or falsified experiment should remain a first-class record with:

- experiment identity;
- failure/falsification class;
- relevant evidence;
- whether failure was scientific, numerical, infrastructure, or resource-gated;
- reusable subproblem IDs where appropriate.

This prevents later agents from unknowingly repeating failed work or interpreting absence of positive evidence as "never tried."

### Architecture friction

One failure should not automatically generate a new abstraction.

A lightweight friction ledger can record recurring seams/failures and justify new infrastructure only when failures cluster. This is useful protection against capable agents overfitting the architecture to the most recent problem.

## 20. Resource truth must appear in evidence state

Climate's execution topology already distinguishes static, portable-CPU, CI-toolchain, CUDA-device, integrated-system, and large-data work.

The survey implies a stronger rule for outputs:

```text
not_run_resource_unavailable
```

must never be rendered as:

```text
passed
```

Likewise:

- mock execution is not live-resource execution;
- CUDA compile is not CUDA runtime verification;
- single-GPU execution is not NCCL/multi-GPU verification;
- fixture data is not real-data validation;
- a CI-installed Julia/Rust/Haskell toolchain proves only what actually ran there.

Whether unavailable-resource tests should fail CI or produce an explicit blocked status can vary by gate. The status itself must remain visible.

## 21. Proposed implementation order after this analysis

This is a dependency order, not a linear project plan.

### A0 — repair machine-readable truth

Before adding new harness layers:

- correct stale module/hazard records introduced by the reverted coarse sanitation change;
- remove dispositions that prescribe whole-file deletion where only subpaths are hazardous;
- add tests preventing the exact `restored implementation / registry still says stubbed` mismatch.

### A1 — unify source-surface discovery

Create one recursively correct source classification consumed by module registration and source gates. Plant root/nested/excluded witnesses.

### A2 — narrow hazard scope and promotion authority

Introduce scoped hazard records and witness-backed resolution. Separate proposal of maturity from promotion acceptance.

### A3 — explicit execution/result states

Define enough common state vocabulary that unavailable, stale, malformed, heuristic, fallback, and valid results cannot collapse at evidence-bearing boundaries.

### A4 — non-vacuous structural gates

Add coverage witnesses for scanners, generated projections, and source selectors.

### A5 — single-home / causal-config guards

Start with high-impact scientific/GPU controls rather than attempting to centralize every constant at once.

### A6 — semantic artifact identity and replay laws

Add identity/fingerprint coverage to expensive derived artifacts and cross-layer conformance tests to the most important translations.

### A7 — adaptive versus protected evidence surfaces

Only after experiment execution is sufficiently real to make the distinction operational.

### A8 — transactional evidence finalization

Needed before protected evidence or automatic claim promotion becomes authoritative.

The CUDA device, full Commons system, and large datasets gate some witnesses but do not block A0-A5.

## 22. What should *not* be copied

The survey also suggests several things Climate should avoid copying mechanically:

- KanForge's domain-specific proof graph, theorem kernel, or search semantics;
- Trader's market/event vocabulary;
- source-specific choices where missing/malformed states are still collapsed;
- direct file-write mechanics merely because the surrounding provenance model is good;
- a universal skip-is-failure rule without considering resource-gated Climate jobs;
- broad file-level invalidation when only a subpath is semantically unsafe;
- an agent framework whose own output can directly declare itself validated.

The goal is not architectural uniformity across repositories. It is a shared discipline around identity, evidence, authority, resource truth, and cross-repository interfaces while preserving domain-specific scientific semantics.

## 23. Acceptance criterion for the harness adaptation

The harness is improving when adding model capability does **not** increase the number of ways a weaker result can masquerade as a stronger one.

Concrete examples:

- a fallback can run, but cannot retain the preferred implementation identity;
- an agent can propose a claim promotion, but cannot grant it without independent witnesses;
- a new nested source file can exist, but cannot escape registration;
- a cache can accelerate work, but cannot survive semantic identity drift unnoticed;
- an unavailable GPU can block a CUDA witness, but cannot become a passing CUDA test;
- a heuristic can remain scientifically interesting, but cannot silently become a physical probability;
- a hazard can be locally quarantined without deleting unrelated useful mathematics;
- a registry can summarize source state, but cannot remain stale after the underlying state changes without a guard noticing.

That is the adaptation target. The next changes should begin with A0 because the current branch already contains a concrete stale-registry example.