# Harness survey deep-dive notes 01

Status: **raw observational notebook; not Climate policy**.

This file extends `docs/CROSS-REPO-HARNESS-SURVEY.md` with findings discovered after the first synthesis. It is intentionally closer to a lab notebook than a design recommendation. Consolidation and Climate adaptation come later.

## 1. LLM-Trader: capability authorization is separate from resource ownership

`signalcore/gpu_authorization.py`, `tests/test_gpu_authorization.py`, and `tests/test_gpu_lock_entry_paths.py` reveal two distinct GPU constraints:

1. the process must hold or be nested under the machine-wide GPU lock/scheduler authority;
2. GPU-entering code must also see an authorization marker set by that authority.

The workload never self-authorizes. Direct execution becomes authorized only as a consequence of successfully acquiring the shared lock. Scheduled execution receives the marker from the scheduler environment.

Two structural scans then make the convention difficult to bypass accidentally:

- every production module that enters CUDA must reference the machine-wide lock;
- every module that performs the relevant CUDA model load must reference the authorization guard.

Both scans contain **non-vacuity assertions** naming known GPU entrypoints. If a future refactor changes the syntax so the scanner sees nothing, the test fails instead of reporting a falsely clean tree.

This is stronger than a source regex saying `cuda` must appear near `lock`: it encodes an authority relationship and separately checks that the checker still observes the expected world.

## 2. Trader: cross-repository contracts are consumer pins, not copied schemas

`research/validation/scheduler_contract.py` consumes the `training-architecture` scheduler contract without re-authoring it.

Trader stores:

- the expected major/schema;
- a fingerprint over the owner's declared ABI surfaces;
- adopted capabilities/date.

At scheduled-run startup it then verifies:

- the owner's shared progress implementation is actually importable;
- the progress major matches;
- the owner can answer `contract --json`;
- the contract major matches;
- the fingerprint over the ABI surfaces matches the pin.

Administrative sections are deliberately outside the fingerprint so a consumer registration or migration-note edit does not create meaningless contract churn.

A standalone run is not forced to depend on the scheduler contract. A scheduled run that elected to enter that boundary fails closed if the shared implementation or pin is incompatible.

This is a useful example of **conditional dependency authority**: a feature is independent when unused, but strict once entered.

## 3. Trader: test mechanics themselves have a single home

`tests/test_test_helpers.py` scans the test suite for cloned mechanical helpers:

- fixture writers;
- URL fakes;
- repository-root calculations;
- subprocess startup fragments;
- shared synthetic-bar generators.

It has a planted `tests/fixtures/clone_fixture.py` containing every banned clone pattern and asserts each one is detected.

This guard is not about production semantics directly. Its purpose is to stop the test suite from splitting into multiple subtly different implementations of the same mechanics. That matters for agent work because copied fixtures can make two tests appear independent when they actually encode divergent assumptions.

This suggests a broader survey question: **which test/reference mechanics are part of the scientific authority and therefore deserve DRY enforcement just as much as production code?**

## 4. Trader: a protected final-test surface is modeled as a spent resource

`research/validation/locked.py` is more than a train/validation split.

A seal commits:

- instrument;
- freeze instant;
- exact release-candidate configuration hash.

After sealing, ordinary experiment runners permanently refuse observations at or beyond the freeze. The special evaluator may access them once. The resulting evidence hash is written back into the seal, and a second evaluation refuses.

Corrupt/partial seals fail loudly; absence means the surface was never frozen.

The conceptual move is important: **untouched evidence is a consumable resource**. Repeated inspection changes its epistemic role even if the bytes never change.

This is directly relevant to any future Climate protected observational period, hidden simulation family, benchmark answer key, or final confirmatory ensemble.

## 5. Trader: failure is structured research state

`research/validation/failure.py` treats failed experiments and rejected proposals as typed knowledge-bearing records rather than terminal log strings.

A failure can carry:

- location;
- local context/config fingerprint;
- attempted mutation;
- violated constraint;
- observed evidence;
- downstream effects;
- reusable diagnosis;
- next-search recommendation.

A reusable diagnosis can be promoted into a stable-ID subproblem linked to its parent hypothesis.

The important property is not the exact enum. It is that failed search branches can become nodes in the research graph without being reinterpreted as positive evidence.

## 6. Trader: cross-layer semantic continuity has dedicated tests

`tests/test_subsystem_kernel_first.py` and `tests/test_subsystem_kf_remainder.py` test meanings that unit tests can miss:

- proposal quantity survives mapping into actual position size;
- sell means short rather than silently changing into close-long semantics;
- aggregate exposure reflects live positions rather than a static default;
- storage implements a protocol instead of defining the event-spine abstraction;
- observable <= decided <= executed timestamps;
- fills join back to proposal/context/outcome records;
- money/equity are independent from ambient wall time;
- data drift becomes an event, not only a hidden manifest change;
- every source declares availability semantics;
- execution semantics handle gap-through stops correctly.

These are tests of **semantic preservation across partitions**. They are distinct from both unit tests and end-to-end smoke tests.

For the Climate survey this raises an analogous class of potential boundaries:

```text
units -> normalized fields -> numerical kernel -> diagnostic -> artifact
calendar -> time window -> transform -> experiment split
method config -> device implementation -> metric result -> evidence record
claim identity -> run identity -> Commons projection
```

No Climate rule is adopted here; the observation is that boundary semantics deserve direct witnesses.

## 7. Trader: generated language surfaces derive from one schema authority

`web/shared/generate_schemas.py` generates JSON schemas and TypeScript interfaces from the Python/Pydantic model authority. Generated files are explicitly marked not to edit.

Together with architecture-generated documentation and recomputed cross-project contract fingerprints, this shows a recurring Trader strategy:

> consumer-friendly duplication is acceptable when it is generated and drift-tested; competing handwritten authority is not.

This distinction is potentially useful for Climate's eventual Python/Rust/Julia/CUDA interfaces.

## 8. Trader: skip-is-failure is itself tested end-to-end

`tests/test_skip_policy.py` does not merely unit-test a pytest hook. It launches real pytest subprocesses against probe tests and demonstrates that:

- an actual skip causes a nonzero run;
- a normal pass still passes.

This is another example of checking the *enforcement mechanism through its real host*, not only the helper function implementing it.

The rule itself remains domain/resource-specific. Climate has intentionally unavailable CUDA and large-data resources in some environments, so the exact Trader policy is not automatically transferable.

## 9. KanForge: publication is downstream of authority

`growth/commit.js`, `digest/auditPack.js`, and `digest/development.js` assemble human- and machine-readable publication records only after proof verification.

The digest carries:

- statements/proofs;
- dependencies;
- assumptions;
- provenance;
- patch streams;
- verified-only hash-chain entries;
- optional training/GRPO records.

`core/hasher.js` makes the chain reproducible and independently verifiable.

The key separation is that publication formatting does not decide truth. It projects the result of the hard verifier path.

### Negative storage finding

These publication writers use ordinary direct file writes rather than atomic `tmp + replace` publication. That does not invalidate the proof semantics, but Climate should not copy the persistence mechanics merely because the artifact/provenance shape is attractive.

## 10. KanForge: learned failure predictors are explicitly not causal truth

`optimization/causal.js` names its own limitation unusually clearly: its transition/failure patterns are sequence statistics, not causal inference.

Predictors may influence search only after support/confidence gates. Very small-support and extreme-confidence rules remain inert. The live path mines from prior-run dataset samples so the current cycle's outcomes cannot train the gate that judges the same cycle.

The comments also identify a remaining weakness: an ablation mode can mine and apply within one run, which is a hypothesis-generating measurement rather than validated gating evidence.

This pattern is important for agent harnessing because an optimizer can be useful without being permitted to redefine the truth boundary.

## 11. KanForge: retrieval authority is enforced after generation

`search/premises.js` uses a simple BM25 baseline to propose relevant premises and can instruct the model to use only that set.

Crucially, prompt wording is not the final enforcement. The complete assembled proof source is inspected at commit time for any corpus premise not present in the retrieved set.

This creates a recurring two-layer structure:

```text
soft guidance / retrieval / learned hint
                  |
                  v
creative proposal generation
                  |
                  v
hard post-generation verifier over the full artifact
```

The hard verifier prevents a clever or confused model from escaping the intended experimental condition merely by ignoring its prompt.

## 12. KanForge: checkpoint coverage is good but corruption semantics remain weak

`test/checkpoint.test.js` verifies:

- event-store round trips;
- checkpoint save/load;
- resume behavior;
- hash-chain verification;
- already-proved lemma skipping.

It does not, in the inspected tests, plant malformed checkpoint JSON and require corruption to differ from absence. `RunCheckpoint.load()` currently returns `null` for both conditions.

This reinforces the earlier negative survey finding: a subsystem can have strong happy-path/resume coverage and still blur an epistemically important failure distinction.

## 13. KanForge versus Trader: live-resource test policy differs

`kanforge/test/blueprint.live.test.js` explicitly runs only against the real Lean REPL and contains no facsimile kernel. When `KANFORGE_REPL_BIN` is unavailable, Node marks those tests skipped.

Trader globally converts skips into failures because skips previously hid a broken live path.

These are not simply contradictory philosophies. They operate under different execution assumptions.

The common invariant appears to be:

> A non-executed test must never be interpreted as positive evidence for the resource-dependent claim it would have tested.

A future Climate harness likely needs an explicit evidence result such as `passed`, `failed`, `not_run_missing_resource`, `not_applicable`, rather than relying on CI color alone. This remains a survey inference, not an adopted implementation.

## 14. Emerging mechanism: checks should prove non-vacuity as well as rejection

A pattern now appears in several independent Trader gates:

1. scan the intended production topology;
2. reject violations;
3. assert that known target surfaces were actually observed;
4. inject a synthetic violation and prove it is rejected where practical.

A checker can therefore fail in two directions:

- it finds a defect;
- its worldview drifted so far that it can no longer see the subsystem it claims to guard.

This is more robust than a clean-result assertion alone.

## 15. Emerging mechanism: the same fact can cross layers only through an explicit semantic witness

Across both repositories, many high-value tests are not checking local implementation details. They check that a fact keeps its meaning while crossing a boundary:

- a knob actually changes the intended computation;
- a proposal quantity survives into execution;
- a retrieved-premise set survives into the committed proof;
- a scheduler contract pin matches the live owner implementation;
- an event links to its originating proposal and later outcome;
- a published aggregate recomputes from raw rows;
- a generated consumer schema recomputes from the source model.

This suggests a useful survey vocabulary:

**semantic edge witness** — a test proving that the meaning carried over one graph edge is the intended meaning, not merely that both endpoint objects exist.

Again, this is nomenclature for analysis, not yet Climate architecture.

## 16. Further surfaces still worth inspecting before adaptation

The survey is not closed. High-value remaining areas include:

### LLM-Trader

- locked-test execution and its tests, not only seal mechanics;
- failure promotion consumers;
- schema drift/generated-file tests;
- GPU scheduler/lock tests around contract fingerprinting and nested ownership;
- event replay as a second consumer of the same spine;
- any deliberate protected-data dependency bans;
- research run publication and stale-artifact behavior.

### KanForge

- dataset contamination checks and oracle partition implementation;
- exact benchmark/checkpoint-set identity;
- live-backend versus mock evidence accounting;
- dataset/training artifact publication;
- resume behavior around partial final artifacts;
- source/statement pin drift tests;
- any guards around report/config registry drift;
- fresh-process versus warm-session authority handling.

No Climate adaptation should be frozen until these remaining surfaces either add a new mechanism or stop materially changing the synthesis.
