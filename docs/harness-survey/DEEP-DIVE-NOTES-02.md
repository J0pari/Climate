# Cross-repository harness survey — deep-dive notes 02

Status: **observational notebook, not Climate policy**.

This file continues the open-ended survey of `J0pari/KanForge` and `J0pari/LLM-Trader`. It records mechanisms found by inspecting code, tests, repository history, and failure handling. It deliberately does not yet prescribe Climate changes.

The purpose of this phase is to observe how capable agent-built systems constrain their own failure modes before deciding which constraints are genuinely generic.

## 1. Fast paths and authority paths are different things

KanForge's history repeatedly distinguishes a fast/warm verification path from an authoritative fresh verification path.

Relevant commits include:

- `8e7da6bc533295bd89742f519be1fa6d369da8b3` — warm verification is explicitly advisory; failures fall through to fresh-environment checks because stale or differently warmed environments produced false negatives.
- `8402d2cab715f258bf0bc912e53a3eb4e2d56448` — exact target statements are supplied to formalization; warm-session poisoning such as already-declared names triggers reset/fresh recheck.
- `8892c0683e5277c4c5beb5933f3887093102bfc8` — import-bearing reuse/repair sources use fresh-only checks.
- `90ce7fae1523e596f2ed63b265047100e06515a4` — ranked reuse may propose candidates, but candidates remain kernel verified.
- `0c676543f82249bd973ac1a3b63f9ce2bc97f353` — warm-first execution is retained as an efficiency technique with fresh fallback and rejection memoization.
- earlier commits (`90075d8`, `8766f58`, `80f9e88`, `2ecda2b`) show the practical tension between fresh verification and real memory/runtime cost.

The important observation is not that fresh execution is always superior. It is that **an optimization path can be useful without being granted authority**.

A recurring architecture shape is:

```text
cheap / cached / warm / approximate path
               |
               v
        proposal or early rejection
               |
               v
expensive / fresh / independent authority path
               |
               v
           commit/evidence
```

The repositories repeatedly pay real performance cost to preserve this distinction when the fast path can be contaminated by stale state.

## 2. Protected evidence can be consumable exactly once

LLM-Trader's locked-final-test machinery is stronger than an ordinary train/holdout convention.

`tests/test_locked_test.py` verifies that:

- a seal can be created and loaded;
- a seal cannot be overwritten casually;
- pre-freeze data remains usable by ordinary workflows;
- post-freeze data is permanently refused by ordinary workflows;
- the dedicated evaluation path may consume the protected surface;
- spending the protected surface does not make it ordinary data afterward;
- a second evaluation is refused;
- the ordinary train/holdout helper also observes the lock;
- corrupt seal state is a loud error rather than a missing-seal equivalent.

The notable design property is that protected evidence has **state and scarcity**. It is not just another dataset split whose results can be inspected repeatedly while tuning the system.

### Caveat found during the survey

The finalization sequence appears to write the result artifact before recording that the seal has been spent. A crash in that interval could plausibly leave a result artifact next to a seal that still claims to be unspent.

This does not negate the protected-surface idea. It does show that one-shot evidence finalization is a transaction/receipt problem if the result and the spent-state marker must move together.

## 3. Consumer contracts test consumer obligations, not the owner's internals

LLM-Trader's `tests/test_scheduler_contract.py` is a useful example of a cross-repository contract boundary.

The tests use fake owner-side modules/manifests to verify the consumer's responsibilities:

- scheduled mode refuses a missing shared progress suite;
- incompatible major versions are refused;
- stale contract fingerprints are refused;
- the pinned compatible contract is accepted;
- standalone mode remains decoupled from the shared scheduler;
- absence of an optional standalone integration does not create a silent substitute;
- the pin declares its adopted major and capabilities;
- the fingerprint covers ABI surfaces but intentionally excludes administrative sections;
- a manifest missing required ABI surfaces cannot be partially fingerprinted;
- a missing or wrong-major pin is a loud error.

The same pattern is used for the shared GPU lock:

- manual launch acquires the shared lock or aborts;
- a held lock is not bypassed locally;
- scheduled execution may run nested under scheduler ownership;
- a missing lock implementation is an abort rather than permission to invent a local substitute.

This is an asymmetric ownership model:

```text
producer repository: owns producer behavior
consumer repository: owns consumer conformance
shared contract: pins the boundary
```

That avoids forcing a consumer test suite to reproduce the producer's implementation tests.

## 4. Cross-layer conformance is stronger than local unit correctness

LLM-Trader's event replay tests treat the recorded execution event spine as an authoritative input and compare an event fold against the ordinary frame fold.

The conformance witnesses include:

- stop-loss exits;
- signal-reversion exits;
- end-of-run liquidation;
- gate rejection;
- proposal-derived frames produced by the agent harness;
- invalid off-bar fills as loud errors;
- evidence that replay does not consult the original signal frame;
- randomized scenarios spanning entries, stops, take-profits, flips, partial fills, gate rejections, and kill-switch behavior;
- an independent accounting rebuild agreeing with the final folded result.

The important mechanism is a **cross-layer semantic continuity law**:

```text
runtime execution -> recorded event spine -> replay fold
```

must preserve the relevant meaning, not merely satisfy independent unit tests at each layer.

This catches wiring and translation defects that can survive excellent local test coverage.

## 5. Destructive cleanup is itself an authority-bearing subsystem

LLM-Trader's `tests/test_run_cleanup.py` is relevant to the earlier Climate mistake of treating a large coarse deletion as sanitation.

Cleanup is guarded by exact liveness predicates:

- completed training may lose per-step checkpoints while retaining the final adapter;
- incomplete training retains the checkpoints required for resume;
- hosting intermediates may be removed only after the candidate is registered;
- an unregistered candidate keeps its intermediates.

The broader observation is that **deletion/pruning is not merely maintenance**. It changes which evidence, resume paths, and implementation options remain available. The deletion predicate therefore deserves direct tests and negative witnesses.

## 6. Derived caches carry semantic identity

LLM-Trader's DPO reference-log-probability cache (`research/validation/build_reference_cache.py`) does not treat a file path as sufficient cache identity.

Its identity incorporates, among other things:

- the SFT adapter weights hash;
- base model identity;
- pinned model revision;
- sequence/tokenization bounds;
- subset size;
- subset seed.

Each cached pair also has a content-derived row hash.

The verifier distinguishes:

- cache absent;
- cache identity stale;
- cache present but row coverage incomplete;
- cache valid and complete.

A stale cache is not silently overwritten or accepted. The build is resumable, but reuse is conditional on semantic identity.

This is a concrete example of **cache != authority**. A derived artifact is trustworthy only under the assumptions and inputs that produced it.

## 7. Adaptive benchmarks are useful but are not independent confirmation

KanForge's `kanforge/bench/checkpointSet.js` constructs the ablation problem set directly from the current mission checkpoint:

- unproved ready lemmas whose dependencies are satisfied are selected first;
- stalled-but-ready lemmas are included as the deadlock-release retry set when ordinary ready work is exhausted;
- the sample is capped and ordered to resemble what the next mission pass would actually attempt;
- ablation recommendations feed back into the next campaign configuration.

This is a strong operational benchmark because it measures components on the **live frontier that matters now**, rather than on a detached toy distribution.

It is also adaptive by construction. As the system learns from that frontier and changes future passes, the measurement distribution moves with the system.

That makes the surface excellent for iterative engineering and weaker as an independent confirmatory surface unless a separate protected benchmark is maintained.

The general distinction observed here is:

```text
adaptive development benchmark
    -> useful for selecting what to try next

protected confirmatory benchmark
    -> useful for claims that need independence from the development loop
```

## 8. Non-execution must remain visible

The two repositories use different policies for resource-dependent tests.

LLM-Trader deliberately turns pytest skips into failures. `tests/test_skip_policy.py` tests the policy itself by spawning real pytest probes:

- a planted `pytest.skip(...)` must make the subprocess fail;
- a normal passing probe must still pass.

The second witness matters: it proves that the enforcement mechanism is sensitive without simply making every run fail.

KanForge has live-kernel tests whose normal behavior may be to skip when the external REPL/kernel is unavailable, while documentation states that mocks are not substitutes for live verification.

These policies are not identical and need not be. Their common property is more important:

**resource execution state is part of the evidence.**

`passed on real resource`, `passed against a fake`, `not executed because resource unavailable`, and `failed on real resource` are materially different states.

## 9. Guards should test their own non-vacuity

Several Trader mechanisms test not only that bad input is rejected but that the guard still observes the intended system surface.

Examples found in the survey include:

- GPU authorization/source scans that verify known GPU-loading homes are still seen;
- skip-is-failure probes that include both a forced skip and a genuine pass;
- clone-detection fixtures that deliberately inject duplicated test mechanics;
- contract-fingerprint tests that prove both intentionally ignored administrative changes and intentionally covered ABI changes.

This is a separate property from ordinary negative testing.

A structural guard can become vacuous because:

- its source selection no longer reaches the relevant files;
- a naming convention changes;
- an exclusion becomes too broad;
- a generated surface stops being generated;
- the checker crashes or returns the permissive default;
- no fixture exercises the class it claims to prohibit.

The source repositories sometimes protect against that by planting a witness that must be observed.

## 10. Soft accelerators repeatedly sit upstream of hard verifiers

KanForge uses several mechanisms that are useful precisely because they are not authoritative:

- ranked proof reuse proposes likely prior lemmas;
- premise retrieval proposes likely relevant material;
- learned sequence statistics/predictors can influence search;
- warm-session checks can cheaply reject or advise;
- evidence-derived component recommendations can guide the next campaign.

The hard boundary remains downstream:

- assembled source is inspected;
- premise/source leakage rules are applied;
- the proof kernel verifies the actual artifact;
- commit-time provenance is attached.

This is a recurring shape:

```text
heuristic / learned / cached accelerator
               |
               v
             proposal
               |
               v
     independent hard verifier
               |
               v
             authority
```

The accelerator is therefore allowed to be imperfect without silently redefining correctness.

## 11. The source repositories also contain ambiguous states

The survey should not idealize either repository.

### KanForge checkpoint ambiguity

Earlier inspection found a checkpoint-loading path that can collapse malformed checkpoint state and absent checkpoint state into the same `null`-like result. Existing checkpoint tests emphasize resume/round-trip/hash-chain behavior; no planted witness was found that requires malformed and absent checkpoint states to remain distinct.

This may be acceptable for a recoverable search workspace if durable evidence lives elsewhere, but it would be unsafe for an evidence authority.

### KanForge publication writes

The provenance/hash-chain model for published verified artifacts is useful, but some publication writers use ordinary direct file writes rather than atomic replacement. Good provenance semantics do not automatically imply robust storage semantics.

### KanForge narrative drift

A top-level README/project-structure description has lagged behind the implementation's deliberate move away from LLM decomposition in the skeleton-generation path. This is a reminder that prose status can become stale even in a heavily tested repository.

### LLM-Trader one-shot finalization gap

As noted above, the locked-test result and the spent seal are not obviously committed as one transaction.

These are useful observations because they show that a good harness is not a binary property. Different authority surfaces can have different strengths.

## 12. Repository history is part of the architecture survey

A number of the strongest mechanisms were easier to understand from commits than from the current tree alone.

KanForge history records concrete reasons for architecture changes:

- a prose-only formalization route once targeted the wrong mathematical problem;
- warm-session state caused false verification outcomes;
- repeated fresh imports caused real memory pressure/OOM behavior;
- cached/reuse paths required fresh verification of assembled transitive closure;
- an LLM decomposition/planning role was removed after it was identified as a source of ungrounded graph structure.

This matters because current constraints often look arbitrary if their failure history is discarded.

A harness survey therefore benefits from asking not just:

> what rule exists?

but:

> what concrete failure made this rule worth its complexity?

## 13. The emerging mechanism families

Without yet turning them into Climate policy, the independent observations are clustering around these families:

1. **authority separation** — proposal/search/advice is not commit/evidence authority;
2. **state distinction** — absent, unknown, invalid, malformed, stale, skipped, empty-valid, and verified are not synonyms;
3. **single semantic homes** — duplicated defaults/contracts/projections are derived or rejected;
4. **causal configuration** — registered knobs must reach the computation they claim to control;
5. **cross-layer conformance** — important meaning is tested across boundaries, not only within modules;
6. **non-vacuous guards** — structural checks contain planted witnesses that prove the checker still sees its target;
7. **semantic cache identity** — cached work is reused only under the identity of the inputs/assumptions that produced it;
8. **protected evidence** — some measurements are scarce and must be kept outside the adaptive development loop;
9. **resource honesty** — unavailable hardware/kernel/tool execution is a recorded state, not a pass;
10. **destructive liveness** — cleanup/deletion requires explicit proof that the removed artifact is no longer authoritative or needed for recovery;
11. **soft accelerator + hard verifier** — heuristics may make work cheaper without receiving authority;
12. **history-backed constraints** — important rules retain enough rationale to avoid later agents deleting them as apparently unnecessary complexity.

These are observations, not yet an implementation checklist.

## 14. Open questions before adapting to Climate

The survey is close to saturation, but several translation questions should remain open until the observations are consolidated:

- Which Climate surfaces genuinely need an authority path distinct from an advisory path?
- Which resource-dependent checks should fail on non-execution, and which should report an explicit blocked/skipped evidence state?
- Which scientific datasets/fixtures should remain adaptive development surfaces versus protected confirmation surfaces?
- How should a protected scientific evaluation be committed transactionally?
- Which Climate derived artifacts need semantic identity beyond a content hash?
- What is the right granularity for a destructive-cleanup/liveness contract?
- Which source scans can be made robust enough to deserve blocking status?
- How should agent-authored proposals be represented so they cannot directly promote method or claim maturity?
- How much of the harness should be shared with Commons versus remain Climate-owned?

The next step after survey saturation should be a **separate adaptation proposal** that maps these observed mechanisms onto Climate's actual failure modes. That document should be reviewable independently from the observations above.