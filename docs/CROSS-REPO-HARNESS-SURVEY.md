# Cross-repository harness survey: KanForge and LLM-Trader

Status: **observational survey, not Climate policy**.

This document records mechanisms observed in `J0pari/KanForge` and `J0pari/LLM-Trader` before deciding which, if any, should become Climate architecture. It intentionally does not begin from a fixed checklist of features to copy. The purpose is to identify structures that repeatedly emerged from concrete failure modes, and to preserve negative findings as carefully as positive ones.

The adaptation step comes later. A pattern appearing here is not yet a Climate requirement.

## 1. Survey method

The survey reads across more than the repositories' headline architecture documents. It includes:

- contributor/agent operating contracts;
- current source topology and implementation seams;
- architecture registries and generated projections;
- static gates and planted negative witnesses;
- runtime event, lifecycle, and persistence code;
- agent proposal and authority boundaries;
- experiment/ablation machinery;
- report self-audits;
- provenance and identity mechanisms;
- recent commit history describing failures and subsequent structural repairs;
- known inconsistencies where prose or loaders lag the intended architecture.

The important unit is not a clever function. It is a recurring relationship between **authority, evidence, failure, and change**.

## 2. LLM-Trader: observed architecture

### 2.1 Architecture is a compiled claim graph

Trader's `architecture.py` is not just documentation. Architectural claims are authored once with IDs, status, scope, code anchors, witnesses, dependencies, and stability. Derived surfaces such as dependency edges, test matrices, event schemas, state-transition descriptions, replay surfaces, and generated documentation are projections of that authored relation.

`tests/test_architecture_surface.py` then acts like a compiler:

- code anchors must resolve;
- witness functions must exist and declare the claim they witness;
- established claims require sufficiently strong witnesses;
- dependencies must resolve and remain acyclic;
- generated documentation must exactly match the registry projection;
- external audit findings that are marked fixed must bind to a claim and witness;
- entrypoint pointers and the prescribed iteration-report shape are test-locked.

The notable property is not the Python representation. It is that architecture statements have **referential integrity** and cannot remain true merely because prose says so.

### 2.2 Static gates constrain where errors may enter

Trader has multiple source-level invariants that share one production-tree topology rather than each scanner inventing its own scope.

Observed examples include:

- banned dependency directions/imports;
- no ambient wall-clock reads inside deterministic computational paths;
- no silent literal/substitute fallbacks;
- no arbitrary configurable magic numbers outside their authority homes;
- no duplicate literal defaults across config classes;
- no registered-but-inert experiment knobs;
- one canonical metric implementation for each named metric;
- explicit GPU authorization before device entry.

Several of these gates contain planted violations. The repository therefore tests both:

1. that production code is currently clean under the rule; and
2. that the rule is capable of failing on the defect class it claims to prevent.

This second property is important for agent harnessing: a static check that has never demonstrated sensitivity can become decorative reassurance.

### 2.3 The arbitrary-default ban is stronger than centralizing constants

Trader does not consider moving a guessed number to `constants.py` sufficient. Its operating contract allows a parameter only when it is:

- adopted from measurement with provenance;
- derived from a declared/principled relation at runtime; or
- required from the caller, with omission failing loudly.

Recent history shows stored guesses being replaced by relations such as a minimum data floor derived from warm-up requirements plus the maximum outcome horizon. Repeated defaults are structurally rejected by `tests/test_config_single_home.py`.

This is semantic DRY: **one authority for a fact**, not merely one textual spelling of a value.

### 2.4 Knob registration is not accepted as evidence of knob function

`tests/test_knob_flow.py` classifies knobs by how their causal reach is witnessed:

- output/frame differences;
- invocation arguments reaching the intended consumer;
- engine-level behavior differences.

`tests/test_registry_coverage.py` requires registered configurable fields to have one of those flow witnesses or a deliberate non-searchable classification.

The defect being prevented is subtle: an inert parameter can make an experiment or ablation appear to compare two methods when both runs execute the same computation.

### 2.5 Agent intelligence is proposal-side, not authority-side

Trader's agent harness treats the model as a proposal source. Proposals become canonical typed events, and downstream execution consumes the same representation during live operation and replay.

The tool surface narrows agent writes to typed proposal and abstention events. Runtime gates remain outside the agent. Unwired tool sources raise rather than returning empty data. A missing broker/account does not manufacture an account snapshot.

The consequence is that a more capable model can improve proposals without becoming the authority for execution truth, evidence identity, or state accounting.

### 2.6 Unavailability is represented differently from emptiness

Trader repeatedly distinguishes:

- source absent;
- valid source whose answer is empty;
- malformed/corrupt state;
- feature not yet licensed by a gate.

Examples:

- unknown data-source names raise with the known registry;
- unwired broker/crypto/options tools raise `ToolError`;
- selection-layer tools raise until their start gate opens;
- malformed lifecycle state raises, while a missing lifecycle file means not started;
- metrics that cannot be computed are documented as `null`, not fabricated.

The architecture therefore spends type/error vocabulary to preserve epistemic distinctions instead of collapsing them into convenient successful values.

### 2.7 Durable state has narrow, shared envelopes

Trader's lifecycle abstraction deliberately stays small: phase, completion, resume pointer, timestamp, with domain-specific fields in typed subclasses. Persistence is atomic (`tmp` + replace). Missing state is `None`; malformed state is corruption and raises.

The event log similarly has a small protocol and typed write boundary. It validates before consuming a sequence ID, uses strict JSON, rejects non-finite/nonserializable payloads, and repairs only a torn final append rather than inventing a replacement event.

This is an example of an abstraction being justified by a real shared need while resisting a framework-sized state machine.

### 2.8 Cross-repository surfaces are recomputable contracts

Trader publishes generated/frozen contract descriptors and tests that committed contract files match their producers. Published fingerprints are recomputed from the committed artifacts.

Recent history also shows a deliberate ownership correction: Trader retired preference/training projections that belonged in `training-architecture` and reduced its boundary to immutable domain evidence. The key pattern is **one owner per transformation**, not convenience copies on both sides of a repository boundary.

### 2.9 Identity binds the choices that produced an object

Recent property tests bind stochastic sampling configuration to candidate identity. Different seeds cannot alias to the same candidate ID; differing payloads under one ID fail closed; canonical serialized form is treated as the equality authority and hashes as indexes.

This matters beyond trading: any Climate candidate/method/run identity that omits precision, preprocessing, seed, metric construction, or other semantics can produce the same class of aliasing bug.

### 2.10 Skips, failures, and friction are retained as information

Trader globally rewrites pytest skips into failures. Its motivation came from a real live-test defect hidden by repeated environmental skips.

Separately, `docs/arch-signals.md` records architectural friction as evidence rather than immediately converting every failure into a task. Repetition/clustering is what justifies architecture work.

These two ideas should not be confused:

- an asserted test that did not execute must not silently contribute to green evidence;
- a single operational annoyance need not automatically generate a new abstraction.

### 2.11 The agent operating procedure is executable in unusual places

Trader's harness extends beyond domain code. A PowerShell command guard refuses progress-hiding pipe patterns and has its own regression test. The end-of-iteration report shape is itself compiler-checked. Commit messages are expected to state `CLAIM_DELTA` and `WITNESS_DELTA`.

This is a broad interpretation of an agent harness: if a repeatedly observed agent behavior causes loss of observability or architectural drift, move the correction from prose into an executable boundary.

## 3. KanForge: observed architecture

### 3.1 Truth authority is intentionally narrower than search authority

KanForge's foundational separation is concise: the LLM proposes, the Lean kernel disposes.

The search system can be rich and adaptive, but a separate `CommitGate` owns:

- pin/context drift checks;
- extraction/assembly of complete proof source;
- whole-source kernel verification;
- premise-lock enforcement;
- complete-source leakage guardrails;
- statement/proof hash material for the run chain.

The commit gate performs no LLM calls. This prevents improvement or failure in the proposal mechanism from changing the meaning of `verified`.

### 3.2 Search can be permissive while commit remains hard

`core/guardrails.js` distinguishes exploratory relaxations from the hard requirements for verified/committed work. Whole assembled source is scanned, not only the small fragment an agent just edited.

The more general mechanism is a **promotion boundary**:

- exploratory objects may exist with weaker status;
- promotion into the durable/reusable set requires the hard authority path;
- temporary grants cannot silently become permanent truth.

### 3.3 A central LLM seam makes budgets and failure types observable

`agent/proposalEngine.js` wraps all live-loop LLM calls in one seam. That seam owns call accounting and the hard call budget, and distinguishes abstention, budget exhaustion, provider failure, and successful proposal.

This is useful agent architecture because it prevents recipe/search code from creating hidden model calls or interpreting a dead provider as a silent model.

The proposal engine also occasionally re-tests failure predictors against the kernel, producing counterfactual evidence rather than allowing a learned veto to become self-confirming forever.

### 3.4 The repository removed an apparently capable LLM stage when its contract was wrong

KanForge's August architecture audit is unusually informative. It concluded that the LLM proof-plan/skeleton stage was a god object: it could invent false intermediate lemmas that the kernel had not yet engaged with, causing the rest of the system to spend large budgets on an ungrounded DAG.

The response was not prompt tuning. The planner role was removed. The current structural seed is deterministic and syntax-derived; DAG growth comes from kernel-engaged artifacts.

The transferable lesson is not “never let an LLM plan.” It is:

> A stage should not be permitted to create a node whose asserted status exceeds the authority available at that stage.

### 3.5 Falsification uses the LLM as candidate generator, never judge

The falsification gate asks the LLM for concrete small counterexample candidates. Each candidate is then checked by the Lean kernel. Only a kernel-verified counterexample can falsify a lemma.

Failure to generate a counterexample is not converted into proof. A transport error similarly carries no positive evidence.

This is a clean example of assigning an LLM to a high-recall creative role while keeping the evidence transition mechanistic.

### 3.6 Reuse stores verified knowledge but retrieval is not truth

The lemma store is content-addressed and written atomically. Underivable index metadata is stored as `null` rather than fabricated. Corrupt entries are tracked separately.

Retrieval can be exact or ranked, but ranked results are explicitly candidates and are re-verified by the kernel before reuse. The store also has a canonicalization preference for shorter proof trajectories without confusing that ranking preference with truth.

The general pattern is important for Climate reuse/caching: **a trusted source artifact can still produce an untrusted retrieval candidate**.

### 3.7 Publication artifacts are self-auditing

KanForge's ablation/trainer reports contain audit blocks that recompute aggregates from raw rows/events. The audit checks row coverage and uniqueness, outcome consistency, budgets, confidence intervals, per-problem/per-recipe summaries, terminal-event coverage, predictor support/confidence, and provenance.

Tests plant fabricated totals, impossible solved/error combinations, missing terminals, incorrect confidence intervals, and missing provenance to prove the report audit catches them.

This reduces dependence on a human or agent noticing that a polished summary disagrees with the underlying event stream.

### 3.8 Incomplete work is prevented from looking complete

The blueprint runner preserves incremental events/checkpoints during long work. Its completion/publication behavior is designed so an unfinished mission cannot leave a stale finished-looking development digest as the apparent current result.

This is particularly relevant to Climate because a stale scientific report or figure can be more dangerous than a missing one.

### 3.9 Provenance distinguishes missing from known-unknown

KanForge's mandatory provenance block requires all core provenance fields. Values that genuinely cannot be known are represented as `unknown:<reason>` rather than omitted.

That does not make the result reproducible, but it preserves the distinction between:

- a field that was forgotten;
- a field that was considered and genuinely unavailable.

### 3.10 Experiment machinery measures the live path instead of maintaining a shadow implementation

The ablation harness constructs the same `TacticLoop` used by the open-problem pipeline with alternate configurations. It measures the live path rather than recreating each strategy in a separate benchmark implementation.

This is a significant DRY property: the measurement instrument changes configuration, not implementation authority.

It also records per-cell failures rather than allowing one failing cell to erase the rest of a comparison, and the report audit later checks the resulting matrix.

### 3.11 The component registry connects configuration, ablation, and UI

One registry describes configurable components, legal options/ranges, safe starting values, and evidence-derived recommendations. Runtime overrides are validated against that schema. Components absent from the registry are not configurable by consumers.

The strongest aspect is the shared control surface. It prevents an ablation tool, a UI, and the live runtime from each maintaining a different list of knobs.

### 3.12 The oracle is a strict partition, not a clever prompt source

KanForge's architecture audit describes an answer-key/oracle partition that grades the general pipeline but must never feed it. This was tightened after problem-specific mathematical guidance leaked into the generic pipeline and was reverted.

This is directly analogous to protected holdouts, hidden labels, and benchmark answer keys in scientific/ML work: a strong oracle is useful only if the dependency graph prevents it from becoming an input.

## 4. Recurring mechanisms across both repositories

The following themes emerged independently enough to deserve further Climate consideration. They are still observations, not adopted policy.

### 4.1 Authority graph, not feature list

Both repositories become easier to reason about when represented as a directed graph of authority:

```text
proposal / hypothesis / retrieved candidate
            |
            v
bounded executable interface
            |
            v
independent authority / verifier
            |
            v
immutable event/artifact
            |
            v
self-audited projection / evidence
            |
            v
promotion / recommendation / reuse
```

Most serious defects can be described as an illegal shortcut edge in this graph.

Examples:

- an LLM plan directly creating authoritative intermediate claims;
- a fallback value directly inhabiting the successful scientific result type;
- a stale handwritten contract bypassing the producer fingerprint;
- an inert knob being treated as a distinct experiment arm;
- an oracle/holdout leaking into candidate generation;
- a cached/retrieved result bypassing fresh verification;
- a demo artifact landing in the canonical evidence path.

### 4.2 One authored relation, many derived views

Repeated successful shapes include:

- one claim graph -> docs/test matrix/dependency graph;
- one component registry -> live config/ablation/UI;
- one event spine -> ledger/portfolio/decision views;
- one evidence package -> downstream consumer projections;
- one source registry -> all data-source construction;
- one verified store -> exact/ranked reuse views.

This is stronger than ordinary code DRY. It minimizes competing authorities.

### 4.3 Negative witnesses are first-class

Both systems repeatedly use tests that deliberately inject the bad thing:

- forbidden import;
- ambient clock read;
- duplicate default;
- silent fallback;
- malformed DAG dependency;
- fabricated report aggregate;
- missing terminal event;
- identity collision;
- illegal registry option.

For an agent-maintained repository, this protects against a dangerous failure mode: weakening a guard while keeping the nominal positive suite green.

### 4.4 Absence, corruption, unknown, null, heuristic, and failure need distinct representations

The two repositories repeatedly improve when these states stop sharing one convenient value.

That observation is especially relevant to Climate's current placeholder problem. Replacing every incomplete routine with an exception would also be too coarse; the architecture needs enough vocabulary to distinguish an explicitly heuristic approximation from a fake implementation, a missing resource, a corrupt artifact, and a mathematically valid result with unknown empirical meaning.

### 4.5 Evidence should be reconstructable downward

Strong reports and claims can be traced toward lower-level records:

```text
claim/recommendation
  <- self-audited report
  <- raw experiment rows/events
  <- run provenance/config
  <- immutable inputs and implementation identity
```

The reverse direction should not be automatic. A successful run does not promote a scientific claim merely because artifacts exist.

### 4.6 Failure histories are part of the architecture

Both repositories contain architecture that makes little sense if viewed only as static design aesthetics. The reason for the mechanism is encoded in prior failures:

- silent skips hid a broken live test;
- duplicated defaults drifted;
- a demo overwrote a real artifact path;
- an LLM planning stage generated false intermediate structure;
- incomplete jobs lacked terminal events;
- warm sessions produced misleading verification failures;
- cross-repo ownership was duplicated;
- an empty evidence ledger looked like success.

Climate should preserve equivalent failure histories close enough to the rules that future agents do not remove an odd-looking constraint because its original motivation has vanished from context.

## 5. Negative and conditional findings

The source repositories should not be idealized.

### 5.1 KanForge checkpoint load conflates absence and corruption

`RunCheckpoint.load()` returns `null` both when no checkpoint exists and when JSON parsing fails. That may be acceptable because verified lemmas are separately write-through and the checkpoint is recoverable working state, but it would be an unsafe pattern for Climate evidence/run manifests where corruption must remain distinguishable from absence.

### 5.2 KanForge recommendation loading also collapses errors to absence

`loadRecommendedDefaults()` returns `null` for missing, malformed, or unreadable recommendation files. If recommendations are advisory only, this degrades safely to the baseline. Climate should not reuse the pattern for evidence-affecting configuration without a stronger corruption distinction.

### 5.3 KanForge currently uses ambient randomness in predictor exploration

`ProposalEngine.shouldExplore()` uses `Math.random()`. The repository has provenance fields for seeds, but this call is not visibly tied to a recorded seeded generator at this seam. This is acceptable only if the exploration result/event is sufficient for the intended analysis; it is not a suitable pattern for Climate evidence-producing stochastic computation without explicit RNG identity.

### 5.4 KanForge prose can lag implementation

The current root README still describes the skeleton as LLM decomposition while `blueprint/skeleton.js` is now explicitly deterministic and LLM-free after the architecture audit. `docs/README.md` tries to assign one owner per design fact, but descriptive entrypoint prose can still lag.

This reinforces the value of generated status surfaces rather than arguing that documentation discipline alone solves drift.

### 5.5 KanForge has no observed GitHub Actions workflow on main

Its local test and live-kernel harnesses are substantial, but no `.github/workflows` directory is currently present. Climate should distinguish “executable locally” from “automatically gated on every PR.”

### 5.6 Trader's universal skip-is-failure policy is context-specific

Turning every skip into a failure solved a real Trader defect. Climate, however, has explicit resource classes such as CUDA-device, large-data, and integrated-system gates that are intentionally unavailable in some environments. A Climate adaptation would likely need *declared resource inapplicability* rather than treating every non-execution as the same failure.

The transferable invariant is narrower: **a test must never count as positive evidence when its required resource did not execute**.

### 5.7 Trader tolerates torn final event-log append by truncation

This is a documented operational choice and avoids fabricating events. For high-value Climate evidence manifests, a stronger receipt or explicit recovery record may be warranted so recovery itself is part of provenance.

### 5.8 Forward-compatible defaulting can reintroduce hidden semantics

Trader's lifecycle `from_dict` ignores unknown fields and supplies dataclass defaults for missing known fields. This is useful for compatible resumability, but any future field whose absence changes scientific/execution meaning must not gain a semantically arbitrary default merely for backward compatibility.

## 6. Questions the survey raises for Climate

These are questions for the adaptation phase, not answers assumed by this survey.

1. What is Climate's narrowest equivalent of a kernel authority for each class of claim?
   - analytic/reference mathematics;
   - numerical solver correctness;
   - physical conservation;
   - statistical inference;
   - empirical climate interpretation;
   - GPU equivalence/performance.

2. Which graphs should be authored once and projected elsewhere?
   - claim/evidence graph;
   - method/capability graph;
   - module/hazard graph;
   - experiment dependency graph;
   - resource/execution graph;
   - data/provenance lineage.

3. Which Climate parameters are legitimate protocol constants, which should be derived, which should be measured recommendations, and which must be caller-required?

4. How should a method prove causal knob flow so an A/B or meta-experiment cannot unknowingly compare identical computations?

5. Which outputs should be impossible to materialize until all required terminal states/evidence are present?

6. Which retrieval/cache/reuse products remain candidates requiring re-verification, even when sourced from previously trusted artifacts?

7. Which protected data/oracle surfaces must be dependency-partitioned so an agent literally cannot import them from proposal/search code?

8. What should count as corruption versus not-started versus unavailable-resource versus explicitly heuristic versus unknown?

9. Where should Climate use generated documentation/projections so prose cannot overstate live capability?

10. Which recurring agent mistakes deserve executable guards, and which should remain friction signals until repeated evidence justifies machinery?

11. How should CPU reference methods, CUDA implementations, and alternative-language implementations share one semantic contract without sharing implementation code?

12. What is the Climate analogue of a commit/promotion gate: the single boundary that must be crossed before a method/result can become reusable evidence?

## 7. Survey implications, deliberately provisional

Without yet selecting exact Climate mechanisms, the survey suggests that the most useful cross-repository inheritance is probably not a particular registry class or testing library. It is a **harness architecture** with these broad properties:

- creative agents have bounded proposal authority;
- hard authorities are independent from proposal generation;
- important facts have one authored home and derived projections;
- gates contain planted negative witnesses;
- successful-looking substitute values are difficult to create accidentally;
- experimental knobs must demonstrate causal reach;
- result summaries self-audit against raw records;
- stochastic and environmental choices bind identity/provenance;
- protected oracles/holdouts are dependency-partitioned;
- incomplete work cannot leave a current artifact that looks completed;
- retrieval and learned heuristics remain advisory until re-verified;
- failure/friction remains queryable evidence rather than vanishing into logs;
- resource-gated claims cannot be promoted by mocks or skipped execution;
- cross-repository ownership is kept narrow enough that the same transformation is not authoritative in two homes.

These are observations to test against Climate's actual source and scientific requirements. The next step is to continue the survey into the less visible enforcement surfaces and then map candidate mechanisms against concrete Climate failure classes before changing Climate policy or source.

## 8. Primary surfaces inspected

### LLM-Trader

- `AGENTS.md`
- `architecture.py`
- `agent/harness.py`
- `agent/tools.py`
- `runtime/eventlog.py`
- `runtime/lifecycle.py`
- `runtime/episodes.py`
- `signalcore/ingest/sources.py`
- `tests/test_architecture_surface.py`
- `tests/test_no_silent_fallbacks.py`
- `tests/test_no_magic_numbers.py`
- `tests/test_config_single_home.py`
- `tests/test_registry_coverage.py`
- `tests/test_knob_flow.py`
- `tests/test_import_bans.py`
- `tests/test_no_ambient_time.py`
- `tests/test_properties.py`
- `tests/test_published_contracts.py`
- `tests/test_sources_registry.py`
- `tests/conftest.py`
- `tests/test_guard.py`
- `scripts/guard.ps1`
- `docs/09-cross-project-patterns.md`
- `docs/arch-signals.md`
- recent commit history through 2026-09-14.

### KanForge

- `README.md`
- `docs/README.md`
- `docs/architecture-audit.md`
- `kanforge/agent/loop.js`
- `kanforge/agent/proposalEngine.js`
- `kanforge/agent/commitGate.js`
- `kanforge/agent/runRecorder.js`
- `kanforge/agent/roles/critic.js`
- `kanforge/agent/roles/normalize.js`
- `kanforge/core/guardrails.js`
- `kanforge/core/checkpoint.js`
- `kanforge/core/provenance.js`
- `kanforge/blueprint/skeleton.js`
- `kanforge/blueprint/falsify.js`
- `kanforge/blueprint/run.js`
- `kanforge/growth/lemmaStore.js`
- `kanforge/config/registry.js`
- `kanforge/bench/ablation.js`
- `kanforge/bench/reportAudit.js`
- `kanforge/test/architectural.test.js`
- `kanforge/test/blueprint.test.js`
- `kanforge/test/causal.test.js`
- `kanforge/test/development.test.js`
- `kanforge/test/ablation.test.js`
- `kanforge/test/reportAudit.test.js`
- recent commit history through 2026-08-22.
