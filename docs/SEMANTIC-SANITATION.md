# Semantic integrity: no plausible lies

Status: **binding cross-cutting safety policy**.

Climate treats implementations that emit plausible-looking values without implementing their advertised semantics as **semantic hazards**, not ordinary TODOs.

## 1. Governing rule

Prefer, in order:

1. a correct implementation with appropriate evidence;
2. an explicit typed/unambiguous failure saying the capability is unavailable;
3. an absent capability with an in-place specification or blueprint;
4. **never** a plausible-output placeholder whose result can be consumed as though it were the intended feature.

Compilation, API completeness, demos, or apparent end-to-end execution are not reasons to retain fake behavior.

**No implicit semantic substitution is the controlling rule.** A required dataset, observation, calibration/parameter authority, configuration, model/runtime, solver, backend, precision, resource class, preprocessing policy, or other scientific/execution semantic either resolves exactly as declared or remains unavailable/failed. Missingness or capability failure must not trigger an undeclared replacement that keeps the original request or success meaning. Alternatives are valid only as separately declared inputs/identities selected explicitly before execution.

## 2. Hazard classes

### S0 — absent / declared unavailable

No implementation exists, or an entrypoint fails immediately with an explicit `Unavailable`/`NotImplemented` diagnostic before producing scientific output.

This is safe by default.

### S1 — incomplete but non-deceptive

Implementation is partial, but unsupported branches fail explicitly and supported branches have accurately narrow names/contracts.

May remain as prototype.

### S2 — heuristic with explicit heuristic semantics

A deliberately approximate method exists and is *actually intended* to be heuristic. Its output type/name says so, and no stronger claim consumes it.

Requires a hypothesis/experiment contract before evidence use.

### S3 — misleading simplification

Code performs a static/default/simplified operation while presenting the result under the name/type/interface of a materially stronger capability.

Examples:

- zero-filled IMFs returned by an EEMD routine;
- fixed humidity injected into a state advertised as observational state;
- fixed eigenvalues used as if computed conditioning diagnostics;
- a CPU/fake-MPI path retaining the identity of a CUDA/NCCL implementation;
- a local preprocessing/evaluation wrapper advertised as equivalent to a native external system while omitting its configuration, provenance, or failure semantics;
- a project-local model runner that impersonates an Anemoi/ACE/ClimSim or other upstream runtime without binding the exact upstream artifact and execution identity;
- a hard-coded probability transform returned as a physical probability.

**S3 must be removed, renamed/retyped as an explicit heuristic, or converted to fail-closed behavior.**

Capability shadowing is an S3/S4 risk even when the local implementation appears numerically plausible. If Climate needs an externally owned capability, the safe forms are an explicit native integration, a clearly distinct bounded reference implementation, or an unavailable state. A simplified local substitute must never retain the upstream system's semantic identity.

A weaker name or type is **not** a repair strategy for an implementation that failed to meet a stronger contract. Renaming/retyping is permitted only when the narrower operation is itself the scientifically or computationally preferable reusable abstraction for a real task, with a contract worth preserving independently. If the stronger capability is the right tool, its implementation must rise to that contract or remain unavailable. Never choose a weaker API merely because it is easier to make compile, test, or return plausible output.

### S4 — fake/plausible-output placeholder

The implementation is known not to implement the advertised capability and emits values likely to flow downstream as valid results.

Examples include fake forecast skill, uninitialized-data diagnostics, stub scientific transforms that return numerically reasonable arrays, or fabricated calibration values.

**S4 is merge-blocking for any touched path.**

## 3. Fail-closed patterns

Preferred failure forms depend on language:

- Rust: `Result::Err` with a dedicated unavailable/unsupported error; `unimplemented!` only at a clearly unreachable prototype boundary.
- Python: raise a dedicated `NotImplementedError`/domain error before constructing an output artifact.
- Fortran: `error stop` at the entrypoint, or remove the procedure/module from the executable build until a real implementation exists.
- C/C++/CUDA: explicit status/error return before output buffers are presented as valid; poison/debug fills are acceptable only if the API also reports failure and evidence paths reject them.
- Julia/Haskell: explicit error/`Either`-style unavailable result rather than a value inhabiting the successful scientific result type.

A sentinel value such as `0`, `NaN`, `1000`, an empty array, or an identity matrix **is not sufficient by itself** when downstream code can ignore the reason and continue.

## 4. Specification requirement for unavailable capabilities

When a substantial capability is absent or intentionally unavailable, its specification or blueprint should contain only forward-looking information needed to implement it correctly:

- intended scientific capability;
- input/output semantics;
- mathematical or physical definition;
- conventional definitions and strong baselines;
- minimum verification required before activation;
- likely implementation/resource partition;
- links to claim/method/experiment IDs where applicable.

Architecture documents should describe the capability and its obligations directly rather than preserving implementation archaeology.

## 5. Priority ordering

Semantic-integrity priority is based on **contamination risk**, not ease of fixing.

1. fake values feeding authoritative or scientific-looking output;
2. placeholder fallbacks hidden behind successful return paths;
3. mislabeled physical probabilities/confidences/risk/timescale outputs;
4. silent synthetic/default observations or state variables;
5. stub transforms returning arrays/tensors under real algorithm names;
6. fake compatibility layers retaining real capability identity;
7. demos/tests that can be mistaken for validation;
8. ordinary TODOs that already fail closed.

`architecture/check_semantic_defaults.py` is the binding source-level structural guard for implicit semantic substitution across the audited scientific/reference surface. Its historical filename is narrower than its responsibility. `architecture/source_gates.py` retains a fallback-marker scan only as supplemental defense in depth: lexical matches can prompt review, but the presence or absence of the word `fallback` is not evidence that substitution is or is not occurring.

## 6. Interaction with maturity and evidence

A module containing an active S3/S4 path cannot be `verified`, `validated`, `replicated`, or `decision-eligible` for a claim that can reach that path.

Execution identity is the most concrete instance of the general rule and is governed by `#ExecutionResolution` in `contracts/climate.cue`. An eligible execution resolves to exactly the identity that was requested. If a required capability is unavailable, the request is ineligible and carries a typed failure without a replacement resolved identity. A CPU path, alternate solver, lower-precision path, compatibility shim, or other implementation may run only when it is requested under its own execution identity; it is not a runtime fallback for a different request. Data, calibration, configuration, and other scientific inputs obey the same resolve-exactly-or-fail-closed principle at their own authority boundaries.

An unavailable stub or blueprint may coexist with a `concept`/`prototype` method descriptor because it cannot manufacture supporting evidence.

## 7. Enforcement requirements

The semantic-integrity system requires:

- a machine-readable hazard ledger with owner/action/status where that adds value;
- negative witnesses proving guards catch plausible-output placeholders and semantic substitution without depending on the spelling `fallback`;
- a binding structural source guard for implicit defaults, semantic coalescing, availability/exception rewrites, and optional-capability substitution;
- supplemental lexical source gates for suspicious markers, explicitly treated as defense in depth rather than semantic proof;
- contract checks that make eligible requested/resolved execution identity mismatch invalid;
- module/method checks preventing maturity promotion while unresolved S3/S4 hazards are reachable;
- explicit allowlists only for correctly named heuristics, never for fake implementations.

The goal is not zero TODO comments. The goal is zero successful-looking execution paths that lie about what computation occurred.

## 8. Activation rule

A scientific capability may become executable when it has:

1. a narrow contract;
2. an independent reference or benchmark where applicable;
3. explicit failure semantics;
4. tests showing unsupported cases fail closed;
5. evidence appropriate to its advertised maturity;
6. no need for fabricated data/results to keep an end-to-end demo green.

A smaller truthful system is preferred to a larger system whose apparent completeness contaminates future reasoning.
