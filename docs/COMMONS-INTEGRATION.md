# Commons integration contract

Status: **contract-ready read integration**. Climate can now submit its existing CPU experiment runtime through Commons `work-scheduler/v1`; live R4 completion still requires a real machine cutover/daemon witness.

The purpose of this interface is to let Commons coordinate Climate without owning Climate's scientific semantics or silently upgrading the strength of its evidence.

## 1. Division of responsibility

### Climate owns

- climate-domain schemas and units;
- dataset identities and preprocessing semantics;
- method descriptors;
- experiment questions, baselines, metrics, falsifiers, and validation policy;
- interpretation of climate-specific outputs;
- scientific claim maturity;
- benchmark definitions;
- domain-specific artifact formats.

### Commons owns at the shared boundary

- repository identity and access policy;
- cross-repository `RunId` and correlation/causation identifiers;
- scheduling and resource leases;
- process sandboxing;
- execution receipts;
- artifact digests and durable lineage edges;
- common event-envelope semantics;
- contract fingerprints and compatibility checks;
- cross-repository evidence taxonomy;
- workspace-level experiment DAGs and promotion decisions where Climate is one evaluator/producer among others.

Neither side should duplicate the other's source of truth. Climate pins the resource-generic Commons `work-scheduler/v1` ABI in `contracts/work-scheduler-pin.json` while retaining the legacy GPU pin only for compatibility history; `architecture/commons_interface.json` is Climate's machine-readable declaration of its side of this boundary.

## 2. Control levels

Climate's integration should progress explicitly.

### `observe`

Commons may:

- inspect repository/ref metadata;
- validate declared paths/contracts;
- read method/experiment descriptors;
- fingerprint contracts;
- collect static findings.

Commons may not execute Climate code or alter the repository.

### `read`

In addition to `observe`, Commons may execute explicitly declared read-only gates/experiments in a sandbox with bounded resources and capabilities.

No repository writes, remote pushes, secret access, or unrestricted network access are implied.

### `write`

Commons may alter the repository only after:

- the repository has a binding contract;
- the relevant execution gates are reproducible;
- receipts/artifacts are immutable and attributable;
- write scope is explicit;
- protected paths/actions are defined;
- rollback/recovery policy exists;
- the write path conforms to `AGENTS.md`, including the direct-`main`, small-coherent-commit, no-history-rewrite policy.

Climate does not use feature branches or PR staging as its repository-write mechanism. If Commons cannot satisfy the direct-`main` policy safely and explicitly, it remains at `read` rather than introducing a conflicting branch workflow.

## 3. Required Climate-facing descriptors

The first stable machine-readable surface should eventually include:

### Repository descriptor

```text
repository_id
repository/build contract version
maturity
supported control level
canonical validation entrypoints
resource capabilities
artifact roots/policy
```

### Method descriptors

See `docs/ARCHITECTURE.md`. Commons should not infer scientific method identity from filenames.

### Experiment specs

Commons should schedule an `ExperimentSpec` or an experiment DAG compiled from one, not a free-form shell command when scientific meaning matters.

### Gate descriptors

A gate should be structured:

```text
Gate {
  gate_id
  purpose
  command/entrypoint
  working_directory
  environment contract
  network policy
  filesystem policy
  GPU/CPU/memory limits
  timeout
  expected artifacts
  evidence class/status produced
}
```

A gate saying only `python script.py` is insufficient for durable automation.

## 4. Commons event envelope usage

Climate durable events should progressively use the shared Commons envelope while keeping climate payloads domain-specific.

Candidate event types:

```text
climate.dataset.materialized.v1
climate.dataset.validated.v1
climate.method.verified.v1
climate.experiment.started.v1
climate.experiment.completed.v1
climate.benchmark.completed.v1
climate.claim.supported.v1
climate.claim.attacked.v1
climate.method.promoted.v1
climate.method.rejected.v1
```

Do not create events merely because an event bus exists. Durable events should represent meaningful state/evidence transitions.

Use Commons fields such as:

```text
run_id
trace_id
correlation_id
causation_id
producer_build
contract_fingerprint
schema_version
semantic_version
```

Climate payloads should retain domain clocks/time coordinates rather than overloading envelope creation time.

## 5. Evidence mapping

Commons' evidence taxonomy should be treated as a coarse cross-repository vocabulary.

Climate may need richer domain detail inside the payload, but must not map upward incorrectly.

Examples:

- ERA5-derived measured field used in analysis -> `Observed` (while retaining that it is a reanalysis/model-assimilated product in Climate metadata);
- CMIP simulation output -> `Simulated`;
- alternative historical policy/model replay -> `Counterfactual` where appropriate;
- method performance on held-out benchmark -> `Behavioral` or `Simulated` depending on the claim and benchmark;
- unvalidated curvature/p-adic/symmetry indicator -> `Heuristic` even if computed exactly;
- mathematical proof about an algorithm -> `Formal`, but that does not make the climate interpretation formal evidence.

Verification status remains orthogonal.

## 6. Artifact lineage

Large outputs should be external artifacts referenced by digest. Commons should be able to follow a lineage such as:

```text
ERA5 source identity
 -> Climate DatasetProjection@digest
 -> ExperimentSpec@digest
 -> RunId
 -> CandidateArtifact@digest
 -> MetricResult@digest
 -> EvidenceRecord@digest
 -> ClaimId
```

Cross-repository lineage might later extend:

```text
Climate benchmark evidence
 -> training projection
 -> adapter
 -> Climate reevaluation
```

If Climate evidence is ever used for model training, the distinction between observed, simulated, counterfactual, and heuristic sources must survive projection.

## 7. Resource interface

Climate workloads vary from tiny statistical tests to GPU/HPC jobs. Do not encode this as repository-specific scheduler code.

An experiment/gate should declare resources such as:

```text
cpu_cores
memory_bytes
gpu_count / GPU capability
scratch_bytes
expected wall time class
network requirement
dataset locality requirement
```

Commons may translate these declarations into concrete leases. Climate should not assume a particular machine, GPU id, or cluster topology in its scientific spec.

`work-scheduler/v1` now represents the current CPU experiment honestly: the Climate adapter submits `resourceClass=cpu`, a positive RAM declaration, the exact Climate experiment spec, explicit repository revision and run scope, and writes only beneath `run-artifacts/commons/`. CPU submission carries zero GPU claim and must not depend on GPU availability. The older `gpu-scheduler/v1` pin remains compatibility history, not the Climate execution route.

## 8. Sandboxing

Climate's multi-language code and data clients can execute arbitrary code and access remote services. Read execution therefore needs explicit capabilities.

Default sandbox posture for automated gates:

- repository checkout read-only;
- separate writable scratch/artifact directory;
- no inherited secrets;
- network denied unless gate declares specific need;
- bounded process spawning;
- bounded CPU/memory/time;
- GPU lease only when requested;
- captured stdout/stderr;
- full resolved environment receipt.

Data-acquisition gates may need network/secrets and should be separated from pure evaluation gates so cached immutable inputs can be evaluated offline.

## 9. Compatibility and contract fingerprints

Commons should fingerprint Climate's external contracts rather than internal source layout. Climate reciprocally verifies the Commons scheduler schema, owner, and ABI fingerprint before trusting its read-only control surface.

Changes requiring deliberate compatibility review include:

- dataset schema/identity semantics;
- experiment spec schema;
- method descriptor schema;
- metric semantics;
- event payload schemas;
- artifact format contracts;
- public gate entrypoints.

Moving an internal Rust file should not move a cross-repository fingerprint unless an external contract changed.

## 10. First Commons-compatible entrypoints

Do not start with dozens of commands. Target a small interface:

```text
climate inspect
climate validate-fixtures
climate verify-method <method_id>
climate run-experiment <experiment-spec>
climate summarize-run <run-manifest>
```

These names are illustrative. The important property is stable semantics and machine-readable outputs.

The first implemented entrypoint should likely be `inspect`: report repository/build structural truth, known placeholders, available toolchains, and declared methods without executing science workloads.

## 11. Integration gates

### Gate A — observable

- binding `AGENTS.md` exists;
- repository descriptor resolves;
- structure can be inspected without executing code;
- known structural defects are emitted as findings;
- Climate is not a Commons governance actor merely because it is registered.

### Gate B — sandbox-readable

- at least one deterministic fixture suite runs in an isolated environment;
- exact environment and command are receipted;
- no undeclared network/secrets/write access occurs;
- outputs are schema validated.

### Gate C — scientific experiment capable

- `ExperimentSpec`, `RunManifest`, `MetricResult`, and `EvidenceRecord` exist;
- baseline and candidate runs use the same orchestration semantics;
- negative controls can be represented;
- failed runs remain first-class evidence.

### Gate D — resource managed

- GPU/CPU/memory resource requests are explicit;
- Commons leases resources transactionally;
- cancellation/worker death cannot leave Climate resources indefinitely owned;
- checkpoint/restart semantics are declared per workload.

### Gate E — write capable

- direct-`main` write policy and authorization are explicit;
- every write is a small coherent commit causally linked to a work order/experiment/issue;
- the target file is re-read from current `main` immediately before writing;
- post-write gates run and are receipted;
- rollback uses new commits rather than history rewriting;
- write authority is least-privilege.

## 12. What Commons must not do

Commons must not:

- classify a Climate method as validated because its gate passed;
- choose physical parameter values as scheduler policy;
- infer evidence class from filenames or language;
- flatten all Climate uncertainty into one confidence number;
- make a scientific claim stronger during cross-repo projection;
- silently rerun a failed scientific job with altered parameters and treat it as the same experiment;
- treat reanalysis, observation, simulation, and counterfactual output as interchangeable.

## 13. What Climate should not duplicate

Once Commons provides these facilities, Climate should avoid building separate incompatible versions of:

- global run IDs;
- workspace-level job scheduling;
- cross-repo GPU leases;
- generic causal lineage;
- generic artifact-addressing conventions;
- generic experiment DAG orchestration;
- cross-repository promotion bookkeeping.

Climate may retain local component scheduling where it is scientifically/numerically part of a simulation, but that distinction must be explicit.

## 7. External artifact evaluation

Climate pins Commons `evaluation-exchange/v1` as an interoperability boundary,
not as scientific authority. A Training model artifact can be presented as an
external artifact reference only when it carries its producer repository,
producer-local alias, full SHA-256 identity, and artifact contract. The local
16-character Training alias is metadata; it is never the cross-repository
identity.

Climate now registers a first deliberately narrow evaluator for
`training.model-artifact/v1`: `external.training_artifact.climate_contract_reasoning.v1`.
Its immutable task set checks reasoning over Climate's binding evidence,
execution-identity, contamination, and cross-repository promotion boundaries.
The evaluator is behavioral only; its specification explicitly excludes
climate prediction, observational skill, physical validity, and Training
promotion.

The evaluator runtime remains unavailable. `architecture/commons_control.py`
resolves the registered specification and then fails closed because
`climate.external-model-runtime/v1` is not yet implemented. That is the
remaining execution blocker; no attestation is issued without a runtime that
binds the exact requested Training artifact digest.

Any eventual evaluation attestation is evidence about the immutable model
artifact. It does not mutate that artifact and cannot accept, reject, deploy,
or promote it; Training retains those decisions.


## 8. Live read-execution witness

The R4 integration witness is executable rather than a prose checklist:

```text
python architecture/commons_live_witness.py \
  --experiment experiments/multirepresentation-ebm-dynamics.v1.json \
  --repository-revision <current-Climate-git-sha> \
  --run-scope <unique-scope> \
  --ram <MiB>
```

The witness submits through `work-scheduler/v1`, polls only the public
`inspect --job` lifecycle, requires a successful terminal job, then validates
that Climate's `outcome.json` has the requested experiment identity and
repository revision. It never reads Commons private queue/state files.

A passing mocked/CI witness proves the harness semantics only. R4 completion
still requires this command to succeed against the machine-local Commons
daemon after the one-writer scheduler-state cutover.
