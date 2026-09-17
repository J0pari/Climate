# Climate GPU engineering specification

Status: **architectural execution specification**. This document defines the GPU execution discipline for Climate: explicit device residency, transfer schedules, deterministic reductions, device-safe identities, source gates, and evidence-bearing hardware metadata.

The behavioral/scientific meaning of methods lives in `docs/ARCHITECTURE.md`, `docs/VALIDATION-AND-EVIDENCE.md`, and method-specific contracts. This document specifies how accelerated implementations must execute when their results are used as scientific evidence. It does not decide which scientific workload should be accelerated first.

## 1. Core principle

**Host authority; device throughput.**

The host/control layer owns:

- experiment identity and configuration;
- dataset selection and provenance;
- scientific interpretation;
- validation policy;
- claim/evidence promotion;
- resource requests;
- artifact registration;
- stopping/cancellation decisions.

The GPU owns explicitly bounded numerical work:

- dense/sparse tensor transforms;
- stencil and local differential operators;
- batched linear algebra;
- reductions;
- spectral transforms;
- ensemble/candidate evaluation;
- selected optimization/autodiff kernels.

A device result must never promote its own scientific interpretation. A kernel may produce `scalar_curvature`; it does not thereby produce `tipping_probability` unless a separately versioned and validated mapping contract exists.

## 2. Why Climate needs a binding GPU spec

Climate's ambitious methods are vulnerable to confusing numerical artifacts with scientific effects. Accelerated paths can combine mathematical transformations, precision choices, factorization, reductions, batching, communication, fallbacks, and resource behavior in ways that make it difficult to distinguish:

- mathematical changes from implementation changes;
- precision effects from hypothesis effects;
- fallback execution from intended hardware execution;
- transfer overhead from kernel cost;
- nondeterministic reductions from unstable science;
- an accelerated implementation failure from a failed scientific idea.

The GPU implementation must therefore be independently specified and differentially testable against reference implementations.

## 3. Workload selection precedes accelerator architecture

No scientific family is the default first GPU target. Acceleration begins only after a useful experiment or numerical path has a demonstrated bottleneck that GPU execution can plausibly remove without changing its scientific semantics.

A geometry/multirepresentation workload is one possible candidate if profiling and experiment demand justify it. In that case a purpose-built subsystem should be a **batched geometric experiment engine**, not a monolithic "curvature application".

Conceptually:

```text
DatasetProjection
      +
GeometryCandidate[N]
      |
      v
state/feature projection
      |
metric construction
      |
metric derivatives
      |
conditioning + inversion
      |
connection coefficients
      |
curvature/contractions
      |
invariant extraction
      |
benchmark metrics
      |
MetricResult[N]
```

`GeometryCandidate` may vary such things as:

- coordinate transform;
- normalization;
- metric family;
- diagonal/full/sparse metric structure;
- physically specified versus statistically learned couplings;
- derivative method;
- regularization;
- invariant selection;
- precision policy.

This design accelerates **hypothesis competition**, not only one favored geometric formulation. It becomes relevant only if the corresponding CPU/reference experiment is scientifically useful and computationally constrained.

## 4. Other suitable batched GPU workloads

The same execution philosophy can support different engines without forcing a single universal kernel architecture.

### 4.1 Information geometry

Batch over parameter candidates, ensemble members, observational subsets, likelihood variants, or Fisher approximations. Maintain a high-level JAX/Python reference path and compare the native GPU implementation against it.

### 4.2 Ultrametric/p-adic teleconnection experiments

Keep feature/time-series data resident and evaluate multiple distance families in matched batches:

- standardized Euclidean;
- correlation distance;
- spectral/coherence distance;
- graph distance;
- learned metric;
- generic ultrametric;
- p-adic encodings.

GPU acceleration is valuable here only when it makes scientifically useful comparison over many representations materially cheaper.

### 4.3 Latent-dimension / representation search

Evaluate many candidate embeddings or added dimensions against held-out predictive, geometric, and numerical criteria. The device layout should support candidate batching without importing any specific search policy.

### 4.4 Topological/sheaf methods

Only after the mathematical implementation uses real complexes/boundary operators and profiling identifies a suitable workload, compile host-side structures into compact device arrays for sparse operations and large fault-injection ensembles. Do not port the abstract object model wholesale to CUDA.

## 5. Device context pattern

Each accelerated subsystem must expose a bounded context whose allocations and lifetimes are inspectable.

A geometric implementation, if justified, could converge toward a shape such as:

```text
CurvatureDeviceContext
  immutable configuration
  device capability record
  stream(s)
  input state batches
  candidate descriptors
  metric workspace
  derivative workspace
  inverse/factorization workspace
  connection workspace
  curvature workspace
  reduction workspace
  output metric/result buffers
  diagnostics/error buffers
```

The context owns allocations. Individual kernels do not perform hidden long-lived allocation.

Every persistent buffer must document:

```text
name
type / precision
logical shape
byte size
layout/order
owner
lifetime
producer(s)
consumer(s)
initialization rule
transfer points
whether contents affect evidence
```

When implementation begins, this table becomes generated or mechanically checked where practical.

## 6. Memory policy

### 6.1 Prefer explicit device memory

Production evidence paths should prefer explicit device allocation and explicit transfers. Managed memory may be useful in exploratory prototypes, but it must not silently become the validated performance/correctness path because page migration can obscure residency and timing behavior.

### 6.2 Pinned host buffers

Use pinned host buffers for planned high-frequency asynchronous transfers when profiling justifies them. Do not pin large memory regions indiscriminately.

### 6.3 Allocation lifetime

Allocation should occur at context/setup boundaries, not inside tight iteration loops. Large workspace requirements must be estimable before execution so Commons or another scheduler can make an honest resource decision.

### 6.4 Artifact arrays are not event payloads

Large GPU outputs should become content-addressed artifacts. Event/evidence records carry digests, shapes, schemas, and provenance rather than embedding arrays.

## 7. Transfer schedule

Every evidence-producing GPU pipeline must document its expected host/device transfer schedule. Normal operation should minimize synchronization and avoid repeatedly transferring bulk state merely for host-side convenience.

For a geometric engine a desirable pattern is:

```text
T0 setup
  H->D immutable configuration/candidate descriptors

T1 input batch
  H->D normalized state/features (or device-local decode if justified)

D-only pipeline
  project -> metric -> derivatives -> factor/invert -> connection
  -> curvature -> invariant/diagnostic reductions

T2 compact results
  D->H MetricResult / diagnostics / selected summaries

T3 exceptional artifact extraction
  D->H full tensors only when explicitly requested for inspection/evidence
```

Full Riemann tensors should not cross the bus on every candidate/run if only contractions are needed for the declared metric.

Transfers are part of profiling evidence. A speedup claim that excludes dominant data movement must say so explicitly.

## 8. Kernel decomposition

Prefer phase-decomposed kernels at real synchronization boundaries rather than one enormous kernel with hidden responsibilities.

A possible geometric decomposition is:

1. `project_state_kernel`
2. `construct_metric_kernel`
3. `metric_derivative_kernel`
4. library-backed or custom factorization/inversion
5. `connection_kernel`
6. `curvature_kernel` or contraction-specific kernels
7. `invariant_reduce_kernel`
8. `diagnostic_kernel`
9. optional `refine_selected_kernel`

This naming is illustrative, not binding API. The binding principle is that each stage has a testable input/output contract.

Kernel fusion is permitted when profiling demonstrates benefit and a differential witness proves semantic equivalence.

## 9. Library-versus-custom-kernel rule

Do not write custom CUDA merely because a quantity is mathematical.

Prefer established libraries for operations they implement well:

- cuBLAS/cuBLASLt for dense linear algebra;
- cuSOLVER for decompositions/solves;
- cuSPARSE for sparse operations;
- cuFFT for FFT workloads;
- CUB for reductions/scans where appropriate.

Custom kernels are justified when:

- layout/domain structure makes a library path materially inefficient;
- fusion avoids dominant memory traffic;
- deterministic behavior requires a controlled reduction;
- the operation is genuinely domain-specific;
- profiling demonstrates a bottleneck.

Every custom replacement for a trusted library path should have a differential benchmark against the reference path.

## 10. Precision policy

Precision is part of method semantics whenever it can change scientific conclusions.

A candidate policy for ambitious geometry is:

```text
FP16/BF16
  optional bulk feature projection / candidate screening

FP32
  common exploratory tensor operations and reductions where conditioned

FP64
  condition-sensitive factorization/inversion
  derivative verification
  final invariants used for numerical validation

CPU/reference precision
  selected verification witnesses and difficult conditioning cases
```

This is not permission to use mixed precision everywhere. Each method must declare:

- input precision;
- accumulator precision;
- output precision;
- refinement rule;
- condition-number or residual trigger for escalation;
- error tolerance against the reference implementation.

A fast low-precision path and a scientific reference path should be separate identities when their numerical semantics differ materially.

## 11. Conditioning and singularity handling

Metric-based methods are especially sensitive to near-singular matrices.

Every inversion/solve path used for evidence must record or bound:

- factorization status;
- condition estimate or a documented proxy;
- regularization/jitter;
- residual norm;
- positive-definiteness test where required;
- regularization/refinement behavior.

No kernel may silently turn singularity into a zero tensor, identity matrix, clamped probability, or successful scientific result.

Conditioning failures are first-class diagnostic outputs.

## 12. Determinism classes

Every evidence-producing accelerated method declares one of three reproducibility classes:

### D0 — bitwise deterministic

Same declared hardware/software configuration and inputs produce bitwise-identical outputs.

Use this for discrete bookkeeping and reductions that determine regression/evidence state where achievable.

### D1 — numerically deterministic

Results may differ at floating-point bit level but remain within a predeclared numerical tolerance and do not alter categorical/scientific conclusions under the test.

### D2 — stochastic

Randomness or nondeterministic execution is intentional. Seeds, RNG algorithm/state identity, sample counts, and uncertainty aggregation are recorded.

"GPU nondeterminism" is not a sufficient reproducibility specification.

## 13. Deterministic reductions

If a reduction directly affects:

- candidate ranking;
- promotion/rejection;
- a regression witness;
- a reported scientific metric;
- conservation/budget closure;

then deterministic reduction order should be preferred when practical.

Avoid unconstrained atomic floating-point accumulation for such values unless the method explicitly declares D1/D2 behavior and sensitivity analysis proves decision stability.

Use fixed tree reductions, warp/block reductions with defined composition, or library modes with understood reproducibility behavior.

## 14. Device-safe strong identities

Raw integer identities are prohibited at important CUDA boundaries once typed infrastructure exists.

Candidate identity families include:

```text
DatasetId
ExperimentId
RunId
MethodId
CandidateId
GridId
EnsembleMemberId
ArtifactId
```

C++/CUDA implementations should use trivially-copyable strong wrappers with explicit construction/extraction when this prevents cross-domain identity mistakes at zero meaningful runtime cost. Indices that are purely local numeric loop coordinates (e.g. `i`, `j`, vertical index) need not become global identities.

The purpose is to prevent a valid integer from the wrong domain silently addressing another buffer or record.

## 15. RNG policy

Evidence paths must not use ambient process RNG state.

Every stochastic GPU method must declare:

- RNG algorithm;
- master seed;
- derivation scheme for candidate/member/thread/subsequence seeds;
- whether execution order can change generated values;
- replay requirements.

Prefer counter-based or otherwise splittable RNG schemes when parallel ordering must not alter streams.

The RNG identity belongs in `RunManifest`.

## 16. CUDA error policy

Unchecked CUDA/runtime/library calls are forbidden in production evidence paths.

Required patterns:

- checked allocation/free;
- checked copies and memset;
- checked stream/event creation;
- checked library calls;
- kernel launch error checks;
- synchronization errors surfaced at explicit boundaries;
- OOM reported as resource failure, not numerical/scientific failure.

Wrappers should distinguish setup/resource errors from method output failures.

A diagnostic mode should additionally support synchronization after selected phases to localize asynchronous failures without imposing that cost on every production run.

## 17. Fail-closed capability identity in evidence runs

Compatibility shims may keep source parseable or support explicitly labeled exploratory builds, but they cannot satisfy a different requested execution identity. `#ExecutionResolution` in `contracts/climate.cue` owns the requested/resolved identity contract for evidence-producing execution.

An accelerated execution is eligible only when the resolved method, implementation build, backend, precision, and resource class exactly match the requested identity. If a requested capability is unavailable, the execution is ineligible with a typed failure and no replacement resolved identity.

Examples:

- an NCCL stub is not NCCL;
- a no-op MPI shim is not distributed execution;
- a manual fallback is not a tensor-core implementation;
- CPU fallback is not a GPU result;
- emulated precision is not native precision.

An alternate implementation may execute only after it is explicitly requested under its own execution identity. Capability discovery may reject a request; it may not rewrite the request to something executable.

Compile-time `REQUIRE_REAL_*` checks are useful, but runtime provenance must still record the resolved environment.

## 18. Resource envelopes

Every accelerated `MethodDescriptor` should eventually declare an estimable resource envelope:

```text
min_compute_capability
required libraries/features
peak VRAM estimate
host RAM estimate
expected temporary workspace
GPU count
multi-GPU requirement/optional status
expected runtime scale model
transfer volume estimate
precision modes
```

Commons should schedule from these declarations rather than parsing CUDA source or trusting ad-hoc prose.

A run that exceeds its declared envelope produces evidence that the envelope is wrong and should not silently expand without record.

## 19. Device evidence metadata

A GPU-backed `RunManifest` must record at minimum when available:

- GPU model;
- compute capability;
- GPU count;
- driver version;
- CUDA runtime/toolkit version;
- relevant library versions;
- kernel implementation/build fingerprint;
- precision mode;
- determinism class;
- stream/concurrency policy where material;
- device resource peak estimates/measurements;
- direct versus Commons-scheduled execution.

Performance evidence additionally records warm-up policy, timing method, repetitions, and whether transfers/setup are included.

## 20. Differential/reference implementation rule

A high-performance GPU implementation should not be its own sole oracle.

For each important accelerated method, maintain at least one independent witness appropriate to scale:

- CPU scalar/reference implementation;
- NumPy/JAX/Julia reference;
- analytic manufactured case;
- high-precision implementation;
- alternate library algorithm.

CI may exercise small fixtures. Larger hardware validation can run separately, but its evidence must bind the same contract/build identity.

## 21. Acceptance tests for accelerated kernels

Before an accelerated implementation can become `verified`, it should usually pass:

1. shape/layout tests;
2. invalid-input and OOM/error propagation tests;
3. tiny exact/analytic fixtures;
4. differential reference tests;
5. precision sweep;
6. conditioning/adversarial cases;
7. determinism-class test;
8. large-enough stress test to exercise indexing/layout;
9. performance characterization without changing scientific tolerances;
10. sanitizer/tooling pass where practical.

For PDE/discretization kernels also include convergence and conservation witnesses.

## 22. Source gates to implement

Climate should adopt machine-enforced source gates specific to scientific computing.

Initial targets:

- no unchecked CUDA/runtime/library calls in production accelerated source;
- no `cudaMallocManaged` in evidence-critical loops unless explicitly allowlisted by contract;
- no ambient RNG in evidence-producing numerical source;
- no anonymous live numerical tolerance/tunable at declared seams;
- no runtime capability substitution in validated/evidence mode;
- no placeholder/stub marker in a method whose descriptor says `verified` or above;
- no metric inversion/solve path without failure/conditioning handling;
- no evidence-affecting reduction with undeclared reproducibility class;
- no new GPU executable bypassing the resource authorization/scheduler adapter once that adapter exists.

Each source gate must have a planted negative test proving it catches the violation. A guard with no negative witness is only an aspiration.

## 23. Multi-GPU policy

Do not start with multi-GPU merely because a candidate implementation can name NCCL/MPI concepts.

First establish a single-GPU implementation with:

- correct memory accounting;
- deterministic/reproducible behavior;
- reference equivalence;
- evidence manifests;
- realistic profiling.

Only introduce distributed execution when a declared experiment cannot reasonably fit or complete on the target single GPU. Multi-GPU decomposition then requires its own numerical equivalence tests, halo/exchange contracts, failure semantics, and evidence metadata.

## 24. Profiling philosophy

Optimization follows measurement.

For each targeted engine capture:

```text
end-to-end wall time
setup/allocation time
H2D / D2H volume and time
per-stage kernel/library time
occupancy / launch configuration where useful
memory-bandwidth or compute limitation
peak VRAM
candidate throughput
energy/power only if it becomes an explicit research metric
```

Prefer optimization that reduces total experiment cost or enables stronger scientific experiments. A locally faster kernel that increases scientific fragility is a regression.

## 25. Activation sequence for a geometry accelerator

Use this sequence only after a geometry/multirepresentation workload has demonstrated scientific usefulness and a real acceleration need:

1. identify the smallest mathematically coherent high-cost slice;
2. establish an independent CPU/reference witness for that slice;
3. define explicit input/output structures, layouts, and precision;
4. select maintained CUDA libraries for generic operations before considering custom kernels;
5. introduce checked CUDA wrappers and a bounded device context;
6. implement the minimum device stages needed by the experiment;
7. add conditioning/residual/failure diagnostics;
8. differential-test GPU versus the independent reference;
9. profile end to end, including transfers and setup;
10. only then consider fusion, tensor-core use, or custom replacements for library operations.

No compatibility path or acceleration feature is retained merely because it already exists; every active path must satisfy the current method contract.

## 26. Relationship to Commons

Climate owns GPU numerical semantics. Commons owns cross-repository scheduling/resource arbitration and shared run/causation identity.

Climate should expose:

```text
resource request
stable experiment/method entrypoint
progress events
run/artifact/evidence results
```

Climate should not grow a second global GPU scheduler. Local streams, kernel scheduling, intra-run multi-GPU coordination, and numerical checkpointing remain Climate concerns.

When Commons resource control is available, production GPU evidence should execute through it except for an explicit direct-development escape path that records itself as direct execution.

## 27. Definition of success

The GPU architecture is working when:

- a scientific method can acquire acceleration without changing its scientific contract;
- multiple candidates can be evaluated efficiently when batching is actually part of the workload;
- device residency and transfers are inspectable;
- GPU and reference implementations can disagree loudly;
- precision/conditioning failures are diagnostics rather than scientific outputs;
- hardware/software provenance travels with evidence;
- a faster implementation cannot silently alter claim maturity;
- Commons can schedule the work without knowing CUDA internals.