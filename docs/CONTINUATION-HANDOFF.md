# Continuation handoff

This is a compact handoff for the next agent working on `J0pari/Climate`. It is a snapshot, not a second status authority. **Before acting, fetch the current `main` head and reread the live authorities below. If `main` has advanced, do not reset or overwrite newer work.**

Live authorities:

- `docs/generated/STATUS.md` — generated module/claim/experiment/sheaf status; do not hand-edit.
- `docs/FORTRAN-PHYSICS-FRONTIER.md` — current Fortran physical/numerical realization boundary.
- `docs/ROADMAP.md` — manually curated research priorities and dependency graph.
- `AGENTS.md` — binding contributor/scientific-evidence rules.
- `architecture/modules/*.json`, `claims/registry.json`, `experiments/*.json`, `methods/sheaf-realization.v1.json` — machine authorities behind the projections.

## Snapshot at handoff

Snapshot date: 2026-09-16.

Canonical branch: `main`.

Pre-handoff canonical head: `9c22b67ee2dcacb32b6aaeb33ba0bed22f12923a` (`Record conservative transport frontier`). This handoff file itself is committed on top of whatever `main` head GitHub accepted at creation time, so re-fetch `main` before using the SHA as anything other than a historical anchor.

At `9c22b67`, generated status reports:

- 49 registered modules: 20 canonical, 24 legacy, 5 reference;
- 23 runnable, 26 prototype;
- all 5 scientific claims remain `concept` with zero supporting evidence records;
- 2 registered experiments;
- sheaf realization: 8/15 obligations reference-realized; all R3–R6 climate-data/empirical obligations remain open.

Exact-head GitHub Actions run `35149951052` on `9c22b67` had these jobs green: architecture/CUE/status drift, Rust canonical + legacy compatibility, canonical Fortran, reference geometry, reference Noether, and reference sheaf cohomology. The only failing job was `fortran-legacy-physics`, the intentionally isolated legacy monolith compile probe. Therefore the workflow-level conclusion is red even when the canonical surface is healthy; inspect jobs rather than equating the aggregate conclusion with canonical failure.

The generated-status checker now prints a unified diff when a projection is stale, and `tests/architecture/test_render_status.py` regression-tests that diagnostic. That test commit is already an ancestor of the current `main` history; do not re-port it.

## Working style the user expects

Act rather than repeatedly asking for permission. Use small coherent commits directly on `main`, verify exact heads in GitHub Actions, and never force-update or rewrite history. Do not assemble work on feature branches, and do not merge old draft PRs merely because they exist.

Edits should be granular and evidence-preserving. Do not broadly rewrite documents or architecture merely to make them conform to a simplified new plan; change the smallest stale or contradictory surface, preserve learned lessons that remain valid, and make disagreements inspectable through narrow commits.

The user strongly rejects temporary/band-aid implementations where the harder final-form abstraction is already knowable. Do the real thing rather than building a knowingly disposable surrogate.

Do **not** simplify semantics merely to make code easier. The repository may gain nuance, technical detail, and semantic structure; it should not smooth difficult distinctions away. Reuse useful legacy material through extraction/reshaping where appropriate, but legacy code is evidence of intended responsibilities, not an authority for architecture or algorithms.

## Scientific and mathematical rules

A strong mathematical name creates an implementation obligation. If the named structure is useful and tractable, realize it correctly and test its defining laws. Do not rename downward merely because the present implementation is weak. Use a weaker abstraction only when it is genuinely the superior scientific/computational tool for that responsibility.

Examples already applied:

- real finite cellular-sheaf operators plus exact cohomology reference, rather than threshold counts called Betti numbers;
- exact finite p-adic residue/ultrametric primitives, while climate-to-p-adic encoding remains an empirical hypothesis;
- actual Kripke semantics with modal-law witnesses;
- actual sparse Clifford algebra rather than decorative gamma-matrix language;
- exact Noether/variational-symmetry reference, while open forced dissipative climate diagnostics use balance-law residuals where those are the better physical object;
- explicit-likelihood Fisher information and Amari alpha-connections, with numerical regularization kept separate from the mathematical Fisher object;
- canonical Levi-Civita geometry separated from any unvalidated climate metric or curvature-to-tipping interpretation.

Formal Lean is a narrow independent mathematical authority. Checked-in proofs establish the scalar sheaf edge obligations O1–O3 and arbitrary-matrix Gram orientation invariance. Formal proof establishes consequences of the declared object; it does not validate climate stalks/restrictions or empirical usefulness.

Use Popperian discipline: every novel method must be testable, ablatable, composable, have declared falsifiers/baselines, and earn its scope. Mathematical correctness is not empirical climate evidence. Performance is not scientific validation.

## Architecture rules

Keep the thin semantic waist and explicit authority boundaries. Governance actors, repositories, components, methods, runs, artifacts, claims, and evidence are distinct identities.

Separate:

1. mathematical/reference authority;
2. canonical portable executable implementation;
3. accelerated/GPU implementation;
4. empirical/scientific interpretation.

Accelerators must be differentially verified against an independent reference/canonical path. Never let GPU speed become evidence for scientific validity.

Generated documentation should only project facts already owned by machine-readable authorities. Human research priorities, interpretation, and scientific judgment remain manually curated. Do not build a documentation compiler for things that inherently require judgment.

## Library-versus-novelty rule

Do not reinvent mature generic numerical infrastructure. Hand-written kernels are appropriate as transparent reference/differential oracles; production generic numerics should usually delegate to maintained libraries through thin semantic wrappers.

Current intended boundaries:

- LAPACK `DGTTRF/DGTTRS` for production-oriented CPU tridiagonal solves; Thomas remains a transparent reference oracle.
- SUNDIALS ARKODE is the preferred production trajectory for adaptive/implicit/IMEX/multirate integration unless a concrete requirement proves another choice better.
- FFTW for future production CPU FFT and cuFFT for GPU; direct DFT remains the reference oracle.
- evaluate RTE+RRTMGP or another established maintained package before implementing generic radiative-transfer/gas-optics/two-stream infrastructure locally.
- PETSc becomes relevant if/when distributed state vectors, MPI domain decomposition, large sparse operators, and serious preconditioning actually justify it; do not add it speculatively.

Climate-owned effort should concentrate on domain-specific variables, units, coordinate semantics, operator construction, conservation/balance laws, Jacobians/preconditioners with atmospheric structure, reproducibility, evidence, and scientifically meaningful experiments.

## Canonical Fortran state

The legacy `climate_physics_core.f90` remains intentionally non-authoritative and non-compiling. Do not make it green by inventing fields or patching parser errors without independently specifying the physical contract first. Its compile probe is a debt sensor.

Canonical Fortran slices currently include:

- fixed-step RK4 reference integration;
- transparent no-pivot tridiagonal/theta oracle;
- LAPACK partial-pivoting tridiagonal backend;
- exact constant-f Coriolis rotation;
- nonuniform geometric-height finite-volume vertical diffusion;
- dry ideal-gas thermodynamics;
- exact moist vapor algebra;
- Murphy–Koop phase-explicit saturation vapor pressure with explicit validity bounds;
- saturation-moisture composition retaining phase/error provenance;
- pressure-coordinate hydrostatic identities;
- conservative extensive mass/tracer budget update;
- direct-DFT/spectral reference primitives.

Read `docs/FORTRAN-PHYSICS-FRONTIER.md` for the exact realized/open boundary. Important open responsibilities at handoff include condensed-water/latent/phase-change thermodynamics, continuity-consistent pressure velocity and other vertical-coordinate dynamics, full momentum dynamics, face-flux construction/reconstruction and CFL/positivity policy, radiation, surface/land/ocean exchange, chemistry, and coupled integration/restart/replay semantics.

The conservative transport kernel deliberately owns only extensive budget algebra. Do not turn it into a bespoke high-order reconstruction framework by inertia; candidate WENO/PPM/other reconstructions should be compared for convergence, monotonicity, conservation, cost, and maintained-library options where practical.

## Highest-leverage next work

Choose from the live frontier rather than following this list mechanically. Good candidates are:

1. **Fortran physical correctness:** add the next independently specifiable physical kernel rather than repairing the monolith. Continuity-consistent pressure velocity / vertical-coordinate dynamics and pressure-gradient/momentum structure are strong candidates because pressure hydrostatics and conservative budgets now provide adjacent contracts.
2. **Transport production semantics:** define geometry/velocity-to-face-mass-flux construction, positivity/monotonicity and CFL policy, then benchmark high-order reconstruction choices instead of presuming the legacy named method is optimal.
3. **Coupled integration:** once explicit/stiff/fast tendency partitions are physically declared, add a thin ARKODE adapter and differential/manufactured witnesses; do not hand-roll another adaptive IMEX stack.
4. **Spectral production path:** add FFTW behind a thin adapter and differential tests against the direct-DFT oracle; cuFFT remains GPU/hardware-gated.
5. **Legacy mathematical-name sweep:** continue identifying names stronger than realized contents. Preferred repair is constructive realization of useful mathematics, not demotion. Keep empirical climate relevance as a separate falsifiable obligation.
6. **Sheaf R3:** actual station cover/nerve, units-bearing climate stalks/restrictions, restriction functoriality, and a real climate-data coboundary are the next structural steps before any empirical sheaf claim can mature.

Do not broaden scope just because an idea is interesting. Every addition must earn its place through a clear responsibility, independent witness, ablation/baseline where relevant, and a composable interface.

## Execution-resource policy

Default to GitHub Actions for portable compilation, Rust/Fortran tests, symbolic references, CUE/architecture guards, and Lean kernel checks. This is cheaper, reproducible, and leaves an immutable execution trail.

Use Codespace sparingly for high-leverage work CI cannot cheaply supply: interactive theorem-prover debugging, materially different toolchain/environment parity, integration exploration requiring a persistent shell, or other targeted probes. Always start a Codespace handoff with `git switch main && git pull --ff-only`; an earlier Work session stayed on the old `architecture-scientific-contract` branch and falsely appeared not to contain the newer sheaf implementation.

Reserve the user's physical gaming laptop primarily for GPU/hardware truths: CUDA execution, device-residency/transfer behavior, numerical CPU↔GPU differential tests, deterministic reductions, resource contention, and realistic performance/VRAM observations. Do not install CUDA in a CPU-only environment merely to claim execution.

## Before making the next change

1. Fetch current `main` and verify it has not advanced past the handoff.
2. Read `docs/generated/STATUS.md`, `docs/FORTRAN-PHYSICS-FRONTIER.md`, `docs/ROADMAP.md`, and relevant module registry records.
3. Inspect exact-head Actions jobs; remember that the expected legacy-physics failure makes the aggregate architecture workflow red.
4. Commit directly to `main` in small coherent changes, preserve machine-authority/generated-status synchronization, and let CI falsify each exact head before moving on to dependent work.
5. Never infer validation or evidence eligibility from successful compilation/tests alone.
