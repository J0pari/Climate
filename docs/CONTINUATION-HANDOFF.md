# Continuation handoff

This is the compact operational handoff for work on `J0pari/Climate`. It is not a second architecture or status authority. Before acting, fetch the current `main` head and reread the live authorities below. Never reset or overwrite newer work.

## Live authorities

- `AGENTS.md` — binding contributor and scientific-evidence rules.
- `docs/ARCHITECTURE.md` — target system shape and semantic boundaries.
- `docs/ROADMAP.md` — live scientific and engineering obligation graph.
- `docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md` — candidate unifying representation/geometry research program.
- `docs/FORTRAN-PHYSICS-FRONTIER.md` — realized and open atmospheric physical/numerical responsibilities.
- `docs/FORTRAN-TIME-INTEGRATION.md` — integration semantics and generic-library boundary.
- `docs/generated/STATUS.md` — generated module/claim/experiment/sheaf status; do not hand-edit.
- `architecture/modules/*.json`, `claims/registry.json`, `experiments/*.json`, `methods/sheaf-realization.v1.json` — machine authorities behind generated status.

## Working style

Commit directly to `main` in small coherent commits. Re-read the current `main` version of every file before writing it. Never force-update or rewrite history.

Edits must be granular and evidence-preserving. Do not broadly rewrite architecture to fit a simplified new plan; preserve valid learned constraints and change only surfaces that are stale, contradictory, or scientifically weaker than the intended design.

README and architecture documents are present/future-facing specifications, not changelogs. Repository history is the archive for superseded implementations and discarded ideas. Use this handoff only for compact operational state that materially affects continuation.

Do not build knowingly disposable surrogate abstractions when the stronger final-form contract is already knowable. Do not simplify scientifically meaningful semantics merely to make code easier.

## Scientific center of gravity

The repository has two coupled scientific programs.

### Structure-preserving climate dynamics

Physical kernels should expose the mathematical structure of the equations they implement rather than relying on post-hoc conservation checks. Important coupled contracts include:

- continuity-consistent mass and tracer fluxes;
- pressure-gradient work paired with thermodynamic/geopotential conversion;
- skew/no-work Coriolis structure;
- energy exchange among kinetic, potential, internal, latent, radiative, and component reservoirs;
- physically signed diffusion, drag, precipitation, radiation, and surface exchange;
- compatible spatial operators whose balance and wave properties are appropriate to rotating geophysical flow.

Noether/Hamiltonian/variational ideas apply selectively where conservative subdynamics actually possess those structures. Forced and dissipative climate dynamics require explicit balance-law and exchange semantics rather than decorative conservation language.

### Multirepresentation climate geometry

The candidate unifying construct is an atlas or structured family of scientifically grounded representations of the same climate state, not one giant feature vector.

Representation maps may contribute as coordinates, factors, fibers, observation maps, constraints, kernels, quotient structures, or comparison geometries. Candidate relationships include common-manifold, product, fibered, quotient, stratified, and local-atlas structure.

Information geometry has a central role where an explicit likelihood supplies Fisher structure, but parameter geometry, state geometry, and distribution/uncertainty geometry remain distinct. Rank-deficient Fisher or observation pullbacks expose unidentifiable directions and must not be regularized into fictitious information by default.

Sheaf methods may supply local-to-global compatibility structure; ultrametric/p-adic methods may supply alternative relational geometry; Clifford structure may represent oriented or multicomponent mode interactions; spectral/Koopman coordinates may provide dynamically coherent views. None is assumed to be a universal coordinate system.

Curvature, geodesics, holonomy, topology, and intrinsic dimension are diagnostics of a declared representation, not universal optimization objectives.

## Library-versus-novelty rule

Do not spend Climate's novelty budget reimplementing mature generic numerical infrastructure.

Use hand-written code when it is a small transparent mathematical/reference oracle or when Climate owns genuinely domain-specific semantics. Production generic numerics should normally delegate to maintained libraries through thin, testable adapters.

Current boundaries include:

- LAPACK `DGTTRF/DGTTRS` for production-oriented CPU tridiagonal solves; transparent Thomas remains a reference oracle.
- SUNDIALS ARKODE as the default production trajectory for adaptive/implicit/IMEX/multirate integration unless a concrete requirement proves a better fit.
- FFTW for production CPU FFT and cuFFT for GPU; direct DFT remains the small deterministic oracle.
- RTE+RRTMGP or another maintained package should be evaluated before implementing generic gas-optics/two-stream radiative-transfer machinery locally.
- PETSc becomes relevant only when distributed state vectors, MPI decomposition, large sparse operators, and serious preconditioning justify it.

Climate-owned effort belongs in physical variables and units, coordinate semantics, discrete operators, balance/exchange laws, Jacobians and atmospheric preconditioning structure, representation maps, physical/statistical constraints, and scientifically meaningful experiments.

## Canonical physical/numerical foundation

The canonical Fortran surface includes:

- fixed-step RK4 reference integration;
- transparent no-pivot tridiagonal/theta reference algebra;
- LAPACK partial-pivoting tridiagonal backend;
- exact constant-`f` Coriolis rotation;
- nonuniform geometric-height finite-volume vertical diffusion;
- dry ideal-gas thermodynamics;
- exact moist-vapor mixture algebra;
- Murphy–Koop phase-explicit saturation vapor pressure;
- saturation-moisture composition;
- pressure-coordinate hydrostatic identities;
- fixed-pressure-coordinate continuity with explicit interface `omega` integration and residual diagnostics;
- conservative extensive mass/tracer budget updates;
- direct-DFT/analytic-signal reference primitives.

Open physical responsibilities include condensed-water/latent thermodynamics, surface-pressure and moving-boundary vertical-coordinate dynamics, full momentum and pressure-gradient structure, continuity-consistent face-flux construction, radiation, surface/land/ocean exchange, chemistry, and coupled integration/restart/replay semantics.

The next physical work should favor a coupled structure that closes mass, momentum, and energy exchanges over another isolated tendency kernel.

## Mathematical and representation foundation

Canonical/reference surfaces include:

- Levi-Civita geometry from explicit metric data;
- Gaussian-mean Fisher information with preserved rank deficiency;
- symbolic information-geometry, Lie-bracket, and variational/Noether references;
- exact finite p-adic/ultrametric primitives;
- sparse Clifford algebra;
- modal-logic semantics;
- finite sheaf numerics plus exact cohomology reference;
- narrow Lean authorities for stable algebraic identities.

Mathematical correctness does not by itself establish climate usefulness. Strong mathematical names must satisfy their defining laws, while empirical interpretations remain separate obligations.

## Repository status

`docs/generated/STATUS.md` is authoritative for counts. At the current documentation handoff it reports:

- 50 registered modules: 21 canonical, 24 noncanonical/compatibility-lifecycle, 5 reference;
- 24 runnable and 26 prototype modules;
- 5 registered scientific claims, all still at `concept` with zero supporting evidence records;
- 2 registered experiment specifications;
- sheaf realization at 8/15 obligations, with climate-data and empirical layers still open.

The intentionally isolated noncanonical Fortran monolith compile probe may keep the aggregate architecture workflow red even when canonical and reference jobs are healthy. Inspect exact-head jobs rather than equating the workflow-level conclusion with canonical failure.

## Highest-leverage next work

Choose from the live roadmap rather than treating this list as a serial plan.

1. Define the first multirepresentation reference experiment on a dynamical system with known shared/product/fibered structure and multiple nonlinear observation views. Compare raw concatenation, linear multiview methods, and established common-manifold/factorized methods before inventing a custom learner.
2. Connect the Fisher-information and information-geometry references to an observation-induced metric while keeping parameter, state, and uncertainty geometry distinct.
3. Advance the physical core through a coupled momentum/pressure-work/continuity/energy contract rather than another isolated tendency routine.
4. Define continuity-consistent geometry/velocity-to-face-mass-flux semantics and then compare reconstruction/positivity strategies rather than assuming a bespoke scheme.
5. Progress the climate-data sheaf through actual station-cover, stalk, restriction, and coboundary semantics.
6. Add production FFT, adaptive integration, radiative transfer, or accelerator paths only behind maintained libraries and independent reference/differential witnesses.

## Execution resources

Use GitHub Actions for portable builds, Rust/Fortran tests, symbolic references, architecture/status checks, and Lean kernel checks.

Use persistent interactive environments only when they materially improve toolchain or integration debugging. Reserve real CUDA hardware for CUDA truths: device execution, memory/transfer behavior, race/synchronization behavior, precision effects, deterministic reductions, and realistic performance/VRAM measurements.

## Before every change

1. Fetch current `main`.
2. Re-read the relevant live authority and exact file being changed.
3. Check whether machine-readable authorities or generated status must change with it.
4. Commit one coherent change directly to `main`.
5. Inspect exact-head CI jobs before depending on that change.
6. Never infer empirical validation from compilation, tests, or mathematical verification alone.
