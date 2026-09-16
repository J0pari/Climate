# Climate

Climate is an experimental research repository for building, checking, and comparing mathematical and numerical methods relevant to climate dynamics.

The repository is deliberately conservative about scientific authority. A mathematical identity, a compiling numerical kernel, a stable long integration, and agreement with observations are different accomplishments. They are tracked separately so that progress in one layer cannot silently promote another.

## Repository authority

The live repository is organized around three implementation lifecycles:

- **canonical** — portable implementations intended to define a maintained computational contract;
- **reference** — independent mathematical or symbolic authorities used to check definitions and analytic identities;
- **legacy** — prototypes retained only while scientifically useful behavior is being extracted behind explicit interfaces.

Objective counts and maturity are generated from machine-readable registries in [`docs/generated/STATUS.md`](docs/generated/STATUS.md). That generated status is the authority for repository inventory; this README intentionally does not duplicate counts.

The architectural and evidence rules are defined in:

- [`AGENTS.md`](AGENTS.md) — binding correctness and evidence requirements;
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — target system architecture;
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — current obligation graph and priorities;
- [`docs/EXECUTION-TOPOLOGY.md`](docs/EXECUTION-TOPOLOGY.md) — execution-resource boundaries;
- [`docs/PHYSICAL-INVARIANTS-REALIZATION.md`](docs/PHYSICAL-INVARIANTS-REALIZATION.md) — physical-law, invariant, balance-budget, and adversarial verification program.

## Physical and numerical core

The maintained Fortran surface is being built as small kernels with explicit state assumptions and analytic or budget witnesses rather than by treating the legacy physics monolith as an authority.

Current canonical pieces include:

- time-integration and linear-implicit references;
- LAPACK tridiagonal solves;
- constant-`f` Coriolis rotation;
- nonuniform vertical diffusion;
- dry ideal-gas thermodynamics;
- moist-vapor algebra;
- saturation vapor pressure and saturation composition;
- pressure-coordinate hydrostatics;
- conservative extensive mass/tracer transport;
- direct spectral/DFT reference calculations.

Each kernel has a deliberately narrow contract. A correct hydrostatic column does not imply a complete dynamical core; a conservative transport update does not imply a complete tracer model; a correct thermodynamic identity does not establish a validated atmospheric parameterization.

The remaining physical program includes momentum and pressure-gradient dynamics, vertical-coordinate boundary semantics, surface-pressure evolution, flux construction and reconstruction, radiation, moist condensed phases, coupling, chemistry, integrated timestep semantics, restart/replay, and end-to-end process budgets.

## Physical-law discipline

Climate treats physically important quantities as model-specific mathematical obligations.

Depending on the governing equations and assumptions, the relevant object may be:

- an exact continuum invariant;
- a local or global balance law;
- a material invariant;
- a Hamiltonian or Poisson-system Casimir;
- a wave-action or pseudomomentum quantity with explicit asymptotic assumptions;
- a semi-discrete invariant of a spatial operator;
- a fully discrete invariant or controlled drift statement;
- an entropy inequality;
- an empirical Earth-system budget.

These categories are not interchangeable.

The implementation strategy is:

```text
physical model and assumptions
        ↓
independent mathematical/reference derivation
        ↓
compatible spatial discretization
        ↓
time-integration and solver contract
        ↓
source/sink/exchange/correction ledger
        ↓
adversarial verification
        ↓
benchmark and empirical validation
```

The first serious higher-order-invariant laboratory is rotating shallow water, because it can exercise mass, wave propagation, rotation, potential vorticity, total energy, potential enstrophy, balance preservation, and controlled dissipation within one compact model. A barotropic-vorticity laboratory provides a smaller precursor for energy/enstrophy-preserving nonlinear operators.

## Numerical engineering principles

The physical core adopts lessons that have proved durable in both scientific modeling and production physics engines:

- prefer discretizations whose algebra encodes the desired cancellations rather than repairing global totals afterward;
- make timestep and substep semantics explicit;
- distinguish physical tendencies from filters, limiters, remapping, projections, positivity repair, and other numerical corrections;
- give every correction a measurable budget contribution;
- expose nonlinear and linear solver residuals rather than inferring correctness from iteration count;
- fail closed when convergence, positivity, ordering, conditioning, or coordinate assumptions are violated;
- make restart, replay, precision, and solver tolerance part of the implementation identity;
- test the same law under timestep refinement, grid refinement, coordinate permutations, orientation changes, unit conversions, boundary changes, and deliberately corrupted operators.

A plausible trajectory is not sufficient evidence of a physically faithful implementation.

## Mathematical reference layer

Independent references live under [`reference/`](reference/). Their purpose is to make exact definitions executable and provide differential checks for portable implementations.

Current reference families include:

- differential geometry;
- information geometry;
- finite-dimensional Lie brackets;
- finite-dimensional point-variational mechanics;
- finite-complex and finite cellular-sheaf cohomology.

A reference proves only the object it implements. Climate interpretation requires an explicit model-specific adapter or derivation plus the appropriate validation layer.

## Experimental mathematical methods

The repository also investigates mathematical representations whose scientific value is not assumed in advance, including:

- Riemannian state-space geometry;
- information-geometric inference;
- ultrametric and p-adic representations;
- sheaf/cohomological consistency methods;
- Clifford/geometric-algebra representations;
- modal-logic scenario semantics;
- learned or inferred latent representations.

These methods must compete against conventional baselines with the same information access. Experiments should state positive controls, negative controls, ablations, primary metrics, uncertainty treatment, and retain/revise/reject criteria before results are inspected.

## Data and empirical validation

Real-data claims require immutable data identity and reproducible preprocessing. The target data spine records:

- source and version/retrieval identity;
- fields, grids, vertical coordinates, calendars, and units;
- quality-control and missingness policy;
- transformation/preprocessing graph;
- artifact digests;
- licensing and citation information.

Synthetic and analytic fixtures are used to verify implementation behavior. They are not substitutes for observational or reanalysis validation.

## Verification

Repository CI separates architectural integrity from implementation-specific witnesses. Important verification classes include:

- analytic and manufactured solutions;
- conservation and balance closure;
- convergence studies;
- differential tests against independent references;
- metamorphic tests under coordinate/unit/orientation changes;
- negative controls and deliberately broken implementations;
- restart and deterministic replay;
- long-time drift and stiffness/timestep stress;
- observational/reanalysis comparison where data and provenance are sufficient.

The legacy Fortran physics monolith remains an isolated compile/debt sensor while coherent physical kernels are extracted. It is not the canonical physics authority.

## Resource boundaries

Some questions cannot be settled in ordinary CPU CI:

- CUDA correctness, races, mixed precision, and performance require a real accelerator;
- large observational/reanalysis validation requires immutable external datasets;
- integrated Commons scheduling/provenance behavior requires the corresponding workspace integration.

The repository prepares deterministic fixtures and contracts before consuming those resources.

## Contribution standard

A strong contribution removes uncertainty rather than merely adding code. Useful changes typically do one or more of the following:

- turn a mathematical name into a correct executable definition;
- extract a physical kernel with explicit units and assumptions;
- add an independent reference or manufactured witness;
- expose a hidden source, sink, correction, or coupling term;
- add a negative control that detects a plausible but wrong implementation;
- establish convergence, refinement, or restart behavior;
- add a fair baseline or ablation;
- improve immutable data provenance;
- retire a prototype after its useful responsibilities have been realized elsewhere.

Claims should remain no stronger than the evidence actually present in the repository.
