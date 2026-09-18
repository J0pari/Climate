# Physical laws, invariants, balance budgets, and verification

Status: **binding research and engineering plan for physical-law realization**.

This document defines how Climate turns mathematical structure into trustworthy climate-model behavior. The goal is a dynamical core whose governing equations, spatial operators, time integration, solvers, coupling, stabilization, and diagnostics agree about the quantities they transport, exchange, conserve, dissipate, or produce.

The standard is stronger than observing that a diagnostic stays nearly constant. Every claimed law must have a declared model, assumptions, discrete realization, runtime accounting, and adversarial witness.

---

## 1. Authority ladder

Physical-law work proceeds through distinct layers:

```text
continuum model + assumptions
        ↓
independent mathematical derivation
        ↓
reference operator / analytic fixture
        ↓
semi-discrete spatial law
        ↓
fully discrete time/solver law
        ↓
runtime source/sink/exchange/correction budget
        ↓
adversarial and convergence verification
        ↓
benchmark / empirical validation
```

Success at one layer does not imply success at the next.

The repository already contains exact finite-dimensional variational machinery in `reference/noether_sympy.py`. Its appropriate role is as one mathematical reference for deliberately defined finite-dimensional Lagrangian systems. Continuum fluids additionally require field-theoretic, Hamiltonian/Poisson, material-transport, and balance-law authorities matched to the actual governing equations.

---

## 2. Law classes

Every physically important quantity must be assigned a semantic class before implementation. The class determines its assumptions, diagnostics, and tests.

### 2.1 Exact continuum invariant

A functional `I[u]` satisfying

```text
dI/dt = 0
```

for a named continuum model, domain, boundary condition, regularity class, and declared absence of forcing/dissipation.

Examples can include total mass or total energy in closed idealized systems.

### 2.2 Local or global balance law

A relation of the form

```text
storage tendency + flux divergence = sources - sinks + exchanges.
```

This is the principal language for radiating, forced, dissipative Earth-system models.

### 2.3 Material invariant

A quantity satisfying a parcel-following relation such as

```text
Dq/Dt = 0
```

under explicitly stated mechanical and thermodynamic assumptions. Material invariance and global integral conservation are separate statements.

### 2.4 Hamiltonian / Poisson invariant

A quantity generated or preserved by a Hamiltonian structure. Noncanonical fluid systems may carry invariants through Poisson degeneracy and Casimir families.

### 2.5 Casimir / potential-vorticity functional

A functional associated with the degeneracy of a noncanonical Poisson bracket. In shallow-water-like systems, PV-dependent integral families and potential enstrophy belong here.

### 2.6 Entropy law

Depending on the model, the relevant statement may be material entropy conservation, a discrete entropy conservation identity, or an entropy inequality. Shock-capturing and irreversible moist physics require production/dissipation semantics rather than simple constancy.

### 2.7 Wave action / pseudomomentum

A perturbative or asymptotic quantity whose validity depends on background-state symmetry, linearization, scale separation, slowly varying coefficients, or another declared approximation.

### 2.8 Statistical cascade law

A scale-dependent flux, spectral slope, structure-function relation, or inertial-range law. These require resolution, forcing-range, dissipation-range, anisotropy, finite-domain, and uncertainty diagnostics.

### 2.9 Semi-discrete invariant

A functional preserved by the spatially discretized continuous-time system. This is where compatible/mimetic operators and energy/enstrophy-preserving constructions are judged.

### 2.10 Fully discrete invariant or controlled drift

A property of the actual time-stepping and nonlinear/linear solve. It must name timestep, substeps, solver tolerances, convergence criteria, precision, and splitting order.

### 2.11 Numerical correction budget

A measured contribution from filtering, limiting, positivity repair, remapping, projection, stabilization, artificial diffusion, clipping, iterative-solver tolerance, or restart/regrid operations.

### 2.12 Empirical Earth-system budget

A comparison against reanalysis or observations in which closure error also includes sampling, retrieval, forcing, unresolved-process, and dataset uncertainty.

---

## 3. Design principle: structure before correction

The preferred numerical method makes the desired physical cancellation an algebraic property of the discrete operators.

Examples:

- internal face fluxes cancel pairwise between adjacent cells;
- pressure work appears with equal-and-opposite conversion terms where appropriate;
- divergence and gradient operators satisfy the intended discrete adjoint relation;
- curl/divergence complexes respect compatible identities;
- mass flux used by tracer transport is the same mass flux that updates the carrier mass;
- coupled interface exchanges are equal-and-opposite between modeled reservoirs;
- skew-symmetric or Hamiltonian operators preserve the intended quadratic form;
- dissipative operators are sign-definite with respect to the quantity they are intended to remove.

Global projection or rescaling may be useful as a diagnostic experiment or tightly specified solver operation, but it is not a substitute for the correct discrete law.

---

## 4. Runtime accounting model

Every timestep should be explainable as a ledger.

For a quantity `Q`, define

```text
Q_after - Q_before
  = boundary_flux
  + resolved_physical_sources
  - resolved_physical_sinks
  + internal_exchange_net
  + coupling_exchange_net
  + numerical_correction
  + solver_residual_effect
  + roundoff_estimate
  + unexplained_residual.
```

For a closed ideal test, the legitimate right-hand-side terms collapse to zero except roundoff and controlled solver error.

For a forced system, closure means the measured state change agrees with the independently accumulated terms.

### Realized streaming ledger substrate

`src/budget_ledger.rs` realizes the common extensive-quantity accounting substrate used by this plan. It is intentionally a sidecar rather than a climate-state container: callers supply before/after extensive totals and process contributions as they stream through local partitions. Contributions are aggregated by typed process class and stable process identity, with compensated floating-point accumulation.

The ledger uses one sign convention—every recorded term is a signed contribution to `Q_after - Q_before`—and never invents a balancing process. Boundary fluxes require explicit boundary classification; physical source/sink sign errors fail closed; numerical correction and solver-residual effects have distinct channels; and the unexplained residual remains visible.

Partition merge requires identical quantity/domain/time/precision identity and rejects repeated partition IDs before double counting. This is the production-facing scaling contract for accounting: state fields need not be gathered globally merely to close a budget. `#BudgetReport` in `contracts/climate.cue` is the language-neutral artifact boundary so Fortran, Python, Rust, and external model wrappers can emit the same accounting semantics without sharing an implementation language. Budget closure through the common ledger is established only by explicit kernel integrations and their witnesses; the ledger's existence alone does not establish system-level closure.

`src/physical_budget_adapters.rs` is the narrow semantic bridge from physical-kernel diagnostics to that ledger. Conservative transport exposes lower/upper boundary contributions and separates nonnegative resolved sources from nonpositive resolved sinks for carrier and tracer mass; the adapter records those terms without reconstructing them from state deltas. Pressure-coordinate energy diagnostics are converted from specific power to extensive energy with explicit mass and timestep: kinetic pressure-gradient work, enthalpy pressure work, and the nonlocal part of geopotential material tendency are internal exchanges, while an explicit local geopotential tendency remains a separate coupling term. Any mismatch in the kernel closure survives as the ledger's unexplained residual rather than being relabeled. Dissipative-operator accounting remains quantity-specific and is not inferred from generic diffusion merely because the operator reduces a norm.

### Required ledger properties

Each term carries:

- quantity identity;
- units;
- sign convention;
- spatial support/domain;
- start/end time;
- process/operator identity;
- precision;
- accumulation method;
- boundary classification;
- whether it is physical, coupling, stabilization, solver, or diagnostic;
- optional uncertainty/error estimate.

A process that mutates state without an accounting channel is incomplete.

---

## 5. Lessons from mature numerical engines

Climate dynamics differ from rigid-body game physics, but mature engines and mature geophysical models converge on several engineering disciplines that are directly transferable.

### 5.1 Timestep semantics are part of the model implementation

Fixed-step engines make deterministic behavior and stability limits visible. Climate should retain the same clarity even when adaptive or multirate methods are used.

Required behavior:

- record every accepted timestep;
- record rejected adaptive steps when relevant to reproducibility;
- record internal substeps;
- separate outer coupling cadence from inner fast-wave/chemistry/microphysics cadence;
- include timestep policy in restart identity;
- test refinement trajectories, not only one nominal timestep.

### 5.2 Substepping is a physical-numerical contract

Fast processes should be resolved at a cadence consistent with their stability and accuracy requirements.

Candidate uses include:

- acoustic/gravity-wave subcycling;
- split-explicit atmospheric dynamics;
- chemistry subcycling;
- microphysics substeps;
- stiff source integration;
- ocean-atmosphere coupling accumulation.

Outer-step budgets must reconcile with the sum of inner-step exchanges.

### 5.3 Solver residuals are first-class diagnostics

Iteration count is not a measure of physical correctness.

Each iterative solve should expose:

- equation residual norm;
- scaled/physical residual where useful;
- convergence reason;
- iteration count;
- tolerance;
- conditioning or preconditioner diagnostics where available;
- resulting contribution to any monitored physical budget.

Reaching an iteration cap is a failed or explicitly degraded solve, not a successful state update with less confidence.

### 5.4 Warm starts are an optimization, not hidden state semantics

When iterative systems use previous solutions, cached multipliers, or previous pressure corrections:

- cold and warm starts must converge to compatible solutions within declared tolerances;
- restart files must include any state needed for bitwise or tolerance-level replay;
- warm-start dependence must not alter the physical fixed point.

### 5.5 Stabilization has its own identity

Numerical stabilization is recorded independently from the physical tendency.

Examples:

- filters;
- hyperdiffusion;
- monotonicity limiters;
- positivity repair;
- divergence damping;
- pressure projection;
- remapping;
- flux correction;
- implicit regularization.

The budget must reveal what each stabilization step changed.

### 5.6 Refusal is preferable to plausible corruption

Fail closed on:

- invalid pressure/vertical-coordinate ordering;
- non-finite state or tendency values;
- unsupported boundary combinations;
- negative masses or depths outside a declared dry-state scheme;
- invalid thermodynamic composition;
- failed nonlinear/linear convergence;
- violated CFL/stability prerequisites when the integrator requires them;
- incompatible grid/operator metadata;
- unit mismatches.

---

## 6. Model ladder

A physical invariant program should advance through compact models whose laws are independently checkable before being embedded in a complete climate core.

### 6.1 Barotropic vorticity / two-dimensional incompressible flow

Purpose: smallest nonlinear fluid laboratory for simultaneous structure and cascade behavior.

Continuum targets:

- vorticity transport;
- kinetic energy;
- enstrophy;
- declared viscous/enstrophy dissipation when enabled.

Implementation targets:

- periodic Cartesian grid first;
- independently derived Arakawa-style Jacobian reference;
- a conventional competitor using the same grid and timestep;
- explicit discrete energy and enstrophy functionals;
- spatial convergence against manufactured or analytic cases;
- long-time nonlinear roll-up/cascade behavior;
- spectral flux diagnostics where resolution permits.

Acceptance requires the structure-preserving operator to demonstrate its intended invariants algebraically/numerically and the competitor to provide a useful negative-control contrast.

### 6.2 Rotating shallow water — first full invariant laboratory

Purpose: compact system combining waves, rotation, balance, mass transport, vorticity, PV, energy, and higher-order invariant structure.

Initial domain:

- periodic beta-plane or f-plane Cartesian geometry;
- positive layer depth;
- closed inviscid baseline;
- explicit metric/grid contract.

Continuum targets:

- total mass;
- total energy;
- potential vorticity `q = (zeta + f)/h`;
- material PV transport in the ideal limit;
- potential enstrophy `1/2 ∫ h q^2 dA`;
- balanced steady states;
- inertia-gravity and Rossby-wave dispersion in the appropriate limits.

Forced/dissipative variants:

- drag;
- viscosity/hyperviscosity;
- mass sources;
- wind forcing;
- boundary fluxes in later nonperiodic tests;
- energy-preserving / potential-enstrophy-dissipating options when intentionally selected.

Required discretization work:

- cell/edge/vertex quantity placement chosen explicitly;
- compatible divergence, gradient, curl, and averaging operators;
- shared mass flux between continuity and advected quantities;
- kinetic/potential energy functional matched to placement;
- PV definition matched to discrete circulation and mass;
- energy/enstrophy exchange algebra derived before optimization;
- time integrator chosen with an explicit fully-discrete drift policy.

Required witnesses:

- constant state;
- solid-body/balanced state where geometry permits;
- linear gravity wave;
- geostrophic adjustment;
- Rossby-wave propagation;
- vortex advection;
- long-time turbulent evolution;
- grid refinement;
- timestep/substep refinement;
- deliberately altered sign/orientation/averaging operators.

### 6.3 Dry compressible Euler

Target quantities:

- mass;
- momentum subject to domain symmetry and boundary conditions;
- kinetic + internal + gravitational total energy;
- entropy conservation for smooth adiabatic inviscid flow;
- entropy-stable/entropy-producing behavior for discontinuous or dissipative variants;
- circulation/vorticity/PV relations with assumptions declared.

Design decisions must explicitly balance:

- energy conservation;
- entropy stability;
- positivity;
- monotonicity;
- shock robustness;
- high-order accuracy.

No single property is promoted without measuring its interaction with the others.

### 6.4 Hydrostatic primitive-equation atmosphere

Target quantities and balances:

- dry-air mass;
- pressure-coordinate continuity;
- total energy for the declared hydrostatic thermodynamic system;
- axial angular momentum with resolved pressure, mountain, friction, and surface torques;
- Ertel-like PV in the declared ideal limit;
- tracer/constituent mass;
- hydrostatic and geostrophic balance preservation.

Existing canonical pressure hydrostatics, Coriolis rotation, thermodynamic algebra, vertical diffusion, and conservative extensive transport are prerequisites, not substitutes for the coupled dynamical law.

### 6.5 Moist atmosphere

Target budgets:

- dry-air mass;
- total water across vapor, liquid, ice, and precipitating categories;
- moist total energy under one documented thermodynamic reference convention;
- phase-change conversion terms;
- precipitation boundary export;
- entropy production for irreversible processes;
- constituent positivity with compensating budget accounting.

Every saturation adjustment or microphysical process emits mass/water/energy receipts.

### 6.6 Radiation and surface/ocean/land/ice exchange

These layers are open-system physics.

Required exchange channels include:

- top-of-atmosphere shortwave and longwave fluxes;
- surface radiative fluxes;
- sensible/latent turbulent fluxes;
- freshwater exchange;
- momentum stress;
- ocean/land/ice storage;
- phase/mass transfer;
- constituent exchange.

Where both sides of an interface are modeled, internal exchange is equal-and-opposite up to explicitly measured numerical/coupling residual.

### 6.7 Wave activity and pseudomomentum

Add these only after the background-state and perturbation system is explicit.

Required metadata:

- basic state;
- perturbation definition;
- linearization order;
- averaging operator;
- scale-separation assumption;
- source/dissipation terms;
- asymptotic error interpretation.

### 6.8 Turbulent cascades

Required diagnostics:

- energy/enstrophy or other relevant scale fluxes;
- spectra and compensated spectra;
- structure functions where useful;
- forcing-band and dissipation-band separation;
- grid-resolution sweep;
- anisotropy/rotation/stratification diagnostics;
- finite-domain effects;
- uncertainty across realizations.

---

## 7. Discrete operator obligations

### 7.1 Topology and geometry are separate contracts

The mesh representation should distinguish:

- incidence/topology;
- metric/geometric factors;
- field placement;
- orientation;
- boundary ownership.

This enables orientation and permutation tests without changing the physical solution.

### 7.2 Compatible operator identities

Where mathematically appropriate, executable tests should cover identities such as:

```text
curl(grad(phi)) = 0
div(curl(A)) = 0
```

or their discrete analogue on the chosen grid.

Adjoint/skew relationships required by energy conservation should be checked directly as matrix/operator identities on tiny fixtures.

### 7.3 Flux consistency

For finite-volume transport:

- each internal face has one authoritative mass flux;
- neighboring cells consume that same flux with opposite orientation;
- tracer extensive flux derives from the same carrier mass flux plus the declared reconstruction;
- remapping and coupling cannot substitute a separately recomputed inconsistent mass flux.

### 7.4 Boundary operators

Boundary conditions are operators with budget semantics.

Each boundary declares:

- impermeable / periodic / prescribed-flux / radiative / open behavior;
- transported quantities;
- sign convention;
- energy/work terms;
- torque terms where relevant;
- tracer/constituent terms.

---

## 8. Time integration obligations

For every integrator used in physical-law tests, record:

- order;
- explicit/implicit/IMEX structure;
- stability assumptions;
- conserved or dissipated quantities known analytically;
- nonlinear solve policy;
- linear solver/preconditioner;
- adaptive controller if any;
- split ordering;
- subcycling policy;
- dense-output/interpolation semantics if coupling uses them.

Tests must distinguish:

- spatial truncation error;
- temporal truncation error;
- algebraic solver error;
- splitting error;
- coupling error;
- roundoff.

---

## 9. Correction ledger

Every non-physical state modification gets a named channel.

Minimum channels:

```text
filter
limiter
positivity_repair
remap
regrid
projection
artificial_diffusion
implicit_regularization
solver_inexactness
restart_conversion
precision_conversion
```

For each channel record:

- state before/after or sufficient delta diagnostics;
- quantity deltas for monitored budgets;
- triggering criterion;
- affected cells/levels;
- magnitude norms;
- whether correction was expected, exceptional, or fatal.

Repeated large correction is a diagnostic failure even when the final state remains bounded.

---

## 10. Adversarial verification matrix

Verification should try to make incorrect implementations look plausible and then ensure the tests still reject them.

### 10.1 Algebraic mutations

Inject test-only variants with:

- one flux sign reversed;
- one face orientation reversed;
- one metric factor omitted;
- one averaging weight changed;
- inconsistent mass/tracer fluxes;
- swapped Coriolis sign;
- missing pressure-work conversion;
- incomplete equal-and-opposite coupling exchange.

Expected result: the appropriate local/global law witness fails with diagnostic localization.

### 10.2 Unit mutations

Test controlled mistakes involving:

- Pa vs hPa;
- K vs degC offsets where applicable;
- mixing ratio vs specific humidity;
- geopotential vs geometric height;
- per-area vs extensive quantities;
- seconds vs days;
- radians vs degrees.

Unit metadata and dimensional tests should reject these before a long integration.

### 10.3 Orientation/permutation metamorphics

Transform:

- cell numbering;
- edge numbering;
- mesh orientation;
- coordinate-axis ordering;
- periodic-domain origin.

After mapping outputs back, physical results and invariant residuals should agree within the declared tolerance.

### 10.4 Timestep stress

Sweep:

- stable small steps;
- nominal steps;
- near-limit steps;
- deliberately invalid steps;
- substep counts;
- adaptive tolerances.

Expected behavior must distinguish controlled convergence, known dissipation, and explicit refusal.

### 10.5 Solver stress

Sweep:

- nonlinear tolerance;
- linear tolerance;
- preconditioner choice;
- iteration cap;
- cold/warm starts;
- ill-conditioned but valid states.

Budget residuals should respond consistently with solver accuracy.

### 10.6 Boundary stress

Compare:

- periodic closure;
- impermeable boundaries;
- prescribed flux;
- open/radiative configurations when implemented.

Global budget changes must equal integrated boundary terms.

### 10.7 Precision stress

Compare FP64 with any reduced-precision path using:

- one-step differential checks;
- long-time drift;
- invariant residual distribution;
- cancellation-sensitive states;
- deterministic reduction behavior where required.

### 10.8 Restart/replay stress

Checkpoint at adversarial times:

- immediately before/after coupling;
- mid-subcycling where supported;
- after a limiter/correction event;
- near solver convergence difficulty.

Restarted evolution must match the declared bitwise or tolerance-level reproducibility contract.

### 10.9 Physical corner states

Include:

- thin layers / low mass;
- strong shear;
- near-saturation thermodynamics;
- near-zero tracer concentrations;
- strong rotation;
- weak/strong stratification;
- steep pressure/height gradients within the model's valid regime.

The expected response is either correct bounded evolution or explicit refusal according to the declared domain.

### 10.10 Long-time behavior

Short-step correctness is insufficient for climate-scale integrations.

Measure:

- secular invariant drift;
- phase error;
- wave amplitude error;
- balance degradation;
- spectral pile-up;
- grid imprinting;
- cumulative correction budget;
- ensemble sensitivity to roundoff/ordering where relevant.

---

## 11. Acceptance gates

A physical-law feature becomes canonical only after the gates appropriate to its class are satisfied.

### Gate A — definition

- governing equations written;
- units declared;
- variables and placement declared;
- domain and boundary conditions declared;
- law class declared;
- assumptions listed.

### Gate B — independent authority

At least one of:

- symbolic derivation;
- analytic derivation encoded as executable fixture;
- independent reference implementation;
- manufactured solution;
- formal proof for a stable finite identity.

### Gate C — discrete structure

- operator identities tested;
- local/internal cancellation demonstrated;
- boundary semantics executable;
- correction channels explicit.

### Gate D — convergence

- spatial refinement;
- temporal refinement;
- solver-tolerance refinement where relevant;
- expected order or asymptotic trend documented.

### Gate E — adversarial rejection

Planted sign, orientation, unit, flux-consistency, or solver errors must be detected by the intended witnesses.

### Gate F — long-time behavior

The relevant invariant/balance, phase, wave, and spectral properties remain within declared envelopes over a duration long enough to expose secular error.

### Gate G — benchmark/empirical interpretation

Only after mathematical/numerical verification should the quantity be interpreted against established benchmark solutions, reanalysis, or observations.

---

## 12. Machine-readable law contracts

Introduce a registry only when the first canonical fluid laboratory requires it. A useful record shape is:

```text
law_id
model_id
law_class
quantity
units
continuum_statement
assumptions
boundary_requirements
spatial_discretization
integrator_contract
physical_source_terms
physical_sink_terms
exchange_terms
numerical_correction_channels
reference_witnesses
adversarial_witnesses
acceptance_tolerances
validation_status
```

The registry should generate status rather than duplicate prose.

---

## 13. Recommended implementation packages

### Package P1 — barotropic invariant laboratory

Deliver:

- compact canonical state/grid contract;
- vorticity-streamfunction inversion using maintained linear algebra;
- structure-preserving nonlinear Jacobian;
- explicit energy/enstrophy diagnostics;
- conventional comparison operator;
- manufactured/analytic tests;
- long-time nonlinear test;
- adversarial mutation tests.

### Package P2 — rotating shallow-water core

Deliver:

- mass and momentum state placement;
- compatible divergence/gradient/curl operators;
- Coriolis/PV flux operator;
- mass-consistent transport;
- discrete energy/PV/potential-enstrophy diagnostics;
- ideal periodic tests;
- forcing/dissipation ledger;
- balanced-state and wave benchmarks;
- timestep/solver/restart adversarial suite.

### Package P3 — physical budget infrastructure

Deliver:

- typed quantity/budget records;
- process identity;
- local/global accumulation;
- correction ledger;
- closure residual computation;
- deterministic serialization;
- restart preservation;
- test helpers for expected cancellation.

### Package P4 — pressure-coordinate momentum dynamics

After shallow-water operator discipline is established, apply it to:

- horizontal pressure-gradient force;
- momentum flux/advection;
- pressure-coordinate continuity coupling;
- axial angular momentum diagnostics;
- kinetic/potential/internal conversion terms;
- hydrostatic/geostrophic balance tests.

### Package P5 — moist total-water and total-energy budget

Deliver:

- condensed-phase thermodynamics;
- one thermodynamic reference convention;
- reversible phase-change witnesses;
- irreversible process receipts;
- precipitation/export accounting;
- positivity policy with conservation receipts;
- coupled column tests.

### Package P6 — coupled interfaces

Deliver equal-and-opposite exchange contracts for atmosphere/ocean/land/ice surfaces before adding broad parameterization complexity.

---

## 14. Relationship to current canonical kernels

Current kernels should become inputs to the physical-law program through explicit composition tests.

### Coriolis rotation

Use as a local skew operator witness. Composition tests should show the expected kinetic-energy behavior for the declared discretization and timestep policy.

### Pressure-coordinate hydrostatics

Use in pressure-gradient and total-energy derivations. Hydrostatic correctness alone does not define horizontal momentum work.

### Pressure-coordinate continuity

Once landed, use its `omega` closure residual as one vertical mass-balance component. Surface-pressure and moving-boundary semantics remain separate obligations.

### Conservative extensive transport

Use as the carrier/tracer budget primitive. Mass-flux construction and reconstruction remain separate dynamical responsibilities.

### Dry/moist thermodynamic algebra

Use to construct energy and water budgets only after a single state-variable/reference-energy convention is fixed.

### Vertical diffusion

Give diffusion a sign-definite dissipation/flux budget and boundary-flux accounting rather than treating it as a generic smoother.

### Time integration

Use reference integrators to separate spatial-law errors from time-discretization errors before selecting production integration strategies.

---

## 15. Practical success criteria

The physical-law program is succeeding when:

- every major prognostic update can explain its contribution to mass, energy, momentum, water, tracer, and other applicable budgets;
- local internal exchanges cancel globally for the mathematically declared reason;
- legitimate forcing and dissipation produce the measured budget change;
- numerical correction is visible and quantitatively small or intentionally controlled;
- planted errors are rejected quickly;
- refinement reduces the appropriate residuals at the expected rate;
- long integrations do not hide secular drift behind bounded-looking fields;
- restarts and solver choices obey explicit reproducibility contracts;
- higher-order quantities such as PV and potential enstrophy are tied to a concrete fluid model and compatible discretization;
- empirical validation is layered on top of verified mathematics rather than used to mask implementation ambiguity.

The target is not maximal formalism. It is a physical core whose behavior remains intelligible under finite precision, finite resolution, stiff coupling, long integration, and hostile testing.
