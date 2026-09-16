# Physical invariants, balance laws, and symmetry realization

Status: **binding research and verification plan for conservation/symmetry work**.

This document defines the work required before Climate may describe Noether structure, higher-order invariants, or conservation laws as being meaningfully connected to deployed climate physics. It is intentionally stricter than a list of mathematical identities. The target is not to decorate an evolving model with conserved quantities after the fact; it is to make the governing equations, discretization, coupling, numerical corrections, and verification machinery agree about what is conserved, what is only balanced, what is deliberately dissipated, and under which assumptions each statement is valid.

The current exact Noether reference remains useful, but it is only the first rung of this ladder.

---

## 1. Current assessment

### 1.1 What is already technically sound

`reference/noether_sympy.py` is a legitimate exact reference for a narrow mathematical problem: finite-dimensional, first-order Lagrangians `L(t, q, qdot)` with point symmetries. It computes the variational invariance residual, Noether charge, Euler-Lagrange residuals, characteristic, and the off-shell Noether identity. Its current tests correctly cover examples such as time translation of a harmonic oscillator, spatial translation of a free particle, rotational symmetry, a Galilean boost requiring a boundary term, an explicit non-symmetry, and rejection of a velocity-dependent generator that does not belong to the declared point-symmetry class.

That is a useful theorem oracle. It is **not** yet a climate-physics realization.

### 1.2 What is not yet realized

The repository does not currently have the layers needed to connect that oracle to the physically important invariants of a climate dynamical core:

- no continuum variational-field implementation for atmosphere/ocean PDEs;
- no Lagrangian particle-label/relabeling symmetry authority for fluid circulation or potential-vorticity results;
- no noncanonical Hamiltonian/Poisson-bracket reference for fluid Casimirs;
- no canonical rotating shallow-water model serving as an invariant laboratory;
- no canonical full momentum equation or pressure-gradient operator;
- no coupled total-energy equation including internal, kinetic, potential, moisture/phase, radiation, surface, and subgrid transfers;
- no axial angular-momentum budget with pressure/mountain/friction/surface torques;
- no canonical vorticity or Ertel potential-vorticity equation;
- no higher-order invariant implementation such as potential enstrophy with its assumptions made executable;
- no runtime process ledger that attributes invariant changes to resolved fluxes, physical sources/sinks, coupling, filters, limiters, remapping, solver tolerances, or roundoff;
- no fully discrete proof or differential witness showing that a chosen space/time discretization preserves the intended semi-discrete invariant;
- no empirical budget closure study against reanalysis/observations that is separate from mathematical correctness.

The legacy `climate_symmetries.hs` contains useful hints about desired budgets and data sources, but it also mixes physically serious ideas with heuristic symmetry detection, variance thresholds, confidence-like values, and broad Noether language. It must remain a source reservoir rather than an authority.

### 1.3 The repository currently overstates the connection in one important place

The README language saying that approximate climate symmetries, via Noether's theorem, correspond to conserved quantities is too broad. Approximate constancy, a low variance diagnostic, a scaling law, or a broken empirical pattern is not enough to invoke Noether. For forced, dissipative, moist, radiating Earth-system dynamics the scientifically useful object is usually a **balance law with explicit production, destruction, flux, and exchange terms**, not an exact conserved Noether charge.

The correct architectural direction is therefore:

> exact theorem oracle -> model-specific continuum law -> discrete law -> runtime budget -> adversarial verification -> empirical/process validation.

Passing an earlier step never promotes a later one automatically.

---

## 2. Vocabulary that must not be blurred

Every invariant-related object must be assigned one of the following semantic classes. These are not interchangeable labels.

### 2.1 Exact continuum invariant

A quantity `I[u]` satisfying `dI/dt = 0` for a precisely declared continuum model, domain, boundary condition, regularity class, and set of absent forcings/dissipations.

Examples may include mass or energy of an inviscid closed system, but only after the relevant assumptions are explicit.

### 2.2 Continuum balance law

A local or global relation of the form

```text
storage tendency + flux divergence = physical sources - physical sinks + exchange terms.
```

This is the normal object for radiating, forced, dissipative climate dynamics. A good balance law is not a failed conservation law; it is often the more physically correct statement.

### 2.3 Casimir / degeneracy invariant

A conserved functional associated with the degeneracy of a noncanonical Poisson structure rather than an ordinary finite-dimensional point symmetry. Potential-enstrophy families in ideal fluid models belong in this discussion. They must not be mislabeled as ordinary point-Noether charges.

### 2.4 Material invariant

A quantity conserved following fluid parcels under a specific idealized set of thermodynamic and mechanical assumptions, such as a form of potential vorticity in adiabatic, frictionless flow. Material conservation is not automatically a global integral conservation statement.

### 2.5 Adiabatic invariant / wave action / pseudomomentum

A quantity conserved only under scale separation, slowly varying backgrounds, linearization, symmetry of the basic state, or another perturbative assumption. These require their own asymptotic error statement.

### 2.6 Statistical scaling relation

An inertial-range exponent, spectral slope, flux plateau, structure-function law, or other statistical relation. Scale invariance in turbulence must live here unless a precise symmetry theorem is actually established. A power law is not by itself a Noether conservation law.

### 2.7 Semi-discrete invariant

A discrete functional preserved exactly by the spatially discretized ODE/PDE system before time discretization. This is where Arakawa-type energy/enstrophy constructions and mimetic/compatible operators belong.

### 2.8 Fully discrete invariant or controlled drift

A property of the actual time-stepping algorithm. A spatial scheme may conserve an invariant in continuous time while the chosen integrator introduces drift. The fully discrete statement must therefore name the time integrator, timestep, solver tolerance, nonlinear convergence policy, and precision.

### 2.9 Numerical correction budget

Any change introduced by clipping, positivity repair, filters, Robert-Asselin-like damping, artificial diffusion, remapping, flux correction, pressure projection, constraint stabilization, iterative-solver tolerance, or restart/regridding. Such corrections must be measured rather than disappearing into the state update.

### 2.10 Empirical Earth-system budget

A comparison of model or diagnostic terms with observations/reanalysis. Closure error here contains sampling, retrieval, forcing, unresolved process, and dataset uncertainty in addition to numerical error. It is not the same verification problem as exact mathematical conservation.

---

## 3. Where Noether is genuinely relevant—and where it is not the whole story

### 3.1 Finite-dimensional variational subsystems

The current SymPy reference is directly applicable to deliberately reduced models whose action and generalized coordinates are explicitly defined. Examples include idealized oscillators, mechanically reduced modes, or carefully derived finite-dimensional truncations.

Required before climate interpretation:

1. derive the reduced Lagrangian from the declared physical model rather than fit it because it produces a desired invariant;
2. identify the exact transformation group and boundary/gauge term;
3. establish the reduction error relative to the parent equations;
4. distinguish a true symmetry from an approximately constant trajectory statistic;
5. test the charge against direct numerical trajectories from an independent integrator.

### 3.2 Continuum fluids require field theory and particle-relabeling structure

Atmospheric and oceanic fluids are not collections of a few generalized coordinates. Important fluid invariants arise from spatial fields, advected quantities, boundary geometry, and relabeling freedom of fluid parcels.

For ideal fluids, circulation, vorticity, helicity, and potential-vorticity results can be connected to particle-relabeling symmetries in a Lagrangian description. This is a deeper and different use of Noether's theorem than the current point-symmetry oracle. A correct implementation therefore needs a field/continuum reference layer or an independently derived Hamiltonian/Poisson authority.

Do **not** generalize the existing `PointSymmetry` API until the required mathematical object is clear. A larger generic API that silently treats field symmetries, generalized symmetries, gauge freedoms, and relabelings as the same thing would make the mathematics less trustworthy, not more.

### 3.3 Casimirs and higher-order invariants need their own authority

In rotating shallow-water and related noncanonical Hamiltonian systems, potential-vorticity-dependent integrals form Casimir families under ideal assumptions. Potential enstrophy is especially important numerically because uncontrolled grid-scale enstrophy can contaminate long integrations and energy cascades.

These quantities should be represented as `Casimir`/`PVInvariant`-type obligations, not squeezed into a point-Noether charge type merely to create conceptual uniformity.

### 3.4 Forced and dissipative climate physics should normally use balance laws

Radiation, surface drag, turbulent mixing, convection, microphysics, precipitation, chemistry, gravity-wave drag, ocean exchange, and data-assimilation increments all break one or more ideal symmetries or transfer quantities between modeled reservoirs.

The useful diagnostic is then something like

```text
energy_after - energy_before
  - resolved_boundary_flux
  - radiation
  - surface_exchange
  - phase-change/internal conversion
  - subgrid_dissipation
  - numerical_correction
  = unexplained_residual.
```

A small unexplained residual is meaningful only when the named terms are independently defined and the signs/units/normalizations are fixed.

---

## 4. The model-specific invariant ladder

The repository should not attempt a universal conservation framework before it has one physically complete model-specific ladder. The following ordering gives increasing realism while keeping the laws independently testable.

### 4.1 Barotropic vorticity / 2-D incompressible flow

Purpose: smallest fluid laboratory for nonlinear invariant preservation.

Target continuum objects:

- circulation/vorticity transport under declared boundary conditions;
- kinetic energy;
- enstrophy;
- controlled viscous/enstrophy dissipation when viscosity is enabled.

Why it matters: the Arakawa Jacobian literature demonstrates that preserving the right discrete nonlinear structure can suppress spurious numerical cascades even when another formally accurate discretization does not. This is a direct example of "real behavior" depending on structure, not just local truncation order.

### 4.2 Rotating shallow water — first high-value canonical target

Purpose: first model that simultaneously exercises mass, wave propagation, rotation, balanced flow, vorticity/PV, energy, and a higher-order invariant.

Target continuum objects for the inviscid closed/periodic idealization:

- total layer mass;
- total energy;
- materially transported potential vorticity `q = (zeta + f) / h` under the appropriate ideal assumptions;
- potential enstrophy `1/2 integral h q^2 dA`;
- more general PV Casimir families only after the basic case is verified.

Target forced/dissipative variants:

- explicit wind stress / drag / viscosity / mass-source contributions;
- energy-conserving but potential-enstrophy-dissipating options where scientifically/numerically appropriate;
- boundary flux terms for nonperiodic domains.

This should be the first place where the project proves that a discrete scheme can preserve two nontrivial invariants simultaneously and where an intentionally nonconserving competitor is used as a negative control.

### 4.3 Dry compressible Euler / ideal-gas atmosphere

Target objects depend on domain and boundary assumptions, but should include:

- total mass;
- component or total momentum where boundary/coordinate symmetries permit it;
- total energy with kinetic + internal + gravitational potential contributions;
- entropy material conservation only for smooth adiabatic inviscid flow, with an entropy inequality/shock-aware statement if shocks are admitted;
- vorticity/circulation/PV relations under their exact assumptions.

The project must not claim simultaneous exact preservation of mutually incompatible discrete properties without demonstrating it. Compressible-flow discretizations often trade strict energy conservation, entropy stability, positivity, shock robustness, and monotonicity; the chosen policy must be explicit.

### 4.4 Hydrostatic / primitive-equation atmosphere

Target objects:

- dry-air mass and pressure-coordinate continuity;
- total energy for the exact declared hydrostatic thermodynamic system;
- axial angular momentum for a closed, torque-free idealization, with mountain/friction/surface torques exposed when present;
- Ertel-like PV in the declared adiabatic/frictionless limit;
- tracer and constituent budgets;
- geostrophic/hydrostatic balance preservation as a separate balanced-state property, not a conservation law.

The existing pressure-coordinate hydrostatics, Coriolis, continuity work, and conservative extensive transport are adjacent prerequisites, but they are not yet a primitive-equation conservation system.

### 4.5 Moist atmosphere

Moist physics makes careless conservation language especially dangerous. Required objects include:

- total dry-air mass;
- total water across vapor/liquid/ice plus precipitation boundary fluxes;
- moist total energy with a single documented thermodynamic reference convention;
- latent/internal/kinetic/potential energy conversions that cancel internally when they should;
- entropy production with irreversible phase change, diffusion, precipitation, and radiation distinguished from reversible transformations;
- constituent positivity without silently creating or destroying total mass/water/energy.

Any saturation adjustment or microphysics scheme must provide a budget receipt. "Clipping q to saturation" is not an acceptable hidden correction.

### 4.6 Radiation and surface/ocean/land coupling

These are fundamentally open-system terms. The primary object is an exchange ledger:

- top-of-atmosphere shortwave/longwave fluxes;
- surface radiative/turbulent fluxes;
- ocean/land/ice storage changes;
- freshwater and constituent fluxes;
- equal-and-opposite interface exchanges where both sides of a coupled system are modeled.

The coupling layer should make violations of action/reaction or energy exchange symmetry visible. This is conceptually analogous to a physics engine ensuring equal-and-opposite impulses across an internal constraint: an internal exchange should not create net system quantity unless the model says it can.

### 4.7 Wave activity and pseudomomentum

Wave-action, Eliassen-Palm, or pseudomomentum diagnostics should be added only with their background-state and perturbation assumptions declared. They are powerful but easy to overinterpret as globally exact conserved quantities.

### 4.8 Turbulence and scale invariance

The repository should explicitly reject the shortcut "power law -> scale symmetry -> Noether charge".

Required evidence for turbulence-scale claims includes:

- flux diagnostics across scales;
- inertial-range extent and resolution sensitivity;
- forcing and dissipation range separation;
- spectral/structure-function uncertainty;
- finite-domain and anisotropy effects;
- comparison against known cascade phenomenology for the declared dimensionality/rotation/stratification regime.

A scaling exponent is a statistical property, not by itself a conservation theorem.

---

## 5. Lessons to import from mature game physics engines

Game physics is not climate physics, but decades of production rigid-body engines contain engineering lessons about enforcing mathematical structure under finite precision, finite timesteps, heterogeneous constraints, and strict runtime budgets. The transferable lesson is **not** to make climate physics game-like; it is to copy the discipline around constraints, residuals, timestep semantics, and failure visibility.

### 5.1 Fixed timestep identity matters

Box2D recommends a fixed timestep because variable steps produce variable behavior and make debugging difficult. Climate should apply the analogous rule to invariant verification:

- every invariant witness records the actual timestep sequence;
- adaptive integration is allowed, but the controller and accepted/rejected steps are part of the implementation identity;
- comparisons must distinguish spatial-discretization error from time-integration error;
- restart/replay must reproduce the same tolerance/controller policy.

### 5.2 Substepping is often more valuable than pretending one large solve is exact

Box2D and PhysX TGS improve difficult constraints by resolving them over smaller internal time increments. Climate analogues include acoustic/gravity-wave subcycling, split-explicit dynamics, microphysics substeps, chemistry subcycling, and multirate integration.

Rules:

- subcycling cadence is explicit configuration;
- external forcing application across substeps is declared rather than accidental;
- exchanges accumulated over substeps must reconcile with the outer-step budget;
- reducing the outer step and increasing substeps become metamorphic tests.

### 5.3 Report constraint residuals; do not infer quality from iteration count

PhysX explicitly supports solver residual reporting and warns that unusually high iteration counts may indicate a bad configuration rather than a need for more brute force.

Climate equivalent:

- nonlinear/implicit solves report equation residuals and invariant/balance residuals separately;
- a solver that reaches its iteration cap cannot silently return a plausible field;
- tolerance tightening should show the expected residual response;
- invariants must not be "repaired" by increasing iterations without identifying whether the error is spatial, temporal, algebraic, or physical.

### 5.4 Stabilization must have a separate identity from physical dynamics

Rigid-body solvers distinguish geometric correction/bias from the velocity state carried forward. Climate should similarly separate:

- physical tendency;
- numerical stabilization/filter;
- positivity/monotonicity correction;
- remapping/regridding;
- iterative projection/correction.

Every one of those channels receives a budget contribution. This prevents an energy fix, tracer clipping, or divergence cleanup from masquerading as physical evolution.

### 5.5 Warm starting is an optimization, not physical memory

Physics engines reuse previous constraint impulses to accelerate convergence. The stored impulse is a numerical initial guess, not a new physical state variable.

Climate analogue: cached Krylov vectors, previous Jacobians, preconditioners, lagged tendencies, and extrapolated solver guesses must not change the declared mathematical solution beyond tolerance. Cold-start vs warm-start differential tests should detect hidden dependence.

### 5.6 Units and numerical scale are part of solver validity

Box2D explicitly tunes tolerances for an MKS-scale operating range. Climate has a much wider dynamic range and cannot rely on accidental scaling.

Required practice:

- SI units or explicit nondimensionalization at interfaces;
- named scales for normalized residuals;
- conditioning diagnostics for pressure, energy, moisture, and momentum equations;
- rescaling metamorphic tests where the underlying physics permits them;
- no absolute epsilon shared blindly between quantities with different units/magnitudes.

### 5.7 Continuous collision detection has an analogue: do not step over fast physics

Game engines need CCD because a finite step can skip a collision entirely. Climate solvers can similarly step over fast waves, stiff chemistry, rapid saturation/phase transitions, or sharp source activation.

The response should be one of:

- smaller timestep/subcycling;
- implicit/IMEX treatment;
- event-aware integration where scientifically meaningful;
- a fail-closed timestep/CFL/stiffness gate.

Not acceptable: allow a large step and then clip the state back into admissibility without recording the correction.

### 5.8 Determinism, replay, pause, and single-step are scientific debugging tools

Box2D treats deterministic execution as important for debugging. Climate evidence-producing kernels should similarly support:

- fixed input -> reproducible output under a declared determinism class;
- single-step state/budget inspection;
- exact restart metadata;
- process-order logging;
- CPU/GPU differential replay where acceleration exists.

Bitwise identity is not always scientifically necessary, but unexplained nondeterminism is not acceptable in a verification path.

---

## 6. Lessons to import from mature geophysical and scientific models

### 6.1 Conservation should be built into flux/operator structure

FV3's development emphasizes conservative finite-volume transport, consistency of momentum/tracer treatment, no false vorticity generation in its shallow-water lineage, and pressure-gradient forces constructed so internal cell-to-cell forces are equal and opposite. These are stronger design statements than checking a global sum after a step.

Climate should prefer operators whose algebra makes the desired cancellation visible.

### 6.2 The same mass flux must mean the same mass flux across equations

MITgcm documents that tracer conservation with a nonlinear free surface requires tracer fluxes to use a form consistent with the continuity integration. The transferable rule is fundamental:

> quantities advected by mass must use a mass flux consistent with the mass continuity equation, or the coupled conservation claim is structurally broken even if each isolated routine looks reasonable.

This directly constrains future work connecting `conservative_transport.f90` to geometry/velocity-derived face fluxes and pressure-coordinate continuity.

### 6.3 Preserve the invariants that control long-time nonlinear behavior

Arakawa-Lamb-type shallow-water schemes demonstrate that simultaneous energy and potential-enstrophy behavior can materially change nonlinear stability, cascade behavior, and flow regime. A higher-order nonconservative method is not automatically better for long climate integrations.

This does not mean "conserve everything." Controlled potential-enstrophy dissipation with minimal energy dissipation can be more physically/numerically appropriate when unresolved small scales must be removed. The key is to choose the invariant/dissipation policy deliberately and test it.

### 6.4 Mimetic/compatible structure is often worth more than local formula fidelity

Divergence, gradient, curl, incidence, and Hodge-like metric operators should satisfy the appropriate discrete identities. This reduces opportunities for spurious sources that no local unit test will catch.

Examples of useful structural witnesses:

- discrete divergence of discrete curl where mathematically applicable;
- gradient/curl compatibility;
- flux antisymmetry across internal faces;
- integration-by-parts / adjointness identities;
- pressure-work cancellation against kinetic/internal energy transfers;
- PV/vorticity identities on the chosen grid;
- metric/orientation invariance on equivalent meshes.

### 6.5 Balanced-state preservation deserves first-class tests

A model can conserve a global quantity and still generate disastrous spurious motion. FV3's hydrostatic-over-topography tests illustrate why rest/geostrophic balance tests belong beside conservation tests.

Climate needs adversarial balanced states such as:

- hydrostatic resting atmosphere over topography;
- geostrophically balanced flow;
- solid-body rotation on the sphere;
- balanced vortex;
- stationary tracer under zero flow.

The expected result is not merely small global drift; local spurious acceleration and wave generation must be bounded and converge appropriately.

### 6.6 Idealized benchmark suites must precede realistic climate claims

A physically serious dynamical core should pass analytic/manufactured and community idealized tests before real-data interpretation. Candidate families include shallow-water spherical tests, baroclinic waves, mountain waves, resting-atmosphere topography, tracer deformation, and known vortical flows.

Realistic full-physics output is a poor debugging oracle because compensating errors can look plausible.

---

## 7. Required architecture

The following objects should eventually become machine-readable authorities. Names here are design targets, not a requirement to implement these exact structs immediately.

### 7.1 `PhysicalLawSpec`

Fields should include:

- `law_id` and semantic version;
- governing model/equation family;
- semantic class: exact invariant / balance / Casimir / material invariant / inequality / statistical relation;
- mathematical statement;
- state variables and units;
- domain and boundary conditions;
- required smoothness/regularity assumptions;
- forcing/dissipation processes that invalidate exact conservation;
- symmetry origin when one exists;
- reference derivation/citation;
- known non-applicability cases.

### 7.2 `DiscreteLawSpec`

Fields:

- parent `PhysicalLawSpec`;
- mesh/grid/coordinate assumptions;
- discrete state locations;
- spatial operators and their defining identities;
- semi-discrete invariant/balance expression;
- time integrator identity;
- precision and nonlinear/linear solver policy;
- explicit numerical dissipation/correction channels;
- expected conservation order/tolerance, with normalization.

### 7.3 `BalanceReceipt`

Every important prognostic update should be able to emit a compact receipt:

```text
quantity_before
quantity_after
resolved_boundary_flux
physical_sources
physical_sinks
internal_exchange_terms
numerical_filter_or_limiter_change
solver_or_projection_correction
roundoff_estimate_or_scale
unexplained_residual
```

Internal exchanges should cancel when summed over the complete modeled system.

### 7.4 `CorrectionLedger`

Any method that changes the state to maintain admissibility records:

- cells/variables affected;
- raw candidate state;
- corrected state;
- reason;
- quantity changes induced by correction;
- whether the correction is mathematically conservative;
- whether the run remains eligible for a given evidence class.

### 7.5 Independent reference paths

The repository should maintain at least two conceptually distinct authorities where practical:

- symbolic/analytic continuum reference;
- canonical portable numerical implementation;
- optional alternative discretization/reference implementation;
- accelerator path only later.

A conservation test that computes the expected answer using the same flux/operator implementation as the candidate is not independent evidence.

---

## 8. Adversarial verification program

The default posture is to try to falsify the conservation claim.

### 8.1 Positive exact cases

For each law, include the simplest state where the invariant should hold exactly or to roundoff:

- uniform/rest states;
- symmetry-generated analytic trajectories;
- periodic waves with known invariant;
- solid-body rotation;
- exact geostrophic/hydrostatic states;
- known shallow-water steady solutions.

### 8.2 Deliberate symmetry breaking

Add one controlled term known to break the invariant and verify that:

1. the invariant changes;
2. the change matches the declared source/sink term;
3. removing the term restores the ideal behavior.

Examples: drag torque, radiative heating, explicit viscosity, mass source, topographic torque, phase conversion, imposed boundary flux.

### 8.3 Sign adversaries

Flip a source/flux sign in a planted negative fixture and require the test to fail. Budget code that still reports closure after a sign flip is probably reusing candidate algebra rather than independently checking it.

### 8.4 Unit adversaries

Inject Pa vs hPa, J/kg vs J, geopotential vs geometric height, mixing ratio vs specific humidity, and mass flux vs velocity confusions in negative fixtures. These must fail structurally or produce a diagnostic mismatch, never a plausible-looking conservation result.

### 8.5 Coordinate/orientation adversaries

Where the mathematics is coordinate invariant or orientation independent:

- reverse face/edge orientation;
- permute cell ordering;
- rotate longitude origin;
- change equivalent map coordinates;
- compare geometrically equivalent meshes.

Invariant changes beyond declared numerical tolerance are defects unless the quantity is explicitly coordinate dependent.

### 8.6 Resolution and refinement

Run systematic grid refinement and record separately:

- state error;
- invariant drift;
- balance residual;
- dispersive/dissipative error;
- wall-clock cost.

Exact discrete conservation with a wrong solution does not pass the benchmark; convergence and conservation are separate axes.

### 8.7 Timestep and substep adversaries

At minimum compare:

- `dt`, `dt/2`, `dt/4`;
- one vs multiple dynamics substeps;
- warm vs cold solver start;
- split-order permutations when operators are not commuting;
- adaptive tolerance sweeps where applicable.

A spatial invariant that degrades with timestep must be described as semi-discrete, not fully discrete.

### 8.8 Boundary-condition adversaries

Repeat laws under periodic, closed/no-flux, prescribed-flux, and open/radiative boundaries where supported. The expected global budget changes with the boundary semantics; tests should make this visible.

### 8.9 Filter/limiter/remap adversaries

Run with each numerical correction disabled/enabled separately. Require the `CorrectionLedger` to explain any invariant change. A monotonic/positive solution may legitimately dissipate an invariant, but that dissipation must be measured and bounded.

### 8.10 Roundoff/precision adversaries

Compare FP64 with reduced precision where supported. Record residual scaling with problem magnitude and reduction order. A GPU/mixed-precision path needs stage-level differential witnesses before its conservation behavior is treated as equivalent.

### 8.11 Restart and replay

Checkpoint mid-trajectory, restart, and require budget continuity across the seam. Restart metadata must include integrator history, multistep state, solver/controller state that materially affects the solution, and stochastic state where relevant.

### 8.12 Long-time adversaries

Short tests can hide secular drift. Include long integrations that reveal:

- monotonic invariant drift;
- energy pile-up at the grid scale;
- checkerboard/computational modes;
- slowly accumulating tracer mass error;
- coupling leaks;
- solver-tolerance bias.

### 8.13 Pathological but valid states

Stress high/low pressure, thin layers, steep topography, strong shear, near-dry/shallow cells where supported, large density ratios, strong stratification, and near-singular coordinates. Invalid states must fail closed; valid extreme states should not require hidden clipping.

---

## 9. Acceptance gates

No quantity should be called "conserved" in canonical documentation until the relevant gates are satisfied.

### Gate I0 — statement integrity

- exact mathematical statement;
- units;
- domain/boundary conditions;
- forcing/dissipation assumptions;
- semantic class correctly identified.

### Gate I1 — independent continuum/reference authority

- symbolic derivation, published derivation reproduced independently, or exact analytic fixture;
- positive and negative symmetry cases;
- no climate interpretation implied by theorem correctness.

### Gate I2 — semi-discrete structure

- discrete operator identities witnessed;
- discrete invariant/balance derived algebraically;
- planted sign/orientation/unit faults detected.

### Gate I3 — fully discrete numerical behavior

- timestep/refinement study;
- solver tolerance/substep sensitivity;
- long-time drift test;
- numerical correction ledger complete.

### Gate I4 — coupled process closure

- every enabled physical process has an exchange/source/sink term;
- internal transfers cancel across components;
- coupling cadence/order sensitivity characterized;
- restart/replay closure passes.

### Gate I5 — external benchmark agreement

- community benchmark or materially independent implementation;
- same initial/boundary conditions;
- declared error metrics;
- failed/negative comparisons retained.

### Gate I6 — empirical/process validation

- observational/reanalysis budget with data provenance and uncertainty;
- unresolved terms acknowledged;
- no promotion from numerical conservation to empirical truth.

---

## 10. Concrete work packages

### P0. Repair authority language now

- keep `reference/noether_sympy.py` as `reference_variational_symmetry`;
- remove README wording that implies broad climate conservation follows from approximate symmetry;
- describe `climate_symmetries.hs` as legacy hypothesis/budget material;
- link this document from the roadmap.

### P1. Build the balance/invariant contract before adding more theorem machinery

Design the machine-readable `PhysicalLawSpec`/`DiscreteLawSpec` concepts and a portable `BalanceReceipt` pattern. Start with existing kernels:

- conservative mass/tracer transport;
- pressure-coordinate hydrostatics/continuity;
- Coriolis rotation;
- vertical diffusion.

For each, state whether the kernel should conserve, exchange, dissipate, or merely diagnose the relevant quantity.

### P2. Add a barotropic-vorticity invariant laboratory

Implement an independently testable 2-D periodic model with at least two nonlinear discretizations:

1. a structure-preserving Arakawa-type Jacobian;
2. a simpler/nonconserving comparison path.

Witness energy/enstrophy behavior, convergence, and long-time cascade contamination. This provides a small hard test of whether the project can preserve a nontrivial fluid invariant for the right reason.

### P3. Add a rotating shallow-water canonical laboratory

This is the highest-value first serious higher-order-invariant target.

Required before `runnable` promotion:

- explicit equations, grid/metric, boundary conditions, and Coriolis semantics;
- mass conservation;
- energy functional;
- PV definition;
- potential enstrophy / Casimir reference;
- discrete operator identities;
- balanced-state tests;
- convergence and long-time tests;
- comparison of conservation-oriented and deliberately different numerical schemes;
- controlled forcing/dissipation variants.

Do not begin with a full 3-D moist atmosphere; that would make invariant failures too hard to localize.

### P4. Connect primitive-equation mechanics only after the pressure-gradient/momentum contract exists

Extend the canonical Fortran dynamics frontier with pressure-gradient and momentum operators whose internal work/force exchanges can be checked. Then connect:

- Coriolis (no isolated kinetic-energy work);
- pressure work;
- continuity/mass flux;
- kinetic/potential/internal energy conversion;
- angular-momentum torque terms.

### P5. Add moist total-water and energy receipts

Only after condensate/latent thermodynamics has an explicit contract. Phase changes must be internally energy-consistent and total-water-consistent before any microphysics scheme is called physically conservative.

### P6. Add continuum field/relabeling mathematical references

Once at least one canonical fluid model exists, add the correct mathematical references for:

- field-theoretic Noether currents for spacetime symmetries;
- particle-relabeling/circulation/PV relationships;
- noncanonical Hamiltonian bracket and Casimirs for the chosen shallow-water/fluid model.

This order is deliberate: build the reference for an actual deployed model, not a broad abstract framework waiting for a use case.

### P7. Add empirical angular-momentum/energy/water budget experiments

Treat legacy source lists as leads, then verify current data products and construct immutable projections. Candidate later experiments include atmospheric angular momentum plus torque terms, TOA/surface energy balance, and total-column water budgets. These are validation studies, not theorem tests.

---

## 11. Priority of higher-order quantities

Not every mathematically conserved quantity deserves implementation. Prioritize by physical/numerical leverage.

1. **Potential enstrophy / PV Casimirs in rotating shallow water** — high leverage for nonlinear vortical dynamics and a classic discriminator between discretizations.
2. **Ertel PV / circulation limits in a dry adiabatic fluid** — high physical relevance but requires a more complete dynamical core.
3. **Axial angular momentum and torques** — physically interpretable and externally diagnosable, but geometry, pressure, surface, and mountain torque bookkeeping must be correct.
4. **Helicity** — useful in selected 3-D inviscid/barotropic contexts; lower priority until the supported model actually has the assumptions and resolution to make it informative.
5. **Wave action / pseudomomentum** — useful for wave-mean-flow interaction but must remain perturbation/background-state scoped.
6. **Arbitrary formal Casimir families** — defer until a concrete diagnostic or numerical design question requires them.

The rule is the same as elsewhere in Climate: mathematical richness does not earn runtime authority by itself.

---

## 12. What "adherence to real behavior" means here

The project should resist two opposite mistakes.

### Mistake A: theorem-first formalism with weak physics

Symptoms:

- exact symbolic charge with no deployed governing equation;
- calling approximate stationarity a symmetry;
- treating all invariants as Noether charges;
- no forcing/dissipation terms;
- no grid/time-step dependence study;
- no benchmark flow.

### Mistake B: engine-style correction that merely looks stable

Symptoms:

- clipping variables until budgets look plausible;
- numerical damping with no energy/enstrophy accounting;
- conservation fixed by global renormalization after each step;
- tuned tolerances that hide instability;
- visually smooth output treated as physical correctness.

The target is between them:

> derive the right law, discretize it with structure-aware operators, expose every intentional violation, and stress it until the remaining residual has a defensible explanation.

A real physics engine is not trustworthy because objects look plausible; it is trustworthy when the solver's approximations, constraints, tolerances, and failure modes are understood. A scientific climate model demands the same engineering discipline plus a much stronger obligation: the equations and closures themselves must also be scientifically justified.

---

## 13. Immediate next decisions

Before implementing more Noether machinery, do these in order:

1. land the pressure-coordinate continuity work independently if its canonical CI lane remains green;
2. repair the README Noether overstatement and link this plan;
3. define the first `PhysicalLawSpec` vocabulary around already-realized mass/tracer and Coriolis kernels;
4. specify the barotropic-vorticity and rotating-shallow-water laboratories, including exact invariants and negative-control discretizations;
5. implement the smallest laboratory and make the adversarial tests fail for the wrong schemes before optimizing anything;
6. only then extend the symbolic authority toward continuum/relabeling/Hamiltonian structure for the model that actually exists.

This sequencing preserves the repository's core rule: a strong mathematical name creates an implementation obligation, but mathematics must be attached to the correct physical object rather than used as a substitute for it.

---

## 14. External precedents to study, not copy blindly

These references are included because they encode hard-earned design lessons relevant to the work above.

### Numerical/geophysical modeling

- Arakawa, A. and V. R. Lamb (1981), *A Potential Enstrophy and Energy Conserving Scheme for the Shallow Water Equations*, Monthly Weather Review, DOI `10.1175/1520-0493(1981)109<0018:APEAEC>2.0.CO;2`.
- Arakawa, A. and Y.-J. G. Hsu (1990), *Energy Conserving and Potential-Enstrophy Dissipating Schemes for the Shallow Water Equations*.
- Salmon, R. (2004), *Poisson-Bracket approach to the construction of energy- and potential-enstrophy-conserving algorithms for the shallow-water equations*.
- Eldred, C. and D. Randall (2017), *Total energy and potential enstrophy conserving schemes for the shallow water equations using Hamiltonian methods – Part 1*, Geoscientific Model Development 10, 791–810.
- Salmon, R. (1988), fluid particle-relabeling / Noether derivations of vorticity-law structure; use as mathematical background, not as automatic authority for a discrete climate model.
- GFDL FV3 design/documentation: `https://www.gfdl.noaa.gov/fv3/` and `https://www.gfdl.noaa.gov/fv3/fv3-key-components/`.
- MITgcm documentation, especially momentum, energy-conservation, and tracer/free-surface consistency sections: `https://mitgcm.readthedocs.io/en/latest/`.
- MPAS-Atmosphere technical documentation: `https://www2.mmm.ucar.edu/projects/mpas/site/documentation.html`.

### Production physics-engine engineering

- Box2D simulation documentation: `https://box2d.org/documentation/md_simulation.html` — fixed timestep, substeps, continuous collision, determinism, persistence/warm starting, tolerances.
- Erin Catto, *Modeling and Solving Constraints* (GDC): `https://box2d.org/files/ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf` — iterative constraints and warm starting.
- NVIDIA PhysX simulation documentation: `https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html` — PGS/TGS, substep behavior, solver iterations, residual reporting, force application.

These systems solve different physical problems. Their value here is methodological: make discretization and solver behavior explicit, measure residuals, expose stabilization, use reproducible stepping, and design tests around known failure modes rather than around attractive output.
