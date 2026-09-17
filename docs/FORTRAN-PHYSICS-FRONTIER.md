# Fortran physics realization frontier

This document defines atmospheric physical and numerical responsibility boundaries, composition rules, and external-library seams. It is a semantic specification, not a hand-maintained inventory of which modules happen to exist.

Objective realization state belongs to the machine-readable module registry and its generated projection in [`docs/generated/STATUS.md`](generated/STATUS.md). Build and witness availability belong to executable CMake/CTest and CI surfaces. Those authorities should be changed directly rather than copied into a prose table here.

Canonical APIs, algorithms, state layouts, numerical methods, and scientific interpretations must be justified by their physical and numerical contracts rather than inherited from any particular source layout.

## Pressure-coordinate composition seam

Pressure-coordinate atmospheric pieces must compose around shared physical quantities rather than independently reconstructed equivalents.

### Coordinate authority

A pressure-coordinate column uses one declared interface ordering and one definition of positive pressure thickness. Hydrostatics, continuity, transport coupling, and pressure-work diagnostics must consume the same coordinate semantics when they participate in one calculation.

The coordinate contract must make explicit:

- interface pressure units and ordering;
- layer pressure thickness;
- lower/upper interface orientation;
- boundary ownership;
- whether pressure levels are fixed or move with surface pressure;
- any mapping to geometric height, hybrid/sigma coordinates, or mass coordinates.

A module that only consumes fixed pressure levels must not imply surface-pressure evolution or a geometric vertical velocity.

### Hydrostatics

Hydrostatic relationships own pressure mass, geopotential thickness, and geometric thickness under their declared thermodynamic and gravity assumptions. They do not own continuity, horizontal momentum, pressure-gradient discretization, or time integration.

Layer-mean virtual temperature is a thermodynamic input to hydrostatics, not a license for the hydrostatic layer to select a moist equation of state, saturation policy, or condensate treatment.

### Continuity and pressure velocity

Pressure-coordinate continuity relates pressure-mean horizontal divergence to interface pressure velocity `omega`. The continuity layer owns the sign/orientation identity and residual diagnostics; it does not choose the horizontal grid, derivative stencil, velocity reconstruction, or boundary-closure policy.

When a coupled finite-volume calculation has authoritative face-integrated mass fluxes, the divergence supplied to continuity must be derived from those same fluxes and the declared cell geometry. A separately recomputed divergence that can disagree with the transported mass flux violates the coupling contract.

### Pressure-gradient and energy exchange

Pressure-coordinate energy coupling must keep these roles distinct:

- horizontal pressure-gradient acceleration and its kinetic-energy power;
- `alpha * omega` pressure work in the thermodynamic/enthalpy equation;
- geopotential material tendency;
- any explicit local geopotential tendency associated with moving boundaries or other declared effects.

The local hydrostatic exchange identity is a diagnostic/coupling contract. It does not choose a horizontal derivative, equation of state, momentum discretization, or time integrator, and it must not be described as discrete total-energy conservation until a coupled discrete momentum/thermodynamic transport scheme demonstrates that property.

### Mass and tracer transport

The extensive transport layer consumes face-integrated carrier-mass fluxes and tracer face states. It owns conservative budget algebra and fail-closed state admissibility, not geometry, velocity-to-flux conversion, high-order reconstruction, limiter design, or timestep/CFL selection.

A coupled transport path has one authoritative carrier-mass flux per face. Carrier mass, tracer mass, pressure-coordinate continuity, and any energy transport depending on carrier mass must use that authority rather than independently approximating the same flux.

## Physical responsibility boundaries

The following responsibilities remain separate scientific or numerical contracts even when several are composed in one atmospheric step:

- **condensed-water thermodynamics:** liquid/ice condensate loading, latent heats, phase transitions, mixed-phase partition, and supersaturation/adjustment policy beyond an equilibrium saturation provider;
- **vertical-coordinate dynamics:** surface-pressure evolution and moving lower boundaries, geometric vertical velocity, boundary-condition/closure policy, and hybrid/sigma/mass-coordinate mappings;
- **momentum dynamics:** pressure-gradient force, advection, metric terms, vertical momentum or diagnostic-omega semantics, diffusion/friction, and their conservation properties;
- **mass-flux construction:** coordinate metrics, face areas, density/velocity coupling, continuity-consistent face mass fluxes, and boundary orientation;
- **tracer reconstruction:** high-order face states, monotonicity/positivity strategy, and accuracy/dispersion controls;
- **radiative transfer:** spectroscopy, gas optics, cloud/aerosol optical properties, and radiative solver semantics;
- **surface/land/ocean exchange:** state contracts, flux sign conventions, conservation, and coupling cadence;
- **chemistry:** species inventory, reaction-mechanism authority, stiffness handling, solver configuration, and mass/element budgets;
- **coupled integration:** tendency partition, operator splitting or IMEX/multirate mapping, error tolerances, conservation monitors, and restart/replay semantics.

Separating these concerns must expose the interfaces between them rather than erasing the missing physics behind a smaller function signature.

## Coupled physical closure

Local correctness of individual kernels is necessary but insufficient. Coupled dynamics must make transfers between reservoirs and equations explicit.

High-priority coupled contracts include:

- horizontal divergence and face mass fluxes that agree with continuity;
- pressure-gradient work paired consistently with thermodynamic/geopotential energy conversion;
- Coriolis terms that remain skew/no-work under the chosen spatial discretization;
- tracer, moisture, and energy transport constructed from compatible carrier-mass fluxes;
- latent/internal-energy exchange for phase change;
- precipitation mass and energy export;
- kinetic-energy loss through drag/diffusion with an explicit decision about dissipative heating;
- surface, radiative, and component fluxes represented as exchanges or external sources with declared signs and budgets.

The dynamical core is therefore judged on coupled mass, momentum, energy, balance, and wave behavior rather than by accumulating independently correct tendency routines.

## Separation of climate semantics from generic numerics

Climate-owned code should define climate variables, coordinates, physical partitions, balance laws, diagnostics, adapters, and verification fixtures. Generic numerical machinery should be delegated when a maintained package provides the required semantics.

- Production adaptive/IMEX/multirate time integration should target SUNDIALS ARKODE unless a concrete requirement demonstrates a better fit.
- CPU tridiagonal factorization/solve uses LAPACK; a transparent Thomas implementation is appropriate only as a reference/differential oracle.
- Production CPU FFT work should use FFTW, with cuFFT for an appropriate GPU path; a direct DFT is useful as a transparent reference oracle.
- Radiative-transfer work should evaluate maintained packages such as RTE+RRTMGP before implementing generic gas-optics or two-stream machinery locally. Climate-owned code should concentrate on scientifically explicit inputs, adapters, diagnostics, validation, and evidence.
- Saturation-vapor-pressure code may implement a named published parameterization when the formula itself is part of the scientific contract; phase selection, validity bounds, and composition semantics remain explicit.
- Production face reconstruction, remapping, linear algebra, optimization, automatic differentiation, and similar commodity machinery should not become project-local infrastructure by default. Candidate methods must earn local implementation only when an independently valuable reference is needed or a climate-specific semantic cannot be expressed through a maintained dependency.

Selecting an external package or published parameterization does not by itself validate a scientific process. Version, configuration, implementation identity, numerical diagnostics, input provenance, and differential/benchmark evidence remain required.

## Canonicalization rule

A physical or numerical responsibility becomes canonical only when all of the following are true:

1. its physical variables, units, coordinates, signs, and validity domain are explicit;
2. invalid or unavailable states fail closed rather than silently clamping or switching implementation identity;
3. executable witnesses test invariants, metamorphic relations, or independent references rather than only example outputs;
4. the module is registered with honest maturity and known gaps;
5. generic machinery is delegated to a maintained library when that reduces bespoke numerical risk;
6. climate-specific composition seams identify which quantities must share one authority across neighboring kernels;
7. the slice does not imply that unresolved neighboring physics has been implemented.
