# Fortran physics realization frontier

This document maps responsibilities found in the legacy Fortran monolith to canonical kernels, external-library boundaries, or explicit open obligations. It is a semantic inventory, not a global implementation sequence.

The legacy source is evidence about intended responsibilities. It is not authoritative for APIs, algorithms, state layout, numerical method, or scientific validity when those choices are internally inconsistent or better handled by maintained libraries.

## Realized canonical slices

| Responsibility | Canonical surface | Current meaning | What it does not establish |
| --- | --- | --- | --- |
| explicit ODE reference stepping | `src/fortran/time_integration.f90` | deterministic classical RK4 oracle with fail-closed callback/state handling | production adaptive/IMEX integration or climate timestep validity |
| tridiagonal reference algebra | `src/fortran/linear_implicit.f90` | transparent Thomas/theta-method oracle with explicit pivot policy | production robustness on pivoting systems |
| production-oriented CPU tridiagonal solve | `src/fortran/lapack_tridiagonal.f90` | LAPACK `DGTTRF`/`DGTTRS` adapter with pivot/residual diagnostics | operator correctness, conditioning acceptance, or physical validity |
| isolated constant-f Coriolis flow | `src/fortran/coriolis_rotation.f90` | exact skew-symmetric 2D rotation preserving the horizontal velocity norm | coupled splitting order, varying-f discretization, or full momentum dynamics |
| geometric-height scalar diffusion | `src/fortran/vertical_diffusion.f90` | nonuniform finite-volume operator with explicit boundaries, conservation, and dissipativity witnesses | pressure/sigma/hybrid-coordinate diffusion, closure validity, or thermodynamic weighting |
| dry ideal-gas identities | `src/fortran/dry_thermodynamics.f90` | Exner, potential-temperature transforms, dry density, and isothermal hydrostatic thickness with explicit SI parameters | moist thermodynamics, atmospheric state evolution, general hydrostatic discretization, or energy closure |
| water-vapor mixture algebra | `src/fortran/moist_vapor_algebra.f90` | exact ideal-gas transforms among vapor pressure, mixing ratio, specific humidity, virtual temperature, and vapor-only moist density | saturation-vapor-pressure law, phase equilibrium, condensate, latent heat, microphysics, or prognostic moisture evolution |
| portable spectral oracle | `src/fortran/spectral_reference.f90` | deterministic direct-DFT/analytic-signal reference behavior | a production FFT implementation or superiority of any climate oscillation method |

## Open physical responsibilities

The following remain explicit obligations rather than being inferred from the presence of neighboring kernels:

- **saturation and condensed-water thermodynamics:** phase-specific equilibrium vapor pressure, liquid/ice convention, latent heats, condensate loading, supersaturation policy, and phase transitions;
- **hydrostatic/state coupling:** a declared vertical coordinate and mass/pressure/geopotential relationship beyond the analytic isothermal identity;
- **momentum dynamics:** pressure-gradient force, advection, metric terms, vertical momentum/diagnostic-omega semantics, diffusion/friction, and their conservation properties;
- **mass and tracer transport:** flux form, positivity/conservation policy, coordinate metrics, and boundary/source semantics;
- **radiative transfer:** spectroscopy, gas optics, cloud/aerosol optical properties, and solver semantics;
- **surface/land/ocean exchange:** state contracts, flux sign conventions, conservation, and coupling cadence;
- **chemistry:** species inventory, reaction mechanism authority, solver choice, stiffness handling, and mass-element budgets;
- **coupled integration:** tendency partition, operator splitting or IMEX/multirate method, error tolerances, conservation monitors, and restart/replay semantics.

## Library boundaries

Generic numerical or domain-standard machinery should be delegated when a maintained package already owns it better than this repository can justify reimplementing it.

- Production adaptive/IMEX/multirate time integration should target SUNDIALS ARKODE unless a concrete requirement demonstrates a better fit.
- CPU tridiagonal factorization/solve uses LAPACK; the hand-written Thomas path remains a reference oracle.
- A future production CPU FFT should use FFTW (and a GPU path cuFFT) while the direct DFT remains the transparent differential oracle.
- Radiative-transfer work should evaluate established maintained packages such as RTE+RRTMGP before implementing generic gas-optics or two-stream machinery locally. Climate-owned code should focus on scientifically explicit inputs, adapters, diagnostics, validation, and evidence.
- Saturation-vapor-pressure implementations must identify their liquid/ice phase model and validity range explicitly; vapor-mixture algebra must not silently select one.

Selecting an external package or published parameterization does not by itself validate a scientific process. Version, configuration, implementation identity, numerical diagnostics, input provenance, and differential/benchmark evidence remain required.

## Current extraction rule

A legacy responsibility becomes canonical only when all of the following are true:

1. its physical variables, units, coordinates, signs, and validity domain are explicit;
2. invalid or unavailable states fail closed rather than silently clamping or switching implementation identity;
3. executable witnesses test invariants, metamorphic relations, or independent references rather than only example outputs;
4. the module is registered with honest maturity and known gaps;
5. generic machinery is delegated to a maintained library when that reduces bespoke numerical risk;
6. the new slice does not imply that unresolved neighboring physics has been implemented.

The legacy compile probe remains useful as a debt sensor while extraction proceeds. Its failure is not a reason to mutate canonical state types until the underlying physical contract is independently specified.
