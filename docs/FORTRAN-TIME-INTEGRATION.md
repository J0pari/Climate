# Fortran time-integration semantics

This document defines the numerical semantics of the canonical Fortran time-integration surface. It is not a transcription of `climate_physics_core.f90`. The legacy source is treated as evidence about intended physical/numerical responsibilities; canonical code may use different algorithms when they provide equal or greater correctness, semantic precision, performance, diagnosability, or future extensibility.

## Design rule: decomposition is not simplification

The canonical surface separates independently testable contracts that were interleaved in the legacy monolith:

- explicit ODE stepping,
- linear implicit solves,
- operator construction,
- source/tendency evaluation,
- physical bounds and conservation policy,
- state diagnostics,
- timestep selection,
- splitting/IMEX policy,
- grid and boundary semantics,
- and climate-specific process coupling.

Separating these concerns is allowed only when the missing semantics remain explicit obligations. A smaller kernel must not silently erase capabilities or physical assumptions that previously existed in one large routine.

## Explicit stepping

`src/fortran/time_integration.f90` currently provides deterministic classical RK4 for a real state vector,

\[
y_{n+1}=y_n+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
\]

The right-hand side is an explicit callback with its own failure status. The numerical kernel does not know the units, physical meaning, or provenance of the state vector. It fails closed on callback failure or non-finite stage values.

This is a reference primitive, not a claim that fixed-step RK4 is the preferred climate integrator.

## Linear implicit stepping

`src/fortran/linear_implicit.f90` represents a tridiagonal linear operator \(L\) with lower, diagonal, and upper bands. For an additive source \(s_n\), the theta step is

\[
(I-\theta\Delta t L)y_{n+1}
=
y_n+\Delta t\left[(1-\theta)Ly_n+s_n\right].
\]

Important semantics:

- `theta=0.5` is the Crank--Nicolson member of this family.
- `theta=1` is backward Euler for the linear operator.
- `explicit_source` is evaluated and owned by the caller; the kernel does not silently re-evaluate or reinterpret it.
- operator coefficients and boundary conditions are caller-owned scientific/numerical contracts.
- timestep positivity is enforced by this climate-facing primitive.
- no implicit claim is made that every physical process belongs in the same linear operator.

## Factorization and performance

The legacy code allocates three dense `nz x nz` matrices while only using their diagonal bands. The canonical solver stores the true tridiagonal representation and performs an O(n) Thomas factorization.

Factorization and solve are separate operations so one factorization can be reused across multiple right-hand sides when the matrix is unchanged. This is important for repeated column solves, ensembles, multiple tracers, and later batched CPU/GPU implementations.

Reuse is only valid when the exact matrix is unchanged. Any change in operator coefficients, boundary conditions, `theta`, or `dt` that changes the left-hand matrix requires a new factorization.

## Numerical policy is explicit

Pivot acceptance uses two caller-supplied tolerances:

\[
\tau = \tau_{abs}+\tau_{rel}\lVert A\rVert_\infty.
\]

A pivot with magnitude less than or equal to `tau` is rejected. The kernel does not embed a universal magic singularity threshold.

Every successful solve reports:

- matrix infinity norm,
- minimum absolute pivot,
- minimum pivot scaled by matrix infinity norm,
- and algebraic residual infinity norm.

These are measurements, not acceptance policy. Higher layers decide what residual or conditioning is acceptable for a given experiment or evidence claim.

## Current witnesses

The portable Fortran witnesses currently require:

- exact integration of a constant tendency by RK4,
- fourth-order RK4 convergence under timestep halving,
- coupled harmonic-oscillator evolution over one period,
- fail-closed RHS and invalid-step handling,
- recovery of a known tridiagonal solution,
- factorization reuse across distinct right-hand sides,
- rejection of a scale-small pivot under relative pivot policy,
- rejection of a pivot that becomes zero during elimination,
- exact agreement with the scalar Crank--Nicolson rational update,
- preservation of the constant-field nullspace of a zero-flux diffusion operator,
- and fail-closed invalid-theta handling.

These witnesses establish software/numerical behavior only. They do not validate a climate discretization, timestep, parameterization, or physical process.

## Legacy mapping

The legacy `time_integrate` routine mixes several different operations:

- explicit tendency application,
- an analytically implicit Coriolis update,
- vertical temperature diffusion,
- moisture forward Euler plus clipping,
- pressure/density updates,
- diagnostic vertical velocity and geopotential,
- diffusion filtering,
- and divergence damping.

Those responsibilities should not be copied into one replacement routine. Each should either map to a canonical kernel with its own witnesses or remain an explicit unresolved obligation.

In particular, the legacy temperature solve suggests a tridiagonal implicit operator, but its exact current implementation is not authoritative: it uses dense storage for banded data, has undeclared/invalid local declaration placement under standard Fortran, has no pivot policy, and refers to state/grid fields that do not exist in the declared types. The canonical layer preserves the useful mathematical intent while discarding those implementation defects.

## Next integration frontier

The next higher layer should define a climate-specific operator adapter with explicit units, vertical coordinate semantics, boundary conditions, and coefficient provenance. That adapter can then be tested against manufactured diffusion problems and conservation/nullspace properties.

After that, the final coupled integrator should be selected by requirements rather than legacy fidelity. Candidate families include higher-order IMEX Runge--Kutta/ARK schemes, operator splitting, semi-Lagrangian methods, exponential integrators for appropriate linearized components, or multirate methods. Selection should compare stability region, conservation behavior, stiffness handling, cost, parallel structure, reproducibility, and compatibility with the physical process partition.

A more sophisticated method may supersede the theta primitive. The theta primitive remains useful as a reference oracle and a directly testable building block even if it is not the final production stepping method.
