# Fortran time-integration semantics

This document defines the numerical semantics of the canonical Fortran time-integration surface and the boundary between Climate-owned physical semantics and maintained generic numerical infrastructure.

## Design rule: decomposition is not simplification

The canonical surface separates independently testable contracts:

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

Separating these concerns is allowed only when the missing semantics remain explicit obligations. A smaller kernel must not erase physical assumptions or coupled responsibilities merely because they live outside that kernel.

## Library boundary: reference locally, use native production numerics through explicit integration

Climate should not spend novelty budget reimplementing mature generic numerical algorithms. External ownership of an integrator or solver does not remove its timestep, tolerance, convergence, restart, or backend semantics from Climate's reproducibility boundary; those semantics must be bound and receipted when they affect evidence.

The hand-written RK4, theta-step, and Thomas-factorization code is small, inspectable, deterministic, and useful as a differential/reference oracle. It is not intended to grow into a bespoke production solver stack.

For production-oriented CPU integration, the preferred trajectory is SUNDIALS ARKODE rather than implementing a project-local adaptive/embedded Runge--Kutta, IMEX, multirate, nonlinear-solver, and error-controller stack. ARKODE supplies explicit, implicit, additive IMEX, and multirate methods plus reusable vector/matrix/nonlinear/linear-solver interfaces and modern Fortran bindings. Climate should supply the equation-derived tendency partition, Jacobians/operators, tolerance semantics, conservation monitors, and evidence capture around that library.

For tridiagonal linear solves, LAPACK `DGTTRF`/`DGTTRS` is the production-oriented CPU backend because it provides partial pivoting and reusable factorization. `src/fortran/linear_implicit.f90` is the transparent no-pivot reference path. Differential tests deliberately cover both a system where the two agree and a system that requires LAPACK pivoting and must be rejected by the reference Thomas path.

PETSc TS is a plausible distributed-system option when Climate has a genuinely domain-decomposed, MPI-scale state representation; it is not justified merely to solve independent vertical columns. The dependency should enter only when its scalable vector/matrix/preconditioner and TS machinery solve an actual repository problem better than the lighter ARKODE/LAPACK path.

GPU production paths should likewise prefer maintained vendor/library solvers where their semantics fit, while keeping CPU reference kernels for differential verification. A GPU library call is not evidence of scientific correctness; implementation identity, precision, determinism, residuals, conditioning, and CPU/GPU agreement remain part of the evidence boundary.

When Climate analyzes an externally owned climate model, the model's native time-integration stack remains authoritative for that run. Climate may compare its consequences through state, tendency, budget, restart, or response artifacts, but should not reconstruct a local integrator and present the result as the external model. If native step/restart metadata are needed for the scientific question, their absence is an integration limitation that must remain explicit.

## Explicit stepping

`src/fortran/time_integration.f90` provides deterministic classical RK4 for a real state vector,

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

The canonical reference solver stores the true tridiagonal representation and performs an O(n) Thomas factorization.

Factorization and solve are separate operations so one factorization can be reused across multiple right-hand sides when the matrix is unchanged. This is important for repeated column solves, ensembles, multiple tracers, and batched CPU/GPU implementations.

Reuse is only valid when the exact matrix is unchanged. Any change in operator coefficients, boundary conditions, `theta`, or `dt` that changes the left-hand matrix requires a new factorization.

The LAPACK backend preserves the same reusable-factorization boundary but permits partial pivoting. The repository wrapper records whether row interchanges occurred and computes an explicit residual against the original matrix so a library success code is not treated as sufficient evidence by itself.

## Numerical policy is explicit

The reference Thomas path uses caller-supplied absolute and relative pivot tolerances:

\[
\tau = \tau_{abs}+\tau_{rel}\lVert A\rVert_\infty.
\]

A pivot with magnitude less than or equal to `tau` is rejected. The kernel does not embed a universal magic singularity threshold.

The LAPACK path does not pretend that LAPACK's singular/non-singular status is a task-specific conditioning policy. It reports matrix scale, the factored U-diagonal scale, pivoting use, and the algebraic residual; higher layers still own scientific/numerical acceptance policy.

These are measurements, not acceptance policy. Higher layers decide what residual or conditioning is acceptable for a given experiment or evidence claim.

## Witnesses

The portable Fortran witnesses require:

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
- fail-closed invalid-theta handling,
- differential agreement between LAPACK and the reference solver when pivoting is unnecessary,
- and successful LAPACK partial pivoting on a system the reference Thomas path must reject.

These witnesses establish software/numerical behavior only. They do not validate a climate discretization, timestep, parameterization, or physical process.

## Climate-specific integration frontier

The higher integration layer must define climate-specific operators with explicit units, coordinate semantics, boundary conditions, coefficient provenance, and physical exchange meaning. Those operators should be tested against manufactured problems, conservation/balance laws, nullspaces, and appropriate wave or mode behavior before being coupled through a general integrator.

The production integrator should be selected by physical and numerical requirements. SUNDIALS ARKODE is the default external candidate because it supplies higher-order adaptive ERK/DIRK/ARK and multirate machinery. Climate's novel work is the physically meaningful explicit/implicit/fast partition, Jacobians or linear operators, tolerance semantics, conservation monitors, reproducibility/evidence capture, and comparative experiments—not reimplementation of generic time-step controllers.

Integrator choice must not define the physics partition after the fact. The partition should follow from the equations and desired discrete structure: conservative or reversible dynamics, stiff dissipative processes, fast wave processes, external forcing, and exchange terms should be separated for scientific reasons first, then mapped onto the capabilities of the chosen solver.

An external method that provides the required adaptivity, stability, implicit/explicit partitioning, or multirate behavior may supersede the theta primitive for production stepping. The theta primitive remains a reference oracle and directly testable building block.
