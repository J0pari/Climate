# Climate

Climate is an experimental research repository for climate dynamics, mathematical representations of climate state, and the numerical structures needed to study them rigorously.

The central scientific goal is not to collect unusual mathematics for its own sake. It is to determine whether physically grounded, statistically grounded, and mathematically distinct representations of the climate system can be integrated into a useful description of climate state and evolution.

Two research programs meet at that point:

- **structure-preserving climate dynamics** — build physical operators whose discrete behavior respects the balances, exchanges, symmetries, constraints, and dissipative structure of the equations they represent;
- **multirepresentation climate geometry** — construct and compare multiple scientifically meaningful views of the same climate state, then determine whether their relationships are best described by shared manifolds, products, fibers, quotients, local atlases, statistical geometry, kernels, sheaf structure, or other appropriate mathematical objects.

Neither program is assumed to have a single final formulation. The repository exists to make those questions precise enough to answer.

## Scientific focus

### Structure-preserving dynamics

Climate dynamics couple mass, momentum, thermodynamics, moisture, radiation, transport, and exchange among atmosphere, ocean, land, ice, and other reservoirs. Numerical implementations should make those relationships explicit rather than treating conservation or balance as an after-the-fact diagnostic.

Important targets include:

- continuity-consistent mass and tracer transport;
- pressure-coordinate hydrostatics and vertical motion;
- momentum dynamics with correct pressure-work and Coriolis structure;
- energy-consistent coupling among kinetic, potential, internal, latent, and radiative reservoirs;
- physically signed dissipation, drag, diffusion, precipitation, and surface exchange;
- spatial operators that preserve the balances and wave properties relevant to rotating geophysical flow;
- time integration chosen to respect the declared physical partition rather than defining it.

Small exact or transparent kernels are used where they clarify the mathematics and provide independent reference behavior. Mature generic numerical machinery should normally come from maintained libraries behind thin, testable interfaces.

### Multirepresentation climate state

A climate state can be viewed through many scientifically distinct maps. Examples include:

- prognostic physical fields and thermodynamic coordinates;
- subsystem and multiscale summaries;
- spectral, modal, Koopman, and oscillatory representations;
- observation-space and retrieval-space representations;
- parameter, likelihood, and uncertainty distributions;
- teleconnection and hierarchical relational structure;
- local data-consistency structures over heterogeneous observing systems;
- learned latent representations constrained by physical or statistical structure.

These views are not assumed to be interchangeable coordinates in one large feature vector. Their relationships are themselves part of the research question.

A candidate representation may contribute as a coordinate chart, factor, fiber, observation map, constraint, kernel, quotient, local compatibility structure, or comparison geometry. Information geometry is especially relevant when an explicit likelihood provides a Fisher metric or another statistically interpretable local geometry. Null directions and non-identifiability are retained as information rather than hidden by automatic regularization.

The repository investigates whether a useful integrated geometry can preserve or reveal:

- physically meaningful neighborhoods of climate state;
- slow/fast and subsystem structure;
- identifiable and weakly observed directions;
- balanced and unbalanced dynamical modes;
- conservation and exchange structure;
- teleconnection organization;
- regime and trajectory geometry;
- predictive structure that is not available from simpler representations.

Curvature, geodesics, holonomy, topology, intrinsic dimension, and related quantities are properties to study after a representation is defined. They are not universal objectives for choosing the representation.

## Mathematical research families

Several mathematical families contribute to this program with different roles and different degrees of centrality:

- **Riemannian and differential geometry** for explicit state-space metrics, connections, curvature, geodesics, pullbacks, and representation maps;
- **information geometry** for likelihood- and distribution-induced geometry, identifiability, statistical distinguishability, and natural-gradient questions;
- **Hamiltonian, variational, Noether, Poisson, Nambu, and compatible-discretization ideas** where conservative subdynamics genuinely possess that structure;
- **sheaf and cohomological methods** for local-to-global consistency over heterogeneous observations and overlapping domains;
- **ultrametric and p-adic representations** as hypotheses about hierarchical teleconnection or relational structure;
- **Clifford/geometric algebra** as a possible representation of oriented, multicomponent, or resonant mode interactions;
- **modal and logical structures** where they give precise semantics to constrained scenario or accessibility questions.

The repository does not assume that every one of these belongs inside a single smooth manifold. A successful integration may be heterogeneous.

## Executable foundations

The executable surface is intentionally narrower than the research agenda. It contains small portable physical kernels, independently checkable mathematical references, canonical numerical primitives, and explicit verification fixtures. Exact module inventory, lifecycle, maturity, registered experiments, and claim status are generated from machine-readable authorities in [`docs/generated/STATUS.md`](docs/generated/STATUS.md).

The physical realization boundary is described in [`docs/FORTRAN-PHYSICS-FRONTIER.md`](docs/FORTRAN-PHYSICS-FRONTIER.md); geometry, sheaf, and other method-specific verification obligations live in their corresponding specifications.

## Architecture

Climate separates four kinds of authority:

1. mathematical or independently checkable reference definitions;
2. canonical portable implementations;
3. optimized or hardware-specific implementations;
4. empirical climate interpretation.

A correct implementation of a mathematical object does not by itself establish that the object is useful for climate science. Likewise, a physical kernel that satisfies its local equations does not by itself establish that a coupled model is correct.

The repository therefore keeps physical state, statistical models, research representations, experiments, evidence, and execution infrastructure distinct. Their interfaces should make scientific assumptions visible rather than burying them in shared mutable state or orchestration code.

Generic numerical infrastructure should be reused where mature implementations exist. Climate-owned engineering effort should concentrate on climate-specific variables, coordinates, operators, balance laws, coupling semantics, representation maps, Jacobians, physical constraints, and scientifically meaningful experiments.

## Research workflow

New scientific ideas should enter as precise mathematical or physical questions, not as implementation-first feature requests. A useful sequence is:

1. define the climate object or process being represented;
2. state the mathematical structure and its domain of applicability;
3. identify established methods and libraries that already solve generic parts of the problem;
4. implement only the climate-specific or independently valuable reference pieces locally;
5. compare candidate structures on systems where the relevant truth is known;
6. evaluate them on climate data or climate-model output only after the representation and numerical semantics are trustworthy.

Reproducibility, ablation, controls, and evidence capture support this work, but they are infrastructure for answering the scientific questions rather than the subject of the research itself.

## Key documents

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — target architecture and semantic boundaries.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — live scientific and engineering obligation graph.
- [`docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md`](docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md) — integrated representation and geometry research program.
- [`docs/FORTRAN-PHYSICS-FRONTIER.md`](docs/FORTRAN-PHYSICS-FRONTIER.md) — realized and open physical/numerical responsibilities.
- [`docs/FORTRAN-TIME-INTEGRATION.md`](docs/FORTRAN-TIME-INTEGRATION.md) — integration semantics and external numerical-library boundaries.
- [`docs/GEOMETRY-VERIFICATION.md`](docs/GEOMETRY-VERIFICATION.md) — mathematical and numerical verification requirements for geometry.
- [`docs/SHEAF-REALIZATION.md`](docs/SHEAF-REALIZATION.md) — sheaf/descent/cohomology realization program.
- [`docs/META-EXPERIMENTATION.md`](docs/META-EXPERIMENTATION.md) — methodology for comparing competing scientific representations and methods.
- [`docs/generated/STATUS.md`](docs/generated/STATUS.md) — generated repository status from machine authorities.
- [`AGENTS.md`](AGENTS.md) — binding repository correctness and contribution rules.

## Scope

Climate is not an operational forecasting system and does not treat experimental mathematical indicators as validated physical predictions. Its purpose is to build and test a technically serious mathematical and physical research framework in which unconventional ideas can be integrated where they add real explanatory or predictive structure and rejected where they do not.