# Climate

Climate is a research codebase for testing climate-physics operators, numerical methods, and alternative representations of climate state under explicit experiment and evidence contracts.

The repository is not an operational forecasting system, a validated unified climate model, or evidence that every registered method is scientifically useful. Registration, execution, numerical verification, empirical validation, and claim promotion are separate states.

## What the repository is for

The code and experiments address questions such as:

- whether discrete climate-physics operators preserve the balances, signs, constraints, and exchanges they are intended to represent;
- whether alternative state representations retain information needed for a declared task better than simpler baselines;
- whether statistical geometry, multiview structure, local-to-global consistency, or other mathematical constructions add measurable value under controlled comparisons;
- whether numerical pathologies such as rank deficiency, poor conditioning, backend differences, or precision changes alter a scientific result;
- how results from external climate-data, model, evaluation, and intercomparison systems can enter a Climate experiment without duplicating the systems that own those capabilities.

These are questions to test. The repository does not assume that any particular mathematical family, representation, or numerical method is useful before evidence supports that conclusion.

## Implementation and evidence are separate

Climate distinguishes several roles that are easy to conflate:

- **reference implementations** provide independently checkable definitions or numerical oracles;
- **canonical implementations** are the repository-owned implementations of narrowly declared objects;
- **ExperimentSpecs** freeze comparisons, data identities, controls, metrics, seeds, and stopping rules;
- **evaluation records** preserve outcomes of declared experiments or bounded exploratory work;
- **claim and evidence registries** control whether an outcome changes the maturity of a scientific claim;
- **execution receipts and local verification** establish what ran for a particular revision and environment.

A canonical implementation is not an endorsement of the scientific method it implements. A passing test does not establish empirical climate validity. A successful synthetic or idealized-model experiment does not automatically transfer to a GCM, reanalysis, observation, or another model class.

The binding maturity and evidence semantics are defined in [docs/VALIDATION-AND-EVIDENCE.md](docs/VALIDATION-AND-EVIDENCE.md); worker instructions are in [AGENTS.md](AGENTS.md). Current repository-local research state is generated in [docs/generated/STATE.md](docs/generated/STATE.md).

## Scientific programs

### Climate dynamics and numerical structure

The physical/numerical work isolates operators and update rules so their declared equations and budgets can be checked independently. Relevant subjects include transport, hydrostatics, forcing response, reservoir exchange, conservation and budget accounting, dissipation signs, time integration, conditioning, and backend/precision behavior.

Small local implementations are appropriate when they define Climate-specific semantics or serve as independent references. Generic numerical algorithms should normally remain owned by maintained numerical libraries.

### Multiple representations of climate state

Climate also compares representations of the same underlying system under explicit tasks and controls. Candidate views may include physical coordinates, modal or spectral summaries, statistical or likelihood coordinates, multiview latent structure, local compatibility data, learned representations, or relational/hierarchical descriptions.

A representation is not selected because it has attractive geometric language. It has to preserve or expose a declared target better than appropriate baselines under the same data and evaluation policy.

The broader research design is described in [docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md](docs/MULTIREPRESENTATION-CLIMATE-MANIFOLD.md) and [docs/META-EXPERIMENTATION.md](docs/META-EXPERIMENTATION.md).

## External systems remain external owners

Climate should not copy mature climate-data, preprocessing, evaluation, training, inference, model-execution, or intercomparison stacks merely to place project-local APIs in front of them.

When an experiment depends on an external system, Climate's responsibility is to bind the exact upstream capability and identity, preserve the configuration and transformations that affect the result, capture native receipts or artifacts, add Climate-specific scientific semantics, and fail closed when the required external path is unavailable.

This boundary is described in [docs/COMMONS-INTEGRATION.md](docs/COMMONS-INTEGRATION.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and the contributor rules in [AGENTS.md](AGENTS.md).

## Experiments and execution

Portable repository checks include:

```text
python -m unittest discover -s tests/architecture -v
python architecture/check_repository_state.py
python architecture/check_claims.py
python architecture/check_methods.py
python architecture/check_modules.py
python architecture/check_planning.py
python architecture/check_experiments.py
python architecture/render_roadmap.py --check
python architecture/render_state.py --check
```

Language-specific references, canonical Rust/Fortran surfaces, mathematical oracles, and registered experiment paths are invoked explicitly from the checkout being evaluated. Hosted CI is intentionally absent: repository pushes must not trigger compute, and `.github/workflows/` is prohibited by the repository-state guard.

### Codespaces

The repository includes a dev-container and a campaign runner for portable CPU experiments:

```text
python architecture/codespaces_campaign.py --list
python architecture/codespaces_campaign.py
```

The runner derives supported experiment identities from existing ExperimentSpecs and the canonical runtime registry. It does not create a second experiment list. Codespaces is an execution venue for work that fits the declared portable CPU/toolchain resource classes; it does not stand in for CUDA hardware, native external runtimes, or large-data confirmation.

Execution-resource semantics are defined in [docs/EXECUTION-RESOURCES.md](docs/EXECUTION-RESOURCES.md).

## Where to look

- [AGENTS.md](AGENTS.md) — worker orientation and repository-editing rules.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — intended architecture and authority boundaries.
- [architecture/planning_graph.json](architecture/planning_graph.json) — sole planning authority.
- [docs/ROADMAP.md](docs/ROADMAP.md) — generated planning-only projection.
- [docs/generated/STATE.md](docs/generated/STATE.md) — generated repository-local research state.
- [experiments/](experiments/) — registered ExperimentSpecs.
- [evaluations/](evaluations/) — committed evaluation and exploratory result records.
- [claims/registry.json](claims/registry.json) and [evidence/registry.json](evidence/registry.json) — scientific claim/evidence authority.
- [methods/registry.json](methods/registry.json) — executable method identities and assumptions.
- [docs/VALIDATION-AND-EVIDENCE.md](docs/VALIDATION-AND-EVIDENCE.md) — verification, validation, and promotion semantics.
- [docs/META-EXPERIMENTATION.md](docs/META-EXPERIMENTATION.md) — comparison and experiment-design rules.
- [docs/EXECUTION-RESOURCES.md](docs/EXECUTION-RESOURCES.md) — resource and execution-environment semantics.

## Nonclaims

The repository does not claim that unusual mathematics is useful because it is unusual, that conventional methods are sufficient because they are conventional, that a canonical implementation is empirically correct, that a passing verification check validates a climate hypothesis, or that one model class can substitute for another without an explicit scientific relation and test.
