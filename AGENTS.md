# Climate worker instructions

Climate is an experimental climate-methods repository. Treat implementation, numerical verification, empirical validation, and scientific claim promotion as distinct.

## Start here

Before substantial work:

1. Resolve the current `main` commit.
2. Read this file and `docs/generated/STATE.md` at that commit.
3. If `main` moved since your previous orientation, inspect the intervening commits before reusing prior conclusions.
4. Read the focused contract for the work you are touching.
5. Check exact-head GitHub Actions when execution status matters.

Repository-local research state is generated into `docs/generated/STATE.md`. Planning authority is `architecture/planning_graph.json`; `docs/ROADMAP.md` is only its projection.

## Worker invariants

- **Write directly to `main` in small coherent commits.** Do not create branches or rewrite history. Before each write, re-read the current `main` version of every file you will modify.
- **Fail closed on required semantics.** Missing data, configuration, calibration, solver/backend, precision, runtime, preprocessing, or resource identity must not silently select a replacement while preserving success semantics. The binding source guard is `architecture/check_semantic_defaults.py`.
- **Do not shadow external owners.** Reuse maintained numerical, climate-data, evaluation, training, inference, model-runtime, and intercomparison systems through thin explicit seams. Climate-owned code should add Climate-specific semantics, boundary checks, or independently valuable references.
- **Do not inflate evidence.** Running is not verification; verification is not empirical validation; evidence from one model/data class does not transfer to another without an explicit scientific relation and test.
- **Planning lives only in the planning graph.** Do not add prose TODO queues, continuation notes, priority lists, or hand-maintained status snapshots.
- **Prefer decisive experiments over discretionary framework work.** When a preregistered R1/R2 experiment or native integration is scientifically ready and its resource is available, run the smallest decisive test before unrelated refinement.
- **Durable prose states contracts and facts, not prestige or edit history.** Use commit history for change narration. `architecture/check_documentation_quality.py` and `architecture/check_durable_text.py` enforce this boundary.
- **Unavailable resources remain unavailable.** Compilation, mocks, source inspection, or local imitation do not substitute for CUDA hardware, native external runtimes, or declared large-data evidence.

## Change-quality gates

- **Preserve semantics before simplifying.** A cleanup, consolidation, abstraction, rename, deletion, or schema reduction must first identify the distinctions, authorities, failure states, provenance, and evidence boundaries it could erase. "Simpler" or "cleaner" is not sufficient justification.
- **Prove equivalence or retire semantics explicitly.** If a change removes a state, field, path, type, authority surface, or execution branch, preserve its meaning with an equivalent representation and witness, or retire it through the owning contract/planning authority. Do not silently collapse distinctions.
- **Migrate semantic dependents together.** A change to an authority, schema, identity model, or generated state surface is incomplete until its renderers, guards, tests, workflows, and affected durable docs agree. Do not leave mixed old/new contracts on `main`.
- **Require a negative witness for collapsed distinctions.** When a refactor reduces representational structure, retain or add a machine check proving that the previously invalid conflation still fails.
- **Keep scope locked.** Do not fold adjacent scientific or architectural work into the current change unless it is required for correctness or the declared completion criteria. Independent obligations stay independent.
- **Do not claim completion before exact-head verification.** If a required check is failing, skipped because of an earlier failure, or still pending, describe the change as unverified/in progress rather than current, complete, or green.

## Focused contracts

- Architecture and ownership boundaries: `docs/ARCHITECTURE.md`
- Verification, validation, evidence, and claim promotion: `docs/VALIDATION-AND-EVIDENCE.md`
- Experiment and method comparison design: `docs/META-EXPERIMENTATION.md`
- Execution resources and evidence boundaries: `docs/EXECUTION-RESOURCES.md`
- GPU work: `docs/GPU-ENGINEERING.md`
- External/Commons integration: `docs/COMMONS-INTEGRATION.md`
- Semantic substitution rules: `docs/SEMANTIC-SANITATION.md`
- Legacy depletion/removal: `docs/LEGACY-DELETION.md`

Method-specific specifications remain authoritative for their own mathematical or physical semantics.

## Required repository checks

Run the checks relevant to the changed surface. The architecture baseline is:

```text
python -m unittest discover -s tests/architecture -v
python architecture/check_repository_state.py
python architecture/check_claims.py
python architecture/check_methods.py
python architecture/check_modules.py
python architecture/check_planning.py
python architecture/check_experiments.py
python architecture/check_semantic_defaults.py
python architecture/check_documentation_quality.py
python architecture/check_durable_text.py
python architecture/render_state.py --check
python architecture/render_roadmap.py --check
```

Use the language-, method-, resource-, and experiment-specific workflows required by the focused contract. A green check proves only what that check actually exercises.
