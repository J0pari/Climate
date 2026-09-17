# Legacy source depletion and deletion policy

Status: **binding repository-structure policy**.

Climate treats non-authoritative monoliths and prototypes as temporary source reservoirs, not as a permanent architecture tier. Git history is the archive for depleted source. The working tree should contain the architecture that contributors are expected to understand, build, test, and extend.

## 1. Governing rule

A legacy source file remains in the working tree only while it contains useful semantics that have not yet been represented elsewhere and those semantics are actively being mined. Once the useful content has been represented by canonical code, a maintained-library boundary, an independent reference, an executable witness, a durable contract, or a planning-graph obligation, the depleted source is deleted.

Do not create a long-lived `legacy/`, `old/`, `archive/`, or compatibility-source tree merely to preserve implementation history. Previous revisions remain available through Git.

## 2. Mining sequence

Before deleting a source reservoir:

1. identify domain-specific equations, contracts, invariants, sign conventions, units, failure semantics, fixtures, and experimentally meaningful hypotheses that are unique to the file;
2. distinguish those from generic numerical machinery, framework glue, arbitrary defaults, plausible-output placeholders, and commodity algorithms better provided by maintained libraries;
3. move reusable semantics into the narrow canonical module, contract, reference, test, or experiment that owns them;
4. put every worthwhile but unrealized obligation in `architecture/planning_graph.json` rather than leaving a TODO in the source;
5. verify that surviving authorities express the intended semantics independently of the legacy implementation;
6. remove build targets, imports, registries, hazard records, compatibility flags, and documentation references whose only purpose was to keep the depleted file live;
7. delete the source file.

Mining preserves meaning, not incidental implementation structure. A local generic optimizer, eigensolver, FFT, ODE solver, remapper, or similar commodity mechanism is not migrated merely because it exists in an old file; the production boundary should name a maintained library and Climate should retain only domain-specific integration or an independently valuable reference oracle.

## 3. Deletion criteria

A source reservoir is deletable when all of the following hold:

- no canonical build, runtime, experiment, or test depends on it;
- no scientifically useful semantic remains available only in that file;
- every useful unfinished responsibility is represented by `architecture/planning_graph.json` with dependencies and completion criteria;
- any mathematical or numerical authority worth retaining has a canonical, reference, or library-backed home with an appropriate witness;
- any still-relevant semantic hazard is attached to surviving source rather than retained solely as archaeology;
- removing the file does not require a weaker replacement, silent fallback, fabricated output, or compatibility shim.

Deletion is preferable to keeping a depleted file for comparison. A historical implementation needed for investigation can be read from the commit that contained it.

## 4. Root layout

The repository root is reserved for top-level manifests, build entrypoints, contributor contracts, and repository-level descriptive files. Scientific implementations belong under their owned package or source tree, such as `src/`, `reference/`, `experiments/`, `formal/`, or another architecture-defined directory.

Root-level `climate_*` prototypes and scientific source files are prohibited. A new scientific source file must enter through an owned package location and the module inventory rather than using the repository root as an integration surface.

## 5. Planning authority

`architecture/planning_graph.json` is the sole repository authority for planned work, priorities, dependencies, blockers, and completion criteria. `docs/ROADMAP.md` is a generated human-readable projection of that graph.

Durable code comments and documentation may explain current contracts, limitations, assumptions, and rationale. They must not maintain parallel TODO lists, future-work queues, priority rankings, or dependency plans. When implementation work remains, create or update the corresponding planning-graph node.

## 6. Hazard and status ledgers

The semantic-hazard ledger describes hazards reachable in live source. It is not a historical defect archive. When a hazardous legacy source is deleted after its useful semantics are mined, the live hazard record is removed as part of the same consolidation boundary; Git history preserves the prior review.

Likewise, the module inventory describes source that exists in the working tree. Deleted reservoirs do not remain registered as fictional modules. Generated status must be refreshed from the surviving registries after deletion.

## 7. Temporary exceptions

A noncanonical source reservoir that cannot yet be deleted must have a specific active planning-graph node describing what remains to be mined and what proves depletion. The exception is about unresolved semantics, not nostalgia, compatibility convenience, or fear of losing history.

A temporary reservoir must not become a production dependency, scientific authority, or excuse to duplicate configuration, constants, solvers, or planning information.
