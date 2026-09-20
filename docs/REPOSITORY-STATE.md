# Repository state and freshness contract

> Generated from `architecture/state_authorities.json`. Do not hand-edit this file.
> State-authority manifest fingerprint: `sha256:fc7fa5f03a5bfd5a690109f3519201de9e75c48ef3e8474231b23ba4fb200833`.

Repository state is **commit-scoped**. A statement about what is planned, realized, evaluated, evidenced, or passing is valid only for the exact resolved `main` commit on which its authorities were read. If `main` moves, cached present-state conclusions are stale until the reorientation sequence below is repeated.

A generated projection being fresh means only that its checked-in bytes match its **declared authority inputs**. It does not mean that the projection is a complete description of repository state, and it does not establish exact-head CI success.

## State surfaces

| Surface | Kind | Authorities | Projection | Scope | Explicit exclusions |
| --- | --- | --- | --- | --- | --- |
| `contract` | `durable_contract` | `AGENTS.md`<br>`docs/ARCHITECTURE.md` | — | Binding contributor semantics and intended architectural contracts. These documents may describe intended structure that is not yet realized. | — |
| `planning` | `machine_authority` | `architecture/planning_graph.json` | `docs/ROADMAP.md` | Planned work, priority, dependencies, blockers, resource class, completion criteria, and planning evidence paths only. | structural realization; scientific evaluation outcomes; claim-evidence promotion; exact-head CI state; commit history |
| `structural_realization` | `machine_authority_set` | `architecture/modules/*.json`<br>`claims/registry.json`<br>`evidence/registry.json`<br>`experiments/*.json`<br>`methods/sheaf-realization.v1.json` | `docs/generated/STATUS.md` | Registered module structure, declared claim maturity/evidence links, ExperimentSpec registration, and sheaf realization authority represented by these exact inputs. | planning graph state; repository evaluation records not promoted through evidence/claim authorities; exact-head CI state; commit history; unregistered implementation facts |
| `scientific_evaluations` | `repository_records` | `evaluations/**` | — | Committed evaluation and exploratory result records. Evaluation records do not become claim evidence merely by existing. | — |
| `execution` | `external_exact_head` | `GitHub Actions workflow runs keyed by exact commit SHA` | — | Execution/CI outcomes for the exact repository revision. A run for another SHA does not establish the current head's execution state. | — |
| `history` | `git_history` | `Git commit history` | — | Change narration and the delta between a previously known revision and current main. | — |

## Mandatory reorientation sequence

Perform this sequence before making a present-state assertion, declaring an obligation done/ready/blocked, selecting the next repository action, or resuming substantial work from an earlier conversational state:

1. Resolve the exact current main commit SHA.
2. Read AGENTS.md and docs/REPOSITORY-STATE.md from that exact SHA.
3. If the resolved SHA differs from the previously known SHA, inspect the intervening commits before reusing any prior repository-state conclusion.
4. Read architecture/planning_graph.json and its docs/ROADMAP.md projection from the same SHA when selecting or describing planned work.
5. Read docs/generated/STATUS.md and the relevant structural machine authorities from the same SHA when describing realized repository structure.
6. Read relevant committed evaluations and evidence/claim records before describing scientific outcomes or promotion state.
7. Resolve GitHub Actions for the exact SHA whenever build, test, experiment, or execution state matters.
8. Only after these steps make a present-state assertion, declare work done/blocked/ready, or select the next repository action.

## Generated-projection semantics

`docs/ROADMAP.md` is a projection of the planning authority only. Its freshness says nothing about implementation, evaluations, evidence promotion, or CI.

`docs/generated/STATUS.md` is a projection of the structural-realization authority set declared in the state-authority manifest. It is intentionally incomplete with respect to planning, committed evaluation records that have not been promoted through evidence authorities, exact-head GitHub Actions, commit history, and unregistered implementation facts.

Therefore, neither projection may be cited alone as proof of the complete current repository state. Present-state claims require reconciliation across the relevant surfaces above at one exact commit.

## Verification commands

```text
python architecture/check_repository_state.py
python architecture/render_repository_state.py --check
python architecture/render_roadmap.py --check
python architecture/render_status.py --check
```
