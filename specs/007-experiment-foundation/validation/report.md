# 007 validation report

**Date**: 2026-09-16. **Result**: Document/installation checks pass. Implementation readiness is conditional on constitution adoption and actual runtime evidence; these are not claimed complete.

## Installed tooling and workflow

- Official Specify CLI **1.0.7**, installed in an isolated Python venv from commit `fe1d00e3ccaf495880aaf90fb0e17679e82f065b` (tag v1.0.7).
- Official generated Codex skills under `.agents/skills/`; managed shared infrastructure updated through the CLI. No skill/template was edited to suppress a rule or make a check pass.
- Existing Copilot integration replaced through official integration commands. An attempted side-by-side installation was rejected by `integration status` as unsafe; the final Codex-only installation passes with **zero modified/missing managed files, invalid manifest paths or unchecked manifests**.
- Followed `speckit-specify`, `speckit-plan`, `speckit-tasks` and `speckit-analyze` instructions, with a separate explicit constitution proposal. Resolved templates, created the local feature pointer and used the official setup scripts. No extensions/hooks were configured before or after the phases.

## Executed checks

Exact commands, exit codes, outputs and UTC timestamps are saved in [commands.json](commands.json). The task template's resolved content is represented there by its hash/length to avoid duplicating the whole upstream template.

| Check | Result | What it establishes |
| --- | --- | --- |
| `specify version` | Exit 0; 1.0.7 | Installed CLI identity. |
| `specify integration status` | Exit 0; OK | Managed skill/scaffold integrity and valid Codex integration. |
| `specify check` | Exit 0 | Environment tool inventory only. The standalone Codex CLI executable and many optional agents are absent here; installed skill files were followed directly in this session. |
| `setup-plan.sh --json` | Exit 0 | Correct007 paths and active plan template; existing plan preserved. |
| `setup-tasks.sh --json` | Exit 0 | Spec/plan present, all four design artifact categories found, task template resolved. |
| `check-prerequisites.sh --json --require-spec --require-tasks --include-tasks` | Exit 0 | Complete feature artifacts discoverable through the official workflow. |
| `validate_artifacts.py` | Exit 0 | 26/26 requirements covered, 52 unique ordered/story-labeled tasks, acyclic dependencies, valid local links, valid Draft2020-12 schema, three positive fixtures and14 rejected negative shapes. |
| `git diff --check` | Exit 0 | No whitespace/error-marker problems in the tracked diff. |
| Read-only semantic analysis | Complete | [Analysis report](analysis.md), with the explicit governance gate below. |

Task distribution: setup/foundation8, US1 execution11, US2 evaluation10, US3 provenance8, US4 operation12, cross-cutting3. All52 implementation checkboxes remain unchecked.

## Meaning and limits

**Subsequent input-coverage audit:** [review-coverage.md](../review-coverage.md) compares all 20 independent recommendations, A01–A19, P01–P17 and the decision session. It finds partial coverage, contained-but-unrepaired defects and an unresolved divergence over immediate false-proof repair. The 26/26 result below is internal coverage of 007 requirements, not complete coverage of those earlier inputs. Document checks passing does not resolve these findings.

The project validator supplements upstream scripts; it is not described as an official Spec Kit validator. Schema fixtures contain synthetic identities and do not establish executed-program correctness, hash integrity or trained-model performance. The read-only analysis checks design consistency and task coverage, not runtime behavior.

**GOV-001 remains a CRITICAL activation gate**: the proposed 2.0.0 constitution is not ratified. T001 gathers actual workstation feasibility evidence; T002 records explicit adoption before subsequent strict implementation promotion. The plan honors this gate; no user approval, maintainer consensus or benchmark has been fabricated.

No application source, arithmetic helper, trainer or evaluation implementation was changed for007. No new foundation runtime tests, Docker/GPU probe, full repository suite or multi-day training was run. Those tests and artifacts are precisely scoped in the implementation tasks and [quickstart](../quickstart.md). Historical known defects therefore remain until the corresponding tasks are implemented.

The deliverable is a complete specification/design/task package with a verified official integration, not a claim that the proposed system already works on Ryzen or that a representation wins the comparison.
