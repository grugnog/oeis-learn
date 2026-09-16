# 007 validation report — expanded scope

**Date**: 2026-09-16. **Result**: Document, integration, schema, task and input-traceability checks pass. Runtime implementation, actual Ryzen evidence and constitution adoption remain pending.

## Workflow and executed checks

Official Specify CLI/Codex skills remain **1.0.7**, pinned to `fe1d00e3ccaf495880aaf90fb0e17679e82f065b`. No upstream skill/template was modified to make a check pass. Used `speckit-plan` for the expanded design, `speckit-tasks` to regenerate the105-task story-organized backlog from the resolved official template, and read-only `speckit-analyze` for consistency. No extension hooks are configured.

Current exact commands, UTC timestamps, exit codes and outputs are in [expanded-commands.json](expanded-commands.json). [commands.json](commands.json) preserves the earlier installation/narrow-design checks as history; its old counts are not the current coverage result.

| Executed check | Result | Scope |
| --- | --- | --- |
| `specify version` | Exit0,1.0.7 | Installed upstream identity |
| `specify integration status` | Exit0,OK | Codex-managed integration integrity |
| `setup-plan.sh --json` | Exit0 | Correct feature paths; existing plan preserved |
| `setup-tasks.sh --json` | Exit0 | Resolved official template and available design artifacts; actual task generation followed the installed skill |
| `check-prerequisites.sh --json --require-spec --require-tasks --include-tasks` | Exit0 | Required feature artifacts discoverable |
| `validate_artifacts.py` | Exit0 | 38FR+14SC,105 tasks/eight stories, acyclic dependencies, local links,68 input dispositions, only LODA deferrals and baseline JSON schema fixtures |
| `git diff --check` | Exit0 | Tracked whitespace/error checks |
| Read-only semantic review | Complete | [analysis.md](analysis.md), retaining GOV-001 |

The baseline schema still passes three positive and14 negative fixtures. Qualified extension records have normative field/behavior tables and explicit implementation-time validator/test tasks; these document checks do not claim those future application validators already exist.

## Completeness and limits

The expanded [review map](../review-coverage.md) resolves all six input-audit gaps at design/task level. Known solver, prover, optimizer, native-runtime, RL, curriculum, replay and discovery defects require owning-path repairs and executable regressions. Packaging/CI, dataset policies, long-program retention and resource summaries are explicit. KV caching is included. Conditional position/capacity/precision/search experiments have concrete bounded protocols; they may retain the baseline with evidence. Only LODA-specific work is deferred.

Internal requirement coverage is **52/52**; external input dispositions are **68/68**. These are different checks. All105 implementation tasks remain unchecked. Story counts: setup/foundation8; US1=11, US2=10, US3=8, US4=12, US5=12, US6=18, US7=13, US8=10; final cross-cutting3.

**GOV-001 remains a CRITICAL activation gate:** constitution2.0.0 is proposed, not ratified. T001 gathers actual workstation evidence and T002 records adoption before further implementation promotion. This revision does not fabricate approval.

No application runtime, helper, trainer, CI job or test implementation was changed or executed in this documentation update. No actual GPU probe, clean-wheel/native application suite, correctness benchmark or training experiment was run. The two-hour engineering and six-hour diagnostic/pilot caps are planning defaults for implementation, not spent time or measured results. Full007 completion requires actual G0–G10 evidence, not document scripts alone.
