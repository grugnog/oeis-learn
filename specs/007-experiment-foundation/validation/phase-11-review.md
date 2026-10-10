# Phase 11 review and remediation

Scope: T103–T105 on merged phase 10 (`686a51659b9e7eefc445c60a068032b8f290db69`).
Only phase 11 is being advanced. Prior incomplete GPU/source/trained-model tasks
remain half-checked. This review does not declare full spec 007 complete.

## Findings and changes

| Finding | Remediation and evidence |
| --- | --- |
| Readiness accepted a 68-entry list containing duplicated, unknown or incompletely mapped findings. | `require_dispositions` binds the original file, current package, actual task checklist and unique exact inventory; regressions reject duplicates, changed source/task hashes, missing mappings, false statuses/rationales and a full-release claim. |
| README claimed unmeasured performance, corpus coverage and universal synthesis/proof soundness. | Replaced with current capabilities, evidence links and explicit acceptance limits. |
| Operational docs described obsolete image paths, planned CLI aliases and incomplete orchestration. | Reconciled profiles/roles, root Docker targets, actual named commands, migration, full-state recovery, finite certificate scope and conditional decisions. User choices remain separate from planning defaults. |
| Handoff lacked copied dependency replay and immutable identities. | New public `foundation handoff` inventories bounded hashed files, rejects unsafe/symlink/duplicate/stale/quota inputs, copies into a staged archive, validates copied dependencies, fsyncs and publishes atomically. Archive completion and release qualification are separate. |
| Documentation slices had no executable driver. | `verify_phase11.py` runs the actual full-model CPU three-update pause/resume, development/finalization/final evaluation, census and measurement slices. It archives failures and prerequisite exclusions honestly. Linux CI executes it without weakening boot/deadline guards. |
| Clean-wheel gate had no current cross-cutting archive. | Installed acceptance exercises packaged governance and handoff. CI archives/replays real G7/G8/G9, six-case wheel and host Docker integration evidence in the matching qualification runtime. Selection follows declared report dependencies, excluding unrelated sampler trials without increasing caps. |

## Validation

Focused local regression after closure remediation: **27 passed, 3 explicit native/gate capability skips**.
The new handoff suite contributes 24 passing tests, including dependency-closure selection with 4,100 unrelated trials and missing/escaping dependencies. The six initially failing
count-only reconciliation regressions passed after remediation; later cases cover
forged task markers, altered checklist identity and an unauthorized release claim.
Compilation and `ruff check --select F,E9` pass for the new modules/drivers/tests.

Local broad CPU execution, Spec Kit checks, packaging round trip and current CI
results are recorded in [implementation-results.md](implementation-results.md).
The Linux/native/wheel workflows are the executable authority for capabilities
unavailable in this workspace. CI artifacts bind the exact tested commit; earlier
phase 10 runs remain historical and are not presented as validation of changed code.

## Semantic design review

The review compares all eight stories, FR-001–038, SC-001–014, 108 tasks,
constitution v2 adoption, contracts and 68 original findings. All 52 requirements
have task coverage; every task has a mapped requirement. Structural coverage does
not establish runtime acceptance. No new constitution conflict was found in this
phase: exact independent arithmetic, strict provenance, visible/private separation,
full continuation, finite proof limits and fail-closed device evidence remain.

| ID | Severity | Finding / disposition |
| --- | --- | --- |
| P11-D1 | Medium, repaired | Operational CLI/build documentation drift; current interfaces now replace planned aliases and deleted paths. Historical design defaults remain identified. |
| P11-D2 | High, repaired | Count-only source finding reconciliation could falsely imply completeness; original inventory and actual task state are now checked. |
| P11-D3 | High, pending acceptance | G0–G6 device/containment, G8 hardware and G10 real-source/live measurements cannot be established by fixtures. Owning tasks and T104/T105 remain `[/]`. |

No unmapped requirements/tasks or new duplicated requirements were found. Custom
requirements/design checklists remain reviewer-owned and untouched. Next action
is the existing workstation/source acceptance, with no automatic experiment or
LODA adoption inferred from structural checks.

## Remaining acceptance

- Actual Ryzen/HIP strict/qualified updates, preflight and host GPU containment.
- Real indexed release source, authoritative metadata and nonempty structural archive.
- Actual trained historical-wrapper/overflow strata and synchronized telemetry.
- Frozen engineering/conditional pilot outcomes within existing two/six-hour caps.

These stay in 007. Only native LODA runtime, paired language comparison and
conditional transcoding questions transfer to 008 after applicable foundation
gates pass. See [handoff.md](handoff.md) and [phase-11-dispositions.json](phase-11-dispositions.json).
