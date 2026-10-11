# Spec 007 phase 11 evidence handoff

The implementation handoff is reviewable, but full 007 release authorization is
pending. T103 documentation is complete; T104/T105 remain `[/]` until the actual
GPU/source/trained-model and bounded live acceptance below is supplied. Earlier
incomplete tasks retain their original partial markers. Archive completion,
checkpoint lifecycle and structural/checklist validation are not gate passes.

## Immutable evidence and replay

[Phase 11 dispositions](phase-11-dispositions.json) reconcile exactly 68 original
inputs (R01–20, A01–19, P01–17, D01–12). They bind the SHA-256 of the original
`review-inputs.json`, the actual `tasks.md` bytes and the current package Python
implementation. Each original title/design disposition, task mapping/marker and
nonempty rationale is checked. Prior phase 10 records retain their original code
identity; their results are historical rather than rebound to this phase.

`foundation handoff --manifest FILE --output NEW_DIR --json` publishes
`inputs.json`, copied `evidence/` and `handoff.json`. The latter binds the input
manifest, file/byte counts, current implementation, freshly replayed G0–G10,
missing gates, and LODA-only deferred scope. `archive_is_authorization` is false.
Use `python -m oeis_learn.evaluation.handoff --archive DIR` in the pinned
qualification runtime. Tampering, missing dependencies and changed implementation
reject. Identical retries replay; changed inputs cannot overwrite the archive.

CI's `archive_phase11.py` selects the owning reports and their declared dependency
closure, excluding unrelated sampler trials, installed environments and prior archives. It replays copied gate
JUnit/native dependencies, the six installed-wheel cases, phase 10/11 and host
Docker integration reports. Assertions require current G7/G9 and G8 software
passes, preserve G8 hardware pending, and prevent full-release authorization.
CI publishes the complete selected gate archive, including its actual native
wheel and resolved image identity, with owning results. The separate CPU
quickstart checkpoint archive is bounded/replayed inside the job; its reports
and slice logs are published without large checkpoint blobs. Original absolute
run-path identities are preserved, not rewritten as portable resume claims.

## Gate status and acceptance ownership

| Gate | Phase 11 evidence / required acceptance | Release status |
| --- | --- | --- |
| G0 | Adopted v2 contract exists; actual device feasibility/preflight bound to the selected run still required. | Pending |
| G1 | Current CPU/native conformance executed by CI; must also bind owning evidence to the accepted baseline checkpoint. | Pending run binding |
| G2 | Visible-only/sealed/private evaluation regressions; actual host Docker roles in CI. | Pending baseline binding |
| G3 | Generic admission/oracle/holdout tests and CPU quickstart pool; real release memberships still required. | Pending release binding |
| G4 | Complete-state CPU recovery/resume tests and quickstart pause/resume; actual baseline binding still required. | Pending baseline binding |
| G5 | Actual HIP updates, allocator/preflight and host-coordinated GPU containment unavailable here. | Pending GPU |
| G6 | Finite-result/scoped-certificate regressions; baseline report must bind executed proof-boundary suites. | Pending baseline binding |
| G7 | Owning repaired solver/prover/optimizer and actual Rust/native qualification in current CI. | PASS |
| G8 | Owning learning/cache/policy/resume software in current CI; actual hardware suite separate. | Software PASS; hardware pending |
| G9 | Logical/streaming/discovery/fresh certificate/native qualification in current CI. | PASS |
| G10 | Six wheel cases, 101 combined integration/Docker cases and 68-input reconciliation PASS; real source audit and qualified measurement decisions missing. | Pending |

Earlier accepted software evidence: phase 10 executable revision
`ba89b9694128775cd136b890f77a249b650a70ae`, CPU/native run
[38088372626](https://github.com/grugnog/oeis-learn/actions/runs/38088372626),
wheel/native/Docker run
[38088372595](https://github.com/grugnog/oeis-learn/actions/runs/38088372595).
Those passed G7/G9 and G8 software, six wheel cases and eight actual Docker role
tests. They are historical records, not authorization for phase 11 changed code.
Current phase 11 tested code is `e8b64943078484f53f7de16ec6fd1aea22910962`:
[CPU/native review](https://github.com/grugnog/oeis-learn/actions/runs/38096255204)
and [installed-wheel/native/Docker](https://github.com/grugnog/oeis-learn/actions/runs/38096255200)
both pass. The package implementation identity is
`sha256:c3253cb1c4ddf4baeab32a6899e422eaa996cbb2dcb44900434b3111b85cda0a`.
Final documentation records these results without changing qualification source
bytes or task markers. Full commands/counts are in
[implementation-results.md](implementation-results.md).

The published
[gate archive](https://github.com/grugnog/oeis-learn/actions/runs/38096255200/artifacts/11686525939)
contains **30 copied files / 41,960,097 bytes**, including the actual native wheel
and resolved image ID. Its input manifest is
`sha256:bd4083ab47c3fbaebb59ed610cde136e933f1d6acdc2e876ed6d13ca475c86e6`;
the freshly replayed report is
`sha256:5a36dbcf9a70d526b127561a824ab977914894a87b9262e9baa28df120a8000a`.
The report preserves G0–G6 pending, G8 hardware pending and G10 source/measurement
pending, with `full_007_complete=false` and `archive_is_authorization=false`.

## Quickstart and experiment decisions

The driver uses exact fixture/profile identities, random full-model weights,
complete manifests/ledger and a genuine visible-only evaluation. Local execution
successfully freezes the fixture cohort and runs measurement dry-run, actual CPU
timings and bounded archive replay. The initial driver supplied malformed census
terms; it now preserves the original indexed records instead. Pool construction fails closed
where Linux `/proc/sys/kernel/random/boot_id` is absent; training/resume and
finalization/evaluation then remain prerequisite failures. The Linux CI driver
at `e8b64943078484f53f7de16ec6fd1aea22910962` executed all 11 slices successfully,
with saved commands/stdout/stderr and no fallback. Its bounded CPU archive also
replayed successfully.

CPU measurement decisions are diagnostic: `qualified=false`, four measured
prompts and a 124-prompt shortfall. All six conditional decisions remain
inconclusive because required diagnostics are unavailable. No GPU throughput, power/thermal,
overflow population, learning improvement or conditional trigger is inferred.
Missing evidence stays unavailable/inconclusive; the baseline is retained. Real
measurement/v2 requires accepted baseline/G7/G8, real release development source,
generic archive, resolved image and the predeclared two/six-hour budgets. The
archive preserves the actual decisions rather than replacing them with an adoption
claim. Real bounded experiments have not been run in this phase.

## Work retained in 007 versus 008

T093, T095–T100 and T102 retain real-source/trained-model/workstation acceptance.
T104/T105 retain the corresponding cross-cutting gate and quickstart acceptance.
Source metadata/audit is a CPU data dependency once supplied; GPU requirements
are separate. Failed/skipped mandatory gates cannot be marked complete by a report.

Only native LODA runtime, paired WAT/LODA training/comparison and conditional
transcoding questions move to spec 008. All applicable 007 gates must pass before
starting those comparisons; no runtime or experiment authorization is added here.
