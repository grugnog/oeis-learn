# Round016 freeze repair

Input: `prepare-research(20261116-191111).zip`, supplied 2026-11-16.

The journal records panel calibration rebind and failed-publication archival as
COMPLETE. Canonical prepare completed in 3015.9 seconds. Freeze failed after
1654.0 seconds with `INVALID / CONTRACT_MISMATCH` and producer error
`ValueError: artifact inventory cap`. Acceptance and readiness remain blocked.
No science or successful freeze is claimed.

The original prepared document is 57,486,558 bytes, SHA-256
`17ede5d64864faa71d40305efc50cfd3f62cdfce63abe19128f0dbb9f4aea72f`.
It contains 9179 references totaling 511,408,016 bytes. This is not a reference
count failure: `verify_refs` uses that error for its cumulative byte limit.

Two reproducible defects were corrected:

1. Study-policy discovery applies 32 MiB to every policy, including the original
   48,779,754-byte calibration input. The scanner now uses the existing 128 MiB
   campaign-document bound and retains 32 MiB for the matching study. Hashes,
   duplicate detection and closed policy validation remain enforced.
2. Freeze recognizes per-task storage only for resource-policy v3, while prepare
   also uses v4. Both now share exact v3/v4 recognition. The integration regression
   uses a 512 MiB retained checkpoint and 160 MiB task-specific states, reproducing
   the false `task_checkpoint_storage:first` deficit before the fix. Canonical
   replay, campaign construction and missing/duplicate/altered storage-record
   rejection run without mocking the storage decision. Only original device,
   checkpoint measurement and trajectory measurements are simulated.

Validation: 143 distinct CPU checks passed; one native-image-dependent check was skipped.
No HIP evidence is claimed. Calibration compatibility now explicitly permits
this host freeze check change while continuing to require fresh qualification.
Old calibration receipts retain original identities; neither measurement code
nor original debit is changed.

The uploaded ZIP does not contain the 1563-byte original traceback, whose recorded
SHA-256 is `841cfa0738de721ef5956102e54904addae93424708bb6a0247e1b02d4327f89`.
The user subsequently supplied `host-traceback.txt`. Its length and hash match
that reference exactly. It confirms `freeze -> validate_studies -> policy ->
verify_refs`, specifically the 32 MiB inventory scan; the diagnosis is confirmed.

## Fresh-source continuation

T227 adds a narrow v8 preparation-owner proof for this terminated host exception.
It checks the original prepare publication, traceback frames/hash, command,
process termination, stdout result, closed allowance and unscored panel report.
Active work, published campaigns, acceptance, scientific execution, changed
training support and unrelated exceptions remain ineligible. The old owner is
fenced under the existing locks; originals and debits remain in history.

The successor uses the existing canonical pipeline. It adopts the original
census and training support, selects the successful corrected panels from the
owner proof, and retains the original round009 calibration donor through that
proof chain. Panel adoption also survives an intervening qualification failure.
Current-source images/base qualification/model qualification and training-only
timing run anew. Canonical preparation authenticates and reuses the exact
calibration receipts, retains their original reservation, then freezes and runs
campaign-bound acceptance before research. No tiny rehearsal or 1296-case
calibration rerun is requested. No assessment command is invoked.

The round016 v2 external helper must not be used with the updated checkout. Use
a fresh session, with the original flags and data/calibration donors:

```bash
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman \
  --session run-state/discovery/013-research-round-018 \
  prepare-research \
  --registration-session run-state/discovery/013-research-inputs-003 \
  --rehearsal-session run-state/discovery/013-hip-007 \
  --reuse-preparation-data run-state/discovery/013-research-round-002 \
  --reuse-calibration-session run-state/discovery/013-research-round-009 \
  --expanded-runtime-budgets \
  --streamlined-research \
  --run-after-prepare
```

Fetch the corrected branch and pin the supplied handoff commit before running.
Keep that checkout fixed while the successor is active. Repeating the same
command retains completed journal steps and canonical resume behavior. An
unknown failure remains visible, not automatically retried as a success.

The previous observed durations were 46.2 minutes base qualification, 16.3 model
qualification, 16.3 evaluation timing, 25.0 trajectory timing, 4.7 construction,
50.3 preparation replay and 27.6 until freeze failed. A rough successor setup
estimate is 3–5 hours including successful freeze/acceptance; the existing
larger operational allowances remain unchanged. This is not a new training ETA.
Diagnostics continue under the successor's `share/` directory, including the
owner proof, original failure frame identities and `panel-data-reuse.json`.

T227 validation: 19 recovery checks cover real hashes, file publication,
ownership transfer/fencing, bounded original failure evidence, changed/active/
exposed work rejection, corrected-material adoption, calibration-donor ancestry,
repeated adoption, intervening failures and normal failure/success ZIP behavior.
Historical Q/timing boundaries are explicitly simulated. Broader reuse and
canonical calibration/freeze integration tests pass locally; actual new-source
HIP qualification and campaign acceptance remain external T220.


## Round017 preflight stop / T228

The workstation's round017 attempt stopped with `data producer changed; fresh
review required` for `progressive_research_census.py` and
`progressive_research_materials.py`. This guard precedes ownership transfer,
images and qualification. The earlier recovery test used unchanged producer
closures and missed the actual transition.

The complete diff from the original qualified 2faa8e43 versions to 4af2358:

- Census grouping includes every relevant non-parameter composition identity,
  including unseen-structure rows, when reserving whole components.
- Material projection invokes the canonical frozen-panel validation before
  publication.

Native generation and training support are unchanged by these edits. Original
panels must nevertheless be replaced by the already authenticated successful
outcome-blind revision. T228 therefore permits only the exact two reviewed hash
pairs, checks that revision before ownership transfer, and requires its adoption
after transfer. The old uncorrected panels cannot pass via this exception.
All unrelated source guards and current-source qualification stay in force.

Validation: 118 tests passed, including the canonical adoption path with actual
producer hashes, repeated adoption, owner fencing, unknown producer/native
changes, absent correction evidence and mandatory test registration. Python 3.13
`-S` accepts the complete round018 handoff in dry-run. No hardware qualification
is claimed. The command above now names round018; round017's source-bound
preflight diagnostics remain retained, with no expensive work to repeat.
