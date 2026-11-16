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
Thus the first defect is an inevitable failure for this inventory, but its exact
position as the first actual exception should be confirmed from the workstation:

```bash
cd ~/workspace/oies-learn &&
cat run-state/discovery/013-research-round-016/prepare-research/frozen/private/host-traceback.txt
```

Do not pull into the pinned checkout or rerun the old v2 helper as a repair.
These changes affect qualified installed code; an updated-source qualification
and correctly rebound continuation are still required. Completed preparation,
original calibration and original qualification remain retained historical
evidence. No automatic qualification transfer or completed hardware recovery is
asserted by this software repair.
