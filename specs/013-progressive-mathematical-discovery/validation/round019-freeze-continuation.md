# Round019 freeze continuation

The 2026-11-17 14:04:48 diagnostic bundle records successful base/model
qualification, timing and preparation. Freeze exhausted its 7,080-second external
execution allowance (7,200 including cleanup), returned 130 with KeyboardInterrupt,
and had no cleanup failure. Acceptance and readiness were blocked. The private
1,813-byte host traceback was excluded from the uploaded ZIP, so its exact last
frame has not been reviewed. Large variant expansion is a plausible contributor;
this evidence does not establish either a deadlock or an exact completion ETA.

Keep workstation source pinned at b616b116dad59a05ad1911687583d8ab1626af13.
Fetch this helper and copy it into run-state; do not pull new source into that
checkout. `continue-round019.py` verifies HEAD, the exact qualified source closure,
active preparation ownership, completed output hashes, qualified image identities,
original allowance binding, exact failed result and clean timeout receipts.

The helper preserves the failed publication and journal, then retries only canonical
freeze. It allocates at most 24 hours from unused original preparation time, after
charging the previous freeze. Scientific graph, settings, token budgets and
calibration remain unchanged. The durable allowance retains spent time and never
refunds an unknown interval. A new failure stops for review; the helper cannot
silently retry it. Completed freeze/acceptance steps are reused on reinvocation.
Canonical acceptance, readiness and run/resume follow; final assessment remains
separate. Existing image evidence is reused without rebuilding.

The supervisor prints elapsed/remaining time every 30 seconds. CPython faulthandler
writes all-thread stacks every five minutes to
`prepare-research/freeze-continuation/logs/0000/stderr.log`. This is a liveness and
debugging aid, not percentage completion. The original and any new bounded host
traceback are copied into public diagnostic logs; no private datasets or weights
are copied. Normal and failure paths use the existing session ZIP packaging under
`share/`. The original frozen failure remains under
`prepare-research/freeze-continuation/original-failed-freeze/`.

## Checkpoints in the updated freeze validator

The qualified round019 checkout at `b616b116` predates these checkpoints. Its
failure receipts and stack traces cannot reconstruct completed trial checks.
Keep that checkout and its images intact. Using the revised validator requires
an explicit source and qualification transition.

For a freeze started with the revised validator, keep both the output directory
and its sibling `<output>.freeze-progress/`. Retry through the same workstation
session with the same arguments. The supervisor retains the original time
allowance and separate attempt logs. An unknown supervisor interruption can use
up the remaining allowance even though validation checkpoints survive.

- `freeze-progress.json` reports the phase and counts for the current attempt.
  The private SQLite database commits each fully validated study training or
  confirmation choice immediately. On retry, freeze checks the current request,
  channel and package bytes before reusing that choice. Other validation stages
  run again if the validator had not yet returned successfully.
- Once validation succeeds, `private/validated-result.json` saves the campaign,
  runtime and ordered task identities before public files are written. If
  publication stops, freeze can reconstruct those exact task files and check
  live capacity without repeating scientific validation. A finished staged
  publication can also be promoted after interruption.
- A successful freeze publishes `result.json`, `campaign.json`, `runtime.json`,
  `tasks/` and `freeze-publication.json` together. Reopening it verifies the
  manifest and roster. Campaign acceptance remains a separate prerequisite for
  research. The private checkpoint files stay on the workstation.
