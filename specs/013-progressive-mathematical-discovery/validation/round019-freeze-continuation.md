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
