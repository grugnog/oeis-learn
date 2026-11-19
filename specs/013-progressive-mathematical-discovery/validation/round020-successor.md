# Round020 workstation handoff

Round019's final freeze stopped in task-template budget validation after
39,668 seconds. Its terminal receipt, traceback, clean exit and closed allowance
are in the shared prepare-research bundle. The reviewed transfer authenticates
that specific session and its predecessors under the existing ownership locks.
It checks the workstation originals before fencing round019. New qualification
runs in round020. The original calibration and corrected panels keep their
existing hashes, scientific budgets and provenance.

Keep a detached source worktree for b616b116. Use the original workstation path
for the successor checkout so historical references to absolute command paths
continue to resolve. Preserve both existing Podman image IDs with round019 tags
before building new images. Keep historical sessions and helper copies intact.
The successor's ordinary launcher is:

```sh
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman --session run-state/discovery/013-research-round-020 \
  prepare-research \
  --registration-session run-state/discovery/013-research-inputs-003 \
  --rehearsal-session run-state/discovery/013-hip-007 \
  --reuse-preparation-data run-state/discovery/013-research-round-002 \
  --reuse-calibration-session run-state/discovery/013-research-round-009 \
  --expanded-runtime-budgets --streamlined-research --run-after-prepare
```

Round020's setup budget carries forward 127,684 seconds of earlier preparation
and freeze charges (rounded up). This leaves 54,116 seconds for preparation and
86,400 seconds for freeze, including supervision. These are ceilings. Before
launching a freeze attempt the supervisor checks the current filesystem against
the successor's prepared graph and writes `prepare-research/storage-preflight.json`.
A capacity block reports the byte and inode deficits. After restoring space,
repeat the same launcher on the same source and session. Completed preparation,
qualification and committed freeze choices are reusable. An unknown interrupted
supervisor interval is charged under the existing allowance policy.

The supervisor packages `share/prepare-research.zip` under the round020 session.
Acceptance and readiness must complete before automatic research. Keep
`prepare-research/frozen.freeze-progress/` for validation and publication
recovery. Round019's unpublished trial checks predate checkpoint support.
