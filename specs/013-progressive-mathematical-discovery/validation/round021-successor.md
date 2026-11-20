# Round020 HIP rehearsal and round021 launch

Round020 froze 352 tasks and validated 15,556 trial choices. Its software suite
completed. Three HIP fixtures used a different base recipe from the selected
learner. Two of those cycles also under-reserved space for the selected weights
and their checkpoint projection. The fourth failing fixture specified the
registered integer zero dropout as a floating-point value.

Run the pinned rehearsal command supplied with the PR commit from the original
round020 checkout. It verifies the journal and frozen output files, uses the
original qualified GPU image and installed wheel, and copies only the two
reviewed test files into an isolated diagnostic test tree. Four HIP cases
exercise the previously failing scenarios. Results and bounded logs are in
`run-state/discovery-tools/round020-hip-rehearsal/diagnostics.zip`. A failed
attempt retains its pytest working files. A successful attempt retains the
results and removes its own temporary pytest data.

## After the rehearsal succeeds

The following command reads the exact tested revision from the rehearsal,
checks all four case outcomes and the container's termination, and tags the two
round020 images before building their successors. Run it from Owen's workstation.
The original directory layout resolves the donor sessions' stored paths.

```bash
(
  set -eu
  cd ~/workspace/oies-learn
  test "$(git rev-parse HEAD)" = f08c2d16943bda3774385589e3b40ae65859ec5a
  revision=$(python3 -S - <<'PY'
import json, re
from pathlib import Path
p = Path('run-state/discovery-tools/round020-hip-rehearsal')
def read(name):
    return json.loads((p / name).read_bytes())
cases = read('case-outcomes.json')
process, cleanup, scope = (read(n) for n in ('process.json', 'cleanup.json', 'scope.json'))
if (len(cases) != 4 or any(c['status'] != 'PASS' for c in cases)
        or process['outcome'] != 'COMPLETE' or process['returncode'] != 0 or process['cleanup_failure']
        or cleanup['acknowledged'] is not True or cleanup['within_bound'] is not True or cleanup['failure']
        or scope['scope'] != 'DIAGNOSTIC_ONLY' or not re.fullmatch('[a-f0-9]{40}', scope['revision'])):
    raise SystemExit('Inspect the original rehearsal results before starting round021.')
print(scope['revision'])
PY
  )
  git diff --quiet
  git diff --cached --quiet
  podman tag 65dfe03e3f4c8fde89b4a83c65f0a5ada510e1ad773545966bf5cf13e4772af7 localhost/oeis-learn:round020-qualification
  podman tag 7c70998d3bc90d58c0f026cbc89e177b62d4ae81091cb0fcadbf452262bb2227 localhost/oeis-learn:round020-gpu
  git switch --detach "$revision"
  python3 -S scripts/discovery_hip_handoff.py \
    --engine podman --session run-state/discovery/013-research-round-021 \
    prepare-research \
    --registration-session run-state/discovery/013-research-inputs-003 \
    --rehearsal-session run-state/discovery/013-hip-007 \
    --reuse-preparation-data run-state/discovery/013-research-round-002 \
    --reuse-calibration-session run-state/discovery/013-research-round-009 \
    --expanded-runtime-budgets --streamlined-research --run-after-prepare
)
```

To resume round021 on the same pinned checkout, repeat the Python launcher above
with the same arguments. Its new qualification and freeze cover the corrected
source. The capacity preflight uses round021's own prepared graph. Freeze saves
validated choices. Acceptance keeps attempts in separate directories under a
single campaign allowance. The launcher rechecks an already published receipt
and clean process termination on recovery. Unproven cleanup blocks another
device attempt. Unknown interrupted intervals consume their full reservation.

Round020's source, images, calibration lineage, corrected panels, task hashes
and failure receipts remain identifiable. Round021 charges round020's measured
time and reserves the rehearsal ceiling from the original setup budget.
Automatic research requires a successful software suite, all seven campaign HIP
cases, and readiness. No scientific training was executed by the rehearsal.
