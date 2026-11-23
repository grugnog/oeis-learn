#!/usr/bin/env bash
# Fetch and invoke this script from the repository root with its immutable commit.
set -euo pipefail
old=fe7b05ffb098e17ebd5588d8677f9460a2f6b910
new=$(git rev-parse --verify "${1:?Pass the published successor commit}^{commit}")
here=$(git rev-parse HEAD)
if [[ "$here" != "$old" && "$here" != "$new" ]]; then
    echo "Expected the round025 checkout or the pinned round026 revision." >&2
    exit 1
fi
git diff --quiet && git diff --cached --quiet || { echo 'Preserve tracked edits before switching.' >&2; exit 1; }
if [[ "$here" != "$new" ]]; then
    for pair in \
        'cb431ee3bb0f273463baa167be8eb6e726bce24bb5f701b557523f41a3250ff4 qualification' \
        '7f518bbdf6a57d6d6c62f81335fac87f45d40bff30ddd599d8bdb766197eaf15 gpu'; do
        read -r id kind <<< "$pair"
        tag="localhost/oeis-learn:round025-$kind"
        if podman image exists "$tag"; then
            test "$(podman image inspect --format '{{.Id}}' "$tag")" = "$id"
        else
            podman tag "$id" "$tag"
        fi
    done
    git switch --detach "$new"
fi
PYTHONPATH=src python3 -BS - <<'PY'
import json, os
from pathlib import Path
from oeis_learn.pipeline_validation import progressive_round025_successor as recovery
from oeis_learn.pipeline_validation.progressive_workstation_state import read
root=Path.cwd()
old=root/recovery.SESSION
evidence=recovery.proof(root,read(old/'preparation-owner.json')['receipt'],read(old/'session.json'),read(old/'journal.json'))
print(f"Verified {len(evidence['original_outputs'])} original capability results and {len(evidence['charges'])} committed debits.",flush=True)
# Check round009 source migration before spending time on image builds.
from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.pipeline_validation.progressive_handoff import original_authority
from oeis_learn.pipeline_validation.progressive_calibration_equivalence import validate
from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file, _json
donor='run-state/discovery/013-research-round-009/prepare-research/qualification/qualification.json'
q, folder=original_authority(root, reference(root, donor))
validate(_json(checked_file(folder,q['source_closure'])), collect_snapshot(root))
preflight=read(root/recovery.BASE/'storage-preflight.json')
fs=os.statvfs(root)
size=fs.f_bavail*fs.f_frsize
fresh=not (root/recovery.SUCCESSOR/'session.json').exists()
# Preparation used roughly 98 GiB of local session data in round025.
# Concurrent cleanup during the old telemetry interval makes the net change misleading.
required=preflight['required_bytes']
print(f"Round025 freeze required {required/2**30:.1f} GiB. Available now {size/2**30:.1f} GiB.",flush=True)
if fresh:
    print(f"Allow room for qualification and timing before freeze; the prior preparation occupied about 98 GiB.",flush=True)
if not (root/recovery.SUCCESSOR/'prepare-research/frozen/campaign.json').exists():
    if size<required or fs.f_favail<preflight['required_inodes']:
        raise SystemExit(f"Before starting, free {max(0,required-size):,} more bytes and {max(0,preflight['required_inodes']-fs.f_favail):,} more inodes; then repeat this command.")
PY
exec python3 -BS scripts/discovery_hip_handoff.py \
    --engine podman --session run-state/discovery/013-research-round-026 prepare-research \
    --registration-session run-state/discovery/013-research-inputs-003 \
    --rehearsal-session run-state/discovery/013-hip-007 \
    --reuse-preparation-data run-state/discovery/013-research-round-002 \
    --reuse-calibration-session run-state/discovery/013-research-round-009 \
    --expanded-runtime-budgets --streamlined-research --run-after-prepare
