#!/usr/bin/env bash
# Fetch and invoke this script from the repository root with its immutable commit.
set -euo pipefail
old=8c2f04cb74819e6b7adf1b8bbfd5262e3936bef5
new=$(git rev-parse --verify "${1:?Pass the published successor commit}^{commit}")
here=$(git rev-parse HEAD)
if [[ "$here" != "$old" && "$here" != "$new" ]]; then
    echo "Expected the round024 checkout or the pinned round025 revision." >&2
    exit 1
fi
git diff --quiet && git diff --cached --quiet || { echo 'Preserve tracked edits before switching.' >&2; exit 1; }
if [[ "$here" != "$new" ]]; then
    for pair in \
        'c2c4489192ab7c75ebf264cd3ea7e7e6ad6fe96ead02594b7b259de8a10118d7 qualification' \
        '170651870873699712db2b922123fc7a91dda1c9357cea17225b58e0a39eab6c gpu'; do
        read -r id kind <<< "$pair"
        tag="localhost/oeis-learn:round024-$kind"
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
from oeis_learn.pipeline_validation import progressive_round024_successor as recovery
from oeis_learn.pipeline_validation.progressive_workstation_state import read
root=Path.cwd()
old=root/recovery.SESSION
evidence=recovery.proof(root,read(old/'preparation-owner.json')['receipt'],read(old/'session.json'),read(old/'journal.json'))
print(f"Verified {len(evidence['original_outputs'])} original results, round024 debits and the HIP rehearsal.",flush=True)
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
with (old/'telemetry.jsonl').open() as stream:
    first=json.loads(next(stream))
headroom=max(0,first['free_disk_bytes']-preflight['available_bytes']) if fresh else 0
required=preflight['required_bytes']
print(f"Round024 freeze required {required/2**30:.1f} GiB. Available now {size/2**30:.1f} GiB.",flush=True)
if fresh:
    print(f"Including the observed setup drop, the estimated starting target is {(required+headroom)/2**30:.1f} GiB.",flush=True)
if not (root/recovery.SUCCESSOR/'prepare-research/frozen/campaign.json').exists():
    if size<required or fs.f_favail<preflight['required_inodes']:
        raise SystemExit(f"Before starting, free {max(0,required-size):,} more bytes and {max(0,preflight['required_inodes']-fs.f_favail):,} more inodes; then repeat this command.")
PY
exec python3 -BS scripts/discovery_hip_handoff.py \
    --engine podman --session run-state/discovery/013-research-round-025 prepare-research \
    --registration-session run-state/discovery/013-research-inputs-003 \
    --rehearsal-session run-state/discovery/013-hip-007 \
    --reuse-preparation-data run-state/discovery/013-research-round-002 \
    --reuse-calibration-session run-state/discovery/013-research-round-009 \
    --expanded-runtime-budgets --streamlined-research --run-after-prepare
