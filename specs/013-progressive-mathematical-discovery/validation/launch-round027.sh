#!/usr/bin/env bash
# Fetch and invoke this script from the repository root with its immutable commit.
set -euo pipefail
old=c7bb2aef73340036319c2566675509cf8e2d06bf
new=$(git rev-parse --verify "${1:?Pass the published successor commit}^{commit}")
here=$(git rev-parse HEAD)
if [[ "$here" != "$old" && "$here" != "$new" ]]; then
    echo "Expected the round026 checkout or the pinned round027 revision." >&2
    exit 1
fi
git diff --quiet && git diff --cached --quiet || { echo 'Preserve tracked edits before switching.' >&2; exit 1; }
if [[ "$here" != "$new" ]]; then
    for pair in \
        'b476ebaa904cd27b202ed6705e38d507c7d6b75135cb8231da46fa1e917e9cfd qualification' \
        '0ddeefde97583dffa17bb337cafa5947862e6fe4ec843ed40cc32509384650bf gpu'; do
        read -r id kind <<< "$pair"
        tag="localhost/oeis-learn:round026-$kind"
        if podman image exists "$tag"; then
            test "$(podman image inspect --format '{{.Id}}' "$tag")" = "$id"
        else
            podman tag "$id" "$tag"
        fi
    done
    git switch --detach "$new"
fi
PYTHONPATH=src python3 -BS - <<'PY'
import os
import runpy
from pathlib import Path
from oeis_learn.pipeline_validation import progressive_round025_successor as recovery
from oeis_learn.pipeline_validation.progressive_workstation_state import read
root=Path.cwd()
# Run the two actual packaging assertions with the host standard library.
checks = runpy.run_path(str(root/'tests/contract/test_progressive_packaging.py'))
checks['test_cpu_inventory_covers_new_tests_without_legacy_name_collision']()
checks['test_diagnostic_templates_ship_identically_and_all_feature_tests_are_registered']()
print("CPU test registration and diagnostic packaging passed.", flush=True)
old=root/recovery.SUCCESSOR
evidence=recovery.retry_proof(root,read(old/'preparation-owner.json')['receipt'],read(old/'session.json'),read(old/'journal.json'))
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
fresh=not (root/recovery.RETRY/'session.json').exists()
# Preparation used roughly 98 GiB of local session data in round025.
# Concurrent cleanup during the old telemetry interval makes the net change misleading.
required=preflight['required_bytes']
print(f"Round025 freeze required {required/2**30:.1f} GiB. Available now {size/2**30:.1f} GiB.",flush=True)
if fresh:
    print(f"Allow room for qualification and timing before freeze; the prior preparation occupied about 98 GiB.",flush=True)
if not (root/recovery.RETRY/'prepare-research/frozen/campaign.json').exists():
    if size<required or fs.f_favail<preflight['required_inodes']:
        raise SystemExit(f"Before starting, free {max(0,required-size):,} more bytes and {max(0,preflight['required_inodes']-fs.f_favail):,} more inodes; then repeat this command.")
PY
exec python3 -BS scripts/discovery_hip_handoff.py \
    --engine podman --session run-state/discovery/013-research-round-027 prepare-research \
    --registration-session run-state/discovery/013-research-inputs-003 \
    --rehearsal-session run-state/discovery/013-hip-007 \
    --reuse-preparation-data run-state/discovery/013-research-round-002 \
    --reuse-calibration-session run-state/discovery/013-research-round-009 \
    --expanded-runtime-budgets --streamlined-research --run-after-prepare
