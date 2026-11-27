#!/usr/bin/env bash
# Called from the workstation repository with the immutable recovery revision.
set -euo pipefail
old=f44099d50b4076d520616431ee8485b27c808c15
new=$(git rev-parse --verify "${1:?Pass the published recovery commit}^{commit}")
previous=609dc06f138c4df3cfb4627e03d9a8a832639254
here=$(git rev-parse HEAD)
git diff --quiet
git diff --cached --quiet
spec=specs/013-progressive-mathematical-discovery/validation
tools=run-state/discovery-tools
mkdir -p "$tools"
if [[ "$here" == "$old" ]]; then
    helper="$tools/rehearse-round023-generator.py"
    tmp=$(mktemp "$tools/round023-generator.XXXXXXXX")
    trap 'rm -f -- "$tmp"' EXIT
    git show "$new:$spec/rehearse-round023-generator.py" > "$tmp"
    if [[ -e "$helper" ]]; then
        cmp "$helper" "$tmp"
    else
        mv "$tmp" "$helper"
    fi
    python3 -S "$helper"
elif [[ "$here" == "$previous" && "$here" != "$new" ]]; then
    if [[ -e run-state/discovery/013-research-round-024/session.json ]]; then
        echo "Existing round024 has a source binding. Preserve that session for review." >&2
        exit 1
    fi
elif [[ "$here" != "$new" ]]; then
    echo "Expected the qualified round023 checkout or the pinned round024 revision." >&2
    exit 1
fi
test -s "$tools/round023-generator-rehearsal/result.json"
if [[ "$here" != "$new" ]]; then
    for pair in \
        '73fd6d9b23574d1beacb8183c0601c94a98508d045309c5bef4097747c36adae qualification' \
        '2da2c7286dd64a5f6a9b760752fb8ae95729991cb6e5f12dace72ae67a4fbe30 gpu'; do
        read -r id kind <<< "$pair"
        tag="localhost/oeis-learn:round023-$kind"
        if podman image exists "$tag"; then
            test "$(podman image inspect --format '{{.Id}}' "$tag")" = "$id"
        else
            podman tag "$id" "$tag"
        fi
    done
    git switch --detach "$new"
fi
PYTHONPATH=src python3 -S - <<'PY'
from pathlib import Path
from oeis_learn.pipeline_validation import progressive_round023_successor as recovery
from oeis_learn.pipeline_validation.progressive_workstation_state import read
root=Path.cwd()
old=root/recovery.SESSION
approval=recovery.proof(root, read(old/'preparation-owner.json')['receipt'],
                        read(old/'session.json'), read(old/'journal.json'))
for name, size in approval['comparison_projection']['estimates'].items():
    print(f'{name}: {size/2**20:.1f} MiB estimated, 512 MiB reader limit', flush=True)
# Estimate capacity before assigning the new owner or rebuilding images.
original=read(root/recovery.BASE/'storage-preflight.json')
estimate=original['required_bytes']+sum(recovery.additions(root,approval).values())
import json
with (old/'telemetry.jsonl').open() as stream:
    first=json.loads(next(stream))
setup_drop=max(0,first['free_disk_bytes']-original['available_bytes'])
fresh=not (root/recovery.SUCCESSOR/'session.json').exists()
target=estimate+(setup_drop if fresh else 0)
from shutil import disk_usage
free=disk_usage(root).free
print(f'Round024 estimated freeze reservation {estimate/2**30:.1f} GiB; free now {free/2**30:.1f} GiB.',flush=True)
if fresh:
    print(f'Round023 used {setup_drop/2**30:.1f} GiB of filesystem headroom before freeze. Starting target {target/2**30:.1f} GiB.',flush=True)
if free<target and not (root/recovery.SUCCESSOR/'prepare-research/frozen/campaign.json').is_file():
    raise SystemExit(f'Free at least {(target-free)/2**30:.1f} GiB more before preparation. Rerun this launcher.')
PY
exec python3 -S scripts/discovery_hip_handoff.py \
    --engine podman --session run-state/discovery/013-research-round-024 prepare-research \
    --registration-session run-state/discovery/013-research-inputs-003 \
    --rehearsal-session run-state/discovery/013-hip-007 \
    --reuse-preparation-data run-state/discovery/013-research-round-002 \
    --reuse-calibration-session run-state/discovery/013-research-round-009 \
    --expanded-runtime-budgets --streamlined-research --run-after-prepare
