#!/usr/bin/env bash
# Fetch and invoke this script from the repository root with its immutable commit.
set -euo pipefail
if (( $# )) && [[ $1 == --round026-space ]]; then
    shift
    # Maintenance exits here, before the normal checkout, image and launch operations.
    exec env PYTHONPATH=src python3 -BS - "$@" <<'PY'
import argparse
from collections import Counter
import fcntl
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import stat

from oeis_learn.pipeline_validation import progressive_round025_successor as recovery
from oeis_learn.pipeline_validation.identity import confined
from oeis_learn.pipeline_validation.progressive_workstation_state import read

parser = argparse.ArgumentParser(prog="launch-round027.sh --round026-space", allow_abbrev=False)
parser.add_argument("--apply", action="store_true", help="discard recorded qualification test scratch")
args = parser.parse_args()
root = Path.cwd().resolve()
stages = Path("prepare-research/base-qualification/pipeline/stages")

def handoff(folder):
    proof = recovery.registration_failure(
        root, read(folder / "preparation-owner.json")["receipt"], read(folder / "session.json"),
        read(folder / "journal.json"), read(folder / "preparation-owner-proof.json"))
    if proof != read(confined(root, Path(recovery.RETRY) / "preparation-owner-proof.json")):
        raise ValueError("the round026 failure evidence differs from the handoff saved by round027")

def mountpoints():
    def unescape(path):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), path)
    return {Path(unescape(line.split()[4])) for line in Path("/proc/self/mountinfo").read_text().splitlines()}

def scratch_roots(folder, mounts):
    found = {}
    nested = {m for m in mounts if m != folder and m.is_relative_to(folder)}
    for path in sorted((folder / stages).glob("*/*/.runtime")):
        if path.resolve() != path or not path.is_dir():
            continue
        if any(m.is_relative_to(path) or path.is_relative_to(m) for m in nested):
            continue
        command = path.parent / "command.json"
        if not command.is_file() or command.is_symlink():
            continue
        try:
            record = json.loads(command.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        argv = record.get("argv") if isinstance(record, dict) else None
        if not isinstance(argv, list):
            continue
        bind = "type=bind,src=" + str(path.parent) + ",dst=/work/output"
        if all(flag in argv for flag in (bind, "TMPDIR=/work/output/.runtime/tmp", "--basetemp=/work/output/.runtime/pytest")):
            info = path.stat()
            found[path] = (info.st_dev, info.st_ino)
    return found

def gib(size):
    return f"{size / (1 << 30):,.3f} GiB"

def inventory(folder, scratch, mounts):
    retained, temporary = Counter(), Counter()
    largest, linked = [], {}
    single_link = 0

    def account(path, area, size, links):
        if area is not None:
            temporary[area] += size
        else:
            rel = path.relative_to(folder)
            depth = len(stages.parts) + 2 if rel.is_relative_to(stages) else 2
            retained[Path(*rel.parts[:depth])] += size
            heapq.heappush(largest, (size, str(rel), links))
            if len(largest) > 15:
                heapq.heappop(largest)

    def failed(exc):
        raise exc

    for directory, dirs, files in os.walk(folder, onerror=failed, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(d for d in dirs if parent / d not in mounts and not (parent / d).is_symlink())
        area = next((p for p in scratch if parent.is_relative_to(p)), None)
        for name in files:
            path = parent / name
            if path in mounts:
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            size = info.st_blocks * 512
            if info.st_nlink == 1:
                account(path, area, size, 1)
                if area is not None:
                    single_link += size
            else:
                # Attribute an inode shared between scratch and retained data to its retained path.
                key = (info.st_dev, info.st_ino)
                if key not in linked or (linked[key][1] is not None and area is None):
                    linked[key] = (path, area, size, info.st_nlink)
    for entry in linked.values():
        account(*entry)
    print("Allocated regular-file blocks in round026:", gib(sum(temporary.values()) + sum(retained.values())))
    print("Eligible scratch, single-link regular-file blocks:", gib(single_link))
    for path in scratch:
        print(f"  {gib(temporary[path]):>13}  {path.relative_to(folder)}")
    print("Largest retained areas, excluding eligible scratch:")
    for name, size in retained.most_common(15):
        print(f"  {gib(size):>13}  {name}")
    print("Largest retained files (allocated size, link count, relative path):")
    for size, name, links in sorted(largest, reverse=True):
        print(f"  {gib(size):>13}  {links:3d}  {name}")

try:
    folder = confined(root, recovery.SUCCESSOR)
    with confined(folder, "session.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mounts = mountpoints()
        scratch = scratch_roots(folder, mounts)
        inventory(folder, scratch, mounts)
        print("Filesystem available:", gib(shutil.disk_usage(folder).free), flush=True)
        handoff(folder)
        print("Verified round026 evidence against round027.")
        if args.apply and scratch:
            if not shutil.rmtree.avoids_symlink_attacks or scratch_roots(folder, mountpoints()) != scratch:
                raise ValueError("scratch directories changed or safe directory removal is unavailable")
            for path in scratch:
                shutil.rmtree(path)
                print("Removed qualification scratch:", path.relative_to(folder), flush=True)
            handoff(folder)
            print("Verified the handoff after cleanup. Filesystem available:", gib(shutil.disk_usage(folder).free))
        elif args.apply:
            print("No eligible round026 qualification scratch remains.")
except (ValueError, KeyError, OSError) as exc:
    parser.exit(2, f"Round026 storage check stopped: {exc}\n")
PY
fi
if (( $# != 1 )) || [[ $1 == -* ]]; then
    echo 'Usage: launch-round027.sh COMMIT | --round026-space [--apply]' >&2
    exit 2
fi
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
