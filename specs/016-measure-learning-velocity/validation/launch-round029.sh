#!/usr/bin/env bash
set -euo pipefail
rev=${1:?full candidate commit required}
action=${2:?round029 action required}
[[ $rev =~ ^[0-9a-f]{40}$ ]]
case "$action" in
  preflight|status) [[ $# == 2 ]]; extra=() ;;
  start-comparative) [[ $# == 4 && $3 =~ ^[1-9][0-9]*$ && $4 =~ ^sha256:[0-9a-f]{64}$ ]] ;;
  verify) extra=()
    if [[ $# == 3 && $3 =~ ^[0-9a-f]{40}$ ]]; then extra=(--resume-initialization "$3")
    else [[ $# == 2 ]]; fi ;;
  qualify|cpu|run) [[ $# -ge 3 && $3 == /* && -f $3 ]]; extra=(--authority "$3")
    if [[ $# == 4 && $action == qualify && $4 =~ ^[0-9a-f]{40}$ ]]; then
      extra+=(--resume-initialization "$4")
    else [[ $# == 3 ]]; fi ;;
  *) exit 2 ;;
esac
workspace=$(pwd -P)
if [[ $(git rev-parse --show-toplevel) != "$workspace" ||
      $(git rev-parse HEAD) != 77667e67176ae32fa16457d52565bae1dbe41afc ]] ||
   git symbolic-ref -q HEAD >/dev/null; then
  echo "Run from the physical root of the detached round027 donor" >&2
  exit 2
fi
[[ $(git rev-parse --verify "${rev}^{commit}") == "$rev" ]]
helper=specs/015-learned-program-search/validation/candidate_source.py
source_root=$(git show "${rev}:${helper}" | python3 -I -B - "$workspace" "$rev")
if [[ $action == start-comparative ]]; then
  grant=$(python3 -I -S -B - "$source_root" "$workspace" "$3" "$4" <<'PY'
from pathlib import Path
import sys
source, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(source / "src"))
from oeis_learn.pipeline_validation.progressive_velocity_wave import authorize
print(authorize(root, source, int(sys.argv[3]), sys.argv[4]))
PY
  )
  echo "Comparative authority: $grant" >&2
  action=run
  extra=(--authority "$grant")
fi
exec python3 -S -B "$source_root/scripts/foundation_run.py" --engine podman discovery round029 \
  --workspace "$workspace" --action "$action" "${extra[@]}"
