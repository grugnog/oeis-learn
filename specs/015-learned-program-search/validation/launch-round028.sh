#!/usr/bin/env bash
set -euo pipefail
rev=${1:?full implementation SHA required}
action=${2:?round028 action required}
[[ $# == 2 && $rev =~ ^[0-9a-f]{40}$ ]]
case "$action" in preflight|audit|status|run|rehearse|verify|collect-audit) ;; *) exit 2 ;; esac
workspace=$(pwd -P)
donor=77667e67176ae32fa16457d52565bae1dbe41afc
if [[ $(git rev-parse --show-toplevel) != "$workspace" || $(git rev-parse HEAD) != "$donor" ]] || git symbolic-ref -q HEAD >/dev/null; then
  echo "Run from the detached round027 donor at $donor" >&2
  exit 2
fi
[[ $(git rev-parse --verify "${rev}^{commit}") == "$rev" ]]
helper=specs/015-learned-program-search/validation/candidate_source.py
source_root=$(git show "${rev}:${helper}" | python3 -I - "$workspace" "$rev")
if [[ $action == collect-audit ]]; then
  exec python3 -I -B "$source_root/specs/015-learned-program-search/validation/collect_audit.py" "$workspace" "$rev"
fi
exec env PYTHONDONTWRITEBYTECODE=1 python3 -B "$source_root/scripts/foundation_run.py" \
  --engine podman discovery round028 --action "$action" --workspace "$workspace"
