#!/usr/bin/env bash
set -euo pipefail
rev=${1:?full implementation SHA required}
action=${2:?round028 action required}
[[ $rev =~ ^[0-9a-f]{40}$ ]]
if [[ $action == collect-receipts ]]; then
  [[ $# == 3 && $3 =~ ^[0-9a-f]{40}$ ]]
else
  [[ $# == 2 ]]
fi
case "$action" in preflight|audit|status|run|rehearse|verify|panels|timing|finalize|collect-final|collect-timing|collect-panels|collect-audit|collect-rehearsal|collect-receipts) ;; *) exit 2 ;; esac
workspace=$(pwd -P)
donor=77667e67176ae32fa16457d52565bae1dbe41afc
if [[ $(git rev-parse --show-toplevel) != "$workspace" || $(git rev-parse HEAD) != "$donor" ]] || git symbolic-ref -q HEAD >/dev/null; then
  echo "Run from the detached round027 donor at $donor" >&2
  exit 2
fi
[[ $(git rev-parse --verify "${rev}^{commit}") == "$rev" ]]
helper=specs/015-learned-program-search/validation/candidate_source.py
source_root=$(git show "${rev}:${helper}" | python3 -I - "$workspace" "$rev")
if [[ $action == collect-receipts ]]; then
  [[ $(git rev-parse --verify "${3}^{commit}") == "$3" ]]
  exec python3 -I -B "$source_root/specs/015-learned-program-search/validation/collect_audit.py" "$workspace" "$rev" "$action" "$3"
fi
if [[ $action == collect-final || $action == collect-timing || $action == collect-panels || $action == collect-audit || $action == collect-rehearsal ]]; then
  exec python3 -I -B "$source_root/specs/015-learned-program-search/validation/collect_audit.py" "$workspace" "$rev" "$action"
fi
exec env PYTHONDONTWRITEBYTECODE=1 python3 -B "$source_root/scripts/foundation_run.py" \
  --engine podman discovery round028 --action "$action" --workspace "$workspace"
