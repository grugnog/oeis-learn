#!/usr/bin/env bash
set -euo pipefail
rev=${1:?full candidate commit required}
action=${2:?round029 action required}
[[ $rev =~ ^[0-9a-f]{40}$ ]]
case "$action" in
  preflight|status|verify) [[ $# == 2 ]]; extra=() ;;
  qualify|cpu|run) [[ $# == 3 && $3 == /* && -f $3 ]]; extra=(--authority "$3") ;;
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
exec python3 -S -B "$source_root/scripts/foundation_run.py" --engine podman discovery round029 \
  --workspace "$workspace" --action "$action" "${extra[@]}"
