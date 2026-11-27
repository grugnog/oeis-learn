#!/usr/bin/env bash
# Extract only the registered host controller and the reviewed evidence reader.
set -euo pipefail
rev=${1:?supply the full follow-up commit}
shift
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full Git commit is required.' >&2; exit 2; }
root=$(git rev-parse --show-toplevel)
cd "$root"
if [[ "$(git rev-parse HEAD)" != 77667e67176ae32fa16457d52565bae1dbe41afc ]]; then
    echo 'Run from the qualified detached round027 checkout at 77667e67176ae32fa16457d52565bae1dbe41afc.' >&2
    exit 2
fi
validation=specs/013-progressive-mathematical-discovery/validation
mkdir -p run-state/discovery-tools
helpers=$(mktemp -d "run-state/discovery-tools/trial00-${rev:0:12}-XXXXXX")
git archive "$rev" "$validation/trial00-followup" "$validation/audit-round027-screening.py" \
    src/oeis_learn/generalization/progressive/trial_evidence.py | tar -x -C "$helpers"
exec python3 -BS "$helpers/$validation/trial00-followup/launch.py" "$@"
