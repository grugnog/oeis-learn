#!/usr/bin/env bash
# From the detached donor: one authorized, disposable HIP rehearsal and its status.
set -euo pipefail
rev=${1:?full implementation commit}
minutes=${2:?approved maximum workstation minutes for learner qualification}
[[ $# == 2 && $rev =~ ^[0-9a-f]{40}$ && $minutes =~ ^[1-9][0-9]*$ ]]
workspace=$(pwd -P)
[[ $(git rev-parse --show-toplevel) == "$workspace" ]]
[[ $(git rev-parse HEAD) == 77667e67176ae32fa16457d52565bae1dbe41afc ]]
if git symbolic-ref -q HEAD >/dev/null; then exit 2; fi
[[ $(git rev-parse --verify "${rev}^{commit}") == "$rev" ]]
helper=specs/015-learned-program-search/validation/candidate_source.py
candidate=$(git show "${rev}:$helper" | python3 -I -B - "$workspace" "$rev")
launcher="$candidate/specs/016-measure-learning-velocity/validation/launch-round029.sh"
bash "$launcher" "$rev" preflight
grant=$(python3 -I -S -B - "$candidate" "$workspace" "$minutes" "$rev" <<'PY'
from pathlib import Path
import sys
source, workspace, minutes, revision = sys.argv[1:]
sys.path.insert(0, str(Path(source) / "src"))
from oeis_learn.pipeline_validation.progressive_velocity_qualification import authorize_learner_check
print(authorize_learner_check(Path(workspace), int(minutes), revision))
PY
)
code=0
bash "$launcher" "$rev" qualify "$grant" || code=$?
bash "$launcher" "$rev" status
exit "$code"
