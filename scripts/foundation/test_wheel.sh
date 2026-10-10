#!/usr/bin/env bash
# Build and install outside checkout; no PYTHONPATH, editable install or source imports.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
parent=${OEIS_WHEEL_WORK_ROOT:-$(dirname "$repo")}
task_root=$(mktemp -d "$parent/installed-wheel-XXXXXX")
# Keep logs and wheel on failure for review; caller chooses artifact retention.
uv build --wheel --out-dir "$task_root/wheels" "$repo"
uv venv --python "$(command -v python)" --no-python-downloads "$task_root/env"
uv pip install --python "$task_root/env/bin/python" 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
uv export --project "$repo" --locked --prune torch --no-emit-project --format requirements.txt --output-file "$task_root/requirements.txt"
uv pip install --python "$task_root/env/bin/python" --require-hashes --no-deps -r "$task_root/requirements.txt"
uv pip install --python "$task_root/env/bin/python" --no-deps "$task_root"/wheels/*.whl
if [[ -n "${OEIS_NATIVE_WHEELS_DIR:-}" ]]; then
  uv pip install --python "$task_root/env/bin/python" --no-deps "$OEIS_NATIVE_WHEELS_DIR"/*.whl
fi
if [[ "${OEIS_REQUIRE_NATIVE:-0}" == 1 ]]; then
  "$task_root/env/bin/python" -I -c 'import oeis_wasm_evaluator; assert oeis_wasm_evaluator.source_identity_bytes()'
fi
mkdir "$task_root/acceptance"
cp "$repo/tests/contract/test_foundation_qualified_cli.py" "$task_root/acceptance/test_installed_cli.py"
cd "$task_root/acceptance"
unset PYTHONPATH
export OEIS_INSTALLED_WHEEL=1
"$task_root/env/bin/python" -I -m pytest -q test_installed_cli.py --junitxml="$task_root/installed-wheel.xml"
"$task_root/env/bin/oeis-learn" foundation --help > "$task_root/installed-help.txt"
printf 'Installed wheel evidence: %s\n' "$task_root"
