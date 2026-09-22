#!/usr/bin/env bash
# Shared project software, not host provisioning. Called as container root.
set -euo pipefail
mode=${1:?cpu or gpu}
case "$mode" in cpu|gpu) ;; *) exit 2 ;; esac
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl build-essential cmake ninja-build pkg-config git
rm -rf /var/lib/apt/lists/*

export RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
export PATH="/opt/cargo/bin:$PATH"
curl --fail --proto '=https' --tlsv1.2 https://sh.rustup.rs -o /tmp/rustup-init.sh
bash /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.90.0
chmod -R a+rX /opt/rustup /opt/cargo

if [[ "$mode" == cpu ]]; then
  # PyO3 0.20 is incompatible with Pi's Python 3.14. Keep Pi itself untouched.
  uv python install 3.12.12
  uv venv --python 3.12.12 --managed-python /opt/oeis-venv
  export VIRTUAL_ENV=/opt/oeis-venv
  export PATH="/opt/oeis-venv/bin:$PATH"
  uv pip install --python "$VIRTUAL_ENV/bin/python" 'torch==2.9.1' \
    --index-url https://download.pytorch.org/whl/cpu
else
  export VIRTUAL_ENV=/opt/venv
  export PATH="/opt/venv/bin:$PATH"
  test -x /opt/venv/bin/python
  python -c 'import torch; assert torch.version.hip; print(torch.__version__, torch.version.hip)' \
    > /tmp/oeis-torch-before
  python -m pip install 'uv==0.8.22'
fi

# Export the committed lock, excluding torch and its exclusive CUDA dependency
# subtree. All other exported packages retain hashes. Never sync generic torch.
uv export --project /opt/oeis-build --locked --prune torch --no-emit-project \
  --format requirements.txt --output-file /tmp/oeis-requirements.txt
uv pip install --python "$VIRTUAL_ENV/bin/python" --require-hashes --no-deps \
  -r /tmp/oeis-requirements.txt
uv pip install --python "$VIRTUAL_ENV/bin/python" 'setuptools==80.9.0' 'wheel==0.45.1'
if [[ "$mode" == gpu ]]; then
  python -c 'import torch; assert torch.version.hip; print(torch.__version__, torch.version.hip)' \
    > /tmp/oeis-torch-after
  cmp /tmp/oeis-torch-before /tmp/oeis-torch-after
fi
uv pip check --python "$VIRTUAL_ENV/bin/python"
chmod -R a+rX "$VIRTUAL_ENV"
