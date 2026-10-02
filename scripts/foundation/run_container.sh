#!/usr/bin/env bash
# Host Docker CLI only. The existing root Dockerfile remains the sole image build.
set -euo pipefail
exec python -m oeis_learn.tracking.foundation_container "$@"
