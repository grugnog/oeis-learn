"""Exercise the public host CLI and its receipts against a real local engine."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from oeis_learn.container_runtime import resolve_engine
from oeis_learn.fresh.common import digest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.docker
def test_public_cpu_validation_and_verified_rerun():
    image = os.environ.get("OEIS_ROLE_TEST_IMAGE")
    if not image:
        pytest.skip("explicit native image and real engine required")
    engine = resolve_engine()
    evidence = ROOT / "tmp-engine-evidence/public-cli"
    evidence.mkdir(parents=True, exist_ok=True)
    output = evidence / "validation"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/foundation_run.py"),
        "--engine",
        engine.name,
        "--image",
        image,
        "validate",
        "--suite",
        "cpu",
        "--output",
        str(output),
    ]
    token = digest([str(ROOT), str(output), "validation"])[7:39]

    def run(name):
        stdout, stderr = evidence / (name + ".stdout.log"), evidence / (name + ".stderr.log")
        with stdout.open("wb") as out, stderr.open("wb") as err:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, timeout=3900)
        if result.returncode:
            # Keep failures diagnosable directly from CI, without downloading
            # an artifact or publishing large checkpoint files.
            for path in [stdout, stderr, *sorted(output.glob("*/*.log"))]:
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - (32 << 10)))
                    print(f"\n==> {path} <==\n" + stream.read().decode(errors="replace"))
            pytest.fail(f"public CLI failed; full logs retained under {evidence}")
        return stdout.read_text()

    try:
        first = run("initial")
        assert "RUN environment" in first and "RUN cpu-regressions" in first
        receipts = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*/receipt.json")}
        assert len(receipts) == 2
        assert all(json.loads(payload)["status"] == "PASS" for payload, _ in receipts.values())
        summary = json.loads((output / "validation.json").read_bytes())
        assert summary["results"]["cpu-regressions"]["counts"]["tests"] > 0
        assert summary["qualified"] is False

        repeated = run("repeat")
        assert "SKIP environment: verified prior success" in repeated
        assert "SKIP cpu-regressions: verified prior success" in repeated
        assert "RUN " not in repeated
        assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in receipts.items())
    finally:
        # Also reclaim the actual worker if the test's outer deadline fires.
        engine.cleanup(token)
