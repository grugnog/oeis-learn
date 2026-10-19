"""Qualification-only bounded native bridge; never used by production callers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import pytest


@pytest.fixture
def synthesis_driver():
    path = Path(os.environ.get("OEIS_SYNTHESIS_DRIVER", "/opt/oeis-build-info/synthesis-core-test-driver"))
    wasm = Path(os.environ.get("OEIS_BIGINT_WASM", "/opt/oeis-build-info/bigint-dependency-smoke.wasm"))
    if not path.is_file() or not wasm.is_file():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("mandatory synthesis qualification driver/artifact missing")
        pytest.skip("requires pinned native qualification image")
    from oeis_learn.synthesis.admission import reference_identity
    def run(operation, source, indices, values=(), **kwargs):
        raw = json.dumps(dict(operation=operation, source=source, indices=list(indices),
                              values=list(values), reference_sha256=reference_identity(), **kwargs)).encode()
        assert len(raw) <= 4*1024**2
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            proc = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            try:
                proc.stdin.write(raw); proc.stdin.close()
                deadline = time.monotonic() + (40 if operation == "execute" else 8)
                while proc.poll() is None:
                    assert time.monotonic() < deadline, "native diagnostic deadline"
                    status = Path(f"/proc/{proc.pid}/status")
                    if status.exists():
                        for line in status.read_text().splitlines():
                            if line.startswith("VmRSS:"):
                                assert int(line.split()[1]) <= 1024**2, "native diagnostic RSS cap"
                    assert stdout.tell() <= 4*1024**2 and stderr.tell() <= 65536, "diagnostic output cap"
                    time.sleep(0.02)
                stdout.seek(0); stderr.seek(0)
                assert proc.returncode == 0, stderr.read(65536).decode()
                return json.loads(stdout.read(4*1024**2+1))
            finally:
                if proc.poll() is None: proc.kill()
                proc.wait(timeout=5)
    run.runtime_sha256 = "sha256:" + hashlib.sha256(wasm.read_bytes()).hexdigest()
    return run
