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
    path = Path(
        os.environ.get("OEIS_SYNTHESIS_DRIVER", "/opt/oeis-build-info/synthesis-core-test-driver")
    )
    wasm = Path(
        os.environ.get("OEIS_BIGINT_WASM", "/opt/oeis-build-info/bigint-dependency-smoke.wasm")
    )
    if not path.is_file() or not wasm.is_file():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("mandatory synthesis qualification driver/artifact missing")
        pytest.skip("requires pinned native qualification image")
    from oeis_learn.synthesis.admission import reference_identity

    def run(operation, source, indices, values=(), **kwargs):
        from oeis_learn.synthesis.contracts import identity

        profile = kwargs.pop("language_profile", "bigint-body/v1")
        policy = kwargs.pop("abstraction_policy", "none")
        seed = kwargs.pop("seed", 20261018)
        # Resolved limits/evidence are harness inputs beside the Phase 1 envelope.
        request = dict(
            protocol="synthesis-worker/v1",
            request_id="qualification",
            operation=operation,
            worker_generation=0,
            contract_sha256=identity(["qualification/v1", profile]),
            language_profile=profile,
            resource_profile="bigint-resources/v1",
            solver_profile="native-fitting/v1",
            candidates=[
                dict(
                    candidate_id="candidate-0",
                    original_source=source,
                    original_source_sha256="sha256:" + hashlib.sha256(source.encode()).hexdigest(),
                    codec_profile="bigint-codec/v1" if profile == "bigint-body/v1" else profile,
                    parameter_ids=[],
                    abstraction_policy=policy,
                )
            ],
            visible_indices=list(indices),
            visible_values=list(values) if operation != "execute" else ["0"] * len(indices),
            remaining_budget=dict(wall_time_ms=2000, attempts=1),
            seed=seed,
        )
        raw = json.dumps(
            dict(request=request, reference_sha256=reference_identity(), **kwargs)
        ).encode()
        assert len(raw) <= 4 * 1024**2
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            proc = subprocess.Popen(
                [str(path)], stdin=subprocess.PIPE, stdout=stdout, stderr=stderr
            )
            try:
                proc.stdin.write(raw)
                proc.stdin.close()
                deadline = time.monotonic() + (40 if operation == "execute" else 8)
                while proc.poll() is None:
                    assert time.monotonic() < deadline, "native diagnostic deadline"
                    status = Path(f"/proc/{proc.pid}/status")
                    try:
                        status_lines = status.read_text().splitlines()
                    except FileNotFoundError:
                        status_lines = []
                    for line in status_lines:
                        if line.startswith("VmRSS:"):
                            assert int(line.split()[1]) <= 1024**2, "native diagnostic RSS cap"
                    assert stdout.tell() <= 4 * 1024**2 and stderr.tell() <= 65536, (
                        "diagnostic output cap"
                    )
                    time.sleep(0.02)
                stdout.seek(0)
                stderr.seek(0)
                assert proc.returncode == 0, stderr.read(65536).decode()
                return json.loads(stdout.read(4 * 1024**2 + 1))
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)

    from oeis_learn.synthesis.contracts import identity

    with path.open("rb") as handle:
        binary_hash = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
    guest_hash = "sha256:" + hashlib.sha256(wasm.read_bytes()).hexdigest()
    run.identity_for = lambda profile: identity(
        dict(guest_sha256=guest_hash, binary_sha256=binary_hash, profile=profile)
    )
    run.runtime_sha256 = run.identity_for("bigint-body/v1")
    return run


@pytest.fixture
def certificate_solver():
    """Loop proofs require the pinned native checker, not a Python fallback."""
    binary = Path(
        os.environ.get("OEIS_CERTIFICATE_SOLVER", "/opt/oeis-build-info/certificate-solver")
    )
    if not binary.is_file():
        if any(os.environ.get(key) == "1" for key in ("OEIS_REQUIRE_BIGINT_BUILD", "OEIS_REQUIRE_NATIVE")):
            pytest.fail("packaged native certificate solver required")
        pytest.skip("requires native certificate solver in qualification image")
    return binary


@pytest.fixture
def fitting_client():
    """Real owning-role client; a required image may never silently skip it."""
    from pathlib import Path
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.contracts import identity
    import os
    binary = Path('/opt/oeis-build-info/synthesis-worker')
    if not binary.is_file():
        if os.environ.get('OEIS_REQUIRE_BIGINT_BUILD') == '1':
            pytest.fail('packaged native synthesis worker required')
        pytest.skip('requires native qualification image')
    with SynthesisClient(identity('retirement-caller-regressions/v1')) as client:
        yield client
