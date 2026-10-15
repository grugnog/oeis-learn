"""Exercise the public host CLI and its receipts against a real local engine."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from oeis_learn.container_runtime import resolve_engine
from oeis_learn.fresh.common import digest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.docker
@pytest.mark.parametrize("launcher", ["fresh", "generic"])
def test_private_compiler_storage_exceeds_small_tmpfs(tmp_path, launcher):
    """Reproduce >64 MiB scratch pressure without needing an actual GPU."""
    image = os.environ.get("OEIS_ROLE_TEST_IMAGE")
    if not image:
        pytest.skip("explicit native image and real engine required")
    engine = resolve_engine()
    image = engine.image(image)
    inputs, output = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    code = """
import json, os, shutil, tempfile
from pathlib import Path
assert shutil.disk_usage('/tmp').total <= 64 << 20
with tempfile.NamedTemporaryFile() as stream:
    path = Path(stream.name)
    assert path.parent == Path('/work/output/.runtime/tmp'), path
    chunk = b'x' * (1 << 20)
    for _ in range(80):
        stream.write(chunk)
    stream.flush()
    assert path.stat().st_size == 80 << 20
for key in ('HOME', 'XDG_CACHE_HOME'):
    path = Path(os.environ[key]) / 'compiler-cache.json'
    path.write_text(json.dumps({'stored': True}))
try:
    Path('/work/input/forbidden').write_text('must fail')
except OSError:
    pass
else:
    raise AssertionError('input mount is writable')
print('PRIVATE COMPILER STORAGE PASS')
"""
    name = "oeis-storage-" + uuid.uuid4().hex
    command = ["python", "-c", code]
    if launcher == "fresh":
        spec = importlib.util.spec_from_file_location(
            "storage_host", ROOT / "scripts/foundation_host.py"
        )
        host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(host)
        argv = host.container_command(
            {
                "engine": engine.name,
                "engine_executable": engine.executable,
                "engine_info": engine.info(),
            },
            image,
            name,
            "worker",
            inputs,
            output,
            command,
        )
    else:
        from oeis_learn.tracking.foundation_container import container_command

        argv = container_command(
            role="worker",
            image=image,
            inputs=inputs,
            output=output,
            name=name,
            command=command,
            engine=engine,
        )
    try:
        result = subprocess.run(argv, capture_output=True, timeout=120)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert b"PRIVATE COMPILER STORAGE PASS" in result.stdout
        for directory in ("home", "cache"):
            assert json.loads(
                (output / ".runtime" / directory / "compiler-cache.json").read_text()
            ) == {"stored": True}
        assert not (inputs / "forbidden").exists()
    finally:
        engine.remove(name)


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
        receipts = {
            p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*/receipt.json")
        }
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
