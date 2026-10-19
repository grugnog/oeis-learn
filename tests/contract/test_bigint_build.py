"""Dependency/artifact smoke checks; not bigint arithmetic qualification."""

import hashlib
import json
import os
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_dependency_closures_and_pins():
    crates = ROOT / "crates"
    guest = tomllib.loads((crates / "oeis_bigint_runtime/Cargo.lock").read_text())["package"]
    native = tomllib.loads((crates / "oeis_synthesis_core/Cargo.lock").read_text())["package"]
    binding = tomllib.loads((crates / "oeis_wasm_evaluator/Cargo.lock").read_text())["package"]
    assert not {p["name"] for p in guest} & {"z3", "z3-sys", "z3-src", "pyo3", "wasmtime"}
    for packages in (native, binding):
        versions = {p["name"]: p["version"] for p in packages}
        assert versions["dashu-int"] == "0.6.2"
        assert versions["z3"] == "0.21.1"
        assert versions["z3-sys"] == "0.13.1"
        assert versions["z3-src"] == "501.0.1"
    assert {p["name"]: p["version"] for p in binding}["wasmtime"] == "20.0.2"
    assert "oeis_wasm_evaluator" not in {p["name"] for p in native}
    for crate in ("oeis_synthesis_core", "oeis_bigint_runtime"):
        package = tomllib.loads((crates / crate / "Cargo.toml").read_text())
        assert "oeis_wasm_evaluator" not in package["dependencies"]


def test_source_inventory_includes_nested_modules_and_locks(tmp_path):
    from oeis_learn.sandbox.native_runtime import source_bytes
    import shutil

    for path in ("crates", "docker/dev", "src/oeis_learn/synthesis"):
        shutil.copytree(
            ROOT / path, tmp_path / path, ignore=shutil.ignore_patterns("target", "__pycache__")
        )
    shutil.copyfile(ROOT / "Dockerfile", tmp_path / "Dockerfile")
    shutil.copyfile(ROOT / "rustfmt.toml", tmp_path / "rustfmt.toml")
    (tmp_path / "src/oeis_learn/sandbox").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "src/oeis_learn/sandbox/preamble.wat",
        tmp_path / "src/oeis_learn/sandbox/preamble.wat",
    )
    binding = tmp_path / "crates/oeis_wasm_evaluator"
    old = source_bytes(binding)
    wheel_script = tmp_path / "docker/dev/build-native-wheel.sh"
    wheel_script.write_text(wheel_script.read_text() + "\n# changed wheel recipe\n")
    assert source_bytes(binding) != old
    wheel_script.write_bytes((ROOT / "docker/dev/build-native-wheel.sh").read_bytes())
    nested = tmp_path / "crates/oeis_synthesis_core/src/nested/new.rs"
    nested.parent.mkdir()
    nested.write_text("// new source must invalidate the native identity\n")
    assert source_bytes(binding) != old
    ignored = tmp_path / "crates/oeis_synthesis_core/target/new.rs"
    ignored.parent.mkdir()
    ignored.write_text("ignored artifact")
    assert b"ignored artifact" not in source_bytes(binding)
    assert b"crates/oeis_bigint_runtime/Cargo.lock\0" in old


def test_fixture_hashes_and_independent_expected_values():
    import math

    manifest = json.loads((ROOT / "tests/fixtures/bigint/manifest.json").read_text())
    for entry in manifest["fixtures"]:
        path = ROOT / "tests/fixtures/bigint" / entry["path"]
        assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    arithmetic = json.loads((ROOT / "tests/fixtures/bigint/arithmetic.json").read_text())
    factorial = next(c for c in arithmetic["cases"] if c["id"] == "factorial-0-99")
    assert factorial["indices"] == list(map(str, range(100)))
    assert factorial["expected"] == [str(math.factorial(n)) for n in range(100)]
    wrong = json.loads((ROOT / "tests/fixtures/bigint/false-fits.json").read_text())
    square = next(c for c in wrong["cases"] if c["id"] == "square-16-versus-4")
    assert square["provisional_assignment"] == {"p0": "4"}
    assert square["production_output"] == "16"
    assert square["visible_value"] == "4"
    assert square["expected_admission"] is False


def test_packaged_artifacts_and_binding_exact_numeral():
    required = os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1"
    info = Path(os.environ.get("OEIS_BIGINT_BUILD_INFO", "/opt/oeis-build-info"))
    if not (info / "bigint-build.json").is_file():
        if required:
            pytest.fail("required packaged bigint smoke artifacts are missing")
        pytest.skip("requires built qualification/gpu image")
    from scripts.inspect_bigint_build import inspect

    report = inspect(ROOT, info)
    saved = json.loads((info / "bigint-build.json").read_text())
    assert report == saved
    assert report["production_enabled"] is False
    assert report["imports"] == []
    for path in (
        info / "bigint-dependency-smoke.wasm",
        info / "core.Cargo.lock",
        info / "guest.Cargo.lock",
    ):
        assert path.is_file() and os.access(path, os.R_OK)


@pytest.mark.parametrize("backend", ["python_wasmtime", "rust"])
def test_retained_signed_256_fixture_execution(backend):
    from oeis_learn.sandbox.pipeline import Runtime
    from oeis_learn.sandbox.runner import decode_i256_limbs

    if backend == "rust":
        try:
            __import__("oeis_wasm_evaluator")
        except ImportError:
            if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
                pytest.fail("required retained native backend is missing")
            pytest.skip("requires the built native extension")
    runtime = Runtime(backend=backend)
    cases = json.loads((ROOT / "tests/fixtures/bigint/signed-256.json").read_text())["cases"]
    for case in cases:
        if "limbs" in case:
            assert str(decode_i256_limbs(case["limbs"])) == case["expected"]
        else:
            result = runtime.evaluate(case["source"], [0])
            assert result.outcome == case.get("failure"), (case["id"], result)
            assert result.outputs == ([case["expected"]] if "expected" in case else [])
