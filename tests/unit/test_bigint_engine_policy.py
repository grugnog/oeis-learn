"""Production policy must be explicit, shared with native code and identity-bound."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.synthesis.contracts import identity


def test_fixed_production_policy_and_zero_control_are_distinct():
    from oeis_learn.synthesis.engine_policy import production_policy, engine_configuration

    policy = production_policy()
    assert policy["version"] == "bigint-engine-policy/v2"
    assert policy["dynamic_memory_reserved_for_growth"] == 65536
    assert policy["static_memory_maximum_size"] == policy["dynamic_memory_guard_size"] == 0
    guest = "sha256:" + "a" * 64
    prod = engine_configuration(guest)
    zero = engine_configuration(guest, zero_control=True)
    assert prod["manifest"]["purpose"] == "production"
    assert zero["manifest"]["purpose"] == "qualification-zero-growth-control"
    assert prod["manifest"]["policy_sha256"] == identity(policy)
    assert prod["sha256"] != zero["sha256"]
    policy["dynamic_memory_reserved_for_growth"] = 0
    assert production_policy()["dynamic_memory_reserved_for_growth"] == 65536


def test_runtime_identity_adds_policy_for_bigint_only(tmp_path):
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.engine_policy import production_policy

    binary = tmp_path / "worker"
    binary.write_bytes(b"fixture native binary")
    guest = tmp_path / "guest.wasm"
    guest.write_bytes(b"fixture guest")
    client = SynthesisClient.__new__(SynthesisClient)
    client.worker = SimpleNamespace(command=[str(binary)])
    client.runtime_identities = {}
    import os

    prior = os.environ.get("OEIS_BIGINT_WASM")
    os.environ["OEIS_BIGINT_WASM"] = str(guest)
    try:
        old = dict(
            binary_sha256="sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest(),
            guest_sha256="sha256:" + hashlib.sha256(guest.read_bytes()).hexdigest(),
            profile="bigint-body/v1",
        )
        assert client.runtime_identity("bigint-body/v1") == identity(
            dict(old, engine_policy_sha256=identity(production_policy()))
        )
        assert client.runtime_identity("bigint-body/v1") != identity(old)
        old["profile"] = "signed-256/v1"
        assert client.runtime_identity("signed-256/v1") == identity(old)
    finally:
        if prior is None:
            os.environ.pop("OEIS_BIGINT_WASM", None)
        else:
            os.environ["OEIS_BIGINT_WASM"] = prior


def test_qualification_driver_identity_matches_client_and_policy(tmp_path, monkeypatch):
    from tests.conftest import synthesis_driver
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.engine_policy import production_policy

    binary = tmp_path / "driver"
    binary.write_bytes(b"fixture diagnostic binary")
    guest = tmp_path / "guest.wasm"
    guest.write_bytes(b"fixture guest")
    monkeypatch.setenv("OEIS_SYNTHESIS_DRIVER", str(binary))
    monkeypatch.setenv("OEIS_BIGINT_WASM", str(guest))
    # Construct the real fixture without invoking native execution or a solver.
    driver = synthesis_driver.__wrapped__()
    client = SynthesisClient.__new__(SynthesisClient)
    client.worker = SimpleNamespace(command=[str(binary)])
    client.runtime_identities = {}
    for profile in ("bigint-body/v1", "wat_i256_checked_v1"):
        manifest = dict(
            binary_sha256="sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest(),
            guest_sha256="sha256:" + hashlib.sha256(guest.read_bytes()).hexdigest(),
            profile=profile,
        )
        old_identity = identity(manifest)
        if profile == "bigint-body/v1":
            manifest["engine_policy_sha256"] = identity(production_policy())
            assert driver.identity_for(profile) != old_identity
        assert (
            driver.identity_for(profile) == client.runtime_identity(profile) == identity(manifest)
        )
    assert driver.runtime_sha256 == driver.identity_for("bigint-body/v1")


def test_native_source_inventory_includes_policy_bytes(tmp_path):
    from oeis_learn.sandbox.native_runtime import source_bytes
    import shutil

    root = Path(__file__).resolve().parents[2]
    for path in ["crates", "docker/dev", "src/oeis_learn/synthesis"]:
        shutil.copytree(
            root / path, tmp_path / path, ignore=shutil.ignore_patterns("target", "__pycache__")
        )
    for path in ["Dockerfile", "rustfmt.toml", "src/oeis_learn/sandbox/preamble.wat"]:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, dest)
    crate = tmp_path / "crates/oeis_wasm_evaluator"
    before = source_bytes(crate)
    policy = tmp_path / "src/oeis_learn/synthesis/engine-policy.json"
    assert b"src/oeis_learn/synthesis/engine-policy.json\0" in before
    policy.write_bytes(policy.read_bytes() + b"\n")
    assert source_bytes(crate) != before


@pytest.mark.parametrize("bad", ["gpu", "prepare", "runtime", "inputs"])
def test_engine_adoption_scope_rejects_unrelated_work_before_engine(bad, monkeypatch):
    import sys
    from tests.unit.test_bigint_phase5 import helper

    extra = {
        "gpu": ["--gpu"],
        "prepare": ["--prepare"],
        "runtime": ["--runtime-profile-only"],
        "inputs": ["--inputs", "fixture"],
    }[bad]
    monkeypatch.setattr(
        sys, "argv", ["qualify", "--engine-adoption-only", *extra, "--output", "unused"]
    )
    with pytest.raises(SystemExit) as exc:
        helper().main()
    assert exc.value.code == 2


@pytest.mark.parametrize("failure", [None, "group", "policy", "source", "profile"])
def test_engine_adoption_runs_only_cpu_groups_and_comparison(tmp_path, monkeypatch, failure):
    import sys, json
    from tests.unit.test_bigint_phase5 import helper
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.synthesis.engine_policy import production_policy

    mod = helper()
    snapshot = {"fixture": "source"}
    out = tmp_path / "adoption"
    policy = dict(manifest=production_policy(), sha256=identity(production_policy()))
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda root: snapshot)

    class Host:
        def __init__(self, args):
            self.engine = SimpleNamespace(name="podman")
            self.policy = dict(runtime={})

        def ensure(self):
            return "sha256:" + "a" * 64

    monkeypatch.setitem(sys.modules, "foundation_local", SimpleNamespace(Host=Host, relative=str))
    seen = []

    def stage(host, image, path, argv, **kwargs):
        seen.append(path.name)
        if path.name == "runtime-profile":
            assert "--compare-memory-growth" in argv and kwargs == dict(seconds=600)
            mod.write(
                path / "benchmark.json",
                dict(
                    status="FAIL" if failure == "profile" else "PASS",
                    profile_input_load=True,
                    compare_memory_growth=True,
                    source_sha256=identity(snapshot),
                ),
            )
        else:
            assert kwargs == dict(seconds=3000)
            mod.write(
                path / "summary.json",
                dict(
                    status="FAIL" if failure == "group" and path.name == "fitting" else "PASS",
                    source_sha256="old"
                    if failure == "source" and path.name == "fitting"
                    else identity(snapshot),
                    artifacts=dict(
                        engine_policy={}
                        if failure == "policy" and path.name == "fitting"
                        else policy
                    ),
                ),
            )

    monkeypatch.setattr(mod, "container_stage", stage)
    monkeypatch.setattr(mod, "launch", lambda *a, **kw: pytest.fail("data/training launch"))
    monkeypatch.setattr(sys, "argv", ["qualify", "--engine-adoption-only", "--output", str(out)])
    assert mod.main() == (0 if failure is None else 1)
    assert seen == (
        ["arithmetic", "fitting"]
        if failure in {"group", "policy", "source"}
        else ["arithmetic", "fitting", "containment", "runtime-profile"]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["scope"] == "engine-adoption-only" and summary["qualified"] is False
    assert summary["expected_engine_policy"] == policy
    assert out.with_suffix(".zip").is_file()
