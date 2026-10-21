"""Read-only collection and visible-only replay; mocks never qualify native/HIP."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from oeis_learn.fresh.common import file_hash, inventory, read, write


def utility():
    path = Path(__file__).resolve().parents[2] / "scripts/diagnose_bigint_run.py"
    spec = importlib.util.spec_from_file_location("bigint_diagnostic_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    from oeis_learn.fresh.bigint_config import defaults
    from tests.unit.test_bigint_diagnostic_evidence import selection

    module = utility()
    run = root / "runs/parent"
    job = root / "run-state" / module.digest("runs/parent")[7:]
    generation = "role-" + "a" * 32
    scoring = "role-" + "b" * 32
    role = job / "roles" / generation
    inputs = job / "inputs" / generation
    parent = dict(
        version="bigint-run/v1", contract_sha256="sha256:" + "c" * 64, image="sha256:" + "d" * 64
    )
    write(run / "run.json", parent)
    write(run / "status.json", dict(status="completed", final_access="sealed"))
    write(run / "budget.json", dict(remaining=6670))
    write(run / "checkpoint-identity.json", dict(selected="unchanged-checkpoint"))
    observation = {
        mode: dict(success=mode == "raw", complete=mode == "raw")
        for mode in ("raw", "rescue", "equal_budget", "shuffled", "absent", "template")
    }
    write(
        run / "comparison.json",
        dict(
            observations={"0:tiny": {"prompt": dict(outcomes=observation)}},
            splits={
                "0:tiny": dict(archive=dict(role_output="/original/machine/roles/" + generation))
            },
        ),
    )
    settings = dict(
        contract_sha256=parent["contract_sha256"],
        recipe=defaults(),
        seconds=63,
        device="cuda",
        final_access="sealed",
    )
    write(inputs / "settings.json", settings)
    write(
        inputs / "prompts.json",
        [dict(id="prompt", indices=list(map(str, range(20))), values=list(map(str, range(20))))],
    )
    selected = selection()
    selected["original_source"] = selected["selected"]["source"]
    selections = {"prompt": dict(selections=dict(raw=selected, rescue=None), complete=False)}
    write(role / "selections/chunk-0000.json", selections)
    refs = {"selections/chunk-0000.json": inventory(role)["selections/chunk-0000.json"]}
    write(role / "result.json", dict(files=refs))
    write(job / "inputs" / scoring / "selections.json", refs)
    write(job / "inputs" / scoring / "settings.json", dict(settings, device="cpu"))
    # Collection must not parse or export private truth, weights or unrelated runtime files.
    (job / "inputs" / scoring / "truth.json").write_text("NOT JSON: PRIVATE TRUTH")
    write(job / "roles" / scoring / "outcomes.json", {"prompt": observation})
    (role / "stdout.log").write_text("generation progress\n")
    (job / "roles" / scoring / "stdout.log").write_text("scoring progress\n")
    (run / "weights.pt").write_bytes(b"private checkpoint weights")
    return module, run, role, settings


def test_collection_preserves_parent_and_exports_specific_visible_evidence(tmp_path):
    module, run, role, settings = fixture(tmp_path)
    before = inventory(tmp_path)
    output = tmp_path / "analysis/collection"
    target = module.collect(tmp_path, "runs/parent", "prompt", output)
    assert all(file_hash(tmp_path / name) == ref["sha256"] for name, ref in before.items())
    summary = read(output / "summary.json")
    case = read(output / "case.json")
    assert summary["candidate_replay_ready"] and not summary["detailed_trace_available"]
    assert summary["cached_fit_matches"] == 0 and summary["scoring_role"] == "role-" + "b" * 32
    assert case["resources"] == settings["recipe"]["execution"]["resources"]
    assert case["fitting_seconds"] == case["raw_seconds"] == 4
    assert len(case["prompt"]["values"]) == 20
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
        assert all("truth" not in name and "weights" not in name for name in bundle.namelist())
        assert b"PRIVATE TRUTH" not in b"".join(bundle.read(n) for n in bundle.namelist())
        manifest = json.loads(bundle.read("archive-manifest.json"))
        import hashlib

        assert all(
            "sha256:" + hashlib.sha256(bundle.read(ref["path"])).hexdigest() == ref["sha256"]
            for ref in manifest["files"]
        )
    with pytest.raises(ValueError, match="fresh output"):
        module.collect(tmp_path, "runs/parent", "prompt", output)


@pytest.mark.parametrize("damage", ["changed_chunk", "traversal", "symlink", "wrong_prompt"])
def test_collector_rejects_changed_or_escaping_selections(tmp_path, damage):
    module, run, role, _ = fixture(tmp_path)
    chunk = role / "selections/chunk-0000.json"
    if damage == "changed_chunk":
        chunk.write_text("{}")
    elif damage == "traversal":
        write(
            role / "result.json",
            dict(files={"selections/../../private.json": dict(bytes=1, sha256="bad")}),
        )
    elif damage == "symlink":
        chunk.unlink()
        chunk.symlink_to(run / "budget.json")
    else:
        selections = read(chunk)
        selections["prompt"]["selections"]["raw"]["prompt_sha256"] = "wrong"
        write(chunk, selections)
        write(
            role / "result.json",
            dict(
                files={"selections/chunk-0000.json": inventory(role)["selections/chunk-0000.json"]}
            ),
        )
    with pytest.raises(ValueError):
        module.collect(tmp_path, "runs/parent", "prompt", tmp_path / "collection")


def test_archive_excludes_role_runtime_cache(tmp_path):
    module = utility()
    write(tmp_path / "output/summary.json", dict(status="COMPLETED"))
    write(tmp_path / "output/.runtime/native-cache/secret.json", dict(unexported=True))
    with zipfile.ZipFile(module.archive(tmp_path / "output")) as bundle:
        assert set(bundle.namelist()) == {"summary.json", "archive-manifest.json"}


def test_visible_replay_keeps_seed_limits_and_separate_debit(tmp_path, monkeypatch):
    from oeis_learn.synthesis import client as clients

    module, run, _, _ = fixture(tmp_path)
    collection = tmp_path / "collection"
    module.collect(tmp_path, "runs/parent", "prompt", collection)
    case = read(collection / "case.json")
    before = inventory(run)
    calls = []

    class Client:
        def __init__(self, contract, **kw):
            assert contract == case["parent_contract_sha256"]
            self.limits = kw["limits"]
            assert self.limits.input_bits == case["resources"]["input_bits"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def runtime_identity(self, profile):
            return "mock-runtime"

        def execute(self, source, indices, **kw):
            assert len(indices) == 20
            return dict(outcome="ok", values=list(indices))

        def fit(self, source, prompt, **kw):
            import time

            calls.append(kw)
            assert source == case["original_source"] and len(prompt.values) == 20
            assert 0 < kw["deadline"] - time.monotonic() <= 4
            return dict(outcome="unsupported", reason="mock diagnostic", admitted=[])

    monkeypatch.setattr(clients, "SynthesisClient", Client)
    replay = tmp_path / "replay"
    assert module.inside(collection / "case.json", replay, 30, case["parent_image"]) == 0
    summary = read(replay / "summary.json")
    budget = read(replay / "budget.json")
    assert calls[0]["seed"] == case["seed"] and calls[0]["policy"] == case["policy"]
    assert summary["same_image"] and not summary["qualified"]
    assert summary["fitting"]["outcome"] == "unsupported"
    from oeis_learn.synthesis.state import BudgetLedger

    assert BudgetLedger.restore(budget).remaining == 0
    assert inventory(run) == before


def test_replay_container_mounts_only_case_script_and_new_output(tmp_path, monkeypatch):
    module, _, _, _ = fixture(tmp_path)
    collection = tmp_path / "collection"
    module.collect(tmp_path, "runs/parent", "prompt", collection)
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import foundation_local
    import qualify_bigint_phase5

    commands, removed = [], []
    engine = SimpleNamespace(
        image=lambda image: image,
        command=lambda **kwargs: commands.append(kwargs) or ["mock-engine"],
        remove=lambda name: removed.append(name),
    )
    monkeypatch.setattr(
        foundation_local,
        "Host",
        lambda args: SimpleNamespace(engine=engine, policy=dict(cpu_memory_bytes=1 << 30, cpus=2)),
    )

    def launch(command, output, **kwargs):
        write(
            output / "summary.json",
            dict(
                status="COMPLETED",
                image="sha256:" + "d" * 64,
                case_sha256=file_hash(collection / "case.json"),
            ),
        )

    monkeypatch.setattr(qualify_bigint_phase5, "launch", launch)
    output = tmp_path / "replay"
    target = module.replay(
        SimpleNamespace(evidence=str(collection), output=str(output), image=None, seconds=30)
    )
    assert target.exists() and removed
    mounts = commands[0]["mounts"]
    assert [(dst, writable) for _, dst, writable in mounts] == [
        ("/work/case.json", False),
        ("/work/diagnose.py", False),
        ("/work/output", True),
    ]
    assert all(Path(src) != tmp_path / "runs/parent" for src, _, _ in mounts)
    assert "gpu" not in commands[0] and "network" not in commands[0]


@pytest.mark.parametrize("failure", ["timeout", "missing_summary", "cleanup"])
def test_replay_failure_still_archives_and_charges_host_budget(tmp_path, monkeypatch, failure):
    module, _, _, _ = fixture(tmp_path)
    collection = tmp_path / "collection"
    module.collect(tmp_path, "runs/parent", "prompt", collection)
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import foundation_local
    import qualify_bigint_phase5
    from oeis_learn.synthesis.state import BudgetLedger

    def remove(name):
        if failure == "cleanup":
            raise RuntimeError("mock cleanup failed")

    engine = SimpleNamespace(
        image=lambda image: image, command=lambda **kw: ["mock-engine"], remove=remove
    )
    monkeypatch.setattr(
        foundation_local,
        "Host",
        lambda args: SimpleNamespace(engine=engine, policy=dict(cpu_memory_bytes=1 << 30, cpus=2)),
    )

    def launch(command, output, **kwargs):
        if failure == "timeout":
            raise TimeoutError("mock role timeout")
        if failure == "cleanup":
            write(
                output / "summary.json",
                dict(
                    status="COMPLETED",
                    image="sha256:" + "d" * 64,
                    case_sha256=file_hash(collection / "case.json"),
                ),
            )

    monkeypatch.setattr(qualify_bigint_phase5, "launch", launch)
    output = tmp_path / "replay"
    with pytest.raises(RuntimeError):
        module.replay(
            SimpleNamespace(evidence=str(collection), output=str(output), image=None, seconds=30)
        )
    assert output.with_suffix(".zip").exists()
    assert BudgetLedger.restore(read(output / "host-budget.json")).remaining == 0
    assert read(output / "host-summary.json")["status"] == "FAIL"


def test_help_is_standard_library_only():
    subprocess.run(
        [sys.executable, "-S", utility().__file__, "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )
