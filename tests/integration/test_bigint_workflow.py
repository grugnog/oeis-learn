"""Persisted integration transitions reuse Phase 2 identity and budget rules."""

import pytest
from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.fresh.bigint_state import DurableRun, CheckpointStore
from oeis_learn.fresh.bigint_inputs import visible_record
from oeis_learn.fresh.bigint_bundle import pack


def test_visible_view_never_exports_continuations_or_rebases_indices():
    row = dict(
        id="sample",
        family="family",
        stage="A",
        indices=list(map(str, range(-7, 93))),
        values=[str(n) for n in range(20)] + ["987654321"] * 80,
        source="(program (return $index))",
        parameter_source="(program (return $index))",
    )
    record = visible_record(row, "concrete")
    assert record["indices"][0] == "-7" and len(record["values"]) == 20
    assert "987654321" not in str(record)
    assert len(record["horizon_indices"]) == 100


def test_debit_checkpoint_resume_and_analysis_do_not_refund_or_change_parent(tmp_path):
    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    manifest = dict(identity="sha256:" + "1" * 64, split_sha256="sha256:" + "2" * 64)
    run = DurableRun.create(
        tmp_path / "run", cfg, manifest, image="sha256:" + "3" * 64, device="cpu"
    )
    run.start("first", 10)
    run.finish("first", "interrupted")
    before = run.ledger.remaining
    resumed = DurableRun.resume(tmp_path / "run", cfg, image="sha256:" + "3" * 64)
    assert resumed.ledger.remaining == before
    with pytest.raises(ValueError, match="image"):
        DurableRun.resume(tmp_path / "run", cfg, image="sha256:" + "4" * 64)
    states = {
        k: dict(value=k)
        for k in ("weights", "optimizer", "rng", "data_order", "selection", "curriculum", "replay")
    }
    store = CheckpointStore(tmp_path / "run/checkpoints", run.contract)
    saved = store.save(states, run.ledger, state="interrupted")
    loaded = store.load(saved)
    assert loaded["states"] == states and loaded["ledger"].remaining == before
    blob = next((tmp_path / "run/checkpoints").glob("*/weights.pt"))
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact"):
        store.load(saved)


def test_core_export_precedes_details_and_excludes_private_inputs(tmp_path):
    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    run = DurableRun.create(
        tmp_path / "run",
        cfg,
        dict(identity="sha256:" + "1" * 64, split_sha256="sha256:" + "2" * 64),
        image="sha256:" + "3" * 64,
        device="cpu",
    )
    from oeis_learn.fresh.common import write

    for name in ("summary.json", "checkpoint-identity.json", "comparison.json"):
        write(run.root / name, dict(status="fixture", final_access="sealed"))
    (run.root / "private").mkdir()
    (run.root / "private/truth.json").write_text("NEVER EXPORT")
    with pytest.raises(ValueError, match="16"):
        pack(run.root, tmp_path / "too-small.zip", max_bytes=1 << 20)
    result = pack(run.root, tmp_path / "report.zip", max_bytes=16 << 20)
    import zipfile

    with zipfile.ZipFile(tmp_path / "report.zip") as archive:
        assert "summary.json" in archive.namelist() and "bundle-manifest.json" in archive.namelist()
        assert all("private" not in name for name in archive.namelist())
    assert result["complete_core"]


def _fixture(device, tmp_path):
    from oeis_learn.fresh.common import write, digest

    recipe = resolve(dict(version="bigint-bootstrap/v1", learning=dict(max_updates=3)))
    run = DurableRun.create(
        tmp_path / "run",
        recipe,
        dict(identity="sha256:" + "1" * 64, split_sha256="sha256:" + "2" * 64),
        image="sha256:" + "3" * 64,
        device=device,
    )
    source = "(program (return (add $index (int 1))))"
    from oeis_learn.decoder.bigint_codec import encode_body

    examples = {
        split: [
            dict(
                id=split,
                family=split,
                stage="A",
                indices=list(map(str, range(-7, 13))),
                values=list(map(str, range(-6, 14))),
                horizon_indices=list(map(str, range(-7, 93))),
                source=source,
                oracle_source=source,
                tokens=encode_body(source),
            )
        ]
        for split in ("training", "parameter_development", "structural_development")
    }
    settings = dict(
        recipe=recipe,
        contract=run.record["contract"],
        contract_sha256=run.contract.sha256,
        device=device,
        seconds=120,
        fixture=True,
        update_limit=1,
        examples_sha256=digest(examples),
    )
    inputs = tmp_path / "inputs"
    write(inputs / "examples.json", examples)
    run.start("first", 130)
    settings["ledger"] = run.ledger.snapshot()
    write(inputs / "settings.json", settings)
    return run, inputs, settings, examples


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_integrated_device_checkpoint_update(tmp_path, device):
    import torch, os, shutil

    if device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        if os.environ.get("OEIS_REQUIRE_BIGINT_GPU") == "1":
            pytest.fail("actual HIP required")
        pytest.skip("actual HIP required")
    from oeis_learn.fresh.bigint_worker import train, models
    from oeis_learn.fresh.common import write, read

    run, inputs, settings, examples = _fixture(device, tmp_path)
    initial, _, _ = models(settings["recipe"], device)
    before = initial.chunk_encoder.projection.weight.detach().cpu().clone()
    del initial
    result = train(inputs, tmp_path / "first")
    assert result["update"] == 1 and result["counts"]["device"].startswith(device)
    state = CheckpointStore(tmp_path / "first/checkpoints", run.contract).load(result["checkpoint"])
    assert set(state["states"]) == set(CheckpointStore.STATES)
    assert not torch.equal(
        before, state["states"]["weights"]["encoder"]["chunk_encoder.projection.weight"]
    )
    run.finish("first", "interrupted")
    resumed = DurableRun.resume(run.root, settings["recipe"], image=run.record["image"])
    resumed.start("second", 130)
    settings["ledger"] = resumed.ledger.snapshot()
    write(inputs / "settings.json", settings)
    shutil.copytree(tmp_path / "first/checkpoints", inputs / "checkpoints")
    shutil.copyfile(tmp_path / "first/selected.json", inputs / "selected.json")
    continued = train(inputs, tmp_path / "second")
    assert continued["update"] == 2 and resumed.ledger.remaining == 7200 - 260
    # The exact resumed data/RNG/optimizer states reproduce two uninterrupted updates.
    clean = tmp_path / "clean"
    clean.mkdir()
    write(clean / "settings.json", dict(settings, update_limit=2))
    write(clean / "examples.json", examples)
    uninterrupted = train(clean, tmp_path / "uninterrupted")
    expected = CheckpointStore(tmp_path / "uninterrupted/checkpoints", run.contract).load(
        uninterrupted["checkpoint"]
    )
    actual = CheckpointStore(tmp_path / "second/checkpoints", run.contract).load(
        continued["checkpoint"]
    )
    for key, tensor in expected["states"]["weights"]["decoder"].items():
        torch.testing.assert_close(
            tensor, actual["states"]["weights"]["decoder"][key], rtol=0, atol=0
        )
    assert read(tmp_path / "second/checkpoint-identity.json")["update"] == 2


def _plan():
    from oeis_learn.fresh.common import digest

    decisions = {
        name: dict(status="completed", decision="frozen fixture")
        for name in ("representation", "curriculum", "replay", "assistance", "encoder")
    }
    h = "sha256:" + "1" * 64
    return dict(
        version="final-evaluation/v1",
        cohort=h,
        decisions=[[key, digest(value)] for key, value in decisions.items()],
        decision_manifest=decisions,
        arms=[["a", 20261016, h, h, h, h, 20]],
        denominator=8,
        controls=["correct", "shuffled", "absent", "template"],
        budget=20,
    )


def test_final_exposure_persists_before_work_and_retry_preserves_debits(tmp_path):
    from oeis_learn.fresh.bigint_state import expose_final, finish_final
    from oeis_learn.fresh.common import read
    import copy

    plan = _plan()
    h = plan["cohort"]
    kwargs = dict(cohort=h, checkpoint=h, recipe=h, portfolio=h, runtime=h, seed=20261016, debit=7)
    arm, state = expose_final(tmp_path, plan, **kwargs)
    assert (
        read(tmp_path / "state.json")["state"] == "exposed"
        and __import__("oeis_learn.synthesis.state", fromlist=["BudgetLedger"])
        .BudgetLedger.restore(state["ledger"])
        .remaining
        == 13
    )
    # Simulate a killed coordinator: stale active arm is interrupted, not refunded.
    arm, state = expose_final(tmp_path, plan, **kwargs)
    assert (
        __import__("oeis_learn.synthesis.state", fromlist=["BudgetLedger"])
        .BudgetLedger.restore(state["ledger"])
        .remaining
        == 6
        and state["arm_spent"][arm] == 14
    )
    finish_final(tmp_path, plan, arm, "interrupted")
    with pytest.raises(ValueError):
        expose_final(tmp_path, plan, **kwargs)
    changed = copy.deepcopy(plan)
    changed["budget"] = 30
    with pytest.raises(ValueError, match="immutable|changed"):
        expose_final(tmp_path, changed, **dict(kwargs, debit=1))
    assert (
        __import__("oeis_learn.synthesis.state", fromlist=["BudgetLedger"])
        .BudgetLedger.restore(read(tmp_path / "state.json")["ledger"])
        .remaining
        == 6
    )
    incomplete = copy.deepcopy(plan)
    incomplete["decision_manifest"]["encoder"]["status"] = "pending"
    with pytest.raises(ValueError, match="decision"):
        expose_final(tmp_path / "fresh", incomplete, **kwargs)


def test_selection_scoring_is_independent_of_private_truth(tmp_path, monkeypatch):
    # Actual native runtime/fitting and separate scoring on exactly the sealed source.
    import os, time
    from pathlib import Path

    if not Path("/opt/oeis-build-info/synthesis-worker").exists():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("native integration image required")
        pytest.skip("native integration image required")
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.fresh.bigint_diagnostics import generate, score
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.decoder.sampler import WatProgramSampler
    from oeis_learn.decoder.bigint_codec import encode_body, BOS_ID
    from types import SimpleNamespace

    source = "(program (return (add $index (int 0))))"
    monkeypatch.setattr(
        WatProgramSampler,
        "sample_trajectories",
        lambda *a, **k: [
            SimpleNamespace(source=source, complete=True, tokens=(BOS_ID, *encode_body(source)))
        ],
    )
    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    encoder, decoder, _ = models(cfg, "cpu")
    rows = [
        dict(id=str(i), indices=list(map(str, range(-7, 13))), values=list(map(str, range(-6, 14))))
        for i in range(2)
    ]
    truth = {
        r["id"]: dict(indices=list(map(str, range(-7, 93))), values=list(map(str, range(-6, 94))))
        for r in rows
    }
    with SynthesisClient("sha256:" + "1" * 64) as native:
        result = generate(encoder, decoder, rows, cfg, native, deadline=time.monotonic() + 120)
        sealed = result["records"]
        before = __import__("copy").deepcopy(sealed)
        scored = score(sealed, truth, native, deadline=time.monotonic() + 120)
        assert (
            scored["splits"]["rescue"]["success"] == 2
            and scored["splits"]["rescue"]["denominator"] == 2
        )
        truth["0"]["values"][-1] = "999"
        changed = score(sealed, truth, native, deadline=time.monotonic() + 120)
        assert changed["splits"]["rescue"]["success"] == 1 and sealed == before
        assert sealed["0"]["selections"]["rescue"]["assignment"]


def test_hidden_mount_rejected_before_model_construction(tmp_path, monkeypatch):
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.fresh.common import write, digest

    run, inputs, settings, examples = _fixture("cpu", tmp_path)
    examples["final"] = [dict(values=["987654321"] * 100)]
    settings["examples_sha256"] = digest(examples)
    write(inputs / "settings.json", settings)
    write(inputs / "examples.json", examples)
    monkeypatch.setattr(
        bigint_worker, "models", lambda *a: pytest.fail("model must not see hidden input")
    )
    with pytest.raises(ValueError, match="hidden/final"):
        bigint_worker.train(inputs, tmp_path / "out")


def test_reports_reject_sealed_final_leaks_and_record_pressure(tmp_path):
    from oeis_learn.fresh.common import write
    import zipfile

    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    root = tmp_path / "run"
    DurableRun.create(
        root,
        cfg,
        dict(identity="sha256:" + "1" * 64, split_sha256="sha256:" + "2" * 64),
        image="sha256:" + "3" * 64,
        device="cpu",
    )
    for name in ("summary.json", "checkpoint-identity.json", "comparison.json"):
        write(root / name, dict(status="fixture"))
    write(root / "details/too-big.json", dict(tokens="x" * (2 << 20)))
    pack(root, tmp_path / "report.zip", max_bytes=16 << 20)
    with zipfile.ZipFile(tmp_path / "report.zip") as z:
        omission = __import__("json").loads(z.read("bundle-manifest.json"))["omitted"][0]
        assert (
            omission["bytes"] > 1 << 20
            and omission["sha256"].startswith("sha256:")
            and omission["reason"] == "detail cap"
        )
    write(root / "summary.json", dict(final_values=["SECRET"]))
    with pytest.raises(ValueError, match="private/final"):
        pack(root, tmp_path / "leak.zip")
