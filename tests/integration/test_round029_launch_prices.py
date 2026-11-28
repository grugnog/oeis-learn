"""CPU pricing replays positive-token bookkeeping at real durable boundaries."""

import pytest

from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.pipeline_validation import progressive_velocity_ledger_prices as prices


def test_bookkeeping_reopens_the_whole_update_that_crossed_each_look(tmp_path):
    order = ["a", "b", "c"]
    lengths = dict(a=2, b=3, c=4)
    frame = prices.frame(order, lengths, 16, looks=(50, 120))
    assert [(k["updates"], k["useful_tokens"]) for k in frame["cuts"]] == [(2, 95), (3, 144)]
    result = prices.measure(tmp_path, frame, order, lengths)
    assert result["updates"] == 3 and result["useful_tokens"] == 144
    assert result["price"]["tail_us"] >= result["price"]["expected_us"] > 0
    assert result["ledger_bytes"] > 0
    assert prices.verify(tmp_path, frame, order, lengths) == result
    saved = tmp_path / "fixture-accounting/budget.jsonl"
    saved.write_bytes(saved.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(ValueError):
        prices.verify(tmp_path, frame, order, lengths)


def test_paid_cpu_batches_resume_without_repaying_work(tmp_path):
    root = tmp_path
    folder = root / "run-state/discovery" / prices.SESSION
    folder.mkdir(parents=True)
    order, lengths = ["a", "b"], {"a": 2, "b": 3}
    frames = {str(b): prices.frame(order, lengths, b, looks=(50, 120)) for b in (16, 32)}
    plan = prices.plan_for("a" * 40, digest("fresh"), digest("cycles"), frames)
    first = prices.run(root, folder, plan, order, lengths, minutes=2, floor=0)
    assert set(first["batches"]) == {"16", "32"}
    assert first["charged_work"]["tokens"] == 0 and first["charged_work"]["attempts"] == 2
    assert first["remaining_us"] + first["charged_work"]["elapsed_us"] == 120_000_000
    assert prices.run(root, folder, plan, order, lengths, minutes=2, floor=0) == first
    assert prices.reopen(root, folder, plan, order, lengths, first["grant"]) == first
    prices.record(root, folder, plan, order, lengths, first)
    with pytest.raises(ValueError, match="both original"):
        prices.record(root, folder, plan, order, lengths, dict(first, errors={"32": "failed"}))
    assert len(list(folder.glob("preparation/update-accounting/*/operations/*/attempt-*"))) == 2


def test_denied_cpu_footprint_keeps_the_full_grant_available(tmp_path):
    root = tmp_path
    folder = root / "run-state/discovery" / prices.SESSION
    folder.mkdir(parents=True)
    lengths, order = {"a": 3}, ["a"]
    frames = {str(b): prices.frame(order, lengths, b, looks=(50, 120)) for b in (16, 32)}
    plan = prices.plan_for("a" * 40, digest("fresh"), digest("cycles"), frames)
    denied = prices.run(root, folder, plan, order, lengths, minutes=2, floor=1 << 60)
    assert denied["remaining_us"] == 120_000_000 and not any(denied["charged_work"].values())
    assert all(v["mounts"][0]["deficit_bytes"] > 0 for v in denied["errors"].values())
    assert not (folder / prices.MARKER).exists()
    completed = prices.run(root, folder, plan, order, lengths, minutes=2, floor=0)
    assert completed["grant"] == denied["grant"] and not completed["errors"]
