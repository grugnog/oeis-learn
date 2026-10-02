"""Regression contracts for phase 6 accounting and bounded telemetry."""

import pytest
from oeis_learn.tracking.budget_ledger import BudgetLedger
from oeis_learn.tracking.foundation_metrics import EventLog, metric


class Clock:
    def __init__(self):
        self.now = dict(monotonic_ns=0, utc_ns=1000, boot_id="boot-a")

    def __call__(self):
        return dict(self.now)

    def advance(self, n):
        self.now["monotonic_ns"] += n
        self.now["utc_ns"] += n


def test_crash_gap_and_clean_pause(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path, "run", 100, clock=clock)
    ledger.transition("training")
    clock.advance(7)
    ledger.heartbeat()
    clock.advance(11)
    recovered = BudgetLedger(tmp_path, "run", 100, clock=clock)
    assert recovered.state["measured_ns"] == 7
    assert recovered.state["estimated_ns"] == 11
    recovered.transition("paused")
    clock.advance(50)
    recovered = BudgetLedger(tmp_path, "run", 100, clock=clock)
    assert recovered.state["charged_budget_ns"] == 18
    recovered.transition("checkpoint")
    clock.advance(3)
    recovered.transition("completed")
    assert recovered.state["charged_budget_ns"] == 21
    assert recovered.state["phase_ns"]["checkpoint"] == 3


def test_cross_boot_and_backwards_clock(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path, "run", 100, clock=clock)
    ledger.transition("training")
    clock.advance(12)
    clock.now.update(boot_id="boot-b", monotonic_ns=0)
    recovered = BudgetLedger(tmp_path, "run", 100, clock=clock)
    assert recovered.state["estimated_ns"] == 12
    clock.now["utc_ns"] -= 1
    with pytest.raises(ValueError, match="clock"):
        recovered.heartbeat()


def test_metrics_are_typed_and_segments_bounded(tmp_path):
    assert metric(0, "count", "counter")["value"] == 0
    assert metric(None, "bytes", "HIP", state="unavailable", reason="CPU")["value"] is None
    for value, state in [(0, "disabled"), (None, "measured"), (float("nan"), "measured")]:
        with pytest.raises(ValueError):
            metric(value, "count", "counter", state=state)
    log = EventLog(tmp_path, segment_bytes=300, quota_bytes=1500)
    event = {"kind": "update", "update": 1, "loss": metric(0, "nats", "CE")}
    log.append("update-1", event)
    assert log.append("update-1", event) is False
    with pytest.raises(ValueError, match="duplicate"):
        log.append("update-1", {"kind": "different"})
    for i in range(5):
        log.append(f"event-{i}", event)
    assert all(p.stat().st_size <= 300 for p in tmp_path.glob("*.jsonl"))
    with pytest.raises(OSError, match="quota"):
        for i in range(100):
            log.append(f"many-{i}", event)


def test_configuration_unknown_inactive_and_stable_effective(tmp_path):
    from pathlib import Path
    import yaml
    from oeis_learn.experiments.config import load_config

    raw = yaml.safe_load(Path("configs/foundation/wat_smoke.yaml").read_text())
    path = tmp_path / "config.yaml"
    for section, key, value in [
        ("training", "mystery", 1),
        ("training", "scheduler", "measured"),
        ("disabled", "scaffolds", "measured"),
    ]:
        from copy import deepcopy

        bad = deepcopy(raw)
        bad[section][key] = value
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            load_config(path)
    path.write_text(yaml.safe_dump(raw))
    first = load_config(path).persist_effective(tmp_path / "first").as_path().read_bytes()
    second = load_config(path).persist_effective(tmp_path / "second").as_path().read_bytes()
    assert first == second


def test_per_program_loss_weights_programs_equally_and_counts_eos():
    import torch
    import torch.nn.functional as F
    from oeis_learn.decoder.program_codec import PAD_ID, EOS_ID, FOUNDATION_VOCAB_SIZE
    from oeis_learn.rl.foundation_sft import per_program_loss

    logits = torch.zeros(2, 3, FOUNDATION_VOCAB_SIZE, requires_grad=True)
    targets = torch.tensor([[EOS_ID, PAD_ID, PAD_ID], [4, 5, EOS_ID]])
    with torch.no_grad():
        logits[0, 0, EOS_ID] = 2
    expected = (
        F.cross_entropy(logits[0, :1], targets[0, :1]) + F.cross_entropy(logits[1], targets[1])
    ) / 2
    actual = per_program_loss(logits, targets)
    assert torch.equal(actual, expected)
    actual.backward()
    assert logits.grad[0, 0, EOS_ID] != 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0


def test_torn_last_event_is_quarantined(tmp_path):
    log = EventLog(tmp_path, segment_bytes=300, quota_bytes=1500)
    log.append("one", {"n": 1})
    path = tmp_path / "00000000.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"sequence":1')
    recovered = EventLog(tmp_path, segment_bytes=300, quota_bytes=1500)
    recovered.append("two", {"n": 2})
    assert len(EventLog(tmp_path, segment_bytes=300, quota_bytes=1500).events) == 2
    assert (tmp_path / "00000000.jsonl.partial").read_bytes() == b'{"sequence":1'


def test_runtime_pins_forward_and_controller_implementations():
    from oeis_learn.rl.foundation_sft import runtime_record

    runtime = runtime_record("cpu")
    for path in (
        "encoder/tri_stream_encoder.py",
        "decoder/wat_decoder.py",
        "decoder/program_codec.py",
        "tracking/budget_ledger.py",
        "tracking/foundation_controller.py",
    ):
        assert path in runtime["implementation"]
    assert runtime["float32_matmul_precision"] == "highest" and not runtime["allow_tf32"]
