"""Independent durable draw/cut and structural novelty requirements."""

import pytest


def policy(**changes):
    from oeis_learn.generalization.progressive.generator import default_policy

    return default_policy(seed=13) | changes


def test_counter_generation_is_reproducible_and_not_coefficient_only():
    from oeis_learn.generalization.progressive.generator import draw, structure_id
    from oeis_learn.synthesis.reference import evaluate

    p = policy()
    a = [draw(p, i) for i in range(40)]
    assert a == [draw(p, i) for i in range(40)]
    assert len({r["structure"] for r in a}) > 12
    assert structure_id("(program (return (add $index (int 3))))") == structure_id(
        "(program (return (add $index (int -9))))"
    )
    assert any("(loop " in r["source"] for r in a)
    assert any("(select " in r["source"] for r in a)
    for r in a:
        result = evaluate(r["source"], list(map(str, range(100))))
        assert result.outcome == "ok"


def test_reserved_ordinal_survives_crash_and_replay_consumes_work(tmp_path):
    from oeis_learn.generalization.progressive.stream import StreamStore

    s = StreamStore(tmp_path, policy())
    with pytest.raises(InterruptedError):
        s.step(fault="after_reserve")
    resumed = StreamStore(tmp_path, policy())
    first = resumed.step()
    assert first["ordinal"] == 0
    assert resumed.state()["draws"] == 1
    assert resumed.state()["work_units"] == 2
    assert resumed.step()["ordinal"] == 1
    with pytest.raises(ValueError):
        StreamStore(tmp_path, policy(seed=14))


def test_forbidden_structure_rejection_and_draw_exhaustion_are_charged(tmp_path):
    from oeis_learn.generalization.progressive.generator import draw
    from oeis_learn.generalization.progressive.stream import StreamStore

    p = policy(max_draws=1)
    p["forbidden_structures"] = [draw(p, 0)["structure"]]
    s = StreamStore(tmp_path, p)
    value = s.step()
    assert s.state()["draws"] == 1
    with pytest.raises(TimeoutError):
        s.step()
    assert value["status"] == "rejected" and value["reason"] == "held_out_structure"


def test_shard_crash_recovery_and_unqualified_rows_never_enter_learner(tmp_path):
    from oeis_learn.generalization.progressive.stream import StreamStore, load_shard

    s = StreamStore(tmp_path, policy(shard_rows=2))
    s.step()
    s.step()
    with pytest.raises(InterruptedError):
        s.seal_shard(fault="after_shard_write")
    r = StreamStore(tmp_path, policy(shard_rows=2))
    assert r.state()["sealed_shards"] == 1
    assert r.step()["ordinal"] == 2
    with pytest.raises(ValueError, match="native"):
        load_shard(tmp_path, r.shards()[0])


def test_byte_work_and_row_caps_are_binding(tmp_path):
    from oeis_learn.generalization.progressive.stream import StreamStore

    s = StreamStore(tmp_path, policy(shard_rows=1, max_work_units=1))
    s.step()
    with pytest.raises(TimeoutError):
        s.step()
    shard = s.seal_shard()
    assert shard["payload"]["accepted"] <= 1
    assert shard["payload"]["bytes"] <= 256 << 20


def test_control_taint_and_parameter_count_keep_whole_literal_rows():
    from oeis_learn.generalization.progressive.generator import draw, structure_id
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw

    source = "(program (local $b0 bigint) (local $c0 control) (set $b0 (int 2)) (loop $c0 (to_control (int 3)) (set $b0 (add (get $b0) $index))) (return (get $b0)))"
    value = dict(draw(policy(), 0), source=source, structure=structure_id(source))
    row, reason = verify_draw(value)
    assert reason is None
    assert row["values"][:3] == ["2", "5", "8"]
    assert row["target_source"] == source
    assert row["target_disposition"] == "unsupported_control_dependency"
    assert row["control_events"]["loop:3"] == 100
    assert row["native_verified"] is False


def test_no_scope_change_or_journal_truncation_on_resume(tmp_path):
    from oeis_learn.generalization.progressive.stream import StreamStore

    s = StreamStore(tmp_path, policy())
    s.step()
    with pytest.raises(ValueError):
        StreamStore(tmp_path, policy(depth=4))
    (tmp_path / "events.jsonl").write_text("")
    with pytest.raises(ValueError, match="truncated"):
        s.state()


def test_finite_signature_exclusion_does_not_claim_all_domain_equivalence(tmp_path):
    from oeis_learn.generalization.progressive.generator import draw, structure_id
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from oeis_learn.synthesis.reference import evaluate

    zero = "(program (return (int 0)))"
    padded = "(program (return (select (lt $index (int 100)) (int 0) $index)))"
    a, _ = verify_draw(dict(draw(policy(), 0), source=zero, structure=structure_id(zero)))
    b, _ = verify_draw(dict(draw(policy(), 0), source=padded, structure=structure_id(padded)))
    assert a["behavior"] == b["behavior"] and a["structure"] != b["structure"]
    assert evaluate(zero, ["100"]).values != evaluate(padded, ["100"]).values
    assert "finite 100-term" in b["detector_limit"]


def test_prefix_and_byte_exclusions_preserve_rejected_draws(tmp_path):
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.progressive.generator import draw
    from oeis_learn.generalization.progressive.stream import StreamStore
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw

    p = policy(max_draws=1)
    row, _ = verify_draw(draw(p, 0))
    p["forbidden_prefixes"] = [digest(row["values"][1:21])]
    s = StreamStore(tmp_path / "excluded", p)
    assert s.step()["reason"] == "finite_prefix_or_shift_exclusion"
    tiny = StreamStore(tmp_path / "bytes", policy(shard_bytes=1, max_draws=1))
    assert tiny.step()["reason"] == "shard_byte_limit"
    assert tiny.seal_shard()["payload"]["bytes"] == 0
