import pytest
from oeis_learn.fresh.learning_policy import Curriculum, checkpoint_key, fixed_panel, score_panel
from oeis_learn.fresh.comparison import promotion, family_bootstrap
from oeis_learn.fresh.cost_policy import screen_cost
from oeis_learn.fresh.replay import SynthesisReplay


def test_denominators_curriculum_two_validations_and_retained_practice():
    ids = tuple(str(i) for i in range(100))
    report = score_panel(ids, {k: dict(complete=True, verified=i < 80) for i, k in enumerate(ids)})
    c = Curriculum(validation_every=100)
    assert not c.observe(report, update=100)
    assert c.observe(report, update=200)
    assert c.stage == 1 and c.earlier_count(8) == 2
    restored = Curriculum.restore(c.snapshot())
    assert restored.stage == 1
    incomplete = score_panel(ids, {"0": dict(complete=True, verified=True)})
    assert incomplete["accuracy"] == 0.01
    assert not restored.observe(incomplete, update=300)
    a = checkpoint_key(report, loss=2, loss_complete=True, update=100)
    b = checkpoint_key(report, loss=1, loss_complete=False, update=200)
    assert a < b
    panel = fixed_panel([dict(id=str(i), stage=i % 4) for i in range(128)])
    assert len(panel) == 32 and len(set(panel)) == 32


def test_transfer_gate_exact_eight_of_128_and_every_comparator():
    full = [
        dict(
            seed=s,
            success=40,
            complete=128,
            denominator=128,
            verified_admissions=True,
            disagreement=False,
            metric="continuation_accuracy_100",
            portfolio="exact_equal_budget",
        )
        for s in (20261016, 20261017, 20261018)
    ]
    controls = {k: [dict(r, success=32) for r in full] for k in ("shuffled", "absent", "template")}
    assert promotion(
        full, controls, [dict(r, success=40) for r in full], [dict(r, success=44) for r in full]
    )["promote"]
    controls["template"][0]["success"] = 33
    assert not promotion(
        full, controls, [dict(r, success=40) for r in full], [dict(r, success=44) for r in full]
    )["promote"]
    controls["template"][0]["success"] = 32
    full[0]["complete"] = 121
    assert not promotion(
        full, controls, [dict(r, success=40) for r in full], [dict(r, success=44) for r in full]
    )["promote"]
    a = family_bootstrap({"f1": [1, 0, 1], "f2": [0, -1, 0]}, seed=17)
    assert a == family_bootstrap({"f1": [1, 0, 1], "f2": [0, -1, 0]}, seed=17)
    assert a["resamples"] == 2000


def test_complete_screen_cost_no_silent_dropping():
    cost = screen_cost(
        hidden_scoring_p95=0.1,
        checkpoint_p95=0.2,
        training_p95=0.1,
        teacher_forcing_p95=0.01,
        generation_p95=0.05,
        fit_p95=0.1,
        reference_p95=0.1,
    )
    assert cost["projected_seconds"] > 0 and cost["within_budget"]
    costly = screen_cost(
        hidden_scoring_p95=0.1,
        checkpoint_p95=0.2,
        training_p95=1,
        teacher_forcing_p95=1,
        generation_p95=10,
        fit_p95=2,
        reference_p95=2,
    )
    assert not costly["within_budget"]


def test_replay_training_only_verified_resumable_deterministic_replacement():
    replay = SynthesisReplay(
        capacity=2,
        track="generic_only",
        representation_comparison=False,
        training_targets=("train1", "train2", "train3"),
    )
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.synthesis.admission import source_identity

    source = "(program (return $index))"

    def record(target):
        verification = dict(
            target=target,
            source_sha256=source_identity(source),
            runtime_sha256="sha256:" + "1" * 64,
            reference_sha256="sha256:" + "2" * 64,
            indices=list(map(str, range(100))),
            values=list(map(str, range(100))),
            admitted=True,
        )
        return dict(
            target=target,
            split="training",
            track="generic_only",
            source=source,
            original_source=source,
            assignment={},
            profile="bigint-body/v1",
            verification=verification,
            independent=True,
            tokens=encode_body(source),
        )

    row = record("train1")
    for target in ("train1", "train2", "train3"):
        replay.admit(record(target))
    with pytest.raises(ValueError):
        replay.admit(record("dev1"))
    assert len(replay.records) == 2
    clone = SynthesisReplay.restore(replay.snapshot())
    assert clone.snapshot() == replay.snapshot()
    assert len(clone.sample(8)) == 2
    for change in [dict(split="final"), dict(track="assisted"), dict(independent=False)]:
        with pytest.raises(ValueError):
            replay.admit(dict(row, **change))
    with pytest.raises(ValueError):
        SynthesisReplay(representation_comparison=True).admit(row)


def test_stopping_resume_and_checkpoint_panel_identity():
    from oeis_learn.fresh.learning_policy import Stopping, choose_checkpoint

    stopping = Stopping()
    assert not stopping.observe(1)
    for _ in range(7):
        assert not stopping.observe(1)
    assert Stopping.restore(stopping.snapshot()).observe(1)
    ids = ("a", "b")
    report = score_panel(ids, {"a": dict(complete=True, verified=True)})
    candidate = dict(checkpoint="one", visible=report, loss=1.0, loss_complete=True, update=100)
    assert choose_checkpoint([candidate], panel_sha256=report["panel_sha256"]) == "one"
    with pytest.raises(ValueError, match="panel"):
        choose_checkpoint([candidate], panel_sha256="changed")
