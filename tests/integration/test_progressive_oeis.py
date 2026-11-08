"""One-cycle mechanics; native rows remain mandatory in the qualification image."""

import pytest


def test_cycle_rejects_second_cycle_and_foreign_track_before_dispatch(tmp_path):
    from oeis_learn.pipeline_validation.progressive_oeis import validate_cycle
    from tests.unit.test_progressive_freeze import campaign
    from oeis_learn.pipeline_validation.identity import seal

    c = campaign()
    with pytest.raises(ValueError, match="OEIS"):
        validate_cycle(tmp_path, c, {})
    c = seal(dict(c, track="oeis_derived"))
    with pytest.raises(ValueError):
        validate_cycle(tmp_path, c, dict(version="progressive-oeis-cycle/v1", cycle=1))


def test_role_input_override_is_not_a_json_permission(tmp_path):
    from oeis_learn.pipeline_validation.progressive_oeis import check_derived

    with pytest.raises(ValueError):
        check_derived(dict(campaign="fake"), campaign={}, task={}, operation="train")


def test_installed_cpu_replay_and_control_update_restore(tmp_path):
    """Precommitted diagnostic admissions exercise real updates, not native acceptance."""
    from tests.helpers.progressive_oeis_fixture import fixture
    from tests.unit.test_progressive_archive import admitted
    from oeis_learn.generalization.progressive.archive import Archive
    from oeis_learn.generalization.progressive.replay import freeze_replay
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import load, reference, atomic_write, seal
    from oeis_learn.pipeline_validation.progressive_oeis import Cycle
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.generalization.progressive.projections import describe
    from oeis_learn.generalization.progressive.recovery import Recovery
    import torch
    from oeis_learn.generalization.progressive.models import construct

    c, m, request, plan = fixture(tmp_path)
    journal = BudgetJournal(
        tmp_path / "run-state/discovery-accounting" / c["identity"][7:],
        c["identity"],
        c["payload"]["caps"],
    )
    cycle = Cycle(tmp_path, tmp_path / "cycle", c, m, request, journal)
    archive = Archive(tmp_path / "precommitted-fixture-archive")
    cut = archive.publish([admitted("a", "first"), admitted("b", "second")])
    for entry in plan["training"][:2]:
        task = cycle.tasks[entry["task"]]
        channel = load(tmp_path / entry["channel"])
        p = load(tmp_path / channel["data"]["artifact"]["path"])
        e, d, _ = construct(p["profile"], device="cpu")
        original = {k: v.detach().clone() for k, v in e.state_dict().items()}
        replay = freeze_replay(archive, cut, arm=task["payload"]["arm"])
        p = seal(dict(p, oeis_replay=replay))
        path = tmp_path / (entry["task"] + "-resolved.json")
        atomic_write(path, p)
        routed = cycle.channel(
            "training_request", "learner", dict(channel["data"], artifact=reference(tmp_path, path))
        )
        result = cycle.cpu_phase(
            "train",
            {"training": [describe(tmp_path, routed)]},
            tmp_path / ("update-" + entry["task"]),
            task,
            90,
        )
        assert result["mechanical_status"] == "COMPLETE", result
        state = tmp_path / "run-state/discovery-learning" / c["identity"][7:] / task["identity"][7:]
        with Recovery(journal, task, compatibility(p), state_root=state) as owner:
            restored = owner.restore()
            assert any(
                not torch.equal(original[k], v)
                for k, v in restored["states"]["weights"]["encoder"].items()
            )
            assert restored["ledger"].updates == 1
            mix = restored["states"]["replay"]
            assert mix["oeis_tokens"] + mix["synthetic_tokens"] == restored["ledger"].consumed
            assert (mix["oeis_tokens"] > 0) == (task["payload"]["arm"] == "replay")
            assert restored["cuts"]["archive"] == (
                cut["identity"] if task["payload"]["arm"] == "replay" else None
            )
            assert owner.restore()["states"]["replay"] == mix


@pytest.mark.parametrize("low", [False, True])
def test_real_native_search_admit_update_evaluate_cycle(tmp_path, fitting_client, low):
    from tests.helpers.progressive_oeis_fixture import fixture
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation.progressive_commands import oeis_cycle

    c, m, request, plan = fixture(tmp_path, low=low)
    journal = BudgetJournal(
        tmp_path / "run-state/discovery-accounting" / c["identity"][7:],
        c["identity"],
        c["payload"]["caps"],
    )
    result = oeis_cycle(tmp_path, tmp_path / "cycle", c, m, request, journal)
    assert result["mechanical_status"] == "COMPLETE", result
    audit = load(tmp_path / "cycle/host/yield-audit.json")["payload"]
    assert audit["autonomous_discoveries"] == 0
    assert result["metrics"]["autonomous_discoveries"] == 0
    if low:
        assert result["reason"] == "OEIS_YIELD_DEFICIT"
        assert result["next_action"] == "DEFER_BRANCH"
        assert not list((tmp_path / "cycle").glob("phase-*-train"))
    else:
        assert len(result["metrics"]["completed_tasks"]) == 4
        assert audit["accepted_groups"] == 2 and audit["invalid_admissions"] == 0
        assert any("hundred-term" in key for key in audit["rejections"])
        cycle = load(tmp_path / "cycle/host/cycle.json")
        assert not cycle["equal_total_compute"]
        assert set(cycle["comparisons"]) == {"starting", "continued_synthetic"}
        assert (
            result["scientific_status"] == "INCOMPLETE"
        )  # tiny panel never passes a research gate
    before = journal.path.read_bytes()
    with pytest.raises(ValueError, match="already attempted"):
        oeis_cycle(tmp_path, tmp_path / "second-cycle", c, m, request, journal)
    assert journal.path.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["cycle", "seed", "track", "initialization", "channel", "overlap"]
)
def test_frozen_cycle_rejects_changed_scope_before_work(tmp_path, change):
    from tests.helpers.progressive_oeis_fixture import fixture
    from oeis_learn.pipeline_validation.progressive_oeis import validate_cycle
    from oeis_learn.generalization.progressive.contracts import seal, load, reference, atomic_write

    c, m, request, p = fixture(tmp_path)
    if change == "cycle":
        p["cycle"] = 1
    if change == "seed":
        c["payload"]["tasks"][1]["seed"] = 9
    if change == "track":
        c["track"] = "synthetic_strict"
    if change == "initialization":
        c["payload"]["tasks"][1]["initializer"]["compatibility"] = "sha256:" + "1" * 64
    if change == "channel":
        p["training"][0]["channel"] = "cycle-plan.json"
    if change == "overlap":
        panel = load(tmp_path / p["panels"]["transfer"]["path"])
        panel["payload"]["rows"][0]["id"] = "train-0"
        atomic_write(tmp_path / "changed-panel.json", seal(panel))
        p["panels"]["transfer"] = reference(tmp_path, "changed-panel.json")
    with pytest.raises(ValueError):
        validate_cycle(tmp_path, seal(c), seal(p))
