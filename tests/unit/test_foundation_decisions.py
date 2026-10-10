from copy import deepcopy
import pytest

from oeis_learn.experiments.decisions import select_trigger, paired_disposition, frozen_protocol


def test_priority_missing_and_nontriggered_dispositions():
    diagnostic = dict(
        encoder_collisions=0,
        new_structures=100,
        plateau_improvements=[0.005] * 3,
        valid_fraction=0.5,
        reward_variance_fraction=0.3,
    )
    selected, states = select_trigger(diagnostic)
    assert selected == "self_training"
    assert states["encoder"] == "not_triggered"
    assert states["grpo"] == "baseline_retained"
    assert states["search"] == "inconclusive"
    assert select_trigger({}) == (
        None,
        dict.fromkeys(frozen_protocol()["priority"], "inconclusive"),
    )


def test_plateau_exact_three_windows_and_capacity_minimum():
    diag = dict(
        encoder_collisions=0,
        new_structures=100,
        plateau_improvements=[0, 0],
        train_loss=0.6,
        validation_loss=0.62,
        updates=1000,
        diverse_programs=9999,
    )
    selected, states = select_trigger(diag)
    assert selected is None
    assert states["self_training"] == "inconclusive"
    assert states["capacity"] == "not_triggered"


@pytest.mark.parametrize(
    "pairs,expected",
    [
        ([(0, 1)], "baseline_retained"),
        ([(0, 2)], "inconclusive"),
        ([(0, 2), (0, 2), (0, 0)], "adopted"),
        ([(100, 106), (100, 106), (100, 0)], "baseline_retained"),
        ([(100, 105), (100, 100), (100, 100)], "baseline_retained"),
    ],
)
def test_first_pair_zero_baseline_confirmation_and_aggregate(pairs, expected):
    assert paired_disposition(pairs) == expected


def test_manifest_unknown_final_truth_and_wrong_caps_rejected(tmp_path):
    from oeis_learn.evaluation.experiment_runner import validate_manifest, fixture_manifest

    manifest = fixture_manifest()
    validate_manifest(manifest)
    for update in (
        {"final_truth": "private.json"},
        {"training_seconds": 21601},
        {"seeds": [1, 2, 3]},
    ):
        invalid = deepcopy(manifest)
        invalid.update(update)
        with pytest.raises(ValueError):
            validate_manifest(invalid)


def test_dry_run_and_actual_fixture_are_immutable(tmp_path):
    from oeis_learn.evaluation.experiment_runner import measure, fixture_manifest

    manifest = fixture_manifest()
    dry = measure(manifest, tmp_path / "dry", dry_run=True)
    assert dry["status"] == "dry_run" and dry["executed"] is False
    live = measure(manifest, tmp_path / "live")
    assert live["executed"] and live["qualified"] is False
    assert live["measurements"]["repeats"] == 30
    assert all(d["status"] == "inconclusive" for d in live["decisions"])
    assert measure(manifest, tmp_path / "live") == live
    invalid = deepcopy(manifest)
    invalid["sources"][0] = "i256.const 7"
    with pytest.raises(ValueError, match="identity|immutable"):
        measure(invalid, tmp_path / "live")


def test_pilot_schedule_charges_crash_and_does_not_repeat_arm(tmp_path):
    from oeis_learn.experiments.decisions import PilotSchedule

    tick = [10]
    pilot = PilotSchedule(tmp_path, "manifest", clock=lambda: tick[0])
    assert pilot.issue()["seconds"] == 1800
    tick[0] += 1800 * 10**9
    pilot.complete()
    pilot.select({"encoder_collisions": 1})
    first = pilot.issue()
    assert first["seed"] == 7001 and first["arm"] == "baseline"
    tick[0] += 3300 * 10**9
    pilot.complete(solved_groups=0)
    assert pilot.issue()["arm"] == "challenger"
    tick[0] += 100 * 10**9
    resumed = PilotSchedule(tmp_path, "manifest", clock=lambda: tick[0])
    assert resumed.charged == 5200 * 10**9 and resumed.interrupted
    with pytest.raises(ValueError, match="interrupted"):
        resumed.issue()


def test_pilot_first_pair_failure_prevents_second_contrast(tmp_path):
    from oeis_learn.experiments.decisions import PilotSchedule

    tick = [0]
    pilot = PilotSchedule(tmp_path, "manifest", clock=lambda: tick[0])
    pilot.issue()
    tick[0] += 1800 * 10**9
    pilot.complete()
    pilot.select({"encoder_collisions": 1})
    for count in (0, 1):
        pilot.issue()
        tick[0] += 3300 * 10**9
        pilot.complete(solved_groups=count)
    with pytest.raises(ValueError, match="stop rule"):
        pilot.issue()
    with pytest.raises(ValueError, match="one post"):
        pilot.select({"encoder_collisions": 1})
    assert pilot.charged == 8400 * 10**9
