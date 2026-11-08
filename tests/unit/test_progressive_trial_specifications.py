"""T038 contracts for T041–T044 durable proposal/trial/pruning implementation."""

import importlib.util

import pytest

pending = importlib.util.find_spec("oeis_learn.generalization.progressive.trials") is None
pytestmark = pytest.mark.xfail(
    pending, strict=True, reason="T041–T044 trial execution remains outside T025–T039"
)


def test_proposal_is_durable_before_dispatch_and_reused_after_interruption(tmp_path):
    from oeis_learn.generalization.progressive.trials import TrialJournal

    journal = TrialJournal(tmp_path, study="sha256:" + "a" * 64)
    with pytest.raises(ValueError):
        journal.dispatch(0)
    proposal = journal.propose(0, parameters={"lr_multiplier": 1}, seed=13)
    assert journal.proposal(0) == proposal
    recovered = TrialJournal(tmp_path, study="sha256:" + "a" * 64)
    assert recovered.proposal(0) == proposal
    with pytest.raises(ValueError):
        recovered.propose(0, parameters={"lr_multiplier": 3}, seed=13)


def test_conflicting_tell_and_confirmation_sampler_feedback_are_rejected(tmp_path):
    from oeis_learn.generalization.progressive.trials import TrialJournal

    j = TrialJournal(tmp_path, study="sha256:" + "a" * 64)
    j.propose(0, parameters={"lr_multiplier": 1}, seed=13)
    j.dispatch(0)
    result = {
        "status": "COMPLETE",
        "tokens": 4000000,
        "elapsed_us": 1000000,
        "value": 0.2,
        "role": "development",
    }
    assert j.tell(0, result) == j.tell(0, result)
    with pytest.raises(ValueError):
        j.tell(0, result | {"value": 0.3})
    j.propose(1, parameters={"lr_multiplier": 0.3}, seed=14)
    j.dispatch(1)
    with pytest.raises(ValueError):
        j.tell(1, result | {"role": "confirmation"})


@pytest.mark.parametrize(
    "coordinate,references,control,expected",
    [
        (999999, 6, False, False),
        (1000000, 4, False, False),
        (1000000, 5, False, True),
        (2000000, 8, True, False),
    ],
)
def test_grace_comparable_references_and_controls(coordinate, references, control, expected):
    from oeis_learn.generalization.progressive.trials import should_prune

    history = [
        {"coordinate": coordinate, "value": 0.5, "status": "COMPLETE", "role": "development"}
        for _ in range(references)
    ]
    assert (
        should_prune(value=0.1, coordinate=coordinate, references=history, control=control)
        == expected
    )


def test_wall_limited_partial_cannot_restart_or_win_equal_token_ranking(tmp_path):
    from oeis_learn.generalization.progressive.trials import TrialJournal

    j = TrialJournal(tmp_path, study="sha256:" + "a" * 64)
    j.propose(0, parameters={"lr_multiplier": 1}, seed=13)
    j.dispatch(0)
    j.tell(
        0,
        {
            "status": "PARTIAL",
            "tokens": 1000000,
            "elapsed_us": 7200000000,
            "value": 0.9,
            "role": "development",
        },
    )
    with pytest.raises(ValueError):
        j.dispatch(0)
    assert 0 not in j.equal_token_eligible(target=4000000)


def test_incomparable_failed_and_confirmation_references_cannot_meet_pruning_quota():
    from oeis_learn.generalization.progressive.trials import should_prune

    refs = [
        {"coordinate": 1000000, "value": 0.5, "status": "COMPLETE", "role": "development"}
        for _ in range(4)
    ]
    refs += [{"coordinate": 2000000, "value": 0.9, "status": "COMPLETE", "role": "development"}] * 5
    refs += [{"coordinate": 1000000, "value": 0.9, "status": "FAILED", "role": "development"}] * 5
    refs += [
        {"coordinate": 1000000, "value": 0.9, "status": "COMPLETE", "role": "confirmation"}
    ] * 5
    assert not should_prune(value=0.1, coordinate=1000000, references=refs, control=False)
