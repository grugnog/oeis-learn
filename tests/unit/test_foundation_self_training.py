import pytest
from oeis_learn.rl.self_training import TrainingTruth, SelfTrainingRounds
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest


def test_training_truth_never_exposes_evaluation_continuation():
    partition = {"version": "training-truth/v1", "partition": "train", "records": [list(range(24))]}
    truth = TrainingTruth(
        partition, expected_sha256=digest(partition), reserved_prefixes=frozenset()
    )
    assert truth.visible() == (tuple(range(20)),)
    assert truth.continuation(list(range(20))) == list(range(24))
    with pytest.raises(ValueError, match="unavailable"):
        truth.continuation([99] * 20)
    reserved = {digest([str(v) for v in range(20)])}
    with pytest.raises(ValueError, match="reserved"):
        TrainingTruth(partition, expected_sha256=digest(partition), reserved_prefixes=reserved)
    partition["partition"] = "development"
    with pytest.raises(ValueError, match="training partition"):
        TrainingTruth(partition, expected_sha256=digest(partition), reserved_prefixes=())


def test_round_snapshot_does_not_change_before_explicit_boundary(tmp_path):
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.data.structural_splits import StructuralHoldouts

    partition = {"version": "training-truth/v1", "partition": "train", "records": [list(range(24))]}
    truth = TrainingTruth(partition, expected_sha256=digest(partition), reserved_prefixes=())
    archive = ProgramArchive(tmp_path, holdouts=StructuralHoldouts())
    rounds = SelfTrainingRounds(archive, truth)
    state = rounds.state_dict()
    restored = SelfTrainingRounds(archive, truth)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert rounds.near_misses(list(range(20))) == ()
    with pytest.raises(ValueError):
        rounds.search([9] * 20, None, None)

    class Runner:
        def run_single(self, *args, **kwargs):
            raise AssertionError("empty archive must not execute")

    rounds.next_round(Runner())
    assert rounds.round == 1 and rounds.snapshot == state["snapshot"]
