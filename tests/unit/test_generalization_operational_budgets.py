"""Independent retained-model-size accounting without allocating model tensors."""
import pytest
from oeis_learn.generalization.resources import checkpoint_reservation
from oeis_learn.generalization.budgets import reserve_storage


def test_acceptance_peak_counts_resume_and_single_projection():
    # Four retained runs with initial/latest/one milestone, one resume projection
    # and one atomic generation: 4*3 + 3 + 1 = 16 full states. Ten weight copies
    # cover origins plus live initializer, eval and proof views.
    result=checkpoint_reservation(snapshot_bytes=1393781168,weights_bytes=421525760,
        milestones=1,arms=2,seeds=1,acceptance_replay=True,selection=True)
    expected=16*1393781168+10*421525760
    assert result['required_checkpoint_bytes']==expected
    assert result['learner_runs']==4
    assert result['phase_generations']=={'training':16,'proof':14}
    assert expected<26<<30
    assert reserve_storage(expected,1,expected)==expected
    with pytest.raises(ValueError):reserve_storage(expected,1,expected-1)


def test_research_selection_and_more_seeds_reserve_their_live_copies():
    result=checkpoint_reservation(snapshot_bytes=100,weights_bytes=10,milestones=2,arms=2,seeds=3,selection=True)
    assert result['phase_generations']=={'training':29,'proof':26,'selection':34}
    assert result['required_checkpoint_bytes']==34*100+12*10
