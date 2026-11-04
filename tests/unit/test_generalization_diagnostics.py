"""Missing feedback and observer schedules are not mastery or kernel credit."""
import copy,json
import pytest


def test_reward_budget_and_two_censored_rewards_pause_persistently():
    from oeis_learn.generalization.diagnostics import FeedbackCoverage
    s=FeedbackCoverage(['a','b'])
    s.reward(complete=False);assert not s.pauses
    s.reward(complete=True);assert s.censored_rewards==0
    s.reward(complete=False);s.reward(complete=False)
    assert s.pauses==['censored_rewards']
    s.reward(complete=True);assert s.pauses==['censored_rewards']
    restored=FeedbackCoverage.restore(json.loads(json.dumps(s.snapshot())),families=['a','b'])
    assert restored.snapshot()==s.snapshot()
    s=FeedbackCoverage(['a','b']);s.reward(complete=False,budget_exhausted=True)
    assert 'required_reward_budget' in s.pauses


def test_anchor_grade_coverage_and_two_missing_semantic_rotations():
    from oeis_learn.generalization.diagnostics import FeedbackCoverage
    s=FeedbackCoverage(['a','b'])
    s.review_block(eligible=8,valid=0);assert not s.pauses
    s.review_block(eligible=0,valid=0);assert s.invalid_review_blocks==0
    s.review_block(eligible=2,valid=0);s.review_block(eligible=1,valid=0)
    assert s.pauses==['invalid_review_grades']
    s.semantic_rotation({'a':False,'b':True});assert not s.retention_incomplete
    s.semantic_rotation({'a':False,'b':True});assert s.retention_incomplete=={'a'}
    with pytest.raises(ValueError):s.semantic_rotation({'a':True})
    with pytest.raises(ValueError):s.review_block(eligible=1,valid=2)
