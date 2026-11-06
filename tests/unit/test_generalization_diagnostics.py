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


def test_common_panels_shadow_observers_and_optional_d_costs():
    from oeis_learn.generalization.diagnostics import observer_schedule,observation_forecast
    families=[f'f{i:02}' for i in range(32)];anchors=[f'a{i:03}' for i in range(128)]
    def schedule(i,arm,**kw):return observer_schedule(i,arm=arm,families=families,anchors=anchors,seed=17,**kw)
    assert [schedule(i*256,'B')['common']['semantic_families'] for i in range(8)]==[families[i:i+4] for i in range(0,32,4)]
    assert schedule(1,'C')['common'] is None
    assert schedule(1,'C',final=True)['common']['loss_probes']==256
    for arm,rewards,grades in [('B',3,8),('C',3,8),('D',4,0)]:
        cycle=[schedule(i,arm) for i in range(32)]
        assert sum(s['reward_start'] is not None for s in cycle)==rewards
        assert sum(s['grade_opportunity'] is not None for s in cycle)==grades
        forecast=observation_forecast(32,arm=arm)
        assert forecast['reward_probe_evaluations']==rewards*4
        assert forecast['grade_opportunities']==grades
        assert forecast['common_loss_probes']==512
        assert forecast['semantic_attempts']==64 and forecast['semantic_seconds']==320
    b=schedule(24,'B')['grade_opportunity'];assert b['anchor']=='a000' and b['shadow']
    assert schedule(31,'B')['grade_opportunity']['anchor']=='a007'
    assert schedule(56,'B')['grade_opportunity']['anchor']=='a008'
    assert schedule(24,'C')['grade_opportunity']['anchor'] is None
    assert schedule(256,'B')['common']==schedule(256,'C')['common']==schedule(256,'D')['common']
    assert all(s['common']['kernel_credit'] is False for s in [schedule(0,'B'),schedule(0,'C')])
    assert observation_forecast(256,arm='C')['common_loss_probes']==512  # final/cadence deduplicate
    assert observation_forecast(1,arm='C')['reward_probe_evaluations']==4  # reserve incomplete block too
