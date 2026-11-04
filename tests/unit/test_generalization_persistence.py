import pytest
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.generalization_persistence import journal,recover,policy,INTERVAL

REF=dict(path='generation/state.json',bytes=1,sha256='sha256:'+'a'*64)


def record(**kw):
    fields=dict(contract='contract',checkpoint=REF,ledger=Ledger(100,consumed=30,updates=3),discarded_tokens=0,
        elapsed_seconds=20.,invocation_base_seconds=0.,closed=False)
    fields.update(kw)
    return journal(**fields)


def test_hourly_policy_and_budget_rollback_are_separate_from_saved_weights():
    assert INTERVAL==3600 and policy({})=='per-update/v1'
    assert policy({'checkpoint_policy':'hourly/v1'})=='hourly/v1'
    ledger=Ledger(100,consumed=10,updates=1)
    result=recover(record(),contract='contract',checkpoint=REF,ledger=ledger,
        discarded_tokens=0,learner_seconds=5,seconds_cap=100,supervised_seconds=25)
    assert result==dict(discarded_tokens=20,learner_seconds=25,rolled_back=True)
    assert ledger.consumed==10 and ledger.updates==1


def test_repeated_recovery_never_refunds_previous_discarded_tokens_or_time():
    r=record(ledger=Ledger(100,consumed=40,updates=4),discarded_tokens=20,invocation_base_seconds=25,elapsed_seconds=40)
    result=recover(r,contract='contract',checkpoint=REF,ledger=Ledger(100,consumed=30,updates=3),
        discarded_tokens=20,learner_seconds=35,seconds_cap=100,supervised_seconds=20)
    assert result['discarded_tokens']==30 and result['learner_seconds']==45


def test_clean_terminal_resume_and_missing_crash_timing():
    live=Ledger(100,consumed=30,updates=3)
    args=dict(contract='contract',checkpoint=REF,ledger=live,discarded_tokens=0,learner_seconds=19,seconds_cap=100)
    assert recover(record(closed=True),**args)['learner_seconds']==20
    assert recover(record(),**args)['learner_seconds']==100


def test_new_latest_before_journal_publication_is_recoverable_but_older_is_not():
    newer=dict(REF,path='new/state.json')
    r=recover(record(),contract='contract',checkpoint=newer,ledger=Ledger(100,consumed=40,updates=4),
        discarded_tokens=0,learner_seconds=23,seconds_cap=100,supervised_seconds=25)
    assert r['discarded_tokens']==0 and not r['rolled_back']
    with pytest.raises(ValueError):recover(record(),contract='contract',checkpoint=newer,ledger=Ledger(100,consumed=10,updates=1),
        discarded_tokens=0,learner_seconds=5,seconds_cap=100,supervised_seconds=25)


@pytest.mark.parametrize('defect',['seal','contract','cap','negative','nan','over_cap','closed_later','time_base'])
def test_invalid_accounting_fails_closed(defect):
    r=record()
    if defect=='seal':r['identity']='bad'
    if defect=='contract':r=seal(dict(r,contract='other'))
    if defect=='cap':r=record(ledger=Ledger(101,consumed=30,updates=3))
    if defect=='negative':r=seal(dict(r,discarded_tokens=-1))
    if defect=='nan':r=dict(r,elapsed_seconds=float('nan'))
    if defect=='over_cap':r=seal(dict(r,discarded_tokens=90))
    if defect=='closed_later':r=record(closed=True)
    if defect=='time_base':r=seal(dict(r,invocation_base_seconds=30))
    with pytest.raises(ValueError):recover(r,contract='contract',checkpoint=REF,ledger=Ledger(100,consumed=10,updates=1),
        discarded_tokens=0,learner_seconds=5,seconds_cap=100,supervised_seconds=25)


def test_bounded_progress_preserves_whole_run_cost_and_loss_trend():
    from oeis_learn.pipeline_validation.generalization_training import TrainingTiming
    timing=TrainingTiming();timing.observe(0,0,0.)
    for i in range(1,2005):
        timing.record_update(dict(loss=2.,gradient_norm=1.))
        timing.observe(i,i*100,float(i))
    assert len(timing.windows)==64 and len(timing.trend)==128
    assert sum(w['updates'] for w in timing.trend)==2000
    assert sum(w['tokens'] for w in timing.trend)==200000
    assert sum(w['seconds'] for w in timing.trend)==2000
    assert all(w['mean_loss']==2. for w in timing.trend)
    assert timing.last_loss==2. and timing.last_gradient_norm==1.


def test_supervisor_time_requires_acknowledged_original_termination():
    from oeis_learn.pipeline_validation.generalization_persistence import supervised_elapsed
    process=dict(outcome='FAIL',returncode=-9,cleanup_failure=None,elapsed_seconds=125.)
    assert supervised_elapsed(process,dict(within_bound=True))==125.
    assert supervised_elapsed(process,dict(within_bound=False)) is None
    assert supervised_elapsed(dict(process,returncode=None),dict(within_bound=True)) is None
    assert supervised_elapsed(None,None) is None
    with pytest.raises(ValueError):supervised_elapsed(dict(process,elapsed_seconds=float('inf')),dict(within_bound=True))
