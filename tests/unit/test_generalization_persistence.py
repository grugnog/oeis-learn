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


def pac_saved():
    from tests.unit.test_generalization_curriculum_state import state,begin,update
    from oeis_learn.generalization.state import pac_snapshot
    from oeis_learn.pipeline_validation.identity import digest
    s=state();begin(s);update(s)
    states=dict(weights={'fixture':1},optimizer={},scheduler={},rng=dict(python=[],numpy=[],torch=[],device=[]),
                data=dict(cursor=s.updates,pending_batch=s.pending),replay={},history={},pins={},selection={})
    identity={k:digest(k) for k in ('contract','source','environment')};identity['inventory']=s.inventory
    return s,pac_snapshot(states,s,**identity),identity


def test_pac_complete_generation_hashes_every_component(tmp_path):
    import json
    from oeis_learn.generalization.checkpoints import pac_generation,verify_pac_generation,PAC_COMPONENTS
    from oeis_learn.pipeline_validation.identity import file_hash,digest
    refs={}
    for name in PAC_COMPONENTS:
        p=tmp_path/(name+'.json');p.write_text(json.dumps({'fixture':name}))
        refs[name]=dict(path=p.name,sha256=file_hash(p),bytes=p.stat().st_size)
    identity={k:digest(k) for k in ('contract','inventory','source','environment')}
    m=pac_generation(refs,root=tmp_path,generation='g1',model_update=1,**identity)
    assert verify_pac_generation(m,root=tmp_path,**identity)==m
    for name in refs:
        missing=dict(refs);missing.pop(name)
        with pytest.raises(ValueError):pac_generation(missing,root=tmp_path,generation='g2',model_update=1,**identity)
    (tmp_path/'optimizer.json').write_text('tampered')
    with pytest.raises(ValueError):verify_pac_generation(m,root=tmp_path,**identity)


def test_pac_restore_identity_and_newer_journal_only_debits_discarded_work():
    from oeis_learn.generalization.state import pac_restore,pac_recover
    from oeis_learn.pipeline_validation.identity import seal,digest
    s,saved,identity=pac_saved()
    restored=pac_restore(saved,**identity)
    assert restored['curriculum'].snapshot()==s.snapshot()
    for key in identity:
        with pytest.raises(ValueError):pac_restore(saved,**dict(identity,**{key:digest('different')}))
    j=seal(dict(version='pac-cost-journal/v1',base=saved['identity'],segment=s.segment,sequence=5,charged_tokens=68,charged_seconds=10.))
    got=pac_recover(saved,j,new_segment='s1',**identity)
    state=got['curriculum']
    assert state.updates==1 and state.charged_tokens==68 and state.discarded_tokens==51
    assert state.charged_seconds==10 and state.kernel.snapshot()==s.kernel.snapshot()
    assert state.streams['a'].snapshot()==s.streams['a'].snapshot()
    assert state.segment=='s1' and state.plan_batch(2)['event_id'].startswith('s1:')
    for bad in (dict(j,kernel={'invented':'newer'}),seal(dict(j,base=digest('stale')))):
        with pytest.raises(ValueError):pac_recover(saved,bad,new_segment='s1',**identity)
    with pytest.raises(ValueError):pac_recover(saved,j,new_segment='s0',**identity)


def test_pac_crash_after_saved_reservation_charges_recomputed_batch_again():
    from oeis_learn.generalization.state import pac_snapshot,pac_recover
    s,saved,identity=pac_saved()
    s.plan_batch(2);s.reserve(17)
    states=saved['states'];states['data']['pending_batch']=s.pending
    saved=pac_snapshot(states,s,**identity)
    j=seal(dict(version='pac-cost-journal/v1',base=saved['identity'],segment=s.segment,sequence=1,charged_tokens=34,charged_seconds=2.))
    got=pac_recover(saved,j,new_segment='s1',**identity)['curriculum']
    assert got.updates==1 and got.discarded_tokens==17 and got.pending['phase']=='planned'
    assert got.pending['event_id'].startswith('s1:')
    assert got.reserve(17) and got.charged_tokens==51
