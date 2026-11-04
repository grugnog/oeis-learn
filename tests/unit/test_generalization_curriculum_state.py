"""Model-bound pure transaction oracle; installed tensor proof belongs to C."""
import copy,json
import pytest
from oeis_learn.pipeline_validation.identity import digest


def state(cap=100000,arm='C'):
    from oeis_learn.generalization.curriculum_state import CurriculumState
    families={f:list(f+str(i) for i in range(16)) for f in ('a','b')}
    probes={f:[f+'p'+str(i) for i in range(8)] for f in families}
    anchors={f+str(i):f for f in families for i in range(4)}
    return CurriculumState(families,probes,anchors,seed=11,arm=arm,token_cap=cap,inventory=digest([families,probes,anchors]),segment='s0')


def panel(s,value):
    b=s.block
    return dict(role='training',block=b['id'],panel=digest(b['probe_ids']),probe_ids=b['probe_ids'],
                model_update=s.updates,loss_sum=value*4,tokens=4)


def begin(s):
    s.start_block();s.record_pre(panel(s,2))


def update(s,grade=None):
    batch=s.plan_batch(2);assert s.reserve(17,grade=grade)
    s.commit(batch['event_id'],success=True,elapsed_seconds=.5)
    return batch


def finish(s):s.record_post(panel(s,1));return s.finish_feedback()


def test_reservation_commit_pending_feedback_and_no_intervening_gradient():
    s=state();begin(s)
    b=s.plan_batch(2);s.reserve(17)
    assert s.charged_tokens==17 and s.updates==0
    with pytest.raises(ValueError):s.plan_batch(2)
    with pytest.raises(ValueError):s.commit('wrong',success=True)
    s.commit(b['event_id'],success=True)
    with pytest.raises(ValueError):s.commit(b['event_id'],success=True)
    with pytest.raises(ValueError):s.start_block()
    for _ in range(7):update(s)
    assert s.updates==8 and s.charged_tokens==136 and s.kernel.valid_updates==0
    with pytest.raises(ValueError):s.plan_batch(2)
    wrong=panel(s,1);wrong['probe_ids']=list(reversed(wrong['probe_ids']))
    with pytest.raises(ValueError):s.record_post(wrong)
    wrong=panel(s,1);wrong['model_update']-=1
    with pytest.raises(ValueError):s.record_post(wrong)
    result=finish(s)
    assert result['raw_gain']==1 and result['work_gain']==1/136
    assert s.kernel.valid_updates==1 and s.scaler.seen==1
    with pytest.raises(ValueError):s.finish_feedback()


def test_failed_optimizer_no_anchor_promotion_and_terminal_partial():
    s=state();begin(s);batch=s.plan_batch(2);s.reserve(17)
    queue=copy.deepcopy(s.review.snapshot());s.commit(batch['event_id'],success=False,elapsed_seconds=.2)
    assert s.updates==0 and s.charged_tokens==17 and s.review.snapshot()==queue
    assert 'optimizer_failure_requires_restore' in s.pauses
    with pytest.raises(ValueError):s.plan_batch(2)
    s=state(cap=60);begin(s)
    for _ in range(3):update(s)
    s.plan_batch(2);assert not s.reserve(17)
    assert s.terminal and s.updates==3 and s.charged_tokens==51 and s.kernel.valid_updates==0
    assert s.last_observation['status']=='terminal_partial'


def test_missing_panel_retains_debits_and_two_censors_pause():
    s=state()
    for _ in range(2):
        s.start_block();s.record_pre(None)
        for _ in range(8):update(s)
        s.record_post(None);result=s.finish_feedback()
        assert result['status']=='censored'
    assert s.charged_tokens==272 and s.updates==16 and s.kernel.valid_updates==s.scaler.seen==0
    assert 'censored_rewards' in s.pauses


def test_snapshot_closed_identity_and_pending_phase_roundtrip():
    from oeis_learn.generalization.curriculum_state import CurriculumState
    s=state();begin(s)
    for _ in range(3):update(s)
    snap=json.loads(json.dumps(s.snapshot()))
    restored=CurriculumState.restore(snap,inventory=s.inventory)
    assert restored.snapshot()==s.snapshot()
    for _ in range(5):assert update(restored)==update(s)
    assert finish(restored)==finish(s) and restored.snapshot()==s.snapshot()
    for field,value in [('version','visit-scheduler/v1'),('inventory',digest('other')),('updates',999)]:
        bad=copy.deepcopy(snap);bad[field]=value
        with pytest.raises(ValueError):CurriculumState.restore(bad,inventory=s.inventory)
    bad=copy.deepcopy(snap);bad['kernel']['valid_updates']+=1
    with pytest.raises(ValueError):CurriculumState.restore(bad,inventory=s.inventory)


def test_pure_binding_rejects_newer_policy_on_older_model():
    from oeis_learn.generalization.state import bind_pac_state
    s=state();begin(s);batch=s.plan_batch(2);s.reserve(17)
    assert bind_pac_state(s.snapshot(),inventory=s.inventory,model_update=0,pending_batch=s.pending).snapshot()==s.snapshot()
    for update,pending in [(1,s.pending),(0,None)]:
        with pytest.raises(ValueError):bind_pac_state(s.snapshot(),inventory=s.inventory,model_update=update,pending_batch=pending)
