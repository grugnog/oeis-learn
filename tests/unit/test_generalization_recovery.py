import pytest
from oeis_learn.generalization.recovery import Workflow, restart
from oeis_learn.generalization.resources import preflight, log_tail, reserve_stage
from oeis_learn.generalization.bundles import bundle_plan, verify_bundle

def test_immutable_terminal_and_restart():
    flow=Workflow('study',('qualify','prepare'))
    flow=flow.start('qualify').finish('qualify','PASS')
    with pytest.raises(ValueError): flow.start('qualify')
    assert flow.status=='PARTIAL'
    flow=flow.start('prepare').finish('prepare','INTERRUPTED')
    assert flow.status=='INTERRUPTED'
    assert restart(flow,old_root='old',new_root='new',reason='interrupted',learner=False)['reason']=='interrupted'
    with pytest.raises(ValueError): restart(flow,old_root='old',new_root='old',reason='interrupted',learner=False)
    with pytest.raises(ValueError): restart(flow,old_root='old',new_root='new',reason='interrupted',learner=True)

def test_reservations_and_log_bounds():
    assert preflight(study_bytes=10,checkpoint_bytes=5,shard_bytes=1,raw_bytes=1,sharing_bytes=1,available_bytes=100)['minutes']==240
    with pytest.raises(ValueError): preflight(study_bytes=100,checkpoint_bytes=5,shard_bytes=1,raw_bytes=1,sharing_bytes=1,available_bytes=90)
    with pytest.raises(ValueError): reserve_stage('cases',elapsed_seconds=200*60,requested_seconds=75*60)
    tail=log_tail(('x'*20000+'\n')*200)
    assert len(tail.encode())<=1024**2 and len(tail.splitlines())<=100

def test_required_results_cannot_be_silently_redacted():
    with pytest.raises(ValueError): bundle_plan([dict(path='weights',bytes=1,private=True,required=True)],mandatory={'weights'})
    with pytest.raises(ValueError): bundle_plan([dict(path='huge',bytes=513*1024**2,private=False,required=True)],mandatory={'huge'})
    assert bundle_plan([dict(path='private',bytes=1,private=True,required=False)],mandatory=set())['omissions']==['private']
    with pytest.raises(ValueError): verify_bundle({'study':'s','stages':[]},expected_study='other',root='.')
