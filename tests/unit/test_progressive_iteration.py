import pytest
from oeis_learn.generalization.progressive.iteration import margin
from oeis_learn.generalization.progressive.contracts import digest, seal


def test_complete_update_probe_publication_margin_precedes_work(tmp_path):
    from tests.unit.test_progressive_recovery_transactions import setup
    from tests.integration.test_progressive_recovery import states
    from oeis_learn.generalization.progressive.recovery import Recovery

    t, j, cuts = setup(tmp_path)
    p = t["payload"]
    p["iteration_budget"] = dict(
        update_us=200,
        probe_us=300,
        checkpoint_us=500,
        producer=dict(path="producer.json", bytes=2, sha256=digest("producer")),
    )
    t = seal(t)
    with Recovery(j, t, digest("learning")) as r:
        r.initialize(states(), cuts)
        r.begin(boot="b", monotonic_us=0)
        before = j.path.read_bytes()
        with pytest.raises(ValueError, match="finalization"):
            r.reserve(
                ["slow-update"],
                1,
                boot="b",
                monotonic_us=r.clock["end"] - p["finalization_us"] - 1000,
            )
        assert j.path.read_bytes() == before
    with pytest.raises(ValueError):
        margin({}, research=True)


@pytest.mark.parametrize('slow_phase',['update','checkpoint'])
def test_slow_iteration_is_refused_before_any_debit_or_slow_work(tmp_path,slow_phase):
    import time
    from oeis_learn.pipeline_validation.progressive_training import Progress
    p=Progress(tmp_path/'progress')
    called=[]
    def slow():called.append(slow_phase);time.sleep(.02)
    # The measured conservative bound includes both phases and the probe;
    # admission occurs before any update, checkpoint or token reservation.
    with pytest.raises(TimeoutError):
        p.preflight(bytes_required=0,inodes_required=0,remaining_us=4999,finalization_us=1000,research=False,work_us=4000)
        slow()
    assert not called


def test_freeze_margin_binds_original_case_model_and_conservative_work(tmp_path):
    from oeis_learn.generalization.progressive.iteration import validate_producer
    from oeis_learn.generalization.progressive.contracts import atomic_write,reference
    from oeis_learn.pipeline_validation.progressive_qualification import matrix
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.fresh.bigint_config import resolve
    # Hand-checked validator fixture only; never a hardware qualification receipt.
    source={'diagnostic':'only'};atomic_write(tmp_path/'source.json',source)
    atomic_write(tmp_path/'q.json',{'source_closure':reference(tmp_path,'source.json')})
    profile=resolve_profile(resolve({'version':'bigint-bootstrap/v1'}),seed=13)
    request=matrix([profile],source=digest(source),image=digest('image'),engine='docker',input_lengths=(20,),microbatches=(16,),hardware={'diagnostic':True})
    case=request['cases'][0]
    producer=seal(dict(version='progressive-hip-case/v1',source=digest(source),scope=digest([request['source'],request['image'],request['engine'],request['hardware'],case]),device='actual_HIP',status='PASS',measurements={'elapsed_us':10}))
    atomic_write(tmp_path/'case.json',producer);atomic_write(tmp_path/'matrix.json',request)
    refs=[dict(kind='qualification',artifact=reference(tmp_path,'case.json')),dict(kind='policy',artifact=reference(tmp_path,'matrix.json'))]
    runtime={'payload':{'scope':{'source':reference(tmp_path,'source.json')['sha256'],'images':[digest('image')],'model':digest({k:v for k,v in profile['model'].items() if k!='seed'})},'qualification':[reference(tmp_path,'q.json')]}}
    plan={'iteration_budget':dict(update_us=80,probe_us=80,checkpoint_us=80,producer=reference(tmp_path,'case.json'))}
    assert validate_producer(tmp_path,plan,runtime,refs)==producer
    plan['iteration_budget']['checkpoint_us']=79
    with pytest.raises(ValueError,match='conservative'):validate_producer(tmp_path,plan,runtime,refs)


def test_actual_slow_update_and_checkpoint_calibration_bounds_next_work(tmp_path,monkeypatch):
    import time
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.pipeline_validation.progressive_qualification import calibrate_iteration
    from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate,Progress
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.progressive.contracts import load,reference
    fixture(tmp_path/'data',interrupt=False)
    package=load(tmp_path/'data/training-input.json')
    update=PolicyUpdate.update;save=GeneralizationCheckpointStore.save;calls=[]
    def slow_update(*a,**kw):
        calls.append('update');time.sleep(.04);return update(*a,**kw)
    def slow_save(*a,**kw):
        calls.append('checkpoint');time.sleep(.04);return save(*a,**kw)
    monkeypatch.setattr(PolicyUpdate,'update',slow_update);monkeypatch.setattr(GeneralizationCheckpointStore,'save',slow_save)
    result=calibrate_iteration(package,tmp_path/'calibration',source=digest('diagnostic-source'),image=digest('image'),device='cpu')
    assert result['device']=='cpu'
    assert result['measurements']['update_us']>=40000 and result['measurements']['checkpoint_us']>=40000
    budget={k:2*v for k,v in result['measurements'].items()}|{'producer':reference(tmp_path,'calibration/producer.json')}
    work=margin({'iteration_budget':budget})
    before=list(calls)
    with pytest.raises(TimeoutError):
        Progress(tmp_path/'progress').preflight(bytes_required=0,inodes_required=0,remaining_us=work+1000,finalization_us=1000,research=False,work_us=work)
        slow_update()
    assert calls==before


def test_external_cleanup_window_bounds_the_learner_clock(tmp_path):
    from tests.unit.test_progressive_recovery_transactions import setup
    from tests.integration.test_progressive_recovery import states
    from oeis_learn.generalization.progressive.recovery import Recovery
    t,j,cuts=setup(tmp_path)
    with Recovery(j,t,digest('learning')) as r:
        r.initialize(states(),cuts)
        debit=dict.fromkeys(t['payload']['caps'],0);debit['elapsed_us']=t['payload']['caps']['elapsed_us']
        charge=j.reserve(t['payload']['id'],'train',0,debit,boot='b',monotonic_us=0)
        r.attach(dict(reservation_key=charge['key'],boot='b',start_us=0,end_us=debit['elapsed_us'],work_end_us=2000000))
        assert r.clock['end']==2000000
        before=j.path.read_bytes()
        with pytest.raises(ValueError,match='finalization'):
            r.reserve(['too-late'],1,boot='b',monotonic_us=2000000-t['payload']['finalization_us'])
        assert j.path.read_bytes()==before
