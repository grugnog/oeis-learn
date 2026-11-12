"""Storage reservations follow actual model scopes without relaxing caps."""
import copy
from types import SimpleNamespace
import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, reference, digest
from oeis_learn.generalization.progressive.storage_plan import CONTEXT_BYTES, publication_reserve
from oeis_learn.pipeline_validation import progressive_research_resources as resources


def test_scoped_finalization_counts_only_training_models_and_evaluation_weights(tmp_path,monkeypatch):
    from tests.unit.test_progressive_continuation import campaign
    from oeis_learn.generalization.progressive.tuning import BASELINE
    c=campaign(); plans=c['payload']['tasks']
    small,large=digest('small'),digest('large')
    for p,scope in zip(plans,(small,large)):
        p['initializer']=dict(mode='random',artifact=None,compatibility=scope)
        p['caps']['disk_bytes']=8<<30
    plans[1]['token_target']=25000000
    plans[1]['trial_choices']=[dict(parameters=dict(BASELINE),compatibility=small,policy=plans[1]['policy']),
                              dict(parameters=dict(BASELINE,shape='larger'),compatibility=large,policy=plans[1]['policy'])]
    evaluation=copy.deepcopy(plans[0]);evaluation.update(id='evaluate',kind='evaluation',token_target=0)
    request=dict(operation='generate',artifacts=dict(model=[dict(task=plans[0]['id'],kind='checkpoint')]))
    atomic_write(tmp_path/'request.json',request)
    evaluation['policy']=reference(tmp_path,'request.json')['sha256'];plans.append(evaluation)
    atomic_write(tmp_path/'policy.json',resources.policy())
    payload=dict(task_plans=plans,checkpoint={},checkpoint_root='original',milestone_pins=1,
        references=[dict(name='research_resource_policy',kind='policy',artifact=reference(tmp_path,'policy.json')),
                    dict(name='generation',kind='policy',artifact=reference(tmp_path,'request.json'))],
        graph=c['payload']['graph'],caps=dict(c['payload']['caps'],disk_bytes=32<<30),deficits=[])
    for node in payload['graph']['nodes']:node['caps']['disk_bytes']=24<<30
    cases=[dict(compatibility=s,package=dict(shards=[[dict(tokens=[1,2])]],update_policy=dict(effective_batch=16))) for s in (small,large)]
    producers=[dict(compatibility=s,checkpoint_bytes=size,weights_bytes=size//4,checkpoint_inodes=28,
                    measurements=dict(update_us=1,probe_us=1,checkpoint_us=1)) for s,size in ((small,100<<20),(large,400<<20))]
    monkeypatch.setattr('oeis_learn.generalization.progressive.storage.checkpoint_size',lambda *a,**kw:(2<<30,28))
    result=resources.finalize(tmp_path,payload,cases,producers)
    assert result['status']=='READY'
    assert plans[0]['caps']['disk_bytes']==2*((100<<20)+CONTEXT_BYTES)+(128<<20)
    assert plans[1]['caps']['disk_bytes']==3*((400<<20)+CONTEXT_BYTES)+(128<<20)
    assert evaluation['caps']['disk_bytes']==(25+128)<<20


def test_shared_package_reserve_is_not_duplicated_inside_each_task(tmp_path,monkeypatch):
    from oeis_learn.generalization.progressive.storage import StorageGuard
    guard=StorageGuard([tmp_path],max_bytes=128, max_inodes=50,free_reserve_bytes=512)
    monkeypatch.setattr('shutil.disk_usage',lambda _:SimpleNamespace(free=1024))
    guard.check(100)
    with pytest.raises(TimeoutError,match='owned storage'):guard.check(129)
    monkeypatch.setattr('shutil.disk_usage',lambda _:SimpleNamespace(free=600))
    with pytest.raises(TimeoutError,match='free storage'):guard.check(100)


def test_next_publication_does_not_require_a_second_empty_campaign_disk():
    storage=dict(checkpoint_bytes=900,checkpoint_inodes=90,milestone_pins=1,
        package_bytes=500,required_bytes=10**12,required_inodes=10**6,
        task_checkpoints=[dict(task='small',checkpoint_bytes=100,checkpoint_inodes=10,milestone_pins=0)])
    assert publication_reserve(storage,dict(id='small'))==(100+500+CONTEXT_BYTES,266)
    legacy=dict(storage);del legacy['task_checkpoints']
    assert publication_reserve(legacy,dict(id='small'))==(900+500+CONTEXT_BYTES,346)


@pytest.mark.parametrize('fallback',[False,True])
def test_projection_copy_has_a_distinct_inode_and_source_survives_changes(tmp_path,monkeypatch,fallback):
    import errno
    import fcntl
    from oeis_learn.generalization.progressive.storage import copy_file
    source=tmp_path/'source';target=tmp_path/'copy'
    source.write_bytes(b'original immutable tensor')
    if fallback:
        def unsupported(*args):raise OSError(errno.EOPNOTSUPP,'no reflink support')
        monkeypatch.setattr(fcntl,'ioctl',unsupported)
    copy_file(source,target)
    assert target.read_bytes()==source.read_bytes()
    assert target.stat().st_ino!=source.stat().st_ino
    target.write_bytes(b'worker changed private copy')
    assert source.read_bytes()==b'original immutable tensor'
    target.unlink()
    assert source.read_bytes()==b'original immutable tensor'


def test_projection_copy_preserves_permission_failures(tmp_path,monkeypatch):
    import errno
    import fcntl
    from oeis_learn.generalization.progressive.storage import copy_file
    source=tmp_path/'source';source.write_bytes(b'tensor')
    def denied(*args):raise OSError(errno.EACCES,'denied')
    monkeypatch.setattr(fcntl,'ioctl',denied)
    with pytest.raises(PermissionError):copy_file(source,tmp_path/'target')


@pytest.mark.parametrize('failure',['setup','worker_reclaimed','worker_unreclaimed'])
def test_projection_cleanup_requires_no_live_reader(tmp_path,monkeypatch,failure):
    from tests.unit.test_progressive_budgets import role_authority
    from tests.contract.test_progressive_roles import artifact
    from oeis_learn.generalization.progressive.budgets import KEYS,role_work_requirements
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.pipeline_validation import progressive_roles as roles
    from tests.unit import test_progressive_freeze as fixture
    original_campaign=fixture.campaign
    def roomy_campaign():
        c=original_campaign()
        c['payload']['tasks'][0]['caps']['disk_bytes']=16<<20
        for node in c['payload']['graph']['nodes']:
            node['caps']['disk_bytes']=16<<20
        c['payload']['storage'].update(required_inodes=10000,available_inodes=10000)
        return seal(c)
    monkeypatch.setattr(fixture,'campaign',roomy_campaign)
    values=[seal(dict(kind='candidate',data={}))]
    a,j,t=role_authority(tmp_path,'admit-training',values)
    c=a['campaign']
    runtime=SimpleNamespace(execution_ready=True,identity=c['payload']['runtime'],track=c['track'],
        purpose=c['purpose'],payload=dict(operations=['admit-training'],scope=dict(images=['image'])))
    item=artifact(tmp_path,'candidate.json','candidate','admission',{})
    def projected(root, destination, *args, **kwargs):
        destination.mkdir(parents=True)
        (destination/'model.pt').write_bytes(b'private copy')
        return {}
    monkeypatch.setattr(roles,'project',projected)
    def fail(**kwargs):
        if failure=='setup':raise RuntimeError('deliberate setup failure')
        return ['explicit test engine boundary']
    def supervise(host,name,command,target,**kwargs):
        atomic_write(target/'supervision/process.json',dict(cleanup_failure=None))
        atomic_write(target/'supervision/cleanup.json',dict(within_bound=True,acknowledged=failure=='worker_reclaimed'))
        raise RuntimeError('deliberate worker failure')
    monkeypatch.setattr('oeis_learn.pipeline_validation.generalization_roles.supervise_owned',supervise)
    host=SimpleNamespace(engine=SimpleNamespace(command=fail),policy=dict(memory_bytes=1<<30,cpus=1))
    cost=dict.fromkeys(KEYS,0)|role_work_requirements('admit-training',values)|dict(elapsed_us=1000000)
    with pytest.raises(RuntimeError,match='deliberate'):
        roles._invoke_reserved(host,'image',tmp_path,'admit-training',{'candidates':[item],'admission_truth':[]},
            output='failed',campaign=c,task=t,journal=j,ordinal=1,reservation=cost,seconds=1,runtime=runtime)
    assert (tmp_path/'failed/private-input').exists()==(failure=='worker_unreclaimed')
