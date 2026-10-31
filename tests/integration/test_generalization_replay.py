import time
import pytest
from oeis_learn.pipeline_validation.generalization_replay import search_round,restore_pool
from oeis_learn.decoder.bigint_codec import encode_body

class NativeFixture:
    def runtime_identity(self,profile):return 'fixture-runtime'
    def execute(self,source,indices,*,deadline):
        assert deadline>time.monotonic()
        return dict(outcome='ok',values=['0']*len(indices))


def task():
    return dict(id='task',role='training',ancestry=['mechanical','training'],source='(program (return (add (int 0) (int 0))))',indices=list(map(str,range(100))),values=['0']*100)


def test_native_replay_admission_and_restore():
    src='(program (return (int 0)))'
    def propose(visible,attempt,deadline):
        assert set(visible)=={'id','indices','values'} and len(visible['values'])==20
        return dict(original_source=src,final_source=src,assignment={},tokens=encode_body(src))
    pool,report=search_round([task()],propose=propose,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    assert len(pool.entries)==1 and report['search_ledger'][0][1]==8
    restored=restore_pool(pool.snapshot(),tasks=[task()],client=NativeFixture(),deadline=time.monotonic()+60)
    assert restored.snapshot()==pool.snapshot()
    assert report['plan']['arms']==['continued_sft','verified_replay']


def test_empty_yield_is_terminal_without_weaker_verification():
    def propose(*args):raise ValueError('no_candidate')
    pool,report=search_round([task()],propose=propose,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    assert not pool.entries and report['plan']['status']=='empty_yield'


def test_mixed_batch_persists_replay_cursor_before_interrupted_update():
    import copy
    import pytest
    from oeis_learn.pipeline_validation.generalization_replay import ReplayBatches
    from oeis_learn.pipeline_validation.generalization_training import train_updates
    from oeis_learn.generalization.budgets import Ledger
    src='(program (return (int 0)))'
    propose=lambda *args:dict(original_source=src,final_source=src,assignment={},tokens=encode_body(src))
    pool,_=search_round([task()],propose=propose,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    base=[[dict(id=str(i),tokens=encode_body(src),values=['0']*20,indices=list(map(str,range(20)))) for i in range(8)]]
    state={'data':dict(cursor=0,pending_batch=None),'replay':pool.snapshot()}
    batches=ReplayBatches(base,pool,[task()],state)
    saved=[]
    def save(s,l):saved.append((copy.deepcopy(s),l))
    def interrupt(batch):raise InterruptedError()
    with pytest.raises(InterruptedError):train_updates(state,Ledger(1000),batches,update=interrupt,save=save)
    persisted,ledger=saved[-1]
    assert persisted['replay']['cursor']==2
    assert len(persisted['data']['pending_batch'])==8
    updated=[]
    result=train_updates(persisted,ledger,batches,update=lambda b:updated.append(b),save=save)
    assert len(updated)==1 and result['state']['replay']['cursor']==2
    assert sum(r['id'].startswith('replay:') for r in updated[0])==2


def test_search_overrun_retains_charged_attempt_evidence(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def overrun(*args):
        clock[0]=61.
        raise ValueError('native deadline cleanup overrun')
    _,report=replay.search_round([task()],propose=overrun,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    assert sum(row[1] for row in report['search_ledger'])==1


def test_interruption_persists_attempt_before_work_and_preserves_measured_time(monkeypatch):
    import pytest
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.];saved=[]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def interrupt(*args):
        assert saved[-1]['search_state']['pending']==['task',60.]
        assert saved[-1]['search_ledger']==[['task',1,0]]
        clock[0]=3.
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        replay.search_round([task()],propose=interrupt,client=NativeFixture(),checkpoint='checkpoint',seed=3,
                            save=lambda pool,report:saved.append(report))
    assert saved[-1]['search_ledger']==[['task',1,3.]]
    assert saved[-1]['status']=='INTERRUPTED'
    assert caught.value.replay_evidence==saved[-1]


def test_late_candidate_is_not_audited_or_admitted(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.];calls=[]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def late(*args):clock[0]=2.;return {}
    monkeypatch.setattr(replay,'admission',lambda *a,**kw:calls.append('audit'))
    pool,report=replay.search_round([task()],propose=late,client=NativeFixture(),checkpoint='checkpoint',seed=3,seconds=1)
    assert not calls and not pool.entries
    assert report['search_ledger']==[['task',1,2.]]
    assert report['search_state']['overruns']==[['task',2.,1.]]
    assert report['status']=='PARTIAL'


def test_failed_reservation_persistence_prevents_work():
    import pytest
    calls=[]
    def save(*args):raise OSError('disk full')
    with pytest.raises(OSError,match='disk full'):
        search_round([task()],propose=lambda *a:calls.append('work'),client=NativeFixture(),checkpoint='c',seed=1,save=save)
    assert calls==[]


def test_reservation_persistence_deadline_prevents_callback(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.];calls=[]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def save(*args):clock[0]=2.
    _,report=replay.search_round([task()],propose=lambda *a:calls.append('work'),client=NativeFixture(),checkpoint='c',seed=1,seconds=1,save=save)
    assert calls==[] and report['status']=='PARTIAL'
    assert report['search_ledger']==[['task',1,2.]]


def test_unexpected_failure_propagates_with_accounting(monkeypatch):
    import pytest
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def fail(*args):clock[0]=5.;raise RuntimeError('native crash')
    with pytest.raises(RuntimeError,match='native crash') as caught:
        replay.search_round([task()],propose=fail,client=NativeFixture(),checkpoint='c',seed=1)
    assert caught.value.replay_evidence['status']=='FAIL'
    assert caught.value.replay_evidence['search_ledger']==[['task',1,5.]]


def test_fitted_literal_round_restores_original_and_transformed_sources():
    src='(program (return (int 0)))'
    candidate=dict(original_source='(program (return (int 9)))',
        transformed_source='(program (return (param.get p0)))',final_source=src,
        assignment={'p0':'0'},tokens=encode_body(src))
    pool,report=search_round([task()],propose=lambda *a:candidate,client=NativeFixture(),checkpoint='checkpoint',seed=7)
    restored=restore_pool(pool.snapshot(),tasks=[task()],client=NativeFixture(),deadline=time.monotonic()+60)
    assert restored.snapshot()==pool.snapshot() and report['admitted']==1
    assert restored.entries[0].record()['original_source']==candidate['original_source']


def test_isolated_search_projects_only_visible_inputs_and_persists_reservation(tmp_path,monkeypatch):
    import json
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.pipeline_validation.generalization_replay import search_isolated,admission
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.generalization.contracts import verify_artifact
    weights=tmp_path/'weights';weights.mkdir();(weights/'weights.pt').write_bytes(b'fixture')
    calls=[]
    source='(program (return (int 0)))'
    def invoke(host,image,workspace,role,artifacts,**kw):
        settings=kw['settings'];calls.append(role)
        saved=json.loads((tmp_path/'search/private/search-state.json').read_text())
        assert saved['report']['search_state']['pending'][0]=='task'
        roots=kw['source_roots']
        if role=='generator':
            assert set(artifacts)=={'visible_prompts','frozen_weights'}
            prompt=json.loads(verify_artifact(roots['visible_prompts'],settings['prompt']))
            assert set(prompt)=={'id','indices','values'} and len(prompt['values'])==20
            return dict(status='PASS',candidate=dict(original_source=source,final_source=source,assignment={},tokens=encode_body(source)))
        assert role=='learner' and set(artifacts)=={'training_targets','verified_training_replay'}
        target=json.loads(verify_artifact(roots['training_targets'],settings['task']))
        candidate=json.loads(verify_artifact(roots['verified_training_replay'],settings['candidate']))
        entry=admission(target,candidate,client=NativeFixture(),checkpoint=settings['checkpoint'],order=settings['order'],deadline=time.monotonic()+30)
        return dict(status='PASS',entry=entry.record())
    monkeypatch.setattr(roles,'project_and_invoke',invoke)
    pool,report=search_isolated(None,tmp_path,tmp_path/'search',images=['cpu-fixture','gpu-fixture'],
        weights_root=weights,checkpoint=reference(weights/'weights.pt',weights),tasks=[task()],recipe={},study='study',seed=4)
    assert calls==['generator','learner']*8 and report['admitted']==1
    assert json.loads((tmp_path/'search/private/pool.json').read_text())==pool.snapshot()
    assert report['current_policy_success'] is None and report['cumulative_discoveries']==1


@pytest.mark.parametrize('admitted',[True,False])
def test_complete_paired_workflow_and_exact_resume_use_current_weights(tmp_path,monkeypatch,admitted):
    import copy,json
    import torch
    from tests.integration.test_generalization_runtime import fixture
    from oeis_learn.pipeline_validation import generalization_commands as commands,generalization_workflow as workflow,generalization_roles as roles,generalization_acceptance as acceptance,generalization_role as role,generalization_scoring as scoring
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.generalization.role_contracts import Role
    from oeis_learn.pipeline_validation.identity import digest
    from oeis_learn.fresh import bigint_worker,pilot_metrics
    from oeis_learn.synthesis import client as native_module
    study,runtime,put=fixture(tmp_path)
    targets=[]
    for i in range(4):
        original=task();original['id']='task' if i==0 else str(i)
        targets.append(dict(original,tokens=encode_body(original['source'])))
        targets[-1].pop('ancestry')
    shard=put('replay-training.json',targets)
    study['progression']['stage']='P4';study['arms']=['continued_sft','verified_replay']
    study['budgets']['target_tokens']=sum(len(t['tokens']) for t in targets)*2
    study['checkpoints']['milestones']=[sum(len(t['tokens']) for t in targets)]
    runtime['purpose']='acceptance';runtime['training']={a:dict(shards=[shard],data_mode='static',batch_size=4) for a in study['arms']}
    runtime['replay']=dict(tasks=put('replay-tasks.json',[task()]),seconds=60)
    truth=[dict(task(),id=k,role='selection',family=digest(k),stratum=k) for k in ('structural','parameter')]
    for row in truth:row.pop('ancestry')
    prompts=[dict(id=r['id'],indices=r['indices'][:20],values=r['values'][:20]) for r in truth]
    runtime['evaluation']=dict(role='selection',prompts=[put('prompts.json',prompts)],targets=[put('truth.json',truth)],checkpoints=[],seconds=60,bank_seeds=None)
    def models(recipe,device):
        torch.manual_seed(recipe['model']['seed'])
        return torch.nn.Linear(1,1).to(device),torch.nn.Linear(1,1).to(device),{}
    def update(encoder,decoder,batch,device,**kwargs):
        optimizer=kwargs['optimizer'];optimizer.zero_grad()
        loss=decoder(encoder(torch.ones(len(batch),1))).square().mean();loss.backward();optimizer.step()
    monkeypatch.setattr(bigint_worker,'models',models);monkeypatch.setattr(bigint_worker,'_batch_loss',update)
    monkeypatch.setattr(pilot_metrics,'measure',lambda *a,**kw:dict(complete=True,loss=1.))
    monkeypatch.setattr(scoring,'score_frozen',lambda *a,**kw:dict(outcomes={r['id']:{'equal_budget':dict(complete=True,success=False)} for r in truth}))
    recipe=json.loads((tmp_path/'recipe.json').read_text());encoder,decoder,_=models(recipe,'cpu')
    torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),tmp_path/'initializer.pt')
    initializer=reference(tmp_path/'initializer.pt',tmp_path)
    plan=dict(recipe=recipe,initializer=dict(weights=initializer))
    class Native(NativeFixture):
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(native_module,'SynthesisClient',Native)
    calls=[]
    def invoke(host,image,workspace,name,artifacts,**kw):
        settings=copy.deepcopy(kw['settings']);operation=kw['operation'];calls.append((name,operation))
        out=workspace/kw['output']/'role';out.mkdir(parents=True)
        if operation=='replay-propose':
            if not admitted:return dict(status='PARTIAL',reason='fixture_no_candidate')
            source='(program (return (int 0)))'
            return dict(status='PASS',candidate=dict(original_source=source,final_source=source,assignment={},tokens=encode_body(source)))
        if operation=='evaluate':
            assert set(artifacts)=={'visible_prompts','frozen_weights'}
            (out/'generation').mkdir();(out/'generation/selections.json').write_text(json.dumps({r['id']:dict(selections={}) for r in prompts}))
            return dict(status='PASS',checkpoint=settings['checkpoint']['sha256'])
        if operation=='train':settings['config']['device']='cpu'
        return role.HANDLERS[(Role(name),operation)](kw['source_roots'],out,settings,seconds=kw['seconds'])
    for module in (roles,commands,workflow,acceptance):monkeypatch.setattr(module,'project_and_invoke',invoke)
    result=workflow.replay(None,tmp_path,tmp_path/'round',tmp_path,study,runtime,plan,acceptance_resume=True)
    assert result['status']=='PASS' and result['rounds']==1
    run=result['runs'][0]
    assert run['cumulative_discoveries']==int(admitted) and run['resume']['resume_equal'] is True
    assert len(run['current_policies'])==2
    assert [r['arm'] for r in run['paired']['runs']]==['continued_sft','verified_replay']
    assert all(e['result']['selection']['measurements'][0]['behavior']['structural'][1]['structural'] is False for e in run['current_policies'])
    assert calls.count(('learner','train'))==4
    assert all(p['changed'] and p['finite'] for p in run['resume']['runs'])


def test_isolated_diagnostic_is_charged_native_audited_and_never_model_yield(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_replay import search_isolated
    from oeis_learn.pipeline_validation import generalization_roles
    from oeis_learn.generalization.acceptance_inputs import DIAGNOSTIC
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_replay import admission
    import json
    (tmp_path/'weights.pt').write_bytes(b'identified fixture weights')
    weight=reference(tmp_path/'weights.pt',tmp_path);calls=[]
    def invoke(host,image,workspace,role,artifacts,**kw):
        calls.append((role,kw['operation']))
        assert role=='learner' and kw['operation']=='replay-audit'
        settings=kw['settings'];roots=kw['source_roots']
        t=json.loads(verify_artifact(roots['training_targets'],settings['task']))
        candidate=json.loads(verify_artifact(roots['verified_training_replay'],settings['candidate']))
        entry=admission(t,candidate,client=NativeFixture(),checkpoint=settings['checkpoint'],order=settings['order'],deadline=time.monotonic()+10)
        return dict(status='PASS',entry=entry.record())
    monkeypatch.setattr(generalization_roles,'project_and_invoke',invoke)
    pool,report=search_isolated(None,tmp_path,tmp_path/'search',images=['cpu','gpu'],weights_root=tmp_path,checkpoint=weight,tasks=[task()],recipe={},study='fixture',seed=7,seconds=30,purpose='acceptance',proposal_policy=DIAGNOSTIC)
    assert len(pool.entries)==1 and report['search_ledger'][0][1]==1
    assert report['plan']['model_discoveries']==0 and report['plan']['proposal_origin']==DIAGNOSTIC
    assert calls==[('learner','replay-audit')]
    with pytest.raises(ValueError,match='research'):
        search_isolated(None,tmp_path,tmp_path/'forbidden',images=['cpu','gpu'],weights_root=tmp_path,checkpoint=weight,tasks=[task()],recipe={},study='fixture',seed=7,seconds=30,purpose='research',proposal_policy=DIAGNOSTIC)
    assert not (tmp_path/'forbidden').exists()
