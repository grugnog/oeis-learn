import json
import pytest
from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.state import STATE_FIELDS
from oeis_learn.pipeline_validation.generalization_training import train_updates


def states():
    s={key:{} for key in STATE_FIELDS}
    s['rng']={key:[] for key in ('python','numpy','torch','device')}
    s['data']=dict(cursor=0,pending_batch=None)
    return s


def test_atomic_full_state_and_tampering(tmp_path):
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    s=states();s['weights']={'fixture':[1,2]}
    ref=store.save(s,Ledger(100),encode=lambda x:json.dumps(x).encode())
    loaded=store.load(ref,decode=json.loads)
    assert loaded['states']==s and loaded['ledger']==Ledger(100)
    (tmp_path/ref['path']).write_text('{}')
    with pytest.raises(ValueError):store.load(ref,decode=json.loads)


def test_pending_batch_survives_interruption_and_budget_boundary(tmp_path):
    batches=[[dict(id='a',tokens=[9,2],values=['0']*20)],[dict(id='b',tokens=[8,7,2],values=['1']*20)]]
    s=states();updates=[];saved=[]
    def save(state,ledger): saved.append((json.loads(json.dumps(state)),ledger))
    def update(batch): updates.append(batch[0]['id']);raise InterruptedError()
    with pytest.raises(InterruptedError):train_updates(s,Ledger(4),batches,update=update,save=save)
    state,ledger=saved[-1]
    assert ledger.pending and ledger.consumed==2
    result=train_updates(state,ledger,batches,update=lambda batch:updates.append(batch[0]['id']),save=save)
    assert result['ledger'].consumed==2 and result['ledger'].updates==1
    assert result['ledger'].pending[1]==3 and result['status']=='COMPLETE'
    assert updates==['a','a']


def test_prepared_static_and_refresh_cursor_and_targets(tmp_path):
    from oeis_learn.pipeline_validation.generalization_training import PreparedBatches
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.decoder.bigint_codec import encode_body
    rows=[]
    for i in range(4):
        source=f'(program (return (int {i})))'
        rows.append(dict(id=str(i),role='training',source=source,tokens=encode_body(source),
                         indices=list(map(str,range(100))),values=[str(i)]*100))
    (tmp_path/'rows.json').write_text(json.dumps(rows))
    ref=reference(tmp_path/'rows.json',tmp_path)
    static=PreparedBatches(tmp_path,[ref],batch_size=2,mode='static',updates=4)
    refresh=PreparedBatches(tmp_path,[ref],batch_size=2,mode='refresh',updates=4)
    assert [r['id'] for r in static[2]]==['0','1']
    assert len(refresh)==2 and len(static)==4
    assert all(len(r['values'])==20 for r in refresh[1])
    assert static[0]==PreparedBatches(tmp_path,[ref],batch_size=2,mode='static',updates=4)[0]
    with pytest.raises(IndexError):refresh[2]
    rows[0]['role']='assessment';(tmp_path/'rows.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):PreparedBatches(tmp_path,[reference(tmp_path/'rows.json',tmp_path)],batch_size=2,mode='static',updates=4)


def test_checkpoint_pins_survive_pruning_and_failed_save(tmp_path):
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    encode=lambda x:json.dumps(x).encode()
    first=store.save(states(),Ledger(100),encode=encode)
    second=store.save(states(),Ledger(100),encode=encode)
    store.retain({'initial':first,'latest':second},('initial','latest'))
    third=store.save(states(),Ledger(100),encode=encode)
    store.retain({'initial':first,'old':second,'latest':third},('initial','latest'))
    assert (tmp_path/first['path']).is_file() and (tmp_path/third['path']).is_file()
    assert not (tmp_path/second['path']).exists()
    def fail(value):raise OSError('injected disk failure')
    with pytest.raises(OSError):store.save(states(),Ledger(100),encode=fail)
    assert json.loads((tmp_path/'generalization-latest.json').read_text())==third


def test_descriptor_uses_committed_exposure_without_deserialization(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.identity import digest
    store=GeneralizationCheckpointStore(tmp_path,digest('contract'),100)
    state=states();state['data']['pending_batch']=[{'tokens':[3,2]}]
    ref=store.save(state,Ledger(100).reserve('batch',2),encode=lambda x:json.dumps(x).encode())
    monkeypatch.setattr(store,'_decode',lambda _:pytest.fail('descriptor must not deserialize private state'))
    descriptor=store.describe(ref,initialization=digest('init'),pins=['latest'])
    assert descriptor['tokens']==0 and descriptor['updates']==0
    assert descriptor['weights']==json.loads((tmp_path/ref['path']).read_text())['states']['weights']
    (tmp_path/descriptor['weights']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):store.describe(ref,initialization=digest('init'),pins=['latest'])


@pytest.mark.parametrize("data_mode",["static","refresh"])
@pytest.mark.parametrize("target_format",["literal","parameterized"])
def test_torch_worker_exact_resume_and_retained_milestones(tmp_path,monkeypatch,target_format,data_mode,capsys,replay_mode=False):
    torch=pytest.importorskip('torch')
    pytest.importorskip('numpy')
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.pipeline_validation.identity import digest
    rows=[]
    for i in range(8 if replay_mode else 4 if data_mode=="refresh" else 2):
        src=f'(program (return (int {i})))' if target_format=='literal' else '(program (return (param.get p0)))'
        if replay_mode and i==0:src='(program (return (add (int 0) (int 0))))'
        rows.append(dict(id='task' if replay_mode and i==0 else str(i),role='training',source=src,tokens=encode_body(src),indices=list(map(str,range(100))),values=[str(i)]*100))
    (tmp_path/'rows.json').write_text(json.dumps(rows));shards=[reference(tmp_path/'rows.json',tmp_path)]
    # Tiny real tensor modules isolate persistence equivalence from model scale.
    def models(recipe,device):
        torch.manual_seed(recipe['model']['seed'])
        return torch.nn.Linear(1,1).to(device),torch.nn.Linear(1,1).to(device),{}
    def loss(encoder,decoder,batch,device,**kw):
        optimizer=kw['optimizer'];optimizer.zero_grad()
        value=torch.tensor([[float(r['values'][0])] for r in batch],device=device)
        result=(decoder(encoder(value))-value).square().mean();result.backward();optimizer.step()
    monkeypatch.setattr(bigint_worker,'models',models);monkeypatch.setattr(bigint_worker,'_batch_loss',loss)
    recipe=dict(model=dict(seed=7),learning=dict(learning_rate=.001),inputs={})
    config=dict(study='study',arm='static',seed=7,recipe=recipe,shards=shards,data_mode=data_mode,batch_size=1,
                target_tokens=sum(len(r['tokens']) for r in rows)*(1 if data_mode=='refresh' else 2),milestones=[len(rows[0]['tokens'])],seconds=30,device='cpu')
    extra={}
    if replay_mode:
        import time
        from tests.integration.test_generalization_replay import NativeFixture,task
        from oeis_learn.pipeline_validation.generalization_replay import search_round
        source='(program (return (int 0)))'
        pool,_=search_round([task()],propose=lambda *a:dict(original_source=source,final_source=source,assignment={},tokens=encode_body(source)),
            client=NativeFixture(),checkpoint='checkpoint',seed=7)
        config['batch_size']=4
        first,second,_=models(recipe,'cpu')
        torch.save(dict(encoder=first.state_dict(),decoder=second.state_dict()),tmp_path/'initializer.pt')
        extra=dict(initializer=reference(tmp_path/'initializer.pt',tmp_path),replay=dict(pool=pool,tasks=[task()],client=NativeFixture()))
    full=run_torch(tmp_path,tmp_path/'full',config,**extra)
    progress=[line for line in capsys.readouterr().err.splitlines() if line.startswith('learner:')]
    phases=[line.split()[1] for line in progress]
    assert phases[0]=='batch-inputs' and phases[-1]=='complete'
    for phase in ('model-init','checkpoint-save','checkpoint-saved','update-start','update-complete','publish-roster'):
        assert phase in phases
    if replay_mode:assert 'initializer-load' in phases and 'replay-restore' in phases
    assert len(progress)<=129 and all(float(line.split('unix=')[1])>0 for line in progress)
    part=run_torch(tmp_path,tmp_path/'part',config,stop_after_updates=1,**extra)
    resumed=run_torch(tmp_path,tmp_path/'resumed',config,resume=(tmp_path/'part/checkpoints',part['checkpoint']),**extra)
    a=GeneralizationCheckpointStore(tmp_path/'full/checkpoints',full['compatibility'],config['target_tokens']).load(full['checkpoint'])
    b=GeneralizationCheckpointStore(tmp_path/'resumed/checkpoints',resumed['compatibility'],config['target_tokens']).load(resumed['checkpoint'])
    assert a['ledger']==b['ledger']
    for component in ('encoder','decoder'):
        for name,value in a['states']['weights'][component].items():
            assert torch.equal(value,b['states']['weights'][component][name])
    assert full['milestones']==resumed['milestones']
    if not replay_mode and data_mode=='static' and target_format=='literal':
        from oeis_learn.pipeline_validation.identity import seal
        roster=json.loads((tmp_path/'part/checkpoints/retained.json').read_text())
        candidates=roster['checkpoints'];selected=roster['latest']
        best=next(c['identity'] for c in candidates if 'initial' in c['pins'])
        decision=seal(dict(version='generalization-selected-pins/v1',roster=roster['identity'],
            decision=digest('selection decision'),selected=selected,best_loss=best,reason='selected'))
        pointer=GeneralizationCheckpointStore(tmp_path/'part/checkpoints',part['compatibility'],config['target_tokens']).selected_copy(tmp_path/'selected',roster,decision)
        continued=run_torch(tmp_path,tmp_path/'continued',config,resume=(tmp_path/'selected',pointer))
        restored=GeneralizationCheckpointStore(tmp_path/'continued/checkpoints',continued['compatibility'],config['target_tokens']).load(continued['checkpoint'])
        assert restored['states']['selection']['checkpoint']==selected
        for member in ('encoder','decoder'):
            for key,value in a['states']['weights'][member].items():
                assert torch.equal(value,restored['states']['weights'][member][key])
        kept=json.loads((tmp_path/'continued/checkpoints/retained.json').read_text())
        assert {best,selected}<={c['identity'] for c in kept['checkpoints']}
        assert kept['best_loss']==best and kept['selected']==selected
    for name,result in [('full',full),('resumed',resumed)]:
        roster=json.loads((tmp_path/name/'checkpoints/retained.json').read_text())
        assert result['retained']==roster['identity']
        assert roster['best_loss'] is None and roster['selected'] is None
        assert len([c for c in roster['checkpoints'] if 'latest' in c['pins']])==1
        for candidate in roster['checkpoints']:
            weights=torch.load(tmp_path/name/'checkpoints'/candidate['weights']['path'],weights_only=True)
            assert set(weights)=={'encoder','decoder'}
        assert any('initial' in c['pins'] for c in roster['checkpoints'])


def test_replay_initialized_torch_resume(tmp_path,monkeypatch,capsys):
    test_torch_worker_exact_resume_and_retained_milestones(tmp_path,monkeypatch,'literal','static',capsys,replay_mode=True)


def test_selection_pins_are_persisted_without_mounting_supervision(tmp_path):
    from oeis_learn.pipeline_validation.identity import digest,seal
    store=GeneralizationCheckpointStore(tmp_path/'original',digest('contract'),100)
    encode=lambda value:json.dumps(value).encode()
    state=states();state['history']={'exposures':{},'milestones':{}}
    first=store.save(state,Ledger(100),encode=encode)
    second=store.save(state,Ledger(100,consumed=2,updates=1),encode=encode)
    descriptors=[store.describe(ref,initialization=digest('random'),pins=[label]) for ref,label in [(first,'initial'),(second,'latest')]]
    roster=seal(dict(version='generalization-retained/v1',checkpoints=descriptors))
    decision=seal(dict(version='generalization-selected-pins/v1',roster=roster['identity'],decision=digest('selection'),
        best_loss=descriptors[0]['identity'],selected=descriptors[1]['identity'],reason='selected'))
    pointer=store.selected_copy(tmp_path/'selected',roster,decision,decode=json.loads,encode=encode)
    restored=GeneralizationCheckpointStore(tmp_path/'selected',digest('contract'),100).load(pointer,decode=json.loads)
    assert restored['states']['selection']['checkpoint']==descriptors[1]['identity']
    assert restored['states']['selection']['best_loss']==descriptors[0]['identity']
    assert set(restored['states']['pins'])=={first['sha256'],second['sha256']}
    assert restored['ledger']==Ledger(100,consumed=2,updates=1)
    changed=seal(dict(decision,selected=digest('unknown')))
    with pytest.raises(ValueError,match='outside'):
        store.selected_copy(tmp_path/'forbidden',roster,changed,decode=json.loads,encode=encode)
    assert not (tmp_path/'forbidden').exists()


def test_model_storage_reservation_blocks_before_any_update(tmp_path,monkeypatch):
    import torch
    from oeis_learn.pipeline_validation.generalization_training import storage_preflight
    from oeis_learn.fresh import bigint_worker
    calls=[]
    def models(recipe,device):
        calls.append(device);return torch.nn.Linear(2,2),torch.nn.Linear(2,1),{}
    monkeypatch.setattr(bigint_worker,'models',models)
    study=dict(checkpoints=dict(milestones=[10,20]),seeds=[7])
    runtime=dict(training={'a':{},'b':{}},replay=None,storage=dict(checkpoint_bytes=32<<20))
    result=storage_preflight({},study,runtime)
    assert result['parameter_bytes']==9*4 and result['pinned_generations_reserved']==13
    assert calls==['cpu']
    runtime['storage']['checkpoint_bytes']=result['required_checkpoint_bytes']-1
    with pytest.raises(ValueError,match='required pins'):storage_preflight({},study,runtime)


def test_slow_preupdate_save_stops_with_resumable_debit(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_training as training
    clock=[0.];saved=[];updates=[]
    monkeypatch.setattr(training.time,'monotonic',lambda:clock[0])
    state=states();batches=[[dict(tokens=[3,2])]]
    def save(state,ledger):
        saved.append((json.loads(json.dumps(state)),ledger));clock[0]+=2
    result=train_updates(state,Ledger(2),batches,update=updates.append,save=save,seconds=1)
    assert result['status']=='PARTIAL' and not updates
    state,ledger=saved[-1]
    assert ledger.consumed==2 and ledger.pending and state['data']['cursor']==1
    result=train_updates(state,ledger,batches,update=updates.append,save=lambda *a:None,seconds=1)
    assert result['status']=='COMPLETE' and result['ledger'].consumed==2 and len(updates)==1


def test_checkpoint_clock_includes_tensor_serialization(tmp_path):
    clock=[0.];state=states();state['scheduler']={'learner_seconds':0.}
    store=GeneralizationCheckpointStore(tmp_path,'contract',2)
    def encode(value):
        clock[0]+=1
        return json.dumps(value).encode()
    def scheduler():
        state['scheduler']['learner_seconds']=clock[0]
        return state['scheduler']
    ref=store.save(state,Ledger(2),encode=encode,scheduler=scheduler)
    loaded=store.load(ref,decode=json.loads)
    assert loaded['states']['scheduler']['learner_seconds']==clock[0]-1
    assert clock[0]>5



def test_installed_resume_probe_on_cpu_is_not_hardware_evidence(tmp_path,monkeypatch):
    from tests.integration.test_generalization_gpu import resume_probe
    from oeis_learn.fresh.bigint_config import resolve
    import torch
    from oeis_learn.fresh import bigint_worker
    def models(recipe,device):
        assert device=='cpu';torch.manual_seed(recipe['model']['seed'])
        return torch.nn.Linear(1,1),torch.nn.Linear(1,1),{}
    def update(encoder,decoder,batch,device,**kwargs):
        optimizer=kwargs['optimizer'];optimizer.zero_grad()
        loss=decoder(encoder(torch.ones(len(batch),1))).square().mean();loss.backward();optimizer.step()
    monkeypatch.setattr(bigint_worker,'models',models);monkeypatch.setattr(bigint_worker,'_batch_loss',update)
    result=resume_probe(tmp_path/'probe',resolve(dict(version='bigint-bootstrap/v1')),'cpu')
    assert result['resume_equal'] is True and result['updates']==2



def test_installed_resume_probe_with_qualified_model_dependencies(tmp_path):
    pytest.importorskip('wasmtime',reason='real model loss imports the pinned container runtime')
    from tests.integration.test_generalization_gpu import resume_probe
    from oeis_learn.fresh.bigint_config import resolve
    assert resume_probe(tmp_path/'probe',resolve(dict(version='bigint-bootstrap/v1')),'cpu')['resume_equal']


def test_identical_checkpoint_blobs_are_linked_but_generations_stay_self_contained(tmp_path):
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    s=states();s['weights']={'fixture':[1,2]};s['optimizer']={'step':1}
    encode=lambda value:json.dumps(value).encode()
    first=store.save(s,Ledger(100),encode=encode)
    s['data']['cursor']=1
    second=store.save(s,Ledger(100),encode=encode)
    a=json.loads((tmp_path/first['path']).read_text());b=json.loads((tmp_path/second['path']).read_text())
    old=tmp_path/a['states']['weights']['path'];new=tmp_path/b['states']['weights']['path']
    assert old!=new and old.stat().st_ino==new.stat().st_ino
    import shutil
    shutil.rmtree((tmp_path/first['path']).parent)
    assert store.load(second,decode=json.loads)['states']==s
    s['weights']['fixture'][0]=3
    third=store.save(s,Ledger(100),encode=encode)
    c=json.loads((tmp_path/third['path']).read_text())
    assert (tmp_path/c['states']['weights']['path']).stat().st_ino!=new.stat().st_ino
    assert store.load(third,decode=json.loads)['states']==s


def test_checkpoint_dedup_does_not_publish_corrupt_reused_bytes(tmp_path):
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    s=states();s['weights']={'fixture':[1,2]}
    encode=lambda value:json.dumps(value).encode()
    first=store.save(s,Ledger(100),encode=encode)
    record=json.loads((tmp_path/first['path']).read_text())
    (tmp_path/record['states']['weights']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):store.save(s,Ledger(100),encode=encode)
    assert json.loads((tmp_path/'generalization-latest.json').read_text())==first


def test_checkpoint_blob_copy_fallback_keeps_exact_state(tmp_path,monkeypatch):
    import os,errno
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    s=states();s['weights']={'fixture':[1,2]}
    encode=lambda value:json.dumps(value).encode()
    first=store.save(s,Ledger(100),encode=encode)
    def unsupported(*a,**k):raise OSError(errno.EOPNOTSUPP,'no hard links')
    monkeypatch.setattr(os,'link',unsupported)
    second=store.save(s,Ledger(100),encode=encode)
    assert first!=second and store.load(second,decode=json.loads)['states']==s


def test_debit_snapshot_reuses_tensors_without_serializing_them(tmp_path):
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    state=states();state['weights']={'tensor':'weights'};state['optimizer']={'tensor':'optimizer'}
    encode=lambda v:json.dumps(v).encode()
    original=store.save(state,Ledger(100),encode=encode)
    state['data']=dict(cursor=1,pending_batch=[dict(tokens=[3,2])])
    def metadata_only(v):
        assert v not in (state['weights'],state['optimizer'])
        return encode(v)
    pointer=store.save(state,Ledger(100).reserve('batch',2),encode=metadata_only,unchanged_tensors=original)
    loaded=store.load(pointer,decode=json.loads)
    assert loaded['states']==state and loaded['ledger'].pending==('batch',2)
    assert store.io_stats['reused_tensor_bytes']>0
    with pytest.raises(ValueError,match='transition'):
        store.save(state,Ledger(100,consumed=2,updates=1),encode=metadata_only,unchanged_tensors=pointer)
    with pytest.raises(ValueError,match='latest'):
        store.save(state,Ledger(100).reserve('batch',2),encode=metadata_only,unchanged_tensors=original)


def test_writer_cache_rejects_same_size_tampering_even_with_restored_mtime(tmp_path):
    import os
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    state=states();state['weights']={'tensor':'abcdef'}
    ref=store.save(state,Ledger(100),encode=lambda v:json.dumps(v).encode())
    store.retain({'first':ref},('first',));hashed=store.io_stats['verified_bytes']
    store.retain({'first':ref},('first',))
    assert store.io_stats['verified_bytes']==hashed and store.io_stats['verification_cache_hits']>0
    record=json.loads((tmp_path/ref['path']).read_text());path=tmp_path/record['states']['weights']['path']
    info=path.stat();raw=path.read_bytes();path.write_bytes(raw.replace(b'abcdef',b'ghijkl'))
    os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns))
    with pytest.raises(ValueError):store.retain({'first':ref},('first',))
    assert json.loads((tmp_path/'generalization-latest.json').read_text())==ref


def test_debit_copy_fallback_is_self_contained_and_atomic(tmp_path,monkeypatch):
    import os,errno,shutil
    store=GeneralizationCheckpointStore(tmp_path,'contract',100);state=states()
    state['weights']={'tensor':'unchanged'}
    encode=lambda v:json.dumps(v).encode()
    first=store.save(state,Ledger(100),encode=encode)
    state['data']=dict(cursor=1,pending_batch=[dict(tokens=[3,2])])
    monkeypatch.setattr(os,'link',lambda *a:(_ for _ in ()).throw(OSError(errno.EOPNOTSUPP,'unsupported')))
    second=store.save(state,Ledger(100).reserve('batch',2),encode=encode,unchanged_tensors=first)
    shutil.rmtree((tmp_path/first['path']).parent)
    assert store.load(second,decode=json.loads)['states']==state
    before=(tmp_path/'generalization-latest.json').read_bytes()
    monkeypatch.setattr(shutil,'copyfileobj',lambda *a:(_ for _ in ()).throw(OSError('disk failed')))
    with pytest.raises(OSError):store.save(state,Ledger(100).reserve('batch',2),encode=encode,unchanged_tensors=second)
    assert (tmp_path/'generalization-latest.json').read_bytes()==before


def test_publication_reserve_and_bounded_complete_timing_windows():
    from oeis_learn.pipeline_validation.generalization_training import publication_reserve,TrainingTiming
    assert publication_reserve(21600)==300 and publication_reserve(30)==6
    t=TrainingTiming();t.observe(0,0,0)
    for i in range(1,601):t.observe(i,i*100,float(i))
    assert len(t.windows)==64
    assert all(w['updates']==8 and w['tokens']==800 and w['seconds']==8 for w in t.windows)
    assert t.windows[-1]['through_updates']==596


def test_worker_terminal_pending_tokens_match_checkpoint_weights(tmp_path,monkeypatch):
    import torch
    from oeis_learn.pipeline_validation import generalization_training as m
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.decoder.bigint_codec import encode_body
    source='(program (return (int 0)))';rows=[dict(id='0',role='training',source=source,tokens=encode_body(source),indices=list(map(str,range(100))),values=['0']*100)]
    (tmp_path/'rows.json').write_text(json.dumps(rows))
    config=dict(study='study',arm='static',seed=7,recipe=dict(model={},learning=dict(learning_rate=.001)),
        shards=[reference(tmp_path/'rows.json',tmp_path)],data_mode='static',batch_size=1,target_tokens=100,milestones=[],seconds=30,device='cpu')
    monkeypatch.setattr(bigint_worker,'models',lambda *a:(torch.nn.Linear(1,1),torch.nn.Linear(1,1),{}))
    monkeypatch.setattr(m,'torch_update',lambda *a,**kw:lambda _:pytest.fail('must not update'))
    allowances=[]
    def partial(state,ledger,batches,*,save,seconds,**kwargs):
        allowances.append(seconds)
        batch=batches[0];state['data']=dict(cursor=1,pending_batch=batch)
        pending=ledger.reserve(m.digest(batch),len(rows[0]['tokens']))
        save(state,pending)
        return dict(status='PARTIAL')
    monkeypatch.setattr(m,'train_updates',partial)
    result=m.run_torch(tmp_path,tmp_path/'output',config)
    assert 0<allowances[0]<24
    assert result['tokens']==0 and result['charged_tokens']==len(rows[0]['tokens']) and result['updates']==0
    roster=json.loads((tmp_path/'output/checkpoints/retained.json').read_text())
    latest=next(r for r in roster['checkpoints'] if r['identity']==roster['latest'])
    assert latest['tokens']==result['tokens']
    assert result['timing']['checkpoint_io']['reused_tensor_bytes']>0
    assert result['timing']['publication_reserve_seconds']==6

    # A fully spent saved allowance publishes a partial result without another
    # update or resetting either time or the reserved exposure.
    store=GeneralizationCheckpointStore(tmp_path/'output/checkpoints',result['compatibility'],100)
    saved=store.load(result['checkpoint']);saved['states']['scheduler']['learner_seconds']=30.
    pointer=store.save(saved['states'],saved['ledger'])
    monkeypatch.setattr(m,'train_updates',lambda *a,**kw:pytest.fail('expired learner must not train'))
    expired=m.run_torch(tmp_path,tmp_path/'expired',config,resume=(store.root,pointer))
    assert expired['status']=='PARTIAL' and expired['learner_seconds']>=30
    assert expired['tokens']==0 and expired['charged_tokens']==result['charged_tokens'] and expired['updates']==0


def test_pac_batches_keep_fixed_order_and_exact_token_charge(tmp_path):
    from tests.integration.test_generalization_preparation import pac_rows
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_pac_training
    from oeis_learn.pipeline_validation.generalization_training import PacPreparedBatches
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.generalization.budgets import charge_tokens
    rows,probes=pac_rows()
    def native(s,i):
        r=evaluate(s,i);return dict(outcome=r.outcome,values=list(r.values))
    bundle=prepare_pac_training(rows,probes,output=tmp_path/'data',execute=native,final_exclusions=[],seed=7,batch_size=2,purpose='acceptance')
    stream=PacPreparedBatches(tmp_path/'data',bundle,arm='B',batch_size=2,updates=100)
    assert [r['id'] for i in range(4) for r in stream[i]]==[r['id'] for r in rows]
    assert stream[0]==stream[4]
    s=states();seen=[];saved=[]
    cost=charge_tokens([r['tokens'] for r in stream[0]],bos=1,pad=0)
    result=train_updates(s,Ledger(cost-1),stream,update=seen.append,save=lambda state,ledger:saved.append(ledger))
    assert not seen and result['ledger'].consumed==0 and result['ledger'].pending[1]==cost


def test_pac_timing_history_preserves_whole_run_totals_and_bounded_trend():
    from oeis_learn.pipeline_validation.generalization_training import TrainingTiming
    timing=TrainingTiming()
    for i in range(2000):
        timing.record_update(dict(loss=2.,gradient_norm=3.))
        timing.observe(i,i*10,float(i))
    timing.checkpoint_seconds=10.;timing.checkpoint_calls=3;timing.update_seconds=20.
    restored=TrainingTiming(timing.snapshot())
    assert restored.snapshot()==timing.snapshot()
    restored.record_update(dict(loss=4.,gradient_norm=5.))
    assert restored.total_loss_observations==2001 and restored.total_loss_sum==4004
    assert restored.checkpoint_seconds==10 and len(restored.trend)<=128 and len(restored.windows)<=64
    with pytest.raises(ValueError):TrainingTiming(dict(timing.snapshot(),extra='forbidden'))
