"""Source-transition counterexamples, using no engine or HIP execution."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_pilot_rebind as m
from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.execution import bind,verify_binding


@pytest.fixture
def pair(monkeypatch):
    design=seal(dict(version='fresh-pilot-design/v2',checkpoint_policy='every-update-full-state; verified byte-identical blob reuse'))
    monkeypatch.setattr(m,'PARENT_PROTOCOL',design['identity'])
    study=dict(hypothesis='frozen '+design['identity'],recipe='P1-refresh-screen/v1',changing_factors=['refresh'],
        model=dict(initialization='random'),progression=dict(stage='P1-screen'),seeds=[20261016],arms=['static','refreshed'],
        budgets=dict(target_tokens=25000000,seconds=21600),checkpoints=dict(milestones=[4000000,8000000,16000000,25000000]),
        data=dict(target_format='literal',sharing_policy='explicit',manifests=['unchanged']),source_closure={'old':'source'})
    runtime=dict(version='generalization-runtime/v2',purpose='research',recipe={'recipe':'unchanged'},images=['old-cpu','old-gpu'],
        preparation=None,training={a:dict(data_mode=mode,batch_size=16,shards=list(range(n))) for a,mode,n in [('static','static',1),('refreshed','refresh',12)]},
        evaluation=None,replay=None,assessment=None,feature={'old':'feature'},storage={})
    return (*bind(study,runtime),design)


def test_transition_only_changes_explicit_authorities_and_persistence_declaration(pair):
    study,runtime,design=pair;before=copy.deepcopy(pair)
    new,sidecar,protocol=m.amend(study,runtime,design,{'current':'source'},['cpu','gpu'],{'current':'feature'})
    verify_binding(new,sidecar)
    assert pair==before and new['identity']!=study['identity']
    assert protocol==seal(dict(design,checkpoint_policy='hourly/v1'))
    expected=dict(study,source_closure={'current':'source'},hypothesis=study['hypothesis'].replace(design['identity'],protocol['identity']))
    for key in ('identity','execution'):expected.pop(key)
    assert {k:v for k,v in new.items() if k not in {'identity','execution'}}==expected
    assert {k:v for k,v in sidecar.items() if k not in {'identity','study','images','feature'}}=={k:v for k,v in runtime.items() if k not in {'identity','study','images','feature'}}


@pytest.mark.parametrize('defect',['seal','protocol','initializer','tokens','seconds','milestones','seeds','shards','batch','preparation','purpose','hypothesis'])
def test_transition_rejects_unreviewed_designs(pair,defect):
    study,runtime,design=copy.deepcopy(pair)
    if defect=='protocol':design=seal(dict(design,extra=True))
    elif defect=='initializer':study['model']['initialization']={'kind':'retained'}
    elif defect=='tokens':study['budgets']['target_tokens']-=1
    elif defect=='seconds':study['budgets']['seconds']+=1
    elif defect=='milestones':study['checkpoints']['milestones']=[25000000]
    elif defect=='seeds':study['seeds']=[1]
    elif defect=='shards':runtime['training']['refreshed']['shards'].pop()
    elif defect=='batch':runtime['training']['refreshed']['batch_size']=32
    elif defect=='preparation':runtime['preparation']={}
    elif defect=='purpose':runtime['purpose']='acceptance'
    elif defect=='hypothesis':study['hypothesis']='not the frozen protocol'
    if defect!='seal':study,runtime=bind(study,runtime)
    else:study['hypothesis']='tampered'
    with pytest.raises(ValueError):m.amend(study,runtime,design,{},['cpu','gpu'],{})


@pytest.fixture
def probe(tmp_path,monkeypatch):
    import fresh_pilot as api
    from oeis_learn.pipeline_validation import generalization_runtime as runtime
    import fresh_pilot_training_probe as probe_api
    folder=tmp_path/'probe';qualified=folder/'acceptance/report/qualified'
    authority=dict(images=['cpu','gpu']);write(qualified/'qualification.json',authority)
    stages=[];training={};originals={};initial=digest('initial');checkpoints={}
    for name in ('preflight','prepare','train-static','train-refreshed'):
        output=folder/'research'/name
        result=dict(manifest='probe/prepared/study.json') if name=='prepare' else {}
        if name.startswith('train'):
            arm=name.removeprefix('train-')
            value=dict(status='PASS',tokens=32000,charged_tokens=32000,discarded_tokens=0,pending_tokens=0,updates=32,checkpoint_policy='hourly/v1',
                timing=dict(version='generalization-training-timing/v1',checkpoint_policy='hourly/v1',invocation_seconds=60,checkpoint_calls=2,journal_writes=65,max_checkpoint_seconds=2,
                    windows=[dict(updates=8,tokens=8000,seconds=2,through_updates=n) for n in (12,20,28)]))
            result=dict(runs=[dict(root='learner',result=seal(value))])
            write(output/'training/learner/training.json',value)
            checkpoint=dict(identity=digest(arm),initialization=initial,weights=dict(sha256=digest('weights-'+arm)))
            checkpoints[arm]=checkpoint;training[arm]=value;originals[arm]=value
            proof=seal(dict(status='PASS',runs=[dict(checkpoint=checkpoint['identity'],weights=checkpoint['weights']['sha256'],initial=initial,
                updates=32,tokens=32000,finite=True,changed=True)],resume_equal=None))
            role=folder/('proof-'+arm)/'proof/role'
            write(role/'result.json',proof);write(role/'training-proof.json',proof)
            write(role/'supervision/process.json',dict(outcome='COMPLETE',returncode=0))
            write(role/'supervision/cleanup.json',dict(acknowledged=True,within_bound=True))
        write(output/'payload.json',{'original':True})
        inventory=[reference(p,output) for p in output.rglob('*.json')]
        receipt=seal(dict(version='generalization-command-result/v2',stage=name.split('-')[0],status='PASS',original=None,result=result,inventory=inventory))
        write(output/'result.json',receipt)
        stages.append(dict(stage=name,status='PASS',exit_code=0,receipt=reference(output/'result.json',tmp_path)))
    protocol=seal(dict(version='fresh-pilot-training-probe/v1',parent_protocol='parent',checkpoint_policy='hourly/v1',first_shard_tokens=16000,research_milestones=4))
    record=seal(dict(version='fresh-pilot-training-probe-result/v1',status='PASS',original_preparation='preparation',protocol=protocol,
        qualification=reference(qualified/'qualification.json',tmp_path),images=authority['images'],stages=stages,training=training,
        proofs={a:dict(status='PASS') for a in training},initialization=initial,forecast=m.assess(originals,16000,research_milestones=4)))
    write(folder/'result.json',record)
    monkeypatch.setattr(m,'APPROVED_PROBE',record['identity'])
    monkeypatch.setattr(runtime,'load',lambda *a:(folder,dict(identity='probe-study'),dict(preparation=None,purpose='research',images=authority['images'])))
    monkeypatch.setattr(probe_api,'verify_plan',lambda *a:None)
    monkeypatch.setattr(api,'latest_checkpoint',lambda learner,*a,**kw:(checkpoints[learner.parents[1].name.removeprefix('train-')],None))
    return tmp_path,folder,qualified,authority,record


def test_probe_recomputes_original_measurements_without_modifying_evidence(probe):
    root,folder,qualified,authority,record=probe
    before={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    assert m.probe_evidence(root,folder,'preparation','parent',qualified,authority)==record
    assert before=={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('defect',['summary','receipt','inventory','missing','training','proof','cleanup','qualification','images','parent','preparation'])
def test_probe_requires_original_receipts_not_summary_pass(probe,defect):
    root,folder,qualified,authority,record=probe;parent='parent';preparation='preparation'
    path=folder/'research/train-static/result.json'
    if defect=='summary':path=folder/'result.json'
    elif defect=='inventory':path=folder/'research/train-static/payload.json'
    elif defect=='training':path=folder/'research/train-static/training/learner/training.json'
    elif defect=='proof':path=folder/'proof-static/proof/role/training-proof.json'
    elif defect=='cleanup':path=folder/'proof-static/proof/role/supervision/cleanup.json'
    elif defect=='qualification':path=qualified/'qualification.json'
    elif defect=='images':authority=dict(images=['changed','gpu'])
    elif defect=='parent':parent='other'
    elif defect=='preparation':preparation='other'
    if defect=='missing':path.unlink()
    elif defect not in {'images','parent','preparation'}:write(path,seal(dict(status='PASS')),exclusive=False)
    with pytest.raises((ValueError,OSError,KeyError)):m.probe_evidence(root,folder,preparation,parent,qualified,authority)


@pytest.fixture
def prepared(tmp_path,monkeypatch,pair):
    from oeis_learn.pipeline_validation import generalization_runtime as runtime_api
    import fresh_pilot_feasibility
    old=tmp_path/'old';prepared=old/'research/prepare';base=prepared/'preparation/role/private/prepared'
    study,runtime,design=pair
    cfg=dict(initializer=None,seeds=[20261016],hypothesis=study['hypothesis'],cells=[])
    for key in ('recipe','descriptors','exclusions'):
        write(old/(key+'.json'),{key:'original'});cfg[key]=str((old/(key+'.json')).relative_to(tmp_path))
        write(base/(key+'.json'),{key:'original'})
        ref=reference(base/(key+'.json'),base)
        if key=='recipe':runtime[key]=ref
        else:study['data'][key]=ref
    study,runtime=bind(study,runtime)
    write(base/'study.json',study);write(base/'runtime.json',runtime)
    write(old/'research/design/design.json',design);write(old/'research/design/blueprint.json',cfg)
    job=seal(dict(version='fresh-pilot-smoke-job/v1',engine='podman',manifest=str((base/'study.json').relative_to(tmp_path)),
        design=reference(old/'research/design/design.json',tmp_path),blueprint=reference(old/'research/design/blueprint.json',tmp_path)))
    write(old/'private/smoke-job.json',job)
    terminal=seal(dict(version='generalization-command-result/v2',stage='prepare',status='PASS',original=None,
        result=dict(manifest=job['manifest']),inventory=[reference(p,prepared) for p in prepared.rglob('*.json')]))
    write(prepared/'result.json',terminal);monkeypatch.setattr(m,'PARENT_PREPARATION',terminal['identity'])
    cells=[dict(name=str(i),status='FEASIBLE') for i in range(16)]
    feasible=seal(dict(status='FEASIBLE',protocol=design['identity'],cells=cells))
    write(old/'feasibility/feasibility.json',feasible)
    write(old/'feasibility/result.json',seal(dict(status='PASS',feasibility=reference(old/'feasibility/feasibility.json',old/'feasibility'))))
    monkeypatch.setattr(runtime_api,'load',lambda *a:(base,study,runtime))
    monkeypatch.setattr(fresh_pilot_feasibility,'verify_prepared',lambda *a:None)
    return tmp_path,old,base


def test_preparation_is_only_candidate_provenance_and_remains_immutable(prepared):
    root,old,base=prepared;before={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    result=m.prepared_evidence(root,old,'podman')
    assert result['manifest']==str((base/'study.json').relative_to(root))
    assert before=={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}


@pytest.mark.parametrize('defect',['payload','manifest','blueprint','design','feasibility','missing-authority','engine'])
def test_original_preparation_mismatch_fails_closed(prepared,monkeypatch,defect):
    root,old,base=prepared;engine='podman'
    if defect=='engine':engine='docker'
    elif defect=='payload':write(base/'recipe.json',{},exclusive=False)
    elif defect=='blueprint':write(old/'recipe.json',{},exclusive=False)
    elif defect=='design':write(old/'research/design/design.json',{},exclusive=False)
    elif defect=='feasibility':write(old/'feasibility/feasibility.json',{},exclusive=False)
    elif defect=='manifest':
        path=old/'private/smoke-job.json';value=json.loads(path.read_text());value['manifest']='elsewhere.json';write(path,seal(value),exclusive=False)
    else:
        path=old/'research/prepare/result.json';value=json.loads(path.read_text())
        value['inventory']=[r for r in value['inventory'] if not r['path'].endswith('/runtime.json')]
        value=seal(value);write(path,value,exclusive=False);monkeypatch.setattr(m,'PARENT_PREPARATION',value['identity'])
    with pytest.raises((ValueError,OSError,KeyError)):m.prepared_evidence(root,old,engine)
