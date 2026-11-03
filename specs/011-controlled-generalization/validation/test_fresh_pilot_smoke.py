import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_pilot_smoke as m
from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference


def outcomes(targets):
    from fresh_pilot_report import CONTROLS
    return dict(result=dict(outcomes={r['id']:{c:dict(success=False,complete=False,
        upstream=dict(complete=False),scoring=dict(outcome='no_selected_candidate')) for c in CONTROLS} for r in targets}))


def test_smoke_accepts_recorded_negatives_and_censoring_but_not_missing_scoring():
    targets=[dict(id='one'),dict(id='two')];scored=outcomes(targets)
    assert m.scoring_recorded(scored,targets)
    scored['result']['outcomes']['one']['raw']['scoring']['outcome']='scoring_incomplete'
    assert not m.scoring_recorded(scored,targets)
    scored=outcomes(targets);scored['result']['outcomes']['one'].pop('bank')
    assert not m.scoring_recorded(scored,targets)


def test_reduced_data_has_same32_families_and_budget_reaches_fresh_rows(tmp_path):
    from fresh_pilot_inputs import research_inputs,blueprint
    from oeis_learn.generalization.refresh import RefreshPlan
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.budgets import charge_tokens
    design=research_inputs()
    cfg=blueprint(tmp_path,design,recipe='recipe.json',source='source.json',images=['cpu','gpu'],feature='feature.json',output='design')
    prepared=tmp_path/'prepared';write(prepared/'study.json',{})
    stream=RefreshPlan(design['descriptors'],seed=design['seed'],capacity=1_000_000)
    for ordinal,cell in enumerate(cfg['cells']):
        if cell['role']=='training':
            rows,_=stream.take(64*ordinal,64)
            rows=[dict(r,tags=['family:'+r['family']]) for r in rows]
        else:
            from oeis_learn.generalization.preparation import reference_candidate
            rows=[reference_candidate(r)[0][0] for r in design['cohorts'][ordinal]['rows'][:2]]
        write(prepared/'cells'/cell['name']/'private/selected.json',rows)
    job=dict(manifest='prepared/study.json',blueprint=reference(tmp_path/'design/blueprint.json',tmp_path),design=reference(tmp_path/'design/design.json',tmp_path))
    cfg,protocol=m.reduced_blueprint(tmp_path,job,tmp_path/'smoke')
    train=[json.loads((tmp_path/'smoke'/(c['name']+'.json')).read_text()) for c in cfg['cells'] if c['role']=='training']
    refresh=[c for c in cfg['cells'] if c['role']=='training']
    assert [c['refresh']['cursor'] for c in refresh]==[0,65536]
    assert all(c['refresh']['draws']==2048 and 'candidates' not in c for c in refresh)
    assert cfg['preparation_seconds']==900
    assert [len(rows) for rows in train]==[32,32]
    assert len({r['family'] for rows in train for r in rows})==32
    assert {r['source'] for r in train[0]}.isdisjoint(r['source'] for r in train[1])
    tokens=lambda rows:charge_tokens([encode_body(r['source']) for r in rows],bos=1,pad=0)
    assert cfg['budgets']['target_tokens']>=tokens(train[0])+tokens(train[1][:16])
    assert cfg['budgets']['target_tokens']>=tokens(train[0])+tokens(train[0][:16])
    assert cfg['budgets']['target_tokens']<=tokens(train[0]+train[1])
    assert cfg['initializer'] is None and cfg['budgets']['seconds']==900
    assert cfg['hypothesis'].endswith(protocol['identity'])


@pytest.mark.parametrize('failure',[None,'preflight','prepare','train-static','train-refreshed',
    'generate-static','score-static','generate-refreshed','score-refreshed','proof','initialization','zero_updates','scoring','plan_drift'])
def test_smoke_gates_canonical_boundaries_and_keeps_failures(tmp_path,monkeypatch,failure):
    import fresh_pilot as api
    from types import SimpleNamespace
    from contextlib import nullcontext
    from oeis_learn.pipeline_validation import generalization_inputs as inputs,generalization_runtime as runtime,generalization_acceptance as acceptance,generalization_lifecycle as lifecycle
    import foundation_local
    monkeypatch.setattr(api,'ROOT',tmp_path)
    monkeypatch.setattr(foundation_local,'Host',lambda args:SimpleNamespace(args=args))
    monkeypatch.setattr(lifecycle,'guard',lambda *a:nullcontext())
    write(tmp_path/'recipe.json',{})
    (tmp_path/'qualified').mkdir()
    study=dict(identity=digest('study'));images=['cpu','gpu'];base=tmp_path/'prepared';base.mkdir()
    refs=[];targets=[]
    for i,stratum in enumerate(('parameter','structural')):
        row=dict(id=str(i),stratum=stratum);targets.append(row)
        write(base/f'truth{i}.json',[row]);write(base/f'prompts{i}.json',[dict(id=str(i))])
        write(base/f'cohort{i}.json',dict(role='selection',private_payloads=[reference(base/f'truth{i}.json',base)],public_payloads=[reference(base/f'prompts{i}.json',base)]))
        refs.append(reference(base/f'cohort{i}.json',base))
    study['data']=dict(manifests=refs)
    sidecar=dict(storage={},images=images,recipe=reference(tmp_path/'recipe.json',tmp_path))
    # Runtime artifact paths are relative to base.
    write(base/'recipe.json',{});sidecar['recipe']=reference(base/'recipe.json',base)
    monkeypatch.setattr(runtime,'load',lambda *a:(base,study,sidecar))
    monkeypatch.setattr(inputs,'assemble_study',lambda *a:dict(manifest='smoke-study.json'))
    def reduced(root,job,output):
        p=seal(dict(first_shard_tokens=10,target_tokens=30));write(output/'protocol.json',p)
        return dict(images=images),p
    monkeypatch.setattr(m,'reduced_blueprint',reduced)
    def verify_plan(*a):
        if failure=='plan_drift':raise ValueError('native smoke selection differs from frozen source plan')
        return dict(minimum_updates=3)
    monkeypatch.setattr(m,'verify_smoke_plan',verify_plan)
    monkeypatch.setattr(acceptance,'training_proof',lambda *a:dict(status='FAIL' if failure=='proof' else 'PASS'))
    monkeypatch.setattr(api,'latest_checkpoint',lambda learner,result,**kw:(dict(initialization=kw['arm'] if failure=='initialization' else 'common'),learner))
    calls=[]
    def invoke(argv,logs,**kw):
        name=logs.name;calls.append(name);output=tmp_path/argv[argv.index('--output')+1]
        result=dict(status='FAIL' if failure==name else 'PASS')
        if name=='prepare':result['result']=dict(manifest='resolved.json')
        if name.startswith('train-'):
            (output/'training/learner').mkdir(parents=True)
            result['result']=dict(runs=[dict(root='learner',result=dict(tokens=20,updates=0 if failure=='zero_updates' else 3))])
        write(output/'result.json',seal(result));return 1 if failure==name else 0
    monkeypatch.setattr(api,'invoke',invoke)
    def evaluate(engine,qualified,folder,manifest,study,protocol,arm,*args):
        command=args[-1]
        for stage in ('generate-','score-'):
            output=folder/(stage+arm);command(stage+arm,['fake','--output',str(output.relative_to(tmp_path))],output)
        return dict(result=dict(outcomes={})) if failure=='scoring' else outcomes(targets),[]
    monkeypatch.setattr(api,'evaluate_arm',evaluate)
    job=seal(dict(version='fresh-pilot-smoke-job/v1',engine='podman',qualified='qualified',manifest='full.json',design={},blueprint={}))
    write(tmp_path/'job.json',job)
    code=m.run(api,'job.json','smoke')
    assert code==(0 if failure is None else 1),(tmp_path/'smoke/result.json').read_text()
    result=json.loads((tmp_path/'smoke/result.json').read_text())
    assert result['status']==('PASS' if failure is None else 'FAIL')
    if failure in m.LIMITS:assert failure in calls
    if failure in {'preflight','prepare','plan_drift'}:assert not any(n.startswith('train-') for n in calls)
    if failure in {'proof','initialization','zero_updates'}:assert not any(n.startswith('generate-') for n in calls)


def test_smoke_prepares_and_passes_real_refresh_runtime_validation(tmp_path,monkeypatch):
    """Exercise the real assembly -> preparation -> resolved runtime boundary."""
    import shutil
    from tests.integration.test_generalization_inputs import blueprint as fixture_blueprint
    from fresh_pilot_inputs import research_inputs,blueprint
    from oeis_learn.pipeline_validation.generalization_inputs import assemble_study
    from oeis_learn.pipeline_validation.generalization_runtime import load,publish_runtime,validate_execution
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_cell,prepare_study,draw_candidates
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.generalization.exclusions import ExclusionIndex
    from oeis_learn.generalization.preparation import exclusion_record
    from oeis_learn.synthesis.reference import evaluate
    monkeypatch.setattr(shutil,'disk_usage',lambda *_:type('Disk',(),dict(free=256<<30))())
    fixture=fixture_blueprint(tmp_path);write(tmp_path/'feature.json',{'fixture':'not qualification'})
    design=research_inputs()
    cfg=blueprint(tmp_path,design,recipe=fixture['recipe'],source=fixture['source_closure'],images=fixture['images'],feature='feature.json',output='design')
    prepared=tmp_path/'original';write(prepared/'study.json',{})
    def native(source,indices):
        result=evaluate(source,indices);return dict(outcome=result.outcome,values=list(result.values))
    exclusion=ExclusionIndex();seen=set();training=0
    for cell in cfg['cells']:
        if cell['role']=='training':
            if training==2:break
            training+=1
            cell=dict(cell,count=32,quotas={k:1 for k in cell['quotas']},refresh=dict(cell['refresh'],draws=2048))
        candidates=dict(cell)
        if 'candidates' in candidates:candidates['candidates']=reference(tmp_path/candidates['candidates'],tmp_path)
        rows,_=draw_candidates(tmp_path,reference(tmp_path/cfg['descriptors'],tmp_path),candidates)
        folder=prepared/'cells'/cell['name']
        ledger=prepare_cell(rows,**{k:cell[k] for k in ('role','holdout','count','quotas','structural','length_boundary')},
            output=folder,execute=native,exclusions=exclusion,seen_sources=seen,unique_sources=cell['role']=='training',seconds=120)
        assert ledger['status']=='PASS'
        selected=json.loads((folder/'private/selected.json').read_text())
        if cell['role']=='training':seen.update(digest(r['source']) for r in selected)
        else:exclusion=exclusion.add_many(exclusion_record(r,cell['role'],cell['holdout']) for r in selected)
    job=dict(manifest='original/study.json',blueprint=reference(tmp_path/'design/blueprint.json',tmp_path),design=reference(tmp_path/'design/design.json',tmp_path))
    reduced,protocol=m.reduced_blueprint(tmp_path,job,tmp_path/'smoke-design')
    assembly=assemble_study(tmp_path,reduced,'smoke-inputs')
    base,study,runtime=load(tmp_path,assembly['manifest'])
    validate_execution(base,study,runtime)
    prep=validate('preparation',json.loads((base/'preparation.json').read_text()))
    output=tmp_path/'smoke-prepared'
    assert prepare_study(base,prep,study,output=output,execute=native,seconds=120)['status']=='PASS'
    resolved,sidecar=publish_runtime(base,output,study,runtime)
    validate_execution(output,resolved,sidecar)
    from fresh_pilot_storage import project
    projected,receipt=project(tmp_path,str((output/'study.json').relative_to(tmp_path)),'execution/private/prepared')
    projected_base,projected_study,projected_runtime=load(tmp_path,projected)
    assert (projected_study,projected_runtime)==(resolved,sidecar)
    validate_execution(projected_base,projected_study,projected_runtime)
    assert receipt['storage']['peak_shard_bytes']<sidecar['storage']['shard_bytes']
    assert not (projected_base/'cells/static/private/selected.json').exists()
    for name in ('static','refresh-01'):
        assert json.loads((output/'cells'/name/'private/selected.json').read_text())==json.loads((tmp_path/'smoke-design'/(name+'.json')).read_text())
    assert resolved['budgets']['target_tokens']==protocol['target_tokens']
    from oeis_learn.pipeline_validation.generalization_training import PreparedBatches,train_updates
    from oeis_learn.generalization.budgets import Ledger
    for arm in ('static','refreshed'):
        view=sidecar['training'][arm]
        batches=PreparedBatches(output,view['shards'],batch_size=16,mode=view['data_mode'],updates=3)
        actual=train_updates(dict(data=dict(cursor=0,pending_batch=None)),Ledger(protocol['target_tokens']),batches,
            update=lambda _:None,save=lambda *_:None,seconds=10)
        assert actual['ledger'].updates==3 and actual['ledger'].consumed>protocol['first_shard_tokens']



def test_exact_batch_budget_handles_parent_selection_with_shorter_programs():
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.pipeline_validation.generalization_training import train_updates
    # A parent-derived cap of 32*10 +16*10 permits only two 16*15 batches.
    batches=[[dict(id=str(i),tokens=[2]*15) for i in range(16)] for _ in range(4)]
    state=lambda:dict(data=dict(cursor=0,pending_batch=None))
    old=train_updates(state(),Ledger(480),batches,update=lambda _:None,save=lambda *_:None,seconds=10)
    assert old['ledger'].updates==2
    assert m.batch_budget([480,480],[480,480])==1440
    new=train_updates(state(),Ledger(m.batch_budget([240,240],[240,240])),batches,
        update=lambda _:None,save=lambda *_:None,seconds=10)
    assert new['ledger'].updates==3 and new['ledger'].consumed==720


@pytest.mark.parametrize('change',[None,'order','tokens','costs','cap','batch_size'])
def test_native_smoke_plan_must_match_frozen_exact_sources_and_tokens(tmp_path,change):
    from oeis_learn.decoder.bigint_codec import encode_body
    arrays=[];refs=[]
    for shard in range(2):
        rows=[dict(source=f'(program (return (int {i+shard*32})))') for i in range(32)]
        rows=[dict(r,tokens=encode_body(r['source'])) for r in rows];arrays.append(rows)
        write(tmp_path/f'{shard}.json',rows);refs.append(reference(tmp_path/f'{shard}.json',tmp_path))
    charges=[m.batch_charges(rows) for rows in arrays];cap=m.batch_budget(*charges)
    protocol=seal(dict(minimum_updates=3,batch_charges=charges,target_tokens=cap,first_shard_tokens=sum(charges[0]),
        training_sources=[digest([r['source'] for r in rows]) for rows in arrays]))
    runtime=dict(training=dict(static=dict(batch_size=16,shards=refs[:1]),refreshed=dict(batch_size=16,shards=refs)))
    study=dict(budgets=dict(target_tokens=cap))
    if change=='order':
        arrays[1].reverse();write(tmp_path/'1.json',arrays[1],exclusive=False);refs[1].update(reference(tmp_path/'1.json',tmp_path))
    if change=='tokens':
        arrays[1][0]['tokens']=[2];write(tmp_path/'1.json',arrays[1],exclusive=False);refs[1].update(reference(tmp_path/'1.json',tmp_path))
    if change=='costs':protocol['batch_charges'][1][0]+=1;protocol=seal(protocol)
    if change=='cap':study['budgets']['target_tokens']-=1
    if change=='batch_size':runtime['training']['static']['batch_size']=8
    if change:
        with pytest.raises(ValueError):m.verify_smoke_plan(tmp_path,study,runtime,protocol)
    else:assert m.verify_smoke_plan(tmp_path,study,runtime,protocol)['minimum_updates']==3
