import copy
import json
from pathlib import Path
import sys
import time

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]/'src'))
from oeis_learn.pipeline_validation.identity import seal,digest


def policy():
    import prospective_evaluation_policy as p
    return p


def row(name='a',*,family='family-a',holdout='parameter',start=0,source=None):
    return dict(id=name,family=family,holdout=holdout,source=source or name,group=name,
                variant=name,values=list(map(str,range(start,start+100))))


def test_shifted_window_and_old_same_role_are_excluded():
    p=policy();candidate=row(start=0)
    index=p.CandidateExclusions([candidate])
    index.observe([row('old',start=80)],role='selection',holdout='parameter')
    assert index.rejected()=={'a':{'finite_window'}}


@pytest.mark.parametrize('field,reason',[('source','source'),('group','group'),('variant','numeric_variant')])
def test_exact_identity_exclusions(field,reason):
    p=policy();a=row();b=row('old',start=1000);b[field]=a[field]
    index=p.CandidateExclusions([a]);index.observe([b],role='training',holdout='parameter')
    assert index.rejected()=={'a':{reason}}


@pytest.mark.parametrize('candidate,old,role,excluded',[
    ('parameter','parameter','training',False),('structural','parameter','training',True),
    ('parameter','structural','assessment',True),('structural','structural','selection',False)])
def test_family_separation_both_directions(candidate,old,role,excluded):
    p=policy();index=p.CandidateExclusions([row(holdout=candidate)])
    index.observe([row('old',start=1000)],role=role,holdout=old)
    assert ('a' in index.rejected())==excluded


def test_exact_windows_do_not_collapse_large_integers():
    p=policy();a=row();b=row('old');a['values']=[str(2**200+i) for i in range(100)]
    b['values']=[str(2**200+i+1) for i in range(0,200,2)]
    index=p.CandidateExclusions([a]);index.observe([b],role='training',holdout='parameter')
    assert not index.rejected()


def test_duplicate_candidate_identity_and_expired_deadline_rejected():
    p=policy()
    with pytest.raises(ValueError):p.CandidateExclusions([row(),row()])
    index=p.CandidateExclusions([row()])
    with pytest.raises(TimeoutError):index.observe([row('old')],role='training',holdout='parameter',deadline=0)


def test_protocol_frozen_counts_and_no_outcome_inputs():
    p=policy();prior=json.loads((HERE/'observable-evaluation-protocol.json').read_text())
    result=p.design(prior)
    assert result==p.design(copy.deepcopy(prior))
    assert result['seed']==20261104
    assert sum(c['count'] for c in result['cells'])==256
    assert result['counts']==dict(certified=153,ambiguous=55,unresolved=48)
    assert 'outcomes' not in json.dumps(result)
    bad=copy.deepcopy(prior);bad['roster'].pop(next(iter(bad['roster'])))
    with pytest.raises(ValueError):p.design(seal(bad))


def test_fixed_reservoir_is_unique_and_same_family_allocation():
    p=policy();d=p.design(json.loads((HERE/'observable-evaluation-protocol.json').read_text()))
    rows=p.draw(d)
    assert len(rows)<=8192
    assert len({r['id'] for r in rows})==len(rows)
    assert {r['family'] for r in rows}==set(d['families'])
    assert all(r['origin']=='generic_random' for r in rows)


def test_roster_freeze_rejects_shortfall_and_duplicate_source():
    p=policy();d=dict(cells=[dict(stratum='parameter',family='family-a',count=2)],families={'family-a':{'category':'certified'}})
    with pytest.raises(ValueError):p.roster([row()],d)
    with pytest.raises(ValueError):p.roster([row(),row('b',source='a')],d)
    actual=p.roster([row(),row('b',start=1000)],d)
    assert len(actual)==2 and all(v['category']=='certified' for v in actual.values())


def small_design():
    from oeis_learn.data.bigint_programs import BigintSampler
    from oeis_learn.generalization.refresh import RefreshPlan
    p=policy();prior=json.loads((HERE/'observable-evaluation-protocol.json').read_text());d=p.design(prior)
    # Two ordinary polynomial families; no model outcome affects the fixture.
    cells=[]
    for s in ('parameter','structural'):
        cell=next(c for c in d['cells'] if c['stratum']==s and d['families'][c['family']]['stage']=='A')
        cells.append(dict(cell,count=1,draws=3))
    d.update(cells=cells,families={c['family']:d['families'][c['family']] for c in cells})
    return seal(d)


@pytest.mark.parametrize('failure',[None,'native','shortfall','tampered'])
def test_native_preparation_never_publishes_failed_cohort(tmp_path,monkeypatch,failure):
    p=policy();d=small_design()
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    old=tmp_path/'old.json';old.write_text('[]')
    ref=reference(old,tmp_path)
    if failure=='tampered':old.write_text('[{}]')
    originals=[dict(root=tmp_path,ref=ref,name='old',role='training',holdout='parameter',count=0)]
    if failure=='shortfall':monkeypatch.setattr(p,'draw',lambda _:[])
    def native(source,indices):
        from oeis_learn.synthesis.reference import evaluate
        return dict(outcome='ok',values=['999']*100 if failure=='native' else list(evaluate(source,indices).values))
    out=tmp_path/'new'
    if failure:
        with pytest.raises(ValueError):p.prepare(d,originals,out,execute=native,seconds=60,workers=1)
        assert not (out/'summary.json').exists()
    else:
        result=p.prepare(d,originals,out,execute=native,seconds=60,workers=1)
        assert result['selected']==2 and len(result['roster'])==2
        assert result['status']=='PASS' and result['original_rows']==0


def test_family_macro_and_incomplete_denominators():
    p=policy();mapping={str(i):dict(category='certified',stratum='parameter' if i==3 else 'structural',family='A' if i<2 else 'B') for i in range(4)}
    scores={arm:{str(i):{'equal_budget':dict(success=i<2,complete=i!=2)} for i in range(4)} for arm in ('static','refreshed')}
    result=p.summarize(mapping,scores,['equal_budget'])
    s=result['cells']['certified/structural']['equal_budget']
    assert s['arms']['static']['family_macro']==.5
    assert s['arms']['static']['tasks']==3 and s['arms']['static']['unresolved']==1
    assert result['primary_interpretable'] is False
    scores['refreshed'].pop('0')
    with pytest.raises(ValueError):p.summarize(mapping,scores,['equal_budget'])


def prepared_fixture(tmp_path,monkeypatch):
    import prospective_evaluation as h
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    d=small_design();root=tmp_path/'new';root.mkdir();checkpoints={}
    for arm in ('static','refreshed'):
        weights=tmp_path/('weights-'+arm);weights.mkdir();(weights/'weights.pt').write_bytes(b'fixture')
        checkpoints[arm]=dict(checkpoint=seal(dict(version='generalization-checkpoint/v1',weights=reference(weights/'weights.pt',weights),arm=arm)),weights_root=weights.name)
    d=seal(dict(d,checkpoints={a:c['checkpoint']['weights'] for a,c in checkpoints.items()}))
    def native(s,i):return dict(outcome='ok',values=evaluate(s,i).values)
    summary=policy().prepare(d,[],root/'prepare/cohort',execute=native,seconds=60,workers=1)
    parent=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['study']
    parent['data']['sharing_policy']='explicit'
    recipe=dict(version='fixture-model/v1',model=dict(seed=7,width=8),learning=dict(learning_rate=.001))
    parent['model'].update(profile=recipe['version'],architecture=digest({'width':8}))
    runtime=dict(purpose='research',storage=dict(study_bytes=2<<30,checkpoint_bytes=1<<30,shard_bytes=256<<20,raw_bytes=128<<20,sharing_bytes=64<<20))
    q=tmp_path/'qualified';q.mkdir();write(q/'qualification.json',seal(dict(status='fixture')));write(q/'feature.json',seal(dict(status='fixture')))
    ref=reference(q/'qualification.json',tmp_path)
    context=dict(study=parent,runtime=runtime,recipe=recipe,originals=[],checkpoints=checkpoints,preparation=ref,parent_manifest=ref)
    authority=dict(images=[digest('cpu'),digest('gpu')]);monkeypatch.setattr(h,'ROOT',tmp_path)
    # Keep production source identity deterministic without asking the fixture root to be a repo.
    import oeis_learn.experiments.source_snapshot as source
    monkeypatch.setattr(source,'collect_snapshot',lambda _:dict(fixture=True))
    return h,root,context,d,q,authority,summary


def test_real_resolved_E0_schema_runtime_and_no_training(tmp_path,monkeypatch):
    h,root,context,d,q,a,summary=prepared_fixture(tmp_path,monkeypatch)
    base,study,runtime,protocol=h.publish_inputs(root,context,d,q,a,summary)
    from oeis_learn.pipeline_validation.generalization_runtime import load,validate_execution,execution_refs
    actual=load(tmp_path,str((base/'study.json').relative_to(tmp_path)))
    assert actual==(base,study,runtime)
    verified=validate_execution(base,study,runtime)
    assert verified['training']=={} and study['budgets']['target_tokens']==0
    assert runtime['assessment'] is None and runtime['evaluation']['role']=='selection'
    assert protocol['identity'] in study['hypothesis']
    execution_refs(base,study,runtime)
    from oeis_learn.pipeline_validation.generalization_workflow import _checkpoints
    selected=_checkpoints(base,runtime['evaluation']['checkpoints'])
    assert len(selected)==2
    from oeis_learn.generalization.contracts import verify_artifact
    for checkpoint,parent in selected:assert verify_artifact(parent,checkpoint['weights'])==b'fixture'
    assert all(set(p)=={'id','indices','values'} for p in json.loads((base/'visible/prompts.json').read_text()))
    assert not (base/'visible/targets.json').exists()


def test_native_output_tamper_blocks_freeze(tmp_path,monkeypatch):
    h,root,context,d,q,a,summary=prepared_fixture(tmp_path,monkeypatch)
    target=root/'prepare/cohort/cells/parameter/private/targets.json';rows=json.loads(target.read_text())
    rows[0]['values'][99]='999999';target.write_text(json.dumps(rows))
    with pytest.raises(ValueError):h.publish_inputs(root,context,d,q,a,summary)
    assert not (root/'research/protocol.json').exists()


@pytest.mark.parametrize('failure',[None,'launch','cleanup','result'])
def test_native_host_real_command_private_projection_and_fail_closed(tmp_path,monkeypatch,failure):
    import prospective_evaluation as h
    from types import SimpleNamespace
    from contextlib import nullcontext
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import supervision,generalization_lifecycle
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    import oeis_learn.experiments.source_snapshot as source
    monkeypatch.setattr(h,'ROOT',tmp_path);monkeypatch.setattr(source,'collect_snapshot',lambda _:dict(fixture=True))
    engine=Engine('podman',executable='podman',details={});monkeypatch.setattr(engine,'user_args',lambda:[])
    monkeypatch.setattr(h,'host_for',lambda _:SimpleNamespace(engine=engine));monkeypatch.setattr(generalization_lifecycle,'guard',lambda _:nullcontext())
    old=tmp_path/'assessment.json';old.write_text('[]');ref=reference(old,tmp_path)
    context=dict(originals=[dict(name='assessment-parameter',role='assessment',holdout='parameter',count=0,ref=ref)])
    target=tmp_path/'prepare';d=small_design();commands=[];cleanup=[]
    def launch(argv,logs,**kw):
        commands.append(argv);logs.mkdir(parents=True)
        assert '--network=none' in argv and '--read-only' in argv and '--device' not in argv
        mounts=[argv[i+1] for i,a in enumerate(argv) if a=='--mount']
        assert len(mounts)==3 and sum('readonly' in m for m in mounts)==2
        assert not any('src='+str(tmp_path)+',' in m for m in mounts)
        assert (target.parent/'private/cohort-inputs/exclusions/cell-0.json').read_text()=='[]'
        assert kw['seconds']<h.LIMITS['prepare']
        if failure=='launch':raise TimeoutError('fixture')
        write(logs/'process.json',dict(outcome='COMPLETE',returncode=0))
        write(target/'cohort/summary.json',seal(dict(status='PASS',design=d['identity'],files=[])))
        write(target/'result.json',seal(dict(status='FAIL' if failure=='result' else 'PASS',summary=reference(target/'cohort/summary.json',target))))
    def reclaim(engine,name,logs,state):
        cleanup.append(True);write(logs/'cleanup.json',dict(acknowledged=failure!='cleanup',within_bound=True))
    monkeypatch.setattr(supervision,'launch',launch);monkeypatch.setattr(supervision,'reclaim',reclaim)
    if failure:
        with pytest.raises((TimeoutError,ValueError)):h.native_prepare('podman',digest('image'),context,d,target)
    else:assert h.native_prepare('podman',digest('image'),context,d,target)['status']=='PASS'
    assert len(commands)==1 and cleanup==[True] and old.read_text()=='[]'


@pytest.mark.parametrize('failure',[None,'qualification','prepare','preflight','generate-static','score-static'])
def test_controller_real_job_wiring_and_no_inference_after_failed_gate(tmp_path,monkeypatch,failure):
    from types import SimpleNamespace
    from contextlib import nullcontext
    import fresh_pilot as api
    import fresh_pilot_training_probe as probe
    from oeis_learn.pipeline_validation import generalization_lifecycle
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_runtime import validate_execution
    h,root,context,d,q,authority,summary=prepared_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(api,'ROOT',tmp_path)
    monkeypatch.setattr(h,'design',lambda _:d)
    monkeypatch.setattr(h,'lineage',lambda *_:context)
    monkeypatch.setattr(h.shutil,'disk_usage',lambda _:SimpleNamespace(free=256<<30))
    monkeypatch.setattr(generalization_lifecycle,'scope',lambda *a,**k:nullcontext())
    events=[]
    def qualification(*a):
        events.append('qualification')
        if failure=='qualification':raise ValueError('fixture qualification failure')
        return q,authority
    def native(*a):
        events.append('prepare')
        if failure=='prepare':raise ValueError('fixture native failure')
        return summary
    monkeypatch.setattr(probe,'qualified',qualification);monkeypatch.setattr(h,'native_prepare',native)
    def invoke(argv,logs,*,seconds):
        stage=logs.name;events.append(stage);output=tmp_path/argv[argv.index('--output')+1];output.mkdir(parents=True)
        if failure==stage:
            write(output/'result.json',seal(dict(status='FAIL')));return 1
        if stage=='preflight':
            base=tmp_path/argv[argv.index('--manifest')+1];s=json.loads(base.read_text());r=json.loads(base.with_name('runtime.json').read_text())
            validate_execution(base.parent,s,r)
        else:
            job=json.loads((tmp_path/argv[argv.index('--job')+1]).read_text())
            protocol=json.loads((tmp_path/job['protocol']['path']).read_text())
            assert protocol['identity'] in json.loads((root/'research/study.json').read_text())['hypothesis']
            assert job['checkpoint']==context['checkpoints']['static' if stage.endswith('static') else 'refreshed']['checkpoint']
            if stage.startswith('generate'):
                assert set(job['artifacts'])=={'visible_prompts','frozen_weights'}
                assert all('truth' not in r and 'assessment' not in r for r in job['source_roots'].values())
                assert protocol['role_seconds']+600<seconds
                g=output/'execution/role/generation'
                write(g/'selections.json',{k:dict(selections={c:None for c in protocol['controls']}) for k in protocol['roster']})
                write(g/'proposal-bank.json',{});write(g/'evaluation.json',dict(status='PASS'))
            else:
                assert set(job['artifacts'])=={'selection_supervision','selection_candidates'}
                assert protocol['scoring_role_seconds']+500<seconds
                candidates=job['settings']['candidates'];rows=json.loads((tmp_path/job['source_roots']['selection_candidates']/candidates['path']).read_text())
                write(output/'execution/role/selection/scoring.json',dict(candidates=candidates,result=dict(outcomes={k:{c:dict(candidate_sha256=None,success=False,complete=True) for c in protocol['controls']} for k in rows})))
        write(output/'result.json',seal(dict(status='PASS')));return 0
    monkeypatch.setattr(api,'invoke',invoke)
    args=SimpleNamespace(engine='podman',pilot='old',prepared='original',qualified='qualified',imported='uniform',cpu_image=None,gpu_image=None)
    result=h.execute(args,root)
    if failure:
        assert result==1 and events[-1]==failure and not (root/'measurements.json').exists()
    else:
        assert result==0
        assert events==['qualification','prepare','preflight','generate-static','score-static','generate-refreshed','score-refreshed']
        assert read_json(root/'measurements.json')['primary_interpretable']
    assert all('train' not in stage for stage in events)


def read_json(path):return json.loads(path.read_text())


def test_failure_export_is_verified_and_excludes_private_truth(tmp_path,monkeypatch):
    import prospective_evaluation as h
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.bundle import verify
    import zipfile
    root=tmp_path/'new';root.mkdir();monkeypatch.setattr(h,'ROOT',tmp_path)
    write(root/'workflow.json',seal(dict(status='FAIL',stages=[],qualification=None)))
    write(root/'private/truth/secret.json',{'secret':'do not export'})
    archive=h.export(root);verify(archive)
    with zipfile.ZipFile(archive) as z:
        assert all('secret' not in n and 'truth' not in n for n in z.namelist())
        assert json.loads(z.read('summary.json'))['selected_scope']=='FAIL'


@pytest.mark.parametrize('failure',[None,'worker-exit','packaging'])
def test_outer_controller_and_export_failures_are_preserved(tmp_path,monkeypatch,failure):
    from types import SimpleNamespace
    import prospective_evaluation as h
    import fresh_pilot as api
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.bundle import verify
    import zipfile
    monkeypatch.setattr(h,'ROOT',tmp_path);monkeypatch.setattr(api,'ROOT',tmp_path)
    for n in ('pilot','prepared','imported','qualified'):(tmp_path/n).mkdir()
    calls=[]
    def invoke(argv,logs,*,seconds):
        calls.append((argv[2],seconds));root=tmp_path/'new'
        if argv[2]=='work':
            write(root/'workflow.json',seal(dict(status='PASS',stages=[],qualification=None)))
            return 1 if failure=='worker-exit' else 0
        if failure=='packaging':return 1
        archive=h.export(root);write(root/'archive-result.json',seal(dict(archive=reference(archive,root))));return 0
    monkeypatch.setattr(api,'invoke',invoke)
    args=SimpleNamespace(command='run',engine='podman',pilot='pilot',prepared='prepared',imported='imported',qualified='qualified',output='new',cpu_image=None,gpu_image=None)
    assert h.run(args)==(0 if failure is None else 1)
    assert calls==[('work',h.CAMPAIGN_SECONDS-h.REPORT_SECONDS),('package',h.REPORT_SECONDS-120)]
    archive=tmp_path/'new'/('emergency.zip' if failure=='packaging' else 'report.zip');verify(archive)
    with zipfile.ZipFile(archive) as z:
        assert json.loads(z.read('summary.json'))['selected_scope']==('PASS' if failure is None else 'FAIL')
    with pytest.raises(ValueError):h.run(args)


@pytest.mark.parametrize('scoring_seconds',[None,1200])
def test_real_shared_scorer_role_respects_frozen_inner_envelope(tmp_path,monkeypatch,scoring_seconds):
    from types import SimpleNamespace
    import fresh_pilot as api
    import foundation_local
    from oeis_learn.pipeline_validation import generalization_commands,generalization_runtime,generalization_roles
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    monkeypatch.setattr(api,'ROOT',tmp_path)
    monkeypatch.setattr(foundation_local,'Host',lambda args:SimpleNamespace(args=args))
    images=[digest('cpu'),digest('gpu')];write(tmp_path/'qualified/qualification.json',dict(images=images))
    protocol=dict(role_seconds=21600)
    if scoring_seconds is not None:protocol['scoring_role_seconds']=scoring_seconds
    protocol=seal(protocol);write(tmp_path/'protocol.json',protocol)
    study=dict(identity=digest('study'),hypothesis=protocol['identity']);runtime=dict(purpose='research',images=images,storage={})
    monkeypatch.setattr(generalization_runtime,'load',lambda *a:(tmp_path,study,runtime))
    calls=[]
    monkeypatch.setattr(generalization_commands,'_qualification',lambda *a:calls.append('qualified'))
    def invoke(*a,**kw):
        assert a[1]==images[0] and a[3]=='selection' and not kw['gpu']
        assert kw['seconds']==(21600 if scoring_seconds is None else scoring_seconds)
        calls.append('scored');return dict(status='PASS')
    monkeypatch.setattr(generalization_roles,'project_and_invoke',invoke)
    (tmp_path/'truth').mkdir();(tmp_path/'candidates').mkdir();(tmp_path/'study').mkdir()
    job=seal(dict(version='fresh-pilot-role/v1',engine='podman',qualified='qualified',manifest='study.json',
        study=study['identity'],study_root='study',protocol=reference(tmp_path/'protocol.json',tmp_path),checkpoint=seal(dict(fixture=True)),
        role='selection',operation='score',artifacts={'selection_supervision':[],'selection_candidates':[]},
        source_roots={'selection_supervision':'truth','selection_candidates':'candidates'},settings=dict(study=study['identity'])))
    write(tmp_path/'job.json',job)
    assert api.role('job.json','output')==0 and calls==['qualified','scored']


@pytest.mark.parametrize('damage',['missing','tampered'])
def test_lineage_missing_or_changed_original_blocks_before_checkpoint_access(tmp_path,monkeypatch,damage):
    p=policy()
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    original=tmp_path/'prepared/research/prepare';original.mkdir(parents=True);(tmp_path/'pilot').mkdir()
    cell=original/'preparation/role/private/prepared/cells/selection-parameter/private/selected.json'
    write(cell,[]);ref=reference(cell,original)
    receipt=seal(dict(status='PASS',stage='prepare',original=None,inventory=[ref] if damage=='tampered' else []))
    write(original/'result.json',receipt);monkeypatch.setattr(p,'PARENT_PREPARATION',receipt['identity'])
    if damage=='tampered':cell.write_text('[{}]')
    with pytest.raises(ValueError,match='original exclusion (cell missing|input changed)'):
        p.lineage(tmp_path,'pilot','prepared',{})


@pytest.mark.parametrize('change',[None,'checkpoint','cohort','arm'])
def test_prospective_role_rejects_refrozen_job_outside_committed_protocol(tmp_path,monkeypatch,change):
    from types import SimpleNamespace
    import fresh_pilot as api
    import foundation_local
    from oeis_learn.pipeline_validation import generalization_commands,generalization_runtime,generalization_roles
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    monkeypatch.setattr(api,'ROOT',tmp_path);monkeypatch.setattr(foundation_local,'Host',lambda args:SimpleNamespace(args=args))
    images=[digest('cpu'),digest('gpu')];write(tmp_path/'qualified/qualification.json',dict(images=images))
    weights=dict(path='weights.pt',bytes=1,sha256=digest('weights'));checkpoint=seal(dict(weights=weights))
    truth=dict(path='truth.json',bytes=1,sha256=digest('truth'))
    protocol=seal(dict(version='prospective-retained-protocol/v1',checkpoint_lineage={'static':dict(checkpoint=checkpoint)},targets=[truth],scoring_role_seconds=1200))
    write(tmp_path/'protocol.json',protocol);study=dict(identity=digest('study'),hypothesis=protocol['identity'])
    runtime=dict(purpose='research',images=images,storage={})
    monkeypatch.setattr(generalization_runtime,'load',lambda *a:(tmp_path,study,runtime))
    calls=[];monkeypatch.setattr(generalization_commands,'_qualification',lambda *a:calls.append('qualified'))
    monkeypatch.setattr(generalization_roles,'project_and_invoke',lambda *a,**k:dict(status='PASS'))
    for n in ('truth','candidates','study'):(tmp_path/n).mkdir()
    job=dict(version='fresh-pilot-role/v1',engine='podman',qualified='qualified',manifest='study.json',arm='static',
        study=study['identity'],study_root='study',protocol=reference(tmp_path/'protocol.json',tmp_path),checkpoint=checkpoint,
        role='selection',operation='score',artifacts={'selection_supervision':[truth],'selection_candidates':[]},
        source_roots={'selection_supervision':'truth','selection_candidates':'candidates'},settings=dict(study=study['identity'],targets=truth))
    if change=='checkpoint':job['checkpoint']=seal(dict(weights=weights,changed=True))
    if change=='cohort':job['settings']['targets']=dict(truth,sha256=digest('other truth'))
    if change=='arm':job.pop('arm')
    write(tmp_path/'job.json',seal(job));assert api.role('job.json','output')==(0 if change is None else 1)
    assert calls==(['qualified'] if change is None else [])


def test_retained_bytes_changed_after_lineage_block_inference(tmp_path,monkeypatch):
    h,root,context,d,q,a,summary=prepared_fixture(tmp_path,monkeypatch)
    (tmp_path/'weights-static/weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='retained weights changed'):h.publish_inputs(root,context,d,q,a,summary)
    assert not (root/'research/protocol.json').exists()


def test_export_preserves_canonical_qualification_build_paths(tmp_path,monkeypatch):
    import prospective_evaluation as h
    from oeis_learn.pipeline_validation import generalization_qualification_io as qio
    from oeis_learn.generalization import feature_acceptance
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.bundle import verify
    import zipfile
    root=tmp_path/'run';root.mkdir();q=tmp_path/'qualified';q.mkdir()
    monkeypatch.setattr(h,'ROOT',tmp_path)
    (q/'build/crates/core').mkdir(parents=True);(q/'build/Dockerfile').write_text('FROM pinned\n');(q/'build/crates/core/Cargo.lock').write_text('version = 4\n')
    write(q/'producer.json',seal(dict(status='PASS')))
    write(q/'qualification.json',seal(dict(producer=dict(receipt=reference(q/'producer.json',q)))))
    write(q/'feature.json',seal(dict(status='PASS')))
    refs=[reference(q/n,q) for n in ('build/Dockerfile','build/crates/core/Cargo.lock','producer.json')]
    monkeypatch.setattr(qio,'public_closure',lambda *_:refs)
    monkeypatch.setattr(feature_acceptance,'feature_artifacts',lambda *a,**k:[])
    write(root/'workflow.json',seal(dict(status='FAIL',error='original quota shortfall',stages=[],qualification='qualified')))
    archive=h.export(root);verify(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.read('payloads/build/Dockerfile')==b'FROM pinned\n'
        assert 'payloads/qualification.json' in z.namelist()


def test_package_failure_preserves_original_error(tmp_path,monkeypatch):
    import prospective_evaluation as h
    import fresh_pilot as api
    from oeis_learn.pipeline_validation.identity import write
    root=tmp_path/'run';root.mkdir();monkeypatch.setattr(h,'ROOT',tmp_path);monkeypatch.setattr(api,'ROOT',tmp_path)
    write(root/'workflow.json',seal(dict(status='FAIL',error='original quota shortfall',stages=[])))
    def fail(*_):raise ValueError('unsafe pinned path')
    monkeypatch.setattr(h,'export',fail)
    assert h.main(['package','--run','run'])==0
    result=read_json(root/'workflow.json')
    assert result['error']=='original quota shortfall'
    assert result['failures'][-1]==dict(stage='export',error='ValueError: unsafe pinned path')


def test_existing_run_diagnose_reads_quota_ledger_without_execution_or_mutation(tmp_path,monkeypatch):
    import prospective_evaluation as h
    import fresh_pilot as api
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.bundle import verify
    import zipfile
    root=tmp_path/'run';root.mkdir();monkeypatch.setattr(h,'ROOT',tmp_path)
    family='sha256:'+'a'*64
    write(root/'workflow.json',seal(dict(status='FAIL',error='export: ValueError',stages=[],qualification=None)))
    cell=root/'prepare/cohort/cells/parameter'
    write(cell/'ledger.json',seal(dict(version='generalization-cell-preparation/v1',role='selection',status='BLOCKED',draws=1681,admitted=1681,rejections=[],shortfalls=['family:'+family+':3'],native_validation='PERFORMED')))
    write(root/'prepare/result.json',seal(dict(status='FAIL',reason='ValueError',error='native cohort shortfall or disagreement')))
    write(root/'prepare/private/secret.json',dict(secret='must stay private'))
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('no execution permitted')
    monkeypatch.setattr(api,'invoke',forbidden);monkeypatch.setattr(h,'native_prepare',forbidden)
    assert h.main(['diagnose','--run','run','--output','diagnostic'])==0
    assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}==before
    archive=tmp_path/'diagnostic/report.zip';verify(archive)
    with zipfile.ZipFile(archive) as z:
        report=json.loads(z.read('preparation-diagnostics.json'))
        assert report['cells'][0]['shortfalls']==['family:'+family+':3']
        assert all('secret' not in n for n in z.namelist())
    with pytest.raises(ValueError):h.main(['diagnose','--run','run','--output','run/child'])


def test_exhausted_family_exports_counts_without_loosening_exclusions(tmp_path):
    p=policy();d=small_design()
    from oeis_learn.generalization.preparation import reference_candidate
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    original=[]
    for r in p.draw(d):
        (exact,_),error=reference_candidate(r);assert error is None;original.append(exact)
    write(tmp_path/'old.json',original)
    files=[dict(root=tmp_path,ref=reference(tmp_path/'old.json',tmp_path),name='old',role='selection',holdout='parameter',count=len(original))]
    def forbidden(*_):raise AssertionError('excluded candidates must not reach native work')
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='shortfalls'):p.prepare(d,files,out,execute=forbidden,workers=1,seconds=60)
    report=read_json(out/'availability.json')
    assert all(r['surviving']==0 and r['deficit']==1 and r['exclusion_reasons']['source']>0 for r in report['families'])
    assert all(set(r)=={'stratum','family','required','reference_eligible','surviving','deficit','exclusion_reasons'} for r in report['families'])
    assert not (out/'summary.json').exists()


@pytest.mark.parametrize('coefficient',[-1024,0,1024])
@pytest.mark.parametrize('prefix,parameter,start,other,formula',[
    ('073880ed','p2',49,'p0',lambda n,k:(2*k+1)*n),
    ('7988b229','p0',0,'p1',lambda n,k:(k-1)*n),
    ('99c9a504','p2',9,'p0',lambda n,k:k*n*n+n),
    ('b44aad9a','p0',0,'p1',lambda n,k:(k-1)*n),
    ('faaf9c87','p0',0,'p1',lambda n,k:k*n*n),
])
def test_low_support_branch_collides_despite_different_full_programs(
        coefficient,prefix,parameter,start,other,formula):
    """Synthetic mechanism counterexamples; no protected original data or saturation claim."""
    import re
    from oeis_learn.generalization.targets import ground
    from oeis_learn.synthesis.reference import evaluate
    d=policy().design(json.loads((HERE/'observable-evaluation-protocol.json').read_text()))
    family=next(f for f in d['families'] if f.startswith('sha256:'+prefix))
    template=d['families'][family]['parameter_source']
    assignment={p:'2' for p in re.findall(r'param.get (p[0-9]+)',template)}
    assignment[parameter]=str(coefficient);alternative=dict(assignment);alternative[other]='3'
    sources=[ground(template,a) for a in (assignment,alternative)]
    values=[list(evaluate(s,tuple(map(str,range(100)))).values) for s in sources]
    expected=[str(formula(n,coefficient)) for n in range(start,start+20)]
    assert sources[0]!=sources[1] and values[0]!=values[1]
    assert values[0][start:start+20]==values[1][start:start+20]==expected
    rows=[dict(id=str(i),family=family,holdout='parameter',source=s,group=str(i),variant=str(i),values=v)
          for i,(s,v) in enumerate(zip(sources,values))]
    index=policy().CandidateExclusions([rows[0]])
    index.observe([rows[1]],role='training',holdout='parameter')
    assert index.rejected()=={'0':{'finite_window'}}
