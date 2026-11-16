"""Full-size construction over explicit synthetic data/qualification boundaries.

These tests do not execute research, claim native provenance or qualify HIP.
"""
import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write,digest,load,record,reference,seal
from oeis_learn.pipeline_validation import progressive_research_bindings as b


@pytest.fixture(scope='module',params=[False,True,'expanded','streamlined'],ids=['original','preparation-v3','runtime-v3','streamlined-v4'])
def compiled(tmp_path_factory,request):
    revised=request.param
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture as learning_fixture
    from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
    root=tmp_path_factory.mktemp('research-construction')
    learning_fixture(root/'original',interrupt=False)
    package=load(root/'original/training-input.json')
    # The native data producer is a named test boundary; no such mutation is
    # available in the production compiler or registered-data adapter.
    shards=[[seal(dict(r,native_verified=True)) for r in shard] for shard in package['shards']]
    rule=seal(dict(batches=[16,32,64]))
    support=seal(dict(version='progressive-prepared-support/v1',policy=rule['identity'],diagnostic=False,
        shards=shards,layout=package['layout'],refresh_updates=1000000,
        **({'refresh_by_batch':{'16':1000000,'32':500000,'64':250000}} if revised else {})))
    summary=seal(dict(status='COMPLETE',diagnostic=False,support=support['identity'],policy=rule,
        maximum_target_tokens=max(len(r['tokens']) for s in shards for r in s)))
    atomic_write(root/'support.json',support)
    atomic_write(root/'recipe.json',package['profile']['base_recipe'])
    (root/'weights.pt').write_bytes(b'EXPLICIT TEST BOUNDARY: never loaded as model weights')
    atomic_write(root/'state.json',dict(test_boundary_only=True))
    manifest=dict(retained=dict(checkpoints=dict(D=dict(root='.',descriptor=dict(
        weights=reference(root,'weights.pt'),state=reference(root,'state.json'))))))
    atomic_write(root/'manifest.json',manifest)
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    with ExposureIndex(root/'original-exposure.sqlite'):pass
    atomic_write(root/'candidates.json',[])
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19])),
        candidates=reference(root,'candidates.json'),exposure=reference(root,'original-exposure.sqlite'),
        recipe=reference(root,'recipe.json'),manifest=reference(root,'manifest.json'),
        **({'census_reuse':seal(dict(TEST_BOUNDARY_ONLY=True))} if revised else {})))
    design=scientific_round(data,expanded=revised in {"expanded","streamlined"},streamlined=revised=="streamlined")
    _,rosters,screening=fixture();panels={};reservations={};pubs=[]
    if revised:
        rosters['development']=[dict(r,group=r['id']) for r in rosters['development']]
        from oeis_learn.generalization.progressive.prepared_panels import screening as select
        screening,_=select([dict(r,role='development') for r in rosters['development']],seed=17,grouping='balanced-at-most-two/v2')
    exposure=record('exposure',dict(previous=None,events=[],detector_limits=['TEST BOUNDARY']),
        created_at='2026-11-11T00:00:00Z',purpose='research',track='synthetic_strict')
    atomic_write(root/'exposure.json',exposure)
    from oeis_learn.generalization.progressive.prepared_panels import cohort
    for name,rows in rosters.items():
        reserved=[dict(id=r['id'],group=r['group'],stratum='parameter' if name.startswith('retention') else r['stratum'],
            role=name,indices=list(map(str,range(100))),values=list(map(str,range(100))),
            source=None if r['stratum']=='oeis' else '(program (return $index))',operations=[]) for r in rows]
        role='assessment' if name=='assessment' else 'development' if name in ('development','retention-development') else 'confirmation'
        panel=cohort(root,root/name,reserved,panel=name,role=role,decision='test-boundary',
            exposure=exposure,created_at=exposure['created_at'])
        atomic_write(root/name/'cohort.json',panel)
        panels[name]=panel;reservations[name]=reserved
        pubs.append(dict(panel=name,role=role,status='COMPLETE',cohort=reference(root,root/name/'cohort.json')))
    bound=dict(roster=b.construct(design,rosters,screening),panels=panels,reservations=reservations,
               screening=screening,support=support,support_summary=summary,exposure=exposure)
    # Only the native data producer is doubled. Actual material binding reads
    # every original reservation/cohort/support byte; inputs() is not patched.
    from oeis_learn.pipeline_validation.progressive_research_bridge import request
    req=request(data);atomic_write(root/'census/request.json',req)
    atomic_write(root/'census/private/panel-reservations.json',reservations)
    native=digest('EXPLICIT NATIVE PRODUCER BOUNDARY')
    census=seal(dict(version='progressive-research-census/v1',request=req['identity'],diagnostic=False,native=native,
        artifacts=[reference(root/'census','private/panel-reservations.json')]))
    atomic_write(root/'census/census.json',census)
    atomic_write(root/'support-summary.json',summary);atomic_write(root/'screening.json',screening)
    materials=seal(dict(version='progressive-research-materials/v1',status='COMPLETE',diagnostic=False,deficits=[],
        census=reference(root,'census/census.json'),request=reference(root,'census/request.json'),native=native,
        panels=pubs,support=reference(root,'support.json'),support_summary=reference(root,'support-summary.json'),
        screening=reference(root,'screening.json'),exposure=reference(root,'exposure.json')))
    for name,value in [('registered-data',data),('design',design),('materials',materials)]:
        atomic_write(root/(name+'.json'),value)
    assert b.inputs(root,data,design,materials)==bound
    atomic_write(root/'qualification/source.json',{'test_boundary_only':True})
    q=dict(source_closure=reference(root/'qualification','source.json'),images=[digest('cpu'),digest('hip')],
           runtime={'execution_policy':digest('TEST EXECUTION POLICY BOUNDARY')})
    atomic_write(root/'qualification/qualification.json',q)
    args=b.compile_plan(root,root/'compiled',data,design,materials,qualification=root/'qualification',checkpoint_bytes=64<<20)
    result=b.finish_plan(*args,data=data,design=design,materials=materials)
    return root,args,result


def test_panel_campaign_rebind_preserves_full_scientific_task_domain(compiled):
    import copy
    from oeis_learn.pipeline_validation.progressive_panel_campaign import invariant_tasks,bind_calibration
    from oeis_learn.pipeline_validation.progressive_calibration_reuse import raw_prepared,named,bind_original_reservation
    from oeis_learn.pipeline_validation.progressive_runtime_timing import feasibility_scopes,TARGET
    root,(_,bound,study,*_),_=compiled
    if not study['version'].endswith('/v4'):return
    materials=load(root/'materials.json');data=load(root/'registered-data.json');design=load(root/'design.json')
    # Change actual cohort membership, then use the real compiler and all 1,296
    # exact-input resolutions. No detector, model or hardware success is mocked.
    reservations=copy.deepcopy(bound['reservations'])
    from oeis_learn.generalization.progressive.prepared_panels import cohort
    material=copy.deepcopy(materials);screen=copy.deepcopy(bound['screening'])
    for name in ('development','confirmation-diversity','confirmation-tuning-0'):
        original_id=reservations[name][0]['id'];reservations[name][0]['id']+='-replacement'
        for row in screen:
            if row['id']==original_id:row['id']+='-replacement'
        role='development' if name=='development' else 'confirmation'
        panel=cohort(root,root/'revised/private'/name,reservations[name],panel=name,role=role,
            decision='test-boundary',exposure=bound['exposure'],created_at=bound['exposure']['created_at'])
        atomic_write(root/'revised'/name/'cohort.json',panel)
        for p in material['panels']:
            if p['panel']==name:p['cohort']=reference(root,root/'revised'/name/'cohort.json')
    atomic_write(root/'revised/screening.json',screen);material['screening']=reference(root,'revised/screening.json')
    atomic_write(root/'revised/private/panel-reservations.json',reservations)
    census=load(root/'census/census.json')
    census=seal(dict(census,artifacts=[reference(root/'revised','private/panel-reservations.json')]))
    atomic_write(root/'revised/census.json',census)
    material['census']=reference(root,'revised/census.json')
    material=seal(material)
    b.build(root,root/'rebound',data,design,material,qualification=root/'qualification',checkpoint_bytes=64<<20)
    old=load(root/'compiled/blueprint.json');new=load(root/'rebound/blueprint.json')
    oldraw=raw_prepared(root,old);newraw=raw_prepared(root,new)
    checked=invariant_tasks(oldraw['payload']['task_plans'],newraw['payload']['task_plans'])
    assert len(checked['derived_seed_changes'])==256 and not checked['mismatches']
    assert oldraw['payload']['task_plans']!=newraw['payload']['task_plans']
    from oeis_learn.pipeline_validation.progressive_outputs import validate_requests
    from oeis_learn.pipeline_validation.progressive_study import validate_frozen
    plan=newraw['payload']
    validate_requests(root,plan['task_plans'],plan['references'],plan['graph'],purpose='research')
    validate_frozen(root,dict(purpose='research',track='synthetic_strict',payload=dict(tasks=plan['task_plans'],references=plan['references'])))
    for field in ('token_target','seed','caps','compatibility'):
        changed=copy.deepcopy(newraw['payload']['task_plans'])
        task=next(p for p in changed if p['kind']=='training')
        task[field]=0
        with pytest.raises(ValueError,match='scientific tasks'):invariant_tasks(oldraw['payload']['task_plans'],changed)
    # Named original measurement authorities are test-only metadata; the real
    # reservation binder and budget guard still verify every retained reference.
    proof=dict(blueprint=reference(root,'compiled/blueprint.json'))
    atomic_write(root/'original-request.json',dict(caps=load(root/'compiled/calibration-plan.json')['caps']))
    proof['request']=reference(root,'original-request.json')
    atomic_write(root/'original-proof.json',proof)
    atomic_write(root/'original-resource-policy.json',dict(test_boundary=True))
    request=seal(dict(cases=feasibility_scopes(),target_tokens=TARGET))
    atomic_write(root/'original-trajectory-request.json',request)
    atomic_write(root/'original-trajectory.json',dict(request=reference(root,'original-trajectory-request.json')))
    for name,path in [('calibration_reuse','original-proof.json'),('research_resource_policy','original-resource-policy.json'),('training_trajectory','original-trajectory.json')]:
        oldraw['payload']['references']=[r for r in oldraw['payload']['references'] if r['name']!=name]
        oldraw['payload']['references'].append(dict(name=name,kind='policy',role='custodian',artifact=reference(root,path)))
    payload=copy.deepcopy(oldraw['payload']);payload.pop('blueprint')
    atomic_write(root/'augmented-plan.json',dict(version='progressive-preparation-plan/v1',payload=payload))
    augmented=copy.deepcopy(old)
    for r in augmented['payload']['references']:
        if r['name']=='preparation_plan':r['artifact']=reference(root,'augmented-plan.json')
    result=bind_calibration(root,seal(augmented),new,root/'replay',bind_reservation=bind_original_reservation)
    rebound=raw_prepared(root,result)
    assert named(result,'iteration_calibration_plan')==named(old,'iteration_calibration_plan')
    for name in ('calibration_reuse','research_resource_policy','training_trajectory'):
        assert named(rebound,name)==named(oldraw,name)
    from oeis_learn.pipeline_validation.progressive_preparation_preflight import verified
    assert verified(root,rebound['payload']['references'])['original_reservation']==named(rebound,'research-reservation-v2')


def test_full_registered_plan_has_real_closed_requests_and_exact_cases(compiled):
    root,(compiler,bound,study,*_),summary=compiled
    assert summary['tasks']==(352 if study['version'].endswith('/v4') else 376) and summary['exact_calibration_cases']==1296
    assert summary['executable'] is False and summary['assessment_exposed'] is False
    from oeis_learn.pipeline_validation.progressive_outputs import validate_requests
    from oeis_learn.pipeline_validation.progressive_study import validate_frozen
    plan=load(root/'compiled/preparation-plan.json',maximum=128<<20)['payload']
    blueprint=load(root/'compiled/blueprint.json')
    record('prepared',dict(plan,blueprint=blueprint['identity']),created_at=blueprint['created_at'],
           purpose='research',track='synthetic_strict')
    assert 4096<len(plan['references'])<16384
    # Literal request validation needs one task per identity, not an expanded
    # list repeating each ID hundreds of times for its finite alternatives.
    validate_requests(root,plan['task_plans'],plan['references'],plan['graph'],purpose='research')
    validate_frozen(root,dict(purpose='research',track='synthetic_strict',payload=dict(tasks=plan['task_plans'],references=plan['references'])))
    requests=[load(root/r['artifact']['path']) for r in plan['references'] if r['kind']=='policy' and r['name'].startswith('requests/')]
    assert len(requests)==summary['tasks']
    from oeis_learn.generalization.progressive.budgets import role_work_requirements
    audit=next(r for r in requests if r['operation']=='audit-language')
    need=role_work_requirements('audit-language',[load(root/p) for p in audit['artifacts']['sources']])
    assert all(audit['reservation'][k]>=v for k,v in need.items())
    assert not any(r['operation']=='assess' for r in requests)
    assert all(not (isinstance(v,str) and '/assessment/' in v) for r in requests for vs in r['artifacts'].values() for v in vs)
    # Independent same-panel baseline + candidate + retention for every pair.
    for item in bound['roster']['comparisons']:
        request=load(root/f"compiled/requests/{item['id']}.json")
        template=load(root/request['artifacts']['public'][0])['data']
        assert len(template['roster'])==640
        assert {r['stratum'] for r in template['roster']}=={'parameter','composition','unseen_structure','oeis','retention'}
    assert (root/'compiled/calibration-plan.json').stat().st_size<4096


def test_compiler_retains_complete_original_finite_domain_and_reference_controls(compiled):
    root,(compiler,bound,study,*_),_=compiled
    rows={t['id']:t for t in compiler.plans}
    assert len(rows['trial-00-1000000-train']['trial_choices'])==162
    assert len(rows['trial-15-4000000-train']['trial_choices'])==162
    assert len(rows['confirmation-1-17']['trial_choices'])==648
    assert len(rows['confirmation-0-17']['trial_choices'])==1
    assert 'trial_choices' not in rows['trial-00-retention-baseline-generate']
    assert rows['confirmation-baseline-1-17-generate']['trial']['ordinal']==0
    assert rows['confirmation-baseline-1-17-generate']['slot']=='tuning-confirmation-1'
    packages=list((root/'compiled/private/packages').glob('*.json'))
    assert packages and all('weights' not in load(p) for p in packages)


def test_immutable_row_validation_is_shared_without_weakening_scope(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.generalization.progressive import shared_support as shared,training_rows
    fixture(tmp_path,interrupt=False);p=load(tmp_path/'training-input.json')
    package=shared.shared(p);before=shared.raw(package)
    calls=[];validate=training_rows.validate_row
    def counted(row,**kw):calls.append(row['identity']);return validate(row,**kw)
    monkeypatch.setattr(training_rows,'validate_row',counted)
    for _ in range(100):shared.validated_rows(package['shards'],native_verified=False)
    assert len(calls)==sum(map(len,p['shards']))
    assert shared.raw(package)==before and seal(package)==package
    with pytest.raises(ValueError,match='native verification'):
        shared.validated_rows(package['shards'],native_verified=True)
    with pytest.raises(ValueError,match='target/codec'):
        shared.validated_rows(package['shards'],native_verified=False,short=True)
    with pytest.raises(TypeError):package['shards'][0][0]['tokens'].append(999)
    # A serialized/mutable copy cannot inherit any in-memory validation.
    import json
    mutable=json.loads(before);row=mutable['shards'][0][0]
    row['tokens']=row['tokens'][:-1];mutable['shards'][0][0]=seal(row)
    with pytest.raises(ValueError,match='target/codec'):
        shared.validated_rows(mutable['shards'],native_verified=False)


def test_shallow_frozen_wrapper_does_not_cache_mutable_descendants(tmp_path):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.generalization.progressive.shared_support import FrozenList,validated_rows
    fixture(tmp_path,interrupt=False);p=load(tmp_path/'training-input.json')
    shards=FrozenList(p['shards']);validated_rows(shards,native_verified=False)
    row=shards[0][0];row['tokens']=row['tokens'][:-1];shards[0][0]=seal(row)
    with pytest.raises(ValueError,match='target/codec'):validated_rows(shards,native_verified=False)


def test_construction_progress_is_packaged_even_when_compilation_fails(tmp_path):
    import json,zipfile
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session
    session=Session(tmp_path,'run-state/progress',{},'prepare-research')
    def fail():
        b.progress(session.folder/'prepare-research/construction','resolving-exact-calibration-inputs',completed_tasks=7,total_tasks=376)
        raise RuntimeError('explicit compiler failure')
    with session.own():session.step('construction',fail)
    with zipfile.ZipFile(session.folder/'share/prepare-research.zip') as archive:
        value=json.loads(archive.read('prepare-research/construction/construction-progress.json'))
        assert value['completed_tasks']==7 and value['total_tasks']==376
        assert set(value)=={'version','identity','phase','unix','completed_tasks','total_tasks'}


@pytest.mark.parametrize('terminate',[False,True])
def test_stdlib_stack_diagnostics_preserve_normal_and_signal_exit(terminate):
    import os,signal,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    code='from oeis_learn.pipeline_validation.progressive_research_bindings import diagnostics;import time,signal\n'
    code+='with diagnostics(interval=0.01):\n print("ready",flush=True)\n '+('signal.pause()' if terminate else 'time.sleep(0.05)')
    child=subprocess.Popen([sys.executable,'-S','-c',code],cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        if terminate:child.send_signal(signal.SIGTERM)
        _,errors=child.communicate(timeout=10)
        assert child.returncode==(-signal.SIGTERM if terminate else 0)
        assert 'research construction stack: '+('termination' if terminate else 'periodic') in errors
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_confirmation_continuation_is_frozen_and_token_caps_are_incremental(compiled):
    root,(compiler,bound,study,*_),_=compiled
    if not study['version'].endswith('/v4'):return
    from oeis_learn.pipeline_validation.progressive_study import confirmation_chains
    import copy
    plans=compiler.plans
    selected=[p for p in plans if p.get('trial',{}).get('phase')=='confirmation']
    confirmation_chains(selected,study)
    byid={p['id']:p for p in plans}
    for p in selected:
        if p['kind']!='training':continue
        assert 12500000<p['caps']['tokens']<12600000
        if p['token_target']==25000000:
            parent=byid[p['initializer']['parent_task']]
            assert parent['trial_choices']==p['trial_choices']
    for fault in ('seed','parent','choice','missing'):
        altered=copy.deepcopy(selected)
        child=next(p for p in altered if p['kind']=='training' and p['token_target']==25000000)
        if fault=='seed':child['seed']+=1
        if fault=='parent':child['initializer']['parent_task']='unrelated'
        if fault=='choice':child['trial_choices']=[]
        if fault=='missing':altered.remove(child)
        with pytest.raises(ValueError):confirmation_chains(altered,study)


def test_finalist_lock_applies_identical_settings_to_both_confirmation_segments(compiled,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_study as s
    from oeis_learn.generalization.progressive.tuning import BASELINE,CONTROL
    root,(compiler,bound,study,*_),_=compiled
    if not study['version'].endswith('/v4'):return
    selected=dict(selected=['0','1'],seeds=[17,19])
    atomic_write(root/'selection-test/finalists.json',selected)
    atomic_write(root/'selection-test/result.json',{})
    journal=SimpleNamespace(root=tmp_path,trials={str(i):dict(proposal=dict(parameters=params)) for i,params in enumerate([BASELINE,CONTROL])})
    monkeypatch.setattr(s,'journal',lambda *a:journal)
    monkeypatch.setattr(s,'policy',lambda *a:study)
    store=SimpleNamespace(campaign=dict(payload=dict(tasks=compiler.plans)),state=dict(choices={},tasks={},outputs={
        'tuning-selection':dict(result=reference(root,'selection-test/result.json'))}))
    store.append=lambda state:setattr(store,'state',state)
    task=dict(payload=dict(id='tuning-selection',trial=dict(study=study['identity'])))
    result=dict(metrics=dict(finalists=reference(root/'selection-test','finalists.json')))
    s.commit_selection(root,store,task,result)
    for position in (0,1):
        for seed in (17,19):
            name=f'confirmation-{position}-{seed}'
            assert store.state['choices'][name]==store.state['choices'][name+'-part1']
    for position in (2,3):
        for seed in (17,19):
            for suffix in ('','-part1'):assert store.state['tasks'][f'confirmation-{position}-{seed}'+suffix]=='SKIPPED'
    original=dict(store.state['choices']);s.commit_selection(root,store,task,result)
    assert store.state['choices']==original
