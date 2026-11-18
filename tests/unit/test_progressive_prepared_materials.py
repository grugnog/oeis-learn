"""Real data projection mechanics; reference execution never qualifies hardware."""
import json

import pytest

from oeis_learn.generalization.progressive.contracts import digest, record, seal
from oeis_learn.generalization.progressive import prepared_support as support
from oeis_learn.generalization.progressive import prepared_panels as panels
from oeis_learn.generalization.progressive.cohorts import publish_many, validate_frozen_panels
from oeis_learn.pipeline_validation.progressive_research_census import reserve_panels


def composition_pair():
    from oeis_learn.generalization.progressive.generator import structure_id
    sources=['(program (return (add (mul $index $index) (int 10000))))',
             '(program (return (add (int 20000) (mul $index $index))))']
    return [dict(id='unseen-'+str(i),stratum='unseen_structure',source=s,structure=structure_id(s),
                 group='original-'+str(i),indices=list(map(str,range(100))),
                 values=[str((i+1)*10000+n*n) for n in range(100)]) for i,s in enumerate(sources)]


@pytest.mark.parametrize('sharing',[False,True])
def test_portable_panel_revision_matches_corrected_allocator(sharing):
    import copy
    from oeis_learn.pipeline_validation.progressive_panel_revision import regroup
    rows=composition_pair()
    rows.extend(dict(r,id=r['id']+'-parameter',stratum='parameter',group='parameter-'+r['id'],
                     values=[str(int(v)+90000) for v in r['values']]) for r in list(rows))
    rows.append(dict(id='oeis',stratum='oeis',group='original-0',family='source-family',
                     indices=list(map(str,range(100))),values=[str(999999+i**3) for i in range(100)]))
    before=copy.deepcopy(rows)
    for seed in range(8):
        req=dict(seed=seed,required_per_stratum=128,minimum_groups=1,
                 panels=['development','confirmation-diversity'],retention_panels=['retention'])
        if sharing:req['reuse']={}
        expected=reserve_panels(rows,seed=seed,required=128,minimum_groups=1,panels=req['panels'],
                                retention_panels=req['retention_panels'],parameter_sharing=sharing)
        assert regroup(rows,req)==expected
    assert rows==before


def test_panel_revision_reconstructs_original_native_database_without_draws(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_panel_revision import reconstruct,regroup
    from oeis_learn.pipeline_validation import progressive_research_census as census
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    from oeis_learn.generalization.progressive.contracts import reference
    from tests.unit.test_progressive_research_census import inputs,ReferenceDouble
    req=inputs(tmp_path);write(tmp_path/'request.json',req)
    original=census.run(tmp_path,tmp_path/'census',req,client=ReferenceDouble())
    materials=dict(census=reference(tmp_path,tmp_path/'census/census.json'),
        request=reference(tmp_path,tmp_path/'request.json'),native=original['native'],diagnostic=True)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(census,'run',lambda *a,**k:pytest.fail('new native draw'))
    rows,roster,result,request,files=reconstruct(tmp_path,materials)
    assert result==original and request==req
    assert regroup(rows,request)[0]==roster
    assert all(p.read_bytes()==raw for p,raw in before.items())
    files['census.sqlite'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):
        reconstruct(tmp_path,materials)


def test_panel_revision_does_not_generate_replacements_for_deficits(tmp_path):
    from oeis_learn.pipeline_validation.progressive_panel_revision import regroup
    rows=composition_pair();req=dict(seed=13,required_per_stratum=128,minimum_groups=16,
        panels=['development','confirmation-diversity'],retention_panels=[],reuse={})
    roster,counts,deficits=regroup(rows,req)
    assert deficits and sum(len(members) for members in roster.values())==2
    assert all(row['id'] in {r['id'] for r in rows} for members in roster.values() for row in members)
    assert any(v['task_deficit'] for v in counts.values())


def test_panel_revision_packages_original_supply_deficits_without_new_work(tmp_path,monkeypatch):
    import zipfile
    from oeis_learn.pipeline_validation import progressive_panel_revision as revision
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,write,NAMES
    from oeis_learn.pipeline_validation.progressive_research_census import run
    from oeis_learn.generalization.progressive.contracts import reference
    from tests.unit.test_progressive_research_census import inputs,ReferenceDouble
    req=inputs(tmp_path);write(tmp_path/'request.json',req)
    c=run(tmp_path,tmp_path/'run-state/original/census',req,client=ReferenceDouble())
    materials=dict(census=reference(tmp_path,tmp_path/'run-state/original/census/census.json'),
        request=reference(tmp_path,tmp_path/'request.json'),native=c['native'],diagnostic=True)
    write(tmp_path/'materials.json',materials)
    s=Session(tmp_path,'run-state/revision',{},'panel-revision')
    script=tmp_path/'run-state/revision.py';script.write_text('fixture')
    # Original ownership is an explicit boundary double; reconstruction and
    # original native outcomes, allocation, journal and ZIP are real.
    monkeypatch.setattr(revision,'authenticated_materials',lambda _: (materials,dict(identity=digest('audit')),reference(tmp_path,tmp_path/'materials.json')))
    before={p:p.read_bytes() for p in (tmp_path/'run-state/original').rglob('*') if p.is_file()}
    NAMES.update({'panel-revision-allocation.json','panel-revision-progress.json'})
    with s.own():
        assert s.step('regroup',lambda:revision.revise_panels(s,script)) is None
    assert s.journal['steps']['panel-revision/regroup']['status']=='BLOCKED'
    with zipfile.ZipFile(s.folder/'share/panel-revision.zip') as z:
        report=json.loads(z.read('panel-revision/panel-revision-allocation.json'))
        assert report['status']=='INCOMPLETE' and report['deficits']
        assert not report['research_ready'] and report['new_draws']==report['model_calls']==0
        assert not any('/private/' in n for n in z.namelist())
    assert all(p.read_bytes()==raw for p,raw in before.items())


def split_composition_panels(root):
    exposure=record('exposure',dict(previous=None,detector_limits=['exact'],events=[]),
        created_at='2026-11-11T00:00:00Z',purpose='acceptance',track='synthetic_strict')
    result=[]
    for row,name in zip(composition_pair(),['development','confirmation-diversity']):
        result.append(panels.cohort(root,root/'private'/name,[dict(row,role=name)],panel=name,
            role='development' if name=='development' else 'confirmation',decision='diversity',
            exposure=exposure,required=1,minimum_groups=1,purpose='acceptance',created_at=exposure['created_at']))
    return result,exposure


def test_unseen_composition_relation_is_indivisible_before_freeze(tmp_path):
    from oeis_learn.generalization.progressive.cohorts import validate_cohort
    rows=composition_pair()
    assert rows[0]['structure']!=rows[1]['structure']
    assert panels.composition(rows[0])==panels.composition(rows[1])
    old,e=split_composition_panels(tmp_path)
    for c in old:validate_cohort(c,exposure=e,other_cohorts=old)
    with pytest.raises(ValueError,match='cross-panel detectable'):
        validate_frozen_panels(tmp_path,old,{e['identity']:e})
    for sharing in (False,True):
        for seed in range(8):
            roster,_,_=reserve_panels(rows,seed=seed,required=128,minimum_groups=1,
                panels=['development','confirmation-diversity'],retention_panels=(),parameter_sharing=sharing)
            owners={r['id']:name for name,members in roster.items() for r in members}
            assert owners['unseen-0']==owners['unseen-1']


def test_panel_audit_classifies_composition_without_exporting_values(tmp_path):
    from oeis_learn.pipeline_validation.progressive_panel_audit import audit
    cs,e=split_composition_panels(tmp_path)
    progress=[]
    result=audit(tmp_path,cs,{e['identity']:e},progress=progress.append)
    assert result['status']=='OVERLAPS_FOUND' and not result['errors']
    assert len(result['panel_pairs'])==1
    reason=result['panel_pairs'][0]['reasons']
    assert set(reason)=={'composition'}
    assert reason['composition']['strata']=={'unseen_structure/unseen_structure':1}
    assert reason['composition']['row_pair_occurrences']==1
    assert progress[-1]==dict(processed=2,total=2,errors=0)
    encoded=json.dumps(result)
    assert 'unseen-0' not in encoded and '(program' not in encoded and '10001' not in encoded
    assert not result['scientific_authority'] and not result['assessment_opened']


def test_panel_audit_detects_shift_and_continues_after_bad_truth(tmp_path):
    from oeis_learn.pipeline_validation.progressive_panel_audit import audit
    from tests.unit.test_progressive_cohorts import fixture
    a,e=fixture(tmp_path,panel='a',counts=(2,0,0,0))
    b,_=fixture(tmp_path,panel='b',counts=(2,0,0,0))
    (tmp_path/a['payload']['rows'][0]['truth']['path']).write_text('PRIVATE_DAMAGED_TRUTH')
    result=audit(tmp_path,[a,b],{e['identity']:e})
    assert result['status']=='INCOMPLETE' and len(result['errors'])==1
    assert sum(p['checked'] for p in result['panels'])==3
    assert result['panel_pairs'][0]['reasons']['window']['shared_keys']>0
    assert 'PRIVATE_DAMAGED_TRUTH' not in json.dumps(result)


def test_panel_audit_respects_parameter_sharing_and_assessment_lock(tmp_path):
    from oeis_learn.pipeline_validation.progressive_panel_audit import audit
    cs,e=split_composition_panels(tmp_path)
    for c in cs:
        c['payload']['rows'][0]['stratum']='parameter'
        c['payload']['rows'][0]['structure']=digest('shared-parameter-structure')
    cs=list(map(seal,cs))
    assert audit(tmp_path,cs,{e['identity']:e})['status']=='NO_DIRECT_OVERLAPS_FOUND'
    cs[1]['payload']['role']='assessment';cs[1]=seal(cs[1])
    with pytest.raises(ValueError,match='reserved assessment'):
        audit(tmp_path,cs,{e['identity']:e})


def test_panel_audit_original_preparation_and_failure_packaging(tmp_path,monkeypatch):
    import zipfile
    from pathlib import Path
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_panel_audit as module
    from oeis_learn.pipeline_validation import supervision
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,write,read,NAMES
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.generalization.progressive.contracts import reference
    from tests.unit.test_progressive_freeze import prepared
    from tests.helpers.progressive_fixture import blueprint
    cs,e=split_composition_panels(tmp_path)
    p=prepared(tmp_path,blueprint())
    refs=[]
    for i,c in enumerate([*cs,e]):
        path=tmp_path/'private'/f'group-{i}.json';write(path,c)
        refs.append(dict(name='group-'+str(i),role='custodian',kind='groups',artifact=reference(tmp_path,path)))
    p['payload']['references']=[r for r in p['payload']['references'] if r['kind']!='groups']+refs
    folder=tmp_path/'run-state/session'
    path=folder/'prepare-research/reservation-repair/prepared/prepared.json';write(path,seal(p))
    result_path=path.parent/'result.json'
    write(result_path,seal(dict(version='discovery-result/v1',operation='prepare',mechanical_status='COMPLETE',
        producer_error=None,missing_evidence=[],metrics=dict(prepared=reference(path.parent,path)))))
    script=tmp_path/'run-state/audit.py';script.write_text('auditor fixture')
    source={'fixture':digest('original')}
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda _:source)
    s=Session(tmp_path,'run-state/session',dict(source=digest(source)),'prepare-research')
    write(folder/'journal.json',seal(dict(version='progressive-workstation-journal/v1',errors=[],steps={
        'prepare-research/freeze':dict(status='FAILED',reason='cross-panel detectable related group leakage'),
        'panel-audit/inspect':dict(status='FAILED',reason='one completed original preparation publication required')})))
    # Exercise production journal serialization; only the external process is a
    # double. The real wrapper records result.json, never prepared.json directly.
    monkeypatch.setattr(supervision,'launch',lambda argv,logs,**kw:write(logs/'process.json',dict(returncode=0)))
    with s.own():
        w=Workflow(s,SimpleNamespace(phase='prepare-research',engine='podman',memory_gib=64,cpus=8,render_node=None))
        w.run_command('prepare','prepare',resume=True,seconds=60,output=str(path.parent.relative_to(tmp_path)))
    assert all(Path(r['path']).name!='prepared.json' for r in s.journal['steps']['prepare-research/prepare']['outputs'])
    s.phase='panel-audit'
    original=read(folder/'journal.json')['steps']
    files={x:x.read_bytes() for x in (tmp_path/'private').rglob('*') if x.is_file()}
    monkeypatch.setattr(module,'main',lambda *a:pytest.fail('nested runner'))
    NAMES.update({'panel-audit.json','panel-audit-progress.json'})
    with s.own():
        result=s.step('inspect-v2',lambda:module.run(s,script))
    assert result['status']=='OVERLAPS_FOUND'
    assert all(x.read_bytes()==b for x,b in files.items())
    for k,v in original.items():assert read(folder/'journal.json')['steps'][k]==v
    with zipfile.ZipFile(folder/'share/panel-audit.zip') as z:
        assert 'panel-audit/panel-audit.json' in z.namelist()
        assert not any('/private/' in n for n in z.namelist())
    with s.own():
        assert s.step('inspect-v2',lambda:pytest.fail('duplicate audit'))==result
    before=path.read_bytes()
    path.write_bytes(before+b' ')
    with pytest.raises(ValueError,match='identity'):
        module.prepared_publication(s)
    path.write_bytes(before)
    before=result_path.read_bytes();result_path.write_bytes(before+b' ')
    with pytest.raises(ValueError,match='identity'):
        module.prepared_publication(s)
    result_path.write_bytes(before)
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda _:{'changed':True})
    with pytest.raises(ValueError,match='original qualified checkout'):
        module.run(s,script)


@pytest.fixture(scope='module')
def reservoir():
    from oeis_learn.generalization.progressive.generator import default_policy,draw
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    rows=[]
    for i in range(512):
        row,_=verify_draw(draw(default_policy(seed=13),i))
        if row:
            rows.append(seal(dict(row,id='mechanical:'+row['draw_identity'])))
    return rows


def test_support_projection_preserves_native_boundary_and_refreshes(reservoir):
    from oeis_learn.generalization.progressive.supply import plan,shard_index
    rule=support.policy(seed=13,shard_rows=16,batches=(16,),target_tokens=1000000)
    data,report=support.project(reservoir,rule,diagnostic=True)
    assert report['status']=='COMPLETE' and report['diagnostic']
    assert report['original_draws']==report['selected_draws']+sum(report['omissions'].values())
    assert data==support.project(list(reversed(reservoir)),rule,diagnostic=True)[0]
    assert not any('id' in r for s in data['shards'] for r in s)
    probes={p['id'] for p in data['layout']['probes'].values()}
    rows=[r for s in data['shards'] for r in s]
    pw=set().union(*(support._windows(r) for r in rows if r['draw_identity'] in probes))
    assert all(not support._windows(r)&pw for r in rows if r['draw_identity'] not in probes)
    structures=set()
    for i,shard in enumerate(data['shards']):
        fresh={r['structure'] for r in shard if support.bin_name(r)=='frontier' and r['draw_identity'] not in probes}
        assert fresh and not fresh&structures
        structures.update(fresh)
        assert shard_index('refreshed','frontier',i*data['refresh_updates'],data['refresh_updates'],len(data['shards']))==i
    package=dict(data,update_policy=dict(effective_batch=16))
    plan(package,1000000,arm='refreshed')
    with pytest.raises(ValueError,match='native verification'):
        support.project(reservoir,rule)


def test_support_refuses_replacement_duplicate_and_missing_bin(reservoir):
    rule=support.policy(seed=13)
    with pytest.raises(ValueError,match='duplicate'):
        support.project([*reservoir,reservoir[0]],rule,diagnostic=True)
    bad=[dict(reservoir[0],membership='evaluation')]
    with pytest.raises(ValueError,match='changed original'):
        support.project(bad,rule,diagnostic=True)
    data,result=support.project([r for r in reservoir if support.bin_name(r)=='familiar'],rule,diagnostic=True)
    assert result['status']=='INCOMPLETE' and not data['shards']
    assert 'training_probe_bins_missing' in result['deficits']
    with pytest.raises(ValueError,match='complete original'):
        support.training_input(data,result,{}, {})


def test_finite_support_cannot_pretend_to_refresh_before_target(reservoir):
    data,result=support.project(reservoir,support.policy(seed=13,max_shards=2),diagnostic=True)
    assert result['status']=='INCOMPLETE'
    assert any('refresh_not_guaranteed' in d for d in result['deficits'])
    assert result['refresh_updates']%8==0


def test_original_relations_survive_reservation_and_projected_truth(tmp_path):
    rows=[dict(id='OEIS:'+str(i),indices=list(map(str,range(10,110))),
               values=[str(100000*i+n*n) for n in range(100)],
               stratum='oeis',family='original-family-'+str(i),group='original-group-'+str(i)) for i in range(30)]
    reservations,_,_=reserve_panels(rows,seed=13,required=128,minimum_groups=16,
        panels=['development','assessment'])
    chosen=next(r for r in reservations['development'] if r['stratum']=='oeis')
    assert chosen['declared_groups']==sorted(['original-family-'+chosen['id'].split(':')[1],
                                            'original-group-'+chosen['id'].split(':')[1]])
    exposure=record('exposure',dict(previous=None,detector_limits=['exact'],events=[]),
        created_at='2026-11-11T00:00:00Z',purpose='acceptance',track='synthetic_strict')
    c=panels.cohort(tmp_path,tmp_path/'projection',[chosen],panel='development',role='development',
        decision='capability',exposure=exposure,required=1,minimum_groups=1,purpose='acceptance',
        created_at=exposure['created_at'])
    row=c['payload']['rows'][0]
    assert set(chosen['declared_groups'])<=set(row['groups'])
    truth=json.loads((tmp_path/row['truth']['path']).read_text())
    assert set(truth)=={'id','indices','values'} and truth['indices'][0]=='10'
    # Missing other strata remain an explicit deficit, never a smaller panel.
    outcomes=publish_many(tmp_path,[('published',c)],exposure=exposure)
    assert outcomes[0]['status']=='INCOMPLETE'
    assert not (tmp_path/'published/public').exists()


def test_family_history_cannot_be_hidden_by_component_group(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.generalization.progressive.indexed_history import open_indexes,check_panel
    from tests.unit.test_progressive_cohorts import fixture
    c,e=fixture(tmp_path,counts=(1,0,0,0))
    with ExposureIndex(tmp_path/'history.sqlite') as index:
        index.add(dict(id='old',indices=list(map(str,range(100))),values=[str(n+10000) for n in range(100)],
                       family='retained-family'),origin='original',role='training')
    e=seal(dict(e,payload=dict(e['payload'],indexes=[reference(tmp_path,'history.sqlite')])))
    c['payload']['rows'][0]['groups']=['new-component','retained-family']
    c=seal(dict(c,payload=dict(c['payload'],exposure=e['identity'])))
    with open_indexes(tmp_path,[e]) as indexes:
        with pytest.raises(ValueError,match='indexed original exposure'):
            check_panel(tmp_path,c,indexes)


def test_reserved_component_is_primary_without_losing_original_family_keys(tmp_path):
    exposure=record('exposure',dict(previous=None,detector_limits=['exact'],events=[]),
        created_at='2026-11-11T00:00:00Z',purpose='acceptance',track='synthetic_strict')
    rows=[dict(id=f'member-{i}',role='development',group='reserved-component',declared_groups=[f'family-{i}'],
               stratum='oeis',indices=list(map(str,range(100))),values=[str(i*1000+n) for n in range(100)])
          for i in range(2)]
    result=panels.cohort(tmp_path,tmp_path/'projection',rows,panel='development',role='development',
        decision='capability',exposure=exposure,required=1,minimum_groups=1,purpose='acceptance',
        created_at=exposure['created_at'])
    projected=result['payload']['rows']
    assert projected[0]['groups'][0]==projected[1]['groups'][0]=='0000000000000000:component:reserved-component'
    for i,row in enumerate(projected):
        assert {'reserved-component',f'family-{i}'}<=set(row['groups'])
    rows[0]['declared_groups']=['0000000000000000:collision']
    with pytest.raises(ValueError,match='namespace'):
        panels.cohort(tmp_path,tmp_path/'collision',rows,panel='development',role='development',
            decision='capability',exposure=exposure,required=1,minimum_groups=1,purpose='acceptance',
            created_at=exposure['created_at'])


def test_batch_publication_rejects_relations_before_first_channel(tmp_path):
    from tests.unit.test_progressive_cohorts import fixture
    a,e=fixture(tmp_path,panel='a')
    b,_=fixture(tmp_path,panel='b')
    b['payload']['rows'][0]['groups']=a['payload']['rows'][0]['groups']
    b=seal(b)
    with pytest.raises(ValueError,match='cross-panel'):
        publish_many(tmp_path,[('first',a),('second',b)],exposure=e)
    assert not (tmp_path/'first').exists()


def test_original_census_to_materials_retains_deficits_and_private_bytes(tmp_path):
    from tests.unit.test_progressive_research_census import inputs,ReferenceDouble
    from oeis_learn.pipeline_validation.progressive_research_census import run
    from oeis_learn.pipeline_validation.progressive_research_materials import build
    from oeis_learn.pipeline_validation.progressive_workstation_state import write,Session
    import zipfile
    req=inputs(tmp_path)
    session=Session(tmp_path,'run-state/projection',{},'prepare-research')
    base=session.folder/'prepare-research'
    write(base/'census/request.json',req)
    run(tmp_path,base/'census',req,client=ReferenceDouble())
    with session.own():
        result=build(tmp_path,str((base/'census/census.json').relative_to(tmp_path)),
            str((base/'census/request.json').relative_to(tmp_path)),
            str((base/'materials').relative_to(tmp_path)),support.policy(seed=req['seed'],shard_rows=req['shard_rows']))
    assert result['status']=='INCOMPLETE' and result['diagnostic']
    assert not result['research_ready'] and not result['original_generation_repeated']
    assert result['panel_censuses'] and result['deficits']
    with zipfile.ZipFile(session.folder/'share/prepare-research.zip') as archive:
        assert 'prepare-research/materials/materials.json' in archive.namelist()
        assert 'prepare-research/materials/support-summary.json' in archive.namelist()
        assert not any('/private/' in name for name in archive.namelist())
        assert all('(program' not in archive.read(name).decode(errors='replace') for name in archive.namelist())
    with pytest.raises(ValueError,match='no automatic replacement'):
        build(tmp_path,str((base/'census/census.json').relative_to(tmp_path)),
            str((base/'census/request.json').relative_to(tmp_path)),
            str((base/'materials').relative_to(tmp_path)),support.policy(seed=req['seed'],shard_rows=req['shard_rows']))


def test_screening_uses_fixed_two_by_sixteen_groups_and_reports_shortages():
    rows=[dict(id=f'{stratum}-{group}-{i}',group=f'{stratum}-{group}',stratum=stratum,role='development')
          for stratum in panels.STRATA for group in range(18) for i in range(8)]
    roster,summary=panels.screening(rows,seed=13)
    assert len(roster)==128 and summary['status']=='COMPLETE'
    assert roster==panels.screening(list(reversed(rows)),seed=13)[0]
    assert set(r['id'] for r in roster)<={r['id'] for r in rows}
    _,short=panels.screening([r for r in rows if r['stratum']!='composition'],seed=13)
    assert short['status']=='INCOMPLETE' and short['deficits']==['screening_support:composition']


@pytest.mark.parametrize('expanded',[False,True])
def test_round_retains_full_optimization_domain_and_separate_panels(tmp_path,expanded):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
    from oeis_learn.pipeline_validation.progressive_research_qualification import selected_profiles,qualification_blueprint
    from oeis_learn.pipeline_validation.progressive_qualification import matrix
    from oeis_learn.pipeline_validation.progressive_research_bridge import request
    from oeis_learn.pipeline_validation.progressive_handoff import named_plan
    write(tmp_path/'recipe.json',resolve(dict(version='bigint-bootstrap/v1')))
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19])),
        recipe=reference(tmp_path,'recipe.json'),candidates={},exposure={}))
    design=scientific_round(data,expanded=expanded)
    assert design['selected_stages']==['capability','diversity','tuning']
    assert design['tuning']['learning_trials']==12 and design['tuning']['sizing_trials']==4
    assert design['tuning']['screen_seconds']==(96*3600 if expanded else 7200)
    assert design['tuning']['screening_looks']==[1000000,2000000,4000000]
    profiles=selected_profiles(tmp_path,data,design)
    assert len(profiles['profiles'])==12
    req=matrix(profiles['profiles'],source=digest('scope'),image=digest('image'),engine='podman',hardware={})
    assert len(req['cases'])==48
    census_request=request(data)
    assert len(census_request['panels'])==7 and len(census_request['retention_panels'])==3
    changed=seal(dict(design,tuning=dict(design['tuning'],learning_trials=1)))
    with pytest.raises(ValueError,match='original prospective'):
        selected_profiles(tmp_path,data,changed)
    write(tmp_path/'matrix.json',req)
    write(tmp_path/'base/qualification.json',{'test_boundary_only':True})
    blueprint=qualification_blueprint(tmp_path,tmp_path/'plans',matrix=tmp_path/'matrix.json',base=tmp_path/'base')
    from oeis_learn.generalization.progressive.contracts import load
    bp=load(blueprint)
    qp=named_plan(tmp_path,bp,'qualification_plan','progressive-qualification-plan/v1',
                  {'matrix','base_qualification','seconds'})
    assert qp['matrix']==reference(tmp_path,'matrix.json') and bp['purpose']=='research'
    assert not bp['payload']['executable']


def test_projected_difficulty_preserves_measured_reference_work_and_unknowns(tmp_path):
    from oeis_learn.generalization.progressive.complexity import measure, validate as validate_difficulty
    source='(program (return (add (mul $index $index) (int 1))))'
    measured=measure(source,reference_steps=600)
    assert measured['nodes']==5 and measured['depth']==3 and measured['work']==600
    unknown=measure(None)
    assert all(unknown[k] is None for k in ('depth','nodes','work'))
    assert all(unknown['unavailable'][k] for k in ('depth','nodes','work'))
    with pytest.raises(ValueError):validate_difficulty(dict(unknown,unavailable={}))
    exposure=record('exposure',dict(previous=None,detector_limits=['exact'],events=[]),
        created_at='2026-11-11T00:00:00Z',purpose='acceptance',track='synthetic_strict')
    rows=[dict(id='known',role='development',group='known',stratum='parameter',source=source,
               difficulty=measured,indices=list(map(str,range(100))),values=[str(i*i+1) for i in range(100)]),
          dict(id='unknown',role='development',group='unknown',stratum='oeis',
               indices=list(map(str,range(100))),values=[str(i**3+99999) for i in range(100)])]
    c=panels.cohort(tmp_path,tmp_path/'new'/'nested',rows,panel='development',role='development',
        decision='capability',exposure=exposure,required=1,minimum_groups=1,purpose='acceptance',created_at=exposure['created_at'])
    assert c['payload']['rows'][0]['difficulty']==measured
    assert c['payload']['rows'][1]['difficulty']==unknown
    from oeis_learn.generalization.progressive.cohorts import census
    counts=census(c,exposure=exposure)['difficulty']
    assert counts['oeis']['work']['unavailable']==1
    assert counts['parameter']['work']['maximum']==600
    assert counts['parameter']['nodes']['minimum']==5


def test_original_reference_work_is_captured_without_reexecuting_draw():
    from oeis_learn.generalization.progressive.generator import default_policy,draw
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    measured={};row,_=verify_draw(draw(default_policy(seed=13),0),measurements=measured)
    assert row is not None and measured['reference_steps']>0
    assert 'difficulty' not in row  # Private census metadata does not enter learner rows.


def test_complexity_counts_ast_nodes_not_literal_digits_or_local_names():
    from oeis_learn.generalization.progressive.complexity import measure
    a=measure('(program (local $b0 bigint) (set $b0 (int 123456789)) (return (get $b0)))')
    assert a['nodes']==3 and a['depth']==2 and a['work'] is None
    assert a['unavailable']['work']
    b=measure('(program (return (select (eq $index (int 0)) (int 99) $index)))')
    assert b['nodes']==6 and b['depth']==3


def test_null_difficulty_cannot_enter_cohort_without_availability_reason(tmp_path):
    from tests.unit.test_progressive_cohorts import fixture
    from oeis_learn.generalization.progressive.contracts import validate
    c,_=fixture(tmp_path)
    c['payload']['rows'][0]['difficulty']['work']=None
    with pytest.raises(ValueError,match='unavailable difficulty'):
        validate('cohort',seal(c))


def test_transfer_strata_use_initial_gradient_not_trimmed_probe_or_refresh_rows(reservoir):
    from oeis_learn.pipeline_validation.progressive_research_census import classify
    rows=reservoir
    data,_=support.project(rows,support.policy(seed=13,shard_rows=16,max_shards=2),diagnostic=True)
    actual,binding=support.transfer_support(data)
    initial={r['structure'] for r in data['shards'][0] if r['draw_identity'] not in binding['probes']}
    assert actual['structures']==initial
    lost=[r for r in rows if r['structure'] not in initial]
    assert lost and all(classify(r,actual)!='parameter' for r in lost)
    assert len(binding['initial_gradient'])==len(data['shards'][0])-2
    assert binding['refreshed_shards']==1


def test_stdlib_custodian_cli_connects_census_and_rejects_stale_output(tmp_path):
    import json,os,subprocess,sys
    from pathlib import Path
    from tests.unit.test_progressive_research_census import inputs,ReferenceDouble
    from oeis_learn.pipeline_validation.progressive_research_census import run
    from oeis_learn.pipeline_validation.progressive_workstation_state import write,sha
    request=inputs(tmp_path);write(tmp_path/'request.json',request)
    run(tmp_path,tmp_path/'nested/census',request,client=ReferenceDouble())
    write(tmp_path/'rule.json',support.policy(seed=request['seed'],shard_rows=request['shard_rows']))
    originals=[p for p in tmp_path.rglob('*') if p.is_file()]
    before={p:sha(p) for p in originals}
    for path in originals:path.chmod(0o400)
    source=Path(__file__).resolve().parents[2]/'src'
    argv=[sys.executable,'-S','-c',
        'import sys;sys.path.insert(0,sys.argv.pop(1));from oeis_learn.pipeline_validation.progressive_research_materials import main;main()',
        str(source),'--root',str(tmp_path),'--census','nested/census/census.json',
        '--request','request.json','--policy','rule.json','--output','new/nested/materials']
    first=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    assert first.returncode==0,first.stderr
    material=json.loads((tmp_path/'new/nested/materials/materials.json').read_text())
    assert material['diagnostic'] and not material['research_ready'] and material['deficits']
    repeated=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    assert repeated.returncode!=0 and 'original materials retained' in repeated.stderr
    assert all(sha(p)==value for p,value in before.items())
    (tmp_path/'partial/output').mkdir(parents=True)
    (tmp_path/'partial/output/sentinel').write_text('retain')
    stale=subprocess.run([*argv[:-1],'partial/output'],capture_output=True,text=True,timeout=60)
    assert stale.returncode!=0 and (tmp_path/'partial/output/sentinel').read_text()=='retain'


def test_batch_specific_cadence_covers_full_horizon_and_refreshes(reservoir):
    from oeis_learn.generalization.progressive.supply import plan
    rule=support.policy(seed=13,batch_cadence=True)
    data,summary=support.project(reservoir,rule,diagnostic=True)
    assert summary['status']=='COMPLETE'
    assert set(data['refresh_by_batch'])=={'16','32','64'}
    for batch in (16,32,64):
        package=support.apply_batch_cadence(dict(data,update_policy={'effective_batch':batch}))
        plan(package,25000000,arm='refreshed')
        assert package['refresh_updates']*summary['maximum_target_tokens']*batch<25000000
    # The support selection is identical; only the frozen timing law changes.
    old,_=support.project(reservoir,support.policy(seed=13),diagnostic=True)
    assert data['shards']==old['shards'] and data['layout']==old['layout']


def test_shorter_horizons_retime_original_support_without_new_data(reservoir):
    import copy
    original,report=support.project(reservoir,support.policy(seed=13,batch_cadence=True),diagnostic=True)
    before=copy.deepcopy((original,report))
    for target in (500000,1000000,2000000):
        data,summary=support.recadence(original,report,target_tokens=target)
        assert summary['status']=='COMPLETE',summary['deficits']
        assert data['shards']==original['shards'] and data['layout']==original['layout']
        assert summary['cadence_origin']['support']==original['identity']
        assert all(b['first_refresh_token_upper_bound']<target for b in summary['bounds'])
        assert all(b['required_shards']<=len(data['shards']) for b in summary['bounds'])
        assert data['refresh_by_batch']!=original['refresh_by_batch']
    assert (original,report)==before
    _,too_early=support.recadence(original,report,target_tokens=2000000,first_look_tokens=1)
    assert too_early['status']=='INCOMPLETE' and len(too_early['deficits'])==3
    for target in (0,True,400000001):
        with pytest.raises(ValueError):support.recadence(original,report,target_tokens=target)


@pytest.mark.parametrize('failure',[None,'construction','prepare','freeze','acceptance','seed-rebind'])
@pytest.mark.parametrize('research_status',['COMPLETE','PAUSED'])
def test_panel_campaign_retains_measurements_and_uses_canonical_execution(tmp_path,monkeypatch,failure,research_status):
    from types import SimpleNamespace
    from pathlib import Path
    import zipfile
    from oeis_learn.pipeline_validation import progressive_panel_campaign as m
    from oeis_learn.pipeline_validation import progressive_research_bindings as b
    from oeis_learn.pipeline_validation import progressive_research_feasibility as f
    from oeis_learn.pipeline_validation import supervision
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,write,NAMES
    from oeis_learn.generalization.progressive.contracts import reference,load
    from oeis_learn.generalization.progressive import campaign as c
    from oeis_learn.pipeline_validation import progressive_device_acceptance as acceptance
    s=Session(tmp_path,'run-state/round',dict(source=digest('source')),'prepare-research')
    base=s.folder/'prepare-research';folder=base/'panel-campaign'
    script=tmp_path/'run-state/continue.py';script.write_text('explicit host script fixture')
    def doc(path,**fields):write(path,seal(fields))
    doc(base/'registered/registered-data.json',test_boundary=True)
    doc(base/'research-round-design.json',test_boundary=True)
    doc(base/'qualification/qualification.json',test_boundary=True)
    doc(base/'reservation-repair/reservation-repair.json',remaining_seconds=100000)
    doc(base/'reservation-repair/prepare-allowance.json',seconds=100000,spent_seconds=100,active=None)
    doc(base/'reservation-repair/blueprint.json',test_boundary=True)
    doc(base/'evaluation-timing/request.json',test_boundary=True)
    doc(base/'preparation-headroom.json',setup_seconds=dict(acceptance=7800))
    doc(base/'frozen/result.json',producer_error=dict(message='cross-panel detectable related group leakage'))
    doc(base/'readiness.json',status='BLOCKED')
    originals={p:p.read_bytes() for p in (base/'reservation-repair').glob('*')}
    visited=[]
    panel=dict(identity=digest('new panels'),materials=reference(tmp_path,base/'registered/registered-data.json'))
    def authenticate(session):
        for row in session.journal['steps'].values():
            if row['status']=='COMPLETE':
                for ref in row.get('outputs',[]):
                    assert reference(tmp_path,ref['path'])==ref
        return SimpleNamespace(bind_reservation=None),panel,dict(test_boundary=True)
    monkeypatch.setattr(m,'authenticate',authenticate)  # Actual source/owner authentication tested separately.
    def roster(root,output,*args):
        doc(output/'private/role-roster.json',test_boundary=True)
        return dict(test_boundary=True)
    monkeypatch.setattr(b,'publish_roster',roster)
    monkeypatch.setattr(f,'estimates',lambda *a,**k:seal(dict(status='READY')))
    monkeypatch.setattr(f,'execution_contract',lambda root,output,**kw:doc(output,test_boundary=True))
    def bind(root,old,new,output,**kw):
        if failure=='seed-rebind':raise ValueError(m.REBIND_ERROR)
        doc(output/'blueprint.json',test_boundary=True)
        return load(output/'blueprint.json')
    monkeypatch.setattr(m,'bind_calibration',bind)  # Full compiler/rebind is tested without this double.
    monkeypatch.setattr(c,'validate_campaign',lambda value:value)
    monkeypatch.setattr(acceptance,'consume_receipt',lambda *a:None)  # Device evidence is an explicit boundary.
    def launch(argv,logs,**kw):
        if '--data' in argv:operation='construction'
        else:
            assert '--engine' in argv and argv[argv.index('--engine')+1]=='podman'
            operation=argv[argv.index('discovery')+1]
        assert operation not in {'qualify','calibrate','rehearse','assess'}
        visited.append((operation,kw['seconds']))
        output=tmp_path/argv[argv.index('--output')+1]
        if operation==failure:raise RuntimeError('explicit external '+operation+' failure')
        doc(Path(logs)/'process.json',returncode=0)
        if operation=='construction':
            doc(output/'construction-summary.json',status='COMPLETE');doc(output/'blueprint.json',test_boundary=True)
        elif operation=='prepare':doc(output/'resource-estimate.json',status='READY')
        elif operation=='freeze':doc(output/'campaign.json',purpose='research',payload=dict(caps=dict(elapsed_us=10000000)))
        elif operation=='acceptance':
            campaign=load(base/'frozen/campaign.json');doc(output/'acceptance.json',campaign=campaign['identity'])
        elif operation=='report':output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(b'EXPLICIT EXTERNAL ARCHIVE BOUNDARY');return
        elif operation=='verify-bundle':doc(output/'verification.json',status='COMPLETE')
        doc(output/'result.json',mechanical_status=research_status if operation=='run' else 'COMPLETE')
    monkeypatch.setattr(supervision,'launch',launch)
    args=SimpleNamespace(phase='prepare-research',engine='podman',memory_gib=64,cpus=8,render_node=None)
    NAMES.update({'freeze-archive.json','continuation.json','continuation-allowance.json'})
    with s.own():
        s.journal['steps'].update({
            'panel-revision/regroup':dict(status='COMPLETE',value=panel,outputs=[],elapsed_seconds=20),
            'prepare-research/prepare':dict(status='COMPLETE',value='original',outputs=[],elapsed_seconds=100),
            'prepare-research/freeze':dict(status='FAILED',reason='original leakage',elapsed_seconds=10),
            'prepare-research/acceptance':dict(status='BLOCKED'),
            'prepare-research/readiness':dict(status='BLOCKED')})
        m.continue_campaign(s,args,script)
    operations=[x[0] for x in visited]
    sequence=['construction','prepare','freeze','acceptance','run','report','verify-bundle']
    expected_first=sequence if failure is None else ['construction'] if failure=='seed-rebind' else sequence[:sequence.index(failure)+1]
    assert operations==expected_first
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    assert load(folder/'continuation.json')['remaining_seconds']==99870
    if failure is None:
        assert s.phase=='run-research' and s.code()==0
        assert load(base/'readiness.json')['research_ready']
        assert s.journal['steps']['prepare-research/prepare']['history'][0]['value']=='original'
        assert s.journal['steps']['prepare-research/freeze']['history'][0]['reason']=='original leakage'
        assert 7200<visited[3][1]<=7800  # Reserved cleanup and startup headroom.
        saved=s.folder/'share/prepare-research.zip'
    else:
        assert s.code()!=0 and s.phase=='prepare-research'
        assert s.journal['steps']['prepare-research/readiness']['status']=='BLOCKED'
        saved=s.folder/'share/prepare-research.zip'
    with zipfile.ZipFile(saved) as z:
        assert 'journal.json' in z.namelist()
        assert not any('/private/' in n for n in z.namelist())
    recovering=failure=='seed-rebind'
    if recovering:
        # Project the v1 journal shape: it had no helper-upgrade stage. All
        # completed construction files and consumed allowance stay unchanged.
        s.journal['steps'].pop('prepare-research/panel-campaign-helper')
        (folder/'continuation-helper.json').unlink();s.save()
        before_allowance=load(folder/'continuation-allowance.json')['spent_seconds']
        monkeypatch.setattr(m,'ORIGINAL_HELPER_SHA',reference(tmp_path,script)['sha256'])
        script=tmp_path/'run-state/continue-v2.py';script.write_text('explicit revised helper fixture')
        failure=None
    visited.clear()
    s=Session(tmp_path,'run-state/round',dict(source=digest('source')),'prepare-research')
    with s.own():m.continue_campaign(s,args,script)
    # Completed stages and archives never repeat. Paused research uses canonical
    # resume and then publishes its new report and verification.
    expected=['resume','report','verify-bundle'] if not failure and research_status=='PAUSED' else []
    if recovering:
        expected=['prepare','freeze','acceptance','run','report','verify-bundle']
        assert s.code()==0
        assert load(folder/'continuation-allowance.json')['spent_seconds']>=before_allowance
        assert s.journal['steps']['prepare-research/panel-campaign-calibration']['history'][0]['reason']==m.REBIND_ERROR
        assert load(folder/'continuation.json')['remaining_seconds']==99870
    assert [x[0] for x in visited]==expected


def test_panel_campaign_authenticates_receipt_and_unchanged_training_bytes(tmp_path):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_panel_campaign import authenticate
    from oeis_learn.generalization.progressive.contracts import reference,atomic_write
    script=tmp_path/'repair.py';script.write_text('def verify_original(root,session):\n session.verified=True\n')
    for n in ('support','support_summary'):atomic_write(tmp_path/(n+'.json'),seal(dict(name=n)))
    materials=seal(dict(status='COMPLETE',diagnostic=False,deficits=[],**{n:reference(tmp_path,n+'.json') for n in ('support','support_summary')}))
    atomic_write(tmp_path/'materials.json',materials)
    row=dict(status='PANELS_READY',deficits=[],training_support_unchanged=True,source=digest('source'),model_calls=0,new_draws=0,
        materials=reference(tmp_path,'materials.json'),original_materials=reference(tmp_path,'materials.json'))
    s=SimpleNamespace(root=tmp_path,binding=dict(source=digest('source')),journal=dict(steps={
        'prepare-research/reservation-repair':dict(status='COMPLETE',value=dict(script=reference(tmp_path,script))),
        'panel-revision/regroup':dict(status='COMPLETE',value=row)}))
    assert authenticate(s)[2]==materials and s.verified
    row['new_draws']=1
    with pytest.raises(ValueError,match='outcome-blind'):authenticate(s)
    row['new_draws']=0
    (tmp_path/'support.json').write_text('tampered')
    with pytest.raises(ValueError):authenticate(s)


@pytest.mark.parametrize('fault',['none','fixed','confirmation','arbitrary','original','budget'])
def test_panel_seed_rebind_is_exact_derivation_not_general_seed_exemption(tmp_path,fault):
    import copy
    from oeis_learn.pipeline_validation.progressive_panel_campaign import invariant_tasks
    from oeis_learn.generalization.progressive.contracts import load
    def trial(study):
        return dict(id='screen',seed=int(digest([study,2])[7:15],16),caps=dict(tokens=4000000),
            trial=dict(study=study,ordinal=2,phase='training'))
    old=[trial(digest('original')),dict(id='fixed',seed=17),
        dict(id='confirmation',seed=19,trial=dict(study=digest('original'),ordinal=2,phase='confirmation'))]
    new=[trial(digest('revised')),*copy.deepcopy(old[1:])]
    if fault=='fixed':new[1]['seed']+=1
    if fault=='confirmation':new[2]['seed']+=1
    if fault=='arbitrary':new[0]['seed']+=1
    if fault=='original':old[0]['seed']+=1
    if fault=='budget':new[0]['caps']['tokens']-=1
    report=tmp_path/'task-rebind.json'
    if fault=='none':
        value=invariant_tasks(old,new,report=report)
        assert value['status']=='PASS' and len(value['derived_seed_changes'])==1
    else:
        with pytest.raises(ValueError,match='scientific tasks'):invariant_tasks(old,new,report=report)
        assert load(report)['status']=='FAIL' and load(report)['mismatches']


def round019_helper():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/'specs/013-progressive-mathematical-discovery/validation/continue-round019.py'
    spec = importlib.util.spec_from_file_location('round019_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round019_original_allowance_and_unknown_intervals():
    helper = round019_helper()
    prior = dict(seconds=261000, spent_seconds=3020.622, active=None)
    assert helper.allocation(prior, dict(elapsed_seconds=7080.426)) == 86400
    assert helper.allocation(dict(prior, seconds=15000), dict(elapsed_seconds=7080)) == pytest.approx(4899.378)
    for change in (dict(active={}), dict(spent_seconds=float('nan')), dict(seconds=4000)):
        with pytest.raises(ValueError):
            helper.allocation(dict(prior, **change), dict(elapsed_seconds=7080))


def test_round019_rejects_unknown_failures():
    helper = round019_helper()
    argv = ['python', '-S', '/root/scripts/foundation_run.py', 'discovery', 'freeze']
    row = dict(status='FAILED')
    result = dict(identity=helper.RESULT, mechanical_status='INTERRUPTED', producer_error=dict(type='KeyboardInterrupt'))
    recorded = dict(argv=argv, seconds=7080, cleanup_seconds=120)
    process = dict(returncode=130, outcome='FAIL', cleanup_failure=None, omitted_bytes=dict(stdout=0, stderr=0),
                   elapsed_seconds=7080.4, failure='TimeoutError: external deadline: '+str(argv))
    helper.timeout_evidence(row, result, process, recorded, argv)
    for changes in (dict(cleanup_failure='unacknowledged'), dict(returncode=-9), dict(failure='other failure'),
                    dict(elapsed_seconds=12), dict(omitted_bytes=dict(stdout=1, stderr=0))):
        with pytest.raises(ValueError):
            helper.timeout_evidence(row, result, dict(process, **changes), recorded, argv)
    with pytest.raises(ValueError):
        helper.timeout_evidence(dict(row, history=[row]), result, process, recorded, argv)


def test_round019_stack_wrapper_keeps_script_imports_and_arguments(tmp_path):
    import subprocess
    import sys
    helper = round019_helper()
    (tmp_path/'sibling.py').write_text('VALUE = 42\n')
    script = tmp_path/'entry.py'
    script.write_text('import sibling,sys,json;print(json.dumps([sibling.VALUE,sys.argv]))\n')
    result = subprocess.run(helper.snapshot_argv([sys.executable, '-S', str(script), '--engine', 'podman']),
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [42, [str(script), '--engine', 'podman']]


@pytest.mark.parametrize('fail_freeze', [False, True])
def test_round019_continuation_preserves_completed_work(tmp_path, monkeypatch, fail_freeze):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, write, read
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow, command
    from oeis_learn.pipeline_validation import supervision, progressive_research_launch
    helper = round019_helper()
    s = Session(tmp_path, helper.SESSION, dict(source=helper.SOURCE), 'prepare-research')
    s.package = lambda: None
    monkeypatch.setattr(helper, 'authenticate', lambda s: None)
    base = s.folder/'prepare-research'
    script = tmp_path/'run-state/helper.py';script.write_text('helper')
    args = SimpleNamespace(engine='podman', memory_gib=64, cpus=8, render_node=None, phase='prepare-research')
    relative = lambda p: str(p.relative_to(tmp_path))
    arguments = dict(blueprint=relative(base/'calibration-reuse/blueprint.json'), qualification=relative(base/'qualification'),
                     prepared=relative(base/'prepared'), output=relative(base/'frozen'))
    argv = command(tmp_path, args, 'freeze', **arguments)
    result = seal(dict(mechanical_status='INTERRUPTED', producer_error=dict(type='KeyboardInterrupt')))
    monkeypatch.setattr(helper, 'RESULT', result['identity'])
    write(base/'frozen/result.json', result)
    write(base/'logs/freeze/command.json', dict(argv=argv, seconds=7080, cleanup_seconds=120))
    write(base/'logs/freeze/process.json', dict(returncode=130, outcome='FAIL', cleanup_failure=None,
        omitted_bytes=dict(stdout=0, stderr=0), elapsed_seconds=7080.4, failure='TimeoutError: external deadline: '+str(argv)))
    prior = seal(dict(seconds=261000, spent_seconds=3020, active=None, binding=seal(dict(scope=s.binding,
        operation='prepare', arguments={k:v for k,v in dict(arguments, output=relative(base/'prepared')).items() if k!='prepared'}))))
    write(base/'prepare-allowance.json', prior)
    s.journal['steps'] = {helper.PREFIX+'freeze':dict(status='FAILED'),
        helper.PREFIX+'execution-preflight':dict(status='COMPLETE', value=dict(setup_seconds=dict(acceptance=7800)))}
    for name in ('registered-data', 'native-census', 'sensitivity', 'materials'):
        s.journal['steps'][helper.PREFIX+name] = dict(status='COMPLETE', value={'original':name})
    calls=[]
    def launch(argv, logs, **kwargs):
        calls.append('freeze')
        assert kwargs['seconds'] == 86280
        write(logs/'process.json', dict(returncode=0))
        if fail_freeze:
            raise RuntimeError('new unknown failure')
        write(base/'frozen/result.json', seal(dict(mechanical_status='COMPLETE')))
        write(base/'frozen/campaign.json', seal(dict(purpose='research')))
    monkeypatch.setattr(supervision, 'launch', launch)
    def acceptance(w, name, operation, **kwargs):
        assert operation == 'acceptance' and kwargs['seconds'] == 7800
        def action():
            calls.append('acceptance')
            return {}, []
        return w.s.step(name, action)
    monkeypatch.setattr(Workflow, 'run_command', acceptance)
    monkeypatch.setattr(Workflow, 'run_after_prepare', lambda w: calls.append('run'))
    def ready(w, *values):
        assert [v['original'] for v in values] == ['registered-data','native-census','sensitivity','materials']
        return {}, []
    monkeypatch.setattr(progressive_research_launch, 'readiness', ready)
    helper.continue_campaign(s, args, script)
    assert calls == (['freeze'] if fail_freeze else ['freeze', 'acceptance', 'run'])
    assert read(base/'prepare-allowance.json') == prior
    assert read(base/'freeze-continuation/original-failed-freeze/result.json') == result
    helper.continue_campaign(s, args, script)
    assert calls == (['freeze'] if fail_freeze else ['freeze', 'acceptance', 'run', 'run'])
    assert s.journal['steps'][helper.PREFIX+'freeze']['history'][0]['status'] == 'FAILED'


def round019_storage_helper():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/'specs/013-progressive-mathematical-discovery/validation/continue-round019-storage.py'
    spec = importlib.util.spec_from_file_location('round019_storage', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('producer_error', [None, {}])
def test_round019_trace_collection_accepts_normal_no_error(tmp_path, producer_error):
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    write(tmp_path/'result.json', seal(dict(producer_error=producer_error, mechanical_status='BLOCKED')))
    round019_helper().copy_trace(tmp_path, tmp_path/'copied.log')
    round019_storage_helper().collect_trace(tmp_path, tmp_path)
    assert not (tmp_path/'copied.log').exists()
    assert not (tmp_path/'collector-error.log').exists()


def test_round019_trace_collection_cannot_mask_original_failure(tmp_path):
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    write(tmp_path/'result.json', seal(dict(producer_error='malformed')))
    helper = round019_storage_helper()
    with pytest.raises(RuntimeError, match='original failure'):
        try:
            raise RuntimeError('original failure')
        finally:
            helper.collect_trace(tmp_path, tmp_path)
    assert 'producer_error must be null or an object' in (tmp_path/'collector-error.log').read_text()


def test_round019_storage_lower_bound_stops_before_expensive_reads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.storage import JOURNAL_BYTES
    helper = round019_storage_helper()
    q = dict(task_plans=[dict(caps=dict(disk_bytes=375541145822))], shard_bytes=26620955,
             package_bytes=536870912, required_inodes=1000000)
    prepared = seal(dict(payload=q))
    monkeypatch.setattr(helper.shutil, 'disk_usage', lambda _: SimpleNamespace(free=291429883904))
    monkeypatch.setattr(helper.os, 'statvfs', lambda _: SimpleNamespace(f_favail=119541676))
    result = helper.capacity(tmp_path, prepared)
    assert result['status'] == 'BLOCKED' and result['scope'] == 'lower_bound'
    assert result['required_bytes'] == 375541145822+JOURNAL_BYTES+26620955+536870912
    assert result['additional_bytes'] > 86*10**9
    # No checkpoint, runtime or private input paths were supplied or read.


def test_round019_storage_allowance_retains_all_previous_costs():
    helper = round019_storage_helper()
    prior = dict(seconds=261000, spent_seconds=3020.622, active=None)
    first = dict(previous_freeze_seconds=7080.426)
    retry = dict(spent_seconds=39212.393, active=None)
    assert helper.remaining_seconds(prior, first, retry) == 86400
    assert helper.remaining_seconds(dict(prior, seconds=60000), first, retry) == pytest.approx(10686.559)
    for invalid in (dict(active={}), dict(spent_seconds=float('nan')), dict(spent_seconds=999999)):
        with pytest.raises(ValueError):
            helper.remaining_seconds(prior, first, dict(retry, **invalid))


@pytest.mark.parametrize('outcome', ['COMPLETE', 'BLOCKED'])
def test_round019_storage_resume_reuses_history_and_reports_true_outcome(tmp_path, monkeypatch, outcome):
    import copy
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, write, read
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation import supervision, progressive_research_launch
    helper = round019_storage_helper()
    s = Session(tmp_path, helper.SESSION, dict(source='qualified'), 'prepare-research')
    s.package = lambda: None
    base = s.folder/'prepare-research';old = base/'freeze-continuation'
    script = tmp_path/'run-state/new-helper.py';script.write_text('new-helper')
    old_script = tmp_path/'run-state/original-helper.py';old_script.write_text('original-helper')
    prior = seal(dict(seconds=261000, spent_seconds=3020, active=None))
    write(base/'prepare-allowance.json', prior)
    first = seal(dict(script=reference(tmp_path, old_script), prior_allowance=reference(tmp_path, base/'prepare-allowance.json'),
                      remaining_seconds=86400, previous_freeze_seconds=7080))
    bound = seal(dict(continuation=first['identity']))
    retry = seal(dict(seconds=86400, spent_seconds=39212.4, active=None, binding=bound,
        attempts=[dict(outcome='AttributeError', error=helper.MASKED_ERROR)]))
    write(old/'continuation-allowance.json', retry)
    result = seal(dict(mechanical_status='BLOCKED', producer_error=None, reason='FREEZE_DEFICIT', missing_evidence=['storage_bytes']))
    monkeypatch.setattr(helper, 'BLOCKED_RESULT', result['identity'])
    write(base/'frozen/result.json', result)
    write(old/'logs/0000/stdout.log', result)
    write(old/'logs/0000/process.json', dict(returncode=2, outcome='FAIL', cleanup_failure=None,
        omitted_bytes=dict(stdout=0, stderr=0), elapsed_seconds=39212.39, failure='RuntimeError: exit 2: original/stderr.log'))
    write(base/'prepared/prepared.json', seal(dict(payload={})))
    s.journal['steps'] = {helper.PREFIX+'freeze':dict(status='FAILED', reason=helper.MASKED_ERROR, binding=bound,
                                                    history=[dict(status='FAILED', reason='old timeout')]),
        helper.PREFIX+'execution-preflight':dict(status='COMPLETE', value=dict(setup_seconds=dict(acceptance=7800)))}
    for n in ('registered-data', 'native-census', 'sensitivity', 'materials'):
        s.journal['steps'][helper.PREFIX+n] = dict(status='COMPLETE', value={})
    monkeypatch.setattr(helper, 'original_helper', lambda _: (SimpleNamespace(snapshot_argv=lambda a:a), first))
    monkeypatch.setattr(helper, 'qualified_images', lambda *a: dict(all_present=True, images=[]))
    monkeypatch.setattr(helper, 'capacity', lambda *a: dict(status='BLOCKED'))
    args = SimpleNamespace(engine='podman', memory_gib=64, cpus=8, render_node=None, phase='prepare-research')
    before = copy.deepcopy(s.journal)
    monkeypatch.setattr(helper, 'qualified_images', lambda *a: dict(all_present=False, images=[]))
    monkeypatch.setattr(helper, 'capacity', lambda *a: (_ for _ in ()).throw(AssertionError('missing image must stop before validation')))
    assert helper.execute(s, args, script, proceed=True) == 2
    assert s.journal == before
    monkeypatch.setattr(helper, 'qualified_images', lambda *a: dict(all_present=True, images=[]))
    monkeypatch.setattr(helper, 'capacity', lambda *a: dict(status='BLOCKED'))
    assert helper.execute(s, args, script, proceed=True) == 2
    assert s.journal == before and (base/'frozen/result.json').exists()
    monkeypatch.setattr(helper, 'capacity', lambda *a: dict(status='READY'))
    assert helper.execute(s, args, script) == 0
    assert s.journal == before  # Check-only cannot archive or execute.
    calls=[]
    def launch(argv, logs, **kwargs):
        assert kwargs['seconds'] == 86280 and kwargs['allowed_returncodes'] == (0,2,3)
        calls.append('freeze')
        write(logs/'process.json', dict(returncode=0 if outcome=='COMPLETE' else 2))
        value = seal(dict(result, mechanical_status=outcome))
        write(base/'frozen/result.json', value)
        if outcome == 'COMPLETE':write(base/'frozen/campaign.json', seal(dict(purpose='research')))
    monkeypatch.setattr(supervision, 'launch', launch)
    def acceptance(w, name, operation, **kwargs):
        return w.s.step(name, lambda: (calls.append('acceptance'), []))
    monkeypatch.setattr(Workflow, 'run_command', acceptance)
    monkeypatch.setattr(Workflow, 'run_after_prepare', lambda w: calls.append('run'))
    monkeypatch.setattr(progressive_research_launch, 'readiness', lambda *a: ({}, []))
    helper.execute(s, args, script, proceed=True)
    assert calls == (['freeze','acceptance','run'] if outcome=='COMPLETE' else ['freeze'])
    assert read(base/'prepare-allowance.json') == prior
    assert read(old/'continuation-allowance.json') == retry
    assert read(base/'storage-continuation/original-blocked-freeze/result.json') == result
    row = s.journal['steps'][helper.PREFIX+'freeze']
    assert len(row['history']) == 2 and row['history'][-1]['reason'] == helper.MASKED_ERROR
    if outcome == 'BLOCKED':
        assert row['reason'] == 'FREEZE_DEFICIT: storage_bytes'
        helper.execute(s, args, script, proceed=True)
        assert calls == ['freeze']
    else:
        monkeypatch.setattr(helper, 'capacity', lambda *a: (_ for _ in ()).throw(AssertionError('completed freeze must be reused')))
        helper.execute(s, args, script, proceed=True)
        assert calls == ['freeze','acceptance','run','run']


def test_round019_full_storage_arithmetic_includes_projection_and_inodes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive import storage, campaign
    helper = round019_storage_helper()
    ref = dict(path='reference.json', bytes=1, sha256='sha256:'+'0'*64)
    q = dict(task_plans=[dict(id='train', kind='training', caps=dict(disk_bytes=100),
                             initializer=dict(mode='weights_only', artifact=ref, origin=ref))],
             references=[], runtime=ref, checkpoint=ref, target_policy=ref,
             checkpoint_root='.', milestone_pins=1, shard_bytes=10, package_bytes=30, required_inodes=1)
    prepared = seal(dict(payload=q, track='synthetic_strict', purpose='research'))
    monkeypatch.setattr(helper.shutil, 'disk_usage', lambda _: SimpleNamespace(free=50 << 30))
    monkeypatch.setattr(helper.os, 'statvfs', lambda _: SimpleNamespace(f_favail=100000))
    monkeypatch.setattr(storage, 'checkpoint_size', lambda *a, **k: (20,3))
    monkeypatch.setattr(storage, 'projection_size', lambda *a: (40,4))
    monkeypatch.setattr(campaign, 'initializer_origin', lambda *a, **k: (None,[dict(path='origin.json', bytes=7)]))
    result = helper.capacity(tmp_path, prepared)
    assert result['status'] == 'READY' and result['scope'] == 'full'
    assert result['required_bytes'] == storage.JOURNAL_BYTES+100+10+30+40+7+20
    assert result['required_inodes'] == (3*3+256)+4+2+16384
    monkeypatch.setattr(helper.os, 'statvfs', lambda _: SimpleNamespace(f_favail=5))
    result = helper.capacity(tmp_path, prepared)
    assert result['status'] == 'BLOCKED' and result['additional_inodes'] > 0


def test_round019_storage_inspection_is_read_only_and_continues_on_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import supervision, discovery_cleanup
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    helper = round019_storage_helper()
    session = SimpleNamespace(root=tmp_path, folder=tmp_path/'run-state/session')
    write(session.folder/'prepare-research/qualification/qualification.json', seal(dict(images=['cpu-id','gpu-id'])))
    output = session.folder/'diagnostics';output.mkdir(parents=True)
    calls=[]
    def launch(argv, logs, **kwargs):
        calls.append(argv)
        assert kwargs['seconds'] == 120
        if len(calls) == 1:raise RuntimeError('engine accounting temporarily unavailable')
    monkeypatch.setattr(supervision, 'launch', launch)
    monkeypatch.setattr(discovery_cleanup, 'inventory', lambda _: dict(groups=[], errors=[]))
    helper.inspect_storage(session, SimpleNamespace(engine='podman'), output)
    assert [a[1] for a in calls] == ['system','info','history','history']
    assert list(output.rglob('disk-inventory.json'))
    assert list(output.rglob('engine-disk-collector-error.log'))
