"""Real data projection mechanics; reference execution never qualifies hardware."""
import json

import pytest

from oeis_learn.generalization.progressive.contracts import digest, record, seal
from oeis_learn.generalization.progressive import prepared_support as support
from oeis_learn.generalization.progressive import prepared_panels as panels
from oeis_learn.generalization.progressive.cohorts import publish_many, validate_frozen_panels
from oeis_learn.pipeline_validation.progressive_research_census import reserve_panels


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


def test_round_retains_full_optimization_domain_and_separate_panels(tmp_path):
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
    design=scientific_round(data)
    assert design['selected_stages']==['capability','diversity','tuning']
    assert design['tuning']['learning_trials']==12 and design['tuning']['sizing_trials']==4
    assert design['tuning']['screen_seconds']==7200
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
