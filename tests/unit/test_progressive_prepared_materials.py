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
            str((base/'materials').relative_to(tmp_path)),support.policy(seed=req['seed']))
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
            str((base/'materials').relative_to(tmp_path)),support.policy(seed=req['seed']))
