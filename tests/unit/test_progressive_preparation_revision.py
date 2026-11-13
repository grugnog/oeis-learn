"""Prospective recovery: preserve originals and test real sampling/publication seams."""
from collections import Counter
import pytest
from oeis_learn.generalization.progressive.contracts import seal, digest


def test_support_conditioned_draws_are_replayable_and_have_declared_strata():
    from oeis_learn.generalization.progressive.research_sampling import Sampler
    from oeis_learn.pipeline_validation.progressive_research_census import features, classify
    from oeis_learn.generalization.progressive.generator import structure_id
    sources=['(program (return (add (mul $index (int 2)) (int 3))))',
             '(program (return (sub (mul $index $index) (int 1))))']
    rows=[dict(source=s,structure=structure_id(s),draw_identity=digest(s)) for s in sources]
    support=seal(dict(identity=None, version='test',shards=[rows],layout={'probes':{}}))
    sample=Sampler(support,seed=7)
    f=dict(structures={r['structure'] for r in rows},nodes=set(),operators=set())
    for s in sources:
        v=features(s);f['nodes'].update(v['nodes']);f['operators'].update(v['operators'])
    seen=Counter()
    for i in range(100):
        d=sample.draw(i)
        assert d==sample.draw(i)
        seen[classify(d,f)]+=1
    assert seen['parameter']>=50 and seen['composition']>20
    assert sample.identity!=Sampler(support,seed=8).identity


def test_revision_screening_accepts_singletons_without_inventing_relations():
    from oeis_learn.generalization.progressive.prepared_panels import screening
    from oeis_learn.generalization.progressive.tuning import validate_screening_panel
    rows=[dict(id=f'{s}-{i}',group=f'{s}-g{i}',stratum=s,role='development')
          for s in ('parameter','composition','unseen_structure','oeis') for i in range(128)]
    chosen,summary=screening(rows,seed=7,grouping='balanced-at-most-two/v2')
    assert summary['status']=='COMPLETE' and len(chosen)==128
    assert all(r in rows for r in chosen)
    validate_screening_panel(chosen,grouping='balanced-at-most-two/v2')
    with pytest.raises(ValueError):validate_screening_panel(chosen)
    broken=[dict(r,group='one') if r['stratum']=='oeis' else r for r in chosen]
    with pytest.raises(ValueError):validate_screening_panel(broken,grouping='balanced-at-most-two/v2')


def test_revision_parameter_structure_sharing_keeps_exact_relations_together():
    from oeis_learn.pipeline_validation.progressive_research_census import reserve_panels
    rows=[dict(id=str(i),structure='same-structure',stratum='parameter',
        values=[str(i*100000+n*n) for n in range(100)]) for i in range(100)]
    rows.append(dict(rows[0],id='shift',values=[str(n*n) for n in range(1,101)]))
    panels,counts,_=reserve_panels(rows,seed=7,required=128,minimum_groups=16,
        panels=['development','assessment'],retention_panels=['retention-development'],
        parameter_sharing=True)
    owners={r['id']:p for p,rs in panels.items() for r in rs}
    # Selection caps can omit the duplicate; assignment itself must not split it.
    if '0' in owners and 'shift' in owners:assert owners['0']==owners['shift']
    assert all(counts[p+'/parameter']['groups']==1 for p in panels)
    assert all(len({r['group'] for r in rs})==1 for rs in panels.values())


def test_numeric_ineligibility_is_distinct_from_bad_original_data():
    from oeis_learn.generalization.progressive.research_sampling import source_eligibility
    row=dict(id='x',indices=list(map(str,range(100))),values=['1']*100)
    assert source_eligibility(row) is None
    row['values'][99]=str(2**4096)
    assert source_eligibility(row)=='outside_bigint_4096_profile'
    row['values'][99]='01'
    with pytest.raises(ValueError,match='canonical'):source_eligibility(row)


def revised_fixture(root, *, candidates=None, client=None):
    from tests.unit.test_progressive_research_census import inputs, ReferenceDouble
    from oeis_learn.pipeline_validation.progressive_research_census import run
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    from oeis_learn.generalization.progressive.contracts import reference
    old=inputs(root)
    if candidates is not None:
        write(root/'candidates.json',candidates)
        old=seal(dict(old,candidates=reference(root,'candidates.json')))
    write(root/'old/request.json',old)
    original=run(root,root/'old',old,client=client or ReferenceDouble())
    reuse=seal(dict(version='progressive-census-data-reuse/v1',
        census=reference(root,'old/census.json'),request=reference(root,'old/request.json'),
        artifacts=[reference(root,root/'old'/r['path']) for r in original['artifacts']],
        native=original['native'],original_source=digest('explicit native producer test boundary'),
        prior_elapsed_seconds=original['elapsed_seconds'],original_draws=40,
        additional_draws=32768,seconds_remaining=old['seconds']-original['elapsed_seconds'],
        hardware_authority_transferred=False,assessment_exposed=False))
    revised=seal(dict(old,version='progressive-research-census-request/v3',reuse=reuse,evaluation_draws=48))
    write(root/'revised/request.json',revised)
    return original,revised


def test_revised_census_reuses_all_draws_and_keeps_original_costs(tmp_path):
    from tests.unit.test_progressive_research_census import ReferenceDouble
    from oeis_learn.pipeline_validation.progressive_research_census import run
    from oeis_learn.pipeline_validation.progressive_research_materials import build
    from oeis_learn.generalization.progressive.prepared_support import policy
    original,req=revised_fixture(tmp_path)
    before={p:p.read_bytes() for p in (tmp_path/'old/private').glob('*.sqlite')}
    class Counting(ReferenceDouble):
        calls=0
        def execute(self,*a,**kw):self.calls+=1;return super().execute(*a,**kw)
    client=Counting();result=run(tmp_path,tmp_path/'revised',req,client=client)
    assert client.calls==24
    assert result['counts']['TRAINING']==original['counts']['TRAINING']
    assert sum(result['counts'].get(k,0) for k in ('TRAINING','EVALUATION','REJECTED','FAILED'))==64
    assert result['elapsed_seconds']>=original['elapsed_seconds']
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert run(tmp_path,tmp_path/'revised',req,client=client)==result and client.calls==24
    materials=build(tmp_path,'revised/census.json','revised/request.json','materials',
        policy(seed=req['seed'],shard_rows=req['shard_rows'],batch_cadence=True))
    assert materials['status']=='INCOMPLETE'  # tiny fixtures cannot establish research supply
    assert not materials['original_generation_repeated']


def test_revision_rejects_changed_native_and_missing_original_bytes(tmp_path):
    from tests.unit.test_progressive_research_census import ReferenceDouble
    from oeis_learn.pipeline_validation.progressive_research_census import run
    _,req=revised_fixture(tmp_path)
    class Foreign(ReferenceDouble):
        def runtime_identity(self,*args):return digest('foreign native')
    with pytest.raises(ValueError,match='native data verifier'):
        run(tmp_path,tmp_path/'bad-native',req,client=Foreign())
    (tmp_path/'old/private/census.sqlite').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):
        run(tmp_path,tmp_path/'revised',req,client=ReferenceDouble())


def test_revision_continues_after_supported_numeric_exclusion(tmp_path):
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    from oeis_learn.pipeline_validation.progressive_research_census import run
    from oeis_learn.generalization.progressive.contracts import reference
    from tests.unit.test_progressive_research_census import ReferenceDouble
    rows=[dict(id='large',indices=list(map(str,range(100))),values=[str(2**4096)]*100),
          dict(id='valid',indices=list(map(str,range(100))),values=[str(9999999+n**3) for n in range(100)])]
    _,req=revised_fixture(tmp_path,candidates=rows)
    result=run(tmp_path,tmp_path/'numeric-revision',req,client=ReferenceDouble())
    assert result['counts']['oeis_ineligible']==1 and result['counts']['oeis_available']==1
    assert not result['counts'].get('oeis_failed')
    assert result['source_dispositions'][0]['id']=='large'


def adoption_fixture(root,monkeypatch):
    """Explicit original-producer boundary; adoption does not qualify this data."""
    from oeis_learn.pipeline_validation import progressive_research_revision as r
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_research_bridge import request
    from tests.unit.test_progressive_research_census import inputs
    small=inputs(root)
    data=seal(dict(version='progressive-registered-data/v1',**{k:small[k] for k in ('candidates','exposure','recipe')}))
    req=request(data);folder=root/'run-state/old';base=folder/'prepare-research/census'
    write(base/'request.json',req)
    artifacts=[]
    for name in ('census.sqlite','training-exposure.sqlite','training-reservoir.json','panel-reservations.json'):
        write(base/'private'/name,dict(test_boundary_only=True));artifacts.append(reference(base,'private/'+name))
    census=seal(dict(version='progressive-research-census/v1',request=req['identity'],diagnostic=False,
        counts=dict(TRAINING=8192,EVALUATION=1344,REJECTED=31424),native=digest('native fixture'),
        elapsed_seconds=1000,artifacts=artifacts,training_started=False,assessment_exposed=False))
    write(base/'census.json',census)
    source=next(iter(r.COMPATIBLE_CENSUS_SOURCES))
    write(folder/'session.json',seal(dict(binding={'source':source})))
    steps={'prepare-research/census-plan':dict(status='COMPLETE',outputs=[reference(root,base/'request.json')]),
        'prepare-research/native-census':dict(status='COMPLETE',elapsed_seconds=1130,
            outputs=[reference(root,base/'census.json'),*[reference(root,base/a['path']) for a in artifacts]])}
    write(folder/'journal.json',seal(dict(steps=steps)))
    return data,folder,steps


def test_adoption_checks_originals_and_has_one_durable_revision_owner(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_research_revision import adopt
    from oeis_learn.pipeline_validation.progressive_research_bridge import request
    from oeis_learn.pipeline_validation.progressive_research_census import validate_request
    data,folder,steps=adoption_fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    reuse=adopt(tmp_path,'run-state/old',tmp_path/'run-state/new',data)
    assert reuse['prior_elapsed_seconds']==1130 and reuse['seconds_remaining']==20470
    assert reuse==adopt(tmp_path,'run-state/old',tmp_path/'run-state/new',data)
    assert not reuse['hardware_authority_transferred']
    assert all(p.read_bytes()==raw for p,raw in before.items())
    revised=request(seal(dict(data,census_reuse=reuse)))
    validate_request(revised)
    with pytest.raises(ValueError,match='limits'):validate_request(seal(dict(revised,evaluation_draws=32768)))
    with pytest.raises(ValueError,match='sibling budget reset'):
        adopt(tmp_path,'run-state/old',tmp_path/'run-state/sibling',data)
    (folder/'prepare-research/census/private/training-reservoir.json').write_text('changed')
    with pytest.raises(ValueError,match='identity'):
        adopt(tmp_path,'run-state/old',tmp_path/'run-state/new',data)


def test_adoption_refuses_revised_inputs_and_any_scientific_execution(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_research_revision import adopt
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    data,folder,steps=adoption_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='inputs changed'):
        adopt(tmp_path,'run-state/old',tmp_path/'run-state/new',seal(dict(data,candidates={})))
    steps['run-research/invocation-0000']={'status':'FAILED'}
    write(folder/'journal.json',seal(dict(steps=steps)))
    with pytest.raises(ValueError,match='unscored'):
        adopt(tmp_path,'run-state/old',tmp_path/'run-state/new',data)


def test_new_screening_rule_is_bound_through_roster_and_study_policy(tmp_path):
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
    from oeis_learn.generalization.progressive.research_roster import construct
    from oeis_learn.generalization.progressive.prepared_panels import screening
    from oeis_learn.pipeline_validation.progressive_study import policy
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    from oeis_learn.generalization.progressive.contracts import reference
    _,panels,_=fixture()
    panels['development']=[dict(r,group=r['id']) for r in panels['development']]
    selected,_=screening([dict(r,role='development') for r in panels['development']],seed=17,
        grouping='balanced-at-most-two/v2')
    design=scientific_round(seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19])),census_reuse={'TEST':True})))
    roster=construct(design,panels,selected)
    assert len(roster['tasks'])==376
    study=seal(dict(version='progressive-study-policy/v2',token_cap=4100000,target_tokens=4000000,
        elapsed_cap_us=7200000000,screening_group_policy='balanced-at-most-two/v2',
        screening=dict(startup_us=1000000,evaluation_us=[1000000]*3,training_us=1000000,finalization_us=1800000000),
        panel=selected,evaluations={},confirmation={}))
    write(tmp_path/'study.json',study)
    campaign={'purpose':'research','payload':{'references':[{'kind':'policy','artifact':reference(tmp_path,'study.json')}]}}
    assert policy(tmp_path,campaign,study['identity'])==study
    bad=seal(dict(study,version='progressive-study-policy/v1'))
    write(tmp_path/'study.json',bad);campaign['payload']['references'][0]['artifact']=reference(tmp_path,'study.json')
    with pytest.raises(ValueError,match='closed'):policy(tmp_path,campaign,bad['identity'])


def test_actual_native_revision_and_material_publication(fitting_client,tmp_path):
    """Runs in mandatory Docker CI; unavailable native service is a local skip."""
    from oeis_learn.pipeline_validation.progressive_research_census import run
    from oeis_learn.pipeline_validation.progressive_research_materials import build
    from oeis_learn.generalization.progressive.prepared_support import policy
    original,req=revised_fixture(tmp_path,client=fitting_client)
    result=run(tmp_path,tmp_path/'revised',req,client=fitting_client)
    assert result['native']==original['native']==fitting_client.runtime_identity('bigint-body/v1')
    assert not result['counts'].get('FAILED') and not result['counts'].get('unattempted')
    assert sum(result['counts'].get(k,0) for k in ('TRAINING','EVALUATION','REJECTED'))==64
    materials=build(tmp_path,'revised/census.json','revised/request.json','materials',
        policy(seed=req['seed'],shard_rows=req['shard_rows'],batch_cadence=True))
    assert materials['diagnostic'] and not materials['research_ready']


def test_parameter_and_retention_allow_structure_only_not_exact_values(tmp_path):
    from tests.unit.test_progressive_cohorts import fixture
    from oeis_learn.generalization.progressive.cohorts import validate_cohort
    a,e=fixture(tmp_path,panel='a',counts=(1,0,0,0))
    b,_=fixture(tmp_path,panel='b',counts=(1,0,0,0))
    ar=a['payload']['rows'][0];br=b['payload']['rows'][0]
    br.update(stratum='retention',structure=ar['structure'],behavior=digest('different exact values'),groups=['independent'])
    b=seal(dict(b,payload=dict(b['payload'],strata=['retention'],group_policy='parameter_structure_sharing')))
    validate_cohort(b,exposure=e,other_cohorts=[a])
    b['payload']['rows'][0]['behavior']=ar['behavior'];b=seal(b)
    with pytest.raises(ValueError,match='leakage'):validate_cohort(b,exposure=e,other_cohorts=[a])


def test_original_sensitivity_is_reused_without_running_cases(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_research_revision import reuse_sensitivity
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    from oeis_learn.pipeline_validation.progressive_research_sensitivity import run
    from oeis_learn.generalization.progressive.contracts import reference
    from tests.unit.test_progressive_sensitivity import scenario
    data,folder,steps=adoption_fixture(tmp_path,monkeypatch)
    design={'seed':13,'repeats':1,'scenarios':[scenario(effect_ppm=0)]}
    old=folder/'prepare-research/sensitivity';write(old/'design.json',design)
    original=run(design,old)
    steps['prepare-research/sensitivity']=dict(status='COMPLETE',elapsed_seconds=12,
        outputs=[reference(tmp_path,old/'sensitivity.json')])
    write(folder/'journal.json',seal(dict(steps=steps)))
    reuse=dict(original_journal=reference(tmp_path,folder/'journal.json'),
        original_session=reference(tmp_path,folder/'session.json'),original_source=digest('original'))
    result,files=reuse_sensitivity(tmp_path,reuse,design,tmp_path/'new')
    assert result==original and len(files)==4
    with pytest.raises(ValueError,match='assumptions'):reuse_sensitivity(tmp_path,reuse,dict(design,seed=14),tmp_path/'bad')
    (old/'case-outcomes.json').write_text('{}')
    with pytest.raises(ValueError):reuse_sensitivity(tmp_path,reuse,design,tmp_path/'bad')
