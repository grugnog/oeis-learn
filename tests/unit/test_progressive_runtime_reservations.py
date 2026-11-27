"""Measured operational revision preserves science and every historical ceiling."""
import pytest
from oeis_learn.generalization.progressive.contracts import seal
from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
from oeis_learn.generalization.progressive.research_roster import construct
from oeis_learn.generalization.progressive.research_budget import reservation
from tests.unit.test_progressive_research_roster import fixture


def test_revision_preserves_every_scientific_role_and_fits_nested_reservations():
    old,panels,screen=fixture()
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    design=scientific_round(data,expanded=True)
    before=construct(old,panels,screen);after=construct(design,panels,screen)
    assert before['version'].endswith('/v1') and after['version'].endswith('/v2')
    assert [{k:v for k,v in r.items() if k!='seconds'} for r in before['tasks']]==[
        {k:v for k,v in r.items() if k!='seconds'} for r in after['tasks']]
    b=reservation(after)
    assert b['status']=='READY' and b['setup_seconds']['prepare']==72*3600+1800
    for ordinal in range(16):
        children=[r for r in after['tasks'] if r['trial'] and r['trial']['ordinal']==ordinal
                  and r['trial']['phase'] not in {'selection','confirmation'}]
        assert sum(r['seconds'] for r in children)+1800<=96*3600
    assert all(r['seconds']<=86400 for r in after['tasks'] if r['operation']=='train')
    assert max(r['seconds'] for r in after['tasks'] if r['operation']=='generate')<=72*3600
    # Recorded service proposals: 44.96h longest role, 64.07h original full trial.
    assert max(r['seconds'] for r in after['tasks'])>161849
    assert b['ancillary_seconds']['evaluation_analysis']>330.485*3600
    assert reservation(before)['selected_maximum_seconds']==312*3600


def test_expanded_trial_requires_explicit_version():
    from oeis_learn.generalization.progressive.tuning import preflight
    kw=dict(startup_us=20_000_000,evaluation_us=[10**10]*3,training_us=10**10,
            finalization_us=1800_000_000,cap_us=96*3600_000_000)
    with pytest.raises(ValueError):preflight(**kw)
    assert preflight(**kw,operational_revision=True)<kw['cap_us']


def test_long_calibration_requires_new_plan_and_request_versions(tmp_path):
    from oeis_learn.pipeline_validation.progressive_handoff import named_plan
    from oeis_learn.pipeline_validation.progressive_calibration import validate_request
    from oeis_learn.generalization.progressive.contracts import atomic_write,reference,digest
    from oeis_learn.generalization.progressive.budgets import KEYS
    caps=dict.fromkeys(KEYS,0);caps.update(elapsed_us=72*3600*1000000,disk_bytes=1<<30)
    plan=seal(dict(version='progressive-calibration-plan/v2',seconds=72*3600,caps=caps,checkpoint_bytes=1<<20,maximum_inodes=100))
    atomic_write(tmp_path/'plan.json',plan)
    blueprint=dict(payload=dict(references=[dict(name='iteration_calibration_plan',artifact=reference(tmp_path,'plan.json'))]))
    assert named_plan(tmp_path,blueprint,'iteration_calibration_plan','progressive-calibration-plan/v1',
        {'seconds','caps','checkpoint_bytes','maximum_inodes'})==plan
    request=seal(dict(version='progressive-calibration-request/v2',seconds=72*3600,caps=caps,checkpoint_bytes=1<<20,maximum_inodes=100,
        source=digest('s'),image=digest('i'),engine='podman',hardware={},execution_policy=digest('e'),blueprint=digest('b'),prepared=digest('p'),
        cases=[dict(compatibility=digest('c'),artifact=dict(path='input.json',bytes=1,sha256=digest('x')))]))
    validate_request(request)
    with pytest.raises(ValueError):validate_request(seal(dict(request,version='progressive-calibration-request/v1')))
    with pytest.raises(ValueError):validate_request(seal(dict(request,seconds=72*3600+1)))


def test_evaluation_deadline_uses_only_validated_installed_authority(tmp_path):
    import time
    from types import SimpleNamespace
    from tests.unit.test_progressive_budgets import role_authority
    from oeis_learn.generalization.progressive.budgets import RoleWorkBudget
    from oeis_learn.pipeline_validation.progressive_evaluation import _maximum_seconds
    from oeis_learn.pipeline_validation.progressive_role import _evaluation_seconds
    values=[seal(dict(kind='candidate',data={}))]
    a,_,_=role_authority(tmp_path,'admit-training',values)
    a['campaign']['payload']['references'].append(dict(name='research-reservation-v2',role='custodian',kind='policy',
        artifact=dict(path='reservation.json',bytes=1,sha256='sha256:'+'a'*64)))
    a['campaign']=seal(a['campaign']);a['task']['payload']['campaign']=a['campaign']['identity'];a['task']=seal(a['task']);a=seal(a)
    budget=RoleWorkBudget(a,'admit-training',values,tmp_path/'installed')
    assert _maximum_seconds(budget)==72*3600
    assert _maximum_seconds(SimpleNamespace(evaluation_seconds=10**9))==21600
    budget.deadline=time.monotonic()+50*3600
    assert 49*3600<_evaluation_seconds(budget)<=50*3600


def test_recovery_setup_headroom_is_counted_with_trajectory_and_preserves_original(tmp_path):
    from oeis_learn.pipeline_validation.progressive_preparation_preflight import reservation as extra, verified
    from oeis_learn.pipeline_validation.progressive_trajectory import reservation as trajectory
    from oeis_learn.pipeline_validation.research_limits import case_allowance,ACCEPTANCE_SECONDS
    from oeis_learn.generalization.progressive.contracts import atomic_write,reference,digest
    _,panels,screen=fixture()
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    outer=reservation(construct(scientific_round(data,expanded=True),panels,screen))
    atomic_write(tmp_path/'roster/outer.json',outer)
    old=(tmp_path/'roster/outer.json').read_bytes()
    value=extra(tmp_path,reference(tmp_path,'roster/outer.json'))
    assert value['additional_seconds']==11800
    assert value['setup_seconds']['base_qualification']>=15600+4*120
    assert case_allowance(value['setup_seconds']['qualify'],maximum=4200)>3600
    assert case_allowance(value['setup_seconds']['acceptance'],maximum=7800)>ACCEPTANCE_SECONDS
    assert value['reserved_qualification_seconds']<=outer['limits_seconds']['qualification_acceptance']
    # Compiler republishes identical original budget bytes at its own path.
    atomic_write(tmp_path/'construction/outer.json',outer)
    atomic_write(tmp_path/'headroom.json',value)
    refs=[dict(name='research-reservation-v2',artifact=reference(tmp_path,'construction/outer.json')),
          dict(name='preparation_headroom',artifact=reference(tmp_path,'headroom.json'))]
    assert verified(tmp_path,refs)==value
    req=dict(identity=digest('request'),cases=[{}]*13,target_tokens=128000)
    combined=trajectory(tmp_path,refs,req,research=True)
    assert combined['reserved_total_seconds']==outer['total_seconds']+11800+36300
    assert combined['reserved_qualification_seconds']==value['reserved_qualification_seconds']
    assert (tmp_path/'roster/outer.json').read_bytes()==old
    altered=dict(outer,limits_seconds=dict(outer['limits_seconds'],qualification_acceptance=1))
    atomic_write(tmp_path/'tight.json',seal(altered))
    with pytest.raises(ValueError,match='outer limits'):extra(tmp_path,reference(tmp_path,'tight.json'))
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    write(tmp_path/'headroom.json',seal(dict(value,additional_seconds=0)))
    refs[1]['artifact']=reference(tmp_path,'headroom.json')
    with pytest.raises(ValueError,match='differs'):verified(tmp_path,refs)


def test_incompatible_calibration_is_reported_before_any_qualification(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_research_launch as launch
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,write
    from oeis_learn.generalization.progressive.contracts import digest
    session=Session(tmp_path,'run-state/new',{},'prepare-research');base=session.folder/'prepare-research'
    for name in ('role-roster','research-prerequisites'):
        session.journal['steps']['prepare-research/'+name]={'status':'COMPLETE'}
    _,panels,screen=fixture()
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    design=scientific_round(data,expanded=True)
    write(base/'roster/research-reservation.json',reservation(construct(design,panels,screen)))
    def invalid(w):raise ValueError('original calibration lineage changed')
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_calibration_reuse_workstation.fenced_original',invalid)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_research_qualification.qualify',lambda *a,**kw:pytest.fail('expensive qualification started'))
    w=SimpleNamespace(root=tmp_path,base=base,s=session,args=SimpleNamespace(reuse_calibration_session='old'))
    with session.own():launch.prepare(w,data,design,{})
    row=session.journal['steps']['prepare-research/execution-preflight']
    assert row['status']=='FAILED' and 'lineage changed' in row['reason']
    assert (session.folder/'share/prepare-research.zip').is_file()
