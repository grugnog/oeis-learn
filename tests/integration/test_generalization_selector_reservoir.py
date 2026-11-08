"""Canonical host boundaries; fake only the new image build/device processes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference


def fixture(root,monkeypatch):
    from tests.integration.test_selector_validation import fixture_campaign
    from tests.unit.test_generalization_selector_reservoir import source_pair,ROOT
    from oeis_learn.pipeline_validation import selector_validation_commands as host
    from oeis_learn.generalization.selector_reservoir import registry
    from oeis_learn.experiments import source_snapshot
    old,current=source_pair()
    for name in registry()['files']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((ROOT/name).read_text())
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda _:current)
    campaign=fixture_campaign(root);inputs=host.discover(root,'campaign')
    qroot=root/'probe/qualification';qroot.mkdir(parents=True)
    write(qroot/'source.json',old);(qroot/'build').mkdir();(qroot/'build/Dockerfile').write_text('pinned fixture')
    from oeis_learn.pipeline_validation.generalization_execution import POLICY_ID
    facts=dict(images=['sha256:'+'a'*64,'sha256:'+'b'*64],
        runtime=dict(hip='6',driver='fixture',python='3.12',torch='2.9',execution_policy=POLICY_ID),
        hardware=dict(hip_available=True,device='fixture',memory_bytes=128<<30),precision='fp32',
        resources=dict(memory_bytes=16<<30,cpus=4,torch_intraop=2,torch_interop=1,native_threads=1,learners=1),
        engine_security=dict(engine='podman',runtime_socket=False,engine_version='fixture',runtime='fixture',
            rootless=True,selinux=False,network=False,read_only_root=True,no_new_privileges=True,capabilities='none'))
    facts.update(source_closure=reference(qroot/'source.json',qroot),build_dependencies=[reference(qroot/'build/Dockerfile',qroot)])
    device=seal(dict(version='generalization-device-evidence/v1',device='hip',loaded_weights='fixture',
        finite_forward=True,finite_backward=True,parameters_changed=True,resume_equal=True,
        execution_policy=facts['runtime']['execution_policy']))
    write(qroot/'device.json',device);device_ref=reference(qroot/'device.json',qroot)
    original=seal(dict(version='generalization-producer/v1',status='PASS',facts=facts,
        suites=['bigint-cpu','bigint-gpu'],capabilities=['weight-load','finite-update','native-exact'],
        artifacts=[device_ref],dependencies=[],feature_acceptance=False))
    write(qroot/'producer.json',original)
    q=seal(dict(version='generalization-qualification/v2',status='PASS',**facts,
        suite_selection=original['suites'],covered_capabilities=original['capabilities'],
        producer=dict(status='PASS',receipt=reference(qroot/'producer.json',qroot),artifacts=[device_ref]),artifact_inventory=[]))
    write(qroot/'qualification.json',q)
    probe=seal(dict(version='pac-selector-validation-plan/v1',phase='probes',inputs=inputs,
        source=digest(old),images=q['images'],engine='podman',selector=host.POLICY,
        budgets=host.budgets(256),controls=list(host.CONTROLS),arms=list(host.ARMS),
        training_tokens=0,assessment='NOT_EXECUTED',qualification='probe/qualification'))
    write(root/'probe/plan.json',probe);scoring={}
    for arm in host.ARMS:
        p=root/'probe'/arm/'scoring.json'
        write(p,seal(dict(status='PASS',study=probe['identity'],trace_incomplete=0,
            metrics=dict(training=dict(equal_budget={v:dict(complete=244) for v in ('original','alternative')})))))
        scoring[arm]=reference(p,root)
    result=seal(dict(version='pac-selector-validation-result/v1',status='PASS',phase='probes',
        plan=probe['identity'],freeze_eligible=True,selector=host.POLICY['identity'],scoring=scoring,
        generation={a:dict(status='PASS') for a in host.ARMS}))
    write(root/'probe/result.json',result)
    parent=host.structural_plan(probe,reference(root/'probe/result.json',root),reference(root/'probe/plan.json',root),'old/execution')
    write(root/'old/plan.json',parent);write(root/'old/execution/plan.json',parent)
    role=root/'old/execution/publication/role'
    required={c['family']:c['count'] for c in parent['protocol']['cells'] if c['family'] in registry()['draws']}
    available=seal(dict(protocol=parent['protocol']['identity'],draws=2288,
        available={c['family']:c['count']+3 for c in parent['protocol']['cells'] if c['family'] not in required and c['count']>0},deficits=required))
    write(role/'availability.json',available)
    write(role/'result.json',seal(dict(status='BLOCKED',reason='FRESH_COHORT_QUOTA_DEFICIT',deficits=required,
        availability=reference(role/'availability.json',role))))
    refs=[dict(name=x['name'],input=x['ref'],rows=x['count']) for x in inputs['originals']]
    refs.extend([dict(name='observed-PAC',input=inputs['observed'],rows=128),dict(name='training-probes',input=inputs['probes'],rows=256)])
    families={f:dict(allowed_assignments=2049 if f.startswith('sha256:3f7f') else 2049**2,
        window_free_assignments=6,window_blocked_assignments=(2049 if f.startswith('sha256:3f7f') else 2049**2)-6,
        all_assignments_blocked=False,category=parent['protocol']['families'][f]['category']) for f in required}
    for i,cell in enumerate(refs):
        cell['families']={f:dict(newly_blocked_assignments=v['window_blocked_assignments'] if i==0 else 0,
            cumulative_window_free_assignments=6) for f,v in families.items()}
    from oeis_learn.pipeline_validation.selector_reservoir_commands import AUDIT_SOURCE
    support=seal(dict(version='pac-fresh-support-audit/v2',status='PASS',scientific_status='DIAGNOSTIC_ONLY',
        freeze_eligible=False,training_tokens=0,assessment='EXCLUSION_ONLY',audit_source=AUDIT_SOURCE,protocol=parent['protocol']['identity'],
        plan=dict(reference(root/'old/execution/plan.json',root),identity=parent['identity']),families=families,cells=refs))
    write(root/'support/result.json',support)
    return parent,current


def test_adopted_amendment_qualifies_new_images_then_freezes_without_rerunning_probes(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import selector_reservoir_commands as amended,selector_validation_commands as host
    from oeis_learn.pipeline_validation import engines,generalization_qualification_io as qualification
    parent,current=fixture(tmp_path,monkeypatch);before=(tmp_path/'old/plan.json').read_bytes();events=[]
    monkeypatch.setattr(host,'probes',lambda *a:pytest.fail('custodian-only amendment reran probes'))
    monkeypatch.setattr(engines.Images,'ensure',lambda self,h,target,*a,**kw:events.append(target) or 'sha256:'+('1' if target=='qualification' else '2')*64)
    def produce(h,workspace,output,*,images,contract,pac_seconds):
        events.append('current CPU/native + HIP')
        assert host.read(tmp_path/'new/plan.json',sealed=True)['identity']==contract
        write(output/'source.json',current)
        original=host.read(tmp_path/'probe/qualification/qualification.json',sealed=True)
        write(output/'qualification.json',seal(dict(original,images=images,source_closure=reference(output/'source.json',output))))
        return dict(status='PASS')
    monkeypatch.setattr(qualification,'produce',produce)
    machine=SimpleNamespace(engine=SimpleNamespace(name='podman'))
    args=SimpleNamespace(previous='old',support='support',output='new')
    assert amended.amend(args,tmp_path,lambda _:machine)==0
    plan=host.read(tmp_path/'new/plan.json',sealed=True)
    assert events==['qualification','gpu','current CPU/native + HIP']
    assert plan['inputs']==parent['inputs'] and plan['probe_result']==parent['probe_result']
    assert plan['images']!=parent['images'] and plan['source']!=parent['source']
    assert sum(c['draws'] for c in plan['protocol']['cells'])==7664
    assert (tmp_path/'old/plan.json').read_bytes()==before and not (tmp_path/'new/execution').exists()
    host.verify_structural_plan(tmp_path,tmp_path/'new/plan.json',plan)
    changed=seal(dict(plan,execution='elsewhere'))
    with pytest.raises(ValueError,match='fixed output'):host.verify_structural_plan(tmp_path,tmp_path/'new/plan.json',changed)
    changed=seal(dict(plan,images=list(reversed(plan['images']))))
    with pytest.raises(ValueError,match='qualification'):host.verify_structural_plan(tmp_path,tmp_path/'new/plan.json',changed)
    qpath=tmp_path/'new/qualification/qualification.json';original=qpath.read_bytes();q=json.loads(original)
    write(qpath,seal(dict(q,resources=dict(q['resources'],cpus=8))),exclusive=False)
    with pytest.raises(ValueError,match='qualified resources'):host.verify_structural_plan(tmp_path,tmp_path/'new/plan.json',plan)
    qpath.write_bytes(original)
    with pytest.raises(ValueError,match='fresh'):amended.amend(args,tmp_path,lambda _:machine)
    def consume(h,workspace,producer_root,output,**kwargs):
        assert producer_root==tmp_path/'new/qualification'
        assert kwargs['required_suites']==['bigint-cpu','bigint-gpu']
        events.append('consume new qualification')
    def publish(h,workspace,frozen,output,role,operation,artifacts,roots,settings,kind):
        assert operation=='selector-structural' and settings['protocol']==plan['protocol']
        assert set(artifacts)=={'mechanical_sources','isolation_truth'}
        assert artifacts['mechanical_sources']==[] and len(artifacts['isolation_truth'])==18
        path=output/'role'
        write(path/'availability.json',seal(dict(protocol=plan['protocol']['identity'],draws=7664,deficits={},
            resources=dict(peak_rss_bytes=1024,memory_ceiling_bytes=14<<30,reference_and_exclusion_seconds=1,index_rows=7664))))
        published=dict(status='PASS',availability=reference(path/'availability.json',path))
        write(path/'result.json',seal(published));events.append('new custody');return published
    monkeypatch.setattr(qualification,'consume_original',consume)
    monkeypatch.setattr(host,'_invoke',publish)
    monkeypatch.setattr(host,'compare',lambda *a:events.append('same paired comparison'))
    assert host.structural(SimpleNamespace(plan='new/plan.json'),tmp_path,lambda _:machine)==0
    assert events[-3:]==['consume new qualification','new custody','same paired comparison']
    assert host.read(tmp_path/'new/execution/admission.json',sealed=True)['availability']['draws']==7664
    with pytest.raises(ValueError,match='already started'):
        host.structural(SimpleNamespace(plan='new/plan.json'),tmp_path,lambda _:machine)


@pytest.mark.parametrize('mutation',['evaluated','census','scoring','source','availability'])
def test_changed_parent_support_or_probe_is_rejected_before_any_build(tmp_path,monkeypatch,mutation):
    from oeis_learn.pipeline_validation import selector_reservoir_commands as amended
    parent,current=fixture(tmp_path,monkeypatch)
    if mutation=='evaluated':(tmp_path/'old/execution/B').mkdir()
    elif mutation=='census':
        p=tmp_path/'support/result.json';v=json.loads(p.read_text());v['cells'][0]['families'][next(iter(v['families']))]['newly_blocked_assignments']-=1;write(p,seal(v),exclusive=False)
    elif mutation=='scoring':(tmp_path/'probe/B/scoring.json').write_text('{}')
    elif mutation=='source':(tmp_path/'probe/qualification/source.json').write_text('{}')
    else:(tmp_path/'old/execution/publication/role/availability.json').write_text('{}')
    with pytest.raises(ValueError):amended.amend(SimpleNamespace(previous='old',support='support',output='new'),tmp_path,lambda _:pytest.fail('rejected input reached engine'))
    assert not (tmp_path/'new').exists()


def test_current_qualification_failure_cannot_appear_frozen(tmp_path,monkeypatch,capsys):
    from oeis_learn.pipeline_validation import selector_validation_commands as host,engines,generalization_qualification_io as qualification
    fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(engines.Images,'ensure',lambda *a,**kw:'sha256:'+'1'*64)
    monkeypatch.setattr(qualification,'produce',lambda *a,**kw:dict(status='FAIL',stages=[dict(stage='image.gpu',reason='fixture mismatch')]))
    args=SimpleNamespace(selector_phase='amend',previous='old',support='support',output='new')
    assert host.main(args,tmp_path,lambda _:SimpleNamespace(engine=SimpleNamespace(name='podman')))==1
    assert 'FROZEN' not in capsys.readouterr().out
    assert (tmp_path/'new/failure.zip').exists() and not (tmp_path/'new/execution').exists()


@pytest.mark.parametrize('private_field',[False,True])
def test_blocked_admission_failure_zip_retains_counts_without_private_payloads(tmp_path,monkeypatch,private_field):
    from oeis_learn.pipeline_validation import selector_validation_commands as host
    import zipfile
    def blocked(args,workspace,host_factory):
        args._selector_output=workspace/'new';args._selector_output.mkdir()
        value=dict(version='pac-selector-availability/v1',protocol='sha256:'+'0'*64,
            draws=7664,available={},deficits={'sha256:'+'1'*64:2},reference_rejections={},exclusion_reasons={})
        if private_field:value['weights']='PRIVATE_PAYLOAD'
        write(args._selector_output/'publication/role/availability.json',seal(value))
        raise ValueError('FRESH_COHORT_QUOTA_DEFICIT')
    monkeypatch.setattr(host,'structural',blocked)
    assert host.main(SimpleNamespace(selector_phase='structural'),tmp_path,lambda _:None)==1
    with zipfile.ZipFile(tmp_path/'new/failure.zip') as z:
        index=json.loads(z.read('admission-diagnostics/index.json'))
        assert index['qualification_evidence'] is False
        assert bool(index['omissions']) is private_field
        assert ('admission-diagnostics/availability.json' in z.namelist()) is (not private_field)
        assert b'PRIVATE_PAYLOAD' not in b''.join(z.read(n) for n in z.namelist())


@pytest.mark.parametrize('exceeded',[False,True])
def test_amended_custody_measures_resources_and_stops_before_native_on_breach(tmp_path,monkeypatch,exceeded):
    from oeis_learn.pipeline_validation import selector_validation_roles as role
    from oeis_learn.generalization import structural,preparation,selector_reservoir as reservoir
    from oeis_learn.synthesis import client
    import resource
    cells=[dict(c,count=1) for c in structural.original_cells()]
    monkeypatch.setattr(structural,'original_cells',lambda:cells)
    protocol=seal(dict(version='pac-selector-structural/v2',cells=[dict(family='fresh',count=1)]))
    monkeypatch.setattr(reservoir,'reservoir_design',lambda:protocol)
    row=dict(id='new',family='fresh',source='source',group='g',variant='v',
        values=list(map(str,range(100))),indices=list(map(str,range(100))))
    unrelated=dict(row,id='other',family='other',source='other',group='other',values=['-1']*100)
    monkeypatch.setattr(reservoir,'draw',lambda _: [row])
    monkeypatch.setattr(preparation,'reference_candidate',lambda r:((r,None),None))
    monkeypatch.setattr(role,'_read',lambda root,ref,cap=None:[row]*128 if ref['path']=='observed' else [] if ref['path']=='probes' else [unrelated])
    monkeypatch.setattr(client,'SynthesisClient',lambda *a,**kw:pytest.fail('resource/quota failure reached native'))
    peak=reservoir.MEMORY_CEILING+1024 if exceeded else 1024
    monkeypatch.setattr(resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=peak//1024))
    settings=dict(protocol=protocol,originals=[dict(c,ref=dict(path=c['name'])) for c in cells],
        observed=dict(path='observed'),probes=dict(path='probes'),study='fixture')
    if exceeded:
        with pytest.raises(ValueError,match='memory ceiling'):role.publish_structural(dict(isolation_truth=tmp_path),tmp_path/'out',settings,seconds=30)
    else:
        result=role.publish_structural(dict(isolation_truth=tmp_path),tmp_path/'out',settings,seconds=30)
        assert result['status']=='BLOCKED'
        assert json.loads((tmp_path/'out/availability.json').read_text())['resources']['peak_rss_bytes']==peak
    assert json.loads((tmp_path/'out/preparation-resources.json').read_text())['peak_rss_bytes']==peak


def test_success_archive_includes_bound_admission_without_changing_score_calculations(tmp_path):
    from oeis_learn.pipeline_validation import selector_validation_commands as host
    from oeis_learn.generalization.selector_validation import paired_metrics
    import zipfile
    output=tmp_path/'report';output.mkdir()
    plan=seal(dict(phase='structural'));write(output/'plan.json',plan)
    admission=seal(dict(version='pac-selector-reservoir-admission/v1',plan=plan['identity'],availability=dict(draws=7664)))
    write(output/'admission.json',admission)
    roster={'case':dict(category='certified',family='f')}
    case=dict(success=False,complete=True,candidate_sha256=None)
    metrics={c:paired_metrics(roster,{'case':case},{'case':case}) for c in host.CONTROLS}
    results={}
    for arm in host.ARMS:
        scored=dict(trace_incomplete=0,roster=roster,metrics=dict(certified=metrics),
            changed={c:0 for c in host.CONTROLS},selector_seconds={c:0 for c in host.CONTROLS},
            outcomes={s:{'case':{c:case for c in host.CONTROLS}} for s in ('original','alternative')})
        write(output/arm/'scoring/role/scoring.json',scored);results[arm]=scored
    result=host.report(tmp_path,plan,output,results,{a:dict(status='PASS') for a in host.ARMS})
    assert result['metrics']['B']['certified']['equal_budget']['original']['success']==0
    assert result['freeze_eligible'] is False
    with zipfile.ZipFile(output/'report.zip') as archive:
        assert json.loads(archive.read('admission.json'))==admission
