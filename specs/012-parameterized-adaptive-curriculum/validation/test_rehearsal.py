"""Temporary harness stop/resume behavior with independent producer fixtures."""
from pathlib import Path
import importlib.util
import pytest
from oeis_learn.pipeline_validation.identity import digest,write
from oeis_learn.pipeline_validation.generalization_preparation import reference


def helper():
    spec=importlib.util.spec_from_file_location('r0_serial',Path(__file__).with_name('run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_rehearsal_is_explicit_before_any_frozen_research_stage():
    h=helper();order=h.stage_order(research=True,arms=['B','C','D'],rehearsal=True)
    assert order[:6]==['admission','qualification','calibration','acceptance','rehearsal','baseline']
    assert 'rehearsal' not in h.stage_order()
    with pytest.raises(ValueError):h.stage_order(arms=['B','C'],rehearsal=True)


def test_stop_then_resume_reuses_all_gates_without_research_initialization(tmp_path,monkeypatch):
    h=helper();arms=['B','C','D'];order=h.stage_order(research=True,arms=arms,rehearsal=True)
    stages=[dict(name=n,operation='prepare',manifest='none',seconds=3600,extra=[],requires=[] if i==0 else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('s'),images=[digest('cpu'),digest('gpu')],research=True,arms=arms,rehearsal=True)
    calls=[]
    def run(stage,target):
        calls.append(stage['name']);write(target/'checked.json',dict(stage=stage['name']))
        return dict(status='PASS',producer=digest(stage),artifacts=[reference(target/'checked.json',target)],
                    process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(within_bound=True,acknowledged=True))
    # Isolate scheduler semantics from the independent archive tests.
    from oeis_learn.pipeline_validation import generalization_archive as archives
    def export(root,target):
        path=target.with_suffix('.zip');path.write_bytes(b'fixture');return path
    monkeypatch.setattr(h,'recover',export)
    monkeypatch.setattr(archives,'verify_archive',lambda path:dict(software_status='PARTIAL',rehearsal_status='PASS'))
    one=h.execute_plan(plan,tmp_path/'r0',workspace=tmp_path,runner=run,stop_after_rehearsal=True)
    assert one['status']=='PARTIAL' and not one['errors'] and calls==order[:5]
    old=(tmp_path/'r0/workflow.json').read_bytes();calls.clear()
    h.execute_plan(plan,tmp_path/'r1',workspace=tmp_path,runner=run,previous=tmp_path/'r0')
    assert calls==order[5:] and (tmp_path/'r0/workflow.json').read_bytes()==old
    assert all(r['reused'] for r in h._read(tmp_path/'r1/workflow.json')['receipts'][:5])


def test_stop_without_a_rehearsal_is_rejected_before_work(tmp_path):
    h=helper();order=h.stage_order();stages=[dict(name=n,operation='prepare',manifest='none',seconds=3600,extra=[],requires=[] if not i else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('s'),images=[digest('cpu'),digest('gpu')])
    with pytest.raises(ValueError):h.execute_plan(plan,tmp_path/'bad',workspace=tmp_path,runner=lambda *a:None,stop_after_rehearsal=True)
    assert not (tmp_path/'bad').exists()


def test_actual_r0_archive_verifies_partial_learners_and_stdlib_scope(tmp_path):
    """Real collector/packer/verifier; synthetic device attestations, not HIP proof."""
    import json,subprocess,sys,os
    from tests.integration.test_generalization_archive import test_pac_complete_synthetic_proof_closure_roundtrips_real_collector_and_verifier as fixture
    from tests.unit.test_generalization_rehearsal import record
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.pipeline_validation.generalization_pac_calibration import forecast
    from oeis_learn.pipeline_validation.generalization_pac_export import export
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    fixture(tmp_path,True)
    # The existing fixture ends with a deliberate corruption; restore its public original.
    import shutil
    shutil.copyfile(tmp_path/'public-payloads/acceptance/proof-1/training-proof.json',tmp_path/'campaign/acceptance/execution/execution/role/acceptance/proof-1/training-proof.json')
    root=tmp_path/'campaign';plan=json.loads((root/'plan.json').read_text());workflow=json.loads((root/'workflow.json').read_text())
    order=helper().stage_order(research=True,arms=['B','C','D'],rehearsal=True)
    plan.update(version='pac-serial-plan/v2',rehearsal=True,research=True,arms=['B','C','D'],stages=[dict(name=n,extra=['context','20261016'] if not i else []) for i,n in enumerate(order)])
    plan=seal(plan);write(root/'plan.json',plan,exclusive=False)
    cal=root/'calibration/execution/execution/role/calibration.json';value=json.loads(cal.read_text())
    value['forecasts']['D']=forecast(value['samples'],updates=value['forecast_updates'],arm='D',existing={});write(cal,seal(value),exclusive=False)
    r=record();r.update(source=plan['source'],environment=json.loads((root/'acceptance/execution/execution/role/acceptance/pac-acceptance.json').read_text())['environment'])
    path=root/'rehearsal/execution/execution/role/rehearsal.json';write(path,seal(r))
    for e in workflow['receipts']:e['artifacts']=[reference(p,root/e['stage']) for p in sorted((root/e['stage']).rglob('*')) if p.is_file()]
    workflow['receipts'].append(dict(stage='rehearsal',status='PASS',producer=digest('R0'),binding=digest('R0binding'),process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True),elapsed_seconds=10,reused=False,directory='rehearsal',artifacts=[reference(path,root/'rehearsal')]))
    workflow.update(plan=plan['identity'],status='PARTIAL',errors=[]);write(root/'workflow.json',seal(workflow),exclusive=False)
    archive=export(root,tmp_path/'r0-report');verdict=verify_archive(archive)
    assert verdict['software_status']=='PARTIAL' and verdict['rehearsal_status']=='PASS' and verdict['scientific_status']=='INCOMPLETE' and not verdict['feature_qualified']
    repo=Path(__file__).resolve().parents[3]
    result=subprocess.run([sys.executable,'-S','-c','import sys;from oeis_learn.pipeline_validation.generalization_archive import verify_archive;assert verify_archive(sys.argv[1])["rehearsal_status"]=="PASS"',str(archive)],cwd=repo,env=dict(os.environ,PYTHONPATH=str(repo/'src')),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    # Missing or resealed false diagnostic must never survive as a passed R0.
    r['runs']['D']['updates']=255;write(path,seal(r),exclusive=False)
    workflow['receipts'][-1]['artifacts']=[reference(path,root/'rehearsal')]
    write(root/'workflow.json',seal(workflow),exclusive=False)
    bad=export(root,tmp_path/'false-r0')
    assert verify_archive(bad)['software_status']=='FAIL'


def test_real_role_job_uses_calibrated_child_graph_not_short_initial_cap(tmp_path,monkeypatch):
    import json
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import rehearsal_envelope
    from oeis_learn.pipeline_validation.identity import seal
    h=helper();e=envelope(purpose='research',arms=['B','C','D'],rehearsal=True)
    e['learner']['work']=300000;e['learner']['outer']=sum(v for k,v in e['learner'].items() if k!='outer')
    plan=dict(arms=['B','C','D'],engine='podman',images=[digest('cpu'),digest('gpu')],source=digest('source'),rehearsal=True)
    target=tmp_path/'role';target.mkdir();observed=[]
    def launch(argv,supervision,terminal,*,seconds):
        observed.append(seconds);write(supervision/'process.json',dict(outcome='COMPLETE',returncode=0))
        job=json.loads((target/'job.json').read_text())
        value=seal(dict(job=job['identity'],source=plan['source'],status='PASS',inventory=[],result={}))
        write(terminal,value);return value
    monkeypatch.setattr(h,'launch_terminal',launch)
    h._role_job(plan,'pac-rehearse',{'runtime':{'envelope':e}},{},{},target,tmp_path)
    job=json.loads((target/'job.json').read_text())
    assert job['envelope']==rehearsal_envelope(e)
    assert observed==[job['envelope']['outer']+9000]


def test_failed_rehearsal_suppresses_research_and_read_only_recovery_verifies(tmp_path):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    h=helper();order=h.stage_order(research=True,arms=['B','C','D'],rehearsal=True)
    stages=[dict(name=n,operation='prepare',manifest='none',seconds=3600,extra=[],requires=[] if not i else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('s'),images=[digest('cpu'),digest('gpu')],research=True,arms=['B','C','D'],rehearsal=True)
    calls=[]
    def runner(stage,target):
        calls.append(stage['name']);write(target/'result.json',dict(status='FAIL' if stage['name']=='rehearsal' else 'PASS'))
        return dict(status='FAIL' if stage['name']=='rehearsal' else 'PASS',producer=digest(stage),artifacts=[reference(target/'result.json',target)],process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True))
    value=h.execute_plan(plan,tmp_path/'run',workspace=tmp_path,runner=runner)
    assert calls==order[:5] and value['status']=='FAIL'
    before=(tmp_path/'run/workflow.json').read_bytes()
    recovered=h.recover(tmp_path/'run',tmp_path/'recovered')
    assert verify_archive(recovered)['software_status']=='FAIL'
    assert (tmp_path/'run/workflow.json').read_bytes()==before


def test_documented_r0_cli_stops_with_zero_only_for_verified_expected_pause(tmp_path,monkeypatch):
    h=helper();monkeypatch.setattr(h,'ROOT',tmp_path);seen=[]
    def build(args):
        seen.append((args.research,args.bandit_only,args.rehearsal,args.stop_after_rehearsal,args.seed))
        return tmp_path/'plan.json',{'engine':'podman'}
    monkeypatch.setattr(h,'build_plan',build)
    def execute(plan,output,**kwargs):
        seen.append(kwargs['stop_after_rehearsal'])
        return dict(status='PARTIAL',stop_reason='REHEARSAL_COMPLETE_RESEARCH_PENDING',rehearsal_status='PASS',export_status='PARTIAL')
    monkeypatch.setattr(h,'execute_plan',execute)
    argv=['run','--engine','podman','--research','--bandit-only','--rehearsal','--stop-after-rehearsal','--seed','20261016','--output','r0']
    assert h.main(argv)==0 and seen==[(True,True,True,True,20261016),True]
    monkeypatch.setattr(h,'execute_plan',lambda *a,**kw:dict(status='FAIL',stop_reason='REHEARSAL_COMPLETE_RESEARCH_PENDING',rehearsal_status='PASS',export_status='FAIL'))
    assert h.main(argv)==1  # a stale pause reason cannot mask failed finalization
