"""No HIP is launched: subprocess boundary is replaced with a recording adapter."""
import importlib.util
from pathlib import Path
import json
import pytest


def helper():
    path=Path(__file__).resolve().parents[2]/'scripts/generalization_hip_serial.py'
    spec=importlib.util.spec_from_file_location('generalization_serial',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_serial_plan_one_qualification_and_no_research(tmp_path):
    h=helper();plan=h.plan('docker','manifest.json','run-state/new')
    assert [p['operation'] for p in plan]==['preflight','qualify','prepare']+['acceptance']*5+['report','verify-bundle']
    assert sum(p['seconds'] for p in plan)+600==14400
    consumers=[p for p in plan if p['operation'] in ('prepare','acceptance')]
    assert all(p['argv'][p['argv'].index('--qualification')+1]=='run-state/new/qualify' for p in consumers)
    assert not any(p['operation'] in ('train','replay') for p in plan)
    assert '--start-step' not in str(plan)


def test_serial_failure_still_reports_and_preserves_ledger(tmp_path):
    h=helper();calls=[];out=tmp_path/'new'
    def run(stage,root,seconds):
        calls.append(stage['name'])
        return 1 if stage['name']=='refresh' else 0
    code=h.execute(h.plan('docker','manifest.json',str(out)),out,runner=run)
    ledger=json.loads((out/'workflow.json').read_text())
    assert code==1 and 'refresh' in calls and 'parameter-targets' not in calls
    assert 'report' in calls
    assert next(r for r in ledger['receipts'] if r['stage']=='refresh')['exit_code']==1
    assert (tmp_path/'new.zip').exists()  # Missing canonical archive invokes recovery.
    from oeis_learn.pipeline_validation.bundle import verify
    assert verify(tmp_path/'new.zip')['summary_state']=='FAIL'
    with pytest.raises(ValueError):h.execute(h.plan('docker','manifest.json',str(out)),out,runner=run)


def test_restart_is_forwarded_to_canonical_endpoints(tmp_path):
    plan=helper().plan('podman','manifest.json','run-state/new',reuse_run='run-state/old')
    assert all('--reuse-run' in row['argv'] for row in plan if row['operation']!='verify-bundle')
    # The helper does not read an old PASS flag or suppress qualification itself.
    assert sum(row['operation']=='qualify' for row in plan)==1


def test_recovery_archive_excludes_unreviewed_stage_payloads(tmp_path):
    h=helper();root=tmp_path/'new';root.mkdir()
    (root/'prepare').mkdir();(root/'prepare/candidates.json').write_text('{"secret_target":"hidden"}')
    ledger=dict(receipts=[dict(exit_code=1)])
    (root/'workflow.json').write_text(json.dumps(ledger))
    archive=h.recover(root,ledger)
    import zipfile
    with zipfile.ZipFile(archive) as z:
        assert 'prepare/candidates.json' not in z.namelist()
        assert set(z.namelist())=={'summary.json','workflow.json','manifest.json'}


def test_original_consumer_uses_fresh_inspection_and_role_verification(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_qualification_io as io
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.pipeline_validation.identity import seal
    producer=tmp_path/'producer';producer.mkdir()
    q=seal(dict(version='generalization-qualification/v2',images=['cpu','gpu'],source_closure='source',build_dependencies=[]))
    (producer/'qualification.json').write_text(json.dumps(q))
    calls=[]
    def inspect(*args,**kwargs):calls.append('inspect');return {'measured':'new'}
    def invoke(*args,**kwargs):
        calls.append('consume');assert kwargs['settings']['current']=={'measured':'new'}
        assert kwargs['source_roots']['public_evidence']==producer
        return dict(status='PASS',original='original-producer')
    monkeypatch.setattr(io,'current_facts',inspect);monkeypatch.setattr(roles,'project_and_invoke',invoke)
    result=io.consume_original(None,tmp_path,producer,tmp_path/'consumer',required_suites=['cpu','hip'],required_capabilities=['load'])
    assert calls==['inspect','consume'] and result['original']=='original-producer'
    def stale(*args,**kwargs):raise ValueError('current source closure changed')
    monkeypatch.setattr(io,'current_facts',stale)
    with pytest.raises(ValueError,match='source closure'):io.consume_original(None,tmp_path,producer,tmp_path/'stale',required_suites=['cpu'],required_capabilities=['load'])
    assert calls==['inspect','consume']


def test_qualification_reserves_post_run_inspection_without_launching_hip(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_qualification_io as io,runner
    from oeis_learn.experiments import source_snapshot
    observed={}
    def execute(stages,output,**kwargs):
        observed.update(kwargs)
        assert len([s for s in stages if s.adapter=='hip'])==1
        assert sum(s.seconds for s in stages)+kwargs['reserve']<=kwargs['cap']
        return 1
    monkeypatch.setattr(runner,'execute',execute)
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:{})
    result=io.produce(None,tmp_path,tmp_path/'qualification',images=['cpu','gpu'],contract={})
    assert result['status']=='FAIL'
    # Two image inspections, one device inspection and authority validation.
    assert observed['cap']+3*120+120<=7200
