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
