"""Installed ownership of PAC follow-ups and thin operator wiring."""
import importlib.util
from pathlib import Path
import pytest


def helper():
    path=Path(__file__).resolve().parents[2]/'specs/012-parameterized-adaptive-curriculum/validation/run.py'
    spec=importlib.util.spec_from_file_location('pac_serial_followups',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('control',['reference-only','fresh-control'])
def test_real_baseline_branch_claims_owned_root_and_verifies_retained_weights(tmp_path,monkeypatch,control):
    import json
    from oeis_learn.pipeline_validation.identity import digest,seal,write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import checkpoint_request
    h=helper();source=tmp_path/'original';source.mkdir();(source/'weights.pt').write_bytes(b'original retained tensors')
    descriptor=seal(dict(version='generalization-checkpoint/v1',kind='full_state',weights=reference(source/'weights.pt',source),
        state=None,initialization=digest('initial'),parent=None,tokens=25000000,updates=100,pins=['latest'],compatibility=digest('old control')))
    study=dict(control_mode=control,arms=['A','B','C'] if control=='fresh-control' else ['B','C'])
    runtime=dict(control=dict(checkpoint=descriptor['identity']))
    write(tmp_path/'prepared/study.json',study);write(tmp_path/'prepared/runtime.json',runtime)
    write(tmp_path/'context.json',dict(checkpoints=dict(static=dict(weights_root='original',checkpoint=descriptor))))
    receipts=dict(acceptance=dict(resolved_manifest='prepared/study.json',resolved_runtime='prepared/runtime.json'))
    plan=dict(stages=[dict(extra=['context.json','20261016'])])
    stage=dict(name='baseline',operation='evaluate',manifest='@admission');target=tmp_path/'baseline';target.mkdir();seen=[]
    def invoke(plan,stage,target,workspace,receipts):
        request=stage['extra'][stage['extra'].index('--checkpoints')+1]
        arm,root,actual=checkpoint_request(workspace,request,study,runtime,{}, {})
        assert arm==('A-reference' if control=='fresh-control' else 'A')
        assert root==target/'private/retained' and actual==descriptor
        assert (root/'weights.pt').read_bytes()==(source/'weights.pt').read_bytes()
        seen.append(request);return dict(status='PASS')
    monkeypatch.setattr(h,'_canonical_command_stage',invoke)
    assert h._canonical_stage(plan,stage,target,tmp_path,receipts)['status']=='PASS'
    assert seen
    # A new helper invocation cannot overwrite previously owned retained bytes.
    with pytest.raises(FileExistsError):h._canonical_stage(plan,stage,target,tmp_path,receipts)
    (source/'weights.pt').write_bytes(b'changed tensor bytes')
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError):h._canonical_stage(plan,stage,other,tmp_path,receipts)
    assert len(seen)==1


@pytest.mark.parametrize('minimal',[False,True])
def test_helper_exception_is_private_locally_and_bound_in_normal_or_minimal_report(tmp_path,monkeypatch,minimal):
    import json,zipfile
    from oeis_learn.pipeline_validation.identity import digest
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.pipeline_validation import generalization_pac_export as exporter
    h=helper();order=h.stage_order()
    plan=h.prepare_plan(tmp_path,engine='podman',stages=[dict(name=n,operation='prepare',manifest='unused',seconds=3600,
        extra=[],requires=[] if i==0 else [order[i-1]]) for i,n in enumerate(order)],inputs=[],source=digest('source'),images=[digest('cpu'),digest('gpu')])
    if minimal:monkeypatch.setattr(exporter,'collect',lambda *a,**kw:(_ for _ in ()).throw(ValueError('private rich export problem')))
    def fail(stage,target):raise FileNotFoundError('/home/private/operator/frozen-input')
    result=h.execute_plan(plan,tmp_path/'failed',workspace=tmp_path,runner=fail)
    assert result['status']=='FAIL' and result['errors'][0]['kind']=='FileNotFoundError'
    payload=tmp_path/'failed/admission/helper-failure.json'
    assert payload.is_file() and (tmp_path/'failed/admission/private/helper-exception.json').is_file()
    public=json.loads(payload.read_text());assert public['location']['function']=='execute_plan'
    assert verify_archive(tmp_path/'failed/report.zip')['software_status']=='FAIL'
    with zipfile.ZipFile(tmp_path/'failed/report.zip') as z:
        assert json.loads(z.read('failures/helper-admission.json'))==public
        assert all(b'/home/private' not in z.read(n) for n in z.namelist())
    canonical=exporter.export(tmp_path/'failed',tmp_path/'canonical')
    with zipfile.ZipFile(canonical) as z:
        assert json.loads(z.read('failures/helper-admission.json'))==public
    original=(tmp_path/'failed/workflow.json').read_bytes()
    h.recover(tmp_path/'failed',tmp_path/'recovered')
    assert (tmp_path/'failed/workflow.json').read_bytes()==original
    payload.write_text('{}')
    bad=exporter.export(tmp_path/'failed',tmp_path/'unverified')
    with zipfile.ZipFile(bad) as z:
        assert 'failures/helper-admission.json' not in z.namelist()
        assert any('helper' in s for s in json.loads(z.read('evidence.json'))['omissions'])
    # Even a resealed workflow cannot turn an arbitrary string into public data.
    from oeis_learn.pipeline_validation.identity import seal,write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    write(payload,seal(dict(public,failure='private teacher payload')),exclusive=False)
    workflow=h._read(tmp_path/'failed/workflow.json')
    workflow['receipts'][0]['artifacts']=[reference(payload,payload.parent) if r['path']=='helper-failure.json' else r
        for r in workflow['receipts'][0]['artifacts']]
    write(tmp_path/'failed/workflow.json',seal(workflow),exclusive=False)
    unsafe=exporter.export(tmp_path/'failed',tmp_path/'unsafe')
    with zipfile.ZipFile(unsafe) as z:
        assert 'failures/helper-admission.json' not in z.namelist()
        assert all(b'private teacher payload' not in z.read(n) for n in z.namelist())


