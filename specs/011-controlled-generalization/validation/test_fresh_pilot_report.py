import importlib.util
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fresh_pilot_report',HERE/'fresh_pilot_report.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def fixture():
    from oeis_learn.pipeline_validation.identity import digest
    study=dict(identity=digest('study'),seeds=[20261016],inference=dict(attempts=8))
    rows=[dict(id=f'row-{i}',role='selection',family=digest('f'+str(i%4)),
        stratum='parameter' if i<128 else 'structural',source='protected',indices=['0']*100,values=['123']*100)
        for i in range(256)]
    return study,rows


def test_fixed_denominators_missing_controls_and_no_private_payloads():
    study,rows=fixture();plan=m.measurement_plan(study,rows)
    assert len(plan['roster'])==2*7*256
    assert 'protected' not in str(plan) and 'values' not in str(plan)
    outcome=dict(complete=True,success=True,scoring=dict(elapsed_seconds=.01))
    scored={'static':dict(result=dict(outcomes={'row-0':{'equal_budget':outcome}}))}
    attempts=m.measurements(plan,rows,scored)
    assert len(attempts['attempts'])==1
    from oeis_learn.generalization.reconciliation import reconcile_measurements
    result=reconcile_measurements(plan,attempts,study)
    cell=result['seeds']['static/equal_budget/parameter']['20261016']
    assert cell['n']==128 and cell['successes']==1 and cell['unresolved']==127
    assert cell['lower']=='1/128' and cell['upper']=='1'
    assert result['costs']['service']['unavailable']==1


def test_no_oracle_union_and_corrupt_outcomes_fail_closed():
    study,rows=fixture();plan=m.measurement_plan(study,rows)
    outcomes={'row-0':{'raw':dict(complete=True,success=True),
        'equal_budget':dict(complete=True,success=False), 'bank':dict(complete=False,success=False)}}
    scored={'static':dict(result=dict(outcomes=outcomes))}
    measured=m.measurements(plan,rows,scored)['attempts']
    assert sorted(str(r['outcome']) for r in measured)==['False','None','True']
    outcomes['row-0']['equal_budget']['success']=1
    with pytest.raises(ValueError):m.measurements(plan,rows,scored)
    outcomes['row-0']['equal_budget']['success']=False
    outcomes['foreign']={}
    with pytest.raises(ValueError):m.measurements(plan,rows,scored)


def test_a_missing_or_assessment_row_cannot_silently_change_the_denominator():
    study,rows=fixture()
    with pytest.raises(ValueError):m.measurement_plan(study,rows[:-1])
    rows[0]['role']='assessment'
    with pytest.raises(ValueError):m.measurement_plan(study,rows)


def test_status_requires_both_exit_code_and_original_terminal_result():
    assert m.terminal_status(0,{'status':'PASS'})=='PASS'
    assert m.terminal_status(1,{'status':'PASS'})=='FAIL'
    assert m.terminal_status(0,None)=='FAIL'
    assert m.terminal_status(3,{'status':'PARTIAL'})=='PARTIAL'
    assert m.terminal_status(130,None)=='INTERRUPTED'


def original_fixture(root,monkeypatch):
    """Synthetic originals exercise offline integrity, not device qualification."""
    import json
    from oeis_learn.pipeline_validation.identity import seal,write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.generalization.contracts import freeze
    from oeis_learn.generalization.qualification import FACTS
    from tests.unit import test_generalization_feature_acceptance as cases
    root.mkdir()
    fixtures=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())
    write(root/'base.json',{});base=reference(root/'base.json',root)
    def refs(value):
        if isinstance(value,dict):
            if set(value)=={'path','bytes','sha256'}:return dict(base)
            return {k:refs(v) for k,v in value.items()}
        if isinstance(value,list):return [refs(v) for v in value]
        return value
    study=refs(fixtures['study']);study['seeds']=[20261016];study=freeze('study',study).record()
    write(root/'device.json',seal(dict(version='generalization-device-evidence/v1',device='hip',
        loaded_weights='synthetic-fixture-not-hardware-evidence',finite_forward=True,finite_backward=True,parameters_changed=True)))
    device=reference(root/'device.json',root)
    q=refs(fixtures['qualification']);facts={k:q[k] for k in FACTS}
    producer=seal(dict(version='generalization-producer/v1',status='PASS',facts=facts,
        suites=q['suite_selection'],capabilities=q['covered_capabilities'],artifacts=[device],dependencies=[],feature_acceptance=False))
    write(root/'producer.json',producer)
    q['producer']=dict(status='PASS',receipt=reference(root/'producer.json',root),artifacts=[device]);q['artifact_inventory']=[device]
    write(root/'qualification.json',freeze('qualification',q).record())
    monkeypatch.setattr(cases,'producer',lambda r:(producer,facts))
    _,_,feature=cases.fixture(root);write(root/'feature.json',feature)
    return study


@pytest.mark.parametrize('status',['PASS','PARTIAL','FAIL','INTERRUPTED'])
def test_public_report_canonical_roundtrip_and_original_controller_failure(tmp_path,monkeypatch,status):
    import json
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import generalization_cli, generalization_roles, generalization_role
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.pipeline_validation.identity import write,seal
    from oeis_learn.generalization.interfaces import EXIT_CODES
    from oeis_learn import container_validation
    import fresh_pilot as runner
    qualified=tmp_path/'qualified';study=original_fixture(qualified,monkeypatch)
    _,rows=fixture();plan=m.measurement_plan(study,rows)
    stages=[]
    for name in m.REQUIRED:
        original=tmp_path/(name+'.json');write(original,seal(dict(status=status if name=='controller' else 'PASS',synthetic=True)))
        stages.append(dict(stage=name,status=status if name=='controller' else 'PASS',
            exit_code=EXIT_CODES[status] if name=='controller' else 0,elapsed_seconds=1,command=['synthetic'],originals=[str(original)]))
    public=tmp_path/'public'
    record=m.assemble_public(public,study=study,plan=plan,targets=rows,scored={},stages=stages,
        qualified=qualified,selection=dict(assessment='SEALED'))
    assert record['software_status']==status
    assert 'protected' not in (public/'fresh/measurement-plan.json').read_text()
    host=SimpleNamespace(engine=SimpleNamespace(image=lambda value:value))
    monkeypatch.setattr(container_validation,'inspect_current_image',lambda *a,**kw:None)
    def invoke(host,image,root,role,artifacts,**kwargs):
        inputs=kwargs.get('source_roots',{}).get('public_evidence',root)
        target=root/kwargs['output']/'role';target.mkdir(parents=True)
        handler=generalization_role.package if kwargs['operation']=='package' else generalization_role.verify_archive
        return handler({'public_evidence':inputs},target,kwargs['settings'],seconds=kwargs['seconds'])
    monkeypatch.setattr(generalization_roles,'project_and_invoke',invoke)
    image='sha256:'+'1'*64
    args=SimpleNamespace(operation='report',inputs='public',manifest='public/fresh/study.json',output='reported',image=image)
    code=generalization_cli.main(args,tmp_path,lambda args:host)
    assert code==EXIT_CODES[status]
    args=SimpleNamespace(operation='verify-bundle',archive='reported.zip',output='checked',image=image)
    verified=generalization_cli.main(args,tmp_path,lambda args:host)
    summary=runner.verified_archive(tmp_path,tmp_path/'reported.zip',tmp_path/'checked',code=code,verified=verified,expected=status)
    assert summary['selected_scope']==status and verify_archive(tmp_path/'reported.zip')['integrity']=='PASS'
    terminal=tmp_path/'checked/result.json';value=json.loads(terminal.read_text())
    value['archive']['sha256']='sha256:'+'0'*64;terminal.write_text(json.dumps(seal(value)))
    with pytest.raises(ValueError,match='binding'):runner.verified_archive(tmp_path,tmp_path/'reported.zip',tmp_path/'checked',code=code,verified=verified,expected=status)


def test_diagnostics_capture_failed_research_controls_without_truth(tmp_path):
    import json
    from oeis_learn.pipeline_validation.identity import write
    write(tmp_path/'research/generate-static/execution/role/result.json',dict(status='FAIL',error='worker deadline',source='SECRET'))
    write(tmp_path/'private/logs/train-static/process.json',dict(returncode=1,elapsed_seconds=12,command=['SECRET']))
    write(tmp_path/'research/private/evaluation/scorer/result.json',dict(status='PASS',error='SECRET'))
    extra=m.diagnostics(tmp_path,tmp_path/'diagnostics')
    text=''.join(p.read_text() for p,_ in extra)
    assert 'worker deadline' in text and '12' in text and 'SECRET' not in text
    assert len(extra)==3


def test_public_export_rejects_oversized_or_symlinked_extra(tmp_path,monkeypatch):
    study=original_fixture(tmp_path/'qualified',monkeypatch);_,rows=fixture();plan=m.measurement_plan(study,rows)
    original=tmp_path/'large.log'
    with original.open('wb') as stream:stream.truncate(501<<20)
    with pytest.raises(ValueError,match='budget'):m.assemble_public(tmp_path/'out',study=study,plan=plan,targets=rows,
        scored={},stages=[],qualified=tmp_path/'qualified',selection={},extra=[(original,'fresh/large.log')])
    link=tmp_path/'link.log';link.symlink_to(original)
    with pytest.raises(ValueError,match='regular'):m.assemble_public(tmp_path/'out2',study=study,plan=plan,targets=rows,
        scored={},stages=[],qualified=tmp_path/'qualified',selection={},extra=[(link,'fresh/link.log')])


def test_preparation_failure_export_keeps_counts_and_cell_without_protected_values(tmp_path):
    from oeis_learn.pipeline_validation.identity import seal,write,digest
    base=tmp_path/'research/prepare/preparation/role/private/prepared'
    write(base/'preparation-result.json',seal(dict(status='BLOCKED',reason='QUOTA_SHORTFALL',cell='static',study=None)))
    write(base/'cells/static/ledger.json',seal(dict(version='generalization-cell-preparation/v1',role='training',status='BLOCKED',draws=20,
        shortfalls=['rows:4','family:'+digest('family')+':4'],
        rejections=[dict(draw=1,reason='cross-role exclusion'),dict(draw=2,reason='SECRET protected program/terms')],
        private_payload='SECRET')))
    write(base/'cells/static/private/targets.json',dict(secret='SECRET'))
    extra=m.diagnostics(tmp_path,tmp_path/'diagnostics')
    summary=m.preparation_diagnostics(tmp_path)
    assert summary['terminal']['cell']=='static'
    assert summary['cells'][0]['shortfalls'][0]=='rows:4'
    assert summary['cells'][0]['rejection_counts']=={'cross-role exclusion':1,'OTHER':1}
    assert 'SECRET' not in ''.join(p.read_text() for p,_ in extra)


def test_preparation_failure_projection_rejects_links_and_unrecognized_labels(tmp_path):
    from oeis_learn.pipeline_validation.identity import seal,write
    ledger=tmp_path/'research/prepare/preparation/role/private/prepared/cells/static/ledger.json'
    write(ledger,seal(dict(version='generalization-cell-preparation/v1',role='training',status='BLOCKED',draws=1,
        shortfalls=['SECRET'],rejections=[])))
    with pytest.raises(ValueError,match='shortfall'):m.preparation_diagnostics(tmp_path)
    ledger.unlink();ledger.symlink_to(tmp_path/'outside.json');write(tmp_path/'outside.json',{})
    with pytest.raises(ValueError,match='symlink'):m.preparation_diagnostics(tmp_path)


def test_reference_shortfall_and_timeout_progress_are_exported_without_sources(tmp_path):
    import json
    from oeis_learn.pipeline_validation.identity import write,seal
    record=seal(dict(version='fresh-pilot-feasibility/v1',status='BLOCKED',training_tokens=200,
        cells=[dict(name='refresh-11',status='BLOCKED',draws=100,admitted=60,selected=60,tokens=200,
                    shortfalls=['rows:4'],source='PRIVATE SOURCE',values=['PRIVATE TERMS'],
                    rejection_counts={'cross-role exclusion':39,'PRIVATE SOURCE':1})]))
    write(tmp_path/'run/feasibility/progress.json',record)
    files=m.diagnostics(tmp_path/'run',tmp_path/'report')
    summary=json.loads((tmp_path/'report/feasibility-summary.json').read_text())
    assert summary['cells'][0]['shortfalls']==['rows:4'] and summary['training_tokens']==200
    assert summary['cells'][0]['rejection_counts']=={'cross-role exclusion':39,'OTHER':1}
    assert 'PRIVATE' not in json.dumps(summary)
    assert any(p.name=='feasibility-summary.json' for p,_ in files)


def test_interrupted_cell_progress_is_diagnostic_only_and_redacted(tmp_path):
    import json
    from oeis_learn.pipeline_validation.identity import seal,write
    cell=tmp_path/'research/prepare/preparation/role/private/prepared/cells/refresh-01'
    record=dict(version='generalization-cell-progress/v1',role='training',phase='native',
        draws=1024,admitted=1000,native_rows=1000,elapsed_seconds=40,
        reference_seconds=2,native_seconds=35,selection_seconds=0)
    write(cell/'progress.json',seal(record))
    result=m.preparation_diagnostics(tmp_path)
    assert result['qualification_evidence'] is False
    assert result['cells'][0]['status']=='INCOMPLETE'
    assert result['cells'][0]['progress']['native_rows']==1000
    m.diagnostics(tmp_path,tmp_path/'export')
    exported=json.loads((tmp_path/'export/preparation-summary.json').read_text())
    assert exported==result
    for extra in ({'source':'private program'},{'native_rows':1025},{'phase':'private program'},{'native_seconds':-1}):
        write(cell/'progress.json',seal(dict(record,**extra)),exclusive=False)
        with pytest.raises(ValueError):m.preparation_diagnostics(tmp_path)
