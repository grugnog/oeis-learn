"""Offline archive boundary checks require no engine or HIP execution."""
import json
import zipfile
import pytest
from oeis_learn.pipeline_validation.bundle import pack
from oeis_learn.pipeline_validation.identity import write


def test_archive_reconciles_evidence_after_safe_extraction(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.generalization import bundles
    root=tmp_path/'input';root.mkdir()
    write(root/'summary.json',{'selected_scope':'PARTIAL'})
    write(root/'evidence.json',{'study':{'path':'study.json'}})
    write(root/'study.json',{'identity':'study'})
    archive=pack(root,mandatory=['summary.json','evidence.json','study.json'])
    called=[]
    def verify(record,*,expected_study,root):
        called.append(expected_study)
        assert json.loads((root/'study.json').read_text())['identity']=='study'
        return dict(integrity='PASS',software_status='PARTIAL',scientific_status='UNMEASURED')
    monkeypatch.setattr(bundles,'verify_bundle',verify)
    result=verify_archive(archive)
    assert result['software_status']=='PARTIAL' and called==['study']


def test_generic_recovery_archive_cannot_claim_generalization_verification(tmp_path):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    root=tmp_path/'input';root.mkdir();write(root/'summary.json',{'selected_scope':'FAIL'})
    archive=pack(root,mandatory=['summary.json'])
    with pytest.raises(ValueError,match='evidence'):verify_archive(archive)


def test_archive_rejects_total_expansion_before_reading_payloads(tmp_path):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('large.json','0'*1025)
    with pytest.raises(ValueError,match='cap'):verify_archive(archive,cap=1024)


def test_final_controller_ledger_is_verified_without_rewriting_payloads(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.pipeline_validation.generalization_finalization import publish
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.generalization import bundles
    root=tmp_path/'run';report=root/'report';report.mkdir(parents=True)
    write(report/'summary.json',dict(selected_scope='PASS'))
    write(report/'evidence.json',{'study':{'path':'study.json'}});write(report/'study.json',{'identity':'study'})
    archive=pack(report,mandatory=['summary.json','evidence.json','study.json'])
    summary=seal(dict(selected_scope='PASS',integrity='PASS',archive=reference(archive,root),scientific_status='UNMEASURED'))
    workflow=seal(dict(status='PASS',plan=[{'name':n} for n in ('report','verify-bundle')],receipts=[dict(stage=n,exit_code=0) for n in ('report','verify-bundle')]))
    verification=root/'verify-bundle';verification.mkdir()
    write(verification/'summary.json',summary)
    write(verification/'result.json',seal(dict(version='generalization-verification-result/v1',status='COMPLETE',archive=summary['archive'],summary=reference(verification/'summary.json',verification),exit_code=0)))
    final=publish(root,workflow,summary)
    monkeypatch.setattr(bundles,'verify_bundle',lambda *a,**kw:dict(integrity='PASS',software_status='PASS'))
    assert verify_archive(final)['software_status']=='PASS'
    target=root/'final-export'
    write(target/'workflow-final.json',seal(dict(workflow,status='FAIL')),exclusive=False)
    (target/'manifest.json').unlink()
    (root/'final-export.zip').unlink(missing_ok=True)
    tampered=pack(target,mandatory=['summary.json','evidence.json','study.json','workflow-final.json','verification-final.json','report-manifest.json','verification-terminal.json'])
    with pytest.raises(ValueError,match='workflow status'):verify_archive(tampered)
