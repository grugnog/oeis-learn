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
