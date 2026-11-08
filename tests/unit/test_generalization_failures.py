"""Root cause survives wrapper/cleanup/export failures without private payloads."""
import json
from pathlib import Path
import pytest
from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference


def test_safe_quota_detail_and_unknown_private_exception():
    from oeis_learn.pipeline_validation.generalization_failures import public_failure
    from oeis_learn.generalization.structural import QuotaDeficit
    family=digest('family')
    detail=public_failure(QuotaDeficit({family:2}))
    assert detail['kind']=='QuotaDeficit' and detail['code']=='structural_quota_deficit'
    assert detail['deficits']=={family:2}
    hidden=public_failure(ValueError('/home/private/path (program hidden-teacher) 123456789'))
    assert 'private/path' not in json.dumps(hidden) and 'hidden-teacher' not in json.dumps(hidden)
    with pytest.raises(ValueError):public_failure(QuotaDeficit({'private teacher':2}))


def test_bounded_verified_discovery_orders_application_before_process_and_cleanup(tmp_path):
    from oeis_learn.pipeline_validation.generalization_failures import stage_failures
    role=tmp_path/'execution/execution/role'
    write(role/'result.json',seal(dict(status='FAIL',reason='QuotaDeficit',error='completed post-exclusion quota deficit; no timeout/reseed retry',private_values=['secret'])))
    write(tmp_path/'supervision/process.json',dict(outcome='FAIL',returncode=1,failure='RuntimeError: /private/host/path'))
    write(role/'supervision/cleanup.json',dict(acknowledged=False,within_bound=False,failure='private cleanup details'))
    refs=[reference(p,tmp_path) for p in tmp_path.rglob('*.json')]
    result=stage_failures(tmp_path,refs=refs)
    assert [f['detail']['kind'] for f in result['failures']]==['QuotaDeficit','RuntimeError','CleanupFailure']
    assert result['failures'][0]['origin']=='execution/execution/role/result.json'
    assert 'secret' not in json.dumps(result) and '/private/host' not in json.dumps(result)
    assert result['failures'][0]['producer_sha256']==reference(role/'result.json',tmp_path)['sha256']
    (role/'result.json').write_text('{}')
    changed=stage_failures(tmp_path,refs=refs)
    assert not any(f['detail']['kind']=='QuotaDeficit' for f in changed['failures'])
    assert changed['omissions']


def test_scanner_never_reads_private_payloads_and_reports_discovery_bound(tmp_path):
    from oeis_learn.pipeline_validation.generalization_failures import stage_failures,public_failure
    from oeis_learn.generalization.structural import QuotaDeficit
    record=seal(dict(status='FAIL',failure=public_failure(QuotaDeficit({digest('f'):3}))))
    write(tmp_path/'role/result.json',record)
    write(tmp_path/'private/role/result.json',seal(dict(status='FAIL',reason='PrivateTeacher',error='secret')))
    details=stage_failures(tmp_path)
    assert len(details['failures'])==1 and details['failures'][0]['detail']['deficits']=={digest('f'):3}
    assert stage_failures(tmp_path,max_entries=1)['omissions']


def test_canonical_negative_transport_preserves_work_cause_before_packaging(tmp_path):
    from oeis_learn.pipeline_validation.generalization_failures import stage_failures
    work=dict(status='FAIL',reason='QuotaDeficit',error='completed post-exclusion quota deficit; no timeout/reseed retry')
    record=seal(dict(version='generalization-command-result/v2',status='FAIL',result=dict(status='FAIL',reason='PACKAGE_FAILED',error='/private/export',work_result=work)))
    write(tmp_path/'result.json',record)
    details=stage_failures(tmp_path)
    assert [r['detail']['kind'] for r in details['failures']]==['QuotaDeficit','PACKAGE_FAILED']
    assert '/private/export' not in json.dumps(details)


def test_canonical_job_writes_negative_terminal_for_host_side_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import generalization_pac_commands as command
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.pipeline_validation.generalization_failures import stage_failures
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:{})
    job=seal(dict(version='pac-role-job/v1',operation='pac-assemble',engine='podman',images=[digest('cpu'),digest('gpu')],
        source=digest({}),artifacts={},source_roots={},settings={},envelope={'outer':3600},qualification=None,native_admission=None))
    write(tmp_path/'job.json',job)
    monkeypatch.setattr(command,'project_and_invoke',lambda *a,**kw:(_ for _ in ()).throw(ValueError('/private/native detail')))
    host=SimpleNamespace(engine=SimpleNamespace(name='podman',image=lambda image:image))
    assert command.main(SimpleNamespace(job='job.json',output='execution',engine='podman'),tmp_path,lambda args:host)==1
    value=json.loads((tmp_path/'execution/result.json').read_text())
    assert value['job']==job['identity'] and value['result']['failure']['kind']=='ValueError'
    details=stage_failures(tmp_path)
    assert details['failures'][0]['detail']['kind']=='ValueError' and '/private/native' not in json.dumps(details)


@pytest.mark.parametrize('message,expected_hash',[
    ('missing artifact','sha256:027455ec0582bfb3d196a67af27b1a6291c34414508aa68ee93b5573cdfaa318'),
    ('invalid frozen execution: missing artifact','sha256:531c31d1def9fd778f37fe34cf4e1dc18bd368beb8b1e56599d2f8694498724e'),
])
def test_reported_safe_literal_errors_are_readable_and_old_hashes_upgrade(tmp_path,message,expected_hash):
    from oeis_learn.pipeline_validation.generalization_failures import public_failure,stage_failures
    detail=public_failure(ValueError(message))
    assert detail['private_detail_sha256']==expected_hash
    assert detail['code']=='missing_artifact'
    # The original producer remains byte-identical; new recovery can recognize
    # only exact known safe constants, never guess an arbitrary private message.
    old=seal(dict(detail,code='private_exception_detail',message='Private exception detail omitted; inspect the original producer result.'))
    write(tmp_path/'role/result.json',seal(dict(status='FAIL',failure=old)))
    immutable=(tmp_path/'role/result.json').read_bytes()
    recovered=stage_failures(tmp_path)
    assert recovered['failures'][0]['detail']['code']=='missing_artifact'
    assert (tmp_path/'role/result.json').read_bytes()==immutable


@pytest.mark.parametrize('raw,truncated',[
    ('/private/host/teacher-source',False),
    ('missing artifact',True),
])
def test_old_unknown_or_truncated_detail_stays_private(tmp_path,raw,truncated):
    from oeis_learn.pipeline_validation.generalization_failures import public_failure,stage_failures
    old=seal(dict(public_failure(ValueError(raw)),code='private_exception_detail',
        message='Private exception detail omitted; inspect the original producer result.',
        private_detail_truncated=truncated))
    write(tmp_path/'role/result.json',seal(dict(status='FAIL',failure=old)))
    result=stage_failures(tmp_path)
    assert result['failures'][0]['detail']==old
    assert '/private/host' not in json.dumps(result)


def test_reported_calibration_missing_source_hash_recovers_readably(tmp_path):
    from oeis_learn.pipeline_validation.generalization_failures import public_failure,stage_failures
    detail=public_failure(KeyError('source'))
    assert detail['private_detail_sha256']=='sha256:a524ceb061775a34f3f255c6328fda28d2833b6fe33db8cff18db11fef01216d'
    assert detail['code']=='missing_source_field'
    old=seal(dict(detail,code='private_exception_detail',message='Private exception detail omitted; inspect the original producer result.'))
    write(tmp_path/'role/result.json',seal(dict(status='FAIL',failure=old)))
    original=(tmp_path/'role/result.json').read_bytes()
    recovered=stage_failures(tmp_path)
    assert recovered['failures'][0]['detail']['message']=='A required source field is missing.'
    assert (tmp_path/'role/result.json').read_bytes()==original
