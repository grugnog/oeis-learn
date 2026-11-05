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
