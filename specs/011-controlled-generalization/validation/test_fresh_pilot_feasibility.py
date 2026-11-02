import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fresh_pilot_feasibility import check,verify_prepared
from oeis_learn.pipeline_validation.identity import seal


def design():
    from tests.integration.test_generalization_preparation import source
    training=[source(1),source(2)]
    cell=dict(name='static',role='training',holdout='parameter',count=2,quotas={},structural=False,rows=training)
    return seal(dict(cohorts=[],training=[cell],target_tokens=1))


def test_full_pass_records_actual_capacity_and_rejects_missing_native_selection(tmp_path):
    result=check(design(),tmp_path/'pass',workers=1)
    assert result['status']=='FEASIBLE' and result['training_tokens']>1
    assert result['native_validation']==result['hardware_validation']=='NOT_RUN'
    assert result['cells'][0]['selected']==2
    with pytest.raises(FileNotFoundError):verify_prepared(result,tmp_path/'missing')
    modified=dict(result,training_tokens=999)
    with pytest.raises(ValueError):verify_prepared(modified,tmp_path/'missing')


def test_later_shard_failure_cannot_promote_earlier_cells(tmp_path):
    d=design();d['training'].append(dict(d['training'][0],name='later',count=3))
    result=check(seal(d),tmp_path/'failed',workers=1)
    assert result['status']=='BLOCKED' and result['reason']=='QUOTA_SHORTFALL'
    assert [c['status'] for c in result['cells']]==['FEASIBLE','BLOCKED']
    assert result['cells'][1]['shortfalls']==['rows:1']
    assert json.loads((tmp_path/'failed/progress.json').read_text())['cells']==result['cells']


def test_token_capacity_and_expired_budget_fail_closed(tmp_path):
    d=design();d['target_tokens']=1_000_000
    assert check(seal(d),tmp_path/'tokens',workers=1)['reason']=='TOKEN_CAPACITY_SHORTFALL'
    assert check(design(),tmp_path/'clock',workers=1,seconds=.000001)['status']=='BLOCKED'


def test_fixed_arithmetic_segments_rejected_before_roster_selection():
    from fresh_pilot_inputs import diverse,NUMERIC_LAW
    from oeis_learn.data.bigint_programs import canonical_family
    source='(program (return (add (add (add $index (param.get p0)) (param.get p1)) (param.get p2))))'
    descriptor=seal(dict(version='generalization-descriptor/v1',generator='test',stage='A',
        family=canonical_family(source),parameter_source=source,numeric_law=NUMERIC_LAW,parameter_names=['p0','p1','p2']))
    assert not diverse(descriptor)
