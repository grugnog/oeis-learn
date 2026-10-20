"""Native visible fitting, independent admission and scoped replay integration."""
from copy import deepcopy
from tests.helpers.native_fitting import fit
from oeis_learn.experiments.qualified_models import replay_record,seal_record


def test_linear_fitting_independent_execution_and_replay(fitting_client):
    source='(program (return (add (mul (param.get p0) $index) (param.get p1))))'
    result=fit(fitting_client,source,[5*n+2 for n in range(20)])
    assert result.outcome == 'verified_solution' and result.constants == (5,2)
    assert replay_record(result.to_qualified())['status'] == 'verified'


def test_inconsistent_fitting_scoped_replay_rejects_tampering(fitting_client):
    source='(program (return (mul (int 2) (param.get p0))))'
    result=fit(fitting_client,source,[1]*20)
    assert result.outcome == 'unsat_in_scope'
    record=result.to_qualified()
    assert replay_record(record)['status'] == 'verified'
    bad=deepcopy(record)
    bad['result']['native']['evidence']['constraints_sha256']='sha256:'+'0'*64
    bad.pop('kind');bad.pop('identity_sha256');bad.pop('schema_version')
    assert replay_record(seal_record('native_grounding_result',**bad))['status'] == 'unknown'
