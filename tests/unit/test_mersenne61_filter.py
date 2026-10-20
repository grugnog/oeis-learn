"""Modular consistency must never stand in for exact integer admission."""
from tests.helpers.native_fitting import fit,equations


def test_modular_consistency_cannot_admit_noninteger_solution(fitting_client):
    # 2*x=1 has modular solutions for the old odd prime, no integer solution.
    result=fit(fitting_client,equations([[2]]),[1])
    assert result.outcome == 'unsat_in_scope' and not result.response['admitted']


def test_prime_multiple_pivot_is_not_a_singular_integer_system(fitting_client):
    p=(1<<61)-1
    result=fit(fitting_client,equations([[p]]),[-p*9])
    assert result.outcome == 'verified_solution' and result.constants == (-9,)
