"""Recurrence-equation and exact-rank regressions against native integer fitting."""
import pytest
from tests.helpers.native_fitting import fit,equations


@pytest.mark.parametrize('matrix,rhs',[
 ([[1,0],[1,1],[2,1],[3,2]], [1,2,3,5]),
 ([[1,0],[2,1],[5,2],[12,5]], [2,5,12,29]),
 ([[1,0],[3,1],[7,3]], [3,7,15]),
 ([[1,0,0],[1,1,0],[2,1,1],[4,2,1]], [1,2,4,7]),
 ([[1,1]], [3]),
 ([[0,1],[0,2]], [2,4]),
 ([[2305843009213693951]], [2305843009213693951*7]),
 ([[2**80+1]], [-9*(2**80+1)]),
])
def test_exact_equations_and_singular_modular_pivots(fitting_client,matrix,rhs):
    result=fit(fitting_client,equations(matrix),rhs)
    assert result.outcome == 'verified_solution',result
    assert all(sum(a*c for a,c in zip(row,result.constants))==b for row,b in zip(matrix,rhs))


@pytest.mark.parametrize('matrix,rhs', [([[2]],[1]), ([[1],[1]],[1,2]), ([[0],[0]],[1,2])])
def test_nonintegral_or_inconsistent_system_never_admitted(fitting_client,matrix,rhs):
    result=fit(fitting_client,equations(matrix),rhs)
    assert result.outcome == 'unsat_in_scope' and not result.response['admitted']
