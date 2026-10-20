"""Original polynomial/ambiguity counterexamples through the one native service."""
from tests.helpers.native_fitting import fit


def test_linear_affine_diophantine_grounding(fitting_client):
    result = fit(fitting_client, '(program (return (add (mul (param.get p0) $index) (param.get p1))))', [5*n+2 for n in range(20)])
    assert result.outcome == 'verified_solution'
    assert result.constants == (5,2)


def test_quadratic_polynomial_diophantine_grounding(fitting_client):
    source = '(program (return (add (add (mul (param.get p0) (mul $index $index)) (mul (param.get p1) $index)) (param.get p2))))'
    result = fit(fitting_client, source, [3*n*n-4*n+7 for n in range(20)])
    assert result.outcome == 'verified_solution' and result.constants == (3,-4,7)


def test_underdetermined_minimal_l1_norm(fitting_client):
    result = fit(fitting_client, '(program (return (add (param.get p0) (param.get p1))))', [3]*20)
    assert result.outcome == 'verified_solution'
    assert sum(result.constants) == 3
    # The frozen service ranks the discovered set, not a claimed global optimum.
    assert result.response['native']['uniqueness'] in {'multiple','incomplete','unchecked'}


def test_inconsistent_linear_diophantine(fitting_client):
    result = fit(fitting_client, '(program (return (add (mul (param.get p0) $index) (param.get p1))))', [1<<n for n in range(20)])
    assert result.outcome == 'unsat_in_scope' and not result.response['admitted']
