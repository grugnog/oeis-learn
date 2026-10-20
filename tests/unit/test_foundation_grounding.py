"""Shared identity, exact range, false-square and visible-only regressions."""
from tests.helpers.native_fitting import fit
from oeis_learn.synthesis.fitting import fit_program


def test_nonlinear_parameter_local_square(fitting_client):
    source='(program (return (mul (param.get p0) (param.get p0))))'
    result=fit(fitting_client,source,[4]*20)
    assert result.outcome == 'verified_solution' and result.constants[0] in (-2,2)
    assert result.source != '(program (return (int 16)))'


def test_large_exact_signed_terms(fitting_client):
    target=-(2**300+13)
    result=fit(fitting_client,'(program (return (add (param.get p0) (int 7))))',[target]*20)
    assert result.outcome == 'verified_solution' and result.constants == (target-7,)


def test_tail_is_not_fitting_data(fitting_client):
    source='(program (return (param.get p0)))'
    a=fit(fitting_client,source,[5]*20+[7]*100)
    b=fit(fitting_client,source,[5]*20+[9]*100)
    assert a.outcome == b.outcome == 'verified_solution'
    assert a.constants == b.constants == (5,)
    assert a.prompt == b.prompt and a.prompt.indices == tuple(map(str,range(20)))


def test_empty_parameter_domain_is_verified(fitting_client):
    source='(program (return (int 1)))'
    assert fit(fitting_client,source,[2]*20).outcome == 'unsat_in_scope'
    assert fit(fitting_client,source,[1]*20).outcome == 'verified_solution'


def test_control_and_invalid_signatures_are_not_unsat():
    result=fit_program('(module (func (result i32) i64.const_?))',[1]*20)
    assert result.outcome == 'unsupported'
