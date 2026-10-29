"""Regression coverage for the post-pilot repairs."""
import pytest


def test_template_roundtrip_preserves_explicit_parameter_policy():
    from oeis_learn.decoder.bigint_codec import encode_body, decode_body, fitting_policy
    from oeis_learn.fresh.template_baseline import template_bank
    for source in template_bank()['sources']:
        assert fitting_policy(source) == fitting_policy(decode_body(encode_body(source))) == 'none'
    assert fitting_policy('(program (return (int 3)))') == 'eligible-literals/v1'
    with pytest.raises(ValueError):
        fitting_policy('(program invalid)')
