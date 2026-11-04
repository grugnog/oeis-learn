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


def test_opt_in_uniform_roster_preserves_historical_default():
    from oeis_learn.pipeline_validation.followup_study import requested_arms
    assert requested_arms(None)==('uniform','family_balanced','decay')
    assert requested_arms(['uniform'])==('uniform',)
    for value in ([],['decay'],['uniform','uniform']):
        with pytest.raises(ValueError):requested_arms(value)


def test_uniform_scope_registers_only_one_training_stage(capsys):
    from types import SimpleNamespace
    from pathlib import Path
    import json
    from oeis_learn.pipeline_validation.followup_study import main
    args=SimpleNamespace(engine='docker',tokens=25000000,seed=20261017,arms=['uniform'],qualification='proof',scope_report=True)
    assert main(args,Path('.'),None)==0
    result=json.loads(capsys.readouterr().out)
    assert [s['id'] for s in result['stages'] if s['adapter']=='followup-train']==['followup.uniform']
    assert sum(s['adapter']=='generalization-qualification' for s in result['stages'])==1
    assert not any(s['adapter'] in {'cpu','hip'} for s in result['stages'])
