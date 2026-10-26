"""Opt-in longer strict recipe, never a change to existing v1/v2 defaults."""
from .pilot_contract import SPEC,PROFILE


def recipe():
    from oeis_learn.fresh.bigint_config import resolve
    from .model_sizes import PROFILES
    return resolve(dict(version='bigint-bootstrap/v3',model_profile=PROFILES['m100'],pilot_profile=PROFILE,
        pipeline=dict(version='pipeline-policy/v1',selection='visible-only/v2',archive='structure-pareto/v2',
            workers=4,overlap=True,lookahead=1,batch_size=16,bookkeeping='hoisted/v1',measurement='ordinary/v1')))


def validate_effective(value):
    if (value['pilot_profile']!=PROFILE or value['data']['training']!=32768 or value['data']['max_families']!=16384
        or value['learning']['batch_size']!=16 or value['learning']['max_updates']!=SPEC['max_updates']
        or value['budgets']['training_seconds']!=86400 or value['budgets']['preparation_seconds']!=9000
        or value['track']!={'kind':'generic_only','foundation':None,'assisted_manifest':None}
        or value['learning']['curriculum'] or value['learning']['replay']):
        raise ValueError('frozen strict learning-pilot recipe required')
