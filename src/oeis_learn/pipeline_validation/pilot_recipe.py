"""Opt-in longer strict recipe, never a change to existing v1/v2 defaults."""
from .pilot_contract import SPEC,PROFILE

PIPELINE=dict(version='pipeline-policy/v1',selection='visible-only/v2',archive='structure-pareto/v2',
    workers=4,overlap=True,lookahead=1,batch_size=16,bookkeeping='hoisted/v1',measurement='ordinary')


def recipe():
    from oeis_learn.fresh.bigint_config import resolve
    from .model_sizes import PROFILES
    return resolve(dict(version='bigint-bootstrap/v3',model_profile=PROFILES['m100'],pilot_profile=PROFILE,
        pipeline=PIPELINE))


def validate_effective(value):
    from oeis_learn.fresh.bigint_config import defaults
    from .model_sizes import PROFILES
    expected=defaults();expected['version']='bigint-bootstrap/v3'
    expected['data'].update(training=32768,max_families=16384)
    expected['model'].update(d_model=768,heads=12,encoder_layers=6,decoder_layers=6,ff=3072)
    expected['learning'].update(batch_size=16,max_updates=SPEC['max_updates'])
    expected['budgets'].update(training_seconds=86400,preparation_seconds=9000)
    expected.update(pipeline=PIPELINE,model_profile=PROFILES['m100'],pilot_profile=PROFILE)
    if {k:v for k,v in value.items() if k!='identity'}!=expected:
        raise ValueError('exact frozen pilot recipe required; changed data/model/solver/seed policy is a new prospective profile')
    if (value['pilot_profile']!=PROFILE or value['data']['training']!=32768 or value['data']['max_families']!=16384
        or value['learning']['batch_size']!=16 or value['learning']['max_updates']!=SPEC['max_updates']
        or value['budgets']['training_seconds']!=86400 or value['budgets']['preparation_seconds']!=9000
        or value.get('pipeline')!=PIPELINE or value['learning']['target_view']!='literal' or value['fitting']['disabled']
        or value['track']!={'kind':'generic_only','foundation':None,'assisted_manifest':None}
        or value['learning']['curriculum'] or value['learning']['replay']):
        raise ValueError('frozen strict learning-pilot recipe required')
