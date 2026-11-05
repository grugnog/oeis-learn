import json
from pathlib import Path
from oeis_learn.generalization.experiments import STAGES,recipe
from oeis_learn.pipeline_validation.identity import seal


def test_recipe_templates_are_frozen_single_factor_and_unresolved():
    for stage in STAGES:
        path=Path('configs/foundation/pipeline_studies/generalization-'+stage+'.json')
        value=json.loads(path.read_text())
        assert seal(value)==value and value['policy']==json.loads(json.dumps(recipe(stage)))
        assert value['changing_factors']==[recipe(stage)['changing_factor']]
        assert value['status']=='PREPARATION_REQUIRED' and not value['automatic_run_authorized']
        assert value['selection_role']!=value['assessment_role']
        assert value['inference']==dict(visible_terms=20,total_terms=100,attempts=8,seconds=60)
    p4=json.loads(Path('configs/foundation/pipeline_studies/generalization-P4.json').read_text())
    assert p4['policy']['milestones']==[1000000,2000000,4000000]


def pac_envelope():
    from oeis_learn.pipeline_validation.generalization_execution import POLICY_ID
    stages={name:dict(work=work,startup=90,publication=120,projection=60,cleanup=30,outer=work+300)
        for name,work in [('validation',600),('preparation',90000),('learner',50000),('generation',25000),
                          ('acceptance',3600),('calibration',1800),('resume',50000),('proof',3600),('export',3600)]}
    return dict(version='pac-envelope/v1',**stages,campaign=sum(s['outer'] for s in stages.values())+3600,
        diagnostics_seconds=1000,batch_size=32,acceptance_tokens=4096,backend=POLICY_ID,
        storage=dict(data_bytes=1000000,data_copies=3,scratch_bytes=1000000,logs_bytes=1000000,package_bytes=1000000,reserved_bytes=32<<30))


def test_pac_all_nested_limits_preserve_longer_reservations_and_reject_clipping():
    import copy,pytest
    from oeis_learn.pipeline_validation.generalization_execution import pac_envelope_graph
    env=pac_envelope();graph=pac_envelope_graph(env,purpose='research')
    assert graph['seconds']==env['campaign']
    for name in ('acceptance','calibration','resume','proof','export','learner','generation','preparation','validation'):
        assert graph['nodes'][name]['effective']==env[name]['outer']
        bad=copy.deepcopy(env);bad[name]['outer']-=1
        with pytest.raises(ValueError):pac_envelope_graph(bad,purpose='research')
    for key in ('work','publication','startup'):
        bad=copy.deepcopy(env);bad['learner'][key]=float('inf')
        with pytest.raises(ValueError):pac_envelope_graph(bad,purpose='research')
    bad=copy.deepcopy(env);bad['backend']='sha256:'+'0'*64
    with pytest.raises(ValueError,match='backend'):pac_envelope_graph(bad,purpose='research')
    bad=copy.deepcopy(env);bad['generation']['work']=21600
    with pytest.raises(ValueError,match='floor'):pac_envelope_graph(bad,purpose='research')
