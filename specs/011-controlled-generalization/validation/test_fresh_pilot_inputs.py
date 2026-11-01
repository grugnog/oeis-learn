import importlib.util
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fresh_pilot_inputs',HERE/'fresh_pilot_inputs.py')
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)


@pytest.fixture(scope='module')
def design():return module.research_inputs()


def test_registered_research_scale_and_fixed_mixture(design):
    assert design['seed']==20261016
    assert len(design['descriptors'])==32
    assert {d['stage'] for d in design['descriptors']}==set('ABCD')
    assert len(design['training'])==12
    expected={'family:'+d['family']:1024 for d in design['descriptors']}
    cursors=[]
    for c in design['training']:
        assert c['count']==32768 and c['quotas']==expected
        assert c['refresh']['draws']==65536 and c['refresh']['capacity']==1_000_000
        cursors.append(c['refresh']['cursor'])
    assert cursors==[65536*i for i in range(12)]
    assert design['token_capacity_floor']>=25_000_000
    assert len(design['training'])+len(design['cohorts'])==16


def test_role_support_is_disjoint_except_intended_parameter_holdout(design):
    training={d['family'] for d in design['descriptors']}
    held=[]
    for cell in design['cohorts']:
        assert cell['count']==128
        assert cell['role'] in {'selection','assessment'}
        assert len({r['source'] for r in cell['rows']})==len(cell['rows'])
        assert all(r['origin']=='generic_random' for r in cell['rows'])
        families={r['family'] for r in cell['rows']}
        if cell['holdout']=='parameter':assert families<=training
        else:
            assert families.isdisjoint(training)
            assert all(families.isdisjoint(old) for old in held)
            held.append(families)
            assert cell['structural']
            assert cell['quotas']['loop_active']>=2
            assert cell['quotas']['branch_active']>=2
    assert len(held)==2


def test_blueprint_is_fresh_and_cannot_consume_imported_weights(design,tmp_path,monkeypatch):
    import shutil
    monkeypatch.setattr(shutil,"disk_usage",lambda *_:type("Disk",(),dict(free=128<<30))())
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_inputs import assemble_study
    from tests.integration.test_generalization_inputs import blueprint
    original=blueprint(tmp_path)
    write(tmp_path/'feature.json',{'fixture':'not qualification'})
    cfg=module.blueprint(tmp_path,design,recipe=original['recipe'],source=original['source_closure'],
        images=original['images'],feature='feature.json',output='research-design')
    assert cfg['initializer'] is None and cfg['purpose']=='research' and cfg['stage']=='P1-screen'
    assert cfg['training']['static']['cells']==['static']
    assert cfg['training']['refreshed']['cells']==[c['name'] for c in design['training']]
    assert cfg['evaluation'] is None and cfg['replay'] is None
    assert cfg['budgets']['target_tokens']==25_000_000
    result=assemble_study(tmp_path,cfg,'assembled')
    import json
    study=json.loads((tmp_path/result['manifest']).read_text())
    assert study['model']['initialization']=='random'
    assert study['seeds']==[20261016] and study['arms']==['static','refreshed']

    from oeis_learn.pipeline_validation.generalization_runtime import load,validate_execution
    base,study,runtime=load(tmp_path,result['manifest'])
    assert validate_execution(base,study,runtime)['preparation']==runtime['preparation']
