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
