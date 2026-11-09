"""Real projected custodian operation, with no model or outcomes in census."""

from tests.unit.test_progressive_cohorts import fixture
from tests.contract.test_progressive_roles import artifact
from oeis_learn.generalization.progressive.projections import project
from oeis_learn.pipeline_validation.progressive_roles import REGISTRY
from oeis_learn.pipeline_validation.progressive_role import execute


def test_projected_installed_census_writes_deficits_without_scoring(tmp_path):
    c, e = fixture(tmp_path)
    cohort = artifact(tmp_path, "cohort-input.json", "cohort", "custodian", c)
    exposure = artifact(tmp_path, "exposure-input.json", "exposure", "custodian", e)
    project(tmp_path, "inputs", "census", {"sources": [cohort, exposure]}, registry=REGISTRY)
    output = tmp_path / "out"
    output.mkdir()
    result = execute("census", inputs=tmp_path / "inputs", output=output)
    assert result["mechanical_status"] == "PARTIAL" and result["scientific_status"] == "UNMEASURED"
    assert set(result["missing_evidence"]) == {
        "parameter",
        "composition",
        "unseen_structure",
        "oeis",
    }
    assert (output / "census.json").is_file()


def test_prepare_assembles_original_prospective_inputs_without_engine(tmp_path):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import atomic_write,reference,load,seal,record
    from oeis_learn.pipeline_validation.progressive_capability import prepare
    from tests.helpers.progressive_fixture import blueprint,envelope
    from tests.unit.test_progressive_freeze import prepared
    b=blueprint();p=prepared(tmp_path,b)
    runtime=envelope('runtime',dict(scope=dict(source='sha256:'+'a'*64,images=['sha256:'+'a'*64],model='sha256:'+'b'*64,language='bigint-body/v1',precision='fp32',device='cpu',execution_policy='sha256:'+'c'*64),qualification=[reference(tmp_path,'qualification.json')],operations=['census'],role_policy='sha256:'+'d'*64,fixture=True))
    atomic_write(tmp_path/'assembled-runtime.json',runtime)
    payload=dict(p['payload'],runtime=reference(tmp_path,'assembled-runtime.json'));payload.pop('blueprint')
    atomic_write(tmp_path/'plan.json',dict(version='progressive-preparation-plan/v1',payload=payload))
    b['payload']['references'].append(dict(name='preparation_plan',kind='prepared',role='custodian',artifact=reference(tmp_path,'plan.json')))
    b=seal(b);atomic_write(tmp_path/'blueprint.json',b)
    result,code=prepare(SimpleNamespace(blueprint='blueprint.json',qualification='.',output='prepared-out'),tmp_path)
    assert result['reason']=='BOUND_OUTCOME_BLIND_PREPARATION'
    assert load(tmp_path/'prepared-out/prepared.json')['payload']['blueprint']==b['identity']
    assert code==2 # original unresolved inputs remain explicit
