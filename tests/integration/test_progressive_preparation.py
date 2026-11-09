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
    runtime=envelope('runtime',dict(scope=dict(source='sha256:'+'a'*64,images=['sha256:'+'a'*64,'sha256:'+'b'*64],model='sha256:'+'b'*64,language='bigint-body/v1',precision='fp32',device='cpu',execution_policy='sha256:'+'c'*64),qualification=[reference(tmp_path,'qualification.json')],operations=['census'],role_policy='sha256:'+'d'*64,fixture=True))
    atomic_write(tmp_path/'assembled-runtime.json',runtime)
    payload=dict(p['payload'],runtime=reference(tmp_path,'assembled-runtime.json'));payload.pop('blueprint')
    atomic_write(tmp_path/'plan.json',dict(version='progressive-preparation-plan/v1',payload=payload))
    b['payload']['references'].append(dict(name='preparation_plan',kind='prepared',role='custodian',artifact=reference(tmp_path,'plan.json')))
    b=seal(b);atomic_write(tmp_path/'blueprint.json',b)
    result,code=prepare(SimpleNamespace(blueprint='blueprint.json',qualification='.',output='prepared-out'),tmp_path)
    assert result['reason']=='BOUND_OUTCOME_BLIND_PREPARATION'
    assert load(tmp_path/'prepared-out/prepared.json')['payload']['blueprint']==b['identity']
    assert code==2 # original unresolved inputs remain explicit


def test_canonical_host_routes_native_owned_outputs_to_public_readiness(tmp_path,monkeypatch,fitting_client):
    from types import SimpleNamespace
    from tests.helpers.progressive_study import build
    from oeis_learn.pipeline_validation import progressive_commands as commands,progressive_study as study
    from oeis_learn.generalization.progressive.contracts import load
    c,p,host,_=build(tmp_path,monkeypatch,capability=True)
    token=commands._INVOCATION.set('native-capability')
    try:result,code=commands.lifecycle_command(SimpleNamespace(operation='run',campaign='study-campaign.json',acceptance='accepted',output='capability-run'),tmp_path,lambda _:host)
    finally:commands._INVOCATION.reset(token)
    assert code==2 # genuine parameter/OEIS support deficits remain explicit
    assert study.journal(tmp_path,c,p).trials['0']['status']=='COMPLETE'
    [readiness]=list((tmp_path/'capability-run').glob('task-*/role/readiness.json'))
    report=load(readiness)
    assert report['status']=='INCOMPLETE' and report['scientific_status']=='UNMEASURED'
    assert {'parameter','oeis'}<=set(report['missing'])
    assert report['strata']['composition']['representation']['supported']==16
    assert report['strata']['unseen_structure']['methods']['retained_D:equal_budget']['tasks']==16
    assert '(program' not in readiness.read_text()
    [decision]=list((tmp_path/'capability-run').glob('task-*/role/decision-bundle.json'))
    bundle=load(decision)
    assert bundle['decision']['payload']['evidence_verdict']=='INCOMPLETE'
    assert bundle['decision']['payload']['next_action']=='RETAIN_BASELINE'
    assert any(r['outcome']=='unresolved' for r in bundle['rows'])
    [coverage]=list((tmp_path/'capability-run').glob('task-*/role/coverage.json'))
    audits=load(coverage)
    assert len(audits)==32 and all(a['version']=='discovery-coverage/v1' for a in audits)
