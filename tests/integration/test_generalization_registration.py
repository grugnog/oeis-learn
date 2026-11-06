from pathlib import Path
from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.pipeline_validation.closures import dependency_closure,compatible_changes


def test_contracts_and_adapters_are_authority_but_recipes_independent():
    root=Path(__file__).resolve().parents[2]
    snapshot=collect_snapshot(root)
    authority='specs/011-controlled-generalization/contracts/artifacts.md'
    assert authority in snapshot
    assert 'src/oeis_learn/pipeline_validation/generalization_preparation.py' in snapshot
    assert not any(k.startswith('configs/foundation/pipeline_studies/') for k in snapshot)
    changed=dict(snapshot);changed[authority]='sha256:'+'0'*64
    assert not compatible_changes(dependency_closure(snapshot,prerequisites={}),dependency_closure(changed,prerequisites={}),[authority])
    assert not compatible_changes(dependency_closure(snapshot,prerequisites={}),dependency_closure(snapshot,prerequisites={}),['unmapped.py'])


def test_new_roles_checkpoints_replay_helper_and_tests_are_in_source_closure():
    root=Path(__file__).resolve().parents[2]
    snapshot=collect_snapshot(root)
    for path in (
        'src/oeis_learn/pipeline_validation/generalization_role.py',
        'src/oeis_learn/pipeline_validation/generalization_runtime.py',
        'src/oeis_learn/pipeline_validation/generalization_session.py',
        'src/oeis_learn/pipeline_validation/generalization_selection_io.py',
        'src/oeis_learn/generalization/acceptance_archive.py',
        'src/oeis_learn/pipeline_validation/generalization_scoring.py',
        'src/oeis_learn/pipeline_validation/generalization_replay.py',
        'src/oeis_learn/fresh/bigint_state.py',
        'scripts/generalization_hip_serial.py',
        'src/oeis_learn/pipeline_validation/generalization_diagnostics.py',
        'tests/integration/test_generalization_diagnostics.py',
        'scripts/generalization_inputs.py',
        'src/oeis_learn/generalization/execution.py',
        'src/oeis_learn/generalization/legacy.py',
        'src/oeis_learn/pipeline_validation/generalization_inputs.py',
        'src/oeis_learn/generalization/bootstrap.py',
        'src/oeis_learn/generalization/acceptance_inputs.py',
        'src/oeis_learn/pipeline_validation/generalization_bootstrap.py',
        'src/oeis_learn/pipeline_validation/generalization_setup.py',
        'tests/unit/test_generalization_bootstrap.py',
        'tests/unit/test_generalization_acceptance_inputs.py',
        'tests/integration/test_generalization_setup.py',
        'tests/unit/test_generalization_convergence.py',
        'tests/unit/test_generalization_legacy.py',
        'tests/integration/test_generalization_inputs.py',
        'tests/integration/test_generalization_gpu.py',
        'tests/integration/test_generalization_training.py',
        'tests/integration/test_generalization_scoring.py',
    ):
        assert path in snapshot
        changed=dict(snapshot);changed[path]='sha256:'+'0'*64
        assert not compatible_changes(dependency_closure(snapshot,prerequisites={}),
            dependency_closure(changed,prerequisites={}),[path])


def test_pac_has_separate_canonical_dispatch_and_closed_binding(tmp_path):
    import copy, json
    import pytest
    from tests.unit.test_generalization_execution import pac_setup
    from oeis_learn.pipeline_validation.identity import seal, digest
    from oeis_learn.pipeline_validation.generalization_execution import bind_pac, verify_pac_runtime
    from oeis_learn.pipeline_validation.generalization_runtime import load
    experiment, control, profiles, _ = pac_setup()
    from tests.integration.test_generalization_recipes import pac_envelope
    from oeis_learn.pipeline_validation.generalization_execution import pac_budget_projection,POLICY_ID
    experiment=seal(dict(experiment,backend=POLICY_ID,budget_graph=pac_budget_projection(pac_envelope())))
    runtime = dict(version='pac-runtime/v1', study=experiment['identity'], purpose='research',
        recipe={'path':'recipe.json','bytes':2,'sha256':digest({})}, images=[digest('cpu'),digest('gpu')],
        source={'path':'source.json','bytes':2,'sha256':digest({})}, control=control, profiles=profiles,
        preparation=None, prepared=None, envelope=pac_envelope(), feature=None, engine='podman')
    experiment, runtime = bind_pac(experiment,runtime)
    verify_pac_runtime(experiment,runtime)
    (tmp_path/'study.json').write_text(json.dumps(experiment));(tmp_path/'runtime.json').write_text(json.dumps(runtime))
    _, actual, sidecar = load(tmp_path,'study.json')
    assert actual==experiment and sidecar==runtime
    for field,value in [('images',[digest('other'),digest('gpu')]),('engine','docker'),('hidden_truth',[])]:
        changed=seal(dict(runtime,**{field:value}))
        with pytest.raises(ValueError):verify_pac_runtime(experiment,changed)
    changed=copy.deepcopy(experiment);changed['version']='generalization-study/v3'
    with pytest.raises(ValueError):verify_pac_runtime(seal(changed),runtime)


def test_pac_model_operations_require_original_admission_before_launch():
    import pytest
    from oeis_learn.pipeline_validation.generalization_execution import require_pac_admission
    for operation in ('train','evaluate','replay'):
        with pytest.raises(ValueError,match='admission|unsupported'):
            require_pac_admission({},dict(purpose='research',prepared=None),operation)


def test_pac_acceptance_routes_through_canonical_host_not_legacy_session(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import generalization_cli as cli,generalization_commands as commands
    (tmp_path/'study.json').write_text(json.dumps(dict(version='PAC-screen/v1')))
    calls=[]
    monkeypatch.setattr(commands,'main',lambda *a:calls.append(a) or 0)
    args=SimpleNamespace(manifest='study.json',operation='acceptance')
    assert cli.main(args,tmp_path,None)==0 and len(calls)==1


def test_pac_source_closure_checks_installed_bytes_before_preparation(tmp_path):
    import pytest
    from tests.unit.test_generalization_execution import pac_setup
    from tests.integration.test_generalization_recipes import pac_envelope
    from oeis_learn.pipeline_validation.generalization_execution import bind_pac,validate_pac_execution,pac_budget_projection,POLICY_ID
    from oeis_learn.pipeline_validation.identity import digest,write,seal
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.fresh.bigint_config import resolve
    exp,control,profiles,_=pac_setup()
    recipe=resolve(dict(version='bigint-bootstrap/v1'))
    write(tmp_path/'recipe.json',recipe);write(tmp_path/'source.json',{'stale.py':digest('stale')})
    exp.update(model=digest({k:v for k,v in recipe['model'].items() if k!='seed'}),optimizer=digest(recipe['learning']),
               source=digest({'stale.py':digest('stale')}),backend=POLICY_ID,budget_graph=pac_budget_projection(pac_envelope()))
    for profile in profiles.values():
        for k in ('model','optimizer','backend'):profile[k]=exp[k]
    runtime=dict(version='pac-runtime/v1',study='',purpose='acceptance',recipe=reference(tmp_path/'recipe.json',tmp_path),
        source=reference(tmp_path/'source.json',tmp_path),images=[digest('cpu'),digest('gpu')],control=control,profiles=profiles,
        preparation=None,prepared=None,envelope=pac_envelope(),feature=None,engine='podman')
    exp,runtime=bind_pac(seal(exp),runtime)
    with pytest.raises(ValueError,match='actual installed source'):
        validate_pac_execution(tmp_path,exp,runtime)


def test_all_pac_handlers_and_delivery_helpers_are_source_bound():
    from oeis_learn.pipeline_validation.generalization_role import HANDLERS
    from oeis_learn.generalization.role_contracts import Role
    expected={('preparation','pac-assemble'),('learner','pac-calibrate'),('learner','pac-campaign-acceptance'),
              ('preparation','pac-publish-evaluation'),('generator','pac-evaluate'),('selection','pac-score')}
    assert {(Role(role),operation) for role,operation in expected}<=set(HANDLERS)
    snapshot=collect_snapshot(Path(__file__).resolve().parents[2])
    for path in [*['src/oeis_learn/pipeline_validation/generalization_pac_'+name+'.py' for name in ['commands','setup','evaluation','calibration','export']]]:
        assert path in snapshot


def test_helper_only_changes_have_separate_complete_identity(tmp_path):
    from oeis_learn.pipeline_validation.generalization_pac_export import helper_identity
    from oeis_learn.pipeline_validation.identity import digest
    folder=tmp_path/'specs/012-parameterized-adaptive-curriculum/validation';folder.mkdir(parents=True)
    (folder/'run.py').write_text('old helper')
    original=helper_identity(tmp_path)
    (folder/'analysis.py').write_text('new analysis')
    assert helper_identity(tmp_path)!=original
    snapshot=collect_snapshot(Path(__file__).resolve().parents[2])
    assert 'specs/012-parameterized-adaptive-curriculum/validation/run.py' not in snapshot
    assert 'specs/012-parameterized-adaptive-curriculum/validation/analysis.py' not in snapshot
