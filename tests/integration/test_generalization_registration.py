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
        'tests/integration/test_generalization_gpu.py',
        'tests/integration/test_generalization_training.py',
        'tests/integration/test_generalization_scoring.py',
    ):
        assert path in snapshot
        changed=dict(snapshot);changed[path]='sha256:'+'0'*64
        assert not compatible_changes(dependency_closure(snapshot,prerequisites={}),
            dependency_closure(changed,prerequisites={}),[path])
