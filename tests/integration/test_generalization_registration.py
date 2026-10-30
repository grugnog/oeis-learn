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
