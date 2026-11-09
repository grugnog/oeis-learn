"""A proposal is not a reservation or permission to change a failed result."""
import copy
import json
from pathlib import Path
import pytest
from oeis_learn.generalization.progressive import diagnostic_blueprints as db
from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
from tests.helpers.progressive_fixture import caps

ROOT = Path(__file__).resolve().parents[2]


def inventory():
    return json.loads((ROOT / 'configs/foundation/progressive/diagnostics.json').read_text())


def concrete(tmp_path, name='seed-disagreement'):
    book = inventory()
    branch = next(b for b in book['branches'] if b['id'] == name)
    branch['selected'] = True
    atomic_write(tmp_path / 'data.json', seal({'version': 'diagnostic-fixture/v1'}))
    ref = reference(tmp_path, 'data.json')
    branch.update(data=[ref], policy=ref, design=ref, panel_slots=['independent'],
                  task_slots=['paired-a', 'paired-b'], caps=caps(), ancillary_caps=caps())
    return book, branch


def test_inventory_covers_every_contract_row_without_executable_defaults():
    book = inventory()
    assert len(book['branches']) == 16
    assert set().union(*(set(b['tasks']) for b in book['branches'])) >= {'T093','T094','T098','T100','T101','T102','T104','T106','T108','T109','T110'}
    result = db.resolve(ROOT, book, observed=[], visited=[], pool=caps())
    assert not result['executable'] and all(b['status'] == 'DISABLED' for b in result['branches'])
    for b in book['branches']:
        assert b['trigger'] and b['question'] and len(b['arms']) >= 2 and b['observations']
        assert b['maximum_visits'] == 1 and b['terminal_action'] in {'RETAIN_BASELINE','DEFER_BRANCH','STOP_AFFECTED'}
        assert b['data'] is None and b['caps'] is None


@pytest.mark.parametrize('state', ['ready','disabled','untriggered','exhausted','unresolved'])
def test_selected_branch_state_is_explicit(tmp_path, state):
    book, branch = concrete(tmp_path)
    observed, visited = [branch['trigger']], []
    if state == 'disabled': branch['selected'] = False
    if state == 'untriggered': observed = []
    if state == 'exhausted': visited = [branch['failure_class']]
    if state == 'unresolved': branch['policy'] = None
    result = db.resolve(tmp_path, book, observed=observed, visited=visited, pool=caps(**{k: v*4 for k,v in caps().items()}))
    row = next(b for b in result['branches'] if b['id'] == branch['id'])
    assert row['status'] == {'ready':'READY','disabled':'DISABLED','untriggered':'UNTRIGGERED','exhausted':'EXHAUSTED','unresolved':'UNRESOLVED'}[state]
    assert result['executable'] is False  # preflight never issues campaign authority


@pytest.mark.parametrize('change', ['assessment','loss','visits','budget','bytes','slot','unknown'])
def test_forbidden_or_unfunded_branch_cannot_become_ready(tmp_path, change):
    book, b = concrete(tmp_path)
    pool = caps(**{k:v*4 for k,v in caps().items()})
    if change == 'assessment': b['assessment'] = True
    if change == 'loss': b['trigger'] = 'training_loss'
    if change == 'visits': b['maximum_visits'] = 2
    if change == 'budget': pool = caps(elapsed_us=0)
    if change == 'bytes': (tmp_path/'data.json').write_text('{}')
    if change == 'slot': b['task_slots'] = ['same','same']
    if change == 'unknown': b['extra_permission'] = True
    with pytest.raises(ValueError):
        db.resolve(tmp_path, book, observed=[b['trigger']], visited=[], pool=pool)


def test_integrity_failure_stops_affected_lineage_without_pool_or_training(tmp_path):
    book = inventory()
    b = next(b for b in book['branches'] if b['id'] == 'integrity-quarantine')
    b['selected'] = True
    result = db.resolve(tmp_path, book, observed=[b['trigger']], visited=[], pool=caps(elapsed_us=0))
    assert next(r for r in result['branches'] if r['id']==b['id'])['status'] == 'STOP_AFFECTED'
    assert not result['executable']
