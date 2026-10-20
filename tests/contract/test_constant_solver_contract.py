"""Historical solver contracts remain readable and never grant replay admission."""
import json
from pathlib import Path
from oeis_learn.synthesis.historical import inspect_record


def test_constant_solver_schema_contract():
    schema=json.loads(Path('specs/004-decoupled-grounding-and-symple-engine/contracts/constant-solver.contract.json').read_text())
    assert schema['title']=='ConstantSolverContract'
    assert {'SolverRequest','SolverResponse'} <= schema['definitions'].keys()


def test_historical_result_is_inspection_only_and_not_mutated():
    original=dict(solver_type='DIOPHANTINE_HNF',constants=[2,5],is_sat=True,grounded_wat='old source')
    view=inspect_record(original)
    assert view['status']=='historical_unreplayed' and view['original']==original
    view['original']['constants'].append(9)
    assert original['constants']==[2,5]
