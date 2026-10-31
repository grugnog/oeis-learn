"""Independent counterexamples from the reconstruction requirements audit."""
import copy
import pytest
from oeis_learn.generalization.selection import select
from oeis_learn.generalization.targets import pair


def test_selection_cannot_rank_different_frozen_denominators():
    a=dict(id='a',tokens=1,role='selection',structural=({'s':'f'},{'s':True}),parameter=({'p':'g'},{'p':True}))
    b=copy.deepcopy(a);b['id']='b';b['structural']=({'other':'f'},{'other':True})
    with pytest.raises(ValueError,match='same frozen'):
        select([a,b])


def test_selection_cannot_change_family_weighting_between_candidates():
    a=dict(id='a',tokens=1,role='selection',structural=({'s':'f','t':'f'},{'s':True,'t':False}),parameter=({'p':'g'},{'p':True}))
    b=copy.deepcopy(a);b['id']='b';b['structural'][0]['t']='other'
    with pytest.raises(ValueError,match='same frozen'):
        select([a,b])


def test_unsupported_control_target_stays_literal_and_counted():
    from oeis_learn.generalization.targets import learning_target
    literal='(program (return (select (lt $index (int 2)) (int 1) (int 0))))'
    parameter='(program (return (select (lt $index (param.get p0)) (int 1) (int 0))))'
    proof=pair(literal,parameter,{'p0':'2'},indices=tuple(map(str,range(100))),sharing='explicit')
    assert proof['disposition']=='unsupported_control_dependency'
    assert learning_target(proof)==(literal,proof['literal_tokens'])


def test_supported_parameter_target_is_preserved():
    from oeis_learn.generalization.targets import learning_target
    literal='(program (return (add $index (int 2))))'
    parameter='(program (return (add $index (param.get p0))))'
    proof=pair(literal,parameter,{'p0':'2'},indices=tuple(map(str,range(100))),sharing='explicit')
    assert learning_target(proof)==(parameter,proof['parameter_tokens'])
