import pytest
from oeis_learn.generalization.exclusions import ExclusionIndex

def row(id,role,family,values,kind='parameter'):
    return dict(id=id,role=role,family=family,holdout=kind,source=id,values=tuple(map(str,values)),group=id,variant=id)

def test_future_shards_prefix_shift_and_family_boundaries():
    sealed=row('sealed','assessment','held',range(100),'structural')
    index=ExclusionIndex((sealed,))
    assert 'structural_family' in index.reasons(row('train','training','held',range(100,200)))
    assert 'finite_window' in index.reasons(row('shift','training','other',range(1,101)))
    assert 'finite_window' in index.reasons(row('prefix','selection','other',list(range(20))+[500]*80))
    safe=row('safe','training','other',range(1000,1100))
    assert not index.reasons(safe)
    index=index.add(safe)
    assert index.scope=='bounded finite windows; not semantic equivalence'
    with pytest.raises(ValueError): index.add(row('collision','selection','another',range(1000,1100)))
