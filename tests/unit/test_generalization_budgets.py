import pytest
from oeis_learn.generalization.budgets import Ledger, charge_tokens, reserve_storage

def test_post_substitution_loss_tokens():
    assert charge_tokens(((1, 8, 9, 2, 0), (1, 6, 2, 0, 0)), bos=1, pad=0) == 5

def test_pending_and_interrupted_debits():
    ledger = Ledger(10).reserve('batch-a', 6)
    assert ledger.consumed == 6 and ledger.updates == 0
    assert ledger.reserve('batch-a', 6) == ledger
    with pytest.raises(ValueError): ledger.reserve('batch-b', 2)
    done = ledger.commit('batch-a')
    assert done.updates == 1 and done.consumed == 6
    stopped = done.reserve('batch-b', 5)
    assert stopped.pending == ('batch-b', 5) and stopped.consumed == 6
    assert stopped.exhausted and stopped.undershoot == 4
    assert Ledger.restore(stopped.snapshot(), cap=10) == stopped
    with pytest.raises(ValueError): Ledger.restore(stopped.snapshot(), cap=11)

def test_storage_and_invalid_counts():
    reserve_storage(100, 3, 301)
    with pytest.raises(ValueError): reserve_storage(100, 4, 301)
    with pytest.raises(ValueError): Ledger(True)
    with pytest.raises(ValueError): Ledger(10).reserve('x', -1)
