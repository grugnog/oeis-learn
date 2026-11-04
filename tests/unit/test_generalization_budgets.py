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


def test_pac_peak_storage_counts_old_new_pins_proofs_and_both_packages():
    from oeis_learn.generalization.resources import pac_peak_storage
    args=dict(arms=['B','C'],snapshot_bytes=100,weights_bytes=10,data_bytes=1000,data_copies=2,
              scratch_bytes=50,logs_bytes=20,package_bytes=30,milestones=4,available_bytes=100000,scratch_kind='disk')
    result=pac_peak_storage(**args)
    # 2*(initial+latest+4 milestone)=12 retained; two additional simultaneous
    # atomic/resume/proof generations; 5 weights, 2 datasets, scratch/logs/packages.
    expected=14*100+5*10+2000+50+20+2*30
    assert result['required_bytes']==expected==3580
    assert pac_peak_storage(**dict(args,available_bytes=expected))['required_bytes']==expected
    for changed in ({'available_bytes':expected-1},{'scratch_kind':'tmpfs'},{'snapshot_bytes':None},{'data_copies':0}):
        with pytest.raises(ValueError):pac_peak_storage(**dict(args,**changed))
    assert pac_peak_storage(**dict(args,arms=['A','B','C','D']))['required_bytes']==4800
