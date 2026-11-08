"""Independent closed diagnostic expectations, not output of the learner under test."""
import copy
import pytest
from oeis_learn.pipeline_validation.identity import digest,seal


def record():
    h=digest('fixture')
    runs={}
    for arm in 'BCD':
        channels={'B':dict(fixed=256),'C':dict(bandit=192,review=32,fallback=32),'D':dict(bandit=256,review=0,fallback=0)}[arm]
        runs[arm]=dict(training_status='PARTIAL',updates=256,channels=channels,tokens=10000,charged_tokens=10000,
            discarded_tokens=0,valid_rewards={'B':0,'C':24,'D':31}[arm],initial_weights=h,checkpoint=h,
            finite=True,changed=True,forward_loss_sum=2.,forward_tokens=64,pauses=[])
    proofs={}
    for arm,r in runs.items():
        value=dict(checkpoint=h,weights=h,initial=h,updates=256,tokens=r['tokens'],finite=True,changed=True)
        proofs[arm]=seal(dict(status='PASS',runs=[value],resume_equal=None))
    proofs['resume']=seal(dict(status='PASS',runs=[dict(proofs['C']['runs'][0]),dict(proofs['C']['runs'][0])],resume_equal=True,pac_equal=True))
    return seal(dict(version='pac-rehearsal/v1',status='PASS',scientific_status='UNMEASURED',purpose='diagnostic',device='hip',
        source=h,environment=h,study=h,inventory=h,initializer=h,seed=20261016,batch_size=32,updates=256,
        target_tokens=25000000,proofs=proofs,cadence='pac-exp3s-leitner/v1',runs=runs,
        resume=dict(arm='C',boundary=129,updates=256,equal=True,pac_equal=True),production_retention_validated=False,semantic_scope='UNMEASURED_R0_DETERMINISTIC_PROOF'))


def test_closed_rehearsal_proof_reconciles_every_arm_and_keeps_scope():
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import verify_rehearsal
    r=record();assert verify_rehearsal(r,source=r['source'],environment=r['environment'])==r


@pytest.mark.parametrize('change',[
    lambda r:r.update(scientific_status='EXPLORATORY_SINGLE_SEED'),
    lambda r:r.update(batch_size=2),lambda r:r.update(cadence='pac-acceptance/v1'),
    lambda r:r.update(production_retention_validated=True),lambda r:r['runs'].pop('D'),
    lambda r:r['runs']['B'].update(training_status='PASS'),lambda r:r['runs']['D'].update(updates=255),
    lambda r:r['runs']['C']['channels'].update(bandit=191),lambda r:r['runs']['C'].update(finite=False),
    lambda r:r['runs']['D'].update(initial_weights=digest('different')),lambda r:r['resume'].update(equal=False),
    lambda r:r['runs']['D'].update(valid_rewards=0),lambda r:r['runs']['C'].update(pauses=['required_reward_budget']),
    lambda r:r['runs']['B'].update(forward_loss_sum=float('inf')),
    lambda r:r['runs']['C'].update(charged_tokens=25000001),
    lambda r:r['proofs'].pop('resume'),
    lambda r:r.update(semantic_scope='production'),
])
def test_resealed_partial_or_false_diagnostic_cannot_pass(change):
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import verify_rehearsal
    r=copy.deepcopy(record());change(r)
    with pytest.raises(ValueError):verify_rehearsal(seal(r))


def test_wrong_producer_cannot_reuse_rehearsal():
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import verify_rehearsal
    with pytest.raises(ValueError):verify_rehearsal(record(),source=digest('new source'))
