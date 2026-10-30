"""Update-boundary learner adapter around Phase B token and resume policies."""
import time
from oeis_learn.generalization.budgets import charge_tokens
from .identity import digest


def train_updates(state,ledger,batches,*,update,save,seconds=21600):
    """Persist pre-update state/debit, then publish complete update state atomically.

    `update` owns model/optimizer/scheduler mutation and refreshes `state`'s tensor
    and RNG entries. On failure the caller reloads the last saved generation.
    Batches are immutable prepared rows; data cursor indexes their frozen order.
    """
    if type(seconds) not in (int,float) or not 0<seconds<=21600:raise ValueError('bounded learner seconds required')
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        batch=state['data']['pending_batch']
        if batch is None:
            cursor=state['data']['cursor']
            if cursor>=len(batches):return dict(status='EXHAUSTED',ledger=ledger,state=state)
            batch=batches[cursor];state['data']['cursor']=cursor+1
            state['data']['pending_batch']=batch
        tokens=charge_tokens([row['tokens'] for row in batch],bos=1,pad=0)
        key=digest(batch);ledger=ledger.reserve(key,tokens)
        save(state,ledger)
        if not ledger.debited:return dict(status='COMPLETE',ledger=ledger,state=state)
        update(batch)
        ledger=ledger.commit(key);state['data']['pending_batch']=None
        save(state,ledger)
        if ledger.exhausted:return dict(status='COMPLETE',ledger=ledger,state=state)
    return dict(status='PARTIAL',ledger=ledger,state=state)


def torch_update(encoder,decoder,optimizer,*,device,scheduler=None):
    """Use the existing real learner implementation; no synthetic loss path."""
    from oeis_learn.fresh.bigint_worker import _batch_loss
    def update(batch):
        encoder.train();decoder.train()
        result=_batch_loss(encoder,decoder,batch,device,optimizer=optimizer,observe=False)
        if scheduler is not None:scheduler.step()
        return result
    return update
