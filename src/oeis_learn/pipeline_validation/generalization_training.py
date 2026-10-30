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


class PreparedBatches:
    """Indexed immutable training shards; static repeats, refresh stops at exhaustion.

    Only the current shard is cached. The cursor is a batch ordinal, so restoring
    a pending batch never advances the stream a second time.
    """
    def __init__(self,root,shards,*,batch_size,mode,updates):
        import json
        from oeis_learn.generalization.contracts import verify_artifact,shape
        from oeis_learn.decoder.bigint_codec import encode_body
        if mode not in {'static','refresh'} or type(batch_size) is not int or not 1<=batch_size<=128 or type(updates) is not int or updates<1:
            raise ValueError('registered bounded data mode required')
        if not shards or len(shards)>16 or sum(r['bytes'] for r in shards)>8<<30:
            raise ValueError('prepared shard ceiling')
        self.root=root;self.shards=tuple(shards);self.mode=mode;self.batch_size=batch_size
        self.sizes=[];self.cache=None;seen=set()
        for ref in shards:
            rows=json.loads(verify_artifact(root,ref,max_bytes=8<<30))
            if not isinstance(rows,list) or not rows:raise ValueError('nonempty training shard required')
            for row in rows:
                shape('training-target',row)
                if row['id'] in seen or row['tokens']!=encode_body(row['source']):raise ValueError('duplicate task or target codec mismatch')
                seen.add(row['id'])
            self.sizes.append(len(rows))
        self.count=sum(self.sizes)
        self.updates=updates if mode=='static' else min(updates,(self.count+batch_size-1)//batch_size)
        self.identity=digest(dict(shards=list(shards),mode=mode,batch_size=batch_size,updates=updates))

    def __len__(self):return self.updates

    def _row(self,index):
        import json
        from oeis_learn.generalization.contracts import verify_artifact
        for shard,size in enumerate(self.sizes):
            if index<size:break
            index-=size
        if self.cache is None or self.cache[0]!=shard:
            self.cache=(shard,json.loads(verify_artifact(self.root,self.shards[shard],max_bytes=8<<30)))
        row=self.cache[1][index]
        return dict(id=row['id'],values=list(row['values'][:20]),indices=list(row['indices'][:20]),tokens=list(row['tokens']))

    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError(index)
        start=index*self.batch_size
        end=start+self.batch_size if self.mode=='static' else min(start+self.batch_size,self.count)
        return [self._row(i%self.count) for i in range(start,end)]
