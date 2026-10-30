"""One-round replay adapters using the existing bounded native client."""
import time
from oeis_learn.generalization.replay_admission import admit
from oeis_learn.generalization.replay_policy import ReplayPool
from oeis_learn.generalization.replay_round import SearchLedger,round_plan
from oeis_learn.decoder.bigint_codec import encode_body
from .identity import seal


def admission(task,candidate,*,client,checkpoint,order,deadline):
    """Candidate flags never substitute for native/reference full-horizon admission."""
    def production(source,indices):
        result=client.execute(source,indices,deadline=deadline)
        if result.get('outcome')!='ok':raise ValueError('native audit incomplete')
        return result['values']
    return admit(task,original_source=candidate['original_source'],final_source=candidate['final_source'],
                 assignment=candidate['assignment'],tokens=candidate['tokens'],origin_checkpoint=checkpoint,
                 production=production,production_identity=client.runtime_identity('bigint-body/v1'),order=order)


def search_round(tasks,*,propose,client,checkpoint,seed,seconds=14400):
    if not 0<seconds<=14400 or len(tasks)>256:raise ValueError('bounded training search required')
    if len({t['id'] for t in tasks})!=len(tasks):raise ValueError('duplicate training search task')
    pool=ReplayPool(seed=seed);ledger=SearchLedger();rejected=[];began=time.monotonic()
    deadline=began+seconds;ordinal=0
    for task in tasks:
        task_deadline=min(deadline,time.monotonic()+60)
        for attempt in range(8):
            if time.monotonic()>=task_deadline:break
            start=time.monotonic();entry=None
            try:
                # The search role gets the registered visible prefix only.
                visible=dict(id=task['id'],indices=list(task['indices'][:20]),values=list(task['values'][:20]))
                candidate=propose(visible,attempt,task_deadline)
                entry=admission(task,candidate,client=client,checkpoint=checkpoint,order=ordinal,deadline=task_deadline)
            except ValueError as exc:rejected.append(dict(task=task['id'],attempt=attempt,reason=str(exc)))
            elapsed=max(time.monotonic()-start,0.000001)
            try:ledger=ledger.charge(task['id'],elapsed)
            except ValueError:
                rejected.append(dict(task=task['id'],attempt=attempt,reason='budget_exhausted'));break
            if time.monotonic()>task_deadline:break
            if entry is not None:pool=pool.add(entry)
            ordinal+=1
        if time.monotonic()>=deadline:break
    return pool,seal(dict(version='generalization-replay-round-result/v1',plan=round_plan(len(pool.entries)),
                          search_ledger=list(ledger.records),rejections=rejected,admitted=len(pool.entries),
                          current_policy_success=None,cumulative_discoveries=len(pool.entries)))


def restore_pool(record,*,tasks,client,deadline):
    roster={task['id']:task for task in tasks}
    def verify(entry):
        row=entry.record()
        if row['task'] not in roster:raise ValueError('replay task absent from training roster')
        checked=admission(roster[row['task']],row,client=client,checkpoint=row['origin_checkpoint'],order=row['order'],deadline=deadline)
        if checked.record()!=row:raise ValueError('replay evidence changed on restore')
        return checked
    return ReplayPool.restore(record,verify=verify)


class ReplayBatches:
    """Adapt the Phase B sampler to the learner's persisted pending-batch protocol.

    The caller must independently reverify the pool before construction, including
    on resume. No sampling occurs when train_updates already has a pending batch.
    """
    def __init__(self,base,pool,tasks,state):
        self.base=base;self.pool=pool;self.state=state
        self.tasks={t['id']:t for t in tasks}
        if len(self.tasks)!=len(tasks) or any(t['role']!='training' for t in tasks):
            raise ValueError('unique training replay roster required')
        if state['replay']!=pool.snapshot():raise ValueError('reverified replay cursor differs from saved state')
        for entry in pool.entries:
            if entry.record()['task'] not in self.tasks:raise ValueError('replay task absent from training roster')

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        batch=self.base[index]
        sampled,next_pool=self.pool.sample(len(batch))
        replay=[]
        for entry in sampled:
            record=entry.record();task=self.tasks[record['task']]
            replay.append(dict(id='replay:'+record['identity'],tokens=list(record['tokens']),
                indices=list(task['indices'][:20]),values=list(task['values'][:20])))
        mixed=batch[:len(batch)-len(replay)]+replay
        self.pool=next_pool;self.state['replay']=next_pool.snapshot()
        return mixed
