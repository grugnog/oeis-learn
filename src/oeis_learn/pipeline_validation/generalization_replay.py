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
                 transformed_source=candidate.get('transformed_source'),
                 assignment=candidate['assignment'],tokens=candidate['tokens'],origin_checkpoint=checkpoint,
                 production=production,production_identity=client.runtime_identity('bigint-body/v1'),order=order)


def search_round(tasks,*,propose,client,checkpoint,seed,seconds=14400,save=None,audit=None):
    """Reserve before callbacks and publish settled evidence even on interruption.

    Orchestrators supply an atomic save(pool_snapshot, report) callback for durable
    crash evidence. Without it, exceptions carry replay_evidence in memory only.
    A persisted pending reservation is unmeasured, never a completed zero-cost run.
    """
    from .identity import number
    number(seconds,positive=True)
    if seconds>14400 or len(tasks)>256:raise ValueError('bounded training search required')
    if len({t['id'] for t in tasks})!=len(tasks):raise ValueError('duplicate training search task')
    pool=ReplayPool(seed=seed);ledger=SearchLedger();rejected=[];began=time.monotonic()
    deadline=began+seconds;ordinal=0;status='PASS'
    def report(state):
        return seal(dict(version='generalization-replay-round-result/v2',status=state,plan=round_plan(len(pool.entries)),
            search_ledger=list(ledger.records),search_state=ledger.snapshot(),rejections=rejected,admitted=len(pool.entries),
            current_policy_success=None,cumulative_discoveries=len(pool.entries)))
    def publish(state):
        result=report(state)
        if save is not None:save(pool.snapshot(),result)
        return result
    for task in tasks:
        task_deadline=min(deadline,time.monotonic()+60)
        for attempt in range(8):
            start=time.monotonic()
            allowance=min(task_deadline-start,ledger.available(task['id']))
            if allowance<=0:
                status='PARTIAL';break
            ledger=ledger.reserve(task['id'],allowance)
            publish('RUNNING')  # Failure to persist prevents the callback from running.
            entry=None;failure=None
            try:
                if time.monotonic()>=task_deadline:raise TimeoutError('reservation persistence exhausted deadline')
                visible=dict(id=task['id'],indices=list(task['indices'][:20]),values=list(task['values'][:20]))
                candidate=propose(visible,attempt,task_deadline)
                if time.monotonic()>=task_deadline:raise TimeoutError('proposal deadline exhausted')
                entry=(audit or admission)(task,candidate,client=client,checkpoint=checkpoint,order=ordinal,deadline=task_deadline)
            except (ValueError,TimeoutError) as exc:
                rejected.append(dict(task=task['id'],attempt=attempt,reason=str(exc)))
            except BaseException as exc:
                failure=exc
                rejected.append(dict(task=task['id'],attempt=attempt,reason=type(exc).__name__))
            finally:
                elapsed=time.monotonic()-start
                ledger=ledger.finish(elapsed)
            late=time.monotonic()>=task_deadline
            if late:
                status='PARTIAL'
                rejected.append(dict(task=task['id'],attempt=attempt,reason='budget_exhausted'))
            if failure is not None:
                result=report('INTERRUPTED' if isinstance(failure,(KeyboardInterrupt,InterruptedError)) else 'FAIL')
                failure.replay_evidence=result
                if save is not None:save(pool.snapshot(),result)
                raise failure
            if entry is not None and not late:pool=pool.add(entry)
            ordinal+=1
            publish('RUNNING')
            if late:break
        if time.monotonic()>=deadline:
            status='PARTIAL';break
    return pool,publish(status)


def propose_visible(root,checkpoint,prompt,recipe,*,client,seed,attempt,device,seconds):
    """One real draw and visible-only native fit, with no training oracle mounted."""
    import torch
    from oeis_learn.generalization.contracts import shape
    from oeis_learn.decoder.bigint_generation import GenerationRequest,sample_batch
    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.synthesis.admission import VisiblePrompt
    from .generalization_evaluation import load_retained
    from oeis_learn.fresh.bigint_diagnostics import _seal
    shape('visible-prompt',prompt)
    if type(attempt) is not int or not 0<=attempt<8 or not 0<seconds<=60:
        raise ValueError('bounded replay draw required')
    deadline=time.monotonic()+seconds
    encoder,decoder,_=load_retained(root,checkpoint,recipe,device=device)
    with torch.no_grad():
        memory=encoder.forward_from_sequences([[int(v) for v in prompt['values']]],device=device)
    generated=sample_batch(decoder,[GenerationRequest(memory,seed,0 if attempt==0 else .8,deadline)],batch_size=1)[0]
    if not generated['complete'] or time.monotonic()>=deadline:raise ValueError('no_complete_proposal')
    source=generated['source'];visible=VisiblePrompt(tuple(prompt['indices']),tuple(prompt['values']))
    fitted=client.fit(source,visible,policy=fitting_policy(source),deadline=deadline)
    chosen=_seal(fitted.get('admitted',[]),visible,cost=0)
    if chosen is None or time.monotonic()>=deadline:raise ValueError('no_visible_fit')
    final=chosen['selected']['source']
    return dict(original_source=source,transformed_source=chosen['transformed_source'],
        final_source=final,assignment=chosen['assignment'],tokens=encode_body(final))


def search_isolated(host,workspace,output,*,images,weights_root,checkpoint,tasks,recipe,study,seed,seconds=14400):
    """Serial role calls: visible proposals and training-only independent audits.

    The ledger is durably written before each model attempt. All role launch,
    copying, fitting, and audit overhead consumes the same measured reservation.
    """
    import json
    from pathlib import Path
    from .generalization_roles import project_and_invoke
    from .generalization_preparation import reference
    from oeis_learn.generalization.replay_admission import AdmittedEntry
    from oeis_learn.fresh.common import write
    from .identity import digest
    workspace=Path(workspace);output=Path(output)
    if output.exists():raise ValueError('fresh replay search output required')
    output.mkdir(parents=True,exist_ok=False)
    def persist(pool,report):write(output/'private/search-state.json',seal(dict(pool=pool,report=report)))
    current={}
    def propose(visible,attempt,deadline):
        key=digest([visible['id'],attempt])[7:];folder=output/'private/attempts'/key
        folder.mkdir(parents=True,exist_ok=False)
        write(folder/'prompt.json',visible)
        prompt=reference(folder/'prompt.json',folder)
        result=project_and_invoke(host,images[1],workspace,'generator',
            {'visible_prompts':[prompt],'frozen_weights':[checkpoint]},
            source_roots={'visible_prompts':folder,'frozen_weights':weights_root},
            output=str((folder/'proposal').relative_to(workspace)),operation='replay-propose',
            settings=dict(checkpoint=checkpoint,prompt=prompt,recipe=recipe,study=study,
                seed=int(digest([seed,visible['id'],attempt])[7:15],16),attempt=attempt,device='cuda'),
            seconds=min(60,max(.001,deadline-time.monotonic())),gpu=True)
        if result['status']!='PASS':raise ValueError(result.get('error',result.get('reason','proposal_failed')))
        current['folder']=folder
        return result['candidate']
    def audit(task,candidate,*,client,checkpoint,order,deadline):
        folder=current['folder'];write(folder/'task.json',task);write(folder/'candidate.json',candidate)
        task_ref=reference(folder/'task.json',folder);candidate_ref=reference(folder/'candidate.json',folder)
        result=project_and_invoke(host,images[0],workspace,'learner',
            {'training_targets':[task_ref],'verified_training_replay':[candidate_ref]},
            source_roots={'training_targets':folder,'verified_training_replay':folder},
            output=str((folder/'audit').relative_to(workspace)),operation='replay-audit',
            settings=dict(task=task_ref,candidate=candidate_ref,study=study,checkpoint=checkpoint,order=order),
            seconds=min(60,max(.001,deadline-time.monotonic())))
        if result['status']!='PASS':raise ValueError(result.get('error',result.get('reason','audit_failed')))
        record=result['entry']
        if seal(record)!=record or record['task_identity']!=digest(task) or record['origin_checkpoint']!=checkpoint or record['order']!=order:
            raise ValueError('audit result binding mismatch')
        return AdmittedEntry(json.dumps(record,sort_keys=True).encode())
    pool,report=search_round(tasks,propose=propose,audit=audit,client=None,checkpoint=checkpoint['sha256'],
        seed=seed,seconds=seconds,save=persist)
    write(output/'private/pool.json',pool.snapshot());write(output/'round.json',report)
    return pool,report


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
