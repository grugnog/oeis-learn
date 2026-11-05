"""Update-boundary learner adapter around Phase B token and resume policies."""
import time
from oeis_learn.generalization.budgets import charge_tokens
from .identity import digest


def publication_reserve(seconds):
    """Keep terminal publication inside the enclosing learner allowance."""
    return min(300.,seconds/5)


class TrainingTiming:
    """Bounded diagnostic windows; never alter optimization/accounting decisions."""
    def __init__(self):
        from collections import deque
        self.windows=deque(maxlen=64);self.start=None;self.anchor=None
        self.checkpoint_seconds=0.;self.update_seconds=0.;self.checkpoint_calls=0;self.max_checkpoint_seconds=0.
        self.trend=[];self.loss_sum=0.;self.loss_count=0;self.last_loss=None;self.last_gradient_norm=None

    def record_update(self,result):
        import math
        if not isinstance(result,dict):return
        for name,attr in (('loss','last_loss'),('gradient_norm','last_gradient_norm')):
            value=result.get(name)
            if value is not None:
                if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite observed training metric required')
                setattr(self,attr,value)
        if result.get('loss') is not None:self.loss_sum+=result['loss'];self.loss_count+=1

    def observe(self,updates,tokens,now):
        if self.start is None:self.start=updates
        if updates-self.start<4:return False
        if self.anchor is None:
            self.anchor=(updates,tokens,now);self.loss_sum=0.;self.loss_count=0;return False
        count,consumed,started=self.anchor
        if updates-count<8:return False
        window=dict(updates=updates-count,tokens=tokens-consumed,seconds=now-started,through_updates=updates,
            loss_sum=self.loss_sum,loss_observations=self.loss_count,mean_loss=self.loss_sum/self.loss_count if self.loss_count else None)
        self.windows.append(window);self.trend.append(dict(window))
        if len(self.trend)>128:
            a,b=self.trend[:2]
            merged={k:a[k]+b[k] for k in ('updates','tokens','seconds','loss_sum','loss_observations')}
            merged.update(through_updates=b['through_updates'],mean_loss=merged['loss_sum']/merged['loss_observations'] if merged['loss_observations'] else None)
            self.trend[:2]=[merged]
        self.loss_sum=0.;self.loss_count=0
        self.anchor=(updates,tokens,now)
        return True


def train_updates(state,ledger,batches,*,update,save,seconds=21600,discarded_tokens=0):
    """Persist pre-update state/debit, then publish complete update state atomically.

    `update` owns model/optimizer/scheduler mutation and refreshes `state`'s tensor
    and RNG entries. On failure the caller reloads the last saved generation.
    Batches are immutable prepared rows; data cursor indexes their frozen order.
    """
    if type(seconds) not in (int,float) or not 0<seconds<=21600:raise ValueError('bounded learner seconds required')
    if type(discarded_tokens) is not int or not 0<=discarded_tokens<=ledger.cap-ledger.consumed:raise ValueError('invalid discarded exposure')
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if ledger.consumed+discarded_tokens==ledger.cap and not ledger.debited:return dict(status='COMPLETE',ledger=ledger,state=state)
        batch=state['data']['pending_batch']
        if batch is None:
            cursor=state['data']['cursor']
            if cursor>=len(batches):return dict(status='EXHAUSTED',ledger=ledger,state=state)
            batch=batches[cursor];state['data']['cursor']=cursor+1
            state['data']['pending_batch']=batch
        tokens=charge_tokens([row['tokens'] for row in batch],bos=1,pad=0)
        key=digest(batch)
        if ledger.pending is None and ledger.consumed+tokens+discarded_tokens>ledger.cap:
            from dataclasses import replace
            ledger=replace(ledger,pending=(key,tokens),debited=False)
        else:ledger=ledger.reserve(key,tokens)
        save(state,ledger)
        if not ledger.debited:return dict(status='COMPLETE',ledger=ledger,state=state)
        if time.monotonic()>=deadline:break
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
        self.sizes=[];self.cache=None;seen=set();self.target_tokens=0
        for ref in shards:
            rows=json.loads(verify_artifact(root,ref,max_bytes=8<<30))
            if not isinstance(rows,list) or not rows:raise ValueError('nonempty training shard required')
            for row in rows:
                shape('training-target',row)
                if row['id'] in seen or row['tokens']!=encode_body(row['source']):raise ValueError('duplicate task or target codec mismatch')
                seen.add(row['id'])
            self.sizes.append(len(rows))
            self.target_tokens+=charge_tokens([r['tokens'] for r in rows],bos=1,pad=0)
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


def run_torch(root,output,config,*,resume=None,stop_after_updates=None,initializer=None,replay=None,initializer_root=None,seconds=None,resume_elapsed=None):
    """Actual existing model/update path with complete update-boundary persistence.

    Caller supplies a verified, frozen execution projection. No selection/assessment
    targets enter this worker. All milestone candidates remain available for the
    separately mounted selection role after training.
    """
    began=time.monotonic()
    import copy,json,random,shutil,sys
    from pathlib import Path
    import numpy as np
    import torch
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.generalization.checkpoints import milestone_decision,pin_plan
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.fresh.common import write
    from .generalization_persistence import policy,INTERVAL,journal,recover
    checkpoint_policy=policy(config);periodic=checkpoint_policy=='hourly/v1'
    progress_count=0;timing=TrainingTiming();tensors_dirty=True
    journal_writes=0;last_full_time=None;last_full_ledger=None
    def progress(phase):
        nonlocal progress_count
        # Public phase/timestamp only; bounded even for long research learners.
        if progress_count<128 or phase=='complete':
            print(f'learner: {phase} unix={time.time():.6f}',file=sys.stderr,flush=True)
            progress_count+=1
    required={'study','arm','seed','recipe','shards','data_mode','batch_size','target_tokens','milestones','seconds','device'}
    if set(config)-{'checkpoint_policy'}!=required:raise ValueError('closed learner execution projection required')
    if config['device'] not in {'cpu','cuda'} or not 0<config['seconds']<=21600:raise ValueError('explicit bounded learner device/time required')
    from .generalization_execution import verify_execution,bind_compatibility
    execution_policy=verify_execution() if config['device']=='cuda' else None
    seconds=config['seconds'] if seconds is None else seconds
    if not 0<seconds<=config['seconds']:raise ValueError('learner allowance exceeds frozen budget')
    reserve=publication_reserve(seconds)
    milestones=config['milestones'];cap=config['target_tokens']
    if milestones!=sorted(set(milestones)) or any(type(m) is not int or not 0<m<=cap for m in milestones):raise ValueError('bounded increasing milestones required')
    progress('batch-inputs')
    batches=PreparedBatches(root,config['shards'],batch_size=config['batch_size'],mode=config['data_mode'],updates=cap)
    if config['data_mode']=='refresh' and batches.target_tokens<cap:raise ValueError('insufficient prepared refresh target-token capacity')
    data_identity=batches.identity
    extension={}
    if initializer is not None:extension['initializer']=initializer
    if replay is not None:
        if set(replay)!={'pool','tasks','client'} or config['data_mode']!='static':
            raise ValueError('verified replay supports only registered static training')
        tasks={task['id']:task for task in replay['tasks']}
        if len(tasks)!=len(replay['tasks']):raise ValueError('duplicate replay training task')
        matched=set()
        for shard in config['shards']:
            for row in json.loads(verify_artifact(root,shard,max_bytes=8<<30)):
                if row['id'] not in tasks:continue
                task=tasks[row['id']]
                if task['role']!='training' or any(task[field]!=row[field] for field in ('source','indices','values')):
                    raise ValueError('replay oracle differs from prepared training task')
                matched.add(row['id'])
        if matched!=set(tasks):raise ValueError('replay task absent from prepared training shards')
        extension['replay']=dict(pool=replay['pool'].snapshot(),tasks=replay['tasks'])
    compatibility=digest(dict(config=config,data=batches.identity,**bind_compatibility(config['device'],extension)))
    output=Path(output)
    if output.exists():raise ValueError('fresh learner output required')
    output.mkdir(parents=True,exist_ok=False)
    recipe=copy.deepcopy(config['recipe']);recipe['model']['seed']=config['seed']
    progress('model-init')
    encoder,decoder,counts=models(recipe,config['device'])
    if initializer is not None and resume is None:
        import io
        progress('initializer-load')
        weights=torch.load(io.BytesIO(verify_artifact(initializer_root or root,initializer)),map_location='cpu',weights_only=True)
        if set(weights)!={'encoder','decoder'}:raise ValueError('weights-only initializer required')
        encoder.load_state_dict(weights['encoder'],strict=True);decoder.load_state_dict(weights['decoder'],strict=True)
    parameters=list(encoder.parameters())+list(decoder.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=recipe['learning']['learning_rate'])
    store=GeneralizationCheckpointStore(output/'checkpoints',compatibility,cap)
    state=dict(weights={},optimizer={},scheduler=dict(learner_seconds=0.),rng={},data=dict(cursor=0,pending_batch=None),
        replay={},history=dict(exposures={},milestones={}),pins={},selection=dict(checkpoint=None,reason='selection_not_run'))
    ledger=Ledger(cap);latest=None;history={};initial=None
    if resume is not None:
        progress('resume-load')
        prior,ref=resume;old=GeneralizationCheckpointStore(prior,compatibility,cap)
        restored=old.load(ref);state=restored['states'];ledger=restored['ledger']
        for member in ('encoder','decoder'):(encoder if member=='encoder' else decoder).load_state_dict(state['weights'][member])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
        if config['device']=='cuda':torch.cuda.set_rng_state_all(state['rng']['device'])
        # Copy only verified pinned generations into the new run-owned store.
        for key,pointer in state['pins'].items():
            old.load(pointer)
            folder=Path(pointer['path']).parent
            shutil.copytree(Path(prior)/folder,store.root/folder,dirs_exist_ok=False)
            history[key]=pointer
        initial=next(iter(state['history']['exposures']),None)
        if periodic:
            pointer_path=Path(prior)/'generalization-latest.json'
            if json.loads(pointer_path.read_text())!=ref:raise ValueError('periodic resume requires latest published snapshot')
            progress_path=Path(prior)/'generalization-progress.json'
            if progress_path.is_symlink() or not progress_path.is_file() or progress_path.stat().st_size>1<<20:
                raise ValueError('original periodic progress journal required')
            restored_accounting=recover(json.loads(progress_path.read_text()),contract=compatibility,checkpoint=ref,ledger=ledger,
                discarded_tokens=state['scheduler'].get('discarded_tokens',0),learner_seconds=state['scheduler']['learner_seconds'],
                seconds_cap=config['seconds'],supervised_seconds=resume_elapsed)
            state['scheduler'].update({k:v for k,v in restored_accounting.items() if k!='rolled_back'})
    # Conservatively debit reduced enclosing allowance (I/O and cleanup reserve).
    previous_seconds=state['scheduler']['learner_seconds']+config['seconds']-seconds;saved_ledger=ledger
    discarded_tokens=state['scheduler'].get('discarded_tokens',0)
    permanent=dict(state['pins']);milestone_keys={int(k):v for k,v in state['history']['milestones'].items()}
    update_tensors=torch_update(encoder,decoder,optimizer,device=config['device'])
    def update(batch):
        nonlocal tensors_dirty
        tensors_dirty=True;started=time.monotonic()
        progress('update-start')
        try:
            result=update_tensors(batch);timing.record_update(result);return result
        finally:
            timing.update_seconds+=time.monotonic()-started
            progress('update-complete')
    class StopAtBoundary(Exception):pass
    def save(current,account,*,force=False):
        nonlocal latest,saved_ledger,initial,history,tensors_dirty,last_full_time,last_full_ledger,journal_writes
        effective=account.consumed-(account.pending[1] if account.debited else 0)
        crossed=milestone_decision(effective,account.pending[1],milestones) if account.pending else ()
        milestone_due=any(m not in milestone_keys and (m==effective or m in crossed) for m in milestones)
        stopped=stop_after_updates is not None and account.pending is None and account.updates>=stop_after_updates
        full=(not periodic or force or latest is None or milestone_due or stopped or
              account.pending is None and time.monotonic()-last_full_time>=INTERVAL)
        if full:full_save(current,account)
        saved_ledger=account
        if periodic:
            write(store.root/'generalization-progress.json',journal(contract=compatibility,checkpoint=latest,ledger=account,
                discarded_tokens=discarded_tokens,elapsed_seconds=previous_seconds+time.monotonic()-began,
                invocation_base_seconds=previous_seconds,closed=False))
            journal_writes+=1
        emitted=account.pending is None and timing.observe(account.updates,effective,time.monotonic())
        if emitted or periodic and (full or account.pending is None):
            write(output/'timing.json',timing_record())
        if stopped:raise StopAtBoundary()

    def timing_record():
        return dict(version='generalization-training-timing/v1',checkpoint_seconds=timing.checkpoint_seconds,
            update_seconds=timing.update_seconds,checkpoint_calls=timing.checkpoint_calls,windows=list(timing.windows),
            checkpoint_io=dict(store.io_stats),checkpoint_policy=checkpoint_policy,journal_writes=journal_writes,
            max_checkpoint_seconds=timing.max_checkpoint_seconds,trend=list(timing.trend),
            last_loss=timing.last_loss,last_gradient_norm=timing.last_gradient_norm,
            invocation_seconds=time.monotonic()-began,
            progress=dict(tokens=saved_ledger.consumed-(saved_ledger.pending[1] if saved_ledger.debited else 0),
                charged_tokens=saved_ledger.consumed+discarded_tokens,discarded_tokens=discarded_tokens,updates=saved_ledger.updates,
                recoverable_tokens=last_full_ledger.consumed-(last_full_ledger.pending[1] if last_full_ledger.debited else 0),
                recoverable_updates=last_full_ledger.updates,checkpoint=latest))

    def full_save(current,account):
        nonlocal latest,initial,history,tensors_dirty,last_full_time,last_full_ledger
        started=time.monotonic()
        progress('checkpoint-save')
        effective=account.consumed-(account.pending[1] if account.debited else 0)

        current['weights']=dict(encoder=encoder.state_dict(),decoder=decoder.state_dict())
        current['optimizer']=optimizer.state_dict()
        current['rng']=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                            device=torch.cuda.get_rng_state_all() if config['device']=='cuda' else [])
        def scheduler():
            current['scheduler']['learner_seconds']=previous_seconds+time.monotonic()-began
            return current['scheduler']
        current['pins']=dict(permanent)
        current['history']['milestones']={str(k):v for k,v in milestone_keys.items()}
        unchanged=latest if latest is not None and not tensors_dirty and account.pending is not None else None
        latest=store.save(current,account,scheduler=scheduler,unchanged_tensors=unchanged)
        key=latest['sha256'];history[key]=latest;current['history']['exposures'][key]=effective
        if initial is None:initial=key;permanent[key]=latest
        crossed=milestone_decision(effective,account.pending[1],milestones) if account.pending else ()
        for milestone in milestones:
            if milestone not in milestone_keys and (milestone==effective or milestone in crossed):
                milestone_keys[milestone]=key;permanent[key]=latest
        pins=pin_plan(current['history']['exposures'],milestones=milestones,best=initial,latest=key,selected=None,
                      through=effective,milestone_pins=milestone_keys)
        keep=set(pins)|set(permanent)
        store.retain(history,tuple(sorted(keep)))
        history={k:v for k,v in history.items() if k in keep}
        tensors_dirty=False
        last_full_time=time.monotonic();last_full_ledger=account
        duration=time.monotonic()-started
        timing.checkpoint_seconds+=duration;timing.checkpoint_calls+=1
        timing.max_checkpoint_seconds=max(timing.max_checkpoint_seconds,duration)
        progress('checkpoint-saved')
    # Restore a snapshot that itself was newly pinned before its reference could
    # appear inside that same snapshot; reconstruct only from its exact ledger.
    if resume is not None:
        prior,ref=resume
        key=ref['sha256'];effective=ledger.consumed-(ledger.pending[1] if ledger.debited else 0)
        state['history']['exposures'][key]=effective
        if initial is None:initial=key
        crossed=milestone_decision(effective,ledger.pending[1],milestones) if ledger.pending else ()
        need=initial==key
        for milestone in milestones:
            if milestone not in milestone_keys and (milestone==effective or milestone in crossed):
                milestone_keys[milestone]=key;need=True
        if need and key not in permanent:
            folder=Path(ref['path']).parent
            shutil.copytree(Path(prior)/folder,store.root/folder,dirs_exist_ok=False)
            permanent[key]=ref;history[key]=ref
    if replay is not None:
        progress('replay-restore')
        from .generalization_replay import restore_pool,ReplayBatches
        snapshot=state['replay'] if resume is not None else replay['pool'].snapshot()
        verified=restore_pool(snapshot,tasks=replay['tasks'],client=replay['client'],deadline=min(time.monotonic()+60,began+config['seconds']-previous_seconds))
        state['replay']=verified.snapshot()
        batches=ReplayBatches(batches,verified,replay['tasks'],state)
    try:
        save(state,ledger)
        remaining=config['seconds']-previous_seconds-(time.monotonic()-began)-reserve
        result=train_updates(state,ledger,batches,update=update,save=save,seconds=remaining,discarded_tokens=discarded_tokens) if remaining>0 else dict(status='PARTIAL')
        status=result['status']
    except StopAtBoundary:status='PARTIAL'
    if periodic:
        # Publish live state once at the orderly terminal boundary, before any
        # descriptor or successful result can refer to it.
        if latest is None or last_full_ledger!=saved_ledger:full_save(state,saved_ledger)

    # These references are private full-state descriptors. Evaluation projects
    # only each descriptor's weights, never its optimizer/RNG or training data.
    from .identity import seal
    published=time.monotonic()
    progress('publish-roster')
    initial_record=json.loads(verify_artifact(store.root,history[initial]))
    initialization=initial_record['states']['weights']['sha256']
    retained=[]
    for key,pointer in sorted(history.items()):
        labels=['retained']
        if key==initial:labels.append('initial')
        if pointer==latest:labels.append('latest')
        labels.extend('milestone:'+str(m) for m,k in milestone_keys.items() if k==key)
        retained.append(state['history'].get('descriptors',{}).get(key) or store.describe(pointer,initialization=initialization,
            parent=initializer['sha256'] if initializer is not None else None,pins=labels))
    roster=seal(dict(version='generalization-retained/v1',study=config['study'],
        arm=config['arm'],seed=config['seed'],checkpoints=retained,
        best_loss=state['selection'].get('best_loss'),selected=state['selection'].get('checkpoint'),
        selection_status=state['selection'].get('reason','selection_not_run'),
        latest=next(c['identity'] for c in retained if c['state']==latest)))
    write(output/'checkpoints/retained.json',roster)
    if periodic:
        write(store.root/'generalization-progress.json',journal(contract=compatibility,checkpoint=latest,ledger=saved_ledger,
            discarded_tokens=discarded_tokens,elapsed_seconds=previous_seconds+time.monotonic()-began,
            invocation_base_seconds=previous_seconds,closed=True))
        journal_writes+=1
    if status=='COMPLETE' and discarded_tokens:status='RECOVERED_WITH_DISCARDED_EXPOSURE'
    result=dict(status='PASS' if status=='COMPLETE' else 'PARTIAL',reason=status,compatibility=compatibility,
        checkpoint=latest,tokens=saved_ledger.consumed-(saved_ledger.pending[1] if saved_ledger.debited else 0),
        charged_tokens=saved_ledger.consumed+discarded_tokens,discarded_tokens=discarded_tokens,checkpoint_policy=checkpoint_policy,pending_tokens=saved_ledger.pending[1] if saved_ledger.pending else 0,
        updates=saved_ledger.updates,undershoot=saved_ledger.cap-saved_ledger.consumed-discarded_tokens,
        milestones={str(m):state['history']['exposures'][k] for m,k in milestone_keys.items()},
        learner_seconds=state['scheduler']['learner_seconds'],data=data_identity,device=config['device'],counts=counts,
        retained=roster['identity'])
    result['timing']=dict(timing_record(),publication_seconds=time.monotonic()-published,publication_reserve_seconds=reserve)
    write(output/'timing.json',result['timing'])
    if execution_policy is not None:result['execution_policy']=execution_policy
    write(output/'training.json',result)
    progress('complete')
    return result


def storage_preflight(recipe,study,runtime):
    """Measure actual CPU model tensors and reserve optimizer/pinned snapshots."""
    if not runtime['training']:
        return dict(required_checkpoint_bytes=0,pinned_generations_reserved=0,kind='evaluation_only_no_learner_snapshots')
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.generalization.budgets import reserve_storage
    encoder,decoder,_=models(recipe,'cpu')
    parameters=list(encoder.parameters())+list(decoder.parameters())
    buffers=list(encoder.buffers())+list(decoder.buffers())
    parameter_bytes=sum(p.numel()*p.element_size() for p in parameters)
    buffer_bytes=sum(p.numel()*p.element_size() for p in buffers)
    # FP32 AdamW has two parameter-sized moments. Reserve serialization metadata,
    # RNG/history, and a bounded full replay snapshot separately from tensor bytes.
    snapshot_bytes=3*parameter_bytes+buffer_bytes+4096*(len(parameters)+len(buffers))+(1<<20)
    if runtime['replay'] is not None:snapshot_bytes+=128<<20
    from oeis_learn.generalization.resources import checkpoint_reservation
    graph=checkpoint_reservation(snapshot_bytes=snapshot_bytes,weights_bytes=parameter_bytes+buffer_bytes,
        milestones=len(study['checkpoints']['milestones']),arms=len(runtime['training']),seeds=len(study['seeds']),
        acceptance_replay=runtime.get('purpose')=='acceptance' and runtime.get('replay') is not None,
        selection=runtime.get('evaluation') is not None)
    reserve_storage(graph['required_checkpoint_bytes'],1,min(32<<30,runtime['storage']['checkpoint_bytes']))
    return dict(parameter_bytes=parameter_bytes,buffer_bytes=buffer_bytes,snapshot_reservation_bytes=snapshot_bytes,
        **graph,device='cpu',kind='measured_tensors_with_serial_peak_reservation')


class PacPreparedBatches(PreparedBatches):
    """Indexed paired targets with a verified family sidecar and separate probes."""
    def __init__(self,root,bundle,*,arm,batch_size,updates):
        import json
        from oeis_learn.generalization.contracts import verify_artifact,shape
        from oeis_learn.generalization.role_contracts import pac_training_refs
        from oeis_learn.pipeline_validation.identity import seal
        from oeis_learn.decoder.bigint_codec import encode_body
        pac_training_refs(bundle,arm)
        if seal(bundle)!=bundle:raise ValueError('PAC prepared bundle identity')
        super().__init__(root,bundle['literal' if arm=='A' else 'parameterized'],batch_size=batch_size,mode='static',updates=updates)
        self.inventory=json.loads(verify_artifact(root,bundle['inventory']))
        if seal(self.inventory)!=self.inventory or self.inventory['role']!='training':raise ValueError('PAC training inventory integrity')
        self.families={f['id']:f['rows'] for f in self.inventory['families']}
        self.probes={f['id']:f['probes'] for f in self.inventory['families']}
        self.anchors={a:f['id'] for f in self.inventory['families'] for a in f['anchors']}
        self.row_families={r:f for f,ids in self.families.items() for r in ids}
        self.rows={}
        for i in range(self.count):
            row=super()._row(i)
            if row['id'] not in self.row_families:raise ValueError('prepared row absent from PAC family sidecar')
            row['family']=self.row_families[row['id']];self.rows[row['id']]=row
        if len(self.rows)!=len(self.row_families):raise ValueError('PAC family inventory/row count mismatch')
        probe_families={r:f for f,ids in self.probes.items() for r in ids}
        self.probe_rows={}
        for row in json.loads(verify_artifact(root,bundle['probes'],max_bytes=8<<30)):
            shape('training-target',row)
            if row['id'] not in probe_families or row['tokens']!=encode_body(row['source']) or row['id'] in self.probe_rows or row['id'] in self.rows:raise ValueError('PAC disjoint encoded training probes required')
            self.probe_rows[row['id']]=dict(id=row['id'],family=probe_families[row['id']],values=row['values'][:20],indices=row['indices'][:20],tokens=row['tokens'])
        if set(self.probe_rows)!=set(probe_families) or any(len(p)!=8 for p in self.probes.values()):raise ValueError('complete PAC probe quota required')
        self.identity=digest(dict(prepared=bundle['identity'],arm=arm,batch_size=batch_size))

    def _row(self,index):
        row=super()._row(index);row['family']=self.row_families[row['id']];return row
