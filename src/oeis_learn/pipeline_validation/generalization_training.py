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


def run_torch(root,output,config,*,resume=None,stop_after_updates=None,initializer=None,replay=None,initializer_root=None):
    """Actual existing model/update path with complete update-boundary persistence.

    Caller supplies a verified, frozen execution projection. No selection/assessment
    targets enter this worker. All milestone candidates remain available for the
    separately mounted selection role after training.
    """
    import copy,json,random,shutil
    from pathlib import Path
    import numpy as np
    import torch
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.generalization.checkpoints import milestone_decision,pin_plan
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.fresh.common import write
    required={'study','arm','seed','recipe','shards','data_mode','batch_size','target_tokens','milestones','seconds','device'}
    if set(config)!=required:raise ValueError('closed learner execution projection required')
    if config['device'] not in {'cpu','cuda'} or not 0<config['seconds']<=21600:raise ValueError('explicit bounded learner device/time required')
    milestones=config['milestones'];cap=config['target_tokens']
    if milestones!=sorted(set(milestones)) or any(type(m) is not int or not 0<m<=cap for m in milestones):raise ValueError('bounded increasing milestones required')
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
    compatibility=digest(dict(config=config,data=batches.identity,**extension))
    output=Path(output)
    if output.exists():raise ValueError('fresh learner output required')
    output.mkdir(parents=True,exist_ok=False)
    recipe=copy.deepcopy(config['recipe']);recipe['model']['seed']=config['seed']
    encoder,decoder,counts=models(recipe,config['device'])
    if initializer is not None and resume is None:
        import io
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
    began=time.monotonic();previous_seconds=state['scheduler']['learner_seconds'];saved_ledger=ledger
    permanent=dict(state['pins']);milestone_keys={int(k):v for k,v in state['history']['milestones'].items()}
    update=torch_update(encoder,decoder,optimizer,device=config['device'])
    class StopAtBoundary(Exception):pass
    def save(current,account):
        nonlocal latest,saved_ledger,initial,history
        effective=account.consumed-(account.pending[1] if account.debited else 0)
        current['weights']=dict(encoder=encoder.state_dict(),decoder=decoder.state_dict())
        current['optimizer']=optimizer.state_dict()
        current['rng']=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                            device=torch.cuda.get_rng_state_all() if config['device']=='cuda' else [])
        current['scheduler']['learner_seconds']=previous_seconds+time.monotonic()-began
        current['pins']=dict(permanent)
        current['history']['milestones']={str(k):v for k,v in milestone_keys.items()}
        latest=store.save(current,account);saved_ledger=account
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
        if stop_after_updates is not None and account.pending is None and account.updates>=stop_after_updates:
            raise StopAtBoundary()
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
        from .generalization_replay import restore_pool,ReplayBatches
        snapshot=state['replay'] if resume is not None else replay['pool'].snapshot()
        verified=restore_pool(snapshot,tasks=replay['tasks'],client=replay['client'],deadline=time.monotonic()+min(60,config['seconds']))
        state['replay']=verified.snapshot()
        batches=ReplayBatches(batches,verified,replay['tasks'],state)
    remaining=config['seconds']-previous_seconds
    try:
        save(state,ledger)
        result=train_updates(state,ledger,batches,update=update,save=save,seconds=remaining) if remaining>0 else dict(status='PARTIAL')
        status=result['status']
    except StopAtBoundary:status='PARTIAL'
    # These references are private full-state descriptors. Evaluation projects
    # only each descriptor's weights, never its optimizer/RNG or training data.
    from .identity import seal
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
    result=dict(status='PASS' if status=='COMPLETE' else 'PARTIAL',reason=status,compatibility=compatibility,
        checkpoint=latest,tokens=saved_ledger.consumed,updates=saved_ledger.updates,undershoot=saved_ledger.undershoot,
        milestones={str(m):state['history']['exposures'][k] for m,k in milestone_keys.items()},
        learner_seconds=state['scheduler']['learner_seconds'],data=data_identity,device=config['device'],counts=counts,
        retained=roster['identity'])
    write(output/'training.json',result)
    return result
