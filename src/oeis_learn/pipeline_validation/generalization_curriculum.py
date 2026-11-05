"""Small installed adapter: existing SFT owns gradients; pure PAC owns decisions."""
import copy
import time
from .identity import digest


def observe_targets(encoder,decoder,rows,*,device):
    """Token-weighted loss and exact target recall with full RNG/mode restoration."""
    import random
    import numpy as np
    import torch
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.rl.sft_trainer import teacher_forced_logits
    from oeis_learn.rl.token_loss import token_cross_entropy
    if not rows:raise ValueError('nonempty training observation')
    modes={m:m.training for model in (encoder,decoder) for m in model.modules()}
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if device=='cuda' else [])
    try:
        encoder.eval();decoder.eval()
        tokens=torch.full((len(rows),max(len(r['tokens']) for r in rows)+1),codec.PAD_ID,dtype=torch.long,device=device)
        for i,r in enumerate(rows):tokens[i,:len(r['tokens'])+1]=torch.tensor([codec.BOS_ID,*r['tokens']],device=device)
        with torch.no_grad():
            logits=teacher_forced_logits(encoder,decoder,[[int(v) for v in r['values']] for r in rows],tokens,device=device,pad_id=codec.PAD_ID)
            target=tokens[:,1:];mask=target.ne(codec.PAD_ID);losses=token_cross_entropy(logits,target)
            if not torch.isfinite(logits).all() or not torch.isfinite(losses[mask]).all():raise FloatingPointError('nonfinite PAC observation')
            value=dict(loss_sum=float(losses[mask].sum().item()),tokens=int(mask.sum().item()),
                recall=[bool(((logits.argmax(-1)==target)|~mask)[i].all().item()) for i in range(len(rows))])
        if device=='cuda':torch.cuda.synchronize()
        return value
    finally:
        for m,training in modes.items():m.training=training
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if device=='cuda':torch.cuda.set_rng_state_all(rng[3])


class PacController:
    def __init__(self,batches,encoder,decoder,*,device,seed,arm,cap,observation_seconds,snapshot=None,acceptance=False):
        from oeis_learn.generalization.curriculum_state import CurriculumState,AcceptanceCurriculumState
        Policy=AcceptanceCurriculumState if acceptance else CurriculumState
        self.batches=batches;self.encoder=encoder;self.decoder=decoder;self.device=device
        self.allowance=observation_seconds;self.observed_seconds=0.;self.observations=0
        self.channels=dict(bandit=0,review=0,fallback=0);self.last_event=None
        self.policy=Policy(batches.families,batches.probes,batches.anchors,seed=seed,arm=arm,
            token_cap=cap,inventory=batches.inventory['identity'],segment='initial')
        if snapshot is not None:
            if set(snapshot)!={'version','policy','observed_seconds','observations','channels','last_event','observer'} or snapshot['version']!='pac-installed-controller/v1':raise ValueError('complete PAC installed observer state required')
            self.policy=Policy.restore(snapshot['policy'],inventory=batches.inventory['identity'])
            if self.policy.cap!=cap or self.policy.arm!=arm or self.policy.seed!=seed:raise ValueError('PAC installed policy identity mismatch')
            self.observed_seconds=snapshot['observed_seconds'];self.observations=snapshot['observations'];self.channels=copy.deepcopy(snapshot['channels']);self.last_event=snapshot['last_event']
        _counters(self.observed_seconds,self.observations,self.channels,{'bandit','review','fallback'},self.policy.updates)
        from .generalization_diagnostics import PacObservers
        self.observer=PacObservers(batches,arm=arm,seed=seed,observe=self.observe,snapshot=snapshot['observer'] if snapshot else None,acceptance=acceptance)

    def snapshot(self):
        return dict(version='pac-installed-controller/v1',policy=self.policy.snapshot(),observed_seconds=self.observed_seconds,
            observations=self.observations,channels=dict(self.channels),last_event=copy.deepcopy(self.last_event),observer=self.observer.snapshot())

    def observe(self,rows):
        if self.observed_seconds>=self.allowance or time.monotonic()>=getattr(self,'deadline',float('inf')):return None
        started=time.monotonic()
        try:
            result=observe_targets(self.encoder,self.decoder,rows,device=self.device)
        except (TimeoutError,FloatingPointError):result=None
        finally:
            self.observed_seconds+=time.monotonic()-started;self.observations+=1
        return result if self.observed_seconds<=self.allowance else None

    def panel(self):
        p=self.policy;b=p.block
        result=self.observe([self.batches.probe_rows[i] for i in b['probe_ids']])
        return None if result is None else dict(role='training',block=b['id'],panel=digest(b['probe_ids']),probe_ids=b['probe_ids'],
            model_update=p.updates,loss_sum=result['loss_sum'],tokens=result['tokens'])

    def run(self,state,ledger,*,update,save,seconds,max_updates=None):
        from oeis_learn.generalization.budgets import charge_tokens
        from oeis_learn.generalization.review import review_slot
        deadline=time.monotonic()+seconds;self.deadline=deadline;p=self.policy
        def persist():
            state['scheduler']['pac']=self.snapshot();save(state,ledger)
        while time.monotonic()<deadline and not p.terminal and not p.pauses:
            # Complete pending feedback before even considering a fresh gradient.
            if p.block and p.block['phase']=='post':p.record_post(self.panel())
            if p.block and p.block['phase']=='feedback':
                self.last_event=p.finish_feedback(budget_exhausted=self.observed_seconds>=self.allowance);persist()
                if p.pauses:break
            if max_updates is not None and p.updates>=max_updates:break
            self.observer.before(p.updates)
            if not p.block and (p.arm=='D' or not p.review_slot(p.updates)):p.start_block()
            if p.block and p.block['phase']=='pre':p.record_pre(self.panel())
            if p.pending is None:p.plan_batch(self.batches.batch_size)
            pending=p.pending;batch=[self.batches.rows[i] for i in pending['rows']]
            tokens=charge_tokens([r['tokens'] for r in batch],bos=1,pad=0);key=digest(batch)
            state['data']['pending_batch']=batch
            if pending['phase']=='planned':
                grade=None;a=pending['anchor']
                if a and p.review.anchors[a]['acquired']:
                    observation=self.observe([self.batches.rows[a]])
                    if observation is not None:grade=dict(anchor=a,model_update=p.updates,recall=observation['recall'][0])
                if not p.reserve(tokens,grade=grade):
                    # Preserve an uncharged oversized pending batch for honest undershoot.
                    from dataclasses import replace
                    ledger=replace(ledger,pending=(key,tokens),debited=False);persist();break
            ledger=ledger.reserve(key,tokens);persist()
            if time.monotonic()>=deadline:break
            try:update(batch)
            except BaseException:
                p.commit(pending['event_id'],success=False);raise
            if self.device=='cuda':
                import torch
                torch.cuda.synchronize()
            p.commit(pending['event_id'],success=True)
            ledger=ledger.commit(key);state['data']=dict(cursor=p.updates,pending_batch=None)
            self.channels[pending['channel']]+=1;self.observer.after(p.updates);persist()
        if p.terminal:self.observer.common(p.updates,final=True)
        state['scheduler']['pac']=self.snapshot()
        return dict(status='COMPLETE' if p.terminal else 'PARTIAL',ledger=ledger,state=state)


def initial_weights_identity(encoder,decoder):
    """Stable tensor-content identity independent of pickle/container serialization."""
    import hashlib
    h=hashlib.sha256()
    for prefix,model in (('encoder',encoder),('decoder',decoder)):
        for name,tensor in sorted(model.state_dict().items()):
            h.update(digest([prefix,name,str(tensor.dtype),list(tensor.shape)]).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return 'sha256:'+h.hexdigest()


class PacFixedController:
    """Use the existing fixed batch loop; shadow observations cannot select rows."""
    observe=PacController.observe
    def __init__(self,batches,encoder,decoder,*,device,seed,arm,cap,observation_seconds,snapshot=None,acceptance=False):
        from .generalization_diagnostics import PacObservers
        self.batches=batches;self.encoder=encoder;self.decoder=decoder;self.device=device
        self.allowance=observation_seconds;self.observed_seconds=0.;self.observations=0;self.policy=None
        self.channels=dict(fixed=0)
        if snapshot:
            if set(snapshot)!={'version','observed_seconds','observations','channels','observer'} or snapshot['version']!='pac-fixed-controller/v1':raise ValueError('closed PAC fixed observer state')
            self.observed_seconds=snapshot['observed_seconds'];self.observations=snapshot['observations'];self.channels=copy.deepcopy(snapshot['channels'])
        _counters(self.observed_seconds,self.observations,self.channels,{'fixed'},sum(self.channels.values()))
        self.observer=PacObservers(batches,arm=arm,seed=seed,observe=self.observe,snapshot=snapshot['observer'] if snapshot else None,acceptance=acceptance)
    def snapshot(self):
        return dict(version='pac-fixed-controller/v1',observed_seconds=self.observed_seconds,observations=self.observations,
            channels=dict(self.channels),observer=self.observer.snapshot())
    def run(self,state,ledger,*,update,save,seconds,max_updates=None):
        from .generalization_training import train_updates
        self.deadline=time.monotonic()+seconds
        updates=ledger.updates
        def actual(batch):
            nonlocal updates
            self.observer.before(updates)
            value=update(batch);updates+=1;self.channels['fixed']+=1;self.observer.after(updates)
            return value
        def persist(s,l):
            s['scheduler']['pac']=self.snapshot();save(s,l)
        result=train_updates(state,ledger,self.batches,update=actual,save=persist,seconds=seconds,
            discarded_tokens=state['scheduler'].get('discarded_tokens',0),pac=True)
        if result['status']=='COMPLETE':self.observer.common(result['ledger'].updates,final=True)
        state['scheduler']['pac']=self.snapshot()
        return result


def semantic_probe(encoder,decoder,row,truth,*,device,seed,recipe,cache_root,seconds=5):
    """One model draw/visible fit and native continuation; training truth only."""
    import random
    import numpy as np
    import torch
    from oeis_learn.decoder.bigint_generation import GenerationRequest,sample_batch
    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.fresh.bigint_diagnostics import _seal
    from oeis_learn.fresh.bigint_worker import client
    deadline=time.monotonic()+seconds
    modes={m:m.training for model in (encoder,decoder) for m in model.modules()}
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if device=='cuda' else [])
    try:
        encoder.eval();decoder.eval()
        with torch.no_grad():
            memory=encoder.forward_from_sequences([[int(v) for v in row['values']]],device=device)
            proposal=sample_batch(decoder,[GenerationRequest(memory,seed,0,deadline)],batch_size=1)[0]
        if time.monotonic()>=deadline or not proposal['complete']:return None
        if proposal.get('source') is None:return False
        visible=VisiblePrompt(tuple(row['indices']),tuple(row['values']))
        with client(dict(recipe=recipe,contract_sha256=digest(['pac-training-probe',row['id']])),cache_root) as native:
            fitted=native.fit(proposal['source'],visible,policy=fitting_policy(proposal['source']),deadline=deadline)
            if not fitted.get('complete') or time.monotonic()>=deadline:return None
            chosen=_seal(fitted.get('admitted',[]),visible,cost=0)
            if chosen is None:return False
            value=native.execute(chosen['selected']['source'],truth['indices'],deadline=deadline)
            if value.get('outcome')!='ok' or time.monotonic()>=deadline:return None
            return value['values']==truth['values']
    except (OSError,RuntimeError,TimeoutError):return None
    finally:
        for m,training in modes.items():m.training=training
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if device=='cuda':torch.cuda.set_rng_state_all(rng[3]);torch.cuda.synchronize()


def attach_semantic(controller,recipe,output,*,seed):
    def semantic(row,truth,seconds):
        if controller.observed_seconds>=controller.allowance or time.monotonic()>=getattr(controller,'deadline',float('inf')):return None
        started=time.monotonic()
        try:
            result=semantic_probe(controller.encoder,controller.decoder,row,truth,device=controller.device,
                seed=int(digest([seed,row['id'],controller.observer.state['semantic_attempts']])[7:15],16),
                recipe=recipe,cache_root=output/'private/pac-semantic',seconds=min(seconds,controller.allowance-controller.observed_seconds,getattr(controller,'deadline',float('inf'))-time.monotonic()))
        finally:controller.observed_seconds+=time.monotonic()-started
        return result if controller.observed_seconds<=controller.allowance else None
    controller.observer.semantic=semantic


def _counters(seconds,observations,channels,keys,updates):
    from .identity import number
    from oeis_learn.generalization.budgets import integer
    number(seconds);integer(observations)
    if set(channels)!=keys:raise ValueError('closed PAC channel counters')
    for value in channels.values():integer(value)
    if sum(channels.values())!=updates:raise ValueError('PAC channel/model clock mismatch')
