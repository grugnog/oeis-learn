"""Pure model-bound PAC transactions; runtime tensors/serialization belong to C.

A successful commit is called only after the optimizer completes. Any failed
optimizer requires whole-generation restore; this module cannot repair tensors.
"""
import copy
import random
from .budgets import integer
from .curriculum import Exp3S,RewardScaler,FamilyStream,ProbeRotation,finite,stream_seed,rng_state,rng_restore
from .review import ReviewQueue,review_slot,review_choice
from .diagnostics import FeedbackCoverage
from oeis_learn.pipeline_validation.identity import digest


class CurriculumState:
    VERSION='pac-model-bound-state/v1'
    POLICY='pac-exp3s-leitner/v1'
    def __init__(self,families,probes,anchors,*,seed,arm,token_cap,inventory,segment):
        integer(token_cap,1);integer(seed)
        if arm not in {'C','D'} or not isinstance(segment,str) or not segment:raise ValueError('adaptive arm/segment required')
        if set(families)!=set(probes) or len({r for rows in probes.values() for r in rows})!=sum(map(len,probes.values())) or len({r for rows in families.values() for r in rows})!=sum(map(len,families.values())):
            raise ValueError('disjoint family row/probe inventory')
        if any(f not in families or a not in families[f] for a,f in anchors.items()):raise ValueError('anchors must be original family rows')
        if set(r for rows in families.values() for r in rows)&set(r for rows in probes.values() for r in rows):raise ValueError('gradient/probe overlap')
        if not isinstance(inventory,str) or len(inventory)!=71 or not inventory.startswith('sha256:') or any(c not in '0123456789abcdef' for c in inventory[7:]):raise ValueError('model-bound inventory identity')
        self.layout_identity=digest([families,probes,anchors])
        self.families=copy.deepcopy(families);self.probe_ids=copy.deepcopy(probes);self.anchor_ids=dict(anchors)
        self.seed=seed;self.arm=arm;self.cap=token_cap;self.inventory=inventory;self.segment=segment;self.segments=[segment]
        self.kernel=Exp3S(sorted(families),seed=stream_seed(seed,'bandit'))
        self.scaler=RewardScaler(seed=seed)
        self.streams={f:FamilyStream(f,rows,seed=seed) for f,rows in families.items()}
        self.probes={f:ProbeRotation(rows) for f,rows in probes.items()}
        self.review=ReviewQueue(anchors);self.coverage=FeedbackCoverage(sorted(families))
        self.fallback_rng=random.Random(stream_seed(seed,'review-fallback'))
        self.updates=0;self.charged_tokens=0;self.useful_tokens=0;self.discarded_tokens=0;self.charged_seconds=0.
        self.reservations=0;self.block=None;self.pending=None;self.pauses=[];self.terminal=False
        self.review_eligible=0;self.review_valid=0;self.last_observation=None

    def _active(self):
        if self.terminal or self.pauses:raise ValueError('paused/terminal policy cannot launch work')

    def start_block(self):
        self._active()
        if self.block or self.pending or self.updates%8 or self.arm=='C' and review_slot(self.updates):
            raise ValueError('one contiguous bandit block at its cycle boundary')
        selection=self.kernel.choose();family=selection['family']
        self.block=dict(id=f'{self.segment}:block:{selection["action"]}',selection=selection,family=family,
                        start_update=self.updates,probe_ids=self.probes[family].take(),pre=None,post=None,
                        completed=0,tokens=0,phase='pre')
        return copy.deepcopy(self.block)

    def _panel(self,value,update):
        if value is None:return None
        b=self.block
        if (set(value)!={'role','block','panel','probe_ids','model_update','loss_sum','tokens'} or value['role']!='training'
            or value['block']!=b['id'] or value['panel']!=digest(b['probe_ids']) or value['probe_ids']!=b['probe_ids'] or value['model_update']!=update):
            raise ValueError('same training panel/model/block binding required')
        integer(value['tokens'],1)
        if finite(value['loss_sum'])<0:raise ValueError('nonnegative token loss sum')
        return copy.deepcopy(value)

    def record_pre(self,value):
        self._active()
        if not self.block or self.block['phase']!='pre' or self.pending:raise ValueError('pending pre observation required')
        self.block['pre']=self._panel(value,self.updates);self.block['phase']='updates'

    def plan_batch(self,batch_size):
        self._active();integer(batch_size,1)
        if self.pending:raise ValueError('at most one pending learner update')
        if self.block:
            if self.block['phase']!='updates' or self.block['completed']>=8:raise ValueError('complete pending observation before gradients')
            choice=dict(family=self.block['family'],anchor=None,channel='bandit')
        else:
            if self.arm!='C':raise ValueError('start bandit block first')
            choice=review_choice(self.review,completed_updates=self.updates,families=sorted(self.families),rng=self.fallback_rng)
        family=choice['family'];anchor=choice['anchor']
        if batch_size>len(self.families[family]):raise ValueError('distinct batch capacity')
        rows=([anchor]+self.streams[family].take(batch_size-1,exclude={anchor}) if batch_size>1 else [anchor]) if anchor else self.streams[family].take(batch_size)
        self.reservations+=1
        self.pending=dict(event_id=f'{self.segment}:update:{self.reservations}',family=family,rows=rows,anchor=anchor,
                          channel=choice['channel'],model_before=self.updates,tokens=None,grade=None,phase='planned')
        return copy.deepcopy(self.pending)

    def reserve(self,tokens,*,grade=None):
        self._active();integer(tokens,1)
        if not self.pending or self.pending['phase']!='planned':raise ValueError('planned exact batch required')
        p=self.pending;a=p['anchor']
        if grade is not None:
            if (a is None or not self.review.anchors[a]['acquired'] or set(grade)!={'anchor','model_update','recall'}
                or grade['anchor']!=a or grade['model_update']!=self.updates or type(grade['recall']) is not bool):
                raise ValueError('pre-update acquired-anchor grade binding')
        if self.charged_tokens+tokens>self.cap:
            self.terminal=True;self.review.finish()
            self.last_observation=dict(status='terminal_partial',updates=self.block['completed'] if self.block else 0,
                                       charged_tokens=self.block['tokens'] if self.block else 0)
            return False
        p.update(tokens=tokens,grade=copy.deepcopy(grade),phase='reserved');self.charged_tokens+=tokens
        return True

    def commit(self,event_id,*,success,elapsed_seconds=0.):
        self._active()
        if not self.pending or self.pending['phase']!='reserved' or self.pending['event_id']!=event_id or type(success) is not bool:
            raise ValueError('one reserved update/event required')
        if finite(elapsed_seconds)<0:raise ValueError('nonnegative charged time')
        p=self.pending
        if not success:
            self.charged_seconds+=elapsed_seconds;p['phase']='failed'
            self.pauses.append('optimizer_failure_requires_restore');return
        new_review=ReviewQueue.restore(self.review.snapshot(),anchors=self.anchor_ids)
        acquired=bool(p['anchor'] and self.review.anchors[p['anchor']]['acquired'])
        new_review.commit(update=self.updates+1,event_id=event_id,designated=p['anchor'],
                          grade=p['grade']['recall'] if p['grade'] else None,
                          exposed=[r for r in p['rows'] if r in self.anchor_ids])
        self.updates+=1;self.useful_tokens+=p['tokens'];self.charged_seconds+=elapsed_seconds;self.review=new_review
        if self.block:
            self.block['completed']+=1;self.block['tokens']+=p['tokens']
            if self.block['completed']==8:self.block['phase']='post'
        else:
            self.review_eligible+=acquired;self.review_valid+=p['grade'] is not None
            if self.updates%32==0:
                self.coverage.review_block(eligible=self.review_eligible,valid=self.review_valid)
                self.review_eligible=self.review_valid=0;self.pauses=list(self.coverage.pauses)
        self.pending=None

    def record_post(self,value):
        self._active()
        if not self.block or self.block['phase']!='post' or self.pending:raise ValueError('complete eight-update block required')
        post=self._panel(value,self.updates)
        if post and self.block['pre'] and post['tokens']!=self.block['pre']['tokens']:raise ValueError('paired panel token counts changed')
        self.block['post']=post;self.block['phase']='feedback'

    def finish_feedback(self,*,budget_exhausted=False):
        self._active()
        if not self.block or self.block['phase']!='feedback' or self.pending:raise ValueError('pending block feedback required')
        b=self.block
        kernel=Exp3S.restore(self.kernel.snapshot());scaler=RewardScaler.restore(self.scaler.snapshot())
        coverage=FeedbackCoverage.restore(self.coverage.snapshot(),families=sorted(self.families))
        complete=b['pre'] is not None and b['post'] is not None
        if complete:
            gain=finite(b['pre']['loss_sum']/b['pre']['tokens']-b['post']['loss_sum']/b['post']['tokens'])
            result=scaler.observe(gain,b['tokens']);reward=result['reward']
        else:result=dict(status='censored',raw_gain=None,reward=None,charged_tokens=b['tokens']);reward=None
        kernel.observe(b['selection'],reward,event_id=f'feedback/{b["selection"]["action"]}',role='training')
        coverage.reward(complete=complete,budget_exhausted=budget_exhausted)
        self.kernel=kernel;self.scaler=scaler;self.coverage=coverage;self.pauses=list(coverage.pauses)
        self.last_observation=dict(result,block=b['id']);self.block=None
        return copy.deepcopy(self.last_observation)

    def snapshot(self):
        return dict(version=self.VERSION,policy=self.POLICY,families=copy.deepcopy(self.families),probes=copy.deepcopy(self.probe_ids),
                    anchors=dict(self.anchor_ids),seed=self.seed,arm=self.arm,cap=self.cap,inventory=self.inventory,layout_identity=self.layout_identity,segment=self.segment,segments=list(self.segments),
                    kernel=self.kernel.snapshot(),scaler=self.scaler.snapshot(),row_streams={f:s.snapshot() for f,s in self.streams.items()},
                    probe_streams={f:p.snapshot() for f,p in self.probes.items()},review=self.review.snapshot(),coverage=self.coverage.snapshot(),
                    fallback_rng=rng_state(self.fallback_rng),updates=self.updates,charged_tokens=self.charged_tokens,useful_tokens=self.useful_tokens,
                    discarded_tokens=self.discarded_tokens,charged_seconds=self.charged_seconds,reservations=self.reservations,
                    block=copy.deepcopy(self.block),pending=copy.deepcopy(self.pending),pauses=list(self.pauses),terminal=self.terminal,
                    review_eligible=self.review_eligible,review_valid=self.review_valid,last_observation=copy.deepcopy(self.last_observation))

    @classmethod
    def restore(cls,state,*,inventory):
        if state.get('version')!=cls.VERSION or state.get('policy')!=cls.POLICY or state.get('inventory')!=inventory:
            raise ValueError('complete named policy/inventory identity required')
        s=cls(state['families'],state['probes'],state['anchors'],seed=state['seed'],arm=state['arm'],token_cap=state['cap'],inventory=inventory,segment=state['segment'])
        if set(state)!=set(s.snapshot()) or state['layout_identity']!=s.layout_identity:raise ValueError('closed complete model-bound state/layout required')
        for key in ('updates','charged_tokens','useful_tokens','discarded_tokens','reservations','review_eligible','review_valid'):
            integer(state[key]);setattr(s,key,state[key])
        if finite(state['charged_seconds'])<0 or type(state['terminal']) is not bool:raise ValueError('state time/terminal')
        s.charged_seconds=state['charged_seconds'];s.terminal=state['terminal']
        if s.charged_tokens>s.cap or s.useful_tokens+s.discarded_tokens>s.charged_tokens or not s.review_valid<=s.review_eligible<=8:raise ValueError('state budget/coverage counters')
        s.kernel=Exp3S.restore(state['kernel']);s.scaler=RewardScaler.restore(state['scaler'])
        if s.kernel.families!=sorted(s.families) or s.kernel.eta!=.001 or s.scaler.seen!=s.kernel.valid_updates:raise ValueError('kernel/scaler identity or clock mismatch')
        if set(state['row_streams'])!=set(s.families) or set(state['probe_streams'])!=set(s.families):raise ValueError('complete family streams required')
        s.streams={f:FamilyStream.restore(state['row_streams'][f],family=f,rows=rows) for f,rows in s.families.items()}
        s.probes={f:ProbeRotation.restore(state['probe_streams'][f],probes=rows) for f,rows in s.probe_ids.items()}
        s.review=ReviewQueue.restore(state['review'],anchors=s.anchor_ids);s.coverage=FeedbackCoverage.restore(state['coverage'],families=sorted(s.families))
        s.fallback_rng=rng_restore(state['fallback_rng'])
        if not 1<=len(state['segments'])<=4096 or len(set(state['segments']))!=len(state['segments']) or state['segment'] not in state['segments'] or any(not isinstance(x,str) or not x for x in state['segments']):raise ValueError('recovery segment history')
        s.segments=list(state['segments'])
        for key in ('block','pending','pauses','last_observation'):setattr(s,key,copy.deepcopy(state[key]))
        if s.review.last_update!=s.updates or s.review.terminal!=s.terminal or not set(s.coverage.pauses)<=set(s.pauses):raise ValueError('model/queue/pause clock mismatch')
        if len(set(s.pauses))!=len(s.pauses) or not set(s.pauses)<={'optimizer_failure_requires_restore','censored_rewards','required_reward_budget','invalid_review_grades'}:raise ValueError('unknown pause')
        if (s.block is None)!=(s.kernel.pending is None):raise ValueError('pending kernel/block mismatch')
        if s.block:
            b=s.block
            if set(b)!={'id','selection','family','start_update','probe_ids','pre','post','completed','tokens','phase'} or b['selection']!=s.kernel.pending or b['family']!=b['selection']['family']:
                raise ValueError('pending block identity')
            for key in ('start_update','completed','tokens'):integer(b[key])
            if b['completed']>8 or b['start_update']+b['completed']!=s.updates or b['phase'] not in {'pre','updates','post','feedback'}:raise ValueError('pending block update/phase mismatch')
            if b['probe_ids'] not in [s.probe_ids[b['family']][i:i+2] for i in (0,2,4,6)]:raise ValueError('pending probe panel')
            if b['tokens']>s.useful_tokens or (b['completed']==0 and b['tokens']!=0) or b['tokens']<b['completed']:raise ValueError('block token debit mismatch')
            if b['start_update']%8 or s.arm=='C' and b['start_update']%32 not in (0,8,16):raise ValueError('block cycle boundary')
            if b['phase']=='pre' and (b['pre'] is not None or b['post'] is not None) or b['phase']!='feedback' and b['post'] is not None or b['phase']=='updates' and b['completed']>=8:raise ValueError('block observation phase')
            if b['pre'] is not None:s._panel(b['pre'],b['start_update'])
            if b['post'] is not None:s._panel(b['post'],s.updates)
            if b['phase'] in {'post','feedback'} and b['completed']!=8 or b['phase']=='pre' and b['completed']!=0:raise ValueError('pending phase/cursor')
        if s.pending:
            p=s.pending
            if set(p)!={'event_id','family','rows','anchor','channel','model_before','tokens','grade','phase'} or p['model_before']!=s.updates or p['phase'] not in {'planned','reserved','failed'} or p['family'] not in s.families:raise ValueError('pending learner update')
            if len(p['rows'])!=len(set(p['rows'])) or not p['rows'] or not set(p['rows'])<=set(s.families[p['family']]):raise ValueError('pending rows')
            if p['event_id']!=f'{s.segment}:update:{s.reservations}' or p['channel'] not in {'bandit','review','fallback'}:raise ValueError('pending event/channel identity')
            if p['phase']!='planned':integer(p['tokens'],1)
            elif p['tokens'] is not None:raise ValueError('unreserved batch has debit')
            if s.block and (p['channel']!='bandit' or p['family']!=s.block['family'] or s.block['phase']!='updates'):raise ValueError('intervening gradient')
            if s.block and p['anchor'] is not None:raise ValueError('bandit update cannot carry review anchor')
            if not s.block:
                if s.arm!='C' or not review_slot(s.updates) or p['channel']=='bandit':raise ValueError('review cycle mismatch')
                due=s.review.choose(s.updates+1)
                if p['anchor']!=due or (due is None)!=(p['channel']=='fallback'):raise ValueError('review obligation mismatch')
            if p['grade'] is not None:
                g=p['grade'];a=p['anchor']
                if a is None or not s.review.anchors[a]['acquired'] or set(g)!={'anchor','model_update','recall'} or g['anchor']!=a or g['model_update']!=s.updates or type(g['recall']) is not bool or p['phase']=='planned':raise ValueError('unbound pre-update review grade')
            if p['phase']=='failed' and 'optimizer_failure_requires_restore' not in s.pauses:raise ValueError('failed optimizer without pause')
        reserved=s.pending['tokens'] if s.pending and s.pending['phase'] in {'reserved','failed'} else 0
        if s.charged_tokens!=s.useful_tokens+s.discarded_tokens+reserved:raise ValueError('unreconciled token ledger')
        completed=s.updates//8 if s.arm=='D' else s.updates//32*3+min(3,s.updates%32//8)
        actions=completed+int(s.block is not None and s.block['completed']<8)
        if s.kernel.action_count!=actions or s.kernel.valid_updates>s.kernel.action_count-int(s.block is not None):raise ValueError('model/kernel action clock mismatch')
        return s
