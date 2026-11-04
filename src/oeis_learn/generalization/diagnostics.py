"""Candidate-only structure diagnostics; no hidden reference needed."""
from oeis_learn.synthesis.reference_parser import parse, SourceError
from oeis_learn.pipeline_validation.identity import digest


def signature(source):
    try:
        p=parse(source,allow_parameters=True)
    except (SourceError,ValueError,TypeError):
        return dict(parse_status='invalid',signature=None,parameters=None,source_hash=digest(source))
    params=set()
    def walk(n):
        if isinstance(n,str): return n
        if n[0]=='param.get': params.add(n[1]); return ('param.get',n[1])
        if n[0]=='int': return ('int','<literal>')
        return tuple(walk(x) for x in n)
    tree=walk(('program',*p.statements,('return',p.result)))
    return dict(parse_status='ok',signature=digest([p.declarations,tree]),parameters=len(params),source_hash=digest(source))


def join_attempts(attempts, fits):
    ids=[a['id'] for a in attempts]
    if len(set(ids))!=len(ids) or set(fits)-set(ids): raise ValueError('attempt identity mismatch')
    result=[]
    for attempt in attempts:
        attrs=signature(attempt.get('source'))
        fit=fits.get(attempt['id'],dict(status='unavailable',reason='historical_field_missing'))
        result.append(dict(attempt,**attrs,duplicate_group=attrs['source_hash'],fit=fit,
                           final_source_hash=digest(fit['final_source']) if fit.get('final_source') is not None else None,
                           first_error=fit.get('reason'),queue_seconds=fit.get('queue_seconds'),
                           service_seconds=fit.get('service_seconds'),cleanup_seconds=fit.get('cleanup_seconds')))
    return tuple(result)


def target_diagnostics(attempts, truth, *, frozen_selection, authorized_selection):
    if not frozen_selection or frozen_selection!=authorized_selection:
        raise ValueError('target diagnostics require frozen assessment authorization')
    return tuple(dict(id=a['id'],finite_match=tuple(a.get('values',()))==tuple(truth)) for a in attempts)


class FeedbackCoverage:
    """Persistent evidence stops, separate from kernel/review mathematical state."""
    VERSION='pac-feedback-coverage/v1'
    def __init__(self,families):
        if not families or len(set(families))!=len(families):raise ValueError('distinct feedback families')
        self.families=list(families);self.censored_rewards=0;self.invalid_review_blocks=0
        self.missing_rotations={f:0 for f in families};self.retention_incomplete=set();self.pauses=[]
    def _pause(self,reason):
        if reason not in self.pauses:self.pauses.append(reason)
    def reward(self,*,complete,budget_exhausted=False):
        if type(complete) is not bool or type(budget_exhausted) is not bool:raise ValueError('explicit reward availability required')
        self.censored_rewards=0 if complete else self.censored_rewards+1
        if self.censored_rewards>=2:self._pause('censored_rewards')
        if budget_exhausted:self._pause('required_reward_budget')
    def review_block(self,*,eligible,valid):
        from .budgets import integer
        integer(eligible);integer(valid)
        if not valid<=eligible<=8:raise ValueError('review coverage counts')
        self.invalid_review_blocks=self.invalid_review_blocks+1 if eligible and not valid else 0
        if self.invalid_review_blocks>=2:self._pause('invalid_review_grades')
    def semantic_rotation(self,completed):
        if set(completed)!=set(self.families) or any(type(v) is not bool for v in completed.values()):raise ValueError('complete semantic rotation coverage required')
        for family,valid in completed.items():
            self.missing_rotations[family]=0 if valid else self.missing_rotations[family]+1
            if self.missing_rotations[family]>=2:self.retention_incomplete.add(family)
    def snapshot(self):
        return dict(version=self.VERSION,families=list(self.families),censored_rewards=self.censored_rewards,
                    invalid_review_blocks=self.invalid_review_blocks,missing_rotations=dict(self.missing_rotations),
                    retention_incomplete=sorted(self.retention_incomplete),pauses=list(self.pauses))
    @classmethod
    def restore(cls,state,*,families):
        from .budgets import integer
        if set(state)!={'version','families','censored_rewards','invalid_review_blocks','missing_rotations','retention_incomplete','pauses'} or state['version']!=cls.VERSION or state['families']!=list(families):raise ValueError('feedback coverage identity')
        for v in (state['censored_rewards'],state['invalid_review_blocks'],*state['missing_rotations'].values()):integer(v)
        if set(state['missing_rotations'])!=set(families) or not set(state['retention_incomplete'])<=set(families) or len(set(state['pauses']))!=len(state['pauses']) or not set(state['pauses'])<={'censored_rewards','required_reward_budget','invalid_review_grades'}:raise ValueError('coverage state')
        if state['censored_rewards']>=2 and 'censored_rewards' not in state['pauses'] or state['invalid_review_blocks']>=2 and 'invalid_review_grades' not in state['pauses']:
            raise ValueError('lost persistent coverage pause')
        if any(n>=2 and f not in state['retention_incomplete'] for f,n in state['missing_rotations'].items()):raise ValueError('lost retention incompleteness')
        obj=cls(families);obj.censored_rewards=state['censored_rewards'];obj.invalid_review_blocks=state['invalid_review_blocks']
        obj.missing_rotations=dict(state['missing_rotations']);obj.retention_incomplete=set(state['retention_incomplete']);obj.pauses=list(state['pauses']);return obj


def observer_schedule(completed_updates,*,arm,families,anchors,seed,final=False):
    """Frozen opportunities, not fabricated observed values or extra gradients."""
    import random
    from .budgets import integer
    from .curriculum import stream_seed
    integer(completed_updates)
    if arm not in {'B','C','D'} or len(families)!=32 or len(set(families))!=32 or len(anchors)!=128 or len(set(anchors))!=128 or type(final) is not bool:
        raise ValueError('frozen PAC observer inventory')
    common=None
    if completed_updates%256==0 or final:
        panel=(completed_updates+255)//256
        start=(panel*4)%32
        common=dict(loss_probes=256,semantic_families=list(families[start:start+4]),semantic_probes=32,
                    attempts_per_probe=1,seconds_per_probe=5,kernel_credit=False,anchor_promotion=False)
    cycle,pos=divmod(completed_updates,32)
    reward=None;grade=None
    if not final and pos in ((0,8,16,24) if arm=='D' else (0,8,16)):
        number=cycle*(4 if arm=='D' else 3)+pos//8
        family=families[random.Random(stream_seed(seed,'shadow-uniform-family',str(number))).randrange(32)] if arm=='B' else None
        reward=dict(block=number,family=family,pre_update=completed_updates,post_update=completed_updates+8,
                    probes_per_evaluation=2,shadow=arm=='B',policy_required=arm!='B',selection_bound=arm!='B')
    if not final and arm!='D' and pos>=24:
        grade=dict(anchor=anchors[(cycle*8+pos-24)%128] if arm=='B' else None,shadow=arm=='B',
                   changes_gradient_order=False if arm=='B' else True,requires_acquired_anchor=True)
    return dict(common=common,reward_start=reward,grade_opportunity=grade)


def observation_forecast(updates,*,arm):
    """Reserve all started blocks plus common initial/final panels and restoration."""
    from .budgets import integer
    integer(updates)
    if arm not in {'B','C','D'}:raise ValueError('PAC observation arm')
    cycles,remainder=divmod(updates,32)
    blocks=(updates+7)//8 if arm=='D' else cycles*3+(min(remainder,24)+7)//8
    grades=0 if arm=='D' else cycles*8+max(0,remainder-24)
    common=1+updates//256+int(updates%256!=0)
    return dict(reward_blocks=blocks,reward_probe_evaluations=4*blocks,grade_opportunities=grades,
                common_loss_probes=common*256,semantic_attempts=common*32,semantic_seconds=common*32*5,
                mode_rng_restorations=2*blocks+grades+2*common,publication_events=common,
                required_reward=arm!='B',costs_equal_across_arms=False)
