"""Pure Graves Eq. 1 EXP3.S and reproducible PAC sampling; no learner imports."""
import copy
import math
import random
from .budgets import integer
from oeis_learn.pipeline_validation.identity import digest


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite number required')
    return value


def logsum(values):
    values=list(values)
    if not values:return -math.inf
    top=max(values)
    return top+math.log(math.fsum(math.exp(v-top) for v in values))


def rng_state(rng):return [rng.getstate()[0],list(rng.getstate()[1]),rng.getstate()[2]]


def rng_restore(state):
    if (not isinstance(state,(list,tuple)) or len(state)!=3 or state[0]!=3
        or not isinstance(state[1],(list,tuple)) or len(state[1])!=625 or state[2] is not None
        or any(type(v) is not int or v<0 for v in state[1])):raise ValueError('complete isolated RNG state required')
    rng=random.Random()
    try:rng.setstate((state[0],tuple(state[1]),state[2]))
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('invalid RNG state') from exc
    return rng


def stream_seed(seed,label,family=''):
    integer(seed)
    return int(digest(['pac-stream/v1',seed,label,family])[7:],16)


class Exp3S:
    VERSION='pac-graves-exp3s/v1'
    FIELDS={'version','families','epsilon','eta','beta','logweights','action_count','valid_updates','rng','pending','last_feedback'}

    def __init__(self,families,*,seed,epsilon=.25,eta=.001,beta=0):
        if not 2<=len(families)<=32 or len(set(families))!=len(families) or any(not isinstance(f,str) or not f for f in families):
            raise ValueError('K distinct ordered families, 2 <= K <= 32')
        if finite(epsilon)!=.25 or finite(eta)<=0 or finite(beta)!=0:raise ValueError('unsupported kernel hyperparameters')
        integer(seed)
        self.families=list(families);self.epsilon=epsilon;self.eta=eta;self.beta=beta
        self.logweights=[-math.log(len(families))]*len(families)
        self.action_count=0;self.valid_updates=0;self.rng=random.Random(seed)
        self.pending=None;self.last_feedback=None

    def weights(self):
        if len(self.logweights)!=len(self.families):raise ValueError('kernel dimension mismatch')
        for w in self.logweights:finite(w)
        # Subtract max first: do not lose log(K) when all inputs are -1e300.
        top=max(self.logweights);shifted=[w-top for w in self.logweights]
        normalizer=logsum(shifted)
        return [math.exp(w-normalizer) for w in shifted]

    def probabilities(self):
        return [(1-self.epsilon)*q+self.epsilon/len(self.families) for q in self.weights()]

    def choose(self):
        if self.pending is not None:raise ValueError('finish pending feedback before another action')
        probabilities=self.probabilities();u=self.rng.random();total=0.;index=len(probabilities)-1
        for i,p in enumerate(probabilities):
            total+=p
            if u<total:index=i;break
        self.action_count+=1
        self.pending=dict(action=self.action_count,family=self.families[index],probability=probabilities[index])
        return copy.deepcopy(self.pending)

    def observe(self,selection,reward,*,event_id,role):
        if role!='training' or self.pending is None or selection!=self.pending:
            raise ValueError('pending selection-time probability and training feedback required')
        if not isinstance(event_id,str) or not event_id or event_id==self.last_feedback:
            raise ValueError('unique feedback event required')
        if reward is not None and not -1<=finite(reward)<=1:raise ValueError('scaled reward range')
        if reward is not None:
            self.weights()  # validate every input before mutation
            top=max(self.logweights);v=[w-top for w in self.logweights]
            a=self.families.index(selection['family'])
            v[a]+=self.eta*reward/selection['probability']
            for w in v:finite(w)
            top=max(v);v=[w-top for w in v]
            alpha=1/(self.valid_updates+2);k=len(v)
            z=[logsum((math.log1p(-alpha)+v[i],math.log(alpha/(k-1))+logsum(v[:i]+v[i+1:]))) for i in range(k)]
            norm=logsum(z);self.logweights=[w-norm for w in z];self.valid_updates+=1
        self.last_feedback=event_id;self.pending=None

    def snapshot(self):
        return dict(version=self.VERSION,families=list(self.families),epsilon=self.epsilon,eta=self.eta,beta=self.beta,
                    logweights=list(self.logweights),action_count=self.action_count,valid_updates=self.valid_updates,
                    rng=rng_state(self.rng),pending=copy.deepcopy(self.pending),last_feedback=self.last_feedback)

    @classmethod
    def restore(cls,state):
        if set(state)!=cls.FIELDS or state['version']!=cls.VERSION:raise ValueError('named Graves kernel state required')
        k=cls(state['families'],seed=0,epsilon=state['epsilon'],eta=state['eta'],beta=state['beta'])
        integer(state['action_count']);integer(state['valid_updates'])
        if state['valid_updates']>state['action_count']:raise ValueError('kernel counter mismatch')
        k.logweights=list(state['logweights']);k.weights()
        k.action_count=state['action_count'];k.valid_updates=state['valid_updates'];k.rng=rng_restore(state['rng'])
        k.pending=copy.deepcopy(state['pending']);k.last_feedback=state['last_feedback']
        if k.last_feedback is not None and (not isinstance(k.last_feedback,str) or not k.last_feedback):raise ValueError('feedback identity')
        if k.pending is not None:
            p=k.pending
            if set(p)!={'action','family','probability'} or p['action']!=k.action_count or p['family'] not in k.families:
                raise ValueError('pending selected action')
            if finite(p['probability'])!=k.probabilities()[k.families.index(p['family'])]:raise ValueError('selection probability mismatch')
        return k
