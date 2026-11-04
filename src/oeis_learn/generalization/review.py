"""Bounded per-anchor Leitner policy; transitions require committed practice."""
import copy
from .budgets import integer


def review_slot(completed_updates):
    integer(completed_updates)
    return completed_updates%32>=24


class ReviewQueue:
    VERSION='pac-leitner-queue/v1'
    def __init__(self,anchors):
        if not 1<=len(anchors)<=128 or any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in anchors.items()):
            raise ValueError('bounded distinct anchor inventory required')
        self.anchors={k:dict(family=v,acquired=False,queue=0,last_exposure=None,due_update=0,last_grade=None,last_event=None)
                      for k,v in sorted(anchors.items())}
        self.last_update=0;self.services=0;self.terminal=False

    def choose(self,next_update):
        integer(next_update,1)
        if self.terminal:return None
        due=[(a['due_update'],key) for key,a in self.anchors.items() if a['due_update']<=next_update]
        return min(due)[1] if due else None

    def commit(self,*,update,event_id,designated,grade,exposed):
        integer(update,1)
        if self.terminal or update<=self.last_update or not isinstance(event_id,str) or not event_id:
            raise ValueError('new committed update required')
        exposed=list(exposed)
        if len(exposed)!=len(set(exposed)) or not set(exposed)<=set(self.anchors):raise ValueError('distinct known exposed anchors required')
        if grade is not None and type(grade) is not bool:raise ValueError('finite complete recall grade or missing')
        if designated is None and grade is not None:raise ValueError('incidental practice cannot supply a grade')
        if designated is not None:
            if designated not in exposed or designated!=self.choose(update):raise ValueError('grade belongs to earliest due designated anchor')
            if not self.anchors[designated]['acquired'] and grade is not None:raise ValueError('acquisition is not graded recall')
        changed=copy.deepcopy(self.anchors)
        for key in exposed:
            a=changed[key]
            if not a['acquired']:a['acquired']=True;a['queue']=0
            elif key==designated and grade is not None:a['queue']=min(4,a['queue']+1) if grade else 0
            a['last_exposure']=update;a['due_update']=update+512*2**a['queue']
            a['last_event']=event_id
            if key==designated:a['last_grade']=grade
        self.anchors=changed;self.last_update=update
        self.services+=designated is not None

    def finish(self):self.terminal=True
    def pending(self):
        return dict(anchors=len(self.anchors),unacquired=sum(not a['acquired'] for a in self.anchors.values()),
                    due=sum(a['due_update']<=self.last_update for a in self.anchors.values()),
                    bound_applies=not self.terminal,due_delay_bound=512,exposure_bound=8704)
    def snapshot(self):
        return dict(version=self.VERSION,anchors=copy.deepcopy(self.anchors),last_update=self.last_update,services=self.services,terminal=self.terminal)
    @classmethod
    def restore(cls,state,*,anchors):
        if set(state)!={'version','anchors','last_update','services','terminal'} or state['version']!=cls.VERSION or set(state['anchors'])!=set(anchors):raise ValueError('review inventory/policy mismatch')
        integer(state['last_update']);integer(state['services'])
        if type(state['terminal']) is not bool or state['services']>state['last_update']:raise ValueError('review counters')
        result=cls(anchors)
        fields={'family','acquired','queue','last_exposure','due_update','last_grade','last_event'}
        for key,a in state['anchors'].items():
            if set(a)!=fields or a['family']!=anchors[key] or type(a['acquired']) is not bool or type(a['queue']) is not int or not 0<=a['queue']<=4:raise ValueError('review anchor state')
            if a['last_grade'] is not None and type(a['last_grade']) is not bool:raise ValueError('review grade')
            if a['acquired']:
                integer(a['last_exposure'],1)
                if a['last_exposure']>state['last_update'] or a['due_update']!=a['last_exposure']+512*2**a['queue'] or not isinstance(a['last_event'],str):raise ValueError('committed anchor exposure')
            elif a!=result.anchors[key]:raise ValueError('unacquired anchor has invented history')
        result.anchors=copy.deepcopy(state['anchors']);result.last_update=state['last_update'];result.services=state['services'];result.terminal=state['terminal'];return result


def review_choice(queue,*,completed_updates,families,rng):
    """Uniform-family fallback consumes only its dedicated RNG, never EXP3.S."""
    if not review_slot(completed_updates) or queue.terminal:raise ValueError('active review slot required')
    if not families or len(set(families))!=len(families) or not {a['family'] for a in queue.anchors.values()}<=set(families):
        raise ValueError('complete review family inventory')
    anchor=queue.choose(completed_updates+1)
    if anchor is None:return dict(channel='fallback',anchor=None,family=families[rng.randrange(len(families))])
    return dict(channel='review',anchor=anchor,family=queue.anchors[anchor]['family'])
