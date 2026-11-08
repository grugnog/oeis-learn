"""Exact, conservative debit-before-work accounting at update boundaries."""
from dataclasses import dataclass, asdict, replace
import time


def remaining_seconds(deadline):
    left=deadline-time.monotonic()
    if left<=0:raise TimeoutError("enclosing deadline exhausted before starting work")
    return left



def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('bounded integer required')
    return value


def charge_tokens(targets, *, bos, pad):
    """Targets are post-substitution output sequences, never encoder inputs; EOS counts."""
    return sum(token not in (bos, pad) for row in targets for token in row)


@dataclass(frozen=True)
class Ledger:
    cap: int
    consumed: int = 0
    updates: int = 0
    pending: tuple[str, int] | None = None
    debited: bool = False

    def __post_init__(self):
        integer(self.cap, 1); integer(self.consumed); integer(self.updates)
        if self.consumed > self.cap or type(self.debited) is not bool:
            raise ValueError('invalid cumulative ledger')
        if self.pending is not None:
            if len(self.pending) != 2 or not isinstance(self.pending[0], str) or not self.pending[0]:
                raise ValueError('pending batch identity required')
            integer(self.pending[1], 1)
        if self.debited and (not self.pending or self.pending[1] > self.consumed):
            raise ValueError('unbacked debit')

    def reserve(self, batch, tokens):
        integer(tokens, 1)
        if self.pending:
            if self.pending != (batch, tokens):
                raise ValueError('pending batch must be retained')
            return self
        debit = self.consumed + tokens <= self.cap
        return replace(self, pending=(batch, tokens), debited=debit,
                       consumed=self.consumed + tokens if debit else self.consumed)

    def commit(self, batch):
        if not self.pending or batch != self.pending[0] or not self.debited:
            raise ValueError('no admitted update')
        return replace(self, pending=None, debited=False, updates=self.updates + 1)

    @property
    def exhausted(self):
        return self.consumed == self.cap or bool(self.pending and not self.debited)

    @property
    def undershoot(self):
        return self.cap - self.consumed

    def snapshot(self):
        return dict(version='generalization-ledger/v1', **asdict(self))

    @classmethod
    def restore(cls, snapshot, *, cap):
        data = dict(snapshot)
        if data.pop('version', None) != 'generalization-ledger/v1' or data.get('cap') != cap or set(data) != {'cap','consumed','updates','pending','debited'}:
            raise ValueError('incompatible ledger')
        if data['pending'] is not None:
            data['pending'] = tuple(data['pending'])
        return cls(**data)


def reserve_storage(snapshot_bytes, required_pins, available_bytes):
    required = integer(snapshot_bytes, 1) * integer(required_pins, 1)
    if required > integer(available_bytes):
        raise ValueError('insufficient checkpoint storage; required pins cannot be pruned')
    return required


def _seconds(value,*,positive=False):
    from oeis_learn.pipeline_validation.identity import number
    return number(value,positive=positive)


def pac_work_allowance(kind,*,forecast,startup,publication,existing):
    floors={'preparation':86400,'learner':43200,'generation':23400}
    if kind not in floors:raise ValueError('PAC work kind')
    for value in (startup,publication,existing):_seconds(value)
    gaps=[]
    if forecast is None:
        gaps=['forecast'];calculated=floors[kind]
    else:calculated=max(floors[kind],3*_seconds(forecast,positive=True)+7200+startup+publication)
    _seconds(calculated)
    return dict(seconds=max(existing,calculated),calibration_gaps=gaps,scientific_portfolio_seconds=60,semantic_probe_seconds=5)


def pac_ancillary_allowance(name,*,observations,overhead,existing):
    if not isinstance(name,str) or not name:raise ValueError('named ancillary reservation')
    _seconds(overhead);_seconds(existing)
    for value in observations:_seconds(value,positive=True)
    if not observations and not existing:raise ValueError('missing measurement and prior justified allowance')
    seconds=max(existing,3*max(observations)+overhead) if observations else existing
    _seconds(seconds)
    return dict(name=name,seconds=seconds,
                calibration_gaps=[] if observations else [name],observations=len(observations))


def pac_deadline_graph(nodes,*,root):
    """Reject inner clipping even when the campaign has ample outer headroom."""
    fields={'requested','effective','work','children','mode','cleanup','cleanup_floor','finalization','overhead','contention','schema_max'}
    visiting=set();done={}
    def visit(name):
        if name in visiting:raise ValueError('cyclic deadline graph')
        if name in done:return done[name]
        if name not in nodes or set(nodes[name])!=fields:raise ValueError('closed complete deadline node')
        n=nodes[name]
        for key in fields-{'children','mode','schema_max'}:_seconds(n[key])
        if n['mode'] not in {'serial','parallel'} or not isinstance(n['children'],list) or len(set(n['children']))!=len(n['children']):raise ValueError('explicit child critical path required')
        if n['cleanup']<n['cleanup_floor'] or name==root and n['finalization']<3600:raise ValueError('protected cleanup/finalization floor')
        if n['effective']<n['requested']:raise ValueError('clipped effective inner deadline')
        if n['schema_max'] is not None:
            _seconds(n['schema_max'],positive=True)
            if n['effective']>n['schema_max']:raise ValueError('short inner/schema cap')
        visiting.add(name)
        children=[visit(child)['effective'] for child in n['children']]
        nested=sum(children) if n['mode']=='serial' else max(children,default=0)
        required=n['work']+nested+n['overhead']+n['contention']+n['cleanup']+n['finalization']
        _seconds(required)
        if n['requested']<required:raise ValueError('parent cannot enclose every child reservation')
        visiting.remove(name)
        done[name]=dict(required=required,requested=n['requested'],effective=n['effective'],schema_max=n['schema_max'])
        return done[name]
    result=visit(root)
    if set(done)!=set(nodes):raise ValueError('unaccounted deadline stages')
    return dict(version='pac-deadline-graph/v1',seconds=result['effective'],nodes=done,automatic_extension=False)
