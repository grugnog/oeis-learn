"""Exact, conservative debit-before-work accounting at update boundaries."""
from dataclasses import dataclass, asdict, replace


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
