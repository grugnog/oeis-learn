"""Persistent exclusions applied to preparation and every future refresh shard."""
from dataclasses import dataclass
from .contracts import digest
from oeis_learn.synthesis.contracts import canonical_integer


def _normalized(row):
    required={'id','role','family','holdout','source','values','group','variant'}
    if set(row)!=required or row['role'] not in {'training','selection','assessment'} or row['holdout'] not in {'structural','parameter'}:
        raise ValueError('exclusion row authority')
    values=tuple(row['values'])
    if len(values)!=100: raise ValueError('100-term exclusion horizon required')
    for value in values: canonical_integer(value,bits=16384)
    return (row['id'],row['role'],row['family'],row['holdout'],digest(row['source']),values,row['group'],row['variant'])


@dataclass(frozen=True,init=False)
class ExclusionIndex:
    records: tuple
    window: int
    scope: str = 'bounded finite windows; not semantic equivalence'

    def __init__(self, rows=(), *, window=20):
        if type(window) is not int or not 20<=window<=100: raise ValueError('window limit')
        object.__setattr__(self,'window',window)
        object.__setattr__(self,'records',tuple(_normalized(row) for row in rows))
        object.__setattr__(self,'scope','bounded finite windows; not semantic equivalence')
        # Initial frozen manifests must also satisfy cross-role separation.
        for i,r in enumerate(self.records):
            if self._reasons(r,self.records[:i]): raise ValueError('initial cross-role collision')

    def _reasons(self,r,others):
        reasons=set(); windows={r[5][i:i+self.window] for i in range(101-self.window)}
        for old in others:
            if old[1]==r[1]: continue
            if (r[3]=='structural' or old[3]=='structural') and r[2]==old[2]: reasons.add('structural_family')
            if r[4]==old[4]: reasons.add('source')
            if r[6]==old[6]: reasons.add('group')
            if r[2]==old[2] and r[7]==old[7]: reasons.add('numeric_variant')
            if any(old[5][i:i+self.window] in windows for i in range(101-self.window)): reasons.add('finite_window')
        return tuple(sorted(reasons))

    def reasons(self,row): return self._reasons(_normalized(row),self.records)

    def add(self,row):
        r=_normalized(row)
        if self._reasons(r,self.records): raise ValueError('cross-role exclusion')
        new=object.__new__(ExclusionIndex)
        object.__setattr__(new,'records',self.records+(r,)); object.__setattr__(new,'window',self.window); object.__setattr__(new,'scope',self.scope)
        return new
