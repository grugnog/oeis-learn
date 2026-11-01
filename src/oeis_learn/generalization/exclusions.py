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


def _keys(r,window):
    return dict(families={r[2]},structural={r[2]} if r[3]=='structural' else set(),
                source={r[4]},group={r[6]},numeric_variant={(r[2],r[7])},
                finite_window={r[5][i:i+window] for i in range(101-window)})


def _reasons(r,keys,roles):
    reasons=set()
    for role,old in roles.items():
        if role==r[1]:continue
        families=old['families'] if r[3]=='structural' else old['structural']
        if r[2] in families:reasons.add('structural_family')
        for name in ('source','group','numeric_variant','finite_window'):
            if not keys[name].isdisjoint(old[name]):reasons.add(name)
    return tuple(sorted(reasons))


@dataclass(frozen=True,init=False)
class ExclusionIndex:
    records: tuple
    window: int
    scope: str = 'bounded finite windows; not semantic equivalence'

    def __init__(self, rows=(), *, window=20):
        from types import MappingProxyType
        if type(window) is not int or not 20<=window<=100:raise ValueError('window limit')
        object.__setattr__(self,'window',window)
        object.__setattr__(self,'records',())
        object.__setattr__(self,'scope','bounded finite windows; not semantic equivalence')
        object.__setattr__(self,'_roles',MappingProxyType({}))
        complete=self.add_many(rows)
        object.__setattr__(self,'records',complete.records)
        object.__setattr__(self,'_roles',complete._roles)

    def reasons(self,row):
        r=_normalized(row)
        return _reasons(r,_keys(r,self.window),self._roles)

    def add_many(self,rows):
        """Atomically extend a frozen index, copying each changed role once.

        All finite windows remain exact tuples, not probabilistic fingerprints.
        Same-role overlap is permitted; every cross-role pair is checked, including
        pairs introduced in this batch. A failed batch leaves its parent intact.
        """
        from types import MappingProxyType
        roles=dict(self._roles);changed=set();added=[]
        for row in rows:
            r=_normalized(row);keys=_keys(r,self.window)
            if _reasons(r,keys,roles):raise ValueError('cross-role exclusion')
            role=r[1]
            if role not in changed:
                roles[role]={k:set(v) for k,v in roles[role].items()} if role in roles else {k:set() for k in keys}
                changed.add(role)
            for key,value in keys.items():roles[role][key].update(value)
            added.append(r)
        if not added:return self
        for role in changed:roles[role]=MappingProxyType({k:frozenset(v) for k,v in roles[role].items()})
        new=object.__new__(ExclusionIndex)
        object.__setattr__(new,'records',self.records+tuple(added))
        object.__setattr__(new,'window',self.window)
        object.__setattr__(new,'scope',self.scope)
        object.__setattr__(new,'_roles',MappingProxyType(roles))
        return new

    def add(self,row):return self.add_many((row,))
