"""Bounded deterministic marginal quota selection and least-privilege projection."""
from dataclasses import dataclass
import json
from .contracts import shape

@dataclass(frozen=True)
class CohortChoice:
    payload: bytes
    shortfalls: tuple[str,...]

    @property
    def rows(self): return tuple(json.loads(self.payload))
    @property
    def complete(self): return not self.shortfalls


def choose(rows, *, count=128, quotas, structural=False, draw_cap=1_000_000):
    if type(count) is not int or not 1<=count<=32768 or not 1<=draw_cap<=1_000_000 or any(type(v) is not int or not 0<=v<=count for v in quotas.values()):
        raise ValueError('invalid frozen quota')
    candidates=[]; seen=set()
    for i,row in enumerate(rows):
        if i>=draw_cap: break
        if row['id'] in seen: raise ValueError('duplicate cohort row')
        seen.add(row['id']); candidates.append(row)
    candidates.sort(key=lambda r:r['id'])
    chosen=[]; counts={k:0 for k in quotas}; families={k:set() for k in ('loop_active','branch_active')}
    # Deterministic bounded greedy feasibility; a shortfall is explicit even if an
    # alternate search could fill it. Never silently relax a marginal or family quota.
    while candidates and len(chosen)<count:
        def gain(r):
            return sum(k in r['tags'] and counts[k]<v for k,v in quotas.items())+sum(structural and k in r['tags'] and len(fs)<2 and r['family'] not in fs for k,fs in families.items())
        best=max(range(len(candidates)),key=lambda i:gain(candidates[i]))
        row=candidates.pop(best); chosen.append(row)
        for k in counts: counts[k]+=k in row['tags']
        for k in families:
            if k in row['tags']: families[k].add(row['family'])
    shortfalls=[]
    if len(chosen)<count: shortfalls.append(f'rows:{count-len(chosen)}')
    shortfalls.extend(f'{k}:{v-counts[k]}' for k,v in sorted(quotas.items()) if counts[k]<v)
    if structural: shortfalls.extend(f'{k}_families:{2-len(fs)}' for k,fs in families.items() if len(fs)<2)
    return CohortChoice(json.dumps(chosen,sort_keys=True).encode(),tuple(shortfalls))


def project(row):
    if row['role'] not in {'training','selection','assessment'} or len(row['values'])!=100 or len(row['indices'])!=100:
        raise ValueError('invalid role/horizon')
    visible={k:row[k] for k in ('id',)}
    visible.update(indices=list(row['indices'][:20]),values=list(row['values'][:20]))
    shape('visible-prompt',visible)
    # Scorer/training payload stays in its own role mount. Opaque ids must already
    # have been assigned by preparation, never encode protected family labels.
    protected={k:row[k] for k in ('id','role','indices','values','source')}
    if row['role']=='training':
        from oeis_learn.decoder.bigint_codec import encode_body
        protected['tokens']=encode_body(row['source'])
        shape('training-target',protected)
    else:
        if row.get('holdout') not in {'parameter','structural'}:raise ValueError('registered protected stratum required')
        protected.update(family=row['family'],stratum=row['holdout'])
        shape('scorer-target',protected)
    return visible,protected


def publish(choices, preparation, proposed, manifests, *, root):
    from .contracts import resolve_study
    shortfalls=tuple(f'{cell}:{reason}' for cell,choice in choices.items() for reason in choice.shortfalls)
    if not choices: shortfalls=('missing cells',)
    return resolve_study(preparation,proposed,manifests,root=root,shortfalls=shortfalls)
