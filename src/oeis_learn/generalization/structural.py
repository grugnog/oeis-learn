"""Versioned pure structural admission; original custodian I/O remains in 011/C.

Inputs to admit are custodian-verified observations, not certificates that arbitrary
caller-provided dictionaries came from native execution. C binds their receipts.
"""
from collections import Counter
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal

def legacy():
    from oeis_learn.generalization import structural_draws
    return structural_draws


def prior_protocol():
    from importlib.resources import files
    return json.loads(files('oeis_learn.generalization').joinpath('policies/structural-protocol.json').read_text())


def design(prior):
    original=legacy().design(prior)
    result={k:v for k,v in original.items() if k!='identity'}
    result.update(version='pac-structural-scope/v1',parent_design=original['identity'],
                  cells=[c for c in original['cells'] if c['stratum']=='structural'],
                  identity_capacity=8192,parameter='DEFERRED',assessment='EXCLUSION_ONLY')
    return seal(result)


def verify(protocol):
    expected=design(prior_protocol())
    if protocol!=expected:raise ValueError('changed frozen structural draw/control policy')


@lru_cache(maxsize=1)
def _draw_bytes(identity):
    protocol=design(prior_protocol())
    return json.dumps(legacy().draw(protocol))


def draw(protocol):
    verify(protocol)
    rows=json.loads(_draw_bytes(protocol['identity']))
    if len(rows)!=2288:raise ValueError('structural draw count changed')
    return rows


def original_cells():
    cells=[dict(name=f'{role}-{holdout}',role=role,holdout=holdout,count=128)
           for role in ('selection','assessment') for holdout in ('parameter','structural')]
    cells.extend(dict(name='static' if i==0 else f'refresh-{i:02}',role='training',holdout='parameter',count=32768) for i in range(12))
    return cells


class QuotaDeficit(ValueError):
    retryable=False
    def __init__(self,deficits):
        self.deficits=deficits
        super().__init__('completed post-exclusion quota deficit; no timeout/reseed retry')


def admit(protocol,draws,candidates,results,*,exclusion_cells,rejected_ids):
    """Freeze exact family quotas after exclusions and full-horizon agreement."""
    expected=draw(protocol)
    if draws!=expected:raise ValueError('original draw identity/order changed')
    wanted=original_cells()
    if (len(exclusion_cells)!=16 or sorted(exclusion_cells,key=lambda x:x['name'])!=sorted(wanted,key=lambda x:x['name'])):
        raise ValueError('all 16 original exclusion cells required')
    by_id={r['id']:r for r in expected}
    if len(by_id)!=len(expected) or not set(rejected_ids)<=set(by_id):raise ValueError('exclusion draw identity')
    seen=set();sources=set();survivors=[]
    for row in candidates:
        key=row['id']
        if key in seen or key not in by_id or row!=by_id[key]:raise ValueError('candidate draw identity')
        seen.add(key)
        if key in rejected_ids:continue
        outcome=results.get(key)
        if not outcome or set(outcome)!={'reference','native'}:raise ValueError('native agreement missing')
        a,b=outcome['reference'],outcome['native']
        if len(a)!=100 or a!=b:raise ValueError('native/reference disagreement')
        from oeis_learn.synthesis.contracts import canonical_integer
        for value in a:canonical_integer(value,bits=16384)
        if row['source'] in sources:continue
        sources.add(row['source']);survivors.append(row)
    counts=Counter(r['family'] for r in survivors)
    deficits={c['family']:c['count']-counts[c['family']] for c in protocol['cells'] if counts[c['family']]<c['count']}
    if deficits:raise QuotaDeficit(deficits)
    selected=[]
    for cell in protocol['cells']:
        selected.extend(sorted((r for r in survivors if r['family']==cell['family']),key=lambda r:r['id'])[:cell['count']])
    roster=legacy().roster(selected,protocol)
    categories=Counter(r['category'] for r in roster.values())
    families={r['family'] for r in roster.values()}
    certified={r['family'] for r in roster.values() if r['category']=='certified'}
    if len(roster)!=128 or len(families)!=27 or len(certified)!=18 or categories!=dict(certified=68,ambiguous=26,unresolved=34):
        raise ValueError('structural allocation changed')
    return roster
