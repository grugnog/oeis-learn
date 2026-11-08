"""Bounded whole-run panel history, separate from the diagnostic event tail.

Keep the initial panel and newest panels. Merge the oldest interior bins when
full; their inclusive endpoints, panel counts and exact rational sums survive.
No interpolation between compacted endpoints is implied.
"""
import copy
import math
from fractions import Fraction

FIELDS=('loss_sum','loss_tokens','loss_completed','loss_censored','semantic_attempts','semantic_completed','semantic_success')
LIMIT=64


def empty_history(*,legacy_panels=0):
    return dict(version='pac-retention-history/v1',compaction='oldest-interior-exact-sums',
                omitted_legacy_panels=legacy_panels,bins=[])


def append_panel(history,coordinates,families):
    point=dict(coordinates=coordinates,families=families)
    totals={f:dict(loss_sum=str(Fraction(str(v['loss']['loss_sum']))) if v['loss'] is not None else '0',
        loss_tokens=v['loss']['tokens'] if v['loss'] is not None else 0,
        loss_completed=int(v['loss'] is not None),loss_censored=int(v['loss'] is None),
        **v['semantic']) for f,v in families.items()}
    history['bins'].append(dict(first=copy.deepcopy(point),last=copy.deepcopy(point),count=1,totals=totals))
    if len(history['bins'])>LIMIT:
        a,b=history['bins'][1:3]
        merged={f:{k:str(Fraction(a['totals'][f][k])+Fraction(b['totals'][f][k])) if k=='loss_sum'
                    else a['totals'][f][k]+b['totals'][f][k] for k in FIELDS} for f in families}
        history['bins'][1:3]=[dict(first=a['first'],last=b['last'],count=a['count']+b['count'],totals=merged)]


def validate_history(history,families,panels):
    if set(history)!={'version','compaction','omitted_legacy_panels','bins'} or history['version']!='pac-retention-history/v1' or history['compaction']!='oldest-interior-exact-sums':raise ValueError('retention history version')
    omitted=history['omitted_legacy_panels']
    if type(omitted) is not int or omitted<0 or len(history['bins'])>LIMIT:raise ValueError('retention history bounds')
    previous=-1;count=omitted
    for b in history['bins']:
        if set(b)!={'first','last','count','totals'} or type(b['count']) is not int or b['count']<1 or set(b['totals'])!=set(families):raise ValueError('retention bin shape')
        for point in (b['first'],b['last']):
            if set(point)!={'coordinates','families'} or set(point['families'])!=set(families):raise ValueError('retention point shape')
            coords=point['coordinates']
            if set(coords)!={'update','tokens','examples','seconds'} or any(type(coords[k]) is not int or coords[k]<0 for k in ('update','tokens','examples')):raise ValueError('retention coordinates')
            if coords['seconds'] is not None and (type(coords['seconds']) not in (int,float) or not math.isfinite(coords['seconds']) or coords['seconds']<0):raise ValueError('retention time')
            for v in point['families'].values():
                if set(v)!={'loss','semantic'}:raise ValueError('retention family shape')
                if v['loss'] is not None and (set(v['loss'])!={'loss_sum','tokens'} or not math.isfinite(v['loss']['loss_sum']) or v['loss']['loss_sum']<0 or type(v['loss']['tokens']) is not int or v['loss']['tokens']<=0):raise ValueError('retention loss')
                _semantic(v['semantic'])
        if not previous<b['first']['coordinates']['update']<=b['last']['coordinates']['update']:raise ValueError('retention order')
        previous=b['last']['coordinates']['update'];count+=b['count']
        for t in b['totals'].values():
            if set(t)!=set(FIELDS) or Fraction(t['loss_sum'])<0 or any(type(t[k]) is not int or t[k]<0 for k in FIELDS if k!='loss_sum'):raise ValueError('retention totals')
            if t['loss_completed']+t['loss_censored']!=b['count']:raise ValueError('retention panel coverage')
            _semantic({k:t[k] for k in FIELDS if k.startswith('semantic_')})
    if count!=panels:raise ValueError('retention history coverage')
    return history


def _semantic(v):
    if set(v)!={'semantic_attempts','semantic_completed','semantic_success'} or any(type(n) is not int or n<0 for n in v.values()) or not v['semantic_success']<=v['semantic_completed']<=v['semantic_attempts']:raise ValueError('retention semantic counts')


def deterministic_observer(snapshot):
    """Comparison-only projection: elapsed wall time cannot be replay-identical."""
    value=copy.deepcopy(snapshot)
    for b in value.get('history',{}).get('bins',[]):
        for key in ('first','last'):b[key]['coordinates']['seconds']=None
    return value
