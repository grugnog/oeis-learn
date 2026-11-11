"""Deterministic native reservoir projection into finite learner support.

This module never generates replacements or inspects evaluation outcomes. A
prospective rule, the complete original reservoir and all omissions are retained.
Only training data determine the refresh cadence; no measured model scores do.
"""
from collections import Counter
from math import ceil

from .contracts import digest, seal
from .training_rows import validate_row


def policy(*, seed, shard_rows=256, max_shards=64, target_tokens=25000000,
           batches=(16, 32, 64)):
    return seal(dict(version='progressive-support-projection-policy/v1', seed=seed,
        shard_rows=shard_rows, max_shards=max_shards, target_tokens=target_tokens,
        batches=list(batches), allocation_block=8, original_refresh_updates=8,
        cadence_rule='smallest multiple of eight covering the complete horizon',
        probe_rule='first counter-ordered row per bin; exclude exact finite overlaps',
        shard_rule='counter-ordered frontier structures, one new structure per refresh',
        minimum_refreshes=1))


def _policy(value):
    try:
        expected=policy(**{k:value[k] for k in ('seed','shard_rows','max_shards','target_tokens','batches')})
    except (KeyError,TypeError) as exc:
        raise ValueError('closed prospective support rule required') from exc
    if expected != value:
        raise ValueError('closed prospective support rule required')
    for key,maximum in [('seed',2**32-1),('shard_rows',4096),('max_shards',64),('target_tokens',400000000)]:
        if type(value[key]) is not int or not 1 <= value[key] <= maximum:
            raise ValueError('bounded support policy required: '+key)
    if not value['batches'] or value['batches'] != sorted(set(value['batches'])) or any(
            type(x) is not int or x not in (16,32,64) for x in value['batches']):
        raise ValueError('registered effective batches required')
    if value['shard_rows'] < 4 or value['max_shards'] < 2:
        raise ValueError('two bins, separate probes and refreshed support required')
    return value


def bin_name(row):
    return 'frontier' if {'loop','select'} & set(row['operations']) else 'familiar'


def _windows(row):
    return {digest(row['values'][i:i+20]) for i in range(81)}


def project(reservoir, rule, *, diagnostic=False):
    rule=_policy(rule)
    if type(diagnostic) is not bool or not isinstance(reservoir,list) or len(reservoir)>65536:
        raise ValueError('bounded original reservoir and explicit purpose required')
    ordered=[]; seen=set()
    for original in reservoir:
        if seal(original)!=original:
            raise ValueError('changed original training reservoir row')
        row=dict(original)
        if 'id' in row:
            if row.pop('id')!='mechanical:'+row['draw_identity']:
                raise ValueError('original mechanical identity differs')
            row=seal(row)
        validate_row(row,native_verified=not diagnostic)
        if row['draw_identity'] in seen:
            raise ValueError('duplicate original training draw')
        seen.add(row['draw_identity']); ordered.append(row)
    # A counter order, not runtime completion order or any learned score.
    ordered.sort(key=lambda r:(r['ordinal'],r['draw_identity']))
    bins={name:[r for r in ordered if bin_name(r)==name] for name in ('familiar','frontier')}
    omissions=Counter(); deficits=[]
    probes={name:rows[0] for name,rows in bins.items() if rows}
    if len(probes)!=2:
        deficits.append('training_probe_bins_missing')
    probe_windows=set().union(*(_windows(r) for r in probes.values()))
    gradient=[]
    for row in ordered:
        if row['draw_identity'] in {r['draw_identity'] for r in probes.values()}:
            continue
        if _windows(row) & probe_windows:
            omissions['probe_overlap']+=1
        else:
            gradient.append(row)
    familiar=[r for r in gradient if bin_name(r)=='familiar']
    frontier=[r for r in gradient if bin_name(r)=='frontier']
    # Hold every canonical structure in a single shard. Refreshes therefore
    # introduce new structural support, rather than new constants on old trees.
    groups={}
    for row in frontier:
        groups.setdefault(row['structure'],[]).append(row)
    slots=min(rule['max_shards'],len(groups))
    if not familiar or slots<2:
        deficits.append('distinct_frontier_and_familiar_support_missing')
    shards=[]
    if familiar and slots>=2 and len(probes)==2:
        assigned=[[] for _ in range(slots)]
        for i,rows in enumerate(groups.values()):
            assigned[i%slots].extend(rows)
        # Initial support includes both probes and both gradient bins. Familiar
        # support always remains shard zero, as required by the installed learner.
        first_familiar=familiar[:max(1,(rule['shard_rows']-2)//2)]
        omissions['familiar_capacity']+=len(familiar)-len(first_familiar)
        for i,rows in enumerate(assigned):
            prefix=[*probes.values(),*first_familiar] if i==0 else []
            keep=rule['shard_rows']-len(prefix)
            shards.append(prefix+rows[:keep])
            omissions['frontier_capacity']+=max(0,len(rows)-keep)
    minimum=min((len(r['tokens']) for s in shards for r in s),default=0)
    maximum=max((len(r['tokens']) for s in shards for r in s),default=0)
    cadence=None; bounds=[]
    if shards:
        updates=ceil(rule['target_tokens']/(minimum*min(rule['batches'])))
        cadence=max(8,8*ceil(updates/(len(shards)*8)))
        for batch in rule['batches']:
            bound=ceil(rule['target_tokens']/(minimum*batch))
            first_refresh_tokens=cadence*maximum*batch
            bounds.append(dict(batch=batch,maximum_updates=bound,
                required_shards=ceil(bound/cadence),
                first_refresh_token_upper_bound=first_refresh_tokens))
            if first_refresh_tokens>=rule['target_tokens']:
                deficits.append('refresh_not_guaranteed_before_target:batch'+str(batch))
    layout=dict(bins=['familiar','frontier'],probes={
        name:dict(role='training',id=row['draw_identity']) for name,row in probes.items()})
    data=seal(dict(version='progressive-prepared-support/v1', policy=rule['identity'],
        diagnostic=diagnostic, shards=shards,layout=layout,refresh_updates=cadence))
    report=seal(dict(version='progressive-support-publication/v1',
        status='INCOMPLETE' if deficits else 'COMPLETE', policy=rule, reservoir=digest(reservoir),
        support=data['identity'], diagnostic=diagnostic, original_draws=len(reservoir),
        selected_draws=sum(map(len,shards)), omissions=dict(omissions),
        probes=len(probes), distinct_frontier_structures=len(groups), available_shards=len(shards),
        minimum_target_tokens=minimum, maximum_target_tokens=maximum,
        refresh_updates=cadence, bounds=bounds, deficits=deficits,
        cadence_revision='prospective outcome-blind rule; original proposal was eight updates',
        native_authority='original census producer required; seals alone do not qualify hardware'))
    return data,report


def training_input(support, report, profile, update_policy):
    """Only complete projected support can become an installed learner package."""
    from .models import validate_profile
    from oeis_learn.pipeline_validation.progressive_training import validate_update_policy
    if (seal(support)!=support or seal(report)!=report or report['status']!='COMPLETE'
            or report['support']!=support['identity'] or report['policy']['identity']!=support['policy']
            or report['diagnostic']!=support['diagnostic']):
        raise ValueError('complete original support publication required')
    validate_profile(profile); validate_update_policy(profile,update_policy)
    if update_policy['effective_batch'] not in report['policy']['batches']:
        raise ValueError('effective batch outside prospective support domain')
    return seal(dict(version='progressive-training-input/v1',profile=profile,
        update_policy=update_policy,shards=support['shards'],layout=support['layout'],
        refresh_updates=support['refresh_updates'],
        scope='DIAGNOSTIC_ONLY' if support['diagnostic'] else 'NATIVE_VERIFIED',diagnostic_fault=None))


def transfer_support(data):
    """Shared initial gradient support, excluding probes and later refreshes.

Parameter/composition witnesses must be available to both static and refreshed
arms at their common start. Later shards and all raw draws still count as prior
exposure for exclusion; they cannot manufacture initial transfer support.
"""
    from oeis_learn.pipeline_validation.progressive_research_census import features
    if seal(data)!=data or data.get('version')!='progressive-prepared-support/v1':
        raise ValueError('original published support required')
    probes={v['id'] for v in data['layout']['probes'].values()}
    rows=[r for r in (data['shards'][0] if data['shards'] else []) if r['draw_identity'] not in probes]
    support=dict(structures=set(),nodes=set(),operators=set())
    for row in rows:
        f=features(row['source']);support['structures'].add(row['structure'])
        support['nodes'].update(f['nodes']);support['operators'].update(f['operators'])
    binding=seal(dict(version='progressive-transfer-support/v1',published=data['identity'],
        policy=data['policy'],initial_gradient=[r['draw_identity'] for r in rows],
        probes=sorted(probes),refreshed_shards=max(0,len(data['shards'])-1),
        structures=sorted(support['structures']),nodes=sorted(support['nodes']),operators=sorted(support['operators']),
        scope='common initial gradient support; probes excluded; all original draws remain exposure exclusions'))
    return support,binding
