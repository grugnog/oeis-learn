"""Frozen, conservative training probe policy; never authorizes a research run."""
import math

ROWS=256
MINIMUM_UPDATES=24
ROLE_SECONDS=1800


def budget(charges):
    if len(charges)!=2 or any(len(c)!=16 or any(type(n) is not int or n<=0 for n in c) for c in charges):
        raise ValueError('two complete 256-row shards required')
    cap=min(2*sum(charges[0]),sum(map(sum,charges)))
    if not 0<cap<=131072 or any(sum(c[:24])>cap for c in (charges[0]*2,charges[0]+charges[1])):
        raise ValueError('probe cannot afford 24 complete updates')
    return cap


def assess(results,first_shard_tokens,*,research_milestones=3):
    if set(results)!={'static','refreshed'}:raise ValueError('both probe arms required')
    if type(research_milestones) is not int or not 0<=research_milestones<=64:raise ValueError('bounded research milestone count required')
    rates=[];policies=set();snapshot_cost=0.
    for result in results.values():
        if (result['status']!='PASS' or result['updates']<MINIMUM_UPDATES or
            result['tokens']<=first_shard_tokens or result['charged_tokens']!=result['tokens']):
            raise ValueError('complete probe with refreshed exposure required')
        timing=result['timing'];windows=timing['windows']
        cadence=timing.get('checkpoint_policy','per-update/v1');policies.add(cadence)
        if cadence not in {'per-update/v1','hourly/v1'}:raise ValueError('registered measured persistence policy required')
        if cadence=='hourly/v1':
            cost=timing.get('max_checkpoint_seconds')
            if (type(cost) not in (int,float) or not math.isfinite(cost) or cost<=0 or
                timing.get('checkpoint_calls',0)<2 or timing.get('journal_writes',0)<result['updates']):
                raise ValueError('measured initial/final snapshots and durable progress required')
            snapshot_cost=max(snapshot_cost,cost)
        if not 2<=len(windows)<=64:raise ValueError('two complete timing windows required')
        previous=4;tokens=0;seconds=0.
        for w in windows:
            if (w['updates']!=8 or type(w['tokens']) is not int or w['tokens']<=0 or
                not isinstance(w['seconds'],(int,float)) or not math.isfinite(w['seconds']) or w['seconds']<=0 or
                w['through_updates']!=previous+8 or w['through_updates']>result['updates']):
                raise ValueError('invalid complete timing window')
            previous=w['through_updates'];tokens+=w['tokens'];seconds+=w['seconds']
            rates.append(w['seconds']/w['tokens'])
        if tokens>result['tokens'] or seconds>timing['invocation_seconds']:raise ValueError('timing exceeds measured work')
    if len(policies)!=1:raise ValueError('paired persistence policies differ')
    work=25_000_000*max(rates)
    snapshot_allowance=(math.ceil(work/3600)+research_milestones+2)*snapshot_cost
    projected=(work+snapshot_allowance)*2+1800
    return dict(status='PASS' if projected<=21600 else 'BLOCKED',
        projected_seconds_per_arm=projected,slowest_seconds_per_token=max(rates),
        checkpoint_policy=next(iter(policies)),snapshot_seconds_allowance=snapshot_allowance,research_milestones=research_milestones,
        margin_factor=2,additional_seconds=1800,proposed_seconds_per_arm=21600,
        target_tokens_per_arm=25_000_000,automatic_research=False,
        interpretation='short diagnostic forecast, not a guarantee or scientific comparison')
