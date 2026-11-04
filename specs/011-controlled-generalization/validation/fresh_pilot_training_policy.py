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


def assess(results,first_shard_tokens):
    if set(results)!={'static','refreshed'}:raise ValueError('both probe arms required')
    rates=[]
    for result in results.values():
        if (result['status']!='PASS' or result['updates']<MINIMUM_UPDATES or
            result['tokens']<=first_shard_tokens or result['charged_tokens']!=result['tokens']):
            raise ValueError('complete probe with refreshed exposure required')
        timing=result['timing'];windows=timing['windows']
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
    projected=25_000_000*max(rates)*2+1800
    return dict(status='PASS' if projected<=21600 else 'BLOCKED',
        projected_seconds_per_arm=projected,slowest_seconds_per_token=max(rates),
        margin_factor=2,additional_seconds=1800,proposed_seconds_per_arm=21600,
        target_tokens_per_arm=25_000_000,automatic_research=False,
        interpretation='short diagnostic forecast, not a guarantee or scientific comparison')
