"""Shared-clock accounting for existing generated portfolios; no score authority."""
from collections import Counter,defaultdict
from oeis_learn.pipeline_validation.identity import seal


def interval_union(intervals):
    end=None;total=0.
    for start,stop in sorted(intervals):
        if stop<start:raise ValueError('reversed service interval')
        total+=max(0.,stop-max(start,end if end is not None else start))
        end=max(stop,end if end is not None else stop)
    return total


def summarize(records,clock):
    if seal(clock)!=clock or any(seal(c)!=c for c in clock['clocks']):
        raise ValueError('changed portfolio clocks')
    batches=defaultdict(list);native=[];failures=Counter();outcomes=Counter();signatures=[];unknown=0
    for task,row in sorted(records.items()):
        for mode,portfolio in sorted(row['details'].items()):
            for d in portfolio['details']:
                if mode!='template':
                    if d.get('generation_batch_id') is None:
                        unknown+=1
                    else:batches[(mode,d['generation_batch_id'],d['generation_seconds'])].append(d)
                c=d['fitting'].get('clock',{})
                if c.get('dispatched') is not None and c.get('completed') is not None:
                    native.append((c['dispatched'],c['completed']))
                if not d['generation']['complete']:failures[d['generation']['reason']]+=1
                outcomes[d['fitting']['outcome']]+=1
                signatures.append(dict(task=task,mode=mode,position=d['position'],
                    seed=d['seed'],tokens=d['tokens'],source_sha256=d['source_sha256'],
                    generation=d['generation'],fitting=d['fitting']['outcome']))
    if any(len(ds)!=ds[0]['generation_batch_size'] or
           any(d['generation_batch_size']!=len(ds) for d in ds) for ds in batches.values()):
        raise ValueError('incomplete or colliding decode batch accounting')
    cc=clock['clocks']
    return seal(dict(version='pac-selector-run-diagnostics/v1',scientific_status='DIAGNOSTIC_ONLY',
        freeze_eligible=False,portfolios=len(cc),decode_batches=len(batches),
        cost_complete=clock['cost_complete'],unidentified_decode_rows=unknown,
        decode_seconds=None if unknown else sum(k[2] for k in batches),
        measured_decode_seconds=sum(k[2] for k in batches),
        cap_batches=sum(any(len(d['tokens'])>=1024 for d in ds) for ds in batches.values()),
        cap_batch_seconds=sum(k[2] for k,ds in batches.items() if any(len(d['tokens'])>=1024 for d in ds)),
        actual_batch_sizes=dict(Counter(str(len(ds)) for ds in batches.values())),
        native_service_union_seconds=interval_union(native),
        portfolio_span_seconds=max(c['finish'] for c in cc)-min(c['start'] for c in cc),
        generation_failures=dict(failures),fitting_outcomes=dict(outcomes),
        attempt_behavior=seal(dict(attempts=signatures))['identity'],
        scope='decode wall includes host sampling; service unions exclude raw calls; not GPU utilization'))
