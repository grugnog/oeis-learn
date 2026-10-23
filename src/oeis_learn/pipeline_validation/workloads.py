"""Material complete-work latency strata; missing observations stay missing."""
from collections import defaultdict
from .identity import number

MANDATORY_STRATA={'updates','validation','generation_native','checkpoint','decode.short',
    'decode.long','decode.mixed','native.large','native.nonlinear','scoring_export'}


def material_strata(attempts,rows):
    groups=defaultdict(list);values=defaultdict(list)
    for row in attempts:
        if row['mode']!='template' and row['generation']['complete'] is True:
            elapsed=number(row['generation_seconds'],positive=True)
            if row['tokens']<=128:values['decode.short'].append(elapsed)
            if row['tokens']>=513:values['decode.long'].append(elapsed)
            if row.get('batch_id') and row['batch_size']>1:
                groups[(row['mode'],row['batch_id'])].append(row)
        elapsed=row.get('native_seconds')
        if row['fitting']['complete'] is True and elapsed is not None:
            number(elapsed,positive=True)
            prompt=rows[row['prompt']]
            if max((abs(int(v)).bit_length() for v in [*prompt['values'],*prompt['indices']]),default=0)>=512:
                values['native.large'].append(elapsed)
            if row.get('native_kind')=='nonlinear':values['native.nonlinear'].append(elapsed)
    for group in groups.values():
        if len({r['tokens'] for r in group})>1:
            values['decode.mixed'].append(max(r['generation_seconds'] for r in group))
    return {name:max(samples) for name,samples in values.items() if samples}
