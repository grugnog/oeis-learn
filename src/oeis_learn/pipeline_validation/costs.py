"""Exact-scope finite cost observations; reduced diagnostic work cannot admit study work."""
from .identity import seal,number,digest
from .workloads import MANDATORY_STRATA
from oeis_learn.fresh.cost_policy import pipeline_forecast,workload

COMPONENTS={'training_p95','teacher_forcing_p95','generation_p95','fit_p95',
    'reference_p95','hidden_scoring_p95','checkpoint_p95','startup','export','cleanup'}
MODES={'raw','rescue','equal_budget','shuffled','absent','template'}


def forecast_trials(rows,scope,*,updates,cadence,budget):
    work=workload(scope);n=work['end_screen_prompts'];missing=[]
    if len(rows)!=3:missing.append('three complete exact-scope baseline observations')
    bindings=set()
    for row in rows:
        try:
            gpu=row['gpu'];score=row['score'];options=row['options']
            bindings.add(digest(dict(source=gpu['source'],weights=gpu['weights'],work=row['work'],
                policy=options.get('policy'),ceiling=gpu['resources'].get('ceiling_bytes'))))
            if not MANDATORY_STRATA <= set(row.get('strata',{})):
                missing.append('representative complete material production strata')
            for value in row.get('strata',{}).values():number(value,positive=True)
            if (row['complete'] is not True or options.get('assessment')!='cost-full' or options.get('scope')!=scope
                    or len(gpu['prompt_ids'])!=n or len(set(gpu['prompt_ids']))!=n or gpu['scheduled_attempts']!=n*32):
                missing.append('full frozen '+scope+' controls')
            if not all(value['qualified'] is True for value in [gpu['resources'],score['resources']]):
                missing.append('qualified aggregate resources')
            splits=score['result']['splits']
            if set(splits)!=MODES or any(splits[k]['complete']!=n or splits[k]['denominator']!=n for k in MODES):
                missing.append('complete full-denominator independent scoring')
            if set(row['cost_components'])!=COMPONENTS:missing.append('complete cost components')
            for value in row['cost_components'].values():number(value)
        except (KeyError,TypeError,ValueError):
            missing.append('valid bound cost observation')
    if len(bindings)!=1:missing.append('identical source, weights, policies, resources and completed work')
    envelopes={} if missing else {name:2*max(row['cost_components'][name] for row in rows) for name in COMPONENTS}
    forecast=pipeline_forecast(envelopes,scope,updates=updates,cadence=cadence,budget=budget)
    return seal(dict(version='pipeline-cost-observations/v1',scope=scope,observations=[r['identity'] for r in rows],
        missing=sorted(set(missing)),envelopes=envelopes,forecast=forecast,
        variability='twice the largest of exactly three observations; field names retain v1 reader compatibility, not a population p95 claim',
        authority='scoped forecast only; separate reviewed study decision required'))
