"""Portable, closed public PAC measurements and deterministic reconciliation.

This module consumes hashes, counts and scored outcomes. It never loads model
state, protected supervision, native tools or temporary validation helpers.
"""
import copy
import math
import re
from fractions import Fraction
from .reconciliation import fields,pac_comparison_report
from .metrics import pac_structural_metrics
from .retention import validate_history,FIELDS
from oeis_learn.pipeline_validation.identity import seal,validate

CONTROLS=('raw','rescue','equal_budget','shuffled','absent','template','bank')
CHANNELS=('fixed','bandit','review','fallback')
COSTS=('preparation_seconds','checkpoint_seconds','update_seconds','publication_seconds','invocation_seconds','review_seconds','semantic_seconds')
DETAILS={'tokens','updates','allocation','channels','review','rewards','observations','observation_seconds','costs','history','loss_panels','learner_seconds'}
INPUTS={'version','experiment','control','profiles','roster','structural_scope','outcomes','diagnostics','stages','scoring','training'}


def identity(value):
    if not isinstance(value,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',value):raise ValueError('public measurement identity')


def number(value,*,integer=False,nullable=False):
    if value is None and nullable:return
    if (type(value) is not int if integer else type(value) not in (int,float)) or not math.isfinite(value) or value<0:raise ValueError('nonnegative measured count/cost required')


def project_training(result):
    """Explicit allowlist; absent original measurements stay unavailable."""
    value=result.get('allocation')
    if value is None or value.get('status')!='RECORDED':return None
    monitoring=value.get('monitoring') or {}
    return dict(tokens=copy.deepcopy(value['tokens']),updates=result.get('updates'),allocation=copy.deepcopy(value.get('allocation')),
        channels=copy.deepcopy(value.get('channels')),review=copy.deepcopy(value.get('review')),rewards=copy.deepcopy(value.get('rewards')),
        observations=value.get('observations'),observation_seconds=value.get('observation_seconds'),
        costs={k:value.get('costs',{}).get(k) for k in COSTS},history=copy.deepcopy(monitoring.get('history')),loss_panels=monitoring.get('loss_panels'),learner_seconds=result.get('learner_seconds'))


def diagnostic_summary(value):
    if value is None:return None,None
    fields(value,DETAILS,'public training diagnostics')
    fields(value['tokens'],('allowed','useful','reserved','discarded','unspent'),'token exposure')
    for n in value['tokens'].values():number(n,integer=True)
    if value['tokens']['allowed']!=sum(v for k,v in value['tokens'].items() if k!='allowed'):raise ValueError('public exposure reconciliation')
    for k in ('updates','observations','loss_panels'):number(value[k],integer=True,nullable=True)
    for k in ('learner_seconds','observation_seconds'):number(value[k],nullable=True)
    fields(value['costs'],COSTS,'learner component costs')
    for n in value['costs'].values():number(n,nullable=True)
    if value['channels'] is not None:
        if not set(value['channels'])<=set(CHANNELS):raise ValueError('unregistered channel')
        for n in value['channels'].values():number(n,integer=True)
        if value['updates'] is not None and sum(value['channels'].values())!=value['updates']:raise ValueError('channel updates')
    for label,keys in [('review',('acquisitions','grades','successes','censored','incidental')),('rewards',('valid','censored'))]:
        if value[label] is not None:
            fields(value[label],keys,label)
            for n in value[label].values():number(n,integer=True)
    allocation=value['allocation'];examples=None
    if allocation is not None:
        if not allocation or len(allocation)>10000:raise ValueError('bounded nonempty family allocation')
        for f,channels in allocation.items():
            identity(f);fields(channels,CHANNELS,'family channels')
            for v in channels.values():
                fields(v,('updates','tokens','examples'),'family exposure')
                if Fraction(v['updates'])<0:raise ValueError('fractional family updates')
                for k in ('tokens','examples'):number(v[k],integer=True)
        rows=[v for f in allocation.values() for v in f.values()]
        if sum(v['tokens'] for v in rows)!=value['tokens']['useful'] or value['updates'] is not None and sum(Fraction(v['updates']) for v in rows)!=value['updates']:raise ValueError('family exposure reconciliation')
        examples=sum(v['examples'] for v in rows)
    retention=None
    if value['history'] is not None:
        if allocation is None or value['loss_panels'] is None:raise ValueError('history family/panel authority missing')
        history=validate_history(value['history'],allocation,value['loss_panels'])
        retention=dict(coverage='PARTIAL_LEGACY' if history['omitted_legacy_panels'] else 'COMPLETE',omitted_legacy_panels=history['omitted_legacy_panels'],
            panels=sum(b['count'] for b in history['bins']),families={f:{k:str(sum((Fraction(b['totals'][f][k]) for b in history['bins']),Fraction())) if k=='loss_sum'
                else sum(b['totals'][f][k] for b in history['bins']) for k in FIELDS} for f in allocation},
            first=history['bins'][0]['first'] if history['bins'] else None,last=history['bins'][-1]['last'] if history['bins'] else None)
    # The legacy aggregate cost contract requires complete integer exposure counts.
    cost=None if examples is None or value['updates'] is None else dict(allowed_tokens=value['tokens']['allowed'],useful_tokens=value['tokens']['useful'],reserved_tokens=value['tokens']['reserved'],
        discarded_tokens=value['tokens']['discarded'],unspent_tokens=value['tokens']['unspent'],examples=examples,updates=value['updates'],seconds=value['learner_seconds'])
    return dict(**copy.deepcopy(value),retention=retention),cost


def recompute(inputs):
    validate(inputs,INPUTS,'pac-research-inputs/v1')
    exp=inputs['experiment'];allowed=set(exp['arms'])|{'A'}
    if not set(inputs['outcomes'])<=allowed or set(inputs['diagnostics'])!=allowed or set(inputs['training'])!=allowed or set(inputs['scoring'])!=set(inputs['outcomes']):raise ValueError('frozen measurement arm binding')
    details={};costs={}
    for arm,value in inputs['diagnostics'].items():details[arm],costs[arm]=diagnostic_summary(value)
    metrics={};outcomes={}
    for arm,controls in inputs['outcomes'].items():
        fields(controls,CONTROLS,'all seven frozen controls')
        identity(inputs['scoring'][arm])
        metrics[arm]={c:pac_structural_metrics(inputs['roster'],v or {},scope=inputs['structural_scope']) for c,v in controls.items()}
        outcomes[arm]=controls['equal_budget'] or {}
    for v in inputs['training'].values():
        if v is not None:identity(v)
    stages=inputs['stages']
    for name,row in stages.items():
        if name not in {'admission','qualification','calibration','acceptance','baseline','analysis',*('train-'+a for a in exp['arms']),*('evaluate-'+a for a in exp['arms'])}:raise ValueError('frozen stage costs')
        fields(row,('producer','binding','elapsed_seconds'),'measured stage cost')
        identity(row['producer']);identity(row['binding']);number(row['elapsed_seconds'],nullable=True)
    result=pac_comparison_report(inputs['roster'],outcomes,experiment=exp,control=inputs['control'],profiles=inputs['profiles'],costs=costs,portable=True,structural_scope=inputs['structural_scope'])
    legacy={a:dict(equal_budget=dict(n=len(inputs['roster']),success=sum(v is True for v in values.values()),complete=sum(v is not None for v in values.values()),unresolved=len(inputs['roster'])-sum(v is not None for v in values.values()),**{('reason:'+reason):sum(('recorded' if k in values else 'missing')==reason for k in inputs['roster']) for reason in ('recorded','missing') if any(('recorded' if k in values else 'missing')==reason for k in inputs['roster'])})) for a,values in outcomes.items()}
    result.update(legacy_count_adapter=legacy,controls=metrics,details=details,stage_costs=copy.deepcopy(stages),stage_cost_totals=dict(
        measured_seconds=sum(v['elapsed_seconds'] for v in stages.values() if v['elapsed_seconds'] is not None),
        unavailable=sum(v['elapsed_seconds'] is None for v in stages.values()),
        scope='observed stage elapsed; learner components overlap and are not summed into compute'))
    return result


def verify_analysis(report,inputs):
    """Verify actual derived bytes; descriptive cases are a separate projection."""
    if seal(report)!=report or report.get('version')!='pac-analysis/v2':raise ValueError('reconstructible PAC analysis version required')
    expected=recompute(inputs)
    extras={'version','identity','inputs','cases','missing_case_categories','case_scope','uncertainty'}
    if set(report)!=set(expected)|extras or any(report[k]!=v for k,v in expected.items() if k!='version'):raise ValueError('analysis differs from recomputed measurements')
    # The version belongs to the report envelope, not the nested comparison API.
    return expected
