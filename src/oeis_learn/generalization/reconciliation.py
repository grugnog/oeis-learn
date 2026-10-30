"""Recompute public evidence projections from a frozen measurement roster.

Times are integer microseconds or null (unavailable), never inferred zero.
Roster keys identify a task/checkpoint/control/split cell uniquely; family and
seed are scorer metadata. Unknown fields fail closed, including private truth.
"""
from dataclasses import asdict
from fractions import Fraction
from .metrics import summarize


def fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('invalid ' + label + ' fields')


def summary(roster, outcomes):
    return {k: str(v) if isinstance(v, Fraction) else v
            for k, v in asdict(summarize(roster, outcomes)).items()}


def reconcile_measurements(invocation, measurements, study):
    fields(invocation, ('version','study','required_stages','roster'), 'effective invocation')
    if invocation['version'] != 'generalization-measurement-plan/v1' or invocation['study'] != study['identity']:
        raise ValueError('measurement plan identity/version')
    roster = invocation['roster']
    if not isinstance(roster, dict) or not roster:
        raise ValueError('frozen nonempty measurement roster required')
    for key, row in roster.items():
        fields(row, ('family','seed','cell','attempts'), 'roster row')
        if not isinstance(key,str) or not key or not all(isinstance(row[k],str) and row[k] for k in ('family','cell')):
            raise ValueError('invalid roster identity')
        if type(row['seed']) is not int or row['seed'] not in study['seeds']:
            raise ValueError('unregistered seed')
        if type(row['attempts']) is not int or not 1 <= row['attempts'] <= study['inference']['attempts']:
            raise ValueError('invalid attempt cap')
    if {r['seed'] for r in roster.values()} != set(study['seeds']):
        raise ValueError('missing registered seed in denominator')
    fields(measurements, ('version','attempts'), 'measurements')
    if measurements['version'] != 'generalization-attempts/v1' or not isinstance(measurements['attempts'],list):
        raise ValueError('invalid attempt payload')
    by_row = {key: {} for key in roster}
    costs = {key: {'measured_us':0,'unavailable':0} for key in ('queue','service','cleanup')}
    for attempt in measurements['attempts']:
        fields(attempt, ('row','ordinal','outcome','queue_us','service_us','cleanup_us'), 'attempt')
        key = attempt['row']; ordinal = attempt['ordinal']
        if key not in roster or type(ordinal) is not int or not 0 <= ordinal < roster[key]['attempts'] or ordinal in by_row[key]:
            raise ValueError('unknown/duplicate/out-of-budget attempt')
        if attempt['outcome'] is not True and attempt['outcome'] is not False and attempt['outcome'] is not None:
            raise ValueError('invalid exact outcome')
        by_row[key][ordinal] = attempt['outcome']
        for phase in costs:
            value = attempt[phase+'_us']
            if value is None: costs[phase]['unavailable'] += 1
            elif type(value) is not int or value < 0: raise ValueError('invalid measured cost')
            else: costs[phase]['measured_us'] += value
    outcomes = {key: True if any(v is True for v in attempts.values()) else
                False if len(attempts)==roster[key]['attempts'] and all(v is False for v in attempts.values()) else None
                for key, attempts in by_row.items()}
    families = {key:r['family'] for key,r in roster.items()}
    total = summary(families, outcomes)
    # Nested cell/seed/family keys avoid collisions and mixing different evaluations.
    cells = {}
    for cell in sorted({r['cell'] for r in roster.values()}):
        cells[cell] = {}
        for seed in sorted({r['seed'] for r in roster.values() if r['cell']==cell}):
            selected = {k:r['family'] for k,r in roster.items() if r['cell']==cell and r['seed']==seed}
            cells[cell][str(seed)] = selected
    seed_summary = {cell:{seed:summary(rows,{k:outcomes[k] for k in rows}) for seed,rows in seeds.items()} for cell,seeds in cells.items()}
    family_summary = {cell:{seed:{family:summary({k:f for k,f in rows.items() if f==family},
                           {k:outcomes[k] for k,f in rows.items() if f==family})
                           for family in sorted(set(rows.values()))} for seed,rows in seeds.items()} for cell,seeds in cells.items()}
    expected = sum(r['attempts'] for r in roster.values())
    return dict(summary=total,families=family_summary,seeds=seed_summary,costs=costs,
                bounds={k:total[k] for k in ('lower','upper','completion','macro_lower','macro_upper')},
                coverage=dict(expected_rows=len(roster),resolved_rows=len(roster)-total['unresolved'],
                              expected_attempts=expected,recorded_attempts=len(measurements['attempts']),
                              missing_attempts=expected-len(measurements['attempts'])))
