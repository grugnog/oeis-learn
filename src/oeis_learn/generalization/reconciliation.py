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


def pac_costs(costs):
    """Unavailable history stays null; equal budgets are not equal useful work."""
    from .budgets import integer,_seconds
    expected={'allowed_tokens','useful_tokens','reserved_tokens','discarded_tokens','unspent_tokens','examples','updates','seconds'}
    if not set(costs)<=set('ABCD'):raise ValueError('unknown arm cost')
    for arm,value in costs.items():
        if value is None:continue
        fields(value,expected,'PAC cost')
        for key in expected-{'seconds'}:integer(value[key])
        if value['seconds'] is not None:_seconds(value['seconds'])
        if value['allowed_tokens']!=sum(value[k] for k in ('useful_tokens','reserved_tokens','discarded_tokens','unspent_tokens')):
            raise ValueError('unreconciled useful/reserved/discarded/unspent allowance')
    present=[v for v in costs.values() if v is not None]
    return dict(arms=costs,equal_allowance=len(present)>=2 and len({v['allowed_tokens'] for v in present})==1,
                equal_useful_exposure=len(present)>=2 and len({v['useful_tokens'] for v in present})==1,
                matched_wall_time=False,interpretation='equal allowance does not establish equal examples, updates or compute')


def pac_comparison_report(roster,outcomes,*,experiment,control,profiles,costs=None):
    from .execution import validate_pac_policy
    from .metrics import pac_structural_metrics,pac_paired_counts,paired_difference
    exp=validate_pac_policy(experiment,control,profiles)
    allowed=set(exp['arms'])|{'A'}
    if not set(outcomes)<=allowed:raise ValueError('unfrozen arm outcome')
    metrics={arm:pac_structural_metrics(roster,values) for arm,values in outcomes.items()}
    # Validate the full denominator even when no arm has any observations.
    pac_structural_metrics(roster,{})
    pairs=[('C','B','allocation_plus_review')]
    if exp['control_mode'] in {'matched','fresh-control'}:pairs.insert(0,('B','A','representation'))
    if 'D' in exp['arms']:pairs.extend([('D','B','allocation'),('C','D','review_substitution')])
    comparisons={};certified={k:r['family'] for k,r in roster.items() if r['category']=='certified'}
    for treatment,baseline,attribution in pairs:
        present=all(a in metrics for a in (treatment,baseline))
        eligible=present and all(metrics[a]['certified']['complete']>=65 for a in (treatment,baseline))
        if present:
            a,b=outcomes[baseline],outcomes[treatment]
            bounds=paired_difference(certified,{k:a[k] for k in certified if k in a},{k:b[k] for k in certified if k in b},macro=True)
            counts=pac_paired_counts(roster,a,b)
        else:bounds=None;counts=None
        comparisons[treatment+'-'+baseline]=dict(attribution=attribution,interpretable=eligible,
            status='EXPLORATORY_SINGLE_SEED' if eligible else 'INCOMPLETE',paired=counts,
            family_macro_difference_bounds=[str(x) for x in bounds] if bounds is not None else None,
            uncertainty_unit='paired_tasks_descriptive_not_training_seeds')
    required=set(exp['arms'])|({'A'} if exp['control_mode']=='matched' else set())
    return dict(version='pac-structural-comparison/v1',experiment=exp['identity'],arms=metrics,
                comparisons=comparisons,missing_arms=sorted(required-set(outcomes)),
                reference_only_A=exp['control_mode']=='reference-only',historical_measurements=control['historical_measurements'],
                parameter_endpoint='DEFERRED',assessment='NOT_EXECUTED',automatic_winner=False,
                seed_uncertainty='unavailable_single_seed',costs=pac_costs(costs) if costs is not None else None,
                positive_gain_required_for_software_acceptance=False)
