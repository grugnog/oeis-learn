"""Installed role adapters for training probes and a separately frozen fresh screen."""
import json
import time
from collections import Counter

from .identity import seal, digest, write
from .generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact, shape
from oeis_learn.generalization.selector_validation import CONTROLS, probe_views, fresh_design, fresh_draw, paired_metrics


def _read(root, ref, cap=64<<20):
    return json.loads(verify_artifact(root, ref, max_bytes=cap))


def _publish(output, prompts, targets, roster, phase, **extra):
    write(output/'visible/prompts.json', prompts)
    write(output/'private/targets.json', targets)
    write(output/'private/roster.json', roster)
    return dict(status='PASS', phase=phase, tasks=len(targets),
        prompts=reference(output/'visible/prompts.json',output),
        targets=reference(output/'private/targets.json',output),
        roster=reference(output/'private/roster.json',output), **extra)


def publish_probes(inputs, output, settings, *, seconds):
    if set(settings) != {'probes','inventory'}:
        raise ValueError('closed training probe publication request')
    root=inputs['mechanical_sources']
    prompts,targets,roster=probe_views(_read(root,settings['probes']),_read(root,settings['inventory']))
    return _publish(output,prompts,targets,roster,'probes')


def publish_structural(inputs, output, settings, *, seconds):
    """Custodian only: original cells never enter generation or scoring channels."""
    from oeis_learn.generalization.structural import original_cells
    from oeis_learn.generalization.structural_draws import CandidateExclusions, roster as make_roster
    from oeis_learn.generalization.preparation import reference_candidate
    from oeis_learn.generalization.cohorts import project
    from oeis_learn.synthesis.client import SynthesisClient
    if set(settings) != {'protocol','originals','observed','probes','study'} or settings['protocol'] != fresh_design():
        raise ValueError('closed frozen fresh structural request')
    declared=[{k:v for k,v in r.items() if k!='ref'} for r in settings['originals']]
    if sorted(declared,key=lambda r:r['name']) != sorted(original_cells(),key=lambda r:r['name']):
        raise ValueError('all sixteen original exclusion cells required')
    deadline=time.monotonic()+seconds
    protocol=settings['protocol'];draws=fresh_draw(protocol);exact=[];rejections=Counter()
    for row in draws:
        if time.monotonic()>=deadline:raise TimeoutError('fresh reference admission deadline')
        value,error=reference_candidate(row)
        if error:rejections[error]+=1
        else:exact.append(dict(value[0],role='selection',holdout='structural'))
    index=CandidateExclusions(exact)
    for item in settings['originals']:
        rows=_read(inputs['isolation_truth'],item['ref'],2<<30)
        if len(rows)!=item['count']:raise ValueError('original exclusion count changed')
        index.observe(rows,role=item['role'],holdout=item['holdout'],deadline=deadline)
    observed=_read(inputs['isolation_truth'],settings['observed'])
    if len(observed)!=128:raise ValueError('complete previously observed PAC structural cohort required')
    index.observe(observed,role='selection',holdout='structural',deadline=deadline)
    # Prepared probe sources contain placeholders; compare exact windows directly.
    for probe in _read(inputs['isolation_truth'],settings['probes']):
        if probe['role']!='training' or len(probe['values'])!=100:raise ValueError('training probe exclusion required')
        values=tuple(probe['values'])
        for i in range(81):
            for key in index.keys['finite_window'].get(values[i:i+20],()):
                index.bad[key].add('training_probe_window')
    bad=index.rejected();eligible=[r for r in exact if r['id'] not in bad]
    unique={}
    for row in eligible:unique.setdefault(row['source'],row)
    eligible=list(unique.values());available=Counter(r['family'] for r in eligible)
    deficits={c['family']:c['count']-available[c['family']] for c in protocol['cells'] if available[c['family']]<c['count']}
    availability=seal(dict(version='pac-selector-availability/v1',protocol=protocol['identity'],
        draws=len(draws),available=dict(available),deficits=deficits,
        reference_rejections=dict(rejections),exclusion_reasons=dict(Counter(v for reasons in bad.values() for v in reasons))))
    write(output/'availability.json',availability)
    if deficits:
        return dict(status='BLOCKED',reason='FRESH_COHORT_QUOTA_DEFICIT',deficits=deficits,
                    availability=reference(output/'availability.json',output),retry='no reseed or quota relaxation')
    selected=[]
    for cell in protocol['cells']:
        selected.extend(sorted((r for r in eligible if r['family']==cell['family']),key=lambda r:r['id'])[:cell['count']])
    # A native failure never substitutes a different row.
    with SynthesisClient(settings['study'],cache_root=output/'private/native') as client:
        for row in selected:
            native=client.execute(row['source'],row['indices'],deadline=min(deadline,time.monotonic()+4))
            if native.get('outcome')!='ok' or native['values']!=row['values']:
                raise ValueError('fresh structural native/reference admission incomplete or disagrees')
    roster=make_roster(selected,protocol);prompts=[];targets=[]
    if len(roster)!=128 or Counter(r['category'] for r in roster.values())!=dict(certified=68,ambiguous=26,unresolved=34):
        raise ValueError('fresh structural allocation changed')
    for row in selected:
        visible,truth=project(dict(row,id=digest(['selection',row['id'],row['source']])))
        prompts.append(visible);targets.append(truth)
    return _publish(output,prompts,targets,roster,'structural',protocol=protocol['identity'],
                    availability=reference(output/'availability.json',output))


def generate(inputs, output, settings, *, seconds):
    from .generalization_role import generator
    return generator(inputs,output,settings,seconds=seconds,pac=True,compare_selectors=True)


def score(inputs, output, settings, *, seconds):
    from .generalization_scoring import score_frozen
    from oeis_learn.synthesis.client import SynthesisClient
    if set(settings)!={'targets','roster','original','alternative','observations','phase','study'}:
        raise ValueError('closed selector comparison scoring request')
    phase=settings['phase']
    if phase not in {'probes','structural'}:raise ValueError('explicit validation phase required')
    rows=_read(inputs['selection_supervision'],settings['targets'])
    roster=_read(inputs['selection_supervision'],settings['roster'])
    for row in rows:
        shape('training-target' if phase=='probes' else 'scorer-target',row)
        if row['role']!=('training' if phase=='probes' else 'selection') or phase=='structural' and row['stratum']!='structural':
            raise ValueError('wrong validation supervision role')
    if len(rows)!=len(roster) or {r['id'] for r in rows}!=set(roster):raise ValueError('complete fixed truth roster required')
    root=inputs['selection_candidates']
    selections={key:_read(root,settings[key],512<<20) for key in ('original','alternative')}
    observations=_read(root,settings['observations'],128<<20)
    for data in (*selections.values(),observations):
        if set(data)!=set(roster):raise ValueError('complete frozen candidate roster required')
    for key in roster:
        if set(observations[key])!=set(CONTROLS):raise ValueError('complete selector observation roster required')
        for data in selections.values():
            if set(data[key]['selections'])!=set(CONTROLS):raise ValueError('all seven controls required')
    scores={};deadline=time.monotonic()+seconds
    with SynthesisClient(settings['study'],cache_root=output/'private/native') as client:
        for key in ('original','alternative'):
            scores[key]=score_frozen(selections[key],{r['id']:r for r in rows},client,deadline=deadline)['outcomes']
    incomplete=0;changed=Counter();selector_seconds=Counter()
    for key,values in observations.items():
        for control,value in values.items():
            changed[control]+=value['changed']
            selector_seconds[control]+=value['elapsed_seconds']
            if not value['complete']:
                incomplete+=1
                case=scores['alternative'][key][control]
                case.update(fallback_complete=case['complete'],fallback_success=case['success'],
                            complete=False,success=False,reason='selector_observation_incomplete')
    metrics={}
    for category in sorted({r['category'] for r in roster.values()}):
        subset={k:r for k,r in roster.items() if r['category']==category}
        metrics[category]={control:paired_metrics(subset,
            {k:scores['original'][k][control] for k in subset},
            {k:scores['alternative'][k][control] for k in subset}) for control in CONTROLS}
    result=seal(dict(version='pac-selector-scoring/v1',status='PASS',phase=phase,study=settings['study'],
        roster=roster,metrics=metrics,outcomes=scores,changed=dict(changed),trace_incomplete=incomplete,
        selector_seconds=dict(selector_seconds),
        assessment='NOT_EXECUTED',training_tokens=0))
    write(output/'scoring.json',result)
    return result
