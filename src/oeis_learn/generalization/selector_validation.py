"""Frozen selector follow-up policy, visible-only comparison, and paired reporting."""
import copy
from collections import Counter

from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.synthesis.contracts import identity
from .selector_coverage import POLICY, rank

CONTROLS = ('raw', 'rescue', 'equal_budget', 'shuffled', 'absent', 'template', 'bank')
ARMS = ('B', 'D')
SEED = 20261108
SELECTION = POLICY['version']


def rerank(envelopes, original, prompt, *, deadline):
    """Receive every admitted envelope before any full-horizon target is mounted."""
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
    if not envelopes:
        if original is not None:
            raise ValueError('original selection missing from eligible pool')
        return None, dict(complete=True, changed=False, eligible=0, candidates={})
    if original is None or len(envelopes) > 64:
        raise ValueError('bounded complete admitted pool required')
    unique = {}
    for envelope in envelopes:
        selected = validate_selection_envelope(envelope)
        if envelope['prompt_sha256'] != prompt.sha256:
            raise ValueError('candidate belongs to another visible prompt')
        unique.setdefault(selected['source_sha256'], envelope)
    before = original['selected']['source_sha256']
    if before not in unique or original not in envelopes:
        raise ValueError('original selection missing from eligible pool')
    rows = [dict(source=e['selected']['source'], source_sha256=k) for k,e in unique.items()]
    observation = rank(rows, prompt, original=before, deadline=deadline)
    observation.update(changed=observation['selected'] != before, eligible=len(rows))
    if not observation['complete']:
        return copy.deepcopy(original), observation
    chosen = copy.deepcopy(unique[observation['selected']])
    selected = chosen['selected']
    body = {k:v for k,v in selected.items() if k not in {'selection','seal_sha256'}}
    body.update(base_selection=selected['selection'], selector_policy=POLICY['identity'])
    chosen['selected'] = dict(body, selection=SELECTION, seal_sha256=identity(body))
    validate_selection_envelope(chosen)
    return chosen, observation


def probe_views(rows, inventory, *, research=True):
    if seal(inventory) != inventory or inventory['role'] != 'training':
        raise ValueError('original sealed training inventory required')
    families = inventory['families']
    membership = {p:f['id'] for f in families for p in f['probes']}
    gradient = {p for f in families for p in f['rows']}
    if (len(membership) != sum(len(f['probes']) for f in families)
            or set(membership)&gradient or len(rows) != len(membership)
            or {r['id'] for r in rows} != set(membership)):
        raise ValueError('complete disjoint training probe panel required')
    if research and (len(families) != 32 or len(rows) != 256 or any(len(f['probes']) != 8 for f in families)):
        raise ValueError('original 32 by 8 training probes required')
    prompts, targets, roster = [], [], {}
    for row in sorted(rows, key=lambda r:r['id']):
        if row['role'] != 'training' or len(row['indices']) != 100 or len(row['values']) != 100:
            raise ValueError('training-only full-horizon probe required')
        prompts.append(dict(id=row['id'], indices=row['indices'][:20], values=row['values'][:20]))
        targets.append(row)
        roster[row['id']] = dict(family=membership[row['id']], category='training')
    return prompts, targets, roster


def fresh_design():
    from .structural import amended_design, prior_protocol
    parent = amended_design(prior_protocol())
    return seal(dict(parent, version='pac-selector-structural/v1', parent_scope=parent['identity'],
        seed=SEED, exclusion='all-16-original-cells-plus-observed-PAC-cohort-and-training-probes/v1',
        retry='no-reseed-no-quota-relaxation', selector=POLICY['identity'], arms=list(ARMS)))


def fresh_draw(protocol):
    from .structural_draws import draw
    if protocol != fresh_design():
        raise ValueError('changed frozen selector structural design')
    return draw(protocol, seed=protocol['seed'])


def paired_metrics(roster, original, alternative):
    if set(original) != set(roster) or set(alternative) != set(roster) or not roster:
        raise ValueError('complete fixed scoring roster required')
    families = sorted({r['family'] for r in roster.values()})
    def measured(outcomes):
        counts = Counter(tasks=len(roster), complete=0, success=0)
        bounds = []
        for family in families:
            rows = [outcomes[k] for k,r in roster.items() if r['family']==family]
            complete = sum(r['complete'] for r in rows)
            success = sum(r['complete'] and r['success'] for r in rows)
            counts.update(complete=complete, success=success)
            bounds.append((success/len(rows), (success+len(rows)-complete)/len(rows)))
        return dict(counts, family_macro_bounds=[sum(x[i] for x in bounds)/len(bounds) for i in (0,1)])
    old, new = measured(original), measured(alternative)
    joint = Counter(count=0, recoveries=0, regressions=0, both_success=0, both_failure=0)
    for k in roster:
        a,b = original[k], alternative[k]
        if a['complete'] and b['complete']:
            joint.update(count=1)
            joint.update([('both_success' if a['success'] else 'recoveries') if b['success']
                          else ('regressions' if a['success'] else 'both_failure')])
    return dict(original=old, alternative=new, joint=dict(joint),
        delta_bounds_pp=[100*(new['family_macro_bounds'][0]-old['family_macro_bounds'][1]),
                         100*(new['family_macro_bounds'][1]-old['family_macro_bounds'][0])],
        interpretation='descriptive family-macro censoring bounds, not confidence intervals')
