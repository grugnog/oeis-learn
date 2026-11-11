"""Prospective measured resource finalization for the registered scientific round.

Only disk partitions decrease. Scientific exposure/time entitlements and every
finite choice remain unchanged; infeasible measured work becomes a deficit.
"""
import json
from oeis_learn.generalization.progressive.contracts import seal,verify_refs

POLICY='progressive-measured-research-resources/v1'


def policy():
    return seal(dict(version=POLICY,metadata_bytes=128<<20,generations=2,
        checkpoint_scope='maximum original retained or exact calibrated full state',
        training_estimate='2x measured update and checkpoint per complete update; two probes per eight updates',
        permits='decrease disk only; no scientific budget or search-domain changes'))


def finalize(root,payload,cases,producers):
    refs=[r['artifact'] for r in payload['references'] if r['name']=='research_resource_policy']
    if not refs:return None
    if len(refs)!=1 or json.loads(verify_refs(root,refs)[0])!=policy():
        raise ValueError('original prospective measured-resource rule required')
    from oeis_learn.generalization.progressive.storage import checkpoint_size
    from oeis_learn.generalization.progressive.trial_scopes import variants
    original_bytes,_=checkpoint_size(root,payload['checkpoint'],checkpoint_root=payload.get('checkpoint_root'),research=True)
    measured=max(original_bytes,max(p['checkpoint_bytes'] for p in producers))
    by_scope={p['compatibility']:p for p in producers}
    inputs={c['compatibility']:c['package'] for c in cases}
    if set(inputs)!=set(by_scope):raise ValueError('complete exact resource measurement inventory required')
    plans=payload['task_plans'];by_id={p['id']:p for p in plans}
    old_total=sum(p['caps']['disk_bytes'] for p in plans)
    old_slots={n['id']:sum(p['caps']['disk_bytes'] for p in plans if p['slot']==n['id']) for n in payload['graph']['nodes']}
    estimates=[];deficits=[]
    policies={r['artifact']['sha256']:r['artifact'] for r in payload['references'] if r['kind']=='policy'}
    for task in plans:
        if task['kind']=='training':
            needed=2*measured+(128<<20)
            worst=None
            parent=task['initializer'].get('parent_task')
            delta=task['token_target']-(by_id[parent]['token_target'] if parent else 0)
            for selected in variants(task):
                key=selected['initializer']['compatibility'];package=inputs[key];producer=by_scope[key]
                minimum=min(len(r['tokens']) for shard in package['shards'] for r in shard)*package['update_policy']['effective_batch']
                updates=(delta+minimum-1)//minimum
                m=producer['measurements']
                elapsed=2*(updates*(m['update_us']+m['checkpoint_us'])+2*((updates+7)//8)*m['probe_us'])
                row=dict(task=task['id'],compatibility=key,maximum_updates=updates,estimated_work_us=elapsed,
                    available_work_us=task['caps']['elapsed_us']-task['startup_us']-task['finalization_us'])
                if worst is None or elapsed>worst['estimated_work_us']:worst=row
            estimates.append(worst)
            if worst['estimated_work_us']>worst['available_work_us']:deficits.append('training_time:'+task['id'])
        else:
            request=json.loads(verify_refs(root,[policies[task['policy']]])[0])
            needed=(measured if request.get('operation')=='generate' else 0)+(128<<20)
        if needed>task['caps']['disk_bytes']:
            deficits.append('measured_task_storage:'+task['id'])
        else:
            task['caps']['disk_bytes']=needed
    for node in payload['graph']['nodes']:
        node['caps']['disk_bytes']+=sum(p['caps']['disk_bytes'] for p in plans if p['slot']==node['id'])-old_slots[node['id']]
    payload['caps']['disk_bytes']+=sum(p['caps']['disk_bytes'] for p in plans)-old_total
    payload['deficits']=sorted(set(payload['deficits']+deficits))
    return seal(dict(version='progressive-research-resource-estimate/v1',status='BLOCKED' if deficits else 'READY',
        deficits=deficits,checkpoint_bytes=measured,storage_before_bytes=old_total,
        storage_after_bytes=sum(p['caps']['disk_bytes'] for p in plans),training=estimates,
        all_choices_retained=True,exposure_caps_unchanged=True,time_caps_unchanged=True,
        interpretation='conservative exact-input feasibility; actual task clocks remain authoritative'))
