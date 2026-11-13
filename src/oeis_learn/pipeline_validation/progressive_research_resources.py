"""Prospective measured resource finalization for the registered scientific round.

Only disk partitions decrease. Scientific exposure/time entitlements and every
finite choice remain unchanged; infeasible measured work becomes a deficit.
"""
import json
from .research_limits import padded_us
from oeis_learn.generalization.progressive.contracts import seal,verify_refs

POLICY='progressive-measured-research-resources/v3'


def policy(*, legacy=False, global_storage=False):
    return seal(dict(version='progressive-measured-research-resources/v1' if legacy else 'progressive-measured-research-resources/v2' if global_storage else POLICY,metadata_bytes=128<<20,generations=2,
        checkpoint_scope='maximum original retained or exact calibrated full state' if legacy or global_storage else 'per-task exact calibrated full state plus 128 MiB context; evaluation weights only',
        training_estimate=('4x measured update and checkpoint per complete update; two probes per eight updates' if legacy else '4x measured updates/probes plus bounded hourly, milestone and terminal snapshots'),
        permits='decrease disk only; no scientific budget or search-domain changes'))


def finalize(root,payload,cases,producers):
    refs=[r['artifact'] for r in payload['references'] if r['name']=='research_resource_policy']
    if not refs:return None
    if len(refs)!=1:raise ValueError('original prospective measured-resource rule required')
    registered=json.loads(verify_refs(root,refs)[0])
    if registered not in (policy(),policy(legacy=True),policy(global_storage=True)):
        raise ValueError('original prospective measured-resource rule required')
    if registered==policy(legacy=True) and any(c['package'].get('checkpoint_policy')=='hourly/v1' for c in cases):
        raise ValueError('hourly resource estimates require the prospective v2 policy')
    from oeis_learn.generalization.progressive.storage import checkpoint_size
    from oeis_learn.generalization.progressive.trial_scopes import variants
    original_bytes,_=checkpoint_size(root,payload['checkpoint'],checkpoint_root=payload.get('checkpoint_root'),research=True)
    measured=max(original_bytes,max(p['checkpoint_bytes'] for p in producers))
    by_scope={p['compatibility']:p for p in producers}
    inputs={c['compatibility']:c['package'] for c in cases}
    if set(inputs)!=set(by_scope):raise ValueError('complete exact resource measurement inventory required')
    plans=payload['task_plans'];by_id={p['id']:p for p in plans}
    scoped = registered == policy()
    from oeis_learn.generalization.progressive.storage_plan import task_checkpoints
    bounds = task_checkpoints(plans, producers, payload['milestone_pins']) if scoped else {}
    old_total=sum(p['caps']['disk_bytes'] for p in plans)
    old_slots={n['id']:sum(p['caps']['disk_bytes'] for p in plans if p['slot']==n['id']) for n in payload['graph']['nodes']}
    estimates=[];deficits=[]
    policies={r['artifact']['sha256']:r['artifact'] for r in payload['references'] if r['kind']=='policy'}
    for task in plans:
        if task['kind']=='training':
            bound=bounds.get(task['id'])
            needed=(bound['checkpoint_bytes']*(2+bound['milestone_pins']) if bound else 2*measured)+(128<<20)
            worst=None
            parent=task['initializer'].get('parent_task')
            delta=task['token_target']-(by_id[parent]['token_target'] if parent else 0)
            for selected in variants(task):
                key=selected['initializer']['compatibility'];package=inputs[key];producer=by_scope[key]
                row=estimate_training(package,producer['measurements'],delta=delta,
                    target=task['token_target'],available=task['caps']['elapsed_us']-task['startup_us']-task['finalization_us'])
                row.update(task=task['id'],compatibility=key)
                if worst is None or row['estimated_work_us']>worst['estimated_work_us']:worst=row
            estimates.append(worst)
            if worst['estimated_work_us']>worst['available_work_us']:deficits.append('training_time:'+task['id'])
        else:
            request=json.loads(verify_refs(root,[policies[task['policy']]])[0])
            model_bytes=measured
            if scoped and request.get('operation')=='generate':
                model_bytes=0
                for item in request['artifacts']['model']:
                    if isinstance(item,str):
                        from oeis_learn.pipeline_validation.identity import confined
                        channel=json.loads(confined(root,item).read_bytes())
                        if channel['kind']=='checkpoint':model_bytes+=channel['data']['artifact']['bytes']
                    elif item['kind']=='checkpoint':
                        source=by_id[item['task']]
                        model_bytes+=max(by_scope[v['initializer']['compatibility']].get('weights_bytes',by_scope[v['initializer']['compatibility']]['checkpoint_bytes']) for v in variants(source))
            needed=(model_bytes if request.get('operation')=='generate' else 0)+(128<<20)
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


def estimate_training(package, measurements, *, delta, target, available):
    """Conservative feasibility, separate from unpadded expected-runtime reports."""
    from .generalization_persistence import policy as checkpoint_policy, INTERVAL
    from oeis_learn.generalization.progressive.milestones import MILESTONES
    minimum=min(len(r['tokens']) for shard in package['shards'] for r in shard)*package['update_policy']['effective_batch']
    updates=(delta+minimum-1)//minimum
    snapshots=updates
    if checkpoint_policy(package)=='hourly/v1':
        snapshots=(2+max(0,available)//(INTERVAL*1000000)+
            sum(target-delta<n<=target for n in MILESTONES))
    m=measurements
    elapsed=padded_us(updates*m['update_us']+snapshots*m['checkpoint_us']+2*((updates+7)//8)*m['probe_us'])
    return dict(maximum_updates=updates,checkpoint_policy=checkpoint_policy(package),
        maximum_snapshots=snapshots,estimated_work_us=elapsed,available_work_us=available)
