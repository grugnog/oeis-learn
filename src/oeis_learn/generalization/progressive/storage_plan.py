"""Per-task measured checkpoint bounds; old campaigns keep global bounds."""
from .trial_scopes import variants

CONTEXT_BYTES = 128 << 20


def task_checkpoints(plans, producers, milestone_pins):
    by_scope = {p['compatibility']: p for p in producers}
    result = {}
    for task in plans:
        if task['kind'] != 'training':
            continue
        rows = [by_scope[v['initializer']['compatibility']] for v in variants(task)]
        pins = min(milestone_pins, sum(m <= task['token_target'] for m in (25000000, 100000000, 400000000)))
        result[task['id']] = dict(
            checkpoint_bytes=max(p['checkpoint_bytes'] for p in rows) + CONTEXT_BYTES,
            checkpoint_inodes=max(p['checkpoint_inodes'] for p in rows) + 256,
            milestone_pins=pins,
        )
    return result


def checkpoint_bound(storage, plan):
    if 'task_checkpoints' in storage:
        rows=[r for r in storage['task_checkpoints'] if r['task']==plan['id']]
        if len(rows)!=1:raise ValueError('one frozen task checkpoint bound required')
        return rows[0]
    return {k: storage[k] for k in ('checkpoint_bytes', 'checkpoint_inodes', 'milestone_pins')}


def publication_reserve(storage, plan):
    """Free space for the next publication, not a second empty campaign disk."""
    bound = checkpoint_bound(storage, plan)
    return (bound['checkpoint_bytes'] + storage['package_bytes'] + CONTEXT_BYTES,
            bound['checkpoint_inodes'] + 256)
